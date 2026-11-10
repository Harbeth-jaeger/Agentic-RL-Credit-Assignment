"""ALFWorld multi-turn rollout implemented on slime's public custom-rollout API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from agentic_credit_assignment.adapters.alfworld import AlfWorldAdapter, AlfWorldAdapterConfig


ACTION_BLOCK = re.compile(r"<action>\s*.*?\s*</action>", re.I | re.S)
OPEN_ACTION_BLOCK = re.compile(r"<action>\s*[^<\r\n]+?\s*$", re.I | re.S)


def action_token_mask(tokenizer, token_ids: list[int], text: str) -> list[int]:
    """Mask only the generated action block; explicit reasoning never enters loss."""
    matches = list(ACTION_BLOCK.finditer(text))
    if matches:
        start, end = matches[-1].span()
    else:
        open_match = OPEN_ACTION_BLOCK.search(text)
        if open_match is None:
            return [0] * len(token_ids)
        start, end = open_match.span()
    mask: list[int] = []
    previous = 0
    for index in range(1, len(token_ids) + 1):
        current = len(tokenizer.decode(token_ids[:index], skip_special_tokens=False))
        mask.append(int(current > start and previous < end))
        previous = current
    return mask


def _generation_tokens(output: dict) -> tuple[str, list[int], list[float]]:
    meta = output["meta_info"]
    pairs = meta.get("output_token_logprobs", [])
    return output["text"], [int(x[1]) for x in pairs], [float(x[0]) for x in pairs]


def _append(
    sample: Sample,
    args,
    ids: list[int],
    logps: list[float],
    mask: list[int],
    *,
    meta_info: dict,
    text: str,
) -> tuple[int, int]:
    begin = sample.response_length
    # Keep slime's rollout replay metadata in sync with the policy tokens.  In
    # particular, Sample owns all top-p/score-centering bookkeeping; bypassing
    # meta_info here makes the later training batch impossible to reconstruct.
    sample.append_response_tokens(
        args,
        tokens=ids,
        log_probs=logps,
        trainable=True,
        meta_info=meta_info,
        text=text,
    )
    sample.loss_mask[-len(ids):] = mask
    return begin, sample.response_length


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    """Run one independent official ALFWorld environment for one slime Sample."""
    rollout_started = time.perf_counter()
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    metadata = sample.metadata or {}
    gamefile = Path(metadata["gamefile"])
    setup_started = time.perf_counter()
    adapter = AlfWorldAdapter(
        AlfWorldAdapterConfig(
            data_root=Path(args.alfworld_data_root), split=metadata.get("split", "train"),
            num_envs=1, seed=int(metadata.get("seed", sample.index or 0)),
            max_steps=int(metadata.get("max_steps", args.alfworld_max_steps)),
            history_length=int(args.alfworld_history_length),
            fixed_gamefile=gamefile,
        )
    )
    trace: list[dict] = []
    response_ids: list[int] = []
    total_reward = 0.0
    setup_seconds = 0.0
    generation_seconds = 0.0
    environment_step_seconds = 0.0
    try:
        # TextWorld's TatSu parser has process-global mutable state. Keep all
        # official environment operations on this event-loop thread; concurrent
        # model generation is distributed across the eight SGLang engines.
        observation = adapter.reset()[0]
        setup_seconds = time.perf_counter() - setup_started
        first_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": observation.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(first_prompt, add_special_tokens=False)
        sample.prompt = first_prompt
        sample.tokens = list(prompt_ids)
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.metadata = metadata
        params = dict(sampling_params)
        params["max_new_tokens"] = int(metadata.get("tokens_per_turn", args.alfworld_tokens_per_turn))
        params["stop"] = ["</action>"]
        params["no_stop_trim"] = True

        max_steps = int(metadata.get("max_steps", args.alfworld_max_steps))
        for turn in range(max_steps):
            headers = None
            if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
                # Keep one multi-turn episode on one engine for KV-prefix reuse,
                # while different session IDs hash across all rollout engines.
                headers = {"X-SMG-Routing-Key": sample.session_id}
            generation_started = time.perf_counter()
            output = await post(
                url,
                {
                    "input_ids": sample.tokens,
                    "sampling_params": params,
                    "return_logprob": True,
                },
                headers=headers,
            )
            turn_generation_seconds = time.perf_counter() - generation_started
            generation_seconds += turn_generation_seconds
            text, ids, logps = _generation_tokens(output)
            token_mask = action_token_mask(tokenizer, ids, text)
            begin, end = _append(
                sample,
                args,
                ids,
                logps,
                token_mask,
                meta_info=output["meta_info"],
                text=text,
            )
            response_ids.extend(ids)
            environment_step_started = time.perf_counter()
            next_obs, parsed, rewards, dones, won = adapter.step([text])
            turn_environment_step_seconds = time.perf_counter() - environment_step_started
            environment_step_seconds += turn_environment_step_seconds
            reward = rewards[0]
            total_reward = max(total_reward, reward, float(won[0]))
            trace.append({
                "turn": turn, "state": observation.graph_state,
                "state_text": observation.feedback, "admissible_actions": list(observation.admissible_actions),
                "raw_output": text, "thought": text.split("<action>", 1)[0],
                "action": parsed[0].action, "protocol_valid": parsed[0].protocol_valid,
                "action_valid": parsed[0].action_valid, "extraction_method": parsed[0].extraction_method,
                "environment_feedback": next_obs[0].feedback, "next_state": next_obs[0].graph_state,
                "environment_reward": reward, "won": won[0],
                "response_span": [begin, end], "action_token_count": sum(token_mask),
                "generation_seconds": turn_generation_seconds,
                "environment_step_seconds": turn_environment_step_seconds,
            })
            observation = next_obs[0]
            if dones[0] or won[0]:
                break

            # Environment text is context, never a policy target. Reuse the official
            # adapter's model-facing prompt and Qwen chat template rather than inventing
            # an ALFWorld state protocol.
            next_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": observation.prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            context_ids = tokenizer.encode(next_text, add_special_tokens=False)
            sample.append_response_tokens(args, tokens=context_ids, trainable=False)
            response_ids.extend(context_ids)

        sample.reward = float(total_reward)
        sample.response = tokenizer.decode(response_ids, skip_special_tokens=False)
        sample.status = Sample.Status.COMPLETED
        sample.metadata.update({
            "task_id": metadata.get("task_id", gamefile.parent.name), "gamefile": str(gamefile),
            "split": metadata.get("split", "train"), "steps": trace,
            "success": bool(total_reward > 0), "truncated": len(trace) >= max_steps,
            "final_reward": float(total_reward), "evaluation": evaluation,
            "timing": {
                "environment_setup_seconds": setup_seconds,
                "model_generation_seconds": generation_seconds,
                "environment_step_seconds": environment_step_seconds,
                "rollout_wall_seconds": time.perf_counter() - rollout_started,
                "num_steps": len(trace),
            },
        })
        sample.train_metadata = {"turn_spans": [x["response_span"] for x in trace]}
        return sample
    finally:
        adapter.close()


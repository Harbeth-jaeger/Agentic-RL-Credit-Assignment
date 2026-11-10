#!/usr/bin/env python3
"""Probe Qwen3.5 native thinking with a minimal ALFWorld action prompt."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

_env_lib = str(Path(sys.prefix) / "lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    part for part in (_env_lib, os.environ.get("LD_LIBRARY_PATH", "")) if part
)
ctypes.CDLL(str(Path(_env_lib) / "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from agentic_credit_assignment.adapters.alfworld import AlfWorldAdapter, AlfWorldAdapterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def choose_gamefile(data_root: Path) -> Path:
    candidates = sorted((data_root / "json_2.1.1/train").glob("**/game.tw-pddl"))
    simple = [path for path in candidates if "pick_and_place_simple" in str(path)]
    if not candidates:
        raise FileNotFoundError(f"No ALFWorld train games below {data_root}")
    return (simple or candidates)[0]


def compact(text: str) -> str:
    return " ".join(text.split())


def build_prompt(
    task: str,
    state: str,
    actions: tuple[str, ...],
    history: list[tuple[str, str, str]],
) -> str:
    if history:
        history_text = "\n".join(
            f"s{i}: {compact(before)}\na{i}: {action}\ns{i + 1}: {compact(after)}"
            for i, (before, action, after) in enumerate(history)
        )
    else:
        history_text = "None."
    action_text = "\n".join(f"- {action}" for action in actions)
    return (
        "You are an agent acting in the ALFWorld text environment.\n"
        f"Task: {task}\n"
        f"Your two most recent interactions:\n{history_text}\n"
        f"Current state: {compact(state)}\n"
        "Choose the single best next action from the allowed actions below.\n"
        f"Allowed actions:\n{action_text}\n"
        "After thinking, output only the exact text of one allowed action. "
        "Do not output an explanation, label, XML tag, or any other text."
    )


def final_answer(generated: str) -> tuple[str, bool]:
    if "</think>" not in generated:
        return "", False
    return generated.rsplit("</think>", 1)[1].strip(), True


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a silent CPU fallback")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()

    gamefile = choose_gamefile(args.data_root).resolve()
    adapter = AlfWorldAdapter(
        AlfWorldAdapterConfig(
            data_root=args.data_root,
            split="train",
            num_envs=1,
            seed=args.seed,
            max_steps=args.steps,
            history_length=0,
            fixed_gamefile=gamefile,
        )
    )
    observation = adapter.reset()[0]
    history: list[tuple[str, str, str]] = []
    valid_count = 0
    step_times: list[float] = []
    trace: list[dict] = []
    total_start = time.perf_counter()

    print(f"model={args.model_path}")
    print(f"gamefile={gamefile}")
    print(f"task={adapter.tasks[0]}")
    print(f"max_new_tokens_safety_ceiling={args.max_new_tokens}")
    print("repair=disabled retry=disabled constrained_decoding=disabled forced_think_close=disabled")

    try:
        for turn in range(args.steps):
            prompt = build_prompt(
                adapter.tasks[0],
                observation.feedback,
                observation.admissible_actions,
                history[-2:],
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            new_ids = generated_ids[0, inputs.input_ids.shape[1] :]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True)
            raw_output = "<think>\n" + generated
            answer, think_closed = final_answer(generated)
            normalized = " ".join(answer.lower().split())
            pool = {" ".join(action.lower().split()): action for action in observation.admissible_actions}
            valid = think_closed and normalized in pool
            chosen = pool.get(normalized, "__invalid_action__")
            valid_count += int(valid)

            before = observation.feedback
            # An invalid generation still consumes a real environment step;
            # the adapter sends a deliberately invalid command and records the
            # resulting feedback without repairing the model's answer.
            next_observations, parsed, rewards, dones, won = adapter.step(
                [chosen if valid else generated]
            )
            next_observation = next_observations[0]
            feedback = next_observation.feedback

            trace.append(
                {
                    "turn": turn,
                    "prompt": prompt,
                    "state": observation.graph_state,
                    "state_text": before,
                    "admissible_actions": list(observation.admissible_actions),
                    "raw_output": raw_output,
                    "think_closed": think_closed,
                    "final_answer": answer,
                    "action_valid": valid,
                    "executed_action": parsed[0].action,
                    "environment_feedback": feedback,
                    "next_state": next_observation.graph_state,
                    "environment_reward": rewards[0],
                    "won": won[0],
                    "generation_seconds": elapsed,
                }
            )

            if args.quiet:
                print(
                    f"step={turn + 1}/{args.steps} valid={valid} "
                    f"think_closed={think_closed} seconds={elapsed:.3f} "
                    f"answer={answer!r}",
                    flush=True,
                )
            else:
                print("\n" + "=" * 100)
                print(f"STEP {turn + 1}/{args.steps}")
                print(f"PROMPT:\n{prompt}")
                print(f"RAW MODEL OUTPUT:\n{raw_output}")
                print(f"FINAL ANSWER: {answer!r}")
                print(f"THINK_CLOSED: {think_closed} ACTION_VALID: {valid}")
                print(f"GENERATION_SECONDS: {elapsed:.3f}")
                print(f"ENVIRONMENT FEEDBACK:\n{feedback}")

            history.append((before, chosen if valid else "__invalid_action__", feedback))
            observation = next_observation
            if dones[0] or won[0]:
                print(f"Episode ended after {turn + 1} interaction(s).")
                break
    finally:
        adapter.close()

    total = time.perf_counter() - total_start
    completed = len(step_times)
    print("\n" + "=" * 100)
    print(f"SUMMARY action_valid={valid_count}/{completed}")
    print(f"STEP_SECONDS={[round(value, 3) for value in step_times]}")
    print(f"MEAN_GENERATION_SECONDS={sum(step_times) / completed:.3f}")
    print(f"TOTAL_SECONDS={total:.3f}")
    if args.output_json is not None:
        payload = {
            "created_at": datetime.now().astimezone().isoformat(),
            "model_path": str(args.model_path),
            "gamefile": str(gamefile),
            "task": adapter.tasks[0],
            "config": {
                "native_thinking": True,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_steps": args.steps,
                "history_length": 2,
                "repair": False,
                "retry": False,
                "constrained_decoding": False,
            },
            "summary": {
                "steps": completed,
                "valid_actions": valid_count,
                "action_valid_rate": valid_count / completed if completed else 0.0,
                "success": any(item["won"] for item in trace),
                "mean_generation_seconds": sum(step_times) / completed if completed else 0.0,
                "total_seconds": total,
            },
            "steps": trace,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output_json)
        print(f"OUTPUT_JSON={args.output_json}")
    return 0 if completed == args.steps and valid_count == args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())

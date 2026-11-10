#!/usr/bin/env python3
"""Run one transparent GraphGPO-style ReAct rollout in ALFWorld."""

from __future__ import annotations

import argparse
import ctypes
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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from agentic_credit_assignment.adapters.alfworld import AlfWorldAdapter, AlfWorldAdapterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gamefile", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Per-step generation budget; native-thinking Qwen3.5 often exceeds 512 tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def choose_gamefile(data_root: Path) -> Path:
    candidates = sorted((data_root / "json_2.1.1/train").glob("**/game.tw-pddl"))
    if not candidates:
        raise FileNotFoundError(f"No ALFWorld train games below {data_root}")
    simple = [path for path in candidates if "pick_and_place_simple" in str(path)]
    return (simple or candidates)[0]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a silent CPU fallback")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = AutoConfig.from_pretrained(args.model_path)
    native_thinking = model_config.model_type == "qwen3_5"
    model_class = (
        AutoModelForImageTextToText
        if model_config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()
    gamefile = (args.gamefile or choose_gamefile(args.data_root)).resolve()
    adapter = AlfWorldAdapter(
        AlfWorldAdapterConfig(
            data_root=args.data_root,
            split="train",
            num_envs=1,
            seed=args.seed,
            max_steps=args.steps,
            history_length=2,
            fixed_gamefile=gamefile,
            native_thinking=native_thinking,
        )
    )

    protocol_valid = 0
    action_valid = 0
    won_episode = False
    step_times: list[float] = []
    total_started = time.perf_counter()
    try:
        observation = adapter.reset()[0]
        print(f"model={args.model_path}")
        print(f"gamefile={gamefile}")
        print(f"task={adapter.tasks[0]}")
        print(
            f"mode=graphgpo_react native_thinking={native_thinking} "
            "repair=off retry=off constrained_decoding=off"
        )
        print(f"sampling=temperature:{args.temperature},max_new_tokens:{args.max_new_tokens}")

        for turn in range(args.steps):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": observation.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=native_thinking,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            new_ids = generated_ids[0, inputs.input_ids.shape[1] :]
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            # Qwen3.5's native chat template places the opening <think> in the
            # input prefix. Reconstruct the complete assistant response before
            # applying the same strict protocol parser used for Qwen2.5.
            raw_output = f"<think>\n{generated_text}" if native_thinking else generated_text
            next_observations, parsed, _, dones, won = adapter.step([raw_output])
            result = parsed[0]
            protocol_valid += int(result.protocol_valid)
            action_valid += int(result.action_valid)
            won_episode = won_episode or won[0]

            print("\n" + "=" * 100)
            print(f"STEP {turn + 1}/{args.steps}")
            print(f"PROMPT:\n{observation.prompt}")
            print(f"RAW MODEL OUTPUT:\n{raw_output}")
            print(
                "PARSE: "
                f"action={result.action!r} method={result.extraction_method} "
                f"protocol_valid={result.protocol_valid} action_valid={result.action_valid}"
            )
            print(f"GENERATION_SECONDS: {elapsed:.3f}")
            print(f"ENVIRONMENT_FEEDBACK:\n{next_observations[0].feedback}")
            observation = next_observations[0]
            if dones[0] or won[0]:
                print(f"EPISODE_ENDED: done={dones[0]} won={won[0]}")
                break
    finally:
        adapter.close()

    completed = len(step_times)
    total = time.perf_counter() - total_started
    protocol_rate = protocol_valid / completed if completed else 0.0
    action_rate = action_valid / completed if completed else 0.0
    print("\n" + "=" * 100)
    print(
        f"SUMMARY steps={completed}/{args.steps} "
        f"protocol_valid={protocol_valid}/{completed} ({protocol_rate:.2%}) "
        f"action_valid={action_valid}/{completed} ({action_rate:.2%}) "
        f"episode_success={int(won_episode)}/1 ({float(won_episode):.2%})"
    )
    print(f"STEP_SECONDS={[round(value, 3) for value in step_times]}")
    if completed:
        print(f"MEAN_GENERATION_SECONDS={sum(step_times) / completed:.3f}")
    print(f"TOTAL_SECONDS={total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

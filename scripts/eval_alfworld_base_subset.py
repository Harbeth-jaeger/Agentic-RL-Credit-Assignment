#!/usr/bin/env python3
"""Evaluate a base checkpoint on a small, auditable ALFWorld subset."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

_env_lib = str(Path(sys.prefix) / "lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    part for part in (_env_lib, os.environ.get("LD_LIBRARY_PATH", "")) if part
)
ctypes.CDLL(str(Path(_env_lib) / "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_credit_assignment.adapters.alfworld import AlfWorldAdapter, AlfWorldAdapterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="valid_unseen", choices=("valid_seen", "valid_unseen"))
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def select_games(data_root: Path, split: str, count: int, seed: int) -> list[Path]:
    games = sorted((data_root / "json_2.1.1" / split).glob("**/game.tw-pddl"))
    if len(games) < count:
        raise ValueError(f"Requested {count} games, but {split} contains only {len(games)}")
    return random.Random(seed).sample(games, count)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a silent CPU fallback")
    if args.num_tasks < 1 or args.max_steps < 1:
        raise ValueError("num-tasks and max-steps must be positive")

    total_started = time.perf_counter()
    torch.manual_seed(args.seed)
    cuda_device = torch.device(args.device)
    if cuda_device.type != "cuda" or cuda_device.index is None:
        raise ValueError(f"Expected an indexed CUDA device such as cuda:0, got {args.device!r}")
    # This evaluator runs in a fresh process, so allocator peaks start at zero.
    # PyTorch 2.11+cu129 on this image rejects the explicit device argument in
    # reset_peak_memory_stats at the C boundary; no reset is needed here.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()
    model_loaded = time.perf_counter()

    gamefiles = select_games(args.data_root, args.split, args.num_tasks, args.seed)
    adapters: list[AlfWorldAdapter] = []
    observations = []
    records: list[dict] = []
    active: list[bool] = []
    try:
        for index, gamefile in enumerate(gamefiles):
            adapter = AlfWorldAdapter(
                AlfWorldAdapterConfig(
                    data_root=args.data_root,
                    split=args.split,
                    num_envs=1,
                    seed=args.seed + index,
                    max_steps=args.max_steps,
                    history_length=args.history_length,
                    fixed_gamefile=gamefile,
                )
            )
            observation = adapter.reset()[0]
            adapters.append(adapter)
            observations.append(observation)
            active.append(True)
            records.append(
                {
                    "task_index": index,
                    "task": adapter.tasks[0],
                    "gamefile": str(gamefile),
                    "success": False,
                    "done": False,
                    "steps": [],
                }
            )

        rollout_started = time.perf_counter()
        for turn in range(args.max_steps):
            indices = [index for index, is_active in enumerate(active) if is_active]
            if not indices:
                break
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": observations[index].prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for index in indices
            ]
            inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(args.device)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generation_seconds = time.perf_counter() - generation_started
            prompt_width = inputs.input_ids.shape[1]
            outputs = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

            for batch_index, task_index in enumerate(indices):
                previous = observations[task_index]
                next_observations, parsed, rewards, dones, won = adapters[task_index].step(
                    [outputs[batch_index]]
                )
                result = parsed[0]
                step_record = {
                        "turn": turn,
                        "prompt": previous.prompt,
                        "rendered_prompt": rendered[batch_index],
                        "state": previous.graph_state,
                        "state_text": previous.feedback,
                        "admissible_actions": list(previous.admissible_actions),
                        "raw_output": outputs[batch_index],
                        "action": result.action,
                        "protocol_valid": result.protocol_valid,
                        "action_valid": result.action_valid,
                        "extraction_method": result.extraction_method,
                        "environment_feedback": next_observations[0].feedback,
                        "next_state": next_observations[0].graph_state,
                        "environment_reward": rewards[0],
                        "won": won[0],
                        "generation_batch_seconds": generation_seconds,
                        "generation_batch_size": len(indices),
                    }
                records[task_index]["steps"].append(step_record)
                print(
                    "STEP_TRACE=" + json.dumps(
                        {
                            "task_index": task_index,
                            **step_record,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                observations[task_index] = next_observations[0]
                records[task_index]["success"] = bool(won[0])
                records[task_index]["done"] = bool(dones[0])
                if dones[0] or won[0]:
                    active[task_index] = False

            print(
                f"turn={turn + 1}/{args.max_steps} active={sum(active)} "
                f"successes={sum(int(record['success']) for record in records)} "
                f"batch_generation_seconds={generation_seconds:.3f}",
                flush=True,
            )
    finally:
        for adapter in adapters:
            adapter.close()

    rollout_seconds = time.perf_counter() - rollout_started
    successes = sum(int(record["success"]) for record in records)
    protocol_count = sum(
        int(step["protocol_valid"]) for record in records for step in record["steps"]
    )
    action_count = sum(
        int(step["action_valid"]) for record in records for step in record["steps"]
    )
    step_count = sum(len(record["steps"]) for record in records)
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "model_path": str(args.model_path),
            "data_root": str(args.data_root),
            "split": args.split,
            "num_tasks": args.num_tasks,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "native_thinking": False,
            "retry": False,
            "repair": False,
            "constrained_decoding": False,
        },
        "summary": {
            "successes": successes,
            "success_rate": successes / args.num_tasks,
            "total_steps": step_count,
            "protocol_valid_rate": protocol_count / step_count if step_count else 0.0,
            "action_valid_rate": action_count / step_count if step_count else 0.0,
            "model_load_seconds": model_loaded - total_started,
            "rollout_seconds": rollout_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "tasks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

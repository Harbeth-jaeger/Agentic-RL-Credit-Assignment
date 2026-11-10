#!/usr/bin/env python3
"""Run independent matrix multiplications on every GPU for a short period.

This utility is intentionally independent of Slime/Ray.  Its only purpose is to
produce real GPU compute for an otherwise idle Kubernetes instance.  By default
it refuses to start if another process already occupies meaningful GPU memory.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuState:
    index: int
    name: str
    memory_used_mib: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep 8 GPUs computing for five minutes (default)."
    )
    parser.add_argument("--duration", type=float, default=300.0, help="seconds")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=8192,
        help="square BF16 matrix dimension; 8192 uses about 384 MiB/GPU",
    )
    parser.add_argument(
        "--max-used-mib",
        type=int,
        default=1024,
        help="refuse to start if a selected GPU already exceeds this usage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore the occupied-GPU safety check",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if args.matrix_size <= 0:
        parser.error("--matrix-size must be positive")
    return args


def query_gpus() -> list[GpuState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    states: list[GpuState] = []
    for line in completed.stdout.splitlines():
        index, name, memory_used = (field.strip() for field in line.split(",", 2))
        states.append(GpuState(int(index), name, int(memory_used)))
    return states


def exercise_gpu(
    gpu_index: int,
    duration: float,
    matrix_size: int,
    result_queue: mp.Queue,
) -> None:
    try:
        import torch

        torch.cuda.set_device(gpu_index)
        device = torch.device("cuda", gpu_index)
        left = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
        right = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
        output = torch.empty_like(left)

        torch.cuda.synchronize(device)
        started = time.monotonic()
        iterations = 0
        while time.monotonic() - started < duration:
            torch.mm(left, right, out=output)
            iterations += 1
            # Bound the CUDA launch queue while keeping synchronization overhead low.
            if iterations % 16 == 0:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        result_queue.put((gpu_index, iterations, None))
    except BaseException as exc:  # Propagate worker failures to the parent process.
        result_queue.put((gpu_index, 0, f"{type(exc).__name__}: {exc}"))


def main() -> int:
    args = parse_args()
    states = query_gpus()
    if len(states) < args.num_gpus:
        print(
            f"ERROR: requested {args.num_gpus} GPUs, but nvidia-smi reports {len(states)}",
            file=sys.stderr,
        )
        return 2

    selected = states[: args.num_gpus]
    occupied = [gpu for gpu in selected if gpu.memory_used_mib > args.max_used_mib]
    if occupied and not args.force:
        details = ", ".join(
            f"GPU {gpu.index}: {gpu.memory_used_mib} MiB" for gpu in occupied
        )
        print(
            "ERROR: selected GPUs appear occupied; refusing to interfere with another "
            f"job ({details}). Use --force only after checking the processes.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Using GPUs {[gpu.index for gpu in selected]} for {args.duration:.0f}s; "
        f"matrix={args.matrix_size}x{args.matrix_size}, dtype=BF16",
        flush=True,
    )
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=exercise_gpu,
            args=(gpu.index, args.duration, args.matrix_size, result_queue),
            name=f"gpu-{gpu.index}-keep-busy",
        )
        for gpu in selected
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    results = [result_queue.get() for _ in processes]
    failures = [(gpu, error) for gpu, _iterations, error in results if error]
    for gpu, iterations, error in sorted(results):
        if error:
            print(f"GPU {gpu}: FAILED: {error}", file=sys.stderr)
        else:
            print(f"GPU {gpu}: completed {iterations} matmuls")
    if failures or any(process.exitcode != 0 for process in processes):
        return 1
    print("All selected GPUs completed the compute window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

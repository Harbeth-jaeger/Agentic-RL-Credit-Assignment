#!/usr/bin/env python3
"""Initialize/finalize exp_001 manifest and create a compact pilot report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess

import yaml


def git_commit(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def init(args) -> None:
    root = args.run_dir
    for relative in ("logs", "checkpoints", "metrics", "samples", "artifacts", "data"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.yaml"
    previous = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "experiment_id": "exp_001",
        "status": "RUNNING",
        "mode": args.mode,
        "started_at": previous.get("started_at", now) if previous else now,
        "ended_at": None,
        "model": {"name": "Qwen2.5-1.5B-Instruct", "path": str(args.model)},
        "benchmark": {"name": "ALFWorld", "train_split": "train", "eval_split": "valid_seen"},
        "algorithm": {
            "name": "BottleneckFlow B+C",
            "episode_base": "GraphGPO step-weighted episode advantage",
            "invalid_action_penalty": False,
            "think_tokens_in_loss": True,
        },
        "hyperparameters": {
            "train_tasks_per_update": args.train_size,
            "rollouts_per_task": args.group_size,
            "eval_tasks": args.val_size,
            "max_steps": args.max_steps,
            "total_training_steps": args.training_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "hardware": {"requested_gpus": 8, "target": "8 x A100 80GB", "minimum_free_headroom_pct": 15},
        "framework": {"trainer": "veRL/verl-agent", "rollout_engine": "vLLM", "strategy": "FSDP"},
        "code": {
            "project_commit": git_commit(args.project_root),
            "verl_agent_commit": git_commit(args.verl_root),
            "verl_agent_branch": "aca/001-bottleneckflow",
        },
        "paths": {
            "run_dir": str(root),
            "train_log": str(root / "logs/train.log"),
            "checkpoints": str(root / "checkpoints"),
            "rollouts": str(root / "samples/rollouts.jsonl"),
            "graphs": str(root / "samples/graphs.jsonl"),
        },
    }
    attempts = list(previous.get("attempts", [])) if previous else []
    attempts.append({"started_at": now, "resume_mode": "auto"})
    manifest["attempts"] = attempts
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")


def finish(args) -> None:
    path = args.run_dir / "manifest.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.exit_code == 0:
        manifest["status"] = "COMPLETED"
    elif args.exit_code in (130, 143):
        manifest["status"] = "STOPPED"
    else:
        manifest["status"] = "FAILED"
    manifest["exit_code"] = args.exit_code
    manifest["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    gpu_path = args.run_dir / "metrics/gpu_memory.csv"
    peaks = {}
    if gpu_path.exists():
        with gpu_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                gpu = row["gpu"]
                used, total = float(row["memory_used_mib"]), float(row["memory_total_mib"])
                previous = peaks.get(gpu, {"used": 0.0, "total": total})
                if used > previous["used"]:
                    peaks[gpu] = {"used": used, "total": total}
        manifest["gpu_memory"] = {
            "per_gpu_peak_mib": {gpu: value["used"] for gpu, value in peaks.items()},
            "max_utilization_pct": max((100 * value["used"] / value["total"] for value in peaks.values()), default=0),
            "minimum_headroom_pct": min((100 * (value["total"] - value["used"]) / value["total"] for value in peaks.values()), default=100),
        }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")


def analyze(args) -> None:
    root = args.run_dir
    metrics_path = root / "metrics/train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()] if metrics_path.exists() else []
    comparison = root / "artifacts/comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "split", "success_rate", "source"])
        writer.writerow(["BottleneckFlow", "valid_seen", "", "pilot eval_metrics.json"])
        writer.writerow(["GraphGPO", "valid_seen", "", "paper reference; not a controlled reproduction"])
    eval_path = root / "metrics/eval_metrics.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    headroom = manifest.get("gpu_memory", {}).get("minimum_headroom_pct")
    timings = {
        key: row[key]
        for row in rows[-1:]
        for key in row
        if key.startswith("timing_s/")
    }
    report = [
        "# Experiment 001 pilot analysis",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- updates recorded: `{len(rows)}`",
        f"- minimum GPU-memory headroom: `{headroom}`%",
        f"- 15% headroom gate: `{'PASS' if headroom is not None and headroom >= 15 else 'FAIL/UNKNOWN'}`",
        f"- rollout audit present: `{(root / 'samples/rollouts.jsonl').exists()}`",
        f"- graph audit present: `{(root / 'samples/graphs.jsonl').exists()}`",
        "",
        "## Last-update stage timings",
        "",
        "```json",
        json.dumps(timings, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evaluation",
        "",
        "```json",
        json.dumps(eval_data, ensure_ascii=False, indent=2),
        "```",
    ]
    (root / "artifacts/analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        steps = [row.get("global_step") for row in rows]
        rewards = [row.get("critic/rewards/mean", row.get("critic/score/mean", 0)) for row in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(steps, rewards, marker="o")
        plt.xlabel("update")
        plt.ylabel("mean reward")
        plt.tight_layout()
        plt.savefig(root / "artifacts/reward_curve.png", dpi=160)
        plt.close()
    except Exception as error:
        (root / "artifacts/reward_curve.error.txt").write_text(str(error), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subs = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", required=True, type=Path)
    start = subs.add_parser("init", parents=[common])
    start.add_argument("--mode", required=True)
    start.add_argument("--model", required=True, type=Path)
    start.add_argument("--project-root", required=True, type=Path)
    start.add_argument("--verl-root", required=True, type=Path)
    for name, typ in (("train-size", int), ("group-size", int), ("val-size", int), ("max-steps", int), ("training-steps", int), ("seed", int), ("learning-rate", float)):
        start.add_argument(f"--{name}", required=True, type=typ)
    done = subs.add_parser("finish", parents=[common])
    done.add_argument("--exit-code", required=True, type=int)
    subs.add_parser("analyze", parents=[common])
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    {"init": init, "finish": finish, "analyze": analyze}[args.command](args)

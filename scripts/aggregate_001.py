#!/usr/bin/env python3
"""Aggregate completed experiment-001 arm evaluations into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    methods = [name for name in ("base", "grpo", "bottleneckflow") if (args.run_root / name).is_dir()]
    rows = []
    result = {"run_root": str(args.run_root.resolve()), "methods": {}}
    for method in methods:
        method_result = {"evaluations": {}}
        for split in ["valid_seen", "valid_unseen"]:
            path = args.run_root / method / f"eval_{split}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            evaluation = {key: payload[key] for key in
                          ("num_games", "successes", "success_rate", "mean_reward", "truncated")}
            method_result["evaluations"][split] = evaluation
            rows.append({"method": method, "split": split, **evaluation})
        metrics_path = args.run_root / method / "metrics.jsonl"
        if metrics_path.is_file():
            updates = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            method_result["training"] = {
                "completed_updates": len(updates),
                "total_episodes": sum(item["episodes"] for item in updates),
                "total_successes": sum(item["successes"] for item in updates),
                "total_wall_seconds": sum(item["elapsed_seconds"] for item in updates),
                "best_train_success_rate": max(item["success_rate"] for item in updates),
                "final_update": updates[-1],
            }
            manifest_path = args.run_root / method / "manifest.json"
            if manifest_path.is_file():
                method_result["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["methods"][method] = method_result
    temporary = args.run_root / "results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.run_root / "results.json")
    with (args.run_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

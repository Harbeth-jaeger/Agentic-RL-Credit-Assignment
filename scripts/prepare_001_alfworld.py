#!/usr/bin/env python3
"""Create slime prompt manifests from official ALFWorld game files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tokens-per-turn", type=int, default=192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid_seen", "valid_unseen"):
        root = args.data_root / "json_2.1.1" / split
        gamefiles = sorted(root.rglob("game.tw-pddl"))
        if not gamefiles:
            raise FileNotFoundError(f"No ALFWorld game files below {root}")
        destination = args.output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for index, path in enumerate(gamefiles):
                row = {"prompt": [{"role": "user", "content":
                    "ALFWorld task is initialized by the custom rollout."}], "metadata": {
                    "gamefile": str(path.resolve()), "split": split,
                    "task_id": path.parent.name, "seed": args.seed + index,
                    # Put runtime limits in each Sample. Ray workers inherit the
                    # manifest reliably; they need not inherit shell variables.
                    "max_steps": args.max_steps,
                    "tokens_per_turn": args.tokens_per_turn}}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(gamefiles)} -> {destination}")

if __name__ == "__main__":
    main()

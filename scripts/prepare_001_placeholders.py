#!/usr/bin/env python3
"""Create the tiny modality/size parquet files verl-agent expects.

ALFWorld itself supplies tasks online; these rows only drive trainer batch sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def write(path: Path, size: int, split: str) -> None:
    rows = [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--val-size", required=True, type=int)
    args = parser.parse_args()
    write(args.output_dir / "train.parquet", args.train_size, "train")
    write(args.output_dir / "test.parquet", args.val_size, "valid_seen")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sample per-GPU memory and utilization until the parent process exits."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    with open(args.output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["timestamp", "gpu", "memory_used_mib", "memory_total_mib", "utilization_pct"])
        while True:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    writer.writerow([stamp, *[item.strip() for item in line.split(",")]])
                handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize arm-specific slime config so Ray workers see exact settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--tokens-per-turn", type=int, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.base.read_text(encoding="utf-8")) or {}
    config.update({
        "aca_method": args.method,
        "aca_audit_dir": str(args.audit_dir),
        "aca_arm_dir": str(args.audit_dir.parent),
        "alfworld_max_steps": args.max_steps,
        "alfworld_tokens_per_turn": args.tokens_per_turn,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids emitter-specific formatting.
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

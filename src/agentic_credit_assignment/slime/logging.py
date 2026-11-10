"""Durable JSON logging hooks for slime evaluation and later case analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    """Save summaries plus complete per-episode traces; keep slime's own logger too."""
    arm_dir = Path(args.aca_arm_dir)
    for split, payload in data.items():
        samples = payload.get("samples", [])
        rewards = [float(value) for value in payload["rewards"]]
        trace_dir = arm_dir / "eval_traces" / f"rollout_{rollout_id}" / split
        trace_paths = []
        for sample in samples:
            path = trace_dir / f"sample_{sample.index}.json"
            _atomic_json(path, {
                "sample_index": sample.index,
                "reward": float(sample.reward),
                "task_id": sample.metadata.get("task_id"),
                "gamefile": sample.metadata.get("gamefile"),
                "split": sample.metadata.get("split", split),
                "success": bool(sample.metadata.get("success", sample.reward > 0)),
                "truncated": bool(sample.metadata.get("truncated", False)),
                "timing": sample.metadata.get("timing", {}),
                "steps": sample.metadata.get("steps", []),
            })
            trace_paths.append(str(path.relative_to(arm_dir)))
        summary = {
            "rollout_id": rollout_id,
            "split": split,
            "num_games": len(rewards),
            "successes": sum(value > 0 for value in rewards),
            "success_rate": sum(value > 0 for value in rewards) / len(rewards) if rewards else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "truncated": sum(bool(x) for x in payload.get("truncated", [])),
            "trace_files": trace_paths,
        }
        _atomic_json(arm_dir / f"eval_{split}_rollout_{rollout_id}.json", summary)
        # The periodic evaluation with the largest rollout id is the latest
        # synchronized model and is what the experiment aggregator consumes.
        current = arm_dir / f"eval_{split}.json"
        if not current.exists() or rollout_id >= json.loads(current.read_text(encoding="utf-8"))["rollout_id"]:
            _atomic_json(current, summary)
    return False

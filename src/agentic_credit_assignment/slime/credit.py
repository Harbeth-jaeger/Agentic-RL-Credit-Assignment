"""Batch graph credit hook and token-level advantage hook for slime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from agentic_credit_assignment.credit import BottleneckConfig, assign_credit
from agentic_credit_assignment.credit.types import Trajectory, Transition


def _trajectory(sample) -> Trajectory:
    steps = sample.metadata["steps"]
    return Trajectory(
        trajectory_id=str(sample.index),
        transitions=tuple(Transition(x["state"], x["action"], x["next_state"]) for x in steps),
        reward=float(sample.reward), success=bool(sample.metadata["success"]),
        truncated=bool(sample.metadata["truncated"]),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_groups(args, groups) -> None:
    """Mutate Samples after a complete same-task GRPO group has been generated."""
    method = os.environ.get("ACA_METHOD", getattr(args, "aca_method", "grpo"))
    audit = Path(os.environ.get("ACA_AUDIT_DIR", args.aca_audit_dir))
    for group in groups:
        samples = list(group)
        trajectories = [_trajectory(sample) for sample in samples]
        config = BottleneckConfig(
            min_success_support=int(args.aca_min_success_support),
            min_success_coverage=float(args.aca_min_success_coverage),
            top_k=int(args.aca_top_k), credit_strength=float(args.aca_credit_strength),
        )
        # The untrained base arm still needs the same trace/audit schema, but it
        # does not have a separate credit estimator.  GRPO's terminal-return
        # assignment is the neutral representation for that read-only arm.
        credit_method = "grpo" if method == "base" else method
        result = assign_credit(credit_method, trajectories, config)
        task_id = samples[0].metadata["task_id"]
        group_id = samples[0].group_index
        for sample, turn_advantages in zip(samples, result.advantages, strict=True):
            sample.train_metadata = {
                **(sample.train_metadata or {}), "turn_advantages": list(turn_advantages),
                "method": method, "task_id": task_id, "group_index": group_id,
            }
            for step, credit in zip(sample.metadata["steps"], turn_advantages, strict=True):
                step["credit"] = credit
            # slime 0.4's DP partitioner currently drops arbitrary
            # train_metadata.  Carry the already-computed dense token credit in
            # the otherwise-unused OPD teacher sidecar, which is an official
            # per-token field preserved by every transport.  OPD is disabled in
            # 001, so this cannot affect any other loss term.
            dense_credit = [0.0] * sample.response_length
            for (begin, end), credit in zip(
                sample.train_metadata["turn_spans"], turn_advantages, strict=True
            ):
                dense_credit[int(begin):min(int(end), sample.response_length)] = [float(credit)] * max(
                    0, min(int(end), sample.response_length) - int(begin)
                )
            sample.teacher_log_probs = dense_credit
            unique_id = (sample.session_id or "no-session").replace("/", "-")
            _atomic_json(
                audit / "rollouts" / f"g{group_id}_s{sample.index}_{unique_id}.json",
                sample.metadata,
            )
        ranked = sorted(
            ({"sample": s.index, "turn": x["turn"], "state": x["state"], "action": x["action"],
              "next_state": x["next_state"], "credit": x.get("credit", 0.0)}
             for s in samples for x in s.metadata["steps"]),
            key=lambda x: abs(x["credit"]), reverse=True,
        )
        nodes = {x["state"] for s in samples for x in s.metadata["steps"]} | {
            x["next_state"] for s in samples for x in s.metadata["steps"]
        }
        group_uid = (samples[0].session_id or "no-session").replace("/", "-")
        _atomic_json(audit / "graphs" / f"group_{group_id}_{group_uid}.json", {
            "task_id": task_id, "group_index": group_id, "method": method,
            "num_nodes": len(nodes), "num_edges": sum(len(t.transitions) for t in trajectories),
            "successes": sum(t.success for t in trajectories), "diagnostics": result.diagnostics,
            "top_credit_edges": ranked[: int(args.aca_audit_top_k)],
        })


def token_advantages(args, rollout_data) -> None:
    """Expand stored per-turn A_it to response tokens; loss_mask excludes think/context."""
    sidecar = rollout_data.get("teacher_log_probs")
    if sidecar is None:
        raise KeyError("001 token-credit sidecar is missing from slime training batch")
    output = [values.to(dtype=torch.float32) for values in sidecar]
    rollout_data["advantages"] = output
    rollout_data["returns"] = [x.clone() for x in output]


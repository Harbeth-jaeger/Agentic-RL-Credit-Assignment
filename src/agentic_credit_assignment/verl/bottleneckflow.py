"""BottleneckFlow adapter for GraphGPO's step-expanded ``DataProto`` batch."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agentic_credit_assignment.credit.assigners import assign_bottleneckflow_from_base
from agentic_credit_assignment.credit.bottleneck import BottleneckConfig, bottleneck_credit_details
from agentic_credit_assignment.credit.types import Trajectory, Transition


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _config(config: Any) -> BottleneckConfig:
    raw = dict(config or {})
    names = {item.name for item in fields(BottleneckConfig)}
    return BottleneckConfig(**{key: value for key, value in raw.items() if key in names})


def _row_scalar(values: torch.Tensor, mask: torch.Tensor, row: int) -> float:
    selected = values[row][mask[row].bool()]
    return float(selected[0].detach().cpu()) if selected.numel() else 0.0


def _build_task_trajectories(batch: Any, task_uid: Any, graph: Any | None = None):
    nt = batch.non_tensor_batch
    indices = np.where(nt["uid"] == task_uid)[0]
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for row in indices:
        by_trajectory[str(nt["traj_uid"][row])].append(int(row))
    trajectories: list[Trajectory] = []
    row_groups: list[list[int]] = []
    for trajectory_id, rows in sorted(by_trajectory.items()):
        rows.sort()
        canonicalize = (
            graph.canonicalizer.canonicalize
            if graph is not None and getattr(graph, "canonicalizer", None) is not None
            else lambda value: value
        )
        transitions = tuple(
            Transition(
                state=str(canonicalize(nt["anchor_obs"][row])),
                action=str(nt.get("executed_actions", nt["text_actions"])[row]),
                next_state=str(canonicalize(nt["next_obs"][row])),
            )
            for row in rows
        )
        reward = float(nt["episode_rewards"][rows[0]])
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                transitions=transitions,
                reward=reward,
                success=reward > 0,
                truncated=str(transitions[-1].next_state) != "success" if transitions else True,
            )
        )
        row_groups.append(rows)
    return trajectories, row_groups


def _graph_record(task_uid: Any, trajectories: list[Trajectory]) -> dict[str, Any]:
    node_frequency: Counter[str] = Counter()
    edge_frequency: Counter[tuple[str, str, str]] = Counter()
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            node_frequency[transition.state] += 1
            node_frequency[transition.next_state] += 1
            edge_frequency[transition.edge] += 1
    return {
        "task_uid": str(task_uid),
        "node_count": len(node_frequency),
        "edge_count": len(edge_frequency),
        "nodes": [
            {"state": state, "frequency": frequency}
            for state, frequency in node_frequency.most_common()
        ],
        "edges": [
            {"state": edge[0], "action": edge[1], "next_state": edge[2], "frequency": frequency}
            for edge, frequency in edge_frequency.most_common()
        ],
    }


def _trace_step(nt: dict[str, Any], row: int, local_step: int) -> dict[str, Any]:
    size = len(nt["uid"])
    return {
        "step": local_step,
        "prompt": str(nt.get("formatted_prompt", np.array([""] * size, dtype=object))[row]),
        "state": str(nt["anchor_obs"][row]),
        "raw_response": str(nt["text_actions"][row]),
        "action": str(nt.get("executed_actions", nt["text_actions"])[row]),
        "action_valid": bool(nt.get("is_action_valid", np.ones(size))[row]),
        "raw_observation": str(nt.get("raw_observations", np.array([""] * size, dtype=object))[row]),
        "next_state": str(nt["next_obs"][row]),
        "environment_reward": float(nt.get("rewards", np.zeros(size))[row]),
        "done": bool(nt.get("environment_dones", np.zeros(size, dtype=bool))[row]),
        "won": float(nt.get("environment_won", np.zeros(size))[row]),
        "goal_condition_success_rate": float(
            nt.get("goal_condition_success_rate", np.zeros(size))[row]
        ),
    }


def append_eval_rollouts(batch: Any, global_step: int, rollout_path: str | None) -> None:
    """Persist every validation trajectory using the same schema as training."""

    if not rollout_path:
        return
    nt = batch.non_tensor_batch
    for task_uid in sorted(set(nt["uid"]), key=str):
        trajectories, row_groups = _build_task_trajectories(batch, task_uid)
        record = {
            "phase": "eval",
            "global_step": global_step,
            "task_uid": str(task_uid),
            "task_description": "",
            "trajectories": [],
        }
        for trajectory, rows in zip(trajectories, row_groups):
            if rows and not record["task_description"]:
                record["task_description"] = str(
                    nt.get("task_description", np.array([""] * len(nt["uid"]), dtype=object))[rows[0]]
                )
            steps = [_trace_step(nt, row, local_step) for local_step, row in enumerate(rows)]
            final = steps[-1] if steps else {}
            record["trajectories"].append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "R": trajectory.reward,
                    "success": trajectory.success,
                    "terminal": {
                        "done": final.get("done", False),
                        "won": final.get("won", 0.0),
                        "goal_condition_success_rate": final.get(
                            "goal_condition_success_rate", 0.0
                        ),
                        "raw_observation": final.get("raw_observation", ""),
                        "next_state": final.get("next_state", ""),
                    },
                    "steps": steps,
                }
            )
        append_jsonl(rollout_path, record)


def apply_bottleneckflow(
    batch: Any,
    episode_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    graphs: dict[Any, Any] | None,
    config: Any,
    global_step: int,
    rollout_path: str | None = None,
    graph_path: str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Apply B+C weights to official episode advantages and emit audits."""

    cfg = _config(config)
    advantages = episode_advantages.clone()
    metrics: dict[str, list[float]] = defaultdict(list)
    nt = batch.non_tensor_batch

    for task_uid in sorted(set(nt["uid"]), key=str):
        trajectories, row_groups = _build_task_trajectories(
            batch, task_uid, None if graphs is None else graphs.get(task_uid)
        )
        base_by_trajectory = [
            _row_scalar(episode_advantages, response_mask, rows[0]) for rows in row_groups
        ]
        result = assign_bottleneckflow_from_base(trajectories, base_by_trajectory, cfg)
        raw_credits, selected, segments, _ = bottleneck_credit_details(trajectories, cfg)

        rollout_record = {
            "phase": "train",
            "global_step": global_step,
            "task_uid": str(task_uid),
            "task_description": "",
            "diagnostics": result.diagnostics,
            "key_states": [asdict(item) for item in selected],
            "segments": [asdict(item) for item in segments],
            "trajectories": [],
        }
        for trajectory, rows, base_advantage, step_advantages, trajectory_credits in zip(
            trajectories, row_groups, base_by_trajectory, result.advantages, raw_credits
        ):
            if len(trajectory_credits) != len(rows):
                trajectory_credits = [0.0] * len(rows)
            if rows and not rollout_record["task_description"]:
                rollout_record["task_description"] = str(
                    nt.get("task_description", np.array([""] * len(nt["uid"]), dtype=object))[rows[0]]
                )
            steps = []
            for local_step, (row, transition, a_it, credit) in enumerate(
                zip(rows, trajectory.transitions, step_advantages, trajectory_credits)
            ):
                omega = a_it / base_advantage if abs(base_advantage) > 1e-12 else 1.0
                advantages[row] = a_it * response_mask[row]
                step_record = _trace_step(nt, row, local_step)
                step_record.update(
                    {
                        "state": transition.state,
                        "action": transition.action,
                        "next_state": transition.next_state,
                        "credit": credit,
                        "omega": omega,
                        "episode_advantage": base_advantage,
                        "A_it": a_it,
                    }
                )
                steps.append(step_record)
                metrics["omega"].append(omega)
                metrics["A_it"].append(a_it)
            rollout_record["trajectories"].append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "R": trajectory.reward,
                    "success": trajectory.success,
                    "terminal": {
                        "done": steps[-1].get("done", False) if steps else False,
                        "won": steps[-1].get("won", 0.0) if steps else 0.0,
                        "goal_condition_success_rate": steps[-1].get(
                            "goal_condition_success_rate", 0.0
                        ) if steps else 0.0,
                        "raw_observation": steps[-1].get("raw_observation", "") if steps else "",
                        "next_state": steps[-1].get("next_state", "") if steps else "",
                    },
                    "steps": steps,
                }
            )
        if rollout_path:
            append_jsonl(rollout_path, rollout_record)
        if graph_path:
            graph = _graph_record(task_uid, trajectories)
            graph.update({"global_step": global_step, "key_states": [asdict(item) for item in selected]})
            append_jsonl(graph_path, graph)

        metrics["selected_nodes"].append(float(result.diagnostics.get("selected_nodes", 0)))
        metrics["credited_edges"].append(float(result.diagnostics.get("credited_edges", 0)))
        metrics["fallback"].append(float(result.diagnostics.get("fallback", "none") != "none"))

    batch.batch["advantages"] = advantages
    batch.batch["returns"] = advantages
    reduced = {
        f"bottleneckflow/{name}_{suffix}": function(values)
        for name, values in metrics.items()
        if values
        for suffix, function in (("mean", np.mean), ("min", np.min), ("max", np.max))
    }
    return batch, {key: float(value) for key, value in reduced.items()}

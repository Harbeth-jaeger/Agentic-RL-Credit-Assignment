"""Project-owned GRPO and graph credit assigners for experiment 001."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

from .types import Trajectory
from .bottleneck import BottleneckConfig, bottleneck_credits, map_edge_credit, select_key_states


@dataclass(frozen=True)
class CreditResult:
    advantages: tuple[tuple[float, ...], ...]
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


def _trajectory_advantages(trajectories: Sequence[Trajectory], eps: float = 1e-6) -> list[float]:
    rewards = [item.reward for item in trajectories]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance <= eps * eps:
        return [0.0] * len(rewards)
    std = math.sqrt(variance)
    return [(reward - mean) / (std + eps) for reward in rewards]


def _constant_turns(trajectories: Sequence[Trajectory], values: Sequence[float]) -> CreditResult:
    return CreditResult(
        advantages=tuple(
            tuple(values[index] for _ in trajectory.transitions)
            for index, trajectory in enumerate(trajectories)
        ),
        diagnostics={"nonzero_turns": sum(value != 0 for value in values)},
    )


def _project_weights(raw: Sequence[float], low: float = 0.5, high: float = 2.0) -> list[float]:
    """Project positive weights to bounds while preserving an unweighted mean of one."""

    if not raw:
        return []
    weights = [min(high, max(low, value)) for value in raw]
    for _ in range(32):
        residual = len(weights) - sum(weights)
        if abs(residual) <= 1e-8:
            break
        free = [i for i, value in enumerate(weights) if low + 1e-10 < value < high - 1e-10]
        if not free:
            break
        delta = residual / len(free)
        for index in free:
            weights[index] = min(high, max(low, weights[index] + delta))
    return weights


def _weighted_turns(
    trajectories: Sequence[Trajectory],
    trajectory_advantages: Sequence[float],
    credits: Sequence[Sequence[float]],
    strength: float = 0.5,
) -> CreditResult:
    output: list[tuple[float, ...]] = []
    changed = 0
    for trajectory, advantage, values in zip(trajectories, trajectory_advantages, credits):
        if len(values) != len(trajectory.transitions):
            raise ValueError("Credit length does not match trajectory length")
        if advantage == 0 or not values:
            output.append(tuple(advantage for _ in values))
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        if variance <= 1e-12:
            output.append(tuple(advantage for _ in values))
            continue
        std = math.sqrt(variance)
        direction = 1.0 if advantage > 0 else -1.0
        raw = [math.exp(direction * strength * (value - mean) / (std + 1e-6)) for value in values]
        weights = _project_weights(raw)
        changed += sum(abs(weight - 1.0) > 1e-6 for weight in weights)
        output.append(tuple(advantage * weight for weight in weights))
    return CreditResult(tuple(output), {"changed_turns": changed})


def _support_maps(trajectories: Sequence[Trajectory]):
    node_success: dict[str, set[str]] = defaultdict(set)
    node_failure: dict[str, set[str]] = defaultdict(set)
    edge_success: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    edge_failure: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for trajectory in trajectories:
        node_target = node_success if trajectory.success else node_failure
        edge_target = edge_success if trajectory.success else edge_failure
        for node in {transition.next_state for transition in trajectory.transitions}:
            node_target[node].add(trajectory.trajectory_id)
        for edge in {transition.edge for transition in trajectory.transitions}:
            edge_target[edge].add(trajectory.trajectory_id)
    return node_success, node_failure, edge_success, edge_failure


def _node_importance(
    trajectories: Sequence[Trajectory], alpha: float = 1.0, min_success_support: int = 2
) -> dict[str, float]:
    successful = [item for item in trajectories if item.success]
    failures = [item for item in trajectories if not item.success]
    if len(successful) < min_success_support:
        return {}
    node_success, node_failure, _, _ = _support_maps(trajectories)
    starts = {item.transitions[0].state for item in trajectories if item.transitions}
    importance: dict[str, float] = {}
    for node, support in node_success.items():
        if node in starts or node in {"success", "failure"} or len(support) < min_success_support:
            continue
        p_success = (len(support) + alpha) / (len(successful) + 2 * alpha)
        p_failure = (len(node_failure[node]) + alpha) / (len(failures) + 2 * alpha)
        score = p_success * max(0.0, math.log(p_success / p_failure))
        if score > 0:
            importance[node] = score
    return importance


def _success_visitation(trajectories: Sequence[Trajectory], trajectory_advantages: Sequence[float]):
    importance = _node_importance(trajectories)
    if not importance:
        result = _constant_turns(trajectories, trajectory_advantages)
        return CreditResult(result.advantages, {"fallback": "no_contrastive_node"})
    credits = [
        [importance.get(transition.next_state, 0.0) for transition in trajectory.transitions]
        for trajectory in trajectories
    ]
    result = _weighted_turns(trajectories, trajectory_advantages, credits)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({"selected_nodes": len(importance), "fallback": "none"})
    return CreditResult(result.advantages, diagnostics)


def _distances_to_success(trajectories: Sequence[Trajectory]) -> dict[str, int]:
    reverse: dict[str, set[str]] = defaultdict(set)
    goals: set[str] = set()
    for trajectory in trajectories:
        if not trajectory.success:
            continue
        for transition in trajectory.transitions:
            reverse[transition.next_state].add(transition.state)
        if trajectory.transitions:
            goals.add(trajectory.transitions[-1].next_state)
    distances = {goal: 0 for goal in goals}
    queue = deque(goals)
    while queue:
        node = queue.popleft()
        for parent in reverse[node]:
            if parent not in distances:
                distances[parent] = distances[node] + 1
                queue.append(parent)
    return distances


def _graph_progress(trajectories: Sequence[Trajectory], trajectory_advantages: Sequence[float]):
    distances = _distances_to_success(trajectories)
    if not distances:
        result = _constant_turns(trajectories, trajectory_advantages)
        return CreditResult(result.advantages, {"fallback": "no_success_path"})
    credits = []
    for trajectory in trajectories:
        values = []
        for transition in trajectory.transitions:
            before = distances.get(transition.state)
            after = distances.get(transition.next_state)
            values.append(float(before - after) if before is not None and after is not None else 0.0)
        credits.append(values)
    result = _weighted_turns(trajectories, trajectory_advantages, credits)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({"reachable_nodes": len(distances), "fallback": "none"})
    return CreditResult(result.advantages, diagnostics)


def _fallback_result(trajectories, trajectory_advantages, diagnostics):
    result = _constant_turns(trajectories, trajectory_advantages)
    merged = dict(result.diagnostics)
    merged.update(diagnostics)
    return CreditResult(result.advantages, merged)


def _node_only(trajectories, trajectory_advantages, config: BottleneckConfig):
    selected, diagnostics = select_key_states(trajectories, config)
    if not selected:
        return _fallback_result(trajectories, trajectory_advantages, diagnostics)
    importance = {item.node: item.importance for item in selected}
    credits = [
        [importance.get(transition.next_state, 0.0) for transition in trajectory.transitions]
        for trajectory in trajectories
    ]
    result = _weighted_turns(
        trajectories, trajectory_advantages, credits, strength=config.credit_strength
    )
    diagnostics.update(result.diagnostics)
    diagnostics["fallback"] = "none"
    return CreditResult(result.advantages, diagnostics)


def _ancestor_flow(trajectories, trajectory_advantages, config: BottleneckConfig):
    selected, diagnostics = select_key_states(trajectories, config)
    if not selected:
        return _fallback_result(trajectories, trajectory_advantages, diagnostics)
    edge_credit: dict[tuple[str, str, str], float] = defaultdict(float)
    max_error = 0.0
    for state in selected:
        prefixes = []
        for trajectory in trajectories:
            if not trajectory.success:
                continue
            prefix = []
            for transition in trajectory.transitions:
                prefix.append(transition.edge)
                if transition.next_state == state.node:
                    prefixes.append(tuple(dict.fromkeys(prefix)))
                    break
        support: dict[tuple[str, str, str], int] = defaultdict(int)
        for prefix in prefixes:
            for edge in prefix:
                support[edge] += 1
        denominator = sum(support.values())
        if denominator == 0:
            continue
        allocated = 0.0
        for edge, count in support.items():
            value = state.importance * count / denominator
            edge_credit[edge] += value
            allocated += value
        max_error = max(max_error, abs(allocated - state.importance))
    if not edge_credit:
        diagnostics["fallback"] = "no_ancestor_credit"
        return _fallback_result(trajectories, trajectory_advantages, diagnostics)
    result = _weighted_turns(
        trajectories, trajectory_advantages, map_edge_credit(trajectories, edge_credit),
        strength=config.credit_strength,
    )
    diagnostics.update(result.diagnostics)
    diagnostics.update({"fallback": "none", "credited_edges": len(edge_credit),
                        "max_conservation_error": max_error})
    return CreditResult(result.advantages, diagnostics)


def _bottleneck_flow(trajectories, trajectory_advantages, config: BottleneckConfig):
    credits, diagnostics = bottleneck_credits(trajectories, config)
    if diagnostics.get("fallback") != "none":
        return _fallback_result(trajectories, trajectory_advantages, diagnostics)
    result = _weighted_turns(
        trajectories, trajectory_advantages, credits, strength=config.credit_strength
    )
    diagnostics.update(result.diagnostics)
    return CreditResult(result.advantages, diagnostics)


def assign_bottleneckflow_from_base(
    trajectories: Sequence[Trajectory],
    trajectory_advantages: Sequence[float],
    bottleneck_config: BottleneckConfig | None = None,
) -> CreditResult:
    """Redistribute an externally computed episode advantage over its steps.

    GraphGPO/veRL computes the episode advantage with step-weighted grouping.
    Experiment 001 deliberately preserves that baseline and only replaces the
    within-trajectory credit distribution, hence this separate entry point.
    """

    if len(trajectories) != len(trajectory_advantages):
        raise ValueError("One base advantage is required for every trajectory")
    return _bottleneck_flow(
        trajectories,
        trajectory_advantages,
        bottleneck_config or BottleneckConfig(),
    )


def assign_credit(
    method: str,
    trajectories: Sequence[Trajectory],
    bottleneck_config: BottleneckConfig | None = None,
) -> CreditResult:
    """Assign turn-level advantages; every method shares the same GRPO outcome base."""

    if not trajectories:
        raise ValueError("At least one trajectory is required")
    trajectory_advantages = _trajectory_advantages(trajectories)
    if method == "grpo":
        return _constant_turns(trajectories, trajectory_advantages)
    if method == "graph_progress":
        return _graph_progress(trajectories, trajectory_advantages)
    if method == "success_visitation":
        return _success_visitation(trajectories, trajectory_advantages)
    config = bottleneck_config or BottleneckConfig()
    if method == "node_only":
        return _node_only(trajectories, trajectory_advantages, config)
    if method == "ancestor_flow":
        return _ancestor_flow(trajectories, trajectory_advantages, config)
    if method == "bottleneckflow":
        return _bottleneck_flow(trajectories, trajectory_advantages, config)
    raise ValueError(f"Unknown credit method: {method}")

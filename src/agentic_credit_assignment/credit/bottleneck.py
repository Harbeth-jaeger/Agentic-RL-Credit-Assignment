"""BottleneckFlow B+C credit assignment for one same-task rollout group."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Sequence

from scipy.stats import beta as beta_distribution

from .types import Trajectory

Edge = tuple[str, str, str]


@dataclass(frozen=True)
class BottleneckConfig:
    min_success_support: int = 2
    min_success_coverage: float = 0.5
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    lcb_quantile: float = 0.05
    top_k: int = 3
    remaining_path_decay: float = 0.1
    coverage_weight: float = 0.5
    earliness_weight: float = 1.0
    efficiency_weight: float = 0.5
    dedup_success_jaccard: float = 0.9
    dedup_max_distance: int = 2
    key_to_key_bonus: float = 1.0
    segment_length_penalty: float = 0.2
    failure_background: float = 0.5
    local_advantage_weight: float = 0.5
    flow_temperature: float = 0.5
    credit_strength: float = 0.5


@dataclass(frozen=True)
class KeyState:
    node: str
    importance: float
    success_support: int
    failure_support: int
    coverage: float
    posterior_mean: float
    lcb: float
    earliness: float
    efficiency: float


@dataclass(frozen=True)
class Segment:
    trajectory_id: str
    source: str
    target: str
    edges: tuple[Edge, ...]


def _support_sets(trajectories: Sequence[Trajectory]):
    node_success: dict[str, set[str]] = defaultdict(set)
    node_failure: dict[str, set[str]] = defaultdict(set)
    edge_success: dict[Edge, set[str]] = defaultdict(set)
    edge_failure: dict[Edge, set[str]] = defaultdict(set)
    for trajectory in trajectories:
        node_map = node_success if trajectory.success else node_failure
        edge_map = edge_success if trajectory.success else edge_failure
        nodes = {transition.state for transition in trajectory.transitions}
        nodes.update(transition.next_state for transition in trajectory.transitions)
        for node in nodes:
            node_map[node].add(trajectory.trajectory_id)
        for edge in {transition.edge for transition in trajectory.transitions}:
            edge_map[edge].add(trajectory.trajectory_id)
    return node_success, node_failure, edge_success, edge_failure


def _lcb(successes: int, failures: int, config: BottleneckConfig) -> float:
    return float(
        beta_distribution.ppf(
            config.lcb_quantile,
            successes + config.beta_alpha,
            failures + config.beta_beta,
        )
    )


def _undirected_distances(trajectories: Sequence[Trajectory], source: str, max_distance: int):
    neighbors: dict[str, set[str]] = defaultdict(set)
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            neighbors[transition.state].add(transition.next_state)
            neighbors[transition.next_state].add(transition.state)
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if distances[node] >= max_distance:
            continue
        for other in neighbors[node]:
            if other not in distances:
                distances[other] = distances[node] + 1
                queue.append(other)
    return distances


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def select_key_states(
    trajectories: Sequence[Trajectory], config: BottleneckConfig
) -> tuple[list[KeyState], dict[str, float | int | str]]:
    """Apply scheme B: support/LCB gate, ranking, then structural dedup."""

    successful = [trajectory for trajectory in trajectories if trajectory.success]
    failures = [trajectory for trajectory in trajectories if not trajectory.success]
    if len(successful) < config.min_success_support:
        return [], {"fallback": "insufficient_successes", "successes": len(successful)}
    node_success, node_failure, _, _ = _support_sets(trajectories)
    starts = {trajectory.transitions[0].state for trajectory in trajectories if trajectory.transitions}
    baseline_lcb = _lcb(len(successful), len(failures), config)
    candidates: list[KeyState] = []
    gate_counts: dict[str, int] = defaultdict(int)
    for node, support in node_success.items():
        if node in starts or node in {"success", "failure"}:
            gate_counts["start_or_terminal"] += 1
            continue
        positive = len(support)
        negative = len(node_failure[node])
        coverage = positive / len(successful)
        if positive < config.min_success_support:
            gate_counts["support"] += 1
            continue
        if coverage < config.min_success_coverage:
            gate_counts["coverage"] += 1
            continue
        lcb = _lcb(positive, negative, config)
        if lcb <= baseline_lcb:
            gate_counts["lcb"] += 1
            continue
        early_values: list[float] = []
        efficiency_values: list[float] = []
        for trajectory in successful:
            first = next(
                (
                    index + 1
                    for index, transition in enumerate(trajectory.transitions)
                    if transition.next_state == node
                ),
                None,
            )
            if first is None:
                continue
            length = len(trajectory.transitions)
            early_values.append(1.0 - first / max(length, 1))
            efficiency_values.append(
                math.exp(-config.remaining_path_decay * max(0, length - first))
            )
        earliness = sum(early_values) / len(early_values)
        efficiency = sum(efficiency_values) / len(efficiency_values)
        posterior_mean = (positive + config.beta_alpha) / (
            positive + negative + config.beta_alpha + config.beta_beta
        )
        importance = (
            lcb
            * coverage**config.coverage_weight
            * max(earliness, 1e-12) ** config.earliness_weight
            * max(efficiency, 1e-12) ** config.efficiency_weight
        )
        candidates.append(
            KeyState(
                node, importance, positive, negative, coverage, posterior_mean, lcb,
                earliness, efficiency,
            )
        )
    candidates.sort(key=lambda item: (-item.importance, item.node))
    selected: list[KeyState] = []
    for candidate in candidates:
        distances = _undirected_distances(
            trajectories, candidate.node, config.dedup_max_distance
        )
        duplicate = False
        for kept in selected:
            if distances.get(kept.node, config.dedup_max_distance + 1) > config.dedup_max_distance:
                continue
            success_similarity = _jaccard(node_success[candidate.node], node_success[kept.node])
            failure_similarity = _jaccard(node_failure[candidate.node], node_failure[kept.node])
            if (
                success_similarity >= config.dedup_success_jaccard
                and failure_similarity >= config.dedup_success_jaccard
            ):
                duplicate = True
                gate_counts["dedup"] += 1
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) == config.top_k:
            break
    diagnostics: dict[str, float | int | str] = {
        "fallback": "none" if selected else "no_key_state",
        "candidate_nodes": len(candidates),
        "selected_nodes": len(selected),
        "task_baseline_lcb": baseline_lcb,
    }
    diagnostics.update({f"filtered_{key}": value for key, value in gate_counts.items()})
    for rank, state in enumerate(selected, start=1):
        prefix = f"key_{rank}"
        diagnostics.update(
            {
                f"{prefix}_node": state.node,
                f"{prefix}_importance": state.importance,
                f"{prefix}_coverage": state.coverage,
                f"{prefix}_lcb": state.lcb,
                f"{prefix}_earliness": state.earliness,
                f"{prefix}_efficiency": state.efficiency,
            }
        )
    return selected, diagnostics


def _segments_to_keys(
    trajectories: Sequence[Trajectory], key_states: Sequence[KeyState]
) -> list[Segment]:
    key_nodes = {state.node for state in key_states}
    segments: list[Segment] = []
    for trajectory in trajectories:
        if not trajectory.success or not trajectory.transitions:
            continue
        milestones: list[tuple[int, str]] = []
        seen: set[str] = set()
        for index, transition in enumerate(trajectory.transitions):
            node = transition.next_state
            if node in key_nodes and node not in seen:
                milestones.append((index, node))
                seen.add(node)
        previous_index = -1
        previous_node = "__start__"
        for index, node in milestones:
            edges = tuple(
                transition.edge
                for transition in trajectory.transitions[previous_index + 1 : index + 1]
            )
            if edges:
                segments.append(Segment(trajectory.trajectory_id, previous_node, node, edges))
            previous_index = index
            previous_node = node
    return segments


def _local_action_advantage(trajectories: Sequence[Trajectory]) -> dict[tuple[str, str], float]:
    state_total: dict[str, set[str]] = defaultdict(set)
    state_success: dict[str, set[str]] = defaultdict(set)
    action_total: dict[tuple[str, str], set[str]] = defaultdict(set)
    action_success: dict[tuple[str, str], set[str]] = defaultdict(set)
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            state_total[transition.state].add(trajectory.trajectory_id)
            action_total[(transition.state, transition.action)].add(trajectory.trajectory_id)
            if trajectory.success:
                state_success[transition.state].add(trajectory.trajectory_id)
                action_success[(transition.state, transition.action)].add(trajectory.trajectory_id)
    result = {}
    for key, support in action_total.items():
        state = key[0]
        action_rate = (len(action_success[key]) + 1.0) / (len(support) + 2.0)
        state_rate = (len(state_success[state]) + 1.0) / (len(state_total[state]) + 2.0)
        result[key] = action_rate - state_rate
    return result


def segment_flow_credit(
    trajectories: Sequence[Trajectory],
    key_states: Sequence[KeyState],
    config: BottleneckConfig,
) -> tuple[dict[Edge, float], dict[str, float | int | str]]:
    """Apply scheme C with target- and segment-level conservation."""

    importance = {state.node: state.importance for state in key_states}
    segments = _segments_to_keys(trajectories, key_states)
    grouped: dict[tuple[str, str], list[Segment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.source, segment.target)].append(segment)
    by_target: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for route in grouped:
        by_target[route[1]].append(route)
    failures = [trajectory for trajectory in trajectories if not trajectory.success]
    _, _, _, edge_failure = _support_sets(trajectories)
    local_advantage = _local_action_advantage(trajectories)
    edge_credit: dict[Edge, float] = defaultdict(float)
    max_target_error = 0.0
    max_segment_error = 0.0
    route_count = 0
    for target, routes in by_target.items():
        route_scores: dict[tuple[str, str], float] = {}
        reaching_ids = {segment.trajectory_id for route in routes for segment in grouped[route]}
        for route in routes:
            route_segments = grouped[route]
            coverage = len({segment.trajectory_id for segment in route_segments}) / len(reaching_ids)
            mean_length = sum(len(segment.edges) for segment in route_segments) / len(route_segments)
            source_importance = importance.get(route[0], 0.0)
            bonus = 1.0 + config.key_to_key_bonus * math.sqrt(
                source_importance * importance[target]
            )
            route_scores[route] = coverage * bonus * math.exp(
                -config.segment_length_penalty * mean_length
            )
        denominator = sum(route_scores.values())
        allocated_target = 0.0
        for route, route_score in route_scores.items():
            mass = importance[target] * route_score / denominator
            allocated_target += mass
            route_segments = grouped[route]
            edge_support: dict[Edge, set[str]] = defaultdict(set)
            for segment in route_segments:
                for edge in set(segment.edges):
                    edge_support[edge].add(segment.trajectory_id)
            logits: dict[Edge, float] = {}
            for edge, support in edge_support.items():
                positive = len(support) / len(route_segments)
                negative = len(edge_failure[edge]) / len(failures) if failures else 0.0
                reliability = len(support) / (len(support) + 1.0)
                value = reliability * max(
                    0.0,
                    positive
                    - config.failure_background * negative
                    + config.local_advantage_weight
                    * local_advantage.get((edge[0], edge[1]), 0.0),
                )
                logits[edge] = value
            if not logits:
                continue
            maximum = max(logits.values())
            weights = {
                edge: math.exp((value - maximum) / config.flow_temperature)
                for edge, value in logits.items()
            }
            weight_sum = sum(weights.values())
            allocated_segment = 0.0
            for edge, weight in weights.items():
                value = mass * weight / weight_sum
                edge_credit[edge] += value
                allocated_segment += value
            max_segment_error = max(max_segment_error, abs(allocated_segment - mass))
            route_count += 1
        max_target_error = max(max_target_error, abs(allocated_target - importance[target]))
    return dict(edge_credit), {
        "segments": len(segments),
        "routes": route_count,
        "credited_edges": len(edge_credit),
        "max_target_conservation_error": max_target_error,
        "max_segment_conservation_error": max_segment_error,
    }


def map_edge_credit(
    trajectories: Sequence[Trajectory], edge_credit: dict[Edge, float]
) -> list[list[float]]:
    credits: list[list[float]] = []
    for trajectory in trajectories:
        seen: set[Edge] = set()
        values: list[float] = []
        for transition in trajectory.transitions:
            if transition.edge in seen:
                values.append(0.0)
            else:
                values.append(edge_credit.get(transition.edge, 0.0))
                seen.add(transition.edge)
        credits.append(values)
    return credits


def bottleneck_credits(
    trajectories: Sequence[Trajectory], config: BottleneckConfig
) -> tuple[list[list[float]], dict[str, float | int | str]]:
    credits, _, _, diagnostics = bottleneck_credit_details(trajectories, config)
    return credits, diagnostics


def bottleneck_credit_details(
    trajectories: Sequence[Trajectory], config: BottleneckConfig
) -> tuple[
    list[list[float]],
    list[KeyState],
    list[Segment],
    dict[str, float | int | str],
]:
    """Return credit plus the node/segment objects needed for experiment audits."""

    key_states, diagnostics = select_key_states(trajectories, config)
    if not key_states:
        return [[] for _ in trajectories], [], [], diagnostics
    segments = _segments_to_keys(trajectories, key_states)
    edge_credit, flow_diagnostics = segment_flow_credit(trajectories, key_states, config)
    diagnostics.update(flow_diagnostics)
    if not edge_credit:
        diagnostics["fallback"] = "no_segment_credit"
        return [[] for _ in trajectories], key_states, segments, diagnostics
    diagnostics["fallback"] = "none"
    return map_edge_credit(trajectories, edge_credit), key_states, segments, diagnostics


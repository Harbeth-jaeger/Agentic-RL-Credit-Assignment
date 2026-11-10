import unittest

from agentic_credit_assignment.credit import Trajectory, Transition
from agentic_credit_assignment.credit.bottleneck import (
    BottleneckConfig,
    segment_flow_credit,
    select_key_states,
)


def make_trajectory(name, states, success):
    transitions = tuple(
        Transition(states[index], f"{states[index]}->{states[index + 1]}", states[index + 1])
        for index in range(len(states) - 1)
    )
    return Trajectory(name, transitions, float(success), success)


class BottleneckCreditTest(unittest.TestCase):
    def setUp(self):
        self.group = [
            make_trajectory("p1", ["s0", "s10", "x1", "s8", "success"], True),
            make_trajectory("p2", ["s0", "s10", "s8", "success"], True),
            make_trajectory("p3", ["s0", "s6", "a", "b", "s8", "success"], True),
            make_trajectory("p4", ["s0", "s6", "c", "d", "e", "s8", "success"], True),
            make_trajectory("n1", ["s0", "s6", "s7", "failure"], False),
            make_trajectory("n2", ["s0", "q", "failure"], False),
            make_trajectory("n3", ["s0", "r", "failure"], False),
            make_trajectory("n4", ["s0", "t", "failure"], False),
            make_trajectory("n5", ["s0", "u", "failure"], False),
            make_trajectory("n6", ["s0", "v", "failure"], False),
        ]
        self.config = BottleneckConfig(top_k=3, dedup_max_distance=1)

    def test_scheme_b_recovers_expected_order(self):
        selected, diagnostics = select_key_states(self.group, self.config)
        self.assertEqual(diagnostics["fallback"], "none")
        self.assertEqual([state.node for state in selected], ["s10", "s8", "s6"])

    def test_scheme_c_prefers_short_key_to_key_route_and_conserves(self):
        selected, _ = select_key_states(self.group, self.config)
        edge_credit, diagnostics = segment_flow_credit(self.group, selected, self.config)
        s10_route = sum(
            value for edge, value in edge_credit.items() if edge[0] in {"s10", "x1"}
        )
        s6_route = sum(
            value
            for edge, value in edge_credit.items()
            if edge[0] in {"s6", "a", "b", "c", "d", "e"} and edge[2] != "s7"
        )
        self.assertGreater(s10_route, s6_route)
        self.assertLessEqual(diagnostics["max_target_conservation_error"], 1e-10)
        self.assertLessEqual(diagnostics["max_segment_conservation_error"], 1e-10)

    def test_repeated_node_counts_as_one_trajectory_support(self):
        loop = make_trajectory("loop", ["s0", "s10", "s0", "s10", "s8", "success"], True)
        group = [loop, *self.group[1:]]
        selected, _ = select_key_states(group, self.config)
        s10 = next(state for state in selected if state.node == "s10")
        self.assertEqual(s10.success_support, 2)

    def test_all_fail_has_explicit_fallback(self):
        group = [
            make_trajectory("n1", ["s0", "x", "failure"], False),
            make_trajectory("n2", ["s0", "y", "failure"], False),
        ]
        selected, diagnostics = select_key_states(group, self.config)
        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["fallback"], "insufficient_successes")


if __name__ == "__main__":
    unittest.main()

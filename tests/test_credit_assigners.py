import unittest

from agentic_credit_assignment.credit import Trajectory, Transition, assign_credit


def trajectory(name, states, reward, success):
    transitions = tuple(
        Transition(states[index], f"a{index}", states[index + 1])
        for index in range(len(states) - 1)
    )
    return Trajectory(name, transitions, reward, success)


class CreditAssignerTest(unittest.TestCase):
    def setUp(self):
        self.group = [
            trajectory("p1", ["s0", "x", "b", "success"], 1.0, True),
            trajectory("p2", ["s0", "y", "b", "success"], 1.0, True),
            trajectory("n1", ["s0", "z", "failure"], 0.0, False),
            trajectory("n2", ["s0", "q", "failure"], 0.0, False),
        ]

    def test_grpo_is_constant_within_trajectory(self):
        result = assign_credit("grpo", self.group)
        self.assertTrue(all(len(set(turns)) == 1 for turns in result.advantages))
        self.assertGreater(result.advantages[0][0], 0)
        self.assertLess(result.advantages[-1][0], 0)

    def test_all_equal_rewards_have_zero_advantage(self):
        group = [trajectory("a", ["s", "x"], 0, False), trajectory("b", ["s", "y"], 0, False)]
        for method in ["grpo", "graph_progress", "success_visitation", "bottleneckflow"]:
            result = assign_credit(method, group)
            self.assertEqual(result.advantages, ((0.0,), (0.0,)))

    def test_success_visitation_changes_turn_weights(self):
        result = assign_credit("success_visitation", self.group)
        self.assertEqual(result.diagnostics["fallback"], "none")
        self.assertGreater(result.diagnostics["changed_turns"], 0)

    def test_bottleneckflow_conserves_each_node_mass(self):
        result = assign_credit("bottleneckflow", self.group)
        self.assertEqual(result.diagnostics["fallback"], "none")
        self.assertLessEqual(result.diagnostics["max_target_conservation_error"], 1e-10)
        self.assertLessEqual(result.diagnostics["max_segment_conservation_error"], 1e-10)
        self.assertGreater(result.diagnostics["credited_edges"], 0)

    def test_unknown_method_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown credit method"):
            assign_credit("mystery", self.group)


if __name__ == "__main__":
    unittest.main()

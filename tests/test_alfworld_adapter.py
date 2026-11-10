import unittest
import json
from pathlib import Path

from agentic_credit_assignment.adapters.alfworld import (
    AlfWorldAdapter,
    AlfWorldAdapterConfig,
    parse_action_output,
)


class ParseActionOutputTest(unittest.TestCase):
    def test_direct_action_is_executable_but_not_reasoning_protocol(self) -> None:
        parsed = parse_action_output("  Go   to cabinet 1  ", ["look", "go to cabinet 1"])
        self.assertFalse(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)
        self.assertEqual(parsed.action, "go to cabinet 1")
        self.assertEqual(parsed.extraction_method, "direct")

    def test_react_think_and_action_tags_follow_protocol(self) -> None:
        parsed = parse_action_output(
            "<think>I should inspect.</think>\n<action>go to cabinet 1</action>",
            ["look", "go to cabinet 1"],
        )
        self.assertTrue(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)
        self.assertEqual(parsed.action, "go to cabinet 1")
        self.assertEqual(parsed.extraction_method, "action_tag")

    def test_action_tag_without_explicit_think_is_protocol_invalid(self) -> None:
        parsed = parse_action_output("<action>look</action>", ["look"])
        self.assertFalse(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)

    def test_truncated_closing_tag_recovers_only_an_exact_action(self) -> None:
        parsed = parse_action_output(
            "<think>Inspect it.</think><action>go to cabinet 1</actio",
            ["look", "go to cabinet 1"],
        )
        self.assertFalse(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)
        self.assertEqual(parsed.action, "go to cabinet 1")
        self.assertEqual(parsed.extraction_method, "truncated_action_tag")

    def test_truncated_non_admissible_action_stays_invalid(self) -> None:
        parsed = parse_action_output(
            "<think>Inspect it.</think><action>go to cabinet 9</actio",
            ["look", "go to cabinet 1"],
        )
        self.assertFalse(parsed.action_valid)

    def test_action_label_is_executable_but_not_protocol_valid(self) -> None:
        parsed = parse_action_output("I should inspect.\nAction: look", ["look"])
        self.assertFalse(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)
        self.assertEqual(parsed.extraction_method, "action_line")

    def test_last_line_exact_action_is_executable(self) -> None:
        parsed = parse_action_output("I should inspect first.\nlook", ["look"])
        self.assertFalse(parsed.protocol_valid)
        self.assertTrue(parsed.action_valid)
        self.assertEqual(parsed.extraction_method, "last_line")

    def test_non_admissible_action_is_invalid(self) -> None:
        parsed = parse_action_output(
            "<think>Try it.</think><action>open fridge 9</action>",
            ["look", "open fridge 1"],
        )
        self.assertFalse(parsed.protocol_valid)
        self.assertFalse(parsed.action_valid)

    def test_missing_action_tags_is_invalid(self) -> None:
        parsed = parse_action_output("I would look around.", ["look"])
        self.assertFalse(parsed.protocol_valid)
        self.assertFalse(parsed.action_valid)
        self.assertEqual(parsed.action, "__invalid_action__")


class GraphStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AlfWorldAdapter(AlfWorldAdapterConfig(data_root=Path("/unused")))
        self.adapter._locations = ["middle of a room"]
        self.adapter._holdings = ["nothing"]
        self.adapter._object_states = [{}]
        self.adapter._object_locations = [{}]

    def test_tracker_records_location_inventory_object_state_and_location(self) -> None:
        self.adapter._update_graph_tracker(0, "go to fridge 1", "You arrive at fridge 1.")
        self.adapter._update_graph_tracker(0, "take apple 1 from fridge 1", "You take apple 1.")
        self.adapter._update_graph_tracker(0, "clean apple 1 with sinkbasin 1", "You clean apple 1.")
        state = json.loads(self.adapter._graph_state("clean apple", ["look"], {}, 0))
        self.assertEqual(state["location"], "fridge 1")
        self.assertEqual(state["holding"], "apple 1")
        self.assertTrue(state["object_states"]["apple 1"]["cleaned"])
        self.assertEqual(state["object_locations"]["apple 1"], "fridge 1")

    def test_failed_action_does_not_mutate_tracker(self) -> None:
        self.adapter._update_graph_tracker(0, "go to fridge 1", "Nothing happens.")
        self.assertEqual(self.adapter._locations[0], "middle of a room")


class GraphGPOPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AlfWorldAdapter(
            AlfWorldAdapterConfig(data_root=Path("/unused"), history_length=2)
        )
        self.adapter._tasks = ["clean some cloth and put it in countertop."]
        self.adapter._histories = [[]]
        self.adapter._locations = ["middle of a room"]
        self.adapter._holdings = ["nothing"]
        self.adapter._object_states = [{}]
        self.adapter._object_locations = [{}]

    def test_initial_prompt_has_explicit_step_location_and_inventory(self) -> None:
        prompt = self.adapter._format_prompt(0, "You are in a room.", ["look", "help"])
        self.assertIn("already taken 0 step(s)", prompt)
        self.assertIn("now at step 1", prompt)
        self.assertIn("Location: middle of a room", prompt)
        self.assertIn("Items in hand (status): nothing", prompt)
        self.assertIn("[look]", prompt)
        self.assertNotIn("help", prompt)

    def test_history_is_numbered_and_pairs_pre_action_state_with_action(self) -> None:
        self.adapter._histories[0] = [
            (
                "You are in a room. Location: middle of a room. "
                "Items in hand (status): nothing.",
                "go to countertop 1",
            )
        ]
        self.adapter._locations[0] = "countertop 1"
        self.adapter._holdings[0] = "cloth 2"
        prompt = self.adapter._format_prompt(
            0, "You pick up the cloth 2.", ["go to sinkbasin 1"]
        )
        self.assertIn("Observation 1:", prompt)
        self.assertIn("Action 1: 'go to countertop 1'", prompt)
        self.assertIn("now at step 2", prompt)
        self.assertIn("Items in hand (status): cloth 2(unprocessed)", prompt)

    def test_native_thinking_prompt_does_not_request_a_second_think_block(self) -> None:
        adapter = AlfWorldAdapter(
            AlfWorldAdapterConfig(
                data_root=Path("/unused"), history_length=2, native_thinking=True
            )
        )
        adapter._tasks = self.adapter._tasks
        adapter._histories = [[]]
        adapter._locations = self.adapter._locations
        adapter._holdings = self.adapter._holdings
        adapter._object_states = self.adapter._object_states
        adapter._object_locations = self.adapter._object_locations
        prompt = adapter._format_prompt(0, "You are in a room.", ["look"])
        self.assertNotIn("reasoning process MUST be enclosed", prompt)
        self.assertNotIn("Once you've finished your reasoning", prompt)
        self.assertTrue(
            prompt.endswith(
                "You should choose an admissible action for current step and present it "
                "within <action> </action> tags."
            )
        )


if __name__ == "__main__":
    unittest.main()

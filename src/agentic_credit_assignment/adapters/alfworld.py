"""Project-owned adapter for the official ALFWorld text environment.

This module depends only on ALFWorld's public environment API. It deliberately
does not import the vendored GraphGPO/verl-agent recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence


_ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_OPEN_ACTION_PATTERN = re.compile(
    r"<action>\s*([^<\r\n]+?)(?:\s*<[^>]*?)?\s*$", re.IGNORECASE | re.DOTALL
)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ACTION_LINE_PATTERN = re.compile(r"^\s*action\s*:\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class AlfWorldAdapterConfig:
    """Configuration required to construct the official ALFWorld benchmark."""

    data_root: Path
    split: str = "train"
    num_envs: int = 2
    seed: int = 0
    max_steps: int = 20
    history_length: int = 2
    num_train_games: int = -1
    num_eval_games: int = -1
    fixed_gamefile: Path | None = None
    native_thinking: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"train", "valid_seen", "valid_unseen"}:
            raise ValueError(f"Unsupported ALFWorld split: {self.split}")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")


@dataclass(frozen=True)
class ParsedAction:
    """A model response parsed against one environment action set."""

    raw_output: str
    action: str
    protocol_valid: bool
    action_valid: bool
    extraction_method: str


@dataclass(frozen=True)
class AlfWorldObservation:
    """One environment's model-facing observation."""

    prompt: str
    feedback: str
    graph_state: str
    admissible_actions: tuple[str, ...]
    gamefile: str | None


def parse_action_output(output: str, admissible_actions: Sequence[str]) -> ParsedAction:
    """Extract an exact admissible action using deterministic syntax fallbacks.

    The baseline-compatible ReAct protocol requires both a complete
    ``<think>...</think>`` block and a complete ``<action>...</action>`` block.
    Exact legal actions recovered by a fallback may still be executed, but are
    counted as protocol-invalid.  No semantic repair, nearest-action fallback,
    constrained decoding, or retry is performed.
    """

    def normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())

    normalized_pool = {normalize(item) for item in admissible_actions}
    stripped = output.strip()
    direct = normalize(stripped)
    tagged = list(_ACTION_PATTERN.finditer(output))
    has_complete_thought = _THINK_PATTERN.search(output) is not None
    candidate = direct
    method = "direct"
    protocol_valid = False
    if tagged:
        candidate = normalize(tagged[-1].group(1))
        method = "action_tag"
        protocol_valid = has_complete_thought and candidate in normalized_pool
    elif bool(direct) and "\n" not in stripped and direct in normalized_pool:
        # Keep accepting a bare exact action for backwards compatibility, but
        # do not count it as compliant with the reasoning-enabled protocol.
        candidate = direct
        method = "direct"
    else:
        truncated = _OPEN_ACTION_PATTERN.search(output)
        action_lines = [
            match.group(1)
            for line in output.splitlines()
            if (match := _ACTION_LINE_PATTERN.match(line)) is not None
        ]
        nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
        if truncated and normalize(truncated.group(1)) in normalized_pool:
            candidate = normalize(truncated.group(1))
            method = "truncated_action_tag"
            # A serving engine may omit a matched stop string.  The action is
            # executable, but the strict baseline protocol is not complete.
            protocol_valid = False
        elif action_lines:
            candidate = normalize(action_lines[-1])
            method = "action_line"
        elif nonempty_lines:
            candidate = normalize(nonempty_lines[-1])
            method = "last_line"
        else:
            candidate = ""
            method = "none"

    action_valid = bool(candidate) and candidate in normalized_pool
    return ParsedAction(
        raw_output=output,
        action=candidate if action_valid else "__invalid_action__",
        protocol_valid=protocol_valid,
        action_valid=action_valid,
        extraction_method=method,
    )


class AlfWorldAdapter:
    """Vectorized ALFWorld interaction loop with project-owned prompting/parsing."""

    def __init__(self, config: AlfWorldAdapterConfig):
        self.config = config
        self._env = None
        self._tasks: list[str] = []
        # Each entry is (observation_before_action, action), matching the
        # GraphGPO prompt rather than pairing an action with its next state.
        self._histories: list[list[tuple[str, str]]] = []
        self._current_feedbacks: list[str] = []
        self._gamefiles: list[str | None] = []
        self._action_pools: list[tuple[str, ...]] = []
        self._locations: list[str] = []
        self._holdings: list[str] = []
        self._object_states: list[dict[str, dict[str, bool]]] = []
        self._object_locations: list[dict[str, str]] = []

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    @staticmethod
    def _extract_task(feedback: str) -> str:
        marker = "Your task is to:"
        index = feedback.find(marker)
        if index < 0:
            raise ValueError(f"Task description missing from ALFWorld feedback: {feedback[:200]!r}")
        return feedback[index + len(marker) :].strip()

    def _official_config(self) -> dict[str, Any]:
        root = self.config.data_root.expanduser().resolve()
        json_root = root / "json_2.1.1"
        required = [
            json_root / "train",
            json_root / "valid_seen",
            json_root / "valid_unseen",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing ALFWorld data directories: {missing}")

        train_path = json_root / "train"
        eval_seen_path = json_root / "valid_seen"
        eval_unseen_path = json_root / "valid_unseen"
        if self.config.fixed_gamefile is not None:
            # The ALFWorld constructor recursively scans data_path. Point it at
            # the selected trial to avoid re-reading the complete CephFS split.
            selected_trial = self.config.fixed_gamefile.expanduser().resolve().parent
            train_path = eval_seen_path = eval_unseen_path = selected_trial

        return {
            "dataset": {
                "data_path": str(train_path),
                "eval_id_data_path": str(eval_seen_path),
                "eval_ood_data_path": str(eval_unseen_path),
                "num_train_games": self.config.num_train_games,
                "num_eval_games": self.config.num_eval_games,
            },
            "env": {
                "goal_desc_human_anns_prob": 0.0,
                "task_types": [1, 2, 3, 4, 5, 6],
                "domain_randomization": False,
                "expert_type": "handcoded",
            },
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": self.config.max_steps}},
        }

    def _holding_status(self, env_index: int) -> str:
        holding = self._holdings[env_index]
        if holding == "nothing":
            return "nothing"
        enabled_states = [
            name
            for name, enabled in self._object_states[env_index].get(holding, {}).items()
            if enabled
        ]
        status = " ".join(enabled_states) if enabled_states else "unprocessed"
        return f"{holding}({status})"

    def _prompt_observation(self, env_index: int, feedback: str) -> str:
        """Append the explicit state summary used by the GraphGPO wrapper.

        Stock ALFWorld does not return these two strings.  They are maintained
        deterministically from successful benchmark actions by our adapter.
        """

        compact_feedback = " ".join(feedback.strip().split())
        return (
            f"{compact_feedback} Location: {self._locations[env_index]}. "
            f"Items in hand (status): {self._holding_status(env_index)}."
        )

    def _format_prompt(self, env_index: int, feedback: str, actions: Sequence[str]) -> str:
        action_text = ", ".join(action for action in actions if action != "help")
        history = self._histories[env_index][-self.config.history_length :]
        if history:
            first_number = len(self._histories[env_index]) - len(history) + 1
            history_text = "\n".join(
                f"[Observation {number}: {observation!r}, Action {number}: {action!r}]"
                for number, (observation, action) in enumerate(history, start=first_number)
            )
        else:
            history_text = "None."

        completed_steps = len(self._histories[env_index])
        current_observation = self._prompt_observation(env_index, feedback)
        response_instruction = (
            "You should choose an admissible action for current step and present it "
            "within <action> </action> tags."
            if self.config.native_thinking
            else (
                "You should first reason step-by-step about the current situation. This "
                "reasoning process MUST be enclosed within <think> </think> tags.\n"
                "Once you've finished your reasoning, you should choose an admissible "
                "action for current step and present it within <action> </action> tags."
            )
        )

        return (
            "You are an expert agent operating in the ALFRED Embodied Environment.\n"
            f"Your task is to: {self._tasks[env_index]}\n"
            f"Prior to this step, you have already taken {completed_steps} step(s). "
            f"Below are the most recent {len(history)} observations and the corresponding "
            f"actions you took:\n{history_text}\n"
            f"You are now at step {completed_steps + 1} and your current observation is: "
            f"{current_observation!r}\n"
            f"Your admissible actions of the current situation are: [{action_text}].\n\n"
            "Now it's your turn to take an action.\n"
            f"{response_instruction}"
        )

    def _graph_state(
        self, feedback: str, actions: Sequence[str], info: dict[str, Any], index: int
    ) -> str:
        """Build a stable, GraphGPO-style environment anchor for graph identity.

        Model history and prompt formatting are deliberately excluded.  When
        ALFWorld exposes structured state fields we retain them; the normalized
        local observation and sorted legal actions remain a deterministic
        fallback for stock environments.
        """

        def normalize_text(value: object) -> str:
            return " ".join(str(value).lower().split())

        def at(key: str) -> object | None:
            value = info.get(key)
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) > index:
                return value[index]
            return value

        payload: dict[str, object] = {
            "observation": normalize_text(feedback),
            "admissible_actions": sorted(normalize_text(action) for action in actions),
            "location": self._locations[index],
            "holding": self._holdings[index],
            "object_states": self._object_states[index],
            "object_locations": self._object_locations[index],
        }
        for key in ("obs_location", "obs_holding", "items_info", "item_location"):
            value = at(key)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def _update_graph_tracker(self, index: int, action: str, feedback: str) -> None:
        """Track the small amount of latent ALFWorld state used by graph identity.

        The tracker follows only confirmed, admissible actions and never changes
        state when the environment reports that nothing happened.  It mirrors
        the useful part of GraphGPO's anchor construction without importing its
        environment implementation.
        """

        if action == "__invalid_action__" or "nothing happens" in feedback.lower():
            return
        normalized = " ".join(action.lower().split())
        location_match = re.match(r"^go to\s+(.+)$", normalized)
        if location_match:
            self._locations[index] = location_match.group(1)

        take_match = re.match(r"^take\s+(.+?)(?:\s+from\s+.+)?$", normalized)
        if take_match:
            item = take_match.group(1)
            self._holdings[index] = item
            self._object_locations[index][item] = self._locations[index]

        put_match = re.match(r"^(?:put|place)\s+(.+?)\s+(?:in|on)\s+(.+)$", normalized)
        if put_match:
            item, receptacle = put_match.groups()
            self._object_locations[index][item] = receptacle
            if self._holdings[index] == item:
                self._holdings[index] = "nothing"

        drop_match = re.match(r"^drop\s+(.+)$", normalized)
        if drop_match:
            item = drop_match.group(1)
            self._object_locations[index][item] = self._locations[index]
            if self._holdings[index] == item:
                self._holdings[index] = "nothing"

        move_match = re.match(r"^move\s+(.+?)\s+to\s+(.+)$", normalized)
        if move_match:
            item, receptacle = move_match.groups()
            self._object_locations[index][item] = receptacle
            if self._holdings[index] == item:
                self._holdings[index] = "nothing"

        state_match = re.match(
            r"^(heat|cool|clean|slice|toggle)\s+(.+?)(?:\s+with\s+.+)?$", normalized
        )
        if state_match:
            verb, item = state_match.groups()
            states = self._object_states[index].setdefault(item, {})
            key = {"heat": "heated", "cool": "cooled", "clean": "cleaned",
                   "slice": "sliced", "toggle": "toggled"}[verb]
            states[key] = not states.get(key, False) if verb == "toggle" else True

    def reset(self) -> list[AlfWorldObservation]:
        """Construct the official vector environment and return initial observations."""

        from alfworld.agents.environment import get_environment

        train_eval = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }[self.config.split]
        base_env = get_environment("AlfredTWEnv")(self._official_config(), train_eval=train_eval)
        if self.config.fixed_gamefile is not None:
            gamefile = self.config.fixed_gamefile.expanduser().resolve()
            if not gamefile.is_file():
                raise FileNotFoundError(f"Fixed ALFWorld gamefile does not exist: {gamefile}")
            # TextWorld accepts repeated gamefiles at registration time. This is
            # how a GRPO group obtains independent resets of the same task.
            base_env.game_files = [str(gamefile)] * self.config.num_envs
            base_env.num_games = self.config.num_envs
        self._env = base_env.init_env(batch_size=self.config.num_envs)
        if hasattr(self._env, "seed"):
            self._env.seed(self.config.seed)

        feedbacks_raw, info = self._env.reset()
        feedbacks = list(feedbacks_raw)
        action_pools = list(info["admissible_commands"])
        gamefiles = list(info.get("extra.gamefile", [None] * self.config.num_envs))
        self._tasks = [self._extract_task(feedback) for feedback in feedbacks]
        self._histories = [[] for _ in range(self.config.num_envs)]
        self._current_feedbacks = feedbacks.copy()
        self._gamefiles = [str(path) if path is not None else None for path in gamefiles]
        self._action_pools = [tuple(pool) for pool in action_pools]
        self._locations = ["middle of a room" for _ in range(self.config.num_envs)]
        self._holdings = ["nothing" for _ in range(self.config.num_envs)]
        self._object_states = [{} for _ in range(self.config.num_envs)]
        self._object_locations = [{} for _ in range(self.config.num_envs)]
        return [
            AlfWorldObservation(
                prompt=self._format_prompt(i, feedbacks[i], action_pools[i]),
                feedback=feedbacks[i],
                graph_state=self._graph_state(feedbacks[i], action_pools[i], info, i),
                admissible_actions=tuple(action_pools[i]),
                gamefile=self._gamefiles[i],
            )
            for i in range(self.config.num_envs)
        ]

    def step(
        self,
        model_outputs: Sequence[str],
    ) -> tuple[list[AlfWorldObservation], list[ParsedAction], list[float], list[bool], list[bool]]:
        """Parse model outputs, step all environments, and build next prompts.

        Returns observations, parsed actions, rewards, done flags, and won flags.
        Invalid model outputs are sent as a deliberately invalid command so the
        smoke test does not hide format failures with a fallback action.
        """

        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        if len(model_outputs) != self.config.num_envs:
            raise ValueError(
                f"Expected {self.config.num_envs} model outputs, got {len(model_outputs)}"
            )

        parsed = [
            parse_action_output(output, self._action_pools[i])
            for i, output in enumerate(model_outputs)
        ]
        commands = [item.action if item.action_valid else "__invalid_action__" for item in parsed]
        pre_action_observations = [
            self._prompt_observation(i, self._current_feedbacks[i])
            for i in range(self.config.num_envs)
        ]
        feedbacks_raw, rewards, dones, info = self._env.step(commands)
        feedbacks = list(feedbacks_raw)
        action_pools = list(info["admissible_commands"])
        self._action_pools = [tuple(pool) for pool in action_pools]
        won = [bool(value) for value in info["won"]]

        for i, (command, feedback) in enumerate(zip(commands, feedbacks)):
            self._update_graph_tracker(i, command, feedback)
            self._histories[i].append((pre_action_observations[i], command))
        self._current_feedbacks = feedbacks.copy()

        observations = [
            AlfWorldObservation(
                prompt=self._format_prompt(i, feedbacks[i], action_pools[i]),
                feedback=feedbacks[i],
                graph_state=self._graph_state(feedbacks[i], action_pools[i], info, i),
                admissible_actions=tuple(action_pools[i]),
                gamefile=self._gamefiles[i],
            )
            for i in range(self.config.num_envs)
        ]
        return observations, parsed, [float(x) for x in rewards], [bool(x) for x in dones], won

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> "AlfWorldAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


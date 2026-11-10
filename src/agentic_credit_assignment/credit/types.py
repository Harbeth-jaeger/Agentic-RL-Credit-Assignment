"""Small, framework-independent rollout records used by credit assigners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    state: str
    action: str
    next_state: str

    @property
    def edge(self) -> tuple[str, str, str]:
        return (self.state, self.action, self.next_state)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    transitions: tuple[Transition, ...]
    reward: float
    success: bool
    truncated: bool = False


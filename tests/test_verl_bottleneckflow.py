from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from agentic_credit_assignment.verl.bottleneckflow import apply_bottleneckflow, append_eval_rollouts


def test_adapter_preserves_shape_and_audits_steps(tmp_path):
    # Four same-task trajectories: two successful paths share the bottleneck k;
    # two failures avoid it. Each row is one environment step as in GraphGPO.
    states = np.array(["s", "k", "s", "k", "s", "x", "s", "y"], dtype=object)
    next_states = np.array(["k", "success", "k", "success", "x", "failure", "y", "failure"], dtype=object)
    actions = np.array(["to-k", "finish", "to-k", "finish", "bad-x", "stop", "bad-y", "stop"], dtype=object)
    trajectory_ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    rewards = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "uid": np.array(["task"] * 8, dtype=object),
            "traj_uid": trajectory_ids,
            "anchor_obs": states,
            "next_obs": next_states,
            "executed_actions": actions,
            "text_actions": np.array([f"<action>{item}</action>" for item in actions], dtype=object),
            "is_action_valid": np.ones(8, dtype=bool),
            "episode_rewards": rewards,
            "rewards": np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float),
            "formatted_prompt": np.array([f"prompt-{i}" for i in range(8)], dtype=object),
            "raw_observations": np.array([f"observation-{i}" for i in range(8)], dtype=object),
            "environment_dones": np.array([False, True] * 4, dtype=bool),
            "environment_won": np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float),
            "goal_condition_success_rate": np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=float),
            "task_description": np.array(["do the task"] * 8, dtype=object),
        },
    )
    mask = torch.ones((8, 3))
    base = torch.tensor([[1.0] * 3] * 4 + [[-1.0] * 3] * 4)
    rollout_path = tmp_path / "rollouts.jsonl"
    graph_path = tmp_path / "graphs.jsonl"

    result, metrics = apply_bottleneckflow(
        batch,
        episode_advantages=base,
        response_mask=mask,
        graphs=None,
        config={"min_success_support": 2, "min_success_coverage": 1.0},
        global_step=1,
        rollout_path=str(rollout_path),
        graph_path=str(graph_path),
    )

    assert result.batch["advantages"].shape == base.shape
    assert torch.all(result.batch["advantages"] * (1 - mask) == 0)
    assert rollout_path.read_text().count("\n") == 1
    assert graph_path.read_text().count("\n") == 1
    assert "bottleneckflow/omega_mean" in metrics
    # Weight projection conserves each trajectory's average episode advantage.
    for start in range(0, 8, 2):
        assert torch.isclose(result.batch["advantages"][start : start + 2].mean(), base[start : start + 2].mean())

    append_eval_rollouts(batch, global_step=1, rollout_path=str(rollout_path))
    records = rollout_path.read_text().splitlines()
    assert len(records) == 2
    assert '"phase": "eval"' in records[1]
    assert '"prompt": "prompt-0"' in records[1]
    assert '"raw_observation": "observation-1"' in records[1]

from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from agentic_credit_assignment.slime.credit import token_advantages
from agentic_credit_assignment.slime.logging import log_eval
from agentic_credit_assignment.slime.rollout import action_token_mask


def test_action_mask_excludes_think_and_includes_action():
    tokenizer = AutoTokenizer.from_pretrained("/cephfs/wenhaozhang/models/Qwen3.5-2B")
    text = "<think>I should inspect the room.</think><action>look</action>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    mask = action_token_mask(tokenizer, ids, text)
    kept = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])
    assert "inspect the room" not in kept
    assert "action" in kept
    assert "look" in kept
    assert len(mask) == len(ids)


def test_token_advantages_reads_transport_safe_sidecar():
    sidecar = [torch.tensor([0.0, 1.5, 1.5]), torch.tensor([-2.0, 0.0])]
    batch = {"teacher_log_probs": sidecar}
    token_advantages(SimpleNamespace(), batch)
    assert [x.tolist() for x in batch["advantages"]] == [[0.0, 1.5, 1.5], [-2.0, 0.0]]
    assert [x.tolist() for x in batch["returns"]] == [[0.0, 1.5, 1.5], [-2.0, 0.0]]


def test_eval_logger_writes_summary_and_step_trace(tmp_path):
    sample = SimpleNamespace(
        index=7,
        reward=1.0,
        metadata={
            "task_id": "task-7",
            "gamefile": "/official/game.tw-pddl",
            "split": "valid_seen",
            "success": True,
            "truncated": False,
            "steps": [{"state": "s0", "action": "look", "credit": 1.0}],
        },
    )
    data = {
        "valid_seen": {
            "samples": [sample],
            "rewards": [1.0],
            "truncated": [False],
        }
    }
    assert log_eval(3, SimpleNamespace(aca_arm_dir=str(tmp_path)), data, {}) is False
    summary = (tmp_path / "eval_valid_seen.json").read_text(encoding="utf-8")
    trace = (tmp_path / "eval_traces/rollout_3/valid_seen/sample_7.json").read_text(
        encoding="utf-8"
    )
    assert '"success_rate": 1.0' in summary
    assert '"action": "look"' in trace

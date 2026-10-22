"""SAHA-CF reward under the detect tool (plan/saha-detect-tool/PLAN.md §D).

    cd /workspace/AdaTooler-V/verltool && python examples/train/hoi/test_saha_cf_detect.py
(plain asserts; also collected by pytest when it is installed)
"""
import json
import os

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl_tool.trainer.ppo.metric_util import compute_saha_cf_metrics
from verl_tool.workers.reward_manager.saha_cf import SAHACounterfactualRewardManager

DETECT = '<tool_call>\n{"name": "detect", "arguments": {}}\n</tool_call>'
GT = {"boxes_1000": [[100, 100, 400, 900], [450, 500, 600, 700]], "num_pairs": 1}
GOOD = '<answer>[{"bbox_2d": [100, 100, 400, 900], "label": "person"}, {"bbox_2d": [450, 500, 600, 700], "label": "cup"}]</answer>'
BAD = '<answer>[{"bbox_2d": [700, 0, 800, 100], "label": "person"}, {"bbox_2d": [0, 0, 50, 50], "label": "cup"}]</answer>'
ANCHOR_PERFECT = [[{"bbox_2d": [100, 100, 400, 900], "label": "person"}, {"bbox_2d": [450, 500, 600, 700], "label": "cup"}]]


def approx(a, b):
    a, b = (list(a), list(b)) if isinstance(a, (list, tuple)) else ([a], [b])
    return len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))


class FakeTokenizer:
    def __init__(self, responses):
        self.responses = responses

    def decode(self, ids, skip_special_tokens=True):
        return self.responses[int(ids[0])]


def rollout(answer, tool=False):
    return "<think>t</think>" + (DETECT + "\nobs\n<think>t</think>" if tool else "") + answer


def run(rows, grounding_sref, agent_loop_scores=False, **kwargs):
    """rows: list of (uid, response). Returns (reward per row, reward_extra_info).
    agent_loop_scores: the batch carries rm_scores, as after the agent loop scored each rollout."""
    os.environ["SAHA_CF_LOG_SUMMARY"] = "0"
    responses = [r for _, r in rows]
    n = len(rows)
    tensors = {"prompts": torch.zeros((n, 4), dtype=torch.long),
               "responses": torch.arange(n).unsqueeze(1),
               "attention_mask": torch.ones((n, 5), dtype=torch.long)}
    if agent_loop_scores:
        tensors["rm_scores"] = torch.zeros((n, 1))
    data = DataProto(
        batch=TensorDict(tensors, batch_size=[n]),
        non_tensor_batch={
            "reward_model": np.array([{"ground_truth": json.dumps(GT), "style": "rule"}] * n, dtype=object),
            "extra_info": np.array([{"task_type": "grounding", "proposal_anchor": ANCHOR_PERFECT}] * n, dtype=object),
            "data_source": np.array(["hico"] * n, dtype=object),
            "uid": np.array([u for u, _ in rows], dtype=object),
        })
    mgr = SAHACounterfactualRewardManager(FakeTokenizer(responses), num_examine=0, alpha=0.6,
                                          grounding_sref=grounding_sref, **kwargs)
    out = mgr(data, return_dict=True)
    return out["reward_tensor"][:, 0].tolist(), out["reward_extra_info"]


def test_detect_call_counts_as_tool():
    _, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(GOOD))], "group_no_tool")
    assert info["i_tool"] == [1.0, 0.0]
    assert info["n_detect"] == [1.0, 0.0]


def test_group_no_tool_rewards_detect_that_beats_its_no_tool_siblings():
    rewards, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(BAD)), ("a", rollout(BAD))], "group_no_tool")
    assert approx(info["s_ref"][0], 0.0)
    assert info["r_tool"][0] > 0
    assert approx(rewards[0], 1.0 + 0.6 * info["r_tool"][0])


def test_full_batch_scoring_ignores_per_rollout_agent_loop_scores():
    rewards, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(BAD))], "group_no_tool",
                        agent_loop_scores=True)
    assert info["has_sref"][0] == 1.0
    assert info["r_tool"][0] > 0
    assert approx(rewards[0], 1.0 + 0.6 * info["r_tool"][0])


def test_group_no_tool_without_no_tool_sibling_gives_no_tool_reward():
    rewards, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(GOOD, tool=True))], "group_no_tool")
    assert info["has_sref"] == [0.0, 0.0]
    assert info["r_tool"] == [0.0, 0.0]
    assert approx(rewards, [1.0, 1.0])


def test_anchor_mode_still_scores_against_the_proposal_anchor():
    # The anchor is perfect, so a correct detect answer gains nothing over it.
    _, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(BAD))], "anchor")
    assert approx(info["s_ref"][0], 1.0)
    assert approx(info["r_tool"][0], 0.0)


def test_strata_use_anchor_score_not_sibling_reference():
    _, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(BAD))], "group_no_tool")
    batch = {k: np.array(v, dtype=object) for k, v in info.items()}
    batch["uid"] = np.array(["a", "a"], dtype=object)
    m = compute_saha_cf_metrics(batch)
    assert approx(m["saha_cf/tool_rate_easy"], 0.5)  # perfect anchor -> both rows are "easy"
    assert approx(m["saha_cf/detect_rate"], 0.5)
    assert approx(m["saha_cf/sref_missing_frac"], 0.0)


def test_grounding_sibling_reference_ignores_referring_setting():
    _, info = run([("a", rollout(GOOD, tool=True)), ("a", rollout(BAD))], "group_no_tool", referring_sref="off")
    assert info["has_sref"][0] == 1.0
    assert info["r_tool"][0] > 0


def test_unknown_grounding_sref_is_rejected():
    try:
        SAHACounterfactualRewardManager(FakeTokenizer([]), num_examine=0, grounding_sref="typo")
    except ValueError:
        return
    raise AssertionError("grounding_sref='typo' was accepted")


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} passed")

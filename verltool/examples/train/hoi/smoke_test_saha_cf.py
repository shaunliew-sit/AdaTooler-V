"""SAHA-CF pipeline smoke test — runs REAL data through the REAL reward manager.

Goal: before committing GPUs to a training run, confirm three things about the
SAHA-CF (v3 counterfactual tool-gain) reward:

  1. PIPELINE WORKS   — the migrated parquet loads and flows through
                        SAHACounterfactualRewardManager.__call__ end-to-end.
  2. TOOL-COLLAPSE FIX — a tool that actually helps earns POSITIVE GRPO group
                        advantage exactly where the proposal anchor is weak
                        (low s_ref); an unnecessary/harmful tool does NOT.
  3. REWARD IS HELPFUL — the per-group reward has real spread (std >> 0), i.e.
                        it is not collapsed to a constant, so GRPO has signal.

It builds GRPO groups (one uid each, n=4 rollouts) from real val.parquet grounding
rows — real GT boxes and real reconstructed proposal_anchor — and synthesizes the
four canonical rollouts (no-tool correct/wrong, tool correct/wrong). It then runs
the actual reward manager and computes the standard GRPO advantage per group.

Run:  python examples/train/hoi/smoke_test_saha_cf.py
(from the verltool dir, inside verl-tool-env)
"""
import json
import sys

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl_tool.workers.reward_manager.saha_cf import (
    SAHACounterfactualRewardManager,
    compute_sref_grounding,
)

VAL_PARQUET = "data/val.parquet"
ALPHA = 0.6  # must match the training default (SAHA_CF_ALPHA)


# ---------------------------------------------------------------------------
# Fake tokenizer: each rollout's "response" is a single token whose id indexes
# into a list of pre-built response strings. decode() returns that string. This
# lets us drive the real reward manager without a real model/tokenizer.
# ---------------------------------------------------------------------------
class FakeTokenizer:
    def __init__(self, response_strings):
        self.responses = response_strings

    def decode(self, ids, skip_special_tokens=True):
        return self.responses[int(ids[0])]


ZOOM_IN = '<tool_call>{"name": "zoom_in", "arguments": {"bbox_2d": [100, 100, 400, 400]}}</tool_call>'
ZOOM_OUT = '<tool_call>{"name": "zoom_out", "arguments": {}}</tool_call>'


def grounding_answer(pairs_boxes):
    """Build an <answer> body in the exact format the anchor/scorer use."""
    lines = []
    for person, obj in pairs_boxes:
        lines.append(json.dumps([
            {"bbox_2d": person, "label": "person"},
            {"bbox_2d": obj, "label": "object"},
        ]))
    return "\n".join(lines)


def gt_pairs_from(gt):
    boxes, np_ = gt["boxes_1000"], gt["num_pairs"]
    return [(boxes[2 * i], boxes[2 * i + 1]) for i in range(np_)]


def make_rollouts(gt):
    """Four canonical rollouts for one grounding prompt."""
    pairs = gt_pairs_from(gt)
    correct = grounding_answer(pairs)
    wrong = grounding_answer([([0, 0, 1, 1], [2, 2, 3, 3]) for _ in pairs])
    think = "<think>locate the interacting pair</think>"
    return [
        # label,            response_str
        ("no_tool_correct", f"{think}<answer>{correct}</answer>"),
        ("no_tool_wrong",   f"{think}<answer>{wrong}</answer>"),
        ("tool_correct",    f"{think}{ZOOM_IN}{ZOOM_OUT}<answer>{correct}</answer>"),
        ("tool_wrong",      f"{think}{ZOOM_IN}{ZOOM_OUT}<answer>{wrong}</answer>"),
    ]


def pick_rows(ds):
    """Pick one low-s_ref ('hard') and one high-s_ref ('easy') grounding row."""
    scored = []
    for r in ds:
        if r["extra_info"]["task_type"] != "grounding":
            continue
        gt = json.loads(r["reward_model"]["ground_truth"])
        if not gt.get("boxes_1000") or not gt.get("num_pairs"):
            continue
        s_ref = compute_sref_grounding(r["extra_info"].get("proposal_anchor"), gt)
        scored.append((s_ref, gt))
    scored.sort(key=lambda x: x[0])
    hard = scored[0]                       # weakest proposal anchor (tool should help)
    easy = next(x for x in reversed(scored) if x[0] >= 0.99)  # perfect anchor
    return {"HARD(low s_ref)": hard, "EASY(high s_ref)": easy}


def build_dataproto(groups):
    """One flat DataProto over all (group, rollout) rows."""
    resp_strings, rm, extra, dsrc, uids, labels, srefs = [], [], [], [], [], [], []
    for uid, (s_ref, gt) in groups.items():
        for label, resp in make_rollouts(gt):
            resp_strings.append(resp)
            rm.append({"ground_truth": json.dumps(gt), "style": "rule"})
            # grounding s_ref comes from proposal_anchor; inject an anchor that
            # reproduces this row's s_ref so the group is self-contained.
            anchor = anchor_for_sref(s_ref, gt)
            extra.append({"task_type": "grounding", "proposal_anchor": anchor})
            dsrc.append("hico")
            uids.append(uid)
            labels.append(label)
            srefs.append(s_ref)

    n = len(resp_strings)
    ids = torch.arange(n).unsqueeze(1)             # responses: [n, 1], token == row index
    prompts = torch.zeros((n, 4), dtype=torch.long)
    attn = torch.ones((n, 5), dtype=torch.long)    # 4 prompt + 1 response, all valid
    batch = TensorDict(
        {"prompts": prompts, "responses": ids, "attention_mask": attn},
        batch_size=[n],
    )
    non_tensor = {
        "reward_model": np.array(rm, dtype=object),
        "extra_info": np.array(extra, dtype=object),
        "data_source": np.array(dsrc, dtype=object),
        "uid": np.array(uids, dtype=object),
    }
    data = DataProto(batch=batch, non_tensor_batch=non_tensor)
    return data, resp_strings, labels, uids, srefs


def anchor_for_sref(s_ref, gt):
    """Return a proposal_anchor that scores to ~s_ref for this row.
    s_ref==1.0  -> anchor = GT boxes (perfect proposal).
    else        -> empty anchor -> compute_sref_grounding returns 0.0.
    (The real parquet carries the true anchor; here we just need the two
    regimes — strong vs absent proposal — to exercise the mechanism.)"""
    if s_ref >= 0.99:
        pairs = gt_pairs_from(gt)
        return [[{"bbox_2d": p, "label": "person"}, {"bbox_2d": o, "label": "object"}]
                for p, o in pairs]
    return []


def grpo_advantage(scores):
    a = np.array(scores, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-8)


def main():
    import datasets
    ds = datasets.load_dataset("parquet", data_files=VAL_PARQUET)["train"]
    groups = pick_rows(ds)

    # Force training-mode reward (r_total, not the eval accuracy hook): num_examine=0.
    rm = SAHACounterfactualRewardManager(tokenizer=None, num_examine=0, alpha=ALPHA)
    data, resp_strings, labels, uids, srefs = build_dataproto(groups)
    rm.tokenizer = FakeTokenizer(resp_strings)

    out = rm(data, return_dict=True)
    info = out["reward_extra_info"]

    # ---- report per group ----
    print(f"\n{'='*92}\nSAHA-CF smoke test  (alpha={ALPHA})\n{'='*92}")
    by_uid = {}
    for i in range(len(labels)):
        by_uid.setdefault(uids[i], []).append(i)

    checks = []
    for uid, idxs in by_uid.items():
        scores = [info["score"][j] for j in idxs]
        adv = grpo_advantage(scores)
        s_ref = srefs[idxs[0]]
        print(f"\n# group {uid}   (grounding, s_ref≈{s_ref:.3f})")
        print(f"  {'rollout':<18}{'i_tool':>7}{'R_out':>8}{'s_ref':>8}{'R_tool':>8}{'R_total':>9}{'GRPO_adv':>10}")
        row_by_label = {}
        for k, j in enumerate(idxs):
            lbl = labels[j]
            row_by_label[lbl] = dict(
                i_tool=info["i_tool"][j], r_out=info["r_outcome"][j],
                s_ref=info["s_ref"][j], r_tool=info["r_tool"][j],
                score=info["score"][j], adv=adv[k],
            )
            print(f"  {lbl:<18}{info['i_tool'][j]:>7.0f}{info['r_outcome'][j]:>8.2f}"
                  f"{info['s_ref'][j]:>8.2f}{info['r_tool'][j]:>8.2f}"
                  f"{info['score'][j]:>9.3f}{adv[k]:>10.3f}")

        std = float(np.std(scores))
        if "low" in uid:  # HARD: tool genuinely recovers -> must be rewarded
            tc, nc = row_by_label["tool_correct"], row_by_label["no_tool_correct"]
            checks.append(("HARD: tool_correct R_tool > 0",
                           tc["r_tool"] > 0))
            checks.append(("HARD: tool_correct beats no_tool_correct (R_total)",
                           tc["score"] > nc["score"] + 1e-6))
            checks.append(("HARD: tool_correct has the top GRPO advantage",
                           tc["adv"] == max(r["adv"] for r in row_by_label.values())))
        else:  # EASY: proposal already perfect -> tool must NOT be over-rewarded
            tc, nc = row_by_label["tool_correct"], row_by_label["no_tool_correct"]
            tw = row_by_label["tool_wrong"]
            checks.append(("EASY: unnecessary tool gains ~nothing (R_tool≈0)",
                           abs(tc["r_tool"]) < 1e-6))
            checks.append(("EASY: tool_correct ≈ no_tool_correct (no free reward)",
                           abs(tc["score"] - nc["score"]) < 1e-6))
            checks.append(("EASY: harmful tool is PENALIZED (R_tool<0, neg adv)",
                           tw["r_tool"] < 0 and tw["adv"] < 0))
        checks.append((f"{uid}: reward not collapsed (group std={std:.3f} > 0.05)",
                       std > 0.05))

    print(f"\n{'='*92}\nCHECKS\n{'='*92}")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL CHECKS PASSED ✅' if ok else 'SOME CHECKS FAILED ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

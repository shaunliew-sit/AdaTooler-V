"""SAHA-CF: Proposal-Conditioned Counterfactual Tool-Gain reward manager.

    R_total = R_format * (R_outcome + alpha * R_tool)
    R_tool  = I_tool * clip(s_final - s_ref, clip_lo, clip_hi)

Where s_ref is the "trust-the-proposal / no-tool" reference:
  - grounding: score of the selected proposal anchor box vs the target GT
               (GT-anchored, policy-independent; built in prepare_train.py).
  - referring: mean R_outcome of no-tool rollouts in the same uid group
               (detached constant from the completed batch; min-count guarded).

The tool earns reward only when zooming beats trusting the proposal. This is the
SAHA v3 reward (docs/saha-v2/reward-v3-counterfactual-spec.md). The frozen
SDS-GRPO manager (sds_grpo.py) is left untouched as the old-area-SDS baseline.

The three reward helpers below (compute_sref_grounding,
compute_counterfactual_tool_reward, resolve_referring_sref) are module-level pure
functions (unit-tested directly under verl-tool-env); the manager wires them to
the verl reward-manager API and reuses the v1 outcome/format scorers.
"""
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register

from .sds_grpo import (
    compute_format_reward,
    compute_grounding_outcome,
    compute_referring_outcome,
    count_zoom_in,
    count_zoom_out,
)


def compute_sref_grounding(anchor: Any, gt_data: dict) -> float:
    """Grounding s_ref: score the proposal anchor as if it were the model's answer.

    ``anchor`` is the list of [person_dict, object_dict] pairs built in
    prepare_train.py (extra_info["proposal_anchor"]). GT-anchored: it depends only
    on the fixed anchor and the GT, never on policy output -> immune to
    baseline-depression hacking.
    """
    if not anchor:
        return 0.0
    lines = "\n".join(json.dumps(pair) for pair in anchor)
    return compute_grounding_outcome(f"<answer>{lines}</answer>", gt_data)


def compute_counterfactual_tool_reward(
    s_final: float,
    s_ref: float,
    i_tool: int,
    clip_lo: float,
    clip_hi: float,
) -> float:
    """R_tool = I_tool * clip(s_final - s_ref, clip_lo, clip_hi).

    The tool earns reward only when the final answer beats the no-tool/proposal
    reference. The asymmetric clip keeps R_tool a bounded tie-breaker around
    R_outcome (Peak-Then-Collapse safeguard). No tool -> 0.0.
    """
    if not i_tool:
        return 0.0
    return float(min(max(s_final - s_ref, clip_lo), clip_hi))


def resolve_referring_sref(rows: list[dict], min_no_tool: int = 1) -> list:
    """Referring s_ref = mean R_outcome of same-uid NO-TOOL siblings.

    Detached constant from the completed batch (sibling rewards, no gradient path
    to the current rollout). Returns a list aligned to ``rows``; ``None`` means
    'no R_tool for this row' (no-tool rows, or tool rows whose uid group has
    fewer than ``min_no_tool`` no-tool siblings).
    """
    by_uid_no_tool: dict = defaultdict(list)
    for r in rows:
        if r["i_tool"] == 0:
            by_uid_no_tool[r["uid"]].append(r["r_outcome"])
    out: list = []
    for r in rows:
        if r["i_tool"] == 0:
            out.append(None)  # no-tool rollouts never earn R_tool
            continue
        siblings = by_uid_no_tool.get(r["uid"], [])
        out.append(float(np.mean(siblings)) if len(siblings) >= min_no_tool else None)
    return out


@register("SAHA-CF")
class SAHACounterfactualRewardManager:
    """Counterfactual tool-gain reward manager (SAHA v3)."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Single knob + calibration (overridable via reward_kwargs in the launcher).
        self.alpha = float(kwargs.get("alpha", 0.6))
        self.clip_lo = float(kwargs.get("clip_lo", -0.5))
        self.clip_hi = float(kwargs.get("clip_hi", 1.0))
        self.referring_sref = kwargs.get("referring_sref", "group_no_tool")  # or "off"
        self.min_no_tool = int(kwargs.get("min_no_tool", 1))

        # Pre-warm NLTK WordNet (thread-safety, mirrors sds_grpo.py).
        try:
            from nltk.corpus import wordnet as _wn
            _wn.ensure_loaded()
        except Exception:
            pass

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                keys = data.meta_info.get("reward_extra_keys", [])
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {k: data.non_tensor_batch[k] for k in keys},
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_printed: dict[str, int] = {}

        # ---- Pass 1: per-item format / outcome / tool counts + grounding s_ref ----
        items: list[dict] = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            response_ids = data_item.batch["responses"]
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_length], skip_special_tokens=True
            )

            ground_truth_raw = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}

            if isinstance(ground_truth_raw, str):
                try:
                    gt_data = json.loads(ground_truth_raw)
                except (json.JSONDecodeError, TypeError):
                    gt_data = {"response": ground_truth_raw}
            elif isinstance(ground_truth_raw, dict):
                gt_data = ground_truth_raw
            else:
                gt_data = {}

            task_type = extra_info.get("task_type", "grounding")
            # GRPO group key (verified: "uid"; fall back to "index" then position).
            uid = data_item.non_tensor_batch.get(
                "uid", data_item.non_tensor_batch.get("index", i)
            )

            r_format = compute_format_reward(response_str, task_type)
            if task_type == "grounding":
                r_outcome = compute_grounding_outcome(response_str, gt_data)
            else:
                r_outcome = compute_referring_outcome(response_str, gt_data)

            n_zoom_in = count_zoom_in(response_str)
            n_zoom_out = count_zoom_out(response_str)
            i_tool = 1 if (n_zoom_in + n_zoom_out) > 0 else 0

            s_ref_g = (
                compute_sref_grounding(extra_info.get("proposal_anchor"), gt_data)
                if task_type == "grounding"
                else None
            )

            items.append({
                "idx": i,
                "uid": uid,
                "task_type": task_type,
                "resp_len": int(valid_response_length),
                "r_format": r_format,
                "r_outcome": r_outcome,
                "i_tool": i_tool,
                "n_zoom_in": n_zoom_in,
                "n_zoom_out": n_zoom_out,
                "s_ref_g": s_ref_g,
                "data_source": data_source,
                "response_str": response_str,
            })

        # ---- referring s_ref via per-uid no-tool mean (detached) ----
        ref_rows = [r for r in items if r["task_type"] != "grounding"]
        if self.referring_sref == "group_no_tool":
            ref_sref = resolve_referring_sref(ref_rows, self.min_no_tool)
        else:
            ref_sref = [None] * len(ref_rows)
        ref_sref_by_idx = {r["idx"]: s for r, s in zip(ref_rows, ref_sref)}

        # ---- Pass 2: R_tool, R_total, logging, reward tensor ----
        for r in items:
            if r["task_type"] == "grounding":
                s_ref = r["s_ref_g"] if r["s_ref_g"] is not None else 0.0
                r_tool = compute_counterfactual_tool_reward(
                    r["r_outcome"], s_ref, r["i_tool"], self.clip_lo, self.clip_hi
                )
            else:
                s_ref = ref_sref_by_idx.get(r["idx"])
                r_tool = (
                    compute_counterfactual_tool_reward(
                        r["r_outcome"], s_ref, r["i_tool"], self.clip_lo, self.clip_hi
                    )
                    if s_ref is not None
                    else 0.0
                )

            r_total = r["r_format"] * (r["r_outcome"] + self.alpha * r_tool)
            accuracy = 1.0 if r["r_outcome"] > 0 else 0.0

            score_dict = {
                "score": r_total,
                "accuracy": accuracy,
                "r_format": r["r_format"],
                "r_outcome": r["r_outcome"],
                "r_tool": r_tool,
                "s_final": r["r_outcome"],
                "s_ref": float(s_ref) if s_ref is not None else float("nan"),
                "tool_gain": r_tool,
                "i_tool": float(r["i_tool"]),
                "n_zoom_in": float(r["n_zoom_in"]),
                "n_zoom_out": float(r["n_zoom_out"]),
                "is_grounding": 1.0 if r["task_type"] == "grounding" else 0.0,
            }
            for key, value in score_dict.items():
                reward_extra_info[key].append(value)

            reward = score_dict["accuracy"] if self.num_examine == 1 else r_total
            reward_tensor[r["idx"], r["resp_len"] - 1] = reward

            ds = r["data_source"]
            already_printed.setdefault(ds, 0)
            if already_printed[ds] < self.num_examine:
                already_printed[ds] += 1
                print(f"[saha-cf response] {r['response_str'][:200]}...")
                for key, value in score_dict.items():
                    print(f"[{key}] {value}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        return reward_tensor

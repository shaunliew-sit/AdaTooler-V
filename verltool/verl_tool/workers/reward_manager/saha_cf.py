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
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register

from .sds_grpo import (
    calculate_iou,
    compute_format_reward,
    compute_grounding_outcome,
    compute_referring_outcome,
    count_zoom_in,
    count_zoom_out,
    parse_grounding_answer,
)


# ---------------------------------------------------------------------------
# Run A — eval-aligned grounding scorer (min-IoU, 10 thresholds).
#
# The frozen sds_grpo.compute_grounding_outcome scores pairs with an AVERAGED IoU
# (0.5*person + 0.5*object) at only {0.5, 0.75} — looser than the eval Average
# Recall, which gates on min(person_iou, object_iou) over 10 thresholds 0.5..0.95
# (eval_hico_ground_sftgrpo_qwen3vl.py:pair_iou + AR sweep). That mismatch let a
# tool action that polishes the already-better box earn reward while moving eval
# AR zero. compute_grounding_outcome_ar mirrors the eval matcher so the reward
# optimizes what eval measures. Selected via SAHA_CF_GROUNDING_METRIC
# (minAR10=default | avg2=frozen v1). sds_grpo.py is NOT modified.
#
# Residual (documented, out of scope for Run A): the reward scores IoU in the
# grid-1000 frame (gt_data carries boxes_1000 only, no width/height), while eval
# scores in pixel space; for non-square images the two IoUs differ slightly.
# Closing it would need width/height threaded into gt_data via a parquet regen
# (not a GRPO-only change).
# ---------------------------------------------------------------------------

_AR_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95 (eval AR sweep)


def _match_pairs_greedy_min(pred_pairs: list, gt_pairs: list, threshold: float) -> int:
    """Greedy one-to-one pair match gated on min(person_iou, object_iou) >= threshold.

    Byte-faithful to eval_hico_ground_sftgrpo_qwen3vl.py::match_pairs_greedy
    (pair_iou = min of the two box IoUs; assign by descending IoU). Returns the
    matched-pair count.
    """
    if not pred_pairs or not gt_pairs:
        return 0
    scores: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_pairs):
        for gi, gt in enumerate(gt_pairs):
            person_iou = calculate_iou(pred[0]["bbox_2d"], gt[0]["bbox_2d"])
            object_iou = calculate_iou(pred[1]["bbox_2d"], gt[1]["bbox_2d"])
            scores.append((min(person_iou, object_iou), pi, gi))
    scores.sort(key=lambda x: x[0], reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    count = 0
    for score, pi, gi in scores:
        if pi in matched_pred or gi in matched_gt:
            continue
        if score >= threshold:
            count += 1
            matched_pred.add(pi)
            matched_gt.add(gi)
    return count


def compute_grounding_outcome_ar(pred_text: str, gt_data: dict) -> float:
    """Grounding outcome = Average Recall over IoU 0.5..0.95 (10 thr), min-IoU pairing.

    Per-sample AR: mean over thresholds of matched/num_gt_pairs. Mirrors the eval
    AR definition (per-sample recall = tp/(tp+fn) with tp=matched, fn=unmatched_gts).
    Same parse + GT reconstruction as sds_grpo.compute_grounding_outcome; only the
    pairing rule (min vs avg) and threshold set ({0.5..0.95} vs {0.5,0.75}) differ.
    """
    pred_pairs = parse_grounding_answer(pred_text)
    boxes_1000 = gt_data.get("boxes_1000", [])
    num_pairs = gt_data.get("num_pairs", 0)

    gt_pairs: list[list[dict]] = []
    for i in range(num_pairs):
        person_idx = i * 2
        object_idx = i * 2 + 1
        if object_idx < len(boxes_1000):
            gt_pairs.append([
                {"bbox_2d": boxes_1000[person_idx]},
                {"bbox_2d": boxes_1000[object_idx]},
            ])

    if not gt_pairs:
        return 1.0 if not pred_pairs else 0.0

    recalls = [
        _match_pairs_greedy_min(pred_pairs, gt_pairs, t) / len(gt_pairs)
        for t in _AR_THRESHOLDS
    ]
    return sum(recalls) / len(recalls)


def resolve_grounding_scorer(name: str | None = None):
    """Select the grounding outcome scorer. minAR10 (eval-aligned, default) | avg2 (frozen v1)."""
    key = (name or os.environ.get("SAHA_CF_GROUNDING_METRIC", "minAR10")).strip().lower()
    if key in ("minar10", "min_ar10", "ar10", "minar", "min"):
        return compute_grounding_outcome_ar
    if key in ("avg2", "avg", "v1", "legacy", "sds"):
        return compute_grounding_outcome
    raise ValueError(
        f"unknown SAHA_CF_GROUNDING_METRIC={key!r} (expected 'minAR10' or 'avg2')"
    )


def compute_sref_grounding(anchor: Any, gt_data: dict, scorer: Any = None) -> float:
    """Grounding s_ref: score the proposal anchor as if it were the model's answer.

    ``anchor`` is the list of [person_dict, object_dict] pairs built in
    prepare_train.py (extra_info["proposal_anchor"]). GT-anchored: it depends only
    on the fixed anchor and the GT, never on policy output -> immune to
    baseline-depression hacking.

    ``scorer`` MUST be the same grounding outcome scorer used for s_final, or the
    counterfactual gain s_final - s_ref would compare two different metrics. Defaults
    to the env-selected scorer (resolve_grounding_scorer).
    """
    if scorer is None:
        scorer = resolve_grounding_scorer()
    # Robust emptiness check: `not anchor` raises on a numpy-array anchor ("ambiguous truth
    # value"). At RL time verl delivers native lists (live runs score s_ref fine), but harden
    # against ndarray inputs from any loader change.
    if anchor is None or len(anchor) == 0:
        # No usable proposal for the target (absent/empty proposal list). s_ref=0
        # means "proposals do not help here", so a tool that lands a correct answer
        # is fully credited as genuine recovery (spec §2.1 missing-proposal case).
        return 0.0
    # `default=` lets json.dumps serialize numpy bbox arrays/scalars (-> .tolist()) without error.
    lines = "\n".join(
        json.dumps(pair, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        for pair in anchor)
    return scorer(f"<answer>{lines}</answer>", gt_data)


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

        # Single knob + calibration. Precedence: reward_kwargs (hydra) > SAHA_CF_*
        # env var > default. The env fallback makes the 4B sweep robust even if the
        # hydra reward_kwargs add is dropped/struct-blocked for an async worker.
        self.alpha = float(kwargs.get("alpha", os.environ.get("SAHA_CF_ALPHA", 0.6)))
        self.clip_lo = float(kwargs.get("clip_lo", os.environ.get("SAHA_CF_CLIP_LO", -0.5)))
        self.clip_hi = float(kwargs.get("clip_hi", os.environ.get("SAHA_CF_CLIP_HI", 1.0)))
        self.referring_sref = kwargs.get(
            "referring_sref", os.environ.get("SAHA_CF_REFERRING_SREF", "group_no_tool")
        )  # "group_no_tool" or "off"
        self.min_no_tool = int(kwargs.get("min_no_tool", os.environ.get("SAHA_CF_MIN_NO_TOOL", 1)))
        # Grounding outcome metric: "minAR10" (eval-aligned min-IoU 10-thr, default) | "avg2"
        # (frozen v1 avg-IoU @{0.5,0.75}). Same scorer drives s_final AND s_ref (Run A).
        self.grounding_metric = kwargs.get(
            "grounding_metric", os.environ.get("SAHA_CF_GROUNDING_METRIC", "minAR10")
        )
        self.grounding_scorer = resolve_grounding_scorer(self.grounding_metric)
        print(
            f"[SAHA-CF] grounding_metric={self.grounding_metric} "
            f"(scorer={self.grounding_scorer.__name__}) alpha={self.alpha} "
            f"clip=[{self.clip_lo},{self.clip_hi}]",
            flush=True,
        )
        # Per-call console dashboard (tool-collapse visibility). Logging only;
        # never affects the reward. Disable with SAHA_CF_LOG_SUMMARY=0.
        self.log_summary = str(
            kwargs.get("log_summary", os.environ.get("SAHA_CF_LOG_SUMMARY", "1"))
        ).lower() not in ("0", "false", "no")

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
                r_outcome = self.grounding_scorer(response_str, gt_data)
            else:
                r_outcome = compute_referring_outcome(response_str, gt_data)

            n_zoom_in = count_zoom_in(response_str)
            n_zoom_out = count_zoom_out(response_str)
            i_tool = 1 if (n_zoom_in + n_zoom_out) > 0 else 0

            s_ref_g = (
                compute_sref_grounding(
                    extra_info.get("proposal_anchor"), gt_data, scorer=self.grounding_scorer
                )
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

            s_ref_defined = s_ref is not None
            # Unclipped gain (for diagnosing how often the clip binds); 0 when no
            # tool or no reference.
            raw_gain = (r["r_outcome"] - s_ref) if (r["i_tool"] and s_ref_defined) else 0.0

            score_dict = {
                "score": r_total,
                "accuracy": accuracy,
                "r_format": r["r_format"],
                "r_outcome": r["r_outcome"],
                "r_tool": r_tool,
                "s_final": r["r_outcome"],
                # 0.0 (never NaN) when there is no reference, so np.mean over the
                # logged list stays finite; has_sref separates a real 0.0 reference
                # (e.g. missing/absent proposal -> recovery case) from "no reference".
                "s_ref": float(s_ref) if s_ref_defined else 0.0,
                "has_sref": 1.0 if s_ref_defined else 0.0,
                "tool_gain_raw": float(raw_gain),
                "tool_gain_clipped": r_tool,
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

        # ---- per-call tool-collapse dashboard (logging only) ----
        # Watch `tool_rate` across steps: collapse = it trends to 0. The
        # counterfactual fix is working when tool_rate stays healthy AND
        # acc(tool) >= acc(notool) with R_tool|tool > 0 (tools earn reward where
        # they genuinely beat trusting the proposal).
        if self.log_summary and len(reward_extra_info.get("i_tool", [])):
            try:
                def _col(k):
                    return np.asarray(reward_extra_info[k], dtype=float)

                def _mean(a, mask=None):
                    a = a if mask is None else a[mask]
                    return float(a.mean()) if a.size else float("nan")

                i_tool = _col("i_tool")
                is_g = _col("is_grounding") > 0.5
                tool = i_tool > 0.5
                acc, r_tool = _col("accuracy"), _col("r_tool")
                gain, s_ref = _col("tool_gain_raw"), _col("s_ref")
                z_in, z_out = _col("n_zoom_in"), _col("n_zoom_out")
                print(
                    f"[SAHA-CF rollout] N={len(i_tool)} "
                    f"tool_rate={_mean(i_tool):.3f}(g={_mean(i_tool, is_g):.3f} "
                    f"r={_mean(i_tool, ~is_g):.3f}) | "
                    f"acc={_mean(acc):.3f} tool={_mean(acc, tool):.3f} "
                    f"notool={_mean(acc, ~tool):.3f} | "
                    f"R_tool|tool={_mean(r_tool, tool):.3f} "
                    f"gain_raw|tool={_mean(gain, tool):.3f} "
                    f"s_ref|g={_mean(s_ref, is_g):.3f} | "
                    f"zoom_in={_mean(z_in):.2f} zoom_out={_mean(z_out):.2f}",
                    flush=True,
                )
            except Exception as e:  # logging must never break training
                print(f"[SAHA-CF rollout] summary skipped: {e}", flush=True)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        return reward_tensor

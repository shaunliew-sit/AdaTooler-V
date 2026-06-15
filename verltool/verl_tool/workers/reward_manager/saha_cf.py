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

Pure reward math lives in the stdlib-only saha_cf_core.py so it is unit-testable
without torch/verl; this module wires it to the verl reward-manager API and the
existing v1 outcome/format scorers.
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
        raise NotImplementedError("SAHA-CF __call__ is implemented in Task 5")

"""Curation gates for generated SFT traces (plan/saha-sft-regen/PLAN.md §6).

A trace is KEPT only if it passes every gate. Gates reuse the LIVE reward scorers so
"correct" means exactly what RL rewards. The anti-override + curriculum gates are what
make the new dataset fix the diagnosed defects.
"""
import json
import re

from verl_tool.workers.reward_manager.sds_grpo import (
    compute_format_reward,
    compute_grounding_outcome,
    compute_referring_outcome,
    parse_grounding_answer,
)
from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding

OUTCOME_KEEP = 0.5      # min s_final vs GT to keep a trace
THINK_MIN_CHARS = 40    # reject thin/empty <think> (the empty-output failure mode)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def gate_no_gt_leak(api_messages, gt_data):
    """Assert no GT answer leaked into the prompt. Checks (a) string-valued GT fields (e.g.
    referring 'response') and (b) the EXACT serialized GT box list (json.dumps(boxes_1000)) —
    a single proposal's [x1,y1,x2,y2] can coincide with a GT box, but the full serialized
    multi-box GT list won't appear from the proposal block, so this avoids false-rejects."""
    blob = json.dumps(api_messages, default=str)
    for k, v in (gt_data or {}).items():
        if isinstance(v, str) and len(v) > 6 and v in blob:
            return False, f"GT leak: {k}={v[:40]!r} found in prompt"
        if k == "boxes_1000" and isinstance(v, (list, tuple)) and len(v) >= 1:
            if json.dumps(list(v)) in blob:
                return False, "GT leak: serialized boxes_1000 found in prompt"
    return True, ""


def gate_format(gpt_turns, final_answer_text, task_type):
    """Every gpt turn must carry a substantive <think>; the FINAL turn must pass the EXACT RL
    format reward (compute_format_reward: <think>...</think>.*<answer>...</answer>, ordered +
    closed). For grounding, the <answer> must also parse to >=1 valid pair (line shape) unless
    it is the explicit no-pairs sentence."""
    for i, t in enumerate(gpt_turns):
        m = re.search(r"<think>(.*?)</think>", t, re.S)
        if not m or len(m.group(1).strip()) < THINK_MIN_CHARS:
            return False, f"turn {i}: missing/thin <think>"
    if compute_format_reward(final_answer_text, task_type) != 1.0:
        return False, "final turn fails RL compute_format_reward (think/answer tags/order)"
    if task_type == "grounding":
        ans_body = re.search(r"<answer>(.*?)</answer>", final_answer_text, re.S)
        body = ans_body.group(1).strip().lower() if ans_body else ""
        if "no valid pairs" not in body and not parse_grounding_answer(final_answer_text):
            return False, "grounding <answer> has no well-formed 2-object line"
    return True, ""


def gate_tool_gain(task_type, n_tool, s_final, s_ref):
    """A TOOL trace must JUSTIFY the tool: the zoom must beat the proposal anchor
    (s_final > s_ref). This rejects the proposal-copy failure — 'zoom, then just copy the
    proposal anchor' (s_final == s_ref, zero tool gain, the exact zero/neg R_tool pattern that
    collapses the tool) — and pointless zooms on already-perfect proposals (s_ref=1.0).
    No-tool traces are exempt (copying a good proposal is the correct easy-case behavior).
    Grounding only; referring s_ref is a train-time per-uid group statistic, not offline-known."""
    if task_type != "grounding" or not n_tool:
        return True, ""
    if s_ref is None:
        return True, ""
    if s_final <= s_ref + 1e-9:
        return False, f"tool_no_gain: s_final={s_final:.2f} <= s_ref={s_ref:.2f} (zoom-then-copy)"
    return True, ""


def score_outcome(task_type, final_answer_text, gt_data):
    if task_type == "grounding":
        return float(compute_grounding_outcome(final_answer_text, gt_data))
    return float(compute_referring_outcome(final_answer_text, gt_data))


def grounding_sref(proposal_anchor, gt_data):
    try:
        return float(compute_sref_grounding(proposal_anchor, gt_data))
    except Exception:
        return 0.0


def _answer_boxes(final_answer_text):
    seg = final_answer_text.split("<answer>")[-1].split("</answer>")[0] if "<answer>" in final_answer_text else ""
    out = []
    for b in re.findall(r'"bbox_2d":\s*\[([^\]]+)\]', seg):
        try:
            out.append([int(float(x)) for x in b.split(",")])
        except Exception:
            pass
    return out


def gate_anti_override(final_answer_text, proposals, gt_boxes):
    """Reject grounding traces that wasted/overrode an available good proposal (Layer 8 defect).
    GT-centric: for each GT box, if a GOOD proposal existed for it (some proposal IoU>0.5)
    but the final answer FAILED to cover that GT box (no answer box IoU>0.5), the model had a
    correct proposal in hand and still missed -> reject. (Answer-centric attribution is unsafe:
    a far-off override box has ~0 IoU with the right GT and would mis-resolve.)"""
    ans = _answer_boxes(final_answer_text)
    if not ans or not gt_boxes:
        return True, ""  # nothing to check (e.g. no-pairs / referring)
    for g in gt_boxes:
        prop_good = max((_iou(p, g) for p in proposals), default=0.0) > 0.5
        ans_hit = max((_iou(a, g) for a in ans), default=0.0) > 0.5
        if prop_good and not ans_hit:
            return False, f"override: good proposal existed for GT {g} but answer missed it"
    return True, ""

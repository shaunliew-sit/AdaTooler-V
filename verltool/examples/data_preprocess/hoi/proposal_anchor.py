"""Grounding proposal-anchor builder for the SAHA v3 counterfactual reward.

Stdlib-only (no datasets/fire/torch) so it can be imported by both the
preprocessing script and a lightweight unit test.

The anchor is the synthetic "proposal-only answer": for each GT pair, the proposal
box best matching the GT person and the GT object by IoU. It is stored in
``extra_info["proposal_anchor"]`` at preprocess time and later scored against the
target GT to produce ``s_ref`` (the trust-the-proposal reference) in the reward
manager. It is GT-anchored: GT is used only to *select* among the given
proposals, so the reference is independent of the policy.
"""
from typing import Any


def iou(b1: list[float], b2: list[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes; 0.0 on malformed input."""
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def build_proposal_anchor_grounding(
    proposals: list[dict[str, Any]],
    boxes_1000: list[list[float]],
    num_pairs: int,
) -> list[list[dict[str, Any]]]:
    """Build the proposal-only grounding answer.

    Args:
        proposals: YOLOE proposals, each with a ``bbox_2d`` 1000-grid box.
        boxes_1000: flat list of GT boxes, pairs at [2i] (person) / [2i+1] (object).
        num_pairs: number of GT person-object pairs.

    Returns:
        List of [person_dict, object_dict] pairs (parser-compatible with
        ``parse_grounding_answer``), or [] if no usable proposals / no pairs.
    """
    if not proposals or num_pairs <= 0:
        return []
    boxes = [p["bbox_2d"] for p in proposals
             if p.get("bbox_2d") and len(p["bbox_2d"]) == 4]
    if not boxes:
        return []

    anchor: list[list[dict[str, Any]]] = []
    for i in range(num_pairs):
        pidx, oidx = 2 * i, 2 * i + 1
        if oidx >= len(boxes_1000):
            break
        gt_p, gt_o = boxes_1000[pidx], boxes_1000[oidx]
        best_p = max(boxes, key=lambda b: iou(b, gt_p))
        best_o = max(boxes, key=lambda b: iou(b, gt_o))
        anchor.append([
            {"bbox_2d": best_p, "label": "person"},
            {"bbox_2d": best_o, "label": "object"},
        ])
    return anchor

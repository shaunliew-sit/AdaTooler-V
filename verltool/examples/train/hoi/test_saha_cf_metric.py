"""Run A unit tests — eval-aligned grounding scorer (min-IoU, 10 thresholds).

Validates that saha_cf.compute_grounding_outcome_ar reproduces the eval AR matching
logic (min(person_iou,object_iou), greedy one-to-one, mean recall over 0.5..0.95),
that it diverges from the frozen avg-IoU@{0.5,0.75} v1 scorer exactly where expected,
and that s_final and s_ref are driven by the SAME selected scorer.

Run (no pytest needed):  python tests/test_saha_cf_metric.py    [in verl-tool-env]
"""
import json
import sys

from verl_tool.workers.reward_manager.saha_cf import (
    _AR_THRESHOLDS,
    _match_pairs_greedy_min,
    compute_grounding_outcome,        # frozen v1 (avg-IoU @{0.5,0.75})
    compute_grounding_outcome_ar,     # Run A (min-IoU, 10 thr)
    compute_sref_grounding,
    resolve_grounding_scorer,
)


def _answer(pairs):
    """pairs: list of (person_box, object_box) -> grounding <answer> text."""
    lines = [
        json.dumps([{"bbox_2d": list(p), "label": "person"},
                    {"bbox_2d": list(o), "label": "object"}])
        for p, o in pairs
    ]
    return "<answer>" + "\n".join(lines) + "</answer>"


def _gt(pairs):
    flat = []
    for p, o in pairs:
        flat.append(list(p))
        flat.append(list(o))
    return {"boxes_1000": flat, "num_pairs": len(pairs)}


def _approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


def test_thresholds_are_eval_ar_sweep():
    assert _AR_THRESHOLDS == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], _AR_THRESHOLDS


def test_perfect_and_miss():
    # Identical boxes -> IoU 1.0 at every threshold -> AR 1.0
    pair = ((0, 0, 100, 100), (200, 200, 300, 300))
    assert _approx(compute_grounding_outcome_ar(_answer([pair]), _gt([pair])), 1.0)
    # Disjoint prediction -> AR 0.0
    bad = ((900, 900, 999, 999), (800, 800, 850, 850))
    assert _approx(compute_grounding_outcome_ar(_answer([bad]), _gt([pair])), 0.0)
    # Empty GT + empty pred -> 1.0 (mirrors v1 edge case)
    assert _approx(compute_grounding_outcome_ar("<answer></answer>", {"boxes_1000": [], "num_pairs": 0}), 1.0)


def test_min_iou_gate():
    # person IoU = 1.0 (identical), object IoU = 0.5 exactly.
    # object boxes: gt [0,0,2,1] (area2), pred [0,0,4,1] (area4); inter 2, union 4 -> 0.5.
    person = (0, 0, 100, 100)
    gt = _gt([(person, (0, 0, 2, 1))])
    ans = _answer([(person, (0, 0, 4, 1))])
    # min-IoU = 0.5 -> counts only at threshold 0.5 -> AR = 1/10
    ar = compute_grounding_outcome_ar(ans, gt)
    assert _approx(ar, 0.1), ar
    # The matcher itself: matched at 0.5, not at 0.55
    pp = [[{"bbox_2d": [0, 0, 100, 100]}, {"bbox_2d": [0, 0, 4, 1]}]]
    gp = [[{"bbox_2d": [0, 0, 100, 100]}, {"bbox_2d": [0, 0, 2, 1]}]]
    assert _match_pairs_greedy_min(pp, gp, 0.5) == 1
    assert _match_pairs_greedy_min(pp, gp, 0.55) == 0


def test_divergence_from_v1():
    # Same case: v1 averages IoU = (1.0+0.5)/2 = 0.75 -> passes BOTH @0.5 and @0.75 -> 1.0
    # Run A uses min = 0.5 -> AR = 0.1. This is exactly the train/eval mismatch Run A fixes.
    person = (0, 0, 100, 100)
    gt = _gt([(person, (0, 0, 2, 1))])
    ans = _answer([(person, (0, 0, 4, 1))])
    v1 = compute_grounding_outcome(ans, gt)
    runA = compute_grounding_outcome_ar(ans, gt)
    assert _approx(v1, 1.0), v1
    assert _approx(runA, 0.1), runA
    assert v1 > runA  # v1 over-credits the polished-person/weak-object case


def test_resolver():
    assert resolve_grounding_scorer("minAR10") is compute_grounding_outcome_ar
    assert resolve_grounding_scorer("avg2") is compute_grounding_outcome
    assert resolve_grounding_scorer(None) is compute_grounding_outcome_ar  # default
    try:
        resolve_grounding_scorer("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sref_uses_selected_scorer():
    # s_ref must route through the SAME scorer as s_final. Pass a sentinel scorer and
    # confirm it is the one invoked (not the hardcoded v1).
    person = (0, 0, 100, 100)
    gt = _gt([(person, (0, 0, 2, 1))])
    anchor = [[{"bbox_2d": list(person), "label": "person"},
               {"bbox_2d": [0, 0, 4, 1], "label": "object"}]]
    called = {}

    def sentinel(text, gt_data):
        called["hit"] = True
        return 0.4242

    assert _approx(compute_sref_grounding(anchor, gt, scorer=sentinel), 0.4242)
    assert called.get("hit")
    # Default (no scorer arg) -> env-selected (minAR10) -> min-IoU AR = 0.1 here
    assert _approx(compute_sref_grounding(anchor, gt), 0.1)


def test_multipair_greedy():
    # Two GT pairs; one perfect, one missed -> AR over thresholds = (1 perfect + 0 missed)/2 = 0.5
    p1 = ((0, 0, 100, 100), (200, 200, 300, 300))     # perfect
    p2 = ((400, 400, 500, 500), (600, 600, 700, 700))  # gt; pred disjoint
    gt = _gt([p1, p2])
    ans = _answer([p1, ((10, 10, 11, 11), (12, 12, 13, 13))])
    # perfect pair contributes recall 1 at all thresholds; missed pair 0 -> per-thr recall 0.5
    assert _approx(compute_grounding_outcome_ar(ans, gt), 0.5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

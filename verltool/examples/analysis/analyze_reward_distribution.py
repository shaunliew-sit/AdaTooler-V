"""
Offline analysis of R_tool reward distribution on raw SFT data.

Loads a random sample from the HOI JSON files, simulates GT tool usage patterns
(derived from plan.md SFT statistics), and validates that the reward signal is
well-shaped:

1. Correct ordering: tool-using samples rank higher than no-tool on high-SDS
2. Correct ordering: no-tool samples rank higher on low-SDS
3. Variance check: std(R_tool) > 0.05 (signal not collapsed)
4. Pattern verification: grounding zoom_in→zoom_out higher than no-tool on high-SDS
5. Referring zoom_in-only not penalized vs zoom_in→zoom_out

Usage:
    cd verltool
    python examples/analysis/analyze_reward_distribution.py \\
        --data_dir /media/shaun/workspace/hoi/dataset/benchmarks_simplified \\
        --n_samples 2000 \\
        --output_dir examples/analysis/output/

Requirements (lightweight):
    pip install numpy matplotlib pandas
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add verltool root to path so we can import sds_grpo directly.
# Stub torch/verl so the script runs without the full training environment.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from unittest.mock import MagicMock

def _register_passthrough(name: str):
    def decorator(cls): return cls
    return decorator

for _mod in ["torch", "verl", "verl.workers", "verl.workers.reward_manager"]:
    sys.modules.setdefault(_mod, MagicMock())
_reg_mock = MagicMock()
_reg_mock.register = _register_passthrough
_reg_mock.REWARD_MANAGER_REGISTRY = {}
sys.modules["verl.workers.reward_manager.registry"] = _reg_mock

from verl_tool.workers.reward_manager.sds_grpo import compute_sds, compute_tool_reward


# ---------------------------------------------------------------------------
# SFT-derived tool usage distributions (from plan.md Section 1)
# ---------------------------------------------------------------------------

# Grounding tool patterns and their probabilities among tool-using samples
GROUNDING_TOOL_PATTERNS = [
    # (n_zoom_in, n_zoom_out, weight)
    (1, 1, 0.449),   # zoom_in → zoom_out (single region)
    (2, 2, 0.274),   # zoom_in → zoom_out → zoom_in → zoom_out (two regions)
    (1, 0, 0.010),   # zoom_in only (rare for grounding)
    (2, 0, 0.070),   # double zoom_in (rare)
    (0, 0, 0.197),   # no tools (already separated out below)
]

# Referring tool patterns
REFERRING_TOOL_PATTERNS = [
    (1, 0, 0.380),   # zoom_in only (most common)
    (0, 0, 0.314),   # no tools
    (1, 1, 0.165),   # zoom_in → zoom_out
    (2, 0, 0.141),   # multi-zoom
]

# Fraction of samples that use tools
GROUNDING_TOOL_FRAC = 0.794  # 1 - 0.206 (no-tool fraction)
REFERRING_TOOL_FRAC = 0.686  # 1 - 0.314


def sample_tool_pattern(task_type: str, rng: random.Random) -> tuple[int, int]:
    """Sample (n_zoom_in, n_zoom_out) from the SFT-derived distribution."""
    if task_type == "grounding":
        patterns = GROUNDING_TOOL_PATTERNS
    else:
        patterns = REFERRING_TOOL_PATTERNS

    choices = [(z_in, z_out) for z_in, z_out, _ in patterns]
    weights = [w for _, _, w in patterns]
    return rng.choices(choices, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(data_dir: str, n_samples: int, seed: int = 42) -> list[dict]:
    """Load a random subset from all 4 HOI JSON files."""
    files = [
        ("hico_ground_train_simplified.json", "grounding"),
        ("swig_ground_train_simplified.json", "grounding"),
        ("hico_referring_train_simplified.json", "referring"),
        ("swig_referring_train_simplified.json", "referring"),
    ]

    rng = random.Random(seed)
    all_raw: list[dict] = []

    for filename, task_type in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        per_file = n_samples // len(files)
        subset = rng.sample(data, min(per_file, len(data)))
        for s in subset:
            s["_task_type"] = task_type
        all_raw.extend(subset)
        print(f"  Loaded {len(subset)} samples from {filename}")

    return all_raw


def compute_sds_for_sample(sample: dict) -> float:
    """Compute SDS from raw sample dict."""
    task_type = sample["_task_type"]
    if task_type == "grounding":
        boxes_1000 = sample.get("boxes_1000", [])
        num_pairs = sample.get("num_pairs", 0)
        return compute_sds(boxes_1000, num_pairs)
    else:
        # For referring: person is at person_box_idx, object at object_box_idx
        boxes_1000 = sample.get("boxes_1000", [])
        person_idx = sample.get("person_box_idx", 0)
        object_idx = sample.get("object_box_idx", 1)
        if person_idx < len(boxes_1000) and object_idx < len(boxes_1000):
            pair_boxes = [boxes_1000[person_idx], boxes_1000[object_idx]]
            return compute_sds(pair_boxes, num_pairs=1)
        return 0.5


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_reward_for_scenarios(sds: float) -> dict[str, float]:
    """Compute R_tool for 3 scenarios given SDS."""
    n_opt = round(2.0 * sds)

    # GT pattern: use n_opt zoom_ins and n_opt zoom_outs (best case GT)
    r_gt, _, _, _ = compute_tool_reward(sds, n_opt, n_opt)

    # No-tool
    r_no_tool, _, _, _ = compute_tool_reward(sds, 0, 0)

    # Over-tooled (2 extra zoom_ins)
    r_over, _, _, _ = compute_tool_reward(sds, n_opt + 2, 0)

    return {"gt": r_gt, "no_tool": r_no_tool, "over": r_over, "n_opt": float(n_opt)}


def sds_bucket(sds: float) -> str:
    """Assign SDS to a named bucket."""
    if sds < 0.3:
        return "low (<0.3)"
    elif sds < 0.7:
        return "medium (0.3-0.7)"
    else:
        return "high (≥0.7)"


def run_ordering_check(records: list[dict]) -> dict:
    """
    Check: for high-SDS samples, GT pattern should rank higher than no-tool.
    For low-SDS samples, no-tool should rank >= GT pattern.
    Returns: dict with pass/fail and fraction correct.
    """
    results: dict[str, dict] = {
        "high": {"correct_gt_above_no_tool": 0, "total": 0},
        "low": {"correct_no_tool_above_gt": 0, "total": 0},
    }

    for r in records:
        sds = r["sds"]
        rewards = r["rewards"]
        bucket = sds_bucket(sds)

        if bucket == "high (≥0.7)":
            results["high"]["total"] += 1
            if rewards["gt"] > rewards["no_tool"]:
                results["high"]["correct_gt_above_no_tool"] += 1
        elif bucket == "low (<0.3)":
            results["low"]["total"] += 1
            if rewards["no_tool"] >= rewards["gt"]:
                results["low"]["correct_no_tool_above_gt"] += 1

    return results


def run_variance_check(r_tool_values: list[float], min_std: float = 0.05) -> dict:
    """Check that R_tool values have sufficient variance."""
    arr = np.array(r_tool_values)
    std = float(np.std(arr))
    return {
        "std": std,
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "passed": std > min_std,
    }


def run_pattern_verification(records: list[dict]) -> dict:
    """
    Verify specific SFT patterns:
    - Grounding zoom_in→zoom_out on high-SDS: avg R_tool > 0
    - Referring zoom_in-only not penalized vs zoom_in→zoom_out
    - No-tool on low-SDS: avg R_tool near 0
    """
    high_sds_grounding = [r for r in records if r["sds"] >= 0.7 and r["task_type"] == "grounding"]
    low_sds = [r for r in records if r["sds"] < 0.3]
    medium_referring = [r for r in records if 0.3 <= r["sds"] < 0.7 and r["task_type"] == "referring"]

    def avg_r_tool(sds_list: list[float], n_in: int, n_out: int) -> float:
        if not sds_list:
            return float("nan")
        vals = [compute_tool_reward(s, n_in, n_out)[0] for s in sds_list]
        return float(np.mean(vals))

    high_sds_vals = [r["sds"] for r in high_sds_grounding]
    low_sds_vals = [r["sds"] for r in low_sds]
    medium_ref_vals = [r["sds"] for r in medium_referring]

    # High-SDS grounding: in→out pattern (n=1,n=1)
    avg_in_out_high = avg_r_tool(high_sds_vals, 1, 1)
    # High-SDS grounding: no tool
    avg_no_tool_high = avg_r_tool(high_sds_vals, 0, 0)

    # Low-SDS: no tool should be neutral (≈0)
    avg_no_tool_low = avg_r_tool(low_sds_vals, 0, 0)
    avg_unnecessary_low = avg_r_tool(low_sds_vals, 1, 0)

    # Medium referring: zoom_in-only vs zoom_in→zoom_out
    avg_in_only_med = avg_r_tool(medium_ref_vals, 1, 0)
    avg_in_out_med = avg_r_tool(medium_ref_vals, 1, 1)

    return {
        "high_sds_grounding": {
            "n_samples": len(high_sds_vals),
            "in_out_R_tool": avg_in_out_high,
            "no_tool_R_tool": avg_no_tool_high,
            "ordering_correct": avg_in_out_high > avg_no_tool_high if high_sds_vals else None,
        },
        "low_sds": {
            "n_samples": len(low_sds_vals),
            "no_tool_R_tool": avg_no_tool_low,  # should be 0.0
            "unnecessary_zoom_R_tool": avg_unnecessary_low,  # should be < 0
            "no_tool_neutral": abs(avg_no_tool_low) < 0.01 if low_sds_vals else None,
            "unnecessary_penalized": avg_unnecessary_low < 0 if low_sds_vals else None,
        },
        "medium_referring": {
            "n_samples": len(medium_ref_vals),
            "zoom_in_only_R_tool": avg_in_only_med,
            "zoom_in_out_R_tool": avg_in_out_med,
            "diff": avg_in_out_med - avg_in_only_med if medium_ref_vals else None,
            "zoom_in_only_not_penalized": avg_in_only_med >= 0 if medium_ref_vals else None,
        },
    }


# ---------------------------------------------------------------------------
# Optional: matplotlib plots
# ---------------------------------------------------------------------------

def plot_distribution(records: list[dict], output_dir: str) -> None:
    """Plot R_tool vs SDS histogram."""
    try:
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        sds_vals = [r["sds"] for r in records]
        r_gt_vals = [r["rewards"]["gt"] for r in records]
        r_no_tool = [r["rewards"]["no_tool"] for r in records]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter: R_tool(GT) vs SDS
        axes[0].scatter(sds_vals, r_gt_vals, alpha=0.3, s=5)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("SDS")
        axes[0].set_ylabel("R_tool (GT pattern)")
        axes[0].set_title("R_tool(GT) vs SDS")
        axes[0].set_xlim(0, 1)

        # Histogram of R_tool(GT) - R_tool(no-tool) [should be positive on high SDS]
        diff = [g - n for g, n in zip(r_gt_vals, r_no_tool)]
        axes[1].hist(diff, bins=50, edgecolor="black", linewidth=0.5)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("R_tool(GT) - R_tool(no-tool)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("GT vs No-Tool Advantage Distribution")

        plt.tight_layout()
        out_path = os.path.join(output_dir, "reward_distribution.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n  Plot saved: {out_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not available — skipping plots)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: str = "/media/shaun/workspace/hoi/dataset/benchmarks_simplified",
    n_samples: int = 2000,
    output_dir: str = "examples/analysis/output",
    seed: int = 42,
    no_plot: bool = False,
) -> None:
    print(f"Loading {n_samples} samples from {data_dir}...")
    raw_samples = load_samples(data_dir, n_samples, seed)
    print(f"Total loaded: {len(raw_samples)}")

    if not raw_samples:
        print("ERROR: No samples loaded. Check data_dir path.")
        sys.exit(1)

    # Build analysis records
    records: list[dict] = []
    r_gt_values: list[float] = []

    for sample in raw_samples:
        sds = compute_sds_for_sample(sample)
        task_type = sample["_task_type"]
        rewards = compute_reward_for_scenarios(sds)
        r_gt_values.append(rewards["gt"])
        records.append({
            "sds": sds,
            "task_type": task_type,
            "rewards": rewards,
        })

    # ---- Print SDS distribution ----
    print("\n=== SDS Distribution ===")
    for lo, hi, label in [
        (0.0, 0.3, "low (<0.3)"),
        (0.3, 0.5, "medium-low"),
        (0.5, 0.7, "medium-high"),
        (0.7, 0.9, "high"),
        (0.9, 1.01, "very high (≥0.9)"),
    ]:
        count = sum(1 for r in records if lo <= r["sds"] < hi)
        pct = 100 * count / len(records)
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")

    # ---- Task distribution ----
    grounding_count = sum(1 for r in records if r["task_type"] == "grounding")
    referring_count = len(records) - grounding_count
    print(f"\n=== Task Distribution ===")
    print(f"  grounding: {grounding_count} ({100*grounding_count/len(records):.1f}%)")
    print(f"  referring:  {referring_count} ({100*referring_count/len(records):.1f}%)")

    # ---- Check 1: Ordering ----
    print("\n=== Check 1: Reward Ordering ===")
    ordering = run_ordering_check(records)

    high = ordering["high"]
    if high["total"] > 0:
        frac_high = high["correct_gt_above_no_tool"] / high["total"]
        status = "PASS" if frac_high > 0.9 else "FAIL"
        print(f"  [HIGH-SDS] GT > no-tool: {frac_high:.1%} ({high['correct_gt_above_no_tool']}/{high['total']}) [{status}]")
    else:
        print("  [HIGH-SDS] No high-SDS samples (check data)")

    low = ordering["low"]
    if low["total"] > 0:
        frac_low = low["correct_no_tool_above_gt"] / low["total"]
        status = "PASS" if frac_low > 0.9 else "FAIL"
        print(f"  [LOW-SDS]  no-tool ≥ GT: {frac_low:.1%} ({low['correct_no_tool_above_gt']}/{low['total']}) [{status}]")
    else:
        print("  [LOW-SDS] No low-SDS samples (check data)")

    # ---- Check 2: Variance ----
    print("\n=== Check 2: Variance ===")
    var_check = run_variance_check(r_gt_values)
    status = "PASS" if var_check["passed"] else "FAIL"
    print(f"  std(R_tool) = {var_check['std']:.4f}  [{status}]  (need > 0.05)")
    print(f"  mean={var_check['mean']:.4f}  min={var_check['min']:.4f}  max={var_check['max']:.4f}")

    # ---- Check 3: Pattern verification ----
    print("\n=== Check 3: Pattern Verification ===")
    patterns = run_pattern_verification(records)

    h = patterns["high_sds_grounding"]
    if h["n_samples"] > 0:
        ord_ok = "PASS" if h["ordering_correct"] else "FAIL"
        print(f"  [HIGH-SDS grounding] n={h['n_samples']}:")
        print(f"    zoom_in→out R_tool = {h['in_out_R_tool']:.4f}")
        print(f"    no-tool     R_tool = {h['no_tool_R_tool']:.4f}")
        print(f"    ordering: {ord_ok}")

    lo = patterns["low_sds"]
    if lo["n_samples"] > 0:
        neutral_ok = "PASS" if lo["no_tool_neutral"] else "FAIL"
        pen_ok = "PASS" if lo["unnecessary_penalized"] else "FAIL"
        print(f"\n  [LOW-SDS] n={lo['n_samples']}:")
        print(f"    no-tool R_tool = {lo['no_tool_R_tool']:.4f} (expect ≈0.0) [{neutral_ok}]")
        print(f"    unnecessary zoom R_tool = {lo['unnecessary_zoom_R_tool']:.4f} (expect <0) [{pen_ok}]")

    med = patterns["medium_referring"]
    if med["n_samples"] > 0:
        not_pen = "PASS" if med["zoom_in_only_not_penalized"] else "FAIL"
        print(f"\n  [MEDIUM-SDS referring] n={med['n_samples']}:")
        print(f"    zoom_in-only R_tool = {med['zoom_in_only_R_tool']:.4f}  [{not_pen}]")
        print(f"    zoom_in→out  R_tool = {med['zoom_in_out_R_tool']:.4f}")
        print(f"    diff (hygiene bonus) = {med['diff']:.4f}")

    # ---- Plot ----
    if not no_plot:
        print("\n=== Generating Plots ===")
        plot_distribution(records, output_dir)

    # ---- Summary ----
    all_pass = all([
        high.get("total", 0) == 0 or (high["correct_gt_above_no_tool"] / high["total"]) > 0.9,
        low.get("total", 0) == 0 or (low["correct_no_tool_above_gt"] / low["total"]) > 0.9,
        var_check["passed"],
        h.get("ordering_correct", True),
        lo.get("no_tool_neutral", True),
        lo.get("unnecessary_penalized", True),
        med.get("zoom_in_only_not_penalized", True),
    ])

    print(f"\n{'='*50}")
    print(f"OVERALL: {'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
    print(f"{'='*50}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze R_tool reward distribution on HOI SFT data")
    parser.add_argument("--data_dir", default="/media/shaun/workspace/hoi/dataset/benchmarks_simplified")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--output_dir", default="examples/analysis/output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        no_plot=args.no_plot,
    )

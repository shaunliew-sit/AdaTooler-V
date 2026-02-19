"""
Post-generation verification of HOI train/val parquet files.

Runs 10 checks to confirm the parquets are well-formed for SDS-GRPO RL training:

1.  Files exist and are non-empty
2.  Row counts: train >> 1000, val == expected val_size
3.  Required columns present
4.  SDS distribution matches expected ranges
5.  Task distribution: grounding ~40%, referring ~60%
6.  SDS values are in [0, 1]
7.  Sample prompt structure: system + user messages with <image> tag
8.  GT data parses correctly as JSON
9.  Image paths in extra_info.images exist on disk (spot-check 100)
10. No NaN/null values in critical fields

Usage:
    cd verltool
    python examples/analysis/verify_parquets.py --parquet_dir data/hoi

Requirements:
    pip install pandas pyarrow numpy
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "data_source",
    "prompt",
    "images",
    "ability",
    "reward_model",
    "extra_info",
]


def _check(name: str, passed: bool, detail: str = "") -> bool:
    """Print check result and return passed status."""
    icon = "PASS" if passed else "FAIL"
    msg = f"  [{icon}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def check_files_exist(parquet_dir: str) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """Check 1: Files exist and are non-empty."""
    train_path = os.path.join(parquet_dir, "train.parquet")
    val_path = os.path.join(parquet_dir, "val.parquet")

    train_ok = os.path.exists(train_path) and os.path.getsize(train_path) > 0
    val_ok = os.path.exists(val_path) and os.path.getsize(val_path) > 0
    passed = train_ok and val_ok

    detail = ""
    if train_ok:
        detail += f"train: {os.path.getsize(train_path) / 1e6:.1f} MB  "
    else:
        detail += f"train: MISSING  "
    if val_ok:
        detail += f"val: {os.path.getsize(val_path) / 1e6:.1f} MB"
    else:
        detail += f"val: MISSING"

    _check("Files exist and non-empty", passed, detail)

    train_df = pd.read_parquet(train_path) if train_ok else pd.DataFrame()
    val_df = pd.read_parquet(val_path) if val_ok else pd.DataFrame()
    return passed, train_df, val_df


def check_row_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, val_size: int = 500) -> bool:
    """Check 2: Row counts are plausible."""
    train_ok = len(train_df) >= 1000
    val_ok = abs(len(val_df) - val_size) <= val_size * 0.1  # within 10% of target

    detail = f"train: {len(train_df):,} rows, val: {len(val_df):,} rows (target val={val_size})"
    return _check("Row counts", train_ok and val_ok, detail)


def check_required_columns(train_df: pd.DataFrame) -> bool:
    """Check 3: Required columns present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in train_df.columns]
    passed = len(missing) == 0
    detail = f"missing: {missing}" if missing else f"all {len(REQUIRED_COLUMNS)} columns present"
    return _check("Required columns", passed, detail)


def check_sds_distribution(train_df: pd.DataFrame) -> bool:
    """Check 4: SDS distribution matches expected ranges from plan.md README.

    Expected (from plan.md):
    - ~40% of samples have SDS < 0.3 (large objects, no zoom)
    - ~23% have SDS 0.9–1.0 (tiny objects, multi-zoom)
    """
    try:
        sds_vals = [row["spatial_difficulty_score"] for row in train_df["extra_info"]]
        sds_arr = np.array(sds_vals, dtype=float)

        low_frac = float(np.mean(sds_arr < 0.3))
        very_high_frac = float(np.mean(sds_arr >= 0.9))

        # Relaxed thresholds: within factor of 2 of expected
        low_ok = 0.2 <= low_frac <= 0.6        # expect ~40%
        very_high_ok = 0.10 <= very_high_frac <= 0.40  # expect ~23%

        for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
            count = np.sum((sds_arr >= lo) & (sds_arr < hi))
            pct = 100 * count / len(sds_arr)
            print(f"         [{lo:.1f},{hi:.1f}): {count:6,} ({pct:5.1f}%)")

        detail = f"SDS<0.3: {low_frac:.1%} (expect ~40%), SDS≥0.9: {very_high_frac:.1%} (expect ~23%)"
        return _check("SDS distribution", low_ok and very_high_ok, detail)

    except (KeyError, TypeError) as e:
        return _check("SDS distribution", False, f"error: {e}")


def check_task_distribution(train_df: pd.DataFrame) -> bool:
    """Check 5: Task distribution approximately 40% grounding / 60% referring."""
    try:
        task_types = [row.get("task_type", "unknown") for row in train_df["extra_info"]]
        ground_frac = sum(1 for t in task_types if t == "grounding") / len(task_types)
        refer_frac = 1.0 - ground_frac

        # Allow range: grounding 30-55%, referring 45-70%
        passed = 0.30 <= ground_frac <= 0.55

        detail = f"grounding: {ground_frac:.1%}, referring: {refer_frac:.1%}"
        return _check("Task distribution", passed, detail)

    except (KeyError, TypeError) as e:
        return _check("Task distribution", False, f"error: {e}")


def check_sds_in_range(train_df: pd.DataFrame) -> bool:
    """Check 6: All SDS values are in [0, 1]."""
    try:
        sds_vals = np.array([row["spatial_difficulty_score"] for row in train_df["extra_info"]], dtype=float)
        in_range = np.all((sds_vals >= 0.0) & (sds_vals <= 1.0))
        n_nan = int(np.sum(np.isnan(sds_vals)))
        detail = f"range [{sds_vals.min():.4f}, {sds_vals.max():.4f}], NaN count: {n_nan}"
        return _check("SDS values in [0, 1]", bool(in_range) and n_nan == 0, detail)
    except (KeyError, TypeError) as e:
        return _check("SDS values in [0, 1]", False, f"error: {e}")


def check_prompt_structure(train_df: pd.DataFrame, n_spot: int = 100) -> bool:
    """Check 7: Prompt has system + user messages; user content has <image> tag."""
    rng = random.Random(42)
    indices = rng.sample(range(len(train_df)), min(n_spot, len(train_df)))

    failures = []
    for idx in indices:
        row = train_df.iloc[idx]
        try:
            prompt = row["prompt"]
            if len(prompt) < 2:
                failures.append(f"row {idx}: only {len(prompt)} messages")
                continue

            roles = [m["role"] for m in prompt]
            if roles[0] != "system":
                failures.append(f"row {idx}: first role is {roles[0]}")
                continue

            user_msg = next((m for m in prompt if m["role"] == "user"), None)
            if not user_msg:
                failures.append(f"row {idx}: no user message")
                continue

            if "<image>" not in user_msg["content"]:
                failures.append(f"row {idx}: no <image> tag in user content")

        except (KeyError, TypeError) as e:
            failures.append(f"row {idx}: {e}")

    passed = len(failures) == 0
    detail = f"spot-checked {len(indices)} rows, {len(failures)} failures"
    if failures:
        detail += f"\n         first failure: {failures[0]}"
    return _check("Prompt structure", passed, detail)


def check_gt_parses(train_df: pd.DataFrame, n_spot: int = 200) -> bool:
    """Check 8: GT data in reward_model.ground_truth parses as JSON."""
    rng = random.Random(42)
    indices = rng.sample(range(len(train_df)), min(n_spot, len(train_df)))

    failures = []
    for idx in indices:
        row = train_df.iloc[idx]
        try:
            gt_raw = row["reward_model"]["ground_truth"]
            gt = json.loads(gt_raw)
            # Verify expected keys
            task_type = row["extra_info"].get("task_type", "grounding")
            if task_type == "grounding":
                if "boxes_1000" not in gt or "num_pairs" not in gt:
                    failures.append(f"row {idx}: grounding GT missing keys, got {list(gt.keys())}")
            else:
                if "response" not in gt:
                    failures.append(f"row {idx}: referring GT missing 'response' key")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            failures.append(f"row {idx}: {e}")

    passed = len(failures) == 0
    detail = f"spot-checked {len(indices)} rows, {len(failures)} failures"
    if failures:
        detail += f"\n         first failure: {failures[0]}"
    return _check("GT data parses as JSON", passed, detail)


def check_image_paths(train_df: pd.DataFrame, n_spot: int = 100) -> bool:
    """Check 9: Image paths in extra_info.images exist on disk."""
    rng = random.Random(42)
    indices = rng.sample(range(len(train_df)), min(n_spot, len(train_df)))

    missing = []
    for idx in indices:
        row = train_df.iloc[idx]
        try:
            images = row["extra_info"].get("images", [])
            for img_path in images:
                if not os.path.exists(img_path):
                    missing.append(img_path)
                    break  # one missing per row is enough
        except (KeyError, TypeError):
            pass

    passed = len(missing) == 0
    detail = f"spot-checked {len(indices)} rows, {len(missing)} missing image paths"
    if missing:
        detail += f"\n         first missing: {missing[0]}"
    return _check("Image paths exist on disk", passed, detail)


def check_no_nulls(train_df: pd.DataFrame) -> bool:
    """Check 10: No NaN/null values in critical fields."""
    critical = ["data_source", "ability"]
    failures = []

    for col in critical:
        if col in train_df.columns:
            null_count = train_df[col].isna().sum()
            if null_count > 0:
                failures.append(f"{col}: {null_count} nulls")

    # Also check SDS in extra_info
    try:
        sds_vals = [row.get("spatial_difficulty_score") for row in train_df["extra_info"]]
        null_sds = sum(1 for v in sds_vals if v is None or (isinstance(v, float) and np.isnan(v)))
        if null_sds > 0:
            failures.append(f"extra_info.spatial_difficulty_score: {null_sds} nulls")
    except Exception as e:
        failures.append(f"SDS null check error: {e}")

    passed = len(failures) == 0
    detail = f"critical columns checked: {critical + ['extra_info.spatial_difficulty_score']}"
    if failures:
        detail += f"\n         failures: {failures}"
    return _check("No null values in critical fields", passed, detail)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    parquet_dir: str = "data/hoi",
    val_size: int = 500,
    n_spot: int = 100,
) -> None:
    print(f"Verifying parquets in: {parquet_dir}")
    print("=" * 60)

    results: list[bool] = []

    # Check 1: Files exist
    ok, train_df, val_df = check_files_exist(parquet_dir)
    results.append(ok)

    if train_df.empty:
        print("\nERROR: Cannot continue — train parquet not loaded.")
        sys.exit(1)

    # Check 2: Row counts
    results.append(check_row_counts(train_df, val_df, val_size))

    # Check 3: Required columns
    results.append(check_required_columns(train_df))

    # Check 4: SDS distribution
    results.append(check_sds_distribution(train_df))

    # Check 5: Task distribution
    results.append(check_task_distribution(train_df))

    # Check 6: SDS in [0, 1]
    results.append(check_sds_in_range(train_df))

    # Check 7: Prompt structure
    results.append(check_prompt_structure(train_df, n_spot))

    # Check 8: GT parses as JSON
    results.append(check_gt_parses(train_df, n_spot * 2))

    # Check 9: Image paths exist
    results.append(check_image_paths(train_df, n_spot))

    # Check 10: No nulls
    results.append(check_no_nulls(train_df))

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print("=" * 60)
    print(f"RESULT: {n_pass}/{n_total} checks passed")

    if n_pass == n_total:
        print("ALL CHECKS PASSED ✓")
        sys.exit(0)
    else:
        print(f"{n_total - n_pass} CHECKS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify HOI parquet files for RL training")
    parser.add_argument("--parquet_dir", default="data/hoi")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--n_spot", type=int, default=100, help="Number of rows to spot-check")
    args = parser.parse_args()
    main(parquet_dir=args.parquet_dir, val_size=args.val_size, n_spot=args.n_spot)

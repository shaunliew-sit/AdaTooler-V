"""Resample the HOI GRPO train parquet by grounding proposal-anchor quality (s_ref).

Motivation (SAHA-CF tool-collapse analysis, 2026-06-18): 57.6% of grounding
training samples have a PERFECT proposal anchor (s_ref=1.0) on which the zoom
tool has zero upside (gain capped at 0, can only hurt). The gradient is
dominated by "tool not needed" cases, so the counterfactual tool reward gets a
weak "keep the tool" signal and tool use collapses. This script rebalances the
*existing* parquet (no new data) toward the tool-opportunity band (s_ref<=0.5),
keeping enough perfect-anchor cases that the model still learns when NOT to zoom.

Strategy (decided 2026-06-18):
  * Grounding: keep ALL non-perfect rows; down-sample perfect (s_ref==1.0) rows
    so perfect becomes PERFECT_TARGET_FRAC of grounding (MODERATE = 0.30).
  * Referring: cannot be profiled offline (s_ref = per-uid no-tool mean at train
    time), so down-sample uniformly to preserve the original grounding:referring
    row ratio (prevents referring dominating the rebalanced batch).
  * Deterministic (fixed seed). Writes a NEW file; never overwrites the source.
  * Verifies round-trip integrity: the nested `extra_info["proposal_anchor"]`
    struct must survive to_parquet or s_ref silently becomes 0.0 at train time.

Usage:
    python examples/data_preprocess/hoi/resample_grpo_by_sref.py \
        --src data/train.parquet --dst data/train_resampled_moderate.parquet \
        --perfect_frac 0.30 --seed 42
"""
import argparse
import json

import numpy as np
import pandas as pd

from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding


def _denumpy(o):
    """Deep-convert nested numpy (pandas/pyarrow read artifact) to native types."""
    if isinstance(o, np.ndarray):
        return [_denumpy(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_denumpy(x) for x in o]
    if isinstance(o, dict):
        return {k: _denumpy(v) for k, v in o.items()}
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _extra_info(row):
    e = row["extra_info"]
    return e if isinstance(e, dict) else json.loads(e)


def _task_type(e):
    return (e if isinstance(e, dict) else json.loads(e)).get("task_type")


def _grounding_sref(row):
    e = _extra_info(row)
    pa = _denumpy(e.get("proposal_anchor"))
    gtraw = row["reward_model"]["ground_truth"]
    gt = json.loads(gtraw) if isinstance(gtraw, str) else _denumpy(gtraw)
    try:
        return float(compute_sref_grounding(pa, gt))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train.parquet")
    ap.add_argument("--dst", default="data/train_resampled_moderate.parquet")
    ap.add_argument("--perfect_frac", type=float, default=0.30,
                    help="target fraction of grounding rows that are perfect-anchor (s_ref==1.0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.src)
    tt = df["extra_info"].apply(_task_type)
    g_idx = df.index[tt == "grounding"].to_numpy()
    r_idx = df.index[tt == "referring"].to_numpy()
    other_idx = df.index[~tt.isin(["grounding", "referring"])].to_numpy()
    print(f"source rows={len(df)}  grounding={len(g_idx)}  referring={len(r_idx)}  other={len(other_idx)}")
    orig_ratio = len(g_idx) / len(r_idx) if len(r_idx) else float("inf")
    print(f"original grounding:referring ratio = {orig_ratio:.4f}")

    # --- grounding: keep all non-perfect, down-sample perfect ---
    srefs = np.array([_grounding_sref(df.loc[i]) for i in g_idx])
    perfect_mask = np.abs(srefs - 1.0) < 1e-6
    perfect_idx = g_idx[perfect_mask]
    nonperfect_idx = g_idx[~perfect_mask]
    n_nonperfect = len(nonperfect_idx)
    # perfect / (perfect + nonperfect) = perfect_frac  =>  perfect = frac/(1-frac)*nonperfect
    keep_perfect = int(round(args.perfect_frac / (1.0 - args.perfect_frac) * n_nonperfect))
    keep_perfect = min(keep_perfect, len(perfect_idx))
    perfect_keep = rng.choice(perfect_idx, size=keep_perfect, replace=False)
    g_keep = np.concatenate([nonperfect_idx, perfect_keep])
    print(f"grounding: nonperfect kept={n_nonperfect}  perfect kept={keep_perfect}/{len(perfect_idx)}  -> grounding total={len(g_keep)}")

    # --- referring: down-sample to preserve original task ratio ---
    target_referring = int(round(len(g_keep) / orig_ratio)) if orig_ratio not in (0, float("inf")) else len(r_idx)
    target_referring = min(target_referring, len(r_idx))
    r_keep = rng.choice(r_idx, size=target_referring, replace=False)
    print(f"referring: kept={len(r_keep)}/{len(r_idx)}  (preserves ratio {len(g_keep)/len(r_keep):.4f})")

    keep_idx = np.concatenate([g_keep, r_keep, other_idx])
    rng.shuffle(keep_idx)
    out = df.loc[keep_idx].reset_index(drop=True)
    print(f"\nresampled total rows={len(out)}  (was {len(df)})")

    out.to_parquet(args.dst, index=False)
    print(f"wrote {args.dst}")

    # --- round-trip integrity: proposal_anchor must survive + s_ref must match ---
    rt = pd.read_parquet(args.dst)
    rtt = rt["extra_info"].apply(_task_type)
    rg = rt.index[rtt == "grounding"].to_numpy()
    rt_srefs = np.array([_grounding_sref(rt.loc[i]) for i in rg])
    nz = int((rt_srefs > 1e-6).sum())
    print(f"\n[round-trip] grounding rows={len(rg)}  s_ref>0 after reload={nz} ({100*nz/len(rg):.1f}%)  mean={rt_srefs.mean():.3f}")
    if nz == 0:
        raise SystemExit("FATAL: all s_ref==0 after round-trip -> proposal_anchor struct did NOT survive to_parquet")
    print("[round-trip] proposal_anchor survived; s_ref intact.")
    print("\nresampled grounding s_ref mix:")
    for lbl, c in [("==1.0", int((np.abs(rt_srefs - 1) < 1e-6).sum())),
                   ("==0.5", int((np.abs(rt_srefs - 0.5) < 1e-6).sum())),
                   ("<=0.5", int((rt_srefs < 0.5 + 1e-6).sum())),
                   ("==0", int((rt_srefs <= 1e-6).sum()))]:
        print(f"  {lbl:6s}: {c:6d} ({100*c/len(rg):4.1f}%)")


if __name__ == "__main__":
    main()

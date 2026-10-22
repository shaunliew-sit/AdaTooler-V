"""Backup hard-heavy resample of the HOI GRPO train parquet.

Goal (2026-06-19, prof feedback): raise the HARD (weak-proposal) fraction so the
counterfactual tool signal is denser. There are NO hard cases outside the data —
all non-perfect grounding rows are already present — so "more hard" = keep ALL
hard and DOWN-SAMPLE easy + referring (clean, no replication, no overfit). Optional
hard replication via --hard_replicate for an even denser variant.

Difficulty (same s_ref the reward uses): HARD = grounding with s_ref<0.5,
EASY = grounding with s_ref>=0.5. Referring has no offline s_ref -> treated as one bucket.

Defaults yield ~40% hard / ~40% easy-grounding / ~20% referring:
  * keep ALL hard grounding rows (optionally replicate x N)
  * easy grounding down-sampled to easy_to_hard_ratio * (kept hard)
  * referring down-sampled to referring_frac of the final total

Deterministic (seed). Writes a NEW file; round-trip-checks proposal_anchor survival.

Usage:
  python examples/data_preprocess/hoi/resample_grpo_hard_heavy.py \
    --src data/train.parquet --dst data/train_resampled_hard_heavy.parquet \
    --easy_to_hard_ratio 1.0 --referring_frac 0.20 --hard_replicate 1 --seed 42
"""
import argparse
import json

import numpy as np
import pandas as pd

from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding


def _denumpy(o):
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


def _task_type(e):
    return (e if isinstance(e, dict) else json.loads(e)).get("task_type")


def _grounding_sref(row):
    e = row["extra_info"]
    e = e if isinstance(e, dict) else json.loads(e)
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
    ap.add_argument("--dst", default="data/train_resampled_hard_heavy.parquet")
    ap.add_argument("--hard_thr", type=float, default=0.5, help="s_ref < thr => HARD")
    ap.add_argument("--easy_to_hard_ratio", type=float, default=1.0,
                    help="kept easy-grounding rows = ratio * kept-hard rows")
    ap.add_argument("--referring_frac", type=float, default=0.20,
                    help="referring rows as a fraction of the final total")
    ap.add_argument("--hard_replicate", type=int, default=1,
                    help="replicate each kept hard row N times (1 = no replication)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.src)
    tt = df["extra_info"].apply(_task_type)
    g_idx = df.index[tt == "grounding"].to_numpy()
    r_idx = df.index[tt == "referring"].to_numpy()
    other_idx = df.index[~tt.isin(["grounding", "referring"])].to_numpy()
    print(f"source rows={len(df)}  grounding={len(g_idx)}  referring={len(r_idx)}  other={len(other_idx)}")

    srefs = np.array([_grounding_sref(df.loc[i]) for i in g_idx])
    hard_mask = srefs < args.hard_thr
    hard_idx = g_idx[hard_mask]
    easy_idx = g_idx[~hard_mask]
    print(f"grounding HARD(s_ref<{args.hard_thr})={len(hard_idx)}  EASY={len(easy_idx)}")

    # keep all hard (optionally replicate)
    hard_keep = np.repeat(hard_idx, args.hard_replicate)
    # easy down-sampled to ratio * kept-hard
    n_easy_target = min(len(easy_idx), int(round(args.easy_to_hard_ratio * len(hard_keep))))
    easy_keep = rng.choice(easy_idx, size=n_easy_target, replace=False)
    g_keep = np.concatenate([hard_keep, easy_keep])

    # referring: make it referring_frac of the final total.
    # total = grounding + referring; referring = frac*total => referring = frac/(1-frac)*grounding
    n_ref_target = int(round(args.referring_frac / (1.0 - args.referring_frac) * len(g_keep)))
    n_ref_target = min(n_ref_target, len(r_idx))
    r_keep = rng.choice(r_idx, size=n_ref_target, replace=False)

    keep_idx = np.concatenate([g_keep, r_keep, other_idx])
    rng.shuffle(keep_idx)
    out = df.loc[keep_idx].reset_index(drop=True)
    n = len(out)
    print(f"\nresampled total rows={n} (was {len(df)})")
    print(f"  HARD grounding : {len(hard_keep):6d}  ({100*len(hard_keep)/n:.1f}% of batch)")
    print(f"  EASY grounding : {len(easy_keep):6d}  ({100*len(easy_keep)/n:.1f}% of batch)")
    print(f"  referring      : {len(r_keep):6d}  ({100*len(r_keep)/n:.1f}% of batch)")
    print(f"  within grounding hard:easy = {len(hard_keep)}:{len(easy_keep)} "
          f"({100*len(hard_keep)/(len(hard_keep)+len(easy_keep)):.0f}% hard)")

    out.to_parquet(args.dst, index=False)
    print(f"wrote {args.dst}")

    # round-trip integrity
    rt = pd.read_parquet(args.dst)
    rtt = rt["extra_info"].apply(_task_type)
    rg = rt.index[rtt == "grounding"].to_numpy()
    rt_srefs = np.array([_grounding_sref(rt.loc[i]) for i in rg])
    nz = int((rt_srefs > 1e-6).sum())
    print(f"[round-trip] grounding={len(rg)}  s_ref>0 after reload={nz} ({100*nz/len(rg):.1f}%)  mean={rt_srefs.mean():.3f}")
    if nz == 0:
        raise SystemExit("FATAL: all s_ref==0 after round-trip -> proposal_anchor did NOT survive")
    print("[round-trip] proposal_anchor survived; s_ref intact.")


if __name__ == "__main__":
    main()

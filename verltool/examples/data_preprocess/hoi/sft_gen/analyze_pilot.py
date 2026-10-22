"""Joint analysis of a pilot dataset: tool-use rate split by task_type and s_ref tier.

Tests the hypothesis "low tool-rate is driven by EASY cases, hard cases DO zoom". Maps each
kept trace's verbatim human prompt back to the source parquet row to recompute s_ref + task.

Usage: python examples/data_preprocess/hoi/sft_gen/analyze_pilot.py <pilot_dir> [--src data/train.parquet]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curate  # noqa


def den(o):
    if isinstance(o, np.ndarray):
        return [den(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [den(x) for x in o]
    if isinstance(o, dict):
        return {k: den(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def tier(s):
    if s is None:
        return "referring"
    if s <= 1e-6:
        return "grounding s_ref=0 (HARD)"
    if s < 0.5 - 1e-6:
        return "grounding s_ref<0.5 (HARD)"
    if abs(s - 0.5) < 1e-6:
        return "grounding s_ref=0.5 (HARD/MID)"
    if s < 1 - 1e-6:
        return "grounding 0.5<s_ref<1 (easyish)"
    return "grounding s_ref=1.0 (EASY)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir")
    ap.add_argument("--src", default="data/train.parquet")
    args = ap.parse_args()

    data = json.load(open(Path(args.pilot_dir) / "dataset.json"))
    print(f"kept samples: {len(data)}")

    df = pd.read_parquet(args.src)
    # map verbatim user prompt -> row index
    def user_of(prompt):
        for m in den(prompt):
            if m.get("role") == "user":
                return m["content"]
        return None
    umap = {}
    for i in range(len(df)):
        u = user_of(df.iloc[i]["prompt"])
        if u is not None:
            umap.setdefault(u, i)

    from collections import defaultdict
    cell = defaultdict(lambda: [0, 0])  # tier -> [tool, no_tool]
    unmatched = 0
    for s in data:
        human = s["conversations"][0]["value"]
        n_tool = len(s["images"]) - 1  # zoom_in count; tool if >=1 OR a zoom_out turn present
        used_tool = any(t["from"] == "observation" for t in s["conversations"])
        ridx = umap.get(human)
        if ridx is None:
            unmatched += 1
            continue
        row = df.iloc[ridx]
        ei = den(row["extra_info"])
        tt = ei.get("task_type")
        if tt == "grounding":
            gtraw = row["reward_model"]["ground_truth"]
            gt = json.loads(gtraw) if isinstance(gtraw, str) else den(gtraw)
            s_ref = curate.grounding_sref(den(ei.get("proposal_anchor")) or [], gt)
        else:
            s_ref = None
        t = tier(s_ref)
        cell[t][0 if used_tool else 1] += 1

    print(f"unmatched (prompt not found in parquet): {unmatched}\n")
    order = ["referring", "grounding s_ref=1.0 (EASY)", "grounding 0.5<s_ref<1 (easyish)",
             "grounding s_ref=0.5 (HARD/MID)", "grounding s_ref<0.5 (HARD)", "grounding s_ref=0 (HARD)"]
    print(f"{'tier':38s} {'tool':>5s} {'no_tool':>8s} {'tool%':>7s}")
    print("-" * 62)
    for t in order:
        if t not in cell:
            continue
        tool, no = cell[t]
        tot = tool + no
        print(f"{t:38s} {tool:5d} {no:8d} {100*tool/tot:6.0f}%")
    # hard aggregate
    hard = [v for k, v in cell.items() if "HARD" in k]
    ht, hn = sum(x[0] for x in hard), sum(x[1] for x in hard)
    if ht + hn:
        print("-" * 62)
        print(f"{'ALL HARD grounding (s_ref<=0.5)':38s} {ht:5d} {hn:8d} {100*ht/(ht+hn):6.0f}%")


if __name__ == "__main__":
    main()

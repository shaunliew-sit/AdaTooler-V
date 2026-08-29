"""Visualize the tool-use-collapse signal over training steps.

Reads the per-step rollout dumps that verl writes to
``trainer.rollout_data_dir`` (the train script sets this to
``verl_step_records/<run_name>/<step>.jsonl``) and prints a per-step table:

    step   N   tool%   acc    acc_tool  acc_notool  R_tool|tool  s_ref|g  zin  zout

How to read it:
  * COLLAPSE  = tool% trends toward 0 over steps (model stops using tools).
  * FIX WORKS = tool% stays healthy while acc rises, AND acc_tool >= acc_notool
                with R_tool|tool > 0 (tools earn reward where they genuinely
                beat trusting the proposal). The counterfactual reward should
                keep tool use alive exactly on the rows where it helps.

Usage:
    python examples/train/hoi/watch_tool_use.py verl_step_records/<run_name>
    # live:
    watch -n 30 'python examples/train/hoi/watch_tool_use.py verl_step_records/<run_name>'
    # inspect the GRPO group sampling for one step (proof of the tool-collapse fix):
    python examples/train/hoi/watch_tool_use.py verl_step_records/<run_name> --groups <step>
"""
import glob
import json
import os
import sys


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return None

    def col(k):
        return [r.get(k) for r in rows if k in r]

    i_tool = col("i_tool")
    acc = col("accuracy")
    is_g = col("is_grounding")
    r_tool = col("r_tool")
    s_ref = col("s_ref")
    z_in, z_out = col("n_zoom_in"), col("n_zoom_out")

    tool_idx = [i for i, t in enumerate(i_tool) if t and t > 0.5]
    notool_idx = [i for i, t in enumerate(i_tool) if not (t and t > 0.5)]
    g_idx = [i for i, g in enumerate(is_g) if g and g > 0.5]

    return {
        "N": len(rows),
        "tool%": 100.0 * _mean(i_tool),
        "acc": _mean(acc),
        "acc_tool": _mean([acc[i] for i in tool_idx]) if acc else float("nan"),
        "acc_notool": _mean([acc[i] for i in notool_idx]) if acc else float("nan"),
        "R_tool|tool": _mean([r_tool[i] for i in tool_idx]) if r_tool else float("nan"),
        "s_ref|g": _mean([s_ref[i] for i in g_idx]) if s_ref else float("nan"),
        "zin": _mean(z_in),
        "zout": _mean(z_out),
    }


def show_groups(path, max_groups=12):
    """Per-GRPO-group view: group rollouts by uid and show, within each group,
    the tool vs no-tool rollouts with their reward and advantage. Proves the
    group-sampling mechanism — collapse = tool rollouts don't out-advantage their
    own siblings. Needs the enriched dump (uid + seq_advantage)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if "uid" not in rows[0] or "seq_advantage" not in rows[0]:
        print("This dump lacks uid/seq_advantage — re-run training after the dump "
              "enrichment (ray_trainer._log_rollout_data).")
        return
    groups = {}
    for r in rows:
        groups.setdefault(r["uid"], []).append(r)
    # prioritize MIXED groups (both tool and no-tool present) — the informative ones
    mixed = [(u, g) for u, g in groups.items()
             if any(x.get("i_tool", 0) for x in g) and any(not x.get("i_tool", 0) for x in g)]
    mixed.sort(key=lambda ug: -_mean([x.get("is_grounding", 0) for x in ug[1]]))  # grounding first
    print(f"{len(groups)} groups, {len(mixed)} mixed (tool & no-tool). Showing up to {max_groups} mixed:\n")
    for u, g in mixed[:max_groups]:
        adv_t = _mean([x["seq_advantage"] for x in g if x.get("i_tool", 0)])
        adv_n = _mean([x["seq_advantage"] for x in g if not x.get("i_tool", 0)])
        won = "TOOL WINS" if adv_t > adv_n else "tool loses"
        sref = _mean([x.get("s_ref") for x in g if x.get("is_grounding", 0)])
        print(f"uid={str(u)[:18]:18}  s_ref|g={sref:.2f}  adv_tool={adv_t:+.3f} vs adv_notool={adv_n:+.3f}  -> {won}")
        for x in sorted(g, key=lambda x: -x.get("i_tool", 0)):
            kind = "TOOL " if x.get("i_tool", 0) else "noTool"
            print(f"    {kind} acc={x.get('accuracy', float('nan')):.0f} "
                  f"score={x.get('score', float('nan')):+.3f} "
                  f"R_tool={x.get('r_tool', 0):+.3f} adv={x.get('seq_advantage', float('nan')):+.3f} "
                  f"zin={x.get('n_zoom_in', 0):.0f} zout={x.get('n_zoom_out', 0):.0f}")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1]
    if "--groups" in sys.argv:
        step = sys.argv[sys.argv.index("--groups") + 1]
        path = os.path.join(run_dir, f"{step}.jsonl")
        if not os.path.exists(path):
            print(f"No dump at {path}")
            sys.exit(1)
        show_groups(path)
        sys.exit(0)
    files = sorted(
        glob.glob(os.path.join(run_dir, "*.jsonl")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else 0,
    )
    if not files:
        print(f"No *.jsonl rollout dumps under {run_dir} yet. "
              f"(trainer.rollout_data_dir writes one per step.)")
        sys.exit(0)

    hdr = f"{'step':>6}{'N':>6}{'tool%':>8}{'acc':>8}{'acc_tool':>10}{'acc_notool':>12}{'R_tool|tool':>13}{'s_ref|g':>9}{'zin':>6}{'zout':>6}"
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        step = os.path.splitext(os.path.basename(f))[0]
        s = summarize(f)
        if not s:
            continue
        print(f"{step:>6}{s['N']:>6}{s['tool%']:>8.1f}{s['acc']:>8.3f}"
              f"{s['acc_tool']:>10.3f}{s['acc_notool']:>12.3f}{s['R_tool|tool']:>13.3f}"
              f"{s['s_ref|g']:>9.3f}{s['zin']:>6.2f}{s['zout']:>6.2f}")


if __name__ == "__main__":
    main()

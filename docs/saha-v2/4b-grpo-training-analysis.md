# 4B SAHA-CF GRPO Training Analysis — eval now vs keep training?

## ✅ CODEX-VERIFIED CONCLUSION (2026-06-25, ran the verification script independently)
Verified numbers (pooled windows):
- **4B (steps 1→164): overall accuracy PLATEAUED** — 0.745 / 0.721 / 0.738 across thirds; **reward_std healthy ~0.38** (not collapsed). Tool policy selective from the start (tr_hard 0.47→0.55 vs tr_easy ~0.19) but the within-group signal is **weak and DECLINING**: group_adv_gap_hard 0.176→0.091→0.066, wins_hard ~0.37–0.40 (<0.5).
- **8B emergence CORRECTED: the decisive within-group signal fired at ~step 600–700, NOT step 205.** 8B group_adv_gap_hard by 100-step window: 0.02/0.09/0.08/0.07/−0.03/−0.03 (steps 1–600) → **0.63 (601–700)**, 0.57, 0.28, 0.79. wins_hard crosses 0.5 only at 601+. (Supersedes the "signal by step 205" claim in [[saha-cf-collapse-resolved]] — that was wrong.)
- **CRUCIAL: overall accuracy is FLAT across the ENTIRE 8B run** (0.784 at steps 1–100 → 0.792 at 901–1000, +0.008). GRPO shaped tool-*selectivity*, not overall accuracy. So waiting for the group signal does NOT materially raise eval-AR.
- **Claim "window-best = step_140" REFUTED** — step 140 has no hard groups (nan); single-step gaps are pure noise (135=0.49, 139=1.10, 142=−0.95). Do not cherry-pick a step.

**VERDICT for the user's question (eval now vs wait):** For a **scale-comparison eval-AR number, eval now** — overall acc is plateaued and the 8B proves overall acc barely moves over 1000 steps, so step ~160 ≈ step ~600 on eval AR. Use the **latest stable checkpoint (global_step_160)**, not a cherry-picked step. More training only buys the within-group hard-tool *selectivity* signal (emerged ~600 in the 8B, may never at 4B/half-params), which is a training-dynamics nicety that barely moves eval AR. Only keep training if you specifically want the 4B to replicate the 8B's late selectivity-emergence story — and that's optional, not blocking, given the 5 Jul deadline and the 4B being a scale-comparison point.

---

Date: 2026-06-25. Decision target: `checkpoints/SAHA-CF/SAHA-CF-a0.6-runB-4b-avg2`
(Qwen3-VL-4B + avg-IoU CTG reward, α=0.6). Records to step ~162 (STILL TRAINING).
8B reference: `SAHA-CF-a0.6-newsft-resampled` (deployed `global_step_1000`).

All numbers reconstructed per-step from `verl_step_records/<run>/<step>.jsonl` by
replicating `metric_util.py::compute_saha_cf_metrics`. **Per-step N is tiny (4B=96,
8B=24 rollouts), so single steps are pure noise — everything below is POOLED over
windows** (4B window=1920–3840 rollouts; 8B window=960).

## 1. 4B trend (pooled, window=40 steps = 3840 rollouts)

| window | acc | reward_std | tool_rate | tr_hard | tr_easy | acc_tool−notool | gain_hard\|tool | grp_adv_gap_hard | grp_wins_hard |
|--------|-----|-----------|-----------|---------|---------|------------------|------------------|------------------|----------------|
| 1–40   | 0.745 | 0.385 | 0.168 | 0.475 | 0.214 | −0.126 | 0.121 | 0.185 | 0.389 |
| 41–80  | 0.720 | 0.382 | 0.168 | 0.541 | 0.183 | −0.195 | 0.162 | 0.012 | 0.263 |
| 81–120 | 0.728 | 0.383 | 0.162 | 0.519 | 0.175 | −0.234 | 0.120 | 0.166 | 0.515 |
| 121–160| 0.742 | 0.374 | 0.159 | 0.530 | 0.195 | −0.145 | 0.176 | 0.196 | 0.333 |
| full 1–162 | 0.735 | 0.381 | 0.164 | 0.511 | 0.191 | −0.176 | 0.145 | 0.130 | 0.380 |

Readout:
- **Overall reward/accuracy: PLATEAUED, not rising.** acc oscillates 0.72–0.745 across
  the whole run with zero upward drift over 160 steps. No learning curve.
- **reward_std HEALTHY** (0.37–0.39 every window, never near the 0.05 collapse floor).
  The GRPO signal is alive; this is not a dead/collapsed run.
- **Tool policy is selective FROM THE START and stays flat.** tool_rate_hard ≈ 0.51 vs
  tool_rate_easy ≈ 0.19 (≈2.7× concentration on weak-proposal cases) in *every* window —
  inherited from SFT, not learned by GRPO. It does not strengthen with steps.
- **gain_hard|tool is positive throughout (~0.12–0.18)** — when the 4B zooms on a hard
  case it does raise the target-IoU, exactly the design intent.
- **BUT the decisive within-group anti-collapse signal is WEAK and FLAT.**
  `group_adv_gap_hard` ≈ 0.13 (full pool) hovering near 0, and
  `group_tool_wins_frac_hard` ≈ 0.38 — **below 0.5**, i.e. on mixed hard groups the tool
  rollout out-advantages its no-tool siblings *less than half the time*. No upward trend.
- `acc_tool_minus_notool` stays **negative** (~−0.18): tool rollouts still don't beat
  no-tool on accuracy overall (the override/capability ceiling from the 8B diagnosis).

## 2. 8B vs 4B at matched steps — the yardstick

8B pooled (window=40), early vs late:

| 8B window | acc | tr_hard | tr_easy | gain_hard\|tool | grp_adv_gap_hard | grp_wins_hard |
|-----------|-----|---------|---------|------------------|------------------|----------------|
| 1–40   | 0.773 | 0.359 | 0.149 | 0.190 | 0.417 | 0.444 |
| 81–120 | 0.767 | 0.542 | 0.164 | 0.115 | −0.088 | 0.091 |
| 161–200| 0.776 | 0.479 | 0.171 | 0.108 | 0.157 | 0.455 |
| 401–440| 0.794 | 0.621 | 0.174 | 0.183 | 0.286 | 0.429 |
| 681–720| 0.769 | 0.750 | 0.203 | 0.382 | 0.566 | 0.667 |
| 881–920| 0.843 | 0.704 | 0.204 | 0.323 | 0.976 | 0.833 |
| 961–1000| 0.783 | 0.845 | 0.283 | 0.317 | 1.465 | 1.000 |

**8B at its matched early window (1–160) is statistically the same as the 4B's whole run:**
acc 0.786 vs 0.735, tr_hard 0.42 vs 0.51, group_adv_gap_hard 0.025 vs 0.130,
wins_hard 0.27 vs 0.38. **The 4B is tracking the 8B's early shape, not lagging or flat.**

The 8B's healthy/converged selective signal — `group_adv_gap_hard` climbing 0.3→1.5 and
`group_tool_wins_frac_hard` 0.5→1.0, gain_hard|tool 0.3+, tr_hard 0.7–0.85 — **only emerges
late, ~steps 600–1000**, NOT at step 205. (Prior "step 205" note was from a separate
eval-stratified analysis with a different metric; the within-group training signal here
does not firm up until ~600+.) The 8B needed ~4–6× more steps than 160 for the group
tournament to flip decisively positive.

## 3. VERDICT

**Keep training the 4B; do NOT eval `global_step_160` as the converged result yet — but
also do NOT expect a large payoff.**

Reasoning, grounded in the numbers:
1. The 4B is **not capability-dead**: reward_std is healthy (~0.38), selectivity is real
   (tr_hard 2.7× tr_easy), and per-rollout tool-gain on hard cases is positive (~0.15).
   So "signal never emerges → eval now" is **not** the right call — the signal exists, it
   is just immature.
2. The 4B at 160 steps sits exactly where the **8B sat at 160 steps**, and the 8B's
   decisive within-group anti-collapse signal (group_adv_gap_hard ≥ 0.5,
   wins_frac_hard ≥ 0.5) did **not** appear until ~step 600+. The 4B has simply **not been
   trained long enough** for the group tournament to flip (currently wins_hard 0.38 < 0.5,
   gap ≈ 0.13, flat).
3. Therefore eval at 160 would capture the run *before* its key metric matures and would
   undersell it.

**Recommendation:**
- **Continue to ~step 400–600**, mirroring the 8B trajectory where group_adv_gap_hard and
  wins_frac_hard first cross/stay above their early-noise band. Watch the live panel
  `saha_cf/group_adv_gap_hard` + `group_tool_wins_frac_hard` (window-pooled, not per-step):
  the GO signal is wins_frac_hard rising and *holding* ≥ 0.5 over a 40-step window, with
  gain_hard|tool ≥ 0.25. That is what "converged & selective" looked like on the 8B.
- **If by ~step 400 group_adv_gap_hard is still flat near 0 and acc has not moved off
  0.73**, declare the 4B capability-bound (the same override/derive-on-miss ceiling
  documented for the 8B, amplified by half the parameters) and eval the best checkpoint —
  in that case pick the window-best, currently around `global_step_140` (gain_hard|tool
  0.176, group_adv_gap_hard 0.196 — the strongest hard-tool window so far).
- **Do not eval `global_step_160` as final now.** If a *checkpoint must be evaluated
  immediately* for a sanity read, use `global_step_140` (best hard-tool window), not 160
  (step 161–162 pool shows acc spike 0.849 on only N=192 — noise, not signal).

### One-line answer
Keep training to ~step 400–600 (the 8B needed ~600 steps for its within-group hard signal
to firm up; the 4B at 160 is on the same early curve, healthy std, selective, but the
decisive `group_tool_wins_frac_hard` is still 0.38 < 0.5 and flat). Re-decide at ~400; if
still flat, it's capability-bound — eval `global_step_140`.

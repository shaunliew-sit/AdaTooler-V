# Run A — Reward-Metric Alignment (GRPO-only) — Design

**Date:** 2026-06-24
**Status:** Approved direction (user chose GRPO-only re-run, metric-align first); design pending review.
**Owner decision log:** α stays 0.6 (one variable); staged smoke → ~200-step probe → full → eval.
**Scope:** Grounding reward scorer only. No SFT regen. No new tools. No new reward terms.

---

## 1. Problem (verified diagnosis)

SAHA-CF (`global_step_1000`) has **lower grounding Average Recall on HARD cases** (proposal misses
GT, `s_ref < 0.5`) than the no-proposal base Qwen3-VL-8B: HICO 4.44 vs 5.16; SWIG 29.05 vs 38.32.

This is **not** a GRPO regression (GRPO ≥ SFT on every eval metric; the base→Ours drop is localized to
the SFT stage, which GRPO partially recovers) and **not** a scorer artifact (base AR reproduces
30.38/38.32 under functionally identical AR math). The decisive, Codex-verified root cause is a
**train/eval metric mismatch**:

| | pairing rule | thresholds | source |
|---|---|---|---|
| **Reward (train)** | `0.5·person_iou + 0.5·object_iou` (avg) | `{0.5, 0.75}` | `sds_grpo.py:75,154-156` (reused by `saha_cf.py`) |
| **Eval AR** | `min(person_iou, object_iou)` | `0.5…0.95` step 0.05 (10) | `eval_hico_ground_sftgrpo_qwen3vl.py:593-597,707` |

Consequence: a tool action that sharpens the already-better box of a pair (person .90→.96, object
stays .55) raises the **reward** (.5→1.0) but leaves **eval AR** (min .55) unchanged. The within-group
hard-case tool gains measured during training (Δacc +0.15…+0.26, late steps) therefore **cannot**
surface in eval AR-Hard. Full diagnosis: memory `saha-cf-hard-regression-diagnosis`.

Two further causes are out of scope for Run A: hard-regime starvation (→ Run B, hard-heavy curriculum)
and a residual capability ceiling on hard box derivation (needs crop-grounded SFT regen — not pursued).

## 2. Goal

Make the GRPO grounding reward optimize the **same** localization target the eval AR measures, so that
hard-case tool gains register. Exactly one change relative to the existing SAHA-CF run.

## 3. The change

All edits in `verl_tool/workers/reward_manager/saha_cf.py`. `sds_grpo.py` is **frozen** (the old
area-SDS baseline) — reuse its helpers by import only.

1. **New scorer** `compute_grounding_outcome_ar(pred_text, gt_data) -> float`:
   - Reuse `parse_grounding_answer` (parsing, not metric) and the existing GT-pair reconstruction from
     `gt_data["boxes_1000"]`/`num_pairs` (mirror `compute_grounding_outcome`).
   - New local greedy matcher: rank candidate (pred,gt) pairs by **`min(person_iou, object_iou)`** and
     gate at threshold on the same min — byte-faithful to `eval_hico_ground_sftgrpo_qwen3vl.py:593-627`.
   - Return **mean over 10 thresholds** `0.5,0.55,…,0.95` of `matched/len(gt_pairs)` (eval AR
     definition). Empty-GT edge case identical to `compute_grounding_outcome`.
2. **Route both grounding paths** through the selected scorer:
   - `r_outcome` for `task_type=="grounding"` (`saha_cf.py:191`).
   - `compute_sref_grounding` (`saha_cf.py:40-60` / call site `:200`) — **must** use the same scorer, or
     `s_final − s_ref` compares two different metrics and the counterfactual gain is meaningless.
   - Implement as a module-level `_GROUNDING_SCORER` selected once from env; `compute_sref_grounding`
     calls it too (pass the callable or read the same selector).
3. **Env switch** `SAHA_CF_GROUNDING_METRIC`: `minAR10` (new, **default for SAHA-CF**) or `avg2`
   (current behavior, for reproducibility / A-B). Mirror the existing `SAHA_CF_*` env-precedence pattern.

**Explicitly unchanged:** reward formula `R_format·(R_outcome + α·R_tool)`; α=0.6 (one knob);
`clip_lo=-0.5`, `clip_hi=1.0`; referring path (`compute_referring_outcome`, group-no-tool `s_ref`);
format reward; all `reward_extra_info` logging keys and the `saha_cf/*` metric panel.

**Constraint compliance:** `sds_grpo.py` untouched ✓ · one knob α ✓ · reward shape unchanged ✓ ·
SDS-v2 factors remain eval-only ✓ · no GT leakage (s_ref is train-only) ✓ · no new tool / no AXPO / no
SAN ✓.

## 4. Risk to manage

`min`-IoU over 10 thresholds (incl. 0.80–0.95) is **stricter** than avg-IoU over {0.5,0.75}, so
`R_outcome` shrinks in scale. This can (a) depress reward std toward the 0.05 collapse floor and
(b) unbalance the α=0.6 mix between `R_outcome` and `R_tool`. Per decision, **α stays 0.6** and this is
checked at the smoke/probe gate; α is only re-swept if the probe shows collapse or a dead tool signal.

## 5. Validation gate (each step gates the next)

1. **Unit test** — `compute_grounding_outcome_ar` reproduces the eval AR on a set of stored eval samples
   (compare to their `matches_per_threshold`-derived AR; must match exactly). At
   `examples/train/hoi/test_saha_cf_metric.py` (`verltool/tests/` is gitignored). DONE — 7/7 pass.
2. **Smoke** — extend `examples/train/hoi/smoke_test_saha_cf.py`: on a balanced subset confirm
   reward std > 0.05 (not collapsed) **and** the hard-group counterfactual gain (`s_final − s_ref`) is
   still positive under `minAR10`.
3. **Probe run (~150–250 steps)** from the same SFT checkpoint, same parquet (`train_resampled_moderate`),
   `reward_manager=SAHA-CF`, `SAHA_CF_GROUNDING_METRIC=minAR10`. Watch
   `saha_cf/group_adv_gap_hard`, `group_tool_wins_frac_hard`, `tool_rate{,_hard}`, `reward_std`, and the
   `[SAHA-CF rollout]` console line. **Pass = tool out-advantages its own no-tool siblings on low-`s_ref`
   groups under the new (eval-aligned) metric**, no collapse.
4. **Full run (~1000 steps)** only if the probe is healthy → re-eval all four tasks on the new
   checkpoint via the existing `run_{hico,swig}_{ground,action}_sftgrpo_eval.sh` (point at the new
   `global_step_*/actor/huggingface`).

## 6. Success criteria

- **Primary:** HICO grounding **AR-Hard ≥ 5.16** (matches/beats no-proposal base), without regressing
  All/Easy AR or the referring metrics vs the current step-1000 model.
- **Secondary:** SWIG grounding AR-Hard improves vs current 1.88 and Ours ≥ base+proposal on hard
  (beating the no-proposal base on SWIG is **not** expected — vocab + capability ceiling).
- **Negative-result value:** if AR-Hard does **not** move, that isolates the residual as capability/SFT-
  bound and justifies either shipping with honest framing or escalating to Run B / SFT regen.

## 7. Out of scope (explicit)

- Run B (hard-heavy curriculum, `train_resampled_hard_heavy.parquet`) — only if Run A helps; separate spec.
- SFT trace regen (needs recorded human decision per spec §10.0) — not triggered by this work.
- Any change to `sds_grpo.py`, the tool set, the reward shape, or referring scoring.

## 8. Paper-text follow-up (independent of the run)

Stale "4B" prose in `sec/4_experiment.tex` (lines ~16, 52, 80, 110) and the commented 4B `tbl_ablation`
block should be updated to 8B; tables are already 8B. Not blocking.

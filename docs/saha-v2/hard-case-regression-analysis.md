# SAHA-CF Hard-Case Regression — Diagnosis, Root Cause, and Fix

**Date:** 2026-06-24 · **Model:** Qwen3-VL-8B (SAHA-CF, ACCV submission) · **Status:** root cause confirmed (multi-agent + Codex-verified); fix (Run A) implemented and a corrected GRPO run is in progress.

---

## 1. Executive summary

The full SAHA model scored **worse than the no-proposal base Qwen3-VL-8B on HARD grounding** (HICO AR-Hard 4.44 < 5.16; SWIG 29.05 ≪ 38.32), contradicting the thesis that tool use should help the hardest cases. Investigation shows this is **not a GRPO regression** and **not a scoring artifact**. The decisive cause is a **train/eval metric mismatch**: the GRPO reward scored grounding with a *looser* localization metric than the eval Average Recall, which **over-credited tool actions by +0.21 on hard cases**. Training therefore "looked fixed" (hard tool-rate rose to 0.72, tool out-accuracied no-tool, advantage gaps strongly positive) while the eval — using the stricter metric — saw near-floor tool outputs and no hard-case improvement. The fix (**Run A**) aligns the reward to the eval metric; a corrected run is training now to test whether the remaining gap is learnable or a capability ceiling.

---

## 2. The observed problem

Grounding Average Recall (AR), stratified by detector-proposal difficulty `s_ref` (HARD = `s_ref < 0.5`, proposal misses the GT pair):

| Method | HICO AR-Hard | SWIG AR (all) | SWIG AR-Hard |
|---|---|---|---|
| Qwen3-VL-8B (no proposal, base) | **5.16** | **38.32** | 8.13 |
| Qwen3-VL-8B + proposal | 3.68 | 35.78 | 6.27 |
| **SAHA (ours, full)** | 4.44 | 29.05 | 1.88 |

Ours beats *proposal+SFT* on hard but trails the *no-proposal base*. The All/Easy buckets are fine (ours ≥ base on HICO All/Easy); the regression is concentrated in **hard grounding**, and the base→ours drop is localized to the **SFT stage** (ablation: SWIG base 38.32 → +proposal 35.78 → **+SFT 28.26** → +GRPO 29.05; GRPO partially *recovers*).

---

## 3. Root-cause ranking (verified)

| # | Cause | Verified? | GRPO-fixable? |
|---|---|---|---|
| 1 | **Train/eval metric mismatch** (reward looser than eval AR) | ✅ Codex-verified in code + numerically | ✅ (Run A) |
| 2 | Proposal over-trust (SFT copies the wrong proposal box ~78% on hard) | ✅ full-set | partial |
| 3 | Capability ceiling (model rarely derives a better box after zoom) | ✅ (near-floor hard AR) | ❌ (needs SFT regen) |
| 4 | "Hard regime starved" (curriculum) | ❌ **false alarm** — artifact of measuring with the wrong metric | n/a |
| 5 | Proposal conditioning is structurally net-negative on hard (base isn't anchored to a wrong proposal) | ✅ | inherent |

**Not** causes: GRPO is not broken (GRPO ≥ SFT on every eval metric, tools more); the base comparison is fair (base AR reproduces 30.38 / 38.32 under functionally identical AR math).

---

## 4. The metric mismatch (cause #1)

| | Pairing rule | IoU thresholds | Source |
|---|---|---|---|
| **Reward (train)** | `0.5·person_iou + 0.5·object_iou` (averaged) | {0.5, 0.75} | `verl_tool/workers/reward_manager/sds_grpo.py:75,154-156` (reused by `saha_cf.py`) |
| **Eval AR** | `min(person_iou, object_iou)` | 0.5…0.95 step 0.05 (10) | `hoi-benchmarks/eval_hico_ground_sftgrpo_qwen3vl.py:593-597,707` |

The reward optimizes a strictly **looser** target. Worked example (one pair): no-tool box `person=0.90, object=0.55`; tool refines the already-good person box `0.90→0.96`, object unchanged.
- Reward (`avg`): 0.5·0.96+0.5·0.55 = 0.755 ≥ 0.75 → outcome jumps **0.50 → 1.0**, tool earns `R_tool`.
- Eval AR (`min`): min stays 0.55 → AR **unchanged**.

The reward pays for polishing the easier box; the eval doesn't. Tool gains in training need not appear in eval AR-Hard.

---

## 5. How the previous run looked during training (the "looks fixed" trap)

Reconstructed from 1000 step-records of `SAHA-CF-a0.6-newsft-resampled` (old `avg2` reward):

| Steps | hard tool-rate | hard tool acc / notool acc | within-group adv gap | reward std |
|---|---|---|---|---|
| 1–100 | 0.40 | 0.41 / 0.32 | +0.02 | 0.37 |
| 301–400 | 0.54 | 0.47 / 0.44 | +0.07 | 0.39 |
| 701–800 | 0.47 | **0.68 / 0.42** | **+0.57** | 0.38 |
| 901–1000 | **0.72** | 0.52 / 0.37 | **+0.79** | 0.38 |

By every training-time signal this was a success: the policy learned to zoom selectively on hard cases, the tool out-accuracied no-tool late, the within-group advantage gap went strongly positive, and the reward never collapsed. This is why it was previously logged as *"tool-collapse resolved."*

---

## 6. The smoking gun — re-scoring the same rollouts both ways

The previous run's hard-grounding rollouts (steps 600–1000) re-scored under both metrics:

| HARD grounding | `avg2` (training optimized) | `minAR10` (eval measures) | inflation |
|---|---|---|---|
| **TOOL** outcome | 0.282 | 0.069 | **+0.213** |
| NOTOOL outcome | 0.164 | 0.047 | +0.117 |
| within-group (tool − notool) gap | **+0.095** | **+0.026** | — |

`avg2` inflated tool localization by **+0.21** on hard, and inflated **tool more than no-tool** (a zoom nails one box, leaves the other mediocre — `avg` rewards the good box, `min` does not). The "tools help hard" signal training chased (**+0.095**) shrank to **+0.026** under the eval metric, at **near-floor absolute** (0.069).

**Causal chain:** old reward over-credits hard zooms (+0.21) → policy drives hard tool-rate to 0.72 → those zoom outputs score near-floor on the strict eval AR → eval hard-AR doesn't improve → combined with proposal-over-trust, ours lands **below the no-proposal base**. Same model, two different rulers.

---

## 7. A corrected false alarm: the data was never starved

An initial hypothesis was that the GRPO mix starved the hard regime (~8.7% hard). That number was measured with the **old `avg2`** `s_ref`. Under the eval-aligned metric, `s_ref` labels shift dramatically:

| parquet | HARD % of grounding (`avg2`) | HARD % of grounding (**`minAR10`**) |
|---|---|---|
| source `train.parquet` | ~26% | **50%** |
| `train_resampled_moderate` (run + probe used this) | ~19% | **73%** (33% of all rows) |

The training data was already **hard-rich** under the metric that matters. So the curriculum was **not** changed — `train_resampled_moderate.parquet` is reused. (`s_ref` distribution is metric-dependent: stricter metric ⇒ proposals "miss" more often ⇒ more hard.)

---

## 8. The fix — Run A (reward-metric alignment)

Implemented in `verl_tool/workers/reward_manager/saha_cf.py` (the frozen `sds_grpo.py` baseline is untouched):
- New `compute_grounding_outcome_ar`: **`min`-IoU pairing, 10 thresholds 0.5–0.95** — byte-faithful to the eval AR.
- Routed through **both** `s_final` and `s_ref`, so the counterfactual gain `s_final − s_ref` is measured on one consistent metric.
- Selected via `SAHA_CF_GROUNDING_METRIC` (`minAR10` default | `avg2` frozen v1).
- Reward shape unchanged: `R_format·(R_outcome + α·R_tool)`, single knob α = 0.6.
- Unit test (`examples/train/hoi/test_saha_cf_metric.py`, 7/7) confirms it reproduces eval-AR logic and diverges from v1 exactly on the polished-person/weak-object case (v1 = 1.0 vs Run A = 0.1).

**Run config (in progress, GPU 3):** `minAR10` reward + batch 4→16 (more hard GRPO groups/step) + CPU offload dropped (H200 143 GB → ~92 s/step vs 131 s) + 1000 steps, on the unchanged `moderate` parquet, restarting from the SFT checkpoint. No SFT regen.

---

## 9. What is expected, and the decision criterion

Run A removes the +0.21 inflation, so any hard gain that now appears is **real and transfers to eval**. But the honest hard benefit is small (+0.026, near-floor), so GRPO may *suppress* hard tooling rather than grow it. The corrected run tests whether the residual is **learnable** or a **capability ceiling**.

**Decisive signal — `saha_cf/tool_rate_hard` over training:**
- **Rises** → GRPO is learning selective, genuinely useful hard tooling → eval hard-AR should finally move → strong result.
- **Decays toward 0** (as the batch-4 probe did, 0.33→0.24 over 69 steps) → capability-bound → the call is SFT trace regen (costly) or **ship with honest framing**: HICO All/Easy win + this metric-mismatch analysis as a contribution (a clean, publishable story either way).

---

## 10. Difficulty stratification is metric-dependent (consistency check)

The paper's HARD/EASY split uses `avg2` `s_ref`. Re-bucketing the *same* eval results under the eval-aligned `minAR10` `s_ref` (consistent with the reward and the eval AR):

- The HARD set **expands monotonically** — the `minAR10` HARD set is a strict **superset** of the `avg2` HARD set (`avg2`-HARD ⊆ `minAR10`-HARD), not a re-partition. It quadruples: HICO 11.2% → **49.7%** hard, SWIG 10.6% → **49.7%** (every `avg2`-HARD pair stays HARD; ~7,600 `avg2`-EASY pairs cross the 0.5 threshold into HARD because their `s_ref` drops under the stricter metric; **0** move the other way). Honest statement under the eval metric: **~half of grounding pairs are "hard"** (the proposal mis-localizes at least one box under min-IoU).

| bucket | `s_ref` metric | HICO base | HICO Full | Δ(Full−base) | SWIG base | SWIG Full | Δ |
|---|---|---|---|---|---|---|---|
| EASY | avg2 | 37.93 | 38.98 | +1.05 | 41.98 | 32.32 | −9.66 |
| EASY | **minAR10** | 55.28 | 60.89 | **+5.61** | 55.94 | 48.56 | −7.38 |
| HARD | avg2 | 5.16 | 4.44 | −0.72 | 7.91 | 1.88 | −6.03 |
| HARD | **minAR10** | 15.43 | 13.09 | **−2.34** | 20.72 | 9.56 | −11.16 |

**Conclusions (robust to the metric choice):**
1. The **hard-case regression (base > Full) is real under both metrics** — it is *not* an artifact of the `avg2` stratification; under the consistent metric it is in fact *larger* (HICO −0.72 → −2.34).
2. On **EASY**, Full's HICO win **grows** under the consistent metric (+1.05 → **+5.61**) — Full clearly helps where the proposal is good.
3. SWIG trails base in both buckets under both metrics (SFT proposal-dependence collapse + vocab ceiling).
4. **Recommendation:** either keep `avg2` (with a footnote) or switch the paper's stratification to `minAR10` for consistency with the reward/eval; the substantive story is unchanged, but `minAR10` is more defensible and reframes difficulty honestly (~50% hard). Built by `hoi-benchmarks/compute_sref_cache_minAR10.py` → `sref_cache_minAR10.json`.

## Appendix — reproduction

- **Eval results analyzed:** `/workspace/hoi-benchmarks/results-sftgrpo/{hico,swig}_{ground,action}_{grpo,sft}/`
- **Difficulty cache:** `/workspace/hoi-benchmarks/sref_cache.json` (HARD = bucket MISS/PARTIAL)
- **Previous run dynamics:** `verl_tool/.../verl_step_records/SAHA-CF-a0.6-newsft-resampled/*.jsonl` (1000 steps)
- **Reward code:** `verl_tool/workers/reward_manager/saha_cf.py` (`compute_grounding_outcome_ar`), frozen `sds_grpo.py`
- **Eval AR scorer:** `hoi-benchmarks/eval_hico_ground_sftgrpo_qwen3vl.py` (`pair_iou=min`, 10-threshold AR)
- **Design spec:** `docs/saha-v2/reward-metric-alignment-runA-design.md`
- **Note:** paper prose still says "4B" in places (`sec/4_experiment.tex` ~lines 16,52,80,110 + commented 4B `tbl_ablation`); the tables are 8B — update before submission.

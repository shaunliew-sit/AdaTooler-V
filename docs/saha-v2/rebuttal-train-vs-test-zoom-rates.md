# Rebuttal: training-time vs test-time zoom rates (SAHA-8B, PTG α=0.6)

Answers the reviewer point *"explain the difference between training-time and test-time
zoom rates (the number)"*. Every number below was recomputed from raw artifacts on this
box (2026-09-01); no number is carried over from a doc.

## Sources
- **Training:** `verltool/verl_step_records/SAHA-CF-a0.6-newsft-resampled/{1..1000}.jsonl`
  — 24,000 logged rollouts of the run that produced the paper's SAHA-8B
  (`global_step_1000`, `docs/saha-v2/checkpoint-inventory.md`).
  Per-rollout fields `i_tool`, `is_grounding`, `s_ref`, `n_zoom_in`.
- **Test:** `/workspace/hoi-benchmarks/results-sftgrpo/{hico,swig}_{ground,action}_{grpo,sft}_8b/`
  — grounding: `*_results_*.json` field `tool_calls`; referring: `*_per_triplet.json`
  field `tool_calls` (the top-level referring json only carries caption/image_id).
- **Hard/Easy strata:** `/workspace/hoi-benchmarks/sref_cache.json`
  (`MISS`/`PARTIAL`/`EASY`; HARD = MISS+PARTIAL = `s_ref < 0.5`, same threshold as the
  training metric `verl_tool/trainer/ppo/metric_util.py:98`). 0 key misses on all 4 sets.

## Test-time zoom rates (greedy, `temperature=0.0`, single pass, full test sets)

| Benchmark | N | zoom rate | HARD (`s_ref<0.5`) | EASY | share HARD/EASY |
|---|---|---|---|---|---|
| HICO grounding | 20,028 | **11.32 %** | **47.71 %** | **6.72 %** | 11.2 / 88.8 % |
| SWIG grounding | 19,333 | **16.79 %** | **45.18 %** | **13.42 %** | 10.6 / 89.4 % |
| HICO referring | 33,405 | **7.70 %** | — | — | — |
| SWIG referring | 19,680 | **3.24 %** | — | — | — |
| **All four** | **92,446** | **9.44 %** | | | |

SFT-8B counterpart (same protocol): HICO-G 9.66 % (hard 42.95 / easy 5.44),
SWIG-G 12.90 % (39.00 / 9.80), HICO-R 7.34 %, SWIG-R 2.20 %, all-four **7.91 %**.
GRPO sharpens the gate: hard 42.95 → 47.71 while easy stays ~6 %.

Exact recombination (the arithmetic to quote):
`0.4771 × 0.1123 + 0.0672 × 0.8877 = 0.1132` → the 11.3 % headline **is** the
47.7 / 6.7 pair reweighted by a test set that is 88.8 % easy.

## Training-time zoom rates (rollouts, `temperature=1.0`, `top_p=1.0`, n=6 per prompt)

| window | overall | grounding | referring | ground HARD | ground EASY | calls/query |
|---|---|---|---|---|---|---|
| steps 1–100 | 13.5 % | 20.9 % | 6.5 % | 39.9 % | 17.1 % | 0.158 |
| steps 401–500 | 15.0 % | 22.6 % | 9.3 % | 50.0 % | 17.0 % | 0.180 |
| steps 901–1000 | **19.1 %** | 30.2 % | 11.0 % | **71.7 %** | 21.2 % | 0.223 |
| last 50 steps | 22.1 % | 38.3 % | 11.1 % | 77.8 % | 27.0 % | 0.260 |
| whole run (1–1000) | 15.5 % | 23.8 % | 8.6 % | 51.3 % | 17.7 % | 0.180 |

(The `fig/training_selectivity.png` curve — hard 42.7 % → 65.2 % — is the smoothed
version of the hard column above; raw bin endpoints are 39.9 → 71.7.)

Training rollout composition: grounding 10,872 / 24,000 (45.3 %); of grounding,
HARD `s_ref<0.5` = 1,980 (**18.2 %**) and perfect anchor `s_ref = 1.0` = 3,138 (**28.9 %**).

## Why the two differ — three causes, none of them a policy change

1. **The split, not the policy.** Training runs on
   `data/train_resampled_moderate.parquet`, deliberately s_ref-rebalanced
   (`examples/data_preprocess/hoi/resample_grpo_by_sref.py`, seed 42): perfect-anchor
   grounding rows down-sampled 57.6 % → 30 % (measured in rollouts: 28.9 %) so gradient
   mass sits in the tool-opportunity band. Result: HARD is **18.2 %** of training
   grounding but only **11.2 %** of HICO test / 10.6 % of SWIG test, and the test suite
   is 88.8 % easy. A marginal rate is a mixture; the mixture weights differ by ~1.6×.
2. **The decoding.** Training zoom rates are measured on **exploratory GRPO rollouts**
   (`temperature=1.0`, `top_p=1.0`, 6 samples/prompt — the script's `n=6`); test is
   **greedy** (`temperature=0.0`, `eval_*_sftgrpo_qwen3vl.py:336,364`), one pass.
   Sampling inflates tool use mostly on easy queries (train easy 17.7–21.2 % vs
   test easy 6.7 %), because a zoom only has to win once in six draws to be logged.
3. **The denominator.** The training-dynamics figure plots the **per-stratum HARD**
   rate; Tab. 4 / the efficiency table report the **marginal over the whole test set**.
   Like-for-like on the hard stratum the two agree in kind: 71.7 % (sampled, hard-heavy
   train) vs 47.7 % (greedy, test). Like-for-like on the marginal, training's
   whole-run 15.5 % vs test 9.44 % across all four sets.

## Drop-in rebuttal sentence (numbers as verified)

> Training and test zoom rates differ because the data and the decoder do, not because
> the policy does: rollouts are sampled at T = 1.0 (six per prompt) on an s_ref-rebalanced
> split where hard queries are 18.2 % of grounding, whereas evaluation is greedy on a test
> set that is 88.8 % easy. The per-stratum rates recombine exactly:
> 47.7 % (hard) × 11.2 % + 6.7 % (easy) × 88.8 % = 11.3 %, the reported HICO figure
> (SWIG: 45.2 / 13.4 → 16.8 %). The gate itself is unchanged and in fact sharpens with
> GRPO (hard 43.0 % → 47.7 % over the SFT model at identical decoding).

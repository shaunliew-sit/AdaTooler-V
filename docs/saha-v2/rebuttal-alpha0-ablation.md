# Rebuttal-ready: α=0 ablation (isolating the PTG tool-gain term)

**Status:** NOT in the paper (Path A). Held for rebuttal. Run 2026-07-02.
**Answers the anticipated reviewer question:** *"Does the tool-gain term add anything over vanilla outcome-only GRPO? Your ablation only varies training stages, not the reward."*

---

## One-paragraph answer (paste-ready)

We ran the requested ablation: an outcome-only variant (α=0, i.e. R = R_format·R_outcome, the tool-gain term removed), trained identically to our model and evaluated against it on a matched, fixed-seed subset of HICO-DET grounding (1,500 samples; 176 Hard / 1,324 Easy by the same s_ref<0.5 split used in the paper). Two findings. (i) The selective zoom behavior is **not** an artifact of the tool-gain reward: the α=0 policy zooms ~44% on Hard vs ~5% on Easy, the same gate our full model shows — consistent with our (softened) claim that the difficulty-aware SFT establishes the gate and GRPO reinforces it. (ii) The tool-gain term nonetheless yields a **small but statistically significant** improvement in Hard-case grounding recall: **+0.98 AR (paired bootstrap 95% CI [+0.37, +1.95], n=176)**, i.e. it makes zooms on unreliable-proposal cases more productive without changing how often the tool is invoked. Pooled AR is unchanged (Hard is ~11% of the set and near the accuracy floor), which is why the effect is best seen on the Hard stratum.

## Numbers (matched 1,500-sample subset, HICO grounding)

| variant | ckpt step | All AR | **Hard AR** | Easy AR | zoom% Hard | zoom% Easy |
|---|---|---|---|---|---|---|
| α=0 (outcome-only GRPO) | 200 | 28.98 | 4.05 | 38.30 | 44.3 | 5.5 |
| α=0.6 (PTG, our model) | 165 | 28.97 | **5.03** | 37.91 | 38.6 | 4.6 |

Paired Hard-AR Δ (α=0.6 − α=0) = **+0.98**, 95% CI [+0.37, +1.95] (2000-resample paired bootstrap, n_hard=176). CI excludes 0.

## Honest caveats (state if pressed)
- Step mismatch: α=0 kept only step 200; the paper's SAHA-4B is step 165. Conservative — α=0 trained *longer* and still trailed on Hard-AR.
- Single seed, single subset, n_hard=176; Hard grounding is near a capability floor (4–5 AR), so the absolute gain is small.

## Provenance / how to reproduce
- α=0 ckpt: `checkpoints/SAHA-CF/SAHA-CF-a0-a0abl-runB-4b-avg2/global_step_200` (train cmd = runB recipe, ALPHA=0; see [[saha-cf-alpha0-ablation-plan]]).
- α=0.6 ckpt: `/workspace/hoi-tool-use-checkpoints/saha-cf-4b-grpo-step165`.
- Subset: `hoi-benchmarks/subset_annotations/hico_ground_test_sub1500_seed42.json`; buckets from `hoi-benchmarks/sref_cache.json` (grounding.hico).
- Results: `hoi-benchmarks/results-sftgrpo/hico_ground_{a0abl_4b_step200,grpo_4b_step165}_sub1500/`.
- Eval needs `VLLM_ATTENTION_BACKEND=TRITON_ATTN` on the H200/CUDA-12.4 box.

---

# Addendum: 8B α=0, FULL benchmarks (run 2026-08-30/31)

**Scope differs from the 4B section above.** That one is a 1,500-sample HICO-grounding
subset at 4B. This one is the **8B** α=0 checkpoint evaluated on the **complete** test
sets of all four benchmarks (92,446 samples total). The two are not interchangeable.

**Motivation:** the 2026-08-29 decision was to *reuse* the 4B α=0 rather than retrain at
8B; this 8B run was done anyway once compute freed up, so the α=0 row can be stated at the
paper's headline model size.

## Checkpoint

- α=0 8B: `/workspace/data/keep_a0_global_step_165` (verified byte-identical copy of
  `checkpoints/SAHA-CF/SAHA-CF-a0-newsft-resampled-8b-minar10/global_step_165`).
- Trained from `/workspace/data/qwen3vl-8b-saha-sft-merged` with `ALPHA=0`, `N=6`,
  `BATCH_SIZE=4`, `SAHA_CF_GROUNDING_METRIC=minAR10`, data
  `train_resampled_moderate.parquet`. Launcher:
  `verltool/examples/train/hoi/run_grpo_v2_alpha0.sh` (single GPU).
- **Stopped at step 165 of a planned 1000.** Training-rollout accuracy showed no trend:
  OLS on per-step means gave slope **−0.067 per 100 steps, 95% CI [−0.138, +0.004]**
  (p=0.067, 4,080 rollouts) — flat, so further steps were not expected to help.

## Results (full test sets, zoom_mode=adaptive, greedy decode via vLLM)

Referring:

| benchmark | n | exact_match | zoom_rate | zoom_out |
|---|---|---|---|---|
| HICO action referring | 33,405 | **24.56** | 7.72 | 0 |
| SWiG action referring | 19,680 | **7.89** | 2.16 | 0 |

Grounding:

| benchmark | n | AR | AR@0.5 | AR@0.75 | ARs | ARm | ARl | zoom_rate | zoom_out |
|---|---|---|---|---|---|---|---|---|---|
| HICO grounding | 20,028 | **29.09** | 51.36 | 28.53 | 3.05 | 17.38 | 36.72 | 11.37 | 0 |
| SWiG grounding | 19,333 | **25.69** | 39.38 | 26.03 | 7.26 | 18.66 | 27.24 | 15.93 | 3 |

All four runs completed every sample (0 missing records); `missing_proposals` was 0, 0, 1, 1.

## Two observations worth stating explicitly

1. **`zoom_out` is effectively unused by this checkpoint**: 3 invocations across 92,446
   eval samples (0 on three of the four benchmarks), and `n_zoom_out = 0.000` at every one
   of the 170 training steps. Since the method is described as having two active tools
   (`zoom_in`, `zoom_out`), say this plainly rather than leaving it implicit — a reviewer
   checking tool statistics will find it.
2. **Small objects are where grounding fails**: HICO ARs 3.05 vs ARl 36.72 (12×). This is
   the difficulty axis PTG targets, so it motivates the method — but it also means the
   headline AR is dominated by large objects.

## Comparison vs the paper's 8B model (α=0.6)

Source: `hoi-benchmarks/results-sftgrpo/*_grpo_8b`, which `checkpoint-inventory.md` records
as `SAHA-CF-a0.6-newsft-resampled/global_step_1000` — the paper's 8B row. Same four full
test sets, identical sample counts, so the rows are directly alignable.

| benchmark | metric | α=0 (step 165) | α=0.6 (step 1000) | Δ |
|---|---|---|---|---|
| HICO action referring | exact_match | 24.56 | 25.55 | **+0.99** |
| SWiG action referring | exact_match | 7.89 | 12.59 | **+4.70** |
| HICO grounding | AR | 29.09 | 31.03 | **+1.94** |
| | AR@0.5 | 51.36 | 54.94 | +3.58 |
| | ARs / ARm / ARl | 3.05 / 17.38 / 36.72 | 3.58 / 19.31 / 38.83 | +0.53 / +1.93 / +2.11 |
| SWiG grounding | AR | 25.69 | 29.05 | **+3.36** |
| | AR@0.5 | 39.38 | 43.39 | +4.01 |
| | ARs / ARm / ARl | 7.26 / 18.66 / 27.24 | 8.09 / 21.25 / 30.78 | +0.83 / +2.59 / +3.54 |

α=0.6 is ahead on all four benchmarks and on every AR breakdown.

### The step mismatch is a real confound — do not ignore it

α=0 ran 165 steps; α=0.6 ran 1000. This is the **opposite** of the 4B ablation above (where
α=0 trained *longer*, making that comparison conservative). Here the deltas are inflated by
whatever the extra 835 steps contribute. Per `saha-accv-rebuttal-ablation-decisions`:

- **Grounding is flat across steps 165→1000** (+0.87 AR HICO). So the grounding deltas
  (+1.94, +3.36) are mostly attributable to the reward and are the defensible rows.
- **Referring is NOT flat** — 4B SWiG referring exact went **7.42 → 14.63** over that same
  range. The +4.70 SWiG referring delta is therefore largely or entirely a training-length
  artifact, not evidence for PTG. Note α=0's 7.89 @165 sits almost exactly where the 4B
  main model was at step 165 (7.42), which is consistent with the gap being about steps.
  (Cross-size, so suggestive rather than conclusive.)

**Recommendation:** lead with grounding. If a referring comparison is needed, evaluate
α=0.6 at `global_step_95` (a surviving checkpoint, closest to 165) rather than quoting the
@1000 delta — or state the step mismatch in the caption. Do not present +4.70 as a reward effect.

### Superseded / do-not-use

`results-sft-grpo-qwen3vl-8b-step{25,1000}` (2026-04-26, AR 25.69 / 27.29) is the
**pre-re-SFT** generation per `checkpoint-inventory.md`. It is not a valid comparison for
these numbers despite the similar directory name.

## Provenance

- Results: `hoi-benchmarks/results-sftgrpo-vanila-grpo/{hico_action,hico_ground,swig_action,swig_ground}_grpo_8b_vanila_grpo/`
- Served by vLLM container `vllm-qwen-inference` as `qwen3VL-8B-SAHA-v2-vanila-GRPO`.
- Scripts: `hoi-benchmarks/run_{hico,swig}_{action,ground}_sftgrpo_eval.sh`, defaults,
  no `VERBOSE`, `MAX_TURNS=6`, `PYTHON_BIN=/opt/conda/envs/verl-tool-env/bin/python`.
- SPICE fails on this box (no Java); BLEU/ROUGE-L/CIDEr/exact-match/AR unaffected.

## Companion ablation: no-SFT (direct GRPO), 8B, full benchmarks

Base `Qwen3-VL-8B-Instruct` + PTG (alpha=0.6), **no SAHA SFT**, stopped at step 120
(training-rollout accuracy flat: slope -0.098 per 100 steps, CI [-0.209, +0.014]).
Checkpoint `checkpoints/SAHA-CF/SAHA-CF-a0.6-base8b-nosft-minar10/global_step_120`.
Evaluated 2026-08-31 on the same four full test sets; all records complete.

| benchmark | metric | value | zoom_rate | zooms/sample | zoom_out |
|---|---|---|---|---|---|
| HICO grounding | AR | **20.06** (AR@0.5 34.78; ARs/m/l 2.01/11.36/25.59) | 99.54 | 1.813 | 3,328 |
| SWiG grounding | AR | **13.12** (AR@0.5 20.06; ARs/m/l 5.14/9.96/13.80) | 98.15 | 1.431 | 2,719 |
| HICO referring | exact_match | **16.85** | 100.00 | 1.590 | 4,043 |
| SWiG referring | exact_match | **6.02** | 99.99 | 1.501 | 2,984 |

### Three-way summary (the table to use)

| Variant | HICO AR | SWiG AR | HICO EM | SWiG EM |
|---|---|---|---|---|
| no-SFT (direct GRPO) | 20.06 | 13.12 | 16.85 | 6.02 |
| alpha=0 (outcome-only) | 29.09 | 25.69 | 24.56 | 7.89 |
| **SAHA (SFT + PTG)** | **31.03** | **29.05** | **25.55** | **12.59** |

Monotonic on all four benchmarks. no-SFT deficits vs SAHA: **-10.97 / -15.93 AR**,
**-8.70 / -6.57 EM** — roughly 5x the alpha=0 gaps, while making ~14x more tool calls.

### Why this row is more robust than the alpha=0 row

The alpha=0 comparison is confounded by training length (165 vs 1000 steps, and referring
is step-sensitive). The no-SFT deficit is **not** the same kind of claim: the failure is a
*missing capability* — no gate at all, zooming on 98-100% of queries — rather than an
undertrained version of the right behaviour. A 120-step run is a fair characterisation of
"GRPO without the SFT stage" on that basis. Prefer it as the headline ablation.

### zoom_out: this reverses the earlier reading

no-SFT invokes `zoom_out` **13,074 times** across the four benchmarks, vs **3** for alpha=0.
So the tool is functional and reachable by the policy; the SFT'd model simply learns it is
not needed. That is a stronger answer to the "two tools claimed, one used" concern than
conceding the 3-invocation statistic alone — the earlier framing in this doc (observation 1)
should be updated to say the trained policy *learns to forgo* zoom_out, not that it is unused.

## Provenance (nosft)

- Results: `hoi-benchmarks/results-sftgrpo-nosft/*_grpo_8b_nosft/`
- Served as `qwen3VL-8B-SAHA-v2-noSFT-GRPO`; same scripts/settings as the alpha=0 runs.

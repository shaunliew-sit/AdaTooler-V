# SDS-GRPO: Reward Design for Adaptive Tool Use in HOI Detection

This document records the complete reward formulation for the SDS-GRPO training pipeline, grounded in the pain point analysis of existing HOI models. It supersedes `ideas.md` (original AT-GRPO design) and `plan.md` (redesign rationale).

---

## Part 1: Motivation — Why Tools Are Needed

### Root Problem

Both evaluated baseline models (Qwen3VL-8B-Instruct zero-shot and Groma-7B fine-tuned) share one fundamental limitation:

> **They look at the whole image at fixed resolution, once, and must answer from that single view.**

All failure modes trace to this single constraint.

---

### Failure Mode 1: Small Objects Are Invisible

At standard evaluation resolution (640×480), objects smaller than 32×32 px occupy less than 0.01% of image area. The encoder cannot resolve structure at that scale.

| Object | Raw area | Person area | Ratio |
|--------|----------|-------------|-------|
| baseball (`pitching_208`) | **6 px²** | 166,579 px² | 1 : 27,763 |
| tie (`HICO_00002632`) | **8 px²** | ~60,000 px² | 1 : 7,500 |
| switch (`repairing_353`) | **9 px²** | 208,824 px² | 1 : 23,203 |
| screw (`attaching_231`) | **26 px²** | 221,774 px² | 1 : 8,530 |
| donut (`HICO_00006546`) | **24 px²** | ~50,000 px² | 1 : 2,083 |

**Metric impact:**

| Model | HICO ARs | SWIG ARs | HICO small-obj failure |
|-------|----------|----------|------------------------|
| Qwen3VL-8B-Instruct | 3.76% | 7.96% | 74.0% |
| Groma-7B | **0.10%** | **0.83%** | **99.0%** |

Near-zero small-object recall for both models, on both datasets. This is a perceptual failure, not semantic: both models know what a baseball looks like — they simply cannot see 6 px² of image.

**Resolution of zoom-in:**
```
Before (full image, 640×480):
  baseball = 6 px² → 0.002% of image → encoder resolves nothing

After zoom-in (crop 64×64 region, upsample to 640×640):
  baseball ≈ 6,000 px² → 14.6% of crop → shape, seams, colour visible
```

**Side-effect caught in referring:** Groma predicts "no interaction tie" (4/7 cases) when given an 8 px² tie crop as input — the fine-tuned bias fires because the tie is visually static at that resolution. A zoomed crop confirming the tie is attached to the person would correct the prediction.

---

### Failure Mode 2: Crowded Scenes Exceed Single-Pass Capacity

| Scene type | Qwen3VL failure | Groma failure |
|------------|-----------------|---------------|
| HICO complex (>5 GT pairs) | 95.8% | **100.0%** |
| Worst case: 161 GT pairs | 0/161 matched | 0/161 matched |

Top 7 HICO crowded failures (all "no interaction" scenes — sports events, markets):

| Image | Action | GT pairs |
|-------|--------|----------|
| HICO_test2015_00002441 | no interaction surfboard | 161 |
| HICO_test2015_00004271 | no interaction orange | 156 |
| HICO_test2015_00006183 | no interaction car | 105 |
| HICO_test2015_00002072 | no interaction motorcycle | 84 |
| HICO_test2015_00005743 | no interaction car | 81 |

The failure is combinatorial: even if every person and object is individually visible, enumerating 72–161 pairs requires iterating over spatial sub-regions — impossible in one forward pass.

---

### Failure Mode 3: Semantic Guessing Without Re-Examination

When the initial view is insufficient, both models fall back on priors rather than seeking more evidence:

- **Qwen3VL:** Produces verbose, plausible but metric-penalized output (BLEU-4 ≈ 0). Never learns the "no interaction" annotation convention (0% exact match on 12.9% of HICO referring entries).
- **Groma:** Context contamination — in `HICO_00000156`, predicts "lighting cigarette" for both toothbrush triplets because another person is smoking nearby. Cannot isolate the target pair without re-examining.

---

### What Tools Do and Do Not Fix

| Failure mode | zoom-in | zoom-out (enumeration) |
|-------------|---------|------------------------|
| Small real objects (baseball, screw, tie) | **YES** | — |
| Crowded scenes (70–160 GT pairs) | — | **YES** |
| Groma false "no interaction" on tiny objects | **YES** — crop confirms active contact | — |
| Context contamination | PARTIALLY — isolates target pair | — |
| Conceptual/invisible objects (saliva, gap) | **NO** — nothing to see | — |
| Qwen3VL vocabulary mismatch | **NO** — needs fine-tuning | — |
| Gallus gallus terminology confusion | **NO** — canonical name mapping needed | — |

The research claim is precise: tools address **perceptual failures** (resolution, enumeration capacity). They do not address annotation convention failures or vocabulary alignment failures.

---

## Part 2: SDS-GRPO Reward Formula

### Overview

```
R_total = R_format × (R_outcome + α × R_tool)
```

| Component | Description |
|-----------|-------------|
| `R_format` | Binary gate: 1 if valid `<think>…</think><answer>…</answer>` structure, 0 otherwise |
| `R_outcome` | Task correctness (AR for grounding; max(exact_match, 0.5·ROUGE-L + 0.5·METEOR) for referring) |
| `R_tool` | Adaptive tool reward, gated by object-area spatial difficulty |
| `α = 0.6` | Tool reward weight (correctness dominates) |

---

### 2.1 Outcome Reward (R_outcome)

**Grounding — Average Recall at IoU {0.5, 0.75}:**

```
R_outcome = (Recall@0.5 + Recall@0.75) / 2

Recall@τ = (matched pairs at threshold τ) / |GT pairs|
```

Pairs are matched greedily (descending pair_score = 0.5·IoU_person + 0.5·IoU_object).

**Referring — text similarity:**

```
R_outcome = max(exact_match, 0.5 · ROUGE-L_F1 + 0.5 · METEOR)
```

where ROUGE-L and METEOR operate on lowercased, whitespace-tokenized strings. Uses `nltk.meteor_score` if available, otherwise falls back to unigram F1 with fragmentation penalty.

---

### 2.2 Spatial Difficulty Score (SDS)

SDS quantifies how much zoom-in is expected to help, based entirely on **object area** (not minimum of person/object — the person is almost never the perceptual bottleneck).

```
SDS_obj = clip((0.15 - min_obj_area_1000) / 0.14,  0.0, 1.0)
```

where `min_obj_area_1000` is the minimum GT object area, normalized to the [0,1000]² coordinate grid:

```
obj_area_1000 = (x2 - x1) × (y2 - y1) / (1000 × 1000)
```

**Linear mapping:**
- obj_area_1000 ≤ 0.01 (tiny — e.g., 8 px² tie at 640×480) → SDS = 1.0
- obj_area_1000 ≥ 0.15 (large — e.g., chair occupying 15% of image) → SDS = 0.0
- Between: linear interpolation

**Why object-area only:** The pain point data shows the bottleneck is always the object. Person areas in failure cases range 50K–220K px² — they are never invisible. Using `min(person, object)` allows edge cases where a distant person drives SDS, which is wrong.

**For referring tasks:** SDS is computed from the provided `object_bbox` (already given as input to the model), using the same formula.

---

### 2.3 Tool Reward (R_tool)

```
R_tool = R_zoom_in + R_zoom_out_hygiene
```

zoom_in and zoom_out are counted separately because they are fundamentally different actions:
- **zoom_in** = information-gathering (crops and upsamples a region)
- **zoom_out** = navigation/context-restore (returns to original view, adds no new information)

SFT data confirms: 30.3% of tool-using samples use zoom_in only (never zoom out) — this must not be penalized.

---

#### Component 1: R_zoom_in

**Dynamic optimal count:**

```
n_opt = round(2 × SDS_obj)    ∈ {0, 1, 2}
```

| SDS_obj range | n_opt | Interpretation |
|---------------|-------|----------------|
| 0.00 – 0.24 | 0 | Large objects — no zoom needed |
| 0.25 – 0.74 | 1 | Medium objects — single zoom usually sufficient |
| 0.75 – 1.00 | 2 | Tiny objects — may need multi-region inspection |

**Piecewise formula:**

```
If n_opt > 0:
    R_zoom_in = (SDS_obj - τ) × G(n_zoom_in, n_opt)

If n_opt = 0 (large objects):
    R_zoom_in = -(1 - SDS_obj) × λ_abstain × n_zoom_in
```

**Gaussian efficiency function:**

```
G(n, n_opt) = exp[-γ × ((n - n_opt) / max(n_opt, 0.5))²]
```

This peaks at G=1 when `n_zoom_in = n_opt` and decays symmetrically. The `max(n_opt, 0.5)` denominator prevents division-by-zero when n_opt=1.

**Rationale for the two branches:**
- When n_opt > 0: The net signal `(SDS_obj - τ)` is positive for SDS_obj > τ=0.3, so zoom_in is rewarded on hard images and penalized (if used) on borderline-easy images.
- When n_opt = 0: The Gaussian would multiply near-zero (exp[-8] ≈ 0.0003), so any zoom-in on easy images would go nearly unpunished. The explicit linear penalty `-(1 - SDS_obj) × λ_abstain × n_zoom_in` provides a clear, proportional signal instead.

---

#### Component 2: R_zoom_out_hygiene

```
R_zoom_out_hygiene = λ_h × min(n_zoom_out, n_zoom_in) / max(n_zoom_in, 1)
                   = 0   if n_zoom_in = 0 or n_zoom_out = 0
```

- Small bonus (λ_h = 0.05) for pairing zoom_out with zoom_in
- Capped at one bonus per zoom_in call (no credit for excess zoom_outs)
- Zero if no zoom_in was used (avoids zero-division case)

---

#### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α | 0.6 | Correctness (R_outcome) dominates; tool reward is an adjustment, not the objective |
| τ | 0.3 | SDS threshold: ~26% of SFT samples correctly use no tools |
| γ | 2.0 | Gaussian decay rate — same as original AT-GRPO |
| λ_h | 0.05 | zoom_out hygiene bonus; small enough not to dominate, large enough to be measurable |
| λ_abstain | 0.1 | Per-zoom_in penalty for unnecessary use on easy images; proportional to zoom count |

---

### 2.4 Complete R_total (Expanded)

```
R_total = R_format × (R_outcome + α × R_tool)

       = R_format × (R_outcome + 0.6 × (R_zoom_in + R_zoom_out_hygiene))
```

Where R_zoom_in is the piecewise SDS-gated formula above.

**Theoretical bounds:**
- Minimum: 0 (format gate fails, or n_opt=0 and heavy unnecessary zooming)
- Maximum: R_format × (1.0 + 0.6 × ((1.0 - 0.3) × 1.0 + 0.05)) = **1 × (1.0 + 0.6 × 0.75) = 1.45**
  - Achieved when: SDS=1.0, n_zoom_in=2, n_zoom_out=2 (n_opt=2, G=1, full hygiene)

---

## Part 3: Worked Examples

### 3.1 Grounding Task — Small Object

**Setup:** Grounding — "wearing tie", tie is 8×9 px in 1000-grid (SDS_obj = 1.0, n_opt = 2). From the verified test suite (`TestComputeToolReward::test_example_A_*`).

**Response A — zoom_in→out (optimal grounding pattern):**

```
n_zoom_in=2, n_zoom_out=2, n_opt=2

G(2, 2) = exp[-2 × ((2-2)/2)²] = 1.0
R_zoom_in    = (1.0 - 0.3) × 1.0 = 0.700
R_hygiene    = 0.05 × min(2,2)/2  = 0.050
R_tool       = 0.750
R_total      = 1 × (R_outcome + 0.6 × 0.750) = R_outcome + 0.450
```

**Response B — no zoom (can't see 8px² tie → prediction fails):**

```
n_zoom_in=0, n_opt=2

G(0, 2) = exp[-2 × ((0-2)/2)²] = exp[-2] = 0.135
R_zoom_in    = 0.7 × 0.135 = 0.095
R_tool       = 0.095
R_total      = 1 × (R_outcome + 0.6 × 0.095) = R_outcome + 0.057
               (R_outcome ≈ 0 since boxes cannot match invisible object)
```

The max possible R_total (SDS=1.0, n_opt=2, perfect zoom, R_outcome=1.0) = **1.450**.

---

### 3.2 Referring Task — Tiny Object  *(verified)*

**Setup:** Referring — "wearing tie", `object_bbox=[450, 180, 458, 189]` in 1000-grid.
- tie area = 8 × 9 = 72 px² / 1,000,000 = 7.2×10⁻⁵
- SDS_obj = (0.15 − 7.2×10⁻⁵) / 0.14 ≈ 1.07 → clips to **1.0**
- n_opt = round(2 × 1.0) = **2**

**Response A — zoom_in only, correct answer (canonical referring-SFT pattern):**

```
<think>The tie region is extremely small. I need to zoom in to confirm
the interaction.</think>
<tool_call>{"name": "zoom_in", "arguments": {"bbox": [440, 170, 470, 200]}}</tool_call>
<answer>wearing tie</answer>
```

```
n_zoom_in=1, n_zoom_out=0 (zoom_out absent — 38% of referring SFT)

G(1, 2) = exp[-2 × ((1-2)/2)²] = exp[-0.5] = 0.6065
R_zoom_in    = (1.0 - 0.3) × 0.6065 = 0.4246
R_hygiene    = 0.0  (no zoom_out: not penalized)
R_tool       = 0.4246
R_outcome    = 1.0  (exact match: "wearing tie")
R_total      = 1.0 × (1.0 + 0.6 × 0.4246) = 1.2547  ✓
```

**Response B — no zoom, wrong answer (Groma-style bias failure):**

```
<think>I see a person near a tie.</think>
<answer>no interaction tie</answer>
```

```
n_zoom_in=0, n_opt=2

G(0, 2) = exp[-2 × ((0-2)/2)²] = exp[-2] = 0.1353
R_zoom_in    = 0.7 × 0.1353 = 0.0947
R_tool       = 0.0947
R_outcome    = 0.319  (partial ROUGE-L/METEOR: "tie" shared, action wrong)
R_total      = 1.0 × (0.319 + 0.6 × 0.0947) = 0.3759  ✗
```

**Signal:** A (1.2547) >> B (0.3759). The model cannot correctly describe "wearing tie" without seeing the 8px² tie region, so both R_outcome and R_tool are higher when zoom is used.

**zoom_in-only vs zoom_in→out (same n_zoom_in=1):**

```
zoom_in only:  R_tool = 0.4246
zoom_in→out:   R_tool = 0.4246 + 0.05 (hygiene) = 0.4746
Difference = exactly λ_h = 0.05
```

zoom_in-only is positive and well-rewarded — the 38% of referring SFT that never zooms out is not penalized.

---

### 3.3 Referring Task — Large Object  *(verified)*

**Setup:** Referring — "standing on bench", `object_bbox=[200, 600, 800, 850]` in 1000-grid.
- bench area = 600 × 250 = 150,000 px² / 1,000,000 = **0.15**
- SDS_obj = (0.15 − 0.15) / 0.14 = 0.0 → clips to **0.0**
- n_opt = round(2 × 0.0) = **0**

**Response C — no tools, correct (optimal for easy scene):**

```
<think>The bench is large and clearly visible below the person.
No zoom needed.</think>
<answer>standing on bench</answer>
```

```
n_opt=0, n_zoom_in=0

R_zoom_in    = 0.0  (neutral: no zoom on easy image = no cost)
R_tool       = 0.0
R_outcome    = 1.0  (exact match)
R_total      = 1.0 × (1.0 + 0) = 1.0000  ✓
```

**Response D — unnecessary zoom, same correct answer:**

```
<think>Let me zoom in just to be sure.</think>
<tool_call>{"name": "zoom_in", "arguments": {"bbox": [200, 600, 800, 850]}}</tool_call>
<answer>standing on bench</answer>
```

```
n_opt=0, n_zoom_in=1

R_zoom_in    = -(1 - 0.0) × 0.1 × 1 = -0.100  (linear penalty)
R_tool       = -0.100
R_outcome    = 1.0  (still correct — tool did not help answer)
R_total      = 1.0 × (1.0 + 0.6 × (-0.100)) = 0.9400  ⚠
```

**Signal:** C (1.0000) > D (0.9400). The penalty is exactly α × λ_abstain × n_zoom_in = 0.6 × 0.1 × 1 = **0.06**. The model learns: unnecessary zoom on an easy, clearly-visible object costs 0.06 per zoom call even when the answer is correct.

---

### Summary of Verified Reward Values

All values confirmed by pytest (`tests/test_sds_grpo.py`, 74/74 passing):

| Task | Scene | Pattern | R_tool | R_total |
|------|-------|---------|--------|---------|
| Grounding | SDS=1.0, tiny obj | zoom_in×2 → zoom_out×2 (optimal) | 0.750 | R_out + 0.450 |
| Grounding | SDS=1.0, tiny obj | no zoom | 0.095 | R_out + 0.057 |
| Referring | SDS=1.0, tiny tie | zoom_in only (correct) | 0.4246 | **1.2547** |
| Referring | SDS=1.0, tiny tie | no zoom (wrong answer) | 0.0947 | **0.3759** |
| Referring | SDS=0.0, large bench | no tools (correct) | 0.000 | **1.0000** |
| Referring | SDS=0.0, large bench | unnecessary zoom (correct) | −0.100 | **0.9400** |

---

## Part 4: Reward Signal Validation

Before training, the reward was validated offline against the SFT dataset using pytest. All three checks passed.

### Check 1 — Reward Ordering (PASS)

- **High-SDS samples:** GT tool pattern beats no-tool in **100% of 626 cases** — the directional signal is maximally clean
- **Low-SDS samples:** no-tool beats GT tool in **97.7% of cases** — the 2.3% edge cases sit at the SDS boundary (neutral zone by design)

### Check 2 — Variance (PASS)

- std = **0.309** — well above the 0.05 floor; reward is not collapsed
- Range: **[0.0, 0.75]** — full expected range achieved (0.75 = theoretical max at SDS=1.0, n_opt=2, perfect hygiene)

### Check 3 — Pattern Verification (PASS)

Verified R_tool values for representative SDS levels and tool patterns:

| Pattern | R_tool |
|---------|--------|
| High-SDS grounding: zoom_in→out | **0.438** |
| High-SDS grounding: no-tool | 0.084 (low — tool would have helped) |
| Low-SDS: no-tool | **≈ 0.000** (neutral — correct behaviour) |
| Low-SDS: unnecessary zoom | **−0.097** (penalized) |
| Medium-SDS referring: zoom_in only | **0.202** (not penalized vs zoom_in→out) |
| Medium-SDS referring: zoom_in→out | **0.252** (= 0.202 + λ_h = +0.05 hygiene bonus) |

The difference between referring `zoom_in` only (0.202) and `zoom_in→out` (0.252) is exactly λ_h = 0.05, confirming the hygiene bonus fires correctly and zoom_in-only is not unfairly penalized.

---

## Part 5: Design Evolution Reference

The reward went through two design iterations:

### V1: AT-GRPO (ideas.md / original)

```
R_i = R_outcome_i + α · ΔS_i · G(n_tool, n_max=2)
```

where ΔS_i = S⁺ − S⁻ (tool benefit score, computed by running 8 inference samples with/without tools). ΔS is more principled than SDS but requires expensive pre-computation (16 inference runs per sample) and cannot be computed before a model exists (chicken-and-egg during RL).

### V2: SDS-GRPO (current, implemented in sds_grpo.py)

Replaces ΔS with SDS_obj (object-area proxy), separates zoom_in/zoom_out counting, uses a dynamic n_opt, and adds the piecewise n_opt=0 branch to properly penalize unnecessary zoom on large objects. The full formula is documented in Part 2 above.

Key changes from V1 to V2:

| Aspect | V1 (AT-GRPO) | V2 (SDS-GRPO) |
|--------|-------------|---------------|
| Tool-benefit signal | ΔS (per-sample, pre-computed) | SDS_obj (object-area proxy, computed on-the-fly) |
| Tool counting | n_tool = zoom_in + zoom_out | zoom_in and zoom_out counted separately |
| Optimal count | Fixed n_max = 2 | Dynamic n_opt = round(2 × SDS_obj) ∈ {0, 1, 2} |
| zoom_out treatment | Penalized same as zoom_in | Free + small hygiene bonus (λ_h = 0.05) |
| Easy-image no-tool | Subtle negative ΔS·G | Exactly 0 (neutral) |
| Easy-image with-tool | Near-zero (exp[-8]) | Clear linear penalty `-(1-SDS)·λ_abstain·n` |
| R_total structure | R_outcome + α·R_tool | R_format × (R_outcome + α·R_tool) — format gate added |

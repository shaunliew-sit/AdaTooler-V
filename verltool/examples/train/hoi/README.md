# SDS-GRPO: Spatial Difficulty-Gated GRPO for HOI Adaptive Tool Use

Train a vision-language model to selectively use zoom tools for Human-Object Interaction (HOI) detection and action recognition, using reinforcement learning with the SDS-GRPO reward.

---

## Reward Design

### Motivation: Why We Need a Tool-Aware Reward

Standard RL rewards only measure whether the answer is correct. But in tool-augmented HOI, **correctness alone is not enough** — the model must also learn *when* to use tools and *when to skip them*:

- **Small objects are invisible without zooming.** A tie at 8 px² or a baseball at 6 px² cannot be resolved by the vision encoder at full-image resolution. Both Qwen3VL (ARs = 3.76%) and Groma (ARs = 0.10%) achieve near-zero recall on small objects. Zooming in turns 6 px² into ~6,000 px², making the object visible.

- **Large objects do not benefit from zooming.** A chair occupying 15% of the image is already clearly visible. Zooming wastes inference turns and can introduce errors. 26.6% of SFT training samples answer correctly without any tools.

- **The two tools serve different purposes.** `zoom_in` gathers new visual information (crops and upsamples a region). `zoom_out` simply restores the full image view — it provides no new information. In 38% of SFT referring samples, the model zooms in and answers directly from the crop without ever zooming out. A reward that counts both tools identically would penalize this valid strategy.

The tool reward makes the total reward **context-dependent**: the same tool usage gets rewarded on hard images and penalized on easy images, teaching the model to zoom adaptively.

### Overview

The total reward for each response:

```
R_total = R_format  *  (R_outcome  +  alpha  *  R_tool)
            ↑               ↑            ↑        ↑
         format gate    task accuracy   weight   tool usage
```

| Symbol | Name | What it is | Value |
|--------|------|-----------|-------|
| **R_format** | Format gate | 1 if the response has valid `<think>...</think>` and `<answer>...</answer>` tags, 0 otherwise. A hard gate: malformed responses get zero reward regardless of content. | 0 or 1 |
| **R_outcome** | Outcome reward | How correct the answer is. For grounding: Average Recall at IoU thresholds {0.5, 0.75}. For referring: max(exact match, 0.5 * ROUGE-L + 0.5 * METEOR). | 0 to 1 |
| **alpha** | Tool weight | Controls how much influence tool usage has on the total reward relative to answer correctness. At alpha = 0.6, correct answers (R_outcome) are always the dominant signal, while tool usage (R_tool) provides up to 60% adjustment on top. This prevents the model from gaming the tool reward at the expense of accuracy. | 0.6 |
| **R_tool** | Tool reward | The adaptive tool usage reward. Positive when the model uses tools appropriately for the difficulty level, negative when tools are used unnecessarily. Defined below. | roughly -0.2 to +0.8 |

**Why alpha = 0.6?** It ensures correctness always dominates. Consider: even a perfect tool reward (R_tool = 0.75) only adds 0.6 * 0.75 = 0.45 to the total, while R_outcome ranges 0 to 1. A model that always gets the right answer (R_outcome = 1.0) but uses tools wastefully (R_tool = -0.2) still scores 1.0 + 0.6 * (-0.2) = 0.88 — better than a model with perfect tool usage but a wrong answer (0.0 + 0.6 * 0.75 = 0.45). **Accuracy first, tool efficiency second.**

### Spatial Difficulty Score (SDS)

SDS measures how hard it is to see the object in the image, based on its size:

```
SDS = clamp( (0.15 - object_area) / 0.14,  0,  1 )
```

| Symbol | What it is |
|--------|-----------|
| **object_area** | Area of the smallest object bounding box across all GT pairs, normalized to [0, 1] in a 1000 x 1000 coordinate grid. |
| **0.15** | Upper threshold: objects larger than 15% of the image area → SDS = 0 (easy, no zoom needed). |
| **0.14** | Scaling denominator (= 0.15 - 0.01), maps the range [0.01, 0.15] linearly to [1.0, 0.0]. |

**Intuition:** Tiny object → SDS near 1.0 → "hard, zoom would help". Large object → SDS near 0.0 → "easy, zoom is wasteful".

| Object | Pixel area | SDS | Meaning |
|--------|-----------|-----|---------|
| Baseball (pitching_208) | 6 px² | **1.00** | Invisible without zoom |
| Tie (HICO_00002632) | 8 px² | **1.00** | Invisible without zoom |
| Screw (attaching_231) | 26 px² | **1.00** | Invisible without zoom |
| Medium object | ~4000 px² | **0.69** | Somewhat hard to see |
| Chair (large) | ~15% of image | **0.00** | Clearly visible |

**Why object area only?** The pain point analysis shows the **object** is always the perceptual bottleneck, not the person. Even in failure cases, persons occupy 50,000–220,000 px². The smallest person is still 2,000x larger than the smallest object (baseball: 6 px² vs person: 166,579 px², a 1:27,763 ratio). Using `min(person, object)` would let irrelevant person-area noise leak into the difficulty signal.

### R_tool: The Adaptive Tool Reward

R_tool has two parts:

```
R_tool = R_zoom_in + R_zoom_out_bonus
```

#### Part 1: R_zoom_in — Rewarding (or Penalizing) the Zoom-In Decision

`zoom_in` is the **information-gathering** action — it crops a region and shows it at higher resolution. Whether this helps depends entirely on the object size.

**Step 1: Determine the optimal number of zoom-ins based on difficulty.**

```
n_opt = round(2 * SDS)
```

| SDS range | n_opt | Meaning |
|-----------|-------|---------|
| 0.00 - 0.24 | **0** | Large objects. No zoom needed. |
| 0.25 - 0.74 | **1** | Medium objects. One zoom usually enough. |
| 0.75 - 1.00 | **2** | Tiny objects. May need to zoom into multiple regions. |

This matches the SFT data: among tool-using samples, 69.1% use 1 zoom_in, 27.5% use 2.

**Step 2: Compute R_zoom_in based on whether the model hit the optimal count.**

*Case 1 — Objects are small or medium (n_opt >= 1): reward zooming, penalize not zooming*

```
R_zoom_in = (SDS - 0.3)  *  efficiency(n_zoom_in, n_opt)
               ↑                        ↑
         difficulty signal     how close to optimal
```

| Symbol | What it is |
|--------|-----------|
| **SDS - 0.3** | The difficulty signal. Positive when SDS > 0.3 (objects are hard to see), negative when SDS < 0.3 (objects are easy). The threshold 0.3 was chosen because 26.6% of SFT samples skip tools entirely — these are the easy images below this threshold. |
| **efficiency** | A Gaussian (bell curve) centered at n_opt. Maximum value of 1.0 when n_zoom_in = n_opt, decays smoothly as n_zoom_in deviates. Formula: `exp(-2 * ((n_zoom_in - n_opt) / n_opt)^2)`. The constant 2 controls steepness — using 1 zoom_in when n_opt = 2 still gets 60.7% efficiency (not a cliff). |

Example efficiency values when n_opt = 2 (tiny objects):

| n_zoom_in | efficiency | Interpretation |
|-----------|-----------|----------------|
| 0 | 0.135 | Severely under-zoomed |
| 1 | 0.607 | Acceptable but suboptimal |
| **2** | **1.000** | **Optimal** |
| 3 | 0.607 | Slightly over-zoomed |

*Case 2 — Objects are large (n_opt = 0): penalize unnecessary zooming*

```
R_zoom_in = -0.1  *  (1 - SDS)  *  n_zoom_in
              ↑         ↑            ↑
           penalty   scales with    per zoom_in
           rate      how easy       (more zooms = more penalty)
                     the image is
```

When objects are large, every unnecessary zoom_in costs `-0.1 * (1 - SDS)`. For a very easy image (SDS = 0), that is -0.1 per zoom_in. If the model skips tools entirely, R_zoom_in = 0 (neutral — the reward comes purely from R_outcome).

#### Part 2: R_zoom_out_bonus — Small Bonus for Context Restore

`zoom_out` returns to the full image view after zooming in. It provides **no new visual information** — the model already saw the full image before the first zoom_in. It is purely a navigation action that helps the model re-orient before making its final prediction.

```
R_zoom_out_bonus = 0.05  *  min(n_zoom_out, n_zoom_in) / n_zoom_in
                    ↑              ↑
               small bonus    capped: can't get more credit
                              than the number of zoom_ins
```

| Scenario | R_zoom_out_bonus | Why |
|----------|-----------------|-----|
| 1 zoom_in, 1 zoom_out | **0.05** | Good practice: restored context |
| 2 zoom_in, 2 zoom_out | **0.05** | Good practice |
| 1 zoom_in, 0 zoom_out | **0.00** | No penalty — 38% of referring SFT does this |
| 0 zoom_in, 0 zoom_out | **0.00** | No tools used |

The bonus is intentionally small (0.05) because zoom_out is optional. The SFT data shows the model often answers correctly from the cropped view alone (38% of referring samples). Penalizing the absence of zoom_out would fight the SFT prior.

### Putting It All Together

```
R_total = R_format  *  ( R_outcome  +  0.6  *  R_tool )
                          ↑              ↑       ↑
                    "is the answer     weight  "did the model zoom
                     correct?"                  appropriately for
                                                this image's difficulty?"
```

The reward is **context-dependent**: the same number of zoom_in calls produces a positive R_tool on a hard image and a negative R_tool on an easy image.

---

## Worked Example

### Setup: Grounding "wearing tie" — HICO_test2015_00002632.jpg

This is a real failure case from the pain point analysis. The tie occupies **8 px²** in the original image. Groma-7B predicts "no interaction tie" for ground truth "wearing tie" because it cannot resolve the object at full-image resolution.

- Object area in 1000-grid: ~30 / 1,000,000 = 0.00003
- **SDS = (0.15 - 0.00003) / 0.14 = 1.00** (maximum difficulty)
- **n_opt = round(2 * 1.0) = 2** (two zoom-ins are optimal)

GRPO generates 4 candidate responses for the same prompt. Each receives a reward.

### Response B (best): zoom_in, zoom_out, zoom_in, zoom_out

The model zooms into the tie region (8 px² becomes ~6,000 px² after crop), confirms the tie is attached to the person's neck, zooms out to restore context, zooms into a second candidate region to verify no other pairs exist, then provides its answer.

```
R_format  = 1.0     (valid <think> and <answer> tags)
R_outcome = 0.920   (excellent Average Recall — tie correctly localized after zooming)

n_zoom_in = 2, n_zoom_out = 2, n_opt = 2

R_zoom_in:
  difficulty signal = SDS - 0.3 = 1.0 - 0.3 = 0.7
  efficiency = exp(-2 * ((2 - 2) / 2)^2) = exp(0) = 1.000  (optimal count!)
  R_zoom_in = 0.7 * 1.000 = 0.700

R_zoom_out_bonus = 0.05 * min(2, 2) / 2 = 0.050

R_tool = 0.700 + 0.050 = 0.750

R_total = 1.0 * (0.920 + 0.6 * 0.750) = 1.370
```

### Response A: zoom_in, zoom_out (single cycle)

The model zooms in once to the tie region and zooms back out. Finds the tie but does not verify other regions.

```
R_outcome = 0.850
n_zoom_in = 1, n_zoom_out = 1, n_opt = 2

R_zoom_in:
  efficiency = exp(-2 * ((1 - 2) / 2)^2) = exp(-0.5) = 0.607
  R_zoom_in = 0.7 * 0.607 = 0.425

R_zoom_out_bonus = 0.050

R_tool = 0.475
R_total = 1.0 * (0.850 + 0.6 * 0.475) = 1.135
```

### Response D: zoom_in only, no zoom_out (the referring-style pattern)

The model zooms in once and answers directly from the cropped view. This pattern is common in referring (38% of SFT) and should not be penalized.

```
R_outcome = 0.800
n_zoom_in = 1, n_zoom_out = 0, n_opt = 2

R_zoom_in = 0.7 * 0.607 = 0.425
R_zoom_out_bonus = 0.000   (no penalty for missing zoom_out)

R_tool = 0.425
R_total = 1.0 * (0.800 + 0.6 * 0.425) = 1.055
```

### Response C (worst): no tools at all

The model tries to answer from the full image. The 8 px² tie is invisible at this resolution.

```
R_outcome = 0.000   (tie invisible — model produces wrong bounding boxes)
n_zoom_in = 0, n_zoom_out = 0, n_opt = 2

R_zoom_in:
  efficiency = exp(-2 * ((0 - 2) / 2)^2) = exp(-2) = 0.135
  R_zoom_in = 0.7 * 0.135 = 0.095

R_tool = 0.095
R_total = 1.0 * (0.000 + 0.6 * 0.095) = 0.057
```

### GRPO Advantage Computation

GRPO normalizes these rewards within the group to compute advantages:

| Response | Pattern | R_total | Advantage | Effect on policy |
|----------|---------|---------|-----------|-----------------|
| B | in → out → in → out | **1.370** | **+1.25** | Strongly reinforced |
| A | in → out | 1.135 | +0.51 | Reinforced |
| D | in (no out) | 1.055 | +0.26 | Mildly reinforced |
| C | no tools | 0.057 | **-2.02** | **Strongly suppressed** |

**What the model learns:** On images with tiny objects (SDS near 1.0), zooming in is essential — the 23x gap between Response B (1.370) and Response C (0.057) creates a strong gradient signal. The model also learns that zoom_out is helpful but optional (Response D is not penalized for skipping it).

### Contrast: Large Object — "sitting on chair" (chair is 15% of image)

```
Object area = 0.15 → SDS = 0.0 → n_opt = 0
```

| Response | Tools | R_tool | R_total (assuming R_outcome = 0.9) |
|----------|-------|--------|------|
| No tools | 0 | **0.000** (neutral) | **0.900** |
| 1 zoom_in + 1 zoom_out | 2 | -0.050 (penalized) | 0.870 |
| 2 zoom_in + 1 zoom_out | 3 | -0.175 (double penalty) | 0.795 |

**What the model learns:** On images with large, clearly visible objects, every unnecessary zoom_in reduces the reward. The best strategy is to skip tools entirely and answer directly.

---

## Prerequisites

- 8x H100/A100 GPUs (80GB)
- SFT checkpoint (Qwen3-VL-8B checkpoint-600)
- HOI datasets: HICO-DET, SWIG-HOI
- YOLO-world proposals (pre-computed)

### Dependencies

```bash
# In the verl-tool conda environment
pip install rouge-score nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Pipeline

### 1. Prepare RL Dataset

```bash
cd verltool
python examples/data_preprocess/hoi/prepare_train.py \
    --data_dir /media/shaun/workspace/hoi/dataset/benchmarks_simplified \
    --hico_img_dir /media/shaun/workspace/hoi/dataset/hico_20160224_det/images/train2015 \
    --swig_img_dir /media/shaun/workspace/hoi/dataset/swig_hoi/images_512 \
    --proposals_dir /media/shaun/workspace/hoi-dataset-curation/output/proposals \
    --local_dir data/hoi \
    --max_pairs 15 \
    --filter_len 8192
```

This produces `data/hoi/train.parquet` and `data/hoi/val.parquet` with pre-computed SDS scores.

### 2. Train with SDS-GRPO

```bash
cd verltool
bash examples/train/hoi/train_qwen3vl.sh [OPTIONAL_MODEL_PATH]
```

Default model: `/media/shaun/workspace/LLaMA-Factory/saves/qwen3-vl-8b/full/sft/checkpoint-600`

Key training parameters:
- **Reward manager**: `SDS-GRPO`
- **GRPO n**: 4 (samples per prompt)
- **Batch size**: 16
- **KL coef**: 0.04
- **GPU memory**: 0.45 (conservative to avoid OOM)
- **Max turns**: 5 (covers 99.7% of SFT tool usage patterns)

### 3. Evaluate

```bash
cd verltool
bash examples/train/hoi/eval.sh CHECKPOINT_PATH STEP_NUMBER
```

Runs grounding and referring evaluation on both HICO and SWIG test sets.

## All Hyperparameters

| Symbol | Name | Value | Why this value |
|--------|------|-------|----------------|
| **alpha** | Tool weight | 0.6 | Ensures correctness dominates: even max R_tool (0.75) only adds 0.45 to the total. Matches AdaTooler-V. |
| **0.3** | SDS threshold | 0.3 | 26.6% of SFT samples skip tools entirely — these are the easy images below this difficulty level. |
| **2** | Gaussian steepness | 2.0 | Smooth decay: using 1 zoom when 2 is optimal still gets 60.7% efficiency (not a cliff). |
| **n_opt** | Optimal zoom_in count | round(2 * SDS) | Maps SDS to {0, 1, 2}. Matches SFT: 69.1% use 1 zoom_in, 27.5% use 2. |
| **0.1** | Abstain penalty rate | 0.1 | Per-zoom_in cost on easy images. Two unnecessary zoom_ins costs -0.2, which at alpha=0.6 reduces R_total by 0.12 — noticeable but not catastrophic. |
| **0.05** | Zoom-out bonus | 0.05 | Small reward for restoring context. Intentionally tiny: zoom_out is optional (38% of referring SFT skips it). |

## SDS Distribution

From validation on 1000 HICO grounding samples:

| SDS Range | % Samples | n_opt | Expected Behavior |
|-----------|-----------|-------|-------------------|
| 0.9-1.0 | ~23% | 2 | Multi-region zoom rewarded |
| 0.7-0.9 | ~21% | 1-2 | Single zoom rewarded |
| 0.3-0.7 | ~16% | 1 | Single zoom optimal |
| 0.0-0.3 | ~40% | 0 | No zoom needed; zooming penalized |

## Files

```
verltool/
  verl_tool/workers/reward_manager/sds_grpo.py   # SDS-GRPO reward manager
  verl_tool/servers/tools/pixel_reasoner.py       # zoom_in/zoom_out tool server
  examples/data_preprocess/hoi/prepare_train.py   # Dataset preparation
  examples/train/hoi/
    train_qwen3vl.sh                              # Training script
    eval.sh                                       # Evaluation script
    README.md                                     # This file
```

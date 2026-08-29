# SAHA checkpoint inventory — which checkpoint is which

**Audited 2026-08-29** on the dev box (`/workspace`). Every row below was verified by
listing the directory and counting `*.safetensors`, not inferred from names.

Paths are as they exist on the dev box. `<CKPT>` = `verltool/checkpoints/SAHA-CF`.

---

## 1. The two numbers you cite in the paper

| paper row | checkpoint | shards | how it was confirmed |
|---|---|---|---|
| **SAHA-8B** | `<CKPT>/SAHA-CF-a0.6-newsft-resampled/global_step_1000/actor/huggingface` | 8 ✅ | `run_{hico,swig}_{ground,action}_sftgrpo_eval.sh:61` hardcodes this as `CHECKPOINT_PATH` |
| **SAHA-4B** | `/workspace/hoi-tool-use-checkpoints/saha-cf-4b-grpo-step165` | 4 ✅ | `docs/saha-v2/rebuttal-alpha0-ablation.md:27`; evals logged as `qwen3VL-4B-SAHA-v2` |

SAHA-8B numbers: HICO-G AR 31.03, SWIG-G AR 29.05, HICO-ref METEOR 29.66
(`hoi-benchmarks/results-sftgrpo/*_grpo_8b`, evaluated 2026-06-21).
SAHA-4B numbers: HICO-G AR 30.35, SWIG-G AR 28.53 (`*_grpo_4b`).

> ⚠️ **`saha-cf-4b-grpo-step165` is the ONLY surviving copy of the paper's 4B model.**
> `SAHA-CF-a0.6-runB-4b-avg2/global_step_165/` was pruned by
> `remove_previous_ckpt_in_save=True` — it now holds only `data.pt` and an empty
> `actor/`. Lose the standalone export and the 4B row needs a retrain. Back it up.

---

## 2. Full inventory of `verltool/checkpoints/SAHA-CF`

| run dir | model | reward | steps saved | latest w/ HF weights | role |
|---|---|---|---|---|---|
| `SAHA-CF-a0.6-newsft-resampled` | 8B | α=0.6, avg2 | 5→1000 | **1000** (8 shards) | **SAHA-8B, paper** |
| `SAHA-CF-a0.6-runB-4b-avg2` | 4B | α=0.6, avg2 | 5→1000 | **1000** (4 shards) | strongest 4B; step 165 = paper row (pruned here) |
| `SAHA-CF-a0-a0abl-runB-4b-avg2` | 4B | **α=0**, avg2 | 5→200 | **200** (4 shards) | outcome-only ablation (rebuttal) |
| `SAHA-CF-a0.6-runA-minAR10-bigbatch` | 8B | α=0.6, **minAR10** | 5→345 | **345** (8 shards) | Run A, not in the paper |
| `SAHA-CF-a0.6-fsdp-tool-agent-sft-checkpoints_qwen3vl-8b_` | 8B | α=0.6 | →145 | 145 | **v1-SFT lineage** (pre-collapse-fix) |
| `SAHA-CF-a0.6-fsdp-resampled-sref30-tool-agent-sft-checkpoints_qwen3vl-8b_` | 8B | α=0.6 | →125 | 125 | **v1-SFT lineage** |

Only the latest step of each run carries HF weights; earlier `global_step_*` dirs
hold FSDP shards (`model_world_size_1_rank_0.pt`) or nothing.

---

## 3. Lineages

**8B**
```
Qwen3-VL-8B-Instruct  (base, /workspace/hoi-tool-use-checkpoints/base-checkpoints/)
  → LoRA SFT (LLaMA-Factory, saves/qwen3-vl-8b/lora/saha_hoi_sft, 2026-06-18)
  → merged: /workspace/data/qwen3vl-8b-saha-sft-merged            [17 GB]
  → GRPO:   SAHA-CF-a0.6-newsft-resampled/global_step_1000        ← SAHA-8B
```

**4B**
```
Qwen3-VL-4B-Instruct  (base)
  → LoRA SFT (saves/qwen3-vl-4b/lora/saha_hoi_sft, checkpoint-400, eval_loss 0.5667)
  → merged: /workspace/data/qwen3vl-4b-saha-sft-merged            [8.3 GB]
  → GRPO:   runB-4b-avg2 → step 165  = SAHA-4B (paper, standalone export only)
                         → step 1000 = strongest 4B
  → GRPO:   a0abl-runB-4b-avg2 → step 200 = α=0 ablation
```

---

## 4. NOT in the SAHA lineage — do not use for SAHA rows

| path | what it is |
|---|---|
| `/workspace/hoi-tool-use-checkpoints/hoi-tool-use-checkpoints/grpo-checkpoints/qwen3vl-sft-grpo-8b/global_step_240` | previous generation: v1 SFT + old SDS-GRPO-era reward, pulled from the GCS bucket. Note the **doubled directory name** — easy to hit by tab-completion. |
| `/workspace/hoi-tool-use-checkpoints/grpo-checkpoints/qwen3vl-sft-grpo-4b/global_step_{40,100,140,1000}` | previous-generation 4B |
| `/workspace/hoi-tool-use-checkpoints/sft-checkpoints/qwen3VL-8B` | the **v1 SFT that collapsed** on tool use |
| `results-sft-grpo-qwen3vl-8b-step{25,1000}` | old 8B evals (AR 25.69 / 27.29), pre-re-SFT |

---

## 5. Which eval produced which result dir

| result dir | checkpoint | notes |
|---|---|---|
| `results-sftgrpo/*_grpo_8b` | 8B newsft-resampled step 1000 | the paper's 8B row |
| `results-sftgrpo/*_grpo_4b` | 4B step 165 | the paper's 4B row |
| `results-sftgrpo/*_grpo_4b_step1000` | 4B runB step 1000 | better than step 165 on every metric |
| `results-sftgrpo/hico_ground_a0abl_4b_step200_sub1500` | α=0 step 200 | 1500-sample subset only; the **full**-set run stalled at 1,029/20,028 and has no metrics file |
| `results-sftgrpo/hico_ground_grpo_4b_step165_sub1500` | 4B step 165 | matched partner for the α=0 comparison |

> ⚠️ **Trap in the eval runners.** `run_*_sftgrpo_eval.sh` defaults `CHECKPOINT_PATH`
> to the **8B** checkpoint but `OUTPUT_DIR` to the **4B** `*_grpo_4b_step1000` dirs.
> The 4B evals were run by overriding `CHECKPOINT_PATH` on the command line. If you
> re-run them, pass **both** variables explicitly or you will write 8B outputs into
> 4B-named directories.

---

## 6. Step sensitivity (matters when comparing checkpoints at different steps)

Full-set 4B, step 165 → step 1000:

| benchmark | metric | 165 | 1000 | Δ |
|---|---|---|---|---|
| HICO grounding | AR | 30.35 | 31.22 | +0.87 |
| SWiG grounding | AR | 28.53 | 28.42 | −0.11 |
| HICO referring | exact | 24.96 | 27.27 | +2.31 |
| SWiG referring | exact | 7.42 | **14.63** | **+7.21** |

**Grounding is flat across steps; referring is not.** A step mismatch is defensible
for a grounding-only comparison (the α=0 ablation) and is *not* defensible for
referring. See [[rebuttal-alpha0-ablation]].

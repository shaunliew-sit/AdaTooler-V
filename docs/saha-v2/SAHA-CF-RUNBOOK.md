# SAHA-CF (v3) — Run Guide for H200 / multi-GPU

How to run the **SAHA v3 counterfactual tool-gain reward** (`reward_manager=SAHA-CF`) on a box where the verl/verl-tool training stack already works (your H200 single-GPU, or 8×A100). The authoritative design is `docs/saha-v2/reward-v3-counterfactual-spec.md`; this file is the operational checklist.

> **Why not the 96 GB single-GPU box:** it has only ~62.5 GB system RAM, and a single GPU forces FSDP to CPU-offload the *whole* 4B optimizer, which OS-OOMs at init. On 8×A100 FSDP **shards** params+optimizer across GPUs (no big offload), and H200 nodes have ~1 TB RAM — both fit. The code/recipe are unchanged; it's purely a memory thing on that one box.

---

## What SAHA-CF is (one paragraph)
`R_total = R_format·(R_outcome + α·R_tool)`, where `R_tool = I_tool·clip(s_final − s_ref)` is a **counterfactual tool-gain** grounded only in target-object GT: grounding `s_ref` = the selected proposal anchor scored vs target GT; referring `s_ref` = the detached per-`uid` no-tool mean. One knob `α`. It replaces the Rev-4 6-term reward. `sds_grpo.py` (`SDS-GRPO`) stays the frozen baseline. Full mechanism + reviewer rebuttal in the spec (§1).

---

## 1. Get the code
The SAHA-CF reward is committed on branch `hoi-new`. On the training box:
```bash
cd /path/to/AdaTooler-V && git pull            # or copy these files:
#   verltool/verl_tool/workers/reward_manager/saha_cf.py        (the reward manager)
#   verltool/examples/data_preprocess/hoi/proposal_anchor.py    (grounding anchor builder)
#   verltool/examples/data_preprocess/hoi/prepare_train.py      (emits proposal_anchor)
```
`saha_cf.py` is **auto-registered** — the `reward_manager/__init__.py` globs every `*.py`, so just setting `reward_model.reward_manager=SAHA-CF` works. Verify:
```bash
python -c "from verl_tool.workers.reward_manager import get_reward_manager_cls; print(get_reward_manager_cls('SAHA-CF').__name__)"
# -> SAHACounterfactualRewardManager
```

## 2. Preprocess: add the `proposal_anchor` field (one-time parquet re-run)
The grounding `s_ref` needs the selected proposal anchor in `extra_info`. Re-run `prepare_train.py` (which now emits it) on the box that has the **raw groma data + YOLOE proposals**:
```bash
cd verltool
python examples/data_preprocess/hoi/prepare_train.py \
  --data_dir <groma_benchmarks_simplified> \
  --hico_img_dir <hico images> --swig_img_dir <swig images> \
  --proposals_dir <yoloe proposals> \
  --local_dir data/hoi --filter_len 8192
# verify the field landed:
python -c "import pandas as pd; d=pd.read_parquet('data/hoi/train.parquet'); print('proposal_anchor' in d['extra_info'].iloc[0])"  # -> True
```
If a parquet row lacks `proposal_anchor` (referring rows, or no proposals), `s_ref` falls back safely (referring → group mean; grounding → 0.0 recovery). So you can also run on existing parquet at reduced fidelity, but the re-run is recommended.

## 3. Wire SAHA-CF into your working launcher
Take your **proven** `examples/train/hoi/train_qwen3vl.sh` (the one that runs on your hardware) and change only the reward:
```bash
reward_manager=SAHA-CF
# ...in the python -m verl_tool.trainer.main_ppo args, add:
    reward_model.reward_manager=$reward_manager \
    +reward_model.reward_kwargs.alpha=${ALPHA:-0.6} \
    +reward_model.reward_kwargs.clip_lo=-0.5 \
    +reward_model.reward_kwargs.clip_hi=1.0 \
    +reward_model.reward_kwargs.referring_sref=group_no_tool \
    +reward_model.reward_kwargs.min_no_tool=1 \
```
Everything else (your sync mode, `AdamW8bit`, `do_offload`, FSDP sharding, GPU count) stays as-is. Keep `export PIXEL_REASONER_BBOX_MODE=grid1000` (the zoom-coordinate fix) before launching the tool server.
Defense-in-depth: the manager also reads `SAHA_CF_ALPHA / SAHA_CF_CLIP_LO / SAHA_CF_CLIP_HI / SAHA_CF_REFERRING_SREF / SAHA_CF_MIN_NO_TOOL` env vars if the hydra `+reward_kwargs` is dropped.

**Multi-GPU note:** set `trainer.n_gpus_per_node=8`, `actor_rollout_ref.actor.fsdp_config.fsdp_size=-1` (full shard) — this is what makes 8×A100 fit without the single-GPU CPU-offload.

## 4. The α sweep (the experiment)
On the 4B SFT (`qwen3VL-4B/hoi_v2_sft`) + a balanced subset, run short GRPO runs over the knob and pick the winner:
```
ALPHA ∈ {0.3, 0.6, 1.0}    referring_sref ∈ {group_no_tool, off}    (n=4 if referring s_ref too sparse at n=2)
```
Pick by, in priority order: (1) outcome not degraded (AR/F1, referring score ≥ no-tool baseline); (2) healthy tool-rate (not 0, not always-on; concentrated on hard strata); (3) positive tool-gain on hard strata; (4) `R_tool` share of total reward flat over training (no Peak-Then-Collapse). Report the α curve as the sensitivity ablation (answers reviewer R4). Never tune on test. Then scale the winner to 8B.

## 5. Verify before the full sweep
```bash
# unit tests (use a python that has verl; pytest installed there)
cd verltool && python -m pytest tests/test_proposal_anchor.py tests/test_saha_cf_reward.py -q   # 10 pass
# 1-step smoke on your hardware:
... your launcher ... trainer.total_training_steps=1 trainer.save_freq=-1
# success = console prints per-rollout [r_tool] [s_ref] [tool_gain_clipped] [is_grounding] and step 1 completes.
```
Per-rollout diagnostics logged: `s_final, s_ref, has_sref, r_tool, tool_gain_raw, tool_gain_clipped, i_tool, n_zoom_in, n_zoom_out, is_grounding`.

---

## Troubleshooting — ONLY if your box has the same env drift this repo's box had
These were fixes for a drifted env (transformers 5.5.4 / vllm 0.11 / CUDA 13.1). On a working H200/A100 env you should NOT need them. Apply only if you hit the matching error.

| Symptom | Fix |
|---|---|
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` (vllm rollout) | transformers 5.x ⊥ vllm ≤0.11. Install the shim: `cp verltool/patches/sitecustomize_transformers5_vllm.py $(python -c "import site;print(site.getsitepackages()[0])")/sitecustomize.py` (also fixes Triton/CUDA-13 ptxas). |
| `Triton only support CUDA 10.0 or higher, but got CUDA version: 13.x` | same shim (forces Triton's bundled ptxas). Not needed on CUDA ≤12.8 hosts. |
| `too many values to unpack` in `verl/models/transformers/qwen3_vl.py` `model.visual(...)` | transformers ≥5.0 returns a dataclass, not a 2-tuple. Make the 3 unpack sites (lines ~149/199/244) version-tolerant: `_vis = model.visual(...); a, b = _vis if isinstance(_vis, tuple) else (_vis.last_hidden_state, _vis.deepstack_features)`. Not needed on transformers 4.57.x. |
| Tokenizer `'list' object has no attribute 'keys'` / `config.rope_scaling` is `None` (loading a transformers-5.0.0 checkpoint under transformers 4.57.x) | the checkpoint is 5.0.0-format. Either use transformers 5.x, OR edit the checkpoint: `tokenizer_config.json` set `extra_special_tokens={}`; `config.json` add `text_config.rope_scaling={"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"}` (value from the base Qwen3-VL-4B-Instruct). Tokenization is byte-identical. |
| `AdamW8bit` / `optimizer_impl=bitsandbytes.optim` | `pip install bitsandbytes`. Note: bnb 8-bit ⊥ FSDP on a *single* GPU here (SIGSEGV at optimizer creation); it works under multi-GPU FSDP sharding (your 8×A100). Alternatives verl supports: `optimizer_impl=torchao.optim optimizer=_AdamW override_optimizer_config={bf16_stochastic_round:true}`. |

**Proven combo for reference (from the 8B GRPO that trained successfully): transformers 4.57.6 + verl stock + plain fp32 AdamW + full offload, on a big-RAM / multi-GPU node.**

---

## Reference
- Design / mechanism / reviewer rebuttal: `docs/saha-v2/reward-v3-counterfactual-spec.md`
- Implementation plan + as-built notes: `plan/saha-v3-simple/implementation-plan.md`
- Env recipe details (this box): memory `saha-cf-training-env.md`

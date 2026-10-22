# SAHA — Detector as a Tool (post-ACCV pivot, decided 2026-10-21)

## Why
ACCV 2026 rejected (meta-review): the formulation removes the *discovery* part of open-vocab HOI
detection (YOLOE proposals were injected into the prompt offline), and the gains do not generalise
to SWIG. The prof's direction is to make the detector a **tool**, so the policy decides **when** to
detect. (v1: on the full image only; detect-on-crop deferred, D5.)

## Decisions (user, 2026-10-21; design reviewed by Codex)
| # | Decision |
|---|---|
| D1 | Scope: keep the current tasks (HOI grounding + action referring). The mAP / image-only triplet evaluation is deferred, not dropped. |
| D2 | Detector = live YOLOE-26l prompt-free (`yoloe-26l-seg-pf.pt`, conf 0.3, max 50), run by the tool server at train **and** test time. No offline proposals in the prompt. v1: `detect` always runs on the full original image. |
| D5 | (2026-10-21) Detect-on-crop is **omitted in v1** to get a first run quickly. Revisit only after v1 results. |
| D3 | SFT: reuse the existing 3,547 traces through a deterministic rewrite (no new teacher run), plus paired no-detect grounding traces from rejection sampling with an SFT model (base model unusable; approved 2026-10-22). Then retrain the LoRA with the same configs. Pilot first. |
| D4 | Reward reference = **Option A**: `s_ref` = mean `R_outcome` of same-uid **no-tool** siblings, for grounding as well as referring (reuses `resolve_referring_sref`). No sibling → `R_tool = 0`; log how often that happens. |

## Verified facts this plan rests on
- **Live YOLOE reproduces the offline proposals.** `yoloe-26l-seg-pf.pt` with ultralytics 8.4.183 on 40 random HICO test images gave the same detections and labels on every image. Boxes drift by at most 3/1000; near-tied boxes can swap order. Median latency is 21 ms per image on an H200. So the SFT observations (the old offline JSON) are what the live tool would return. (Probe: `yoloe_repro_probe.py` in the session scratchpad.)
- **`ultralytics==8.4.183` is in `/workspace/envs/verl-tool-env`**, installed with `--no-deps` alongside matplotlib 3.10.9, polars 2.0.0, ultralytics-thop 2.2.3 and nvidia-ml-py.
  - Its declared `opencv-python` dependency is deliberately **not** installed: `opencv-python-headless 4.11` already provides `cv2`, and a second cv2 wheel conflicts. So `pip check` reports that one line.
  - Plain `pip install ultralytics` would pull numpy 2 and opencv 5 and break verl.
- **The tool server already gets the original image.** The agent loop sends `extra_fields["images"]`; `extra_info` is cleared on purpose. A live detector needs no extra plumbing and no GT.
- **SFT traces that depend on the proposals:**
  - Grounding: 2,293 of 2,293 mention them; 2,202 copy a proposal box into the answer.
  - Referring: 462 of 1,254 mention them; 792 never do.
  - 37 traces have two `<answer>` blocks in one gpt turn; drop them.
- **Budget:** a detect observation costs 648 tokens at the median (p90 1,774, max 3,186) with the Qwen3-VL tokenizer. It replaces the same tokens the prompt used to carry, but they now sit in the response. Raise the response/observation length budgets by about 2k tokens over the current launcher.
- **Detector cost:** YOLOE-26l median latency is 209 ms on CPU and 26 ms on GPU (H200), so CPU is viable for the tool server. The launcher now uses `TOOL_WORKERS=4` and `MAX_CONCURRENT=BATCH_SIZE*N`; check throughput in G4.

## Components (minimal change; old pipelines stay runnable as baselines)

### A. `detect` tool — `verl_tool/servers/tools/pixel_reasoner.py`
- Schema: `detect()` with no arguments (like `zoom_out`). It runs YOLOE on the ORIGINAL image (`env['images'][0]`), whatever has been zoomed.
- Observation (text only, no image):
  `Here are the detected object proposals:\n<candidate_objects>\n{json}\n</candidate_objects>`
  with `[{bbox_2d, label, confidence, id}]`, `indent=2`, the same format the prompts used. The SFT rewrite emits exactly this string.
- Boxes are in the original image's 0–1000 grid (normalised by the size of the image the tool receives), so they can be copied straight into `<answer>`. No crop-frame tracking is needed in v1.
- YOLOE is lazy-loaded once per worker process.
  - Env vars: `DETECT_TOOL_WEIGHTS` (default `/workspace/saha-hoi/yoloe-26l-seg-pf.pt`), `DETECT_TOOL_DEVICE` (default `cpu`; set it to a GPU id when memory allows), `DETECT_TOOL_CONF=0.3`, `DETECT_TOOL_MAX=50`.
- Add `detect` to `valid_mcp_func_names` and the dispatch. zoom_in/zoom_out are unchanged.
- The tool description says that detect boxes are full-image 0–1000 coordinates.
- Return an explicit observation on error (e.g. detector failure) and mark the call `valid=False`, so attempted and successful calls can be counted separately.

### B. Prompts — one shared converter, `examples/data_preprocess/hoi/detect_tool_prompt.py`
- `SYSTEM_PROMPT_DETECT`: the current system prompt plus a `detect` function schema.
- `convert_user_prompt(text) -> (new_text, proposals_json)`: removes the `<candidate_objects>`/`<proposals>` block and its framing sentences, and rewords "candidate proposals" to "objects returned by `detect` (if you call it)". The task, the answer format and the referring person/object boxes stay. One function serves:
  - B1: migrating the GRPO parquets (`data/*.parquet` → `data/*_detect.parquet`; the originals are kept). `extra_info.proposal_anchor` stays, for EVAL/metric stratification only.
  - B2: the SFT rewrite (C).
  - B3: the eval prompts (E).

### C. SFT data — `examples/data_preprocess/hoi/sft_gen/convert_to_detect_tool.py`
For each trace:
- Replace `system` and the first human turn using B.
- **Needs detect** (every grounding trace, plus referring traces whose gpt text mentions the proposals):
  - Insert a gpt turn: `<think>{short templated reason}</think>\n<tool_call>\n{"name": "detect", "arguments": {}}\n</tool_call>`
  - Then an observation turn holding the A-format string built from the offline JSON.
  - The observation is text only, so the image list and zoom `target_image` indices are unchanged.
- **Otherwise** (792 referring traces): no detect turn.
- Drop the 37 double-answer traces.
- **C2, paired no-detect grounding traces:** run the base Qwen3-VL-8B (vLLM, the B prompt, tools offered but not required) on the SFT grounding prompts; keep GT-scored answers (the live grounding scorer, outcome ≥ 0.5) that call no tool. Pair them with the same prompt's detect trace, stratified by dataset and `s_ref` tier; aim for about 25–35% no-detect on grounding. Hold out a validation split.
- **Split by ORIGINAL IMAGE before pairing** (train/val), so detect and no-detect versions of the same image, or repeated queries, never straddle the split.
- Output `/workspace/data/hoi_sft_detect/`, registered in LlamaFactory `dataset_info.json`. Same LoRA YAMLs with `dataset` **and** `dataset_dir` changed. Pilot (4B, small subset) before the full run.
- Audits: no-GT-leak check; detect-rate per task; tool-call JSON validity; observation formatting matches the RL agent loop's wrapping.

### D. Reward — `verl_tool/workers/reward_manager/saha_cf.py` (`sds_grpo.py` stays untouched)
- Add `count_detect`. Set `I_tool = 1` if there is any zoom_in, zoom_out or detect call. Log `n_detect` per rollout.
- New option `SAHA_CF_GROUNDING_SREF=group_no_tool`, the default for detect-tool runs: grounding rows use `resolve_referring_sref` like referring. `anchor` stays selectable for the old pipeline.
- Same form `R_format·(R_outcome + α·R_tool)`, still one knob α.
- **Keep the strata separate from the reward reference.** The existing hard/easy metrics are keyed on `s_ref`, and Option A changes its meaning (it is also undefined for no-tool rows). Log the old proposal-anchor score as a separate `s_anchor` from `extra_info.proposal_anchor`, and key the hard/easy strata on `s_anchor`.
- Count calls from the tool execution records (attempted vs valid), not by regex alone; regex counts are a fallback.
- Metrics (`metric_util.compute_saha_cf_metrics`): add `detect_rate{,_grounding,_referring}`, `zoom_rate`, and `sref_missing_frac` (tool rows with no no-tool sibling). If `sref_missing_frac` stays high, raise it with the user; Option B (a forced no-tool rollout) is the fallback.

### E. Eval — `/workspace/hoi-benchmarks` (separate repo; coordinate, another agent may be working there)
- The eval harness builds its own prompts with proposals. It needs a detect-tool mode: the B3 prompt, plus the same `detect` implementation (import it or copy the tool logic with the frame tracking).
- Report detect rate and zoom rate per stratum alongside the existing metrics, on HICO and SWIG.
- Comparison rows: no-proposal base; old proposal-injected SAHA; detect-tool SFT; detect-tool SFT+GRPO.

### F. Training launcher — `examples/train/hoi/run_grpo_detect_tool.sh`
A copy of `run_grpo_v2_alpha0.sh` with:
- the `*_detect.parquet` data
- `SAHA_CF_GROUNDING_SREF=group_no_tool`
- the `DETECT_TOOL_*` env vars
- the new SFT checkpoint
- ALPHA taken from the env (default 0.6)

### G. Tests (smoke before any long run)
1. **Detector equivalence through the real transport:** send the image the way the agent loop does (encoded data URL → decode) and run detect. On ≥500 HICO + SWIG test images, compare with `test_proposals` (same detection set, boxes within 3/1000; report how often the order or ids change).
   The tool receives the dataset-resized image re-encoded as JPEG (`encode_image_url`), not the original file, so the test must use that path.
2. Converter: round-trip on 20 SFT traces, checked for prompt/obs format and the indices.
3. Reward unit tests:
   - a detect-only rollout counts as tool
   - grounding s_ref from siblings
   - R_tool is 0 when there is no sibling
4. A one-step GRPO smoke run with the tool server running YOLOE.

## Order
A → G1 → B → **E (eval detect-tool mode)** → C (+ audits) → C2 → SFT pilot (evaluated with E, including an **always-detect** comparison row) → D → G3 → F → G4 → full runs.
Each step gets an independent Codex check through Herdr before moving on.

## Explicitly NOT doing now
- mAP / image-only HOI detection (D1)
- a text-prompted detector
- a per-call cost term (it would add a second knob)
- a forced no-tool rollout (Option B)

## Run log 2026-10-21 (unattended run; decision trail in `.audit/saha-detect-tool.tsv`)
- **A done.** The `detect` tool runs YOLOE on the ORIGINAL file, `extra_info.detect_image`, which the agent loop forwards to the tool server.
  - The agent loop's image copy is resized and JPEG re-encoded. On that copy YOLOE gave 37.5% recall vs the offline proposals; on the original file 99.2% recall and precision (G1, 500 images). Gate: `smoke_test_detect_tool.py` exits 1 below 95%.
  - The 0–1000 conversion is `round(px / size * 1000)`, which is what reproduces the offline files. The `(size-1)` formula in today's saha-hoi code gave boxes above 1000.
- **B done.** Outputs: `detect_tool_prompt.py`, `migrate_to_detect_tool.py`, and `data/{val,train_resampled_moderate}_detect.parquet`.
  - The zoom_in schema wording now says 0–1000 (the data's old text said native pixels).
- **C done, with one deviation.** No-detect grounding answers are sampled from the **SAHA SFT 8B** (`qwen3vl-8b-saha-sft-merged`), not the base model.
  - Pilot on 64 prompts: the base model called a tool on 256/256 samples when tools were offered, formatted 0/64 without tools, and gave 8/64 with a `<think>` prefill, still talking about detecting. The SAHA SFT gave 35/64.
  - Prompt for the sampler: tool guideline and tool schema removed. Filters: format, minAR10 ≥ 0.5, and no tool/detect/proposal talk.
  - Result: `/workspace/data/hoi_sft_detect`, 4,112 train / 398 val, split by original image. No-detect is 31% of grounding.
- **Env fix (LlamaFactory).** Orphan CUDA-13 nvidia wheels had overwritten the cu12 cuDNN libraries (`CUDNN_STATUS_NOT_INITIALIZED` on driver 550). I removed them and force-reinstalled the cu12 wheels.
- **SFT.** 8B LoRA, `LlamaFactory/examples/train_lora/qwen3vl_8b_saha_detect_sft_lora_1gpu.yaml`, GPU 0. Merge with `examples/merge_lora/qwen3vl_8b_saha_detect_merge.yaml` → `/workspace/data/qwen3vl-8b-saha-detect-sft-merged`.
- **D done.** `SAHA_CF_GROUNDING_SREF=group_no_tool`; detect counts in `I_tool`; `s_anchor` keys the strata. Tests: `examples/train/hoi/test_saha_cf_detect.py`; the old `smoke_test_saha_cf.py` still passes.
- **F written.** `examples/train/hoi/run_grpo_detect_tool.sh`. `train_qwen3vl_v2.sh` gained `MAX_RESPONSE_LENGTH` and `grounding_sref` passthroughs; the defaults are unchanged.
- **SFT done** (12:28, 2h24m). train_loss 0.512, final eval_loss 0.487. The merge writes transformers-5 tokenizer/config files that crash verl's transformers 4.57 (`extra_special_tokens` is a list). Those were moved to `_llamafactory_tok_bak/` and the base model's non-weight files copied in, as was done for `qwen3vl-8b-saha-sft-merged`. Weight keys equal the base (750).
- **First-move check** (`check_sft_detect_policy.py`, 200 val prompts ×4, T=1): grounding detect 94.8% / answer 5.2%; referring answer 66.1% / detect 30.5% / zoom_in 3.2% / malformed 0.2%. Grounding detects more than the SFT mix (69%). At 95% only ~27% of N=6 grounding groups contain a no-tool sibling (Codex), so most grounding detect rows would get `R_tool=0`. **Open, for the user:** accept, or raise no-detect exploration (N, the SFT mix, or a reference fallback).
- **Env fix (verl).** The GRPO recipe uses `optim=AdamW8bit` from `bitsandbytes.optim`; the lockfile env lacked it. Installed `bitsandbytes==0.48.2 --no-deps` (deps already satisfied) and added it to `verl_tool_env.lock.txt`.
- **Reward bug, pre-existing (fixed).** verl's agent loop scores each rollout alone (batch of 1, `agent_loop.py` ~496, whenever `reward_model.enable=False`), and `saha_cf.__call__` returned those `rm_scores` on the trainer's full-batch call. So no same-uid sibling was ever visible: every sibling `s_ref` was missing and `R_tool=0`. This also hit **referring `group_no_tool` in all earlier SAHA-CF runs** (e.g. `runA-minAR10-bigbatch`: 15/15 referring tool rows with a no-tool sibling had `has_sref=0`). Fix: drop the early return; the full-batch call recomputes. Regression test `test_full_batch_scoring_ignores_per_rollout_agent_loop_scores` fails on the old code. Codex PASS.
- **G4 GRPO smoke, 1 step, GPU 0: PASS.** `logs/grpo_detect_smoke.log`, records `verl_step_records/SAHA-CF-a0.6-detect-tool-8b-smoke-minar10/1.jsonl`. Live detect calls return YOLOE JSON; `sref_missing_frac` 0.0; detect_rate grounding 0.75 / referring 0.25; R_tool signs follow the sibling gap; 152 s/step. One checkpoint is **82 GB** (model+optimizer+hf_model); `/workspace` is at 100% (82 GB free).

## Decisions 2026-10-22 (user)
- The old proposal-injected runs are dropped. New training and eval use the detect-tool SFT model, so the pre-existing reward bug needs no cleanup for past results.
- Sampling no-detect traces with an SFT model is approved. Re-sampling, if needed, uses `qwen3vl-8b-saha-detect-sft-merged`.
- Keep the current SFT for now. Run a 20-step GRPO pilot (GPU 0, `DATA_TAG=detect-tool-8b-pilot`, `SAVE_FREQ=20`) and watch `sref_missing_frac` and `detect_rate_grounding`. If most grounding tool rollouts still lack a reference, retrain the SFT with more no-detect grounding traces.
- The smoke checkpoint was deleted. `train_qwen3vl_v2.sh` takes `SAVE_FREQ` (default 5); only the latest actor checkpoint is kept, but a save briefly needs about 2×82 GB.
- hoi-benchmarks detect-tool mode waits until the eval requirements are clear.

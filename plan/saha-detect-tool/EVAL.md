# Eval requirements: detect-tool mode in hoi-benchmarks (draft 2026-10-22)

Status: APPROVED 2026-10-22 (user), including max-model-len 16384 in detect mode and the never-/always-detect rows. Nothing in `/workspace/hoi-benchmarks` has been edited.

## Goal
Evaluate the detect-tool models with exactly the previous protocol. Only two things change:
1. The prompt has no proposals, and a `detect` tool is offered.
2. Tool rates are reported per task, with detect counted separately.

Anything not listed under "What changes" stays byte-for-byte as it is today.

## What stays the same (user decision)
| Item | Today (verified in hoi-benchmarks) |
|---|---|
| Benchmarks / splits | HICO + SWIG; grounding + action referring. Annotations: `/workspace/Groma/groma_data/benchmarks_simplified/{hico,swig}_{ground,action_referring}_test_simplified.json`. Images: HICO `test2015`, SWIG `images_512`. The sub1500 ablation subsets stay as well. |
| Grounding metrics | AR over IoU 0.5–0.95 (pair IoU = min(person, object), i.e. minAR10), AR@0.5, AR@0.75, ARs/ARm/ARl |
| Referring metrics | BLEU, METEOR, ROUGE_L, CIDEr, exact_match |
| Decoding | vLLM OpenAI server, temperature 0, max_tokens 4096 per turn |
| Turn limit | `MAX_TURNS=6`, as in the run scripts today |
| zoom_in / zoom_out | Unchanged: crop the original image from 0–1000 coords; zoom_out is text-only |
| Rows | Base, SFT, GRPO, plus the zoom ablation rows (adaptive / never / always / random). New SFT and GRPO rows come from `qwen3vl-8b-saha-detect-sft-merged` and the new GRPO run. |

## What changes
1. **Prompt.** The system prompt gains the `detect` schema, and the user prompt loses the proposals block.
   - Single source of truth: generate the text byte-exact from the training data, as `_gen_sftgrpo_prompts.py` already does for the old prompts. The source is `AdaTooler-V/verltool/data/train_resampled_moderate_detect.parquet`, and the output is a new `sftgrpo_prompts_detect.py`. No hand-written prompt text.
2. **detect tool at test time.**
   - Runs YOLOE-26l prompt-free (`yoloe-26l-seg-pf.pt`, conf 0.3, max 50) on the ORIGINAL image file, never on the resized copy (on the copy, recall drops to 37.5%; see PLAN run log).
   - Copy the TRAINING implementation verbatim (`AdaTooler-V/verltool/verl_tool/servers/tools/pixel_reasoner.py`: `_bbox_to_1000`, `detect_objects`, `format_detect_observation`) into `hoi-benchmarks/detect_tool.py`. Do NOT reuse `efficiency_breakdown.detect_live`: it scales float pixels directly, while training first rounds to integer pixels (x=1.3, w=512 gives 3 vs 2). It also passes a filename rather than a PIL RGB image and relies on the implicit device. `efficiency_breakdown.py` also has uncommitted edits from the other agent.
   - Observation text: `"Here are the detected object proposals:\n<candidate_objects>\n" + json + "\n</candidate_objects>"`, text only, no image.
   - The detector is loaded once per eval process, on the eval GPU.
3. **Unknown tool names.** Today an unknown tool name silently falls through to a "zoomed-in view" observation, so `detect` would be mis-executed without anyone noticing. The detect branch must come before that fall-through. An unknown name should be logged and counted as `invalid_tool`.
4. **Context length.** The detect JSON adds roughly 1–2.5k tokens per call, and the vLLM server runs with `--max-model-len 8192`. Training used 8192 prompt + 6144 response. Raise the limit to 16384 in detect mode only, so answers are not cut off. **This is the one deliberate departure from "same as before".** Old rows are not re-run.
5. **Tool-rate reporting.** Per task, in `_metrics.json` and the summary table:
   - `detect_rate`: the fraction of samples with ≥1 executed detect.
   - `zoom_rate`: unchanged definition.
   - `any_tool_rate`, `detect_calls_per_sample`, `zooms_per_sample`, `invalid_tool`.
   - Per sample: `num_detect`, `num_zoom_in`, `tool_calls` (in order).
   - Also accuracy split by tool use (`acc | detect` vs `acc | no tool`). This mirrors the training metric `saha_cf/acc_tool_minus_notool`.
6. **Optional rows (recommended, cheap).** `never-detect` and `always-detect` (detect forced on the first turn), through the existing `ZoomAblationPolicy` pattern. They show whether the model's choice to call detect beats both fixed policies, which answers the reviewer's "discovery" point. The PLAN already lists always-detect.

7. **Rows (complete list).** Each model row is re-run in detect mode: Base (Qwen3-VL-8B-Instruct), the new SFT and the new GRPO. The policy rows on the full test sets are adaptive (the model decides), never-detect and always-detect. The zoom ablation rows (adaptive / never / always / random) are re-run on the same `sub1500_seed42` subsets, for the new GRPO. The old proposal-injected rows stay as they are, as the reference.
8. **Outcome by tool use.** Training's comparison is any-tool vs no-tool. Report per task, split by (no tool / any tool / detect / zoom):
   - Grounding: AR and AR@0.5.
   - Referring: METEOR, BLEU and exact_match. CIDEr stays in the overall metrics only (see "Detect-mode protocol changes").
   - Count attempted calls (a parsed `<tool_call>`) separately from valid ones (executed with no error).

## Known eval-vs-training differences (preserved, documented, not changed)
These were in the old protocol too; changing them would break comparability with the old rows.
- **Turns.** Eval runs 6 generations and can execute a tool on the last one. Training (`max_turns=5`) executes tools only before the final generation.
- **Decoding.** Eval uses T=0 and up to 4096 tokens per turn. Training used T=1, 2048 tokens per action, a 6144 total response budget and an 8192-token observation limit.
- **zoom_in.** Eval crops the original image unpadded and ignores `target_image`, and sends JPEG q90. Training crops with padding by image index and uses the default JPEG quality.
- **History.** Grounding eval prunes old images to "[zoomed image]"; training keeps them. Referring eval keeps history.
- **Message framing.** Eval observations are plain user messages, without SFT's `<tool_response>` wrapper. For detect, check in the smoke that the model still reads the observation correctly. If it does not, add the wrapper in detect mode only and record why.

### Detect-mode protocol changes
These apply only with `--tool-mode detect`. `--tool-mode proposals` is unchanged, so the old rows are unaffected.
- **Grounding image pruning** (`detect_tool.prune_to_current_view`). The proposals-mode rule keeps an image only in the last user message. A detect observation is a text-only user message, so that rule would remove every image and the model would answer blind. The detect-mode rule keeps the current view in the last user message that carries an image, and older images become "[zoomed image]" as before.
- **zoom_out changes too.** In proposals mode, the text-only zoom_out observation removes every image. In detect mode, the original image comes back, in the last image-carrying message.
- **Stale crop caption.** When that message was a zoom_in observation, its image is replaced by the current view (the original image after a zoom_out), but its caption "Here is the zoomed-in view of the requested region." stays.
- **Referring outcome split.** The per-group split (item 8) reports METEOR, BLEU and exact_match. It does not report CIDEr: CIDEr recomputed within a group uses that group's document frequencies, so its values are not comparable across groups. The overall metrics keep CIDEr.
- **always-detect.** When the forced detect replaces the model's turn 0, the discarded turn is kept per sample (`detect_stats.replaced_model_turn`) and counted as `forced_replaced_model_turn_total`, by attempt kind in `replaced_model_attempts`. The injected executions are counted apart, as `detect_injected_total`.
- **Server check.** Each evaluator reads the server's `/v1/models` and stops unless it serves `--expected-model` (as its `root`, the loaded checkpoint; its `id` only when there is no `root`) with `max_model_len` >= 16384.

## How to build it with minimal disruption
- A new flag, `--tool-mode {proposals,detect}`, defaulting to `proposals`. It goes in the 4 eval scripts (`eval_{hico,swig}_{ground,action_referring}_sftgrpo_qwen3vl.py`), and the default keeps the old behaviour identical.
- A new driver, `run_detect_eval.sh`, covers the 4 tasks and writes to `results-detect-tool/{task}_{sft,grpo}_8b`. It does NOT edit the existing `run_*_sftgrpo_eval.sh`, which have uncommitted edits from the other agent.
- The 4 eval files are near-duplicates, so the detect branch and its counters go into one small shared module (`detect_tool.py`) that all 4 import.
- The detect prompts come from a NEW generator script writing `sftgrpo_prompts_detect.py`. Never run `_gen_sftgrpo_prompts.py` unchanged: it overwrites `sftgrpo_prompts.py`.

## Acceptance checks (before any real eval)
1. **Equivalence.** On 200 test images (HICO and SWIG), the eval's detect observation string must be byte-identical to `pixel_reasoner.format_detect_observation(pixel_reasoner.detect_objects(path))`, with the same weights and device. Report the number of exact matches; anything below 200/200 is a FAIL to explain.
2. **Prompt equality.** The generated detect prompts must equal the `*_detect.parquet` prompts byte-for-byte.
3. **Regression.** `--tool-mode proposals` on 50 samples must give results identical to the current code.
4. **Smoke.** 20 samples per task with the new SFT model must show detect calls executed (not falling through), non-zero `detect_rate`, and no truncation errors.
5. A Codex check through Herdr.

## Decisions (user, 2026-10-22)
- Raise `--max-model-len` to 16384 in detect mode. This is an explicit protocol change for the new rows only. Training measured the detect observation at a median of 648 and up to about 3.2k tokens. The smoke also logs how often 8192 would have truncated.
- Add the never-detect and always-detect rows.

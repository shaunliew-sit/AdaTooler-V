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
   - Reuse `efficiency_breakdown.py` `load_detector` / `detect_live` (same `round(px*1000/size)` grid as training) and `format_proposals` (same JSON as the training tool).
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

## How to build it with minimal disruption
- A new flag, `--tool-mode {proposals,detect}`, defaulting to `proposals`. It goes in the 4 eval scripts (`eval_{hico,swig}_{ground,action_referring}_sftgrpo_qwen3vl.py`), and the default keeps the old behaviour identical.
- A new driver, `run_detect_eval.sh`, covers the 4 tasks and writes to `results-detect-tool/{task}_{sft,grpo}_8b`. It does NOT edit the existing `run_*_sftgrpo_eval.sh`, which have uncommitted edits from the other agent.
- The 4 eval files are near-duplicates, so the detect branch and its counters go into one small shared module (`detect_tool.py`) that all 4 import.

## Acceptance checks (before any real eval)
1. **Equivalence.** On 200 test images, the eval's detect observation must equal the training tool server's output (`pixel_reasoner.format_detect_observation`) byte-for-byte, or match ≥ 99% of boxes, as in G1.
2. **Prompt equality.** The generated detect prompts must equal the `*_detect.parquet` prompts byte-for-byte.
3. **Regression.** `--tool-mode proposals` on 50 samples must give results identical to the current code.
4. **Smoke.** 20 samples per task with the new SFT model must show detect calls executed (not falling through), non-zero `detect_rate`, and no truncation errors.
5. A Codex check through Herdr.

## Decisions (user, 2026-10-22)
- Raise `--max-model-len` to 16384 in detect mode.
- Add the never-detect and always-detect rows.

# HOI SFT trace regeneration (gpt-5.5 + Responses API)

Rebuilds the SFT dataset to fix the tool-collapse SFT ceiling (copy-only/GT-narration,
override-of-good-proposals, target_image drift). Plan: `plan/saha-sft-regen/PLAN.md`.

## Files
- `prompts.py` — `GEN_DEVELOPER_PROMPT` (generation-only steering: per-turn `<think>`,
  trust-good-proposals/anti-override, zoom-discipline) + Responses tool schema + obs strings.
- `crop.py` — reuses the live `pixel_reasoner.crop` (grid1000) so SFT crops == RL crops.
- `curate.py` — keep-gates: no-GT-leak, per-turn `<think>`, outcome≥0.5 (live scorers),
  GT-centric anti-override, and **tool-gain** (a grounding TOOL trace must beat the proposal
  anchor `s_final>s_ref` — rejects zoom-then-copy / pointless zoom; empty-anchor tool traces
  are rejected since `s_ref` is then unreliable). NO-tool grounding copies of good proposals are
  kept. **Referring tool traces are outcome-gated only** (no offline `s_ref`), so a zero-gain
  referring zoom can pass — acceptable for the pilot; revisit if referring tool-rate looks padded.
- `generate.py` — seeds from `data/train.parquet`, runs the Responses tool loop
  (reasoning=high, `previous_response_id` chaining), assembles ShareGPT, writes dataset.
- `probe_openai.py` / `probe_responses.py` — API de-risk probes (already run).

## Run
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate verl-tool-env
cd /workspace/AdaTooler-V/verltool

# 1) SMOKE TEST (3 samples, 1 worker) — inspect output before scaling
python examples/data_preprocess/hoi/sft_gen/generate.py --smoke 3

# 2) PILOT (100, hard-weighted, concurrent) — read stats.json (yield, cost, sref tiers)
python examples/data_preprocess/hoi/sft_gen/generate.py --n 100 --task both --hard_frac 0.6 --workers 8
```
Output: `/workspace/data/hoi_sft_regen_v2/{dataset.json, images/<idx_uid>/N.jpg, stats.json}`.
Reads `OPENAI_API_KEY` from `/workspace/AdaTooler-V/.env`. **Spends real credits — pilot first.**

For the full 10k run, switch to the **Batch API** (50% off) once the pilot validates yield/cost
(the loop logic is identical; batch submits each round as a JSONL and polls).

## Caveats / TODO before full run
- **LLaMA-Factory registration (B3):** `dataset.json` uses a top-level `system` field +
  `conversations`(human/gpt/observation) + relative `images` paths. Add a `dataset_info.json`
  entry mapping `system`/`messages`/`images` and point `image_dir` at the dataset root.
- **Thread safety:** one shared OpenAI client across worker threads (fine for httpx; if flaky
  transport errors appear, switch to one client per thread).
- **Batch + previous_response_id:** verify chaining works across separate batch rounds during
  the pilot before committing the 10k batch run.
- After generation, run the copy-rate / override / s_ref-coverage audits on the NEW dataset (B4)
  to confirm the defects are fixed before any SFT.

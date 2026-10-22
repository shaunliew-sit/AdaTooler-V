# SAHA-CF SFT Trace Regeneration — Plan (gpt-5.5 + Batch API)

**Status:** PLAN ONLY — awaiting human approval before execution (CLAUDE.md §10.0 gate).
**Date:** 2026-06-18. **Author:** analysis + codex-verified.

---

## 1. Why we are doing this

The tool-use collapse is now traced (memory `saha-cf-tool-collapse-root-cause`, Layers 1–10) to a
**reused-v1-SFT capability/habit ceiling**, not a reward/coord/template bug. Three concrete SFT defects:

1. **Copy-only / GT-narration.** 70.6% of tool answer boxes are verbatim proposal copies; "derived"
   boxes were GT-conditioned, so the model learned to *narrate* derivation, not *do* it. At RL time
   (no GT) its freehand boxes are wrong (`s_final 0.41` when it derives).
2. **Override-of-good-proposals.** After a (correct) zoom, the model discards a geometrically perfect
   proposal and re-estimates a wrong box: 38% of derived answers override a good proposal and miss GT.
3. **Tool-schema drift.** Old SFT tool calls omit `target_image`; live RL requires it → 5% of live
   tool calls rejected `Missing parameters: target_image`.

**Goal of the regen:** a new SFT dataset that teaches the *discrimination* the reward wants —
**trust a good proposal (copy it), zoom + genuinely derive only when the proposal is weak/missing,
and never override a good proposal** — with prompts/format byte-aligned to the *current RL pipeline*
(not the old SFT), so SFT↔RL drift is removed and changes are minimal.

---

## 2. Key decision: seed from the existing `train.parquet`

Data-availability audit result: the canonical inputs are **already in
`/workspace/AdaTooler-V/verltool/data/train.parquet`** (281,408 rows):
- verbatim RL **system+user prompt** (incl. `<tools>` block, `<candidate_objects>`/`<proposals>`),
- **image path** (100% resolve on disk under `/workspace/data/...`),
- **GT** (`reward_model.ground_truth`: `boxes_1000`, `num_pairs`, or referring `response`),
- **proposals** (embedded in the prompt) and **`extra_info.proposal_anchor`** + computable `s_ref`.

Raw TRAIN YOLO proposal files are absent but **not needed** (embedded). Crop images are not stored;
we render them at generation time with the **fixed `grid1000` `pixel_reasoner` crop** so SFT crops ==
RL crops exactly.

**Consequence:** seeding from the parquet gives "minimal change" for free — the SFT prompt IS the RL
prompt. We only generate the assistant turns.

---

## 3. Output format (align to CURRENT RL, ShareGPT for LLaMA-Factory)

Per-sample ShareGPT object `{"conversations":[...], "images":[...]}`:
- **system** turn = RL system message verbatim (the `<tools>` block with `zoom_in`/`zoom_out`,
  `target_image` required). *(Use LLaMA-Factory ShareGPT `system` field or a `system`-role turn.)*
- **human** turn = RL user message verbatim (`<image>\n` + task + proposals + guidelines).
- **gpt** turns = `<think>…</think>` then either `<tool_call>{...}</tool_call>` (compact JSON) or final `<answer>…</answer>`.
- **observation** turns = tool results, using the **exact pixel_reasoner strings**:
  - zoom_in → `"<image>\nHere is the zoomed-in view of the requested region."`
  - zoom_out → `"Returned to the full original image view."`
- **tool_call schema (RL-aligned, fixes drift):**
  - `{"name":"zoom_in","arguments":{"bbox_2d":[x1,y1,x2,y2],"target_image":1}}` — **include `target_image`**, coords 0–1000.
  - `{"name":"zoom_out","arguments":{}}`.
- **answer:** grounding = one `[{person},{object}]` JSON array **per line** (≤2 objs/line) or
  `no valid pairs found`; referring = bare `"{verb}ing {object}"`.
- **images invariant:** `len(images) == 1 + (#zoom_in)`; zoom_out adds none; index 0 = original, then one crop per zoom_in (paths under the new dataset dir).

Format reward gate (must pass for the trace to be RL-useful): final message must match
`<think>…</think>.*<answer>…</answer>` (DOTALL). Grounding scoring = avg Recall@IoU{0.5,0.75},
pair_score = 0.5·IoU_person + 0.5·IoU_object (**both boxes must be good**). Referring = exact / ROUGE-L+METEOR.

---

## 4. Generation architecture — iterative Batch rounds (codex-verified CORRECT)

Batch API is one-shot, so a multi-turn tool loop = **N sequential batch rounds** (original used 7):

```
seed conversations from parquet rows (NO GT in any field)
for round k in 0..MAX_ROUNDS(=6):
    build one JSONL: one request per in-flight conversation, custom_id = row uid
    submit batch (model=gpt-5.5, tools=[zoom_in,zoom_out], reasoning.effort=high)
    poll until complete; match results by custom_id (order NOT guaranteed)
    for each result:
        parse <think> + (tool_call | answer)
        if tool_call: validate bbox_2d; render crop (grid1000 pixel_reasoner);
                      append crop as a NEW user message w/ image_url (see gotcha);
                      conversation stays in-flight
        else (answer or no-think or malformed): retire conversation
    checkpoint completed custom_ids to disk (resume-safe vs 24h expiry)
```

Then **offline curation/rejection-filter** (§6) → convert kept traces to ShareGPT → write dataset.

---

## 5. Technical decisions (with codex verdicts + fixes)

| # | Decision | Verdict | Action |
|---|---|---|---|
| 1 | Iterative batch rounds, match by `custom_id`, checkpoint completed | CORRECT | implement resume on `custom_id`; batch canary before each big round |
| 2 | API surface | **RESOLVED → Responses API** | Probe (`sft_gen/probe_openai.py`): Chat Completions **rejects** `reasoning_effort`+function tools for gpt-5.5 ("use /v1/responses"). Responses API works (`sft_gen/probe_responses.py`): reasoning=high + tools + image OK. Multi-turn via **`store=True` + `previous_response_id`** confirmed (send only the new crop each round; prior context reused + prompt-cached). Alt: `store=False` + `include=["reasoning.encrypted_content"]`. **Pilot run SYNC** (concurrent, fast iteration, no 24h wait); **full 10k run BATCH** (50% off). |
| 3 | Visible `<think>` for the SFT trace | **RESOLVED → enforce per-turn** | At effort=high the visible `<think>` is THIN (~150 chars; real reasoning hidden) and is **ABSENT on the final answer turn** (model emitted `<answer>` only). RL format gate needs BOTH `<think>` and `<answer>` in the final turn. Fix: prompt to emit a visible `<think>` on EVERY assistant turn; enrich via `reasoning.summary:"detailed"`; **hard post-filter rejects any turn missing `<think>`** (same empty-output failure seen in RL). |
| 4 | Images across rounds | RISKY | **use OpenAI File API file-ids / HTTPS URLs, not inline base64** (batch file cap 200MB); `detail:"low"` for carried-forward crops, `"high"`/`auto` for the active crop; monitor per-round JSONL size |
| 5 | No GT in prompt; rejection-filter on GT offline | CORRECT | audit every request field (system/user/tool) for GT leakage before submit |
| 6 | Cost | RISKY | **100-sample pilot** to measure `usage` p50/p95, then extrapolate |

**Codex gotchas folded in:** Chat Completions `role:"tool"` is text-only → return the crop as a new
`role:"user"` image message; validate `bbox_2d` before rendering (malformed args crash a whole round);
handle parallel tool_calls in one message; newer models may need `developer` role not `system`
(pilot will reveal); expired batches still bill completed reqs → checkpoint; rejection filtering
shifts distribution → track acceptance rate **stratified by `s_ref` tier**; one model per batch file.

---

## 6. Curation — engineer the behaviors we want (the whole point)

Generation alone won't fix the defects; **curation must enforce the target behavior**:

1. **Anti-leakage:** GT never in any request field.
2. **Format gate:** keep only traces with non-empty `<think>` + valid `<answer>` (+ RL format regex).
3. **Outcome gate (rejection filter):** keep grounding traces with `s_final ≥ τ` vs GT (same
   `compute_grounding_outcome`); referring with exact/high score.
4. **Anti-override gate:** on samples where a proposal already covers the target well (`s_ref` high),
   keep traces whose answer **matches the good proposal** (copy), and **drop** traces that overrode it
   into a worse box. On weak-proposal samples (`s_ref ≤ 0.5`), keep traces where the zoom produced a
   **derived box that beats the anchor** (`s_final > s_ref`) — these are the genuine tool-gain traces
   the reward needs.
5. **Curriculum (matches the GRPO resample):** weight **tool-using** traces toward `s_ref ≤ 0.5`
   (the band where the tool has upside), keep enough easy/no-tool traces to teach "don't zoom when the
   proposal is perfect." Track kept-distribution by `s_ref` tier so SFT isn't skewed to lucky easy wins.

---

## 7. Cost & pilot

Codex ballpark: ~**$1,400** for ~10k samples × ~5 rounds at batch pricing ($2.5/$15 per 1M), main
driver = re-sent image tokens; **could be 3–5× higher** if images/detail uncontrolled. → **Mandatory
100-sample pilot** measuring real `usage` before any full run.

**Probe usage data points (sync, simple 1-proposal case):** round1 in=555/out=250/reasoning=171;
round2 (after crop, via previous_response_id) in=905/out=566/reasoning=516. Real samples have larger
prompts (full proposal list) + up to ~6 rounds → pilot will give p50/p95. `previous_response_id`
re-bills prior tokens but at the prompt-**cached** rate (cheaper); detail:low on carried crops helps.

---

## 8. Proposed scope (OPEN — needs your call, §10)

- Original: 10k attempted → 8,773 kept (3,881 grounding + 4,892 referring; hico 2,921 + swig 5,852).
- Recommendation: **pilot 100 (50 grounding hard `s_ref≤0.5` + 25 grounding easy + 25 referring)**,
  then a **~10k full run weighted toward grounding `s_ref≤0.5`**. Reuse v1's no-tool easy traces where
  still valid to cut cost (they were not the defect).

---

## 9. Work breakdown (after approval)

- **B0 Pilot harness** (`examples/data_preprocess/hoi/sft_gen/`): parquet→requests, batch submit/poll,
  crop renderer (import fixed pixel_reasoner), curation, ShareGPT writer. Run 100-sample pilot.
- **B1** Decide Chat Completions vs Responses from pilot; lock cost.
- **B2** Full generation run (iterative batches) with checkpointing.
- **B3** Curation + dataset assembly → `/workspace/data/hoi_sft_regen_v2/` (+ `dataset_info.json`).
- **B4** Sanity: re-run the copy-rate / override / `s_ref`-coverage audits on the NEW dataset to confirm
  defects are fixed before any SFT.
- **B5** (separate gate) SFT training on the new dataset.

Output dataset dir: **`/workspace/data/`** (per request) — e.g. `/workspace/data/hoi_sft_regen_v2/`.

---

## 10. Open decisions for you (approval gate)

1. **Scope/size** — pilot-100 then ~10k weighted to hard grounding? or grounding-only? or a different N?
2. **reasoning.effort** — `high` (better traces, ~costlier) vs `medium` (cheaper). Default: high.
3. **Referring** — regenerate too, or grounding-only this round (referring wasn't the main defect)?
4. **Reuse v1 easy/no-tool traces** to cut cost, or regenerate everything fresh?
5. Confirm **pilot-first** (strongly recommended) before any full run.

Nothing is executed until these are confirmed.

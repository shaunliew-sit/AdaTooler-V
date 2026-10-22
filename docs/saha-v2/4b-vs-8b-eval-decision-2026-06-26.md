# SAHA 4B GRPO eval — analysis & 4B-vs-8B main-model decision (2026-06-26)

**Question (user):** now that the SAHA-4B GRPO eval is done, does 4B show improvement that supports
our stand, or should we stick with the 8B we already have?

**Short answer: stick with 8B as the main model. Keep 4B as a scale-comparison row** — it delivers two
defensible messages (GRPO helps at *both* scales; monotone scaling), but 4B **regresses below the vanilla
Qwen3-VL-4B base on SWIG referring**, which is exactly the reviewer attack we are trying to escape. 8B
fixes that. Promoting 4B to main would re-open the wound.

**Data source:** `/workspace/hoi-benchmarks/results-sftgrpo/{hico,swig}_{ground,action}_{sft_4b,grpo_4b}`
(SAHA-4B), `…_{sft,grpo}` (SAHA-8B), and `results-baseline-qwen3vl-4b-instruct/…` (vanilla 4B).
All numbers ×100. Independently re-extracted twice (script + raw-file agent) — 12/12 match. The 8B cells
reproduce the verified paper tables exactly (HICO-G GRPO 31.03, SWIG-G GRPO 29.05, HICO-ref METEOR 29.66),
so the same pipeline gives the 4B numbers trustworthy provenance.

---

## 1. The numbers

### Grounding (AR, ×100)
| | Base-4B¹ | SAHA-4B SFT | SAHA-4B **GRPO** | GRPO−SFT | Base-8B¹ | SAHA-8B SFT | SAHA-8B **GRPO** |
|---|---|---|---|---|---|---|---|
| HICO | 9.28 | 29.62 | **30.35** | **+0.73** | 30.38 | 30.04 | **31.03** |
| SWIG | 18.07 | 27.85 | **28.53** | **+0.68** | **38.32** | 28.26 | 29.05 |

> ⚠ **Base-8B SWIG grounding = 38.32 ≫ SAHA-8B-GRPO 29.05.** Both bases are no-proposal vanilla VLMs.
> Base-8B is a *strong* free-grounder (esp. SWIG), Base-4B is weak (HICO 9.28) — so the grounding "base"
> is NOT scale-neutral, and the SWIG-grounding regression is a real same-scale-8B fact (the paper's
> conceded proposal-dependence / vocab-ceiling limitation), not a cross-scale artifact. See §3.

### Action Referring (×100)
| | metric | Base-4B¹ | SAHA-4B SFT | SAHA-4B **GRPO** | GRPO−SFT | Base-8B¹ | SAHA-8B **GRPO** |
|---|---|---|---|---|---|---|---|
| HICO | METEOR | 24.95 | 28.52 | **28.92** | +0.40 | 25.01 | **29.66** |
| HICO | exact | 18.92 | 24.58 | **24.96** | +0.38 | 16.99 | 25.55 |
| HICO | BERTScore-F1 | 82.12 | 83.97 | **84.24** | +0.27 | 81.84 | 84.73 |
| SWIG | METEOR | **18.18** | 15.57 | 16.07 | +0.49 | 19.08 | **19.36** |
| SWIG | BERTScore-F1 | **75.48** | 74.40 | 74.89 | +0.49 | 77.31 | 77.90 |

**Same-scale (SAHA-GRPO vs its OWN vanilla base):**
| | HICO-G AR | SWIG-G AR | HICO-Ref BS/MET | SWIG-Ref BS/MET |
|---|---|---|---|---|
| **4B-GRPO − Base-4B** | +21.07 (prop) | +10.46 (prop) | +2.12 / +3.97 | **−0.59 / −2.11** |
| **8B-GRPO − Base-8B** | +0.65 | **−9.27** | +2.89 / +4.65 | **+0.59 / +0.28** |

Two facts the base-8B column makes explicit: (a) the **SWIG-grounding regression is same-scale at 8B**
(38.32→29.05), the paper's conceded proposal-dependence limitation — not a 4B-vs-8B artifact; (b) **8B flips
the SWIG-referring regression positive against its own base** (BS +0.59, MET +0.28) whereas 4B stays below
its base (BS −0.59, MET −2.11). The referring base barely changes with scale (HICO MET 24.95→25.01), so the
growing SAHA gain over base (HICO MET +3.97→+4.65) is the model learning, not a stronger base.

¹ **Base-4B is the vanilla, NO-PROPOSAL Qwen3-VL-4B** (verified: plain grounding prompt, no `<candidate_objects>`).
So the grounding gap base→SAHA (9.28→30.35) is *mostly the proposal*, not our training — do **not** headline it.
² BERTScore-F1 for the two GRPO-4B referring runs was computed 2026-06-26 with `calculate_bertscore.py`
(microsoft/deberta-v2-xxlarge-mnli, rescale_with_baseline=True, lang=en) — identical config to every other
BERTScore cell, so apples-to-apples. Files: `…_grpo_4b/…_per_triplet_bertscore_deberta-v2-xxlarge-mnli_metrics.json`.

---

## 2. What the 4B run proves (supports our stand)

**Finding 1 — GRPO beats SFT on all four 4B cells, on every metric.** HICO-G +0.73, SWIG-G +0.68,
HICO-ref METEOR +0.40, SWIG-ref METEOR +0.49 — and the now-computed BERTScore-F1 agrees (HICO-ref +0.27,
SWIG-ref +0.49), as do exact-match (HICO +0.38). The CTG reward + GRPO improves over the frozen SFT at 4B
in the *same direction* as at 8B. This is the single most useful 4B result: it shows the training contribution is **not an
8B-only artifact**, and directly answers **Yongkang's "inconsistent performance across model size"** — GRPO
is a consistent net positive at both scales.

**Finding 2 — scaling is monotone and sensible.** SAHA-4B-GRPO actually edges out SAHA-8B-**SFT** on both
grounding tasks (HICO 30.35 > 30.04; SWIG 28.53 > 28.26), and sits just below 8B-GRPO. The ladder
**4B-SFT < {4B-GRPO ≈ 8B-SFT} < 8B-GRPO** holds on every grounding cell — a clean "method scales with
capacity" story and evidence the 4B is a *credible* compact point, not a broken run.

**Finding 3 — HICO referring is a real 4B win over the vanilla base.** SAHA-4B-GRPO beats Base-4B on
HICO referring (METEOR 28.92 vs 24.95, **+3.97**; exact +6.04; SFT-stage BERTScore 83.97 vs 82.12). So the
original reviewer line "*no improvement over Qwen3-VL-4B*" is **already false on HICO referring at 4B**.

---

## 3. Why we still keep 8B as the main model

**Reason 1 — 4B regresses below the vanilla base on SWIG referring (the reviewer's exact attack).**
SAHA-4B-GRPO METEOR 16.07 **< 18.18** Base-4B (−2.11), **and BERTScore-F1 74.89 < 75.48 (−0.59)** — both
metrics agree the 4B SWIG-referring policy sits below the vanilla base. The proposal/format adaptation hurts
SWIG referring at 4B and GRPO only partially recovers it (METEOR 15.57→16.07, BERTScore 74.40→74.89, still
under the base). This is precisely R2/Indri's "little-to-no improvement over the 4B baseline" — and at 4B it
is *worse than no improvement*. **At 8B this flips to a clean win: SWIG referring METEOR 19.36 > 18.18,
BERTScore 77.90 > 75.48.**
8B is what rescues the referring story; 4B-as-main would hand the reviewer back their strongest point.

**Reason 2 — 8B dominates 4B on every cell (now fully confirmed, BERTScore included), by the most on the
weakest axis.** 8B-GRPO − 4B-GRPO: HICO-G +0.67, SWIG-G +0.52, HICO-ref METEOR +0.74 / BERTScore +0.49,
**SWIG-ref METEOR +3.29 / BERTScore +3.01**. The largest gap is exactly where 4B is fragile (SWIG referring),
so scale buys the most where we most need it. (With the GRPO-4B BERTScore now computed there is no remaining
cell where the 8B≥4B claim is unverified.)

**Reason 3 — the headline claims in the paper are 8B claims.** The verified §4 story (referring best
general VLM at all strata incl. Hard +4.14; grounding 11/12 cols vs 8B+proposal; selectivity 47.7/6.7)
is all 8B. None of it reproduces at 4B strength. Switching main to 4B would require re-deriving and
weakening every headline.

---

## 4. Recommended use of the 4B numbers in the paper

Add a compact **scale-comparison** block (NOT a new main model). It does positive work:
- **Lead message:** "GRPO improves over SFT at both 4B and 8B (HICO-G +0.73/+0.99, SWIG-G +0.68/+0.79),
  and performance scales monotonically with capacity" → answers Yongkang (consistency) + R2 (GRPO adds value).
- **Honest line:** at 4B the SWIG-referring regime is below the vanilla base; capacity (8B) is needed to
  turn the proposal-conditioned referring policy net-positive there. Frame as a *capability-scaling*
  observation, consistent with the SWIG-grounding vocabulary-ceiling diagnostic already in the paper.
- **Do NOT** present the Base-4B→SAHA-4B grounding jump (9.28→30.35) as a result — it is the proposal,
  and foregrounding it re-triggers the "proposals do the work" critique. Use **SFT-4B→GRPO-4B** to isolate
  the training contribution.

### Loose ends before any 4B row ships
1. ~~Compute BERTScore-F1 for the two GRPO-4B referring runs.~~ **DONE 2026-06-26** — HICO 84.24, SWIG 74.89
   (same deberta-v2-xxlarge-mnli + rescale config as all rows). The 4B referring row is now metric-complete.
2. **Base-4B is no-proposal.** For an apples-to-apples 4B *grounding* contribution either (a) report only the
   SFT-4B→GRPO-4B delta, or (b) run a 4B**+proposal** base if a fair grounding comparison row is wanted.
3. ~~Hard/Easy s_ref strata not computed for 4B.~~ **DONE 2026-06-26** — computed via `stratify_all_tasks.py`
   (`models_4b.json`); the 8B `grpo` cross-check row reproduced the published paper strata *exactly* (HICO-G
   Easy 38.98 / Hard 4.44; HICO-ref Hard MET 25.63 / BS-F1 81.55; SWIG-ref 19.36/77.90), validating the pipeline.

### Paper tables updated 2026-06-26 (Codex-verified, "SAFE TO FINALIZE")
- `depd/tbl_action_grounding.tex` — added a **\methodname-4B** row (HICO + SWIG), full All/Easy/Hard strata.
- `depd/tbl_action_referring.tex` — added a **\methodname-4B** row, full strata (LLM-Judge blank, as for 8B).
- `depd/tbl_ablation.tex` — added a **Qwen3-VL-4B backbone** block (base / +proposal / +SFT / Full); base &
  +proposal cells reuse the published main-table 4B numbers, SFT/Full from the verified 4B eval.
- All inserts cross-checked against `scratchpad/stratified_4b.json` and the existing main-table cells; column
  counts uniform; purple `\multirow` bumped −8→−9. SAHA-4B is labeled as a scale point; 8B stays "Ours".

---

## 5. Decision to confirm with the prof
**Main model = 8B** (unchanged). **4B = scale-comparison row** carrying the "GRPO helps at both scales +
monotone scaling" message, with an honest note on the 4B SWIG-referring regression. Reasoning above is
fully grounded in the just-completed eval; no re-run needed for the decision itself (only the BERTScore
fill if a 4B referring row is published).

---

### Claim → Evidence map
- *GRPO>SFT on all 4 4B cells* | `*_grpo_4b` vs `*_sft_4b` metrics (30.35>29.62; 28.53>27.85; 28.92>28.52; 16.07>15.57) | **supported**
- *4B-GRPO ≈ 8B-SFT on grounding* | 30.35 vs 30.04; 28.53 vs 28.26 | **supported**
- *8B>4B every cell* | deltas in §3 Reason 2 | **supported**
- *4B SWIG referring < vanilla base* | METEOR 16.07 < 18.18; BERTScore 74.89 < 75.48 (both agree) | **supported**
- *HICO referring 4B beats base* | METEOR 28.92 vs 24.95 (+3.97); BERTScore 84.24 vs 82.12 (+2.12) | **supported**
- *Base-4B is no-proposal* | grounding prompt has no `<candidate_objects>` (agent-verified) | **supported**
- *8B ≥ 4B on every cell* | BERTScore gap closed (HICO +0.49, SWIG +3.01); no unverified cell remains | **supported**

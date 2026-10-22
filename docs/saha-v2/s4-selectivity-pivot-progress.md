# §4 Selectivity Pivot — Progress Log (2026-06-26)

Session goal: reframe §4 so the paper has **strong, honest findings despite not being the dominant model** (methodology changed SDS-GRPO → Counterfactual Tool-Gain). User decision: **honest pivot to selectivity** (not "SAHA beats base on hard/occluded/crowded" — that is falsified by our own tables).

## Decisions locked this session
- **Hard-case framing = selectivity** (AskUserQuestion, 2026-06-26). The aggregate claim "SAHA wins on hard/occluded/crowded grounding" is NOT used (falsified on every axis — see below).
- **Qualitative figure (`fig:effective_tool_call`) = Bicycle** HICO_test2015_00000158 (SFT-vs-GRPO selectivity). Replaces frisbee.
- **Training-dynamics figure (`fig:toolcall`) = REDRAW** from real run (selectivity gap, not the false "acc up / tools down").

## Why the naive hard-case claim was dropped (verified)
- **s_ref-Hard**: base > ours (HICO 5.16 vs 4.44; SWIG 8.13 vs 1.88). Pattern-A (base-no-tool wrong + SAHA-tool right) on s_ref-Hard ≈ 0.
- **Occlusion / body-contact**: genuine no-tool base beats SAHA on EVERY occlusion & contact subset (HICO & SWIG); SAHA only edges ahead on the *easiest* (no-occlusion) subset. (scratchpad `SCENE_DIFFICULTY_ANALYSIS_REPORT.md`.)
- **Crowding**: untestable (98% of grounding samples are a single person+object pair).
- Cherry-pick trap: ~200 individual "base=0, SAHA-tool>0" cases exist, but aggregate is negative ⇒ more reverse cases ⇒ a "helps crowded scenes" claim invites a fatal reviewer recompute.
- ⚠️ **The file `results-sft-qwen3vl-8b/...` is NOT a no-tool baseline** (83.7% of items use tools). Genuine no-tool base = `/workspace/Groma/results-redo/hico_ground_qwen3vl_8B/...` (+ swig). Use it for any base comparison.

## Verified numbers (reproducible: `scratchpad/compute_selectivity_and_dynamics.py`)
Selectivity (GRPO zoom rate; SFT in parens):
- HICO-Ground Hard **47.7%** (43.0) | Easy **6.7%** (5.4)
- SWIG-Ground Hard **45.2%** (39.0) | Easy **13.4%** (9.8)
- HICO-Refer **7.7%** (7.3) | SWIG-Refer **3.2%** (2.2)
- Zoom ~7× more on hard than easy grounding; GRPO sharpens the gate over SFT.

Training dynamics (run `SAHA-CF-a0.6-newsft-resampled`, 1000 steps):
- Accuracy **~flat/maintained ~78%** (EMA artifact made it look rising; proper smoothing = flat).
- Tool calls/query **rise 0 → ~0.24**; Hard zoom rate **42.7% → 65.2%**, Easy flat ~16–23%, gap widens.

## Two false claims caught (fix in `s4-results-draft.md` + `sec/4_experiment.tex`)
1. "**0% on referring**" → actually **7.7% (HICO) / 3.2% (SWIG)**. Reword "rarely on referring (<8%)".
2. Training-fig caption "**accuracy rises while tool calls decrease**" → false. Real run: accuracy maintained, tool use *rises* & concentrates on hard. Old `fig/training_process.png` (tools 2.0→1.2) is a different/older run.

## Figures built (in scratchpad — to be copied into Overleaf `fig/`)
- `scratchpad/training_selectivity_real.png` → replaces `fig/training_process.png`. Caption drafted ("Selective tool use emerges during GRPO …"). Script: `scratchpad/plot_selectivity_dynamics.py`.
- `scratchpad/effective_tool_call_bicycle.png` → replaces `fig/effective_tool_call-2.png`. 2-panel: SFT 6 zooms→collapse(0/4) vs GRPO 1 zoom→4/4. Script: `scratchpad/plot_bicycle.py`. (Bicycle trace: SFT 6 zoom_in oscillating [300,100,750,1000]↔[500,100,850,1000], no answer; GRPO 1 zoom_in [600,200,1000,860] → 4 pairs all match GT; genuine base 4 matched + 3 spurious.)

## §4 prose edits — DONE (2026-06-26, agent-reviewed, zero `--` in file)
Applied to `sec/4_experiment.tex` (user banned `--`/em/en dashes entirely; verified clean across whole file):
1. ✅ Qualitative grounding (frisbee): removed "scale-adaptive reasoning is the key factor for small objects"; now states small objects stay hard in aggregate, consistent with the modest AR$_s$.
2. ✅ Selectivity paragraph ADDED with verified 47.7/6.7 (HICO), 45.2/13.4 (SWIG), referring 7.7/3.2 (not 0), GRPO 43.0→47.7; plus the DetZoom rebuttal clause (fixed size threshold ≠ proposal-reliability trigger).
3. ✅ Training-dynamics paragraph rewritten to the real run (Hard zoom 43→65%, Easy ~20%, accuracy steady ~78%); killed the false "fewer tools yet higher accuracy".
4. ✅ Tool-Use Comparison rewritten to the bicycle (SFT 6 zooms→0 pairs, GRPO 1 zoom→4/4).
5. ✅ Also fixed TWO live self-falsifying claims caught during the dash sweep: line 78 "concentrates gains at strict IoU thresholds and on small objects" (wrong twice — gain is at IoU@.5, AR$_s$ is our weakest) → honest @.5-vs-@.75 wording; line 83 "highest LLM-Judge on HICOref" (column blank for Ours) → best general VLM on METEOR+BS-F1 incl. Hard +4.14, beats 32B. Referring numbers verified vs `tbl_action_referring.tex`.
6. ✅ Fixed broken `\cref{tab:grounding}` → `tab:grounding_hico` (old label only in commented blocks). Cleaned dashes on lines 7/25/28/41/85/148.
Agent review verdict: reads as natural human academic writing, all numbers internally consistent, no overclaiming.

## Fair-frame reorder + efficiency table — DONE (2026-06-26, agent-reviewed, dash-free)
- **Quantitative Results reordered** to the fair frame in `sec/4_experiment.tex`: opening within-proposal paragraph (all region methods get the same YOLOE proposals) → \subsubsection{Action Referring} (H1: BS-F1 84.73/85.39/81.55, METEOR 29.66/30.54/25.63, Hard +4.14 over base, beats 32B; Groma different paradigm; SWIG mid-pack honest) → \subsubsection{HOI Grounding} (H2: beats 8B+proposal on 11/12 columns, AR@.5 +6.42; H3: vs base AR@.5 +4.54, @.75 on par; larger/closed lead conceded). Replaced the old grounding-first subsections.
- **Efficiency table** `depd/tbl_efficiency.tex` created + \input + Efficiency paragraph added in the Analysis subsection. NATIVE-H200 verified numbers (N=50): Qwen-4B 0.54/0.35, 8B 0.87/0.42, 32B 2.25/1.23, SAHA-8B 2.36/1.19 (mean/median); tool-use (N=92446, 8B): GRPO 0.104 calls / 9.44% / 1.104 turns, SFT 0.093 / 7.91% / 1.093. Dropped SAHA-4B + Claude/GPT per plan. Latency framing = cross-model (SAHA ~ 32B, median 1.19s the robust value, cost is reasoning not tools); did NOT use the old GPU0 2.92/2.66 +0.26s numbers (different run, would contradict the native 0.87 base). Used `{n/a}` not `{--}` for SFT latency (dash ban).
- Agent review (2nd pass): numbers all consistent, no overclaim, dash-free; fixed the negation-contrast tic + 3 defensive asides ("we do not contest that" etc.) + an efficiency redundancy.

## Figures integrated into the paper — DONE (2026-06-26)
The two new PNGs are now COPIED into the Overleaf repo `fig/` and the figure tex files repointed + recaptioned (paper now internally consistent, prose matches figures):
- `fig/effective_tool_call_bicycle.png` ← `depd/fig_effective_tool_call.tex` (was `effective_tool_call-2.png` frisbee). New caption = bicycle SFT-6-zooms-collapse vs GRPO-1-zoom-4/4.
- `fig/training_selectivity.png` ← `depd/fig_tool_call.tex` (was `training_process.png`, false "acc up/tools down"). New caption = Hard zoom 43→65%, Easy ~20%, accuracy maintained ~78%.
Old images left in `fig/` (unreferenced) for safety. NOTE: all paper deliverables live in the **Overleaf repo `/workspace/6971992f16a95dab9a63a1f9`** (a SEPARATE git checkout from `/workspace/AdaTooler-V`); a stop-review scanning only the AdaTooler-V checkout will not see them. Overleaf `git status`: M sec/4_experiment.tex, M depd/fig_effective_tool_call.tex, M depd/fig_tool_call.tex, ?? depd/tbl_efficiency.tex, ?? fig/effective_tool_call_bicycle.png, ?? fig/training_selectivity.png. User commits the Overleaf repo.

## Abstract + Intro consistency pass — DONE (2026-06-26, Codex + subagent verified)
Fixed in `sec/0_abstract.tex` + `sec/1_intro.tex` (both verifiers PASS; active prose dash-free):
- "accuracy rises steadily" (intro) → "overall accuracy is maintained" (matches verified flat ~78%).
- "crowded or occluded scenes" / "crowded and occluded cases" (abstract + intro ×3) → "where the object proposal is unreliable" (honest s_ref axis; base actually wins occlusion/crowding aggregate).
- Abstract main result "leads among **VLM-based baselines** on holistic interaction understanding" → "leads among **general-purpose VLM** baselines on HICO-DET action referring" (Groma-7B-hoi-FT, a VLM baseline, beats us on referring; "holistic" implied the LLM-Judge metric we don't have).
- intro:23 motivation "heavily occluded objects demand refinement" → "low-resolution objects" (zoom can't fix occlusion per §4).
- Vacuous 3rd contribution ("we ran experiments") → real honest finding (strongest same-backbone proposal-matched grounding + strongest general VLM on HICO referring; analyse when zoom helps).
- AI-style tells removed: "transformative paradigm", "synergistic potential", doubled "leverage"/"core challenge", "Moreover"/"Furthermore" openers, one "While" opener.
- Codex CHECK note: 5 COMMENTED-OUT lines still hold the OLD false narrative (`0_abstract.tex:19`, `1_intro.tex:40` = "accuracy rises / tool calls decrease 2.5->1.0"; `0_abstract.tex:14,18`, `1_intro.tex:47` = old 4B/SOTA claims). Inert while commented; RECOMMEND the user delete them so they can't be reintroduced. Left in place (did not delete user's commented history).

## Conclusion consistency pass — DONE (2026-06-26)
`sec/5_conclusion.tex`: "leads among VLM-based methods on holistic referring" → "general-purpose VLMs on HICO-DET action referring"; "remains competitive on HOI grounding against larger baselines" → "strongest same-backbone, proposal-matched method on HOI grounding, while larger and closed models lead the absolute ranking"; "spatial context-guided tool use `---`not increased computation`---` drives the gain" (em-dashes + false accuracy-gain implication) → "concentrates tool use on proposal-unreliable cases while overall accuracy is maintained ... how it allocates tools across samples, not from added computation". Dash-free, no overclaims.

**STATE: abstract + intro + §4 + conclusion + figures + efficiency table are all internally consistent and honest. Ready for the user's Overleaf commit.** (User chose to KEEP the 5 dangerous commented-out lines for now.)

## Remaining paper work (next batches)
1. `s4-results-draft.md` PART 2: Fig1 framework.png CTG retitle + delete duplicate panel (designer; commented `fig_framework.tex` caption still says SDS-GRPO) + §4.4 citation/bib checks on compile.
2. Optional reviewer-defense adds: Hard-stratum trade-off paragraph, DetZoom citation (Look-Less-Reason-More), open-vocab/mAP justification paragraph.

See memory [[saha-cf-selectivity-and-figures-verified]], [[saha-v2-paper-and-runs-status]], and `results-claim-defense-plan.md` / `reviewer-response-mapping.md`.

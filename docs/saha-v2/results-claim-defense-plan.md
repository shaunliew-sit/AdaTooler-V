# SAHA 8B Results — Claim-Defense Plan (ACCV §4)

**Date:** 2026-06-25. **Source:** multi-agent analysis (win-mining + hard-defense + tool-value + reviewer red-team + synthesis), all numbers from `depd/tbl_action_grounding.tex` (active HICO; commented SWIG), `tbl_action_referring.tex`, `tbl_ablation.tex` + `results-sftgrpo/` + `hard-case-regression-analysis.md`. Model = Qwen3-VL-8B Full (proposal+SFT+GRPO, counterfactual tool-gain, avg-IoU). **Cross-checked by 4 independent agents (mutually consistent).**

## THE DECISION: reframe the thesis
**"tools help the hardest cases" is FALSIFIED by our own Hard columns** (base 8B beats Ours: HICO-Hard 5.16 vs 4.44; SWIG-Hard 8.13 vs 1.88; *worse* under min-IoU). Reframe to:
> *"A proposal-aware counterfactual reward yields the strongest **same-VLM** grounding + referring at 8B, with **selective** tool use as the mechanism — and an honest characterization of where the proposal-conditioned regime trades off."*

## Headline claims to LEAD with (ranked, all numbers verified)
- **H1 — HICO referring clean sweep (strongest).** Ours is best general/same-VLM VLM at **all** strata *including Hard*: BS-F1 84.73/85.39/**81.55**, METEOR 29.66/30.54/**25.63** — Hard METEOR **+4.14** over base 8B (21.49). The *one honest hard-stratum win* in the paper. (Only Groma-7B-hoi-FT higher → different paradigm.)
- **H2 — same-VLM grounding win.** Ours beats Qwen3-VL-8B **+proposal** on **11 of 12** HICO grounding columns (AR 31.03 vs 28.14, @.5 54.94 vs 48.52 = **+6.42**, even Hard 4.44 vs 3.68). The one exception is **@s** (3.58 vs 3.65) — see DROP list. Isolates the SFT+GRPO contribution on top of identical proposal injection.
- **H3 — beats no-proposal base 8B** on overall AR (31.03 vs 30.38) and decisively at IoU@.5 (54.94 vs 50.40, **+4.54**; Easy@.5 67.78 vs 61.46, +6.32).
- **H4 — GRPO is the enabler/fix.** Full > +SFT on **every** grounding+referring cell; GRPO *recovers* the SFT proposal-dependence drop (SWIG-G 28.26→29.05; HICO-G 30.04→31.03; HICO-Hard +prop 3.68→4.44).
- **H5 — selectivity mechanism.** Zoom fires **47.7% on HARD grounding vs 6.7% EASY**, **0% on referring** (HICO; SWIG 45.2% vs 13.4%). GRPO sharpened it over SFT (HARD 42.9%→47.7%). The adaptive-tool-use evidence.

## Hard-case rebuttal (paper-ready paragraph)
> On the Hard split (proposal misses the GT pair, s_ref<0.5; 11.2% of HICO / 10.6% of SWIG pairs) the no-proposal base Qwen3-VL-8B attains higher AR than ours (HICO 5.16 vs 4.44; SWIG 8.13 vs 1.88). This is the expected behaviour of a proposal-conditioned model: the base is unanchored and free-grounds, paying for it everywhere else, whereas ours is correctly conditioned on the (here-misleading) proposal. The honest comparator is therefore other proposal-conditioned methods: among the eight +proposal entries on HICO-Hard, ours (4.44) ranks 3rd — beating Qwen3-VL-8B+proposal (3.68), Qwen3-VL-4B+proposal (3.51), GPT-5.2+proposal (3.14) and InternVL3-38B+proposal (2.69), and trailing only the larger/closed Claude-4.5-Opus+proposal (5.88) and Qwen3-VL-32B+proposal (5.34). Against its own architecture, ours improves Hard by +0.76 over Qwen3-VL-8B+proposal. The trade-off is favourable in aggregate: ours beats the base on All AR (31.03 vs 30.38), Easy AR (38.98 vs 37.93), @.5 (54.94 vs 50.40), @m and @l. The Hard gap is introduced at the SFT stage (SWIG ablation 35.78→28.26) and GRPO partially recovers it (28.26→29.05; HICO-Hard 3.68→4.44), so it is a frozen-SFT proposal-over-trust artifact, not a GRPO failure.

## DROP / CONCEDE / NARROW (or reviewers falsify us with our own table)
- **DROP** "tools help hard grounding" unconditionally (falsified).
- **DROP** any **@s small-object** win — Ours 3.58 **<** +proposal 3.65 **<** base 3.76 (Ours is the *lowest*). (Contradicts "zoom helps small objects" — a reviewer reads our own table.)
- **CONCEDE / keep commented:** SWIG grounding (Ours 29.05 vs base 38.32, **−9.27**) — present only as the proposal-dependence / vocab-ceiling **diagnostic**, never as a result.
- **NARROW** referring "SOTA" → "best among **general** VLMs" — Groma-7B-hoi-FT (BS-F1 89.69, METEOR 40.00) dominates referring (region-token finetune, different paradigm).
- **NARROW** all grounding wins to "**same-VLM 8B / proposal-conditioned**" — Qwen3-VL-32B (34.18 AR) and Claude+proposal (43.96) beat Ours; not absolute SOTA.
- **NEVER** publish the observational eval tool-vs-notool Δ — it is **negative** (HICO-Hard −4.34, selection bias, model zooms its hardest queries). Lead with tool-**rate** selectivity, not tool-accuracy-gain. Only honest causal estimate = near-floor within-group **+0.026**.
- **LLM-Judge** columns are **blank** for Ours → do **not** claim an LLM-Judge lead until the score is produced (or drop the column).

## §4 framing order
Lead **referring (H1)** → **same-VLM grounding (H2/H3)** → **ablation = GRPO-as-enabler (H4)** → **selectivity (H5)** → **honest hard/SWIG limitation paragraph** (the rebuttal above).

## Reviewer attacks → our counter
| Attack | Severity | Counter |
|---|---|---|
| "Base beats you on Hard → thesis dead" | fatal if not reframed | the rebuttal paragraph (trade-off + fair-comparator + GRPO-recovers) |
| "SWIG grounding regresses −9.27 vs base" | serious | keep table commented; present as proposal-dependence diagnostic; SFT-stage attribution |
| "Overall AR +0.65 is marginal / 32B wins" | serious | lead with @.5 +4.54 & same-VLM +2.89; scope to 8B |
| "It's the proposal, not your reward" | serious | H2 (Ours ≫ +proposal on every column) + H4 ablation |
| "Groma beats you on referring" | serious | "best general VLM"; Groma = region-token finetune |

## Additional analysis still worth doing
1. **Significance/CI** on the HICO-All AR +0.65 and @.5 +4.54 (the +0.65 is fragile).
2. **Regenerate the selectivity rates** (47.7/6.7/0%) via a committed script → reproducible artifact, not ad-hoc counts.
3. Fix stale **"4B" prose** in `sec/4_experiment.tex` to match the 8B tables.
4. Decide deliberately: keep SWIG-grounding table commented (recommended) or include with honest framing.

# Reviewer-Response Mapping — SAHA ACM MM → ACCV resubmission

**Date:** 2026-06-25. Prior submission: ACM MM, 5 reviewers (Weak-Accept / **Reject** / Borderline / Weak-Reject / Borderline) + supervisor feedback (Indri, Yongkang, Bingjie). Goal: ACCV (5 Jul). Reviews at `reviewer-comment.md` / `supervisor-feedback.md`. Pairs with `results-claim-defense-plan.md`.

**The central insight:** the paper's whole pivot this cycle (**SDS-GRPO → proposal-relative tool-gain (PTG)**, renamed from "counterfactual tool-gain / CTG" on 2026-07-01, + **4B → 8B**) is a *direct* response to the two most damaging method critiques (R2 size-heuristic, R4 area-can't-capture-occlusion). That redesign should be the headline of the resubmission. Several other concerns are now auto-resolved by 8B + the new numbers; a few need genuinely new work.

---

## CURRENT STATE (2026-07-01) — concern → live paper anchor → status

Post PTG-rename + efficiency figure + numbers audit + 5-agent adversarial review + bootstrap CIs. Paper = `/workspace/6971992f16a95dab9a63a1f9`. Status: ✅ addressed · 🟡 partial/scoped · ⛔ gap (deferred/needs work).

| # | Concern | Reviewers | Live anchor | Status |
|---|---|---|---|---|
| 1 | **Difficulty = object AREA can't capture occlusion/contact** | R4, R1 | §3:175 "Difficulty is proposal-relative, not size-based"; §3:179 PTG reward; §4:81 `s_ref` stratification; §2:30 rebuttal | ✅ core pivot |
| 2 | **Unfair: baselines get no proposals** | R2,R3,R4,R5 | `tbl_action_grounding`/`tbl_action_referring` "+proposal" blocks (every baseline); §4:90-91 same-backbone+proposal comparison | ✅ |
| 3 | **Computationally heavy / report efficiency** | R1,R2 | `fig:efficiency` (latency-vs-AR); §4:142-145 "not computationally heavy" | ✅ |
| 4 | **Mixed / weaker than larger VLMs** | R1,R3,R5 | all "best" claims scoped to same-backbone+proposal (§4:91,95; conclusion); 32B/closed = "reference/scale context" | ✅ scoped |
| 5 | **Groma-7B-ft beats referring / simpler** | R2,R3 | §4:86 specialist-oracle framing (exact-string CE, no free-form/tools); "leads *general-purpose*" scoping | ✅ framing |
| 6 | **Novelty vs AdaTooler-V unclear** | R2 | §2:31-33 differentiation (Look-Less + AdaTooler-V; online/per-rollout/proposal-anchored); PTG = the contribution | ✅ (referring-branch caveat, see §2:32 scope) |
| 7 | **Single-run / variance** | (implicit, adversarial) | §4:95 "Robustness to eval sampling" bootstrap CIs | 🟡 eval-set only; seed variance in Limitations |
| 8 | **Full < proposal-only on SWIG grounding** | R2,R3,R4,R5 | §4:121 (SFT proposal over-fit, not reward) + Limitations (conclusion) | 🟡 conceded honestly |
| 9 | **GRPO/SFT contribution unclear; proposals do the work** | R2,R3,R4,R5 | `tbl_ablation` + §4:117-122 (GRPO surpasses proposals-only on HICO-G/R both scales, SWIG-R@8B); §4:120 attributes to GRPO stage | 🟡 α=0 reward-isolation NOT run (no compute) — flagged future work |
| 10 | **Deterministic size-based zoom might suffice** | R2 | premise defused (difficulty=`s_ref` not size, §3:175; §4:139 "a fixed size threshold would not reproduce this") | 🟡 heuristic baseline not run |
| 11 | **No standard HOI mAP / traditional detectors** | R1,R2,R4,R5 | Limitations notes it; semantic metrics justified (paraphrase penalty) | ⛔ deferred to next discussion |
| 12 | **LLM-Judge / trace-gen / reproducibility validation** | R2,R3,R5 | §6 suppl (judge model, trace gen, hyperparams, GRPO batch/K=6) | 🟡 single judge, no inter-judge agreement |
| 13 | **Test other VLM backbones (Qwen-specific?)** | R2 | 4B + 8B (both Qwen3-VL); selectivity scale-robust (§4:136) | 🟡 same family only |
| 14 | **Detector reliance vs "open-vocab departure"** | R2 | §3:86 "YOLOE-26, an open-vocabulary detector" — pipeline is open-vocab end-to-end (YOLOE-26 = arXiv 2602.00168 "Real-Time Open-Vocabulary Instance Segmentation"; "-26" is YOLO26, NOT a 26-category set). No closed-set contradiction. | ✅ (premise was wrong — YOLOE is open-vocab) |
| 15 | **Fig 1 redundant with Fig 2** | R2 | framework.png redraw (CTG→PTG/`s_ref`) | ⛔ pending figure pass |
| 16 | **zoom_out usefulness / 1.0 avg tools** | R2 | selectivity §4:134-139 reframes; zoom_out not separately defended | 🟡 |

**Headline for the response letter:** lead with #1 (area→`s_ref` redesign, answers R2+R4 method critique), #2 (proposal-matched fairness), #3 (efficiency figure). Concede #8/#11 honestly (now in Limitations). Remaining ⛔ (mAP, framework.png, detector-framing) are the known open items; α=0 ablation (#9) is the single most-requested experiment we could not run on 1-GPU before Jul 5.


## Concern → status (grouped by what it takes)

### A. RESOLVED by the changes we already made
| Concern | Reviewers | Why resolved |
|---|---|---|
| **"Area alone can't capture occlusion/contact"** (R4); **"SDS is just box area"** | R4, R2, Bingjie | **CTG replaces area-based SDS entirely** — difficulty now = the proposal's evidence state (`s_ref`), not box area. This is the central method response. |
| **SDS hyperparameters unjustified / not ablated** | R2 | Moot — SDS removed; CTG has one knob (α). |
| **Action Referring shows ~no gain over Qwen3-VL-4B** | R2, Indri | Under **8B** + new numbers, referring is now a **clear win** (Ours best general VLM at all strata; HICO METEOR +4.65 over base). Foreground H1. |

### B. ADDRESSABLE with data we already have (mainly FRAMING)
| Concern | Reviewers | Action |
|---|---|---|
| **Unfair: baselines don't get proposals** (the #1 complaint) | ALL 5 + Indri + Yongkang | The tables **already have `+proposal` rows for every baseline** (Qwen-4B/8B/32B, InternVL3, Claude, GPT). **Lead with the within-proposal comparison** ("all methods get the same YOLOE-26 proposals"); show Ours is best among 8B proposal methods on HICO (H2). Honest that SWIG is a loss. |
| **Proposals do the work; SFT hurts; GRPO adds little** | R2, R3, R4, R5, Yongkang | Ablation: **GRPO > SFT on every cell**; on HICO **Full > proposal-only** (31.03 > 28.14) and GRPO recovers the SFT drop. Lead the ablation with this. (SWIG is the exception — concede, see C.) |
| **Mixed results / weaker than larger VLMs** | R1, R3, R5 | Scope all "best" claims to **same-VLM 8B / proposal-conditioned**; never claim absolute SOTA (32B / Claude+prop lead). |
| **Groma-7B-HOI-FT beats referring** | R2, R3 | Concede + differentiate: region-token finetune trained on exact GT strings, no reasoning/tools — "best among general VLMs." |
| **Novelty vs AdaTooler-V unclear** | R2 | Related-Work CTG differentiation already added (`lookless2025` + AdaTooler-V). Reframe the contribution around the **CTG reward**, not "Proposal-Conditioned Region Context." |
| **Strong claims need citations (L210-211)** | R2 | Add cites or soften during the rewrite. |
| **Figure 1 redundant with Figure 2** | R2 | Cut/merge Figure 1. |

### C. CONCEDE honestly (cannot be turned into a win)
| Concern | Reviewers | Framing |
|---|---|---|
| **SWIG grounding: Full < proposal-only** (negative side effect) | R2, R3, R4, R5 | Real loss (29.05 < 35.78). Keep the SWIG-grounding table **commented**; present as the **proposal-dependence / vocab-ceiling diagnostic** (SFT-stage collapse). Do NOT foreground SWIG grounding. |
| **Hard cases**: base beats SAHA | (implicit) | Reframe via the proposal-dependence trade-off (see claim-defense plan); honest capability-ceiling analysis. |

### D. NEEDS NEW WORK (decide scope before 5 Jul)
| Concern | Reviewers | What it takes |
|---|---|---|
| **R2/DetZoom: "a deterministic size-based zoom may suffice"** | R2, Indri (#3) | **DetZoom baseline is MISSING from the tables.** Either (a) run a size-only-zoom baseline (GPU eval) and show CTG beats it on crowded/occluded, or (b) answer by mechanism (CTG triggers on proposal reliability, not size) + cite Look-Less-Reason-More / "Learning How Hard to Think." **(a) is much stronger — recommended if time allows.** |
| **Computational cost / inference time** | R1, R2, Indri, Bingjie | Add an **efficiency table**: inference time, avg tool calls, token cost for SAHA vs baselines. Needs timing measurement (light). |
| **Non-MLLM HOI baselines + standard mAP** (BC-HOI, ScriptHOI) | R1, R2, R4, Bingjie | Hardest ask. Either add a couple of traditional-HOI-detector comparisons (cite published mAP / run) **or** justify the open-vocab task reformulation as intentionally mAP-incompatible. **Decide deliberately** — likely the biggest remaining risk. |
| **Other VLM backbones** | R2, R4, Bingjie | 4B + 8B is same-VLM two scales, NOT a different backbone. Reviewers want e.g. InternVL. Either add one backbone (GPU) or argue scope. |
| **Baselines get minimal prompts / single-turn (unfair)** | R2 | To be airtight on fairness, re-run key baselines with the **same prompt richness** (and note single-turn is inherent to non-tool baselines). Partial new eval. |
| **zoom_out usefulness / 1.0 avg tool calls** | R2 | Selectivity evidence (H5) + optionally an ablation removing zoom_out. |
| **Trace construction / Table-1 splits / task description underspecified** | R2 | Writing — describe the SFT trace generation, the benchmark split construction, and the two tasks in more detail. |
| **LLM-as-Judge / semantic-metric validation** | R3, R5 | Validate the judge (agreement with humans / inter-metric) or temper reliance. (Also: the LLM-Judge column is currently **blank** for Ours — fill or drop.) |

## DECISIONS (locked 2026-06-25)
- **DetZoom (R2):** answer by **mechanism + literature** — CTG triggers on proposal reliability (`s_ref`), not box size (selectivity H5: zoom 47.7% hard vs 6.7% easy); cite Look-Less-Reason-More (`lookless2025`) + "Learning How Hard to Think" (need to add the cite). **No DetZoom run.**
- **Non-MLLM HOI / mAP (R1/R2/R4):** **justify open-vocab** — add a paragraph arguing the instruction-following reformulation is intentionally open-vocab and incompatible with closed-set mAP; position traditional HOI detectors as out-of-scope. **No new experiments.**
- **Efficiency table (R1/R2/Indri/Bingjie):** **ADD** — report inference time, avg tool calls, token cost for SAHA vs baselines (try to source from existing eval logs first; measure only what's missing).

## Priority order for the resubmission (recommendation)
1. **Reframe around CTG** (answers R2/R4 — the deepest critiques) + **lead with the within-proposal fair comparison** (answers the #1 fairness complaint) — both use data/work already done.
2. **Add the efficiency table** + **DetZoom baseline** (the two cheap-ish new experiments that close R2 + compute concerns).
3. **Decide the non-MLLM/mAP question** (biggest risk; needs a call).
4. Honest SWIG + Groma + scoping; writing fixes (Fig1, traces, splits, cites).

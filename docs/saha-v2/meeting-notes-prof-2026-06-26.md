# Meeting Notes — SAHA ACCV Resubmission (Prof update, 2026-06-26)

**Deadline:** 5 Jul 2026. Prior: ACM MM, 5 reviewers (Reject/Borderline/Weak). All numbers verified vs live Overleaf tables + reproducible scripts on 2026-06-26 (Codex + agent cross-checked).

---

## 1. TL;DR
- **Method changed:** area-based SDS-GRPO → **Counterfactual Tool-Gain (CTG)** reward (difficulty = proposal evidence `s_ref`, not box area). Backbone **8B**.
- **We are NOT the top/dominant model, and we do not claim to be.** Larger/closed models (Qwen-32B, Claude+proposal, Groma-7B-hoi-FT) lead some metrics; the no-proposal base wins the hardest grounding split.
- **Our story:** strongest **same-backbone, proposal-matched** system + **selective tool use** as a novel mechanism + one honest hard-case win (referring), with limits conceded.

## 2. What we established this session
- **The method = Counterfactual Tool-Gain (CTG) reward.** CTG credits a zoom only when the answer *after* the tool scores higher than simply trusting the external object proposal (the gain `s_final − s_ref`), so tool use is rewarded by its actual improvement over the proposal rather than by any hand-set difficulty heuristic; one knob α balances it against the task reward.
- **Verified the selective tool-use mechanism** on the deployed 8B model with a reproducible script: the policy zooms much more on proposal-unreliable cases than on easy ones, and GRPO sharpens this gate over SFT.
- **Quantified the computational cost** (the efficiency table the reviewers asked for): selective, lightweight tool use.
- **Established a consistent, honest result narrative:** removed every self-falsifying claim so the tables, figures, abstract, intro, and conclusion all agree and survive a reviewer re-checking our own tables.

## 3. Results (summary, verified — numbers in the claim map at the end)
- **Action Referring (our cleanest win):** the strongest general-purpose VLM on HICO referring at every difficulty level, including the hardest split, and ahead of a model four times larger. A specialized region-token model trained directly on the GT action strings scores higher, but it is a non-reasoning, non-tool paradigm (conceded).
- **HOI Grounding (same-VLM win):** under matched proposals, the strongest single-8B-backbone method; it beats the same backbone+proposal on nearly every metric and the no-proposal base at the usable-localization (IoU@.5) operating point.
- **Selectivity (the mechanism):** the policy zooms far more on proposal-unreliable cases than on easy ones, and almost never on referring; GRPO sharpens this gate over SFT.
- **Computational efficiency:** selective and lightweight — a tool fires on under a tenth of queries, and the 8B model's latency is comparable to a 32B model, with the cost concentrated on the queries that actually need evidence.
- **Honest limits (conceded, framed in §4 and §5):** small-object recall, the hardest grounding split (no-proposal base is higher), and SWIG grounding overall (regresses vs base).

## 4. The hard-case zoom hypothesis: what we claimed vs what actually happens
**What we hypothesized (the original thesis):** zoom-in tools resolve the hardest cases (small, occluded, or crowded objects, and pairs where the detector proposal is unreliable), so a tool-using model should beat a no-tool baseline exactly on the hard split. The earlier draft leaned on this ("tools help the hardest cases", "scale-adaptive reasoning is key for small objects", "gathers evidence in crowded/occluded scenes").

**What the data actually shows (verified this session):**
- **On aggregate hard grounding the hypothesis is false.** The no-proposal base beats SAHA on the HICO Hard split (AR 5.16 vs 4.44) and on SWIG; across both datasets there are essentially **zero** cases where the no-tool base fails and SAHA's zoom rescues it.
- **Same on the occlusion / crowding axis:** stratifying by occlusion or body-contact, the base wins every subset; crowding is untestable (98% of grounding samples are a single person+object pair).
- **Why:** `zoom_in`/`zoom_out` crop *visible* pixels and cannot reveal an occluded part; and a proposal-conditioned model trades away the base's "free-grounding" on the few misleading-proposal pairs (which is what lets the base win the hard split) in exchange for its gains on AR@.5, Easy, and referring.

**What is true instead (the salvageable, honest finding):**
- The policy is **selectively triggered by hard cases** — it zooms far more on proposal-unreliable pairs than easy ones, and GRPO sharpens this. Zoom is *deployed where the difficulty is*, even though it does not out-score the base there.
- **Reasoning, not the crop, wins the one hard-case result:** on hard *referring* SAHA beats the base by +4.14 METEOR, because holistic description (not box geometry) decides that score.

**Implication:** we pivoted the thesis from "tools help the hardest cases" to "**selective, proposal-aware tool use**" and removed the small-object / crowded-occluded / tools-beat-base-on-hard claims. This is the honest core that survives a reviewer recomputing our own tables.

## 5. "Not dominant on accuracy, but has plus-points" → how we address it (core of the meeting)
**The concern (raised by multiple reviewers):** SAHA does not top the leaderboard on raw accuracy. Larger and closed models lead the absolute ranking, the no-proposal base beats us on the hardest grounding split, and SWIG grounding overall regresses. The reviewers' question is what value the method has if it is not dominant.

**Where the current Overleaf already answers this:** the paper's Experiments section scopes every claim to same-VLM / proposal-matched comparisons; the **Efficiency** paragraph opens with "\methodname is not the fastest system in the table" and pivots to its selective, lightweight tool-use advantage; the **Selectivity** and **Ablation** paragraphs establish the adaptive policy itself as the contribution. So the "not dominant but has plus-points" answer is written into the results section, not left implicit.

**Our response (7 moves):**
1. **Scope every claim — no absolute-SOTA wording.** "Strongest same-VLM 8B under matched proposals" + "best general-purpose VLM on referring." Defensible and still strong.
2. **Lead with the fair within-proposal comparison** (all methods get the same YOLOE-26 proposals) → answers the #1 fairness complaint and isolates what SFT+GRPO adds (11/12 columns).
3. **Sell the mechanism, not the leaderboard.** Selectivity is a novel, reproducible property of the CTG reward; the contribution is the adaptive policy, not topping a metric.
4. **One genuine hard-case win:** Hard referring +4.14 METEOR, where reasoning (not box geometry) decides.
5. **Efficiency angle:** lightweight, selective tool use (32B-level latency at 8B size, tool on <10% of queries).
6. **Concede losses and reframe them, do not hide.** SWIG grounding + hard-split grounding = expected proposal-dependence trade-off (the base free-grounds, which helps on the few misleading-proposal pairs but costs it on AR@.5, Easy, and referring). Fair comparator = other +proposal methods, where ours ranks **3rd of 8** on the HICO Hard split. **GRPO does not cause the regression** (GRPO ≥ SFT on every cell); the drop originates at the SFT stage (SWIG 35.78→28.26, GRPO recovers to 29.05). Present SWIG grounding as a diagnostic, not a headline.
7. **Removed all self-falsifying claims** so a reviewer can't kill us with our own tables: dropped the false LLM-Judge lead, "accuracy rises" (it is **maintained ~78%**), the small-object win (we are *lowest* on AR_s), and "helps in crowded/occluded scenes" (base wins those; honest axis = proposal-unreliable).

## 6. Decisions needed from prof
1. **Main model = 8B?** (4B kept as the compact scale-comparison point.)
2. **No DetZoom experiment** — answer R2 by mechanism + literature (CTG fires on proposal reliability, not size)?
3. **No non-MLLM / mAP experiment** — justify open-vocab as out-of-scope? *(highest residual risk — needs explicit buy-in)*
4. **Other-backbone (InternVL) ablation** — argue scope, or spend GPU time?
5. **LLM-Judge column blank for Ours** — produce the score, or drop the column?

## 7. Remaining work
- **Fig 1 `framework.png`** — retitle "SDS-GRPO Reward" → "CTG Reward", delete the duplicate instruction panel, add the reward-loop arrows (designer).
- Compile once → resolve §4.4 `\cref`/bib (`bercovich2025llamanemotro`, `singh2025openai`).
- Optional reviewer-defense adds: explicit Hard-stratum trade-off paragraph, DetZoom citation (`lookless2025`), open-vocab/mAP justification paragraph.
- Abstract still has no quantitative numbers (optional to add).

---
### Claim → evidence (for anything quoted)
- Referring best general VLM / Hard +4.14 → `tbl_action_referring.tex` — supported
- Grounding 11/12 cols vs 8B+proposal, @.5 +6.42 → `tbl_action_grounding.tex` — supported
- Selectivity 47.7 vs 6.7 → `compute_selectivity_and_dynamics.py` over `results-sftgrpo/` — supported
- Efficiency 9.44% / 1.19s median → `efficiency_figures_bundle/` + `depd/tbl_efficiency.tex` — supported (open-source rows only)
- SWIG grounding **overall AR** 29.05 vs base 38.32; HICO **Hard-split AR** ours 4.44 vs base 5.16 → tables — supported (concede)

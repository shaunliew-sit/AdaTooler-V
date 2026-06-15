# SAHA Reward v3 — Proposal-Conditioned Counterfactual Tool-Gain

**Date:** 2026-06-15
**Status:** APPROVED design (user, 2026-06-15). Authoritative for the **reward design and GRPO credit only**. Supersedes `docs/saha-v2/latest-decision.md` §8 (6-term reward) and §9.3 (Rung-2 SAN as the collapse fix). Everything else in `latest-decision.md` — proposals, `zoom_in`/`zoom_out`, evaluation protocol, falsification gates, SFT reuse — is retained.
**Validated by:** web novelty/collision agent + codex code-grounded audit (2026-06-15); outcomes in `plan/saha-v3-simple/notes.md`.
**Lineage:** prof directive 2026-06-15 — simplify to ≤2 reward variables, ground the tool reward in target-object GT only, make proposals the decision criterion for crowded/occluded scenes, align with the frozen SFT, fix GRPO tool-collapse.

---

## 1. Mechanism & Motivation (paper-ready)

**Thesis.** External object proposals do not merely hand the model a box. They (1) **discretize** a crowded scene into countable candidates, (2) supply a **localization prior** the VLM cannot produce over a hidden object, and (3) expose an **observable difficulty signal** the model reads to decide *when* to gather more evidence with a tool. The contribution is a reward that converts (3) into a trained, proposal-conditioned tool-use policy — because the single best proposal is usually **wrong precisely in crowded/occluded scenes**, and that error is the headroom the tool fills.

### 1.1 What breaks, and what the proposal fixes

| Scene | VLM failure without proposals | What the proposal contributes |
|---|---|---|
| **Crowded** (many people/objects) | Free-form grounding binds the *salient* instance, not the *queried* one → wrong person↔object pairing | Enumerates discrete candidates: "where is it?" (continuous search) becomes "**which of these?**" (verification) — the reliable operation where pairing errors get caught |
| **Occluded** (target half-hidden) | Model misses the target or defaults to the visible co-occurring object; boxes are loose | Detector fires on partially-visible objects via shape/context priors the VLM lacks → an **anchor box** to verify and tighten, instead of inventing a box over hidden pixels |

Non-target proposals (e.g. a knife/dining-table in a "cooking fish" scene) are **input context cues only — never scored**. We never label a proposal good or bad; we only have GT for the query's target.

### 1.2 The causal chain the reward exploits

1. In a crowded/occluded scene the best single proposal (the **anchor**, `s_ref`) is usually low — the detector binds the wrong instance or boxes the occluded object loosely.
2. The tool reward `R_tool = s_final − s_ref` therefore has **large positive headroom exactly in hard scenes**: zooming to disambiguate (crowded) or tighten over the hidden object (occluded) makes `s_final` beat the low anchor → large reward.
3. In a clean scene the anchor is already high → no headroom → zooming earns nothing → the model learns to trust the proposal.

The reward gradient **automatically concentrates tool-learning on the crowded/occluded slice**, with no hand-tuned occlusion threshold and no GT for proposal quality. The model learns a proposal-conditioned trigger: *when the proposal set looks crowded/conflicting/low-confidence (count, mutual overlap, confidence, boundary-touch — all model-observable), gather evidence; otherwise answer directly.* This learned, evidence-based trigger is the **robust decision criterion** the reviewers demanded.

### 1.3 Tool routing from proposal structure
- **Many overlapping candidate boxes near the target** = assignment/crowding problem → **`zoom_out`** for scene context to bind the right pair.
- **A single low-confidence / partial / boundary-touching box** = detail/occlusion problem → **`zoom_in`** to recover the hidden evidence.

The proposal set tells the model *which kind of hard* the scene is, so one policy handles both.

### 1.4 How this answers the reviewers (and stays honest)
- **R4 ("area alone can't capture occlusion/contact"):** difficulty now comes from proposal *evidence state*, not box area.
- **R2 ("is it just a deterministic size-based zoom?"):** DetZoom (size-only) is blind to crowding/occlusion → it should lose on exactly these strata. That contrast is the experiment proving reasoning adds value.
- **Falsification:** only real if zooming recovers evidence the proposal lacked. The hard-large / proposal-reliability diagnostics + the evidence-grounding ablation (does removing the tool observation change the answer?) verify it is not placebo. If SAHA does not beat proposal-only and DetZoom on the occluded/crowded rows, narrow the claim.

### 1.5 Tool-Trigger Mechanism & Reviewer Rebuttal (R2 / R4)

**Mechanism.** There is **no hand-coded trigger**. The policy emits `zoom_in`/`zoom_out` autonomously, conditioned on the observable proposals; `R_tool` trains it to zoom only when zooming beats trusting the proposal. The trigger is **learned, not verbalized** (the frozen SFT cannot verbalize a proposal rationale, and we do not reward one).

**Claim discipline (web-validated 2026-06-15).** "We learn *when* to zoom (vs a difficulty heuristic)" is the 2025-2026 field default (DeepEyes, Pixel-Reasoner, Chain-of-Focus, Mini-o3, VisRL) — **table-stakes, not a contribution**. Frame the novelty as the **proposal-relative necessity reward** (the tool must beat the *detector-proposal* anchor) **+ HOI proposal-quality stratification**, never as "learned triggering." See §5.

**R4 ("area-only SDS can't capture occlusion/contact").** Answered by *removing* the area heuristic: `s_ref` (the proposal-trust score) drops under occlusion, contact ambiguity, and proposal unreliability **regardless of object size**, so the trigger fires exactly where area-gating fails. There is no area threshold left to be wrong.

**R2 ("maybe a deterministic size-based zoom is enough").** Answered head-to-head: **DetZoom** (deterministic size zoom) is in every main table, and the learned trigger must beat it on the occluded/crowded/wrong-proposal strata. Supported by 2510.01681 ("beats tool-VLMs without handcrafted rules") and "Learning How Hard to Think" (ICLR 2025; heuristic-only gating lands far below the learned policy).

**Honest gate (DetZoom falsification).** If the trained tool-rate turns out to be ≈ a function of size, DetZoom ties us → we have NOT beaten the heuristic → narrow the claim. The proof of separation is a **read-only diagnostic**: tool-rate × occlusion / proposal-IoU strata (eval-only SDS-v2 factors). It adds **zero reward variables** and no SFT change.

**Drop-in rebuttal sentence:**
```text
Unlike area-based gating, our zoom policy is trained by a proposal-relative
necessity signal — rewarded only when zooming beats trusting the detector
proposal — which absorbs occlusion, contact ambiguity, and proposal
unreliability that area cannot. We show the learned trigger concentrates on
those cases at fixed object size and outperforms a deterministic size-based
zoom (DetZoom); heuristic-only gating provably underperforms learned necessity
policies [2510.01681; Learning How Hard to Think, ICLR 2025].
```

---

## 2. Reward Design (final form)

```
R_total = R_format · (R_outcome + α · R_tool)        # single knob: α (v1 default 0.6)

R_tool  = I_tool · clip( s_final − s_ref , −c_neg, +c_pos )     # counterfactual tool-gain
```
- `R_format`, `R_outcome`: **unchanged** from `verltool/verl_tool/workers/reward_manager/sds_grpo.py` (format gate; grounding `0.5·person_iou+0.5·object_iou` + AR; referring ROUGE-L/METEOR or judge).
- `I_tool` = 1 if ≥1 valid tool call, else 0 (derived from the existing `count_zoom_in + count_zoom_out > 0`; no new signal needed).
- `s_final` = outcome score of the final answer ∈ [0,1] (same scorer as `R_outcome`).
- `s_ref` = **no-tool / trust-the-proposal reference** for the same sample:
  - **Grounding:** `s_ref` = `compute_grounding_outcome` of the **selected proposal anchor box(es)** vs target GT. Offline, deterministic, **GT-anchored → policy cannot move it**.
  - **Referring:** `s_ref` = mean outcome of **no-tool rollouts in the same `uid` group** — a **detached constant** computed from sibling rollouts' already-computed outcome scores (rewards here are NumPy floats fed to GRPO as a precomputed tensor, so this is "detached w.r.t. the current rollout," not autograd stop-gradient). **Minimum-count guard:** require ≥1 no-tool sibling; if the group has none, drop `R_tool` for that sample (`R_tool=0`, reduces to v1 behavior). **`n=2` reality:** at the sweep's default `n=2` a `uid` group splits at most 1 tool / 1 no-tool, so referring `s_ref` is a single-sample mean (noisy) and is often dropped. Mitigation, predeclared in config: either (a) raise `n` for referring (e.g. `n=4`) to stabilize the no-tool mean, or (b) accept sparse referring `R_tool` and let standard GRPO advantage carry referring — decided empirically in the gate sweep.
- **Behavior:** tool beats the proposal-trust baseline → `+`; tool ties/hurts → `~0`/`−`; no tool → `R_tool = 0` (R_outcome carries it). The asymmetric `clip` keeps `R_tool` a **tie-breaker around R_outcome**, not a dominant term (Peak-Then-Collapse safeguard).

### 2.1 Grounding anchor selection
The anchor = the **target-matching proposal**: among the given proposals, the person/object pair whose boxes best match the query target (selected at preprocess time using target GT, since GT exists offline at training). `s_ref` = the score that pair would earn. The tool is rewarded only when the model's final answer beats **the best a proposal could give** → an honest "value beyond proposal-only" signal (claim C3). If this proves too strict in the gate sweep (tool rarely beats the best proposal), relax to the label/role-matched proposal; predeclare the choice in config.

### 2.2 The ≤2-variable promise
Tunable surface = `{α}` primarily, with `{c_neg, c_pos}` clip bounds as calibration (not per-sample heuristics). No per-factor SDS weights, no λ-vector, no decision/correction/copy/redundancy terms.

---

## 3. GRPO Tool-Collapse Fix

**Default = the counterfactual reward itself + standard vanilla GRPO advantage.** No SAN by default.

Rationale (both validators, decisive): vanilla GRPO advantage already subtracts the `uid`-group mean (`verl/.../core_algos.py:301-325`). Adding a two-stratum tool/no-tool SAN advantage on top **double-centers the same tool-vs-no-tool contrast** → redundant/unstable. The counterfactual reward already breaks collapse on its own: on hard samples no-tool rollouts score low and a useful tool earns `R_outcome + α·gain` → positive group advantage for tools *exactly where tools help*; on easy samples there is no gain to farm.

**Two-stratum SAN (Stratified GRPO, 2510.06214) is retained only as an ablation row**, run iff tool-rate curves on hard strata still show collapse after the reward-only default. It is never the default and never claimed as novel — cite 2510.06214 as its source.

---

## 4. Safeguards (literature-mandated)

1. **Detached `s_ref`.** Grounding `s_ref` is GT-anchored → immune to baseline-depression hacking. Referring `s_ref` = group no-tool mean is policy-coupled → compute it as a **detached constant from the completed rollout batch** (sibling rewards, no gradient path to the current rollout) + `clip`, to block the "keep no-tool siblings bad to farm gain" exploit.
2. **Asymmetric clip, small α.** Keep `R_tool` a tie-breaker; plot per-term reward share over training. If `R_tool`'s share grows, that is a hacking signal, not progress (Peak-Then-Collapse 2605.26037).
3. **No double-count.** `s_ref` and any SAN stratum baseline must never both be "group no-tool mean." Default ships reward-only; SAN is a separate ablation.
4. **Per-rollout JSONL logging** of `s_final, s_ref, tool_gain_raw, tool_gain_clipped, i_tool, n_zoom_in, n_zoom_out, uid, task_type` — the diagnostic backbone for the falsification gates.

---

## 5. Novelty & Citations

- **CLAIM DISCIPLINE (web-validated 2026-06-15):** "we *learn* when to zoom (vs a hand-crafted difficulty heuristic)" is **table-stakes, NOT a contribution** — it is the 2025-2026 field default (DeepEyes 2505.14362, Pixel-Reasoner 2505.15966, Chain-of-Focus 2505.15436, Mini-o3 2509.07969, VisRL 2503.07523 all learn the trigger). **Do not pitch the paper as "learned adaptive triggering."** The contribution is the **proposal-relative reward** (`s_ref` = trusting the proposal; the tool must beat *that*) + **HOI proposal-quality stratification** + the fair proposal-only/DetZoom baselines.
- **Defensible novel kernel:** a **GT-anchored, proposal-relative "beat-the-proposal-baseline" counterfactual tool reward** in HOI. No exact prior found.
- **Closest sibling — cite + differentiate: "Look Less, Reason More" (2510.01681).** Its **Pixel Necessity Rollouts** (success with vs without the operation) is conceptually the same "is the tool truly beneficial" signal as our `s_final − s_ref`. Differentiation: their baseline is a **no-op / direct-answer** reference; ours is the **external-proposal anchor** (`s_ref` = trust the detector proposal), and we condition/stratify on **proposal quality** in **HOI** — they do neither. This is now the FIRST citation to manage after AdaTooler-V.
- **Differentiate from AdaTooler-V / AT-GRPO (2512.16918):** ours is an **online, per-rollout, proposal-anchored** counterfactual; theirs is an **offline frozen-72B-teacher scalar ΔS** scaling the whole sample. Do not adopt their reward form as the main method.
- **Cite Stratified GRPO (2510.06214)** as the source of the SAN ablation — do not claim it.
- **Ammunition for reviewer R2 ("is it just a size heuristic?"):** "Look Less, Reason More" (2510.01681) beats tool-augmented VLMs "without handcrafted rules"; "Learning How Hard to Think" (ICLR 2025) shows an entropy/heuristic-only gating policy lands "far below the full learned adapter." Cite both to argue a hand-crafted scalar gate cannot recover the learned policy's gains.
- **Uncertainty/confidence gating** (Adaptive VLM Routing 2603.12823, etc.) is **orthogonal and weaker** than a learned policy — cite as future work, **do NOT implement** (violates the ≤2-variable + frozen-SFT constraints).
- **Citation-verification TODO (before any draft):** confirm arXiv IDs — AutoTool is **2512.13278 / 2511.14650** (not 2605.19852); Proof-of-Use (2510.10931) and 2510.01681 / 2603.12823 to be re-verified. Run the `citation-verification` pass.

---

## 6. What Changes vs Rev-4 (and which CLAUDE.md constraints this supersedes)

**Removed from the reward:** `R_proposal_correction`, `R_proposal_decision`, `R_bad_proposal_copy`, `R_redundant_tool_use` (as separate terms), the 5-factor SDS-v2-as-reward, and the `proposal_action` (trust/correct/reject) verbalization (frozen SFT can't emit it — R6 gate).

**SDS-v2 factors survive eval-only** as stratification axes (size/occlusion/contact/proposal/scene), never as reward coefficients.

**Superseded CLAUDE.md hard-constraints (to update on this approval):**
- ~~"6-term proposal-first hierarchy `min(λ_corr,λ_copy) ≥ λ_tool`"~~ → obsolete (those terms removed).
- ~~"every rollout JSONL carries all five SDS-v2 factors + proposal_action"~~ → replaced by the §4 counterfactual logging schema; SDS-v2 factors logged for eval strata only.
- ~~"§9.3 Rung-2 SAN = the fix for GRPO tool-collapse"~~ → counterfactual reward is the default fix; SAN demoted to ablation.

**Still in force (unchanged):** keep proposals (YOLOE); `zoom_in`/`zoom_out` only; no `inspect_pose`; no detector-free; no AXPO; no new-GRPO-algorithm claim; reuse v1 SFT (no trace generation without a recorded human decision); never adopt AT-GRPO ΔS as the main method.

---

## 7. Implementation Change Set (from codex audit — for the writing-plans phase)

1. **`examples/data_preprocess/hoi/prepare_train.py`** — add the selected proposal **anchor pair** as a structured field in `extra_info` / `reward_model.ground_truth` (proposals are prompt-text only today). **Forces a one-time parquet re-run.**
2. **`workers/reward_manager/sds_grpo.py`** (or a new `saha_cf.py` registered manager) —
   - new helper `compute_counterfactual_tool_reward(s_final, s_ref, i_tool, clip_lo, clip_hi)`;
   - new helper to serialize the proposal anchor pair into parser-compatible text → score via existing `compute_grounding_outcome` (promote single box → person+object pair; coords already 1000×1000);
   - restructure `__call__` (currently a per-item loop, `:486-583`) into **two passes** so referring `s_ref` can use the per-`uid` no-tool mean;
   - extend `score_dict` logging (§4 schema).
3. **`sds_v2.py`** — no change; remains registered but unused by the reward (eval-only factor lib).
4. **Config** — `α`, `c_neg`, `c_pos`, grounding-anchor-selection mode, referring-`s_ref` fallback flag.
5. **Keep `sds_grpo.py` (`SDS-GRPO`) registered and untouched** as the frozen "old area-SDS" baseline row.

---

## 8. Experiment Plan

- **Substrate:** resume GRPO from `/workspace/hoi/checkpoints/qwen3VL-4B/hoi_v2_sft` on a **balanced ~2–3k subset** (stratify by scene crowding / proposal quality so hard cases aren't washed out).
- **Sweep (parallel/sequential short runs on one 96 GB GPU):** `α ∈ {0.3, 0.6, 1.0}`, clip bounds, grounding-anchor mode, referring-`s_ref` variant. Single-GPU recipe: full FT + CPU offload, n=2, batch 2.
- **Selection:** winner by val outcome (AR/F1, referring score) **and** tool-gain on hard strata, with a non-collapsed tool-rate curve.
- **Scale:** promote the winning reward config to 8B (`qwen3VL-8B/hoi_v2_sft`) for the final tables.

---

## 9. Falsification Gates (retained from latest-decision.md)

Proposal-only gate · Hard-large gate · DetZoom gate · Good-proposal gate · Logging gate · Evidence-grounding gate. (Decision-parse and stratum-stability gates are dropped — the terms they guarded no longer exist.) If a gate fails, narrow the claim rather than ignore it.

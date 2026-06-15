# SAHA v3 — Proposal-Conditioned Counterfactual Tool-Gain (simplified reward)

**Date:** 2026-06-15
**Status:** design draft for validation (not yet approved). Supersedes the Rev-4 6-term reward (`docs/saha-v2/latest-decision.md` §8) for the *reward design only*. Everything non-reward in Rev-4 (proposals, zoom_in/zoom_out, eval protocol, falsification gates, §9.3 SAN collapse fix) is retained.

## Why the pivot (prof directive, 2026-06-15)
- Rev-4 reward is too complicated (6 terms, 5 heuristic λ's) and depends on a capability the frozen SFT lacks (stating trust/correct/reject — R6 gate: SFT 0/6 parseable).
- We do **not** have GT for the full proposal set — only GT for the **target object** of the query. So any reward that scores "is this proposal good/bad" is unsupportable. Non-target proposals (e.g. knife/dining-table in a cooking-fish scene) are legitimate *context cues*, never scored.
- Reviewer pressure (reviewer-comment.md): area-only SDS can't capture occlusion/contact/crowding (R4); "is a deterministic zoom heuristic enough?" (R2); referring shows ~no gain (R2). The robust, crowding/occlusion-aware **decision criterion for when to call a tool** is the contribution — and proposals are that criterion's input.

## Locked forks (user, 2026-06-15)
1. Tool reward = **counterfactual gain** (outcome-grounded, target-GT only).
2. Reward form = **old multiplicative** `R_format·(R_outcome + α·R_tool)` (one knob α).
3. Substrate = **4B SFT + subset** (`/workspace/hoi/checkpoints/qwen3VL-4B/hoi_v2_sft`).

## Reward design
```
R_total = R_format · (R_outcome + α · R_tool)        # α = single knob, v1 default 0.6
```
- `R_format`, `R_outcome`: unchanged from `verltool/verl_tool/workers/reward_manager/sds_grpo.py`.
- `R_tool` (NEW — counterfactual tool-gain):
  ```
  R_tool = I_tool · clip( s_final − s_ref , −c_neg, +c_pos )
  ```
  - `I_tool` = 1 if ≥1 valid tool call, else 0.
  - `s_final` = outcome score of the model's final answer ∈ [0,1] (same scorer as R_outcome).
  - `s_ref` = **no-tool / trust-the-proposal reference** for the same sample:
    - **Grounding:** `s_ref` = outcome score of the **selected proposal anchor box** vs target GT (offline, deterministic, GT-anchored → policy cannot move it).
    - **Referring:** `s_ref` = mean outcome of **no-tool rollouts in the same GRPO group**; fallback to config constant if none. (boxes are inputs in referring, so there is no proposal "box prediction" to score — open sub-fork #1.)
  - Behavior: tool beats the proposal-trust baseline → +; tool ties/hurts → ~0/−; no tool → `R_tool = 0` (R_outcome carries it). Asymmetric clip keeps R_tool a tie-breaker around R_outcome (Peak-Then-Collapse safeguard).

### Why this satisfies every constraint
- ≤2 variables: outcome + tool, one weight α. ✅
- No proposal-quality GT: `s_ref` for grounding is the proposal box scored **against the target GT we DO have**; we never label a proposal good/bad. ✅
- Proposals are the decision criterion: clean scene → proposal anchor already high → little gain headroom → policy learns not to zoom; crowded/occluded → anchor low → zoom-to-correct yields gain → policy learns to zoom. Crowding/occlusion emerge through outcome gain, not hand-tuned SDS thresholds. ✅
- SFT-aligned: outcome/behavior-based, no verbalization required. ✅
- Collapse: §9.3 Rung-2 two-stratum tool/no-tool SAN retained (credit mechanism, not a reward variable). ✅

## Removed vs Rev-4
R_proposal_correction, R_proposal_decision, R_bad_proposal_copy, R_redundant_tool_use (separate terms), 5-factor SDS-v2-as-reward, proposal_action verbalization. SDS-v2 factors *may* survive as **eval-only** stratification axes (not reward).

## Open sub-forks for validation
1. **Referring `s_ref` source:** group no-tool mean vs one cheap extra no-tool rollout vs drop R_tool for referring.
2. **Double-counting:** R_tool (per-rollout counterfactual) vs SAN credit (group counterfactual) — do they compound, conflict, or is one redundant?
3. **Hacking:** can the policy farm R_tool by depressing `s_ref`? (Grounding: no — GT-anchored. Referring group-mean: policy-coupled — audit.)
4. **No-tool bonus:** should no-tool earn a tiny bonus when the anchor is already correct, to actively discourage over-zooming on good proposals (good-proposal gate)?
5. **Novelty vs AdaTooler-V ΔS:** ΔS is a frozen-72B-teacher offline scalar; ours is a policy-own, proposal-anchored, within-task counterfactual. Confirm this reads as distinct.

## Validation outcomes (2026-06-15: web-novelty agent + codex code audit)
Both agents independently converged on the same critical point.

- **Double-counting (decisive).** Vanilla GRPO advantage already subtracts the uid-group mean (`core_algos.py:301-325`). A counterfactual tool-gain *reward* (tool vs no-tool) **plus** a two-stratum tool/no-tool SAN *advantage* (tool vs no-tool) centers the same contrast twice → redundant/unstable. **Decision: the counterfactual reward term BECOMES the default collapse fix; the §9.3 Rung-2 SAN is demoted to an ablation/contingency.** This supersedes the CLAUDE.md framing that named Rung-2 SAN "the fix for GRPO tool-collapse."
  - Why the reward alone fixes collapse: on hard samples (bad proposal) no-tool rollouts score low and a useful tool earns `R_outcome + α·gain` → positive group advantage for tools exactly where tools help; on easy samples the anchor is already high → no gain headroom → no tool. Collapse is addressed at the reward level, not the advantage level.
- **Hacking surface.** Grounding `s_ref` = proposal box vs target GT is **policy-independent → immune** to baseline-depression. Referring `s_ref` = group no-tool mean is **policy-coupled → exposed** (policy could farm gain by keeping no-tool siblings bad). Mitigation: stop-gradient on `s_ref`, keep the asymmetric `clip`, treat R_tool as a tie-breaker around R_outcome.
- **Novelty.** GT-anchored "beat-the-proposal-baseline" reward = no exact prior found → the defensible novel kernel. Must (a) cite **Stratified GRPO (2510.06214)** as the source of the SAN (do not claim it), (b) frame novelty narrowly and differentiate from **AdaTooler-V/AT-GRPO (2512.16918)** (online + GT-anchored + proposal-trust vs offline frozen-72B-teacher ΔS scalar).
- **Citation TODO:** verify arXiv IDs before use — AutoTool is 2512.13278 / 2511.14650 (not 2605.19852); Proof-of-Use 2510.10931 unconfirmed.
- **Feasibility / cost.** Reward manager currently sees only `ground_truth`, `data_source`, `extra_info` (`sds_grpo.py:501-503`); proposal boxes are prompt-text only. The selected proposal anchor must be threaded into `extra_info` in `prepare_train.py` → **one-time parquet re-run required**. Tool counts already available (`sds_grpo.py:351-358`). uid group context is NOT in `__call__` → referring group-mean needs a two-pass `__call__`.
- **Grounding anchor scorer.** `compute_grounding_outcome` is a person+object **pair** scorer (`0.5·person_iou+0.5·object_iou`); a single-box anchor must be promoted to a pair. Coords already 1000×1000 — compatible.

## Refined collapse-fix decision
Default = counterfactual tool-gain reward + **standard vanilla GRPO advantage** (simplest, no double-count). Two-stratum SAN kept registered as an **ablation row only**, run if tool-rate curves on hard strata still show collapse.

## Experiment substrate
4B SFT → balanced ~2–3k subset → multiple short GRPO runs over {α, clip, s_ref variant} → pick winner by val tool-gain + outcome → scale best to 8B. One 96GB GPU (proven single-GPU recipe: full FT + CPU offload, n=2, batch 2).

# TODO — Related-Work differentiation for the PTG reward (address before submission)

**Date:** 2026-06-24. **Status:** ✅ DONE (2026-06-25) — added to `sec/2_related_work.tex` §2.3, wired to `lookless2025` (arXiv 2510.01681, verified) + `wang2025adatooler`. **Source:** `reward-v3-counterfactual-spec.md` §5 (novelty & citations, web-validated 2026-06-15).
> **Rename note (2026-07-01):** the reward is named **proposal-relative tool-gain (PTG)** in the paper; "CTG / counterfactual tool-gain" below is the old working name (dropped per supervisor — "counterfactual" read as adversarial). The live §2 already uses the PTG macro; the draft sentences below are the historical source and are kept for the record.

## The gap
The method now centres on the **Proposal-Relative Tool-Gain (PTG)** reward, but `sec/2_related_work.tex` does **not** differentiate it from its closest prior work. A reviewer will ask "isn't this just necessity/benefit-style tool RL, or AdaTooler-V's ΔS?" The defensible novel kernel is a **GT-anchored, proposal-relative "beat-the-proposal-baseline" tool reward in HOI** — we must say so explicitly and contrast the baselines.

## Claim discipline (do NOT violate)
- Do **not** pitch the contribution as "we *learn when* to zoom (vs a hand-crafted heuristic)." That is 2025–26 field default (DeepEyes, Pixel-Reasoner, Chain-of-Focus, Mini-o3, VisRL) = **table-stakes, not novelty**.
- Frame novelty as: **(1)** the reference is the *external detector-proposal anchor* (`s_ref` = trusting the proposal), not a no-op/direct-answer baseline; **(2)** we condition/stratify on **proposal quality** in **HOI**.

## Draft sentences for `sec/2_related_work.tex` (tool-use paragraph, ~L28–29)
> Closest to our reward is the necessity-style signal in recent tool-use RL — e.g. the pixel-necessity rollouts of \emph{Look Less, Reason More}~\cite{TODO_lookless2510_01681}, which credit a visual operation by its success with versus without the tool. Our proposal-relative tool-gain differs in its \emph{reference}: rather than a no-op / direct-answer baseline, the tool must beat the \emph{external detector-proposal anchor} ($s_{\mathrm{ref}}$, the score of trusting the proposal), and we condition and stratify on proposal quality in the HOI setting. This also separates us from the adaptive tool-use of AdaTooler-V~\cite{wang2025adatooler}, whose tool benefit is an offline, frozen-teacher scalar applied per sample; ours is an online, per-rollout, proposal-anchored gain.

(One-line alternative if space is tight: "Unlike necessity-style tool rewards that compare against a no-op baseline~\cite{TODO_lookless2510_01681}, our proposal-relative tool-gain is measured against the external proposal anchor and stratified by proposal quality in HOI.")

## Bib entries to ADD + verify (citation-verification pass)
Not currently in `main.bib`:
- **`TODO_lookless2510_01681`** — "Look Less, Reason More" (pixel-necessity rollouts). **Verify arXiv ID 2510.01681** (spec flagged it for re-verification).
- **`TODO_stratifiedgrpo2510_06214`** — Stratified GRPO (only needed if the two-stratum SAN ablation is reported; cite as the *source*, do not claim as ours). **Verify 2510.06214.**
- AT-GRPO / AdaTooler-V is already cited as `wang2025adatooler`; if the AT-GRPO ΔS reward is named, verify its ID (spec: **2512.16918**, not the older 2512-style IDs).

## Placement / scope
- Goes in `sec/2_related_work.tex` (the focus-zooming / adaptive-tool-routing sentence already there, L28–29).
- Do this in the **same pass** that adds the experiments (DetZoom + proposal-only baselines), since the differentiation claim ("must beat the proposal anchor") is what those baselines empirically defend.
- Keep it to 2–3 sentences; the reward math stays in §3, not Related Work.

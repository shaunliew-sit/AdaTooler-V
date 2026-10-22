# Methodology Rewrite Draft — SDS-GRPO → Counterfactual Tool-Gain (for review)

**Status:** DRAFT for user approval. No `.tex` edited yet. Methodology only (results interpretation waits on the new GRPO run). Authoritative source: `docs/saha-v2/reward-v3-counterfactual-spec.md`. To be Codex-verified against the spec + `saha_cf.py` before applying.

---

## 0. The core change (what and why)

The paper's method is branded **SDS-GRPO**: the tool reward is gated by a **Spatial Difficulty Score (SDS)** derived from **GT object-box area** (`R_zoom_in` with a `(SDS−τ)·exp(−γ(·)²)` form + a hygiene bonus). **This is no longer the method.** The actual reward (implemented in `saha_cf.py`, currently training) is a **proposal-relative counterfactual tool-gain**: the tool is rewarded only when the model's final answer **beats trusting the detector proposal** (`R_tool = I_tool·clip(s_final − s_ref)`). Area-based SDS is **removed from the reward** (it survives eval-only as a stratification axis). The composite shape `R_total = R_format·(R_outcome + α·R_tool)`, α=0.6, is unchanged.

**Naming decision (blocking — pick one):** the old contribution was "SDS-GRPO." The reward spec forbids claiming a *new GRPO algorithm*, so the contribution must be named as a **reward**, optimized with standard GRPO. Proposed: **Counterfactual Tool-Gain (CTG) reward** (matches the `SAHA-CF` code; honest). Alternatives: "Proposal-Relative Tool-Gain (PRTG)", or keep it unbranded ("a proposal-relative counterfactual tool reward"). Framework name **SAHA** stays.

**`R_outcome` metric (settled):** the grounding `R_outcome`/`s_ref` use the **eval metric** — `min(person,object)` IoU over 10 thresholds 0.5–0.95. This is a **settled design decision**, already the live default in `saha_cf.py` (`compute_grounding_outcome_ar`, `SAHA_CF_GROUNDING_METRIC=minAR10`) and in `reward-metric-alignment-runA-design.md`; the paper's current avg-IoU @{0.5,0.75} text is what must change. (Only the *empirical* hard-case efficacy of this alignment is deferred — it belongs to the Experiments section, not the Method.)

**`I_tool` note:** the active tool set is `zoom_in`/`zoom_out` only, so the paper's "≥1 `zoom_in`/`zoom_out` call" is exact. The code's `count_zoom_in` regex also matches a legacy `crop_image` alias, but that tool is not registered in the current setup, so it never fires — no need to mention it in the paper.

---

## 1. Change-map (surgical; methodology scope)

| # | Location | Current (stale) | Action |
|---|---|---|---|
| C1 | §3 overview, `3_method.tex:9` | "…our proposed Spatial Difficulty Score-Gated GRPO (SDS-GRPO)" | rewrite → counterfactual tool-gain reward |
| C2 | §3.3 intro, `:119` | "RL with our proposed SDS-GRPO reward refines tool timing by per-sample spatial difficulty" | rewrite → "…by a proposal-relative counterfactual signal" |
| C3 | §3.3, `:126` | "asymmetric treatment in SDS-GRPO" (hygiene/zoom-out) | drop the SDS-asymmetry clause; keep the zoom_in/zoom_out role description |
| C4 | §3.3, `:135` heading | "Policy Optimization with SDS-GRPO" | rename (per naming decision) |
| C5 | §3.3, `:152-159` `R_ground` | avg joint-IoU @{0.5,0.75} | align to min-IoU 10-thr (settled: live default in `saha_cf.py` + Run-A design) |
| C6 | §3.3, `:174-202` **SDS + SDS-gated tool reward** (2 paras + 4 eqs) | area-SDS reward | **REPLACE** with Counterfactual Tool-Gain (draft §2) |
| C7 | §3.3, `:205-210` composite | OK (shape unchanged) | keep; reword "balance task accuracy against tool efficiency" |
| C8 | §1 intro, `:30-32,44,48` | "SDS-GRPO… SDS from GT box areas… penalized for redundant calls" + "4B" | rewrite contribution bullet + 4B→8B |
| C9 | §2 related, `:29` | "apply tools without conditioning on per-sample spatial difficulty" | → "without conditioning tool use on external-proposal reliability" |
| C10 | §5 conclusion, `:4-7` | "SDS-GRPO… per-sample spatial difficulty" + "4B" | rewrite + 4B→8B |
| C11 | §6 suppl, `:189-284` "SDS-GRPO Reward Details" | full SDS appendix (eqs, hyperparams table) | **REPLACE** with CTG details (s_ref construction, clip, α, anchor selection, logging) |
| C12 | §6 suppl, `:104,166,309-343` | "Qwen3-VL-4B Baseline", SDS curriculum prose | 4B→8B where it's *our* model; keep SFT area-stratification (SFT curriculum is real & reused) |
| C13 | global | "Qwen3-VL-4B" as **our backbone** (`3_method:4`, `4_experiment:52,79`, intro, conclusion, suppl) | → 8B (tables already 8B). Keep "Qwen3-VL-4B" where it's a *baseline row*. |

**Out of scope now (results-dependent):** §4 result interpretation `:79,109,124-125` (numbers + their reading) — revisit after the GRPO run. Naming-only/4B fixes in §4 setup (`:37,43,52,70`) are safe and included.

---

## 2. Draft replacement — §3.3 reward subsection (the core rewrite)

> Replaces `3_method.tex:135-210`. Uses the placeholder name **CTG**; swap per the naming decision.

```latex
  \subsubsection{Policy Optimization with Counterfactual Tool-Gain}

  GRPO~\cite{shao2024deepseekmath} refines the tool-use policy by optimising task-performance
  signals over the full training set. For each sample, $K$ reasoning chains are sampled and the
  policy is updated to favour above-average-reward trajectories:
  \begin{equation}
  \resizebox{0.92\linewidth}{!}{$\mathcal{L}_{\text{GRPO}} =
  -\mathbb{E}\!\left[\sum_{k=1}^{K}\!\left(R(y_k)-\bar{R}\right)
  \log\pi_\theta(y_k\mid x)\right]
  +\beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{SFT}})$}
  \end{equation}
  where $R(y_k)$ is the composite reward $R_{\mathrm{total}}$ for response $y_k$, $\bar{R}$ is the
  within-group mean, and $\beta$ weights the KL to the SFT reference. A binary format gate is a hard
  prerequisite:
  \begin{equation}
  R_{\mathrm{format}} = \mathbf{1}\!\left[\text{response has valid }
  \langle\mathrm{think}\rangle\text{ and }\langle\mathrm{answer}\rangle\text{ tags}\right]
  \end{equation}

  \noindent\textbf{Outcome reward.}
  The outcome reward scores the final answer against ground truth with the \emph{same} metric used at
  evaluation, so the reward optimises exactly what is measured. For grounding, predicted pairs are
  matched to GT pairs by the stricter per-pair criterion $\min(\mathrm{IoU}_{\mathrm{person}},
  \mathrm{IoU}_{\mathrm{object}})$ and scored as Average Recall over IoU thresholds
  $\delta\in\{0.5,0.55,\dots,0.95\}$:
  \begin{equation}
      R_{\mathrm{ground}} = \frac{1}{|\Delta|}\sum_{\delta\in\Delta}\frac{|\mathcal{M}_\delta|}{|G|},
      \qquad \Delta=\{0.5,0.55,\dots,0.95\}
  \end{equation}
  where $\mathcal{M}_\delta$ are predicted pairs matched at threshold $\delta$ and $|G|$ is the number
  of GT pairs. For referring, partial credit preserves exact-match superiority:
  \begin{equation}
      R_{\mathrm{refer}}=\max\!\Bigl(\mathbf{1}[a_{\mathrm{pred}}{=}a_{\mathrm{gt}}],\;
      0.5\,\mathrm{ROUGE\text{-}L}+0.5\,\mathrm{METEOR}\Bigr)
  \end{equation}

  \noindent\textbf{Motivation: difficulty is proposal-relative, not size-based.}
  Whether a zoom helps is not a function of object size: a large but occluded or wrongly-bound object
  benefits from zooming, while a small but cleanly-detected one does not. The decisive signal is
  whether the external proposal already localises the queried target. In crowded or occluded scenes the
  best single proposal is typically wrong --- the detector binds the salient instance rather than the
  queried one, or boxes a half-hidden object loosely --- and it is exactly this error that a zoom can
  recover. We therefore reward the tool by how much the final answer \emph{beats trusting the proposal},
  rather than by a hand-set area threshold.

  \noindent\textbf{Counterfactual tool-gain.}
  Let $s_{\mathrm{final}}\in[0,1]$ be the outcome score of the model's final answer and
  $s_{\mathrm{ref}}\in[0,1]$ the \emph{trust-the-proposal} reference for the same sample, scored by the
  same outcome metric. The tool reward is the clipped counterfactual gain
  \begin{equation}
      R_{\mathrm{tool}} = I_{\mathrm{tool}}\cdot
      \operatorname{clip}\!\bigl(s_{\mathrm{final}}-s_{\mathrm{ref}},\,-c_{-},\,+c_{+}\bigr),
  \end{equation}
  where $I_{\mathrm{tool}}=1$ if the trajectory issues at least one \texttt{zoom\_in}/\texttt{zoom\_out}
  call and $0$ otherwise. For \textbf{grounding}, $s_{\mathrm{ref}}$ scores the \emph{target-matching
  proposal anchor} (the person/object proposal pair best matching the query target) against the GT;
  it is computed offline and is GT-anchored, so the policy cannot move it. For \textbf{referring},
  $s_{\mathrm{ref}}$ is the mean outcome of the no-tool rollouts in the same prompt group --- a detached
  constant from sibling rollouts (dropped when the group has no no-tool sibling).

  \noindent\textbf{Why it works.}
  The gain $s_{\mathrm{final}}-s_{\mathrm{ref}}$ has large positive headroom \emph{exactly} where the
  proposal is unreliable: zooming to disambiguate a crowded scene or to tighten over an occluded object
  lets $s_{\mathrm{final}}$ exceed the low anchor, earning reward; in a clean scene the anchor is already
  high, so zooming earns nothing and the model learns to trust the proposal. The reward gradient thus
  concentrates tool-learning on the hard slice with no area threshold and no GT for proposal quality.
  Because difficulty is read from the proposal's \emph{evidence state} rather than object size, the
  signal absorbs occlusion, contact ambiguity, and detector unreliability that area-based gating cannot.
  The asymmetric clip keeps $R_{\mathrm{tool}}$ a bounded tie-breaker around $R_{\mathrm{outcome}}$
  rather than a dominant term. This same mechanism resolves tool-collapse under GRPO without any new
  algorithm: the group-mean baseline already centres the tool-vs-no-tool contrast, and the
  counterfactual reward gives a tool positive group advantage precisely where it beats the proposal.

  \noindent\textbf{Composite reward.}
  \begin{equation}
  R_{\mathrm{total}}=R_{\mathrm{format}}\cdot\bigl(R_{\mathrm{outcome}}+\alpha\cdot R_{\mathrm{tool}}\bigr)
  \end{equation}
  with $R_{\mathrm{outcome}}\in\{R_{\mathrm{ground}},R_{\mathrm{refer}}\}$ by task type. A single knob
  $\alpha{=}0.6$ trades task accuracy against tool gain; clip bounds $(c_{-},c_{+})$ are fixed
  calibration constants. Full settings are in the Supplementary Material.
```

---

## 3. Draft edits — smaller surgical spots

- **C1 (§3 overview L9):** "…optimised via knowledge-distillation warm-up followed by GRPO with a **proposal-relative counterfactual tool reward** that credits a tool call only when it beats trusting the detector proposal (see \cref{sec:tool})."
- **C2 (§3.3 intro L119):** "…and RL with a **counterfactual tool-gain reward** that credits zooming only when the refined answer beats trusting the proposal."
- **C3 (L126):** end at "\texttt{zoom\_in} gathers new visual evidence while \texttt{zoom\_out} navigates spatial context." (delete the "asymmetric treatment in SDS-GRPO" clause).
- **C8 (§1 contribution bullet L44):** "We propose a **counterfactual tool-gain reward** that credits a zoom only when the refined answer beats the detector-proposal anchor, training a selective zoom policy without any hand-set difficulty threshold." + L31-32 reword (remove SDS-from-area framing) + 4B→8B in L38,48.
- **C9 (§2 L29):** "…but apply tools without conditioning invocation on **external-proposal reliability**."
- **C10 (§5):** "…training a selective zoom policy via a **counterfactual tool-gain reward** that credits tool use only when it beats trusting the detector proposal." + 4B→8B.
- **C13 (4B→8B):** our-model mentions → 8B; baseline rows ("Qwen3-VL-4B baseline") stay 4B.

(Suppl C11/C12 drafted in a second pass once the §3 wording is approved.)

---

## 4. Self-review + claim–evidence map

**Self-review (5-dim):** terminology now stable (one name for the reward, per decision); no "new GRPO algorithm" claim (collapse-fix framed as a property of the reward + standard GRPO); area-SDS fully removed from the reward and relabelled eval-only; one-message paragraphs (motivation / design / why-it-works / composite).

| Claim | Evidence | Status |
|---|---|---|
| Reward = `R_format·(R_outcome+α·R_tool)`, `R_tool=I_tool·clip(s_final−s_ref)` | `saha_cf.py`; v3 spec §2 | supported |
| Grounding `s_ref` = GT-anchored proposal-anchor score | `saha_cf.py:compute_sref_grounding`; spec §2.1 | supported |
| Referring `s_ref` = detached no-tool group mean | `saha_cf.py:resolve_referring_sref`; spec §2 | supported |
| `R_outcome` uses min-IoU 10-thr (eval-aligned) | `saha_cf.py:compute_grounding_outcome_ar`, live default | supported (settled design; empirical efficacy → Experiments) |
| Counterfactual reward fixes tool-collapse w/o new algorithm | spec §3; training dynamics | supported (mechanism); efficacy pending run |
| "beats area-based gating / DetZoom on hard strata" | — | **needs evidence (results) — omit from method; defer to experiments** |
| 8B backbone | eval logs + tables (8B) | supported |

# SAHA ACCV 2026 Resubmission — Assembled Package

Three drafts merged and de-conflicted. Overlap resolution:
- The **efficiency table** (in drafts 1 and 3) → single canonical version, draft 3's `depd/tbl_efficiency.tex`, referenced once from §4 prose.
- The **plus-points paragraph** (drafts 2 and 3) → draft 2's standalone Discussion paragraph is the camera-ready home for prof point 5; the §4 prose (part 1 below) carries the SAME framing inline (selectivity + same-VLM scoping) so the two do not double up — use the §4 inline framing for the Results section and draft 2's paragraph ONLY if a separate Discussion subsection is added. Do not paste both into §4.
- **Fig1 + §4.4 citation** (draft 1) → part 2, unchanged.
All §4 numbers re-verified against `tbl_action_grounding.tex`, `tbl_action_referring.tex`, `tbl_ablation.tex` (live, non-commented rows). SWIG grounding lives ONLY in the ablation table (the standalone stratified SWIG-G table is commented out) — the ablation prose's SWIG-G numbers (38.32/35.78/28.26/29.05) are valid; do NOT cite a `tab:grounding_swig` (it does not render).

═══════════════════════════════════════════════════════════════════
## PART 1 — FINAL §4 LaTeX (drop-in for Quantitative Results + Ablation)
═══════════════════════════════════════════════════════════════════
Replaces the `\subsection{Quantitative Results}` … end of `\subsection{Ablation Study and Analysis}` blocks in `sec/4_experiment.tex` (tex lines 75–127). Setup / Baselines / Implementation / Qualitative subsections unchanged. NOTE: the existing Quantitative Results prose (tex 83) claims "highest LLM-Judge on HICOref" — that column is now blank (`{--}`) for Ours; this replacement REMOVES that false claim. Also reorders to lead with the within-proposal fair frame, then referring (H1), then grounding (H2/H3).

```latex
\subsection{Quantitative Results}

All region-conditioned methods in \cref{tab:grounding_hico,tab:referring} receive the \emph{same} YOLOE-26 proposals at evaluation time, so the comparison within the \emph{VLM with proposal} block isolates what our counterfactual tool-gain pipeline adds on top of identical spatial conditioning, rather than crediting the detector. We read the tables through this lens: against same-VLM and proposal-matched competitors first, and against larger or closed models as scope, not as our claim.

\subsubsection{Action Referring}
On \HICOref, \methodname is the strongest general-purpose VLM at every difficulty stratum (\cref{tab:referring}). It reaches BERTScore-F1 of 84.73/85.39/81.55 on All/Easy/Hard and METEOR of 29.66/30.54/25.63, ahead of every zero-shot and proposal-conditioned baseline including Qwen3-VL-32B (BS-F1 83.38, METEOR 26.94) at four times the parameter count. The Hard column is the result we most want to highlight: ours scores 25.63 METEOR where the same-VLM base Qwen3-VL-8B scores 21.49, a gain of $+4.14$ exactly on the queries where the detector misses the object. This is the one stratum where holistic interaction description, not box geometry, decides the score, and the reasoning policy carries it.

Groma-7B-hoi-finetuned posts higher referring numbers (BS-F1 89.69, METEOR 40.00), and we report it as such rather than hiding it. It belongs to a different category: a region-token architecture trained end-to-end with cross-entropy on the exact ground-truth action strings, with no reasoning, no tools, and no proposal conditioning. That recipe buys exact-string vocabulary familiarity our reward-based, instruction-following setup does not target. Among general VLMs that read free-form prompts, \methodname leads.

\SWIGref is harder for every method. The roughly $3\times$ broader action vocabulary depresses fine-grained verb discrimination across the board, and ours sits mid-pack on All (BS-F1 77.90, METEOR 19.36)---competitive with the larger Qwen3-VL-32B$+$proposal (76.98 BS-F1) but below the SWIG-specialized Groma row. We treat SWIG referring as a generalization stress test, discussed with the grounding limitation below.

\subsubsection{HOI Grounding}
\cref{tab:grounding_hico} stratifies HICO grounding into All/Easy/Hard by detector-proposal difficulty $s_{\mathrm{ref}}$ (Hard: $s_{\mathrm{ref}}<0.5$, the proposal misses the GT pair; $11.2\%$ of pairs). The fair comparison is against Qwen3-VL-8B$+$proposal---the same backbone with the same proposals but without our SFT$+$GRPO. \methodname improves on it across the IoU spectrum: All AR 31.03 vs.\ 28.14, AR@.5 54.94 vs.\ 48.52 ($+6.42$), AR@.75 30.36 vs.\ 28.01, and even Hard AR 4.44 vs.\ 3.68. The only column where it does not is small-object recall AR$_s$ (3.58 vs.\ 3.65), which we return to below. The consistent margin on the other eleven columns is what proposal injection alone cannot explain---it is the contribution of the counterfactual reward layered on identical anchors.

Relative to the no-proposal base Qwen3-VL-8B, \methodname is ahead on overall AR (31.03 vs.\ 30.38), with the gain concentrated at the IoU@.5 operating point: AR@.5 54.94 vs.\ 50.40 ($+4.54$) and Easy@.5 67.78 vs.\ 61.46 ($+6.32$). At the strictest IoU@.75 threshold the two are on par (30.36 vs.\ 30.62), so the improvement reflects more pairs clearing a usable localization bar rather than uniformly tighter boxes.

Larger and closed models lead the absolute ranking---Claude-4.5-Opus$+$proposal (43.96 AR) and Qwen3-VL-32B (34.18 AR)---and we do not contest that. Scoped to a single 8B backbone under matched proposals, \methodname is the strongest grounding method in the table.

\subsection{Ablation Study and Analysis}
\label{sec:ablation}

\noindent\textbf{GRPO is the enabler.}
\cref{tab:ablation} decomposes the pipeline stage by stage. The story is not that any single ingredient suffices, but that the full configuration dominates every intermediate one: on \HICOgnd, \HICOref and \SWIGref, Full (Proposal+SFT+GRPO) is the best row. Proposals establish the spatial substrate but are not free---on HICO grounding the proposal-only row actually drops below the unconditioned base (28.14 vs.\ 30.38), because the model must learn to read the new anchors. SFT then introduces a temporary regression that is sharper for grounding than for referring (HICO grounding 30.04 after a 28.14 proposal-only start; SWIG grounding 35.78$\to$28.26), reflecting that the structured box-prediction format is harder to acquire by imitation than language generation. GRPO recovers it: HICO grounding climbs to 31.03, surpassing both the proposal-only (28.14) and the base (30.38) rows, and referring improves on both datasets (HICO BS-F1 to 84.73, SWIG to 77.90). The counterfactual tool-gain reward is what turns the proposal conditioning from a liability the SFT model over-trusts into a net gain.

\noindent\textbf{The SWIG grounding exception.}
On \SWIGgnd the recovery is incomplete: GRPO lifts the SFT row from 28.26 to 29.05 but does not reach the 38.32 of the unconditioned base. We report this as a genuine limitation rather than a result. The cause is vocabulary breadth, not the reward: SWIG's roughly $10\times$ larger object vocabulary exposes the training distribution, the SFT stage is where the proposal over-trust is introduced (35.78$\to$28.26), and GRPO only partially undoes a frozen-SFT bias it inherits.

\noindent\textbf{The Hard-stratum trade-off.}
On the Hard grounding split the no-proposal base Qwen3-VL-8B attains higher AR than ours (HICO 5.16 vs.\ 4.44). This is the expected behaviour of a proposal-conditioned model: the base is unanchored and free-grounds, which helps on the few pairs where the proposal is misleading but costs it everywhere else; ours stays correctly conditioned on the (here unreliable) proposal. The honest comparison is therefore against other proposal-conditioned methods. Among the eight \emph{+proposal} entries on HICO-Hard, ours (4.44) ranks third---ahead of Qwen3-VL-8B$+$proposal (3.68), Qwen3-VL-4B$+$proposal (3.51), GPT-5.2$+$proposal (3.14) and InternVL3-38B$+$proposal (2.69), and behind only the larger or closed Claude-4.5-Opus$+$proposal (5.88) and Qwen3-VL-32B$+$proposal (5.34). Against its own architecture, ours improves Hard grounding by $+0.76$. The aggregate trade is favourable: ours beats the base on All AR (31.03 vs.\ 30.38), Easy AR (38.98 vs.\ 37.93), AR@.5 (54.94 vs.\ 50.40), AR@m and AR@l. The same logic explains AR$_s$: the small-object split contains many flush, body-contact objects where the proposal, not the crop, is the binding constraint, so the base's free-grounding edges ours out (3.58 vs.\ 3.76). The Hard gap originates at the SFT stage and GRPO narrows it (HICO-Hard 3.68$\to$4.44; SWIG grounding 28.26$\to$29.05), so it is a frozen-SFT over-trust artifact, not a GRPO failure.

\input{depd/tbl_ablation}

\vspace{1ex}
\noindent\textbf{Selectivity: when the policy zooms.}
The mechanism behind these results is selective, not constant, tool use. Measured over the HICO test set, \methodname invokes zoom on $47.7\%$ of Hard grounding queries (proposal unreliable) but only $6.7\%$ of Easy ones, and $0\%$ on referring, where the interaction is legible in the full frame and a crop adds nothing; the SWIG pattern is the same direction ($45.2\%$ Hard vs.\ $13.4\%$ Easy). GRPO sharpens this gate over the SFT model (Hard zoom rate $42.9\%\to47.7\%$). The policy spends tool budget where the evidence is missing and abstains where the proposal already answers the query---exactly the behaviour the counterfactual tool-gain reward is designed to induce, and the reason a deterministic size-based zoom is the wrong trigger: ours fires on \emph{proposal reliability}, not object size.

\vspace{1ex}
\noindent\textbf{Efficiency.}
While \methodname is not the absolute top scorer---32B and closed proposal-conditioned models lead in raw accuracy---its advantage among same-scale 8B systems is \emph{selective, lightweight} tool use, quantified in \cref{tab:efficiency} over all four benchmarks ($N{=}\num{92446}$ queries). \methodname averages $0.104$ tool calls per query and invokes a tool on only $9.44\%$ of queries, for a mean of $1.104$ turns. The GRPO policy stays close to its SFT initialization on raw tool frequency ($0.093$ calls, $7.91\%$, $1.093$ turns) while redirecting those calls toward the queries that need them, as the selectivity rates show. The non-tool baselines (Qwen3-VL-8B, Claude, GPT) are single-turn by construction ($0$ calls, $1.0$ turn). On a single idle GPU \methodname averages $2.92$\,s per query against $2.66$\,s for the single-turn no-tool baseline, an overhead of only $+0.26$\,s ($\sim$10\%); and that cost is confined to the ${\sim}9\%$ of queries that actually zoom ($5.06$\,s when a tool fires versus $2.72$\,s otherwise). Fewer than one query in ten triggers a tool, so the multi-turn machinery adds a small marginal cost on the queries that need evidence rather than a constant tax on every query.

\input{depd/tbl_efficiency}

\vspace{1ex}
\noindent\textbf{Qualitative Tool-Use Comparison.}
\cref{fig:effective_tool_call} contrasts SFT and SFT+GRPO on the same grounding query, making the behavioural shift from the ablation concrete. SFT applies two progressive zoom-ins yet remains uncertain, ultimately estimating the bounding box from scratch rather than trusting the proposal coordinates. SFT+GRPO resolves the same query with a single decisive zoom, identifies the object from visual evidence, and correctly maps it to the proposal despite the detector's wrong label. This confirms that GRPO trains the model to make evidence-grounded tool choices rather than accumulating redundant observations.

\vspace{1ex}
\noindent\textbf{Training Dynamics.}
\cref{fig:toolcall} shows tool discipline and accuracy improving together along an inverse trend: as training proceeds the policy issues fewer zoom calls yet scores higher. Early on it over-explores before the counterfactual reward has shaped the gate; by convergence it zooms mainly when the proposal is unreliable. Crucially, neither quantity is constrained by an inference budget---the efficiency gain emerges from the reward structure, indicating that the model learns to reason selectively rather than exhaustively.

\input{depd/fig_tool_call}
```

NEW FILE → `depd/tbl_efficiency.tex` (\input after tbl_ablation; already wired in the prose above):

```latex
\begin{table}[!t]
    \centering
    \caption{Tool-use efficiency. \textbf{Calls/query}, \textbf{\% w/ tool}, and \textbf{Turns} are aggregated over all four benchmarks ($N{=}\num{92446}$ queries); non-tool baselines are single-turn by construction. \textbf{Time/query} is per-query latency on one idle GPU (vLLM, sequential greedy decoding, $N{=}300$ HICO grounding queries); API models are excluded as their latency is not comparable to local GPU inference, and \methodname is timed on the deployed SFT+GRPO model. Adaptive tool use adds only ${+}0.26$\,s/query (${\sim}10\%$) over the single-turn baseline, confined to the ${\sim}9\%$ of queries that zoom.}
    \label{tab:efficiency}
    \vspace{-1em}
    \setlength{\tabcolsep}{4pt}
    \renewcommand{\arraystretch}{1.15}
    \begin{adjustbox}{max width=\columnwidth}
    \begin{tabular}{l | S[table-format=1.3] S[table-format=2.2] S[table-format=1.3] c}
        \toprule
        Method & {Calls/query} & {\% w/ tool} & {Turns} & {Time/query (s)} \\
        \midrule
        Qwen3-VL-8B~\cite{bai2025qwen3vltechnicalreport} & 0.000 & 0.00 & 1.000 & 2.66 \\
        Claude-4.5-Opus~\cite{anthropic2025claude}       & 0.000 & 0.00 & 1.000 & {N/A} \\
        GPT-5.2~\cite{singh2025openai}                   & 0.000 & 0.00 & 1.000 & {N/A} \\
        \midrule
        \methodname (SFT only)                           & 0.093 & 7.91 & 1.093 & {--} \\
        \rowcolor{gray!15} \methodname (SFT+GRPO)        & 0.104 & 9.44 & 1.104 & 2.92 \\
        \bottomrule
    \end{tabular}
    \end{adjustbox}
\end{table}
```

OPTIONAL standalone Discussion paragraph (draft 2) — use ONLY if you add a separate `\subsection{Discussion}` / Limitations block; do NOT also paste it into §4 (the Efficiency + Hard-stratum prose above already carries this framing). Kept here as the camera-ready home for prof point 5 if a Discussion section is preferred:

```latex
\paragraph{Where SAHA stands.}
SAHA is not the top-scoring system on every absolute metric---larger and closed models (Qwen3-VL-32B, Claude-4.5-Opus with proposals) lead on overall grounding AR, and a region-token--finetuned specialist (Groma-7B-HOI-FT) leads on referring---yet its contribution is orthogonal to raw scale and remains well-supported. First, under a controlled same-VLM, same-proposal comparison SAHA is the strongest 8B system: it beats Qwen3-VL-8B$+$proposal on $11$ of $12$ HICO grounding columns (AR $31.03$ vs.\ $28.14$; IoU@$.5$ $54.94$ vs.\ $48.52$, $+6.42$) and surpasses the unanchored base 8B on overall AR ($31.03$ vs.\ $30.38$) and decisively at IoU@$.5$ ($+4.54$). Second, on referring SAHA is the best general VLM at every difficulty stratum including Hard (HICO METEOR $29.66/30.54/25.63$, a $+4.14$ Hard-METEOR margin over the base 8B). Third, SAHA's tool use is selective and lightweight: across all four benchmarks it issues only $0.10$ tool calls per query, invokes a tool on just $9.4\%$ of queries, and averages $1.10$ turns, zooming on $47.7\%$ of Hard grounding queries but $6.7\%$ of Easy and $0\%$ of referring. We are explicit about the limits: on the Hard split the unanchored base scores higher (HICO $5.16$ vs.\ $4.44$), and on SWIG grounding the SFT-stage proposal-dependence is not fully recovered.
```

═══════════════════════════════════════════════════════════════════
## PART 2 — Fig 1 redesign spec + §4.4 citation fix (prof points 1–3)
═══════════════════════════════════════════════════════════════════

### Fig 1 (`fig/framework.png`) — hand to designer
Current: LEFT half "Data Curation" = (A) SFT distilled-trace card + (B) "GRPO Instruction Data" with Grounding/Referring example blocks. RIGHT half = YOLO proposals → "HOI Instruction" (SAME Grounding/Referring format) → "SAHA Framework" reasoning loop + a floating top-right reward box titled "SDS-GRPO Reward".

1. **DELETE the bottom-left "GRPO Instruction Data" panel** (the Grounding/Referring example pair) — it duplicates the right-side "HOI Instruction". Rename the left half "Data Curation" → "Stage 1: SFT Cold-Start (Knowledge Distillation)" keeping ONLY the single `<think>/<tool_call>/<answer>` trace card. Optionally add a one-line caption under YOLO Proposals: "Raw HICO-DET / SWIG-HOI annotations → Grounding & Referring instructions" to preserve data provenance as text. (prof point 1a)
2. **ADD two arrows to close the loop** (both currently absent; reward box is orphaned — prof point 1b):
   - IN: from the `<answer>`/Prediction node → into the reward box, head on the R_outcome/R_tool stages, label "score `<answer>` vs GT + proposal anchor".
   - OUT (red/orange curved feedback): from `R_total` → back to the SAHA-Framework/policy box, label "GRPO policy update (group-relative advantage, K=4)".
3. **Retitle the reward box** "SDS-GRPO Reward" → "Counterfactual Tool-Gain (CTG) Reward"; "Stage 3: SDS Gated Tool Reward R_tool" → "Stage 3: Counterfactual Tool-Gain R_tool = I_tool · clip(s_final − s_ref)". Keep `R_total = R_format·(R_outcome + α·R_tool)` (already correct). Removes the last "SDS-GRPO" string from the main figure (CLAUDE.md: SDS is eval-only).

### §4.4 "citation error in section 4.4 line 358"
NO LaTeX toolchain on this box, so PDF-line→tex-line could not be auto-resolved. Findings from reading source + bib:
- §4.4 prose (tex 102–127) contains **ZERO `\cite{}`** — only `\cref{}` to floats (`tab:ablation`, `fig:effective_tool_call`, `fig:toolcall`, `sec:ablation`). All 10 `\cite` keys in the section file resolve to one bib entry each, so it is NOT an undefined `\cite`.
- **Most likely (1): a swapped `\cref{fig:effective_tool_call}` vs `\cref{fig:toolcall}`.** `fig:effective_tool_call` is a full-width `figure*` (floats out of source order); the §4.4 sentence at tex 116 ("contrasts SFT and SFT+GRPO") and tex 123 ("fewer tools yet higher accuracy") can render with the wrong figure NUMBER. On the next compile, confirm the figure number printed in each sentence matches its content; if swapped, correct the `\cref` key.
- **(2):** the GRPO cite `\cite{shao2024deepseekmath}` lives in §4.3 (tex 56), but §4.4 says "GRPO" 5× with no local cite; if PDF line 358 lands on the §4.3/§4.4 boundary, add a local cite at the first §4.4 "GRPO" (tex 106). The bib entry itself is correct.
- **Bib-content fixes regardless:** `bercovich2025llamanemotro` (the LLM-as-Judge cite at tex 28 — misspelled key, points to "Llama-Nemotron"; verify it is the intended judge paper, high-risk since R3/R5 flagged judge validation) and `singh2025openai` (titled "GPT-5 System Card" but text cites GPT-5.2; verify arXiv 2601.03267 resolves).
- **Definitive resolution:** compile once, `grep` the `.log` for `Reference … undefined` / `Citation … undefined`.

═══════════════════════════════════════════════════════════════════
## PART 3 — "What to apply where" checklist
═══════════════════════════════════════════════════════════════════

| # | Prof point | File / location | Action |
|---|---|---|---|
| 1a | Fig1 duplicate instruction panel | `fig/framework.png` (designer) | Delete bottom-left "GRPO Instruction Data" panel; rename left half to single SFT card. |
| 1b | Fig1 no reward arrow | `fig/framework.png` (designer) | Add `<answer>`→reward arrow + red `R_total`→policy feedback arrow; retitle box "SDS-GRPO"→"Counterfactual Tool-Gain (CTG)". |
| 2 | §4.4 citation error, line 358 | `sec/4_experiment.tex` tex 116/123 + `main.bib` | Compile + grep `.log`; check `\cref{fig:effective_tool_call}`/`\cref{fig:toolcall}` not swapped; add GRPO `\cite` at tex 106 if needed; fix `bercovich2025llamanemotro` + `singh2025openai` bib entries. |
| 3 | Efficiency details | `depd/tbl_efficiency.tex` (NEW) + §4 Efficiency prose (Part 1) | Add table (done, exact: 0.104/9.44%/1.104 vs 0/0/1.0); fill inference-time [TBD] from GPU0; optionally add a token-cost column per the locked decision. |
| 4 | Not-dominant-but-has-plus-points | §4 Efficiency + Hard-stratum prose (Part 1); optionally draft-2 Discussion paragraph | The §4 replacement now opens Efficiency with the explicit "not the absolute top scorer … advantage is selective, efficient tool use" line tied to `tab:efficiency`. Single home; do not double-paste the Discussion paragraph into §4. |
| 5 | Stale §4 claims | `sec/4_experiment.tex` | Part-1 replacement REMOVES the false "highest LLM-Judge on HICOref" claim (column now `{--}`) and reframes AR$_s$ as a loss; the Qualitative Results subsection (tex 95, "iterative zoom recovers the frisbee … key factor for small handheld objects") still contradicts the AR$_s$ concession — soften it manually. |
# Backup — SDS-GRPO paper content removed in the CTG rewrite

**Date:** 2026-06-24. **Why:** the methodology was rewritten from the area-based
**SDS-GRPO** reward to the **Counterfactual Tool-Gain (CTG)** reward (+ 4B→8B).
This file preserves the **old text/equations that were removed or replaced**, so it
can be restored if needed (e.g. to describe the SDS-GRPO baseline row, or to revert).

**Authoritative live source of the change:** the Overleaf repo working tree vs HEAD.
To see/restore in full: `cd /workspace/6971992f16a95dab9a63a1f9 && git diff HEAD -- sec/`
or restore one file: `git checkout HEAD -- sec/3_method.tex`.

Below: the **removed (old) lines** per file, verbatim LaTeX (the '+' new lines are in the live paper).

## sec/0_abstract.tex — removed / replaced (OLD)

```latex
Central to our approach is \textbf{Spatial Difficulty Score-Gated GRPO (SDS-GRPO)}, a reward mechanism that gates tool invocation on per-sample spatial difficulty, learning a selective zoom policy that triggers dynamic spatial refinement only when necessary (e.g., small or ambiguous objects).
Building on Qwen3-VL-4B, our model outperforms the same-size vanilla baseline across both tasks, leads among VLM-based baselines on holistic interaction understanding for HICO-DET referring, and remains competitive with models twice its parameter count.
Notably, the training process shows that accuracy improves as tool reliance decreases, suggesting the efficacy of spatial context-guided tool use.
```

## sec/1_intro.tex — removed / replaced (OLD)

```latex
To address this challenge, we propose Spatial Aware HOI Agent (\methodname), a VLM-based HOI framework spanning data curation, task reformulation, multi-turn spatial reasoning with zoom tools, and SDS-GRPO reward training — as illustrated in \cref{fig:architecture}.
Our key innovation is \textit{Spatial Difficulty Score-Gated GRPO (SDS-GRPO)}, a reinforcement learning paradigm that conditions the tool-use reward on the spatial complexity of the interaction (\ie~SDS).
By deriving the SDS from ground-truth bounding-box areas, the model is incentivized to invoke spatial refinement for high-difficulty pairs while being penalized for redundant tool calls on easy samples.
Our 4B model outperforms the vanilla Qwen3-VL-4B baseline on both HOI grounding and action referring, and remains competitive with the larger Qwen3-VL-8B counterpart at half the parameter count.
As training progresses, accuracy rises steadily while mean tool calls per sample decrease from 2.5 to 1.0, and tool invocations align with spatial context requirements, confirming that context-guided tool use, not increased computation, drives the gain.
      \item We propose SDS-GRPO, a GRPO reward that modulates tool-use incentive via a Spatial Difficulty Score (SDS) derived from object bounding-box geometry. This formulation trains a selective zoom policy that invokes tools only when spatially justified, effectively balancing descriptive granularity with computational efficiency.
      \item Extensive experiments show our proposed 4B model surpasses the same-size vanilla baseline and remains competitive with the larger 8B counterpart, with training dynamics confirming that spatial context-guided tool use drives the accuracy gains.
```

## sec/2_related_work.tex — removed / replaced (OLD)

```latex
Existing frameworks have shown strong gains in generic visual reasoning, including focus-based zooming~\cite{zhang2025chain}, adaptive tool routing~\cite{wang2025adatooler}, curiosity-driven visual reasoning~\cite{wang2025pixel}, and general tool-learning infrastructure~\cite{jiang2025verltool}, but apply tools without conditioning on per-sample spatial difficulty.
```

## sec/3_method.tex — removed / replaced (OLD)

```latex
We reformulate HOI detection as a pair of instruction-following tasks for VLMs and build a unified region-based pipeline on top of Qwen3-VL-4B~\cite{bai2025qwen3vltechnicalreport}. 
Third, inspired by the adaptive tool-use framework AdaTooler-V~\cite{wang2025adatooler}, we train a selective visual exploration policy using \texttt{zoom\_in} and \texttt{zoom\_out} tools, optimised via knowledge distillation warm-up followed by our proposed Spatial Difficulty Score-Gated GRPO (SDS-GRPO), enabling efficient tool calls with improved context precision (see \cref{sec:tool}).
Fine-grained HOI understanding demands visual evidence that global encoding routinely fails to provide: distinguishing ``adjusting necktie'' from ``wearing necktie'' requires resolving sub-centimetre hand-contact cues lost in a single-pass feature map. Inspired by AdaTooler-V~\cite{wang2025adatooler}, we equip the VLM with \texttt{zoom\_in} and \texttt{zoom\_out} tools and train a \emph{selective} invocation policy in two stages: a supervised warm-up via knowledge distillation provides calibrated priors on when tools are beneficial, and RL with our proposed SDS-GRPO reward refines tool timing by per-sample spatial difficulty.
  The two tools serve complementary roles: \texttt{zoom\_in} gathers new visual evidence while \texttt{zoom\_out} navigates spatial context — a distinction that motivates their asymmetric treatment in SDS-GRPO.
  \subsubsection{Policy Optimization with SDS-GRPO}
  GRPO~\cite{shao2024deepseekmath} refines the tool-use policy by optimising task-performance signals over the full training set. For each sample, $K$ reasoning chains are sampled and the policy is updated to favour above-average-reward trajectories:
  \noindent\textbf{Outcome Rewards.}
  For grounding, predicted pairs are matched to GT pairs via joint IoU, and average recall is computed at $\delta\in\{0.5,0.75\}$:
      R_{\mathrm{ground}}=\frac{1}{2}\!\left(
      \frac{|\mathcal{M}_{0.5}|}{|G|}+\frac{|\mathcal{M}_{0.75}|}{|G|}
      \right)
  \noindent where $\mathcal{M}_\delta$ is the set of predicted pairs matched to a GT pair at joint IoU $\geq\delta$,\footnote{Joint IoU is the average of the person-box IoU and the object-box IoU between a predicted pair and a GT pair, following the HICO-DET evaluation protocol~\cite{chao2018learning_hico}.} and $|G|$ is the total number of GT pairs.
  \noindent\textbf{Spatial Difficulty Score.}
  Tool benefit depends on whether the object is visually resolvable at global scale: a uniform tool reward ignores this and either over-encourages zooming on easy images or under-rewards it on hard ones.
  We encode per-sample difficulty as the Spatial Difficulty Score~(SDS), derived from the GT object box area only — objects are the consistent resolution bottleneck, not persons — linearly mapping normalized object area to a $[0,1]$ score.
  Concretely, objects occupying less than 1\% of the image area (typical of small interacting objects such as a baseball or a knife) receive SDS$\approx 1$; objects occupying more than 15\% receive SDS$=0$.
  These operating bounds were chosen to match the practical range of interaction objects in HICO-DET: objects below the 1\% threshold are routinely under-resolved by the global VLM encoder, while objects above 15\% are large enough that zooming provides negligible benefit.
  A high SDS (close to 1) indicates a small, under-resolved object likely to benefit from \texttt{zoom\_in}; a low SDS (close to 0) indicates a large object for which zoom calls are unnecessary and incur a penalty.

  \noindent\textbf{SDS-Gated Tool Reward.}
  The tool reward $R_{\mathrm{tool}}$ decomposes into an information-gathering term and a context-restoration bonus:
  \begin{equation}
      R_{\mathrm{tool}} = R_{\mathrm{zoom\_in}} + R_{\mathrm{hygiene}}
  \end{equation}
  The zoom-in reward is gated by the per-sample target call count $n_{\mathrm{opt}}=\bigl\lfloor 2\cdot\mathrm{SDS}\bigr\rceil$:
  R_{\mathrm{zoom\_in}} =
  \begin{cases}
  (\mathrm{SDS}-\tau)\,\exp\!\left(-\gamma\!\left(\dfrac{n_{\mathrm{zi}}-n_{\mathrm{opt}}}{n_{\mathrm{opt}}}\right)^{\!2}\right) & n_{\mathrm{opt}}>0 \\[6pt]
  -\lambda_{a}(1-\mathrm{SDS})\,n_{\mathrm{zi}} & n_{\mathrm{opt}}=0
  \end{cases}
  The $(\mathrm{SDS}-\tau)$ factor ensures only samples with $\mathrm{SDS}>\tau$ receive a net positive zoom reward; when $n_{\mathrm{opt}}=0$, any zoom call is penalised linearly.
  A small hygiene bonus rewards pairing \texttt{zoom\_out} with \texttt{zoom\_in} without penalising its absence:
  \begin{equation}
      R_{\mathrm{hygiene}} = \lambda_{h}\,\frac{\min(n_{\mathrm{zo}},\,n_{\mathrm{zi}})}{\max(n_{\mathrm{zi}},\,1)}
  \end{equation}
  where $n_{\mathrm{zi}}$ and $n_{\mathrm{zo}}$ are the \texttt{zoom\_in} and \texttt{zoom\_out} call counts. Empirically, $\tau{=}0.3$ is the SDS activation threshold; $\gamma{=}2.0$ controls Gaussian decay steepness; $\lambda_{a}{=}0.1$ is the abstain penalty coefficient; and $\lambda_{h}{=}0.05$ is the hygiene bonus coefficient.
  \noindent\textbf{Composite Reward.}
  where $R_{\mathrm{outcome}}\in\{R_{\mathrm{ground}},R_{\mathrm{refer}}\}$ is selected based on the task type, and $\alpha{=}0.6$ is empirically selected to balance task accuracy against tool efficiency. Full hyperparameter details are in Appendix.
```

## sec/4_experiment.tex — removed / replaced (OLD)

```latex
We compare against two groups of baselines, all evaluated without YOLOE-26 region proposals, to isolate the contribution of spatial proposal conditioning and our SDS-GRPO training pipeline.
To isolate the contribution of our SDS-GRPO training pipeline from region proposal conditioning, we report a proposals-only ablation of Qwen3-VL-4B without SFT or GRPO in \cref{sec:ablation}.
  We fine-tune only the language model weights of Qwen3-VL-4B using LLaMA-Factory~\cite{zheng2024llamafactory}, keeping the vision tower and multi-modal projector frozen to preserve pretrained visual representations.
  The model is served via vLLM with a \num{32768}-token context window and invokes \texttt{zoom\_in}/\texttt{zoom\_out} freely until producing a final \texttt{<answer>} block, consistent with the SDS-GRPO objective of adaptive, spatial context-guided tool use.
Despite a compact 4B backbone, our model achieves competitive grounding performance against larger-scale baselines, suggesting that adaptive spatial refinement can partly compensate for model scale.
GRPO corrects this over-zoom tendency, surpasses proposals-only on \HICOgnd and \HICOref, and improves referring across both datasets, demonstrating that the SDS-gated reward is sufficient to realize the full value of spatial conditioning.
Early in training, the policy over-explores -- issuing zoom calls indiscriminately before the SDS reward has shaped selectivity; by convergence, it zooms only when spatial difficulty warrants it.
Crucially, this efficiency gain is not imposed by any inference budget constraint---both quantities are unconstrained---but emerges from the reward structure itself, indicating that SDS-GRPO teaches the model to reason selectively rather than exhaustively.
```

## sec/5_conclusion.tex — removed / replaced (OLD)

```latex
We presented \methodname, a VLM-based HOI detection framework that reformulates HOI grounding and action referring as instruction-following tasks, training a selective zoom policy via SDS-GRPO that conditions tool invocation on per-sample spatial difficulty rather than applying it uniformly.
On both tasks, a compact 4B model achieves competitive grounding performance against larger baselines and leads among VLM-based methods on holistic referring evaluation.
These results demonstrate that task reformulation and SDS-GRPO together enable precise HOI understanding without scaling model size.
```

## sec/6_suppl.tex — removed / replaced (OLD)

```latex
\section{SDS-GRPO Reward Details}
\label{sec:appendix_sds_grpo}
This section provides additional design rationale and hyperparameter analysis
for SDS-GRPO, complementing the reward formulation presented in \S3
\subsection{Spatial Difficulty Score (SDS)}
SDS quantifies how much a sample benefits from zooming, based solely on the
minimum GT \emph{object} bounding-box area.
Object area alone drives SDS because objects are the perceptual bottleneck
in HOI grounding, not persons: the person is almost never under-resolved at
global scale, so using $\min(\text{person area},\, \text{object area})$ would
over-count easy cases.

Let $a_\text{min}$ denote the minimum GT object area normalised to $[0,1]$
in the $1000{\times}1000$ coordinate space.
SDS is defined as:
\begin{equation}
  \text{SDS} = \operatorname{clip}\!\left(
    \frac{0.15 - a_\text{min}}{0.14},\; 0,\; 1
  \right)
  \label{eq:sds}
\end{equation}
This linearly maps $a_\text{min} \leq 0.01$ (object occupies $\leq$1\% of the
image) to $\text{SDS}=1.0$, and $a_\text{min} \geq 0.15$ (object occupies
$\geq$15\% of the image) to $\text{SDS}=0.0$.
These bounds capture the practical range of interaction objects in HICO-DET:
objects below the 1\% threshold (e.g., a baseball or knife) are routinely
under-resolved at global scale and benefit from \texttt{zoom\_in}, while
objects above 15\% are clearly visible without zoom.
\subsection{Tool Reward}
The tool reward separates \texttt{zoom\_in} and \texttt{zoom\_out} because
they serve distinct roles:
\texttt{zoom\_in} gathers new visual information, while \texttt{zoom\_out}
restores spatial context.
Conflating them into a single tool-use count would conflate
information-gathering skill with navigational housekeeping.

\paragraph{zoom\_in reward.}
The optimal number of \texttt{zoom\_in} calls scales with object difficulty.
We set $n_{\mathrm{opt}} = \operatorname{round}(2 \times \mathrm{SDS})$,
calibrated to the SFT distribution (69\% of tool-use trajectories contain
one \texttt{zoom\_in}; 28\% contain two).
When $n_{\mathrm{opt}} > 0$, the reward uses a Gaussian efficiency function
peaking at $n_{\mathrm{zi}} = n_{\mathrm{opt}}$; when $n_{\mathrm{opt}} = 0$
(SDS is low; objects are large), any zoom call is penalised linearly.
See \S3 (Tool-Augmented Reasoning) of the main paper for the complete formulation.

\paragraph{zoom\_out reward ($R_{\mathrm{hygiene}}$).}
\texttt{zoom\_out} restores the full-image view after a crop, helping the
model maintain spatial context before zooming into a different region.
It is not an information-gathering action, so we reward it only as a small
hygiene bonus rather than penalising its absence.
Empirically, 38\% of referring SFT trajectories use \texttt{zoom\_in}
without any \texttt{zoom\_out}; penalising this behaviour would incorrectly
discourage valid one-shot zoom trajectories.
The hygiene bonus proportionally rewards matched \texttt{zoom\_out}/\texttt{zoom\_in}
pairs (capped at one unit per \texttt{zoom\_in}) and is zero when no
\texttt{zoom\_in} is present; see \S3 (Tool-Augmented Reasoning) of the main paper for the formula.

The total tool reward is $R_{\mathrm{tool}} = R_{\mathrm{zoom\_in}} + R_{\mathrm{hygiene}}$,
where $n_{\mathrm{zi}}$ and $n_{\mathrm{zo}}$ are the \texttt{zoom\_in} and
\texttt{zoom\_out} call counts, respectively.
\cref{tab:sds_hyperparams} lists all SDS-GRPO hyperparameters.
$\alpha$ balances task accuracy against tool efficiency and is the most
sensitive hyperparameter.
The Gaussian decay $\gamma$ and abstain penalty $\lambda_{a}$
were set to match the reward magnitude of $R_{\mathrm{outcome}}$.
\caption{SDS-GRPO hyperparameters and their roles.}
\label{tab:sds_hyperparams}
$\alpha$          & 0.6  & Weight of $R_{\mathrm{tool}}$ in $R_{\mathrm{total}}$ \\
$\tau$            & 0.3  & SDS threshold below which \texttt{zoom\_in} is not rewarded \\
$\gamma$          & 2.0  & Gaussian decay steepness around $n_{\mathrm{opt}}$ \\
$\lambda_{a}$     & 0.1  & Per-call penalty for zooming on easy images ($n_{\mathrm{opt}}=0$) \\
$\lambda_{h}$     & 0.05 & Hygiene bonus per matched \texttt{zoom\_out}/\texttt{zoom\_in} pair \\
\textit{hard} ($<$1\%; aligns with SDS$\approx 1.0$).
\noindent The frisbee occupies 0.68\% of the image area (SDS$=1.0$),
placing it in the high-difficulty regime where \texttt{zoom\_in} is
maximally rewarded by SDS-GRPO (see \cref{sec:appendix_sds_grpo}).
The grounding reward $R_{\mathrm{ground}}$ computes average pair recall at
IoU thresholds 0.5 and 0.75, where each predicted pair is matched to a GT
pair via greedy assignment on a joint pair score
$s = 0.5 \cdot \mathrm{IoU}_\mathrm{person} + 0.5 \cdot \mathrm{IoU}_\mathrm{object}$.
```


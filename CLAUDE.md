# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚡ ACTIVE PLAN: SAHA v2 — Proposal-Aware Evidence Difficulty (ACCV 2026, submission 5 Jul 2026)

This repo is now the training codebase for the **SAHA ACCV 2026 revision**. Before ANY code change, read in order:

1. `docs/saha-v2/reward-v3-counterfactual-spec.md` — **THE authoritative reward/credit design (2026-06-15, approved). Wins on reward, GRPO credit, and the crowded/occluded mechanism.** Reward v3 = `R_format·(R_outcome + α·R_tool)`, counterfactual tool-gain, replaces the Rev-4 6-term reward.
2. `docs/saha-v2/latest-decision.md` — Rev 4 spec. Authoritative for everything **non-reward** (proposals, eval protocol, falsification gates, SFT reuse, novelty map). Where it conflicts with the v3 spec on reward/credit, the v3 spec wins.
3. `docs/saha-v2/adatooler-v-reading-note.md` — before touching reward code (novelty differentiation).
4. `plan/saha-v3-simple/` — design notes + implementation plan for the v3 reward.

**Hard constraints (any session, any agent):**
- Keep external object-region proposals (YOLOE) and proposal injection.
- Active tools = `zoom_in` and `zoom_out` ONLY. Never implement/register `inspect_pose` or any new tool.
- Never switch to detector-free. Never implement AXPO (cite-only).
- Never claim a new GRPO algorithm; never write "unbiased" without a derivation.
- Never adopt AT-GRPO's `ΔS · exp(−γ(·)²)` reward as the main method — it exists only as the labeled baseline (`reward_manager=AdaTooler-V`, latest-decision §11.7).
- **Reward = `R_format·(R_outcome + α·R_tool)`, ONE knob α** (v3 spec §2). `R_tool = I_tool·clip(s_final − s_ref)` is a counterfactual tool-gain grounded ONLY in target-object GT (grounding `s_ref` = selected proposal anchor scored vs target GT; referring `s_ref` = detached per-`uid` no-tool mean). **Never** reintroduce the Rev-4 6-term reward (`R_proposal_correction/decision/bad_copy/redundant`), per-factor SDS-v2 reward weights, or the `proposal_action` (trust/correct/reject) verbalization — the frozen SFT cannot emit it (R6 gate).
- SDS-v2 factors (`D_size, D_occlusion, D_contact, D_proposal, D_scene`) are **EVAL-ONLY** stratification axes now — NOT reward terms. Do not wire them into the reward.
- No GT leakage: GT (target box, proposal-quality strata) is training/eval-only; inference prompts use model-observable proposal signals only.
- REUSE the v1 SFT checkpoint. NEVER start SFT trace generation without a recorded human decision (latest-decision §10.0).
- **GRPO tool-collapse fix = the counterfactual reward itself + standard GRPO advantage (default).** The two-stratum tool/no-tool SAN (cite Stratified GRPO 2510.06214) is an **ABLATION only** — never run it together with the counterfactual reward as the default (it double-centers the tool-vs-no-tool contrast).
- Work order (v3): one-time **parquet re-run** to thread the proposal anchor into `extra_info` → reward helpers + two-pass `__call__` + per-rollout logging → smoke tests → **4B SFT + balanced subset** reward sweep (`α`, clip, referring-`s_ref` variant) → pick winner → scale to 8B. Plan in `plan/saha-v3-simple/`.
- Stop and report missing paths/fields; never silently invent a different method.

**Status of the existing pipelines under SAHA v2:**
- `sds_grpo.py` (`reward_manager=SDS-GRPO`, area-only SDS, `R_format×(R_outcome+0.6·R_tool)`) is now the **"old area-SDS" baseline row** (spec §11.4). Keep it registered, runnable, and untouched. Note: the `tests/test_sds_grpo.py` suite cited by `reward_design.md` was never committed — Wave-1 task B0 (execution-plan §2-B) recreates it from the documented verified values before other code lands.
- `AdaTooler-V.py` (`reward_manager=AdaTooler-V`) is the **optional AT-GRPO-style baseline** (spec §11.7). Never the default.
- The SAHA method (**v3 reward, 2026-06-15**) = a NEW/updated reward manager implementing the §2 counterfactual tool-gain (`docs/saha-v2/reward-v3-counterfactual-spec.md`). The GRPO tool-collapse fix is the **counterfactual reward** (a useful tool earns `R_outcome + α·gain` → positive group advantage exactly where the proposal anchor is low), NOT a SAN — the SAN is an ablation only. SDS-v2 factor library is retained eval-only. Train on 4B SFT + balanced subset first; scale the winning reward config to 8B.

**Drift checks (run before claiming done):** `rg -n "inspect_pose|detector-free" verltool/ --type py` → nothing active; `rg -n "Delta S|Tool Benefit|AT-GRPO|AdaTooler" verltool/ --type py` → `AdaTooler-V.py` + docs only; `rg -rn "prepare_sft|trace_gen|generate_traces" --type py verltool/` → empty.

**Sibling repos:** `/workspace/hoi/saha-hoi` (SAHA v1 code), `/workspace/hoi/hoi-benchmarks` (eval harness — gate run + DetZoom + eval modes live there), `/workspace/hoi/checkpoints` (shared model checkpoints). Spec source-of-truth mirror: `paper-reading/saha-reference-notes/`.

---

## Project Overview

AdaTooler-V is a multimodal LLM (MLLM) that performs **adaptive tool-use** for image and video reasoning tasks. It uses **AT-GRPO** (Adaptive Tool-GRPO), a reinforcement learning algorithm that adjusts reward scales based on a Tool Benefit Score, encouraging the model to invoke tools only when they provide genuine improvements. Built on top of [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), which itself extends the [verl](https://github.com/volcengine/verl) RL framework. The repo also hosts the HOI (human-object interaction) RL pipeline that SAHA v2 builds on.

## Setup

```bash
cd verltool
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```

## Key Commands

### Data Preprocessing
```bash
cd verltool
python examples/data_preprocess/pixel_reasoner/prepare_train.py \
  --dataset_path=AdaTooler-V/AdaTooler-V-300k \
  --local_dir=data/AdaTooler-V \
  --version max_8192 \
  --include_videos=True \
  --filter_len=8192
```

### Stage 1: SFT Cold-Start (via LLaMA-Factory)
```bash
llamafactory-cli train sft_configs/qwen2_5vl_full_sft.yaml
```

### Stage 2: RL Training — AdaTooler-V (original)
```bash
cd verltool
bash examples/train/AdaTooler-V/train_qwen25vl.sh
```
Requires 8x H100/A100 GPUs (80GB). Uses GRPO with `n=8`, batch size 32, tensor parallelism size 2.

### Stage 2: RL Training — HOI / SDS-GRPO (Qwen3-VL)
```bash
cd verltool
bash examples/train/hoi/train_qwen3vl.sh /path/to/sft_model
```
Script as committed assumes multi-GPU (3 GPUs/node). **SAHA v2 compute reality (2026-06-12): ONE NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB).** A proven single-GPU recipe exists at `/workspace/hoi/saha-hoi/verltool/examples/train/hoi/train_saha_hoi_grpo_single_gpu.sh` (full fine-tune + CPU offload `do_offload=True`, n=2, batch 2, micro-batch 1, `gpu_memory_utilization=0.45` — NOT LoRA); porting it here (default ckpt → `hoi_v2_sft`, reward → `SAHA-V2`) + a 1-step memory smoke test are post-gate work items (execution-plan R7 / OD-2b).
SAHA v2 decisions (2026-06-12, recorded in `docs/saha-v2/execution-plan.md`):
- SFT checkpoint reused per spec §10.0: `/workspace/hoi/checkpoints/qwen3VL-8B/hoi_v2_sft` (Qwen3-VL-8B, full SFT, eval loss 0.442). Training restarts from this SFT.
- `qwen3VL-8B-SFT-GRPO/global_step_240` = previous-generation trained model, NOT in this lineage; gate rows R3–R6 run on `hoi_v2_sft` ("SFT-only current behavior"). Older 2B/4B checkpoints are previous iterations.
- Gate-run grounding scorer = current proposals-eval convention (both-pass + avg-IoU rank), min-IoU secondary where cheap.
- zoom_in coordinate question (risk R6) is settled by the A-7 smoke test, then a human decides — never fix silently.

### Evaluation
```bash
cd verltool
bash examples/train/AdaTooler-V/eval.sh
```

### Eval Service (OpenAI-compatible API)
```bash
cd verltool/eval_service
bash scripts/start_api_service.sh
```

## Architecture

```
AdaTooler-V/
├── checkpoints/
│   └── qwen3VL-2B/               # SFT cold-start checkpoint (Qwen3-VL-2B, 3 epochs, loss≈0.51)
├── sftconfigs/                    # LLaMA-Factory SFT YAML configs
│   └── qwen2.5-vl.yaml           # Qwen2.5-VL-7B full SFT config (vision tower frozen)
└── verltool/                      # Core project (verl-tool fork)
    ├── verl/                      # Git submodule: verl RL framework (volcengine/verl)
    ├── verl_tool/                 # Main package (installed via pip install -e)
    │   ├── agent_loop/            # Multi-turn agent-tool interaction loop
    │   ├── servers/               # Tool server + individual tool implementations
    │   ├── trainer/               # PPO/GRPO training entry points + Hydra configs
    │   ├── workers/               # Reward managers + rollout workers
    │   └── utils/                 # Dataset utilities (audio, etc.)
    ├── examples/
    │   ├── train/AdaTooler-V/     # Training and eval shell scripts (original)
    │   ├── train/hoi/             # HOI SDS-GRPO training script (train_qwen3vl.sh)
    │   └── data_preprocess/       # Data preparation scripts
    ├── data/hoi/                  # HOI training data (train.parquet, val.parquet)
    ├── eval_service/              # FastAPI inference server with tool calling
    ├── benchmarks/                # Git submodules: bigcodebench, evalplus, LiveCodeBench, etc.
    ├── patches/                   # Model-specific patches (e.g., qwen_2_5_omni.patch)
    └── scripts/                   # Misc utilities (frame extraction, visualization)
```

### Core Components

**Agent Loop** (`verl_tool/agent_loop/`): Manages multi-turn interactions between the LLM and tools during training. `verltool_agent_loop.py` intercepts LLM outputs, routes tool calls to the tool server, and feeds observations back. Handles vision/video processing via `vision_process.py`.

**Tool Server** (`verl_tool/servers/`): FastAPI-based server (`serve.py` spawns worker subprocesses with consistent hashing on trajectory IDs). Available tools include `pixel_reasoner` (vision crop/frame tools), `python_code`, `google_search`, `text_browser`, `bash_terminal`, `mcp_interface`, and others. Each tool is a self-contained Python file in `servers/tools/`.

**Trainer** (`verl_tool/trainer/`): Entry point is `main_ppo.py` which initializes a Ray cluster and configures training via Hydra. Agent-specific config lives in `config/verltool/agent.yaml`. Derives from verl's PPO trainer with agent loop integration.

**Reward Managers** (`verl_tool/workers/reward_manager/`): Primary managers:
- `AdaTooler-V.py` — AT-GRPO reward: calculates Tool Benefit Scores and adaptively scales rewards (original pipeline; under SAHA v2 this is the optional labeled baseline, spec §11.7).
- `sds_grpo.py` — SDS-GRPO reward for HOI tasks: `R_total = R_format * (R_outcome + alpha * R_tool)`. Uses Spatial Difficulty Score (SDS) gated on object bounding-box area to decide when zoom_in/zoom_out tool use should be rewarded or penalized. Registered as `"SDS-GRPO"`. Under SAHA v2 this is the frozen "old area-SDS" baseline (spec §11.4) — do not modify it.
- `sds_v2.py` + `saha_v2.py` (PLANNED, Phases B/C — see `docs/saha-v2/execution-plan.md`) — SDS-v2 factor library and the proposal-first SAHA v2 reward (`R_task + λ_corr·R_proposal_correction + λ_dec·R_proposal_decision + λ_tool·R_useful_tool_evidence − λ_red·R_redundant_tool_use − λ_copy·R_bad_proposal_copy`, normalize-then-sum, all terms logged per rollout).

Other reward managers exist for different tasks (torl, acecoder, sqlcoder, etc.).

**Eval Service** (`eval_service/`): OpenAI-compatible `/chat/completions` endpoint. Loads models via vLLM, communicates with the tool server for multi-turn tool calling during inference. Configured via `config.py` (ModelConfig, ToolConfig, ServerConfig).

### Training Flow

1. The training script starts the **tool server** (pixel_reasoner with 4 workers) on a random port
2. Launches `verl_tool.trainer.main_ppo` with Hydra config overrides
3. Ray initializes actor/critic/rollout workers across GPUs
4. During rollout, the agent loop generates responses, detects `</tool_call>` tokens, sends tool requests to the server, and appends observations
5. The reward manager computes rewards (AT-GRPO or SDS-GRPO depending on pipeline)
6. GRPO updates the policy

### Key Training Parameters

- `enable_agent=True` — enables agent mode with tool calling
- `enable_mtrl=True` — multi-turn RL training
- `max_turns=3` (AdaTooler-V) / `5` (HOI) — max tool interaction turns
- `mask_observations=True` — masks tool observations for KL loss/gradient
- `action_stop_tokens='</tool_call>'` — triggers tool server call
- `gpu_memory_utilization=0.45–0.5` — keep low to avoid vLLM OOM (raise requires `do_offload=True`)

### SDS-GRPO Reward Signal (HOI)

Validated reward properties (verified against dataset before training):
- High-SDS samples: GT tool pattern beats no-tool in 100% of cases
- Low-SDS samples: no-tool beats GT tool in ~98% of cases
- Reward std = 0.309 (well above 0.05 floor — not collapsed)
- Range: [0.0, 0.75]

| Pattern | R_tool |
|---|---|
| High-SDS grounding: zoom_in→out | 0.438 |
| High-SDS grounding: no-tool | 0.084 |
| Low-SDS: no-tool | ≈0.000 (neutral) |
| Low-SDS: unnecessary zoom | -0.097 (penalized) |
| Medium-SDS referring: zoom_in only | 0.202 |
| Medium-SDS referring: zoom_in→out | 0.252 (+0.05 hygiene bonus) |

## Git Submodules

The `verltool/verl/` directory is a submodule pointing to `volcengine/verl`. After cloning, always run `git submodule update --init --recursive`. Benchmarks under `verltool/benchmarks/` are also submodules.

## Troubleshooting

- **Shared memory errors**: Lower `data.dataloader_num_workers`
- **CUDA OOM during vLLM rollout**: Set `actor_rollout_ref.rollout.enforce_eager=True`
- **CUDA OOM during training**: Set `use_dynamic_bs=False`
- **vLLM OOM and stuck**: Lower `gpu_memory_utilization` to 0.4-0.5, or enable `do_offload=True` if above 0.6

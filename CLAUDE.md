# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚡ ACTIVE PLAN: SAHA — Detector as a Tool (post-ACCV pivot, decided 2026-10-21)

ACCV 2026 **rejected** the proposal-injected SAHA (meta-review: the formulation removes HOI *discovery* because YOLOE proposals were fed in offline, and gains don't generalise to SWIG). **New direction (prof + user, 2026-10-21):** the YOLOE detector becomes a **tool** the policy calls when it decides to (v1: full image only; detect-on-crop deferred). **Before ANY code change, read `plan/saha-detect-tool/PLAN.md`.** It holds the decisions (D1–D4), the verified facts, the component list and the work order.

**Hard constraints (any session, any agent):**
- **Tools = `detect`, `zoom_in`, `zoom_out`.** `detect()` (no arguments, full original image in v1) runs **live** YOLOE-26l prompt-free (`yoloe-26l-seg-pf.pt`, conf 0.3, max 50) inside the tool server at train AND test time, and returns boxes in the ORIGINAL image's 0–1000 grid. Never inject offline proposals into detect-tool prompts. No other new tools (no `inspect_pose`, no text-prompted detector) without a recorded decision.
- Scope = the current tasks (HOI grounding + action referring, with the person/object boxes still given). The mAP / image-only triplet evaluation is deferred (D1), not dropped.
- **Reward = `R_format·(R_outcome + α·R_tool)`, ONE knob α.** `R_tool = I_tool·clip(s_final − s_ref)`. `I_tool` counts detect, zoom_in and zoom_out. In detect-tool runs `s_ref` = the detached mean `R_outcome` of same-`uid` **no-tool** siblings for BOTH tasks (Option A, `SAHA_CF_GROUNDING_SREF=group_no_tool`). No sibling → `R_tool=0`; log `sref_missing_frac`.
  - Never reintroduce the Rev-4 6-term reward, per-factor SDS-v2 weights, the `proposal_action` verbalization, or a per-call cost term (second knob).
  - Paper name: proposal-relative tool-gain (PTG); its wording must be revisited now that proposals are a tool output. `SAHA-CF`, `saha_cf`, `SAHA_CF_*` and `saha_cf/*` stay as INTERNAL tags; do NOT rename them (checkpoint resume + WandB history).
- **SFT = reuse the existing traces** (`/workspace/data/hoi_sft_final`, 3,547) through a deterministic rewrite: the prompt loses its proposals; grounding traces and proposal-referencing referring traces get a `detect` call + observation carrying the original YOLOE JSON. Add paired no-detect grounding traces by rejection sampling, then retrain the LoRA. The sampler is an SFT model, not the base model, which could not produce usable answers (approved 2026-10-22). The current set came from `qwen3vl-8b-saha-sft-merged`; any re-sampling uses the detect-tool SFT (`qwen3vl-8b-saha-detect-sft-merged`). The old proposal-injected runs are dropped; all new training and eval use the detect-tool model. **NEVER run a new teacher (gpt-5.5) trace generation without a recorded human decision.**
- GRPO tool-collapse fix = the tool-gain reward + standard GRPO advantage. The stratified tool/no-tool SAN is an ablation only. Never implement AXPO (cite-only). Never claim a new GRPO algorithm; never write "unbiased" without a derivation. AT-GRPO's `ΔS·exp(−γ(·)²)` exists only as the labeled baseline (`reward_manager=AdaTooler-V`).
- No GT leakage: GT is training/eval-only. The tool server never receives GT. `extra_info.proposal_anchor` is kept for eval/metric stratification only. SDS-v2 factors are eval-only stratification axes.
- Keep the old pipelines runnable as baselines: `sds_grpo.py` (old area-SDS, frozen, untouched), proposal-injected SAHA-CF (`SAHA_CF_GROUNDING_SREF=anchor`, `data/{train,val}.parquet`), `AdaTooler-V.py`. Detect-tool data goes to NEW files (`*_detect.parquet`, `/workspace/data/hoi_sft_detect`); never overwrite the old ones.
- **Env:** `ultralytics==8.4.183` is in `/workspace/envs/verl-tool-env`, installed `--no-deps`. Never `pip install ultralytics` plainly: it pulls numpy 2 + opencv 5 and breaks verl. `pip check` flagging `opencv-python` is expected, because `opencv-python-headless` provides cv2.
- Every implementation step gets an independent Codex check through Herdr before the next step. Stop and report missing paths/fields; never silently invent a different method.

**Previous plan (history; superseded on proposals/tools, the grounding `s_ref` (anchor → no-tool siblings) and the SFT data (rewritten for detect)):** SAHA v2/v3 proposal-injected PTG reward. See `docs/saha-v2/reward-v3-counterfactual-spec.md`, `docs/saha-v2/latest-decision.md`, `plan/saha-v3-simple/`, and `docs/saha-v2/meeting-notes-prof-2026-06-26.md` (what the hard-case evidence did and did not show). Monitoring still applies: `saha_cf/*` WandB metrics via `verl_tool/trainer/ppo/metric_util.py::compute_saha_cf_metrics`, `examples/train/hoi/watch_tool_use.py`, and `docs/saha-v2/SAHA-CF-RUNBOOK.md`. The parquet `proposal_anchor` migration is documented in `examples/data_preprocess/hoi/migrate_add_proposal_anchor.py`.

**Drift checks (run before claiming done):**
- `rg -n "inspect_pose|detector-free" verltool/ --type py` → nothing active
- `rg -n "Delta S|Tool Benefit|AT-GRPO|AdaTooler" verltool/ --type py` → `AdaTooler-V.py` + docs only

**Sibling repos:** `/workspace/saha-hoi` (SAHA v1 code; YOLOE weights `yoloe-26l-seg-pf.pt`), `/workspace/hoi-benchmarks` (eval harness; needs a detect-tool mode, see PLAN §E), `/workspace/hoi-tool-use-checkpoints` (shared checkpoints: `base-checkpoints/`, `sft-checkpoints/`, `grpo-checkpoints/`; `test_proposals/` = old offline YOLOE output). SFT runs in `/workspace/LlamaFactory` (env `/workspace/envs/llamafactory`).

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

**This machine (8×H200, driver 550, 2026-10-21):** the env lives on the persistent volume at `/workspace/envs/verl-tool-env` (the container's `/opt/conda` is wiped when the container is removed). Activate it with `conda activate /workspace/envs/verl-tool-env`. Rebuild it with `bash verltool/scripts/setup_verl_tool_env.sh`, which installs exact versions from `verltool/scripts/verl_tool_env.lock.txt` (a pip freeze of the verified env: torch 2.8.0+cu128, vllm 0.11.0, transformers 4.57.6, numpy 1.26.4) plus the sha-pinned flash-attn 2.8.3 wheel. Unpinned pip pulls transformers 5 / numpy 2, which break verl. After an intentional upgrade, regenerate the lockfile. The script creates the env from conda-forge only, so no conda ToS prompt. It first recreates the image symlink `/media/shaun/workspace/hoi/dataset → /workspace/data`, which the parquets' absolute image paths need. It lives on the container overlay and disappears with the container.

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
Script as committed assumes multi-GPU (3 GPUs/node). **Compute (2026-10-21): this machine has 8×H200 (shared; pick idle GPUs).** The current local launcher is `examples/train/hoi/run_grpo_v2_alpha0.sh` (proposal-injected SAHA SFT model). Detect-tool runs will use `run_grpo_detect_tool.sh` (PLAN §F).
**Checkpoints (2026-10-21):** the SAHA SFT used by recent GRPO runs is the LoRA SFT on `/workspace/data/hoi_sft_final` merged into `/workspace/data/qwen3vl-{4b,8b}-saha-sft-merged`. `/workspace/hoi-tool-use-checkpoints/sft-checkpoints/qwen3VL-{4B,8B}` (`hoi_v2_sft`, full SFT, 8B eval loss 0.442) is the older lineage. Detect-tool runs start from a NEW LoRA SFT on the rewritten data (PLAN §C), not from either.

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
- `saha_cf.py` (`reward_manager=SAHA-CF`) — the SAHA tool-gain reward `R_format·(R_outcome + α·R_tool)`. Proposal-injected runs use the grounding anchor `s_ref`; detect-tool runs use same-uid no-tool siblings (PLAN §D). The Rev-4 six-term `saha_v2` reward was never adopted and must not be built.

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

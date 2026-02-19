# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdaTooler-V is a multimodal LLM (MLLM) that performs **adaptive tool-use** for image and video reasoning tasks. It uses **AT-GRPO** (Adaptive Tool-GRPO), a reinforcement learning algorithm that adjusts reward scales based on a Tool Benefit Score, encouraging the model to invoke tools only when they provide genuine improvements. Built on top of [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), which itself extends the [verl](https://github.com/volcengine/verl) RL framework.

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
Requires 8x H100/A100 GPUs (80GB). Uses GRPO with `n=4`, batch size 16, tensor parallelism size 2.
Current SFT checkpoint: `/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-2B`

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

**Reward Managers** (`verl_tool/workers/reward_manager/`): Two primary managers:
- `AdaTooler-V.py` — AT-GRPO reward: calculates Tool Benefit Scores and adaptively scales rewards (original pipeline).
- `sds_grpo.py` — SDS-GRPO reward for HOI tasks: `R_total = R_format * (R_outcome + alpha * R_tool)`. Uses Spatial Difficulty Score (SDS) gated on object bounding-box area to decide when zoom_in/zoom_out tool use should be rewarded or penalized. Registered as `"SDS-GRPO"`.

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

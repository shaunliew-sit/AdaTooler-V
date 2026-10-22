#!/bin/bash
# Local (non-PBS) launcher for the alpha=0 reward ablation on the SAHA SFT model.
#
# Same experiment as examples/train/hoi/train_hoi_grpo_v2_alpha0.pbs, but for this
# workstation instead of NSCC: no `module load`, no PBS GPU assignment, local
# model/data paths, and a GPU subset you pick (this box is SHARED).
#
# ALPHA=0 -> R_total = R_format*R_outcome: the PTG tool term is off, i.e. vanilla
# GRPO reward on the SAHA SFT model. ALPHA is part of run_name, so this cannot
# collide with the a0.6 run and DATA_TAG stays the same for comparability.
#
# Usage:
#   cd /workspace/AdaTooler-V/verltool
#   bash examples/train/hoi/run_grpo_v2_alpha0.sh                  # GPU 0
#   GPUS=3 bash examples/train/hoi/run_grpo_v2_alpha0.sh           # use a different GPU
#   TOTAL_STEPS=50 bash examples/train/hoi/run_grpo_v2_alpha0.sh   # short probe
#
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # .../AdaTooler-V/verltool
cd "$REPO_ROOT"

# ── Conda env ──
# The recipe needs the verl-tool env (torch/vllm/verl/flash-attn). Activate it
# yourself, or set CONDA_ENV to have this script do it.
CONDA_ENV=${CONDA_ENV:-verl-tool-env}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        echo "WARNING: conda env '$CONDA_ENV' not active/found - using $(which python3)" >&2
    fi
fi
export LD_LIBRARY_PATH=${CONDA_PREFIX:-/usr}/lib:${LD_LIBRARY_PATH:-}

# ── GPUs ──
# This machine is shared: pick only idle GPUs (nvidia-smi) or the run will OOM
# against someone else's job. PBS did this for us; here it is manual.
# Single GPU by default -- the 8-GPU PBS sizing (BATCH_SIZE=16, MAX_CONCURRENT=96)
# does not fit one card, so the knobs below stay at the dev-box defaults.
GPUS=${GPUS:-0}
export CUDA_VISIBLE_DEVICES=$GPUS
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export NGPUS=$NUM_GPUS
# `ray stop --force` is node-wide and would kill co-tenant jobs on this box.
export SKIP_RAY_STOP=${SKIP_RAY_STOP:-1}

# ── Diagnostics ──
echo "=== Job Info ==="
echo "Host:  $(hostname)"
echo "Date:  $(date)"
echo "PWD:   $(pwd)"
echo "Env:   ${CONDA_DEFAULT_ENV:-none} ($(which python3))"
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" \
    || { echo "FATAL: torch not importable - build/activate the '$CONDA_ENV' conda env first." >&2; exit 1; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (using $NUM_GPUS GPUs)"
echo ""

# ── Logging ──
export NCCL_DEBUG=INFO
export WANDB_PROJECT=SAHA-CF
# Log live so the run is watchable at wandb.ai/<entity>/SAHA-CF while it trains
# (credentials come from ~/.netrc). Set WANDB_MODE=offline to log locally instead --
# then `wandb sync $WANDB_DIR/wandb/offline-run-*` once the run is done.
export WANDB_MODE=${WANDB_MODE:-online}
# Local model paths must not be validated as HF repo IDs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$(pwd):$(pwd)/verl:${PYTHONPATH:-}

# ── Training knobs (read by train_qwen3vl_v2.sh) ──
export TP=1
export N=${N:-6}
export BATCH_SIZE=${BATCH_SIZE:-4}
export PPO_MINI=${PPO_MINI:-$BATCH_SIZE}
export MAX_CONCURRENT=${MAX_CONCURRENT:-$((BATCH_SIZE * N))}
export TOOL_WORKERS=${TOOL_WORKERS:-4}
export DO_OFFLOAD=${DO_OFFLOAD:-True}
export GPU_MEM=${GPU_MEM:-0.30}
export ALPHA=0
export TOTAL_STEPS=${TOTAL_STEPS:-1000}
export SAHA_CF_GROUNDING_METRIC=${SAHA_CF_GROUNDING_METRIC:-minAR10}
export DATA_TAG=${DATA_TAG:-newsft-resampled-8b}
export RESUME_MODE=${RESUME_MODE:-auto}
export TRAIN_PARQUET=${TRAIN_PARQUET:-$(pwd)/data/train_resampled_moderate.parquet}
export VAL_PARQUET=${VAL_PARQUET:-$(pwd)/data/val.parquet}

# ── Model path (local mirror of the NSCC merged SFT checkpoint) ──
MODEL_PATH=${1:-${MODEL_PATH:-/workspace/data/qwen3vl-8b-saha-sft-merged}}

# ── Pre-flight ──
for p in "$MODEL_PATH/config.json" "$TRAIN_PARQUET" "$VAL_PARQUET"; do
    [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done
# The parquet stores ABSOLUTE image paths from the machine that built it
# (/media/shaun/workspace/hoi/dataset/...). On this box the datasets live under
# /workspace/data, bridged by a symlink; without it every rollout loses its image.
python3 - "$TRAIN_PARQUET" <<'PYCHK' || exit 1
import os, sys, pyarrow.parquet as pq
row = next(pq.ParquetFile(sys.argv[1]).iter_batches(batch_size=1)).to_pandas()
imgs = list(row["extra_info"][0]["images"])
missing = [p for p in imgs if not os.path.exists(p)]
if missing:
    print(f"FATAL: image path from the parquet does not resolve: {missing[0]}", file=sys.stderr)
    print("       Fix: mkdir -p /media/shaun/workspace/hoi && "
          "ln -sfn /workspace/data /media/shaun/workspace/hoi/dataset", file=sys.stderr)
    sys.exit(1)
print(f"images OK (checked {len(imgs)}): {imgs[0]}")
PYCHK

echo "=== Starting GRPO Training (alpha=0 ablation, SAHA SFT model) ==="
echo "Model: $MODEL_PATH"
echo "Data:  $TRAIN_PARQUET"
echo "Steps: $TOTAL_STEPS   alpha=$ALPHA  n=$N  batch=$BATCH_SIZE"
echo ""

if bash examples/train/hoi/train_qwen3vl_v2.sh "$MODEL_PATH"; then
    echo "=== Training Complete ==="
else
    rc=$?
    echo "=== Training FAILED (exit $rc) - see the traceback above ===" >&2
    echo "End: $(date)" >&2
    exit $rc
fi
echo "End: $(date)"

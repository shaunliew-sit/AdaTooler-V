#!/bin/bash
# Local launcher for SAHA GRPO with the detector as a tool (plan/saha-detect-tool/PLAN.md §F).
#
# Same recipe as run_grpo_v2_alpha0.sh, changed only where the detect tool needs it:
# detect-tool parquets (no proposals in the prompt, extra_info.detect_image), the
# detect-tool SFT model, grounding s_ref = same-uid no-tool siblings
# (SAHA_CF_GROUNDING_SREF=group_no_tool), a longer response budget for the detect
# observation, and the YOLOE settings for the tool server.
#
# Usage:
#   cd /workspace/AdaTooler-V/verltool
#   bash examples/train/hoi/run_grpo_detect_tool.sh                  # GPU 0, alpha 0.6
#   TOTAL_STEPS=1 bash examples/train/hoi/run_grpo_detect_tool.sh    # smoke
#   ALPHA=0 bash examples/train/hoi/run_grpo_detect_tool.sh          # alpha=0 ablation
#
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # .../AdaTooler-V/verltool
cd "$REPO_ROOT"

# ── Conda env ──
# The recipe needs the verl-tool env (torch/vllm/verl/flash-attn). Activate it
# yourself, or set CONDA_ENV to have this script do it.
CONDA_ENV=${CONDA_ENV:-/workspace/envs/verl-tool-env}
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
# wandb login done on this box (2026-10-22, ~/.netrc). Redo it if the container is recreated.
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
export ALPHA=${ALPHA:-0.6}
export TOTAL_STEPS=${TOTAL_STEPS:-1000}
export SAHA_CF_GROUNDING_METRIC=${SAHA_CF_GROUNDING_METRIC:-minAR10}
export DATA_TAG=${DATA_TAG:-detect-tool-8b}
export RESUME_MODE=${RESUME_MODE:-auto}
export TRAIN_PARQUET=${TRAIN_PARQUET:-$(pwd)/data/train_resampled_moderate_detect.parquet}
export VAL_PARQUET=${VAL_PARQUET:-$(pwd)/data/val_detect.parquet}
export SAHA_CF_GROUNDING_SREF=group_no_tool
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144}
# YOLOE in the tool server. CPU (~0.2 s/call) keeps it off the training GPU; set a
# GPU index to speed it up when there is memory to spare.
export DETECT_TOOL_DEVICE=${DETECT_TOOL_DEVICE:-cpu}
export DETECT_TOOL_WEIGHTS=${DETECT_TOOL_WEIGHTS:-/workspace/saha-hoi/yoloe-26l-seg-pf.pt}

# ── Model path: the detect-tool SFT (LlamaFactory qwen3vl_8b_saha_detect_merge.yaml) ──
MODEL_PATH=${1:-${MODEL_PATH:-/workspace/data/qwen3vl-8b-saha-detect-sft-merged}}

# ── Pre-flight ──
for p in "$MODEL_PATH/config.json" "$TRAIN_PARQUET" "$VAL_PARQUET" "$DETECT_TOOL_WEIGHTS"; do
    [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done
# The parquet stores ABSOLUTE image paths from the machine that built it
# (/media/shaun/workspace/hoi/dataset/...). On this box the datasets live under
# /workspace/data, bridged by a symlink; without it every rollout loses its image.
python3 - "$TRAIN_PARQUET" <<'PYCHK' || exit 1
import os, sys, pyarrow.parquet as pq
row = next(pq.ParquetFile(sys.argv[1]).iter_batches(batch_size=1)).to_pandas()
imgs = list(row["extra_info"][0]["images"])
detect_image = row["extra_info"][0].get("detect_image")
if not detect_image or not os.path.exists(detect_image):
    print(f"FATAL: extra_info.detect_image missing or unresolved ({detect_image}); "
          "build the parquet with migrate_to_detect_tool.py", file=sys.stderr)
    sys.exit(1)
import ultralytics  # noqa: F401  (the tool server needs it for detect)
missing = [p for p in imgs if not os.path.exists(p)]
if missing:
    print(f"FATAL: image path from the parquet does not resolve: {missing[0]}", file=sys.stderr)
    print("       Fix: mkdir -p /media/shaun/workspace/hoi && "
          "ln -sfn /workspace/data /media/shaun/workspace/hoi/dataset", file=sys.stderr)
    sys.exit(1)
print(f"images OK (checked {len(imgs)}): {imgs[0]}")
PYCHK

echo "=== Starting GRPO Training (detect tool, alpha=$ALPHA) ==="
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

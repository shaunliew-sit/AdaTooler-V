#!/bin/bash
# SDS-GRPO HOI Evaluation Script
#
# Evaluates a trained HOI model on grounding and referring tasks.
# Uses the eval_hoi_agent.py evaluation pipeline.
#
# Usage:
#   cd verltool
#   bash examples/train/hoi/eval.sh [CHECKPOINT_PATH] [STEP]
#
# Prerequisites:
#   - Trained checkpoint with actor/huggingface directory
#   - Test annotation files in benchmarks_simplified/
#   - HICO/SWIG test images
set -x

# Paths
CHECKPOINT_PATH=${1:-"checkpoints/SDS-GRPO/global_step_100/actor/huggingface"}
STEP=${2:-"100"}
DATA_DIR="/media/shaun/workspace/hoi/dataset/benchmarks_simplified"
HICO_IMG_DIR="/media/shaun/workspace/hoi/dataset/hico_20160224_det/images/test2015"
SWIG_IMG_DIR="/media/shaun/workspace/hoi/dataset/swig_hoi/images_512"
PROPOSALS_DIR="/media/shaun/workspace/hoi-dataset-curation/output/proposals"
EVAL_SCRIPT_DIR="/media/shaun/workspace/hoi/verl-tool/examples/eval/hoi"
OUTPUT_BASE="results/sds_grpo_step${STEP}"

# vLLM server config
PORT=8000
MODEL_NAME="hoi-sds-grpo"
TP_SIZE=4

echo "=============================================="
echo "SDS-GRPO HOI Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Step:       $STEP"
echo "Output:     $OUTPUT_BASE"
echo "=============================================="

# Check checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    find checkpoints -name "huggingface" -type d 2>/dev/null | head -10
    exit 1
fi

# Start vLLM server
echo "Starting vLLM server on port $PORT with TP=$TP_SIZE..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$CHECKPOINT_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port $PORT \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size $TP_SIZE \
    --enable-auto-tool-choice \
    --tool-call-parser hermes &
VLLM_PID=$!
echo "vLLM server started (pid=$VLLM_PID)"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Start tool server for agent mode
host=$(hostname -i | awk '{print $1}')
tool_port=$(shuf -i 31000-32000 -n 1)
tool_server_url=http://$host:$tool_port/get_observation
python -m verl_tool.servers.serve --host $host --port $tool_port --tool_type "pixel_reasoner" --workers_per_tool 4 &
TOOL_PID=$!
echo "Tool server started (pid=$TOOL_PID) at $tool_server_url"
sleep 5

# ----- HICO Grounding -----
echo ""
echo "=== HICO Grounding Evaluation ==="
python ${EVAL_SCRIPT_DIR}/eval_hoi_agent.py \
    --task grounding \
    --dataset hico \
    --ann-file ${DATA_DIR}/hico_ground_test_simplified.json \
    --img-prefix ${HICO_IMG_DIR} \
    --proposals-dir ${PROPOSALS_DIR} \
    --endpoint http://localhost:${PORT}/v1 \
    --model ${MODEL_NAME} \
    --concurrency 8 \
    --save-thinking \
    --output-dir ${OUTPUT_BASE}/hico_grounding

# ----- HICO Referring -----
echo ""
echo "=== HICO Referring Evaluation ==="
python ${EVAL_SCRIPT_DIR}/eval_hoi_agent.py \
    --task referring \
    --dataset hico \
    --ann-file ${DATA_DIR}/hico_referring_test_simplified.json \
    --img-prefix ${HICO_IMG_DIR} \
    --proposals-dir ${PROPOSALS_DIR} \
    --endpoint http://localhost:${PORT}/v1 \
    --model ${MODEL_NAME} \
    --concurrency 8 \
    --bertscore-gpu 4 \
    --save-thinking \
    --output-dir ${OUTPUT_BASE}/hico_referring

# ----- SWIG Grounding -----
echo ""
echo "=== SWIG Grounding Evaluation ==="
python ${EVAL_SCRIPT_DIR}/eval_hoi_agent.py \
    --task grounding \
    --dataset swig \
    --ann-file ${DATA_DIR}/swig_ground_test_simplified.json \
    --img-prefix ${SWIG_IMG_DIR} \
    --proposals-dir ${PROPOSALS_DIR} \
    --endpoint http://localhost:${PORT}/v1 \
    --model ${MODEL_NAME} \
    --concurrency 8 \
    --save-thinking \
    --output-dir ${OUTPUT_BASE}/swig_grounding

# ----- SWIG Referring -----
echo ""
echo "=== SWIG Referring Evaluation ==="
python ${EVAL_SCRIPT_DIR}/eval_hoi_agent.py \
    --task referring \
    --dataset swig \
    --ann-file ${DATA_DIR}/swig_referring_test_simplified.json \
    --img-prefix ${SWIG_IMG_DIR} \
    --proposals-dir ${PROPOSALS_DIR} \
    --endpoint http://localhost:${PORT}/v1 \
    --model ${MODEL_NAME} \
    --concurrency 8 \
    --bertscore-gpu 4 \
    --save-thinking \
    --output-dir ${OUTPUT_BASE}/swig_referring

# Cleanup
echo ""
echo "=== Cleanup ==="
kill -9 $VLLM_PID 2>/dev/null
kill -9 $TOOL_PID 2>/dev/null
echo "Done. Results saved to ${OUTPUT_BASE}/"

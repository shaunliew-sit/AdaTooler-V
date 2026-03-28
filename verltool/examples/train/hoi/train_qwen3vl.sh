#!/bin/bash
# SDS-GRPO Training for HOI (Qwen3-VL-4B)
#
# Usage:
#   cd verltool
#   bash examples/train/hoi/train_qwen3vl.sh [MODEL_PATH]
#
# 
set -x
unset ROCR_VISIBLE_DEVICES

# ── Kill stale Ray actors / GPU contexts from previous failed runs ──
ray stop --force 2>/dev/null || true
sleep 2

# ── Fix PBS UUID-based CUDA_VISIBLE_DEVICES for vllm compatibility ──
# PBS assigns GPUs as UUIDs (e.g. GPU-4f6613e0-...) but vllm requires integer indices (0,1,2,...)
if [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
    NEW_IDS=""
    IFS=',' read -ra UUIDS <<< "$CUDA_VISIBLE_DEVICES"
    for uuid in "${UUIDS[@]}"; do
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
              | grep "$uuid" | awk -F',' '{print $1}' | tr -d ' ')
        NEW_IDS="${NEW_IDS},${idx}"
    done
    export CUDA_VISIBLE_DEVICES="${NEW_IDS#,}"
    echo "Remapped CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# Model — override via first CLI argument
model_name=${1:-"/home/users/sit/a104059/scratch/checkpoints/qwen3-vl-4b"}

# Data
train_data=[$(pwd)/data/hoi/train.parquet]
val_data=[$(pwd)/data/hoi/val.parquet]

# RL config
rl_alg=grpo
reward_manager=SDS-GRPO
n=4
batch_size=8
ppo_mini_batch_size=8

# Sequence lengths
max_prompt_length=8192
max_response_length=4096
max_action_length=2048
max_obs_length=8192
free_cache_engine=True
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Sampling
temperature=1.0
top_p=1.0

# Agent config
enable_agent=True
action_stop_tokens='</tool_call>'
max_turns=5
mask_observations=True
enable_mtrl=True

# Training
strategy="fsdp"
lr=5e-7
kl_loss_coef=0.04
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl

# GPU config
n_gpus_per_node=8
n_nodes=1
tensor_model_parallel_size=2
gpu_memory_utilization=0.40
do_offload=True
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1

# Micro batch sizes
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1

# Misc
additional_eos_token_ids=[151645]  # <|im_end|>
max_num_batched_tokens=10000
rollout_mode='async'

# Run name
model_pretty_name=$(echo $model_name | tr '[:upper:]' '[:lower:]' | awk -F'/' '{print $(NF-1)"_"$NF}' | tr -c 'a-z0-9_.-' '_')
#run_name="${reward_manager}-${strategy}-tool-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}"
run_name="${reward_manager}-${strategy}-tool-agent-${model_pretty_name}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
# wandb needs a local (non-NFS) temp dir — HPC home/scratch dirs are too slow for the 30s timeout
export WANDB_DIR=/tmp/${USER}_wandb_$$
mkdir -p $WANDB_DIR
# Write action stop tokens to temp file (verl cannot pass special strings as params)
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# Start tool server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "pixel_reasoner" --workers_per_tool 4 &
server_pid=$!
echo "Tool server (pid=$server_pid) started at $tool_server_url"

# Wait for server startup
sleep 10

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=64 \
    data.dataloader_num_workers=4 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache_engine \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$(pwd)/verl_step_records/$run_name \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=1000 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/home/users/sit/a104059/scratch/AdaTooler-V/verltool/checkpoints/SDS-GRPO/SDS-GRPO-fsdp-tool-agent-actor_huggingface_/global_step_820 \

pkill -P -9 $server_pid
kill -9 $server_pid
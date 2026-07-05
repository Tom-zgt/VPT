#!/usr/bin/env bash
# ============================================================================
# Training launcher for VPT (dual-stream flow + role/seg Wan LoRA fine-tuning).
#
# Prerequisite: prepare the data first (see data_preparation/README.md):
#   data/videos_data/videos/       RGB clips (WISA-80K)
#   data/latent_data/flow_videos/  RAFT optical-flow      (auxiliary modality)
#   data/latent_data/seg_videos/   Qwen3-VL + SAM3 maps   (auxiliary modality)
# The flow/seg roots are read from WanModelSpecification defaults; role/flow
# (triple-discrete) timestep decoupling and auxiliary-loss annealing are enabled
# automatically by the trainer when those latents are present.
#
# Standard torchrun launch (single- or multi-node). Topology via env vars:
# NNODES, NODE_RANK, NPROC_PER_NODE, MASTER_ADDR, MASTER_PORT. Single node needs
# none of them; for N nodes run this on each node with the same MASTER_ADDR/PORT
# and a distinct NODE_RANK (0..N-1). All knobs below are env-overridable.
# ============================================================================
set -euo pipefail

# ---- paths (override via env) ----------------------------------------------
SCRIPT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_SELF}/../../../../../.." && pwd)}"
SCRIPT_DIR="examples/training/sft/wan/3dgs_dissolve"

# Base Wan model: a local dir or a HuggingFace repo id.
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
TRAINING_DATASET_CONFIG="${TRAINING_DATASET_CONFIG:-${SCRIPT_DIR}/training.json}"
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-${SCRIPT_DIR}/videophy.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/vpt_train}"

# ---- training hyper-parameters (paper defaults) ----------------------------
TRAIN_STEPS="${TRAIN_STEPS:-2000}"        # released LoRA is the step-2000 checkpoint
BATCH_SIZE="${BATCH_SIZE:-6}"             # per-GPU; 8 GPUs -> global batch 48
LR="${LR:-5e-5}"
RANK="${RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-1000}"
RESUME="${RESUME:-latest}"                # "latest" to resume, or a step dir
REPORT_TO="${REPORT_TO:-none}"            # none | wandb (set WANDB_MODE=offline for offline)

# ---- distributed topology (standard torchrun) ------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DP_DEGREE=$(( NNODES * NPROC_PER_NODE ))

cd "$REPO_ROOT"

echo "-------------------- train.sh --------------------"
echo "nodes=$NNODES x gpus=$NPROC_PER_NODE (dp=$DP_DEGREE)  train_steps=$TRAIN_STEPS  lr=$LR  batch=$BATCH_SIZE"
echo "dataset=$TRAINING_DATASET_CONFIG  output_dir=$OUTPUT_DIR"
echo "--------------------------------------------------"

torchrun --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  train.py \
  --parallel_backend ptd --pp_degree 1 --dp_degree "$DP_DEGREE" --dp_shards 1 --cp_degree 1 --tp_degree 1 \
  --model_name wan \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_PATH" \
  --dataset_config "$TRAINING_DATASET_CONFIG" \
  --dataset_shuffle_buffer_size 1 \
  --dataloader_num_workers 1 \
  --flow_weighting_scheme logit_normal \
  --training_type lora \
  --seed 42 \
  --batch_size "$BATCH_SIZE" \
  --train_steps "$TRAIN_STEPS" \
  --rank "$RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules 'blocks.*(to_q|to_k|to_v|to_out.0)' \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps "$CHECKPOINTING_STEPS" \
  --checkpointing_limit 10 \
  --resume_from_checkpoint "$RESUME" \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --lr "$LR" \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 1000 \
  --lr_num_cycles 0 \
  --beta1 0.9 --beta2 0.99 --weight_decay 1e-4 --epsilon 1e-8 --max_grad_norm 1.0 \
  --validation_dataset_file "$VALIDATION_DATASET_FILE" \
  --validation_steps "$VALIDATION_STEPS" \
  --tracker_name finetrainers-wan-vpt \
  --output_dir "$OUTPUT_DIR" \
  --init_timeout 2600 --nccl_timeout 2600 \
  --report_to "$REPORT_TO"

echo "-------------------- done: checkpoints in ${OUTPUT_DIR} --------------------"

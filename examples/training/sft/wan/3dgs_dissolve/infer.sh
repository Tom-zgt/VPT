#!/usr/bin/env bash
# ============================================================================
# Inference / reproduction launcher for the dual-stream (flow + seg) Wan model.
#
# This runs the trained LoRA in *inference-only* mode: it drives `train.py`
# with `--train_steps 1`, so the training loop is skipped and only the
# validation path runs, generating one video per prompt into --save_videos_dir.
#
# Standard torchrun launch (single- or multi-node). Set the usual torchrun
# topology via env vars: NNODES, NODE_RANK, NPROC_PER_NODE, MASTER_ADDR,
# MASTER_PORT. Single node needs none of them. For N nodes, run this script on
# each node with the same MASTER_ADDR/MASTER_PORT and a distinct NODE_RANK
# (0..N-1). All other knobs are overridable via env vars too.
#
# Triple-discrete (flow/seg) timestep conditioning is auto-enabled at load time:
# CustomWanPipeline detects `cond_proj` weights in the LoRA checkpoint and swaps
# in WanTripleDiscreteTimeTextImageEmbedding. During inference all three streams
# share the main diffusion timestep, so NO extra "multi_timestep_mode" flag is
# needed (and none is defined in args.py).
# ============================================================================
set -euo pipefail

# ---- paths (override via env) ----------------------------------------------
# Repo root = directory that contains train.py. Defaults to this script's repo.
SCRIPT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_SELF}/../../../../../.." && pwd)}"
SCRIPT_DIR="examples/training/sft/wan/3dgs_dissolve"

# Base Wan model: a local dir or a HuggingFace repo id.
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
# Path to the trained VPT LoRA checkpoint (.safetensors). REQUIRED.
LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-./checkpoints/vpt_lora/pytorch_lora_weights.safetensors}"

# ---- benchmark selection ---------------------------------------------------
# videophy | videophy2 | vbench | vbench2 | ood
BENCHMARK="${BENCHMARK:-videophy}"
case "$BENCHMARK" in
  videophy)  VALIDATION_DATASET_FILE="${SCRIPT_DIR}/videophy.json"  ;;
  videophy2) VALIDATION_DATASET_FILE="${SCRIPT_DIR}/videophy2.json" ;;
  vbench)    VALIDATION_DATASET_FILE="${SCRIPT_DIR}/vbench.json"    ;;
  vbench2)   VALIDATION_DATASET_FILE="${SCRIPT_DIR}/vbench2.json"   ;;
  ood)       VALIDATION_DATASET_FILE="${SCRIPT_DIR}/ood.json"       ;;
  *) echo "Unknown BENCHMARK=$BENCHMARK" >&2; exit 2 ;;
esac

SAVE_VIDEOS_DIR="${SAVE_VIDEOS_DIR:-./outputs/wan_eval_${BENCHMARK}}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/wan_infer_workdir}"

# ---- guidance knobs --------------------------------------------------------
WAN_VALIDATION_GUIDANCE_SCALE="${WAN_VALIDATION_GUIDANCE_SCALE:-1.8}"
WAN_VALIDATION_FLOW_GUIDANCE_SCALE="${WAN_VALIDATION_FLOW_GUIDANCE_SCALE:-0.0}"
WAN_VALIDATION_SEG_GUIDANCE_SCALE="${WAN_VALIDATION_SEG_GUIDANCE_SCALE:-0.0}"
WAN_VALIDATION_INNER_GUIDANCE="${WAN_VALIDATION_INNER_GUIDANCE:-1}"   # 1=inner (chain) CFG, 0=text-only CFG

# ---- eval env consumed by the validation code ------------------------------
export WAN_EVAL_STYLE="$BENCHMARK"
export WAN_EVAL_VIDEO_FPS="${WAN_EVAL_VIDEO_FPS:-16}"
case "$BENCHMARK" in
  videophy|videophy2) export WAN_SKIP_EXISTING_VIDEO="${WAN_SKIP_EXISTING_VIDEO:-1}" ;;
  *)                  export WAN_SKIP_EXISTING_VIDEO="${WAN_SKIP_EXISTING_VIDEO:-0}" ;;
esac
export WAN_VALIDATION_MAX_VIDEOS_PER_FORWARD="${WAN_VALIDATION_MAX_VIDEOS_PER_FORWARD:-5}"
export WAN_VALIDATION_GUIDANCE_SCALE WAN_VALIDATION_FLOW_GUIDANCE_SCALE \
       WAN_VALIDATION_SEG_GUIDANCE_SCALE WAN_VALIDATION_INNER_GUIDANCE

# inner-guidance boolean flag
if [[ "$WAN_VALIDATION_INNER_GUIDANCE" =~ ^(0|false|no|off)$ ]]; then
  INNER_FLAG=(--no-validation_enable_inner_guidance)
else
  INNER_FLAG=(--validation_enable_inner_guidance)
fi

# ---- distributed topology (standard torchrun) ------------------------------
# Single node: defaults are fine. Multi-node: set NNODES, NODE_RANK,
# MASTER_ADDR, MASTER_PORT identically-ish across nodes (distinct NODE_RANK).
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DP_DEGREE=$(( NNODES * NPROC_PER_NODE ))

cd "$REPO_ROOT"

echo "-------------------- infer.sh --------------------"
echo "benchmark=$BENCHMARK  nodes=$NNODES x gpus=$NPROC_PER_NODE (dp=$DP_DEGREE)"
echo "gs=$WAN_VALIDATION_GUIDANCE_SCALE flow=$WAN_VALIDATION_FLOW_GUIDANCE_SCALE seg=$WAN_VALIDATION_SEG_GUIDANCE_SCALE inner=$WAN_VALIDATION_INNER_GUIDANCE"
echo "save_videos_dir=$SAVE_VIDEOS_DIR"
echo "--------------------------------------------------"

torchrun --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  train.py \
  --parallel_backend ptd --pp_degree 1 --dp_degree "$DP_DEGREE" --dp_shards 1 --cp_degree 1 --tp_degree 1 \
  --model_name wan \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_PATH" \
  --save_videos_dir "$SAVE_VIDEOS_DIR" \
  --val_num_per_prompt 1 \
  --val_batch_size 1 \
  --validation_guidance_scale "$WAN_VALIDATION_GUIDANCE_SCALE" \
  --validation_flow_guidance_scale "$WAN_VALIDATION_FLOW_GUIDANCE_SCALE" \
  --validation_seg_guidance_scale "$WAN_VALIDATION_SEG_GUIDANCE_SCALE" \
  "${INNER_FLAG[@]}" \
  --dataset_config "${SCRIPT_DIR}/training.json" \
  --dataset_shuffle_buffer_size 1 \
  --dataloader_num_workers 1 \
  --flow_weighting_scheme logit_normal \
  --training_type lora \
  --load_lora_weights11 "$LORA_WEIGHTS_PATH" \
  --seed 42 \
  --batch_size 6 \
  --train_steps 1 \
  --rank 32 \
  --lora_alpha 32 \
  --target_modules 'blocks.*(to_q|to_k|to_v|to_out.0)' \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 1000 \
  --checkpointing_limit 10 \
  --resume_from_checkpoint latest \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --lr 5e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 1000 \
  --lr_num_cycles 0 \
  --beta1 0.9 --beta2 0.99 --weight_decay 1e-4 --epsilon 1e-8 --max_grad_norm 1.0 \
  --validation_dataset_file "$VALIDATION_DATASET_FILE" \
  --validation_steps 1 \
  --tracker_name "finetrainers-wan-${BENCHMARK}" \
  --output_dir "$OUTPUT_DIR" \
  --init_timeout 2600 --nccl_timeout 2600 \
  --report_to none

echo "-------------------- done: videos in ${SAVE_VIDEOS_DIR} --------------------"

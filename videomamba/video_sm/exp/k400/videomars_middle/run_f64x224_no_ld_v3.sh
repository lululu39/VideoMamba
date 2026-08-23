#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="video-mars-middle-k400-f64x224-no-ld-v3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_SM_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_DIR="/mnt/localssd/experiments/videovit/${RUN_NAME}"
OUTPUT_DIR="${RUN_DIR}/ckpt"
IMAGE_CHECKPOINT="/mnt/localssd/experiments/videovit/video-vit-middle-imagenet-1k-pretrain/ckpt/best_checkpoint.pth"
DATA_PATH="/mnt/localssd/dataset/videovit/k400/kinetics_400"
VIDEO_PREFIX="${DATA_PATH}/videos_320"

test -f "${IMAGE_CHECKPOINT}"
test -f "${DATA_PATH}/train.csv"
test -f "${DATA_PATH}/val.csv"
test -d "${VIDEO_PREFIX}"
mkdir -p "${OUTPUT_DIR}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate videovit

unset WANDB_BASE_URL
export OMP_NUM_THREADS=1
export PYTHONPATH="${VIDEO_SM_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

cd "${VIDEO_SM_DIR}"
exec "${CONDA_PREFIX}/bin/torchrun" \
    --nnodes=1 \
    --nproc-per-node=8 \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    run_class_finetuning.py \
    --model videomars_middle \
    --finetune "${IMAGE_CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --prefix "${VIDEO_PREFIX}" \
    --data_set Kinetics_sparse \
    --split ',' \
    --nb_classes 400 \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 4 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 64 \
    --num_workers 12 \
    --warmup_epochs 1 \
    --tubelet_size 1 \
    --epochs 10 \
    --lr 2e-4 \
    --warmup_lr 2e-6 \
    --min_lr 2e-6 \
    --layer_decay 1.0 \
    --drop_path 0.8 \
    --mlp_ratio 3.0 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --fw_base_lr 0.01 \
    --fw_update_group_size 8 \
    --fw_update_layer_group_size 16 \
    --muon_update_steps 5 \
    --mars_mask_ratio 0.75 \
    --mars_cnn_dim 64 \
    --use_checkpoint \
    --checkpoint_num 32 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
    --bf16 \
    --wandb \
    --wandb_entity LVSM-Experiment \
    --wandb_project videosft \
    --wandb_run_name "${RUN_NAME}"

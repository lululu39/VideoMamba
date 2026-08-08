#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="video-vit-middle-imagenet-1k-pretrain"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_SM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="/mnt/localssd/experiments/videovit/${RUN_NAME}"
OUTPUT_DIR="${RUN_DIR}/ckpt"

mkdir -p "${OUTPUT_DIR}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate videovit

unset WANDB_BASE_URL
export OMP_NUM_THREADS=1
export PYTHONPATH="${IMAGE_SM_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

cd "${IMAGE_SM_DIR}"
exec torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    main.py \
    --root_dir_train /mnt/localssd/dataset/videovit/imagenet-1k/train \
    --meta_file_train /mnt/localssd/dataset/videovit/imagenet-1k/meta/train.txt \
    --root_dir_val /mnt/localssd/dataset/videovit/imagenet-1k/val \
    --meta_file_val /mnt/localssd/dataset/videovit/imagenet-1k/meta/val.txt \
    --model videovit_middle \
    --input-size 224 \
    --batch-size 128 \
    --num_workers 16 \
    --epochs 300 \
    --warmup-epochs 20 \
    --lr 5e-4 \
    --warmup-lr 5e-7 \
    --min-lr 5e-6 \
    --weight-decay 0.05 \
    --drop-path 0.5 \
    --clip-grad 5.0 \
    --mlp-ratio 3.0 \
    --no-model-ema \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 \
    --dist-eval \
    --wandb \
    --wandb-entity LVSM-Experiment \
    --wandb-project videosft \
    --wandb-run-name "${RUN_NAME}"

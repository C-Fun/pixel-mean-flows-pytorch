#!/bin/bash

# ── Devices ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# ── Module Path ───────────────────────────────────────────────────────────────
export PYTHONPATH=$PYTHONPATH:"./pMF/"

# ── Model ─────────────────────────────────────────────────────────────────────
HF_REPO_ID=Lyy0725/pMF
HF_FILENAME=pMF-B-16.pt
MODEL_NAME=pmfDiT_B_16
IMG_SIZE=256
FID_REF_URL=https://raw.githubusercontent.com/LTH14/JiT/refs/heads/main/fid_stats/jit_in${IMG_SIZE}_stats.npz

# ── Run parameters ────────────────────────────────────────────────────────────
WORKDIR=./b16_fid_output
NUM_SAMPLES=20000
GEN_BSZ=64
SAMPLE_SEED=42
NUM_SAMPLING_STEPS=1
CFG_OMEGA=7.5
INTERVAL_MIN=0.1
INTERVAL_MAX=0.8
SAVE_SAMPLES=false   # set to true to keep generated images after FID evaluation

# ── Launch ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAVE_SAMPLES_FLAG=""
if [ "$SAVE_SAMPLES" = "true" ]; then
    SAVE_SAMPLES_FLAG="--save-samples"
fi

torchrun --nproc-per-node=$NUM_GPUS "$SCRIPT_DIR/pMF_eval.py" \
    --workdir "$WORKDIR" \
    --hf-repo-id $HF_REPO_ID \
    --hf-filename $HF_FILENAME \
    --model $MODEL_NAME \
    --img-size $IMG_SIZE \
    --fid-ref $FID_REF_URL \
    --num-samples $NUM_SAMPLES \
    --gen-bsz $GEN_BSZ \
    --sample-seed $SAMPLE_SEED \
    --num-sampling-steps $NUM_SAMPLING_STEPS \
    --cfg-omega $CFG_OMEGA \
    --interval-min $INTERVAL_MIN \
    --interval-max $INTERVAL_MAX \
    $SAVE_SAMPLES_FLAG
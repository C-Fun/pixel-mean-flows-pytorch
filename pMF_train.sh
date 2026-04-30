#!/bin/bash

# ─── Environment ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# ─── Config & output ─────────────────────────────────────────────────────────
CONFIG="configs/pMF_B_16_flowers102.yml"
WORKDIR="./workdir/pmf_flowers102"

# ─── Launch ──────────────────────────────────────────────────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS}"
else
  LAUNCHER="python"
fi

$LAUNCHER pMF_train.py \
  --config  "${CONFIG}" \
  --workdir "${WORKDIR}"
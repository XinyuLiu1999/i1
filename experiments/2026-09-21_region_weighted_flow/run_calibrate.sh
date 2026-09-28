#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SFT_MANIFEST:?Set SFT_MANIFEST to the completed experiment cache.jsonl}"
: "${CALIBRATION_WORKDIR:?Set a fresh CALIBRATION_WORKDIR for measurements}"
SFT_INIT=${SFT_INIT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}
TRAIN_PYTHON=${TRAIN_PYTHON:-/root/miniconda3/envs/i1_sft/bin/python}
TRAIN_GPUS=${TRAIN_GPUS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export REGION_WANDB=0 PERCEPTUAL_WEIGHT=0 REGION_WEIGHT=0
mkdir -p "$CALIBRATION_WORKDIR"

"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TRAIN_GPUS" \
  "$EXPERIMENT_DIR/calibrate.py" --config "$EXPERIMENT_DIR/config.py" \
  --manifest "$SFT_MANIFEST" --init_from "$SFT_INIT" --workdir "$CALIBRATION_WORKDIR" \
  --fsdp "$TRAIN_GPUS" --tp 1 --batch_size 32 --grad_accum 1 --no_compile \
  --calibration-batches "${CALIBRATION_BATCHES:-64}" --completion-config "" "$@" \
  2>&1 | tee -a "$CALIBRATION_WORKDIR/calibration.log"

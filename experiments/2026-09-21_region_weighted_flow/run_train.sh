#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SFT_MANIFEST:?Set SFT_MANIFEST to the completed experiment cache.jsonl}"
SFT_INIT=${SFT_INIT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}
: "${SFT_WORKDIR:?Set a distinct SFT_WORKDIR for each loss arm}"
TRAIN_PYTHON=${TRAIN_PYTHON:-/root/miniconda3/envs/i1_sft/bin/python}
TRAIN_GPUS=${TRAIN_GPUS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-online}
: "${REGION_WEIGHT:?Set an explicitly calibrated REGION_WEIGHT, or 0 for the control}"
export REGION_WEIGHT
export PERCEPTUAL_WEIGHT=${PERCEPTUAL_WEIGHT:-0}

completion_args=(--completion-config "")
step_args=()
if [[ -n ${TRAIN_STEPS:-} ]]; then
  step_args=(--total_steps "$TRAIN_STEPS")
fi
if [[ ${DISABLE_AUTO_SHUTDOWN:-0} != 1 ]]; then
  : "${GPU_COMPLETION_CONFIG:?Set the private mode-0600 completion config, or DISABLE_AUTO_SHUTDOWN=1}"
  completion_args=(--completion-config "$GPU_COMPLETION_CONFIG")
fi
mkdir -p "$SFT_WORKDIR"
export WANDB_DIR=${WANDB_DIR:-$SFT_WORKDIR/wandb}
mkdir -p "$WANDB_DIR"

# The shared trainer resumes checkpoint.pt automatically and verifies experiment
# provenance. Completion occurs only after a successful final saved checkpoint.
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TRAIN_GPUS" \
  "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.py" \
  --manifest "$SFT_MANIFEST" --init_from "$SFT_INIT" --workdir "$SFT_WORKDIR" \
  --fsdp "$TRAIN_GPUS" --tp 1 --batch_size 32 --grad_accum 1 \
  "${step_args[@]}" --no_compile "${completion_args[@]}" "$@" \
  2>&1 | tee -a "$SFT_WORKDIR/train.log"

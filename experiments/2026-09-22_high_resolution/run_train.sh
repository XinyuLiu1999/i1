#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESOLUTION=${RESOLUTION:-2048}
case "$RESOLUTION" in
  1536|2048) ;;
  *) echo "RESOLUTION must be 1536 or 2048" >&2; exit 2 ;;
esac
: "${SFT_MANIFEST:?Set SFT_MANIFEST to the matching high-resolution cache.jsonl or original-image manifest}"
: "${SFT_WORKDIR:?Set a separate SFT_WORKDIR for this resolution experiment}"
SFT_INIT=${SFT_INIT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}
TRAIN_PYTHON=${TRAIN_PYTHON:-/root/miniconda3/envs/i1_sft/bin/python}
TRAIN_GPUS=${TRAIN_GPUS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
mkdir -p "$SFT_WORKDIR"
SFT_WORKDIR=$(cd -- "$SFT_WORKDIR" && pwd)
export WANDB_DIR=${WANDB_DIR:-$SFT_WORKDIR/wandb}
mkdir -p "$WANDB_DIR"
export PYTHONPATH="$EXPERIMENT_DIR/../../torch_train${PYTHONPATH:+:$PYTHONPATH}"
step_args=()
if [[ -n ${TRAIN_STEPS:-} ]]; then
  step_args=(--total_steps "$TRAIN_STEPS")
fi
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TRAIN_GPUS" \
  -m training.main --config "$EXPERIMENT_DIR/config_${RESOLUTION}.py" \
  --manifest "$SFT_MANIFEST" --init_from "$SFT_INIT" --workdir "$SFT_WORKDIR" \
  --fsdp "$TRAIN_GPUS" --tp 1 --batch_size 32 --grad_accum 4 --no_compile \
  --completion-config "" "${step_args[@]}" "$@" \
  2>&1 | tee -a "$SFT_WORKDIR/train.log"

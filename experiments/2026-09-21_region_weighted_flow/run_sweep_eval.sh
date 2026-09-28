#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_PYTHON=${TRAIN_PYTHON:-/root/miniconda3/envs/i1_sft/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$TRAIN_PYTHON" -u "$EXPERIMENT_DIR/evaluate_sweep.py" "$@"

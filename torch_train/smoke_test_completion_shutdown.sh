#!/usr/bin/env bash
# Run a few production-shaped SFT steps, save a checkpoint, and ask the GPU task
# manager to shut down the VM. This test is intentionally destructive: run it
# only after stopping any valuable training process on this host.

set -euo pipefail

if [[ "${CONFIRM_GPU_SHUTDOWN:-}" != "YES" ]]; then
  echo "Refusing to run: this test powers off the configured GPU VM." >&2
  echo "Re-run with CONFIRM_GPU_SHUTDOWN=YES after stopping production training." >&2
  exit 2
fi

DENSE_PROJECT="${DENSE_PROJECT:-/cephfs/liuxinyu/DenseText-Project}"
SFT_MANIFEST="${SFT_MANIFEST:-$DENSE_PROJECT/artifacts/textdense_primary_english_captioned_v4_precompute/cache_1024/cache.jsonl}"
SFT_INIT="${SFT_INIT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}"
GPU_COMPLETION_CONFIG="${GPU_COMPLETION_CONFIG:-$DENSE_PROJECT/local_captioning/completion_config.json}"
SFT_SMOKE_CONFIG="${SFT_SMOKE_CONFIG:-$DENSE_PROJECT/i1/torch_train/configs/sft_1024_shutdown_smoke.py}"
SFT_SMOKE_ROOT="${SFT_SMOKE_ROOT:-$DENSE_PROJECT/artifacts/sft_completion_shutdown_smoke}"
SFT_SMOKE_STEPS="${SFT_SMOKE_STEPS:-3}"
SFT_PYTHON="${SFT_PYTHON:-/root/miniconda3/envs/i1_sft/bin/python}"
SFT_TORCHRUN="${SFT_TORCHRUN:-/root/miniconda3/envs/i1_sft/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES GPU_COMPLETION_CONFIG

if ! [[ "$SFT_SMOKE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT_SMOKE_STEPS must be a positive integer; got $SFT_SMOKE_STEPS" >&2
  exit 2
fi

for required_file in \
  "$SFT_MANIFEST" \
  "$SFT_INIT" \
  "$GPU_COMPLETION_CONFIG" \
  "$SFT_SMOKE_CONFIG" \
  "$SFT_PYTHON" \
  "$SFT_TORCHRUN"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file does not exist: $required_file" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; run this test on the eight-GPU training VM." >&2
  exit 2
fi

active_compute="$(nvidia-smi --query-compute-apps=pid,process_name \
  --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_compute" ]]; then
  echo "Refusing to overlap the destructive smoke test with active GPU processes:" >&2
  echo "$active_compute" >&2
  exit 2
fi

export PYTHONPATH="$DENSE_PROJECT/i1/torch_train${PYTHONPATH:+:$PYTHONPATH}"
"$SFT_PYTHON" - <<'PY'
import os
import torch
from training.completion import load_completion_config

config = load_completion_config(os.environ["GPU_COMPLETION_CONFIG"])
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print(f"Validated shutdown credentials for {len(config['vmids'])} VM(s); 8 GPUs visible.")
PY

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SFT_SMOKE_WORKDIR="$SFT_SMOKE_ROOT/$run_id"
if [[ -e "$SFT_SMOKE_WORKDIR" ]]; then
  echo "Refusing to reuse smoke workdir: $SFT_SMOKE_WORKDIR" >&2
  exit 2
fi
mkdir -p "$SFT_SMOKE_WORKDIR"
printf '%s\n' "$SFT_SMOKE_WORKDIR" > "$SFT_SMOKE_ROOT/latest_run.txt"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/cephfs/liuxinyu/.cache/data_juicer/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Starting $SFT_SMOKE_STEPS SFT steps in $SFT_SMOKE_WORKDIR"
echo "A successful final checkpoint will be followed by the real VM shutdown request."
cd "$DENSE_PROJECT/i1/torch_train"
set -o pipefail
"$SFT_TORCHRUN" \
  --standalone \
  --nproc_per_node=8 \
  -m training.main \
  --config "$SFT_SMOKE_CONFIG" \
  --manifest "$SFT_MANIFEST" \
  --init_from "$SFT_INIT" \
  --workdir "$SFT_SMOKE_WORKDIR" \
  --fsdp 8 \
  --batch_size 32 \
  --grad_accum 4 \
  --total_steps "$SFT_SMOKE_STEPS" \
  --ckpt_steps "$SFT_SMOKE_STEPS" \
  --log_every 1 \
  --no_compile \
  --completion-config "$GPU_COMPLETION_CONFIG" \
  2>&1 | tee "$SFT_SMOKE_WORKDIR/train.log"

echo "The task manager acknowledged completion; VM shutdown may be asynchronous."

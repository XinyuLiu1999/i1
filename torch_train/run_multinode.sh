#!/usr/bin/env bash
# Run once per node. Platform WORLD_SIZE/RANK mean node count/node rank here;
# torchrun replaces them with process count/global rank in its workers.
set -euo pipefail

TRAIN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$TRAIN_DIR"
# Explicit TRAIN_PYTHON bypasses Conda, e.g. for another host or CPU checks.
CONDA_SH=${CONDA_SH:-/user/lxy8802/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-i1_sft}
if [[ -z ${TRAIN_PYTHON:-} ]]; then
  if [[ ! -f $CONDA_SH ]]; then
    echo "ERROR: Conda initialization script not found: $CONDA_SH" >&2
    exit 2
  fi
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  TRAIN_PYTHON=python
fi
# Pre-downloaded model cache. Model loading is offline by default; W&B is separate.
export HF_HOME="${HF_HOME:-/user/lxy8802/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/user/lxy8802/.cache/data_juicer/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE
# W&B: an environment key wins; otherwise read the `export WANDB_API_KEY=...`
# line from WANDB_KEY_FILE. That .bashrc returns early in non-interactive
# shells, so parse the one line instead of sourcing it. An empty key still
# allows credentials previously saved by `wandb login`.
WANDB_KEY_FILE=${WANDB_KEY_FILE:-/user/lxy8802/.bashrc}
if [[ -z ${WANDB_API_KEY:-} && -r $WANDB_KEY_FILE ]]; then
  WANDB_API_KEY=$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?WANDB_API_KEY=[\"']?([^\"'[:space:]]*)[\"']?.*/\2/p" \
    "$WANDB_KEY_FILE" | tail -n 1)
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-DenseText-SFT}"
export WANDB_MODE="${WANDB_MODE:-online}"
dry_run=0
check_communication=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --check-communication) check_communication=1; shift ;;
    *) break ;;
  esac
done
fail() { echo "ERROR: $*" >&2; exit 2; }
integer() {
  [[ $2 =~ ^(0|[1-9][0-9]*)$ && ${#2} -le 9 ]] || fail "$1 must be a nonnegative integer: $2"
}
positive() { integer "$1" "$2"; (( $2 > 0 )) || fail "$1 must be positive"; }

NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-}}
positive NNODES "$NNODES"
if (( NNODES > 1 )); then
  [[ -n $NODE_RANK ]] || fail "Set platform RANK or NODE_RANK for a multi-node job"
  [[ -n ${MASTER_ADDR:-} ]] || fail "MASTER_ADDR is required for a multi-node job"
fi
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
integer NODE_RANK "$NODE_RANK"
(( NODE_RANK < NNODES )) || fail "NODE_RANK must be less than NNODES"
positive MASTER_PORT "$MASTER_PORT"
(( MASTER_PORT <= 65535 )) || fail "MASTER_PORT must be <= 65535"

CHECK_BACKEND=${CHECK_BACKEND:-nccl}
[[ $CHECK_BACKEND == nccl || $CHECK_BACKEND == gloo ]] || fail "CHECK_BACKEND must be nccl or gloo"
if [[ -z ${GPUS_PER_NODE:-} ]]; then
  (( dry_run == 0 )) || fail "Set GPUS_PER_NODE explicitly for --dry-run"
  GPUS_PER_NODE=$("$TRAIN_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
fi
positive GPUS_PER_NODE "$GPUS_PER_NODE"
TOTAL_PROCESSES=$(( NNODES * GPUS_PER_NODE ))

launch=("$TRAIN_PYTHON" -m torch.distributed.run
  --nnodes "$NNODES" --nproc_per_node "$GPUS_PER_NODE"
  --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"
  --max_restarts 0)

if (( check_communication )); then
  launch+=(-m training.distributed_smoke --backend "$CHECK_BACKEND" "$@")
else
  # These are owned by the launcher so its topology/batch checks remain valid.
  for arg in "$@"; do
    case "$arg" in
      --config|--config=*|--workdir|--workdir=*|--manifest|--manifest=*|--init_from|--init_from=*|--resume|--resume=*|--fsdp|--fsdp=*|--tp|--tp=*|--batch_size|--batch_size=*|--grad_accum|--grad_accum=*|--completion-config|--completion-config=*)
        fail "Set $arg through the documented launcher environment variables instead" ;;
    esac
  done
  SFT_CONFIG=${SFT_CONFIG:-$TRAIN_DIR/configs/sft_1024.py}
  if [[ -z ${SFT_RESUME:-} ]]; then
    SFT_INIT=${SFT_INIT:-$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt}
  fi
  [[ -n ${SFT_MANIFEST:-} ]] || fail "Set SFT_MANIFEST to the shared dataset manifest"
  [[ -n ${SFT_WORKDIR:-} ]] || fail "Set SFT_WORKDIR to the same shared output directory on every node"
  [[ -z ${SFT_INIT:-} || -z ${SFT_RESUME:-} ]] || fail "Set only one of SFT_INIT and SFT_RESUME"
  TP_SIZE=${TP_SIZE:-1}
  positive TP_SIZE "$TP_SIZE"
  (( GPUS_PER_NODE % TP_SIZE == 0 )) || fail "TP_SIZE must divide GPUS_PER_NODE (TP stays within a node)"
  DP_WORLD=$(( TOTAL_PROCESSES / TP_SIZE ))
  FSDP_SIZE=${FSDP_SIZE:-$DP_WORLD}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
  GRAD_ACCUM=${GRAD_ACCUM:-4}
  positive FSDP_SIZE "$FSDP_SIZE"
  positive GLOBAL_BATCH_SIZE "$GLOBAL_BATCH_SIZE"
  positive GRAD_ACCUM "$GRAD_ACCUM"
  (( DP_WORLD % FSDP_SIZE == 0 )) || fail "FSDP_SIZE must divide total data-parallel ranks ($DP_WORLD)"
  (( GLOBAL_BATCH_SIZE % (DP_WORLD * GRAD_ACCUM) == 0 )) ||
    fail "GLOBAL_BATCH_SIZE must be divisible by DP_WORLD * GRAD_ACCUM ($(( DP_WORLD * GRAD_ACCUM )))"
  echo "data_parallel_ranks=$DP_WORLD fsdp=$FSDP_SIZE tp=$TP_SIZE global_batch=$GLOBAL_BATCH_SIZE microbatch_per_gpu=$(( GLOBAL_BATCH_SIZE / DP_WORLD / GRAD_ACCUM )) grad_accum=$GRAD_ACCUM"
  launch+=(-m training.main --config "$SFT_CONFIG" --manifest "$SFT_MANIFEST"
    --workdir "$SFT_WORKDIR" --fsdp "$FSDP_SIZE" --tp "$TP_SIZE"
    --batch_size "$GLOBAL_BATCH_SIZE" --grad_accum "$GRAD_ACCUM"
    --no_compile --completion-config "")
  if [[ -n ${SFT_RESUME:-} ]]; then
    launch+=(--resume "$SFT_RESUME")
  else
    launch+=(--init_from "$SFT_INIT")
  fi
  launch+=("$@")
  if (( dry_run == 0 )); then
    [[ -f $SFT_CONFIG ]] || fail "Config not found: $SFT_CONFIG"
    [[ -e $SFT_MANIFEST ]] || fail "Dataset manifest not found: $SFT_MANIFEST"
    [[ -f ${SFT_RESUME:-$SFT_INIT} ]] || fail "Initialization/resume checkpoint not found"
    # The trainer prioritizes workdir/checkpoint.pt. Require an explicit resume
    # of that file instead of accidentally continuing a different run.
    if [[ -e $SFT_WORKDIR/checkpoint.pt ]]; then
      [[ -n ${SFT_RESUME:-} && $SFT_RESUME -ef $SFT_WORKDIR/checkpoint.pt ]] ||
        fail "Workdir contains checkpoint.pt; use a new directory or set SFT_RESUME to that file"
    fi
  fi
fi

echo "node_rank=$NODE_RANK nodes=$NNODES processes_per_node=$GPUS_PER_NODE total_processes=$TOTAL_PROCESSES master=$MASTER_ADDR:$MASTER_PORT"
printf 'Command:'
printf ' %q' "${launch[@]}"
printf '\n'
if (( dry_run )); then
  exit 0
fi
if (( !check_communication )) || [[ $CHECK_BACKEND == nccl ]]; then
  visible=$("$TRAIN_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
  (( visible >= GPUS_PER_NODE )) || fail "Requested $GPUS_PER_NODE GPUs but only $visible are visible"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONPATH="$TRAIN_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${SFT_WORKDIR:-} ]]; then
  mkdir -p "$SFT_WORKDIR/logs"
  export WANDB_DIR=${WANDB_DIR:-$SFT_WORKDIR/wandb}
  mkdir -p "$WANDB_DIR"
  exec > >(tee -a "$SFT_WORKDIR/logs/node_${NODE_RANK}.log") 2>&1
fi
if (( !check_communication && NODE_RANK == 0 )) && [[ $WANDB_MODE == online ]]; then
  echo "Authenticating W&B on node 0..."
  # Read WANDB_API_KEY (or saved credentials) without putting the key in argv.
  # No interactive prompt in a platform job; fail before starting GPU workers.
  "$TRAIN_PYTHON" -m wandb login --verify </dev/null ||
    fail "W&B login failed; set WANDB_API_KEY or check saved credentials and network access"
fi
echo "Launching node $NODE_RANK on $(hostname) at $(date -Is)"
# Only torchrun should define process ranks. Do not leak platform node ranks.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
exec "${launch[@]}"

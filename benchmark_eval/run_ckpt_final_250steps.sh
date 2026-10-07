#!/usr/bin/env bash
set -euo pipefail

# No arguments. Generate all benchmarks in order, one worker per visible GPU.
# Images, inputs, manifests, and logs live under each benchmark's
# ckpt-final-250steps directory. Reruns resume with the same GPU count.
if (( $# != 0 )); then
    echo "This script takes no arguments." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
I1_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/user/lxy8802/miniforge3/envs/i1_sft/bin/python"
CHECKPOINT="/backup/user/lxy8802/i1/densetext_sft_1024_run001/checkpoint.pt-000132658"
# Match the existing DenseText runner's sibling-repository layout.
BIZGENEVAL_DATA="$I1_ROOT/../BizGenEval/assets/bizgeneval.jsonl"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing i1_sft Python: $PYTHON_BIN" >&2
    exit 1
fi
for required_file in \
    "$CHECKPOINT" \
    "$BIZGENEVAL_DATA" \
    "$I1_ROOT/jax/inference/prompts/longtext_complex_rewrite.jsonl" \
    "$I1_ROOT/jax/inference/prompts/CVTG-2K_complex_rewrite.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done

cd -- "$I1_ROOT"
# Ignore stale GPU_IDS overrides; respect the scheduler's CUDA visibility mask.
GPU_LIST="$("$PYTHON_BIN" - <<'PY'
from benchmark_eval.text_benchmarks import check_cuda, gpu_ids
import sys
ids = gpu_ids(None, sys.executable)
check_cuda(ids, sys.executable)
print(",".join(ids))
PY
)"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --stage generate
    --gpu-ids "$GPU_LIST"
    --limit 0
    --resolution 1024
    --seed 42
    --num-steps 250
    --caption-overflow truncate
    --cfg-scale 12.0
    --cfg-rescale 1.0
    --inference-timestep-shift 0.3
    --diffusion-batch-size 1
    --vae-batch-size 1
)

for benchmark in bizgeneval longtext cvtg-2k; do
    mkdir -p "$SCRIPT_DIR/$benchmark/ckpt-final-250steps"
done

echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPU_LIST"
echo "[1/3] Generating BizGenEval (official prompts, 1024 aspect-ratio buckets)"
"$PYTHON_BIN" "$SCRIPT_DIR/bizgeneval/densetext_eval.py" \
    "${COMMON_ARGS[@]}" \
    --data-path "$BIZGENEVAL_DATA" \
    --native-text-context \
    --output-root "$SCRIPT_DIR/bizgeneval/ckpt-final-250steps"

stage=2
for benchmark in longtext cvtg-2k; do
    echo "[$stage/3] Generating $benchmark (complex_rewrite)"
    # Omitting --text-num-tokens preserves text_num_tokens=null.
    "$PYTHON_BIN" "$SCRIPT_DIR/text_benchmarks.py" \
        "${COMMON_ARGS[@]}" \
        --benchmark "$benchmark" \
        --prompt-variant complex_rewrite \
        --output-root "$SCRIPT_DIR/$benchmark/ckpt-final-250steps"
    stage=$((stage + 1))
done

echo "All three benchmarks completed: benchmark_eval/{bizgeneval,longtext,cvtg-2k}/ckpt-final-250steps"

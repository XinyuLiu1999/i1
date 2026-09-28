#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
I1_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$I1_ROOT/.." && pwd)"

BIZGENEVAL_ROOT="${BIZGENEVAL_ROOT:-$PROJECT_ROOT/../T2IBenchs/BizGenEval}"
DATA_PATH="${DATA_PATH:-$BIZGENEVAL_ROOT/assets/bizgeneval.jsonl}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$PROJECT_ROOT/artifacts/sft_1024_full_20260917_094936/checkpoint.pt-000006262}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/bizgeneval_sft_context_comparison_107}"

DEFAULT_PYTHON="/root/miniconda3/envs/i1_sft/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${GENERATION_PYTHON:-$DEFAULT_PYTHON}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_LAUNCH_DELAY="${GPU_LAUNCH_DELAY:-10}"
PROMPT_COUNT="${PROMPT_COUNT:-107}"
TOKEN_THRESHOLD="${TOKEN_THRESHOLD:-1024}"
SELECTION="${SELECTION:-dataset-order}"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-1}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-1}"
CFG_SCALE="${CFG_SCALE:-12}"
CFG_RESCALE="${CFG_RESCALE:-1.0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

for required_file in "$DATA_PATH" "$SFT_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done
for positive_integer in "$PROMPT_COUNT" "$TOKEN_THRESHOLD" "$NUM_STEPS" "$DIFFUSION_BATCH_SIZE" "$VAE_BATCH_SIZE"; do
    if ! [[ "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer, got: $positive_integer" >&2
        exit 1
    fi
done
if ! [[ "$GPU_LAUNCH_DELAY" =~ ^[0-9]+$ ]]; then
    echo "GPU_LAUNCH_DELAY must be a non-negative integer" >&2
    exit 1
fi
case "$SELECTION" in
    dataset-order|longest) ;;
    *) echo "SELECTION must be dataset-order or longest" >&2; exit 1 ;;
esac
case "$PREPARE_ONLY" in
    0|1) ;;
    *) echo "PREPARE_ONLY must be 0 or 1" >&2; exit 1 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
        echo "GPU_IDS must contain unique non-negative integers; got: $GPU_IDS" >&2
        exit 1
    fi
    SEEN_GPUS[$gpu_id]=1
done
if (( ${#GPU_ARRAY[@]} == 0 )); then
    echo "GPU_IDS cannot be empty" >&2
    exit 1
fi

INPUT_DIR="$OUTPUT_ROOT/inputs"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_context_comparison.py" \
    --input "$DATA_PATH" \
    --output-dir "$INPUT_DIR" \
    --count "$PROMPT_COUNT" \
    --threshold "$TOKEN_THRESHOLD" \
    --selection "$SELECTION"

PREPARED_DATA="$INPUT_DIR/bizgeneval_i1.jsonl"
OUTPUT_NAMES="$INPUT_DIR/output_names.txt"
ALL_TOKENS_CONTEXT="$(tr -d '[:space:]' < "$INPUT_DIR/max_text_tokens.txt")"
if ! [[ "$ALL_TOKENS_CONTEXT" =~ ^[1-9][0-9]*$ ]] || (( ALL_TOKENS_CONTEXT <= TOKEN_THRESHOLD )); then
    echo "Invalid all-token context length: $ALL_TOKENS_CONTEXT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
{
    echo "checkpoint=$SFT_CHECKPOINT"
    echo "data=$DATA_PATH"
    echo "prompt_count=$PROMPT_COUNT"
    echo "selection=$SELECTION"
    echo "truncate_context=$TOKEN_THRESHOLD"
    echo "all_tokens_context=$ALL_TOKENS_CONTEXT"
    echo "num_steps=$NUM_STEPS"
    echo "seed=$SEED"
    echo "gpu_ids=$GPU_IDS"
} > "$OUTPUT_ROOT/run_config.txt"

echo "Comparison contexts: truncate=$TOKEN_THRESHOLD, all_tokens=$ALL_TOKENS_CONTEXT"
if (( PREPARE_ONLY == 1 )); then
    echo "Preparation complete; PREPARE_ONLY=1, so generation was not started."
    exit 0
fi

ACTIVE_PIDS=()
cleanup_children() {
    if (( ${#ACTIVE_PIDS[@]} > 0 )); then
        echo "Stopping ${#ACTIVE_PIDS[@]} generation worker(s)..." >&2
        kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
        wait "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup_children INT TERM

run_arm() {
    local label="$1"
    local text_context="$2"
    local overflow="$3"
    local image_dir="$OUTPUT_ROOT/images/$label"
    local log_dir="$OUTPUT_ROOT/logs/$label"
    mkdir -p "$image_dir" "$log_dir"

    local worker_count="${#GPU_ARRAY[@]}"
    if (( worker_count > PROMPT_COUNT )); then
        worker_count="$PROMPT_COUNT"
    fi
    local base_count=$((PROMPT_COUNT / worker_count))
    local extra_count=$((PROMPT_COUNT % worker_count))
    local -a stage_pids=()
    local -a stage_labels=()
    local worker_idx prompt_start prompt_count prompt_end gpu_id worker_seed log_file pid

    echo "Generating $label: $PROMPT_COUNT prompts on $worker_count GPU(s), context=$text_context"
    for ((worker_idx = 0; worker_idx < worker_count; worker_idx++)); do
        prompt_count="$base_count"
        if (( worker_idx < extra_count )); then
            prompt_count=$((prompt_count + 1))
            prompt_start=$((worker_idx * prompt_count))
        else
            prompt_start=$((extra_count * (base_count + 1) + (worker_idx - extra_count) * base_count))
        fi
        prompt_end=$((prompt_start + prompt_count))
        gpu_id="${GPU_ARRAY[$worker_idx]}"
        worker_seed=$((SEED + worker_idx))
        log_file="$log_dir/gpu${gpu_id}.log"

        echo "  GPU $gpu_id: prompts [$prompt_start, $prompt_end), log: $log_file"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$I1_ROOT/torch_inference/generate.py" \
            --checkpoint "$SFT_CHECKPOINT" \
            --prompts-jsonl "$PREPARED_DATA" \
            --jsonl-height-key _i1_height \
            --jsonl-width-key _i1_width \
            --output-names-file "$OUTPUT_NAMES" \
            --start-idx "$prompt_start" \
            --end-idx "$prompt_end" \
            --rewrite-prompt false \
            --skip-existing \
            --resolution 1024 \
            --caption-overflow "$overflow" \
            --text-num-tokens "$text_context" \
            --num-steps "$NUM_STEPS" \
            --seed "$worker_seed" \
            --diffusion-batch-size "$DIFFUSION_BATCH_SIZE" \
            --vae-batch-size "$VAE_BATCH_SIZE" \
            --cfg-scale "$CFG_SCALE" \
            --cfg-rescale "$CFG_RESCALE" \
            --outdir "$image_dir" > "$log_file" 2>&1 &
        pid=$!
        stage_pids+=("$pid")
        stage_labels+=("GPU $gpu_id")
        ACTIVE_PIDS+=("$pid")
        if (( GPU_LAUNCH_DELAY > 0 && worker_idx + 1 < worker_count )); then
            sleep "$GPU_LAUNCH_DELAY"
        fi
    done

    local failed=0
    for worker_idx in "${!stage_pids[@]}"; do
        if ! wait "${stage_pids[$worker_idx]}"; then
            echo "$label ${stage_labels[$worker_idx]} failed; see $log_dir" >&2
            failed=1
        fi
    done
    ACTIVE_PIDS=()
    if (( failed )); then
        return 1
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/validate_images.py" \
        --names "$OUTPUT_NAMES" \
        --image-dir "$image_dir" \
        --data "$PREPARED_DATA" \
        --geometry native_buckets
}

# The first arm intentionally truncates. The second uses overflow=error so it
# fails rather than silently dropping a token if the computed context is wrong.
run_arm "truncate_1024" "$TOKEN_THRESHOLD" "truncate"
run_arm "all_tokens" "$ALL_TOKENS_CONTEXT" "error"

"$PYTHON_BIN" "$SCRIPT_DIR/build_context_comparison.py" --output-root "$OUTPUT_ROOT"
echo "Comparison complete: $OUTPUT_ROOT/comparison.html"

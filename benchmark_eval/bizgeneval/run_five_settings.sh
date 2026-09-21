#!/usr/bin/env bash
set -euo pipefail

# Run all BizGenEval prompts under five checkpoint/context configurations:
#   1. Starting checkpoint, truncated to its native 256-token context.
#   2. Starting checkpoint, extended to the longest prompt (no truncation).
#   3. Full-data SFT checkpoint, truncated to its native 1024-token context.
#   4. DenseText-captioned SFT checkpoint, truncated to 1024 tokens.
#   5. DenseText-captioned SFT checkpoint, extended to the longest prompt.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
I1_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$I1_ROOT/.." && pwd)"

BIZGENEVAL_ROOT="${BIZGENEVAL_ROOT:-$PROJECT_ROOT/../BizGenEval}"
DATA_PATH="${DATA_PATH:-$BIZGENEVAL_ROOT/assets/bizgeneval.jsonl}"
START_CHECKPOINT="${START_CHECKPOINT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}"
FULL_SFT_CHECKPOINT="${FULL_SFT_CHECKPOINT:-$PROJECT_ROOT/artifacts/sft_1024_full_20260917_094936/checkpoint.pt-000006262}"
CAPTIONED_SFT_CHECKPOINT="${CAPTIONED_SFT_CHECKPOINT:-$PROJECT_ROOT/artifacts/sft_densetext_captioned_v4_1024/checkpoint.pt-000006245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/bizgeneval_five_settings}"

DEFAULT_PYTHON="/root/miniconda3/envs/i1_sft/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${GENERATION_PYTHON:-$DEFAULT_PYTHON}"

TOKENIZER="${TOKENIZER:-google/t5gemma-2b-2b-ul2-it}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_LAUNCH_DELAY="${GPU_LAUNCH_DELAY:-10}"
LIMIT="${LIMIT:-0}"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-1}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-1}"
CFG_SCALE="${CFG_SCALE:-12}"
CFG_RESCALE="${CFG_RESCALE:-1.0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

for required_file in \
    "$DATA_PATH" \
    "$START_CHECKPOINT" \
    "$FULL_SFT_CHECKPOINT" \
    "$CAPTIONED_SFT_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done

for integer_value in "$GPU_LAUNCH_DELAY" "$LIMIT"; do
    if ! [[ "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "Expected a non-negative integer, got: $integer_value" >&2
        exit 1
    fi
done
for positive_integer in "$NUM_STEPS" "$DIFFUSION_BATCH_SIZE" "$VAE_BATCH_SIZE"; do
    if ! [[ "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer, got: $positive_integer" >&2
        exit 1
    fi
done
case "$PREPARE_ONLY" in
    0|1) ;;
    *) echo "PREPARE_ONLY must be 0 or 1; got: $PREPARE_ONLY" >&2; exit 1 ;;
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
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_inputs.py" \
    --input "$DATA_PATH" \
    --output-dir "$INPUT_DIR" \
    --limit "$LIMIT" \
    --tokenizer "$TOKENIZER"

PREPARED_DATA="$INPUT_DIR/bizgeneval_i1.jsonl"
OUTPUT_NAMES="$INPUT_DIR/output_names.txt"
MAX_PROMPT_TOKENS="$(tr -d '[:space:]' < "$INPUT_DIR/max_text_tokens.txt")"
NUM_PROMPTS="$(wc -l < "$OUTPUT_NAMES")"
if ! [[ "$MAX_PROMPT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid maximum prompt length: $MAX_PROMPT_TOKENS" >&2
    exit 1
fi
ALL_TOKENS_CONTEXT="$MAX_PROMPT_TOKENS"
# The SFT checkpoints cannot be loaded below their native 1024-token context.
if (( ALL_TOKENS_CONTEXT < 1024 )); then
    ALL_TOKENS_CONTEXT=1024
fi

mkdir -p "$OUTPUT_ROOT"
{
    echo "data=$DATA_PATH"
    echo "start_checkpoint=$START_CHECKPOINT"
    echo "full_sft_checkpoint=$FULL_SFT_CHECKPOINT"
    echo "captioned_sft_checkpoint=$CAPTIONED_SFT_CHECKPOINT"
    echo "prompt_count=$NUM_PROMPTS"
    echo "max_prompt_tokens=$MAX_PROMPT_TOKENS"
    echo "all_tokens_context=$ALL_TOKENS_CONTEXT"
    echo "num_steps=$NUM_STEPS"
    echo "seed=$SEED"
    echo "gpu_ids=$GPU_IDS"
} > "$OUTPUT_ROOT/run_config.txt"

echo "Prepared $NUM_PROMPTS prompts; maximum length is $MAX_PROMPT_TOKENS tokens."
echo "All-token model context is $ALL_TOKENS_CONTEXT."
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
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup_children EXIT

run_setting() {
    local label="$1"
    local checkpoint="$2"
    local text_context="$3"
    local overflow="$4"
    local geometry="$5"
    local image_dir="$OUTPUT_ROOT/images/$label"
    local log_dir="$OUTPUT_ROOT/logs/$label"
    local -a geometry_args=()

    case "$geometry" in
        square)
            geometry_args=(--height 1024 --width 1024)
            ;;
        native_buckets)
            geometry_args=(--jsonl-height-key _i1_height --jsonl-width-key _i1_width)
            ;;
        *)
            echo "Unknown geometry mode: $geometry" >&2
            return 1
            ;;
    esac
    mkdir -p "$image_dir" "$log_dir"

    local worker_count="${#GPU_ARRAY[@]}"
    if (( worker_count > NUM_PROMPTS )); then
        worker_count="$NUM_PROMPTS"
    fi
    local base_count=$((NUM_PROMPTS / worker_count))
    local extra_count=$((NUM_PROMPTS % worker_count))
    local -a stage_pids=()
    local -a stage_labels=()
    local worker_idx prompt_start prompt_count prompt_end gpu_id worker_seed log_file pid

    echo
    echo "Generating $label: $NUM_PROMPTS prompts on $worker_count GPU(s), context=$text_context"
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
            --checkpoint "$checkpoint" \
            --prompts-jsonl "$PREPARED_DATA" \
            "${geometry_args[@]}" \
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
            --device "$DEVICE" \
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
        --geometry "$geometry"
}

# Load the newly trained DenseText-captioned checkpoint first so both of its
# requested settings finish before inference starts on either comparison model.
run_setting "04_captioned_sft_6245_truncate_1024" "$CAPTIONED_SFT_CHECKPOINT" 1024 "truncate" "native_buckets"
run_setting "05_captioned_sft_6245_all_tokens" "$CAPTIONED_SFT_CHECKPOINT" "$ALL_TOKENS_CONTEXT" "error" "native_buckets"
run_setting "01_start_truncate_256" "$START_CHECKPOINT" 256 "truncate" "square"
run_setting "02_start_all_tokens" "$START_CHECKPOINT" "$ALL_TOKENS_CONTEXT" "error" "square"
run_setting "03_full_sft_6262_truncate_1024" "$FULL_SFT_CHECKPOINT" 1024 "truncate" "native_buckets"

echo
echo "Generation complete: $OUTPUT_ROOT/images"

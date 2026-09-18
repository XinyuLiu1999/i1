#!/usr/bin/env bash
set -euo pipefail

# Compare the pretrained initialization and the step-2500 SFT checkpoint on
# N prompts from CVTG-2K, LongText-Bench, and a category-stratified BizGenEval
# subset. By default, use the existing complex rewrites so both checkpoints
# see identical rewritten text. LongText generates four images per prompt.
# Dynamic text context keeps the starting checkpoint at its native 256 tokens
# for short prompts and uses 1024 only when needed; the SFT checkpoint is
# natively 1024 tokens.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

START_CHECKPOINT="${START_CHECKPOINT:-/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$PROJECT_ROOT/artifacts/sft_1024_full_20260917_094936/checkpoint.pt-000002500}"
DEFAULT_PYTHON="/root/miniconda3/envs/i1_sft/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

NUM_PROMPTS="${NUM_PROMPTS:-10}"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-1}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-4}"
CFG_SCALE="${CFG_SCALE:-12}"
CFG_RESCALE="${CFG_RESCALE:-1.0}"
TEXT_NUM_TOKENS="${TEXT_NUM_TOKENS:-1024}"
DYNAMIC_TEXT_CONTEXT="${DYNAMIC_TEXT_CONTEXT:-true}"
PROMPT_VARIANT="${PROMPT_VARIANT:-complex_rewrite}"
GPU_IDS="${GPU_IDS:-0}"
# Stagger large checkpoint loads to reduce simultaneous host-RAM and storage pressure.
GPU_LAUNCH_DELAY="${GPU_LAUNCH_DELAY:-10}"

CVTG_SOURCE="$PROJECT_ROOT/i1/jax/inference/prompts/CVTG-2K.json"
LONGTEXT_SOURCE="$PROJECT_ROOT/i1/benchmark_eval/longtext/text_prompts.jsonl"
LONGTEXT_GENERATOR_SOURCE="$PROJECT_ROOT/i1/jax/inference/prompts/longtext.jsonl"
BIZGENEVAL_SOURCE="${BIZGENEVAL_SOURCE:-/cephfs/liuxinyu/BizGenEval/assets/bizgeneval.jsonl}"

case "$PROMPT_VARIANT" in
    original)
        CVTG_PROMPT_SET="CVTG-2K"
        LONGTEXT_PROMPT_SET="longtext"
        ;;
    simple_rewrite)
        CVTG_PROMPT_SET="CVTG-2K_simple_rewrite"
        LONGTEXT_PROMPT_SET="longtext_simple_rewrite"
        ;;
    complex_rewrite)
        CVTG_PROMPT_SET="CVTG-2K_complex_rewrite"
        LONGTEXT_PROMPT_SET="longtext_complex_rewrite"
        ;;
    *)
        echo "PROMPT_VARIANT must be original, simple_rewrite, or complex_rewrite; got: $PROMPT_VARIANT" >&2
        exit 1
        ;;
esac

case "$DYNAMIC_TEXT_CONTEXT" in
    true)
        TEXT_CONTEXT_ARGS=(--dynamic-text-context)
        TEXT_CONTEXT_LABEL="dynamictext${TEXT_NUM_TOKENS}"
        ;;
    false)
        TEXT_CONTEXT_ARGS=()
        TEXT_CONTEXT_LABEL="text${TEXT_NUM_TOKENS}"
        ;;
    *)
        echo "DYNAMIC_TEXT_CONTEXT must be true or false, got: $DYNAMIC_TEXT_CONTEXT" >&2
        exit 1
        ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/sft_1024_full_20260917_094936/inference_compare_${NUM_PROMPTS}_${PROMPT_VARIANT}_${TEXT_CONTEXT_LABEL}}"
CVTG_SELECTED_SOURCE="$PROJECT_ROOT/i1/jax/inference/prompts/${CVTG_PROMPT_SET}.json"
LONGTEXT_SELECTED_SOURCE="$PROJECT_ROOT/i1/jax/inference/prompts/${LONGTEXT_PROMPT_SET}.jsonl"

for required_file in \
    "$START_CHECKPOINT" \
    "$SFT_CHECKPOINT" \
    "$CVTG_SOURCE" \
    "$LONGTEXT_SOURCE" \
    "$LONGTEXT_GENERATOR_SOURCE" \
    "$CVTG_SELECTED_SOURCE" \
    "$LONGTEXT_SELECTED_SOURCE" \
    "$BIZGENEVAL_SOURCE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done

if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROMPTS must be a positive integer, got: $NUM_PROMPTS" >&2
    exit 1
fi

if ! [[ "$TEXT_NUM_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEXT_NUM_TOKENS must be a positive integer, got: $TEXT_NUM_TOKENS" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "GPU_IDS must be a comma-separated list of non-negative integers; got: $GPU_IDS" >&2
        exit 1
    fi
    if [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
        echo "GPU_IDS contains duplicate GPU $gpu_id: $GPU_IDS" >&2
        exit 1
    fi
    SEEN_GPUS[$gpu_id]=1
done

if ! [[ "$GPU_LAUNCH_DELAY" =~ ^[0-9]+$ ]]; then
    echo "GPU_LAUNCH_DELAY must be a non-negative integer, got: $GPU_LAUNCH_DELAY" >&2
    exit 1
fi

if ! cmp -s "$LONGTEXT_SOURCE" "$LONGTEXT_GENERATOR_SOURCE"; then
    echo "LongText benchmark prompts differ from the inference prompt copy:" >&2
    echo "  $LONGTEXT_SOURCE" >&2
    echo "  $LONGTEXT_GENERATOR_SOURCE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
BIZGENEVAL_INPUT_DIR="$OUTPUT_ROOT/inputs/bizgeneval_stratified${NUM_PROMPTS}"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_bizgeneval_subset.py" \
    --input "$BIZGENEVAL_SOURCE" \
    --output-dir "$BIZGENEVAL_INPUT_DIR" \
    --limit "$NUM_PROMPTS" \
    --selection stratified
BIZGENEVAL_PROMPTS="$BIZGENEVAL_INPUT_DIR/metadata.jsonl"
BIZGENEVAL_OUTPUT_NAMES="$BIZGENEVAL_INPUT_DIR/output_names.txt"

ACTIVE_PIDS=()
cleanup_children() {
    if (( ${#ACTIVE_PIDS[@]} > 0 )); then
        echo "Stopping ${#ACTIVE_PIDS[@]} inference worker(s)..." >&2
        kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
        wait "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup_children INT TERM

run_generation_sharded() {
    local checkpoint_label="$1"
    local checkpoint_path="$2"
    local benchmark_dir="$3"
    local samples_per_prompt="$4"
    local prompt_mode="$5"
    local prompt_source="$6"
    local caption_overflow="$7"
    local output_names_file="${8:-}"

    local outdir="$OUTPUT_ROOT/$checkpoint_label/$benchmark_dir"
    local logdir="$OUTPUT_ROOT/logs/$checkpoint_label"
    mkdir -p "$outdir"
    mkdir -p "$logdir"

    echo
    echo "[$checkpoint_label] $benchmark_dir -> $outdir"
    local worker_count="${#GPU_ARRAY[@]}"
    if (( worker_count > NUM_PROMPTS )); then
        worker_count="$NUM_PROMPTS"
    fi

    local base_prompts=$((NUM_PROMPTS / worker_count))
    local extra_prompts=$((NUM_PROMPTS % worker_count))
    local worker_idx prompt_start prompt_count prompt_end start_idx end_idx gpu_id worker_seed log_file pid
    local -a stage_pids=()
    local -a stage_labels=()
    local -a prompt_args=()
    local -a output_name_args=()

    case "$prompt_mode" in
        prompt-set)
            prompt_args=(--prompt-set "$prompt_source")
            ;;
        prompts-file)
            prompt_args=(--prompts-file "$prompt_source")
            ;;
        prompts-jsonl)
            prompt_args=(--prompts-jsonl "$prompt_source")
            ;;
        *)
            echo "Unknown prompt mode: $prompt_mode" >&2
            return 1
            ;;
    esac
    if [[ -n "$output_names_file" ]]; then
        output_name_args=(--output-names-file "$output_names_file")
    fi

    for ((worker_idx = 0; worker_idx < worker_count; worker_idx++)); do
        prompt_count="$base_prompts"
        if (( worker_idx < extra_prompts )); then
            prompt_count=$((prompt_count + 1))
            prompt_start=$((worker_idx * prompt_count))
        else
            prompt_start=$((extra_prompts * (base_prompts + 1) + (worker_idx - extra_prompts) * base_prompts))
        fi
        prompt_end=$((prompt_start + prompt_count))
        start_idx=$((prompt_start * samples_per_prompt))
        end_idx=$((prompt_end * samples_per_prompt))
        gpu_id="${GPU_ARRAY[$worker_idx]}"
        worker_seed=$((SEED + worker_idx))
        log_file="$logdir/${benchmark_dir}_gpu${gpu_id}.log"

        echo "  GPU $gpu_id: prompts [$prompt_start, $prompt_end), sample indices [$start_idx, $end_idx), log: $log_file"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$SCRIPT_DIR/generate.py" \
            --checkpoint "$checkpoint_path" \
            --skip-existing \
            "${prompt_args[@]}" \
            "${output_name_args[@]}" \
            --start-idx "$start_idx" \
            --end-idx "$end_idx" \
            --rewrite-prompt false \
            --caption-overflow "$caption_overflow" \
            --text-num-tokens "$TEXT_NUM_TOKENS" \
            "${TEXT_CONTEXT_ARGS[@]}" \
            --resolution 1024 \
            --num-steps "$NUM_STEPS" \
            --seed "$worker_seed" \
            --device "$DEVICE" \
            --diffusion-batch-size "$DIFFUSION_BATCH_SIZE" \
            --vae-batch-size "$VAE_BATCH_SIZE" \
            --cfg-scale "$CFG_SCALE" \
            --cfg-rescale "$CFG_RESCALE" \
            --outdir "$outdir" >"$log_file" 2>&1 &
        pid=$!
        stage_pids+=("$pid")
        stage_labels+=("GPU $gpu_id")
        ACTIVE_PIDS+=("$pid")

        if (( GPU_LAUNCH_DELAY > 0 && worker_idx + 1 < worker_count )); then
            sleep "$GPU_LAUNCH_DELAY"
        fi
    done

    echo "  Launched ${#stage_pids[@]} workers. Monitor with:"
    echo "    tail -f $logdir/${benchmark_dir}_gpu*.log"

    local failed=0
    for worker_idx in "${!stage_pids[@]}"; do
        if ! wait "${stage_pids[$worker_idx]}"; then
            echo "  ${stage_labels[$worker_idx]} failed; see its log." >&2
            failed=1
        fi
    done
    ACTIVE_PIDS=()
    if (( failed )); then
        return 1
    fi
}

run_generation_sharded "starting_checkpoint" "$START_CHECKPOINT" "cvtg-2k" 1 "prompt-set" "$CVTG_PROMPT_SET" "error"
run_generation_sharded "starting_checkpoint" "$START_CHECKPOINT" "longtext" 4 "prompt-set" "$LONGTEXT_PROMPT_SET" "error"
run_generation_sharded "starting_checkpoint" "$START_CHECKPOINT" "bizgeneval" 1 "prompts-jsonl" "$BIZGENEVAL_PROMPTS" "error" "$BIZGENEVAL_OUTPUT_NAMES"
run_generation_sharded "step_000002500" "$SFT_CHECKPOINT" "cvtg-2k" 1 "prompt-set" "$CVTG_PROMPT_SET" "error"
run_generation_sharded "step_000002500" "$SFT_CHECKPOINT" "longtext" 4 "prompt-set" "$LONGTEXT_PROMPT_SET" "error"
run_generation_sharded "step_000002500" "$SFT_CHECKPOINT" "bizgeneval" 1 "prompts-jsonl" "$BIZGENEVAL_PROMPTS" "error" "$BIZGENEVAL_OUTPUT_NAMES"

echo
echo "Done. Comparison samples are in: $OUTPUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
I1_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$I1_ROOT/.." && pwd)"

BIZGENEVAL_ROOT="${BIZGENEVAL_ROOT:-/cephfs/liuxinyu/BizGenEval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/bizgeneval_start_vs_sft6262}"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-python}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:-$BIZGENEVAL_ROOT/config/evaluation_config.yaml}"
CHECKPOINT_SET="${CHECKPOINT_SET:-both}"

PREPARED_DATA="$OUTPUT_ROOT/inputs/bizgeneval_i1.jsonl"
OUTPUT_NAMES="$OUTPUT_ROOT/inputs/output_names.txt"
for required_file in "$PREPARED_DATA" "$OUTPUT_NAMES" "$EVALUATION_CONFIG"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        echo "Run run_generation.sh first (LIMIT and OUTPUT_ROOT must match)." >&2
        exit 1
    fi
done
case "$CHECKPOINT_SET" in
    both|starting|sft) ;;
    *) echo "CHECKPOINT_SET must be both, starting, or sft; got: $CHECKPOINT_SET" >&2; exit 1 ;;
esac
if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "Set GEMINI_API_KEY or GOOGLE_API_KEY before evaluation." >&2
    exit 1
fi

evaluate_checkpoint() {
    local label="$1"
    local geometry="$2"
    local image_dir="$OUTPUT_ROOT/images/$label"
    local result_dir="$OUTPUT_ROOT/eval_results/$label"
    local summary_dir="$OUTPUT_ROOT/summaries/$label"

    "$EVALUATION_PYTHON" "$SCRIPT_DIR/validate_images.py" \
        --names "$OUTPUT_NAMES" \
        --image-dir "$image_dir" \
        --data "$PREPARED_DATA" \
        --geometry "$geometry"
    (
        cd "$BIZGENEVAL_ROOT"
        "$EVALUATION_PYTHON" -m evaluation.image_evaluation \
            --data_path "$PREPARED_DATA" \
            --img_dir "$image_dir" \
            --save_dir "$result_dir" \
            --config_path "$EVALUATION_CONFIG"
        "$EVALUATION_PYTHON" "$SCRIPT_DIR/validate_results.py" \
            --data "$PREPARED_DATA" \
            --result-dir "$result_dir"
        "$EVALUATION_PYTHON" -m evaluation.summarize \
            --data_path "$PREPARED_DATA" \
            --result_dir "$result_dir" \
            --save_dir "$summary_dir"
    )
}

if [[ "$CHECKPOINT_SET" == "both" || "$CHECKPOINT_SET" == "starting" ]]; then
    evaluate_checkpoint "starting_checkpoint" "square"
fi
if [[ "$CHECKPOINT_SET" == "both" || "$CHECKPOINT_SET" == "sft" ]]; then
    evaluate_checkpoint "checkpoint_000006262" "native_buckets"
fi

if [[ "$CHECKPOINT_SET" == "both" ]]; then
    mkdir -p "$OUTPUT_ROOT/comparison"
    for grouping in domain dimension; do
        "$EVALUATION_PYTHON" "$SCRIPT_DIR/compare_summaries.py" \
            --starting "$OUTPUT_ROOT/summaries/starting_checkpoint/summary_by_${grouping}.csv" \
            --sft "$OUTPUT_ROOT/summaries/checkpoint_000006262/summary_by_${grouping}.csv" \
            --output "$OUTPUT_ROOT/comparison/summary_by_${grouping}.csv"
    done
fi

echo "Evaluation complete: $OUTPUT_ROOT/summaries"

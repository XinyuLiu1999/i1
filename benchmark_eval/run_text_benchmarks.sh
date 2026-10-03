#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/text_benchmarks.py" "$@"
fi
exec conda run --no-capture-output -n i1_sft python "$SCRIPT_DIR/text_benchmarks.py" "$@"

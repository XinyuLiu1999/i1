#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# All generation and evaluation subprocesses inherit this interpreter.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/densetext_eval.py" "$@"
fi
exec conda run --no-capture-output -n i1_sft python "$SCRIPT_DIR/densetext_eval.py" "$@"

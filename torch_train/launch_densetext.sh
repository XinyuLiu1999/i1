#!/usr/bin/env bash
# Platform start script (same on every node). Settings live in SFT_ENV_FILE;
# extra trainer flags are passed through, e.g. --total_steps 60000.
set -euo pipefail
source "${SFT_ENV_FILE:-/user/lxy8802/i1/torch_train/configs/densetext_1024.env}"
exec bash /user/lxy8802/i1/torch_train/run_multinode.sh "$@"

#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m matcher.eval.eval_runner

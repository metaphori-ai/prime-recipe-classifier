#!/usr/bin/env bash
# Evaluate a locally served mlx model (mlx_lm.server on :8080) on recipe-classify.
# Usage (from anywhere):  bash scripts/vf_run_mlx.sh <model-path-as-served> [n=439] [concurrency=4]
set -euo pipefail
MODEL="${1:?model path exactly as passed to mlx_lm.server --model}"; N="${2:-439}"; C="${3:-4}"
cd "$(dirname "$0")/../env"
exec uv run --env-file environments/recipe_classify/configs/mlx.env vf-run recipe-classify \
  -m "$MODEL" -n "$N" -r 1 -c "$C" @ environments/recipe_classify/configs/mlx.toml

#!/usr/bin/env bash
# Run the recipe-classify baseline against a local Ollama model.
# Usage (from anywhere):  bash scripts/vf_run_ollama.sh qwen3.5:2b [n=50] [concurrency=2] [config=ollama.toml]
set -euo pipefail
MODEL="${1:?model tag, e.g. qwen3.5:2b}"; N="${2:-50}"; C="${3:-2}"; CFG="${4:-ollama.toml}"
cd "$(dirname "$0")/../env"
exec uv run --env-file environments/recipe_classify/configs/ollama.env vf-run recipe-classify \
  -m "$MODEL" -n "$N" -r 1 -c "$C" @ "environments/recipe_classify/configs/$CFG"

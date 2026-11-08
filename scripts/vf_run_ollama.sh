#!/usr/bin/env bash
# Run the recipe-classify baseline against a local Ollama model. Usage (from env/):  bash ../scripts/vf_run_ollama.sh qwen3.5:2b 50
set -euo pipefail
MODEL="${1:?model tag, e.g. qwen3.5:2b}"; N="${2:-50}"; C="${3:-2}"
cd "$(dirname "$0")/../env"
exec uv run --env-file environments/recipe_classify/configs/ollama.env vf-run recipe-classify \
  -m "$MODEL" -n "$N" -r 1 -c "$C" @ environments/recipe_classify/configs/ollama.toml

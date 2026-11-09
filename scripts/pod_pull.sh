#!/usr/bin/env bash
# Pull everything worth keeping off the pod before it is destroyed. rsync, so safe to re-run; later runs only move deltas.
#   bash scripts/pod_pull.sh [host=pod] [weights|all]     (default: all)
# What comes down and where:
#   runs/sft-2b-sonnet-pod/      HF-layout SFT weights (B'', ~4 GB) — convert_dcp_to_bf16 output
#   pod-archive/outputs/         every prime-rl run dir: configs, logs, monitors (metrics.jsonl, traces/*.zst), broadcasts/
#                                (safetensors of the last two RL steps); DCP checkpoints/ and wandb/ caches are skipped (tens of GB)
#   pod-archive/models/          assembled model dirs (grpo-base-v2-step22; sft-2b-sonnet symlink resolved)
#   pod-archive/tools/           the patched convert_dcp_to_bf16.py + .orig (upstream bug: _tied_weights_keys None)
#   pod-archive/*.log            launcher logs
set -euo pipefail
HOST=${1:-pod}; WHAT=${2:-all}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/pod-archive"; mkdir -p "$DEST/outputs" "$DEST/models" "$DEST/tools" "$ROOT/runs"
RS=(rsync -az --progress --partial)   # macOS ships openrsync: no --info=

SFT_RUN=$(ssh "$HOST" 'ls -dt prime-rl/outputs/sft_sonnet--qwen3.5-2b--*/checkpoints/step_320/weights 2>/dev/null | head -1')
if [[ -n "$SFT_RUN" ]]; then
  echo "== SFT weights: $SFT_RUN -> runs/sft-2b-sonnet-pod/"
  "${RS[@]}" "$HOST:$SFT_RUN/" "$ROOT/runs/sft-2b-sonnet-pod/"
else
  echo "!! no sft weights dir found on $HOST"
fi
[[ "$WHAT" == "weights" ]] && exit 0

echo "== run outputs (no DCP checkpoints, no wandb)"
"${RS[@]}" --exclude 'checkpoints/' --exclude 'wandb/' "$HOST:prime-rl/outputs/" "$DEST/outputs/"
echo "== models (symlinks resolved)"
"${RS[@]}" --copy-links "$HOST:prime-intellect/models/" "$DEST/models/"
echo "== patched tool + logs"
"${RS[@]}" "$HOST:prime-rl/tools/convert_dcp_to_bf16.py" "$HOST:prime-rl/tools/convert_dcp_to_bf16.py.orig" "$DEST/tools/"
"${RS[@]}" "$HOST:~/*.log" "$DEST/" 2>/dev/null || true
echo "== done"; du -sh "$ROOT/runs/sft-2b-sonnet-pod" "$DEST"/* 2>/dev/null

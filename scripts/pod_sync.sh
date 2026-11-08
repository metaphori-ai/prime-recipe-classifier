#!/usr/bin/env bash
# Sync the environment package + frozen train set to the pod. Run FROM THE MAC:
#   bash scripts/pod_sync.sh <ssh-host-or-alias> [ssh-port]
# Excludes: .venv, outputs, raw/gold/silver data, .env, runs/. Only what training needs.
set -euo pipefail
HOST="${1:?ssh host, e.g. root@1.2.3.4 or a ~/.ssh/config alias}"; PORT="${2:-22}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
ssh -p "$PORT" "$HOST" 'mkdir -p ~/prime-intellect/env ~/prime-intellect/data/dataset ~/prime-intellect/data/sft'

# --- secrets: build ~/.pod-env on the Mac from local credentials, ship it, never echo it ---
# PRIME_API_KEY / PRIME_TEAM_ID from ~/.prime/config.json; HF_TOKEN from $HF_TOKEN or the HF CLI token cache.
TMP_ENV="$(mktemp)"; chmod 600 "$TMP_ENV"
python3 - "$TMP_ENV" <<'PY'
import json, os, sys, pathlib
out = sys.argv[1]; lines = []
cfg = pathlib.Path.home() / ".prime" / "config.json"
if cfg.exists():
    c = json.load(open(cfg))
    if c.get("api_key"): lines.append(f'export PRIME_API_KEY="{c["api_key"]}"')
    if c.get("team_id"): lines.append(f'export PRIME_TEAM_ID="{c["team_id"]}"')
hf = os.environ.get("HF_TOKEN")
if not hf:
    for cand in (pathlib.Path.home()/".cache"/"huggingface"/"token", pathlib.Path.home()/".huggingface"/"token"):
        if cand.exists(): hf = cand.read_text().strip(); break
if hf: lines.append(f'export HF_TOKEN="{hf}"')
wb = os.environ.get("WANDB_API_KEY")
if wb: lines.append(f'export WANDB_API_KEY="{wb}"')
open(out, "w").write("\n".join(lines) + "\n")
print(f"pod-env: {len(lines)} secrets staged ({', '.join(l.split('=')[0].split()[-1] for l in lines)})")
PY
scp -P "$PORT" -q "$TMP_ENV" "$HOST:.pod-env" && rm -f "$TMP_ENV"
ssh -p "$PORT" "$HOST" 'chmod 600 ~/.pod-env; grep -q "source ~/.pod-env" ~/.bashrc || echo "[ -f ~/.pod-env ] && source ~/.pod-env" >> ~/.bashrc'
rsync -avz -e "ssh -p $PORT" --delete \
  --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' \
  "$SRC/env/" "$HOST:~/prime-intellect/env/"
rsync -avz -e "ssh -p $PORT" \
  "$SRC/data/dataset/train.jsonl" "$SRC/data/dataset/eval.jsonl" "$SRC/data/dataset/manifest.json" \
  "$HOST:~/prime-intellect/data/dataset/"
rsync -avz -e "ssh -p $PORT" "$SRC/data/sft/train.jsonl" "$SRC/data/sft/valid.jsonl" "$HOST:~/prime-intellect/data/sft/"
rsync -avz -e "ssh -p $PORT" "$SRC/taxonomy.yaml" "$HOST:~/prime-intellect/taxonomy.yaml"
echo "synced. On the pod: cd ~/prime-rl && uv pip install -e ~/prime-intellect/env -e ~/prime-intellect/env/environments/recipe_classify"

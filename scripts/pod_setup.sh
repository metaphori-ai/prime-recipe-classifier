#!/usr/bin/env bash
# One-time setup on a fresh Prime pod (Ubuntu + CUDA). Run ON THE POD after SSH-ing in:
#   bash pod_setup.sh
# Then, FROM THE MAC, sync the project (see scripts/pod_sync.sh).
# RUNPOD (learned 2026-11-16, after a disk resize wiped the container disk and a finished SFT with it):
#   - /root (overlay) is fast but is ERASED on restart, resize or stop. /workspace is the network volume and survives.
#   - /workspace does not allow chmod: uv's cache, git clones and venvs fail there. Keep prime-rl, caches and in-progress runs
#     on /root, and copy every FINISHED artefact (exported weights, eval folders, logs) to /workspace/save right away.
#   - Before this script:  echo 'export UV_CACHE_DIR=/root/.cache/uv' >> ~/.bashrc; echo 'export HF_HOME=/root/.cache/huggingface' >> ~/.bashrc
#                          source ~/.bashrc; mkdir -p /workspace/save /root/prime-intellect /home/ubuntu
#                          ln -sfn /root/prime-intellect /home/ubuntu/prime-intellect
#   - Trainer checkpoints are ~25 GB each for a 2B model: size the disk for it or use --trainer.ckpt.keep-last 1.
#   - If the installer dies on a 503 from a wheel index: cd ~/prime-rl && uv sync --all-extras (do not re-run the installer).
set -euo pipefail
cd ~
# 0. Driver check FIRST. prime-rl installs a CUDA-13 stack; the driver must support it (driver >= 580 / "CUDA Version: 13.x"
#    in the nvidia-smi header). A pod image with a 12.4 driver fails at import time after a 5-minute install. Pick a newer image.
nvidia-smi | head -4 || true   # head closes the pipe early; with pipefail + set -e the SIGPIPE (exit 141) killed the script silently on RunPod
DRV_CUDA=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
echo "driver supports CUDA ${DRV_CUDA:-unknown}"
if [ -n "${DRV_CUDA:-}" ] && [ "${DRV_CUDA%%.*}" -lt 13 ]; then
  echo "WARNING: driver supports CUDA ${DRV_CUDA} < 13. prime-rl's default wheels need CUDA 13."
  echo "Fix (worked on nebius Ubuntu 22 image, 2026-11-08): sudo apt update && sudo apt install -y nvidia-driver-580-server"
  echo "then either reboot (VM may need a dashboard Stop/Resume to come back) or swap modules in place:"
  echo "  sudo systemctl stop nvidia-persistenced nvidia-dcgm; sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia; sudo modprobe nvidia nvidia_uvm"
  read -r -p "Continue anyway? [y/N] " yn; [ "${yn:-N}" = "y" ] || exit 1
fi
# 0b. NVIDIA DCGM (nv-hostengine) listens on tcp/5555 — the same default port as prime-rl's orchestrator→trainer ZMQ
#     transport. It is only telemetry; stop it or the orchestrator dies with "Address already in use".
sudo systemctl stop nvidia-dcgm 2>/dev/null && sudo systemctl disable nvidia-dcgm 2>/dev/null || true
# 1. prime-rl + uv (their installer handles uv, clone, uv sync, flash-attn)
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
cd ~/prime-rl
uv run python -c "import flash_attn; print('flash-attn ok')"
nvidia-smi --query-gpu=name,memory.total --format=csv
# 2. secrets: pod_sync.sh ships ~/.pod-env (PRIME_API_KEY, PRIME_TEAM_ID, HF_TOKEN, optionally WANDB_API_KEY).
grep -q "source ~/.pod-env" ~/.bashrc || echo "[ -f ~/.pod-env ] && source ~/.pod-env" >> ~/.bashrc
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
[ -f ~/.pod-env ] && source ~/.pod-env
# wandb: non-interactive if WANDB_API_KEY was shipped, else log in by hand in an interactive shell afterwards.
if [ -n "${WANDB_API_KEY:-}" ]; then uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "wandb ok"; else echo "wandb: run 'uv run wandb login' interactively"; fi
# 3. placeholders the sync step fills: ~/prime-intellect/{env,data/dataset}
mkdir -p ~/prime-intellect
echo "setup done. Now run scripts/pod_sync.sh from the Mac, then: cd ~/prime-rl && uv pip install -e ~/prime-intellect/env -e ~/prime-intellect/env/environments/recipe_classify"

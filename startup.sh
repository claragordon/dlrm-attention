#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a clean Python venv with a consistent PyTorch CUDA stack
# that supports torch.nn.attention.flex_attention.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu121}"

echo "[startup] repo: $ROOT_DIR"
echo "[startup] venv: $VENV_DIR"
echo "[startup] torch index: $TORCH_INDEX_URL"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[startup] creating virtualenv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[startup] upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "[startup] removing potentially conflicting torch/cuda packages"
pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
pip freeze | grep '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y >/dev/null 2>&1 || true

echo "[startup] clearing pip cache"
pip cache purge >/dev/null 2>&1 || true

echo "[startup] removing leftover torch/nvidia directories"
SITE_PACKAGES="$("$VENV_DIR/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
find "$SITE_PACKAGES" -maxdepth 1 -type d -name 'nvidia*' -exec rm -rf {} + || true
find "$SITE_PACKAGES" -maxdepth 1 -type d -name 'torch*' -exec rm -rf {} + || true

echo "[startup] installing nightly torch stack"
pip install --pre torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "[startup] validating torch + flex_attention"
unset LD_LIBRARY_PATH || true
"$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
from torch.nn.attention.flex_attention import flex_attention
print("flex_attention: AVAILABLE")
PY

echo "[startup] done"

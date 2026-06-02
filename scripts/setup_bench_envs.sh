#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCH_PYTHON="${BENCH_PYTHON:-python3.12}"
MAX_JOBS="${MAX_JOBS:-4}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

echo "Syncing project .venv"
uv sync --extra bench
uv pip install --python .venv/bin/python setuptools wheel

mkdir -p .bench

setup_flash_env() {
  local name="$1"
  local flash_version="$2"
  local env_dir=".bench/${name}"
  local python_bin="${env_dir}/bin/python"

  echo "Creating ${env_dir}"
  uv venv "$env_dir" --python "$BENCH_PYTHON" --clear

  echo "Installing PyTorch ${TORCH_VERSION} in ${env_dir}"
  uv pip install --python "$python_bin" --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION}"

  echo "Installing build and benchmark dependencies in ${env_dir}"
  uv pip install --python "$python_bin" setuptools wheel packaging ninja matplotlib

  echo "Installing flash-attn==${flash_version} in ${env_dir}"
  MAX_JOBS="$MAX_JOBS" uv pip install --python "$python_bin" "flash-attn==${flash_version}" --no-build-isolation
}

setup_flash_env fa1 "${FA1_VERSION:-1.0.9}"
setup_flash_env fa2 "${FA2_VERSION:-2.8.3}"

echo "Done"

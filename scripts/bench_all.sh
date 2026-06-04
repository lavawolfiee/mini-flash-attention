#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULT_DIR="${RESULT_DIR:-benchs/results/$(date +%Y%m%d_%H%M%S)}"
N_VALUES="${N_VALUES:-1024 2048 4096 8192}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
INPUT_BUFFERS="${INPUT_BUFFERS:-4}"
FLUSH_L2_MB="${FLUSH_L2_MB:-256}"
BATCH="${BATCH:-1}"
HEADS="${HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-64}"
DTYPE="${DTYPE:-fp16}"
GPU="${GPU:-}"

if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "Using physical GPU ${GPU} via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "$RESULT_DIR"

run_backend() {
  local backend="$1"
  local python_bin="$2"
  local out_file="${RESULT_DIR}/${backend}.jsonl"

  if [[ ! -x "$python_bin" ]]; then
    echo "Skipping ${backend}: ${python_bin} does not exist"
    return
  fi

  echo "Running ${backend}"
  if "$python_bin" benchs/benchmark_forward.py \
      --backend "$backend" \
      --out "$out_file" \
      --n-values $N_VALUES \
      --batch "$BATCH" \
      --heads "$HEADS" \
      --head-dim "$HEAD_DIM" \
      --dtype "$DTYPE" \
      --warmup "$WARMUP" \
      --iters "$ITERS" \
      --input-buffers "$INPUT_BUFFERS" \
      --flush-l2-mb "$FLUSH_L2_MB"; then
    echo "Saved ${out_file}"
  else
    echo "Skipping ${backend}: benchmark failed"
  fi
}

run_backend ours .venv/bin/python
run_backend wmma .venv/bin/python
run_backend torch_old .venv/bin/python
run_backend torch_fp16 .venv/bin/python
run_backend torch_fp16_compile .venv/bin/python
run_backend fa1 .bench/fa1/bin/python
run_backend fa2 .bench/fa2/bin/python

shopt -s nullglob
result_files=("$RESULT_DIR"/*.jsonl)
if (( ${#result_files[@]} > 0 )); then
  .venv/bin/python benchs/plot_results.py "${result_files[@]}" --out-dir "$RESULT_DIR"
  echo "Plots saved to ${RESULT_DIR}"
else
  echo "No result files were produced"
fi

# Forward Benchmarks

The benchmark measures forward-only attention runtime with CUDA events and peak
allocated CUDA memory. All calls run under `torch.inference_mode()` with
`dropout_p=0.0` for FlashAttention backends.

FA1 and FA2 run in separate virtual environments because both install a
`flash_attn` Python package with different compiled extensions. Layout
conversion, such as `transpose(1, 2).contiguous()`, is done while creating input
buffers and is not included in the timed forward loop. Runtime plots and
TFLOPs/s use median latency; mean latency is kept in JSONL for reference.

Run the project environment and optional FlashAttention environments:

```bash
scripts/setup_bench_envs.sh
```

Run all available backends and create grouped bar plots:

```bash
scripts/bench_all.sh
```

Run on a specific physical GPU:

```bash
GPU=2 scripts/bench_all.sh
```

The default result directory is `benchs/results/<timestamp>/`. It contains
`runtime_bars.png`, `memory_bars.png`, `tflops_bars.png`, and one JSONL file per
backend. You can override the common parameters with environment variables:

```bash
N_VALUES="1024 2048 4096 8192 16384" WARMUP=20 ITERS=100 scripts/bench_all.sh
```

Run a single backend manually:

```bash
.venv/bin/python benchs/benchmark_forward.py --backend ours --out benchs/results/ours.jsonl
.venv/bin/python benchs/benchmark_forward.py --backend torch --out benchs/results/torch.jsonl
.bench/fa1/bin/python benchs/benchmark_forward.py --backend fa1 --out benchs/results/fa1.jsonl
.bench/fa2/bin/python benchs/benchmark_forward.py --backend fa2 --out benchs/results/fa2.jsonl
```

Build a plot from existing JSONL files:

```bash
.venv/bin/python benchs/plot_results.py benchs/results/*.jsonl --out-dir benchs/results
```

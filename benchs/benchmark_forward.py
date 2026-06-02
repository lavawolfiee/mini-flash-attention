import argparse
import gc
import importlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch


PreparedInput = tuple[object, ...]


class Backend(Protocol):
    name: str

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        ...

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool
    dtype: torch.dtype
    warmup: int
    iters: int
    input_buffers: int
    flush_l2_mb: int


@dataclass(frozen=True)
class BenchResult:
    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool
    dtype: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    tflops_s: float
    peak_memory_mb: float
    warmup: int
    iters: int
    input_buffers: int
    flush_l2_mb: int
    torch_version: str
    cuda_version: str | None
    gpu: str
    python: str


class OursBackend:
    name = "ours"

    def __init__(self) -> None:
        from mini_flash_attention import forward

        self.forward = forward

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        return q, k, v

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        q, k, v = inputs
        return self.forward(q, k, v, causal=causal)


class WmmaBackend:
    name = "wmma"

    def __init__(self) -> None:
        from mini_flash_attention import forward_wmma

        self.forward = forward_wmma

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        return q, k, v

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        q, k, v = inputs
        return self.forward(q, k, v, causal=causal)


class TorchBackend:
    name = "torch"

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        return q, k, v

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        q, k, v = inputs
        scale = q.shape[-1] ** -0.5
        scores = q.float() @ k.float().transpose(-2, -1) * scale

        if causal:
            seq_len = q.shape[-2]
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        return (probs @ v.float()).to(q.dtype)


class FlashAttention1Backend:
    name = "fa1"

    def __init__(self) -> None:
        module = importlib.import_module("flash_attn.flash_attn_interface")
        self.forward = module.flash_attn_unpadded_func

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        batch, heads, seq_len, head_dim = q.shape
        q_bnhd = q.transpose(1, 2).contiguous().view(batch * seq_len, heads, head_dim)
        k_bnhd = k.transpose(1, 2).contiguous().view(batch * seq_len, heads, head_dim)
        v_bnhd = v.transpose(1, 2).contiguous().view(batch * seq_len, heads, head_dim)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=q.device,
            dtype=torch.int32,
        )
        return q_bnhd, k_bnhd, v_bnhd, cu_seqlens, seq_len

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        q, k, v, cu_seqlens, max_seqlen = inputs
        return self.forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            int(max_seqlen),
            int(max_seqlen),
            0.0,
            softmax_scale=None,
            causal=causal,
        )


class FlashAttention2Backend:
    name = "fa2"

    def __init__(self) -> None:
        module = importlib.import_module("flash_attn")
        self.forward = module.flash_attn_func

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PreparedInput:
        return (
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )

    def run(self, inputs: PreparedInput, causal: bool) -> torch.Tensor:
        q, k, v = inputs
        return self.forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention forward implementations.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["ours", "wmma", "torch", "fa1", "fa2"],
        help="Implementation to benchmark. FA1 and FA2 should run in separate environments.",
    )
    parser.add_argument("--out", type=Path, default=Path("benchs/results/forward.jsonl"))
    parser.add_argument("--n-values", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--input-buffers", type=int, default=4)
    parser.add_argument("--flush-l2-mb", type=int, default=256)
    return parser.parse_args()


def load_backend(name: str) -> Backend:
    if name == "ours":
        return OursBackend()
    if name == "wmma":
        return WmmaBackend()
    if name == "torch":
        return TorchBackend()
    if name == "fa1":
        return FlashAttention1Backend()
    if name == "fa2":
        return FlashAttention2Backend()
    raise ValueError(f"Unknown backend: {name}")


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def make_inputs(config: BenchConfig, backend: Backend) -> list[PreparedInput]:
    shape = (config.batch, config.heads, config.seq_len, config.head_dim)
    inputs = []

    for seed in range(config.input_buffers):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        q = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        k = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        v = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        # Expensive layout conversion for FA1/FA2 happens here, outside timed runs.
        inputs.append(backend.prepare(q, k, v))

    return inputs


def make_l2_flush_buffer(flush_l2_mb: int) -> torch.Tensor | None:
    if flush_l2_mb <= 0:
        return None

    element_size = torch.empty((), dtype=torch.float32).element_size()
    elements = flush_l2_mb * 1024 * 1024 // element_size
    return torch.empty((elements,), device="cuda", dtype=torch.float32)


def flush_l2(buffer: torch.Tensor | None) -> None:
    if buffer is None:
        return

    buffer.fill_(1.0)
    torch.cuda.synchronize()


def benchmark_one(
    backend: Backend,
    inputs: list[PreparedInput],
    config: BenchConfig,
    l2_flush_buffer: torch.Tensor | None,
) -> BenchResult:
    with torch.inference_mode():
        for i in range(config.warmup):
            flush_l2(l2_flush_buffer)
            out = backend.run(inputs[i % len(inputs)], config.causal)
            del out

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Prepared inputs and backend metadata are the memory baseline.
        baseline_memory = torch.cuda.memory_allocated()

        timings_ms = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(config.iters):
            flush_l2(l2_flush_buffer)

            start.record()
            out = backend.run(inputs[i % len(inputs)], config.causal)
            end.record()
            end.synchronize()

            timings_ms.append(start.elapsed_time(end))
            del out

        torch.cuda.synchronize()

    median_ms = statistics.median(timings_ms)
    mean_ms = sum(timings_ms) / len(timings_ms)
    peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    return BenchResult(
        backend=backend.name,
        batch=config.batch,
        heads=config.heads,
        seq_len=config.seq_len,
        head_dim=config.head_dim,
        causal=config.causal,
        dtype=str(config.dtype).replace("torch.", ""),
        median_ms=median_ms,
        mean_ms=mean_ms,
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        tflops_s=attention_tflops_s(config, median_ms),
        peak_memory_mb=peak_memory / 1024**2,
        warmup=config.warmup,
        iters=config.iters,
        input_buffers=config.input_buffers,
        flush_l2_mb=config.flush_l2_mb,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        python=sys.version.split()[0],
    )


def attention_tflops_s(config: BenchConfig, runtime_ms: float) -> float:
    if runtime_ms <= 0:
        return 0.0

    if config.causal:
        attended_tokens = config.seq_len * (config.seq_len + 1) / 2
    else:
        attended_tokens = config.seq_len * config.seq_len

    flops = 4 * config.batch * config.heads * attended_tokens * config.head_dim
    return flops / (runtime_ms / 1000) / 1e12


def print_header(args: argparse.Namespace, backend: Backend) -> None:
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"backend: {backend.name}")
    print(
        f"dtype={args.dtype}, B={args.batch}, H={args.heads}, "
        f"D={args.head_dim}, causal={args.causal}"
    )
    print(
        f"warmup={args.warmup}, iters={args.iters}, "
        f"input_buffers={args.input_buffers}, flush_l2_mb={args.flush_l2_mb}"
    )
    print()
    print("     N | backend |  median ms |    mean ms |     min ms |     max ms |   TFLOPs/s | peak memory MB")
    print("-" * 101)


def print_result(result: BenchResult) -> None:
    print(
        f"{result.seq_len:>6} | {result.backend:<7} | "
        f"{result.median_ms:>10.3f} | {result.mean_ms:>10.3f} | "
        f"{result.min_ms:>10.3f} | {result.max_ms:>10.3f} | "
        f"{result.tflops_s:>10.2f} | "
        f"{result.peak_memory_mb:>14.1f}"
    )


def append_result(path: Path, result: BenchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.set_grad_enabled(False)
    backend = load_backend(args.backend)
    dtype = parse_dtype(args.dtype)

    print_header(args, backend)

    for seq_len in args.n_values:
        inputs = None
        l2_flush_buffer = None
        config = BenchConfig(
            batch=args.batch,
            heads=args.heads,
            seq_len=seq_len,
            head_dim=args.head_dim,
            causal=args.causal,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            input_buffers=args.input_buffers,
            flush_l2_mb=args.flush_l2_mb,
        )

        try:
            inputs = make_inputs(config, backend)
            l2_flush_buffer = make_l2_flush_buffer(config.flush_l2_mb)
            result = benchmark_one(backend, inputs, config, l2_flush_buffer)
            append_result(args.out, result)
            print_result(result)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{seq_len:>6} | {backend.name:<7} | OOM")
        finally:
            del inputs
            del l2_flush_buffer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

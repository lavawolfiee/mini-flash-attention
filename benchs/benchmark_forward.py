import argparse
import gc
from dataclasses import dataclass
from typing import Callable

import torch

from mini_flash_attention import forward


@dataclass(frozen=True)
class BenchConfig:
    B: int
    H: int
    D: int
    N: int
    causal: bool
    dtype: torch.dtype
    warmup: int
    iters: int
    buffer_size: int
    flush_l2_mb: int


@dataclass(frozen=True)
class BenchResult:
    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    peak_memory_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mini flash attention forward.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--buffer-size", type=int, default=4)
    parser.add_argument("--flush-l2-mb", type=int, default=256)
    return parser.parse_args()


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = q.float() @ k.float().transpose(-2, -1) * scale

    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)


def make_inputs(config: BenchConfig) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    inputs = []
    shape = (config.B, config.H, config.N, config.D)

    for seed in range(config.buffer_size):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        q = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        k = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        v = torch.randn(shape, device="cuda", dtype=config.dtype, generator=generator).contiguous()
        inputs.append((q, k, v))

    return inputs


def make_l2_flush_buffer(flush_l2_mb: int) -> torch.Tensor | None:
    if flush_l2_mb <= 0:
        return None

    elements = flush_l2_mb * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
    return torch.empty((elements,), device="cuda", dtype=torch.float32)


def flush_l2(buffer: torch.Tensor | None) -> None:
    if buffer is None:
        return

    buffer.fill_(1.0)
    torch.cuda.synchronize()


def benchmark_one(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    config: BenchConfig,
    l2_flush_buffer: torch.Tensor | None,
) -> BenchResult:
    with torch.inference_mode():
        for i in range(config.warmup):
            q, k, v = inputs[i % len(inputs)]
            flush_l2(l2_flush_buffer)
            out = fn(q, k, v, config.causal)
            del out

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()

        timings_ms = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(config.iters):
            q, k, v = inputs[i % len(inputs)]
            flush_l2(l2_flush_buffer)

            start.record()
            out = fn(q, k, v, config.causal)
            end.record()
            end.synchronize()

            timings_ms.append(start.elapsed_time(end))
            del out

        torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    return BenchResult(
        name=name,
        mean_ms=sum(timings_ms) / len(timings_ms),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        peak_memory_mb=peak_memory / 1024**2,
    )


def print_result(n: int, result: BenchResult) -> None:
    print(
        f"{n:>6} | {result.name:<10} | "
        f"{result.mean_ms:>10.3f} | {result.min_ms:>10.3f} | "
        f"{result.max_ms:>10.3f} | {result.peak_memory_mb:>14.1f}"
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.set_grad_enabled(False)

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"dtype: fp16, B=1, H=8, D=64, causal=false")
    print(f"warmup={args.warmup}, iters={args.iters}, buffer_size={args.buffer_size}")
    print(f"flush_l2_mb={args.flush_l2_mb}")
    print()
    print("     N | impl       |    mean ms |     min ms |     max ms | peak memory MB")
    print("-" * 78)

    for n in args.n_values:
        inputs = None
        l2_flush_buffer = None
        config = BenchConfig(
            B=1,
            H=8,
            D=64,
            N=n,
            causal=False,
            dtype=torch.float16,
            warmup=args.warmup,
            iters=args.iters,
            buffer_size=args.buffer_size,
            flush_l2_mb=args.flush_l2_mb,
        )

        try:
            inputs = make_inputs(config)
            l2_flush_buffer = make_l2_flush_buffer(config.flush_l2_mb)

            kernel_result = benchmark_one("kernel", forward, inputs, config, l2_flush_buffer)
            naive_result = benchmark_one("naive", naive_attention, inputs, config, l2_flush_buffer)

            print_result(n, kernel_result)
            print_result(n, naive_result)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{n:>6} | OOM")
        finally:
            del inputs
            del l2_flush_buffer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

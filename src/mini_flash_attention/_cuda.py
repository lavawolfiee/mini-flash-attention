import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _load_extension():
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parent
    sources = [
        root / "flash_attn.cpp",
        root / "flash_attn_cuda.cu",
    ]
    cutlass_include = root.parents[1] / "third_party" / "cutlass" / "include"

    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    _EXTENSION = load(
        name="flash_attn_cuda",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_include_paths=[str(cutlass_include)],
        verbose=False,
    )
    return _EXTENSION


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return _load_extension().forward(q, k, v, causal)

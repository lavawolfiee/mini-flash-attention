#include <cmath>
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

void launch_flash_attn_forward(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int BH,
    int N,
    int D,
    float scale,
    bool causal,
    cudaStream_t stream);

torch::Tensor flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16");
  TORCH_CHECK(k.scalar_type() == torch::kFloat16, "k must be float16");
  TORCH_CHECK(v.scalar_type() == torch::kFloat16, "v must be float16");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, N, D]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q");
  TORCH_CHECK(v.sizes() == q.sizes(), "v must have the same shape as q");

  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto N = q.size(2);
  const auto D = q.size(3);

  TORCH_CHECK(D > 0, "D must be greater than 0");
  TORCH_CHECK(B * H <= std::numeric_limits<int>::max(), "B * H is too large");
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "N is too large");
  TORCH_CHECK(D <= std::numeric_limits<int>::max(), "D is too large");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::zeros_like(q);

  const int BH = static_cast<int>(B * H);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  launch_flash_attn_forward(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      BH,
      static_cast<int>(N),
      static_cast<int>(D),
      scale,
      causal,
      at::cuda::getCurrentCUDAStream());

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attn_forward, "Mini Flash Attention forward");
}

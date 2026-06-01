#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

template <typename T, typename U>
constexpr auto ceil_div(T a, U b) {
  using R = std::common_type_t<T, U>;
  return (static_cast<R>(a) + static_cast<R>(b) - R{1}) / static_cast<R>(b);
}

inline void check_cuda(cudaError_t error) {
  if (error == cudaSuccess) {
    return;
  }

  throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
}

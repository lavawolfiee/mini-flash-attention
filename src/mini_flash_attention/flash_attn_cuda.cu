#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.cuh"


__forceinline__ __device__ float ReduceWarpSum(float x) {
    constexpr unsigned mask = 0xffffffff;
    x += __shfl_xor_sync(mask, x, 16);
    x += __shfl_xor_sync(mask, x, 8);
    x += __shfl_xor_sync(mask, x, 4);
    x += __shfl_xor_sync(mask, x, 2);
    x += __shfl_xor_sync(mask, x, 1);
    return x; // full sum in every lane
}

template <size_t BLOCK_SIZE, size_t D, size_t Br, size_t Bc>
__global__ void FlashAttnKernel(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int BH,
    int N,
    float scale,
    bool causal
) {
    constexpr size_t NUM_WARPS = ceil_div(BLOCK_SIZE, 32);
    constexpr size_t D_PER_THREAD = D / 32;
    constexpr size_t ROWS_PER_WARP = Br / NUM_WARPS;
    static_assert(D % 32 == 0);
    static_assert(Br % NUM_WARPS == 0);

    size_t headIdx = blockIdx.y;
    size_t qBaseIdx = blockIdx.x * Br;
    size_t tid = threadIdx.x;
    size_t warpId = tid / 32;
    size_t laneId = tid % 32;
    size_t warpBaseRow = warpId * ROWS_PER_WARP;

    // moving ptrs to the beggining of the working area
    q += (headIdx * N + qBaseIdx) * D;
    out += (headIdx * N + qBaseIdx) * D;
    k += headIdx * N * D;
    v += headIdx * N * D;

    // loading q block into smem
    __shared__ half Qs[Br * D];

    for (size_t i = tid; i < Br * D; i += BLOCK_SIZE) {
        // coalesced memory access
        Qs[i] = q[i];
    }

    // each warp processes on q row
    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float o[ROWS_PER_WARP][D_PER_THREAD];

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;

        #pragma unroll
        for (size_t k = 0; k < D_PER_THREAD; ++k) {
            o[r][k] = 0.0f;
        }
    }

    // iterating over K/Vs
    for (size_t i = 0; i < N; i += Bc) {
        // loading Bc K/Vs into smem
        __shared__ half Ks[Bc * D];
        __shared__ half Vs[Bc * D];

        for (size_t j = tid; j < Bc * D; j += BLOCK_SIZE) {
            Ks[j] = k[j];
        }
        for (size_t j = tid; j < Bc * D; j += BLOCK_SIZE) {
            Vs[j] = v[j];
        }
        __syncthreads();  // signals that keys/values are loaded into smem

        // processing loaded K/Vs by all warps
        #pragma unroll
        for (size_t kIdx = 0; kIdx < Bc; ++kIdx) {
            #pragma unroll
            for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
                float s = 0.0f;

                #pragma unroll
                for (size_t j = laneId, k = 0; k < D_PER_THREAD; j += 32, ++k) {
                    s += __half2float(__hmul(Qs[(warpBaseRow + r) * D + j], Ks[kIdx * D + j]));
                }

                s = ReduceWarpSum(s) * scale;

                // online softmax calculation
                float mNew = fmaxf(m[r], s);
                float alpha = __expf(m[r] - mNew);
                float p = __expf(s - mNew);

                denom[r] = denom[r] * alpha + p;
                m[r] = mNew;

                #pragma unroll
                for (size_t j = laneId, k = 0; k < D_PER_THREAD; j += 32, ++k) {
                    o[r][k] = o[r][k] * alpha + p * __half2float(Vs[kIdx * D + j]);
                }
            }
        }

        // moving k/v ptrs
        k += Bc * D;
        v += Bc * D;

        __syncthreads();  // signals that K/Vs in smem were used and are no longer needed
    }

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        float denomInv = 1.0f / denom[r];

        for (size_t j = laneId, k = 0; k < D_PER_THREAD; j += 32, ++k) {
            out[(warpBaseRow + r) * D + j] = __float2half(o[r][k] * denomInv);
        }
    }
}


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
    cudaStream_t stream
) {
    constexpr size_t BLOCK_SIZE = 512;
    constexpr size_t ROWS_PER_WARP = 8;
    constexpr size_t WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    constexpr size_t Br = ROWS_PER_WARP * WARPS_PER_BLOCK;
    constexpr size_t Bc = 128;

    if (N % Br != 0 || N % Bc != 0) {
        throw std::runtime_error("Only N (seq len) divisible by " + std::to_string(Br) + " and " + std::to_string(Bc) + " are supported yet");
    }
    if (causal) {
        throw std::runtime_error("Causal attention is not yet supported");
    }

    // each block processes a Qblock of size Br in one head
    // for now, one warp processes on q row
    dim3 gridDim(ceil_div(N, Br), BH);
    dim3 blockSize(BLOCK_SIZE);

    if (D == 32) {
        FlashAttnKernel<BLOCK_SIZE, 32, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
            q, k, v, out, BH, N, scale, causal
        );
        check_cuda(cudaGetLastError());
    } else if (D == 64) {
        FlashAttnKernel<BLOCK_SIZE, 64, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
            q, k, v, out, BH, N, scale, causal
        );
        check_cuda(cudaGetLastError());
    } else if (D == 128) {
        FlashAttnKernel<BLOCK_SIZE, 128, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
            q, k, v, out, BH, N, scale, causal
        );
        check_cuda(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported D");
    }
}

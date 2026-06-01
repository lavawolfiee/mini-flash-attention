#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "utils.cuh"


using namespace nvcuda;


__forceinline__ __device__ float ReduceWarpSum(float x) {
    constexpr unsigned mask = 0xffffffff;
    x += __shfl_xor_sync(mask, x, 16);
    x += __shfl_xor_sync(mask, x, 8);
    x += __shfl_xor_sync(mask, x, 4);
    x += __shfl_xor_sync(mask, x, 2);
    x += __shfl_xor_sync(mask, x, 1);
    return x; // full sum in every lane
}


__forceinline__ __device__ float ReduceWarpMax(float x) {
    constexpr unsigned mask = 0xffffffff;
    x = fmaxf(x, __shfl_xor_sync(mask, x, 16));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 8));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 4));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 2));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 1));
    return x; // full max in every lane
}


template <size_t BLOCK_SIZE, size_t D, size_t Br, size_t Bc>
__global__ void FlashAttnWmmaQKKernel(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int BH,
    int N,
    float scale,
    bool causal
) {
    constexpr size_t NUM_WARPS = BLOCK_SIZE / 32;
    constexpr size_t ROWS_PER_WARP = Br / NUM_WARPS;
    constexpr size_t D_PER_THREAD = D / 32;

    static_assert(D == 64);
    static_assert(Br == 64);
    static_assert(Bc == 64);
    static_assert(BLOCK_SIZE == 128);
    static_assert(NUM_WARPS == 4);
    static_assert(ROWS_PER_WARP == 16);
    static_assert(D_PER_THREAD == 2);

    size_t headIdx = blockIdx.y;
    size_t qBaseIdx = blockIdx.x * Br;
    size_t tid = threadIdx.x;
    size_t warpId = tid / 32;
    size_t laneId = tid % 32;
    size_t warpBaseRow = warpId * ROWS_PER_WARP;

    // moving ptrs to the beginning of the working area
    q += (headIdx * N + qBaseIdx) * D;
    out += (headIdx * N + qBaseIdx) * D;
    k += headIdx * N * D;
    v += headIdx * N * D;

    // Q/K/V are stored in shared memory in simple row-major layout.
    // This is not the final FA2 swizzled layout, but it is much easier to debug.
    __shared__ half Qs[Br * D];      // [Br, D] = [64, 64]
    __shared__ half Ks[Bc * D];      // [Bc, D] = [64, 64]
    __shared__ half Vs[Bc * D];      // [Bc, D] = [64, 64]

    // Scores is used twice:
    // 1. after WMMA QK^T it stores raw fp32 scores [Br, Bc]
    // 2. after softmax it stores fp32 probabilities P [Br, Bc]
    __shared__ float Scores[Br * Bc];

    // loading Q block once: global -> shared
    for (size_t i = tid; i < Br * D; i += BLOCK_SIZE) {
        Qs[i] = q[i];
    }
    __syncthreads();

    // Each warp owns 16 Q rows.
    // Each lane owns 2 output dimensions: lane and lane + 32.
    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float o[ROWS_PER_WARP][D_PER_THREAD];

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;

        #pragma unroll
        for (size_t dd = 0; dd < D_PER_THREAD; ++dd) {
            o[r][dd] = 0.0f;
        }
    }

    // iterating over K/V blocks
    for (size_t kvBaseIdx = 0; kvBaseIdx < N; kvBaseIdx += Bc) {
        // loading K/V block: global -> shared
        for (size_t i = tid; i < Bc * D; i += BLOCK_SIZE) {
            Ks[i] = k[kvBaseIdx * D + i];
            Vs[i] = v[kvBaseIdx * D + i];
        }
        __syncthreads();

        // ------------------------------------------------------------
        // 1. Compute S = Q @ K^T for this block using WMMA.
        //
        // One warp computes S tile [16, 64]:
        //   rows: its own 16 Q rows
        //   cols: all 64 K rows in current K block
        //
        // We split 64 K columns into 4 chunks of 16 columns.
        // For each [16,16] S tile:
        //   S[16,16] = Q[16,64] @ K_chunk[16,64]^T
        // ------------------------------------------------------------

        #pragma unroll
        for (size_t colBlock = 0; colBlock < Bc; colBlock += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qFrag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kFrag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sFrag;

            wmma::fill_fragment(sFrag, 0.0f);

            #pragma unroll
            for (size_t dBlock = 0; dBlock < D; dBlock += 16) {
                // Q fragment: [16 rows, 16 dims], row-major.
                const half* qTilePtr = Qs + (warpBaseRow * D + dBlock);

                // K fragment is interpreted as B matrix [D, 16] in column-major.
                // Ks is row-major [K_row, D], so col-major B with ldm=D maps:
                // B[d, col] = Ks[col, d].
                const half* kTilePtr = Ks + (colBlock * D + dBlock);

                wmma::load_matrix_sync(qFrag, qTilePtr, D);
                wmma::load_matrix_sync(kFrag, kTilePtr, D);
                wmma::mma_sync(sFrag, qFrag, kFrag, sFrag);
            }

            // Store raw fp32 scores to shared memory.
            // Scores layout: [Br, Bc], row-major.
            float* sTilePtr = Scores + (warpBaseRow * Bc + colBlock);
            wmma::store_matrix_sync(sTilePtr, sFrag, Bc, wmma::mem_row_major);
        }

        __syncthreads();

        // ------------------------------------------------------------
        // 2. Blockwise online softmax.
        //
        // Unlike the previous scalar version, we do not update softmax
        // statistics for every single key separately.
        //
        // For each row:
        //   blockMax = max scores in current K block
        //   mNew = max(old m, blockMax)
        //   old O and denominator are rescaled once per K block
        //   Scores[row, col] is overwritten with P[row, col]
        // ------------------------------------------------------------

        #pragma unroll
        for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
            size_t row = warpBaseRow + r;

            float localMax = -INFINITY;

            // each lane checks two columns: lane and lane + 32
            #pragma unroll
            for (size_t dd = 0; dd < 2; ++dd) {
                size_t col = laneId + dd * 32;
                float s = Scores[row * Bc + col] * scale;
                localMax = fmaxf(localMax, s);
            }

            float blockMax = ReduceWarpMax(localMax);
            float mNew = fmaxf(m[r], blockMax);
            float alpha = __expf(m[r] - mNew);

            // rescale old output accumulator once per K/V block
            denom[r] *= alpha;

            #pragma unroll
            for (size_t dd = 0; dd < D_PER_THREAD; ++dd) {
                o[r][dd] *= alpha;
            }

            float localDenomAdd = 0.0f;

            // convert current score block into probabilities in shared memory
            #pragma unroll
            for (size_t dd = 0; dd < 2; ++dd) {
                size_t col = laneId + dd * 32;
                float s = Scores[row * Bc + col] * scale;
                float p = __expf(s - mNew);

                Scores[row * Bc + col] = p;
                localDenomAdd += p;
            }

            float denomAdd = ReduceWarpSum(localDenomAdd);
            denom[r] += denomAdd;
            m[r] = mNew;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // 3. Compute O += P @ V.
        //
        // This first WMMA version intentionally keeps P@V scalar.
        // It is slower than real FA2, but it keeps the code readable:
        //   - QK^T already uses Tensor Cores
        //   - softmax is already blockwise
        //   - work split is sliced-Q
        //
        // Next step will be replacing this part with WMMA too.
        // ------------------------------------------------------------

        #pragma unroll
        for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
            size_t row = warpBaseRow + r;

            #pragma unroll
            for (size_t dd = 0; dd < D_PER_THREAD; ++dd) {
                size_t d = laneId + dd * 32;

                float acc = 0.0f;

                #pragma unroll
                for (size_t col = 0; col < Bc; ++col) {
                    float p = Scores[row * Bc + col];
                    float vVal = __half2float(Vs[col * D + d]);
                    acc += p * vVal;
                }

                o[r][dd] += acc;
            }
        }

        __syncthreads();
    }

    // final normalization and store
    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        float denomInv = 1.0f / denom[r];

        #pragma unroll
        for (size_t dd = 0; dd < D_PER_THREAD; ++dd) {
            size_t d = laneId + dd * 32;
            out[(warpBaseRow + r) * D + d] = __float2half(o[r][dd] * denomInv);
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
    constexpr size_t BLOCK_SIZE = 128;
    constexpr size_t Br = 64;
    constexpr size_t Bc = 64;

    if (D != 64) {
        throw std::runtime_error("This WMMA prototype supports only D=64 yet");
    }
    if (N % Br != 0 || N % Bc != 0) {
        throw std::runtime_error("Only N (seq len) divisible by " + std::to_string(Br) + " and " + std::to_string(Bc) + " are supported yet");
    }
    if (causal) {
        throw std::runtime_error("Causal attention is not yet supported");
    }

    // each block processes one Q block in one BH item
    // grid.x: Q blocks
    // grid.y: batch * heads
    dim3 gridDim(ceil_div(N, Br), BH);
    dim3 blockSize(BLOCK_SIZE);

    FlashAttnWmmaQKKernel<BLOCK_SIZE, 64, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
        q, k, v, out, BH, N, scale, causal
    );
    check_cuda(cudaGetLastError());
}
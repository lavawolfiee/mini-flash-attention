#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "utils.cuh"


using namespace nvcuda;
using namespace cute;


// -----------------------------------------------------------------------------
// Small warp reductions used by the blockwise softmax.
// -----------------------------------------------------------------------------

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


// -----------------------------------------------------------------------------
// Scale a CuTe register accumulator tensor by rows.
//
// accO is a register tensor produced by TiledMMA.
// cO is an identity-coordinate tensor partitioned exactly like accO.
// For each accumulator element we recover its logical row and multiply by alpha.
//
// This is the important part: no O shared-memory roundtrip is needed.
// -----------------------------------------------------------------------------

template <typename AccTensor, typename CoordTensor>
__forceinline__ __device__ void ScaleAccOByRow(
    AccTensor& accO,
    CoordTensor const& cO,
    float const* alpha
) {
    #pragma unroll
    for (int i = 0; i < size(accO); ++i) {
        auto coord = cO(i);
        int row = int(get<0>(coord));
        accO(i) *= alpha[row];
    }
}


// -----------------------------------------------------------------------------
// Final row-wise normalization by denominator.
// Same idea as ScaleAccOByRow, but multiplier is 1 / denom[row].
// -----------------------------------------------------------------------------

template <typename AccTensor, typename CoordTensor>
__forceinline__ __device__ void NormalizeAccOByRow(
    AccTensor& accO,
    CoordTensor const& cO,
    float const* denom
) {
    #pragma unroll
    for (int i = 0; i < size(accO); ++i) {
        auto coord = cO(i);
        int row = int(get<0>(coord));
        accO(i) *= 1.0f / denom[row];
    }
}


// -----------------------------------------------------------------------------
// CuTe bridge kernel.
//
// Design:
//   - D = 64
//   - Br = 64
//   - Bc = 64
//   - 4 warps per CTA
//   - each warp owns 16 Q rows
//
// This version keeps your current easy-to-read QK^T path:
//   QK^T: WMMA -> shared Scores
//
// But replaces scalar output accumulator:
//   old: float o[16][2]
//   new: CuTe register tensor accO [16,64] per warp
//
// P@V is done by CuTe MMA directly into accO.
// -----------------------------------------------------------------------------

template <size_t BLOCK_SIZE, size_t D, size_t Br, size_t Bc>
__global__ void FlashAttnCutePVKernel(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int BH,
    int N,
    float scale,
    bool causal
) {
    using Element = cutlass::half_t;
    using ElementAccum = float;

    constexpr size_t NUM_WARPS = BLOCK_SIZE / 32;
    constexpr size_t ROWS_PER_WARP = Br / NUM_WARPS;

    static_assert(D == 64);
    static_assert(Br == 64);
    static_assert(Bc == 64);
    static_assert(BLOCK_SIZE == 128);
    static_assert(NUM_WARPS == 4);
    static_assert(ROWS_PER_WARP == 16);

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

    // Reinterpret CUDA half pointers as CUTLASS half_t pointers for CuTe.
    auto qElem = reinterpret_cast<Element const*>(q);
    auto kElem = reinterpret_cast<Element const*>(k);
    auto vElem = reinterpret_cast<Element const*>(v);

    // -------------------------------------------------------------------------
    // Shared memory.
    //
    // Q/K/V layout is intentionally simple row-major, like in your WMMA version.
    // Scores is still used for raw fp32 scores.
    // P is half probabilities for CuTe P@V MMA.
    //
    // Scores is reused at the very end as temporary float output storage.
    // This final store is once per CTA, not once per K/V block.
    // -------------------------------------------------------------------------

    __shared__ Element Qs[Br * D];       // [64, 64]
    __shared__ Element Ks[Bc * D];       // [64, 64]
    __shared__ Element Vs[Bc * D];       // [64, 64]
    __shared__ Element P[Br * Bc];       // [64, 64], half probabilities
    __shared__ float Scores[Br * Bc];    // [64, 64], raw scores / final temp O

    // -------------------------------------------------------------------------
    // CuTe MMA setup for P @ V.
    //
    // The hardware atom below is Ampere fp16 inputs -> fp32 accum.
    // TiledMMA describes a warp-level tile. We use it per warp:
    //   P_warp [16, 64] @ V [64, 64] -> accO [16, 64]
    // -------------------------------------------------------------------------

    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    auto tiledMma = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _64, _16>{}
    );

    auto thrMma = tiledMma.get_slice(laneId);

    // -------------------------------------------------------------------------
    // Shared-memory CuTe views for one warp's P tile and the whole V tile.
    //
    // P_warp shape: [16, 64]
    // V shape:      [64, 64]
    //
    // Both are row-major here to keep the first version readable.
    // Later you can switch V to a better swizzled/transposed layout.
    // -------------------------------------------------------------------------

    Tensor sP = make_tensor(
        make_smem_ptr(P + warpBaseRow * Bc),
        make_layout(make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{}), LayoutRight{})
    );

    // V is physically stored as Vs[key, d] row-major:
    //   address = key * D + d
    //
    // But CuTe MMA B operand is logical B[N,K].
    // For O = P[M,K] @ V[K,N], we expose V as Vt[N,K]:
    //   sVt[d, key] = Vs[key, d]
    //   address = d + key * D
    Tensor sVt = make_tensor(
        make_smem_ptr(Vs),
        make_layout(
            make_shape(Int<D>{}, Int<Bc>{}),     // [N,K]
            make_stride(Int<1>{}, Int<D>{})      // address = d + key * D
        )
    );

    // Identity tensor for output coordinates [16, 64].
    // Partitioned like C/accO, it tells us for every accumulator register
    // which logical (row, col) it corresponds to.
    Tensor cO = make_identity_tensor(
        make_shape(Int<ROWS_PER_WARP>{}, Int<D>{})
    );

    // Partition P/V/O according to the MMA layout.
    Tensor tPrP = thrMma.partition_A(sP);
    Tensor tVrV = thrMma.partition_B(sVt);
    Tensor tOcO = thrMma.partition_C(cO);

    // Register accumulator for O_warp [16, 64].
    // O lives in registers across all K/V blocks.
    Tensor accO = thrMma.make_fragment_C(tOcO);
    clear(accO);

    // -------------------------------------------------------------------------
    // Load Q once: global -> shared.
    // -------------------------------------------------------------------------

    for (size_t i = tid; i < Br * D; i += BLOCK_SIZE) {
        Qs[i] = qElem[i];
    }
    __syncthreads();

    // Online softmax stats per warp-owned row.
    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float alpha[ROWS_PER_WARP];

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;
        alpha[r] = 1.0f;
    }

    // -------------------------------------------------------------------------
    // Main loop over K/V blocks.
    // -------------------------------------------------------------------------

    for (size_t kvBaseIdx = 0; kvBaseIdx < N; kvBaseIdx += Bc) {
        // ---------------------------------------------------------------------
        // Load K/V block: global -> shared.
        // ---------------------------------------------------------------------

        for (size_t i = tid; i < Bc * D; i += BLOCK_SIZE) {
            Ks[i] = kElem[kvBaseIdx * D + i];
            Vs[i] = vElem[kvBaseIdx * D + i];
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 1. Compute S = Q @ K^T.
        //
        // This keeps your current WMMA implementation. It is already fast and
        // easy to read. The next possible step is rewriting this part in CuTe too.
        //
        // One warp computes Scores[16, 64].
        // ---------------------------------------------------------------------

        #pragma unroll
        for (size_t colBlock = 0; colBlock < Bc; colBlock += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qFrag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kFrag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sFrag;

            wmma::fill_fragment(sFrag, 0.0f);

            #pragma unroll
            for (size_t dBlock = 0; dBlock < D; dBlock += 16) {
                const half* qTilePtr = reinterpret_cast<const half*>(Qs + warpBaseRow * D + dBlock);
                const half* kTilePtr = reinterpret_cast<const half*>(Ks + colBlock * D + dBlock);

                wmma::load_matrix_sync(qFrag, qTilePtr, D);
                wmma::load_matrix_sync(kFrag, kTilePtr, D);
                wmma::mma_sync(sFrag, qFrag, kFrag, sFrag);
            }

            float* sTilePtr = Scores + warpBaseRow * Bc + colBlock;
            wmma::store_matrix_sync(sTilePtr, sFrag, Bc, wmma::mem_row_major);
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Blockwise online softmax.
        //   - find block max per row
        //   - compute alpha[row] for old accO rescale
        //   - write P[row, col] in half for Tensor Core P@V
        // ---------------------------------------------------------------------

        #pragma unroll
        for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
            size_t row = warpBaseRow + r;

            float localMax = -INFINITY;

            // each lane checks two columns: lane and lane + 32
            #pragma unroll
            for (size_t colLocalChunk = 0; colLocalChunk < 2; ++colLocalChunk) {
                size_t col = laneId + colLocalChunk * 32;
                float s = Scores[row * Bc + col] * scale;
                localMax = fmaxf(localMax, s);
            }

            float blockMax = ReduceWarpMax(localMax);
            float mNew = fmaxf(m[r], blockMax);

            alpha[r] = __expf(m[r] - mNew);
            denom[r] *= alpha[r];

            float localDenomAdd = 0.0f;

            // Convert current score block into P half in shared memory.
            // P is used by CuTe MMA in the next section.
            #pragma unroll
            for (size_t colLocalChunk = 0; colLocalChunk < 2; ++colLocalChunk) {
                size_t col = laneId + colLocalChunk * 32;
                float s = Scores[row * Bc + col] * scale;
                float p = __expf(s - mNew);

                P[row * Bc + col] = Element(p);
                localDenomAdd += p;
            }

            float denomAdd = ReduceWarpSum(localDenomAdd);
            denom[r] += denomAdd;
            m[r] = mNew;
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // 3. Rescale old accO in registers.
        //
        // This replaces:
        //   o[row][d] *= alpha[row]
        //
        // but does it on the CuTe MMA accumulator tensor.
        // No shared-memory O temp is used.
        // ---------------------------------------------------------------------

        ScaleAccOByRow(accO, tOcO, alpha);

        // ---------------------------------------------------------------------
        // 4. Compute accO += P @ V through CuTe MMA.
        //
        // P is [16,64] for this warp.
        // V is [64,64] for this CTA.
        // accO is [16,64] register accumulator.
        // ---------------------------------------------------------------------

        cute::gemm(tiledMma, tPrP, tVrV, accO);

        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Final normalization in registers:
    //   accO[row, :] /= denom[row]
    // -------------------------------------------------------------------------

    NormalizeAccOByRow(accO, tOcO, denom);

    // -------------------------------------------------------------------------
    // Store accO.
    //
    // For the first readable CuTe bridge version we store accO to shared once,
    // then write the final output using your old lane mapping.
    //
    // This is only one final store/read, not once per K/V block, so it should be
    // much cheaper than the previous WMMA-PV shared roundtrip.
    //
    // Scores memory is no longer needed, so we reuse it as float O temp:
    //   Scores[warpBaseRow : warpBaseRow+16, 0:64]
    // -------------------------------------------------------------------------

    Tensor sO = make_tensor(
        make_smem_ptr(Scores + warpBaseRow * D),
        make_layout(make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}), LayoutRight{})
    );

    Tensor tOsO = thrMma.partition_C(sO);

    copy(accO, tOsO);

    __syncwarp();

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        size_t row = warpBaseRow + r;

        // each lane writes two output dimensions: lane and lane + 32
        #pragma unroll
        for (size_t colLocalChunk = 0; colLocalChunk < 2; ++colLocalChunk) {
            size_t d = laneId + colLocalChunk * 32;
            out[row * D + d] = __float2half(Scores[row * D + d]);
        }
    }
}


// -----------------------------------------------------------------------------
// Launcher.
// -----------------------------------------------------------------------------

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
        throw std::runtime_error("Only D=64 is supported yet");
    }
    if (N % Br != 0 || N % Bc != 0) {
        throw std::runtime_error("Only N divisible by " + std::to_string(Br) + " and " + std::to_string(Bc) + " are supported yet");
    }
    if (causal) {
        throw std::runtime_error("Causal attention is not yet supported");
    }

    // each block processes one Q block in one BH item
    // grid.x: Q blocks
    // grid.y: batch * heads
    dim3 gridDim(::ceil_div(N, Br), BH);
    dim3 blockSize(BLOCK_SIZE);

    FlashAttnCutePVKernel<BLOCK_SIZE, 64, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
        q, k, v, out, BH, N, scale, causal
    );
    check_cuda(cudaGetLastError());
}

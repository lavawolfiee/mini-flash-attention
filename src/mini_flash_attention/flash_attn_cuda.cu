#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "utils.cuh"


using namespace cute;


// -----------------------------------------------------------------------------
// Warp-level reductions.
// Each call returns the final reduced value in every lane of the warp.
// -----------------------------------------------------------------------------

__forceinline__ __device__ float ReduceWarpSum(float x) {
    constexpr unsigned mask = 0xffffffff;
    x += __shfl_xor_sync(mask, x, 16);
    x += __shfl_xor_sync(mask, x, 8);
    x += __shfl_xor_sync(mask, x, 4);
    x += __shfl_xor_sync(mask, x, 2);
    x += __shfl_xor_sync(mask, x, 1);
    return x;
}


__forceinline__ __device__ float ReduceWarpMax(float x) {
    constexpr unsigned mask = 0xffffffff;
    x = fmaxf(x, __shfl_xor_sync(mask, x, 16));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 8));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 4));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 2));
    x = fmaxf(x, __shfl_xor_sync(mask, x, 1));
    return x;
}


// -----------------------------------------------------------------------------
// Shared-memory layouts.
//
// Q, K and P are logically row-major matrices.
// The XOR swizzle changes only their physical placement in shared memory.
// This reduces bank conflicts for Tensor Core loads, while keeping indexing
// logical: tensor(row, col).
// -----------------------------------------------------------------------------

template <size_t Rows, size_t Cols>
CUTE_HOST_DEVICE constexpr auto MakeRowMajorSwizzledLayout() {
    return composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<Cols>{}),
            make_stride(Int<Cols>{}, Int<1>{})
        )
    );
}


// V is consumed by the MMA instruction as a B operand with logical shape [N, K].
// For P @ V:
//
//   P:  [M, K] = [16, 64]
//   V:  [K, N] = [64, 64]
//   O:  [M, N] = [16, 64]
//
// The SM80 TN MMA atom expects B as [N, K], so shared V is exposed as
// logical Vt[d, key] = V[key, d].
template <size_t Rows, size_t Cols, size_t LeadingDim>
CUTE_HOST_DEVICE constexpr auto MakeTransposedSwizzledLayout() {
    return composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<Cols>{}),
            make_stride(Int<1>{}, Int<LeadingDim>{})
        )
    );
}


// -----------------------------------------------------------------------------
// Row-wise operations on CuTe register accumulator tensors.
//
// CuTe accumulator tensors are register fragments with a layout.  The matching
// identity tensor cTensor gives the logical coordinate of every register element.
// This lets us do row-dependent operations such as:
//
//   acc[row, :] *= alpha[row]
//   acc[row, :] /= denom[row]
//
// without storing the accumulator to shared memory.
// -----------------------------------------------------------------------------

template <typename AccTensor, typename CoordTensor>
__forceinline__ __device__ void ScaleAccByRow(
    AccTensor& accTensor,
    CoordTensor const& cTensor,
    float const* alpha
) {
    #pragma unroll
    for (int i = 0; i < size(accTensor); ++i) {
        auto coord = cTensor(i);
        int row = int(get<0>(coord));
        accTensor(i) *= alpha[row];
    }
}


template <typename AccTensor, typename CoordTensor>
__forceinline__ __device__ void NormalizeAccByRow(
    AccTensor& accTensor,
    CoordTensor const& cTensor,
    float const* denom
) {
    #pragma unroll
    for (int i = 0; i < size(accTensor); ++i) {
        auto coord = cTensor(i);
        int row = int(get<0>(coord));
        accTensor(i) *= 1.0f / denom[row];
    }
}


// -----------------------------------------------------------------------------
// Blockwise online softmax for one warp-owned score tile.
//
// Input:
//   accS: register tile with logical shape [ROWS_PER_WARP, Bc]
//         containing Q @ K^T scores for the current K-block.
//   cS:   identity-coordinate tensor partitioned like accS.
//   sP:   this warp's shared P tile [ROWS_PER_WARP, Bc].
//
// Output:
//   sP(row, col) stores softmax probabilities in fp16.
//   m[row], denom[row], alpha[row] update online softmax state.
//
// For each row, this implements:
//
//   m_new     = max(m_old, max(scores_block))
//   alpha     = exp(m_old - m_new)
//   denom     = denom * alpha + sum(exp(scores_block - m_new))
//   P_block   = exp(scores_block - m_new)
//
// The old output accumulator is rescaled later by alpha[row].
// -----------------------------------------------------------------------------

template <
    typename AccTensor,
    typename CoordTensor,
    typename PTensor,
    typename Element,
    size_t ROWS_PER_WARP
>
__forceinline__ __device__ void SoftmaxAccSToPShared(
    AccTensor& accS,
    CoordTensor const& cS,
    PTensor& sP,
    float scale,
    float* m,
    float* denom,
    float* alpha
) {
    float localMax[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        localMax[r] = -INFINITY;
    }

    // Each thread owns a subset of the score tile. First collect per-row maxima
    // over the elements owned by this thread, then reduce across the warp.
    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));

        float s = float(accS(i)) * scale;
        localMax[row] = fmaxf(localMax[row], s);
    }

    float mNew[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        float blockMax = ReduceWarpMax(localMax[r]);
        mNew[r] = fmaxf(m[r], blockMax);
        alpha[r] = __expf(m[r] - mNew[r]);
        denom[r] *= alpha[r];
    }

    float localDenomAdd[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        localDenomAdd[r] = 0.0f;
    }

    // Convert the current score block into probabilities and write them to
    // the warp-local shared P tile.  sP already carries the swizzled layout.
    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));
        int col = int(get<1>(coord));

        float s = float(accS(i)) * scale;
        float p = __expf(s - mNew[row]);

        sP(row, col) = Element(p);
        localDenomAdd[row] += p;
    }

    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        float denomAdd = ReduceWarpSum(localDenomAdd[r]);
        denom[r] += denomAdd;
        m[r] = mNew[r];
    }
}


// -----------------------------------------------------------------------------
// FlashAttention forward prototype, D = 64.
//
// Data layout expected by the kernel:
//   q, k, v, out are interpreted as [BH, N, D] contiguous.
//
// Work decomposition:
//   CTA/block: one (BH item, Q block)
//   Br:        64 query rows per CTA
//   Bc:        64 key/value rows per K/V tile
//   Warps:     4 warps per CTA
//   Per warp:  16 query rows
//
// Main loop for a fixed Q block:
//   1. Load Q once to shared.
//   2. For every K/V block:
//        a. Load K and V to shared.
//        b. Compute accS = Q @ K^T in registers using CuTe MMA.
//        c. Run blockwise online softmax over accS and write P to shared.
//        d. Rescale accO by alpha[row] in registers.
//        e. Accumulate accO += P @ V using CuTe MMA.
//   3. Normalize accO by denom[row] and store output.
// -----------------------------------------------------------------------------

template <size_t BLOCK_SIZE, size_t D, size_t Br, size_t Bc>
__global__ void FlashAttnCuteKernel(
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

    constexpr int NUM_WARPS = int(BLOCK_SIZE / 32);
    constexpr int ROWS_PER_WARP = int(Br / NUM_WARPS);

    static_assert(D == 64);
    static_assert(Br == 64);
    static_assert(Bc == 64);
    static_assert(BLOCK_SIZE == 128);
    static_assert(NUM_WARPS == 4);
    static_assert(ROWS_PER_WARP == 16);

    int headIdx = int(blockIdx.y);
    int qBaseIdx = int(blockIdx.x) * int(Br);
    int tid = int(threadIdx.x);
    int warpId = tid >> 5;
    int laneId = tid & 31;
    int warpBaseRow = warpId * ROWS_PER_WARP;

    // Move pointers to this CTA's working region.
    q += (headIdx * N + qBaseIdx) * int(D);
    out += (headIdx * N + qBaseIdx) * int(D);
    k += headIdx * N * int(D);
    v += headIdx * N * int(D);

    auto qElem = reinterpret_cast<Element const*>(q);
    auto kElem = reinterpret_cast<Element const*>(k);
    auto vElem = reinterpret_cast<Element const*>(v);
    auto outElem = reinterpret_cast<Element*>(out);

    // Shared staging buffers.  They are addressed only through CuTe tensors
    // below, so the logical layout can differ from physical shared placement.
    __shared__ Element Qs[Br * D];       // logical Q  [64, 64]
    __shared__ Element Ks[Bc * D];       // logical K  [64, 64]
    __shared__ Element Vs[Bc * D];       // logical Vt [64, 64]
    __shared__ Element P[Br * Bc];       // logical P  [64, 64]

    // Full-tile shared tensors.  Warp-local Q/P views are derived with
    // local_tile(), so writes and reads use one consistent swizzled layout.
    Tensor sQFull = make_tensor(
        make_smem_ptr(Qs),
        MakeRowMajorSwizzledLayout<Br, D>()
    );

    Tensor sKFull = make_tensor(
        make_smem_ptr(Ks),
        MakeRowMajorSwizzledLayout<Bc, D>()
    );

    Tensor sPFull = make_tensor(
        make_smem_ptr(P),
        MakeRowMajorSwizzledLayout<Br, Bc>()
    );

    Tensor sVtFull = make_tensor(
        make_smem_ptr(Vs),
        MakeTransposedSwizzledLayout<D, Bc, D>()
    );

    // Tensor Core MMA atom for Ampere:
    //   A: fp16 [M, K]
    //   B: fp16 [N, K]
    //   C: fp32 [M, N]
    //
    // The same tiled MMA shape is used for QK^T and P@V.
    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    auto tiledMma = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _64, _16>{}
    );

    auto thrMma = tiledMma.get_slice(laneId);

    // Warp-local Q and P tiles.
    // Each warp owns 16 query rows and all 64 columns of the tile.
    Tensor sQ = local_tile(
        sQFull,
        make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
        make_coord(warpId, 0)
    );

    Tensor sP = local_tile(
        sPFull,
        make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{}),
        make_coord(warpId, 0)
    );

    // Identity tensors are used to recover logical row/column coordinates
    // of each register element in CuTe accumulator fragments.
    Tensor cS = make_identity_tensor(
        make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{})
    );

    Tensor cO = make_identity_tensor(
        make_shape(Int<ROWS_PER_WARP>{}, Int<D>{})
    );

    // Partition the logical tiles according to the MMA layout.  The resulting
    // tensors describe the registers and shared-memory fragments owned by
    // this lane for the warp-level MMA.
    Tensor tQrQ = thrMma.partition_A(sQ);
    Tensor tKrK = thrMma.partition_B(sKFull);
    Tensor tScS = thrMma.partition_C(cS);

    Tensor tPrP = thrMma.partition_A(sP);
    Tensor tVrV = thrMma.partition_B(sVtFull);
    Tensor tOcO = thrMma.partition_C(cO);

    // Output accumulator for this warp's [16, 64] output tile.
    // It remains in registers across all K/V blocks.
    Tensor accO = thrMma.make_fragment_C(tOcO);
    clear(accO);

    // Load this CTA's Q block once.
    for (int i = tid; i < int(Br * D); i += int(BLOCK_SIZE)) {
        int row = i / int(D);
        int col = i % int(D);
        sQFull(row, col) = qElem[i];
    }

    __syncthreads();

    // Online softmax state for the 16 rows owned by this warp.
    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float alpha[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;
        alpha[r] = 1.0f;
    }

    // Iterate over K/V tiles.
    for (int kvBaseIdx = 0; kvBaseIdx < N; kvBaseIdx += int(Bc)) {
        // Load K as logical [key, d].
        // Load V as logical Vt[d, key], because TN MMA expects B as [N, K].
        for (int i = tid; i < int(Bc * D); i += int(BLOCK_SIZE)) {
            int key = i / int(D);
            int d = i % int(D);

            sKFull(key, d) = kElem[kvBaseIdx * int(D) + i];
            sVtFull(d, key) = vElem[kvBaseIdx * int(D) + i];
        }

        __syncthreads();

        // 1. Scores for this K block:
        //      accS = Q_warp[16,64] @ K_block[64,64]^T
        //
        // accS is a fp32 register tensor.  Raw scores are not written to shared.
        Tensor accS = thrMma.make_fragment_C(tScS);
        clear(accS);

        cute::gemm(tiledMma, tQrQ, tKrK, accS);

        // 2. Blockwise online softmax.
        //
        // Produces fp16 P in shared memory for the following P@V MMA and updates
        // the running softmax statistics m/denom/alpha.
        SoftmaxAccSToPShared<decltype(accS), decltype(tScS), decltype(sP),
                             Element, ROWS_PER_WARP>(
            accS,
            tScS,
            sP,
            scale,
            m,
            denom,
            alpha
        );

        // P is written and read only by the same warp.
        __syncwarp();

        // 3. Online softmax requires the previous output accumulator to be
        // rescaled when the row maximum changes.
        ScaleAccByRow(accO, tOcO, alpha);

        // 4. Accumulate this block's contribution:
        //
        //      accO += P[16,64] @ V[64,64]
        //
        // P is read from shared, V is read as logical Vt[d,key], and the result
        // accumulates directly into fp32 registers.
        cute::gemm(tiledMma, tPrP, tVrV, accO);

        // All warps share K/V buffers, so wait before overwriting them.
        __syncthreads();
    }

    // Normalize the accumulated output:
    //
    //   O[row, :] = accO[row, :] / denom[row]
    NormalizeAccByRow(accO, tOcO, denom);

    // Store this warp's [16,64] output tile.
    Tensor gO = make_tensor(
        make_gmem_ptr(outElem + warpBaseRow * int(D)),
        make_layout(
            make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
            make_stride(Int<D>{}, Int<1>{})
        )
    );

    Tensor tOgO = thrMma.partition_C(gO);

    auto rO = make_fragment_like<Element>(accO);

    #pragma unroll
    for (int i = 0; i < size(accO); ++i) {
        rO(i) = Element(accO(i));
    }

    copy(rO, tOgO);
}


// -----------------------------------------------------------------------------
// Launcher
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
        throw std::runtime_error("This kernel supports only D=64");
    }
    if (N % Br != 0 || N % Bc != 0) {
        throw std::runtime_error(
            "Only N divisible by " + std::to_string(Br) +
            " and " + std::to_string(Bc) + " is supported"
        );
    }
    if (causal) {
        throw std::runtime_error("Causal attention is not supported yet");
    }

    dim3 gridDim(::ceil_div(N, Br), BH);
    dim3 blockSize(BLOCK_SIZE);

    FlashAttnCuteKernel<BLOCK_SIZE, 64, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
        q, k, v, out, BH, N, scale, causal
    );

    check_cuda(cudaGetLastError());
}

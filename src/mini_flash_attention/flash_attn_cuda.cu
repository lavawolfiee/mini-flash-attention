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
// Warp reductions
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
// Shared-memory swizzled layouts.
//
// Important:
//   - These layouts describe the full shared tile.
//   - Warp-local tiles are created by local_tile(full_tensor, ...).
//   - We do NOT create a separate "warp swizzled layout".
//     This keeps all shared writes/reads consistent and reduces layout pressure.
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


// Logical transposed layout for V:
//
//   logical sVt[d, key] = physical V[key, d]
//
// Shape:  [D, Bc]
// Stride: [1, D]
//
// We write V through this tensor and read it through the same tensor.
// This avoids inconsistent "write row-major, read transposed-swizzled" behavior.
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
// cTensor is an identity-coordinate tensor partitioned exactly like accTensor.
// For every register element accTensor(i), cTensor(i) tells us its logical
// coordinate, e.g. (row, col).
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
// Convert accS register scores into:
//   1. row-wise softmax stats update
//   2. P half shared tile for P @ V
//
// accS: register tensor with logical shape [ROWS_PER_WARP, Bc].
// cS:   identity-coordinate tensor partitioned like accS.
// sP:   this warp's shared probability tile [ROWS_PER_WARP, Bc].
//       It already has the correct swizzled layout.
//       Therefore we write sP(row, col), not P[pLayout(...)].
//
// NOTE:
//   This version intentionally keeps alpha[ROWS_PER_WARP], as requested.
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

    // First pass over this thread's accumulator registers:
    // collect local row maxima.
    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));

        float s = float(accS(i)) * scale;
        localMax[row] = fmaxf(localMax[row], s);
    }

    float mNew[ROWS_PER_WARP];

    // Reduce row maxima across the warp.
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

    // Second pass:
    // convert scores into probabilities P in swizzled shared memory.
    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));
        int col = int(get<1>(coord));

        float s = float(accS(i)) * scale;
        float p = __expf(s - mNew[row]);

        // Correct swizzled write:
        // sP is this warp's [16,64] tile obtained from the full swizzled P tensor.
        sP(row, col) = Element(p);

        localDenomAdd[row] += p;
    }

    // Reduce probability sums across the warp.
    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        float denomAdd = ReduceWarpSum(localDenomAdd[r]);
        denom[r] += denomAdd;
        m[r] = mNew[r];
    }
}


// -----------------------------------------------------------------------------
// Full CuTe-QK + CuTe-PV FlashAttention forward prototype.
//
// This version keeps the same algorithmic structure as your current CuTe kernel,
// but fixes swizzling:
//
//   - one full swizzled tensor per shared tile;
//   - warp-local Q/P views are created with local_tile;
//   - P is written through sP(row,col), not raw P[pLayout(...)].
//   - V is written through sVtFull(d,key) and read through the same layout.
//
// Still intentionally keeps:
//   - alpha[ROWS_PER_WARP]
//   - shared P half tile
//   - D=64, Br=64, Bc=64, non-causal only
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

    // Move pointers to the beginning of the working area.
    q += (headIdx * N + qBaseIdx) * int(D);
    out += (headIdx * N + qBaseIdx) * int(D);
    k += headIdx * N * int(D);
    v += headIdx * N * int(D);

    auto qElem = reinterpret_cast<Element const*>(q);
    auto kElem = reinterpret_cast<Element const*>(k);
    auto vElem = reinterpret_cast<Element const*>(v);
    auto outElem = reinterpret_cast<Element*>(out);

    // -------------------------------------------------------------------------
    // Shared memory.
    //
    // No float Scores[64,64].
    // Q/K/V/P use small XOR swizzle in shared memory.
    // -------------------------------------------------------------------------

    __shared__ Element Qs[Br * D];       // [64, 64]
    __shared__ Element Ks[Bc * D];       // [64, 64]
    __shared__ Element Vs[Bc * D];       // storage for logical Vt [D, Bc]
    __shared__ Element P[Br * Bc];       // [64, 64], half probabilities

    // -------------------------------------------------------------------------
    // Full shared tensors with swizzled layouts.
    //
    // These are the only shared layouts we create.
    // We do not create qWarpSwizzledLayout / pWarpSwizzledLayout anymore.
    // -------------------------------------------------------------------------

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

    // Logical Vt:
    //   sVtFull(d, key) = V[key, d]
    Tensor sVtFull = make_tensor(
        make_smem_ptr(Vs),
        MakeTransposedSwizzledLayout<D, Bc, D>()
    );

    // -------------------------------------------------------------------------
    // CuTe MMA setup.
    //
    // SM80_16x8x16_F32F16F16F32_TN:
    //   A: [M,K]
    //   B: [N,K]
    //   C: [M,N]
    // -------------------------------------------------------------------------

    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    auto tiledMma = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _64, _16>{}
    );

    // One warp-level MMA per warp.
    auto thrMma = tiledMma.get_slice(laneId);

    // -------------------------------------------------------------------------
    // Warp-local Q/P tiles.
    //
    // local_tile preserves the full swizzled layout and only creates a logical
    // [16,64] view for this warp.
    // -------------------------------------------------------------------------

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

    // Identity tensors for logical coordinates.
    Tensor cS = make_identity_tensor(
        make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{})
    );

    Tensor cO = make_identity_tensor(
        make_shape(Int<ROWS_PER_WARP>{}, Int<D>{})
    );

    // MMA partitions.
    Tensor tQrQ = thrMma.partition_A(sQ);
    Tensor tKrK = thrMma.partition_B(sKFull);
    Tensor tScS = thrMma.partition_C(cS);

    Tensor tPrP = thrMma.partition_A(sP);
    Tensor tVrV = thrMma.partition_B(sVtFull);
    Tensor tOcO = thrMma.partition_C(cO);

    // Output accumulator lives in registers across the whole K/V loop.
    Tensor accO = thrMma.make_fragment_C(tOcO);
    clear(accO);

    // -------------------------------------------------------------------------
    // Load Q once: global -> swizzled shared.
    // -------------------------------------------------------------------------

    for (int i = tid; i < int(Br * D); i += int(BLOCK_SIZE)) {
        int row = i / int(D);
        int col = i % int(D);

        // Correct swizzled write through the tensor layout.
        sQFull(row, col) = qElem[i];
    }

    __syncthreads();

    // Online softmax stats for this warp's 16 rows.
    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float alpha[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;
        alpha[r] = 1.0f;
    }

    // -------------------------------------------------------------------------
    // Main loop over K/V blocks.
    // -------------------------------------------------------------------------

    for (int kvBaseIdx = 0; kvBaseIdx < N; kvBaseIdx += int(Bc)) {
        // ---------------------------------------------------------------------
        // Load K/V block: global -> swizzled shared.
        //
        // K is written/read as logical [key,d].
        // V is written/read as logical Vt[d,key] = V[key,d].
        // ---------------------------------------------------------------------

        for (int i = tid; i < int(Bc * D); i += int(BLOCK_SIZE)) {
            int key = i / int(D);
            int d = i % int(D);

            sKFull(key, d) = kElem[kvBaseIdx * int(D) + i];
            sVtFull(d, key) = vElem[kvBaseIdx * int(D) + i];
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // 1. Compute accS = Q @ K^T using CuTe MMA.
        //
        // accS is a register tensor with logical shape [16,64].
        // Raw scores never touch shared memory.
        // ---------------------------------------------------------------------

        Tensor accS = thrMma.make_fragment_C(tScS);
        clear(accS);

        cute::gemm(tiledMma, tQrQ, tKrK, accS);

        // ---------------------------------------------------------------------
        // 2. Softmax directly from accS registers.
        //
        // Writes P into this warp's swizzled shared P tile.
        // Keeps alpha[16] as requested.
        // ---------------------------------------------------------------------

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

        // P was written by all lanes of this warp.
        // Other warps write different P rows, so warp sync is enough before
        // this warp reads its own sP.
        __syncwarp();

        // ---------------------------------------------------------------------
        // 3. Rescale old output accumulator in registers.
        // ---------------------------------------------------------------------

        ScaleAccByRow(accO, tOcO, alpha);

        // ---------------------------------------------------------------------
        // 4. accO += P @ V through CuTe MMA.
        // ---------------------------------------------------------------------

        cute::gemm(tiledMma, tPrP, tVrV, accO);

        // All warps must finish reading K/V/P before next iteration overwrites.
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Final normalization in registers.
    // -------------------------------------------------------------------------

    NormalizeAccByRow(accO, tOcO, denom);

    // -------------------------------------------------------------------------
    // Store accO directly to global output.
    // -------------------------------------------------------------------------

    Tensor gO = make_tensor(
        make_gmem_ptr(outElem + warpBaseRow * int(D)),
        make_layout(
            make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
            make_stride(Int<D>{}, Int<1>{})
        )
    );

    Tensor tOgO = thrMma.partition_C(gO);

    // Create a register tensor with the same layout as accO,
    // but with half elements for the final global store.
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
        throw std::runtime_error("This CuTe prototype supports only D=64 yet");
    }
    if (N % Br != 0 || N % Bc != 0) {
        throw std::runtime_error("Only N divisible by " + std::to_string(Br) + " and " + std::to_string(Bc) + " are supported yet");
    }
    if (causal) {
        throw std::runtime_error("Causal attention is not yet supported");
    }

    dim3 gridDim(::ceil_div(N, Br), BH);
    dim3 blockSize(BLOCK_SIZE);

    FlashAttnCuteKernel<BLOCK_SIZE, 64, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
        q, k, v, out, BH, N, scale, causal
    );

    check_cuda(cudaGetLastError());
}

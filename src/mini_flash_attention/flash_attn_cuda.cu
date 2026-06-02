#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
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
// Row-wise operations on CuTe register accumulator tensors.
//
// cTensor is an identity-coordinate tensor partitioned exactly like accTensor.
// So for every register element accTensor(i), cTensor(i) tells us its logical
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
// P:    shared probabilities [Br, Bc], but this warp writes only its 16 rows.
// -----------------------------------------------------------------------------

template <
    typename AccTensor,
    typename CoordTensor,
    typename Element,
    size_t ROWS_PER_WARP,
    size_t Bc,
    size_t Bc_PAD
>
__forceinline__ __device__ void SoftmaxAccSToPShared(
    AccTensor& accS,
    CoordTensor const& cS,
    Element* P,
    size_t warpBaseRow,
    float scale,
    float* m,
    float* denom,
    float* alpha
) {
    float localMax[ROWS_PER_WARP];

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
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
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        float blockMax = ReduceWarpMax(localMax[r]);
        mNew[r] = fmaxf(m[r], blockMax);
        alpha[r] = __expf(m[r] - mNew[r]);
        denom[r] *= alpha[r];
    }

    float localDenomAdd[ROWS_PER_WARP];

    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        localDenomAdd[r] = 0.0f;
    }

    // Second pass:
    // convert scores into probabilities P in shared memory.
    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));
        int col = int(get<1>(coord));

        float s = float(accS(i)) * scale;
        float p = __expf(s - mNew[row]);

        P[(warpBaseRow + row) * Bc_PAD + col] = Element(p);
        localDenomAdd[row] += p;
    }

    // Reduce probability sums across the warp.
    #pragma unroll
    for (size_t r = 0; r < ROWS_PER_WARP; ++r) {
        float denomAdd = ReduceWarpSum(localDenomAdd[r]);
        denom[r] += denomAdd;
        m[r] = mNew[r];
    }
}


// -----------------------------------------------------------------------------
// Full CuTe-QK + CuTe-PV FlashAttention forward prototype.
//
// This version removes shared float Scores[Br, Bc].
// QK scores live in accS register tensor.
// Softmax reads accS directly and writes only P half shared.
// Output accumulator accO lives in registers across all K/V blocks.
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

    // Move pointers to the beginning of the working area.
    q += (headIdx * N + qBaseIdx) * D;
    out += (headIdx * N + qBaseIdx) * D;
    k += headIdx * N * D;
    v += headIdx * N * D;

    auto qElem = reinterpret_cast<Element const*>(q);
    auto kElem = reinterpret_cast<Element const*>(k);
    auto vElem = reinterpret_cast<Element const*>(v);
    auto outElem = reinterpret_cast<Element*>(out);

    // -------------------------------------------------------------------------
    // Shared memory.
    //
    // No float Scores[64,64] anymore.
    // Q/K/V are still simple row-major for readability.
    // P is half shared because this bridge version uses shared P for P@V.
    //
    // Shared memory:
    //   Qs: 8 KB
    //   Ks: 8 KB
    //   Vs: 8 KB
    //   P:  8 KB
    // Total ~32 KB/block instead of ~40 KB/block in the previous Scores version.
    // -------------------------------------------------------------------------

    constexpr size_t D_PAD = 72;
    constexpr size_t Bc_PAD = 72;

    __shared__ Element Qs[Br * D_PAD];       // [64, 64]
    __shared__ Element Ks[Bc * D_PAD];       // [64, 64]
    __shared__ Element Vs[Bc * D_PAD];       // [64, 64]
    __shared__ Element P[Br * Bc_PAD];       // [64, 64], half probabilities

    // -------------------------------------------------------------------------
    // CuTe MMA setup.
    //
    // SM80_16x8x16_F32F16F16F32_TN:
    //   A is logical [M,K]
    //   B is logical [N,K]
    //   C is logical [M,N]
    //
    // We use it twice:
    //   1. accS = Q [16,64] @ K^T [64,64] -> [16,64]
    //      A = Q [M,K]
    //      B = K as [N,K] = [64,64]
    //
    //   2. accO += P [16,64] @ V [64,64] -> [16,64]
    //      A = P [M,K]
    //      B = V as [N,K], i.e. logical Vt[d,key]
    // -------------------------------------------------------------------------

    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    auto tiledMma = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _64, _16>{}
    );

    // We run one warp-level MMA per warp, so slice by laneId.
    auto thrMma = tiledMma.get_slice(laneId);

    // -------------------------------------------------------------------------
    // Warp-local shared views.
    //
    // Q_warp: [16, 64], row-major
    // K:      [64, 64], logical B[N,K], row-major as [key, d]
    // P_warp: [16, 64], row-major
    // Vt:     [64, 64], logical B[N,K], where Vt[d, key] = V[key, d]
    //
    // Important:
    //   V is physically stored as Vs[key, d] row-major.
    //   For B operand in TN MMA we expose it as logical [N,K] = [D,Bc]:
    //     sVt(d, key) -> Vs[key, d]
    // -------------------------------------------------------------------------

    Tensor sQ = make_tensor(
        make_smem_ptr(Qs + warpBaseRow * D_PAD),
        make_layout(
            make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
            make_stride(Int<D_PAD>{}, Int<1>{})
        )
    );

    Tensor sK = make_tensor(
        make_smem_ptr(Ks),
        make_layout(
            make_shape(Int<Bc>{}, Int<D>{}),       // [N,K] for QK
            make_stride(Int<D_PAD>{}, Int<1>{})        // Ks[key, d]
        )
    );

    Tensor sP = make_tensor(
        make_smem_ptr(P + warpBaseRow * Bc_PAD),
        make_layout(
            make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{}),
            make_stride(Int<Bc_PAD>{}, Int<1>{})
        )
    );

    Tensor sVt = make_tensor(
        make_smem_ptr(Vs),
        make_layout(
            make_shape(Int<D>{}, Int<Bc>{}),       // [N,K] for P@V
            make_stride(Int<1>{}, Int<D_PAD>{})        // sVt[d,key] = Vs[key,d]
        )
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
    Tensor tKrK = thrMma.partition_B(sK);
    Tensor tScS = thrMma.partition_C(cS);

    Tensor tPrP = thrMma.partition_A(sP);
    Tensor tVrV = thrMma.partition_B(sVt);
    Tensor tOcO = thrMma.partition_C(cO);

    // Output accumulator lives in registers across the whole K/V loop.
    Tensor accO = thrMma.make_fragment_C(tOcO);
    clear(accO);

    // -------------------------------------------------------------------------
    // Load Q once: global -> shared.
    // -------------------------------------------------------------------------

    for (size_t i = tid; i < Br * D; i += BLOCK_SIZE) {
        size_t row = i / D;
        size_t col = i % D;
        Qs[row * D_PAD + col] = qElem[i];
    }
    __syncthreads();

    // Online softmax stats for this warp's 16 rows.
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
            size_t row = i / D;
            size_t col = i % D;
            Ks[row * D_PAD + col] = kElem[kvBaseIdx * D + i];
            Vs[row * D_PAD + col] = vElem[kvBaseIdx * D + i];
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 1. Compute accS = Q @ K^T using CuTe MMA.
        //
        // accS is a register tensor with logical shape [16,64].
        // This replaces:
        //   WMMA QK -> store sFrag to shared Scores
        //
        // Now raw scores never touch shared memory.
        // ---------------------------------------------------------------------

        Tensor accS = thrMma.make_fragment_C(tScS);
        clear(accS);

        cute::gemm(tiledMma, tQrQ, tKrK, accS);

        // ---------------------------------------------------------------------
        // 2. Softmax directly from accS registers.
        //
        // This writes P half to shared for the following P@V MMA.
        // It also updates m/denom and computes alpha for old accO rescale.
        // ---------------------------------------------------------------------

        SoftmaxAccSToPShared<decltype(accS), decltype(tScS), Element, ROWS_PER_WARP, Bc, Bc_PAD>(
            accS,
            tScS,
            P,
            warpBaseRow,
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
        //
        // This is the online-softmax rescale:
        //   accO[row, :] *= alpha[row]
        //
        // No shared O temp is used.
        // ---------------------------------------------------------------------

        ScaleAccByRow(accO, tOcO, alpha);

        // ---------------------------------------------------------------------
        // 4. accO += P @ V through CuTe MMA.
        //
        // P:  [16,64] from shared
        // Vt: [64,64] logical [N,K] view over Vs[key,d]
        // accO: [16,64] register tensor
        // ---------------------------------------------------------------------

        cute::gemm(tiledMma, tPrP, tVrV, accO);

        // All warps must finish reading Vs before it is overwritten
        // by the next K/V block load.
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Final normalization in registers:
    //   accO[row, :] /= denom[row]
    // -------------------------------------------------------------------------

    NormalizeAccByRow(accO, tOcO, denom);

    // -------------------------------------------------------------------------
    // Store accO directly to global output.
    //
    // Output is [16,64] for this warp.
    // We create a global tensor view and partition it like C.
    // Then we convert fp32 accumulator to half and copy to global.
    // -------------------------------------------------------------------------

    Tensor gO = make_tensor(
        make_gmem_ptr(outElem + warpBaseRow * D),
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

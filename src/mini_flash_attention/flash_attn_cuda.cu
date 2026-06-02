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

template <size_t Rows, size_t Cols>
CUTE_HOST_DEVICE constexpr auto MakeRowMajorSwizzledLayout() {
    return composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<Rows>{}, Int<Cols>{}),
                    make_stride(Int<Cols>{}, Int<1>{}))
    );
}

// Logical Vt[d, key] = V[key, d]. This is the B operand layout for P @ V.
template <size_t Rows, size_t Cols, size_t LeadingDim>
CUTE_HOST_DEVICE constexpr auto MakeTransposedSwizzledLayout() {
    return composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<Rows>{}, Int<Cols>{}),
                    make_stride(Int<1>{}, Int<LeadingDim>{}))
    );
}

// SM80 m16n8k16: reinterpret an accumulator-layout tile [M, N]
// as an A-register-layout tile [M, K] for the following P @ V MMA.
template <typename Layout0>
__forceinline__ __device__ auto ConvertLayoutAccToAregs(Layout0 accLayout) {
    using X = Underscore;
    static_assert(decltype(size<0>(accLayout))::value == 4);
    static_assert(decltype(rank(accLayout))::value == 3);

    auto l = logical_divide(accLayout, Shape<X, X, _2>{});
    return make_layout(
        make_layout(get<0>(l), get<2, 0>(l)),
        get<1>(l),
        get<2, 1>(l)
    );
}

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

// Blockwise online softmax.
// accS is fp32 register scores. rP is fp16 register probabilities.
// P never goes through shared memory.
template <
    typename AccTensor,
    typename CoordTensor,
    typename PTensor,
    size_t ROWS_PER_WARP
>
__forceinline__ __device__ void SoftmaxAccSToPRegs(
    AccTensor& accS,
    CoordTensor const& cS,
    PTensor& rP,
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

    #pragma unroll
    for (int i = 0; i < size(accS); ++i) {
        auto coord = cS(i);
        int row = int(get<0>(coord));

        float s = float(accS(i)) * scale;
        float p = __expf(s - mNew[row]);

        rP(i) = typename PTensor::value_type(p);
        localDenomAdd[row] += p;
    }

    #pragma unroll
    for (int r = 0; r < int(ROWS_PER_WARP); ++r) {
        float denomAdd = ReduceWarpSum(localDenomAdd[r]);
        denom[r] += denomAdd;
        m[r] = mNew[r];
    }
}

// Register/shared GEMM: acc += rA @ sB.
// Keep only one B K-slice in registers at a time to avoid high register pressure.
template <typename TiledMma, typename TensorA, typename TensorB, typename TensorC>
__forceinline__ __device__ void GemmRSmem(
    TiledMma tiledMma,
    TensorA const& rA,
    TensorB const& sB,
    TensorC& acc
) {
    #pragma unroll
    for (int kBlock = 0; kBlock < size<2>(rA); ++kBlock) {
        auto rB = make_fragment_like<typename TensorB::value_type>(sB(_, _, kBlock));
        copy(sB(_, _, kBlock), rB);
        cute::gemm(tiledMma, rA(_, _, kBlock), rB, acc);
    }
}

// FlashAttention forward, D=64, non-causal.
// q/k/v/out are contiguous [BH, N, D].
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
    static_assert(BLOCK_SIZE % 32 == 0);
    static_assert(Br % 16 == 0);
    static_assert(Bc % 16 == 0);
    static_assert(ROWS_PER_WARP == 16);

    int headIdx = int(blockIdx.y);
    int qBaseIdx = int(blockIdx.x) * int(Br);
    int tid = int(threadIdx.x);
    int warpId = tid >> 5;
    int laneId = tid & 31;
    int warpBaseRow = warpId * ROWS_PER_WARP;

    q += (headIdx * N + qBaseIdx) * int(D);
    out += (headIdx * N + qBaseIdx) * int(D);
    k += headIdx * N * int(D);
    v += headIdx * N * int(D);

    auto qElem = reinterpret_cast<Element const*>(q);
    auto kElem = reinterpret_cast<Element const*>(k);
    auto vElem = reinterpret_cast<Element const*>(v);
    auto outElem = reinterpret_cast<Element*>(out);

    // Shared memory holds only Q, K and V/Vt. P stays in registers.
    __shared__ Element Qs[Br * D];
    __shared__ Element Ks[Bc * D];
    __shared__ Element Vs[Bc * D];

    Tensor sQFull = make_tensor(
        make_smem_ptr(Qs),
        MakeRowMajorSwizzledLayout<Br, D>()
    );

    Tensor sKFull = make_tensor(
        make_smem_ptr(Ks),
        MakeRowMajorSwizzledLayout<Bc, D>()
    );

    Tensor sVtFull = make_tensor(
        make_smem_ptr(Vs),
        MakeTransposedSwizzledLayout<D, Bc, D>()
    );

    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    // QK: [16,D] @ [Bc,D]^T -> [16,Bc]
    auto tiledMmaQK = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, Int<Bc>, _16>{}
    );

    // PV: [16,Bc] @ [D,Bc]^T -> [16,D]
    auto tiledMmaPV = make_tiled_mma(
        MmaAtom{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, Int<D>, _16>{}
    );

    auto thrMmaQK = tiledMmaQK.get_slice(laneId);
    auto thrMmaPV = tiledMmaPV.get_slice(laneId);

    Tensor sQ = local_tile(
        sQFull,
        make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
        make_coord(warpId, 0)
    );

    Tensor cS = make_identity_tensor(make_shape(Int<ROWS_PER_WARP>{}, Int<Bc>{}));
    Tensor cO = make_identity_tensor(make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}));

    Tensor tQrQ = thrMmaQK.partition_A(sQ);
    Tensor tKrK = thrMmaQK.partition_B(sKFull);
    Tensor tScS = thrMmaQK.partition_C(cS);

    Tensor tVrV = thrMmaPV.partition_B(sVtFull);
    Tensor tOcO = thrMmaPV.partition_C(cO);

    Tensor accO = thrMmaPV.make_fragment_C(tOcO);
    clear(accO);

    for (int i = tid; i < int(Br * D); i += int(BLOCK_SIZE)) {
        int row = i / int(D);
        int col = i % int(D);
        sQFull(row, col) = qElem[i];
    }

    __syncthreads();

    float m[ROWS_PER_WARP];
    float denom[ROWS_PER_WARP];
    float alpha[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        m[r] = -INFINITY;
        denom[r] = 0.0f;
        alpha[r] = 1.0f;
    }

    for (int kvBaseIdx = 0; kvBaseIdx < N; kvBaseIdx += int(Bc)) {
        for (int i = tid; i < int(Bc * D); i += int(BLOCK_SIZE)) {
            int key = i / int(D);
            int d = i % int(D);

            sKFull(key, d) = kElem[kvBaseIdx * int(D) + i];
            sVtFull(d, key) = vElem[kvBaseIdx * int(D) + i];
        }

        __syncthreads();

        // 1. QK^T scores in fp32 registers.
        Tensor accS = thrMmaQK.make_fragment_C(tScS);
        clear(accS);

        cute::gemm(tiledMmaQK, tQrQ, tKrK, accS);

        // 2. Softmax directly into fp16 register fragment rP.
        auto rP = make_fragment_like<Element>(accS);

        SoftmaxAccSToPRegs<decltype(accS), decltype(tScS),
                           decltype(rP), ROWS_PER_WARP>(
            accS,
            tScS,
            rP,
            scale,
            m,
            denom,
            alpha
        );

        // 3. Online-softmax rescale of old output.
        ScaleAccByRow(accO, tOcO, alpha);

        // 4. Reinterpret rP as an A-register fragment and compute P @ V.
        Tensor tOrP = make_tensor(rP.data(), ConvertLayoutAccToAregs(rP.layout()));

        GemmRSmem(tiledMmaPV, tOrP, tVrV, accO);

        __syncthreads();
    }

    NormalizeAccByRow(accO, tOcO, denom);

    Tensor gO = make_tensor(
        make_gmem_ptr(outElem + warpBaseRow * int(D)),
        make_layout(
            make_shape(Int<ROWS_PER_WARP>{}, Int<D>{}),
            make_stride(Int<D>{}, Int<1>{})
        )
    );

    Tensor tOgO = thrMmaPV.partition_C(gO);

    auto rO = make_fragment_like<Element>(accO);

    #pragma unroll
    for (int i = 0; i < size(accO); ++i) {
        rO(i) = Element(accO(i));
    }

    copy(rO, tOgO);
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
    constexpr size_t D_HEAD = 64;

    // BLOCK_SIZE is derived so that every warp owns 16 query rows.
    constexpr size_t Br = 128;
    constexpr size_t Bc = 128;
    constexpr size_t BLOCK_SIZE = (Br / 16) * 32;

    static_assert(Br % 16 == 0);
    static_assert(BLOCK_SIZE == (Br / 16) * 32);

    constexpr size_t STATIC_SMEM_BYTES =
        (Br * D_HEAD + Bc * D_HEAD + Bc * D_HEAD) * sizeof(cutlass::half_t);

    static_assert(STATIC_SMEM_BYTES <= 48 * 1024,
        "This static-shared version supports up to 48 KiB shared memory. "
        "For larger padded tiles, switch to dynamic shared memory.");

    if (D != int(D_HEAD)) {
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

    FlashAttnCuteKernel<BLOCK_SIZE, D_HEAD, Br, Bc><<<gridDim, blockSize, 0, stream>>>(
        q, k, v, out, BH, N, scale, causal
    );

    check_cuda(cudaGetLastError());
}

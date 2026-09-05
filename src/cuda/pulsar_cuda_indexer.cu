#include "pulsar_cuda_internal.h"

/* The indexer compressed cache is always MXKV-FP4-packed
 * (PULSAR_MXKV_FP4_ROWBYTES(128) = 68 B/row: 64 E2M1 nibble pairs low-nibble-first
 * + 4 E8M0 block-32 scales); the score kernels read packed rows and nothing
 * else.  head_dim must be 128. */


__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            /* Larger value wins; on exact ties prefer the lower index. */
            const bool take_right = (vr > vl) || (vr == vl && ir < il);
            if (take_right) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = sm_idx[0];
}


__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}


__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}


/* Apply one bitonic compare-exchange to a single element.
 *
 * Replicates EXACTLY the shared-memory network in indexer_topk_pow2_kernel:
 * for the pair (low = i, high = i^j) the original writes
 *     swap = desc ? better(b,a) : better(a,b)      (a = low, b = high)
 *     vals[low]  = swap ? b : a
 *     vals[high] = swap ? a : b
 * so the element at `low` takes the partner iff (desc ? better(p,o)
 * : better(o,p)) and the element at `high` takes it iff the mirrored
 * predicate holds.  Both members of a pair share the same `k` bit
 * (j < k), so `desc` can be computed from either index.  No comparison
 * result is reinterpreted, so this is bit-identical for every input,
 * including NaN and +-0. */
__device__ __forceinline__ static void topk_ce_apply(float &ov, uint32_t &oi,
                                                     float pv, uint32_t pi,
                                                     bool desc, bool is_low) {
    const bool take = (is_low == desc) ? topk_score_better(pv, pi, ov, oi)
                                       : topk_score_better(ov, oi, pv, pi);
    if (take) {
        ov = pv;
        oi = pi;
    }
}


/* Bitonic sort of SORT_N (score,index) pairs held in registers, 4 elements per
 * thread, launched with SORT_N/4 threads.  Element i = 4*threadIdx.x + m lives
 * in v[m]/x[m].
 *
 * Same network, same comparator, same pair directions as the all-shared
 * indexer_topk_pow2_kernel loop -- only the data movement changes:
 *   j <  4   partner is in the same thread   -> pure registers
 *   j <  128 partner is in the same warp     -> __shfl_xor_sync
 *   j >= 128 partner is in another warp      -> shared memory (vectorised)
 * That leaves only the j >= 128 stages touching shared memory: 15 of the 78
 * (and 30 of the 78 barriers) for SORT_N 4096, 10 of the 66 for the 2048
 * instantiation -- which is what this kernel is actually bound by.
 *
 * On entry the input must already be in sv[]/sx[]; on exit the sorted result
 * is in the registers (shared memory contents are stale). */
template <uint32_t SORT_N>
__device__ __forceinline__ static void topk_bitonic_sort_regs4(
        float *sv, uint32_t *sx, float (&v)[4], uint32_t (&x)[4]) {
    const uint32_t tid = threadIdx.x;
    float4 *sv4 = reinterpret_cast<float4 *>(sv);
    uint4 *sx4 = reinterpret_cast<uint4 *>(sx);
    {
        const float4 lv = sv4[tid];
        const uint4 lx = sx4[tid];
        v[0] = lv.x; v[1] = lv.y; v[2] = lv.z; v[3] = lv.w;
        x[0] = lx.x; x[1] = lx.y; x[2] = lx.z; x[3] = lx.w;
    }

    #pragma unroll
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        #pragma unroll
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            if (j >= 128u) {
                __syncthreads();
                sv4[tid] = make_float4(v[0], v[1], v[2], v[3]);
                sx4[tid] = make_uint4(x[0], x[1], x[2], x[3]);
                __syncthreads();
                const uint32_t pt = tid ^ (j >> 2u);
                const float4 pv = sv4[pt];
                const uint4 px = sx4[pt];
                const float pva[4] = {pv.x, pv.y, pv.z, pv.w};
                const uint32_t pxa[4] = {px.x, px.y, px.z, px.w};
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], pva[m], pxa[m],
                                  (i & k) == 0u, (i & j) == 0u);
                }
            } else if (j >= 4u) {
                const uint32_t lane_mask = j >> 2u;
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const float pv = __shfl_xor_sync(0xffffffffu, v[m], lane_mask);
                    const uint32_t px = __shfl_xor_sync(0xffffffffu, x[m], lane_mask);
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], pv, px,
                                  (i & k) == 0u, (i & j) == 0u);
                }
            } else {
                const float ov[4] = {v[0], v[1], v[2], v[3]};
                const uint32_t oi[4] = {x[0], x[1], x[2], x[3]};
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const uint32_t mm = m ^ j;
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], ov[mm], oi[mm],
                                  (i & k) == 0u, (i & j) == 0u);
                }
            }
        }
    }
}


__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}


/* The radix key must induce the SAME order as topk_score_better, which
 * compares VALUES: -0.0 == +0.0, and under FTZ (--use_fast_math) a denormal
 * compares equal to zero.  The bit-pattern key does not: it ranks +0.0 above
 * -0.0 and any denormal above +0.0, so on a zero-score tie the CUB kernel and
 * the bitonic kernels selected DIFFERENT rows (L172).  Zero-score ties are the
 * common case here -- every scorer ReLUs each head before the weighted sum,
 * and the FTZ epilogue keeps the sign of a flushed zero.  Canonicalise the
 * whole zero class to +0.0 before packing; `v == 0.0f` is true for -0.0 and,
 * under FTZ, for every denormal.  NaN cannot be ordered by either relation and
 * is not a score any scorer emits; it keeps the key's deterministic rank. */
__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    if (v == 0.0f) v = 0.0f;
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}


__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}


__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}


template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}


template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}


template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) out[i] = x[m];
    }
}


template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) selected[(uint64_t)t * top_k + i] = x[m];
    }
}


template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) dst[i] = x[m];
    }
}


/* Configure indexer_topk_8192_cub_kernel's opt-in dynamic shared memory ONCE and
 * cache the result. The device, its max-optin-smem, and the per-function attribute
 * are process-invariant, so the cudaGetDevice + cudaDeviceGetAttribute +
 * cudaFuncSetAttribute triple must not run per launch (it did, per layer per token
 * at >=16k ctx — squarely on the decode launch floor). Returns the dynamic smem
 * byte count if the cub path is usable on this device, or 0 if the caller must fall
 * back to the pow2 kernel. Decode is single-threaded; a first-call race is benign
 * (cudaFuncSetAttribute is idempotent). */
static int indexer_topk_cub_smem(void) {
    static int cached = -1; /* -1 uninit, 0 unavailable, >0 smem bytes */
    if (cached >= 0) return cached;
    using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
    const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
    int dev = 0, max_optin_smem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&max_optin_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        max_optin_smem < smem ||
        cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
        cached = 0;
        return 0;
    }
    cached = smem;
    return smem;
}


static int indexer_scores_launch(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    /* ONE scorer (L173, L176): the block-scaled MXFP4 tier
     * (src/cuda/pulsar_cuda_indexer_mxfp4.cu) feeds the stored MXFP4 rows
     * straight to the SM120 tensor cores, for every row count.  Everything
     * this file once dispatched around it is gone: the one-row SIMT kernel
     * (L173), and the generic per-(comp,row) kernel with its descriptor
     * (banked) arm -- the engine scores a banked span as same-bank runs
     * through pulsar_gpu_indexer_scores_decode_run_tensor and REFUSES a span
     * whose runs are not consecutive positions (gpu_prefill.cpp), so the
     * descriptor arm's only caller was the smoke test that compared it with
     * itself, and the generic kernel's only remaining shapes were that test's
     * 4-head fixtures.  A shape the tier does not take is refused by name.
     *
     * D5's cross-tier operand unification is structural: Q arrives as the
     * producer's packed E2M1 rows (L090.4), the tier decodes the same bytes
     * the store wrote; no per-entry round-trip.  FIDELITY: the suite-v1 KL
     * run cleared the original flip; tests/idx_quant_fidelity.cc's top-k
     * overlap was the component evidence. */
    const uint64_t comp_bytes = (uint64_t)n_comp * PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        head_dim != 128u ||   /* packed rows are the 68-byte head_dim-128 layout */
        q->bytes < (uint64_t)n_tokens * n_head * PULSAR_MXKV_FP4_ROWBYTES(128u) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < comp_bytes ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        fprintf(stderr, "pulsar: indexer scores rejected: n_tokens=%u n_comp=%u head_dim=%u "
                        "(index_comp %llu B, scores %llu B) -- refusing\n",
                n_tokens, n_comp, head_dim,
                (unsigned long long)(index_comp ? index_comp->bytes : 0ull),
                (unsigned long long)(scores ? scores->bytes : 0ull));
        return 0;
    }
    if (causal && ratio == 0) {
        fprintf(stderr, "pulsar: indexer scores rejected: causal scan with compression ratio 0 -- refusing\n");
        return 0;
    }
    if (n_head != 64u) {
        fprintf(stderr, "pulsar: indexer scores: n_head %u has no kernel (the MXFP4 tier tiles 64 heads) -- refusing\n",
                n_head);
        return 0;
    }
    /* Say so once: this tier changes the numbers, so "did it engage" must be
     * answerable from a log rather than inferred from a timing delta. */
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: indexer scorer = block-scaled MXFP4 tier "
                        "(packed E2M1 Q, consumed natively; every row count)\n");
    }
    return pulsar_gpu_indexer_scores_mxfp4(
            (float *)scores->ptr, (const pulsar_mxkv_pack_t *)q->ptr,
            (const float *)weights->ptr,
            (const pulsar_mxkv_pack_t *)index_comp->ptr,
            n_comp, n_tokens, pos0, n_head, head_dim, ratio, scale,
            causal ? 1 : 0);
}


int pulsar_gpu_indexer_score_one_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}


int pulsar_gpu_indexer_scores_decode_batch_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}


int pulsar_gpu_indexer_scores_decode_run_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *bank_index_comp,
        uint32_t                n_comp,
        uint32_t                run_n,
        uint32_t                run_pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    /* One same-bank consecutive-position run through the block-scaled MXFP4
     * tier (L121).  The tier's causal mask is position-derived, so the bank's
     * comp operand only ever reads rows below (run_pos0+run_n)/ratio -- the
     * bank frontier under the L120 position-truth invariant -- and every
     * masked score in the n_comp-stride row is written -INF without a read.
     * n_comp here is the scores stride / scan bound (the step-top superset),
     * NOT this bank's frontier. */
    const uint32_t vis_max = (run_pos0 + run_n) / ratio;
    if (!scores || !q || !weights || !bank_index_comp ||
        n_comp == 0 || run_n == 0 || ratio == 0 ||
        n_head != 64u || head_dim != 128u ||
        q->bytes < (uint64_t)run_n * n_head * PULSAR_MXKV_FP4_ROWBYTES(128u) ||
        weights->bytes < (uint64_t)run_n * n_head * sizeof(float) ||
        scores->bytes < (uint64_t)run_n * n_comp * sizeof(float) ||
        bank_index_comp->bytes < (uint64_t)vis_max * PULSAR_MXKV_FP4_ROWBYTES(128u)) {
        fprintf(stderr, "pulsar: indexer decode-run scores rejected: run_n=%u run_pos0=%u n_comp=%u visible rows=%u "
                        "(bank index comp %llu B) -- refusing\n", run_n, run_pos0, n_comp, vis_max,
                (unsigned long long)(bank_index_comp ? bank_index_comp->bytes : 0ull));
        return 0;
    }
    return pulsar_gpu_indexer_scores_mxfp4(
            (float *)scores->ptr, (const pulsar_mxkv_pack_t *)q->ptr,
            (const float *)weights->ptr,
            (const pulsar_mxkv_pack_t *)bank_index_comp->ptr,
            n_comp, run_n, run_pos0, n_head, head_dim, ratio, scale, 1);
}


int pulsar_gpu_indexer_topk_tensor(
        pulsar_gpu_tensor       *selected,
        const pulsar_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        fprintf(stderr, "pulsar: indexer top-k rejected: n_tokens=%u n_comp=%u top_k=%u (scores %llu B, selected %llu B) "
                        "-- refusing\n", n_tokens, n_comp, top_k,
                (unsigned long long)(scores ? scores->bytes : 0ull),
                (unsigned long long)(selected ? selected->bytes : 0ull));
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u) {
        if (n_comp == 4096u) {
            const int smem = indexer_topk_cub_smem();
            if (smem > 0) {
                indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                             (const float *)scores->ptr,
                                                                             n_comp, n_tokens, top_k);
                return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            const int smem = indexer_topk_cub_smem();
            if (smem > 0) {
                indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                             (const float *)scores->ptr,
                                                                             n_comp, n_tokens, top_k);
                return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u) {
        /* Chunk width is chosen by batch shape, and ONLY by batch shape.
         *
         * A bitonic sort costs N*log2(N)*(log2(N)+1)/2 compare-exchanges, so
         * two 2048-chunks are cheaper in total work than one 4096-chunk
         * (2*66*2048 = 270k vs 78*4096 = 319k) and they run as two INDEPENDENT
         * blocks instead of one.  Which of those matters depends entirely on
         * n_tokens:
         *
         *   decode  (n_tokens == 1): the grid is (1, n_chunks) -- five 2048-wide
         *     blocks at n_comp 9216 (three with the 4096 chunk) on a 48-SM GPU.
         *     The stage is pure block latency, so halving the
         *     per-block network depth is the whole game.  Measured standalone at
         *     n_comp=9216 (the ctx-32k value): chunk stage 33.3 -> 22.1 us,
         *     whole chain 59.8 -> 50.4 us; in-engine decode_topk 70.2 -> 60.2 us.
         *
         *   prefill (n_tokens >= 2): the grid is (n_tokens, n_chunks) and is
         *     already saturated by token parallelism.  There is no latency to
         *     recover, so narrower chunks buy nothing and merely double the
         *     block count.  Keep 4096 there -- byte-for-byte the pre-existing
         *     prefill configuration, so this change structurally cannot regress
         *     prefill.
         *
         * Widening/narrowing is output-neutral: same comparator, same bitonic
         * network, same "top-512 of each chunk, then merge" reduction.
         * topk_score_better is a strict total order on (score, index) for every
         * NaN-free input (indices are distinct; FTZ makes +-0 and denormals
         * compare equal, and the -INFINITY/UINT32_MAX pads are literally
         * identical elements), so the selected top-512 sequence is uniquely
         * determined and independent of how the input is chunked.  Verified
         * bitwise over 400,896 standalone selections spanning uniform,
         * heavy-tie, all-equal, +-0-only, +-inf, denormal, coarse-quantised and
         * two-valued inputs at n_comp 8193..131072, and over 440,745,984
         * in-engine selections at ctx 36864: 0 differing.  NaN input is the one
         * exception -- NaN compares "not better" in both directions, so the
         * comparator is not an ordering there and NO restructuring can preserve
         * it; a NaN score already makes the selection arbitrary.
         *
         * Either width keeps the final merge inside SORT_N=4096: the tree loop
         * below reduces to at most PULSAR_CUDA_TOPK_MERGE_GROUP(=8) * 512 = 4096
         * candidates. */
        const uint32_t chunk_n = (n_tokens == 1u) ? 2048u : 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > PULSAR_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + PULSAR_CUDA_TOPK_MERGE_GROUP - 1u) / PULSAR_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        if (chunk_n == 2048u) {
            indexer_topk_chunk_pow2_kernel<2048><<<grid_chunks, 512>>>(cur,
                                                                        (const float *)scores->ptr,
                                                                        n_comp,
                                                                        n_tokens,
                                                                        top_k,
                                                                        candidate_stride);
        } else {
            indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                        (const float *)scores->ptr,
                                                                        n_comp,
                                                                        n_tokens,
                                                                        top_k,
                                                                        candidate_stride);
        }
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > PULSAR_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + PULSAR_CUDA_TOPK_MERGE_GROUP - 1u) / PULSAR_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    PULSAR_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}


int pulsar_gpu_argmax_tensor(
        pulsar_gpu_tensor       *out_idx,
        const pulsar_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}


/* ===== MXFP8 (E4M3 + per-32 E8M0) weight matmul via cuBLASLt block-scaling.
 * Hardware MX tensor cores on GB10 (~2x FP16 prefill). Scale tensor uses the
 * Blackwell 128x4 tile swizzle (see blackwell_mx_scale_swizzle). ===== */
cublasLtHandle_t g_cublaslt = NULL;


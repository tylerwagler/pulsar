/* CSA2 candidate pool (L218, DeepSeek-V4.1): the reference's
 * select_candidate_blocks, level one of the two-level top-k.
 *
 * The candidate source's indexer (layer 20) scores every compressed position
 * a query can reach; this pool keeps the `topk_blocks` highest-scoring
 * blocks of `block_size` consecutive positions per query, the block holding
 * the query's newest position pinned in (it is only partly filled and could
 * otherwise be outscored by an older, full block), blocks the query cannot
 * reach at all (score -inf) dropped even when fewer than topk_blocks are
 * reachable.  Every later index source (24/28/32/36) scores with its own
 * weights but only inside the mask -- pulsar_gpu_candidate_mask_scores_tensor
 * puts -inf outside it before that layer's top-k.
 *
 * Selection is by threshold: the k-th largest block score is found with a
 * 4-pass 8-bit radix select over the monotone key of the f32 score, then a
 * block is kept when its score exceeds the threshold, or equals it and lies
 * within the first (k - #greater) such blocks in position order.  torch.topk
 * breaks ties arbitrarily; ours is deterministic and position-first, which is
 * the one place this can differ from the reference (a tie between two blocks'
 * fp32 maxima), by decision.
 *
 * Own translation unit so tests/candidate_kernel_test.cu can #include it and
 * drive the real kernels against the reference math in C. */
#include "pulsar_cuda_internal.h"

/* Monotone f32 -> u32 key: larger float, larger key (sign folded). */
__device__ __forceinline__ static uint32_t cand_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* Per (row, block): the block's max score, the query's newest block pinned to
 * +inf.  scores are [n_rows][n_comp] with -inf already at positions the query
 * cannot reach; vis = the row's reachable count, (pos + 1) / ratio. */
__global__ static void cand_block_max_kernel(float *bscore, const float *scores,
                                             uint32_t n_comp, uint32_t n_blocks, uint32_t block_size,
                                             uint32_t pos0, uint32_t ratio,
                                             const int32_t * __restrict__ positions) {
    const uint32_t row = blockIdx.y;
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint32_t pos = positions ? (uint32_t)positions[row] : pos0 + row;
    uint32_t vis = (pos + 1u) / ratio;
    if (vis > n_comp) vis = n_comp;
    const float *sr = scores + (uint64_t)row * n_comp;
    float m = -INFINITY;
    const uint32_t c0 = b * block_size, c1 = min(n_comp, c0 + block_size);
    for (uint32_t c = c0; c < c1; c++) m = fmaxf(m, sr[c]);
    if (vis != 0u && b == (vis - 1u) / block_size) m = INFINITY;
    bscore[(uint64_t)row * n_blocks + b] = m;
}

/* One 256-thread block per row: radix-select the k-th largest block score,
 * then emit the mask.  bscore is that row's [n_blocks]; mask is [mask_words]
 * u32, bit b set when block b is a candidate. */
__global__ static void cand_select_kernel(uint32_t *mask, const float *bscore,
                                          uint32_t n_blocks, uint32_t mask_words, uint32_t k) {
    const uint32_t row = blockIdx.x;
    const float *bs = bscore + (uint64_t)row * n_blocks;
    uint32_t *mr = mask + (uint64_t)row * mask_words;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix;   /* key prefix fixed so far */
    __shared__ uint32_t s_remain;   /* rank still to resolve within the prefix */
    __shared__ uint32_t s_tie_take; /* ties to keep */
    __shared__ uint32_t s_scan[256];
    const uint32_t tid = threadIdx.x;
    if (k > n_blocks) k = n_blocks;
    if (tid == 0) { s_prefix = 0u; s_remain = k; }
    __syncthreads();
    /* 4 passes, most significant byte first: find the byte of the k-th largest
     * key while keeping the higher bytes fixed. */
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0u;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const uint32_t pmask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint32_t b = tid; b < n_blocks; b += blockDim.x) {
            const uint32_t key = cand_key(bs[b]);
            if ((key & pmask) == (prefix & pmask)) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            /* walk bins from the top; the bin where the running count reaches
             * remain holds the k-th largest key's byte */
            uint32_t remain = s_remain, acc = 0u;
            int bin = 255;
            for (; bin >= 0; bin--) {
                if (acc + hist[bin] >= remain) break;
                acc += hist[bin];
            }
            if (bin < 0) bin = 0;
            s_prefix = prefix | ((uint32_t)bin << shift);
            s_remain = remain - acc;
        }
        __syncthreads();
    }
    const uint32_t thr_key = s_prefix;
    /* count strictly-greater blocks; the rest of k comes from ties, in
     * position order (a block scan over "== thr" flags) */
    uint32_t gt = 0u;
    for (uint32_t b = tid; b < n_blocks; b += blockDim.x) gt += cand_key(bs[b]) > thr_key ? 1u : 0u;
    s_scan[tid] = gt;
    __syncthreads();
    for (uint32_t stride = 1u; stride < 256u; stride <<= 1u) {
        const uint32_t v = tid >= stride ? s_scan[tid - stride] : 0u;
        __syncthreads();
        s_scan[tid] += v;
        __syncthreads();
    }
    if (tid == 0) s_tie_take = k > s_scan[255] ? k - s_scan[255] : 0u;   /* s_scan[255] = blocks strictly above */
    __syncthreads();
    const uint32_t tie_take = s_tie_take;
    /* -inf never qualifies (the reference drops picks that came back -inf) */
    const uint32_t ninf_key = cand_key(-INFINITY);
    for (uint32_t w = tid; w < mask_words; w += blockDim.x) mr[w] = 0u;
    __syncthreads();
    /* ties are numbered in position order with a chunked scan: chunk c holds
     * blocks [c*256, c*256+256) */
    uint32_t ties_before = 0u;
    for (uint32_t c0 = 0; c0 < n_blocks; c0 += 256u) {
        const uint32_t b = c0 + tid;
        const bool in = b < n_blocks;
        const uint32_t key = in ? cand_key(bs[b]) : 0u;
        const uint32_t is_tie = (in && key == thr_key && key != ninf_key) ? 1u : 0u;
        s_scan[tid] = is_tie;
        __syncthreads();
        for (uint32_t stride = 1u; stride < 256u; stride <<= 1u) {
            const uint32_t v = tid >= stride ? s_scan[tid - stride] : 0u;
            __syncthreads();
            s_scan[tid] += v;
            __syncthreads();
        }
        const uint32_t tie_rank = ties_before + s_scan[tid] - is_tie;   /* exclusive */
        const bool keep = in && key != ninf_key &&
                          (key > thr_key || (is_tie && tie_rank < tie_take));
        if (keep) atomicOr(&mr[b >> 5], 1u << (b & 31u));
        ties_before += s_scan[255];
        __syncthreads();
    }
}

/* scores[row][c] = -inf where block c / block_size is outside the row's mask */
__global__ static void cand_mask_scores_kernel(float *scores, const uint32_t *mask,
                                               uint32_t n_comp, uint32_t mask_words, uint32_t block_size) {
    const uint32_t row = blockIdx.y;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_comp) return;
    const uint32_t b = c / block_size;
    const uint32_t bit = mask[(uint64_t)row * mask_words + (b >> 5)] >> (b & 31u);
    if ((bit & 1u) == 0u) scores[(uint64_t)row * n_comp + c] = -INFINITY;
}

uint32_t pulsar_gpu_candidate_mask_words(uint32_t n_comp_cap, uint32_t block_size) {
    if (block_size == 0) return 0;
    const uint32_t n_blocks = (n_comp_cap + block_size - 1u) / block_size;
    return (n_blocks + 31u) / 32u;
}

int pulsar_gpu_candidate_blocks_tensor(pulsar_gpu_tensor *mask, pulsar_gpu_tensor *bscore_scratch,
                                       const pulsar_gpu_tensor *scores,
                                       uint32_t n_comp, uint32_t n_rows, uint32_t mask_words,
                                       uint32_t block_size, uint32_t topk_blocks,
                                       uint32_t pos0, uint32_t ratio, const pulsar_gpu_tensor *positions) {
    if (!mask || !bscore_scratch || !scores || n_comp == 0 || n_rows == 0 || block_size == 0 || topk_blocks == 0 || ratio == 0) {
        fprintf(stderr, "pulsar: candidate blocks: bad arguments -- refusing\n");
        return 0;
    }
    const uint32_t n_blocks = (n_comp + block_size - 1u) / block_size;
    if (mask_words < (n_blocks + 31u) / 32u ||
        mask->bytes < (uint64_t)n_rows * mask_words * sizeof(uint32_t) ||
        bscore_scratch->bytes < (uint64_t)n_rows * n_blocks * sizeof(float) ||
        scores->bytes < (uint64_t)n_rows * n_comp * sizeof(float) ||
        (positions && positions->bytes < (uint64_t)n_rows * sizeof(int32_t))) {
        fprintf(stderr, "pulsar: candidate blocks: operand too small (rows %u, comp %u, blocks %u, mask words %u) -- refusing\n",
                n_rows, n_comp, n_blocks, mask_words);
        return 0;
    }
    dim3 grid((n_blocks + 255u) / 256u, n_rows, 1);
    cand_block_max_kernel<<<grid, 256>>>((float *)bscore_scratch->ptr, (const float *)scores->ptr,
                                         n_comp, n_blocks, block_size, pos0, ratio,
                                         positions ? (const int32_t *)positions->ptr : NULL);
    if (!cuda_ok(cudaGetLastError(), "candidate block-max launch")) return 0;
    cand_select_kernel<<<n_rows, 256>>>((uint32_t *)mask->ptr, (const float *)bscore_scratch->ptr,
                                        n_blocks, mask_words, topk_blocks);
    return cuda_ok(cudaGetLastError(), "candidate select launch");
}

int pulsar_gpu_candidate_mask_scores_tensor(pulsar_gpu_tensor *scores, const pulsar_gpu_tensor *mask,
                                            uint32_t n_comp, uint32_t n_rows, uint32_t mask_words, uint32_t block_size) {
    if (!scores || !mask || n_comp == 0 || n_rows == 0 || block_size == 0 ||
        scores->bytes < (uint64_t)n_rows * n_comp * sizeof(float) ||
        mask->bytes < (uint64_t)n_rows * mask_words * sizeof(uint32_t) ||
        mask_words < ((n_comp + block_size - 1u) / block_size + 31u) / 32u) {
        fprintf(stderr, "pulsar: candidate mask: bad operands (rows %u, comp %u, mask words %u) -- refusing\n",
                n_rows, n_comp, mask_words);
        return 0;
    }
    dim3 grid((n_comp + 255u) / 256u, n_rows, 1);
    cand_mask_scores_kernel<<<grid, 256>>>((float *)scores->ptr, (const uint32_t *)mask->ptr, n_comp, mask_words, block_size);
    return cuda_ok(cudaGetLastError(), "candidate mask launch");
}

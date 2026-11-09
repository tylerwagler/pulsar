/** @file
 *  The CandidateSelector chain head's kernels: score table and walk (L299). */
/* CandidateSelector chain head (L299).  The contract is pulsar_cuda_candsel.h. */
#include "pulsar_cuda_candsel.h"

#include <math.h>
#include <stdio.h>

namespace {

/// @param b bf16 bits.
/// @return the value as f32 (exact).
__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

/// Threads per score-table CTA.
constexpr uint32_t kThreads = 256;

/* The score table: one CTA a (slot, bank) -- the slot's predecessor rows q[i] = pred[p_i] * hp (f32) and its
 * candidates' successor rows (bf16) in shared memory, then every (predecessor, candidate) score.  The slots are
 * independent (slot s's predecessors are slot s - 1's candidates, all of them), so they run side by side. */
/// Grid (slots, n_banks), kThreads threads, k * rank * 6 B dynamic shared memory (q f32 then su bf16, each [k][rank]).
/// @param slots slots per bank.
/// @param k candidates per slot.
/// @param rank codebook width R.
/// @param cand candidate ids, [n_banks][slots][k].
/// @param unary candidate scores, f32 [n_banks][slots][k].
/// @param hp projected pass rows, bf16 [rank] at hp + (b * hp_bank + s) * hp_ld.
/// @param hp_bank hp's bank stride, rows.
/// @param hp_ld hp's row stride, elements.
/// @param anchor each bank's carry id, [n_banks].
/// @param pred predecessor codebook, bf16 [vocab][rank].
/// @param succ successor codebook, bf16 [vocab][rank].
/// @param scores out: f32 [n_banks][slots][k][k].
__global__ void __launch_bounds__(kThreads) candsel_scores_kernel(uint32_t slots, uint32_t k, uint32_t rank,
                                                                  const int32_t *__restrict__ cand,
                                                                  const float *__restrict__ unary,
                                                                  const uint16_t *__restrict__ hp, uint64_t hp_bank,
                                                                  uint64_t hp_ld, const int32_t *__restrict__ anchor,
                                                                  const uint16_t *__restrict__ pred,
                                                                  const uint16_t *__restrict__ succ, float *scores) {
    extern __shared__ __align__(16) unsigned char smem[];
    float *q = reinterpret_cast<float *>(smem);                             /* [k][rank] */
    uint16_t *su = reinterpret_cast<uint16_t *>(q + (size_t)k * rank);      /* [k][rank] */
    const uint32_t s = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const int32_t *cb = cand + (uint64_t)b * slots * k;
    const uint16_t *h = hp + ((uint64_t)b * hp_bank + s) * hp_ld;
    for (uint32_t e = tid; e < k * rank; e += kThreads) {
        const uint32_t i = e / rank, r = e % rank;
        const int32_t p = s == 0 ? anchor[b] : cb[(s - 1) * k + i];
        q[e] = bf(pred[(uint64_t)p * rank + r]) * bf(h[r]);
        su[e] = succ[(uint64_t)cb[s * k + i] * rank + r];
    }
    __syncthreads();
    const float *un = unary + ((uint64_t)b * slots + s) * k;
    float *out = scores + ((uint64_t)b * slots + s) * k * k;
    for (uint32_t e = tid; e < k * k; e += kThreads) {
        const uint32_t i = e / k, c = e % k;
        const float *qi = q + (size_t)i * rank;
        const uint16_t *sc = su + (size_t)c * rank;
        float acc = 0.f;
        for (uint32_t r = 0; r < rank; r++) acc = fmaf(bf(sc[r]), qi[r], acc);
        out[e] = un[c] + acc;
    }
}

/* The walk: one thread a bank, slot by slot the first maximum of the row the previous slot chose. */
/// Grid ceil(n_banks / 32), 32 threads, one thread per bank.
/// @param n_banks banks.
/// @param slots slots per bank.
/// @param k candidates per slot.
/// @param cand candidate ids, [n_banks][slots][k].
/// @param scores the score table, f32 [n_banks][slots][k][k].
/// @param chain_ids out: the chosen ids, [n_banks][slots].
/// @param chain_rows out: the chosen candidate indices, [n_banks][slots].
__global__ void candsel_walk_kernel(uint32_t n_banks, uint32_t slots, uint32_t k, const int32_t *__restrict__ cand,
                                    const float *__restrict__ scores, int32_t *chain_ids, int32_t *chain_rows) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_banks) return;
    uint32_t prev = 0;
    for (uint32_t s = 0; s < slots; s++) {
        const float *rw = scores + (((uint64_t)b * slots + s) * k + prev) * k;
        uint32_t j = 0;
        for (uint32_t c = 1; c < k; c++)
            if (rw[c] > rw[j]) j = c;
        chain_rows[(uint64_t)b * slots + s] = (int32_t)j;
        chain_ids[(uint64_t)b * slots + s] = cand[((uint64_t)b * slots + s) * k + j];
        prev = j;
    }
}

}  // namespace

int pulsar_candsel_launch(uint32_t n_banks, uint32_t slots, uint32_t k, uint32_t rank, uint32_t vocab,
                          const int32_t *cand, const float *unary, const uint16_t *hp, uint64_t hp_bank,
                          uint64_t hp_ld, const int32_t *anchor, const uint16_t *pred, const uint16_t *succ,
                          float *scores, int32_t *chain_ids, int32_t *chain_rows, cudaStream_t stream) {
    if (!cand || !unary || !hp || !anchor || !pred || !succ || !scores || !chain_ids || !chain_rows || n_banks == 0 ||
        n_banks > 65535u || slots == 0 || slots > PULSAR_CANDSEL_SLOTS_MAX || k == 0 || k > PULSAR_CANDSEL_TOPK_MAX ||
        rank == 0 || rank > PULSAR_CANDSEL_RANK_MAX || vocab == 0 || hp_ld < rank || hp_bank < slots) {
        fprintf(stderr, "pulsar: candidate selector refused: a NULL buffer, or banks / slots / k / rank outside the "
                        "kernel's (k <= %u, rank <= %u, slots <= %u)\n", PULSAR_CANDSEL_TOPK_MAX,
                PULSAR_CANDSEL_RANK_MAX, PULSAR_CANDSEL_SLOTS_MAX);
        return -1;
    }
    const size_t sm = (size_t)k * rank * (sizeof(float) + sizeof(uint16_t));
    candsel_scores_kernel<<<dim3(slots, n_banks), kThreads, sm, stream>>>(slots, k, rank, cand, unary, hp, hp_bank,
                                                                          hp_ld, anchor, pred, succ, scores);
    candsel_walk_kernel<<<(n_banks + 31u) / 32u, 32, 0, stream>>>(n_banks, slots, k, cand, scores, chain_ids, chain_rows);
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "pulsar: candidate selector launch failed: %s\n", cudaGetErrorString(e));
    return -3;
}

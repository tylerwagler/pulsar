/* CandidateSelector chain head (L299).  The contract is pulsar_cuda_candsel.h. */
#include "pulsar_cuda_candsel.h"

#include <math.h>
#include <stdio.h>

namespace {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

constexpr uint32_t kThreads = 256;

/* One CTA a bank, its slots in order: the slot's predecessor rows q[i] = pred[p_i] * hp (f32) and its candidates'
 * successor rows (bf16) in shared memory, every (predecessor, candidate) score, then the walk's step at this slot
 * (thread 0: the first maximum of the row the previous slot chose). */
__global__ void __launch_bounds__(kThreads) candsel_kernel(uint32_t slots, uint32_t k, uint32_t rank,
                                                           const int32_t *__restrict__ cand,
                                                           const float *__restrict__ unary,
                                                           const uint16_t *__restrict__ hp, uint64_t hp_bank,
                                                           uint64_t hp_ld, const int32_t *__restrict__ anchor,
                                                           const uint16_t *__restrict__ pred,
                                                           const uint16_t *__restrict__ succ, float *scores,
                                                           int32_t *chain_ids, int32_t *chain_rows) {
    extern __shared__ __align__(16) unsigned char smem[];
    float *q = reinterpret_cast<float *>(smem);                             /* [k][rank] */
    uint16_t *su = reinterpret_cast<uint16_t *>(q + (size_t)k * rank);      /* [k][rank] */
    __shared__ float row[PULSAR_CANDSEL_TOPK_MAX * PULSAR_CANDSEL_TOPK_MAX];
    __shared__ int32_t s_prev;
    const uint32_t b = blockIdx.x, tid = threadIdx.x;
    const int32_t *cb = cand + (uint64_t)b * slots * k;
    if (tid == 0) s_prev = 0;
    for (uint32_t s = 0; s < slots; s++) {
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
            const float v = un[c] + acc;
            out[e] = v;
            row[e] = v;
        }
        __syncthreads();
        if (tid == 0) {
            const float *rw = row + (size_t)s_prev * k;
            uint32_t j = 0;
            for (uint32_t c = 1; c < k; c++)
                if (rw[c] > rw[j]) j = c;
            chain_rows[(uint64_t)b * slots + s] = (int32_t)j;
            chain_ids[(uint64_t)b * slots + s] = cb[s * k + j];
            s_prev = (int32_t)j;
        }
        __syncthreads();
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
    candsel_kernel<<<n_banks, kThreads, sm, stream>>>(slots, k, rank, cand, unary, hp, hp_bank, hp_ld, anchor, pred,
                                                      succ, scores, chain_ids, chain_rows);
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "pulsar: candidate selector launch failed: %s\n", cudaGetErrorString(e));
    return -3;
}

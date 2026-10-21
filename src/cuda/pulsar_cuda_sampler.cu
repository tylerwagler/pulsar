/* pulsar_cuda_sampler.cu -- THE sampler, on the device (L302 step 1).
 *
 * Every lane draws its tokens here: the plain decode, the batched decode, the speculative verify (its walk too,
 * pulsar_gpu_spec_walk), the drafters' sampled drafts.  The logits never leave the device; a round reads back a few
 * bytes a row.  The rule is the one the host sampler (tokenizer.cpp until L302) defined, kept BYTE FOR BYTE:
 *
 *   - a row's value at id i is its logit, or -inf where the row's token mask (L303's hook; L272's tool-name mask
 *     rides it) clears bit i; a non-finite value is no candidate;
 *   - greedy (temperature <= 0, or the row's GREEDY flag -- the toolsample positions) is the argmax: the lowest id
 *     whose value equals the row's finite maximum (IEEE ==, so +0 and -0 tie), -1 with no finite value;
 *   - otherwise the nucleus: candidates in descending value, ascending id on ties (-0 is +0); probabilities
 *     p = expf((v - max) * (1 / T)) -- the host's -ffast-math build multiplies by the reciprocal, so this does; the
 *     candidate set is the top-k (k <= 1024), or every finite value (min_p tiny), or the min-p prefilter's superset
 *     (min_p > 1e-30; full-nucleus arm when top_p == 1 and the sparse predicate holds); `sum` is the host's serial
 *     float sum -- over the candidates in sorted order, or (prefilter arm) over every finite value in id order, or
 *     +inf (full-nucleus arm: the top-p test is inert); the cut walks the sorted candidates: stop before one under
 *     p0 * min_p (never the first), add it, stop after the sum reaches top_p (filtered * (1 / sum) >= top_p);
 *     emitted probabilities p * (1 / filtered);
 *   - the draw at the row's key: an exponential race, u = hash(key, id) in (0, 1), score -log(u) / p in double,
 *     the first strict minimum in the emitted order; p <= 0 never runs.
 *
 * The transcendental functions are the host libm's, transcribed (pulsar_cuda_libm.cuh), and every float operation
 * is an explicit _rn intrinsic; this TU builds without --use_fast_math (Makefile), so subnormal probabilities
 * survive as they did on the host.  sampler-device-gate holds the device to a frozen host reference
 * (tests/sampler_host_ref.cpp, the host sampler as it was, built with the production flags) over every mode.
 *
 * Shape: one block per row (rows run on separate SMs) after the prep stage's scans spread over the GPU.  A row whose
 * candidate set exceeds the block's shared-memory capacity (a near-flat min-p row) is finished by the slot kernel on
 * a global scratch slot -- the same row function, a stable radix sort in global memory instead of the shared
 * bitonic: the same order (the key is total), so the same bytes.  The full arm (min_p <= 1e-30 without top-k: every
 * finite value a candidate) is known from the knobs, so the launch routes those rows to their own pipeline: the
 * device radix sort, the probabilities, the sum, the cut and the race over the GPU.
 *
 * The host's long serial sums (the id-order sum over a whole row, the full arm's sorted one, the logprob scorer's
 * double one) keep their bits but not their seriality: block_serial_sum runs them as integer scans on the running
 * sum's ulp grid, and only the adds that leave a binade or round a tie run one at a time. */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_libm.cuh"

#include <cub/device/device_radix_sort.cuh>

namespace {

constexpr uint32_t SMP_THREADS = 1024;   /* the row kernel's block */
constexpr uint32_t SLOT_THREADS = 512;   /* the slot kernel's block (its radix count matrix is 16 x 512 words) */
constexpr uint32_t SMP_CAP = 2048;       /* candidates the row kernel sorts in shared memory */
constexpr uint32_t SMP_SLOTS = PULSAR_GPU_SAMPLE_SLOTS;

constexpr float MINP_PREFILTER_SLACK = 1.0f - 4e-6f;
constexpr float MINP_PREFILTER_MIN = 1e-30f;
constexpr float SPARSE_MINP_MIN = 0.02f;

enum : uint32_t { PATH_TOPK = 0, PATH_MINP = 1, PATH_FULL = 2 };

__host__ __device__ __forceinline__ uint32_t f32_bits(float v) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(v);
#else
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return u;
#endif
}
__host__ __device__ __forceinline__ bool finite_bits(float v) { return (f32_bits(v) & 0x7f800000u) != 0x7f800000u; }

/* descending value, -0 as +0: unsigned ascending order of the key is the candidates' order (ids break ties) */
__device__ __forceinline__ uint32_t desc_key(float f) {
    uint32_t u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;
    const uint32_t asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ~asc;
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* the race score of candidate `id` at probability p (p > 0) */
__device__ __forceinline__ double race_score(uint64_t key, int32_t id, float p) {
    const uint64_t h = mix64(key ^ ((uint64_t)(uint32_t)id * 0xd1b54a32d192ed03ull));
    const double u = __dmul_rn(__dadd_rn((double)(h >> 11), 0.5), 1.0 / 9007199254740992.0);
    return __ddiv_rn(-plm_log(u), (double)p);
}

struct row_view {
    const float *x;
    const uint32_t *mask;   /* the row's mask words, or NULL */
    uint32_t n;
    __device__ __forceinline__ float operator()(uint32_t i) const {
        float v = x[i];
        if (mask && !((mask[i >> 5] >> (i & 31u)) & 1u)) v = __uint_as_float(0xff800000u);
        return v;
    }
};

/* block-wide reductions over THREADS threads; `red` is THREADS-wide shared scratch */
template <uint32_t THREADS>
__device__ void block_argmax(const row_view &rv, float *redv, int32_t *redi, float *out_v, int32_t *out_i) {
    const uint32_t tid = threadIdx.x;
    float bv = 0.0f;
    int32_t bi = -1;
    for (uint32_t i = tid; i < rv.n; i += THREADS) {
        const float v = rv(i);
        if (!finite_bits(v)) continue;
        if (bi < 0 || v > bv) { bv = v; bi = (int32_t)i; }
    }
    redv[tid] = bv;
    redi[tid] = bi;
    __syncthreads();
    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = redv[tid + s], vl = redv[tid];
            const int32_t ir = redi[tid + s], il = redi[tid];
            if (ir >= 0 && (il < 0 || vr > vl || (vr == vl && ir < il))) { redv[tid] = vr; redi[tid] = ir; }
        }
        __syncthreads();
    }
    *out_v = redv[0];
    *out_i = redi[0];
    __syncthreads();
}

template <uint32_t THREADS>
__device__ uint32_t block_sum_u32(uint32_t v, uint32_t *red) {
    const uint32_t tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const uint32_t r = red[0];
    __syncthreads();
    return r;
}

/* inclusive scan of one value per thread (Hillis-Steele over shared) */
template <uint32_t THREADS>
__device__ uint32_t block_scan_incl(uint32_t v, uint32_t *red) {
    const uint32_t tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (uint32_t off = 1; off < THREADS; off <<= 1) {
        const uint32_t t = tid >= off ? red[tid - off] : 0u;
        __syncthreads();
        red[tid] += t;
        __syncthreads();
    }
    const uint32_t r = red[tid];
    __syncthreads();
    return r;
}

/* block argmin over (score, index): the smallest score, the lowest index among equal scores -- the first strict
 * minimum of the host's in-order loop.  Every thread receives the winner. */
template <uint32_t THREADS>
__device__ void block_argmin(double *s, uint32_t *i, double *wd, uint32_t *wi) {
    double v = *s;
    uint32_t x = *i;
    for (uint32_t off = 16; off > 0; off >>= 1) {
        const double vo = __shfl_down_sync(0xffffffffu, v, off);
        const uint32_t xo = __shfl_down_sync(0xffffffffu, x, off);
        if (vo < v || (vo == v && xo < x)) { v = vo; x = xo; }
    }
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (lane == 0) { wd[warp] = v; wi[warp] = x; }
    __syncthreads();
    if (warp == 0) {
        v = lane < THREADS / 32u ? wd[lane] : __longlong_as_double(0x7ff0000000000000ll);
        x = lane < THREADS / 32u ? wi[lane] : UINT32_MAX;
        for (uint32_t off = 16; off > 0; off >>= 1) {
            const double vo = __shfl_down_sync(0xffffffffu, v, off);
            const uint32_t xo = __shfl_down_sync(0xffffffffu, x, off);
            if (vo < v || (vo == v && xo < x)) { v = vo; x = xo; }
        }
        if (lane == 0) { wd[0] = v; wi[0] = x; }
    }
    __syncthreads();
    *s = wd[0];
    *i = wi[0];
    __syncthreads();
}

/* bitonic sort of keys[0, m) ascending in shared memory (m <= SMP_CAP), padded to a power of two with ~0 */
template <uint32_t THREADS>
__device__ void bitonic_sort(uint64_t *keys, uint32_t m) {
    uint32_t p2 = 1;
    while (p2 < m) p2 <<= 1;
    for (uint32_t i = m + threadIdx.x; i < p2; i += THREADS) keys[i] = ~0ull;
    __syncthreads();
    for (uint32_t k = 2; k <= p2; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < p2; i += THREADS) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    const uint64_t a = keys[i], b = keys[l];
                    const bool up = (i & k) == 0;
                    if (up ? a > b : a < b) { keys[i] = b; keys[l] = a; }
                }
            }
            __syncthreads();
        }
    }
}

/* The k (1 <= k <= 1024, k <= the finite count) smallest (desc key, id) of the row, sorted, into keys[0, k): the
 * k-th key K* by an 8-bit radix select, every key below it, and the lowest ids of K*'s tie class -- exactly the k the
 * host's insertion (strict '>' / '<', earlier ids first) keeps, in its order. */
template <uint32_t THREADS>
__device__ void select_topk(const row_view &rv, uint32_t k, uint64_t *keys, uint32_t *hist, uint32_t *cnt,
                            uint32_t *redu) {
    const uint32_t tid = threadIdx.x;
    uint32_t prefix = 0, want = k;
    for (int pass = 3; pass >= 0; pass--) {
        const uint32_t sh = (uint32_t)pass * 8u;
        const uint32_t hmask = pass == 3 ? 0u : (0xffffffffu << (sh + 8u));
        for (uint32_t d = tid; d < 256; d += THREADS) hist[d] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < rv.n; i += THREADS) {
            const float v = rv(i);
            if (!finite_bits(v)) continue;
            const uint32_t dk = desc_key(v);
            if ((dk & hmask) == (prefix & hmask)) atomicAdd(&hist[(dk >> sh) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t c = 0, d = 0;
            while (c + hist[d] < want) c += hist[d++];
            *cnt = d | (c << 8);   /* the digit, and how many of the class sit below it (< 2^24) */
        }
        __syncthreads();
        const uint32_t d = *cnt & 255u, below = *cnt >> 8;
        prefix |= d << sh;
        want -= below;
        __syncthreads();
    }
    const uint32_t kstar = prefix;   /* `want` of its tie class are kept, the lowest ids first */
    if (tid == 0) *cnt = 0;
    __syncthreads();
    for (uint32_t i = tid; i < rv.n; i += THREADS) {
        const float v = rv(i);
        if (!finite_bits(v)) continue;
        const uint32_t dk = desc_key(v);
        if (dk < kstar) keys[atomicAdd(cnt, 1u)] = ((uint64_t)dk << 32) | i;
    }
    __syncthreads();
    const uint32_t nless = *cnt;
    const uint32_t chunk = (rv.n + THREADS - 1u) / THREADS;
    const uint32_t lo = min(tid * chunk, rv.n), hi = min(lo + chunk, rv.n);
    uint32_t c = 0;
    for (uint32_t i = lo; i < hi; i++) {
        const float v = rv(i);
        c += finite_bits(v) && desc_key(v) == kstar ? 1u : 0u;
    }
    uint32_t pos = block_scan_incl<THREADS>(c, redu) - c;
    for (uint32_t i = lo; i < hi && pos < want; i++) {
        const float v = rv(i);
        if (finite_bits(v) && desc_key(v) == kstar) keys[nless + pos++] = ((uint64_t)kstar << 32) | i;
    }
    __syncthreads();
    bitonic_sort<THREADS>(keys, k);
}

/* What the row function writes per row (the host reads the token; the gate also the dist). */
typedef pulsar_gpu_sample_out row_out_t;

/* Per-launch arguments (by value). */
struct launch_args {
    const float *logits;
    uint64_t row_stride;       /* floats */
    uint32_t n_vocab;
    uint32_t n_rows;
    const uint32_t *mask;      /* [mask rows][mask_words] or NULL */
    uint32_t mask_words;
    const int32_t *id_map;     /* [n_vocab]: row index -> token id (a drafter's draft vocabulary), or NULL */
    pulsar_gpu_sample_req req[PULSAR_GPU_SAMPLE_ROWS_MAX];
    row_out_t *out;            /* [n_rows] */
    uint32_t *pending;         /* [n_rows]: 1 = the row kernel left it to a slot */
    uint64_t full_rows;        /* bit r: row r is on the full arm (its own pipeline, not the row kernel's) */
    uint32_t plan_vocab;       /* the vocabulary the knobs' plan reads (the row's width; a chain slot's sparse row's) */
    float *pvals;              /* [n_rows][n_vocab]: the id-order sum's values, computed over the GPU, or NULL */
    uint64_t idsum_rows;       /* bit r: row r's id-order sum is idsums[r] (sr_sum_kernel's) */
    const double *idsums;
    /* slot scratch: per slot keys [2][n_vocab] u32, ids [2][n_vocab] u32, probs [n_vocab] f32, then (aligned) the
     * slot kernel's radix counts or the full arm's state and sort scratch */
    uint8_t *slots;
    uint64_t slot_bytes;
    /* the gate's dist dump: per row [n, ids[dist_cap], probs[dist_cap]] (u32 words), or NULL */
    uint32_t *dist;
    uint32_t dist_cap;
};

/* block_serial_sum's shared state */
template <typename T>
struct serial_sum_smem {
    unsigned long long wsum[32];
    uint32_t wmin[32];
    T s;
    uint32_t start, stop, nan;
};

/* The shared memory of a row: the THREADS-wide scratch and (row kernel) the candidate store. */
template <uint32_t THREADS>
struct row_smem {
    float redv[THREADS];
    int32_t redi[THREADS];
    uint32_t redu[THREADS];
    uint64_t keys[SMP_CAP];
    float probs[SMP_CAP];
    serial_sum_smem<float> ssum;
    double redd[32];
    uint32_t redw[32];
    uint32_t hist[256];
    uint32_t cnt;
    float fsum;
    uint32_t filtered;
    int32_t win;
};

/* A row's candidate list, wherever it lives: sorted (key, id) and its probabilities in sorted order. */
struct cand_list {
    uint32_t n;
    uint64_t *skeys;     /* shared: (desc_key << 32 | id) sorted */
    uint32_t *gids;      /* global: ids in sorted order (slot path) */
    float *probs;        /* shared or global, sorted order */
    const int32_t *map;  /* the row's ids -> the draw's ids (a drafter's draft vocabulary), or NULL */
    __device__ __forceinline__ int32_t id(uint32_t i) const {
        const int32_t r = skeys ? (int32_t)(uint32_t)skeys[i] : (int32_t)gids[i];
        return map ? map[r] : r;
    }
};

/* Stable LSD radix sort of (key, id) pairs in global memory by key, 4-bit digits, one block: each thread owns a
 * contiguous chunk, so counting per (digit, thread) and scattering each chunk in order keeps the sort stable. */
template <uint32_t THREADS>
__device__ void global_radix_sort(uint32_t *k0, uint32_t *v0, uint32_t *k1, uint32_t *v1, uint32_t n, uint32_t *cnt) {
    const uint32_t tid = threadIdx.x;
    const uint32_t chunk = (n + THREADS - 1u) / THREADS;
    const uint32_t lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    for (uint32_t pass = 0; pass < 8; pass++) {
        const uint32_t sh = pass * 4u;
        uint32_t c[16];
        for (uint32_t d = 0; d < 16; d++) c[d] = 0;
        for (uint32_t i = lo; i < hi; i++) c[(k0[i] >> sh) & 15u]++;
        for (uint32_t d = 0; d < 16; d++) cnt[d * THREADS + tid] = c[d];
        __syncthreads();
        /* exclusive scan over (digit, thread) in digit-major order: one thread per digit row, then the row bases */
        if (tid < 16) {
            uint32_t s = 0;
            for (uint32_t t = 0; t < THREADS; t++) {
                const uint32_t v = cnt[tid * THREADS + t];
                cnt[tid * THREADS + t] = s;
                s += v;
            }
            cnt[16 * THREADS + tid] = s;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t s = 0;
            for (uint32_t d = 0; d < 16; d++) {
                const uint32_t v = cnt[16 * THREADS + d];
                cnt[16 * THREADS + d] = s;
                s += v;
            }
        }
        __syncthreads();
        for (uint32_t d = 0; d < 16; d++) c[d] = cnt[16 * THREADS + d] + cnt[d * THREADS + tid];
        for (uint32_t i = lo; i < hi; i++) {
            const uint32_t d = (k0[i] >> sh) & 15u;
            const uint32_t o = c[d]++;
            k1[o] = k0[i];
            v1[o] = v0[i];
        }
        __syncthreads();
        uint32_t *t = k0; k0 = k1; k1 = t;
        t = v0; v0 = v1; v1 = t;
    }
    /* 8 passes: the result is back in the first buffers */
}

/* ---- the prep stage: the row's scans spread over the GPU ---------------------------------------------------------
 *
 * One block a row reads a 1 MB row at one SM's bandwidth, once for the argmax and once or twice more for the
 * candidates (four more for a top-k's radix select).  The prep kernels do those passes with PREP_CHUNKS blocks a row:
 * a scan (each chunk's first maximum and finite count; a top-k row's first radix histogram), the top-k row's two
 * further radix levels (each after a decide kernel that fixes the level's digit), and the collection (the row's
 * maximum reduced from the chunks', then every candidate appended to the row's list -- the order is free: the row
 * kernel sorts by (desc key, id), a total order).  The row kernel then starts from the maximum and the list; a list
 * past its shared store, and the arm that needs a serial sum over the whole row in id order, keep their own passes.
 * The arithmetic is the row kernel's own (the same is_cand, the same expf): the candidates are the same set. */

constexpr uint32_t PREP_CHUNKS = 32;
constexpr uint32_t PREP_THREADS = 256;
constexpr uint32_t HIST1 = 2048, HIST2 = 2048, HIST3 = 1024;   /* desc key bits 31..21, 20..10, 9..0 */

/* A row's prep record (scratch, zeroed through `zero_end` before every launch). */
struct row_prep {
    uint32_t n_cand;            /* candidates appended (past SMP_CAP: the list overflowed) */
    uint32_t d1, d2, d3;        /* the top-k row's radix digits, fixed by the decide kernels */
    uint32_t want1, want2, want3;
    uint32_t hist[HIST1 + HIST2 + HIST3];
    uint32_t zero_end;          /* everything above is zeroed per launch */
    float maxv;
    int32_t maxi;
    uint32_t n_fin;
    float pmax[PREP_CHUNKS];
    int32_t pidx[PREP_CHUNKS];
    uint32_t pfin[PREP_CHUNKS];
    uint64_t cand[SMP_CAP];
};

/* A row's knobs as the sampler reads them, and its path. */
struct row_plan {
    bool refuse, greedy;
    uint32_t path;
    float T, top_p, min_p, rcpT, prefilter;
    int top_k;
    bool full;
};

/* the knobs clamped and the path -- comparisons only, so the launch (host) routes the full arm by the same rule */
__host__ __device__ __forceinline__ void plan_knobs(const pulsar_gpu_sample_req &q, row_plan *pl) {
    row_plan &p = *pl;
    p.greedy = (q.flags & PULSAR_GPU_SAMPLE_GREEDY) != 0;
    p.T = q.temperature;
    p.top_p = q.top_p;
    p.min_p = q.min_p;
    p.top_k = q.top_k;
    p.refuse = !p.greedy && (!finite_bits(p.T) || !finite_bits(p.top_p) || !finite_bits(p.min_p));
    p.greedy = p.greedy || (!p.refuse && p.T <= 0.0f);
    if (p.top_p <= 0.0f || p.top_p > 1.0f) p.top_p = 1.0f;
    if (p.min_p < 0.0f) p.min_p = 0.0f;
    if (p.top_k <= 0 || p.top_k > 1024) p.top_k = p.top_k <= 0 ? 0 : 1024;
    p.path = p.top_k > 0 ? PATH_TOPK : p.min_p > MINP_PREFILTER_MIN ? PATH_MINP : PATH_FULL;
}

/* a row the full arm's pipeline draws */
__host__ __device__ __forceinline__ bool row_on_full_arm(const pulsar_gpu_sample_req &q) {
    row_plan p;
    plan_knobs(q, &p);
    return !p.refuse && !p.greedy && p.path == PATH_FULL;
}

/* THE full-nucleus predicate of the min-p arm, min_p >= max(0.02, 2.5 * 2^-24 * n_vocab) with top_p == 1 (plan_row
 * has the argument); a product and comparisons, the same on the host (the launch routes by it) */
__host__ __device__ __forceinline__ bool minp_full_nucleus(const row_plan &p, uint32_t n_vocab) {
#ifdef __CUDA_ARCH__
    const float band = __fmul_rn(1.4901161193847656e-7f, (float)n_vocab);
#else
    const float band = 1.4901161193847656e-7f * (float)n_vocab;
#endif
    return p.top_p == 1.0f && p.min_p >= (band > SPARSE_MINP_MIN ? band : SPARSE_MINP_MIN);
}

__device__ __forceinline__ row_plan plan_row(const pulsar_gpu_sample_req &q, uint32_t n_vocab) {
    row_plan p;
    plan_knobs(q, &p);
    p.rcpT = p.refuse || p.greedy ? 0.0f : __fdiv_rn(1.0f, p.T);
    p.prefilter = 0.0f;
    p.full = false;
    if (p.path == PATH_MINP) {
        p.prefilter = __fmul_rn(p.min_p, MINP_PREFILTER_SLACK);
        /* L295 F7: THE sparse min-p predicate, min_p >= max(0.02, 2.5 * 2^-24 * n_vocab).  With top_p == 1 the cut's
         * `filtered * (1 / sum) >= 1` test is redundant with the min-p cut: while a further survivor exists, sum >=
         * filtered + min_p (every survivor's p >= min_p times the max's, which is expf(0) == 1), so the ratio stays at
         * least min_p / (filtered + min_p) under 1, which clears the rounding band (2 * 2^-24) for filtered <= n_vocab
         * when min_p > 2 * 2^-24 * n_vocab; the 2.5 is a 25% margin.  So the arm drops the sum (+inf) and the expf
         * below the floor. */
        p.full = minp_full_nucleus(p, n_vocab);
    }
    return p;
}

/* a row whose sum is the id-order serial sum over its whole vocabulary (the min-p arm off its full nucleus): the
 * launch runs those sums over the GPU before the row kernel (sr_sum_kernel) */
__host__ __device__ __forceinline__ bool row_idsum(const pulsar_gpu_sample_req &q, uint32_t n_vocab) {
    row_plan p;
    plan_knobs(q, &p);
    return !p.refuse && !p.greedy && p.path == PATH_MINP && !minp_full_nucleus(p, n_vocab);
}

/* THE candidate predicate of the min-p and full arms (a top-k row's is its key against K*) -- the row kernel's and
 * the prep's (the full arm's rows of a launch take their own pipeline; a chain's slot runs the arm in the block).  floor: the full-nucleus arm's bound (an entry under it has
 * p < prefilter however it rounds), else -inf; *pp receives p where it was computed. */
__device__ __forceinline__ bool row_is_cand(const row_plan &p, uint32_t i, float v, float floor_logit, float maxlogit,
                                            uint32_t max_id, float *pp) {
    if (!finite_bits(v)) return false;
    if (p.path == PATH_FULL) return true;
    if (v < floor_logit && i != max_id) return false;
    const float pr = plm_expf(__fmul_rn(__fsub_rn(v, maxlogit), p.rcpT));
    *pp = pr;
    return pr >= p.prefilter || i == max_id;
}

__device__ __forceinline__ float row_floor(const row_plan &p, float maxv) {
    return p.path == PATH_MINP && p.full ? __fmaf_rn(__fsub_rn(logf(p.min_p), 1e-3f), p.T, maxv)
                                         : __uint_as_float(0xff800000u);
}

__device__ __forceinline__ row_view prep_view(const launch_args &a, const pulsar_gpu_sample_req &q) {
    row_view rv;
    rv.x = a.logits + (size_t)q.row * a.row_stride;
    rv.n = a.n_vocab;
    rv.mask = a.mask && q.mask_row != PULSAR_GPU_SAMPLE_NO_MASK ? a.mask + (size_t)q.mask_row * a.mask_words : nullptr;
    return rv;
}

/* chunk c of a row: [lo, hi) */
__device__ __forceinline__ void prep_chunk(uint32_t n, uint32_t c, uint32_t *lo, uint32_t *hi) {
    const uint32_t per = (n + PREP_CHUNKS - 1u) / PREP_CHUNKS;
    *lo = min(c * per, n);
    *hi = min(*lo + per, n);
}

/* chunk c of gridDim.x (the passes that keep no per-chunk record run wider) */
__device__ __forceinline__ void grid_chunk(uint32_t n, uint32_t c, uint32_t *lo, uint32_t *hi) {
    const uint32_t per = (n + gridDim.x - 1u) / gridDim.x;
    *lo = min(c * per, n);
    *hi = min(*lo + per, n);
}
constexpr uint32_t WIDE_CHUNKS = 128;   /* the wide passes' blocks a row */

/* scan: the chunk's first maximum and finite count; a top-k row's first histogram */
__global__ void __launch_bounds__(PREP_THREADS) prep_scan_kernel(const __grid_constant__ launch_args a, row_prep *prep) {
    __shared__ float rv_[PREP_THREADS];
    __shared__ int32_t ri_[PREP_THREADS];
    __shared__ uint32_t ru_[PREP_THREADS];
    const uint32_t row = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    const row_plan p = plan_row(q, a.n_vocab);
    row_prep *pr = prep + row;
    const row_view rv = prep_view(a, q);
    uint32_t lo, hi;
    prep_chunk(rv.n, c, &lo, &hi);
    const bool hist = !p.refuse && !p.greedy && p.path == PATH_TOPK;
    float bv = 0.0f;
    int32_t bi = -1;
    uint32_t fin = 0;
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
        const float v = rv(i);
        if (!finite_bits(v)) continue;
        fin++;
        if (bi < 0 || v > bv) { bv = v; bi = (int32_t)i; }
        if (hist) atomicAdd(&pr->hist[desc_key(v) >> 21], 1u);
    }
    rv_[tid] = bv;
    ri_[tid] = bi;
    ru_[tid] = fin;
    __syncthreads();
    for (uint32_t s = PREP_THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = rv_[tid + s], vl = rv_[tid];
            const int32_t ir = ri_[tid + s], il = ri_[tid];
            if (ir >= 0 && (il < 0 || vr > vl || (vr == vl && ir < il))) { rv_[tid] = vr; ri_[tid] = ir; }
            ru_[tid] += ru_[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        pr->pmax[c] = rv_[0];
        pr->pidx[c] = ri_[0];
        pr->pfin[c] = ru_[0];
    }
}

/* the row's first maximum and finite count from its chunks' (the block argmax's rule: a larger value, or an equal one
 * at a lower id) */
__device__ __forceinline__ void prep_reduce_max(const row_prep *pr, float *maxv, int32_t *maxi, uint32_t *n_fin) {
    float bv = 0.0f;
    int32_t bi = -1;
    uint32_t fin = 0;
    for (uint32_t c = 0; c < PREP_CHUNKS; c++) {
        const float v = pr->pmax[c];
        const int32_t i = pr->pidx[c];
        fin += pr->pfin[c];
        if (i >= 0 && (bi < 0 || v > bv || (v == bv && i < bi))) { bv = v; bi = i; }
    }
    *maxv = bv;
    *maxi = bi;
    *n_fin = fin;
}

/* decide level L of a top-k row's radix select: the digit whose cumulative count reaches the keys still wanted
 * (the k-th smallest key's), and how many of the level's class sit below it */
template <uint32_t L>
__global__ void __launch_bounds__(PREP_THREADS) prep_decide_kernel(const __grid_constant__ launch_args a, row_prep *prep) {
    __shared__ uint32_t sc[PREP_THREADS];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const row_plan p = plan_row(a.req[row], a.n_vocab);
    if (p.refuse || p.greedy || p.path != PATH_TOPK) return;
    row_prep *pr = prep + row;
    const uint32_t nb = L == 1 ? HIST1 : L == 2 ? HIST2 : HIST3;
    const uint32_t *h = pr->hist + (L == 1 ? 0u : L == 2 ? HIST1 : HIST1 + HIST2);
    uint32_t want;
    if (L == 1) {
        float mv;
        int32_t mi;
        uint32_t nf;
        prep_reduce_max(pr, &mv, &mi, &nf);
        want = min((uint32_t)p.top_k, nf);
    } else {
        want = L == 2 ? pr->want1 : pr->want2;
    }
    /* each thread a contiguous run of bins; a scan of the runs' sums finds the digit */
    const uint32_t per = nb / PREP_THREADS;
    uint32_t s = 0;
    for (uint32_t b = 0; b < per; b++) s += h[tid * per + b];
    sc[tid] = s;
    __syncthreads();
    for (uint32_t off = 1; off < PREP_THREADS; off <<= 1) {
        const uint32_t t = tid >= off ? sc[tid - off] : 0u;
        __syncthreads();
        sc[tid] += t;
        __syncthreads();
    }
    const uint32_t before = sc[tid] - s;   /* bins below this thread's run */
    if (want == 0) return;                 /* no finite value: the row kernel refuses it */
    if (before < want && sc[tid] >= want) {
        uint32_t c = before, d = tid * per;
        while (c + h[d] < want) c += h[d++];
        if (L == 1) { pr->d1 = d; pr->want1 = want - c; }
        else if (L == 2) { pr->d2 = d; pr->want2 = want - c; }
        else { pr->d3 = d; pr->want3 = want - c; }
    }
}

/* level 2 / 3 histograms over the keys of the decided prefix */
template <uint32_t L>
__global__ void __launch_bounds__(PREP_THREADS) prep_hist_kernel(const __grid_constant__ launch_args a, row_prep *prep) {
    const uint32_t row = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    const row_plan p = plan_row(q, a.n_vocab);
    if (p.refuse || p.greedy || p.path != PATH_TOPK) return;
    row_prep *pr = prep + row;
    const row_view rv = prep_view(a, q);
    uint32_t lo, hi;
    prep_chunk(rv.n, c, &lo, &hi);
    const uint32_t prefix = L == 2 ? pr->d1 : (pr->d1 << 11) | pr->d2;
    const uint32_t shift = L == 2 ? 21u : 10u;
    uint32_t *h = pr->hist + (L == 2 ? HIST1 : HIST1 + HIST2);
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
        const float v = rv(i);
        if (!finite_bits(v)) continue;
        const uint32_t dk = desc_key(v);
        if ((dk >> shift) != prefix) continue;
        atomicAdd(&h[L == 2 ? (dk >> 10) & 2047u : dk & 1023u], 1u);
    }
}

/* the id-order sum's values of a min-p row (not full-nucleus), over the GPU: expf's double arithmetic is the cost a
 * row's single block cannot carry -- the sum then reads them (sample_row) */
__global__ void __launch_bounds__(PREP_THREADS) prep_pvals_kernel(const __grid_constant__ launch_args a,
                                                                  const row_prep *prep) {
    const uint32_t row = blockIdx.y, c = blockIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    const row_plan p = plan_row(q, a.plan_vocab);
    const row_prep *pr = prep + row;
    if (p.refuse || p.greedy || p.path != PATH_MINP || p.full || pr->maxi < 0) return;
    const row_view rv = prep_view(a, q);
    float *pv = a.pvals + (size_t)row * a.n_vocab;
    uint32_t lo, hi;
    grid_chunk(rv.n, c, &lo, &hi);
    for (uint32_t i = lo + threadIdx.x; i < hi; i += PREP_THREADS) {
        float x = 0.0f;
        (void)row_is_cand(p, i, rv(i), __uint_as_float(0xff800000u), pr->maxv, (uint32_t)pr->maxi, &x);
        pv[i] = x;
    }
}

/* the row's maximum (block 0 records it) and its candidates appended to its list */
__global__ void __launch_bounds__(PREP_THREADS) prep_collect_kernel(const __grid_constant__ launch_args a,
                                                                    row_prep *prep) {
    const uint32_t row = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    const row_plan p = plan_row(q, a.n_vocab);
    row_prep *pr = prep + row;
    float maxv;
    int32_t maxi;
    uint32_t n_fin;
    prep_reduce_max(pr, &maxv, &maxi, &n_fin);
    if (c == 0 && tid == 0) {
        pr->maxv = maxv;
        pr->maxi = maxi;
        pr->n_fin = n_fin;
    }
    if (p.refuse || p.greedy || maxi < 0 || ((a.full_rows >> row) & 1u)) return;
    const row_view rv = prep_view(a, q);
    uint32_t lo, hi;
    prep_chunk(rv.n, c, &lo, &hi);
    const uint32_t kstar = (pr->d1 << 21) | (pr->d2 << 10) | pr->d3;
    const float floor_logit = row_floor(p, maxv);
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
        const float v = rv(i);
        bool take;
        if (p.path == PATH_TOPK) {
            take = finite_bits(v) && desc_key(v) <= kstar;   /* below K*, and K*'s whole tie class */
        } else {
            float pp;
            take = row_is_cand(p, i, v, floor_logit, maxv, (uint32_t)maxi, &pp);
        }
        if (!take) continue;
        const uint32_t at = atomicAdd(&pr->n_cand, 1u);
        if (at < SMP_CAP) pr->cand[at] = ((uint64_t)desc_key(v) << 32) | i;
    }
}

/* Serially sum v[0, n) in order (one thread) -- the host's float accumulation, bit for bit; four at a time from
 * shared memory so the loads overlap the dependent adds. */
__device__ __forceinline__ float serial_sum_from(float s, const float *v, uint32_t n) {
    uint32_t i = 0;
    if ((((uintptr_t)v) & 15u) == 0)
        for (; i + 4u <= n; i += 4u) {
            const float4 x = *reinterpret_cast<const float4 *>(v + i);
            s = __fadd_rn(s, x.x);
            s = __fadd_rn(s, x.y);
            s = __fadd_rn(s, x.z);
            s = __fadd_rn(s, x.w);
        }
    for (; i < n; i++) s = __fadd_rn(s, v[i]);
    return s;
}
__device__ __forceinline__ float serial_sum(const float *p, uint32_t n) { return serial_sum_from(0.0f, p, n); }

/* ---- THE serial sum, in parallel and to the bit ----------------------------------------------------------------
 *
 * The host summed non-negative values one at a time in round-to-nearest (float in the sampler, double in the logprob
 * scorer).  While the running sum s stays inside one binade its representable values are the multiples of one ulp u,
 * s = m * u (m below `top`, 2^24 or 2^53; the subnormals share the lowest normal binade's u), so s + p rounds to
 * (m + rint(p / u)) * u -- integer arithmetic, which a block scan does in any order.  Two cases are not that: a value
 * exactly halfway between grid points (the rounding's tie goes to an even m + k, which depends on m) and the add
 * that takes the sum out of its binade (the next grid is coarser).  Both are taken serially, by the hardware add,
 * and the scan restarts after them on the sum's new grid -- inside the tile, from the values each thread holds in
 * registers.  A sum crosses a binade a few dozen times, and an exact tie is rare, so a row costs about one pass. */

constexpr uint32_t SUM_E = 8;   /* values a thread holds a tile */

template <typename T> struct sum_grid_traits;
template <> struct sum_grid_traits<float> {
    static constexpr uint32_t MANT = 23u, LSB = 150u, EMAX = 255u;   /* value = M * 2^(max(e, 1) - LSB) */
    static __device__ __forceinline__ uint64_t bits(float v) { return __float_as_uint(v); }
    static __device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
    static __device__ __forceinline__ float scaled(uint64_t m, int k) { return ldexpf((float)(uint32_t)m, -k); }
};
template <> struct sum_grid_traits<double> {
    static constexpr uint32_t MANT = 52u, LSB = 1075u, EMAX = 2047u;
    static __device__ __forceinline__ uint64_t bits(double v) { return (uint64_t)__double_as_longlong(v); }
    static __device__ __forceinline__ double add(double a, double b) { return __dadd_rn(a, b); }
    static __device__ __forceinline__ double scaled(uint64_t m, int k) { return ldexp((double)m, -k); }
};

/* s's grid: s = m * 2^-k exactly, m < 2^(MANT + 1); finite false for a NaN or infinite s */
template <typename T>
struct sum_grid {
    bool finite;
    int k;
    uint64_t m;
    __device__ __forceinline__ explicit sum_grid(T s) {
        typedef sum_grid_traits<T> tr;
        const uint64_t b = tr::bits(s);
        const uint32_t e = (uint32_t)(b >> tr::MANT) & tr::EMAX;
        const uint64_t mant = b & ((1ull << tr::MANT) - 1u);
        finite = e != tr::EMAX;
        m = e ? mant | (1ull << tr::MANT) : mant;
        k = (int)tr::LSB - (int)(e ? e : 1u);
    }
    __device__ __forceinline__ T value(uint64_t mm) const { return sum_grid_traits<T>::scaled(mm, k); }
};

/* rint(p * 2^k) from p's bits (integer ops: no FP64), clamped at `top` (any such value leaves the binade); *serial
 * set for an exact tie or a NaN p */
template <typename T>
__device__ __forceinline__ uint64_t grid_q(T p, int k, bool *serial) {
    typedef sum_grid_traits<T> tr;
    constexpr uint64_t TOP = 1ull << (tr::MANT + 1u);
    const uint64_t b = tr::bits(p);
    const uint32_t e = (uint32_t)(b >> tr::MANT) & tr::EMAX;
    if (e == tr::EMAX) { *serial = true; return 0; }
    const uint64_t M = e ? (b & ((1ull << tr::MANT) - 1u)) | (1ull << tr::MANT) : b & ((1ull << tr::MANT) - 1u);
    if (M == 0) return 0;   /* +0 (a -0 cannot arise) */
    const int sh = (int)(e ? e : 1u) - (int)tr::LSB + k;
    if (sh >= 0) return sh > (int)tr::MANT || M >= (TOP >> sh) ? TOP : M << sh;
    const int r = -sh;
    if (r > (int)tr::MANT + 2) return 0;   /* M < 2^(MANT + 1) <= half the step: under half */
    const uint64_t rem = M & ((1ull << r) - 1u), half = 1ull << (r - 1);
    uint64_t q = M >> r;
    if (rem > half) q++;
    else if (rem == half) *serial = true;
    return q;
}

/* Saturating at 2^62: a prefix that large has left any binade, and the saturating sum is still associative. */
__device__ __forceinline__ uint64_t sat_add(uint64_t a, uint64_t b) {
    constexpr uint64_t CAP = 1ull << 62;
    const uint64_t s = a + b;
    return s > CAP || s < a ? CAP : s;
}

/* Block-wide exclusive (saturating) scan of v (THREADS threads); *total receives the sum. */
template <uint32_t THREADS>
__device__ __forceinline__ uint64_t block_scan_excl_u64(uint64_t v, unsigned long long *wsum, uint64_t *total) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    uint64_t x = v, ex = 0;
    for (uint32_t off = 1; off < 32u; off <<= 1) {
        const uint64_t o = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x = sat_add(x, o);
    }
    {   /* the warp-exclusive prefix: the inclusive one of the lane below */
        const uint64_t o = __shfl_up_sync(0xffffffffu, x, 1);
        ex = lane ? o : 0u;
    }
    if (lane == 31u) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint64_t w = lane < THREADS / 32u ? wsum[lane] : 0u;
        for (uint32_t off = 1; off < 32u; off <<= 1) {
            const uint64_t o = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w = sat_add(w, o);
        }
        if (lane < THREADS / 32u) wsum[lane] = w;
    }
    __syncthreads();
    const uint64_t r = sat_add(ex, warp ? wsum[warp - 1u] : 0u);
    *total = wsum[THREADS / 32u - 1u];
    __syncthreads();
    return r;
}

/* The block's least c. */
template <uint32_t THREADS>
__device__ __forceinline__ uint32_t block_min_u32(uint32_t c, uint32_t *wmin) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    c = __reduce_min_sync(0xffffffffu, c);
    if (lane == 0) wmin[warp] = c;
    __syncthreads();
    uint32_t r = UINT32_MAX;
    for (uint32_t w = 0; w < THREADS / 32u; w++) r = min(r, wmin[w]);
    __syncthreads();
    return r;
}

/* One tile of the serial sum: the values v (thread t holds [ib, ib + E)), from index `start` to t_end, onto sm.s (the
 * sum before `start`).  On return sm.s is the sum through t_end - 1 -- or, HAS_PRED, through the first index whose
 * prefix satisfies pred, which sm.stop then names; sm.nan is set when the sum went NaN.  Every thread takes part. */
template <typename T, uint32_t THREADS, uint32_t E, bool HAS_PRED, typename P>
__device__ void tile_walk(const T (&v)[E], uint32_t ib, uint32_t start, uint32_t t_end, P pred, serial_sum_smem<T> &sm) {
    typedef sum_grid_traits<T> tr;
    constexpr uint64_t TOP = 1ull << (tr::MANT + 1u);
    const uint32_t tid = threadIdx.x;
    for (;;) {
        const sum_grid<T> g(sm.s);
        if (!g.finite) {   /* a NaN sum stays NaN (an infinite one cannot arise: every value is <= 1) */
            __syncthreads();
            if (tid == 0) sm.nan = 1;
            __syncthreads();
            return;
        }
        uint64_t q[E], loc = 0;
        uint32_t ser = 0;
#pragma unroll
        for (uint32_t e = 0; e < E; e++) {
            bool f = false;
            q[e] = ib + e >= start && ib + e < t_end ? grid_q<T>(v[e], g.k, &f) : 0u;
            if (f) ser |= 1u << e;
            loc = sat_add(loc, q[e]);
        }
        uint64_t total;
        const uint64_t excl = block_scan_excl_u64<THREADS>(loc, sm.wsum, &total);
        uint32_t hit = UINT32_MAX;
        uint64_t S = excl;
#pragma unroll
        for (uint32_t e = 0; e < E; e++) {
            S = sat_add(S, q[e]);
            const uint32_t i = ib + e;
            if (hit != UINT32_MAX || i < start || i >= t_end) continue;
            bool h = ((ser >> e) & 1u) || g.m + S >= TOP;
            if (!h && HAS_PRED) h = pred(g.value(g.m + S));
            if (h) hit = i;
        }
        const uint32_t first = HAS_PRED || (g.m + total >= TOP) || __syncthreads_or(ser != 0u)
                                   ? block_min_u32<THREADS>(hit, sm.wmin)
                                   : UINT32_MAX;
        if (first == UINT32_MAX) {
            if (tid == 0) sm.s = g.value(g.m + total);
            __syncthreads();
            return;
        }
        if (first >= ib && first < ib + E) {
            const uint32_t e0 = first - ib;
            uint64_t Sb = excl, qf = 0;   /* the prefix before `first`, and its q */
            T pv = T(0);
#pragma unroll
            for (uint32_t e = 0; e < E; e++) {
                if (e < e0) Sb = sat_add(Sb, q[e]);
                if (e == e0) { qf = q[e]; pv = v[e]; }
            }
            if (!((ser >> e0) & 1u) && g.m + Sb + qf < TOP) {   /* pred at an exact prefix */
                sm.s = g.value(g.m + Sb + qf);
                sm.stop = first;
            } else {   /* a tie, a NaN or the binade's edge: the hardware add */
                const T ns = tr::add(g.value(g.m + Sb), pv);
                sm.s = ns;
                if (HAS_PRED && pred(ns)) sm.stop = first;
            }
            sm.start = first + 1u;
        }
        __syncthreads();
        start = sm.start;
        const bool done = sm.stop != UINT32_MAX || start >= t_end;
        __syncthreads();
        if (done) return;
    }
}

/* The serial round-to-nearest sum of value(0) .. value(n - 1) in one block, stopped after the first i whose prefix
 * satisfies pred (HAS_PRED; *stop_at = i, else UINT32_MAX) -- a row's few values (a chain's slot, a slot kernel's
 * row); a row's whole vocabulary takes the multi-block sum below.  Every thread returns the sum. */
template <typename T, uint32_t THREADS, uint32_t E, bool HAS_PRED, typename V, typename P>
__device__ T block_serial_sum(uint32_t n, V value, P pred, uint32_t *stop_at, serial_sum_smem<T> &sm) {
    const uint32_t tid = threadIdx.x;
    if (tid == 0) { sm.s = T(0); sm.stop = UINT32_MAX; sm.nan = 0; }
    __syncthreads();
    for (uint32_t t0 = 0; t0 < n; t0 += THREADS * E) {
        if (sm.stop != UINT32_MAX || sm.nan) break;
        const uint32_t ib = t0 + tid * E, t_end = min(n, t0 + THREADS * E);
        T v[E];
#pragma unroll
        for (uint32_t e = 0; e < E; e++) v[e] = ib + e < n ? value(ib + e) : T(0);
        tile_walk<T, THREADS, E, HAS_PRED>(v, ib, t0, t_end, pred, sm);
    }
    if (stop_at) *stop_at = sm.stop;
    const T out = sm.s;
    __syncthreads();
    return out;
}

/* ---- the serial sum over a row's whole vocabulary, over the GPU (decoupled look-back on the ulp grid) ------------
 *
 * Tiles of SR_TILE values, a block each, take their order by ticket (so every tile's predecessors have started).  A
 * tile first guesses the grid its start will sit on (the grid of an approximate prefix: the tiles' double sums
 * before it) and publishes its integer aggregate on that grid -- Q, the sum of rint(p / u), and whether any of its
 * values ties.  Its exact start is then the nearest predecessor's exact end plus the aggregates between, whenever
 * those share the grid, hold no tie and leave the binade unreached (an aggregate is shift-invariant on its grid); else
 * it waits for its predecessor's exact end.  From the exact start, tile_walk gives the exact end (one scan pass when
 * nothing ties or crosses), which the tile publishes.  The few tiles where the sum leaves a binade chain; the rest
 * resolve from the look-back. */

constexpr uint32_t SR_THREADS = 512, SR_E = 8, SR_TILE = SR_THREADS * SR_E;
constexpr uint32_t SR_ROWS = PULSAR_GPU_SAMPLE_ROWS_MAX;

struct sr_status {
    uint32_t flag;            /* 0 nothing, 1 the aggregate, 2 the exact end too */
    int32_t k;                /* the aggregate's grid (u = 2^-k) */
    uint32_t tie;             /* the aggregate holds a tie or a NaN */
    uint32_t stop;            /* HAS_PRED: the first index whose prefix satisfies pred, in this tile, or UINT32_MAX */
    unsigned long long Q;     /* the aggregate */
    double end;               /* the exact sum through the tile's end (or through `stop`) */
};

/* A row's sum: its values (T at vals), their count (*n_ptr, or n when n_ptr is NULL); HAS_PRED's prefix test is
 * prefix * (1 / *sum_ptr) >= top_p (the cut, against the row's sum from a pass before). */
struct sr_row {
    const void *vals;
    const uint32_t *n_ptr;
    uint32_t n;
    float top_p;
    const double *sum_ptr;
};

struct sr_args {
    sr_row row[SR_ROWS];
    uint32_t n_rows, tiles;   /* tiles: the launch's grid width (the widest row's) */
    sr_status *st;            /* [n_rows][tiles] */
    double *approx;           /* [n_rows][tiles] */
    uint32_t *ticket;         /* [n_rows] */
    double *sum;              /* [n_rows]: the sum (HAS_PRED: through the stop) */
    uint32_t *stop;           /* [n_rows]: HAS_PRED's first index, or UINT32_MAX */
};

__device__ __forceinline__ uint32_t sr_n(const sr_row &r) { return r.n_ptr ? *r.n_ptr : r.n; }

/* each tile's double sum, any order: the approximate prefix the tiles' grid guesses read */
template <typename T>
__global__ void __launch_bounds__(SR_THREADS) sr_approx_kernel(const __grid_constant__ sr_args a) {
    __shared__ double red[SR_THREADS / 32u];
    const uint32_t t = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    if (tid == 0 && t == 0) a.ticket[r] = 0u;   /* (the sum kernel's tickets; this kernel runs first) */
    const uint32_t n = sr_n(a.row[r]);
    const T *v = (const T *)a.row[r].vals;
    double x = 0.0;
    for (uint32_t i = t * SR_TILE + tid; i < min(n, (t + 1u) * SR_TILE); i += SR_THREADS) x += (double)v[i];
    for (uint32_t off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
    if ((tid & 31u) == 0) red[tid >> 5] = x;
    __syncthreads();
    if (tid == 0) {
        double y = 0.0;
        for (uint32_t w = 0; w < SR_THREADS / 32u; w++) y += red[w];
        a.approx[(size_t)r * a.tiles + t] = y;
        a.st[(size_t)r * a.tiles + t].flag = 0u;
    }
}

template <typename T, bool HAS_PRED>
__global__ void __launch_bounds__(SR_THREADS) sr_sum_kernel(const __grid_constant__ sr_args a) {
    typedef sum_grid_traits<T> tr;
    constexpr uint64_t TOP = 1ull << (tr::MANT + 1u);
    __shared__ serial_sum_smem<T> sm;
    __shared__ uint32_t tile_s;
    __shared__ T start_s;
    const uint32_t r = blockIdx.y, tid = threadIdx.x;
    const sr_row &row = a.row[r];
    const uint32_t n = sr_n(row);
    if (tid == 0) tile_s = atomicAdd(&a.ticket[r], 1u);
    __syncthreads();
    const uint32_t t = tile_s;
    if ((uint64_t)t * SR_TILE >= n) return;
    sr_status *st = a.st + (size_t)r * a.tiles;
    volatile sr_status *vst = st;
    const T *vals = (const T *)row.vals;
    const uint32_t t0 = t * SR_TILE, t_end = min(n, t0 + SR_TILE), ib = t0 + tid * SR_E;
    T v[SR_E];
#pragma unroll
    for (uint32_t e = 0; e < SR_E; e++) v[e] = ib + e < t_end ? vals[ib + e] : T(0);
    const float rs = HAS_PRED ? __fdiv_rn(1.0f, (float)*row.sum_ptr) : 0.0f;
    auto pred = [&](T pre) { return HAS_PRED && __fmul_rn((float)pre, rs) >= row.top_p; };
    /* the aggregate on the guessed grid */
    double est = 0.0;
    if (tid < 32u) {
        for (uint32_t j = tid; j < t; j += 32u) est += a.approx[(size_t)r * a.tiles + j];
        for (uint32_t off = 16; off > 0; off >>= 1) est += __shfl_down_sync(0xffffffffu, est, off);
    }
    if (tid == 0) start_s = (T)est;
    __syncthreads();
    const sum_grid<T> gg(start_s);
    uint64_t loc = 0, q[SR_E];
    uint32_t ser = 0;
#pragma unroll
    for (uint32_t e = 0; e < SR_E; e++) {
        bool f = false;
        q[e] = ib + e < t_end ? grid_q<T>(v[e], gg.k, &f) : 0u;
        ser |= f ? 1u << e : 0u;
        loc = sat_add(loc, q[e]);
    }
    uint64_t total;
    const uint64_t excl = block_scan_excl_u64<SR_THREADS>(loc, sm.wsum, &total);
    const bool tie = __syncthreads_or(ser != 0u) || !gg.finite;
    if (tid == 0) {
        vst[t].k = gg.k;
        vst[t].tie = tie ? 1u : 0u;
        vst[t].Q = total;
        __threadfence();
        vst[t].flag = 1u;
    }
    /* the exact start: the nearest exact end behind and the agreeing aggregates between */
    if (tid == 0) {
        T s0 = T(0);
        if (t > 0) {
            uint64_t acc = 0;
            int32_t j = (int32_t)t - 1;
            bool exact = false;
            T base = T(0);
            while (j >= 0) {
                uint32_t f;
                while ((f = vst[j].flag) == 0u) { }
                __threadfence();
                if (f == 2u) { base = (T)vst[j].end; exact = true; break; }
                if (vst[j].k != gg.k || vst[j].tie) break;
                acc = sat_add(acc, vst[j].Q);
                j--;
            }
            if (!exact && j < 0) { base = T(0); exact = true; }   /* the row's start */
            bool ok = false;
            if (exact) {
                const sum_grid<T> bg(base);
                ok = bg.finite && bg.k == gg.k && bg.m + acc < TOP;
                if (ok) s0 = bg.value(bg.m + acc);
                else if (acc == 0u) { s0 = base; ok = true; }   /* the base is the predecessor's own end */
            }
            if (!ok) {   /* the span leaves the grid: the predecessor's exact end */
                while (vst[t - 1].flag != 2u) { }
                __threadfence();
                s0 = (T)vst[t - 1].end;
            }
        }
        sm.s = s0;
        sm.stop = UINT32_MAX;
        sm.nan = 0;
        sm.start = t0;
    }
    __syncthreads();
    /* the start sits on the guessed grid (the usual case): the aggregate's scan is the tile's walk, but for the first
     * value that ties, crosses the binade or meets pred from the exact start -- found without a second scan */
    const sum_grid<T> sg(sm.s);
    bool walked = false;
    if (sg.finite && sg.k == gg.k && !tie) {
        uint32_t hit = UINT32_MAX;
        uint64_t S = excl;
#pragma unroll
        for (uint32_t e = 0; e < SR_E; e++) {
            S = sat_add(S, q[e]);
            if (hit != UINT32_MAX || ib + e >= t_end) continue;
            if (sg.m + S >= TOP || (HAS_PRED && pred(sg.value(sg.m + S)))) hit = ib + e;
        }
        const uint32_t first = HAS_PRED || sg.m + total >= TOP ? block_min_u32<SR_THREADS>(hit, sm.wmin) : UINT32_MAX;
        if (first == UINT32_MAX) {
            if (tid == 0) sm.s = sg.value(sg.m + total);
            walked = true;
        }
    }
    __syncthreads();
    if (!walked) tile_walk<T, SR_THREADS, SR_E, HAS_PRED>(v, ib, t0, t_end, pred, sm);
    if (tid == 0) {
        vst[t].stop = sm.stop;
        vst[t].end = (double)sm.s;
        __threadfence();
        vst[t].flag = 2u;
    }
}

/* the row's result: the last tile's end, or (HAS_PRED) the first tile's stop and its prefix */
__global__ void sr_final_kernel(const __grid_constant__ sr_args a, bool has_pred) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= a.n_rows) return;
    const uint32_t n = sr_n(a.row[r]);
    const sr_status *st = a.st + (size_t)r * a.tiles;
    const uint32_t nt = (n + SR_TILE - 1u) / SR_TILE;
    a.sum[r] = nt ? st[nt - 1u].end : 0.0;
    a.stop[r] = UINT32_MAX;
    if (has_pred)
        for (uint32_t t = 0; t < nt; t++)
            if (st[t].stop != UINT32_MAX) {
                a.sum[r] = st[t].end;
                a.stop[r] = st[t].stop;
                break;
            }
}

/* THE row: argmax, the path, the candidates, the cut, the race.  `pp` (the row kernel) is the row's prep -- its maximum
 * and candidate list (a list past the shared store leaves the row pending for a slot); `slot` (the slot kernel, pp
 * NULL) is a global scratch slot: the row's own passes, a stable radix sort in global memory. */
template <uint32_t THREADS>
__device__ void sample_row(const launch_args &a, uint32_t row, row_smem<THREADS> &sm, const row_prep *pp,
                           uint8_t *slot) {
    const uint32_t tid = threadIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    const row_view rv = prep_view(a, q);
    row_out_t *o = a.out + row;
    uint32_t *dist = a.dist ? a.dist + (size_t)row * (1u + 2u * a.dist_cap) : nullptr;
    auto finish = [&](int32_t tok, uint32_t status) {
        if (tid == 0) {
            o->token = tok;
            o->status = status;
            if (dist && (status != PULSAR_GPU_SAMPLE_OK || tok < 0)) dist[0] = 0;
        }
    };
    auto dump_point = [&](int32_t tok) {   /* a point mass (greedy) for the gate's dump */
        if (tid == 0 && dist && tok >= 0) {
            dist[0] = 1;
            dist[1] = (uint32_t)tok;
            dist[1 + a.dist_cap] = __float_as_uint(1.0f);
        }
    };

    if (tid == 0) o->probe_p = 0.0f;   /* before the first barrier: a race-loop thread may set it below */
    const row_plan p = plan_row(q, a.plan_vocab);
    if (p.refuse) {
        finish(-1, PULSAR_GPU_SAMPLE_REFUSED_KNOB);
        return;
    }
    float maxv;
    int32_t maxi;
    if (pp) {
        maxv = pp->maxv;
        maxi = pp->maxi;
    } else {
        block_argmax<THREADS>(rv, sm.redv, sm.redi, &maxv, &maxi);
    }
    if (p.greedy) {
        if (maxi >= 0 && a.id_map) maxi = a.id_map[maxi];
        if (tid == 0 && maxi >= 0 && maxi == q.probe) o->probe_p = 1.0f;
        finish(maxi, maxi < 0 ? PULSAR_GPU_SAMPLE_REFUSED_EMPTY : PULSAR_GPU_SAMPLE_OK);
        dump_point(maxi);
        return;
    }
    if (maxi < 0) {
        finish(-1, PULSAR_GPU_SAMPLE_REFUSED_EMPTY);
        return;
    }
    const float top_p = p.top_p, min_p = p.min_p, rcpT = p.rcpT;
    /* a prep list past the shared store: the slot kernel finishes this row from scratch */
    if (pp && pp->n_cand > SMP_CAP) {
        if (tid == 0) a.pending[row] = 1u;
        return;
    }

    cand_list cl;
    cl.map = a.id_map;
    cl.n = 0;
    cl.skeys = nullptr;
    cl.gids = nullptr;
    cl.probs = nullptr;
    float sum = 0.0f;
    bool sum_inert = false;
    float maxlogit = 0.0f;   /* the sorted head's value: the probabilities' anchor */

    /* slot scratch layout */
    uint32_t *gk0 = nullptr, *gv0 = nullptr, *gk1 = nullptr, *gv1 = nullptr;
    float *gp = nullptr;
    uint32_t *gcnt = nullptr;
    if (slot) {
        gk0 = (uint32_t *)slot;
        gv0 = gk0 + a.n_vocab;
        gk1 = gv0 + a.n_vocab;
        gv1 = gk1 + a.n_vocab;
        gp = (float *)(gv1 + a.n_vocab);
        gcnt = (uint32_t *)(gp + a.n_vocab);   /* 17 x THREADS words */
    }

    if (p.path == PATH_TOPK) {
        uint32_t k;
        if (pp) {
            /* the prep's list: every key below K* and K*'s tie class; sorted, its first k are the host's k */
            k = min((uint32_t)p.top_k, pp->n_fin);
            const uint32_t m = pp->n_cand;
            for (uint32_t i = tid; i < m; i += THREADS) sm.keys[i] = pp->cand[i];
            __syncthreads();
            bitonic_sort<THREADS>(sm.keys, m);
        } else {
            uint32_t nfin = 0;
            for (uint32_t i = tid; i < rv.n; i += THREADS) nfin += finite_bits(rv(i)) ? 1u : 0u;
            nfin = block_sum_u32<THREADS>(nfin, sm.redu);
            k = min((uint32_t)p.top_k, nfin);
            select_topk<THREADS>(rv, k, sm.keys, sm.hist, &sm.cnt, sm.redu);
        }
        cl.n = k;
        cl.skeys = sm.keys;
        maxlogit = rv((uint32_t)sm.keys[0]);
        for (uint32_t i = tid; i < k; i += THREADS)
            sm.probs[i] = plm_expf(__fmul_rn(__fsub_rn(rv((uint32_t)sm.keys[i]), maxlogit), rcpT));
        __syncthreads();
        cl.probs = sm.probs;
        if (tid == 0) sm.fsum = serial_sum(sm.probs, k);
        __syncthreads();
        sum = sm.fsum;
    } else {
        /* PATH_MINP, and PATH_FULL in a block (a chain's slot: a launch's full-arm rows take their own pipeline):
         * the candidate predicate, and (min-p, not full-nucleus) the id-order sum */
        const uint32_t max_id = (uint32_t)maxi;
        const float floor_logit = row_floor(p, maxv);
        const bool full_arm = p.path == PATH_FULL;
        maxlogit = maxv;
        const bool idsum = !full_arm && !p.full;
        if (tid == 0) { sm.cnt = 0; sm.fsum = 0.0f; }
        __syncthreads();
        uint32_t n;
        if (!pp) {   /* the count (the row kernel has the prep's) */
            uint32_t mine = 0;
            for (uint32_t i = tid; i < rv.n; i += THREADS) {
                float pr;
                mine += row_is_cand(p, i, rv(i), floor_logit, maxlogit, max_id, &pr) ? 1u : 0u;
            }
            n = block_sum_u32<THREADS>(mine, sm.redu);
        } else {
            n = pp->n_cand;
        }
        if (idsum && ((a.idsum_rows >> row) & 1u)) {   /* the launch's sum over the GPU (sr_sum_kernel) */
            if (tid == 0) sm.fsum = (float)a.idsums[row];
            __syncthreads();
        } else if (idsum) {   /* a chain slot's few values: the id-order serial sum in the block (the host's bits) */
            auto value = [&](uint32_t i) {
                float pr = 0.0f;   /* a non-finite value adds +0 to the sum: the host skipped it, same bits */
                (void)row_is_cand(p, i, rv(i), floor_logit, maxlogit, max_id, &pr);
                return pr;
            };
            const float fs = block_serial_sum<float, THREADS, SUM_E, false>(rv.n, value, [](float) { return false; },
                                                                            nullptr, sm.ssum);
            if (tid == 0) sm.fsum = fs;
            __syncthreads();
        }
        if (idsum) sum = sm.fsum;
        if (p.full) { sum = __uint_as_float(0x7f800000u); sum_inert = true; }
        cl.n = n;
        if (full_arm && n > SMP_CAP) {   /* a launch routes these to the full arm's pipeline: none reaches here */
            finish(-1, PULSAR_GPU_SAMPLE_REFUSED_KNOB);
            return;
        }
        if (n <= SMP_CAP) {
            if (pp) {
                for (uint32_t i = tid; i < n; i += THREADS) sm.keys[i] = pp->cand[i];
            } else {
                if (tid == 0) sm.cnt = 0;
                __syncthreads();
                for (uint32_t i = tid; i < rv.n; i += THREADS) {
                    float pr;
                    if (row_is_cand(p, i, rv(i), floor_logit, maxlogit, max_id, &pr))
                        sm.keys[atomicAdd(&sm.cnt, 1u)] = ((uint64_t)desc_key(rv(i)) << 32) | i;
                }
            }
            __syncthreads();
            bitonic_sort<THREADS>(sm.keys, n);
            cl.skeys = sm.keys;
            if (full_arm) maxlogit = rv((uint32_t)sm.keys[0]);   /* the sorted head */
            for (uint32_t i = tid; i < n; i += THREADS)
                sm.probs[i] = plm_expf(__fmul_rn(__fsub_rn(rv((uint32_t)sm.keys[i]), maxlogit), rcpT));
            __syncthreads();
            cl.probs = sm.probs;
            if (full_arm) {   /* the serial sum in sorted order */
                if (tid == 0) sm.fsum = serial_sum(sm.probs, n);
                __syncthreads();
                sum = sm.fsum;
            }
        } else {
            if (!slot) {   /* past the shared store: the slot kernel finishes this row */
                if (tid == 0) a.pending[row] = 1u;
                return;
            }
            /* ordered compaction into the slot (ascending id), the stable sort, the probabilities in sorted order */
            const uint32_t chunk = (rv.n + THREADS - 1u) / THREADS;
            const uint32_t lo = min(tid * chunk, rv.n), hi = min(lo + chunk, rv.n);
            uint32_t c = 0;
            for (uint32_t i = lo; i < hi; i++) {
                float pr;
                c += row_is_cand(p, i, rv(i), floor_logit, maxlogit, max_id, &pr) ? 1u : 0u;
            }
            uint32_t pos = block_scan_incl<THREADS>(c, sm.redu) - c;
            for (uint32_t i = lo; i < hi; i++) {
                float pr;
                if (row_is_cand(p, i, rv(i), floor_logit, maxlogit, max_id, &pr)) {
                    gk0[pos] = desc_key(rv(i));
                    gv0[pos] = i;
                    pos++;
                }
            }
            __syncthreads();
            global_radix_sort<THREADS>(gk0, gv0, gk1, gv1, n, gcnt);
            cl.gids = gv0;
            for (uint32_t i = tid; i < n; i += THREADS)
                gp[i] = plm_expf(__fmul_rn(__fsub_rn(rv(gv0[i]), maxlogit), rcpT));
            __syncthreads();
            cl.probs = gp;
        }
    }
    if (!sum_inert && (sum <= 0.0f || !finite_bits(sum))) {
        finish(-1, PULSAR_GPU_SAMPLE_REFUSED_MASS);
        return;
    }
    /* the cut (serial, as the host's loop) */
    if (tid == 0) {
        const float min_e = __fmul_rn(cl.probs[0], min_p);
        const float rs = __fdiv_rn(1.0f, sum);
        float fs = 0.0f;
        uint32_t f = 0;
        for (uint32_t i = 0; i < cl.n; i++) {
            const float p = cl.probs[i];
            if (i > 0 && p < min_e) break;
            fs = __fadd_rn(fs, p);
            f++;
            if (__fmul_rn(fs, rs) >= top_p) break;
        }
        sm.fsum = fs;
        sm.filtered = f;
    }
    __syncthreads();
    const uint32_t f = sm.filtered;
    const float rf = __fdiv_rn(1.0f, sm.fsum);
    /* the race over the emitted candidates: the first strict minimum in emitted order */
    double best = __longlong_as_double(0x7ff0000000000000ll);
    uint32_t besti = UINT32_MAX;
    for (uint32_t i = tid; i < f; i += THREADS) {
        const float pe = __fmul_rn(cl.probs[i], rf);
        if (dist && i < a.dist_cap) {
            dist[1 + i] = (uint32_t)cl.id(i);
            dist[1 + a.dist_cap + i] = __float_as_uint(pe);
        }
        if (cl.id(i) == q.probe) o->probe_p = pe;
        if (!(pe > 0.0f)) continue;
        const double s = race_score(q.key, cl.id(i), pe);
        if (s < best) { best = s; besti = i; }   /* i rises per thread: the first minimum of this thread's share */
    }
    block_argmin<THREADS>(&best, &besti, sm.redd, sm.redw);
    const uint32_t wi = besti;
    const int32_t tok = wi == UINT32_MAX ? cl.id(0) : cl.id(wi);
    if (tid == 0 && dist) dist[0] = f;
    finish(tok, PULSAR_GPU_SAMPLE_OK);
}

__global__ void __launch_bounds__(SMP_THREADS) sample_rows_kernel(const __grid_constant__ launch_args a,
                                                                   const row_prep *prep) {
    __shared__ row_smem<SMP_THREADS> sm;
    if (threadIdx.x == 0) a.pending[blockIdx.x] = 0u;
    if ((a.full_rows >> blockIdx.x) & 1u) return;   /* the full arm's pipeline draws it */
    __syncthreads();
    sample_row<SMP_THREADS>(a, blockIdx.x, sm, prep + blockIdx.x, nullptr);
}

/* The rows the row kernel left pending, round-robin over the slots in row order. */
__global__ void __launch_bounds__(SLOT_THREADS) sample_slots_kernel(const __grid_constant__ launch_args a) {
    __shared__ row_smem<SLOT_THREADS> sm;
    uint32_t rank = 0;
    for (uint32_t r = 0; r < a.n_rows; r++) {
        if (!a.pending[r]) continue;
        if (rank++ % SMP_SLOTS != blockIdx.x) continue;
        sample_row<SLOT_THREADS>(a, r, sm, nullptr, a.slots + (size_t)blockIdx.x * a.slot_bytes);
        __syncthreads();
    }
}

/* ---- the full arm (min_p <= 1e-30, no top-k): every finite value is a candidate --------------------------------
 *
 * The host routes these rows here (their path follows from the knobs alone): the row's (desc key, id) pairs sorted
 * by the device radix sort (stable, so ascending id on ties -- the host's order; a non-finite value keys ~0 and sorts
 * last), the probabilities over the GPU, the serial sum on one thread over staged tiles (stopped where the sum
 * saturates), the cut as a search over its prefixes and the race over the GPU (each chunk's first minimum, then the
 * first minimum of those).  Rows go FULL_GROUP at a time, one scratch
 * slot each. */

constexpr uint32_t FULL_GROUP = SMP_SLOTS;
/* a full-arm row's state: the candidates, the cut and the sum it emits by, each race chunk's first minimum */
struct full_state {
    uint32_t n, f, status, ncut;   /* ncut: the cut's walk length (the floor), 0 for a refused row */
    float fs;
    uint32_t ub_bits;   /* the race estimates' least upper bound (float bits) */
    double pbest[PREP_CHUNKS];
    uint32_t pidx[PREP_CHUNKS];
};

__host__ __device__ __forceinline__ uint64_t full_state_off(uint32_t n_vocab) {
    return ((uint64_t)n_vocab * 5u * sizeof(uint32_t) + 255u) & ~255ull;
}

/* the rows of one pass, slot j for row[j]; the sort leaves its output in the alternate buffers or not */
struct full_group {
    uint32_t row[FULL_GROUP];
    uint32_t n;
    uint32_t sorted_in_alt;
};

/* a slot's arrays: keys and ids in (keygen), the sorted ids, the probabilities in sorted order, the row's state */
struct full_slot {
    uint32_t *k_in, *v_in;
    const uint32_t *ids;
    float *p;
    full_state *st;
};

__device__ __forceinline__ full_slot full_slot_of(const launch_args &a, const full_group &g, uint32_t j) {
    uint8_t *base = a.slots + (size_t)j * a.slot_bytes;
    uint32_t *k0 = (uint32_t *)base, *v0 = k0 + a.n_vocab, *k1 = v0 + a.n_vocab, *v1 = k1 + a.n_vocab;
    full_slot s;
    s.k_in = k0;
    s.v_in = v0;
    s.ids = g.sorted_in_alt ? v1 : v0;
    s.p = (float *)(v1 + a.n_vocab);
    s.st = (full_state *)(base + full_state_off(a.n_vocab));
    return s;
}

/* the sort's input, and the row's state reset (the floor search lowers f from n) */
__global__ void __launch_bounds__(PREP_THREADS) full_keygen_kernel(const __grid_constant__ launch_args a,
                                                                   const full_group g, const row_prep *prep) {
    const uint32_t j = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const uint32_t row = g.row[j];
    const pulsar_gpu_sample_req q = a.req[row];
    const row_view rv = prep_view(a, q);
    const full_slot s = full_slot_of(a, g, j);
    if (c == 0 && tid == 0) {
        const uint32_t n = prep[row].n_fin;
        a.out[row].probe_p = 0.0f;
        s.st->n = n;
        s.st->f = n;
        s.st->ub_bits = 0x7f800000u;
        s.st->status = n == 0 ? PULSAR_GPU_SAMPLE_REFUSED_EMPTY : PULSAR_GPU_SAMPLE_OK;
    }
    uint32_t lo, hi;
    grid_chunk(rv.n, c, &lo, &hi);
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
        const float v = rv(i);
        s.k_in[i] = finite_bits(v) ? desc_key(v) : 0xffffffffu;
        s.v_in[i] = i;
    }
}

__global__ void __launch_bounds__(PREP_THREADS) full_probs_kernel(const __grid_constant__ launch_args a,
                                                                  const full_group g, const row_prep *prep) {
    const uint32_t j = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const uint32_t row = g.row[j];
    const pulsar_gpu_sample_req q = a.req[row];
    const row_plan p = plan_row(q, a.n_vocab);
    const row_view rv = prep_view(a, q);
    const full_slot s = full_slot_of(a, g, j);
    const uint32_t n = prep[row].n_fin;
    if (n == 0) return;
    const float maxlogit = rv(s.ids[0]);   /* the sorted head */
    uint32_t lo, hi;
    grid_chunk(n, c, &lo, &hi);
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS)
        s.p[i] = plm_expf(__fmul_rn(__fsub_rn(rv(s.ids[i]), maxlogit), p.rcpT));
}

/* the cut's floor: the host's loop stops before the first i > 0 with p under p0 * min_p (min_p <= 1e-30 here: only
 * a subnormal tail is under it) -- the least such i over the row lowers f */
__global__ void __launch_bounds__(PREP_THREADS) full_floor_kernel(const __grid_constant__ launch_args a,
                                                                  const full_group g) {
    __shared__ uint32_t red[PREP_THREADS];
    const uint32_t j = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const row_plan p = plan_row(a.req[g.row[j]], a.n_vocab);
    const full_slot s = full_slot_of(a, g, j);
    const uint32_t n = s.st->n;
    if (n == 0) return;
    const float min_e = __fmul_rn(s.p[0], p.min_p);
    uint32_t lo, hi;
    grid_chunk(n, c, &lo, &hi);
    uint32_t stop = UINT32_MAX;
    for (uint32_t i = max(lo, 1u) + tid; i < hi; i += PREP_THREADS)   /* i rises: this thread's first is its least */
        if (s.p[i] < min_e) { stop = i; break; }
    red[tid] = stop;
    __syncthreads();
    for (uint32_t off = PREP_THREADS / 2u; off > 0; off >>= 1) {
        if (tid < off) red[tid] = min(red[tid], red[tid + off]);
        __syncthreads();
    }
    if (tid == 0 && red[0] != UINT32_MAX) atomicMin(&s.st->f, red[0]);
}

/* THE serial sum in sorted order (the host's float accumulation) runs over the GPU (sr_sum_kernel), then the cut:
 * the walk again up to the floor (f from the floor search), stopped after the first candidate whose prefix
 * * (1 / sum) reaches top_p.  Between the two: the mass test, and each row's walk length. */
__global__ void full_mass_kernel(const __grid_constant__ launch_args a, const full_group g, const double *sum) {
    const uint32_t j = threadIdx.x;
    if (j >= g.n) return;
    const full_slot s = full_slot_of(a, g, j);
    if (s.st->status == PULSAR_GPU_SAMPLE_OK) {
        const float x = (float)sum[j];
        if (!(x > 0.0f) || !finite_bits(x)) s.st->status = PULSAR_GPU_SAMPLE_REFUSED_MASS;
    }
    s.st->ncut = s.st->status == PULSAR_GPU_SAMPLE_OK ? s.st->f : 0u;
}

/* the cut's result: the top-p stop, else the floor (or every candidate), and the prefix there */
__global__ void full_cut_kernel(const __grid_constant__ launch_args a, const full_group g, const double *prefix,
                                const uint32_t *stop) {
    const uint32_t j = threadIdx.x;
    if (j >= g.n) return;
    const full_slot s = full_slot_of(a, g, j);
    if (s.st->status != PULSAR_GPU_SAMPLE_OK) return;
    s.st->f = stop[j] != UINT32_MAX ? stop[j] + 1u : s.st->ncut;
    s.st->fs = (float)prefix[j];
}

/* A float estimate of race_score within a relative 1e-5 (the error is a few ulps): u = (m + 0.5) * 2^-53 is exact in
 * double; -log(u) from u itself under 1/2, from log1p(-(1 - u)) above (so a u near 1 keeps its digits). */
constexpr float RACE_EST_TOL = 1e-5f;
__device__ __forceinline__ float race_score_est(uint64_t key, int32_t id, float p) {
    const uint64_t h = mix64(key ^ ((uint64_t)(uint32_t)id * 0xd1b54a32d192ed03ull));
    const double u = __dmul_rn(__dadd_rn((double)(h >> 11), 0.5), 1.0 / 9007199254740992.0);
    const float e = u < 0.5 ? -logf((float)u) : -log1pf(-(float)__dsub_rn(1.0, u));
    return __fdiv_rn(e, p);
}

/* the race over the emitted candidates in two passes: the estimates' least upper bound over the row, then the exact
 * score (double, the host's) only where the estimate's lower bound reaches it -- every other candidate's score is
 * provably above the minimum, so the first strict minimum is the same.  Pass 1 also writes the probe and the dump. */
__global__ void __launch_bounds__(PREP_THREADS) full_race_bound_kernel(const __grid_constant__ launch_args a,
                                                                       const full_group g) {
    __shared__ float red[PREP_THREADS];
    const uint32_t j = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const uint32_t row = g.row[j];
    const pulsar_gpu_sample_req q = a.req[row];
    const full_slot s = full_slot_of(a, g, j);
    if (s.st->status != PULSAR_GPU_SAMPLE_OK) return;
    row_out_t *o = a.out + row;
    uint32_t *dist = a.dist ? a.dist + (size_t)row * (1u + 2u * a.dist_cap) : nullptr;
    const uint32_t f = s.st->f;   /* >= 1: the first candidate is always taken */
    const float rf = __fdiv_rn(1.0f, s.st->fs);
    float ub = __uint_as_float(0x7f800000u);
    uint32_t lo, hi;
    prep_chunk(f, c, &lo, &hi);
    for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
        const float pe = __fmul_rn(s.p[i], rf);
        const int32_t id = a.id_map ? a.id_map[s.ids[i]] : (int32_t)s.ids[i];
        if (dist && i < a.dist_cap) {
            dist[1 + i] = (uint32_t)id;
            dist[1 + a.dist_cap + i] = __float_as_uint(pe);
        }
        if (id == q.probe) o->probe_p = pe;
        if (!(pe > 0.0f)) continue;
        ub = fminf(ub, __fmul_rn(race_score_est(q.key, id, pe), 1.0f + RACE_EST_TOL));
    }
    red[tid] = ub;
    __syncthreads();
    for (uint32_t off = PREP_THREADS / 2u; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fminf(red[tid], red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) atomicMin(&s.st->ub_bits, __float_as_uint(red[0]));   /* non-negative: bits order as values */
}

/* pass 2: each chunk's first strict minimum of the exact scores among the candidates that can reach the bound */
__global__ void __launch_bounds__(PREP_THREADS) full_race_kernel(const __grid_constant__ launch_args a,
                                                                 const full_group g) {
    __shared__ double wd[32];
    __shared__ uint32_t wi[32];
    const uint32_t j = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const uint32_t row = g.row[j];
    const pulsar_gpu_sample_req q = a.req[row];
    const full_slot s = full_slot_of(a, g, j);
    double best = __longlong_as_double(0x7ff0000000000000ll);
    uint32_t besti = UINT32_MAX;
    if (s.st->status == PULSAR_GPU_SAMPLE_OK) {
        const uint32_t f = s.st->f;
        const float rf = __fdiv_rn(1.0f, s.st->fs), bound = __uint_as_float(s.st->ub_bits);
        uint32_t lo, hi;
        prep_chunk(f, c, &lo, &hi);
        for (uint32_t i = lo + tid; i < hi; i += PREP_THREADS) {
            const float pe = __fmul_rn(s.p[i], rf);
            if (!(pe > 0.0f)) continue;
            const int32_t id = a.id_map ? a.id_map[s.ids[i]] : (int32_t)s.ids[i];
            if (__fmul_rn(race_score_est(q.key, id, pe), 1.0f - RACE_EST_TOL) > bound) continue;
            const double sc = race_score(q.key, id, pe);
            if (sc < best) { best = sc; besti = i; }
        }
    }
    block_argmin<PREP_THREADS>(&best, &besti, wd, wi);
    if (tid == 0) {
        s.st->pbest[c] = best;
        s.st->pidx[c] = besti;
    }
}

__global__ void full_final_kernel(const __grid_constant__ launch_args a, const full_group g) {
    const uint32_t j = blockIdx.x;
    if (threadIdx.x != 0 || j >= g.n) return;
    const uint32_t row = g.row[j];
    const full_slot s = full_slot_of(a, g, j);
    row_out_t *o = a.out + row;
    uint32_t *dist = a.dist ? a.dist + (size_t)row * (1u + 2u * a.dist_cap) : nullptr;
    if (s.st->status != PULSAR_GPU_SAMPLE_OK) {
        o->token = -1;
        o->status = s.st->status;
        if (dist) dist[0] = 0;
        return;
    }
    double best = __longlong_as_double(0x7ff0000000000000ll);
    uint32_t bi = UINT32_MAX;
    for (uint32_t c = 0; c < PREP_CHUNKS; c++)   /* chunks in index order: the first strict minimum */
        if (s.st->pbest[c] < best) { best = s.st->pbest[c]; bi = s.st->pidx[c]; }
    const uint32_t w = bi == UINT32_MAX ? 0u : bi;
    o->token = a.id_map ? a.id_map[s.ids[w]] : (int32_t)s.ids[w];
    o->status = PULSAR_GPU_SAMPLE_OK;
    if (dist) dist[0] = s.st->f;
}

/* ---- logprobs (the host scorers' bytes: tests/sampler_host_ref.cpp pulsar_logits_top_logprobs / _token_logprob) --- */

constexpr uint32_t LP_GROUP = 8;   /* rows a logprob pass takes (its scratch: LP_GROUP rows of doubles) */
constexpr uint32_t LP_TK_CHUNKS = 32, LP_TK_THREADS = 256, LP_EXP_CHUNKS = 4u * 32u;
struct lp_head {
    float maxv;
    int32_t maxi;
};

struct lp_args {
    const float *logits;
    uint64_t row_stride;
    uint32_t n_vocab;
    const uint32_t *mask;
    uint32_t mask_words;
    uint32_t mask_row[PULSAR_GPU_SAMPLE_ROWS_MAX];
    const int32_t *token;      /* the scored token of row r at token[r * token_stride], or NULL */
    uint32_t token_stride;
    uint32_t top_k;
    pulsar_gpu_logprob_row *out;
    uint32_t row0;             /* this launch's first row (a launch takes LP_GROUP rows) */
    lp_head *head;             /* [LP_GROUP] each row's first maximum */
    double *ex;                /* [LP_GROUP][n_vocab] exp(v - max) of the finite values, 0 elsewhere */
    const double *sums;        /* [LP_GROUP] their serial sums (sr_sum_kernel) */
    uint64_t *tk;              /* [LP_GROUP][LP_TK_CHUNKS][PULSAR_GPU_LOGPROB_TOPK_MAX] each chunk's best keys */
    uint32_t *tk_n;            /* [LP_GROUP][LP_TK_CHUNKS][2]: each chunk's kept count and finite count */
};

struct lp_smem {
    uint64_t keys[1024];   /* the chunks' best keys, merged (LP_TK_CHUNKS * PULSAR_GPU_LOGPROB_TOPK_MAX <= 1024) */
    uint32_t cnt;
};
static_assert(LP_TK_CHUNKS * PULSAR_GPU_LOGPROB_TOPK_MAX <= 1024u, "the merge's store");


__device__ __forceinline__ row_view lp_view(const lp_args &a, uint32_t row) {
    row_view rv;
    rv.x = a.logits + (size_t)row * a.row_stride;
    rv.n = a.n_vocab;
    rv.mask = a.mask && a.mask_row[row] != PULSAR_GPU_SAMPLE_NO_MASK
                  ? a.mask + (size_t)a.mask_row[row] * a.mask_words : nullptr;
    return rv;
}

/* a chunk of a row (32-aligned, so a mask view stays word-aligned) */
__device__ __forceinline__ void lp_chunk(uint32_t n, uint32_t c, uint32_t *lo, uint32_t *hi) {
    const uint32_t per = ((n + LP_TK_CHUNKS - 1u) / LP_TK_CHUNKS + 31u) & ~31u;
    *lo = min(c * per, n);
    *hi = min(*lo + per, n);
}

/* pass 2b, LP_TK_CHUNKS blocks a row: each chunk's best min(k, its finite count) keys (select_topk's), ids made the
 * row's -- the row's best k are among the chunks' best k, in the same total order */
__global__ void __launch_bounds__(LP_TK_THREADS) logprobs_topk_kernel(const __grid_constant__ lp_args a) {
    __shared__ uint64_t keys[1024];
    __shared__ uint32_t hist[256], redu[LP_TK_THREADS], cnt;
    const uint32_t c = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    const row_view rv = lp_view(a, a.row0 + j);
    uint32_t lo, hi;
    lp_chunk(rv.n, c, &lo, &hi);
    row_view cv = rv;
    cv.x = rv.x + lo;
    cv.n = hi - lo;
    cv.mask = rv.mask ? rv.mask + lo / 32u : nullptr;
    uint32_t nfin = 0;
    for (uint32_t i = tid; i < cv.n; i += LP_TK_THREADS) nfin += finite_bits(cv(i)) ? 1u : 0u;
    nfin = block_sum_u32<LP_TK_THREADS>(nfin, redu);
    const uint32_t kc = min(min(a.top_k, rv.n), nfin);
    if (kc > 0) select_topk<LP_TK_THREADS>(cv, kc, keys, hist, &cnt, redu);
    uint64_t *out = a.tk + ((size_t)j * LP_TK_CHUNKS + c) * PULSAR_GPU_LOGPROB_TOPK_MAX;
    for (uint32_t i = tid; i < kc; i += LP_TK_THREADS) out[i] = keys[i] + lo;   /* the id is the low word */
    if (tid == 0) {
        a.tk_n[((size_t)j * LP_TK_CHUNKS + c) * 2u] = kc;
        a.tk_n[((size_t)j * LP_TK_CHUNKS + c) * 2u + 1u] = nfin;
    }
}

/* pass 1, one block a row: the first finite maximum */
__global__ void __launch_bounds__(SMP_THREADS) logprobs_max_kernel(const __grid_constant__ lp_args a) {
    __shared__ float redv[SMP_THREADS];
    __shared__ int32_t redi[SMP_THREADS];
    const uint32_t j = blockIdx.x;
    float maxv;
    int32_t maxi;
    block_argmax<SMP_THREADS>(lp_view(a, a.row0 + j), redv, redi, &maxv, &maxi);
    if (threadIdx.x == 0) a.head[j] = lp_head{maxv, maxi};
}

/* pass 2, LP_EXP_CHUNKS blocks a row: exp(v - max) in double, the host scorer's (plm_exp), for every finite value */
__global__ void __launch_bounds__(PREP_THREADS) logprobs_exp_kernel(const __grid_constant__ lp_args a) {
    const uint32_t j = blockIdx.y, c = blockIdx.x;
    if (a.head[j].maxi < 0) return;
    const row_view rv = lp_view(a, a.row0 + j);
    const double dmax = (double)a.head[j].maxv;
    double *ex = a.ex + (size_t)j * a.n_vocab;
    const uint32_t per = (rv.n + LP_EXP_CHUNKS - 1u) / LP_EXP_CHUNKS;
    const uint32_t lo = min(c * per, rv.n), hi = min(lo + per, rv.n);
    for (uint32_t i = lo + threadIdx.x; i < hi; i += PREP_THREADS) {
        const float v = rv(i);
        ex[i] = finite_bits(v) ? plm_exp(__dsub_rn((double)v, dmax)) : 0.0;   /* +0.0: the host skipped it */
    }
}

/* pass 3 (after the double sum of exp(v - max) over the finite values in id order -- sr_sum_kernel, the host's
 * accumulation), one block a row: log(sum) once; the scored token as _token_logprob's compiled arithmetic has it
 * ((logit - max) - log(sum)), the top-k as _top_logprobs' does (logit - (log(sum) + max)) -- the host's -ffast-math
 * build reassociated one and not the other. */
__global__ void __launch_bounds__(SMP_THREADS) logprobs_rows_kernel(const __grid_constant__ lp_args a) {
    __shared__ lp_smem sm;
    const uint32_t tid = threadIdx.x, j = blockIdx.x, row = a.row0 + j;
    const row_view rv = lp_view(a, row);
    pulsar_gpu_logprob_row *o = a.out + row;
    if (a.head[j].maxi < 0) {
        if (tid == 0) { o->n_top = 0; o->have_token = 0; }
        return;
    }
    const double dmax = (double)a.head[j].maxv;
    const double lsum = plm_log(a.sums[j]);
    if (tid == 0) {
        const int32_t t = a.token ? a.token[(size_t)row * a.token_stride] : -1;
        o->have_token = t >= 0 && (uint32_t)t < rv.n ? 1u : 0u;
        if (o->have_token) {
            const float lg = rv((uint32_t)t);
            o->token.id = t;
            o->token.logit = lg;
            o->token.logprob = finite_bits(lg) ? __double2float_rn(__dsub_rn(__dsub_rn((double)lg, dmax), lsum))
                                               : PULSAR_GPU_LOGPROB_NEG_INF;
        }
    }
    const uint32_t k = min(a.top_k, rv.n);
    if (k == 0) {
        if (tid == 0) o->n_top = 0;
        return;
    }
    /* the chunks' best, merged: one sort of their union, its first min(k, finite) */
    if (tid == 0) {
        uint32_t m = 0, nfin = 0;
        for (uint32_t c = 0; c < LP_TK_CHUNKS; c++) {
            const uint32_t *cn = a.tk_n + ((size_t)j * LP_TK_CHUNKS + c) * 2u;
            const uint64_t *ck = a.tk + ((size_t)j * LP_TK_CHUNKS + c) * PULSAR_GPU_LOGPROB_TOPK_MAX;
            for (uint32_t i = 0; i < cn[0]; i++) sm.keys[m++] = ck[i];
            nfin += cn[1];
        }
        sm.cnt = m | (min(k, nfin) << 16);
    }
    __syncthreads();
    const uint32_t m = sm.cnt & 0xffffu, kf = sm.cnt >> 16;
    bitonic_sort<SMP_THREADS>(sm.keys, m);
    const double logsum = __dadd_rn(lsum, dmax);
    for (uint32_t i = tid; i < k; i += SMP_THREADS) {
        pulsar_gpu_token_score ts;
        if (i < kf) {
            ts.id = (int32_t)(uint32_t)sm.keys[i];
            ts.logit = rv((uint32_t)ts.id);
            ts.logprob = __double2float_rn(__dsub_rn((double)ts.logit, logsum));
        } else {
            ts.id = -1;
            ts.logit = PULSAR_GPU_LOGPROB_NEG_INF;
            ts.logprob = PULSAR_GPU_LOGPROB_NEG_INF;
        }
        o->top[i] = ts;
    }
    if (tid == 0) o->n_top = k;
}

/* ---- the speculative walk (L288 sdet's rule): a draft is accepted iff the target drew it ------------------------ */

struct walk_args {
    const row_out_t *rows;     /* the sampler's rows of the verify */
    uint32_t n_runs;
    pulsar_gpu_spec_run run[PULSAR_GPU_SAMPLE_ROWS_MAX];
    pulsar_gpu_spec_walk_out *out;
};

/* One thread a run: rows row0 .. row0 + k are the target's draws after the base and after each draft; the walk
 * keeps drafts while the row's draw IS the draft, and the first draw that is not -- or row k's on a full accept --
 * is the carry.  A refused row stops the walk with its status. */
__global__ void spec_walk_kernel(const __grid_constant__ walk_args a) {
    const uint32_t r = threadIdx.x;
    if (r >= a.n_runs) return;
    const pulsar_gpu_spec_run &run = a.run[r];
    uint32_t c = 0;
    int32_t carry = -1;
    uint32_t status = PULSAR_GPU_SAMPLE_OK, n_label = 0;
    pulsar_gpu_spec_walk_out &o = a.out[r];
    for (;;) {
        const row_out_t t = a.rows[run.row0 + c];
        if (t.status != PULSAR_GPU_SAMPLE_OK || t.token < 0) {
            status = t.status != PULSAR_GPU_SAMPLE_OK ? t.status : PULSAR_GPU_SAMPLE_REFUSED_EMPTY;
            break;
        }
        if (c == run.k) { carry = t.token; break; }
        const bool ok = t.token == run.pend[c];
        o.label[c] = (run.greedy_rows >> c) & 1u ? (ok ? 1.0f : 0.0f) : t.probe_p;
        o.hit[c] = ok ? 1u : 0u;
        n_label = c + 1u;
        if (!ok) { carry = t.token; break; }
        c++;
    }
    o.n_label = n_label;
    a.out[r].commit = c;
    a.out[r].carry = carry;
    a.out[r].status = status;
}

/* ---- a block drafter's sampled chains (L298 rows): one block a chain, every slot's draw in it -------------------
 *
 * A slot's row is a sparse row of the vocabulary: the head's few candidates, every other id -inf.  THE sampler on it
 * reads only its finite entries, so the block hands it those, in ascending id (the order its sums and tie-breaks
 * read) with the ids as the row's id map, and plans the knobs against the vocabulary's width (the sparse min-p
 * predicate reads it).  The same function as every other draw (sample_row), so the same bytes as the full-width row;
 * the next slot's row follows from the pick, so the walk stays in the block -- one launch, one readback. */

constexpr uint32_t CHAIN_THREADS = 256;

struct chain_args {
    const int32_t *cand;
    const float *scores;
    uint32_t slots, topk, vocab;
    pulsar_gpu_chain_req req[PULSAR_GPU_CHAINS_MAX];
    int32_t *out;
};

__global__ void __launch_bounds__(CHAIN_THREADS) sample_chains_kernel(const __grid_constant__ chain_args c) {
    __shared__ row_smem<CHAIN_THREADS> sm;
    __shared__ launch_args la;
    __shared__ float row[PULSAR_GPU_CHAIN_TOPK_MAX];
    __shared__ int32_t ids[PULSAR_GPU_CHAIN_TOPK_MAX];
    __shared__ row_out_t o;
    __shared__ uint32_t pend, prev, stop;
    const uint32_t b = blockIdx.x, tid = threadIdx.x, TK = c.topk;
    const pulsar_gpu_chain_req &q = c.req[b];
    const int32_t *cand = c.cand + (size_t)q.chain * c.slots * TK;
    const float *sc = c.scores + (size_t)q.chain * c.slots * TK * TK;
    int32_t *out = c.out + (size_t)b * c.slots;
    if (tid == 0) {
        memset(&la, 0, sizeof(la));
        la.logits = row;
        la.n_rows = 1;
        la.plan_vocab = c.vocab;
        la.id_map = ids;
        la.out = &o;
        la.pending = &pend;
        la.req[0].temperature = q.temperature;
        la.req[0].top_p = q.top_p;
        la.req[0].min_p = q.min_p;
        la.req[0].top_k = q.top_k;
        la.req[0].mask_row = PULSAR_GPU_SAMPLE_NO_MASK;
        la.req[0].probe = -1;
        prev = 0;
        stop = 0;
    }
    for (uint32_t j = 0; j < c.slots; j++) {
        __syncthreads();
        if (stop || j >= q.n) {
            if (tid == 0) out[j] = -1;
            continue;
        }
        if (tid == 0) {   /* the slot's finite entries in ascending id, a repeated id at its last score */
            const int32_t *ci = cand + (size_t)j * TK;
            const float *rw = sc + ((size_t)j * TK + prev) * TK;
            uint32_t m = 0;
            for (uint32_t k = 0; k < TK; k++) {
                const int32_t id = ci[k];
                if (id < 0 || (uint32_t)id >= c.vocab) continue;
                uint32_t at = 0;
                while (at < m && ids[at] < id) at++;
                if (at < m && ids[at] == id) {
                    row[at] = rw[k];
                    continue;
                }
                for (uint32_t t = m; t > at; t--) { ids[t] = ids[t - 1]; row[t] = row[t - 1]; }
                ids[at] = id;
                row[at] = rw[k];
                m++;
            }
            la.n_vocab = m;
            la.row_stride = m;
            la.req[0].key = q.key[j];
        }
        __syncthreads();
        sample_row<CHAIN_THREADS>(la, 0u, sm, nullptr, nullptr);
        __syncthreads();
        if (tid == 0) {
            const int32_t tok = o.status == PULSAR_GPU_SAMPLE_OK ? o.token : -1;
            out[j] = tok;
            uint32_t pick = 0;
            const int32_t *ci = cand + (size_t)j * TK;
            while (pick < TK && ci[pick] != tok) pick++;
            if (tok < 0 || pick == TK) stop = 1;
            else prev = pick;
        }
    }
}

}  // namespace

/* the full arm's sort scratch for a row of n_vocab (the double-buffer form: the slot's own arrays alternate) */
static uint64_t full_sort_bytes(uint32_t n_vocab) {
    size_t bytes = 0;
    cub::DoubleBuffer<uint32_t> k(nullptr, nullptr), v(nullptr, nullptr);
    if (cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, v, (int)n_vocab, 0, 32, cudaStreamPerThread) != cudaSuccess)
        return UINT64_MAX / 8u;   /* refused below: no scratch is that large */
    return (bytes + 255u) & ~255ull;
}

static uint64_t full_sort_off(uint32_t n_vocab) {
    return full_state_off(n_vocab) + ((sizeof(full_state) + 255u) & ~(uint64_t)255u);
}

/* a slot: the five row arrays, then the slot kernel's radix counts or the full arm's state and sort scratch */
static uint64_t sample_slot_bytes(uint32_t n_vocab) {
    const uint64_t counts = (uint64_t)n_vocab * 5u * sizeof(uint32_t) + 17ull * SLOT_THREADS * sizeof(uint32_t);
    const uint64_t full = full_sort_off(n_vocab) + full_sort_bytes(n_vocab);
    return ((counts > full ? counts : full) + 255u) & ~255ull;
}

/* scratch: the slots, the pending flags, the rows' prep records */
static uint64_t sample_prep_off(uint32_t n_vocab) {
    return ((uint64_t)SMP_SLOTS * sample_slot_bytes(n_vocab) +
            (uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * sizeof(uint32_t) + 255u) & ~255ull;
}

static uint64_t sample_pvals_off(uint32_t n_vocab) {
    return (sample_prep_off(n_vocab) + (uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * sizeof(row_prep) + 255u) & ~255ull;
}

/* the GPU-wide serial sums' scratch: the tiles' status and approximate sums, the tickets, two passes' results */
struct sr_scratch {
    sr_status *st;
    double *approx;
    uint32_t *ticket;
    double *sum[2];
    uint32_t *stop[2];
};
static uint32_t sr_tiles(uint32_t n_vocab) { return (n_vocab + SR_TILE - 1u) / SR_TILE; }
static uint64_t sr_scratch_bytes(uint32_t n_vocab) {
    const uint64_t t = sr_tiles(n_vocab);
    return (uint64_t)SR_ROWS * t * (sizeof(sr_status) + sizeof(double)) + (uint64_t)SR_ROWS * 4u * 6u + 256u;
}
static sr_scratch sr_carve(uint8_t *p, uint32_t n_vocab) {
    const uint64_t t = sr_tiles(n_vocab);
    sr_scratch c;
    c.st = (sr_status *)p;
    p += SR_ROWS * t * sizeof(sr_status);
    c.approx = (double *)p;
    p += SR_ROWS * t * sizeof(double);
    for (int k = 0; k < 2; k++) { c.sum[k] = (double *)p; p += SR_ROWS * sizeof(double); }
    c.ticket = (uint32_t *)p;
    p += SR_ROWS * sizeof(uint32_t);
    for (int k = 0; k < 2; k++) { c.stop[k] = (uint32_t *)p; p += SR_ROWS * sizeof(uint32_t); }
    return c;
}

/* one GPU-wide serial sum over sa's rows into pass k's results */
template <typename T, bool HAS_PRED>
static bool sr_run(sr_args &sa, const sr_scratch &c, int k, uint32_t n_vocab) {
    sa.tiles = sr_tiles(n_vocab);
    sa.st = c.st;
    sa.approx = c.approx;
    sa.ticket = c.ticket;
    sa.sum = c.sum[k];
    sa.stop = c.stop[k];
    const dim3 grid(sa.tiles, sa.n_rows);
    sr_approx_kernel<T><<<grid, SR_THREADS>>>(sa);
    sr_sum_kernel<T, HAS_PRED><<<grid, SR_THREADS>>>(sa);
    sr_final_kernel<<<1, 64>>>(sa, HAS_PRED);
    static_assert(SR_ROWS <= 64, "sr_final_kernel's one block");
    return cuda_ok(cudaGetLastError(), "device serial sum");
}

static uint64_t sample_sr_off(uint32_t n_vocab) {
    return (sample_pvals_off(n_vocab) + (uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * n_vocab * sizeof(float) + 255u) &
           ~255ull;
}

uint64_t pulsar_gpu_sample_scratch_bytes(uint32_t n_vocab) {
    return sample_sr_off(n_vocab) + sr_scratch_bytes(n_vocab);
}

int pulsar_gpu_sample_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                           uint32_t n_vocab, uint32_t logits_rows, uint32_t n_rows, const pulsar_gpu_sample_req *reqs,
                           const pulsar_gpu_tensor *mask, uint32_t mask_words, const pulsar_gpu_tensor *id_map,
                           pulsar_gpu_tensor *scratch, pulsar_gpu_tensor *out, uint64_t out_offset_bytes,
                           pulsar_gpu_tensor *dist, uint32_t dist_cap) {
    if (!logits || !reqs || !scratch || !out || n_rows == 0 || n_rows > PULSAR_GPU_SAMPLE_ROWS_MAX || n_vocab == 0 ||
        row_stride < n_vocab || logits_offset_bytes > logits->bytes ||
        logits_rows == 0 ||
        (uint64_t)(logits_rows - 1u) * row_stride * sizeof(float) + (uint64_t)n_vocab * sizeof(float) >
            logits->bytes - logits_offset_bytes ||
        scratch->bytes < pulsar_gpu_sample_scratch_bytes(n_vocab) ||
        out_offset_bytes > out->bytes || out->bytes - out_offset_bytes < (uint64_t)n_rows * sizeof(row_out_t) ||
        (mask && mask_words < (n_vocab + 31u) / 32u) || (id_map && id_map->bytes < (uint64_t)n_vocab * 4u) ||
        (dist && dist->bytes < (uint64_t)n_rows * (1u + 2u * dist_cap) * sizeof(uint32_t))) {
        fprintf(stderr, "pulsar: device sampler: bad launch (%u rows of %u, stride %u) -- refusing\n", n_rows, n_vocab,
                row_stride);
        return 0;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        if (reqs[r].row >= logits_rows) {
            fprintf(stderr, "pulsar: device sampler: draw %u reads row %u of %u -- refusing\n", r, reqs[r].row,
                    logits_rows);
            return 0;
        }
        if (reqs[r].mask_row != PULSAR_GPU_SAMPLE_NO_MASK &&
            (!mask || (uint64_t)(reqs[r].mask_row + 1u) * mask_words * sizeof(uint32_t) > mask->bytes)) {
            fprintf(stderr, "pulsar: device sampler: row %u names mask row %u outside the mask -- refusing\n", r,
                    reqs[r].mask_row);
            return 0;
        }
    }
    launch_args a;
    memset(&a, 0, sizeof(a));
    a.logits = (const float *)((const char *)logits->ptr + logits_offset_bytes);
    a.row_stride = row_stride;
    a.n_vocab = n_vocab;
    a.plan_vocab = n_vocab;
    a.n_rows = n_rows;
    a.mask = mask ? (const uint32_t *)mask->ptr : nullptr;
    a.mask_words = mask_words;
    a.id_map = id_map ? (const int32_t *)id_map->ptr : nullptr;
    memcpy(a.req, reqs, (size_t)n_rows * sizeof(*reqs));
    a.out = (row_out_t *)((char *)out->ptr + out_offset_bytes);
    a.slot_bytes = sample_slot_bytes(n_vocab);
    a.slots = (uint8_t *)scratch->ptr;
    a.pending = (uint32_t *)((uint8_t *)scratch->ptr + (uint64_t)SMP_SLOTS * a.slot_bytes);
    a.dist = dist ? (uint32_t *)dist->ptr : nullptr;
    a.dist_cap = dist_cap;
    /* the prep: the rows' scans over PREP_CHUNKS blocks a row, the top-k rows' radix levels when a row is one */
    row_prep *prep = (row_prep *)((uint8_t *)scratch->ptr + sample_prep_off(n_vocab));
    a.pvals = (float *)((uint8_t *)scratch->ptr + sample_pvals_off(n_vocab));
    const sr_scratch src = sr_carve((uint8_t *)scratch->ptr + sample_sr_off(n_vocab), n_vocab);
    bool any_topk = false;
    for (uint32_t r = 0; r < n_rows; r++) {
        any_topk |= reqs[r].top_k > 0;
        if (row_on_full_arm(reqs[r])) a.full_rows |= 1ull << r;
        if (row_idsum(reqs[r], n_vocab)) a.idsum_rows |= 1ull << r;
    }
    a.idsums = src.sum[0];
    const dim3 pg(PREP_CHUNKS, n_rows);
    if (!cuda_ok(cudaMemset2DAsync(prep, sizeof(row_prep), 0, offsetof(row_prep, zero_end), n_rows,
                                   cudaStreamPerThread), "device sampler prep zero"))
        return 0;
    prep_scan_kernel<<<pg, PREP_THREADS>>>(a, prep);
    if (any_topk) {
        prep_decide_kernel<1><<<n_rows, PREP_THREADS>>>(a, prep);
        prep_hist_kernel<2><<<pg, PREP_THREADS>>>(a, prep);
        prep_decide_kernel<2><<<n_rows, PREP_THREADS>>>(a, prep);
        prep_hist_kernel<3><<<pg, PREP_THREADS>>>(a, prep);
        prep_decide_kernel<3><<<n_rows, PREP_THREADS>>>(a, prep);
    }
    prep_collect_kernel<<<pg, PREP_THREADS>>>(a, prep);
    if (a.idsum_rows) {   /* the id-order sums, over the GPU, before the row kernel reads them */
        prep_pvals_kernel<<<dim3(WIDE_CHUNKS, n_rows), PREP_THREADS>>>(a, prep);
        sr_args sa;
        memset(&sa, 0, sizeof(sa));
        sa.n_rows = n_rows;
        for (uint32_t r = 0; r < n_rows; r++) {
            sa.row[r].vals = a.pvals + (size_t)r * n_vocab;
            sa.row[r].n = (a.idsum_rows >> r) & 1u ? n_vocab : 0u;
        }
        if (!sr_run<float, false>(sa, src, 0, n_vocab)) return 0;
    }
    sample_rows_kernel<<<n_rows, SMP_THREADS>>>(a, prep);
    if (!cuda_ok(cudaGetLastError(), "device sampler rows")) return 0;
    sample_slots_kernel<<<SMP_SLOTS, SLOT_THREADS>>>(a);
    if (!cuda_ok(cudaGetLastError(), "device sampler slots")) return 0;
    /* the full arm's rows, a slot each, FULL_GROUP at a time (the slot kernel above is done with the slots) */
    const uint64_t sort_bytes = full_sort_bytes(n_vocab);
    for (uint32_t r0 = 0; r0 < n_rows;) {
        full_group g;
        memset(&g, 0, sizeof g);
        for (; r0 < n_rows && g.n < FULL_GROUP; r0++)
            if ((a.full_rows >> r0) & 1u) g.row[g.n++] = r0;
        if (g.n == 0) break;
        full_keygen_kernel<<<dim3(WIDE_CHUNKS, g.n), PREP_THREADS>>>(a, g, prep);
        for (uint32_t j = 0; j < g.n; j++) {
            uint8_t *base = a.slots + (uint64_t)j * a.slot_bytes;
            uint32_t *k0 = (uint32_t *)base, *v0 = k0 + n_vocab, *k1 = v0 + n_vocab, *v1 = k1 + n_vocab;
            cub::DoubleBuffer<uint32_t> kb(k0, k1), vb(v0, v1);
            size_t tb = sort_bytes;
            if (!cuda_ok(cub::DeviceRadixSort::SortPairs(base + full_sort_off(n_vocab), tb, kb, vb, (int)n_vocab, 0,
                                                          32, cudaStreamPerThread),
                         "device sampler full-arm sort"))
                return 0;
            /* the pass count is fixed by the key width, so where the sorted ids land is the same for every slot */
            g.sorted_in_alt = vb.selector;
        }
        full_probs_kernel<<<dim3(WIDE_CHUNKS, g.n), PREP_THREADS>>>(a, g, prep);
        full_floor_kernel<<<dim3(WIDE_CHUNKS, g.n), PREP_THREADS>>>(a, g);
        {   /* the sum in sorted order, the mass test, then the cut's walk (both over the GPU) */
            sr_args sa;
            memset(&sa, 0, sizeof(sa));
            sa.n_rows = g.n;
            for (uint32_t j = 0; j < g.n; j++) {
                uint8_t *base = a.slots + (uint64_t)j * a.slot_bytes;
                full_state *st = (full_state *)(base + full_state_off(n_vocab));
                row_plan pl;
                plan_knobs(reqs[g.row[j]], &pl);
                sa.row[j].vals = base + (uint64_t)n_vocab * 4u * sizeof(uint32_t);   /* the probabilities */
                sa.row[j].n_ptr = &st->n;
                sa.row[j].top_p = pl.top_p;
            }
            if (!sr_run<float, false>(sa, src, 0, n_vocab)) return 0;
            full_mass_kernel<<<1, 32>>>(a, g, src.sum[0]);
            for (uint32_t j = 0; j < g.n; j++) {
                full_state *st = (full_state *)(a.slots + (uint64_t)j * a.slot_bytes + full_state_off(n_vocab));
                sa.row[j].n_ptr = &st->ncut;
                sa.row[j].sum_ptr = src.sum[0] + j;
            }
            if (!sr_run<float, true>(sa, src, 1, n_vocab)) return 0;
            full_cut_kernel<<<1, 32>>>(a, g, src.sum[1], src.stop[1]);
        }
        full_race_bound_kernel<<<dim3(PREP_CHUNKS, g.n), PREP_THREADS>>>(a, g);
        full_race_kernel<<<dim3(PREP_CHUNKS, g.n), PREP_THREADS>>>(a, g);
        full_final_kernel<<<g.n, 32>>>(a, g);
        if (!cuda_ok(cudaGetLastError(), "device sampler full arm")) return 0;
    }
    return 1;
}

static uint64_t lp_sr_off(uint32_t n_vocab) { return (256u + (uint64_t)LP_GROUP * n_vocab * sizeof(double) + 255u) & ~255ull; }

static uint64_t lp_tk_off(uint32_t n_vocab) { return (lp_sr_off(n_vocab) + sr_scratch_bytes(n_vocab) + 255u) & ~255ull; }
static constexpr uint64_t LP_TK_BYTES =
    (uint64_t)LP_GROUP * LP_TK_CHUNKS * (PULSAR_GPU_LOGPROB_TOPK_MAX * sizeof(uint64_t) + 2u * sizeof(uint32_t));

uint64_t pulsar_gpu_logprobs_scratch_bytes(uint32_t n_vocab) {
    static_assert(LP_GROUP * sizeof(lp_head) <= 256u, "the logprob scratch's head");
    return lp_tk_off(n_vocab) + LP_TK_BYTES;
}

int pulsar_gpu_logprobs_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                             uint32_t n_vocab, uint32_t n_rows, const pulsar_gpu_tensor *mask, uint32_t mask_words,
                             const uint32_t *mask_rows, const pulsar_gpu_tensor *tokens, uint64_t tokens_offset_bytes,
                             uint32_t token_stride, uint32_t top_k, pulsar_gpu_tensor *scratch,
                             pulsar_gpu_tensor *out) {
    if (!logits || !out || !scratch || scratch->bytes < pulsar_gpu_logprobs_scratch_bytes(n_vocab) || n_rows == 0 || n_rows > PULSAR_GPU_SAMPLE_ROWS_MAX || n_vocab == 0 || row_stride < n_vocab ||
        top_k > PULSAR_GPU_LOGPROB_TOPK_MAX || logits_offset_bytes > logits->bytes ||
        (uint64_t)(n_rows - 1u) * row_stride * sizeof(float) + (uint64_t)n_vocab * sizeof(float) >
            logits->bytes - logits_offset_bytes ||
        out->bytes < (uint64_t)n_rows * sizeof(pulsar_gpu_logprob_row) || (mask && mask_words < (n_vocab + 31u) / 32u) ||
        (mask_rows && !mask)) {
        fprintf(stderr, "pulsar: device logprobs: bad launch (%u rows of %u, top %u) -- refusing\n", n_rows, n_vocab,
                top_k);
        return 0;
    }
    lp_args a;
    memset(&a, 0, sizeof(a));
    a.logits = (const float *)((const char *)logits->ptr + logits_offset_bytes);
    a.row_stride = row_stride;
    a.n_vocab = n_vocab;
    a.mask = mask ? (const uint32_t *)mask->ptr : nullptr;
    a.mask_words = mask_words;
    for (uint32_t r = 0; r < n_rows; r++) a.mask_row[r] = mask_rows ? mask_rows[r] : PULSAR_GPU_SAMPLE_NO_MASK;
    a.token = tokens ? (const int32_t *)((const char *)tokens->ptr + tokens_offset_bytes) : nullptr;
    a.token_stride = token_stride;
    a.top_k = top_k;
    a.out = (pulsar_gpu_logprob_row *)out->ptr;
    a.head = (lp_head *)scratch->ptr;
    a.ex = (double *)((uint8_t *)scratch->ptr + 256u);
    const sr_scratch src = sr_carve((uint8_t *)scratch->ptr + lp_sr_off(n_vocab), n_vocab);
    a.tk = (uint64_t *)((uint8_t *)scratch->ptr + lp_tk_off(n_vocab));
    a.tk_n = (uint32_t *)(a.tk + (size_t)LP_GROUP * LP_TK_CHUNKS * PULSAR_GPU_LOGPROB_TOPK_MAX);
    for (uint32_t r0 = 0; r0 < n_rows; r0 += LP_GROUP) {
        const uint32_t g = min(LP_GROUP, n_rows - r0);
        a.row0 = r0;
        logprobs_max_kernel<<<g, SMP_THREADS>>>(a);
        logprobs_exp_kernel<<<dim3(LP_EXP_CHUNKS, g), PREP_THREADS>>>(a);
        logprobs_topk_kernel<<<dim3(LP_TK_CHUNKS, g), LP_TK_THREADS>>>(a);
        sr_args sa;
        memset(&sa, 0, sizeof(sa));
        sa.n_rows = g;
        for (uint32_t j = 0; j < g; j++) {
            sa.row[j].vals = a.ex + (size_t)j * n_vocab;
            sa.row[j].n = n_vocab;
        }
        if (!sr_run<double, false>(sa, src, 0, n_vocab)) return 0;
        a.sums = src.sum[0];
        logprobs_rows_kernel<<<g, SMP_THREADS>>>(a);
        if (!cuda_ok(cudaGetLastError(), "device logprobs")) return 0;
    }
    return 1;
}

int pulsar_gpu_spec_walk(const pulsar_gpu_tensor *rows, uint64_t rows_offset_bytes, uint32_t n_rows,
                         const pulsar_gpu_spec_run *runs, uint32_t n_runs, pulsar_gpu_tensor *out,
                         uint64_t out_offset_bytes) {
    bool ok = rows && runs && out && n_runs > 0 && n_runs <= PULSAR_GPU_SAMPLE_ROWS_MAX &&
              rows_offset_bytes + (uint64_t)n_rows * sizeof(row_out_t) <= rows->bytes &&
              out_offset_bytes + (uint64_t)n_runs * sizeof(pulsar_gpu_spec_walk_out) <= out->bytes;
    for (uint32_t r = 0; ok && r < n_runs; r++) ok = runs[r].k <= 16u && runs[r].row0 + runs[r].k < n_rows;
    if (!ok) {
        fprintf(stderr, "pulsar: device spec walk: bad launch (%u runs over %u rows) -- refusing\n", n_runs, n_rows);
        return 0;
    }
    walk_args a;
    memset(&a, 0, sizeof(a));
    a.rows = (const row_out_t *)((const char *)rows->ptr + rows_offset_bytes);
    a.n_runs = n_runs;
    memcpy(a.run, runs, (size_t)n_runs * sizeof(*runs));
    a.out = (pulsar_gpu_spec_walk_out *)((char *)out->ptr + out_offset_bytes);
    spec_walk_kernel<<<1, 32>>>(a);
    return cuda_ok(cudaGetLastError(), "device spec walk");
}

/* ---- the rows' identity digest (the TP mirror's cross-rank check, L243; L302: the rows stay on the device) ------ */

/* One block: each thread folds a contiguous run of 64-bit words with a multiply-xor chain seeded by its position,
 * and the chains fold in thread order -- a deterministic function of the bytes, the row count and the width. */
__global__ static void __launch_bounds__(256) rows_digest_kernel(const uint64_t *w, uint64_t n_words, uint32_t n_rows,
                                                          uint32_t width, uint64_t *out) {
    __shared__ uint64_t part[256];
    const uint64_t K = 0x9E3779B97F4A7C15ull;
    const uint64_t chunk = (n_words + 255u) / 256u;
    const uint64_t lo = min((uint64_t)threadIdx.x * chunk, n_words), hi = min(lo + chunk, n_words);
    uint64_t h = (K ^ ((uint64_t)n_rows << 32 | width)) + threadIdx.x * K;
    for (uint64_t i = lo; i < hi; i++) h = (h ^ w[i]) * K;
    part[threadIdx.x] = h ^ (h >> 29);
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t f = K;
        for (uint32_t t = 0; t < 256; t++) f = (f ^ part[t]) * 0x100000001b3ull;
        *out = f;
    }
}

int pulsar_gpu_rows_digest(const pulsar_gpu_tensor *rows, uint64_t offset_bytes, uint32_t n_rows, uint32_t width,
                           pulsar_gpu_tensor *out, uint64_t out_offset_bytes) {
    const uint64_t bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!rows || !out || (offset_bytes & 7u) || (bytes & 7u) || offset_bytes + bytes > rows->bytes ||
        out_offset_bytes + 8u > out->bytes || (out_offset_bytes & 7u)) {
        fprintf(stderr, "pulsar: rows digest: bad launch (%u rows x %u) -- refusing\n", n_rows, width);
        return 0;
    }
    rows_digest_kernel<<<1, 256>>>((const uint64_t *)((const char *)rows->ptr + offset_bytes), bytes / 8u, n_rows,
                                   width, (uint64_t *)((char *)out->ptr + out_offset_bytes));
    return cuda_ok(cudaGetLastError(), "rows digest");
}

int pulsar_gpu_sample_chains(const pulsar_gpu_tensor *cand, uint64_t cand_offset_bytes, const pulsar_gpu_tensor *scores,
                             uint64_t scores_offset_bytes, uint32_t slots, uint32_t topk, uint32_t vocab,
                             const pulsar_gpu_chain_req *reqs, uint32_t n_chains, pulsar_gpu_tensor *out,
                             uint64_t out_offset_bytes) {
    bool ok = cand && scores && reqs && out && n_chains > 0 && n_chains <= PULSAR_GPU_CHAINS_MAX && slots > 0 &&
              slots <= PULSAR_GPU_CHAIN_SLOTS_MAX && topk > 0 && topk <= PULSAR_GPU_CHAIN_TOPK_MAX && vocab > 0 &&
              out_offset_bytes <= out->bytes && out->bytes - out_offset_bytes >= (uint64_t)n_chains * slots * 4u &&
              cand_offset_bytes <= cand->bytes && scores_offset_bytes <= scores->bytes;
    for (uint32_t i = 0; ok && i < n_chains; i++) {
        const uint64_t ch = reqs[i].chain;
        ok = reqs[i].n <= slots && (ch + 1u) * slots * topk * 4u <= cand->bytes - cand_offset_bytes &&
             (ch + 1u) * slots * topk * topk * 4u <= scores->bytes - scores_offset_bytes;
    }
    if (!ok) {
        fprintf(stderr, "pulsar: device sampler: bad chain launch (%u chains of %u slots, top %u) -- refusing\n",
                n_chains, slots, topk);
        return 0;
    }
    chain_args c;
    memset(&c, 0, sizeof(c));
    c.cand = (const int32_t *)((const char *)cand->ptr + cand_offset_bytes);
    c.scores = (const float *)((const char *)scores->ptr + scores_offset_bytes);
    c.slots = slots;
    c.topk = topk;
    c.vocab = vocab;
    memcpy(c.req, reqs, (size_t)n_chains * sizeof(*reqs));
    c.out = (int32_t *)((char *)out->ptr + out_offset_bytes);
    sample_chains_kernel<<<n_chains, CHAIN_THREADS>>>(c);
    return cuda_ok(cudaGetLastError(), "device sampler chains");
}

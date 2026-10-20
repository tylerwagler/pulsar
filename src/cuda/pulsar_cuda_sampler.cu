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
 * Shape: one block per row (rows run on separate SMs).  A row whose candidate set exceeds the block's shared-memory
 * capacity (min_p == 0 without top-k, a near-flat row) is finished by the slot kernel on a global scratch slot --
 * the same row function, a stable radix sort in global memory instead of the shared bitonic: the same order (the
 * key is total), so the same bytes. */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_libm.cuh"

namespace {

constexpr uint32_t SMP_THREADS = 1024;   /* the row kernel's block */
constexpr uint32_t SLOT_THREADS = 512;   /* the slot kernel's block (its radix count matrix is 16 x 512 words) */
constexpr uint32_t SMP_CAP = 2048;       /* candidates the row kernel sorts in shared memory */
constexpr uint32_t SMP_SLOTS = PULSAR_GPU_SAMPLE_SLOTS;

constexpr float MINP_PREFILTER_SLACK = 1.0f - 4e-6f;
constexpr float MINP_PREFILTER_MIN = 1e-30f;
constexpr float SPARSE_MINP_MIN = 0.02f;

enum : uint32_t { PATH_TOPK = 0, PATH_MINP = 1, PATH_FULL = 2 };

__device__ __forceinline__ bool finite_bits(float v) { return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u; }

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
    pulsar_gpu_sample_req req[PULSAR_GPU_SAMPLE_ROWS_MAX];
    row_out_t *out;            /* [n_rows] */
    uint32_t *pending;         /* [n_rows]: 1 = the row kernel left it to a slot */
    /* slot scratch: per slot keys [2][n_vocab] u32, ids [2][n_vocab] u32, probs [n_vocab] f32 */
    uint8_t *slots;
    uint64_t slot_bytes;
    /* the gate's dist dump: per row [n, ids[dist_cap], probs[dist_cap]] (u32 words), or NULL */
    uint32_t *dist;
    uint32_t dist_cap;
};

/* The shared memory of a row: the THREADS-wide scratch and (row kernel) the candidate store. */
template <uint32_t THREADS>
struct row_smem {
    float redv[THREADS];
    int32_t redi[THREADS];
    uint32_t redu[THREADS];
    float tile[THREADS];
    uint64_t keys[SMP_CAP];
    float probs[SMP_CAP];
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
    __device__ __forceinline__ int32_t id(uint32_t i) const {
        return skeys ? (int32_t)(uint32_t)skeys[i] : (int32_t)gids[i];
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

/* Serially sum probs[0, n) in order (one thread) -- the host's float accumulation, bit for bit. */
__device__ __forceinline__ float serial_sum(const float *p, uint32_t n) {
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) s = __fadd_rn(s, p[i]);
    return s;
}

/* THE row: argmax, the path, the candidates, the cut, the race.  `slot` NULL = the row kernel (shared candidates
 * only; a row past SMP_CAP is left pending), else the slot kernel's global scratch. */
template <uint32_t THREADS>
__device__ void sample_row(const launch_args &a, uint32_t row, row_smem<THREADS> &sm, uint8_t *slot) {
    const uint32_t tid = threadIdx.x;
    const pulsar_gpu_sample_req q = a.req[row];
    row_view rv;
    rv.x = a.logits + (size_t)row * a.row_stride;
    rv.n = a.n_vocab;
    rv.mask = a.mask && q.mask_row != PULSAR_GPU_SAMPLE_NO_MASK ? a.mask + (size_t)q.mask_row * a.mask_words : nullptr;
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
    float maxv;
    int32_t maxi;
    const bool greedy_flag = (q.flags & PULSAR_GPU_SAMPLE_GREEDY) != 0;
    float T = q.temperature, top_p = q.top_p, min_p = q.min_p;
    int top_k = q.top_k;
    if (!greedy_flag && (!finite_bits(T) || !finite_bits(top_p) || !finite_bits(min_p))) {
        finish(-1, PULSAR_GPU_SAMPLE_REFUSED_KNOB);
        return;
    }
    block_argmax<THREADS>(rv, sm.redv, sm.redi, &maxv, &maxi);
    if (greedy_flag || T <= 0.0f) {
        if (tid == 0 && maxi >= 0 && maxi == q.probe) o->probe_p = 1.0f;
        finish(maxi, maxi < 0 ? PULSAR_GPU_SAMPLE_REFUSED_EMPTY : PULSAR_GPU_SAMPLE_OK);
        dump_point(maxi);
        return;
    }
    if (maxi < 0) {
        finish(-1, PULSAR_GPU_SAMPLE_REFUSED_EMPTY);
        return;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > 1024) top_k = top_k <= 0 ? 0 : 1024;
    const float rcpT = __fdiv_rn(1.0f, T);
    const uint32_t path = top_k > 0 ? PATH_TOPK : min_p > MINP_PREFILTER_MIN ? PATH_MINP : PATH_FULL;

    cand_list cl;
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

    if (path == PATH_TOPK) {
        uint32_t nfin = 0;
        for (uint32_t i = tid; i < rv.n; i += THREADS) nfin += finite_bits(rv(i)) ? 1u : 0u;
        nfin = block_sum_u32<THREADS>(nfin, sm.redu);
        const uint32_t k = min((uint32_t)top_k, nfin);
        select_topk<THREADS>(rv, k, sm.keys, sm.hist, &sm.cnt, sm.redu);
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
        /* PATH_MINP / PATH_FULL: the candidate predicate, and (MINP, not full-nucleus) the id-order sum */
        bool full = false;
        float floor_logit = __uint_as_float(0xff800000u), prefilter = 0.0f;
        const uint32_t max_id = (uint32_t)maxi;
        if (path == PATH_MINP) {
            prefilter = __fmul_rn(min_p, MINP_PREFILTER_SLACK);
            const float band = __fmul_rn(1.4901161193847656e-7f, (float)rv.n);
            full = top_p == 1.0f && min_p >= (band > SPARSE_MINP_MIN ? band : SPARSE_MINP_MIN);
            /* a bound, not a decision: an entry under it has p < prefilter however it rounds */
            if (full) floor_logit = __fmaf_rn(__fsub_rn(logf(min_p), 1e-3f), T, maxv);
            maxlogit = maxv;
        }
        auto is_cand = [&](uint32_t i, float v, float *pp) -> bool {
            if (!finite_bits(v)) return false;
            if (path == PATH_FULL) return true;
            if (v < floor_logit && i != max_id) return false;
            const float p = plm_expf(__fmul_rn(__fsub_rn(v, maxlogit), rcpT));
            *pp = p;
            return p >= prefilter || i == max_id;
        };
        /* count, and the id-order serial sum where the arm keeps one */
        const bool idsum = path == PATH_MINP && !full;
        if (tid == 0) { sm.cnt = 0; sm.fsum = 0.0f; }
        __syncthreads();
        uint32_t mine = 0;
        for (uint32_t base = 0; base < rv.n; base += THREADS) {
            const uint32_t i = base + tid;
            float p = 0.0f;   /* a non-finite value adds +0 to the sum: the host skipped it, same bits */
            const bool c = i < rv.n && is_cand(i, rv(i), &p);
            mine += c ? 1u : 0u;
            if (idsum) {
                sm.tile[tid] = i < rv.n ? p : 0.0f;
                __syncthreads();
                if (tid == 0) {
                    float s = sm.fsum;
                    const uint32_t m = min(THREADS, rv.n - base);
                    for (uint32_t j = 0; j < m; j++) s = __fadd_rn(s, sm.tile[j]);
                    sm.fsum = s;
                }
                __syncthreads();
            }
        }
        const uint32_t n = block_sum_u32<THREADS>(mine, sm.redu);
        if (idsum) sum = sm.fsum;
        if (path == PATH_MINP && full) { sum = __uint_as_float(0x7f800000u); sum_inert = true; }
        cl.n = n;
        if (n <= SMP_CAP) {
            if (tid == 0) sm.cnt = 0;
            __syncthreads();
            for (uint32_t i = tid; i < rv.n; i += THREADS) {
                float p;
                if (is_cand(i, rv(i), &p)) sm.keys[atomicAdd(&sm.cnt, 1u)] = ((uint64_t)desc_key(rv(i)) << 32) | i;
            }
            __syncthreads();
            bitonic_sort<THREADS>(sm.keys, n);
            cl.skeys = sm.keys;
            if (path == PATH_FULL) maxlogit = rv((uint32_t)sm.keys[0]);
            for (uint32_t i = tid; i < n; i += THREADS)
                sm.probs[i] = plm_expf(__fmul_rn(__fsub_rn(rv((uint32_t)sm.keys[i]), maxlogit), rcpT));
            __syncthreads();
            cl.probs = sm.probs;
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
                float p;
                c += is_cand(i, rv(i), &p) ? 1u : 0u;
            }
            uint32_t pos = block_scan_incl<THREADS>(c, sm.redu) - c;
            for (uint32_t i = lo; i < hi; i++) {
                float p;
                if (is_cand(i, rv(i), &p)) { gk0[pos] = desc_key(rv(i)); gv0[pos] = i; pos++; }
            }
            __syncthreads();
            global_radix_sort<THREADS>(gk0, gv0, gk1, gv1, n, gcnt);
            cl.gids = gv0;
            if (path == PATH_FULL) maxlogit = rv(gv0[0]);
            for (uint32_t i = tid; i < n; i += THREADS)
                gp[i] = plm_expf(__fmul_rn(__fsub_rn(rv(gv0[i]), maxlogit), rcpT));
            __syncthreads();
            cl.probs = gp;
        }
        if (path == PATH_FULL) {
            if (tid == 0) sm.fsum = serial_sum(cl.probs, n);
            __syncthreads();
            sum = sm.fsum;
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

__global__ void __launch_bounds__(SMP_THREADS) sample_rows_kernel(const __grid_constant__ launch_args a) {
    __shared__ row_smem<SMP_THREADS> sm;
    if (threadIdx.x == 0) a.pending[blockIdx.x] = 0u;
    __syncthreads();
    sample_row<SMP_THREADS>(a, blockIdx.x, sm, nullptr);
}

/* The rows the row kernel left pending, round-robin over the slots in row order. */
__global__ void __launch_bounds__(SLOT_THREADS) sample_slots_kernel(const __grid_constant__ launch_args a) {
    __shared__ row_smem<SLOT_THREADS> sm;
    uint32_t rank = 0;
    for (uint32_t r = 0; r < a.n_rows; r++) {
        if (!a.pending[r]) continue;
        if (rank++ % SMP_SLOTS != blockIdx.x) continue;
        sample_row<SLOT_THREADS>(a, r, sm, a.slots + (size_t)blockIdx.x * a.slot_bytes);
        __syncthreads();
    }
}

/* ---- logprobs (the host's pulsar_logits_top_logprobs / _token_logprob, byte for byte) -------------------------- */

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
};

struct lp_smem {
    float redv[SMP_THREADS];
    int32_t redi[SMP_THREADS];
    uint32_t redu[SMP_THREADS];
    double tile[SMP_THREADS];
    uint64_t keys[1024];
    uint32_t hist[256];
    uint32_t cnt;
    double sum;
};

/* One block a row: the max (the first finite maximum's value), the double sum of exp(v - max) over the finite
 * values in id order (serial: the host's accumulation), log(sum) once; the scored token as _token_logprob's
 * compiled arithmetic has it ((logit - max) - log(sum)), the top-k as _top_logprobs' does (logit - (log(sum) +
 * max)) -- the host's -ffast-math build reassociated one and not the other. */
__global__ void __launch_bounds__(SMP_THREADS) logprobs_rows_kernel(const __grid_constant__ lp_args a) {
    __shared__ lp_smem sm;
    const uint32_t tid = threadIdx.x, row = blockIdx.x;
    row_view rv;
    rv.x = a.logits + (size_t)row * a.row_stride;
    rv.n = a.n_vocab;
    rv.mask = a.mask && a.mask_row[row] != PULSAR_GPU_SAMPLE_NO_MASK
                  ? a.mask + (size_t)a.mask_row[row] * a.mask_words : nullptr;
    pulsar_gpu_logprob_row *o = a.out + row;
    float maxv;
    int32_t maxi;
    block_argmax<SMP_THREADS>(rv, sm.redv, sm.redi, &maxv, &maxi);
    if (maxi < 0) {
        if (tid == 0) { o->n_top = 0; o->have_token = 0; }
        return;
    }
    const double dmax = (double)maxv;
    if (tid == 0) sm.sum = 0.0;
    __syncthreads();
    for (uint32_t base = 0; base < rv.n; base += SMP_THREADS) {
        const uint32_t i = base + tid;
        double e = 0.0;   /* the host skipped a non-finite value; +0.0 leaves the sum's bits alone */
        if (i < rv.n) {
            const float v = rv(i);
            if (finite_bits(v)) e = plm_exp(__dsub_rn((double)v, dmax));
        }
        sm.tile[tid] = e;
        __syncthreads();
        if (tid == 0) {
            double s = sm.sum;
            const uint32_t m = min(SMP_THREADS, rv.n - base);
            for (uint32_t j = 0; j < m; j++) s = __dadd_rn(s, sm.tile[j]);
            sm.sum = s;
        }
        __syncthreads();
    }
    const double lsum = plm_log(sm.sum);
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
    uint32_t nfin = 0;
    for (uint32_t i = tid; i < rv.n; i += SMP_THREADS) nfin += finite_bits(rv(i)) ? 1u : 0u;
    nfin = block_sum_u32<SMP_THREADS>(nfin, sm.redu);
    const uint32_t kf = min(k, nfin);
    select_topk<SMP_THREADS>(rv, kf, sm.keys, sm.hist, &sm.cnt, sm.redu);
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

}  // namespace

uint64_t pulsar_gpu_sample_slot_bytes(uint32_t n_vocab) {
    return (uint64_t)n_vocab * 5u * sizeof(uint32_t) + 17ull * SLOT_THREADS * sizeof(uint32_t);
}

uint64_t pulsar_gpu_sample_scratch_bytes(uint32_t n_vocab) {
    return (uint64_t)SMP_SLOTS * ((pulsar_gpu_sample_slot_bytes(n_vocab) + 255u) & ~255ull) +
           (uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * sizeof(uint32_t);
}

int pulsar_gpu_sample_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                           uint32_t n_vocab, uint32_t n_rows, const pulsar_gpu_sample_req *reqs,
                           const pulsar_gpu_tensor *mask, uint32_t mask_words, pulsar_gpu_tensor *scratch,
                           pulsar_gpu_tensor *out, uint64_t out_offset_bytes, pulsar_gpu_tensor *dist,
                           uint32_t dist_cap) {
    if (!logits || !reqs || !scratch || !out || n_rows == 0 || n_rows > PULSAR_GPU_SAMPLE_ROWS_MAX || n_vocab == 0 ||
        row_stride < n_vocab || logits_offset_bytes > logits->bytes ||
        (uint64_t)(n_rows - 1u) * row_stride * sizeof(float) + (uint64_t)n_vocab * sizeof(float) >
            logits->bytes - logits_offset_bytes ||
        scratch->bytes < pulsar_gpu_sample_scratch_bytes(n_vocab) ||
        out_offset_bytes > out->bytes || out->bytes - out_offset_bytes < (uint64_t)n_rows * sizeof(row_out_t) ||
        (mask && mask_words < (n_vocab + 31u) / 32u) ||
        (dist && dist->bytes < (uint64_t)n_rows * (1u + 2u * dist_cap) * sizeof(uint32_t))) {
        fprintf(stderr, "pulsar: device sampler: bad launch (%u rows of %u, stride %u) -- refusing\n", n_rows, n_vocab,
                row_stride);
        return 0;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
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
    a.n_rows = n_rows;
    a.mask = mask ? (const uint32_t *)mask->ptr : nullptr;
    a.mask_words = mask_words;
    memcpy(a.req, reqs, (size_t)n_rows * sizeof(*reqs));
    a.out = (row_out_t *)((char *)out->ptr + out_offset_bytes);
    a.slot_bytes = (pulsar_gpu_sample_slot_bytes(n_vocab) + 255u) & ~255ull;
    a.slots = (uint8_t *)scratch->ptr;
    a.pending = (uint32_t *)((uint8_t *)scratch->ptr + (uint64_t)SMP_SLOTS * a.slot_bytes);
    a.dist = dist ? (uint32_t *)dist->ptr : nullptr;
    a.dist_cap = dist_cap;
    sample_rows_kernel<<<n_rows, SMP_THREADS>>>(a);
    if (!cuda_ok(cudaGetLastError(), "device sampler rows")) return 0;
    sample_slots_kernel<<<SMP_SLOTS, SLOT_THREADS>>>(a);
    return cuda_ok(cudaGetLastError(), "device sampler slots");
}

int pulsar_gpu_logprobs_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                             uint32_t n_vocab, uint32_t n_rows, const pulsar_gpu_tensor *mask, uint32_t mask_words,
                             const uint32_t *mask_rows, const pulsar_gpu_tensor *tokens, uint64_t tokens_offset_bytes,
                             uint32_t token_stride, uint32_t top_k, pulsar_gpu_tensor *out) {
    if (!logits || !out || n_rows == 0 || n_rows > PULSAR_GPU_SAMPLE_ROWS_MAX || n_vocab == 0 || row_stride < n_vocab ||
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
    logprobs_rows_kernel<<<n_rows, SMP_THREADS>>>(a);
    return cuda_ok(cudaGetLastError(), "device logprobs");
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

/* pulsar_tessera.cu -- the host side of Tessera's fused window kernel, value (BF16, folded) family (L255).
 *
 * Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
 *
 * Contracts in pulsar_tessera.h.  The device code is tessera_routed_fused_window.cuh, upstream byte for byte;
 * this file replaces upstream's torch host entries (``routed_fused_forward``, ``dense_forward``,
 * ``token_sum``) and the Python owner's routing prep (``FusedRoutedWindowMoE._routing``) with plain CUDA.
 * The Params it fills are the ones those entries fill, field for field -- the comments name the upstream
 * source of each rule.  Built WITHOUT --use_fast_math (Makefile): the SwiGLU epilogue's expf and divide must
 * round as upstream's -O3 build rounds them. */
#include "pulsar_tessera.h"

#include <stdio.h>

namespace {

bool g_host_failed = false;   // set by the vendored header's host hooks during one launch call

void tessera_host_cuda(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return;
    fprintf(stderr, "pulsar: tessera: %s: %s\n", what, cudaGetErrorString(e));
    g_host_failed = true;
}

void tessera_host_refuse_pair(int mode, int r_lo, bool two, int tile_words, int K) {
    fprintf(stderr, "pulsar: tessera: the %s launch does not decode the run pair (r_lo %d, %s) that tile_words %d "
                    "fixes at K %d -- refusing\n",
            mode == 2 ? "down/dense" : "gate/up", r_lo, two ? "two runs" : "one run", tile_words, K);
    g_host_failed = true;
}

/* upstream's TORCH_CHECK(cond, msg) in launch (the piece-major guards, 37742e0f): a false condition refuses the
 * launch the same way -- the value family never sets piece_major, so these never fire here */
void tessera_host_check(bool ok, const char *msg) {
    if (ok) return;
    fprintf(stderr, "pulsar: tessera: %s -- refusing\n", msg);
    g_host_failed = true;
}

} // namespace

#define TESSERA_ROUTED_FUSED_FP8 0
#define TESSERA_HOST_CUDA(expr) tessera_host_cuda((expr), #expr)
#define TESSERA_HOST_CHECK(cond, msg) tessera_host_check((cond), (msg))
#define TESSERA_HOST_REFUSE_PAIR(mode, r_lo, two, tile_words, K) tessera_host_refuse_pair((mode), (r_lo), (two), (tile_words), (K))
/* nvcc #549-D "cm_nxt is used before its value is set" fires inside the vendored header (upstream 37742e0f
 * routed_fused_window.cu, the producer loop's advance_micro; present since 1381c3b7): the last chunk's advance
 * copies a next-chunk ColMap that was never loaded, and the copy is dead -- the next work item reloads cm_cur
 * (load_prev at kc0) before any read.  Upstream's own build carries the same copy; patching it would break the
 * byte-for-byte device code the gate holds to Tessera's build, so the one diagnostic is silenced for this one
 * include and restored after it. */
#pragma nv_diag_suppress 549
#include "tessera_routed_fused_window.cuh"
#pragma nv_diag_default 549

namespace {

constexpr int kRouteThreads = 256;
constexpr int kScanThreads = 1024;   // one block scans every expert: E <= kScanThreads

int sm_count() {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            sms = 0;
    }
    return sms;
}

int max_smem() {
    static int v = -1;
    if (v < 0) {
        int dev = 0;
        v = cudaGetDevice(&dev) == cudaSuccess ? max_dynamic_smem_bytes(dev) : 0;
    }
    return v;
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* The checks upstream's host entries make on one projection (``check_words``, ``check_run_tables``, the
 * tile_words range, ``check_slot``), plus the tile geometry the kernel assumes (N a multiple of BN). */
bool proj_ok(const pulsar_tessera_proj *w, int mode, const char *name) {
    const char *why = nullptr;
    if (!w || !w->words || !w->table || !w->init || !w->has_init || !w->wscale || !w->runs || !w->bdesc)
        why = "a null plane";
    else if (w->E <= 0 || w->K % BK != 0 || w->K < 4 * BK)
        why = "K must be a multiple of 32 and at least 128";
    else if (w->N <= 0 || w->N % BN != 0)
        why = "N must be a positive multiple of 128";
    else if (!aligned16(w->words) || w->words_stride % 4 != 0)
        why = "words must be 16-byte aligned with a stride that is a multiple of 4 words";
    else if (!aligned16(w->bdesc))
        why = "bdesc must be 16-byte aligned";
    else if (w->tile_words < w->K * 16 * RATE_MIN || w->tile_words > w->K * 16 * RATE_MAX || w->tile_words % 16 != 0)
        why = "tile_words must be 16 * (sum of the column rates), rates 1..8";
    else if (w->slot_words % 4 != 0 || w->slot_words < 4 || w->slot_words > SLOT_WORDS_MAX)
        why = "slot_words must be a multiple of 4 in [4, SLOT_WORDS_MAX]";
    else if (smem_bytes(mode, w->slot_words) > max_smem())
        why = "the launch's dynamic shared memory exceeds the device's opt-in limit";
    if (!why) return true;
    fprintf(stderr, "pulsar: tessera: %s (E %d, N %d, K %d, tile_words %d, slot_words %d): %s -- refusing\n", name,
            w ? w->E : 0, w ? w->N : 0, w ? w->K : 0, w ? w->tile_words : 0, w ? w->slot_words : 0, why);
    return false;
}

size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

/* The layout of one call's workspace; ``base`` null sizes it. */
struct MoeWs {
    int32_t *counts, *offsets, *item_off, *flat_sorted, *counters;
    float *rw_sorted;
    uint16_t *act, *routed;
};

size_t moe_ws_layout(uint8_t *base, int E, long P, int I, int H, MoeWs *o) {
    size_t used = 0;
    auto take = [&](size_t bytes) -> void * {
        const size_t off = align256(used);
        used = off + bytes;
        return base ? base + off : nullptr;
    };
    MoeWs m{};
    m.counts = (int32_t *)take((size_t)E * 4);
    m.offsets = (int32_t *)take((size_t)(E + 1) * 4);
    m.item_off = (int32_t *)take((size_t)(E + 1) * 4);
    m.counters = (int32_t *)take(2 * 4);
    m.flat_sorted = (int32_t *)take((size_t)P * 4);
    m.rw_sorted = (float *)take((size_t)P * 4);
    m.act = (uint16_t *)take((size_t)P * I * 2);
    m.routed = (uint16_t *)take((size_t)P * H * 2);
    if (o) *o = m;
    return used;
}

/* counts[e] = routes to expert e.  An id outside [0, E) is an impossible router output: trap. */
__global__ void route_count_kernel(const int32_t *__restrict__ ids, long P, int E, int32_t *__restrict__ counts) {
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int e = ids[p];
    if (e < 0 || e >= E) __trap();
    atomicAdd(&counts[e], 1);
}

/* offsets = exclusive prefix of counts, item_off = exclusive prefix of ceil(counts / BM) -- upstream's two
 * cumsums (``_routing``).  One block; each thread owns one expert (E <= kScanThreads). */
__global__ void __launch_bounds__(kScanThreads) route_scan_kernel(const int32_t *__restrict__ counts, int E,
                                                                   int32_t *__restrict__ offsets,
                                                                   int32_t *__restrict__ item_off) {
    __shared__ int32_t warp_c[kScanThreads / 32], warp_s[kScanThreads / 32];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const int c = e < E ? counts[e] : 0;
    const int s = (c + BM - 1) / BM;
    int ic = c, is = s;   // inclusive warp scans
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int uc = __shfl_up_sync(0xffffffffu, ic, o), us = __shfl_up_sync(0xffffffffu, is, o);
        if (lane >= o) { ic += uc; is += us; }
    }
    if (lane == 31) { warp_c[warp] = ic; warp_s[warp] = is; }
    __syncthreads();
    if (warp == 0) {
        int wc = warp_c[lane], ws = warp_s[lane];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int uc = __shfl_up_sync(0xffffffffu, wc, o), us = __shfl_up_sync(0xffffffffu, ws, o);
            if (lane >= o) { wc += uc; ws += us; }
        }
        warp_c[lane] = wc;
        warp_s[lane] = ws;
    }
    __syncthreads();
    const int bc = warp ? warp_c[warp - 1] : 0, bs = warp ? warp_s[warp - 1] : 0;
    if (e < E) {
        offsets[e + 1] = bc + ic;
        item_off[e + 1] = bs + is;
    }
    if (e == 0) { offsets[0] = 0; item_off[0] = 0; }
}

/* The stable argsort of the flattened ids (``torch.argsort(ids, stable=True)``): expert e's routes land at
 * offsets[e] .. offsets[e + 1] in route order.  One block per expert walks every route in order, ranking its
 * own with warp ballots, so the order is exactly upstream's.  rw_sorted follows (``weights[order]``). */
__global__ void __launch_bounds__(kRouteThreads) route_place_kernel(const int32_t *__restrict__ ids,
                                                                     const float *__restrict__ weights, long P,
                                                                     const int32_t *__restrict__ offsets,
                                                                     int32_t *__restrict__ flat_sorted,
                                                                     float *__restrict__ rw_sorted) {
    __shared__ int warp_n[kRouteThreads / 32];
    const int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = offsets[e];
    const int end = offsets[e + 1];
    for (long p0 = 0; p0 < P && base < end; p0 += kRouteThreads) {
        const long p = p0 + threadIdx.x;
        const bool mine = p < P && ids[p] == e;
        const unsigned ballot = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) warp_n[warp] = __popc(ballot);
        __syncthreads();
        int before = 0, total = 0;
        #pragma unroll
        for (int w = 0; w < kRouteThreads / 32; ++w) {
            before += w < warp ? warp_n[w] : 0;
            total += warp_n[w];
        }
        if (mine) {
            const int q = base + before + __popc(ballot & ((1u << lane) - 1u));
            flat_sorted[q] = (int32_t)p;
            rw_sorted[q] = weights[p];
        }
        base += total;
        __syncthreads();
    }
}

void announce(const char *what, int N, int K, int E, int extra_a, int extra_b) {
    fprintf(stderr, "pulsar: L255 tessera %s = Tessera fused window kernel (value family, folded bf16), "
                    "N %d K %d E %d (%d, %d)\n", what, N, K, E, extra_a, extra_b);
}

} // namespace

extern "C" int pulsar_tessera_dense_k_split(int M, int N, int K, int sms, int tile_words) {
    /* upstream ``routed_fused.dense_k_split``, operation for operation (the t(S) comparison in double) */
    const long items0 = (long)((M + BM - 1) / BM) * (N / BN);
    const int nk = K / BK;
    if (items0 >= sms || M <= 0) return 1;
    const long wire = (long)N * (long)tile_words * 4 / 512;
    const long cap = (sms + items0 - 1) / items0;
    int best_s = 1;
    double best_t = 0.0;
    for (int s = 1; s <= (nk < cap ? nk : (int)cap); ++s) {
        const long used = (long)s * items0 < sms ? (long)s * items0 : sms;
        const double t = (double)wire * sms / (double)used + 2.0 * s * M * (double)N * 4;
        if (s == 1 || t < best_t) { best_s = s; best_t = t; }
    }
    return best_s;
}

extern "C" size_t pulsar_tessera_dense_workspace_bytes(const pulsar_tessera_proj *w, int M) {
    if (!w || M <= 0) return 0;
    const int s = pulsar_tessera_dense_k_split(M, w->N, w->K, sm_count(), w->tile_words);
    return 256 + (s > 1 ? (size_t)s * M * w->N * 4 : 0);
}

extern "C" int pulsar_tessera_dense_launch(const pulsar_tessera_proj *w, const uint16_t *x, int M, uint16_t *out,
                                           long out_stride, void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!proj_ok(w, 2, "dense")) return -1;
    if (w->E != 1 || !x || !out || M <= 0 || out_stride < w->N || out_stride % 2 != 0) {
        fprintf(stderr, "pulsar: tessera: dense launch: E %d (must be 1), M %d, out_stride %ld (even, >= N %d) -- "
                        "refusing\n", w->E, M, out_stride, w->N);
        return -1;
    }
    const int sms = sm_count();
    const int S = pulsar_tessera_dense_k_split(M, w->N, w->K, sms, w->tile_words);
    const size_t need = pulsar_tessera_dense_workspace_bytes(w, M);
    if (!ws || ws_bytes < need || !aligned16(ws) || (S > 1 && out_stride % 4 != 0)) {
        fprintf(stderr, "pulsar: tessera: dense launch: workspace %zu B < %zu B, or split %d with out_stride %ld "
                        "(needs a multiple of 4) -- refusing\n", ws_bytes, need, S, out_stride);
        return -1;
    }
    static int announced = 0;
    if (!announced) { announced = 1; announce("dense", w->N, w->K, 1, M, S); }
    int32_t *counter = (int32_t *)ws;
    float *partial = S > 1 ? (float *)((uint8_t *)ws + 256) : nullptr;

    /* upstream ``dense_forward`` (routed_fused_window.cu), the value family's fields */
    Params p{};
    p.x = x;
    p.a_scale = nullptr;
    p.words0 = w->words;
    p.table0 = w->table;
    p.init0 = w->init;
    p.has_init0 = w->has_init;
    p.wscale0 = w->wscale;
    p.runs0 = w->runs;
    p.bdesc0 = w->bdesc;
    p.words_stride = w->words_stride;
    p.tile_words = w->tile_words;
    p.slot_words = w->slot_words;
    p.K = w->K;
    p.N = w->N;
    p.E = 1;
    p.counter = counter;
    p.n_blocks = w->N / BN;
    p.top_k = 1;
    p.a_row_mode = 2;
    p.mul_weight = 0;
    p.limit = __builtin_huge_valf();
    p.out = out;
    p.out_stride = out_stride;
    p.inter = w->N;
    p.rows_x = M;
    p.k_split = S;
    p.partial = partial;

    g_host_failed = false;
    if (cudaMemsetAsync(counter, 0, sizeof(int32_t), stream) != cudaSuccess) g_host_failed = true;
    if (S > 1) {
        launch<false, 2, true, true>(p, sms, stream);
        const long quads = (long)M * (w->N / 4);
        const int threads = 256;
        dense_reduce_kernel<false><<<(unsigned)((quads + threads - 1) / threads), threads, 0, stream>>>(
            partial, nullptr, w->wscale, out, out_stride, S, M, w->N);
        tessera_host_cuda(cudaGetLastError(), "dense_reduce_kernel");
    } else {
        launch<false, 2, true, false>(p, sms, stream);
    }
    return g_host_failed ? -1 : 0;
}

extern "C" size_t pulsar_tessera_moe_workspace_bytes(const pulsar_tessera_proj *gate,
                                                     const pulsar_tessera_proj *down, int T, int top_k) {
    if (!gate || !down || T <= 0 || top_k <= 0) return 0;
    return moe_ws_layout(nullptr, gate->E, (long)T * top_k, gate->N, down->N, nullptr);
}

extern "C" int pulsar_tessera_moe_launch(const pulsar_tessera_proj *gate, const pulsar_tessera_proj *up,
                                         const pulsar_tessera_proj *down, const uint16_t *x, int T, int top_k,
                                         const int32_t *expert_ids, const float *weights, uint16_t *out,
                                         void *ws, size_t ws_bytes, cudaStream_t stream) {
    if (!proj_ok(gate, 0, "moe gate") || !proj_ok(up, 0, "moe up") || !proj_ok(down, 2, "moe down")) return -1;
    const int E = gate->E, H = gate->K, I = gate->N;
    if (up->E != E || up->K != H || up->N != I || up->words_stride != gate->words_stride ||
        up->tile_words != gate->tile_words || up->slot_words != gate->slot_words || down->E != E ||
        down->K != I || down->N != H || E > kScanThreads || !x || !out || !expert_ids || !weights || T <= 0 ||
        top_k <= 0) {
        fprintf(stderr, "pulsar: tessera: moe launch: gate/up (E %d N %d K %d / E %d N %d K %d) and down (E %d N %d "
                        "K %d) do not form one stack of at most %d experts, or T %d / top_k %d / a null pointer -- "
                        "refusing\n", gate->E, gate->N, gate->K, up->E, up->N, up->K, down->E, down->N, down->K,
                kScanThreads, T, top_k);
        return -1;
    }
    const long P = (long)T * top_k;
    const size_t need = moe_ws_layout(nullptr, E, P, I, H, nullptr);
    if (!ws || ws_bytes < need) {
        fprintf(stderr, "pulsar: tessera: moe launch: workspace %zu B < %zu B -- refusing\n", ws_bytes, need);
        return -1;
    }
    static int announced = 0;
    if (!announced) { announced = 1; announce("moe", H, I, E, T, top_k); }
    MoeWs m;
    moe_ws_layout((uint8_t *)ws, E, P, I, H, &m);
    const int sms = sm_count();
    g_host_failed = false;

    /* the routing prep (upstream ``FusedRoutedWindowMoE._routing``) */
    if (cudaMemsetAsync(m.counts, 0, (size_t)E * 4, stream) != cudaSuccess ||
        cudaMemsetAsync(m.counters, 0, 2 * 4, stream) != cudaSuccess)
        g_host_failed = true;
    route_count_kernel<<<(unsigned)((P + kRouteThreads - 1) / kRouteThreads), kRouteThreads, 0, stream>>>(
        expert_ids, P, E, m.counts);
    route_scan_kernel<<<1, kScanThreads, 0, stream>>>(m.counts, E, m.offsets, m.item_off);
    route_place_kernel<<<E, kRouteThreads, 0, stream>>>(expert_ids, weights, P, m.offsets, m.flat_sorted,
                                                        m.rw_sorted);
    tessera_host_cuda(cudaGetLastError(), "routing prep");

    /* gate/up + SwiGLU into the route-sorted activation (upstream ``__call__``: mode 0, a_row_mode 0, no
     * weight, limit inf, counter slot 0) */
    Params p{};
    p.x = x;
    p.words0 = gate->words;
    p.words1 = up->words;
    p.table0 = gate->table;
    p.table1 = up->table;
    p.init0 = gate->init;
    p.init1 = up->init;
    p.has_init0 = gate->has_init;
    p.has_init1 = up->has_init;
    p.wscale0 = gate->wscale;
    p.wscale1 = up->wscale;
    p.runs0 = gate->runs;
    p.runs1 = up->runs;
    p.bdesc0 = gate->bdesc;
    p.bdesc1 = up->bdesc;
    p.words_stride = gate->words_stride;
    p.tile_words = gate->tile_words;
    p.slot_words = gate->slot_words;
    p.K = H;
    p.N = I;
    p.E = E;
    p.offsets = m.offsets;
    p.flat_sorted = m.flat_sorted;
    p.rw_sorted = m.rw_sorted;
    p.item_off = m.item_off;
    p.counter = &m.counters[0];
    p.n_blocks = I / HALF;
    p.top_k = top_k;
    p.a_row_mode = 0;
    p.mul_weight = 0;
    p.limit = __builtin_huge_valf();
    p.out = m.act;
    p.out_stride = I;
    p.inter = I;
    p.rows_x = T;
    p.k_split = 1;
    p.partial = nullptr;
    if (!g_host_failed) launch<false, 0>(p, sms, stream);

    /* down, weighted, into route-ordered rows (mode 2, a_row_mode 1, mul_weight, counter slot 1) */
    Params d = p;
    d.x = m.act;
    d.words0 = down->words;
    d.words1 = nullptr;
    d.table0 = down->table;
    d.table1 = nullptr;
    d.init0 = down->init;
    d.init1 = nullptr;
    d.has_init0 = down->has_init;
    d.has_init1 = nullptr;
    d.wscale0 = down->wscale;
    d.wscale1 = nullptr;
    d.runs0 = down->runs;
    d.runs1 = nullptr;
    d.bdesc0 = down->bdesc;
    d.bdesc1 = nullptr;
    d.words_stride = down->words_stride;
    d.tile_words = down->tile_words;
    d.slot_words = down->slot_words;
    d.K = I;
    d.N = H;
    d.counter = &m.counters[1];
    d.n_blocks = H / BN;
    d.a_row_mode = 1;
    d.mul_weight = 1;
    d.out = m.routed;
    d.out_stride = H;
    d.inter = H;
    d.rows_x = (int)P;
    if (!g_host_failed) launch<false, 2>(d, sms, stream);

    /* out[t] = bf16(sum_j routed[t * top_k + j]) in fixed j order (upstream ``token_sum``) */
    const long vecs = (long)T * (H / 8);
    token_sum_kernel<<<(unsigned)((vecs + 255) / 256), 256, 0, stream>>>(m.routed, out, T, top_k, H);
    tessera_host_cuda(cudaGetLastError(), "token_sum_kernel");
    return g_host_failed ? -1 : 0;
}

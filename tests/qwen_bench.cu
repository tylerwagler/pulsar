/* Qwen S4 microbenchmark (L251): the router, the MoE block, the Gated Residual
 * read / write and the PLE injection at the decode widths T = 1, 4, 8, 16 (plus
 * a 128-row prefill chunk), DRAM-cold, with the recipe's formats (routed experts
 * EXL3 K=4, shared expert and PLE projections EXL3 K=5, GR low-rank MXFP8).
 *
 * DRAM-cold: every timed call reads weights no earlier call left in L2 (24 MB on
 * GB10) -- the per-layer weights (router, shared expert, GR, PLE projections)
 * cycle through replicas totalling >= 96 MB, and the routed experts are a full
 * 512-expert pool at the real size (1.26 GB at K=4) routed by random weights on
 * rotating inputs, so each call touches its own ~10 T experts.  Launches are
 * queued behind a spin kernel and timed with events (GPU time back to back).
 * "wire" = the bytes the call must read (weights; for the MoE the distinct
 * experts the call's routing touched) and "GB/s" = wire / time; the box streams
 * ~240-280 GB/s (expert_stream_probe).
 *
 *   tests/qwen_bench [iters=60]
 */
#include "../src/pulsar_gpu.h"
#include "qwen_ref.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using namespace qref;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0x2545F4914F6CDD1DULL;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndn(void) {
    const double u1 = ((double)rnd() + 0.5) / 4294967296.0, u2 = ((double)rnd() + 0.5) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
/* fill device bytes with a hash -- random trellis words / codes without a host round trip */
__global__ void fill_kernel(uint32_t *p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
        p[i] = x;
    }
}
/* E4M3 codes that are never NaN (0x7f / 0xff) */
__global__ void sanitize_e4m3(uint8_t *p, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if ((p[i] & 0x7f) == 0x7f) p[i] ^= 0x01;
}

template <typename F>
static double time_calls(int iters, F fn) {
    const cudaStream_t st = cudaStreamPerThread;
    for (int i = 0; i < 3; i++) fn(i);
    CK(cudaStreamSynchronize(st));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    spin_kernel<<<1, 1, 0, st>>>(150000000LL);
    CK(cudaEventRecord(t0, st));
    const auto h0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn(i);
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    CK(cudaEventRecord(t1, st));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (host_ms > 55.0) fprintf(stderr, "  (warning: enqueue took %.1f ms, may exceed the spin -- host-bound timing)\n", host_ms);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms * 1000.0 / iters;
}

static void *dfill(size_t bytes, uint32_t seed) {
    void *p = nullptr;
    CK(cudaMalloc(&p, bytes + 64));
    fill_kernel<<<1024, 256>>>((uint32_t *)p, (bytes + 3) / 4, seed);
    return p;
}
template <typename T> static T *up(const std::vector<T> &v) {
    T *d = nullptr;
    CK(cudaMalloc((void **)&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static void *dz(size_t n) { void *p = nullptr; CK(cudaMalloc(&p, n + 16)); CK(cudaMemset(p, 0, n + 16)); return p; }
static std::vector<uint16_t> rnd_bf(size_t n, double s) { std::vector<uint16_t> v(n); for (auto &b : v) b = to_bf(rndn() * s); return v; }

/* EXL3 slices with random words and realistic fp16 scales, on the device */
static uint8_t *exl3_slices(int in, int out, int k2, int count, size_t *stride_out) {
    uint64_t trellis = 0, scales = 0, stride = 0;
    exl3_expert_layout(in, out, k2, &trellis, &scales, &stride);
    uint8_t *d = (uint8_t *)dfill(stride * count, 0x1234u + k2 * 7 + in);
    std::vector<uint16_t> s(in + out);
    for (int i = 0; i < in; i++) s[i] = exl3_f32_to_f16((float)(0.025 * (0.5 + (rnd() % 1000) / 1000.0) * ((rnd() & 1) ? 1 : -1)));
    for (int i = 0; i < out; i++) s[in + i] = exl3_f32_to_f16((float)(0.02 * ((rnd() & 1) ? 1 : -1)));
    for (int c = 0; c < count; c++) CK(cudaMemcpy(d + (size_t)c * stride + trellis, s.data(), s.size() * 2, cudaMemcpyHostToDevice));
    *stride_out = stride;
    return d;
}
static pulsar_qwen_slot slot_for(const std::vector<uint16_t> &x, int T, int n) {
    const int kbp = pulsar_mx_kbp(n);
    std::vector<uint8_t> q((size_t)T * n), sf(pulsar_mx_sf_slab_bytes(T, kbp), 0);
    std::vector<double> v(n);
    for (int r = 0; r < T; r++) {
        for (int k = 0; k < n; k++) v[k] = bf(x[(size_t)r * n + k]);
        mx_encode_row(v.data(), n, r, kbp, q.data() + (size_t)r * n, sf.data(), nullptr);
    }
    return {up(q), up(sf), kbp};
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 60;
    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }
    const int widths[] = {1, 4, 8, 16, 128};
    const int NX = 48;                               /* rotating input sets */
    printf("qwen-bench: DRAM-cold, us per call (GB/s of wire bytes)\n");

    /* ---- inputs: NX sets of 128 rows */
    std::vector<std::vector<uint16_t>> xs(NX);
    std::vector<pulsar_qwen_slot> xslot(NX);
    std::vector<uint16_t *> xdev(NX);
    for (int i = 0; i < NX; i++) {
        xs[i] = rnd_bf((size_t)128 * H, 0.7);
        xdev[i] = up(xs[i]);
        xslot[i] = slot_for(xs[i], 128, H);
    }

    /* ---- router + MoE */
    const int RR = 40;                               /* router / shared replicas: 40 x ~5.8 MB */
    std::vector<uint16_t *> wr(RR), wsg(RR);
    for (int i = 0; i < RR; i++) { wr[i] = up(rnd_bf((size_t)E * H, 0.02)); wsg[i] = up(rnd_bf(H, 0.02)); }
    size_t sg_stride = 0, sd_stride = 0, eg_stride = 0, ed_stride = 0;
    uint8_t *shg = exl3_slices(H, MID, 10, RR, &sg_stride), *shu = exl3_slices(H, MID, 10, RR, &sg_stride);
    uint8_t *shd = exl3_slices(MID, H, 10, RR, &sd_stride);
    uint8_t *eg = exl3_slices(H, 2 * MID, 8, E, &eg_stride);
    uint8_t *ed = exl3_slices(MID, H, 8, E, &ed_stride);
    uint64_t tg = 0, td = 0, sc = 0, st = 0;
    exl3_expert_layout(H, 2 * MID, 8, &tg, &sc, &st);
    exl3_expert_layout(MID, H, 8, &td, &sc, &st);
    std::vector<const void *> tgv(2 * E), tdv(2 * E);
    for (int e = 0; e < E; e++) {
        tgv[2 * e] = eg + e * eg_stride; tgv[2 * e + 1] = eg + e * eg_stride + tg;
        tdv[2 * e] = ed + e * ed_stride; tdv[2 * e + 1] = ed + e * ed_stride + td;
    }
    const void *const *dtg = (const void *const *)up(tgv), *const *dtd = (const void *const *)up(tdv);
    std::vector<pulsar_qwen_moe_weights> mw(RR);
    for (int i = 0; i < RR; i++) {
        mw[i].router_w = wr[i]; mw[i].shared_gate_w = wsg[i];
        mw[i].gate_up_table = dtg; mw[i].down_table = dtd;
        mw[i].k2_gate_up = 8; mw[i].k2_down = 8;
        mw[i].shared_gate = {shg + i * sg_stride, 10, H, MID};
        mw[i].shared_up = {shu + i * sg_stride, 10, H, MID};
        mw[i].shared_down = {shd + i * sd_stride, 10, MID, H};
    }
    const size_t mws = pulsar_qwen_moe_workspace_bytes(&mw[0], 128);
    void *mwsp = dz(mws);
    float *out = (float *)dz((size_t)128 * H * 4), *lg = (float *)dz((size_t)128 * (E + 1) * 4);
    float *rw = (float *)dz(128 * TOPK * 4), *rs = (float *)dz(128 * 4);
    int32_t *sel = (int32_t *)dz(128 * TOPK * 4);
    uint32_t *nf = (uint32_t *)dz(4);
    const double router_bytes = (double)(E + 1) * H * 2;
    const double shared_bytes = 2.0 * sg_stride + sd_stride;
    printf("\nrouter (bf16 %d x %d = %.2f MB) | MoE block (experts K=4: gate_up %.2f MB + down %.2f MB, shared K=5 %.2f MB, router)\n",
           E, H, router_bytes / 1e6, eg_stride / 1e6, ed_stride / 1e6, shared_bytes / 1e6);
    printf("%5s | %-18s | %-18s | %-12s | %-22s | %s\n", "T", "router", "shared (3 dense)", "experts hit", "MoE block", "routed alone (block - router - shared)");
    for (int T : widths) {
        const double t_router = time_calls(iters, [&](int i) {
            pulsar_qwen_router_launch(xdev[i % NX], wr[i % RR], wsg[i % RR], T, H, E, TOPK, lg, sel, rw, rs, 0);
        });
        const size_t lws = pulsar_qwen_linear_workspace_bytes(&mw[0].shared_gate, T);
        void *lw = dz(lws);
        float *y1 = (float *)dz((size_t)T * MID * 4), *y2 = (float *)dz((size_t)T * H * 4);
        pulsar_qwen_slot hq = slot_for(std::vector<uint16_t>((size_t)T * MID, 0x3f00), T, MID);
        const double t_shared = time_calls(iters, [&](int i) {
            const pulsar_qwen_moe_weights &m = mw[i % RR];
            pulsar_qwen_linear_launch(&m.shared_gate, &xslot[i % NX], T, y1, lw, lws, 0);
            pulsar_qwen_linear_launch(&m.shared_up, &xslot[i % NX], T, y1, lw, lws, 0);
            pulsar_qwen_linear_launch(&m.shared_down, &hq, T, y2, lw, lws, 0);
        });
        /* distinct experts per call, averaged over the rotation */
        double hit = 0;
        for (int i = 0; i < NX; i++) {
            pulsar_qwen_router_launch(xdev[i], wr[i % RR], wsg[i % RR], T, H, E, TOPK, lg, sel, rw, rs, 0);
            std::vector<int32_t> s((size_t)T * TOPK);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(s.data(), sel, s.size() * 4, cudaMemcpyDeviceToHost));
            hit += (double)std::set<int32_t>(s.begin(), s.end()).size() / NX;
        }
        const double t_moe = time_calls(iters, [&](int i) {
            pulsar_qwen_moe_launch(&mw[i % RR], xdev[i % NX], &xslot[i % NX], T, out, mwsp, mws, nf, 1u, 0);
        });
        const double moe_bytes = router_bytes + shared_bytes + hit * ((double)eg_stride + ed_stride);
        const double routed_bytes = hit * ((double)eg_stride + ed_stride);
        const double t_routed = t_moe - t_router - t_shared;
        printf("%5d | %7.1f (%5.0f)    | %7.1f (%5.0f)    | %6.1f       | %8.1f (%5.0f GB/s)  | %7.1f us (%5.0f GB/s over %.1f MB)\n",
               T, t_router, router_bytes / t_router / 1e3, t_shared, shared_bytes / t_shared / 1e3, hit, t_moe,
               moe_bytes / t_moe / 1e3, t_routed, routed_bytes / t_routed / 1e3, routed_bytes / 1e6);
        cudaFree(lw); cudaFree(y1); cudaFree(y2); cudaFree(hq.q); cudaFree(hq.sf);
    }

    /* ---- Gated Residual */
    const int RG = 16;
    std::vector<pulsar_qwen_gr_weights> gw(RG);
    const size_t dq = (size_t)R * HC, dsf = pulsar_mx_sf_slab_bytes(R, pulsar_mx_kbp(HC));
    const size_t uq = (size_t)HC * R, usf = pulsar_mx_sf_slab_bytes(HC, pulsar_mx_kbp(R));
    for (int i = 0; i < RG; i++) {
        uint8_t *a = (uint8_t *)dfill(dq, 11 + i), *b = (uint8_t *)dfill(uq, 211 + i);
        sanitize_e4m3<<<256, 256>>>(a, dq);
        sanitize_e4m3<<<256, 256>>>(b, uq);
        std::vector<uint8_t> s1(dsf), s2(usf);
        for (auto &v : s1) v = (uint8_t)(118 + rnd() % 4);
        for (auto &v : s2) v = (uint8_t)(120 + rnd() % 4);
        gw[i].norm_w = up(rnd_bf(HC, 0.1));
        gw[i].down = {a, up(s1), R, HC};
        gw[i].up = {b, up(s2), HC, R};
        gw[i].inject = up(rnd_bf((size_t)S * HC, 0.01));
    }
    const double gr_bytes = (double)dq + dsf + uq + usf + (double)S * HC * 2 + HC * 2;
    std::vector<uint16_t *> streams(NX);
    for (int i = 0; i < NX; i++) streams[i] = up(rnd_bf((size_t)128 * HC, 0.8));
    uint16_t *xo = (uint16_t *)dz((size_t)128 * H * 2);
    pulsar_qwen_slot xq = {(uint8_t *)dz((size_t)128 * H), (uint8_t *)dz(pulsar_mx_sf_slab_bytes(128, pulsar_mx_kbp(H))), pulsar_mx_kbp(H)};
    float *inj = (float *)dz(128 * S * 4);
    const size_t gws = pulsar_qwen_gr_workspace_bytes(128);
    void *gwsp = dz(gws);
    printf("\nGated Residual (W_down + W_up MXFP8 + inject bf16 = %.2f MB per site)\n", gr_bytes / 1e6);
    printf("%5s | %-20s | %-20s | %s\n", "T", "read (site)", "read (mixer)", "write");
    for (int T : widths) {
        const double t_read = time_calls(iters, [&](int i) {
            pulsar_qwen_gr_read_launch(&gw[i % RG], streams[i % NX], T, xo, &xq, inj, gwsp, gws, 0);
        });
        const double t_mix = time_calls(iters, [&](int i) {
            pulsar_qwen_gr_weights m = gw[i % RG];
            m.inject = nullptr;
            pulsar_qwen_gr_read_launch(&m, streams[i % NX], T, xo, &xq, nullptr, gwsp, gws, 0);
        });
        const double t_write = time_calls(iters, [&](int i) {
            pulsar_qwen_gr_write_launch(streams[i % NX], out, inj, T, 0);
        });
        printf("%5d | %7.1f (%5.0f GB/s) | %7.1f (%5.0f GB/s) | %6.1f\n", T, t_read, gr_bytes / t_read / 1e3, t_mix,
               (gr_bytes - S * HC * 2.0) / t_mix / 1e3, t_write);
    }

    /* ---- PLE */
    const int RP = 6;
    size_t k_stride = 0, v_stride = 0;
    uint8_t *kp = exl3_slices(H, HC, 10, RP, &k_stride), *vp = exl3_slices(H, H, 10, RP, &v_stride);
    std::vector<pulsar_qwen_ple_weights> pw(RP);
    for (int i = 0; i < RP; i++) {
        pw[i].key_proj = {kp + i * k_stride, 10, H, HC};
        pw[i].value_proj = {vp + i * v_stride, 10, H, H};
        pw[i].norm_key = up(rnd_bf(HC, 0.1)); pw[i].norm_query = up(rnd_bf(HC, 0.1));
        pw[i].norm_conv = up(rnd_bf(HC, 0.1)); pw[i].conv_w = up(rnd_bf((size_t)HC * 4, 0.3));
    }
    const double ple_bytes = (double)k_stride + v_stride + 4.0 * HC * 2 + HC * 8.0;
    std::vector<uint16_t *> embs(NX);
    for (int i = 0; i < NX; i++) embs[i] = up(rnd_bf((size_t)128 * H, 0.05));
    float *state = (float *)dz((size_t)128 * PULSAR_QWEN_PLE_STATE * HC * 4);
    printf("\nPLE injection (key K=5 %.2f MB + value K=5 %.2f MB; the row gather is host I/O, not here)\n",
           k_stride / 1e6, v_stride / 1e6);
    for (int T : widths) {
        /* decode: T sequences of one row each; the 128-row chunk: one sequence */
        const bool chunk = T > 16;
        const int n_seq = chunk ? 1 : T;
        std::vector<int32_t> rs(T), rj(T), sf(n_seq), sr(n_seq);
        for (int r = 0; r < T; r++) { rs[r] = chunk ? 0 : r; rj[r] = chunk ? r : 0; }
        for (int q = 0; q < n_seq; q++) { sf[q] = chunk ? 0 : q; sr[q] = chunk ? T : 1; }
        pulsar_qwen_rows pr = {up(rs), up(rj), up(sf), up(sr), n_seq};
        const size_t pws = pulsar_qwen_ple_workspace_bytes(&pw[0], T);
        void *pwsp = dz(pws);
        const double t_ple = time_calls(iters, [&](int i) {
            pulsar_qwen_ple_launch(&pw[i % RP], embs[i % NX], streams[i % NX], T, &pr, state, pwsp, pws, 0);
        });
        printf("%5d | %7.1f us (%5.0f GB/s)%s\n", T, t_ple, ple_bytes / t_ple / 1e3, chunk ? "  [one sequence, a prefill chunk]" : "");
        cudaFree(pwsp);
    }
    CK(cudaDeviceSynchronize());
    return 0;
}

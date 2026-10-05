/* Gated DeltaNet gate (L251) -- model-free, links the PRODUCTION object
 * (src/cuda/pulsar_cuda_gdn.o, the engine's NVCCFLAGS incl. fast-math).
 *
 *   1. fidelity   a 257-token prefill from random non-zero states vs the double
 *                 host authority (tests/gdn_ref.h): output, recurrent state,
 *                 conv state.
 *   2. decode     40 one-token calls == one 40-token prefill, BIT-EXACT
 *                 (output, both states).
 *   3. chunking   the 257 tokens split at 1, 3, 19, 36, 41, 141 == one call,
 *                 bit-exact.
 *   4. batching   five 13-row sequences on scattered slots in one call ==
 *                 each alone, and 16 decode rows batched == each alone,
 *                 bit-exact.
 *   5. A8 slot    the producer's E4M3 bytes + ue8m0 scales == the canonical
 *                 warp encoder (pulsar_mx_emit_block) run on its own f32 output.
 *   6. mutations  each must grade >= 100x the Frobenius bar: one A_log sign, one
 *                 recurrent-state element, one channel's conv taps reversed;
 *                 and the reference with the GQA map tiled or the q scale
 *                 dropped must FAIL against the kernel.
 *   7. refusals   contract violations return -1 and launch nothing.
 *
 * The fixture is live: heavy-tailed inputs with outlier channels, decays from
 * ~0.5 to ~0.9999, random initial states; the log prints rms and median error
 * so a degenerate (all-zero, all-saturated) fixture is visible. */
#include "cuda/pulsar_cuda_gdn.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "gdn_ref.h"

#include <algorithm>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static const int QKV = PULSAR_GDN_QKV_DIM, VD = PULSAR_GDN_V_DIM, NV = PULSAR_GDN_NV;
static const size_t CSF = PULSAR_GDN_CONV_STATE_FLOATS, RSF = PULSAR_GDN_REC_STATE_FLOATS;
static const double BAR = 1e-6;       /* ||err|| / ||ref|| (relative Frobenius)            */
static const double BAR_MAX = 1e-5;   /* max |err| / max |ref|, the worst single element  */
static int g_fail = 0;

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static double urand() {                /* splitmix64 -> [0,1) */
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (double)((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
}
static double nrand() { double u = urand() + 1e-12, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

template <class T> static T *dalloc(size_t n) { T *p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
template <class T> static void h2d(T *d, const T *h, size_t n) { CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> static void d2h(T *h, const T *d, size_t n) { CK(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost)); }

static void check(bool ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "ok" : "FAIL"); vprintf(fmt, ap); printf("\n");
    va_end(ap);
    if (!ok) g_fail = 1;
}

/* max |got - ref| / rms(ref), and the median |err| / rms. */
/* fro = ||got - ref|| / ||ref||; max = max |got - ref| / max |ref|; med = the
 * median |got - ref| / rms(ref) (a live fixture has a nonzero one); rms of ref. */
struct Err { double fro, max, med, rms; };
static Err grade(const float *got, const double *ref, size_t n) {
    double ss = 0, se = 0, am = 0, em = 0;
    for (size_t i = 0; i < n; i++) { ss += ref[i] * ref[i]; am = std::max(am, fabs(ref[i])); }
    const double rms = sqrt(ss / (double)n);
    std::vector<double> e(n);
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)got[i] - ref[i]);
        se += d * d; em = std::max(em, d); e[i] = d / rms;
    }
    std::nth_element(e.begin(), e.begin() + n / 2, e.end());
    return {sqrt(se / ss), em / am, e[n / 2], rms};
}

/* The canonical encoder: one warp per aligned 32-column block. */
__global__ void ref_emit_kernel(const float *x, int rows, int kbp, __nv_fp8_e4m3 *data, unsigned char *scale) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * 32 + threadIdx.x;
    if (row < rows) pulsar_mx_emit_block(x[(size_t)row * VD + col], (uint32_t)col, (uint32_t)row, VD, kbp, data, scale);
}

/* bf16 round (RNE) in place, keeping a uint16 copy: the container stores these
 * four tensors bf16 and the kernel widens them, so both the kernel input and
 * the double reference must read the rounded values. */
static void bf16_round(std::vector<float> &v, std::vector<uint16_t> &b) {
    b.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t u; memcpy(&u, &v[i], 4); u = (u + 0x8000u) & 0xFFFF0000u;
        memcpy(&v[i], &u, 4); b[i] = (uint16_t)(u >> 16);
    }
}

struct Rig {
    static const int ROWS = 512, SLOTS = 20;
    std::vector<float> conv_w, A_log, dt_bias, norm_w;      /* the REFERENCE values (bf16-rounded) */
    std::vector<uint16_t> conv_w_b, A_log_b, dt_bias_b, norm_w_b;
    std::vector<float> qkv, z, a, b;
    uint16_t *d_conv_w, *d_A_log, *d_dt_bias, *d_norm_w;
    float *d_qkv, *d_z, *d_a, *d_b;
    float *d_cs, *d_rs;                       /* state pools */
    int32_t *d_slot;
    void *d_scratch; size_t scratch_bytes;
    float *d_out; __nv_fp8_e4m3 *d_q; unsigned char *d_s; int kbp;
    size_t slab;

    void init() {
        conv_w.resize((size_t)QKV * 4); A_log.resize(NV); dt_bias.resize(NV); norm_w.resize(128);
        for (auto &v : conv_w) v = (float)(urand() * 1.2 - 0.6);
        for (int h = 0; h < NV; h++) {
            A_log[h] = (float)log(0.02 + urand() * 2.0);
            dt_bias[h] = (float)(-5.0 + urand() * 5.0);
        }
        for (auto &v : norm_w) v = (float)(0.5 + urand());
        bf16_round(conv_w, conv_w_b); bf16_round(A_log, A_log_b);
        bf16_round(dt_bias, dt_bias_b); bf16_round(norm_w, norm_w_b);   /* sizes for the dalloc below */
        qkv.resize((size_t)ROWS * QKV); z.resize((size_t)ROWS * VD); a.resize((size_t)ROWS * NV); b.resize((size_t)ROWS * NV);
        std::vector<float> chs(QKV);
        for (int c = 0; c < QKV; c++) chs[c] = (float)((urand() < 0.02 ? 8.0 : 1.0) * (0.3 + urand()));
        for (size_t i = 0; i < qkv.size(); i++) qkv[i] = (float)(nrand() * chs[i % QKV]);
        for (auto &v : z) v = (float)(nrand() * 1.5);
        for (auto &v : a) v = (float)(nrand() * 1.5);
        for (auto &v : b) v = (float)(nrand() * 1.5);
        d_conv_w = dalloc<uint16_t>(conv_w_b.size()); d_A_log = dalloc<uint16_t>(NV); d_dt_bias = dalloc<uint16_t>(NV); d_norm_w = dalloc<uint16_t>(128);
        upload_weights();
        d_qkv = dalloc<float>(qkv.size()); d_z = dalloc<float>(z.size()); d_a = dalloc<float>(a.size()); d_b = dalloc<float>(b.size());
        h2d(d_qkv, qkv.data(), qkv.size()); h2d(d_z, z.data(), z.size()); h2d(d_a, a.data(), a.size()); h2d(d_b, b.data(), b.size());
        d_cs = dalloc<float>(CSF * SLOTS); d_rs = dalloc<float>(RSF * SLOTS);
        d_slot = dalloc<int32_t>(ROWS);
        scratch_bytes = pulsar_gdn_scratch_bytes(ROWS); d_scratch = dalloc<char>(scratch_bytes);
        d_out = dalloc<float>((size_t)ROWS * VD);
        kbp = pulsar_mx_kbp(VD);
        slab = pulsar_mx_sf_slab_bytes(ROWS, kbp);
        d_q = dalloc<__nv_fp8_e4m3>((size_t)ROWS * VD); d_s = dalloc<unsigned char>(slab);
    }
    /* Round in place and refresh the bf16 copy on EVERY upload, so a mutation
     * of the f32 vector reaches the kernel AND the reference sees the same
     * widened value.  (The container stores these four bf16; the kernel widens.) */
    void upload_weights() {
        bf16_round(conv_w, conv_w_b); bf16_round(A_log, A_log_b);
        bf16_round(dt_bias, dt_bias_b); bf16_round(norm_w, norm_w_b);
        h2d(d_conv_w, conv_w_b.data(), conv_w_b.size()); h2d(d_A_log, A_log_b.data(), (size_t)NV);
        h2d(d_dt_bias, dt_bias_b.data(), (size_t)NV); h2d(d_norm_w, norm_w_b.data(), (size_t)128);
    }
    pulsar_gdn_weights weights() const { return {d_conv_w, d_A_log, d_dt_bias, d_norm_w}; }
    gdn_ref_weights ref_weights() const { return {conv_w.data(), A_log.data(), dt_bias.data(), norm_w.data()}; }

    /* One call over input rows row0..: slots.size() sequences of seq_rows rows
     * each, sequence i on slot slots[i]; output rows land at the same row
     * indices of d_out (d_q from row 0). */
    int call(int row0, int seq_rows, const std::vector<int> &slots, bool emit = true) {
        std::vector<int32_t> rs;
        for (int s : slots) for (int r = 0; r < seq_rows; r++) rs.push_back(s);
        h2d(d_slot, rs.data(), rs.size());
        pulsar_gdn_call c{};
        c.n_seq = (int)slots.size(); c.seq_rows = seq_rows;
        c.row_slot = d_slot; c.conv_state = d_cs; c.rec_state = d_rs;
        c.qkv = d_qkv + (size_t)row0 * QKV; c.ld_qkv = QKV;
        c.z = d_z + (size_t)row0 * VD; c.ld_z = VD;
        c.a = d_a + (size_t)row0 * NV; c.ld_a = NV;
        c.b = d_b + (size_t)row0 * NV; c.ld_b = NV;
        c.scratch = d_scratch; c.scratch_bytes = scratch_bytes;
        c.out_f32 = d_out + (size_t)row0 * VD;
        if (emit) { c.out_e4m3 = d_q; c.out_scale = d_s; c.out_kbp = kbp; }
        const pulsar_gdn_weights w = weights();
        const int rc = pulsar_gdn_forward(&w, &c, 0);
        CK(cudaDeviceSynchronize());
        return rc;
    }
    void set_slot(int s, const std::vector<float> &cs, const std::vector<float> &rs) {
        h2d(d_cs + CSF * s, cs.data(), CSF); h2d(d_rs + RSF * s, rs.data(), RSF);
    }
    void get_slot(int s, std::vector<float> &cs, std::vector<float> &rs) {
        cs.resize(CSF); rs.resize(RSF);
        d2h(cs.data(), d_cs + CSF * s, CSF); d2h(rs.data(), d_rs + RSF * s, RSF);
    }
    std::vector<float> out(int row0, int n) {
        std::vector<float> o((size_t)n * VD); d2h(o.data(), d_out + (size_t)row0 * VD, o.size()); return o;
    }
};

static bool same(const std::vector<float> &x, const std::vector<float> &y) {
    return x.size() == y.size() && memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

int main() {
    Rig R; R.init();
    std::vector<float> cs0(CSF), rs0(RSF);
    for (auto &v : cs0) v = (float)nrand();
    for (auto &v : rs0) v = (float)(nrand() * 0.05);
    const gdn_ref_weights rw = R.ref_weights();
    double dmin = 1, dmax = 0;
    for (int h = 0; h < NV; h++) for (int t = 0; t < Rig::ROWS; t++) {
        const double sp_in = R.a[(size_t)t * NV + h] + R.dt_bias[h];
        const double d = exp(-exp((double)R.A_log[h]) * (sp_in > 20 ? sp_in : log1p(exp(sp_in))));
        dmin = std::min(dmin, d); dmax = std::max(dmax, d);
    }
    printf("gdn_gate: shapes qkv %d, v %d, heads %d/%d; fixture decays %.3g..%.6g\n", QKV, VD, PULSAR_GDN_NK, NV, dmin, dmax);

    /* ---- 1. fidelity vs the double reference -------------------------------- */
    const int T = 257;
    printf("1. fidelity: one %d-token prefill from random states vs the double reference (bars: fro %.0e, max %.0e)\n", T, BAR, BAR_MAX);
    std::vector<double> rconv(cs0.begin(), cs0.end()), rrec(rs0.begin(), rs0.end()), ry((size_t)T * VD);
    gdn_ref_seq(rw, T, R.qkv.data(), QKV, R.z.data(), VD, R.a.data(), NV, R.b.data(), NV, rconv.data(), rrec.data(), ry.data());
    R.set_slot(3, cs0, rs0);
    check(R.call(0, T, {3}) == 0, "prefill call returns 0");
    std::vector<float> y1 = R.out(0, T), cs1, rs1;
    R.get_slot(3, cs1, rs1);
    const Err ey = grade(y1.data(), ry.data(), y1.size());
    const Err es = grade(rs1.data(), rrec.data(), rs1.size());
    const Err ec = grade(cs1.data(), rconv.data(), cs1.size());
    check(ey.fro < BAR && ey.max < BAR_MAX && ey.med > 0 && ey.rms > 0.1,
          "output          fro %.2e  max %.2e  median %.2e  (rms of ref %.3g)", ey.fro, ey.max, ey.med, ey.rms);
    check(es.fro < BAR && es.max < BAR_MAX && es.med > 0 && es.rms > 1e-3,
          "recurrent state fro %.2e  max %.2e  median %.2e  (rms of ref %.3g)", es.fro, es.max, es.med, es.rms);
    check(ec.fro == 0.0, "conv state      fro %.2e (a copy of the last three inputs: exact)", ec.fro);

    /* ---- 5. the A8 slot (on the fidelity run's rows) ------------------------- */
    printf("5. A8 slot: producer E4M3 + ue8m0 == the canonical warp encoder on its own f32 output\n");
    {
        std::vector<unsigned char> q1((size_t)T * VD), s1(R.slab);
        d2h(q1.data(), reinterpret_cast<unsigned char *>(R.d_q), q1.size()); d2h(s1.data(), R.d_s, s1.size());
        __nv_fp8_e4m3 *rq = dalloc<__nv_fp8_e4m3>((size_t)T * VD); unsigned char *rsd = dalloc<unsigned char>(R.slab);
        CK(cudaMemset(rsd, 0, R.slab)); CK(cudaMemset(rq, 0, (size_t)T * VD));
        std::vector<unsigned char> s_pre(R.slab);
        ref_emit_kernel<<<dim3(T, VD / 32), 32>>>(R.d_out, T, R.kbp, rq, rsd);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned char> q2(q1.size()), s2(R.slab);
        d2h(q2.data(), reinterpret_cast<unsigned char *>(rq), q2.size()); d2h(s2.data(), rsd, s2.size());
        size_t dq = 0, ds = 0, nz = 0;
        for (size_t i = 0; i < q1.size(); i++) { dq += q1[i] != q2[i]; nz += (q1[i] & 0x7f) != 0; }
        for (size_t i = 0; i < s1.size(); i++) ds += s1[i] != s2[i];
        check(dq == 0 && ds == 0 && nz > q1.size() * 9 / 10,
              "data bytes differing %zu / %zu, scale bytes differing %zu / %zu, nonzero codes %.1f%%",
              dq, q1.size(), ds, s1.size(), 100.0 * (double)nz / (double)q1.size());
        CK(cudaFree(rq)); CK(cudaFree(rsd));
    }

    /* ---- 2. decode == prefill ------------------------------------------------ */
    printf("2. decode: 40 one-token calls == one 40-token prefill (bit-exact)\n");
    {
        const int N = 40;
        R.set_slot(0, cs0, rs0); R.call(0, N, {0});
        std::vector<float> yp = R.out(0, N), csp, rsp; R.get_slot(0, csp, rsp);
        R.set_slot(0, cs0, rs0);
        for (int t = 0; t < N; t++) R.call(t, 1, {0});
        std::vector<float> yd = R.out(0, N), csd, rsd; R.get_slot(0, csd, rsd);
        check(same(yp, yd) && same(csp, csd) && same(rsp, rsd), "output %s, conv state %s, recurrent state %s",
              same(yp, yd) ? "identical" : "DIFFER", same(csp, csd) ? "identical" : "DIFFER", same(rsp, rsd) ? "identical" : "DIFFER");
    }

    /* ---- 3. chunk neutrality ------------------------------------------------- */
    printf("3. chunking: %d tokens split at 1, 3, 19, 36, 41, 141 == one call (bit-exact)\n", T);
    {
        const int cuts[] = {0, 1, 3, 19, 36, 41, 141, T};
        R.set_slot(5, cs0, rs0);
        for (int i = 0; i + 1 < (int)(sizeof cuts / sizeof cuts[0]); i++) R.call(cuts[i], cuts[i + 1] - cuts[i], {5});
        std::vector<float> yc = R.out(0, T), csc, rsc; R.get_slot(5, csc, rsc);
        check(same(y1, yc) && same(cs1, csc) && same(rs1, rsc), "output %s, conv state %s, recurrent state %s",
              same(y1, yc) ? "identical" : "DIFFER", same(cs1, csc) ? "identical" : "DIFFER", same(rs1, rsc) ? "identical" : "DIFFER");
    }

    /* ---- 4. batching --------------------------------------------------------- */
    printf("4. batching: five 13-row sequences and 16 decode rows == each sequence alone (bit-exact)\n");
    {
        const int L = 13;
        const std::vector<int> lens(5, L), slots = {4, 0, 12, 7, 19};
        std::vector<std::vector<float>> cs_i(lens.size()), rs_i(lens.size());
        for (size_t i = 0; i < lens.size(); i++) {         /* distinct starting states */
            cs_i[i] = cs0; rs_i[i] = rs0;
            for (auto &v : cs_i[i]) v *= (float)(0.5 + 0.25 * (double)i);
            for (auto &v : rs_i[i]) v *= (float)(1.0 + 0.5 * (double)i);
        }
        const int base = 300;
        for (size_t i = 0; i < lens.size(); i++) R.set_slot(slots[i], cs_i[i], rs_i[i]);
        R.call(base, L, slots);
        int rows = 0; for (int l : lens) rows += l;
        std::vector<float> yb = R.out(base, rows);
        std::vector<std::vector<float>> csb(lens.size()), rsb(lens.size());
        for (size_t i = 0; i < lens.size(); i++) R.get_slot(slots[i], csb[i], rsb[i]);
        bool ok = true; int r = base;
        for (size_t i = 0; i < lens.size(); i++) {
            R.set_slot(1, cs_i[i], rs_i[i]);
            R.call(r, L, {1});
            std::vector<float> ya = R.out(r, lens[i]), csa, rsa; R.get_slot(1, csa, rsa);
            ok = ok && memcmp(ya.data(), yb.data() + (size_t)(r - base) * VD, ya.size() * sizeof(float)) == 0
                    && same(csa, csb[i]) && same(rsa, rsb[i]);
            r += lens[i];
        }
        check(ok, "5 x 13 rows on slots {4,0,12,7,19}: every sequence identical to its solo run");

        const int M = 16;
        std::vector<int> sl(M);
        for (int i = 0; i < M; i++) { sl[i] = i; R.set_slot(i, cs_i[i % lens.size()], rs_i[(i + 2) % lens.size()]); }
        R.call(100, 1, sl);
        std::vector<float> yM = R.out(100, M);
        std::vector<std::vector<float>> csM(M), rsM(M);
        for (int i = 0; i < M; i++) R.get_slot(i, csM[i], rsM[i]);
        bool ok2 = true;
        for (int i = 0; i < M; i++) {
            R.set_slot(19, cs_i[i % lens.size()], rs_i[(i + 2) % lens.size()]);
            R.call(100 + i, 1, {19});
            std::vector<float> ya = R.out(100 + i, 1), csa, rsa; R.get_slot(19, csa, rsa);
            ok2 = ok2 && memcmp(ya.data(), yM.data() + (size_t)i * VD, VD * sizeof(float)) == 0 && same(csa, csM[i]) && same(rsa, rsM[i]);
        }
        check(ok2, "16 decode rows batched == each row alone");
    }

    /* ---- 6. mutations -------------------------------------------------------- */
    printf("6. mutations (each must grade fro >= 100x the bar)\n");
    {
        /* (a) one A_log sign */
        const float keep = R.A_log[5];
        R.A_log[5] = -keep; R.upload_weights();
        R.set_slot(3, cs0, rs0); R.call(0, T, {3});
        std::vector<float> ym = R.out(0, T);
        Err e = grade(ym.data(), ry.data(), ym.size());
        check(e.fro > 100 * BAR, "A_log[5] sign flipped: output grades fro %.2e", e.fro);
        R.A_log[5] = keep; R.upload_weights();
        /* (b) one recurrent-state element, graded after 8 tokens (a fast-decaying
         * head would forget it over 257) */
        std::vector<double> c8(cs0.begin(), cs0.end()), r8(rs0.begin(), rs0.end()), y8((size_t)8 * VD);
        gdn_ref_seq(rw, 8, R.qkv.data(), QKV, R.z.data(), VD, R.a.data(), NV, R.b.data(), NV, c8.data(), r8.data(), y8.data());
        std::vector<float> rsm = rs0; rsm[(size_t)17 * 128 * 128 + 40 * 128 + 99] += 0.5f;
        R.set_slot(3, cs0, rsm); R.call(0, 8, {3});
        std::vector<float> csx, rsx; R.get_slot(3, csx, rsx);
        e = grade(rsx.data(), r8.data(), rsx.size());
        check(e.fro > 100 * BAR, "S[17][40][99] += 0.5 in the initial state: recurrent state grades fro %.2e", e.fro);
        /* (c) one k channel's conv taps reversed */
        const int c = 2048 + 5 * 128 + 7;
        std::swap(R.conv_w[c * 4 + 0], R.conv_w[c * 4 + 3]); std::swap(R.conv_w[c * 4 + 1], R.conv_w[c * 4 + 2]);
        R.upload_weights();
        R.set_slot(3, cs0, rs0); R.call(0, T, {3});
        ym = R.out(0, T);
        e = grade(ym.data(), ry.data(), ym.size());
        check(e.fro > 100 * BAR, "conv taps of channel %d reversed: output grades fro %.2e", c, e.fro);
        std::swap(R.conv_w[c * 4 + 0], R.conv_w[c * 4 + 3]); std::swap(R.conv_w[c * 4 + 1], R.conv_w[c * 4 + 2]);
        R.upload_weights();
        /* (d), (e) a wrong reference must fail against the kernel's (correct) run */
        const gdn_ref_mut muts[2] = {GDN_REF_GQA_TILED, GDN_REF_NO_QSCALE};
        const char *names[2] = {"GQA map tiled (h % 16)", "q scale dropped"};
        for (int m = 0; m < 2; m++) {
            std::vector<double> mc(cs0.begin(), cs0.end()), mr(rs0.begin(), rs0.end()), my((size_t)T * VD);
            gdn_ref_seq(rw, T, R.qkv.data(), QKV, R.z.data(), VD, R.a.data(), NV, R.b.data(), NV, mc.data(), mr.data(), my.data(), muts[m]);
            e = grade(y1.data(), my.data(), y1.size());
            check(e.fro > 100 * BAR, "reference with %s: the kernel grades fro %.2e against it", names[m], e.fro);
        }
    }

    /* ---- 7. refusals --------------------------------------------------------- */
    printf("7. refusals (each returns -1 and leaves the output untouched)\n");
    {
        CK(cudaMemset(R.d_out, 0x7f, (size_t)4 * VD * sizeof(float)));
        std::vector<float> before = R.out(0, 4);
        const pulsar_gdn_weights w = R.weights();
        auto base = [&]() {
            std::vector<int32_t> sl = {2, 2, 2, 2};
            h2d(R.d_slot, sl.data(), 4);
            pulsar_gdn_call c{};
            c.n_seq = 1; c.seq_rows = 4; c.row_slot = R.d_slot;
            c.conv_state = R.d_cs; c.rec_state = R.d_rs;
            c.qkv = R.d_qkv; c.ld_qkv = QKV; c.z = R.d_z; c.ld_z = VD; c.a = R.d_a; c.ld_a = NV; c.b = R.d_b; c.ld_b = NV;
            c.scratch = R.d_scratch; c.scratch_bytes = R.scratch_bytes; c.out_f32 = R.d_out;
            return c;
        };
        struct { const char *what; void (*mut)(pulsar_gdn_call &); } cases[] = {
            {"qkv NULL", [](pulsar_gdn_call &c) { c.qkv = nullptr; }},
            {"ld_qkv not a multiple of 4", [](pulsar_gdn_call &c) { c.ld_qkv = QKV + 2; }},
            {"scratch one byte short", [](pulsar_gdn_call &c) { c.scratch_bytes = pulsar_gdn_scratch_bytes(4) - 1; }},
            {"no output", [](pulsar_gdn_call &c) { c.out_f32 = nullptr; }},
            {"A8 slot with a wrong kbp", [](pulsar_gdn_call &c) { c.out_e4m3 = c.scratch; c.out_scale = c.scratch; c.out_kbp = 191; }},
            {"seq_rows 0", [](pulsar_gdn_call &c) { c.seq_rows = 0; }},
        };
        for (auto &cs : cases) {
            pulsar_gdn_call c = base(); cs.mut(c);
            const int rc = pulsar_gdn_forward(&w, &c, 0);
            CK(cudaDeviceSynchronize());
            check(rc == -1 && same(before, R.out(0, 4)), "%s -> %d", cs.what, rc);
        }
    }

    /* ---- 7. tensor parallelism (L266 step 7): a rank's heads ------------------------------------
     * The GDN is head-local -- conv per channel, L2 norms per head, the recurrence and gated norm per V
     * head (reading K head h / 3) -- so a rank running ITS heads (tp_ranks 2: 8 key heads + the 24 value
     * heads reading them) must give exactly the full layer's columns and state for those heads. */
    {
        const int T7 = 97, NKL = PULSAR_GDN_NK / 2, NVL = NV / 2, DKH = PULSAR_GDN_DK;
        const int QKVL = 2 * NKL * DKH + NVL * DKH, VDL = NVL * DKH;
        printf("7. tp_ranks 2: each rank's %d key + %d value heads vs the full layer's, %d tokens\n", NKL, NVL, T7);
        check(pulsar_gdn_conv_state_floats(2) == (size_t)3 * QKVL && pulsar_gdn_rec_state_floats(2) == RSF / 2,
              "per-rank state sizes: conv %zu, recurrent %zu floats", pulsar_gdn_conv_state_floats(2),
              pulsar_gdn_rec_state_floats(2));
        R.set_slot(5, cs0, rs0);
        check(R.call(0, T7, {5}, false) == 0, "full-layer call");
        const std::vector<float> yfull = R.out(0, T7);
        std::vector<float> csF, rsF;
        R.get_slot(5, csF, rsF);
        for (int r = 0; r < 2; r++) {
            /* the rank's channels in the full layer's order: its q heads, its k heads, its v heads */
            std::vector<int> ch;
            for (int i = 0; i < NKL * DKH; i++) ch.push_back(r * NKL * DKH + i);
            for (int i = 0; i < NKL * DKH; i++) ch.push_back(PULSAR_GDN_NK * DKH + r * NKL * DKH + i);
            for (int i = 0; i < VDL; i++) ch.push_back(2 * PULSAR_GDN_NK * DKH + r * VDL + i);
            std::vector<float> qkvl((size_t)T7 * QKVL), zl((size_t)T7 * VDL), csl((size_t)3 * QKVL);
            std::vector<uint16_t> cwl((size_t)QKVL * 4);
            for (int t = 0; t < T7; t++) {
                for (int i = 0; i < QKVL; i++) qkvl[(size_t)t * QKVL + i] = R.qkv[(size_t)t * QKV + ch[i]];
                for (int i = 0; i < VDL; i++) zl[(size_t)t * VDL + i] = R.z[(size_t)t * VD + r * VDL + i];
            }
            for (int i = 0; i < QKVL; i++) {
                for (int k = 0; k < 4; k++) cwl[(size_t)i * 4 + k] = R.conv_w_b[(size_t)ch[i] * 4 + k];
                for (int j = 0; j < 3; j++) csl[(size_t)j * QKVL + i] = cs0[(size_t)j * QKV + ch[i]];
            }
            float *dq = dalloc<float>(qkvl.size()), *dz = dalloc<float>(zl.size()), *dcs = dalloc<float>(csl.size());
            float *drs = dalloc<float>(RSF / 2), *dout = dalloc<float>((size_t)T7 * VDL);
            uint16_t *dcw = dalloc<uint16_t>(cwl.size());
            int32_t *dslot = dalloc<int32_t>(T7);
            h2d(dq, qkvl.data(), qkvl.size()); h2d(dz, zl.data(), zl.size()); h2d(dcs, csl.data(), csl.size());
            h2d(drs, rs0.data() + (size_t)r * (RSF / 2), RSF / 2); h2d(dcw, cwl.data(), cwl.size());
            CK(cudaMemset(dslot, 0, (size_t)T7 * sizeof(int32_t)));
            pulsar_gdn_call c{};
            c.tp_ranks = 2; c.n_seq = 1; c.seq_rows = T7; c.row_slot = dslot; c.conv_state = dcs; c.rec_state = drs;
            c.qkv = dq; c.ld_qkv = QKVL; c.z = dz; c.ld_z = VDL;
            c.a = R.d_a + (size_t)r * NVL; c.ld_a = NV; c.b = R.d_b + (size_t)r * NVL; c.ld_b = NV;
            c.scratch = R.d_scratch; c.scratch_bytes = R.scratch_bytes; c.out_f32 = dout;
            const pulsar_gdn_weights wl = {dcw, R.d_A_log + (size_t)r * NVL, R.d_dt_bias + (size_t)r * NVL, R.d_norm_w};
            const int rc = pulsar_gdn_forward(&wl, &c, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> yl((size_t)T7 * VDL), csr(csl.size()), rsr(RSF / 2);
            d2h(yl.data(), dout, yl.size()); d2h(csr.data(), dcs, csr.size()); d2h(rsr.data(), drs, rsr.size());
            size_t dy = 0, dc = 0;
            for (int t = 0; t < T7; t++)
                for (int i = 0; i < VDL; i++) dy += memcmp(&yl[(size_t)t * VDL + i], &yfull[(size_t)t * VD + r * VDL + i], 4) != 0;
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < QKVL; i++) dc += memcmp(&csr[(size_t)j * QKVL + i], &csF[(size_t)j * QKV + ch[i]], 4) != 0;
            const bool rs_same = memcmp(rsr.data(), rsF.data() + (size_t)r * (RSF / 2), (RSF / 2) * sizeof(float)) == 0;
            check(rc == 0 && dy == 0 && dc == 0 && rs_same,
                  "rank %d: rc %d; output %zu, conv state %zu elements differ; recurrent state %s", r, rc, dy, dc,
                  rs_same ? "bit-identical" : "DIFFERS");
            cudaFree(dq); cudaFree(dz); cudaFree(dcs); cudaFree(drs); cudaFree(dout); cudaFree(dcw); cudaFree(dslot);
        }
    }

    printf("gdn_gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}

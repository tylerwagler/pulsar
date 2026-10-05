/* QSA ATTENTION gate (L251 S3): pulsar_gpu_qsa_forward against a double-precision
 * host reference, model-free.
 *
 * FIXTURES (deterministic, generated per (sequence, position) so no copy is kept)
 *   A  3000 tokens, random projections with outlier K/V channels: rows past
 *      position 2050 select (nb > 512), so the dense and the selecting lanes
 *      both run, and the selecting rows carry every tail length.
 *   T  2600 tokens whose indexer keys have zero rotary dims and one of 24
 *      patterns per block: many blocks score EXACTLY equal, so the top-512
 *      boundary falls inside a tie and the tie rule (lower block wins) decides.
 *   B  40000 tokens (nb up to 10000: the engine top-k's chunk + merge tree),
 *      checked on sampled rows.
 *
 * CHECKS (each against the host reference unless it says otherwise)
 *   G1  cache encode: every K/V element is the E4M3 of the double value at the
 *       block's MX scale; a byte may differ only at a rounding midpoint (K is
 *       f32 on the device, double here).  Block keys within one bf16 rounding.
 *   G2  selection == the double top-512 by (score desc, block asc); a block may
 *       differ only if its score is within 1e-5 of the 512th (f32 vs double).
 *   G3  fixture T: selection IDENTICAL, and the boundary really is a tie.
 *   G4  the gated output vs double attention over the device's own cache and
 *       selection (so G4 grades the attention arithmetic alone).
 *   G5  the o_proj E4M3 slot == the MX encode of the f32 tap, swizzle included.
 *   G6  decode == prefill: one row per call, odd chunks, and a two-sequence
 *       batch all give the canonical run's bytes (outputs, selection, cache).
 *   G7  mutations: each deliberately wrong host variant must FAIL its check --
 *       query/gate order swapped, interleaved RoPE pairs, w in place of 1 + w,
 *       the tie rule reversed, the tail dropped, pooling after the norm, the
 *       block key roped at the block's end, one cache byte flipped.
 *
 * usage: qsa_attn_gate            (needs a device; ~1-2 min on GB10) */
#include "pulsar_gpu.h"
#include "pulsar_cuda_mx.cuh"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t HD = PULSAR_QSA_HEAD_DIM, NQ = PULSAR_QSA_N_HEAD, NKV = PULSAR_QSA_N_KV;
constexpr uint32_t IH = PULSAR_QSA_IDX_HEADS, ID = PULSAR_QSA_IDX_DIM, TOPB = PULSAR_QSA_TOP_BLOCKS;
constexpr uint32_t QIN = PULSAR_QSA_Q_IN, KVIN = PULSAR_QSA_KV_IN, IIN = PULSAR_QSA_IDX_IN, OUT = PULSAR_QSA_OUT_DIM;
constexpr uint32_t REC = PULSAR_QSA_KV_TOKEN_BYTES;
constexpr uint32_t KBP_OUT = OUT / 32u;   /* 192, a multiple of 4: the slot's KBp */

int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); } else { printf("  ok    "); } \
                              printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)

/* ------------------------------------------------------------------ fixture */
uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
struct Rng {
    uint64_t s;
    double u() { s = splitmix(s); return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    float n() { return (float)(sqrt(-2.0 * log(u())) * cos(6.283185307179586 * u())); }
};

enum { FIX_A = 0, FIX_T = 1, FIX_B = 2 };

/* The CONTAINER stores the norms bf16 (format-maps/qwen38fn-u-e4-d5.json); the
 * f32 arrays are the double reference and are rounded in place, so the kernel
 * and the reference read the same values. */
struct Norms { float q[HD], k[HD], iq[ID], ik[ID]; uint16_t q16[HD], k16[HD], iq16[ID], ik16[ID]; };
Norms g_norm;
float g_pattern[24][ID];

void gen_norms() {
    Rng r{0x5eed0001ull};
    for (uint32_t i = 0; i < HD; i++) g_norm.q[i] = 0.3f * r.n();
    for (uint32_t i = 0; i < HD; i++) g_norm.k[i] = 0.3f * r.n();
    for (uint32_t i = 0; i < ID; i++) g_norm.iq[i] = 0.3f * r.n();
    for (uint32_t i = 0; i < ID; i++) g_norm.ik[i] = 0.3f * r.n();
    auto r16 = [](float *f, uint16_t *b, int n) {
        for (int i = 0; i < n; i++) {
            uint32_t u; memcpy(&u, &f[i], 4); u = (u + 0x8000u) & 0xFFFF0000u;
            memcpy(&f[i], &u, 4); b[i] = (uint16_t)(u >> 16);
        }
    };
    r16(g_norm.q, g_norm.q16, HD); r16(g_norm.k, g_norm.k16, HD);
    r16(g_norm.iq, g_norm.iq16, ID); r16(g_norm.ik, g_norm.ik16, ID);
    for (int p = 0; p < 24; p++) for (uint32_t d = 0; d < ID; d++) g_pattern[p][d] = d < 64 ? 0.f : r.n();
}

/* The four projection rows of token `pos` of fixture `fix`. */
void gen_row(int fix, uint32_t pos, float *qg, float *k, float *v, float *idx) {
    Rng r{splitmix(((uint64_t)fix << 48) ^ ((uint64_t)pos << 8) ^ 0x77ull)};
    for (uint32_t i = 0; i < QIN; i++) qg[i] = 1.5f * r.n();
    for (uint32_t i = 0; i < KVIN; i++) k[i] = (i % 97u == 5u ? 20.f : 2.f) * r.n();
    for (uint32_t i = 0; i < KVIN; i++) v[i] = (i % 61u == 7u ? 30.f : 1.f) * r.n();
    for (uint32_t i = 0; i < IIN; i++) idx[i] = r.n();
    if (fix == FIX_T) {
        const uint32_t pat = (uint32_t)(splitmix(pos / 4u + 991u) % 24u);
        for (uint32_t d = 0; d < ID; d++) idx[IH * ID + d] = g_pattern[pat][d];
    }
}

/* ------------------------------------------------------------ host reference */
struct Mut {
    bool swap_qg = false, rope_interleaved = false, plain_w = false, tie_high = false,
         no_tail = false, pool_after_norm = false, bkey_rope_end = false;
};
float g_inv[PULSAR_QSA_ROT_DIM / 2];

void ref_norm_rope(double *x, uint32_t n, const float *w, uint32_t pos, const Mut &m, bool rope = true) {
    double ss = 0;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    const double r = 1.0 / sqrt(ss / n + (double)PULSAR_QSA_RMS_EPS);
    for (uint32_t i = 0; i < n; i++) x[i] = x[i] * r * (m.plain_w ? (double)w[i] : 1.0 + (double)w[i]);
    if (!rope) return;
    for (uint32_t i = 0; i < PULSAR_QSA_ROT_DIM / 2; i++) {
        const float ang = (float)pos * g_inv[i];   /* the f32 angle transformers forms */
        const double c = cos((double)ang), s = sin((double)ang);
        const uint32_t a = m.rope_interleaved ? 2 * i : i, b = m.rope_interleaved ? 2 * i + 1 : i + 32;
        const double x0 = x[a], x1 = x[b];
        x[a] = x0 * c - x1 * s;
        x[b] = x1 * c + x0 * s;
    }
}

double e4m3_decode(uint8_t b) {
    const int e = (b >> 3) & 15, mt = b & 7;
    if (e == 15 && mt == 7) return NAN;
    const double v = e == 0 ? mt * ldexp(1.0, -9) : (1.0 + mt / 8.0) * ldexp(1.0, e - 7);
    return (b & 0x80) ? -v : v;
}
/* RNE to E4M3 of |a| < 256 (the MX scale keeps it there); frac = distance data for the midpoint test */
double e4m3_round(double a, double *frac) {
    const double sa = fabs(a);
    if (sa == 0) { *frac = 0; return 0; }
    double step = sa < ldexp(1.0, -6) ? ldexp(1.0, -9) : ldexp(1.0, (int)floor(log2(sa)) - 3);
    const double t = sa / step;
    *frac = t - floor(t);
    const double q = nearbyint(t) * step;
    return a < 0 ? -q : q;
}
int mx_exp(double amax) {
    if (amax <= 0) return -127;
    int e;
    frexp(amax, &e);           /* amax = f * 2^e, f in [0.5, 1) */
    return std::min(127, std::max(-127, (e - 1) - 7));
}
bool near_pow2(double a) { int e; const double f = frexp(a, &e); return fabs(f - 0.5) < 1e-5 * 0.5 || fabs(f - 1.0) < 1e-5; }

double bf16_to_d(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

/* ------------------------------------------------------------ device harness */
struct Seq {
    int fix;
    uint32_t T, cap;
    pulsar_gpu_tensor *kv, *bkey, *stage;
    pulsar_qsa_seq desc() const { return {kv, bkey, stage, cap}; }
};
struct Dev {
    uint32_t max_rows;
    pulsar_gpu_tensor *qg, *k, *v, *idx, *slot, *slot_sc, *tap, *sel, *ws;
    pulsar_gpu_tensor *qn, *kn, *iqn, *ikn;
    uint32_t max_ctx;
};
Dev g_dev;

pulsar_gpu_tensor *upload(const void *p, uint64_t bytes) {
    pulsar_gpu_tensor *t = pulsar_gpu_tensor_alloc(bytes);
    if (!t || !pulsar_gpu_tensor_write(t, 0, p, bytes)) { printf("alloc/write failed\n"); exit(2); }
    return t;
}
pulsar_gpu_tensor *zeros(uint64_t bytes) {
    pulsar_gpu_tensor *t = pulsar_gpu_tensor_alloc(bytes);
    if (!t || !pulsar_gpu_tensor_fill_f32(t, 0.f, bytes / 4)) { printf("alloc failed\n"); exit(2); }
    return t;
}

void dev_init(uint32_t max_rows, uint32_t max_ctx) {
    Dev &d = g_dev;
    d.max_rows = max_rows;
    d.max_ctx = max_ctx;
    d.qg = zeros((uint64_t)max_rows * QIN * 4);
    d.k = zeros((uint64_t)max_rows * KVIN * 4);
    d.v = zeros((uint64_t)max_rows * KVIN * 4);
    d.idx = zeros((uint64_t)max_rows * IIN * 4);
    d.slot = zeros((uint64_t)max_rows * OUT);
    d.slot_sc = zeros(pulsar_mx_sf_slab_bytes((int)max_rows, (int)KBP_OUT));
    d.tap = zeros((uint64_t)max_rows * OUT * 4);
    d.sel = zeros((uint64_t)max_rows * TOPB * 4);
    d.ws = zeros(pulsar_gpu_qsa_workspace_bytes(max_rows, max_ctx));
    d.qn = upload(g_norm.q16, HD * 2);
    d.kn = upload(g_norm.k16, HD * 2);
    d.iqn = upload(g_norm.iq16, ID * 2);
    d.ikn = upload(g_norm.ik16, ID * 2);
}

Seq make_seq(int fix, uint32_t T) {
    Seq s;
    s.fix = fix;
    s.T = T;
    s.cap = (T + 3u) & ~3u;
    s.kv = zeros((uint64_t)s.cap * REC);
    s.bkey = zeros((uint64_t)(s.cap / 4) * PULSAR_QSA_BKEY_BYTES);
    s.stage = zeros(PULSAR_QSA_STAGE_BYTES);
    return s;
}
void seq_free(Seq &s) {
    pulsar_gpu_tensor_free(s.kv);
    pulsar_gpu_tensor_free(s.bkey);
    pulsar_gpu_tensor_free(s.stage);
}

/* What a run keeps per (sequence, position). */
struct RowOut {
    std::vector<float> out;        /* [6144], kept when `keep` */
    std::vector<uint32_t> sel;     /* [512] */
    std::vector<uint8_t> slot;     /* [6144 + 192]: data then the row's scale bytes */
    uint64_t hash = 0;             /* FNV of out + sel */
};
uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Piece { uint32_t seq, pos0, n; };   /* one sequence's run inside a call */

/* One forward call over the pieces; results into outs[seq][pos]. */
void call(std::vector<Seq> &seqs, const std::vector<Piece> &pieces,
          std::vector<std::vector<RowOut>> &outs, const std::function<bool(uint32_t, uint32_t)> &keep) {
    Dev &d = g_dev;
    uint32_t n = 0;
    for (auto &p : pieces) n += p.n;
    if (n > d.max_rows) { printf("call too wide\n"); exit(2); }
    std::vector<float> qg((size_t)n * QIN), k((size_t)n * KVIN), v((size_t)n * KVIN), idx((size_t)n * IIN);
    std::vector<uint32_t> rs(n), rp(n);
    uint32_t r = 0;
    for (auto &p : pieces) {
        for (uint32_t i = 0; i < p.n; i++, r++) {
            rs[r] = p.seq;
            rp[r] = p.pos0 + i;
            gen_row(seqs[p.seq].fix, rp[r], &qg[(size_t)r * QIN], &k[(size_t)r * KVIN], &v[(size_t)r * KVIN],
                    &idx[(size_t)r * IIN]);
        }
    }
    pulsar_gpu_tensor_write(d.qg, 0, qg.data(), qg.size() * 4);
    pulsar_gpu_tensor_write(d.k, 0, k.data(), k.size() * 4);
    pulsar_gpu_tensor_write(d.v, 0, v.data(), v.size() * 4);
    pulsar_gpu_tensor_write(d.idx, 0, idx.data(), idx.size() * 4);
    pulsar_gpu_tensor_fill_f32(d.slot_sc, 0.f, pulsar_gpu_tensor_bytes(d.slot_sc) / 4);
    std::vector<pulsar_qsa_seq> sd;
    for (auto &s : seqs) sd.push_back(s.desc());
    pulsar_qsa_layer L{(const uint16_t *)pulsar_gpu_tensor_device_ptr(d.qn), (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.kn),
                       (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.iqn), (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.ikn)};
    pulsar_qsa_io io{};
    io.qg = d.qg; io.k = d.k; io.v = d.v; io.idx = d.idx;
    io.out_e4m3 = pulsar_gpu_tensor_device_ptr(d.slot);
    io.out_scale = pulsar_gpu_tensor_device_ptr(d.slot_sc);
    io.out_sf_pitch = (int)KBP_OUT;
    io.tap_out_f32 = d.tap;
    io.tap_sel = d.sel;
    if (!pulsar_gpu_qsa_forward(&L, sd.data(), (uint32_t)sd.size(), rs.data(), rp.data(), n, &io, d.ws)) {
        printf("FAIL  pulsar_gpu_qsa_forward refused\n");
        exit(1);
    }
    std::vector<float> tap((size_t)n * OUT);
    std::vector<uint32_t> sel((size_t)n * TOPB);
    std::vector<uint8_t> slot((size_t)n * OUT), sc(pulsar_gpu_tensor_bytes(d.slot_sc));
    pulsar_gpu_tensor_read(d.tap, 0, tap.data(), tap.size() * 4);
    pulsar_gpu_tensor_read(d.sel, 0, sel.data(), sel.size() * 4);
    pulsar_gpu_tensor_read(d.slot, 0, slot.data(), slot.size());
    pulsar_gpu_tensor_read(d.slot_sc, 0, sc.data(), sc.size());
    for (uint32_t i = 0; i < n; i++) {
        RowOut &o = outs[rs[i]][rp[i]];
        std::vector<uint8_t> srow(&slot[(size_t)i * OUT], &slot[(size_t)i * OUT] + OUT);
        for (uint32_t kb = 0; kb < KBP_OUT; kb++) srow.push_back(sc[pulsar_mx_sfoff((int)i, (int)kb, (int)KBP_OUT)]);
        /* the hash covers everything a row emits: the f32 tap, the selection, and the E4M3 slot + its scales */
        o.hash = fnv(srow.data(), srow.size(), fnv(&sel[(size_t)i * TOPB], TOPB * 4, fnv(&tap[(size_t)i * OUT], OUT * 4)));
        if (keep(rs[i], rp[i])) {
            o.out.assign(&tap[(size_t)i * OUT], &tap[(size_t)i * OUT] + OUT);
            o.sel.assign(&sel[(size_t)i * TOPB], &sel[(size_t)i * TOPB] + TOPB);
            o.slot = srow;
        }
    }
}

/* A run: a schedule of calls over fresh caches. */
struct Run {
    std::vector<std::vector<RowOut>> outs;
    std::vector<std::vector<uint8_t>> kv, bkey;
};
Run run(std::vector<Seq> &seqs, const std::vector<std::vector<Piece>> &schedule,
        const std::function<bool(uint32_t, uint32_t)> &keep) {
    Run R;
    R.outs.resize(seqs.size());
    for (size_t s = 0; s < seqs.size(); s++) {
        R.outs[s].assign(seqs[s].T, RowOut());
        pulsar_gpu_tensor_fill_f32(seqs[s].kv, 0.f, pulsar_gpu_tensor_bytes(seqs[s].kv) / 4);
        pulsar_gpu_tensor_fill_f32(seqs[s].bkey, 0.f, pulsar_gpu_tensor_bytes(seqs[s].bkey) / 4);
        pulsar_gpu_tensor_fill_f32(seqs[s].stage, 1e30f, PULSAR_QSA_STAGE_BYTES / 4);  /* stale stage = loud */
    }
    for (auto &c : schedule) call(seqs, c, R.outs, keep);
    R.kv.resize(seqs.size());
    R.bkey.resize(seqs.size());
    for (size_t s = 0; s < seqs.size(); s++) {
        R.kv[s].resize((size_t)seqs[s].T * REC);
        R.bkey[s].resize((size_t)(seqs[s].T / 4) * PULSAR_QSA_BKEY_BYTES);
        pulsar_gpu_tensor_read(seqs[s].kv, 0, R.kv[s].data(), R.kv[s].size());
        pulsar_gpu_tensor_read(seqs[s].bkey, 0, R.bkey[s].data(), R.bkey[s].size());
    }
    return R;
}
std::vector<std::vector<Piece>> chunks(uint32_t seq, uint32_t T, const std::vector<uint32_t> &sizes) {
    std::vector<std::vector<Piece>> s;
    uint32_t p = 0, i = 0;
    while (p < T) {
        const uint32_t n = std::min(T - p, sizes[std::min<size_t>(i, sizes.size() - 1)]);
        s.push_back({{seq, p, n}});
        p += n;
        i++;
    }
    return s;
}

/* ------------------------------------------------------------ checks */
struct Tok { std::vector<double> q[NQ], gate[NQ], k[NKV], v[NKV], iq[IH]; std::vector<float> rawk; };

Tok host_tok(int fix, uint32_t pos, const Mut &m) {
    std::vector<float> qg(QIN), k(KVIN), v(KVIN), idx(IIN);
    gen_row(fix, pos, qg.data(), k.data(), v.data(), idx.data());
    Tok t;
    for (uint32_t h = 0; h < NQ; h++) {
        const float *qa = &qg[(size_t)h * 2 * HD], *ga = qa + HD;
        if (m.swap_qg) std::swap(qa, ga);
        t.q[h].assign(qa, qa + HD);
        t.gate[h].assign(ga, ga + HD);
        ref_norm_rope(t.q[h].data(), HD, g_norm.q, pos, m);
    }
    for (uint32_t g = 0; g < NKV; g++) {
        t.k[g].assign(&k[g * HD], &k[g * HD] + HD);
        ref_norm_rope(t.k[g].data(), HD, g_norm.k, pos, m);
        t.v[g].assign(&v[g * HD], &v[g * HD] + HD);
    }
    for (uint32_t h = 0; h < IH; h++) {
        t.iq[h].assign(&idx[h * ID], &idx[h * ID] + ID);
        ref_norm_rope(t.iq[h].data(), ID, g_norm.iq, pos, m);
    }
    t.rawk.assign(&idx[IH * ID], &idx[IH * ID] + ID);
    return t;
}

std::vector<double> host_bkey(int fix, uint32_t b, const Mut &m) {
    std::vector<double> x(ID, 0.0);
    for (uint32_t j = 0; j < 4; j++) {
        Tok t = host_tok(fix, 4 * b + j, Mut());
        std::vector<double> kk(t.rawk.begin(), t.rawk.end());
        if (m.pool_after_norm) ref_norm_rope(kk.data(), ID, g_norm.ik, 0, m, false);
        for (uint32_t d = 0; d < ID; d++) x[d] += kk[d] / 4.0;
    }
    if (m.pool_after_norm) {   /* rope only: the norm already ran per token */
        std::vector<double> y = x;
        for (uint32_t i = 0; i < PULSAR_QSA_ROT_DIM / 2; i++) {
            const float ang = (float)(4 * b) * g_inv[i];
            const double c = cos((double)ang), s = sin((double)ang);
            y[i] = x[i] * c - x[i + 32] * s;
            y[i + 32] = x[i + 32] * c + x[i] * s;
        }
        return y;
    }
    ref_norm_rope(x.data(), ID, g_norm.ik, m.bkey_rope_end ? 4 * b + 3 : 4 * b, m);
    return x;
}

/* G1: K/V encode of token `pos`; returns unexplained mismatches, counts midpoint allowances. */
uint64_t check_encode(const uint8_t *rec, const Tok &t, uint64_t *allow) {
    uint64_t bad = 0;
    for (uint32_t kvsel = 0; kvsel < 2; kvsel++) {
        for (uint32_t g = 0; g < NKV; g++) {
            const std::vector<double> &x = kvsel ? t.v[g] : t.k[g];
            const uint8_t *data = rec + (kvsel ? (NKV + g) : g) * HD;
            const uint8_t *scl = rec + 2 * NKV * HD + (kvsel ? (NKV + g) : g) * (HD / 32);
            for (uint32_t b = 0; b < HD / 32; b++) {
                double amax = 0;
                for (uint32_t e = 0; e < 32; e++) amax = std::max(amax, fabs(x[b * 32 + e]));
                const int se = (int)scl[b] - 127;
                if (se != mx_exp(amax)) {
                    if (near_pow2(amax)) (*allow)++; else { bad++; continue; }
                }
                for (uint32_t e = 0; e < 32; e++) {
                    double frac;
                    const double want = e4m3_round(x[b * 32 + e] / ldexp(1.0, se), &frac);
                    const double got = e4m3_decode(data[b * 32 + e]);
                    if (got != want) {
                        if (fabs(frac - 0.5) < 1e-4) (*allow)++; else bad++;
                    }
                }
            }
        }
    }
    return bad;
}

/* The double top-512 by (score desc, block asc) -- or block desc under the mutant. */
std::vector<uint32_t> host_select(const Tok &t, const std::vector<std::vector<double>> &bk, uint32_t nb,
                                  const Mut &m, std::vector<double> *scores) {
    std::vector<double> &sc = *scores;
    sc.assign(nb, 0.0);
    for (uint32_t b = 0; b < nb; b++) {
        double s = 0;
        for (uint32_t h = 0; h < IH; h++) {
            double dot = 0;
            for (uint32_t d = 0; d < ID; d++) dot += t.iq[h][d] * bk[b][d];
            s += dot > 0 ? dot : 0;
        }
        sc[b] = s;
    }
    std::vector<uint32_t> ids(nb);
    for (uint32_t b = 0; b < nb; b++) ids[b] = b;
    std::partial_sort(ids.begin(), ids.begin() + TOPB, ids.end(), [&](uint32_t a, uint32_t b) {
        if (sc[a] != sc[b]) return sc[a] > sc[b];
        return m.tie_high ? a > b : a < b;
    });
    ids.resize(TOPB);
    std::sort(ids.begin(), ids.end());
    return ids;
}

/* G4: max |ours - ref| / max|ref| for one row, attention over the DEVICE cache + selection. */
double check_row_out(const Seq &s, const std::vector<uint8_t> &kv, uint32_t pos, const RowOut &o, const Mut &m) {
    Tok t = host_tok(s.fix, pos, m);
    const uint32_t n = pos + 1, nb = n / 4;
    std::vector<uint32_t> list;
    if (nb <= TOPB) {
        for (uint32_t j = 0; j < n; j++) list.push_back(j);
    } else {
        for (uint32_t i = 0; i < TOPB; i++) for (uint32_t j = 0; j < 4; j++) list.push_back(o.sel[i] * 4 + j);
        if (!m.no_tail) for (uint32_t j = nb * 4; j < n; j++) list.push_back(j);
    }
    std::vector<double> kd(list.size() * NKV * HD), vd(list.size() * NKV * HD);
    for (size_t i = 0; i < list.size(); i++) {
        const uint8_t *rec = &kv[(size_t)list[i] * REC];
        for (uint32_t g = 0; g < NKV; g++) {
            for (uint32_t d = 0; d < HD; d++) {
                kd[(i * NKV + g) * HD + d] = e4m3_decode(rec[g * HD + d]) * ldexp(1.0, (int)rec[2 * NKV * HD + g * 8 + d / 32] - 127);
                vd[(i * NKV + g) * HD + d] = e4m3_decode(rec[(NKV + g) * HD + d]) * ldexp(1.0, (int)rec[2 * NKV * HD + (NKV + g) * 8 + d / 32] - 127);
            }
        }
    }
    double maxref = 0, maxerr = 0;
    std::vector<double> p(list.size()), acc(HD);
    for (uint32_t h = 0; h < NQ; h++) {
        const uint32_t g = h / (NQ / NKV);
        double mx = -INFINITY;
        for (size_t i = 0; i < list.size(); i++) {
            double dot = 0;
            for (uint32_t d = 0; d < HD; d++) dot += t.q[h][d] * kd[(i * NKV + g) * HD + d];
            p[i] = dot / 16.0;
            mx = std::max(mx, p[i]);
        }
        double l = 0;
        for (size_t i = 0; i < list.size(); i++) { p[i] = exp(p[i] - mx); l += p[i]; }
        std::fill(acc.begin(), acc.end(), 0.0);
        for (size_t i = 0; i < list.size(); i++)
            for (uint32_t d = 0; d < HD; d++) acc[d] += p[i] * vd[(i * NKV + g) * HD + d];
        for (uint32_t d = 0; d < HD; d++) {
            const double y = acc[d] / l / (1.0 + exp(-t.gate[h][d]));
            maxref = std::max(maxref, fabs(y));
            maxerr = std::max(maxerr, fabs(y - (double)o.out[h * HD + d]));
        }
    }
    return maxerr / maxref;
}

/* Run fn(i) for i in [0, n) on all cores; returns in order. */
template <class F>
void par_for(size_t n, F fn) {
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; w++) th.emplace_back([&, w] { for (size_t i = w; i < n; i += nt) fn(i); });
    for (auto &t : th) t.join();
}

struct SelStats { uint64_t rows = 0, identical = 0, near_tie_rows = 0, bad_rows = 0, tie_at_boundary = 0; };

SelStats check_selection(const Seq &s, const Run &R, const std::vector<uint32_t> &rows, const Mut &m,
                         double near_tol) {
    const uint32_t nbk = s.T / 4;
    std::vector<std::vector<double>> bk(nbk, std::vector<double>(ID));
    for (uint32_t b = 0; b < nbk; b++)
        for (uint32_t d = 0; d < ID; d++) {
            uint16_t u;
            memcpy(&u, &R.bkey[0][(size_t)b * PULSAR_QSA_BKEY_BYTES + d * 2], 2);
            bk[b][d] = bf16_to_d(u);
        }
    SelStats st;
    std::vector<int> res(rows.size());
    std::vector<int> tie(rows.size());
    par_for(rows.size(), [&](size_t i) {
        const uint32_t pos = rows[i];
        Tok t = host_tok(s.fix, pos, m);
        std::vector<double> sc;
        const uint32_t nb = (pos + 1) / 4;
        std::vector<uint32_t> want = host_select(t, bk, nb, m, &sc);
        const std::vector<uint32_t> &got = R.outs[0][pos].sel;
        std::vector<double> sorted(sc);
        std::nth_element(sorted.begin(), sorted.begin() + (TOPB - 1), sorted.end(), std::greater<double>());
        const double s512 = sorted[TOPB - 1];
        std::nth_element(sorted.begin(), sorted.begin() + TOPB, sorted.end(), std::greater<double>());
        tie[i] = sorted[TOPB] == s512;
        if (std::equal(want.begin(), want.end(), got.begin())) { res[i] = 0; return; }
        std::vector<uint32_t> diff;
        std::set_symmetric_difference(want.begin(), want.end(), got.begin(), got.end(), std::back_inserter(diff));
        bool ok = true;
        for (uint32_t b : diff) {
            if (b >= nb || fabs(sc[b] - s512) > near_tol * s512) ok = false;
        }
        res[i] = ok ? 1 : 2;
    });
    for (size_t i = 0; i < rows.size(); i++) {
        st.rows++;
        st.identical += res[i] == 0;
        st.near_tie_rows += res[i] == 1;
        st.bad_rows += res[i] == 2;
        st.tie_at_boundary += tie[i];
    }
    return st;
}

}  // namespace

int main() {
    if (!pulsar_gpu_init()) { printf("no device\n"); return 2; }
    gen_norms();
    pulsar_qsa_inv_freq(g_inv);
    const uint32_t TA = 3000, TT = 2600, TB = 40000;
    dev_init(2048, TB + 4);
    printf("qsa_attn_gate: fixtures A %u tok, T %u tok (ties), B %u tok\n", TA, TT, TB);

    /* ---------------- fixture A: the canonical run (chunks of 1000, not a multiple of 4) */
    std::vector<Seq> SA{make_seq(FIX_A, TA)};
    auto all = [](uint32_t, uint32_t) { return true; };
    Run A = run(SA, chunks(0, TA, {1000}), all);

    printf("G1 cache encode (fixture A, every token)\n");
    {
        std::vector<uint64_t> bad(TA), allow(TA);
        par_for(TA, [&](size_t p) { Tok t = host_tok(FIX_A, (uint32_t)p, Mut()); bad[p] = check_encode(&A.kv[0][p * REC], t, &allow[p]); });
        uint64_t nb = 0, na = 0;
        for (uint32_t p = 0; p < TA; p++) { nb += bad[p]; na += allow[p]; }
        CHECK(nb == 0, "K/V E4M3/MX32: %llu unexplained of %llu elements (%llu at a rounding midpoint / power-of-two amax)",
              (unsigned long long)nb, (unsigned long long)TA * 1024, (unsigned long long)na);
        double worst = 0;
        uint64_t over = 0;
        for (uint32_t b = 0; b < TA / 4; b++) {
            std::vector<double> x = host_bkey(FIX_A, b, Mut());
            double rms = 0;
            for (double v : x) rms += v * v;
            rms = sqrt(rms / ID);
            for (uint32_t d = 0; d < ID; d++) {
                uint16_t u;
                memcpy(&u, &A.bkey[0][(size_t)b * PULSAR_QSA_BKEY_BYTES + d * 2], 2);
                const double e = fabs(bf16_to_d(u) - x[d]);
                worst = std::max(worst, e / (fabs(x[d]) + 1e-6 * rms));
                if (e > ldexp(fabs(x[d]), -8) + 1e-6 * rms) over++;
            }
        }
        CHECK(over == 0, "block keys: %u blocks, worst |bf16 - double| / |double| = %.3g (bound 2^-8 = %.3g)",
              TA / 4, worst, ldexp(1.0, -8));
    }

    printf("G2 selection vs the double top-512 (fixture A, rows 2051..2999)\n");
    std::vector<uint32_t> sel_rows;
    for (uint32_t p = 2051; p < TA; p++) sel_rows.push_back(p);
    {
        SelStats st = check_selection(SA[0], A, sel_rows, Mut(), 1e-5);
        CHECK(st.bad_rows == 0 && st.identical + st.near_tie_rows == st.rows,
              "%llu rows: %llu identical, %llu differ only inside 1e-5 of the 512th score, %llu wrong",
              (unsigned long long)st.rows, (unsigned long long)st.identical,
              (unsigned long long)st.near_tie_rows, (unsigned long long)st.bad_rows);
        CHECK(st.identical * 10 >= st.rows * 9, "selection non-degenerate: identical on >= 90%% of rows");
    }

    printf("G4 gated output vs double attention (fixture A, sampled rows)\n");
    std::vector<uint32_t> out_rows;
    for (uint32_t p = 0; p < TA; p++) {
        if (p < 40 || (p >= 2040 && p < 2064) || p % 29 == 0 || p + 8 >= TA) out_rows.push_back(p);
    }
    double g4_bar = 2e-5;
    {
        std::vector<double> err(out_rows.size());
        par_for(out_rows.size(), [&](size_t i) { err[i] = check_row_out(SA[0], A.kv[0], out_rows[i], A.outs[0][out_rows[i]], Mut()); });
        std::vector<double> se(err);
        std::sort(se.begin(), se.end());
        CHECK(se.back() <= g4_bar, "%zu rows: max rel err %.3g, median %.3g (bar %.0e)", err.size(), se.back(),
              se[se.size() / 2], g4_bar);
    }

    printf("G5 o_proj E4M3 slot == MX encode of the f32 tap (fixture A, every row)\n");
    {
        uint64_t bad = 0, allow = 0;
        for (uint32_t p = 0; p < TA; p++) {
            const RowOut &o = A.outs[0][p];
            for (uint32_t kb = 0; kb < KBP_OUT; kb++) {
                double amax = 0;
                for (uint32_t e = 0; e < 32; e++) amax = std::max(amax, fabs((double)o.out[kb * 32 + e]));
                const int se = (int)o.slot[OUT + kb] - 127;
                if (se != mx_exp(amax)) { if (near_pow2(amax)) allow++; else { bad++; continue; } }
                for (uint32_t e = 0; e < 32; e++) {
                    double frac;
                    if (e4m3_round((double)o.out[kb * 32 + e] / ldexp(1.0, se), &frac) != e4m3_decode(o.slot[kb * 32 + e])) bad++;
                }
            }
        }
        CHECK(bad == 0, "%u rows x 6144: %llu mismatched bytes (%llu power-of-two scale choices)", TA,
              (unsigned long long)bad, (unsigned long long)allow);
    }

    printf("G6 decode == prefill (fixture A)\n");
    auto same_run = [&](const Run &X, const Run &Y, uint32_t seq, uint32_t T, const char *what) {
        uint64_t d_out = 0, d_kv = 0, d_bk = 0;
        for (uint32_t p = 0; p < T; p++) d_out += X.outs[seq][p].hash != Y.outs[seq][p].hash;
        d_kv = X.kv[seq] != Y.kv[seq];
        d_bk = X.bkey[seq] != Y.bkey[seq];
        CHECK(d_out == 0 && d_kv == 0 && d_bk == 0, "%s: %llu of %u rows differ; KV bytes %s; block keys %s", what,
              (unsigned long long)d_out, T, d_kv ? "DIFFER" : "identical", d_bk ? "DIFFER" : "identical");
    };
    {
        auto none = [](uint32_t, uint32_t) { return false; };
        Run D = run(SA, chunks(0, TA, {1}), none);
        same_run(A, D, 0, TA, "one row per call (3000 calls)");
        Run C = run(SA, chunks(0, TA, {1, 2, 3, 5, 7, 513, 1027, 2, 900, 11}), none);
        same_run(A, C, 0, TA, "odd chunks 1,2,3,5,7,513,1027,2,900,11,...");
    }

    /* ---------------- fixture T: the tie rule */
    printf("G3 exact ties at the top-512 boundary (fixture T)\n");
    std::vector<Seq> ST{make_seq(FIX_T, TT)};
    Run TR = run(ST, chunks(0, TT, {1024}), all);
    std::vector<uint32_t> t_rows;
    for (uint32_t p = 2051; p < TT; p++) t_rows.push_back(p);
    {
        SelStats st = check_selection(ST[0], TR, t_rows, Mut(), 0.0);
        CHECK(st.identical == st.rows, "%llu rows: %llu identical", (unsigned long long)st.rows, (unsigned long long)st.identical);
        CHECK(st.tie_at_boundary * 2 >= st.rows, "the 512th and 513th scores are EQUAL on %llu of %llu rows (the rule is exercised)",
              (unsigned long long)st.tie_at_boundary, (unsigned long long)st.rows);
    }

    printf("G6b two sequences in one batch == each alone (A + T)\n");
    {
        std::vector<Seq> two{SA[0], ST[0]};
        std::vector<std::vector<Piece>> sch;
        uint32_t pa = 0, pt = 0, i = 0;
        const uint32_t wa[] = {700, 1, 3, 1300, 1, 1, 996}, wt[] = {5, 1, 1000, 1, 2, 1591, 0};
        while (pa < TA || pt < TT) {
            std::vector<Piece> c;
            const uint32_t na = std::min(TA - pa, i < 7 ? wa[i] : 64u), nt = std::min(TT - pt, i < 7 ? wt[i] : 64u);
            if (nt) c.push_back({1, pt, nt});
            if (na) c.push_back({0, pa, na});
            pa += na; pt += nt; i++;
            if (!c.empty()) sch.push_back(c);
        }
        auto none = [](uint32_t, uint32_t) { return false; };
        Run M2 = run(two, sch, none);
        same_run(A, M2, 0, TA, "fixture A rows inside mixed batches");
        Run M2t;
        M2t.outs = {M2.outs[1]};
        M2t.kv = {M2.kv[1]};
        M2t.bkey = {M2.bkey[1]};
        same_run(TR, M2t, 0, TT, "fixture T rows inside mixed batches");
    }

    /* ---------------- fixture B: deep */
    printf("fixture B (%u tokens): chunked prefill 2048 vs 1999, then decode the last 64\n", TB);
    {
        std::vector<Seq> SB{make_seq(FIX_B, TB)};
        std::vector<uint32_t> b_rows;
        for (uint32_t p = 2051; p < TB; p += 1237) b_rows.push_back(p);
        for (uint32_t p = TB - 5; p < TB; p++) b_rows.push_back(p);
        auto keepB = [&](uint32_t, uint32_t p) { return std::binary_search(b_rows.begin(), b_rows.end(), p); };
        std::sort(b_rows.begin(), b_rows.end());
        Run B1 = run(SB, chunks(0, TB, {2048}), keepB);
        Run B2 = run(SB, chunks(0, TB, {1999}), keepB);
        same_run(B1, B2, 0, TB, "chunks 2048 vs 1999");
        std::vector<std::vector<Piece>> sch = chunks(0, TB - 64, {2048});
        for (uint32_t p = TB - 64; p < TB; p++) sch.push_back({{0, p, 1}});
        Run B3 = run(SB, sch, keepB);
        same_run(B1, B3, 0, TB, "prefill to 39936 then 64 single-row decodes");
        SelStats st = check_selection(SB[0], B1, b_rows, Mut(), 1e-5);
        CHECK(st.bad_rows == 0, "selection at depth: %llu rows, %llu identical, %llu near-tie, %llu wrong",
              (unsigned long long)st.rows, (unsigned long long)st.identical, (unsigned long long)st.near_tie_rows,
              (unsigned long long)st.bad_rows);
        std::vector<double> err(b_rows.size());
        par_for(b_rows.size(), [&](size_t i) { err[i] = check_row_out(SB[0], B1.kv[0], b_rows[i], B1.outs[0][b_rows[i]], Mut()); });
        CHECK(*std::max_element(err.begin(), err.end()) <= g4_bar, "output at depth: %zu rows, max rel err %.3g",
              err.size(), *std::max_element(err.begin(), err.end()));
        seq_free(SB[0]);
    }

    /* ---------------- G7 mutations: each wrong reference must FAIL */
    printf("G7 mutations (each must be CAUGHT)\n");
    {
        auto g4_fails = [&](const Mut &m) {
            double worst = 0;
            std::vector<double> err(out_rows.size());
            par_for(out_rows.size(), [&](size_t i) { err[i] = check_row_out(SA[0], A.kv[0], out_rows[i], A.outs[0][out_rows[i]], m); });
            for (double e : err) worst = std::max(worst, e);
            return worst > g4_bar;
        };
        Mut m;
        m = Mut(); m.swap_qg = true;           CHECK(g4_fails(m), "query/gate halves swapped -> G4 fails");
        m = Mut(); m.rope_interleaved = true;  CHECK(g4_fails(m), "RoPE on interleaved pairs -> G4 fails");
        m = Mut(); m.plain_w = true;           CHECK(g4_fails(m), "norm weight w instead of 1 + w -> G4 fails");
        m = Mut(); m.no_tail = true;           CHECK(g4_fails(m), "the open tail dropped -> G4 fails");
        {
            m = Mut(); m.tie_high = true;
            SelStats st = check_selection(ST[0], TR, t_rows, m, 0.0);
            CHECK(st.identical < st.rows, "tie rule reversed -> G3 fails on %llu of %llu rows",
                  (unsigned long long)(st.rows - st.identical), (unsigned long long)st.rows);
        }
        auto bkey_fails = [&](const Mut &mm) {
            uint64_t over = 0;
            for (uint32_t b = 0; b < TA / 4; b += 7) {
                std::vector<double> x = host_bkey(FIX_A, b, mm);
                for (uint32_t d = 0; d < ID; d++) {
                    uint16_t u;
                    memcpy(&u, &A.bkey[0][(size_t)b * PULSAR_QSA_BKEY_BYTES + d * 2], 2);
                    if (fabs(bf16_to_d(u) - x[d]) > ldexp(fabs(x[d]), -8) + 1e-6) over++;
                }
            }
            return over > 0;
        };
        m = Mut(); m.pool_after_norm = true;   CHECK(bkey_fails(m), "pool after the norm -> G1 block keys fail");
        m = Mut(); m.bkey_rope_end = true;     CHECK(bkey_fails(m), "block key roped at the block end -> G1 block keys fail");
        {
            std::vector<uint8_t> rec(&A.kv[0][1234 * REC], &A.kv[0][1234 * REC] + REC);
            rec[300] ^= 0x08;
            uint64_t allow = 0;
            CHECK(check_encode(rec.data(), host_tok(FIX_A, 1234, Mut()), &allow) > 0, "one K byte flipped -> G1 fails");
        }
    }

    printf("G8 tensor parallelism (L266 step 7): each rank's KV head + its 12 query heads == the full layer's (fixture A)\n");
    {
        const uint32_t REC2 = pulsar_qsa_kv_token_bytes(2), QIN2 = QIN / 2u, KVIN2 = KVIN / 2u, OUT2 = OUT / 2u;
        CHECK(REC2 == REC / 2u, "a rank's KV record is half the full one (%u of %u B)", REC2, REC);
        Dev &d = g_dev;
        for (uint32_t r = 0; r < 2; r++) {
            Seq s;
            s.fix = FIX_A; s.T = TA; s.cap = (TA + 3u) & ~3u;
            s.kv = zeros((uint64_t)s.cap * REC2);
            s.bkey = zeros((uint64_t)(s.cap / 4) * PULSAR_QSA_BKEY_BYTES);
            s.stage = zeros(PULSAR_QSA_STAGE_BYTES);
            pulsar_gpu_tensor_fill_f32(s.stage, 1e30f, PULSAR_QSA_STAGE_BYTES / 4);
            uint64_t dout = 0, dkv = 0;
            for (uint32_t p0 = 0; p0 < TA; p0 += 1000u) {
                const uint32_t n = std::min(1000u, TA - p0);
                std::vector<float> qg(QIN), k(KVIN), v(KVIN), idx((size_t)n * IIN);
                std::vector<float> qg2((size_t)n * QIN2), k2((size_t)n * KVIN2), v2((size_t)n * KVIN2);
                std::vector<uint32_t> rs(n, 0u), rp(n);
                for (uint32_t i = 0; i < n; i++) {
                    rp[i] = p0 + i;
                    gen_row(FIX_A, rp[i], qg.data(), k.data(), v.data(), &idx[(size_t)i * IIN]);
                    memcpy(&qg2[(size_t)i * QIN2], &qg[(size_t)r * QIN2], QIN2 * 4);   /* heads 12r..12r+11, q + gate */
                    memcpy(&k2[(size_t)i * KVIN2], &k[(size_t)r * HD], HD * 4);
                    memcpy(&v2[(size_t)i * KVIN2], &v[(size_t)r * HD], HD * 4);
                }
                pulsar_gpu_tensor_write(d.qg, 0, qg2.data(), qg2.size() * 4);
                pulsar_gpu_tensor_write(d.k, 0, k2.data(), k2.size() * 4);
                pulsar_gpu_tensor_write(d.v, 0, v2.data(), v2.size() * 4);
                pulsar_gpu_tensor_write(d.idx, 0, idx.data(), idx.size() * 4);
                pulsar_gpu_tensor_fill_f32(d.slot_sc, 0.f, pulsar_gpu_tensor_bytes(d.slot_sc) / 4);
                pulsar_qsa_seq sd = s.desc();
                pulsar_qsa_layer L{(const uint16_t *)pulsar_gpu_tensor_device_ptr(d.qn), (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.kn),
                                   (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.iqn), (const uint16_t *)pulsar_gpu_tensor_device_ptr(d.ikn)};
                pulsar_qsa_io io{};
                io.tp_ranks = 2;
                io.qg = d.qg; io.k = d.k; io.v = d.v; io.idx = d.idx;
                io.out_e4m3 = pulsar_gpu_tensor_device_ptr(d.slot);
                io.out_scale = pulsar_gpu_tensor_device_ptr(d.slot_sc);
                io.out_sf_pitch = (int)(OUT2 / 32u);
                io.tap_out_f32 = d.tap;
                if (!pulsar_gpu_qsa_forward(&L, &sd, 1, rs.data(), rp.data(), n, &io, d.ws)) {
                    CHECK(false, "rank %u: pulsar_gpu_qsa_forward refused at %u", r, p0);
                    break;
                }
                std::vector<float> tap((size_t)n * OUT2);
                pulsar_gpu_tensor_read(d.tap, 0, tap.data(), tap.size() * 4);
                for (uint32_t i = 0; i < n; i++)
                    dout += memcmp(&tap[(size_t)i * OUT2], &A.outs[0][p0 + i].out[(size_t)r * OUT2], OUT2 * 4) != 0;
            }
            std::vector<uint8_t> kv((size_t)TA * REC2);
            pulsar_gpu_tensor_read(s.kv, 0, kv.data(), kv.size());
            for (uint32_t p = 0; p < TA; p++) {
                const uint8_t *mine = &kv[(size_t)p * REC2], *full = &A.kv[0][(size_t)p * REC];
                dkv += memcmp(mine, full + r * HD, HD) != 0;                              /* K data */
                dkv += memcmp(mine + HD, full + (NKV + r) * HD, HD) != 0;                 /* V data */
                dkv += memcmp(mine + 2 * HD, full + 2 * NKV * HD + r * (HD / 32), HD / 32) != 0;            /* K scales */
                dkv += memcmp(mine + 2 * HD + HD / 32, full + 2 * NKV * HD + (NKV + r) * (HD / 32), HD / 32) != 0;  /* V scales */
            }
            CHECK(dout == 0 && dkv == 0, "rank %u: %llu of %u rows' gated output and %llu K/V record parts differ from the full layer's",
                  r, (unsigned long long)dout, TA, (unsigned long long)dkv);
            seq_free(s);
        }
    }

    seq_free(SA[0]);
    seq_free(ST[0]);
    printf("\nqsa_attn_gate: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

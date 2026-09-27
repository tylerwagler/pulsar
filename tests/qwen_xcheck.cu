/* Qwen S4 cross-check tool (L251): runs the production router / MoE block / GR /
 * PLE launchers on REAL weights and captured activations handed over as files,
 * so tools/qwen/s4_xcheck.py can compare them with transformers' own modules
 * (and with exllamav3's reconstruct for the EXL3 weights).  Not a gate: it needs
 * the files the driver writes.  Where it helps the comparison it also writes the
 * double host reference (tests/qwen_ref.h) for the first rows.
 *
 *   qwen_xcheck router DIR N
 *       x.bin bf16 [N][2560], wr.bin bf16 [512][2560], wsg.bin bf16 [2560]
 *       -> sel.bin i32 [N][10], wts.bin f32 [N][10], sgate.bin f32 [N], logits.bin f32 [N][513]
 *   qwen_xcheck moe DIR N k2_gate_up k2_down k2_sg k2_su k2_sd
 *       x.bin, wr.bin, wsg.bin, experts.idx (u32 n, u32 ids[n]), exp_gate.bin / exp_up.bin /
 *       exp_down.bin (n slices each), shared_gate.bin / shared_up.bin / shared_down.bin
 *       -> out.bin f32 [N][2560], sel.bin, wts.bin, sgate.bin, out_m1.bin (each row alone)
 *   qwen_xcheck gr DIR N
 *       streams.bin bf16 [N][10240], norm.bin bf16 [10240], down.bin / up.bin (MXFP8_LT:
 *       codes then the swizzled E8M0 slab), inject.bin bf16 [4][10240] (absent = the mixer)
 *       -> x.bin bf16 [N][2560], inj.bin f32 [N][4], ref_x.bin f64 [min(N,16)][2560], ref_inj.bin
 *   qwen_xcheck ple DIR N k2_key k2_value
 *       emb.bin bf16 [N][2560], streams.bin bf16 [N][10240], seqs.bin (i32 n_seq, i32 len[n_seq];
 *       rows consecutive per sequence, every sequence starts fresh), key.bin / value.bin (EXL3
 *       slices), norm_key.bin / norm_query.bin / norm_conv.bin bf16 [10240], conv.bin bf16 [10240][4]
 *       -> out.bin bf16 [N][10240], ref_out.bin bf16 [min(len0,12)][10240] (sequence 0) */
#include "../src/pulsar_gpu.h"
#include "qwen_ref.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace qref;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::string g_dir;
static bool exists(const char *name) { struct stat st; return stat((g_dir + "/" + name).c_str(), &st) == 0; }
template <typename T> static std::vector<T> slurp(const char *name, size_t n) {
    const std::string path = g_dir + "/" + name;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::vector<T> v(n);
    const size_t got = fread(v.data(), sizeof(T), n, f);
    const int extra = fgetc(f);
    fclose(f);
    if (got != n || extra != EOF) { fprintf(stderr, "%s: expected exactly %zu x %zu bytes\n", path.c_str(), n, sizeof(T)); exit(2); }
    return v;
}
static void spill(const char *name, const void *p, size_t n) {
    const std::string path = g_dir + "/" + name;
    FILE *f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    fclose(f);
}
template <typename T> static T *up(const std::vector<T> &v) {
    T *d = nullptr;
    CK(cudaMalloc((void **)&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> down(const T *d, size_t n) {
    std::vector<T> v(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void *dalloc(size_t n) { void *p = nullptr; CK(cudaMalloc(&p, n + 16)); CK(cudaMemset(p, 0, n + 16)); return p; }
static size_t slice_bytes(int in, int out, int k2) { uint64_t t = 0, s = 0; exl3t_layout(in, out, k2, &t, &s); return s; }
static linear load_linear(const char *name, int in, int out, int k2) {
    linear l; l.in = in; l.out = out; l.k2 = k2;
    l.bytes = slurp<uint8_t>(name, slice_bytes(in, out, k2));
    return l;
}
static pulsar_qwen_linear dev_linear(const linear &l) {
    pulsar_qwen_linear d;
    d.w = up(l.bytes); d.k2 = l.k2; d.in = l.in; d.out = l.out;
    return d;
}
/* the block input's A8 slot, encoded from its bf16 rows the way the GR read emits it */
static pulsar_qwen_slot slot_of(const std::vector<uint16_t> &x, int N, int n) {
    const int kbp = pulsar_mx_kbp(n);
    std::vector<uint8_t> q((size_t)N * n), sf(pulsar_mx_sf_slab_bytes(N, kbp), 0);
    std::vector<double> v(n);
    for (int r = 0; r < N; r++) {
        for (int k = 0; k < n; k++) v[k] = bf(x[(size_t)r * n + k]);
        mx_encode_row(v.data(), n, r, kbp, q.data() + (size_t)r * n, sf.data(), nullptr);
    }
    return {up(q), up(sf), kbp};
}

static int do_router(int N) {
    const auto x = slurp<uint16_t>("x.bin", (size_t)N * H), wr = slurp<uint16_t>("wr.bin", (size_t)E * H),
               wsg = slurp<uint16_t>("wsg.bin", H);
    uint16_t *dwr = up(wr), *dwsg = up(wsg);
    const int C = 8192;
    std::vector<int32_t> sel((size_t)N * TOPK);
    std::vector<float> wts((size_t)N * TOPK), sg(N), lg((size_t)N * (E + 1));
    uint16_t *dx = (uint16_t *)dalloc((size_t)C * H * 2);
    float *dl = (float *)dalloc((size_t)C * (E + 1) * 4), *dw = (float *)dalloc((size_t)C * TOPK * 4), *ds = (float *)dalloc(C * 4);
    int32_t *dsel = (int32_t *)dalloc((size_t)C * TOPK * 4);
    for (int r0 = 0; r0 < N; r0 += C) {
        const int n = std::min(C, N - r0);
        CK(cudaMemcpy(dx, &x[(size_t)r0 * H], (size_t)n * H * 2, cudaMemcpyHostToDevice));
        if (pulsar_qwen_router_launch(dx, dwr, dwsg, n, H, E, TOPK, dl, dsel, dw, ds, 0)) return 1;
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&sel[(size_t)r0 * TOPK], dsel, (size_t)n * TOPK * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&wts[(size_t)r0 * TOPK], dw, (size_t)n * TOPK * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&sg[r0], ds, (size_t)n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&lg[(size_t)r0 * (E + 1)], dl, (size_t)n * (E + 1) * 4, cudaMemcpyDeviceToHost));
    }
    spill("sel.bin", sel.data(), sel.size() * 4);
    spill("wts.bin", wts.data(), wts.size() * 4);
    spill("sgate.bin", sg.data(), sg.size() * 4);
    spill("logits.bin", lg.data(), lg.size() * 4);
    printf("router: %d rows\n", N);
    return 0;
}

template <typename T> static std::vector<T> slurp_all(const char *name) {
    const std::string path = g_dir + "/" + name;
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || st.st_size % sizeof(T)) { fprintf(stderr, "cannot size %s\n", path.c_str()); exit(2); }
    return slurp<T>(name, (size_t)st.st_size / sizeof(T));
}

static int do_moe(int N, int k2gu, int k2d, int k2sg, int k2su, int k2sd) {
    const auto x = slurp<uint16_t>("x.bin", (size_t)N * H), wr = slurp<uint16_t>("wr.bin", (size_t)E * H),
               wsg = slurp<uint16_t>("wsg.bin", H);
    const auto idx = slurp_all<uint32_t>("experts.idx");
    const uint32_t n = idx.at(0);
    if (idx.size() != 1 + (size_t)n) { fprintf(stderr, "experts.idx: count %u vs %zu ids\n", n, idx.size() - 1); return 2; }
    const size_t sgu = slice_bytes(H, MID, k2gu), sd = slice_bytes(MID, H, k2d);
    const auto eg = slurp<uint8_t>("exp_gate.bin", n * sgu), eu = slurp<uint8_t>("exp_up.bin", n * sgu),
               ed = slurp<uint8_t>("exp_down.bin", n * sd);
    uint8_t *deg = up(eg), *deu = up(eu), *ded = up(ed);
    /* unquantized experts point at a poisoned slot: the tool refuses a routing that reaches one */
    std::vector<int> slot_of_expert(E, -1);
    for (uint32_t i = 0; i < n; i++) slot_of_expert.at(idx[1 + i]) = (int)i;
    uint64_t tgu = 0, sgu2 = 0, td = 0, sd2 = 0;
    exl3t_layout(H, MID, k2gu, &tgu, &sgu2);
    exl3t_layout(MID, H, k2d, &td, &sd2);
    std::vector<const void *> tg(2 * E), tu(2 * E), tdn(2 * E);
    for (int e = 0; e < E; e++) {
        const int s = slot_of_expert[e] < 0 ? 0 : slot_of_expert[e];
        tg[2 * e] = deg + (size_t)s * sgu;  tg[2 * e + 1] = deg + (size_t)s * sgu + tgu;
        tu[2 * e] = deu + (size_t)s * sgu;  tu[2 * e + 1] = deu + (size_t)s * sgu + tgu;
        tdn[2 * e] = ded + (size_t)s * sd;  tdn[2 * e + 1] = ded + (size_t)s * sd + td;
    }
    pulsar_qwen_moe_weights w;
    w.router_w = up(wr);
    w.shared_gate_w = up(wsg);
    w.gate_table = (const void *const *)up(tg);
    w.up_table = (const void *const *)up(tu);
    w.down_table = (const void *const *)up(tdn);
    w.k2_gate_up = k2gu; w.k2_down = k2d;
    const linear shg = load_linear("shared_gate.bin", H, PULSAR_QWEN_SHARED_MID, k2sg),
                 shu = load_linear("shared_up.bin", H, PULSAR_QWEN_SHARED_MID, k2su),
                 shd = load_linear("shared_down.bin", PULSAR_QWEN_SHARED_MID, H, k2sd);
    w.shared_gate = dev_linear(shg); w.shared_up = dev_linear(shu); w.shared_down = dev_linear(shd);
    uint16_t *dx = up(x);
    const pulsar_qwen_slot xs = slot_of(x, N, H);
    /* the routing first, so a row that reaches an unquantized expert is refused, not run */
    float *lg = (float *)dalloc((size_t)N * (E + 1) * 4), *rw = (float *)dalloc((size_t)N * TOPK * 4), *rs = (float *)dalloc(N * 4);
    int32_t *sel = (int32_t *)dalloc((size_t)N * TOPK * 4);
    if (pulsar_qwen_router_launch(dx, w.router_w, w.shared_gate_w, N, H, E, TOPK, lg, sel, rw, rs, 0)) return 1;
    const auto Sel = down(sel, (size_t)N * TOPK);
    for (size_t i = 0; i < Sel.size(); i++)
        if (slot_of_expert.at(Sel[i]) < 0) { fprintf(stderr, "row %zu routes to expert %d, which the driver did not quantize\n", i / TOPK, Sel[i]); return 3; }
    spill("sel.bin", Sel.data(), Sel.size() * 4);
    spill("wts.bin", down(rw, (size_t)N * TOPK).data(), (size_t)N * TOPK * 4);
    spill("sgate.bin", down(rs, N).data(), (size_t)N * 4);
    float *out = (float *)dalloc((size_t)N * H * 4);
    uint32_t *nf = (uint32_t *)dalloc(4);
    const size_t wsb = pulsar_qwen_moe_workspace_bytes(&w, N);
    void *ws = dalloc(wsb);
    if (pulsar_qwen_moe_launch(&w, dx, &xs, N, out, ws, wsb, nf, 0x7351u, 0)) return 1;
    const auto O = down(out, (size_t)N * H);
    spill("out.bin", O.data(), O.size() * 4);
    /* every row alone (T = 1): the decode width */
    std::vector<float> O1((size_t)N * H);
    for (int r = 0; r < N; r++) {
        const std::vector<uint16_t> xr(x.begin() + (size_t)r * H, x.begin() + (size_t)(r + 1) * H);
        const pulsar_qwen_slot s1 = slot_of(xr, 1, H);
        if (pulsar_qwen_moe_launch(&w, dx + (size_t)r * H, &s1, 1, out, ws, wsb, nf, 0x7351u, 0)) return 1;
        CK(cudaMemcpy(&O1[(size_t)r * H], out, H * 4, cudaMemcpyDeviceToHost));
        cudaFree(s1.q); cudaFree(s1.sf);
    }
    spill("out_m1.bin", O1.data(), O1.size() * 4);
    const auto NF = down(nf, 1);
    printf("moe: %d rows, %u quantized experts, M=1 rows %s, non-finite flag 0x%x\n", N, n,
           memcmp(O1.data(), O.data(), O.size() * 4) ? "DIFFER" : "bit-identical", NF[0]);
    return NF[0] != 0;
}

static mx8 load_mx8(const char *name, int out, int in) {
    mx8 m;
    m.out = out; m.in = in;
    const size_t nq = (size_t)out * in, nsf = pulsar_mx_sf_slab_bytes(out, pulsar_mx_kbp(in));
    const auto b = slurp<uint8_t>(name, nq + nsf);
    m.q.assign(b.begin(), b.begin() + nq);
    m.sf.assign(b.begin() + nq, b.end());
    return m;
}

static int do_gr(int N) {
    const auto st = slurp<uint16_t>("streams.bin", (size_t)N * HC), norm = slurp<uint16_t>("norm.bin", HC);
    const mx8 dn = load_mx8("down.bin", R, HC), upw = load_mx8("up.bin", HC, R);
    const bool site = exists("inject.bin");
    std::vector<uint16_t> inj_w;
    if (site) inj_w = slurp<uint16_t>("inject.bin", (size_t)S * HC);
    pulsar_qwen_gr_weights w;
    w.norm_w = up(norm);
    w.down = {up(dn.q), up(dn.sf), R, HC};
    w.up = {up(upw.q), up(upw.sf), HC, R};
    w.inject = site ? up(inj_w) : nullptr;
    uint16_t *dst = up(st), *xo = (uint16_t *)dalloc((size_t)N * H * 2);
    pulsar_qwen_slot xs = {(uint8_t *)dalloc((size_t)N * H), (uint8_t *)dalloc(pulsar_mx_sf_slab_bytes(N, pulsar_mx_kbp(H))), pulsar_mx_kbp(H)};
    float *inj = (float *)dalloc((size_t)N * S * 4);
    const size_t wsb = pulsar_qwen_gr_workspace_bytes(N);
    void *ws = dalloc(wsb);
    if (pulsar_qwen_gr_read_launch(&w, dst, N, xo, &xs, site ? inj : nullptr, ws, wsb, 0)) return 1;
    const auto X = down(xo, (size_t)N * H);
    const auto I = down(inj, (size_t)N * S);
    spill("x.bin", X.data(), X.size() * 2);
    spill("inj.bin", I.data(), I.size() * 4);
    const int nr = std::min(N, 16);
    std::vector<double> rx((size_t)nr * H), ri((size_t)nr * S);
    for (int t = 0; t < nr; t++) {
        const gr_out o = gr_read(&st[(size_t)t * HC], norm.data(), dn, upw, site ? inj_w.data() : nullptr);
        std::copy(o.x.begin(), o.x.end(), rx.begin() + (size_t)t * H);
        for (int j = 0; j < S; j++) ri[(size_t)t * S + j] = o.inj[j];
    }
    spill("ref_x.bin", rx.data(), rx.size() * 8);
    spill("ref_inj.bin", ri.data(), ri.size() * 8);
    printf("gr: %d rows (%s), host reference for %d\n", N, site ? "site" : "mixer", nr);
    return 0;
}

static int do_ple(int N, int k2k, int k2v) {
    const auto emb = slurp<uint16_t>("emb.bin", (size_t)N * H);
    auto st = slurp<uint16_t>("streams.bin", (size_t)N * HC);
    const auto seqs = slurp_all<int32_t>("seqs.bin");
    const int n_seq = seqs.at(0);
    if ((int)seqs.size() != 1 + n_seq) { fprintf(stderr, "seqs.bin: bad count\n"); return 2; }
    const linear key = load_linear("key.bin", H, HC, k2k), value = load_linear("value.bin", H, H, k2v);
    const auto nk = slurp<uint16_t>("norm_key.bin", HC), nq = slurp<uint16_t>("norm_query.bin", HC),
               nc = slurp<uint16_t>("norm_conv.bin", HC), cw = slurp<uint16_t>("conv.bin", (size_t)HC * 4);
    pulsar_qwen_ple_weights w;
    w.key_proj = dev_linear(key); w.value_proj = dev_linear(value);
    w.norm_key = up(nk); w.norm_query = up(nq); w.norm_conv = up(nc); w.conv_w = up(cw);
    std::vector<int32_t> rs(N), rj(N), sf(n_seq), sr(n_seq);
    int r = 0;
    for (int q = 0; q < n_seq; q++) {
        sf[q] = r; sr[q] = seqs[1 + q];
        for (int j = 0; j < seqs[1 + q]; j++, r++) { if (r >= N) { fprintf(stderr, "seqs exceed N\n"); return 2; } rs[r] = q; rj[r] = j; }
    }
    if (r != N) { fprintf(stderr, "seqs cover %d of %d rows\n", r, N); return 2; }
    pulsar_qwen_rows pr = {up(rs), up(rj), up(sf), up(sr), n_seq};
    float *state = (float *)dalloc((size_t)n_seq * PULSAR_QWEN_PLE_STATE * HC * 4);
    uint16_t *de = up(emb), *ds = up(st);
    const size_t wsb = pulsar_qwen_ple_workspace_bytes(&w, N);
    void *ws = dalloc(wsb);
    if (pulsar_qwen_ple_launch(&w, de, ds, N, &pr, state, ws, wsb, 0)) return 1;
    const auto O = down(ds, (size_t)N * HC);
    spill("out.bin", O.data(), O.size() * 2);
    linear k2 = key, v2 = value;
    k2.dequant(); v2.dequant();
    const int nr = std::min(seqs[1], 12);
    std::vector<std::vector<double>> hist;
    for (int t = 0; t < nr; t++)
        ple_token(&emb[(size_t)t * H], k2, v2, nk.data(), nq.data(), nc.data(), cw.data(), &st[(size_t)t * HC], hist);
    spill("ref_out.bin", st.data(), (size_t)nr * HC * 2);
    printf("ple: %d rows in %d sequences, host reference for %d rows of sequence 0\n", N, n_seq, nr);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s router|moe|gr|ple DIR N [k2...]\n", argv[0]); return 2; }
    const std::string mode = argv[1];
    g_dir = argv[2];
    const int N = atoi(argv[3]);
    if (!pulsar_gpu_init()) { fprintf(stderr, "no GPU\n"); return 2; }
    if (mode == "router") return do_router(N);
    if (mode == "moe" && argc == 9) return do_moe(N, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]));
    if (mode == "gr") return do_gr(N);
    if (mode == "ple" && argc == 6) return do_ple(N, atoi(argv[4]), atoi(argv[5]));
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

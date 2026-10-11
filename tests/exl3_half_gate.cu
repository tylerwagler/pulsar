/* EXL3 expert-half gate (L269 W1): a rank's half of every EXL3 routed expert is a BYTE-EXACT piece of the
 * whole, and the routed arm run on the two halves IS the whole run, split where tensor parallelism splits it.
 *
 * WHY THIS EXISTS
 * ---------------
 * Under DeepSeek TP every rank holds half of every routed expert -- gate / up by output columns, down by input
 * rows, the same intermediate range on all three (pulsar_tp_owned_range) -- and runs every selected expert over
 * that half; the FFN exchange sums the two ranks' outputs.  For CUTLASS MXFP4 that is 4g-2 (L241); this is the
 * EXL3 twin, which is what lets V4.1 (all-EXL3 experts) fit the pair.  A two-Spark run would show a wrong cut
 * only as "the logits differ"; here it is checked on one GPU, through the production pieces: the builder
 * (pulsar_gpu_register_exl3_expert_half, reading the stack through the model fd like the engine does) and the
 * routed entry (pulsar_gpu_routed_moe_batch_tensor with expert_split, the call moe.cpp's front door makes).
 *
 * WHAT IT PROVES (the V4.1 routed shape: 5120 -> 2304 -> 5120, halves of 1152 = 9 x 128; 16 experts, top-6)
 *   1. BYTES, every rate the routed arms read (gate / up K2 K2.5 K3 K4 K6; down K2 K2.5 K3 K4 K5 K6): both
 *      ranks' halves, as the production builder made them, equal an INDEPENDENT statement of the cut -- every
 *      16x16 tile of the half is the whole's tile at the shifted index, suh / svh the whole's ranges -- so the
 *      two halves partition every expert, and each half is exl3_expert_layout of the half shape.  The host cut
 *      (exl3_expert_cut_apply, what tp_slice.cpp's rows reader uses) equals the device bytes too.
 *   2. CHAIN, decode rows (1, 5 and 16 tokens: the trellis GEMV) and prefill rows (33 and 300 tokens: the L287
 *      tensor-core GEMM): the whole stack and each rank's half through the routed entry, then
 *        - the gate and up z of each half == the whole's columns [lo, hi), bit for bit;
 *        - the folded mid's E4M3 slot of each half == the whole's columns: every value byte and every per-32
 *          scale (the 32-groups and the 128 Hadamard blocks align with the cut), bit for bit;
 *        - out_0 + out_1 (what the FFN exchange computes) vs the whole: NOT bit-exact by construction -- the
 *          down's K reduction is split at the cut and the two f32 partials are added in a new order (rotation
 *          and svh are linear, applied per partial) -- so it is bounded: relative Frobenius <= 1e-5 and the
 *          worst element <= 1e-4 of the output's max |value|.
 *   3. REFUSALS: a misaligned cut and an out-of-range cut refused by the builder and by the host cut; the
 *      routed entry under expert_split refuses mixed formats (EXL3 with CUTLASS MXFP4) and IQ2 stacks, and a
 *      half width off the 128 granule.
 *
 * usage: tests/exl3_half_gate [SCRATCH_DIR]   (default "."; the whole stacks go to an unlinked file there,
 * read back through the model fd like a checkpoint shard) */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/pulsar_cuda_qwen.h"   /* pulsar_gpu_weight_range_ptr */
#include "engine/exl3_trellis.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t g_rng = 0x269A1u;
static uint32_t rnd(void) { uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return g_rng = x; }
static float rndf(void) { return (float)(rnd() >> 8) * (1.0f / 16777216.0f); }

static const int H = 5120, MID = 2304, E = 16, TOPK = 6, NR = 2;
static const uint64_t HALF = MID / NR;

/* one stack's bytes on the host: E experts of [trellis | suh | svh], random trellis words (any word is a valid
 * tile), suh ~ +-0.025, svh ~ +-1 (the scales real quants carry) */
static std::vector<uint8_t> make_stack(int k, int n, int k2, uint64_t *stride, uint64_t *trellis) {
    uint64_t sc = 0;
    if (!exl3_expert_layout(k, n, k2, trellis, &sc, stride)) { printf("layout refused\n"); exit(1); }
    std::vector<uint8_t> b(*stride * E);
    for (int e = 0; e < E; e++) {
        uint8_t *p = b.data() + e * *stride;
        for (uint64_t i = 0; i < *trellis; i += 4) { const uint32_t r = rnd(); memcpy(p + i, &r, 4); }
        uint16_t *s = (uint16_t *)(p + *trellis);
        for (int i = 0; i < k + n; i++) {
            const float m = (0.5f + rndf()) * (i < k ? 0.025f : 1.0f);
            s[i] = exl3_f32_to_f16((rnd() & 1u) ? -m : m);
        }
    }
    return b;
}

/* The independent statement of a half (not exl3_expert_cut): tile (kt, nt) of the half is the whole's tile
 * (kt, nt + lo/16) for an OUT half, (kt + lo/16, nt) for an IN half; suh / svh are the whole's ranges. */
static bool half_matches(const uint8_t *whole, const uint8_t *half, int k, int n, int k2, bool in_axis, uint64_t lo,
                         uint64_t hi) {
    const uint64_t tile = (uint64_t)exl3_words_per_tile(k2) * 2u, w = hi - lo;
    const uint64_t hk = in_axis ? w : k, hn = in_axis ? n : w;
    const uint64_t wtr = (uint64_t)(k / 16) * (n / 16) * tile, htr = hk / 16 * (hn / 16) * tile;
    for (uint64_t kt = 0; kt < hk / 16; kt++)
        for (uint64_t nt = 0; nt < hn / 16; nt++) {
            const uint64_t wk = in_axis ? kt + lo / 16 : kt, wn = in_axis ? nt : nt + lo / 16;
            if (memcmp(half + (kt * (hn / 16) + nt) * tile, whole + (wk * (n / 16) + wn) * tile, tile)) return false;
        }
    const uint8_t *wsuh = whole + wtr, *wsvh = whole + wtr + k * 2u, *hsuh = half + htr, *hsvh = half + htr + hk * 2u;
    if (in_axis) return !memcmp(hsuh, wsuh + lo * 2u, w * 2u) && !memcmp(hsvh, wsvh, n * 2u);
    return !memcmp(hsuh, wsuh, k * 2u) && !memcmp(hsvh, wsvh + lo * 2u, w * 2u);
}

struct run_out {
    std::vector<float> gz, uz, out;
    std::vector<uint8_t> mq, msf;   /* the folded mid's E4M3 slot: [pairs][mid] values, per (pair, 32-group) scale */
};

/* one call of the production routed entry; returns false on a refusal */
static bool routed(const void *map, uint64_t size, const uint64_t off[3], int gk2, int dk2, uint64_t mid, int split,
                   pulsar_gpu_tensor *x, pulsar_gpu_tensor *sel, pulsar_gpu_tensor *wts, int n_tok, run_out *r) {
    const uint64_t pairs = (uint64_t)n_tok * TOPK;
    uint64_t gs = 0, gt = 0, ds = 0, dt = 0, sc = 0;
    if (!exl3_expert_layout(H, mid, gk2, &gt, &sc, &gs) || !exl3_expert_layout(mid, H, dk2, &dt, &sc, &ds)) return false;
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc((uint64_t)n_tok * H * 4);
    pulsar_gpu_tensor *up = pulsar_gpu_tensor_alloc(pairs * mid * 4), *md = pulsar_gpu_tensor_alloc(pairs * mid * 4);
    pulsar_gpu_tensor *down = pulsar_gpu_tensor_alloc(pairs * H * 4);
    if (!pulsar_gpu_mxfp8_act_cache_encode_f32(x, (uint64_t)n_tok, H)) { printf("x encode failed\n"); exit(1); }
    const uint32_t gtype = gk2 == 4 ? PULSAR_TENSOR_EXL3M_K2 : gk2 == 5 ? PULSAR_TENSOR_EXL3M_K2H : gk2 == 6 ? PULSAR_TENSOR_EXL3M_K3
                         : gk2 == 8 ? PULSAR_TENSOR_EXL3M_K4 : gk2 == 10 ? PULSAR_TENSOR_EXL3M_K5 : PULSAR_TENSOR_EXL3M_K6;
    const uint32_t dtype = dk2 == 4 ? PULSAR_TENSOR_EXL3M_K2 : dk2 == 5 ? PULSAR_TENSOR_EXL3M_K2H : dk2 == 6 ? PULSAR_TENSOR_EXL3M_K3
                         : dk2 == 8 ? PULSAR_TENSOR_EXL3M_K4 : dk2 == 10 ? PULSAR_TENSOR_EXL3M_K5 : PULSAR_TENSOR_EXL3M_K6;
    const int ok = pulsar_gpu_routed_moe_batch_tensor(out, up, md, down, map, size, off[0], off[1], off[2], gtype, dtype,
                                                      gs, gt, ds, dt, H, (uint32_t)mid, H, sel, wts, E, TOPK, 10.0f, x,
                                                      0, (uint32_t)n_tok, (uint32_t)split);
    bool good = ok != 0 && cudaDeviceSynchronize() == cudaSuccess;
    if (good && r) {
        r->gz.resize(pairs * mid); r->uz.resize(pairs * mid); r->out.resize((size_t)n_tok * H);
        good = pulsar_gpu_tensor_read(up, 0, r->gz.data(), pairs * mid * 4) &&
               pulsar_gpu_tensor_read(md, 0, r->uz.data(), pairs * mid * 4) &&
               pulsar_gpu_tensor_read(out, 0, r->out.data(), (uint64_t)n_tok * H * 4);
        const void *q = NULL, *s = NULL;
        int kbp = 0;
        good = good && pulsar_gpu_mxfp8_act_cache_get_e4m3(md, pairs, mid, &q, &s, &kbp) && q && s;
        if (good) {
            r->mq.resize(pairs * mid);
            std::vector<uint8_t> sf(pulsar_mx_sf_slab_bytes((int)pairs, kbp));
            good = cudaMemcpy(r->mq.data(), q, r->mq.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
                   cudaMemcpy(sf.data(), s, sf.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
            r->msf.resize(pairs * (mid / 32));
            for (uint64_t p = 0; p < pairs; p++)
                for (uint64_t g = 0; g < mid / 32; g++) r->msf[p * (mid / 32) + g] = sf[pulsar_mx_sfoff((int)p, (int)g, kbp)];
        }
    }
    pulsar_gpu_act_slot_drop(md);
    pulsar_gpu_tensor_free(out); pulsar_gpu_tensor_free(up); pulsar_gpu_tensor_free(md); pulsar_gpu_tensor_free(down);
    return good;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const std::string dir = argc > 1 ? argv[1] : ".";
    if (!pulsar_gpu_init()) { printf("exl3-half-gate: no GPU\n"); return 2; }
    printf("exl3-half-gate (L269 W1): V4.1 routed shape %d -> %d -> %d, %d experts, top-%d, %d ranks (halves of %llu)\n",
           H, MID, H, E, TOPK, NR, (unsigned long long)HALF);
    static const char key = 0;   /* the registry key the halves resolve under (the engine, in the engine) */
    uint64_t key_next = 1;
    /* (gate / up rate, down rate): every rate each kind reads (exl3_arm_has_rate PAIR / DOWN); the pair has no K5 */
    const int rates[][2] = {{4, 4}, {5, 5}, {6, 6}, {8, 8}, {8, 10}, {12, 12}};
    double worst_rel = 0, worst_max = 0;
    for (const auto &rk : rates) {
        const int gk2 = rk[0], dk2 = rk[1];
        printf("-- gate/up K=%g, down K=%g\n", gk2 / 2.0, dk2 / 2.0);
        uint64_t gs = 0, gt = 0, ds = 0, dt = 0;
        const std::vector<uint8_t> G = make_stack(H, MID, gk2, &gs, &gt), U = make_stack(H, MID, gk2, &gs, &gt),
                                   D = make_stack(MID, H, dk2, &ds, &dt);
        /* the stacks as a checkpoint shard: written, mapped, read back through the fd */
        const std::string path = dir + "/.exl3_half_gate." + std::to_string(getpid());
        const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) { printf("cannot create %s\n", path.c_str()); return 2; }
        unlink(path.c_str());
        const uint64_t off[3] = {0, G.size(), G.size() + U.size()}, fsize = off[2] + D.size();
        bool wrote = true;
        for (const std::vector<uint8_t> *v : {&G, &U, &D})
            wrote = wrote && write(fd, v->data(), v->size()) == (ssize_t)v->size();
        void *map = wrote ? mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        if (map == MAP_FAILED || !pulsar_gpu_set_model_fd_for_map(fd, map)) { printf("scratch map failed\n"); return 2; }
        /* (never unmapped: the range table keys on the mapping's address) */

        /* 1. BYTES: both ranks' halves of all three stacks, through the production builder */
        uint64_t hoff[NR][3];
        for (int r = 0; r < NR; r++) {
            const uint64_t lo = r * HALF, hi = lo + HALF;
            for (int s = 0; s < 3; s++) {
                const bool in_axis = s == 2;
                const int k = in_axis ? MID : H, n = in_axis ? H : MID, k2 = in_axis ? dk2 : gk2;
                const std::vector<uint8_t> &W = s == 0 ? G : s == 1 ? U : D;
                hoff[r][s] = (key_next++) << 32;
                const int built = pulsar_gpu_register_exl3_expert_half(&key, hoff[r][s], map, off[s], E, k, n, k2,
                                                                       in_axis, lo, hi);
                CHECK(built, "rank %d stack %d: the builder refused", r, s);
                exl3_cut c;
                if (!built || !exl3_expert_cut(k, n, k2, in_axis, lo, hi, 0, hi - lo, &c)) { g_fail = 1; continue; }
                const void *dev = pulsar_gpu_weight_range_ptr(&key, hoff[r][s], c.dst_stride * E, "exl3 half gate");
                std::vector<uint8_t> got(c.dst_stride * E), cut(c.dst_stride);
                CHECK(dev && cudaMemcpy(got.data(), dev, got.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                      "rank %d stack %d: the registered half does not resolve", r, s);
                int bad_ind = 0, bad_cut = 0;
                for (int e = 0; e < E; e++) {
                    const uint8_t *we = W.data() + e * c.src_stride, *he = got.data() + e * c.dst_stride;
                    bad_ind += !half_matches(we, he, k, n, k2, in_axis, lo, hi);
                    exl3_expert_cut_apply(&c, we, cut.data());
                    bad_cut += memcmp(cut.data(), he, c.dst_stride) != 0;
                }
                CHECK(bad_ind == 0 && bad_cut == 0, "rank %d %s [%llu,%llu): %d experts differ from the tile statement, "
                      "%d from the host cut", r, s == 0 ? "gate" : s == 1 ? "up" : "down", (unsigned long long)lo,
                      (unsigned long long)hi, bad_ind, bad_cut);
            }
        }
        printf("  1. bytes: %d ranks x gate / up (output columns) + down (input rows), %d experts each: %s\n", NR, E,
               g_fail ? "FAIL" : "every tile, suh and svh == the whole's, host cut == device");

        /* 2. CHAIN: decode rows (GEMV) and prefill rows (tensor-core GEMM) */
        for (const int n_tok : {1, 5, 16, 33, 300}) {
            const bool decode = n_tok <= 16;
            const uint64_t pairs = (uint64_t)n_tok * TOPK;
            std::vector<float> xh((size_t)n_tok * H), wh(pairs);
            std::vector<int32_t> sh(pairs);
            for (float &v : xh) v = (rndf() * 2.0f - 1.0f) * (0.25f + 4.0f * rndf());
            for (int t = 0; t < n_tok; t++)
                for (int k = 0; k < TOPK; k++) {
                    int e;
                    bool dup;
                    do { e = (int)(rnd() % E); dup = false; for (int q = 0; q < k; q++) dup |= sh[t * TOPK + q] == e; } while (dup);
                    sh[t * TOPK + k] = e;
                    wh[t * TOPK + k] = 0.05f + 0.3f * rndf();
                }
            pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc(xh.size() * 4), *sel = pulsar_gpu_tensor_alloc(pairs * 4),
                              *wts = pulsar_gpu_tensor_alloc(pairs * 4);
            pulsar_gpu_tensor_write(x, 0, xh.data(), xh.size() * 4);
            pulsar_gpu_tensor_write(sel, 0, sh.data(), pairs * 4);
            pulsar_gpu_tensor_write(wts, 0, wh.data(), pairs * 4);
            pulsar_gpu_matmul_set_batch_decode_rows(decode ? n_tok : 0);
            run_out whole, half[NR];
            bool ran = routed(map, fsize, off, gk2, dk2, MID, 0, x, sel, wts, n_tok, &whole);
            for (int r = 0; r < NR; r++)
                ran = ran && routed(&key, UINT64_MAX / 2u, hoff[r], gk2, dk2, HALF, 1, x, sel, wts, n_tok, &half[r]);
            pulsar_gpu_matmul_set_batch_decode_rows(0);
            CHECK(ran, "%d tokens: the routed entry refused", n_tok);
            if (ran) {
                uint64_t zbad = 0, qbad = 0, sbad = 0;
                for (int r = 0; r < NR; r++)
                    for (uint64_t p = 0; p < pairs; p++) {
                        const uint64_t w0 = p * MID + r * HALF, h0 = p * HALF;
                        zbad += memcmp(&whole.gz[w0], &half[r].gz[h0], HALF * 4) != 0;
                        zbad += memcmp(&whole.uz[w0], &half[r].uz[h0], HALF * 4) != 0;
                        qbad += memcmp(&whole.mq[w0], &half[r].mq[h0], HALF) != 0;
                        sbad += memcmp(&whole.msf[p * (MID / 32) + r * (HALF / 32)], &half[r].msf[p * (HALF / 32)],
                                       HALF / 32) != 0;
                    }
                double num = 0, den = 0, mx = 0, amax = 0;
                for (size_t i = 0; i < whole.out.size(); i++) {
                    const float sum = half[0].out[i] + half[1].out[i];   /* the exchange's f32 add, rank order */
                    const double d = (double)sum - (double)whole.out[i];
                    num += d * d;
                    den += (double)whole.out[i] * whole.out[i];
                    mx = fmax(mx, fabs(d));
                    amax = fmax(amax, fabs((double)whole.out[i]));
                }
                const double rel = den > 0 ? sqrt(num / den) : 0, rmax = amax > 0 ? mx / amax : 0;
                worst_rel = fmax(worst_rel, rel);
                worst_max = fmax(worst_max, rmax);
                const bool ok = zbad == 0 && qbad == 0 && sbad == 0 && rel <= 1e-5 && rmax <= 1e-4 && std::isfinite(rel);
                CHECK(ok, "%d tokens", n_tok);
                printf("  2. %-7s %3d tokens: gate/up z rows differing %llu, mid E4M3 value rows %llu, scale rows %llu "
                       "(of %llu x %d); out_0 + out_1 vs whole: rel Frob %.3g, max %.3g of max|out| -- %s\n",
                       decode ? "decode" : "prefill", n_tok, (unsigned long long)zbad, (unsigned long long)qbad,
                       (unsigned long long)sbad, (unsigned long long)pairs, NR, rel, rmax, ok ? "ok" : "FAIL");
            }
            pulsar_gpu_act_slot_drop(x);
            pulsar_gpu_tensor_free(x); pulsar_gpu_tensor_free(sel); pulsar_gpu_tensor_free(wts);
        }

        /* 3. REFUSALS (once, at the first rate) */
        if (gk2 == 4) {
            exl3_cut c;
            CHECK(!exl3_expert_cut(H, MID, gk2, false, 64, 64 + HALF, 0, HALF, &c), "host cut took a cut off the 128 block");
            CHECK(!exl3_expert_cut(MID, H, dk2, true, HALF, MID + 128, 0, HALF + 128, &c), "host cut took a range past the dim");
            CHECK(!pulsar_gpu_register_exl3_expert_half(&key, (key_next++) << 32, map, off[0], E, H, MID, gk2, 0, 64,
                                                        64 + HALF), "the builder took a cut off the 128 block");
            CHECK(!pulsar_gpu_register_exl3_expert_half(&key, (key_next++) << 32, map, off[2], E, MID, H, dk2, 1, HALF,
                                                        MID + 128), "the builder took a range past the dim");
            pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc(H * 4), *sel = pulsar_gpu_tensor_alloc(TOPK * 4),
                              *wts = pulsar_gpu_tensor_alloc(TOPK * 4), *o = pulsar_gpu_tensor_alloc(H * 4),
                              *u = pulsar_gpu_tensor_alloc(TOPK * MID * 4), *m = pulsar_gpu_tensor_alloc(TOPK * MID * 4),
                              *d = pulsar_gpu_tensor_alloc(TOPK * H * 4);
            const int32_t s6[TOPK] = {0, 1, 2, 3, 4, 5};
            const float w6[TOPK] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
            std::vector<float> x1(H, 0.5f);
            pulsar_gpu_tensor_write(x, 0, x1.data(), H * 4);
            pulsar_gpu_tensor_write(sel, 0, s6, sizeof(s6));
            pulsar_gpu_tensor_write(wts, 0, w6, sizeof(w6));
            pulsar_gpu_mxfp8_act_cache_encode_f32(x, 1, H);
            auto split_call = [&](uint32_t gtype, uint32_t dtype, uint32_t mid) {
                return pulsar_gpu_routed_moe_batch_tensor(o, u, m, d, &key, UINT64_MAX / 2u, hoff[0][0], hoff[0][1],
                                                          hoff[0][2], gtype, dtype, gs, gt, ds, dt, H, mid, H, sel, wts,
                                                          E, TOPK, 10.0f, x, 0, 1, 1);
            };
            pulsar_gpu_matmul_set_batch_decode_rows(1);
            CHECK(!split_call(PULSAR_TENSOR_EXL3M_K2, PULSAR_TENSOR_CUTLASS_MXFP4, HALF), "split took EXL3 gate / MXFP4 down");
            CHECK(!split_call(PULSAR_TENSOR_CUTLASS_MXFP4, PULSAR_TENSOR_EXL3M_K2, HALF), "split took MXFP4 gate / EXL3 down");
            CHECK(!split_call(PULSAR_TENSOR_IQ2_XXS_MMQ_K, PULSAR_TENSOR_IQ2_XXS_MMQ_K, HALF), "split took IQ2 stacks");
            CHECK(!split_call(PULSAR_TENSOR_EXL3M_K2, PULSAR_TENSOR_EXL3M_K2, HALF - 64), "split took a half off the 128 granule");
            pulsar_gpu_matmul_set_batch_decode_rows(0);
            (void)cudaGetLastError();
            printf("  3. refusals: misaligned / out-of-range cuts (host cut, builder); mixed EXL3 + MXFP4, IQ2 and an "
                   "off-granule half at the split entry: %s\n", g_fail ? "see FAIL lines" : "all refused");
            pulsar_gpu_act_slot_drop(x);
            for (pulsar_gpu_tensor *t : {x, sel, wts, o, u, m, d}) pulsar_gpu_tensor_free(t);
        }
    }
    printf("worst out_0 + out_1 vs whole over every rate and width: rel Frob %.3g, max %.3g of max|out| "
           "(bounds 1e-5 / 1e-4: f32 reassociation of the down's split K)\n", worst_rel, worst_max);
    printf(g_fail ? "EXL3-HALF GATE FAIL\n" : "EXL3-HALF GATE PASS\n");
    return g_fail ? 1 : 0;
}

/* L151-C STAGE 0 / L169 PROBE: can ONE tensor-core tile configuration be both
 * M-neutral and flat against the engine's dispatch on the <= 16-row dense step?
 *
 * MODEL-DEPENDENT (real MXFP8_LT / bf16 weights): `make cuda-fixed-tile-probe`.
 * For each dense shape and M in {1..16} it times, on the same f32 rows:
 *   engine   -- the engine's dispatch with the M rows declared DECODE rows
 *               (pulsar_gpu_matmul_set_batch_decode_rows(M): the M-independent
 *               GEMV/nt arms, the served verify-step shape; L167), reading the
 *               activation plane its producer emitted for exactly M rows
 *   variants -- the probe's CUTLASS GEMMs (fixed_tile_gemm_kernels.cu):
 *               MXFP8: the 128x128x128 block-scaled tile plain (S=1), split-K
 *               S in {2,4,8} and stream-K, fed the engine's own E4M3 bytes and
 *               the LT weight slabs; bf16: four fixed mma.sync tiles
 * and reports per shape:
 *   1. speed: us per call and the EFFECTIVE weight bandwidth (weight bytes /
 *      time) to read against the part's DRAM (~273 GB/s on GB10);
 *   2. neutrality: rows [0, M) of every variant's M-row call are BYTE-IDENTICAL
 *      to the same rows of its own 16-row call (Y/N per variant), and the
 *      scheduler's decomposition (effective splits, stream-K units, CTAs) is
 *      the same at M as at 16 (Y/N per variant);
 *   3. distance: max |S=1 - engine| and max |variant - S=1| -- split-K and
 *      stream-K reduce in a different (fixed) order than the single-accumulator
 *      tile, so they are different numbers; reported, not hidden;
 *   4. layout: how many weight scale bytes the CUTLASS SFB layout places at a
 *      different offset than the LT slab, summed over the rotated layers
 *      (0 = the LT slab is usable as-is).
 *
 * DRAM-honest timing (L169): the 40 launches ROTATE across different layers'
 * weights of the same shape (layer L, L+1, ... wrapping), R layers chosen so the
 * rotation footprint is at least twice the device's L2 -- a loop over one
 * layer's 4-9 MB weight reads it from L2 at 600 GB/s, above the DRAM the served
 * step actually streams from.  Neutrality and distance are measured on layer L
 * alone.  Activation packing (the producer's job in the engine) sits outside
 * the timed loop on both sides; each variant's launch pays its own barrier-
 * workspace reset (split-K/stream-K arrivals are counters).
 *
 * Host wall-clock around 40 launches + one sync per arm, as in the L151 sweep. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "fixed_tile_gemm_probe.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_us(void) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_rand(pulsar_gpu_tensor *t, uint64_t n_floats, uint32_t seed) {
    std::vector<float> h(n_floats);
    uint32_t s = seed * 2654435761u + 12345u;
    for (uint64_t i = 0; i < n_floats; i++) {
        s = s * 1664525u + 1013904223u;
        h[i] = ((float)(s >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
    pulsar_gpu_tensor_write(t, 0, h.data(), n_floats * sizeof(float));
}

typedef struct {
    const pulsar_engine *e;
    pulsar_gpu_tensor *x, *out;
    uint64_t in_dim, out_dim;
    bool is_bf16;
    std::vector<const pulsar_tensor *> w;   /* rotation set; w[0] = layer L, the neutrality/distance layer */
    std::vector<ft_weight *> ftw;  ft_act *fta;   /* MXFP8 arm, one weight per rotated layer */
    std::vector<fb_weight *> fbw;  fb_act *fba;   /* bf16 arm */
    std::vector<ft_plan *> ftp;                   /* current (variant, M): one plan per rotated layer */
    std::vector<fb_plan *> fbp;
} ctx_t;

/* launch i of a loop reads layer i % R */
static bool engine_launch(void *v, uint32_t M, int i) {
    ctx_t *c = (ctx_t *)v;
    const pulsar_tensor *w = c->w[(size_t)i % c->w.size()];
    if (c->is_bf16)
        return pulsar_gpu_matmul_bf16_tensor(c->out, c->e->model.map, c->e->model.size,
                                             w->abs_offset, c->in_dim, c->out_dim, c->x, M) != 0;
    return pulsar_gpu_matmul_mxfp8_tensor(c->out, c->e->model.map, c->e->model.size,
                                          w->abs_offset, c->in_dim, c->out_dim, c->x, M) != 0;
}
static bool ft_launch(void *v, uint32_t, int i) {
    ctx_t *c = (ctx_t *)v;
    if (c->is_bf16) return fb_plan_run(c->fbp[(size_t)i % c->fbp.size()]) == 0;
    return ft_plan_run(c->ftp[(size_t)i % c->ftp.size()]) == 0;
}

typedef bool (*launch_fn)(void *, uint32_t, int);
static double time_launches(launch_fn fn, void *ctx, uint32_t M, int reps) {
    for (int i = 0; i < 3; i++) if (!fn(ctx, M, i)) return -1.0;
    if (!pulsar_gpu_end_commands() || ft_sync() != 0) return -1.0;
    const double t0 = now_us();
    for (int i = 0; i < reps; i++) if (!fn(ctx, M, i)) return -1.0;
    if (!pulsar_gpu_end_commands() || ft_sync() != 0) return -1.0;
    return (now_us() - t0) / reps;
}

static bool read_out(const pulsar_gpu_tensor *t, std::vector<float> &h, uint64_t n) {
    h.resize(n);
    return pulsar_gpu_tensor_read(t, 0, h.data(), n * sizeof(float)) != 0;
}

static double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b, uint64_t n) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) { const double d = fabs((double)a[i] - (double)b[i]); if (d > m) m = d; }
    return m;
}

static uint64_t count_byte_mismatch(const std::vector<float> &a, const std::vector<float> &b, uint64_t n) {
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], sizeof(float)) != 0) bad++;
    return bad;
}

static bool same_decomp(const ft_decomp &a, const ft_decomp &b) {
    return a.splits == b.splits && a.sk_units == b.sk_units && a.sk_tiles == b.sk_tiles &&
           a.big_units == b.big_units && a.ctas == b.ctas;
}

/* Plans for (variant, M) on every rotated layer.  Returns 0, or the first
 * non-zero plan_make code (1 = CUTLASS can_implement refused). */
static int make_plans(ctx_t *c, int v, uint32_t M) {
    const size_t R = c->w.size();
    if (c->is_bf16) {
        c->fbp.assign(R, nullptr);
        for (size_t r = 0; r < R; r++) {
            const int rc = fb_plan_make(c->fbw[r], c->fba, v, (int)M, c->out, &c->fbp[r]);
            if (rc) return rc;
        }
    } else {
        c->ftp.assign(R, nullptr);
        for (size_t r = 0; r < R; r++) {
            const int rc = ft_plan_make(c->ftw[r], c->fta, v, (int)M, c->out, &c->ftp[r]);
            if (rc) return rc;
        }
    }
    return 0;
}
static void release_plans(ctx_t *c) {
    for (ft_plan *p : c->ftp) ft_plan_release(p);
    for (fb_plan *p : c->fbp) fb_plan_release(p);
    c->ftp.clear(); c->fbp.clear();
}

/* The producer's activation for the engine column: exactly the M rows the
 * consumer will read (a bf16 plane or an E4M3 slot), the way the fused norms
 * emit per step.  An M-row plane is row-local, so it is byte-identical to the
 * first M rows of a wider one; the encode is outside the timed loop. */
static bool encode_for_engine(ctx_t *c, uint32_t M) {
    const int ok = c->is_bf16 ? pulsar_gpu_bf16_act_encode_f32(c->x, M, c->in_dim)
                              : pulsar_gpu_mxfp8_act_cache_encode_f32(c->x, M, c->in_dim);
    return ok && pulsar_gpu_end_commands();
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [LAYER]\n", argv[0]); return 2; }
    const uint32_t il = argc > 2 ? (uint32_t)atoi(argv[2]) : 4u;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (il >= PULSAR_N_LAYER) { fprintf(stderr, "layer %u past n_layer %u\n", il, (unsigned)PULSAR_N_LAYER); return 1; }
    const pulsar_layer_weights *L = &e->weights.layer[il];
    if (!L->attn_q_b || !L->attn_output_b || !L->ffn_gate_shexp || !L->attn_q_a || !L->ffn_gate_inp || !e->weights.output) {
        fprintf(stderr, "layer %u lacks a needed tensor\n", il);
        return 1;
    }
    int sm_count = 0;
    unsigned long long l2_bytes = 0;
    if (ft_device_info(&sm_count, &l2_bytes)) return 1;
    typedef pulsar_tensor *pulsar_layer_weights::*layer_field;
    struct { const char *name; layer_field f; const pulsar_tensor *single; } shapes[] = {
        {"attn_q_b", &pulsar_layer_weights::attn_q_b, NULL}, {"attn_output_b", &pulsar_layer_weights::attn_output_b, NULL},
        {"ffn_gate_shexp", &pulsar_layer_weights::ffn_gate_shexp, NULL}, {"attn_q_a", &pulsar_layer_weights::attn_q_a, NULL},
        {"router (bf16)", &pulsar_layer_weights::ffn_gate_inp, NULL}, {"output head (bf16)", NULL, e->weights.output},
    };
    const uint32_t Ms[] = {1, 2, 4, 5, 8, 9, 12, 16};
    const int NM = (int)(sizeof(Ms) / sizeof(Ms[0]));
    const uint32_t MMAX = 16;
    const int reps = 40;
    int rc = 0;
    printf("layer %u (+ rotation); device: %d SMs, L2 %.1f MB; us per call (%d launches + 1 sync), (GB/s) = weight bytes / time.\n"
           "engine = the dispatch with M rows declared decode (GEMV/nt arms, <= %u rows), producer plane for exactly M rows;\n"
           "MXFP8 variants = fixed CUTLASS 128x128x128 block-scaled tile: S=1 plain, S=2/4/8 split-K (deterministic, "
           "fixed slices), streamK; bf16 variants = fixed mma.sync tiles.\n\n",
           il, sm_count, l2_bytes / 1e6, reps, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const char *name = shapes[si].name;
        const pulsar_tensor *base = shapes[si].single ? shapes[si].single : e->weights.layer[il].*shapes[si].f;
        const bool is_bf16 = base->type == PULSAR_TENSOR_BF16;
        if (!is_bf16 && base->type != PULSAR_TENSOR_MXFP8_LT) { printf("%s: type %u is neither MXFP8_LT nor BF16, skipped\n\n", name, (unsigned)base->type); continue; }
        ctx_t c;
        c.e = e; c.x = c.out = NULL; c.in_dim = base->dim[0]; c.out_dim = base->dim[1]; c.is_bf16 = is_bf16;
        c.fta = NULL; c.fba = NULL;
        const int K = (int)c.in_dim, N = (int)c.out_dim;
        const double wbytes = is_bf16 ? (double)K * (double)N * 2.0 : (double)K * (double)N * 1.03;
        /* rotation set: layer L first, then L+1, ... (wrapping) with the same
         * type and dims, until the footprint is >= 2x L2 or the layers run out */
        size_t R_target = (size_t)ceil(2.0 * (double)l2_bytes / wbytes);
        if (R_target < 1) R_target = 1;
        std::vector<uint32_t> layers;
        if (shapes[si].single) {
            c.w.push_back(base);
        } else {
            for (uint32_t k = 0; k < PULSAR_N_LAYER && c.w.size() < R_target; k++) {
                const uint32_t l = (il + k) % PULSAR_N_LAYER;
                const pulsar_tensor *t = e->weights.layer[l].*shapes[si].f;
                if (t && t->type == base->type && t->dim[0] == base->dim[0] && t->dim[1] == base->dim[1]) {
                    c.w.push_back(t);
                    layers.push_back(l);
                }
            }
        }
        const size_t R = c.w.size();
        const double footprint = wbytes * (double)R;
        c.x = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.in_dim * sizeof(float));
        c.out = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.out_dim * sizeof(float));
        if (!c.x || !c.out) { fprintf(stderr, "alloc failed for %s\n", name); return 1; }
        fill_rand(c.x, (uint64_t)MMAX * c.in_dim, 7u + (uint32_t)si);
        /* weights: every rotated layer; slabs straight from the mmap
         * (MXFP8_LT: [data N*K][scale rup(N,128)*rup(K/32,4)]) */
        unsigned long long sfb_mismatch = 0;
        for (size_t r = 0; r < R; r++) {
            const uint8_t *host = (const uint8_t *)e->model.map + c.w[r]->abs_offset;
            if (is_bf16) {
                fb_weight *w = fb_prepare(host, N, K);
                if (!w) { fprintf(stderr, "fb_prepare failed for %s layer %zu\n", name, r); return 1; }
                c.fbw.push_back(w);
            } else {
                unsigned long long mis = 0;
                ft_weight *w = ft_prepare(host, host + (size_t)N * K, N, K, &mis);
                if (!w) { fprintf(stderr, "ft_prepare failed for %s layer %zu\n", name, r); return 1; }
                c.ftw.push_back(w);
                sfb_mismatch += mis;
            }
        }
        if (is_bf16) { c.fba = fb_act_prepare(N, K); if (!c.fba) return 1; }
        else         { c.fta = ft_act_prepare(N, K); if (!c.fta) return 1; }
        printf("%s  K=%d N=%d  (%.1f MB%s)  rotation: R=%zu layer%s", name, K, N, wbytes / 1e6, is_bf16 ? " bf16" : "",
               R, R == 1 ? "" : "s");
        if (!layers.empty()) {
            printf(" [%u", layers[0]);
            for (size_t r = 1; r < layers.size(); r++) printf(",%u", layers[r]);
            printf("]");
        }
        printf(", footprint %.0f MB %s L2 %.1f MB\n", footprint / 1e6, footprint > (double)l2_bytes ? ">" : "<= (NOT DRAM-honest)", l2_bytes / 1e6);
        if (!is_bf16)
            printf("  SFB layout vs LT slab over %zu layer%s: %llu of %llu scale bytes at a different offset%s\n",
                   R, R == 1 ? "" : "s", sfb_mismatch, (unsigned long long)R * N * (K / 32),
                   sfb_mismatch == 0 ? "  -> SAME swizzle, LT slab usable as SFB" : "  -> re-layout needed");
        const int NV = is_bf16 ? fb_nvariants() : ft_nvariants();
        /* neutrality references: each variant's 16-row output on layer L, and
         * its decomposition at 16 rows */
        std::vector<std::vector<float>> refs(NV);
        std::vector<ft_decomp> decomp16(NV);
        std::vector<float> cur, cur_s1, ref_eng;
        if ((is_bf16 ? fb_pack(c.fba, c.x, (int)MMAX) : ft_pack(c.fta, c.x, (int)MMAX)) != 0) { rc = 1; break; }
        if (!is_bf16) printf("  decomposition at M=16:");
        for (int v = 0; v < NV && !rc; v++) {
            const int mk = make_plans(&c, v, MMAX);
            if (mk) { fprintf(stderr, "variant %s at M=16 refused (plan_make %d) for %s\n", is_bf16 ? fb_variant_name(v) : ft_variant_name(v), mk, name); rc = 1; break; }
            if (!ft_launch(&c, MMAX, 0) || ft_sync() != 0) { fprintf(stderr, "variant %d at M=16 failed for %s\n", v, name); rc = 1; break; }
            if (!read_out(c.out, refs[v], (uint64_t)MMAX * N)) { rc = 1; break; }
            if (!is_bf16) {
                ft_plan_decomp(c.ftp[0], &decomp16[v]);
                printf("  %s: splits %d, ctas %u", ft_variant_name(v), decomp16[v].splits, decomp16[v].ctas);
                if (decomp16[v].sk_units) printf(" (sk_units %u, sk_tiles %u)", decomp16[v].sk_units, decomp16[v].sk_tiles);
                if (decomp16[v].big_units) printf(" (big %u)", decomp16[v].big_units);
                printf(";");
            }
            release_plans(&c);
        }
        if (!is_bf16) printf("\n");
        if (rc) break;
        printf("  %5s | %14s |", "M", "engine");
        for (int v = 0; v < NV; v++) printf(" %13s", is_bf16 ? fb_variant_name(v) : ft_variant_name(v));
        printf(" | %10s | neutral(rows==M16) | decomp==M16 | max|d| vs S=1 per variant\n", "|S1-eng|");
        for (int mi = 0; mi < NM; mi++) {
            const uint32_t M = Ms[mi];
            /* the engine's own dispatch for M decode-kind rows (L167: row kind chooses) */
            if (!encode_for_engine(&c, M)) { fprintf(stderr, "activation encode failed for %s at M=%u\n", name, M); rc = 1; break; }
            if (!pulsar_gpu_matmul_set_batch_decode_rows((int)M)) { rc = 1; break; }
            const double te = time_launches(engine_launch, &c, M, reps);
            (void)pulsar_gpu_matmul_set_batch_decode_rows(0);
            if (te < 0) fprintf(stderr, "engine dispatch failed at M=%u for %s (its refusal, if any, is printed above)\n", M, name);
            if (!read_out(c.out, ref_eng, (uint64_t)M * N)) { rc = 1; break; }
            /* the probe's rows: packed once per M, outside the loop, like the engine's */
            if ((is_bf16 ? fb_pack(c.fba, c.x, (int)M) : ft_pack(c.fta, c.x, (int)M)) != 0) { rc = 1; break; }
            printf("  %5u | %9.1f(%3.0f) |", M, te, te > 0 ? wbytes / te / 1e3 : 0.0);
            double deng = 0;
            std::string neutral, decomp_same, dist;
            for (int v = 0; v < NV; v++) {
                const int mk = make_plans(&c, v, M);
                if (mk) {
                    release_plans(&c);
                    fprintf(stderr, "variant %s refused at M=%u for %s (plan_make %d)\n", is_bf16 ? fb_variant_name(v) : ft_variant_name(v), M, name, mk);
                    printf(" %13s", "refused");
                    neutral += " -"; decomp_same += " -"; if (v > 0) dist += " refused";
                    continue;
                }
                const double tf = time_launches(ft_launch, &c, M, reps);
                if (tf < 0) { fprintf(stderr, "variant %d failed at M=%u for %s\n", v, M, name); rc = 1; break; }
                /* layer L once more for the byte checks: the loop ended on layer (reps-1) % R */
                if (!ft_launch(&c, M, 0) || ft_sync() != 0 || !read_out(c.out, cur, (uint64_t)M * N)) { rc = 1; break; }
                neutral += count_byte_mismatch(cur, refs[v], (uint64_t)M * N) == 0 ? " Y" : " N";
                if (!is_bf16) {
                    ft_decomp d; ft_plan_decomp(c.ftp[0], &d);
                    decomp_same += same_decomp(d, decomp16[v]) ? " Y" : " N";
                } else {
                    decomp_same += " Y";   /* one CTA per tile, identity swizzle: nothing to decompose */
                }
                if (v == 0) { deng = max_abs_diff(cur, ref_eng, (uint64_t)M * N); cur_s1 = cur; }
                else {
                    char buf[24];
                    snprintf(buf, sizeof buf, " %.2e", cur_s1.empty() ? 0.0 : max_abs_diff(cur, cur_s1, (uint64_t)M * N));
                    dist += buf;
                }
                release_plans(&c);
                printf(" %8.1f(%3.0f)", tf, wbytes / tf / 1e3);
            }
            if (rc) { release_plans(&c); break; }
            printf(" | %10.3e | %18s | %11s |%s\n", deng, neutral.c_str(), decomp_same.c_str(), dist.c_str());
        }
        printf("\n");
        for (ft_weight *w : c.ftw) ft_release(w);
        for (fb_weight *w : c.fbw) fb_release(w);
        if (c.fta) ft_act_release(c.fta);
        if (c.fba) fb_act_release(c.fba);
        pulsar_gpu_tensor_free(c.x);
        pulsar_gpu_tensor_free(c.out);
        if (rc) break;
    }
    pulsar_engine_close(e);
    return rc;
}

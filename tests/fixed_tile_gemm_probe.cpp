/* L151-C STAGE 0 PROBE: can ONE tensor-core tile configuration be both
 * M-neutral and cuBLASLt-flat on the <= 16-row dense step?
 *
 * MODEL-DEPENDENT (real MXFP8_LT weights): `make cuda-fixed-tile-probe`.
 * For each dense shape and M in {1..16} it times three arms on the same
 * f32 activation rows:
 *   unarmed  -- the default dispatch (nt <= 4 rows, cuBLASLt above)
 *   armed    -- M-neutral (nt kernels up to 16 rows)
 *   ft64/128 -- the probe's CUTLASS block-scaled MXFP8 x MXFP8 GEMM with a
 *               fixed 128 x TN x 128 tile (fixed_tile_gemm_kernels.cu), fed the
 *               engine's own E4M3 bytes and the LT weight slabs
 * and answers the three stage-0 questions from plans/L151-C-fixed-tile-gemm.md:
 *   1. speed: ft time vs cuBLASLt at M = 9 (go: within ~1.2x) and flatness
 *      (M = 16 within 10% of M = 1);
 *   2. neutrality: rows [0, M) of the M-row ft call are BYTE-IDENTICAL to the
 *      same rows of the 16-row call, for every M (mismatching floats counted);
 *   3. layout: how many weight scale bytes the CUTLASS SFB layout places at a
 *      different offset than the LT slab (0 = the LT slab is usable as-is).
 * Also printed: max |ft - unarmed| and max |ft - armed| per shape, so the
 * accumulation-order distance to both existing arms is on record.
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
    const pulsar_tensor *w;
    pulsar_gpu_tensor *x, *out;
    uint64_t in_dim, out_dim;
    ft_ctx *ft;     /* MXFP8 arm */
    fb_ctx *fb;     /* bf16 arm */
    int tn;
} ctx_t;

static bool engine_launch(void *v, uint32_t M) {
    ctx_t *c = (ctx_t *)v;
    if (c->w->type == PULSAR_TENSOR_BF16)
        return pulsar_gpu_matmul_bf16_tensor(c->out, c->e->model.map, c->e->model.size,
                                             c->w->abs_offset, c->in_dim, c->out_dim, c->x, M) != 0;
    return pulsar_gpu_matmul_mxfp8_tensor(c->out, c->e->model.map, c->e->model.size,
                                          c->w->abs_offset, c->in_dim, c->out_dim, c->x, M) != 0;
}
static bool ft_launch(void *v, uint32_t M) {
    ctx_t *c = (ctx_t *)v;
    if (c->fb) return fb_run(c->fb, c->tn, c->x, (int)M, c->out) == 0;
    return ft_run(c->ft, c->tn, c->x, (int)M, c->out) == 0;
}

typedef bool (*launch_fn)(void *, uint32_t);
static double time_launches(launch_fn fn, void *ctx, uint32_t M, int reps) {
    for (int i = 0; i < 3; i++) if (!fn(ctx, M)) return -1.0;
    if (!pulsar_gpu_end_commands() || ft_sync() != 0) return -1.0;
    const double t0 = now_us();
    for (int i = 0; i < reps; i++) if (!fn(ctx, M)) return -1.0;
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [LAYER]\n", argv[0]); return 2; }
    const uint32_t il = argc > 2 ? (uint32_t)atoi(argv[2]) : 4u;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    const pulsar_layer_weights *L = &e->weights.layer[il];
    if (il >= PULSAR_N_LAYER || !L->attn_q_b || !L->attn_output_b || !L->ffn_gate_shexp || !L->attn_q_a ||
        !L->ffn_gate_inp || !e->weights.output) {
        fprintf(stderr, "layer %u lacks a needed tensor\n", il);
        return 1;
    }
    struct { const char *name; const pulsar_tensor *w; } shapes[] = {
        {"attn_q_b", L->attn_q_b}, {"attn_output_b", L->attn_output_b},
        {"ffn_gate_shexp", L->ffn_gate_shexp}, {"attn_q_a", L->attn_q_a},
        {"router (bf16)", L->ffn_gate_inp}, {"output head (bf16)", e->weights.output},
    };
    const uint32_t Ms[] = {1, 2, 4, 5, 8, 9, 12, 16};
    const int NM = (int)(sizeof(Ms) / sizeof(Ms[0]));
    const uint32_t MMAX = 16;
    const int reps = 40;
    int rc = 0;
    printf("layer %u; us per call (%d launches + 1 sync). unarmed = default dispatch (nt <= 4, cuBLASLt above); "
           "armed = M-neutral nt; ft64/ft128 = fixed CUTLASS tile (MXFP8: 128xTNx128 block-scaled; bf16: 64xTNx32 mma.sync)\n\n", il, reps);
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const pulsar_tensor *w = shapes[si].w;
        const bool is_bf16 = w->type == PULSAR_TENSOR_BF16;
        if (!is_bf16 && w->type != PULSAR_TENSOR_MXFP8_LT) { printf("%s: type %u is neither MXFP8_LT nor BF16, skipped\n\n", shapes[si].name, (unsigned)w->type); continue; }
        ctx_t c;
        memset(&c, 0, sizeof c);
        c.e = e; c.w = w; c.in_dim = w->dim[0]; c.out_dim = w->dim[1];
        const int K = (int)c.in_dim, N = (int)c.out_dim;
        c.x = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.in_dim * sizeof(float));
        c.out = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.out_dim * sizeof(float));
        if (!c.x || !c.out) { fprintf(stderr, "alloc failed for %s\n", shapes[si].name); return 1; }
        fill_rand(c.x, (uint64_t)MMAX * c.in_dim, 7u + (uint32_t)si);
        /* L158: arm the E4M3 slot and let one unarmed 16-row call quantise into
         * it, as the engine's producers do -- the dense GEMM no longer has an
         * f32 fallback for an unarmed activation (one format or an error). */
        if (!is_bf16) {
            pulsar_gpu_mxfp8_act_cache_arm(c.x, MMAX, c.in_dim);
            pulsar_gpu_matmul_set_batch_mneutral(0);
            if (!engine_launch(&c, MMAX) || !pulsar_gpu_end_commands()) {
                fprintf(stderr, "warm call failed for %s\n", shapes[si].name);
                return 1;
            }
        }
        /* LT slabs straight from the mmap: [data N*K][scale rup(N,128)*rup(K/32,4)] */
        const uint8_t *host = (const uint8_t *)e->model.map + w->abs_offset;
        const size_t data_bytes = (size_t)N * K;
        unsigned long long sfb_mismatch = 0;
        double wbytes;
        if (is_bf16) {
            c.fb = fb_prepare(host, N, K);
            if (!c.fb) { fprintf(stderr, "fb_prepare failed for %s\n", shapes[si].name); return 1; }
            wbytes = (double)K * (double)N * 2.0;
            printf("%s  K=%d N=%d  (%.1f MB bf16)\n", shapes[si].name, K, N, wbytes / 1e6);
        } else {
            c.ft = ft_prepare(host, host + data_bytes, N, K, &sfb_mismatch);
            if (!c.ft) { fprintf(stderr, "ft_prepare failed for %s\n", shapes[si].name); return 1; }
            wbytes = (double)K * (double)N * 1.03;
            printf("%s  K=%d N=%d  (%.1f MB)   SFB layout vs LT slab: %llu of %llu scale bytes at a different offset%s\n",
                   shapes[si].name, K, N, wbytes / 1e6, sfb_mismatch, (unsigned long long)N * (K / 32),
                   sfb_mismatch == 0 ? "  -> SAME swizzle, LT slab usable as SFB" : "  -> re-layout needed");
        }
        printf("  %5s | %9s %9s | %9s %9s | %8s %8s | %10s %10s | %s\n", "M", "unarmed", "armed", "ft64", "ft128",
               "ft64GB/s", "ft128GBs", "|ft64-un|", "|ft64-arm|", "neutral(ft64/ft128: rows==M16)");
        /* neutrality references: the 16-row ft outputs */
        std::vector<float> ref64, ref128, cur, ref_un, ref_arm;
        for (int tn = 64; tn <= 128; tn += 64) {
            c.tn = tn;
            if (!ft_launch(&c, 16) || ft_sync() != 0) { fprintf(stderr, "ft%d at M=16 failed for %s\n", tn, shapes[si].name); rc = 1; break; }
            if (!read_out(c.out, tn == 64 ? ref64 : ref128, (uint64_t)16 * N)) { rc = 1; break; }
        }
        if (rc) break;
        for (int mi = 0; mi < NM; mi++) {
            const uint32_t M = Ms[mi];
            pulsar_gpu_matmul_set_batch_mneutral(0);
            const double tu = time_launches(engine_launch, &c, M, reps);
            if (!read_out(c.out, ref_un, (uint64_t)M * N)) { rc = 1; break; }
            pulsar_gpu_matmul_set_batch_mneutral((int)M);
            const double ta = time_launches(engine_launch, &c, M, reps);
            if (!read_out(c.out, ref_arm, (uint64_t)M * N)) { rc = 1; break; }
            pulsar_gpu_matmul_set_batch_mneutral(0);
            double tf[2] = {-1, -1}, dun = 0, darm = 0;
            uint64_t bad[2] = {0, 0};
            for (int ti = 0; ti < 2; ti++) {
                c.tn = ti == 0 ? 64 : 128;
                tf[ti] = time_launches(ft_launch, &c, M, reps);
                if (tf[ti] < 0) { fprintf(stderr, "ft%d failed at M=%u for %s\n", c.tn, M, shapes[si].name); rc = 1; break; }
                if (!read_out(c.out, cur, (uint64_t)M * N)) { rc = 1; break; }
                const std::vector<float> &ref = ti == 0 ? ref64 : ref128;
                for (uint64_t i = 0; i < (uint64_t)M * N; i++)
                    if (memcmp(&cur[i], &ref[i], sizeof(float)) != 0) bad[ti]++;
                if (ti == 0) { dun = max_abs_diff(cur, ref_un, (uint64_t)M * N); darm = max_abs_diff(cur, ref_arm, (uint64_t)M * N); }
            }
            if (rc) break;
            printf("  %5u | %9.1f %9.1f | %9.1f %9.1f | %8.0f %8.0f | %10.3e %10.3e | %s / %s%s\n", M, tu, ta, tf[0], tf[1],
                   tf[0] > 0 ? wbytes / tf[0] / 1e3 : 0.0, tf[1] > 0 ? wbytes / tf[1] / 1e3 : 0.0, dun, darm,
                   bad[0] == 0 ? "YES" : "NO", bad[1] == 0 ? "YES" : "NO",
                   (bad[0] || bad[1]) ? "  <-- rows differ from the 16-row call" : "");
        }
        printf("\n");
        if (c.ft) ft_release(c.ft);
        if (c.fb) fb_release(c.fb);
        pulsar_gpu_tensor_free(c.x);
        pulsar_gpu_tensor_free(c.out);
        if (rc) break;
    }
    pulsar_engine_close(e);
    return rc;
}

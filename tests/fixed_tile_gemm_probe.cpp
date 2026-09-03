/* L151-C STAGE 0 PROBE: can ONE tensor-core tile configuration be both
 * M-neutral and cuBLASLt-flat on the <= 16-row dense step?
 *
 * MODEL-DEPENDENT (real MXFP8_LT weights): `make cuda-fixed-tile-probe`.
 * For each dense shape and M in {1..16} it times two arms on the same
 * f32 activation rows:
 *   engine   -- the engine's dispatch with the M rows declared DECODE rows
 *               (pulsar_gpu_matmul_set_batch_decode_rows(M): the M-independent
 *               GEMV/nt arms, the served verify-step shape; L167)
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
 * Also printed: max |ft - engine| per shape, so the accumulation-order
 * distance to the engine's arm is on record.
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
    printf("layer %u; us per call (%d launches + 1 sync). engine = the dispatch with M rows declared decode "
           "(GEMV/nt arms, <= %u rows); ft64/ft128 = fixed CUTLASS tile (MXFP8: 128xTNx128 block-scaled; "
           "bf16: 64xTNx32 mma.sync)\n\n",
           il, reps, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
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
        if (!(is_bf16 ? pulsar_gpu_bf16_act_encode_f32(c.x, MMAX, c.in_dim)
                      : pulsar_gpu_mxfp8_act_cache_encode_f32(c.x, MMAX, c.in_dim)) ||
            !pulsar_gpu_end_commands()) {
            fprintf(stderr, "activation encode failed for %s\n", shapes[si].name);
            return 1;
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
        const int NV = is_bf16 ? fb_nvariants() : ft_nvariants();
        printf("  %5s | %9s |", "M", "engine");
        for (int v = 0; v < NV; v++) printf(" %12s", is_bf16 ? fb_variant_name(v) : ft_variant_name(v));
        printf(" | %10s | neutral per variant (rows == M16 rows)\n", "|v0-eng|");
        /* neutrality references: each variant's 16-row output */
        std::vector<std::vector<float>> refs(NV);
        std::vector<float> cur, ref_eng;
        for (int v = 0; v < NV && !rc; v++) {
            c.tn = v;
            if (!ft_launch(&c, 16) || ft_sync() != 0) { fprintf(stderr, "variant %d at M=16 failed for %s\n", v, shapes[si].name); rc = 1; break; }
            if (!read_out(c.out, refs[v], (uint64_t)16 * N)) { rc = 1; break; }
        }
        if (rc) break;
        for (int mi = 0; mi < NM; mi++) {
            const uint32_t M = Ms[mi];
            /* the engine's own dispatch for M decode-kind rows (L167: row kind chooses) */
            if (!pulsar_gpu_matmul_set_batch_decode_rows((int)M)) { rc = 1; break; }
            const double te = time_launches(engine_launch, &c, M, reps);
            (void)pulsar_gpu_matmul_set_batch_decode_rows(0);
            if (!read_out(c.out, ref_eng, (uint64_t)M * N)) { rc = 1; break; }
            printf("  %5u | %9.1f |", M, te);
            double deng = 0;
            std::string flags;
            for (int v = 0; v < NV; v++) {
                c.tn = v;
                const double tf = time_launches(ft_launch, &c, M, reps);
                if (tf < 0) { fprintf(stderr, "variant %d failed at M=%u for %s\n", v, M, shapes[si].name); rc = 1; break; }
                if (!read_out(c.out, cur, (uint64_t)M * N)) { rc = 1; break; }
                uint64_t bad = 0;
                for (uint64_t i = 0; i < (uint64_t)M * N; i++)
                    if (memcmp(&cur[i], &refs[v][i], sizeof(float)) != 0) bad++;
                flags += bad == 0 ? " Y" : " N";
                if (v == 0) deng = max_abs_diff(cur, ref_eng, (uint64_t)M * N);
                printf(" %7.1f(%3.0f)", tf, wbytes / tf / 1e3);
            }
            if (rc) break;
            printf(" | %10.3e |%s\n", deng, flags.c_str());
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

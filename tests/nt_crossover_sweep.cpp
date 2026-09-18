/* L151 SWEEP: the dense GEMM dispatch per production shape at every
 * decode-sized row count, both row kinds.
 *
 * MODEL-DEPENDENT (real weights, real cache, real dispatch): `make cuda-nt-sweep`.
 * For each production dense shape and M in {1..16} it times the engine's own
 * wrapper call twice: with the M rows declared DECODE rows
 * (pulsar_gpu_matmul_set_batch_decode_rows(M): the M-independent GEMV/nt
 * arms, what a verify step or drafter forward of M rows runs) and declared
 * PREFILL rows (count 0: cuBLASLt/tensor-core at every M, what an M-row
 * remainder chunk runs).  Since L167 row KIND chooses the arm, not row count,
 * so the two columns are the two lanes' costs for the same shape (rule 8:
 * sweep kernels across M).
 *
 * Host wall-clock around 40 launches + one sync, per (shape, M, kind); GB/s
 * is weight bytes / decode time, the bandwidth-bound ceiling being ~273 GB/s
 * on the GB10.
 *
 * MoE experts are NOT here: their two tiers (per-expert vs grouped cutlass)
 * take routing and a real batch; L146 measured them in-engine (grouped loses
 * at 9-12 rows). Attention is not a GEMM. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double now_us(void) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_rand(pulsar_gpu_tensor *t, uint64_t n_floats, uint32_t seed) {
    float *h = (float *)malloc(n_floats * sizeof(float));
    uint32_t s = seed;
    for (uint64_t i = 0; i < n_floats; i++) {
        s = s * 1664525u + 1013904223u;
        h[i] = ((float)(s >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
    pulsar_gpu_tensor_write(t, 0, h, n_floats * sizeof(float));
    free(h);
}

typedef bool (*launch_fn)(void *ctx, uint32_t n_tok);

static double time_launches(launch_fn fn, void *ctx, uint32_t n_tok, int reps) {
    for (int i = 0; i < 3; i++) if (!fn(ctx, n_tok)) return -1.0;
    if (!pulsar_gpu_end_commands()) return -1.0;
    const double t0 = now_us();
    for (int i = 0; i < reps; i++) if (!fn(ctx, n_tok)) return -1.0;
    if (!pulsar_gpu_end_commands()) return -1.0;
    return (now_us() - t0) / reps;
}

/* ---- dense GEMM through the format-resolving wrappers ---- */
typedef struct {
    const pulsar_engine *e;
    const pulsar_tensor *w;
    pulsar_gpu_tensor *x, *out;
    uint64_t in_dim, out_dim;
} gemm_ctx;

static bool gemm_launch(void *vctx, uint32_t n_tok) {
    gemm_ctx *c = (gemm_ctx *)vctx;
    if (c->w->type == PULSAR_TENSOR_BF16)
        return pulsar_gpu_matmul_bf16_tensor(c->out, c->e->model.map, c->e->model.size,
                                             c->w->abs_offset, c->in_dim, c->out_dim, c->x, n_tok) != 0;
    return pulsar_gpu_matmul_mxfp8_tensor(c->out, c->e->model.map, c->e->model.size,
                                          c->w->abs_offset, c->in_dim, c->out_dim, c->x, n_tok) != 0;
}

/* L222 COLD PASS.  time_launches() repeats ONE weight 40 times, so the small
 * shapes are L2-resident in it (its own GB/s column reads above the ~273 GB/s
 * cold DRAM figure at M = 1).  A real step reads each dense weight ONCE, and the
 * 43 layers' copies of one shape sum to far more than L2, so this pass makes one
 * call per layer with no per-call sync and divides by the layer count: every
 * timed call streams its own weight from DRAM, which is what the production
 * verify row actually pays.  Reference re-runs this in layer order so the layer
 * k+1 weight cannot still be the one layer k wanted. */
typedef const pulsar_tensor *(*shape_pick)(const pulsar_engine *e, uint32_t il);

static double time_pass_cold(launch_fn fn, void *ctx, const pulsar_engine *e,
                            shape_pick pick, uint32_t n_tok, int *calls_out) {
    /* warm the LAUNCH path (not the weight) on the first layer, then time the
     * whole layer sweep as one block so the sync is amortised exactly as the
     * warm pass amortises it. */
    if (!fn(ctx, n_tok) || !pulsar_gpu_end_commands()) return -1.0;
    int calls = 0;
    const double t0 = now_us();
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_tensor *w = pick(e, il);
        if (!w) continue;
        ((gemm_ctx *)ctx)->w = w;
        if (!fn(ctx, n_tok)) return -1.0;
        calls++;
    }
    if (!pulsar_gpu_end_commands()) return -1.0;
    const double us = now_us() - t0;
    if (calls_out) *calls_out = calls;
    return calls ? us / calls : -1.0;
}

/* ---- the grouped attention-output 'a' + 'b' pair ---- */
typedef struct {
    const pulsar_engine *e;
    const pulsar_layer_weights *L;
    pulsar_gpu_tensor *heads, *low, *out;
    uint64_t group_dim, rank;
    uint32_t n_groups;
} attnout_ctx;

static bool attnout_launch(void *vctx, uint32_t n_tok) {
    attnout_ctx *c = (attnout_ctx *)vctx;
    return pulsar_gpu_attention_output_batch_tensor(
               c->out, c->low, c->e->model.map, c->e->model.size,
               c->L->attn_output_a->abs_offset, c->L->attn_output_b->abs_offset,
               c->group_dim, c->rank, c->n_groups, PULSAR_N_EMBD, c->heads, n_tok) != 0;
}

static const pulsar_tensor *pick_layer_shape(const pulsar_engine *e, uint32_t il) { return NULL; }
static const pulsar_tensor *pick_q_a(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].attn_q_a; }
static const pulsar_tensor *pick_q_b(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].attn_q_b; }
static const pulsar_tensor *pick_kv(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].attn_kv; }
static const pulsar_tensor *pick_out_b(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].attn_output_b; }
static const pulsar_tensor *pick_gate_shexp(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].ffn_gate_shexp; }
static const pulsar_tensor *pick_down_shexp(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].ffn_down_shexp; }
static const pulsar_tensor *pick_router(const pulsar_engine *e, uint32_t il) { return e->weights.layer[il].ffn_gate_inp; }

static const char *type_name(uint32_t t) {
    return t == PULSAR_TENSOR_BF16 ? "bf16" : "mxfp8";
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
    if (il >= PULSAR_N_LAYER || !L->attn_q_a || !L->attn_q_b || !L->attn_kv || !L->attn_output_a ||
        !L->attn_output_b || !L->ffn_gate_shexp || !L->ffn_down_shexp || !L->ffn_gate_inp ||
        !e->weights.output) {
        fprintf(stderr, "layer %u lacks a needed tensor\n", il);
        return 1;
    }
    struct { const char *name; const pulsar_tensor *w; shape_pick pick; } shapes[] = {
        {"attn_q_a", L->attn_q_a, pick_q_a}, {"attn_q_b", L->attn_q_b, pick_q_b},
        {"attn_kv", L->attn_kv, pick_kv},
        {"attn_output_b", L->attn_output_b, pick_out_b}, {"ffn_gate_shexp", L->ffn_gate_shexp, pick_gate_shexp},
        {"ffn_down_shexp", L->ffn_down_shexp, pick_down_shexp}, {"router (bf16)", L->ffn_gate_inp, pick_router},
        {"output head (bf16)", e->weights.output, NULL},
    };
    const bool cold = getenv("NTSWEEP_COLD") != NULL;
    if (cold)
        printf("NTSWEEP_COLD: one call per LAYER, no per-call sync, /n_layers -- each timed call streams its\n"
               "              own weight from DRAM (the warm mode repeats one weight and is L2-served).\n"
               "              The head/grouped blocks have no per-layer twin and stay warm.\n\n");
    const uint32_t Ms[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
    const int NM = (int)(sizeof(Ms) / sizeof(Ms[0]));
    const uint32_t MMAX = 16;
    const int reps = 40;
    printf("layer %u; times are us per call (%d launches + 1 sync); 'decode' = rows declared decode "
           "(GEMV/nt arms, <= %u rows), 'prefill' = rows declared prefill (tensor-core at every M)\n\n",
           il, reps, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
    printf("%-20s %-6s %5s | %9s %9s | %7s | %s\n", "shape", "fmt", "M", "decode", "prefill", "ratio", "decode GB/s (weights/time)");
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const pulsar_tensor *w = shapes[si].w;
        gemm_ctx c;
        c.e = e; c.w = w; c.in_dim = w->dim[0]; c.out_dim = w->dim[1];
        c.x = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.in_dim * sizeof(float));
        c.out = pulsar_gpu_tensor_alloc((uint64_t)MMAX * c.out_dim * sizeof(float));
        if (!c.x || !c.out) { fprintf(stderr, "alloc failed for %s\n", shapes[si].name); return 1; }
        fill_rand(c.x, (uint64_t)MMAX * c.in_dim, 7u + (uint32_t)si);
        /* L151-D (2026-09-03): arm the E4M3 activation slot and let one
         * unarmed 16-row call quantise into it, exactly as the engine's
         * producers do.  Without this the armed column ran the f32 nt kernel
         * (no slot -> mxfp8_mmvq_deint_nt_kernel), NOT the A8 twin production
         * runs -- the instrument measured a kernel the served lane never
         * executes.  The engine announces "verify-batch GEMV W8A8" once per
         * shape when the A8 arm fires; the run script greps for it. */
        /* L158 inc 4: this tool is x's PRODUCER, so it encodes x itself; the
         * GEMMs no longer quantise for anyone. */
        if (w->type != PULSAR_TENSOR_BF16 &&
            (!pulsar_gpu_mxfp8_act_cache_encode_f32(c.x, MMAX, c.in_dim) || !pulsar_gpu_end_commands())) {
            fprintf(stderr, "activation encode failed for %s\n", shapes[si].name);
            return 1;
        }
        const double wbytes = (double)c.in_dim * (double)c.out_dim *
                              (w->type == PULSAR_TENSOR_BF16 ? 2.0 : 1.03);
        for (int mi = 0; mi < NM; mi++) {
            const uint32_t M = Ms[mi];
            double td, tp;
            int ncall = 0;
            if (cold && shapes[si].pick) {
                if (!pulsar_gpu_matmul_set_batch_decode_rows((int)M)) return 1;
                td = time_pass_cold(gemm_launch, &c, e, shapes[si].pick, M, &ncall);
                (void)pulsar_gpu_matmul_set_batch_decode_rows(0);
                tp = time_pass_cold(gemm_launch, &c, e, shapes[si].pick, M, NULL);
                c.w = w;   /* restore for the next shape's warm path / wbytes */
            } else {
                if (!pulsar_gpu_matmul_set_batch_decode_rows((int)M)) return 1;
                td = time_launches(gemm_launch, &c, M, reps);
                (void)pulsar_gpu_matmul_set_batch_decode_rows(0);
                tp = time_launches(gemm_launch, &c, M, reps);
            }
            printf("%-20s %-6s %5u | %9.1f %9.1f | %7.2f | %6.0f%s\n", shapes[si].name, type_name(w->type),
                   M, td, tp, td > 0 && tp > 0 ? tp / td : 0.0, td > 0 ? wbytes / td / 1e3 : 0.0,
                   cold ? (shapes[si].pick ? "  (cold, per-layer)" : "  (warm: no per-layer twin)") : "");
        }
        pulsar_gpu_mxfp8_act_cache_disarm();
        pulsar_gpu_tensor_free(c.x);
        pulsar_gpu_tensor_free(c.out);
        printf("\n");
    }
    /* grouped attention output */
    {
        attnout_ctx a;
        a.e = e; a.L = L;
        a.n_groups = PULSAR_N_OUT_GROUP;
        a.group_dim = (uint64_t)PULSAR_N_HEAD_DIM * (PULSAR_N_HEAD / a.n_groups);
        a.rank = PULSAR_N_LORA_O;
        const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
        a.heads = pulsar_gpu_tensor_alloc_elt((uint64_t)MMAX * q_dim, PULSAR_HEADS_ELT_SIZE, PULSAR_HEADS_ELT_FMT);
        a.low = pulsar_gpu_tensor_alloc((uint64_t)MMAX * a.n_groups * a.rank * sizeof(float));
        a.out = pulsar_gpu_tensor_alloc((uint64_t)MMAX * PULSAR_N_EMBD * sizeof(float));
        if (!a.heads || !a.low || !a.out) { fprintf(stderr, "attn-out alloc failed\n"); return 1; }
        /* heads may be a 16-bit carrier: write zeros-ish bytes; timing is data-independent */
        {
            const uint64_t bytes = pulsar_gpu_tensor_bytes(a.heads);
            uint8_t *h = (uint8_t *)calloc(1, bytes);
            for (uint64_t i = 0; i < bytes; i += 7) h[i] = (uint8_t)(i * 31u);
            pulsar_gpu_tensor_write(a.heads, 0, h, bytes);
            free(h);
        }
        const double wbytes = ((double)L->attn_output_a->elements + (double)L->attn_output_b->elements) * 1.03;
        for (int mi = 0; mi < NM; mi++) {
            const uint32_t M = Ms[mi];
            if (!pulsar_gpu_matmul_set_batch_decode_rows((int)M)) return 1;
            const double td = time_launches(attnout_launch, &a, M, reps);
            (void)pulsar_gpu_matmul_set_batch_decode_rows(0);
            const double tp = time_launches(attnout_launch, &a, M, reps);
            printf("%-20s %-6s %5u | %9.1f %9.1f | %7.2f | %6.0f\n", "attn_output a+b", "mxfp8", M, td, tp,
                   td > 0 && tp > 0 ? tp / td : 0.0, td > 0 ? wbytes / td / 1e3 : 0.0);
        }
        pulsar_gpu_tensor_free(a.heads);
        pulsar_gpu_tensor_free(a.low);
        pulsar_gpu_tensor_free(a.out);
    }
    pulsar_engine_close(e);
    return 0;
}

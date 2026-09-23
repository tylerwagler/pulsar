/* 4g HEAD-SPLIT gate (L241): a rank's slice of the attention projections is a
 * BYTE-IDENTICAL piece of the whole.
 *
 * WHY THIS EXISTS
 * ---------------
 * Slice 4g splits attention by OUTPUT GROUP: a rank runs attn_q_b over its
 * heads' rows, the attention core on those heads, and attn_output_a over its
 * groups' rows, then the group gathers `low` and every rank runs attn_output_b
 * whole.  Every value a rank computes is a head- or group-independent piece of
 * the single-box computation, so the split is bit-exact by construction --
 * IF the row slices the engine registers resolve to the right bytes of the
 * pre-stored MXFP8_LT layout (data rows + the 128-row scale band) on BOTH GEMM
 * arms.  That is the part a two-Spark run would only show as "the logits
 * differ", and it is checkable on one GPU with the real weights: run the whole
 * projection and each fabricated rank's slice on the same input, and compare
 * the bytes column range by column range.
 *
 * WHAT IT PROVES
 * --------------
 *   A. attn_q_b: for n = 2, 3, 4 ranks and every rank, the sliced GEMM's rows
 *      equal the whole GEMM's columns [lo, hi) -- decode rows (the batched
 *      GEMV arm) and prefill rows (the cuBLASLt tensor-core arm).
 *   B. attn_output_a: the grouped 'a' projection over the owned groups, fed
 *      the compact owned-heads encoding, equals the whole projection's columns
 *      [g_lo*rank, g_hi*rank) -- both arms again.
 *   C. Negative controls: a slice off the 128-row scale band is refused at
 *      registration; a registered slice resolved with other dims is refused
 *      at the GEMM.
 *
 * Layer 0's weights, random inputs from a fixed seed, in the runner's shared
 * engine (no second load).  The engine is borrowed: opened through
 * gate_engine_open, never closed here. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"
#include "tp/pulsar_tp.h"
#include "gate_entry.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_rng = 0x4C241u;
static float frand(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float)(g_rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even */
    return (uint16_t)(u >> 16);
}

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; fprintf(stderr, "  FAIL  "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

/* A. attn_q_b whole vs every rank's row slice, one regime. */
static void check_q_b(pulsar_engine *e, const pulsar_layer_weights *L, uint32_t n_rows, int decode) {
    const uint64_t q_rank = L->attn_q_a->dim[1];
    const uint64_t hd = PULSAR_N_HEAD_DIM;
    const uint32_t groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / groups;
    const uint64_t q_full = (uint64_t)PULSAR_N_HEAD * hd;
    const char *regime = decode ? "decode rows (GEMV arm)" : "prefill rows (tensor-core arm)";

    float *hx = (float *)malloc((size_t)n_rows * q_rank * sizeof(float));
    float *hfull = (float *)malloc((size_t)n_rows * q_full * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)n_rows * q_rank; i++) hx[i] = frand();
    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)n_rows * q_rank * sizeof(float));
    pulsar_gpu_tensor *full = pulsar_gpu_tensor_alloc((uint64_t)n_rows * q_full * sizeof(float));
    if (!hx || !hfull || !x || !full) { CHECK(0, "q_b: allocation"); goto out; }
    pulsar_gpu_tensor_write(x, 0, hx, (uint64_t)n_rows * q_rank * sizeof(float));
    /* The engine feeds attn_q_b from a producer-emitted E4M3 slot (q_a_norm's
     * epilogue); this gate is the producer of its own x, so it emits the same
     * way.  One emit serves the whole GEMM and every slice: identical operand. */
    if (!pulsar_gpu_mxfp8_act_emit_f32(x, n_rows, q_rank)) { CHECK(0, "q_b: x E4M3 emit refused"); goto out; }
    pulsar_gpu_matmul_set_batch_decode_rows(decode ? (int)n_rows : 0);
    if (!gpu_graph_matmul_mxfp8_rows_named_tensor("gate attn_q_b whole", 0, 0, full, &e->model, L->attn_q_b,
                                                  q_rank, q_full, 0, q_full, x, n_rows)) {
        CHECK(0, "q_b whole GEMM refused (%s)", regime);
        goto out;
    }
    pulsar_gpu_tensor_read(full, 0, hfull, (uint64_t)n_rows * q_full * sizeof(float));
    for (int n = 2; n <= 4; n++) {
        for (int r = 0; r < n; r++) {
            uint32_t g_lo = 0, g_hi = 0;
            if (!pulsar_tp_owned_range(r, (uint32_t)n, groups, &g_lo, &g_hi) || g_hi <= g_lo) {
                CHECK(0, "owned_range(%d of %d) over %u groups", r, n, groups);
                continue;
            }
            const uint64_t lo = (uint64_t)g_lo * group_heads * hd, hi = (uint64_t)g_hi * group_heads * hd;
            if (!pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(&e->model, L->attn_q_b),
                                                      L->attn_q_b->abs_offset, q_rank, q_full, lo, hi)) {
                CHECK(0, "q_b: register rows [%llu,%llu) for rank %d of %d",
                      (unsigned long long)lo, (unsigned long long)hi, r, n);
                continue;
            }
            pulsar_gpu_tensor *sl = pulsar_gpu_tensor_alloc((uint64_t)n_rows * (hi - lo) * sizeof(float));
            float *hsl = (float *)malloc((size_t)n_rows * (hi - lo) * sizeof(float));
            if (!sl || !hsl) { CHECK(0, "q_b slice allocation"); if (sl) pulsar_gpu_tensor_free(sl); free(hsl); continue; }
            if (!gpu_graph_matmul_mxfp8_rows_named_tensor("gate attn_q_b slice", 0, 0, sl, &e->model, L->attn_q_b,
                                                          q_rank, q_full, lo, hi, x, n_rows)) {
                CHECK(0, "q_b slice GEMM refused (rank %d of %d, %s)", r, n, regime);
            } else {
                pulsar_gpu_tensor_read(sl, 0, hsl, (uint64_t)n_rows * (hi - lo) * sizeof(float));
                uint64_t bad = 0;
                for (uint32_t row = 0; row < n_rows; row++)
                    if (memcmp(hsl + (uint64_t)row * (hi - lo), hfull + (uint64_t)row * q_full + lo,
                               (size_t)(hi - lo) * sizeof(float)) != 0) bad++;
                CHECK(bad == 0, "q_b: rank %d of %d rows [%llu,%llu): %llu of %u rows differ from the whole (%s)",
                      r, n, (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)bad, n_rows, regime);
                if (bad == 0)
                    printf("  ok    attn_q_b rank %d/%d rows [%6llu,%6llu) == whole  (%s, %u rows)\n",
                           r, n, (unsigned long long)lo, (unsigned long long)hi, regime, n_rows);
            }
            pulsar_gpu_tensor_free(sl);
            free(hsl);
        }
    }
out:
    pulsar_gpu_matmul_set_batch_decode_rows(0);
    if (x) { pulsar_gpu_act_slot_drop(x); pulsar_gpu_tensor_free(x); }   /* the slot dies with its buffer (L159) */
    if (full) pulsar_gpu_tensor_free(full);
    free(hx);
    free(hfull);
}

/* B. attn_output_a whole vs every rank's group slice, one regime. */
static void check_out_a(pulsar_engine *e, const pulsar_layer_weights *L, uint32_t n_rows, int decode) {
    const uint64_t hd = PULSAR_N_HEAD_DIM;
    const uint32_t groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / groups;
    const uint64_t group_dim = (uint64_t)group_heads * hd;
    const uint64_t rank = PULSAR_N_LORA_O;
    const uint64_t heads_full = (uint64_t)groups * group_dim;   /* elements per row */
    const uint64_t low_full = (uint64_t)groups * rank;
    const char *regime = decode ? "decode rows (GEMV arm)" : "prefill rows (tensor-core arm)";

    uint16_t *hheads = (uint16_t *)malloc((size_t)n_rows * heads_full * sizeof(uint16_t));
    float *hlow = (float *)malloc((size_t)n_rows * low_full * sizeof(float));
    pulsar_gpu_tensor *heads = pulsar_gpu_tensor_alloc((uint64_t)n_rows * heads_full * PULSAR_HEADS_ELT_SIZE);
    pulsar_gpu_tensor *low = pulsar_gpu_tensor_alloc((uint64_t)n_rows * low_full * sizeof(float));
    if (!hheads || !hlow || !heads || !low) { CHECK(0, "out_a: allocation"); goto out; }
    for (uint64_t i = 0; i < (uint64_t)n_rows * heads_full; i++) hheads[i] = f32_to_bf16(frand());
    pulsar_gpu_tensor_write(heads, 0, hheads, (uint64_t)n_rows * heads_full * PULSAR_HEADS_ELT_SIZE);
    pulsar_gpu_matmul_set_batch_decode_rows(decode ? (int)n_rows : 0);
    if (!pulsar_gpu_mxfp8_gact_emit_heads(heads, n_rows, groups, group_dim) ||
        !pulsar_gpu_attention_output_a_tensor(low, tensor_map_base(&e->model, L->attn_output_a),
                                              tensor_map_size(&e->model, L->attn_output_a),
                                              L->attn_output_a->abs_offset, group_dim, rank, groups,
                                              heads, n_rows)) {
        pulsar_gpu_mxfp8_gact_disarm();
        CHECK(0, "out_a whole projection refused (%s)", regime);
        goto out;
    }
    pulsar_gpu_mxfp8_gact_disarm();
    pulsar_gpu_tensor_read(low, 0, hlow, (uint64_t)n_rows * low_full * sizeof(float));
    for (int n = 2; n <= 4; n++) {
        for (int r = 0; r < n; r++) {
            uint32_t g_lo = 0, g_hi = 0;
            if (!pulsar_tp_owned_range(r, (uint32_t)n, groups, &g_lo, &g_hi) || g_hi <= g_lo) {
                CHECK(0, "owned_range(%d of %d) over %u groups", r, n, groups);
                continue;
            }
            const uint32_t own = g_hi - g_lo;
            const uint64_t a_lo = (uint64_t)g_lo * rank, a_hi = (uint64_t)g_hi * rank;
            if (!pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(&e->model, L->attn_output_a),
                                                      L->attn_output_a->abs_offset, group_dim, low_full, a_lo, a_hi)) {
                CHECK(0, "out_a: register rows [%llu,%llu) for rank %d of %d",
                      (unsigned long long)a_lo, (unsigned long long)a_hi, r, n);
                continue;
            }
            /* the compact owned-heads block a TP rank's attention writes */
            const uint64_t own_heads = (uint64_t)own * group_dim;
            uint16_t *hc = (uint16_t *)malloc((size_t)n_rows * own_heads * sizeof(uint16_t));
            float *hsl = (float *)malloc((size_t)n_rows * own * rank * sizeof(float));
            pulsar_gpu_tensor *ch = pulsar_gpu_tensor_alloc((uint64_t)n_rows * own_heads * PULSAR_HEADS_ELT_SIZE);
            pulsar_gpu_tensor *sl = pulsar_gpu_tensor_alloc((uint64_t)n_rows * own * rank * sizeof(float));
            if (!hc || !hsl || !ch || !sl) {
                CHECK(0, "out_a slice allocation");
            } else {
                for (uint32_t row = 0; row < n_rows; row++)
                    memcpy(hc + (uint64_t)row * own_heads,
                           hheads + (uint64_t)row * heads_full + (uint64_t)g_lo * group_dim,
                           (size_t)own_heads * sizeof(uint16_t));
                pulsar_gpu_tensor_write(ch, 0, hc, (uint64_t)n_rows * own_heads * PULSAR_HEADS_ELT_SIZE);
                int ok = pulsar_gpu_mxfp8_gact_emit_heads(ch, n_rows, own, group_dim) &&
                         pulsar_gpu_attention_output_a_tensor(sl, tensor_map_base(&e->model, L->attn_output_a),
                                                              tensor_map_size(&e->model, L->attn_output_a),
                                                              L->attn_output_a->abs_offset + a_lo * group_dim,
                                                              group_dim, rank, own, ch, n_rows);
                pulsar_gpu_mxfp8_gact_disarm();
                if (!ok) {
                    CHECK(0, "out_a slice projection refused (rank %d of %d, %s)", r, n, regime);
                } else {
                    pulsar_gpu_tensor_read(sl, 0, hsl, (uint64_t)n_rows * own * rank * sizeof(float));
                    uint64_t bad = 0;
                    for (uint32_t row = 0; row < n_rows; row++)
                        if (memcmp(hsl + (uint64_t)row * own * rank, hlow + (uint64_t)row * low_full + a_lo,
                                   (size_t)own * rank * sizeof(float)) != 0) bad++;
                    CHECK(bad == 0, "out_a: rank %d of %d groups [%u,%u): %llu of %u rows differ from the whole (%s)",
                          r, n, g_lo, g_hi, (unsigned long long)bad, n_rows, regime);
                    if (bad == 0)
                        printf("  ok    attn_output_a rank %d/%d groups [%u,%u) == whole  (%s, %u rows)\n",
                               r, n, g_lo, g_hi, regime, n_rows);
                }
            }
            if (ch) pulsar_gpu_tensor_free(ch);
            if (sl) pulsar_gpu_tensor_free(sl);
            free(hc);
            free(hsl);
        }
    }
out:
    pulsar_gpu_matmul_set_batch_decode_rows(0);
    if (heads) pulsar_gpu_tensor_free(heads);
    if (low) pulsar_gpu_tensor_free(low);
    free(hheads);
    free(hlow);
}

/* C. the refusals a wrong slice must produce. */
static void check_refusals(pulsar_engine *e, const pulsar_layer_weights *L) {
    const uint64_t q_rank = L->attn_q_a->dim[1];
    const uint64_t q_full = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    fprintf(stderr, "  (the two refusals below are EXPECTED -- negative controls)\n");
    const int off_band = pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(&e->model, L->attn_q_b),
                                                              L->attn_q_b->abs_offset, q_rank, q_full, 64, 64 + 4096);
    CHECK(off_band == 0, "a slice starting inside a 128-row scale band was ACCEPTED");
    if (off_band == 0) printf("  ok    a slice off the 128-row scale band is refused at registration\n");
    /* half the rows of the n=4 rank-0 slice check_q_b registered at [0, 16*512):
     * the same data offset, a row count nobody registered -- the resolver knows
     * no such slice and the whole tensor has other dims, so the GEMM refuses */
    uint32_t g_lo = 0, g_hi = 0;
    pulsar_tp_owned_range(0, 4u, PULSAR_N_OUT_GROUP, &g_lo, &g_hi);
    const uint64_t hi = (uint64_t)g_hi * (PULSAR_N_HEAD / PULSAR_N_OUT_GROUP) * PULSAR_N_HEAD_DIM;
    const uint32_t n_rows = 2;
    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)n_rows * q_rank * sizeof(float));
    pulsar_gpu_tensor *o = pulsar_gpu_tensor_alloc((uint64_t)n_rows * hi * sizeof(float));
    if (x && o) {
        float *hx = (float *)calloc((size_t)n_rows * q_rank, sizeof(float));
        if (hx) { pulsar_gpu_tensor_write(x, 0, hx, (uint64_t)n_rows * q_rank * sizeof(float)); free(hx); }
        pulsar_gpu_mxfp8_act_emit_f32(x, n_rows, q_rank);
        pulsar_gpu_matmul_set_batch_decode_rows((int)n_rows);
        const bool wrong = gpu_graph_matmul_mxfp8_rows_named_tensor("gate attn_q_b wrong dims", 0, 0, o, &e->model,
                                                                    L->attn_q_b, q_rank, q_full, 0, hi / 2, x, n_rows);
        pulsar_gpu_matmul_set_batch_decode_rows(0);
        CHECK(!wrong, "an unregistered row range at a registered slice's offset was ACCEPTED");
        if (!wrong) printf("  ok    an unregistered row range at a registered slice's offset is refused at the GEMM\n");
    } else {
        CHECK(0, "refusal probe allocation");
    }
    if (x) { pulsar_gpu_act_slot_drop(x); pulsar_gpu_tensor_free(x); }
    if (o) pulsar_gpu_tensor_free(o);
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    g_fail = 0;
    g_rng = 0x4C241u;
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.dspark_disable = true;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    const pulsar_layer_weights *L = &e->weights.layer[0];
    if (!L->attn_q_a || !L->attn_q_b || !L->attn_output_a) {
        fprintf(stderr, "layer 0 has no attention projections\n");
        gate_engine_close(e);
        return 1;
    }
    printf("tp head-split gate: layer 0, %u heads in %u output groups, q_rank %llu, lora_o %u\n",
           (unsigned)PULSAR_N_HEAD, (unsigned)PULSAR_N_OUT_GROUP,
           (unsigned long long)L->attn_q_a->dim[1], (unsigned)PULSAR_N_LORA_O);
    check_q_b(e, L, 3, 1);
    check_q_b(e, L, 6, 0);
    check_out_a(e, L, 3, 1);
    check_out_a(e, L, 6, 0);
    check_refusals(e, L);
    gate_engine_close(e);
    printf("TP HEAD-SPLIT GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

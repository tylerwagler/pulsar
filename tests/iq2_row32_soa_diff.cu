/* IQ2_XXS SoA dispatch differential.
 *
 * Built to chase the depth-4102 prefill-gate failure on a type-42 (SoA) model:
 * all 129280 logits non-finite.  It cleared the two suspects and caught the
 * real one.  This TU #includes the shipped .cu, so the file-static kernel
 * templates are visible and the REAL kernels are driven -- not copies.
 *
 * Same synthetic weights in both layouts (the SoA planes are an exact repack
 * of the packed buffer), so every SoA arm must be bit-identical to its packed
 * reference:
 *
 *   A  row32<0>              packed   reference for the tile8 arms
 *   B  row32<1>              SoA      == A  (was wrongly suspected)
 *   C  rowspan<32,1>         SoA      == A  (control: known-good arm)
 *   D  qwarp32<IQ2>          packed   reference for the qwarp32 arms
 *   E  qwarp32<IQ2SoA>       SoA      == D  (THE FIX)
 *   F  qwarp32<IQ2>          SoA      must DIFFER -- negative control
 *
 * F is the pre-fix dispatch: routed_moe_launch_mixed40's non-tiled branch had
 * no gate_soa arm, so a chunk with n_tokens < 128 (the 6-token tail of a 4102
 * prefill) read SoA planes with the PACKED reader -- the fp16 scale came out
 * of quantised weight bytes.  It reproduces as ~46k non-finite outputs, which
 * is the observed engine failure.  If F ever stops differing, this harness has
 * gone inert and proves nothing, so that case fails the run.
 *
 * Shape defaults to the shipped v5mx geometry (192 experts, mid_dim 2048), and
 * that matters: the bug is invisible at toy sizes.  Override: argv = n_tokens
 * [n_total_expert] [mid_dim].
 *
 * Build (needs the engine objects; this TU replaces pulsar_cuda_moe.o):
 *   OBJS=$(ls src/cuda/*.o src/engine/*.o | grep -v pulsar_cuda_moe.o)
 *   nvcc -O3 --use_fast_math -arch=sm_120f -DPULSAR_HC_F32 \
 *        -Isrc -Isrc/cuda -Isrc/engine -o /tmp/r32diff \
 *        tests/iq2_row32_soa_diff.cu $OBJS -lm -Xcompiler -pthread \
 *        -L$CUDA_HOME/targets/sbsa-linux/lib -lcudart -lcublas -lcublasLt
 */
#include "../src/cuda/pulsar_cuda_moe.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

/* Deliberately small: the failing shape is a SHORT chunk (few pairs, many
 * empty tiles), not a wide one.  in_dim stays 4096 so xq_blocks == 16, which
 * is the shipped value and the one that keeps the <=16 staging branch live. */
/* Shape defaults to the SHIPPED v5mx shape -- the small-shape run passed, so
 * the reproducer needs the real 192-expert / 2048-row geometry (d-plane offset
 * ~402 MB) rather than a toy one.  Override from argv to bisect which
 * dimension matters. */
static uint32_t N_EXPERT_TOTAL = 192;
static uint32_t N_EXPERT_USED  = 6;
static uint32_t IN_DIM         = 4096;
static uint32_t MID_DIM        = 2048;
static uint32_t XQ_BLOCKS, ROW_BLOCKS;
static uint64_t ROW_BYTES, EXPERT_BYTES, TOTAL_BYTES, NBLK_TOTAL, D_OFF;
static void shape_init(void) {
    XQ_BLOCKS    = IN_DIM / 256;
    ROW_BLOCKS   = IN_DIM / 256;
    ROW_BYTES    = (uint64_t)ROW_BLOCKS * 66ull;
    EXPERT_BYTES = (uint64_t)MID_DIM * ROW_BYTES;
    TOTAL_BYTES  = (uint64_t)N_EXPERT_TOTAL * EXPERT_BYTES;
    NBLK_TOTAL   = (uint64_t)N_EXPERT_TOTAL * MID_DIM * ROW_BLOCKS;
    D_OFF        = NBLK_TOTAL * 64ull;
}

static uint32_t rng = 12345u;
static uint32_t xrand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* AoS [u16 d][u16 qs[32]] -> q plane (64 B/blk) + d plane (2 B/blk). */
static void repack_soa(const uint8_t *packed, uint8_t *soa, uint64_t nblk) {
    uint16_t *dpl = (uint16_t *)(soa + nblk * 64ull);
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *src = packed + b * 66ull;
        memcpy(&dpl[b], src, 2);
        memcpy(soa + b * 64ull, src + 2, 64);
    }
}

int main(int argc, char **argv) {
    const uint32_t n_tokens = (argc > 1) ? (uint32_t)atoi(argv[1]) : 6u;
    if (argc > 2) N_EXPERT_TOTAL = (uint32_t)atoi(argv[2]);
    if (argc > 3) MID_DIM        = (uint32_t)atoi(argv[3]);
    shape_init();
    printf("# TOTAL_BYTES=%llu  NBLK_TOTAL=%llu  D_OFF=%llu\n",
           (unsigned long long)TOTAL_BYTES, (unsigned long long)NBLK_TOTAL, (unsigned long long)D_OFF);
    printf("# n_tokens=%u  mid_dim=%u  xq_blocks=%u  experts=%u/%u\n",
           n_tokens, MID_DIM, XQ_BLOCKS, N_EXPERT_USED, N_EXPERT_TOTAL);

    /* ---- weights ---------------------------------------------------------- */
    uint8_t *h_gate = (uint8_t *)malloc(TOTAL_BYTES);
    uint8_t *h_up   = (uint8_t *)malloc(TOTAL_BYTES);
    for (uint64_t i = 0; i < TOTAL_BYTES; i++) { h_gate[i] = (uint8_t)xrand(); h_up[i] = (uint8_t)xrand(); }
    /* keep the per-block fp16 scale small and finite so sums stay in range */
    for (uint64_t b = 0; b < NBLK_TOTAL; b++) {
        uint16_t d = 0x1c00u | (uint16_t)(xrand() & 0x03ffu);   /* ~1.0 .. 2.0 */
        memcpy(h_gate + b * 66ull, &d, 2);
        memcpy(h_up   + b * 66ull, &d, 2);
    }
    uint8_t *h_gate_soa = (uint8_t *)malloc(TOTAL_BYTES);
    uint8_t *h_up_soa   = (uint8_t *)malloc(TOTAL_BYTES);
    repack_soa(h_gate, h_gate_soa, NBLK_TOTAL);
    repack_soa(h_up,   h_up_soa,   NBLK_TOTAL);

    /* ---- activations ------------------------------------------------------ */
    const uint64_t n_xq = (uint64_t)n_tokens * XQ_BLOCKS;
    cuda_block_q8_K *h_xq = (cuda_block_q8_K *)malloc(n_xq * sizeof(cuda_block_q8_K));
    for (uint64_t i = 0; i < n_xq; i++) {
        h_xq[i].d = 0.01f + (float)(xrand() % 100) * 0.0001f;
        for (int j = 0; j < CUDA_QK_K; j++) h_xq[i].qs[j] = (int8_t)(xrand() & 0x7f) - 64;
        for (int j = 0; j < CUDA_QK_K / 16; j++) {
            int32_t s = 0;
            for (int k = 0; k < 16; k++) s += h_xq[i].qs[j * 16 + k];
            h_xq[i].bsums[j] = (int16_t)s;
        }
    }

    /* ---- routing (dense: every token hits N_EXPERT_USED distinct experts) -- */
    const uint32_t pair_count = n_tokens * N_EXPERT_USED;
    uint32_t *h_pairs = (uint32_t *)malloc(pair_count * sizeof(uint32_t));
    uint32_t *h_counts = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    uint32_t *h_off = (uint32_t *)calloc(N_EXPERT_TOTAL + 1, sizeof(uint32_t));
    uint32_t *exp_of = (uint32_t *)malloc(pair_count * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            uint32_t e = (t + s) % N_EXPERT_TOTAL;
            exp_of[p] = e; h_counts[e]++;
        }
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) h_off[e + 1] = h_off[e] + h_counts[e];
    uint32_t *cur = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            h_pairs[h_off[exp_of[p]] + cur[exp_of[p]]++] = t * N_EXPERT_USED + s;
        }

    /* ---- tiles (width 8, matching tile8) ---------------------------------- */
    uint32_t cap = 0;
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) cap += (h_counts[e] + 7u) / 8u;
    if (!cap) cap = 1;
    uint32_t *h_te = (uint32_t *)malloc(cap * sizeof(uint32_t));
    uint32_t *h_ts = (uint32_t *)malloc(cap * sizeof(uint32_t));
    uint32_t n_tiles = 0;
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++)
        for (uint32_t s = 0; s < h_counts[e]; s += 8u) { h_te[n_tiles] = e; h_ts[n_tiles] = s; n_tiles++; }
    printf("# tiles=%u pairs=%u\n", n_tiles, pair_count);

    float *h_w = (float *)malloc((uint64_t)n_tokens * N_EXPERT_USED * sizeof(float));
    for (uint32_t i = 0; i < n_tokens * N_EXPERT_USED; i++) h_w[i] = 0.1f + (float)(xrand() % 100) * 0.001f;

    /* ---- device ----------------------------------------------------------- */
    char *d_gate, *d_up, *d_gate_soa, *d_up_soa;
    cuda_block_q8_K *d_xq;
    uint32_t *d_pairs, *d_off, *d_counts, *d_total, *d_te, *d_ts;
    float *d_w, *d_outA, *d_outB, *d_outC, *d_outD, *d_outE, *d_outF;
    float *d_g1, *d_u1;
    int32_t *d_sel;
    const uint64_t out_n = (uint64_t)pair_count * MID_DIM;

    CK(cudaMalloc(&d_gate, TOTAL_BYTES));      CK(cudaMalloc(&d_up, TOTAL_BYTES));
    CK(cudaMalloc(&d_gate_soa, TOTAL_BYTES));  CK(cudaMalloc(&d_up_soa, TOTAL_BYTES));
    CK(cudaMalloc(&d_xq, n_xq * sizeof(cuda_block_q8_K)));
    CK(cudaMalloc(&d_pairs, pair_count * sizeof(uint32_t)));
    CK(cudaMalloc(&d_off, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t)));
    CK(cudaMalloc(&d_counts, N_EXPERT_TOTAL * sizeof(uint32_t)));
    CK(cudaMalloc(&d_total, sizeof(uint32_t)));
    CK(cudaMalloc(&d_te, cap * sizeof(uint32_t)));
    CK(cudaMalloc(&d_ts, cap * sizeof(uint32_t)));
    CK(cudaMalloc(&d_w, (uint64_t)n_tokens * N_EXPERT_USED * sizeof(float)));
    CK(cudaMalloc(&d_outA, out_n * sizeof(float)));
    CK(cudaMalloc(&d_outB, out_n * sizeof(float)));
    CK(cudaMalloc(&d_outC, out_n * sizeof(float)));
    CK(cudaMalloc(&d_outD, out_n * sizeof(float)));
    CK(cudaMalloc(&d_outE, out_n * sizeof(float)));
    CK(cudaMalloc(&d_outF, out_n * sizeof(float)));
    CK(cudaMalloc(&d_g1, out_n * sizeof(float)));
    CK(cudaMalloc(&d_u1, out_n * sizeof(float)));
    CK(cudaMalloc(&d_sel, (uint64_t)n_tokens * N_EXPERT_USED * sizeof(int32_t)));
    {   /* qwarp32 indexes weights/experts per (token, slot), not by sorted pair */
        int32_t *h_sel = (int32_t *)malloc((uint64_t)n_tokens * N_EXPERT_USED * sizeof(int32_t));
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t sl = 0; sl < N_EXPERT_USED; sl++)
                h_sel[t * N_EXPERT_USED + sl] = (int32_t)exp_of[t * N_EXPERT_USED + sl];
        CK(cudaMemcpy(d_sel, h_sel, (uint64_t)n_tokens * N_EXPERT_USED * sizeof(int32_t), cudaMemcpyHostToDevice));
        free(h_sel);
    }

    CK(cudaMemcpy(d_gate, h_gate, TOTAL_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_up, h_up, TOTAL_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gate_soa, h_gate_soa, TOTAL_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_up_soa, h_up_soa, TOTAL_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xq, h_xq, n_xq * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pairs, h_pairs, pair_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off, h_off, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_counts, h_counts, N_EXPERT_TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_total, &n_tiles, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_te, h_te, n_tiles * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ts, h_ts, n_tiles * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, h_w, (uint64_t)n_tokens * N_EXPERT_USED * sizeof(float), cudaMemcpyHostToDevice));

    CK(cudaMemset(d_outA, 0, out_n * sizeof(float)));
    CK(cudaMemset(d_outB, 0, out_n * sizeof(float)));
    CK(cudaMemset(d_outC, 0, out_n * sizeof(float)));
    CK(cudaMemset(d_outD, 0, out_n * sizeof(float)));
    CK(cudaMemset(d_outE, 0, out_n * sizeof(float)));
    CK(cudaMemset(d_outF, 0, out_n * sizeof(float)));

    dim3 g32((MID_DIM + 31u) / 32u, n_tiles);
    dim3 gsp((MID_DIM + 31u) / 32u, n_tiles);   /* ROW_SPAN=32 -> same grid */

    moe_gate_up_mid_expert_tile8_row32_kernel<0><<<g32, 256>>>(
        d_outA, d_gate, d_up, 0ull, 0ull, d_xq, d_pairs, d_off, d_counts,
        d_total, d_te, d_ts, d_w, EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    moe_gate_up_mid_expert_tile8_row32_kernel<1><<<g32, 256>>>(
        d_outB, d_gate_soa, d_up_soa, D_OFF, D_OFF, d_xq, d_pairs, d_off, d_counts,
        d_total, d_te, d_ts, d_w, EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    moe_gate_up_mid_expert_tile8_rowspan_kernel<32, 1><<<gsp, 256>>>(
        d_outC, d_gate_soa, d_up_soa, D_OFF, D_OFF, d_xq, d_pairs, d_off, d_counts,
        d_total, d_te, d_ts, d_w, EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    /* ---- qwarp32: the path routed_moe_launch_mixed40 takes for a SHORT chunk
     * (n_tokens < 128).  This is where the missing gate_soa arm lived. ------- */
    dim3 qgrid((MID_DIM + 127u) / 128u, pair_count, 1);
    moe_gate_up_mid_qwarp32_kernel<GateUpDotIQ2><<<qgrid, 256>>>(
        d_g1, d_u1, d_outD, d_gate, d_up, 0ull, 0ull, d_xq, d_sel, d_w,
        EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    moe_gate_up_mid_qwarp32_kernel<GateUpDotIQ2SoA><<<qgrid, 256>>>(
        d_g1, d_u1, d_outE, d_gate_soa, d_up_soa, D_OFF, D_OFF, d_xq, d_sel, d_w,
        EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    /* NEGATIVE CONTROL: the pre-fix dispatch -- packed reader pointed at SoA
     * planes.  Must NOT match; this is the defect being fixed. */
    moe_gate_up_mid_qwarp32_kernel<GateUpDotIQ2><<<qgrid, 256>>>(
        d_g1, d_u1, d_outF, d_gate_soa, d_up_soa, 0ull, 0ull, d_xq, d_sel, d_w,
        EXPERT_BYTES, ROW_BYTES, XQ_BLOCKS, MID_DIM, N_EXPERT_USED, 0.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    float *A = (float *)malloc(out_n * sizeof(float));
    float *B = (float *)malloc(out_n * sizeof(float));
    float *C = (float *)malloc(out_n * sizeof(float));
    CK(cudaMemcpy(A, d_outA, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(B, d_outB, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(C, d_outC, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    float *D = (float *)malloc(out_n * sizeof(float));
    float *E = (float *)malloc(out_n * sizeof(float));
    float *F = (float *)malloc(out_n * sizeof(float));
    CK(cudaMemcpy(D, d_outD, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(E, d_outE, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(F, d_outF, out_n * sizeof(float), cudaMemcpyDeviceToHost));

    struct Cmp { const char *name; const float *x; };
    Cmp cmps[3] = { { "row32<1>", B }, { "rowspan<32,1>", C }, { "qwarp32<IQ2SoA> (THE FIX)", E } };
    int fail = 0;
    for (int c = 0; c < 3; c++) {
        uint64_t ndiff = 0, nnan = 0, first = ~0ull;
        double maxrel = 0.0;
        const float *ref = (c == 2) ? D : A;   /* qwarp32 folds in a different order than tile8 */
        for (uint64_t i = 0; i < out_n; i++) {
            const float a = ref[i], b = cmps[c].x[i];
            if (!isfinite(b) && isfinite(a)) { nnan++; if (first == ~0ull) first = i; continue; }
            if (a != b) {
                ndiff++;
                if (first == ~0ull) first = i;
                const double den = fabs((double)a) > 1e-30 ? fabs((double)a) : 1e-30;
                const double rel = fabs((double)a - (double)b) / den;
                if (rel > maxrel) maxrel = rel;
            }
        }
        printf("%-26s ndiff=%llu nonfinite=%llu maxrel=%.3g",
               cmps[c].name, (unsigned long long)ndiff, (unsigned long long)nnan, maxrel);
        if (ndiff || nnan) {
            fail = 1;
            const uint64_t i = first;
            const uint32_t pair = (uint32_t)(i / MID_DIM), row = (uint32_t)(i % MID_DIM);
            printf("  FIRST i=%llu pair=%u row=%u tok=%u slot=%u  ref=%.9g got=%.9g",
                   (unsigned long long)i, pair, row, pair / N_EXPERT_USED, pair % N_EXPERT_USED,
                   ref[i], cmps[c].x[i]);
        } else {
            printf("  EXACT");
        }
        printf("\n");
    }
    /* negative control must differ -- if it matches, the harness is not
     * actually exercising the layout difference and proves nothing. */
    uint64_t nc_bad = 0, nc_nan = 0;
    for (uint64_t i = 0; i < out_n; i++) {
        if (!isfinite(F[i])) nc_nan++;
        else if (F[i] != D[i]) nc_bad++;
    }
    printf("%-26s differs=%llu nonfinite=%llu  %s\n", "packed-reader-on-SoA (BUG)",
           (unsigned long long)(nc_bad + nc_nan), (unsigned long long)nc_nan,
           (nc_bad + nc_nan) ? "REPRODUCED (expected)" : "*** NOT REPRODUCED -- harness inert ***");
    if (!(nc_bad + nc_nan)) fail = 1;
    printf("%s\n", fail ? "DIFF: FAIL" : "DIFF: PASS");
    return fail;
}

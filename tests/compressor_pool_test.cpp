/* HOST oracle for the 0731 Compressor's state machine and pooling.
 *
 * The compressor is the largest unrestored piece of the 0731 path (PLAN 96
 * s18.2) and its state machine is the part most likely to be wrong QUIETLY: a
 * mis-stashed carry still produces a plausible pooled row, just from the wrong
 * tokens.  So this is written oracle-first, the way the row codec and the row
 * packer were.
 *
 * This file is a CPU transcription of the SHIPPED reference
 * (/mnt/models/dsv4-flash-0731/inference/model.py, class Compressor) and is
 * graded against vectors from gate-baseline/l218-v41/compressor_ref.py.  It is
 * deliberately a transcription rather than a shared fixture: when the device
 * gate lands it must compare the KERNEL against this, and an oracle that shares
 * code with the thing under test can be wrong in the same direction.
 *
 * SCOPE: pooling + state only.  Not covered here, because they are separate
 * pieces with their own gates: RMSNorm, RoPE, the act_quant/Hadamard step, and
 * the cache write itself.
 *
 * What the vectors pin:
 *   ratio4      overlap (coff 2): a prefill with a partial group, a decode that
 *               does not cross a group boundary, then two that do -- the concat
 *               of the two halves and the post-compress state shift.
 *   ratio4short a prefill SHORTER than one group.  The carry was never written,
 *               so this is the case where the state's -inf/zero initialisation
 *               is what makes the first group correct at all.
 *   ratio128    the non-overlap path: no dim split, no shift, softmax over all
 *               128 positions.
 *
 * The one subtlety a plausible-looking transcription gets wrong: the softmax is
 * over the POSITION axis per head_dim column, not over the columns.  After
 * overlap_transform, column c comes from the previous group's first half while
 * column c+d comes from the current group's second half, so the two halves have
 * DIFFERENT weight vectors.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

#include "compressor_pool_fixture.h"

#define MAX_ROWS 256   /* coff(2) * ratio(128) */
#define MAX_W      8

static unsigned g_checks, g_failures, g_printed;
#define CHECK_DETAIL_MAX 12

static void check(int ok, const char *fmt, ...) {
    g_checks++;
    if (ok) return;
    g_failures++;
    if (g_printed++ >= CHECK_DETAIL_MAX) {
        if (g_printed == CHECK_DETAIL_MAX + 1) puts("FAIL: ... (further failures counted only)");
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fputs("FAIL: ", stdout);
    vprintf(fmt, ap);
    putchar('\n');
    va_end(ap);
}

/* The reference is f32; the vectors are f64.  The generated values are O(10), so
 * this is a relative-ish absolute bound well above f32 round-off (~1e-6) and far
 * below any real transcription error. */
#define TOL 2e-5f

static int close_enough(float a, float b) {
    if (isinf(a) || isinf(b)) return isinf(a) && isinf(b) && ((a < 0) == (b < 0));
    return fabsf(a - b) <= TOL;
}

/* ---- the transcription -------------------------------------------------- */

typedef struct {
    int ratio, coff, head, w;
    float kv[MAX_ROWS][MAX_W];
    float score[MAX_ROWS][MAX_W];
} cstate;

static void cstate_init(cstate *st, int ratio, int coff, int head) {
    st->ratio = ratio;
    st->coff = coff;
    st->head = head;
    st->w = coff * head;
    /* model.py:309-310 -- kv zero, score -inf. */
    for (int r = 0; r < MAX_ROWS; r++)
        for (int c = 0; c < MAX_W; c++) {
            st->kv[r][c] = 0.0f;
            st->score[r][c] = -INFINITY;
        }
}

static void linear(const float *x, const float *w, int n_out, int n_in, float *out) {
    for (int o = 0; o < n_out; o++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += w[o * n_in + k] * x[k];
        out[o] = acc;
    }
}

/* softmax over `npos` positions in column `col`, independently for each column.
 * `stride` is the row stride of the score rows -- MAX_W here, NOT w: the state
 * rows are padded to MAX_W while the generated fixture rows are packed. */
static void softmax_pos(const float *score, int stride, int npos, int col, float *wts) {
    float mx = -INFINITY;
    for (int r = 0; r < npos; r++) {
        const float v = score[r * stride + col];
        if (v != -INFINITY && v > mx) mx = v;
    }
    double sum = 0.0;
    for (int r = 0; r < npos; r++) {
        const float v = score[r * stride + col];
        wts[r] = (v == -INFINITY) ? 0.0f : expf(v - mx);
        sum += wts[r];
    }
    for (int r = 0; r < npos; r++) wts[r] = (float)(wts[r] / sum);
}

/* (kv * score.softmax(positions)).sum(positions), per column. */
static void pool(const float kv[][MAX_W], const float score[][MAX_W], int npos, int d, float *out) {
    static float wts[MAX_ROWS];
    for (int c = 0; c < d; c++) {
        softmax_pos(&score[0][0], MAX_W, npos, c, wts);
        double acc = 0.0;
        for (int r = 0; r < npos; r++) acc += (double)wts[r] * kv[r][c];
        out[c] = (float)acc;
    }
}

/* model.py:313.  [g][ratio][2d] -> [g][2*ratio][d], group 0's first half = value. */
static void overlap_transform(const float in[][MAX_W], int ngroup, int ratio, int d, float value,
                              float out[][MAX_W]) {
    for (int g = 0; g < ngroup; g++) {
        for (int r = 0; r < 2 * ratio; r++)
            for (int c = 0; c < d; c++) out[g * 2 * ratio + r][c] = value;
        for (int r = 0; r < ratio; r++)
            for (int c = 0; c < d; c++) {
                out[g * 2 * ratio + ratio + r][c] = in[g * ratio + r][d + c];
                if (g > 0) out[g * 2 * ratio + r][c] = in[(g - 1) * ratio + r][c];
            }
    }
}

/* Returns the number of pooled rows written (0 when the reference returns
 * early), and sets *first to the cache row the first one goes to. */
static int compressor_forward(const float *x, int seqlen, int start_pos, const float *wkv,
                              const float *wgate, const float *ape, cstate *st,
                              float out_rows[][MAX_W], int *first) {
    const int ratio = st->ratio, coff = st->coff, d = st->head, d2 = st->w;
    static float kv[MAX_ROWS][MAX_W], score[MAX_ROWS][MAX_W];
    static float gkv[MAX_ROWS][MAX_W], gsc[MAX_ROWS][MAX_W];
    static float tkv[MAX_ROWS][MAX_W], tsc[MAX_ROWS][MAX_W];

    for (int t = 0; t < seqlen; t++) {
        linear(x + t * 2, wkv, d2, 2, kv[t]);
        linear(x + t * 2, wgate, d2, 2, score[t]);
    }

    if (start_pos == 0) {
        const int should_compress = seqlen >= ratio;
        const int remainder = seqlen % ratio;
        const int cutoff = seqlen - remainder;
        const int offset = (coff == 2) ? ratio : 0;
        if (coff == 2 && cutoff >= ratio) {
            for (int r = 0; r < ratio; r++)
                for (int c = 0; c < d2; c++) {
                    st->kv[r][c] = kv[cutoff - ratio + r][c];
                    st->score[r][c] = score[cutoff - ratio + r][c] + ape[r * d2 + c];
                }
        }
        if (remainder > 0) {
            for (int r = 0; r < remainder; r++)
                for (int c = 0; c < d2; c++) {
                    st->kv[offset + r][c] = kv[cutoff + r][c];
                    st->score[offset + r][c] = score[cutoff + r][c] + ape[r * d2 + c];
                }
        }
        if (!should_compress) return 0;
        const int ngroup = cutoff / ratio;
        for (int g = 0; g < ngroup; g++)
            for (int r = 0; r < ratio; r++)
                for (int c = 0; c < d2; c++) {
                    gkv[g * ratio + r][c] = kv[g * ratio + r][c];
                    gsc[g * ratio + r][c] = score[g * ratio + r][c] + ape[r * d2 + c];
                }
        const int npos = ratio;
        if (coff == 2) {
            overlap_transform(gkv, ngroup, ratio, d, 0.0f, tkv);
            overlap_transform(gsc, ngroup, ratio, d, -INFINITY, tsc);
            for (int g = 0; g < ngroup; g++)
                pool(&tkv[g * 2 * ratio], &tsc[g * 2 * ratio], 2 * ratio, d, out_rows[g]);
        } else {
            for (int g = 0; g < ngroup; g++)
                pool(&gkv[g * ratio], &gsc[g * ratio], npos, d, out_rows[g]);
        }
        *first = 0;
        return ngroup;
    }

    const int should_compress = (start_pos + 1) % ratio == 0;
    const int slot = start_pos % ratio;
    float sc1[MAX_W];
    for (int c = 0; c < d2; c++) sc1[c] = score[0][c] + ape[slot * d2 + c];

    if (coff == 2) {
        for (int c = 0; c < d2; c++) {
            st->kv[ratio + slot][c] = kv[0][c];
            st->score[ratio + slot][c] = sc1[c];
        }
        if (!should_compress) return 0;
        for (int r = 0; r < ratio; r++)
            for (int c = 0; c < d; c++) {
                tkv[r][c] = st->kv[r][c];
                tsc[r][c] = st->score[r][c];
                tkv[ratio + r][c] = st->kv[ratio + r][d + c];
                tsc[ratio + r][c] = st->score[ratio + r][d + c];
            }
        pool(tkv, tsc, 2 * ratio, d, out_rows[0]);
        for (int r = 0; r < ratio; r++)
            for (int c = 0; c < d2; c++) {
                st->kv[r][c] = st->kv[ratio + r][c];
                st->score[r][c] = st->score[ratio + r][c];
            }
        *first = start_pos / ratio;
        return 1;
    }

    for (int c = 0; c < d2; c++) {
        st->kv[slot][c] = kv[0][c];
        st->score[slot][c] = sc1[c];
    }
    if (!should_compress) return 0;
    pool(st->kv, st->score, ratio, d, out_rows[0]);
    *first = start_pos / ratio;
    return 1;
}

/* ---- the gate ----------------------------------------------------------- */

static void run_case(const char *name, int ratio, int coff, int head, int w, int nstep,
                     const pulsar_comp_step *steps, const float *wkv, const float *wgate,
                     const float *ape) {
    cstate st;
    cstate_init(&st, ratio, coff, head);
    int nrows_total = 0;

    for (int s = 0; s < nstep; s++) {
        const pulsar_comp_step *stp = &steps[s];
        float rows[MAX_ROWS][MAX_W];
        int first = -1;
        memset(rows, 0, sizeof rows);

        const int n = compressor_forward(stp->x, stp->len, stp->start, wkv, wgate, ape, &st, rows, &first);

        check(n == stp->nrow, "%s step %d (start_pos %d, len %d): produced %d rows, reference %d",
              name, s, stp->start, stp->len, n, stp->nrow);
        check(n == 0 || first == stp->first,
              "%s step %d: first cache row %d, reference %d", name, s, first, stp->first);

        for (int r = 0; r < n && r < stp->nrow; r++)
            for (int c = 0; c < head; c++) {
                const float want = stp->row[r * head + c];
                const float got = rows[r][c];
                check(close_enough(got, want),
                      "%s step %d row %d col %d: got %.8g want %.8g", name, s, r, c, got, want);
            }
        nrows_total += n;

        /* the state after every step: this is the part that is silently wrong */
        const int nstate = coff * ratio;
        for (int r = 0; r < nstate; r++)
            for (int c = 0; c < w; c++) {
                check(close_enough(st.kv[r][c], stp->kv[r * w + c]),
                      "%s step %d: state.kv[%d][%d] got %.8g want %.8g",
                      name, s, r, c, st.kv[r][c], stp->kv[r * w + c]);
                check(close_enough(st.score[r][c], stp->sc[r * w + c]),
                      "%s step %d: state.score[%d][%d] got %.8g want %.8g",
                      name, s, r, c, st.score[r][c], stp->sc[r * w + c]);
            }
    }
    printf("--- %-12s ratio %-4d coff %d  %d steps, %d pooled rows\n",
           name, ratio, coff, nstep, nrows_total);
}

int main(void) {
    printf("compressor pool gate: 0731 state machine + pooling, vs the reference\n");

    static const pulsar_comp_step *const cases[3] = { RATIO4_STEPS, RATIO4SHORT_STEPS, RATIO128_STEPS };

    run_case("ratio4", 4, RATIO4_COFF, RATIO4_HEAD, RATIO4_W, RATIO4_NSTEP, cases[0],
             &RATIO4_WKV[0][0], &RATIO4_WGATE[0][0], &RATIO4_APE[0][0]);
    run_case("ratio4short", 4, RATIO4SHORT_COFF, RATIO4SHORT_HEAD, RATIO4SHORT_W, RATIO4SHORT_NSTEP, cases[1],
             &RATIO4SHORT_WKV[0][0], &RATIO4SHORT_WGATE[0][0], &RATIO4SHORT_APE[0][0]);
    run_case("ratio128", 128, RATIO128_COFF, RATIO128_HEAD, RATIO128_W, RATIO128_NSTEP, cases[2],
             &RATIO128_WKV[0][0], &RATIO128_WGATE[0][0], &RATIO128_APE[0][0]);

    /* The invariants that make the arithmetic meaningful, checked directly
     * rather than inferred from a passing vector compare. */
    check(RATIO4_COFF == 2 && RATIO128_COFF == 1,
          "overlap must be exactly (ratio == 4): coff 2 at 4, 1 at 128");
    check(RATIO4_W == 2 * RATIO4_HEAD && RATIO128_W == RATIO128_HEAD,
          "the compression width is coff * head_dim");

    printf("compressor pool gate: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

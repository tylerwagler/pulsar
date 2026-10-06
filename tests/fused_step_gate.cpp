/* L260 fusion phase B -- pulsar_session_decode_fused: verify rows and prompt
 * chunks in ONE forward.
 *
 * One fused step on a 4-bank pool:
 *   rows [0, 4)  decode / verify: bank 0 a 3-row run (a verify-shaped base +
 *                2 drafts) at C0A, bank 1 a 1-row run at C0B;
 *   then         two FINAL prefill runs from zero on reused (dirty, then
 *                invalidated) banks: bank 2 over [0, K) of prompt P2 and bank 3
 *                over [0, K3) of prompt P3, in BOTH orders.
 * Checks:
 *   (1) the 4 decode rows' logits are BYTE-IDENTICAL to the same 4 rows run as a
 *       decode-only step (PULSAR_MSEQ_HEAD_ALL_ROWS) -- the prompt rows behind
 *       them do not perturb the verify;
 *   (2) each prefill run's headed last row matches a one-chunk classic prefill of
 *       its prompt: next token exact, and rel-RMS within the prompt's own
 *       ENVELOPE -- max(1e-2, 2 x the largest rel-RMS between that classic prefill
 *       in one chunk and in two, split at K/2 and at K/2 + 2: a split on a
 *       compressor group boundary can be exactly neutral, one off it is not).  A fused run shares its forward with other rows, so its
 *       batch shape differs from the classic chunk's; batch-dependent numerics are
 *       allowed (L241) and graded by the reference gate, and on a sensitive prompt
 *       a LEGAL re-chunking alone moves the last row by several percent (this
 *       story's [2000, 2256): 5.5e-2).  The envelope is that legal spread; KV
 *       corruption lands far outside it;
 *   (3) each prefill bank's drafter prompt ring (the last <= 128 positions'
 *       anchor hiddens, per anchor layer) matches the one-chunk classic ring within
 *       the same kind of envelope, and its window bounds are exact -- a fused
 *       prompt drafts from the context a classic one does;
 *   (4) *out_n_rows == 6 (4 decode rows + 2 headed prefill runs).
 * K = 130 and 2048, K3 = 256.
 * LONG leg: bank 2 holds the story's [0, L_PRE) (classic), and ONE fused step
 * carries its continuation [L_PRE, L_PRE + KL) behind the 4 decode rows.  Past
 * the indexer's top-k the step's attention runs in prefill-slice spans, so spans
 * start INSIDE the prompt run: each span must see only ITS OWN decode rows (L260:
 * the spans after the first took the step's decode-row count as theirs, and the
 * prompt row at each span start was split-K'd and inverse-roped twice).  Checked
 * as (1)-(3) against the classic continuation over the same prefix, and SHARPER:
 *   (5) decode-row neutrality -- the continuation's headed row is BYTE-IDENTICAL
 *       to the SAME rows run as a mixed step with no decode rows in front (the
 *       dense and MoE dispatch run the decode prefix as its own call and attention
 *       is per row, so the decode rows must not move a prompt row).  A legal
 *       classic re-chunk moves this continuation's last row ~0.33, so (2)'s
 *       envelope cannot see a corrupted row; this does: before the per-span fix
 *       the row was 0.58 off both references (the ring, 0.37), after it 0 here,
 *       3.7e-4 from classic and the ring exact.  (The ring is not compared here:
 *       a plain mixed step does not fill the drafter's prompt ring -- (3) checks
 *       it against classic.)
 *   usage: PULSAR_MSEQ_BANKS=4 ./tests/fused_step_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define C0A   300
#define C0B   170
#define OFF_A 2400
#define OFF_B 2800
#define OFF_3 2000   /* bank 3's prompt P3: not P2, so no row of the step duplicates another */
#define K3    256
#define L_PRE 4096   /* ratio-4 sources: 1024 compressed rows > top-k 512 -> the span path */
#define KL    1024   /* spans of 512 start at batch rows 512 and 1024, inside the run */

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

static double rel_rms(const float *a, const float *ref, uint64_t n) {
    double se = 0, sr = 0;
    for (uint64_t i = 0; i < n; i++) { const double d = (double)a[i] - ref[i]; se += d * d; sr += (double)ref[i] * ref[i]; }
    return sr > 0 ? sqrt(se / sr) : (se > 0 ? 1e9 : 0.0);
}

/* one bank's ring, in floats -- PULSAR_N_EMBD is the loaded model's shape, so
 * this is read at run time, never at static init */
static uint64_t ring_floats(void) { return (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_N_EMBD; }

/* a bank's drafter prompt ring (3 anchor layers) and window bounds */
struct ring { float *h[3]; uint32_t lo, n; };
static void ring_free(ring *r) { for (int i = 0; i < 3; i++) { free(r->h[i]); r->h[i] = NULL; } }
static bool ring_read(pulsar_gpu_graph *g, uint32_t bank, ring *r) {
    for (int i = 0; i < 3; i++) {
        r->h[i] = (float *)malloc(ring_floats() * sizeof(float));
        if (!g->banks.dspark_prompt[i] ||
            !pulsar_gpu_tensor_read(g->banks.dspark_prompt[i], (uint64_t)bank * g->banks.dspark_prompt_bank_bytes,
                                    r->h[i], ring_floats() * sizeof(float))) {
            fprintf(stderr, "drafter ring read failed: bank %u anchor layer %d\n", bank, i);
            return false;
        }
    }
    r->lo = g->ms_dspark_prompt_lo[bank];
    r->n = g->ms_dspark_prompt_n[bank];
    return true;
}
/* worst anchor-layer rel-RMS over the ring slots the window [lo, n) holds */
static double ring_rel(const ring *a, const ring *ref) {
    const uint32_t win = PULSAR_DSPARK_DRAFT_WINDOW;
    const uint64_t row = PULSAR_N_EMBD;
    double worst = 0;
    for (int i = 0; i < 3; i++) {
        double se = 0, sr = 0;
        for (uint32_t p = ref->lo; p < ref->n; p++) {
            if (p + win < ref->n) continue;   /* only the last `win` positions are in the ring */
            const float *x = a->h[i] + (uint64_t)(p % win) * row, *y = ref->h[i] + (uint64_t)(p % win) * row;
            for (uint64_t k = 0; k < row; k++) { const double d = (double)x[k] - y[k]; se += d * d; sr += (double)y[k] * y[k]; }
        }
        const double r = sr > 0 ? sqrt(se / sr) : 0.0;
        if (r > worst) worst = r;
    }
    return worst;
}

/* the 4 decode rows */
static void decode_rows(pulsar_multiseq_req *rq) {
    for (int j = 0; j < 3; j++) { rq[j].bank = 0; rq[j].pos = C0A + j; rq[j].token = g_toks.v[OFF_A + C0A + j]; }
    rq[3].bank = 1; rq[3].pos = C0B; rq[3].token = g_toks.v[OFF_B + C0B];
}

static bool populate_decoders(pulsar_session *s) {
    return gate_populate_bank(s, 0, g_toks.v + OFF_A, C0A, NULL, "decoder A") &&
           gate_populate_bank(s, 1, g_toks.v + OFF_B, C0B, NULL, "decoder B");
}

/* reference (1): the 4 decode rows alone */
static bool decode_only(float *lg4) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    bool ok = gate_pool_fits(s, 4) && populate_decoders(s);
    pulsar_multiseq_req rq[4];
    decode_rows(rq);
    char e[256];
    uint32_t nr = 0;
    if (ok && pulsar_session_decode_mixed(s, rq, 4, lg4, 4 * (int)PULSAR_N_VOCAB, &nr, PULSAR_MSEQ_HEAD_ALL_ROWS,
                                          e, sizeof e) != 0) {
        fprintf(stderr, "decode-only reference: %s\n", e);
        ok = false;
    }
    if (ok && nr != 4) { fprintf(stderr, "decode-only reference returned %u rows\n", nr); ok = false; }
    pulsar_session_free(s);
    return ok;
}

/* a prompt's classic reference: the one-chunk prefill's last row and ring, and
 * its ENVELOPE -- how far the same prompt moves when prefilled in two chunks
 * instead of one, the larger over a split at K/2 and at K/2 + 2 */
struct classic_ref { float *lg; ring r; double env_lg, env_ring; };
/* `pre` > 0: the prompt is [off, off + pre + K), the first `pre` tokens prefilled
 * as their own sync and the K-token continuation measured */
static bool classic_split(int off, int K, int split, const classic_ref *c, double *dl, double *dr, int pre = 0) {
    const uint64_t V = PULSAR_N_VOCAB;
    float *two = (float *)malloc(V * sizeof(float));
    ring r2 = {};
    char e[256];
    pulsar_session *s = NULL;
    pulsar_tokens t = { .v = g_toks.v + off, .len = pre + K, .cap = pre + K };
    bool ok = pulsar_session_create(&s, g_e, 8192) == 0 &&
              gate_populate_bank(s, 0, g_toks.v + off, pre > 0 ? pre : split, NULL, "classic two chunks");
    if (ok && pre > 0 && !gate_prefill_suffix_classic(s, &t, pre, pre + split, e, sizeof e)) {
        fprintf(stderr, "%s\n", e);
        ok = false;
    }
    if (ok && !gate_prefill_suffix_classic(s, &t, pre + split, pre + K, e, sizeof e)) { fprintf(stderr, "%s\n", e); ok = false; }
    if (ok) {
        gpu_graph_bank_counters_capture(&s->graph, 0);
        pulsar_session_copy_logits(s, two, (int)V);
        ok = ring_read(&s->graph, 0, &r2);
    }
    if (ok) {
        *dl = rel_rms(two, c->lg, V);
        *dr = ring_rel(&r2, &c->r);
    }
    if (s) pulsar_session_free(s);
    ring_free(&r2);
    free(two);
    return ok;
}
static bool classic_reference(int off, int K, classic_ref *c, int pre = 0) {
    const uint64_t V = PULSAR_N_VOCAB;
    c->lg = (float *)malloc(V * sizeof(float));
    pulsar_session *s = NULL;
    pulsar_tokens t = { .v = g_toks.v + off, .len = pre + K, .cap = pre + K };
    char e[256];
    bool ok = pulsar_session_create(&s, g_e, 8192) == 0 && gate_pool_fits(s, 4) &&
              gate_populate_bank(s, 0, g_toks.v + off, pre > 0 ? pre : K, NULL, "classic one chunk");
    if (ok && pre > 0 && !gate_prefill_suffix_classic(s, &t, pre, pre + K, e, sizeof e)) {
        fprintf(stderr, "%s\n", e);
        ok = false;
    }
    if (ok) {
        gpu_graph_bank_counters_capture(&s->graph, 0);
        pulsar_session_copy_logits(s, c->lg, (int)V);
        ok = ring_read(&s->graph, 0, &c->r);
    }
    if (s) pulsar_session_free(s);
    c->env_lg = c->env_ring = 0;
    const int splits[2] = {K / 2, K / 2 + 2};
    for (int i = 0; ok && i < 2; i++) {
        double dl = 0, dr = 0;
        ok = classic_split(off, K, splits[i], c, &dl, &dr, pre);
        if (dl > c->env_lg) c->env_lg = dl;
        if (dr > c->env_ring) c->env_ring = dr;
    }
    return ok;
}

/* the fused step; order 0 = bank 2's run first, 1 = bank 3's first */
static bool fused(int K, int order, float *lg, uint32_t *nr, ring *r2, ring *r3) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    /* banks 2 and 3 first hold another conversation, then are invalidated: the
     * served shape (a reused bank) */
    bool ok = gate_pool_fits(s, 4) &&
              gate_populate_bank(s, 2, g_toks.v + 600, 900, NULL, "dirty bank 2") &&
              gate_populate_bank(s, 3, g_toks.v + 1500, 700, NULL, "dirty bank 3") &&
              populate_decoders(s);
    for (uint32_t b = 2; ok && b < 4; b++) {
        if (!gpu_graph_bank_repoint(g, b)) { fprintf(stderr, "bank %u repoint failed\n", b); ok = false; break; }
        pulsar_session_invalidate(s);
        gpu_graph_bank_counters_capture(g, b);
    }
    const uint32_t n = 4 + (uint32_t)K + K3;
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc(n * sizeof(*rq));
    decode_rows(rq);
    const int at2 = order == 0 ? 4 : 4 + K3, at3 = order == 0 ? 4 + K : 4;
    for (int j = 0; j < K; j++) { rq[at2 + j].bank = 2; rq[at2 + j].pos = j; rq[at2 + j].token = g_toks.v[j]; }
    for (int j = 0; j < K3; j++) { rq[at3 + j].bank = 3; rq[at3 + j].pos = j; rq[at3 + j].token = g_toks.v[OFF_3 + j]; }
    pulsar_fused_shape sh;
    memset(&sh, 0, sizeof sh);
    sh.n_dec = 4;
    sh.n_pf = 2;
    sh.head_last[0] = 1;
    sh.head_last[1] = 1;
    char e[256];
    float *raw = (float *)malloc(6 * (size_t)PULSAR_N_VOCAB * sizeof(float));
    if (ok && pulsar_session_decode_fused(s, rq, n, &sh, raw, 6 * (int)PULSAR_N_VOCAB, nr, e, sizeof e) != 0) {
        fprintf(stderr, "fused step: %s\n", e);
        ok = false;
    }
    /* lg: the decode rows, then bank 2's row, then bank 3's -- whatever the order */
    const uint64_t V = PULSAR_N_VOCAB;
    if (ok) {
        memcpy(lg, raw, 4 * V * sizeof(float));
        memcpy(lg + 4 * V, raw + (order == 0 ? 4 : 5) * V, V * sizeof(float));
        memcpy(lg + 5 * V, raw + (order == 0 ? 5 : 4) * V, V * sizeof(float));
    }
    if (ok) ok = ring_read(g, 2, r2) && ring_read(g, 3, r3);
    free(raw);
    free(rq);
    pulsar_session_free(s);
    return ok;
}

/* the LONG leg's fused step: the 4 decode rows, then bank 2's continuation
 * [L_PRE, L_PRE + KL) over its classic prefix */
static bool fused_long(float *lg, uint32_t *nr, ring *r2) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 8192) != 0) return false;
    bool ok = gate_pool_fits(s, 4) &&
              gate_populate_bank(s, 2, g_toks.v, L_PRE, NULL, "long prefix bank 2") &&
              populate_decoders(s);
    const uint32_t n = 4 + KL;
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc(n * sizeof(*rq));
    decode_rows(rq);
    for (int j = 0; j < KL; j++) { rq[4 + j].bank = 2; rq[4 + j].pos = L_PRE + j; rq[4 + j].token = g_toks.v[L_PRE + j]; }
    pulsar_fused_shape sh;
    memset(&sh, 0, sizeof sh);
    sh.n_dec = 4;
    sh.n_pf = 1;
    sh.head_last[0] = 1;
    char e[256];
    if (ok && pulsar_session_decode_fused(s, rq, n, &sh, lg, 5 * (int)PULSAR_N_VOCAB, nr, e, sizeof e) != 0) {
        fprintf(stderr, "long fused step: %s\n", e);
        ok = false;
    }
    if (ok) ok = ring_read(&s->graph, 2, r2);
    free(rq);
    pulsar_session_free(s);
    return ok;
}

/* (5)'s reference: the continuation alone, as a mixed step */
static bool prompt_only_long(float *lg1) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 8192) != 0) return false;
    bool ok = gate_pool_fits(s, 4) &&
              gate_populate_bank(s, 2, g_toks.v, L_PRE, NULL, "long prefix bank 2") &&
              populate_decoders(s);
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc(KL * sizeof(*rq));
    for (int j = 0; j < KL; j++) { rq[j].bank = 2; rq[j].pos = L_PRE + j; rq[j].token = g_toks.v[L_PRE + j]; }
    char e[256];
    uint32_t nr = 0;
    if (ok && pulsar_session_decode_mixed(s, rq, KL, lg1, (int)PULSAR_N_VOCAB, &nr, 1u, e, sizeof e) != 0) {
        fprintf(stderr, "prompt-only long step: %s\n", e);
        ok = false;
    }
    if (ok && nr != 1) { fprintf(stderr, "prompt-only long step headed %u rows\n", nr); ok = false; }
    free(rq);
    pulsar_session_free(s);
    return ok;
}

static void check_prompt(const char *what, const float *row, const ring *rg, const classic_ref *c, uint32_t end) {
    const uint64_t V = PULSAR_N_VOCAB;
    const int a = (int)argmax_f32(row, V), b = (int)argmax_f32(c->lg, V);
    const double r = rel_rms(row, c->lg, V), lim = fmax(1e-2, 2.0 * c->env_lg);
    printf("%s FINAL ROW vs classic: next %d vs %d %s | rel-RMS %.3e (envelope %.3e, limit %.3e)\n", what, a, b,
           a == b ? "MATCH" : "MISMATCH", r, c->env_lg, lim);
    if (a != b || r > lim) { fprintf(stderr, "GATE FAIL: %s final row\n", what); g_fail = 1; }
    const double rr = ring_rel(rg, &c->r), rlim = fmax(1e-2, 2.0 * c->env_ring);
    const bool bounds = rg->lo == c->r.lo && rg->n == end && c->r.n == end;
    printf("%s DRAFTER RING: window [%u, %u) vs classic [%u, %u) %s | rel-RMS %.3e (envelope %.3e, limit %.3e)\n",
           what, rg->lo, rg->n, c->r.lo, c->r.n, bounds ? "EXACT" : "MISMATCH", rr, c->env_ring, rlim);
    if (!bounds || rr > rlim) { fprintf(stderr, "GATE FAIL: %s drafter ring\n", what); g_fail = 1; }
}

int GATE_ENTRY(int argc, char **argv) {
    g_e = NULL;
    memset(&g_toks, 0, sizeof g_toks);
    g_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options o;
    memset(&o, 0, sizeof o);
    o.model_path = argv[1];
    o.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&g_e, &o) != 0 || !pulsar_engine_has_spec_rounds(g_e)) {
        fprintf(stderr, "engine open failed or no drafter\n");
        return 1;
    }
    if (!gate_load_story(g_e, &g_toks, L_PRE + KL)) { gate_engine_close(g_e); return 1; }
    const uint64_t V = PULSAR_N_VOCAB;
    float *dec_ref = (float *)malloc(4 * V * sizeof(float));
    float *lg = (float *)malloc(6 * V * sizeof(float));
    if (!decode_only(dec_ref)) { fprintf(stderr, "GATE FAIL: decode-only reference\n"); g_fail = 1; }
    classic_ref c3 = {};
    if (!g_fail && !classic_reference(OFF_3, K3, &c3)) { fprintf(stderr, "GATE FAIL: classic P3 reference\n"); g_fail = 1; }
    const int Ks[2] = {130, 2048};
    for (int ki = 0; ki < 2 && !g_fail; ki++) {
        const int K = Ks[ki];
        classic_ref c2 = {};
        if (!classic_reference(0, K, &c2)) { fprintf(stderr, "GATE FAIL: classic P2 K=%d reference\n", K); g_fail = 1; break; }
        for (int order = 0; order < 2 && !g_fail; order++) {
            ring r2 = {}, r3 = {};
            uint32_t nr = 0;
            const char *ord = order == 0 ? "P2,P3" : "P3,P2";
            if (!fused(K, order, lg, &nr, &r2, &r3)) {
                fprintf(stderr, "GATE FAIL: fused step K=%d order %s\n", K, ord);
                g_fail = 1;
                break;
            }
            if (nr != 6) { fprintf(stderr, "GATE FAIL: K=%d out_n_rows %u, want 6\n", K, nr); g_fail = 1; }
            const long fd = gate_first_diff(lg, dec_ref, (long)(4 * V));
            printf("K=%d %s DECODE ROWS vs decode-only step: %s (first diff %ld)\n", K, ord,
                   fd < 0 ? "BYTE-IDENTICAL" : "DIFFER", fd);
            if (fd >= 0) { fprintf(stderr, "GATE FAIL: K=%d the prompt rows perturbed the verify rows\n", K); g_fail = 1; }
            char what[48];
            snprintf(what, sizeof what, "K=%d %s bank 2 (P2, %d)", K, ord, K);
            check_prompt(what, lg + 4 * V, &r2, &c2, (uint32_t)K);
            snprintf(what, sizeof what, "K=%d %s bank 3 (P3, %d)", K, ord, K3);
            check_prompt(what, lg + 5 * V, &r3, &c3, K3);
            ring_free(&r2);
            ring_free(&r3);
        }
        ring_free(&c2.r);
        free(c2.lg);
    }
    if (!g_fail) {
        classic_ref cl = {};
        ring rl = {};
        uint32_t nr = 0;
        if (!classic_reference(0, KL, &cl, L_PRE)) { fprintf(stderr, "GATE FAIL: classic LONG reference\n"); g_fail = 1; }
        else if (!fused_long(lg, &nr, &rl)) { fprintf(stderr, "GATE FAIL: LONG fused step\n"); g_fail = 1; }
        else {
            if (nr != 5) { fprintf(stderr, "GATE FAIL: LONG out_n_rows %u, want 5\n", nr); g_fail = 1; }
            const long fd = gate_first_diff(lg, dec_ref, (long)(4 * V));
            printf("LONG DECODE ROWS vs decode-only step: %s (first diff %ld)\n",
                   fd < 0 ? "BYTE-IDENTICAL" : "DIFFER", fd);
            if (fd >= 0) { fprintf(stderr, "GATE FAIL: LONG the prompt rows perturbed the verify rows\n"); g_fail = 1; }
            char what[64];
            snprintf(what, sizeof what, "LONG bank 2 ([%d, %d) over a classic prefix)", L_PRE, L_PRE + KL);
            check_prompt(what, lg + 4 * V, &rl, &cl, (uint32_t)(L_PRE + KL));
            float *lg1 = (float *)malloc(V * sizeof(float));
            if (!prompt_only_long(lg1)) { fprintf(stderr, "GATE FAIL: LONG prompt-only reference\n"); g_fail = 1; }
            else {
                const long pd = gate_first_diff(lg + 4 * V, lg1, (long)V);
                printf("LONG DECODE-ROW NEUTRALITY: headed row vs the same rows without decode rows: %s "
                       "(first diff %ld, rel-RMS %.3e)\n",
                       pd < 0 ? "BYTE-IDENTICAL" : "DIFFER", pd, rel_rms(lg + 4 * V, lg1, V));
                if (pd >= 0) {
                    fprintf(stderr, "GATE FAIL: LONG the decode rows moved the prompt rows\n");
                    g_fail = 1;
                }
            }
            free(lg1);
        }
        ring_free(&rl);
        ring_free(&cl.r);
        free(cl.lg);
    }
    ring_free(&c3.r);
    free(c3.lg);
    free(dec_ref);
    free(lg);
    pulsar_tokens_free(&g_toks);
    gate_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "FUSED-STEP GATE: FAIL\n"); return 1; }
    printf("FUSED-STEP GATE: PASS\n");
    return 0;
}

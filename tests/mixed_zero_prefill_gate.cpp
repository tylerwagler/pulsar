/* L260 fusion phase A -- a FROM-ZERO prefill run inside a mixed batch.
 *
 * The spec lane's fused step carries a queued prompt's chunks in the same
 * pulsar_session_decode_mixed forward as the live decode rows, and a prompt's
 * FIRST chunk starts at position 0.  This gate co-schedules, in ONE mixed step,
 * a decode row of bank 0 (prefilled classically to C0) and a K-row prefill run
 * of a fresh bank 1 at positions [0, K), and checks:
 *
 *   (1) bank 1 COHERENT vs a classic from-zero prefill of the same K tokens:
 *       next token exact, last-position full-vocab logit rel-RMS < 1e-2 (the
 *       mixed_prefill_gate oracle -- KV corruption is large, last-ulp drift is
 *       not), and its greedy continuation coherent (reported, first token must
 *       match);
 *   (2) bank 0's decode row NEUTRAL: its logits equal the same row decoded alone
 *       (argmax exact, rel-RMS < 1e-3; byte identity reported).
 *
 * K = 130 (not ratio-aligned) and 2048 (a prefill chunk), each on a CLEAN bank 1
 * and on a DIRTY one -- bank 1 first holds 2,500 tokens of another conversation,
 * then is invalidated, as a server bank is reused: a per-row arm that read state
 * below position 0 (raw ring, compressor carry, KV) would pass clean and fail
 * dirty.
 *   usage: PULSAR_MSEQ_BANKS=2 ./tests/mixed_zero_prefill_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "gate_entry.h"
#include "gate_fixture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NGEN 24
#define C0   130

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

static double rel_rms(const float *a, const float *ref, int n) {
    double se = 0, sr = 0;
    for (int i = 0; i < n; i++) { const double d = (double)a[i] - ref[i]; se += d * d; sr += (double)ref[i] * ref[i]; }
    return sr > 0 ? sqrt(se / sr) : (se > 0 ? 1e9 : 0.0);
}

/* greedy-decode NGEN tokens on `bank`, frontier F, out[0] already the first token */
static bool decode_cont(pulsar_session *s, uint32_t bank, int F, int *out) {
    const int vocab = (int)PULSAR_N_VOCAB;
    float *lg = (float *)malloc((size_t)vocab * sizeof(float));
    char e[256];
    bool ok = true;
    for (int i = 1; i < NGEN && ok; i++) {
        pulsar_multiseq_req r = {.bank = bank, .pos = F + i - 1, .token = out[i - 1]};
        uint32_t nr = 0;
        if (pulsar_session_decode_mixed(s, &r, 1, lg, vocab, &nr, 0u, e, sizeof e) != 0) {
            fprintf(stderr, "cont step %d: %s\n", i, e);
            ok = false;
            break;
        }
        out[i] = (int)argmax_f32(lg, (uint64_t)vocab);
    }
    free(lg);
    return ok;
}

/* classic from-zero prefill of [0, K) on bank 0, then decode NGEN */
static bool classic_ref(int K, float *lg_out, int *out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    int t0 = -1;
    bool ok = gate_populate_bank(s, 0, g_toks.v, K, &t0, "classic ref");
    if (ok) {
        pulsar_session_copy_logits(s, lg_out, (int)PULSAR_N_VOCAB);
        out[0] = t0;
        ok = decode_cont(s, 0, K, out);
    }
    pulsar_session_free(s);
    return ok;
}

/* bank 0's decode row at C0 alone (the neutrality reference) */
static bool solo_decode_row(float *lg_out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    bool ok = gate_populate_bank(s, 0, g_toks.v + 1000, C0, NULL, "solo partner");
    char e[256];
    uint32_t nr = 0;
    pulsar_multiseq_req r = {.bank = 0, .pos = C0, .token = g_toks.v[1000 + C0]};
    if (ok && pulsar_session_decode_mixed(s, &r, 1, lg_out, (int)PULSAR_N_VOCAB, &nr, 0u, e, sizeof e) != 0) {
        fprintf(stderr, "solo decode row: %s\n", e);
        ok = false;
    }
    pulsar_session_free(s);
    return ok;
}

/* ONE mixed step: bank 0 decode row at C0, then bank 1's K-row run from 0. */
static bool mixed_zero(int K, bool dirty, float *dec_lg, float *pf_lg, int *out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    bool ok = gate_pool_fits(s, 2);
    if (ok && dirty) ok = gate_populate_bank(s, 1, g_toks.v + 500, 2500, NULL, "dirty bank 1");
    if (ok) ok = gate_populate_bank(s, 0, g_toks.v + 1000, C0, NULL, "partner");
    if (ok && !gpu_graph_bank_repoint(g, 1)) { fprintf(stderr, "bank 1 repoint failed\n"); ok = false; }
    if (ok) {
        pulsar_session_invalidate(s);
        gpu_graph_bank_counters_capture(g, 1);   /* bank 1: a fresh, empty frontier */
    }
    const int vocab = (int)PULSAR_N_VOCAB;
    float *lg = (float *)malloc(2 * (size_t)vocab * sizeof(float));
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc((size_t)(K + 1) * sizeof(*rq));
    rq[0].bank = 0; rq[0].pos = C0; rq[0].token = g_toks.v[1000 + C0];
    for (int j = 0; j < K; j++) { rq[1 + j].bank = 1; rq[1 + j].pos = j; rq[1 + j].token = g_toks.v[j]; }
    char e[256];
    uint32_t nr = 0;
    if (ok && pulsar_session_decode_mixed(s, rq, (uint32_t)(K + 1), lg, 2 * vocab, &nr, 0u, e, sizeof e) != 0) {
        fprintf(stderr, "mixed step (decode + from-zero K=%d): %s\n", K, e);
        ok = false;
    }
    if (ok && nr != 2) { fprintf(stderr, "mixed step returned %u logit rows, want 2\n", nr); ok = false; }
    if (ok) {
        memcpy(dec_lg, lg, (size_t)vocab * sizeof(float));
        memcpy(pf_lg, lg + vocab, (size_t)vocab * sizeof(float));
        out[0] = (int)argmax_f32(pf_lg, (uint64_t)vocab);
        ok = decode_cont(s, 1, K, out);
    }
    free(lg);
    free(rq);
    pulsar_session_free(s);
    return ok;
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
    if (gate_engine_open(&g_e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!gate_load_story(g_e, &g_toks, 3000 + NGEN)) { gate_engine_close(g_e); return 1; }
    const int vocab = (int)PULSAR_N_VOCAB;
    float *ref_lg = (float *)malloc((size_t)vocab * sizeof(float));
    float *pf_lg = (float *)malloc((size_t)vocab * sizeof(float));
    float *dec_lg = (float *)malloc((size_t)vocab * sizeof(float));
    float *solo_lg = (float *)malloc((size_t)vocab * sizeof(float));
    if (!solo_decode_row(solo_lg)) { fprintf(stderr, "GATE FAIL: solo decode reference\n"); g_fail = 1; }
    const int Ks[2] = {130, 2048};
    for (int vi = 0; vi < 4 && !g_fail; vi++) {
        const int K = Ks[vi % 2];
        const bool dirty = vi >= 2;
        const char *bk = dirty ? "dirty" : "clean";
        int ref[NGEN], mix[NGEN];
        if (!classic_ref(K, ref_lg, ref)) { fprintf(stderr, "GATE FAIL: classic from-zero K=%d\n", K); g_fail = 1; break; }
        if (!mixed_zero(K, dirty, dec_lg, pf_lg, mix)) { fprintf(stderr, "GATE FAIL: mixed from-zero K=%d (%s bank)\n", K, bk); g_fail = 1; break; }
        const double r1 = rel_rms(pf_lg, ref_lg, vocab);
        int first_div = -1;
        for (int i = 0; i < NGEN; i++) if (mix[i] != ref[i]) { first_div = i; break; }
        printf("K=%d %s PREFILL-FROM-ZERO: next %d vs classic %d %s | last-pos rel-RMS %.3e | continuation %s (first diff %d of %d)\n",
               K, bk, mix[0], ref[0], mix[0] == ref[0] ? "MATCH" : "MISMATCH", r1,
               first_div < 0 ? "identical" : "diverges", first_div, NGEN);
        if (mix[0] != ref[0] || r1 >= 1e-2) { fprintf(stderr, "GATE FAIL: K=%d %s bank 1 not coherent with classic from zero\n", K, bk); g_fail = 1; }
        const double r0 = rel_rms(dec_lg, solo_lg, vocab);
        const long fd = gate_first_diff(dec_lg, solo_lg, vocab);
        const int a0 = (int)argmax_f32(dec_lg, (uint64_t)vocab), as = (int)argmax_f32(solo_lg, (uint64_t)vocab);
        printf("K=%d %s DECODE NEUTRALITY: argmax %d vs solo %d %s | rel-RMS %.3e | %s\n", K, bk, a0, as,
               a0 == as ? "MATCH" : "MISMATCH", r0, fd < 0 ? "byte-identical" : "not byte-identical");
        if (a0 != as || r0 >= 1e-3) { fprintf(stderr, "GATE FAIL: K=%d co-scheduled decode row perturbed\n", K); g_fail = 1; }
    }
    free(ref_lg); free(pf_lg); free(dec_lg); free(solo_lg);
    pulsar_tokens_free(&g_toks);
    gate_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "MIXED-ZERO-PREFILL GATE: FAIL\n"); return 1; }
    printf("MIXED-ZERO-PREFILL GATE: PASS\n");
    return 0;
}

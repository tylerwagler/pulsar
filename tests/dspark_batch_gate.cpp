/* L150 GATE: the batched redraft equals the serialized redraft, per bank, byte
 * for byte.
 *
 * MODEL-DEPENDENT (drafter-merged model), run on the GB10 via
 * `make cuda-dspark-batch-gate`. Drives the SERVER's batched speculative lane
 * exactly as server_sched.cpp does -- per bank round_begin under its restore,
 * ONE shared decode_mixed ALL_ROWS forward, per bank round_end (which, in this
 * lane, DEFERS the drafter) -- and then redrafts the same rounds twice on the
 * same verified state:
 *   (a) SERIALIZED: pulsar_session_spec_redraft_batch on each bank alone;
 *   (b) BATCHED:    one call over all banks;
 * and requires the per-bank draft ids, confidence scores (float bits) and
 * trimmed pending counts to be IDENTICAL. Then commits (b) and moves on, for
 * several ticks. The drafts are transient device state, so (a) and (b) read
 * the same rings and anchors; each bank draws from its own rng, and the two
 * passes start from equal rng copies, so the sampled path must match too.
 *
 * Runs two shapes: all banks greedy, and bank 0 greedy with banks 1-2 sampled
 * at temperature 1.0 / min_p 0.05 (the production shape; the mixed set
 * exercises the greedy-first / sampled-second grouping). argv[3] = draft depth
 * (0 = engine default) so the Makefile can also pin depth 1, whose attention
 * takes a distinct single-row fast path.
 *
 * WHY THIS AND NOT THE MULTISEQ FINGERPRINTS: with exact verification the
 * emitted tokens never depend on the drafts, so a token-stream gate cannot see
 * a wrong drafter. Only the drafts themselves can. */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            fprintf(stderr, "DSPARK-BATCH GATE FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            g_fail = 1; \
        } \
    } while (0)

enum { NB = 3, ROWS = 32 };
static int g_nb = NB;   /* banks in play this run (argv[4], <= NB) */
static const int g_prompt_off[NB] = {0, 401, 700};
static const int g_prompt_len[NB] = {130, 258, 160};

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bool bank_prefill(pulsar_session *s, int k) {
    pulsar_gpu_graph *g = &s->graph;
    char err[256];
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) return false;
    pulsar_session_invalidate(s);
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    const int off = g_prompt_off[k], len = g_prompt_len[k];
    if (off + len > g_toks.len) return false;
    p.v = (int *)malloc((size_t)len * sizeof(int));
    memcpy(p.v, g_toks.v + off, (size_t)len * sizeof(int));
    p.len = p.cap = len;
    bool ok = pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
    if (!ok) fprintf(stderr, "bank %d prefill failed: %s\n", k, err);
    if (ok) pulsar_session_bank_state_save(s, (uint32_t)k);
    pulsar_tokens_free(&p);
    return ok;
}

typedef struct {
    int      present;
    uint32_t n_draft, keep;
    int      sampled;
    int32_t  ids[17];
    float    conf[16];
} draft_result;

static void peek_all(pulsar_spec_round **r, int n, draft_result *out) {
    for (int b = 0; b < n; b++) {
        memset(&out[b], 0, sizeof(out[b]));
        out[b].present = pulsar_session_spec_redraft_peek(r[b], out[b].ids, out[b].conf,
                                                         &out[b].n_draft, &out[b].keep,
                                                         &out[b].sampled);
    }
}

/* One server-shaped tick up to (and including) the deferred round_end. Returns
 * the number of banks still live, or -1. */
static int tick_to_round_end(pulsar_session *s, pulsar_spec_round **r, const float *temps,
                             uint64_t *rngs, int *first_tok, uint32_t *row0, float *logits,
                             int vocab) {
    char err[256];
    pulsar_multiseq_req reqs[ROWS];
    uint32_t rows = 0;
    const int eos = pulsar_token_eos(g_e);
    for (int b = 0; b < g_nb; b++) {
        if (!pulsar_session_bank_state_restore(s, (uint32_t)b)) return -1;
        const int first = pulsar_session_spec_next_base(s, temps[b], 0, 1.0f, 0.05f, &rngs[b]);
        first_tok[b] = first;
        if (pulsar_session_spec_round_begin(s, r[b], first, 64, 17, temps[b], 0, 1.0f, 0.05f,
                                            err, sizeof(err)) != 0) {
            fprintf(stderr, "round_begin bank %d: %s\n", b, err);
            return -1;
        }
        if (rows + pulsar_spec_round_n_rows(r[b]) > ROWS) return -1;
        row0[b] = rows;
        rows += pulsar_spec_round_fill_reqs(r[b], (uint32_t)b, first, reqs + rows);
        pulsar_session_bank_state_save(s, (uint32_t)b);
    }
    pulsar_session_spec_arm_capture(s, rows);
    uint32_t got = 0;
    const int rc = pulsar_session_decode_mixed(s, reqs, rows, logits, (int)(rows * (uint32_t)vocab),
                                               &got, PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof(err));
    pulsar_session_spec_arm_capture(s, 0u);
    if (rc != 0 || got != rows) {
        fprintf(stderr, "decode_mixed failed (rc=%d got=%u rows=%u): %s\n", rc, got, rows, err);
        return -1;
    }
    int live = 0;
    for (int b = 0; b < g_nb; b++) {
        if (!pulsar_session_bank_state_restore(s, (uint32_t)b)) return -1;
        int accepted[17];
        const int na = pulsar_session_spec_round_end(s, r[b], first_tok[b], eos, temps[b], 0, 1.0f,
                                                     0.05f, &rngs[b], logits, row0[b], accepted, 17,
                                                     err, sizeof(err));
        if (na < 0) { fprintf(stderr, "round_end bank %d: %s\n", b, err); return -1; }
        live++;
        pulsar_session_bank_state_save(s, (uint32_t)b);
    }
    return live;
}

static int run_shape(const char *name, const float *temps, int ticks) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) { CHECK(0, "%s: session create", name); return 0; }
    if ((int)gpu_graph_bank_pool_count(&s->graph) < g_nb) {
        CHECK(0, "%s: pool has %u banks, need %d (PULSAR_MSEQ_BANKS)", name,
              gpu_graph_bank_pool_count(&s->graph), g_nb);
        pulsar_session_free(s);
        return 0;
    }
    for (int b = 0; b < g_nb; b++)
        if (!bank_prefill(s, b)) { CHECK(0, "%s: prefill bank %d", name, b); pulsar_session_free(s); return 0; }
    const int vocab = pulsar_engine_logits_width(g_e);
    float *logits = (float *)malloc((size_t)ROWS * (size_t)vocab * sizeof(float));
    pulsar_spec_round *r[NB];
    for (int b = 0; b < g_nb; b++) r[b] = pulsar_spec_round_new();
    uint64_t rngs[NB] = {0x2545F4914F6CDD1Dull, 0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull};
    int compared = 0, deepest = 0, shallowest = 99;
    char err[256];
    for (int t = 0; t < ticks; t++) {
        int first_tok[NB];
        uint32_t row0[NB];
        if (tick_to_round_end(s, r, temps, rngs, first_tok, row0, logits, vocab) < 0) {
            CHECK(0, "%s: tick %d failed", name, t);
            break;
        }
        /* (a) serialized, from rng copies */
        uint64_t rng_a[NB], rng_b[NB];
        memcpy(rng_a, rngs, sizeof(rngs));
        memcpy(rng_b, rngs, sizeof(rngs));
        draft_result ser[NB], bat[NB];
        /* stage capture: the drafter's base logits rows and post-head hidden
         * rows per bank after each serialized redraft (rows [0, n_draft) of
         * the forward), to say WHERE a batched difference enters */
        float *ser_logits[NB] = {NULL, NULL, NULL};
        float *ser_hidden[NB] = {NULL, NULL, NULL};
        uint32_t ser_nd[NB] = {0, 0, 0};
        /* L150 stage dump (diagnostic): with L150_DUMP_DIR set (and the engine's
         * PULSAR_CUDA_GRAPH_DUMP_PREFIX/NAME/LAYER pointing at the dsp_ stages),
         * tick 0's serialized forwards dump under <dir>/ser_b<b>_ and the
         * batched one under <dir>/bat_, so the first differing stage can be
         * read offline (tools/dumpcmp_rows.py DIR n0 n1 n2 --batched t0_bat --solo 't0_ser_b{k}'). */
        const char *dump_dir = getenv("L150_DUMP_DIR");
        for (int b = 0; b < g_nb; b++) {
            pulsar_spec_round *one = r[b];
            const uint32_t bank = (uint32_t)b;
            uint64_t *rp = &rng_a[b];
            if (dump_dir) {
                char pfx[512];
                snprintf(pfx, sizeof(pfx), "%s/%s_ser_b%d", dump_dir, t == 0 ? "t0" : "tn", b);
                setenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX", pfx, 1);
            }
            if (pulsar_session_spec_redraft_batch(s, &one, &bank, &rp, 1, err, sizeof(err)) != 0)
                CHECK(0, "%s: tick %d serialized redraft bank %d: %s", name, t, b, err);
            draft_result d;
            memset(&d, 0, sizeof(d));
            if (pulsar_session_spec_redraft_peek(one, d.ids, d.conf, &d.n_draft, &d.keep, &d.sampled)) {
                ser_nd[b] = d.n_draft;
                ser_logits[b] = (float *)malloc((size_t)d.n_draft * (size_t)vocab * sizeof(float));
                ser_hidden[b] = (float *)malloc((size_t)d.n_draft * PULSAR_N_EMBD * sizeof(float));
                pulsar_gpu_tensor_read(s->graph.spec_logits, 0, ser_logits[b],
                                       (uint64_t)d.n_draft * (uint64_t)vocab * sizeof(float));
                pulsar_gpu_tensor_read(s->graph.batch_ffn_cur, 0, ser_hidden[b],
                                       (uint64_t)d.n_draft * PULSAR_N_EMBD * sizeof(float));
            }
        }
        peek_all(r, g_nb, ser);
        /* (b) batched, from equal rng copies; the rounds still hold their
         * requests (peek does not consume them) */
        {
            uint32_t banks[NB] = {0, 1, 2};
            uint64_t *rps[NB] = {&rng_b[0], &rng_b[1], &rng_b[2]};
            if (dump_dir) {
                char pfx[512];
                snprintf(pfx, sizeof(pfx), "%s/%s_bat", dump_dir, t == 0 ? "t0" : "tn");
                setenv("PULSAR_CUDA_GRAPH_DUMP_PREFIX", pfx, 1);
            }
            if (pulsar_session_spec_redraft_batch(s, r, banks, rps, g_nb, err, sizeof(err)) != 0)
                CHECK(0, "%s: tick %d batched redraft: %s", name, t, err);
        }
        peek_all(r, g_nb, bat);
        /* stage comparison: batched rows sit at [base_row_b, +n_draft) with the
         * greedy banks first in bank order (all-greedy: bank order) */
        {
            float *bl = (float *)malloc((size_t)ROWS * (size_t)vocab * sizeof(float));
            float *bh = (float *)malloc((size_t)ROWS * PULSAR_N_EMBD * sizeof(float));
            uint32_t total = 0;
            for (int b = 0; b < g_nb; b++) total += ser_nd[b];
            pulsar_gpu_tensor_read(s->graph.spec_logits, 0, bl, (uint64_t)total * (uint64_t)vocab * sizeof(float));
            pulsar_gpu_tensor_read(s->graph.batch_ffn_cur, 0, bh, (uint64_t)total * PULSAR_N_EMBD * sizeof(float));
            /* row order: greedy banks (temps == 0) first, then sampled, each in bank order */
            uint32_t off = 0;
            for (int pass = 0; pass < 2; pass++)
                for (int b = 0; b < g_nb; b++) {
                    if ((pass == 0) != (temps[b] <= 0.0f)) continue;
                    if (!ser_logits[b]) continue;
                    for (uint32_t k = 0; k < ser_nd[b]; k++) {
                        const float *sl = ser_logits[b] + (size_t)k * vocab;
                        const float *ql = bl + (size_t)(off + k) * vocab;
                        const float *sh = ser_hidden[b] + (size_t)k * PULSAR_N_EMBD;
                        const float *qh = bh + (size_t)(off + k) * PULSAR_N_EMBD;
                        int nl = 0, nh = 0; float ml = 0, mh = 0;
                        for (int v = 0; v < vocab; v++) if (sl[v] != ql[v]) { nl++; float d = fabsf(sl[v] - ql[v]); if (d > ml) ml = d; }
                        for (uint32_t v = 0; v < PULSAR_N_EMBD; v++) if (sh[v] != qh[v]) { nh++; float d = fabsf(sh[v] - qh[v]); if (d > mh) mh = d; }
                        if (nl || nh)
                            printf("  STAGE %s tick %d bank %d row %u: hidden differs in %d/%u (max %.3g), logits differ in %d/%d (max %.3g)\n",
                                   name, t, b, k, nh, PULSAR_N_EMBD, (double)mh, nl, vocab, (double)ml);
                    }
                    off += ser_nd[b];
                }
            free(bl); free(bh);
        }
        for (int b = 0; b < g_nb; b++) { free(ser_logits[b]); free(ser_hidden[b]); }
        for (int b = 0; b < g_nb; b++) {
            CHECK(ser[b].present == bat[b].present, "%s: tick %d bank %d present %d vs %d", name, t, b,
                  ser[b].present, bat[b].present);
            if (!ser[b].present || !bat[b].present) continue;
            CHECK(ser[b].n_draft == bat[b].n_draft && ser[b].keep == bat[b].keep &&
                  ser[b].sampled == bat[b].sampled,
                  "%s: tick %d bank %d n_draft/keep/sampled %u/%u/%d vs %u/%u/%d", name, t, b,
                  ser[b].n_draft, ser[b].keep, ser[b].sampled, bat[b].n_draft, bat[b].keep, bat[b].sampled);
            const uint32_t nd = ser[b].n_draft < bat[b].n_draft ? ser[b].n_draft : bat[b].n_draft;
            for (uint32_t k = 0; k <= nd; k++)
                CHECK(ser[b].ids[k] == bat[b].ids[k], "%s: tick %d bank %d draft id[%u] %d vs %d",
                      name, t, b, k, ser[b].ids[k], bat[b].ids[k]);
            for (uint32_t k = 0; k < nd; k++)
                CHECK(memcmp(&ser[b].conf[k], &bat[b].conf[k], sizeof(float)) == 0,
                      "%s: tick %d bank %d conf[%u] %.9g vs %.9g", name, t, b, k,
                      (double)ser[b].conf[k], (double)bat[b].conf[k]);
            CHECK(rng_a[b] == rng_b[b], "%s: tick %d bank %d rng state diverged", name, t, b);
            compared++;
            if ((int)nd > deepest) deepest = (int)nd;
            if ((int)nd < shallowest) shallowest = (int)nd;
        }
        /* commit (b) under each bank's switch, as the server does */
        for (int b = 0; b < g_nb; b++) {
            if (!pulsar_session_bank_state_restore(s, (uint32_t)b)) { CHECK(0, "%s: restore %d", name, b); break; }
            pulsar_session_spec_redraft_commit(s, r[b]);
            pulsar_session_bank_state_save(s, (uint32_t)b);
        }
        memcpy(rngs, rng_b, sizeof(rngs));
    }
    printf("%s: %d bank-ticks compared, draft depth %d..%d%s\n", name, compared,
           compared ? shallowest : 0, compared ? deepest : 0,
           g_fail ? "  (see FAIL lines)" : "  IDENTICAL");
    for (int b = 0; b < g_nb; b++) pulsar_spec_round_free(r[b]);
    free(logits);
    pulsar_session_free(s);
    return compared;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [TICKS] [DRAFT_DEPTH] [BANKS<=3]\n", argv[0]); return 2; }
    const int ticks = argc > 2 ? atoi(argv[2]) : 8;
    const int depth = argc > 3 ? atoi(argv[3]) : 0;
    g_nb = argc > 4 ? atoi(argv[4]) : NB;
    if (g_nb < 1 || g_nb > NB) { fprintf(stderr, "bad bank count\n"); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.dspark_draft_tokens = depth;
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_dspark(g_e)) {
        fprintf(stderr, "DSPARK-BATCH GATE: model has no drafter\n");
        pulsar_engine_close(g_e);
        return 1;
    }
    char *text = read_file("tests/long_context_story_prompt.txt");
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof(g_toks));
    pulsar_tokenize_text(g_e, text, &g_toks);
    free(text);

    const float greedy[NB] = {0.0f, 0.0f, 0.0f};
    const float mixed[NB] = {0.0f, 1.0f, 1.0f};
    const int c1 = run_shape("greedy x3", greedy, ticks);
    const int c2 = run_shape("greedy + sampled x2", mixed, ticks);
    CHECK(c1 >= ticks && c2 >= ticks, "too few bank-ticks compared (%d, %d)", c1, c2);
    printf("DSPARK BATCH GATE (depth %s, %d banks): %s\n", depth ? "pinned" : "default", g_nb, g_fail ? "FAIL" : "PASS");
    pulsar_tokens_free(&g_toks);
    pulsar_engine_close(g_e);
    return g_fail ? 1 : 0;
}

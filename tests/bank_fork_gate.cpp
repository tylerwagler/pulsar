/* Tier-2 PATH-A FORK gate (plan-33 increment A) — the fork==cold oracle.
 *
 * Proves the full-prefix in-memory fork produces a bank BYTE-IDENTICAL to a cold
 * full prefill: fork src->dst, prefill an IDENTICAL suffix into (a) the forked dst
 * and (b) a fresh COLD bank, then assert their committed comp/index KV frontiers
 * are byte-identical (FNV over the D2H rows) AND their next-token argmax matches.
 * Also checks the STRUCTURAL anti-contamination: a fork whose request prefix does
 * NOT match the source's committed history is REFUSED (no device write).
 *
 * MODEL-DEPENDENT, GPU-resident, needs PULSAR_MSEQ_BANKS>=3 (src + fork-dst + cold-dst).
 * Run manually under the memory discipline (hold temp/gpu.lock, drop_caches, no
 * foreign ds4 process). NOT part of `make test`.
 *
 * usage: PULSAR_MSEQ_BANKS=3 ./tests/bank_fork_gate MODEL [N_CACHED L]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
/* seed for greedy pulsar_session_sample calls (was &(uint64_t){7} compound literals;
 * greedy never consumes RNG state, so one shared object is behavior-identical). */
static uint64_t g_seed7 = 7;


static int g_fail;
#define CHECK(c, ...) do { if(!(c)){ fprintf(stderr,"FORK FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); g_fail=1; } } while(0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb"); if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(b); return NULL; }
    fclose(fp); b[n] = '\0'; if (len_out) *len_out = (size_t)n; return b;
}

/* FNV-1a fold of bank `bank`'s captured comp+index frontier rows (raw D2H). */
static uint64_t checksum_bank_kv(pulsar_session *s, uint32_t bank) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = PULSAR_ENGINE_MAINKV_ROWBYTES;
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t h = 1469598103934665603ull;
    uint8_t *buf = (uint8_t *)malloc(64u * 1024u * 1024u);
    if (!buf) return 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t ncomp = g->ms_n_comp[bank][il];
        if (ncomp) {
            pulsar_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, bank);
            if (!v || pulsar_gpu_tensor_read(v, 0, buf, (uint64_t)ncomp * attn_row) == 0) { pulsar_gpu_tensor_free(v); free(buf); return 0; }
            pulsar_gpu_tensor_free(v);
            for (uint64_t i = 0; i < (uint64_t)ncomp * attn_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
        }
        {   /* one emit writes the comp row AND, where the source runs an
             * indexer, the index-K row: one frontier */
            const uint32_t nidx = ncomp;
            if (nidx && gpu_graph_layer_has_index_pool(il)) {
                pulsar_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, bank);
                if (!v || pulsar_gpu_tensor_read(v, 0, buf, (uint64_t)nidx * idx_row) == 0) { pulsar_gpu_tensor_free(v); free(buf); return 0; }
                pulsar_gpu_tensor_free(v);
                for (uint64_t i = 0; i < (uint64_t)nidx * idx_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            }
        }
    }
    free(buf);
    return h;
}

/* Prefill bank `bank` (fresh) to the first `len` tokens of toks. */
static bool prefill_bank_cold(pulsar_session *s, uint32_t bank, int *toks, int len) {
    if (!pulsar_session_bank_state_restore(s, bank)) return false;
    pulsar_session_invalidate(s);
    pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks; p.len = p.cap = len;
    char err[256];
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync bank %u: %s\n", bank, err); return false; }
    gpu_graph_bank_counters_capture(&s->graph, bank);
    return true;
}

int GATE_ENTRY(int argc, char **argv) {
    g_seed7 = 7;
    g_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [N_CACHED L]\n", argv[0]); return 2; }
    /* ORACLE VALIDITY: n_cached must be a multiple of the PREFILL CHUNK (4096
     * default; also of 128 = LCM of compress ratios). A resumed prefill aligns
     * its chunks to absolute prefill_cap boundaries (imatrix.c chunked_range),
     * so with an aligned cut every resumed chunk has the SAME batch shape as the
     * cold run's — bit-identical kernels — and byte-identity is a fair oracle.
     * An UNALIGNED cut runs a short realign chunk through the per-token
     * compressor fallback (a different code path than cold's batched one):
     * last-ulp GEMM differences then propagate to every later row — that is the
     * engine's existing warm-continuation numerics, NOT a fork bug, and it
     * breaks the byte oracle spuriously. */
    /* Default shape (L163): the identity must hold at any length, so the
     * length buys boundary coverage, not strength.  The source must be whole
     * prefill chunks (checked below: the oracle's premise is that the suffix
     * on the forked bank and the cold run chunk at the same edges), so it is
     * one chunk; the total, 8601, puts a chunk edge (8192) inside both the
     * suffix and the cold run, wraps the raw ring (window 128) many times
     * over, and sits off every grid the caches are laid out on (8601 % 4 =
     * 1, % 128 = 25).  The 12288/20480 shape this replaced was 3 and 5 whole
     * chunks and a multiple of 128 -- aligned everywhere -- at 2.4x the
     * tokens. */
    const int n_cached = argc > 2 ? atoi(argv[2]) : 4096;
    const int L = argc > 3 ? atoi(argv[3]) : 8601;
    if ((n_cached % 4096) != 0) {
        fprintf(stderr, "n_cached %d must be a multiple of the prefill chunk (4096) "
                        "for the byte-identity oracle (see header comment)\n", n_cached);
        return 2;
    }
    const int ctx = L + 4096;
    if (n_cached >= L) { fprintf(stderr, "need N_CACHED < L\n"); return 2; }

    pulsar_engine *e = NULL; pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    pulsar_tokens base; memset(&base, 0, sizeof base);
    pulsar_session *s = NULL;
    int *toks = NULL;
    int rc = 1;
    {
    size_t tl = 0; char *text = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!text) { fprintf(stderr, "prompt read failed\n"); goto done; }
    pulsar_tokenize_text(e, text, &base); free(text);
    if (base.len < 256) { fprintf(stderr, "prompt too short\n"); goto done; }

    if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); goto done; }
    const uint32_t pool = gpu_graph_bank_pool_count(&s->graph);
    fprintf(stderr, "fork_gate: pool banks=%u ctx=%d n_cached=%d L=%d\n", pool, ctx, n_cached, L);
    if (pool < 3) { fprintf(stderr, "need PULSAR_MSEQ_BANKS>=3\n"); goto done; }

    toks = (int *)malloc((size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) toks[i] = base.v[i % base.len];

    /* 1. SRC = bank 0 prefilled to the shared prefix [0, n_cached). */
    CHECK(prefill_bank_cold(s, 0, toks, n_cached), "src prefill");
    pulsar_gpu_synchronize();
    const uint64_t sum_src = checksum_bank_kv(s, 0);
    CHECK(sum_src != 0, "src checksum");
    fprintf(stderr, "fork_gate: src(bank0) prefilled to %d, checksum=%016" PRIx64 "\n", n_cached, sum_src);

    /* 2. NEGATIVE: a fork whose prefix does NOT match src is REFUSED (no write). */
    {
        int *wrong = (int *)malloc((size_t)n_cached * sizeof(int));
        for (int i = 0; i < n_cached; i++) wrong[i] = toks[i];
        wrong[n_cached / 2] ^= 0x1;   /* one token differs */
        const int rc = pulsar_session_bank_fork(s, 0, 2, wrong, n_cached, n_cached);
        CHECK(rc != 0, "fork ACCEPTED a mismatched prefix (anti-contamination broken)");
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "src left fork-pinned after refusal");
        fprintf(stderr, "fork_gate: mismatch-prefix fork refused rc=%d : %s\n", rc, rc != 0 ? "OK" : "FAIL");
        free(wrong);
        /* src must be untouched by the refused fork. */
        gpu_graph_bank_counters_capture(&s->graph, 0);
        pulsar_gpu_synchronize();
        CHECK(checksum_bank_kv(s, 0) == sum_src, "src KV changed by a REFUSED fork");
    }

    /* 3. FORK src(0) -> dst(1). Must validate + clone + mirror counters. */
    {
        /* src is cur (bank 0 installed); its committed history is s->checkpoint. */
        const int rc = pulsar_session_bank_fork(s, 0, 1, toks, n_cached, n_cached);
        CHECK(rc == 0, "full-prefix fork refused rc=%d", rc);
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "src still pinned after fork");
        /* immediate fork bit-identity: dst frontier == src frontier. */
        pulsar_gpu_synchronize();
        const uint64_t sum_fork = checksum_bank_kv(s, 1);
        CHECK(sum_fork == sum_src, "forked dst KV != src KV immediately after clone (%016" PRIx64 " vs %016" PRIx64 ")", sum_fork, sum_src);
        fprintf(stderr, "fork_gate: fork(0->1) rc=%d immediate dst checksum=%016" PRIx64 " (==src: %s)\n",
                rc, sum_fork, sum_fork == sum_src ? "YES" : "NO");
    }

    /* 4. Prefill the IDENTICAL suffix [n_cached, L) into the FORKED dst (bank 1)
     *    and a COLD full prefill into bank 2. */
    /* fork dst: install bank 1 (carries src's checkpoint at n_cached), sync to L. */
    CHECK(pulsar_session_bank_state_restore(s, 1), "install forked bank 1");
    {
        pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks; p.len = p.cap = L;
        char err[256];
        CHECK(pulsar_session_sync(s, &p, err, sizeof err) == 0, "fork-dst suffix sync: %s", err);
        /* LIVE frontier assert (catches a silently short-circuited sync HERE,
         * while bank 1 is still installed — pulsar_session_pos reads the live
         * checkpoint, not a possibly-stale carry). */
        CHECK(pulsar_session_pos(s) == L, "fork-dst live pos %d != %d after suffix sync",
              pulsar_session_pos(s), L);
        gpu_graph_bank_counters_capture(&s->graph, 1);
    }
    const int tok_fork = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
    pulsar_gpu_synchronize();
    const uint64_t sum_fork_full = checksum_bank_kv(s, 1);
    /* SAVE bank 1's live state before switching away — the final bank_pos(1)
     * read below reads the CARRY once bank 2 is installed; without this save it
     * reads the stale fork-time carry (the original gate-A false "pos" failure). */
    pulsar_session_bank_state_save(s, 1);

    /* cold: bank 2 full prefill of [0, L). */
    CHECK(prefill_bank_cold(s, 2, toks, L), "cold full prefill bank 2");
    const int tok_cold = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
    pulsar_gpu_synchronize();
    const uint64_t sum_cold = checksum_bank_kv(s, 2);

    /* 5. THE ORACLE: fork+suffix must be byte-identical to cold full prefill. */
    CHECK(sum_fork_full == sum_cold,
          "fork+suffix KV NOT byte-identical to cold (%016" PRIx64 " vs %016" PRIx64 ")",
          sum_fork_full, sum_cold);
    CHECK(tok_fork == tok_cold, "fork+suffix next-token %d != cold %d", tok_fork, tok_cold);
    CHECK(pulsar_session_bank_pos(s, 1) == L, "fork-dst pos %d != %d", pulsar_session_bank_pos(s, 1), L);
    fprintf(stderr, "fork_gate: fork+suffix checksum=%016" PRIx64 " cold=%016" PRIx64 " (byte-identical: %s); next-token fork=%d cold=%d\n",
            sum_fork_full, sum_cold, sum_fork_full == sum_cold ? "YES" : "NO", tok_fork, tok_cold);

    /* ============ P6: SERVER-CONFIG work-skipped TEETH (dst == cur) ==========
     * The value bug: the server provisions dst, INSTALLS it as cur, and
     * INVALIDATES the live session (s->checkpoint.len -> 0) BEFORE forking.
     * The fork==cold phase forks into a NON-cur dst and re-installs it, so it never
     * exercised this path — and it left the frontier at 0, so the server
     * re-prefilled 100% of the prompt (fork "fired" but skipped ZERO work; the
     * correctness gates stayed green because re-prefill-from-0 IS byte-identical).
     * These teeth assert bank_pos(dst) == cut in the dst==cur config, so a
     * 0-benefit fork can never pass green again.  Since L264 the dst also takes
     * src's grid checkpoints, so its continuation resumes from the grid point
     * under the fork frontier instead of prefilling from 0. */
    {
        const int FLEN = 8192;

        /* ---- FULL fork, dst==cur: bank_pos(dst) must be n_cached, not 0 ---- */
        CHECK(prefill_bank_cold(s, 0, toks, FLEN), "P6 full src prefill");
        pulsar_session_bank_state_save(s, 0);
        CHECK(pulsar_session_bank_state_restore(s, 1), "P6 install dst(1) as cur");
        pulsar_session_invalidate(s);                    /* mimic provision: cur len -> 0 */
        CHECK(pulsar_session_pos(s) == 0, "P6 precondition: dst==cur must start at 0");
        CHECK(pulsar_session_bank_fork(s, 0, 1, toks, FLEN, FLEN) == 0, "P6 full fork (dst==cur) refused");
        CHECK(pulsar_session_bank_pos(s, 1) == FLEN,
              "P6 FULL dst==cur bank_pos %d != %d (WORK-SKIPPED REGRESSION: "
              "server would re-prefill from 0)", pulsar_session_bank_pos(s, 1), FLEN);
        CHECK(pulsar_session_pos(s) == FLEN, "P6 full live pos %d != %d", pulsar_session_pos(s), FLEN);
        fprintf(stderr, "fork_gate: P6 FULL fork dst==cur bank_pos=%d (==%d: %s)\n",
                pulsar_session_bank_pos(s, 1), FLEN,
                pulsar_session_bank_pos(s, 1) == FLEN ? "YES" : "NO");
        {   /* L196: the served shape continues on the forked cur -- the resume
             * must start at the grid point under the fork frontier, not cold
             * (provision's invalidate had zeroed the live prefill frontier). */
            int *toks6f = (int *)malloc((size_t)(FLEN + 64) * sizeof(int));
            memcpy(toks6f, toks, (size_t)FLEN * sizeof(int));
            for (int i = 0; i < 64; i++) toks6f[FLEN + i] = base.v[(i + 777) % base.len];
            pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks6f; p.len = p.cap = FLEN + 64;
            char e6f[256];
            CHECK(pulsar_session_sync(s, &p, e6f, sizeof e6f) == 0, "P6 full continuation sync: %s", e6f);
            CHECK(pulsar_session_resume_origin(s) == (int)(((uint32_t)FLEN / PULSAR_RESUME_GRID) * PULSAR_RESUME_GRID),
                  "P6 FULL dst==cur continuation resumed from %d, want the grid point under the fork frontier %d "
                  "(a served full fork must not go cold)", pulsar_session_resume_origin(s), FLEN);
            CHECK(pulsar_session_pos(s) == FLEN + 64, "P6 full continuation pos %d != %d", pulsar_session_pos(s), FLEN + 64);
            free(toks6f);
        }

    }

    fprintf(stderr, "BANK-FORK GATE: %s\n", g_fail ? "FAIL" : "PASS");
    rc = g_fail ? 1 : 0;
    }
done:
    free(toks);
    pulsar_session_free(s);
    pulsar_tokens_free(&base);
    gate_engine_close(e);
    return rc;
}

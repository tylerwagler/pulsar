/* Shared fixture for the multi-bank engine gates (algo_stability_gate,
 * mixed_neutrality_gate, multiseq_decode_gate).  Before 2026-09-02 (L153) each
 * carried its own copy of: read the story prompt, tokenize it, and populate a
 * bank -- repoint the graph's views, invalidate, classic sync, capture the
 * bank's frontier counters, take the argmax.  Three copies of a fixture that
 * MUST agree (the gates compare each other's shapes) with nothing enforcing it.
 *
 * Header-only, static inline: the gates are single-TU programs. */
#pragma once

#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATE_STORY_PROMPT "tests/long_context_story_prompt.txt"

/* L183: prefill toks[start, end) as a classic CONTINUATION at `start` -- the
 * chunked prefill entered at the checkpoint, which is what pulsar_session_sync
 * did before L183.  Since L183 the public sync recomputes a resume from the
 * last chunk-grid boundary (a resume is a cold prefill), so a gate that
 * compares the MIXED lane's prefill of [start, end) against the classic lane
 * on the SAME chunking needs this entry; through sync it would compare the
 * mixed continuation against a cold prefill and measure chunk dependence
 * instead of lane parity (that is what the chunk-neutrality gate measures).
 * Leaves the session's checkpoint at `end` and its logits at the last row. */
static inline bool gate_prefill_suffix_classic(pulsar_session *s, const pulsar_tokens *toks,
                                               int start, int end, char *err, size_t errlen) {
    if (!s || !toks || start < 0 || end <= start || end > toks->len) {
        snprintf(err, errlen, "classic suffix: bad range [%d, %d)", start, end);
        return false;
    }
    if (s->checkpoint.len != start) {
        snprintf(err, errlen, "classic suffix: checkpoint at %d, suffix starts at %d", s->checkpoint.len, start);
        return false;
    }
    bool cancelled = false;
    if (!gpu_graph_prefill_chunked_range(&s->graph, &s->engine->model, &s->engine->weights, toks,
                                         (uint32_t)start, (uint32_t)(end - start), s->logits,
                                         false, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &cancelled)) {
        snprintf(err, errlen, "classic suffix prefill [%d, %d) failed", start, end);
        return false;
    }
    pulsar_tokens p = *toks; p.len = end;
    pulsar_tokens_copy(&s->checkpoint, &p);
    s->checkpoint_valid = true;
    return true;
}

static inline char *gate_read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp); buf[n] = '\0'; if (len_out) *len_out = (size_t)n; return buf;
}

/* Tokenize the story fixture into *toks (zeroed first).  false + a message on
 * a missing file or a prompt shorter than `need` tokens. */
static inline bool gate_load_story(pulsar_engine *e, pulsar_tokens *toks, int need) {
    size_t tl = 0;
    char *text = gate_read_file(GATE_STORY_PROMPT, &tl);
    if (!text) { fprintf(stderr, "prompt read failed (%s)\n", GATE_STORY_PROMPT); return false; }
    memset(toks, 0, sizeof(*toks));
    pulsar_tokenize_text(e, text, toks);
    free(text);
    if (toks->len < need) { fprintf(stderr, "prompt too short (%d<%d)\n", toks->len, need); return false; }
    return true;
}

/* First differing float index, or -1 if byte-identical over n floats. */
static inline long gate_first_diff(const float *a, const float *b, long n) {
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return -1;
}

/* Populate bank `bank` of session s from the token view [v, v+len): repoint
 * the graph at the bank (when a pool exists), invalidate (no prefix reuse
 * across banks), classic-sync the prompt, capture the bank's frontier
 * counters, and hand back the next token (argmax) through *argtok when asked.
 * The session does not take ownership of v. */
static inline bool gate_populate_bank(pulsar_session *s, uint32_t bank, const int *v, int len,
                                      int *argtok, const char *what) {
    pulsar_gpu_graph *g = &s->graph;
    char err[256];
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, bank)) {
        fprintf(stderr, "%s: bank %u repoint failed\n", what, bank);
        return false;
    }
    pulsar_session_invalidate(s);
    pulsar_tokens p = { .v = (int *)v, .len = len, .cap = len };
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
        fprintf(stderr, "%s: bank %u sync failed: %s\n", what, bank, err);
        return false;
    }
    gpu_graph_bank_counters_capture(g, bank);
    if (argtok) *argtok = pulsar_session_argmax(s);
    return true;
}

/* The pool must hold at least `need` banks; false + a message otherwise. */
static inline bool gate_pool_fits(pulsar_session *s, uint32_t need) {
    const uint32_t have = gpu_graph_bank_pool_count(&s->graph);
    if (have >= need) return true;
    fprintf(stderr, "pool too small: %u < %u (set PULSAR_MSEQ_BANKS)\n", have, need);
    return false;
}

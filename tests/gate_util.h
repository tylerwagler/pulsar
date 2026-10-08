/* Shared helpers for every gate, PUBLIC API ONLY (L278).  A family-generic gate includes this and nothing of the
 * engine's internals; tests/gate_fixture.h adds DeepSeek's session-graph helpers on top of it for the gates that
 * grade that family's mechanisms.  Gates migrate their private copies here as they are touched (the inventory in
 * the L278 row counted CHECK counters in ~50 files, read_file in ~17, FNV in 9).
 *
 * Header-only, static inline: the gates are single-TU programs, and the runner links them side by side. */
#pragma once

#include "pulsar.h"
#include "pulsar_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GATE_STORY_PROMPT "tests/long_context_story_prompt.txt"

/* A named assertion that counts: prints "ok  " or "FAIL" and the message, and bumps `fails` on a failure. */
#define GATE_CHECK(fails, cond, ...) do { \
        const bool gate_ok_ = (cond); \
        printf("  %s  ", gate_ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); printf("\n"); \
        if (!gate_ok_) (fails)++; \
    } while (0)

/* Monotonic seconds. */
static inline double gate_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Canonical FNV-1a 64 (the reference-capture blobs and tokens.bin hash with this basis). */
static inline uint64_t gate_fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* The whole file, NUL-terminated; NULL on any failure (never exits: a runner gate returns). */
static inline char *gate_read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (n < 0) { fclose(fp); return NULL; }
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

/* The session's logits row into `out` (width floats); false when the copy is short. */
static inline bool gate_logits(pulsar_session *s, float *out, int width) {
    return pulsar_session_copy_logits(s, out, width) == width;
}

/* First differing float index, or -1 if byte-identical over n floats. */
static inline long gate_first_diff(const float *a, const float *b, long n) {
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return -1;
}

/* L260: the per-row bound for a decode row compared ACROSS the width-exact
 * boundary (pulsar_gpu_matmul_decode_exact_rows): KL(narrow || wide) over the
 * softmax.  Measured worst rows, width 12/16 vs narrower (2026-10-02, all three
 * cuBLASLt decode arms): 4.7e-2 on the B300 reference text, 0.39 on GATE 5R's
 * replayed-prompt rows.  Used by the dspark-batch and algo-stability gates.
 * L262: GATE 5R no longer uses it (its replayed off-distribution rows reached
 * 0.82 at 8c76d808 while the reference text moved < 0.05) -- 5R separates
 * corruption with GATE_5R_FLOOR_FRACTION, and FIDELITY across widths is graded
 * against the B300 reference by cuda-decode-reference-gate-*. */
#define GATE_DECODE_WIDTH_KL_TOL 1.0

/* L262 GATE 5R: a graded row passes when its KL vs the solo row is at most this
 * fraction of the step's own cross-bank floor (the smallest KL of a row against
 * ANOTHER bank's row at the same position -- what a bank / frontier / KV mix-up
 * produces), measured in the same step.  4: at 8c76d808 the worst graded row was
 * 0.82 against a floor of 8.49 (ratio 0.097). */
#define GATE_5R_FLOOR_FRACTION 0.25

/* L262 cuda-decode-reference-gate-*: a w-wide decode step's last row, against
 * the SAME tokens over the SAME prefill decoded as w one-row steps (the width
 * effect alone), must stay within this fraction of the one-row run's own
 * distance from the B300 reference -- KL(serial || wide) <= max(FRACTION *
 * KL(ref || serial), FLOOR) -- and keep the serial or the reference top-1.
 * Measured at 3d5d2430, widths 16..32: worst ratio 0.14 (code depth 3840 at 24:
 * 0.036 vs KL(ref||serial) 0.25), the tiny-KL depths under 1e-5 absolute. */
#define GATE_REF_WIDTH_KL_FRACTION 0.5
#define GATE_REF_WIDTH_KL_FLOOR 1e-4

/* Top-2 logit margin above which a top-1 flip cannot be quantization noise (spec_sampling_gate's greedy
 * gate; cuda-decode-reference-gate's every-row leg).  Calibrated on the shipped type-43 artifact, where
 * decisive positions measure 6.0-14.8 and ambiguous ones 0.19-1.94; 2.0 sits in the empty band with ~3x
 * headroom either side. */
#define GATE_DECISIVE_MARGIN 2.0f

/* KL(ref || cur) over the softmax of two logit rows, in double. */
static inline double gate_row_kl(const float *ref, const float *cur, long n) {
    double mr = -INFINITY, mc = -INFINITY;
    for (long i = 0; i < n; i++) { if (ref[i] > mr) mr = ref[i]; if (cur[i] > mc) mc = cur[i]; }
    double zr = 0, zc = 0;
    for (long i = 0; i < n; i++) { zr += exp((double)ref[i] - mr); zc += exp((double)cur[i] - mc); }
    const double lzr = mr + log(zr), lzc = mc + log(zc);
    double kl = 0;
    for (long i = 0; i < n; i++) {
        const double lp = (double)ref[i] - lzr;
        kl += exp(lp) * (lp - ((double)cur[i] - lzc));
    }
    return kl < 0 ? 0 : kl;
}

/* True when a decode step of `rows` rows takes width-dependent arithmetic. */
static inline bool gate_width_inexact(int rows) { return rows > pulsar_gpu_matmul_decode_exact_rows(); }

/* L284: populate bank `bank` of session s from the token view [v, v+len) through the PUBLIC bank API -- repoint
 * the session at the bank, invalidate (no prefix reuse across banks), sync the prompt, save the bank's state (its
 * frontier counters and host carry) -- and hand back the next token (argmax) through *argtok when asked.  The
 * family-generic twin of gate_fixture.h's gate_populate_bank (which reaches DeepSeek's session graph directly);
 * on DeepSeek the two leave the same device state.  The session does not take ownership of v. */
static inline bool gate_bank_sync(pulsar_session *s, uint32_t bank, const int *v, int len, int *argtok,
                                  const char *what) {
    char err[256] = "";
    if (pulsar_session_bank_repoint(s, bank) != 0) {
        fprintf(stderr, "%s: bank %u repoint failed\n", what, bank);
        return false;
    }
    pulsar_session_invalidate(s);
    pulsar_tokens p = { .v = (int *)v, .len = len, .cap = len };
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
        fprintf(stderr, "%s: bank %u sync failed: %s\n", what, bank, err);
        return false;
    }
    pulsar_session_bank_state_save(s, bank);
    if (argtok) *argtok = pulsar_session_argmax(s);
    return true;
}

/* The session's bank pool holds at least `need` banks; false + a message otherwise (public API). */
static inline bool gate_bank_pool_fits(pulsar_session *s, int need) {
    const int have = pulsar_session_bank_count(s);
    if (have >= need) return true;
    fprintf(stderr, "bank pool too small: %d < %d (size it through pulsar_engine_set_bank_pool)\n", have, need);
    return false;
}

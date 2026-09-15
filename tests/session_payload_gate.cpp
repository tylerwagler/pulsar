/* Session payload SAVE -> LOAD round-trip gate.
 *
 * Nothing tested this before 2026-08-18, and the gap hid a real bug: the
 * raw-ring payload path still described the ring as "__half containers" long
 * after it became a packed row, so save indexed a 584 B/row buffer at a
 * 1024 B stride and read past the end of the allocation.  It compiled, no gate
 * touched it, and both of its f16 converters were marked PULSAR_MAYBE_UNUSED --
 * which suppressed the one warning that would have fired when their last real
 * caller disappeared.
 *
 * The payload format then changed twice more in a day (v4: attn comp stored
 * packed; v5: raw ring + indexer comp stored packed), so this gate exists to
 * make "a session survives a round trip" a thing the build can state rather
 * than a thing we assume.
 *
 * WHAT IT PROVES, in the order that localises a failure:
 *   1. THE LINEAR LANES, byte for byte.  FNV-1a over each kv source's attn comp
 *      rows, its index-K rows, and both compressor state lanes, folded to the
 *      size the ALLOCATION holds rather than the size the payload believes it
 *      should carry.  That distinction is the point: a payload that silently
 *      drops the tail of a lane balances its own accounting and round-trips its
 *      own header, so only a fold at the allocation's size can see it.  (v11: a
 *      0731 ratio-4 state lane is 32768 B and the payload carried 8192.)
 *   2. THE RAW RING, functionally.  Byte-comparing a ring is awkward (rows sit
 *      at pos % cap and the tail is legitimately stale), so instead both
 *      sessions evaluate the SAME next token and their full-vocab logits must
 *      match bit for bit.  That runs the restored raw window, comp rows and
 *      indexer selection through attention, which is the property a session
 *      restore actually owes its caller.
 *
 * A byte copy round trip should be EXACT -- v5 re-encodes nothing -- so this
 * gate compares for equality, not tolerance.  If it ever needs a tolerance,
 * something re-encodes and that is the finding.
 *
 * MODEL-DEPENDENT, GPU-resident.  Run under the usual memory discipline.
 *
 * usage: ./tests/session_payload_gate MODEL [L]
 *
 * L220: a runner function like the other model gates (tests/gate_entry.h), so
 * the battery reuses the broker's engine instead of paying a 92 GB cold load
 * for this gate alone.  Every session it creates is registered with
 * g_sessions so a CHECK failure frees it before returning -- in the runner the
 * next gate has to fit beside whatever this one leaked.
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Slots are cleared as each session is freed, so release is idempotent. */
static pulsar_session *g_sessions[3];

static int payload_gate_release(pulsar_engine *e) {
    for (int i = 0; i < 3; i++) {
        if (g_sessions[i]) pulsar_session_free(g_sessions[i]);
        g_sessions[i] = NULL;
    }
    gate_engine_close(e);
    return 1;
}

#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "SESSION-PAYLOAD GATE FAIL: " __VA_ARGS__); \
    fprintf(stderr, "\n"); return payload_gate_release(e); } } while (0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n <= 0) { fclose(fp); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { free(buf); fclose(fp); return NULL; }
    buf[n] = 0;
    fclose(fp);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

/* FNV-1a over every linear-from-row-0 lane the payload carries: each kv
 * source's comp rows, its index-K rows where an indexer runs, and both
 * compressor state lanes.  (The raw ring is the one lane that is NOT linear --
 * rows sit at pos % cap -- so it is checked functionally, by the logits below.)
 *
 * The fold runs to pulsar_gpu_tensor_bytes(), i.e. over exactly what the
 * ALLOCATION holds, not over what session_payload.cpp believes it should hold.
 * That distinction is the whole point: a payload that carries fewer bytes than
 * the lane holds still balances its own accounting and still round-trips its own
 * header, so only a fold at the allocation's size can see the missing tail.  The
 * v11 fix exists because a V4 ratio-4 state lane is 32768 B and the payload
 * carried a quarter of it. */
static bool fold_tensor(uint64_t *h, pulsar_gpu_tensor *t, uint8_t *buf, size_t cap,
                        uint64_t *bytes_io) {
    /* A lane the allocation never made (V4.1's indexer compressor, a ratio-1
     * source's pending group) contributes nothing and is not an error. */
    if (!t) return true;
    const uint64_t n = pulsar_gpu_tensor_bytes(t);
    if (n == 0) return true;
    if (n > cap) return false;
    if (pulsar_gpu_tensor_read(t, 0, buf, n) == 0) return false;
    for (uint64_t i = 0; i < n; i++) { *h ^= buf[i]; *h *= 1099511628211ull; }
    *bytes_io += n;
    return true;
}

static uint64_t checksum_lanes(pulsar_session *s, const char *tag) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = PULSAR_ENGINE_MAINKV_ROWBYTES;
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    const size_t cap = 64u * 1024u * 1024u;
    bool ok = true;
    uint64_t h = 1469598103934665603ull;
    uint64_t attn_rows = 0, idx_rows = 0, attn_state = 0, idx_state = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER && ok; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t ncomp = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
        if (ncomp) {
            const uint64_t n = (uint64_t)ncomp * attn_row;
            if (pulsar_gpu_tensor_read(g->layer_attn_comp_cache[il], 0, buf, n) == 0) { ok = false; break; }
            for (uint64_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            attn_rows += ncomp;
        }
        /* One emit writes the comp row AND the index-K row: one frontier, and
         * the index pool exists only where an indexer runs -- 0731's ratio-128
         * (HCA) sources publish none, so their lane is NULL by construction. */
        if (ncomp && pulsar_attn_runs_indexer(pulsar_layer_attn_layout(il)->mode)) {
            const uint64_t n = (uint64_t)ncomp * idx_row;
            if (pulsar_gpu_tensor_read(g->layer_index_comp_cache[il], 0, buf, n) == 0) { ok = false; break; }
            for (uint64_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            idx_rows += ncomp;
        }
        ok = fold_tensor(&h, g->layer_attn_state_kv[il], buf, cap, &attn_state) &&
             fold_tensor(&h, g->layer_attn_state_score[il], buf, cap, &attn_state) &&
             fold_tensor(&h, g->layer_index_state_kv[il], buf, cap, &idx_state) &&
             fold_tensor(&h, g->layer_index_state_score[il], buf, cap, &idx_state);
    }
    free(buf);
    if (!ok) return 0;
    fprintf(stderr, "  %-8s attn_comp_rows=%llu (%llu B/row)  idx_comp_rows=%llu (%llu B/row)  "
                    "attn_state=%llu B  idx_state=%llu B  fnv=%016llx\n",
            tag, (unsigned long long)attn_rows, (unsigned long long)attn_row,
            (unsigned long long)idx_rows, (unsigned long long)idx_row,
            (unsigned long long)attn_state, (unsigned long long)idx_state,
            (unsigned long long)h);
    return h;
}

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [L]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 2048;
    const int ctx = L + 4096;

    pulsar_engine *e = NULL; pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t tl = 0; char *text = read_file("tests/long_context_story_prompt.txt", &tl);
    CHECK(text != NULL, "prompt read failed");
    pulsar_tokens base; memset(&base, 0, sizeof base);
    pulsar_tokenize_text(e, text, &base); free(text);
    CHECK(base.len >= L, "prompt has %d tokens, need >= %d", base.len, L);

    const int width = pulsar_engine_logits_width(e);
    CHECK(width > 0, "bad logits width %d", width);
    /* The token both sessions will evaluate after the restore point.  Taken from
     * the prompt so it is a plausible continuation rather than noise. */
    const int probe_tok = base.v[L];

    char err[256];

    /* ---- session A: prefill, checksum, save ---- */
    pulsar_session *a = NULL;
    CHECK(pulsar_session_create(&a, e, ctx) == 0, "session A create failed");
    g_sessions[0] = a;
    pulsar_tokens p; memset(&p, 0, sizeof p);
    p.v = base.v; p.len = p.cap = L;
    CHECK(pulsar_session_sync(a, &p, err, sizeof err) == 0, "A sync: %s", err);

    fprintf(stderr, "session_payload_gate: L=%d ctx=%d width=%d payload=%llu B\n",
            L, ctx, width, (unsigned long long)pulsar_session_payload_bytes(a));
    const uint64_t fnv_a = checksum_lanes(a, "saved");
    CHECK(fnv_a != 0, "checksum of session A failed");

    FILE *fp = tmpfile();
    CHECK(fp != NULL, "tmpfile failed");
    const uint64_t pbytes = pulsar_session_payload_bytes(a);
    CHECK(pulsar_session_save_payload(a, fp, err, sizeof err) == 0, "save: %s", err);

    /* The size the engine advertises must be the size it actually wrote -- the
     * accounting has been wrong before (it sized packed rows at the f32 stride,
     * over-reserving the disk cache by 3.5x). */
    const long written = ftell(fp);
    CHECK(written > 0, "ftell after save failed");
    CHECK((uint64_t)written == pbytes,
          "payload_bytes() said %llu but save wrote %ld", (unsigned long long)pbytes, written);

    /* A evaluates the probe token from its LIVE state -- the reference. */
    CHECK(pulsar_session_eval(a, probe_tok, err, sizeof err) == 0, "A eval: %s", err);
    float *ref = (float *)malloc((size_t)width * sizeof(float));
    CHECK(ref != NULL, "alloc ref");
    CHECK(pulsar_session_copy_logits(a, ref, width) == width, "A copy_logits");

    /* ---- session B: load, checksum, evaluate the same token ---- */
    rewind(fp);
    pulsar_session *b = NULL;
    CHECK(pulsar_session_create(&b, e, ctx) == 0, "session B create failed");
    g_sessions[1] = b;
    CHECK(pulsar_session_load_payload(b, fp, pbytes, err, sizeof err) == 0, "load: %s", err);
    fclose(fp);

    const uint64_t fnv_b = checksum_lanes(b, "restored");
    CHECK(fnv_b != 0, "checksum of session B failed");
    CHECK(fnv_a == fnv_b,
          "payload lanes differ across the round trip: saved=%016llx restored=%016llx "
          "-- a comp pool, an index-K pool or a compressor state lane did not survive as bytes",
          (unsigned long long)fnv_a, (unsigned long long)fnv_b);

    CHECK(pulsar_session_eval(b, probe_tok, err, sizeof err) == 0, "B eval: %s", err);
    float *got = (float *)malloc((size_t)width * sizeof(float));
    CHECK(got != NULL, "alloc got");
    CHECK(pulsar_session_copy_logits(b, got, width) == width, "B copy_logits");

    int ndiff = 0; float worst = 0.0f; int first = -1;
    for (int i = 0; i < width; i++) {
        if (ref[i] != got[i]) {
            ndiff++;
            const float d = ref[i] > got[i] ? ref[i] - got[i] : got[i] - ref[i];
            if (d > worst) worst = d;
            if (first < 0) first = i;
        }
    }
    fprintf(stderr, "  probe token %d: %d/%d logits differ (worst |delta| %.6g, first idx %d)\n",
            probe_tok, ndiff, width, (double)worst, first);
    CHECK(ndiff == 0,
          "restored session decodes differently: %d/%d logits differ, worst %.6g. "
          "The lanes above matched byte for byte, so this is the RAW RING",
          ndiff, width, (double)worst);

    /* ---- corruption case: the v10 digest must refuse a one-byte flip ----
     * The digest is the only thing between a damaged disk cache and silent
     * wrong attention, so assert it fires on a payload that differs in exactly
     * one byte.  Offset pbytes-9 is the last DATA byte (the trailing 8 are the
     * digest itself): a structural field would trip the reader's length check
     * first, so this offset is what reaches, and must exercise, the digest
     * comparison. */
    {
        FILE *fc = tmpfile();
        CHECK(fc != NULL, "tmpfile for corruption case failed");
        CHECK(pulsar_session_save_payload(a, fc, err, sizeof err) == 0, "save 2: %s", err);
        /* This file's length, not the first save's: the eval above advanced the
         * session, so the payload grew. */
        const long len2 = ftell(fc);
        CHECK(len2 > 16, "corruption payload too small (%ld B)", len2);
        const long at = len2 - 9;
        CHECK(fseek(fc, at, SEEK_SET) == 0, "corruption seek failed");
        const int byte = fgetc(fc);
        CHECK(byte != EOF, "corruption read failed");
        CHECK(fseek(fc, at, SEEK_SET) == 0, "corruption seek-back failed");
        CHECK(fputc(byte ^ 0xFF, fc) != EOF, "corruption write failed");
        fflush(fc);
        rewind(fc);
        pulsar_session *c = NULL;
        CHECK(pulsar_session_create(&c, e, ctx) == 0, "session C create failed");
        g_sessions[2] = c;
        err[0] = '\0';
        const int crc = pulsar_session_load_payload(c, fc, (uint64_t)len2, err, sizeof err);
        CHECK(crc != 0 && strstr(err, "digest mismatch") != NULL,
              "CORRUPTED PAYLOAD ACCEPTED: one flipped byte at offset %ld loaded "
              "clean or failed without the digest (rc=%d, err='%s') -- the v10 "
              "digest did not fire", at, crc, err);
        printf("corruption at byte %ld refused (rc=%d): %s\n", at, crc, err);
        pulsar_session_free(c);
        g_sessions[2] = NULL;
        fclose(fc);
    }

    free(ref); free(got); free(base.v);
    printf("SESSION-PAYLOAD GATE: PASS (v%u, %llu B, comp fnv %016llx, logits identical)\n",
           (unsigned)PULSAR_SESSION_PAYLOAD_VERSION, (unsigned long long)pbytes,
           (unsigned long long)fnv_a);
    payload_gate_release(e);
    return 0;
}

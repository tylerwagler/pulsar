/* Session payload SAVE -> LOAD round-trip gate.
 *
 * Nothing tested this before 2026-08-18, and the gap hid a real bug: the
 * raw-ring payload path still described the ring as "__half containers" long
 * after it became PULSAR_ATTN_PACK, so save indexed a 584 B/row buffer at a
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
 *   1. KV BYTES.  FNV-1a over the attn comp cache and the indexer comp cache of
 *      the restored session equals the same fold over the saved one.  These are
 *      the two caches v4 and v5 changed the storage format of, and a mismatch
 *      names which one.
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
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "SESSION-PAYLOAD GATE FAIL: " __VA_ARGS__); \
    fprintf(stderr, "\n"); return 1; } } while (0)

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

/* FNV-1a over the two caches whose STORAGE FORMAT the payload changed.  Rows are
 * linear from 0 in both, unlike the raw ring, so a byte fold is well defined. */
static uint64_t checksum_comp_caches(pulsar_session *s, const char *tag) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = PULSAR_ENGINE_IDXFP4_ROWBYTES;
    uint64_t h = 1469598103934665603ull;
    uint64_t attn_rows = 0, idx_rows = 0;
    uint8_t *buf = (uint8_t *)malloc(64u * 1024u * 1024u);
    if (!buf) return 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t ncomp = gpu_graph_n_comp(g, il);
        if (ncomp) {
            const uint64_t n = (uint64_t)ncomp * attn_row;
            if (pulsar_gpu_tensor_read(g->layer_attn_comp_cache[il], 0, buf, n) == 0) { free(buf); return 0; }
            for (uint64_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            attn_rows += ncomp;
        }
        if (ratio == 4) {
            const uint32_t nidx = gpu_graph_n_index_comp(g, il);
            if (nidx) {
                const uint64_t n = (uint64_t)nidx * idx_row;
                if (pulsar_gpu_tensor_read(g->layer_index_comp_cache[il], 0, buf, n) == 0) { free(buf); return 0; }
                for (uint64_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ull; }
                idx_rows += nidx;
            }
        }
    }
    free(buf);
    fprintf(stderr, "  %-8s attn_comp_rows=%llu (%llu B/row)  idx_comp_rows=%llu (%llu B/row)  fnv=%016llx\n",
            tag, (unsigned long long)attn_rows, (unsigned long long)attn_row,
            (unsigned long long)idx_rows, (unsigned long long)idx_row,
            (unsigned long long)h);
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [L]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 2048;
    const int ctx = L + 4096;

    pulsar_engine *e = NULL; pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

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
    pulsar_tokens p; memset(&p, 0, sizeof p);
    p.v = base.v; p.len = p.cap = L;
    CHECK(pulsar_session_sync(a, &p, err, sizeof err) == 0, "A sync: %s", err);

    fprintf(stderr, "session_payload_gate: L=%d ctx=%d width=%d payload=%llu B\n",
            L, ctx, width, (unsigned long long)pulsar_session_payload_bytes(a));
    const uint64_t fnv_a = checksum_comp_caches(a, "saved");
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
    CHECK(pulsar_session_load_payload(b, fp, pbytes, err, sizeof err) == 0, "load: %s", err);
    fclose(fp);

    const uint64_t fnv_b = checksum_comp_caches(b, "restored");
    CHECK(fnv_b != 0, "checksum of session B failed");
    CHECK(fnv_a == fnv_b,
          "comp caches differ across the round trip: saved=%016llx restored=%016llx "
          "-- the attn (v4) or indexer (v5) rows did not survive as bytes",
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
          "The comp caches matched, so this is the RAW RING or the compressor state",
          ndiff, width, (double)worst);

    free(ref); free(got); free(base.v);
    printf("SESSION-PAYLOAD GATE: PASS (v%u, %llu B, comp fnv %016llx, logits identical)\n",
           (unsigned)PULSAR_SESSION_PAYLOAD_VERSION, (unsigned long long)pbytes,
           (unsigned long long)fnv_a);
    return 0;
}

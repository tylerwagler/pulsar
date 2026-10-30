/* L264: a session's KV as a segment chain (pulsar_kvchain.h). */
#include "pulsar_kvchain.h"
#include "pulsar_kvtext.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {
struct seg_ctx {
    pulsar_session *s;
    int G_prev, G;
    char key[41];       ///< the key the store will name it by (the same rule, pulsar_segstore_child_key)
    const char *text;
    const pulsar_kvchain_trailer *trailer;
};
int write_payload(FILE *fp, void *ud, char *err, size_t errlen) {
    const seg_ctx *c = (const seg_ctx *)ud;
    /* L334: the chain keeps its spans raw (the disk chain is a later phase); they enter as positions here */
    return pulsar_session_save_segment(c->s, fp, TokPos{c->G_prev}, TokPos{c->G}, c->key, err, errlen);
}
int write_trailer(FILE *fp, void *ud, char *err, size_t errlen) {
    const seg_ctx *c = (const seg_ctx *)ud;
    return c->trailer->write(fp, c->trailer->ud, c->text, err, errlen);
}
pulsar_tokens view(const pulsar_tokens *t, int from, int to) {
    pulsar_tokens v;
    memset(&v, 0, sizeof(v));
    v.v = t->v + from;
    v.len = v.cap = to - from;
    return v;
}
enum put_rc { PUT_FAILED, PUT_NO_CUT, PUT_STORED };
/* Store the bank's segment [G_prev, G) under `parent` (its key into key_out).  PUT_NO_CUT: [G_prev, G) is not a
 * segment span (an image block lies across G); PUT_FAILED: `r->err` says why.  A segment the store holds is not
 * written again -- the same key -- unless the store's copy lacks the state this bank holds at G (pulsar_segstore_put
 * replaces it in place); only a write counts in r. */
put_rc kvchain_put(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const pulsar_tokens *toks,
                   const char *parent, int G_prev, int G, const pulsar_kvchain_trailer *trailer,
                   pulsar_kvchain_persist_result *r, char key_out[41]) {
    uint64_t state_bytes = 0;
    const uint64_t payload = pulsar_session_segment_bytes(s, TokPos{G_prev}, TokPos{G}, &state_bytes);
    if (payload == 0) return PUT_NO_CUT;
    const pulsar_tokens span = view(toks, G_prev, G);
    size_t text_len = 0;
    char *text = pulsar_kvtext_render_tokens_text(e, &span, &text_len);
    if (!text) {
        snprintf(r->err, sizeof(r->err), "segment [%d,%d) could not be rendered", G_prev, G);
        return PUT_FAILED;
    }
    uint64_t tbytes = 0;
    if (trailer && !trailer->size(trailer->ud, text, &tbytes)) tbytes = 0;
    seg_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.s = s;
    ctx.G_prev = G_prev;
    ctx.G = G;
    ctx.text = text;
    ctx.trailer = trailer;
    pulsar_segstore_child_key(parent, text, text_len, ctx.key);
    pulsar_segstore_seg had;
    const bool held = pulsar_segstore_get(st, ctx.key, &had) && (had.state_bytes > 0 || state_bytes == 0);
    const bool ok = pulsar_segstore_put(st, parent, (uint32_t)G_prev, (uint32_t)G, text, text_len, payload, state_bytes,
                                        write_payload, tbytes, tbytes ? write_trailer : NULL, &ctx, key_out, r->err,
                                        sizeof(r->err));
    free(text);
    if (!ok) return PUT_FAILED;
    if (!held) {
        r->written++;
        r->bytes += payload + tbytes;
    }
    return PUT_STORED;
}
}  // namespace

int pulsar_kvchain_grid(pulsar_session *s) { return pulsar_session_prefill_cap(s).raw(); }

int pulsar_kvchain_segment_ends(int prev, int deepest, int grid, bool close, int *ends, int cap) {
    if (grid <= 0 || prev < 0 || deepest <= prev) return 0;
    int m = 0;
    for (int g = (prev / grid + 1) * grid; g <= deepest && m < cap; g += grid) ends[m++] = g;
    if (close && deepest % grid != 0 && m < cap) ends[m++] = deepest;
    return m;
}

int pulsar_kvchain_persist(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, int min_tokens, bool close,
                           const pulsar_kvchain_trailer *trailer, pulsar_kvchain_persist_result *out) {
    pulsar_kvchain_persist_result r;
    memset(&r, 0, sizeof(r));
    const pulsar_tokens *toks = st ? pulsar_session_tokens(s) : NULL;
    /* The chain ends at the bank's deepest grid checkpoint at or before the persist end: the rows below it are the
     * prefill's (a checkpoint is valid only while the bank holds that history), and its state is the tail's.  No
     * checkpoint is captured inside an image block.  L281: the end is past every block the session records (its
     * segment carries the record), short of the first it does not. */
    const TokPos end_pos = toks ? pulsar_session_persist_end(s) : TokPos{0};
    if (end_pos.is_none()) {
        snprintf(r.err, sizeof(r.err), "the history's image blocks are malformed; nothing persisted");
        if (out) *out = r;
        return 0;
    }
    const int end = end_pos.raw();
    const int deepest = end > 0 ? pulsar_session_checkpoint_best(s, end_pos).raw() : 0;
    const int grid = pulsar_kvchain_grid(s);
    if (deepest <= 0 || deepest < min_tokens || grid <= 0) {
        if (out) *out = r;
        return 0;
    }
    /* L323: the parent is the deepest stored GRID-ENDED chain of this history -- a tail is never extended */
    char parent[41] = "";
    int prev = 0;
    {
        const pulsar_tokens head = view(toks, 0, deepest);
        size_t full_len = 0;
        char *full = pulsar_kvtext_render_tokens_text(e, &head, &full_len);
        const pulsar_segstore_end grid_end = { (uint32_t)grid, false };
        pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
        const int c = full ? pulsar_segstore_lookup(st, full, full_len, &grid_end, chain, PULSAR_KVCHAIN_MAX) : 0;
        free(full);
        if (c > 0 && (int)chain[c - 1].G <= deepest) {
            const pulsar_tokens upto = view(toks, 0, (int)chain[c - 1].G);
            size_t upto_len = 0;
            free(pulsar_kvtext_render_tokens_text(e, &upto, &upto_len));
            if (upto_len == chain[c - 1].text_end) {
                memcpy(parent, chain[c - 1].key, 41);
                prev = (int)chain[c - 1].G;
                /* a stored segment of it without the state this bank holds at its end gets it, in place (the same
                 * key), so a branch parting in the cell above resumes there and not a cell lower */
                for (int i = 0; i < c; i++) {
                    const TokPos Gi{(int)chain[i].G};
                    if (chain[i].state_bytes || pulsar_session_checkpoint_best(s, Gi) != Gi)
                        continue;
                    char key[41];
                    if (kvchain_put(st, e, s, toks, i ? chain[i - 1].key : "", (int)chain[i].G_prev, (int)chain[i].G,
                                    trailer, &r, key) == PUT_FAILED) break;
                }
            }
        }
    }
    int ends[PULSAR_KVCHAIN_MAX];
    const int m = r.err[0] ? 0 : pulsar_kvchain_segment_ends(prev, deepest, grid, close, ends, PULSAR_KVCHAIN_MAX);
    for (int k = 0; k < m; k++) {
        char key[41];
        const put_rc rc = kvchain_put(st, e, s, toks, parent, prev, ends[k], trailer, &r, key);
        if (rc == PUT_NO_CUT && ends[k] != deepest) continue;   /* an image block across this grid point: no cut */
        if (rc != PUT_STORED) {
            if (!r.err[0]) snprintf(r.err, sizeof(r.err), "[%d,%d) is not a segment span", prev, ends[k]);
            break;
        }
        memcpy(parent, key, 41);
        prev = ends[k];
    }
    r.end = prev;
    memcpy(r.key, parent, 41);
    if (out) *out = r;
    return r.written;
}

int pulsar_kvchain_find(pulsar_segstore *st, const char *text, size_t text_len, const uint64_t *image_hashes,
                        int n_images, int min_tokens, pulsar_segstore_seg *chain, int cap, uint64_t *bytes, char *err,
                        size_t errlen) {
    if (bytes) *bytes = 0;
    if (err && errlen) err[0] = '\0';
    if (!st || !text) return 0;
    const pulsar_segstore_end state_end = { 0u, true };   /* L323: a restore needs the state at the chain's end */
    int n = pulsar_segstore_lookup(st, text, text_len, &state_end, chain, cap);
    /* L281: how far the request's images let the chain go -- every segment's image records must be the request's
     * next images, by content and in order; the chain stops at the last segment below the first that is not */
    {
        int k = 0;   /* the request's next image */
        for (int i = 0; i < n; i++) {
            uint64_t pb = 0, rec[64];
            uint32_t nr = 0;
            FILE *fp = pulsar_segstore_open_payload(st, chain[i].key, &pb);
            const int prc = fp ? pulsar_session_segment_images(fp, pb, rec, 64, &nr) : 1;
            if (fp) fclose(fp);
            bool match = prc == 0 && nr <= 64;
            for (uint32_t r = 0; match && r < nr; r++) {   /* k advances only past an image that matched */
                match = k < n_images && image_hashes && image_hashes[k] == rec[r];
                if (match) k++;
            }
            if (!match) {
                if (err && errlen)
                    snprintf(err, errlen, "chain stops at %u: segment [%u,%u) holds %s (image %d of the request)",
                             i ? chain[i - 1].G : 0u, chain[i].G_prev, chain[i].G,
                             prc != 0 ? "unreadable image records" : k >= n_images ? "an image the request does not carry"
                                                                                     : "a different image", k);
                n = i;
                break;
            }
        }
    }
    while (n > 0 && chain[n - 1].state_bytes == 0) n--;   /* L323: back to the last state below an image cut */
    if (n == 0 || (int)chain[n - 1].G < min_tokens) return 0;
    if (bytes) *bytes = pulsar_kvchain_load_bytes(chain, n);
    return n;
}

uint64_t pulsar_kvchain_load_bytes(const pulsar_segstore_seg *chain, int n) {
    uint64_t bytes = 0;
    for (int i = 0; i < n; i++) bytes += pulsar_kvchain_segment_load_bytes(chain, n, i);
    return bytes;
}

uint64_t pulsar_kvchain_cached_bytes(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n) {
    uint64_t bytes = 0;
    for (int i = 0; i < n; i++)
        bytes += pulsar_segstore_cached_bytes(st, chain[i].key,
                                              chain[i].payload_bytes - (i < n - 1 ? chain[i].state_bytes : 0u), true);
    return bytes;
}

uint64_t pulsar_kvchain_segment_load_bytes(const pulsar_segstore_seg *chain, int n, int i) {
    return chain[i].payload_bytes - (i < n - 1 ? chain[i].state_bytes : 0u) + chain[i].trailer_bytes;
}

int pulsar_kvchain_load_slice_with(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int *next, int k,
                                   pulsar_kvchain_segment_loader load, void *ud, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!st || !next || *next < 0 || *next >= n || k <= 0) {
        if (err && errlen) snprintf(err, errlen, "chain slice [%d, +%d) is not inside a chain of %d", next ? *next : -1,
                                    k, n);
        return 1;
    }
    const int end = *next + k < n ? *next + k : n;
    for (int i = *next; i < end; i++) {
        uint64_t pb = 0;
        FILE *fp = pulsar_segstore_open_payload(st, chain[i].key, &pb);
        char lerr[256] = "segment file unreadable";
        int G = 0;
        const int rc = fp ? load(ud, fp, pb, i == n - 1, &G, chain[i].key, lerr, sizeof(lerr)) : 1;
        if (fp) fclose(fp);
        if (rc != 0) {
            if (err && errlen)
                snprintf(err, errlen, "segment %s [%u,%u) failed to load (%s) -- dropped with its chain",
                         chain[i].key, chain[i].G_prev, chain[i].G, lerr);
            pulsar_segstore_drop(st, chain[i].key);
            return 1;
        }
        *next = i + 1;
    }
    return 0;
}

static int session_segment_loader(void *ud, FILE *fp, uint64_t bytes, bool last, int *G, const char *key, char *err,
                                  size_t errlen) {
    TokPos at{0};
    const int rc = pulsar_session_load_segment((pulsar_session *)ud, fp, bytes, last, G ? &at : NULL, key, err, errlen);
    if (G) *G = at.raw();
    return rc;
}

int pulsar_kvchain_load_slice(pulsar_segstore *st, pulsar_session *s, const pulsar_segstore_seg *chain, int n,
                              int *next, int k, char *err, size_t errlen) {
    const int rc = pulsar_kvchain_load_slice_with(st, chain, n, next, k, session_segment_loader, s, err, errlen);
    if (rc != 0) pulsar_session_invalidate(s);
    return rc;
}

int pulsar_kvchain_load_check(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                              const pulsar_segstore_seg *chain, int n, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    const pulsar_tokens *loaded = pulsar_session_tokens(s);
    /* L196: a chain whose text ends with an EOS its tokens never sampled (or the
     * reverse) would have the rest tokenised from the wrong byte. */
    if (!loaded || loaded->len != (int)chain[n - 1].G ||
        !pulsar_kvtext_text_ends_with_live(e, text, chain[n - 1].text_end, loaded)) {
        if (err && errlen)
            snprintf(err, errlen, "chain to %u disagrees with its tokens at the boundary -- dropped", chain[n - 1].G);
        pulsar_segstore_drop(st, chain[n - 1].key);
        pulsar_session_invalidate(s);
        return 0;
    }
    pulsar_segstore_touch(st, chain, n);
    return (int)chain[n - 1].G;
}

int pulsar_kvchain_load(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                        const pulsar_segstore_seg *chain, int n, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!st || !text || n <= 0) return 0;
    /* the one-shot load is the sliced load's one slice: the whole chain */
    int next = 0;
    if (pulsar_kvchain_load_slice(st, s, chain, n, &next, n, err, errlen) != 0) return 0;
    return pulsar_kvchain_load_check(st, e, s, text, chain, n, err, errlen);
}

int pulsar_kvchain_restore(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                           size_t text_len, const uint64_t *image_hashes, int n_images, int min_tokens,
                           pulsar_segstore_seg *chain, int cap, int *n_out, char *err, size_t errlen) {
    if (n_out) *n_out = 0;
    const int n = pulsar_kvchain_find(st, text, text_len, image_hashes, n_images, min_tokens, chain, cap, NULL, err,
                                      errlen);
    if (n == 0) return 0;
    /* a chain cut at an image still loads (find said why in `err`); a load failure says its own why instead */
    char lerr[384] = "";
    const int G = pulsar_kvchain_load(st, e, s, text, chain, n, lerr, sizeof(lerr));
    if (lerr[0] && err && errlen) snprintf(err, errlen, "%s", lerr);
    if (G > 0 && n_out) *n_out = n;
    return G;
}


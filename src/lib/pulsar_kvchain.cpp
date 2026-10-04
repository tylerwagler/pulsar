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
    return pulsar_session_save_segment(c->s, fp, c->G_prev, c->G, c->key, err, errlen);
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
}  // namespace

int pulsar_kvchain_persist(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, int min_tokens,
                           const pulsar_kvchain_trailer *trailer, pulsar_kvchain_persist_result *out) {
    pulsar_kvchain_persist_result r;
    memset(&r, 0, sizeof(r));
    const pulsar_tokens *toks = st ? pulsar_session_tokens(s) : NULL;
    /* The bank's grid checkpoints, deepest first: where new segments can end. */
    int Gs[PULSAR_KVCHAIN_MAX];
    int n = 0;
    if (toks)
        for (int G = pulsar_session_checkpoint_best(s, toks->len); G > 0 && n < PULSAR_KVCHAIN_MAX;
             G = pulsar_session_checkpoint_best(s, G - 1)) Gs[n++] = G;
    if (n == 0 || Gs[0] < min_tokens) {
        if (out) *out = r;
        return 0;
    }
    char parent[41] = "";
    int prev = 0;
    {
        const pulsar_tokens head = view(toks, 0, Gs[0]);
        size_t full_len = 0;
        char *full = pulsar_kvtext_render_tokens_text(e, &head, &full_len);
        pulsar_segstore_seg chain[PULSAR_KVCHAIN_MAX];
        const int c = full ? pulsar_segstore_lookup(st, full, full_len, chain, PULSAR_KVCHAIN_MAX) : 0;
        free(full);
        if (c > 0 && (int)chain[c - 1].G <= Gs[0]) {
            const pulsar_tokens upto = view(toks, 0, (int)chain[c - 1].G);
            size_t upto_len = 0;
            free(pulsar_kvtext_render_tokens_text(e, &upto, &upto_len));
            if (upto_len == chain[c - 1].text_end) {
                memcpy(parent, chain[c - 1].key, 41);
                prev = (int)chain[c - 1].G;
            }
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        const int G = Gs[k];
        if (G <= prev) continue;
        const pulsar_tokens span = view(toks, prev, G);
        size_t text_len = 0;
        char *text = pulsar_kvtext_render_tokens_text(e, &span, &text_len);
        uint64_t tbytes = 0;
        if (!text || (trailer && !trailer->size(trailer->ud, text, &tbytes))) tbytes = 0;
        seg_ctx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.s = s;
        ctx.G_prev = prev;
        ctx.G = G;
        ctx.text = text;
        ctx.trailer = trailer;
        if (text) pulsar_segstore_child_key(parent, text, text_len, ctx.key);
        const uint64_t payload = pulsar_session_segment_bytes(s, prev, G);
        char key[41];
        const bool ok = text && pulsar_segstore_put(st, parent, (uint32_t)prev, (uint32_t)G, text, text_len, payload,
                                                    write_payload, tbytes, tbytes ? write_trailer : NULL, &ctx, key,
                                                    r.err, sizeof(r.err));
        free(text);
        if (!ok) {
            if (!r.err[0]) snprintf(r.err, sizeof(r.err), "segment [%d,%d) could not be rendered", prev, G);
            break;
        }
        r.written++;
        r.bytes += payload + tbytes;
        memcpy(parent, key, 41);
        prev = G;
    }
    r.end = prev;
    memcpy(r.key, parent, 41);
    if (out) *out = r;
    return r.written;
}

int pulsar_kvchain_restore(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                           size_t text_len, int min_tokens, pulsar_segstore_seg *chain, int cap, int *n_out,
                           char *err, size_t errlen) {
    if (n_out) *n_out = 0;
    if (err && errlen) err[0] = '\0';
    if (!st || !text) return 0;
    const int n = pulsar_segstore_lookup(st, text, text_len, chain, cap);
    if (n == 0 || (int)chain[n - 1].G < min_tokens) return 0;
    for (int i = 0; i < n; i++) {
        uint64_t pb = 0;
        FILE *fp = pulsar_segstore_open_payload(st, chain[i].key, &pb);
        char lerr[256] = "segment file unreadable";
        int G = 0;
        const int rc = fp ? pulsar_session_load_segment(s, fp, pb, i == n - 1, &G, chain[i].key, lerr, sizeof(lerr)) : 1;
        if (fp) fclose(fp);
        if (rc != 0) {
            if (err && errlen)
                snprintf(err, errlen, "segment %s [%u,%u) failed to load (%s) -- dropped with its chain",
                         chain[i].key, chain[i].G_prev, chain[i].G, lerr);
            pulsar_segstore_drop(st, chain[i].key);
            pulsar_session_invalidate(s);
            return 0;
        }
    }
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
    if (n_out) *n_out = n;
    return (int)chain[n - 1].G;
}

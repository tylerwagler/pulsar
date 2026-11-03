/** @file
 *  The segment-chain persist / restore of a session's KV (pulsar_kvchain.h). */
/* L264: a session's KV as a segment chain (pulsar_kvchain.h). */
#include "pulsar_kvchain.h"
#include "pulsar_kvtext.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {
/** A non-owning pulsar_tokens over t[from, to); its `v` aliases t's storage and must not be freed.
 *  @param t source tokens
 *  @param from first index (inclusive)
 *  @param to end index (exclusive)
 *  @return the borrowed view */
pulsar_tokens view(const pulsar_tokens *t, int from, int to) {
    pulsar_tokens v;
    memset(&v, 0, sizeof(v));
    v.v = t->v + from;
    v.len = v.cap = to - from;
    return v;
}
/** L347 I3a: a reserve that waits for room in the store's RAM pool: `reserve` (a pulsar_segstore_reserve or
 *  _reserve_rp call) again after the writer made room, while it says there is none (-2) and the writer can wait.
 *  @tparam F       a callable `int()` returning the reserve's code
 *  @param writer   the writer (its wait_room)
 *  @param need     the file's bytes
 *  @param reserve  the reserve
 *  @param r        receives no_room when the writer could not make room
 *  @return the last reserve's code (-2: no room) */
template <typename F>
int reserve_with_room(const pulsar_kvchain_writer *writer, uint64_t need, F reserve, pulsar_kvchain_persist_result *r) {
    int rc = reserve();
    while (rc == -2 && writer->wait_room(writer->ud, need)) rc = reserve();
    if (rc == -2) r->no_room = true;
    return rc;
}
/** Outcome of kvchain_put. */
enum put_rc { PUT_FAILED, PUT_NO_CUT, PUT_STORED };
/* Store the bank's segment [G_prev, G) under `parent` (its key into key_out).  PUT_NO_CUT: [G_prev, G) is not a
 * segment span (an image block lies across G); PUT_FAILED: `r->err` says why.  A segment the store holds is not
 * written again -- the same key -- nor one being written; only a write counts in r.  L338 / L347 I3a: the rows are
 * copied out here into the job's image and the file is written by `writer`. */
/** Store one segment of the bank (see the comment above).
 *  @param st the segment store
 *  @param e the engine, for rendering the span's text
 *  @param s the session whose bank holds the rows
 *  @param toks the session's tokens; [G_prev, G) is rendered
 *  @param parent the parent segment's key ("" for a root)
 *  @param G_prev span start (token position)
 *  @param G span end (token position)
 *  @param trailer optional trailer writer (NULL = none)
 *  @param writer where the copied-out segment goes
 *  @param r counters updated on a write; `r->err` set on PUT_FAILED
 *  @param key_out receives the segment's key
 *  @return PUT_STORED, PUT_NO_CUT or PUT_FAILED */
put_rc kvchain_put(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const pulsar_tokens *toks,
                   const char *parent, int G_prev, int G, const pulsar_kvchain_trailer *trailer,
                   const pulsar_kvchain_writer *writer, pulsar_kvchain_persist_result *r, char key_out[41]) {
    const uint64_t payload = pulsar_session_segment_bytes(s, TokPos{G_prev}, TokPos{G}, NULL);
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
    pulsar_segstore_job *job = NULL;
    const uint64_t need = pulsar_segstore_seg_file_bytes(text_len, payload, tbytes);
    const int rc = reserve_with_room(writer, need, [&]() {
        return pulsar_segstore_reserve(st, parent, (uint32_t)G_prev, (uint32_t)G, text, text_len, payload, tbytes,
                                       key_out, &job, r->err, sizeof(r->err));
    }, r);
    if (rc <= 0) {   /* held or being written (0), or refused (-1 / -2, err) */
        free(text);
        return rc == 0 ? PUT_STORED : PUT_FAILED;
    }
    /* the bank's rows are copied out now (the bank moves on); the file is the writer's */
    bool ok = pulsar_session_copy_out_segment(s, TokPos{G_prev}, TokPos{G}, key_out, pulsar_segstore_job_payload(job),
                                              payload, r->err, sizeof(r->err)) == 0;
    if (ok && tbytes) {   /* a memory stream over the trailer's place in the image (one byte past it is the image's) */
        FILE *m = fmemopen(pulsar_segstore_job_trailer(job), (size_t)tbytes + 1u, "wb");
        ok = m && trailer->write(m, trailer->ud, text, r->err, sizeof(r->err)) == 0 && fflush(m) == 0 &&
             (uint64_t)ftello(m) == tbytes;
        if (m) fclose(m);
        if (!ok && !r->err[0]) snprintf(r->err, sizeof(r->err), "segment [%d,%d): its trailer was not written", G_prev,
                                        G);
    }
    free(text);
    if (!ok) {
        (void)pulsar_segstore_commit(st, job, false);
        return PUT_FAILED;
    }
    writer->submit(writer->ud, job);
    r->written++;
    r->bytes += payload + tbytes;
    return PUT_STORED;
}
/* L347: the restore point of segment `key` at G, copied out and handed to the writer (`key` may still be the
 * writer's) with `flags`, unless the store has one there -- the same key is the same state: it only gains the flags.
 * *wrote: a new one was handed over (its job in *job_out, not yet submitted).  False: `r->err` says why. */
/** Store one restore point of a persisted chain (see above).
 *  @param st     the segment store
 *  @param src    where the state at G comes from
 *  @param link   the segment ending at G (stored, or being written)
 *  @param flags  its PULSAR_SEGSTORE_RP_* flags
 *  @param end    it is the chain's end: its commit drops the restore points it supersedes
 *  @param writer where it goes
 *  @param r      counters updated on a write; `r->err` set on failure
 *  @return true when the store holds (or is writing) the restore point, or `src` holds no checkpoint at G */
bool kvchain_put_rp(pulsar_segstore *st, const pulsar_kvchain_rp_source *src, const pulsar_kvchain_link &link,
                    uint32_t flags, bool end, const pulsar_kvchain_writer *writer, pulsar_kvchain_persist_result *r) {
    pulsar_segstore_seg had;
    if (pulsar_segstore_get(st, link.key, &had) && had.rp_bytes > 0u) {
        if (pulsar_segstore_flag_rp(st, link.key, flags)) return true;
        snprintf(r->err, sizeof(r->err), "restore point %.40s at %d: its flags could not be written", link.key, link.G);
        return false;
    }
    const uint64_t bytes = src->bytes(src->ud, link.G);
    if (bytes == 0u) return true;
    pulsar_segstore_job *job = NULL;
    const int rc = reserve_with_room(writer, pulsar_segstore_rp_file_bytes(bytes), [&]() {
        return pulsar_segstore_reserve_rp(st, link.key, bytes, flags, &job, r->err, sizeof(r->err));
    }, r);
    if (rc <= 0) return rc == 0;   /* one in flight or waiting gained the flags (0), or refused */
    if (src->copy(src->ud, link.G, link.key, pulsar_segstore_job_payload(job), bytes, r->err, sizeof(r->err)) != 0) {
        (void)pulsar_segstore_commit(st, job, false);
        return false;
    }
    if (end) pulsar_segstore_job_supersede(job);
    writer->submit(writer->ud, job);
    r->rp_bytes += bytes;
    r->rp_written++;
    return true;
}
/** The installed bank's checkpoints as a pulsar_kvchain_rp_source::bytes.  @param ud the pulsar_session
 *  @param G the position  @return pulsar_session_restore_point_bytes at G (0: no checkpoint there) */
uint64_t session_rp_bytes(void *ud, int G) {
    return pulsar_session_restore_point_bytes((pulsar_session *)ud, TokPos{G});
}
/** The installed bank's checkpoints as a pulsar_kvchain_rp_source::copy.  @param ud the session  @param G the position
 *  @param key the segment's key  @param dst the restore point's place  @param bytes its size  @param err reason
 *  @param errlen bytes at err  @return pulsar_session_copy_out_restore_point's status: 0 = copied */
int session_rp_copy(void *ud, int G, const char *key, uint8_t *dst, uint64_t bytes, char *err, size_t errlen) {
    return pulsar_session_copy_out_restore_point((pulsar_session *)ud, TokPos{G}, key, dst, bytes, err, errlen);
}
/** The inline writer's wait_room: nothing is ever in flight, so the pool's room is what eviction makes now.
 *  @param ud the store  @param file_bytes the file's bytes  @return pulsar_segstore_ram_make_room */
bool inline_wait_room(void *ud, uint64_t file_bytes) {
    return pulsar_segstore_ram_make_room((pulsar_segstore *)ud, file_bytes);
}
/** The inline writer's submit: write (sealed), commit.  @param ud the store  @param job the filled job */
void inline_submit(void *ud, pulsar_segstore_job *job) {
    char err[256];
    const bool ok = pulsar_segstore_job_write(job, err, sizeof err);
    (void)pulsar_segstore_commit((pulsar_segstore *)ud, job, ok);
}
}  // namespace

pulsar_kvchain_writer pulsar_kvchain_writer_inline(pulsar_segstore *st) {
    pulsar_kvchain_writer w = { st, inline_wait_room, inline_submit };
    return w;
}

int pulsar_kvchain_grid(pulsar_session *s) { return pulsar_session_prefill_cap(s).raw(); }

int pulsar_kvchain_segment_ends(int prev, int deepest, int grid, bool tail, int *ends, int cap) {
    if (grid <= 0 || prev < 0 || deepest <= prev) return 0;
    int m = 0;
    for (int g = (prev / grid + 1) * grid; g <= deepest && m < cap; g += grid) ends[m++] = g;
    if (tail && deepest % grid != 0 && m < cap) ends[m++] = deepest;
    return m;
}

bool pulsar_kvchain_rp_periodic(int G, int grid) {
    return grid > 0 && G > 0 && G % (grid * PULSAR_KVCHAIN_RP_EVERY_CELLS) == 0;
}

void pulsar_kvchain_place_rps(pulsar_segstore *st, const pulsar_kvchain_link *links, int n, int grid, unsigned what,
                              const pulsar_kvchain_rp_source *src, const pulsar_kvchain_writer *writer,
                              pulsar_kvchain_persist_result *r) {
    const uint64_t dropped0 = pulsar_segstore_superseded_count(st);
    for (int i = 0; i < n; i++) {
        const bool end = i == n - 1 && (what & PULSAR_KVCHAIN_RP);
        uint32_t flags = pulsar_kvchain_rp_periodic(links[i].G, grid) ? PULSAR_SEGSTORE_RP_PERIODIC : 0u;
        if (end && (what & PULSAR_KVCHAIN_ANCHOR)) flags |= PULSAR_SEGSTORE_RP_ANCHOR;
        if (end && (what & PULSAR_KVCHAIN_PIN)) flags |= PULSAR_SEGSTORE_RP_PINNED;   /* L347 I3b */
        if (!end && flags == 0u) continue;
        pulsar_kvchain_persist_result one;
        memset(&one, 0, sizeof one);
        const bool ok = kvchain_put_rp(st, src, links[i], flags, end, writer, &one);
        r->rp_bytes += one.rp_bytes;
        r->rp_written += one.rp_written;
        r->no_room = r->no_room || one.no_room;
        if (!ok && !r->err[0]) memcpy(r->err, one.err, sizeof(r->err));
    }
    r->rp_dropped += (int)(pulsar_segstore_superseded_count(st) - dropped0);
}

int pulsar_kvchain_persist(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, int min_tokens, unsigned what,
                           const pulsar_kvchain_trailer *trailer, const pulsar_kvchain_writer *writer,
                           pulsar_kvchain_persist_result *out) {
    pulsar_kvchain_persist_result r;
    memset(&r, 0, sizeof(r));
    if (!writer || !writer->wait_room || !writer->submit) {
        snprintf(r.err, sizeof(r.err), "persist: no segment writer");
        if (out) *out = r;
        return 0;
    }
    if ((what & PULSAR_KVCHAIN_ANCHOR) && !(what & PULSAR_KVCHAIN_RP)) {
        snprintf(r.err, sizeof(r.err), "persist: an anchor is a restore point (PULSAR_KVCHAIN_ANCHOR without _RP)");
        if (out) *out = r;
        return 0;
    }
    if ((what & PULSAR_KVCHAIN_PIN) && !(what & PULSAR_KVCHAIN_ANCHOR)) {   /* L347 I3b */
        snprintf(r.err, sizeof(r.err), "persist: only an anchor is pinned (PULSAR_KVCHAIN_PIN without _ANCHOR)");
        if (out) *out = r;
        return 0;
    }
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
            }
        }
    }
    int ends[PULSAR_KVCHAIN_MAX];
    const int m = pulsar_kvchain_segment_ends(prev, deepest, grid, (what & PULSAR_KVCHAIN_TAIL) != 0u, ends,
                                              PULSAR_KVCHAIN_MAX);
    /* the chain's segments this persist passes, for the restore point policy: the new ones, or the stored end alone */
    pulsar_kvchain_link links[PULSAR_KVCHAIN_MAX];
    int n_links = 0;
    for (int k = 0; k < m; k++) {
        char key[41];
        const put_rc rc = kvchain_put(st, e, s, toks, parent, prev, ends[k], trailer, writer, &r, key);
        if (rc == PUT_NO_CUT && ends[k] != deepest) continue;   /* an image block across this grid point: no cut */
        if (rc != PUT_STORED) {
            if (!r.err[0]) snprintf(r.err, sizeof(r.err), "[%d,%d) is not a segment span", prev, ends[k]);
            break;
        }
        memcpy(parent, key, 41);
        prev = ends[k];
        memcpy(links[n_links].key, key, 41);
        links[n_links++].G = prev;
    }
    if (n_links == 0 && prev > 0) {
        memcpy(links[0].key, parent, 41);
        links[n_links++].G = prev;
    }
    r.end = prev;
    memcpy(r.key, parent, 41);
    /* L347 I2: the restore points, when the bank holds its checkpoints there -- also where the write stopped short
     * (its error is kept: the first one says why the chain is short) */
    const pulsar_kvchain_rp_source src = { s, session_rp_bytes, session_rp_copy };
    pulsar_kvchain_place_rps(st, links, n_links, grid, what, &src, writer, &r);
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
    while (n > 0 && chain[n - 1].rp_bytes == 0) n--;   /* L347: back to the last restore point below an image cut */
    if (n == 0 || (int)chain[n - 1].G < min_tokens) return 0;
    if (bytes) *bytes = pulsar_kvchain_load_bytes(chain, n);
    return n;
}

uint64_t pulsar_kvchain_load_bytes(const pulsar_segstore_seg *chain, int n) {
    uint64_t bytes = 0;
    for (int i = 0; i < n; i++) bytes += pulsar_kvchain_segment_load_bytes(chain, n, i);
    return bytes;
}

uint64_t pulsar_kvchain_resident_bytes(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n) {
    uint64_t bytes = 0;
    for (int i = 0; i < n; i++) bytes += pulsar_segstore_resident_bytes(st, chain[i].key, true, i == n - 1);
    return bytes;
}

uint64_t pulsar_kvchain_segment_load_bytes(const pulsar_segstore_seg *chain, int n, int i) {
    return chain[i].payload_bytes + chain[i].trailer_bytes + (i == n - 1 ? chain[i].rp_bytes : 0u);
}

namespace {
/* L347 I3b (plans/124 D7): a content fault drops only what it invalidates -- a segment's rows take the key and every
 * key below it (no chain through it can load); a restore point takes itself alone (the rows stay, for a chain that
 * ends further on).  Anything else drops nothing. */
/** Drop what a fault of segment `e` invalidates (nothing for a transient failure).
 *  @param st  the store
 *  @param e   the segment
 *  @param why PULSAR_SEGMENT_BAD_ROWS, _BAD_RP or _TRANSIENT
 *  @return the slice functions' code: 1 (dropped) or 2 (transient, kept) */
int kvchain_fault(pulsar_segstore *st, const pulsar_segstore_seg &e, pulsar_segment_status why) {
    if (why == PULSAR_SEGMENT_BAD_ROWS) pulsar_segstore_drop(st, e.key);
    else if (why == PULSAR_SEGMENT_BAD_RP) (void)pulsar_segstore_drop_rp(st, e.key);
    return why == PULSAR_SEGMENT_TRANSIENT ? 2 : 1;
}
/** What a fault drops, for the message.  @param why the fault  @return a phrase */
const char *kvchain_fault_word(pulsar_segment_status why) {
    return why == PULSAR_SEGMENT_BAD_ROWS ? "dropped with every segment below it"
         : why == PULSAR_SEGMENT_BAD_RP   ? "its restore point dropped, its rows kept"
                                          : "kept (not the stored bytes' fault)";
}
/** Map segment `i` of the chain (its rows, and its restore point when it is the last): the parts' bytes from RAM or
 *  read off disk, digest-checked by the store.  The caller unmaps them (kvchain_unmap) when this returns OK.
 *  @param st the store  @param chain the chain  @param n its length  @param i the segment
 *  @param seg receives the rows  @param pb their size  @param rp receives the restore point (last only)
 *  @param rb its size  @param err receives the reason  @param errlen bytes at err
 *  @return PULSAR_SEGMENT_OK; _BAD_ROWS / _BAD_RP (a part missing, unreadable or bad); _TRANSIENT (no RAM room now) */
pulsar_segment_status kvchain_map(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int i,
                                  const uint8_t **seg, uint64_t *pb, const uint8_t **rp, uint64_t *rb, char *err,
                                  size_t errlen) {
    *seg = *rp = NULL;
    *pb = *rb = 0;
    const int mrc = pulsar_segstore_map_part(st, chain[i].key, PULSAR_SEGSTORE_PART_ROWS, seg, pb, NULL, err, errlen);
    if (mrc != 0) return mrc == 2 ? PULSAR_SEGMENT_TRANSIENT : PULSAR_SEGMENT_BAD_ROWS;
    if (i != n - 1) return PULSAR_SEGMENT_OK;
    const int rrc = pulsar_segstore_map_part(st, chain[i].key, PULSAR_SEGSTORE_PART_RP, rp, rb, NULL, err, errlen);
    if (rrc == 0) return PULSAR_SEGMENT_OK;
    pulsar_segstore_unmap_part(st, chain[i].key, PULSAR_SEGSTORE_PART_ROWS);
    return rrc == 2 ? PULSAR_SEGMENT_TRANSIENT : PULSAR_SEGMENT_BAD_RP;
}
/** End a kvchain_map that returned OK.  @param st the store  @param chain the chain  @param n its length
 *  @param i the segment */
void kvchain_unmap(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int i) {
    pulsar_segstore_unmap_part(st, chain[i].key, PULSAR_SEGSTORE_PART_ROWS);
    if (i == n - 1) pulsar_segstore_unmap_part(st, chain[i].key, PULSAR_SEGSTORE_PART_RP);
}
/** The slice functions' argument check.  @param st the store  @param n the chain's length  @param next the slice's
 *  first segment  @param k its length  @param err receives the reason  @param errlen bytes at err
 *  @return true when [*next, *next + k) is a slice of the chain */
bool kvchain_slice_ok(const pulsar_segstore *st, int n, const int *next, int k, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (st && next && *next >= 0 && *next < n && k > 0) return true;
    if (err && errlen)
        snprintf(err, errlen, "chain slice [%d, +%d) is not inside a chain of %d", next ? *next : -1, k, n);
    return false;
}
}  // namespace

int pulsar_kvchain_verify_slice_with(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int *next, int k,
                                     pulsar_kvchain_segment_checker check, void *ud, char *err, size_t errlen) {
    if (!kvchain_slice_ok(st, n, next, k, err, errlen)) return 2;
    const int end = *next + k < n ? *next + k : n;
    for (int i = *next; i < end; i++) {
        const uint8_t *seg = NULL, *rp = NULL;
        uint64_t pb = 0, rb = 0;
        char lerr[256] = "";
        pulsar_segment_status why = kvchain_map(st, chain, n, i, &seg, &pb, &rp, &rb, lerr, sizeof lerr);
        if (why == PULSAR_SEGMENT_OK) {
            why = check(ud, seg, pb, i == n - 1, rp, rb, lerr, sizeof lerr);
            kvchain_unmap(st, chain, n, i);
        }
        if (why != PULSAR_SEGMENT_OK) {
            if (err && errlen)
                snprintf(err, errlen, "segment %s [%u,%u) failed its check before the load (%s) -- %s", chain[i].key,
                         chain[i].G_prev, chain[i].G, lerr, kvchain_fault_word(why));
            return kvchain_fault(st, chain[i], why);
        }
        *next = i + 1;
    }
    return 0;
}

int pulsar_kvchain_load_slice_with(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int *next, int k,
                                   pulsar_kvchain_segment_loader load, void *ud, char *err, size_t errlen) {
    if (!kvchain_slice_ok(st, n, next, k, err, errlen)) return 2;
    const int end = *next + k < n ? *next + k : n;
    for (int i = *next; i < end; i++) {
        const bool last = i == n - 1;
        const uint8_t *seg = NULL, *rp = NULL;
        uint64_t pb = 0, rb = 0;
        char lerr[256] = "";
        /* L347 I3a: each part from its RAM copy, or read off disk and digest-checked by the store */
        pulsar_segment_status why = kvchain_map(st, chain, n, i, &seg, &pb, &rp, &rb, lerr, sizeof lerr);
        int G = 0;
        if (why == PULSAR_SEGMENT_OK) {
            why = load(ud, seg, pb, last, rp, rb, &G, chain[i].key, lerr, sizeof(lerr));
            kvchain_unmap(st, chain, n, i);
        }
        if (why != PULSAR_SEGMENT_OK) {
            if (err && errlen)
                snprintf(err, errlen, "segment %s [%u,%u) failed to load (%s) -- %s", chain[i].key, chain[i].G_prev,
                         chain[i].G, lerr, kvchain_fault_word(why));
            return kvchain_fault(st, chain[i], why);
        }
        *next = i + 1;
    }
    return 0;
}

void pulsar_kvchain_pin(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, bool pin) {
    for (int i = 0; i < n; i++) {
        if (pin) pulsar_segstore_pin(st, chain[i].key);
        else pulsar_segstore_unpin(st, chain[i].key);
    }
}

/** The pulsar_kvchain_segment_checker over a session's installed bank (pulsar_session_check_segment).
 *  @param ud the pulsar_session  @param seg the segment's payload  @param bytes its size  @param last the chain's last
 *  @param rp its restore point when `last`  @param rp_bytes its size  @param err receives the fault
 *  @param errlen size of `err`  @return pulsar_session_check_segment's status */
static pulsar_segment_status session_segment_checker(void *ud, const uint8_t *seg, uint64_t bytes, bool last,
                                                     const uint8_t *rp, uint64_t rp_bytes, char *err, size_t errlen) {
    return pulsar_session_check_segment((pulsar_session *)ud, seg, bytes, last, rp, rp_bytes, err, errlen);
}

int pulsar_kvchain_verify_slice(pulsar_segstore *st, pulsar_session *s, const pulsar_segstore_seg *chain, int n,
                                int *next, int k, char *err, size_t errlen) {
    return pulsar_kvchain_verify_slice_with(st, chain, n, next, k, session_segment_checker, s, err, errlen);
}

/** The pulsar_kvchain_segment_loader that loads into a session's installed bank (pulsar_session_load_segment).
 *  @param ud       the pulsar_session
 *  @param seg      the segment's payload
 *  @param bytes    the payload's size
 *  @param last     restore the state at the segment's end too
 *  @param rp       the restore point at its end when `last`, else NULL
 *  @param rp_bytes its size
 *  @param G        receives the position the bank stands at, or NULL
 *  @param key      the segment's key
 *  @param err      receives a reason on failure
 *  @param errlen   size of `err`
 *  @return pulsar_session_load_segment's status */
static pulsar_segment_status session_segment_loader(void *ud, const uint8_t *seg, uint64_t bytes, bool last,
                                                    const uint8_t *rp, uint64_t rp_bytes, int *G, const char *key,
                                                    char *err, size_t errlen) {
    TokPos at{0};
    const pulsar_segment_status rc = pulsar_session_load_segment((pulsar_session *)ud, seg, bytes, last, rp, rp_bytes,
                                                                 G ? &at : NULL, key, err, errlen);
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


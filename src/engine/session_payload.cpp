#include "pulsar_engine_internal.h"

/* L281: one image block record in a payload or a segment -- u32 start, u32 end, u64 content as two u32, and
 * (L268) the block's 2D grid, u32 rows and u32 columns */
#define SEGMENT_IMAGE_U32 6u
static void image_rec_pack(const pulsar_image_block *b, uint32_t (&rec)[SEGMENT_IMAGE_U32]) {
    rec[0] = b->start;
    rec[1] = b->end;
    rec[2] = (uint32_t)b->content;
    rec[3] = (uint32_t)(b->content >> 32);
    rec[4] = b->grid_h;
    rec[5] = b->grid_w;
}
static pulsar_image_block image_rec_unpack(const uint32_t (&rec)[SEGMENT_IMAGE_U32]) {
    return (pulsar_image_block){ rec[0], rec[1], (uint64_t)rec[2] | (uint64_t)rec[3] << 32, rec[4], rec[5] };
}
#include "lib/pulsar_writeback.h"

void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}



static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}



static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}



/* ---- payload digest ------------------------------------------------------
 *
 * L219: a 64-bit rolling digest over every payload byte, appended after the
 * data.  The store is atomic, but media bit-rot or an offline edit inside a
 * multi-GB payload used to load silently: a flipped KV row decrypts as
 * plausible values and a lowered n_comp quietly drops compressed recall.  With
 * the digest a damaged file is a cache miss (re-prefill), not wrong attention.
 * Format v10 added the trailing 8 bytes; v9 files refuse on the version.
 *
 * The update is CHUNK-INDEPENDENT: bytes are folded 8 at a time with a partial
 * word carried in `tail`, so a writer that hashes one 64 MB buffer and a reader
 * that hashes it in ten pieces compute the same digest.  Not cryptographic --
 * integrity against corruption, not tampering. */
typedef struct {
    uint64_t h;      ///< running digest
    uint64_t tail;   ///< pending bytes, first-arrived at the low end
    uint32_t ntail;  ///< 0..7 pending bytes
} payload_digest;

static void payload_digest_init(payload_digest *d) {
    d->h = UINT64_C(1469598103934665603);
    d->tail = 0;
    d->ntail = 0;
}

static uint64_t payload_digest_mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= UINT64_C(0x9E3779B97F4A7C15);
    return (h << 31) | (h >> 33);
}

static void payload_digest_update(payload_digest *d, const void *ptr, uint64_t bytes) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (d->ntail != 0 && bytes != 0) {
        d->tail |= (uint64_t)(*p++) << (8u * d->ntail);
        d->ntail++;
        bytes--;
        if (d->ntail == 8u) {
            d->h = payload_digest_mix(d->h, d->tail);
            d->tail = 0;
            d->ntail = 0;
        }
    }
    while (bytes >= 8u) {
        uint64_t w = 0;
        memcpy(&w, p, sizeof(w));
        d->h = payload_digest_mix(d->h, w);
        p += 8;
        bytes -= 8;
    }
    while (bytes != 0) {
        d->tail |= (uint64_t)(*p++) << (8u * d->ntail);
        d->ntail++;
        bytes--;
    }
}

static uint64_t payload_digest_final(const payload_digest *d) {
    uint64_t h = d->h;
    if (d->ntail != 0) h = payload_digest_mix(h, d->tail);
    return h;
}



/* One payload stream: the file plus the digest covering everything read or
 * written through it.  A private type so the file and its digest cannot drift
 * apart: only these helpers touch the stream, and every one of them folds the
 * bytes it moved. */
typedef struct {
    FILE *fp;
    payload_digest digest;
    pulsar_writeback wb;   ///< write side only: written pages go to disk and leave the cache as they stream (L261)
} payload_io;

static int payload_write_bytes(payload_io *io, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, io->fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        payload_digest_update(&io->digest, p, n);
        p += n;
        bytes -= n;
    }
    pulsar_writeback_step(&io->wb);
    return 0;
}



static int payload_read_bytes(payload_io *io, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = (uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, io->fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        payload_digest_update(&io->digest, p, n);
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



static int payload_write_u32(payload_io *io, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(io, b, sizeof(b), err, errlen);
}



static int payload_read_u32(payload_io *io, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (payload_read_bytes(io, b, sizeof(b), remaining, err, errlen) != 0) return 1;
    *v = payload_get_u32(b);
    return 0;
}

/* L281: an image section -- the count, then each block's record (payload, segment and kv-state payload) */
static int payload_write_images(payload_io *io, const pulsar_image_block *b, uint32_t n, char *err, size_t errlen) {
    if (payload_write_u32(io, n, err, errlen) != 0) return 1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rec[SEGMENT_IMAGE_U32];
        image_rec_pack(&b[i], rec);
        for (uint32_t k = 0; k < SEGMENT_IMAGE_U32; k++)
            if (payload_write_u32(io, rec[k], err, errlen) != 0) return 1;
    }
    return 0;
}



/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(payload_io *io, const pulsar_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        /* Name the numbers: a bare "smaller than the payload" says nothing about
         * WHICH span overran, and this is the refusal a checkpoint hits when a
         * payload's live-row sizing and an allocation disagree (L218 s122). */
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "session tensor is smaller than the payload (offset %llu + %llu > %llu bytes)",
                 (unsigned long long)offset, (unsigned long long)bytes,
                 (unsigned long long)(tensor ? pulsar_gpu_tensor_bytes(tensor) : 0));
        payload_set_err(err, errlen, msg);
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (pulsar_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(io, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static int payload_read_tensor_span(payload_io *io, pulsar_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "session tensor is smaller than the payload (offset %llu + %llu > %llu bytes)",
                 (unsigned long long)offset, (unsigned long long)bytes,
                 (unsigned long long)(tensor ? pulsar_gpu_tensor_bytes(tensor) : 0));
        payload_set_err(err, errlen, msg);
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(io, buf, n, remaining, err, errlen) != 0) return 1;
        if (pulsar_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}



/* ===== L264 S4 / L265: disk segments ====================================
 * One segment = tokens [G_prev, G), every append-only POOL's rows for those
 * positions, and the grid checkpoint at G (pulsar.h, pulsar_session_save_segment).
 * What the pools and the slot are is the model's (kv_state.h); this framing is
 * not.  The header pins the model's layout by a digest of what makes the bytes
 * readable -- the state model's name, its grid, the slot size and every pool's
 * granularity and row width -- so a segment written under another layout is
 * refused, never misread. */
#define SEGMENT_U32_FIELDS 8u
/* L281 (v3): after the span's tokens, the image blocks lying wholly inside the span -- a u32 count, then per block
 * u32 start, u32 end, u64 content (two u32) -- so a chain carries rows past an image together with WHICH image they
 * are (image_identity.cpp).  A span boundary is a grid checkpoint, never inside a block, so every block a chain holds
 * is in exactly one segment.  Each rank's copy carries them: a worker's load sets the same identity as the leader's. */

/* the session's records of the blocks inside [G_prev, G) */
static uint32_t segment_images(const pulsar_session *s, uint32_t G_prev, uint32_t G, const pulsar_image_block **first) {
    uint32_t n = 0;
    *first = NULL;
    for (uint32_t i = 0; i < s->live_images.n; i++) {
        const pulsar_image_block *b = &s->live_images.b[i];
        if (b->start >= G_prev && b->end <= G) {
            if (!n) *first = b;
            n++;
        }
    }
    return n;
}

/* The bank holds nothing: a chain's root replaces its history, a failed load leaves it empty.  The
 * session's invalidate forgets the view (a family whose invalidate leaves the device state alone has its
 * lanes written by the chain's last restore and its counters by set_frontier_stale), and the bank's
 * checkpoints go with the history they described. */
static void segment_clear(pulsar_session *s, pulsar_ckpt_store *st, uint32_t bank) {
    s->live_images.n = 0;   /* L281: the bank holds no block */
    pulsar_session_family_invalidate(s);
    pulsar_ckpt_drop_bank(st, bank);
}

static uint32_t segment_pools(pulsar_ckpt_store *st, pulsar_kv_pool *pools) {
    const uint32_t n = st->ops->pools(st->state, pools, PULSAR_KV_POOLS_MAX);
    return n <= PULSAR_KV_POOLS_MAX ? n : 0u;   /* a model past the bound refuses every segment */
}

static uint64_t segment_layout_digest(pulsar_ckpt_store *st) {
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; } };
    for (const char *c = st->ops->name; *c; c++) mix((uint8_t)*c);
    if (st->artifact) mix(st->artifact);   /* 0 leaves DeepSeek's digests as they were */
    mix(st->ops->resume_grid);
    mix(st->slot_bytes);
    mix(n);
    for (uint32_t i = 0; i < n; i++) { mix(pools[i].tokens_per_row); mix(pools[i].row_bytes); }
    return h;
}

/* Pool rows [t0 / tokens_per_row, end(t1)) of every pool: a segment's span (grid points, whole rows) or a
 * kv-state payload's [0, frontier) -- its open row too, unless the pool writes whole rows only (kv_state.h).
 * The one sizing and the one copy of pool rows either carries. */
static uint32_t pool_row(const pulsar_kv_pool &p, uint32_t t) {
    return (t + (p.whole_rows ? 0u : p.tokens_per_row - 1u)) / p.tokens_per_row;
}

static uint64_t pools_span_bytes(const pulsar_kv_pool *pools, uint32_t n, uint32_t t0, uint32_t t1) {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < n; i++) bytes += (uint64_t)(pool_row(pools[i], t1) - t0 / pools[i].tokens_per_row) * pools[i].row_bytes;
    return bytes;
}

static int pools_span_io(payload_io *io, const pulsar_kv_pool *pools, uint32_t n, uint32_t t0, uint32_t t1, bool write,
                         uint8_t *buf, uint64_t *remaining, char *err, size_t errlen) {
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n; i++) {
        const uint32_t row0 = t0 / pools[i].tokens_per_row, rows = pool_row(pools[i], t1) - row0;
        if (!rows) continue;
        if (!pools[i].rows) {   /* L284 #3: an evicted bank's pools are gone until alloc_physical re-backs them */
            payload_set_err(err, errlen, "a KV pool of the installed bank has no physical (an evicted bank)");
            return 1;
        }
        const uint64_t off = (uint64_t)row0 * pools[i].row_bytes, bytes = (uint64_t)rows * pools[i].row_bytes;
        rc = write ? payload_write_tensor_span(io, pools[i].rows, off, bytes, buf, PULSAR_SESSION_IO_CHUNK, err, errlen)
                   : payload_read_tensor_span(io, pools[i].rows, off, bytes, buf, PULSAR_SESSION_IO_CHUNK, remaining,
                                              err, errlen);
    }
    return rc;
}

static bool segment_span_ok(const pulsar_ckpt_store *st, uint32_t G_prev, uint32_t G) {
    const uint32_t grid = st->ops->resume_grid;
    return G > G_prev && G_prev % grid == 0u && G % grid == 0u;
}

/* the bytes of a segment over [G_prev, G) carrying `n_images` image records (a v2 segment has no image section:
 * written before L281, when no chain held a block, it loads as carrying none; v3, L281's 4-u32 records, never
 * shipped and is refused) */
static uint64_t segment_bytes_for(pulsar_ckpt_store *st, uint32_t G_prev, uint32_t G, uint32_t n_images,
                                  uint32_t version = PULSAR_SESSION_SEGMENT_VERSION) {
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    uint64_t bytes = (uint64_t)SEGMENT_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)(G - G_prev) * sizeof(uint32_t);
    if (version >= 4u) bytes += sizeof(uint32_t) + (uint64_t)n_images * SEGMENT_IMAGE_U32 * sizeof(uint32_t);
    bytes += st->slot_bytes;
    bytes += pools_span_bytes(pools, n, G_prev, G);
    return bytes + sizeof(uint64_t);   /* the trailing digest */
}

/* L281: a stored segment's image records, read from its payload's head without loading it (a v2 segment carries
 * none).  The bytes are the payload digest's to vouch for; this only decides how far a restore goes, and the load
 * that follows re-reads and checks everything. */
int pulsar_session_segment_images(FILE *fp, uint64_t bytes, uint64_t *hashes, uint32_t cap, uint32_t *n_out) {
    *n_out = 0;
    uint32_t h[SEGMENT_U32_FIELDS];
    if (!fp || bytes < sizeof(h) || fread(h, sizeof(uint32_t), SEGMENT_U32_FIELDS, fp) != SEGMENT_U32_FIELDS) return 1;
    if (h[0] != PULSAR_SESSION_SEGMENT_MAGIC || h[7] <= h[6]) return 1;
    if (h[1] == 2u) return 0;
    if (h[1] != PULSAR_SESSION_SEGMENT_VERSION) return 1;
    if (fseek(fp, (long)((uint64_t)(h[7] - h[6]) * sizeof(uint32_t)), SEEK_CUR) != 0) return 1;
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 || n > PULSAR_IMAGE_BLOCKS_MAX) return 1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rec[SEGMENT_IMAGE_U32];
        if (fread(rec, sizeof(uint32_t), SEGMENT_IMAGE_U32, fp) != SEGMENT_IMAGE_U32) return 1;
        if (i < cap) hashes[i] = (uint64_t)rec[2] | (uint64_t)rec[3] << 32;
    }
    *n_out = n;
    return 0;
}

uint64_t pulsar_session::segment_bytes(uint32_t G_prev, uint32_t G) {
    pulsar_ckpt_store *st = pulsar_session_kv_store(this);
    if (!segment_span_ok(st, G_prev, G)) return 0;
    const pulsar_image_block *first = NULL;
    return segment_bytes_for(st, G_prev, G, segment_images(this, G_prev, G, &first));
}

int pulsar_session::save_segment(FILE *fp, uint32_t G_prev, uint32_t G, char *err, size_t errlen) {
    auto *s = this;
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    if (!fp || !s->checkpoint_valid || !segment_span_ok(st, G_prev, G) || G > (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "save segment: the session does not hold that grid span");
        return 1;
    }
    if (!st->ops->holds(st->state, G)) {
        payload_set_err(err, errlen, "save segment: the bank's rows do not reach the segment's end");
        return 1;
    }
    pulsar_gpu_tensor *slab = NULL;
    uint64_t slot_off = 0;
    if (!pulsar_ckpt_locate(st, bank, G, &slab, &slot_off)) {
        payload_set_err(err, errlen, "save segment: the bank holds no grid checkpoint at the segment's end");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "save segment: failed to synchronize accelerator");
        return 1;
    }
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, fp);
    const uint64_t layout = segment_layout_digest(st);
    const uint32_t header[SEGMENT_U32_FIELDS] = {
        PULSAR_SESSION_SEGMENT_MAGIC, PULSAR_SESSION_SEGMENT_VERSION,
        (uint32_t)layout, (uint32_t)(layout >> 32), (uint32_t)st->slot_bytes, st->ops->resume_grid, G_prev, G,
    };
    for (uint32_t i = 0; i < SEGMENT_U32_FIELDS; i++)
        if (payload_write_u32(&io, header[i], err, errlen) != 0) return 1;
    for (uint32_t p = G_prev; p < G; p++)
        if (payload_write_u32(&io, (uint32_t)s->checkpoint.v[p], err, errlen) != 0) return 1;
    {
        const pulsar_image_block *b = NULL;
        const uint32_t n_img = segment_images(s, G_prev, G, &b);
        if (payload_write_images(&io, b, n_img, err, errlen) != 0) return 1;
    }
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = payload_write_tensor_span(&io, slab, slot_off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    if (rc == 0) rc = pools_span_io(&io, pools, n, G_prev, G, true, buf, NULL, err, errlen);
    free(buf);
    if (rc != 0) return rc;
    const uint64_t dg = payload_digest_final(&io.digest);
    if (fwrite(&dg, 1, sizeof(dg), fp) != sizeof(dg)) {
        payload_set_err(err, errlen, "save segment: failed to write the digest");
        return 1;
    }
    pulsar_writeback_finish(&io.wb);
    return 0;
}

int pulsar_session::load_segment(FILE *fp, uint64_t bytes, bool last, uint32_t *G_out, char *err, size_t errlen) {
    auto *s = this;
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    if (G_out) *G_out = 0u;
    if (!fp) { payload_set_err(err, errlen, "load segment: no stream"); return 1; }
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, NULL);
    uint64_t remaining = bytes;
    uint32_t h[SEGMENT_U32_FIELDS];
    for (uint32_t i = 0; i < SEGMENT_U32_FIELDS; i++)
        if (payload_read_u32(&io, &h[i], &remaining, err, errlen) != 0) return 1;
    if (h[0] != PULSAR_SESSION_SEGMENT_MAGIC || (h[1] != PULSAR_SESSION_SEGMENT_VERSION && h[1] != 2u)) {
        payload_set_err(err, errlen, "load segment: unsupported segment version");
        return 1;
    }
    const uint64_t layout = segment_layout_digest(st);
    if (h[2] != (uint32_t)layout || h[3] != (uint32_t)(layout >> 32) || h[4] != (uint32_t)st->slot_bytes ||
        h[5] != st->ops->resume_grid) {
        payload_set_err(err, errlen, "load segment: written for a different state layout");
        return 1;
    }
    const uint32_t G_prev = h[6], G = h[7];
    if (!segment_span_ok(st, G_prev, G) || G >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "load segment: span is not a grid span inside this context");
        return 1;
    }
    /* A chain loads in order: a root resets the bank, every later segment
     * continues one that stands exactly at its start. */
    if (G_prev == 0u) {
        segment_clear(s, st, bank);
    } else {
        char why[192];
        if (!s->checkpoint_valid || (uint32_t)s->checkpoint.len != G_prev ||
            !st->ops->stands_at(st->state, G_prev, why, sizeof(why))) {
            payload_set_err(err, errlen, "load segment: the bank does not stand at the segment's start");
            return 1;
        }
    }
    token_vec toks = {0};
    for (uint32_t p = G_prev; p < G; p++) {
        uint32_t t = 0;
        if (payload_read_u32(&io, &t, &remaining, err, errlen) != 0) { token_vec_free(&toks); return 1; }
        token_vec_push(&toks, (int)t);
    }
    /* L281: the span's image records, appended to the identity at commit (the size check needs their count) */
    pulsar_image_block imgs[PULSAR_IMAGE_BLOCKS_MAX];
    uint32_t n_img = 0;
    if (h[1] >= 4u && payload_read_u32(&io, &n_img, &remaining, err, errlen) != 0) { token_vec_free(&toks); return 1; }
    if (n_img > PULSAR_IMAGE_BLOCKS_MAX - s->live_images.n) {
        token_vec_free(&toks);
        payload_set_err(err, errlen, "load segment: more image blocks than a session holds");
        return 1;
    }
    for (uint32_t i = 0; i < n_img; i++) {
        uint32_t rec[SEGMENT_IMAGE_U32];
        for (uint32_t k = 0; k < SEGMENT_IMAGE_U32; k++)
            if (payload_read_u32(&io, &rec[k], &remaining, err, errlen) != 0) { token_vec_free(&toks); return 1; }
        imgs[i] = image_rec_unpack(rec);
        if (imgs[i].start < G_prev || imgs[i].end > G || imgs[i].end <= imgs[i].start) {
            token_vec_free(&toks);
            payload_set_err(err, errlen, "load segment: an image record lies outside its span");
            return 1;
        }
    }
    if (bytes != segment_bytes_for(st, G_prev, G, n_img, h[1])) {
        token_vec_free(&toks);
        payload_set_err(err, errlen, "load segment: size does not match its span");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&toks);
        payload_set_err(err, errlen, "load segment: failed to synchronize accelerator");
        return 1;
    }
    /* Device writes begin: from here a failure leaves the bank holding nothing. */
    auto fail = [&]() { token_vec_free(&toks); segment_clear(s, st, bank); return 1; };
    pulsar_gpu_tensor *slab = NULL;
    uint64_t slot_off = 0;
    uint32_t slot = 0;
    if (!pulsar_ckpt_claim(st, bank, G, &slab, &slot_off, &slot)) {
        payload_set_err(err, errlen, "load segment: no grid-checkpoint slot");
        return fail();
    }
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = payload_read_tensor_span(&io, slab, slot_off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK,
                                      &remaining, err, errlen);
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    if (rc == 0) rc = pools_span_io(&io, pools, n, G_prev, G, false, buf, &remaining, err, errlen);
    free(buf);
    if (rc != 0) return fail();
    uint64_t stored_dg = 0;
    if (remaining != sizeof(uint64_t) || fread(&stored_dg, 1, sizeof(stored_dg), fp) != sizeof(stored_dg) ||
        stored_dg != payload_digest_final(&io.digest)) {
        payload_set_err(err, errlen, "load segment: digest mismatch (corrupt segment)");
        return fail();
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "load segment: failed to synchronize accelerator after the copy");
        return fail();
    }
    /* Commit: the history, the frontier, then the checkpoint (after the frontier
     * write, which could otherwise drop it).  The lanes and window describe no
     * position yet -- stale until the last segment restores its checkpoint. */
    for (int i = 0; i < toks.len; i++) token_vec_push(&s->checkpoint, toks.v[i]);
    token_vec_free(&toks);
    for (uint32_t i = 0; i < n_img; i++) s->live_images.b[s->live_images.n++] = imgs[i];   /* L281 */
    st->ops->set_frontier_stale(st->state, G);
    pulsar_ckpt_commit(st, bank, slot, G);
    s->prefill_frontier = (int)G;   /* L195: a prefill reached G (the family's own counter is set_frontier_stale's) */
    s->checkpoint_valid = true;
    /* the chain's last checkpoint: the family's standalone restore where it has one (DeepSeek: the history and the
     * drafter window with it), else the store's alone (Qwen: its sync owns the rest) */
    const pulsar_family_bank_ops *bops = s->engine->family->banks;
    if (last && !(bops->restore_checkpoint ? bops->restore_checkpoint(s, G) : pulsar_ckpt_restore(st, bank, G))) {
        payload_set_err(err, errlen, "load segment: restoring the chain's last checkpoint failed");
        segment_clear(s, st, bank);
        return 1;
    }
    if (last) s->logits_stale = true;   /* a restore moves the KV, not the logits (restore_checkpoint sets it too) */
    if (G_out) *G_out = G;
    return 0;
}



/* ===== L284: the kv-state payload =========================================
 * Every model's session payload is a segment's parts at the session's frontier T (pulsar.h,
 * PULSAR_SESSION_KV_PAYLOAD_MAGIC): the same layout digest, image section, slot walk, pool copy and digested
 * stream, so the bytes have one authority -- the state model -- and the format carries nothing a segment's
 * reader does not already validate.  What it adds over a segment is the frontier: the state model's FRONTIER
 * walk at T (not a grid point's), the pools' open rows (where a pool has them) and the trailing pools, the
 * logits, the prefill frontier, and the resume checkpoint a sync that does not extend T restores. */
#define KVP_U32_FIELDS 8u
/* L284: DeepSeek's retired graph-format payload (v3..v15, "DSV4"), refused by name */
#define KVP_RETIRED_DSV4_MAGIC UINT32_C(0x34565344)

/* the frontier slot's size: the state model's frontier walk, aligned as a grid slot is (checkpoint.cpp) */
static uint64_t kvp_frontier_bytes(const pulsar_ckpt_store *st) {
    uint64_t bytes = 0;
    (void)st->ops->walk(st->state, -1, true, NULL, 0, st->ops->min_checkpoint(st->state), &bytes);
    return (bytes + 255u) & ~(uint64_t)255u;
}

/* every pool, then every trailing pool; false past the bound */
static bool kvp_pools(pulsar_ckpt_store *st, pulsar_kv_pool *out, uint32_t *n) {
    const uint32_t a = st->ops->pools(st->state, out, PULSAR_KV_POOLS_MAX);
    if (a > PULSAR_KV_POOLS_MAX) return false;
    const uint32_t b = st->ops->trailing_pools ? st->ops->trailing_pools(st->state, out + a, PULSAR_KV_POOLS_MAX - a) : 0u;
    if (b > PULSAR_KV_POOLS_MAX - a) return false;
    *n = a + b;
    return true;
}

/* the segment's layout digest, extended by every pool the payload carries (whole-row or open), the frontier
 * slot's size and the logits width */
static uint64_t kvp_layout_digest(pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t width) {
    uint64_t h = segment_layout_digest(st);
    auto mix = [&h](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; } };
    mix(n);
    for (uint32_t i = 0; i < n; i++) { mix(pools[i].tokens_per_row); mix(pools[i].row_bytes); mix(pools[i].whole_rows); }
    mix(kvp_frontier_bytes(st));
    mix(width);
    return h;
}

static uint64_t kvp_bytes_for(const pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t T,
                              uint32_t G, uint32_t n_img, uint32_t width) {
    uint64_t bytes = (uint64_t)KVP_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)T * sizeof(uint32_t);
    bytes += sizeof(uint32_t) + (uint64_t)n_img * SEGMENT_IMAGE_U32 * sizeof(uint32_t);
    bytes += (uint64_t)width * sizeof(float);
    bytes += (G ? st->slot_bytes : 0u) + kvp_frontier_bytes(st);   /* the resume checkpoint, the frontier */
    bytes += pools_span_bytes(pools, n, 0u, T);
    return bytes + sizeof(uint64_t);   /* the trailing digest */
}

/* What a save of the session writes, decided once: payload_bytes and the save read the same plan. */
struct kvp_plan {
    pulsar_ckpt_store *st;
    uint32_t bank, T, PF, G, width, n_pools;
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    uint64_t bytes;
};

static bool kvp_plan_save(pulsar_session *s, kvp_plan *p, char *err, size_t errlen) {
    p->st = pulsar_session_kv_store(s);
    p->bank = pulsar_session_live_bank(s);
    p->width = (uint32_t)pulsar_engine_logits_width(s->engine);
    if (!s->checkpoint_valid || s->logits_stale || s->checkpoint.len <= 0) {
        payload_set_err(err, errlen, "session has no live frontier to save (sync it first)");
        return false;
    }
    p->T = (uint32_t)s->checkpoint.len;
    char why[192];
    if (!p->st->ops->frontier_at(p->st->state, p->T, why, sizeof(why))) {
        char msg[256];
        snprintf(msg, sizeof(msg), "session payload: %s", why);
        payload_set_err(err, errlen, msg);
        return false;
    }
    p->PF = (uint32_t)pulsar_session_bank_prefill_frontier(s, p->bank);
    if (p->PF > p->T) {
        payload_set_err(err, errlen, "session payload: the prefill frontier lies past the session's tokens");
        return false;
    }
    if (!kvp_pools(p->st, p->pools, &p->n_pools)) {
        payload_set_err(err, errlen, "session payload: the model has more pools than a payload carries");
        return false;
    }
    /* the deepest checkpoint at or below the prefill frontier's grid point: what a sync that does not extend
     * the frontier restores */
    p->G = pulsar_ckpt_best(p->st, p->bank, pulsar_ckpt_grid_floor(p->st, p->PF));
    p->bytes = kvp_bytes_for(p->st, p->pools, p->n_pools, p->T, p->G, s->live_images.n, p->width);
    return true;
}

uint64_t pulsar_session::payload_bytes() {
    auto *s = this;
    kvp_plan p;
    char err[256];
    return kvp_plan_save(s, &p, err, sizeof(err)) ? p.bytes : 0u;
}

int pulsar_session::save_payload(FILE *fp, char *err, size_t errlen) {
    auto *s = this;
    if (!fp) { payload_set_err(err, errlen, "session payload: no stream"); return 1; }
    if (pulsar_session_is_mirrored(s)) {
        payload_set_err(err, errlen, "session payload: a tensor-parallel session's payload is not mirrored");
        return 1;
    }
    kvp_plan p;
    if (!kvp_plan_save(s, &p, err, errlen)) return 1;
    pulsar_ckpt_store *st = p.st;
    pulsar_gpu_tensor *slab = NULL;
    uint64_t slot_off = 0;
    if (p.G && !pulsar_ckpt_locate(st, p.bank, p.G, &slab, &slot_off)) {
        payload_set_err(err, errlen, "session payload: the resume checkpoint vanished while saving");
        return 1;
    }
    /* the frontier's slot, staged: the state model's frontier walk at T, over zeros (the slot's alignment tail
     * -- and DeepSeek's window head below its first raw_window positions -- is written too, so the same state
     * always saves the same bytes) */
    const uint64_t fb = kvp_frontier_bytes(st);
    pulsar_gpu_tensor *front = pulsar_gpu_tensor_alloc(fb);
    if (!front || pulsar_gpu_tensor_fill_f32(front, 0.0f, fb / sizeof(float)) == 0 ||
        !st->ops->walk(st->state, 0, true, front, 0, p.T, NULL) || pulsar_gpu_synchronize() == 0) {
        pulsar_gpu_tensor_free(front);
        payload_set_err(err, errlen, "session payload: staging the frontier's state failed");
        return 1;
    }
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, fp);
    const uint64_t layout = kvp_layout_digest(st, p.pools, p.n_pools, p.width);
    const uint32_t header[KVP_U32_FIELDS] = {
        PULSAR_SESSION_KV_PAYLOAD_MAGIC, PULSAR_SESSION_KV_PAYLOAD_VERSION, (uint32_t)layout, (uint32_t)(layout >> 32),
        p.T, p.PF, p.G, p.width,
    };
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < KVP_U32_FIELDS; i++) rc = payload_write_u32(&io, header[i], err, errlen);
    for (uint32_t i = 0; rc == 0 && i < p.T; i++) rc = payload_write_u32(&io, (uint32_t)s->checkpoint.v[i], err, errlen);
    if (rc == 0) rc = payload_write_images(&io, s->live_images.b, s->live_images.n, err, errlen);
    if (rc == 0) rc = payload_write_bytes(&io, s->logits, (uint64_t)p.width * sizeof(float), err, errlen);
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    if (rc == 0 && p.G)
        rc = payload_write_tensor_span(&io, slab, slot_off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0) rc = payload_write_tensor_span(&io, front, 0, fb, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0) rc = pools_span_io(&io, p.pools, p.n_pools, 0u, p.T, true, buf, NULL, err, errlen);
    free(buf);
    pulsar_gpu_tensor_free(front);
    if (rc != 0) return rc;
    const uint64_t dg = payload_digest_final(&io.digest);
    if (fwrite(&dg, 1, sizeof(dg), fp) != sizeof(dg)) {
        payload_set_err(err, errlen, "session payload: failed to write the digest");
        return 1;
    }
    pulsar_writeback_finish(&io.wb);
    return 0;
}

int pulsar_session::load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    auto *s = this;
    if (!fp) { payload_set_err(err, errlen, "session payload: no stream"); return 1; }
    if (pulsar_session_is_mirrored(s)) {
        payload_set_err(err, errlen, "session payload: a tensor-parallel session's payload is not mirrored");
        return 1;
    }
    spec_lookahead_reset(s);
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    const uint32_t width = (uint32_t)pulsar_engine_logits_width(s->engine);
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    uint32_t n = 0;
    if (!kvp_pools(st, pools, &n)) {
        payload_set_err(err, errlen, "session payload: the model has more pools than a payload carries");
        return 1;
    }
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, NULL);
    uint64_t remaining = payload_bytes;
    uint32_t h[KVP_U32_FIELDS];
    for (uint32_t i = 0; i < KVP_U32_FIELDS; i++)
        if (payload_read_u32(&io, &h[i], &remaining, err, errlen) != 0) return 1;
    if (h[0] == KVP_RETIRED_DSV4_MAGIC) {
        payload_set_err(err, errlen, "session payload: a DeepSeek graph-format (DSV4) payload, retired in L284 -- "
                                     "re-prefill");
        return 1;
    }
    if (h[0] != PULSAR_SESSION_KV_PAYLOAD_MAGIC || h[1] != PULSAR_SESSION_KV_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    const uint64_t layout = kvp_layout_digest(st, pools, n, width);
    if (h[2] != (uint32_t)layout || h[3] != (uint32_t)(layout >> 32) || h[7] != width) {
        payload_set_err(err, errlen, "session payload: written for a different state layout");
        return 1;
    }
    const uint32_t T = h[4], PF = h[5], G = h[6];
    if (T == 0u || T >= (uint32_t)s->ctx_size || PF > T || G > PF || G % st->ops->resume_grid != 0u ||
        (G && G < st->ops->min_checkpoint(st->state))) {
        payload_set_err(err, errlen, "session payload: its frontier, prefill frontier or grid point does not fit "
                                     "this session");
        return 1;
    }
    token_vec toks = {0};
    pulsar_image_identity imgs;
    imgs.n = 0;
    float *logits = (float *)xmalloc((size_t)width * sizeof(float));
    auto drop = [&]() { token_vec_free(&toks); free(logits); return 1; };
    for (uint32_t i = 0; i < T; i++) {
        uint32_t t = 0;
        if (payload_read_u32(&io, &t, &remaining, err, errlen) != 0) return drop();
        token_vec_push(&toks, (int)t);
    }
    uint32_t n_img = 0;
    if (payload_read_u32(&io, &n_img, &remaining, err, errlen) != 0) return drop();
    if (n_img > PULSAR_IMAGE_BLOCKS_MAX || payload_bytes != kvp_bytes_for(st, pools, n, T, G, n_img, width)) {
        payload_set_err(err, errlen, "session payload: size does not match its header");
        return drop();
    }
    for (uint32_t i = 0; i < n_img; i++) {
        uint32_t rec[SEGMENT_IMAGE_U32];
        for (uint32_t k = 0; k < SEGMENT_IMAGE_U32; k++)
            if (payload_read_u32(&io, &rec[k], &remaining, err, errlen) != 0) return drop();
        const pulsar_image_block b = image_rec_unpack(rec);
        if (b.end <= b.start || b.end > T || (i && b.start < imgs.b[i - 1].end)) {
            payload_set_err(err, errlen, "session payload: its image records are out of order or outside its tokens");
            return drop();
        }
        imgs.b[imgs.n++] = b;
    }
    if (payload_read_bytes(&io, logits, (uint64_t)width * sizeof(float), &remaining, err, errlen) != 0) return drop();
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "session payload: failed to synchronize accelerator");
        return drop();
    }
    /* Device writes begin: from here a failure leaves the bank holding nothing. */
    segment_clear(s, st, bank);
    pulsar_gpu_tensor *front = NULL;
    auto fail = [&]() { pulsar_gpu_tensor_free(front); segment_clear(s, st, bank); return drop(); };
    pulsar_gpu_tensor *slab = NULL;
    uint64_t slot_off = 0;
    uint32_t slot = PULSAR_CKPT_SLOTS_MAX;
    if (G && !pulsar_ckpt_claim(st, bank, G, &slab, &slot_off, &slot)) {
        payload_set_err(err, errlen, "session payload: no grid-checkpoint slot");
        return fail();
    }
    const uint64_t fb = kvp_frontier_bytes(st);
    front = pulsar_gpu_tensor_alloc(fb);
    if (!front) {
        payload_set_err(err, errlen, "session payload: staging the frontier's state failed");
        return fail();
    }
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    if (G) rc = payload_read_tensor_span(&io, slab, slot_off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, &remaining,
                                         err, errlen);
    if (rc == 0) rc = payload_read_tensor_span(&io, front, 0, fb, buf, PULSAR_SESSION_IO_CHUNK,
                                               &remaining, err, errlen);
    if (rc == 0) rc = pools_span_io(&io, pools, n, 0u, T, false, buf, &remaining, err, errlen);
    free(buf);
    if (rc != 0) return fail();
    uint64_t stored_dg = 0;
    if (remaining != sizeof(uint64_t) || fread(&stored_dg, 1, sizeof(stored_dg), fp) != sizeof(stored_dg) ||
        stored_dg != payload_digest_final(&io.digest)) {
        payload_set_err(err, errlen, "session payload: digest mismatch (corrupt payload)");
        return fail();
    }
    /* Commit: the frontier's counters, its lanes from the staged slot, then the resume checkpoint. */
    if (!st->ops->install_frontier(st->state, T, PF) || !st->ops->walk(st->state, 1, true, front, 0, T, NULL) ||
        pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "session payload: installing the frontier's state failed");
        return fail();
    }
    st->ops->restored(st->state);
    pulsar_gpu_tensor_free(front);
    if (G) pulsar_ckpt_commit(st, bank, slot, G);
    token_vec_free(&s->checkpoint);
    s->checkpoint = toks;
    s->live_images = imgs;
    memcpy(s->logits, logits, (size_t)width * sizeof(float));
    free(logits);
    s->prefill_frontier = (int)PF;   /* the session's mirror (DeepSeek's own: L195), as a fused prompt chunk sets it */
    s->checkpoint_valid = true;
    s->logits_stale = false;
    spec_lookahead_reset(s);
    return 0;
}



int pulsar_session::save_snapshot(pulsar_session_snapshot *snap, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    const uint64_t bytes = s->payload_bytes();
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX - 1u) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    /* ONE EXTRA BYTE, DELIBERATELY.  fmemopen's write mode appends a NUL at the
     * next position on fclose, and when the buffer is filled EXACTLY to capacity
     * that NUL lands on its last byte: the trailing payload byte is silently
     * zeroed, which for a compressor state lane is a corrupted float and for a
     * snapshot of a live session is a restore that is not the state it saved.
     * Budget the terminator and keep snap->len at the real payload size.  (dev
     * hit the same trap from the other side when the digest made a single
     * flipped byte observable -- it was the DIGEST's last byte that showed it,
     * which is why the trap predates it.) */
    const uint64_t capacity = bytes + 1u;
    if (snap->cap < capacity) {
        uint8_t *p = (uint8_t *)realloc(snap->ptr, (size_t)capacity);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = capacity;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)capacity, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = s->save_payload(fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}



int pulsar_session::load_snapshot(const pulsar_session_snapshot *snap, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = s->load_payload(fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}



void pulsar_session_snapshot_free(pulsar_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

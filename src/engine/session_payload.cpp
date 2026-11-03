/** @file session_payload.cpp
 *  @brief Session payloads, disk segments and snapshots: serialisation of a session's KV and state. */
#include "pulsar_engine_internal.h"

/* L281: one image block record in a payload or a segment -- u32 start, u32 end, u64 content as two u32, and
 * (L268) the block's 2D grid, u32 rows and u32 columns */
/** u32 words of one image block record (see above). */
#define SEGMENT_IMAGE_U32 6u
/** Pack an image block into its record.  @param b the block  @param rec receives the record */
static void image_rec_pack(const pulsar_image_block *b, uint32_t (&rec)[SEGMENT_IMAGE_U32]) {
    rec[0] = b->start;
    rec[1] = b->end;
    rec[2] = (uint32_t)b->content;
    rec[3] = (uint32_t)(b->content >> 32);
    rec[4] = b->grid_h;
    rec[5] = b->grid_w;
}
/** Unpack an image block record.  @param rec the record  @return the block */
static pulsar_image_block image_rec_unpack(const uint32_t (&rec)[SEGMENT_IMAGE_U32]) {
    return (pulsar_image_block){ rec[0], rec[1], (uint64_t)rec[2] | (uint64_t)rec[3] << 32, rec[4], rec[5] };
}
#include "lib/pulsar_kvdigest.h"
#include "lib/pulsar_writeback.h"
#include "lib/pulsar_stall.h"

void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}



/** Store `v` little-endian.  @param out receives 4 bytes  @param v the value */
static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}



/** Load a little-endian u32.  @param in 4 bytes  @return the value */
static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}



/* ---- payload digest ------------------------------------------------------
 *
 * L219: every payload (kv-state, segment, restore point) ends with the 64-bit digest of its other bytes, so a damaged
 * file is a cache miss (re-prefill), not wrong attention.  The digest is lib/pulsar_kvdigest.h's (v2, four lanes, L347
 * I3a), the one authority the segment store's writer and reader share: a kv-state payload digests as it streams
 * (payload_io below); a segment or restore point is copied out with its digest slot zeroed and sealed by the store's
 * writer, and checked by the store's reader when it comes off disk. */

/* One payload stream: the file plus the digest covering everything read or
 * written through it.  A private type so the file and its digest cannot drift
 * apart: only these helpers touch the stream, and every one of them folds the
 * bytes it moved. */
/** One payload stream: the file and the digest of every byte moved through it (see above). */
typedef struct {
    /// the payload file; borrowed (the caller opened it and closes it)
    FILE *fp;
    /// digest of every byte read or written through this stream so far
    pulsar_kvdigest digest;
    pulsar_writeback wb;   ///< write side only: written pages go to disk and leave the cache as they stream (L261)
} payload_io;

/** Write bytes through the stream, folding them into its digest, then advance its writeback.
 *  @param io     the stream
 *  @param ptr    the bytes
 *  @param bytes  how many
 *  @param err    receives the failure
 *  @param errlen bytes of `err`
 *  @return 0, or 1 on a short write */
static int payload_write_bytes(payload_io *io, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, io->fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        pulsar_kvdigest_update(&io->digest, p, n);
        p += n;
        bytes -= n;
    }
    pulsar_writeback_step(&io->wb);
    return 0;
}



/** Read bytes through the stream, folding them into its digest.
 *  @param io        the stream
 *  @param ptr       receives the bytes
 *  @param bytes     how many
 *  @param remaining the payload bytes left, decremented (NULL = unbounded)
 *  @param err       receives the failure
 *  @param errlen    bytes of `err`
 *  @return 0, or 1 when truncated or on a short read */
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
        pulsar_kvdigest_update(&io->digest, p, n);
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



/** Write a little-endian u32 through the stream.  @param io the stream  @param v the value
 *  @param err receives the failure  @param errlen bytes of `err`  @return 0, or 1 on failure */
static int payload_write_u32(payload_io *io, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(io, b, sizeof(b), err, errlen);
}



/** Read a little-endian u32 through the stream.  @param io the stream  @param v receives the value
 *  @param remaining payload bytes left (NULL = unbounded)  @param err receives the failure
 *  @param errlen bytes of `err`  @return 0, or 1 on failure */
static int payload_read_u32(payload_io *io, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (payload_read_bytes(io, b, sizeof(b), remaining, err, errlen) != 0) return 1;
    *v = payload_get_u32(b);
    return 0;
}

/* L281: an image section -- the count, then each block's record (payload, segment and kv-state payload) */
/// @param io     the stream
/// @param b      [n] the blocks
/// @param n      blocks
/// @param err    receives the failure
/// @param errlen bytes of `err`
/// @return 0, or 1 on failure
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
/// @param io     the stream
/// @param tensor the device tensor read from
/// @param offset the span's byte offset in it
/// @param bytes  the span's length in bytes
/// @param buf    the staging buffer
/// @param cap    bytes of `buf`
/// @param err    receives the failure
/// @param errlen bytes of `err`
/// @return 0, or 1 when the span is out of range or a copy failed
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



/** Read a span from the stream into a device tensor through a host staging buffer.
 *  @param io        the stream
 *  @param tensor    the device tensor written
 *  @param offset    the span's byte offset in it
 *  @param bytes     the span's length in bytes
 *  @param buf       the staging buffer
 *  @param cap       bytes of `buf`
 *  @param remaining payload bytes left (NULL = unbounded)
 *  @param err       receives the failure
 *  @param errlen    bytes of `err`
 *  @return 0, or 1 when the span is out of range or a copy failed */
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
 * One segment = tokens [G_prev, G) and every append-only POOL's rows for those
 * positions (pulsar.h, pulsar_session_copy_out_segment).  What the pools and the slot
 * are is the model's (kv_state.h); this framing is not.  The header pins the
 * model's layout by a digest of what makes the bytes readable -- the state
 * model's name, its grid, the slot size and every pool's granularity and row
 * width -- so a segment written under another layout is refused, never misread.
 *
 * L347 (v7): a segment is ROWS ONLY.  The stream is the header (word 8: flags,
 * none defined, 0), the tokens, the image records, the pool rows and the rows'
 * digest.  The state at G -- the grid checkpoint's slot -- is a RESTORE POINT, a
 * payload of its own (pulsar_session_copy_out_restore_point, below) stored beside the
 * segment under the same key: a chain load reads the rows of every segment and
 * one restore point, its last segment's. */
/** u32 words of a segment's header. */
#define SEGMENT_U32_FIELDS 9u
/* L281 (v3): after the span's tokens, the image blocks lying wholly inside the span -- a u32 count, then per block
 * u32 start, u32 end, u64 content (two u32) -- so a chain carries rows past an image together with WHICH image they
 * are (image_identity.cpp).  A span boundary is never inside a block (segment_cut_ok; a checkpoint never is), so
 * every block a chain holds is in exactly one segment.  Each rank's copy carries them: a worker's load sets the same
 * identity as the leader's. */

/* the session's records of the blocks inside [G_prev, G): one run of the records (pulsar_image_identity_span), or
 * refused when the records are out of order -- the save writes the run from *first as it stands.  Said once a
 * process (the chain asks the size of every span it persists; a refused size is its "no cut") */
/// @param s      the session (its live image identity)
/// @param G_prev the span's start
/// @param G      the span's end
/// @param first  receives the first such record (borrowed), NULL when none
/// @param n      receives how many records lie wholly inside the span
/// @return false (said once) when the records are not in position order
static bool segment_images(const pulsar_session *s, uint32_t G_prev, uint32_t G, const pulsar_image_block **first,
                           uint32_t *n) {
    uint32_t at = 0;
    *first = NULL;
    if (!pulsar_image_identity_span(&s->live_images, G_prev, G, &at, n)) {
        *n = 0;
        if (pulsar_refusal_first("segment image records"))
            fprintf(stderr, "pulsar: segment [%u, %u): the session's %u image records are not in position order -- "
                            "refusing the segment\n", G_prev, G, s->live_images.n);
        return false;
    }
    if (*n) *first = &s->live_images.b[at];
    return true;
}

/* The bank holds nothing: a chain's root replaces its history, a failed load leaves it empty.  The
 * session's invalidate forgets the view (a family whose invalidate leaves the device state alone has its
 * lanes written by the chain's last restore and its counters by set_frontier_stale), and the bank's
 * checkpoints go with the history they described. */
/// @param s    the session
/// @param st   its checkpoint store
/// @param bank the installed bank
static void segment_clear(pulsar_session *s, pulsar_ckpt_store *st, uint32_t bank) {
    s->live_images.n = 0;   /* L281: the bank holds no block */
    pulsar_session_family_invalidate(s);
    pulsar_ckpt_drop_bank(st, bank);
}

/** The installed bank's pools.  @param st the store  @param pools receives up to PULSAR_KV_POOLS_MAX pools
 *  @return the count, 0 when the model has more than PULSAR_KV_POOLS_MAX */
static uint32_t segment_pools(pulsar_ckpt_store *st, pulsar_kv_pool *pools) {
    const uint32_t n = st->ops->pools(st->state, pools, PULSAR_KV_POOLS_MAX);
    return n <= PULSAR_KV_POOLS_MAX ? n : 0u;   /* a model past the bound refuses every segment */
}

/** FNV-1a over what makes a segment's bytes readable: the state model's name, the artifact (when set), the bank's
 *  runtime adapter (L338: only when on -- a plain bank's digest is as it was), the grid, the slot size and every
 *  pool's tokens per row and row bytes.
 *  @param st   the store
 *  @param bank the bank the segment is written from / read into
 *  @return the digest */
static uint64_t segment_layout_digest(pulsar_ckpt_store *st, uint32_t bank) {
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; } };
    for (const char *c = st->ops->name; *c; c++) mix((uint8_t)*c);
    if (st->artifact) mix(st->artifact);   /* 0 leaves DeepSeek's digests as they were */
    if (st->bank_adapter && st->bank_adapter[bank]) for (const char *c = "runtime-adapter"; *c; c++) mix((uint8_t)*c);
    mix(st->ops->resume_grid);
    mix(st->slot_bytes);
    mix(n);
    for (uint32_t i = 0; i < n; i++) { mix(pools[i].tokens_per_row); mix(pools[i].row_bytes); }
    return h;
}

/* Pool rows [t0 / tokens_per_row, end(t1)) of every pool: a segment's span (grid points, whole rows) or a
 * kv-state payload's [0, frontier) -- its open row too, unless the pool writes whole rows only (kv_state.h).
 * The one sizing and the one copy of pool rows either carries. */
/** The row a span ending at t ends on: floor for a whole-rows pool, else ceil (the open row carried).
 *  @param p the pool  @param t the position, in tokens  @return the row index (exclusive end) */
static uint32_t pool_row(const pulsar_kv_pool &p, uint32_t t) {
    return (t + (p.whole_rows ? 0u : p.tokens_per_row - 1u)) / p.tokens_per_row;
}

/// @param pools [n] the pools
/// @param n     pools
/// @param t0    the span's start, in tokens
/// @param t1    the span's end, in tokens
/// @return the bytes of every pool's rows in the span
static uint64_t pools_span_bytes(const pulsar_kv_pool *pools, uint32_t n, uint32_t t0, uint32_t t1) {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < n; i++) bytes += (uint64_t)(pool_row(pools[i], t1) - t0 / pools[i].tokens_per_row) * pools[i].row_bytes;
    return bytes;
}

/** Write (or read) every pool's rows of the span [t0, t1) through the stream.
 *  @param io        the stream
 *  @param pools     [n] the pools
 *  @param n         pools
 *  @param t0        the span's start, in tokens
 *  @param t1        the span's end, in tokens
 *  @param write     write (else read into the pools)
 *  @param buf       a PULSAR_SESSION_IO_CHUNK staging buffer
 *  @param remaining payload bytes left on a read (NULL = unbounded)
 *  @param err       receives the failure
 *  @param errlen    bytes of `err`
 *  @return 0, or 1 on failure (an evicted pool included) */
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

/** Is [G_prev, G) a segment span: non-empty, both ends on the resume grid?
 *  @param st the store (its grid)  @param G_prev the start  @param G the end  @return true when it is */
static bool segment_span_ok(const pulsar_ckpt_store *st, uint32_t G_prev, uint32_t G) {
    const uint32_t grid = st->ops->resume_grid;
    return G > G_prev && G_prev % grid == 0u && G % grid == 0u;
}

/* L323: a segment may END at G only where no image block of the history lies across it (a checkpoint never does;
 * a rows-only grid cut must check) -- every block a chain holds is then inside exactly one segment */
/// @param s the session (its live image identity)
/// @param G the cut, in tokens
/// @return false when an image block straddles G
static bool segment_cut_ok(const pulsar_session *s, uint32_t G) {
    for (uint32_t i = 0; i < s->live_images.n; i++)
        if (s->live_images.b[i].start < G && G < s->live_images.b[i].end) return false;
    return true;
}

/* the bytes of a segment over [G_prev, G) carrying `n_images` image records.  Only the current version loads: older
 * segments are refused and the chain rewrites itself (on-disk KV carries no back-compat, Tyler 2026-10-13). */
/// @param st       the store
/// @param G_prev   the span's start
/// @param G        the span's end
/// @param n_images image records carried
/// @return the segment's size in bytes
static uint64_t segment_bytes_for(pulsar_ckpt_store *st, uint32_t G_prev, uint32_t G, uint32_t n_images) {
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    uint64_t bytes = (uint64_t)SEGMENT_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)(G - G_prev) * sizeof(uint32_t);
    bytes += sizeof(uint32_t) + (uint64_t)n_images * SEGMENT_IMAGE_U32 * sizeof(uint32_t);
    bytes += pools_span_bytes(pools, n, G_prev, G);
    return bytes + sizeof(uint64_t);   /* the rows' digest */
}

/* ===== L347: restore points ===============================================
 * The state at a grid point G: the installed bank's checkpoint slot at G, the
 * one thing a chain's load restores besides rows (pulsar.h,
 * pulsar_session_copy_out_restore_point).  The stream is the header -- magic
 * "SRP1", version, the segment layout digest (lo, hi), the slot size, the
 * resume grid, G, flags (none defined, 0) -- the slot, and the digest of both. */
/** u32 words of a restore point's header. */
#define RESTORE_POINT_U32_FIELDS 8u

/** The bytes of a restore point: header, slot, digest.  @param st the store (its slot size)  @return bytes */
static uint64_t restore_point_bytes_for(const pulsar_ckpt_store *st) {
    return (uint64_t)RESTORE_POINT_U32_FIELDS * sizeof(uint32_t) + st->slot_bytes + sizeof(uint64_t);
}

/* L281: a stored segment's image records, read from its payload's head without loading it.  The bytes are the
 * payload digest's to vouch for; this only decides how far a restore goes, and the load that follows re-reads and
 * checks everything. */
int pulsar_session_segment_images(FILE *fp, uint64_t bytes, uint64_t *hashes, uint32_t cap, uint32_t *n_out) {
    *n_out = 0;
    uint32_t h[SEGMENT_U32_FIELDS];
    if (!fp || bytes < sizeof(h) || fread(h, sizeof(uint32_t), SEGMENT_U32_FIELDS, fp) != SEGMENT_U32_FIELDS) return 1;
    if (h[0] != PULSAR_SESSION_SEGMENT_MAGIC || h[7] <= h[6]) return 1;
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

uint64_t pulsar_session::segment_bytes(uint32_t G_prev, uint32_t G, uint64_t *state_bytes) {
    pulsar_ckpt_store *st = pulsar_session_kv_store(this);
    if (state_bytes) *state_bytes = 0u;
    if (!segment_span_ok(st, G_prev, G) || !segment_cut_ok(this, G_prev) || !segment_cut_ok(this, G)) return 0;
    if (state_bytes) *state_bytes = restore_point_bytes(G);
    const pulsar_image_block *first = NULL;
    uint32_t n_img = 0;
    if (!segment_images(this, G_prev, G, &first, &n_img)) return 0;
    return segment_bytes_for(st, G_prev, G, n_img);
}

uint64_t pulsar_session::restore_point_bytes(uint32_t G) {
    pulsar_ckpt_store *st = pulsar_session_kv_store(this);
    if (G == 0u || G % st->ops->resume_grid != 0u || !pulsar_ckpt_holds(st, pulsar_session_live_bank(this), G))
        return 0;
    return restore_point_bytes_for(st);
}

/* ===== L347 I3a: segments and restore points through memory ================
 * A save COPIES OUT: the header, tokens and image records are written by the host straight into the caller's buffer
 * (the segment store's pinned image of the file), the device bytes -- every pool's rows of the span, or the checkpoint
 * slot -- by one device -> host copy per span on the copy stream, fenced after the work queued on this thread's stream
 * (pulsar_gpu_copy_out: no device drain), and the digest slot (the last 8 bytes) is left zero: the store's writer
 * seals it (pulsar_kvdigest_seal over [0, bytes - 8)).  A load reads the same bytes from memory -- a pinned image, so
 * each pool span is one host -> device DMA -- and checks no digest: the store checked it when it read the file, and a
 * RAM-resident image never left this process.  The byte layout is the file's. */

/** A cursor writing little-endian u32s and raw bytes into a caller's buffer, bounded by its size. */
struct mem_out {
    uint8_t *p;      ///< the buffer
    uint64_t cap;    ///< its bytes
    uint64_t at;     ///< bytes written so far
    bool ok;         ///< false once a write would overrun
    /** Write a u32.  @param v the value */
    void u32(uint32_t v) {
        if (!ok || cap - at < 4u) { ok = false; return; }
        payload_put_u32(p + at, v);
        at += 4u;
    }
    /** Reserve `n` bytes for a device copy.  @param n bytes  @return where they go, NULL on overrun */
    uint8_t *take(uint64_t n) {
        if (!ok || cap - at < n) { ok = false; return NULL; }
        uint8_t *q = p + at;
        at += n;
        return q;
    }
};

/** A cursor reading little-endian u32s and raw bytes from a buffer, bounded by its size. */
struct mem_in {
    const uint8_t *p;   ///< the buffer
    uint64_t len;       ///< its bytes
    uint64_t at;        ///< bytes read so far
    /** Read a u32.  @param v receives it  @return false past the end */
    bool u32(uint32_t *v) {
        if (len - at < 4u) return false;
        *v = payload_get_u32(p + at);
        at += 4u;
        return true;
    }
    /** Take `n` bytes.  @param n bytes  @return them, NULL past the end */
    const uint8_t *take(uint64_t n) {
        if (len - at < n) return NULL;
        const uint8_t *q = p + at;
        at += n;
        return q;
    }
};

int pulsar_session::copy_out_restore_point(uint32_t G, uint8_t *dst, uint64_t bytes, char *err, size_t errlen) {
    pulsar_ckpt_store *st = pulsar_session_kv_store(this);
    const uint32_t bank = pulsar_session_live_bank(this);
    pulsar_ckpt_slot_at at;
    if (!dst || G == 0u || G % st->ops->resume_grid != 0u || !pulsar_ckpt_locate(st, bank, G, &at)) {
        payload_set_err(err, errlen, "copy-out restore point: the installed bank holds no checkpoint there");
        return 1;
    }
    if (bytes != restore_point_bytes_for(st)) {
        payload_set_err(err, errlen, "copy-out restore point: the buffer is not the restore point's size");
        return 1;
    }
    mem_out o = { dst, bytes, 0, true };
    const uint64_t layout = segment_layout_digest(st, bank);
    const uint32_t header[RESTORE_POINT_U32_FIELDS] = {
        PULSAR_SESSION_RESTORE_POINT_MAGIC, PULSAR_SESSION_RESTORE_POINT_VERSION,
        (uint32_t)layout, (uint32_t)(layout >> 32), (uint32_t)st->slot_bytes, st->ops->resume_grid, G, 0u,
    };
    for (uint32_t i = 0; i < RESTORE_POINT_U32_FIELDS; i++) o.u32(header[i]);
    uint8_t *slot_dst = o.take(st->slot_bytes);
    uint8_t *dg = o.take(PULSAR_KVDIGEST_BYTES);
    if (!o.ok || o.at != bytes) {
        payload_set_err(err, errlen, "copy-out restore point: its layout overran the buffer");
        return 1;
    }
    memset(dg, 0, PULSAR_KVDIGEST_BYTES);   /* the writer's to seal */
    /* L347 I4: on a discrete card the slot already IS host memory (its capture landed: pulsar_ckpt_locate waited for
     * that one copy), so this is a memcpy; on GB10 the device copy-out */
    pulsar_stall_mark("restore point copy-out: state slot to host");
    if (!pulsar_ckpt_slot_get(st, &at, slot_dst)) {
        payload_set_err(err, errlen, "copy-out restore point: the slot copy failed");
        return 1;
    }
    return 0;
}

int pulsar_session::copy_out_segment(uint32_t G_prev, uint32_t G, uint8_t *dst, uint64_t bytes, char *err,
                                     size_t errlen) {
    auto *s = this;
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    if (!dst || !s->checkpoint_valid || !segment_span_ok(st, G_prev, G) || G > (uint32_t)s->checkpoint.len) {
        payload_set_err(err, errlen, "copy-out segment: the session does not hold that grid span");
        return 1;
    }
    if (!st->ops->holds(st->state, G)) {
        payload_set_err(err, errlen, "copy-out segment: the bank's rows do not reach the segment's end");
        return 1;
    }
    if (!segment_cut_ok(s, G_prev) || !segment_cut_ok(s, G)) {
        payload_set_err(err, errlen, "copy-out segment: an image block lies across the span's edge");
        return 1;
    }
    const pulsar_image_block *b = NULL;
    uint32_t n_img = 0;
    if (!segment_images(s, G_prev, G, &b, &n_img)) {
        payload_set_err(err, errlen, "copy-out segment: the image records are not in position order");
        return 1;
    }
    if (bytes != segment_bytes_for(st, G_prev, G, n_img)) {
        payload_set_err(err, errlen, "copy-out segment: the buffer is not the segment's size");
        return 1;
    }
    mem_out o = { dst, bytes, 0, true };
    const uint64_t layout = segment_layout_digest(st, bank);
    const uint32_t header[SEGMENT_U32_FIELDS] = {
        PULSAR_SESSION_SEGMENT_MAGIC, PULSAR_SESSION_SEGMENT_VERSION,
        (uint32_t)layout, (uint32_t)(layout >> 32), (uint32_t)st->slot_bytes, st->ops->resume_grid, G_prev, G,
        0u,
    };
    for (uint32_t i = 0; i < SEGMENT_U32_FIELDS; i++) o.u32(header[i]);
    for (uint32_t p = G_prev; p < G; p++) o.u32((uint32_t)s->checkpoint.v[p]);
    o.u32(n_img);
    for (uint32_t i = 0; i < n_img; i++) {   /* L281: the span's image records */
        uint32_t rec[SEGMENT_IMAGE_U32];
        image_rec_pack(&b[i], rec);
        for (uint32_t k = 0; k < SEGMENT_IMAGE_U32; k++) o.u32(rec[k]);
    }
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    pulsar_gpu_copy_span spans[PULSAR_KV_POOLS_MAX];
    uint32_t n_spans = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t row0 = G_prev / pools[i].tokens_per_row, rows = pool_row(pools[i], G) - row0;
        if (!rows) continue;
        if (!pools[i].rows) {   /* L284 #3: an evicted bank's pools are gone until alloc_physical re-backs them */
            payload_set_err(err, errlen, "copy-out segment: a KV pool of the installed bank has no physical (an "
                                         "evicted bank)");
            return 1;
        }
        const uint64_t span = (uint64_t)rows * pools[i].row_bytes;
        spans[n_spans++] = { pools[i].rows, (uint64_t)row0 * pools[i].row_bytes, o.take(span), span };
    }
    uint8_t *dg = o.take(PULSAR_KVDIGEST_BYTES);
    if (!o.ok || o.at != bytes) {
        payload_set_err(err, errlen, "copy-out segment: its layout overran the buffer");
        return 1;
    }
    memset(dg, 0, PULSAR_KVDIGEST_BYTES);   /* the writer's to seal */
    pulsar_stall_mark("segment copy-out: pool rows to host");
    if (pulsar_gpu_copy_out(spans, n_spans) == 0) {
        payload_set_err(err, errlen, "copy-out segment: the device copy failed");
        return 1;
    }
    return 0;
}

/* L347: read and check a restore point's header for the segment ending at G (the slot follows on `in`). */
/// @param in        the restore point, at its first byte
/// @param st        the store
/// @param bank      the installed bank
/// @param G         the chain's end: the restore point must be the state there
/// @param err       receives the refusal
/// @param errlen    bytes of `err`
/// @return 0, or 1 on a short read or a header that is not this layout's restore point at G
static int restore_point_header_read(mem_in *in, pulsar_ckpt_store *st, uint32_t bank, uint32_t G, char *err,
                                     size_t errlen) {
    uint32_t h[RESTORE_POINT_U32_FIELDS];
    for (uint32_t i = 0; i < RESTORE_POINT_U32_FIELDS; i++)
        if (!in->u32(&h[i])) {
            payload_set_err(err, errlen, "load segment: truncated restore point");
            return 1;
        }
    if (h[0] != PULSAR_SESSION_RESTORE_POINT_MAGIC || h[1] != PULSAR_SESSION_RESTORE_POINT_VERSION) {
        payload_set_err(err, errlen, "load segment: unsupported restore point version");
        return 1;
    }
    const uint64_t layout = segment_layout_digest(st, bank);
    if (h[2] != (uint32_t)layout || h[3] != (uint32_t)(layout >> 32) || h[4] != (uint32_t)st->slot_bytes ||
        h[5] != st->ops->resume_grid) {
        payload_set_err(err, errlen, "load segment: restore point written for a different state layout");
        return 1;
    }
    if (h[6] != G) {
        payload_set_err(err, errlen, "load segment: the restore point is not the state at the segment's end");
        return 1;
    }
    if (h[7] != 0u) {
        payload_set_err(err, errlen, "load segment: unknown restore point flags");
        return 1;
    }
    return 0;
}

/** L347 I3b: a segment (and its restore point) parsed and checked against the installed bank's layout, nothing
 *  written (segment_parse). */
struct segment_view {
    uint32_t G_prev = 0;          ///< the span's start
    uint32_t G = 0;               ///< ... and its end
    const uint8_t *toks = NULL;   ///< its G - G_prev token ids
    uint32_t n_img = 0;           ///< its image records
    pulsar_image_block imgs[PULSAR_IMAGE_BLOCKS_MAX];   ///< ... unpacked
    mem_in in{};                  ///< the segment, positioned at its pool rows
    mem_in rin{};                 ///< the restore point, positioned at its slot (when `last`)
};

/* L347 I3b: every check of a segment's bytes and its restore point's against the layout of `bank` -- what a load
 * checks before its first device write, and nothing of the bank's state (where it stands, its images, its slots).  A
 * fault here is the bytes' (pulsar_segment_status). */
/// @param st       the store
/// @param bank     the installed bank
/// @param ctx_size the session's context
/// @param seg      the segment's bytes
/// @param bytes    their size
/// @param last     the chain's last segment (its restore point is checked)
/// @param rp       the restore point when `last`, else NULL
/// @param rp_bytes its size
/// @param v        receives the parse
/// @param err      receives the fault
/// @param errlen   bytes of `err`
/// @return PULSAR_SEGMENT_OK, _BAD_ROWS, _BAD_RP, or _TRANSIENT for a caller error (no bytes, `rp` against `last`)
static pulsar_segment_status segment_parse(pulsar_ckpt_store *st, uint32_t bank, int ctx_size, const uint8_t *seg,
                                           uint64_t bytes, bool last, const uint8_t *rp, uint64_t rp_bytes,
                                           segment_view *v, char *err, size_t errlen) {
    if (!seg) { payload_set_err(err, errlen, "load segment: no bytes"); return PULSAR_SEGMENT_TRANSIENT; }
    if (last != (rp != NULL)) {
        payload_set_err(err, errlen, last ? "load segment: the chain's last segment has no restore point to restore"
                                          : "load segment: a restore point given for a segment that is not the last");
        return PULSAR_SEGMENT_TRANSIENT;
    }
    v->in = { seg, bytes, 0 };
    uint32_t h[SEGMENT_U32_FIELDS];
    for (uint32_t i = 0; i < SEGMENT_U32_FIELDS; i++)
        if (!v->in.u32(&h[i])) { payload_set_err(err, errlen, "load segment: truncated segment"); return PULSAR_SEGMENT_BAD_ROWS; }
    if (h[0] != PULSAR_SESSION_SEGMENT_MAGIC || h[1] != PULSAR_SESSION_SEGMENT_VERSION) {
        payload_set_err(err, errlen, "load segment: unsupported segment version");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    const uint64_t layout = segment_layout_digest(st, bank);
    if (h[2] != (uint32_t)layout || h[3] != (uint32_t)(layout >> 32) || h[4] != (uint32_t)st->slot_bytes ||
        h[5] != st->ops->resume_grid) {
        payload_set_err(err, errlen, "load segment: written for a different state layout");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    v->G_prev = h[6];
    v->G = h[7];
    if (!segment_span_ok(st, v->G_prev, v->G) || v->G >= (uint32_t)ctx_size) {
        payload_set_err(err, errlen, "load segment: span is not a grid span inside this context");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    if (h[8] != 0u) {
        payload_set_err(err, errlen, "load segment: unknown header flags");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    v->toks = v->in.take((uint64_t)(v->G - v->G_prev) * sizeof(uint32_t));
    if (!v->toks || !v->in.u32(&v->n_img)) {
        payload_set_err(err, errlen, "load segment: truncated segment");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    if (v->n_img > PULSAR_IMAGE_BLOCKS_MAX) {
        payload_set_err(err, errlen, "load segment: more image blocks than a session holds");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    for (uint32_t i = 0; i < v->n_img; i++) {   /* L281: the span's image records */
        uint32_t rec[SEGMENT_IMAGE_U32];
        for (uint32_t k = 0; k < SEGMENT_IMAGE_U32; k++)
            if (!v->in.u32(&rec[k])) {
                payload_set_err(err, errlen, "load segment: truncated segment");
                return PULSAR_SEGMENT_BAD_ROWS;
            }
        v->imgs[i] = image_rec_unpack(rec);
        if (v->imgs[i].start < v->G_prev || v->imgs[i].end > v->G || v->imgs[i].end <= v->imgs[i].start) {
            payload_set_err(err, errlen, "load segment: an image record lies outside its span");
            return PULSAR_SEGMENT_BAD_ROWS;
        }
    }
    if (bytes != segment_bytes_for(st, v->G_prev, v->G, v->n_img)) {
        payload_set_err(err, errlen, "load segment: size does not match its span");
        return PULSAR_SEGMENT_BAD_ROWS;
    }
    /* L347: the chain's last segment restores the state at G from its restore point */
    v->rin = { rp, rp_bytes, 0 };
    if (last) {
        if (rp_bytes != restore_point_bytes_for(st)) {
            payload_set_err(err, errlen, "load segment: the restore point's size does not match the state layout");
            return PULSAR_SEGMENT_BAD_RP;
        }
        if (restore_point_header_read(&v->rin, st, bank, v->G, err, errlen) != 0) return PULSAR_SEGMENT_BAD_RP;
    }
    return PULSAR_SEGMENT_OK;
}

pulsar_segment_status pulsar_session::check_segment(const uint8_t *seg, uint64_t bytes, bool last, const uint8_t *rp,
                                                    uint64_t rp_bytes, char *err, size_t errlen) {
    segment_view v;
    return segment_parse(pulsar_session_kv_store(this), pulsar_session_live_bank(this), ctx_size, seg, bytes, last, rp,
                         rp_bytes, &v, err, errlen);
}

pulsar_segment_status pulsar_session::load_segment(const uint8_t *seg, uint64_t bytes, bool last, const uint8_t *rp,
                                                   uint64_t rp_bytes, uint32_t *G_out, char *err, size_t errlen) {
    auto *s = this;
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const uint32_t bank = pulsar_session_live_bank(s);
    if (G_out) *G_out = 0u;
    /* L347 I3b: every check of the bytes first -- a content fault leaves the bank as it was */
    segment_view v;
    const pulsar_segment_status parsed = segment_parse(st, bank, s->ctx_size, seg, bytes, last, rp, rp_bytes, &v, err,
                                                       errlen);
    if (parsed != PULSAR_SEGMENT_OK) return parsed;
    const uint32_t G_prev = v.G_prev, G = v.G;
    mem_in &in = v.in;
    mem_in &rin = v.rin;
    /* A chain loads in order: a root resets the bank, every later segment continues one that stands exactly at its
     * start (checked before anything is written).  From here every refusal is the bank's or the device's: transient. */
    if (G_prev != 0u) {
        char why[192];
        if (!s->checkpoint_valid || (uint32_t)s->checkpoint.len != G_prev ||
            !st->ops->stands_at(st->state, G_prev, why, sizeof(why))) {
            payload_set_err(err, errlen, "load segment: the bank does not stand at the segment's start");
            return PULSAR_SEGMENT_TRANSIENT;
        }
    }
    if (v.n_img > PULSAR_IMAGE_BLOCKS_MAX - (G_prev ? s->live_images.n : 0u)) {
        payload_set_err(err, errlen, "load segment: more image blocks than the session holds room for");
        return PULSAR_SEGMENT_TRANSIENT;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "load segment: failed to synchronize accelerator");
        return PULSAR_SEGMENT_TRANSIENT;
    }
    /* Device writes begin: from here a failure leaves the bank holding nothing. */
    if (G_prev == 0u) segment_clear(s, st, bank);
    /* L347 I3b: a failure here is the device's or the bank's (transient); the two layout checks below are the bytes'
     * (segment_parse already sized them, so they hold unless the pools disagree with the size rule) */
    auto fail = [&](pulsar_segment_status why) { segment_clear(s, st, bank); return why; };
    pulsar_ckpt_slot_at at = {};
    if (last && !pulsar_ckpt_claim(st, bank, G, &at)) {
        payload_set_err(err, errlen, "load segment: no grid-checkpoint slot");
        return fail(PULSAR_SEGMENT_TRANSIENT);
    }
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t n = segment_pools(st, pools);
    pulsar_stall_mark("segment load: host to pool rows");
    for (uint32_t i = 0; i < n; i++) {   /* one host -> device copy per pool span */
        const uint32_t row0 = G_prev / pools[i].tokens_per_row, rows = pool_row(pools[i], G) - row0;
        if (!rows) continue;
        if (!pools[i].rows) {
            payload_set_err(err, errlen, "load segment: a KV pool of the installed bank has no physical (an evicted "
                                         "bank)");
            return fail(PULSAR_SEGMENT_TRANSIENT);
        }
        const uint64_t span = (uint64_t)rows * pools[i].row_bytes;
        const uint8_t *src = in.take(span);
        if (!src || pulsar_gpu_tensor_write(pools[i].rows, (uint64_t)row0 * pools[i].row_bytes, src, span) == 0) {
            payload_set_err(err, errlen, "load segment: restoring a pool's rows failed");
            return fail(src ? PULSAR_SEGMENT_TRANSIENT : PULSAR_SEGMENT_BAD_ROWS);
        }
    }
    if (bytes - in.at != PULSAR_KVDIGEST_BYTES) {
        payload_set_err(err, errlen, "load segment: its rows do not end at its digest");
        return fail(PULSAR_SEGMENT_BAD_ROWS);
    }
    /* L347: the chain's last segment -- the state at G from its restore point */
    if (last) {
        pulsar_stall_mark("segment load: restore point to the checkpoint slot");
        const uint8_t *slot_src = rin.take(st->slot_bytes);
        if (!slot_src || rp_bytes - rin.at != PULSAR_KVDIGEST_BYTES) {
            payload_set_err(err, errlen, "load segment: the restore point's slot does not end at its digest");
            return fail(PULSAR_SEGMENT_BAD_RP);
        }
        if (!pulsar_ckpt_slot_put(st, &at, slot_src)) {
            payload_set_err(err, errlen, "load segment: restoring the restore point's slot failed");
            return fail(PULSAR_SEGMENT_TRANSIENT);
        }
    }
    /* Commit: the history, the frontier, then the restore point's checkpoint (after the frontier write, which could
     * otherwise drop it).  The lanes and window describe no position yet -- stale until the last segment restores
     * its checkpoint. */
    for (uint32_t p = 0; p < G - G_prev; p++) token_vec_push(&s->checkpoint, (int)payload_get_u32(v.toks + 4u * p));
    for (uint32_t i = 0; i < v.n_img; i++) s->live_images.b[s->live_images.n++] = v.imgs[i];   /* L281 */
    st->ops->set_frontier_stale(st->state, G);
    if (last) pulsar_ckpt_commit(st, bank, at.slot, G);
    s->prefill_frontier = (int)G;   /* L195: a prefill reached G (the family's own counter is set_frontier_stale's) */
    s->checkpoint_valid = true;
    /* the chain's last checkpoint: the family's standalone restore where it has one (DeepSeek: the history and the
     * drafter window with it), else the store's alone (Qwen: its sync owns the rest) */
    const pulsar_family_bank_ops *bops = s->engine->family->banks;
    if (last && !(bops->restore_checkpoint ? bops->restore_checkpoint(s, G) : pulsar_ckpt_restore(st, bank, G))) {
        payload_set_err(err, errlen, "load segment: restoring the chain's last checkpoint failed");
        return fail(PULSAR_SEGMENT_TRANSIENT);
    }
    if (last) s->logits_stale = true;   /* a restore moves the KV, not the logits (restore_checkpoint sets it too) */
    if (G_out) *G_out = G;
    return PULSAR_SEGMENT_OK;
}



/* ===== L284: the kv-state payload =========================================
 * Every model's session payload is a segment's parts at the session's frontier T (pulsar.h,
 * PULSAR_SESSION_KV_PAYLOAD_MAGIC): the same layout digest, image section, slot walk, pool copy and digested
 * stream, so the bytes have one authority -- the state model -- and the format carries nothing a segment's
 * reader does not already validate.  What it adds over a segment is the frontier: the state model's FRONTIER
 * walk at T (not a grid point's), the pools' open rows (where a pool has them) and the trailing pools, the
 * logits, the prefill frontier, and the resume checkpoint a sync that does not extend T restores. */
/** u32 words of a kv-state payload's header. */
#define KVP_U32_FIELDS 8u
/* L284: DeepSeek's retired graph-format payload (v3..v15, "DSV4"), refused by name */
/** The retired "DSV4" payload magic (little-endian), refused by name. */
#define KVP_RETIRED_DSV4_MAGIC UINT32_C(0x34565344)

/* The session's draft-depth controller (spec_depth.h) -- the speculation state a frontier carries, every family's
 * (a drafter without an adaptive policy holds it at zero).  It persists across a session's requests on purpose
 * (pulsar_spec_carry_state::spec_adaptive_depth), so a restored session drafts at the saved one's depth and runs
 * its controller from the saved counters -- not from whatever its bank served last.  So does the copy drafter's
 * measured acceptance (pulsar_spec_carry_state::copy_accept, L288): it prices the copy rounds, and the request it was
 * measured in goes on past the frontier (a new request resets it, as it would have).  The rest of the speculation
 * state is a request's lookahead (the carry, the pendings, the quench): a load resets it as every request
 * boundary does (spec_lookahead_reset), and a live session voids it on its next plain step, so neither side
 * holds it at the frontier's next round.  Format v3; the copy acceptance v5; its age (copy_unseen, L308: a stale
 * estimate's next copy chain is a probe) v6. */
/** u32 words of the speculation record (see above). */
#define KVP_SPEC_U32 6u
/** Pack the speculation state a frontier carries.  @param sp the carry state  @param rec receives the record */
static void kvp_spec_pack(const pulsar_spec_carry_state *sp, uint32_t (&rec)[KVP_SPEC_U32]) {
    rec[0] = (uint32_t)sp->spec_adaptive_depth;
    rec[1] = sp->spec_depth_down_forgiven ? 1u : 0u;
    rec[2] = sp->spec_depth_rounds_since_up;
    rec[3] = sp->spec_depth_climb_cooldown;
    memcpy(&rec[4], &sp->copy_accept, sizeof(float));
    rec[5] = sp->copy_unseen;
}
/* false = the record holds no state a session could (0 = not yet moved; a depth is 1..16; an acceptance is a rate
 * in [0, 1], 0 = not measured; its age is any count) */
/// @param rec the record
/// @return true when every field is in range
static bool kvp_spec_valid(const uint32_t (&rec)[KVP_SPEC_U32]) {
    float c;
    memcpy(&c, &rec[4], sizeof c);
    return rec[0] <= 16u && rec[1] <= 1u && rec[2] <= 255u && rec[3] <= 255u && c >= 0.0f && c <= 1.0f;
}
/** Unpack a speculation record into the carry state (its other fields untouched).  @param rec the record
 *  @param sp receives the fields */
static void kvp_spec_unpack(const uint32_t (&rec)[KVP_SPEC_U32], pulsar_spec_carry_state *sp) {
    sp->spec_adaptive_depth = (int)rec[0];
    sp->spec_depth_down_forgiven = rec[1] != 0u;
    sp->spec_depth_rounds_since_up = (uint8_t)rec[2];
    sp->spec_depth_climb_cooldown = (uint8_t)rec[3];
    memcpy(&sp->copy_accept, &rec[4], sizeof(float));
    sp->copy_unseen = rec[5];
}

/* the frontier slot's size: the state model's frontier walk, aligned as a grid slot is (checkpoint.cpp) */
/// @param st the store
/// @return bytes
static uint64_t kvp_frontier_bytes(const pulsar_ckpt_store *st) {
    uint64_t bytes = 0;
    (void)st->ops->walk(st->state, -1, true, NULL, 0, st->ops->min_checkpoint(st->state), &bytes);
    return (bytes + 255u) & ~(uint64_t)255u;
}

/* every pool, then every trailing pool; false past the bound */
/// @param st  the store
/// @param out receives the pools (PULSAR_KV_POOLS_MAX entries)
/// @param n   receives the count
/// @return false past the bound
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
/// @param st    the store
/// @param pools [n] the payload's pools
/// @param n     pools
/// @param width the logits width
/// @param bank  the bank the payload is written from / read into (its runtime adapter, L338)
/// @return the digest
static uint64_t kvp_layout_digest(pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t width,
                                  uint32_t bank) {
    uint64_t h = segment_layout_digest(st, bank);
    auto mix = [&h](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; } };
    mix(n);
    for (uint32_t i = 0; i < n; i++) { mix(pools[i].tokens_per_row); mix(pools[i].row_bytes); mix(pools[i].whole_rows); }
    mix(kvp_frontier_bytes(st));
    mix(width);
    return h;
}

/** A kv-state payload's size in bytes (the trailing digest included).
 *  @param st    the store
 *  @param pools [n] the pools carried
 *  @param n     pools
 *  @param T     the frontier, in tokens
 *  @param G     the resume checkpoint, 0 = none
 *  @param n_img image records carried
 *  @param width the logits width, in floats
 *  @return bytes */
static uint64_t kvp_bytes_for(const pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t T,
                              uint32_t G, uint32_t n_img, uint32_t width) {
    uint64_t bytes = (uint64_t)KVP_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)T * sizeof(uint32_t);
    bytes += sizeof(uint32_t) + (uint64_t)n_img * SEGMENT_IMAGE_U32 * sizeof(uint32_t);
    bytes += (uint64_t)KVP_SPEC_U32 * sizeof(uint32_t);
    bytes += (uint64_t)width * sizeof(float);
    bytes += (G ? st->slot_bytes : 0u) + kvp_frontier_bytes(st);   /* the resume checkpoint, the frontier */
    bytes += pools_span_bytes(pools, n, 0u, T);
    return bytes + sizeof(uint64_t);   /* the trailing digest */
}

/* What a save of the session writes, decided once: payload_bytes and the save read the same plan. */
/** A session save's plan (see above): filled by kvp_plan_save, read by payload_bytes and save_payload. */
struct kvp_plan {
    /// the session's kv store (pulsar_session_kv_store); borrowed
    pulsar_ckpt_store *st;
    /** @var kvp_plan::T
     *  the frontier saved, in tokens: the session's checkpoint length */
    /** @var kvp_plan::PF
     *  the bank's prefill frontier (tokens), <= T */
    /** @var kvp_plan::G
     *  the resume checkpoint's grid point (tokens): the deepest checkpoint at or below PF's grid floor; 0 = none */
    /** @var kvp_plan::width
     *  the logits row width in floats (pulsar_engine_logits_width) */
    /** @var kvp_plan::n_pools
     *  entries of pools[] in use */
    /// the live bank saved
    uint32_t bank, T, PF, G, width, n_pools;
    /// [n_pools] the append-only pools the payload carries: pools() then trailing_pools() (kvp_pools)
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    /// the payload's total size in bytes, the trailing digest included (kvp_bytes_for)
    uint64_t bytes;
};

/** Plan a save of the session's live bank (see kvp_plan); every check is made here.
 *  @param s      the session
 *  @param p      receives the plan
 *  @param err    receives the refusal
 *  @param errlen bytes of `err`
 *  @return false (err filled) when the session cannot be saved */
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
    p->PF = (uint32_t)pulsar_session_bank_prefill_frontier(s, p->bank).raw();   /* L334: the payload's raw field */
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
    pulsar_ckpt_slot_at at = {};
    if (p.G && !pulsar_ckpt_locate(st, p.bank, p.G, &at)) {
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
    pulsar_kvdigest_init(&io.digest);
    pulsar_writeback_init(&io.wb, fp);
    const uint64_t layout = kvp_layout_digest(st, p.pools, p.n_pools, p.width, pulsar_session_live_bank(s));
    const uint32_t header[KVP_U32_FIELDS] = {
        PULSAR_SESSION_KV_PAYLOAD_MAGIC, PULSAR_SESSION_KV_PAYLOAD_VERSION, (uint32_t)layout, (uint32_t)(layout >> 32),
        p.T, p.PF, p.G, p.width,
    };
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < KVP_U32_FIELDS; i++) rc = payload_write_u32(&io, header[i], err, errlen);
    for (uint32_t i = 0; rc == 0 && i < p.T; i++) rc = payload_write_u32(&io, (uint32_t)s->checkpoint.v[i], err, errlen);
    if (rc == 0) rc = payload_write_images(&io, s->live_images.b, s->live_images.n, err, errlen);
    uint32_t spec_rec[KVP_SPEC_U32];
    kvp_spec_pack(&s->spec, spec_rec);
    for (uint32_t i = 0; rc == 0 && i < KVP_SPEC_U32; i++) rc = payload_write_u32(&io, spec_rec[i], err, errlen);
    const float *live = rc == 0 ? s->live_host() : NULL;   /* L302: the payload carries the live row */
    if (rc == 0 && !live) {
        snprintf(err, errlen, "payload: the live row could not be read from the device");
        rc = 1;
    }
    if (rc == 0) rc = payload_write_bytes(&io, live, (uint64_t)p.width * sizeof(float), err, errlen);
    uint8_t *buf = s->io_buf;   /* pinned once at create (pulsar_session::io_buf) */
    if (rc == 0 && p.G)   /* L347 I4: a host-tier slot is written from its pinned bytes */
        rc = at.host ? payload_write_bytes(&io, at.host, st->slot_bytes, err, errlen)
                     : payload_write_tensor_span(&io, at.slab, at.off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, err,
                                                 errlen);
    if (rc == 0) rc = payload_write_tensor_span(&io, front, 0, fb, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0) rc = pools_span_io(&io, p.pools, p.n_pools, 0u, p.T, true, buf, NULL, err, errlen);
    pulsar_gpu_tensor_free(front);
    if (rc != 0) return rc;
    const uint64_t dg = pulsar_kvdigest_final(&io.digest);
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
    pulsar_kvdigest_init(&io.digest);
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
    const uint64_t layout = kvp_layout_digest(st, pools, n, width, bank);
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
    uint32_t spec_rec[KVP_SPEC_U32];
    for (uint32_t k = 0; k < KVP_SPEC_U32; k++)
        if (payload_read_u32(&io, &spec_rec[k], &remaining, err, errlen) != 0) return drop();
    if (!kvp_spec_valid(spec_rec)) {
        payload_set_err(err, errlen, "session payload: its speculation record holds no state a session could");
        return drop();
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
    pulsar_ckpt_slot_at at = {};
    if (G && !pulsar_ckpt_claim(st, bank, G, &at)) {
        payload_set_err(err, errlen, "session payload: no grid-checkpoint slot");
        return fail();
    }
    const uint64_t fb = kvp_frontier_bytes(st);
    front = pulsar_gpu_tensor_alloc(fb);
    if (!front) {
        payload_set_err(err, errlen, "session payload: staging the frontier's state failed");
        return fail();
    }
    uint8_t *buf = s->io_buf;   /* pinned once at create (pulsar_session::io_buf) */
    int rc = 0;
    if (G)   /* L347 I4: a host-tier slot is read into its pinned bytes */
        rc = at.host ? payload_read_bytes(&io, at.host, st->slot_bytes, &remaining, err, errlen)
                     : payload_read_tensor_span(&io, at.slab, at.off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK,
                                                &remaining, err, errlen);
    if (rc == 0) rc = payload_read_tensor_span(&io, front, 0, fb, buf, PULSAR_SESSION_IO_CHUNK,
                                               &remaining, err, errlen);
    if (rc == 0) rc = pools_span_io(&io, pools, n, 0u, T, false, buf, &remaining, err, errlen);
    if (rc != 0) return fail();
    uint64_t stored_dg = 0;
    if (remaining != sizeof(uint64_t) || fread(&stored_dg, 1, sizeof(stored_dg), fp) != sizeof(stored_dg) ||
        stored_dg != pulsar_kvdigest_final(&io.digest)) {
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
    if (G) pulsar_ckpt_commit(st, bank, at.slot, G);
    token_vec_free(&s->checkpoint);
    s->checkpoint = toks;
    s->live_images = imgs;
    const bool live_ok = s->live_from_host(logits);
    free(logits);
    if (!live_ok) return fail();
    s->prefill_frontier = (int)PF;   /* the session's mirror (DeepSeek's own: L195), as a fused prompt chunk sets it */
    s->checkpoint_valid = true;
    s->logits_stale = false;
    spec_lookahead_reset(s);
    kvp_spec_unpack(spec_rec, &s->spec);
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

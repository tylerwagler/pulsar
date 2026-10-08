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



/* An index-K pool exists only where the layer runs an indexer: 0731's ratio-128
 * (HCA) layers publish compressed KV and no indexer at all, so their
 * layer_index_comp_cache is never allocated.  One emit writes a comp row AND an
 * index-K row, so the row COUNT here is the source's own n_comp -- there is no
 * second counter to drift.
 *
 * v10 wrote the pool for every kv source, which is sound only on V4.1 (all four
 * of its sources run an indexer) and made EVERY 0731 checkpoint refuse to save:
 * "session tensor is smaller than the payload (offset 0 + 2176 > 0 bytes)" is
 * layer 3 -- ratio 128, no indexer -- at 4096 tokens, 32 rows x 68 B. */
static bool layer_has_index_pool(uint32_t il) {
    return pulsar_attn_runs_indexer(pulsar_layer_attn_layout(il)->mode);
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
/* The frontier's recurrent lanes, in the one order save and load walk them:
 * per layer the attention compressor's (kv, score) and the V4 indexer
 * compressor's (kv, score), each at the size its ALLOCATION holds -- a span
 * shorter than the lane round-trips its own accounting and leaves the lane's
 * tail primed (v11: a 0731 ratio-4 lane is 32768 B and the payload carried
 * 8192). */
static void payload_frontier_lanes(pulsar_gpu_graph *g, uint32_t il, pulsar_gpu_tensor *out[4]) {
    out[0] = g->layer_attn_state_kv[il];
    out[1] = g->layer_attn_state_score[il];
    out[2] = g->layer_index_state_kv[il];
    out[3] = g->layer_index_state_score[il];
}

/* The raw window a payload carries: the last raw_window positions before the
 * frontier, in logical order (the reader re-scatters them into its own ring). */
static uint32_t payload_raw_rows(const pulsar_gpu_graph *g, uint32_t tokens) {
    return tokens < g->raw_window ? tokens : g->raw_window;
}

static uint64_t session_payload_live_tensor_bytes(pulsar_gpu_graph *g, uint32_t tokens, bool has_ckpt) {
    /* Comp rows are sized in the format the pools hold them in (MAIN rows; v4
     * had sized them at the f32 stride, over-reserving the disk cache 3.5x). */
    uint64_t bytes = has_ckpt ? g->ckpt.slot_bytes : 0u;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        bytes += (uint64_t)payload_raw_rows(g, tokens) * pulsar_kv_row_bytes(PULSAR_KV_ROW_RING);
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint64_t rows = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
        bytes += rows * pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP);
        if (layer_has_index_pool(il)) bytes += rows * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX);
        pulsar_gpu_tensor *lanes[4];
        payload_frontier_lanes(g, il, lanes);
        for (int k = 0; k < 4; k++) if (lanes[k]) bytes += pulsar_gpu_tensor_bytes(lanes[k]);
    }
    return bytes;
}

/* The resume checkpoint a payload carries: the deepest at or below the prefill
 * frontier's grid point -- what a sync that does not extend the frontier
 * exactly would restore.  0 when the session holds none (a short one). */
static uint32_t payload_resume_checkpoint(pulsar_session *s) {
    const uint32_t pf = s->prefill_frontier < 0 ? 0u : (uint32_t)s->prefill_frontier;
    return pulsar_ckpt_best(&s->graph->ckpt, gpu_graph_cur_bank(s->graph), pulsar_ckpt_grid_floor(&s->graph->ckpt, pf));
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



/* v5: the indexer comp cache is written in the format it is held in, exactly as
 * v4 did for the attention comp cache.  It used to dequantise MXKV-FP4 rows into
 * a 512 B/row f32 staging buffer, write that, and re-pack on load -- 68 B of
 * content stored as 512 B, and a re-encode on every restore. */
static int payload_write_index_comp(payload_io *io, pulsar_gpu_graph *g, uint32_t il,
                                    uint32_t row0, uint32_t n_rows, uint8_t *buf, size_t cap,
                                    char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t rb = pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX);
    return payload_write_tensor_span(io, g->layer_index_comp_cache[il], (uint64_t)row0 * rb, (uint64_t)n_rows * rb,
                                     buf, cap, err, errlen);
}

static int payload_read_index_comp(payload_io *io, pulsar_gpu_graph *g, uint32_t il,
                                   uint32_t row0, uint32_t n_rows, uint8_t *buf, size_t cap,
                                   uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t rb = pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX);
    /* Straight into the packed cache.  The old load path RE-ENCODED, which only
     * stayed safe because it used an exact integer-math scale bucket -- and
     * 2026-08-18 measured that the fast-math bucket is NOT value-idempotent
     * (removing an analogous double-quantise on the attn side moved decode
     * acceptance).  Copying bytes cannot invoke an idempotence it never relies
     * on, which is the same reason v4 gave for the attention rows, and it is
     * why the re-encode machinery could then be deleted outright. */
    return payload_read_tensor_span(io, g->layer_index_comp_cache[il], (uint64_t)row0 * rb, (uint64_t)n_rows * rb,
                                    buf, cap, remaining, err, errlen);
}

/* The comp cache is written in the format it is HELD in: MAIN rows.
 *
 * Payload v3 stored f32. Saving dequantised 584 B rows into 2048 B, wrote that,
 * and loading read it back and re-encoded -- a full round trip through a format
 * neither end holds, for a file 3.5x larger than its own contents. The f32 buffer
 * that round trip needed was the last f32 KV allocation in the engine.
 *
 * It also removes the re-encode entirely, which is worth more than the bytes: the
 * load side needed the EXACT-scale repack because "the fast-math quantize bucket
 * is not bit-idempotent at scale boundaries". Copying packed bytes cannot lose an
 * idempotence it never invokes. */
static int payload_write_attn_comp_pack(payload_io *io, pulsar_gpu_graph *g, uint32_t il,
                                    uint32_t row0, uint32_t n_rows, uint8_t *buf, size_t cap,
                                    char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t rb = pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP);
    return payload_write_tensor_span(io, g->layer_attn_comp_cache[il], (uint64_t)row0 * rb, (uint64_t)n_rows * rb,
                                     buf, cap, err, errlen);
}

static int payload_read_attn_comp_pack(payload_io *io, pulsar_gpu_graph *g, uint32_t il,
                                   uint32_t row0, uint32_t n_rows, uint8_t *buf, size_t cap,
                                   uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t rb = pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP);
    /* Straight into the packed cache: the file holds exactly what it holds, so
     * there is no staging buffer and no re-encode on either side.  Under KV4
     * this is what makes save/load safe at all -- an FP4 re-encode misrounds
     * ~33% of blocks, so the bytes ARE the values.  The version plus the
     * h[13] stride refuse files from any earlier row format. */
    return payload_read_tensor_span(io, g->layer_attn_comp_cache[il], (uint64_t)row0 * rb, (uint64_t)n_rows * rb,
                                    buf, cap, remaining, err, errlen);
}



/* L284: the kv-state payload (below the segments, built from their parts) -- the payload of a model whose
 * state ops declare the frontier (kv_state.h); DeepSeek's graph keeps the format of this section. */
static bool kvp_model(pulsar_session *s);
static uint64_t kvp_payload_bytes(pulsar_session *s);
static int kvp_save(pulsar_session *s, FILE *fp, char *err, size_t errlen);
static int kvp_load(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* A restore replaces the state every speculative lookahead was conditioned on: the carry token, the
 * pre-drafted pendings, the quench.  Dropped up front (a restore failing midway may already have
 * overwritten what they read) and again at the commit. */
static void payload_drop_lookahead(pulsar_session *s) {
    s->spec.spec_carry_valid = false;
    pulsar_spec_drop_pendings(&s->spec);
    spec_quench_reset(s);
}

uint64_t pulsar_session::payload_bytes() {
    auto *s = this;
    if (s && kvp_model(s)) return kvp_payload_bytes(s);
    if (!s || !s->checkpoint_valid) return 0;
    pulsar_gpu_graph *g = s->graph;
    uint64_t bytes = (uint64_t)PULSAR_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += sizeof(uint32_t) + (uint64_t)s->live_images.n * SEGMENT_IMAGE_U32 * sizeof(uint32_t);   /* L281 v14 */
    bytes += (uint64_t)PULSAR_N_VOCAB * sizeof(float);   /* the frontier's logits */
    bytes += sizeof(uint32_t);                            /* the prefill frontier */
    /* ONE per-layer row-count array.  There were two while the index-K pool had
     * its own counter; the second line outlived it, so every advertised size was
     * PULSAR_N_LAYER * 4 (172 B on 0731) larger than what save_payload wrote.
     * That is not a cosmetic mismatch: save_snapshot sizes the memory stream from
     * THIS number and load_snapshot hands the same number back as the payload
     * length, so a restore of a live session ended on "KV checkpoint has trailing
     * payload bytes" -- the bench caught it, the disk path (which passes the real
     * file size) did not.  The gate's written == payload_bytes() check is the
     * guard; keep them one fact. */
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len, payload_resume_checkpoint(s) != 0u);
    /* v10: the trailing digest, appended raw after the data. */
    bytes += sizeof(uint64_t);
    return bytes;
}



/* Raw-ring row spans, one WINDOW row (pulsar_kv_row_bytes(PULSAR_KV_ROW_RING))
 * at the row's physical slot. */
static int payload_raw_row(payload_io *io, pulsar_gpu_graph *g, uint32_t il, uint32_t pos, bool write,
                           uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen) {
    const uint64_t rb = pulsar_kv_row_bytes(PULSAR_KV_ROW_RING);
    const uint64_t off = (uint64_t)(pos % g->raw_cap) * rb;
    return write ? payload_write_tensor_span(io, g->layer_raw_cache[il], off, rb, buf, cap, err, errlen)
                 : payload_read_tensor_span(io, g->layer_raw_cache[il], off, rb, buf, cap, remaining, err, errlen);
}

int pulsar_session::save_payload(FILE *fp, char *err, size_t errlen) {
    auto *s = this;
    if (s && kvp_model(s)) return kvp_save(s, fp, err, errlen);
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->prefill_frontier < 0 || s->prefill_frontier > s->checkpoint.len) {
        payload_set_err(err, errlen, "prefill frontier lies outside the checkpoint");
        return 1;
    }
    pulsar_gpu_graph *g = s->graph;
    if (g->ms_comp_state_stale[gpu_graph_cur_bank(g)]) {
        payload_set_err(err, errlen, "session's state is stale (rewound off its grid checkpoints); sync it first");
        return 1;
    }
    /* L264: the resume checkpoint, carried beside the frontier state so a
     * restored session that is then synced to a prompt it does not extend
     * resumes exactly where this one would. */
    const uint32_t G = payload_resume_checkpoint(s);
    pulsar_gpu_tensor *slab = NULL;
    uint64_t slot_off = 0;
    if (G != 0u && !pulsar_ckpt_locate(&g->ckpt, gpu_graph_cur_bank(g), G, &slab, &slot_off)) {
        payload_set_err(err, errlen, "session's resume checkpoint vanished while saving");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, fp);

    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 checkpoint slot bytes,
     *   13 main (comp pool) row bytes, 14 indexer fp4 row bytes,
     *   15 the resume checkpoint's grid point (0 = none),
     *   16 window (raw ring) row bytes.
     *
     * 13/14/16 are the STORAGE FORMAT, not the shape.  Fields 9-11 already caught a
     * file written for a different model; these catch one written for a
     * different row LAYOUT at the same shape -- which is what a KV format change
     * produces, and which the version alone was guarding until 2026-08-18.
     */
    uint32_t header[PULSAR_SESSION_PAYLOAD_U32_FIELDS] = {
        PULSAR_SESSION_PAYLOAD_MAGIC,
        PULSAR_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        PULSAR_N_LAYER,
        PULSAR_N_HEAD_DIM,
        PULSAR_N_INDEXER_HEAD_DIM,
        PULSAR_N_VOCAB,
        (uint32_t)g->ckpt.slot_bytes,
        /* the row strides -- with the payload version, refuse any
         * earlier-format file */
        (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP),
        (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX),
        G,
        (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_RING),
    };
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(&io, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(&io, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    /* L281 (v14): which images the checkpoint's sentinel blocks are -- a payload carries the rows, so it carries
     * the records, or its restore would pair them with whatever the session held before */
    if (payload_write_images(&io, s->live_images.b, s->live_images.n, err, errlen) != 0) return 1;
    if (payload_write_bytes(&io, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    if (payload_write_u32(&io, (uint32_t)s->prefill_frontier, err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(&io, gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il), err, errlen) != 0) return 1;
    }

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    if (G != 0u)
        rc = payload_write_tensor_span(&io, slab, slot_off, g->ckpt.slot_bytes,
                                       buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    const uint32_t ck = (uint32_t)s->checkpoint.len;
    const uint32_t raw_live = payload_raw_rows(g, ck);
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* The frontier's raw window, in logical position order. */
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++)
            rc = payload_raw_row(&io, g, il, ck - raw_live + r, true, buf, PULSAR_SESSION_IO_CHUNK, NULL, err, errlen);
        if (rc != 0 || !gpu_graph_layer_is_kv_source(il)) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous; the index-K rows the same emits wrote follow where an
         * indexer runs.  Then the frontier's recurrent lanes. */
        const uint32_t rows = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
        rc = payload_write_attn_comp_pack(&io, g, il, 0u, rows, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
        if (rc == 0 && layer_has_index_pool(il))
            rc = payload_write_index_comp(&io, g, il, 0u, rows, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
        pulsar_gpu_tensor *lanes[4];
        payload_frontier_lanes(g, il, lanes);
        for (int k = 0; rc == 0 && k < 4; k++)
            if (lanes[k]) rc = payload_write_tensor_span(&io, lanes[k], 0, pulsar_gpu_tensor_bytes(lanes[k]),
                                                         buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
    }
    free(buf);
    if (rc == 0) {
        /* The digest covers every byte above; it is appended RAW so it does not
         * cover itself.  load_payload requires exactly these 8 bytes to remain
         * after the data and refuses on mismatch. */
        const uint64_t dg = payload_digest_final(&io.digest);
        if (fwrite(&dg, 1, sizeof(dg), fp) != sizeof(dg)) {
            payload_set_err(err, errlen, "failed to write session payload digest");
            return 1;
        }
        pulsar_writeback_finish(&io.wb);
    }
    return rc;
}



int pulsar_session::load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    if (kvp_model(s)) return kvp_load(s, fp, payload_bytes, err, errlen);
    payload_drop_lookahead(s);
    /* L264: same argument for the bank's grid checkpoints -- they reference the
     * rows this load overwrites. */
    pulsar_ckpt_drop_bank(&s->graph->ckpt, gpu_graph_cur_bank(s->graph));
    payload_io io;
    io.fp = fp;
    payload_digest_init(&io.digest);
    pulsar_writeback_init(&io.wb, NULL);   /* a read stream */

    uint64_t remaining = payload_bytes;
    uint32_t h[PULSAR_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(&io, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != PULSAR_SESSION_PAYLOAD_MAGIC || h[1] != PULSAR_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    pulsar_gpu_graph *g = s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_slot_bytes = h[12];
    const uint32_t saved_grid = h[15];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != PULSAR_N_LAYER || h[9] != PULSAR_N_HEAD_DIM ||
        h[10] != PULSAR_N_INDEXER_HEAD_DIM || h[11] != PULSAR_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different Pulsar layout");
        return 1;
    }
    /* Storage format, checked separately from shape: a KV format change keeps
     * head_dim and moves the row STRIDE, so the checks above would pass it.
     * Every row span below is addressed with these strides, so a mismatch here
     * is the difference between refusing a file and decoding noise into a cache. */
    if (h[13] != (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP) ||
        h[14] != (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX) ||
        h[16] != (uint32_t)pulsar_kv_row_bytes(PULSAR_KV_ROW_RING))
    {
        payload_set_err(err, errlen,
                        "KV checkpoint row strides differ from this build "
                        "(window/main/indexer storage format changed)");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout, and
     * raw_cap only says where a window row lives in a ring -- the rows travel
     * by position, so neither has to match. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window || saved_slot_bytes != (uint32_t)g->ckpt.slot_bytes) {
        payload_set_err(err, errlen, "KV checkpoint's window or grid-checkpoint layout does not match this runtime");
        return 1;
    }
    if (saved_grid % g->ckpt.ops->resume_grid != 0u || saved_grid > saved_tokens) {
        payload_set_err(err, errlen, "KV checkpoint's resume grid point is not a grid point inside its token count");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(&io, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    /* L281 (v14): the image records, installed at commit */
    pulsar_image_identity new_images;
    new_images.n = 0;
    {
        uint32_t n_img = 0;
        bool ok = payload_read_u32(&io, &n_img, &remaining, err, errlen) == 0;
        if (ok && n_img > PULSAR_IMAGE_BLOCKS_MAX) {
            payload_set_err(err, errlen, "KV checkpoint carries more image blocks than a session holds");
            ok = false;
        }
        for (uint32_t i = 0; ok && i < n_img; i++) {
            uint32_t rec[SEGMENT_IMAGE_U32];
            for (uint32_t k = 0; ok && k < SEGMENT_IMAGE_U32; k++)
                ok = payload_read_u32(&io, &rec[k], &remaining, err, errlen) == 0;
            const pulsar_image_block b = image_rec_unpack(rec);
            if (ok && (b.end <= b.start || b.end > saved_tokens || (i && b.start < new_images.b[i - 1].end))) {
                payload_set_err(err, errlen, "KV checkpoint's image records are out of order or outside its tokens");
                ok = false;
            }
            if (ok) new_images.b[new_images.n++] = b;
        }
        if (!ok) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
    }
    if (payload_read_bytes(&io, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t saved_pf = 0;
    if (payload_read_u32(&io, &saved_pf, &remaining, err, errlen) != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (saved_pf > saved_tokens || saved_grid > saved_pf) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint's prefill frontier is inconsistent with its tokens and grid point");
        return 1;
    }
    uint32_t n_comp[PULSAR_MAX_LAYER];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(&io, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        /* The frontier's rows, and the resume checkpoint references rows
         * [0, G/ratio): a file that holds fewer would resume onto rows it never
         * carried. */
        const bool source = gpu_graph_layer_is_kv_source(il);
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il] ||
            (source && n_comp[il] != saved_tokens / pulsar_layer_compress_ratio(il))) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }

    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    uint32_t slot = PULSAR_CKPT_SLOTS_MAX;
    if (saved_grid != 0u) {
        pulsar_gpu_tensor *slab = NULL;
        uint64_t slot_off = 0;
        if (!pulsar_ckpt_claim(&g->ckpt, gpu_graph_cur_bank(g), saved_grid, &slab, &slot_off, &slot)) {
            payload_set_err(err, errlen, "no grid-checkpoint slot for the restored bank");
            rc = 1;
        } else {
            rc = payload_read_tensor_span(&io, slab, slot_off, g->ckpt.slot_bytes,
                                          buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        }
    }
    const uint32_t raw_live = payload_raw_rows(g, saved_tokens);
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Re-scatter the window into the current ring: the file holds rows by
         * position, not by the writer's ring layout. */
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++)
            rc = payload_raw_row(&io, g, il, saved_tokens - raw_live + r, false, buf, PULSAR_SESSION_IO_CHUNK,
                                 &remaining, err, errlen);
        if (rc != 0 || !gpu_graph_layer_is_kv_source(il)) continue;
        rc = payload_read_attn_comp_pack(&io, g, il, 0u, n_comp[il], buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        if (rc == 0 && layer_has_index_pool(il))
            rc = payload_read_index_comp(&io, g, il, 0u, n_comp[il], buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        pulsar_gpu_tensor *lanes[4];
        payload_frontier_lanes(g, il, lanes);
        for (int k = 0; rc == 0 && k < 4; k++)
            if (lanes[k]) rc = payload_read_tensor_span(&io, lanes[k], 0, pulsar_gpu_tensor_bytes(lanes[k]),
                                                        buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    /* v10: the trailing 8 bytes are the digest over every byte above.  A
     * mismatch is a corrupted payload -- refuse it (the caller treats this as a
     * cache miss and re-prefills), never decode byte-rot into a live cache. */
    if (remaining != sizeof(uint64_t)) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint is missing its payload digest");
        return 1;
    }
    uint64_t stored_dg = 0;
    if (fread(&stored_dg, 1, sizeof(stored_dg), fp) != sizeof(stored_dg)) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to read session payload digest");
        return 1;
    }
    if (stored_dg != payload_digest_final(&io.digest)) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint digest mismatch (corrupt payload)");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    s->live_images = new_images;   /* L281 */
    const uint32_t bank = gpu_graph_cur_bank(g);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        gpu_graph_set_n_comp(g, bank, il, n_comp[il]);
    }
    /* The frontier is installed first: a commit before it could be dropped by a
     * frontier write.  The bank now stands at its frontier with that frontier's
     * window, lanes and logits -- live, decodable as is -- and holds the resume
     * checkpoint a non-extending sync restores. */
    if (slot != PULSAR_CKPT_SLOTS_MAX) pulsar_ckpt_commit(&g->ckpt, gpu_graph_cur_bank(g), slot, saved_grid);
    g->ms_comp_state_stale[bank] = false;
    s->prefill_frontier = (int)saved_pf;   /* L195: the next sync resumes from the grid point below it */
    s->checkpoint_valid = true;
    /* a restored state invalidates any in-flight speculative lookahead: the
     * carry token, pre-drafted pendings, AND the drafter's context-KV ring
     * were all conditioned on the replaced state. Leaving the ring makes the
     * next drafts (and therefore the verify batch shapes) depend on whatever
     * ran before the restore — the source of run-to-run tie flips. */
    payload_drop_lookahead(s);
    for (int li = 0; li < 3; li++) g->dspark_n_raw[li] = 0;
    g->dspark_prompt_n = 0;
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

/* The bank holds nothing: a chain's root replaces its history, a failed load leaves it empty.  A
 * bank-pool family's lanes are written by the chain's last restore and its counters by
 * set_frontier_stale, so its host view and its checkpoints are all there is to clear. */
static void segment_clear(pulsar_session *s, pulsar_ckpt_store *st, uint32_t bank) {
    s->live_images.n = 0;   /* L281: the bank holds no block */
    if (FAMILY_BANKS(s)) {
        s->engine->family->session->invalidate(s);
        pulsar_ckpt_drop_bank(st, bank);
    } else {
        s->invalidate();
    }
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

/* Pool rows [t0 / tokens_per_row, ceil(t1 / tokens_per_row)) of every pool: a segment's span (grid points,
 * whole rows) or a kv-state payload's [0, frontier) (its open row too).  The one sizing and the one copy of
 * pool rows either carries. */
static uint32_t pool_row(const pulsar_kv_pool &p, uint32_t t) { return (t + p.tokens_per_row - 1u) / p.tokens_per_row; }

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
    if (!FAMILY_BANKS(s)) s->prefill_frontier = (int)G;   /* a bank-pool family's frontier is set_frontier_stale's */
    s->checkpoint_valid = true;
    if (last && !(FAMILY_BANKS(s) ? pulsar_ckpt_restore(st, bank, G) : s->restore_checkpoint(G))) {
        payload_set_err(err, errlen, "load segment: restoring the chain's last checkpoint failed");
        segment_clear(s, st, bank);
        return 1;
    }
    if (last) s->logits_stale = true;   /* a restore moves the KV, not the logits (restore_checkpoint sets it too) */
    if (G_out) *G_out = G;
    return 0;
}



/* ===== L284: the kv-state payload =========================================
 * A payload of a model whose state ops declare the frontier (kv_state.h; Qwen) is a segment's parts at the
 * session's frontier T (pulsar.h, PULSAR_SESSION_KV_PAYLOAD_MAGIC): the same layout digest, image section,
 * slot walk, pool copy and digested stream, so the bytes have one authority -- the state model -- and the
 * format carries nothing a segment's reader does not already validate.  What it adds over a segment is the
 * frontier: the walk's slot taken AT T (not at a grid point), the pools' open rows and the trailing pools,
 * the logits, the prefill frontier, and the resume checkpoint a sync that does not extend T restores. */
#define KVP_U32_FIELDS 8u

static bool kvp_model(pulsar_session *s) {
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    return st && st->ops && st->ops->frontier_at;
}

/* every pool, then every trailing pool; false past the bound */
static bool kvp_pools(pulsar_ckpt_store *st, pulsar_kv_pool *out, uint32_t *n) {
    const uint32_t a = st->ops->pools(st->state, out, PULSAR_KV_POOLS_MAX);
    if (a > PULSAR_KV_POOLS_MAX) return false;
    const uint32_t b = st->ops->trailing_pools(st->state, out + a, PULSAR_KV_POOLS_MAX - a);
    if (b > PULSAR_KV_POOLS_MAX - a) return false;
    *n = a + b;
    return true;
}

/* the segment's layout digest, extended by every pool the payload carries and the logits width */
static uint64_t kvp_layout_digest(pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t width) {
    uint64_t h = segment_layout_digest(st);
    auto mix = [&h](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; } };
    mix(n);
    for (uint32_t i = 0; i < n; i++) { mix(pools[i].tokens_per_row); mix(pools[i].row_bytes); }
    mix(width);
    return h;
}

static uint64_t kvp_bytes_for(const pulsar_ckpt_store *st, const pulsar_kv_pool *pools, uint32_t n, uint32_t T,
                              uint32_t G, uint32_t n_img, uint32_t width) {
    uint64_t bytes = (uint64_t)KVP_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)T * sizeof(uint32_t);
    bytes += sizeof(uint32_t) + (uint64_t)n_img * SEGMENT_IMAGE_U32 * sizeof(uint32_t);
    bytes += (uint64_t)width * sizeof(float);
    bytes += (G ? st->slot_bytes : 0u) + st->slot_bytes;   /* the resume checkpoint, the frontier */
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
    if (!p->st->ops->frontier_at(p->st->state, p->T, &p->PF, why, sizeof(why)) || p->PF > p->T) {
        char msg[256];
        snprintf(msg, sizeof(msg), "session payload: %s", why);
        payload_set_err(err, errlen, msg);
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

static uint64_t kvp_payload_bytes(pulsar_session *s) {
    kvp_plan p;
    char err[256];
    return kvp_plan_save(s, &p, err, sizeof(err)) ? p.bytes : 0u;
}

static int kvp_save(pulsar_session *s, FILE *fp, char *err, size_t errlen) {
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
    /* the frontier's slot, staged: the state model's walk at T, over zeros (the slot's alignment tail is
     * written too, so the same state always saves the same bytes) */
    pulsar_gpu_tensor *front = pulsar_gpu_tensor_alloc(st->slot_bytes);
    if (!front || pulsar_gpu_tensor_fill_f32(front, 0.0f, st->slot_bytes / sizeof(float)) == 0 ||
        !st->ops->walk(st->state, 0, front, 0, p.T, NULL) || pulsar_gpu_synchronize() == 0) {
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
    if (rc == 0) rc = payload_write_tensor_span(&io, front, 0, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
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

static int kvp_load(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (pulsar_session_is_mirrored(s)) {
        payload_set_err(err, errlen, "session payload: a tensor-parallel session's payload is not mirrored");
        return 1;
    }
    payload_drop_lookahead(s);
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
    front = pulsar_gpu_tensor_alloc(st->slot_bytes);
    if (!front) {
        payload_set_err(err, errlen, "session payload: staging the frontier's state failed");
        return fail();
    }
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    if (G) rc = payload_read_tensor_span(&io, slab, slot_off, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, &remaining,
                                         err, errlen);
    if (rc == 0) rc = payload_read_tensor_span(&io, front, 0, st->slot_bytes, buf, PULSAR_SESSION_IO_CHUNK, &remaining,
                                               err, errlen);
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
    if (!st->ops->install_frontier(st->state, T, PF) || !st->ops->walk(st->state, 1, front, 0, T, NULL) ||
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
    s->checkpoint_valid = true;
    s->logits_stale = false;
    payload_drop_lookahead(s);
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

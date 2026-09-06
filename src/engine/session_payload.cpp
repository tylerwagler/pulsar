#include "pulsar_engine_internal.h"

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



static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}



static int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = (uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



static int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}



static int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}



static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > PULSAR_SESSION_IO_CHUNK ? PULSAR_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}



static uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = pulsar_compress_coff(ratio);
    return (uint64_t)coff * PULSAR_N_HEAD_DIM * coff * ratio * sizeof(float);
}



static uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = pulsar_compress_coff(ratio);
    return (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}



/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical GPU tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : PULSAR_N_SWA;
    /* L195: a restored checkpoint resumes from the grid point G <= it (at most
     * PULSAR_RESUME_GRID - 1 below) after a warm-up over the PULSAR_WARMUP_TOKENS
     * tokens before G, whose attention reaches raw_window further down: the
     * window a fresh bank needs is [G - warmup - raw_window, checkpoint). */
    rows += PULSAR_RESUME_GRID - 1u + PULSAR_WARMUP_TOKENS;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}



/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    /* v4 stores raw AND comp rows in the format the caches hold them in, so
     * both are PULSAR_ATTN_PACK rows.  This sized them at the f32 stride --
     * 2048 B against the real 584 B -- which over-reserved the disk cache by
     * 3.5x on the KV bulk of every payload. */
    /* L111: the comp pool's row size follows the active PULSAR_KV4 format;
     * raw rows stay E4M3. */
    const uint64_t comp_row = gpu_graph_attn_comp_cache_row_bytes();
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) * comp_row;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il) * PULSAR_ENGINE_IDXFP4_ROWBYTES;
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const pulsar_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (pulsar_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static int payload_read_tensor_span(FILE *fp, pulsar_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
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
static int payload_write_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                    uint32_t n_rows, uint8_t *buf, size_t cap,
                                    char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_ENGINE_IDXFP4_ROWBYTES;
    return payload_write_tensor_span(fp, g->layer_index_comp_cache[il], 0, bytes,
                                     buf, cap, err, errlen);
}

static int payload_read_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                   uint32_t n_rows, uint8_t *buf, size_t cap,
                                   uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_ENGINE_IDXFP4_ROWBYTES;
    /* Straight into the packed cache.  The old load path RE-ENCODED, which only
     * stayed safe because it used an exact integer-math scale bucket -- and
     * 2026-08-18 measured that the fast-math bucket is NOT value-idempotent
     * (removing an analogous double-quantise on the attn side moved decode
     * acceptance).  Copying bytes cannot invoke an idempotence it never relies
     * on, which is the same reason v4 gave for the attention rows, and it is
     * why the re-encode machinery could then be deleted outright. */
    return payload_read_tensor_span(fp, g->layer_index_comp_cache[il], 0, bytes,
                                    buf, cap, remaining, err, errlen);
}

/* The comp cache is written in the format it is HELD in: PULSAR_ATTN_PACK rows.
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
static int payload_write_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                        uint32_t n_rows, uint8_t *buf, size_t cap,
                                        char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * gpu_graph_attn_comp_cache_row_bytes();
    return payload_write_tensor_span(fp, g->layer_attn_comp_cache[il], 0, bytes,
                                     buf, cap, err, errlen);
}

static int payload_read_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                       uint32_t n_rows, uint8_t *buf, size_t cap,
                                       uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * gpu_graph_attn_comp_cache_row_bytes();
    /* Straight into the packed cache: the file holds exactly what it holds, so
     * there is no staging buffer and no re-encode on either side.  Under KV4
     * this is what makes save/load safe at all -- an FP4 re-encode misrounds
     * ~33% of blocks, so the bytes ARE the values.  The version (v7) plus the
     * h[13] stride refuse files from any earlier row format. */
    return payload_read_tensor_span(fp, g->layer_attn_comp_cache[il], 0, bytes,
                                    buf, cap, remaining, err, errlen);
}




/* Raw-ring row spans.  The ring is PULSAR_ATTN_PACK, like every other KV buffer
 * -- one row is PULSAR_ENGINE_ATTN_PACK_ROWBYTES, not PULSAR_N_HEAD_DIM __half
 * containers.  This path still said f16 long after the ring stopped being f16:
 * it read at a 1024 B stride from a 584 B/row buffer and reinterpreted packed
 * bytes as halves, so it walked the wrong rows AND past the end of the
 * allocation.  Nothing caught it because ->bytes is a number and __half is a
 * legal way to look at those bytes. */
static int payload_write_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    return payload_write_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_ENGINE_ATTN_PACK_ROWBYTES,
            (uint64_t)PULSAR_ENGINE_ATTN_PACK_ROWBYTES, buf, cap, err, errlen);
}

static int payload_read_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                char *err, size_t errlen) {
    return payload_read_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_ENGINE_ATTN_PACK_ROWBYTES,
            (uint64_t)PULSAR_ENGINE_ATTN_PACK_ROWBYTES, buf, cap, remaining, err, errlen);
}

uint64_t pulsar_session::payload_bytes() {
    auto *s = this;
    if (!s || !s->checkpoint_valid) return 0;
    const pulsar_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)PULSAR_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
}



int pulsar_session_write_staged_payload(const pulsar_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}



void pulsar_session_payload_file_free(pulsar_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}



int pulsar_session::stage_payload(pulsar_session_payload_file *out,
                              const char *stage_dir,
                              char *err, size_t errlen) {
    auto *s = this;
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    /* L110 F5: stage in the CALLER's disk directory, not /tmp. On the GB10
     * serving box /tmp is tmpfs, and unified memory means a multi-GiB staged
     * payload competes with GPU KV banks for the same physical pool -- worst
     * at deep-session stores, exactly when memory is tightest. NULL falls
     * back to /tmp for callers with no disk dir (tests). */
    char tmpl[1024];
    const int n = snprintf(tmpl, sizeof tmpl, "%s/ds4-session-payload.XXXXXX",
                           stage_dir && stage_dir[0] ? stage_dir : "/tmp");
    if (n < 0 || (size_t)n >= sizeof tmpl - 32) {
        payload_set_err(err, errlen, "session payload staging dir path too long");
        return 1;
    }
    int fd = mkstemp(tmpl);
    /* Rename onto the house "<name>.tmp.<pid>" convention so the kvstore's
     * boot-scan orphan GC reclaims a crash-abandoned staged file (it lives in
     * the store dir now); the live-pid check keeps in-flight saves safe when
     * two servers share a cache dir. Best-effort: on rename failure keep the
     * mkstemp name (worst case a crash leaks one skipped file, /tmp-era
     * behavior). */
    char named[1056];
    if (fd >= 0) {
        snprintf(named, sizeof named, "%s.tmp.%ld", tmpl, (long)getpid());
        if (rename(tmpl, named) == 0) memcpy(tmpl, named, strlen(named) + 1);
    }
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = s->save_payload(fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = pulsar_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}



int pulsar_session::save_payload(FILE *fp, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    pulsar_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    if (s->prefill_frontier < 0 || s->prefill_frontier > s->checkpoint.len) {
        payload_set_err(err, errlen, "prefill frontier lies outside the checkpoint");
        return 1;
    }
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below,
     *   13 attn-pack row bytes, 14 indexer fp4 row bytes,
     *   15 prefill frontier (L195: the last position a PREFILL wrote; the resume grid point derives from it).
     *
     * 13/14 are the STORAGE FORMAT, not the shape.  Fields 9-11 already caught a
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
        raw_live,
        /* the row stride (one unified format; 384) -- with the payload
         * version, refuses any earlier-format file */
        (uint32_t)gpu_graph_attn_comp_cache_row_bytes(),
        (uint32_t)PULSAR_ENGINE_IDXFP4_ROWBYTES,
        (uint32_t)s->prefill_frontier,
    };
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il), err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il), err, errlen) != 0) return 1;
    }

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_raw_row(fp, g, il, phys,
                                       buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
            rc = payload_write_attn_comp_pack(fp, g, il,
                                              gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il),
                                              buf,
                                              PULSAR_SESSION_IO_CHUNK,
                                              err,
                                              errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_index_comp(fp, g, il,
                                          gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il),
                                          buf,
                                          PULSAR_SESSION_IO_CHUNK,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}



int pulsar_session::load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    /* drop speculative lookahead up front, not just on success: a restore
     * that fails midway may already have overwritten GPU state the carry
     * and pendings were conditioned on */
    s->spec.spec_carry_valid = false;
    pulsar_spec_drop_pendings(&s->spec);
    spec_quench_reset(s);
    uint64_t remaining = payload_bytes;
    uint32_t h[PULSAR_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != PULSAR_SESSION_PAYLOAD_MAGIC || h[1] != PULSAR_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_prefill_frontier = h[15];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
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
    if (h[13] != (uint32_t)gpu_graph_attn_comp_cache_row_bytes() ||
        h[14] != (uint32_t)PULSAR_ENGINE_IDXFP4_ROWBYTES)
    {
        payload_set_err(err, errlen,
                        "KV checkpoint row strides differ from this build "
                        "(attn/indexer storage format changed)");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_prefill_frontier > saved_tokens) {
        payload_set_err(err, errlen, "KV checkpoint's prefill frontier lies outside its token count");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[PULSAR_MAX_LAYER];
    uint32_t n_index_comp[PULSAR_MAX_LAYER];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
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
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_raw_row(fp, g, il, phys,
                                      buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
            rc = payload_read_attn_comp_pack(fp, g, il,
                                             n_comp[il],
                                             buf,
                                             PULSAR_SESSION_IO_CHUNK,
                                             &remaining,
                                             err,
                                             errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_index_comp(fp, g, il,
                                         n_index_comp[il],
                                         buf,
                                         PULSAR_SESSION_IO_CHUNK,
                                         &remaining,
                                         err,
                                         errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) = n_comp[il];
        gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il) = n_index_comp[il];
    }
    /* L124: a loaded payload replaces the conversation; the undo ring's
     * entries describe the previous one's stores.  Zero it (the L120
     * projection span restarts itself on the first gap; this ring cannot). */
    s->graph.r128_undo_head = 0u;
    s->graph.r128_undo_n = 0u;
    s->graph.r128_perrow_chunk = false;
    s->prefill_frontier = (int)saved_prefill_frontier;   /* L195: the next sync resumes from the grid point below it */
    s->checkpoint_valid = true;
    /* a restored state invalidates any in-flight speculative lookahead: the
     * carry token, pre-drafted pendings, AND the drafter's context-KV ring
     * were all conditioned on the replaced state. Leaving the ring makes the
     * next drafts (and therefore the verify batch shapes) depend on whatever
     * ran before the restore — the source of run-to-run tie flips. */
    s->spec.spec_carry_valid = false;
    pulsar_spec_drop_pendings(&s->spec);
    spec_quench_reset(s);
    for (int li = 0; li < 3; li++) g->dspark_n_raw[li] = 0;
    g->dspark_prompt_n = 0;
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
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = (uint8_t *)realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
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

/* The tensor-parallel row all-reduce (4g-2 / v14; family-generic since L266): a step's f32 rows
 * [n_rows][n_embd] summed across the group in place, through the row lane (decode / verify widths over
 * RDMA), the bulk lane (prefill chunks over RDMA), or the host big gate (TCP, and any transport without
 * those).  Every lane's combine is the all-reduce's own + peer add in rank order, so the ranks hold
 * the same bytes.  DeepSeek's graph and Qwen's state each lend their transport, slab mappings, stage
 * ticket and exchange seq (pulsar_tp_rows). */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool pulsar_tp_allreduce_rows(const pulsar_tp_rows *g, uint32_t il, uint32_t n_tokens,
                              pulsar_gpu_tensor *t, pulsar_gpu_tensor *addend,
                              const char *what) {
    if (!g->tp) return 1;
    const uint64_t nelt = (uint64_t)n_tokens * g->n_embd;
    const uint64_t bytes = nelt * sizeof(float);
    /* Decode and verify rows on a pair ride the ROW LANE (4g-2): enqueued
     * entirely on the stream -- stage, publish, combine -- while the
     * transport's proxy thread moves the rows; the engine thread never waits.
     * Chosen by row count, like the staging split below.  The combine is the
     * all-reduce's own + peer add, same operands, same order -- bit-identical.
     * The addend is folded by the stage kernel itself (t + addend, the
     * engine's add_kernel arithmetic; addend zeroed), so the fold costs no
     * launch of its own on the decode path. */
    if (n_tokens <= PULSAR_TP_BATCH_MAX_ROWS && pulsar_tp_row_lane(g->tp) &&
        (uint64_t)g->n_embd * sizeof(float) == pulsar_tp_vec_bytes(g->tp)) {
        pulsar_tp_row_lane_layout_t L;
        pulsar_tp_row_lane_layout(g->tp, &L);
        uint8_t *slab = (uint8_t *)g->slab_dev;
        uint64_t first = 0, exch = 0;
        bool ok = slab && g->ticket &&
                  pulsar_tp_row_lane_begin(g->tp, n_tokens, &first, &exch) != 0;
        if (ok) ok = pulsar_gpu_tp_stage_publish(t, addend, slab, L.out_off, L.vec_bytes,
                                                 first, L.n_slots, n_tokens, slab + L.desc_off,
                                                 exch, g->ticket) != 0;
        if (ok) ok = pulsar_gpu_tp_combine_sum(t, slab, L.in_off, L.vec_bytes,
                                               first, L.n_slots, n_tokens, slab + L.done_off,
                                               exch, slab + L.err_off, L.timeout_ns) != 0;
        if (!ok) fprintf(stderr, "pulsar: tp row lane: layer %u %s exchange refused (%u rows)\n", il, what, n_tokens);
        return ok;
    }
    /* Prefill-sized exchanges on a pair ride the BULK LANE (v14): the same
     * stream-enqueued stage / publish / combine, but the rows go GPU-direct
     * through the registered bulk buffer and cross as RDMA writes on the
     * second QP -- no host copy, no handshake.  A payload larger than one
     * bulk buffer is cut into buffer-sized row runs, each its own exchange.
     * The combine is the all-reduce's own + peer add -- bit-identical to the
     * host big gate below, which remains the lane for transports with no bulk
     * buffer (TCP).  The stage kernel folds the addend on its way into the
     * out-region (t + addend, addend zeroed: the row lane's stage arithmetic)
     * and the combine reads this rank's partial back from there, so t is
     * written once, by the combine (L260: the copy-engine stage and the
     * separate add + fill passes cost ~1.2 ms per 32 MiB exchange). */
    if (n_tokens > PULSAR_TP_BATCH_MAX_ROWS && pulsar_tp_bulk_lane(g->tp) && g->slab_dev && g->bulk_dev) {
        pulsar_tp_row_lane_layout_t L;
        pulsar_tp_row_lane_layout(g->tp, &L);
        pulsar_tp_bulk_layout_t BL;
        pulsar_tp_bulk_layout(g->tp, &BL);
        uint8_t *slab = (uint8_t *)g->slab_dev;
        uint8_t *bulk = (uint8_t *)g->bulk_dev;
        const uint64_t row_bytes = (uint64_t)g->n_embd * sizeof(float);
        const uint64_t piece_rows = BL.cap_bytes / row_bytes;
        bool ok = piece_rows > 0;
        for (uint64_t r0 = 0; ok && r0 < n_tokens; r0 += piece_rows) {
            const uint64_t rows = n_tokens - r0 < piece_rows ? n_tokens - r0 : piece_rows;
            const uint64_t off = r0 * row_bytes, len = rows * row_bytes;
            uint64_t exch = 0;
            uint32_t buf = 0;
            ok = pulsar_tp_bulk_begin(g->tp, len, &exch, &buf) != 0 &&
                 pulsar_gpu_tp_bulk_stage(t, addend, off, bulk + BL.out_off, len) != 0 &&
                 pulsar_gpu_tp_publish_bulk(slab + L.desc_off, exch, len,
                                            PULSAR_TP_DESC_BULK_FLAG | (uint64_t)buf) != 0 &&
                 pulsar_gpu_tp_bulk_combine_sum(t, off, bulk + BL.out_off, bulk + BL.in_off[buf], len,
                                                slab + L.done_off,
                                                exch, slab + L.err_off, L.timeout_ns) != 0;
        }
        if (!ok) fprintf(stderr, "pulsar: tp bulk lane: layer %u %s exchange refused (%u rows)\n",
                         il, what, n_tokens);
        return ok;
    }
    /* Two staging shapes, chosen by ROW COUNT because that is the slab's sizing
     * boundary rather than a semantic variant (so nothing selects it by flag):
     *
     *  - <= PULSAR_TP_BATCH_MAX_ROWS (a decode/verify step) stages in the
     *    registered slab's OWN batch region, so the transport rides DIRECT over
     *    RDMA -- its out/in pointers are already inside the slab -- instead of
     *    copying the payload through those same regions to reach registered
     *    memory.
     *  - a bigger prefill chunk does not fit there at all (a 2048-row chunk is
     *    ~33 MB against a batch region of 8 rows) and keeps its own buffer,
     *    which the transport stages through the slab in message-sized pieces.
     * The host gate folds the addend with the same add up front, then zeroes
     * it: byte for byte the values the lanes' stage kernels leave behind. */
    if (addend &&
        (pulsar_gpu_add_tensor(t, t, addend, (uint32_t)nelt) == 0 ||
         pulsar_gpu_tensor_fill_f32(addend, 0.0f, nelt) == 0)) {
        fprintf(stderr, "pulsar: tp: layer %u %s addend fold refused (%u rows)\n", il, what, n_tokens);
        return false;
    }
    float *out = NULL;
    float *in = NULL;
    bool heap = false;
    if (n_tokens <= PULSAR_TP_BATCH_MAX_ROWS) {
        out = (float *)pulsar_tp_slab_batch_out(g->tp, il);
        in  = (float *)pulsar_tp_slab_batch_in(g->tp, il);
        if (!out || !in) {
            fprintf(stderr, "pulsar: tp gate: no slab batch region for layer %u "
                            "(%u rows) -- refusing\n", il, n_tokens);
            return false;
        }
    } else {
        out = (float *)xmalloc(bytes ? bytes : sizeof(float));
        in  = (float *)xmalloc(bytes ? bytes : sizeof(float));
        if (!out || !in) {
            free(out);
            free(in);
            fprintf(stderr, "pulsar: tp prefill big-gate out of memory (%llu bytes)\n",
                    (unsigned long long)bytes);
            return false;
        }
        heap = true;
    }
    bool ok = pulsar_gpu_tensor_read(t, 0, out, bytes) != 0 &&
              pulsar_tp_row_lane_check(g->tp) != 0;   /* stream drained: every row-lane exchange done */
    if (ok) {
        ok = pulsar_tp_allreduce_sum(g->tp, il, ++*g->seq,
                                     out, in, bytes) != 0;
    }
    if (ok) {
        ok = pulsar_gpu_tensor_write(t, 0, out, bytes) != 0;
    }
    if (heap) {
        free(out);
        free(in);
    }
    return ok;
}

/* The tensor-parallel vocab gather (4g-2; family-generic since L266): this rank projects only ITS vocab range
 * (`head(lo, width, dst)` writes the packed slice: row r at r * width), every rank's range is gathered, and
 * the assembled full logits land in `out` [n_rows][n_vocab] in rank order -- every rank ends with the same
 * full rows and samples on its own.  A row-lane pair gathers on the stream; any other transport stages
 * through the host.  The lanes' contract in full is gpu_decode.cpp's tp_vocab_split comment. */
uint64_t pulsar_tp_vocab_own_bytes_for(uint32_t n_ranks, uint64_t vec_bytes, uint32_t n_vocab, uint32_t max_rows) {
    const uint64_t stride = ((uint64_t)n_vocab + n_ranks - 1u) / n_ranks;
    return ((uint64_t)max_rows * stride * sizeof(float) + vec_bytes - 1u) / vec_bytes * vec_bytes;
}

uint64_t pulsar_tp_vocab_own_bytes(pulsar_tp *tp, uint32_t n_vocab, uint32_t max_rows) {
    if (!tp || !pulsar_tp_row_lane(tp)) return 0;
    return pulsar_tp_vocab_own_bytes_for(pulsar_tp_n_ranks(tp), pulsar_tp_vec_bytes(tp), n_vocab, max_rows);
}

bool pulsar_tp_vocab_gather(const pulsar_tp_vocab *g, uint32_t n_rows, const pulsar_tp_head_fn &head,
                            pulsar_gpu_tensor *out) {
    const uint32_t n_vocab = g->n_vocab;
    const uint32_t n_ranks = pulsar_tp_n_ranks(g->tp);
    const uint32_t stride = (n_vocab + n_ranks - 1u) / n_ranks;
    const int rank = pulsar_tp_rank(g->tp);
    uint32_t lo = 0, hi = 0;
    if (!pulsar_tp_owned_range(rank, n_ranks, n_vocab, &lo, &hi)) {
        fprintf(stderr, "pulsar: tp vocab range refused (rank=%d n_ranks=%u n_vocab=%u)\n",
                rank, n_ranks, n_vocab);
        return false;
    }
    if (pulsar_tp_row_lane(g->tp)) {
        uint32_t plo = 0, phi = 0;
        if (!pulsar_tp_owned_range(1 - rank, n_ranks, n_vocab, &plo, &phi) ||
            n_rows == 0 || n_rows > g->max_rows || !g->vocab_own || !g->slab_dev ||
            !g->ticket) {
            fprintf(stderr, "pulsar: tp vocab gather refused (%u rows, scratch %s) -- refusing\n",
                    n_rows, g->vocab_own ? "ok" : "MISSING");
            return false;
        }
        pulsar_tp_row_lane_layout_t L;
        pulsar_tp_row_lane_layout(g->tp, &L);
        uint8_t *slab = (uint8_t *)g->slab_dev;
        const uint64_t vf = L.vec_bytes / sizeof(float);
        const uint32_t msgs = (uint32_t)(((uint64_t)n_rows * stride + vf - 1u) / vf);
        pulsar_gpu_tensor *own = pulsar_gpu_tensor_view(g->vocab_own, 0,
                                                        (uint64_t)n_rows * (hi - lo) * sizeof(float));
        bool ok = own && head(lo, hi - lo, own) &&
                  pulsar_gpu_tp_scatter_cols(out, own, n_rows, hi - lo, n_vocab, lo) != 0;
        pulsar_gpu_tensor_free(own);
        for (uint32_t m0 = 0; ok && m0 < msgs; m0 += PULSAR_TP_BATCH_MAX_ROWS) {
            const uint32_t m = msgs - m0 < PULSAR_TP_BATCH_MAX_ROWS ? msgs - m0 : PULSAR_TP_BATCH_MAX_ROWS;
            uint64_t first = 0, exch = 0;
            pulsar_gpu_tensor *chunk = pulsar_gpu_tensor_view(g->vocab_own, (uint64_t)m0 * L.vec_bytes,
                                                              (uint64_t)m * L.vec_bytes);
            ok = chunk && pulsar_tp_row_lane_begin(g->tp, m, &first, &exch) != 0 &&
                 pulsar_gpu_tp_stage_publish(chunk, NULL, slab, L.out_off, L.vec_bytes, first, L.n_slots, m,
                                             slab + L.desc_off, exch, g->ticket) != 0 &&
                 pulsar_gpu_tp_combine_scatter(out, slab, L.in_off, L.vec_bytes, first, L.n_slots, m,
                                               (uint64_t)m0 * vf, n_rows, phi - plo, n_vocab, plo,
                                               slab + L.done_off, exch, slab + L.err_off,
                                               L.timeout_ns) != 0;
            pulsar_gpu_tensor_free(chunk);
        }
        if (!ok) fprintf(stderr, "pulsar: tp vocab gather: row lane refused (%u rows, %u messages)\n",
                         n_rows, msgs);
        return ok;
    }
    const uint64_t slice_bytes = (uint64_t)n_rows * stride * sizeof(float);
    const uint64_t full_bytes  = (uint64_t)n_rows * n_vocab * sizeof(float);
    pulsar_gpu_tensor *slice = pulsar_gpu_tensor_view(out, 0, slice_bytes);
    if (!slice) {
        fprintf(stderr, "pulsar: tp vocab slice view refused (rows=%u stride=%u)\n",
                n_rows, stride);
        return false;
    }
    float *own     = (float *)xmalloc(slice_bytes);
    float *scratch = (float *)xmalloc(slice_bytes);
    float *full    = (float *)xmalloc(full_bytes);
    bool ok = own && scratch && full;
    if (!ok) fprintf(stderr, "pulsar: tp vocab staging out of memory (%llu + %llu bytes)\n",
                      (unsigned long long)slice_bytes, (unsigned long long)full_bytes);
    /* This rank's range only.  slice 4d inc 1 made the range expressible: the
     * head weight is [vocab, in_dim] row-major, so it is a contiguous row range. */
    if (ok) ok = head(lo, hi - lo, slice);
    /* The head wrote the slice PACKED: row r of the GEMM output starts at
     * r * (hi - lo), because the GEMM's out_dim IS the range width.  The gather
     * wants rows at the padded pitch `stride` (one byte count per exchange
     * round), so the packed rows are read into scratch and RE-PITCHED here --
     * reading them straight into a stride-pitched buffer mis-placed every row
     * after the first on any rank whose range is shorter than the stride (every
     * uneven split), and the tail zero-fill then overwrote the next row's head.
     * n=2 over 129280 is even, which is why the pair never showed it. */
    const uint64_t packed_bytes = (uint64_t)n_rows * (hi - lo) * sizeof(float);
    if (ok) ok = pulsar_gpu_tensor_read(slice, 0, scratch, packed_bytes) != 0;
    if (ok) {
        for (uint32_t r = 0; r < n_rows; r++) {
            memcpy(own + (uint64_t)r * stride,
                   scratch + (uint64_t)r * (hi - lo),
                   (uint64_t)(hi - lo) * sizeof(float));
            memset(own + (uint64_t)r * stride + (hi - lo), 0,
                   (uint64_t)(stride - (hi - lo)) * sizeof(float));
        }
    }
    if (ok) {
        ok = pulsar_tp_allgather_rows(g->tp, PULSAR_TP_NON_LAYER_TAG,
                                       ++*g->seq, full, own, scratch,
                                       n_rows, n_vocab, 1u) != 0;
    }
    if (ok) ok = pulsar_gpu_tensor_write(out, 0, full, full_bytes) != 0;
    pulsar_gpu_tensor_free(slice);
    free(own);
    free(scratch);
    free(full);
    return ok;
}

#include "pulsar_engine_internal.h"






pulsar_gpu_tensor *gpu_graph_tensor_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* L120 value-half: bank the tail (up to 8 positions) of a chunk's staged
 * projections into the ratio-4 ring — the aligned prefill paths' equivalent
 * of the per-row deposit, so a rewind shortly after a continuation still
 * finds its replay span covered. */
static bool gpu_graph_proj_ring_deposit_tail(pulsar_gpu_graph *g, uint32_t il,
                                             uint32_t pos0, uint32_t n_tokens,
                                             uint32_t width, bool indexer) {
    const uint32_t tail = n_tokens < 8u ? n_tokens : 8u;
    bool ok = true;
    for (uint32_t k = 0; ok && k < tail; k++) {
        const uint32_t t = n_tokens - tail + k;
        pulsar_gpu_tensor *kv = gpu_graph_tensor_row_view(g->batch_comp_kv, t, width);
        pulsar_gpu_tensor *sc = gpu_graph_tensor_row_view(g->batch_comp_sc, t, width);
        ok = kv && sc &&
             gpu_graph_proj_ring_deposit(g, il, pos0 + t, kv, sc, indexer);
        pulsar_gpu_tensor_free(sc);
        pulsar_gpu_tensor_free(kv);
    }
    return ok;
}


/* Row view into a Q buffer (L045). Same as gpu_graph_tensor_row_view but
 * strides by PULSAR_Q_ELT_SIZE — use this (not the generic helper) for
 * batch_q and q. The generic one strides by sizeof(float), which against a
 * narrowed buffer lands at double the intended offset: a silent wrong answer,
 * not a fault. Same reasoning as gpu_graph_hc_row_view below. */
pulsar_gpu_tensor *gpu_graph_q_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * PULSAR_Q_ELT_SIZE,
                                 row_values * PULSAR_Q_ELT_SIZE);
}


/* Row view into the attention-output (heads) buffer (L033). Strides by
 * PULSAR_HEADS_ELT_SIZE — use this, not the generic helper, for batch_heads and
 * heads. Third instance of the same hazard after Q and HC: the generic view
 * strides by sizeof(float) and against a narrowed buffer lands at double the
 * intended offset, which compiles clean and computes a wrong answer. */
pulsar_gpu_tensor *gpu_graph_heads_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * PULSAR_HEADS_ELT_SIZE,
                                 row_values * PULSAR_HEADS_ELT_SIZE);
}


/* Row view into an HC residual CARRIER buffer (BF16 storage; task #62). Same as
 * gpu_graph_tensor_row_view but strides by PULSAR_HC_ELT_SIZE, not sizeof(float) —
 * use this (not the generic helper) for cur_hc/next_hc/after_*_hc bases. */
pulsar_gpu_tensor *gpu_graph_hc_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * PULSAR_HC_ELT_SIZE,
                                 row_values * PULSAR_HC_ELT_SIZE);
}



/* Upload prompt token ids for kernels that need token-aware hash routing. */
bool gpu_graph_upload_prompt_tokens(
        pulsar_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = (int32_t *)xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) {
        tokens[i] = prompt->v[pos0 + i];
        /* L188: the embed kernel clamps a negative id to 0 -- refuse it here */
        if (tokens[i] < 0 || tokens[i] >= (int32_t)PULSAR_N_VOCAB) {
            fprintf(stderr, "pulsar: prefill token %d at position %u is not a vocab id -- refusing\n",
                    tokens[i], pos0 + i);
            free(tokens);
            return false;
        }
    }

    const bool ok = pulsar_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}



/* Rebuild the ratio-4 compressor state from a chunk's tail: the last COMPLETE
 * group of four (when the chunk has one) plus the partial group of
 * rem = n_tokens % 4 rows, taken from the chunk's OWN kv / score projections
 * (batch_comp_kv/_sc, the prefill arm).  pulsar_gpu_compressor_prefill_state_ratio4_tensor
 * lays them out as the decode store does (rows 0..3 complete, 4 + phase
 * partial).  Until L168 the tail was the last four rows written at 0..3
 * regardless of alignment, which dropped the partial group for every whole
 * prompt with n_tokens % 4 != 0.  Until L183 the tail was RE-PROJECTED on the
 * decode arm (the nt kernels, 4..7 rows) in the name of parity with the decode
 * store; that arm gave different bytes at 4 and at 5 rows for the plain
 * weights, so the state depended on the chunk's remainder and no parity was
 * had.  The prefill arm is M-neutral since L183: the rows a chunk projected
 * are the rows any chunking would have projected, and the state copies them.
 * Chunk starts are ratio-aligned at every caller (chunk multiples, or 0); an
 * unaligned start is refused, not laid out by the wrong phase. */
static bool gpu_graph_refresh_ratio4_compressor_state(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        pulsar_gpu_tensor *state_kv,
        pulsar_gpu_tensor *state_score,
        const pulsar_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens == 0u) {
        fprintf(stderr, "pulsar: ratio-4 compressor state rebuild on an empty chunk at pos0=%u -- refusing\n", pos0);
        return false;
    }
    if ((pos0 % 4u) != 0u) {
        fprintf(stderr, "pulsar: ratio-4 compressor state rebuild: chunk start %u is not ratio-aligned "
                        "(n_tokens=%u) -- refusing\n", pos0, n_tokens);
        return false;
    }
    if (!g || !model || !state_kv || !state_score || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }
    const uint32_t rem = n_tokens % 4u;
    const uint32_t n_full = n_tokens >= 4u ? 4u : 0u;
    const uint32_t n_tail = n_full + rem;

    /* The tail rows are copied out of batch_comp_kv/_sc into comp_tail_kv/_sc
     * rather than read in place: gpu_graph_proj_ring_deposit_tail reads that
     * buffer's last eight rows AFTER this rebuild, and the rebuild kernel must
     * not write there (L171: writing into batch_comp_kv rows 0..n_tail-1
     * handed the ring tail tokens under head positions for every 5..11-token
     * chunk).  Stream-ordered async copies: the consumer is the rebuild kernel
     * on the same stream. */
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t src_off = (uint64_t)(n_tokens - n_tail) * row_bytes;
    bool ok = pulsar_gpu_tensor_copy_async(g->comp_tail_kv, 0, g->batch_comp_kv, src_off, (uint64_t)n_tail * row_bytes) != 0 &&
              pulsar_gpu_tensor_copy_async(g->comp_tail_sc, 0, g->batch_comp_sc, src_off, (uint64_t)n_tail * row_bytes) != 0;
    if (ok) {
        ok = pulsar_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->comp_tail_kv,
                                                              g->comp_tail_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - n_tail,
                                                              n_full,
                                                              rem) != 0;
    }
    return ok;
}



/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding, gathered on the device from the uploaded token ids
 * (pulsar_gpu_embed_tokens_hc_tensor) at every n_tokens.  The gather is an
 * exact copy -- each bf16 embedding row into the bf16 HC carrier -- so no
 * numerics live here.  The host build-and-upload twin that served n_tokens <
 * 512 until L167 was a second implementation of the same copy (rule 1) and is
 * deleted with its helpers embed_token_f16 and pulsar_store_hc_carrier_f32. */
bool gpu_graph_upload_prompt_embeddings_hc(
        pulsar_gpu_tensor   *out_hc,
        pulsar_gpu_tensor   *tokens,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    if (!tokens) {
        fprintf(stderr, "pulsar: prompt embedding gather needs the device token tensor "
                        "(gpu_graph_upload_prompt_tokens first) -- refusing\n");
        return false;
    }
    return pulsar_gpu_embed_tokens_hc_tensor(out_hc,
                                            tokens,
                                            model->map,
                                            model->size,
                                            weights->token_embd->abs_offset,
                                            (uint32_t)weights->token_embd->dim[1],
                                            n_tokens,
                                            PULSAR_N_EMBD,
                                            PULSAR_N_HC) != 0;
}



bool gpu_graph_warmup_prefill_kernels(
        pulsar_gpu_graph   *g,
        const pulsar_model   *model,
        const pulsar_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed) return true;

    /*
     * The first batched F16 matmul can pay GPU's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;

    bool ok = pulsar_gpu_begin_commands() != 0;
    /* L159: the bf16 core reads the norm's bf16 plane and refuses without one,
     * so the warmup runs the same producer the graph runs, on the same buffers
     * (their contents are scratch here; the output is overwritten). */
    void *warm_b = NULL;
    if (ok) ok = pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_tokens, hc_dim, &warm_b) != 0;
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, warm_b, g->batch_cur_hc,
                                                      (uint32_t)hc_dim, n_tokens, PULSAR_RMS_EPS, 0) != 0;
    if (ok) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_tokens, hc_dim);
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_hc_mix,
                                             model,
                                             weights->layer[0].hc_attn_fn,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "pulsar: GPU prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}
















/** Operand set for one indexed-attention span: indexer score -> top-k ->
 * indexed attention over rows [s0, s0+sn) of the batch.
 *
 * Exists to collapse two loops into one body. The chunked and zero-prefix
 * paths ran identical logic over different operands, so the CALLER resolves
 * every multiseq difference into this struct and the span code stays single-
 * form. `mseq` is the one exception -- it controls only whether per-span
 * descriptor views get built.
 */
struct gpu_graph_span_ops {
    pulsar_gpu_tensor       *comp_src;  ///< attention comp-cache operand
    pulsar_gpu_tensor       *raw_src;  ///< raw KV cache operand
    pulsar_gpu_tensor       *index_src;  ///< indexer comp-cache operand
    const pulsar_gpu_tensor *index_bases;  ///< per-bank base table, or NULL
    const pulsar_gpu_tensor *comp_bases;  ///< per-bank base table, or NULL
    uint32_t                 comp_cap;  ///< per-bank stride, 0 when scalar
    uint32_t                 n_banks;  ///< 1 when scalar
    bool                     mseq;          ///< build per-span descriptor views for the banked path
};

static bool gpu_graph_indexed_attention_span(
        pulsar_gpu_graph           *g,
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        uint32_t                    il,
        uint32_t                    s0,
        uint32_t                    sn,
        uint32_t                    spos0,
        uint64_t                    q_dim,
        uint32_t                    n_comp,
        uint32_t                    ratio,
        float                       index_scale,
        uint32_t                    n_raw,
        uint32_t                    raw_start,
        const struct gpu_graph_span_ops *op) {
    /* Q is the producer's PACKED rows now (L090.4): per-token stride is
     * n_head packed rows, in BYTES -- the element-size trap the sq_view
     * comment below warns about, avoided by construction. */
    pulsar_gpu_tensor *iq_view = pulsar_gpu_tensor_view(g->batch_indexer_qp,
            (uint64_t)s0 * PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES,
            (uint64_t)sn * PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES);
    pulsar_gpu_tensor *iw_view = pulsar_gpu_tensor_view(g->batch_indexer_weights,
            (uint64_t)s0 * PULSAR_N_INDEXER_HEAD * sizeof(float),
            (uint64_t)sn * PULSAR_N_INDEXER_HEAD * sizeof(float));
    /* BYTE offset and BYTE length -- must follow the Q element size, or a
     * narrowed buffer is viewed at double the intended offset. */
    pulsar_gpu_tensor *sq_view = pulsar_gpu_tensor_view(g->batch_q,
            (uint64_t)s0 * q_dim * PULSAR_Q_ELT_SIZE,
            (uint64_t)sn * q_dim * PULSAR_Q_ELT_SIZE);
    pulsar_gpu_tensor *sh_view = pulsar_gpu_tensor_view(g->batch_heads,
            (uint64_t)s0 * q_dim * PULSAR_HEADS_ELT_SIZE,
            (uint64_t)sn * q_dim * PULSAR_HEADS_ELT_SIZE);
    /* Multiseq: per-span descriptor views (rows s0..s0+sn).  The scalar raw
     * span/start are ignored in banked mode -- they are derived per row. */
    pulsar_gpu_tensor *sp_view = op->mseq
        ? pulsar_gpu_tensor_view(g->batch_positions,
                              (uint64_t)s0 * sizeof(int32_t),
                              (uint64_t)sn * sizeof(int32_t))
        : NULL;
    pulsar_gpu_tensor *ss_view = op->mseq
        ? pulsar_gpu_tensor_view(g->batch_seq_id,
                              (uint64_t)s0 * sizeof(int32_t),
                              (uint64_t)sn * sizeof(int32_t))
        : NULL;
    bool ok = iq_view && iw_view && sq_view && sh_view &&
              (!op->mseq || (sp_view && ss_view));

    /* L121: a banked multi-row span is scored per bank run through the
     * block-scaled MXFP4 tier against the bank's own comp slab (the generic
     * per-(comp,row) descriptor kernel was rows x depth-linear: 26 ms/layer
     * at depth 24.5k vs the 0.2 ms indexed attention it feeds).  Each run is
     * shape-identical to the classic non-banked case because
     * gpu_graph_multiseq_step_begin admits no batch whose per-bank rows are
     * not one contiguous run of consecutive positions (gpu_diag.cpp: the "not
     * contiguous" / "not consecutive within its run" rejections).  The run
     * shape is re-checked here and a violation REFUSES.  Until L167 it sent
     * the whole span to the generic descriptor kernel instead -- a different
     * accumulation order selected by batch layout, unreachable behind
     * step_begin's check and measured by nothing.  Single-row spans take the
     * direct-one tier (bit-identical to classic single-token decode). */
    bool span_runs_conform = true;
    for (uint32_t t = 1; op->mseq && span_runs_conform && t < sn; t++) {
        const uint32_t a = s0 + t;
        if (g->ms_seq_id[a] == g->ms_seq_id[a - 1u] &&
            g->ms_positions[a] != g->ms_positions[a - 1u] + 1)
            span_runs_conform = false;
    }
    if (ok && !span_runs_conform) {
        fprintf(stderr, "pulsar: indexer span at layer %u: a bank's %u rows are not consecutive "
                        "positions (step_begin admits no such batch) -- refusing\n", il, sn);
        ok = false;
    }
    /* Every banked span, one-row spans included, scores each bank run through
     * the tier in scalar mode (L173: a one-row descriptor launch used to reach
     * the deleted SIMT kernel; a one-row run here is the same arithmetic as
     * the classic one-row decode, which is what solo-vs-banked comparisons
     * depend on). */
    if (ok && op->mseq) {
        for (uint32_t r0 = 0; ok && r0 < sn; ) {
            uint32_t rn = 1;
            while (r0 + rn < sn &&
                   g->ms_seq_id[s0 + r0 + rn] == g->ms_seq_id[s0 + r0]) rn++;
            const uint32_t bank  = (uint32_t)g->ms_seq_id[s0 + r0];
            const uint32_t rpos0 = (uint32_t)g->ms_positions[s0 + r0];
            pulsar_gpu_tensor *rq = pulsar_gpu_tensor_view(g->batch_indexer_qp,
                    (uint64_t)(s0 + r0) * PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES,
                    (uint64_t)rn * PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES);
            pulsar_gpu_tensor *rw = pulsar_gpu_tensor_view(g->batch_indexer_weights,
                    (uint64_t)(s0 + r0) * PULSAR_N_INDEXER_HEAD * sizeof(float),
                    (uint64_t)rn * PULSAR_N_INDEXER_HEAD * sizeof(float));
            pulsar_gpu_tensor *rs = pulsar_gpu_tensor_view(g->indexer_scores,
                    (uint64_t)r0 * n_comp * sizeof(float),
                    (uint64_t)rn * n_comp * sizeof(float));
            pulsar_gpu_tensor *rb = gpu_graph_bank_index_comp_view(g, il, bank);
            ok = rq && rw && rs && rb &&
                 pulsar_gpu_indexer_scores_decode_run_tensor(rs, rq, rw, rb,
                        n_comp, rn, rpos0,
                        PULSAR_N_INDEXER_HEAD,
                        PULSAR_N_INDEXER_HEAD_DIM,
                        ratio, index_scale) != 0;
            pulsar_gpu_tensor_free(rb);
            pulsar_gpu_tensor_free(rs);
            pulsar_gpu_tensor_free(rw);
            pulsar_gpu_tensor_free(rq);
            r0 += rn;
        }
    } else if (ok) {
        ok = pulsar_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                          iq_view,
                                                          iw_view,
                                                          op->index_src,
                                                          n_comp,
                                                          sn,
                                                          spos0,
                                                          PULSAR_N_INDEXER_HEAD,
                                                          PULSAR_N_INDEXER_HEAD_DIM,
                                                          ratio,
                                                          index_scale) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("indexer_scores", g->indexer_scores,
                                      (uint64_t)n_comp * sn, il, spos0);
    }
    if (ok) {
        ok = pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                           g->indexer_scores,
                                           n_comp,
                                           sn,
                                           PULSAR_N_INDEXER_TOP_K) != 0;
        if (ok) {
            gpu_graph_debug_dump_i32_tensor("indexer_topk", g->comp_selected,
                                              (uint64_t)sn * PULSAR_N_INDEXER_TOP_K, il, spos0);
        }
    }
    if (ok) {
        ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(sh_view,
                                                                  model->map,
                                                                  model->size,
                                                                  layer->attn_sinks->abs_offset,
                                                                  sq_view,
                                                                  op->raw_src,
                                                                  op->comp_src,
                                                                  g->comp_selected,
                                                                  sn,
                                                                  spos0,
                                                                  n_raw,
                                                                  g->raw_cap,
                                                                  raw_start,
                                                                  n_comp,
                                                                  PULSAR_N_INDEXER_TOP_K,
                                                                  g->raw_window,
                                                                  ratio,
                                                                  PULSAR_N_HEAD,
                                                                  PULSAR_N_HEAD_DIM,
                                                                  sp_view, ss_view,
                                                                  op->comp_bases,
                                                                  op->comp_cap,
                                                                  op->n_banks,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
    }
    pulsar_gpu_tensor_free(ss_view);
    pulsar_gpu_tensor_free(sp_view);
    pulsar_gpu_tensor_free(sh_view);
    pulsar_gpu_tensor_free(sq_view);
    pulsar_gpu_tensor_free(iw_view);
    pulsar_gpu_tensor_free(iq_view);
    return ok;
}


bool gpu_graph_encode_layer_attention_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint32_t n_groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    /* Grouped E4M3 for the attn-output "a" projection, emitted by the fp16
     * attention epilogue (head dims [0, n_nope)) and rope_tail (the rest).
     * Declared here because the two producers sit in different scopes below
     * and the note() must see both.  Left NULL unless every precondition
     * holds -- see the eligibility comment at the attention call. */
    void    *gact_data = NULL, *gact_scale = NULL;
    int      gact_kbp = 0, gact_emitted = 0;
    uint64_t gact_slab = 0;
    /* Banked multiseq step (Tier-2): rows are independent sessions — per-row
     * position/bank from the host mirrors (gpu_graph_multiseq_step_begin),
     * per-bank compressor frontiers, banked kernel operands (whole pool +
     * device descriptor arrays), scalar counters = read-only supersets.
     * step_begin rejects position-0 rows, so zero_prefix is never multiseq. */
    const bool mseq = g->batch_multiseq;
    /* Single-sequence prefill has been dequantising the packed comp
     * cache into an f32 shadow and reading that: 2048 B/row instead of 584, on
     * the rows that dominate the attention tile, plus a whole dequant pass.
     * Every prefill attention consumer reads PULSAR_ATTN_PACK rows as of
     * 2026-08-18, so there is nothing left to ask the backend about and no
     * shadow to choose: the packed pool goes straight to all of them.
     * Bit-exact by construction -- packed rows decode to exactly the values the
     * f32 cache would hold. */
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    if (mseq && (pos0 == 0 || n_tokens > g->batch_multiseq_rows ||
                 (uint32_t)g->ms_positions[0] != pos0)) {
        fprintf(stderr, "pulsar: multiseq layer batch rejected: rows/pos0 do not "
                        "match the armed step (pos0=%u n_tokens=%u rows=%u)\n",
                pos0, n_tokens, g->batch_multiseq_rows);
        return false;
    }
    const bool zero_prefix = pos0 == 0;
    /* ⚠ DISARM THE GROUPED-ACTIVATION CACHE FIRST, EVERY LAYER.
     * g_gact is a single-entry cache keyed on (batch_heads->ptr, n_tokens,
     * n_groups, group_dim).  Every layer of a prefill shares that pointer and
     * those dims, so the key is IDENTICAL across all 43 layers: a layer that
     * reaches attention through an arm which never re-slots the cache would
     * otherwise inherit the previous layer's `valid` and hand the "a" GEMM
     * layer N-1's activations -- the [[L035]] / C1 stale-cache failure, a hit
     * that is well-formed, current-shaped, and wrong.
     *
     * This lived inside the per-token fallback until 2026-08-15, so only that
     * arm was protected.  It was reachable: a 62-token prefill gives ratio-128
     * layers n_comp == 0, no arm claims the batch, the fallback runs and notes
     * the cache, and the next ratio-4 layer takes the static-mixed arm and
     * skips the disarm.  The n_tokens >= 128 floor on the raw launcher was
     * suppressing it by keeping fp16 -- the only producer that sets *mx_out --
     * off that path at small batch.  Caught by multiseq_frontier_gate S1
     * populate-order invariance. */
    pulsar_gpu_mxfp8_gact_disarm();
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? (uint32_t *)xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? (uint32_t *)xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    pulsar_gpu_tensor *hc_mix_view = pulsar_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *hc_split_view = pulsar_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    /* batch_attn_cur is dump-only and NULL when dumps are off (L090.1); the
     * view must follow, and the ok-chain must not treat that NULL as failure. */
    pulsar_gpu_tensor *attn_cur_view = g->batch_attn_cur
            ? pulsar_gpu_tensor_view(g->batch_attn_cur, 0,
                                  (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float))
            : NULL;
    pulsar_gpu_tensor *after_attn_hc_view = pulsar_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * PULSAR_HC_ELT_SIZE);  ///< carrier
    bool ok = hc_mix_view && hc_split_view && after_attn_hc_view &&
              (attn_cur_view || !g->batch_attn_cur);
    /* The f16 activation slot and its flat_hc_skip_f32 companion are gone with
     * the last F16 weight (2026-08-16).  They existed so an F16 mix GEMM could
     * read a 2-byte activation and the widest f32 store in the layer could be
     * skipped; with the mix weight at F32 that skip is impossible by
     * construction -- cublasSgemm needs exactly the store it was skipping. */
    void *attn_norm_q = NULL, *attn_norm_sf = NULL; int attn_norm_kbp = 0;
    void *attn_norm_b = NULL;
    uint32_t attn_norm_keep_from = 0u;
    /* batch_flat_hc's reader is the "plain F32" GEMM below, which since
     * L079/L087 is not f32 at all: pulsar_gpu_matmul_f32_tensor resolves a bf16
     * copy of the weight and runs the SHARED bf16 core, because these weights'
     * source math is bf16.  So this buffer has a BF16 consumer, and without a
     * producer-side copy that consumer converted hc_dim floats per layer per
     * step -- the T3 census named it as in_dim=16384, cover=0, at every n_tok
     * the run produced.  Emit from the epilogue instead: the value is already
     * in a register and already scaled, so this deletes the convert rather
     * than moving it.
     *
     * Still NO e4m3 arm() here -- that half of the old note stands.  There is
     * no MXFP8 consumer of this buffer, so an E4M3 slot would claim one of the
     * six and reset validity bits nobody reads. */
    void *flat_hc_b = NULL;
    if (ok && !pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_tokens,
                                        (uint64_t)hc_dim, &flat_hc_b)) {
        fprintf(stderr, "pulsar: flat_hc: no bf16 slot (n_tok=%u hc_dim=%u) -- refusing (L159)\n",
                n_tokens, (uint32_t)hc_dim);
        ok = false;
    }
    /* L157: the f32 rows of batch_flat_hc are a dead store when the bf16 copy
     * exists -- their only consumer is hc_attn_fn's GEMM, which runs the shared
     * bf16 core (L079/L087) and reads the slot.  d967327 (2026-08-14) skipped
     * exactly this store for a measured, bit-exact +4.0% prefill; the F32
     * detour of 08-16 made it impossible (cuBLAS SGEMM needed the f32) and it
     * was deleted with the F16 sweep; the bf16-core migration made it legal
     * again and nobody put it back.  Same predicate family as attn_norm's skip:
     * the mixed-batch split's offset views read f32 and key no slot, and dumps
     * read f32.  Declared to the cache so a slot miss refuses, never converts
     * unwritten bytes. */
    const bool flat_skip_f32 = flat_hc_b &&
                               pulsar_gpu_matmul_batch_decode_rows() == 0 &&
                               !gpu_graph_f32_store_observed_any();
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      flat_hc_b,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS,
                                                      flat_skip_f32 ? 1 : 0) != 0;
    /* note() only after the kernel SUCCEEDED -- validity must not outlive a
     * failed launch, or the consumer reads a slot that was never written. */
    if (ok && flat_hc_b) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_tokens,
                                                  (uint64_t)hc_dim);
    if (ok && flat_skip_f32) {
        pulsar_gpu_act_note_f32_skipped_for(g->batch_flat_hc, n_tokens, (uint64_t)hc_dim, 0u);
        static int announced_fhs = 0;
        if (!announced_fhs) {
            announced_fhs = 1;
            fprintf(stderr, "pulsar: flat_hc f32 store SKIPPED (n_tok=%u, %.1f MiB/layer x2)\n",
                    n_tokens, (double)n_tokens * hc_dim * sizeof(float) / (1024.0 * 1024.0));
        }
    }
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_attn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    /* L183 census taps: the carrier the layer starts from, the mix GEMM's
     * rows, and (below) the split coefficients -- the three per-row inputs of
     * hc_attn_post that had no dump. */
    if (ok) gpu_graph_debug_dump_hc_tensor("hc_cur", g->batch_cur_hc, (uint64_t)n_tokens * hc_dim, il, pos0);
    if (ok) gpu_graph_debug_dump_tensor("hc_mix_raw", hc_mix_view, (uint64_t)n_tokens * mix_hc, il, pos0);
    {
        /* ...and the E4M3 encoding too: batch_attn_norm feeds seven MXFP8
         * projections, every one of which would otherwise wait on a separate
         * quantize pass over the whole tensor. */
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_attn_norm, n_tokens, PULSAR_N_EMBD,
                                                        &attn_norm_q, &attn_norm_sf,
                                                        &attn_norm_kbp)) {
            fprintf(stderr, "pulsar: attn_norm: no E4M3 slot (n_tok=%u in_dim=%u) -- refusing (L189)\n",
                    n_tokens, (unsigned)PULSAR_N_EMBD);
            ok = false;
        }
        /* ...and the bf16 copy: batch_attn_norm also feeds the BF16-weight
         * compressors and indexer_proj, which staged their own convert. */
        if (ok && !pulsar_gpu_bf16_act_slot(g->batch_attn_norm, n_tokens, PULSAR_N_EMBD,
                                            &attn_norm_b)) {
            fprintf(stderr, "pulsar: attn_norm: no bf16 slot -- refusing (L159)\n");
            ok = false;
        }
        /* Every consumer of batch_attn_norm is a GEMM reading the E4M3 or the
         * bf16 plane -- the ratio-4 compressor's last-four-rows view included:
         * it goes through the bf16 core, which finds the plane's window by row
         * offset (act_slot_find_window), so the f32 bytes under the view are
         * never read.  The rows are stored only when a dump wants them, or
         * when the mixed-batch split is armed (its offset views key no slot). */
        attn_norm_keep_from = 0u;
        if (attn_norm_q && attn_norm_b &&
            pulsar_gpu_matmul_batch_decode_rows() == 0 &&
            !gpu_graph_f32_store_observed_any()) {
            attn_norm_keep_from = n_tokens;
            static int announced_ans = 0;
            if (!announced_ans) {
                announced_ans = 1;
                fprintf(stderr, "pulsar: attn_norm f32 store SKIPPED (n_tok=%u, %.1f MiB/layer)\n",
                        n_tokens,
                        (double)n_tokens * PULSAR_N_EMBD * sizeof(float) / (1024.0 * 1024.0));
            }
        }
        /* The pre-norm carrier is a dead store unless a dump wants it -- see
         * the kernel's `out` note. */
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
                                                                 gpu_graph_f32_store_observed("hc_attn_pre", il, pos0)
                                                                     ? attn_cur_view : NULL,
                                                                 g->batch_attn_norm,
                                                                 attn_norm_q,
                                                                 attn_norm_sf,
                                                                 attn_norm_kbp,
                                                                 attn_norm_b,
                                                                 attn_norm_keep_from,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_cur_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_attn_scale->abs_offset,
                                                                 layer->hc_attn_base->abs_offset,
                                                                 layer->attn_norm->abs_offset,
                                                                 n_tokens,
                                                                 PULSAR_N_EMBD,
                                                                 PULSAR_N_HC,
                                                                 PULSAR_N_HC_SINKHORN_ITER,
                                                                 PULSAR_HC_EPS,
                                                                 PULSAR_RMS_EPS,
        layer->attn_norm->type == PULSAR_TENSOR_BF16) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    /* batch_attn_norm is now final for this layer and feeds up to seven MXFP8
     * projections below (q_a, kv, attn compressor kv+gate, indexer compressor
     * kv+gate, indexer_proj), each of which would otherwise re-quantize the
     * identical [n_tokens x n_embd] f32 tensor.  Arm the quantize-once cache
     * here -- immediately after the ONLY writes to this buffer (the fused-norm
     * and standalone-norm branches above), which is what keeps a later cache
     * hit coherent -- and disarm at the single exit below. */
    if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_attn_norm, n_tokens, PULSAR_N_EMBD);
    /* arm() invalidates both encodings; the f16 one the hc-fused norm already
     * wrote into the cache's slot IS current for this exact tensor. */
    if (ok && attn_norm_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    if (ok && attn_norm_b) pulsar_gpu_bf16_act_note(g->batch_attn_norm, n_tokens, PULSAR_N_EMBD);
    if (ok && attn_norm_keep_from) pulsar_gpu_mxfp8_act_cache_note_f32_skipped(attn_norm_keep_from);
    if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      g->batch_qr,
                                                      model,
                                                      layer->attn_q_a,
                                                      PULSAR_N_EMBD,
                                                      q_rank,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          PULSAR_N_EMBD,
                                                          PULSAR_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
        }
        /* batch_qr_norm feeds the MXFP8 attn_q_b (and the indexer's q_b) as
         * E4M3, so the norm emits that encoding itself instead of leaving a
         * whole-tensor quantize pass for the GEMM to wait on.  This is the
         * second live E4M3 buffer in the layer -- batch_attn_norm is still
         * armed -- which is why the activation cache had to grow per-buffer
         * slots first (647a606); with one slot this arm would have silently
         * invalidated batch_attn_norm and its later consumers would have
         * re-quantized from f32. */
        void *qr_norm_q = NULL, *qr_norm_sf = NULL; int qr_norm_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_qr_norm, n_tokens,
                                                        (uint64_t)q_rank,
                                                        &qr_norm_q, &qr_norm_sf,
                                                        &qr_norm_kbp)) {
            fprintf(stderr, "pulsar: qr_norm: no E4M3 slot (n_tok=%u in_dim=%u) -- refusing (L189)\n",
                    n_tokens, (unsigned)q_rank);
            ok = false;
        }
        /* DEAD-STORE ELIMINATION.  Both readers of batch_qr_norm are MXFP8
         * GEMMs (attn_q_b below, and the indexer's q_b) and both take the
         * E4M3 the norm emits, so the f32 store has no arithmetic consumer.
         *
         * It does have a DIAGNOSTIC one: the "q_lora_norm" dump just below
         * reads the buffer as f32.  A dump that silently shows bytes from a
         * previous call is worse than no dump, so the skip yields to it --
         * checked with the same debug predicate the dump itself uses, so the
         * two cannot disagree.  Unlike batch_attn_norm there is no offset
         * VIEW of this buffer anywhere (checked), which is what makes it
         * eliminable at all. */
        /* Same mixed-batch condition as the shared_mid skip below -- see the
         * comment there for why the cache-lookup invariant does not cover the
         * prefix split. */
        const bool qr_skip_f32 = (qr_norm_q != NULL) &&
                                 pulsar_gpu_matmul_batch_decode_rows() == 0 &&
                                 !gpu_graph_f32_store_observed("q_lora_norm", il, pos0);
        if (ok) ok = pulsar_gpu_dsv4_qkv_rms_norm_rows_mx_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             PULSAR_N_HEAD_DIM,
                                                             n_tokens,
                                                             PULSAR_RMS_EPS,
                                                             qr_norm_q,
                                                             qr_norm_sf,
                                                             qr_norm_kbp,
        layer->attn_q_a_norm->type == PULSAR_TENSOR_BF16, layer->attn_kv_a_norm->type == PULSAR_TENSOR_BF16,
                                                             qr_skip_f32) != 0;
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_qr_norm, n_tokens, (uint64_t)q_rank);
        if (ok && qr_norm_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
        if (ok && qr_skip_f32) pulsar_gpu_mxfp8_act_cache_note_f32_skipped(n_tokens);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_b",
                                                          il,
                                                          pos0,
                                                          g->batch_q,
                                                          model,
                                                          layer->attn_q_b,
                                                          q_rank,
                                                          q_dim,
                                                          g->batch_qr_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_q_tensor("Qraw", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        /* WHERE the Q head-norm + tail rope runs, never WHICH attention kernel.
         * Shipped: deferred into the fp16 attention kernel's Q-fragment build
         * (q_prep), so batch_q stays RAW and the normed+roped Q exists only in
         * the kernel's registers.  A "Qcur" dump needs that intermediate in
         * memory, so it runs the standalone head_rms_norm_rope_tail kernel
         * first and hands attention pre-normed Q (q_prep NULL).  The two are
         * bit-exact (shared rope core, replicated reduction -- attn_f16.cu),
         * and the attention launch is the same fp16 kernel either way (L166:
         * there is no other attention kernel; a device without the tier is
         * refused by the attention launch, not routed elsewhere). */
        const bool prefill_q_defer = !gpu_graph_f32_store_observed("Qcur", il, pos0);
        g->q_prep_active = 0;
        bool prefill_q_norm_rope_fused = false;
        if (ok && prefill_q_defer) {
            memset(&g->q_prep, 0, sizeof g->q_prep);
            g->q_prep.eps = PULSAR_RMS_EPS;
            g->q_prep.n_rot = PULSAR_N_ROT;
            g->q_prep.n_ctx_orig = compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0;
            g->q_prep.freq_base = freq_base;
            g->q_prep.freq_scale = freq_scale;
            g->q_prep.ext_factor = ext_factor;
            g->q_prep.attn_factor = attn_factor;
            g->q_prep.beta_fast = PULSAR_ROPE_YARN_BETA_FAST;
            g->q_prep.beta_slow = PULSAR_ROPE_YARN_BETA_SLOW;
            g->q_prep_active = 1;
            prefill_q_norm_rope_fused = true;  ///< deferred into attention
        } else if (ok) {
            prefill_q_norm_rope_fused =
                pulsar_gpu_head_rms_norm_rope_tail_tensor(g->batch_q,
                                                       n_tokens,
                                                       PULSAR_N_HEAD,
                                                       PULSAR_N_HEAD_DIM,
                                                       PULSAR_N_ROT,
                                                       pos0,
                                                       compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                       false,
                                                       freq_base,
                                                       freq_scale,
                                                       ext_factor,
                                                       attn_factor,
                                                       PULSAR_ROPE_YARN_BETA_FAST,
                                                       PULSAR_ROPE_YARN_BETA_SLOW,
                                                       PULSAR_RMS_EPS,
                                                       mseq ? g->batch_positions : NULL) != 0;
        }
        /* The separate head-norm + rope-tail pair that used to live here was
         * reachable ONLY by asking for a "Qnorm" dump: the fused kernel never
         * materialises that intermediate, so the dump request forced a
         * different pair of kernels and the dumped numbers were, by the file's
         * own warning, not the numbers production computes.  A debug
         * affordance that changes what it observes cannot diagnose what it
         * observes, so it is gone along with the "Qnorm" dump.  Prefill Q now
         * has exactly two places to be normed: inside attention (shipped) or
         * by the standalone kernel above (the "Qcur" dump); same numbers,
         * same attention kernel after it.  L045 stage 2.
         *
         * If the post-norm/pre-rope intermediate is ever genuinely needed, the
         * honest way to get it is an optional store from the SHIPPED kernel,
         * not a second code path that only debuggers take. */
        if (!prefill_q_norm_rope_fused && ok) {
            fprintf(stderr, "pulsar: prefill Q reached neither the deferred nor the fused "
                            "norm+rope path -- refusing rather than leaving Q unnormalised\n");
            ok = false;
        }
        if (ok) {
            gpu_graph_debug_dump_q_tensor("Qcur", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
    }
    if (ok) ok = pulsar_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            PULSAR_N_HEAD_KV,
                                            PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            mseq ? g->batch_positions : NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    /* One pass: round-trip batch_kv in place (nope E4M3, rope bf16) AND emit the
     * packed rows attention will read.  This was pulsar_gpu_dsv4_fp8_kv_quantize_tensor
     * followed by attention reading the f32 staging directly -- so the chunk's own
     * KV was multiplied at 4 bytes/element while every later chunk read the same
     * rows out of the ring at 584 B.
     *
     * This is now the ONLY quantise of these rows.  It used to argue that its
     * fast-math scale matched the ring store's, so the chunk's rows and the
     * ring's rows "agree byte for byte" -- but that was two quantisations of the
     * same data hoping to land identically, and the second one was fed an
     * already-round-tripped buffer.  The ring store now scatters THESE bytes
     * (pulsar_gpu_store_raw_kv_batch_packed_tensor), so the agreement is
     * structural and there is nothing left to argue about.
     *
     * The in-place round-trip of batch_kv is OBSERVER-ONLY now.  It carried a
     * comment claiming "the compressor reads it afterwards" -- FALSE (verified
     * 2026-08-23): the attention compressor's inputs are
     * batch_comp_kv/batch_comp_sc, produced by GEMMs on batch_attn_norm, and
     * nothing in that chain touches batch_kv.  Its one real reader was the
     * per-token fallback below, which re-quantised a row of this buffer into
     * the ring; that now scatters the packed bytes instead, so after this pack
     * the only thing that ever looks at batch_kv is a dump or the range sweep
     * (L094 item 4).  ~8 MiB x 43 layers of stores per chunk. */
    if (ok) ok = pulsar_gpu_attn_pack_quantize_store_tensor(g->batch_kv,
                                                           g->batch_kv_pack,
                                                           0u,
                                                           n_tokens,
                                                           PULSAR_N_HEAD_DIM,
                                                           PULSAR_N_ROT,
                                                           gpu_graph_f32_store_observed_any()) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = pulsar_gpu_store_raw_kv_batch_packed_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv_pack,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    PULSAR_N_HEAD_DIM,
                                                                    NULL, NULL, 1) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = pulsar_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv_pack,
                                                          n_tokens,
                                                          g->raw_window,
                                                          PULSAR_N_HEAD,
                                                          PULSAR_N_HEAD_DIM,
                                                          mseq ? g->batch_positions : NULL,
                                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = pulsar_gpu_store_raw_kv_batch_packed_tensor(mseq ? gpu_graph_bank_raw_pool(g, il)
                                                    : g->layer_raw_cache[il],
                                                 g->batch_kv_pack,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 PULSAR_N_HEAD_DIM,
                                                 mseq ? g->batch_positions : NULL,
                                                 mseq ? g->batch_seq_id : NULL,
                                                 mseq ? nb : 1) != 0;
        if (ok) {
            ok = pulsar_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                        : g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   mseq ? 0 : n_raw,
                                                                   g->raw_cap,
                                                                   mseq ? 0 : raw_start,
                                                                    g->raw_window,
                                                                    PULSAR_N_HEAD,
                                                                    PULSAR_N_HEAD_DIM,
                                                                    0,
                                                                    mseq ? g->batch_positions : NULL,
                                                                    mseq ? g->batch_seq_id : NULL,
                                                                    0,
                                                                    mseq ? nb : 1,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = pulsar_compress_coff(ratio);
        const uint32_t comp_width = coff * PULSAR_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "pulsar: GPU layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->attn_compressor_kv,
                                             PULSAR_N_EMBD,
                                             comp_width,
                                             g->batch_attn_norm,
                                             n_tokens) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->attn_compressor_gate,
                                                     PULSAR_N_EMBD,
                                                     comp_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
        }
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        /* Stage-B save: keep this batch's per-position compressor projections so
         * a partial spec accept can roll the pool state forward without a
         * transformer replay. Must run here -- the indexer section below reuses
         * batch_comp_kv/sc. */
        if (ok && g->spec_comp_save_n && g->spec_comp_kv_save[il]) {
            uint32_t sn = g->spec_comp_save_n;
            if (sn > n_tokens) sn = n_tokens;
            if (sn > PULSAR_SPEC_LOGITS_ROWS + 1u) sn = PULSAR_SPEC_LOGITS_ROWS + 1u;
            const uint64_t sb = (uint64_t)sn * comp_width * sizeof(float);
            /* ASYNC: nothing on the host reads these. They are written here and
             * consumed only by gpu_graph_dspark_compressor_rollforward, which
             * hands row views straight to the update kernels -- same stream, so
             * stream order already gives the ordering the blocking copy was
             * providing. As cudaMemcpy they were ~124 synchronous D2D per spec
             * step (41 compressor layers + 21 indexer layers, two each), and
             * each one stalled the host for a copy no host code was waiting
             * on. See L038. */
            ok = pulsar_gpu_tensor_copy_async(g->spec_comp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                 pulsar_gpu_tensor_copy_async(g->spec_comp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
        }
        /* The comp bound the attention launch below hands the kernels for
         * the WHOLE batch. Both arms assign it before that use: zero_prefix
         * from this chunk's own emit count, the else arm at its tail. */
        uint32_t n_comp = 0u;
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "pulsar: GPU layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "pulsar: GPU graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            pulsar_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = gpu_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                ok = attn_comp_target != NULL &&
                     pulsar_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         PULSAR_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         PULSAR_N_ROT,
                                                         compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         PULSAR_ROPE_YARN_BETA_FAST,
                                                         PULSAR_ROPE_YARN_BETA_SLOW,
                                                         PULSAR_RMS_EPS) != 0;
                if (ok && n_comp != 0) {
                    ok = gpu_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                /* Every whole prompt, including one shorter than the window:
                 * the rebuild lays out the complete group (if any) and the
                 * partial rows the way the decode store does (L168). */
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_ape,
                                                                     PULSAR_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                /* STAGE 1b: n_comp is this ALIGNED chunk's frontier for ONE
                 * sequence. Under mseq it was written to the scalar as a
                 * read-only cross-bank superset; with the scalar gone that
                 * write would land on cur_bank's real row and clobber it. The
                 * banked arm publishes per bank at 1438, so skip it here. */
                if (!mseq) gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) = n_comp;
                if (gpu_graph_store_commits(g, mseq) && ratio == 4)
                    ok = gpu_graph_proj_ring_deposit_tail(g, il, pos0, n_tokens,
                                                          comp_width, false);
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * PULSAR_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                gpu_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                gpu_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            /* Classic aligned-chunk fast path: one contiguous ratio-aligned run
             * on the single-session cache. */
            const bool aligned_chunk = !mseq &&
                                       (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            /* LEVER 2 (plan-34): a BANKED step whose whole batch is a single
             * same-bank contiguous run (step_begin guarantees contiguity, so
             * seq_id[0]==seq_id[last] => one run => one bank) that is ratio-
             * aligned reuses the SAME batched replay/pool kernels the classic
             * aligned path trusts, keyed at the bank's frontier — one launch
             * per stage instead of the per-row loop's N launches.  Byte-
             * identical by construction: the batched pool recomputes the same
             * candidate window the recurrent per-row state carries (proven
             * equivalent by the classic aligned-vs-per-row equivalence), and it
             * leaves the bank state in the exact configuration classic leaves
             * (compressor_prefill*_tensor's tail re-seed == the per-row shift).
             * A mixed step's decode rows are length-1 runs (seq_id[0]!=
             * seq_id[last]) and fall through to the per-row loop unchanged. */
            const uint32_t run_bank = mseq ? (uint32_t)g->ms_seq_id[0] : 0u;
            const bool mseq_aligned_run = mseq &&
                (uint32_t)g->ms_seq_id[n_tokens - 1u] == run_bank &&
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk || mseq_aligned_run) {
                /* One batched aligned-run emit, keyed either at the single
                 * session's frontier/state lanes or at a bank's.  LEVER 2
                 * (plan-34) is the banked case: a step whose whole batch is one
                 * same-bank contiguous ratio-aligned run reuses these same
                 * batched kernels instead of the per-row loop's N launches
                 * (measured 13.835 -> 3.354 ms/layer at K=2048).  quantize_fp8
                 * is false in both: the pack commit is the single fp8 quantizer. */
                const bool banked = mseq_aligned_run;
                const uint32_t bank = banked ? run_bank : 0u;
                const uint32_t comp_before = banked ? g->ms_n_comp[bank][il]
                                                    : gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                /* Banked state lanes are OWNED views and must be freed; the
                 * single-session ones are borrowed and must not be. */
                pulsar_gpu_tensor *st_kv = NULL, *st_sc = NULL;
                if (ok) {
                    if (banked) {
                        st_kv = gpu_graph_bank_attn_state_kv_view(g, il, bank);
                        st_sc = gpu_graph_bank_attn_state_score_view(g, il, bank);
                        ok = st_kv && st_sc;
                    } else {
                        st_kv = g->layer_attn_state_kv[il];
                        st_sc = g->layer_attn_state_score[il];
                    }
                }
                pulsar_gpu_tensor *attn_comp_target =
                    ok ? gpu_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
                if (ok && ratio == 4) {
                    ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target, st_kv, st_sc,
                            g->batch_comp_kv, g->batch_comp_sc,
                            model->map, model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM, pos0, n_tokens, PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                } else if (ok) {
                    ok = pulsar_gpu_compressor_prefill_tensor(
                            attn_comp_target, st_kv, st_sc,
                            g->batch_comp_kv, g->batch_comp_sc,
                            model->map, model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM, ratio, pos0, n_tokens, PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = banked
                        ? gpu_graph_commit_attn_comp_stage_bank(g, il, bank, comp_before, comp_chunk)
                        : gpu_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g, model,
                            st_kv, st_sc,
                            layer->attn_compressor_ape, PULSAR_N_HEAD_DIM, comp_width,
                            pos0, n_tokens);
                }
                if (ok) {
                    if (banked) g->ms_n_comp[bank][il] = comp_before + comp_chunk;
                    else        gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il)    = comp_before + comp_chunk;
                    if (gpu_graph_store_commits(g, banked) && ratio == 4)
                        ok = gpu_graph_proj_ring_deposit_tail(g, il, pos0, n_tokens,
                                                              comp_width, false);
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    gpu_graph_debug_dump_tensor("KVcompress", attn_comp_target,
                                                  (uint64_t)comp_chunk * PULSAR_N_HEAD_DIM, il, pos0);
                    gpu_graph_debug_dump_tensor("attn_state_kv", st_kv,
                                                  (uint64_t)comp_width * coff * ratio, il, pos0);
                    gpu_graph_debug_dump_tensor("attn_state_score", st_sc,
                                                  (uint64_t)comp_width * coff * ratio, il, pos0);
                }
                gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
                if (banked) {
                    pulsar_gpu_tensor_free(st_sc);
                    pulsar_gpu_tensor_free(st_kv);
                }
            } else {
                /* Per-row compressor loop.  Multiseq: row t belongs to bank
                 * ms_seq_id[t] at absolute position ms_positions[t] — the
                 * pool/emit run against THAT bank's state lanes and comp
                 * cache at ITS frontier (ms_n_comp[bank][il]), and bump only
                 * that bank's counter; the scalar layer_n_comp stays the
                 * step-top superset (read-only here — the §6.1 race class is
                 * structurally unreachable).  Per-bank ratio groups are
                 * independent, and the pack-mode f32 stage row is safe to
                 * share across banks: each iteration's emit packs it before
                 * the next iteration's kernels run on the same stream. */
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
                    const uint32_t bank = mseq ? (uint32_t)g->ms_seq_id[t] : 0u;
                    /* STAGE 1b: one storage, selected by bank id. Non-mseq is
                     * simply the current bank -- there is no separate scalar
                     * for it to be "the classic case" in any more. */
                    uint32_t *const n_comp_slot =
                        &g->ms_n_comp[mseq ? bank : gpu_graph_cur_bank(g)][il];
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && *n_comp_slot >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = *n_comp_slot;
                    pulsar_gpu_tensor *ms_st_kv = mseq
                        ? gpu_graph_bank_attn_state_kv_view(g, il, bank) : NULL;
                    pulsar_gpu_tensor *ms_st_sc = mseq
                        ? gpu_graph_bank_attn_state_score_view(g, il, bank) : NULL;
                    /* Packing stages in the shared f32 row and commits
                     * bank-aware below, so no per-bank comp view is needed. */
                    pulsar_gpu_tensor *ms_target = NULL;
                    ok = kv_view && sc_view &&
                         (!mseq || (ms_st_kv && ms_st_sc));
                    /* L124: pre-store slot capture on ratio-128 layers.  The
                     * classic (non-mseq, non-spec-armed) per-row extension is
                     * a per-position store like decode's; mseq goes through
                     * bank state views the lane machinery doesn't cover
                     * (undo log is zeroed across bank hand-offs anyway).
                     * note_pos for this path rides the same per-position
                     * hook decode uses (encode_token_raw_swa) when the row
                     * is the token eval; the chunked sync path notes below. */
                    if (ok && gpu_graph_store_commits(g, mseq) && ratio == 128u) {
                        ok = gpu_graph_r128_undo_capture(g, il, pos);
                        g->r128_perrow_chunk = true;
                    }
                    if (ok) {
                        ok = pulsar_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            mseq ? ms_st_kv : g->layer_attn_state_kv[il],
                                                            mseq ? ms_st_sc : g->layer_attn_state_score[il],
                                                            ms_target ? ms_target
                                                                      : gpu_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            PULSAR_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            gpu_graph_attn_comp_update_row(comp_row),
                                                            PULSAR_N_ROT,
                                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            PULSAR_ROPE_YARN_BETA_FAST,
                                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                                            PULSAR_RMS_EPS) != 0;
                    }
                    /* L120 value-half: classic sync extensions commit as they
                     * store; deposit the projection row.  Spec-armed passes
                     * (classic block eval) and mseq candidate rows never
                     * deposit — their committed positions are banked by the
                     * Stage A replay / Stage B rollforward instead. */
                    if (ok && gpu_graph_store_commits(g, mseq) && ratio == 4)
                        ok = gpu_graph_proj_ring_deposit(g, il, pos, kv_view,
                                                         sc_view, false);
                    if (ok && emit) {
                        pulsar_gpu_tensor *comp_row_view = ms_target
                            ? pulsar_gpu_tensor_view(ms_target,
                                                  (uint64_t)comp_row * PULSAR_N_HEAD_DIM * sizeof(float),
                                                  (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float))
                            : gpu_graph_attn_comp_row_view(g, il, comp_row);
                        /* comp_row_view aliases the f32 stage; the commit
                         * below quantizes+packs and roundtrips the stage in
                         * place, so the dump happens after it. */
                        ok = comp_row_view != NULL;
                        if (ok) {
                            ok = mseq
                                ? gpu_graph_commit_attn_comp_stage_bank(g, il, bank, comp_row, 1)
                                : gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                        }
                        if (ok) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          PULSAR_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        pulsar_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) (*n_comp_slot)++;
                    if (comp_counts) comp_counts[t] = *n_comp_slot;
                    pulsar_gpu_tensor_free(ms_target);
                    pulsar_gpu_tensor_free(ms_st_sc);
                    pulsar_gpu_tensor_free(ms_st_kv);
                    pulsar_gpu_tensor_free(sc_view);
                    pulsar_gpu_tensor_free(kv_view);
                }
            }
            /* L139. The attention launch below takes ONE comp bound for the
             * whole batch: each kernel row derives its visible rows from its
             * own position and CLAMPS to this (pulsar_cuda_attention.cu names
             * it "the cross-bank superset clamp"). Classic: the session's
             * post-emit frontier, as it always was. Banked: the max over the
             * batch of each row's emit-inclusive bound, (pos+1)/ratio -- the
             * value step_begin computes as sup[] and step_end asserts every
             * batched bank reached. Derived from the positions, never read
             * from a bank's row: stage 1b made this read resolve to cur_bank's
             * row, and cur_bank is the device VIEW binding, not a batch
             * member -- with a prefill bank installed it clamped every deeper
             * decode bank to the prefill bank's depth (mixed-neutrality gate
             * 4: bank 1 diverged, bank 0 sat below the clamp). The follow-up
             * that zeroed it as "dead" made the kernels skip compressed
             * attention outright (n_comp == 0 is "no comp operand"). */
            if (mseq) {
                /* The step's superset, computed and cap-checked once in
                 * step_begin (L178: this used to re-derive it from
                 * ms_positions with no cap check). */
                n_comp = g->batch_comp_sup[il];
            } else {
                n_comp = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
            }
        }

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * PULSAR_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "pulsar: GPU layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->indexer_compressor_kv,
                                                 PULSAR_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->indexer_compressor_gate,
                                                         PULSAR_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens) != 0;
            }
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            /* Stage-B save (indexer variant; see the attn compressor hook). */
            if (ok && g->spec_comp_save_n && g->spec_icomp_kv_save[il]) {
                uint32_t sn = g->spec_comp_save_n;
                if (sn > n_tokens) sn = n_tokens;
                if (sn > PULSAR_SPEC_LOGITS_ROWS + 1u) sn = PULSAR_SPEC_LOGITS_ROWS + 1u;
                const uint64_t sb = (uint64_t)sn * index_width * sizeof(float);
                /* ASYNC for the same reason as the attn compressor save above:
                 * read only by the rollforward's update kernels, same stream. */
                ok = pulsar_gpu_tensor_copy_async(g->spec_icomp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                     pulsar_gpu_tensor_copy_async(g->spec_icomp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
            }
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_q,
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM,
                                                          g->batch_qr_norm,
                                                          n_tokens);
            /* Fused rope + QAT: one launch over batch_indexer_q instead of the
             * old rope_tail + qat pair (bit-exact, see the kernel note). */
            if (ok) ok = pulsar_gpu_dsv4_indexer_rope_qat_tensor(g->batch_indexer_q,
                                                    g->batch_indexer_qp,
                                                    n_tokens,
                                                    PULSAR_N_INDEXER_HEAD,
                                                    PULSAR_N_INDEXER_HEAD_DIM,
                                                    PULSAR_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    PULSAR_ROPE_YARN_BETA_FAST,
                                                    PULSAR_ROPE_YARN_BETA_SLOW,
                                                    mseq ? g->batch_positions : NULL) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_weights,
                                              model,
                                              layer->indexer_proj,
                                                     PULSAR_N_EMBD,
                                                     PULSAR_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "pulsar: GPU layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = pulsar_gpu_compressor_prefill_tensor(g->idx_comp_stage,
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             PULSAR_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             PULSAR_N_ROT,
                                                             compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             PULSAR_ROPE_YARN_BETA_FAST,
                                                             PULSAR_ROPE_YARN_BETA_SLOW,
                                                             PULSAR_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = pulsar_gpu_dsv4_indexer_qat_pack_tensor(g->idx_comp_stage,
                                                                g->layer_index_comp_cache[il],
                                                                0,
                                                                n_comp,
                                                                PULSAR_N_INDEXER_HEAD_DIM,
                                                                gpu_graph_f32_store_observed_any()) != 0;
                    /* plan-33 inc C: boundary-row restore (whole-prefill site). */
                    if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                            g->banks.n_banks ? g->banks.cur_bank : 0u, 0, n_comp, true);
                }
                /* Same as the attention compressor above: every whole prompt,
                 * complete group plus partial rows (L168). */
                if (ok) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_ape,
                                                                     PULSAR_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    /* STAGE 1b: indexer twin of the attn guard at 1316 -- under
                     * mseq this was the read-only superset; the banked arm
                     * publishes per bank at 1829. */
                    if (!mseq) gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il) = n_comp;
                    if (gpu_graph_store_commits(g, mseq))
                        ok = gpu_graph_proj_ring_deposit_tail(g, il, pos0, n_tokens,
                                                              index_width, true);
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      g->idx_comp_stage,
                                                      (uint64_t)n_comp * PULSAR_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                /* Classic aligned fast path; LEVER 2 adds the banked single-
                 * same-bank-aligned-run variant (see the attn emit section). */
                const bool aligned_chunk = !mseq &&
                                           (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                const uint32_t run_bank = mseq ? (uint32_t)g->ms_seq_id[0] : 0u;
                const bool mseq_aligned_run = mseq &&
                    (uint32_t)g->ms_seq_id[n_tokens - 1u] == run_bank &&
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk || mseq_aligned_run) {
                    /* One batched aligned-run indexer emit, keyed either at the
                     * single session's frontier/state lanes or at a bank's
                     * (LEVER 2, plan-34).  fp4 stages in the shared
                     * idx_comp_stage -- one bank per step, so no aliasing --
                     * and packs into the destination index cache. */
                    const bool banked = mseq_aligned_run;
                    const uint32_t bank = banked ? run_bank
                                                 : (g->banks.n_banks ? g->banks.cur_bank : 0u);
                    const uint32_t index_before = banked ? g->ms_n_index_comp[bank][il]
                                                         : gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il);
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    /* Banked views are OWNED and must be freed; the
                     * single-session pointers are borrowed and must not be. */
                    pulsar_gpu_tensor *bank_idx = NULL;
                    pulsar_gpu_tensor *ist_kv = NULL, *ist_sc = NULL, *idx_dst = NULL;
                    if (ok) {
                        if (banked) {
                            bank_idx = gpu_graph_bank_index_comp_view(g, il, bank);
                            ist_kv = gpu_graph_bank_index_state_kv_view(g, il, bank);
                            ist_sc = gpu_graph_bank_index_state_score_view(g, il, bank);
                            idx_dst = bank_idx;
                            ok = bank_idx && ist_kv && ist_sc;
                        } else {
                            ist_kv = g->layer_index_state_kv[il];
                            ist_sc = g->layer_index_state_score[il];
                            idx_dst = g->layer_index_comp_cache[il];
                        }
                    }
                    pulsar_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = pulsar_gpu_tensor_view(
                                g->idx_comp_stage,
                                (uint64_t)index_before * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view, ist_kv, ist_sc,
                                g->batch_comp_kv, g->batch_comp_sc,
                                model->map, model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                PULSAR_N_INDEXER_HEAD_DIM, pos0, n_tokens, PULSAR_N_ROT,
                                compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                freq_base, freq_scale, ext_factor, attn_factor,
                                PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                                PULSAR_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_view,
                                                                    idx_dst,
                                                                    index_before,
                                                                    index_chunk,
                                                                    PULSAR_N_INDEXER_HEAD_DIM,
                                                                    gpu_graph_f32_store_observed_any()) != 0;
                        /* plan-33 inc C: boundary-row restore (chunked emit site —
                         * the replay-from-R path that recomputes row R/4). */
                        if (ok) ok = gpu_graph_emit_keep_restore(g, il, bank,
                                index_before, index_chunk, true);
                    }
                    if (ok) {
                        ok = gpu_graph_refresh_ratio4_compressor_state(g, model,
                                ist_kv, ist_sc,
                                layer->indexer_compressor_ape, PULSAR_N_INDEXER_HEAD_DIM,
                                index_width, pos0, n_tokens);
                    }
                    if (ok) {
                        if (banked) g->ms_n_index_comp[bank][il] = index_before + index_chunk;
                        else        gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il)    = index_before + index_chunk;
                        if (gpu_graph_store_commits(g, banked))
                            ok = gpu_graph_proj_ring_deposit_tail(g, il, pos0, n_tokens,
                                                                  index_width, true);
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        gpu_graph_debug_dump_tensor("indexer_KVcompress", index_view,
                                                      (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM, il, pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_kv", ist_kv,
                                                      (uint64_t)index_width * coff * ratio, il, pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_score", ist_sc,
                                                      (uint64_t)index_width * coff * ratio, il, pos0);
                    }
                    pulsar_gpu_tensor_free(index_view);
                    if (banked) {
                        pulsar_gpu_tensor_free(ist_sc);
                        pulsar_gpu_tensor_free(ist_kv);
                        pulsar_gpu_tensor_free(bank_idx);
                    }
                } else {
                    /* Per-row indexer compressor loop; multiseq semantics as
                     * in the attn emit loop above (bank state lanes, bank
                     * frontier row, bank counter bump; scalar = read-only
                     * superset).  The fp4 stage rows are indexed by the
                     * bank-LOCAL frontier: two banks at the same frontier
                     * share a stage row safely because each iteration's emit
                     * packs it before the next iteration's kernels run. */
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
                        const uint32_t bank = mseq ? (uint32_t)g->ms_seq_id[t] : 0u;
                        uint32_t *const n_index_slot =
                            &g->ms_n_index_comp[mseq ? bank : gpu_graph_cur_bank(g)][il];
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && *n_index_slot >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        /* L120 value-half: same commit-time deposit rule as
                         * the attn per-row loop above. */
                        if (gpu_graph_store_commits(g, mseq) &&
                            !gpu_graph_proj_ring_deposit(g, il, pos, kv_view,
                                                         sc_view, true)) {
                            pulsar_gpu_tensor_free(sc_view);
                            pulsar_gpu_tensor_free(kv_view);
                            ok = false;
                            break;
                        }
                        const uint32_t index_row = *n_index_slot;
                        pulsar_gpu_tensor *ms_st_kv = mseq
                            ? gpu_graph_bank_index_state_kv_view(g, il, bank) : NULL;
                        pulsar_gpu_tensor *ms_st_sc = mseq
                            ? gpu_graph_bank_index_state_score_view(g, il, bank) : NULL;
                        pulsar_gpu_tensor *ms_cache = mseq
                            ? gpu_graph_bank_index_comp_view(g, il, bank) : NULL;
                        ok = kv_view && sc_view &&
                             (!mseq || (ms_st_kv && ms_st_sc && ms_cache));
                        if (ok) {
                            ok = pulsar_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                mseq ? ms_st_kv : g->layer_index_state_kv[il],
                                                                mseq ? ms_st_sc : g->layer_index_state_score[il],
                                                                g->idx_comp_stage,
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                PULSAR_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                PULSAR_N_ROT,
                                                                compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                PULSAR_ROPE_YARN_BETA_FAST,
                                                                PULSAR_ROPE_YARN_BETA_SLOW,
                                                                PULSAR_RMS_EPS) != 0;
                        }
                        if (ok && emit) {
                            pulsar_gpu_tensor *index_row_view = pulsar_gpu_tensor_view(
                                    g->idx_comp_stage,
                                    (uint64_t)index_row * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else {
                                ok = pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_row_view,
                                                                           mseq ? ms_cache
                                                                                : g->layer_index_comp_cache[il],
                                                                           index_row,
                                                                           1,
                                                                           PULSAR_N_INDEXER_HEAD_DIM,
                                                                           gpu_graph_f32_store_observed_any()) != 0;
                                pulsar_gpu_tensor_free(index_row_view);
                            }
                            /* plan-33 inc C: boundary-row restore (banked emit). */
                            if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                                    mseq ? (uint32_t)bank
                                         : (g->banks.n_banks ? g->banks.cur_bank : 0u),
                                    index_row, 1, true);
                        }
                        if (ok && emit) (*n_index_slot)++;
                        if (index_counts) index_counts[t] = *n_index_slot;
                        pulsar_gpu_tensor_free(ms_cache);
                        pulsar_gpu_tensor_free(ms_st_sc);
                        pulsar_gpu_tensor_free(ms_st_kv);
                        pulsar_gpu_tensor_free(sc_view);
                        pulsar_gpu_tensor_free(kv_view);
                    }
                }
            }
        }

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);

            ok = pulsar_gpu_store_raw_kv_batch_packed_tensor(mseq ? gpu_graph_bank_raw_pool(g, il)
                                                        : g->layer_raw_cache[il],
                                                     g->batch_kv_pack,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     PULSAR_N_HEAD_DIM,
                                                     mseq ? g->batch_positions : NULL,
                                                     mseq ? g->batch_seq_id : NULL,
                                                     mseq ? nb : 1) != 0;
            if (ok && ratio == 4 && n_comp > PULSAR_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                /* PULSAR_PREFILL_SLICE: run [score -> top-k -> indexed attention]
                 * over <=slice-token spans so indexer_scores only ever holds
                 * one span of rows.  Per-token math is keyed on the absolute
                 * position (pos0+t) and the raw window/comp visibility are
                 * recomputed per span exactly like a smaller chunk, so slicing
                 * is bit-identical; slice==0 (unset) is one full-chunk span
                 * with pointer-identical arguments. */
                const uint32_t slice = gpu_graph_prefill_slice();
                const uint32_t span = (slice != 0u && slice < n_tokens) ? slice : n_tokens;
                /* Hoisted out of the span loop.  This selects the comp source and,
                 * on the non-native path, DEQUANTS all n_comp packed rows into the
                 * shared f32 shadow.  Both inputs (il, n_comp) are loop-invariant
                 * and the span body only READS the comp cache, so evaluating it per
                 * span re-dequanted every row once per span for an identical
                 * result.  Bit-exact: same rows, same kernel, same destination —
                 * only the redundant repeats are gone. */
                pulsar_gpu_tensor *span_comp_src =
                    mseq ? gpu_graph_bank_attn_comp_pool(g, il)
                         : g->layer_attn_comp_cache[il];
                const struct gpu_graph_span_ops sop = {
                    /* comp_src   */ span_comp_src,
                    /* raw_src    */ mseq ? gpu_graph_bank_raw_pool(g, il) : g->layer_raw_cache[il],
                    /* index_src  */ mseq ? gpu_graph_bank_index_comp_pool(g, il)
                                          : g->layer_index_comp_cache[il],
                    /* index_bases*/ mseq ? gpu_graph_bank_index_comp_bases(g, il) : NULL,
                    /* comp_bases */ mseq ? gpu_graph_bank_attn_comp_bases(g, il) : NULL,
                    /* comp_cap   */ mseq ? g->layer_comp_cap[il] : 0u,
                    /* n_banks    */ mseq ? nb : 1u,
                    /* mseq       */ mseq,
                };
                /* The span hands n_comp -- the ATTENTION frontier -- to the indexer as
                 * its row count and score stride too.  The indexer keeps its own
                 * frontier (ms_n_index_comp); the two are equal by construction on a
                 * ratio-4 layer and step_end asserts it after the step.  Assert it
                 * BEFORE the launch as well, per bank in the batch, so a divergence
                 * refuses instead of scoring rows past the indexer's frontier (L178). */
                if (ok && mseq) {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        if (t > 0 && g->ms_seq_id[t] == g->ms_seq_id[t - 1]) continue;
                        const uint32_t b = (uint32_t)g->ms_seq_id[t];
                        if (g->ms_n_index_comp[b][il] != g->ms_n_comp[b][il]) {
                            fprintf(stderr, "pulsar: layer %u bank %u: indexer comp frontier %u != attention comp "
                                            "frontier %u before the indexed span -- refusing\n",
                                    il, b, g->ms_n_index_comp[b][il], g->ms_n_comp[b][il]);
                            ok = false;
                        }
                    }
                }
                for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += span) {
                    const uint32_t sn = n_tokens - s0 < span ? n_tokens - s0 : span;
                    const uint32_t spos0 = pos0 + s0;
                    const uint32_t s_n_raw = gpu_graph_raw_span_for_batch(g, spos0, sn);
                    const uint32_t s_raw_start = gpu_graph_raw_start_for_span(g,
                                                                                spos0 + sn - 1u,
                                                                                s_n_raw);
                    ok = gpu_graph_indexed_attention_span(g, model, layer, il,
                            s0, sn, spos0, q_dim, n_comp, ratio, index_scale,
                            mseq ? 0u : s_n_raw, mseq ? 0u : s_raw_start,
                            &sop);
                }
            } else if (ok) {
                ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                         model->map,
                                                                         model->size,
                                                                         layer->attn_sinks->abs_offset,
                                                                         g->batch_q,
                                                                         mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                              : g->layer_raw_cache[il],
                                                                         mseq ? gpu_graph_bank_attn_comp_pool(g, il)
                                                                              : g->layer_attn_comp_cache[il],
                                                                         n_tokens,
                                                                         pos0,
                                                                         mseq ? 0 : n_raw,
                                                                         g->raw_cap,
                                                                         mseq ? 0 : raw_start,
                                                                         n_comp,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          PULSAR_N_HEAD,
                                                                          PULSAR_N_HEAD_DIM,
                                                                          0,
                                                                          mseq ? g->batch_positions : NULL,
                                                                          mseq ? g->batch_seq_id : NULL,
                                                                          mseq ? gpu_graph_bank_attn_comp_bases(g, il) : NULL,
                                                                          mseq ? g->layer_comp_cap[il] : 0,
                                                                          mseq ? nb : 1,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > PULSAR_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
            /* PULSAR_PREFILL_SLICE: same span loop as the chunked branch.  The
             * zero-prefix case is the decode-batch entry with pos0 == 0
             * (zero_prefix means pos0 == 0, same launcher, causal), so a span at
             * offset s0 scores the same per-token values with pos0 = s0.  Attention per span keeps
             * first_raw_pos == 0 by passing n_raw = s0 + sn with raw_start 0. */
            const uint32_t zslice = gpu_graph_prefill_slice();
            const uint32_t zspan = (zslice != 0u && zslice < n_tokens) ? zslice : n_tokens;
            /* The packed cache straight in, like every other span site. This
             * branch built the f32 shadow unconditionally and was the ONLY
             * source of attn_pack_dequant launches in production. */
            pulsar_gpu_tensor *zspan_comp_src = g->layer_attn_comp_cache[il];
            const struct gpu_graph_span_ops zsop = {
                /* comp_src   */ zspan_comp_src,
                /* raw_src    */ g->layer_raw_cache[il],
                /* index_src  */ g->layer_index_comp_cache[il],
                /* index_bases*/ NULL,
                /* comp_bases */ NULL,
                /* comp_cap   */ 0u,
                /* n_banks    */ 1u,
                /* mseq       */ false,
            };
            for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += zspan) {
                const uint32_t sn = n_tokens - s0 < zspan ? n_tokens - s0 : zspan;
                const uint32_t spos0 = pos0 + s0;
                ok = gpu_graph_indexed_attention_span(g, model, layer, il,
                        s0, sn, spos0, q_dim, n_comp, ratio, index_scale,
                        s0 + sn, 0u,
                        &zsop);
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            /* Whole batch goes through this ONE call, so the completeness
             * requirement the raw site documents holds here by construction:
             * either the fp16 tier writes the encoding for every token or
             * mx_out stays 0 and the "a" GEMM quantizes as before.  This is
             * the per-layer traffic carrier -- the raw site runs twice a
             * prefill, this one for every layer (L039 item 2; D1 measured the
             * quantize pass it replaces at 117 ms / 43 launches). */
            /* A missing slot used to zero the pointers and let the "a" GEMM
             * quantize the heads in a separate pass -- a second arithmetic
             * for the same conversation, chosen by whether a scratch
             * reservation succeeded, with no message (L174; the L158 shape). */
            if (!pulsar_gpu_mxfp8_gact_slot(g->batch_heads, n_tokens, n_groups, group_dim,
                                            &gact_data, &gact_scale, &gact_kbp, &gact_slab)) {
                fprintf(stderr, "pulsar: layer %u: grouped E4M3 activation slot for %u x %u x %u heads "
                                "unavailable -- refusing (no quantize-pass fallback)\n",
                        il, n_tokens, n_groups, group_dim);
                ok = false;
            }
            if (ok) ok = pulsar_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv_pack,
                                                                       /* Packed pool straight in.  This called
                                                                        * gpu_graph_attn_comp_read_cache -- a full
                                                                        * dequantise of 584 B rows into a 2048 B f32
                                                                        * shadow, on the SHIPPED path, because the
                                                                        * consumer could not read packed.  The launcher
                                                                        * has ONE arm now (L166): the fp16 tier reads
                                                                        * the ATTN_PACK pool natively through comp_kv;
                                                                        * no shadow, no second kernel. */
                                                                       mseq ? gpu_graph_bank_attn_comp_pool(g, il)
                                                                            : g->layer_attn_comp_cache[il],
                                                                       gact_data, gact_scale, gact_kbp,
                                                                       (uint32_t)gact_slab, n_groups,
                                                                       PULSAR_N_HEAD_DIM - PULSAR_N_ROT,
                                                                       &gact_emitted,
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       PULSAR_N_HEAD,
                                                                       PULSAR_N_HEAD_DIM,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
            if (!gact_emitted) { gact_data = NULL; gact_scale = NULL; }
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && mseq && !batch_attention_done) {
        /* Every multiseq-legal shape is handled by the banked branches above;
         * the fallback below is classic single-session (shadow reads, scalar
         * spans).  Reaching it banked would silently compute the wrong rows. */
        fprintf(stderr, "pulsar: multiseq layer batch rejected: unsupported shape "
                        "(layer %u n_tokens=%u raw_cap=%u)\n",
                il, n_tokens, g->raw_cap);
        ok = false;
    }
    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        /* ELIGIBILITY.  batch_heads is written by this call for the raw
         * PREFIX and by the per-token loop below for the rest, so the fused
         * encoding is only complete when the whole batch went through here.
         * Anything less and part of the E4M3 buffer is never written -- a
         * wrong answer, not a slow one -- so the fusion is refused rather
         * than partially applied.  (Threading the indexed per-token path is
         * the follow-up that lifts this restriction.) */
        /* (the cache was disarmed for this layer at the top of the encode) */
        if (ok && raw_prefix_tokens == n_tokens &&
            !pulsar_gpu_mxfp8_gact_slot(g->batch_heads, n_tokens, n_groups, group_dim,
                                        &gact_data, &gact_scale, &gact_kbp, &gact_slab)) {
            gact_data = NULL; gact_scale = NULL; gact_kbp = 0; gact_slab = 0;
        }
        if (raw_prefix_tokens != 0) {
            ok = pulsar_gpu_attention_prefill_raw_heads_mx_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv_pack,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              PULSAR_N_HEAD,
                                                              PULSAR_N_HEAD_DIM,
                                                              gact_data, gact_scale, gact_kbp,
                                                              (uint32_t)gact_slab, n_groups,
                                                              PULSAR_N_HEAD_DIM - PULSAR_N_ROT,
                                                              &gact_emitted,
                                          mseq ? g->batch_positions : NULL,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
        }
        if (!gact_emitted) { gact_data = NULL; gact_scale = NULL; }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                bool have_topk = false;
                /* The indexer ranks ITS compressed rows and the attention
                 * folds the selected ids over ITS compressed rows: the two
                 * frontiers must agree or an id is out of range on one side.
                 * The arm used to gate on cur_comp and pass cur_index, and a
                 * disagreement surfaced only as a silent top_k > n_comp
                 * refusal inside the ranking entry (L174). */
                if (ratio == 4 && cur_comp != cur_index) {
                    fprintf(stderr, "pulsar: layer %u pos %u: attention comp frontier %u != indexer comp frontier %u "
                                    "-- refusing\n", il, pos, cur_comp, cur_index);
                    ok = false;
                    break;
                }

                if (ratio == 4 && cur_comp > PULSAR_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                    pulsar_gpu_tensor *indexer_q_view = pulsar_gpu_tensor_view(
                            g->batch_indexer_qp,
                            (uint64_t)t * PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES,
                            (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_ENGINE_IDXFP4_ROWBYTES);
                    pulsar_gpu_tensor *indexer_w_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_weights, t, PULSAR_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         pulsar_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            PULSAR_N_INDEXER_HEAD,
                                                            PULSAR_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       PULSAR_N_INDEXER_TOP_K) != 0;
                    pulsar_gpu_tensor_free(indexer_w_view);
                    pulsar_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        have_topk = true;
                        n_selected = PULSAR_N_INDEXER_TOP_K < cur_index
                            ? PULSAR_N_INDEXER_TOP_K
                            : cur_index;
                        /* Mirror of the batch path's dump at :545.  This deep
                         * per-token path had no selection dump, which made
                         * "did the top-k SELECTION change?" unanswerable
                         * exactly where it matters: the L033 flip's only
                         * regressing depth (story@4096) is a PRUNED depth, and
                         * every unpruned depth moved closer to source.
                         * Discrete reselection vs continuous rounding is the
                         * whole verdict question, and this is the instrument
                         * that answers it. */
                        gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                g->comp_selected, (uint64_t)n_selected, il, pos);
                    }
                }

                pulsar_gpu_tensor *q_view = gpu_graph_q_row_view(g->batch_q, t, q_dim);
                /* Row t of the PACKED rows the kv_path already emitted, not a
                 * row of the f32 staging.  This site used to re-quantise
                 * batch_kv row t into the ring -- a SECOND quantise of rows the
                 * pack had already round-tripped, which is exactly the
                 * not-bit-idempotent double-quantise this file's header warns
                 * about and which the batched arms fixed by scattering bytes
                 * (L094 item 4).  Copying the bytes makes the ring agree with
                 * what attention read by construction. */
                pulsar_gpu_tensor *kv_pack_view = pulsar_gpu_tensor_view(
                        g->batch_kv_pack,
                        (uint64_t)t * PULSAR_ENGINE_ATTN_PACK_ROWBYTES,
                        PULSAR_ENGINE_ATTN_PACK_ROWBYTES);
                pulsar_gpu_tensor *heads_view = gpu_graph_heads_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_pack_view && heads_view;
                if (ok && !zero_prefix) {
                    /* n_tokens=1 with pos0=pos puts the row at pos % raw_cap --
                     * the same slot the f32 store targeted (attn_pack_ring_slot). */
                    ok = pulsar_gpu_store_raw_kv_batch_packed_tensor(g->layer_raw_cache[il],
                                                       kv_pack_view,
                                                       g->raw_cap,
                                                       pos,
                                                       1u,
                                                       PULSAR_N_HEAD_DIM,
                                                       NULL, NULL, 1u) != 0;
                }
                if (ok && have_topk && n_selected != 0) {
                    ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              /* Native packed read: this sits in a PER-TOKEN loop and
                                                                               * cur_comp grows per token, so the shadow was rebuilt
                                                                               * for every token. */
                                                                              g->layer_attn_comp_cache[il],
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              PULSAR_N_HEAD,
                                                                              PULSAR_N_HEAD_DIM,
                                                                              NULL, NULL, NULL, 0, 1,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
                } else if (ok) {
                    /* No selection this token: the same one-row step the
                     * batched decode takes, through the same entry -- the
                     * fp16 tier for this shape, with q_prep honoured like the
                     * indexed sibling above.  (The single-token entry this
                     * replaced always ran the f32 kernel and had no q_prep
                     * parameter, L164.) */
                    ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(heads_view,
                            model->map, model->size, layer->attn_sinks->abs_offset,
                            q_view, g->layer_raw_cache[il],
                            cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                            1, pos, n_raw, g->raw_cap, raw_start, cur_comp,
                            g->raw_window, ratio, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM,
                            0, NULL, NULL, NULL, 0, 1,
                            g->q_prep_active ? &g->q_prep : NULL) != 0;
                }
                pulsar_gpu_tensor_free(heads_view);
                pulsar_gpu_tensor_free(kv_pack_view);
                pulsar_gpu_tensor_free(q_view);
            }
        }
    }

    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    /* Second half of the grouped encoding: this rewrites head dims
     * [n_nope, head_dim) in place, so it owns exactly the MX blocks the
     * attention epilogue deliberately skipped.  Only reached with slots when
     * that epilogue actually ran (gact_emitted). */
    if (ok) ok = pulsar_gpu_rope_tail_mx_tensor(g->batch_heads,
                                            n_tokens,
                                            PULSAR_N_HEAD,
                                            PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            mseq ? g->batch_positions : NULL,
                                            gact_data, gact_scale, gact_kbp,
                                            (uint32_t)gact_slab, n_groups) != 0;
    /* BOTH producers have now run: the encoding is complete and the "a" GEMM
     * may consume it instead of running its own quantise pass. */
    if (ok && gact_data) pulsar_gpu_mxfp8_gact_note();
    /* L158 inc 4: the attention arms without the E4M3 epilogue (the indexed
     * per-token span path, a raw prefix with an indexed remainder) leave no
     * grouped encoding.  The attention STAGE emits it here, after the inverse
     * rope, so the 'a' projection reads the same encoding on every arm; the
     * consumer's quantise-from-heads fallback is gone. */
    if (ok && !gact_data) ok = pulsar_gpu_mxfp8_gact_emit_heads(g->batch_heads, n_tokens, n_groups, group_dim) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) {
        ok = pulsar_gpu_attention_output_batch_tensor(g->batch_attn_out,
                                                   g->batch_attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   layer->attn_output_b->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   PULSAR_N_EMBD,
                                                   g->batch_heads,
                                                   n_tokens) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) {
        ok = pulsar_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                            g->batch_attn_out,
                                            g->batch_cur_hc,
                                            hc_split_view,
                                            PULSAR_N_EMBD,
                                            PULSAR_N_HC) != 0;
    }
    if (ok) gpu_graph_debug_dump_tensor("hc_split_coef", hc_split_view, (uint64_t)n_tokens * mix_hc, il, pos0);
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    pulsar_gpu_mxfp8_act_cache_disarm();
    pulsar_gpu_tensor_free(after_attn_hc_view);
    pulsar_gpu_tensor_free(attn_cur_view);
    pulsar_gpu_tensor_free(hc_split_view);
    pulsar_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
    return ok;
}



/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
/* L119 verdict: per-layer FFN graph segments were built, made bitwise
 * (parity-keyed), and MEASURED A NET LOSS on GB10 (-13% solo at 87% replay
 * rate): cudaGraphLaunch is expensive on integrated Blackwell, and 43 small
 * graph launches per round lose to eager launches the host already hides at
 * 92% busy. Deleted; the output head keeps capture (1 dense graph/round,
 * +1-2% measured). Full chain: pulsar-notes rows/L119.md. */
bool gpu_graph_encode_layer_ffn_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (!routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                        &gate_expert_bytes, &gate_row_bytes,
                                        &down_expert_bytes, &down_row_bytes)) {
        return false;
    }
    pulsar_gpu_tensor *hc_mix_view = pulsar_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *hc_split_view = pulsar_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *ffn_cur_view = pulsar_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    pulsar_gpu_tensor *next_hc_view = pulsar_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * PULSAR_HC_ELT_SIZE);  ///< carrier
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    void *ffn_norm_q = NULL, *ffn_norm_sf = NULL; int ffn_norm_kbp = 0;
    void *ffn_norm_b = NULL;
    uint32_t ffn_norm_keep_from = 0u;
    /* Same bf16 epilogue as the attention side, same reason -- see the note
     * there.  Still no e4m3 arm(). */
    void *flat_hc_b_ffn = NULL;
    if (ok && !pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_tokens,
                                        (uint64_t)hc_dim, &flat_hc_b_ffn)) {
        fprintf(stderr, "pulsar: ffn flat_hc: no bf16 slot -- refusing (L159)\n");
        ok = false;
    }
    /* L157: same dead-store skip as the attention-side flat_hc norm, same
     * predicate; the consumer is hc_ffn_fn's GEMM on the shared bf16 core. */
    const bool flat_skip_f32_ffn = flat_hc_b_ffn &&
                                   pulsar_gpu_matmul_batch_decode_rows() == 0 &&
                                   !gpu_graph_f32_store_observed_any();
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      flat_hc_b_ffn,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS,
                                                      flat_skip_f32_ffn ? 1 : 0) != 0;
    if (ok && flat_hc_b_ffn) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_tokens,
                                                      (uint64_t)hc_dim);
    if (ok && flat_skip_f32_ffn)
        pulsar_gpu_act_note_f32_skipped_for(g->batch_flat_hc, n_tokens, (uint64_t)hc_dim, 0u);
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_ffn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    {
        /* Same E4M3 epilogue as the attention norm above: batch_ffn_norm feeds
         * the router-logits and shared-expert MXFP8 GEMMs, which would otherwise
         * quantize the whole [n_tokens x n_embd] tensor in a separate
         * bandwidth-bound pass. */
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_ffn_norm, n_tokens, PULSAR_N_EMBD,
                                                        &ffn_norm_q, &ffn_norm_sf,
                                                        &ffn_norm_kbp)) {
            fprintf(stderr, "pulsar: ffn_norm: no E4M3 slot (n_tok=%u in_dim=%u) -- refusing (L189)\n",
                    n_tokens, (unsigned)PULSAR_N_EMBD);
            ok = false;
        }
        /* ...and the bf16 copy for the router's BF16 GEMM (ffn_gate_inp). */
        if (ok && !pulsar_gpu_bf16_act_slot(g->batch_ffn_norm, n_tokens, PULSAR_N_EMBD,
                                            &ffn_norm_b)) {
            fprintf(stderr, "pulsar: ffn_norm: no bf16 slot -- refusing (L159)\n");
            ok = false;
        }
        /* ffn_norm keeps NO f32 rows under the skip: its one offset reuse --
         * the output head's scratch view (the L035 site) -- WRITES its rows
         * before reading them, so it never sees ours. */
        ffn_norm_keep_from = 0u;
        /* Every MoE tier -- grouped, mixed-type, MMQ, the small-batch FFN GEMV
         * -- reads the producer's E4M3 for x and refuses without it, so once
         * the slot is armed nothing dereferences the f32 rows: ~128 MiB per
         * prefill of dead stores unless kept.  Two readers still want them
         * and say so: a dump, and the imatrix collector, which sums x^2 over
         * the f32 rows on the host (imatrix.cpp) and marks the graph while it
         * runs. */
        if (ffn_norm_q && ffn_norm_b && !g->imatrix_f32_rows &&
            pulsar_gpu_matmul_batch_decode_rows() == 0 &&
            !gpu_graph_f32_store_observed_any()) {
            ffn_norm_keep_from = n_tokens;
            static int announced_fns = 0;
            if (!announced_fns) {
                announced_fns = 1;
                fprintf(stderr, "pulsar: ffn_norm f32 store SKIPPED "
                                "(n_tok=%u, %.1f MiB/layer)\n", n_tokens,
                        (double)n_tokens * PULSAR_N_EMBD * sizeof(float) /
                        (1024.0 * 1024.0));
            }
        }
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
                                                                 gpu_graph_f32_store_observed("hc_ffn_pre", il, pos0)
                                                                     ? ffn_cur_view : NULL,
                                                                 g->batch_ffn_norm,
                                                                 ffn_norm_q,
                                                                 ffn_norm_sf,
                                                                 ffn_norm_kbp,
                                                                 ffn_norm_b,
                                                                 ffn_norm_keep_from,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_after_attn_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_ffn_scale->abs_offset,
                                                                 layer->hc_ffn_base->abs_offset,
                                                                 layer->ffn_norm->abs_offset,
                                                                 n_tokens,
                                                                 PULSAR_N_EMBD,
                                                                 PULSAR_N_HC,
                                                                 PULSAR_N_HC_SINKHORN_ITER,
                                                                 PULSAR_HC_EPS,
                                                                 PULSAR_RMS_EPS,
        layer->ffn_norm->type == PULSAR_TENSOR_BF16) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    /* Attention is finished for this layer, so re-keying the single-slot
     * activation cache onto ffn_norm cannot strand a live attn_norm encoding. */
    if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_ffn_norm, n_tokens, PULSAR_N_EMBD);
    if (ok && ffn_norm_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    if (ok && ffn_norm_b) pulsar_gpu_bf16_act_note(g->batch_ffn_norm, n_tokens, PULSAR_N_EMBD);
    if (ok && ffn_norm_keep_from) pulsar_gpu_mxfp8_act_cache_note_f32_skipped(ffn_norm_keep_from);
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_router_logits,
                                              model,
                                              layer->ffn_gate_inp,
                                             PULSAR_N_EMBD,
                                             PULSAR_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = pulsar_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      gpu_graph_f32_store_observed("ffn_moe_probs", il, pos0)
                                                          ? g->batch_router_probs : NULL,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      PULSAR_N_EXPERT,
                                                      PULSAR_N_EXPERT_USED,
                                                      PULSAR_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * PULSAR_N_EXPERT_USED, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED, il, pos0);
    }

    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos0);

#define PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_gate, \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          PULSAR_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_up, \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          PULSAR_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        void *shmid_q = NULL, *shmid_sf = NULL; int shmid_kbp = 0; \
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_shared_mid, n_tokens, \
                                                        (uint64_t)shared_dim, \
                                                        &shmid_q, &shmid_sf, &shmid_kbp)) { \
            fprintf(stderr, "pulsar: shared_mid: no E4M3 slot (n_tok=%u in_dim=%u) -- refusing (L189)\n", \
                    n_tokens, (unsigned)shared_dim); \
            ok = false; \
        } \
        /* DEAD-STORE ELIMINATION. batch_shared_mid's only reader is the MXFP8 \
         * shared_down GEMM immediately below, and the swiglu epilogue hands it \
         * the E4M3 encoding directly, so the f32 store has no consumer at all. \
         * Dropping it removes n_tokens*shared_dim*4 B per layer (32 MiB at a \
         * 4096-token chunk, ~1.4 GB across 43 layers) and is BIT-EXACT: the \
         * encoding written is byte-for-byte what it was before. \
         * \
         * Gated on the SAME predicate as the emission (shmid_q non-NULL), per \
         * the rule that cost this exact file a wrong answer once already -- a \
         * fusion whose arm and emission use different predicates arms the \
         * cache off a buffer it never wrote. The arm/note pair follows on the \
         * next two lines with no intervening arm, which is what makes the \
         * consumer's cache hit certain rather than likely; \
         * act_f32_absent_hazard() in pulsar_cuda_matmul.cu is the loud \
         * backstop if that adjacency is ever broken by reordering. */ \
        /* ⚠ AND mixed-batch must be DISARMED.  cuda-mixed-neutrality-gate \
         * caught this: at n_dec=2 of 66 the mxfp8 dispatch splits the batch \
         * and recurses on OFFSET row pointers, which key no cache slot, so \
         * BOTH halves quantize from f32 -- the store we just skipped.  The \
         * backstop refused (correctly) and the GEMM failed.  d967327's \
         * predicate required batch_decode_rows == 0 for exactly this reason; \
         * dropping it was my error.  The invariant "valid => every arm takes \
         * A8" holds only for arms that LOOK UP the cache, and the split does \
         * not. */ \
        const int shmid_skip_f32 = (shmid_q != NULL) && \
                                   pulsar_gpu_matmul_batch_decode_rows() == 0; \
        if (ok) ok = pulsar_gpu_swiglu_mx_tensor(g->batch_shared_mid, \
                                             g->batch_shared_gate, \
                                             g->batch_shared_up, \
                                             (uint32_t)((uint64_t)n_tokens * shared_dim), \
                                             PULSAR_SWIGLU_CLAMP_EXP, \
                                             1.0f, \
                                             shmid_q, shmid_sf, shmid_kbp, \
                                             (uint32_t)shared_dim, \
                                             shmid_skip_f32) != 0; \
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_shared_mid, n_tokens, (uint64_t)shared_dim); \
        if (ok && shmid_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8(); \
        if (ok && shmid_skip_f32) pulsar_gpu_mxfp8_act_cache_note_f32_skipped(n_tokens); \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              g->batch_shared_out, \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              PULSAR_N_EMBD, \
                                                                              g->batch_shared_mid, \
                                                                              n_tokens); \
        if (ok) { \
            gpu_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out, \
                                          (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0); \
        } \
    } while (0)

    if (ok) {
        ok = pulsar_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               tensor_map_base(model, layer->ffn_gate_exps),
                                               tensor_map_size(model, layer->ffn_gate_exps),
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               pulsar_layer_n_expert(il),
                                               PULSAR_N_EXPERT_USED,
                                               PULSAR_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens) != 0;
    }
    if (ok) {
        /* ARM-DEPENDENT: batch_routed_up is only written by the MMQ arms
         * (where it serves as raw-gate scratch); on 40/40-grouped and mixed
         * case-A layers this dump shows a PREVIOUS layer's bytes. Same below
         * for ffn_moe_down: the grouped path sums straight from its padded
         * GEMM output and never touches batch_routed_down. */
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * down_in_dim;
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                      routed_mid_elems, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * PULSAR_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
#undef PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT

    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_batch_ffn_out(g) &&
             pulsar_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * PULSAR_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = pulsar_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              PULSAR_N_EMBD,
                                              PULSAR_N_HC) != 0;
    }
    else if (ok) {
        ok = pulsar_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    /* Mirror the attention encode's disarm.  The activation cache is keyed only
     * on (ptr, n_tok, in_dim), and the batch output heads REUSE batch_ffn_norm
     * as their output_norm scratch at the same n_tok and in_dim -- so an armed
     * key surviving this function lets the vocab GEMM hit layer-42's stale
     * E4M3 activations and emit silently wrong logits.  Arm/disarm must pair
     * within one encode. */
    pulsar_gpu_mxfp8_act_cache_disarm();
    pulsar_gpu_tensor_free(next_hc_view);
    pulsar_gpu_tensor_free(ffn_cur_view);
    pulsar_gpu_tensor_free(hc_split_view);
    pulsar_gpu_tensor_free(hc_mix_view);
    return ok;
}



/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool gpu_graph_encode_layer_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    /* L124: the per-row-capture flag is per CHUNK; clear it on entry to the
     * first layer so a failed chunk cannot leak it into a later aligned
     * chunk's note block (reviewer finding 3 -- a note without a capture
     * restores stale lane bytes). */
    if (il == 0u) g->r128_perrow_chunk = false;

    bool ok = gpu_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (!ok) {
        fprintf(stderr, "pulsar: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = gpu_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
        if (!ok) {
            fprintf(stderr, "pulsar: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        pulsar_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* L120 value-half: after the LAST layer of a committed (non-mseq,
     * non-spec-armed) chunk, every ratio-4 layer has banked the chunk's
     * tail-8 projections — advance the deposited span once per position. */
    if (ok && il + 1u == PULSAR_N_LAYER &&
        gpu_graph_store_commits(g, g->batch_multiseq != 0)) {
        const uint32_t tail = n_tokens < 8u ? n_tokens : 8u;
        for (uint32_t k = 0; k < tail; k++)
            gpu_graph_proj_ring_note_pos(g, pos0 + n_tokens - tail + k);
        /* L124: note ONLY when the per-row arm captured this chunk's
         * ratio-128 slots (a note without a capture would restore stale
         * lane bytes).  The aligned batch arm doesn't capture -- and doesn't
         * need to: a rewind can never target into an aligned chunk, and the
         * newest-first walk stops before any pre-chunk entry. */
        if (g->r128_perrow_chunk) {
            const uint32_t utail = n_tokens < PULSAR_REWIND_RING_DEPTH ? n_tokens : PULSAR_REWIND_RING_DEPTH;
            for (uint32_t k = 0; k < utail; k++)
                gpu_graph_r128_undo_note_pos(g, pos0 + n_tokens - utail + k);
        }
        g->r128_perrow_chunk = false;
    }
    /* Fused spec loop (P2): when armed, capture the drafter's anchor hidden for
     * every batch position at the anchor layers, so the last-accepted position's
     * hidden is available without a replay decode. Off (0) during prefill and
     * plain decode. */
    if (ok && g->dspark_capture_batch_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_target_h_batch[slot]) {
                /* L190 D2: the anchor layer matched but its capture buffer is
                 * missing.  Breaking with ok untouched let the round seed from
                 * STALE rows and lose acceptance silently -- the class
                 * gpu_decode.cpp's KV seed documents. */
                fprintf(stderr, "pulsar: drafter anchor capture: layer %u is anchor slot %d but "
                                "its batch buffer is not allocated -- refusing\n", il, slot);
                ok = false;
                break;
            }
            uint32_t cap_n = g->dspark_capture_batch_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!pulsar_gpu_dspark_hc_mean_reduce_batch(g->dspark_target_h_batch[slot],
                                                     g->batch_cur_hc,
                                                     PULSAR_N_EMBD, PULSAR_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    /* Bulk prefill capture for drafter retraining (PULSAR_DSPARK_PREFILL_DUMP):
     * same reduction as the verify capture above, but over EVERY chunk position
     * into the per-layer bulk buffers. Armed only by the prefill path. */
    if (ok && g->dspark_bulk_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_bulk_h[slot]) {
                /* same class as the verify capture above: armed with a
                 * missing buffer is an impossible state, not a skip */
                fprintf(stderr, "pulsar: drafter bulk capture: layer %u is anchor slot %d but "
                                "its bulk buffer is not allocated -- refusing\n", il, slot);
                ok = false;
                break;
            }
            uint32_t cap_n = g->dspark_bulk_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!pulsar_gpu_dspark_hc_mean_reduce_batch(g->dspark_bulk_h[slot],
                                                     g->batch_cur_hc,
                                                     PULSAR_N_EMBD, PULSAR_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    return ok;
}









/* Stage-B no-replay rollback for the fused spec loop: after restoring the
 * pre-batch frontier snapshot, roll ONLY the recurrent compressor/indexer pool
 * state forward through the committed batch positions using the projections
 * saved during the verify batch (spec_comp_*_save). Bit-identical to what a
 * transformer replay would produce: the same per-token update kernels run on
 * the same input rows in the same order -- minus the 43-layer forward. The
 * comp-cache rows and raw KV need no work (the batch already wrote the
 * committed positions' rows from identical state; rejected rows are position-
 * addressed and get overwritten). Counters are set by formula. The pooled-row
 * emit goes to a scratch sink (cache rows are already correct). */
/* L189: a rollforward that stops at layer il leaves layers 0..il-1 advanced to
 * pos0+n_positions and layers il.. at the restored frontier -- the pool state
 * is HALF-ADVANCED and no per-layer commit can undo the layers already
 * rolled (their update kernels are recurrent).  The exit therefore names the
 * layer, position and kernel, and says so; the caller (session_spec.cpp,
 * "DSpark fused state update failed") invalidates the checkpoint, which is
 * what makes the half-advanced state unreachable. */
static bool rollforward_fail(uint32_t il, uint32_t pos, const char *what) {
    fprintf(stderr, "pulsar: compressor rollforward FAILED at layer %u pos %u (%s) -- layers 0..%u are "
                    "advanced to the committed frontier, layers %u.. are not: the pool state is "
                    "half-advanced and the caller must invalidate the checkpoint (it does)\n",
            il, pos, what, il == 0u ? 0u : il - 1u, il);
    return false;
}

bool gpu_graph_dspark_compressor_rollforward(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        const pulsar_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions,
        uint32_t          save_row0) {
    if (!g || !model || !weights) {
        fprintf(stderr, "pulsar: compressor rollforward: NULL graph/model/weights -- refusing\n");
        return false;
    }
    if (n_positions == 0) return true;
    if (save_row0 + n_positions > PULSAR_SPEC_LOGITS_ROWS + 1u || !g->spec_comp_scratch_row) {
        fprintf(stderr, "pulsar: compressor rollforward: save rows %u+%u exceed the %u-row save slab, or no "
                        "scratch row (%d) -- refusing before any layer moved\n",
                save_row0, n_positions, (unsigned)PULSAR_SPEC_LOGITS_ROWS + 1u,
                g->spec_comp_scratch_row != NULL);
        return false;
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const pulsar_layer_weights *layer = &weights->layer[il];
        const uint32_t coff = pulsar_compress_coff(ratio);
        const uint32_t comp_width = coff * PULSAR_N_HEAD_DIM;
        const uint32_t index_width = 2u * PULSAR_N_INDEXER_HEAD_DIM;
        const float freq_base = layer_rope_freq_base(il);
        const float freq_scale = layer_rope_freq_scale(il);
        const float ext_factor = PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
        float attn_factor = 1.0f;
        if (ext_factor != 0.0f && freq_scale > 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        if (!g->spec_comp_kv_save[il] || !g->spec_comp_sc_save[il])
            return rollforward_fail(il, pos0, "no saved compressor projections for this layer");
        for (uint32_t t = 0; t < n_positions; t++) {
            const uint32_t pos = pos0 + t;
            pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->spec_comp_kv_save[il], save_row0 + t, comp_width);
            pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->spec_comp_sc_save[il], save_row0 + t, comp_width);
            /* L124: pre-store slot capture (ratio-128 layers). */
            if (ratio == 128u && !gpu_graph_r128_undo_capture(g, il, pos)) {
                pulsar_gpu_tensor_free(sc_view);
                pulsar_gpu_tensor_free(kv_view);
                return rollforward_fail(il, pos, "ratio-128 undo capture");
            }
            bool ok = kv_view && sc_view &&
                pulsar_gpu_compressor_update_tensor(kv_view, sc_view,
                        g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                        g->spec_comp_scratch_row,
                        model->map, model->size,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        layer->attn_compressor_norm->abs_offset,
                        layer->attn_compressor_norm->type,
                        PULSAR_N_HEAD_DIM, ratio, pos, 0,
                        PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                        PULSAR_RMS_EPS) != 0;
            /* L120 value-half: rollforward positions are the round's
             * COMMITTED prefix -- the batched lane's deposit point (the
             * ONE-STATE-MODEL stage 3 contract at gpu_graph_store_commits
             * names it; a fully accepted round has no rollforward and
             * deposits nothing, by decision). */
            const bool attn_updated = ok;
            if (ok && ratio == 4)
                ok = gpu_graph_proj_ring_deposit(g, il, pos, kv_view, sc_view, false);
            pulsar_gpu_tensor_free(sc_view);
            pulsar_gpu_tensor_free(kv_view);
            if (!ok)
                return rollforward_fail(il, pos, attn_updated ? "attention projection-ring deposit"
                                                              : "attention compressor update (row view or kernel)");
            if (ratio == 4 && g->spec_icomp_kv_save[il]) {
                pulsar_gpu_tensor *ikv = gpu_graph_tensor_row_view(g->spec_icomp_kv_save[il], save_row0 + t, index_width);
                pulsar_gpu_tensor *isc = gpu_graph_tensor_row_view(g->spec_icomp_sc_save[il], save_row0 + t, index_width);
                ok = ikv && isc &&
                    pulsar_gpu_compressor_update_tensor(ikv, isc,
                            g->layer_index_state_kv[il], g->layer_index_state_score[il],
                            g->spec_comp_scratch_row,
                            model->map, model->size,
                            layer->indexer_compressor_ape->abs_offset,
                            layer->indexer_compressor_ape->type,
                            layer->indexer_compressor_norm->abs_offset,
                            layer->indexer_compressor_norm->type,
                            PULSAR_N_INDEXER_HEAD_DIM, ratio, pos, 0,
                            PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                const bool idx_updated = ok;
                if (ok)
                    ok = gpu_graph_proj_ring_deposit(g, il, pos, ikv, isc, true);
                pulsar_gpu_tensor_free(isc);
                pulsar_gpu_tensor_free(ikv);
                if (!ok)
                    return rollforward_fail(il, pos, idx_updated ? "indexer projection-ring deposit"
                                                                 : "indexer compressor update (row view or kernel)");
            }
        }
        gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) = (pos0 + n_positions) / ratio;
        if (ratio == 4) gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il) = (pos0 + n_positions) / ratio;
    }
    /* L120 value-half: the span is committed across every layer now. */
    for (uint32_t t = 0; t < n_positions; t++)
        gpu_graph_proj_ring_note_pos(g, pos0 + t);
    for (uint32_t t2 = 0; t2 < n_positions; t2++)
        gpu_graph_r128_undo_note_pos(g, pos0 + t2);
    return true;
}

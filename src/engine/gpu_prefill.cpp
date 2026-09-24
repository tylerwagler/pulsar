#include "pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"






pulsar_gpu_tensor *gpu_graph_tensor_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
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
        /* L188: the embed kernel clamps a negative id to 0 -- refuse it here.
         *
         * L216: an image block's slots carry `vocab_size + role` sentinel ids,
         * so the bound is not simply n_vocab.  The roles are 0..IMAGE_END and
         * nothing else may sit at or above vocab_size, which keeps this a real
         * check rather than a hole.  pulsar_session::sync refuses such an id
         * when the request carries no image, so reaching here with one means a
         * merge will fill the row; the embedder zero-masks it until then. */
        if (tokens[i] < 0 ||
            tokens[i] > (int32_t)(PULSAR_N_VOCAB + PULSAR_VISION_ROLE_IMAGE_END)) {
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



/* ============================================================================
 * CSA2 KV production (L218, DeepSeek-V4.1).
 *
 * A kv SOURCE layer (mode FULL) turns each complete group of `ratio` tokens
 * into one compressed row and one index-K row; every other compressing layer
 * reads those through pulsar_layer_attn_layout(il)->kv_source and produces
 * nothing.  The reference's order, kept exactly:
 *
 *   latent  = Compressor(x)                  pre-RoPE, bf16-exact f32 rows
 *   index K = fp4(RoPE(k_norm(wk(latent))))  from the UNROTATED latent
 *   comp KV = fp4(RoPE(latent))              at the group's first position
 *
 * gpu_graph_csa2_emit_rows writes n rows at once from `n` latent rows staged
 * at attn_comp_stage[0, n): the index-K projection first (it needs the
 * unrotated latent), then the rotation in place, then the pack into the
 * source's comp pool.  Bank-aware: `bank` selects the pool and the frontier
 * under a multiseq step (the caller resolved it from the row's seq_id); the
 * single-session graph is bank 0 == the installed views.
 * ============================================================================ */
static bool gpu_graph_csa2_emit_rows(
        pulsar_gpu_graph           *g,
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        uint32_t                    il,
        bool                        banked,
        uint32_t                    bank,
        uint32_t                    n_rows,
        uint32_t                    cache_row0,
        uint32_t                    pos_first,
        uint32_t                    ratio) {
    if (n_rows == 0) return true;
    if (cache_row0 > g->layer_comp_cap[il] || n_rows > g->layer_comp_cap[il] - cache_row0) {
        fprintf(stderr, "pulsar: kv source %u: compressed rows %u+%u exceed the pool cap %u -- refusing\n",
                il, cache_row0, n_rows, g->layer_comp_cap[il]);
        return false;
    }
    if (n_rows > g->attn_comp_stage_cap || n_rows > g->comp_cap) {
        fprintf(stderr, "pulsar: kv source %u: %u latent rows exceed the staging cap %u -- refusing\n",
                il, n_rows, g->attn_comp_stage_cap);
        return false;
    }
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);

    pulsar_gpu_tensor *latent = pulsar_gpu_tensor_view(g->attn_comp_stage, 0,
                                                       (uint64_t)n_rows * PULSAR_N_HEAD_DIM * sizeof(float));
    pulsar_gpu_tensor *comp_dst = banked ? gpu_graph_bank_attn_comp_view(g, il, bank) : g->layer_attn_comp_cache[il];
    /* The POOL's output, before the rotation -- dev's `KVprelatent`.  This is
     * the only point at which it can be observed: everything below mutates
     * `latent` in place, first the tail rope and then the pack's f32 writeback
     * (the pack is handed `latent` as BOTH its read source and its
     * f32-observation output whenever any f32 dump is armed, so it writes the
     * quantised round-trip back over the row it just read).  A dump taken
     * after them is the quantised row, and comparing a quantised row against
     * an unquantised model is what made this compressor look 1.4% wrong. */
    gpu_graph_debug_dump_tensor("KVprelatent", latent,
                                (uint64_t)n_rows * PULSAR_N_HEAD_DIM, il, pos_first);
    /* The projection staging and the index pool are V4.1's alone: under
     * indexer_own_compressor both are the indexer compressor's, and a view
     * opened here would only shadow the buffer it stages through. */
    pulsar_gpu_tensor *idx = g_pulsar_shape.indexer_own_compressor ? NULL
                         : pulsar_gpu_tensor_view(g->idx_comp_stage, 0,
                                                  (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
    pulsar_gpu_tensor *idx_dst = g_pulsar_shape.indexer_own_compressor ? NULL
                              : (banked ? gpu_graph_bank_index_comp_view(g, il, bank) : g->layer_index_comp_cache[il]);
    bool ok = latent && comp_dst && (g_pulsar_shape.indexer_own_compressor || (idx && idx_dst));

    /* index K: wk on the unrotated latent, k_norm, RoPE at the group position,
     * FP4 pack into the source's index-K pool.
     *
     * V4.1 ONLY.  When the indexer owns its compressor (0731) the index-K row is
     * pooled by the indexer's OWN weights over the batch norm -- it is not a
     * projection of the kv source's latent at all -- and gpu_graph_index_comp_*
     * has already written it into the same pool by the time this runs. */
    if (ok && !g_pulsar_shape.indexer_own_compressor) {
        if (ok) ok = gpu_graph_matmul_plain_tensor(idx, model, layer->indexer_k,
                                                   PULSAR_N_HEAD_DIM, PULSAR_N_INDEXER_HEAD_DIM, latent, n_rows) != 0;
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(idx, idx, tensor_map_base(model, layer->indexer_k_norm), tensor_map_size(model, layer->indexer_k_norm),
                                                            layer->indexer_k_norm->abs_offset,
                                                            PULSAR_N_INDEXER_HEAD_DIM, n_rows, PULSAR_RMS_EPS, NULL,
                                                            layer->indexer_k_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) ok = pulsar_gpu_rope_tail_strided_tensor(idx, n_rows, PULSAR_N_INDEXER_HEAD_DIM, PULSAR_N_ROT,
                                                         pos_first, ratio, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                                                         freq_base, freq_scale, ext_factor, attn_factor,
                                                         PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = pulsar_gpu_indexer_fp4_pack_tensor(idx, idx_dst, cache_row0, n_rows,
                                                             PULSAR_N_INDEXER_HEAD_DIM,
                                                             gpu_graph_f32_store_observed_any()) != 0;
        if (ok) gpu_graph_debug_dump_tensor("indexer_KVcompress", idx,
                                            (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM, il, pos_first);
    }

    /* comp KV: the same latent rotated in place, packed into the comp pool */
    if (ok) ok = pulsar_gpu_rope_tail_strided_tensor(latent, n_rows, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
                                                     pos_first, ratio, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                                                     freq_base, freq_scale, ext_factor, attn_factor,
                                                     PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = pulsar_gpu_kv_comp_pack_tensor(gpu_graph_f32_store_observed_any() ? latent : NULL, latent,
                                               comp_dst, cache_row0, n_rows, PULSAR_N_HEAD_DIM) != 0;
    if (ok) gpu_graph_debug_dump_tensor("KVcompress", latent,
                                        (uint64_t)n_rows * PULSAR_N_HEAD_DIM, il, pos_first);
    if (banked) {
        pulsar_gpu_tensor_free(idx_dst);
        pulsar_gpu_tensor_free(comp_dst);
    }
    pulsar_gpu_tensor_free(idx);
    pulsar_gpu_tensor_free(latent);
    return ok;
}



/* ---------------------------------------------------------------------------
 * V4's indexer-OWN compressor (0731).
 *
 * V4.1's index key is a projection of the kv source's latent -- wk + k_norm,
 * fp4-packed in gpu_graph_csa2_emit_rows.  0731 is different at the root: the
 * reference's Indexer builds `Compressor(args, compress_ratio, head_dim = 128,
 * rotate = True)` and `self.compressor(x, start_pos)` pools the index key from
 * its OWN wkv/wgate over the batch norm, with its own ape, its own state lane
 * and its own rope + Hadamard tail.  So one source compresses the same tokens
 * TWICE, and this is the second one.
 *
 * A pooled row lands in the source's index pool at the SAME row index the
 * attention comp row lands at -- one emit, one frontier -- because the scorer
 * reads the index pool with the comp frontier as its row count and stride
 * (the `n_comp` gpu_graph_indexed_attention_span is handed).
 *
 * Both arms below therefore run BESIDE the attention compressor's, on the same
 * rows, and the same helper folds the attention compressor's ape.  The two
 * compressors are the same kernels at a different head dim and with different
 * weights -- that is the whole difference, and it is why the pair can share a
 * frontier without either being derived from the other.
 * ------------------------------------------------------------------------ */
static bool gpu_graph_index_comp_prefill(
        pulsar_gpu_graph           *g,
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        uint32_t                    il,
        bool                        banked,
        uint32_t                    bank,
        uint32_t                    n_tokens,
        uint32_t                    out_row0,
        uint32_t                    pos0,
        uint32_t                    ratio,
        float                       freq_base,
        float                       freq_scale,
        float                       ext_factor,
        float                       attn_factor) {
    const uint32_t width = pulsar_comp_row_width(ratio, PULSAR_N_INDEXER_HEAD_DIM);
    const uint32_t n_groups = n_tokens / ratio;
    pulsar_gpu_tensor *kv = pulsar_gpu_tensor_view(g->batch_index_comp_kv, 0,
                                                   (uint64_t)n_tokens * width * sizeof(float));
    pulsar_gpu_tensor *sc = pulsar_gpu_tensor_view(g->batch_index_comp_sc, 0,
                                                   (uint64_t)n_tokens * width * sizeof(float));
    pulsar_gpu_tensor *latent = pulsar_gpu_tensor_view(g->idx_comp_stage, 0,
                                                       (uint64_t)n_groups * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
    /* The index pool and the indexer's state lane are per-bank under a multiseq
     * step and the installed views otherwise -- the same split the attention
     * lane uses, and the reason this helper takes `bank` at all: a step that
     * mixes banks writes each row to ITS bank's pool. */
    pulsar_gpu_tensor *comp = banked ? gpu_graph_bank_index_comp_view(g, il, bank)
                                     : g->layer_index_comp_cache[il];
    pulsar_gpu_tensor *st_kv = banked ? gpu_graph_bank_index_state_kv_view(g, il, bank)
                                      : g->layer_index_state_kv[il];
    pulsar_gpu_tensor *st_sc = banked ? gpu_graph_bank_index_state_score_view(g, il, bank)
                                      : g->layer_index_state_score[il];
    const bool lanes = kv && sc && latent && comp && st_kv && st_sc;
    if (!lanes) {
        fprintf(stderr, "pulsar: index source %u: %s index lane missing for bank %u "
                        "(batch %s/%s, pool %s, state %s/%s) -- refusing\n",
                il, banked ? "banked" : "classic", bank,
                kv ? "ok" : "MISSING", sc ? "ok" : "MISSING",
                comp ? "ok" : "MISSING", st_kv ? "ok" : "MISSING", st_sc ? "ok" : "MISSING");
    }
    const bool ok = lanes &&
              pulsar_gpu_indexer_compressor_prefill_tensor(
                      comp, latent, st_kv, st_sc, sc, kv,
                      tensor_map_base(model, layer->indexer_compressor_ape), tensor_map_size(model, layer->indexer_compressor_ape),
                      layer->indexer_compressor_ape->abs_offset, layer->indexer_compressor_ape->type,
                      layer->indexer_compressor_norm->abs_offset,
                      layer->indexer_compressor_norm->type,
                      out_row0, PULSAR_N_INDEXER_HEAD_DIM, ratio, pos0, n_tokens,
                      PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                      freq_base, freq_scale, ext_factor, attn_factor,
                      PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                      PULSAR_RMS_EPS) != 0;
    if (ok) gpu_graph_debug_dump_tensor("indexer_KVcompress", latent,
                                        (uint64_t)n_groups * PULSAR_N_INDEXER_HEAD_DIM, il, pos0);
    if (banked) {
        pulsar_gpu_tensor_free(st_sc);
        pulsar_gpu_tensor_free(st_kv);
        pulsar_gpu_tensor_free(comp);
    }
    pulsar_gpu_tensor_free(latent);
    pulsar_gpu_tensor_free(sc);
    pulsar_gpu_tensor_free(kv);
    return ok;
}


/* The per-row twin: one token, through the indexer's own compressor.  `emitted`
 * is 1 exactly when this token closed a group and the row was packed at
 * `out_row`; the caller advances the shared frontier on that, not on the return
 * value. */
static bool gpu_graph_index_comp_update(
        pulsar_gpu_graph           *g,
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        uint32_t                    il,
        bool                        banked,
        uint32_t                    bank,      /* which BANK this row belongs to */
        uint32_t                    row,       /* which BATCH ROW to read */
        uint32_t                    pos,       /* that row's absolute POSITION */
        uint32_t                    out_row,
        uint32_t                    ratio,
        float                       freq_base,
        float                       freq_scale,
        float                       ext_factor,
        float                       attn_factor,
        int                        *emitted) {
    const uint32_t width = pulsar_comp_row_width(ratio, PULSAR_N_INDEXER_HEAD_DIM);
    pulsar_gpu_tensor *kv = gpu_graph_tensor_row_view(g->batch_index_comp_kv, row, width);
    pulsar_gpu_tensor *sc = gpu_graph_tensor_row_view(g->batch_index_comp_sc, row, width);
    pulsar_gpu_tensor *latent = pulsar_gpu_tensor_view(g->idx_comp_stage, 0,
                                                       (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
    /* Per-row twin of the prefill arm above: BOTH banks the row can belong to
     * and the row's own batch slot are per-bank, and the caller's `bank` may
     * differ from the installed one on every token of a mixed step. */
    pulsar_gpu_tensor *comp = banked ? gpu_graph_bank_index_comp_view(g, il, bank)
                                     : g->layer_index_comp_cache[il];
    pulsar_gpu_tensor *st_kv = banked ? gpu_graph_bank_index_state_kv_view(g, il, bank)
                                      : g->layer_index_state_kv[il];
    pulsar_gpu_tensor *st_sc = banked ? gpu_graph_bank_index_state_score_view(g, il, bank)
                                      : g->layer_index_state_score[il];
    const bool lanes = kv && sc && latent && comp && st_kv && st_sc;
    if (!lanes) {
        fprintf(stderr, "pulsar: index source %u: %s index lane missing for bank %u "
                        "(row %u, pool %s, state %s/%s) -- refusing\n",
                il, banked ? "banked" : "classic", bank, row,
                comp ? "ok" : "MISSING", st_kv ? "ok" : "MISSING", st_sc ? "ok" : "MISSING");
    }
    const bool ok = lanes &&
              pulsar_gpu_indexer_compressor_update_tensor(
                      comp, latent, st_kv, st_sc, sc, kv,
                      tensor_map_base(model, layer->indexer_compressor_ape), tensor_map_size(model, layer->indexer_compressor_ape),
                      layer->indexer_compressor_ape->abs_offset, layer->indexer_compressor_ape->type,
                      layer->indexer_compressor_norm->abs_offset,
                      layer->indexer_compressor_norm->type,
                      out_row, PULSAR_N_INDEXER_HEAD_DIM, ratio, pos,
                      PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                      freq_base, freq_scale, ext_factor, attn_factor,
                      PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                      PULSAR_RMS_EPS, emitted) != 0;
    if (banked) {
        pulsar_gpu_tensor_free(st_sc);
        pulsar_gpu_tensor_free(st_kv);
        pulsar_gpu_tensor_free(comp);
    }
    pulsar_gpu_tensor_free(latent);
    pulsar_gpu_tensor_free(sc);
    pulsar_gpu_tensor_free(kv);
    return ok;
}


/* The 0731 compressor folds an absolute-position embedding into its score
 * before pooling -- the reference's prefill branch does `score = score + self.ape`
 * and its decode branch `score += self.ape[start_pos % ratio]` -- while V4.1's
 * plain projections carry none, which is what `compressor_ape` records.  Both
 * arms go through here so the batch path and the per-row path fold the same ape
 * the same way; the reference's two branches agree and so must ours, or a chunk
 * boundary would change the answer. */
static bool gpu_graph_comp_ape_fold(
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        pulsar_gpu_tensor          *sc,
        uint32_t                    width,
        uint32_t                    ratio,
        uint32_t                    pos0,
        uint32_t                    n_tokens) {
    if (!g_pulsar_shape.compressor_ape) return true;
    const pulsar_tensor *ape = layer->attn_compressor_ape;
    if (!ape) {
        fprintf(stderr, "pulsar: the 0731 compressor needs its absolute-position embedding -- refusing\n");
        return false;
    }
    return pulsar_gpu_csa2_comp_ape_add_tensor(sc, tensor_map_base(model, ape), tensor_map_size(model, ape),
                                               ape->abs_offset, ape->type,
                                               width, ratio, pos0, n_tokens) != 0;
}

/* The per-position twin of the fold above: the store adds the ape as it writes
 * the lane (pulsar_gpu_csa2_ape), so the row buffer is never modified and no
 * separate launch runs -- the pre-CSA2 store's shape (L239).  Returns false only
 * when the profile needs an ape and the layer has none; *out is NULL when the
 * profile has no ape (V4.1). */
static bool gpu_graph_comp_ape_desc(
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        pulsar_gpu_csa2_ape        *desc,
        const pulsar_gpu_csa2_ape **out) {
    *out = NULL;
    if (!g_pulsar_shape.compressor_ape) return true;
    const pulsar_tensor *ape = layer->attn_compressor_ape;
    if (!ape) {
        fprintf(stderr, "pulsar: the 0731 compressor needs its absolute-position embedding -- refusing\n");
        return false;
    }
    desc->model_map = tensor_map_base(model, ape);
    desc->model_size = tensor_map_size(model, ape);
    desc->offset = ape->abs_offset;
    desc->type = ape->type;
    *out = desc;
    return true;
}


/* Run kv source `il`'s compressor over this batch's rows (batch_comp_kv/sc
 * hold the kv / score projections of every row) and emit what completes.
 * Three arms, all bit-equivalent by construction:
 *   - aligned run: pos0 on a group boundary, one bank -- one batched pool over
 *     the whole run, the trailing partial group left in the state;
 *   - per row: anything else (an unaligned start finishing a pending group, or
 *     a multiseq step mixing banks) -- store each row, emit at each boundary.
 * A zero-prefix chunk is the aligned run at row 0.  comp_counts[t] receives
 * the compressed rows visible to row t after its own emit, (pos+1)/ratio. */
static bool gpu_graph_csa2_produce(
        pulsar_gpu_graph           *g,
        const pulsar_model         *model,
        const pulsar_layer_weights *layer,
        uint32_t                    il,
        uint32_t                    pos0,
        uint32_t                    n_tokens,
        bool                        mseq,
        uint32_t                   *comp_counts) {
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    const bool has_state = ratio > 1u;
    /* The compressor's projection rows are coff wide (2*head_dim where the layer
     * overlaps) while the latent it emits stays head_dim.  Every stride and size
     * below is one or the other -- mixing them reads a neighbouring row and
     * still looks like a number. */
    const uint32_t comp_width = pulsar_comp_row_width(ratio, PULSAR_N_HEAD_DIM);
    /* V4's indexer compresses its OWN index key (the helpers above), so this
     * source runs a second compression over the same rows; V4.1's index key is a
     * projection of the latent and needs no second pass. */
    const bool own_index = g_pulsar_shape.indexer_own_compressor &&
                           pulsar_attn_runs_indexer(pulsar_layer_attn_layout(il)->mode);
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    /* A banked graph has the indexer's lane too (the slab carries iskv/issc,
     * and gpu_graph_bank_index_state_*_view hands back the active bank's), so
     * the two index helpers below take the bank and select it themselves.  The
     * refusal that used to stand here -- "the slab carries no index twin" --
     * was true until L218 s124 and is deleted with the premise. */
    /* Stage-B save: keep this batch's per-position compressor projections so
     * a partial spec accept can roll the pending group forward without a
     * transformer replay (gpu_graph_dspark_compressor_rollforward), and a
     * rewind inside the round can rebuild a pending slot.  Ratio 1 keeps no
     * state, so there is nothing to roll. */
    if (has_state && g->spec_comp_save_n && g->spec_comp_kv_save[il]) {
        uint32_t sn = g->spec_comp_save_n;
        if (sn > n_tokens) sn = n_tokens;
        if (sn > PULSAR_SPEC_LOGITS_ROWS + 1u) sn = PULSAR_SPEC_LOGITS_ROWS + 1u;
        const uint64_t sb = (uint64_t)sn * comp_width * sizeof(float);
        /* ASYNC: read only by the rollforward's kernels on the same stream. */
        if (pulsar_gpu_tensor_copy_async(g->spec_comp_kv_save[il], 0, g->batch_comp_kv, 0, sb) == 0 ||
            pulsar_gpu_tensor_copy_async(g->spec_comp_sc_save[il], 0, g->batch_comp_sc, 0, sb) == 0)
            return false;
        /* V4's indexer owns a second lane over the same rows and needs its own
         * save for the same reason (see the rollforward).  Both are RAW
         * projections: the ape each compressor folds before its store is
         * applied by the rollforward, exactly as the live path applies it. */
        if (own_index) {
            const uint32_t iw = pulsar_comp_row_width(ratio, PULSAR_N_INDEXER_HEAD_DIM);
            const uint64_t ib = (uint64_t)sn * iw * sizeof(float);
            if (!g->spec_icomp_kv_save[il] || !g->spec_icomp_sc_save[il] ||
                pulsar_gpu_tensor_copy_async(g->spec_icomp_kv_save[il], 0, g->batch_index_comp_kv, 0, ib) == 0 ||
                pulsar_gpu_tensor_copy_async(g->spec_icomp_sc_save[il], 0, g->batch_index_comp_sc, 0, ib) == 0)
                return false;
        }
    }
    const uint32_t run_bank = mseq ? (uint32_t)g->ms_seq_id[0] : gpu_graph_cur_bank(g);
    const bool one_bank = !mseq || (uint32_t)g->ms_seq_id[n_tokens - 1u] == run_bank;
    const bool aligned = one_bank && (pos0 % ratio) == 0u;
    bool ok = true;
    if (aligned) {
        /* Banked state lanes are OWNED views and must be freed; the single-
         * session ones are borrowed. */
        pulsar_gpu_tensor *st_kv = NULL, *st_sc = NULL;
        if (has_state) {
            if (mseq) {
                st_kv = gpu_graph_bank_attn_state_kv_view(g, il, run_bank);
                st_sc = gpu_graph_bank_attn_state_score_view(g, il, run_bank);
                ok = st_kv && st_sc;
            } else {
                st_kv = g->layer_attn_state_kv[il];
                st_sc = g->layer_attn_state_score[il];
            }
        }
        const uint32_t before = g->ms_n_comp[run_bank][il];
        const uint32_t n_groups = n_tokens / ratio;
        /* a run starting on a group boundary rebuilds the state from scratch */
        g->ms_comp_state_stale[run_bank] = false;
        if (ok && before != pos0 / ratio) {
            fprintf(stderr, "pulsar: kv source %u bank %u: frontier %u is not position-true at %u (ratio %u) -- refusing\n",
                    il, run_bank, before, pos0, ratio);
            ok = false;
        }
        if (ok && n_groups > g->attn_comp_stage_cap) {
            fprintf(stderr, "pulsar: kv source %u: %u groups exceed the staging cap %u -- refusing\n",
                    il, n_groups, g->attn_comp_stage_cap);
            ok = false;
        }
        if (ok) ok = gpu_graph_comp_ape_fold(model, layer, g->batch_comp_sc, comp_width, ratio, pos0, n_tokens);
        if (ok) ok = pulsar_gpu_csa2_compressor_prefill_tensor(g->attn_comp_stage, g->batch_comp_kv, g->batch_comp_sc,
                                                               st_kv, st_sc, tensor_map_base(model, layer->attn_compressor_norm), tensor_map_size(model, layer->attn_compressor_norm),
                                                               layer->attn_compressor_norm->abs_offset,
                                                               layer->attn_compressor_norm->type,
                                                               PULSAR_N_HEAD_DIM, ratio, pos0, n_tokens,
                                                               PULSAR_RMS_EPS) != 0;
        /* The indexer's own compression, before emit_rows: it needs this batch's
         * index projections and writes the index pool emit_rows would otherwise
         * fill from the latent -- and unlike the comp row it has no latent of its
         * own to hand over, so it must run here rather than inside that call. */
        if (ok && own_index) ok = gpu_graph_index_comp_prefill(g, model, layer, il, mseq, run_bank,
                                                               n_tokens, before, pos0,
                                                               ratio, freq_base, freq_scale, ext_factor, attn_factor);
        if (ok) ok = gpu_graph_csa2_emit_rows(g, model, layer, il, mseq, run_bank, n_groups, before, pos0, ratio);
        /* plan-33 inc C: the partial-fork boundary row -- byte-restore it over
         * whatever the emit just recomputed.  Both lanes: one emit writes the
         * comp row and (V4.1) the index-K row, and V4's own index compressor
         * wrote its row before this call. */
        if (ok) ok = gpu_graph_emit_keep_restore(g, il, run_bank, before, n_groups, false);
        if (ok && own_index) ok = gpu_graph_emit_keep_restore(g, il, run_bank, before, n_groups, true);
        /* L120 value half: the rows both stores just consumed go into the ring,
         * which is what a later rewind replays to rebuild the overlap's carry. */
        if (ok) ok = gpu_graph_proj_ring_deposit(g, il, pos0, 0u, n_tokens);
        if (ok) {
            g->ms_n_comp[run_bank][il] = before + n_groups;
            for (uint32_t t = 0; t < n_tokens; t++) comp_counts[t] = (pos0 + t + 1u) / ratio;
            if (has_state) {
                const uint64_t lane_floats = (uint64_t)pulsar_comp_state_rows(ratio) * comp_width;
                gpu_graph_debug_dump_tensor("attn_state_kv", st_kv, lane_floats, il, pos0);
                gpu_graph_debug_dump_tensor("attn_state_score", st_sc, lane_floats, il, pos0);
            }
        }
        if (mseq) {
            pulsar_gpu_tensor_free(st_sc);
            pulsar_gpu_tensor_free(st_kv);
        }
        return ok;
    }
    /* Per-row: row t belongs to bank ms_seq_id[t] at position ms_positions[t]
     * (or to the current bank at pos0 + t); the store lands in THAT bank's
     * pending slot and an emit lands at ITS frontier.  Per-bank groups are
     * independent, and the shared staging row is safe across banks because
     * each emit packs it before the next row's kernels run on the stream. */
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
        const uint32_t bank = mseq ? (uint32_t)g->ms_seq_id[t] : gpu_graph_cur_bank(g);
        uint32_t *const n_comp_slot = &g->ms_n_comp[bank][il];
        pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
        pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
        pulsar_gpu_tensor *st_kv = NULL, *st_sc = NULL;
        if (has_state) {
            st_kv = mseq ? gpu_graph_bank_attn_state_kv_view(g, il, bank) : g->layer_attn_state_kv[il];
            st_sc = mseq ? gpu_graph_bank_attn_state_score_view(g, il, bank) : g->layer_attn_state_score[il];
        }
        pulsar_gpu_tensor *latent_row = pulsar_gpu_tensor_view(g->attn_comp_stage, 0,
                                                               (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
        int emitted = 0;
        if (has_state) {
            /* A stale pending group (rewound mid-group past the verify saves)
             * can only be joined at a group boundary; a store elsewhere would
             * pool a wrong token in. */
            if (pos % ratio == 0u) g->ms_comp_state_stale[bank] = false;
            else if (g->ms_comp_state_stale[bank]) {
                fprintf(stderr, "pulsar: kv source %u bank %u: store at %u would extend a stale pending group -- refusing\n",
                        il, bank, pos);
                ok = false;
            }
        }
        pulsar_gpu_csa2_ape ape_desc;
        const pulsar_gpu_csa2_ape *ape = NULL;
        if (ok) ok = gpu_graph_comp_ape_desc(model, layer, &ape_desc, &ape);
        ok = ok && kv_view && sc_view && latent_row && (!has_state || (st_kv && st_sc)) &&
             pulsar_gpu_csa2_compressor_update_tensor(latent_row, kv_view, sc_view, st_kv, st_sc,
                                                      tensor_map_base(model, layer->attn_compressor_norm), tensor_map_size(model, layer->attn_compressor_norm),
                                                      layer->attn_compressor_norm->abs_offset,
                                                      layer->attn_compressor_norm->type, ape,
                                                      PULSAR_N_HEAD_DIM, ratio, pos, PULSAR_RMS_EPS, &emitted) != 0;
        /* The indexer's own compressor walks the SAME rows on the same schedule:
         * it stores this token whether or not the group closes, and its frontier
         * IS the attention compressor's (one emit, one row -- the scorer reads
         * the index pool with the comp frontier as its stride).  So it is called
         * unconditionally and the two `emitted` flags are compared: a disagreement
         * would mean the second pool's row indices had drifted off the first's,
         * which no later check would notice. */
        int idx_emitted = 0;
        if (ok && own_index) ok = gpu_graph_index_comp_update(g, model, layer, il, mseq, bank, t, pos,
                                                              *n_comp_slot, ratio,
                                                              freq_base, freq_scale, ext_factor, attn_factor,
                                                              &idx_emitted);
        if (ok && own_index && (idx_emitted != 0) != (emitted != 0)) {
            fprintf(stderr, "pulsar: index source %u: the indexer's compressor and the attention's "
                            "disagree about the group boundary at %u (ratio %u) -- refusing\n", il, pos, ratio);
            ok = false;
        }
        /* L120 value half: this row's own slot in the ring, after both stores. */
        if (ok) ok = gpu_graph_proj_ring_deposit(g, il, pos, t, 1u);
        if (ok && emitted) {
            const uint32_t row = *n_comp_slot;
            if (row != pos / ratio) {
                fprintf(stderr, "pulsar: kv source %u bank %u: frontier %u is not position-true at %u (ratio %u) -- refusing\n",
                        il, bank, row, pos, ratio);
                ok = false;
            }
            if (ok) ok = gpu_graph_csa2_emit_rows(g, model, layer, il, mseq, bank, 1u, row, pos + 1u - ratio, ratio);
            /* plan-33 inc C: same boundary-row restore as the batched arm. */
            if (ok) ok = gpu_graph_emit_keep_restore(g, il, bank, row, 1u, false);
            if (ok && own_index) ok = gpu_graph_emit_keep_restore(g, il, bank, row, 1u, true);
            if (ok) (*n_comp_slot)++;
        }
        if (ok) comp_counts[t] = *n_comp_slot;
        pulsar_gpu_tensor_free(latent_row);
        if (mseq) {
            pulsar_gpu_tensor_free(st_sc);
            pulsar_gpu_tensor_free(st_kv);
        }
        pulsar_gpu_tensor_free(sc_view);
        pulsar_gpu_tensor_free(kv_view);
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
        pulsar_gpu_graph     *g,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    if (!g->prefill_tokens) {
        fprintf(stderr, "pulsar: prompt embedding gather needs the device token tensor "
                        "(gpu_graph_upload_prompt_tokens first) -- refusing\n");
        return false;
    }
    if (!pulsar_gpu_embed_tokens_hc_tensor(g->batch_cur_hc,
                                          g->prefill_tokens,
                                          tensor_map_base(model, weights->token_embd),
                                          tensor_map_size(model, weights->token_embd),
                                          weights->token_embd->abs_offset,
                                          (uint32_t)weights->token_embd->dim[1],
                                          n_tokens,
                                          PULSAR_N_EMBD,
                                          PULSAR_N_HC)) return false;
    /* the pre-mix is born with the stream: layer 0's attention collapses with
     * the identity (make_identity_pre_mix), every later sublayer with what the
     * one before it derived */
    return pulsar_gpu_hc_pre_identity_tensor(g->batch_hc_pre, n_tokens, PULSAR_N_HC) != 0;
}



/* Scatter a MERGED image span into the HC carrier.
 *
 * vision_merge_span() produces one bf16 embedding per span position on the HOST
 * (the sentinel's learned vector, or an aligner row for an IMAGE slot).  The
 * reference writes that block into h BEFORE expanding to hc_mult copies --
 * `merge_image_embeddings` runs, then `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`
 * -- so every HC stream carries the SAME merged row.  That is what this
 * replicates: n_hc copies of one n_embd row, not a broadcast the downstream
 * kernels would have to know about.
 *
 * The rows are raw bf16 bits (pulsar_hc_t is __nv_bfloat16, 2 bytes), so the
 * copies are memcpys and no numerics live here.  The carrier layout is
 * [token][hc][embd], contiguous per token, so one tensor_write covers the span.
 *
 * Returns false, without writing, if the span does not fit the carrier. */
bool gpu_graph_write_vision_span(
        pulsar_gpu_tensor *out_hc,
        const uint16_t    *rows,      /* n_rows * PULSAR_N_EMBD bf16 bits */
        uint32_t           n_rows,
        uint32_t           row0,      /* the span's first row within the chunk */
        uint32_t           n_tokens) {
    if (!out_hc || !rows || n_rows == 0) return false;
    if (row0 > n_tokens || n_rows > n_tokens - row0) return false;
    const uint64_t row_elt = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const size_t per_row = (size_t)row_elt * PULSAR_HC_ELT_SIZE;
    if (pulsar_gpu_tensor_bytes(out_hc) < (uint64_t)n_tokens * per_row) return false;

    uint16_t *stage = (uint16_t *)malloc((size_t)n_rows * per_row);
    if (!stage) return false;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint16_t *src = rows + (size_t)r * PULSAR_N_EMBD;
        uint16_t *dst = (uint16_t *)(void *)((char *)stage + (size_t)r * per_row);
        for (uint32_t h = 0; h < PULSAR_N_HC; h++)
            memcpy(dst + (size_t)h * PULSAR_N_EMBD, src,
                   (size_t)PULSAR_N_EMBD * PULSAR_HC_ELT_SIZE);
    }
    const bool ok = pulsar_gpu_tensor_write(out_hc, (uint64_t)row0 * per_row, stage,
                                            (uint64_t)n_rows * per_row) != 0;
    free(stage);
    return ok;
}



/* The reference's image-preprocessing args.  Patch size and downsample ratio are
 * tower dims the binder already validates against the tensors; the other three
 * are the checkpoint's policy constants (see PULSAR_VISION_MAX_N_TOKEN).  Built
 * here rather than at each call so there is one place that knows the mapping. */
static void vision_default_args(pulsar_vision_args *a) {
    a->patch_size       = (int)PULSAR_VISION_PATCH;
    a->downsample_ratio = (int)PULSAR_VISION_DOWNSAMPLE;
    a->max_n_token      = PULSAR_VISION_MAX_N_TOKEN;
    a->min_pixels       = PULSAR_VISION_MIN_PIXELS;
    a->max_wh_ratio     = PULSAR_VISION_MAX_WH_RATIO;
}

/* Decode + preprocess + encode + scatter, for every image span inside the chunk.
 *
 * The span EXTENT comes from the prompt's own sentinel ids (vision_span_extent),
 * not from the image, so an image whose span is not actually in the prompt is a
 * refusal rather than a silently misplaced block.  An image whose span lies in
 * another chunk is skipped: the chunk planner has already refused any request
 * that would split one, so this only ever skips images that a later chunk owns.
 *
 * The reference does exactly this merge between `h = self.embed(input_ids)` and
 * the first layer, which is why the caller runs it right after the embedding
 * gather. */
/* Timed at the caller-visible boundary: this is the ViT encode + scatter a
 * prepared-span/merged-row cache would remove on a repeated image (L226). */
bool gpu_graph_merge_image_spans(pulsar_gpu_tensor *out_hc, const pulsar_model *model,
                                 const int32_t *ids, int n_ids,
                                 const pulsar_vision_request *vr,
                                 uint32_t pos0, uint32_t n_tokens) {
    if (!vr || vr->n_images <= 0) return true;      /* text-only: nothing to do */
    if (!out_hc || !model || !vr->weights || !ids || n_ids <= 0) return false;

    pulsar_vision_args args;
    vision_default_args(&args);

    for (int i = 0; i < vr->n_images; i++) {
        const pulsar_image_ref *img = &vr->images[i];
        if (!img->bytes || img->len == 0 || img->start_pos < 0) return false;
        int span_len = 0;
        if (!vision_span_extent(ids, n_ids, (int)PULSAR_N_VOCAB, img->start_pos, &span_len)) {
            fprintf(stderr, "pulsar: image %d claims a span at token %d that the prompt does not "
                            "carry (no IMAGE_START sentinel there, or no IMAGE_END after it)\n",
                    i, img->start_pos);
            return false;
        }
        const uint32_t s0 = (uint32_t)img->start_pos;
        if (s0 < pos0 || s0 + (uint32_t)span_len > pos0 + n_tokens) continue;   /* another chunk */

        pulsar_vision_prepared prep = {};
        const double vp_t0 = now_sec();
        if (!vision_prepare_image(img->bytes, img->len, &args, (int)s0, (int)PULSAR_N_VOCAB, &prep)) {
            fprintf(stderr, "pulsar: image %d at token %d failed to decode/preprocess\n", i, img->start_pos);
            return false;
        }
        const double vp_prep_ms = (now_sec() - vp_t0) * 1000.0;
        const int cap = prep.span_len * (int)PULSAR_N_EMBD;
        uint16_t *rows = (uint16_t *)malloc((size_t)cap * sizeof(uint16_t));
        if (!rows) { vision_prepared_free(&prep); return false; }
        int n_rows = 0;
        int cache_hit = 0;
        const double vp_enc_t0 = now_sec();
        const bool merged = vision_merge_span_cached(vr->weights, model, &prep,
                                                     img->bytes, img->len, &args,
                                                     rows, cap, &n_rows, &cache_hit) &&
                            n_rows == prep.span_len &&
                            gpu_graph_write_vision_span(out_hc, rows, (uint32_t)n_rows,
                                                        s0 - pos0, n_tokens);
        free(rows);
        fprintf(stderr, "pulsar: image %d: decode+preprocess %.1f ms, %s %.1f ms (%d span rows)\n",
                i, vp_prep_ms,
                cache_hit ? "CACHED encode (tower skipped)" : "ViT encode+scatter",
                (now_sec() - vp_enc_t0) * 1000.0, prep.span_len);
        vision_prepared_free(&prep);
        if (!merged) {
            fprintf(stderr, "pulsar: image %d at token %d failed to merge (span %d rows)\n",
                    i, img->start_pos, prep.span_len);
            return false;
        }
    }
    return true;
}



/* Per-chunk image-span VISIBILITY, computed once and uploaded once for the
 * chunk's attention launches.
 *
 * The reference makes an image span the ONE bidirectional region: a query
 * inside [IMAGE_START, IMAGE_END] sees `left` back to the span's first slot and
 * `right` forward to its end (get_image_visible), on top of the sliding window.
 * The kernel cannot express that forward reach from the window alone, so it
 * takes the two counts.  The arrays are stored left-half then right-half in one
 * tensor (`prefill_cap` int32 each) so there is one allocation and one layout.
 *
 * A TEXT chunk -- and every later chunk of an image request, which is all text --
 * matches no sentinel and leaves vision_visible_tokens at 0.  That is what keeps
 * the text prefill bit-identical: the attention entries then receive NULL
 * pointers and the kernel runs its pre-L216 plan literally. */
bool gpu_graph_upload_vision_visible(pulsar_gpu_graph *g, const int32_t *ids,
                                     int n_ids, uint32_t start, uint32_t n_tokens) {
    if (!g || !g->vision_visible) return false;
    g->vision_visible_tokens = 0;
    if (!g->vision_req || g->vision_req->n_images <= 0) return true;   /* text: nothing to compute */
    if (!ids || n_ids <= 0 || start > (uint32_t)n_ids ||
        n_tokens > (uint32_t)n_ids - start || n_tokens > g->prefill_cap) {
        return false;
    }
    const int32_t *chunk = ids + start;
    int has_span = 0;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (chunk[i] >= (int32_t)PULSAR_N_VOCAB) { has_span = 1; break; }
    }
    if (!has_span) return true;   /* this chunk owns no sentinel: NULL path */

    int32_t *host = (int32_t *)xmalloc(2ull * n_tokens * sizeof(int32_t));
    if (!host) return false;
    int32_t *left = host, *right = host + n_tokens;
    vision_image_visible(chunk, (int)n_tokens, (int)PULSAR_N_VOCAB,
                         PULSAR_VISION_MAX_N_TOKEN, left, right);

    /* The widest raw span any query now reaches is [q - left[q], q + right[q]];
     * its length is left+right+1, at least the sliding window.  The kernel's
     * raw-row scratch covers the reference's own clamps (AF16_RAWROWS in
     * pulsar_cuda_attn_f16.cu derives from PULSAR_VISION_MAX_N_TOKEN), so the
     * only remaining bound is the ring itself.  A span the ring cannot hold
     * cannot be attended in one pass: refuse rather than clip it, because a
     * clipped span is a silently different (causal-only) answer. */
    uint32_t widest = 0;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const int64_t w = (int64_t)left[i] + (int64_t)right[i] + 1;
        if (w > (int64_t)widest) widest = (uint32_t)w;
    }
    if (widest > g->raw_cap) {
        fprintf(stderr, "pulsar: image visibility at token %u needs %u raw rows but the raw ring "
                        "holds %u -- refusing (an image span must be attendable in one pass)\n",
                start, widest, g->raw_cap);
        free(host);
        return false;
    }

    const uint64_t half = (uint64_t)g->prefill_cap * sizeof(int32_t);
    const bool ok = pulsar_gpu_tensor_write(g->vision_visible, 0, left,
                                            (uint64_t)n_tokens * sizeof(int32_t)) != 0 &&
                    pulsar_gpu_tensor_write(g->vision_visible, half, right,
                                            (uint64_t)n_tokens * sizeof(int32_t)) != 0;
    free(host);
    if (!ok) return false;
    g->vision_visible_tokens = n_tokens;
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: image-span visibility armed for prefill attention "
                        "(%u-token chunk, widest reach %u raw rows)\n", n_tokens, widest);
    }
    return true;
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
















/* CSA2 (L218): what happens between an index source's scores and its top-k,
 * for batch rows [row0, row0 + n) whose scores sit at g->indexer_scores rows
 * [0, n): the candidate source publishes the block mask for those rows (from
 * its unmasked scores), a later index source scores only inside the published
 * mask.  `positions` (mseq) or pos0 + row give each row's reach. */
static bool gpu_graph_csa2_candidates(pulsar_gpu_graph *g, const pulsar_layer_attn *attn, uint32_t il,
                                      uint32_t row0, uint32_t n, uint32_t n_comp, uint32_t ratio,
                                      uint32_t pos0, const pulsar_gpu_tensor *positions) {
    if (!attn->candidate_source && !attn->uses_candidates) return true;
    const uint32_t mw = g->cand_mask_words;
    pulsar_gpu_tensor *mask = pulsar_gpu_tensor_view(g->cand_mask, (uint64_t)row0 * mw * sizeof(uint32_t),
                                                     (uint64_t)n * mw * sizeof(uint32_t));
    if (!mask) return false;
    bool ok = true;
    if (attn->candidate_source) {
        ok = pulsar_gpu_candidate_blocks_tensor(mask, g->cand_bscore, g->indexer_scores, n_comp, n, mw,
                                                PULSAR_CANDIDATE_BLOCK_SIZE, PULSAR_CANDIDATE_TOPK_BLOCKS,
                                                pos0, ratio, positions) != 0;
        if (ok) gpu_graph_debug_dump_i32_tensor("candidate_mask", mask, (uint64_t)n * mw, il, pos0);
    } else {
        ok = pulsar_gpu_candidate_mask_scores_tensor(g->indexer_scores, mask, n_comp, n, mw,
                                                     PULSAR_CANDIDATE_BLOCK_SIZE) != 0;
    }
    pulsar_gpu_tensor_free(mask);
    return ok;
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
        const pulsar_layer_attn    *attn,
        uint32_t                    s0,
        uint32_t                    sn,
        uint32_t                    spos0,
        uint64_t                    q_dim,
        uint32_t                    n_head,      ///< the OWNED heads (slice 4g)
        uint64_t                    sinks_off,   ///< the first owned head's sink
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
            (uint64_t)s0 * PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX),
            (uint64_t)sn * PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX));
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
    /* CSA2: the selection lives at ABSOLUTE batch rows so the index source's
     * REUSE members find it; an index source writes it here, a REUSE layer
     * reads it and runs no indexer. */
    pulsar_gpu_tensor *sel_view = pulsar_gpu_tensor_view(g->comp_selected,
            (uint64_t)s0 * PULSAR_N_INDEXER_TOP_K * sizeof(uint32_t),
            (uint64_t)sn * PULSAR_N_INDEXER_TOP_K * sizeof(uint32_t));
    /* FULL / REINDEX score and select here; REUSE reads the selection its index
     * source wrote.  The unindexed and window modes never reach this span. */
    const bool selects = pulsar_attn_runs_indexer(attn->mode);
    /* L216 image-span visibility for THIS span's rows, offset by s0 into the
     * chunk's arrays: the kernel indexes the counts by its own token axis, the
     * same axis positions uses.  A text chunk leaves vision_visible_tokens at 0
     * and passes NULL (the pre-L216 path); a banked multiseq span never carries
     * vision. */
    pulsar_gpu_tensor *vleft_view = NULL, *vright_view = NULL;
    if (!op->mseq && g->vision_visible_tokens != 0u) {
        vleft_view = pulsar_gpu_tensor_view(g->vision_visible,
                (uint64_t)s0 * sizeof(int32_t), (uint64_t)sn * sizeof(int32_t));
        vright_view = pulsar_gpu_tensor_view(g->vision_visible,
                ((uint64_t)g->prefill_cap + s0) * sizeof(int32_t),
                (uint64_t)sn * sizeof(int32_t));
    }
    bool ok = iq_view && iw_view && sq_view && sh_view && sel_view &&
              (!op->mseq || (sp_view && ss_view)) &&
              (!g->vision_visible_tokens || op->mseq || (vleft_view && vright_view));

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
    if (ok && selects && op->mseq) {
        for (uint32_t r0 = 0; ok && r0 < sn; ) {
            uint32_t rn = 1;
            while (r0 + rn < sn &&
                   g->ms_seq_id[s0 + r0 + rn] == g->ms_seq_id[s0 + r0]) rn++;
            const uint32_t bank  = (uint32_t)g->ms_seq_id[s0 + r0];
            const uint32_t rpos0 = (uint32_t)g->ms_positions[s0 + r0];
            pulsar_gpu_tensor *rq = pulsar_gpu_tensor_view(g->batch_indexer_qp,
                    (uint64_t)(s0 + r0) * PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX),
                    (uint64_t)rn * PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX));
            pulsar_gpu_tensor *rw = pulsar_gpu_tensor_view(g->batch_indexer_weights,
                    (uint64_t)(s0 + r0) * PULSAR_N_INDEXER_HEAD * sizeof(float),
                    (uint64_t)rn * PULSAR_N_INDEXER_HEAD * sizeof(float));
            pulsar_gpu_tensor *rs = pulsar_gpu_tensor_view(g->indexer_scores,
                    (uint64_t)r0 * n_comp * sizeof(float),
                    (uint64_t)rn * n_comp * sizeof(float));
            pulsar_gpu_tensor *rb = gpu_graph_bank_index_comp_view(g, attn->kv_source, bank);   /* the source's index K */
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
    } else if (ok && selects) {
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
    if (ok && selects) {
        gpu_graph_debug_dump_tensor("indexer_scores", g->indexer_scores,
                                      (uint64_t)n_comp * sn, il, spos0);
        ok = gpu_graph_csa2_candidates(g, attn, il, s0, sn, n_comp, ratio, spos0, sp_view);
    }
    if (ok && selects) {
        ok = pulsar_gpu_indexer_topk_tensor(sel_view,
                                           g->indexer_scores,
                                           n_comp,
                                           sn,
                                           PULSAR_N_INDEXER_TOP_K) != 0;
        if (ok) {
            gpu_graph_debug_dump_i32_tensor("indexer_topk", sel_view,
                                              (uint64_t)sn * PULSAR_N_INDEXER_TOP_K, il, spos0);
        }
    }
    if (ok) {
        ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(sh_view,
                                                                  tensor_map_base(model, layer->attn_sinks),
                                                                  tensor_map_size(model, layer->attn_sinks),
                                                                  sinks_off,
                                                                  sq_view,
                                                                  op->raw_src,
                                                                  op->comp_src,
                                                                  sel_view,
                                                                  sn,
                                                                  spos0,
                                                                  n_raw,
                                                                  g->raw_cap,
                                                                  raw_start,
                                                                  n_comp,
                                                                  PULSAR_N_INDEXER_TOP_K,
                                                                  g->raw_window,
                                                                  ratio,
                                                                  n_head,
                                                                  PULSAR_N_HEAD_DIM,
                                                                  sp_view, ss_view,
                                                                  op->comp_bases,
                                                                  op->comp_cap,
                                                                  op->n_banks,
                                          g->q_prep_active ? &g->q_prep : NULL,
                                          vleft_view, vright_view) != 0;
    }
    pulsar_gpu_tensor_free(sel_view);
    pulsar_gpu_tensor_free(vright_view);
    pulsar_gpu_tensor_free(vleft_view);
    pulsar_gpu_tensor_free(ss_view);
    pulsar_gpu_tensor_free(sp_view);
    pulsar_gpu_tensor_free(sh_view);
    pulsar_gpu_tensor_free(sq_view);
    pulsar_gpu_tensor_free(iw_view);
    pulsar_gpu_tensor_free(iq_view);
    return ok;
}


/* Slice 4g (L241): gather the owned output groups' `low` rows into the full
 * [tokens][n_groups_total * rank] block on every rank, between the attention
 * output's stage 'a' (owned groups) and stage 'b' (whole).  Stage 'a' wrote
 * `low` PACKED at the owned width; this re-pitches it to the gather's padded
 * per-rank stride (a whole number of groups, see pulsar_tp_allgather_rows),
 * exchanges in rank order -- rank order IS group order -- and writes the full
 * block back over the same buffer.  Concatenation, never a sum: every rank's
 * groups are distinct pieces of one row.  Rides the same monotonic seq as the
 * FFN big gate, so the two exchanges of a layer stay in lockstep across ranks. */
static bool tp_attn_gather_low(pulsar_gpu_graph *g, uint32_t il, uint32_t n_tokens,
                               uint32_t rank, uint32_t n_groups_total) {
    if (!g->tp) {
        fprintf(stderr, "pulsar: layer %u attention owns output groups [%u,%u) of %u but has no "
                        "TP transport to gather the rest -- refusing\n",
                il, g->tp_group_lo, g->tp_group_hi, n_groups_total);
        return false;
    }
    const uint32_t n_ranks = pulsar_tp_n_ranks(g->tp);
    const uint64_t own_dim  = (uint64_t)(g->tp_group_hi - g->tp_group_lo) * rank;
    const uint64_t stride   = (uint64_t)((n_groups_total + n_ranks - 1u) / n_ranks) * rank;
    const uint64_t full_dim = (uint64_t)n_groups_total * rank;
    float *packed  = (float *)xmalloc((size_t)n_tokens * own_dim * sizeof(float));
    float *own     = (float *)calloc((size_t)n_tokens * stride, sizeof(float));   /* padded tail stays zero */
    float *scratch = (float *)calloc((size_t)n_tokens * stride, sizeof(float));
    float *full    = (float *)xmalloc((size_t)n_tokens * full_dim * sizeof(float));
    bool ok = packed && own && scratch && full;
    if (!ok) fprintf(stderr, "pulsar: tp attention gather out of memory (layer %u, %u rows)\n", il, n_tokens);
    if (ok) ok = pulsar_gpu_tensor_read(g->batch_attn_low, 0, packed,
                                        (uint64_t)n_tokens * own_dim * sizeof(float)) != 0;
    if (ok) {
        for (uint32_t r = 0; r < n_tokens; r++)
            memcpy(own + (uint64_t)r * stride, packed + (uint64_t)r * own_dim, own_dim * sizeof(float));
        ok = pulsar_tp_allgather_rows(g->tp, il, ++g->tp_prefill_seq, full, own, scratch,
                                      n_tokens, n_groups_total, rank) != 0;
        if (!ok) fprintf(stderr, "pulsar: tp attention gather failed (layer %u, %u rows)\n", il, n_tokens);
    }
    if (ok) ok = pulsar_gpu_tensor_write(g->batch_attn_low, 0, full,
                                         (uint64_t)n_tokens * full_dim * sizeof(float)) != 0;
    free(packed);
    free(own);
    free(scratch);
    free(full);
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
    /* Slice 4g (L241): this rank computes the attention heads of the OUTPUT
     * GROUPS it owns, [g->tp_group_lo, g->tp_group_hi) of n_groups_total --
     * whole heads, whole groups.  `n_head`, `q_dim` and `n_groups` below are
     * the OWNED counts and every kernel in this block runs on them: q is the
     * compact [tokens][n_head][head_dim] the sliced attn_q_b writes, the sinks
     * start at the first owned head, the grouped 'a' projection covers the
     * owned groups, and `low` is gathered across the group before 'b' (which
     * every rank runs whole, on identical input).  Every value a rank computes
     * is a head- or group-independent piece of the single-box computation, so
     * the split is bit-exact against one box and across ranks.  On one box the
     * owned range is the whole layer and nothing here changes. */
    const uint32_t n_groups_total = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups_total;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;
    const uint32_t g_lo = g->tp_group_lo, g_hi = g->tp_group_hi;
    if (g_hi <= g_lo || g_hi > n_groups_total) {
        fprintf(stderr, "pulsar: layer %u attention: this graph owns output groups [%u,%u) of %u "
                        "-- refusing\n", il, g_lo, g_hi, n_groups_total);
        return false;
    }
    const uint32_t n_groups = g_hi - g_lo;
    const uint32_t n_head = n_groups * group_heads;
    const uint32_t h_lo = g_lo * group_heads;
    const uint64_t q_dim = (uint64_t)n_head * PULSAR_N_HEAD_DIM;
    const uint64_t q_dim_full = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint64_t sinks_off = layer->attn_sinks->abs_offset + (uint64_t)h_lo * sizeof(float);
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    /* CSA2 (L218): the pools this layer attends over live at its kv source. */
    const pulsar_layer_attn *attn = pulsar_layer_attn_layout(il);
    const uint32_t src = compressed ? attn->kv_source : il;
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
     * Every prefill attention consumer reads the packed rows as of
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
    /* L216 image-span visibility for THIS chunk, or NULL/NULL when the chunk has
     * no span (vision_visible_tokens == 0 -- every text chunk).  The two halves
     * live in one tensor, left at 0 and right at prefill_cap counts.  The views
     * are passed to every whole-chunk attention launch below; the indexed span
     * path offsets its own from these in gpu_graph_indexed_attention_span. */
    const bool chunk_vis = g->vision_visible_tokens == n_tokens && n_tokens != 0u;
    pulsar_gpu_tensor *vis_left_view = chunk_vis
            ? pulsar_gpu_tensor_view(g->vision_visible, 0,
                                     (uint64_t)n_tokens * sizeof(int32_t)) : NULL;
    pulsar_gpu_tensor *vis_right_view = chunk_vis
            ? pulsar_gpu_tensor_view(g->vision_visible,
                                     (uint64_t)g->prefill_cap * sizeof(int32_t),
                                     (uint64_t)n_tokens * sizeof(int32_t)) : NULL;
    bool ok = hc_mix_view && hc_split_view && after_attn_hc_view &&
              (attn_cur_view || !g->batch_attn_cur) &&
              (!chunk_vis || (vis_left_view && vis_right_view));
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
                                                                 g->batch_hc_pre,
                                                                 hc_mix_view,
                                                                 g->batch_cur_hc,
                                                                 tensor_map_base(model, layer->hc_attn_scale),
                                                                 tensor_map_size(model, layer->hc_attn_scale),
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
                                                             tensor_map_base(model, layer->attn_q_a_norm),
                                                             tensor_map_size(model, layer->attn_q_a_norm),
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
        if (ok) ok = gpu_graph_matmul_mxfp8_rows_named_tensor("attn_q_b",
                                                               il,
                                                               pos0,
                                                               g->batch_q,
                                                               model,
                                                               layer->attn_q_b,
                                                               q_rank,
                                                               q_dim_full,
                                                               (uint64_t)h_lo * PULSAR_N_HEAD_DIM,
                                                               (uint64_t)(h_lo + n_head) * PULSAR_N_HEAD_DIM,
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
         * memory, so it runs the standalone kernel first and hands attention
         * pre-normed Q (q_prep NULL).  The two are bit-exact (shared rope core,
         * replicated reduction -- attn_f16.cu), and the attention launch is the
         * same fp16 kernel either way (L166).  The per-head RMS half is
         * `PULSAR_Q_HEAD_NORM`'s: 0731 normalises Q per head there, V4.1 has no
         * per-head pass (its q_norm is on the low-rank latent, before wq_b). */
        const bool prefill_q_defer = !gpu_graph_f32_store_observed("Qcur", il, pos0);
        g->q_prep_active = 0;
        bool prefill_q_norm_rope_fused = false;
        if (ok && prefill_q_defer) {
            memset(&g->q_prep, 0, sizeof g->q_prep);
            g->q_prep.eps = PULSAR_Q_HEAD_NORM ? PULSAR_RMS_EPS : 0.0f;
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
            /* One call either way; the profile picks the kernel, exactly as it
             * picks the fused prologue's scale. */
            prefill_q_norm_rope_fused = (PULSAR_Q_HEAD_NORM
                ? pulsar_gpu_head_rms_norm_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            n_head,
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
                                            mseq ? g->batch_positions : NULL)
                : pulsar_gpu_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            n_head,
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
                                            mseq ? g->batch_positions : NULL)) != 0;
        }
        /* The separate head-norm + rope-tail pair that used to live here was
         * reachable ONLY by asking for a "Qnorm" dump: the fused kernel never
         * materialises that intermediate, so the dump request forced a
         * different pair of kernels and the dumped numbers were, by the file's
         * own warning, not the numbers production computes.  A debug
         * affordance that changes what it observes cannot diagnose what it
         * observes, so it is gone along with the "Qnorm" dump.  Prefill Q now
         * has exactly two places to be normed and roped: inside attention
         * (shipped) or by the standalone kernel above (the "Qcur" dump); same
         * numbers, same attention kernel after it.  L045 stage 2.
         *
         * If the post-norm/pre-rope intermediate is ever genuinely needed, the
         * honest way to get it is an optional store from the SHIPPED kernel,
         * not a second code path that only debuggers take. */
        if (!prefill_q_norm_rope_fused && ok) {
            fprintf(stderr, "pulsar: prefill Q reached neither the deferred nor the standalone "
                            "rope path -- refusing rather than leaving Q unrotated\n");
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
    if (ok) ok = pulsar_gpu_kv_ring_pack_tensor(gpu_graph_f32_store_observed_any() ? g->batch_kv : NULL,
                                              g->batch_kv, g->batch_kv_pack, 0u, n_tokens, PULSAR_N_HEAD_DIM,
                                              NULL, NULL, 1u, 0u) != 0;
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
                                                          tensor_map_base(model, layer->attn_sinks),
                                                          tensor_map_size(model, layer->attn_sinks),
                                                          sinks_off,
                                                          g->batch_q,
                                                          g->batch_kv_pack,
                                                          n_tokens,
                                                          g->raw_window,
                                                          n_head,
                                                          PULSAR_N_HEAD_DIM,
                                                          mseq ? g->batch_positions : NULL,
                                                          g->q_prep_active ? &g->q_prep : NULL,
                                                          vis_left_view, vis_right_view) != 0;
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
                                                                   tensor_map_base(model, layer->attn_sinks),
                                                                   tensor_map_size(model, layer->attn_sinks),
                                                                   sinks_off,
                                                                   g->batch_q,
                                                                   mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                        : g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   mseq ? 0 : n_raw,
                                                                   g->raw_cap,
                                                                   mseq ? 0 : raw_start,
                                                                    g->raw_window,
                                                                    n_head,
                                                                    PULSAR_N_HEAD_DIM,
                                                                    0,
                                                                    mseq ? g->batch_positions : NULL,
                                                                    mseq ? g->batch_seq_id : NULL,
                                                                    0,
                                                                    mseq ? nb : 1,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && compressed) {
        /* CSA2 (L218): KV production belongs to the kv SOURCE alone (FULL /
         * FULL_UNINDEXED) -- it runs its compressor over this batch and emits
         * the comp rows every later layer up to the next source attends over,
         * plus the index-K rows when it is also an index source (V4.1 always,
         * 0731 only on its ratio-4 layers).  A member layer (REINDEX / REUSE)
         * produces nothing and reads its source's pools.  The indexer QUERY
         * side belongs to every indexed layer; a REUSE layer attends with its
         * index source's top-k. */
        if (pulsar_attn_owns_kv(attn->mode)) {
            /* ONE production body for both profiles, and for BOTH kinds of kv
             * source.  They differ in which weights project the rows and in
             * whether the indexer compresses its own index key; neither
             * difference is visible here, because the projections are named by
             * `own` and gpu_graph_csa2_produce branches on the profile for the
             * second compression.  Control flow is identical.
             *
             * A FULL_UNINDEXED source (0731's ratio-128 HCA layers) has no
             * indexer at all, so it needs no index-key weights and produces no
             * top-k: `indexed` is false for it.  It is its own kv source and
             * attends over that pool with no selection, which is the mixed
             * branch below -- the reference's `get_compress_topk_idxs` hands the
             * unindexed layers every visible compressed position, in order, with
             * no top-k at all. */
            const bool indexed = pulsar_attn_runs_indexer(attn->mode);
            const bool own = g_pulsar_shape.indexer_own_compressor && indexed;
            const bool have_comp = layer->attn_compressor_kv && layer->attn_compressor_norm &&
                                   (ratio == 1u || layer->attn_compressor_gate) &&
                                   (!indexed ||
                                    (own ? (layer->indexer_compressor_kv && layer->indexer_compressor_gate &&
                                            layer->indexer_compressor_norm && layer->indexer_compressor_ape)
                                         : (layer->indexer_k && layer->indexer_k_norm)));
            if (!have_comp) {
                fprintf(stderr, "pulsar: kv source %u is missing its %s weights -- refusing\n", il,
                        !indexed ? "compressor"
                                 : own ? "compressor / indexer-compressor" : "compressor / index-key");
                ok = false;
            }
            const uint32_t comp_width = pulsar_comp_row_width(ratio, PULSAR_N_HEAD_DIM);
            const uint32_t index_width = pulsar_comp_row_width(ratio, PULSAR_N_INDEXER_HEAD_DIM);
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv, model, layer->attn_compressor_kv,
                                                       PULSAR_N_EMBD, comp_width, g->batch_attn_norm, n_tokens) != 0;
            if (ok && ratio > 1u) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc, model, layer->attn_compressor_gate,
                                                                     PULSAR_N_EMBD, comp_width, g->batch_attn_norm, n_tokens) != 0;
            if (ok) gpu_graph_debug_dump_tensor("attn_comp_kv_raw", g->batch_comp_kv,
                                                (uint64_t)comp_width * n_tokens, il, pos0);
            if (ok && ratio > 1u) gpu_graph_debug_dump_tensor("attn_comp_score_raw", g->batch_comp_sc,
                                                              (uint64_t)comp_width * n_tokens, il, pos0);
            /* The indexer's own projections, at its own head dim: a separate
             * compression of the same rows, over the same normed activation. */
            if (ok && own) ok = gpu_graph_matmul_plain_tensor(g->batch_index_comp_kv, model, layer->indexer_compressor_kv,
                                                              PULSAR_N_EMBD, index_width, g->batch_attn_norm, n_tokens) != 0;
            if (ok && own) ok = gpu_graph_matmul_plain_tensor(g->batch_index_comp_sc, model, layer->indexer_compressor_gate,
                                                              PULSAR_N_EMBD, index_width, g->batch_attn_norm, n_tokens) != 0;
            if (ok) ok = gpu_graph_csa2_produce(g, model, layer, il, pos0, n_tokens, mseq, comp_counts);
        } else {
            /* The source ran earlier in this same layer sweep (S < il), so its
             * emits for every row of this batch are in place: the visible row
             * count is the position law, not a counter this layer moves. */
            for (uint32_t t = 0; t < n_tokens; t++) {
                const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
                comp_counts[t] = (pos + 1u) / ratio;
            }
        }
        /* The comp bound the attention launch below hands the kernels for the
         * WHOLE batch: banked, the step's superset computed and cap-checked
         * once in step_begin (L178); classic, the source's frontier. */
        uint32_t n_comp = mseq ? g->batch_comp_sup[src] : gpu_graph_n_comp(g, gpu_graph_cur_bank(g), src);

        /* The QUERY side belongs to the layers that run an indexer: FULL and
         * REINDEX publish a top-k, REUSE reads one it did not compute, and
         * FULL_UNINDEXED has no indexer and no top-k at all.  Spelled with the
         * named authority rather than `mode != REUSE`, which quietly included the
         * unindexed layer and then asked it for query weights its artifact does
         * not carry. */
        if (ok && pulsar_attn_runs_indexer(attn->mode)) {
            if (!layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "pulsar: index source %u is missing indexer query weights -- refusing\n", il);
                ok = false;
            }
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_q,
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM,
                                                          g->batch_qr_norm,
                                                          n_tokens);
            /* Fused rope + FP4 pack: one launch over batch_indexer_q instead of
             * a rope_tail + pack pair (bit-exact, see the kernel note).  V4's
             * Indexer is built `rotate=True`, so the reference rotates q with the
             * same Hadamard as its key (`q = rotate_activation(q)` before the fp4
             * quant).  The two sides of the product must be rotated together, so
             * the profile picks the launch and the arguments are shared. */
            const bool rotate_q = g_pulsar_shape.indexer_own_compressor;
            if (ok) ok = (rotate_q ? pulsar_gpu_dsv4_indexer_rope_qat_tensor
                                   : pulsar_gpu_indexer_rope_fp4_pack_tensor)(
                                                    g->batch_indexer_q,
                                                    g->batch_indexer_qp,
                                                    n_tokens,
                                                    PULSAR_N_INDEXER_HEAD,
                                                    PULSAR_N_INDEXER_HEAD_DIM,
                                                    PULSAR_N_ROT,
                                                    pos0,
                                                    (uint32_t)PULSAR_ROPE_ORIG_CTX,
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
            /* The top-k path needs a top-k.  A FULL_UNINDEXED source never has
             * one however deep the pool grows, so it takes the mixed branch at
             * every depth -- which is exactly what the reference's unindexed
             * layers do (`get_compress_topk_idxs`, every visible compressed
             * position, in order). */
            if (ok && pulsar_attn_reads_index(attn->mode) && n_comp > PULSAR_N_INDEXER_TOP_K) {
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
                    mseq ? gpu_graph_bank_attn_comp_pool(g, src)
                         : g->layer_attn_comp_cache[src];
                const struct gpu_graph_span_ops sop = {
                    /* comp_src   */ span_comp_src,
                    /* raw_src    */ mseq ? gpu_graph_bank_raw_pool(g, il) : g->layer_raw_cache[il],
                    /* index_src  */ mseq ? gpu_graph_bank_index_comp_pool(g, src)
                                          : g->layer_index_comp_cache[src],
                    /* index_bases*/ mseq ? gpu_graph_bank_index_comp_bases(g, src) : NULL,
                    /* comp_bases */ mseq ? gpu_graph_bank_attn_comp_bases(g, src) : NULL,
                    /* comp_cap   */ mseq ? g->layer_comp_cap[src] : 0u,
                    /* n_banks    */ mseq ? nb : 1u,
                    /* mseq       */ mseq,
                };
                /* The span hands n_comp -- the source's frontier -- to the indexer as
                 * its row count and score stride too: one emit writes the comp row
                 * AND the index-K row, so the two pools share one frontier. */
                for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += span) {
                    const uint32_t sn = n_tokens - s0 < span ? n_tokens - s0 : span;
                    const uint32_t spos0 = pos0 + s0;
                    const uint32_t s_n_raw = gpu_graph_raw_span_for_batch(g, spos0, sn);
                    const uint32_t s_raw_start = gpu_graph_raw_start_for_span(g,
                                                                                spos0 + sn - 1u,
                                                                                s_n_raw);
                    ok = gpu_graph_indexed_attention_span(g, model, layer, il, attn,
                            s0, sn, spos0, q_dim, n_head, sinks_off, n_comp, ratio, index_scale,
                            mseq ? 0u : s_n_raw, mseq ? 0u : s_raw_start,
                            &sop);
                }
            } else if (ok) {
                ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                         tensor_map_base(model, layer->attn_sinks),
                                                                         tensor_map_size(model, layer->attn_sinks),
                                                                         sinks_off,
                                                                         g->batch_q,
                                                                         mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                              : g->layer_raw_cache[il],
                                                                         mseq ? gpu_graph_bank_attn_comp_pool(g, src)
                                                                              : g->layer_attn_comp_cache[src],
                                                                         n_tokens,
                                                                         pos0,
                                                                         mseq ? 0 : n_raw,
                                                                         g->raw_cap,
                                                                         mseq ? 0 : raw_start,
                                                                         n_comp,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          n_head,
                                                                          PULSAR_N_HEAD_DIM,
                                                                          0,
                                                                          mseq ? g->batch_positions : NULL,
                                                                          mseq ? g->batch_seq_id : NULL,
                                                                          mseq ? gpu_graph_bank_attn_comp_bases(g, src) : NULL,
                                                                          mseq ? g->layer_comp_cap[src] : 0,
                                                                          mseq ? nb : 1,
                                          g->q_prep_active ? &g->q_prep : NULL) != 0;
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = pulsar_attn_reads_index(attn->mode) &&
                                         compressed && n_comp > PULSAR_N_INDEXER_TOP_K;
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
             * source of comp-row dequant launches in production. */
            pulsar_gpu_tensor *zspan_comp_src = g->layer_attn_comp_cache[src];
            const struct gpu_graph_span_ops zsop = {
                /* comp_src   */ zspan_comp_src,
                /* raw_src    */ g->layer_raw_cache[il],
                /* index_src  */ g->layer_index_comp_cache[src],
                /* index_bases*/ NULL,
                /* comp_bases */ NULL,
                /* comp_cap   */ 0u,
                /* n_banks    */ 1u,
                /* mseq       */ false,
            };
            for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += zspan) {
                const uint32_t sn = n_tokens - s0 < zspan ? n_tokens - s0 : zspan;
                const uint32_t spos0 = pos0 + s0;
                ok = gpu_graph_indexed_attention_span(g, model, layer, il, attn,
                        s0, sn, spos0, q_dim, n_head, sinks_off, n_comp, ratio, index_scale,
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
                                                                       tensor_map_base(model, layer->attn_sinks),
                                                                       tensor_map_size(model, layer->attn_sinks),
                                                                       sinks_off,
                                                                       g->batch_q,
                                                                       g->batch_kv_pack,
                                                                       /* Packed pool straight in.  This called
                                                                        * gpu_graph_attn_comp_read_cache -- a full
                                                                        * dequantise of 584 B rows into a 2048 B f32
                                                                        * shadow, on the SHIPPED path, because the
                                                                        * consumer could not read packed.  The launcher
                                                                        * has ONE arm now (L166): the fp16 tier reads
                                                                        * the MAIN pool natively through comp_kv;
                                                                        * no shadow, no second kernel. */
                                                                       mseq ? gpu_graph_bank_attn_comp_pool(g, src)
                                                                            : g->layer_attn_comp_cache[src],
                                                                       gact_data, gact_scale, gact_kbp,
                                                                       (uint32_t)gact_slab, n_groups,
                                                                       PULSAR_N_HEAD_DIM - PULSAR_N_ROT,
                                                                       &gact_emitted,
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       n_head,
                                                                       PULSAR_N_HEAD_DIM,
                                          g->q_prep_active ? &g->q_prep : NULL,
                                          vis_left_view, vis_right_view) != 0;
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
                                                              tensor_map_base(model, layer->attn_sinks),
                                                              tensor_map_size(model, layer->attn_sinks),
                                                              sinks_off,
                                                              g->batch_q,
                                                              g->batch_kv_pack,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              n_head,
                                                              PULSAR_N_HEAD_DIM,
                                                              gact_data, gact_scale, gact_kbp,
                                                              (uint32_t)gact_slab, n_groups,
                                                              PULSAR_N_HEAD_DIM - PULSAR_N_ROT,
                                                              &gact_emitted,
                                          mseq ? g->batch_positions : NULL,
                                          g->q_prep_active ? &g->q_prep : NULL,
                                          vis_left_view, vis_right_view) != 0;
        }
        if (!gact_emitted) { gact_data = NULL; gact_scale = NULL; }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                uint32_t n_selected = 0;
                bool have_topk = false;
                /* The indexer ranks the source's index-K rows and the attention
                 * folds the selected ids over the source's comp rows; one emit
                 * writes both, so cur_comp bounds both. */
                /* CSA2: the selection for row t lives at absolute row t of the
                 * shared buffer; an index source writes it, a REUSE layer reads
                 * its index source's. */
                pulsar_gpu_tensor *sel_t = compressed ? pulsar_gpu_tensor_view(g->comp_selected,
                        (uint64_t)t * PULSAR_N_INDEXER_TOP_K * sizeof(uint32_t),
                        (uint64_t)PULSAR_N_INDEXER_TOP_K * sizeof(uint32_t)) : NULL;
                if (compressed && !sel_t) { ok = false; break; }
                /* Selection belongs to the layers that run an indexer (FULL,
                 * REINDEX); a REUSE layer reads the shared one and a
                 * FULL_UNINDEXED layer has none to make. */
                if (compressed && pulsar_attn_runs_indexer(attn->mode) &&
                    cur_comp > PULSAR_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                    pulsar_gpu_tensor *indexer_q_view = pulsar_gpu_tensor_view(
                            g->batch_indexer_qp,
                            (uint64_t)t * PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX),
                            (uint64_t)PULSAR_N_INDEXER_HEAD * pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX));
                    pulsar_gpu_tensor *indexer_w_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_weights, t, PULSAR_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         pulsar_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[src],
                                                            cur_comp,
                                                            PULSAR_N_INDEXER_HEAD,
                                                            PULSAR_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         gpu_graph_csa2_candidates(g, attn, il, t, 1u, cur_comp, ratio, pos, NULL) &&
                         pulsar_gpu_indexer_topk_tensor(sel_t,
                                                       g->indexer_scores,
                                                       cur_comp,
                                                       1,
                                                       PULSAR_N_INDEXER_TOP_K) != 0;
                    pulsar_gpu_tensor_free(indexer_w_view);
                    pulsar_gpu_tensor_free(indexer_q_view);
                }
                /* As above, the top-k path is the top-k layers' alone: past 512
                 * compressed rows an unindexed source still attends over every
                 * one of them. */
                if (compressed && pulsar_attn_reads_index(attn->mode) &&
                    cur_comp > PULSAR_N_INDEXER_TOP_K) {
                    if (ok) {
                        have_topk = true;
                        n_selected = PULSAR_N_INDEXER_TOP_K < cur_comp
                            ? PULSAR_N_INDEXER_TOP_K
                            : cur_comp;
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
                                sel_t, (uint64_t)n_selected, il, pos);
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
                        (uint64_t)t * pulsar_kv_row_bytes(PULSAR_KV_ROW_RING),
                        pulsar_kv_row_bytes(PULSAR_KV_ROW_RING));
                pulsar_gpu_tensor *heads_view = gpu_graph_heads_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_pack_view && heads_view;
                if (ok && !zero_prefix) {
                    /* n_tokens=1 with pos0=pos puts the row at pos % raw_cap --
                     * the same slot the f32 store targeted (pulsar_kv_ring_slot). */
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
                                                                              tensor_map_base(model, layer->attn_sinks),
                                                                              tensor_map_size(model, layer->attn_sinks),
                                                                              sinks_off,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              /* Native packed read: this sits in a PER-TOKEN loop and
                                                                               * cur_comp grows per token, so the shadow was rebuilt
                                                                               * for every token. */
                                                                              g->layer_attn_comp_cache[src],
                                                                              sel_t,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              n_head,
                                                                              PULSAR_N_HEAD_DIM,
                                                                              NULL, NULL, NULL, 0, 1,
                                          g->q_prep_active ? &g->q_prep : NULL,
                                          /* A per-token row never carries image visibility: an
                                           * image chunk is handled whole by the arms above. */
                                          NULL, NULL) != 0;
                } else if (ok) {
                    /* No selection this token: the same one-row step the
                     * batched decode takes, through the same entry -- the
                     * fp16 tier for this shape, with q_prep honoured like the
                     * indexed sibling above.  (The single-token entry this
                     * replaced always ran the f32 kernel and had no q_prep
                     * parameter, L164.) */
                    ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(heads_view,
                            tensor_map_base(model, layer->attn_sinks), tensor_map_size(model, layer->attn_sinks), sinks_off,
                            q_view, g->layer_raw_cache[il],
                            cur_comp ? g->layer_attn_comp_cache[src] : NULL,
                            1, pos, n_raw, g->raw_cap, raw_start, cur_comp,
                            g->raw_window, ratio, n_head, PULSAR_N_HEAD_DIM,
                            0, NULL, NULL, NULL, 0, 1,
                            g->q_prep_active ? &g->q_prep : NULL) != 0;
                }
                pulsar_gpu_tensor_free(heads_view);
                pulsar_gpu_tensor_free(kv_pack_view);
                pulsar_gpu_tensor_free(q_view);
                pulsar_gpu_tensor_free(sel_t);
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
     * that epilogue actually ran (gact_emitted).
     *
     * L210: DECODE rows with the fused Q prep already carry this rotation --
     * the split-K combine applies it as it stores the heads (the same two
     * roundings in the same order; the decode blob did not move) -- so this
     * launch covers the rows past them: a mixed step's prefill rows, or
     * nothing.  Those rows never come with grouped slots (the epilogue that
     * emits them is the dense prefill arm's, which a step with decode rows
     * does not take); a slot here with decode rows present is a contradiction,
     * refused rather than half-emitted. */
    const int rope_dec_rows = pulsar_gpu_matmul_batch_decode_rows();
    const uint32_t rope_row0 = (g->q_prep_active && rope_dec_rows > 0)
                             ? ((uint32_t)rope_dec_rows < n_tokens ? (uint32_t)rope_dec_rows : n_tokens) : 0u;
    if (ok && rope_row0 != 0u && gact_data) {
        fprintf(stderr, "pulsar: layer %u: grouped heads slots with %u decode rows -- the split-K combine "
                        "roped those rows without emitting; refusing\n", il, rope_row0);
        ok = false;
    }
    if (ok && rope_row0 < n_tokens) {
        const uint32_t rope_rows = n_tokens - rope_row0;
        pulsar_gpu_tensor *heads_rows = rope_row0 ? pulsar_gpu_tensor_view(g->batch_heads,
                                                        (uint64_t)rope_row0 * q_dim * PULSAR_HEADS_ELT_SIZE,
                                                        (uint64_t)rope_rows * q_dim * PULSAR_HEADS_ELT_SIZE)
                                                  : g->batch_heads;
        pulsar_gpu_tensor *pos_rows = (mseq && rope_row0) ? pulsar_gpu_tensor_view(g->batch_positions,
                                                        (uint64_t)rope_row0 * sizeof(int32_t),
                                                        (uint64_t)rope_rows * sizeof(int32_t))
                                                  : (mseq ? g->batch_positions : NULL);
        ok = heads_rows && (!mseq || pos_rows) &&
             pulsar_gpu_rope_tail_mx_tensor(heads_rows,
                                            rope_rows,
                                            n_head,
                                            PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT,
                                            pos0 + rope_row0,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            pos_rows,
                                            gact_data, gact_scale, gact_kbp,
                                            (uint32_t)gact_slab, n_groups) != 0;
        if (rope_row0) {
            if (heads_rows) pulsar_gpu_tensor_free(heads_rows);
            if (pos_rows) pulsar_gpu_tensor_free(pos_rows);
        }
    }
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
    /* Stage 'a' over the owned groups (the registered row slice of attn_output_a
     * on a TP rank; the whole tensor on one box), the gather of `low` across
     * the group, then stage 'b' whole on every rank. */
    if (ok) {
        ok = pulsar_gpu_attention_output_a_tensor(g->batch_attn_low,
                                                  tensor_map_base(model, layer->attn_output_a),
                                                  tensor_map_size(model, layer->attn_output_a),
                                                  layer->attn_output_a->abs_offset +
                                                      (uint64_t)g_lo * rank * group_dim,
                                                  group_dim,
                                                  rank,
                                                  n_groups,
                                                  g->batch_heads,
                                                  n_tokens) != 0;
    }
    if (ok && n_groups != n_groups_total) ok = tp_attn_gather_low(g, il, n_tokens, rank, n_groups_total);
    if (ok) {
        ok = pulsar_gpu_attention_output_b_tensor(g->batch_attn_out,
                                                  tensor_map_base(model, layer->attn_output_a),
                                                  tensor_map_size(model, layer->attn_output_a),
                                                  layer->attn_output_b->abs_offset,
                                                  (uint64_t)n_groups_total * rank,
                                                  PULSAR_N_EMBD,
                                                  g->batch_attn_low,
                                                  n_tokens) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups_total * rank,
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
    pulsar_gpu_tensor_free(vis_right_view);
    pulsar_gpu_tensor_free(vis_left_view);
    pulsar_gpu_tensor_free(attn_cur_view);
    pulsar_gpu_tensor_free(hc_split_view);
    pulsar_gpu_tensor_free(hc_mix_view);
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
/* Slice 4b: one prefill big-gate exchange for `il` over the whole chunk.  Read
 * this layer's n_embd-width routed contribution to host, swap it with the peer
 * via the pair transport, and write the combined (local + peer) value back so
 * the FFN's HC expansion below carries the fully-summed layer output.  Called
 * only when g->tp is armed; returns 0 (fail loud) on any tensor/transport
 * failure.  Host staging is a transient cost on this first wiring; the D2H/H2D
 * can move onto the registerable GB10 slab later. */
static bool tp_prefill_big_gate(pulsar_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    if (!g->tp) return 1;
    const uint64_t nelt = (uint64_t)n_tokens * PULSAR_N_EMBD;
    const uint64_t bytes = nelt * sizeof(float);
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
     *    which the transport stages through the slab in message-sized pieces. */
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
    bool ok = pulsar_gpu_tensor_read(g->batch_routed_out, 0, out, bytes) != 0;
    if (ok) {
        ok = pulsar_tp_allreduce_sum(g->tp, il, ++g->tp_prefill_seq,
                                     out, in, bytes) != 0;
    }
    if (ok) {
        ok = pulsar_gpu_tensor_write(g->batch_routed_out, 0, out, bytes) != 0;
    }
    if (heap) {
        free(out);
        free(in);
    }
    return ok;
}

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
                                                                 g->batch_hc_pre,
                                                                 hc_mix_view,
                                                                 g->batch_after_attn_hc,
                                                                 tensor_map_base(model, layer->hc_ffn_scale),
                                                                 tensor_map_size(model, layer->hc_ffn_scale),
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
                                             layer->n_expert,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    /* All three router offsets below are offsets WITHIN THE LAYER, so the
     * mapping the kernel resolves them against must be the layer's own.  It
     * used to be taken from ffn_exp_probs_b, which the hash-routed layers 0..2
     * do not carry (they carry ffn_gate_tid2eid instead).  tensor_map_base() is
     * total, so a NULL tensor yields m->map -- the FIRST shard -- while the
     * offsets still point into the layer's own shard: in a one-file model those
     * are the same mapping and nothing shows, but a multi-shard checkpoint made
     * layers 0..2 read their routing table and VL bias out of the wrong file.
     * ffn_gate_inp is required on every layer, and weights_bind_layer asserts
     * that a layer's tensors share one mapping, which is what makes it a valid
     * anchor for the other two tensors' offsets. */
    const void *router_map = tensor_map_base(model, layer->ffn_gate_inp);
    const uint64_t router_map_size = tensor_map_size(model, layer->ffn_gate_inp);
    if (ok) ok = pulsar_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      gpu_graph_f32_store_observed("ffn_moe_probs", il, pos0)
                                                          ? g->batch_router_probs : NULL,
                                                      router_map,
                                                      router_map_size,
                                                      layer->ffn_exp_probs_b
                                                          ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid
                                                          ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid
                                                          ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      layer->n_expert,
                                                      layer->n_expert_used,
                                                      PULSAR_EXPERT_WEIGHT_SCALE,
                                                      n_tokens,
                                                      layer->ffn_exp_probs_b_vl ? layer->ffn_exp_probs_b_vl->abs_offset : 0,
                                                      PULSAR_N_VOCAB,
                                                      layer->ffn_exp_probs_b_vl != NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * layer->n_expert, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * layer->n_expert, il, pos0);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * layer->n_expert_used, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * layer->n_expert_used, il, pos0);
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

    /* The range is over THIS layer's routed experts -- the count the kernel is
     * handed below (n_expert_present), not the target table's entry for `il`.
     * The drafter reaches this encoder with ITS OWN layer index, and a REAP'd
     * or V4.1 layout gives the two tables different numbers; a range wider
     * than the kernel's total is a refusal there, and one narrower silently
     * drops experts. */
    uint32_t exp_lo = 0u;
    uint32_t exp_hi = layer->n_expert_present;
    if (ok && g->tp) {
        /* Slice 4c: under TP each rank computes ONLY its owned expert slice
         * (single authority), so the all-reduce at tp_prefill_big_gate sums the
         * owned partials into the correct full routed sum.  Co-gated with the
         * combine: a narrowed partial is only ever emitted when the all-reduce
         * runs, and never summed twice. */
        if (!pulsar_tp_owned_range(pulsar_tp_rank(g->tp),
                                          pulsar_tp_n_ranks(g->tp), exp_hi,
                                          &exp_lo, &exp_hi)) {
            fprintf(stderr, "pulsar: tp owned-expert range refused (rank=%d n=%u)\n",
                    pulsar_tp_rank(g->tp), pulsar_tp_n_ranks(g->tp));
            ok = false;
        }
    }
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
                                               layer->n_expert_present,
                                               layer->n_expert_used,
                                               PULSAR_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               exp_lo,
                                               exp_hi) != 0;
    }
    if (ok) {
        /* ARM-DEPENDENT: batch_routed_up is only written by the MMQ arms
         * (where it serves as raw-gate scratch); on 40/40-grouped and mixed
         * case-A layers this dump shows a PREVIOUS layer's bytes. Same below
         * for ffn_moe_down: the grouped path sums straight from its padded
         * GEMM output and never touches batch_routed_down. */
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * layer->n_expert_used * down_in_dim, il, pos0);
    }
    if (ok) {
        /* ARM-DEPENDENT, like ffn_moe_up_clamped above: every routed arm now
         * emits the folded SwiGLU leaf as E4M3 from its own epilogue (L219), so
         * this f32 scratch no longer holds the leaf anywhere -- on the MMQ
         * gate/up arms it is the RAW UP the pair GEMM wrote (same caveat as
         * ffn_moe_up_clamped), and on the pure type-40 arm it is stale.  The
         * element count reads the LAYER's router width, not the target's: a
         * drafter layer routes 128 experts / top-3 (L218 audit risk #2). */
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * layer->n_expert_used * down_in_dim;
        gpu_graph_debug_dump_tensor("ffn_moe_mid_raw_up", g->batch_routed_mid,
                                      routed_mid_elems, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * layer->n_expert_used * PULSAR_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
#undef PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT

    /* Slice 4b: with the group armed, exchange this layer's owned routed
     * partial and fold the peers' in BEFORE the HC expansion below, so the layer
     * output carried into the next layer is the full routed sum.  One big gate
     * per layer for the whole chunk (amortized, not per token).  The ffn_out
     * sum is formed ONCE, from the combined routed value, so the single-box
     * branch and the TP branch never both write it. */
    if (ok && keep_ffn_out && !g->tp) {
        ok = gpu_graph_ensure_batch_ffn_out(g) &&
             pulsar_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * PULSAR_N_EMBD)) != 0;
    }
    if (ok && g->tp) {
        ok = tp_prefill_big_gate(g, il, n_tokens);
        if (ok && keep_ffn_out) {
            ok = gpu_graph_ensure_batch_ffn_out(g) &&
                 pulsar_gpu_add_tensor(g->batch_ffn_out,
                                      g->batch_shared_out,
                                      g->batch_routed_out,
                                      (uint32_t)((uint64_t)n_tokens * PULSAR_N_EMBD)) != 0;
        }
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



/* Capture the DSpark drafter anchors at layer `il` from `batch_cur_hc`, which
 * holds the layer's INPUT when this is called at layer entry and its OUTPUT
 * when called after the HC swap -- the caller picks the point, this function
 * owns the capture.  Both the fused-spec verify anchor and the retraining bulk
 * anchor are captured here so their placement cannot drift apart (dev keeps
 * them adjacent for the same reason).
 *
 * An armed capture whose buffer is missing is an impossible state, not a skip:
 * breaking with ok untouched let the round seed from STALE rows and lose
 * acceptance silently -- the class gpu_decode.cpp's KV seed documents (L190 D2).
 * Returns false, having captured nothing further, if any armed capture fails. */
static bool dspark_capture_anchors(pulsar_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    struct { uint32_t n; pulsar_gpu_tensor **dst; const char *what; } cap[2] = {
        { g->dspark_capture_batch_n, g->dspark_target_h_batch, "anchor" },
        { g->dspark_bulk_n,          g->dspark_bulk_h,          "bulk"   },
    };
    for (int c = 0; c < 2; c++) {
        if (!cap[c].n) continue;
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!cap[c].dst[slot]) {
                fprintf(stderr, "pulsar: drafter %s capture: layer %u is anchor slot %d but "
                                "its buffer is not allocated -- refusing\n", cap[c].what, il, slot);
                return false;
            }
            uint32_t cap_n = cap[c].n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!pulsar_gpu_dspark_hc_mean_reduce_batch(cap[c].dst[slot], g->batch_cur_hc,
                                                       PULSAR_N_EMBD, PULSAR_N_HC, cap_n)) {
                return false;
            }
            break;
        }
    }
    return true;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool gpu_graph_encode_layer_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = true;
    /* DRAFTER ANCHORS: the reference appends `h.mean(dim=2)` for its target
     * layers at exactly one point in a layer, and the point is NOT the same in
     * both profiles.  V4.1 (Transformer.forward) appends BEFORE `h = layer(h)`,
     * so its drafter is conditioned on the anchor layer's INPUT; 0731 appends
     * AFTER the layer, so its drafter sees that layer's OUTPUT.  The two arms
     * are therefore not interchangeable and `dspark_anchor_after` names which
     * one this artifact needs -- feeding the wrong arm's hidden is a structural
     * difference (measured 20-64% off across every conditioning array), not a
     * rounding one.
     *
     * Fused spec loop (P2): when armed, capture the drafter's anchor hidden for
     * every batch position at the anchor layers, so the last-accepted position's
     * hidden is available without a replay decode. Off (0) during prefill and
     * plain decode. */
    if (!g_pulsar_shape.dspark_anchor_after) ok = dspark_capture_anchors(g, il, n_tokens);
    if (ok) ok = gpu_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
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
    /* 0731 takes its anchors HERE, after the swap, so the drafter is
     * conditioned on the layer's OUTPUT hidden -- see the note at the top. */
    if (ok && g_pulsar_shape.dspark_anchor_after) ok = dspark_capture_anchors(g, il, n_tokens);
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

/* Stage-B rollback (L218 form).  After a partial accept the verify batch has
 * left every kv source's state at the LAST candidate row; the caller restored
 * the pre-batch snapshot (pending group + frontier) and now commits positions
 * [pos0, pos0 + n).  The comp and index-K rows those positions emitted are
 * position-addressed and already in the pools (a candidate row extends the
 * committed prefix contiguously, so an accepted group's row was written at
 * its true index), so nothing is re-emitted: the ratio-2 sources' pending
 * slots are re-stored from the projections saved during the batch -- a
 * group's slot layout IS the group, a consumed group's slots are simply
 * overwritten -- and every source's frontier is set by the position law.
 * Ratio-1 sources keep no state and need only the frontier. */
bool gpu_graph_dspark_compressor_rollforward(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        const pulsar_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions,
        uint32_t          save_row0) {
    (void)model;
    (void)weights;
    if (!g) {
        fprintf(stderr, "pulsar: compressor rollforward: NULL graph -- refusing\n");
        return false;
    }
    if (n_positions == 0) return true;
    if (save_row0 + n_positions > PULSAR_SPEC_LOGITS_ROWS + 1u) {
        fprintf(stderr, "pulsar: compressor rollforward: save rows %u+%u exceed the %u-row save slab -- refusing "
                        "before any layer moved\n",
                save_row0, n_positions, (unsigned)PULSAR_SPEC_LOGITS_ROWS + 1u);
        return false;
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_is_kv_source(il)) continue;
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio > 1u) {
            if (!g->spec_comp_kv_save[il] || !g->spec_comp_sc_save[il])
                return rollforward_fail(il, pos0, "no saved compressor projections for this kv source");
            /* The saved rows are the compressor PROJECTIONS, so their width is
             * the layer's coff width (2*head_dim at ratio 4), NOT head_dim.  The
             * view said head_dim, so every ratio-4 source stored a 2048-byte row
             * into a kernel that requires 4096 -- `bad operands`, loudly, on the
             * first verify batch.  V4.1's ratio-2 sources have coff 1 and agreed
             * by accident, which is why this survived until a V4 model ran. */
            const uint32_t comp_width = pulsar_comp_row_width(ratio, PULSAR_N_HEAD_DIM);
            /* The update kernel below needs a [1][head_dim] f32 scratch for the
             * row it pools at a group boundary.  The value is discarded -- the
             * pool's own row is already committed -- only the lane side effects
             * (store, pool, shift) matter here. */
            pulsar_gpu_tensor *latent_row = pulsar_gpu_tensor_view(g->attn_comp_stage, 0,
                                                                   (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
            if (!latent_row) return rollforward_fail(il, pos0, "no latent-row scratch for the rollforward");
            for (uint32_t t = 0; t < n_positions; t++) {
                const uint32_t pos = pos0 + t;
                pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->spec_comp_kv_save[il], save_row0 + t, comp_width);
                pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->spec_comp_sc_save[il], save_row0 + t, comp_width);
                /* The reference adds the ape to a row BEFORE it stores it, so
                 * the saved (raw) score row is not what the lane holds.  Fold it
                 * here or the rolled-forward pending group is a different
                 * function of its own inputs than the live path's. */
                int emitted = 0;
                pulsar_gpu_csa2_ape ape_desc;
                const pulsar_gpu_csa2_ape *ape = NULL;
                const bool ok = kv_view && sc_view &&
                    gpu_graph_comp_ape_desc(model, &weights->layer[il], &ape_desc, &ape) &&
                    /* THE UPDATE KERNEL, not a bare store.  This function's own
                     * comment promises "same update kernels, same rows, same
                     * order", and the live path's per-token update is what stores
                     * the row, POOLS the group when it closes and SHIFTS the
                     * completed group into the carry half.  A store alone never
                     * promotes a completed group, so the carry keeps whatever the
                     * rejected batch left there and the NEXT group's pool reads
                     * exactly that: the compressed row for the group at positions
                     * 28..31 came out ~80% off dev's while every one of its raw
                     * projections was bit-exact (L218 s121). */
                    pulsar_gpu_csa2_compressor_update_tensor(latent_row, kv_view, sc_view,
                            g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                            tensor_map_base(model, weights->layer[il].attn_compressor_norm), tensor_map_size(model, weights->layer[il].attn_compressor_norm),
                            weights->layer[il].attn_compressor_norm->abs_offset,
                            weights->layer[il].attn_compressor_norm->type, ape,
                            PULSAR_N_HEAD_DIM, ratio, pos, PULSAR_RMS_EPS, &emitted) != 0;
                pulsar_gpu_tensor_free(sc_view);
                pulsar_gpu_tensor_free(kv_view);
                if (!ok) return rollforward_fail(il, pos, "pending-slot update (row view or kernel)");
            }
            pulsar_gpu_tensor_free(latent_row);
            /* V4's indexer lane is a second recurrent state over the same rows:
             * roll it from ITS saved projections, or the next group pools
             * positions the rejected draft put there.  The SAME reasoning as the
             * attention lane above: through the UPDATE kernel, not a bare store,
             * or a replayed prefix that closes a group never promotes it into the
             * indexer's carry half.  Two differences from the attention lane:
             * this kernel applies the ape ITSELF (so no caller-side fold), and it
             * writes the pool row it emits -- `out_row` is the frontier the closed
             * group occupies (pos/ratio), whose row the live batch already wrote,
             * so the replay rewrites it with the same bytes. */
            if (g_pulsar_shape.indexer_own_compressor && pulsar_attn_runs_indexer(pulsar_layer_attn_layout(il)->mode)) {
                if (!g->spec_icomp_kv_save[il] || !g->spec_icomp_sc_save[il])
                    return rollforward_fail(il, pos0, "no saved indexer-compressor projections for this kv source");
                const pulsar_layer_weights *lw = &weights->layer[il];
                const uint32_t iw = pulsar_comp_row_width(ratio, PULSAR_N_INDEXER_HEAD_DIM);
                const float idx_freq_base = layer_rope_freq_base(il);
                const float idx_freq_scale = layer_rope_freq_scale(il);
                const float idx_ext_factor = PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
                float idx_attn_factor = 1.0f;
                if (idx_ext_factor != 0.0f && idx_freq_scale > 0.0f)
                    idx_attn_factor /= 1.0f + 0.1f * logf(1.0f / idx_freq_scale);
                pulsar_gpu_tensor *idx_latent = pulsar_gpu_tensor_view(g->idx_comp_stage, 0,
                                                    (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!idx_latent) return rollforward_fail(il, pos0, "no indexer latent scratch for the rollforward");
                for (uint32_t t = 0; t < n_positions; t++) {
                    const uint32_t pos = pos0 + t;
                    pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->spec_icomp_kv_save[il], save_row0 + t, iw);
                    pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->spec_icomp_sc_save[il], save_row0 + t, iw);
                    int idx_emitted = 0;
                    const bool ok = kv_view && sc_view &&
                        pulsar_gpu_indexer_compressor_update_tensor(
                                g->layer_index_comp_cache[il], idx_latent,
                                g->layer_index_state_kv[il], g->layer_index_state_score[il],
                                sc_view, kv_view,
                                tensor_map_base(model, lw->indexer_compressor_ape), tensor_map_size(model, lw->indexer_compressor_ape),
                                lw->indexer_compressor_ape->abs_offset, lw->indexer_compressor_ape->type,
                                lw->indexer_compressor_norm->abs_offset,
                                lw->indexer_compressor_norm->type,
                                pos / ratio, PULSAR_N_INDEXER_HEAD_DIM, ratio, pos,
                                PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                                idx_freq_base, idx_freq_scale, idx_ext_factor, idx_attn_factor,
                                PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                                PULSAR_RMS_EPS, &idx_emitted) != 0;
                    pulsar_gpu_tensor_free(sc_view);
                    pulsar_gpu_tensor_free(kv_view);
                    if (!ok) return rollforward_fail(il, pos, "indexer pending-slot update");
                }
                pulsar_gpu_tensor_free(idx_latent);
            }
        }
        gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) = (pos0 + n_positions) / ratio;
    }
    return true;
}

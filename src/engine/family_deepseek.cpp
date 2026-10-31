/** @file family_deepseek.cpp
 *  @brief The DeepSeek V4 family descriptor (0731, Vision-Exp, V4.1). */
/* family_deepseek.cpp -- the DeepSeek V4 family descriptor (L251 S1).
 *
 * The first family: DeepSeek V4 Flash (0731), Vision-Exp and V4.1 Flash, one
 * architecture with two shape profiles (shape_profiles.cpp).  Every entry
 * points at the engine as it stood at L250 -- the load and device steps moved
 * out of pulsar_engine::open, the session create/destroy/price bodies, and the
 * pulsar_session methods -- so a DeepSeek lane runs the same code, in the same
 * order, as before the interface existed.  The graph (pulsar_session::graph)
 * is this family's session state. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "spec_internal.h"

/* L272 B15: the drafter the opened artifact carries -- DSpark when its dspark.* tensors loaded. */
/// @param e the engine
/// @return DSPARK or NONE
static pulsar_drafter_kind ds4_drafter(pulsar_engine *e) {
    return e->has_dspark() ? PULSAR_DRAFTER_DSPARK : PULSAR_DRAFTER_NONE;
}

/// pulsar_family::quant_bits.
/// @param e the engine
/// @return 4, 2, 20 + the EXL3 half-bit rate, or 0
static int ds4_quant_bits(pulsar_engine *e) {
    /* Report the routed-expert precision tier actually present, derived from
     * the loaded tensor types (was hardcoded 2, which under-reported the mixed
     * IQ2 + MXFP4/type-40 build as pure 2-bit). Any 4-bit routed format
     * (MXFP4 E2M1 / CUTLASS type-40) anywhere in gate/up/down makes this a
     * 4-bit-tier model; otherwise the 2-bit floor (IQ2_XXS / Q2_K); 0 if no
     * routed experts. pulsar_engine_model_id() is the profile, so
     * this is the model-variant discriminator in the KV segment store's
     * identity (pulsar_segstore_identity): a value change puts a build on a
     * fresh store (one-time re-prefill). */
    /* EXL3 (L245) is its own value space -- 20 + the rate in half-bit units
     * (24 = K2, 25 = K2.5, 26 = K3; the highest rate present wins) -- so an
     * EXL3 artifact never shares KV with the IQ2 (2) or MXFP4 (4) tier of the
     * same model id. */
    int bits = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_tensor *proj[3] = {
            e->weights.layer[il].ffn_gate_exps,
            e->weights.layer[il].ffn_up_exps,
            e->weights.layer[il].ffn_down_exps,
        };
        for (int k = 0; k < 3; k++) {
            const pulsar_tensor *t = proj[k];
            if (!t) continue;
            if (t->type == PULSAR_TENSOR_CUTLASS_MXFP4)
                return 4;
            const int k2 = exl3_type_k2(t->type);
            if (k2) {
                if (20 + k2 > bits) bits = 20 + k2;
                continue;
            }
            if (bits == 0) bits = 2;
        }
    }
    return bits;
}

/// @return PULSAR_N_VOCAB
static uint32_t ds4_logits_width(const pulsar_engine *) { return PULSAR_N_VOCAB; }

/// pulsar_family::tp_shape.
/// @param[out] n_layer exchange slots
/// @param[out] n_embd all-reduce width
/// @param[out] n_vocab vocab
static void ds4_tp_shape(const pulsar_engine *, uint32_t *n_layer, uint32_t *n_embd, uint32_t *n_vocab) {
    *n_layer = PULSAR_N_LAYER;
    *n_embd = PULSAR_N_EMBD;
    *n_vocab = PULSAR_N_VOCAB;
}

/// @return the shape profile's name
static const char *ds4_model_name(const pulsar_engine *) { return PULSAR_MODEL_SHAPE_NAME; }
/* The id each profile is served as, spelled like the other families' (qwen3.8-flash-next, nemotron-3.5-super): variant
 * 1 was V4 Pro's before L218 made it V4.1 Flash, and served "deepseek-v4-pro" until 2026-10-30. */
/// @return the API model id
static const char *ds4_served_model_id(const pulsar_engine *) {
    return g_pulsar_shape.variant == PULSAR_VARIANT_V41 ? "deepseek-v4.1-flash" : "deepseek-v4-flash";
}

/* The conversation format follows the profile: 0731 has its own template,
 * Vision-Exp and V4.1 share one (pulsar_shape::variant). */
/// @return the profile's chat format
static pulsar_chat_format ds4_chat_format(const pulsar_engine *) {
    return g_pulsar_shape.variant == PULSAR_VARIANT_V4 ? PULSAR_CHAT_DS4_V4 : PULSAR_CHAT_DS4_V41;
}

/* 0 and 1 are the two profiles' ids since before families existed; disk-KV
 * files carry them. */
/// @return PULSAR_MODEL_VARIANT
static int ds4_model_id(const pulsar_engine *) { return (int)PULSAR_MODEL_VARIANT; }
/* YaRN: the original context the rope was trained at, times the scaling factor (both validated at load) */
/// @return tokens
static uint64_t ds4_trained_context(const pulsar_engine *) {
    return (uint64_t)((double)PULSAR_ROPE_ORIG_CTX * (double)PULSAR_ROPE_SCALE_FACTOR);
}

/* L284: the family's sync is the core's (sync_driver.cpp) over its ops.
 *
 * The state agrees with the view when nothing has made the bank another position's: no multiseq step left the
 * per-bank carry to re-establish (mseq_dirty), no rewind left the compressor lanes stale, and no kv source's
 * compressed frontier is AHEAD of the view (L148: the bank kept decoding after the view was recorded -- the plain
 * batched lane folds generated tokens back late).  A view that does not agree is resumed from a grid checkpoint
 * (the restore re-establishes every frontier at G) or rebuilt from 0, never continued. */
/// @param s the session
/// @return true when the bank's state is the view's
static bool ds4_sync_state_agrees(pulsar_session *s) {
    const pulsar_gpu_graph *g = s->graph;
    const uint32_t bank = gpu_graph_cur_bank(g);
    if (s->mseq_dirty || g->ms_comp_state_stale[bank]) return false;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++)
        if (gpu_graph_layer_is_kv_source(il) &&
            gpu_graph_n_comp(g, bank, il) > (uint32_t)s->checkpoint.len / pulsar_layer_compress_ratio(il))
            return false;
    return true;
}

/* Position 0 on the installed bank: the frontiers zeroed, its checkpoints dropped, the compressor lanes the
 * canonical empty group -- the one place this session's per-bank truth is re-established, so the multiseq mark
 * goes too (other banks hold other slots' positions; a reset here says nothing about them). */
/// @param s the session
/// @return false when the graph reset fails
static bool ds4_sync_reset_bank(pulsar_session *s) {
    if (!gpu_graph_reset_prefill_state(s->graph)) return false;
    gpu_graph_engram_forget(s->graph, gpu_graph_cur_bank(s->graph));
    s->mseq_dirty = false;
    return true;
}

/* One prefill chunk (pulsar_prefill_chunk_fn): the layer-major forward of rows [pos0, pos0 + rows), the last chunk
 * headed into the session's logits (a non-final chunk's head would be overwritten unread), mid-chunk display
 * progress, and the prefill frontier (L195) advanced to the chunk's end. */
/// @param s the session
/// @param prompt the prompt
/// @param pos0 first row
/// @param rows rows
/// @param last head this chunk
/// @return false (printed) on a failed forward
static bool ds4_prefill_chunk(pulsar_session *s, const pulsar_tokens *prompt, uint32_t pos0, uint32_t rows, bool last,
                              void *) {
    pulsar_engine *e = s->engine;
    if (!gpu_graph_prefill_layer_major(s->graph, &e->model, &e->weights, prompt, pos0, rows, last ? s->dev.live : NULL,
                                       false, NULL, s->display_progress, s->display_progress_ud)) {
        if (pulsar_gpu_synchronize() == 0)
            fprintf(stderr, "pulsar: GPU synchronize after chunked prefill failure also failed\n");
        return false;
    }
    if (last) s->dev.host_stale = true;   /* L302: the head wrote the live row on the device */
    s->prefill_frontier = (int)(pos0 + rows);
    return true;
}

/* The prefill from `start`: the host history above it goes (trim_history: the image records, the lookahead, the
 * drafter's window rows -- a cold rebuild's included), the prefill frontier stands at it, and the sync's images are
 * lent to the graph for the merge (gpu_graph_prefill_layer_major reads g->vision_req) for exactly this prefill. */
/// @param s the session
/// @param prompt the prompt
/// @param start first position to prefill
/// @return pulsar_prefill_loop's
static int ds4_sync_prefill(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start) {
    if ((int)start < s->checkpoint.len) s->trim_history((int)start);
    s->prefill_frontier = (int)start;
    const pulsar_vision_request vreq = { s->sync_images, s->sync_n_images, s->engine };
    pulsar_gpu_graph *g = s->graph;
    const pulsar_vision_request *prev = g->vision_req;
    g->vision_req = s->sync_n_images > 0 ? &vreq : NULL;
    const int rc = pulsar_prefill_loop(s, prompt, start, ds4_prefill_chunk, NULL);
    g->vision_req = prev;
    return rc;
}

/// @param s the session
/// @param[out] out the graph's prefill shape
static void ds4_sync_prefill_shape(pulsar_session *s, pulsar_prefill_shape *out) {
    *out = gpu_graph_prefill_shape(s->graph);
}

/// DeepSeek's ops for the core sync driver.
static const pulsar_sync_ops k_ds4_sync = {
    /* .name          = */ "deepseek-v4",
    /* .state_agrees  = */ ds4_sync_state_agrees,
    /* .reset_bank    = */ ds4_sync_reset_bank,
    /* .prefill       = */ ds4_sync_prefill,
    /* .prefill_shape = */ ds4_sync_prefill_shape,
};

/* ONE LANE (L129/L130): the row is a 1-row batch of the batched step the server decodes with, on the session's own
 * bank (bank 0 maps to the classic tensors when no pool is allocated), so every tool built on the classic API
 * measures the code production runs.  The core's eval refused a view the state does not agree with before this
 * (the multiseq mark, a stale compressor, a frontier ahead -- ds4_sync_state_agrees), which is what makes a
 * one-bank batch on these counters this bank's truth. */
/// @param s the session
/// @param token the token
/// @param err message buffer
/// @param errlen its size
/// @return 0 decoded; 1 + err on a failure
static int ds4_eval_row(pulsar_session *s, int token, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    int ms_tok[1] = { token };
    int32_t ms_pos[1] = { (int32_t)s->checkpoint.len };
    int32_t ms_bank[1] = { (int32_t)(s->graph->banks.n_banks ? s->graph->banks.cur_bank : 0u) };
    /* rc: 0 = recoverable pre-arm reject, 1 = success, else fatal mid-sweep */
    if (gpu_graph_decode_multiseq_batch(s->graph, &e->model, &e->weights, ms_tok, ms_pos, ms_bank, 1u, NULL,
                                        NULL, 0u, /*capture_cur=*/true, NULL) == 1 &&
        s->live_from_rows(s->graph->spec_logits, 0))
        return 0;
    if (err) snprintf(err, errlen, "%s decode failed", pulsar_backend_name(e->backend));
    return 1;
}

/// pulsar_family_session_ops::decode_multiseq: pulsar_session::decode_multiseq.
/// @param s the session
/// @param reqs [n] requests
/// @param n rows
/// @param[out] logits rows
/// @param logits_cap the TOTAL float capacity of `logits` (all rows, not one row)
/// @param err message buffer
/// @param errlen its size
/// @return the session method's
static int ds4_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n,
                               float *logits, int logits_cap, char *err, size_t errlen) {
    return s->decode_multiseq(reqs, n, logits, logits_cap, err, errlen);
}

/// pulsar_family_session_ops::decode_mixed: pulsar_session::decode_mixed.
/// @param s the session
/// @param reqs requests
/// @param n_rows rows
/// @param[out] logits rows
/// @param logits_cap the TOTAL float capacity of `logits` (all rows, not one row)
/// @param[out] out_n_rows rows headed
/// @param max_head_runs prompt runs headed at most
/// @param err message buffer
/// @param errlen its size
/// @return the session method's
static int ds4_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                            float *logits, int logits_cap, uint32_t *out_n_rows,
                            uint32_t max_head_runs, char *err, size_t errlen) {
    return s->decode_mixed(reqs, n_rows, logits, logits_cap, out_n_rows, max_head_runs, err, errlen);
}

/* L260: an invalidated bank is EMPTY on the device too -- its compressed frontier at the position law for 0, its
 * compressor group empty, its checkpoints and the drafter's window rows gone (rewind(0)) -- so any lane may start it
 * at position 0 and no next conversation can restore this one's rows. */
/// @param s the session
static void ds4_invalidate(pulsar_session *s) { s->rewind(0); }
/// pulsar_family_session_ops::decode_fused: pulsar_session::decode_fused.
/// @param s the session
/// @param reqs requests
/// @param n_rows rows
/// @param shape the step's shape
/// @param[out] logits rows
/// @param logits_cap the TOTAL float capacity of `logits` (all rows, not one row)
/// @param[out] out_n_rows rows headed
/// @param err message buffer
/// @param errlen its size
/// @return the session method's
static int ds4_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                            const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                            char *err, size_t errlen) {
    return s->decode_fused(reqs, n_rows, shape, logits, logits_cap, out_n_rows, err, errlen);
}

/// DeepSeek's session forward.
static const pulsar_family_session_ops k_ds4_session_ops = {
    /* .create          = */ pulsar_ds4_session_create,
    /* .destroy         = */ pulsar_ds4_session_destroy,
    /* .sync            = */ &k_ds4_sync,
    /* .eval_row        = */ ds4_eval_row,
    /* .decode_multiseq = */ ds4_decode_multiseq,
    /* .decode_mixed    = */ ds4_decode_mixed,
    /* .invalidate      = */ ds4_invalidate,
    /* .decode_fused    = */ ds4_decode_fused,
    /* .fused_heads_max = */ PULSAR_SPEC_ROW_BUDGET,   /* n_dec + headed runs <= the spec-logits block */
};

/* L281: DeepSeek's image geometry -- a sentinel is `vocab_size + role`, and a block is compressor pads, IMAGE_START,
 * the image rows, IMAGE_END (vision_span_extent, the one scan the planner and the merge share). */
/// @param id a token id
/// @return id >= PULSAR_N_VOCAB
static bool ds4_is_image_sentinel(const pulsar_engine *, int32_t id) { return id >= (int32_t)PULSAR_N_VOCAB; }
/// @param ids the sequence
/// @param n its length
/// @param start where to look
/// @param[out] len the block's length
/// @return true when a block starts at start
static bool ds4_image_block_extent(const pulsar_engine *, const int32_t *ids, int n, int start, int *len) {
    return vision_span_extent(ids, n, (int)PULSAR_N_VOCAB, start, len) != 0;
}
/* L268: the rest of DeepSeek's image front -- its placeholder token, one image's sentinel block (vision.cpp), and its
 * tower's block rows over the core's cache */
/// @param e the engine
/// @return the vocab's image id
static int ds4_image_placeholder_id(const pulsar_engine *e) { return e->vocab.image_id; }
/// @param img the image
/// @param[in,out] out ids appended
/// @param err message buffer
/// @param errlen its size
/// @return vision_ds4_expand's
static bool ds4_image_expand(const pulsar_engine *, const pulsar_image_ref *img, pulsar_tokens *out, char *err,
                             size_t errlen) {
    return vision_ds4_expand(img, out, err, errlen);
}
/// @return PULSAR_N_EMBD
static uint32_t ds4_image_row_width(const pulsar_engine *) { return (uint32_t)PULSAR_N_EMBD; }
/// @param e the engine
/// @param img the image
/// @param block_len the block's rows
/// @param tower cached tower rows, or NULL
/// @param n_tower their count
/// @param[out] tower_out a fresh encode's rows
/// @param[out] n_tower_out their count
/// @param[out] out the block's rows
/// @param err message buffer
/// @param errlen its size
/// @return vision_ds4_block_rows's
static bool ds4_image_block_rows(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *, int block_len,
                                 const uint16_t *tower, int n_tower, uint16_t **tower_out, int *n_tower_out,
                                 uint16_t *out, char *err, size_t errlen) {
    return vision_ds4_block_rows(&e->vision_weights, &e->model, img, block_len, tower, n_tower, tower_out,
                                 n_tower_out, out, err, errlen);
}
/// DeepSeek's image front.
static const pulsar_family_vision k_ds4_vision = {
    /* .is_sentinel      = */ ds4_is_image_sentinel,
    /* .block_extent     = */ ds4_image_block_extent,
    /* .placeholder_text = */ PULSAR_IMAGE_PLACEHOLDER,
    /* .placeholder_id   = */ ds4_image_placeholder_id,
    /* .expand           = */ ds4_image_expand,
    /* .row_width        = */ ds4_image_row_width,
    /* .block_rows       = */ ds4_image_block_rows,
};

/// The DeepSeek V4 family.
const pulsar_family PULSAR_FAMILY_DEEPSEEK4 = {
    /* .id           = */ PULSAR_FAMILY_ID_DEEPSEEK4,
    /* .arch         = */ "deepseek4",
    /* .name         = */ "DeepSeek V4",
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SPEC | PULSAR_FAMILY_CAP_PAYLOAD |
                          PULSAR_FAMILY_CAP_REWIND | PULSAR_FAMILY_CAP_VISION | PULSAR_FAMILY_CAP_TP |
                          PULSAR_FAMILY_CAP_IMATRIX | PULSAR_FAMILY_CAP_CHAT |
                          PULSAR_FAMILY_CAP_SEGMENTS | PULSAR_FAMILY_CAP_MIXED_PREFILL,
    /* .load         = */ pulsar_ds4_family_load,
    /* .after_gpu    = */ pulsar_ds4_family_after_gpu,
    /* .logits_width = */ ds4_logits_width,
    /* .model_name   = */ ds4_model_name,
    /* .served_id    = */ ds4_served_model_id,
    /* .chat_format  = */ ds4_chat_format,
    /* .model_id     = */ ds4_model_id,
    /* .trained_ctx  = */ ds4_trained_context,
    /* .rope_id      = */ pulsar_family_rope_fixed,
    /* .tp_shape     = */ ds4_tp_shape,
    /* .drafter      = */ ds4_drafter,
    /* .quant_bits   = */ ds4_quant_bits,
    /* .spec         = */ &k_ds4_spec_target,
    /* .session      = */ &k_ds4_session_ops,
    /* .tokenizer    = */ &k_ds4_tokenizer,
    /* .tp_slices    = */ pulsar_ds4_tp_slices,
    /* .act_kind     = */ PULSAR_ACT_KIND_SLOT,
    /* .vision       = */ &k_ds4_vision,
    /* .banks        = */ &k_ds4_bank_ops,
    /* .imatrix      = */ &k_ds4_imatrix,
    /* .driver       = */ NULL,   /* L295 F4: its own forward (gpu_decode.cpp, gpu_prefill.cpp) */
};

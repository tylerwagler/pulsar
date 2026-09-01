#include "pulsar_engine_internal.h"


/* Read an HC residual CARRIER (BF16 storage under task #62) into an f32 host
 * buffer, expanding each stored sample (BF16->f32 is an exact bit-extension:
 * the stored 16 bits are the high half of the f32). Used ONLY by the dev-only
 * layer-0 parity self-test (since removed) and the env-gated DSpark
 * dumps — never the production decode path. n is a sample count. */
/* Host-side f32 -> HC carrier store (task #62). Round-to-nearest-even so a host
 * staged write matches the GPU's __float2bfloat16 store path. NaN is
 * CANONICALIZED to 0x7FFF, which is what cvt.rn.bf16.f32 emits on sm_80+ (and
 * what CUDA's software path returns) — passing the payload through in the high
 * half would NOT match. Inf needs no special case: it rounds exactly through
 * the RNE path below. `dst` is raw carrier bytes, n a sample count. */
void pulsar_store_hc_carrier_f32(void *dst, const float *src, uint64_t n) {
    uint16_t *d = (uint16_t *)dst;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t x;
        memcpy(&x, &src[i], sizeof(x));
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) {       /* NaN -> canonical */
            d[i] = 0x7FFFu;
        } else {
            const uint32_t bias = 0x7FFFu + ((x >> 16) & 1u);   /* RNE; Inf exact */
            d[i] = (uint16_t)((x + bias) >> 16);
        }
    }
}


int pulsar_read_hc_carrier_f32(const pulsar_gpu_tensor *t, uint64_t off_elems,
                            float *out, uint64_t n) {
    uint16_t *tmp = (uint16_t *)xmalloc((size_t)n * sizeof(uint16_t));
    int rc = pulsar_gpu_tensor_read((pulsar_gpu_tensor *)t, off_elems * PULSAR_HC_ELT_SIZE,
                                 tmp, n * PULSAR_HC_ELT_SIZE);
    if (rc == 0) {
        /* Read failed and left tmp uninitialized — do NOT convert it. Callers
         * (the DSpark dumps at session.cpp) memset `out` to zero beforehand and
         * ignore the return, relying on "zeros on failure"; converting garbage
         * here would silently write heap noise into the dump. */
        free(tmp);
        return 0;
    }
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)tmp[i] << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
    free(tmp);
    return rc;
}

/* The same, for the stored Q buffer (f16 since L045).  A plain f32 read
 * would take n*sizeof(float) bytes from a buffer holding
 * n*PULSAR_Q_ELT_SIZE -- a 2x out-of-bounds read.  (The tensor's esz field
 * records the width now, but tensor_read is deliberately byte-oriented, so
 * this wrapper is still the only f32-decoding host read of a Q buffer.) */
static_assert(PULSAR_Q_ELT_SIZE == 2u,
              "pulsar_read_q_f32 decodes f16; update it if the Q element type moves");
int pulsar_read_q_f32(const pulsar_gpu_tensor *t, uint64_t off_elems,
                      float *out, uint64_t n) {
    uint16_t *tmp = (uint16_t *)xmalloc((size_t)n * sizeof(uint16_t));
    int rc = pulsar_gpu_tensor_read((pulsar_gpu_tensor *)t, off_elems * PULSAR_Q_ELT_SIZE,
                                 tmp, n * PULSAR_Q_ELT_SIZE);
    if (rc == 0) {
        /* Same contract as the HC reader: a failed read leaves tmp
         * uninitialised, so do NOT convert it into the caller's buffer. */
        free(tmp);
        return 0;
    }
    for (uint64_t i = 0; i < n; i++) out[i] = f16_to_f32(tmp[i]);
    free(tmp);
    return rc;
}






























static bool gpu_graph_decode_kv_store(
        pulsar_gpu_tensor *kv,
        pulsar_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {

    return pulsar_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             PULSAR_N_HEAD_DIM,
                                             PULSAR_N_ROT,
                                             gpu_graph_f32_store_observed_any()) != 0;
}




/* PULSAR_PREFILL_SLICE=<N>: process the prefill [indexer score -> top-k ->
 * indexed attention] sequence in <=N-token slices so the two ctx-scaling f32
 * work buffer (indexer_scores) is allocated with only N token
 * rows instead of prefill_cap.  Defaults to 512 (validated bit-exact);
 * 0 restores the historical full-chunk buffers. */
uint32_t gpu_graph_prefill_slice(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("PULSAR_PREFILL_SLICE");
        long v = (e && e[0]) ? strtol(e, NULL, 10) : 512;
        cached = v > 0 ? v : 0;
    }
    return (uint32_t)cached;
}

uint64_t gpu_graph_attn_comp_cache_row_bytes(void) {
    /* One row format for every KV buffer since the L111 unification; this
     * asks the backend so the seam stays a question, not an assumption. */
    return pulsar_gpu_attn_pack_rowbytes(PULSAR_N_HEAD_DIM);
}

/* The comment that stood here described gpu_graph_attn_comp_read_cache: a
 * dequantise of the packed comp pool into an f32 shadow, for prefill consumers
 * that could only read f32.  Both are gone -- every prefill consumer reads
 * PULSAR_ATTN_PACK rows (2026-08-18), and the attention kernels no longer take a
 * comp format parameter at all -- but the comment outlived the function and
 * ended up describing the unrelated one below. */

static bool gpu_graph_weight_is_plain_or_mxfp8(const pulsar_tensor *w) {
    return pulsar_weight_is_plain_or_mxfp8(w->type);
}




pulsar_gpu_tensor *gpu_graph_attn_comp_update_target(
        pulsar_gpu_graph *g,
        uint32_t       il) {
    (void)il;
    return g->attn_comp_stage;
}



uint32_t gpu_graph_attn_comp_update_row(uint32_t row) {
    (void)row;
    return 0u;
}



bool gpu_graph_commit_attn_comp_stage(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    {
        /* Quantize+pack the `rows` f32 rows staged in attn_comp_stage into the
         * packed comp cache at first_row.  The kernel CAN also fp8-roundtrip
         * the stage rows in place so the stage keeps the exact f32-pipeline
         * values -- but attn_comp_stage has no compute reader (every use is an
         * output target or the offset-0 row view feeding a "KVcompress" dump),
         * so that writeback is observer-only and now runs only when one is
         * watching (L094).  The pack itself MUST
         * be the only fp8 quantize of the row: re-quantizing an already-
         * roundtripped block is NOT bit-idempotent when the block amax sits on
         * a scale boundary (the recomputed scale can shift one step and
         * re-round small values, e.g. subnormal ties) — that is why the
         * prefill/replay producers pass quantize_fp8=false under pack. */
        if (rows == 0) return true;
        if (!g || il >= PULSAR_N_LAYER || !g->layer_attn_comp_cache[il] || !g->attn_comp_stage) {
            return false;
        }
        if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row) {
            return false;
        }
        if (pulsar_gpu_attn_pack_quantize_store_tensor(g->attn_comp_stage,
                                                    g->layer_attn_comp_cache[il],
                                                    first_row, rows,
                                                    PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
                                                    gpu_graph_f32_store_observed_any()) == 0) {
            return false;
        }
        /* plan-33 inc C: byte-replace the ratio-4 boundary row after any commit
         * below the fork keep threshold (no-op when ms_emit_keep is 0). */
        return gpu_graph_emit_keep_restore(g, il,
                g->banks.n_banks ? g->banks.cur_bank : 0u, first_row, rows, false);
    }
}



bool gpu_graph_commit_attn_comp_stage_bank(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       bank,
        uint32_t       first_row,
        uint32_t       rows) {
    if (rows == 0) return true;
    if (!g || il >= PULSAR_N_LAYER || !g->attn_comp_stage) return false;
    if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }
    pulsar_gpu_tensor *cache = gpu_graph_bank_attn_comp_view(g, il, bank);
    if (!cache) return false;
    const bool ok = pulsar_gpu_attn_pack_quantize_store_tensor(
            g->attn_comp_stage, cache, first_row, rows,
            PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
            gpu_graph_f32_store_observed_any()) != 0;
    pulsar_gpu_tensor_free(cache);
    /* plan-33 inc C: boundary-row restore for the explicit-bank commit path. */
    return ok && gpu_graph_emit_keep_restore(g, il, bank, first_row, rows, false);
}



pulsar_gpu_tensor *gpu_graph_attn_comp_row_view(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    (void)il; (void)row;
    return pulsar_gpu_tensor_view(g->attn_comp_stage,
                               0,
                               (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float));
}



pulsar_gpu_tensor *gpu_graph_attn_comp_prefill_target(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    (void)il; (void)first_row; (void)rows;
    return g->attn_comp_stage;
}



void gpu_graph_attn_comp_prefill_target_free(pulsar_gpu_tensor *t) {
    /* The target is always the persistent attn_comp_stage, which must not be
     * freed. Kept as a named no-op so the call sites still read as paired. */
    (void)t;
}



/* Encode one DS4 decode layer on GPU.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);


bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);


bool gpu_graph_decode_stage_profile_enabled(uint32_t il);


bool gpu_graph_matmul_plain_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);



/* Decode-only fused RMSNorm + HC-mix GEMV.  Byte-identical to the
 * rms_norm_plain -> matmul_f16 pair it replaces (see pulsar_cuda_hc_router.cu);
 * it exists because that pair ran a 1-block kernel and then a 24-block kernel
 * with a 64 KB f32 scratch round trip between them, for ~5.4% of decode.
 * src_hc is an HC residual carrier (BF16 under task #62); the fused kernel
 * reads it via pulsar_hc_load, exactly as rms_norm_plain_tensor does.
 * Non-F16 mix weights keep the original two-kernel path. */
static bool gpu_graph_norm_mix_plain(
        pulsar_gpu_graph        *g,
        const pulsar_model      *model,
        const pulsar_tensor     *w,
        uint64_t              hc_dim,
        uint64_t              out_dim,
        const pulsar_gpu_tensor *src_hc,
        pulsar_gpu_tensor       *out) {
    /* Any storage the fused kernel can read takes the fusion; only an fp8 mix
     * weight still needs the unfused pair. Was F16-only, which quietly dropped
     * the fusion -- and its ~5.4% of decode -- as soon as hc_*_fn moved. */
    if (w->type == PULSAR_TENSOR_BF16 ||
        w->type == PULSAR_TENSOR_F32) {
        return pulsar_gpu_hc_norm_mix_tensor(out, model->map, model->size,
                                              w->abs_offset, hc_dim, out_dim,
                                              src_hc, PULSAR_RMS_EPS,
                                              w->type) != 0;
    }
    if (!pulsar_gpu_rms_norm_plain_tensor(g->flat_hc, src_hc, (uint32_t)hc_dim, PULSAR_RMS_EPS)) return false;
    return gpu_graph_matmul_plain_tensor(out, model, w, hc_dim, out_dim, g->flat_hc, 1);
}






bool gpu_graph_dspark_project_main_x(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w) {
    const uint64_t E = PULSAR_N_EMBD;
    const uint64_t concat_dim = 3ull * E;

    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h[i]) return false;
    }

    /* Persistent scratch: this runs up to 5x per fused spec step and each
     * cudaMalloc/cudaFree pair serializes the device. */
    pulsar_gpu_tensor *target_concat = g->dspark_concat;
    pulsar_gpu_tensor *proj_out = g->dspark_proj_out;
    if (!target_concat || !proj_out) return false;

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        ok = pulsar_gpu_tensor_copy(target_concat, (uint64_t)i * E * sizeof(float),
                                 g->dspark_target_h[i], 0, E * sizeof(float)) != 0;
        if (!ok) break;
    }

    if (ok) {
        ok = pulsar_gpu_matmul_mxfp8_tensor(proj_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->main_proj->abs_offset,
                                          concat_dim, E,
                                          target_concat, 1) != 0;
    }

    if (ok) {
        ok = pulsar_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            proj_out,
                                            dspark_model->map,
                                            dspark_model->size,
                                            w->main_norm->abs_offset,
                                            (uint32_t)E,
                                            PULSAR_RMS_EPS,
        w->main_norm->type == PULSAR_TENSOR_BF16) != 0;
    }

    return ok;
}

void gpu_graph_dspark_seed_draft_kv(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        uint32_t                 n_rows) {
    const uint64_t kv_bytes = (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float);
    /* Persistent scratch (dspark_seed_*): the fused loop seeds up to 5 rows per
     * step across 3 layers; per-call cudaMalloc/cudaFree here was ~9 device-
     * serializing pairs per call. */
    pulsar_gpu_tensor *kv_out = g->dspark_seed_kv;
    pulsar_gpu_tensor *kv_norm = g->dspark_seed_norm;
    pulsar_gpu_tensor *kv_rot = g->dspark_seed_rot;
    if (!kv_out || !kv_norm || !kv_rot) return;

    /* ⚠ THE THREE LAYERS MUST ADVANCE IN LOCKSTEP, so a partial failure rolls the
     * whole seed back.  This loop used to `continue` past a failed layer without
     * advancing dspark_n_raw[li], leaving that layer's ring counter behind its
     * siblings' PERMANENTLY: every later draft then RoPEs at a different sequence
     * position per layer.  The verifier still rejects the bad drafts, so output
     * stays exact -- which is precisely why it was invisible.  What degrades is
     * ACCEPTANCE, silently, for the rest of the session.
     *
     * Rolling the counters back is enough to stay consistent: rows already
     * written sit at ring slots the counters no longer reach, so nothing reads
     * them and the next successful seed overwrites the same slots. */
    const uint32_t n_raw_entry[3] = {
        g->dspark_n_raw[0], g->dspark_n_raw[1], g->dspark_n_raw[2]
    };
    bool seeded = true;
    for (int li = 0; li < 3 && seeded; li++) {
        if (!pulsar_gpu_matmul_mxfp8_tensor(kv_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->layer[li].attn_kv->abs_offset,
                                          PULSAR_N_EMBD, PULSAR_N_HEAD_DIM,
                                          g->dspark_main_x, 1)) {
            seeded = false;
            break;
        }
        if (!pulsar_gpu_rms_norm_weight_tensor(kv_norm, kv_out,
                                             dspark_model->map,
                                             dspark_model->size,
                                             w->layer[li].attn_kv_a_norm->abs_offset,
                                             PULSAR_N_HEAD_DIM, PULSAR_RMS_EPS,
        w->layer[li].attn_kv_a_norm->type == PULSAR_TENSOR_BF16)) {
            seeded = false;
            break;
        }
        /* Seed one KV row per committed position.  Each row is RoPE'd at its OWN
         * sequence position (dspark_n_raw[li]) and fp8-rounded, matching the
         * draft-forward KV and the DSparkAttention reference (main_kv is rotated
         * at start_pos).  kv_norm holds the un-rotated vector; we rotate a fresh
         * copy per position so multi-row seeds (accepted drafts) land at distinct
         * positions rather than all sharing the first row's rotation. */
        for (uint32_t i = 0; i < n_rows; i++) {
            const uint32_t pos = g->dspark_n_raw[li];
            const uint32_t row = pos % PULSAR_DSPARK_DRAFT_WINDOW;
            /* Both copies were unchecked: a failed stage used to leave the row
             * unwritten while the counter still advanced past it, i.e. the ring
             * would be read as if it held a seeded position. */
            if (!pulsar_gpu_tensor_copy(kv_rot, 0, kv_norm, 0, kv_bytes)) {
                seeded = false;
                break;
            }
            if (!pulsar_gpu_rope_tail_tensor(kv_rot, 1, PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
                                     pos, 0, false,
                                     (float)PULSAR_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
                                     PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW, NULL)) {
                seeded = false;
                break;
            }
            /* No fake-quantise before the store.  The TARGET path this seed has
             * to agree with (gpu_graph_decode_kv_store -> attn_pack_store_kernel
             * with x = kv) quantises the true f32 exactly ONCE, packing to the
             * ring and writing back in the same pass.  Round-tripping here first
             * made the seed quantise twice, which is the ~5%-misround pattern
             * norm_kv warns about and would let a seeded row differ from the
             * target's row for the same token -- the one thing this seed exists
             * to make identical. */
            /* Store through the ring's own writer, not a byte copy.  This was
             * pulsar_gpu_tensor_copy at row*kv_bytes -- an f32 row at a 2048 B
             * stride -- which stopped being the ring's layout when it became
             * PULSAR_ATTN_PACK (584 B) in 157cd1d.  It then failed its bounds
             * check, the seed aborted, and acceptance collapsed far enough that
             * the yield-quench dropped the request to plain decode.  A writer
             * that bypasses the store API is a writer the next format change
             * will miss, which is exactly what happened. */
            if (!pulsar_gpu_store_raw_kv_tensor(g->dspark_raw_cache[li], kv_rot,
                                                PULSAR_DSPARK_DRAFT_WINDOW, row,
                                                PULSAR_N_HEAD_DIM)) {
                seeded = false;
                break;
            }
            g->dspark_n_raw[li]++;
        }
    }
    if (!seeded) {
        g->dspark_n_raw[0] = n_raw_entry[0];
        g->dspark_n_raw[1] = n_raw_entry[1];
        g->dspark_n_raw[2] = n_raw_entry[2];
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                    "pulsar: WARNING drafter KV seed failed (%u row(s)) -- rolled the "
                    "three layers back to %u/%u/%u to keep them in step; this step's "
                    "draft is unseeded and acceptance will dip until the next seed\n",
                    n_rows, n_raw_entry[0], n_raw_entry[1], n_raw_entry[2]);
        }
    }
}

bool gpu_graph_dspark_draft_forward(
        pulsar_gpu_graph          *g,
        const pulsar_model         *base_model,
        const pulsar_weights       *base_weights,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        pulsar_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft) {
    /* L106 K2a: the drafter hand-rolls its attention half and never passes
     * through the batch encode whose first act is the gact disarm -- so a
     * gact entry armed and noted by a prior prefill chunk or spec verify of
     * EQUAL WIDTH would satisfy the cache probe when a deeper draft
     * (n_draft >= 5) takes the tensor-core "a" arm, handing the drafter the
     * MAIN model's activation encoding: corrupted draft logits, acceptance
     * collapse (output stays correct -- spec verify is protected).  Unreachable
     * at today's n_draft <= 4; a PREREQUISITE for L092 deeper drafts.  Disarm
     * here, unconditionally, exactly as the batch encode does per layer. */
    pulsar_gpu_mxfp8_gact_disarm();
    if (!g || !base_model || !base_weights || !dspark_model || !w ||
        !base_logits_out || n_draft == 0 || n_draft > 16 ||
        n_draft > g->prefill_cap)
        return false;

    if (!base_weights->token_embd || !base_weights->output)
        return false;

    /* Embed N draft tokens via main model's F16 token_embd → HC-expand.
     * L104 fix B: persistent graph-owned upload tensor (n_draft <= 16) -- this
     * was a cudaMalloc + blocking write + cudaFree per drafter forward. */
    pulsar_gpu_tensor *tokens_t = g->dspark_embed_tokens;
    if (!tokens_t) return false;
    if (!pulsar_gpu_tensor_write(tokens_t, 0, draft_ids, (uint64_t)n_draft * sizeof(int32_t)))
        return false;
    bool ok = pulsar_gpu_embed_tokens_hc_tensor(g->batch_cur_hc,
                                              tokens_t,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->token_embd->abs_offset,
                                              PULSAR_N_VOCAB,
                                              n_draft,
                                              PULSAR_N_EMBD,
                                              PULSAR_N_HC) != 0;
    if (!ok) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint32_t n_groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;

    const int prev_comp = g->comp_ratio_override;
    g->comp_ratio_override = 0;

    for (uint32_t li = 0; li < 3 && ok; li++) {
        const pulsar_layer_weights *layer = &w->layer[li];
        const uint32_t raw_cap = PULSAR_DSPARK_DRAFT_WINDOW;
        const uint32_t q_rank = (uint32_t)layer->attn_q_a->dim[1];
        /* Draft queries/KV sit at the frontier (the current main_kv was seeded at
         * dspark_n_raw[li]-1 just before this forward), so RoPE them at the real
         * position -- NOT 0.  The reference DSparkAttention rotates draft Q/KV at
         * start_pos+1 while the seeded main_kv is at start_pos. */
        const uint32_t pos0 = g->dspark_n_raw[li];

        /* --- HC pre-processing --- */
        /* Create views from batch working set */
        pulsar_gpu_tensor *hc_mix_view = pulsar_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        pulsar_gpu_tensor *hc_split_view = pulsar_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        pulsar_gpu_tensor *ffn_cur_view = pulsar_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_draft * PULSAR_N_EMBD * sizeof(float));
        ok = hc_mix_view && hc_split_view && ffn_cur_view;
        /* RMS norm: flat HC from batch_cur_hc */
        if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(
            g->batch_flat_hc, g->batch_cur_hc,
            (uint32_t)hc_dim, n_draft, PULSAR_RMS_EPS) != 0;
        /* HC → mix projection */
        if (ok) ok = gpu_graph_matmul_plain_tensor(
            hc_mix_view, dspark_model,
            layer->hc_attn_fn,
            hc_dim, mix_hc, g->batch_flat_hc, n_draft);
        /* HC split + weighted sum → attn_cur (E-dim) */
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_tensor(
            ffn_cur_view, hc_split_view, hc_mix_view,
            g->batch_cur_hc,
            dspark_model->map, dspark_model->size,
            layer->hc_attn_scale->abs_offset,
            layer->hc_attn_base->abs_offset,
            PULSAR_N_EMBD, PULSAR_N_HC,
            PULSAR_N_HC_SINKHORN_ITER, PULSAR_HC_EPS) != 0;
        /* Input RMS norm → batch_attn_norm */
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, ffn_cur_view,
            dspark_model->map, dspark_model->size,
            layer->attn_norm->abs_offset,
            PULSAR_N_EMBD, n_draft, PULSAR_RMS_EPS,
        layer->attn_norm->type == PULSAR_TENSOR_BF16) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_norm", g->batch_attn_norm,
                                             (uint64_t)n_draft * PULSAR_N_EMBD, li, pos0);
        /* L106 K2b: batch_attn_norm feeds TWO MXFP8 GEMMs (attn_q_a, attn_kv).
         * Unarmed, each independently re-quantised the same values from f32 --
         * the duplicate encode the act-cache design note called "the sanctioned
         * miss", per layer per draft.  Arming makes the first GEMM's quantise
         * land in the slot and the second hit it; bit-identical either way
         * (same encoder, same bytes), one encode instead of two. */
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_attn_norm, n_draft,
                                               PULSAR_N_EMBD);
        /* --- Q projection --- */
        if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(
            g->batch_qr, dspark_model->map, dspark_model->size,
            layer->attn_q_a->abs_offset,
            PULSAR_N_EMBD, q_rank, g->batch_attn_norm, n_draft) != 0;
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(
            g->batch_qr_norm, g->batch_qr,
            dspark_model->map, dspark_model->size,
            layer->attn_q_a_norm->abs_offset,
            q_rank, n_draft, PULSAR_RMS_EPS,
        layer->attn_q_a_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(
            g->batch_q, dspark_model->map, dspark_model->size,
            layer->attn_q_b->abs_offset,
            q_rank, PULSAR_N_HEAD * PULSAR_N_HEAD_DIM,
            g->batch_qr_norm, n_draft) != 0;
        /* Q head-norm + RoPE */
        if (ok) ok = pulsar_gpu_head_rms_norm_rope_tail_tensor(
            g->batch_q, n_draft,
            PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
            pos0, 0, false,
            (float)PULSAR_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW, PULSAR_RMS_EPS,
            NULL) != 0;

        /* --- KV projection --- */
        if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(
            g->batch_kv_raw, dspark_model->map, dspark_model->size,
            layer->attn_kv->abs_offset,
            PULSAR_N_EMBD, PULSAR_N_HEAD_DIM,
            g->batch_attn_norm, n_draft) != 0;
        /* K2b: last consumer of the armed attn_norm ran; disarm so a later
         * width-matched buffer reuse cannot hit this entry (same rule as the
         * head entries' redundant second lock). */
        pulsar_gpu_mxfp8_act_cache_disarm();
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(
            g->batch_kv, g->batch_kv_raw,
            dspark_model->map, dspark_model->size,
            layer->attn_kv_a_norm->abs_offset,
            PULSAR_N_HEAD_DIM, n_draft, PULSAR_RMS_EPS,
        layer->attn_kv_a_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) ok = pulsar_gpu_rope_tail_tensor(
            g->batch_kv, n_draft,
            PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
            pos0, 0, false,
            (float)PULSAR_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW, NULL) != 0;
        /* No fake-quantise pass here.  batch_kv has no reader after the store
         * below, and that store packs the rows itself -- so round-tripping first
         * only fed already-quantized values to a fresh quantizer, which is the
         * ~5%-misround pattern norm_kv warns about.  Packing the true f32 once
         * is both cheaper and closer to what the target path does for the same
         * token, which is what verification compares against. */

        /* --- Store draft KV transiently in ring buffer for attention --- */
        const uint32_t saved_n_raw = g->dspark_n_raw[li];
        const uint32_t kv_store_pos = saved_n_raw % raw_cap;
        if (ok) ok = pulsar_gpu_store_raw_kv_batch_tensor(
            g->dspark_raw_cache[li], g->batch_kv,
            raw_cap, kv_store_pos, n_draft, PULSAR_N_HEAD_DIM,
            NULL, NULL, 1) != 0;
        const uint32_t vis_raw = saved_n_raw + n_draft;
        const uint32_t cap_raw = vis_raw < raw_cap ? vis_raw : raw_cap;
        const uint32_t raw_start = vis_raw > raw_cap
            ? (vis_raw - raw_cap) % raw_cap : 0;

        /* --- Non-causal raw batch attention ---
         * Queries are at positions [saved_n_raw, saved_n_raw+n_draft).
         * Visible raw entries span [0, vis_raw) — all cached + current draft rows. */
        if (ok) ok = pulsar_gpu_attention_decode_raw_batch_heads_tensor(
            g->batch_heads,
            dspark_model->map, dspark_model->size,
            layer->attn_sinks->abs_offset,
            g->batch_q, g->dspark_raw_cache[li],
            n_draft, saved_n_raw,
            cap_raw, raw_cap, raw_start,
            0,
            PULSAR_N_HEAD, PULSAR_N_HEAD_DIM,
            1,
            NULL, NULL, 0, 1,
                                          NULL /* q pre-normed */) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_heads", g->batch_heads,
                                             (uint64_t)n_draft * PULSAR_N_HEAD * PULSAR_N_HEAD_DIM, li, pos0);
        /* Inverse-rotate the attention output's rope dims before the o
         * projection (reference: apply_rotary_emb(o, freqs_cis, inverse=True);
         * the verify/prefill path does the same via its "kqv_back" rope).
         * This call was MISSING here: wo_a/wo_b consumed position-rotated
         * rope dims -- 64/512 dims per head scrambled in every drafter block,
         * the ~30-point acceptance bug (#4). Verified vs the torch reference:
         * the engine attention output matches exactly WITHOUT inverse rope
         * (cos 0.99999) and diverges with it (cos 0.57), so everything
         * downstream of this line computed on corrupted features. */
        if (ok) ok = pulsar_gpu_rope_tail_tensor(
            g->batch_heads, n_draft,
            PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
            pos0, 0, true,
            (float)PULSAR_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW, NULL) != 0;
        /* --- Attention output projection (LoRA grouped) --- */
        if (ok) ok = pulsar_gpu_attention_output_batch_tensor(
            g->batch_attn_out, g->batch_attn_low,
            dspark_model->map, dspark_model->size,
            layer->attn_output_a->abs_offset,
            layer->attn_output_b->abs_offset,
            group_dim, rank, n_groups, PULSAR_N_EMBD,
            g->batch_heads, n_draft) != 0;
        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_out", g->batch_attn_out,
                                             (uint64_t)n_draft * PULSAR_N_EMBD, li, pos0);

        /* --- HC expand + split → batch_after_attn_hc --- */
        /* View sized to the real draft count: the CUDA side infers n_tokens from
         * out_hc->bytes, and the raw batch tensor is allocated at prefill capacity
         * (4096) -- passing it unviewed made every drafter block expand the FULL
         * capacity, ~2.8 ms per call x3 blocks = ~8 ms of pure waste per spec step. */
        if (ok) {
            pulsar_gpu_tensor *after_attn_view = pulsar_gpu_tensor_view(
                g->batch_after_attn_hc, 0,
                (uint64_t)n_draft * PULSAR_N_HC * PULSAR_N_EMBD * PULSAR_HC_ELT_SIZE);   /* carrier */
            ok = after_attn_view &&
                 pulsar_gpu_hc_expand_split_tensor(
                     after_attn_view, g->batch_attn_out,
                     g->batch_cur_hc, g->batch_hc_split,
                     PULSAR_N_EMBD, PULSAR_N_HC) != 0;
            pulsar_gpu_tensor_free(after_attn_view);
        }

        /* --- FFN batch (reuses existing function) --- */
        if (ok) ok = gpu_graph_encode_layer_ffn_batch(
            g, dspark_model, layer, li, pos0, n_draft);

        if (ok) gpu_graph_debug_dump_hc_tensor("dsp_after_attn_hc", g->batch_after_attn_hc,
                                             (uint64_t)n_draft * PULSAR_N_HC * PULSAR_N_EMBD, li, pos0);
        /* --- HC swap for next layer --- */
        if (ok) {
            pulsar_gpu_tensor *tmp = g->batch_cur_hc;
            g->batch_cur_hc = g->batch_next_hc;
            g->batch_next_hc = tmp;
        }
        if (ok) gpu_graph_debug_dump_hc_tensor("dsp_block_out", g->batch_cur_hc,
                                             (uint64_t)n_draft * PULSAR_N_HC * PULSAR_N_EMBD, li, pos0);
        /* Draft KV is transient; dspark_n_raw remains at the persistent count.
         * Committed positions are seeded via gpu_graph_dspark_seed_draft_kv(). */

        /* Views over the batch working set are per-iteration host structs;
         * free them each layer or they leak on every speculative block. */
        pulsar_gpu_tensor_free(hc_mix_view);
        pulsar_gpu_tensor_free(hc_split_view);
        pulsar_gpu_tensor_free(ffn_cur_view);
    }

    g->comp_ratio_override = prev_comp;

    /* Batch output head → N-token logits in g->spec_logits.  Use the DSpark
     * drafter's OWN hc_head + norm (dspark.2.*) with the shared vocab head, NOT
     * the main model's output head (which was corrupting the draft logits). */
    if (ok) {
        ok = gpu_graph_encode_dspark_output_head_batch(
            g, dspark_model, w, base_model, base_weights, n_draft, PULSAR_N_VOCAB);
    }

    /* The output head already wrote into g->spec_logits.  Callers that pass a
     * distinct buffer get a copy; the session passes g->spec_logits itself, so
     * skip the self-copy (a same-buffer cudaMemcpy is undefined). */
    if (ok && base_logits_out && base_logits_out != g->spec_logits) {
        const uint64_t logits_bytes = (uint64_t)n_draft * PULSAR_N_VOCAB * sizeof(float);
        ok = pulsar_gpu_tensor_copy(base_logits_out, 0,
                                  g->spec_logits, 0,
                                  logits_bytes) != 0;
    }

    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on GPU. */
bool gpu_graph_encode_output_head(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    bool ok = gpu_graph_norm_mix_plain(g, (const pulsar_model *)model, weights->output_hc_fn,
                                       hc_dim, PULSAR_N_HC, g->cur_hc, g->output_pre);
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_pre", g->output_pre, PULSAR_N_HC, PULSAR_N_LAYER, 0);
    }
    if (ok) ok = pulsar_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    PULSAR_N_HC,
                                                    PULSAR_HC_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_weights", g->output_weights, PULSAR_N_HC, PULSAR_N_LAYER, 0);
    }
    if (ok) ok = pulsar_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_N_HC) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc", g->output_embd, PULSAR_N_EMBD, PULSAR_N_LAYER, 0);
    }
    if (ok) ok = pulsar_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_RMS_EPS,
        weights->output_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_norm", g->output_norm, PULSAR_N_EMBD, PULSAR_N_LAYER, 0);
    }
    if (ok) {
        if (weights->output->type == PULSAR_TENSOR_BF16)
            ok = pulsar_gpu_matmul_bf16_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
        else
            ok = pulsar_gpu_matmul_mxfp8_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, PULSAR_N_LAYER, 0);
    }
    return ok;
}



/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and MXFP8 vocab projection.  For
 * tiny speculative suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
static bool gpu_graph_encode_output_head_batch_impl(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim);

/* L119 segment bracket: the output head is round-invariant at fixed n_tokens
 * (2026-08-27 audit — norm/mix/weights/vocab GEMM take only n_tokens and
 * vocab_dim; no position, no frontier, no readback), so at decode widths it
 * is captured once per n_tokens and replayed. Distinct head row counts
 * (mixed-lane head_cap modes) key separately. */
bool gpu_graph_encode_output_head_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens >= 1 && n_tokens <= 16 && g->banks.n_banks) {
        /* L119: parity-keyed like the FFN bracket — the head reads the
         * sweep-final hidden buffer, whose identity alternates per sweep
         * (odd layer count x per-layer pointer swap). */
        const uint64_t hc_parity =
            (uintptr_t)(const void *)g->batch_cur_hc >
            (uintptr_t)(const void *)g->batch_next_hc ? 1ull : 0ull;
        const uint64_t key = (hc_parity << 40) |
                             (0xFFull << 16) | (2ull << 8) | n_tokens;
        const int st = pulsar_gpu_seg_enter(key);
        if (st == 2) return true;
        if (st == 1) {
            const bool ok = gpu_graph_encode_output_head_batch_impl(
                    g, model, weights, n_tokens, vocab_dim);
            if (pulsar_gpu_seg_exit(key, ok ? 1 : 0)) return true;
            /* Capture failed (key now poisoned): the recorded work never ran,
             * and a mid-capture violation can fail an otherwise-good body —
             * run the body for real regardless of ok; a REAL body failure
             * simply fails again here and propagates. */
            return gpu_graph_encode_output_head_batch_impl(
                    g, model, weights, n_tokens, vocab_dim);
        }
    }
    return gpu_graph_encode_output_head_batch_impl(g, model, weights, n_tokens, vocab_dim);
}

static bool gpu_graph_encode_output_head_batch_impl(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap ||
        n_tokens > PULSAR_SPEC_LOGITS_ROWS || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    pulsar_gpu_tensor *output_pre = NULL;
    pulsar_gpu_tensor *output_weights = NULL;
    pulsar_gpu_tensor *output_embd = NULL;
    pulsar_gpu_tensor *output_norm = NULL;
    pulsar_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = pulsar_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * PULSAR_N_HC * sizeof(float));
    output_weights = pulsar_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * PULSAR_N_HC * sizeof(float));
    output_embd = pulsar_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    output_norm = pulsar_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    /* These heads REUSE batch_ffn_norm as output_norm scratch at the same n_tok
     * and in_dim the layer encodes armed it with, so an activation cache that
     * outlived a layer would hit here and quantize the vocab GEMM from stale
     * activations.  The FFN encode disarms at its exit; this is the second lock
     * on the same door, and it is free. */
    pulsar_gpu_mxfp8_act_cache_disarm();
    logits = pulsar_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre,
                                                 (const pulsar_model *)model,
                                                 weights->output_hc_fn,
                                             hc_dim,
                                             PULSAR_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = pulsar_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    PULSAR_N_HC,
                                                    PULSAR_HC_EPS) != 0;
    if (ok) ok = pulsar_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_N_HC) != 0;
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       PULSAR_N_EMBD,
                                                       n_tokens,
                                                       PULSAR_RMS_EPS,
        weights->output_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok) {
        if (weights->output->type == PULSAR_TENSOR_BF16)
            ok = pulsar_gpu_matmul_bf16_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
        else
            ok = pulsar_gpu_matmul_mxfp8_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
    }

    pulsar_gpu_tensor_free(logits);
    pulsar_gpu_tensor_free(output_norm);
    pulsar_gpu_tensor_free(output_embd);
    pulsar_gpu_tensor_free(output_weights);
    pulsar_gpu_tensor_free(output_pre);
    return ok;
}

/* DSpark drafter output head.  Collapses the drafter's final HC with the DSpark
 * block's OWN head (dspark.2.hc_head_fn/scale/base) and norm (dspark.2.norm),
 * then projects to vocab with the SHARED main output head (self.head in the
 * reference).  The plain gpu_graph_encode_output_head_batch used the MAIN model's
 * output_hc and output_norm weights for the drafter -- wrong weights that
 * corrupted the draft base logits (base0_hit was ~29%). */
bool gpu_graph_encode_dspark_output_head_batch(
        pulsar_gpu_graph            *g,
        const pulsar_model          *dspark_model,
        const pulsar_dspark_weights *dw,
        const pulsar_model          *base_model,
        const pulsar_weights        *bw,
        uint32_t                  n_tokens,
        uint64_t                  vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap ||
        n_tokens > PULSAR_SPEC_LOGITS_ROWS || !g->spec_logits) return false;
    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    pulsar_gpu_tensor *output_pre = pulsar_gpu_tensor_view(g->batch_hc_mix, 0, (uint64_t)n_tokens * PULSAR_N_HC * sizeof(float));
    pulsar_gpu_tensor *output_weights = pulsar_gpu_tensor_view(g->batch_hc_split, 0, (uint64_t)n_tokens * PULSAR_N_HC * sizeof(float));
    pulsar_gpu_tensor *output_embd = pulsar_gpu_tensor_view(g->batch_ffn_cur, 0, (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    pulsar_gpu_tensor *output_norm = pulsar_gpu_tensor_view(g->batch_ffn_norm, 0, (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    /* These heads REUSE batch_ffn_norm as output_norm scratch at the same n_tok
     * and in_dim the layer encodes armed it with, so an activation cache that
     * outlived a layer would hit here and quantize the vocab GEMM from stale
     * activations.  The FFN encode disarms at its exit; this is the second lock
     * on the same door, and it is free. */
    pulsar_gpu_mxfp8_act_cache_disarm();
    pulsar_gpu_tensor *logits = pulsar_gpu_tensor_view(g->spec_logits, 0, (uint64_t)n_tokens * vocab_dim * sizeof(float));
    bool ok = output_pre && output_weights && output_embd && output_norm && logits;
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                                     (uint32_t)hc_dim, n_tokens, PULSAR_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre, dspark_model, dw->hc_head_fn,
                                               hc_dim, PULSAR_N_HC, g->batch_flat_hc, n_tokens) != 0;
    if (ok) ok = pulsar_gpu_output_hc_weights_tensor(output_weights, output_pre,
                                                  dspark_model->map, dspark_model->size,
                                                  dw->hc_head_scale->abs_offset,
                                                  dw->hc_head_base->abs_offset,
                                                  PULSAR_N_HC, PULSAR_HC_EPS) != 0;
    if (ok) ok = pulsar_gpu_hc_weighted_sum_tensor(output_embd, g->batch_cur_hc, output_weights,
                                                PULSAR_N_EMBD, PULSAR_N_HC) != 0;
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(output_norm, output_embd,
                                                     dspark_model->map, dspark_model->size,
                                                     dw->final_norm->abs_offset,
                                                     PULSAR_N_EMBD, n_tokens, PULSAR_RMS_EPS,
        dw->final_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok) {
        if (bw->output->type == PULSAR_TENSOR_BF16)
            ok = pulsar_gpu_matmul_bf16_tensor(logits, base_model->map, base_model->size,
                                            bw->output->abs_offset, PULSAR_N_EMBD, vocab_dim,
                                            output_norm, n_tokens) != 0;
        else
            ok = pulsar_gpu_matmul_mxfp8_tensor(logits, base_model->map, base_model->size,
                                             bw->output->abs_offset, PULSAR_N_EMBD, vocab_dim,
                                             output_norm, n_tokens) != 0;
    }
    pulsar_gpu_tensor_free(logits);
    pulsar_gpu_tensor_free(output_norm);
    pulsar_gpu_tensor_free(output_embd);
    pulsar_gpu_tensor_free(output_weights);
    pulsar_gpu_tensor_free(output_pre);
    return ok;
}



bool gpu_graph_matmul_plain_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == PULSAR_TENSOR_F32) {
        return pulsar_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == PULSAR_TENSOR_BF16) {
        return pulsar_gpu_matmul_bf16_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    /* FP8_E4M3 and MXFP8_LT are the same numbers in two layouts, and
     * pulsar_gpu_matmul_mxfp8_tensor already tells them apart: an offset
     * registered as LT (gguf.cpp) resolves straight to the mmap, a plain one is
     * de-interleaved once into a device buffer. Both land on the same kernels.
     *
     * MXFP8_LT was absent here until 2026-08-17, and that absence was load-bearing
     * -- it is why the 21 indexer.attn_q_b tensors ship as plain type 38 and pay a
     * second resident copy. tensor_expect_plain_or_mxfp8 rejected type 41 on
     * purpose so the gap failed at load rather than dispatching into nothing,
     * which is exactly what it did when a repacked artifact was tried. Adding the
     * arm is what makes that repack legal. */
    /* MXFP8_LT ONLY.  The FP8_E4M3 (type-38) disjunct that used to sit here was
     * provably dead: gguf.cpp's loader calls pulsar_die("plain MXFP8 weight in
     * artifact") on type 38 since L060, so no such weight can reach any
     * dispatcher.  Its membership in the accept set is deliberate sequencing --
     * pass validation, then die at cache time with the actionable repack
     * message -- but the dispatch branch itself served nothing (L083 C6). */
    if (w->type == PULSAR_TENSOR_MXFP8_LT) {
        return pulsar_gpu_matmul_mxfp8_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    /* Reached only if a type is IN pulsar_weight_is_plain_or_mxfp8 but has no
     * arm above -- i.e. the two drifted. Say which, because the old message
     * ("does not support") reads like an artifact problem when it is ours. */
    fprintf(stderr, "pulsar: plain matmul has no arm for %s%s\n",
            tensor_type_name(w->type),
            pulsar_weight_is_plain_or_mxfp8(w->type)
                ? " -- but it IS in the accept set; an arm is missing here"
                : " (and it is correctly absent from the accept set)");
    return false;
}



bool gpu_graph_matmul_mxfp8_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = pulsar_gpu_matmul_mxfp8_tensor(out,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok) != 0;
    return ok;
}



/* =========================================================================
 * GPU Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * GPU tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */










/* =========================================================================
 * GPU Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing GPU backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */



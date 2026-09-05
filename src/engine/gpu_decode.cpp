#include "pulsar_engine_internal.h"


/* Read an HC residual CARRIER (BF16 storage under task #62) into an f32 host
 * buffer, expanding each stored sample (BF16->f32 is an exact bit-extension:
 * the stored 16 bits are the high half of the f32). Used ONLY by the dev-only
 * layer-0 parity self-test (since removed) and the env-gated DSpark
 * dumps — never the production decode path. n is a sample count. */
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


































/* Prefill score slice, in rows: the prefill [indexer score -> top-k -> indexed
 * attention] sequence runs in <= slice-token spans so indexer_scores (the one
 * ctx-scaling f32 work buffer with a token dimension) is allocated with slice
 * rows instead of prefill_cap.  512, one number one place (validated bit-exact;
 * the PULSAR_PREFILL_SLICE env override went with L159 inc 4). */
uint32_t gpu_graph_prefill_slice(void) {
    /* The prefill score slice, in rows.  The env override had no caller
     * (L159 inc 4); one number, one place. */
    return 512u;
}

uint64_t gpu_graph_attn_comp_cache_row_bytes(void) {
    /* One row format for every KV buffer since the L111 unification; this
     * asks the backend so the seam stays a question, not an assumption. */
    return pulsar_gpu_attn_pack_rowbytes(PULSAR_N_HEAD_DIM);
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
 * reads it via pulsar_hc_load, exactly as pulsar_gpu_rms_norm_plain_rows_tensor does.
 * L159: the unfused norm -> plain-GEMM arm for other weight formats was a
 * fallback nothing ran (hc_*_fn is bf16); it is deleted, other formats refuse. */
static bool gpu_graph_norm_mix_plain(
        const pulsar_model      *model,
        const pulsar_tensor     *w,
        uint64_t              hc_dim,
        uint64_t              out_dim,
        const pulsar_gpu_tensor *src_hc,
        pulsar_gpu_tensor       *out) {
    /* Any storage the fused kernel can read takes the fusion; only an fp8 mix
     * weight still needs the unfused pair. Was F16-only, which quietly dropped
     * the fusion -- and its ~5.4% of decode -- as soon as hc_*_fn moved. */
    if (w->type != PULSAR_TENSOR_BF16 && w->type != PULSAR_TENSOR_F32) {
        fprintf(stderr, "pulsar: hc mix weight type %d has no fused norm+mix kernel -- refusing\n",
                (int)w->type);
        return false;
    }
    return pulsar_gpu_hc_norm_mix_tensor(out, model->map, model->size,
                                          w->abs_offset, hc_dim, out_dim,
                                          src_hc, PULSAR_RMS_EPS,
                                          w->type) != 0;
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
    /* The projection is one DECODE row (the drafter's conditioning); the GEMV
     * under it takes the M-independent arm.  Restored on exit, so a caller
     * mid-step (the fused loop seeds up to 5 rows per step) keeps its count. */
    pulsar_decode_rows_scope rows(1u);
    if (!rows.ok()) return false;

    /* L158 (2026-09-03): this was the ONE dense GEMV in the served lane still
     * multiplying f32 activations.  Three tensor copies built an f32 concat,
     * nothing armed a slot, and the GEMV fell to the f32 kernel -- W8A32 on
     * the drafter's projection while every other projection ran W8A8 (found
     * the day the f32 fallback was deleted: four spec gates refused here).
     * A8 is producer-side: the concat is now emitted as E4M3 straight into
     * the slot, the f32 concat is never written, and the GEMV reads the
     * encoding.  No slot means no GEMV -- one format or an error. */
    bool ok = true;
    void *cq = NULL, *csf = NULL;
    int ckbp = 0;
    pulsar_gpu_mxfp8_act_cache_arm(target_concat, 1, concat_dim);
    if (!pulsar_gpu_mxfp8_act_cache_e4m3_slot(target_concat, 1, concat_dim, &cq, &csf, &ckbp)) {
        static int said = 0;
        if (!said) { said = 1; fprintf(stderr, "pulsar: dspark main-x concat: no E4M3 slot -- refusing\n"); }
        ok = false;
    }
    if (ok) ok = pulsar_gpu_dspark_concat3_e4m3(cq, csf, ckbp,
                                                g->dspark_target_h[0], g->dspark_target_h[1],
                                                g->dspark_target_h[2], (uint32_t)E) != 0;
    if (ok) {
        pulsar_gpu_mxfp8_act_cache_note_mxfp8();
        pulsar_gpu_mxfp8_act_cache_note_f32_skipped(1u);
        ok = pulsar_gpu_matmul_mxfp8_tensor(proj_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->main_proj->abs_offset,
                                          concat_dim, E,
                                          target_concat, 1) != 0;
    }
    pulsar_gpu_mxfp8_act_cache_disarm();

    /* main_x feeds the seed-KV GEMVs (gpu_graph_dspark_seed_draft_kv, called
     * right after this): the norm emits E4M3 into main_x's slot and leaves it
     * ARMED for the seed, which disarms when done.  The f32 row is written
     * only when the spec dump (session_spec.cpp) will read it. */
    void *mx_q = NULL, *mx_sf = NULL; int mx_kbp = 0;
    if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->dspark_main_x, 1, E, &mx_q, &mx_sf, &mx_kbp)) {
        static int said = 0;
        if (!said) { said = 1; fprintf(stderr, "pulsar: dspark main-x norm: no E4M3 slot -- refusing\n"); }
        ok = false;
    }
    if (ok) {
        ok = pulsar_gpu_rms_norm_weight_mx_tensor(gpu_graph_spec_dump_active() ? g->dspark_main_x : NULL,
                                               proj_out,
                                               dspark_model->map,
                                               dspark_model->size,
                                               w->main_norm->abs_offset,
                                               (uint32_t)E,
                                               PULSAR_RMS_EPS,
                                               mx_q, mx_sf, mx_kbp,
        NULL,
        w->main_norm->type == PULSAR_TENSOR_BF16) != 0;
    }
    if (ok) {
        pulsar_gpu_mxfp8_act_cache_arm(g->dspark_main_x, 1, E);
        pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    }

    return ok;
}

bool gpu_graph_dspark_seed_draft_kv(
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
    if (!kv_out || !kv_norm || !kv_rot) {
        fprintf(stderr, "pulsar: drafter KV seed: seed scratch not allocated -- refusing\n");
        return false;
    }
    /* One DECODE row per layer GEMV (kv projection of main_x). */
    pulsar_decode_rows_scope rows(1u);
    if (!rows.ok()) return false;

    /* ⚠ THE THREE LAYERS MUST ADVANCE IN LOCKSTEP, so a partial failure rolls the
     * whole seed back and the call REFUSES.  This loop used to `continue` past a
     * failed layer without advancing dspark_n_raw[li], leaving that layer's ring
     * counter behind its siblings' PERMANENTLY: every later draft then RoPEs at
     * a different sequence position per layer.  The verifier still rejects the
     * bad drafts, so output stays exact -- which is precisely why it was
     * invisible.  What degrades is ACCEPTANCE, silently, for the rest of the
     * session.  Until L167 the rollback was followed by a once-only WARNING and
     * the step drafted unseeded -- the same silent acceptance dip, one step
     * wide; now the caller's spec round refuses instead.
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
    /* L158: last consumer of main_x's E4M3 encoding (armed by the projection). */
    pulsar_gpu_mxfp8_act_cache_disarm();
    if (!seeded) {
        g->dspark_n_raw[0] = n_raw_entry[0];
        g->dspark_n_raw[1] = n_raw_entry[1];
        g->dspark_n_raw[2] = n_raw_entry[2];
        fprintf(stderr,
                "pulsar: drafter KV seed failed (%u row(s)); the three layers are rolled "
                "back to %u/%u/%u -- refusing the spec round rather than drafting unseeded\n",
                n_rows, n_raw_entry[0], n_raw_entry[1], n_raw_entry[2]);
        return false;
    }
    return true;
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
    /* L150: the single-bank forward is the banked one at n_banks == 1 -- the
     * scalar-position code path, byte-identical to before. */
    return gpu_graph_dspark_draft_forward_banks(g, base_model, base_weights, dspark_model, w,
                                                base_logits_out, draft_ids, n_draft,
                                                1u, NULL, NULL, NULL);
}

/* L150: the drafter forward over the rows of several banks at once. Row t
 * belongs to bank row_bank[t] (ids in [0, g->banks.n_banks)); rows of one bank
 * are contiguous and in draft order; bank b has bank_n_draft[b] rows and its
 * three drafter rings hold bank_n_raw[b][li] positions. n_banks == 1 with the
 * three arrays NULL is the classic single-bank forward on the CURRENT bank's
 * repointed ring views (scalar positions, unchanged). With n_banks > 1 the
 * per-row position arrays go up in ONE write and the attention/store calls
 * take the banked arm over the bank-major slabs: rope and KV-store positions
 * are each row's true position (n_raw_b + k); the attention VISIBILITY
 * position is the bank's last draft position (n_raw_b + n_draft_b - 1) for
 * every row of that bank, which reproduces today's non-causal window
 * [0, n_raw_b + n_draft_b) and ring start exactly (rows/L150.md). */
bool gpu_graph_dspark_draft_forward_banks(
        pulsar_gpu_graph          *g,
        const pulsar_model         *base_model,
        const pulsar_weights       *base_weights,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        pulsar_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft,
        uint32_t                n_banks,
        const uint32_t          *row_bank,
        const uint32_t         (*bank_n_raw)[3],
        const uint32_t          *bank_n_draft) {
    const bool banked = row_bank != NULL;
    if (banked && (n_banks == 0 || !bank_n_raw || !bank_n_draft || !g->dspark_row_meta ||
                   g->banks.n_banks == 0 || n_draft > PULSAR_SPEC_LOGITS_ROWS ||
                   !g->banks.dspark_raw[0] || !g->banks.dspark_raw[1] || !g->banks.dspark_raw[2]))
        return false;
    /* per-row device arrays: rope/store position per layer, visibility per
     * layer, bank id -- one upload for the whole forward */
    pulsar_gpu_tensor *meta_rope[3] = {NULL, NULL, NULL};
    pulsar_gpu_tensor *meta_vis[3] = {NULL, NULL, NULL};
    pulsar_gpu_tensor *meta_seq = NULL;
    if (banked) {
        int32_t meta[7 * PULSAR_SPEC_LOGITS_ROWS];
        uint32_t k = 0, prev_bank = UINT32_MAX;
        for (uint32_t t = 0; t < n_draft; t++) {
            const uint32_t b = row_bank[t];
            if (b >= g->banks.n_banks) return false;
            k = (b == prev_bank) ? k + 1u : 0u;
            prev_bank = b;
            for (uint32_t li = 0; li < 3; li++) {
                meta[(li * 2 + 0) * PULSAR_SPEC_LOGITS_ROWS + t] = (int32_t)(bank_n_raw[b][li] + k);
                meta[(li * 2 + 1) * PULSAR_SPEC_LOGITS_ROWS + t] =
                    (int32_t)(bank_n_raw[b][li] + bank_n_draft[b] - 1u);
            }
            meta[6 * PULSAR_SPEC_LOGITS_ROWS + t] = (int32_t)b;
        }
        if (!pulsar_gpu_tensor_write(g->dspark_row_meta, 0, meta, sizeof(meta))) return false;
        const uint64_t rb = (uint64_t)PULSAR_SPEC_LOGITS_ROWS * sizeof(int32_t);
        bool vok = true;
        for (uint32_t li = 0; li < 3; li++) {
            meta_rope[li] = pulsar_gpu_tensor_view(g->dspark_row_meta, (uint64_t)(li * 2 + 0) * rb,
                                                   (uint64_t)n_draft * sizeof(int32_t));
            meta_vis[li] = pulsar_gpu_tensor_view(g->dspark_row_meta, (uint64_t)(li * 2 + 1) * rb,
                                                  (uint64_t)n_draft * sizeof(int32_t));
            vok = vok && meta_rope[li] && meta_vis[li];
        }
        meta_seq = pulsar_gpu_tensor_view(g->dspark_row_meta, 6u * rb,
                                          (uint64_t)n_draft * sizeof(int32_t));
        if (!vok || !meta_seq) {
            for (uint32_t li = 0; li < 3; li++) {
                pulsar_gpu_tensor_free(meta_rope[li]);
                pulsar_gpu_tensor_free(meta_vis[li]);
            }
            pulsar_gpu_tensor_free(meta_seq);
            return false;
        }
    }
    struct MetaGuard {
        pulsar_gpu_tensor **r, **v, *s;
        ~MetaGuard() {
            for (int i = 0; i < 3; i++) { pulsar_gpu_tensor_free(r[i]); pulsar_gpu_tensor_free(v[i]); }
            pulsar_gpu_tensor_free(s);
        }
    } meta_guard = { meta_rope, meta_vis, meta_seq };
    /* Every row of a drafter forward -- banked or the single-bank one -- is a
     * DECODE row: declare the count so every dense GEMM, the attn-out GEMV,
     * the heads and the MoE under it take the M-independent arms whatever the
     * batch width (until L167 the single-bank forward declared nothing and
     * its 5..16-row GEMMs took cuBLASLt by row count).  The setter refuses a
     * forward wider than the cap; the caller groups banks to fit it. */
    pulsar_decode_rows_scope rows(n_draft);
    if (!rows.ok()) return false;
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

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint32_t n_groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;

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
        /* RMS norm: flat HC from batch_cur_hc.
         *
         * The consumer below (hc_attn_fn, a BF16-weight GEMM through the shared
         * bf16 core) takes the producer-emitted BF16 copy when the slot for
         * (batch_flat_hc, n_draft, hc_dim) is noted valid -- the same slot the
         * drafter's output head notes at the SAME key one pass earlier. So this
         * producer must acquire the slot (which resets that note) and emit its
         * own copy, exactly as the batch layer encoder does; a producer that
         * skips the slot would leave the head's stale note standing and the
         * GEMM would read last pass's rows. (As cherry-picked, de459a8 also
         * passed NULL as the OUTPUT here and the flat buffer as the bf16 slot:
         * the wrapper refused, every drafter forward failed, and the drafter
         * was silently dead on dev from a5208ca until this fix -- no gate in
         * that landing batch looked at drafts. rows/L150.md.) */
        void *flat_b = NULL;
        if (ok && !pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_draft, hc_dim, &flat_b)) {
            fprintf(stderr, "pulsar: drafter flat_hc: no bf16 slot -- refusing (L159)\n");
            ok = false;
        }
        if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(
            g->batch_flat_hc, flat_b, g->batch_cur_hc,
            (uint32_t)hc_dim, n_draft, PULSAR_RMS_EPS, 0) != 0;
        if (ok && flat_b) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_draft, hc_dim);
        /* HC → mix projection */
        if (ok) ok = gpu_graph_matmul_plain_tensor(
            hc_mix_view, dspark_model,
            layer->hc_attn_fn,
            hc_dim, mix_hc, g->batch_flat_hc, n_draft);
        /* HC split + weighted sum + input RMS norm -> batch_attn_norm, with the
         * E4M3 encoding emitted at the producer (L158, 2026-09-03).  Until today
         * this was a split-sum kernel plus a plain f32 norm, and the two GEMVs
         * below read f32 -- the drafter's attention ran W8A32 while the target's
         * ran W8A8 (the f32 GEMV arms were deleted in L158 and refused here).
         * Same fused producer the target's batch layer uses (gpu_prefill.cpp);
         * the pre-norm carrier still lands in ffn_cur_view. */
        void *dn_q = NULL, *dn_sf = NULL; int dn_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_attn_norm, n_draft, PULSAR_N_EMBD,
                                                        &dn_q, &dn_sf, &dn_kbp)) {
            fprintf(stderr, "pulsar: drafter attn_norm: no E4M3 slot -- refusing\n");
            ok = false;
        }
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
            ffn_cur_view, g->batch_attn_norm, dn_q, dn_sf, dn_kbp,
            NULL /* no bf16 consumer in the drafter */,
            gpu_graph_f32_store_observed_any() ? 0u : n_draft /* f32 rows only for a dump */,
            hc_split_view, hc_mix_view, g->batch_cur_hc,
            dspark_model->map, dspark_model->size,
            layer->hc_attn_scale->abs_offset,
            layer->hc_attn_base->abs_offset,
            layer->attn_norm->abs_offset,
            n_draft, PULSAR_N_EMBD, PULSAR_N_HC,
            PULSAR_N_HC_SINKHORN_ITER, PULSAR_HC_EPS, PULSAR_RMS_EPS,
            layer->attn_norm->type == PULSAR_TENSOR_BF16) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_norm", g->batch_attn_norm,
                                             (uint64_t)n_draft * PULSAR_N_EMBD, li, pos0);
        /* batch_attn_norm feeds TWO MXFP8 GEMMs (attn_q_a, attn_kv).  The fused
         * norm emitted its E4M3 into the armed slot once; both GEMMs read it. */
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_attn_norm, n_draft,
                                               PULSAR_N_EMBD);
        if (ok) pulsar_gpu_mxfp8_act_cache_note_mxfp8();   /* L158: the fused norm emitted it */
        /* --- Q projection --- */
        if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(
            g->batch_qr, dspark_model->map, dspark_model->size,
            layer->attn_q_a->abs_offset,
            PULSAR_N_EMBD, q_rank, g->batch_attn_norm, n_draft) != 0;
        /* L158: q_a_norm emits E4M3 for the q_b GEMV (was f32 -> W8A32). */
        void *dq_q = NULL, *dq_sf = NULL; int dq_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->batch_qr_norm, n_draft, q_rank,
                                                        &dq_q, &dq_sf, &dq_kbp)) {
            fprintf(stderr, "pulsar: drafter q_a_norm: no E4M3 slot -- refusing\n");
            ok = false;
        }
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_mx_tensor(
            NULL /* no f32 reader: attn_q_b reads the E4M3 slot keyed on batch_qr_norm */, g->batch_qr,
            dspark_model->map, dspark_model->size,
            layer->attn_q_a_norm->abs_offset,
            q_rank, n_draft, PULSAR_RMS_EPS,
            dq_q, dq_sf, dq_kbp,
            NULL,
            layer->attn_q_a_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_qr_norm, n_draft, q_rank);
        if (ok) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
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
            banked ? meta_rope[li] : NULL) != 0;

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
        NULL,
        layer->attn_kv_a_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) ok = pulsar_gpu_rope_tail_tensor(
            g->batch_kv, n_draft,
            PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT,
            pos0, 0, false,
            (float)PULSAR_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
            banked ? meta_rope[li] : NULL) != 0;
        /* No fake-quantise pass here.  batch_kv has no reader after the store
         * below, and that store packs the rows itself -- so round-tripping first
         * only fed already-quantized values to a fresh quantizer, which is the
         * ~5%-misround pattern norm_kv warns about.  Packing the true f32 once
         * is both cheaper and closer to what the target path does for the same
         * token, which is what verification compares against. */

        /* --- Store draft KV transiently in ring buffer for attention --- */
        const uint32_t saved_n_raw = g->dspark_n_raw[li];
        const uint32_t kv_store_pos = saved_n_raw % raw_cap;
        if (ok) {
            if (banked)
                /* row t -> bank row_bank[t]'s ring at slot rope_pos[t] % raw_cap */
                ok = pulsar_gpu_store_raw_kv_batch_tensor(
                    g->banks.dspark_raw[li], g->batch_kv,
                    raw_cap, 0u, n_draft, PULSAR_N_HEAD_DIM,
                    meta_rope[li], meta_seq, g->banks.n_banks) != 0;
            else
                ok = pulsar_gpu_store_raw_kv_batch_tensor(
                    g->dspark_raw_cache[li], g->batch_kv,
                    raw_cap, kv_store_pos, n_draft, PULSAR_N_HEAD_DIM,
                    NULL, NULL, 1) != 0;
        }
        const uint32_t vis_raw = saved_n_raw + n_draft;
        const uint32_t cap_raw = vis_raw < raw_cap ? vis_raw : raw_cap;
        const uint32_t raw_start = vis_raw > raw_cap
            ? (vis_raw - raw_cap) % raw_cap : 0;

        /* --- Non-causal raw batch attention ---
         * Queries are at positions [saved_n_raw, saved_n_raw+n_draft).
         * Visible raw entries span [0, vis_raw) — all cached + current draft rows.
         * Banked: per row, the kernel derives the window from the VISIBILITY
         * position (the bank's last draft position) with window = raw_cap, which
         * yields exactly cap_raw / raw_start / first_raw_pos of the scalar path
         * for that bank; scalar n_raw/raw_start are ignored there. */
        if (ok) {
            if (banked)
                ok = pulsar_gpu_attention_decode_raw_batch_heads_tensor(
                    g->batch_heads,
                    dspark_model->map, dspark_model->size,
                    layer->attn_sinks->abs_offset,
                    g->batch_q, g->banks.dspark_raw[li],
                    n_draft, 0u,
                    0u, raw_cap, 0u,
                    raw_cap,
                    PULSAR_N_HEAD, PULSAR_N_HEAD_DIM,
                    1,
                    meta_vis[li], meta_seq, 0, g->banks.n_banks,
                    NULL /* q pre-normed */) != 0;
            else
                ok = pulsar_gpu_attention_decode_raw_batch_heads_tensor(
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
        }

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
            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
            banked ? meta_rope[li] : NULL) != 0;
        /* L158 inc 4: the drafter's raw attention has no E4M3 epilogue, so the
         * attention stage emits the grouped encoding here, after the inverse
         * rope, and the attn-out 'a' projection reads it -- the consumer's
         * quantise-from-heads fallback is gone. */
        if (ok) ok = pulsar_gpu_mxfp8_gact_emit_heads(g->batch_heads, n_draft, n_groups, group_dim) != 0;
        /* --- Attention output projection (LoRA grouped) --- */
        if (ok) ok = pulsar_gpu_attention_output_batch_tensor(
            g->batch_attn_out, g->batch_attn_low,
            dspark_model->map, dspark_model->size,
            layer->attn_output_a->abs_offset,
            layer->attn_output_b->abs_offset,
            group_dim, rank, n_groups, PULSAR_N_EMBD,
            g->batch_heads, n_draft) != 0;
        pulsar_gpu_mxfp8_gact_disarm();
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

/* Encode the final HC collapse, output norm, and vocab projection on GPU for
 * ONE row (g->cur_hc): the last row of a whole/chunked prefill.  Its logits are
 * the first decode output -- the same distribution the classic decode step
 * produces with the one-row GEMV -- so the row is declared a DECODE row and
 * the vocab GEMM takes the M-independent arm (as it did before L167, by row
 * count).  The callers are the prefill lane, which declares nothing itself. */
bool gpu_graph_encode_output_head(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    pulsar_decode_rows_scope rows(1u);
    if (!rows.ok()) return false;
    bool ok = gpu_graph_norm_mix_plain((const pulsar_model *)model, weights->output_hc_fn,
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
    /* L158: an MXFP8 head (the drafter's) reads its activation from the E4M3
     * slot, so the output norm emits it; a bf16 head (the target's) takes the
     * bf16 core, which converts elementwise -- same bytes whichever side does
     * it, so no slot is needed there. */
    const bool head_mx = weights->output->type != PULSAR_TENSOR_BF16;
    void *hn_q = NULL, *hn_sf = NULL; int hn_kbp = 0;
    if (ok && head_mx && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->output_norm, 1, PULSAR_N_EMBD,
                                                              &hn_q, &hn_sf, &hn_kbp)) {
        fprintf(stderr, "pulsar: output norm: no E4M3 slot for the MXFP8 head -- refusing\n");
        ok = false;
    }
    /* L159: a bf16 head reads the producer's bf16 plane; the GEMM core no
     * longer converts f32 on a miss.  Same slot as the E4M3 half: one key, two
     * planes, the head fills the one its weight format needs. */
    void *hn_b = NULL;
    if (ok && !head_mx && !pulsar_gpu_bf16_act_slot(g->output_norm, 1, PULSAR_N_EMBD, &hn_b)) {
        fprintf(stderr, "pulsar: output norm: no bf16 slot for the bf16 head -- refusing\n");
        ok = false;
    }
    if (ok) ok = pulsar_gpu_rms_norm_weight_mx_tensor(g->output_norm,
                                                     g->output_embd,
                                                     model->map,
                                                     model->size,
                                                     weights->output_norm->abs_offset,
                                                     PULSAR_N_EMBD,
                                                     PULSAR_RMS_EPS,
                                                     hn_q, hn_sf, hn_kbp,
        hn_b,
        weights->output_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_norm", g->output_norm, PULSAR_N_EMBD, PULSAR_N_LAYER, 0);
    }
    if (ok && hn_b) pulsar_gpu_bf16_act_note(g->output_norm, 1, PULSAR_N_EMBD);
    if (ok) {
        if (!head_mx) {
            ok = pulsar_gpu_matmul_bf16_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
        } else {
            pulsar_gpu_mxfp8_act_cache_arm(g->output_norm, 1, PULSAR_N_EMBD);
            pulsar_gpu_mxfp8_act_cache_note_mxfp8();
            ok = pulsar_gpu_matmul_mxfp8_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, PULSAR_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
            pulsar_gpu_mxfp8_act_cache_disarm();
        }
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

    /* The output head's hc projection is a BF16-weight GEMM (output_hc_fn via
     * pulsar_gpu_matmul_f32_tensor -> the shared bf16 core), so emit the bf16
     * copy from the norm epilogue rather than letting the GEMM convert
     * hc_dim floats every step.  L086 T3. */
    void *out_flat_b = NULL;
    if (ok && !pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_tokens,
                                        (uint64_t)hc_dim, &out_flat_b)) {
        fprintf(stderr, "pulsar: verify head flat_hc: no bf16 slot -- refusing (L159)\n");
        ok = false;
    }
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, out_flat_b,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS, 0) != 0;
    if (ok && out_flat_b) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_tokens,
                                                   (uint64_t)hc_dim);
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
    void *on_b = NULL;
    if (ok && weights->output->type == PULSAR_TENSOR_BF16 &&
        !pulsar_gpu_bf16_act_slot(output_norm, n_tokens, PULSAR_N_EMBD, &on_b)) {
        fprintf(stderr, "pulsar: verify output norm: no bf16 slot for the bf16 head -- refusing\n");
        ok = false;
    }
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       PULSAR_N_EMBD,
                                                       n_tokens,
                                                       PULSAR_RMS_EPS,
        on_b,
        weights->output_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok && on_b) pulsar_gpu_bf16_act_note(output_norm, n_tokens, PULSAR_N_EMBD);
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

    pulsar_gpu_act_slot_drop(output_norm);   /* L159: planes die with the buffer */
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
    /* Same as the main output head above: dw->hc_head_fn is a BF16-weight
     * GEMM, so the epilogue emits the copy it needs. */
    void *dsp_flat_b = NULL;
    if (ok && !pulsar_gpu_bf16_act_slot(g->batch_flat_hc, n_tokens,
                                        (uint64_t)hc_dim, &dsp_flat_b)) {
        fprintf(stderr, "pulsar: dspark head flat_hc: no bf16 slot -- refusing (L159)\n");
        ok = false;
    }
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, dsp_flat_b, g->batch_cur_hc,
                                                     (uint32_t)hc_dim, n_tokens, PULSAR_RMS_EPS, 0) != 0;
    if (ok && dsp_flat_b) pulsar_gpu_bf16_act_note(g->batch_flat_hc, n_tokens,
                                                   (uint64_t)hc_dim);
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre, dspark_model, dw->hc_head_fn,
                                               hc_dim, PULSAR_N_HC, g->batch_flat_hc, n_tokens) != 0;
    if (ok) ok = pulsar_gpu_output_hc_weights_tensor(output_weights, output_pre,
                                                  dspark_model->map, dspark_model->size,
                                                  dw->hc_head_scale->abs_offset,
                                                  dw->hc_head_base->abs_offset,
                                                  PULSAR_N_HC, PULSAR_HC_EPS) != 0;
    if (ok) ok = pulsar_gpu_hc_weighted_sum_tensor(output_embd, g->batch_cur_hc, output_weights,
                                                PULSAR_N_EMBD, PULSAR_N_HC) != 0;
    void *dn_b = NULL;
    if (ok && bw->output->type == PULSAR_TENSOR_BF16 &&
        !pulsar_gpu_bf16_act_slot(output_norm, n_tokens, PULSAR_N_EMBD, &dn_b)) {
        fprintf(stderr, "pulsar: dspark output norm: no bf16 slot for the bf16 head -- refusing\n");
        ok = false;
    }
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(output_norm, output_embd,
                                                     dspark_model->map, dspark_model->size,
                                                     dw->final_norm->abs_offset,
                                                     PULSAR_N_EMBD, n_tokens, PULSAR_RMS_EPS,
        dn_b,
        dw->final_norm->type == PULSAR_TENSOR_BF16) != 0;
    if (ok && dn_b) pulsar_gpu_bf16_act_note(output_norm, n_tokens, PULSAR_N_EMBD);
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
    pulsar_gpu_act_slot_drop(output_norm);   /* L159: planes die with the buffer */
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



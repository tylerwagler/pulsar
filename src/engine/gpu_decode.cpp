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
                                             PULSAR_N_ROT) != 0;
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
    return PULSAR_ENGINE_ATTN_PACK_ROWBYTES;
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
         * packed comp cache at first_row.  The kernel also fp8-roundtrips the
         * stage rows in place (identical to the plain quantize the f32 path
         * runs), so the stage keeps the exact f32-pipeline values.  This MUST
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
                                                    PULSAR_N_HEAD_DIM, PULSAR_N_ROT) == 0) {
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
            PULSAR_N_HEAD_DIM, PULSAR_N_ROT) != 0;
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


bool gpu_graph_encode_decode_layer(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        pulsar_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint32_t n_groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = g->comp_ratio_override >= 0
        ? g->comp_ratio_override != 0
        : pulsar_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    bool ok = true;
    const bool decode_stage_profile = gpu_graph_decode_stage_profile_enabled(il);
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define PULSAR_CUDA_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = gpu_graph_norm_mix_plain(g, model, layer->hc_attn_fn,
                                          hc_dim, mix_hc, g->cur_hc, g->hc_mix);
    if (ok) {
        /* A8 on the decode path: emit the E4M3 + ue8m0 encoding from the norm
         * epilogue so the mmvq GEMVs multiply in the format the SOURCE uses
         * (dynamic e4m3) instead of against f32.  Decode was the last place
         * W8A32 survived.  FIDELITY, not speed -- a GEMV's traffic is dominated
         * by the weight matrix, not the shared activation vector. */
        void *an_q = NULL, *an_sf = NULL; int an_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->attn_norm, 1, PULSAR_N_EMBD,
                                                        &an_q, &an_sf, &an_kbp)) {
            an_q = NULL; an_sf = NULL; an_kbp = 0;
        }
        ok = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         an_q, an_sf, an_kbp,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         1u,
                                                         PULSAR_N_EMBD,
                                                         PULSAR_N_HC,
                                                         PULSAR_N_HC_SINKHORN_ITER,
                                                         PULSAR_HC_EPS,
                                                         PULSAR_RMS_EPS,
        layer->attn_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->attn_norm, 1, PULSAR_N_EMBD);
        if (ok && an_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, PULSAR_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, PULSAR_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)PULSAR_N_HC * PULSAR_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, PULSAR_N_EMBD, il, pos);
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->attn_norm, PULSAR_N_EMBD, il, pos);
    }
    bool qkv_pair_projected = false;
    if (ok) {
        qkv_pair_projected = pulsar_gpu_matmul_mxfp8_pair_tensor(g->qr,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a->abs_offset,
                                                             layer->attn_kv->abs_offset,
                                                             PULSAR_N_EMBD,
                                                             q_rank,
                                                             PULSAR_N_HEAD_DIM,
                                                             g->attn_norm,
                                                             1) != 0;
    }
    if (ok && !qkv_pair_projected) ok = pulsar_gpu_matmul_mxfp8_tensor(g->qr,
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_q_a->abs_offset,
                                                                    PULSAR_N_EMBD,
                                                                    q_rank,
                                                                    g->attn_norm,
                                                                    1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    /* qr_norm feeds the MXFP8 attn_q_b GEMV; emit its E4M3 too so no decode
     * GEMV is left multiplying against f32. */
    void *qn_q = NULL, *qn_sf = NULL; int qn_kbp = 0;
    {
        if (ok && !qkv_pair_projected) ok = pulsar_gpu_matmul_mxfp8_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  PULSAR_N_EMBD, PULSAR_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, PULSAR_N_HEAD_DIM, il, pos);
        }
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->qr_norm, 1, (uint64_t)q_rank,
                                                        &qn_q, &qn_sf, &qn_kbp)) {
            qn_q = NULL; qn_sf = NULL; qn_kbp = 0;
        }
        if (ok) ok = pulsar_gpu_dsv4_qkv_rms_norm_rows_mx_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             PULSAR_N_HEAD_DIM,
                                                             1,
                                                             PULSAR_RMS_EPS,
                                                             qn_q, qn_sf, qn_kbp,
        layer->attn_q_a_norm->type == PULSAR_TENSOR_BF16, layer->attn_kv_a_norm->type == PULSAR_TENSOR_BF16,
                                                             /* q_skip_f32: */ 0) != 0;
    }
    if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->qr_norm, 1, (uint64_t)q_rank);
    if (ok && qn_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->kv, PULSAR_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    bool decode_q_norm_rope_fused = false;
    if (ok) {
        decode_q_norm_rope_fused =
            pulsar_gpu_head_rms_norm_rope_tail_tensor(g->q,
                                                   1,
                                                   PULSAR_N_HEAD,
                                                   PULSAR_N_HEAD_DIM,
                                                   PULSAR_N_ROT,
                                                   pos,
                                                   compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                   false,
                                                   freq_base,
                                                   freq_scale,
                                                   ext_factor,
                                                   attn_factor,
                                                   PULSAR_ROPE_YARN_BETA_FAST,
                                                   PULSAR_ROPE_YARN_BETA_SLOW,
                                                   PULSAR_RMS_EPS,
                                                   NULL,
                                                   /* q_f16: */ 0) != 0;
    }
    /* The separate head-norm + rope-tail pair here was reachable ONLY via a
     * "Qnorm" dump request, and the fused kernel never materialises that
     * intermediate -- so the dump forced different kernels and, by this file's
     * own former warning, produced numbers production does not compute.
     * Removed with the dump: a debug affordance that changes what it observes
     * cannot diagnose what it observes.  L045 stage 2. */
    if (!decode_q_norm_rope_fused && ok) {
        fprintf(stderr, "pulsar: decode Q norm+rope did not run -- refusing rather than "
                        "leaving Q unnormalised\n");
        ok = false;
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (ok) ok = pulsar_gpu_rope_tail_tensor(g->kv, 1, PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT, pos,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                                            NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->kv, PULSAR_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = gpu_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    PULSAR_CUDA_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->kv, PULSAR_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    pulsar_gpu_tensor *comp_cache = NULL;
    pulsar_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    static int decode_index_stage_env = -1;
    const bool decode_index_stage_profile = gpu_graph_env_flag("PULSAR_CUDA_INDEXER_STAGE_PROFILE", &decode_index_stage_env);
    if (ok && compressed) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * PULSAR_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_kv) ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_gate) ||
            layer->attn_compressor_kv->dim[0] != PULSAR_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != PULSAR_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "pulsar: GPU graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok) {
            /* Two sequential calls, not a fusion: the F16 pair fast path is
             * gone with the last F16 weight, and the f16 and bf16 kernels were
             * structurally identical, so collapsing to one arm cost nothing
             * measurable.  Its `else` -- an identical copy of these same two
             * calls, guarded by `if (ok)` inside an `else` of `if (ok)` --
             * outlived it as unreachable code until 2026-08-18. */
            ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                layer->attn_compressor_kv,
                                                PULSAR_N_EMBD, comp_width,
                                                g->attn_norm, 1) &&
                 gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                layer->attn_compressor_gate,
                                                PULSAR_N_EMBD, comp_width,
                                                g->attn_norm, 1);
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = pulsar_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        gpu_graph_attn_comp_update_target(g, il),
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
        if (ok && emit) {
            pulsar_gpu_tensor *comp_row_view = gpu_graph_attn_comp_row_view(g, il, comp_row);
            /* comp_row_view aliases the f32 stage; the commit below quantizes,
             * packs and roundtrips the stage in place, so the KVcompress dump
             * happens after it. */
            if (!comp_row_view) {
                ok = false;
            }
            if (ok) ok = gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
            if (ok && comp_row_view) {
                gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, PULSAR_N_HEAD_DIM, il, pos);
            }
            pulsar_gpu_tensor_free(comp_row_view);
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * PULSAR_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_kv) ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_gate) ||
                layer->indexer_compressor_kv->dim[0] != PULSAR_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != PULSAR_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "pulsar: GPU graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok) {
                ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                    layer->indexer_compressor_kv,
                                                    PULSAR_N_EMBD, index_width,
                                                    g->attn_norm, 1) &&
                     gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                    layer->indexer_compressor_gate,
                                                    PULSAR_N_EMBD, index_width,
                                                    g->attn_norm, 1);
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = pulsar_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
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
            if (ok && emit) {
                pulsar_gpu_tensor *index_row_view = pulsar_gpu_tensor_view(
                        g->idx_comp_stage,
                        (uint64_t)index_row * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_row_view,
                                                               g->layer_index_comp_cache[il],
                                                               index_row,
                                                               1,
                                                               PULSAR_N_INDEXER_HEAD_DIM) != 0;
                    pulsar_gpu_tensor_free(index_row_view);
                }
                /* plan-33 inc C: boundary-row restore (decode-fallback site). */
                if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                        g->banks.n_banks ? g->banks.cur_bank : 0u, index_row, 1, true);
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                gpu_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > PULSAR_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_attn_q_b) ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "pulsar: GPU graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_proj) ||
                           layer->indexer_proj->dim[0] != PULSAR_N_EMBD ||
                           layer->indexer_proj->dim[1] != PULSAR_N_INDEXER_HEAD)) {
                    fprintf(stderr, "pulsar: GPU graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_q,
                                                              model,
                                                              layer->indexer_attn_q_b,
                                                              q_rank,
                                                              indexer_q_dim,
                                                              g->qr_norm,
                                                              1);
                /* Fused rope + QAT, mirroring the prefill site. */
                if (ok) ok = pulsar_gpu_dsv4_indexer_rope_qat_tensor(g->indexer_q, 1,
                                                        PULSAR_N_INDEXER_HEAD,
                                                        PULSAR_N_INDEXER_HEAD_DIM,
                                                        PULSAR_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        PULSAR_ROPE_YARN_BETA_FAST,
                                                        PULSAR_ROPE_YARN_BETA_SLOW,
                                                        NULL) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_weights, model,
                                                             layer->indexer_proj,
                                                             PULSAR_N_EMBD, PULSAR_N_INDEXER_HEAD,
                                                             g->attn_norm, 1);
                const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) {
                    ok = pulsar_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                PULSAR_N_INDEXER_HEAD,
                                                                PULSAR_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                }
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           PULSAR_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * PULSAR_N_INDEXER_TOP_K.
                     */
                    n_selected = PULSAR_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? PULSAR_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    pulsar_layer_compress_ratio(il),
                    PULSAR_N_HEAD,
                    PULSAR_N_HEAD_DIM,
                    NULL,
                    NULL,
                    NULL,
                    0,
                    1,
                                          NULL /* q pre-normed */) != 0;
            if (ok && decode_index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = pulsar_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         n_comp,
                                                         PULSAR_N_HEAD, PULSAR_N_HEAD_DIM) != 0;
        }
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = pulsar_gpu_rope_tail_tensor(g->heads,
                                            1, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT, pos,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc = !gpu_graph_directional_steering_attn_enabled(g);
    if (ok && fuse_attn_out_hc) {
        ok = pulsar_gpu_attention_output_low_tensor(g->attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   g->heads) != 0;
        if (ok) {
            ok = pulsar_gpu_matmul_fp8_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        PULSAR_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        PULSAR_N_EMBD,
                                                        PULSAR_N_HC) != 0;
        }
    } else if (ok) {
        ok = pulsar_gpu_attention_output_batch_tensor(g->attn_out,
                                                     g->attn_low,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_output_a->abs_offset,
                                                     layer->attn_output_b->abs_offset,
                                                     group_dim, rank,
                                                     n_groups, PULSAR_N_EMBD,
                                                     g->heads, 1) != 0;
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->attn_out, PULSAR_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = pulsar_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, PULSAR_N_EMBD, PULSAR_N_HC) != 0;
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = gpu_graph_norm_mix_plain(g, model, layer->hc_ffn_fn,
                                          hc_dim, mix_hc, g->after_attn_hc, g->hc_mix);
    if (ok) {
        /* Same A8 emission as the attention norm above: batch_ffn_norm feeds
         * the router logits and the shared gate/up GEMVs. */
        void *fn_q = NULL, *fn_sf = NULL; int fn_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->ffn_norm, 1, PULSAR_N_EMBD,
                                                        &fn_q, &fn_sf, &fn_kbp)) {
            fn_q = NULL; fn_sf = NULL; fn_kbp = 0;
        }
        ok = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         fn_q, fn_sf, fn_kbp,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         1u,
                                                         PULSAR_N_EMBD,
                                                         PULSAR_N_HC,
                                                         PULSAR_N_HC_SINKHORN_ITER,
                                                         PULSAR_HC_EPS,
                                                         PULSAR_RMS_EPS,
        layer->ffn_norm->type == PULSAR_TENSOR_BF16) != 0;
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->ffn_norm, 1, PULSAR_N_EMBD);
        if (ok && fn_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, PULSAR_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, PULSAR_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)PULSAR_N_HC * PULSAR_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, PULSAR_N_EMBD, il, pos);
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, PULSAR_N_EMBD, il, pos);
    }
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (ok) {
        ok = routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                            &gate_expert_bytes, &gate_row_bytes,
                                            &down_expert_bytes, &down_row_bytes);
    }
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 PULSAR_N_EMBD, PULSAR_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = pulsar_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                PULSAR_N_EXPERT,
                                                PULSAR_N_EXPERT_USED,
                                                PULSAR_EXPERT_WEIGHT_SCALE,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    PULSAR_CUDA_PROFILE_DECODE_STAGE("router");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, PULSAR_N_EXPERT, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, PULSAR_N_EXPERT, il, pos);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, PULSAR_N_EXPERT_USED, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, PULSAR_N_EXPERT_USED, il, pos);
    }
    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_down_hc = !keep_ffn_out;
    if (ok) ok = pulsar_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 tensor_map_base(model, layer->ffn_gate_exps),
                                                 tensor_map_size(model, layer->ffn_gate_exps),
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 pulsar_layer_n_expert(il),
                                                 PULSAR_N_EXPERT_USED, PULSAR_SWIGLU_CLAMP_EXP, g->ffn_norm,
                                                 il) != 0;
    PULSAR_CUDA_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)PULSAR_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)PULSAR_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)PULSAR_N_EXPERT_USED * PULSAR_N_EMBD, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, PULSAR_N_EMBD, il, pos);
    }
    if (ok) {
        /* A8 for shared_down: emit shared_mid's E4M3 from the SwiGLU epilogue
         * so the down GEMV multiplies in the source's format instead of f32.
         * Prefill has done this since gpu_prefill.cpp:2300 -- decode was simply
         * still calling the plain SwiGLU, which is why shared_down was the last
         * dense decode GEMV left on W8A32. A miss is not an error: the slot
         * comes back NULL and the epilogue behaves as before. */
        void *sm_q = NULL, *sm_sf = NULL; int sm_kbp = 0;
        if (ok && !pulsar_gpu_mxfp8_act_cache_e4m3_slot(g->shared_mid, 1, (uint64_t)shared_dim,
                                                        &sm_q, &sm_sf, &sm_kbp)) {
            sm_q = NULL; sm_sf = NULL; sm_kbp = 0;
        }
        ok = pulsar_gpu_shared_gate_up_swiglu_mxfp8_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         PULSAR_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm,
                                                         PULSAR_SWIGLU_CLAMP_EXP,
                                                         sm_q, sm_sf, sm_kbp) != 0;
        if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->shared_mid, 1, (uint64_t)shared_dim);
        if (ok && sm_q) pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && fuse_shared_down_hc) {
        ok = pulsar_gpu_shared_down_hc_expand_mxfp8_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         PULSAR_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         PULSAR_N_EMBD,
                                                         PULSAR_N_HC) != 0;
    } else if (ok) {
        ok = pulsar_gpu_matmul_mxfp8_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, PULSAR_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, PULSAR_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_ffn_out(g) &&
             pulsar_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, PULSAR_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, PULSAR_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = pulsar_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        PULSAR_N_EMBD,
                                        PULSAR_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = pulsar_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_N_HC) != 0;
    }
    PULSAR_CUDA_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef PULSAR_CUDA_PROFILE_DECODE_STAGE
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    if (ok) gpu_graph_capture_dspark_target_hc(g, il);
    return ok;
}

void gpu_graph_capture_dspark_target_hc(pulsar_gpu_graph *g, uint32_t il) {
    int slot = -1;
    for (int i = 0; i < 3; i++) {
        if (il == g->dspark_target_layer_ids[i]) { slot = i; break; }
    }
    if (slot < 0 || !g->dspark_target_h[slot]) return;

    pulsar_gpu_dspark_hc_mean_reduce(g->dspark_target_h[slot],
                                   g->after_ffn_hc,
                                   PULSAR_N_EMBD, PULSAR_N_HC);
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
    if (!g || !base_model || !base_weights || !dspark_model || !w ||
        !base_logits_out || n_draft == 0 || n_draft > 16 ||
        n_draft > g->prefill_cap)
        return false;

    if (!base_weights->token_embd || !base_weights->output)
        return false;

    /* Embed N draft tokens via main model's F16 token_embd → HC-expand */
    pulsar_gpu_tensor *tokens_t = pulsar_gpu_tensor_alloc((uint64_t)n_draft * sizeof(int32_t));
    if (!tokens_t) return false;
    if (!pulsar_gpu_tensor_write(tokens_t, 0, draft_ids, (uint64_t)n_draft * sizeof(int32_t))) {
        pulsar_gpu_tensor_free(tokens_t);
        return false;
    }
    bool ok = pulsar_gpu_embed_tokens_hc_tensor(g->batch_cur_hc,
                                              tokens_t,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->token_embd->abs_offset,
                                              PULSAR_N_VOCAB,
                                              n_draft,
                                              PULSAR_N_EMBD,
                                              PULSAR_N_HC) != 0;
    pulsar_gpu_tensor_free(tokens_t);
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
        pulsar_gpu_tensor *attn_cur_view = pulsar_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_draft * PULSAR_N_EMBD * sizeof(float));
        ok = hc_mix_view && hc_split_view && attn_cur_view;
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
            attn_cur_view, hc_split_view, hc_mix_view,
            g->batch_cur_hc,
            dspark_model->map, dspark_model->size,
            layer->hc_attn_scale->abs_offset,
            layer->hc_attn_base->abs_offset,
            PULSAR_N_EMBD, PULSAR_N_HC,
            PULSAR_N_HC_SINKHORN_ITER, PULSAR_HC_EPS) != 0;
        /* Input RMS norm → batch_attn_norm */
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, attn_cur_view,
            dspark_model->map, dspark_model->size,
            layer->attn_norm->abs_offset,
            PULSAR_N_EMBD, n_draft, PULSAR_RMS_EPS,
        layer->attn_norm->type == PULSAR_TENSOR_BF16) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_norm", g->batch_attn_norm,
                                             (uint64_t)n_draft * PULSAR_N_EMBD, li, pos0);
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
            NULL,
                                                   /* q_f16: */ 0) != 0;

        /* --- KV projection --- */
        if (ok) ok = pulsar_gpu_matmul_mxfp8_tensor(
            g->batch_kv_raw, dspark_model->map, dspark_model->size,
            layer->attn_kv->abs_offset,
            PULSAR_N_EMBD, PULSAR_N_HEAD_DIM,
            g->batch_attn_norm, n_draft) != 0;
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
        pulsar_gpu_tensor_free(attn_cur_view);
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
bool gpu_graph_encode_output_head_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

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
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
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
    if (w->type == PULSAR_TENSOR_FP8_E4M3 || w->type == PULSAR_TENSOR_MXFP8_LT) {
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

uint32_t gpu_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("PULSAR_CUDA_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= PULSAR_N_LAYER) split_after_layers = (uint32_t)v;
    }
    return split_after_layers;
}


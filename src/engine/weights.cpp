#include "pulsar_engine_internal.h"



static float required_f32(const pulsar_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "pulsar: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}



static bool required_bool(const pulsar_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "pulsar: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}



static pulsar_tensor *required_tensor(const pulsar_model *m, const char *name) {
    pulsar_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "pulsar: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}



static pulsar_tensor *required_tensorf(const pulsar_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) pulsar_die("tensor name is too long");
    return required_tensor(m, name);
}



/* Formatted OPTIONAL lookup: NULL when the artifact does not carry it.  Used for
 * the tensors a shipped checkpoint may legitimately omit -- the router's
 * correction bias is the one that bit (see the binding site).  The caller is
 * responsible for the "absent" arm; a silent NULL reaching a deref is the
 * failure this naming is meant to make visible. */
static pulsar_tensor *optional_tensorf(const pulsar_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) pulsar_die("tensor name is too long");
    return model_find_tensor(m, name);
}



/* Shape half of the layout validators. Both callers check the TYPE their own
 * way and then need exactly this; keeping one copy stops the two drifting, which
 * is a live failure mode in this file (two banked emit blocks had silently
 * stopped emitting dumps their classic twins still emitted). */
static void tensor_expect_dims(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t->ndim != ndim) {
        fprintf(stderr,
                "pulsar: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "pulsar: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}


static void tensor_expect_layout(
        const pulsar_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    tensor_expect_dims(t, ndim, d0, d1, d2);
}



/* MXFP8 workhorse weight: either the classic interleaved type (FP8_E4M3, 38) or
 * its pre-stored device layout (MXFP8_LT, 41). Both share dims and byte
 * accounting; the FP8 matmul resolver dispatches on the registered offset. */
static void tensor_expect_mxfp8(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type == PULSAR_TENSOR_FP8_E4M3)
        tensor_expect_layout(t, PULSAR_TENSOR_FP8_E4M3, ndim, d0, d1, d2);
    else if (t->type == PULSAR_TENSOR_MXFP8_LT)
        tensor_expect_layout(t, PULSAR_TENSOR_MXFP8_LT, ndim, d0, d1, d2);
    else
        pulsar_die("tensor has unsupported weight type; expected FP8_E4M3 or MXFP8_LT");
}
static void tensor_expect_plain_or_mxfp8(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    /* Membership comes from pulsar_weight_is_plain_or_mxfp8 -- see the note on
     * it. This function used to restate the set, and drifted from the
     * dispatcher it is supposed to mirror.
     *
     * MXFP8_LT was rejected here until 2026-08-17, deliberately and correctly:
     * gpu_graph_matmul_plain_tensor had no type-41 branch, so such a tensor
     * would have passed load and dispatched into nothing. The guard did its job
     * -- a repacked artifact was tried that day and died HERE, at load, instead
     * of misbehaving. The arm exists now, so the type is in the set. */
    if (!pulsar_weight_is_plain_or_mxfp8(t->type)) {
        fprintf(stderr, "pulsar: tensor %.*s has type %s, which the plain matmul "
                        "cannot dispatch\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        pulsar_die("unsupported weight type for the plain matmul path");
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}




/* For weights whose consumer reads f32 OR bf16 storage, chosen per tensor at
 * load (see pulsar_w_load in the CUDA internal header).  These are tensors the
 * checkpoint holds in bf16 and we used to widen to f32 for no reason -- the
 * drafter's markov head and confidence projection, and the norm weights.
 * Accepting both keeps one binary able to load an artifact from either side of
 * that change; every caller must pass the tensor's actual type down to the
 * kernel, never assume. */
static void tensor_expect_f32_or_bf16(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type == PULSAR_TENSOR_BF16)
        tensor_expect_layout(t, PULSAR_TENSOR_BF16, ndim, d0, d1, d2);
    else
        tensor_expect_layout(t, PULSAR_TENSOR_F32, ndim, d0, d1, d2);
}

static void tensor_expect_plain_layout(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    /* Accepts the three NON-fp8 arms of gpu_graph_matmul_plain_tensor, which is
     * what every caller here feeds. BF16 was missing until 2026-08-16 and the
     * drafter's router is bf16 now, so an otherwise-correct artifact died at
     * load with "expected F16 or F32". The main model's router is checked by
     * tensor_expect_plain_or_mxfp8 and had already been extended -- the drafter
     * goes through THIS one, and changing the type policy without walking every
     * validator that sees it is exactly how that gets missed. */
    if (t->type != PULSAR_TENSOR_BF16 &&
        t->type != PULSAR_TENSOR_F32) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %s, expected F32 or BF16\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}



/* The two routed-expert types the engine still reads.  IQ2_XXS (16),
 * IQ2_XXS_SOA (42), Q2_K (10) and FP4_E2M1 (39) were dropped: a scan of the
 * shipped artifact found only types 0/1/26/38/40/41/44 in the file, none of the
 * four is ever synthesised at load (they can only arrive FROM a gguf), and the
 * kernels behind them are gone.  Refusing here is what keeps that honest -- an
 * old artifact now fails to load with a clear message instead of dispatching
 * into a reader that no longer exists. */
static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == PULSAR_TENSOR_IQ2_XXS_MMQ_K ||
           type == PULSAR_TENSOR_CUTLASS_MXFP4;
}



static PULSAR_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    /* IQ2_XXS_MMQ_K (44) is a pure permutation of the raw IQ2_XXS blocks (L202)
     * MMQ's aligned-SoA layout.  Same 66 B/block, so row bytes, expert stride
     * and tensor size are unchanged; that is why it drops into the existing
     * offset machinery.  Only the kernel's read pattern differs. */
    case PULSAR_TENSOR_IQ2_XXS_MMQ_K: return sizeof(block_iq2_xxs);
    default:                 pulsar_die("unsupported routed expert tensor type");
    }
    return 0;
}



PULSAR_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const pulsar_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) pulsar_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}



/* Computes (gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes)
 * for any supported routed-expert quant combo, centralizing the
 * dispatch-site pattern `row_bytes = routed_expert_row_bytes(t); expert_bytes =
 * t->dim[1] * row_bytes` that's repeated across gpu_prefill.cpp/gpu_decode.cpp.
 *
 * For CUTLASS_MXFP4 (type 40) "row_bytes" has no ordinary per-row meaning --
 * the tensor is expert-major ColumnMajor+swizzle with no per-row byte stride
 * at all. It instead carries the data/SF split point within each expert's
 * block: the SF blob starts *row_bytes bytes into that expert's slice, and
 * *expert_bytes is the full [data + SF] stride to the next expert. Callers
 * that dispatch on gate->type == PULSAR_TENSOR_CUTLASS_MXFP4 must read it that
 * way; only the CUTLASS MoE path does. */
bool routed_expert_gate_down_layout(
        const pulsar_tensor *gate,
        const pulsar_tensor *down,
        uint64_t         *gate_expert_bytes,
        uint64_t         *gate_row_bytes,
        uint64_t         *down_expert_bytes,
        uint64_t         *down_row_bytes) {
    /* NOTE: gate and down are NOT always the same type -- gate/up and down
     * formats pair freely per layer by design (see
     * tensor_expect_routed_expert_combo). Each side's layout is computed
     * independently so MIXED layers (cutlass_mxfp4 on one side, iq2/q2k on the
     * other) resolve correctly: the CUTLASS_MXFP4 side yields stride/split-point,
     * the dp4a side yields ordinary expert/row byte counts. */
    if (!gate || !down) return false;

    if (gate->type == PULSAR_TENSOR_CUTLASS_MXFP4) {
        uint64_t gate_sf, gate_stride;
        cutlass_mxfp4_expert_layout(gate->dim[0], gate->dim[1],
                                     gate_row_bytes, &gate_sf, &gate_stride);
        *gate_expert_bytes = gate_stride;
    } else {
        *gate_row_bytes = routed_expert_row_bytes(gate);
        if (*gate_row_bytes == 0 || gate->dim[1] > UINT64_MAX / *gate_row_bytes) return false;
        *gate_expert_bytes = gate->dim[1] * *gate_row_bytes;
    }

    if (down->type == PULSAR_TENSOR_CUTLASS_MXFP4) {
        uint64_t down_sf, down_stride;
        cutlass_mxfp4_expert_layout(down->dim[0], down->dim[1],
                                     down_row_bytes, &down_sf, &down_stride);
        *down_expert_bytes = down_stride;
    } else {
        *down_row_bytes = routed_expert_row_bytes(down);
        if (*down_row_bytes == 0 || down->dim[1] > UINT64_MAX / *down_row_bytes) return false;
        *down_expert_bytes = down->dim[1] * *down_row_bytes;
    }
    return true;
}



/* The CUDA routed-MoE dispatcher selects kernels per role and per layer:
 * gate/up (always a matching pair -- the fused gate+up kernels assume one
 * format) in {IQ2_XXS, Q2_K, MXFP4}, down in {IQ2_XXS, Q2_K, MXFP4}, in any
 * pairing, and the combo may differ layer to layer (prisma per-layer
 * allocation). CUTLASS_MXFP4 is the exception: the grouped tensor-core GEMM
 * path runs the whole expert FFN in one dispatch, so cutlass gate/up
 * requires cutlass down (and vice versa). Reject anything else at load with
 * one clear error instead of a silent kernel-dispatch failure at the first
 * MoE layer. */
static void tensor_expect_routed_expert_combo(
        const pulsar_tensor *gate,
        const pulsar_tensor *up,
        const pulsar_tensor *down) {
    /* gate/up must match (the fused gate+up kernels assume one format). Each of
     * gate/up and down is independently either IQ2_XXS_MMQ_K (44, read by the MMQ
     * arms) or CUTLASS_MXFP4 (40) -- the GPU MoE path handles all-cutlass
     * (uniform, grouped/gemv), all-MMQ, AND the two MIXED shapes via
     * per-projection dispatch, which is what the shipped artifact needs: its 43
     * routed layers are 9 all-40, 27 all-44 and 7 mixed.
     *
     * The old dp4a types (IQ2_XXS 16, IQ2_XXS_SOA 42, Q2_K 10, FP4_E2M1 39) are
     * gone along with their kernels, so the former "bad_mix" cross (CUTLASS
     * against a legacy type-39 side) can no longer be expressed and its check
     * went with them. */
    const bool gate_up_pair = gate->type == up->type;
    const bool gate_ok = tensor_is_routed_expert_type(gate->type);
    const bool down_ok = tensor_is_routed_expert_type(down->type);
    if (gate_up_pair && gate_ok && down_ok) return;
    fprintf(stderr,
            "pulsar: unsupported routed expert quant combo at tensor %.*s: "
            "gate=%s up=%s down=%s\n"
            "  gate/up must match, and each of gate/up and down must be one of:",
            (int)gate->name.len,
            gate->name.ptr,
            tensor_type_name(gate->type),
            tensor_type_name(up->type),
            tensor_type_name(down->type));
    /* DERIVED from tensor_is_routed_expert_type(), for the reason spelled out on
     * weights_reject_unsupported_types().  This message used to name
     * "iq2_xxs_mmq (43)" by hand; that reader was deleted (L202) and 43 is
     * refused, so the engine was telling users to repack their artifact into a
     * type it then rejects -- L207's failure mode, in the one message L207
     * missed. */
    for (uint32_t t = 0; t < 256u; ++t) {
        if (tensor_is_routed_expert_type(t)) {
            fprintf(stderr, " %s (%u)", tensor_type_name(t), t);
        }
    }
    fprintf(stderr, "\n  the combo may differ per layer\n");
    exit(1);
}



static void tensor_expect_routed_expert(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_dims(t, ndim, d0, d1, d2);
}



/* V4.1 (L218): the head collapses the stream with the LAST FFN's pre-mix, so
 * there are no output_hc_* tensors -- the head is the norm and the projection. */
bool weights_have_output_head(const pulsar_weights *w) {
    return w && w->output_norm && w->output;
}



static bool weights_have_partial_output_head(const pulsar_weights *w) {
    return w && (w->output_norm || w->output);
}



static bool weights_layer_has_required(const pulsar_layer_weights *l, uint32_t il) {
    if (!l) return false;
    if (!l->hc_attn_fn ||
        !l->hc_attn_scale ||
        !l->hc_attn_base ||
        !l->attn_norm ||
        !l->attn_q_a ||
        !l->attn_q_a_norm ||
        !l->attn_q_b ||
        !l->attn_kv ||
        !l->attn_kv_a_norm ||
        !l->attn_sinks ||
        !l->attn_output_a ||
        !l->attn_output_b ||
        !l->hc_ffn_fn ||
        !l->hc_ffn_scale ||
        !l->hc_ffn_base ||
        !l->ffn_norm ||
        !l->ffn_gate_inp ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        !l->ffn_gate_shexp ||
        !l->ffn_up_shexp ||
        !l->ffn_down_shexp)
    {
        return false;
    }
    /* 0731: a hash-routed layer without its table cannot route at all. */
    if (il < PULSAR_N_HASH_LAYER && !l->ffn_gate_tid2eid) return false;

    const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
    if (pulsar_attn_owns_kv(a->mode) &&
        (!l->attn_compressor_kv ||
         !l->attn_compressor_norm ||
         (a->ratio > 1 && !l->attn_compressor_gate) ||
         (g_pulsar_shape.compressor_ape && !l->attn_compressor_ape) ||
         (!g_pulsar_shape.indexer_own_compressor && (!l->indexer_k || !l->indexer_k_norm))))
    {
        return false;
    }
    if (pulsar_attn_runs_indexer(a->mode) &&
        (!l->indexer_attn_q_b || !l->indexer_proj ||
         (g_pulsar_shape.indexer_own_compressor &&
          (!l->indexer_compressor_ape || !l->indexer_compressor_kv ||
           !l->indexer_compressor_gate || !l->indexer_compressor_norm))))
    {
        return false;
    }
    return true;
}



const pulsar_layer_weights *weights_first_bound_layer(const pulsar_weights *w) {
    if (!w) return NULL;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (weights_layer_has_required(&w->layer[il], il)) return &w->layer[il];
    }
    return NULL;
}



/* Verify every tensor type and dimension used by the specialized pipeline.
 * Token embedding and output head are validated when present. */
static void weights_validate_layout(
        const pulsar_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    const uint64_t hc_dim = (uint64_t)PULSAR_N_EMBD * PULSAR_N_HC;
    const uint64_t hc_mix_dim = 2u * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;

    if (!w) pulsar_die("internal error: missing weights while validating layout");
    if (layer_start >= PULSAR_N_LAYER) pulsar_die("invalid first layer in weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = PULSAR_N_LAYER - 1u;
    if (layer_end >= PULSAR_N_LAYER || layer_end < layer_start) {
        pulsar_die("invalid layer range in weight layout validation");
    }

    if (require_token_embd && !w->token_embd) pulsar_die("required token embedding tensor is missing");
    if (w->token_embd) {
        /* BF16 upstream. F16 and BF16 are both 2 bytes here, so this costs
         * nothing to store either way -- but f16 is a LOSSY copy of a bf16
         * source (it flushes anything under 5.96e-8 to zero) while bf16 is an
         * exact one. There is no version of this where f16 is the better
         * choice; it was simply never revisited. */
        tensor_expect_layout(w->token_embd, PULSAR_TENSOR_BF16, 2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
    }

    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) pulsar_die("required output head tensors are missing");
    if (weights_have_partial_output_head(w) && !have_output) pulsar_die("partial output head in GGUF");
    if (have_output) {
        tensor_expect_f32_or_bf16(w->output_norm,  1, PULSAR_N_EMBD, 0, 0);
        /* Output head is BF16 (source format, kept lossless by a dedicated BF16
         * matmul) or MXFP8 (routed to the FP8 matmul).  Source ships this head
         * bf16, so MXFP8 is BELOW source here -- the one place in the model
         * where a dense tensor is.  Both are accepted so the two can be
         * A/B'd against the same binary. */
        if (w->output->type == PULSAR_TENSOR_BF16)
            tensor_expect_layout(w->output,      PULSAR_TENSOR_BF16, 2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
        else
            tensor_expect_mxfp8(w->output,       2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const pulsar_layer_weights *l = &w->layer[il];
        if (!weights_layer_has_required(l, il)) {
            fprintf(stderr, "pulsar: required tensors for layer %u are missing\n", il);
            exit(1);
        }

        tensor_expect_plain_or_mxfp8(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  PULSAR_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   PULSAR_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_f32_or_bf16(l->attn_norm,  1, PULSAR_N_EMBD, 0, 0);
        tensor_expect_mxfp8(l->attn_q_a,        2, PULSAR_N_EMBD, PULSAR_N_LORA_Q, 0);
        tensor_expect_f32_or_bf16(l->attn_q_a_norm,  1, PULSAR_N_LORA_Q, 0, 0);
        tensor_expect_mxfp8(l->attn_q_b,        2, PULSAR_N_LORA_Q, q_dim, 0);
        tensor_expect_mxfp8(l->attn_kv,         2, PULSAR_N_EMBD, PULSAR_N_HEAD_DIM, 0);
        tensor_expect_f32_or_bf16(l->attn_kv_a_norm,  1, PULSAR_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     PULSAR_TENSOR_F32,  1, PULSAR_N_HEAD, 0, 0);
        tensor_expect_mxfp8(l->attn_output_a,   2, PULSAR_N_HEAD_DIM * (PULSAR_N_HEAD / PULSAR_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_mxfp8(l->attn_output_b,   2, out_low_dim, PULSAR_N_EMBD, 0);

        /* The compressor lives on the kv sources only, the indexer's q/score
         * weights on every index source.  A ratio-1 compressor is a plain
         * projection + norm (no gate).  On 0731 the compressor width is
         * coff * head_dim (coff 2 at ratio 4, 1 at ratio 128 -- pulsar_compress_coff)
         * and both compressors carry an ape; the index KEY comes from the
         * indexer's own compressor instead of the kv source's latent. */
        const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
        if (pulsar_attn_owns_kv(a->mode)) {
            const uint64_t comp_width = (uint64_t)pulsar_compress_coff(a->ratio) * PULSAR_N_HEAD_DIM;
            if (g_pulsar_shape.compressor_ape) {
                tensor_expect_plain_or_mxfp8(l->attn_compressor_ape, 2, comp_width, a->ratio, 0);
            }
            tensor_expect_plain_or_mxfp8(l->attn_compressor_kv, 2, PULSAR_N_EMBD, comp_width, 0);
            if (a->ratio > 1) tensor_expect_plain_or_mxfp8(l->attn_compressor_gate, 2, PULSAR_N_EMBD, comp_width, 0);
            tensor_expect_f32_or_bf16(l->attn_compressor_norm, 1, PULSAR_N_HEAD_DIM, 0, 0);
            if (!g_pulsar_shape.indexer_own_compressor) {
                tensor_expect_plain_or_mxfp8(l->indexer_k, 2, PULSAR_N_HEAD_DIM, PULSAR_N_INDEXER_HEAD_DIM, 0);
                tensor_expect_f32_or_bf16(l->indexer_k_norm, 1, PULSAR_N_INDEXER_HEAD_DIM, 0, 0);
            }
        }
        if (pulsar_attn_runs_indexer(a->mode)) {
            const uint64_t index_q_dim = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2ull * PULSAR_N_INDEXER_HEAD_DIM;
            tensor_expect_plain_or_mxfp8(l->indexer_attn_q_b, 2, PULSAR_N_LORA_Q, index_q_dim, 0);
            tensor_expect_plain_or_mxfp8(l->indexer_proj, 2, PULSAR_N_EMBD, PULSAR_N_INDEXER_HEAD, 0);
            if (g_pulsar_shape.indexer_own_compressor) {
                tensor_expect_plain_or_mxfp8(l->indexer_compressor_ape, 2, index_width, a->ratio, 0);
                tensor_expect_plain_or_mxfp8(l->indexer_compressor_kv, 2, PULSAR_N_EMBD, index_width, 0);
                tensor_expect_plain_or_mxfp8(l->indexer_compressor_gate, 2, PULSAR_N_EMBD, index_width, 0);
                tensor_expect_f32_or_bf16(l->indexer_compressor_norm, 1, PULSAR_N_INDEXER_HEAD_DIM, 0, 0);
            }
        }

        tensor_expect_plain_or_mxfp8(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   PULSAR_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    PULSAR_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_f32_or_bf16(l->ffn_norm,  1, PULSAR_N_EMBD, 0, 0);
        /* Router + bias stay padded to the full n_expert; only the expert
         * weight tensors are dense-trimmed to the per-layer survivor count
         * (== n_expert for un-pruned models). */
        const uint32_t n_layer_expert = pulsar_layer_n_expert(il);
        tensor_expect_plain_or_mxfp8(l->ffn_gate_inp, 2, PULSAR_N_EMBD, PULSAR_N_EXPERT, 0);
        /* OPTIONAL: absent from Vision-Exp's serving artifact; the router has a
         * bias-less arm and the bind below is optional_tensorf. */
        if (l->ffn_exp_probs_b)
            tensor_expect_layout(l->ffn_exp_probs_b, PULSAR_TENSOR_F32, 1, PULSAR_N_EXPERT, 0, 0);
        if (l->ffn_gate_tid2eid) {
            /* [n_expert_used, n_vocab]: one row of expert ids per token id. */
            tensor_expect_layout(l->ffn_gate_tid2eid, PULSAR_TENSOR_I32, 2,
                                 PULSAR_N_EXPERT_USED, PULSAR_N_VOCAB, 0);
        }
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, PULSAR_N_EMBD, PULSAR_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, PULSAR_N_EMBD, PULSAR_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, PULSAR_N_FF_EXP, PULSAR_N_EMBD, n_layer_expert);
        tensor_expect_routed_expert_combo(l->ffn_gate_exps,
                                          l->ffn_up_exps,
                                          l->ffn_down_exps);
        tensor_expect_mxfp8(l->ffn_gate_shexp, 2, PULSAR_N_EMBD, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_up_shexp,   2, PULSAR_N_EMBD, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_down_shexp, 2, PULSAR_N_FF_EXP, PULSAR_N_EMBD, 0);
    }
}



static bool pulsar_shape_matches_metadata(
        const pulsar_shape *s,
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return s->n_layer == n_layer &&
           s->n_embd == n_embd &&
           s->n_vocab == n_vocab &&
           s->n_head == n_head &&
           s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim &&
           s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot &&
           s->n_lora_q == n_lora_q &&
           s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group &&
           s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used &&
           s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared &&
           s->n_swa == n_swa &&
           s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k &&
           s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}



static void pulsar_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    /* The ONE model-identity dispatch in the engine, and it runs once at load.
     * The geometry alone identifies the profile: 0731 and V4.1 differ in six of
     * these fields (layers 43/40, embd 4096/5120, experts 256/384, ff_exp
     * 2048/2304, q_lora 1024/1280, indexer heads 64/32), so nothing downstream
     * needs to test which model is loaded.
     * plans/96-two-profiles-one-engine.md s4. */
    static const pulsar_shape *const profiles[] = { &PULSAR_SHAPE_V4, &PULSAR_SHAPE_V41 };

    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++) {
        if (pulsar_shape_matches_metadata(profiles[i],
                                       n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                       n_head_dim, n_value_dim, n_rot, n_lora_q,
                                       n_lora_o, n_out_group, n_expert,
                                       n_expert_used, n_ff_exp, n_expert_shared,
                                       n_swa, n_indexer_head,
                                       n_indexer_head_dim, n_indexer_top_k, n_hc,
                                       n_hc_sinkhorn_iter)) {
            g_pulsar_shape = *profiles[i];
            /* The CUDA module packs the rows, so it needs the family too.  The
             * profile stays the authority; this is the seam's copy of one fact,
             * pushed once, here. */
            pulsar_gpu_set_kv_row_style(g_pulsar_shape.kv_row_style);
            return;
        }
    }

    fprintf(stderr,
            "pulsar: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}



/* Read an int32/uint32 array key into `out` (capacity `cap`), returning its
 * length.  Dies when the key is missing, the wrong type, longer than `cap`, or
 * carries a negative value. */
static uint32_t model_read_u32_array(const pulsar_model *m, const char *key, uint32_t *out, uint32_t cap) {
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "pulsar: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len > cap) {
        fprintf(stderr, "pulsar: %s has %llu entries, at most %u are meaningful\n",
                key, (unsigned long long)arr.len, cap);
        exit(1);
    }
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &out[i])) pulsar_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            if (v < 0) pulsar_die("metadata array contains a negative value");
            out[i] = (uint32_t)v;
        }
    }
    return (uint32_t)arr.len;
}



/* The per-layer ratios and the source sets that shape the attention layout.
 *
 * compress_ratios is the artifact's on BOTH models and is required.  The source
 * sets and the candidate pool are the PROFILE's: 0731 compresses every layer
 * >= 2 on its own and has no candidate pool, so its artifact carries none of
 * those keys, while V4.1's artifact carries all four and carries no
 * hash_layer_count.  When a key IS present it must agree with the profile --
 * pulsar_attn_layout_install enforces that against g_pulsar_shape.
 * plans/96-SPIKE0-results.md S2. */
static void validate_attention_layout_metadata(const pulsar_model *m) {
    uint32_t ratios[PULSAR_MAX_LAYER];
    const uint32_t n_ratio = model_read_u32_array(m, "deepseek4.attention.compress_ratios", ratios, PULSAR_MAX_LAYER);
    if (n_ratio < PULSAR_N_LAYER) pulsar_die("deepseek4.attention.compress_ratios is shorter than the layer count");

    const pulsar_shape *sh = &g_pulsar_shape;
    pulsar_array_ref probe;

    /* Undeclared -> the profile's set (the degenerate 0731 case: every
     * compressed layer is its own source).  Declared -> the artifact's, which
     * the installer then checks against the profile. */
    uint32_t kv_sources[PULSAR_MAX_ATTN_SOURCE], index_sources[PULSAR_MAX_ATTN_SOURCE];
    uint32_t n_kv, n_index;
    if (model_get_array(m, "deepseek4.attention.kv_source_layers", &probe)) {
        n_kv = model_read_u32_array(m, "deepseek4.attention.kv_source_layers", kv_sources, PULSAR_MAX_ATTN_SOURCE);
    } else {
        n_kv = sh->n_kv_source;
        memcpy(kv_sources, sh->kv_source_layer, sizeof(kv_sources));
    }
    if (model_get_array(m, "deepseek4.attention.index_source_layers", &probe)) {
        n_index = model_read_u32_array(m, "deepseek4.attention.index_source_layers", index_sources, PULSAR_MAX_ATTN_SOURCE);
    } else {
        n_index = sh->n_index_source;
        memcpy(index_sources, sh->index_source_layer, sizeof(index_sources));
    }

    /* Same rule for the candidate pool: absent means "this model has none",
     * which is only consistent with a profile that has none (-1). */
    int32_t candidate = sh->candidate_source_layer;
    uint32_t declared_candidate = 0;
    if (model_get_u32(m, "deepseek4.attention.candidate_source_layer", &declared_candidate)) {
        if (declared_candidate > (uint32_t)INT32_MAX) {
            pulsar_die("deepseek4.attention.candidate_source_layer is out of range");
        }
        candidate = (int32_t)declared_candidate;
    }
    uint32_t topk_blocks = sh->candidate_topk_blocks, block_size = sh->candidate_block_size;
    (void)model_get_u32(m, "deepseek4.attention.candidate_topk_blocks", &topk_blocks);
    (void)model_get_u32(m, "deepseek4.attention.candidate_block_size", &block_size);
    if (topk_blocks != sh->candidate_topk_blocks || block_size != sh->candidate_block_size) {
        fprintf(stderr, "pulsar: candidate pool is %u blocks of %u, %s expects %u of %u\n",
                topk_blocks, block_size, PULSAR_MODEL_SHAPE_NAME,
                sh->candidate_topk_blocks, sh->candidate_block_size);
        exit(1);
    }
    pulsar_attn_layout_install(ratios, kv_sources, n_kv, index_sources, n_index, candidate);
}



static void config_expect_f32(const char *name, float got, float expected);



/* REAP ds4-compact-v1 expert pruning: populate g_pulsar_layer_expert_count from
 * reap.layer.keep_count. Absent/disabled -> array stays zeroed (every layer
 * falls back to the full n_expert). When present, each entry must be in
 * [1, n_expert]; the router/bias tensors stay padded to n_expert and only the
 * expert weight tensors are dense-trimmed to keep_count. */
static void validate_reap_metadata(const pulsar_model *m) {
    memset(g_pulsar_layer_expert_count, 0, sizeof(g_pulsar_layer_expert_count));

    bool enabled = false;
    if (!model_get_bool(m, "reap.enabled", &enabled) || !enabled) return;

    const char *key = "reap.layer.keep_count";
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        pulsar_die("reap.enabled is set but reap.layer.keep_count is missing or not an int32/uint32 array");
    }
    if (arr.len < PULSAR_N_LAYER) {
        pulsar_die("reap.layer.keep_count is shorter than the layer count");
    }

    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) pulsar_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            if (v < 0) pulsar_die("reap.layer.keep_count contains a negative value");
            got = (uint32_t)v;
        }
        if (got == 0 || got > PULSAR_N_EXPERT) {
            fprintf(stderr,
                    "pulsar: reap.layer.keep_count[%u]=%u out of range [1, %u]\n",
                    il, got, PULSAR_N_EXPERT);
            exit(1);
        }
        g_pulsar_layer_expert_count[il] = got;
    }
}



static void validate_swiglu_clamp_metadata(const pulsar_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "pulsar: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < PULSAR_N_LAYER) {
        pulsar_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < PULSAR_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) pulsar_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, PULSAR_SWIGLU_CLAMP_EXP);
    }
}



static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "pulsar: expected %s=%u for %s, got %u\n",
            name, expected, PULSAR_MODEL_SHAPE_NAME, got);
    exit(1);
}



/* Relative tolerance: the old absolute 1e-6 floor let an artifact's rms eps of
 * 1e-20 pass against a 1e-6 profile (and vice versa) without a word -- every
 * RMSNorm wrong, no message (L218 audit risk #1).  A value of 0 must match 0. */
static void config_expect_f32(const char *name, float got, float expected) {
    const float tol = fabsf(expected) * 1.0e-6f;
    if (fabsf(got - expected) <= tol) return;
    fprintf(stderr, "pulsar: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, PULSAR_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}



static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "pulsar: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", PULSAR_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}



static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 PULSAR_N_LAYER);
}



/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
void config_validate_model(const pulsar_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    pulsar_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         PULSAR_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        PULSAR_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         PULSAR_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     PULSAR_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      PULSAR_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    PULSAR_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          PULSAR_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    PULSAR_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        PULSAR_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        PULSAR_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        PULSAR_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   PULSAR_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        PULSAR_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, PULSAR_N_EXPERT_SHARED);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   PULSAR_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     PULSAR_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, PULSAR_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    PULSAR_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, PULSAR_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, PULSAR_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_attention_layout_metadata(m);

    validate_reap_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = PULSAR_ROPE_ORIG_CTX;
    model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != PULSAR_ROPE_ORIG_CTX) {
        fprintf(stderr, "pulsar: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)PULSAR_ROPE_ORIG_CTX, PULSAR_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, PULSAR_ROPE_FREQ_BASE);
    float rope_scale_factor = PULSAR_ROPE_SCALE_FACTOR;
    model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, PULSAR_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = PULSAR_ROPE_YARN_BETA_FAST;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, PULSAR_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = PULSAR_ROPE_YARN_BETA_SLOW;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, PULSAR_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, PULSAR_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, PULSAR_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    /* THE ARTIFACT IS THE AUTHORITY, not the profile.  This is the one text
     * config field the shipped checkpoints disagree on, and they disagree
     * ACROSS the profile split rather than along it: 0731 declares 1e-6 while
     * BOTH Vision-Exp and V4.1 declare 1e-20.  It is a kernel parameter
     * (PULSAR_RMS_EPS == g_pulsar_shape.rms_eps), not a shape, so pinning it per
     * profile made the compiled default refuse a real artifact -- Vision-Exp
     * (v5-vexp-full256) failed to load with "expected 1e-06, got 1e-20" until
     * this was read off the artifact instead.  Only the known pair is accepted:
     * anything else exits rather than silently retuning the norm of every
     * layer. */
    if (rms_eps != PULSAR_V4_RMS_EPS && rms_eps != PULSAR_V41_RMS_EPS) {
        fprintf(stderr, "pulsar: attention.layer_norm_rms_epsilon=%.9g is neither 0731's %.9g "
                        "nor Vision-Exp's/V4.1's %.9g\n",
                (double)rms_eps, (double)PULSAR_V4_RMS_EPS, (double)PULSAR_V41_RMS_EPS);
        exit(1);
    }
    g_pulsar_shape.rms_eps = rms_eps;
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, PULSAR_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);

    /* The hash-routed layer count follows the same rule as the source sets: it
     * is the PROFILE's.  The 0731 artifact declares it (3); V4.1's does not and
     * has none.  Declared -> it must agree.  The restored hash-routing arm reads
     * PULSAR_N_HASH_LAYER, so a wrong value here would route by id in a model
     * that has no id table. */
    uint32_t declared_hash = 0;
    if (model_get_u32(m, "deepseek4.hash_layer_count", &declared_hash) &&
        declared_hash != (uint32_t)PULSAR_N_HASH_LAYER) {
        fprintf(stderr, "pulsar: hash_layer_count is %u, %s expects %u\n",
                declared_hash, PULSAR_MODEL_SHAPE_NAME, (unsigned)PULSAR_N_HASH_LAYER);
        exit(1);
    }
}



/* Weight formats the engine still decodes -- exactly the seven types present in
 * the shipped artifact (a full scan of its 1406 tensors found 0/1/26/38/40/41/43
 * and nothing else).  Q4_K, Q8_0, Q2_K (10), IQ2_XXS (16), FP4_E2M1 (39) and
 * IQ2_XXS_SOA (42) have been removed along with their readers.
 * Rejecting up front gives one clear error instead of failing on the first
 * per-tensor layout check -- or, worse, dispatching into a deleted arm.
 *
 * BF16 (30) is BACK as of 2026-08-15, after 0f6e1dd removed it as unreachable.
 * That removal was correct at the time and is not being second-guessed: the
 * artifact genuinely had no BF16 tensor.  It does now, because the tensors that
 * are bf16 or f32 upstream are stored bf16 instead of f16 -- f16 was deleting
 * ~11% of the small weights in some of them -- and because the output head goes
 * back to bf16 to match source.  Both need this type accepted. */
static bool weights_tensor_type_supported(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_F32:
    case PULSAR_TENSOR_I32:
    case PULSAR_TENSOR_BF16:
    case PULSAR_TENSOR_FP8_E4M3:
    case PULSAR_TENSOR_MXFP8_LT:
    case PULSAR_TENSOR_CUTLASS_MXFP4:
    case PULSAR_TENSOR_IQ2_XXS_MMQ_K:
    case PULSAR_TENSOR_FP8_E4M3_SOA_K:
        return true;
    default:
        return false;
    }
}



static void weights_reject_unsupported_types(const pulsar_model *m) {
    bool seen[256] = { false };
    bool any = false;

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const uint32_t type = m->tensors[i].type;
        if (weights_tensor_type_supported(type)) continue;
        if (type < 256 && seen[type]) continue;
        if (type < 256) seen[type] = true;
        fprintf(stderr,
                "pulsar: unsupported weight tensor type %s (first tensor: %.*s)\n",
                tensor_type_name(type),
                (int)m->tensors[i].name.len,
                m->tensors[i].name.ptr);
        any = true;
    }
    if (any) {
        /* DERIVED from weights_tensor_type_supported(), not a second list kept
         * in step by hand.  This is the only thing a user sees when an artifact
         * is refused, and a list naming types the engine no longer reads sends
         * them looking for a bug in their file instead of repacking it -- which
         * is exactly what it did after the reader for iq2_xxs_mmq (43) was
         * removed and the prose still advertised it (L207). */
        fprintf(stderr, "pulsar: supported weight tensor types:");
        for (uint32_t t = 0; t < 256u; ++t) {
            const char *tname = tensor_type_name(t);
            if (tname != NULL && weights_tensor_type_supported(t)) {
                fprintf(stderr, " %s (%u)", tname, t);
            }
        }
        fprintf(stderr, "\n");
        exit(1);
    }
}



static void weights_bind_output(pulsar_weights *w, const pulsar_model *m, bool required, bool optional) {
    /* 0731's HC head mix.  REQUIRED where the profile says the head computes its
     * own coefficients: a 0731 artifact missing the group would otherwise
     * collapse with V4.1's carried pre -- the wrong arithmetic for the model,
     * and it would not fail.  V4.1 has no such tensors. */
    if (g_pulsar_shape.hc_head_mix) {
        w->output_hc_fn    = required_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale = required_tensor(m, "output_hc_scale.weight");
        w->output_hc_base  = required_tensor(m, "output_hc_base.weight");
    }
    if (required) {
        w->output_norm      = required_tensor(m, "output_norm.weight");
        w->output           = required_tensor(m, "output.weight");
        return;
    }
    if (!optional) return;

    w->output_norm      = model_find_tensor(m, "output_norm.weight");
    w->output           = model_find_tensor(m, "output.weight");
    if (weights_have_partial_output_head(w) && !weights_have_output_head(w)) {
        pulsar_die("partial output head in GGUF");
    }
}



static void weights_bind_layer(pulsar_layer_weights *l, const pulsar_model *m, uint32_t il) {
    const pulsar_layer_attn *attn = pulsar_layer_attn_layout(il);
    l->n_expert = PULSAR_N_EXPERT;
    l->n_expert_used = PULSAR_N_EXPERT_USED;
    l->n_expert_present = pulsar_layer_n_expert(il);

    l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
    l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
    l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
    l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
    l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
    l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
    /* The compressor lives on every kv source; the indexer's q/score weights on
     * every index source; a REUSE layer owns neither.  Which index-KEY family a
     * model uses is the profile's: V4.1 projects it from the kv source's latent
     * (indexer.attn_k / k_norm), 0731 compresses its own
     * (indexer_compressor_*).  Any of these tensors on a layer whose mode does
     * not own it is a wrong artifact. */
    if (pulsar_attn_owns_kv(attn->mode)) {
        if (g_pulsar_shape.compressor_ape) {
            l->attn_compressor_ape = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
        }
        l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
        if (attn->ratio > 1) l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
        l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
        if (!g_pulsar_shape.indexer_own_compressor) {
            l->indexer_k        = required_tensorf(m, "blk.%u.indexer.attn_k.weight", il);
            l->indexer_k_norm   = required_tensorf(m, "blk.%u.indexer.k_norm.weight", il);
        }
    }
    if (pulsar_attn_runs_indexer(attn->mode)) {
        l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
        l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
        if (g_pulsar_shape.indexer_own_compressor) {
            l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
            l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
            l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
            l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
        }
    }
    static const char *const attn_owned[] = {
        "attn_compressor_ape.weight",
        "attn_compressor_kv.weight", "attn_compressor_gate.weight", "attn_compressor_norm.weight",
        "indexer.attn_k.weight", "indexer.k_norm.weight", "indexer.attn_q_b.weight", "indexer.proj.weight",
        "indexer_compressor_ape.weight", "indexer_compressor_kv.weight",
        "indexer_compressor_gate.weight", "indexer_compressor_norm.weight",
    };
    const pulsar_tensor *const attn_bound[] = {
        l->attn_compressor_ape,
        l->attn_compressor_kv, l->attn_compressor_gate, l->attn_compressor_norm,
        l->indexer_k, l->indexer_k_norm, l->indexer_attn_q_b, l->indexer_proj,
        l->indexer_compressor_ape, l->indexer_compressor_kv,
        l->indexer_compressor_gate, l->indexer_compressor_norm,
    };
    for (size_t i = 0; i < sizeof(attn_owned) / sizeof(attn_owned[0]); i++) {
        char name[128];
        const int n = snprintf(name, sizeof(name), "blk.%u.%s", il, attn_owned[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) pulsar_die("tensor name is too long");
        if (!attn_bound[i] && model_find_tensor(m, name)) {
            fprintf(stderr, "pulsar: layer %u carries %s but its CSA2 mode does not own it -- refusing\n", il, name);
            exit(1);
        }
    }
    l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
    l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
    l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
    l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    /* OPTIONAL, like the drafter's: text-only artifacts do not ship it. */
    l->ffn_exp_probs_b = optional_tensorf(m, "blk.%u.exp_probs_b.bias", il);
    /* 0731's leading layers route by token id.  Required exactly on the hash
     * layers and REFUSED on any other, so an artifact cannot carry a table that
     * would silently replace the gate's routing. */
    if (il < PULSAR_N_HASH_LAYER) {
        l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
    } else {
        char tid_name[128];
        const int n = snprintf(tid_name, sizeof(tid_name), "blk.%u.ffn_gate_tid2eid.weight", il);
        if (n < 0 || (size_t)n >= sizeof(tid_name)) pulsar_die("tensor name is too long");
        if (model_find_tensor(m, tid_name)) {
            fprintf(stderr, "pulsar: layer %u carries ffn_gate_tid2eid but only the first %u "
                            "layers are hash-routed -- refusing\n",
                    il, (unsigned)PULSAR_N_HASH_LAYER);
            exit(1);
        }
    }
    l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
}



/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
void weights_bind(pulsar_weights *w, const pulsar_model *m) {
    memset(w, 0, sizeof(*w));
    weights_reject_unsupported_types(m);

    w->token_embd = required_tensor(m, "token_embd.weight");
    weights_bind_output(w, m, true, false);

    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        weights_bind_layer(&w->layer[il], m, il);
    }

    weights_validate_layout(w, 0, PULSAR_N_LAYER - 1u, true, true);
}





void weights_free(pulsar_weights *w) {
    memset(w, 0, sizeof(*w));
}

static void dspark_weights_validate_layout(const pulsar_dspark_weights *w) {
    const uint32_t E = w->embed_dim;
    const uint32_t V = w->vocab_size;
    /*
     * The tensor-layout checks below are self-referential: they validate every
     * DSpark tensor against E/V read from the support GGUF's own metadata.  But
     * the runtime drives DSpark with the target model's compiled constants
     * (PULSAR_N_EMBD embed buffers, PULSAR_N_VOCAB logits stride), so a support model
     * built for a different base would pass this validation and then produce
     * misaligned reads or garbage drafts.  Pin E/V to the target here.
     */
    if (E != PULSAR_N_EMBD || V != PULSAR_N_VOCAB) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "dspark: support model shape (embed=%u vocab=%u) does not match "
                 "target (embed=%u vocab=%u)",
                 E, V, (uint32_t)PULSAR_N_EMBD, (uint32_t)PULSAR_N_VOCAB);
        pulsar_die(msg);
    }
    const uint64_t hc_dim = (uint64_t)E * PULSAR_N_HC;
    const uint64_t hc_mix_dim = 2u * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;

    tensor_expect_mxfp8(w->main_proj, 2, 3ull * E, E, 0);
    tensor_expect_f32_or_bf16(w->main_norm, 1, E, 0, 0);

    for (int li = 0; li < 3; li++) {
        const pulsar_layer_weights *l = &w->layer[li];
        tensor_expect_plain_layout(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale, PULSAR_TENSOR_F32, 1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base, PULSAR_TENSOR_F32, 1, hc_mix_dim, 0, 0);
        tensor_expect_f32_or_bf16(l->attn_norm, 1, E, 0, 0);
        tensor_expect_mxfp8(l->attn_q_a, 2, E, PULSAR_N_LORA_Q, 0);
        tensor_expect_f32_or_bf16(l->attn_q_a_norm, 1, PULSAR_N_LORA_Q, 0, 0);
        tensor_expect_mxfp8(l->attn_q_b, 2, PULSAR_N_LORA_Q, q_dim, 0);
        tensor_expect_mxfp8(l->attn_kv, 2, E, PULSAR_N_HEAD_DIM, 0);
        tensor_expect_f32_or_bf16(l->attn_kv_a_norm, 1, PULSAR_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks, PULSAR_TENSOR_F32, 1, PULSAR_N_HEAD, 0, 0);
        tensor_expect_mxfp8(l->attn_output_a, 2,
                             PULSAR_N_HEAD_DIM * (PULSAR_N_HEAD / PULSAR_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_mxfp8(l->attn_output_b, 2, out_low_dim, E, 0);
        tensor_expect_plain_layout(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale, PULSAR_TENSOR_F32, 1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base, PULSAR_TENSOR_F32, 1, hc_mix_dim, 0, 0);
        tensor_expect_f32_or_bf16(l->ffn_norm, 1, E, 0, 0);
        tensor_expect_plain_layout(l->ffn_gate_inp, 2, E, PULSAR_N_DSPARK_EXPERT, 0);
        if (l->ffn_exp_probs_b)
            tensor_expect_layout(l->ffn_exp_probs_b, PULSAR_TENSOR_F32, 1, PULSAR_N_DSPARK_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, E, PULSAR_N_FF_EXP, PULSAR_N_DSPARK_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, E, PULSAR_N_FF_EXP, PULSAR_N_DSPARK_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, PULSAR_N_FF_EXP, E, PULSAR_N_DSPARK_EXPERT);
        tensor_expect_routed_expert_combo(l->ffn_gate_exps,
                                          l->ffn_up_exps,
                                          l->ffn_down_exps);
        tensor_expect_mxfp8(l->ffn_gate_shexp, 2, E, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_up_shexp,   2, E, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_down_shexp, 2, PULSAR_N_FF_EXP, E, 0);
    }

    /* bf16 upstream; f32 here until 2026-08-15. markov_w2 is the single
     * largest above-source tensor in the model (66 MB of pure width) and the
     * markov step streams all of it every draft position.
     *
     * markov_w2 is K-MAJOR (L213): dims (vocab, 256), element (v, i) at
     * i * vocab + v, so the markov kernels' warps read contiguous memory. The
     * source and markov_w1 are (256, vocab). An artifact still carrying the
     * v-major markov_w2 fails this dims check and refuses to load -- there is
     * no runtime transpose; gguf-tools/gguf_transpose_bf16_tensor.py migrates
     * an existing artifact in place and --verify proves the result.
     *
     * Its storage is one of three (L213 step 2; pulsar_markov_w2_fmt is the
     * type -> arm mapping): f32 or bf16 (the source's widths), or
     * FP8_E4M3_SOA_K (46: MXFP8 numerics -- E8M0 scale plane then E4M3 payload
     * plane -- the shipped table; alpha-neutral to bf16, 289 -> 161 us).  Plain
     * type 38 is NOT accepted anywhere in the artifact.  Any other type refuses
     * here; the dims contract is the same for all three. */
    tensor_expect_f32_or_bf16(w->markov_w1, 2, 256, V, 0);
    switch (w->markov_w2->type) {
    case PULSAR_TENSOR_F32:
    case PULSAR_TENSOR_BF16:
    case PULSAR_TENSOR_FP8_E4M3_SOA_K:
        tensor_expect_layout(w->markov_w2, w->markov_w2->type, 2, V, 256, 0);
        break;
    default:
        pulsar_die("dspark markov_w2: storage must be f32, bf16 or fp8_e4m3_soa_k (46)");
    }
    tensor_expect_f32_or_bf16(w->confidence_proj, 1, E + 256, 0, 0);
    tensor_expect_f32_or_bf16(w->final_norm, 1, E, 0, 0);
}

void dspark_weights_bind(pulsar_dspark_weights *w, const pulsar_model *m) {
    memset(w, 0, sizeof(*w));
    weights_reject_unsupported_types(m);

    w->embed_dim = required_u32(m, "deepseek_v4_dspark.embedding_length");
    /* The drafter's own shape keys, checked against what this engine was built
     * to draft: block_size is the depth it was trained to draft (the depth
     * controller's ceiling must not exceed it), the noise token fills the
     * draft slots past the first (forward_embed), the markov rank is the
     * width the k-major markov kernels are written for. */
    {
        const uint32_t n_expert = required_u32(m, "deepseek_v4_dspark.expert_count");
        const uint32_t n_used = required_u32(m, "deepseek_v4_dspark.expert_used_count");
        const uint32_t block_size = required_u32(m, "deepseek_v4_dspark.block_size");
        const uint32_t noise_id = required_u32(m, "deepseek_v4_dspark.noise_token_id");
        const uint32_t markov_rank = required_u32(m, "deepseek_v4_dspark.markov_rank");
        if (n_expert != (uint32_t)PULSAR_N_DSPARK_EXPERT || n_used != (uint32_t)PULSAR_N_DSPARK_EXPERT_USED ||
            block_size < (uint32_t)PULSAR_SPEC_DEPTH_MAX || noise_id != (uint32_t)PULSAR_DSPARK_NOISE_TOKEN_ID ||
            markov_rank != 256u) {
            char msg[320];
            snprintf(msg, sizeof(msg),
                     "dspark: artifact experts %u/%u, block_size %u, noise_token_id %u, markov_rank %u vs this "
                     "engine's %u/%u, depth ceiling %d, noise token %d, markov rank 256 -- refusing",
                     n_expert, n_used, block_size, noise_id, markov_rank,
                     (unsigned)PULSAR_N_DSPARK_EXPERT, (unsigned)PULSAR_N_DSPARK_EXPERT_USED,
                     (int)PULSAR_SPEC_DEPTH_MAX, (int)PULSAR_DSPARK_NOISE_TOKEN_ID);
            pulsar_die(msg);
        }
    }
    w->main_proj = required_tensor(m, "dspark.main_proj.weight");
    w->main_norm = required_tensor(m, "dspark.main_norm.weight");

    for (int li = 0; li < 3; li++) {
        pulsar_layer_weights *l = &w->layer[li];
        l->n_expert = PULSAR_N_DSPARK_EXPERT;
        l->n_expert_used = PULSAR_N_DSPARK_EXPERT_USED;
        l->n_expert_present = PULSAR_N_DSPARK_EXPERT;
        l->hc_attn_fn      = required_tensorf(m, "dspark.%d.hc_attn_fn.weight", li);
        l->hc_attn_scale   = required_tensorf(m, "dspark.%d.hc_attn_scale.weight", li);
        l->hc_attn_base    = required_tensorf(m, "dspark.%d.hc_attn_base.weight", li);
        l->attn_norm       = required_tensorf(m, "dspark.%d.attn_norm.weight", li);
        l->attn_q_a        = required_tensorf(m, "dspark.%d.attn_q_a.weight", li);
        l->attn_q_a_norm   = required_tensorf(m, "dspark.%d.attn_q_a_norm.weight", li);
        l->attn_q_b        = required_tensorf(m, "dspark.%d.attn_q_b.weight", li);
        l->attn_kv         = required_tensorf(m, "dspark.%d.attn_kv.weight", li);
        l->attn_kv_a_norm  = required_tensorf(m, "dspark.%d.attn_kv_a_norm.weight", li);
        l->attn_sinks      = required_tensorf(m, "dspark.%d.attn_sinks.weight", li);
        l->attn_output_a   = required_tensorf(m, "dspark.%d.attn_output_a.weight", li);
        l->attn_output_b   = required_tensorf(m, "dspark.%d.attn_output_b.weight", li);
        l->hc_ffn_fn       = required_tensorf(m, "dspark.%d.hc_ffn_fn.weight", li);
        l->hc_ffn_scale    = required_tensorf(m, "dspark.%d.hc_ffn_scale.weight", li);
        l->hc_ffn_base     = required_tensorf(m, "dspark.%d.hc_ffn_base.weight", li);
        l->ffn_norm        = required_tensorf(m, "dspark.%d.ffn_norm.weight", li);
        l->ffn_gate_inp    = required_tensorf(m, "dspark.%d.ffn_gate_inp.weight", li);
        /* the drafter's trained router bias (L216: every shipped 0731 artifact
         * routed its drafter without it; V4.1 ships one per drafter layer) */
        l->ffn_exp_probs_b = optional_tensorf(m, "dspark.%d.exp_probs_b.bias", li);
        l->ffn_gate_exps   = required_tensorf(m, "dspark.%d.ffn_gate_exps.weight", li);
        l->ffn_up_exps     = required_tensorf(m, "dspark.%d.ffn_up_exps.weight", li);
        l->ffn_down_exps   = required_tensorf(m, "dspark.%d.ffn_down_exps.weight", li);
        l->ffn_gate_shexp  = required_tensorf(m, "dspark.%d.ffn_gate_shexp.weight", li);
        l->ffn_up_shexp    = required_tensorf(m, "dspark.%d.ffn_up_shexp.weight", li);
        l->ffn_down_shexp  = required_tensorf(m, "dspark.%d.ffn_down_shexp.weight", li);
    }

    w->markov_w1        = required_tensor(m, "dspark.2.markov_head.markov_w1.weight");
    w->markov_w2        = required_tensor(m, "dspark.2.markov_head.markov_w2.weight");
    w->confidence_proj  = required_tensor(m, "dspark.2.confidence_head.proj.weight");
    w->final_norm       = required_tensor(m, "dspark.2.norm.weight");

    w->vocab_size = (uint32_t)w->markov_w1->dim[1];

    /* dspark_target_layer_ids: the artifact says which target layers' INPUT
     * hiddens the drafter conditions on (V4.1: 37, 38, 39 of 40).  No
     * derivation from the layer count -- a drafter trained on other anchors
     * would run and draft garbage. */
    w->target_layer_ids[0] = required_u32(m, "dspark.target_layer_ids.0");
    w->target_layer_ids[1] = required_u32(m, "dspark.target_layer_ids.1");
    w->target_layer_ids[2] = required_u32(m, "dspark.target_layer_ids.2");
    for (int i = 0; i < 3; i++) {
        if (w->target_layer_ids[i] >= PULSAR_N_LAYER || (i && w->target_layer_ids[i] <= w->target_layer_ids[i - 1])) {
            pulsar_die("dspark.target_layer_ids must be ascending target layer indices");
        }
    }

    /* The DRAFTER's own HC head mix, from block 2 -- the block whose hidden
     * feeds the head.  Required for the same reason as the main model's above:
     * a 0731 drafter missing it would otherwise collapse with the carried pre
     * and draft from a wrong head without failing. */
    if (g_pulsar_shape.hc_head_mix) {
        w->hc_head_fn    = required_tensor(m, "dspark.2.hc_head_fn.weight");
        w->hc_head_scale = required_tensor(m, "dspark.2.hc_head_scale.weight");
        w->hc_head_base  = required_tensor(m, "dspark.2.hc_head_base.weight");
    }

    dspark_weights_validate_layout(w);
}

































/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */





















































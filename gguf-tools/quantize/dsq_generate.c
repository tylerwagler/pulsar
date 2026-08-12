#include "dsq_internal.h"

/* =====
 * Tensor generation
 */

byte_buf f32_to_type(const float *src, int64_t n, ds4q_type type, int64_t ncols, const float *imat) {
    if (ncols <= 0 || n % ncols != 0) die("bad ncols for tensor conversion");
    byte_buf out = {0};
    if (type == DS4Q_TYPE_F32) {
        out.size = (size_t)n * sizeof(float);
        out.data = xmalloc(out.size);
        memcpy(out.data, src, out.size);
        return out;
    }
    if (type == DS4Q_TYPE_F16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_f16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (type == DS4Q_TYPE_BF16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_bf16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (type == DS4Q_TYPE_MXFP8_LT) {
        /* Quantize to the plain type-38 block layout, then swizzle it into the
         * pre-stored device layout (ds4q_pack_mxfp8_lt) in one pass. */
        if (ncols % 32 != 0) die("mxfp8_lt ncols is not divisible by 32");
        const int64_t nrows = n / ncols;
        const size_t tmp_size = (size_t)nrows * ds4q_row_size(DS4Q_TYPE_FP8_E4M3, ncols);
        uint8_t *tmp = xmalloc(tmp_size);
        ds4q_quantize_chunk(DS4Q_TYPE_FP8_E4M3, src, tmp, 0, nrows, ncols, NULL);
        out.size = ds4q_mxfp8_lt_bytes(nrows, ncols);
        out.data = xmalloc(out.size);
        ds4q_pack_mxfp8_lt(tmp, out.data, nrows, ncols);
        free(tmp);
        return out;
    }
    if (!ds4q_can_quantize(type)) die("unsupported quant target type");
    if (ncols % ds4q_block_size(type) != 0) die("ncols is not divisible by quant block size");
    const int64_t nrows = n / ncols;
    out.size = (size_t)nrows * ds4q_row_size(type, ncols);
    out.data = xmalloc(out.size);

    float *synthetic = NULL;
    const float *im_ptr = imat;
    if (!im_ptr && ds4q_requires_imatrix(type)) {
        synthetic = xcalloc((size_t)ncols, sizeof(float));
        for (int64_t r = 0; r < nrows; r++) {
            const float *row = src + (size_t)r * (size_t)ncols;
            for (int64_t c = 0; c < ncols; c++) synthetic[c] += row[c] * row[c];
        }
        im_ptr = synthetic;
    }
    size_t written = ds4q_quantize_chunk(type, src, out.data, 0, nrows, ncols, im_ptr);
    free(synthetic);
    if (written != out.size) die("ds4q_quantize_chunk wrote unexpected byte count");
    return out;
}

static byte_buf i64_to_i32(const st_value *src) {
    if (strcmp(src->dtype, "I64") != 0) die("expected I64 source for I32 tensor");
    const int64_t n = value_nelements(src);
    if (src->nbytes != (size_t)n * sizeof(int64_t)) die("bad I64 byte size");
    byte_buf out = { .size = (size_t)n * sizeof(int32_t), .data = xmalloc((size_t)n * sizeof(int32_t)) };
    int32_t *dst = (int32_t *)out.data;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = load_i64_le(src->data + (size_t)i * 8);
        if (v < INT32_MIN || v > INT32_MAX) die("I64 value out of I32 range");
        dst[i] = (int32_t)v;
    }
    return out;
}

size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    if (type == DS4Q_TYPE_CUTLASS_MXFP4) {
        /* Not a fixed-stride row format: per expert [K=ne[0], N=ne[1]] the
         * blob is E2M1 data + a padded swizzled SF tile, expert-major. */
        if (n_dims < 2) die("cutlass_mxfp4 tensor must be at least 2D");
        size_t nbytes = ds4q_cutlass_mxfp4_bytes(ne[1], ne[0]);
        for (int i = 2; i < n_dims; i++) nbytes *= (size_t)ne[i];
        return nbytes;
    }
    size_t nbytes = ds4q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) nbytes *= (size_t)ne[i];
    return nbytes;
}

static void check_reversed_shape(const char *gguf_name, const st_info *info, const tensor_meta *tmpl) {
    int nd = tensor_n_dims(tmpl);
    int skip = 0;
    while (info->n_dims - skip > nd && info->shape[skip] == 1) skip++;
    if (info->n_dims - skip != nd) {
        fprintf(stderr, "error: rank mismatch for %s\n", gguf_name);
        exit(1);
    }
    for (int i = 0; i < nd; i++) {
        if (tmpl->ne[i] != info->shape[info->n_dims - 1 - i]) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    }
}

static byte_buf generate_regular(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                 ds4q_type target, const imatrix_store *imatrix) {
    char *hf_name = hf_name_for_regular(gguf_name);
    tensor_entry *te = db_tensor(db, hf_name, NULL);
    check_reversed_shape(gguf_name, &te->info, tmpl);
    if (target == DS4Q_TYPE_I32) {
        st_value sv = db_read(db, hf_name);
        byte_buf b = i64_to_i32(&sv);
        st_value_free(&sv);
        free(hf_name);
        return b;
    }
    if (target == DS4Q_TYPE_CUTLASS_MXFP4)
        die("cutlass_mxfp4 is only supported for routed-expert tensors");
    /* Only the workhorse weights (attn q/kv/output, shared experts, output
     * head, main_proj) route through the engine's MXFP8_LT-aware FP8 resolver;
     * any other tensor stored as type 41 would decode to garbage on the plain
     * matmul path, so refuse it here (mirrors _WORKHORSE_BASES in
     * tools/mxfp8_prestore/repack_mxfp8_lt.py). */
    if (target == DS4Q_TYPE_MXFP8_LT && !is_mxfp8_lt_workhorse(gguf_name)) {
        fprintf(stderr, "error: mxfp8_lt is only valid for workhorse tensors "
                "(attn/shared/output/main_proj), not %s\n", gguf_name);
        exit(1);
    }
    if (!is_quantizable_target(target)) die("unsupported regular target type");
    int64_t n = 0;
    float *f32 = NULL;
    if (strcmp(te->info.dtype, "F8_E4M3") == 0) {
        if (!str_ends(hf_name, ".weight")) die("FP8 tensor without .weight suffix");
        char *scale_name = xstrdup(hf_name);
        strcpy(scale_name + strlen(scale_name) - strlen(".weight"), ".scale");
        if (!db_has(db, scale_name)) die("missing FP8 scale tensor");
        st_value w = db_read(db, hf_name);
        st_value s = db_read(db, scale_name);
        f32 = dequant_fp8_weight(&w, &s, &n);
        st_value_free(&w);
        st_value_free(&s);
        free(scale_name);
    } else {
        st_value w = db_read(db, hf_name);
        f32 = tensor_to_f32(&w, &n);
        st_value_free(&w);
    }
    const char *names[2] = { gguf_name, hf_name };
    const float *imat = imatrix_find(imatrix, names, 2, tmpl->ne[0], -1, 0);
    byte_buf b = f32_to_type(f32, n, target, tmpl->ne[0], imat);
    free(f32);
    free(hf_name);
    return b;
}

typedef struct {
    st_db *db;
    const char *gguf_name;
    const tensor_meta *tmpl;
    ds4q_type target;
    int n_experts;
    const imatrix_store *imatrix;
    expert_tensor expert;
    const char *wid;
    int64_t ncols;
    int64_t nrows;
    size_t per_expert;
    byte_buf *out;
    int next;
    int done;
    pthread_mutex_t lock;
} expert_job;

static void generate_one_expert(expert_job *j, int xid) {
    char prefix[256];
    if (j->expert.is_mtp)
        snprintf(prefix, sizeof(prefix), "mtp.%d.ffn.experts.%d.%s", j->expert.layer, xid, j->wid);
    else
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d.%s", j->expert.layer, xid, j->wid);
    char weight_name[320];
    char scale_name[320];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(scale_name, sizeof(scale_name), "%s.scale", prefix);
    st_value w = db_read(j->db, weight_name);
    st_value s = db_read(j->db, scale_name);
    if (w.n_dims != 2 || w.shape[0] != j->nrows || w.shape[1] * 2 != j->ncols) die("expert shape mismatch");
    if (j->target == DS4Q_TYPE_CUTLASS_MXFP4) {
        /* Byte-lossless: the QAT source already stores E2M1 nibbles + E8M0
         * block scales; copy them verbatim into the CUTLASS B layout. */
        if (strcmp(w.dtype, "I8") != 0 || strcmp(s.dtype, "F8_E8M0") != 0)
            die("cutlass_mxfp4 requires an E2M1(I8)+E8M0 QAT source");
        if (s.n_dims != 2 || s.shape[0] != j->nrows || s.shape[1] * 32 != j->ncols)
            die("expert scale shape mismatch");
        ds4q_pack_cutlass_mxfp4(w.data, s.data,
                                j->out->data + (size_t)xid * j->per_expert,
                                j->nrows, j->ncols);
        st_value_free(&w);
        st_value_free(&s);
        return;
    }
    int64_t n = 0;
    float *f32 = dequant_fp4_weight(&w, &s, &n);
    const char *names[3] = { j->gguf_name, weight_name, NULL };
    const float *imat = imatrix_find(j->imatrix, names, 2, j->ncols, xid, j->n_experts);
    byte_buf q = f32_to_type(f32, n, j->target, j->ncols, imat);
    if (q.size != j->per_expert) die("expert quantized size mismatch");
    memcpy(j->out->data + (size_t)xid * j->per_expert, q.data, q.size);
    free(q.data);
    free(f32);
    st_value_free(&w);
    st_value_free(&s);
}

static void *expert_worker(void *arg) {
    expert_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int xid = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (xid >= j->n_experts) break;
        generate_one_expert(j, xid);
        pthread_mutex_lock(&j->lock);
        int done = ++j->done;
        if (done % 32 == 0 || done == j->n_experts) {
            fprintf(stderr, "generate_expert_tensor: layer %d %s %d/%d experts\n",
                    j->expert.layer, j->wid, done, j->n_experts);
        }
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

static byte_buf generate_expert(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    expert_tensor e = parse_expert_tensor(gguf_name);
    if (!e.is_expert) die("not an expert tensor");
    if (!is_quantizable_target(target)) die("unsupported expert target type");
    /* IQ2_XXS_MMQ is a TENSOR-GLOBAL permutation of IQ2_XXS: one d plane across
     * every block of every expert, then one q plane. The per-expert workers
     * below therefore quantize the BASE type into their own slices, and the
     * permutation runs once after the join. Doing it per-expert (the shape the
     * MXFP8_LT branch in f32_to_type uses, which is per-row and so composes
     * fine) would emit per-expert planes: same bytes, wrong arrangement, loads
     * clean, reads garbage. */
    const bool want_iq2_mmq = (target == DS4Q_TYPE_IQ2_XXS_MMQ);
    if (want_iq2_mmq) target = DS4Q_TYPE_IQ2_XXS;
    const char *wid = expert_part_name(e.part);
    const int64_t ncols = tmpl->ne[0];
    const int64_t nrows = tmpl->ne[1];
    const size_t per_expert = target == DS4Q_TYPE_CUTLASS_MXFP4
        ? ds4q_cutlass_mxfp4_bytes(nrows, ncols)
        : (size_t)nrows * ds4q_row_size(target, ncols);
    byte_buf out = { .size = per_expert * (size_t)n_experts, .data = xmalloc(per_expert * (size_t)n_experts) };
    ds4q_quantize_init(target);
    int worker_count = n_threads > 0 ? n_threads : 8;
    if (worker_count < 1) worker_count = 1;
    if (worker_count > n_experts) worker_count = n_experts;
    fprintf(stderr, "generate_expert_tensor: layer %d %s using %d worker%s\n",
            e.layer, wid, worker_count, worker_count == 1 ? "" : "s");
    expert_job job = {
        .db = db, .gguf_name = gguf_name, .tmpl = tmpl, .target = target,
        .n_experts = n_experts, .imatrix = imatrix, .expert = e, .wid = wid,
        .ncols = ncols, .nrows = nrows, .per_expert = per_expert, .out = &out,
    };
    pthread_mutex_init(&job.lock, NULL);
    pthread_t *threads = xcalloc((size_t)worker_count, sizeof(threads[0]));
    for (int i = 1; i < worker_count; i++) pthread_create(&threads[i], NULL, expert_worker, &job);
    expert_worker(&job);
    for (int i = 1; i < worker_count; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&job.lock);
    free(threads);
    if (want_iq2_mmq) {
        /* Whole-tensor permutation, size-preserving. This is what
         * gguf-tools/repack_iq2_mmq.py used to do as a separate ~92 GB
         * copy-and-rewrite pass over the finished artifact. */
        const int64_t nblk = (int64_t)(out.size / 66);
        if ((size_t)nblk * 66 != out.size) die("iq2_xxs_mmq: tensor is not a whole number of 66 B blocks");
        uint8_t *packed = xmalloc(out.size);
        ds4q_pack_iq2_xxs_mmq((const uint8_t *)out.data, packed, nblk);
        free(out.data);
        out.data = packed;
    }
    return out;
}

byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    if (parse_expert_tensor(name).is_expert) {
        return generate_expert(db, name, tmpl, target, n_experts, n_threads, imatrix);
    }
    return generate_regular(db, name, tmpl, target, imatrix);
}


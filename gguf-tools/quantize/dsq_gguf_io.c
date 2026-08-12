#include "dsq_internal.h"

/* =====
 * Minimal GGUF reader/writer
 *
 * GGUF metadata is copied as raw KV records from the template.  Tensor infos
 * are rewritten with the new target types and offsets.  This keeps the tool C
 * only and independent from general-purpose GGUF libraries.
 */

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_checked_len_fp(fp, "GGUF string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            if (n > bytes_remaining_fp(fp, "GGUF array") / sz)
                die("GGUF array size exceeds the remaining file bytes");
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type");
    skip_bytes_fp(fp, sz);
}

static size_t gguf_string_size(const char *s) {
    return sizeof(uint64_t) + strlen(s);
}

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static bool is_imatrix_kv_key(const char *key) {
    return str_starts(key, "quantize.imatrix.");
}

static size_t extra_imatrix_kv_size(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    size_t n = 0;
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_SHA256) + 4 + gguf_string_size(im->sha256);
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES) + 4 + 8;
    if (im->dataset) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_DATASET) + 4 + gguf_string_size(im->dataset);
    if (im->chunks > 0) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS) + 4 + 8;
    return n;
}

static uint64_t extra_imatrix_kv_count(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    return 2 + (im->dataset ? 1 : 0) + (im->chunks > 0 ? 1 : 0);
}

static void write_imatrix_kvs(FILE *fp, const imatrix_store *im) {
    if (!imatrix_enabled(im)) return;
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_SHA256);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, im->sha256);

    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, (uint64_t)im->n_entries);

    if (im->dataset) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_DATASET);
        write_u32(fp, GGUF_TYPE_STRING);
        write_gguf_string(fp, im->dataset);
    }
    if (im->chunks > 0) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS);
        write_u32(fp, GGUF_TYPE_UINT64);
        write_u64(fp, (uint64_t)im->chunks);
    }
}

/* Raw bytes of the named KV records, concatenated in the order requested.
 * Fails if any is missing: a merged artifact without dspark.target_layer_ids
 * dies in dspark_weights_bind() at load, which is a long way from here. */
static uint8_t *gguf_take_kvs(const char *path, const char *const *want, int n_want,
                              size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0)
        die("bad GGUF template");
    (void)read_u32_le_fp(fp, "GGUF version");
    (void)read_u64_le_fp(fp, "GGUF tensor count");
    const uint64_t n_kv = read_u64_le_fp(fp, "GGUF KV count");

    byte_span *found = xcalloc((size_t)n_want, sizeof(found[0]));
    bool *seen = xcalloc((size_t)n_want, sizeof(seen[0]));
    const off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < n_kv; i++) {
        const off_t rec_start = ftello(fp);
        char *key = read_gguf_string_fp(fp);
        const uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        skip_gguf_value_fp(fp, type);
        const off_t rec_end = ftello(fp);
        if (rec_start < 0 || rec_end < rec_start) die("GGUF ftell failed");
        for (int k = 0; k < n_want; k++) {
            if (strcmp(key, want[k]) != 0) continue;
            if (seen[k]) die("duplicate KV key in dspark template");
            seen[k] = true;
            found[k] = (byte_span){ .start = (size_t)(rec_start - kv_start),
                                    .end = (size_t)(rec_end - kv_start) };
        }
        free(key);
    }
    const off_t kv_end = ftello(fp);
    if (kv_end < kv_start) die("GGUF ftell failed");
    for (int k = 0; k < n_want; k++) {
        if (!seen[k]) {
            fprintf(stderr, "error: %s: missing required kv %s\n", path, want[k]);
            exit(1);
        }
    }
    const size_t blob_len = (size_t)(kv_end - kv_start);
    uint8_t *blob = xmalloc(blob_len ? blob_len : 1);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (blob_len && fread(blob, 1, blob_len, fp) != blob_len) die("GGUF KV read failed");
    fclose(fp);

    size_t total = 0;
    for (int k = 0; k < n_want; k++) total += found[k].end - found[k].start;
    uint8_t *out = xmalloc(total ? total : 1);
    size_t pos = 0;
    for (int k = 0; k < n_want; k++) {
        const size_t n = found[k].end - found[k].start;
        memcpy(out + pos, blob + found[k].start, n);
        pos += n;
    }
    free(blob);
    free(found);
    free(seen);
    *out_len = total;
    return out;
}

/* KVs the engine needs to bind the drafter; mirrors MERGE_KEYS in the
 * merge_dspark_gguf.py this replaces. */
static const char *const DSPARK_MERGE_KEYS[] = {
    "deepseek_v4_dspark.embedding_length",
    "dspark.target_layer_ids.0",
    "dspark.target_layer_ids.1",
    "dspark.target_layer_ids.2",
};
#define N_DSPARK_MERGE_KEYS ((int)(sizeof(DSPARK_MERGE_KEYS) / sizeof(DSPARK_MERGE_KEYS[0])))

void gguf_append_dspark(gguf_file *g, const char *dspark_path) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (str_starts(g->tensors[i].name, "dspark."))
            die("main template already contains dspark.* tensors");

    gguf_file d = load_gguf_metadata(dspark_path);
    if (d.alignment != g->alignment) die("dspark template alignment differs from main");

    size_t kv_add_len = 0;
    uint8_t *kv_add = gguf_take_kvs(dspark_path, DSPARK_MERGE_KEYS,
                                    N_DSPARK_MERGE_KEYS, &kv_add_len);

    g->tensors = xrealloc(g->tensors, (size_t)(g->n_tensors + d.n_tensors) * sizeof(g->tensors[0]));
    for (uint64_t i = 0; i < d.n_tensors; i++) {
        g->tensors[g->n_tensors + i] = d.tensors[i];   /* steals d's name pointers */
        d.tensors[i].name = NULL;
    }
    g->n_tensors += d.n_tensors;

    g->kv_raw = xrealloc(g->kv_raw, g->kv_raw_len + kv_add_len);
    memcpy(g->kv_raw + g->kv_raw_len, kv_add, kv_add_len);
    g->kv_raw_len += kv_add_len;
    g->n_kv += N_DSPARK_MERGE_KEYS;
    free(kv_add);

    hmap_free(&g->tensor_map);
    char **keys = xmalloc((size_t)g->n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g->n_tensors; i++) keys[i] = g->tensors[i].name;
    hmap_build(&g->tensor_map, keys, (int)g->n_tensors);
    free(keys);

    free(d.path);
    free(d.kv_raw);
    free(d.tensors);
    hmap_free(&d.tensor_map);

    fprintf(stderr, "dspark: appended %" PRIu64 " tensors and %d kv from %s\n",
            d.n_tensors, N_DSPARK_MERGE_KEYS, dspark_path);
}

gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF template");
    }
    g.version = read_u32_le_fp(fp, "GGUF version");
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    byte_span *kv_keep = xcalloc((size_t)g.n_kv, sizeof(kv_keep[0]));
    uint64_t n_kv_keep = 0;

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        off_t rec_start = ftello(fp);
        if (rec_start < 0 || rec_start < kv_start) die("GGUF ftell failed");
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t n = read_u32_le_fp(fp, "GGUF expert count");
            if (n <= (uint32_t)INT_MAX) g.n_experts = (int)n;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT64) {
            uint64_t n = read_u64_le_fp(fp, "GGUF expert count");
            if (n <= (uint64_t)INT_MAX) g.n_experts = (int)n;
        } else {
            skip_gguf_value_fp(fp, type);
        }
        off_t rec_end = ftello(fp);
        if (rec_end < 0 || rec_end < rec_start) die("GGUF ftell failed");

        /*
         * Template GGUFs may already carry imatrix provenance from a previous
         * quantization.  Drop those keys and write the current run's keys later,
         * otherwise the output can contain duplicate GGUF metadata with stale
         * and new values.
         */
        if (!is_imatrix_kv_key(key)) {
            kv_keep[n_kv_keep++] = (byte_span){
                .start = (size_t)(rec_start - kv_start),
                .end = (size_t)(rec_end - kv_start),
            };
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");
    size_t kv_full_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_full = xmalloc(kv_full_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (kv_full_len && fread(kv_full, 1, kv_full_len, fp) != kv_full_len) die("GGUF KV read failed");

    for (uint64_t i = 0; i < n_kv_keep; i++) g.kv_raw_len += kv_keep[i].end - kv_keep[i].start;
    g.kv_raw = xmalloc(g.kv_raw_len);
    size_t kv_pos = 0;
    for (uint64_t i = 0; i < n_kv_keep; i++) {
        size_t n = kv_keep[i].end - kv_keep[i].start;
        memcpy(g.kv_raw + kv_pos, kv_full + kv_keep[i].start, n);
        kv_pos += n;
    }
    g.n_kv = n_kv_keep;
    free(kv_full);
    free(kv_keep);
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    char **keys = xmalloc((size_t)g.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) keys[i] = g.tensors[i].name;
    hmap_build(&g.tensor_map, keys, (int)g.n_tensors);
    free(keys);
    fclose(fp);
    return g;
}

byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name) {
    int idx = hmap_get(&g->tensor_map, name);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in GGUF: %s\n", name);
        exit(1);
    }
    const tensor_meta *t = &g->tensors[idx];
    byte_buf b = { .size = t->size, .data = xmalloc(t->size) };
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    if (fseeko(fp, (off_t)(g->data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek GGUF", path);
    if (b.size && fread(b.data, 1, b.size, fp) != b.size) die_errno("read GGUF tensor", path);
    fclose(fp);
    return b;
}

uint64_t fnv1a64_bytes(const uint8_t *data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

output_context build_output_context(const gguf_file *tmpl, const quant_policy *policy, const imatrix_store *im) {
    output_context out = {0};
    out.n_tensors = tmpl->n_tensors;
    out.n_kv_extra = extra_imatrix_kv_count(im);
    out.alignment = tmpl->alignment;
    out.tensors = xcalloc((size_t)out.n_tensors, sizeof(out.tensors[0]));
    size_t tensor_info = 0;
    size_t off = 0;
    for (uint64_t i = 0; i < out.n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        tensor_meta *dst = &out.tensors[i];
        *dst = *src;
        dst->name = src->name;
        ds4q_type type = policy_type(policy, src->name, src);
        if (type == DS4Q_TYPE_COUNT) type = src->type;
        if (type != DS4Q_TYPE_I32 && !is_quantizable_target(type)) die("unsupported planned tensor type");
        if (ds4q_can_quantize(type) && src->ne[0] % ds4q_block_size(type) != 0) die("ne[0] not divisible by block size");
        dst->type = type;
        dst->size = tensor_nbytes(type, src->ne, src->n_dims);
        dst->new_offset = off;
        off += ds4q_pad(dst->size, tmpl->alignment);
        tensor_info += gguf_string_size(dst->name) + 4 + (size_t)dst->n_dims * 8 + 4 + 8;
    }
    out.tensor_bytes = off;
    out.meta_size = 4 + 4 + 8 + 8 + tmpl->kv_raw_len + extra_imatrix_kv_size(im) + tensor_info;
    out.data_offset = ds4q_pad(out.meta_size, tmpl->alignment);
    return out;
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

void write_full_gguf(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                            const char *out_path, int n_experts, int n_threads,
                            const imatrix_store *imatrix, const reap_map *reap) {
    FILE *fp = fopen(out_path, "wb");
    if (!fp) die_errno("open output", out_path);
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, tmpl->version);
    write_u64(fp, tmpl->n_tensors);
    write_u64(fp, tmpl->n_kv + out_ctx->n_kv_extra);
    if (fwrite(tmpl->kv_raw, 1, tmpl->kv_raw_len, fp) != tmpl->kv_raw_len) die("write GGUF KV failed");
    write_imatrix_kvs(fp, imatrix);
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *t = &out_ctx->tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0) die("ftell failed");
    if ((size_t)pos > out_ctx->data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, out_ctx->data_offset - (size_t)pos);

    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s -> %s\n", i + 1, out_ctx->n_tensors, dst->name, ds4q_type_name(dst->type));
        byte_buf data = generate_tensor(db, dst->name, src, dst->type, n_experts, n_threads, imatrix, reap);
        size_t expected = dst->size;
        if (data.size != expected) {
            fprintf(stderr, "error: generated size mismatch for %s: got %zu expected %zu\n", dst->name, data.size, expected);
            exit(1);
        }
        if (fwrite(data.data, 1, data.size, fp) != data.size) die_errno("write tensor", out_path);
        size_t padded = ds4q_pad(data.size, out_ctx->alignment);
        write_padding(fp, padded - data.size);
        fprintf(stderr, "       generated %.2f MiB\n", (double)data.size / 1048576.0);
        free(data.data);
    }
    fclose(fp);
}

void print_plan(const gguf_file *tmpl, const output_context *out_ctx) {
    size_t tensor_bytes = 0;
    size_t changed = 0;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        tensor_bytes += out_ctx->tensors[i].size;
        const tensor_meta *src = &tmpl->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        if (src->type != dst->type) {
            changed++;
            printf("type_change: %s %s -> %s\n", dst->name, ds4q_type_name(src->type), ds4q_type_name(dst->type));
        }
    }
    printf("n_tensors: %" PRIu64 "\n", out_ctx->n_tensors);
    printf("meta_bytes: %zu\n", out_ctx->data_offset);
    printf("tensor_bytes_unpadded: %zu\n", tensor_bytes);
    printf("approx_file_bytes: %zu\n", out_ctx->data_offset + out_ctx->tensor_bytes);
    printf("type_changes: %zu\n", changed);
}


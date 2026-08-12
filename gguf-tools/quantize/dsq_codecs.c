#include "dsq_internal.h"

/* =====
 * DeepSeek V4 data conversion
 */

static float e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

static float bf16_to_f32_bits(uint16_t bits) {
    return ds4q_bf16_to_f32(bits);
}

int64_t value_nelements(const st_value *v) {
    int64_t n = 1;
    for (int i = 0; i < v->n_dims; i++) n *= v->shape[i];
    return n;
}

float *tensor_to_f32(const st_value *t, int64_t *n_out) {
    const int64_t n = value_nelements(t);
    float *out = xmalloc((size_t)n * sizeof(float));
    if (strcmp(t->dtype, "F32") == 0) {
        if (t->nbytes != (size_t)n * sizeof(float)) die("bad F32 byte size");
        memcpy(out, t->data, t->nbytes);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad BF16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32_bits(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad F16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = ds4q_f16_to_f32(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F8_E4M3") == 0) {
        if (t->nbytes != (size_t)n) die("bad F8_E4M3 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = e4m3fn_to_f32(t->data[i]);
    } else {
        fprintf(stderr, "error: cannot convert HF dtype directly: %s\n", t->dtype);
        exit(1);
    }
    if (n_out) *n_out = n;
    return out;
}

float *dequant_fp8_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    if (strcmp(w->dtype, "F8_E4M3") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP8 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP8 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t in_dim = w->shape[1];
    const int64_t block_out = 128;
    const int64_t block_in = 128;
    if (out_dim <= 0 || in_dim <= 0) die("FP8 dims must be positive");
    if (out_dim % block_out || in_dim % block_in) die("FP8 dims are not divisible by 128");
    const int64_t scale_rows = out_dim / block_out;
    const int64_t scale_cols = in_dim / block_in;
    if (scale->shape[0] != scale_rows || scale->shape[1] != scale_cols) die("FP8 scale shape mismatch");
    /* The loop bounds come from the header shape while data comes from
     * data_offsets; a shape declared over a short data range is heap OOB
     * (upstream ds4 a968c08). */
    if ((uint64_t)out_dim > SIZE_MAX / sizeof(float) / (uint64_t)in_dim)
        die("FP8 tensor shape overflows");
    if (w->nbytes < (uint64_t)out_dim * (uint64_t)in_dim)
        die("FP8 weight bytes shorter than the declared shape");
    if (scale->nbytes < (uint64_t)scale_rows * (uint64_t)scale_cols)
        die("FP8 scale bytes shorter than the declared shape");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t ob = 0; ob < scale_rows; ob++) {
        for (int64_t ib = 0; ib < scale_cols; ib++) {
            const float s = e8m0_to_f32(scale->data[(size_t)ob * (size_t)scale_cols + (size_t)ib]);
            for (int64_t r = 0; r < block_out; r++) {
                const int64_t row = ob * block_out + r;
                const size_t base = (size_t)row * (size_t)in_dim + (size_t)ib * (size_t)block_in;
                for (int64_t c = 0; c < block_in; c++) {
                    out[base + (size_t)c] = e4m3fn_to_f32(w->data[base + (size_t)c]) * s;
                }
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

float *dequant_fp4_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    static const float fp4_table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    if (strcmp(w->dtype, "I8") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP4 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP4 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t packed_in = w->shape[1];
    if (out_dim <= 0 || packed_in <= 0) die("FP4 dims must be positive");
    if (packed_in > INT64_MAX / 2) die("FP4 packed dim overflows");
    const int64_t in_dim = packed_in * 2;
    if (in_dim % 32) die("FP4 in_dim is not divisible by 32");
    const int64_t n_blocks = in_dim / 32;
    if (scale->shape[0] != out_dim || scale->shape[1] != n_blocks) die("FP4 scale shape mismatch");
    if ((uint64_t)out_dim > SIZE_MAX / sizeof(float) / (uint64_t)in_dim)
        die("FP4 tensor shape overflows");
    if (w->nbytes < (uint64_t)out_dim * (uint64_t)packed_in)
        die("FP4 weight bytes shorter than the declared shape");
    if (scale->nbytes < (uint64_t)out_dim * (uint64_t)n_blocks)
        die("FP4 scale bytes shorter than the declared shape");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t b = 0; b < n_blocks; b++) {
            const float s = e8m0_to_f32(scale->data[(size_t)r * (size_t)n_blocks + (size_t)b]);
            const size_t wbase = ((size_t)r * (size_t)n_blocks + (size_t)b) * 16;
            const size_t obase = (size_t)r * (size_t)in_dim + (size_t)b * 32;
            for (int64_t j = 0; j < 16; j++) {
                const uint8_t q = w->data[wbase + (size_t)j];
                out[obase + (size_t)(2*j + 0)] = fp4_table[q & 0x0f] * s;
                out[obase + (size_t)(2*j + 1)] = fp4_table[(q >> 4) & 0x0f] * s;
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

/* =====
 * Imatrix
 */

void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    /* Hashed before parsing so the recorded identity covers the file as it was
     * read, not just the parts this loader happens to understand. */
    im->sha256 = sha256_file_hex(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n_entries = read_i32_fp(fp, "imatrix entry count");
    if (n_entries < 1) die("imatrix has no entries");
    im->entries = xcalloc((size_t)n_entries, sizeof(im->entries[0]));
    im->n_entries = n_entries;
    for (int i = 0; i < n_entries; i++) {
        int32_t len = read_i32_fp(fp, "imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall = read_i32_fp(fp, "imatrix calls");
        int32_t nval = read_i32_fp(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count");
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    if (fgetc(fp) != EOF) {
        if (fseeko(fp, -1, SEEK_CUR) == 0) {
            im->chunks = read_i32_fp(fp, "imatrix chunks");
            int32_t dlen = read_i32_fp(fp, "imatrix dataset length");
            if (dlen > 0 && dlen < (1 << 20)) {
                im->dataset = xmalloc((size_t)dlen + 1);
                if (fread(im->dataset, 1, (size_t)dlen, fp) == (size_t)dlen) {
                    im->dataset[dlen] = '\0';
                } else {
                    free(im->dataset);
                    im->dataset = NULL;
                }
            }
        }
    }
    fclose(fp);
    char **keys = xmalloc((size_t)n_entries * sizeof(keys[0]));
    for (int i = 0; i < n_entries; i++) keys[i] = im->entries[i].name;
    hmap_build(&im->map, keys, n_entries);
    free(keys);
    fprintf(stderr, "loaded imatrix %s: %d entries%s%s\n",
            path, n_entries, im->dataset ? ", dataset=" : "", im->dataset ? im->dataset : "");
}

bool imatrix_enabled(const imatrix_store *im) {
    return im && im->n_entries > 0;
}

const float *imatrix_find(
        const imatrix_store *im,
        const char **names,
        int n_names,
        int64_t ncols,
        int expert_id,
        int n_experts) {
    if (!imatrix_enabled(im)) return NULL;
    char tmp[4096];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < n_names; i++) {
            if (!names[i]) continue;
            const char *candidate = names[i];
            if (expert_id >= 0 && pass < 2) {
                snprintf(tmp, sizeof(tmp), "%s.expert%s%d", names[i], pass == 0 ? "." : "_", expert_id);
                candidate = tmp;
            } else if (pass < 2) {
                continue;
            }
            int idx = hmap_get(&im->map, candidate);
            if (idx < 0) continue;
            const imatrix_entry *e = &im->entries[idx];
            if ((int64_t)e->n_values == ncols) return e->values;
            if (expert_id >= 0 && n_experts > 0 && (int64_t)e->n_values == ncols * (int64_t)n_experts) {
                return e->values + (size_t)expert_id * (size_t)ncols;
            }
            fprintf(stderr, "error: imatrix size mismatch for %s: got %d expected %" PRId64 "\n",
                    candidate, e->n_values, ncols);
            exit(1);
        }
    }
    if (im->strict) {
        fprintf(stderr, "error: missing imatrix entry for %s\n", names[0] ? names[0] : "(unnamed)");
        exit(1);
    }
    return NULL;
}

void imatrix_free(imatrix_store *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].values);
    }
    free(im->entries);
    free(im->file);
    free(im->sha256);
    free(im->dataset);
    hmap_free(&im->map);
    memset(im, 0, sizeof(*im));
}


/* =====
 * REAP survivor map (ds4-compact-v1) -- see reap_map in dsq_internal.h.
 */

static int *reap_int_array(const json_doc *d, int tok, int expect, const char *what) {
    if (tok < 0 || d->v[tok].type != JT_ARRAY) die("reap survivor map: bad array");
    const int n = d->v[tok].size;
    if (expect > 0 && n != expect) die("reap survivor map: wrong array length");
    int *out = xcalloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int i = tok + 1, k = 0;
    while (i < d->len && d->v[i].parent == tok) {
        out[k++] = (int)json_i64(d, i);
        i = json_skip(d, i);
    }
    if (k != n) die("reap survivor map: truncated array");
    (void)what;
    return out;
}

void reap_load(reap_map *rm, const char *path) {
    memset(rm, 0, sizeof(*rm));
    size_t len = 0;
    char *text = read_file(path, &len);
    json_doc d = json_parse_text(text, len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("reap survivor map is not a JSON object");

    const int t_layout = json_obj_get(&d, 0, "layout");
    if (t_layout < 0) die("reap survivor map: missing layout");
    rm->layout = json_strdup_tok(&d, t_layout);

    const int t_keep = json_obj_get(&d, 0, "keep_count");
    if (t_keep < 0 || d.v[t_keep].type != JT_ARRAY) die("reap survivor map: missing keep_count");
    const int n_layers = d.v[t_keep].size;
    rm->n_layers = n_layers;
    rm->keep = reap_int_array(&d, t_keep, n_layers, "keep_count");
    rm->policy = reap_int_array(&d, json_obj_get(&d, 0, "policy"), n_layers, "policy");

    rm->survivors = xcalloc((size_t)n_layers, sizeof(int *));
    const int t_surv = json_obj_get(&d, 0, "survivors");
    if (t_surv < 0 || d.v[t_surv].type != JT_OBJECT)
        die("reap survivor map: missing or malformed survivors");
    for (int i = t_surv + 1; i < d.len && d.v[i].parent == t_surv;) {
        char *lk = json_strdup_tok(&d, i);
        const int L = atoi(lk);
        free(lk);
        const int v = i + 1;
        if (L < 0 || L >= n_layers) die("reap survivor map: survivor layer out of range");
        rm->survivors[L] = reap_int_array(&d, v, rm->keep[L], "survivors entry");
        /* Ascending original ids: dense slot order must be deterministic, and
         * trim_reap.py asserts the same invariant on its side. */
        for (int j = 1; j < rm->keep[L]; j++)
            if (rm->survivors[L][j] <= rm->survivors[L][j - 1])
                die("reap survivor map: survivors are not strictly ascending");
        i = json_skip(&d, v);
    }
    int n_pruned = 0;
    for (int L = 0; L < n_layers; L++) {
        if (rm->policy[L] == 2 && !rm->survivors[L])
            die("reap survivor map: policy-2 layer has no survivor list");
        if (rm->policy[L] == 1 && rm->survivors[L])
            die("reap survivor map: policy-1 layer carries a survivor list");
        if (rm->policy[L] == 2) n_pruned++;
    }
    rm->enabled = true;
    json_free(&d);
    free(text);
    fprintf(stderr, "reap: %s, %d layers, %d pruned\n", rm->layout, n_layers, n_pruned);
}

void reap_free(reap_map *rm) {
    if (!rm || !rm->enabled) return;
    for (int L = 0; L < rm->n_layers; L++) free(rm->survivors[L]);
    free(rm->survivors);
    free(rm->keep);
    free(rm->policy);
    free(rm->layout);
    memset(rm, 0, sizeof(*rm));
}

int reap_keep(const reap_map *rm, int layer, int n_expert) {
    if (!rm || !rm->enabled || layer < 0 || layer >= rm->n_layers) return n_expert;
    if (rm->policy[layer] != 2) return n_expert;
    return rm->keep[layer];
}

int reap_src_expert(const reap_map *rm, int layer, int slot) {
    if (!rm || !rm->enabled || layer < 0 || layer >= rm->n_layers) return slot;
    if (rm->policy[layer] != 2 || !rm->survivors[layer]) return slot;
    if (slot < 0 || slot >= rm->keep[layer]) die("reap: dense slot out of range");
    return rm->survivors[layer][slot];
}

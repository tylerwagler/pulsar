/* safetensors.cpp -- the safetensors container.
 *
 * A safetensors checkpoint is a DIRECTORY of per-layer shards rather than one
 * file, and its metadata is JSON text.  This TU
 * bridges both differences so that nothing downstream of model_open changes:
 *
 *   * every shard is mmapped, and each tensor records its shard through the
 *     existing ext_map/ext_size fields -- the same mechanism --expert-overlay
 *     already uses, so tensor_data()/tensor_map_base() and the CUDA device-range
 *     cache (keyed on host_base) need no changes at all;
 *
 *   * the metadata is re-encoded ONCE into a typed value blob, so every
 *     existing kv consumer keeps reading values through cursor_at();
 *
 *   * the routed experts are re-stacked.  The file stores one tensor per expert
 *     because that is the source checkpoint's scheme, but a projection's 256
 *     per-expert tensors are contiguous and gap-free in these files (the writer
 *     asserts it), so the engine's single base + expert_bytes stride arithmetic
 *     sees exactly the byte layout it has always seen.  Contiguity is verified
 *     here rather than trusted.
 *
 * What it deliberately does NOT do is convert any payload: types 44/41/46/40 are
 * kernel read patterns and are mapped straight to their PULSAR_TENSOR_* ids.
 */
#include "pulsar_engine_internal.h"

#include "pulsar_json.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void st_die(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    pulsar_die(buf);
}

/** Allocations tied to the model's lifetime: the engine keeps the tensor
 * directory for as long as the model is open. */
static void *st_alloc(size_t n) {
    void *p = calloc(1, n ? n : 1);
    if (!p) st_die("safetensors: out of memory (%zu bytes)", n);
    return p;
}

static char *st_strdup(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)st_alloc(n + 1);
    memcpy(d, s, n);
    return d;
}

/* ------------------------------------------------------------------ shard --- */

typedef struct {
    char *name;         /**< tensor name as it appears in the file */
    char *dtype;        /**< "U8", "BF16", ... */
    uint64_t off0, off1;/**< data_offsets, relative to the shard's data buffer */
} st_tensor;

typedef struct {
    char *key;
    char *value;
} st_meta;

typedef struct {
    char *name;             /**< basename, for deterministic ordering */
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint64_t buf_off;       /**< 8 + header length: where data_offsets start */
    st_tensor *tensors;
    uint64_t n_tensors;
    st_meta *meta;
    uint64_t n_meta;
} st_shard;

static const char *st_meta_get(const st_shard *s, const char *key) {
    for (uint64_t i = 0; i < s->n_meta; i++) {
        if (strcmp(s->meta[i].key, key) == 0) return s->meta[i].value;
    }
    return NULL;
}

static const st_tensor *st_tensor_find(const st_shard *s, const char *name) {
    for (uint64_t i = 0; i < s->n_tensors; i++) {
        if (strcmp(s->tensors[i].name, name) == 0) return &s->tensors[i];
    }
    return NULL;
}

/* ------------------------------------------------------------ json helpers --- */

static bool st_obj_key(const char **p, const char *key) {
    /* Leave *p at the value of `key` inside the object that *p points at.
     * Rescans from the object start, so callers may look up any number of keys
     * in any order without depending on document order. */
    const char *q = *p;
    json_ws(&q);
    if (*q != '{') return false;
    q++;
    json_ws(&q);
    if (*q == '}') return false;
    for (;;) {
        char *k = NULL;
        if (!json_string(&q, &k)) return false;
        bool hit = strcmp(k, key) == 0;
        free(k);
        json_ws(&q);
        if (*q != ':') return false;
        q++;
        if (hit) { *p = q; return true; }
        if (!json_skip_value(&q)) return false;
        json_ws(&q);
        if (*q == '}') return false;
        if (*q != ',') return false;
        q++;
        json_ws(&q);
    }
}

static bool st_u64_array(const char **p, uint64_t *out, uint32_t *n_out, uint32_t max) {
    const char *q = *p;
    json_ws(&q);
    if (*q != '[') return false;
    q++;
    uint32_t n = 0;
    json_ws(&q);
    if (*q == ']') { q++; *p = q; *n_out = 0; return true; }
    for (;;) {
        double v = 0.0;
        if (!json_number(&q, &v)) return false;
        if (v < 0) return false;
        if (n < max) out[n] = (uint64_t)v;
        n++;
        json_ws(&q);
        if (*q == ']') { q++; break; }
        if (*q != ',') return false;
        q++;
        json_ws(&q);
    }
    if (n > max) return false;
    *n_out = n;
    *p = q;
    return true;
}

/* ------------------------------------------------------------- kv re-encode --- */

/** A growable byte blob holding the metadata values, typed by PULSAR_META_*.
 * cursor_at() reads metadata through this, so its layout must be that encoding
 * the existing consumers already decode. */
typedef struct {
    uint8_t *p;
    uint64_t len, cap;
} st_blob;

static void blob_need(st_blob *b, uint64_t n) {
    if (b->len + n <= b->cap) return;
    uint64_t cap = b->cap ? b->cap : 4096;
    while (cap < b->len + n) cap *= 2;
    uint8_t *np = (uint8_t *)realloc(b->p, cap);
    if (!np) st_die("safetensors: out of memory growing the metadata blob");
    b->p = np;
    b->cap = cap;
}

static void blob_raw(st_blob *b, const void *src, uint64_t n) {
    blob_need(b, n);
    memcpy(b->p + b->len, src, n);
    b->len += n;
}

static void blob_le(st_blob *b, uint64_t v, int width) {
    uint8_t t[8];
    for (int i = 0; i < width; i++) t[i] = (uint8_t)(v >> (8 * i));
    blob_raw(b, t, (uint64_t)width);
}

static void blob_str(st_blob *b, const char *s) {
    uint64_t n = strlen(s);
    blob_le(b, n, 8);
    blob_raw(b, s, n);
}

static uint32_t meta_code_for_type_name(const char *t) {
    if (!strcmp(t, "u8")) return PULSAR_META_UINT8;
    if (!strcmp(t, "i8")) return PULSAR_META_INT8;
    if (!strcmp(t, "u16")) return PULSAR_META_UINT16;
    if (!strcmp(t, "i16")) return PULSAR_META_INT16;
    if (!strcmp(t, "u32")) return PULSAR_META_UINT32;
    if (!strcmp(t, "i32")) return PULSAR_META_INT32;
    if (!strcmp(t, "f32")) return PULSAR_META_FLOAT32;
    if (!strcmp(t, "bool")) return PULSAR_META_BOOL;
    if (!strcmp(t, "string")) return PULSAR_META_STRING;
    if (!strcmp(t, "array")) return PULSAR_META_ARRAY;
    if (!strcmp(t, "u64")) return PULSAR_META_UINT64;
    if (!strcmp(t, "i64")) return PULSAR_META_INT64;
    if (!strcmp(t, "f64")) return PULSAR_META_FLOAT64;
    st_die("safetensors: unknown metadata type name '%s'", t);
    return 0;
}

static void blob_value(st_blob *b, const char **p, uint32_t type);

static void blob_array(st_blob *b, const char **p) {
    /* {"__array__":"u32","n":N,"v":[...]}.
     * st_obj_key() always rescans from the object's '{', so every lookup starts
     * from `obj` -- not from wherever the previous value left off. */
    const char *obj = *p;
    const char *q = obj;
    char *ename = NULL;
    if (!st_obj_key(&q, "__array__")) st_die("safetensors: metadata array has no __array__");
    if (!json_string(&q, &ename)) st_die("safetensors: metadata __array__ is not a string");
    uint32_t etype = meta_code_for_type_name(ename);
    free(ename);
    blob_le(b, etype, 4);
    q = obj;
    if (!st_obj_key(&q, "n")) st_die("safetensors: metadata array has no n");
    double dn = 0.0;
    if (!json_number(&q, &dn)) st_die("safetensors: metadata array n is not a number");
    uint64_t n = (uint64_t)dn;
    blob_le(b, n, 8);
    q = obj;
    if (!st_obj_key(&q, "v")) st_die("safetensors: metadata array has no v");
    json_ws(&q);
    if (*q != '[') st_die("safetensors: metadata array v is not an array");
    q++;
    json_ws(&q);
    if (*q == ']') st_die("safetensors: metadata array v is empty but n=%llu",
                          (unsigned long long)n);
    for (uint64_t i = 0; i < n; i++) {
        blob_value(b, &q, etype);
        json_ws(&q);
        if (i + 1 < n) {
            if (*q != ',') st_die("safetensors: metadata array v is short");
            q++;
        }
    }
    json_ws(&q);
    if (*q != ']') st_die("safetensors: metadata array v is long");
    q++;
    *p = q;
}

static void blob_value(st_blob *b, const char **p, uint32_t type) {
    const char *q = *p;
    json_ws(&q);
    switch (type) {
    case PULSAR_META_STRING: {
        char *s = NULL;
        if (!json_string(&q, &s)) st_die("safetensors: metadata string did not parse");
        blob_str(b, s);
        free(s);
        break;
    }
    case PULSAR_META_BOOL: {
        bool v = false;
        if (!json_bool(&q, &v)) st_die("safetensors: metadata bool did not parse");
        uint8_t byte = v ? 1 : 0;
        blob_raw(b, &byte, 1);
        break;
    }
    case PULSAR_META_ARRAY:
        blob_array(b, &q);
        break;
    case PULSAR_META_FLOAT32: {
        double v = 0.0;
        if (!json_number(&q, &v)) st_die("safetensors: metadata number did not parse");
        float f = (float)v;
        blob_raw(b, &f, 4);
        break;
    }
    case PULSAR_META_FLOAT64: {
        double v = 0.0;
        if (!json_number(&q, &v)) st_die("safetensors: metadata number did not parse");
        blob_raw(b, &v, 8);
        break;
    }
    case PULSAR_META_UINT8:
    case PULSAR_META_INT8:
    case PULSAR_META_UINT16:
    case PULSAR_META_INT16:
    case PULSAR_META_UINT32:
    case PULSAR_META_INT32:
    case PULSAR_META_UINT64: {
        double v = 0.0;
        if (!json_number(&q, &v)) st_die("safetensors: metadata integer did not parse");
        int w = (type == PULSAR_META_UINT8 || type == PULSAR_META_INT8) ? 1
              : (type == PULSAR_META_UINT16 || type == PULSAR_META_INT16) ? 2
              : (type == PULSAR_META_UINT32 || type == PULSAR_META_INT32) ? 4 : 8;
        blob_le(b, (uint64_t)(int64_t)v, w);
        break;
    }
    default:
        st_die("safetensors: metadata value type %u is not supported", type);
    }
    *p = q;
}

/* ---------------------------------------------------------------- parsing --- */


static void st_parse_header(st_shard *s) {
    if (s->size < 8) st_die("safetensors: %s is too small", s->name);
    uint64_t hlen = 0;
    memcpy(&hlen, s->map, 8);
    if (hlen == 0 || hlen > s->size - 8) {
        st_die("safetensors: %s has an implausible header length (%llu)",
               s->name, (unsigned long long)hlen);
    }
    s->buf_off = 8 + hlen;

    /* The header is padded with SPACES for alignment, not NUL-terminated, so
     * copy it once and terminate it: the scanner walks a C string. */
    char *text = (char *)st_alloc((size_t)hlen + 1);
    memcpy(text, s->map + 8, (size_t)hlen);

    const char *p = text;
    json_ws(&p);
    if (*p != '{') st_die("safetensors: %s header is not an object", s->name);
    p++;
    json_ws(&p);
    if (*p == '}') st_die("safetensors: %s header has no tensors", s->name);

    uint64_t cap_t = 4096, cap_m = 64;
    s->tensors = (st_tensor *)st_alloc(cap_t * sizeof(*s->tensors));
    s->meta = (st_meta *)st_alloc(cap_m * sizeof(*s->meta));

    for (;;) {
        char *key = NULL;
        if (!json_string(&p, &key)) st_die("safetensors: %s header key did not parse", s->name);
        json_ws(&p);
        if (*p != ':') st_die("safetensors: %s header: expected ':' after '%s'", s->name, key);
        p++;

        if (strcmp(key, "__metadata__") == 0) {
            const char *q = p;
            json_ws(&q);
            if (*q != '{') st_die("safetensors: %s __metadata__ is not an object", s->name);
            q++;
            json_ws(&q);
            while (*q != '}') {
                char *mk = NULL, *mv = NULL;
                if (!json_string(&q, &mk)) st_die("safetensors: %s __metadata__ key bad", s->name);
                json_ws(&q);
                if (*q != ':') st_die("safetensors: %s __metadata__ expected ':'", s->name);
                q++;
                if (!json_string(&q, &mv)) {
                    st_die("safetensors: %s __metadata__[%s] is not a string", s->name, mk);
                }
                if (s->n_meta == cap_m) {
                    cap_m *= 2;
                    s->meta = (st_meta *)realloc(s->meta, cap_m * sizeof(*s->meta));
                    if (!s->meta) st_die("safetensors: out of memory growing metadata");
                }
                s->meta[s->n_meta].key = mk;
                s->meta[s->n_meta].value = mv;
                s->n_meta++;
                json_ws(&q);
                if (*q == '}') break;
                if (*q != ',') st_die("safetensors: %s __metadata__ expected ','", s->name);
                q++;
                json_ws(&q);
            }
            p = q + 1;
            free(key);
        } else {
            if (s->n_tensors == cap_t) {
                cap_t *= 2;
                s->tensors = (st_tensor *)realloc(s->tensors, cap_t * sizeof(*s->tensors));
                if (!s->tensors) st_die("safetensors: out of memory growing the tensor list");
            }
            st_tensor *t = &s->tensors[s->n_tensors];
            memset(t, 0, sizeof(*t));
            t->name = key;

            const char *q = p;
            if (!st_obj_key(&q, "dtype") || !json_string(&q, &t->dtype)) {
                st_die("safetensors: %s: %s has no dtype", s->name, key);
            }
            q = p;
            uint64_t shape[PULSAR_MAX_DIMS];
            uint32_t nd = 0;
            if (!st_obj_key(&q, "shape") || !st_u64_array(&q, shape, &nd, PULSAR_MAX_DIMS)) {
                st_die("safetensors: %s: %s has a bad shape", s->name, key);
            }
            q = p;
            uint64_t off[2];
            uint32_t no = 0;
            if (!st_obj_key(&q, "data_offsets") || !st_u64_array(&q, off, &no, 2) || no != 2) {
                st_die("safetensors: %s: %s has bad data_offsets", s->name, key);
            }
            if (off[1] < off[0] || off[1] > s->size - s->buf_off) {
                st_die("safetensors: %s: %s data_offsets are outside the file", s->name, key);
            }
            t->off0 = off[0];
            t->off1 = off[1];
            s->n_tensors++;

            if (!json_skip_value(&p)) st_die("safetensors: %s: %s is malformed", s->name, key);
        }

        json_ws(&p);
        if (*p == '}') break;
        if (*p != ',') st_die("safetensors: %s header: expected ','", s->name);
        p++;
        json_ws(&p);
    }
}

static uint32_t st_layout_type(const char *layout, const char *dtype) {
    /* "native" is the one layout whose id depends on something other than its
     * name -- the payload is the file's own dtype, needing no interpretation --
     * which is why it is handled here rather than in the name table. */
    if (!strcmp(layout, "native")) {
        if (!strcmp(dtype, "F32")) return PULSAR_TENSOR_F32;
        if (!strcmp(dtype, "I32")) return PULSAR_TENSOR_I32;
        if (!strcmp(dtype, "BF16")) return PULSAR_TENSOR_BF16;
        st_die("safetensors: native tensor with unsupported dtype '%s'", dtype);
    }
    if (!strcmp(layout, "fp8_e4m3")) {
        /* Legacy interleaved MXFP8.  The runtime used to convert this at first
         * use into exactly the bytes mxfp8_lt already holds -- a second
         * resident copy of every such weight beside the mapping -- and that
         * path is gone, so a checkpoint declaring it is refused by NAME rather
         * than loaded into a layout nothing reads. */
        st_die("safetensors: '%s' is the legacy interleaved MXFP8 layout; this "
               "build serves the pre-stored 'mxfp8_lt' only", layout);
    }
    const int id = tensor_type_from_name(layout);
    if (id < 0) st_die("safetensors: unknown layout id '%s'", layout);
    return (uint32_t)id;
}

/** Exact on-disk bytes for a declared tensor.  The declaration must account for
 * the payload exactly; a mismatch is a wrong file and is refused rather than
 * loaded with a guessed size. */
static uint64_t st_bytes_for(uint32_t type, const uint64_t *dim, uint32_t nd) {
    uint64_t elements = 1;
    for (uint32_t i = 0; i < nd; i++) elements *= dim[i];
    switch (type) {
    case PULSAR_TENSOR_F32:
    case PULSAR_TENSOR_I32:
        return elements * 4;
    case PULSAR_TENSOR_BF16:
        return elements * 2;
    case PULSAR_TENSOR_IQ2_XXS_MMQ_K:
        if (nd < 2 || elements % 256) st_die("safetensors: IQ2 tensor has a bad shape");
        return elements / 256 * 66;
    case PULSAR_TENSOR_MXFP8_LT:
    case PULSAR_TENSOR_FP8_E4M3_SOA_K: {
        /* ne order: dim[0] = in (cols), dim[1] = out (rows) */
        if (nd < 2) st_die("safetensors: MXFP8 tensor has a bad shape");
        uint64_t cols = dim[0], rows = dim[1];
        if (cols % 32) st_die("safetensors: MXFP8 tensor has a non-multiple-of-32 input dim");
        uint64_t kb_pad = ((cols / 32) + 3) / 4 * 4;
        if (type == PULSAR_TENSOR_MXFP8_LT) {
            uint64_t rows_pad = (rows + 127) / 128 * 128;
            return rows * cols + rows_pad * kb_pad;
        }
        return rows * cols + rows * (cols / 32);
    }
    default:
        st_die("safetensors: no byte model for tensor type %u", type);
        return 0;
    }
}

typedef struct {
    pulsar_tensor *v;
    uint64_t n, cap;
} st_dir;

static pulsar_tensor *dir_add(st_dir *d) {
    if (d->n == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        pulsar_tensor *nv = (pulsar_tensor *)realloc(d->v, (size_t)d->cap * sizeof(*nv));
        if (!nv) st_die("safetensors: out of memory growing the tensor directory");
        d->v = nv;
    }
    pulsar_tensor *t = &d->v[d->n++];
    memset(t, 0, sizeof(*t));
    return t;
}

static void dir_push(st_dir *d, const char *gguf_name, uint32_t type,
                     const uint64_t *dim, uint32_t nd, const uint8_t *base,
                     uint64_t shard_size, uint64_t abs_offset, uint64_t bytes) {
    pulsar_tensor *t = dir_add(d);
    /* The name is the engine's CANONICAL tensor name (blk.N.* / dspark.N.*),
     * borrowed from the declaration text but kept alive for the model's
     * lifetime: every downstream lookup is by that name, so nothing after
     * model_open learns what the container called the tensor. */
    t->name.ptr = gguf_name;
    t->name.len = strlen(gguf_name);
    t->ndim = nd;
    for (uint32_t i = 0; i < nd; i++) t->dim[i] = dim[i];
    t->type = type;
    t->rel_offset = 0;
    t->abs_offset = abs_offset;
    t->elements = 1;
    for (uint32_t i = 0; i < nd; i++) t->elements *= dim[i];
    t->bytes = bytes;
    t->ext_map = base;
    t->ext_size = shard_size;
}

static void st_add_declared(st_dir *d, st_shard *s) {
    const char *md = st_meta_get(s, "pulsar.tensors");
    if (!md) return;
    const char *p = md;
    json_ws(&p);
    if (*p != '{') st_die("safetensors: %s pulsar.tensors is not an object", s->name);
    p++;
    json_ws(&p);
    if (*p == '}') return;
    for (;;) {
        char *hf = NULL;
        if (!json_string(&p, &hf)) st_die("safetensors: %s pulsar.tensors key bad", s->name);
        json_ws(&p);
        if (*p != ':') st_die("safetensors: %s pulsar.tensors expected ':'", s->name);
        p++;

        const char *q = p;
        char *layout = NULL, *gguf_name = NULL;
        if (!st_obj_key(&q, "layout") || !json_string(&q, &layout)) {
            st_die("safetensors: %s: %s has no layout", s->name, hf);
        }
        q = p;
        if (!st_obj_key(&q, "gguf_name") || !json_string(&q, &gguf_name)) {
            st_die("safetensors: %s: %s has no gguf_name", s->name, hf);
        }
        q = p;
        uint64_t dim[PULSAR_MAX_DIMS];
        uint32_t nd = 0;
        if (!st_obj_key(&q, "dims_ne") || !st_u64_array(&q, dim, &nd, PULSAR_MAX_DIMS)) {
            st_die("safetensors: %s: %s has no dims_ne", s->name, hf);
        }

        const st_tensor *ft = st_tensor_find(s, hf);
        if (!ft) st_die("safetensors: %s declares '%s' but the file does not contain it",
                        s->name, hf);
        uint32_t type = st_layout_type(layout, ft->dtype);
        uint64_t want = st_bytes_for(type, dim, nd);
        if (want != ft->off1 - ft->off0) {
            st_die("safetensors: %s: %s declares %llu bytes but holds %llu",
                   s->name, hf, (unsigned long long)want,
                   (unsigned long long)(ft->off1 - ft->off0));
        }
        dir_push(d, gguf_name, type, dim, nd, s->map, s->size,
                 s->buf_off + ft->off0, want);
        free(hf);
        free(layout);

        if (!json_skip_value(&p)) st_die("safetensors: %s pulsar.tensors entry bad", s->name);
        json_ws(&p);
        if (*p == '}') break;
        if (*p != ',') st_die("safetensors: %s pulsar.tensors expected ','", s->name);
        p++;
        json_ws(&p);
    }
}

static void st_add_expert_stacks(st_dir *d, st_shard *s) {
    const char *md = st_meta_get(s, "pulsar.experts");
    if (!md) return;
    const char *p = md;
    json_ws(&p);
    if (*p != '[') st_die("safetensors: %s pulsar.experts is not an array", s->name);
    p++;
    json_ws(&p);
    if (*p == ']') return;
    for (;;) {
        const char *q = p;
        char *gguf_name = NULL, *part = NULL, *layout = NULL;
        uint64_t dim2[2];
        uint32_t nd2 = 0;
        double dn = 0.0, db = 0.0;
        if (!st_obj_key(&q, "gguf_name") || !json_string(&q, &gguf_name)) st_die("safetensors: expert entry has no gguf_name");
        q = p;
        if (!st_obj_key(&q, "part") || !json_string(&q, &part)) st_die("safetensors: expert entry has no part");
        q = p;
        if (!st_obj_key(&q, "layout") || !json_string(&q, &layout)) st_die("safetensors: expert entry has no layout");
        q = p;
        if (!st_obj_key(&q, "n_experts") || !json_number(&q, &dn)) st_die("safetensors: expert entry has no n_experts");
        q = p;
        if (!st_obj_key(&q, "expert_bytes") || !json_number(&q, &db)) st_die("safetensors: expert entry has no expert_bytes");
        q = p;
        if (!st_obj_key(&q, "dims_per_expert_ne") || !st_u64_array(&q, dim2, &nd2, 2) || nd2 != 2) {
            st_die("safetensors: expert entry has no dims_per_expert_ne");
        }
        uint64_t n_experts = (uint64_t)dn;
        uint64_t expert_bytes = (uint64_t)db;

        char ns[16];
        int layer = -1;
        if (sscanf(gguf_name, "blk.%d.", &layer) == 1) {
            memcpy(ns, "layers", 7);
        } else if (sscanf(gguf_name, "dspark.%d.", &layer) == 1) {
            memcpy(ns, "mtp", 4);
        } else {
            st_die("safetensors: expert family '%s' is neither blk.N nor dspark.N", gguf_name);
        }

        /* The per-expert tensors must be contiguous in expert order: the
         * kernels address them as base + xid * expert_bytes from ONE pointer, so
         * a gap or a reordering here would silently read the wrong expert. */
        uint64_t first_off = 0;
        for (uint64_t e = 0; e < n_experts; e++) {
            char tn[160];
            snprintf(tn, sizeof(tn), "%s.%d.ffn.experts.%llu.%s.weight",
                     ns, layer, (unsigned long long)e, part);
            const st_tensor *ft = st_tensor_find(s, tn);
            if (!ft) st_die("safetensors: %s: missing %s", s->name, tn);
            if (ft->off1 - ft->off0 != expert_bytes) {
                st_die("safetensors: %s: %s holds %llu bytes, not expert_bytes=%llu",
                       s->name, tn, (unsigned long long)(ft->off1 - ft->off0),
                       (unsigned long long)expert_bytes);
            }
            if (e == 0) {
                first_off = ft->off0;
            } else if (ft->off0 != first_off + e * expert_bytes) {
                st_die("safetensors: %s: expert %llu of %s is not contiguous "
                       "(offset %llu, expected %llu) -- the engine reads experts "
                       "by stride from one base pointer",
                       s->name, (unsigned long long)e, gguf_name,
                       (unsigned long long)ft->off0,
                       (unsigned long long)(first_off + e * expert_bytes));
            }
        }

        uint64_t dim[PULSAR_MAX_DIMS];
        dim[0] = dim2[0];
        dim[1] = dim2[1];
        dim[2] = n_experts;
        uint32_t type = st_layout_type(layout, "U8");
        dir_push(d, gguf_name, type, dim, 3, s->map, s->size,
                 s->buf_off + first_off, n_experts * expert_bytes);
        free(part);
        free(layout);

        if (!json_skip_value(&p)) st_die("safetensors: %s pulsar.experts entry bad", s->name);
        json_ws(&p);
        if (*p == ']') break;
        if (*p != ',') st_die("safetensors: %s pulsar.experts expected ','", s->name);
        p++;
        json_ws(&p);
    }
}

static void st_build_kv(pulsar_model *m, const st_shard *s) {
    const char *md = st_meta_get(s, "pulsar.kv");
    if (!md) st_die("safetensors: no shard carries pulsar.kv; %s is not a checkpoint", s->name);
    const char *p = md;
    json_ws(&p);
    if (*p != '[') st_die("safetensors: pulsar.kv is not an array");
    p++;
    json_ws(&p);
    if (*p == ']') st_die("safetensors: pulsar.kv is empty");

    st_blob b = {0};
    uint64_t cap = 128;
    m->kv = (pulsar_kv *)st_alloc(cap * sizeof(m->kv[0]));
    for (;;) {
        const char *q = p;
        char *key = NULL, *tname = NULL;
        if (!st_obj_key(&q, "key") || !json_string(&q, &key)) st_die("safetensors: kv entry has no key");
        q = p;
        if (!st_obj_key(&q, "type") || !json_string(&q, &tname)) st_die("safetensors: kv entry has no type");
        q = p;
        if (!st_obj_key(&q, "value")) st_die("safetensors: kv entry has no value");
        uint32_t type = meta_code_for_type_name(tname);
        free(tname);

        if (m->n_kv == cap) {
            cap *= 2;
            pulsar_kv *nv = (pulsar_kv *)realloc(m->kv, (size_t)cap * sizeof(*nv));
            if (!nv) st_die("safetensors: out of memory growing the kv table");
            m->kv = nv;
        }
        pulsar_kv *kv = &m->kv[m->n_kv++];
        kv->key.ptr = key;
        kv->key.len = strlen(key);
        kv->type = type;
        kv->value_pos = b.len;
        blob_value(&b, &q, type);

        if (!json_skip_value(&p)) st_die("safetensors: pulsar.kv entry bad");
        json_ws(&p);
        if (*p == ']') break;
        if (*p != ',') st_die("safetensors: pulsar.kv expected ','");
        p++;
        json_ws(&p);
    }
    m->kv_base = b.p;
    m->kv_size = b.len;
}

static int st_cmp_name(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static bool st_is_shard(const char *n) {
    /* Let the compiler count the extension: the first version hard-coded 11 for
     * a 12-character suffix and rejected every shard. */
    static const char kExt[] = ".safetensors";
    const size_t e = sizeof(kExt) - 1;
    const size_t l = strlen(n);
    return l >= e && strcmp(n + l - e, kExt) == 0;
}

void safetensors_open(pulsar_model *m, const char *path, bool gpu_mapping) {
    /* Two shapes, one loader: a DIRECTORY of shards, or a single file that IS
     * the whole checkpoint.  The engine does not care which -- the tensor table
     * records each tensor's mapping either way. */
    struct stat pst;
    if (stat(path, &pst) == -1) pulsar_die_errno("cannot stat model", path);
    const bool single_file = S_ISREG(pst.st_mode);

    char **names = NULL;
    uint64_t n = 0, cap = 64;
    names = (char **)st_alloc(cap * sizeof(*names));
    if (single_file) {
        const char *slash = strrchr(path, '/');
        names[n++] = st_strdup(slash ? slash + 1 : path);
    } else {
        DIR *dir = opendir(path);
        if (!dir) pulsar_die_errno("cannot open model directory", path);
        for (struct dirent *de = readdir(dir); de; de = readdir(dir)) {
            if (!st_is_shard(de->d_name)) continue;
            if (n == cap) {
                cap *= 2;
                names = (char **)realloc(names, (size_t)cap * sizeof(*names));
                if (!names) st_die("safetensors: out of memory listing shards");
            }
            names[n++] = st_strdup(de->d_name);
        }
        closedir(dir);
    }
    if (n == 0) st_die("safetensors: no *.safetensors shards in %s", path);
    qsort(names, (size_t)n, sizeof(*names), st_cmp_name);

    m->n_shards = n;
    m->shard_fd = (int *)st_alloc((size_t)n * sizeof(*m->shard_fd));
    m->shard_map = (const uint8_t **)st_alloc((size_t)n * sizeof(*m->shard_map));
    m->shard_size = (uint64_t *)st_alloc((size_t)n * sizeof(*m->shard_size));
    st_shard *shards = (st_shard *)st_alloc((size_t)n * sizeof(*shards));

    const int mmap_flags = gpu_mapping ? MAP_SHARED : MAP_PRIVATE;
    for (uint64_t i = 0; i < n; i++) {
        char full[4096];
        const int w = single_file ? snprintf(full, sizeof(full), "%s", path)
                                  : snprintf(full, sizeof(full), "%s/%s", path, names[i]);
        if (w <= 0 || (size_t)w >= sizeof(full)) st_die("safetensors: shard path is too long");
        st_shard *s = &shards[i];
        s->name = names[i];
        s->fd = open(full, O_RDONLY);
        if (s->fd == -1) pulsar_die_errno("cannot open shard", full);
        struct stat st;
        if (fstat(s->fd, &st) == -1) pulsar_die_errno("cannot stat shard", full);
        if (st.st_size < 32) st_die("safetensors: %s is too small", full);
        void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, s->fd, 0);
        if (map == MAP_FAILED) pulsar_die_errno("cannot mmap shard", full);
        s->map = (const uint8_t *)map;
        s->size = (uint64_t)st.st_size;
        st_parse_header(s);
        m->shard_fd[i] = s->fd;
        m->shard_map[i] = s->map;
        m->shard_size[i] = s->size;
        /* PULSAR_VERIFY_RANGES: name each mapping, so the device-range hashes
         * printed by the runtime can be attributed to a file. */
        if (getenv("PULSAR_VERIFY_RANGES")) {
            printf("SHARD %llu %p %s\n", (unsigned long long)i, (const void *)s->map, full);
            fflush(stdout);
        }
    }

    /* The primary shard is the one carrying the full KV block; it is also the
     * only one that needs the architecture keys. */
    const st_shard *primary = NULL;
    for (uint64_t i = 0; i < n; i++) {
        if (st_meta_get(&shards[i], "pulsar.kv")) { primary = &shards[i]; break; }
    }
    if (!primary) st_die("safetensors: no shard in %s carries pulsar.kv", path);
    st_build_kv(m, primary);

    st_dir d = {0};
    for (uint64_t i = 0; i < n; i++) {
        st_add_declared(&d, &shards[i]);
        st_add_expert_stacks(&d, &shards[i]);
    }
    if (d.n == 0) st_die("safetensors: %s declares no tensors", path);
    m->tensors = d.v;
    m->n_tensors = d.n;
    m->version = 0;
    m->alignment = 32;
    m->tensor_data_pos = 0;
    /* A harmless default for the few helpers that take the model mapping when a
     * tensor has no ext_map; every tensor built above sets one. */
    m->map = m->shard_map[0];
    m->size = m->shard_size[0];
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = m->tensors[i].bytes;
        }
    }
    m->mapped_bytes = 0;
    for (uint64_t i = 0; i < m->n_shards; i++) m->mapped_bytes += m->shard_size[i];
    fprintf(stderr, "pulsar: safetensors model: %llu shards, %llu tensors, "
                    "%.2f GiB mapped, %llu metadata keys\n",
            (unsigned long long)m->n_shards, (unsigned long long)m->n_tensors,
            (double)m->mapped_bytes / 1073741824.0, (unsigned long long)m->n_kv);

    /* PULSAR_VERIFY_TENSORS=<name prefix>: hash the bytes the KERNELS will read
     * (tensor_map_base(m,t) + t->abs_offset) for every tensor under that
     * prefix.  Gate 3 and the reference reader both validated the FILE; this is
     * the only check of what the engine actually resolves, which is where a
     * multi-mapping model can go wrong while the file stays perfect.
     * An EMPTY prefix means every tensor, which is what diffs this container's
     * whole inventory against the artifact it replaced. */
    const char *vfypfx = getenv("PULSAR_VERIFY_TENSORS");
    if (vfypfx) {
        const size_t plen = strlen(vfypfx);
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const pulsar_tensor *t = &m->tensors[i];
            if (t->name.len < plen || memcmp(t->name.ptr, vfypfx, plen) != 0) continue;
            const uint8_t *base = (const uint8_t *)tensor_map_base(m, t);
            const uint8_t *p = base + t->abs_offset;
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint64_t b = 0; b < t->bytes; b++) {
                h ^= (uint64_t)p[b];
                h *= 0x100000001b3ull;
            }
            printf("VERIFY %.*s %016llx %llu\n",
                   (int)t->name.len, t->name.ptr, (unsigned long long)h,
                   (unsigned long long)t->bytes);
        }
        fflush(stdout);
    }
}

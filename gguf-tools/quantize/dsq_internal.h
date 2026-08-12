/*
 * dsq_internal.h — shared internal header for the deepseek4-quantize TUs.
 * Carries the shared prologue, cross-TU types, and declarations for the
 * symbols referenced across module boundaries. Edit freely (there is no
 * generator). Split from deepseek4-quantize.c; original doc below.
 */
#ifndef DSQ_INTERNAL_H
#define DSQ_INTERNAL_H

/*
 * DeepSeek V4 Flash/Pro HF -> GGUF quantizer.
 *
 * This is a plain C, model-specific version of the DS4 quantization pipeline.
 * It deliberately keeps only the pieces needed by the DeepSeek V4 Flash and
 * Pro GGUF recipes used by this repository:
 *
 * - safetensors index/header loading;
 * - FP8 E4M3 + E8M0 dequantization for dense tensors;
 * - packed FP4 + E8M0 dequantization for routed experts;
 * - local Q2_K and IQ2_XXS quantization;
 * - GGUF metadata/tensor-order reuse from an existing template GGUF.
 *
 * The optional imatrix is the legacy llama.cpp binary .dat format emitted by
 * ds4's collector.  DS4 stores one packed vector per routed tensor, laid out as
 * n_experts consecutive per-expert importance vectors.  When no external
 * imatrix is supplied and IQ2_XXS requires one, this tool falls back to the
 * same synthetic weight-energy heuristic used by the old generator:
 * each column importance is sum(row[column]^2) over the dequantized weight.
 */

#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(_WIN32)
#error "deepseek4-quantize.c currently targets POSIX systems"
#endif

/* Content hash, not a path: a path records where someone's disk was, a hash
 * records WHICH imatrix it was. Replaced "quantize.imatrix.file" on 2026-08-12;
 * nothing outside this tool ever read that key. */
#define DS4_KV_QUANTIZE_IMATRIX_SHA256    "quantize.imatrix.sha256"
#define DS4_KV_QUANTIZE_IMATRIX_DATASET   "quantize.imatrix.dataset"
#define DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES "quantize.imatrix.entries_count"
#define DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS  "quantize.imatrix.chunks_count"
#define DS4_GGUF_DEFAULT_ALIGNMENT 32

/* ===== Shared types (moved verbatim from the original sections) ===== */

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

typedef enum {
    JT_OBJECT,
    JT_ARRAY,
    JT_STRING,
    JT_PRIMITIVE,
} json_type;

typedef struct {
    json_type type;
    int start;
    int end;
    int parent;
    int size;
} json_tok;

typedef struct {
    json_tok *v;
    int len;
    int cap;
    const char *js;
    int js_len;
} json_doc;

typedef struct {
    char *key;
    int value;
} hslot;

typedef struct {
    hslot *slots;
    int cap;
} hmap;

#define MAX_DIMS 8

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint64_t begin;
    uint64_t end;
} st_info;

typedef struct {
    char *name;
    char *file;
} weight_map_entry;

typedef struct {
    char *name;
    st_info info;
} tensor_entry;

typedef struct {
    char *file;
    char *path;
    uint64_t data_base;
    tensor_entry *tensors;
    int n_tensors;
    int cap_tensors;
    hmap tensor_map;
    FILE *fp;
    pthread_mutex_t lock;
    bool loaded;
} shard;

typedef struct {
    char *hf_dir;
    weight_map_entry *weights;
    int n_weights;
    hmap weight_map;
    shard *shards;
    int n_shards;
    int cap_shards;
    pthread_mutex_t lock;
} st_db;

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint8_t *data;
    size_t nbytes;
} st_value;

typedef struct {
    char *name;
    float *values;
    int n_values;
} imatrix_entry;

typedef struct {
    char *file;
    char *sha256;      /* hex content hash; what identifies the imatrix */
    char *dataset;
    imatrix_entry *entries;
    int n_entries;
    hmap map;
    int chunks;
    bool strict;
} imatrix_store;

/* REAP survivor map (ds4-compact-v1). Lets the quantizer emit an already-pruned
 * artifact instead of writing a full-256 intermediate for a downstream tool to
 * re-slice. policy 1 = layer keeps every expert (hash-routed via tid2eid, which
 * indexes experts by id and so structurally cannot be pruned); policy 2 = layer
 * is trimmed to keep[L] survivors, listed in ascending ORIGINAL expert id.
 * Router and bias stay padded to n_expert -- pruned bias slots get -1e30 so
 * they can never win top-k. See gguf-tools/reap/trim_reap.py, which this
 * replaces, and validate_reap_metadata() in the engine's weights.cpp. */
typedef struct {
    bool  enabled;
    char *layout;
    int   n_layers;
    int  *keep;        /* [n_layers] survivor count */
    int  *policy;      /* [n_layers] 1 = untouched, 2 = pruned */
    int **survivors;   /* [n_layers][keep[L]] original expert ids; NULL if policy 1 */
} reap_map;

void reap_load(reap_map *rm, const char *path);
void reap_free(reap_map *rm);
/* Survivor count for a layer (n_expert when the layer is untouched or no map). */
int reap_keep(const reap_map *rm, int layer, int n_expert);
/* Original expert id backing dense slot `slot` (identity when untouched). */
int reap_src_expert(const reap_map *rm, int layer, int slot);

typedef enum { EXP_NONE, EXP_W1, EXP_W2, EXP_W3 } expert_part;

typedef struct {
    bool is_expert;
    bool is_mtp;
    int layer;
    expert_part part;
} expert_tensor;

typedef struct {
    const char *gguf;
    const char *hf;
} name_map;

typedef struct {
    char *prefix;
    ds4q_type type;
} type_override;

typedef struct {
    ds4q_type routed_w1, routed_w2, routed_w3;
    ds4q_type attention_proj, attention, shared, embedding, output, dense;
    type_override *overrides;
    int n_overrides;
} quant_policy;

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
} tensor_meta;

typedef struct {
    uint8_t *data;
    size_t size;
} byte_buf;

typedef struct {
    size_t start;
    size_t end;
} byte_span;

typedef struct {
    char *path;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint8_t *kv_raw;
    size_t kv_raw_len;
    size_t alignment;
    int n_experts;
    size_t data_offset;
    tensor_meta *tensors;
    hmap tensor_map;
} gguf_file;

typedef struct {
    tensor_meta *tensors;
    uint64_t n_tensors;
    uint64_t n_kv_extra;
    size_t meta_size;
    size_t data_offset;
    size_t tensor_bytes;
    size_t alignment;
} output_context;

/* ===== Cross-TU declarations ===== */

/* dsq_sha256.c */
/* Hex SHA-256 of a whole file, streamed. Caller frees. Self-tests on first use. */
char *sha256_file_hex(const char *path);

/* dsq_util.c */
void die(const char *msg);
void die_errno(const char *what, const char *path);
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t sz);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);
char *path_join(const char *a, const char *b);
bool str_starts(const char *s, const char *prefix);
bool str_ends(const char *s, const char *suffix);
char *read_file(const char *path, size_t *len_out);
uint64_t read_u64_le_fp(FILE *fp, const char *what);
uint64_t bytes_remaining_fp(FILE *fp, const char *what);
uint64_t read_checked_len_fp(FILE *fp, const char *what);
uint32_t read_u32_le_fp(FILE *fp, const char *what);
int32_t read_i32_fp(FILE *fp, const char *what);
uint16_t load_u16_le(const uint8_t *p);
int64_t load_i64_le(const uint8_t *p);

/* dsq_json.c */
json_doc json_parse_text(const char *js, size_t len);
void json_free(json_doc *d);
bool json_tok_eq(const json_doc *d, int tok, const char *s);
char *json_strdup_tok(const json_doc *d, int tok);
int json_skip(const json_doc *d, int tok);
int json_obj_get(const json_doc *d, int obj, const char *key);
int64_t json_i64(const json_doc *d, int tok);
void hmap_build(hmap *m, char **keys, int n);
int hmap_get(const hmap *m, const char *key);
void hmap_free(hmap *m);

/* dsq_safetensors.c */
void st_value_free(st_value *v);
void db_open(st_db *db, const char *hf_dir);
void db_close(st_db *db);
bool db_has(const st_db *db, const char *name);
tensor_entry *db_tensor(st_db *db, const char *name, shard **shard_out);
st_value db_read(st_db *db, const char *name);

/* dsq_codecs.c */
int64_t value_nelements(const st_value *v);
float *tensor_to_f32(const st_value *t, int64_t *n_out);
float *dequant_fp8_weight(const st_value *w, const st_value *scale, int64_t *n_out);
float *dequant_fp4_weight(const st_value *w, const st_value *scale, int64_t *n_out);
void imatrix_load(imatrix_store *im, const char *path, bool strict);
bool imatrix_enabled(const imatrix_store *im);
const float *imatrix_find(const imatrix_store *im, const char **names, int n_names,
                          int64_t ncols, int expert_id, int n_experts);
void imatrix_free(imatrix_store *im);

/* dsq_names.c */
expert_tensor parse_expert_tensor(const char *name);
const char *expert_part_name(expert_part p);
char *hf_name_for_regular(const char *gguf_name);
int tensor_n_dims(const tensor_meta *t);
ds4q_type policy_type(const quant_policy *p, const char *name, const tensor_meta *tmpl);
ds4q_type parse_type(const char *raw);
bool is_quantizable_target(ds4q_type type);
bool is_mxfp8_lt_workhorse(const char *name);
void policy_load_format_map(quant_policy *p, const char *path);

/* dsq_generate.c */
byte_buf f32_to_type(const float *src, int64_t n, ds4q_type type, int64_t ncols, const float *imat);
size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims);
byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl,
                         ds4q_type target, int n_experts, int n_threads,
                         const imatrix_store *imatrix, const reap_map *reap);

/* dsq_gguf_io.c */
gguf_file load_gguf_metadata(const char *path);
/* Append the drafter template's tensors + its required KVs to the main
 * template, so one quantizer pass emits the merged artifact directly. */
void gguf_append_dspark(gguf_file *g, const char *dspark_path);
byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name);
uint64_t fnv1a64_bytes(const uint8_t *data, size_t n);
output_context build_output_context(const gguf_file *tmpl, const quant_policy *policy,
                                    const imatrix_store *im);
void write_full_gguf(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                     const char *out_path, int n_experts, int n_threads,
                     const imatrix_store *imatrix, const reap_map *reap);
void print_plan(const gguf_file *tmpl, const output_context *out_ctx);

/* dsq_probe.c */
void run_mse_probe(st_db *db, const gguf_file *tmpl, const imatrix_store *im,
                   int n_experts, int n_threads, int sample, const char *out_path);

#endif /* DSQ_INTERNAL_H */

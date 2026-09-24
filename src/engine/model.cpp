#include "pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"



/* Pulsar's tensor LAYOUT vocabulary, in one table so that a name -> id and an
 * id -> name answer cannot disagree.  The byte model is deliberately NOT here:
 * a container must account for its own declaration's bytes exactly, and does
 * that where the declaration is read (st_bytes_for for the fixed-rate layouts,
 * cutlass_mxfp4_expert_layout for the block-scaled one).  The names are the
 * strings a checkpoint declares, and the native layouts carry their dtype's
 * name because that is what a histogram of a checkpoint should print. */
static const struct { const char *name; uint32_t id; } pulsar_layout_names[] = {
    {"f32",            PULSAR_TENSOR_F32},
    {"i32",            PULSAR_TENSOR_I32},
    {"bf16",           PULSAR_TENSOR_BF16},
    {"mxfp8_lt",       PULSAR_TENSOR_MXFP8_LT},
    {"iq2_xxs_mmq_k",  PULSAR_TENSOR_IQ2_XXS_MMQ_K},
    {"cutlass_mxfp4",  PULSAR_TENSOR_CUTLASS_MXFP4},
    {"fp8_e4m3_soa_k", PULSAR_TENSOR_FP8_E4M3_SOA_K},
    {"exl3m_k2",       PULSAR_TENSOR_EXL3M_K2},
    {"exl3m_k2h",      PULSAR_TENSOR_EXL3M_K2H},
    {"exl3m_k3",       PULSAR_TENSOR_EXL3M_K3},
};



const char *tensor_type_name(uint32_t type) {
    for (size_t i = 0; i < sizeof(pulsar_layout_names) / sizeof(pulsar_layout_names[0]); i++) {
        if (pulsar_layout_names[i].id == type) return pulsar_layout_names[i].name;
    }
    return "unknown";
}



int tensor_type_from_name(const char *name) {
    for (size_t i = 0; i < sizeof(pulsar_layout_names) / sizeof(pulsar_layout_names[0]); i++) {
        if (!strcmp(pulsar_layout_names[i].name, name)) return (int)pulsar_layout_names[i].id;
    }
    return -1;
}



/* PULSAR_TENSOR_CUTLASS_MXFP4 is expert-major: one
 * ColumnMajor E2M1 data blob (N*K/2 bytes) followed by one swizzled E8M0
 * scale-factor blob per expert. The CUTLASS Sm1xxBlkScaledConfig tile atom
 * pads both N and K up to a multiple of 128 for the SF blob, so this is
 * NOT a uniform per-element byte rate, so it has no entry in the
 * layout-name table's byte model.
 * (For DS4's actual rich-expert shapes both dims are already multiples of
 * 128, so sf_bytes == n*k/32 exactly with no padding -- but callers must
 * not assume that holds for arbitrary shapes.) */
void cutlass_mxfp4_expert_layout(uint64_t k, uint64_t n,
                                  uint64_t *data_bytes, uint64_t *sf_bytes,
                                  uint64_t *stride) {
    uint64_t k_pad = (k + 127) / 128 * 128;
    uint64_t n_pad = (n + 127) / 128 * 128;
    *data_bytes = n * k / 2;
    *sf_bytes = (n_pad / 32) * k_pad;
    *stride = *data_bytes + *sf_bytes;
}



pulsar_cursor cursor_at(const pulsar_model *m, uint64_t pos) {
    /* A safetensors model's metadata values live in a synthesized buffer, not
     * in the mapping; a container that keeps its values in the file leaves
     * kv_base NULL and is unaffected. */
    pulsar_cursor c = {
        .base = m->kv_base ? m->kv_base : m->map,
        .size = m->kv_base ? m->kv_size : m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}



static pulsar_kv *model_find_kv(const pulsar_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (pulsar_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}



static bool model_get_string(const pulsar_model *m, const char *key, pulsar_str *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != PULSAR_META_STRING) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}



bool model_get_u32(const pulsar_model *m, const char *key, uint32_t *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != PULSAR_META_UINT32) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}



static bool model_get_u64(const pulsar_model *m, const char *key, uint64_t *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != PULSAR_META_UINT64) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u64(&c, out);
}



bool model_get_u64_compat(const pulsar_model *m, const char *key, uint64_t *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == PULSAR_META_UINT64) {
        return cursor_u64(&c, out);
    }
    if (kv->type == PULSAR_META_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}



bool model_get_f32_compat(const pulsar_model *m, const char *key, float *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == PULSAR_META_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == PULSAR_META_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == PULSAR_META_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == PULSAR_META_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}



bool model_get_bool(const pulsar_model *m, const char *key, bool *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != PULSAR_META_BOOL) return false;
    pulsar_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}



bool model_get_array(const pulsar_model *m, const char *key, pulsar_array_ref *out) {
    pulsar_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != PULSAR_META_ARRAY) return false;

    pulsar_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}



void model_close(pulsar_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}






/* Open a model.  A pulsar model is a safetensors checkpoint: a DIRECTORY of
 * per-layer shards, or one file carrying the same declaration.  Everything the
 * engine reads -- metadata values, tensor layouts, offsets, shapes -- is
 * declared inside it. */
void model_open(pulsar_model *m, const char *path, bool gpu_mapping) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    /* GGUF is not a container this engine has any more.  The magic is checked
     * ONLY so that the refusal names the file's real format and says what to do
     * about it, instead of surfacing as a JSON parse error from the reader
     * below.  Nothing here reads a GGUF. */
    struct stat pst;
    if (stat(path, &pst) == -1) pulsar_die_errno("cannot stat model", path);
    if (!S_ISDIR(pst.st_mode)) {
        if (pst.st_size < (off_t)sizeof(uint32_t)) pulsar_die("model file is too small");
        int probe = open(path, O_RDONLY);
        if (probe == -1) pulsar_die_errno("cannot open model", path);
        uint32_t magic = 0;
        const ssize_t got = read(probe, &magic, sizeof(magic));
        (void)close(probe);
        if (got == (ssize_t)sizeof(magic) && magic == PULSAR_GGUF_MAGIC) {
            pulsar_die("this is a GGUF file, and pulsar reads safetensors "
                       "checkpoints only -- either a directory of per-layer "
                       "shards or a single file carrying the same declaration. "
                       "The GGUF container was retired in favour of the "
                       "declared-layout safetensors artifact.");
        }
    }
    safetensors_open(m, path, gpu_mapping);
}



static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}



void model_summary(const pulsar_model *m) {
    pulsar_str name = {0, 0};
    pulsar_str arch = {0, 0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u32(m, "deepseek4.block_count", &layers);
    model_get_u64(m, "deepseek4.context_length", &ctx_train);
    model_get_u32(m, "deepseek4.attention.head_count", &n_head);
    model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv);
    model_get_u32(m, "deepseek4.attention.key_length", &head_dim);
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads);
    model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim);
    model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k);
    model_get_u32(m, "deepseek4.expert_count", &n_expert);
    model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used);
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("container: safetensors, %" PRIu64 " shards, %" PRIu64 " metadata keys, "
           "%" PRIu64 " tensors\n",
           m->n_shards, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    printf("file size: ");
    /* mapped_bytes, not size: size is ONE shard on this path (the primary), and
     * reporting a single shard here reads as if the model were tiny.  This was
     * the third place that re-derived the sum, and the second that got it
     * wrong -- it is one field now (pulsar_model.mapped_bytes). */
    print_size(m->mapped_bytes);
    printf("\n");
    printf("tensor bytes described by the directory: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < PULSAR_TENSOR_TYPE_COUNT; type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}



pulsar_tensor *model_find_tensor(const pulsar_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}



static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *sa = (const accelerator_tensor_span *)a;
    const accelerator_tensor_span *sb = (const accelerator_tensor_span *)b;
    if (sa->base < sb->base) return -1;
    if (sa->base > sb->base) return 1;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}



static uint64_t accelerator_cuda_preload_span_bytes(void) {
    /* 1 GiB spans; the env override had no caller (L159 inc 4). */
    return 1024ull * 1048576ull;
}



static bool accelerator_span_filter_contains(uint64_t off,
                                             uint64_t bytes,
                                             const uint64_t *span_offsets,
                                             const uint64_t *span_sizes,
                                             uint32_t span_count) {
    if (span_count == 0) return true;
    if (bytes == 0) return true;
    const uint64_t end = off + bytes;
    if (end < off) return false;
    for (uint32_t i = 0; i < span_count; i++) {
        const uint64_t span_end = span_offsets[i] + span_sizes[i];
        if (span_end < span_offsets[i]) return false;
        if (off >= span_offsets[i] && end <= span_end) return true;
    }
    return false;
}



/* Slice 4f (L237): the routed-expert stacks are the tensors residency splits.
 * The container presents each projection's experts as ONE contiguous run
 * (safetensors: st_add_expert_stacks asserts off(e) == off(0) + e*expert_bytes;
 * GGUF: the 3-D tensor is stored that way), so a rank's owned experts are one
 * byte sub-span and the peer-owned experts two holes that are never staged and
 * never read. */
static bool model_tensor_is_expert_stack(const pulsar_tensor *t) {
    static const char suffix[] = "_exps.weight";
    const size_t sl = sizeof(suffix) - 1;
    if (t->ndim != 3 || t->dim[2] == 0 || t->bytes == 0) return false;
    if (t->name.len < sl || memcmp(t->name.ptr + t->name.len - sl, suffix, sl) != 0) return false;
    return t->bytes % t->dim[2] == 0;
}

bool pulsar_model_expert_stack_owned_span(const pulsar_model *m, const pulsar_tensor *t,
                                          uint64_t *off, uint64_t *bytes) {
    *off = 0;
    *bytes = t->bytes;
    if (!model_tensor_is_expert_stack(t)) return false;
    if (m->tp_n_ranks <= 1) return true;
    const uint64_t n = t->dim[2];
    const uint64_t expert_bytes = t->bytes / n;
    if (n > UINT32_MAX ||
        !pulsar_tp_owned_byte_span(m->tp_rank, m->tp_n_ranks, (uint32_t)n, expert_bytes, off, bytes)) {
        pulsar_die("routed-expert stack: the owned byte span was refused (rank/group mismatch)");
    }
    return true;
}

uint64_t pulsar_model_peer_expert_bytes(const pulsar_model *m) {
    if (!m || m->tp_n_ranks <= 1) return 0;
    uint64_t peer = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor *t = &m->tensors[i];
        uint64_t off = 0, bytes = 0;
        if (pulsar_model_expert_stack_owned_span(m, t, &off, &bytes)) peer += t->bytes - bytes;
    }
    return peer;
}

static bool accelerator_prepare_model_tensor_spans(const pulsar_model *m,
                                                   const uint64_t *span_offsets,
                                                   const uint64_t *span_sizes,
                                                   uint32_t span_count,
                                                   const char *skip_prefix,
                                                   uint64_t *prepared_out) {
    uint64_t cap = m->n_tensors;
    if (cap == 0) {
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    accelerator_tensor_span *spans = (accelerator_tensor_span *)xmalloc((size_t)cap * sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint32_t i = 0; i < span_count; i++) {
        if (span_offsets[i] > m->size ||
            span_sizes[i] == 0 ||
            span_sizes[i] > m->size - span_offsets[i]) {
            free(spans);
            return false;
        }
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        /* A model plus --expert-overlay: swapped tensors live in the DONOR's
         * mapping with donor-relative offsets, and are prepared separately by
         * accelerator_prepare_expert_overlay.  A safetensors model ALSO sets
         * ext_map on every tensor -- to its own shard -- so the skip must key
         * on the container, not on the field being set. */
        if (m->n_shards == 0 && t->ext_map) continue;
        if (skip_prefix &&
            t->name.len > strlen(skip_prefix) &&
            memcmp(t->name.ptr, skip_prefix, strlen(skip_prefix)) == 0) continue;
        const uint8_t *base = t->ext_map ? t->ext_map : m->map;
        const uint64_t map_size = t->ext_map ? t->ext_size : m->size;
        if (t->abs_offset > map_size || t->bytes > map_size - t->abs_offset) {
            free(spans);
            return false;
        }
        if (!accelerator_span_filter_contains(t->abs_offset, t->bytes,
                                              span_offsets, span_sizes, span_count)) {
            continue;
        }
        /* Slice 4f: a routed-expert stack is staged over the OWNED experts
         * only.  Peer-owned experts are two holes of whole experts (MBs), far
         * wider than the 64 KiB merge slack below, so no merged span can
         * bridge into them. */
        uint64_t sub_off = 0, sub_bytes = t->bytes;
        (void)pulsar_model_expert_stack_owned_span(m, t, &sub_off, &sub_bytes);
        if (sub_bytes == 0) continue;
        spans[nspan++] = (accelerator_tensor_span){
            .base = base,
            .map_size = map_size,
            .off = t->abs_offset + sub_off,
            .end = t->abs_offset + sub_off + sub_bytes,
        };
    }
    if (nspan == 0) {
        free(spans);
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    const int tty = pulsar_log_is_tty(stderr);
    const uint64_t progress_step = (tty ? 2ull : 16ull) * 1073741824ull;
    uint64_t next_progress = progress_step;
    double last_progress = now_sec();
    uint64_t prepared = 0;
    uint64_t merged = 0;

    const char *accelerator_name = "CUDA";

    fprintf(stderr, "%spulsar: %s preparing model tensor mappings%s",
            tty ? "\r\033[K" : "",
            accelerator_name,
            tty ? ": 0.00 GiB" : "\n");
    fflush(stderr);

    for (uint64_t i = 0; i < nspan;) {
        const uint8_t *base = spans[i].base;
        const uint64_t map_size = spans[i].map_size;
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        /* A tensor of 1 MiB or more starts its OWN span: the arena page-aligns
         * span bases (cuda_model_arena_alloc), and only a tensor at the head of
         * a span inherits that alignment.  Merged behind a 24 B bias, a routed
         * expert block sat 24 B into a page for all 256 experts (L239). */
        while (i < nspan &&
               spans[i].base == base &&
               spans[i].off <= end + 65536u &&
               spans[i].end - off <= max_span &&
               spans[i].end - spans[i].off < 1048576u) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        char label[96];
        snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
        if (pulsar_gpu_cache_model_range(base, map_size, off, end - off, label) == 0) {
            if (tty) fputc('\n', stderr);
            fprintf(stderr,
                    "pulsar: accelerator failed to prepare model tensor span %" PRIu64
                    " at offset %" PRIu64 "\n",
                    merged, off);
            free(spans);
            return false;
        }
        prepared += end - off;
        merged++;

        const double now = now_sec();
        if (prepared >= next_progress || now - last_progress >= (tty ? 2.0 : 10.0)) {
            if (tty) {
                fprintf(stderr, "\r\033[Kpulsar: %s preparing model tensor mappings: %.2f GiB",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            } else {
                fprintf(stderr, "pulsar: %s prepared model tensor mappings %.2f GiB\n",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            }
            fflush(stderr);
            last_progress = now;
            while (next_progress <= prepared) next_progress += progress_step;
        }
    }

    if (tty) fputc('\n', stderr);
    /* The total, once: the progress lines above are ticks every 16 GiB (or 2 s
     * on a tty), so a log's last tick under-reads the sum -- and under TP the
     * sum is the residency claim itself (L237). */
    fprintf(stderr, "pulsar: %s staged %.2f GiB of model tensors in %" PRIu64 " span(s)\n",
            accelerator_name, (double)prepared / 1073741824.0, merged);
    free(spans);
    if (prepared_out) *prepared_out = prepared;
    return true;
}



bool accelerator_cache_model_tensors(pulsar_backend backend,
                                            const pulsar_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count,
                                     const char *skip_prefix) {
    if (backend != PULSAR_BACKEND_CUDA) return true;
    /* A shard table exists for a safetensors model; m->map/size is only shard 0
     * there, so it is not the validity test. */
    if (!m || m->size == 0) return false;
    if (m->n_shards == 0 && !m->map) return false;
    /* Slice 4f: announce the residency lane (rule 5) with the bytes it
     * withholds, so a load log states which experts this rank holds. */
    if (m->tp_n_ranks > 1) {
        fprintf(stderr, "pulsar: TP residency: rank %d/%u stages only its owned experts of every "
                        "stack; %.2f GiB of peer-owned expert bytes are never staged or read\n",
                m->tp_rank, m->tp_n_ranks,
                (double)pulsar_model_peer_expert_bytes(m) / 1073741824.0);
    }
    /* Register each MXFP8 weight's offset so the workhorse matmul executes
     * ONLY registered tensors (per-tensor routing; unregistered offsets are
     * rejected at dispatch). Runs before the weight-cache early-out so it
     * applies in the mmap path too. (Said "the DIRECT_MODEL early-out" when
     * that switch existed; the alternative load strategies are gone -- L030.) */
    uint64_t n_fp8 = 0, n_fp8_lt = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const pulsar_tensor *t = &m->tensors[i];
        if (t->type == PULSAR_TENSOR_MXFP8_LT) {
            /* Register the offset as pre-stored so the FP8 matmul's resolver
             * points cuBLASLt straight at the mapping instead of converting.
             * This is the ONLY MXFP8 storage: a checkpoint that declares the
             * legacy interleaved layout is refused by name at load
             * (st_layout_type), so no plain-38 tensor can exist here. */
            pulsar_gpu_register_fp8_weight(tensor_map_base(m, t), t->abs_offset);
            pulsar_gpu_register_fp8_lt_weight(tensor_map_base(m, t), t->abs_offset);
            n_fp8++;
            n_fp8_lt++;
        }
    }
    if (n_fp8 > 0)
        fprintf(stderr, "pulsar: %llu MXFP8 workhorse weights detected -> FP8 matmul path"
                " (%llu pre-stored MXFP8_LT, zero-copy)\n",
                (unsigned long long)n_fp8, (unsigned long long)n_fp8_lt);
    /* n_fp8 == n_fp8_lt by construction now -- a plain one dies above. The line
     * still prints both because a future divergence should be visible, and
     * because 369/390 is exactly how the double-store was spotted. */
    const double t0 = now_sec();
    uint64_t prepared = 0;
    if (!accelerator_prepare_model_tensor_spans(m, span_offsets, span_sizes, span_count, skip_prefix, &prepared)) {
        return false;
    }
    const double t1 = now_sec();
    const char *accelerator_name = "CUDA";
    fprintf(stderr,
            "pulsar: %s startup model preparation covered %.2f GiB of tensor spans in %.3fs\n",
            accelerator_name, (double)prepared / 1073741824.0, t1 - t0);
    return true;
}



/* Return the in-place tensor payload inside the mapped model (or inside the
 * overlay file's mapping for --expert-overlay swapped tensors). */
const void *tensor_data(const pulsar_model *m, const pulsar_tensor *t) {
    return (const uint8_t *)tensor_map_base(m, t) + t->abs_offset;
}



/* Pre-populate device access for --expert-overlay swapped tensor spans.
 * Must run at startup: the lazy weight-cache copy path cannot execute inside
 * GPU graph encode (synchronous cudaMemcpy during capture fails), which is
 * where an unprepared overlay span would otherwise first be touched. */
bool accelerator_prepare_expert_overlay(pulsar_backend backend,
                                        const pulsar_model *base,
                                        const pulsar_model *overlay) {
    if (backend != PULSAR_BACKEND_CUDA) return true;
    uint64_t prepared = 0;
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        const pulsar_tensor *t = &base->tensors[i];
        if (t->ext_map != overlay->map || t->bytes == 0) continue;
        char label[96];
        snprintf(label, sizeof(label), "overlay:%.*s",
                 (int)(t->name.len < 80 ? t->name.len : 80), t->name.ptr);
        if (pulsar_gpu_cache_external_range(overlay->map, overlay->fd,
                                         t->abs_offset, t->bytes, label) == 0) {
            fprintf(stderr, "pulsar: failed to prepare expert-overlay span for %.*s\n",
                    (int)t->name.len, t->name.ptr);
            return false;
        }
        prepared += t->bytes;
    }
    fprintf(stderr, "pulsar: expert overlay prepared %.2f GiB for device access\n",
            (double)prepared / 1073741824.0);
    return true;
}



/* Swap routed-expert tensor entries in the base model for same-named entries
 * from an overlay model (--expert-overlay PATH:PREFIX). Only 3D `*_exps.*`
 * tensors whose name starts with the prefix are swapped, so the overlay never
 * touches FP8-registered dense weights or streaming offset tables. A swapped
 * entry keeps the overlay file's abs_offset and records the overlay mapping in
 * ext_map/ext_size; consumers resolve the payload via tensor_data()/
 * tensor_map_base(). Returns the number of tensors swapped. */
static bool str_span_contains(const char *s, size_t len, const char *needle) {
    const size_t nlen = strlen(needle);
    if (len < nlen) return false;
    for (size_t i = 0; i + nlen <= len; i++) {
        if (memcmp(s + i, needle, nlen) == 0) return true;
    }
    return false;
}

uint32_t model_apply_expert_overlay(pulsar_model *base, const pulsar_model *overlay,
                                    const char *prefix) {
    const size_t plen = strlen(prefix);
    uint32_t swapped = 0;
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        pulsar_tensor *t = &base->tensors[i];
        if (t->name.len < plen || memcmp(t->name.ptr, prefix, plen) != 0) continue;
        /* Tensor names are length-prefixed, not NUL-terminated: every match
         * below must stay bounded by name.len. */
        if (t->ndim != 3 || !str_span_contains(t->name.ptr, t->name.len, "_exps.")) continue;
        const pulsar_tensor *ov = NULL;
        for (uint64_t j = 0; j < overlay->n_tensors; j++) {
            const pulsar_tensor *c = &overlay->tensors[j];
            if (c->name.len == t->name.len &&
                memcmp(c->name.ptr, t->name.ptr, t->name.len) == 0) {
                ov = c;
                break;
            }
        }
        if (!ov) {
            fprintf(stderr, "pulsar: expert overlay is missing tensor: %.*s\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        if (ov->ndim != t->ndim ||
            memcmp(ov->dim, t->dim, sizeof(ov->dim)) != 0) {
            fprintf(stderr, "pulsar: expert overlay shape mismatch for %.*s\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        if (ov->type == PULSAR_TENSOR_CUTLASS_MXFP4) {
            /* The CUTLASS grouped-GEMM prefill path device-asserts when its
             * expert weights come from an overlay range (observed 2026-07-02,
             * root cause not yet chased). The measurement harness only needs
             * plain MXFP4/IQ2/Q2K donors, so refuse rather than corrupt. */
            fprintf(stderr, "pulsar: expert overlay does not support CUTLASS "
                            "type-40 donor tensors yet (%.*s)\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        t->type = ov->type;
        t->rel_offset = ov->rel_offset;
        t->abs_offset = ov->abs_offset;
        t->elements = ov->elements;
        t->bytes = ov->bytes;
        t->ext_map = overlay->map;
        t->ext_size = overlay->size;
        if (t->bytes > base->max_tensor_bytes) base->max_tensor_bytes = t->bytes;
        swapped++;
    }
    return swapped;
}





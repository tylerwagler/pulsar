/* tp_slice.cpp -- L272 P4b: every tensor-parallel slice a family asks for goes through here.
 *
 * A rank's share of a stored weight is one of a few OPERATIONS on a format: a row slice or an input-column
 * (K) slice of an MXFP8_LT weight registered with the backend, the half of every routed CUTLASS MXFP4 expert,
 * a slice built on the host (an EXL3 output-column or input-row gather, the GDN conv's channels) and kept on
 * the device, or a contiguous range of an expert stack staged as is.  The family says which of its tensors
 * split and how (pulsar_family::tp_slices); these primitives do it.
 *
 * RECORD MODE (pulsar_tp_record_begin): instead of touching the device, each primitive writes one canonical
 * line -- the operation, the tensor, its geometry and range, and a byte hash of a host-built slice.  A host
 * test opens a family's real model inspect-only as rank r of n and records its slices (tests/tp_plan_test.cpp),
 * so a change to how slices are declared is checked line for line against the recorded golden before any pair
 * runs it. */
#include "pulsar_engine_internal.h"
#include "cuda/pulsar_cuda_qwen.h"

static FILE *g_tp_record;

void pulsar_tp_record_begin(FILE *f) { g_tp_record = f; }
void pulsar_tp_record_end(void) { g_tp_record = NULL; }
bool pulsar_tp_recording(void) { return g_tp_record != NULL; }

static void rec_name(const pulsar_tensor *t) { fprintf(g_tp_record, "%.*s", (int)t->name.len, t->name.ptr); }

bool pulsar_tp_slice_fp8_rows(const pulsar_model *m, const pulsar_tensor *t, uint64_t in_dim, uint64_t out_full,
                              uint64_t lo, uint64_t hi) {
    if (g_tp_record) {
        fprintf(g_tp_record, "fp8_rows ");
        rec_name(t);
        fprintf(g_tp_record, " off %llu in %llu out %llu rows [%llu,%llu)\n", (unsigned long long)t->abs_offset,
                (unsigned long long)in_dim, (unsigned long long)out_full, (unsigned long long)lo, (unsigned long long)hi);
        return true;
    }
    return pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(m, t), t->abs_offset, in_dim, out_full, lo, hi) != 0;
}

bool pulsar_tp_slice_fp8_kslice(const pulsar_model *m, const pulsar_tensor *t, uint64_t in_full, uint64_t out_dim,
                                uint64_t lo, uint64_t hi, const void *key) {
    if (g_tp_record) {
        fprintf(g_tp_record, "fp8_kslice ");
        rec_name(t);
        fprintf(g_tp_record, " off %llu in %llu out %llu cols [%llu,%llu) key engine+tensor\n",
                (unsigned long long)t->abs_offset, (unsigned long long)in_full, (unsigned long long)out_dim,
                (unsigned long long)lo, (unsigned long long)hi);
        return true;
    }
    return pulsar_gpu_register_fp8_lt_kslice(tensor_map_base(m, t), t->abs_offset, in_full, out_dim, lo, hi, key,
                                             pulsar_tp_kslice_key_offset(t)) != 0;
}

bool pulsar_tp_slice_mxfp4_half(const void *key_map, const pulsar_model *m, const pulsar_tensor *t, uint32_t n_expert,
                                uint64_t k, uint64_t n, int k_half, uint64_t lo, uint64_t hi, uint64_t src_stride,
                                uint64_t src_data, uint64_t dst_stride, uint64_t dst_data) {
    if (g_tp_record) {
        fprintf(g_tp_record, "mxfp4_half ");
        rec_name(t);
        fprintf(g_tp_record, " off %llu key %llu experts %u k %llu n %llu %s [%llu,%llu) src %llu/%llu dst %llu/%llu\n",
                (unsigned long long)t->abs_offset, (unsigned long long)pulsar_tp_expert_half_offset(m, t), n_expert,
                (unsigned long long)k, (unsigned long long)n, k_half ? "kcols" : "rows", (unsigned long long)lo,
                (unsigned long long)hi, (unsigned long long)src_stride, (unsigned long long)src_data,
                (unsigned long long)dst_stride, (unsigned long long)dst_data);
        return true;
    }
    return pulsar_gpu_register_mxfp4_expert_half(key_map, pulsar_tp_expert_half_offset(m, t), tensor_map_base(m, t),
                                                 t->abs_offset, n_expert, k, n, k_half, lo, hi, src_stride, src_data,
                                                 dst_stride, dst_data) != 0;
}

/* FNV-1a over a built slice: the record's proof that the gathered bytes are the same */
static uint64_t fnv(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

pulsar_gpu_tensor *pulsar_tp_slice_built(const pulsar_tensor *t, const char *how, const uint8_t *bytes, size_t n,
                                         bool *ok) {
    if (g_tp_record) {
        fprintf(g_tp_record, "built %s ", how);
        rec_name(t);
        fprintf(g_tp_record, " bytes %zu fnv %016llx\n", n, (unsigned long long)fnv(bytes, n));
        return NULL;
    }
    pulsar_gpu_tensor *g = pulsar_gpu_tensor_alloc(n);
    if (!g || !pulsar_gpu_tensor_write(g, 0, bytes, n)) {
        pulsar_gpu_tensor_free(g);
        *ok = false;
        return NULL;
    }
    return g;
}

bool pulsar_tp_slice_stage_range(const pulsar_model *m, const pulsar_tensor *t, uint64_t rel, uint64_t bytes,
                                 const char *what) {
    if (g_tp_record) {
        fprintf(g_tp_record, "stage_range ");
        rec_name(t);
        fprintf(g_tp_record, " off %llu +%llu bytes %llu\n", (unsigned long long)t->abs_offset, (unsigned long long)rel,
                (unsigned long long)bytes);
        return true;
    }
    return pulsar_qwen_weight_ptr(tensor_map_base(m, t), t->abs_offset + rel, bytes, what) != NULL;
}

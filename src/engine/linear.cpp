/* linear.cpp -- L272 P4c: the dense linear's one front door, for every family.
 *
 * A dense linear is a weight (a model's tensor, or a row range of it) times an activation.  The family names the
 * weight and hands the activation as it produced it; the kernel arm is the core's choice by (stored format,
 * activation), from the one table admission also reads (pulsar_dense_arm_for, weight_format.cpp):
 *
 *   the MX SLOT -- an activation armed in the backend's cache, keyed by the f32 buffer it was produced from
 *   (DeepSeek's producers emit its E4M3 and / or bf16 planes; pulsar_linear_slot):
 *       mxfp8_lt   -> the E4M3 plane, cuBLASLt (a row range, or the rank's plan's K slice: registered at open, TP)
 *       bf16, f32  -> the bf16 plane, cuBLAS (a row range is offset arithmetic)
 *   raw bf16 ROWS -- a device row pointer (Qwen's producers; pulsar_linear_rows_ref):
 *       mxfp8_lt   -> the W8A16 split-K GEMV / MMA
 *       EXL3       -> the EXL3 dense arm at its rate
 *       bf16       -> the BF16 rows arm (L295 F7: tensor-core rows kernel at decode widths, cuBLASLt beyond)
 *
 * Every arm is today's launcher, unchanged -- this is where a family stops choosing kernels.  A pairing the table
 * has not refuses by name (rule 1); filling one (an EXL3 weight against the slot's bf16 plane, a bf16 weight
 * against raw rows) is an arm here and a line in the table, for every family at once. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_qwen.h"

#include <vector>

static void say_no_arm(const pulsar_tensor *w, const char *against) {
    fprintf(stderr, "pulsar: linear: %.*s is %s, which has no dense arm against %s -- refusing\n", (int)w->name.len,
            w->name.ptr, tensor_type_name(w->type), against);
}

const void *pulsar_weight_device_ptr(const pulsar_model *m, const pulsar_tensor *t, const char *what) {
    if (const void *slice = pulsar_tp_built_ptr(m, t)) return slice;   /* the rank's TP slice (tp_slice.cpp) */
    const void *p = pulsar_gpu_weight_range_ptr(tensor_map_base(m, t), t->abs_offset, t->bytes, what);
    if (!p) fprintf(stderr, "pulsar: no device copy of %.*s (%s) -- refusing\n", (int)t->name.len, t->name.ptr, what);
    return p;
}

bool pulsar_linear_slot(pulsar_gpu_tensor *out, const pulsar_model *m, const pulsar_tensor *w, uint64_t in_dim,
                        uint64_t row_lo, uint64_t row_hi, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    /* a weight the rank's plan K-sliced (TP, row-parallel): its input columns [k_lo, k_hi) are the rank's, the
     * slice registered under the engine key -- the output is a partial the exchange sums */
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t k_lo = 0, k_hi = 0;
    const void *key = NULL;
    const bool kslice = pulsar_tp_slice_of(m, w, &op, &k_lo, &k_hi, &key) && op == PULSAR_TP_OP_FP8_K;
    if (row_hi <= row_lo || w->ndim < 2 || row_hi > w->dim[1] || in_dim != (kslice ? k_hi - k_lo : w->dim[0]) ||
        (kslice && (row_lo != 0 || row_hi != w->dim[1]))) {
        fprintf(stderr, "pulsar: linear: rows [%llu,%llu) x %llu of %.*s [%llu x %llu] -- refusing\n",
                (unsigned long long)row_lo, (unsigned long long)row_hi, (unsigned long long)in_dim, (int)w->name.len,
                w->name.ptr, (unsigned long long)w->dim[0], (unsigned long long)(w->ndim >= 2 ? w->dim[1] : 0));
        return false;
    }
    const void *map = tensor_map_base(m, w);
    const uint64_t size = tensor_map_size(m, w), rows = row_hi - row_lo;
    pulsar_dense_arm arm = pulsar_dense_arm_for(w->type, PULSAR_ACT_SLOT_E4M3);
    if (arm == PULSAR_DENSE_ARM_NONE) arm = pulsar_dense_arm_for(w->type, PULSAR_ACT_SLOT_BF16);
    switch (arm) {
    case PULSAR_DENSE_ARM_MXFP8_SLOT:
        if (kslice)
            return pulsar_gpu_matmul_mxfp8_tensor(out, key, UINT64_MAX / 2u, pulsar_tp_kslice_key_offset(w), in_dim, rows,
                                                  x, n_tok) != 0;
        /* a whole tensor is its own offset; a row slice is the offset registered at open, parent + row_lo * in
         * (the scale plane is not row-addressable, so an unregistered slice refuses in the backend) */
        return pulsar_gpu_matmul_mxfp8_tensor(out, map, size, w->abs_offset + row_lo * in_dim, in_dim, rows, x,
                                              n_tok) != 0;
    case PULSAR_DENSE_ARM_BF16_PLANE:
        return pulsar_gpu_matmul_bf16_tensor(out, map, size, w->abs_offset + row_lo * in_dim * sizeof(uint16_t),
                                             in_dim, rows, x, n_tok) != 0;
    case PULSAR_DENSE_ARM_F32_PLANE:
        return pulsar_gpu_matmul_f32_tensor(out, map, size, w->abs_offset + row_lo * in_dim * sizeof(float), in_dim,
                                            rows, x, n_tok) != 0;
    default:
        say_no_arm(w, "the MX slot (E4M3 / bf16 plane)");
        return false;
    }
}

bool pulsar_linear_rows_ref(const pulsar_model *m, const pulsar_tensor *t, int in, int out, bool prompt,
                            const char *what, pulsar_rows_linear *l) {
    const pulsar_dense_arm arm = pulsar_dense_arm_for(t->type, PULSAR_ACT_ROWS_BF16);
    if (arm != PULSAR_DENSE_ARM_MXFP8_ROWS && arm != PULSAR_DENSE_ARM_EXL3_ROWS && arm != PULSAR_DENSE_ARM_BF16_ROWS) {
        say_no_arm(t, "raw bf16 rows");
        return false;
    }
    l->w = pulsar_weight_device_ptr(m, t, what);
    l->in = in;
    l->out = out;
    /* the launcher's arm: k2 = the EXL3 rate, or 0 with the mxfp8_lt E8M0 plane (after the [out][in] E4M3), or the
     * BF16 flag (L295 F7) */
    l->k2 = arm == PULSAR_DENSE_ARM_EXL3_ROWS ? exl3_type_k2(t->type) : 0;
    l->sf = arm == PULSAR_DENSE_ARM_MXFP8_ROWS && l->w ? (const uint8_t *)l->w + (uint64_t)out * in : NULL;
    l->prompt = prompt;
    l->bf16 = arm == PULSAR_DENSE_ARM_BF16_ROWS;
    return l->w != NULL;
}

pulsar_gpu_tensor *pulsar_draft_head_build(const pulsar_model *m, const pulsar_tensor *head, int in, int vocab,
                                           const int32_t *ids, uint32_t n, const char *what) {
    constexpr int kChunk = 256;   /* one-hot rows a launch: y is kChunk x vocab f32 */
    const uint64_t bytes = pulsar_qwen_mxfp8_bytes((int)n, in);
    bool ok = n > 0 && (int64_t)n < vocab && in > 0 && in % kChunk == 0 && bytes != 0;
    for (uint32_t r = 0; ok && r < n; r++) ok = ids[r] >= 0 && ids[r] < vocab && (r == 0 || ids[r] > ids[r - 1]);
    if (!ok) {
        fprintf(stderr, "pulsar: %s: a draft head of %u ascending ids under %d over %d inputs (a multiple of %d) -- "
                        "refusing\n", what, n, vocab, in, kChunk);
        return NULL;
    }
    pulsar_rows_linear l{};
    if (!pulsar_linear_rows_ref(m, head, in, vocab, false, what, &l)) return NULL;
    const uint64_t ws_bytes = pulsar_rows_linear_workspace_bytes(&l, kChunk);
    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)kChunk * in * 2u);
    pulsar_gpu_tensor *y = pulsar_gpu_tensor_alloc((uint64_t)kChunk * vocab * 4u);
    pulsar_gpu_tensor *ws = pulsar_gpu_tensor_alloc(ws_bytes ? ws_bytes : 256u);
    pulsar_gpu_tensor *dids = pulsar_gpu_tensor_alloc((uint64_t)n * 4u);
    pulsar_gpu_tensor *wb = pulsar_gpu_tensor_alloc((uint64_t)n * in * 2u);
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc(bytes);
    ok = x && y && ws && dids && wb && out && pulsar_gpu_tensor_write(dids, 0, ids, (uint64_t)n * 4u);
    /* the head's rows through its own arm: the linear of the one-hot rows e_j0 .. e_{j0 + kChunk - 1} is the head's
     * input columns j0.. (y[i][v] = W[v][j0 + i]), gathered at the ids into the bf16 [n][in] staging */
    std::vector<uint16_t> onehot((size_t)kChunk * in, 0);
    for (int j0 = 0; ok && j0 < in; j0 += kChunk) {
        for (int i = 0; i < kChunk; i++) {
            if (j0 > 0) onehot[(size_t)i * in + (j0 - kChunk) + i] = 0;
            onehot[(size_t)i * in + j0 + i] = 0x3f80u;   /* bf16 1.0 */
        }
        ok = pulsar_gpu_tensor_write(x, 0, onehot.data(), onehot.size() * 2u) &&
             pulsar_rows_linear_launch(&l, (const uint16_t *)pulsar_gpu_tensor_device_ptr(x), kChunk,
                                       (float *)pulsar_gpu_tensor_device_ptr(y), pulsar_gpu_tensor_device_ptr(ws),
                                       ws_bytes, 0) == 0 &&
             pulsar_head_draft_gather_launch((const float *)pulsar_gpu_tensor_device_ptr(y), (uint64_t)vocab, kChunk,
                                             (const int32_t *)pulsar_gpu_tensor_device_ptr(dids), n,
                                             (uint16_t *)pulsar_gpu_tensor_device_ptr(wb) + j0, (uint64_t)in, 0) == 0 &&
             pulsar_gpu_synchronize() != 0;
    }
    /* stored mxfp8_lt by the producers' encoder: the mxfp8_lt rows arm's weight */
    ok = ok && pulsar_qwen_bf16_to_mxfp8((const uint16_t *)pulsar_gpu_tensor_device_ptr(wb), NULL, (int)n, in,
                                         pulsar_gpu_tensor_device_ptr(out), 0) == 0 &&
         pulsar_gpu_synchronize() != 0;
    for (pulsar_gpu_tensor *t : {x, y, ws, dids, wb}) pulsar_gpu_tensor_free(t);
    if (!ok) {
        fprintf(stderr, "pulsar: %s: the draft head (%u rows of %.*s) could not be built -- refusing\n", what, n,
                (int)head->name.len, head->name.ptr);
        pulsar_gpu_tensor_free(out);
        return NULL;
    }
    fprintf(stderr, "pulsar: %s: draft head: %u of %d rows of %.*s (%s), read through its rows arm, stored MXFP8 "
                    "(%.1f MB)\n", what, n, vocab, (int)head->name.len, head->name.ptr, tensor_type_name(head->type),
            (double)bytes / 1e6);
    return out;
}

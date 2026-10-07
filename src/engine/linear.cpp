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
 *
 * Every arm is today's launcher, unchanged -- this is where a family stops choosing kernels.  A pairing the table
 * has not refuses by name (rule 1); filling one (an EXL3 weight against the slot's bf16 plane, a bf16 weight
 * against raw rows) is an arm here and a line in the table, for every family at once. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "cuda/pulsar_cuda_qwen.h"

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
    if (arm != PULSAR_DENSE_ARM_MXFP8_ROWS && arm != PULSAR_DENSE_ARM_EXL3_ROWS) {
        say_no_arm(t, "raw bf16 rows");
        return false;
    }
    l->w = pulsar_weight_device_ptr(m, t, what);
    l->in = in;
    l->out = out;
    /* the launcher's arm: k2 = the EXL3 rate, or 0 with the mxfp8_lt E8M0 plane (after the [out][in] E4M3) */
    l->k2 = arm == PULSAR_DENSE_ARM_EXL3_ROWS ? exl3_type_k2(t->type) : 0;
    l->sf = arm == PULSAR_DENSE_ARM_MXFP8_ROWS && l->w ? (const uint8_t *)l->w + (uint64_t)out * in : NULL;
    l->prompt = prompt;
    return l->w != NULL;
}

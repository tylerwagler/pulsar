/* tp_slice.cpp -- L272 P4b: the tensor-parallel plan.  A family DECLARES its rank's slices; the core does them.
 *
 * A declaration is a tensor, an axis and the rank's ranges along it -- OUT (a linear's output rows: dim[1], or a
 * plain tensor's outermost dim), IN (a linear's input: dim[0]) or EXPERTS (whole experts of a stack: dim[2]).
 * The OPERATION is the core's choice, by the tensor's format (one table, plan_op):
 *
 *   MXFP8_LT        OUT  -> fp8_rows     a row slice registered with the backend (the stored tensor stays staged)
 *   MXFP8_LT        IN   -> fp8_kslice   an input-column slice registered, keyed by (engine, tensor)
 *   EXL3            OUT  -> exl3_cols    the output columns gathered on the host, kept on the device
 *   EXL3            IN   -> exl3_rows    the input rows gathered likewise
 *   bf16 / f32      OUT  -> view         one range: the forward offsets into the staged tensor
 *                                 -> <type>_channels   several ranges: gathered and kept on the device
 *   CUTLASS MXFP4   OUT/IN on a stack -> mxfp4_half    the [lo, hi) half of every expert, built by the backend
 *   any stack       EXPERTS      -> stage_range        the rank's whole experts staged as stored
 *
 * Any other pairing refuses by name at load.  So does an operation the family's forward does not read
 * (pulsar_family::tp_reads): until one launcher serves every format (P4c) a slice the forward would not look up
 * computes the full tensor in silence.  A tensor whose slice REPLACES it (built, a half, a staged range) is never
 * staged whole: the plan is the residency rule (pulsar_model::tp_unstaged).
 *
 * The plan is built at open after the family's load, before the inspect-only exit (pulsar_tp_plan_build), and
 * run after the GPU is up (pulsar_tp_plan_run).  RECORD MODE (pulsar_tp_record_begin): the run writes one
 * canonical line per slice -- the operation, the tensor, its geometry and range, a byte hash of a host-built
 * slice -- instead of touching the device.  tests/tp_plan_test.cpp records a real model's plan as rank r of n
 * and make tp-plan-gate diffs it against tests/tp-plan-golden/. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "cuda/pulsar_cuda_qwen.h"

#include <unordered_map>
#include <vector>

struct pulsar_tp_plan {
    std::vector<pulsar_tp_slice> slices;
    std::vector<pulsar_tp_op> ops;    ///< per slice, chosen at build
    std::unordered_map<const pulsar_tensor *, pulsar_gpu_tensor *> built;   ///< the host-built slices, by the tensor
};

static FILE *g_tp_record;

void pulsar_tp_record_begin(FILE *f) { g_tp_record = f; }
void pulsar_tp_record_end(void) { g_tp_record = NULL; }

static const char *axis_name(pulsar_tp_axis a) {
    return a == PULSAR_TP_AXIS_OUT ? "out" : a == PULSAR_TP_AXIS_IN ? "in" : "experts";
}

static const char *op_name(pulsar_tp_op op) {
    switch (op) {
    case PULSAR_TP_OP_FP8_ROWS: return "fp8_rows";
    case PULSAR_TP_OP_FP8_K: return "fp8_kslice";
    case PULSAR_TP_OP_EXL3_COLS: return "exl3_cols";
    case PULSAR_TP_OP_EXL3_ROWS: return "exl3_rows";
    case PULSAR_TP_OP_VIEW: return "view";
    case PULSAR_TP_OP_GATHER: return "gather";
    case PULSAR_TP_OP_MXFP4_HALF: return "mxfp4_half";
    case PULSAR_TP_OP_EXPERTS: return "stage_range";
    case PULSAR_TP_OP_NONE: break;
    }
    return "none";
}

static void tname(const pulsar_tensor *t, char *buf, size_t n) {
    snprintf(buf, n, "%.*s", (int)t->name.len, t->name.ptr);
}

bool pulsar_tp_plan_add(pulsar_tp_plan *p, pulsar_model *m, const pulsar_tensor *t, pulsar_tp_axis axis, uint32_t n,
                        const uint64_t *lo, const uint64_t *hi) {
    if (!t || n == 0 || n > PULSAR_TP_SLICE_RANGES) {
        fprintf(stderr, "pulsar: TP plan: a slice of %s with %u ranges -- refusing\n", t ? "a tensor" : "no tensor", n);
        return false;
    }
    pulsar_tp_slice s{};
    s.m = m;
    s.t = t;
    s.axis = axis;
    s.n = n;
    for (uint32_t i = 0; i < n; i++) { s.lo[i] = lo[i]; s.hi[i] = hi[i]; }
    p->slices.push_back(s);
    return true;
}

bool pulsar_tp_plan_add1(pulsar_tp_plan *p, pulsar_model *m, const pulsar_tensor *t, pulsar_tp_axis axis, uint64_t lo,
                         uint64_t hi) {
    return pulsar_tp_plan_add(p, m, t, axis, 1, &lo, &hi);
}

/* ---- the operation a slice takes, by its tensor's format --------------------------------------------------- */

static bool plain_type(uint32_t type) { return type == PULSAR_TENSOR_BF16 || type == PULSAR_TENSOR_F32; }

/* The extent of the axis, or 0 = the tensor has no such axis. */
static uint64_t axis_full(const pulsar_tensor *t, pulsar_tp_axis axis) {
    if (axis == PULSAR_TP_AXIS_EXPERTS) return t->ndim == 3 ? t->dim[2] : 0;
    if (plain_type(t->type)) return axis == PULSAR_TP_AXIS_OUT && t->ndim >= 1 ? t->dim[t->ndim - 1] : 0;
    if (t->ndim < 2) return 0;
    return axis == PULSAR_TP_AXIS_OUT ? t->dim[1] : t->dim[0];
}

static pulsar_tp_op plan_op(const pulsar_tp_slice *s) {
    const pulsar_tensor *t = s->t;
    if (s->axis == PULSAR_TP_AXIS_EXPERTS) return t->ndim == 3 && t->bytes % t->dim[2] == 0 ? PULSAR_TP_OP_EXPERTS
                                                                                            : PULSAR_TP_OP_NONE;
    if (t->type == PULSAR_TENSOR_MXFP8_LT && t->ndim == 2)
        return s->axis == PULSAR_TP_AXIS_OUT ? PULSAR_TP_OP_FP8_ROWS : PULSAR_TP_OP_FP8_K;
    if (exl3_type_k2(t->type) && t->ndim == 2)
        return s->axis == PULSAR_TP_AXIS_OUT ? PULSAR_TP_OP_EXL3_COLS : PULSAR_TP_OP_EXL3_ROWS;
    if (plain_type(t->type) && s->axis == PULSAR_TP_AXIS_OUT) return s->n == 1 ? PULSAR_TP_OP_VIEW : PULSAR_TP_OP_GATHER;
    if (t->type == PULSAR_TENSOR_CUTLASS_MXFP4 && t->ndim == 3) return PULSAR_TP_OP_MXFP4_HALF;
    return PULSAR_TP_OP_NONE;
}

/* The ranges are in bounds, ascending and disjoint; only a gather takes several; a format's cut is aligned. */
static bool plan_ranges_ok(const pulsar_tp_slice *s, pulsar_tp_op op) {
    const uint64_t full = axis_full(s->t, s->axis);
    const bool multi = op == PULSAR_TP_OP_EXL3_COLS || op == PULSAR_TP_OP_GATHER;
    const uint64_t align = op == PULSAR_TP_OP_EXL3_COLS || op == PULSAR_TP_OP_EXL3_ROWS || op == PULSAR_TP_OP_MXFP4_HALF
                               ? 128u : 1u;
    if (!full || (s->n > 1 && !multi)) return false;
    for (uint32_t i = 0; i < s->n; i++) {
        if (s->lo[i] >= s->hi[i] || s->hi[i] > full || s->lo[i] % align || s->hi[i] % align) return false;
        if (i && s->lo[i] < s->hi[i - 1]) return false;
    }
    return true;
}

static bool op_replaces(pulsar_tp_op op) {
    return op == PULSAR_TP_OP_EXL3_COLS || op == PULSAR_TP_OP_EXL3_ROWS || op == PULSAR_TP_OP_GATHER ||
           op == PULSAR_TP_OP_MXFP4_HALF || op == PULSAR_TP_OP_EXPERTS;
}

bool pulsar_tp_plan_build(pulsar_engine *e) {
    if (e->tp_plan) return true;
    if (!e->family->tp_slices) {
        fprintf(stderr, "pulsar: family %s declares no tensor-parallel slices -- refusing\n", e->family->name);
        return false;
    }
    pulsar_tp_plan *p = new pulsar_tp_plan();
    e->tp_plan = p;
    e->model.tp_plan = p;
    if (!e->family->tp_slices(e, p)) return false;
    uint32_t count[9] = {0};
    uint64_t unstaged = 0;
    for (const pulsar_tp_slice &s : p->slices) {
        const pulsar_tp_op op = plan_op(&s);
        char nm[256];
        tname(s.t, nm, sizeof(nm));
        if (op == PULSAR_TP_OP_NONE || !plan_ranges_ok(&s, op)) {
            fprintf(stderr, "pulsar: TP plan: no %s slice of %s (%s, %u range(s) along %s) -- refusing\n",
                    op == PULSAR_TP_OP_NONE ? "operation for a" : "aligned in-bounds", nm, tensor_type_name(s.t->type),
                    s.n, axis_name(s.axis));
            return false;
        }
        if (!(e->family->tp_reads & op)) {
            fprintf(stderr, "pulsar: TP plan: %s's forward does not read a %s slice (%s is %s) -- refusing (L272 P4c)\n",
                    e->family->name, op_name(op), nm, tensor_type_name(s.t->type));
            return false;
        }
        p->ops.push_back(op);
        for (uint32_t b = 0; b < 9; b++) if ((uint32_t)op == 1u << b) count[b]++;
        if (op_replaces(op)) {
            /* the residency rule: the model that holds the tensor (a merged drafter aliases the main model) */
            pulsar_model *owner = s.t >= e->model.tensors && s.t < e->model.tensors + e->model.n_tensors ? &e->model : s.m;
            if (!owner->tp_unstaged) owner->tp_unstaged = (uint8_t *)xcalloc((size_t)owner->n_tensors, 1);
            if (!owner->tp_unstaged[s.t - owner->tensors]) unstaged += s.t->bytes;
            owner->tp_unstaged[s.t - owner->tensors] = 1;
        }
    }
    fprintf(stderr, "pulsar: TP plan: rank %d/%u, %zu slices (fp8 rows %u, fp8 K %u, exl3 cols %u, exl3 rows %u, "
                    "views %u, gathers %u, mxfp4 halves %u, expert ranges %u); %.2f GiB of stored tensors unstaged\n",
            e->model.tp_rank, e->model.tp_n_ranks, p->slices.size(), count[0], count[1], count[2], count[3], count[4],
            count[5], count[6], count[7], (double)unstaged / 1073741824.0);
    return true;
}

void pulsar_tp_plan_free(pulsar_engine *e) {
    if (!e->tp_plan) return;
    for (auto &kv : e->tp_plan->built) pulsar_gpu_tensor_free(kv.second);
    delete e->tp_plan;
    e->tp_plan = NULL;
    e->model.tp_plan = NULL;
}

const void *pulsar_tp_built_ptr(const pulsar_model *m, const pulsar_tensor *t) {
    if (!m->tp_plan) return NULL;
    const auto it = m->tp_plan->built.find(t);
    return it == m->tp_plan->built.end() ? NULL : pulsar_gpu_tensor_device_ptr(it->second);
}

/* ---- EXL3 [trellis | suh | svh], trellis in (k-tile, n-tile, word) order (exl3_trellis.h) ---------------------
 * An output-column slice is, per k-tile row, a run of n-tiles plus that range of svh (suh whole); an input-row
 * slice is a run of k-tile rows plus that range of suh (svh whole).  Every cut is 128-aligned, so the Hadamard
 * blocks on both sides stay whole and a slice is bytes copied -- the rank's matmul is exactly the full one's
 * rows / columns (a K slice's output is a partial the all-reduce sums). */
static bool exl3_tile_geometry(const pulsar_tensor *t, uint64_t *K, uint64_t *N, uint64_t *trellis, uint64_t *tile) {
    const int k2 = exl3_type_k2(t->type);
    uint64_t sc = 0, stride = 0;
    *K = t->dim[0];
    *N = t->dim[1];
    if (!k2 || !exl3_expert_layout(*K, *N, k2, trellis, &sc, &stride) || stride != t->bytes) return false;
    *tile = *trellis / ((*K / 16u) * (*N / 16u));
    return true;
}

static bool exl3_slice_cols(const uint8_t *src, const pulsar_tp_slice *s, std::vector<uint8_t> &out) {
    uint64_t K = 0, N = 0, tr = 0, tile = 0;
    if (!exl3_tile_geometry(s->t, &K, &N, &tr, &tile)) return false;
    for (uint64_t kt = 0; kt < K / 16u; kt++)
        for (uint32_t i = 0; i < s->n; i++) {
            const uint8_t *p = src + (kt * (N / 16u) + s->lo[i] / 16u) * tile;
            out.insert(out.end(), p, p + (s->hi[i] - s->lo[i]) / 16u * tile);
        }
    out.insert(out.end(), src + tr, src + tr + K * 2u);                       /* suh: the whole input */
    for (uint32_t i = 0; i < s->n; i++)
        out.insert(out.end(), src + tr + K * 2u + s->lo[i] * 2u, src + tr + K * 2u + s->hi[i] * 2u);
    return true;
}

static bool exl3_slice_rows(const uint8_t *src, const pulsar_tp_slice *s, std::vector<uint8_t> &out) {
    uint64_t K = 0, N = 0, tr = 0, tile = 0;
    const uint64_t k0 = s->lo[0], k1 = s->hi[0];
    if (!exl3_tile_geometry(s->t, &K, &N, &tr, &tile)) return false;
    out.insert(out.end(), src + k0 / 16u * (N / 16u) * tile, src + k1 / 16u * (N / 16u) * tile);
    out.insert(out.end(), src + tr + k0 * 2u, src + tr + k1 * 2u);            /* suh: the rank's inputs */
    out.insert(out.end(), src + tr + K * 2u, src + tr + K * 2u + N * 2u);       /* svh: the whole output */
    return true;
}

/* ---- the run -------------------------------------------------------------------------------------------------- */

/* FNV-1a over a built slice: the record's proof that the gathered bytes are the same */
static uint64_t fnv(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static bool run_built(pulsar_tp_plan *p, const pulsar_tp_slice *s, const char *how, const std::vector<uint8_t> &b,
                      const char *nm) {
    if (g_tp_record) {
        fprintf(g_tp_record, "built %s %s bytes %zu fnv %016llx\n", how, nm, b.size(), (unsigned long long)fnv(b.data(), b.size()));
        return true;
    }
    pulsar_gpu_tensor *g = pulsar_gpu_tensor_alloc(b.size());
    if (!g || !pulsar_gpu_tensor_write(g, 0, b.data(), b.size())) {
        pulsar_gpu_tensor_free(g);
        return false;
    }
    p->built[s->t] = g;
    return true;
}

static bool run_slice(pulsar_engine *e, pulsar_tp_plan *p, const pulsar_tp_slice *s, pulsar_tp_op op, uint64_t *bytes) {
    const pulsar_model *m = s->m;
    const pulsar_tensor *t = s->t;
    const unsigned long long off = (unsigned long long)t->abs_offset, lo = s->lo[0], hi = s->hi[0];
    char nm[256];
    tname(t, nm, sizeof(nm));
    switch (op) {
    case PULSAR_TP_OP_FP8_ROWS:
        if (g_tp_record) {
            fprintf(g_tp_record, "fp8_rows %s off %llu in %llu out %llu rows [%llu,%llu)\n", nm, off,
                    (unsigned long long)t->dim[0], (unsigned long long)t->dim[1], lo, hi);
            return true;
        }
        return pulsar_gpu_register_fp8_lt_row_slice(tensor_map_base(m, t), t->abs_offset, t->dim[0], t->dim[1], lo, hi) != 0;
    case PULSAR_TP_OP_FP8_K:
        if (g_tp_record) {
            fprintf(g_tp_record, "fp8_kslice %s off %llu in %llu out %llu cols [%llu,%llu) key engine+tensor\n", nm, off,
                    (unsigned long long)t->dim[0], (unsigned long long)t->dim[1], lo, hi);
            return true;
        }
        return pulsar_gpu_register_fp8_lt_kslice(tensor_map_base(m, t), t->abs_offset, t->dim[0], t->dim[1], lo, hi, e,
                                                 pulsar_tp_kslice_key_offset(t)) != 0;
    case PULSAR_TP_OP_VIEW:
        if (g_tp_record)
            fprintf(g_tp_record, "view %s off %llu rows [%llu,%llu) of %llu\n", nm, off, lo, hi,
                    (unsigned long long)axis_full(t, s->axis));
        return true;   /* the forward offsets into the staged tensor */
    case PULSAR_TP_OP_EXL3_COLS:
    case PULSAR_TP_OP_EXL3_ROWS: {
        std::vector<uint8_t> b;
        const uint8_t *src = (const uint8_t *)tensor_data(m, t);
        if (!(op == PULSAR_TP_OP_EXL3_COLS ? exl3_slice_cols(src, s, b) : exl3_slice_rows(src, s, b))) {
            fprintf(stderr, "pulsar: TP: %s does not slice as EXL3 tiles -- refusing\n", nm);
            return false;
        }
        *bytes += b.size();
        return run_built(p, s, op_name(op), b, nm);
    }
    case PULSAR_TP_OP_GATHER: {
        /* rows of the outermost dim, each t->bytes / full long */
        const uint64_t row = t->bytes / axis_full(t, s->axis);
        const uint8_t *src = (const uint8_t *)tensor_data(m, t);
        std::vector<uint8_t> b;
        for (uint32_t i = 0; i < s->n; i++) b.insert(b.end(), src + s->lo[i] * row, src + s->hi[i] * row);
        char how[32];
        snprintf(how, sizeof(how), "%s_channels", tensor_type_name(t->type));
        *bytes += b.size();
        return run_built(p, s, how, b, nm);
    }
    case PULSAR_TP_OP_MXFP4_HALF: {
        /* a stack [k][n] x experts: OUT halves the n rows (gate / up), IN the k columns (down) */
        const bool k_half = s->axis == PULSAR_TP_AXIS_IN;
        const uint64_t k = t->dim[0], n = t->dim[1];
        const uint32_t n_exp = (uint32_t)t->dim[2];
        uint64_t sd = 0, ssf = 0, ss = 0, hd = 0, hsf = 0, hs = 0;
        cutlass_mxfp4_expert_layout(k, n, &sd, &ssf, &ss);
        cutlass_mxfp4_expert_layout(k_half ? hi - lo : k, k_half ? n : hi - lo, &hd, &hsf, &hs);
        *bytes += (uint64_t)n_exp * hs;
        if (g_tp_record) {
            fprintf(g_tp_record, "mxfp4_half %s off %llu key %llu experts %u k %llu n %llu %s [%llu,%llu) src %llu/%llu "
                                 "dst %llu/%llu\n", nm, off, (unsigned long long)pulsar_tp_expert_half_offset(m, t), n_exp,
                    (unsigned long long)k, (unsigned long long)n, k_half ? "kcols" : "rows", lo, hi,
                    (unsigned long long)ss, (unsigned long long)sd, (unsigned long long)hs, (unsigned long long)hd);
            return true;
        }
        return pulsar_gpu_register_mxfp4_expert_half(e, pulsar_tp_expert_half_offset(m, t), tensor_map_base(m, t),
                                                     t->abs_offset, n_exp, k, n, k_half ? 1 : 0, lo, hi, ss, sd, hs,
                                                     hd) != 0;
    }
    case PULSAR_TP_OP_EXPERTS: {
        const uint64_t stride = t->bytes / t->dim[2], rel = lo * stride, n = (hi - lo) * stride;
        *bytes += n;
        if (g_tp_record) {
            fprintf(g_tp_record, "stage_range %s off %llu +%llu bytes %llu\n", nm, off, (unsigned long long)rel,
                    (unsigned long long)n);
            return true;
        }
        return pulsar_qwen_weight_ptr(tensor_map_base(m, t), t->abs_offset + rel, n, "TP expert range") != NULL;
    }
    case PULSAR_TP_OP_NONE: break;
    }
    return false;
}

bool pulsar_tp_plan_run(pulsar_engine *e) {
    pulsar_tp_plan *p = e->tp_plan;
    if (!p) return true;
    uint64_t bytes = 0;
    for (size_t i = 0; i < p->slices.size(); i++)
        if (!run_slice(e, p, &p->slices[i], p->ops[i], &bytes)) {
            char nm[256];
            tname(p->slices[i].t, nm, sizeof(nm));
            fprintf(stderr, "pulsar: TP rank %d: the %s slice of %s could not be built -- refusing\n", e->model.tp_rank,
                    op_name(p->ops[i]), nm);
            return false;
        }
    /* L272 B9: resident weights the model's staged count never sees (pulsar_engine::weights_resident_bytes) */
    if (!g_tp_record) e->tp_built_bytes += bytes;
    fprintf(stderr, "pulsar: TP rank %d: %zu slices %s, %.2f GiB built or staged\n", e->model.tp_rank, p->slices.size(),
            g_tp_record ? "recorded" : "in place", (double)bytes / 1073741824.0);
    return true;
}

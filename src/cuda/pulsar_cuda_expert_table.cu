/* PLAN 94 phase 1 (L217): the per-expert ADDRESS TABLE.
 *
 * Owns the ONE place an expert's address is computed.  Before this existed every
 * consumer derived `base + e*stride` itself (type 40 in two kernels and three GEMV
 * arms, type 44 in two planes); now they read a device array instead, which is what
 * lets phase 2 point an expert at a slot the cache assigns, moves and evicts at
 * runtime -- an address no arithmetic over `e` can express.
 *
 * It lives in its own TU because that rule is worth a gate that drives it directly
 * (tests/expert_table_gate.cu) rather than one that stands up a model and a graph to
 * observe it, and because phase 2 will change the FILL while leaving every consumer
 * untouched.  Nothing here knows about the MoE arms.
 */
#include "pulsar_cuda_internal.h"

/* PLAN 94 phase 1 (L217): the per-expert ADDRESS TABLE for a routed type-40
 * (CUTLASS MXFP4) stack.
 *
 * Today every consumer derives an expert's bytes arithmetically: the grouped
 * setup writes `ptrB[e] = B_base + e*B_stride` and the GEMV arms compute
 * `gate_w + e*stride` inline.  That makes an expert's address a FACT OF THE
 * MODEL LAYOUT, which is exactly what phase 2 has to break: an expert will live
 * in a slot that the slab cache assigns, moves and evicts at runtime, and no
 * arithmetic over `e` can express it.
 *
 * So the address becomes data.  One device array of `n_total` pointers per
 * stack, built ONCE per (base, stride, n_total) from the same arithmetic the
 * consumers used to perform, and read by them instead.  This commit is
 * deliberately bit-identical -- the contents are `base + e*stride` -- and its
 * whole value is that it isolates the kernel-side change from the allocation
 * and IO that phase 2 adds: when the slab lands, only the FILL changes.
 *
 * Ownership follows L188's non-finite flag and the fp8 pointer cache: entries
 * point into the per-engine model arena, so they are freed at backend cleanup
 * (`pulsar_gpu_cleanup` -> `mxfp4_expert_tables_clear`).  A later engine open in
 * the same process typically maps the model at the same base address, so a
 * surviving entry would false-positive and serve a dangling pointer. */
struct expert_table {
    const void *base;                 /* the arena base the entries were derived from */
    uint64_t    stride0;              /* plane 0's per-expert stride, in bytes */
    uint64_t    off1;                 /* plane 1's offset from base (planes == 2 only) */
    uint64_t    stride1;              /* plane 1's per-expert stride, in bytes */
    uint32_t    n_total;
    uint32_t    planes;               /* 1 = type 40 (data; SF at +data_bytes), 2 = type 44 (d,q) */
    const void **entries;             /* device array [planes * n_total] */
    struct expert_table *next;
};

static struct expert_table *g_expert_tables = NULL;

__global__ static void expert_table_fill_kernel(
        const void **entries, const uint8_t *base,
        uint64_t stride0, uint64_t off1, uint64_t stride1, uint32_t n_total, uint32_t planes) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total) return;
    entries[(size_t)e * planes] = base + (size_t)e * stride0;
    if (planes > 1) entries[(size_t)e * planes + 1] = base + off1 + (size_t)e * stride1;
}

static const void *const *expert_table_get(const void *base, uint32_t n_total, uint32_t planes,
                                           uint64_t stride0, uint64_t off1, uint64_t stride1) {
    if (!base || !n_total || !stride0) return NULL;
    for (struct expert_table *t = g_expert_tables; t; t = t->next)
        if (t->base == base && t->stride0 == stride0 && t->off1 == off1 && t->stride1 == stride1 &&
            t->n_total == n_total && t->planes == planes) return t->entries;
    const void **entries = NULL;
    const size_t bytes = (size_t)n_total * planes * sizeof(*entries);
    if (!cuda_ok(cudaMalloc((void **)&entries, bytes), "expert table alloc")) return NULL;
    expert_table_fill_kernel<<<(n_total + 255u) / 256u, 256>>>(
        entries, (const uint8_t *)base, stride0, off1, stride1, n_total, planes);
    if (!cuda_ok(cudaGetLastError(), "expert table fill launch")) {
        (void)cudaFree((void *)entries);
        return NULL;
    }
    struct expert_table *t = (struct expert_table *)calloc(1, sizeof(*t));
    if (!t) {
        (void)cudaFree((void *)entries);
        return NULL;
    }
    t->base = base; t->stride0 = stride0; t->off1 = off1; t->stride1 = stride1;
    t->n_total = n_total; t->planes = planes; t->entries = entries;
    t->next = g_expert_tables;
    g_expert_tables = t;
    return entries;
}

const uint8_t *const *mxfp4_expert_table(const void *base, uint64_t stride, uint32_t n_total) {
    return (const uint8_t *const *)expert_table_get(base, n_total, 1, stride, 0, 0);
}

/* The TYPE 44 arm: IQ2_XXS_MMQ_K is two planes per expert, at
 *   d[e] = base + e*nb*M halves,  q[e] = base + align64(E*nb*M*2) + e*nb*8*M uint2s
 * (L202's layout, ds4_mmq_d2r.cu:918-926).  Both must be rebased together when
 * an expert moves, so the table is ONE array of [d,q] pairs rather than two
 * arrays -- the pair IS the eviction unit. */
const void *const *iq2_expert_table(const void *base, uint32_t n_total, uint32_t nb, uint32_t M) {
    const uint64_t stride0 = (uint64_t)nb * M * 2ull;                    /* half per element   */
    const uint64_t off1    = (stride0 * n_total + 63ull) & ~63ull;       /* q plane base       */
    const uint64_t stride1 = (uint64_t)nb * M * 8ull * sizeof(uint2);    /* uint2 per element  */
    return expert_table_get(base, n_total, 2, stride0, off1, stride1);
}

/* The EXL3 arm (L245): one self-contained [trellis | scales] slice per expert
 * (exl3_expert_layout), so both planes share the expert stride and plane 1 is
 * `split` bytes into the slice: entries [t,s] with t[e] = base + e*stride,
 * s[e] = t[e] + split.  Per-expert contiguity is what lets the restack code
 * treat an EXL3 stack like any other; the table exists so the kernels never do
 * the address arithmetic themselves. */
const void *const *exl3_expert_table(const void *base, uint32_t n_total, uint64_t stride, uint64_t split) {
    if (split == 0 || split >= stride) return NULL;
    return expert_table_get(base, n_total, 2, stride, split, stride);
}

void mxfp4_expert_tables_clear(void) {
    for (struct expert_table *t = g_expert_tables; t; ) {
        struct expert_table *next = t->next;
        (void)cudaFree((void *)t->entries);
        free(t);
        t = next;
    }
    g_expert_tables = NULL;
}

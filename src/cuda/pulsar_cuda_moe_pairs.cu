/* The routed-expert SORTED-PAIR builders: one pass counting the slots each
 * expert was routed, an exclusive prefix sum turning those counts into pair-slot
 * offsets, and one pass scattering each pair index into its expert's run.  The
 * grouped CUTLASS MXFP4 lane and the mixed type-40/44 lane both build their
 * expert-major schedule this way.
 *
 * They live in their own TU because they contain the one place a ROUTER ID
 * becomes an ARRAY INDEX, and that rule -- including the fail-closed flag in
 * `pulsar_gpu_routed_moe_route_oob_take` -- is worth a test that drives it
 * directly (tests/moe_route_bounds_gate.cu) instead of one that has to stand up
 * a model, a graph and a scratch reservation to observe it.  Nothing else in
 * this file knows about the MoE arms; the callers pass the arm tag in.
 */
#include "pulsar_cuda_internal.h"

/* PLAN 94 phase 1 (L217): the routed-expert ROUTE BOUNDS flag.
 *
 * `selected[]` carries one expert id per (token, slot), chosen by the router
 * over the GATE tensor's expert dimension -- which every REAP artifact pads to
 * 256 -- while the routed stacks hold only `n_expert_present` experts and the
 * pair maps (`counts`, `cursors`, `offsets`), the padded row maps and
 * `sorted_pairs` are all sized for exactly those.  Nothing remaps ids: the
 * artifact's own convention is that a pruned expert can never win a slot.
 *
 * The IQ2_XXS MMQ arm is safe against a violation because ggml's id helper
 * only carries in-range ids and drops the rest (ds4_mmq.cu:368-379).  The
 * grouped CUTLASS MXFP4 arm was NOT: the two sorted-pair builders below index
 * `counts`/`cursors` with the raw id, so a single id >= n_expert_present writes
 * past them -- into `offsets`, the padded row maps and `sorted_pairs`.  That is
 * silently wrong routing at best and a corrupt scratch reservation at worst,
 * with nothing printed either way.
 *
 * A NEGATIVE id is not a violation: the router's NaN path emits -1 by design,
 * and both builders have always clamped it to expert 0.  That behaviour is
 * preserved exactly; only the upper bound is new.
 *
 * This guard is the prerequisite for PLAN 94's per-expert pointer table, where
 * an out-of-range id becomes a wild READ of model memory instead of a scratch
 * write.  Same discipline as L188's non-finite flag: the first writer records
 * it on the device, the host reads it at the step's existing stream drain, and
 * the step is REFUSED by name -- no sanitizer, no silent clamp-and-continue.
 *
 * Code layout: bits 0..7 = layer_index + 1 (0 means "clear"), bits 8..15 = arm. */
__device__ static uint32_t g_moe_route_oob = 0u;

/** Record `code` in the route-bounds flag (first writer wins). */
__device__ static __forceinline__ void moe_flag_route_oob(uint32_t code) {
    atomicCAS(&g_moe_route_oob, 0u, code);
}

/** Clamp `expert_i` into [0, n_total), flagging an upper-bound violation.
 *  `expert_i` is an lvalue so both builders share one authority for the rule. */
__device__ static __forceinline__ void moe_clamp_expert_id(int32_t &expert_i, uint32_t n_total, uint32_t oob_code) {
    if (expert_i < 0) {
        expert_i = 0;                       /* the router's NaN path: by design, not a violation */
    } else if ((uint32_t)expert_i >= n_total) {
        moe_flag_route_oob(oob_code);
        expert_i = 0;                       /* in bounds so the maps stay consistent; the step is refused */
    }
}

/** TP OWNERSHIP (slice 4c): is `expert_i` in this rank's slice of the routed
 *  experts?  A peer-owned pair contributes NOTHING to this rank's partial --
 *  the group's all-reduce sums the ranks' partials into the full routed sum,
 *  so a rank that also counted a peer's pair would double it.  Dropping it
 *  here is what leaves the grouped GEMM's group for that expert at M=0 (no
 *  bytes, no FLOPs) and is the whole byte/FLOP cut of the split.
 *
 *  The full range [0, n_total) is the single-rank path: every pair is owned
 *  and the predicate is inert.  The bound comes from the ONE authority,
 *  `pulsar_tp_owned_range` (src/tp) -- the kernel never recomputes it.
 *
 *  BOTH builders call this with the same arguments, and it runs AFTER
 *  `moe_clamp_expert_id`: an out-of-range id is folded to 0 and flagged first,
 *  so the route-bounds refusal still fires on every rank no matter which slice
 *  happens to own expert 0. */
__device__ static __forceinline__ bool moe_pair_owned(uint32_t expert_i, uint32_t expert_lo, uint32_t expert_hi) {
    return expert_i >= expert_lo && expert_i < expert_hi;
}

int pulsar_gpu_routed_moe_route_oob_take(uint32_t *layer_index, const char **arm) {
    uint32_t code = 0u;
    if (!cuda_ok(cudaMemcpyFromSymbol(&code, g_moe_route_oob, sizeof code, 0, cudaMemcpyDeviceToHost),
                 "routed MoE route-bounds flag read")) return -1;
    if (code == 0u) return 0;
    const uint32_t zero = 0u;
    if (!cuda_ok(cudaMemcpyToSymbol(g_moe_route_oob, &zero, sizeof zero, 0, cudaMemcpyHostToDevice),
                 "routed MoE route-bounds flag clear")) return -1;
    if (layer_index) *layer_index = (code & 0xffu) - 1u;
    if (arm) {
        switch (code >> 8) {
        case MOE_OOB_ARM_GROUPED_COUNT:   *arm = "grouped CUTLASS MXFP4 count"; break;
        case MOE_OOB_ARM_GROUPED_SCATTER: *arm = "grouped CUTLASS MXFP4 scatter"; break;
        case MOE_OOB_ARM_MIXED_COUNT:     *arm = "mixed type-40/type-44 count"; break;
        case MOE_OOB_ARM_MIXED_SCATTER:   *arm = "mixed type-40/type-44 scatter"; break;
        default:                          *arm = "unknown arm"; break;
        }
    }
    return 1;
}

__global__ void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total,
        uint32_t oob_code,
        uint32_t expert_lo,
        uint32_t expert_hi) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    moe_clamp_expert_id(expert_i, n_total, oob_code);
    if (!moe_pair_owned((uint32_t)expert_i, expert_lo, expert_hi)) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

/* Exclusive prefix sum over the per-expert counts.  This ran as <<<1,1>>>: one
 * CUDA thread walking 256 experts serially, on every CUTLASS layer of every
 * step, with two more launches either side of it.  Now a single block scans in
 * shared memory.
 *
 * The results are IDENTICAL, not merely equivalent: these are uint32 counts and
 * integer addition is associative and exact, so reassociating the sum cannot
 * move a value.  That is why this is safe to change under a bit-exactness
 * regime where the same edit on floats would not be.
 *
 * Chunked so expert_count > blockDim still works; the model's 256 fits one
 * pass. */
__global__ void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
    __shared__ uint32_t sh[256];
    __shared__ uint32_t base;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) base = 0u;
    __syncthreads();

    for (uint32_t chunk = 0; chunk < expert_count; chunk += 256u) {
        const uint32_t e = chunk + tid;
        const uint32_t v = (e < expert_count) ? counts[e] : 0u;
        sh[tid] = v;
        __syncthreads();
        for (uint32_t off = 1u; off < 256u; off <<= 1) {
            const uint32_t add = (tid >= off) ? sh[tid - off] : 0u;
            __syncthreads();
            sh[tid] += add;
            __syncthreads();
        }
        if (e < expert_count) {
            const uint32_t excl = base + sh[tid] - v;   /* inclusive -> exclusive */
            offsets[e] = excl;
            cursors[e] = excl;
        }
        __syncthreads();
        if (tid == 255u) base += sh[255];
        __syncthreads();
    }
    if (tid == 0) offsets[expert_count] = base;
}

__global__ void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total,
        uint32_t oob_code,
        uint32_t expert_lo,
        uint32_t expert_hi) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    moe_clamp_expert_id(expert_i, n_total, oob_code);
    if (!moe_pair_owned((uint32_t)expert_i, expert_lo, expert_hi)) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

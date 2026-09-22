/* MOE ROUTE BOUNDS GATE -- PLAN 94 phase 1 (L217).
 *
 * `selected[]` holds one router id per (token, slot).  Nothing remaps those ids,
 * and every REAP artifact pads the router's expert dimension to 256 while the
 * routed stacks hold only `n_expert_present` experts -- so the id space and the
 * pair maps (`counts`, `cursors`, `offsets`, `sorted_pairs`) are sized from two
 * DIFFERENT numbers.  The artifact's convention is that a pruned expert can
 * never win a slot; the engine's job is to not be silently wrong if one does.
 *
 * Before this gate's guard existed, the two sorted-pair builders indexed
 * `counts`/`cursors` with the raw id: one id >= n_total wrote past those arrays
 * into `offsets`, the padded row maps and `sorted_pairs` -- wrong routing at
 * best, a corrupt scratch reservation at worst, and nothing printed either way.
 * A wrong expert ADDRESS has the same signature as the mis-ordered router that
 * `cuda-reap-router-audit` exists for: fluent output, no crash.  That is why
 * PLAN 94 phase 1 lands this guard before it introduces a per-expert pointer
 * table, where the same id becomes a wild read of model memory.
 *
 * WHAT THIS GATE PROVES, and why it is not a re-statement of the kernel:
 *   1. every array is passed with canary words on BOTH sides, so an
 *      out-of-range index is caught as a tripped canary rather than as a
 *      plausible-looking count -- the failure mode is memory, not arithmetic;
 *   2. the schedule is checked STRUCTURALLY (each expert's run in
 *      `sorted_pairs` holds exactly the pairs whose clamped id is that expert,
 *      and the runs tile [0, pair_count) exactly), which is the invariant an
 *      overflow breaks;
 *   3. a NEGATIVE id must NOT raise the flag: the router's NaN path emits -1 by
 *      design and has always been clamped to expert 0, so flagging it would
 *      turn a designed route into a refused step;
 *   4. the flag names BOTH the layer and the arm, and taking it clears it --
 *      an untaken flag would be charged to the next step's bytes.
 * Remove the clamp and check 1 or 2 fails; flag a negative id and check 3
 * fails; forget the clear and check 4 fails.  Model-free: it includes the
 * builders' TU and links no engine, so it runs in seconds.
 *
 * usage: ./tests/moe_route_bounds_gate
 */
#include "../src/cuda/pulsar_cuda_moe_pairs.cu"

#include <cstdio>
#include <cstring>
#include <vector>

/* The builders' TU reaches cuda_ok() only from the flag take; the engine's lives
 * in pulsar_cuda_runtime.cu.  This stub reproduces its contract (1 on success)
 * and prints CUDA's own message, so a real launch failure is reported by name
 * here instead of being swallowed. */
int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "moe-route-bounds: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

#define CUDA_OK(call) do { \
        cudaError_t e_ = (call); \
        if (e_ != cudaSuccess) { \
            fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: %s: %s\n", #call, cudaGetErrorString(e_)); \
            return 1; \
        } \
    } while (0)

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static const uint32_t kCanary = 0xC0FFEEu;

/* Run one count + prefix + scatter over `ids`, with canary words on both sides
 * of every array.  Returns the clamped ids the builders actually used (recovered
 * from the schedule) so the caller can check the rule, not just the absence of a
 * crash. */
struct Schedule {
    std::vector<uint32_t> counts;   /* [n_total] */
    std::vector<uint32_t> offsets;  /* [n_total + 1] */
    std::vector<uint32_t> pairs;    /* [pair_count], expert-major */
    bool canary_intact = true;
};

static int run_schedule(const std::vector<int32_t> &ids, uint32_t n_total,
                        uint32_t oob_code, Schedule &out,
                        uint32_t expert_lo, uint32_t expert_hi) {
    const uint32_t pair_count = (uint32_t)ids.size();
    CHECK(n_total > 0 && n_total <= 256, "n_total %u outside the prefix kernel's one-block range", n_total);

    const size_t cnt_n = n_total + 2;            /* canary | n_total | canary */
    const size_t off_n = (n_total + 1) + 2;      /* canary | n_total+1 | canary */
    const size_t pr_n  = pair_count + 2;

    std::vector<uint32_t> cnt_h(cnt_n, kCanary), off_h(off_n, kCanary), cur_h(cnt_n, kCanary),
                          pr_h(pr_n, kCanary);
    uint32_t *cnt = nullptr, *off = nullptr, *cur = nullptr, *pr = nullptr;
    int32_t  *sel = nullptr;
    CUDA_OK(cudaMalloc(&cnt, cnt_n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&off, off_n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&cur, cnt_n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&pr,  pr_n  * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&sel, pair_count * sizeof(int32_t)));

    /* Every device array starts ZEROED in its body, as the engine's scratch
     * reservation does (memsetAsync before the launch), and holds the canary
     * outside it. */
    CUDA_OK(cudaMemcpy(cnt, cnt_h.data(), cnt_n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(off, off_h.data(), off_n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(cur, cur_h.data(), cnt_n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(pr,  pr_h.data(),  pr_n  * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(cnt + 1, 0, n_total * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(cur + 1, 0, n_total * sizeof(uint32_t)));
    CUDA_OK(cudaMemcpy(sel, ids.data(), pair_count * sizeof(int32_t), cudaMemcpyHostToDevice));

    const uint32_t blocks = (pair_count + 255u) / 256u;
    moe_count_sorted_pairs_kernel<<<blocks, 256>>>(cnt + 1, sel, pair_count, n_total, oob_code,
                                                   expert_lo, expert_hi);
    moe_prefix_sorted_pairs_kernel<<<1, 256>>>(off + 1, cur + 1, cnt + 1, n_total);
    moe_scatter_sorted_pairs_kernel<<<blocks, 256>>>(pr + 1, cur + 1, sel, pair_count, n_total, oob_code,
                                                     expert_lo, expert_hi);
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(cnt_h.data(), cnt, cnt_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(off_h.data(), off, off_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(pr_h.data(),  pr,  pr_n  * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(cur_h.data(), cur, cnt_n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* (1) the canaries: an index outside [0, n_total) lands on one of these. */
    if (cnt_h[0] != kCanary || cnt_h[cnt_n - 1] != kCanary) {
        fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: counts[] canary tripped (id escaped [0,%u))\n", n_total);
        out.canary_intact = false;
    }
    if (cur_h[0] != kCanary || cur_h[cnt_n - 1] != kCanary) {
        fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: cursors[] canary tripped (id escaped [0,%u))\n", n_total);
        out.canary_intact = false;
    }
    if (off_h[0] != kCanary || off_h[off_n - 1] != kCanary) {
        fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: offsets[] canary tripped (id escaped [0,%u])\n", n_total);
        out.canary_intact = false;
    }
    if (pr_h[0] != kCanary || pr_h[pr_n - 1] != kCanary) {
        fprintf(stderr, "MOE-ROUTE-BOUNDS FAIL: sorted_pairs[] canary tripped (id escaped [0,%u))\n", n_total);
        out.canary_intact = false;
    }
    g_fail |= !out.canary_intact;

    out.counts.assign(cnt_h.begin() + 1, cnt_h.begin() + 1 + n_total);
    out.offsets.assign(off_h.begin() + 1, off_h.begin() + 1 + n_total + 1);
    out.pairs.assign(pr_h.begin() + 1, pr_h.begin() + 1 + pair_count);

    CUDA_OK(cudaFree(cnt)); CUDA_OK(cudaFree(off)); CUDA_OK(cudaFree(cur));
    CUDA_OK(cudaFree(pr));  CUDA_OK(cudaFree(sel));
    return g_fail;
}

/* Single-rank convenience: the full range owns every pair, so the ownership
 * predicate is inert -- which is exactly what makes the non-TP path
 * byte-identical to what it was before slice 4c. */
static int run_schedule(const std::vector<int32_t> &ids, uint32_t n_total,
                        uint32_t oob_code, Schedule &out) {
    return run_schedule(ids, n_total, oob_code, out, 0u, n_total);
}

/* (6) TP OWNERSHIP (slice 4c): with this rank owning [lo,hi), the schedule must
 * hold EXACTLY the pairs whose (clamped) expert is in range, each exactly once,
 * and every peer-owned expert's run must be EMPTY.  A peer-owned pair appearing
 * in a rank's partial would be summed twice by the all-reduce, which is the
 * whole correctness claim of the split. */
static void check_owned(const char *what, const std::vector<int32_t> &ids, uint32_t n_total,
                        uint32_t lo, uint32_t hi, const Schedule &s) {
    const uint32_t pair_count = (uint32_t)ids.size();
    std::vector<int> seen(pair_count, 0);
    for (uint32_t e = 0; e < n_total; e++) {
        if (!(e >= lo && e < hi)) {
            CHECK(s.offsets[e] == s.offsets[e + 1],
                  "%s: peer-owned expert %u has run [%u,%u), expected empty",
                  what, e, s.offsets[e], s.offsets[e + 1]);
            continue;
        }
        for (uint32_t p = s.offsets[e]; p < s.offsets[e + 1]; p++) {
            if (p >= pair_count) { CHECK(false, "%s: expert %u's run reaches %u, past pair_count %u",
                                         what, e, p, pair_count); return; }
            const uint32_t pair = s.pairs[p];
            CHECK(pair < pair_count, "%s: sorted_pairs[%u] = %u out of range", what, p, pair);
            if (pair >= pair_count) continue;
            const int32_t raw = ids[pair];
            const uint32_t clamped = (raw < 0 || (uint32_t)raw >= n_total) ? 0u : (uint32_t)raw;
            CHECK(clamped == e, "%s: pair %u (id %d -> expert %u) landed in expert %u's run",
                  what, pair, raw, clamped, e);
            seen[pair]++;
        }
    }
    for (uint32_t i = 0; i < pair_count; i++) {
        const int32_t raw = ids[i];
        const uint32_t clamped = (raw < 0 || (uint32_t)raw >= n_total) ? 0u : (uint32_t)raw;
        const int want = (clamped >= lo && clamped < hi) ? 1 : 0;
        CHECK(seen[i] == want, "%s: pair %u (expert %u) appears %d times, expected %d",
              what, i, clamped, seen[i], want);
    }
}

/* (2) the schedule is a tiling: run e covers exactly the pairs whose clamped id
 * is e, and the runs tile [0, pair_count). */
static void check_schedule(const char *what, const std::vector<int32_t> &ids, uint32_t n_total,
                           const Schedule &s) {
    const uint32_t pair_count = (uint32_t)ids.size();
    uint32_t total = 0;
    for (uint32_t e = 0; e < n_total; e++) {
        CHECK(s.offsets[e] == total, "%s: offsets[%u] = %u, expected %u (runs must be contiguous)",
              what, e, s.offsets[e], total);
        for (uint32_t p = s.offsets[e]; p < s.offsets[e + 1]; p++) {
            CHECK(p < pair_count, "%s: expert %u's run reaches %u, past pair_count %u",
                  what, e, p, pair_count);
            if (p >= pair_count) return;
            const uint32_t pair = s.pairs[p];
            CHECK(pair < pair_count, "%s: sorted_pairs[%u] = %u out of range", what, p, pair);
            if (pair >= pair_count) continue;
            const int32_t raw = ids[pair];
            const uint32_t clamped = (raw < 0 || (uint32_t)raw >= n_total) ? 0u : (uint32_t)raw;
            CHECK(clamped == e, "%s: pair %u (id %d -> expert %u) landed in expert %u's run",
                  what, pair, raw, clamped, e);
            total++;
        }
    }
    CHECK(s.offsets[n_total] == pair_count, "%s: offsets[%u] = %u, expected pair_count %u",
          what, n_total, s.offsets[n_total], pair_count);
    CHECK(total == pair_count, "%s: the runs cover %u pairs, expected %u", what, total, pair_count);
}

/* Every pair index must appear exactly once across the schedule. */
static void check_permutation(const char *what, const std::vector<int32_t> &ids, const Schedule &s) {
    std::vector<int> seen(ids.size(), 0);
    for (uint32_t p = 0; p < s.pairs.size(); p++) {
        if (s.pairs[p] >= seen.size()) { CHECK(false, "%s: pair index %u out of range", what, s.pairs[p]); continue; }
        seen[s.pairs[p]]++;
    }
    for (size_t i = 0; i < seen.size(); i++)
        CHECK(seen[i] == 1, "%s: pair %zu appears %d times in the schedule, expected 1", what, i, seen[i]);
}

static void expect_flag(const char *what, uint32_t want_layer, const char *want_arm, int want) {
    uint32_t layer = 0xFFFFFFFFu;
    const char *arm = NULL;
    const int got = pulsar_gpu_routed_moe_route_oob_take(&layer, &arm);
    CHECK(got == want, "%s: flag take returned %d, expected %d", what, got, want);
    if (want > 0) {
        CHECK(layer == want_layer, "%s: flag names layer %u, expected %u", what, layer, want_layer);
        CHECK(arm && want_arm && strcmp(arm, want_arm) == 0,
              "%s: flag names arm '%s', expected '%s'", what, arm ? arm : "(null)", want_arm);
    }
    /* An untaken flag would be charged to the NEXT step's bytes (the L188
     * hazard): after a successful take it must read clear. */
    if (want > 0) expect_flag(what, 0, NULL, 0);
}

int main(void) {
    const uint32_t n_total = 6;
    const uint32_t layer = 3;

    /* Control 1: every id in range.  Nothing may move and nothing may flag --
     * this is the bit-identity half of the guard. */
    {
        std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 0, 1};
        Schedule s;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), s)) return 1;
        check_schedule("in-range", ids, n_total, s);
        check_permutation("in-range", ids, s);
        const uint32_t want[6] = {2, 2, 1, 1, 1, 1};
        for (uint32_t e = 0; e < n_total; e++) CHECK(s.counts[e] == want[e], "in-range: counts[%u] = %u, expected %u", e, s.counts[e], want[e]);
        expect_flag("in-range", 0, NULL, 0);
        printf("in-range: 8 ids over 6 experts, no flag, schedule tiles\n");
    }

    /* Control 2: the router's designed NaN route.  -1 clamps to expert 0 and is
     * NOT a violation. */
    {
        std::vector<int32_t> ids = {0, -1, 2, 3, 4, 5, 0, 1};
        Schedule s;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), s)) return 1;
        check_schedule("negative", ids, n_total, s);
        check_permutation("negative", ids, s);
        CHECK(s.counts[0] == 3, "negative: counts[0] = %u, expected 3 (ids 0, -1, 0)", s.counts[0]);
        expect_flag("negative", 0, NULL, 0);
        printf("negative: -1 routed to expert 0 and did NOT flag\n");
    }

    /* The guard: ids one past the end and far past it.  Both must be clamped
     * (canaries + tiling) and the flag must name the layer and the arm. */
    {
        std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, (int32_t)n_total, 999};
        Schedule s;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), s)) return 1;
        check_schedule("over-range", ids, n_total, s);
        check_permutation("over-range", ids, s);
        CHECK(s.counts[0] == 3, "over-range: counts[0] = %u, expected 3 (id 0 + two clamped)", s.counts[0]);
        CHECK(s.counts[1] == 1, "over-range: counts[1] = %u, expected 1 (the in-range id 1, and ONLY it)"
                                " -- clamping must not disturb a valid neighbour", s.counts[1]);
        CHECK(s.offsets[n_total] == 8, "over-range: every pair must still be placed (offsets[%u] = %u)", n_total, s.offsets[n_total]);
        expect_flag("over-range", layer, "grouped CUTLASS MXFP4 count", 1);
        printf("over-range: id %u and 999 clamped, flag named layer %u / count arm\n", n_total, layer);
    }

    /* The scatter arm has its own tag, and it must flag on its own. */
    {
        std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 7, 1};
        Schedule s;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_SCATTER), s)) return 1;
        check_schedule("scatter-arm", ids, n_total, s);
        check_permutation("scatter-arm", ids, s);
        expect_flag("scatter-arm", layer, "grouped CUTLASS MXFP4 scatter", 1);
        printf("scatter arm: flagged by name\n");
    }

    /* The mixed lane's tags come from the same enum. */
    {
        std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 9, 1};
        Schedule s;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_MIXED_COUNT), s)) return 1;
        check_schedule("mixed-arm", ids, n_total, s);
        expect_flag("mixed-arm", layer, "mixed type-40/type-44 count", 1);
        printf("mixed arm: flagged by name\n");
    }

    /* A 256-expert stack with an id of exactly 256: the boundary case that a
     * REAP artifact at full width would hit first. */
    {
        std::vector<int32_t> ids(256, 0);
        ids[255] = 256;
        Schedule s;
        if (run_schedule(ids, 256, moe_route_oob_code(0, MOE_OOB_ARM_GROUPED_COUNT), s)) return 1;
        check_schedule("full-width", ids, 256, s);
        check_permutation("full-width", ids, s);
        CHECK(s.counts[0] == 256, "full-width: counts[0] = %u, expected 256", s.counts[0]);
        expect_flag("full-width", 0, "grouped CUTLASS MXFP4 count", 1);
        printf("full width: id 256 at n_total 256 clamped, not written past\n");
    }

    /* TP ownership (slice 4c): rank 1 of 2 owns [3,6), rank 0 owns [0,3).  Each
     * rank's schedule must hold exactly its own half -- and the two halves must
     * be DISJOINT and COMPLETE, which is the property the split's correctness
     * rests on once the all-reduce sums them. */
    {
        std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 3, 5, 0, 4, 2, 1};
        Schedule s1, s0, sf;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), s1, 3u, 6u)) return 1;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), s0, 0u, 3u)) return 1;
        if (run_schedule(ids, n_total, moe_route_oob_code(layer, MOE_OOB_ARM_GROUPED_COUNT), sf)) return 1;
        check_owned("owned-rank1", ids, n_total, 3u, 6u, s1);
        check_owned("owned-rank0", ids, n_total, 0u, 3u, s0);
        /* In-range ids only: neither rank may raise the route-bounds flag. */
        expect_flag("owned-rank1", 0, NULL, 0);
        expect_flag("owned-rank0", 0, NULL, 0);
        /* Disjoint + complete: every expert's full run is the sum of the two
         * halves', and the owned runs tile each rank's own partial. */
        uint32_t n1 = 0, n0 = 0;
        for (uint32_t e = 0; e < n_total; e++) {
            const uint32_t r1 = s1.offsets[e + 1] - s1.offsets[e];
            const uint32_t r0 = s0.offsets[e + 1] - s0.offsets[e];
            const uint32_t rf = sf.offsets[e + 1] - sf.offsets[e];
            CHECK(r1 + r0 == rf, "owned: expert %u's halves (%u + %u) do not sum to the full run %u",
                  e, r1, r0, rf);
            n1 += r1; n0 += r0;
        }
        CHECK(n1 + n0 == (uint32_t)ids.size(),
              "owned: the two partials cover %u + %u pairs, expected %zu", n1, n0, ids.size());
        uint32_t lo1 = 0;
        for (uint32_t e = 3; e < 6; e++) {
            CHECK(s1.offsets[e] == lo1, "owned-rank1: its owned runs do not tile [0,%u)", n1);
            lo1 = s1.offsets[e + 1];
        }
        CHECK(lo1 == n1, "owned-rank1: partial run ends at %u, expected %u", lo1, n1);
        printf("tp ownership: rank halves are disjoint and complete (%u + %u pairs)\n", n1, n0);
    }

    if (g_fail) { fprintf(stderr, "MOE-ROUTE-BOUNDS GATE FAIL\n"); return 1; }
    printf("MOE-ROUTE-BOUNDS GATE PASS\n");
    return 0;
}

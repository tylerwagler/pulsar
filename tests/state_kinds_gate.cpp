/* L295 F3: the declared-state gate -- the core's state kinds (src/engine/state_kinds.h) over declarations that are
 * no served family's.  Model-free: the device, no weights, seconds.
 *
 *   ./tests/state_kinds_gate          (make state-kinds-gate)
 *
 * PLAN A is an Inkling-Small-shaped stub (plan 120): 42 layers, the 7 global ones (5, 11, ..., 41) with a per-token
 * KV POOL (8 KV heads x 128 x K|V bf16 = 4 KiB a token), the 35 window ones with a RING of W - 1 + the verify width
 * slots (W = 512; L295 F1's block), and every layer's SCONV state -- kind (c), FIXED, 3 rows x (1024 + 1024 + 4096 +
 * 4096) bf16 = 60 KiB, the four short-conv streams (K, V, the attention delta, the MLP delta), the state after each
 * verify row captured (SNAPSHOT).  Graded: the sizes the declaration implies (60 KiB a layer = 2.5 MiB a sequence,
 * the ring, the pool rows), the dry price == the allocation, the demand-paged share == the pools' bytes, the
 * checkpoint slot's bytes and ORDER (slot by slot, decl by decl) and a walk round trip, the pools, reset, the
 * physical residency of a bank, and a SNAPSHOT commit (keep < R restores the kept row's state; keep == R leaves the
 * verify's).
 *
 * PLAN B grades the other captures on a small plan: a STAGED block (STAGE: the block restored and the kept rows of
 * the open block rewritten from their records), a DELTA state (the commit hook called at every keep with the kept
 * rows' updates), a host REPLAY state (restored and replayed over the kept tokens), and a drafter slot whose pool
 * trails (a payload's trailing pool, never a segment's, never evicted, never captured).  And refusals: a declaration
 * the core cannot run is refused by name at the create. */
#include "pulsar_engine_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static int n_fail;

static void check(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s  ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) n_fail++;
}

static std::vector<uint8_t> rd(pulsar_gpu_tensor *t, uint64_t off, uint64_t n) {
    std::vector<uint8_t> v(n);
    if (!pulsar_gpu_tensor_read(t, off, v.data(), n)) v.clear();
    return v;
}
static bool wr(pulsar_gpu_tensor *t, uint64_t off, const std::vector<uint8_t> &v) {
    return pulsar_gpu_tensor_write(t, off, v.data(), v.size()) != 0;
}
static std::vector<uint8_t> pattern(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    for (uint64_t i = 0; i < n; i++) v[i] = (uint8_t)(seed * 131u + i * 7u + (i >> 9));
    return v;
}
static bool all_zero(const std::vector<uint8_t> &v) {
    for (uint8_t b : v) if (b) return false;
    return !v.empty();
}

/* ---- plan A: Inkling-shaped ---------------------------------------------------------------------------------- */

enum { A_SCONV = 0, A_KV_GLOBAL, A_KV_WINDOW };
static constexpr uint64_t SCONV_BYTES = 3ull * (1024 + 1024 + 4096 + 4096) * 2;   /* 61440 */
static constexpr uint64_t KV_ROW = 2ull * 8 * 128 * 2;                           /* 4096 */
static constexpr uint32_t WINDOW = 512;

static void plan_inkling(pulsar_state_plan *p) {
    *p = pulsar_state_plan{};
    p->name = "inkling-stub";
    p->resume_grid = 128;
    p->split_invariant = true;
    p->ckpt_slots = 2;
    pulsar_state_plan_add(p, {"sconv", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_SNAPSHOT, false, false, 0, SCONV_BYTES,
                              0, 0, NULL, NULL, NULL});
    pulsar_state_plan_add(p, {"kv_global", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1, KV_ROW, 0, 0,
                              NULL, NULL, NULL});
    pulsar_state_plan_add(p, {"kv_window", PULSAR_STATE_RING, PULSAR_STATE_CAPTURE_NONE, false, false, WINDOW, KV_ROW,
                              0, 0, NULL, NULL, NULL});
    p->n_trunk = p->n_slots = 42;
    for (uint32_t il = 0; il < 42; il++)
        p->slot[il] = 1u << A_SCONV | (il % 6 == 5 ? 1u << A_KV_GLOBAL : 1u << A_KV_WINDOW);
}

static void plan_a(void) {
    fprintf(stderr, "state-kinds gate: plan A (Inkling-shaped: 7 global pools, 35 window rings, 42 SConv states)\n");
    static pulsar_state_plan p;
    plan_inkling(&p);
    const uint32_t banks = 2, ctx = 8190;   /* off a row boundary: the pools need no rounding at 1 token a row */
    const uint64_t ring = (uint64_t)(WINDOW - 1u + PULSAR_STATE_VERIFY_ROWS) * KV_ROW;
    check(pulsar_state_decl_bank_bytes(&p.decl[A_SCONV]) == 61440 && 42 * SCONV_BYTES / 1048576.0 > 2.45 &&
          42 * SCONV_BYTES / 1048576.0 < 2.47, "SConv: 60 KiB a layer, %.2f MiB a sequence (plan 120: 2.5 MiB)",
          42 * SCONV_BYTES / 1048576.0);
    check(pulsar_state_decl_bank_bytes(&p.decl[A_KV_WINDOW]) == ring, "the window ring holds W - 1 + %u slots: %llu B",
          PULSAR_STATE_VERIFY_ROWS, (unsigned long long)ring);
    /* the price is the create run dry */
    uint64_t dry = 0, managed = 0;
    pulsar_gpu_tensor_dry_begin();
    pulsar_state_set *d = pulsar_state_create(NULL, &p, NULL, banks, ctx, PULSAR_STATE_VERIFY_ROWS);
    if (d) pulsar_state_destroy(d);
    pulsar_gpu_tensor_dry_end(&dry, &managed);
    const uint64_t before = pulsar_gpu_tensor_alloc_bytes_current();
    pulsar_state_set *ss = pulsar_state_create(NULL, &p, NULL, banks, ctx, PULSAR_STATE_VERIFY_ROWS);
    check(ss != NULL, "the declaration builds (%u banks x %u tokens)", banks, ctx);
    if (!ss) return;
    const uint64_t got = pulsar_gpu_tensor_alloc_bytes_current() - before;
    check(dry > 0 && got == dry, "allocated %llu B == the dry price %llu B", (unsigned long long)got,
          (unsigned long long)dry);
    check(managed == (uint64_t)banks * 7 * ctx * KV_ROW, "demand-paged %llu B == banks x 7 global layers x ctx x 4 KiB",
          (unsigned long long)managed);
    check(pulsar_state_pool_bytes_at(ss, 1000) == 7ull * 1000 * KV_ROW, "a bank's 1000 touched tokens: %llu B of pool",
          (unsigned long long)pulsar_state_pool_bytes_at(ss, 1000));
    const uint64_t slot = 42 * SCONV_BYTES + 35 * ring;
    check(ss->ckpt.slot_bytes == ((slot + 255u) & ~255ull), "a checkpoint slot: every SConv state and ring (%llu B)",
          (unsigned long long)ss->ckpt.slot_bytes);
    pulsar_kv_pool pools[PULSAR_KV_POOLS_MAX];
    const uint32_t np = ss->ops.pools(ss, pools, PULSAR_KV_POOLS_MAX);
    bool pools_ok = np == 7;
    for (uint32_t i = 0; pools_ok && i < np; i++)
        pools_ok = pools[i].tokens_per_row == 1 && pools[i].row_bytes == KV_ROW &&
                   pools[i].rows == pulsar_state_pool(ss, 5 + 6 * i, A_KV_GLOBAL, 0);
    check(pools_ok, "the installed bank's pools: the 7 global layers' KV in layer order (%u)", np);
    check(ss->ops.trailing_pools(ss, pools, PULSAR_KV_POOLS_MAX) == 0, "no drafter slot: no trailing pool");

    /* the walk: bank 1 installed, every lane a distinct pattern, its slab order slot by slot, then a round trip */
    ss->live_bank = 1;
    bool ok = true;
    for (uint32_t k = 0; ok && k < 42; k++) {
        ok = wr(pulsar_state_tensor(ss, k, A_SCONV), SCONV_BYTES, pattern(SCONV_BYTES, 2 * k));
        if (ok && k % 6 != 5) ok = wr(pulsar_state_tensor(ss, k, A_KV_WINDOW), ring, pattern(ring, 2 * k + 1));
    }
    pulsar_gpu_tensor *slab = pulsar_gpu_tensor_alloc(ss->ckpt.slot_bytes);
    uint64_t walked = 0;
    ok = ok && slab && ss->ops.walk(ss, 0, false, slab, 0, 128, &walked) && pulsar_gpu_synchronize();
    check(ok && walked == slot, "the walk copies %llu B (the slot's)", (unsigned long long)walked);
    uint64_t off = 0;
    bool order = ok;
    for (uint32_t k = 0; order && k < 42; k++) {
        order = rd(slab, off, SCONV_BYTES) == pattern(SCONV_BYTES, 2 * k);
        off += SCONV_BYTES;
        if (order && k % 6 != 5) {
            order = rd(slab, off, ring) == pattern(ring, 2 * k + 1);
            off += ring;
        }
    }
    check(order, "the slot's ORDER: slot by slot, each slot's decls in id order (sconv, then its ring)");
    ok = ok && pulsar_state_reset_bank(ss, 1);
    check(ok && all_zero(rd(pulsar_state_tensor(ss, 7, A_SCONV), SCONV_BYTES, SCONV_BYTES)) &&
          all_zero(rd(pulsar_state_tensor(ss, 7, A_KV_WINDOW), ring, ring)) && ss->bank_pos[1] == 0,
          "a reset zeroes the bank's state and its counters");
    ok = ok && ss->ops.walk(ss, 1, false, slab, 0, 128, NULL) && pulsar_gpu_synchronize();
    check(ok && rd(pulsar_state_tensor(ss, 41, A_SCONV), SCONV_BYTES, SCONV_BYTES) == pattern(SCONV_BYTES, 82) &&
          rd(pulsar_state_tensor(ss, 40, A_KV_WINDOW), ring, ring) == pattern(ring, 81) &&
          all_zero(rd(pulsar_state_tensor(ss, 41, A_SCONV), 0, SCONV_BYTES)),
          "the walk back restores bank 1's lanes, and only bank 1's");
    pulsar_gpu_tensor_free(slab);

    /* physical residency: bank 0's pools go and come back; the rest stays */
    /* work/kv-vmm: rows are mapped before they are written -- 300 tokens of bank 0 are 7 pools' whole chunks */
    pulsar_state_set_pos(ss, 0, 300);
    const uint64_t chunk = pulsar_gpu_commit_chunk_bytes();
    const uint64_t mapped0 = pulsar_state_bank_committed_bytes(ss, 0);
    check(mapped0 == 0, "a fresh bank maps nothing (%llu B)", (unsigned long long)mapped0);
    const int32_t cpos[2] = {299, 120}, cbank[2] = {0, 1};
    /* each pool to its furthest row + 1 + the read slack (capped at the capacity), in whole chunks */
    const auto mapped_for = [&](uint64_t T) {
        T = T + PULSAR_STATE_POOL_READ_SLACK > ss->ctx ? ss->ctx : T + PULSAR_STATE_POOL_READ_SLACK;
        return 7ull * ((T * KV_ROW + chunk - 1) / chunk) * chunk;
    };
    check(pulsar_state_pools_cover(ss, cpos, cbank, 2) && pulsar_state_bank_committed_bytes(ss, 0) == mapped_for(300) &&
              pulsar_state_bank_committed_bytes(ss, 1) == mapped_for(121),
          "covering a step's rows maps each bank's pools to its furthest row: %llu / %llu B (chunks of %llu B)",
          (unsigned long long)pulsar_state_bank_committed_bytes(ss, 0),
          (unsigned long long)pulsar_state_bank_committed_bytes(ss, 1), (unsigned long long)chunk);
    /* gpuq #492: the driver does not refuse before the kernel's margin, so every commit is gated by the floor */
    {
        const uint64_t before = pulsar_state_bank_committed_bytes(ss, 1);
        pulsar_gpu_set_commit_floor(UINT64_MAX / 4);   /* no box has this much free */
        const int32_t far_pos[1] = {(int32_t)(ss->ctx - 1)}, far_bank[1] = {1};
        const bool refused = !pulsar_state_pools_cover(ss, far_pos, far_bank, 1);
        pulsar_gpu_set_commit_floor(0);
        check(refused && pulsar_state_bank_committed_bytes(ss, 1) == before,
              "a commit under the floor is refused before any create (bank 1 still %llu B mapped)",
              (unsigned long long)before);
    }
    std::vector<uint8_t> row = pattern(KV_ROW, 9);
    check(wr(pulsar_state_pool(ss, 5, A_KV_GLOBAL, 0), 299ull * KV_ROW, row) &&
              rd(pulsar_state_pool(ss, 5, A_KV_GLOBAL, 0), 299ull * KV_ROW, KV_ROW) == row,
          "a mapped row reads back what was written");
    pulsar_state_bank_free(ss, 0);
    check(pulsar_state_bank_evicted(ss, 0) && pulsar_state_bank_committed_bytes(ss, 0) == 0 &&
              pulsar_state_pool(ss, 5, A_KV_GLOBAL, 0) == NULL && !pulsar_state_bank_evicted(ss, 1) &&
              pulsar_state_bank_committed_bytes(ss, 1) > 0 && pulsar_state_tensor(ss, 5, A_SCONV) != NULL,
          "free_physical: bank 0's pools unmapped (0 B, no pool handed out), bank 1 whole");
    check(!pulsar_state_pools_cover_bank(ss, 0, 10), "an evicted bank's rows are refused, not mapped");
    check(pulsar_state_bank_alloc(ss, 0) && !pulsar_state_bank_evicted(ss, 0) && pulsar_state_bank_alloc(ss, 0),
          "alloc_physical re-backs it (idempotent)");

    /* a SNAPSHOT commit: a verify of two runs (bank 0 rows [0, 3) at 40, bank 1 rows [3, 8) at 100) */
    const uint32_t run_first[3] = {0, 3, 8};
    const int32_t bank[8] = {0, 0, 0, 1, 1, 1, 1, 1}, pos[8] = {40, 41, 42, 100, 101, 102, 103, 104};
    const int32_t tok[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    pulsar_state_verify_begin(ss, 2, run_first, bank, pos);
    ok = pulsar_state_verify_capture(ss);
    for (uint32_t r = 0; ok && r < 8; r++)   /* what the op writes: the state after each row, at its step row */
        ok = wr(pulsar_state_capture_tensor(ss, A_SCONV),
                pulsar_state_capture_offset(ss, 11, A_SCONV) + r * SCONV_BYTES, pattern(SCONV_BYTES, 500 + r));
    ok = ok && wr(pulsar_state_tensor(ss, 11, A_SCONV), SCONV_BYTES, pattern(SCONV_BYTES, 999));   /* the verify's end */
    ok = ok && pulsar_state_commit(ss, 1, tok + 3, 2) && pulsar_gpu_synchronize();
    check(ok && rd(pulsar_state_tensor(ss, 11, A_SCONV), SCONV_BYTES, SCONV_BYTES) == pattern(SCONV_BYTES, 504),
          "SNAPSHOT keep 2 of bank 1's 5 rows: its state is the capture of step row 4 (the run's second)");
    ok = wr(pulsar_state_tensor(ss, 11, A_SCONV), SCONV_BYTES, pattern(SCONV_BYTES, 999)) &&
         pulsar_state_commit(ss, 1, tok + 3, 5) && pulsar_gpu_synchronize();
    check(ok && rd(pulsar_state_tensor(ss, 11, A_SCONV), SCONV_BYTES, SCONV_BYTES) == pattern(SCONV_BYTES, 999),
          "SNAPSHOT keep 5 of 5: the verify's own state stays");
    check(!pulsar_state_commit(ss, 1, tok + 3, 6) && !pulsar_state_commit(ss, 3, tok, 1),
          "a commit past the run, or of a bank the verify did not carry, is refused");
    pulsar_state_destroy(ss);
}

/* ---- plan B: the other captures, a drafter slot, refusals --------------------------------------------------- */

enum { B_STAGE = 0, B_POOL, B_DELTA, B_HOST, B_TRAIL };
static constexpr uint64_t STAGE_ROW = 512, STAGE_REC = 2560, STAGE_OFF = 2048, DELTA_BYTES = 256, DELTA_UPD = 64;

static struct { int calls; uint32_t bank, keep; const void *rows; } g_delta;
static bool delta_commit(void *, uint32_t bank, const void *rows, uint32_t keep) {
    g_delta.calls++;
    g_delta.bank = bank;
    g_delta.keep = keep;
    g_delta.rows = rows;
    return true;
}
/* host state: the last two tokens (oldest first) -- what an n-gram context is */
static bool host_replay(const void *, void *state, const int32_t *tokens, uint32_t keep) {
    int32_t *c = (int32_t *)state;
    for (uint32_t i = 0; i < keep; i++) { c[0] = c[1]; c[1] = tokens[i]; }
    return true;
}
static void host_init(const void *, void *state) { ((int32_t *)state)[0] = ((int32_t *)state)[1] = -1; }

static void plan_others(pulsar_state_plan *p) {
    *p = pulsar_state_plan{};
    p->name = "captures-stub";
    p->resume_grid = 128;
    p->split_invariant = true;
    p->ckpt_slots = 1;
    pulsar_state_plan_add(p, {"stage", PULSAR_STATE_STAGED, PULSAR_STATE_CAPTURE_STAGE, false, false, 4, STAGE_ROW,
                              STAGE_REC, STAGE_OFF, NULL, NULL, NULL});
    pulsar_state_plan_add(p, {"pool", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 4, 256, 0, 0, NULL,
                              NULL, NULL});
    pulsar_state_plan_add(p, {"delta", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_DELTA, false, false, 0, DELTA_BYTES,
                              DELTA_UPD, 0, delta_commit, NULL, NULL});
    pulsar_state_plan_add(p, {"host", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_REPLAY, true, false, 0, 8, 0, 0, NULL,
                              host_replay, host_init});
    pulsar_state_plan_add(p, {"trail", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1, 128, 0, 0, NULL,
                              NULL, NULL});
    p->n_trunk = 3;
    p->n_slots = 4;
    p->slot[0] = 1u << B_DELTA;
    p->slot[1] = 1u << B_STAGE | 1u << B_POOL;
    p->slot[2] = 1u << B_DELTA;
    p->slot[3] = 1u << B_STAGE | 1u << B_TRAIL;   /* the drafter slot: no capture, a trailing pool */
    p->session = 1u << B_HOST;
}

static void plan_b(void) {
    fprintf(stderr, "state-kinds gate: plan B (STAGE, DELTA, REPLAY captures; a drafter slot; refusals)\n");
    static pulsar_state_plan p;
    plan_others(&p);
    pulsar_state_set *ss = pulsar_state_create(NULL, &p, NULL, 2, 1001, PULSAR_STATE_VERIFY_ROWS);
    check(ss != NULL, "the declaration builds");
    if (!ss) return;
    check(ss->ctx == 1004, "the pools' capacity: whole rows of every pool and block (1001 -> %u)", ss->ctx);
    check(((int32_t *)pulsar_state_host(ss, B_HOST, 1))[0] == -1, "a fresh bank's host state is its init");
    pulsar_kv_pool pools[8];
    check(ss->ops.pools(ss, pools, 8) == 1 && pools[0].tokens_per_row == 4 &&
          ss->ops.trailing_pools(ss, pools, 8) == 1 && pools[0].tokens_per_row == 1 && pools[0].row_bytes == 128,
          "the trunk's pool is a segment's; the drafter slot's trails");
    /* gpuq #497 (B5): the touched bytes are the trunk's -- what a spill frees; the drafter slot's pool stays mapped */
    {
        const int32_t pos[1] = {40}, bk[1] = {1};
        check(pulsar_state_pools_cover(ss, pos, bk, 1) && pulsar_state_bank_committed_bytes(ss, 1) > 0 &&
                  pulsar_gpu_tensor_committed_bytes(pulsar_state_pool(ss, 3, B_TRAIL, 1)) > 0,
              "a step maps the trunk's pool and the drafter slot's");
    }
    pulsar_state_bank_free(ss, 1);
    check(pulsar_state_bank_evicted(ss, 1) && pulsar_state_pool(ss, 3, B_TRAIL, 1) != NULL &&
              pulsar_gpu_tensor_committed_bytes(pulsar_state_pool(ss, 3, B_TRAIL, 1)) > 0 &&
              pulsar_state_bank_committed_bytes(ss, 1) == 0,
          "free_physical leaves a drafter slot's pool resident (mapped), and the bank's touched KV is 0");
    check(pulsar_state_bank_alloc(ss, 1), "and alloc_physical re-backs the trunk's");
    check(ss->cap_rows[B_DELTA] && ss->cap_rows[B_STAGE] && ss->cap_runs[B_STAGE] && ss->cap_host[B_HOST] &&
          !ss->cap_rows[B_TRAIL] && pulsar_gpu_tensor_bytes(ss->cap_runs[B_STAGE]) >= 1ull * 16 * 4 * STAGE_ROW,
          "captures for the trunk's decls only (the drafter slot's stage is not captured)");

    /* one verify: bank 1, rows [0, 6) at positions 9 .. 14 (the open block at 9 is [8, 12); keep 4 ends at 13) */
    const uint32_t run_first[2] = {0, 6};
    const int32_t bank[6] = {1, 1, 1, 1, 1, 1}, pos[6] = {9, 10, 11, 12, 13, 14}, tok[6] = {70, 71, 72, 73, 74, 75};
    const uint64_t blk = 4 * STAGE_ROW;
    int32_t *h = (int32_t *)pulsar_state_host(ss, B_HOST, 1);
    h[0] = 50;
    h[1] = 51;
    bool ok = wr(pulsar_state_tensor(ss, 1, B_STAGE), blk, pattern(blk, 7));   /* bank 1's block before the step */
    pulsar_state_verify_begin(ss, 1, run_first, bank, pos);
    ok = ok && pulsar_state_verify_capture(ss);
    h[0] = h[1] = 0;   /* the step advanced it */
    ok = ok && wr(pulsar_state_tensor(ss, 1, B_STAGE), blk, pattern(blk, 8));   /* ... and wrote the block */
    std::vector<uint8_t> recs = pattern(6 * STAGE_REC, 9);
    ok = ok && wr(pulsar_state_capture_tensor(ss, B_STAGE), pulsar_state_capture_offset(ss, 1, B_STAGE), recs);
    g_delta = {};
    ok = ok && pulsar_state_commit(ss, 1, tok, 4) && pulsar_gpu_synchronize();
    /* the block as the step found it, then the kept rows of the block open at 13 ([12, 16)): position 12 = row 3 */
    std::vector<uint8_t> want = pattern(blk, 7);
    memcpy(&want[(12 % 4) * STAGE_ROW], &recs[3 * STAGE_REC + STAGE_OFF], STAGE_ROW);
    check(ok && rd(pulsar_state_tensor(ss, 1, B_STAGE), blk, blk) == want,
          "STAGE keep 4: the block restored, the kept row of the block open at 13 rewritten from its record");
    const uint8_t *upd = (const uint8_t *)pulsar_gpu_tensor_device_ptr(ss->cap_rows[B_DELTA]);
    check(g_delta.calls == 2 && g_delta.bank == 1 && g_delta.keep == 4 &&
          g_delta.rows == upd + (uint64_t)1 * PULSAR_STATE_VERIFY_ROWS * DELTA_UPD,
          "DELTA: the commit applies each slot's kept updates (2 slots, the last at its capture rows)");
    check(h[0] == 72 && h[1] == 73, "REPLAY: the bank's host state before the step, replayed over the kept tokens (%d %d)",
          h[0], h[1]);
    g_delta = {};
    h[0] = h[1] = 0;
    ok = pulsar_state_commit(ss, 1, tok, 6) && pulsar_gpu_synchronize();
    check(ok && g_delta.calls == 2 && g_delta.keep == 6 && h[0] == 0,
          "keep == R: DELTA still applies (the verify left the state); nothing else moves");
    pulsar_state_destroy(ss);

    /* refusals, by name, at the create */
    pulsar_state_plan bad;
    plan_others(&bad);
    bad.decl[B_POOL].tokens = 3;
    check(!pulsar_state_create(NULL, &bad, NULL, 1, 64, 0), "a pool whose rows the resume grid does not close: refused");
    plan_others(&bad);
    bad.decl[B_STAGE].capture_off = STAGE_REC;
    check(!pulsar_state_create(NULL, &bad, NULL, 1, 64, 0), "a STAGE record without room for its row: refused");
    plan_others(&bad);
    bad.slot[0] |= 1u << B_HOST;
    check(!pulsar_state_create(NULL, &bad, NULL, 1, 64, 0), "host state on a layer slot: refused");
    plan_others(&bad);
    bad.decl[B_DELTA].commit = NULL;
    check(!pulsar_state_create(NULL, &bad, NULL, 1, 64, 0), "a DELTA without its commit: refused");
}

/* gpuq #504: the chunk pool (runtime POOL mode, once per process -- so last).  A small pool: the chunks mapped before
 * it are adopted, a step maps from the free list, an empty list refuses the step by name (-2, nothing created), a spill
 * returns the bank's chunks to the list (nothing released), and the rows map again from them. */
static void plan_pool(void) {
    fprintf(stderr, "state-kinds gate: the chunk pool\n");
    static pulsar_state_plan p;
    plan_inkling(&p);
    pulsar_state_set *ss = pulsar_state_create(NULL, &p, NULL, 2, 8190, PULSAR_STATE_VERIFY_ROWS);
    check(ss != NULL, "a 2-bank set for the pool");
    if (!ss) return;
    const uint64_t chunk = pulsar_gpu_commit_chunk_bytes();
    const int32_t first[1] = {99}, b0[1] = {0};
    check(pulsar_state_pools_cover(ss, first, b0, 1), "bank 0's first rows map before the pool (DIRECT)");
    const uint64_t adopted = pulsar_gpu_chunk_mapped_bytes() / chunk;
    char err[256] = "";
    check(pulsar_gpu_chunk_pool_reserve(7 * chunk, err, sizeof err) && pulsar_gpu_chunk_pool_on(),
          "the pool reserves 7 chunks (%s)", err[0] ? err : "ok");
    uint64_t total = 0, in_use = 0, free_c = 0;
    pulsar_gpu_chunk_pool_stats(&total, &in_use, &free_c, NULL);
    check(total == 7 + adopted && in_use == adopted && free_c == 7,
          "the %llu chunks mapped before it are the pool's: total %llu, in use %llu, free %llu",
          (unsigned long long)adopted, (unsigned long long)total, (unsigned long long)in_use,
          (unsigned long long)free_c);
    /* bank 1: 7 pools, one chunk each covers its first rows -- the list's 7 chunks */
    const int32_t one[1] = {99}, b1[1] = {1};
    check(pulsar_state_pools_cover(ss, one, b1, 1) && pulsar_gpu_chunk_pool_free_bytes() == 0,
          "bank 1 maps its 7 pools' first chunks from the list (free now %llu B)",
          (unsigned long long)pulsar_gpu_chunk_pool_free_bytes());
    const int32_t far[1] = {8000};
    check(!pulsar_state_pools_cover(ss, far, b1, 1), "an empty list refuses bank 1's next rows (nothing created)");
    pulsar_gpu_chunk_pool_stats(&total, &in_use, &free_c, NULL);
    check(total == 7 + adopted && free_c == 0, "the refusal created no chunk (total %llu)", (unsigned long long)total);
    pulsar_state_bank_free(ss, 1);
    pulsar_gpu_chunk_pool_stats(&total, &in_use, &free_c, NULL);
    check(free_c == 7 && total == 7 + adopted, "a spill returns bank 1's chunks to the list (free %llu, total %llu)",
          (unsigned long long)free_c, (unsigned long long)total);
    check(pulsar_state_bank_alloc(ss, 1) && pulsar_state_pools_cover(ss, one, b1, 1) &&
              pulsar_gpu_chunk_pool_free_bytes() == 0,
          "the readmit maps them again from the list");
    std::vector<uint8_t> row = pattern(KV_ROW, 11);
    check(wr(pulsar_state_pool(ss, 5, A_KV_GLOBAL, 1), 99ull * KV_ROW, row) &&
              rd(pulsar_state_pool(ss, 5, A_KV_GLOBAL, 1), 99ull * KV_ROW, KV_ROW) == row,
          "a row on a pooled chunk reads back what was written");
    pulsar_state_destroy(ss);
    pulsar_gpu_chunk_pool_stats(&total, &in_use, &free_c, NULL);
    check(free_c == total, "the set's destroy returns every chunk to the list (free %llu of %llu)",
          (unsigned long long)free_c, (unsigned long long)total);
}

int main(void) {
    if (!pulsar_gpu_init()) {
        fprintf(stderr, "state-kinds gate: no device\n");
        return 2;
    }
    plan_a();
    plan_b();
    plan_pool();
    if (n_fail) {
        fprintf(stderr, "STATE-KINDS GATE FAIL (%d)\n", n_fail);
        return 1;
    }
    fprintf(stderr, "STATE-KINDS GATE PASS\n");
    return 0;
}

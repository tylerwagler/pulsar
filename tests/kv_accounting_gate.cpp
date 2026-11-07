/* L357: the touched-KV accounting tracks the physical memory, for every family on the core step driver (Qwen first)
 * -- the intent of the DeepSeek-only tests/accounting_gate.cpp, through the public accounting API, without its graph
 * internals and without its rewind leg (a recurrent family has no rewind).
 *
 *   ./tests/kv_accounting_gate <model> [step_mib]
 *
 * pulsar_session_touched_kv_bytes / pulsar_session_bank_touched_kv_bytes are what the server's eviction guard
 * triggers on and budgets with: the bytes a bank's demand-paged pools (cudaMallocManaged, physical on touch) hold.
 * This gate measures them against the device's own count of its free memory (pulsar_gpu_mem_info: cudaMemGetInfo,
 * the device, never MemAvailable):
 *   K0  the fill is big enough to measure: each increment grows bank 0's pools by at least `step_mib` (default 1024)
 *       MiB by the engine's own price (pulsar_session_quantum_growth_bytes_per_bank), at least 4x the tolerance floor
 *   K1  bank 0 of a 2-bank session (pulsar_engine_set_bank_pool) syncs a warm level -- every distinct token of the
 *       tiled story seen, full prefill chunks run, so the one-time working set is in place -- then two increments of
 *       `step` tokens; at each, after the device's free count settles, the physical it lost exceeds the touched KV it
 *       gained by no more than max(TOL_FLOOR_MIB, TOL_REL of the gain) -- the direction that would under-budget the
 *       guard -- and the session's gain is bank 0's alone.  Physical may LAG the touched count: a prefill faults
 *       pages ahead of the rows it counts (2026-11-07 on Qwen: +655 MiB physical against +1050 touched for the first
 *       increment, +1026 for the second), so a shortfall is reported, and K2's release is the two-sided check
 *   K2  bank 1 installed, bank 0's physical freed (pulsar_session_bank_free_physical): its touched KV falls to 0, the
 *       session's falls by exactly what bank 0 held, and the device's free count rises by at least 3/4 of bank 0's
 *       trunk pools within 60 s (GB10 hands a freed managed range back asynchronously: bank_residency_gate R1) and
 *       by no more than them plus the tolerance.  A drafter slot's pools are never evicted (state_kinds.h), so the
 *       freed bytes are the trunk's -- read from the declared state (gate_fixture.h), the one reach past the API.
 * The increments CANCEL the session's eager floor (the fixed state, the checkpoint slabs, the scratch), so what
 * moves is the demand-paged pools.  Measured on the whole device: run it on a quiet box (the battery's discipline).
 * A runner gate (tests/gates_runner.cpp `kv-accounting-gate`). */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_fixture.h"   /* the declared state: the trunk's share of a bank's pools (K2) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

static const double MIB = 1024.0 * 1024.0;
#define TOL_FLOOR_MIB 256u      /* managed pages fault in at up to 2 MiB per (slot, decl) tensor, plus device slack */
#define TOL_REL       0.12      /* ... or this fraction of the measured gain, whichever is larger */
#define N_INC         2         /* increments measured after the warm level */
#define PRICE_ROWS    65536     /* the context the engine's per-bank price is read at, to size the step */

/* the tolerance for a gain of `gain` bytes */
static int64_t tol_for(int64_t gain) {
    const int64_t floor_b = (int64_t)TOL_FLOOR_MIB << 20, rel = (int64_t)(TOL_REL * (double)(gain < 0 ? -gain : gain));
    return floor_b > rel ? floor_b : rel;
}

/* the device's free bytes once they stop moving -- two reads a second apart within 8 MiB, for at most 30 s (another
 * session's freed range still returning would read as this fill's growth); *waited_s receives the wait */
static uint64_t settled_free(double *waited_s) {
    const double t0 = gate_now_s();
    (void)pulsar_gpu_synchronize();
    uint64_t prev = 0, now = 0, total = 0;
    pulsar_gpu_mem_info(&prev, &total);
    for (int i = 0; i < 30; i++) {
        usleep(1000 * 1000);
        pulsar_gpu_mem_info(&now, &total);
        const uint64_t moved = now > prev ? now - prev : prev - now;
        prev = now;
        if (moved < (8u << 20)) break;
    }
    *waited_s = gate_now_s() - t0;
    return prev;
}

/* bank `b`'s TRUNK pools at `rows` positions -- what free_physical returns (a drafter slot's pools stay) */
static uint64_t trunk_pool_bytes(const pulsar_session *s, uint64_t rows) {
    const pulsar_state_plan *p = s->states->plan;
    uint64_t bytes = 0;
    for (uint32_t k = 0; k < p->n_trunk; k++)
        for (uint32_t id = 0; id < p->n_decl; id++)
            if ((p->slot[k] >> id & 1u) && p->decl[id].kind == PULSAR_STATE_POOL)
                bytes += (rows + p->decl[id].tokens - 1u) / p->decl[id].tokens * p->decl[id].bytes;
    return bytes;
}

static int round_up(int v, int to) { return (v + to - 1) / to * to; }

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [step_mib]\n", argv[0]); return 2; }
    const int step_mib = argc > 2 ? atoi(argv[2]) : 1024;
    if (step_mib < 4 * (int)TOL_FLOOR_MIB) {
        fprintf(stderr, "kv-accounting: step_mib %d is under 4x the tolerance floor (%u MiB): it would measure noise\n",
                step_mib, TOL_FLOOR_MIB);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "kv-accounting: %s did not open\n", argv[1]); return 2; }
    pulsar_tokens base;
    if (!gate_load_story(e, &base, 256)) { gate_engine_close(e); return 2; }

    /* the levels: the engine's price of a bank's pools per position sizes a step of step_mib */
    const uint64_t price = pulsar_engine_demand_paged_bytes_per_bank(e, PRICE_ROWS).raw();
    if (price == 0) {
        fprintf(stderr, "kv-accounting: %s prices no demand-paged KV per bank\n", pulsar_engine_family_name(e));
        pulsar_tokens_free(&base);
        gate_engine_close(e);
        return 1;
    }
    const double per_row = (double)price / PRICE_ROWS;
    const int step = round_up((int)((double)step_mib * MIB / per_row) + 1, 4096);
    const int warm = round_up(base.len > 8192 ? base.len : 8192, 4096);
    int level[1 + N_INC];
    for (int i = 0; i <= N_INC; i++) level[i] = warm + i * step;
    const int peak = level[N_INC], ctx = peak + 4096;
    printf("kv-accounting: %s, %.1f B a position by the engine's price; warm %d, %d increments of %d tokens (ctx %d)\n",
           pulsar_engine_family_name(e), per_row, warm, N_INC, step, ctx);

    pulsar_engine_set_bank_pool(2);
    pulsar_session *s = NULL;
    int *toks = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, ctx) != 0 || !s || pulsar_session_bank_count(s) != 2 ||
        gate_state_slots(s) == 0) {
        CHECK(false, "a 2-bank session with declared state at ctx %d (%s)", ctx, pulsar_engine_family_name(e));
        if (s) pulsar_session_free(s);
        pulsar_tokens_free(&base);
        gate_engine_close(e);
        return 1;
    }
    /* K0 */
    const uint64_t step_price = pulsar_session_quantum_growth_bytes_per_bank(s, KvRows{(uint64_t)step}).raw();
    CHECK(step_price >= (uint64_t)step_mib << 20, "K0 an increment of %d tokens grows a bank's pools by %.1f MiB "
          "(want >= %d MiB, 4x the %u MiB tolerance floor at least)", step, step_price / MIB, step_mib, TOL_FLOOR_MIB);

    toks = (int *)malloc((size_t)peak * sizeof(int));
    bool ok = toks != NULL && pulsar_session_bank_state_restore(s, 0);
    for (int i = 0; ok && i < peak; i++) toks[i] = base.v[i % base.len];
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = toks;
    p.len = p.cap = warm;
    ok = ok && pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
    CHECK(ok, "K1 bank 0 warmed to %d tokens: %s", warm, err);

    /* K1 */
    double waited = 0;
    uint64_t free_prev = ok ? settled_free(&waited) : 0;
    uint64_t touched_prev = pulsar_session_touched_kv_bytes(s).raw();
    uint64_t bank_prev = pulsar_session_bank_touched_kv_bytes(s, 0).raw();
    printf("kv-accounting: warm baseline: device free %.1f MiB (settled in %.1f s), touched %.1f MiB (bank 0 %.1f)\n",
           free_prev / MIB, waited, touched_prev / MIB, bank_prev / MIB);
    for (int i = 1; ok && i <= N_INC; i++) {
        p.len = p.cap = level[i];
        err[0] = '\0';
        ok = pulsar_session_sync(s, &p, err, sizeof(err)) == 0;
        CHECK(ok, "K1 bank 0 synced to %d tokens: %s", level[i], err);
        if (!ok) break;
        const uint64_t free_now = settled_free(&waited);
        const uint64_t touched = pulsar_session_touched_kv_bytes(s).raw();
        const uint64_t bank_now = pulsar_session_bank_touched_kv_bytes(s, 0).raw();
        const int64_t dphys = (int64_t)free_prev - (int64_t)free_now;
        const int64_t dtouched = (int64_t)touched - (int64_t)touched_prev;
        const int64_t dbank = (int64_t)bank_now - (int64_t)bank_prev;
        const int64_t diff = dphys - dtouched, tol = tol_for(dtouched);
        CHECK(dbank == dtouched, "K1 %d -> %d: the session's touched KV grew by bank 0's alone (%.1f vs %.1f MiB)",
              level[i - 1], level[i], dtouched / MIB, dbank / MIB);
        CHECK(diff <= tol, "K1 %d -> %d: physical %+.1f MiB vs touched %+.1f MiB: %+.1f MiB (%+.1f%%), "
              "tolerance %.0f MiB (free settled in %.1f s)", level[i - 1], level[i], dphys / MIB, dtouched / MIB,
              diff / MIB, dtouched ? 100.0 * (double)diff / (double)dtouched : 0.0, tol / MIB, waited);
        free_prev = free_now;
        touched_prev = touched;
        bank_prev = bank_now;
    }

    /* K2 */
    if (ok) {
        pulsar_session_bank_state_save(s, 0);
        ok = pulsar_session_bank_state_restore(s, 1);
        CHECK(ok, "K2 bank 1 installed (bank 0 idle)");
    }
    if (ok) {
        const uint64_t t0 = pulsar_session_bank_touched_kv_bytes(s, 0).raw();
        const uint64_t all0 = pulsar_session_touched_kv_bytes(s).raw();
        const uint64_t trunk = trunk_pool_bytes(s, (uint64_t)peak);
        const uint64_t f0 = settled_free(&waited);
        const bool freed = pulsar_session_bank_free_physical(s, 0);
        const uint64_t t0_after = pulsar_session_bank_touched_kv_bytes(s, 0).raw();
        const uint64_t all1 = pulsar_session_touched_kv_bytes(s).raw();
        CHECK(freed && pulsar_session_bank_is_evicted(s, 0) && t0_after == 0 && all0 - all1 == t0 && t0 > 0,
              "K2 bank 0 freed: evicted, its touched KV %.1f -> %.1f MiB, the session's %.1f -> %.1f MiB", t0 / MIB,
              t0_after / MIB, all0 / MIB, all1 / MIB);
        /* GB10 returns a freed managed range asynchronously: watch the device's free count for up to 60 s */
        uint64_t f1 = f0, total = 0;
        const double w0 = gate_now_s();
        while (freed && gate_now_s() - w0 < 60.0 && (double)(f1 > f0 ? f1 - f0 : 0) < 0.75 * (double)trunk) {
            usleep(250 * 1000);
            uint64_t f = 0;
            pulsar_gpu_mem_info(&f, &total);
            if (f > f1) f1 = f;
        }
        const int64_t rise = (int64_t)f1 - (int64_t)f0;
        CHECK(freed && (double)rise >= 0.75 * (double)trunk && rise <= (int64_t)trunk + tol_for((int64_t)trunk),
              "K2 the device's free memory rose %.1f MiB within %.1f s: bank 0's trunk pools %.1f MiB (want >= 3/4 of "
              "them, <= them + %.0f MiB)", rise / MIB, gate_now_s() - w0, trunk / MIB, tol_for((int64_t)trunk) / MIB);
    }

    free(toks);
    pulsar_session_free(s);
    pulsar_tokens_free(&base);
    gate_engine_close(e);
    printf(n_fail ? "KV-ACCOUNTING GATE FAIL (%d)\n" : "KV-ACCOUNTING GATE PASS\n", n_fail);
    return n_fail != 0;
}

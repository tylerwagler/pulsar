/** @file
 *  pulsar-server's entry point: option parsing, the KV budget / memory-floor arithmetic, engine and session
 *  startup, the listener and the shutdown drain.
 */
#include "pulsar_server_internal.h"
#include "lib/pulsar_segstore.h"
#include "pulsar_lock.hpp"
#include "pulsar_gpu.h"
#include "pulsar_ctxmem.h"
#include "pulsar_grammar.h"
#include "lib/pulsar_writeback.h"

#include <string>

#include <malloc.h>



#ifndef PULSAR_SERVER_TEST   /* main() only: the harnesses that include this file run no server */
/* =========================================================================
 * Startup warmup generation (F1 unledgered-spike fix, task #32).
 *
 * The CUDA stack allocates a large first-generation working set LAZILY —
 * cuBLASLt workspaces, FP8/MXFP4 staging, GEMV activation buffers — on the
 * first prefill/decode that exercises each kernel family (measured on the
 * GB10, 2026-07-17: ~9.2 GiB, materializing in ~1.5 s on the first client
 * burst; see pulsar_server_internal.h, PULSAR_SERVER_PROCESS_OVERHEAD_BYTES).
 * Every MemAvailable-based admission check made before that burst is
 * therefore stale-high by that amount: the floor check legally admits
 * sessions whose room the first generation then consumes, and the box
 * craters straight through watchdog floors mid-request (F1 symptom B:
 * 6.3 -> 2.9 GiB in ~3 s with the session ledger clean).
 *
 * Running one throwaway generation here — a full prefill chunk plus a tail,
 * then a few tokens through the production decode path (speculative when
 * the drafter is loaded, so its buffers materialize too) — moves that cliff
 * to startup, BEFORE the listener opens and before the admission budget is
 * derived.  After this returns, MemAvailable is the truth every later
 * check needs it to be.
 *
 * Best-effort: a warmup failure is logged and serving continues (the cost
 * is only that the first real generation pays the materialization instead).
 * The session state is shrunk to a single token afterwards so the warmup
 * transcript can never win slot routing, hit the disk KV min-tokens floor,
 * or collide with a real conversation.
 *
 * L306: the warmup is also the CUDA-graph catalog's declaration window
 * (pulsar_gpu_seg_catalog_open): every graphed shape it runs -- the verify
 * curve's walk over every verify width, the drafter's sweep over its bank
 * counts and attention bounds -- is captured before the listener opens.  A
 * declared shape whose capture FAILED is not best-effort: it would refuse
 * every live step of that shape, so the server refuses to start (returns
 * false) and the log names the key. */
/// @brief Warm every graphed shape before the listener opens (the startup warmup the comment above describes).
/// @param engine the loaded engine
/// @param session the server's one session
/// @param ctx_size its context, in tokens (bounds the warmup prompt)
/// @return false when a declared shape's capture failed: the server refuses to start
static bool server_warmup_generation(pulsar_engine *engine, pulsar_session *session,
                                     int ctx_size) {
    const double t0 = server_now_sec();
    pulsar_gpu_seg_catalog_open();
    const HostBytes avail_before = server_mem_available_bytes();
    /* One full prefill-cap chunk plus a tail: exercises the widened chunk
     * prefill shape AND the sub-chunk tail shape.  The session's resolved
     * cap is used (the config value can be 0 = engine default). */
    const int prefill_cap = pulsar_session_prefill_cap(session).raw();
    int want = prefill_cap + 64;
    if (want < 64) want = 64;
    if (want > ctx_size - 256) want = ctx_size - 256;
    if (want < prefill_cap) {
        /* A small --ctx-size truncates the warmup below one full prefill
         * chunk: the chunked-prefill working set is NOT materialized here
         * (single-shot path only) and the measured budget will read
         * stale-high — still bounded by the static formula's min(). */
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: warmup prompt truncated to %d tokens "
                   "(< prefill cap %d): chunked-prefill workspaces are not "
                   "materialized at startup; first long prompt pays them",
                   want, prefill_cap);
    }
    pulsar_tokens body = {0}, prompt = {0};
    if (!pulsar_tokenize_text(engine, "The quick brown fox jumps over the lazy dog. ", &body))
        body.len = 0;   /* said by the engine: no warmup prompt, and the warmup below fails by name */
    if (body.len > 0 && want > 0) {
        while (prompt.len < want)
            for (int i = 0; i < body.len && prompt.len < want; i++)
                pulsar_tokens_push(&prompt, body.v[i]);
    }
    char err[256];
    err[0] = '\0';
    bool ok = prompt.len > 0 &&
              pulsar_session_sync(session, &prompt, err, sizeof(err)) == 0;
    int emitted = 0;
    if (ok) {
        uint64_t rng = 0x5eed5eed5eed5eedULL;
        TokId toks[17];
        while (emitted < 12) {
            /* whichever drafter runs behind the round API (L272 B12: Qwen's MTP warmed with argmax before) */
            if (pulsar_engine_has_spec_rounds(engine)) {
                const int n = pulsar_session_generate_speculative(
                        session, 0.0f, 0, 1.0f, 0.0f, &rng, 12 - emitted,
                        toks,
                        (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                /* Contract (session.cpp): eos arrives as a returned token; 0 means
                 * a rejected step (bad args / dirty multiseq state), not eos. */
                if (n <= 0) { ok = false; break; }
                emitted += n;
                if (pulsar_token_is_stop(engine, toks[n - 1])) break;
            } else {
                const TokId tok = pulsar_session_argmax(session);
                if (pulsar_token_is_stop(engine, tok)) break;
                if (pulsar_session_eval(session, tok, err, sizeof(err)) != 0) {
                    ok = false;
                    break;
                }
                emitted++;
            }
        }
    }
    /* L301: the spec lane's verify curve, on this bank at the warmup's frontier (the prompt's depth, the drafter
     * warm): loaded from the cache under $XDG_CACHE_HOME (else ~/.cache) /pulsar/verify-curve, or profiled -- every
     * verify width, ~6 forwards each -- and cached there.  On a pair (L315) the leader opens it, every rank running
     * the profile in step, and the leader's pick sends each round's choice to the workers.  A server with no batched
     * spec lane (no drafter, no bank pool) has none: its row budget alone admits drafts. */
    if (ok && pulsar_engine_has_spec_rounds(engine) &&
        pulsar_engine_has_fused_step(engine) && pulsar_session_bank_count(session) > 1) {
        char dir[4096] = "";
        const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
        if (xdg && xdg[0]) snprintf(dir, sizeof dir, "%s/pulsar/verify-curve", xdg);
        else if (home && home[0]) snprintf(dir, sizeof dir, "%s/.cache/pulsar/verify-curve", home);
        for (char *p = dir + 1; dir[0] && *p; p++) {   /* mkdir -p */
            if (*p != '/') continue;
            *p = '\0';
            (void)mkdir(dir, 0755);
            *p = '/';
        }
        if (dir[0] && mkdir(dir, 0755) != 0 && errno != EEXIST) {
            server_log(PULSAR_LOG_WARNING, "pulsar-server: verify curve cache %s unusable (%s): profiled, not cached",
                       dir, strerror(errno));
            dir[0] = '\0';
        }
        char cerr[256] = "";
        if (pulsar_session_verify_curve_open(session, dir, PULSAR_VERSION_STR, cerr, sizeof cerr) != 0)
            server_log(PULSAR_LOG_WARNING, "pulsar-server: no verify curve (%s): the spec lane's row budget alone "
                       "admits drafts", cerr);
    }
    /* Shrink the live state to one token: below every routing/storage
     * threshold, and the next real prompt rebuilds from zero anyway. */
    if (prompt.len > 0) {
        prompt.len = 1;
        char shrink_err[256];
        (void)pulsar_session_sync(session, &prompt, shrink_err,
                               sizeof(shrink_err));
    }
    pulsar_tokens_free(&prompt);
    pulsar_tokens_free(&body);
    const uint64_t graph_failed = pulsar_gpu_seg_catalog_close();
    const HostBytes avail_after = server_mem_available_bytes();
    if (ok) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: warmup generation: %d prompt + %d decode tokens "
                   "in %.1f s; MemAvailable %.2f -> %.2f GiB "
                   "(first-generation working set %.2f GiB materialized)",
                   want, emitted, server_now_sec() - t0,
                   (double)avail_before.raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)avail_after.raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)(avail_before > avail_after
                                ? avail_before - avail_after : HostBytes{}).raw() /
                       (1024.0 * 1024.0 * 1024.0));
    } else {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: warmup generation failed (%s); the first real "
                   "generation will pay the deferred CUDA allocations instead",
                   err[0] ? err : "empty warmup prompt");
    }
    if (graph_failed) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: %llu graphed shape(s) of the warmup catalog FAILED or refused their capture "
                   "(the L306 lines above name them) -- refusing to start",
                   (unsigned long long)graph_failed);
        return false;
    }
    return true;
}
#endif /* PULSAR_SERVER_TEST */



void server_warn_if_exposed(const char *host) {
    if (strcmp(host, "127.0.0.1") != 0 && strcmp(host, "localhost") != 0 && strcmp(host, "::1") != 0)
        server_log(PULSAR_LOG_WARNING, "pulsar-server: bound to %s with no authentication: anyone who can reach it can use it",
                   host);
}

/// @brief A strictly positive int option value; exits(2) with a log line on anything else.
/// @param s the value text
/// @param opt the option's name, for the message
/// @return the value
static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}



/* Non-negative int: a rank index (0 is the leader), which parse_int_arg's
 * strictly-positive rule would refuse; the engine checks 0 <= R < N at open. */
/// @param s the value text
/// @param opt the option's name, for the message
/// @return the value, >= 0 (exits(2) on anything else)
static int parse_index_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

/// @brief A float option value in [minv, maxv]; exits(2) with a log line on anything else.
/// @param s the value text
/// @param opt the option's name, for the message
/// @param minv the least value accepted
/// @param maxv the greatest value accepted
/// @return the value
static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}



/// @brief The value after option argv[*i]; exits(2) when there is none.
/// @param i the option's index, advanced to the value's
/// @param argc argument count
/// @param argv the arguments
/// @param opt the option's name, for the message
/// @return the value (argv storage)
static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}



#ifndef PULSAR_SERVER_TEST   /* main() only: the harnesses that include this file run no server */
/* Tier-2 overcommit (task #55). PULSAR_OVERCOMMIT switches the pool auto-size from
 * "N-bank pool must fit the budget in FULL" to "only the EAGER floor (raw ring +
 * state lanes + shared) for N banks must fit; the ctx-scaled comp/index are
 * VA-only, demand-paged, and NOT charged at admission". This lets a large --ctx
 * (e.g. the 1M default) come up with N>1 banks all 1M-CAPABLE, each paying only
 * for the KV it actually touches. DEFAULT ON as of v0.3.0: the increment-2
 * proactive eviction guard has landed (server_sched.cpp worker_batched_decode_quantum),
 * so banks that grow toward 1M are bounded by LRU-idle eviction before the
 * physical budget is breached. PULSAR_OVERCOMMIT=0/off/false reverts to classic
 * full-charge admission; PULSAR_MSEQ_BANKS still pins and bypasses auto-sizing.
 * Read once at startup, never on a hot path. */
/// @return false only when PULSAR_OVERCOMMIT is 0 / off / false
static bool server_overcommit_enabled(void) {
    const char *v = getenv("PULSAR_OVERCOMMIT");
    if (!v || !v[0]) return true; /* default ON (v0.3.0) */
    return !(v[0] == '0' || !strcasecmp(v, "off") || !strcasecmp(v, "false"));
}
#endif /* PULSAR_SERVER_TEST */




/* Admission-control budget (Tier 1 §1.4). The session budget is the
 * unified-memory headroom left for per-session GPU state after the shared
 * resident weights, a fixed process-overhead reserve, and the free-memory
 * floor (PULSAR_SERVER_MEM_FLOOR_BYTES) — committing the full budget must still
 * leave the floor free. Returns 0 if the reserves already exceed usable. */
/// @param weights_resident_bytes the resident weights, bytes (their host mapping: on unified memory the pool's)
/// @return the session budget in bytes (0 when the reserves exceed PULSAR_SERVER_USABLE_BYTES)
KvBytes server_kv_budget_bytes(HostBytes weights_resident_bytes) {
    const HostBytes usable{PULSAR_SERVER_USABLE_BYTES};
    const HostBytes reserved = weights_resident_bytes +
                               HostBytes{PULSAR_SERVER_PROCESS_OVERHEAD_BYTES} +
                               HostBytes{PULSAR_SERVER_MEM_FLOOR_BYTES};
    if (reserved >= usable) return KvBytes{};
    /* unified memory's formula (server_kv_budget_for_device): the box's one pool, what is left of it room for KV */
    return pulsar_kv_room_bytes(pulsar_unified_device_bytes(usable - reserved));
}

/* L338: a DISCRETE card's sessions live in its own memory, which server_kv_budget_bytes' constants (GB10's unified
 * 121 GiB and its process reserve) do not describe -- on a 24 GB RTX 6000 they budgeted 84.6 GiB, auto-sized 16
 * banks, and the create died out of device memory at bank 11.  There the budget is what the device has free once the
 * weights are staged, less server_discrete_margin_bytes of the working set (L347 I4: the estimate at sizing, checked
 * against the warmup's measurement before the listener opens). */
/// @param free_bytes        the device's free memory, the weights staged
/// @param working_set_bytes the working set the margin keeps (PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES at sizing)
/// @return the session budget in bytes (0 when the margin exceeds what is free)
KvBytes server_kv_budget_discrete_bytes(DeviceBytes free_bytes, DeviceBytes working_set_bytes) {
    const DeviceBytes margin = server_discrete_margin_bytes(working_set_bytes);
    return free_bytes > margin ? pulsar_kv_room_bytes(free_bytes - margin) : KvBytes{};
}

#ifndef PULSAR_SERVER_TEST   /* main() only: the harnesses that include this file run no server */
/* The budget for the device this process runs on: the discrete formula when the device says its memory is its own,
 * else (GB10, or a device that did not answer) the unified one. */
/// @param weights_resident_bytes the resident weights, bytes (the unified formula's)
/// @return the session budget in bytes
static KvBytes server_kv_budget_for_device(HostBytes weights_resident_bytes) {
    if (pulsar_gpu_integrated() == 0) {
        uint64_t free_b = 0, total_b = 0;
        pulsar_gpu_mem_info(&free_b, &total_b);
        return server_kv_budget_discrete_bytes(DeviceBytes{free_b},
                                               DeviceBytes{PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES});
    }
    return server_kv_budget_bytes(weights_resident_bytes);
}

/** L335 P6: the session's startup price -- the ONE authority for the figures pool sizing, the admission check, the
 * create's reconciliation, the ledger's per-bank charge and the guard's eager floor read.  Under overcommit the banked
 * cost is affine in N, banked(N) = shared + N x (eager + demand), and admission charges only shared + N x eager (the
 * demand-paged comp/index is VA, physical on touch); without it admission charges the full price.  Each method is the
 * expression main() computed inline before (the oc_* triple and its sums). */
struct server_session_price {
    KvBytes shared;            ///< overcommit: banked(1) less one bank's marginal; zero without overcommit
    KvBytes eager_per_bank;    ///< overcommit: one bank's eager floor, its marginal less the demand-paged part
    KvBytes demand_per_bank;   ///< overcommit: one bank's demand-paged comp/index (VA reserved, NOT charged)
    bool overcommit;           ///< admission charges the eager floor rather than the full price
    pulsar_engine *engine;     ///< the engine whose dry create is the full price
    int ctx;                   ///< the per-bank context priced, tokens
    /// @param n banks
    /// @return the full price of an n-bank pool: the create run dry (demand-paged VA at full size); 0 if unpriceable
    KvBytes full(uint32_t n) const { return pulsar_engine_session_cost_bytes_banked(engine, ctx, (int)n); }
    /// @param n banks
    /// @return what admission charges for an n-bank pool: shared + n x eager under overcommit, else full(n)
    KvBytes admission(uint32_t n) const { return overcommit ? shared + eager_per_bank.times(n) : full(n); }
};

/// @brief The session price at `ctx` (server_session_price); the overcommit split is measured only under overcommit.
/// @param engine the engine
/// @param ctx the per-bank context, tokens
/// @param overcommit admission charges the eager floor (server_overcommit_enabled)
/// @return the price
static server_session_price server_session_price_of(pulsar_engine *engine, int ctx, bool overcommit) {
    server_session_price p{KvBytes{}, KvBytes{}, KvBytes{}, overcommit, engine, ctx};
    if (overcommit) {
        const KvBytes banked1 = p.full(1);
        const KvBytes banked2 = p.full(2);
        const KvBytes marginal = banked2 > banked1 ? banked2 - banked1 : KvBytes{};
        p.demand_per_bank = pulsar_engine_demand_paged_bytes_per_bank(engine, ctx);
        p.eager_per_bank = marginal > p.demand_per_bank ? marginal - p.demand_per_bank : KvBytes{};
        p.shared = banked1 > marginal ? banked1 - marginal : KvBytes{};
    }
    return p;
}
#endif /* PULSAR_SERVER_TEST */

/* Admission rule: the already-committed live-session cost plus this request's
 * estimated cost must fit under the budget. Gates the startup session here and
 * every bank provisioning in the scheduler (provision_bank's ledger guard,
 * server::ledger_admits_bank, L391); an over-budget provisioning attempt leaves
 * the job queued until an eviction releases a charge (plan Tier 1 §1.4).
 * Non-static: the scheduler calls it; unit tests live in this TU. */
/// @param kv_budget_bytes the admission budget, bytes
/// @param committed_bytes the live sessions' committed cost, bytes
/// @param incoming_bytes the new session's estimated cost, bytes
/// @return true when committed + incoming fits the budget (overflow-safe)
bool server_kv_admits(KvBytes kv_budget_bytes, KvBytes committed_bytes, KvBytes incoming_bytes) {
    if (incoming_bytes > kv_budget_bytes) return false;      /* guards overflow + lone fit */
    return committed_bytes <= kv_budget_bytes - incoming_bytes;
}

/* The headroom predicate every floor check is built on: the reading must keep
 * the floor after `need` more bytes.  The floor is kernel/OS breathing room
 * only — process-fixed costs live in the overhead reserve the ledger already
 * subtracted (see PULSAR_SERVER_MEM_FLOOR_BYTES in pulsar_server_internal.h for
 * the 2026-07-15 sizing).  avail == 0 means the reading was unreadable: fail
 * closed — this guard exists precisely for when other accounting is wrong, so
 * it must not silently disarm itself.  A KV need is asked through
 * server_kv_headroom_admits, which reads the memory the KV occupies (L391). */
/// @brief server_headroom_admits on one memory's unit (HostBytes or DeviceBytes): the two overloads' one body.
/// @param avail the reading (0 = unreadable)
/// @param need the bytes about to be taken
/// @param floor the floor
/// @return true when the floor stays
template <class B>
static bool server_headroom_admits_in(B avail, B need, B floor) {
    if (avail == B{} || need > B::max() - floor) return false;
    return avail >= need + floor;
}

bool server_headroom_admits(HostBytes avail, HostBytes need, HostBytes floor) {
    return server_headroom_admits_in(avail, need, floor);
}

bool server_headroom_admits(DeviceBytes avail, DeviceBytes need, DeviceBytes floor) {
    return server_headroom_admits_in(avail, need, floor);
}

bool server_kv_headroom_admits(const server_mem_reading &m, KvBytes kv_need) {
    const DeviceBytes need = pulsar_kv_device_bytes(kv_need);   /* the KV lives in the device's memory */
    if (!m.discrete) return server_headroom_admits(m.kv_avail, need, m.kv_floor);
    if (!server_headroom_admits(m.host_avail, HostBytes{}, m.host_floor)) return false;
    return kv_need == KvBytes{} || server_headroom_admits(m.kv_avail, need, m.kv_floor);
}

KvBytes server_bank_provision_demand(bool growth_guarded, KvBytes kv_bank_bytes) {
    return growth_guarded ? KvBytes{} : kv_bank_bytes;
}

KvBytes server_bank_charge(KvBytes admitted, uint32_t priced_banks, uint32_t bank) {
    if (priced_banks == 0 || bank >= priced_banks) return KvBytes{};
    return KvBytes{admitted.raw() / priced_banks + (bank < admitted.raw() % priced_banks ? 1u : 0u)};
}

KvBytes server_bank_charges(KvBytes admitted, uint32_t priced_banks, uint32_t from, uint32_t to) {
    KvBytes sum{};
    for (uint32_t b = from; b < to && b < priced_banks; b++) sum += server_bank_charge(admitted, priced_banks, b);
    return sum;
}

bool server_ledger_self_check(KvBytes admitted, uint32_t priced_banks, uint32_t chat_banks, KvBytes reserved,
                              KvBytes budget, char *err, size_t errlen) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    if (admitted == KvBytes{} || chat_banks == 0 || chat_banks > priced_banks) {
        snprintf(err, errlen, "a pool of %u chat bank(s) priced at %llu bytes over %u bank(s) is no pool", chat_banks,
                 (unsigned long long)admitted.raw(), priced_banks);
        return false;
    }
    const KvBytes all_live = reserved + server_bank_charges(admitted, priced_banks, 0, chat_banks);
    if (all_live != admitted) {
        snprintf(err, errlen,
                 "with every chat bank live the ledger holds %llu bytes (%llu reserved + %u chat bank charges) but "
                 "the pool was admitted at %llu over %u bank(s)",
                 (unsigned long long)all_live.raw(), (unsigned long long)reserved.raw(), chat_banks,
                 (unsigned long long)admitted.raw(), priced_banks);
        return false;
    }
    if (!server_kv_admits(budget, KvBytes{}, admitted)) {
        snprintf(err, errlen, "the pool's admitted price %.2f GiB is past the budget %.2f GiB",
                 (double)admitted.raw() / gib, (double)budget.raw() / gib);
        return false;
    }
    return true;
}

/* The one floor (pulsar_server_internal.h, "The memory floor"): set once at startup, before the listener opens and
 * the worker starts, and read-only after that. */
/// The memory floor in bytes (server_mem_floor / server_mem_floor_init); main thread writes it before any other
/// thread starts.
static HostBytes g_server_mem_floor{PULSAR_SERVER_MEM_FLOOR_BYTES};

HostBytes server_mem_floor(void) { return g_server_mem_floor; }

HostBytes server_mem_floor_init(HostBytes zone_high_bytes, HostBytes step_bytes) {
    g_server_mem_floor = HostBytes{PULSAR_SERVER_MEM_FLOOR_BYTES} + zone_high_bytes + step_bytes;
    return g_server_mem_floor;
}

HostBytes server_floor_step_bytes(bool discrete, KvBytes chunk_kv, HostBytes persist) {
    /* a chunk's KV pages are host memory only on unified memory, where the device allocates from the host's pool */
    if (discrete) return persist;
    return pulsar_unified_host_bytes(pulsar_kv_device_bytes(chunk_kv)) + persist;
}

HostBytes server_zoneinfo_high_bytes(const char *zoneinfo, uint64_t page_bytes) {
    uint64_t pages = 0;
    bool any = false;
    for (const char *p = zoneinfo; p && *p;) {
        const char *eol = strchr(p, '\n');
        unsigned long long v = 0;
        char tail = 0;
        /* a zone's watermark line, "        high     497208" (pages): the word alone, never a per-cpu "high:" */
        if (sscanf(p, " high %llu%c", &v, &tail) == 2 && tail == '\n') {
            pages += v;
            any = true;
        }
        if (!eol) break;
        p = eol + 1;
    }
    return HostBytes{any ? pages * page_bytes : 0};   /* the kernel's reserve enters the floor's unit here */
}

HostBytes server_zone_high_watermark_bytes(void) {
    FILE *fp = fopen("/proc/zoneinfo", "r");
    if (!fp) return HostBytes{};
    std::string text;
    char line[512];
    while (fgets(line, sizeof line, fp)) text += line;
    fclose(fp);
    const long pg = sysconf(_SC_PAGESIZE);
    return server_zoneinfo_high_bytes(text.c_str(), pg > 0 ? (uint64_t)pg : 4096u);
}

HostBytes server_persist_host_bytes(int tokens) {
    const uint64_t t = tokens > 0 ? (uint64_t)tokens : 0u;
    return HostBytes{PULSAR_WRITEBACK_STEP + 2u * t * PERSIST_TEXT_BYTES_PER_TOKEN};
}

/* L319: the resident-KV ceiling every touched-KV decision keys on (the decode guard's spills, L280's prompt
 * verdict, L314's preempt / readmit): the demand-paged KV the box can hold, measured -- what was touched at the
 * post-warmup reading plus that reading's MemAvailable above the floor -- and never more than the ledger's share
 * (budget less the eager floor).  Before L319 it was the ledger's share alone, and the ledger's budget is not a
 * physical number in pooled mode (its startup clamp lifts it to slot 0 + one session estimate): on Nemotron at
 * --ctx 1M the ceiling was 9.05 GiB against 5.16 GiB of headroom, the guard never spilled, MemAvailable reached
 * zero, and the GPU stalled on unbacked pages (a production box, 2026-10-22) or earlyoom killed the server (the GB10 host).  avail 0
 * (/proc/meminfo unreadable) is 0: the caller refuses to start rather than run unguarded. */
/// @param kv_budget_bytes the ledger's budget
/// @param eager_bytes the eager floor already resident
/// @param touched_bytes the KV resident now
/// @param avail_bytes the headroom reading now, MemAvailable or the device's free memory (0 = unreadable)
/// @param floor the floor that reading keeps free (server_guard_floor)
/// @return the resident-KV budget in bytes; 0 when the reading is at or under the floor
KvBytes server_guard_touched_budget(KvBytes kv_budget_bytes, KvBytes eager_bytes, KvBytes touched_bytes,
                                    DeviceBytes avail_bytes, DeviceBytes floor) {
    if (avail_bytes <= floor) return KvBytes{};
    const KvBytes ledger = kv_budget_bytes > eager_bytes ? kv_budget_bytes - eager_bytes : KvBytes{};
    const KvBytes headroom = pulsar_kv_room_bytes(avail_bytes - floor);
    const KvBytes physical = touched_bytes > KvBytes::max() - headroom ? KvBytes::max() : touched_bytes + headroom;
    return ledger < physical ? ledger : physical;
}



DeviceBytes server_guard_avail_bytes(bool discrete) {
    if (!discrete) return pulsar_unified_device_bytes(server_mem_available_bytes());   /* unified: one pool */
    uint64_t free_b = 0, total_b = 0;
    pulsar_gpu_mem_info(&free_b, &total_b);
    return DeviceBytes{free_b};   /* the device's reading enters its unit here */
}

DeviceBytes server_guard_floor(bool discrete) {
    return discrete ? DeviceBytes{PULSAR_SERVER_DISCRETE_GUARD_FLOOR_BYTES}
                    : pulsar_unified_device_bytes(server_mem_floor());   /* unified: the host's floor is the pool's */
}

void server::close_resources() {
    auto *s = this;
    server_judge_close(s);   /* L338 P2.4 */
    for (int i = 0; i < s->tp_rank_json_n; i++) free(s->tp_rank_json[i]);
    free(s->tp_rank_json);
    free(s->tp_rank_ids);
    s->tp_rank_json = NULL;
    s->tp_rank_ids = NULL;
    s->tp_rank_json_n = 0;
    s->tp = NULL;
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    if (s->capture) {
        fclose(s->capture);
        s->capture = NULL;
    }
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_mutex_destroy(&s->capture_mu);
    pthread_cond_destroy(&s->stream_cv);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    /* The ONE session (server.sess — classic mode == slot 0 alone, pool mode
     * == N banks over it) is freed exactly once here; slots are pure bank
     * descriptors and own nothing to free. */
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) {
        live_tool_state_free(&s->slots[i].responses_live);
        live_tool_state_free(&s->slots[i].anthropic_live);
        s->slot_terminal_clear(&s->slots[i]);   /* L380: a yielded checkpoint's kept turn */
        s->slots[i].provisioned = false;
    }
    if (s->sess) pulsar_session_free(s->sess);
    pulsar_grammar_compiler_free(s->grammar);   /* L303: the grammar compiler and its token table */
    s->grammar = NULL;
    delete s->token_bytes;
    s->token_bytes = NULL;
    s->sess = NULL;
    pulsar_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}



/// @brief Print pulsar-server's help (PULSAR_HELP_SERVER), or one topic of it.
/// @param fp    the stream (stdout for --help, stderr after a bad option)
/// @param topic the topic asked (NULL: the whole help)
static void usage(FILE *fp, const char *topic) {
    pulsar_help_print(fp, PULSAR_HELP_SERVER, topic);
}



/// @return the backend the server opens by default (CUDA)
static pulsar_backend default_server_backend(void) {
    return PULSAR_BACKEND_CUDA;
}



/* Default gguf resolution, in order: the project gguf/ directory (canonical
 * production filename, then the generic name), the current directory, then
 * $PULSAR_MODEL_DIR/<canonical> when that env var is set.  No store path is
 * baked into the binary.  Returns NULL if nothing readable — callers decide
 * whether that is fatal (main model) or a warning (drafter).  Heap results
 * intentionally live for the process lifetime (they become engine paths). */
/// @brief "dir/name" when that path is readable.
/// @param dir the directory
/// @param name the entry
/// @return a malloc'd path (process lifetime), or NULL
static const char *resolve_gguf_at(const char *dir, const char *name) {
    size_t n = strlen(dir) + 1 + strlen(name) + 1;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/%s", dir, name);
    if (access(p, R_OK) == 0) return p;
    free(p);
    return NULL;
}

/* Naming convention: an ACTIVE-POINTER symlink selects what a bare
 * `pulsar-server` runs (the drafter ships merged in the checkpoint, so there is
 * no separate dspark pointer).  Deploy = repoint the symlink.
 *
 * The pointer used to be `gguf/model.gguf`, and this looked for a GGUF.  The
 * engine reads a safetensors checkpoint -- a shard directory or a single file --
 * so the pointer is `model` (or `model.safetensors`) beside the working
 * directory, then in $PULSAR_MODEL_DIR.  The old name is deliberately NOT kept
 * as a fallback: resolving a GGUF would only defer the refusal to the engine,
 * which names the format and what to do about it. */
/// @return the default model pointer path (./model, ./model.safetensors, then the same under $PULSAR_MODEL_DIR),
/// or NULL when none is readable
static const char *resolve_default_model(void) {
    static const char *const pointers[] = { "model", "model.safetensors" };
    const char *dir = getenv("PULSAR_MODEL_DIR");
    for (size_t i = 0; i < sizeof(pointers) / sizeof(pointers[0]); i++) {
        const char *p;
        if (access(pointers[i], R_OK) == 0) return pointers[i];
        if (dir && dir[0] && (p = resolve_gguf_at(dir, pointers[i])) != NULL) return p;
    }
    return NULL;
}

/* Default directory for the disk KV cache, which is ON by default: the
 * conversation-bounce restore (sub-second vs full re-prefill TTFT) must not
 * depend on remembering --kv-disk-dir.  Placement follows XDG:
 * $XDG_CACHE_HOME/pulsar/kv-<model>, else ~/.cache/pulsar/kv-<model>.  The user
 * cache dir is chosen over <dirname(model)>/ because the model store may be
 * read-only or a network mount, while checkpoints are latency-sensitive,
 * regenerable, per-user state.
 *
 * <model> keys the directory by model identity.  The store header only
 * self-validates the model VARIANT (Flash/Pro, pulsar_engine_model_id) and the
 * routed-expert quant bits — NOT the exact weight artifact — so two different
 * ggufs of the same shape sharing a directory could cross-restore each
 * other's checkpoints.  gguf/model.gguf is an ACTIVE-POINTER symlink by
 * convention; its realpath basename is the versioned artifact name and
 * therefore the right identity key.
 *
 * L280: --rope-yarn F suffixes the key with "-rope-yarnF".  A segment's file
 * name is the hash of its TEXT, so a native and a YaRN server sharing one
 * directory would write the same names for the same conversation (each
 * renaming over the other's file, each under its own budget); with their own
 * directories both profiles keep their stores across a restart that switches
 * them.  The store identity (pulsar_engine_kv_identity) still guards an
 * explicit --kv-disk-dir both name.  Returns a heap path (process lifetime),
 * or NULL when no cache home can be resolved. */
/// @param model_path the model path (its basename, realpath-resolved, keys the directory)
/// @param rope_yarn_factor the --rope-yarn factor (non-zero: the key gets a "-rope-yarnF" suffix)
/// @return the default cache directory (heap, process lifetime), or NULL
static char *server_default_kv_disk_dir(const char *model_path, double rope_yarn_factor) {
    char resolved[PATH_MAX];
    const char *src = model_path ? model_path : "";
    if (model_path && realpath(model_path, resolved)) src = resolved;
    const char *base = strrchr(src, '/');
    base = base ? base + 1 : src;
    size_t len = strlen(base);
    if (len > 5 && !strcasecmp(base + len - 5, ".gguf")) len -= 5;
    char key[128];
    if (len == 0) {
        strcpy(key, "model");
    } else {
        if (len > sizeof(key) - 32) len = sizeof(key) - 32;   /* room for the rope suffix below */
        for (size_t j = 0; j < len; j++) {
            unsigned char ch = (unsigned char)base[j];
            key[j] = (isalnum(ch) || ch == '.' || ch == '_' || ch == '-') ?
                     (char)ch : '_';
        }
        key[len] = '\0';
    }
    if (rope_yarn_factor != 0.0) {
        const size_t at = strlen(key);
        snprintf(key + at, sizeof(key) - at, "-rope-yarn%g", rope_yarn_factor);
    }
    buf b = {0};
    /* Per the XDG spec a non-absolute XDG_CACHE_HOME is ignored (a relative
     * value would key the cache off whatever the process cwd happens to be). */
    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] == '/') {
        buf_printf(&b, "%s/pulsar/kv-%s", xdg, key);
    } else {
        const char *home = getenv("HOME");
        if (!home || !home[0]) return NULL;
        buf_printf(&b, "%s/.cache/pulsar/kv-%s", home, key);
    }
    return buf_take(&b);
}



/* Resolve the effective disk KV cache directory after option parsing.
 * Explicit --kv-disk-dir PATH behaves exactly as before; --kv-disk-dir "" or
 * --no-kv-disk opts out; otherwise the default above is filled in.  An
 * unresolvable or unusable directory must never be fatal — the server runs
 * with the disk cache disabled (kv_cache_open handles the unusable case). */
/// @param c the parsed config; kv_disk_dir is filled (NULL = disk cache off)
void server_resolve_kv_disk_dir(server_config *c) {
    if (c->kv_disk_disable) {
        c->kv_disk_dir = NULL;
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: disk KV cache disabled by flag");
        return;
    }
    if (c->kv_disk_dir) return; /* explicit path: unchanged behavior */
    c->kv_disk_dir = server_default_kv_disk_dir(c->engine.model_path, c->engine.rope_yarn_factor);
    if (c->kv_disk_dir) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache default dir %s "
                   "(--kv-disk-dir PATH overrides; --no-kv-disk disables)",
                   c->kv_disk_dir);
    } else {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache disabled (neither XDG_CACHE_HOME "
                   "nor HOME is set, no default directory)");
    }
}



/// @brief Parse pulsar-server's command line into a config over the defaults; --help prints usage and exits(0), an
/// unknown or malformed option exits(2).
/// @param argc argument count
/// @param argv the arguments
/// @return the config (its strings point into argv or static storage)
server_config server_parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "model",
            .backend = default_server_backend(),
        },
        .host = "127.0.0.1",   /* L354: no auth, so exposing it is a choice: --host */
        .port = 8000,
        /* v0.3.0 default: 1M context with overcommit ON (see
         * server_overcommit_enabled). Every session is grow-to-1M-capable, but
         * the ctx-scaled KV is demand-paged, so a short session pays only for
         * what it touches — admission charges just the eager floor (~4 banks come
         * up at ~6.4 GiB, NOT 4x the 1M worst case), and the increment-2 eviction
         * guard bounds banks that actually grow toward 1M. An operator who wants
         * a hard per-session cap passes a smaller --ctx; PULSAR_OVERCOMMIT=0 reverts
         * to classic full-charge admission (1M => N=1 single session). */
        .ctx_size = 1048576,
        .default_tokens = 393216,
    };
    c.kv_disk_space_mb = 131072;   /* L354: a default that bounds it; an explicit 0 is unbounded */
    c.judge_banks = -1;   /* L338 P2.4: one reserved bank when the container carries an adapter */
    c.reasoning_default = -1;   /* L338 P4: the family's default unless --reasoning */
    c.kv_cache = kv_cache_default_options();
    c.kv_ram_mb = -1;   /* L347 I3a: kv_ram_plan_for's default unless --kv-ram-mb */

    bool directional_steering_scale_set = false;
    /* Did -m/--model name a path?  The default below is a POINTER name, and
     * resolving it is skippable only if the user said nothing -- keying that on
     * a filename literal is what this replaced. */
    bool defaulted_model_path = true;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            const char *m = need_arg(&i, argc, argv, arg);
            if (c.n_models == SERVER_MODELS_MAX) {
                fprintf(stderr, "pulsar-server: at most %d models (-m) in one process\n", SERVER_MODELS_MAX);
                exit(2);
            }
            c.models[c.n_models++] = m;
            c.engine.model_path = c.models[0];
            defaulted_model_path = false;
        } else if (!strcmp(arg, "--no-dspark")) {
            c.engine.dspark_disable = true;
        } else if (!strcmp(arg, "--drafter")) {
            c.engine.drafter = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--reasoning")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "on")) c.reasoning_default = 1;
            else if (!strcmp(v, "off")) c.reasoning_default = 0;
            else {
                fprintf(stderr, "pulsar-server: --reasoning takes on or off\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--judge-banks")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (strcmp(v, "0") && strcmp(v, "1")) {
                fprintf(stderr, "pulsar-server: --judge-banks takes 0 or 1\n");
                exit(2);
            }
            c.judge_banks = v[0] - '0';
        } else if (!strcmp(arg, "--rope-yarn")) {
            c.engine.rope_yarn_factor = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 1.0f, 64.0f);
        } else if (!strcmp(arg, "--spec-depth")) {
            c.engine.spec_depth = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--spec-tau")) {
            const char *v = need_arg(&i, argc, argv, arg);
            const float t = strcmp(v, "off") ? parse_float_arg(v, arg, 0.0f, 1.0f) : 0.0f;
            c.engine.spec_tau = t > 0.0f ? t : -1.0f;   /* 0 / off: no stop */
        } else if (!strcmp(arg, "--tp-role")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "leader")) c.engine.tp_role = 1;
            else if (!strcmp(v, "worker")) c.engine.tp_role = 2;
            else c.engine.tp_role = parse_int_arg(v, arg) != 0 ? 1 : 0;
        } else if (!strcmp(arg, "--tp-peer")) {
            c.engine.tp_peer = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--tp-port")) {
            c.engine.tp_port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--tp-rank")) {
            c.engine.tp_rank = parse_index_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--tp-nranks")) {
            c.engine.tp_nranks = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--tp-peers")) {
            c.engine.tp_peers = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--capture-requests")) {
            c.capture_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            /* An empty value opts out of the default-on disk cache; the last
             * kv-disk flag on the command line wins. */
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
            c.kv_disk_disable = !c.kv_disk_dir[0];
        } else if (!strcmp(arg, "--no-kv-disk")) {
            c.kv_disk_dir = NULL;
            c.kv_disk_disable = true;
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_index_arg(need_arg(&i, argc, argv, arg), arg);   /* 0 is unbounded */
        } else if (!strcmp(arg, "--kv-ram-mb")) {
            c.kv_ram_mb = parse_int_arg(need_arg(&i, argc, argv, arg), arg);   /* >= 1, else the start is refused */
        } else if (!strcmp(arg, "--kv-pinned-max-mb")) {   /* L347 I3b: the pinned anchors' budget */
            c.kv_cache.pinned_max_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: unknown option: %s", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    /* Production defaults: when -m is not given, resolve the active-pointer
     * checkpoint (cwd first, then the model store) so a bare `pulsar-server` is
     * the full validated launch.  Keyed on the DEFAULT, not on a filename --
     * this used to compare against the literal "ds4flash.gguf", so passing that
     * name explicitly silently triggered the default resolution. */
    if (defaulted_model_path) {
        const char *m = resolve_default_model();
        if (m) {
            c.engine.model_path = m;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: default model %s", m);
        }
        c.models[c.n_models++] = c.engine.model_path;
    }
    return c;
}



#ifndef PULSAR_SERVER_TEST

/* L264 S4e: the segment store's removal hook on a TP leader. */
/// @param ud the engine
/// @param key the removed segment's key (40 hex digits + NUL)
/// @param rp_only only its restore point was removed (L347)
static void kv_segment_removed(void *ud, const char key[41], bool rp_only) {
    pulsar_engine_segment_dropped((pulsar_engine *)ud, key, rp_only ? 1 : 0);
}

/// @brief pulsar-server: parse the options, open the engine (an encoder or parser engine, or several -m, serves through encode_serve), size and warm the pool, open the disk KV cache, run
/// the listener until a stop signal, drain and close.
/// @param argc argument count
/// @param argv the arguments
/// @return 0 after a clean shutdown, 1 when startup failed
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = server_parse_options(argc, argv);
    server_resolve_kv_disk_dir(&cfg);
    int stall_limit_s = 0;
    {
        char werr[256];
        if (!server_stall_limit_parse(getenv("PULSAR_STALL_WATCHDOG_S"), &stall_limit_s, werr, sizeof werr)) {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: refusing to start: PULSAR_STALL_WATCHDOG_S: %s", werr);
            return 2;
        }
    }
    /* L264 S4e: a TP worker keeps its own copy of every disk KV segment the
     * leader's store names, under its own KV directory. */
    char *tp_kv_dir = cfg.kv_disk_dir ? (char *)malloc(strlen(cfg.kv_disk_dir) + 8) : NULL;
    if (tp_kv_dir) sprintf(tp_kv_dir, "%s/tp-seg", cfg.kv_disk_dir);
    cfg.engine.tp_kv_dir = tp_kv_dir;
    /* Every rank tells the group its build (the NODE frame), so the leader's
     * /health can show a mixed-build pair for what it is.  Stamped here, not in
     * the transport: this TU is rebuilt whenever HEAD moves (Makefile), and a
     * library object carrying the -D string would serve a stale one. */
    cfg.engine.build_id = PULSAR_VERSION_STR;

    pulsar_engine *engine = NULL;
    /* L321: several -m: one engine group (one process, one GPU) -- encoders and parsers, served by their own
     * endpoints */
    if (cfg.n_models > 1) {
        pulsar_engine_options opts[SERVER_MODELS_MAX];
        pulsar_engine *engines[SERVER_MODELS_MAX] = {};
        for (int i = 0; i < cfg.n_models; i++) {
            opts[i] = cfg.engine;
            opts[i].model_path = cfg.models[i];
        }
        if (pulsar_engine_open_group(engines, opts, (uint32_t)cfg.n_models) != 0) return 1;
        int rc = 0;
        for (int i = 0; i < cfg.n_models; i++)
            if (pulsar_engine_encoder(engines[i]) == PULSAR_ENCODER_NONE && !pulsar_engine_is_parser(engines[i])) {
                server_log(PULSAR_LOG_DEFAULT, "pulsar-server: %s is a %s model; a server with several models serves "
                           "encoders and parsers only (a chat model serves alone) -- refusing", cfg.models[i],
                           pulsar_engine_family_name(engines[i]));
                rc = 1;
            }
        if (rc == 0) rc = encode_serve(engines, cfg.n_models, &cfg);
        for (int i = cfg.n_models; i-- > 0;) pulsar_engine_close(engines[i]);
        free(tp_kv_dir);
        return rc;
    }
    if (pulsar_engine_open(&engine, &cfg.engine) != 0) return 1;
    /* Slice 4e (L238): a TP worker rank serves no HTTP and schedules nothing;
     * it applies the leader's frames until the leader stops it. */
    if (pulsar_engine_is_tp_worker(engine)) {
        char werr[512];
        const int wrc = pulsar_tp_worker_run(engine, werr, sizeof(werr));
        if (wrc != 0) server_log(PULSAR_LOG_DEFAULT, "pulsar-server: TP worker stopped on a failure: %s", werr);
        pulsar_engine_close(engine);
        return wrc;
    }
    /* L321: an encoder or a parser serves its own endpoints (no session, no chat, no scheduler) */
    if (pulsar_engine_encoder(engine) != PULSAR_ENCODER_NONE || pulsar_engine_is_parser(engine)) {
        const int rc = encode_serve(&engine, 1, &cfg);
        pulsar_engine_close(engine);
        free(tp_kv_dir);
        return rc;
    }
    /* A family whose tokenizer and chat renderer this build does not carry
     * cannot take a request: refused at startup, by name (L251). */
    if (!pulsar_engine_has_tokenizer(engine)) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: the %s family has no tokenizer or chat "
                   "renderer in this build -- refusing to serve", pulsar_engine_family_name(engine));
        pulsar_engine_close(engine);
        return 1;
    }
    /* L284: the spec lane lands a verify step's rows in buffers of PULSAR_SPEC_LOGITS_ROWS rows; a family whose
     * verify width (pulsar_engine_fused_heads_max) passes them is refused here, once, by name. */
    if (pulsar_engine_has_spec_rounds(engine) && pulsar_engine_fused_heads_max(engine) > PULSAR_SPEC_LOGITS_ROWS) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: the %s family verifies %u rows a step; the spec lane holds %u "
                   "-- refusing to serve", pulsar_engine_family_name(engine), pulsar_engine_fused_heads_max(engine),
                   (unsigned)PULSAR_SPEC_LOGITS_ROWS);
        pulsar_engine_close(engine);
        return 1;
    }

    /* The one authoritative speculation line: only the opened engine knows
     * whether a drafter exists (an external gguf OR dspark.* tensors merged
     * into the main artifact), so the state is logged here, never at parse. */
    /* L272 P1: whichever drafter the family loaded runs behind the one round API; --no-dspark turns
     * drafting off for every family. */
    if (pulsar_engine_drafter(engine) == PULSAR_DRAFTER_DSPARK) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: speculative decoding active (merged drafter, adaptive draft depth, start %d)",
                   pulsar_engine_spec_depth(engine));
    } else if (pulsar_engine_has_spec_rounds(engine)) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: %s speculative decoding active (%s drafter, greedy or sampled; %u verify rows a round)",
                   pulsar_engine_family_name(engine),
                   pulsar_engine_drafter(engine) == PULSAR_DRAFTER_BLOCK ? "block" : "MTP",
                   pulsar_engine_fused_heads_max(engine));
    } else if (cfg.engine.dspark_disable) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: speculative decoding disabled by --no-dspark");
    } else {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: no dspark.* tensors in the artifact; "
                   "running without speculative decoding");
    }

    {
        char ctxmem_line[256];
        server_log(PULSAR_LOG_DEFAULT, "%s",
                   pulsar_context_memory_line(ctxmem_line, sizeof ctxmem_line, "pulsar-server",
                                              cfg.engine.backend, cfg.ctx_size,
                                              cfg.engine.prefill_chunk));
    }

    /* Admission control (Tier 1 §1.4): compute the session budget from the
     * real resident weight footprint and the TRUE per-session cost (full graph
     * + prefill working set + drafter state, pulsar_engine_session_cost_bytes),
     * then gate the slot's graph allocation on it. The scheduler runs the same
     * predicate for every lazily provisioned slot. */
    const HostBytes weights_resident = pulsar_engine_weights_resident_bytes(engine);
    const KvBytes kv_budget = server_kv_budget_for_device(weights_resident);

    /* Tier-2 pool auto-sizing (batching ON by default, auto-fit). Batching is
     * the default: the bank count is DERIVED from --ctx so a moderate ctx yields
     * a multi-bank pool while a huge ctx (e.g. 1M) correctly yields N=1 (exact
     * classic behavior, no startup failure). PULSAR_MSEQ_BANKS, when set, PINS the
     * count and skips auto-sizing.  The engine owns the number: the server asks
     * it (pulsar_engine_bank_pool) and, when it derives N itself, sets it
     * (pulsar_engine_set_bank_pool) -- L159 retired the setenv round trip.  The
     * fit probe uses the _banked cost variant, which prices an explicit N. */
    const int pool_auto_max = PULSAR_SESSION_POOL_AUTO_MAX; /* fast-lane boundary */

    /* Tier-2 overcommit (task #55, increment 1). Precompute the per-bank split of
     * the banked session cost into an EAGER floor (raw ring + state lanes +
     * dspark-banked) and a DEMAND-PAGED comp/index term (VA-only, physical on
     * touch). The cost is exactly affine in N: banked(N) = shared + N*(eager +
     * demand), so eager = (banked(2)-banked(1)) - demand and shared = banked(1) -
     * (banked(2)-banked(1)). Overcommit charges only shared + N*eager (+ optional
     * touched reserve) at admission, so a huge --ctx yields N>1 banks instead of
     * N=1. DEFAULT ON as of v0.3.0 — the increment-2 proactive-eviction guard has
     * landed and was stress-validated (banks growing toward 1M are bounded by
     * LRU-idle eviction before the physical budget is breached; see
     * server_overcommit_enabled). PULSAR_OVERCOMMIT=0 reverts to full-charge N=1. */
    const bool overcommit = server_overcommit_enabled();
    const server_session_price price = server_session_price_of(engine, cfg.ctx_size, overcommit);

    int pool_pinned = 0;
    (void)pulsar_engine_bank_pool(&pool_pinned);
    if (pool_pinned) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 pool: PULSAR_MSEQ_BANKS pinned by operator "
                   "(auto-sizing skipped)");
    } else {
        /* Fit table across reference contexts — how many banks (1..cap) fit
         * within the admission budget at each ctx. Operator-facing sizing aid. */
        static const int ref_ctx[] = {32768, 65536, 131072, 262144, 524288, 1048576};
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 pool fit table (budget %.1f GiB, cap %d banks):",
                   (double)kv_budget.raw() / (1024.0 * 1024.0 * 1024.0), pool_auto_max);
        for (size_t ci = 0; ci < sizeof(ref_ctx) / sizeof(ref_ctx[0]); ci++) {
            const double cost1 =
                (double)pulsar_engine_session_cost_bytes_banked(engine, ref_ctx[ci], 1).raw()
                / (1024.0 * 1024.0 * 1024.0);
            int fitN = 1;
            double costN = cost1;
            for (int N = pool_auto_max; N >= 1; N--) {
                const KvBytes c =
                    pulsar_engine_session_cost_bytes_banked(engine, ref_ctx[ci], N);
                if (c > KvBytes{} && server_kv_admits(kv_budget, KvBytes{}, c)) {
                    fitN = N;
                    costN = (double)c.raw() / (1024.0 * 1024.0 * 1024.0);
                    break;
                }
            }
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server:   ctx %7d: fits %d bank(s) (1-bank %.2f GiB, "
                       "%d-bank %.2f GiB)",
                       ref_ctx[ci], fitN, cost1, fitN, costN);
        }
        /* Auto-derive N for the requested ctx: largest N (1..cap) whose cost fits
         * the budget (which already reserves overhead + the 4 GiB floor). Always
         * at least 1, so no ctx ever fails to start here. Overcommit charges only
         * the eager floor + reserve; the classic path charges the full N-bank
         * cost (so a huge ctx yields N=1). */
        const double gib = 1024.0 * 1024.0 * 1024.0;
        int chosen = 1;
        if (overcommit) {
            for (int N = pool_auto_max; N >= 1; N--) {
                if (server_kv_admits(kv_budget, KvBytes{}, price.admission((uint32_t)N))) { chosen = N; break; }
            }
        } else {
            for (int N = pool_auto_max; N >= 1; N--) {
                const KvBytes c = price.full((uint32_t)N);
                if (c > KvBytes{} && server_kv_admits(kv_budget, KvBytes{}, c)) { chosen = N; break; }
            }
        }
        pulsar_engine_set_bank_pool((uint32_t)chosen);
        if (overcommit) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 OVERCOMMIT auto-sized to %d bank(s) for --ctx %d: "
                       "eager floor %.2f GiB/bank charged; "
                       "demand-paged VA %.2f GiB/bank reserved (physical on touch, NOT "
                       "charged); shared %.2f GiB; admission est %.2f GiB (batching %s)",
                       chosen, cfg.ctx_size,
                       (double)price.eager_per_bank.raw() / gib,
                       (double)price.demand_per_bank.raw() / gib, (double)price.shared.raw() / gib,
                       (double)price.admission((uint32_t)chosen).raw() / gib,
                       chosen > 1 ? "ON" : "OFF");
        } else {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 pool auto-sized to %d bank(s) for --ctx %d "
                       "(batching %s)", chosen, cfg.ctx_size,
                       chosen > 1 ? "ON" : "OFF (single-session, ctx too large for a pool)");
        }
    }

    /* Full N-bank cost is what the allocator will reserve (incl. the
     * demand-paged VA, which cudaMallocManaged counts at full size). Under
     * overcommit, admission is gated on the eager-floor est instead so the pool
     * comes up at N>1; the full price is what the create is checked against. */
    const uint32_t final_banks = pulsar_engine_bank_pool(NULL);
    const KvBytes full_banked_est = price.full(final_banks);   /* pulsar_engine_session_cost_bytes: the same pool */
    const KvBytes overcommit_admission_est = overcommit ? price.admission(final_banks) : KvBytes{};
    const KvBytes session_est = overcommit ? overcommit_admission_est : full_banked_est;
    if (pulsar_gpu_integrated() == 0) {   /* L338: the discrete card's own memory (server_kv_budget_for_device) */
        uint64_t free_b = 0, total_b = 0;
        pulsar_gpu_mem_info(&free_b, &total_b);
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: session admission: discrete GPU -- device %.1f GiB, free after the weights %.1f GiB "
                   "(weights_resident=%.1f GiB), margin %.2f GiB (working set estimate %.2f + reserve %.2f; checked "
                   "against the warmup), budget=%.1f GiB",
                   (double)total_b / (1024.0 * 1024.0 * 1024.0), (double)free_b / (1024.0 * 1024.0 * 1024.0),
                   (double)weights_resident.raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)server_discrete_margin_bytes(DeviceBytes{PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES}).raw() /
                       (1024.0 * 1024.0 * 1024.0),
                   (double)PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES / (1024.0 * 1024.0 * 1024.0),
                   (double)PULSAR_SERVER_DISCRETE_GUARD_FLOOR_BYTES / (1024.0 * 1024.0 * 1024.0),
                   (double)kv_budget.raw() / (1024.0 * 1024.0 * 1024.0));
    } else {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: session admission: usable=%.1f GiB, weights_resident=%.1f GiB, "
                   "overhead=%.1f GiB, floor=%.1f GiB, budget=%.1f GiB",
                   (double)PULSAR_SERVER_USABLE_BYTES / (1024.0 * 1024.0 * 1024.0),
                   (double)weights_resident.raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)PULSAR_SERVER_PROCESS_OVERHEAD_BYTES / (1024.0 * 1024.0 * 1024.0),
                   (double)PULSAR_SERVER_MEM_FLOOR_BYTES / (1024.0 * 1024.0 * 1024.0),
                   (double)kv_budget.raw() / (1024.0 * 1024.0 * 1024.0));
    }
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: session admission: per-session true cost est=%.2f GiB (ctx=%d)",
               (double)session_est.raw() / (1024.0 * 1024.0 * 1024.0),
               cfg.ctx_size);
    if (session_est == KvBytes{} || !server_kv_admits(kv_budget, KvBytes{}, session_est)) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: session admission REJECTED: est %.2f GiB exceeds budget %.2f GiB "
                   "(reduce --ctx-size)",
                   (double)session_est.raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)kv_budget.raw() / (1024.0 * 1024.0 * 1024.0));
        pulsar_engine_close(engine);
        return 1;
    }

    /* L347 I4: the device's free memory before the create -- the warmup's working set is measured against it */
    const bool discrete = pulsar_gpu_integrated() == 0;
    DeviceBytes dev_free_before_create{};
    if (discrete) {
        uint64_t free_b = 0, total_unused = 0;
        pulsar_gpu_mem_info(&free_b, &total_unused);
        dev_free_before_create = DeviceBytes{free_b};
    }
    pulsar_session *session = NULL;
    if (pulsar_session_create(&session, engine, cfg.ctx_size) != 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to create %s session",
                   pulsar_backend_name(cfg.engine.backend));
        pulsar_engine_close(engine);
        return 1;
    }
    /* The price and the allocation are one piece of code; check they agree
     * and commit the actual to the ledger.  Under overcommit the admission
     * est is the eager floor only; the FULL banked price is what the create
     * allocated (the demand-paged managed VA counts at full size). */
    const KvBytes session_actual =
        server_reconciled_session_cost(0, cfg.ctx_size,
                                       overcommit ? full_banked_est : session_est,
                                       pulsar_session_resident_bytes(session));
    if (session_actual == KvBytes{}) {
        pulsar_session_free(session);
        pulsar_engine_close(engine);
        return 1;
    }

    /* Materialize the lazy first-generation CUDA working set BEFORE deriving
     * the admission budget or opening the listener (F1, task #32). */
    /* L347 I3a: the disk cache's host pool -- the RAM tier on a discrete-GPU host, the copy-out ring on a unified one.
     * Planned BEFORE the warmup (battery #590, server-kv-headroom-live on GB10): the pool is host memory the cache
     * allocates (and touches) when it opens, after the post-warmup MemAvailable reading the admission budget and the 2b
     * guard's ceiling are cut from -- so that reading charges it (kv_pool_charge below).  Uncharged, the 1 GiB ring came
     * out of the guard's headroom and MemAvailable fell 1.02 GiB, below the floor. */
    kv_ram_plan ram = {0, false};
    {
        const uint64_t mem_total = kv_host_mem_total();   /* a container's limit, not the host's RAM (L338, CT 130) */
        kv_ram_plan_for(pulsar_gpu_integrated(), mem_total, cfg.kv_ram_mb, &ram.bytes, &ram.retain);
    }
    /* the host pool the cache opens with */
    const HostBytes kv_pool_charge{cfg.kv_disk_dir && cfg.kv_disk_dir[0] ? ram.bytes : 0u};
    if (!server_warmup_generation(engine, session, cfg.ctx_size)) {
        pulsar_session_free(session);
        pulsar_engine_close(engine);
        return 1;
    }

    /* Re-derive the admission budget from MEASURED post-warmup MemAvailable
     * (the est==actual discipline applied to the budget itself): the static
     * USABLE/OVERHEAD constants drift as kernels and workspaces change
     * (measured drift ~1 GiB by 2026-07-17, two days after the constants
     * were sized), while the measured number cannot.  The static formula is
     * kept as an upper bound — it still protects against a page-cache- or
     * UVM-inflated MemAvailable reading — and slot 0's RESIDENT bytes are
     * added back because they are already excluded from MemAvailable.  A parse failure keeps the
     * static budget: the live floor check in provision_slot fails closed on
     * its own read. */
    /* The one floor every MemAvailable check reads (pulsar_server_internal.h, "The memory floor"), derived now that
     * the session's per-step growth is known: the OS margin, the kernel's high watermarks, and one worker step's
     * host/driver demand -- a prefill chunk's new KV pages on one bank plus a persist at the full context. */
    {
        const HostBytes zone_high = server_zone_high_watermark_bytes();
        const KvBytes chunk_kv = pulsar_session_quantum_growth_bytes_per_bank(
            session, pulsar_kv_rows(pulsar_session_prefill_cap(session)));
        const HostBytes persist = server_persist_host_bytes(cfg.ctx_size);
        const HostBytes step = server_floor_step_bytes(discrete, chunk_kv, persist);
        const HostBytes floor = server_mem_floor_init(zone_high, step);
        const double mib = 1024.0 * 1024.0;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: memory floor %.0f MiB = OS margin %.0f + kernel high watermarks %.0f%s + one "
                   "step's demand %.0f (a %d-token prefill chunk's KV %.0f + a persist at %d tokens %.0f)",
                   (double)floor.raw() / mib, (double)PULSAR_SERVER_MEM_FLOOR_BYTES / mib,
                   (double)zone_high.raw() / mib, zone_high != HostBytes{} ? "" : " (/proc/zoneinfo unreadable)",
                   (double)step.raw() / mib, pulsar_session_prefill_cap(session).raw(),
                   (double)(step - persist).raw() / mib,   /* the chunk's KV the floor holds: none on a discrete card */
                   cfg.ctx_size, (double)persist.raw() / mib);
    }
    KvBytes kv_budget_final = kv_budget;
    HostBytes warm_avail{};   /* L319: the touched-KV ceiling's measurement (below) */
    DeviceBytes warm_dev_free{};   /* L338: its discrete-card reading, the device's free memory at the same point */
    KvBytes warm_touched{};
    {
        /* What of slot 0 is RESIDENT now, and so already out of MemAvailable:
         * its priced cost less the pool's demand-paged comp/index VA that no
         * warmup row touched.  A pooled session prices every bank's comp/index
         * at full --ctx (the managed VA the create reserves), but a page exists
         * only once a row lands on it; adding the whole price back counted
         * ~38 GiB of VA as memory (2026-09-29, the pair at --ctx 1M: budget
         * 61.3 GiB, box full at ~12 GiB of KV).  A single session allocates its
         * comp/index eagerly, so all of it is resident. */
        const int banks_now = pulsar_session_bank_count(session);
        const KvBytes demand_va = banks_now > 1
            ? pulsar_engine_demand_paged_bytes_per_bank(engine, cfg.ctx_size).times((uint64_t)banks_now)
            : KvBytes{};
        const KvBytes touched_now = pulsar_session_touched_kv_bytes(session);
        const KvBytes untouched_va = demand_va > touched_now ? demand_va - touched_now : KvBytes{};
        const KvBytes resident_now = session_actual > untouched_va ? session_actual - untouched_va : KvBytes{};
        /* L347: the disk cache's host pool, opened after this reading, is charged to it now */
        const HostBytes avail_raw = server_mem_available_bytes();
        const HostBytes avail_now = avail_raw > kv_pool_charge ? avail_raw - kv_pool_charge : HostBytes{};
        warm_avail = avail_now;
        warm_touched = touched_now;
        if (discrete) warm_dev_free = server_guard_avail_bytes(true);   /* the same reading */
        if (discrete) {
            /* L347 I4: the card's working set, measured -- the KV budget is VRAM, which MemAvailable (the host's) does
             * not describe.  Past the estimate the banks were sized with, the start is refused (rule 9). */
            uint64_t free_b = 0, total_unused = 0;
            pulsar_gpu_mem_info(&free_b, &total_unused);
            const DeviceBytes dev_free_now{free_b};
            const DeviceBytes ws = server_discrete_working_set_bytes(dev_free_before_create, dev_free_now, resident_now);
            const double gib = 1024.0 * 1024.0 * 1024.0;
            if (ws > DeviceBytes{PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES}) {
                server_log(PULSAR_LOG_ERROR,
                           "pulsar-server: refusing to start: the warmup's working set is %.2f GiB (device free %.2f "
                           "GiB before the create, %.2f after the warmup, session resident %.2f) -- past the %.2f GiB "
                           "estimate the banks were sized with (PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES); lower "
                           "--ctx or PULSAR_MSEQ_BANKS, or re-measure the estimate (L347 I4)",
                           (double)ws.raw() / gib, (double)dev_free_before_create.raw() / gib,
                           (double)dev_free_now.raw() / gib, (double)resident_now.raw() / gib,
                           (double)PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES / gib);
                pulsar_session_free(session);
                pulsar_engine_close(engine);
                return 1;
            }
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: session admission: discrete GPU working set measured %.2f GiB (estimate %.2f; "
                       "device free %.2f GiB before the create, %.2f after the warmup, session resident %.2f); budget "
                       "%.2f GiB; the 2b guard (device free - %.2f GiB) is the run-time backstop",
                       (double)ws.raw() / gib, (double)PULSAR_SERVER_DISCRETE_WORKING_SET_EST_BYTES / gib,
                       (double)dev_free_before_create.raw() / gib, (double)dev_free_now.raw() / gib,
                       (double)resident_now.raw() / gib, (double)kv_budget_final.raw() / gib,
                       (double)PULSAR_SERVER_DISCRETE_GUARD_FLOOR_BYTES / gib);
        } else if (avail_now > HostBytes{}) {
            const HostBytes floor = server_mem_floor();
            const HostBytes headroom = avail_now > floor ? avail_now - floor : HostBytes{};
            /* unified memory (this branch's): the host's headroom is the pool's, room for KV */
            KvBytes measured = resident_now + pulsar_kv_room_bytes(pulsar_unified_device_bytes(headroom));
            /* The measured budget is one-shot and PERMANENT, while the
             * MemAvailable it derives from can read transiently low at
             * startup (a previous model process still releasing memory;
             * UVM/MemTotal shrink on driver 610).  A budget refusal is
             * classed "eviction helps" by the scheduler, so a collapsed
             * budget would evict-thrash slot 0 forever instead of ever
             * provisioning slot 1.  Clamp: the budget always structurally
             * permits at least one more session (slot 0's actual + one
             * session estimate — the extra-slot ctx is capped at slot 0's,
             * so its est can never exceed session_est); when memory is
             * GENUINELY tight, the LIVE per-attempt floor check
             * (server_kv_headroom_admits — truthful post-warmup) is the guard,
             * and it recovers naturally when memory frees. */
            const KvBytes budget_min = resident_now + session_est;
            if (measured < budget_min) {
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: session admission: measured budget "
                           "%.2f GiB clamped up to %.2f GiB (slot 0 resident + "
                           "one session est): MemAvailable %.2f GiB reads low "
                           "at startup; the live MemAvailable floor check "
                           "remains the per-attempt guard",
                           (double)measured.raw() / (1024.0 * 1024.0 * 1024.0),
                           (double)budget_min.raw() / (1024.0 * 1024.0 * 1024.0),
                           (double)avail_now.raw() / (1024.0 * 1024.0 * 1024.0));
                measured = budget_min;
            }
            if (measured < kv_budget_final) kv_budget_final = measured;
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: session admission: measured budget %.2f GiB "
                       "(MemAvailable %.2f GiB post-warmup - floor %.2f GiB "
                       "+ slot 0 resident %.2f GiB [priced %.2f, untouched VA %.2f]; "
                       "static bound %.2f GiB)",
                       (double)kv_budget_final.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)avail_now.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)floor.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)resident_now.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)session_actual.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)untouched_va.raw() / (1024.0 * 1024.0 * 1024.0),
                       (double)kv_budget.raw() / (1024.0 * 1024.0 * 1024.0));
        }
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.started = time(NULL);          /* uptime origin reported by /health */
    /* L357: the /test/ routes (http_server.cpp) answer only when the test harness asks for them; said, so a
     * production log shows it if they were ever on */
    {
        const char *tr = getenv("PULSAR_SERVER_TEST_ROUTES");
        s.test_routes = tr && !strcmp(tr, "1");
        if (s.test_routes)
            server_log(PULSAR_LOG_WARNING, "pulsar-server: test routes ON (PULSAR_SERVER_TEST_ROUTES=1): "
                       "POST /test/guard-budget answers -- never set this in production");
    }
    /* Slot 0 is provisioned here at the configured --ctx-size over the ONE
     * session (s.sess). Tier-2: if the created session is bank-pooled
     * (PULSAR_MSEQ_BANKS>1), every other slot maps to one of its banks; the
     * scheduler provisions banks 1..pool_banks-1 lazily as pure host
     * bookkeeping. In classic mode (pool_banks==0) slot 0 is the only slot —
     * a job needing another one queues. */
    const int pool_banks = pulsar_session_bank_count(session);
    /* Hard clamp to the slots array: an engine-side PULSAR_MSEQ_BANKS pin
     * beyond POOL_CAP used to walk the server's slots[] out of bounds. */
    int pool_banks_clamped = pool_banks;
    if (pool_banks_clamped > PULSAR_SESSION_POOL_CAP) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: engine bank pool %d clamped to slot cap %d",
                   pool_banks_clamped, PULSAR_SESSION_POOL_CAP);
        pool_banks_clamped = PULSAR_SESSION_POOL_CAP;
    }
    s.pool_banks = pool_banks_clamped > 1 ? pool_banks_clamped : 0;
    s.reasoning_default = cfg.reasoning_default < 0 ? 0 : cfg.reasoning_default + 1;   /* L338 P4: --reasoning */
    if (s.reasoning_default)
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: --reasoning %s: a request that sends no thinking control %s",
                   cfg.reasoning_default ? "on" : "off", cfg.reasoning_default ? "thinks" : "does not think");
    /* L338 P2.4: the judge's reserved bank -- the pool's last, past the chat pool (no chat slot is it).  The chat
     * pool keeps at least two banks (a pool); a smaller one serves no judge. */
    int judge_bank = -1;
    {
        const bool has = pulsar_engine_has_adapter(engine);
        const int want = cfg.judge_banks < 0 ? (has ? 1 : 0) : cfg.judge_banks;
        if (want > 0 && !has) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: --judge-banks %d: the container carries no adapter -- "
                       "refusing to start", want);
            return 1;
        }
        if (want > 0 && s.pool_banks - 1 >= 2) {
            s.pool_banks -= 1;
            judge_bank = s.pool_banks;
        } else if (want > 0) {
            server_log(PULSAR_LOG_WARNING, "pulsar-server: no /v1/systemone: the bank pool (%d) leaves no chat pool of "
                       "two beside a judge bank (set PULSAR_MSEQ_BANKS >= 3 or a smaller --ctx)", pool_banks_clamped);
        }
    }
    /* The mixed lane fits a prefill run of kstep = prefill_chunk - POOL_CAP rows
     * beside up to POOL_CAP decode rows; a pinned chunk below POOL_CAP + 1
     * clamps kstep to 1 and the step exceeds the chunk cap on every quantum
     * (every decode slot errors, then the decode-only retry errors too).
     * PULSAR_CUDA_PREFILL_CHUNK accepts any value >= 1, so refuse the pooled
     * configuration here instead (L177). */
    if (s.pool_banks && cfg.engine.prefill_chunk != 0 &&
        cfg.engine.prefill_chunk < (uint32_t)PULSAR_SESSION_POOL_CAP + 1u) {
        server_log(PULSAR_LOG_ERROR,
                   "pulsar-server: prefill_chunk %u is below the pooled minimum %d "
                   "(PULSAR_SESSION_POOL_CAP + 1): the batched lanes cannot fit a step -- refusing to start",
                   cfg.engine.prefill_chunk, PULSAR_SESSION_POOL_CAP + 1);
        return 1;
    }
    s.live_bank = BankId{0};
    /* L118 (everything is a batch): the three-way scheduler, its
     * spec_max_live crossover knob (PULSAR_SERVER_SPEC_MAX_LIVE), and the
     * PULSAR_CLASSIC_DECODE A/B hatch are DELETED — decode runs through the
     * batched quanta at every n >= 1, a solo session being a batch of one.
     * Parity evidence: ledger row L118. */
    /* plan-34 phase-2 inc 5: fused mixed-batch lane. Default OFF (OPT-IN via
     * PULSAR_MIXED_BATCH=1). It is a workload-dependent TRADE, not a safe blanket
     * default — three measured regimes:
     *   (a) shallow / synchronized concurrent  -> NEUTRAL (teb --perf-only 2026-07-27:
     *       ON==OFF within noise at d0/d4096, all concurrencies).
     *   (b) one big prefill overlapping a few shallow decoders (plan-34, K=8000
     *       ndec=3, chunk 8) -> better tail/throughput (the reason it exists).
     *   (c) MULTIPLE DEEP concurrent streams -> REGRESSES HARD. teb --perf-only
     *       d8192/c4 reproduced ON 265 pp / 16.0 tg vs OFF 373 / 30.6 = ~-29%
     *       prefill, ~-48% decode (the deep KV decode step is already bandwidth-
     *       saturated, so folding prefill in displaces decode catastrophically).
     * Was briefly flipped default-ON 2026-07-26 on the (a)/(b) reasoning; the
     * teb A/B (2026-07-27) found regime (c) and Tyler reverted it to opt-in —
     * a default that can halve decode t/s at deep-context concurrency is a
     * footgun on a 1M-capable box. Re-enable per-deployment only for a workload
     * verified to be shallow-bursty. DO NOT re-flip default-on without a
     * depth/concurrency guard (auto-disable in regime (c)). PULSAR_MIXED_CHUNK
     * overrides the chunk (c8 = plan-34 tail point). Phase-1 warm-fork TTFT is
     * v0.3.0's headline continuous-batching win. One startup read, no hot-path
     * getenv. Only engages in pool mode. */
    {
        /* Default ON in pool mode WITH the deep-concurrent guard below — the
         * condition the opt-in revert set for re-flipping ("do NOT re-flip
         * default-on without a depth/concurrency guard"). The guard bounds the
         * aggregate committed depth of the active decode set: regime (c)
         * measured NEUTRAL at 4x4096=16384 aggregate rows and -29% pp/-48% tg
         * at 4x8192=32768, so the default threshold 16384 stops fusing exactly
         * where fusing stops paying, and regimes (a)/(b) keep their win.
         * PULSAR_MIXED_BATCH=0 still forces the lane fully off;
         * PULSAR_MIXED_DEEP_GUARD_ROWS overrides the threshold (0 = no guard). */
        const char *mb = getenv("PULSAR_MIXED_BATCH");
        /* L251/L284: this lane's prompt rows ride decode_mixed beside the decode rows -- a family whose
         * decode_mixed carries none (Qwen: its prompts ride the spec lane's fused step) does not run it */
        s.mixed_batch_enabled = s.pool_banks > 0 && pulsar_engine_has_mixed_prefill(engine) &&
                                !(mb && (mb[0] == '0' || !strcasecmp(mb, "off")));
        s.mixed_chunk_tokens = 8;          /* the env knobs for these had no caller (L159 inc 4) */
        s.mixed_deep_guard_rows = 16384;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: fused mixed-batch lane %s (chunk=%d/step, deep guard=%d rows)",
                   s.mixed_batch_enabled ? "ENABLED (default; opt out with PULSAR_MIXED_BATCH=0)"
                                         : "disabled (PULSAR_MIXED_BATCH)",
                   s.mixed_chunk_tokens,
                   s.mixed_deep_guard_rows);
    }
    s.n_slots = 1;
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) s.slots[i].bank = BankId{i}; /* slot i holds bank i */
    if (s.pool_banks > 0) {
        /* The pool's ledger (L391, server::ledger_open_pool): the admitted
         * price split evenly over the banks it was priced over -- ALL the
         * session's banks, the judge's included, each one's eager floor being
         * resident from the create.  Startup commits the charges of the banks
         * no chat slot holds (the judge's; before L391 its share was divided
         * out and committed nowhere, so a full pool sat one charge under the
         * estimate); each chat bank's is committed as it is provisioned, so
         * committed == session_est, to the byte, when all are live.  The
         * self-check refuses a ledger that does not add up to that or that the
         * budget does not hold.
         * Pool mode: NO chat bank is provisioned at boot — bank 0 is an
         * ordinary bank provisioned on first use like the rest.  The
         * old boot-provisioned slot 0 was the root of a whole wart family:
         * the routing phantom (first conversation routed "in place" to the
         * empty boot slot but landing on bank 1), the un-provisionable /
         * un-evictable squatter, and the compensating special cases each of
         * those needed. */
        char ledger_err[320];
        if (!s.ledger_open_pool(kv_budget_final, session_est, (uint32_t)pool_banks, ledger_err, sizeof ledger_err)) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: refusing to start: the session-cost ledger does not add up "
                       "-- %s (L391)", ledger_err);
            pulsar_session_free(session);
            pulsar_engine_close(engine);
            return 1;
        }
        /* L347 I4: where the checkpoints are (the store's placement): a discrete card's are pinned host RAM */
        char ckpt_where[96];
        const HostBytes ckpt_host = pulsar_session_ckpt_host_bytes(session);
        if (ckpt_host != HostBytes{})
            snprintf(ckpt_where, sizeof ckpt_where, "in pinned host RAM (%.2f GiB), not in the price",
                     (double)ckpt_host.raw() / (1024.0 * 1024.0 * 1024.0));
        else
            snprintf(ckpt_where, sizeof ckpt_where, "in device memory, in the price");
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 shared pool ACTIVE: %d banks, "
                   "unified batch decode (L118), per-bank marginal %.2f GiB "
                   "(pool resident actual %.2f GiB; grid checkpoints %s); ledger: admitted %.2f GiB over %d "
                   "bank(s), %d chat bank(s) charged as provisioned, %.2f GiB committed now for the rest",
                   pool_banks,
                   (double)s.bank_charge(BankId{0}).raw() / (1024.0 * 1024.0 * 1024.0),
                   (double)session_actual.raw() / (1024.0 * 1024.0 * 1024.0), ckpt_where,
                   (double)s.ledger_admitted_bytes.raw() / (1024.0 * 1024.0 * 1024.0), pool_banks, s.pool_banks,
                   (double)s.ledger_reserved_bytes.raw() / (1024.0 * 1024.0 * 1024.0));
    } else {
        s.ledger_open_classic(kv_budget_final, session_actual);
    }
    /* L338 / L391: where the KV lives, for every headroom check (server::mem_reading), guard or no guard */
    s.guard_discrete = (pulsar_gpu_integrated() == 0);
    /* Tier-2 task #55 increment 2b — proactive-eviction guard. Enabled only under
     * overcommit with N>1 banks (banks are 1M-capable but total physical is
     * bounded). The guard keeps touched (demand-paged) KV under budget − eager by
     * spilling LRU-idle banks to local fast disk + cudaFree; a returning bank
     * reloads bit-identically. Decisions use the deterministic touched accounting,
     * NEVER the coarse cudaMemGetInfo gauge. PULSAR_SERVER_KV_BUDGET_OVERRIDE (bytes)
     * lowers the budget for the memory-safety smoke so the guard fires at modest
     * fills while the box stays far from real OOM. */
    if (overcommit && s.pool_banks > 0) {
        /* L338: a discrete card's guard reads the device (server_guard_avail_bytes), and the ledger's share does not
         * cap it unless the test hook below sets a budget. */
        s.guard_floor = server_guard_floor(s.guard_discrete);
        KvBytes budget = s.guard_discrete ? KvBytes::max() : kv_budget_final;
        /* Test hook (restored L264, 2026-10-04: its callers are the guard legs of
         * tools/tp2-l264 -- the spill/restore through the segment chain has no
         * other way to fire at a test's fills). */
        const char *ov = getenv("PULSAR_SERVER_KV_BUDGET_OVERRIDE");
        if (ov && ov[0]) {
            const unsigned long long b = strtoull(ov, NULL, 10);
            if (b > 0) {
                budget = KvBytes{(uint64_t)b};
                server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: KV budget OVERRIDDEN to %.2f GiB (test hook)",
                           (double)budget.raw() / (1024.0 * 1024.0 * 1024.0));
            }
        }
        s.guard_eager_bytes = overcommit_admission_est;   /* eager floor resident (the ledger's figure) */
        /* L319: the ceiling is the smaller of the ledger's share and the measured headroom (see
         * server_guard_touched_budget); a pool with no measured headroom does not start -- unguarded, its banks
         * would grow until the box runs dry.  2026-10-29: every guard decision reads the LIVE budget under this
         * ceiling (server::guard_budget). */
        s.guard_kv_budget = budget;
        /* the guard's reading: the device's free memory, or on unified memory MemAvailable (the one pool) */
        const DeviceBytes guard_warm_avail = s.guard_discrete ? warm_dev_free : pulsar_unified_device_bytes(warm_avail);
        s.guard_touched_ceiling = server_guard_touched_budget(s.guard_kv_budget, s.guard_eager_bytes, warm_touched,
                                                              guard_warm_avail, s.guard_floor);
        s.guard_enabled = (s.guard_touched_ceiling > KvBytes{});
        const double gib = 1024.0 * 1024.0 * 1024.0;
        if (!s.guard_enabled) {
            server_log(PULSAR_LOG_ERROR,
                       "pulsar-server: refusing to start: a %d-bank pool has no touched-KV headroom (%s %.2f GiB "
                       "post-warmup, floor %.2f GiB; kv budget %.2f − eager %.2f GiB) -- lower --ctx or "
                       "PULSAR_MSEQ_BANKS (L319)",
                       s.pool_banks, s.guard_discrete ? "device free" : "MemAvailable",
                       (double)guard_warm_avail.raw() / gib,
                       (double)s.guard_floor.raw() / gib,
                       (double)budget.raw() / gib, (double)s.guard_eager_bytes.raw() / gib);
            pulsar_session_free(session);
            pulsar_engine_close(engine);
            return 1;
        }
        if (s.guard_discrete && budget == KvBytes::max())
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 2b guard ENABLED (discrete GPU): touched ceiling %.2f GiB = touched %.2f "
                       "+ device free %.2f − floor %.2f post-warmup; every decision reads the live budget under it "
                       "(touched + device free − floor now); spills persist to the disk KV cache",
                       (double)s.guard_touched_ceiling.raw() / gib, (double)warm_touched.raw() / gib,
                       (double)guard_warm_avail.raw() / gib, (double)s.guard_floor.raw() / gib);
        else
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 2b guard ENABLED: touched ceiling %.2f GiB = min(kv budget %.2f − eager "
                       "%.2f, touched %.2f + %s %.2f − floor %.2f post-warmup); every decision reads the live "
                       "budget under it (touched + %s − floor now); spills persist to the disk KV cache",
                       (double)s.guard_touched_ceiling.raw() / gib, (double)budget.raw() / gib,
                       (double)s.guard_eager_bytes.raw() / gib, (double)warm_touched.raw() / gib,
                       s.guard_discrete ? "device free" : "MemAvailable", (double)guard_warm_avail.raw() / gib,
                       (double)s.guard_floor.raw() / gib, s.guard_discrete ? "device free" : "MemAvailable");
    }
    s.sess = session;                /* the one session; slot 0 describes it */
    /* ctx_size on slot 0 is the pool's shared per-bank ctx reference and is
     * read regardless of provisioning state.  Classic mode (no pool) keeps
     * the boot session live on slot 0; pool mode provisions bank 0 lazily
     * like every other bank. */
    s.slots[0].ctx_size = cfg.ctx_size;
    s.pool_ctx_size = cfg.ctx_size;
    /* Unconditional: the context-scaled KV one bank holds is a property of the
     * shape and the ctx, not of how admission happens to be gated. */
    s.kv_bank_bytes = pulsar_engine_demand_paged_bytes_per_bank(engine, cfg.ctx_size);
    if (s.pool_banks == 0) {
        s.slots[0].provisioned = true;
        s.slots[0].state = SLOT_IDLE;
    }
    /* Slot-routing trivial-match threshold (choose_slot_for_job): the token
     * depth at which a shared prefix stops meaning "same rendered template
     * header" and starts meaning "same conversation". Derived once per model
     * rather than hardcoded: tokenize the largest template-injected text two
     * UNRELATED conversations can share — BOS plus the longest fixed
     * reasoning-effort preamble (prompt_render.cpp renders both before any
     * client content; the preambles only appear at PULSAR_THINK_HIGH/MAX,
     * unreachable below a 384K context, but a threshold sized for them stays
     * correct on boxes where they ARE reachable and costs nothing here) —
     * then allow PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS of incidental
     * prologue overlap on top (see the constant's comment for the sizing
     * evidence). */
    {
        const int hdr_len = server_family_for_engine(engine)->trivial_header_tokens(engine);   /* the loaded family's (L272 B6) */
        s.slot_trivial_common_tokens =
            hdr_len + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: slot routing: trivial-match threshold %d tokens "
                   "(template header %d + incidental allowance %d)",
                   s.slot_trivial_common_tokens, hdr_len,
                   PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS);
    }
    /* L329 (a): the shared-prefix wait's bound (server::prefix_wait_decide) -- read once, announced */
    {
        const char *pw = getenv("PULSAR_SERVER_PREFIX_WAIT_MS");
        s.prefix_wait_ms = server_prefix_wait_parse(pw);
        if (s.prefix_wait_ms < 0) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: PULSAR_SERVER_PREFIX_WAIT_MS=\"%s\" is not whole "
                       "milliseconds (0 = off, at most a day) -- refusing to start", pw);
            pulsar_session_free(session);
            pulsar_engine_close(engine);
            return 1;
        }
        if (s.pool_banks > 1)
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: prefix wait: a queued request waits up to %d ms for a cut "
                       "another prompt's prefill is filling, then adopts it (PULSAR_SERVER_PREFIX_WAIT_MS; 0 = off)",
                       s.prefix_wait_ms);
    }
    /* L330: the restore slice's test hook -- read once, refused when malformed (announced with the disk tier below) */
    {
        const char *rs = getenv("PULSAR_SERVER_KV_RESTORE_SLICE_MS");
        s.restore_slice_ms = server_restore_slice_parse(rs);
        if (s.restore_slice_ms < 0) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: PULSAR_SERVER_KV_RESTORE_SLICE_MS=\"%s\" is not whole "
                       "milliseconds in 1..600000 (unset: one prefill chunk's priced time a slice) -- refusing to "
                       "start", rs);
            pulsar_session_free(session);
            pulsar_engine_close(engine);
            return 1;
        }
    }
    /* L272 B5: the template's turn markers, for the anchor cold store and the routing anchor. */
    if (!pulsar_chat_turn_markers(engine, &s.turn_markers))
        server_log(PULSAR_LOG_WARNING, "pulsar-server: the tokenizer does not spell the chat turn markers: "
                   "the anchor cold store and the last-turn routing anchor are off");
    s.default_tokens = cfg.default_tokens;
    s.tool_mem.max_entries = PULSAR_TOOL_MEMORY_DEFAULT_MAX_IDS;
    if (cfg.kv_disk_dir &&
        !kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                       pulsar_engine_kv_identity(engine),
                       cfg.kv_cache, ram))
    {
        /* Never fatal: an uncreatable/read-only directory logs its reason in
         * kv_cache_open; state the consequence once and serve without disk
         * restore. */
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache disabled (directory %s unusable); "
                   "serving without disk restore",
                   cfg.kv_disk_dir);
    }
    s.kv_cache_publish_store();   /* L276: /metrics sees the store as it opened (a no-op when disabled) */
    /* L264 S4e: on a tensor-parallel group every worker keeps its own copy of
     * each segment the store names (the engine mirrors save and load).  Every
     * key the store removes from here on is dropped on the workers too, and the
     * keys it holds now reconcile them -- copies left by a crash, or by orphans
     * the open just pruned, go. */
    if (s.kv.enabled && pulsar_engine_is_tp(engine)) {
        pulsar_segstore_set_removed_hook(s.kv.st, kv_segment_removed, engine);
        char *keys = NULL;
        const int n_keys = pulsar_segstore_keys(s.kv.st, &keys);
        if (pulsar_engine_segment_reconcile(engine, keys, n_keys) != 0)
            server_log(PULSAR_LOG_WARNING, "pulsar-server: tp: the disk KV segment reconcile could not be sent; "
                                           "orphan worker copies stay until the next bring-up");
        else
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: disk KV segments mirrored across the TP group "
                                           "(%d held, workers reconciled)", n_keys);
        free(keys);
    }
    /* L264: a guard spill persists the bank's history to the disk KV cache; with
     * no cache there is nowhere to put it, and freeing the bank would lose the
     * conversation outright. */
    if (s.guard_enabled && !s.kv.enabled) {
        s.guard_enabled = false;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 2b guard DISABLED: it spills to the disk KV cache, which is off");
    }
    /* L330 (rule 5): how a disk chain comes back */
    if (s.kv.enabled && s.restore_slice_ms > 0)
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: disk KV restores are SLICED between rounds (L330): a slice reads the segments "
                   "%d ms reads (PULSAR_SERVER_KV_RESTORE_SLICE_MS, test hook; one segment while the read meter is "
                   "unmeasured); the job reserves the chain's KV from bind", s.restore_slice_ms);
    else if (s.kv.enabled)
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV restores are SLICED between rounds (L330): a request's chain, or a spilled "
                   "bank's history, loads the segments one prefill chunk's priced time reads per worker turn (one "
                   "segment while the read meter or the prefill fit is unmeasured); the job reserves the chain's KV "
                   "from bind");
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_cond_init(&s.stream_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    pthread_mutex_init(&s.capture_mu, NULL);
    if (cfg.capture_path) {
        /* L216: the served workload as a corpus -- one JSON line per accepted
         * request {ts, path, body}; the body is the client's JSON verbatim. */
        s.capture = fopen(cfg.capture_path, "a");
        if (!s.capture) {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to open capture file %s: %s",
                       cfg.capture_path, strerror(errno));
            s.close_resources();
            return 1;
        }
        setvbuf(s.capture, NULL, _IOLBF, 0);
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: capturing accepted requests to %s", cfg.capture_path);
    }
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            s.close_resources();
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: tracing session to %s", cfg.trace_path);
    }

    /* Seed the /metrics snapshots (slot-0 position, spec-decode config like
     * max_draft) before any client thread can scrape: send_metrics never
     * calls into the engine (CUDA-state audit, pulsar_server_internal.h). This
     * runs before the worker thread starts, so it is still single-threaded
     * engine access. */
    /* L338 P2.4: the judge on its reserved bank (single-threaded: before the worker starts) */
    if (judge_bank >= 0) {
        char jerr[320];
        if (!server_judge_open(&s, cfg.engine.model_path, judge_bank, jerr, sizeof jerr)) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: /v1/systemone: %s -- refusing to start", jerr);
            s.close_resources();
            return 1;
        }
    }
    s.publish_metrics_snapshot();
    /* Same window: still single-threaded, and the TP records are final. */
    s.build_tp_health();

    pthread_t worker;
    if (pthread_create(&worker, NULL, server_worker_main, &s) != 0) die("failed to start worker");
    s.worker_thread = worker;
    s.stall_limit_s = stall_limit_s;
    if (!s.watchdog_start()) die("failed to start the stall watchdog");

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(worker, NULL);
        s.watchdog_stop();
        s.close_resources();
        return 1;
    }
    g_listen_fd = lfd;
    server_warn_if_exposed(cfg.host);
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: listening on http://%s:%d (pulsar %s)",
               cfg.host, cfg.port, PULSAR_VERSION_STR);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        pthread_mutex_lock(&s.mu);
        const int at_cap = s.clients >= PULSAR_SERVER_MAX_CLIENTS;
        if (!at_cap) s.clients++;
        pthread_mutex_unlock(&s.mu);
        if (at_cap) {
            s.count_503(SERVER_503_CONNECTIONS);   /* L378 */
            http_error_retry(fd, 503, "too many connections", HTTP_RETRY_BUSY_S);
            close(fd);
            continue;
        }
        client_arg *ca = (client_arg *)server_xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    /* /metrics/stream subscribers are parked on their own condition and are
     * not woken by `cv`. Without this they would sleep out a full keepalive
     * interval each before noticing, and the client drain below would wait on
     * them — a shutdown that hangs for as long as the slowest keepalive. */
    pthread_cond_broadcast(&s.stream_cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(worker, NULL);
    s.watchdog_stop();
    {
        pulsar::ScopedLock lk(&s.mu);
        while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    }

    for (int i = 0; i < s.n_slots; i++) {
        session_slot *sl = &s.slots[i];
        /* L330: a guard-spilled bank's history is on disk already (the spill persisted it): nothing to store */
        if (!sl->provisioned || sl->spilled) continue;
        /* Tier-2: install this slot's bank so tokens/store read ITS frontier,
         * not the pool's last-live cursor (worker has joined; single-threaded).
         * Must go through server_bank_switch, NOT a bare state_restore (a
         * spilled bank's slabs were cudaFree'd, so it is skipped above).  The
         * bare restore left gpu_graph_bank_repoint to fail partway through the
         * layer loop, which both lost THIS slot's snapshot and left the graph
         * half-repointed with cur_bank stale — so the next slot whose bank
         * matched that stale cur_bank early-returned "success" over mixed views
         * and silently lost its snapshot too.  Skip the slot if it can't be
         * installed rather than persisting another conversation's frontier. */
        if (s.pool_banks > 0 && !s.bank_switch(sl->bank)) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: slot %d bank %d could not be installed at shutdown; "
                       "skipping its KV persist", i, sl->bank.raw());
            continue;
        }
        const pulsar_tokens *tokens = pulsar_session_tokens(s.sess);
        if (s.kv.enabled && tokens && tokens->len >= s.kv.opt.min_tokens) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: persisting slot %d KV cache before shutdown tokens=%d",
                       i, tokens->len);
            s.kv_cache_persist(sl, KV_PERSIST_SHUTDOWN);
        }
    }
    /* L378: the run's last ledger line, after the shutdown persists (the store's I/O published first) */
    s.kv_cache_publish_store();
    (void)s.kv_ledger_log(true);
    s.close_resources();
    free(tp_kv_dir);
    return 0;
}


#endif /* PULSAR_SERVER_TEST */

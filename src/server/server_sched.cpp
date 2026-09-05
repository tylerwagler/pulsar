/* Scheduler/worker (split move-only from generate.cpp): worker_main and its
 * quantum loop (classic / batched / fused mixed-batch lanes), job->slot
 * routing (choose_slot_for_job) and enqueue, lazy provisioning + the
 * admission/ledger/mem-floor helpers, LRU eviction and the warm-fork
 * make-room path, the proactive spill/restore guard, Tier-2 bank
 * switching, and the frontier/committed-pos readers. */
#include "pulsar_server_internal.h"
#include "pulsar_lock.hpp"



/* Tier-2 shared-pool bank switch (defined lower, near provision_slot). Installs
 * `bank`'s device views + host carry on the pool session, saving the outgoing
 * bank first. No-op in classic mode or when `bank` is already live. Called at
 * the top of every per-slot worker op (gen_begin, generate_job_step,
 * worker_evict_one) so all s->sess touches inside that op are live-correct. */
/* Returns false only when the target is a guard-spilled bank whose restore FAILED
 * (KV unrecoverable): the bank is NOT installed and cur_bank is unchanged, so the
 * caller MUST fail the request rather than sample/commit on the wrong bank's KV
 * (review finding 1). True on success (or classic mode / no-op). */
/* plan-33 inc D: evict ONE non-trunk victim (LRU-superseded preferred, else
 * plain LRU) so a warm fork has a free bank — trunk always protected. Defined
 * near worker_evict_one; forward-declared for the routing path above it. */



bool server::enqueue(job *j) {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    s->n_queued++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}



/* =========================================================================
 * Increment 3: job→slot binding + round-robin scheduler.
 *
 * Everything below runs on the single GPU worker thread. Client threads only
 * enqueue jobs (enqueue above, under mu) and block on the per-job condvar;
 * they never touch a slot or a pulsar_session. Over-capacity requests stay in
 * the FIFO queue until a slot frees (plan Tier 1 §1.4 "queue them until a
 * slot frees"); the queue is bounded by PULSAR_SERVER_MAX_CLIENTS, since every
 * queued job is one connected client thread.
 * ========================================================================= */

/* Context a request needs from a slot: prompt plus generation budget (plus a
 * small allowance for tool-error recovery injections), capped at the largest
 * (startup) slot so every request can always run on slot 0. */
int server::job_needed_ctx(const job *j) const {
    const auto *s = this; (void)s;
    int64_t need = (int64_t)j->req.prompt.len +
                   (int64_t)(j->req.max_tokens > 0 ? j->req.max_tokens
                                                   : s->default_tokens) +
                   64;
    const int ref_ctx = s->pool_banks > 0 ? s->pool_ctx_size
                                          : s->slots[0].ctx_size;
    if (need > ref_ctx) need = ref_ctx;
    if (need < 1) need = 1;
    return (int)need;
}



/* Check the price admission charged against the bytes the allocator committed.
 * Both come from the same allocation code (the price is that code run dry), so
 * they are equal, or an allocation inside the create window bypassed the
 * tensor primitives -- an accounting hole the ledger must not paper over.
 * Returns the bytes to commit (the actual), 0 on a mismatch: the caller
 * refuses to start. */
uint64_t server_reconciled_session_cost(int slot_idx, int ctx,
                                        uint64_t est_bytes,
                                        uint64_t actual_bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: slot %d session cost: priced=%.2f GiB actual=%.2f GiB (ctx=%d)",
               slot_idx, (double)est_bytes / gib, (double)actual_bytes / gib, ctx);
    if (actual_bytes == 0 || est_bytes != actual_bytes) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: SESSION COST MISMATCH: priced %llu bytes, allocated %llu "
                   "-- the price is the allocator run dry, so an allocation inside "
                   "pulsar_session_create bypassed the tensor primitives; refusing to start",
                   (unsigned long long)est_bytes, (unsigned long long)actual_bytes);
        return 0;
    }
    return actual_bytes;
}



/* MemAvailable from /proc/meminfo, in bytes (0 on parse failure — the caller
 * fails closed and refuses provisioning). One read per slot-provisioning
 * attempt — never on a token/layer hot path. Coarse by design: driver 610's
 * UVM accounting lags MemAvailable, and under UVM pressure MemTotal itself
 * shrinks, so this is a belt-and-suspenders floor check on top of the
 * ledger, not a precise gauge. */
uint64_t server_mem_available_bytes(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    }
    fclose(fp);
    return (uint64_t)kib * 1024ull;
}



/* Lazily provision a fresh slot. In pool mode (production) this is
 * provision_bank: pure host bookkeeping over the ONE pool session. In classic
 * mode (pool_banks == 0) only slot 0 exists — the single session created at
 * startup — so provisioning always refuses and the job queues for slot 0.
 * Returns NULL when the pool is at capacity or MemAvailable would drop below
 * the floor — the job then waits in the queue, exactly like an admission
 * rejection.
 * Since increment 4, an evicted slot leaves a hole (provisioned == false)
 * below n_slots; provisioning reuses the lowest hole before extending the
 * pool, so n_slots stays the high-water published count every reader iterates
 * by (they all skip unprovisioned entries). *refusal reports WHY provisioning
 * failed: the eviction path only acts on refusals eviction can actually
 * relieve (a full pool or a full ledger — never the MemAvailable floor,
 * which freed CUDA memory does not promptly move; see worker_try_bind). */
/* provision_refusal (enum) moved to pulsar_server_internal.h so the server
 * member methods provision_bank/provision_slot/choose_slot_for_job can name it. */

/* ===== Tier-2 shared-pool bank plumbing =================================== */

/* Lazy bank switch on the shared pool session. server_bank_switch(s, b) saves
 * the currently-installed bank's host carry + frontier counters and installs
 * bank b's (device views + carry). Idempotent when b is already live, and a
 * no-op in classic mode. The switch-away save is what keeps every idle bank's
 * carry current, so the per-bank frontier readers (pulsar_session_bank_*) and
 * server_slot_frontier_pos are always correct for non-live banks. */
/* Tier-2 2b: restore a guard-spilled bank's physical + raw KV from disk before it
 * is installed. Defined below (near the guard); forward-declared for the switch. */

bool server::bank_switch(int bank) {
    auto *s = this;
    if (s->pool_banks <= 0) return true;          /* classic mode */
    if (bank < 0 || bank >= s->pool_banks) return true;
    if (bank == s->live_bank && !s->slots[bank].spilled) return true; /* already installed */
    pulsar_session *pool = s->sess;
    /* Snapshot the outgoing bank so its carry stays readable while idle, and
     * record its committed frontier for metrics/routing. live_bank < 0 is the
     * post-batched-quantum sentinel: the pool holds no single bank's clean
     * state (multiseq left a cross-bank superset), so there is nothing safe to
     * save — just install the target (bank_state_restore clears the poison). */
    if (s->live_bank >= 0 && s->live_bank != bank) {
        s->slots[s->live_bank].committed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)s->live_bank);
    }
    /* Tier-2 2b: a guard-spilled target has no physical — reallocate + reload its
     * KV bit-identically from disk before installing (the reload-stall cost). On
     * restore failure the bank is NOT installed and cur_bank is unchanged; return
     * false so the caller fails the request (finding 1). */
    if (s->slots[bank].spilled) {
        if (!s->bank_restore_spilled(bank)) return false;
    }
    /* The state restore can ALSO fail (gpu_graph_bank_repoint rejects a bank
     * whose slabs are missing, e.g. after a partially-failed realloc).  Ignoring
     * it published live_bank over a half-repointed view set with cur_bank
     * unchanged, so the caller ran engine work against mixed banks instead of
     * failing the request.  Fail loud here — this is the one check that catches
     * a bad repoint before any token is generated. */
    if (!pulsar_session_bank_state_restore(pool, (uint32_t)bank)) return false;
    s->live_bank = bank;
    s->slots[bank].committed_pos = pulsar_session_pos(pool);
    return true;
}

int server::slot_frontier_pos(const session_slot *sl) const {
    const auto *s = this;
    if (!sl || !sl->provisioned) return 0;
    if (!s->sess) return 0; /* host-only server (unit tests): no session to read */
    if (s->pool_banks > 0) return pulsar_session_bank_pos(s->sess, sl->bank);
    return pulsar_session_pos(s->sess);
}

/* Human name for a pulsar_session_bank_fork_partial refusal code, for the
 * routing logs (a bare "refused" hid the ring-scrolled class for a day). */
static const char *server_fork_rc_name(int rc) {
    switch (rc) {
    case PULSAR_FORK_OK:            return "ok";
    case PULSAR_FORK_EINVAL:        return "einval";
    case PULSAR_FORK_SHALLOW:       return "cut-too-shallow";
    case PULSAR_FORK_NOHIST:        return "no-history";
    case PULSAR_FORK_MISMATCH:      return "token-mismatch";
    case PULSAR_FORK_EVICTED:       return "src-evicted";
    case PULSAR_FORK_RING_SCROLLED: return "ring-scrolled";
    case PULSAR_FORK_COPY_FAIL:     return "copy-fail";
    default:                        return "unknown";
    }
}

int server::slot_common_prefix(const session_slot *sl,
                               const pulsar_tokens *prompt) const {
    const auto *s = this;
    pulsar_prefix_match m;
    s->slot_prefix_match(sl, prompt, &m);
    return m.live_cut;
}

/* L115: the one prefix-reuse question, asked of a slot.  Callers that need
 * the request-side count (accounting, prefill bounds) read `prompt_cut`;
 * callers that need KV rows or a cut point read `live_cut`.  Across a seam
 * these differ, and keeping them in one struct is what stops a live-side
 * count from being used to index the request array. */
void server::slot_prefix_match(const session_slot *sl, const pulsar_tokens *prompt,
                               pulsar_prefix_match *out) const {
    const auto *s = this;
    out->live_cut = 0;
    out->prompt_cut = 0;
    out->seamed = false;
    if (!sl || !sl->provisioned) return;
    if (s->pool_banks > 0)
        pulsar_session_bank_prefix_match(s->sess, sl->bank, prompt, out);
    else
        pulsar_session_prefix_match(s->sess, prompt, out);
}

/* Provision a bank in the shared pool (Tier-2). No GPU allocation happens here:
 * the whole n-bank pool was allocated and admitted ONCE at startup, so this is
 * pure host bookkeeping — find a free bank slot, install it, reset it to an
 * empty conversation (so gen_begin sees pos 0 / no stale prefix), and charge
 * the even-split per-bank marginal to the ledger. The only runtime pressure is
 * the demand-paged comp/index pages a bank touches as it fills; the belt-and-
 * suspenders MemAvailable floor still guards each provision. Returns NULL with
 * *refusal set on a full pool or a tight box (never on a create/admission
 * failure — there is no runtime create). */
/* provision_bank's MemAvailable verdict (L179 branch 14). The FIRST bank is
 * exempt (see the caller); from the second on, an unreadable gauge (avail ==
 * 0) or a box below floor + marginal refuses. */
static bool server_bank_floor_refuses(int n_provisioned, uint64_t avail, uint64_t marginal) {
    return n_provisioned > 0 && (avail == 0 || !server_mem_floor_admits(avail, marginal));
}

bool warn_limiter_due(warn_limiter *w, double now_sec, double period_sec, unsigned *skipped) {
    if (w->last_sec != 0.0 && now_sec - w->last_sec < period_sec) {
        w->suppressed++;
        return false;
    }
    *skipped = w->suppressed;
    w->suppressed = 0;
    w->last_sec = now_sec;
    return true;
}

session_slot *server::provision_bank(provision_refusal *refusal) {
    auto *s = this;
    *refusal = PROVISION_OK;
    int idx = -1;
    /* Pool mode: bank 0 is an ordinary bank (no boot pre-provisioning, no
     * pinning) — the uniform loop from 0 replaced the boot-phantom /
     * empty-reuse / soft-evict special-case family (2026-08-10). */
    int n_provisioned = 0;
    for (int i = 0; i < s->pool_banks; i++) {
        if (s->slots[i].provisioned) { n_provisioned++; continue; }
        if (idx < 0) idx = i;
    }
    if (idx < 0) { *refusal = PROVISION_REFUSED_POOL_FULL; return NULL; }
    /* Belt-and-suspenders: refuse if the box is physically tight (fail closed on
     * an unreadable gauge), matching provision_slot. The marginal is what the
     * bank may still demand-page as it fills.
     * The FIRST bank is exempt: before the 2026-08-10 slot-0 rework it was
     * boot-provisioned and never floor-gated, and an empty-pool floor refusal
     * is a hang — worker_main's wait predicate stays false (head queued,
     * nothing active), so the worker hard-spins re-reading /proc/meminfo and
     * the first request never completes.  One bank's eager floor is small;
     * admission pressure on a tight box belongs to the SECOND bank onward. */
    const uint64_t avail = server_mem_available_bytes();
    if (server_bank_floor_refuses(n_provisioned, avail, s->bank_marginal_bytes)) {
        /* A per-request condition: the head job is re-tried every quantum and
         * the next job meets the same floor.  One line per period with the
         * count it swallowed (L190 C1); a once-per-process line hid every
         * refusal after the first. */
        unsigned skipped = 0;
        if (warn_limiter_due(&s->mem_floor_warn, server_now_sec(),
                             PULSAR_SERVER_MEM_FLOOR_WARN_SEC, &skipped)) {
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: bank provisioning refused: MemAvailable %.2f GiB "
                       "below floor for marginal %.2f GiB (job queued; %u refusals "
                       "in the last %.0f s not logged)",
                       (double)avail / (1024.0 * 1024.0 * 1024.0),
                       (double)s->bank_marginal_bytes / (1024.0 * 1024.0 * 1024.0),
                       skipped, PULSAR_SERVER_MEM_FLOOR_WARN_SEC);
        }
        *refusal = PROVISION_REFUSED_MEM_FLOOR;
        return NULL;
    }
    session_slot *sl = &s->slots[idx];
    pulsar_session *pool = s->sess;
    /* Install and reset the bank to an empty conversation with a valid (empty)
     * host carry, so routing/metrics read pos 0 and gen_begin cold-prefills. A
     * free (SLOT_EVICTED) bank is never guard-spilled (spilled banks stay
     * provisioned), so this never restores from disk -- but the state restore
     * that installs the bank can still refuse (bank_switch's contract: a bank
     * whose slabs are missing).  Then the bank is NOT installed and live_bank
     * still names another bank; resetting and saving "bank idx" here would
     * clobber THAT bank's carry with an empty one.  Refuse the provision and
     * let worker_try_bind fail the job (eviction cannot relieve this). */
    if (!s->bank_switch(idx)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: bank %d install failed (state restore refused); "
                   "provisioning refused, job fails", idx);
        *refusal = PROVISION_REFUSED_CREATE_FAIL;
        return NULL;
    }
    pulsar_session_invalidate(pool);
    pulsar_session_bank_state_save(pool, (uint32_t)idx);
    sl->provisioned = true;
    sl->bank = (uint32_t)idx;
    sl->committed_pos = 0;
    sl->state = SLOT_IDLE;
    sl->ctx_size = s->pool_ctx_size;              /* every bank shares pool ctx */
    sl->est_cost_bytes = s->bank_marginal_bytes;
    sl->tokens_emitted = 0;
    sl->prefill_counted = 0;
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    sl->continued_last_store_tokens = 0;
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes += s->bank_marginal_bytes;
    if (idx >= s->n_slots) s->n_slots = idx + 1;
    pthread_mutex_unlock(&s->mu);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: provisioned bank %d (pooled) marginal=%.2f GiB "
               "total committed=%.2f GiB / %.2f GiB",
               idx,
               (double)s->bank_marginal_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_committed_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0));
    return sl;
}

session_slot *server::provision_slot(int ctx,
                                    provision_refusal *refusal) {
    auto *s = this;
    if (s->pool_banks > 0) { (void)ctx; return s->provision_bank(refusal); }
    /* Classic mode (pool_banks == 0): the ONE session IS slot 0, created at
     * startup — there is no independent session to create, so a request that
     * needs another slot queues until slot 0 frees. */
    (void)ctx;
    *refusal = PROVISION_REFUSED_POOL_FULL;
    return NULL;
}



/* Context a lazily provisioned slot would be created with for this job: the
 * secondary-slot default, raised to the job's need, capped at slot 0's ctx.
 * Shared by the provisioning path and the eviction could-it-help precheck so
 * they price the same session shape. */
int server::provision_ctx_for_job(const job *j) const {
    const auto *s = this;
    int ctx = PULSAR_SERVER_EXTRA_SLOT_CTX_TOKENS;
    const int cap_ctx = s->pool_banks > 0 ? s->pool_ctx_size
                                          : s->slots[0].ctx_size;
    if (ctx > cap_ctx) ctx = cap_ctx;
    const int needed = s->job_needed_ctx(j);
    if (ctx < needed) ctx = needed;
    return ctx;
}



/* Trivial-match classifier for the router's choose-vs-provision decision
 * (unit-tested in cli_main.cpp). A candidate slot's token-prefix match is
 * TRIVIAL — grounds to prefer provisioning a fresh slot over reusing (and
 * clobbering) the candidate — only when BOTH hold:
 *   - common < share_ceiling: the match is no deeper than the SHARED PREFIX
 *     two UNRELATED conversations from the same client render before their
 *     task-specific content. The static startup floor covers BOS + the
 *     reasoning-effort preamble, but a tools-advertising client (Claude Code)
 *     also renders a fixed ~1.7 KB tool-instruction block plus its schemas and
 *     usually a stable system prompt into that prefix, byte-identical across
 *     its conversations — hundreds to thousands of tokens the startup constant
 *     cannot see. So the caller raises share_ceiling to this job's chat anchor
 *     (the last user marker before the first assistant, i.e. the end of the
 *     shared scaffolding), and a match no deeper than that does not indicate
 *     the same conversation;
 *   - slot_pos - common >= protect_floor: reuse would destroy a meaningful
 *     amount of some conversation's warm KV. When the slot holds less than
 *     that past the match, clobbering costs at worst a sub-threshold
 *     re-prefill (sub-second) — always cheaper than a multi-GiB,
 *     multi-second session create — and this clause also keeps short
 *     same-conversation continuations (whose common covers nearly the whole
 *     slot state) on their warm slot. An empty slot (slot_pos == 0) is
 *     never "clobbered": it is simply free.
 * protect_floor stays the static startup threshold; only share_ceiling is
 * lifted per job, so raising the ceiling never changes what counts as a slot
 * "worth protecting".
 * Deliberate semantic change from v0.2.0 (common == 0 && pos > 0): a slot
 * holding only a sub-threshold warm tail past the match is now reused
 * (clobbered) rather than protected by a fresh provisioning — protecting
 * <threshold tokens of KV is never worth seconds of session create. */
bool server_slot_match_is_trivial(int common, int slot_pos,
                                  int share_ceiling, int protect_floor) {
    return common < share_ceiling && slot_pos - common >= protect_floor;
}



/* CROSS-WIRE ROOT FIX (L179 branch 3): the router's divergent-match verdict.
 * In-place continuation is safe ONLY for a linear extension of THIS bank's
 * own conversation (best_common == frontier). When best_common < frontier the
 * bank holds a DIFFERENT conversation past the shared prefix (typically just
 * the system-prompt header); continuing in place REWINDS-and-clobbers it. On
 * the shared pool session that conversation is often still live under
 * concurrency, so the two interleave and a request decodes another's KV (the
 * cross-wire). Deep divergent matches (best_common >= warm_partial_min)
 * already FORKED and never reach here.
 *   NOT_DIVERGENT : no best, or best_common >= frontier -- the caller
 *                   continues on best as before.
 *   FRESH         : divergent and a fresh bank was provisioned (the shared
 *                   prefix is cheap and prefix-cached): route there.
 *   QUEUE         : divergent, no fresh bank, and some job is active: queue
 *                   rather than clobber -- the active job frees a bank as it
 *                   finishes AND there is a live conversation worth protecting.
 *   IN_PLACE      : divergent, no fresh bank, NOTHING active: no bank will
 *                   ever free (idle banks may all be protected), so queuing
 *                   would LIVE-LOCK the worker -- and with no live reader there
 *                   is nothing to corrupt -- so continue in place (progress
 *                   over warmth). The corruption only ever happened while a
 *                   concurrent conversation was live.
 * The side effects (provisioning, the log line, *refusal) stay in the caller. */
enum divergent_route { ROUTE_NOT_DIVERGENT, ROUTE_FRESH, ROUTE_QUEUE, ROUTE_IN_PLACE };

static int divergent_route_decision(bool have_best, int best_common, int frontier,
                                    bool provisioned_fresh, bool any_active) {
    if (!have_best || best_common >= frontier) return ROUTE_NOT_DIVERGENT;
    if (provisioned_fresh) return ROUTE_FRESH;
    return any_active ? ROUTE_QUEUE : ROUTE_IN_PLACE;
}

/* Warm-advance-in-place (L179 branch 5): at a pool full of LIVE banks a PARTIAL
 * fork consumes the requester's OWN trunk (pulsar_session_bank_fork_partial with
 * src == dst, the engine's documented truncate-reuse degenerate) instead of
 * evicting anyone -- this is what makes capacity == banks work. Eligible iff no
 * destination was provisioned, the route is a partial cut (a FULL fork has no
 * cut and only ever forks into a distinct bank) and the refusal was POOL_FULL:
 * any other refusal (admission, mem floor, create fail) is not a "full of live
 * banks" verdict and falls through to the cold path. */
static bool warm_inplace_eligible(bool have_dst, bool partial, provision_refusal fr) {
    return !have_dst && partial && fr == PROVISION_REFUSED_POOL_FULL;
}

/* The commit after a successful in-place cut. The cut moved the frontier
 * BACKWARD to the engine's R-aligned resume position; the pre-truncation
 * continued-store watermark would then refuse every continued disk checkpoint
 * until the new conversation outgrows the old frontier (gen_begin only resets
 * it when cached == 0, which a successful cut is exactly not) -- so it is
 * reset here, with the committed position. Nothing else on the slot moves. */
static void warm_inplace_commit(session_slot *sl, int resume_pos) {
    sl->committed_pos = resume_pos;
    sl->continued_last_store_tokens = 0;
}

/* Route the job to a slot. Preferences, in order:
 *   1. A live-tool-state continuation binds to the slot that owns its call
 *      ids (waiting for it if busy — running it elsewhere could only 409).
 *      A continuation whose prompt cannot fit its owner slot's context can
 *      never run: it must not run elsewhere (the live tool state exists only
 *      on the owner), and leaving it queued would wedge the FIFO forever
 *      behind an unbindable head — so it is failed explicitly through
 *      *reject_ctx with the same context_length_exceeded client error the
 *      front door sends (http_server.cpp / request_exceeds_context; the front
 *      door checks against slot 0's ctx and cannot see the owner's smaller
 *      one).
 *   2. A free fitting slot whose live thinking binding byte-matches the
 *      request's visible transcript is that conversation's warm continuation
 *      and wins outright (thinking_live_binds_prompt): for thinking chats
 *      the client replays visible content while the slot's frontier holds
 *      the hidden reasoning, so the token common prefix understates
 *      relatedness and must not out-vote the binding.
 *   3. Among free slots with enough context, the longest common token prefix
 *      wins, keeping a client's follow-ups on their warm KV.
 *   4. A job whose best token match is TRIVIAL — header-deep only, against a
 *      slot holding meaningful warm state past the match
 *      (server_slot_match_is_trivial) — prefers a fresh lazily provisioned
 *      slot over clobbering that conversation (budget permitting); with the
 *      pool exhausted it falls back to the warmest free slot exactly like
 *      the single-session server did. (Through v0.2.0 this gate required
 *      common == 0, which rendered chat traffic can never produce — every
 *      rendered prompt shares the template header, measured 4–9 common
 *      tokens across distinct conversations — so sequential conversations
 *      always clobbered slot 0 and the pool never provisioned; task #24
 *      bounce repro, fixed in task #30.)
 * Returns NULL when the job must wait for a slot to free — except when
 * *reject_ctx is set nonzero (the owner slot's ctx_size), which means the
 * job can never run and must be failed, not left queued. *waiting_owner is
 * set when the NULL means "the continuation's owner slot is busy": eviction
 * cannot help that job, only the owner finishing can. *clobbers is set when
 * the returned slot would overwrite another conversation's warm KV — the
 * caller may prefer evict(LRU)+provision over that (increment 4). *refusal
 * reports why a fresh provisioning was refused (PROVISION_OK when none was
 * attempted or it succeeded) so the eviction path can act only on refusals
 * eviction relieves. */
session_slot *server::choose_slot_for_job(job *j, int *reject_ctx,
                                         bool *waiting_owner, bool *clobbers,
                                         provision_refusal *refusal) {
    auto *s = this;
    *waiting_owner = false;
    *clobbers = false;
    *refusal = PROVISION_OK;
    session_slot *owner = NULL;
    if (j->req.api == API_RESPONSES && j->req.responses_live_call_ids.len > 0) {
        owner = s->responses_live_slot_for_ids(&j->req.responses_live_call_ids);
    } else if (j->req.api == API_ANTHROPIC &&
               j->req.anthropic_live_call_ids.len > 0) {
        owner = s->anthropic_live_slot_for_ids(&j->req.anthropic_live_call_ids);
    }
    if (owner) {
        if (request_exceeds_context(&j->req, owner->ctx_size)) {
            *reject_ctx = owner->ctx_size;
            return NULL;
        }
        *waiting_owner = owner->active_job != NULL;
        return owner->active_job ? NULL : owner;
    }

    const int needed = s->job_needed_ctx(j);
    session_slot *best = NULL;
    int best_common = -1;
    session_slot *bound = NULL;   /* thinking-binding match (preference 2) */
    size_t bound_visible = 0;     /* longest key = most recent frontier */
    static const int route_debug = getenv("PULSAR_ROUTE_DEBUG") != NULL;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (route_debug) {
            const int dbg_common = sl->provisioned && !sl->active_job
                ? s->slot_common_prefix(sl, &j->req.prompt) : -1;
            const pulsar_tokens *bank_toks = sl->provisioned
                ? pulsar_session_bank_tokens(s->sess, sl->bank) : NULL;
            const int bt = (bank_toks && dbg_common >= 0 &&
                            dbg_common < bank_toks->len) ? bank_toks->v[dbg_common] : -1;
            const int pt = (dbg_common >= 0 && dbg_common < j->req.prompt.len)
                ? j->req.prompt.v[dbg_common] : -1;
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: route-scan slot %d bank %u: active=%d "
                       "provisioned=%d ctx=%d needed=%d frontier=%d common=%d "
                       "bank[c]=%d prompt[c]=%d",
                       i, sl->bank, sl->active_job != NULL, (int)sl->provisioned,
                       sl->ctx_size, needed,
                       sl->provisioned ? s->slot_frontier_pos(sl) : -1,
                       dbg_common, bt, pt);
        }
        if (sl->active_job || !sl->provisioned) continue;
        if (sl->ctx_size < needed) continue;
        const size_t visible =
            s->thinking_live_binds_prompt(sl, &j->req,
                                       s->slot_frontier_pos(sl));
        if (visible > bound_visible) {
            bound_visible = visible;
            bound = sl;
        }
        const int common = s->slot_common_prefix(sl, &j->req.prompt);
        if (common > best_common) {
            best_common = common;
            best = sl;
        }
    }
    if (bound) {
        /* Same never-silent rule as the fork log below: a thinking-bind hit
         * routes onto live KV whose retained reasoning EXCEEDS the visible
         * transcript, so whether it fired must be answerable from the log
         * (the 2026-08-09 eval-variance hunt could not tell). */
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: route: thinking-bind bank %u visible=%zu "
                   "prompt=%zu",
                   bound->bank, bound_visible,
                   j->req.prompt_text ? strlen(j->req.prompt_text) : 0);
        return bound;
    }
    /* The shared-prefix ceiling is per job, not a global constant: a
     * tools-advertising client renders a large fixed tool/system block before
     * any task content, so two UNRELATED conversations share far more than the
     * startup BOS+preamble threshold. This job's chat anchor (end of the shared
     * scaffolding) is that ceiling; without it, every fresh conversation's
     * template-deep match reads as "related" and clobbers an unrelated warm
     * slot (the round-robin bounce under Claude Code). protect_floor stays the
     * static threshold. */
    int share_ceiling = s->slot_trivial_common_tokens;
    const int job_anchor = kv_cache_chat_anchor_pos(&s->kv, &j->req.prompt,
                                                    pulsar_token_user(s->engine),
                                                    pulsar_token_assistant(s->engine));
    if (job_anchor > 0) {
        const int anchor_ceiling =
            job_anchor + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
        if (anchor_ceiling > share_ceiling) share_ceiling = anchor_ceiling;
    }
    const bool best_clobbers_warm_state =
        best && server_slot_match_is_trivial(best_common,
                                             s->slot_frontier_pos(best),
                                             share_ceiling,
                                             s->slot_trivial_common_tokens);
    if (!best || best_clobbers_warm_state) {
        if (best_clobbers_warm_state) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: slot routing: best match is trivial "
                       "(common=%d pos=%d ceiling=%d floor=%d anchor=%d); "
                       "preferring a fresh slot",
                       best_common, s->slot_frontier_pos(best),
                       share_ceiling, s->slot_trivial_common_tokens, job_anchor);
        }
        session_slot *fresh = s->provision_slot(s->provision_ctx_for_job(j),
                                             refusal);
        if (fresh) {
            if (route_debug)
                server_log(PULSAR_LOG_DEFAULT,
                           "pulsar-server: route-branch: FRESH bank %u "
                           "(best_common %d, best %d)",
                           fresh->bank, best_common,
                           best ? (int)best->bank : -1);
            return fresh;
        }
        /* Pool full of idle stale banks: without eviction HERE, every new
         * conversation falls through to the same trivial-match bank and
         * clobbers it serially while the other banks sit pinned with dead
         * state forever — measured 2026-08-10: an 8-bank pool pre-churned by
         * 12 one-shots served every later conversation cold off bank 1, and
         * warm-turn TTFT never engaged (2.35 s vs the 0.6 s a fresh pool
         * gives).  Evict an LRU idle bank (live-tool owners protected, no
         * trunk to preserve) and retry once, mirroring the fork path. */
        if (*refusal == PROVISION_REFUSED_POOL_FULL && s->fork_make_room(NULL)) {
            fresh = s->provision_slot(s->provision_ctx_for_job(j), refusal);
            if (fresh) return fresh;
        }
    }
    /* plan-33 inc B/D: warm FORK routing. `best`'s committed history shares a
     * token prefix `best_common` with the request (validated bit-for-bit inside
     * the fork before any device write). Instead of continuing IN PLACE (which
     * consumes the trunk — today's behavior), fork the trunk into another bank
     * and continue there, leaving the trunk INTACT so a divergent sibling keeps
     * matching it:
     *   - inc B FULL fork   : best_common == trunk frontier  -> pulsar_session_bank_fork
     *                         (dst resumes at the exact frontier; re-prefill suffix).
     *   - inc D PARTIAL cut : warm_partial_min <= best_common < frontier
     *                         -> pulsar_session_bank_fork_partial (engine aligns down
     *                         to R, byte-stashes the ratio-4 boundary row; dst's
     *                         committed history becomes tokens[0..R), re-prefill [R..).
     * A free bank is used first; else one non-trunk victim is evicted (make_room,
     * LRU-superseded preferred). Any refusal (history moved, evicted src, cut
     * below R, no evictable bank) degrades to today's path — never a client error. */
    const int frontier = best ? s->slot_frontier_pos(best) : 0;
    /* `best_common <= prompt.len`, NOT `<`. A client that ROLLS BACK or compacts
     * history resends a prompt that is a strict prefix of the trunk's committed
     * tokens, giving best_common == prompt.len < frontier. Under `<` that case
     * failed warm_ok, skipped both fork paths, and fell into the cross-wire guard
     * below — which correctly refuses to clobber the trunk and provisions a FRESH
     * bank, i.e. a FULL cold prefill of an already-resident prompt. Measured
     * 2026-08-11: `common 164812 frontier 165045 prompt 164812` re-prefilled
     * 123,852 tokens (~2.5 min) with the whole prompt already in bank 0. It also
     * broke the invariant the cross-wire comment below asserts ("deep divergent
     * matches already FORKED above and never reach here"). The partial cut handles
     * it exactly: align down from best_common, re-prefill only the remainder.
     *
     * `full` keeps the strict `<`: a FULL fork exists to re-prefill a suffix, and
     * best_common == prompt.len == frontier means the trunk IS the prompt, with no
     * suffix and no divergence. That is a plain in-place continuation and must stay
     * one — forking there would copy a bank to do nothing. */
    const bool warm_ok = s->pool_banks > 0 && s->warm_fork_enabled && best &&
                         !best_clobbers_warm_state && !best->active_job &&
                         best_common > 0 && best_common <= j->req.prompt.len;
    const bool full    = warm_ok && best_common == frontier &&
                         best_common < j->req.prompt.len;                    /* inc B */
    /* inc D geometry alone isn't enough: the engine's raw ring may have
     * scrolled past the cut (typical after a client compacts history on a
     * deep bank), which makes every partial fork AND the in-place advance
     * permanently infeasible — the ring only moves forward. Probe before
     * proposing, so a doomed FORK-partial routes to the divergent/fresh path
     * immediately instead of re-attempting an impossible fork every quantum
     * (observed 2026-08-10: ~30 s of refusal retries stalled a compacted
     * 42k-token turn before an eviction finally let it cold-prefill). */
    const bool partial_geom = warm_ok && !full && best_common >= s->warm_partial_min &&
                              best_common < frontier;
    const int  partial_rc   = partial_geom
        ? pulsar_session_bank_fork_partial_feasible(s->sess, best->bank, best_common)
        : PULSAR_FORK_OK;
    const bool partial = partial_geom && partial_rc == PULSAR_FORK_OK;       /* inc D */
    /* Always-on routing-decision inputs, so a 0-fork count is never silent (the
     * verbose KVCACHE stream; one line per bind, not per token). Confirmed nuance:
     * re-tokenized generated tail rarely reproduces the trunk's exact frontier, so
     * best_common < frontier (the PARTIAL path) is the common case; full is the
     * rare exact-continuation. */
    char infeasible[48] = "";
    if (partial_geom && !partial)
        snprintf(infeasible, sizeof infeasible, " [partial-infeasible: %s]",
                 server_fork_rc_name(partial_rc));
    if (s->pool_banks > 0 && s->warm_fork_enabled)
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: route: best bank %d common %d frontier %d prompt %d "
                   "partial_min %d -> %s%s%s%s",
                   best ? (int)best->bank : -1, best_common, frontier,
                   j->req.prompt.len, s->warm_partial_min,
                   full ? "FORK-full" : partial ? "FORK-partial" : "in-place/cold",
                   best_clobbers_warm_state ? " [trivial]" : "",
                   (best && best->active_job) ? " [busy]" : "",
                   infeasible);
    if (full || partial) {
        provision_refusal fr;
        session_slot *dst = s->provision_slot(s->provision_ctx_for_job(j), &fr);
        if (!dst && fr == PROVISION_REFUSED_POOL_FULL &&
            s->fork_make_room(best, /*superseded_only=*/true)) {
            /* Freed a SUPERSEDED victim; the trunk was protected. Retry once.
             * Live banks are never evicted for a fork: under cyclic
             * multi-tenant traffic the LRU live bank is exactly the next
             * returning conversation (the measured domino), and the partial
             * case has a strictly better option below — advance IN PLACE. */
            dst = s->provision_slot(s->provision_ctx_for_job(j), &fr);
        }
        if (warm_inplace_eligible(dst != NULL, partial, fr)) {
            /* Pool full of LIVE banks: consume the requester's OWN trunk (see
             * warm_inplace_eligible) -- cut to the R-aligned common, then
             * re-prefill only the suffix. Every returning conversation advances
             * its own bank warm, nobody's state dies. The trunk is not
             * preserved for siblings — at a full pool that luxury costs another
             * conversation its warmth. */
            const int rc = pulsar_session_bank_fork_partial(
                    s->sess, best->bank, best->bank,
                    j->req.prompt.v, j->req.prompt.len, best_common);
            if (rc == 0) {
                warm_inplace_commit(best, pulsar_session_bank_pos(s->sess, best->bank));
                server_log(PULSAR_LOG_DEFAULT,
                           "pulsar-server: warm-advance-in-place: bank %u cut "
                           "(frontier %d, common %d) resume %d; no eviction",
                           best->bank, frontier, best_common,
                           best->committed_pos);
                *clobbers = false;
                return best;
            }
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: warm-advance-in-place refused (bank %u, %s); "
                       "falling through", best->bank, server_fork_rc_name(rc));
        }
        if (dst && dst != best) {
            pulsar_session *pool = s->sess;
            const int rc = full
                ? pulsar_session_bank_fork(pool, best->bank, dst->bank,
                                        j->req.prompt.v, j->req.prompt.len, best_common)
                : pulsar_session_bank_fork_partial(pool, best->bank, dst->bank,
                                                j->req.prompt.v, j->req.prompt.len,
                                                best_common);
            if (rc == 0) {
                /* FULL resumes at best_common; PARTIAL resumes at the engine's
                 * R-aligned cut (read it back rather than recompute the align). */
                dst->committed_pos = full ? best_common
                                          : pulsar_session_bank_pos(pool, dst->bank);
                server_log(PULSAR_LOG_DEFAULT,
                           "pulsar-server: warm-fork-%s: trunk bank %u (frontier %d, "
                           "common %d) -> bank %u (resume %d); trunk preserved",
                           full ? "full" : "partial", best->bank, frontier,
                           best_common, dst->bank, dst->committed_pos);
                *clobbers = false;
                return dst;
            }
            /* Refused (history moved / evicted src / cut below R): dst stays a
             * fresh empty bank — cold on it is safe and beats clobbering best. */
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: warm-fork-%s refused (bank %u, %s); cold on bank %u",
                       full ? "full" : "partial", best->bank,
                       server_fork_rc_name(rc), dst->bank);
            *clobbers = false;
            return dst;
        }
        /* No free/evictable bank for the fork: fall through to the divergent
         * guard below (reuses in place only for a linear continuation; otherwise
         * queues rather than clobber). */
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: warm-fork-%s wanted (bank %u common %d) but no free bank",
                   full ? "full" : "partial", best->bank, best_common);
    }
    /* CROSS-WIRE ROOT FIX: the verdict is divergent_route_decision's (see it
     * for the four routes); the provisioning, the log line and *refusal are
     * this caller's. */
    if (divergent_route_decision(best != NULL, best_common, frontier, false, false)
            != ROUTE_NOT_DIVERGENT) {
        session_slot *fresh = s->provision_slot(s->provision_ctx_for_job(j), refusal);
        bool any_active = false;
        for (int i = 0; i < s->n_slots; i++)
            if (s->slots[i].active_job) { any_active = true; break; }
        switch (divergent_route_decision(best != NULL, best_common, frontier, fresh != NULL,
                                         any_active)) {
        case ROUTE_FRESH:
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: divergent match bank %u (common %d < frontier %d): "
                       "fresh bank %u, no in-place clobber",
                       best->bank, best_common, frontier, fresh->bank);
            *clobbers = false;
            return fresh;
        case ROUTE_QUEUE:
            if (*refusal == PROVISION_OK) *refusal = PROVISION_REFUSED_POOL_FULL;
            return NULL;   /* an active job will free a bank; worker retries */
        case ROUTE_IN_PLACE:
        case ROUTE_NOT_DIVERGENT:
            break;         /* in place: no live reader; a queue would deadlock */
        }
    }
    *clobbers = best_clobbers_warm_state;
    return best;
}



/* =========================================================================
 * Increment 4: LRU eviction of idle slots.
 *
 * Sessions were previously never freed at runtime: once the admission budget
 * was consumed by idle-but-warm sessions, new conversations either queued
 * forever (no fitting free slot) or silently clobbered the warmest idle slot
 * chosen by scan order. Now, when the queue head cannot be placed cleanly —
 * no fitting free slot, or only a slot whose warm KV belongs to another
 * conversation — AND provisioning was refused by a constraint eviction can
 * actually relieve (a full pool or a full admission ledger), the worker
 * evicts the least-recently-serviced IDLE slot: snapshot its KV to the disk
 * kv cache (the same LRU store used for shutdown/cold checkpoints — plan
 * Tier 1 §1.3 "spill target"), free the session, release its ACTUAL
 * committed bytes from the ledger, and retry placement (the freed budget +
 * slot entry let provisioning succeed). The clobber fallback remains when
 * eviction cannot help: it is an in-place eviction of that one slot
 * (gen_begin disk-stores the displaced state), so correctness is identical —
 * LRU eviction just picks a better victim and keeps warmer conversations
 * alive.
 *
 * MemAvailable-floor refusals deliberately do NOT trigger eviction. Measured
 * on the GB10 (driver 610, 2026-07-14 smoke): freeing two 2.5 GiB sessions
 * moved MemAvailable only 5.98 -> 6.07 GiB — cudaFree'd memory does not
 * promptly return to the kernel's gauge, so post-eviction provisioning kept
 * refusing on the same floor while two warm conversations were lost for
 * nothing (and the server degraded to one effective slot). A floor refusal
 * means the machine is physically tight; the honest response is to queue the
 * head until a slot frees, not to churn snapshots.
 *
 * Restore path: none of it is new. gen_begin's cache resolution already
 * prefers a disk-text snapshot over cold prefill (kv_cache_try_load), and the
 * "evict" snapshot is keyed exactly like a shutdown checkpoint — by rendered
 * transcript bytes, or by the visible protocol transcript when a live
 * responses/thinking binding covers the frontier. A returning client binds to
 * any free slot and restores from disk there; no extra routing metadata is
 * needed because the kvstore's text-prefix index IS the metadata. Live
 * tool-state continuations of an evicted slot get the protocol's honest 409
 * (their sampled frontier is gone), exactly like a server restart.
 *
 * Slot 0 is PINNED — never evicted: (a) client threads read
 * pulsar_session_ctx(s->sess) lock-free (http_server.cpp) under the
 * CUDA-state audit's immutable-after-startup exception, so freeing that
 * session would be a data race; (b) slot 0 is the only slot guaranteed to fit
 * any admissible request (job_needed_ctx caps at its ctx), which preserves
 * the scheduler invariant that an all-idle pool always binds the head (the
 * worker sleeps on the condvar when nothing is active) and guarantees the
 * pool never reaches zero slots; (c) as the largest session it is the most
 * expensive one to bring back. When small sessions need room, evicting the
 * idle secondaries (this code) frees the same budget without touching it.
 *
 * Like lazy provisioning, eviction is a deliberate multi-second quantum
 * overshoot on the single GPU worker thread: the snapshot forces a full
 * device sync + a multi-GiB D2H copy + a disk write, and pulsar_session_free
 * tears down the graph. It happens only at a scheduling boundary (bind time),
 * and only when the alternative is queueing forever.
 * ========================================================================= */

uint64_t server_ledger_release(uint64_t committed_total, uint64_t slot_cost) {
    if (slot_cost > committed_total) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: EVICTION LEDGER UNDERFLOW: releasing %.2f GiB from "
                   "%.2f GiB committed — provision/evict pairing is out of sync; "
                   "clamping the ledger to 0 (the MemAvailable floor remains the "
                   "backstop against the resulting over-admission)",
                   (double)slot_cost / (1024.0 * 1024.0 * 1024.0),
                   (double)committed_total / (1024.0 * 1024.0 * 1024.0));
        return 0;
    }
    return committed_total - slot_cost;
}



int server_evict_pick_victim(const session_slot *slots, int n_slots,
                             const bool *protect, bool allow_slot0) {
    int victim = -1;
    /* Slot 0 competes in LRU only when the caller says so (pool mode, where
     * bank 0 is just another bank): an idle conversation squatting bank 0 was
     * otherwise IMMORTAL, silently shrinking the evictable pool to N-1 and
     * seeding the churned-pool LRU domino (measured 2026-08-10: a one-shot's
     * leftovers on bank 0 forced an 8-conversation load into 7 banks, and the
     * shortage evicted a live conversation every turn forever). */
    for (int i = allow_slot0 ? 0 : 1; i < n_slots; i++) {
        const session_slot *sl = &slots[i];
        if (!sl->provisioned || sl->active_job) continue;
        if (protect && protect[i]) continue;
        if (victim < 0 ||
            sl->last_serviced_us < slots[victim].last_serviced_us ||
            (sl->last_serviced_us == slots[victim].last_serviced_us &&
             sl->est_cost_bytes < slots[victim].est_cost_bytes))
        {
            victim = i;
        }
    }
    return victim;
}



/* Mark slots some QUEUED live-tool-state continuation still needs: that KV
 * frontier exists only on its owner slot, so evicting it would turn the
 * queued job into a 409 the moment it binds. The queue is snapshotted under
 * mu; the job pointers stay valid afterwards because only this worker pops
 * jobs and each client thread blocks on its job condvar until then. The
 * owner lookups (tool_mu + session pos) run after mu is released — the two
 * locks are never nested. */
void server::worker_protect_queued_owner_slots(bool protect[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    memset(protect, 0, sizeof(protect[0]) * PULSAR_SESSION_POOL_CAP);
    job *queued[PULSAR_SERVER_MAX_CLIENTS];
    int n = 0;
    pthread_mutex_lock(&s->mu);
    for (job *q = s->head; q && n < PULSAR_SERVER_MAX_CLIENTS; q = q->next) {
        queued[n++] = q;
    }
    pthread_mutex_unlock(&s->mu);
    for (int i = 0; i < n; i++) {
        const request *r = &queued[i]->req;
        session_slot *owner = NULL;
        if (r->api == API_RESPONSES && r->responses_live_call_ids.len > 0) {
            owner = s->responses_live_slot_for_ids(&r->responses_live_call_ids);
        } else if (r->api == API_ANTHROPIC &&
                   r->anthropic_live_call_ids.len > 0) {
            owner = s->anthropic_live_slot_for_ids(&r->anthropic_live_call_ids);
        }
        if (owner) protect[owner - s->slots] = true;
    }
}

/* Soft eviction protection: OR into protect[] every bank that is some QUEUED
 * job's best USABLE warm match. Without this the fresh-path domino recurs:
 * job A's eviction lands on job B's warm trunk, and B — often the very next
 * bind — cold-replays its whole history. "Usable" is the operative word: a
 * match whose partial cut the raw ring has scrolled past (a compacted client)
 * is dead warmth and stays evictable — protecting it would evict live warmth
 * in its stead. Best-effort by contract: callers retry without this overlay
 * when it leaves no victim, so binding always progresses. Worker thread only
 * (slot_common_prefix reads engine host carries). */

/* The overlay's usability rule (L179 branch 6): a queued job's best match
 * protects its bank iff best_common >= warm_partial_min AND it is either a
 * full fork (best_common == frontier) or a partial cut the raw ring can still
 * replay (feasible_rc == PULSAR_FORK_OK). A ring-scrolled cut is dead warmth
 * and stays evictable; best_common == prompt.len (bank holds the whole
 * prompt) rides the partial predicate too -- conservative, never protects
 * dead warmth. */
static bool warm_match_usable(int best_common, int warm_partial_min, int frontier,
                              int feasible_rc) {
    if (best_common < warm_partial_min) return false;
    return best_common == frontier || feasible_rc == PULSAR_FORK_OK;
}

void server::worker_protect_queued_warm_matches(bool protect[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    if (s->pool_banks <= 0) return;
    job *queued[PULSAR_SERVER_MAX_CLIENTS];
    int n = 0;
    pthread_mutex_lock(&s->mu);
    for (job *q = s->head; q && n < PULSAR_SERVER_MAX_CLIENTS; q = q->next) {
        queued[n++] = q;
    }
    pthread_mutex_unlock(&s->mu);
    for (int i = 0; i < n; i++) {
        const job *q = queued[i];
        int best_i = -1, best_common = 0;
        for (int k = 0; k < s->n_slots && k < PULSAR_SESSION_POOL_CAP; k++) {
            const session_slot *sl = &s->slots[k];
            if (!sl->provisioned) continue;
            const int common = s->slot_common_prefix(sl, &q->req.prompt);
            if (common > best_common) { best_common = common; best_i = k; }
        }
        if (best_i < 0) continue;
        const session_slot *sl = &s->slots[best_i];
        const int frontier = s->slot_frontier_pos(sl);
        /* The ring probe is only consulted for a PARTIAL cut at/above the
         * minimum; a full fork or a below-minimum match needs none. */
        const bool probe = best_common >= s->warm_partial_min && best_common != frontier;
        const int feasible_rc = probe
            ? pulsar_session_bank_fork_partial_feasible(s->sess, sl->bank, best_common)
            : PULSAR_FORK_OK;
        if (warm_match_usable(best_common, s->warm_partial_min, frontier, feasible_rc))
            protect[best_i] = true;
    }
}



/* Pointless-eviction guard #2: evicting is only worth its cost if releasing
 * idle sessions can actually admit the provisioning the head job needs.
 * If even reclaiming EVERY unprotected idle slot leaves admission refusing,
 * skip eviction entirely — the head is genuinely waiting for a busy slot to
 * free, and evicting warm idle sessions would only churn snapshots. (Host
 * arithmetic only: pulsar_engine_session_cost_bytes is the same sizing code the
 * allocator uses, no CUDA work; runs only on failed bind attempts. Guard #1
 * is the provisioning-refusal reason check in worker_try_bind.) */
bool server::worker_eviction_could_help(const job *j,
                                       const bool *protect) {
    auto *s = this;
    const uint64_t est =
        pulsar_engine_session_cost_bytes(s->engine, s->provision_ctx_for_job(j));
    if (est == 0) return false;
    /* Model the MemAvailable floor too (2026-07-15 review): a POOL_FULL
     * refusal never consulted it, so on a physically tight box the gate
     * could open, one warm session be evicted, and the re-provisioning then
     * refuse on MEM_FLOOR anyway — the exact churn the refusal-reason gate
     * exists to prevent. Eviction does not promptly raise MemAvailable (see
     * the block comment above), so if the floor refuses NOW it will refuse
     * after the eviction too; skip. Fail closed on an unreadable gauge,
     * matching provision_slot. One /proc read per failed bind attempt. */
    const uint64_t avail = server_mem_available_bytes();
    if (!server_mem_floor_admits(avail, est)) return false;
    uint64_t reclaimable = 0;
    bool any = false;
    for (int i = 1; i < s->n_slots; i++) {
        const session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->active_job || (protect && protect[i])) continue;
        reclaimable += sl->est_cost_bytes;
        any = true;
    }
    if (!any) return false;
    pthread_mutex_lock(&s->mu);
    const uint64_t committed = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    const uint64_t after = committed > reclaimable ? committed - reclaimable : 0;
    return server_kv_admits(s->kv_budget_bytes, after, est);
}



/* Evict one idle slot (LRU victim): snapshot to the disk kv cache when
 * possible, free the session, release the ledger, and leave the slot entry
 * (provisioned == false) for provision_slot to reuse. Failure honesty: a failed or
 * unavailable snapshot only costs the returning client a re-prefill — the
 * eviction itself proceeds, and the response always belongs to the right
 * conversation because the freed KV can never be read again. Returns false
 * when nothing is evictable. Worker thread only. */
/* The eviction reset (L179 branch 11): the slot becomes a reusable hole --
 * unprovisioned, SLOT_EVICTED, no gen/job, no ctx, no ledger cost, no
 * scheduler bookkeeping, no continued-store watermark. Only these move: the
 * bank id stays (slot i -> bank i for the pool's life), `spilled` stays for
 * the caller's spill-file/physical reconciliation right after, committed_pos
 * stays (slot_frontier_pos reads 0 for an unprovisioned slot and provision
 * zeroes it on reuse), and the protocol live bindings are the caller's to
 * clear AFTER the snapshot store (their keying reads them). Returns the ctx
 * the slot was admitted for, for the caller's log line. */
static int evict_reset_slot_fields(session_slot *sl) {
    const int evicted_ctx = sl->ctx_size;
    sl->provisioned = false;
    sl->gen = NULL;
    sl->active_job = NULL;
    sl->state = SLOT_EVICTED;
    sl->ctx_size = 0;
    sl->est_cost_bytes = 0;
    sl->tokens_emitted = 0;
    sl->prefill_counted = 0;
    sl->last_serviced_us = 0;
    sl->continued_last_store_tokens = 0;
    return evicted_ctx;
}

bool server::worker_evict_one(bool protect[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    /* plan-33: protect any bank that is a live fork SOURCE mid-clone from disk
     * eviction (belt-and-suspenders — fork and evict are both worker-thread ops
     * and never interleave, but the invariant "a pinned source is never freed"
     * must hold for both eviction paths). */
    if (s->pool_banks > 0 && s->sess) {
        for (int i = 0; i < s->n_slots && i < PULSAR_SESSION_POOL_CAP; i++) {
            if (pulsar_session_bank_fork_pinned(s->sess, s->slots[i].bank)) protect[i] = true;
        }
    }
    const int vi = server_evict_pick_victim(s->slots, s->n_slots, protect,
                                            /*allow_slot0=*/s->pool_banks > 0);
    if (vi < 0) return false;
    session_slot *sl = &s->slots[vi];

    /* Tier-2: install the victim's bank so the snapshot store below reads its
     * OWN frontier (not the pool's live cursor). server_bank_switch restores a
     * guard-spilled victim first; if that restore fails we cannot snapshot its KV,
     * so skip it (this path is discarding the conversation anyway). */
    const bool switched = s->bank_switch(sl->bank);
    const pulsar_tokens *tokens = switched ? pulsar_session_tokens(s->sess) : NULL;
    const int live_tokens = tokens ? tokens->len : 0;
    bool stored = false;
    if (switched && s->kv.enabled && live_tokens >= s->kv.opt.min_tokens) {
        stored = s->kv_cache_store_current(sl, "evict");
    }
    if (!stored && live_tokens > 0) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: slot %d evicting WITHOUT a disk snapshot (%s); "
                   "a returning client re-prefills (correctness unaffected)",
                   vi,
                   !s->kv.enabled ? "kv disk cache disabled"
                   : live_tokens < s->kv.opt.min_tokens
                       ? "conversation below kv-cache min-tokens"
                       : "snapshot write failed");
    }
    /* The live protocol bindings describe a sampled frontier that is about to
     * lose its GPU state; clear them AFTER the store (snapshot keying reads
     * them). A later continuation of those ids gets the protocol's 409. */
    s->responses_live_clear(sl);
    s->anthropic_live_clear(sl);
    s->thinking_live_clear(sl);

    const uint64_t committed = sl->est_cost_bytes;
    uint64_t freed = 0;
    /* Pooled mode only (classic mode never provisions a slot >= 1, so the
     * picker returns -1 above): the pool session (s->sess, shared by every
     * bank) persists. "Evicting" bank vi means reset it to an empty
     * conversation and drop its host carry so a fresh conversation can reuse
     * it — no pulsar_session_free (that would tear down the whole pool). The
     * demand-paged comp/index pages this bank touched stay resident; the
     * ledger releases only this bank's even-split marginal. */
    pulsar_session_invalidate(s->sess);
    pulsar_session_bank_state_save(s->sess, (uint32_t)sl->bank);
    freed = committed; /* logical release; no allocator delta to verify */
    const int evicted_ctx = evict_reset_slot_fields(sl);
    /* Tier-2 2b: a slot being evicted for reuse must not carry a stale guard-spill
     * flag or leave an orphan spill file (invariant: physical freed IFF spilled).
     * server_bank_switch above restored a spilled victim (physical present, flag
     * cleared); belt-and-suspenders in case that restore failed. */
    if (s->pool_banks > 0 && sl->spilled) {
        char spath[600];
        snprintf(spath, sizeof spath, "%s/spill-bank-%u.kv", s->spill_dir, (unsigned)sl->bank);
        remove(spath);
        (void)pulsar_session_bank_alloc_physical(s->sess, sl->bank); /* empty physical for reuse */
        sl->spilled = false;
    }
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes = server_ledger_release(s->kv_committed_bytes, committed);
    const uint64_t committed_now = s->kv_committed_bytes;
    /* The evicted conversation must replay from a checkpoint on its next turn,
     * which is the dominant tail-latency source on a busy pool — worth a
     * counter of its own rather than only a log line. */
    s->m_evictions++;
    pthread_mutex_unlock(&s->mu);
    protect[vi] = true; /* freed hole; never a candidate again this round */
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: evicted slot %d ctx=%d tokens=%d snapshot=%s "
               "released=%.2f GiB (allocator freed %.2f GiB) "
               "committed now %.2f / %.2f GiB, MemAvailable %.2f GiB",
               vi, evicted_ctx, live_tokens, stored ? "disk" : "none",
               (double)freed / (1024.0 * 1024.0 * 1024.0),
               (double)freed / (1024.0 * 1024.0 * 1024.0),
               (double)committed_now / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)server_mem_available_bytes() / (1024.0 * 1024.0 * 1024.0));
    return true;
}



/* plan-33 inc D victim policy: an idle bank is LRU-SUPERSEDED when its whole
 * committed history is a token-prefix of ANOTHER live bank's history (a sibling
 * that already extends past it) — its KV is redundant, so evicting it loses the
 * least. Returns such a slot's index (the least-recently-served among them), or
 * -1. Pure host reads (pulsar_session_bank_tokens / _common_prefix are the same
 * host-carry reads routing already uses on idle banks; no CUDA, no install). */
/* The supersession scan itself (L179 branch 6), over injected per-slot facts:
 * a is a CANDIDATE iff eligible[a] (provisioned, idle, not fork-pinned -- the
 * caller's engine reads), not protected, and hist_len[a] > 0 (an empty bank
 * is plain LRU's business). Slot k SUPERSEDES a iff frontier[k] > hist_len[a]
 * (k is strictly longer; an unprovisioned k has frontier 0) AND common[a][k]
 * >= hist_len[a] (a's whole history is k's prefix). Among superseded
 * candidates the least-recently-served (smallest last_us, first index on a
 * tie) wins; -1 when none is superseded. common[a] may be NULL for a
 * non-candidate row and common[a][k] is only read once frontier[k] >
 * hist_len[a] holds (the caller fills it under the same test). */
static int superseded_pick_core(int n_slots, const bool *protect, const bool *eligible,
                                const int *hist_len, const int *frontier,
                                const int *const *common, const uint64_t *last_us) {
    int victim = -1;
    for (int i = 0; i < n_slots; i++) {
        if (!eligible[i] || (protect && protect[i]) || hist_len[i] <= 0) continue;
        bool superseded = false;
        for (int k = 0; k < n_slots && !superseded; k++) {
            if (k == i) continue;
            if (frontier[k] > hist_len[i] && common[i][k] >= hist_len[i]) superseded = true;
        }
        if (!superseded) continue;
        if (victim < 0 || last_us[i] < last_us[victim]) victim = i;
    }
    return victim;
}

int server::pick_superseded_idle(const bool *protect) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    if (!pool) return -1;
    bool eligible[PULSAR_SESSION_POOL_CAP];
    int hist_len[PULSAR_SESSION_POOL_CAP];
    int frontier[PULSAR_SESSION_POOL_CAP];
    uint64_t last_us[PULSAR_SESSION_POOL_CAP];
    const pulsar_tokens *hist[PULSAR_SESSION_POOL_CAP];
    int common_rows[PULSAR_SESSION_POOL_CAP][PULSAR_SESSION_POOL_CAP];
    const int *common[PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < s->n_slots; i++) {
        const session_slot *a = &s->slots[i];
        eligible[i] = a->provisioned && !a->active_job &&
                      !pulsar_session_bank_fork_pinned(pool, a->bank);
        hist[i] = eligible[i] ? pulsar_session_bank_tokens(pool, a->bank) : NULL;
        hist_len[i] = hist[i] ? hist[i]->len : 0;
        frontier[i] = s->slot_frontier_pos(a);      /* 0 when unprovisioned */
        last_us[i] = a->last_serviced_us;
        common[i] = common_rows[i];
    }
    /* Host-carry prefix reads (the same reads routing does on idle banks; no
     * CUDA), only where the core will look: candidate rows, strictly-longer
     * columns. */
    for (int i = 0; i < s->n_slots; i++) {
        if (!eligible[i] || (protect && protect[i]) || hist_len[i] <= 0) continue;
        for (int k = 0; k < s->n_slots; k++)
            common_rows[i][k] = (k != i && frontier[k] > hist_len[i])
                ? pulsar_session_bank_common_prefix(pool, s->slots[k].bank, hist[i]) : -1;
    }
    return superseded_pick_core(s->n_slots, protect, eligible, hist_len, frontier, common, last_us);
}

/* Evict exactly one NON-trunk victim so a warm fork gets a free bank. Trunk is
 * always protected (a sibling still matches it); LRU-superseded victims go
 * first, else plain LRU (worker_evict_one's picker). Reuses the proven eviction
 * body (snapshot + ledger release + bank reset). Worker thread only; returns
 * true when a bank was freed. */
bool server::fork_make_room(const session_slot *trunk, bool superseded_only) {
    auto *s = this;
    if (s->pool_banks <= 0) return false;
    bool protect[PULSAR_SESSION_POOL_CAP];
    s->worker_protect_queued_owner_slots(protect);      /* live-tool owners */
    /* trunk == NULL is the FRESH-SLOT caller: a new conversation that
     * matched nothing worth keeping, so only live owners are protected. */
    const int ti = trunk ? (int)(trunk - s->slots) : -1;
    if (ti >= 0 && ti < PULSAR_SESSION_POOL_CAP) protect[ti] = true;  /* NEVER the trunk */
    const int sup = s->pick_superseded_idle(protect);
    if (sup >= 0) {
        /* Force worker_evict_one onto the superseded pick by protecting all others. */
        bool only[PULSAR_SESSION_POOL_CAP];
        for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) only[i] = (i != sup);
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: warm-fork make-room: evicting LRU-superseded bank %u "
                   "(trunk bank %u preserved)",
                   s->slots[sup].bank, trunk ? trunk->bank : (uint32_t)-1);
        return s->worker_evict_one(only);
    }
    /* No superseded victim.  superseded_only callers stop here: evicting a
     * LIVE conversation's bank under cyclic multi-tenant traffic hits
     * exactly the next returning conversation (the LRU domino, measured
     * 2026-08-10) — those callers have a better option (advance in place). */
    if (superseded_only) return false;
    /* Plain LRU among unprotected idle, trunk still safe. Two-phase: first
     * avoid banks that are a queued job's usable warm match (the fresh-path
     * domino), then — if that leaves no victim — retry without the overlay,
     * because binding must progress. */
    bool with_warm[PULSAR_SESSION_POOL_CAP];
    memcpy(with_warm, protect, sizeof with_warm);
    s->worker_protect_queued_warm_matches(with_warm);
    if (s->worker_evict_one(with_warm)) return true;
    return s->worker_evict_one(protect);
}



/* Bind the head job to a slot if routing allows it. Strict FIFO: when the
 * head must wait (its owner slot is busy, or no fitting slot is free), later
 * jobs wait behind it — simple and starvation-free. Returns true when the
 * head was consumed: bound to a slot, or failed explicitly (a continuation
 * that cannot fit its owner slot — see choose_slot_for_job). When the head
 * cannot be placed cleanly (nothing fits, or only a warm slot it would
 * clobber), it is not waiting on a busy owner, and the provisioning refusal
 * is one eviction can relieve (full pool / full ledger — never the
 * MemAvailable floor, see the increment-4 block above), idle slots are
 * evicted LRU-first until the head binds without clobbering or eviction
 * stops helping — then the clobber fallback binds it exactly like the
 * increment-3 scheduler did. */
bool server::worker_try_bind() {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    job *j = s->head; /* peek: only the worker pops */
    /* Nothing queued means nothing is blocked; without this the gauge would
     * keep reporting the last reason long after the queue drained. */
    if (!j) s->m_queue_block_reason = 0;
    pthread_mutex_unlock(&s->mu);
    if (!j) return false;

    /* Reap a head whose client disconnected while queued BEFORE the bind
     * machinery: choose_slot_for_job can evict warm banks and spend seconds
     * inside pulsar_session_create — none of it for a dead fd. The client
     * thread only flags and stays parked on j->cv until this signal, so the
     * worker remains the sole popper (the invariant the queued-job protect
     * helpers rely on). No HTTP error write: the socket is gone. */
    pthread_mutex_lock(&j->mu);
    const bool head_cancelled = j->cancelled;
    pthread_mutex_unlock(&j->mu);
    if (head_cancelled) {
        pthread_mutex_lock(&s->mu);
        s->head = j->next;
        if (!s->head) s->tail = NULL;
        if (s->n_queued > 0) s->n_queued--;
        pthread_mutex_unlock(&s->mu);
        j->next = NULL;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: dropping queued job for disconnected client fd=%d",
                   j->fd);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
        return true;
    }

    int reject_ctx = 0;
    bool waiting_owner = false;
    bool clobbers = false;
    provision_refusal refusal = PROVISION_OK;
    session_slot *sl = s->choose_slot_for_job(j, &reject_ctx, &waiting_owner,
                                           &clobbers, &refusal);
    if ((!sl || clobbers) && !waiting_owner && reject_ctx == 0 &&
        (refusal == PROVISION_REFUSED_POOL_FULL ||
         refusal == PROVISION_REFUSED_ADMISSION))
    {
        bool protect[PULSAR_SESSION_POOL_CAP];
        s->worker_protect_queued_owner_slots(protect);
        if (s->worker_eviction_could_help(j, protect)) {
            while ((!sl || clobbers) &&
                   (refusal == PROVISION_REFUSED_POOL_FULL ||
                    refusal == PROVISION_REFUSED_ADMISSION))
            {
                /* Refresh owner protection every iteration (2026-07-15
                 * review): each pass through choose_slot_for_job can stall
                 * for seconds inside pulsar_session_create, and a live-tool
                 * continuation enqueued during that stall would be invisible
                 * to a one-shot snapshot — its owner could then be evicted
                 * into an avoidable 409. Holes are re-derived from
                 * provisioned == false (the refresh clears worker_evict_one's
                 * hole marks). */
                s->worker_protect_queued_owner_slots(protect);
                for (int i = 0; i < s->n_slots; i++) {
                    if (!s->slots[i].provisioned) protect[i] = true;
                }
                /* Two-phase: prefer victims that are no queued job's usable
                 * warm match; fall back to owner-only protection when the
                 * overlay leaves nothing evictable (progress over warmth). */
                bool with_warm[PULSAR_SESSION_POOL_CAP];
                memcpy(with_warm, protect, sizeof with_warm);
                s->worker_protect_queued_warm_matches(with_warm);
                if (!s->worker_evict_one(with_warm) &&
                    !s->worker_evict_one(protect)) break;
                sl = s->choose_slot_for_job(j, &reject_ctx, &waiting_owner,
                                         &clobbers, &refusal);
            }
        }
    }
    if (!sl && reject_ctx > 0) {
        /* The job can never run: pop it and send the front door's
         * context_length_exceeded client error (against the owner slot's
         * context), then wake the client thread exactly like
         * worker_finish_slot does for a completed job. */
        pthread_mutex_lock(&s->mu);
        s->head = j->next;
        if (!s->head) s->tail = NULL;
        if (s->n_queued > 0) s->n_queued--;
        pthread_mutex_unlock(&s->mu);
        j->next = NULL;
        http_error_context_length_exceeded(j->fd, &j->req,
                                           j->req.prompt.len, reject_ctx);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
        return true;
    }
    if (!sl && refusal == PROVISION_REFUSED_CREATE_FAIL) {
        /* The bank the pool offered could not be installed (provision_bank:
         * state restore refused).  Eviction does not relieve it and retrying
         * every quantum would hard-spin on an empty pool (worker_main's wait
         * predicate), so answer the client with a 500 and pop the job, the
         * way the context-length reject above does. */
        s->note_provision_refusal(j, refusal);
        pthread_mutex_lock(&s->mu);
        s->head = j->next;
        if (!s->head) s->tail = NULL;
        if (s->n_queued > 0) s->n_queued--;
        pthread_mutex_unlock(&s->mu);
        j->next = NULL;
        const char *emsg = "session bank could not be installed (state restore refused)";
        if (j->req.api == API_ANTHROPIC) http_error_anthropic(j->fd, 500, emsg);
        else http_error(j->fd, 500, emsg);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
        return true;
    }
    if (!sl) {
        /* Still queued. A no-op when the head is merely waiting on a busy
         * owner slot (refusal stays PROVISION_OK there). */
        s->note_provision_refusal(j, refusal);
        return false;
    }

    pthread_mutex_lock(&s->mu);
    s->m_queue_block_reason = 0; /* the head bound — the queue is moving again */
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    if (s->n_queued > 0) s->n_queued--;
    s->n_generating++;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;

    s->generate_job_begin(sl, j);
    return true;
}



/* Publish the /metrics snapshots — per-slot KV position/context and the
 * engine spec-decode counters — into plain server fields under mu. Client
 * threads must never call into the engine (CUDA-state audit,
 * pulsar_server_internal.h), so the worker exports these at startup (cli_main,
 * before the worker thread runs), after binds, and once per quantum;
 * send_metrics reads only the snapshots. Host-int copies, no GPU work. */
void server::publish_metrics_snapshot() {
    auto *s = this;
    pulsar_spec_metrics m;
    pulsar_engine_spec_metrics(s->engine, &m);
    pthread_mutex_lock(&s->mu);
    for (int i = 0; i < s->n_slots; i++) {
        /* Bank-aware: a non-live bank's pos is its saved carry, never the pool's
         * live cursor (server_slot_frontier_pos). Pure host read, no CUDA. */
        s->m_slot_pos[i] = s->slot_frontier_pos(&s->slots[i]);
        s->m_slot_ctx[i] = s->slots[i].ctx_size;
        /* Phase + prefill progress: pure host reads of the slot's gen_state,
         * published here because send_metrics runs on a client thread and must
         * not walk worker-owned structures. Phase is stored +1 so an unbound
         * slot is 0 = idle; prefill_last_current stays -1 until the first
         * progress callback fires, so clamp it here. */
        const session_slot *sl = &s->slots[i];
        const gen_state *g = sl->active_job ? sl->gen : NULL;
        s->m_slot_phase[i]         = g ? (int)g->phase + 1 : 0;
        s->m_slot_prefill_done[i]  = (g && g->prefill_last_current > 0) ? g->prefill_last_current : 0;
        s->m_slot_prefill_total[i] = (g && g->prefill_total > 0) ? g->prefill_total : 0;
        /* L112: adaptive draft depth, live-or-carry (pure host read). */
        s->m_slot_depth[i] = sl->provisioned
                ? pulsar_session_bank_spec_depth(s->sess, sl->bank) : 0;
    }
    s->m_spec = m;
    s->m_gen_tokens = s->w_gen_tokens;
    s->m_prefill_chunk_tokens = s->w_prefill_chunk_tokens;
    s->m_spec_overflow_rounds = s->w_spec_overflow_rounds;
    s->m_spec_thr_cut_rows = s->w_spec_thr_cut_rows;
    s->m_decode_lane = s->w_decode_lane;
    pthread_mutex_unlock(&s->mu);
}



/* Fold one finished request's timings into the /metrics histograms. Called by
 * the worker from generate_job_end, where req_timings has just been computed;
 * this only reads that struct, so it adds no hot-path work. */
void server::observe_request_timings(const req_timings *t, double e2e_s) {
    auto *s = this;
    if (!t->valid) return;
    pthread_mutex_lock(&s->mu);
    /* ttft_s is 0 when no token was ever emitted (an errored or empty
     * request); recording that as a zero-latency success would flatter the
     * histogram, so skip it. */
    if (t->ttft_s > 0.0)
        pulsar_hist_observe(&s->m_h_ttft, pulsar_hist_seconds_bounds, t->ttft_s);
    if (t->decode_n > 0 && t->decode_s > 0.0)
        pulsar_hist_observe(&s->m_h_tpot, pulsar_hist_seconds_bounds,
                            t->decode_s / (double)t->decode_n);
    if (e2e_s > 0.0)
        pulsar_hist_observe(&s->m_h_e2e, pulsar_hist_seconds_bounds, e2e_s);
    pulsar_hist_observe(&s->m_h_prompt_tok, pulsar_hist_tokens_bounds, (double)t->prompt_n);
    pulsar_hist_observe(&s->m_h_gen_tok, pulsar_hist_tokens_bounds, (double)t->decode_n);
    s->m_requests_finished++;
    pthread_mutex_unlock(&s->mu);
}



/* Bump a /metrics counter under mu. Worker-thread callers only; the lock is
 * what keeps the value coherent for send_metrics reading on a client thread.
 * Never called with mu already held — the mutex is not recursive. */
void server::count_metric(uint64_t *counter) {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    (*counter)++;
    pthread_mutex_unlock(&s->mu);
}



/* Record why the head job could not be bound. Counted once per job, not once
 * per bind attempt: the worker retries the head every quantum, so counting
 * each attempt would turn one stuck request into thousands of "refusals". The
 * companion gauge reports the reason the queue is blocked right now. */
void server::note_provision_refusal(job *j, provision_refusal refusal) {
    auto *s = this;
    const bool countable = refusal > PROVISION_OK && refusal < PROVISION_REFUSAL_COUNT;
    pthread_mutex_lock(&s->mu);
    /* Waiting on a busy owner slot is not an admission refusal, so report the
     * queue as unblocked rather than leaving the previous reason standing. */
    s->m_queue_block_reason = countable ? (int)refusal + 1 : 0;
    if (countable && j->refusal_counted != refusal) {
        j->refusal_counted = refusal;
        s->m_refusals[refusal]++;
    }
    pthread_mutex_unlock(&s->mu);
}



/* Detach a finished job from its slot and wake its client thread. */
void server::worker_finish_slot(session_slot *sl) {
    auto *s = this;
    job *j = sl->active_job;
    s->generate_job_end(sl);
    pthread_mutex_lock(&s->mu);
    if (s->n_generating > 0) s->n_generating--;
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_lock(&j->mu);
    j->done = true;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
}



/* The single GPU worker (increment 3): a round-robin scheduler over the slot
 * pool. Each pass binds queued jobs to free slots (FIFO), then advances ONE
 * runnable slot by one quantum — a prefill chunk, or up to
 * PULSAR_SERVER_DECODE_QUANTUM_TOKENS decode tokens — and flushes that slot's
 * deferred client bytes. With a single active job this degenerates to the
 * increment-2 loop (quantum after quantum on one slot, with only a queue
 * peek — one mutex op, no GPU work — between quanta), so single-client
 * output is byte-identical. All pulsar_session_* and CUDA work stays on this
 * thread. On shutdown the scheduler keeps stepping bound jobs (the decode
 * loop observes g_stop_requested) and drains the queue, exactly like the
 * increment-2 worker.
 *
 * Quantum overshoot: binding may lazily provision a slot, and
 * provision_slot's pulsar_session_create is a multi-GiB allocation that can
 * take SECONDS — every bound slot stalls for its duration. Binding may also
 * EVICT idle slots first (increment 4): snapshot-to-disk (full device sync +
 * multi-GiB D2H + disk write) plus session teardown, then the provisioning
 * on top. These are the largest quantum overshoots in the system (larger
 * than the DSpark ≤17-token fused burst) and are deliberate single-thread
 * design: the CUDA-state audit (pulsar_server_internal.h) rules out a second
 * GPU thread, and both happen only at a scheduling boundary. */
/* A slot is eligible for the batched decode lanes when it is in steady-state
 * decode. L116: tool-call requests are admitted — measured 2026-08-26, the old
 * has_tools exclusion forced ALL agent traffic onto the classic per-slot RR,
 * which is aggregate-CONSERVED (c1/c2/c4 = 17.8/18.5/17.6 t/s) while the
 * spec-batched lane scales (19.2/22.7/26.4). The machinery the exclusion
 * comment cited is lane-shared today: the temp-0 tool-payload gating lives in
 * gen_resolve_sampling_decode (every lane's sampling authority), and the
 * think/tool recovery paths are host-side in gen_emit_token + gen_step_finish,
 * both of which the batched lanes already drive. CONTRACT CAVEAT the admission
 * inherits: the M>=2 batched forward is ~1 ULP off the M=1 path, so a
 * forced-greedy tool-payload span's near-tie argmax can depend on what else is
 * co-scheduled — same accepted property the plain batched lane has always had
 * for explicit temp-0 requests (characterized 2026-07-22); output remains
 * valid and self-consistent with the bank's KV.
 * Prefill/init/finish slots are serviced per-slot as usual. */
static bool slot_is_batchable_decode(const session_slot *sl) {
    const gen_state *g = sl->gen;
    return sl->active_job && g && g->phase == GEN_DECODE;
}

/* worker_main's lane select over the gathered decode set (L179 branch 2).
 * 0 = idle, 3 = spec-batched (inc 6): every decoder can speculate, none has
 * joined a plain batch (n_batched == 0 -- no lane switch mid-conversation)
 * and the drafter is loaded; 2 = plain batched otherwise (L118: every n_dec
 * >= 1 is a batch, a solo session is a batch of one). 1 is the retired
 * classic lane: reachable only with n_dec >= 1 and no pool, which the
 * gather loop never produces. Lane 3 keeps the spec_decode counters
 * advancing; lane 2 does not. */
static int server_pick_decode_lane(int pool_banks, bool has_dspark, session_slot *const *dec,
                                   int n_dec, int n_batched) {
    bool all_spec = has_dspark && n_dec >= 1 && n_batched == 0;
    for (int i = 0; all_spec && i < n_dec; i++) {
        const gen_state *dg = dec[i]->gen;
        if (!dg || !dg->dspark_spec_enabled || dg->batch_active)
            all_spec = false;
    }
    const bool use_spec_batched = pool_banks > 0 && all_spec;
    const bool use_batched = use_spec_batched || (pool_banks > 0 && n_dec >= 1);
    return n_dec <= 0 ? 0 : (use_spec_batched ? 3 : (use_batched ? 2 : 1));
}

/* Tier-2 §5 batched decode quantum: ONE shared multiseq weight sweep drives up
 * to PULSAR_SERVER_DECODE_QUANTUM_TOKENS steps across every supplied decode slot.
 * Each slot samples its OWN logits row with its OWN sampler/RNG and streams
 * through gen_emit_token (the shared emit path), so per-request
 * output/stop/streaming is byte-identical in shape to the classic lane; only the
 * forward numerics differ (batched vs single-token path). NOTE: this is NOT
 * co-scheduling-neutral for greedy (temp-0) output. The M>=2 co-batched forward's
 * logits differ ~1 ULP from the M=1 single-sequence path (different GEMM tiling /
 * reduction order); at greedy temp-0 that can flip a near-tie argmax, so a
 * request's greedy continuation depends on what else is co-scheduled. This is
 * distribution-preserving and INVISIBLE under sampling (temp>0) — same class as
 * batched inference elsewhere (vLLM et al.) — but it is a real greedy-determinism
 * property, characterized 2026-07-22. The multiseq gate proves N=2-vs-N=3
 * neutrality (bank output invariant to OTHER banks), which is a WEAKER invariant
 * than batched==solo and does NOT cover this ULP. L116: tool-call requests
 * ride this lane too — their forced-greedy payload spans inherit exactly this
 * property (see slot_is_batchable_decode), and their temp-0 gating comes from
 * gen_resolve_sampling_decode like every other lane.
 * Slots that stop are set to GEN_FINISH and dropped; the worker finishes
 * them via the per-slot path, which reconciles their host checkpoint against the
 * tokens committed here. Leaves the pool multiseq-poisoned with live_bank = -1
 * so the next server_bank_switch does a real bank restore. */
/* ==== Tier-2 task #55 increment 2b: proactive-eviction guard =================
 *
 * At each decode-quantum boundary, if the live banks growing by one quantum would
 * push total resident KV past the budget, SPILL the LRU-idle smallest-frontier
 * bank: its comp/index KV -> local fast disk (raw, bit-identical), then cudaFree
 * its physical (the only primitive that returns physical on GB10; 2a/2b gates).
 * The conversation stays bound to the slot and is restored (alloc + reload) on its
 * next decode via server_bank_switch. Decisions use the DETERMINISTIC touched
 * accounting + MemAvailable floor backstop, never the coarse cudaMemGetInfo gauge.
 * Worker thread only. */

bool server::bank_restore_spilled(int bank) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    char path[600];
    snprintf(path, sizeof path, "%s/spill-bank-%d.kv", s->spill_dir, bank);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: restore open %s failed: %s",
                   path, strerror(errno));
        s->count_metric(&s->m_restore_failures);
        return false;
    }
    char err[128];
    const double t0 = server_now_sec();
    const int rc = pulsar_session_bank_kv_load(pool, (uint32_t)bank, fp, err, sizeof err);
    fclose(fp);
    if (rc != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: kv_load bank %d failed: %s", bank, err);
        s->count_metric(&s->m_restore_failures);
        return false;
    }
    s->slots[bank].spilled = false;
    remove(path);
    s->count_metric(&s->m_restores);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: guard RESTORED bank %d from disk (%.1f ms reload stall)",
               bank, (server_now_sec() - t0) * 1e3);
    return true;
}

/* Spill one idle bank: install it, snapshot its KV to disk, save its host carry,
 * repoint AWAY (free_physical refuses the cur bank), then cudaFree its physical. */
bool server::spill_bank(session_slot *victim) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    const uint32_t vb = victim->bank;
    /* The victim is never spilled (the pick excludes spilled banks), so this
     * never reloads from disk, but the installing state restore can refuse.
     * Then live_bank still names another bank and the kv_save below would
     * snapshot THAT bank's rings under the victim's file name (L190 A4).
     * Abort the spill; the caller stops spilling this quantum. */
    if (!s->bank_switch((int)vb)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: guard: bank %u install failed (state restore refused); "
                   "spill aborted", vb);
        return false;
    }
    char path[600];
    snprintf(path, sizeof path, "%s/spill-bank-%u.kv", s->spill_dir, vb);
    /* Durability: free_physical below drops the bank's only other copy, so this
     * file must be COMPLETE on disk before we get there.  Plain fopen+fclose
     * does not guarantee that — a crash or power loss can leave a truncated or
     * zero-length spill sitting under the final name, and the restore path then
     * refuses it, so that conversation 500s permanently.  Use the same
     * write-tmp / fsync / rename contract the disk-KV store already uses
     * (pulsar_kvstore.cpp): this path never inherited it. */
    char tmp[672];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: spill open %s failed: %s",
                   tmp, strerror(errno));
        return false;
    }
    char err[128];
    const double t0 = server_now_sec();
    const int rc = pulsar_session_bank_kv_save(pool, vb, fp, err, sizeof err);
    /* fsync BEFORE the rename, or the rename can become visible while the
     * contents are still only in page cache. */
    const bool synced = rc == 0 && fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    const int fc = fclose(fp);
    if (rc != 0 || !synced || fc != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: kv_save bank %u failed: %s",
                   vb, rc ? err : (synced ? "close" : "fsync"));
        remove(tmp);
        return false;
    }
    if (rename(tmp, path) != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: spill rename %s failed: %s",
                   path, strerror(errno));
        remove(tmp);
        return false;
    }
    /* Persist the rename itself so the entry survives a crash.  Best-effort:
     * some filesystems reject a directory fsync, and the contents are already
     * durable above. */
    {
        int dfd = ::open(s->spill_dir, O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) {
            (void)fsync(dfd);
            ::close(dfd);
        }
    }
    pulsar_session_bank_state_save(pool, vb);         /* preserve host carry for restore */
    if (!pulsar_session_bank_state_restore(pool, 0)) { remove(path); return false; }
    s->live_bank = 0;
    /* Finding 4: free_physical returns false ONLY on a precondition refusal (nothing
     * freed, bank still live) — abort the spill and drop the unused disk snapshot.
     * A true return means the bank IS evicted (slabs freed), so mark it spilled
     * unconditionally — no half-evicted state (spilled=false over freed slabs). */
    if (!pulsar_session_bank_free_physical(pool, vb)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: guard: free_physical bank %u refused (still cur?) — spill aborted", vb);
        remove(path);
        return false;
    }
    victim->spilled = true;
    victim->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    s->guard_evictions++;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: guard EVICTED bank %u -> disk (%.1f ms save, physical freed); "
               "touched %.2f GiB / budget %.2f GiB, evictions %llu",
               vb, (server_now_sec() - t0) * 1e3,
               (double)pulsar_session_touched_kv_bytes(pool) / gib,
               (double)s->guard_touched_budget / gib,
               (unsigned long long)s->guard_evictions);
    return true;
}

/* LRU-idle smallest-frontier victim: NOT bank 0 (pinned), NOT in the live decode
 * set, no active job, not already spilled. -1 if none. */
int server::guard_pick_victim(session_slot **dec, int n) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    int best = -1;
    uint64_t best_us = UINT64_MAX, best_touched = UINT64_MAX;
    for (int i = 1; i < s->n_slots; i++) {         /* bank 0 pinned */
        session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->spilled || sl->active_job) continue;
        /* plan-33: never free a bank that is a live fork SOURCE mid-clone. */
        if (pulsar_session_bank_fork_pinned(pool, sl->bank)) continue;
        bool live = false;
        for (int k = 0; k < n; k++) if (dec[k] == sl) { live = true; break; }
        if (live) continue;
        const uint64_t t = pulsar_session_bank_touched_kv_bytes(pool, sl->bank);
        if (sl->last_serviced_us < best_us ||
            (sl->last_serviced_us == best_us && t < best_touched)) {
            best = i; best_us = sl->last_serviced_us; best_touched = t;
        }
    }
    return best;
}

/* The guard's control law (L179 branch 7): how many spills a projected breach
 * needs. projected = touched + delta; no breach (projected <= bound) is 0.
 * Otherwise victims are spilled in the given (LRU) order, each dropping its
 * per_victim_drop from touched, until the projection fits -- the MINIMUM
 * number (usually ONE), never the whole idle set (finding 2: the cascade bug).
 * With the victims exhausted and the breach still standing it returns
 * n_victims -- the count it CAN do; the caller spills those and then
 * back-pressures (proceeds and lets the MemAvailable floor guard), so with no
 * victims at all it returns 0 and spills nothing. */
static int guard_spill_plan(uint64_t touched, uint64_t delta, uint64_t bound,
                            const uint64_t *per_victim_drop, int n_victims) {
    int k = 0;
    while (touched + delta > bound && k < n_victims) {
        const uint64_t drop = per_victim_drop[k++];
        touched = drop >= touched ? 0 : touched - drop;
    }
    return k;
}

/* Every guard victim in pick order: guard_pick_victim is the ONE authority
 * for the filter and the LRU key; the slots already ordered are appended to
 * its live-exclusion set so each call yields the next. At most n_slots
 * entries. */
static int guard_victim_order(server *s, session_slot **dec, int n, int *order) {
    session_slot *excl[2 * PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < n; i++) excl[i] = dec[i];
    int m = 0;
    while (n + m < 2 * PULSAR_SESSION_POOL_CAP) {
        const int vi = s->guard_pick_victim(excl, n + m);
        if (vi < 0) break;
        order[m] = vi;
        excl[n + m] = &s->slots[vi];
        m++;
    }
    return m;
}

void server::guard_maybe_evict(session_slot **dec, int n) {
    auto *s = this;
    if (!s->guard_enabled || s->pool_banks <= 0 || n <= 0) return;
    pulsar_session *pool = s->sess;
    const uint64_t dpb = pulsar_session_quantum_growth_bytes_per_bank(
            pool, (uint32_t)PULSAR_SERVER_DECODE_QUANTUM_TOKENS);
    const uint64_t delta = (uint64_t)n * dpb;      /* all n live banks grow */
    const uint64_t bound = s->guard_touched_budget;
    /* No breach, no plan: the common quantum costs one gauge read, as before
     * the plan-then-execute split (review of abc847f). */
    if (pulsar_session_touched_kv_bytes(pool) + delta <= bound) return;
    /* Finding 2: free_physical zeroes a spilled bank's frontier, so a spill
     * drops touched by exactly that bank's touched bytes (the pool gauge IS
     * the per-bank sum, gpu_graph_touched_kv_bytes). guard_spill_plan walks
     * that arithmetic over the LRU victim order and evicts exactly the minimum
     * (usually ONE) per breach, NOT the whole idle set (the cascade bug). The
     * per-quantum count log lets the smoke assert no cascade. */
    int order[PULSAR_SESSION_POOL_CAP];
    uint64_t drop[PULSAR_SESSION_POOL_CAP];
    const int n_victims = guard_victim_order(s, dec, n, order);
    for (int k = 0; k < n_victims; k++)
        drop[k] = pulsar_session_bank_touched_kv_bytes(pool, s->slots[order[k]].bank);
    const int plan = guard_spill_plan(pulsar_session_touched_kv_bytes(pool), delta, bound,
                                      drop, n_victims);
    int spilled_this_quantum = 0;
    for (int k = 0; k < plan; k++) {
        if (!s->spill_bank(&s->slots[order[k]])) break;   /* spill failed — stop */
        spilled_this_quantum++;
        s->count_metric(&s->m_spills);
    }
    if (spilled_this_quantum == n_victims) {
        /* Every idle victim is spilled (or there was none): if the breach still
         * stands, back-pressure — proceed and let the live floor guard.
         * (Evicting a LIVE growing bank would thrash; the MemAvailable watchdog
         * is the hard backstop.) */
        const uint64_t projected = pulsar_session_touched_kv_bytes(pool) + delta;
        static uint64_t last_warn_us;
        const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
        if (projected > bound && now_us - last_warn_us > 5000000ull) {
            last_warn_us = now_us;
            const double gib = 1024.0 * 1024.0 * 1024.0;
            server_log(PULSAR_LOG_WARNING,
                "pulsar-server: guard: projected touched %.2f GiB > budget %.2f GiB but NO "
                "idle victim — back-pressure (MemAvailable floor is the backstop)",
                (double)projected / gib, (double)bound / gib);
        }
    }
    if (spilled_this_quantum > 0) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: guard: spilled %d bank(s) this quantum (touched now %.2f GiB / %.2f GiB)",
                   spilled_this_quantum,
                   (double)pulsar_session_touched_kv_bytes(pool) / (1024.0*1024.0*1024.0),
                   (double)bound / (1024.0*1024.0*1024.0));
    }
}

/* L148: park the live bank before a batched quantum poisons the pool.
 *
 * A batched step invalidates the pool session's live checkpoint (decode_mixed
 * sets checkpoint_valid = false) and the quantum then sets live_bank = -1. If
 * the bank that was live is NOT in this batch -- a slot mid-prefill, or one
 * that just finished and whose folded history sits only in the pool copy --
 * nothing ever copies its checkpoint to its carry: the next bank_state_restore
 * hands back the stale carry (the prompt, or nothing), pos() reads 0 or the
 * prompt length while the bank's KV frontier is hundreds of rows ahead, and
 * the next step is rejected ("first position 0 <= 0", "frontier not
 * position-true"). Seen on the drafter-off lane with three repeat-prompt
 * clients (rows/L148.md). Saving it here, while its checkpoint is still valid,
 * is what bank_switch does on every hand-off; this is the hand-off the
 * batched quanta skipped. */
static bool park_live_bank_needed(int live_bank, int pool_banks, session_slot *const *dec, int n,
                                  const session_slot *extra) {
    if (live_bank < 0 || live_bank >= pool_banks) return false;
    for (int i = 0; i < n; i++)
        if (dec[i] && (int)dec[i]->bank == live_bank) return false;
    if (extra && (int)extra->bank == live_bank) return false;
    return true;
}

static void park_live_bank(server *s, session_slot **dec, int n, const session_slot *extra) {
    if (!park_live_bank_needed(s->live_bank, s->pool_banks, dec, n, extra)) return;
    s->slots[s->live_bank].committed_pos = pulsar_session_pos(s->sess);
    pulsar_session_bank_state_save(s->sess, (uint32_t)s->live_bank);
}

/* L118 per-quantum client-liveness poll, shared by the three batched lanes
 * (L179 branch 13). A slot is abandoned iff it is mid-decode (GEN_DECODE),
 * its client is gone and -- on the lanes that ask for it -- its batch_feed is
 * valid. "Gone" is either signal: the socket hung up (gen_client_disconnected:
 * POLLRDHUP on fd), or the slot's writer has already FAILED (EPIPE, stall
 * timeout, pending overflow, shutdown -- genmsg.cpp slot_writer_send/flush/
 * drain).  A failed writer used to be read only by the next byte-producing
 * emit, so a stream holding bytes back (tool-tag or stop-sequence hold)
 * decoded on for a client that could no longer hear it (L190 C3).  The plain
 * and mixed lanes require batch_feed_valid (their abandon path also clears
 * it, so the pending commit is dropped exactly once); the spec lane does not
 * (its per-slot epilogue finishes a dead client through the normal path).
 * The asymmetry is the caller's, stated at each call. */
static bool lane_should_abandon(const gen_state *g, bool require_batch_feed, int fd) {
    if (!g || g->phase != GEN_DECODE) return false;
    if (require_batch_feed && !g->batch_feed_valid) return false;
    if (g->writer.failed) return true;
    return gen_client_disconnected(fd);
}

/* The abandon itself, shared by the three lanes: name the signal, stop the
 * slot (drop_feed on the lanes whose pending commit must be dropped once). */
static void lane_abandon(gen_state *g, bool drop_feed) {
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: %s, abandoning generation after %d tokens",
               g->writer.failed ? "client stream failed" : "client disconnected",
               g->completion);
    if (drop_feed) g->batch_feed_valid = false;
    g->phase = GEN_FINISH;
}

void server::worker_batched_decode_quantum(session_slot **dec, int n) {
    auto *s = this;
    if (n <= 0) return;
    pulsar_session *pool = s->sess;
    const int vocab = pulsar_engine_logits_width(s->engine);
    park_live_bank(s, dec, n, NULL);

    /* Tier-2 2b: proactive-eviction guard — BEFORE the weight sweep grows the live
     * banks, spill LRU-idle banks if this quantum's projected growth would breach
     * the resident-KV budget. A dec bank that was spilled while idle is restored
     * transparently by server_bank_switch in the ENTRY loop below. */
    s->guard_maybe_evict(dec, n);

    /* L118: per-quantum client-liveness poll (classic-loop parity). NOTE:
     * mid-flight continued disk-KV stores are NOT ported to this plain lane —
     * batch_feed/batch_pending tokens reconcile into the session history only
     * at per-slot finish, so there is no point in the quantum where
     * s->checkpoint is token-true for a bank. Finish-time stores are
     * unchanged; the spec lane (the default path) carries the mid-flight
     * cadence. */
    for (int i = 0; i < n; i++) {
        gen_state *pg = dec[i]->gen;
        if (pg && lane_should_abandon(pg, /*require_batch_feed=*/true, pg->j->fd))
            lane_abandon(pg, /*drop_feed=*/true);
    }

    /* ENTRY: a slot not yet in the batch samples its first feed token from its
     * bank's current classic logits while that bank is briefly live, then
     * CAPTURES its per-bank frontier counters (ms_n_comp[bank]) that the
     * multiseq driver's position-true check reads. The explicit save is
     * essential: server_bank_switch only captures the OUTGOING bank, so a bank
     * entering the batch while it is already the live bank (e.g. it just
     * prefilled and was never switched away) would otherwise carry a STALE
     * ms_n_comp and the driver would reject the step ("frontier not
     * position-true"). Redundant when the bank was captured on a prior
     * switch-away — but harmless (same value). */
    for (int i = 0; i < n; i++) {
        session_slot *sl = dec[i];
        gen_state *g = sl->gen;
        if (g->batch_feed_valid) continue;
        /* Finding 1: a failed spill restore must NOT sample this slot's feed token
         * on another bank's KV — fail the slot cleanly and drop it from the batch. */
        if (!s->bank_switch(sl->bank)) {
            snprintf(g->err, sizeof g->err,
                     "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
            g->finish = "error";
            g->batch_feed_valid = false;
            g->phase = GEN_FINISH;
            continue;
        }
        float temp, top_p, min_p; int top_k;
        gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
        g->batch_feed_token =
            pulsar_session_sample(pool, temp, top_k, top_p, min_p, &g->rng);
        if (g->batch_feed_token < 0) {
            snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        /* The bank is live and its classic logits are what this token was drawn
         * from; after the multiseq step below they describe nothing (the step
         * leaves s->logits untouched by contract), so capture here or never. */
        logprob_capture_session(&g->logprobs, pool, g->batch_feed_token);
        g->batch_feed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
        g->batch_feed_valid = true;
        g->batch_active = true;
    }

    float *logits = (float *)server_xmalloc((size_t)n * (size_t)vocab * sizeof(float));
    pulsar_multiseq_req reqs[PULSAR_SESSION_POOL_CAP];
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < PULSAR_SERVER_DECODE_QUANTUM_TOKENS; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;            /* already stopped */
            if (g_stop_requested || g->completion >= g->max_tokens) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            if (g->batch_feed_pos >= pulsar_session_ctx(pool)) {
                g->batch_feed_valid = false; g->finish = "length";
                g->phase = GEN_FINISH; continue;
            }
            reqs[m].bank = sl->bank;
            reqs[m].pos = g->batch_feed_pos;
            reqs[m].token = g->batch_feed_token;
            live_idx[m] = i;
            m++;
        }
        if (m == 0) break;

        char err[96];
        /* plan-34 inc 1: route the decode-only lane through the mixed entry
         * (n_rows == n_dec, still exactly 1 row per bank — no prefill rows, no
         * mixing yet). Byte-identical to pulsar_session_decode_multiseq; the only
         * change is the heap descriptor scratch. */
        const int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m,
                                                 logits, (uint32_t)m * vocab,
                                                 NULL, 0u, err, sizeof(err));
        s->live_bank = -1;   /* pool is multiseq-poisoned: no clean live bank */
        if (rc != 0) {
            for (int q = 0; q < m; q++) {
                gen_state *g = dec[live_idx[q]]->gen;
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "batched decode failed: %s", err);
                g->batch_feed_valid = false;
                g->phase = GEN_FINISH;
            }
            break;
        }

        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            const int committed = g->batch_feed_token; /* committed at batch_feed_pos */
            g->batch_feed_pos++;
            sl->committed_pos = g->batch_feed_pos;
            sl->tokens_emitted++;   /* L118: classic-loop accounting parity */
            pulsar_tokens_push(&g->batch_pending, committed);
            if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
            /* Route THIS slot's stream writes through its own deferral queue
             * (the worker-thread-local writer is shared across slots). */
            slot_writer_install(&g->writer);
            if (s->gen_emit_token(sl, committed)) {
                g->batch_feed_valid = false;
                g->phase = GEN_FINISH;
                continue;
            }
            const float *row = logits + (size_t)q * (size_t)vocab;
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
            g->batch_feed_token =
                pulsar_sample_logits(row, vocab, temp, top_k, top_p, min_p, &g->rng);
            if (g->batch_feed_token < 0) {
                snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
                g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
                continue;
            }
            /* This bank's row IS the target distribution at batch_feed_pos, and
             * it is overwritten by the next step's sweep. */
            logprob_capture_row(&g->logprobs, row, vocab, g->batch_feed_token);
        }
    }
    free(logits);
    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
}


/* L049 increment 1 / L117: confidence-ranked cross-bank K allocation for the
 * spec lane's shared forward (L179 branch 1). surv[i][j] is bank i's survival
 * (cumprod of drafter confidences) at pending position j, npend[i] its pending
 * count (0 for a bank not in GEN_DECODE), n_live the number of decoding banks
 * (one base row each). Returns 1 on OVERFLOW (base rows + every pending >
 * PULSAR_SPEC_ROW_BUDGET), else 0.
 *
 * ISOLATION INVARIANT (the lane gate caught the first version of this
 * violating it): the carry-derived pending counts can be stale or inherited
 * from a bank's PREVIOUS request, so a cap derived from them must never bind
 * when the budget does not -- otherwise a bank's round shape couples to its
 * partner's bank history. When everything fits, every bank is admitted whole
 * (k_alloc[i] == npend[i]) and the caller applies the old unconditional cap,
 * so the sweep is bit-identical to the pre-allocator lane; the ranked
 * allocation engages ONLY on overflow, where the old behavior (arbitrary
 * whole-bank sit-out) was itself partner-coupled and strictly worse.
 *
 * Under overflow: admit the global best (bank, position) until the row budget
 * (PULSAR_SPEC_ROW_BUDGET - n_live) is spent; survival is monotone within a
 * bank so each bank's admitted set is a prefix. The cost table (L117/L136)
 * stops admission once the best remaining candidate's survival is below thr
 * (= marginal row ms / live ms-per-token EMA): every remaining candidate is
 * then <= it, and *thr_cut_rows counts the rows that cut left unadmitted (0
 * when the cut did not fire). */
static int spec_alloc_rows(const float surv[][16], const uint32_t *npend, int n, int n_live,
                           float thr, int *k_alloc, int *thr_cut_rows) {
    uint32_t demand = (uint32_t)n_live;
    for (int i = 0; i < n; i++) demand += npend[i];
    *thr_cut_rows = 0;
    const bool overflow = demand > (int)PULSAR_SPEC_ROW_BUDGET;
    if (!overflow) {
        for (int i = 0; i < n; i++) k_alloc[i] = (int)npend[i];
        return 0;
    }
    for (int i = 0; i < n; i++) k_alloc[i] = 0;
    int budget = (int)PULSAR_SPEC_ROW_BUDGET - n_live;
    while (budget > 0) {
        int bi = -1;
        float bv = -1.0f;
        for (int i = 0; i < n; i++) {
            if ((uint32_t)k_alloc[i] < npend[i] && surv[i][k_alloc[i]] > bv) {
                bv = surv[i][k_alloc[i]];
                bi = i;
            }
        }
        if (bi < 0) break;
        if (bv < thr) {
            /* L136: everything left is priced out by the cost table while
             * budget remains -- the only rows whose fate marginal_ms decides. */
            for (int i = 0; i < n; i++) *thr_cut_rows += (int)npend[i] - k_alloc[i];
            break;
        }
        k_alloc[bi]++;
        budget--;
    }
    return 1;
}

/* plan-34 inc 6: the SPEC batched quantum. Same skeleton as
 * worker_batched_decode_quantum, but each sweep runs one speculative ROUND
 * per bank instead of one token: per bank under its restored state we draw
 * the base token (carry or fresh), begin the round (guards, frontier
 * snapshot, checkpoint push), and contribute its rows; ONE decode_mixed
 * ALL_ROWS forward covers every bank's rows with the drafter capture +
 * Stage-B saves armed; then per bank round_end walks its slice, rolls state,
 * redrafts, and we emit the accepted tokens through the same slot machinery
 * the plain lane uses. Tokens per weight-stream compound: batching x
 * acceptance ([[L076]]).
 *
 * Emission mirrors gen_decode's L073 discipline: a mid-emit stop (tool-call
 * end, stop string) rewinds the ghost tail so the bank's history never
 * carries tokens the client did not see (pulsar_session_rewind also clears
 * the pendings/carry, which is exactly right -- they were conditioned on the
 * ghosts). */
void server::worker_spec_batched_quantum(session_slot **dec, int n) {
    auto *s = this;
    if (n <= 0) return;
    pulsar_session *pool = s->sess;
    const int vocab = pulsar_engine_logits_width(s->engine);
    const int eos_token = pulsar_token_eos(s->engine);
    park_live_bank(s, dec, n, NULL);

    s->guard_maybe_evict(dec, n);

    /* L118: the per-quantum client-liveness poll the classic loop always had.
     * A dead client's slot finishes through the normal per-slot epilogue (the
     * final write fails harmlessly on the dead fd). */
    for (int i = 0; i < n; i++) {
        gen_state *pg = dec[i]->gen;
        if (pg && lane_should_abandon(pg, /*require_batch_feed=*/false, pg->j->fd))
            lane_abandon(pg, /*drop_feed=*/false);
    }

    pulsar_spec_round *rounds[PULSAR_SESSION_POOL_CAP] = {0};
    int first_tok[PULSAR_SESSION_POOL_CAP];
    uint32_t row0s[PULSAR_SESSION_POOL_CAP];
    int row_nb[PULSAR_SESSION_POOL_CAP];   /* rows this bank contributed (base+K) */
    int live_idx[PULSAR_SESSION_POOL_CAP];
    pulsar_multiseq_req reqs[PULSAR_SPEC_LOGITS_ROWS];
    int accepted[PULSAR_SPEC_LOGITS_ROWS + 1];
    /* ALL_ROWS caps the shared forward at 16 rows (the spec-logits ceiling).
     * L123: the landing buffer lives on the server (allocated once) — the
     * per-quantum malloc re-faulted 16.5 MB of demand-zero pages every
     * quantum. */
    if (!s->spec_lane_logits)
        s->spec_lane_logits = (float *)server_xmalloc(
                (size_t)PULSAR_SPEC_LOGITS_ROWS * (size_t)vocab * sizeof(float));
    float *logits = s->spec_lane_logits;

    int emitted_total = 0;
    const double quantum_t0 = server_now_sec();   /* L117 EMA numerator */
    while (emitted_total < PULSAR_SERVER_DECODE_QUANTUM_TOKENS) {
        /* ---- L049 increment 1: confidence-ranked cross-bank K allocation.
         * At <=16 total rows the shared forward's marginal row cost is
         * near-flat, so the win is ALLOCATION under the cap, not budget
         * sizing (vLLM #47808's cost-table argmax is increment 2 if that
         * ever stops being true). Score every (bank, position) slot by the
         * drafter's survival probability -- cumprod of per-position
         * confidences, read switch-free from the saved carries, which are
         * fresh at sweep start because every bank saved at the previous
         * round's end -- and admit the global best until the row budget is
         * spent. Survival is monotone, so each bank's admitted set is a
         * prefix. Banks with no/invalid pendings get K=0 and still run
         * their base row. When everything fits, every slot is admitted and
         * the lane behaves exactly as before this allocator existed. */
        int k_alloc[PULSAR_SESSION_POOL_CAP];
        bool k_overflow = false;
        {
            float surv[PULSAR_SESSION_POOL_CAP][16];
            uint32_t npend[PULSAR_SESSION_POOL_CAP];
            int n_live = 0;
            for (int i = 0; i < n; i++) {
                npend[i] = 0;
                gen_state *ag = dec[i]->gen;
                if (!ag || ag->phase != GEN_DECODE) continue;
                n_live++;
                float conf[16];
                const uint32_t np =
                    pulsar_session_bank_pending_confs(pool, (uint32_t)dec[i]->bank, conf);
                float p = 1.0f;
                for (uint32_t j2 = 0; j2 < np; j2++) {
                    p *= conf[j2];
                    surv[i][j2] = p;
                }
                npend[i] = np;
            }
            /* L117 (L049 inc 2): under overflow the ranked admission also
             * consults the COST TABLE — stop admitting once the next
             * candidate's survival is worth less than a marginal row
             * costs (value denominator = live EMA of ms per emitted
             * token).  Binds ONLY under overflow — when everything fits
             * the old unconditional cap is byte-identical (isolation
             * invariant, see spec_alloc_rows). vLLM #47808 is the same
             * design upstream.
             * L111/L121 established the cost is DEPTH-FLAT (the old
             * 8.4→11 ramp was the naive score kernel's rows x depth
             * term, not a property of the engine).  L136 refresh
             * (ROWCOST 2026-08-31, dev 87eec09): 6.4 ms/row @2048,
             * 5.9 @24576 — the 8.0 measured on 08-27 predated L129's
             * MoE fusion, which moved the whole sweep ~9%.  A stage
             * decomposition (L134) puts ~83% of this in routed-MoE
             * expert compute, so expect the number to move with MoE
             * kernel work, not with KV/indexer work. */
            const float marginal_ms = 6.0f;
            const float ema = s->spec_ms_per_tok_ema > 1.0f ?
                              s->spec_ms_per_tok_ema : 45.0f;
            int thr_cut_rows = 0;
            k_overflow = spec_alloc_rows(surv, npend, n, n_live, marginal_ms / ema,
                                         k_alloc, &thr_cut_rows) != 0;
            if (k_overflow) s->w_spec_overflow_rounds++;
            s->w_spec_thr_cut_rows += (uint64_t)thr_cut_rows;
        }
        /* ---- assemble: per bank, base draw + round begin + rows ---------- */
        uint32_t rows = 0;
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g || g->phase != GEN_DECODE) continue;
            if (g_stop_requested || g->completion >= g->max_tokens) {
                if (g->completion >= g->max_tokens && !g->finish) g->finish = "length";
                g->phase = GEN_FINISH;
                continue;
            }
            if (!s->bank_switch(sl->bank)) {
                snprintf(g->err, sizeof g->err,
                         "bank %u state restore failed (evicted KV unrecoverable)",
                         (unsigned)sl->bank);
                g->finish = "error";
                g->phase = GEN_FINISH;
                continue;
            }
            if (pulsar_session_pos(pool) >= pulsar_session_ctx(pool)) {
                g->finish = "length";
                g->phase = GEN_FINISH;
                continue;
            }
            const uint32_t k_cap_rows = k_overflow
                ? 1u + (uint32_t)k_alloc[i]
                : pulsar_session_spec_next_rows_max(pool);
            if (rows + k_cap_rows > PULSAR_SPEC_ROW_BUDGET) {
                /* Over the shared-forward row budget even at the allocated
                 * K (can only happen when an earlier bank EOS'd/errored and
                 * the sweep shape shifted): sit this sweep out BEFORE the
                 * base draw or round_begin touch anything -- the carry
                 * (possibly a rejection residual, whose exact emission the
                 * acceptance proof needs) and the pendings stay intact. */
                pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
                continue;
            }
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
            const int first = pulsar_session_spec_next_base(pool, temp, top_k,
                                                         top_p, min_p, &g->rng);
            if (first < 0) {
                snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
                g->finish = "error";
                g->phase = GEN_FINISH;
                continue;
            }
            if (first == eos_token) {
                /* generate_speculative's short-circuit: emit EOS, never eval it. */
                slot_writer_install(&g->writer);
                if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
                (void)s->gen_emit_token(sl, first);
                pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
                g->phase = GEN_FINISH;
                emitted_total++;
                continue;
            }
            if (!rounds[i]) rounds[i] = pulsar_spec_round_new();
            char err[160];
            /* accepted_cap = allocated K + the base token: round_begin's own
             * trim (K <= accepted_cap-1) enforces the allocation, and its
             * validity checks may trim further -- the allocation is an
             * upper bound, never a promise. */
            if (pulsar_session_spec_round_begin(pool, rounds[i], first,
                                             g->max_tokens - g->completion,
                                             k_overflow ? k_alloc[i] + 1
                                                        : (int)(sizeof(accepted) / sizeof(accepted[0])),
                                             temp, top_k, top_p, min_p,
                                             err, sizeof err) != 0) {
                snprintf(g->err, sizeof g->err, "spec round begin failed: %s", err);
                g->finish = "error";
                g->phase = GEN_FINISH;
                continue;
            }
            if (rows + pulsar_spec_round_n_rows(rounds[i]) > PULSAR_SPEC_ROW_BUDGET) {
                /* Unreachable: the pre-begin budget check bounds n_batch from
                 * above (begin only trims). Defensive backstop, checked
                 * BEFORE fill_reqs writes, so a future change to begin's row
                 * math cannot overflow reqs[]. */
                pulsar_session_spec_round_abort(pool, rounds[i]);
                pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
                continue;
            }
            const uint32_t nb = pulsar_spec_round_fill_reqs(rounds[i], sl->bank,
                                                         first, reqs + rows);
            row0s[m] = rows;
            row_nb[m] = (int)nb;
            first_tok[m] = first;
            live_idx[m] = i;
            rows += nb;
            m++;
            pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
        }
        if (m == 0) break;

        /* ---- ONE shared forward over every bank's rows ------------------- */
        char err[160];
        pulsar_session_spec_arm_capture(pool, rows);
        uint32_t got = 0;
        const int rc = pulsar_session_decode_mixed(pool, reqs, rows, logits,
                                                 (int)(rows * (uint32_t)vocab),
                                                 &got, PULSAR_MSEQ_HEAD_ALL_ROWS,
                                                 err, sizeof err);
        pulsar_session_spec_arm_capture(pool, 0u);
        s->live_bank = -1;   /* pool is multiseq-poisoned until a bank_switch */
        if (rc != 0 || got != rows) {
            for (int q = 0; q < m; q++) {
                session_slot *sl = dec[live_idx[q]];
                gen_state *g = sl->gen;
                if (s->bank_switch(sl->bank)) {
                    pulsar_session_spec_round_abort(pool, rounds[live_idx[q]]);
                    pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
                }
                g->finish = "error";
                snprintf(g->err, sizeof g->err, "spec batched forward failed: %s", err);
                g->phase = GEN_FINISH;
            }
            break;
        }

        /* ---- per bank: finish the round, emit, persist ------------------- */
        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            if (!s->bank_switch(sl->bank)) {
                g->finish = "error";
                snprintf(g->err, sizeof g->err,
                         "bank %u restore failed after spec forward", (unsigned)sl->bank);
                g->phase = GEN_FINISH;
                continue;
            }
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
            /* The round's base position -- NOT pulsar_session_pos() here: during
             * the round the checkpoint already spans the verify batch. */
            const int pos_before = pulsar_spec_round_saved_len(rounds[live_idx[q]]);
            const int na = pulsar_session_spec_round_end(pool, rounds[live_idx[q]],
                                                      first_tok[q], eos_token,
                                                      temp, top_k, top_p, min_p,
                                                      &g->rng, logits, row0s[q],
                                                      accepted,
                                                      (int)(sizeof(accepted) / sizeof(accepted[0])),
                                                      err, sizeof err);
            if (na < 0) {
                g->finish = "error";
                snprintf(g->err, sizeof g->err, "spec round end failed: %s", err);
                g->phase = GEN_FINISH;
                continue;
            }
            /* L155 tripwire: the round must hand back exactly the positions it
             * committed (the engine clamps commit at EOS and trims to the
             * accepted count).  The ghost rewind below counts na - emitted and
             * cannot see a frontier that ran ahead of na, so this is the only
             * place that would notice; it says so if the invariant ever breaks. */
            if (pulsar_session_pos(pool) != pos_before + na)
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: spec batched round bank %u: frontier %d != "
                           "%d + %d accepted -- the bank carries tokens the client "
                           "will not see (L155)",
                           (unsigned)sl->bank, pulsar_session_pos(pool), pos_before, na);
            slot_writer_install(&g->writer);
            int done = 0;
            bool stopped = false;
            for (int t = 0; t < na; t++) {
                if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
                done = t + 1;
                if (s->gen_emit_token(sl, accepted[t])) { stopped = true; break; }
            }
            if (done < na) {
                /* L073, batched-lane edition: committed-but-never-emitted
                 * tokens rewind, whatever ended the emission. */
                const int ghost = na - done;
                const int target = pulsar_session_pos(pool) - ghost;
                pulsar_session_rewind(pool, target);
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: spec batched round bank %u: rewound %d "
                           "ghost tokens to pos %d",
                           (unsigned)sl->bank, ghost, target);
            }
            if (stopped) g->phase = GEN_FINISH;
            sl->committed_pos = pulsar_session_pos(pool);
            sl->tokens_emitted += (uint64_t)done;
            /* L119: request-scoped DSpark accounting — the round's truth,
             * accumulated here where it is unambiguous (row_nb = base+K rows
             * this bank contributed; na = base + accepted drafts). */
            g->req_spec_rounds++;
            if (row_nb[q] > 1) g->req_spec_draft += (uint64_t)(row_nb[q] - 1);
            if (na > 1) g->req_spec_accepted += (uint64_t)(na - 1);
            g->req_spec_gen += (uint64_t)done;
            /* L118: continued disk-KV store — the classic loop's cadence,
             * ported to the one point where it is valid in this lane: the
             * bank is live (round-end bank_switch) and round_end + the ghost
             * rewind left s->checkpoint token-true for it. Same tool-span
             * suppression as the classic loop. */
            if (!stopped && g->phase == GEN_DECODE) {
                const request *rq = &g->j->req;
                const dsml_decode_state ds =
                    rq->kind == REQ_CHAT && rq->has_tools ?
                        g->dsml_tracker.decode : DSML_DECODE_OUTSIDE;
                if (!(rq->kind == REQ_CHAT && rq->has_tools &&
                      (g->saw_tool_start || dsml_decode_state_is_tool(ds))))
                    s->kv_cache_maybe_store_continued(sl);
            }
            pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
            emitted_total += done;
        }
        /* L150: round_end deferred every bank's redraft; run ONE drafter pass
         * over all of them (the engine reads their saved ring counters and the
         * bank-major rings, and never switches banks), then stamp each bank's
         * draft into its shadow under this scheduler's own bank switch. A
         * device failure here is not fatal: the banks keep no pendings and take
         * a plain n=1 step next round. */
        {
            pulsar_spec_round *live_rounds[PULSAR_SPEC_LOGITS_ROWS];
            uint32_t live_banks[PULSAR_SPEC_LOGITS_ROWS];
            uint64_t *live_rngs[PULSAR_SPEC_LOGITS_ROWS];
            session_slot *live_slots[PULSAR_SPEC_LOGITS_ROWS];
            int nl = 0;
            for (int q = 0; q < m && nl < (int)PULSAR_SPEC_LOGITS_ROWS; q++) {
                session_slot *sl = dec[live_idx[q]];
                if (!rounds[live_idx[q]] || !sl->gen || sl->gen->phase != GEN_DECODE) continue;
                live_rounds[nl] = rounds[live_idx[q]];
                live_banks[nl] = (uint32_t)sl->bank;
                live_rngs[nl] = &sl->gen->rng;
                live_slots[nl] = sl;
                nl++;
            }
            char rerr[160];
            if (nl > 0 &&
                pulsar_session_spec_redraft_batch(pool, live_rounds, live_banks, live_rngs, nl,
                                                  rerr, sizeof rerr) != 0)
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: batched redraft failed: %s (banks take a plain step)",
                           rerr);
            for (int q = 0; q < nl; q++) {
                if (!s->bank_switch(live_banks[q])) {
                    /* bank_switch's contract: a failed state restore fails the
                     * request.  Skipping the commit left the bank's ring holding
                     * an uncommitted draft and the slot decoding on (L190 D1). */
                    gen_state *g = live_slots[q]->gen;
                    snprintf(g->err, sizeof g->err,
                             "bank %u state restore failed before redraft commit",
                             (unsigned)live_banks[q]);
                    g->finish = "error";
                    g->phase = GEN_FINISH;
                    continue;
                }
                pulsar_session_spec_redraft_commit(pool, live_rounds[q]);
                pulsar_session_bank_state_save(pool, live_banks[q]);
            }
        }
        /* /metrics granularity: publish per ROUND, not per quantum.  A
         * quantum (16 tokens) takes ~1 s at deep-context rates, and a
         * scraper polling faster than the publish cadence sees its deltas
         * beat into 0 / 2x-rate flapping (the pulsar-tui square wave).
         * One mutex+copy per round (~0.35 s) is host noise. */
        s->publish_metrics_snapshot();
    }
    if (emitted_total > 0) {
        const float ms_per_tok = (float)((server_now_sec() - quantum_t0) * 1e3 /
                                         (double)emitted_total);
        s->spec_ms_per_tok_ema = s->spec_ms_per_tok_ema <= 0.0f ?
                ms_per_tok : 0.9f * s->spec_ms_per_tok_ema + 0.1f * ms_per_tok;
    }
    for (int i = 0; i < n; i++) {
        if (rounds[i]) pulsar_spec_round_free(rounds[i]);
        dec[i]->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
}

/* plan-34 phase-2 inc 5 — find ONE prefilling slot to FOLD into the fused mixed
 * quantum (P=1). Admissible = main-prefill (not cold), already past its FIRST chunk
 * (bank pos>0, so the driver's pos-0 reject is satisfied — the first chunk stays
 * classic), and with more than one fold-chunk of prompt still left (the FINAL tail
 * <= chunk stays classic; it carries the prefill->decode completion bookkeeping).
 * Its bank is necessarily DISTINCT from every decode bank (different phase). NULL
 * when the flag is off, not in pool mode, or nothing qualifies. */
/* Deep-concurrent guard (see the enable block in cli_main.cpp): fusing a
 * prefill chunk into a decode quantum whose banks already read a deep
 * aggregate KV working set displaces bandwidth-saturated decode. Blocks while
 * two or more active decoders' summed committed depth exceeds guard_rows;
 * guard_rows <= 0 is the guard off. n_dec/deep are reported for the caller's
 * log line (L179 branch 12). */
static bool mixed_deep_guard_blocks(const session_slot *slots, int n_slots, int guard_rows,
                                    int *n_dec_out, long *deep_out) {
    if (guard_rows <= 0) return false;
    long deep = 0;
    int n_dec = 0;
    for (int i = 0; i < n_slots; i++) {
        const session_slot *dl = &slots[i];
        if (dl->provisioned && dl->active_job && dl->state == SLOT_DECODING) {
            deep += dl->committed_pos;
            n_dec++;
        }
    }
    if (n_dec_out) *n_dec_out = n_dec;
    if (deep_out) *deep_out = deep;
    return n_dec >= 2 && deep > (long)guard_rows;
}

session_slot *server::worker_find_fuse_prefill() {
    auto *s = this;
    if (!s->mixed_batch_enabled || s->pool_banks <= 0) return NULL;
    /* Those prefills take the classic (unfused) path instead. */
    int n_dec = 0;
    long deep = 0;
    if (mixed_deep_guard_blocks(s->slots, s->n_slots, s->mixed_deep_guard_rows, &n_dec, &deep)) {
        static long last_logged = -1;
        if (deep != last_logged) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: fused lane paused by deep guard (%d decoders, %ld aggregate rows > %d)",
                       n_dec, deep, s->mixed_deep_guard_rows);
            last_logged = deep;
        }
        return NULL;
    }
    pulsar_session *pool = s->sess;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        gen_state *g = sl->gen;
        if (!sl->active_job || !g || g->phase != GEN_PREFILL_MAIN || !g->prompt_for_sync ||
            g->no_fuse)
            continue;
        const int P = pulsar_session_bank_pos(pool, sl->bank);
        const int len = g->prompt_for_sync->len;
        if (P <= 0 || P >= len) continue;                       /* first chunk / done: classic */
        return sl;                                              /* P=1: first qualifier */
    }
    return NULL;
}

/* plan-34 phase-2 inc 5 — FUSED mixed-batch quantum. One decode quantum whose EVERY
 * step folds a small (s->mixed_chunk_tokens) prefill run for `pf` into the SAME
 * pulsar_session_decode_mixed sweep as the decode banks (true continuous batching,
 * P=1). Decode banks advance exactly as worker_batched_decode_quantum — the inc-4
 * neutrality gate proves a co-scheduled prefill does not perturb them — so their
 * per-request output is unchanged in shape. The prefill advances up to
 * QUANTUM*chunk tokens, SPREAD uniformly across the steps so no single decode step
 * eats a whole chunk (the p99 lever vs the time-slice's per-interval decode stall).
 * pf's FIRST chunk (pos 0) and FINAL tail (<= chunk) stay CLASSIC: the tail's
 * pulsar_session_sync carries the prefill->decode completion (kv-cache store,
 * gen_stream_begin), so this never reimplements that handoff. Reconciliation of
 * pf's bank is the exact recipe the decode lane uses (bank_state_restore +
 * note_committed_tokens). */

/* LEVER 1 (L179 branch 10): how many of a fused step's runs get a head. On an
 * INTERMEDIATE prefill sub-chunk (kthis > 0 rows that do not reach len this
 * step) with decode banks to head (m > 0), only the m decode runs' logits are
 * emitted -- the prefill run's intermediate logits are unused and the head
 * takes the single-block identity path (no two-block resync, no wasted
 * prefill head). On the FINAL sub-chunk (pos_now + kthis == len), a pure-decode
 * step (kthis == 0) or a prefill-only step (m == 0) the cap is 0 = every run
 * (the prefill head IS consumed). */
static uint32_t mixed_head_cap(int kthis, int m, int pos_now, int len) {
    return (kthis > 0 && m > 0 && pos_now + kthis < len) ? (uint32_t)m : 0u;
}

/* The verdict on a fused step's rc (L179 branch 10). rc == 1 is the engine's
 * RECOVERABLE reject (nothing committed); when the step folded prefill rows
 * (kthis > 0) the reject is charged to the prefill run -- e.g. its bank's
 * frontier is not position-true after a cache-warm resume -- and the prefill
 * stops folding (no_fuse: classic from now on). The co-scheduled decode banks
 * are not harmed: with m > 0 the step is RETRIED decode-only, with m == 0 the
 * quantum STOPS (only the prefill was in it). rc == 0, a hard failure, and a
 * recoverable reject on a pure-decode step (kthis == 0) all PROCEED to the
 * caller's normal rc handling. */
enum mixed_step_verdict { MIXED_PROCEED, MIXED_GIVEUP_RETRY_DECODE, MIXED_GIVEUP_STOP };

static int mixed_prefill_giveup(int rc, int kthis, int m) {
    if (rc != 1 || kthis <= 0) return MIXED_PROCEED;
    return m > 0 ? MIXED_GIVEUP_RETRY_DECODE : MIXED_GIVEUP_STOP;
}

void server::worker_mixed_batch_quantum(session_slot **dec, int n, session_slot *pf) {
    auto *s = this;
    if (n <= 0 || !pf || !pf->gen || !pf->gen->prompt_for_sync) return;
    pulsar_session *pool = s->sess;
    const int vocab = pulsar_engine_logits_width(s->engine);
    park_live_bank(s, dec, n, pf);
    gen_state *pg = pf->gen;
    const pulsar_tokens *pp = pg->prompt_for_sync;

    s->guard_maybe_evict(dec, n);

    /* L118: per-quantum client-liveness poll (classic-loop parity; same
     * continued-store caveat as the plain lane). */
    for (int i = 0; i < n; i++) {
        gen_state *lg = dec[i]->gen;
        if (lg && lane_should_abandon(lg, /*require_batch_feed=*/true, lg->j->fd))
            lane_abandon(lg, /*drop_feed=*/true);
    }

    /* Decode-bank ENTRY — identical to worker_batched_decode_quantum. */
    for (int i = 0; i < n; i++) {
        session_slot *sl = dec[i];
        gen_state *g = sl->gen;
        if (g->batch_feed_valid) continue;
        if (!s->bank_switch(sl->bank)) {
            snprintf(g->err, sizeof g->err,
                     "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        float temp, top_p, min_p; int top_k;
        gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
        g->batch_feed_token = pulsar_session_sample(pool, temp, top_k, top_p, min_p, &g->rng);
        if (g->batch_feed_token < 0) {
            snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        logprob_capture_session(&g->logprobs, pool, g->batch_feed_token);
        g->batch_feed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
        g->batch_feed_valid = true; g->batch_active = true;
    }

    /* Prefill-bank ENTRY: make pf live at its committed frontier P0 and capture its
     * per-bank counters so the mixed driver reads pf's TRUE starting frontier. */
    if (!s->bank_switch(pf->bank)) {
        snprintf(pg->err, sizeof pg->err, "prefill bank %u restore failed", (unsigned)pf->bank);
        pg->finish = "error"; pg->phase = GEN_FINISH; return;
    }
    const int P0 = pulsar_session_pos(pool);
    pulsar_session_bank_state_save(pool, (uint32_t)pf->bank);
    {   /* one-shot observability: confirm the fused lane actually engaged. */
        static int logged = 0;
        if (logged < 3) { logged++;
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: FUSED mixed quantum engaged (prefill bank %u at pos %d/%d, "
                       "%d decode banks, chunk %d/step)",
                       (unsigned)pf->bank, P0, pp->len, n, s->mixed_chunk_tokens);
        }
    }

    int kstep = s->mixed_chunk_tokens;
    const int cap = pulsar_session_prefill_cap(pool);   /* mixed-step row ceiling */
    if (kstep > cap - PULSAR_SESSION_POOL_CAP) kstep = cap - PULSAR_SESSION_POOL_CAP;
    if (kstep < 1) kstep = 1;
    const int len = pp->len;
    int pf_done = 0;
    bool reached_end = false;                     /* prefill reached len this quantum */
    bool pf_giveup = false;                        /* prefill rejected -> stop folding it */

    const size_t reqcap = (size_t)PULSAR_SESSION_POOL_CAP + (size_t)kstep;
    pulsar_multiseq_req *reqs = (pulsar_multiseq_req *)server_xmalloc(reqcap * sizeof(*reqs));
    float *logits = (float *)server_xmalloc((size_t)(PULSAR_SESSION_POOL_CAP + 1) * (size_t)vocab * sizeof(float));
    /* last-position (len-1) logits captured from the final fused prefill run — the
     * decode seed for the prefill->decode handoff (byte-identical to classic per
     * the inc-4 gate: the fused run's last-of-run logits match classic-resume). */
    float *pf_last_logits = (float *)server_xmalloc((size_t)vocab * sizeof(float));
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < PULSAR_SERVER_DECODE_QUANTUM_TOKENS; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;
            if (g_stop_requested || g->completion >= g->max_tokens) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            if (g->batch_feed_pos >= pulsar_session_ctx(pool)) {
                g->batch_feed_valid = false; g->finish = "length"; g->phase = GEN_FINISH; continue;
            }
            reqs[m].bank = sl->bank; reqs[m].pos = g->batch_feed_pos;
            reqs[m].token = g->batch_feed_token; live_idx[m] = i; m++;
        }
        /* Fold this step's ascending prefill sub-chunk from P0+pf_done, all the way
         * to len (the FINAL sub-chunk carries the last-position logits used for the
         * prefill->decode handoff — no non-aligned classic tail resume, so the
         * prefill output stays byte-identical to a cold prefill). */
        const int pos_now = P0 + pf_done;
        int kthis = 0;
        if (!pf_giveup && pos_now < len) {
            kthis = kstep;
            if (pos_now + kthis > len) kthis = len - pos_now;
        }
        for (int j = 0; j < kthis; j++) {
            reqs[m + j].bank = pf->bank;
            reqs[m + j].pos = pos_now + j;
            reqs[m + j].token = pp->v[pos_now + j];
        }
        const int nrows = m + kthis;
        if (nrows == 0) break;

        char err[96];
        uint32_t n_runs = 0;
        /* LEVER 1: the head cap is mixed_head_cap's (see it). */
        const uint32_t head_cap = mixed_head_cap(kthis, m, pos_now, len);
        int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)nrows, logits,
                (int)((size_t)(m + (kthis > 0 ? 1 : 0)) * (size_t)vocab), &n_runs, head_cap, err, sizeof err);
        const int verdict = mixed_prefill_giveup(rc, kthis, m);
        if (verdict != MIXED_PROCEED) {
            /* RECOVERABLE reject charged to the PREFILL run (see
             * mixed_prefill_giveup): stop folding this prefill (classic via
             * no_fuse); the co-scheduled decode banks retry this step alone. */
            pf_giveup = true; pg->no_fuse = true;
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: fused prefill rejected (%s): this prefill runs classic "
                       "from now on; the %d decode bank(s) retry this step alone", err, m);
            if (verdict == MIXED_GIVEUP_RETRY_DECODE) {
                rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m, logits,
                        (int)((size_t)m * (size_t)vocab), &n_runs, 0u, err, sizeof err);
            } else { s->live_bank = -1; break; }   /* only prefill this step: just stop */
            kthis = 0;                              /* prefill did not advance */
        }
        s->live_bank = -1;
        if (rc != 0) {
            for (int q = 0; q < m; q++) {
                gen_state *g = dec[live_idx[q]]->gen;
                g->finish = "error";
                snprintf(g->err, sizeof g->err, "fused mixed decode failed: %s", err);
                g->batch_feed_valid = false; g->phase = GEN_FINISH;
            }
            break;   /* real decode failure -> stop */
        }
        pf_done += kthis;
        /* L114 counter: fused sub-chunks bypass server_progress_cb — tick the
         * chunk-granular counter through the per-slot watermark instead (the
         * classic callback uses the same watermark, so the two sites compose
         * without double-counting; pos_now + kthis is the same position
         * coordinate the callback's chunk_end reports). */
        if (kthis > 0 && pos_now + kthis > pf->prefill_counted) {
            s->w_prefill_chunk_tokens +=
                (uint64_t)(pos_now + kthis - pf->prefill_counted);
            pf->prefill_counted = pos_now + kthis;
        }
        if (kthis > 0 && pos_now + kthis == len) {
            /* final prefill sub-chunk: capture the len-1 logits (prefill run row =
             * index m, last-of-run) as the decode seed for the handoff. */
            memcpy(pf_last_logits, logits + (size_t)m * (size_t)vocab, (size_t)vocab * sizeof(float));
            reached_end = true;
        }

        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            const int committed = g->batch_feed_token;
            g->batch_feed_pos++; sl->committed_pos = g->batch_feed_pos;
            sl->tokens_emitted++;   /* L118: classic-loop accounting parity */
            pulsar_tokens_push(&g->batch_pending, committed);
            if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
            slot_writer_install(&g->writer);
            if (s->gen_emit_token(sl, committed)) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            const float *row = logits + (size_t)q * (size_t)vocab;
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
            g->batch_feed_token = pulsar_sample_logits(row, vocab, temp, top_k, top_p, min_p, &g->rng);
            if (g->batch_feed_token < 0) {
                snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
                g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
                continue;
            }
            logprob_capture_row(&g->logprobs, row, vocab, g->batch_feed_token);
        }
    }
    free(reqs); free(logits);

    /* Reconcile the prefill bank (same recipe as a decode bank leaving the lane):
     * install its driver-maintained frontier (P0+pf_done) and advance the host
     * checkpoint by exactly the committed prefill tokens, so its next chunk resumes
     * correctly and any store sees the true frontier. */
    if (pf_done > 0 && pg->phase == GEN_PREFILL_MAIN) {
        if (pulsar_session_bank_state_restore(pool, pf->bank)) {
            pulsar_session_note_committed_tokens(pool, &pp->v[P0], pf_done);
            pf->committed_pos = P0 + pf_done;
            s->live_bank = (int)pf->bank;
            if (reached_end) {
                /* Prefill complete: seed the decode with the fused last-position
                 * logits (byte-identical to classic) and run the SAME prefill->
                 * decode handoff the classic path uses (kv-cache store, SSE start,
                 * GEN_DECODE_INIT). No non-aligned classic tail resume => the
                 * request's decode is byte-identical to a cold classic prefill. */
                pulsar_session_set_logits(pool, pf_last_logits, vocab);
                slot_writer_install(&pg->writer);
                s->gen_stream_begin(pf);
                slot_writer_flush(&pg->writer);
            }
        } else {
            snprintf(pg->err, sizeof pg->err, "prefill bank %u reconcile failed", (unsigned)pf->bank);
            pg->finish = "error"; pg->phase = GEN_FINISH;
        }
    }
    free(pf_last_logits);

    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
    pf->last_serviced_us = now_us;
}

void *worker_main(void *arg) {
    server *s = (server *)arg;
    int rr = 0; /* round-robin cursor: first slot index to consider next */
    for (;;) {
        bool bound = false;
        while (s->worker_try_bind()) bound = true;
        if (bound) s->publish_metrics_snapshot();

        int n_active = 0;
        for (int i = 0; i < s->n_slots; i++) {
            if (s->slots[i].active_job) n_active++;
            /* L006 heartbeat: swept over ALL bound slots, not just the one
             * about to get a quantum. The slot that most needs a keepalive is
             * precisely the one starved behind another job's long prefill —
             * which is never the slot being advanced. Non-blocking; a no-op
             * unless that slot has been silent >= PULSAR_SERVER_HEARTBEAT_MS. */
            if (s->slots[i].gen) (void)gen_stream_heartbeat(s->slots[i].gen);
        }
        if (n_active == 0) {
            /* With every slot free, choose_slot_for_job never returns NULL
             * (provision_bank floor-gates only the SECOND bank onward, so an
             * empty pool always provisions its first bank), so an unbound
             * head cannot reach this wait: sleeping on the condvar until new
             * work or shutdown is safe.  If a refusal path is ever added
             * that can leave head set with nothing active, this predicate
             * becomes a hard spin — keep the first-bank guarantee. */
            bool quit, had_head;
            {
                pulsar::ScopedLock lk(&s->mu);
                had_head = s->head != NULL;
                while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
                quit = !s->head && s->stopping;   /* read shared state under the lock */
            }
            if (quit) break;
            /* A head that was already queued when nothing is active is a head
             * worker_try_bind just refused (provisioning refusal, waiting
             * owner). Re-attempting it in a tight loop burns a core on
             * /proc/meminfo reads — poll instead. Fresh work arriving through
             * the condvar wait has had_head == false and binds immediately. */
            if (had_head) usleep(10000);
            continue;
        }

        /* Gather steady-state batchable decode slots. n_batched > 0 means a
         * plain batch is already in flight — those slots stay on the plain
         * lane until they finish (no mid-conversation lane switch, which
         * would need stale-logits reconciliation). L118: every decode slot is
         * batchable and every n >= 1 arms a batched quantum; the Tier-2
         * three-way lane that lived here is deleted. */
        session_slot *dec[PULSAR_SESSION_POOL_CAP];
        int n_dec = 0, n_batched = 0;
        if (s->pool_banks > 0) {
            for (int i = 0; i < s->n_slots; i++) {
                if (slot_is_batchable_decode(&s->slots[i])) {
                    dec[n_dec++] = &s->slots[i];
                    if (s->slots[i].gen->batch_active) n_batched++;
                }
            }
        }
        /* inc 6: the SPEC batched lane -- rounds, not tokens -- when every
         * batchable decode slot can speculate and no plain batch is mid-
         * flight (no lane switch mid-conversation; same reconciliation rule
         * as classic->batched). Slots with spec off (logprobs requests) keep
         * the whole group on the plain lane this quantum rather than
         * splitting the sweep. A bank whose yield-quench has latched stays in
         * this lane but degrades naturally to 1-row rounds: round_end's
         * redraft respects the latch, so its pendings stay empty -- correct
         * output, base-token rounds, only the per-row capture as overhead. */
        /* L118 (everything is a batch): the batched quanta run at EVERY
         * n_dec >= 1 — a solo session is a batch of one. The classic per-slot
         * decode lane, its A/B hatch, and the spec_max_live crossover are
         * DELETED (P4; parity evidence in rows/L118.md). Spec-batched when
         * every decoder can speculate, plain-batched otherwise. */
        /* Record the lane for /metrics. Only the spec lane runs the fused verify
         * loop, so this is what tells a scraper whether the spec_decode_*
         * counters describe the present or some earlier single-request stretch. */
        s->w_decode_lane = server_pick_decode_lane(s->pool_banks,
                                                   pulsar_engine_has_dspark(s->engine),
                                                   dec, n_dec, n_batched);
        const bool use_spec_batched = s->w_decode_lane == 3;
        const bool use_batched = s->w_decode_lane >= 2;

        if (use_batched) {
            /* plan-34 inc 5: when the fused lane is armed and a prefilling slot is
             * admissible (P=1), FOLD its next chunk into the decode sweep instead of
             * the separate classic prefill advance below. Flag OFF (or nothing
             * admissible) => pf_fuse==NULL => today's exact decode-quantum +
             * separate-prefill time-slice, byte-identical. */
            session_slot *pf_fuse = NULL;
            if (use_spec_batched) {
                s->worker_spec_batched_quantum(dec, n_dec);
            } else {
                pf_fuse = s->worker_find_fuse_prefill();
                if (pf_fuse) s->worker_mixed_batch_quantum(dec, n_dec, pf_fuse);
                else         s->worker_batched_decode_quantum(dec, n_dec);
            }
            /* Finish any slot the batched quantum stopped (per-slot path
             * reconciles its checkpoint). Then also advance ONE non-decode
             * active slot (prefill/init/finish) so prompt ingest never starves
             * behind a long batched decode. */
            for (int i = 0; i < n_dec; i++) {
                session_slot *d = dec[i];
                if (d->gen && d->gen->phase != GEN_DECODE) {
                    s->generate_job_step(d);
                    if (d->gen) slot_writer_flush(&d->gen->writer);
                    d->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
                    if (!d->gen || d->gen->phase == GEN_DONE)
                        s->worker_finish_slot(d);
                }
            }
            session_slot *other = NULL;
            for (int k = 0; k < s->n_slots; k++) {
                session_slot *c = &s->slots[(rr + k) % s->n_slots];
                /* inc 5: skip the slot already advanced in-band by the fused
                 * quantum (pf_fuse) so it does not also run a classic chunk. */
                if (c->active_job && c != pf_fuse && !slot_is_batchable_decode(c) &&
                    !(c->gen && c->gen->batch_active)) {
                    other = c;
                    rr = (int)(c - s->slots) + 1;
                    break;
                }
            }
            if (other && other->gen && other->gen->phase != GEN_DONE) {
                s->generate_job_step(other);
                if (other->gen) slot_writer_flush(&other->gen->writer);
                other->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
                if (!other->gen || other->gen->phase == GEN_DONE)
                    s->worker_finish_slot(other);
            }
            s->publish_metrics_snapshot();
            continue;
        }

        session_slot *sl = NULL;
        for (int k = 0; k < s->n_slots; k++) {
            session_slot *c = &s->slots[(rr + k) % s->n_slots];
            if (c->active_job) {
                sl = c;
                rr = (int)(c - s->slots) + 1;
                break;
            }
        }
        if (!sl) continue; /* unreachable: n_active > 0 */

        /* L118: this fall-through only runs when NO decode slot exists
         * (any GEN_DECODE slot arms use_batched above), so it services
         * prefill/init/finish slots only. The L112 paired-decode mirror
         * that lived here is deleted with the classic lane. */
        if (sl->gen && sl->gen->phase != GEN_DONE) {
            s->generate_job_step(sl);
            if (sl->gen) slot_writer_flush(&sl->gen->writer);
            sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
        }
        if (!sl->gen || sl->gen->phase == GEN_DONE) {
            s->worker_finish_slot(sl);
        }
        s->publish_metrics_snapshot(); /* /metrics: once per quantum */
    }
    return NULL;
}


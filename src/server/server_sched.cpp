/* Scheduler/worker (split move-only from generate.cpp): worker_main and its
 * quantum loop (classic / batched / fused mixed-batch lanes), job->slot
 * routing (choose_slot_for_job) and enqueue, lazy provisioning + the
 * admission/ledger/mem-floor helpers, LRU eviction and the warm
 * make-room path, the proactive spill/restore guard, Tier-2 bank
 * switching, and the frontier/committed-pos readers. */
#include "pulsar_server_internal.h"
#include "pulsar_nvtx.h"
#include "pulsar_lock.hpp"

#include <algorithm>



/* Tier-2 shared-pool bank switch (defined lower, near provision_slot). Installs
 * `bank`'s device views + host carry on the pool session, saving the outgoing
 * bank first. No-op in classic mode or when `bank` is already live. Called at
 * the top of every per-slot worker op (gen_begin, generate_job_step,
 * worker_evict_one) so all s->sess touches inside that op are live-correct. */
/* Returns false only when the target is a guard-spilled bank whose restore FAILED
 * (KV unrecoverable): the bank is NOT installed and cur_bank is unchanged, so the
 * caller MUST fail the request rather than sample/commit on the wrong bank's KV
 * (review finding 1). True on success (or classic mode / no-op). */
/* Evict ONE idle victim (a dominated bank preferred, else plain LRU) so a fresh
 * conversation has a bank; live-tool owners protected. Defined near
 * worker_evict_one; forward-declared for the routing path above it. */



bool server::enqueue(job *j) {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    j->seq = ++s->job_seq;
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    s->n_queued++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}

/* L329 (a): a waiter is passed over, so the job the worker binds or drops need not be the head (mu held) */
static void queue_unlink_locked(server *s, job *j) {
    job *prev = NULL, *q = s->head;
    for (; q && q != j; q = q->next) prev = q;
    if (q != j) return;   /* not queued */
    if (prev) prev->next = j->next;
    else s->head = j->next;
    if (s->tail == j) s->tail = prev;
    if (s->n_queued > 0) s->n_queued--;
    j->next = NULL;
}

void server::queue_unlink(job *j) {
    pthread_mutex_lock(&mu);
    queue_unlink_locked(this, j);
    pthread_mutex_unlock(&mu);
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

/* The guard's live budget (pulsar_server_internal.h, server::guard_budget).  Before 2026-10-29 every decision read
 * the startup ceiling, i.e. it priced the post-warmup headroom as KV's alone for the life of the process: whatever
 * else grew took the floor (the work hang).  One /proc read per 0.25 s at most -- a decision reads it, never a
 * token. */
KvBytes server::guard_budget() {
    const double now = server_now_sec();
    if (guard_avail_at == 0.0 || now - guard_avail_at >= 0.25) {
        guard_avail = KvBytes{server_mem_available_bytes()};
        guard_avail_at = now;
    }
    const KvBytes live =
        server_guard_touched_budget(guard_kv_budget, guard_eager_bytes, pulsar_session_touched_kv_bytes(sess),
                                    guard_avail);
    const KvBytes b = live < guard_touched_ceiling ? live : guard_touched_ceiling;
    /* rule 5: the budget the guard is keeping, said when it moves by half a GiB or more */
    const KvBytes step{512ull << 20};
    if ((b > guard_budget_said ? b - guard_budget_said : guard_budget_said - b) >= step) {
        const double gib = 1024.0 * 1024.0 * 1024.0;
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: guard: live touched-KV budget %.2f GiB (ceiling %.2f; touched %.2f + MemAvailable "
                   "%.2f − floor %.2f)%s",
                   (double)b.raw() / gib, (double)guard_touched_ceiling.raw() / gib,
                   (double)pulsar_session_touched_kv_bytes(sess).raw() / gib, (double)guard_avail.raw() / gib,
                   (double)server_mem_floor().raw() / gib,
                   guard_avail != KvBytes{} ? "" : " -- MemAvailable unreadable: fail closed");
        guard_budget_said = b;
    }
    return b;
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
 * server_slot_frontier_pos are always correct for non-live banks.  L330: a
 * guard-spilled bank installs EMPTY (bank_unspill; its history to `was`). */

bool server::bank_switch(int bank, spilled_history *was) {
    auto *s = this;
    if (s->pool_banks <= 0) return true;          /* classic mode */
    if (bank < 0 || bank >= s->pool_banks) return true;
    if (s->slots[bank].preempted) return false;   /* L314: its physical is freed until its readmission reloads it */
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
    /* Tier-2 2b: a guard-spilled target has no physical.  L330: it is re-backed EMPTY -- its history stays on disk,
     * where the spill put it, and the job bound to it restores it a slice per worker turn (gen_begin); before, this
     * read the whole chain here, inside whichever worker turn installed the bank.  A bank that cannot be re-backed is
     * NOT installed and cur_bank is unchanged; return false so the caller fails the request (finding 1). */
    if (s->slots[bank].spilled) return s->bank_unspill(bank, was);
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

int server::slot_common_prefix(const session_slot *sl,
                               const pulsar_tokens *prompt) const {
    const auto *s = this;
    pulsar_prefix_match m;
    s->slot_prefix_match(sl, prompt, &m);
    return m.live_cut.raw();
}

/* L115: the one prefix-reuse question, asked of a slot.  Callers that need
 * the request-side count (accounting, prefill bounds) read `prompt_cut`;
 * callers that need KV rows or a cut point read `live_cut`.  Across a seam
 * these differ, and keeping them in one struct is what stops a live-side
 * count from being used to index the request array. */
void server::slot_prefix_match(const session_slot *sl, const pulsar_tokens *prompt,
                               pulsar_prefix_match *out) const {
    const auto *s = this;
    out->live_cut = LiveToks{0};
    out->prompt_cut = PromptToks{0};
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
 * 0) or a box below floor + marginal refuses -- unless the bank's pages are
 * already RESIDENT (a recycled bank: its engine high-water is non-zero).
 * Reinstalling such a bank pages in nothing, so a tight box has no reason to
 * refuse it; its growth past the high-water is the guard's to police, like any
 * live bank's.  Refusing it was the 2026-09-29 pair's serialized c2: the pool
 * was full, LRU eviction freed a bank, and the floor then refused to hand that
 * same bank to the queued job. */
static bool server_bank_floor_refuses(int n_provisioned, bool pages_resident,
                                      uint64_t avail, uint64_t marginal) {
    return n_provisioned > 0 && !pages_resident &&
           (avail == 0 || !server_mem_floor_admits(avail, marginal));
}

/* The free bank to provision: the one with the most resident pages (engine
 * high-water bytes), lowest index on a tie, so a tight box reuses what it has
 * already paged in before touching a bank that never was.  resident[i] is
 * read only where provisioned[i] is false.  -1 when every bank is provisioned. */
static int server_pick_free_bank(const bool *provisioned, const KvBytes *resident, int n) {
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (provisioned[i]) continue;
        if (best < 0 || resident[i] > resident[best]) best = i;
    }
    return best;
}

/* A refusal one idle-bank eviction can relieve.  A full pool or a full ledger
 * always; the MemAvailable floor only in pool mode, where an evicted bank keeps
 * its pages and provision_bank reuses that resident hole without paging in
 * anything (server_pick_free_bank prefers it, server_bank_floor_refuses admits
 * it).  In classic mode an eviction frees nothing the floor can see. */
static bool server_refusal_evictable(provision_refusal r, bool pool_mode) {
    return r == PROVISION_REFUSED_POOL_FULL || r == PROVISION_REFUSED_ADMISSION ||
           (pool_mode && r == PROVISION_REFUSED_MEM_FLOOR);
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
    /* Pool mode: bank 0 is an ordinary bank (no boot pre-provisioning, no
     * pinning) — the uniform loop from 0 replaced the boot-phantom /
     * empty-reuse / soft-evict special-case family (2026-08-10). */
    int n_provisioned = 0;
    bool provisioned[PULSAR_SESSION_POOL_CAP];
    KvBytes resident[PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < s->pool_banks; i++) {
        provisioned[i] = s->slots[i].provisioned;
        resident[i] = provisioned[i] ? KvBytes{}
                    : pulsar_session_bank_touched_kv_bytes(s->sess, s->slots[i].bank);
        if (provisioned[i]) n_provisioned++;
    }
    const int idx = server_pick_free_bank(provisioned, resident, s->pool_banks);
    if (idx < 0) { *refusal = PROVISION_REFUSED_POOL_FULL; return NULL; }
    /* Belt-and-suspenders: refuse if the box is physically tight (fail closed on
     * an unreadable gauge), matching provision_slot, for a bank that would page
     * in fresh memory; a recycled bank's pages are already resident (see
     * server_bank_floor_refuses). The marginal is what the bank may still
     * demand-page as it fills.
     * The FIRST bank is exempt: before the 2026-08-10 slot-0 rework it was
     * boot-provisioned and never floor-gated, and an empty-pool floor refusal
     * is a hang — worker_main's wait predicate stays false (head queued,
     * nothing active), so the worker hard-spins re-reading /proc/meminfo and
     * the first request never completes.  One bank's eager floor is small;
     * admission pressure on a tight box belongs to the SECOND bank onward. */
    const uint64_t avail = server_mem_available_bytes();
    if (server_bank_floor_refuses(n_provisioned, resident[idx] > KvBytes{}, avail, s->bank_marginal_bytes)) {
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

session_slot *server::empty_bank() {
    auto *s = this;
    if (s->pool_banks <= 0) return NULL;
    /* server_pick_free_bank's rule over the empty banks: the most resident pages, the lowest index on a tie */
    bool taken[PULSAR_SESSION_POOL_CAP];
    KvBytes resident[PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < s->n_slots; i++) {
        const session_slot *c = &s->slots[i];
        taken[i] = !c->provisioned || c->active_job || c->spilled || c->preempted || s->slot_frontier_pos(c) != 0;
        resident[i] = taken[i] ? KvBytes{} : pulsar_session_bank_touched_kv_bytes(s->sess, c->bank);
    }
    const int i = server_pick_free_bank(taken, resident, s->n_slots);
    return i < 0 ? NULL : &s->slots[i];
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



/* L275: the bank's last-turn anchor -- the position of the last user marker among
 * the tokens the bank PREFILLED, when a completed exchange (an assistant turn)
 * precedes it.  An agent client sends side requests beside a turn that share the
 * conversation through that marker and diverge INSIDE the last user turn: Claude
 * Code's session recap (the main prompt + a recap instruction, sent when the
 * terminal is unfocused), its subagent summaries ("Describe your most recent
 * action in 3-5 words...", ~150 tokens after the tool result) and its tool-use
 * summaries.  Measured on the pair 2026-10-06: every such request took a fresh
 * bank, evicted an idle one and reloaded the chain from disk (301 provisions,
 * 188 evictions in three hours); the next real turn of a conversation whose
 * bank had been evicted came back from disk on another bank.
 * -1 when the markers are unknown (server::turn_markers zeroed: the tokenizer
 * does not spell them), when the bank holds no completed exchange (a
 * first-turn bank: two unrelated conversations behind one system prompt share
 * the marker position and part of the first message, the 2026-10-04 clobber), or
 * when there is no user marker at all. */
int server_route_turn_anchor(const pulsar_tokens *bank, int prefilled, const pulsar_turn_markers *m) {
    if (!bank || !bank->v || !m || m->n_user <= 0 || m->n_assistant <= 0) return -1;
    if (prefilled > bank->len) prefilled = bank->len;
    int last_user = -1, first_assistant = -1;
    for (int i = 0; i < prefilled; i++) {
        const int role = pulsar_turn_marker_at(m, bank->v, prefilled, i);
        if (role == 1) last_user = i;
        else if (role == 2 && first_assistant < 0) first_assistant = i;
    }
    return first_assistant >= 0 && last_user > first_assistant ? last_user : -1;
}

/* L311: the bank's GENERATION-HEADER anchor -- the last token of the last assistant marker among the tokens the bank
 * PREFILLED (its last prompt's generation header: the history of complete messages before it is that prompt's message
 * endpoint), first turn included.  A request that carries that whole marker is this conversation's next turn whatever
 * the template did to the reply after it: a template that truncates history reasoning (Nemotron's
 * truncate_history_thinking, always on: the bank holds "<think>\nREASONING</think>answer", the next prompt re-renders
 * the turn "<think></think>answer") or a client that strips it diverges right after the header, past every complete
 * message (the 2026-10-21 opencode report: every Nemotron turn took a fresh bank).  The 2026-10-04 clobber diverged
 * inside the first user message, before any header.  -1 when the markers are unknown or there is none. */
int server_route_header_anchor(const pulsar_tokens *bank, int prefilled, const pulsar_turn_markers *m) {
    if (!bank || !bank->v || !m || m->n_assistant <= 0) return -1;
    if (prefilled > bank->len) prefilled = bank->len;
    for (int i = prefilled - 1; i >= 0; i--)
        if (pulsar_turn_marker_at(m, bank->v, prefilled, i) == 2) return i + m->n_assistant - 1;
    return -1;
}

/* L322: the bank's route anchor -- the shallower of its last-turn anchor and its generation-header anchor that exists.
 * Each alone is the evidence server_route_in_place reads (a request matching past it is this conversation's: its next
 * turn, its side request, an edit of its last turn), so the anchor is the first one the match may pass.  L311 combined
 * them by the DEEPER one, so a bank that also held a header past its last user marker -- always, once a side request
 * had run on it -- lost the last-turn rule: Claude Code's SUGGESTION MODE call (the conversation + ~340 tokens, after
 * every finished turn) ran in place and left its header at the bank's end, the next real turn diverged five tokens
 * past the real turn's end, inside that call's user turn, and took a fresh bank, which reloaded the whole chain from
 * disk (2026-10-24, Qwen at 178-193k: 32-34 GB, 22-25 s a turn).  -1 when neither exists. */
int server_route_anchor(const pulsar_tokens *bank, int prefilled, const pulsar_turn_markers *m) {
    const int turn = server_route_turn_anchor(bank, prefilled, m);
    const int header = server_route_header_anchor(bank, prefilled, m);
    if (turn < 0) return header;
    return header < 0 || turn < header ? turn : header;
}

/* L264: does the job continue on its best bank?  Yes when it carries everything
 * that bank PREFILLED (common >= prefilled: it IS that conversation's next turn,
 * whatever it did to the bank's generated tail -- an agent client drops the
 * previous turn's reasoning, so the echo diverges exactly there), when it
 * carries the bank's last user marker (common > anchor, L275: it is this
 * conversation's side request or an edit of its last turn, and what the resume
 * discards is at most that one turn), or when what the resume would discard is
 * under the protect floor.  A job that shares only a prefix -- a long common
 * system prompt, the measured case (2026-10-04: four SWE-agent conversations
 * behind one 2.3k-token prompt all landed on bank 0 and clobbered one another
 * every turn) -- diverges INSIDE the bank's first user turn, before any anchor,
 * so it takes a fresh bank, which restores the shared prefix from the disk
 * chain. */
bool server_route_in_place(int common, int score, int frontier, int prefilled, int protect_floor,
                           int anchor) {
    return common >= prefilled || (anchor >= 0 && common > anchor) || frontier - score < protect_floor;
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
 *   2. Among free slots with enough context, the highest SCORE wins (L264):
 *      where the bank's sync will resume (L325: the one resume rule,
 *      pulsar_session_bank_resume_at); the longer match breaks a tie.
 *   3. The winner continues in place unless that would discard meaningful
 *      warm state of a DIFFERENT conversation (server_route_in_place); then a
 *      fresh lazily provisioned slot is preferred (budget permitting, after
 *      an eviction if that is what it takes), and with the pool exhausted it
 *      falls back to the winner exactly like the single-session server did.
 * Returns NULL when the job must wait for a slot to free — except when
 * *reject_ctx is set nonzero (the owner slot's ctx_size), which means the
 * job can never run and must be failed, not left queued. *waiting_owner is
 * set when the NULL means "the continuation's owner slot is busy" -- or (L261)
 * "the deepest match is a busy slot whose client is gone": eviction cannot
 * help that job, only that slot finishing can. *clobbers is set when
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
    int best_common = -1, best_score = -1;
    static const int route_debug = getenv("PULSAR_ROUTE_DEBUG") != NULL;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (sl->active_job || !sl->provisioned || sl->ctx_size < needed) continue;
        const int common = s->slot_common_prefix(sl, &j->req.prompt);
        /* L325: the bank scores where its sync resumes -- pulsar_session_bank_resume_at, the one resume rule over the
         * token-seam stitch that the bank's own resume reads and an adopt (bank_adopt_point, L322) reads too: its grid
         * checkpoints AND its records (prompt end, message endpoint) within the match and its prefill frontier, or
         * the whole match for a history it continues.  The route scored the grid checkpoint below the match alone (a
         * record only for a bank holding the whole prompt, L289), so a bank whose turn-end record stood above its
         * thinned grid ranked below a shallower bank. */
        const int score = pulsar_session_bank_resume_at(s->sess, sl->bank, &j->req.prompt).raw();
        if (route_debug)
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: route-scan slot %d bank %u: frontier=%d common=%d score=%d",
                       i, sl->bank, s->slot_frontier_pos(sl), common, score);
        if (score > best_score || (score == best_score && common > best_common)) {
            best_score = score;
            best_common = common;
            best = sl;
        }
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
    const int job_anchor = kv_cache_chat_anchor_pos(&s->kv, &j->req.prompt, &s->turn_markers);
    if (job_anchor > 0) {
        const int anchor_ceiling =
            job_anchor + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
        if (anchor_ceiling > share_ceiling) share_ceiling = anchor_ceiling;
    }
    /* L261: an interrupted client re-sends before the server has seen the old
     * stream drop, so the conversation's bank is still BUSY with a request
     * nobody will read -- and routing past it took the disk and a fresh bank
     * (25,306 tokens re-prefilled, pair, 2026-10-01).  When the deepest match
     * for this job is a busy slot whose client is gone and whose prompt is
     * fully prefilled (decode init / decode), wait for it: the lanes'
     * per-quantum liveness poll abandons it within a quantum and the abandon
     * keeps the bank's history, and *waiting_owner keeps the caller from
     * evicting or provisioning meanwhile.  Not mid-prefill: that abandon
     * invalidates the bank (gen_prefill_fail), so waiting would only delay. */
    {
        session_slot *gone = NULL;
        int gone_common = -1;
        for (int i = 0; i < s->n_slots; i++) {
            session_slot *sl = &s->slots[i];
            if (!sl->active_job || !sl->provisioned || sl->ctx_size < needed) continue;
            if (!sl->gen || (sl->gen->phase != GEN_DECODE_INIT && sl->gen->phase != GEN_DECODE)) continue;
            const bool writer_failed = sl->gen->writer.failed;
            if (!writer_failed && !gen_client_disconnected(sl->active_job->fd)) continue;
            const int common = s->slot_common_prefix(sl, &j->req.prompt);
            if (common > gone_common) {
                gone_common = common;
                gone = sl;
            }
        }
        if (gone && gone_common > best_common && gone_common >= share_ceiling) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: slot routing: deepest match is busy bank %u whose client "
                       "is gone (common=%d, best free %d); waiting for it to abandon",
                       gone->bank, gone_common, best_common);
            *waiting_owner = true;
            return NULL;
        }
    }
    const int frontier = best ? s->slot_frontier_pos(best) : 0;
    const int prefilled = best && s->sess ? pulsar_session_bank_prefill_frontier(s->sess, best->bank) : 0;
    /* L275: the bank's last-turn anchor, from the history it prefilled; L311: or its generation header's; L322: the
     * shallower of the two */
    const int anchor = best && s->sess
        ? server_route_anchor(pulsar_session_bank_tokens(s->sess, best->bank), prefilled, &s->turn_markers)
        : -1;
    const bool in_place = best && server_route_in_place(best_common, best_score, frontier, prefilled,
                                                        s->slot_trivial_common_tokens, anchor);
    /* One line per bind decision: which bank, how deep the match, where it resumes, what it costs the bank, and
     * where the bank's last turn starts (a head that waits re-routes every quantum; its line repeats only when the
     * decision changes). */
    const uint64_t route_key = best ? 1u + (((uint64_t)best->bank << 1 | (in_place ? 1u : 0u)) ^
                                            ((uint64_t)(uint32_t)best_common << 17) ^ ((uint64_t)(uint32_t)best_score << 40))
                                    : 0u;
    if (best && route_key != j->route_logged) {
        j->route_logged = route_key;
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: route: best bank %u common %d resume %d frontier %d prefilled %d "
                   "anchor %d prompt %d -> %s",
                   best->bank, best_common, best_score, frontier, prefilled, anchor, j->req.prompt.len,
                   in_place ? "in place" : "fresh preferred");
    }
    if (in_place) return best;
    /* L328 (3): a provisioned bank that holds nothing IS a fresh bank.  A prompt queued on KV leaves the bank its try
     * provisioned behind, empty, and provisioning another on every retry filled the pool until the make-room eviction
     * took a conversation's bank for it (gpuq #463 / #473 / #474: banks 3..15 provisioned on retries, slot 0 evicted,
     * B on bank 0) -- the retry now takes the same bank again. */
    session_slot *empty = s->empty_bank();
    /* L280: a fresh bank is preferred to keep the best bank's conversation -- but the prompt's KV lands in it, and
     * when that cannot fit the resident-KV budget even by spilling idle banks (prompt_kv_check on the empty bank),
     * continuing on the best bank reuses its resident pages instead (the same clobber the pool-exhausted fallback
     * below takes; gen_begin persists the displaced history first).  Only when THAT fits. */
    if (best) {
        int order[PULSAR_SESSION_POOL_CAP], n_spill = 0;
        KvBytes need_fresh{}, need_best{};
        const prompt_kv_verdict vf = s->prompt_kv_check(empty, j, order, &n_spill, &need_fresh);
        if (vf == PROMPT_KV_WAIT || vf == PROMPT_KV_REFUSE) {
            const prompt_kv_verdict vb = s->prompt_kv_check(best, j, order, &n_spill, &need_best);
            if (vb == PROMPT_KV_FITS || vb == PROMPT_KV_SPILL) {
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: route: a fresh bank would add %.2f GiB of KV past the budget; continuing "
                           "on bank %u instead (it adds %.2f GiB)",
                           (double)need_fresh.raw() / (1024.0 * 1024.0 * 1024.0), best->bank,
                           (double)need_best.raw() / (1024.0 * 1024.0 * 1024.0));
                *clobbers = true;
                return best;
            }
        }
    }
    session_slot *fresh = empty ? empty : s->provision_slot(s->provision_ctx_for_job(j), refusal);
    /* Pool full of idle stale banks: without eviction HERE, every new
     * conversation falls through to the same bank and clobbers it serially
     * while the other banks sit pinned with dead state forever -- measured
     * 2026-08-10: an 8-bank pool pre-churned by 12 one-shots served every later
     * conversation cold off bank 1.  Evict an LRU idle bank (live-tool owners
     * protected) and retry once. */
    if (!fresh && server_refusal_evictable(*refusal, s->pool_banks > 0) && s->fresh_make_room())
        fresh = s->provision_slot(s->provision_ctx_for_job(j), refusal);
    if (fresh) {
        if (route_debug)
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: route-branch: FRESH bank %u", fresh->bank);
        return fresh;
    }
    *clobbers = best != NULL;
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
 * MemAvailable-floor refusals trigger eviction in POOL mode only
 * (server_refusal_evictable).  Classic mode: freeing two 2.5 GiB sessions
 * moved MemAvailable only 5.98 -> 6.07 GiB (GB10, driver 610, 2026-07-14) --
 * cudaFree'd memory does not promptly return to the kernel's gauge, so an
 * eviction there cannot relieve the floor and only churns snapshots.  Pool
 * mode frees nothing either, and needs nothing freed: the evicted bank keeps
 * its pages, provision_bank picks that resident hole first and reinstalls it
 * without paging anything in (server_bank_floor_refuses).  Refusing it queued
 * every new conversation behind a running one on a box that had served for a
 * while (the 2026-09-29 pair: c2 ran serially).
 *
 * Restore path: none of it is new. gen_begin's cache resolution already
 * takes a disk chain when it is the deepest state (L318, server_resume_pick), and the
 * "evict" snapshot is keyed exactly like a shutdown checkpoint — by rendered
 * transcript bytes, or by the visible protocol transcript when a live
 * responses/thinking binding covers the frontier. A returning client binds to
 * any free slot and restores from disk there; no extra routing metadata is
 * needed because the segment store's text-prefix lookup IS the metadata. Live
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
                             const bool *protect, bool allow_slot0, const bool *dominated) {
    int victim = -1;
    /* Slot 0 competes in LRU only when the caller says so (pool mode, where
     * bank 0 is just another bank): an idle conversation squatting bank 0 was
     * otherwise IMMORTAL, silently shrinking the evictable pool to N-1 and
     * seeding the churned-pool LRU domino (measured 2026-08-10: a one-shot's
     * leftovers on bank 0 forced an 8-conversation load into 7 banks, and the
     * shortage evicted a live conversation every turn forever).  L323: a
     * dominated slot (server::dominated_slots) goes before any LRU choice. */
    for (int i = allow_slot0 ? 0 : 1; i < n_slots; i++) {
        const session_slot *sl = &slots[i];
        if (!sl->provisioned || sl->active_job) continue;
        if (protect && protect[i]) continue;
        const bool dom = dominated && dominated[i], vdom = victim >= 0 && dominated && dominated[victim];
        if (victim < 0 || (dom && !vdom) ||
            (dom == vdom &&
             (sl->last_serviced_us < slots[victim].last_serviced_us ||
              (sl->last_serviced_us == slots[victim].last_serviced_us &&
               sl->est_cost_bytes < slots[victim].est_cost_bytes))))
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
    PULSAR_NVTX_FN();
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
 * bind — cold-replays its whole history. "Usable" (L264): the bank holds a grid
 * checkpoint inside the job's byte match, i.e. the job would resume there
 * rather than from 0; a match with none below it is no warmth worth keeping.
 * Best-effort by contract: callers retry without this overlay when it leaves
 * no victim, so binding always progresses. Worker thread only
 * (slot_common_prefix reads engine host carries). */
void server::worker_protect_queued_warm_matches(bool protect[PULSAR_SESSION_POOL_CAP]) {
    PULSAR_NVTX_FN();
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
        int best_i = -1, best_score = 0;
        for (int k = 0; k < s->n_slots && k < PULSAR_SESSION_POOL_CAP; k++) {
            const session_slot *sl = &s->slots[k];
            if (!sl->provisioned) continue;
            /* L325: the route's score (pulsar_session_bank_resume_at, choose_slot_for_job) */
            const int score = pulsar_session_bank_resume_at(s->sess, sl->bank, &q->req.prompt).raw();
            if (score > best_score) { best_score = score; best_i = k; }
        }
        if (best_i >= 0) protect[best_i] = true;
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
    PULSAR_NVTX_FN();
    auto *s = this;
    if (s->pool_banks > 0) {
        /* Pool mode: a provision costs one bank's ledger marginal, and the
         * hole an eviction leaves keeps its pages, which provision_bank reuses
         * without consulting the floor (server_bank_floor_refuses).  So an
         * eviction helps when some idle bank can go and the ledger then admits
         * one marginal.  (The classic arithmetic below prices a whole
         * session: at --ctx 1M that is the 44 GiB pool, which no box's
         * MemAvailable ever admits, so this gate never opened in pool mode.) */
        uint64_t reclaimable = 0;
        bool any = false;
        for (int i = 0; i < s->pool_banks; i++) {
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
        return server_kv_admits(s->kv_budget_bytes, after, s->bank_marginal_bytes);
    }
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
    return evicted_ctx;
}

bool server::worker_evict_one(bool protect[PULSAR_SESSION_POOL_CAP]) {
    PULSAR_NVTX_FN();
    auto *s = this;
    bool dominated[PULSAR_SESSION_POOL_CAP];
    s->dominated_slots(dominated);
    const int vi = server_evict_pick_victim(s->slots, s->n_slots, protect,
                                            /*allow_slot0=*/s->pool_banks > 0, dominated);
    if (vi < 0) return false;
    session_slot *sl = &s->slots[vi];
    if (dominated[vi])
        server_log(PULSAR_LOG_KVCACHE, "pulsar-server: evict: bank %u is dominated (another bank holds its whole "
                                       "history)", sl->bank);

    /* Tier-2: install the victim's bank so the snapshot store below reads its
     * OWN frontier (not the pool's live cursor).  L330: a guard-spilled victim
     * installs empty -- its history is on disk already, where the spill put it --
     * so there is nothing to store; one that cannot be re-backed is skipped (this
     * path is discarding the conversation anyway). */
    const bool switched = s->bank_switch(sl->bank);
    const pulsar_tokens *tokens = switched ? pulsar_session_tokens(s->sess) : NULL;
    const int live_tokens = tokens ? tokens->len : 0;
    bool stored = false;
    if (switched && s->kv.enabled && live_tokens >= s->kv.opt.min_tokens) {
        /* L264 S4: persisting is idempotent -- a chain the store already holds
         * writes nothing -- so "stored" is "the store holds this history". */
        s->kv_cache_persist(sl, "evict", true);
        stored = true;
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
     * flag (invariant: physical freed IFF spilled).  server_bank_switch above
     * re-backed a spilled victim (physical present, flag cleared); this is the
     * case where that failed. */
    if (s->pool_banks > 0 && sl->spilled) {
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



/* L323: the ONE bank-dominance predicate, over injected per-slot facts (server::dominated_slots reads them).  Slot a
 * is DOMINATED when it holds a history (hist_len[a] > 0; an empty bank is plain LRU's business) and another slot k
 * -- resident or spilled, any provisioned bank (an unprovisioned one has hist_len 0) -- holds a's whole history as
 * its prefix (common[a][k] >= hist_len[a]) and reaches at least as far: hist_len[k] > hist_len[a], or an identical
 * copy at a lower index (so of two identical copies exactly one is dominated).  Its KV is redundant -- every request
 * it serves, k serves at least as deep, and a spilled k's history is on disk -- so the guard spills it, and the
 * make-room eviction evicts it, before any LRU choice.  common[a][k] is read only where the length test holds (the
 * caller fills it there).  Before L323 this was plan-33 inc D's "LRU-superseded" pick, consulted by fresh_make_room
 * alone and over resident banks only; the guard spilled by LRU and kept dominated copies resident (2026-10-24: 9 of
 * 16 banks at 150-205k, mostly copies of one conversation). */
static bool bank_dominated_core(int a, int n_slots, const int *hist_len, const int *const *common) {
    if (hist_len[a] <= 0) return false;
    for (int k = 0; k < n_slots; k++) {
        if (k == a || hist_len[k] < hist_len[a] || (hist_len[k] == hist_len[a] && k > a)) continue;
        if (common[a][k] >= hist_len[a]) return true;
    }
    return false;
}

/* L323/L324: the common-prefix matrix of the slots' histories, the ONE fill both the dominance predicate and the
 * duplicate census read: hist_len[i] (0 = no history; unprovisioned slots have none) and, where hist_len[i] > 0 and
 * hist_len[k] >= hist_len[i] (k != i), common_rows[i][k] = their common token prefix; -1 elsewhere.  Host-carry
 * prefix reads (the same reads routing does on idle banks; no CUDA), only where a reader looks: two conversations
 * part within a few thousand tokens, so a compare runs long only between copies of one. */
static void bank_common_matrix(server *s, int hist_len[PULSAR_SESSION_POOL_CAP],
                               int common_rows[PULSAR_SESSION_POOL_CAP][PULSAR_SESSION_POOL_CAP]) {
    pulsar_session *pool = s->sess;
    const pulsar_tokens *hist[PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < s->n_slots; i++) {
        const session_slot *a = &s->slots[i];
        hist[i] = a->provisioned ? pulsar_session_bank_tokens(pool, a->bank) : NULL;
        hist_len[i] = hist[i] ? hist[i]->len : 0;
    }
    for (int i = 0; i < s->n_slots; i++)
        for (int k = 0; k < s->n_slots; k++)
            common_rows[i][k] = (hist_len[i] > 0 && k != i && hist_len[k] >= hist_len[i])
                ? pulsar_session_bank_common_prefix(pool, s->slots[k].bank, hist[i]) : -1;
}

void server::dominated_slots(bool dominated[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) dominated[i] = false;
    if (!s->sess || s->pool_banks <= 0) return;
    int hist_len[PULSAR_SESSION_POOL_CAP];
    int common_rows[PULSAR_SESSION_POOL_CAP][PULSAR_SESSION_POOL_CAP];
    const int *common[PULSAR_SESSION_POOL_CAP];
    bank_common_matrix(s, hist_len, common_rows);
    for (int i = 0; i < s->n_slots; i++) common[i] = common_rows[i];
    for (int i = 0; i < s->n_slots; i++) dominated[i] = bank_dominated_core(i, s->n_slots, hist_len, common);
}

/* L324: the common prefix of slots a and b from the matrix (read in the direction bank_common_matrix fills). */
static int dup_pair_common(int a, int b, const int *hist_len, const int *const *common) {
    return hist_len[a] <= hist_len[b] ? common[a][b] : common[b][a];
}

void kv_dup_census_core(int n, const int *hist_len, const bool *resident, const int *const *common, int grid,
                        KvBytes page_bytes, kv_dup_census *out, uint32_t *deep) {
    memset(out, 0, sizeof *out);
    out->grid = grid;
    out->page_bytes = page_bytes;
    if (deep)
        for (int a = 0; a < n; a++) deep[a] = 0;
    if (grid <= 0) return;
    int pages[PULSAR_SESSION_POOL_CAP];
    int max_pages = 0;
    for (int a = 0; a < n; a++) {
        pages[a] = resident[a] && hist_len[a] > 0 ? hist_len[a] / grid : 0;
        if (pages[a] > 0) out->banks++;
        out->pages_resident += (uint64_t)pages[a];
        max_pages = std::max(max_pages, pages[a]);
    }
    /* Page p: the banks that hold it whole fall into classes by "same ids to (p+1)*grid" (an equivalence: prefix
     * agreement is transitive).  A class's first member stands for it; the others are its extra copies. */
    for (int p = 0; p < max_pages; p++) {
        const int end = (p + 1) * grid;
        for (int a = 0; a < n; a++) {
            if (pages[a] <= p) continue;
            bool first = true;
            for (int b = 0; b < a && first; b++)
                if (pages[b] > p && dup_pair_common(a, b, hist_len, common) >= end) first = false;
            if (!first) continue;
            out->pages_distinct++;
            for (int b = a + 1; b < n; b++)
                if (pages[b] > p && dup_pair_common(a, b, hist_len, common) >= end) { out->pages_shared++; break; }
        }
    }
    out->pages_duplicate = out->pages_resident - out->pages_distinct;
    out->dup_bytes = page_bytes.times(out->pages_duplicate);
    for (int a = 0; a < n; a++)
        for (int b = a + 1; b < n; b++) {
            if (pages[a] <= 0 || pages[b] <= 0) continue;
            const int c = dup_pair_common(a, b, hist_len, common);
            if (c < grid) continue;
            const int cls = c >= PULSAR_KV_DUP_DEEP_TOKENS ? 1 : 0;
            out->pairs[cls]++;
            out->longest[cls] = std::max(out->longest[cls], c);
            if (cls && deep) deep[a] |= 1u << b;
        }
}

void server::kv_dup_census_update() {
    auto *s = this;
    if (!s->sess || s->pool_banks <= 0) return;
    int hist_len[PULSAR_SESSION_POOL_CAP];
    int common_rows[PULSAR_SESSION_POOL_CAP][PULSAR_SESSION_POOL_CAP];
    const int *common[PULSAR_SESSION_POOL_CAP];
    bool resident[PULSAR_SESSION_POOL_CAP];
    bank_common_matrix(s, hist_len, common_rows);
    for (int i = 0; i < s->n_slots; i++) {
        common[i] = common_rows[i];
        resident[i] = s->slots[i].provisioned && !pulsar_session_bank_is_evicted(s->sess, s->slots[i].bank);
    }
    const int grid = pulsar_kvchain_grid(s->sess);
    const KvBytes page_bytes = grid > 0 ? pulsar_session_quantum_growth_bytes_per_bank(s->sess, KvRows{(uint32_t)grid})
                                        : KvBytes{};
    kv_dup_census c;
    uint32_t deep[PULSAR_SESSION_POOL_CAP];
    kv_dup_census_core(s->n_slots, hist_len, resident, common, grid, page_bytes, &c, deep);
    /* a pair that shares more than the preamble: once per pair until it stops (a bank that moves on clears it) */
    for (int a = 0; a < s->n_slots; a++) {
        const uint32_t fresh = deep[a] & ~s->w_dup_deep[a];
        for (int b = a + 1; b < s->n_slots; b++) {
            if (!(fresh >> b & 1u)) continue;
            const int shared = dup_pair_common(a, b, hist_len, common);
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: kv dup: banks %u and %u share %d tokens (%d whole pages of %d, %.2f GiB held "
                       "twice) -- deeper than the preamble (%d)", s->slots[a].bank, s->slots[b].bank, shared,
                       shared / grid, grid,
                       (double)page_bytes.times((uint64_t)(shared / grid)).raw() / (1024.0 * 1024.0 * 1024.0),
                       PULSAR_KV_DUP_DEEP_TOKENS);
        }
        s->w_dup_deep[a] = deep[a];
    }
    pthread_mutex_lock(&s->mu);
    s->m_dup = c;
    s->m_dup_censuses++;
    s->m_dup_bytes_peak = std::max(s->m_dup_bytes_peak, c.dup_bytes);
    pthread_mutex_unlock(&s->mu);
}

/* Evict exactly one NON-trunk victim so a warm fork gets a free bank. Trunk is
 * always protected (a sibling still matches it); dominated victims go first,
 * else plain LRU (worker_evict_one's picker). Reuses the proven eviction
 * body (snapshot + ledger release + bank reset). Worker thread only; returns
 * true when a bank was freed. */
bool server::fresh_make_room() {
    auto *s = this;
    if (s->pool_banks <= 0) return false;
    bool protect[PULSAR_SESSION_POOL_CAP];
    s->worker_protect_queued_owner_slots(protect);      /* live-tool owners */
    /* Two-phase: first avoid banks that are a queued job's usable warm match
     * (the fresh-path domino), then — if that leaves no victim — retry without
     * the overlay, because binding must progress. */
    bool with_warm[PULSAR_SESSION_POOL_CAP];
    memcpy(with_warm, protect, sizeof with_warm);
    s->worker_protect_queued_warm_matches(with_warm);
    if (s->worker_evict_one(with_warm)) return true;
    return s->worker_evict_one(protect);
}



/* =========================================================================
 * L329 (a): the shared-prefix wait (PULSAR_SERVER_PREFIX_WAIT_MS_DEFAULT in pulsar_server_internal.h says why).
 *
 * Production 2026-10-23 16:08 (Qwen, Claude Code fanning out 11 subagents over a ~25k-token prompt sharing 20-24k):
 * five requests bound at 16:08:23 adopted the leader's 12288 checkpoint and each prefilled [12288, 20480) again,
 * six more at 16:08:50 adopted 20480; 108k prompt tokens prefilled for ~68k of distinct work, the leader's prompt
 * done after 100.6 s.  Here a request that a bound prefill will serve deeper waits in the queue for it.
 * ========================================================================= */

static bool slot_prefilling(const session_slot *c);

const char *const prefix_wait_names[PREFIX_WAIT_OUTCOMES] = {"started", "reached", "left", "timeout"};

int server_prefix_wait_pick(int have, int prompt_len, int grid, int min_gain, const prefix_lead *leads, int n,
                            int *mark) {
    int best = -1;
    *mark = 0;
    if (grid <= 0) return -1;
    for (int i = 0; i < n; i++) {
        const prefix_lead *l = &leads[i];
        int cut = l->shared;
        if (cut > prompt_len - 1) cut = prompt_len - 1;   /* the request's sync evaluates at least its last token */
        if (cut > l->len - 1) cut = l->len - 1;           /* inside the leader's prompt: its prefill stops there */
        cut = cut / grid * grid;
        /* a cut the leader stands at or past is served now (have), and one the adopt would not take is not waited
         * for (server_resume_memory_depth: more than min_gain past what serves it) */
        if (cut <= l->pos || cut <= have + min_gain) continue;
        if (cut > *mark || (cut == *mark && l->pos > leads[best].pos)) {
            best = i;
            *mark = cut;
        }
    }
    return best;
}

prefix_wait_verdict server_prefix_wait_check(bool live, int leader_pos, int mark, double waited_ms, double bound_ms) {
    if (!live) return PREFIX_WAIT_LEFT;
    if (leader_pos >= mark) return PREFIX_WAIT_REACHED;
    if (waited_ms >= bound_ms) return PREFIX_WAIT_TIMEOUT;
    return PREFIX_WAIT_HOLD;
}

int server_prefix_mark_merge(int cur, int want, int pos) {
    return cur <= pos || cur > want ? want : cur;
}

int server_prefix_wait_parse(const char *text) {
    if (!text || !*text) return PULSAR_SERVER_PREFIX_WAIT_MS_DEFAULT;
    long v = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        v = v * 10 + (*p - '0');
        if (v > 24L * 3600 * 1000) return -1;   /* past a day is no bound */
    }
    return (int)v;
}

void server::prefix_wait_release(job *const *queued, int n) {
    auto *s = this;
    const double now = server_now_sec();
    for (int k = 0; k < n; k++) {
        prefix_wait *w = &queued[k]->pw;
        if (!w->leader) continue;
        session_slot *ls = &s->slots[w->slot];
        const bool live = slot_prefilling(ls) && ls->active_job->seq == w->leader;
        const int pos = live ? pulsar_session_bank_pos(s->sess, (uint32_t)ls->bank) : 0;
        const double waited_ms = (now - w->since) * 1e3;
        const prefix_wait_verdict v = server_prefix_wait_check(live, pos, w->mark, waited_ms, (double)s->prefix_wait_ms);
        if (v == PREFIX_WAIT_HOLD) {
            ls->gen->share_mark = server_prefix_mark_merge(ls->gen->share_mark, w->mark, pos);
            continue;
        }
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: prefix wait %s after %.1f s: a queued request (prompt %d) waited for bank %u to "
                   "stand at %d (it stands at %d%s); it binds now", prefix_wait_names[v], waited_ms / 1e3,
                   queued[k]->req.prompt.len, ls->bank, w->mark, pos, live ? "" : ", its prefill ended");
        pthread_mutex_lock(&s->mu);
        s->m_prefix_waits[v]++;
        if (v == PREFIX_WAIT_REACHED) s->m_prefix_wait_gain += (uint64_t)(w->mark - w->have);
        pthread_mutex_unlock(&s->mu);
        w->leader = 0;
    }
}

bool server::prefix_wait_decide(job *j) {
    auto *s = this;
    prefix_wait *w = &j->pw;
    if (w->decided) return false;
    w->decided = true;   /* once: a released waiter never waits again, so waits never chain */
    const request *r = &j->req;
    /* the adopt is what serves the cut: a pooled, single-GPU text request that no live tool state pins to its bank
     * (pulsar_session_bank_adopt refuses a tensor-parallel engine and an image history) */
    if (s->prefix_wait_ms <= 0 || s->pool_banks <= 1 || !s->sess || pulsar_engine_is_tp(s->engine) ||
        r->n_images > 0 || r->prompt.len <= 0 || r->responses_live_call_ids.len > 0 ||
        r->anthropic_live_call_ids.len > 0)
        return false;
    prefix_lead leads[PULSAR_SESSION_POOL_CAP];
    int lead_slot[PULSAR_SESSION_POOL_CAP];
    int n = 0, have = 0;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->spilled || sl->preempted) continue;
        /* the deepest state a bank serves the prompt from now: its own resume, which is what an adopt of it reads */
        const int at = pulsar_session_bank_resume_at(s->sess, (uint32_t)sl->bank, &r->prompt).raw();
        if (at > have) have = at;
        if (!slot_prefilling(sl) || sl->gen->j->req.n_images > 0 || !sl->gen->prompt_for_sync) continue;
        const pulsar_tokens *lp = sl->gen->prompt_for_sync;
        const int m = lp->len < r->prompt.len ? lp->len : r->prompt.len;
        int shared = 0;
        while (shared < m && lp->v[shared] == r->prompt.v[shared]) shared++;
        leads[n] = {shared, pulsar_session_bank_pos(s->sess, (uint32_t)sl->bank), lp->len};
        lead_slot[n++] = i;
    }
    int mark = 0;
    const int pick = server_prefix_wait_pick(have, r->prompt.len, (int)pulsar_session_resume_grid(s->sess),
                                             PULSAR_SERVER_ADOPT_MIN_GAIN, leads, n, &mark);
    if (pick < 0) return false;
    session_slot *ls = &s->slots[lead_slot[pick]];
    w->leader = ls->active_job->seq;
    w->slot = lead_slot[pick];
    w->mark = mark;
    w->have = have;
    w->since = server_now_sec();
    ls->gen->share_mark = server_prefix_mark_merge(ls->gen->share_mark, mark, leads[pick].pos);
    pthread_mutex_lock(&s->mu);
    s->m_prefix_waits[PREFIX_WAIT_HOLD]++;
    pthread_mutex_unlock(&s->mu);
    server_log(PULSAR_LOG_KVCACHE,
               "pulsar-server: prefix wait: a queued request (prompt %d) waits for bank %u's prefill to stand at %d "
               "(shares %d of its %d; it stands at %d, the best state for the request now %d; bound %d ms)",
               r->prompt.len, ls->bank, mark, leads[pick].shared, leads[pick].len, leads[pick].pos, have,
               s->prefix_wait_ms);
    return true;
}



/* Bind the head job to a slot if routing allows it. Strict FIFO: when the
 * head must wait (its owner slot is busy, or no fitting slot is free), later
 * jobs wait behind it — simple and starvation-free.  L329 (a): a job waiting
 * for another prompt's prefill is not the head; it keeps its place and is
 * passed over until it may go on (bounded). Returns true when the
 * head was consumed: bound to a slot, or failed explicitly (a continuation
 * that cannot fit its owner slot — see choose_slot_for_job). When the head
 * cannot be placed cleanly (nothing fits, or only a warm slot it would
 * clobber), it is not waiting on a busy owner, and the provisioning refusal
 * is one eviction can relieve (server_refusal_evictable: full pool / full
 * ledger, and in pool mode the MemAvailable floor -- see the increment-4
 * block above), idle slots are
 * evicted LRU-first until the head binds without clobbering or eviction
 * stops helping — then the clobber fallback binds it exactly like the
 * increment-3 scheduler did. */
bool server::worker_try_bind() {
    PULSAR_NVTX_FN();
    auto *s = this;
    /* peek: only the worker pops, so the queued jobs stay valid once the lock is dropped (a client thread only
     * appends at the tail and flags its own job cancelled) */
    job *queued[PULSAR_SERVER_MAX_CLIENTS];
    int n_queued = 0;
    pthread_mutex_lock(&s->mu);
    for (job *q = s->head; q && n_queued < PULSAR_SERVER_MAX_CLIENTS; q = q->next) queued[n_queued++] = q;
    /* Nothing queued means nothing is blocked; without this the gauge would
     * keep reporting the last reason long after the queue drained. */
    if (n_queued == 0) s->m_queue_block_reason = 0;
    pthread_mutex_unlock(&s->mu);
    if (n_queued == 0) return false;

    /* L329 (a): the head is the first queued job that does not wait for another prompt's prefill -- the waiters
     * that may go on first go on (they kept their places), a job never seen before decides its wait once here --
     * or the first cancelled one, reaped below */
    s->prefix_wait_release(queued, n_queued);
    job *j = NULL;
    bool head_cancelled = false;
    for (int k = 0; k < n_queued && !j; k++) {
        job *q = queued[k];
        pthread_mutex_lock(&q->mu);
        head_cancelled = q->cancelled;
        pthread_mutex_unlock(&q->mu);
        if (head_cancelled || (!q->pw.leader && !s->prefix_wait_decide(q))) j = q;
    }
    if (!j) {
        pthread_mutex_lock(&s->mu);
        s->m_queue_block_reason = 0;   /* every queued job waits for a prefill: not an admission refusal */
        pthread_mutex_unlock(&s->mu);
        return false;
    }

    /* Reap a head whose client disconnected while queued BEFORE the bind
     * machinery: choose_slot_for_job can evict warm banks and spend seconds
     * inside pulsar_session_create — none of it for a dead fd. The client
     * thread only flags and stays parked on j->cv until this signal, so the
     * worker remains the sole popper (the invariant the queued-job protect
     * helpers rely on). No HTTP error write: the socket is gone. */
    if (head_cancelled) {
        s->queue_unlink(j);
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
        server_refusal_evictable(refusal, s->pool_banks > 0))
    {
        bool protect[PULSAR_SESSION_POOL_CAP];
        s->worker_protect_queued_owner_slots(protect);
        if (s->worker_eviction_could_help(j, protect)) {
            while ((!sl || clobbers) && server_refusal_evictable(refusal, s->pool_banks > 0))
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
        s->queue_unlink(j);
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
        s->queue_unlink(j);
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
    /* L280: the prompt's KV must fit the resident-KV budget before its prefill or chain load starts (the guard
     * only watches decode quanta): spill idle banks, or keep the head queued for a busy one, or answer 503 --
     * never bind it into memory pressure. */
    {
        bool kv_refused = false;
        if (!s->prompt_kv_admit(sl, j, &kv_refused)) {
            if (!kv_refused) return false;
            s->queue_unlink(j);
            pthread_mutex_lock(&j->mu);
            j->done = true;
            pthread_cond_signal(&j->cv);
            pthread_mutex_unlock(&j->mu);
            return true;
        }
    }

    pthread_mutex_lock(&s->mu);
    s->m_queue_block_reason = 0; /* the head bound — the queue is moving again */
    queue_unlink_locked(s, j);
    s->n_generating++;
    pthread_mutex_unlock(&s->mu);

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
    /* L328: the resident KV and the admission base (host reads of the banks' high-waters, as the guard's) */
    const KvBytes kv_touched = s->guard_enabled ? pulsar_session_touched_kv_bytes(s->sess) : KvBytes{};
    const KvBytes kv_reserved = s->guard_enabled ? s->kv_reserved_bytes(NULL) : KvBytes{};
    const KvBytes kv_budget = s->guard_enabled ? s->guard_budget() : KvBytes{};
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
        /* L112: adaptive draft depth, live-or-carry (pure host read) -- DSpark's; the Qwen MTP lane's
         * depth is fixed and the family refuses the query. */
        s->m_slot_depth[i] = sl->provisioned && pulsar_engine_has_spec_rounds(s->engine)
                ? pulsar_session_bank_spec_depth(s->sess, sl->bank) : 0;
    }
    s->m_spec = m;
    s->m_kv_touched = kv_touched;
    s->m_kv_reserved = kv_reserved;
    s->m_kv_budget = kv_budget;
    s->m_gen_tokens = s->w_gen_tokens;
    s->m_prefill_chunk_tokens = s->w_prefill_chunk_tokens;
    s->m_spec_overflow_rounds = s->w_spec_overflow_rounds;
    s->m_spec_pick_cut_rows = s->w_spec_pick_cut_rows;
    s->m_spec_grammar_cut_rows = s->w_spec_grammar_cut_rows;
    pulsar_engine_verify_curve_view(s->engine, &s->m_verify_curve);
    pulsar_engine_spec_calibration(s->engine, &s->m_spec_calib);
    s->m_drafter_name = pulsar_engine_drafter_name(s->engine);
    memcpy(s->m_spec_phase_ms, s->w_spec_phase_ms, sizeof s->m_spec_phase_ms);
    s->m_spec_phase_rounds = s->w_spec_phase_rounds;
    memcpy(s->m_spec_syncs, s->w_spec_syncs, sizeof s->m_spec_syncs);
    s->m_decode_lane = s->w_decode_lane;
    /* Bump and wake the /metrics/stream subscribers. A generation counter
     * rather than a payload comparison on purpose: "did anything move" over a
     * dozen fields is exactly the check pulsar-gui got wrong, because the
     * derived rates are window-relative and differ on every sample even when
     * nothing happened. The worker already knows when it published. */
    s->metrics_generation++;
    pthread_cond_broadcast(&s->stream_cv);
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
    PULSAR_NVTX_FN();
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

void server::worker_service_slot(session_slot *sl) {
    if (sl->gen && sl->gen->phase != GEN_DONE) {
        generate_job_step(sl);
        if (sl->gen) slot_writer_flush(&sl->gen->writer);
        sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    }
    if (!sl->gen || sl->gen->phase == GEN_DONE) worker_finish_slot(sl);
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
/* A slot's draw request (THE sampler, L302) -- L303: through the grammar's mask while a region is live; the row
 * itself stays the target distribution the logprob capture reads.  *mask is the caller's storage; the draw request
 * reads it.  false with g->err set: the grammar failed (the request fails). */
static bool gen_draw_req(server *s, gen_state *g, float temp, int top_k, float top_p, float min_p,
                         std::vector<uint32_t> *mask, pulsar_row_draw *d) {
    const bool live = gen_constrain_live(g);
    if (live) {
        mask->resize(gen_constrain_words(s));
        if (!gen_constrain_mask(s, g, mask->data())) return false;
    }
    *d = pulsar_row_draw{temp, top_k, top_p, min_p, &g->rng, false, live ? mask->data() : NULL};
    return true;
}

/* the slot's draw from the pool session's live row.  -1 with g->err set: the sampler refused the row, or the
 * grammar failed. */
static int gen_sample_pool(server *s, gen_state *g, pulsar_session *pool, float temp, int top_k, float top_p,
                           float min_p) {
    std::vector<uint32_t> mask;
    pulsar_row_draw d;
    if (!gen_draw_req(s, g, temp, top_k, top_p, min_p, &mask, &d)) return -1;
    const int tok = pulsar_session_draw(pool, &d);
    if (tok < 0) snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
    return tok;
}

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
    return sl->active_job && !sl->preempted && g && g->phase == GEN_DECODE;   /* L314: a parked decoder is not */
}

/* Decode-lane selection, ONE rule for the whole sweep (L179): the spec-batched
 * lane (3) takes every decoder when a drafter runs behind the round API
 * (pulsar_engine_has_spec_rounds), every decoder speculates (spec_enabled: a
 * logprobs request does not), none sits in the plain multiseq batch, and the
 * decoders are no more than one shared verify forward carries, one base row each
 * (pulsar_engine_fused_heads_max, the family's verify width -- L284: DeepSeek 32;
 * L288 C1: Qwen 32 too; the allocator shortens the drafts to fit, spec_alloc_rows);
 * the plain batched lane (2) otherwise; the classic lane (1) with no pool.  Lane 3
 * keeps the spec_decode counters advancing; lane 2 does not.  Where lane 3 COULD
 * carry every decoder (server_spec_lane_carries), whether it does is priced
 * (L284 lane cost, lane_price_pick): the spec lane is offered to this pick only
 * when it is the faster, and decoders left in the plain batch rejoin it through
 * server::batch_leave before this pick (L271, L284). */
/* L271 / L284: whether the spec lane could take every decoder: the drafter runs behind the round API, they are
 * no more than one verify forward carries, and each speculates (the lane pick's own conditions, batch membership
 * aside: a decoder in the plain batch leaves it when the price picks spec, so a leave is never undone by the next
 * pick).  L303: a constrained decoder speculates too -- the round's rows carry its masks. */
static bool server_spec_lane_carries(bool spec_rounds, uint32_t verify_rows, session_slot *const *dec, int n_dec) {
    if (!spec_rounds || n_dec <= 0 || (uint32_t)n_dec > verify_rows) return false;
    for (int i = 0; i < n_dec; i++) {
        const gen_state *dg = dec[i]->gen;
        if (!dg || !dg->spec_enabled) return false;
    }
    return true;
}

/* ==== L284 lane cost: ONE priced choice between the plain and spec lanes, every family ====================
 *
 * Tyler 2026-10-08: "Make the model behaviors match."  The spec lane's tokens a round over its round's cost
 * against the plain lane's decoders a step over its step's cost, one rule for every family.  Each lane's price
 * for the N decoders at hand, tokens/s, is
 *   - MEASURED where that lane has run at this N: the tokens it committed over its steps' wall, EW
 *     (lane_price::meas_*, reset when N moves);
 *   - else PREDICTED from the engine's fits (pulsar_engine_lane_cost, one structure, step_ms = flat + row *
 *     rows) and the spec lane's measured yield per bank per round (lane_price::tau / rho):
 *         plain = N / plain_ms(N)
 *         spec  = N * tau / spec_ms(N * rho)       (rows past the verify width: the allocator rations the
 *                                                   drafts, so the accepted drafts prorate with those kept)
 * A prediction only ever starts a MEASUREMENT (a probe of the other lane); the decoders change lanes on
 * measured prices at their N.  A linear fit extrapolated past the rows it has seen was 15% off on DeepSeek at
 * N8 (spec priced 84 tok/s against the 99.5 it serves) and moved eight decoders to a slower plain lane.
 * No family constant enters: the families' verify widths, their drafters' acceptance and their
 * kernels' costs are all in the measured numbers.  (L284 specrows measured why it must be priced: Qwen's spec
 * lane at ~0.55 acceptance lost to plain at N >= 3, 64.6 vs 67.2 tok/s at N3 and 76.0 vs 91.1 at N6, where a
 * lone DeepSeek stream gains 11% from it.)
 *
 * The controller's own constants, not a deployment's:
 *   - hysteresis: the other lane must price LANE_PRICE_MARGIN faster, and the lane in force has run
 *     LANE_PRICE_HOLD_STEPS steps since it was chosen or last weighed, before a probe or a switch (a switch
 *     costs every decoder a bank restore);
 *   - no number, no decision: a lane with no price at this N, or a predicted price past the margin, is
 *     probed -- run until LANE_PRICE_PROBE_STEPS of its steps are in -- and the probe's end weighs the two,
 *     the lane in force keeping ties within the margin;
 *   - prices go stale with the content (acceptance) and the context (cost): once the lane in force has run
 *     LANE_PRICE_REPROBE_STEPS steps since its last weighing, the other lane is probed again.  Counted in
 *     steps, not quanta (a spec quantum at twelve decoders is one round): one 32-step probe per 512 steps is
 *     ~6% of the decode time, so at the measured 10-15% gap probing costs about 1% at most;
 *   - but a re-measure is evidence-based, not a clock (L289): while the lane in force's MEASURED price clears
 *     the other lane's price at this N -- measured, or predicted by its fit (L288) -- by LANE_PRICE_CLEAR, a flip
 *     needs the other lane to move by more than its price moves between weighings, so the scheduled re-measure
 *     waits LANE_PRICE_CLEAR_SPAN times as long -- the bound on how old a measurement may grow before it is taken
 *     again.  The lane in force is measured every step it runs, so the clearance is read fresh at each pick; a
 *     narrower gap, an unmeasured lane in force or no price for the other lane re-measures on the plain schedule.
 *     (A fit's misprediction seen so far was optimistic for the other lane -- DeepSeek N8, plain 111 predicted
 *     against 80 measured -- and the clearance is 25%: a cleared prediction does not hide a faster lane.)  (L288, Qwen one stream: spec 51-78 tok/s against
 *     plain 33.8-36.1 measured, and the 512-step clock re-probed plain 13 times in 48 requests at ~0.67 s each,
 *     ~9% of a probed request; the plain price it re-measured moved 7% end to end.) */
#define LANE_PRICE_MARGIN        0.04
#define LANE_PRICE_HOLD_STEPS    32u
#define LANE_PRICE_PROBE_STEPS   32u
#define LANE_PRICE_REPROBE_STEPS 512u
#define LANE_PRICE_CLEAR         1.25   /* six margins: past the 7% the plain price drifted over L288's whole run */
#define LANE_PRICE_CLEAR_SPAN    8u     /* 4096 steps: ~16k tokens of one Qwen stream before the price is re-taken */
#define LANE_PRICE_ALPHA         0.0625f   /* the measured prices' and yield's EW weight: a probe's 32 steps
                                              carry ~7/8 of the price it ends on */

const char *const server_round_phase_name[SERVER_ROUND_PHASE_COUNT] = {"plan", "forward", "round_end", "emit",
                                                                       "draft", "commit"};

/* One decode step on lane `li` (PULSAR_LANE_PLAIN / _SPEC) by `n` decoders: `tokens` committed in `ms`. */
static void lane_price_measure(lane_price *lp, int li, int n, int tokens, double ms) {
    if (n <= 0 || tokens < 0 || !(ms > 0.0)) return;
    if (n != lp->meas_n) {   /* another decoder count: what was measured prices it no longer */
        memset(lp->meas_tok, 0, sizeof lp->meas_tok);
        memset(lp->meas_ms, 0, sizeof lp->meas_ms);
        memset(lp->meas_steps, 0, sizeof lp->meas_steps);
        lp->meas_n = n;
    }
    const double a = lp->meas_steps[li] ? (double)LANE_PRICE_ALPHA : 1.0;
    lp->meas_tok[li] += a * ((double)tokens - lp->meas_tok[li]);
    lp->meas_ms[li] += a * (ms - lp->meas_ms[li]);
    lp->meas_steps[li]++;
}

/* One plain decode step: `n` decoders committed a token each in `ms`. */
static void lane_price_observe_plain(lane_price *lp, int n, double ms) {
    lane_price_measure(lp, PULSAR_LANE_PLAIN, n, n, ms);
}

/* One spec decode round: `tokens` committed by `banks` banks over `rows` verify rows in `ms`. */
static void lane_price_observe_spec(lane_price *lp, int tokens, int rows, int banks, double ms) {
    if (banks <= 0 || rows <= 0 || tokens < 0) return;
    lane_price_measure(lp, PULSAR_LANE_SPEC, banks, tokens, ms);
    const float tau = (float)tokens / (float)banks, rho = (float)rows / (float)banks;
    if (lp->tau <= 0.0f) {
        lp->tau = tau;
        lp->rho = rho;
        return;
    }
    lp->tau += LANE_PRICE_ALPHA * (tau - lp->tau);
    lp->rho += LANE_PRICE_ALPHA * (rho - lp->rho);
}

/* The lanes' predicted prices for `n` decoders, tokens/s (0: no number). */
static double lane_price_plain(const pulsar_lane_cost *c, int n) {
    const double ms = pulsar_lane_cost_ms(c, (double)n);
    return ms > 0.0 ? 1000.0 * (double)n / ms : 0.0;
}
static double lane_price_spec(const pulsar_lane_cost *c, const lane_price *lp, int n, uint32_t verify_rows) {
    if (!(lp->tau > 0.0f) || !(lp->rho >= 1.0f)) return 0.0;
    double rows = (double)n * lp->rho, toks = (double)n * lp->tau;
    if (rows > (double)verify_rows) {
        const double drafts = rows - (double)n, kept = (double)verify_rows - (double)n;
        toks = (double)n + (toks - (double)n) * (kept > 0.0 ? kept / drafts : 0.0);
        rows = (double)verify_rows;
    }
    const double ms = pulsar_lane_cost_ms(c, rows);
    return ms > 0.0 ? 1000.0 * toks / ms : 0.0;
}
/* Lane `li`'s measured price at `n` decoders, tokens/s (0: not measured there). */
static double lane_price_measured(const lane_price *lp, int li, int n) {
    if (lp->meas_n != n || lp->meas_steps[li] < LANE_PRICE_PROBE_STEPS || !(lp->meas_ms[li] > 0.0)) return 0.0;
    return 1000.0 * lp->meas_tok[li] / lp->meas_ms[li];
}

/* The priced lane (2 plain, 3 spec) for `n` decoders the spec lane could carry; called once a quantum.  The
 * fits' step counts (pulsar_lane_cost::n) time the hold and the probes.  Updates lp's prices and reason. */
static int lane_price_pick(lane_price *lp, const pulsar_lane_cost *plain, const pulsar_lane_cost *spec, int n,
                           uint32_t verify_rows) {
    if (lp->lane == 0) {   /* the spec lane until a price says otherwise */
        lp->lane = 3;
        lp->held_n0 = spec->n;
        lp->deferral = 0;
    }
    const double mp = lane_price_measured(lp, PULSAR_LANE_PLAIN, n);
    const double msp = lane_price_measured(lp, PULSAR_LANE_SPEC, n);
    lp->meas_plain = mp > 0.0;
    lp->meas_spec = msp > 0.0;
    lp->t_plain = lp->meas_plain ? mp : lane_price_plain(plain, n);
    lp->t_spec = lp->meas_spec ? msp : lane_price_spec(spec, lp, n, verify_rows);
    const int inc = lp->lane, other = inc == 3 ? 2 : 3;
    const double t_inc = inc == 3 ? lp->t_spec : lp->t_plain, t_other = other == 3 ? lp->t_spec : lp->t_plain;
    const bool other_measured = other == 3 ? lp->meas_spec : lp->meas_plain;
    const bool inc_measured = inc == 3 ? lp->meas_spec : lp->meas_plain;
    if (lp->probe) {
        const uint32_t steps = (lp->probe == 3 ? spec->n : plain->n) - lp->probe_n0;
        if (steps < LANE_PRICE_PROBE_STEPS || (!other_measured && steps < 4u * LANE_PRICE_PROBE_STEPS)) {
            lp->why = "measuring";
            return lp->probe;
        }
        lp->probe = 0;
        if (other_measured && (t_inc <= 0.0 || t_other > t_inc * (1.0 + LANE_PRICE_MARGIN))) {
            lp->lane = other;
            lp->why = "measured faster";
        } else {
            lp->why = "measured: the lane in force holds";
        }
        lp->held_n0 = lp->lane == 3 ? spec->n : plain->n;
        lp->deferral = 0;
        return lp->lane;
    }
    if (!inc_measured) {
        /* the lane in force is unmeasured at this N (it just moved): whichever lane is measured here first, the
         * better-predicted one is (one decoder left beside a plain batch of four went 32 plain steps before the
         * spec lane was probed); a misprediction costs the probe that corrects it */
        if (!other_measured && t_inc > 0.0 && t_other > t_inc * (1.0 + LANE_PRICE_MARGIN)) {
            lp->lane = other;
            lp->held_n0 = other == 3 ? spec->n : plain->n;
            lp->deferral = 0;
            lp->why = "priced faster: measuring";
            return other;
        }
        lp->why = "measuring the lane in force";
        return inc;
    }
    const uint32_t ran = (inc == 3 ? spec->n : plain->n) - lp->held_n0;
    if (ran < LANE_PRICE_HOLD_STEPS) {
        lp->why = "held";
        return inc;
    }
    const bool faster = t_other > t_inc * (1.0 + LANE_PRICE_MARGIN);
    if (faster && other_measured) {   /* both measured at this N: the switch needs no probe */
        lp->lane = other;
        lp->held_n0 = other == 3 ? spec->n : plain->n;
        lp->deferral = 0;
        lp->why = "measured faster";
        return other;
    }
    /* L289: the lane in force, measured, clears the other by LANE_PRICE_CLEAR -- the re-measure waits.  L288: the
     * other's price measured at this N or predicted by its fit (a new N drops the measurements, so with the decoder
     * count moving the other lane is mostly predicted, and the 512-step clock re-probed it inside every busy
     * stretch: a 32-step plain probe at four Qwen decoders, ~2 s at 96 tok/s against the spec lane's 165) */
    const bool clears = t_other > 0.0 && t_inc >= t_other * LANE_PRICE_CLEAR;
    const uint32_t reprobe = clears ? LANE_PRICE_REPROBE_STEPS * LANE_PRICE_CLEAR_SPAN : LANE_PRICE_REPROBE_STEPS;
    if (t_other <= 0.0 || faster || ran >= reprobe) {
        lp->probe = other;
        lp->probe_n0 = other == 3 ? spec->n : plain->n;
        lp->why = t_other <= 0.0 ? "no price here: measuring" : faster ? "priced faster: measuring" : "re-measuring";
        return other;
    }
    if (clears && ran >= LANE_PRICE_REPROBE_STEPS) {
        if (lp->deferral == 0) lp->deferral = 1;   /* the caller announces it once a weighing */
        lp->why = "clears the other lane: the re-measure waits";
        return inc;
    }
    lp->why = "priced";
    return inc;
}

static int server_pick_decode_lane(int pool_banks, bool spec_rounds, uint32_t verify_rows,
                                   session_slot *const *dec, int n_dec, int n_batched) {
    bool all_spec = spec_rounds && n_dec >= 1 && (uint32_t)n_dec <= verify_rows && n_batched == 0;
    for (int i = 0; all_spec && i < n_dec; i++) {
        const gen_state *dg = dec[i]->gen;
        if (!dg || !dg->spec_enabled || dg->batch_active)
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

bool server::bank_unspill(int bank, spilled_history *was) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    if (!pulsar_session_bank_alloc_physical(pool, (uint32_t)bank) ||
        !pulsar_session_bank_state_restore(pool, (uint32_t)bank)) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: bank %d could not be re-backed", bank);
        s->count_metric(&s->m_restore_failures);
        return false;
    }
    s->live_bank = bank;
    /* the host view is the bank's history as it was spilled (its carry); its rows are gone with the physical */
    const pulsar_tokens *hist = pulsar_session_tokens(pool);
    if (was && hist && hist->len > 0) {
        size_t len = 0;
        was->text = render_tokens_text(s->engine, hist, &len);
        /* L281: the bank's own image records are what its chain must carry back */
        const int n_img = pulsar_session_image_hashes(pool, was->hashes, 64);
        was->n_img = n_img < 64 ? n_img : 64;
    }
    pulsar_session_invalidate(pool);
    pulsar_session_bank_state_save(pool, (uint32_t)bank);
    s->slots[bank].spilled = false;
    s->slots[bank].committed_pos = 0;
    s->count_metric(&s->m_restores);
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: guard: spilled bank %d re-backed empty (its history %s)", bank,
               was && was->text ? "restores from its segment chain in slices" : "stays on disk");
    return true;
}

/* Spill one idle bank: install it, persist its history as segments (only what
 * the store lacks), save its host carry, repoint AWAY (free_physical refuses the
 * cur bank), then cudaFree its physical. */
bool server::spill_bank(session_slot *victim) {
    auto *s = this;
    pulsar_stall_mark("guard spill");
    pulsar_session *pool = s->sess;
    const uint32_t vb = victim->bank;
    /* The victim is never spilled (the pick excludes spilled banks), so this
     * never reloads, but the installing state restore can refuse; then the
     * persist below would write ANOTHER bank's history (L190 A4).  Abort. */
    /* L328 (2): the pool parks on another resident bank while vb's physical goes (bank 0 is a victim like any) */
    const int park = s->park_bank((int)vb);
    if (park < 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: bank %u is the only resident bank -- nowhere to park the "
                   "pool, spill refused", vb);
        return false;
    }
    if (!s->bank_switch((int)vb)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: guard: bank %u install failed (state restore refused); "
                   "spill aborted", vb);
        return false;
    }
    const double t0 = server_now_sec();
    s->kv_cache_persist(victim, "spill", true);
    pulsar_session_bank_state_save(pool, vb);         /* preserve host carry for restore */
    if (!pulsar_session_bank_state_restore(pool, (uint32_t)park)) return false;
    s->live_bank = park;
    /* Finding 4: free_physical returns false ONLY on a precondition refusal (nothing
     * freed, bank still live).  A true return means the bank IS evicted (slabs
     * freed), so mark it spilled unconditionally -- no half-evicted state. */
    if (!pulsar_session_bank_free_physical(pool, vb)) {
        /* free_physical refuses only the installed bank or one past the pool (and then frees nothing) */
        const int n_banks = pulsar_session_bank_count(pool);
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: guard: free_physical bank %u refused (%s) -- spill aborted, its KV stays resident",
                   vb, (int)vb == s->live_bank    ? "it is still the installed bank"
                       : vb >= (uint32_t)n_banks ? "the bank is past the pool"
                                                 : "the engine refused an idle bank; see its log line above");
        return false;
    }
    victim->spilled = true;
    victim->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    s->guard_evictions++;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: guard EVICTED bank %u -> segments (%.1f ms persist, physical freed); "
               "touched %.2f GiB / budget %.2f GiB, evictions %llu",
               vb, (server_now_sec() - t0) * 1e3,
               (double)pulsar_session_touched_kv_bytes(pool).raw() / gib,
               (double)s->guard_budget().raw() / gib,
               (unsigned long long)s->guard_evictions);
    return true;
}

/* ==== L314: preempt a running decoder through its terminal checkpoint, readmit it =====================
 *
 * After Flare (core-design.md, "Capacities and memory"): when a prompt cannot be admitted and no idle bank can be
 * spilled, the NEWEST running decoder publishes its terminal checkpoint and is readmitted later with its history and
 * its remaining budget.  Here the terminal checkpoint is the bank's frontier PAYLOAD at its consumed count (the
 * kv-state payload, every family's: every pool's rows, the frontier walk, the logits the next draw reads, the
 * draft-depth controller -- what the payload-frontier gate proves decodes on as the saved session would), written to
 * the disk KV dir, and the bank's physical is freed.  The job itself stays bound and parked: its gen_state keeps the
 * stream, the budget, the stop / grammar (L303) / penalty state and the sampler stream (L302), whose key is (seed,
 * tokens drawn) -- so a readmitted request draws exactly the tokens an uninterrupted one would (greedy and keyed
 * sampling alike; beside other decoders the batched lanes are nondeterministic by design, L118).  Readmission is
 * first-preempted first, as soon as the bank's resident KV fits the budget again.  An image request keeps its bank
 * (as Flare's: its image features are not reconstructed). */
static int guard_victim_order(server *s, session_slot **dec, int n, int *order, bool *dominated_out = nullptr);
static prompt_kv_verdict server_prompt_kv_verdict(KvBytes touched, KvBytes delta, KvBytes bound,
                                                  const KvBytes *drop, int n_victims, bool busy_may_free,
                                                  int *n_spill);

/* L326: the resident KV an admission decision starts from -- what is touched now plus what every admitted job will
 * still add before it decodes (want[i] = its bank's growth to its prompt's length, have[i] = what the bank holds;
 * a bank already past its prompt adds nothing).  Admitting a prompt touches none of its KV until its prefill runs,
 * so `touched` alone let L314's readmission retake, in the same worker turn, the room a preemption had just made
 * for that prompt (gpuq #472: PREEMPTED, READMITTED 0.0 s later, the guard over budget with no victim). */
static KvBytes server_kv_reserved(KvBytes touched, const KvBytes *want, const KvBytes *have, int n) {
    KvBytes r = touched;
    for (int i = 0; i < n; i++)
        if (want[i] > have[i]) r += want[i] - have[i];
    return r;
}

/* Does the slot's job still owe its bank the prompt's KV?  Bound with no gen yet, restoring a disk chain (L330: the
 * chain's rows not yet read land in the bank slice by slice, so its room is held from bind), or prefilling.  A parked
 * decoder holds nothing until its readmission (that decision's own subject); a decoder's bank holds its prompt -- and
 * its per-bank figure can lag the pool's while it decodes in a batch (gpuq #473: a decoding A read 0.02 GiB short of
 * its prompt, which kept the preemption B needed from fitting, and B ended up on bank 0). */
static bool server_slot_reserves(const session_slot *c) {
    if (!c->active_job || c->preempted) return false;
    return !c->gen || c->gen->phase == GEN_RESTORE || c->gen->phase == GEN_PREFILL_COLD ||
           c->gen->phase == GEN_PREFILL_MAIN;
}

/* L328 (2): where the pool parks while a bank's physical is freed (free_physical refuses the installed bank), pure:
 * bank 0 when it is resident and not the one freed -- what every guard move used before bank 0 could be spilled --
 * else the lowest other resident bank; -1 when the freed bank is the only resident one (the move is refused).  Any
 * resident bank serves: parking installs its views and carry and nothing reads it until the next switch, and
 * DeepSeek's batched launches name a resident bank's slab as their nominal operand
 * (gpu_graph_bank_attn_comp_pool), reading every bank through the base-pointer table. */
static int server_park_pick(const bool *resident, int n, int except) {
    if (n > 0 && resident[0] && except != 0) return 0;
    for (int b = 1; b < n; b++)
        if (resident[b] && b != except) return b;
    return -1;
}

/* L328 (2): does preempting a decoder of `freed` bytes make a prompt adding `need` fit, pure -- beside the reserved
 * base, once the idle banks (`idle`, their bytes summed) are spilled too: the re-check after the preemption spills
 * them (PROMPT_KV_SPILL).  Before bank 0 could be spilled, an idle bank 0 holding the warmup's high-water was no
 * victim, and this fit ignored every idle bank (gpuq #474: 0.12 GiB resident against W + A's 0.10). */
static bool server_preempt_fits(KvBytes reserved, KvBytes freed, KvBytes idle, KvBytes need, KvBytes budget) {
    const KvBytes gone = freed + idle;
    return (reserved > gone ? reserved - gone : KvBytes{}) + need <= budget;
}

/* L328 (5): what a readmission must fit beside the reserved base: the decoder's KV and one decode quantum's growth of
 * every decoder that steps once it is back (n_decoding running + itself).  Without the growth a decoder the guard
 * preempted because the decoders' next quantum did not fit was readmitted at once -- its KV alone fits -- and the
 * guard preempted it again the next quantum. */
static KvBytes server_readmit_need(KvBytes bytes, int n_decoding, KvBytes growth) {
    return bytes + growth.times((uint64_t)(n_decoding + 1));
}

/* L328 (5): the decode guard preempts while that makes progress: beside another decoder, or beside a job that holds
 * reserved room (a prompt prefilling or a chain restoring, whose end frees it).  A lone decoder with nothing bound
 * beside it would only come straight back -- nothing it waits on ever ends -- so it proceeds (back-pressure, the
 * MemAvailable floor the backstop). */
static bool server_guard_may_preempt(int n_decoding, bool reserving) {
    return n_decoding > 1 || reserving;
}

int server::park_bank(int except) {
    auto *s = this;
    bool resident[PULSAR_SESSION_POOL_CAP];
    const int n = s->pool_banks < PULSAR_SESSION_POOL_CAP ? s->pool_banks : PULSAR_SESSION_POOL_CAP;
    for (int b = 0; b < n; b++) resident[b] = !pulsar_session_bank_is_evicted(s->sess, (uint32_t)b);
    return server_park_pick(resident, n, except);
}

KvBytes server::kv_reserved_bytes(const session_slot *except) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    KvBytes want[PULSAR_SESSION_POOL_CAP], have[PULSAR_SESSION_POOL_CAP];
    int n = 0;
    for (int i = 0; i < s->n_slots && n < PULSAR_SESSION_POOL_CAP; i++) {
        const session_slot *c = &s->slots[i];
        if (c == except || !server_slot_reserves(c)) continue;
        const int len = c->active_job->req.prompt.len > 0 ? c->active_job->req.prompt.len : 0;
        want[n] = pulsar_session_quantum_growth_bytes_per_bank(pool, KvRows{(uint32_t)len});
        have[n] = pulsar_session_bank_touched_kv_bytes(pool, c->bank);
        n++;
    }
    return server_kv_reserved(pulsar_session_touched_kv_bytes(pool), want, have, n);
}

bool server::preempt_newest_decoder(const session_slot *keep, KvBytes need) {
    auto *s = this;
    if (s->pool_banks <= 1 || !s->kv.enabled || !s->kv.dir || pulsar_engine_is_tp(s->engine)) return false;
    bool tried[PULSAR_SESSION_POOL_CAP] = {};
    /* L328 (2): the idle banks the re-check after a preemption spills -- bank 0 among them -- are room too */
    KvBytes idle{};
    {
        int order[PULSAR_SESSION_POOL_CAP];
        session_slot *self[1] = {(session_slot *)keep};
        const int n_victims = guard_victim_order(s, self, keep ? 1 : 0, order);
        for (int k = 0; k < n_victims; k++) idle += pulsar_session_bank_touched_kv_bytes(s->sess, s->slots[order[k]].bank);
    }
    for (;;) {
        /* the newest decoder not yet tried, never an image request's */
        session_slot *v = NULL;
        for (int i = 0; i < s->n_slots; i++) {
            session_slot *c = &s->slots[i];
            if (c == keep || tried[i] || !slot_is_batchable_decode(c) || c->spilled || c->active_job->req.n_images > 0)
                continue;
            /* a preemption that does not make the prompt fit only churns (the prompt waits either way) */
            const KvBytes freed = pulsar_session_bank_touched_kv_bytes(s->sess, c->bank);
            /* L326: the admitted jobs' growth included */
            if (!server_preempt_fits(s->kv_reserved_bytes(keep), freed, idle, need, s->guard_budget())) continue;
            if (!v || c->gen->t0 > v->gen->t0) v = c;
        }
        if (!v) return false;
        tried[v - s->slots] = true;
        gen_state *g = v->gen;
        if (g->batch_active && !s->batch_leave(v)) continue;   /* the plain batch's sampled token is committed first */
        const int park = s->park_bank((int)v->bank);   /* L328 (2): where the pool parks while v's physical goes */
        if (park < 0 || g->phase != GEN_DECODE || !s->bank_switch((int)v->bank)) continue;
        pulsar_session *pool = s->sess;
        const double t0 = server_now_sec();
        const KvBytes bytes = pulsar_session_bank_touched_kv_bytes(pool, v->bank);
        const uint64_t payload_bytes = pulsar_session_payload_bytes(pool);   /* the installed victim's */
        char name[64];
        snprintf(name, sizeof name, "preempt-bank-%u.kvp", v->bank);
        char *path = path_join(s->kv.dir, name);
        char err[256] = "";
        /* the payload refuses a bank that is no exact frontier (a drafter mid-draft between rounds): that decoder keeps
         * its bank and the next newest is tried */
        FILE *fp = fopen(path, "wb");
        bool ok = fp && pulsar_session_save_payload(pool, fp, err, sizeof err) == 0;
        if (fp) ok = fclose(fp) == 0 && ok;
        if (ok) {
            pulsar_session_bank_state_save(pool, v->bank);
            ok = pulsar_session_bank_state_restore(pool, (uint32_t)park);   /* free_physical refuses the installed bank */
            if (ok) s->live_bank = park;
        }
        ok = ok && pulsar_session_bank_free_physical(pool, v->bank);
        if (!ok) {
            server_log(PULSAR_LOG_WARNING, "pulsar-server: preemption of bank %u declined (%s); it keeps decoding",
                       v->bank, err[0] ? err : "its payload could not be written or its physical freed");
            unlink(path);
            free(path);
            continue;
        }
        v->preempted = true;
        v->preempt_bytes = bytes;
        v->preempt_t = t0;
        v->preempt_path = path;
        s->count_metric(&s->m_preemptions);
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: PREEMPTED bank %u's decoder at %d tokens (%d generated): terminal checkpoint "
                   "%.1f MiB in %.1f ms, physical freed (%.2f GiB)", v->bank, pulsar_session_bank_pos(pool, v->bank),
                   g->completion_total + g->completion, (double)payload_bytes / 1048576.0,
                   (server_now_sec() - t0) * 1e3, (double)bytes.raw() / (1024.0 * 1024.0 * 1024.0));
        return true;
    }
}

bool server::worker_readmit_preempted() {
    auto *s = this;
    session_slot *v = NULL;
    for (int i = 0; i < s->n_slots; i++)
        if (s->slots[i].preempted && (!v || s->slots[i].preempt_t < v->preempt_t)) v = &s->slots[i];
    if (!v) return false;
    pulsar_session *pool = s->sess;
    if (s->guard_enabled) {
        /* the admission arithmetic over the idle banks, as a prompt's (L280): fits, or fits once idle banks spill --
         * from the same reserved base (L326: an admitted prompt's coming prefill counts, so the room a preemption
         * made for it is not retaken before it lands; the decoder stays parked until a job ends and frees it) */
        int order[PULSAR_SESSION_POOL_CAP], n_spill = 0;
        KvBytes drop[PULSAR_SESSION_POOL_CAP];
        const int n_victims = guard_victim_order(s, NULL, 0, order);
        for (int k = 0; k < n_victims; k++) drop[k] = pulsar_session_bank_touched_kv_bytes(pool, s->slots[order[k]].bank);
        bool busy = false;   /* another job whose end may make room */
        int n_decoding = 0;
        for (int i = 0; i < s->n_slots; i++) {
            busy = busy || (&s->slots[i] != v && s->slots[i].active_job && !s->slots[i].preempted);
            if (slot_is_batchable_decode(&s->slots[i])) n_decoding++;
        }
        /* L328 (5): its KV and the next quantum's growth of every decoder stepping once it is back */
        const KvBytes want = server_readmit_need(
                v->preempt_bytes, n_decoding,
                pulsar_session_quantum_growth_bytes_per_bank(pool, KvRows{PULSAR_SERVER_DECODE_QUANTUM_TOKENS}));
        const prompt_kv_verdict pv = server_prompt_kv_verdict(s->kv_reserved_bytes(v), want,
                                                              s->guard_budget(), drop, n_victims, busy, &n_spill);
        if (pv == PROMPT_KV_WAIT) return false;
        if (pv == PROMPT_KV_REFUSE) {
            /* nothing running will make room and no spill does: it can never come back -- ended by name (Flare: a
             * request whose history exceeds the whole pool ends) */
            snprintf(v->gen->err, sizeof v->gen->err,
                     "a preempted decode no longer fits the KV budget (%.2f GiB beside %.2f GiB resident, budget %.2f "
                     "GiB) and nothing running can make room",
                     (double)v->preempt_bytes.raw() / (1024.0 * 1024.0 * 1024.0),
                     (double)pulsar_session_touched_kv_bytes(pool).raw() / (1024.0 * 1024.0 * 1024.0),
                     (double)s->guard_budget().raw() / (1024.0 * 1024.0 * 1024.0));
            v->gen->finish = "error";
            v->gen->phase = GEN_FINISH;
            server_log(PULSAR_LOG_WARNING, "pulsar-server: bank %u: %s", v->bank, v->gen->err);
            unlink(v->preempt_path);
            free(v->preempt_path);
            v->preempt_path = NULL;
            v->preempted = false;
            /* an empty bank again, its history gone with it (never a view over rows it no longer holds) */
            if (pulsar_session_bank_alloc_physical(pool, v->bank) && pulsar_session_bank_state_restore(pool, v->bank)) {
                s->live_bank = (int)v->bank;
                pulsar_session_invalidate(pool);
                pulsar_session_bank_state_save(pool, v->bank);
            }
            return true;
        }
        for (int k = 0; k < n_spill; k++) {
            if (!s->spill_bank(&s->slots[order[k]])) return false;
            s->count_metric(&s->m_spills);
        }
        if (s->kv_reserved_bytes(v) + want > s->guard_budget()) return false;
    }
    const double t0 = server_now_sec();
    char err[256] = "";
    bool ok = pulsar_session_bank_alloc_physical(pool, v->bank) && pulsar_session_bank_state_restore(pool, v->bank);
    const bool installed = ok;
    if (ok) s->live_bank = (int)v->bank;
    FILE *fp = ok ? fopen(v->preempt_path, "rb") : NULL;
    if (fp) {
        fseek(fp, 0, SEEK_END);
        const long n = ftell(fp);
        rewind(fp);
        ok = n > 0 && pulsar_session_load_payload(pool, fp, (uint64_t)n, err, sizeof err) == 0;
        fclose(fp);
    } else {
        ok = false;
    }
    gen_state *g = v->gen;
    if (ok) {
        pulsar_session_bank_state_save(pool, v->bank);
        v->committed_pos = pulsar_session_pos(pool);
        s->count_metric(&s->m_readmissions);
        /* L312's age clock restarts here: the parked time was the preemption's, not a wait a round causes, and an
         * age credit for it would hold back the prompt the decoder was preempted for */
        g->sched_out_t = server_now_sec();
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: READMITTED bank %u's decoder at %d tokens after %.1f s (%.1f ms "
                   "reload)", v->bank, pulsar_session_pos(pool), t0 - v->preempt_t, (server_now_sec() - t0) * 1e3);
    } else {
        snprintf(g->err, sizeof g->err, "a preempted decode could not be reloaded: %s",
                 err[0] ? err : "its bank could not be re-backed or its payload read");
        g->finish = "error";
        g->phase = GEN_FINISH;
        server_log(PULSAR_LOG_WARNING, "pulsar-server: bank %u: %s", v->bank, g->err);
        if (installed) pulsar_session_invalidate(pool);   /* only the bank's own state, never another's */
    }
    unlink(v->preempt_path);
    free(v->preempt_path);
    v->preempt_path = NULL;
    v->preempted = false;
    return true;
}

/* LRU-idle smallest-frontier victim: NOT in the live decode set, no active job,
 * not already spilled (L328: bank 0 is a victim like any other -- the pool parks
 * on another resident bank, server::park_bank). -1 if none.  L323: a dominated
 * bank (server::dominated_slots; `dominated` may be NULL) goes before any LRU
 * choice. */
int server::guard_pick_victim(session_slot **dec, int n, const bool *dominated) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    int best = -1;
    bool best_dom = false;
    uint64_t best_us = UINT64_MAX;
    KvBytes best_touched = KvBytes::max();
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->spilled || sl->active_job) continue;
        bool live = false;
        for (int k = 0; k < n; k++) if (dec[k] == sl) { live = true; break; }
        if (live) continue;
        const KvBytes t = pulsar_session_bank_touched_kv_bytes(pool, sl->bank);
        const bool dom = dominated && dominated[i];
        if (best < 0 || (dom && !best_dom) ||
            (dom == best_dom && (sl->last_serviced_us < best_us ||
                                 (sl->last_serviced_us == best_us && t < best_touched)))) {
            best = i; best_dom = dom; best_us = sl->last_serviced_us; best_touched = t;
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
static int guard_spill_plan(KvBytes touched, KvBytes delta, KvBytes bound,
                            const KvBytes *per_victim_drop, int n_victims) {
    int k = 0;
    while (touched + delta > bound && k < n_victims) {
        const KvBytes drop = per_victim_drop[k++];
        touched = drop >= touched ? KvBytes{} : touched - drop;
    }
    return k;
}

/* Every guard victim in pick order: guard_pick_victim is the ONE authority
 * for the filter and the LRU key; the slots already ordered are appended to
 * its live-exclusion set so each call yields the next. At most n_slots
 * entries. */
static int guard_victim_order(server *s, session_slot **dec, int n, int *order, bool *dominated_out) {
    session_slot *excl[2 * PULSAR_SESSION_POOL_CAP];
    for (int i = 0; i < n; i++) excl[i] = dec[i];
    /* L323: the dominance facts once for the whole order -- a spill leaves every bank's history (the host carry)
     * as it was, so they hold across it */
    bool dominated[PULSAR_SESSION_POOL_CAP];
    s->dominated_slots(dominated);
    int m = 0;
    while (n + m < 2 * PULSAR_SESSION_POOL_CAP) {
        const int vi = s->guard_pick_victim(excl, n + m, dominated);
        if (vi < 0) break;
        order[m] = vi;
        excl[n + m] = &s->slots[vi];
        m++;
    }
    if (dominated_out) memcpy(dominated_out, dominated, sizeof dominated);
    return m;
}

/* L280: the prompt's KV admission, as arithmetic.  touched = the pool's resident KV now, delta = what binding the
 * job adds (its bank's growth to the prompt's length), bound = the guard's budget, drop[] = the idle victims'
 * resident bytes in spill order.  FITS when touched + delta <= bound; SPILL (*n_spill victims, the minimum,
 * guard_spill_plan's law) when spilling makes it fit; else WAIT when a busy bank will become a victim once its job
 * ends (busy_may_free), else REFUSE.  A prompt whose growth alone is past the bound is REFUSE at once: no spill or
 * wait ever makes it fit.  Spilling is never planned when it cannot make the prompt fit (a spill that does not
 * clear the breach only churns the disk). */
static prompt_kv_verdict server_prompt_kv_verdict(KvBytes touched, KvBytes delta, KvBytes bound,
                                                  const KvBytes *drop, int n_victims, bool busy_may_free,
                                                  int *n_spill) {
    *n_spill = 0;
    /* a job that adds nothing (a continuation on its own resident bank) never waits on a breach it did not cause */
    if (delta == KvBytes{} || touched + delta <= bound) return PROMPT_KV_FITS;
    if (delta > bound) return PROMPT_KV_REFUSE;
    const int k = guard_spill_plan(touched, delta, bound, drop, n_victims);
    KvBytes t = touched;
    for (int i = 0; i < k; i++) t = drop[i] >= t ? KvBytes{} : t - drop[i];
    if (k > 0 && t + delta <= bound) {
        *n_spill = k;
        return PROMPT_KV_SPILL;
    }
    return busy_may_free ? PROMPT_KV_WAIT : PROMPT_KV_REFUSE;
}

prompt_kv_verdict server::prompt_kv_check(const session_slot *sl, const job *j, int *order, int *n_spill,
                                          KvBytes *need) {
    auto *s = this;
    *n_spill = 0;
    *need = KvBytes{};
    if (!s->guard_enabled || s->pool_banks <= 0) return PROMPT_KV_FITS;
    pulsar_session *pool = s->sess;
    const int len = j->req.prompt.len > 0 ? j->req.prompt.len : 0;
    const KvBytes want = pulsar_session_quantum_growth_bytes_per_bank(pool, KvRows{(uint32_t)len});
    /* NULL: an empty bank */
    const KvBytes have = sl ? pulsar_session_bank_touched_kv_bytes(pool, sl->bank) : KvBytes{};
    *need = want > have ? want - have : KvBytes{};
    /* the victims: every idle bank but the job's own (guard_pick_victim's filter: never a bound or spilled bank) */
    session_slot *self[1] = {(session_slot *)sl};
    const int n_victims = guard_victim_order(s, self, sl ? 1 : 0, order);
    KvBytes drop[PULSAR_SESSION_POOL_CAP];
    for (int k = 0; k < n_victims; k++) drop[k] = pulsar_session_bank_touched_kv_bytes(pool, s->slots[order[k]].bank);
    /* a bank bound to a job now is a victim once that job ends */
    bool busy = false;
    for (int i = 0; i < s->n_slots && !busy; i++)
        busy = &s->slots[i] != sl && s->slots[i].provisioned && !s->slots[i].spilled && s->slots[i].active_job;
    /* L326: from the reserved base -- prompts bound earlier in this worker turn have touched nothing yet */
    return server_prompt_kv_verdict(s->kv_reserved_bytes(sl), *need, s->guard_budget(), drop, n_victims, busy,
                                    n_spill);
}

bool server::prompt_kv_admit(session_slot *sl, job *j, bool *refused) {
    auto *s = this;
    *refused = false;
    int order[PULSAR_SESSION_POOL_CAP];
    int n_spill = 0;
    KvBytes need{};
    prompt_kv_verdict v = s->prompt_kv_check(sl, j, order, &n_spill, &need);
    /* L314: no idle bank can make room: the newest running decoders are preempted through their terminal checkpoints,
     * one at a time, while that can (a prompt whose growth alone is past the budget never fits) */
    while ((v == PROMPT_KV_WAIT || v == PROMPT_KV_REFUSE) && need <= s->guard_budget() &&
           s->preempt_newest_decoder(sl, need))
        v = s->prompt_kv_check(sl, j, order, &n_spill, &need);
    if (v == PROMPT_KV_FITS) return true;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const KvBytes touched = s->kv_reserved_bytes(sl);   /* the base the verdict read (L326) */
    if (v == PROMPT_KV_SPILL) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: kv admission: a %d-token prompt adds %.2f GiB of KV to bank %u (resident %.2f + "
                   "%.2f > budget %.2f GiB): spilling %d idle bank(s) first",
                   j->req.prompt.len, (double)need.raw() / gib, sl->bank, (double)touched.raw() / gib,
                   (double)need.raw() / gib, (double)s->guard_budget().raw() / gib, n_spill);
        for (int k = 0; k < n_spill; k++) {
            if (!s->spill_bank(&s->slots[order[k]])) break;
            s->count_metric(&s->m_spills);
        }
        /* a spill that failed leaves the breach: decide again on what is resident now */
        KvBytes again{};
        int n2 = 0;
        const prompt_kv_verdict v2 = s->prompt_kv_check(sl, j, order, &n2, &again);
        if (v2 == PROMPT_KV_FITS) return true;
        if (v2 != PROMPT_KV_REFUSE) {
            s->note_provision_refusal(j, PROVISION_REFUSED_KV_TOUCHED);
            return false;
        }
    } else if (v == PROMPT_KV_WAIT) {
        if (j->refusal_counted != PROVISION_REFUSED_KV_TOUCHED)   /* once per job (rule 9): the worker retries it */
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: kv admission: a %d-token prompt adds %.2f GiB of KV to bank %u (resident %.2f + "
                       "%.2f > budget %.2f GiB) and no idle bank can be spilled: queued until a busy bank's job ends",
                       j->req.prompt.len, (double)need.raw() / gib, sl->bank, (double)touched.raw() / gib,
                       (double)need.raw() / gib, (double)s->guard_budget().raw() / gib);
        s->note_provision_refusal(j, PROVISION_REFUSED_KV_TOUCHED);
        return false;
    }
    char msg[512];
    snprintf(msg, sizeof msg,
             "not enough KV memory for this prompt now: its %d tokens need %.2f GiB of KV beside %.2f GiB the server "
             "keeps resident (budget %.2f GiB), and no idle conversation can be moved to disk to make room; retry "
             "later or shorten the prompt",
             j->req.prompt.len, (double)need.raw() / gib, (double)touched.raw() / gib,
             (double)s->guard_budget().raw() / gib);
    server_log(PULSAR_LOG_WARNING, "pulsar-server: kv admission REFUSED (503): %s", msg);
    s->note_provision_refusal(j, PROVISION_REFUSED_KV_TOUCHED);
    *refused = true;
    if (j->req.api == API_ANTHROPIC) http_error_anthropic(j->fd, 503, msg);
    else http_error_retry(j->fd, 503, msg, HTTP_RETRY_GOING_AWAY_S);
    return false;
}

/* ==== 2026-10-29: the per-chunk memory re-check of a prefill =================================================
 * Each prefill quantum (one chunk) starts only when MemAvailable covers that chunk's new KV over the floor.  The
 * admission verdict (L280) priced the whole prompt when the job bound; this re-check is against what the box holds
 * NOW, because memory nobody accounts for goes meanwhile.  Short of it, the guard's own remedies run first, while
 * there is something to free -- spill the least-recently-used idle bank, else preempt the newest decoder (L314) --
 * and only then is the request refused (503, retry): never proceed.  Nothing is invalidated on a refusal: the bank
 * keeps every chunk it computed.  A fused / mixed rider is not refused there (a remedy would move banks under the
 * step being assembled): it leaves the fused lane, and its classic quantum runs this whole rule. */
const char *const prefill_mem_step_names[PREFILL_MEM_STEPS] = { "go", "spill", "preempt", "refuse" };

prefill_mem_step server_prefill_mem_step(KvBytes avail, KvBytes need, KvBytes floor, int n_idle_victims,
                                         bool preemptible) {
    if (server_headroom_admits(avail, need, floor)) return PREFILL_MEM_GO;
    if (n_idle_victims > 0) return PREFILL_MEM_SPILL;
    if (preemptible) return PREFILL_MEM_PREEMPT;
    return PREFILL_MEM_REFUSE;
}

KvBytes server::prefill_chunk_need() {
    return pulsar_session_quantum_growth_bytes_per_bank(sess, KvRows{(uint32_t)pulsar_session_prefill_cap(sess)});
}

bool server::prefill_headroom_ok(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    const KvBytes need = s->prefill_chunk_need(), floor = server_mem_floor();
    const double gib = 1024.0 * 1024.0 * 1024.0;
    KvBytes avail{server_mem_available_bytes()};
    bool preempt_tried = false;
    const char *why = "nothing left to free";   /* the refusal says which: a REFUSE verdict, or a remedy that failed */
    int spilled = 0;
    for (int round = 0; round <= PULSAR_SESSION_POOL_CAP; round++) {
        int order[PULSAR_SESSION_POOL_CAP], useful[PULSAR_SESSION_POOL_CAP];
        session_slot *self[1] = {sl};
        const int n_order = s->guard_enabled && s->pool_banks > 0 ? guard_victim_order(s, self, 1, order) : 0;
        int n_idle = 0;   /* in the guard's pick order, the idle banks whose spill frees something */
        for (int k = 0; k < n_order; k++)
            if (pulsar_session_bank_touched_kv_bytes(s->sess, s->slots[order[k]].bank) > KvBytes{})
                useful[n_idle++] = order[k];
        bool decoders = false;
        for (int i = 0; i < s->n_slots && !preempt_tried; i++)
            decoders |= &s->slots[i] != sl && slot_is_batchable_decode(&s->slots[i]);
        const prefill_mem_step step = server_prefill_mem_step(avail, need, floor, n_idle, decoders && !preempt_tried);
        if (step == PREFILL_MEM_GO) return true;
        if (step == PREFILL_MEM_REFUSE) break;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: prefill on bank %u: MemAvailable %.2f GiB does not cover the next chunk's %.2f GiB "
                   "over the %.2f GiB floor -- %s first",
                   sl->bank, (double)avail.raw() / gib, (double)need.raw() / gib, (double)floor.raw() / gib,
                   step == PREFILL_MEM_SPILL ? "spilling the least-recently-used idle bank"
                                             : "preempting the newest decoder");
        if (step == PREFILL_MEM_SPILL) {
            if (!s->spill_bank(&s->slots[useful[0]])) { why = "a spill failed (its line above)"; break; }
            s->count_metric(&s->m_spills);
            spilled++;
        } else {
            preempt_tried = true;
            if (!s->preempt_newest_decoder(sl, need)) continue;   /* none could go: the next round refuses */
        }
        if (!s->bank_switch(sl->bank)) { why = "the bank could not be reinstalled after a remedy"; break; }
        avail = KvBytes{server_mem_available_bytes()};
    }
    s->count_metric(&s->m_prefill_mem_refusals);
    snprintf(g->err, sizeof g->err,
             "not enough memory to continue this prompt now (MemAvailable %.2f GiB, next chunk %.2f GiB, floor "
             "%.2f GiB); the part computed is kept, retry later",
             (double)avail.raw() / gib, (double)need.raw() / gib, (double)floor.raw() / gib);
    unsigned skipped = 0;
    if (warn_limiter_due(&s->prefill_mem_warn, server_now_sec(), PULSAR_SERVER_MEM_FLOOR_WARN_SEC, &skipped))
        server_log(PULSAR_LOG_WARNING, "pulsar-server: prefill REFUSED (503) on bank %u at %d of %d tokens -- %s after "
                   "%d spill(s)%s: %s (%u more refused since the last line)", sl->bank,
                   pulsar_session_bank_pos(s->sess, (uint32_t)sl->bank), g->prompt_tokens, why, spilled,
                   preempt_tried ? " and a preemption try" : "", g->err, skipped);
    pulsar_session_set_cancel(s->sess, NULL, NULL);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    s->trace_event(g->trace_id, "prefill refused: %s", g->err);
    if (g->j->req.stream && g->progress.headers_sent) {
        if (!g->progress.stream_failed) (void)sse_error_event(g->j->fd, &g->j->req, g->err);
    } else if (g->j->req.api == API_ANTHROPIC) {
        http_error_anthropic(g->j->fd, 503, g->err);
    } else {
        http_error_retry(g->j->fd, 503, g->err, HTTP_RETRY_GOING_AWAY_S);
    }
    g->phase = GEN_DONE;
    return false;
}

bool server::prefill_headroom_fused_ok(session_slot *sl) {
    if (server_headroom_admits(KvBytes{server_mem_available_bytes()}, prefill_chunk_need(), server_mem_floor()))
        return true;
    sl->gen->no_fuse = "the memory floor (its classic quantum re-checks it, with the guard's remedies)";
    return false;
}

int server::guard_maybe_evict(session_slot *const *dec, int n, session_slot **kept) {
    auto *s = this;
    for (int i = 0; i < n; i++) kept[i] = dec[i];
    if (!s->guard_enabled || s->pool_banks <= 0 || n <= 0) return n;
    pulsar_session *pool = s->sess;
    const KvBytes dpb = pulsar_session_quantum_growth_bytes_per_bank(
            pool, KvRows{(uint32_t)PULSAR_SERVER_DECODE_QUANTUM_TOKENS});
    const KvBytes bound = s->guard_budget();
    /* L328: from the reserved base, as every admission decision -- a prompt admitted (or a chain restoring) beside
     * these decoders holds its room, and their growth must not take it (a prefill and a restore are not guarded:
     * their admission is) */
    const KvBytes base = s->kv_reserved_bytes(NULL);
    /* No breach, no plan: the common quantum costs one walk of the slots. */
    if (base + dpb.times((uint64_t)n) <= bound) return n;
    /* Finding 2: free_physical zeroes a spilled bank's frontier, so a spill
     * drops touched by exactly that bank's touched bytes (the pool gauge IS
     * the per-bank sum, pulsar_session_touched_kv_bytes). guard_spill_plan walks
     * that arithmetic over the LRU victim order and evicts exactly the minimum
     * (usually ONE) per breach, NOT the whole idle set (the cascade bug). The
     * per-quantum count log lets the smoke assert no cascade. */
    int order[PULSAR_SESSION_POOL_CAP];
    KvBytes drop[PULSAR_SESSION_POOL_CAP];
    bool dominated[PULSAR_SESSION_POOL_CAP];
    const int n_victims = guard_victim_order(s, kept, n, order, dominated);
    for (int k = 0; k < n_victims; k++)
        drop[k] = pulsar_session_bank_touched_kv_bytes(pool, s->slots[order[k]].bank);
    const int plan = guard_spill_plan(base, dpb.times((uint64_t)n), bound, drop, n_victims);
    int spilled_this_quantum = 0;
    for (int k = 0; k < plan; k++) {
        if (dominated[order[k]])
            server_log(PULSAR_LOG_KVCACHE, "pulsar-server: guard: bank %u is dominated (another bank holds its whole "
                                           "history) -- spilled first", s->slots[order[k]].bank);
        if (!s->spill_bank(&s->slots[order[k]])) break;   /* spill failed — stop */
        spilled_this_quantum++;
        s->count_metric(&s->m_spills);
    }
    if (spilled_this_quantum == n_victims) {
        /* L328 (5): every idle victim is spilled (or there was none) and the breach stands: the newest decoders are
         * preempted through their terminal checkpoints (L314), one at a time while one makes the others' growth fit
         * -- where that makes progress (server_guard_may_preempt).  Before, the guard proceeded past its budget here whenever every bank was live,
         * the MemAvailable floor the only backstop (L319 item 2; gpuq #472's "NO idle victim -- back-pressure"). */
        bool reserving = false;
        for (int i = 0; i < s->n_slots && !reserving; i++) reserving = server_slot_reserves(&s->slots[i]);
        while (s->kv_reserved_bytes(NULL) + dpb.times((uint64_t)n) > bound && server_guard_may_preempt(n, reserving) &&
               s->preempt_newest_decoder(NULL, n > 0 ? dpb.times((uint64_t)(n - 1)) : KvBytes{})) {
            int m = 0;
            for (int i = 0; i < n; i++)
                if (slot_is_batchable_decode(kept[i])) kept[m++] = kept[i];
            if (m == n) break;   /* the preempted decoder was not in this quantum's set: nothing here moved */
            n = m;
        }
        /* A candidate's batch_leave commits the plain batch's sampled token, and that token can END its decode (a stop,
         * the budget, a failed step: GEN_FINISH) -- then it was not preempted, and whether or not another was, it must
         * not ride this quantum: the ENTRY loops read batch_feed_valid, not the phase, and would decode past its end.
         * The worker's post-quantum pass finishes it. */
        {
            int m = 0;
            for (int i = 0; i < n; i++)
                if (slot_is_batchable_decode(kept[i])) kept[m++] = kept[i];
            n = m;
        }
        /* What still stands is back-pressure: proceed and let the live floor guard (a lone decoder with nothing
         * bound beside it, or no preemption that makes the rest fit). */
        const KvBytes projected = s->kv_reserved_bytes(NULL) + dpb.times((uint64_t)n);
        static uint64_t last_warn_us;
        const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
        if (projected > bound && now_us - last_warn_us > 5000000ull) {
            last_warn_us = now_us;
            const double gib = 1024.0 * 1024.0 * 1024.0;
            server_log(PULSAR_LOG_WARNING,
                "pulsar-server: guard: projected touched %.2f GiB > budget %.2f GiB but NO "
                "idle victim and no preemption makes room — back-pressure (MemAvailable floor is the backstop)",
                (double)projected.raw() / gib, (double)bound.raw() / gib);
        }
    }
    if (spilled_this_quantum > 0) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: guard: spilled %d bank(s) this quantum (touched now %.2f GiB / %.2f GiB)",
                   spilled_this_quantum,
                   (double)pulsar_session_touched_kv_bytes(pool).raw() / (1024.0*1024.0*1024.0),
                   (double)bound.raw() / (1024.0*1024.0*1024.0));
    }
    return n;
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

/* L293: the request's token budget is spent (or the server is stopping).  The ONE check every site that commits a
 * decode token reads first: the plain and mixed lanes' steps, the spec lane's round assembly (whose round_begin then
 * caps the accepted run at the remaining budget), and a decoder leaving the plain batch for the spec lane.  The leave
 * committed the batch's sampled token without it: a plain quantum whose last step emitted the budget's last token
 * left the slot in GEN_DECODE with the next token sampled, and a switch to the spec lane emitted it -- 65 for 64. */
static bool gen_budget_spent(const gen_state *g) {
    return g_stop_requested || g->completion >= g->max_tokens;
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

/* The plain and mixed batched lanes' step epilogue: every decode row's committed token emitted, then the rows of
 * the slots that go on drawn in ONE device launch (THE sampler over the step's headed rows: row q is dec[live_idx[q]]'s,
 * each at its own stream's key, a live grammar region through its mask -- L302, L303), and each draw's logprobs captured
 * from the same row.  A slot that stopped draws nothing (its stream is not stepped). */
static void lane_emit_and_draw(server *s, session_slot **dec, const int *live_idx, int m, pulsar_session *pool) {
    uint32_t rows[PULSAR_SESSION_POOL_CAP];
    pulsar_row_draw d[PULSAR_SESSION_POOL_CAP];
    std::vector<uint32_t> masks[PULSAR_SESSION_POOL_CAP];
    gen_state *who[PULSAR_SESSION_POOL_CAP];
    uint32_t nd = 0;
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
        float temp, top_p, min_p; int top_k;
        gen_resolve_sampling_decode(g, &temp, &top_k, &top_p, &min_p);
        if (!gen_draw_req(s, g, temp, top_k, top_p, min_p, &masks[nd], &d[nd])) {   /* L303: g->err says why */
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        rows[nd] = (uint32_t)q;
        who[nd++] = g;
    }
    if (nd == 0) return;
    int tok[PULSAR_SESSION_POOL_CAP];
    if (pulsar_session_draw_rows(pool, rows, d, nd, tok) != 0)
        for (uint32_t k = 0; k < nd; k++) tok[k] = PULSAR_SAMPLE_REFUSED;
    for (uint32_t k = 0; k < nd; k++) {
        gen_state *g = who[k];
        g->batch_feed_token = tok[k];
        if (tok[k] < 0) {
            snprintf(g->err, sizeof g->err, "sampler refused a degenerate logits row (L188)");
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        /* This bank's row IS the target distribution at batch_feed_pos, and
         * it is overwritten by the next step's sweep. */
        logprob_capture_row(&g->logprobs, pool, rows[k], tok[k]);
    }
}

void server::worker_batched_decode_quantum(session_slot **dec, int n, int quantum_tokens) {
    PULSAR_NVTX_FN();
    auto *s = this;
    if (n <= 0) return;
    pulsar_session *pool = s->sess;
    park_live_bank(s, dec, n, NULL);

    /* Tier-2 2b: proactive-eviction guard — BEFORE the weight sweep grows the live
     * banks, spill LRU-idle banks if this quantum's projected growth would breach
     * the resident-KV budget (L328: or preempt the newest decoders; the quantum
     * steps the ones it keeps). */
    session_slot *kept[PULSAR_SESSION_POOL_CAP];
    n = s->guard_maybe_evict(dec, n, kept);
    dec = kept;
    if (n <= 0) return;

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
        g->batch_feed_token = gen_sample_pool(s, g, pool, temp, top_k, top_p, min_p);
        if (g->batch_feed_token < 0) {
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

    pulsar_multiseq_req reqs[PULSAR_SESSION_POOL_CAP];
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < quantum_tokens; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;            /* already stopped */
            if (gen_budget_spent(g)) {
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

        const double step_t0 = server_now_sec();   /* L284: this step's cost observation */
        char err[96];
        /* plan-34 inc 1: route the decode-only lane through the mixed entry
         * (n_rows == n_dec, still exactly 1 row per bank — no prefill rows, no
         * mixing yet). Byte-identical to pulsar_session_decode_multiseq; the only
         * change is the heap descriptor scratch. */
        const int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m, NULL, 0, NULL, 0u, err, sizeof(err));
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

        lane_emit_and_draw(s, dec, live_idx, m, pool);
        /* L284: the plain lane's step cost, measured as the spec lane's round is -- the rows the forward
         * carried against the step's wall, sampling and emission included -- into the same fit structure
         * (pulsar_engine_lane_cost), so the lane choice weighs one clock against the other */
        const double step_ms = (server_now_sec() - step_t0) * 1e3;
        pulsar_engine_lane_cost_observe(s->engine, PULSAR_LANE_PLAIN, (uint32_t)m, step_ms);
        lane_price_observe_plain(&s->w_lane_price, m, step_ms);
    }
    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
}


/* L301: the round's verify pick (verify_curve.cpp, one rule for every drafter; L049 / L117 / L284 ranked the same
 * candidates against a marginal threshold).  surv[i][j] is bank i's survival (the product of its drafter's
 * confidences up to pending position j; a confidence under 0 counts as 1), npend[i] its pending count (0 for a bank
 * not in GEN_DECODE), n_live the decoding banks (one base row each, never cut), row_budget the rows the round may
 * carry (the family's verify width less the round's finishing chunks).  Every (bank, position) is ranked by survival
 * -- then bank, then position, so each bank's admitted set is a prefix of its chain -- and the engine's pick takes
 * the T best that maximise the round's tokens per ms on the measured curve (no curve: as many as the budget holds).
 * k_alloc[i] = bank i's share; tokens[i] (when given) = the survivals of its verified drafts (L312: its expected
 * tokens past the base, what the round selection values its decode by).  Returns the pending drafts left unverified;
 * *overflow = the demand passed the budget. */
static int spec_pick_rows(const pulsar_engine *e, const float surv[][16], const uint32_t *npend, int n, int n_live,
                          int row_budget, int *k_alloc, bool *overflow, double *tokens = NULL) {
    struct cand { float surv; int i; uint32_t j; };
    cand c[PULSAR_SESSION_POOL_CAP * 16];
    uint32_t m = 0;
    for (int i = 0; i < n; i++) {
        k_alloc[i] = 0;
        for (uint32_t j = 0; j < npend[i] && j < 16u; j++) c[m++] = {surv[i][j], i, j};
    }
    *overflow = (uint32_t)n_live + m > (uint32_t)(row_budget > 0 ? row_budget : 0);
    std::sort(c, c + m, [](const cand &a, const cand &b) {
        return a.surv != b.surv ? a.surv > b.surv : a.i != b.i ? a.i < b.i : a.j < b.j;
    });
    float sv[PULSAR_SESSION_POOL_CAP * 16];
    for (uint32_t t = 0; t < m; t++) sv[t] = c[t].surv;
    const uint32_t T = pulsar_engine_verify_pick(e, (uint32_t)n_live, sv, m,
                                                 row_budget > 0 ? (uint32_t)row_budget : 0u);
    for (uint32_t t = 0; t < T; t++) k_alloc[c[t].i]++;
    if (tokens) {
        for (int i = 0; i < n; i++) tokens[i] = 0.0;
        for (uint32_t t = 0; t < T; t++) tokens[c[t].i] += (double)c[t].surv;
    }
    return (int)(m - T);
}


/* L260 fusion: where a chunk that does NOT finish its prompt may end is SCHED_CHUNK_ALIGN (sched_value.h) -- a
 * multiple of the widest compressor group (ratio 128).  Re-chunking a prompt at such a point left its last row
 * unchanged where an unaligned split moved it 5.5e-2 (the classic envelope fused_step_gate measures, story prompt),
 * and an unaligned cut falls wherever the round's leftover room does -- so a prompt's chunking, and its numerics with
 * it, would depend on the load beside it. */
static_assert(SCHED_PROMPTS_MAX == PULSAR_SESSION_POOL_CAP, "the round selection weighs every bank's prompt");

/* L312: the round selection's preferences (sched_value.cpp says what each one means; the simulator behind these
 * numbers is pulsar_test --fused-sched, its runs in rows/L312.md).  Value is counted in prompt tokens.
 *   out_value 64: an output token is worth 64 prompt tokens (Flare's default; decode over prefill, Tyler 2026-10-15).
 *   fairness 4 per ms, split evenly within each class (the decoding streams, the prefilling prompts), the prompts'
 *     class weighing prompt_weight 2 against the streams' 1: while both wait, neither class starves the other.
 *   age_price 0.006 per ms^2 a stream: holding four streams one more second costs what ~1500 prompt tokens are
 *     worth -- the gap's price.
 *   adapt 1000 ms: a price follows its service within about a second (a fused round's length).
 * Measured (gpuq #411, Qwen DFlash2, 4 streams decoding throughout while a 53k prompt prefills): TTFT 136 s, the
 * streams 81 tok/s, gap p99 0.86 s / max 1.38 s; prompts=1: 243 s / 101 / 0.84 / 1.04; out=16: 117 s / 68 / 0.78 /
 * 0.99; L304's budget: the prompt never finished while the streams ran (gap p99 0.97 s / max 1.89 s).
 * PULSAR_SERVER_SCHED="out=..,fair=..,prompts=..,age=..,adapt=.." (read once, announced) moves them for a
 * measurement sweep; a malformed value refuses to start the lane. */
static const sched_prefs *server_sched_prefs() {
    static sched_prefs v;
    static bool init = false;
    if (init) return &v;
    init = true;
    v = {64.0, 4.0, 2.0, 0.006, 1000.0};
    const char *e = getenv("PULSAR_SERVER_SCHED");
    if (!e || !*e) return &v;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", e);
    for (char *save = NULL, *t = strtok_r(buf, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(t, '=');
        char *end = NULL;
        const double x = eq ? strtod(eq + 1, &end) : 0.0;
        double *dst = !eq ? NULL
                      : !strncmp(t, "out=", 4)     ? &v.out_value
                      : !strncmp(t, "fair=", 5)    ? &v.fairness
                      : !strncmp(t, "prompts=", 8) ? &v.prompt_weight
                      : !strncmp(t, "age=", 4)     ? &v.age_price
                      : !strncmp(t, "adapt=", 6)   ? &v.adapt_ms
                                                   : NULL;
        if (!dst || !end || *end || !(x >= 0.0) || (dst == &v.adapt_ms && !(x > 0.0))) {
            server_log(PULSAR_LOG_ERROR, "pulsar-server: PULSAR_SERVER_SCHED=\"%s\": \"%s\" is not out=, fair=, "
                       "prompts=, age= or adapt= with a number (adapt > 0) -- refusing", e, t);
            exit(2);
        }
        *dst = x;
    }
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: round selection preferences out=%g fair=%g prompts=%g age=%g "
               "adapt=%g ms (PULSAR_SERVER_SCHED)", v.out_value, v.fairness, v.prompt_weight, v.age_price,
               v.adapt_ms);
    return &v;
}

static bool slot_prefilling(const session_slot *c) {
    /* L314: a parked (preempted) bank is never a rider -- its physical is freed until its readmission */
    return c->active_job && !c->preempted && c->gen &&
           (c->gen->phase == GEN_PREFILL_COLD || c->gen->phase == GEN_PREFILL_MAIN);
}

static bool slot_fusable_prefill(const session_slot *c) {
    const gen_state *g = c->gen;
    return c->active_job && g && g->phase == GEN_PREFILL_MAIN && !g->no_fuse &&
           g->j->req.n_images == 0 && g->prompt_for_sync;
}

bool server::fusion_enabled() const {
    return pool_banks > 0 && pulsar_engine_has_fused_step(engine);
}

bool server::fuse_prepare(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    pulsar_session *pool = s->sess;
    if (!s->bank_switch(sl->bank)) {
        snprintf(g->err, sizeof g->err, "bank %u state restore failed (evicted KV unrecoverable)",
                 (unsigned)sl->bank);
        s->gen_prefill_fail(sl, false);
        return false;
    }
    const int pos = pulsar_session_pos(pool);
    if (pos == 0) {
        pulsar_session_invalidate(pool);     /* a fresh conversation: the bank empty on the device too */
    } else if (pulsar_session_common_prefix(pool, g->prompt_for_sync) != pos) {
        g->no_fuse = "needs a rewind/stitch";   /* the classic sync owns that */
    } else if (const int at = pulsar_session_bank_resume_at(pool, (uint32_t)sl->bank, g->prompt_for_sync).raw();
               at >= 0 && at != pos) {
        /* L284: a bank-pool family's sync would not continue this history where it stands (decode rows past
         * its prefill: it resumes from a checkpoint below); a fused chunk starts only where the sync would */
        g->no_fuse = "resumes below the bank frontier";
    } else if (pulsar_session_bank_comp_stale(pool, (uint32_t)sl->bank)) {
        /* A fused round would extend the stale compressor group at the frontier
         * and fail the step on both ranks (the pair 2026-10-02 17:11:55); the
         * classic sync's resume starts at a grid point, a group boundary. */
        g->no_fuse = "stale compressor state";
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: bank %d compressor state is stale at %d: "
                   "the prompt takes the classic sync (grid-point resume), not a fused round",
                   sl->bank, pos);
    }
    pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
    s->live_bank = -1;
    g->fuse_ready = !g->no_fuse;
    return g->fuse_ready;
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
 * acceptance ([[L076]]).  Each per-bank phase is ONE engine call over every
 * bank (L260: pulsar_session_spec_*_batch, one mirrored frame on a pair).
 *
 * Emission mirrors gen_decode's L073 discipline: a mid-emit stop (tool-call
 * end, stop string) rewinds the ghost tail so the bank's history never
 * carries tokens the client did not see (pulsar_session_rewind also clears
 * the pendings/carry, which is exactly right -- they were conditioned on the
 * ghosts). */
void server::worker_spec_batched_quantum(session_slot **dec, int n, int quantum_tokens) {
    PULSAR_NVTX_FN();
    auto *s = this;
    if (n <= 0) return;
    pulsar_session *pool = s->sess;
    park_live_bank(s, dec, n, NULL);

    session_slot *kept[PULSAR_SESSION_POOL_CAP];   /* L328: the decoders the guard did not preempt */
    n = s->guard_maybe_evict(dec, n, kept);
    dec = kept;
    if (n <= 0) return;

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
    /* L260: one step per bank per phase; each bank keeps its own accepted[]
     * because the batched round_end finishes every bank before any emits. */
    pulsar_spec_step steps[PULSAR_SESSION_POOL_CAP];
    int step_slot[PULSAR_SESSION_POOL_CAP];
    int accepted[PULSAR_SESSION_POOL_CAP][PULSAR_SPEC_LOGITS_ROWS + 1];

    int emitted_total = 0;
    const bool fuse_on = s->fusion_enabled();
    /* L260 fusion: the prompt rows one round carries, at most -- a classic
     * chunk's worth (prefill_cap) less the verify rows' budget, so a fused step
     * is never wider than a classic one.  It used to be capped at 2048, and a
     * 4096-row chunk runs ~9% more tokens/s than a 2048-row one (pulsar-bench,
     * one Spark: 1272 vs 1165 -- the routed experts are dequantised once per
     * tile, whatever the chunk).  Which chunks ride, and how wide, is L312's
     * round selection (sched_value.cpp): value per time on the engine's measured
     * costs, net of the decoding streams' wait -- it replaced L304's fixed round
     * budget (PULSAR_SERVER_FUSED_ROUND_MS) and L260's "a continuation rides only
     * the first round of a full decode quantum", whose interplay held a prompt's
     * last <= 4096 rows until every stream beside it had finished (L312). */
    /* L284 #2: the rows one step heads -- the verify's and the finishing chunks' -- are the family's cap
     * (DeepSeek: the spec-logits block; Qwen: its logits slab, the widest step its kernels keep decode-width
     * arms), the one authority the verify budget below is cut from, fused or not (L284: the lane pick admits up
     * to this many decoders and the allocator rations their drafts within it).  A finishing chunk rides only
     * while every decode bank keeps at least its base row.  The landing buffers here hold
     * PULSAR_SPEC_LOGITS_ROWS rows; startup refused a family past them (cli_main). */
    const uint32_t heads_max = pulsar_engine_fused_heads_max(s->engine);
    const uint32_t prefill_cap = (uint32_t)pulsar_session_prefill_cap(pool);
    const uint32_t fuse_rows = prefill_cap > heads_max ? prefill_cap - heads_max : 0u;
    const sched_prefs *prefs = server_sched_prefs();
    pulsar_multiseq_req *fused_reqs = NULL;
    /* rule 5 (L284): each prefilling prompt's verdict over this quantum -- the rows it rode in how many
     * rounds, and why it last did not ride -- printed once at the quantum's end */
    struct rider_verdict { int rounds, rows; const char *why; };
    rider_verdict rv[PULSAR_SESSION_POOL_CAP];
    memset(rv, 0, sizeof rv);
    int round_ix = 0;
    while (emitted_total < quantum_tokens) {
        const bool first_round = round_ix++ == 0;
        const double round_t0 = server_now_sec();   /* L263: this round's cost observation */
        pulsar_gpu_sync_stats sync0;                  /* L302: and its synchronisations */
        pulsar_gpu_sync_stats_get(&sync0);
        const int round_emitted0 = emitted_total;   /* L284: and its yield */
        /* ---- L260 fusion: the queued prompts that may ride this round's forward, one chunk each, up to fuse_rows
         * prompt rows in all; a chunk that finishes its prompt gets a head row, reserved out of the verify rows'
         * budget below. */
        struct fused_run { session_slot *sl; int p0, k; bool fin; };
        static_assert(PULSAR_SESSION_POOL_CAP <= PULSAR_FUSED_PF_MAX, "a fused step carries every bank's chunk");
        fused_run fr[PULSAR_SESSION_POOL_CAP];
        int n_fr = 0, n_fin = 0;
        uint32_t pf_rows = 0;
        session_slot *ps[PULSAR_SESSION_POOL_CAP];   /* L312: the prompts the round selection weighs */
        gen_state *pgs[PULSAR_SESSION_POOL_CAP];
        sched_prompt sp[PULSAR_SESSION_POOL_CAP];
        double sp_rate[PULSAR_SESSION_POOL_CAP];
        int n_ps = 0;
        for (int k = 0; k < s->n_slots; k++) {
            session_slot *c = &s->slots[k];
            if (!slot_prefilling(c)) continue;
            gen_state *pg = c->gen;
            const char *why = NULL;   /* rule 5: the verdict the quantum prints for this prompt */
            if (!fuse_on) why = "fusion off";
            else if (pg->phase != GEN_PREFILL_MAIN) why = "cold-store phase";
            else if (pg->j->req.n_images > 0) why = "image prompt";
            else if (pg->no_fuse) why = pg->no_fuse;
            if (why) { rv[k].why = why; continue; }
            if (gen_client_disconnected(pg->j->fd)) {
                s->gen_prefill_abandon(c);
                s->worker_finish_slot(c);   /* L282: released where abandoned (nothing in this quantum steps it) */
                continue;
            }
            /* 2026-10-29: a rider's chunk is a prefill chunk -- the per-chunk memory re-check; short of it the
             * prompt leaves the fused lane, and its classic quantum runs the remedies or refuses it */
            if (!s->prefill_headroom_fused_ok(c)) { rv[k].why = pg->no_fuse; continue; }
            if (!pg->fuse_ready && !s->fuse_prepare(c)) { rv[k].why = pg->no_fuse; continue; }
            const int p0 = pulsar_session_bank_pos(pool, (uint32_t)c->bank);
            if (pg->prompt_for_sync->len - p0 <= 0) { rv[k].why = "nothing left"; continue; }
            ps[n_ps] = c;
            pgs[n_ps] = pg;
            sp[n_ps] = {(uint32_t)p0, (uint32_t)pg->prompt_for_sync->len, pulsar_session_resume_grid(pool),
                        pg->sched_inv_ms, pg->sched_last_ride};
            n_ps++;
        }
        /* ---- L301: the verify pick (spec_pick_rows): every (bank, position) by its drafter's survival -- the
         * product of per-position confidences, read switch-free from the saved carries, which are fresh at sweep
         * start because every bank saved at the previous round's end -- and the count that maximises the round's
         * tokens per ms on the engine's measured verify curve, within the row budget: the family's verify width
         * (heads_max) less this round's finishing chunks (L284: a finishing chunk used to leave the allocation at
         * the full width and sit a whole bank out of the round, assemble's SKIPPED).  It runs at the full width
         * first -- the decode-only round, what the selection values the streams by -- and again when finishing
         * chunks took head rows.  Banks with no or invalid pendings get K=0 and still run their base row. */
        int k_alloc[PULSAR_SESSION_POOL_CAP];
        double k_tok[PULSAR_SESSION_POOL_CAP];
        double dec_rate[PULSAR_SESSION_POOL_CAP];   /* L312: each live decoder's own rate, its price adapts against */
        uint32_t n_live_w = 0;
        float surv[PULSAR_SESSION_POOL_CAP][16];
        uint32_t npend[PULSAR_SESSION_POOL_CAP];
        for (int i = 0; i < n; i++) {
            npend[i] = 0;
            gen_state *ag = dec[i]->gen;
            if (!ag || ag->phase != GEN_DECODE) continue;
            float conf[16];
            int32_t pbase, ptok[16];
            uint32_t np = pulsar_session_bank_pendings(pool, (uint32_t)dec[i]->bank, &pbase, ptok, conf);
            /* L309: a chain under a grammar ends before its first draft the grammar refuses */
            if (np > 0 && ag->gate) {
                const int legal = gen_constrain_draft_prefix(s, ag, pbase, ptok, (int)np);
                if (legal < 0) {
                    ag->finish = "error";
                    ag->phase = GEN_FINISH;
                    continue;
                }
                if ((uint32_t)legal < np && s->w_spec_grammar_cut_rows == 0)   /* rule 5: the cut ran */
                    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: grammar cut a pending chain: %d of %u drafts "
                               "before the grammar's first refusal (L309)", legal, np);
                s->w_spec_grammar_cut_rows += np - (uint32_t)legal;
                np = (uint32_t)legal;
            }
            n_live_w++;
            float p = 1.0f;
            for (uint32_t j2 = 0; j2 < np; j2++) {
                p *= conf[j2] < 0.0f ? 1.0f : conf[j2];
                surv[i][j2] = p;
            }
            npend[i] = np;
        }
        bool overflow = false;
        int pick_cut =
            spec_pick_rows(s->engine, surv, npend, n, (int)n_live_w, (int)heads_max, k_alloc, &overflow, k_tok);
        /* ---- L312: the round selection (sched_value.cpp) over the decode-only round just picked and every
         * prompt's candidate chunks.  A price starts at a 1/n share of its request's own rate (Flare). */
        sched_choice ch;
        {
            sched_round sr;
            memset(&sr, 0, sizeof sr);
            sr.n_live = n_live_w;
            sr.n_dec = (uint32_t)n;
            sr.heads_max = heads_max;
            const double now = server_now_sec();
            const double n_req = (double)n_live_w + (double)n_ps;
            uint32_t li = 0;
            for (int i = 0; i < n; i++) {
                gen_state *ag = dec[i]->gen;
                dec_rate[i] = 0.0;
                if (!ag || ag->phase != GEN_DECODE) continue;
                sr.rows += 1u + (uint32_t)k_alloc[i];
                sr.tokens[li] = 1.0 + k_tok[i];
                dec_rate[i] = sched_decode_rate_max(s->engine, sr.tokens[li], 1u + (uint32_t)k_alloc[i]);
                if (!(ag->sched_inv_ms > 0.0) && dec_rate[i] > 0.0) ag->sched_inv_ms = n_req / dec_rate[i];
                sr.inv_ms[li] = ag->sched_inv_ms;
                if (ag->sched_out_t > 0.0 && now > ag->sched_out_t) sr.age_ms += (now - ag->sched_out_t) * 1e3;
                li++;
            }
            for (int i = 0; i < n_ps; i++) {
                sp_rate[i] = sched_prompt_rate_max(s->engine, &sr, &sp[i], fuse_rows);
                if (!(sp[i].inv_ms > 0.0) && sp_rate[i] > 0.0) sp[i].inv_ms = pgs[i]->sched_inv_ms = n_req / sp_rate[i];
            }
            const uint32_t probe = (s->w_sched_probes & 1u) ? 6u * SCHED_CHUNK_ALIGN : 2u * SCHED_CHUNK_ALIGN;
            sched_choose(s->engine, prefs, &sr, sp, n_ps, fuse_rows,
                         first_round && quantum_tokens >= PULSAR_SERVER_DECODE_QUANTUM_TOKENS, probe, &ch);
            if (n_ps > 0 && !s->w_sched_announced[ch.priced ? 1 : 0]) {   /* rule 5: which rule chose, once each */
                s->w_sched_announced[ch.priced ? 1 : 0] = true;
                if (ch.priced)
                    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: round selection ran priced: %d prompts beside %u "
                               "decoders, decode-only %.0f ms (score %.3g) vs chosen %.0f ms (score %.3g)", n_ps,
                               n_live_w, ch.t_dec_ms, ch.score_dec, ch.t_ms, ch.score);
                else
                    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: round selection ran unpriced (%s): a prompt's rest "
                               "rides whole, a continuation %u rows on a full quantum's first round",
                               ch.t_dec_ms > 0.0 ? "no fused-step fit yet" : "no verify curve", probe);
            }
        }
        for (int i = 0; i < n_ps; i++) {
            session_slot *c = ps[i];
            /* L329 (a): a chunk that would cross the cut a queued request waits for ends on it (a resume-grid point,
             * so 128-aligned and captured: note_prefilled's grid rule); it cannot finish the prompt, which ends past
             * the cut */
            const int mk = pgs[i]->share_mark;
            if (ch.q[i] > 0u && mk > (int)sp[i].p0 && sp[i].p0 + ch.q[i] > (uint32_t)mk) {
                ch.q[i] = (uint32_t)mk - sp[i].p0;
                ch.fin[i] = false;
            }
            if (ch.q[i] == 0u) {
                rv[c - s->slots].why = ch.priced ? "worth less than the wait it causes this round"
                                                 : "unpriced: a continuation rides a full quantum's first round";
                continue;
            }
            fr[n_fr].sl = c;
            fr[n_fr].p0 = (int)sp[i].p0;
            fr[n_fr].k = (int)ch.q[i];
            fr[n_fr].fin = ch.fin[i];
            c->state = SLOT_PREFILLING;
            pgs[i]->sched_last_ride = ++s->w_sched_rounds;
            n_fin += fr[n_fr].fin ? 1 : 0;
            pf_rows += ch.q[i];
            n_fr++;
            if (!ch.priced) s->w_sched_probes++;
        }
        if (n_fin > 0)
            pick_cut = spec_pick_rows(s->engine, surv, npend, n, (int)n_live_w, (int)heads_max - n_fin, k_alloc,
                                 &overflow);
        if (overflow) s->w_spec_overflow_rounds++;
        s->w_spec_pick_cut_rows += (uint64_t)pick_cut;
        /* ---- assemble: ONE call restores each bank, draws its base, begins
         * its round and lays its rows, saving it after (L260: one mirrored
         * frame for every bank, where each bank paid three verdicts). */
        int ns = 0;
        for (int i = 0; i < n && ns < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g || g->phase != GEN_DECODE) continue;
            if (gen_budget_spent(g)) {
                if (g->completion >= g->max_tokens && !g->finish) g->finish = "length";
                g->phase = GEN_FINISH;
                continue;
            }
            /* A bound bank is never a guard victim (guard_pick_victim: no active job), so a decoder's bank is never
             * spilled; one that were would decode over freed rows -- refused by name (L330: a spilled bank installs
             * empty, its job restores it before it resolves) */
            if (s->slots[sl->bank].spilled) {
                snprintf(g->err, sizeof g->err, "bank %u is spilled under its decoder", (unsigned)sl->bank);
                g->finish = "error";
                g->phase = GEN_FINISH;
                continue;
            }
            if (!rounds[i]) rounds[i] = pulsar_spec_round_new();
            pulsar_spec_step *st = &steps[ns];
            memset(st, 0, sizeof(*st));
            st->bank = (uint32_t)sl->bank;
            st->round = rounds[i];
            gen_resolve_sampling_decode(g, &st->temperature, &st->top_k, &st->top_p, &st->min_p);
            st->rng = &g->rng;
            /* L303: a fresh base draw goes through the grammar's mask (a carry was drawn under one) */
            if (gen_constrain_live(g)) {
                st->masks = gen_constrain_base_mask(s, g);
                st->mask_rows = 1u;
                if (!st->masks) {
                    g->finish = "error";
                    g->phase = GEN_FINISH;
                    continue;
                }
            }
            st->max_tokens = g->max_tokens - g->completion;
            /* accepted_cap = allocated K + the base token: round_begin's own
             * trim (K <= accepted_cap-1) enforces the allocation, and its
             * validity checks may trim further -- the allocation is an
             * upper bound, never a promise. */
            st->k_alloc = k_alloc[i];
            st->accepted_cap = k_alloc[i] + 1;
            step_slot[ns++] = i;
        }
        /* The batch's first restore overwrites whatever is installed: park the
         * live bank first, exactly as bank_switch's switch-away save does. */
        if (s->live_bank >= 0) {
            s->slots[s->live_bank].committed_pos = pulsar_session_pos(pool);
            pulsar_session_bank_state_save(pool, (uint32_t)s->live_bank);
            s->live_bank = -1;
        }
        uint32_t rows = 0;
        if (ns > 0)
            (void)pulsar_session_spec_assemble_batch(pool, steps, ns,
                                                     heads_max - (uint32_t)n_fin, reqs, &rows);
        int m = 0;
        for (int j = 0; j < ns; j++) {
            const pulsar_spec_step *st = &steps[j];
            const int i = step_slot[j];
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            switch (st->status) {
            case PULSAR_SPEC_STEP_OK:
                row0s[m] = st->row0;
                row_nb[m] = (int)st->n_rows;
                first_tok[m] = st->first_token;
                live_idx[m] = i;
                m++;
                break;
            case PULSAR_SPEC_STEP_SKIPPED:
                break;
            case PULSAR_SPEC_STEP_EOS:
                /* generate_speculative's short-circuit: emit EOS, never eval it. */
                slot_writer_install(&g->writer);
                if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
                (void)s->gen_emit_token(sl, st->first_token);
                g->phase = GEN_FINISH;
                emitted_total++;
                break;
            case PULSAR_SPEC_STEP_LENGTH:
                g->finish = "length";
                g->phase = GEN_FINISH;
                break;
            case PULSAR_SPEC_STEP_RESTORE_FAILED:
                snprintf(g->err, sizeof g->err,
                         "bank %u state restore failed (evicted KV unrecoverable)",
                         (unsigned)sl->bank);
                g->finish = "error";
                g->phase = GEN_FINISH;
                break;
            default:
                if (st->first_token < 0) snprintf(g->err, sizeof g->err, "%s", st->err);
                else snprintf(g->err, sizeof g->err, "spec round begin failed: %s", st->err);
                g->finish = "error";
                g->phase = GEN_FINISH;
                break;
            }
        }
        if (m == 0) break;

        /* ---- L260 fusion: a round's BASE token was sampled from the carried
         * logits when the round began, so the forward cannot change it; a
         * fused forward is a prompt chunk long (~2 s at 2048 rows), so each
         * live bank streams its base token before it rather than after.  A
         * prompt that just finished prefill gets its first token at once
         * instead of a chunk later (c2: TTFT min 3.3 s against the 1.6 s its
         * own prefill took).  Round end then emits from the first draft on. */
        bool base_sent[PULSAR_SESSION_POOL_CAP], base_stop[PULSAR_SESSION_POOL_CAP];
        bool constrain_failed[PULSAR_SESSION_POOL_CAP] = {};
        for (int q = 0; q < m; q++) {
            base_sent[q] = base_stop[q] = false;
            if (n_fr == 0) continue;
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            slot_writer_install(&g->writer);
            if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
            base_stop[q] = s->gen_emit_token(sl, first_tok[q]);
            base_sent[q] = true;
            g->sched_out_t = server_now_sec();   /* L312: a delivery resets the stream's age clock */
        }

        /* ---- ONE shared forward over every bank's rows ------------------- */
        const double fwd_t0 = server_now_sec();   /* L288: the round's phases (server::w_spec_phase_ms) */
        char err[160];
        pulsar_session_spec_arm_capture(pool, rows);
        uint32_t got = 0;
        int rc;
        if (n_fr > 0) {
            /* L260 fusion: the verify rows, then each riding prompt's chunk */
            if (!fused_reqs)
                fused_reqs = (pulsar_multiseq_req *)server_xmalloc(
                        ((size_t)PULSAR_SPEC_LOGITS_ROWS + fuse_rows) * sizeof(*fused_reqs));
            memcpy(fused_reqs, reqs, (size_t)rows * sizeof(*reqs));
            pulsar_fused_shape shape;
            memset(&shape, 0, sizeof shape);
            shape.n_dec = rows;
            shape.n_pf = (uint32_t)n_fr;
            uint32_t at = rows;
            for (int r = 0; r < n_fr; r++) {
                const pulsar_tokens *t = fr[r].sl->gen->prompt_for_sync;
                /* L329 (b2): the rider's next chunk of PLE rows, read while this round computes */
                (void)pulsar_session_prompt_read_ahead(pool, (uint32_t)fr[r].sl->bank, t, fr[r].p0 + fr[r].k);
                for (int j = 0; j < fr[r].k; j++) {
                    fused_reqs[at].bank = (uint32_t)fr[r].sl->bank;
                    fused_reqs[at].pos = fr[r].p0 + j;
                    fused_reqs[at].token = t->v[fr[r].p0 + j];
                    at++;
                }
                shape.head_last[r] = fr[r].fin ? 1u : 0u;
            }
            rc = pulsar_session_decode_fused(pool, fused_reqs, at, &shape, NULL, 0, &got, err, sizeof err);
            if (rc == 0 && got == rows + (uint32_t)n_fin) got = rows;   /* the verify rows' share */
            /* Rule 5: the lane announces that it RAN, once, with its shape --
             * otherwise only its refusals print and "it ran" rests on inference
             * (the pair's fusion measurement, 2026-10-01). */
            if (rc == 0) {
                static int announced = 0;
                if (!announced) {
                    announced = 1;
                    server_log(PULSAR_LOG_DEFAULT,
                               "pulsar-server: fused step ran: %u verify rows + %u prompt rows in %d runs (%d finishing)",
                               rows, pf_rows, n_fr, n_fin);
                }
            }
            else if (rc == 0) rc = -1;
        } else {
            rc = pulsar_session_decode_mixed(pool, reqs, rows, NULL, 0, &got, PULSAR_MSEQ_HEAD_ALL_ROWS, err,
                                             sizeof err);
        }
        pulsar_session_spec_arm_capture(pool, 0u);
        const double fwd_t1 = server_now_sec();
        s->live_bank = -1;   /* pool is multiseq-poisoned until a bank_switch */
        if (rc != 0 || got != rows) {
            for (int r = 0; r < n_fr; r++) {
                gen_state *pg = fr[r].sl->gen;
                snprintf(pg->err, sizeof pg->err, "fused prefill forward failed: %s", err);
                s->gen_prefill_fail(fr[r].sl, false);   /* shared by every rider: no one file is implicated */
                s->worker_finish_slot(fr[r].sl);        /* L282 */
            }
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

        /* L304 / L327: the fused forward's wall against its prompt runs and its verify rows (whose forward the engine
         * takes off on its verify curve) -- the prompt chunk cost fit the round selection prices the next by */
        if (n_fr > 0) {
            uint32_t fq[PULSAR_SESSION_POOL_CAP], fc[PULSAR_SESSION_POOL_CAP];
            for (int r = 0; r < n_fr; r++) {
                fq[r] = (uint32_t)fr[r].k;
                fc[r] = (uint32_t)fr[r].p0;
            }
            pulsar_engine_fused_cost_observe(s->engine, (uint32_t)n_fr, fq, fc, rows, (fwd_t1 - fwd_t0) * 1e3);
        }

        /* ---- L260 fusion: each riding chunk joins its bank (checkpoint +
         * prefill frontier); a finished prompt takes its head row as the bank's
         * next-token logits and starts its stream (the classic sync's epilogue),
         * then decodes from the next pass. */
        for (int r = 0, fin_row = 0; r < n_fr; r++) {
            session_slot *c = fr[r].sl;
            gen_state *pg = c->gen;
            const int head = fr[r].fin ? fin_row++ : -1;
            if (!s->bank_switch(c->bank) ||
                pulsar_session_note_prefilled(pool, pg->prompt_for_sync->v + fr[r].p0, fr[r].k, head) != 0) {
                snprintf(pg->err, sizeof pg->err, "bank %u: the fused prompt chunk could not be recorded",
                         (unsigned)c->bank);
                s->gen_prefill_fail(c, false);
                s->worker_finish_slot(c);   /* L282 (the bank stays assigned: the save below reads it) */
            } else {
                /* The classic sync's per-chunk callback: slot progress for
                 * /metrics, the SSE headers and keepalive (a long prompt riding
                 * fused rounds was silent to its client for minutes), the
                 * prefill log line with its rate, and the L114 counter through
                 * the per-slot watermark. */
                gen_prefill_progress_cb(pg, "prefill_chunk", fr[r].p0 + fr[r].k,
                                        pg->prompt_for_sync->len);
                rider_verdict *v = &rv[c - s->slots];
                v->rounds++;
                v->rows += fr[r].k;
                if (fr[r].fin) {
                    slot_writer_install(&pg->writer);
                    s->gen_stream_begin(c);
                }
            }
            if (s->live_bank == (int)c->bank) {
                pulsar_session_bank_state_save(pool, (uint32_t)c->bank);
                s->live_bank = -1;
            }
        }

        /* ---- finish every bank's round in ONE call (restore, round_end,
         * save per bank), then emit and persist per bank ------------------ */
        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            pulsar_spec_step *st = &steps[q];
            memset(st, 0, sizeof(*st));
            st->bank = (uint32_t)sl->bank;
            st->round = rounds[live_idx[q]];
            /* L296 toolsample: the request's params, and the tool-call override at each of the round's positions
             * (row i: the token after the base and drafts 0..i-1) as the plain lane resolves it token by token --
             * one resolution for the round drew a crossing's first payload tokens greedy.  The round's tokens are
             * its rows' (reqs, laid by the assemble); a base streamed before the forward is in g->text already. */
            gen_resolve_sampling(&g->j->req, &st->temperature, &st->top_k, &st->top_p, &st->min_p);
            {
                int toks[PULSAR_SPEC_LOGITS_ROWS];
                int nt = 0;
                for (int i = base_sent[q] ? 1 : 0; i < row_nb[q]; i++) toks[nt++] = reqs[row0s[q] + (uint32_t)i].token;
                st->greedy_rows = gen_spec_greedy_rows(s, g, toks, nt, base_sent[q]);
                /* L303: each verify row's mask -- the gate walked through the round's tokens and back */
                const int mr = gen_constrain_round(s, g, toks, nt, base_sent[q], &st->masks);
                if (mr < 0) {
                    /* the grammar failed (g->err says how): masks with no rows make the round refuse and undo
                     * itself, so this request fails and the others' rounds go on */
                    static const uint32_t no_rows = 0;
                    st->masks = &no_rows;
                    st->mask_rows = 0u;
                    constrain_failed[q] = true;
                } else {
                    st->mask_rows = (uint32_t)mr;
                }
            }
            st->rng = &g->rng;
            st->first_token = first_tok[q];
            st->row0 = row0s[q];
            st->accepted = accepted[q];
            st->accepted_cap = (int)(PULSAR_SPEC_LOGITS_ROWS + 1);
        }
        (void)pulsar_session_spec_round_end_batch(pool, steps, m);
        const double end_t1 = server_now_sec();
        int done_of[PULSAR_SESSION_POOL_CAP] = {};   /* L312: each live bank's tokens this round */
        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            const pulsar_spec_step *st = &steps[q];
            if (st->status == PULSAR_SPEC_STEP_RESTORE_FAILED) {
                g->finish = "error";
                snprintf(g->err, sizeof g->err,
                         "bank %u restore failed after spec forward", (unsigned)sl->bank);
                g->phase = GEN_FINISH;
                continue;
            }
            if (st->status != PULSAR_SPEC_STEP_OK) {
                g->finish = "error";
                if (!constrain_failed[q]) snprintf(g->err, sizeof g->err, "spec round end failed: %s", st->err);
                g->phase = GEN_FINISH;
                continue;
            }
            const int na = st->n_accepted;
            /* L155 tripwire: the round must hand back exactly the positions it
             * committed (the engine clamps commit at EOS and trims to the
             * accepted count).  The ghost rewind below counts na - emitted and
             * cannot see a frontier that ran ahead of na, so this is the only
             * place that would notice; it says so if the invariant ever breaks. */
            if (st->pos_after != st->pos_before + na)
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: spec batched round bank %u: frontier %d != "
                           "%d + %d accepted -- the bank carries tokens the client "
                           "will not see (L155)",
                           (unsigned)sl->bank, st->pos_after, st->pos_before, na);
            slot_writer_install(&g->writer);
            int done = 0;
            bool stopped = false;
            int t0 = 0;
            if (base_sent[q] && na > 0) {
                /* streamed before the forward (above); it is accepted[0] */
                done = t0 = 1;
                stopped = base_stop[q];
            }
            for (int t = t0; !stopped && t < na; t++) {
                if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
                done = t + 1;
                if (s->gen_emit_token(sl, st->accepted[t])) { stopped = true; break; }
            }
            if (stopped) g->phase = GEN_FINISH;
            /* L264 S4: decode creates no grid checkpoints, so there is nothing
             * for the disk cache to persist here; the next prefill does it. */
            if (done < na) {
                if (!s->bank_switch(sl->bank)) {
                    snprintf(g->err, sizeof g->err,
                             "bank %u restore failed after spec round end", (unsigned)sl->bank);
                    g->finish = "error";
                    g->phase = GEN_FINISH;
                    continue;
                }
                if (done < na) {
                    /* L073, batched-lane edition: committed-but-never-emitted
                     * tokens rewind, whatever ended the emission.  A family whose
                     * state cannot rewind (Qwen's recurrent layers) has its bank's
                     * view invalidated instead, so the next sync prefills it
                     * rather than continuing a history the client never saw. */
                    const int ghost = na - done;
                    const int target = pulsar_session_pos(pool) - ghost;
                    if (pulsar_engine_can_rewind(s->engine)) {
                        pulsar_session_rewind(pool, target);
                        server_log(PULSAR_LOG_KVCACHE,
                                   "pulsar-server: spec batched round bank %u: rewound %d "
                                   "ghost tokens to pos %d",
                                   (unsigned)sl->bank, ghost, target);
                    } else {
                        pulsar_session_invalidate(pool);
                        server_log(PULSAR_LOG_KVCACHE,
                                   "pulsar-server: spec batched round bank %u: %d committed tokens past "
                                   "the stop; bank view invalidated (no rewind on this family's state)",
                                   (unsigned)sl->bank, ghost);
                    }
                }
                pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
                s->live_bank = -1;   /* saved: the next install need not save it again */
            }
            sl->committed_pos = pulsar_session_bank_pos(pool, (uint32_t)sl->bank);
            sl->tokens_emitted += (uint64_t)done;
            /* L119: request-scoped DSpark accounting — the round's truth,
             * accumulated here where it is unambiguous (row_nb = base+K rows
             * this bank contributed; na = base + accepted drafts). */
            g->req_spec_rounds++;
            if (row_nb[q] > 1) g->req_spec_draft += (uint64_t)(row_nb[q] - 1);
            if (na > 1) g->req_spec_accepted += (uint64_t)(na - 1);
            g->req_spec_gen += (uint64_t)done;
            emitted_total += done;
            done_of[q] = done;
            if (done > 0) g->sched_out_t = server_now_sec();
        }
        /* L150: round_end deferred every bank's redraft; run ONE drafter pass
         * over all of them (the engine reads their saved ring counters and the
         * bank-major rings, and never switches banks), then stamp each bank's
         * draft into its shadow in one batched commit. A
         * device failure here is not fatal: the banks keep no pendings and take
         * a plain n=1 step next round. */
        const double draft_t0 = server_now_sec();
        double draft_t1 = draft_t0;
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
            draft_t1 = server_now_sec();
            /* One call stamps every bank's draft into its shadow (restore,
             * commit, save per bank). */
            for (int q = 0; q < nl; q++) {
                memset(&steps[q], 0, sizeof(steps[q]));
                steps[q].bank = live_banks[q];
                steps[q].round = live_rounds[q];
            }
            if (nl > 0) (void)pulsar_session_spec_redraft_commit_batch(pool, steps, nl);
            for (int q = 0; q < nl; q++) {
                if (steps[q].status != PULSAR_SPEC_STEP_OK) {
                    /* A failed state restore fails the request: skipping the
                     * commit left the bank's ring holding an uncommitted draft
                     * and the slot decoding on (L190 D1). */
                    gen_state *g = live_slots[q]->gen;
                    snprintf(g->err, sizeof g->err,
                             "bank %u state restore failed before redraft commit",
                             (unsigned)live_banks[q]);
                    g->finish = "error";
                    g->phase = GEN_FINISH;
                }
            }
        }
        /* L312: every request the round selection weighed holds the round it was given: each prompt its rows, each
         * stream its tokens, over the round's wall (sched_service_observe; a request that left mid-round is
         * skipped). */
        const double round_t1 = server_now_sec();
        {
            const double wall_ms = (round_t1 - round_t0) * 1e3;
            for (int i = 0; i < n_ps; i++)
                if (ps[i]->gen == pgs[i])
                    sched_service_observe(&pgs[i]->sched_inv_ms, wall_ms, (double)ch.q[i], sp_rate[i],
                                          prefs->adapt_ms);
            for (int q = 0; q < m; q++) {
                gen_state *g = dec[live_idx[q]]->gen;
                if (g)
                    sched_service_observe(&g->sched_inv_ms, wall_ms, (double)done_of[q], dec_rate[live_idx[q]],
                                          prefs->adapt_ms);
            }
        }
        /* /metrics granularity: publish per ROUND, not per quantum.  A
         * quantum (16 tokens) takes ~1 s at deep-context rates, and a
         * scraper polling faster than the publish cadence sees its deltas
         * beat into 0 / 2x-rate flapping (the pulsar-tui square wave).
         * One mutex+copy per round (~0.35 s) is host noise. */
        /* L263: the round's cost, measured -- the rows the forward carried
         * against its wall time, redraft included (the next round's drafting
         * is part of this one's step).  A round that carried prompt rows is
         * not a decode round and is left out. */
        if (n_fr == 0 && rows > 0) {
            const double cut[SERVER_ROUND_PHASE_COUNT + 1] = {round_t0, fwd_t0, fwd_t1, end_t1, draft_t0, draft_t1,
                                                              round_t1};
            for (int p = 0; p < SERVER_ROUND_PHASE_COUNT; p++) s->w_spec_phase_ms[p] += (cut[p + 1] - cut[p]) * 1e3;
            s->w_spec_phase_rounds++;
            pulsar_gpu_sync_stats sync1;
            pulsar_gpu_sync_stats_get(&sync1);
            s->w_spec_syncs[0] += sync1.stream_syncs - sync0.stream_syncs;
            s->w_spec_syncs[1] += sync1.d2h - sync0.d2h;
            s->w_spec_syncs[2] += sync1.d2h_bytes - sync0.d2h_bytes;
            s->w_spec_syncs[3] += sync1.h2d - sync0.h2d;
            const double round_ms = (round_t1 - round_t0) * 1e3, fwd_ms = (fwd_t1 - fwd_t0) * 1e3,
                         draft_ms = (draft_t1 - draft_t0) * 1e3;
            pulsar_engine_lane_cost_observe(s->engine, PULSAR_LANE_SPEC, rows, round_ms);
            /* L284 lane cost: the same round's yield, per bank -- what the lane choice prices the spec lane's
             * rows by (lane_price_pick) */
            lane_price_observe_spec(&s->w_lane_price, emitted_total - round_emitted0, (int)rows, m, round_ms);
            /* L301: and the verify curve's online correction -- the forward against its profile, the rest of the
             * round outside the forward and the drafting, the drafting */
            pulsar_engine_verify_curve_observe(s->engine, (uint32_t)m, rows, fwd_ms, round_ms - fwd_ms - draft_ms,
                                               draft_ms);
            if (!s->w_spec_pick_announced) {   /* rule 5: the pick ran, with the curve it priced on */
                s->w_spec_pick_announced = true;
                pulsar_verify_curve_view v;
                pulsar_engine_verify_curve_view(s->engine, &v);
                server_log(PULSAR_LOG_DEFAULT, "pulsar-server: verify pick ran: %u decoders, %u rows (%s)",
                           n_live_w, rows, v.valid ? "priced on the measured verify curve"
                                                   : "no verify curve: the row budget alone");
            }
        }
        s->publish_metrics_snapshot();
        /* L260 fusion: a prompt finished this round -- end the quantum so its
         * first-token init runs at the top of the next pass and it joins the
         * very next round. */
        if (n_fin > 0) break;
    }
    free(fused_reqs);
    for (int k = 0; k < s->n_slots; k++) {
        if (!rv[k].rounds && !rv[k].why) continue;
        const session_slot *c = &s->slots[k];
        const int len = c->gen && c->gen->prompt_for_sync ? c->gen->prompt_for_sync->len : -1;
        server_log(PULSAR_LOG_PREFILL, "pulsar-server: fuse: bank %d at %d/%d rode %d rows in %d of %d rounds%s%s",
                   c->bank, pulsar_session_bank_pos(pool, (uint32_t)c->bank), len, rv[k].rows, rv[k].rounds,
                   round_ix, rv[k].why ? "; last miss: " : "", rv[k].why ? rv[k].why : "");
    }
    for (int i = 0; i < n; i++) {
        if (rounds[i]) pulsar_spec_round_free(rounds[i]);
        dec[i]->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
}

/* L271 / L272 P1 / L284: a decoder leaves the plain multiseq batch to rejoin the spec-batched lane.  The
 * bank is made live again (bank_state_restore clears the multiseq poison), the host checkpoint catches up
 * to what multiseq committed, and the token the batch sampled (its logprob already captured) is fed,
 * counted and emitted as the batch's next step would have -- so the session's logits are fresh for the
 * round API's base draw, which draws the token after it. */
bool server::batch_leave(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    pulsar_session *pool = s->sess;
    if (!s->bank_switch(sl->bank)) {
        snprintf(g->err, sizeof g->err, "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
        g->finish = "error";
        g->phase = GEN_FINISH;
        return false;
    }
    if (g->batch_pending.len > 0)
        pulsar_session_note_committed_tokens(pool, g->batch_pending.v, g->batch_pending.len);
    pulsar_tokens_free(&g->batch_pending);
    g->batch_active = false;
    if (!g->batch_feed_valid) return true;
    /* L293: the budget is read before the commit, as the batch's next step would have: a spent budget drops the
     * sampled token (never committed, never emitted) and finishes the slot -- truncate the run, never the stream */
    if (gen_budget_spent(g)) {
        g->batch_feed_valid = false;
        g->phase = GEN_FINISH;
        return true;
    }
    /* the token the batch sampled (its logprob already captured) is committed here as the batch's next
     * step would have: fed, counted, emitted -- so the session's logits are fresh for the family's
     * generate, which draws the token after it */
    const int tok = g->batch_feed_token;
    g->batch_feed_valid = false;
    char err[160] = "";
    if (pulsar_session_eval(pool, tok, err, sizeof err) != 0) {
        snprintf(g->err, sizeof g->err, "leaving the batch: %s", err);
        g->finish = "error";
        g->phase = GEN_FINISH;
        return false;
    }
    sl->committed_pos = pulsar_session_pos(pool);
    sl->tokens_emitted++;
    if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
    slot_writer_install(&g->writer);
    if (s->gen_emit_token(sl, tok)) g->phase = GEN_FINISH;
    return true;
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
    PULSAR_NVTX_FN();
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
        if (g->share_mark > P) continue;                        /* L329 (a): the classic sync stops at the cut */
        /* 2026-10-29: the fused run is prefill chunks too -- short of the floor it goes classic (prefill_headroom_ok) */
        if (!s->prefill_headroom_fused_ok(sl)) continue;
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
    PULSAR_NVTX_FN();
    auto *s = this;
    if (n <= 0 || !pf || !pf->gen || !pf->gen->prompt_for_sync) return;
    pulsar_session *pool = s->sess;
    park_live_bank(s, dec, n, pf);
    gen_state *pg = pf->gen;
    const pulsar_tokens *pp = pg->prompt_for_sync;

    session_slot *kept[PULSAR_SESSION_POOL_CAP];   /* L328: the decoders the guard did not preempt */
    n = s->guard_maybe_evict(dec, n, kept);
    dec = kept;
    if (n <= 0) return;

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
        g->batch_feed_token = gen_sample_pool(s, g, pool, temp, top_k, top_p, min_p);
        if (g->batch_feed_token < 0) {
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
    bool pf_at_grid = false;                       /* prefill reached a grid point: commit there (L264) */

    const size_t reqcap = (size_t)PULSAR_SESSION_POOL_CAP + (size_t)kstep;
    pulsar_multiseq_req *reqs = (pulsar_multiseq_req *)server_xmalloc(reqcap * sizeof(*reqs));
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < PULSAR_SERVER_DECODE_QUANTUM_TOKENS; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;
            if (gen_budget_spent(g)) {
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
        if (!pf_giveup && !pf_at_grid && pos_now < len) {
            kthis = kstep;
            if (pos_now + kthis > len) kthis = len - pos_now;
            /* L264: never step over a grid point -- stop there for this quantum,
             * so the commit below lands on it and the bank keeps a checkpoint
             * (note_prefilled); the decode banks run on. */
            const int grid = (int)pulsar_session_resume_grid(pool);
            const int next_grid = (pos_now / grid + 1) * grid;
            if (pos_now + kthis >= next_grid && next_grid < len) {
                kthis = next_grid - pos_now;
                pf_at_grid = true;
            }
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
        int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)nrows, NULL, 0, &n_runs, head_cap, err,
                                             sizeof err);
        const int verdict = mixed_prefill_giveup(rc, kthis, m);
        if (verdict != MIXED_PROCEED) {
            /* RECOVERABLE reject charged to the PREFILL run (see
             * mixed_prefill_giveup): stop folding this prefill (classic via
             * no_fuse); the co-scheduled decode banks retry this step alone. */
            pf_giveup = true; pg->no_fuse = "fused step gave up";
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: fused prefill rejected (%s): this prefill runs classic "
                       "from now on; the %d decode bank(s) retry this step alone", err, m);
            if (verdict == MIXED_GIVEUP_RETRY_DECODE) {
                rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m, NULL, 0, &n_runs, 0u, err, sizeof err);
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
            /* final prefill sub-chunk: the len-1 row (the prefill run's, headed row m, last-of-run) becomes the
             * prefill bank's next-token row -- the decode seed for the handoff, on the device (L302) */
            if (pulsar_session_step_row_to_bank(pool, (uint32_t)m, (uint32_t)pf->bank) != 0) {
                snprintf(pg->err, sizeof pg->err, "prefill bank %u: its last row could not be kept", (unsigned)pf->bank);
                pg->finish = "error"; pg->phase = GEN_FINISH;
                break;
            }
            reached_end = true;
        }

        lane_emit_and_draw(s, dec, live_idx, m, pool);
    }
    free(reqs);

    /* Reconcile the prefill bank (same recipe as a decode bank leaving the lane):
     * install its driver-maintained frontier (P0+pf_done) and advance the host
     * checkpoint by exactly the committed prefill tokens, so its next chunk resumes
     * correctly and any store sees the true frontier. */
    if (pf_done > 0 && pg->phase == GEN_PREFILL_MAIN) {
        /* note_prefilled, not note_committed: these are PREFILL rows, so the
         * bank's prefill frontier moves with them (a resume computes its grid
         * point from it) and a commit that ends on the grid is a checkpoint. */
        if (pulsar_session_bank_state_restore(pool, pf->bank) &&
            pulsar_session_note_prefilled(pool, &pp->v[P0], pf_done, -1) == 0) {
            pf->committed_pos = P0 + pf_done;
            s->live_bank = (int)pf->bank;
            if (reached_end) {
                /* Prefill complete: the restore above installed the fused last-position row (kept for the
                 * bank at the final sub-chunk; byte-identical to classic) -- run the SAME prefill->decode
                 * handoff the classic path uses (kv-cache store, SSE start, GEN_DECODE_INIT). No non-aligned
                 * classic tail resume => the request's decode is byte-identical to a cold classic prefill. */
                slot_writer_install(&pg->writer);
                s->gen_stream_begin(pf);
                slot_writer_flush(&pg->writer);
            }
        } else {
            snprintf(pg->err, sizeof pg->err, "prefill bank %u reconcile failed", (unsigned)pf->bank);
            pg->finish = "error"; pg->phase = GEN_FINISH;
        }
    }

    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
    pf->last_serviced_us = now_us;
}

/* L260: the decode budget of one scheduler iteration, and which non-decode
 * slot it advances.  A prompt's prefill chunk and the running streams' decode
 * alternate one-for-one; a full decode quantum (16 tokens -- several spec
 * rounds, ~0.3-0.6 s) before EVERY queued prompt's chunk put ~0.5 s of decode
 * in front of each prompt's first token (pair, 5 x 2K prompts: ~9.8 s to the
 * last first token against ~7.4 s of prefill).  So while a queued prompt is
 * within one chunk of finishing, the running streams get ONE round per chunk
 * -- what a chunked-prefill step with decode rows riding along gives them --
 * and a long multi-chunk prompt still alternates with full quanta, so it
 * cannot starve decode (the 214 s stall). */
static int worker_decode_budget(server *s) {
    const int cap = pulsar_session_prefill_cap(s->sess);
    for (int i = 0; i < s->n_slots; i++) {
        const session_slot *c = &s->slots[i];
        if (!slot_prefilling(c)) continue;
        const gen_state *g = c->gen;
        const pulsar_tokens *target = g->phase == GEN_PREFILL_COLD ? &g->cold_prefix : g->prompt_for_sync;
        if (!target) continue;
        const int left = target->len - pulsar_session_bank_pos(s->sess, (uint32_t)c->bank);
        if (left <= cap) return 1;
    }
    return PULSAR_SERVER_DECODE_QUANTUM_TOKENS;
}

/* The next non-decode slot to advance, round-robin from *rr.  A slot past its
 * prefill (first-token init, finish) goes before another prompt's chunk: its
 * step is cheap, and round-robin alone made a prompt that had just finished
 * prefill wait for every other queued prompt before its first token (one
 * Spark, 10 x 1.8k prompts: TTFT min 14.2 s where 1.6 s is possible). */
static session_slot *worker_pick_step(server *s, int *rr, const session_slot *skip,
                                      bool (*eligible)(const session_slot *)) {
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < s->n_slots; k++) {
            session_slot *c = &s->slots[(*rr + k) % s->n_slots];
            if (c == skip || !eligible(c)) continue;
            /* L330: a restore slice costs what a prompt chunk does */
            if (pass == 0 && (slot_prefilling(c) || (c->gen && c->gen->phase == GEN_RESTORE))) continue;
            *rr = (int)(c - s->slots) + 1;
            return c;
        }
    }
    return NULL;
}

static bool slot_steppable_beside_decode(const session_slot *c) {
    return c->active_job && !c->preempted && !slot_is_batchable_decode(c) && !(c->gen && c->gen->batch_active);
}

static bool slot_active(const session_slot *c) { return c->active_job != NULL && !c->preempted; }

/* L260 fusion: beside a spec quantum a riding prompt advances inside the
 * quantum's forwards, so the iteration's classic step leaves it alone. */
static bool slot_steppable_beside_fused(const session_slot *c) {
    return slot_steppable_beside_decode(c) && !slot_fusable_prefill(c);
}

void *worker_main(void *arg) {
    server *s = (server *)arg;
    __atomic_store_n(&s->worker_tid, (int)syscall(SYS_gettid), __ATOMIC_RELAXED);   /* the stall dump's /proc */
    int rr = 0; /* round-robin cursor: first slot index to consider next */
    for (;;) {
        pulsar_stall_mark("worker: scheduling");   /* the watchdog's heartbeat (pulsar_stall.h) */
        bool bound = false;
        while (s->worker_try_bind()) bound = true;
        /* L314: a preempted decoder whose KV fits again is reloaded -- once no queued prompt waits on KV (the prompt
         * it was preempted for binds first; readmitting ahead of it would only preempt it again) */
        {
            bool kv_wait;
            {
                pulsar::ScopedLock lk(&s->mu);
                /* L329 (a): the head a waiter is passed over for (pw is worker-thread state) */
                const job *h = s->head;
                while (h && h->pw.leader) h = h->next;
                kv_wait = h && h->refusal_counted == PROVISION_REFUSED_KV_TOUCHED;
            }
            if (!kv_wait && s->worker_readmit_preempted()) bound = true;
        }
        if (bound) s->publish_metrics_snapshot();

        int n_active = 0, n_parked = 0;
        for (int i = 0; i < s->n_slots; i++) {
            if (s->slots[i].active_job && !s->slots[i].preempted) n_active++;
            if (s->slots[i].preempted) n_parked++;
            /* L006 heartbeat: swept over ALL bound slots, not just the one
             * about to get a quantum. The slot that most needs a keepalive is
             * precisely the one starved behind another job's long prefill —
             * which is never the slot being advanced. Non-blocking; a no-op
             * unless that slot has been silent >= PULSAR_SERVER_HEARTBEAT_MS. */
            if (s->slots[i].gen) (void)gen_stream_heartbeat(s->slots[i].gen);
        }
        if (n_active == 0 && n_parked > 0) {
            usleep(10000);   /* L314: only parked decoders, waiting for room: poll their readmission */
            continue;
        }
        if (n_active == 0) {
            /* With every slot free, choose_slot_for_job never returns NULL
             * (provision_bank floor-gates only the SECOND bank onward, so an
             * empty pool always provisions its first bank), so an unbound
             * head cannot reach this wait: sleeping on the condvar until new
             * work or shutdown is safe.  If a refusal path is ever added
             * that can leave head set with nothing active, this predicate
             * becomes a hard spin — keep the first-bank guarantee. */
            /* L311: an idle worker with no queued job first completes the finished turns' terminal checkpoints, one
             * a pass (a request that arrives meanwhile binds before the next) */
            {
                bool queued;
                {
                    pulsar::ScopedLock lk(&s->mu);
                    queued = s->head != NULL;
                }
                if (!queued && (pulsar_stall_mark("terminal checkpoint"), s->terminal_checkpoint_step())) {
                    s->publish_metrics_snapshot();
                    continue;
                }
            }
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

        /* L260: a prompt that finished prefill takes its first-token init
         * (host bookkeeping, no GPU work) HERE, so it rides this iteration's
         * decode round.  Stepped after the quantum as an "other" slot, it
         * waited out a full quantum it was not in (c2 on one Spark: ~600 ms
         * of the second prompt's TTFT). */
        for (int i = 0; i < s->n_slots; i++) {
            session_slot *c = &s->slots[i];
            if (!c->active_job || !c->gen || c->gen->phase != GEN_DECODE_INIT) continue;
            s->generate_job_step(c);
            if (c->gen) slot_writer_flush(&c->gen->writer);
            c->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
            if (!c->gen || c->gen->phase == GEN_DONE) s->worker_finish_slot(c);
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
        /* L271: a lone decoder still in the plain batch (it shared it with a request that has since
         * finished -- Claude Code's side calls) rejoins speculation (L272 P1: every family).
         * L284: and so do decoders the spec lane can carry again -- a third stream beside Qwen's pair (spec
         * banks 2) moved all three to the plain batch, and the pair stayed there after it finished: every later
         * prompt took the classic sync, none rode a fused step -- when the price picks the spec lane (below) */
        /* L284 lane cost: where the spec lane could carry every decoder, the faster lane by measured price takes
         * them (lane_price_pick, one rule for every family) */
        const uint32_t verify_rows = pulsar_engine_fused_heads_max(s->engine);
        const bool spec_carries = s->pool_banks > 0 &&
                                  server_spec_lane_carries(pulsar_engine_has_spec_rounds(s->engine), verify_rows,
                                                           dec, n_dec);
        int priced = 0;
        if (spec_carries) {
            const pulsar_lane_cost plain_cost = pulsar_engine_lane_cost(s->engine, PULSAR_LANE_PLAIN);
            const pulsar_lane_cost spec_cost = pulsar_engine_lane_cost(s->engine, PULSAR_LANE_SPEC);
            priced = lane_price_pick(&s->w_lane_price, &plain_cost, &spec_cost, n_dec, verify_rows);
        }
        if (priced == 3 && n_batched > 0) {
            for (int i = 0; i < n_dec; i++)
                if (dec[i]->gen->batch_active) s->batch_leave(dec[i]);
            n_batched = 0;
        }
        const int lane = server_pick_decode_lane(s->pool_banks, pulsar_engine_has_spec_rounds(s->engine) && priced != 2,
                                                 verify_rows, dec, n_dec, n_batched);
        /* rule 5: the lane a decode sweep runs is announced when it changes (only lane 3 fuses prompts), with the
         * prices it was chosen by when it was priced */
        if (lane != s->w_decode_lane && lane >= 2) {
            const lane_price *lp = &s->w_lane_price;
            if (priced)
                server_log(PULSAR_LOG_GENERATION,
                           "pulsar-server: decode lane %d -> %d (%d decoders, %d in the plain batch; spec %.1f%s vs "
                           "plain %.1f%s tok/s: %s)",
                           s->w_decode_lane, lane, n_dec, n_batched, lp->t_spec, lp->meas_spec ? "" : " predicted",
                           lp->t_plain, lp->meas_plain ? "" : " predicted", lp->why);
            else
                server_log(PULSAR_LOG_GENERATION,
                           "pulsar-server: decode lane %d -> %d (%d decoders, %d in the plain batch)",
                           s->w_decode_lane, lane, n_dec, n_batched);
        }
        /* L289: a deferred re-measure is a decision the lane line does not show (the lane holds): said once a
         * weighing, with the prices that cleared it */
        if (priced && s->w_lane_price.deferral == 1) {
            const lane_price *lp = &s->w_lane_price;
            const bool spec_in = lp->lane == 3;
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: decode lane %d holds (%d decoders; %s %.1f measured clears %s %.1f%s tok/s, "
                       "x%.2f >= %.2f): the re-measure waits up to %u steps",
                       lp->lane, n_dec, spec_in ? "spec" : "plain", spec_in ? lp->t_spec : lp->t_plain,
                       spec_in ? "plain" : "spec", spec_in ? lp->t_plain : lp->t_spec,
                       (spec_in ? lp->meas_plain : lp->meas_spec) ? " measured" : " predicted",
                       spec_in ? lp->t_spec / lp->t_plain : lp->t_plain / lp->t_spec, LANE_PRICE_CLEAR,
                       LANE_PRICE_REPROBE_STEPS * LANE_PRICE_CLEAR_SPAN);
            s->w_lane_price.deferral = 2;
        }
        s->w_decode_lane = lane;
        const bool use_spec_batched = s->w_decode_lane == 3;
        const bool use_batched = s->w_decode_lane >= 2;

        if (use_batched) {
            /* plan-34 inc 5: when the fused lane is armed and a prefilling slot is
             * admissible (P=1), FOLD its next chunk into the decode sweep instead of
             * the separate classic prefill advance below. Flag OFF (or nothing
             * admissible) => pf_fuse==NULL => today's exact decode-quantum +
             * separate-prefill time-slice, byte-identical. */
            session_slot *pf_fuse = NULL;
            const int budget = worker_decode_budget(s);
            if (use_spec_batched) {
                pulsar_stall_mark("spec batched quantum");
                s->worker_spec_batched_quantum(dec, n_dec, budget);
            } else {
                pf_fuse = s->worker_find_fuse_prefill();
                if (pf_fuse) {
                    pulsar_stall_mark("mixed batch quantum");
                    s->worker_mixed_batch_quantum(dec, n_dec, pf_fuse);
                } else {
                    pulsar_stall_mark("batched decode quantum");
                    s->worker_batched_decode_quantum(dec, n_dec, budget);
                }
            }
            /* Finish any slot the batched quantum stopped (per-slot path
             * reconciles its checkpoint). Then also advance ONE non-decode
             * active slot (prefill/init/finish) so prompt ingest never starves
             * behind a long batched decode. */
            for (int i = 0; i < n_dec; i++)
                if (dec[i]->gen && dec[i]->gen->phase != GEN_DECODE) s->worker_service_slot(dec[i]);
            /* inc 5: skip the slot already advanced in-band by the fused
             * quantum (pf_fuse) so it does not also run a classic chunk. */
            session_slot *other = worker_pick_step(s, &rr, pf_fuse,
                                                   use_spec_batched && s->fusion_enabled()
                                                       ? slot_steppable_beside_fused
                                                       : slot_steppable_beside_decode);
            /* L282: before, the finish sat INSIDE `phase != GEN_DONE`, so a slot already done was picked
             * every quantum and never released -- 23 minutes on the pair (2026-10-07), six slots held and two
             * prompts starved behind them while five banks decoded */
            if (other) s->worker_service_slot(other);
            s->publish_metrics_snapshot();
            continue;
        }

        session_slot *sl = worker_pick_step(s, &rr, NULL, slot_active);
        if (!sl) continue; /* unreachable: n_active > 0 */

        /* L118: this fall-through only runs when NO decode slot exists
         * (any GEN_DECODE slot arms use_batched above), so it services
         * prefill/init/finish slots only. The L112 paired-decode mirror
         * that lived here is deleted with the classic lane. */
        s->worker_service_slot(sl);
        s->publish_metrics_snapshot(); /* /metrics: once per quantum */
    }
    return NULL;
}


/* spec_mtp.cpp -- the chained-MTP drafter (spec_mtp.h; L295 F6, L296 N9): the lane a family's declared MTP modules run
 * on -- the lockstep with the trunk, the absorb of a round's kept rows, the draft chains in lockstep, the drafter hooks
 * of the declared state -- behind the speculation round API (session_spec.cpp) and the core driver's verify target
 * (step_driver.cpp PULSAR_DRIVER_SPEC_TARGET).
 *
 * The lane's arithmetic is the family's (its module steps); what is here is only which rows a step carries, in which
 * order, and what each bank's parked row is.  The rules (L251 MTP-SPEC, L272 P1, L288, L289) were Qwen's drafter's
 * (spec_qwen.cpp until L296 N9), lifted with no change of order; Qwen and Nemotron-H both draft here. */
#include "pulsar_engine_internal.h"
#include "spec_mtp.h"

#include <math.h>

namespace {

pulsar_mtp_lane *LANE(const pulsar_session *s) { return s->driver ? s->driver->mtp : NULL; }
const char *ARCH(const pulsar_session *s) { return s->engine->family->arch; }

/* the module step j of a chain (1-based) runs: one module recursively, N modules each its own step, the last again */
uint32_t chain_module(const pulsar_mtp_lane *l, uint32_t j) { return (j < l->modules ? j : l->modules) - 1u; }

/* the parked rows of every bank (drafter slot n_trunk, the family's pending decl) */
pulsar_gpu_tensor *pend_tensor(const pulsar_session *s) {
    const pulsar_state_set *ss = s->states;
    return pulsar_state_tensor(ss, ss->plan->n_trunk, LANE(s)->ops->pend_decl);
}

/* One module step; on a failure its rows' banks lose their parked row (their module state has a hole the chain must
 * not read past). */
bool step(pulsar_session *s, const pulsar_mtp_step &m) {
    pulsar_mtp_lane *l = LANE(s);
    if (l->ops->forward(s, &m)) return true;
    fprintf(stderr, "pulsar: %s: the MTP step failed; its banks' drafter state is dropped\n", ARCH(s));
    for (uint32_t r = 0; r < m.n; r++)
        if ((uint32_t)m.bank[r] < l->n_banks) l->pend_pos[m.bank[r]] = UINT32_MAX;
    return false;
}

pulsar_mtp_step step_of(uint32_t module, pulsar_step_mode mode, bool decode_rows, const int32_t *tok,
                        const int32_t *pos, const int32_t *bank, uint32_t n, const pulsar_gpu_tensor *cond) {
    pulsar_mtp_step m{};
    m.module = module;
    m.mode = mode;
    m.decode_rows = decode_rows;
    m.tokens = tok;
    m.pos = pos;
    m.bank = bank;
    m.n = n;
    m.cond = cond;
    return m;
}

}  // namespace

/* ---- the lane -------------------------------------------------------------------------------------------------- */

pulsar_mtp_lane *pulsar_mtp_lane_create(pulsar_session *s, const pulsar_mtp_ops *ops, uint32_t n_banks,
                                        uint32_t max_rows) {
    const pulsar_engine *e = s->engine;
    const pulsar_state_plan *p = &e->state_plan;
    const uint32_t mods = ops->modules(e);
    const uint64_t cb = ops->cond_bytes(e);
    /* the declaration the lane runs on: a drafter slot per module, the pending decl FIXED on the first, its row the
     * conditioning row, the plan's hooks the lane's */
    const char *why = mods == 0 || p->n_slots != p->n_trunk + mods ? "a drafter slot per module"
                      : ops->pend_decl >= p->n_decl || !(p->slot[p->n_trunk] >> ops->pend_decl & 1u)
                          ? "the pending decl on the first drafter slot"
                      : p->decl[ops->pend_decl].kind != PULSAR_STATE_FIXED || p->decl[ops->pend_decl].host ||
                              p->decl[ops->pend_decl].capture != PULSAR_STATE_CAPTURE_NONE
                          ? "a FIXED device pending decl with no capture"
                      : p->decl[ops->pend_decl].bytes != cb ? "the pending decl one conditioning row"
                      : p->drafter_at != pulsar_mtp_drafter_at || p->drafter_frontier != pulsar_mtp_drafter_frontier
                          ? "the lane's drafter hooks"
                      : (ops->stage_save == NULL) != (ops->stage_restore == NULL) ? "a stage save and its restore"
                                                                                   : NULL;
    if (why) {
        fprintf(stderr, "pulsar: %s: the MTP declaration lacks %s (%u modules, %u drafter slots) -- refusing the "
                        "session\n", e->family->arch, why, mods, p->n_slots - p->n_trunk);
        return NULL;
    }
    pulsar_mtp_lane *l = (pulsar_mtp_lane *)xcalloc(1, sizeof(*l));
    l->ops = ops;
    l->modules = mods;
    l->n_banks = n_banks;
    l->cond_bytes = cb;
    l->head_width = ops->head_width(e);
    l->cond = pulsar_gpu_tensor_alloc((uint64_t)max_rows * cb);
    l->pend_pos = (uint32_t *)xmalloc(n_banks * sizeof(uint32_t));
    l->stage_dirty = (bool *)xcalloc(n_banks, sizeof(bool));
    for (uint32_t b = 0; b < n_banks; b++) l->pend_pos[b] = UINT32_MAX;
    if (!l->cond) {
        pulsar_mtp_lane_free(l);
        return NULL;
    }
    return l;
}

void pulsar_mtp_lane_free(pulsar_mtp_lane *l) {
    if (!l) return;
    pulsar_gpu_tensor_free(l->cond);
    free(l->pend_pos);
    free(l->stage_dirty);
    free(l);
}

/* At T the parked row is the trunk's at T - 1 (none at 0) and no stage holds draft rows (L289: the walk writes the
 * true tail). */
void pulsar_mtp_drafter_at(void *ud, uint32_t bank, uint32_t T) {
    pulsar_mtp_lane *l = ((pulsar_driver_state *)ud)->mtp;
    if (!l) return;
    l->pend_pos[bank] = T - 1u;
    l->stage_dirty[bank] = false;
}

/* A payload's frontier T: the parked row is the trunk's at T - 1 and no stage holds draft rows (a chain consumes the
 * parked row and writes into the stage until the round's absorb puts both back). */
bool pulsar_mtp_drafter_frontier(void *ud, uint32_t bank, uint32_t T, char *why, size_t whylen) {
    const pulsar_mtp_lane *l = ((const pulsar_driver_state *)ud)->mtp;
    if (l && (T == 0u || l->pend_pos[bank] != T - 1u || l->stage_dirty[bank])) {
        snprintf(why, whylen, "bank %u's MTP lane is mid-draft (pending row %d, stage %s); step it first", bank,
                 l->pend_pos[bank] == UINT32_MAX ? -1 : (int)l->pend_pos[bank],
                 l->stage_dirty[bank] ? "holds draft rows" : "clean");
        return false;
    }
    return true;
}

/* ---- the lockstep (the driver's absorb) ----------------------------------------------------------------------------
 * After a trunk step over rows (tokens, pos, bank) -- step rows [row0, row0 + n), the trunk's rows there (L284 #2: a
 * fused step's prompt run sits behind its verify rows) -- every row whose next token is now known gets its module row,
 * and each bank's last row is parked.  PREFILL: one bank, consecutive positions -- the bank's parked row (position
 * P - 1, if any) pairs with tokens[0], row i with tokens[i + 1], and the last row is parked.  DECODE: one row per bank
 * -- each bank's parked row pairs with the token this step fed, and this step's row is parked.  Module 0 runs: the
 * lockstep row is a chain's first step. */
bool pulsar_mtp_absorb(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                       const int32_t *bank, uint32_t n, uint32_t row0, bool) {
    pulsar_mtp_lane *l = LANE(s);
    if (!l) return true;
    const pulsar_mtp_ops *ops = l->ops;
    pulsar_gpu_tensor *pend = pend_tensor(s);
    const uint64_t cb = l->cond_bytes;
    std::vector<int32_t> mt(n + 1u), mp(n + 1u), mb(n + 1u);
    uint32_t k = 0;
    if (!pend) {
        fprintf(stderr, "pulsar: %s: the MTP lane has no pending rows -- refusing the step\n", ARCH(s));
        return false;
    }
    bool ok = true;
    if (mode == PULSAR_STEP_PREFILL) {
        const uint32_t b = (uint32_t)bank[0];
        if (l->pend_pos[b] != UINT32_MAX && l->pend_pos[b] + 1u == (uint32_t)pos[0]) {
            ok = pulsar_gpu_tensor_copy_async(l->cond, 0, pend, (uint64_t)b * cb, cb) != 0;
            mt[k] = tokens[0]; mp[k] = pos[0] - 1; mb[k] = (int32_t)b; k++;
        }
        if (ok && n > 1) ok = ops->trunk_rows(s, row0, n - 1u, l->cond, (uint64_t)k * cb);
        for (uint32_t i = 0; i + 1u < n; i++) { mt[k] = tokens[i + 1]; mp[k] = pos[i]; mb[k] = (int32_t)b; k++; }
        if (ok) ok = ops->trunk_rows(s, row0 + n - 1u, 1u, pend, (uint64_t)b * cb);
        if (ok) l->pend_pos[b] = (uint32_t)pos[n - 1];
    } else {
        for (uint32_t j = 0; ok && j < n; j++) {
            const uint32_t b = (uint32_t)bank[j];
            if (l->pend_pos[b] != UINT32_MAX && l->pend_pos[b] + 1u == (uint32_t)pos[j]) {
                ok = pulsar_gpu_tensor_copy_async(l->cond, (uint64_t)k * cb, pend, (uint64_t)b * cb, cb) != 0;
                mt[k] = tokens[j]; mp[k] = pos[j] - 1; mb[k] = (int32_t)b; k++;
            }
            if (ok) ok = ops->trunk_rows(s, row0 + j, 1u, pend, (uint64_t)b * cb);
            if (ok) l->pend_pos[b] = (uint32_t)pos[j];
        }
    }
    if (!ok || k == 0) return ok;
    pulsar_mtp_step m = step_of(0u, mode, false, mt.data(), mp.data(), mb.data(), k, l->cond);
    /* the probe: every row headed, appended to its rows (k <= the head width: a longer chunk is not probed) */
    pulsar_mtp_probe *pr = l->probe;
    if (pr && k <= PULSAR_DRIVER_HEADS_MAX && pr->n + k <= pr->cap) {
        m.head_n = k;
        m.logits = pr->logits + (size_t)pr->n * l->head_width;
    }
    ok = step(s, m);
    if (ok && m.head_n) {
        for (uint32_t r = 0; r < k; r++) pr->pos[pr->n + r] = mp[r];
        pr->n += k;
    }
    return ok;
}

/* ---- the absorb of a round's kept rows ------------------------------------------------------------------------------
 * The committed rows of the last verify, any number of banks (each bank's rows one run, in the step's row order):
 * every bank's stage back to what its chain saved, the kept rows' module rows (trunk row i, next[i]) of EVERY bank in
 * one step (L288: one run a bank, decode rows) -- after the lockstep row of a bank whose parked row no chain consumed
 * -- and each bank's last kept row parked for its next draft.  At most one row a kept row: the verify width. */
bool pulsar_mtp_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *next,
                              uint32_t n) {
    pulsar_mtp_lane *l = LANE(s);
    pulsar_state_set *ss = s->states;
    if (!l) {
        fprintf(stderr, "pulsar: %s: an MTP absorb with no MTP lane -- refusing\n", ARCH(s));
        return false;
    }
    const pulsar_mtp_ops *ops = l->ops;
    const uint64_t cb = l->cond_bytes;
    pulsar_gpu_tensor *pend = pend_tensor(s);
    int32_t tok[PULSAR_DRIVER_HEADS_MAX], pos[PULSAR_DRIVER_HEADS_MAX], bank[PULSAR_DRIVER_HEADS_MAX];
    uint32_t m = 0;   /* the step's rows */
    bool ok = n <= PULSAR_DRIVER_HEADS_MAX && pend;
    for (uint32_t i = 0; ok && i < n;) {
        const uint32_t b = banks[i];
        uint32_t j = i + 1;
        while (j < n && banks[j] == b) j++;
        const uint32_t k = j - i;
        for (uint32_t r = i + 1; ok && r < j; r++) ok = rows[r] == rows[i] + (r - i);
        if (!ok || b >= ss->n_banks || ss->bank_pos[b] < k) {
            fprintf(stderr, "pulsar: %s: bank %u's absorbed rows are not one run of its kept rows -- refusing\n",
                    ARCH(s), b);
            ok = false;
            break;
        }
        const uint32_t p0 = ss->bank_pos[b] - k;   /* the commit moved the counter past the kept rows */
        if (l->stage_dirty[b]) {
            ok = ops->stage_restore(s, b);
            l->stage_dirty[b] = false;
        }
        /* a row still parked at p0 - 1 -- no chain consumed it (the round's first after a sync, or a round whose
         * drafter did not run) -- pairs with the run's base token, the lockstep row a chain's step 1 would have
         * written: the module's state has no hole at p0 - 1 */
        if (ok && l->pend_pos[b] != UINT32_MAX && l->pend_pos[b] + 1u == p0 && rows[i] < l->n_vtok) {
            ok = pulsar_gpu_tensor_copy_async(l->cond, (uint64_t)m * cb, pend, (uint64_t)b * cb, cb) != 0;
            tok[m] = l->vtok[rows[i]];
            pos[m] = (int32_t)p0 - 1;
            bank[m++] = (int32_t)b;
        }
        if (ok && k > 1) ok = ops->trunk_rows(s, rows[i], k - 1u, l->cond, (uint64_t)m * cb);
        for (uint32_t r = 0; r + 1 < k; r++, m++) {
            tok[m] = next[i + r];
            pos[m] = (int32_t)(p0 + r);
            bank[m] = (int32_t)b;
        }
        i = j;
    }
    if (ok && m > 0) ok = step(s, step_of(0u, PULSAR_STEP_PREFILL, true, tok, pos, bank, m, l->cond));
    for (uint32_t i = 0; i < n;) {
        const uint32_t b = banks[i];
        uint32_t j = i + 1;
        while (j < n && banks[j] == b) j++;
        bool parked = ok && b < ss->n_banks && ss->bank_pos[b] >= j - i;
        if (parked) parked = ops->trunk_rows(s, rows[j - 1], 1u, pend, (uint64_t)b * cb);
        if (b < ss->n_banks) l->pend_pos[b] = parked ? ss->bank_pos[b] - 1u : UINT32_MAX;
        ok = ok && parked;
        i = j;
    }
    if (!ok)
        fprintf(stderr, "pulsar: %s: the MTP absorb failed; its banks draft nothing until their next row\n", ARCH(s));
    return ok;
}

/* ---- the draft chains ------------------------------------------------------------------------------------------------
 * The draft chains of `n` banks in lockstep -- step j runs every bank still drafting as ONE module step, one row a
 * bank, every row headed and picked in one launch.  Bank banks[i]'s chain follows its record's carry x at the bank's
 * next position, into the record: refined[0] = x, refined[1..keep] = the drafts, conf[] the head's top probability at
 * each position, and for sampled drafts q(d) with q's support.  The stop rule runs position by position: the first
 * draft whose confidence is under tau is its chain's last, so no module step is spent past it.  A row's bytes do not
 * depend on the rows beside it (one row a bank, the decode arms, a per-row pick), so each chain is the one its bank
 * drafts alone. */
static bool mtp_draft_chains(pulsar_session *s, const uint32_t *banks, uint64_t *const *rngs,
                             spec_redraft_req *const *reqs, uint32_t n, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_mtp_lane *l = LANE(s);
    pulsar_state_set *ss = s->states;
    if (!l) {
        snprintf(err, errlen, "%s: speculative decoding needs the MTP lane (the artifact's modules)", ARCH(s));
        return false;
    }
    const pulsar_mtp_ops *ops = l->ops;
    const uint64_t cb = l->cond_bytes;
    const float tau = pulsar_spec_tau(e);
    pulsar_gpu_tensor *pend = pend_tensor(s);
    if (n > PULSAR_DRIVER_HEADS_MAX || !pend) {
        snprintf(err, errlen, "%s: %u draft chains in one lockstep (at most %u)", ARCH(s), n, PULSAR_DRIVER_HEADS_MAX);
        return false;
    }
    struct chain {
        uint32_t b, p, K;       /* the bank, the carry's position, the chain's depth */
        bool sampled, live;     /* live: the chain takes the next step */
        uint32_t row;           /* its row in the last step */
        float run;              /* its survival: the product of its drafts' confidences */
        bool copy;              /* L288: its drafts are a copy chain; it takes step 1 only (its module row) */
    } c[PULSAR_DRIVER_HEADS_MAX];
    for (uint32_t i = 0; i < n; i++) {
        spec_redraft_req *q = reqs[i];
        chain &ch = c[i];
        ch.b = banks[i];
        ch.p = ss->bank_pos[ch.b];
        ch.sampled = q->temperature > 0.0f;
        ch.live = false;
        ch.copy = q->copied;   /* L288: the record holds a copy chain (pulsar_spec_draft_req_init) */
        ch.run = 1.0f;
        if (!ch.copy) {
            q->keep = 0;
            q->have_conf = true;
            q->refined[0] = q->next_base;
        }
        if (ch.sampled && !rngs[i]) {
            snprintf(err, errlen, "%s: sampled speculation needs the request's rng", ARCH(s));
            return false;
        }
        /* no parked row (a failed module step, a fresh restore): this round verifies the carry alone */
        if (l->pend_pos[ch.b] == UINT32_MAX || l->pend_pos[ch.b] + 1u != ch.p) continue;
        /* the drafts and the carry must fit the context */
        if (q->n_draft > PULSAR_MTP_DRAFT_MAX) {
            snprintf(err, errlen, "%s: a chain of %u drafts is past the verify capture's %u", ARCH(s), q->n_draft,
                     PULSAR_MTP_DRAFT_MAX);
            return false;
        }
        ch.K = q->n_draft;
        if ((uint64_t)ch.p + 1u + ch.K > (uint64_t)ss->ctx) ch.K = ss->ctx > ch.p + 1u ? ss->ctx - ch.p - 1u : 0u;
        if (ch.K == 0) continue;
        ch.live = true;
        /* L289: the verify will embed [x, d_1 ..]: a family may start reading what they need as each is known */
        if (ops->prefetch) ops->prefetch(s, ch.b, &q->refined[0], ch.copy ? 1u + q->keep : 1u, true);
    }
    /* L301: every survival the round has drafted so far -- what a further step's drafts are weighed beside by the
     * verify pick (pulsar_spec_chain_step) -- and the drafting wall spent so far, which the round pays too */
    const double draft_t0 = now_sec();
    float drafted[PULSAR_DRIVER_HEADS_MAX * PULSAR_MTP_DRAFT_MAX];
    uint32_t n_drafted = 0;
    /* L288 copy drafts: a record that holds a copy chain (pulsar_spec_draft_req_init) keeps it; its drafts outrank the
     * module's drafts by their survival like any other.  It still takes step 1 when its bank has a parked row -- the
     * lockstep row writes the bank's module row at p - 1, which the next round's chain conditions on -- and its pick
     * there is dropped. */
    for (uint32_t i = 0; i < n; i++) {
        if (!c[i].copy) continue;
        float run = 1.0f;
        for (uint32_t k = 0; k < reqs[i]->keep; k++) drafted[n_drafted++] = run *= reqs[i]->conf[k];
    }
    bool any_sampled = false;
    for (uint32_t i = 0; i < n; i++) any_sampled |= c[i].live && !c[i].copy && c[i].sampled;
    /* L289: the picks are the device's; L302: so are the sampled chains' draws (THE sampler over the headed rows, at
     * each draft's key), one launch a step -- the rows never come back */
    const pulsar_gpu_tensor *q_map = NULL;
    const pulsar_gpu_tensor *q_rows = any_sampled ? ops->q_rows(s, &q_map) : NULL;
    if (any_sampled && !q_rows) {
        snprintf(err, errlen, "%s: the MTP head's rows are not on the device", ARCH(s));
        return false;
    }
    int32_t drawn[PULSAR_DRIVER_HEADS_MAX];
    pulsar_mtp_pick pick[PULSAR_DRIVER_HEADS_MAX];
    /* position j of chain i from its row's pick: its confidence (whether it is the chain's last), then its draft:
     * the pick (greedy) or its draw from q over its row (sampled, drawn[i] by the step's launch) */
    auto draft = [&](uint32_t i, uint32_t j) -> bool {
        if (c[i].copy) {   /* its drafts are the copy; step 1 wrote its module row */
            c[i].live = false;
            return true;
        }
        spec_redraft_req *q = reqs[i];
        const pulsar_mtp_pick &pk = pick[c[i].row];
        q->conf[j - 1] = pk.conf;
        q->keep = j;
        c[i].run *= pk.conf;
        drafted[n_drafted++] = c[i].run;   /* j <= K <= PULSAR_MTP_DRAFT_MAX (refused above) */
        if (pulsar_spec_conf_stops(pk.conf, tau) || j == c[i].K) c[i].live = false;   /* the chain's last */
        if (!c[i].sampled) {
            q->refined[j] = pk.id;
            if (ops->prefetch) ops->prefetch(s, c[i].b, &q->refined[j], 1, false);
            return true;
        }
        if (drawn[i] < 0) return false;
        q->refined[j] = drawn[i];
        if (ops->prefetch) ops->prefetch(s, c[i].b, &q->refined[j], 1, false);
        return true;
    };
    int32_t tok[PULSAR_DRIVER_HEADS_MAX], pos[PULSAR_DRIVER_HEADS_MAX], bank[PULSAR_DRIVER_HEADS_MAX];
    pulsar_gpu_tensor *out = ops->out_rows(s);
    bool ok = out != NULL;
    for (uint32_t j = 1; ok; j++) {
        /* the step's rows: step 1 is every chain's lockstep row (the parked row at p - 1, x); step j > 1 feeds each
         * live chain's last output row with d_{j-1} at p + j - 2 */
        uint32_t m = 0;
        bool headed = false;   /* a chain that drafts from this step's pick rides it (a copy chain needs no head) */
        for (uint32_t i = 0; ok && i < n; i++) {
            chain &ch = c[i];
            if (!ch.live) continue;
            headed |= !ch.copy;
            ok = j == 1 ? pulsar_gpu_tensor_copy_async(l->cond, (uint64_t)m * cb, pend, (uint64_t)ch.b * cb, cb) != 0
                        : pulsar_gpu_tensor_copy_async(l->cond, (uint64_t)m * cb, out, (uint64_t)ch.row * cb, cb) != 0;
            tok[m] = reqs[i]->refined[j - 1];
            pos[m] = (int32_t)(ch.p + j) - 2;
            bank[m] = (int32_t)ch.b;
            ch.row = m++;
        }
        if (!ok || m == 0) break;
        const double step_t0 = now_sec();
        pulsar_mtp_step ms = step_of(chain_module(l, j), PULSAR_STEP_DECODE, false, tok, pos, bank, m, l->cond);
        if (headed) {
            ms.head_n = m;
            ms.pick = pick;
        }
        ok = step(s, ms);
        /* the sampled live chains' drafts at this step, one launch: chain i's row, the key of the token it proposes
         * (draft j sits j places past the carry: the key the target's draw for that token reads) */
        if (ok && headed && any_sampled) {
            pulsar_gpu_sample_req dq[PULSAR_DRIVER_HEADS_MAX];
            pulsar_gpu_sample_out dout[PULSAR_DRIVER_HEADS_MAX];
            uint32_t drow[PULSAR_DRIVER_HEADS_MAX], who[PULSAR_DRIVER_HEADS_MAX], nd = 0;
            for (uint32_t i = 0; i < n; i++) {
                drawn[i] = -1;
                if (!c[i].live || c[i].copy || !c[i].sampled) continue;
                const spec_redraft_req *q = reqs[i];
                pulsar_gpu_sample_req &r = dq[nd];
                memset(&r, 0, sizeof(r));
                r.temperature = q->temperature;
                r.top_k = q->top_k;
                r.top_p = q->top_p;
                r.min_p = q->min_p;
                r.key = pulsar_sample_key(*rngs[i], j);
                r.probe = -1;
                drow[nd] = c[i].row;
                who[nd++] = i;
            }
            if (nd > 0) ok = s->draw_rows_width(q_rows, l->head_width, q_map, drow, nd, dq, dout);
            for (uint32_t k = 0; ok && k < nd; k++) drawn[who[k]] = dout[k].token;
        }
        if (j == 1) {
            for (uint32_t i = 0; i < n; i++) {
                if (!c[i].live) continue;
                const uint32_t b = c[i].b;
                l->pend_pos[b] = UINT32_MAX;   /* consumed; the absorb re-parks */
                /* the stage after the bank's last TRUE row: the chain below writes draft rows into it */
                if (!ok || !ops->stage_save) continue;
                ok = ops->stage_save(s, b);
                l->stage_dirty[b] = true;
            }
        }
        for (uint32_t i = 0; ok && i < n; i++)
            if (c[i].live) ok = draft(i, j);
        /* the step's wall, picks and draws included (the picks' readback waited for the device): the drafter's step
         * price.  Off TP only (a group's ranks must draft alike). */
        if (ok && headed && !e->tp) lane_cost_fit_observe(&e->draft_step_cost, m, (now_sec() - step_t0) * 1e3);
        /* L301: the live chains whose next drafts raise the round's tokens per ms on the verify curve once the step
         * that drafts them is paid (pulsar_spec_chain_step, the verify pick's rule) take it */
        if (ok) {
            float run[PULSAR_DRIVER_HEADS_MAX];
            uint32_t at[PULSAR_DRIVER_HEADS_MAX], nl = 0;
            bool go[PULSAR_DRIVER_HEADS_MAX];
            for (uint32_t i = 0; i < n; i++) {
                if (!c[i].live) continue;
                run[nl] = c[i].run;
                at[nl++] = i;
            }
            (void)pulsar_spec_chain_step(s, n, drafted, n_drafted, run, nl, (now_sec() - draft_t0) * 1e3, go);
            for (uint32_t x = 0; x < nl; x++) c[at[x]].live = go[x];
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s: the MTP draft chain failed (see the log)", ARCH(s));
        return false;
    }
    return true;
}

/* The single lane's draft: the chain into a record, the record stamped into the bank's pendings. */
uint32_t pulsar_mtp_draft(pulsar_session *s, int next_base, bool, float temperature, int top_k, float top_p,
                          float min_p, uint64_t *rng) {
    spec_redraft_req rec;
    memset(&rec, 0, sizeof rec);
    pulsar_spec_draft_req_init(s, &rec, next_base, temperature, top_k, top_p, min_p);
    char err[200];
    uint32_t keep = 0;
    spec_redraft_req *req = &rec;
    const uint32_t b = s->states->live_bank;
    if (mtp_draft_chains(s, &b, &rng, &req, 1, err, sizeof err)) {
        pulsar_spec_redraft_stamp(s, &rec);
        keep = rec.keep;
    } else {
        fprintf(stderr, "pulsar: %s (the round takes a plain step next)\n", err);
        pulsar_spec_drop_pendings(&s->spec);
    }
    return keep;
}

/* The batched lane's draft: every bank's chain in lockstep (mtp_draft_chains), PULSAR_DRIVER_HEADS_MAX banks at a
 * time. */
int pulsar_mtp_draft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n,
                           char *err, size_t errlen) {
    uint32_t b[PULSAR_DRIVER_HEADS_MAX];
    uint64_t *r[PULSAR_DRIVER_HEADS_MAX];
    spec_redraft_req *q[PULSAR_DRIVER_HEADS_MAX];
    uint32_t m = 0;
    for (int i = 0; i <= n; i++) {
        if (i < n) {
            spec_redraft_req *rq = &rounds[i]->redraft;
            if (!rq->valid || rq->n_draft == 0) continue;
            if (banks[i] >= s->states->n_banks) {
                snprintf(err, errlen, "%s: the draft's bank %u is outside the pool", ARCH(s), banks[i]);
                return -1;
            }
            b[m] = banks[i];
            r[m] = rngs[i];
            q[m++] = rq;
        }
        if (m == PULSAR_DRIVER_HEADS_MAX || (i == n && m > 0)) {
            if (!mtp_draft_chains(s, b, r, q, m, err, errlen)) return -1;
            for (uint32_t k = 0; k < m; k++) q[k]->done = true;
            m = 0;
        }
    }
    return 0;
}

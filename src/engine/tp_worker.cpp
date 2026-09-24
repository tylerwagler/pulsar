#include "pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"
#include <unistd.h>

/* ---------------------------------------------------------------------------
 * Slice 4e, the WORKER RECEIVE LOOP (L238).
 *
 * A rank above 0 drives nothing of its own: it never opens a listener and
 * never reads a prompt.  It runs this one loop -- receive a frame from the
 * leader, look the session up in a registry keyed by the CREATE ORDINAL,
 * apply the operation through the session member, ack -- until the leader
 * sends STOP or the transport dies.  "Only the leader's arguments are
 * authoritative" is therefore literally true: a worker has no arguments.
 *
 * The ordinal is the key because both ranks hand it out in the same order
 * (pulsar_session::create, ++tp_session_seq); a frame naming a session this
 * rank never created is a divergence and is refused by name.
 *
 * Acked frames (create, sync, eval, batched and mixed decode, bank restore /
 * repoint / fork) always answer,
 * even with a refusal: a rank that stays silent hangs the leader into its
 * deadline instead of failing it at once.  Void frames (destroy, rewind,
 * invalidate, rng state) cannot answer; a refusal there marks the pair failed
 * and every later acked frame carries the refusal back.  A pair already
 * marked failed keeps CONSUMING frames and refusing them, so the leader's
 * stream stays aligned with this rank's until STOP arrives.
 * ------------------------------------------------------------------------ */

struct pulsar_tp_worker_slot {
    uint64_t id;            /* the create ordinal the leader names */
    pulsar_session *s;
    float *logits;          /* this rank's decode output; sized on first use */
    uint32_t logits_rows;
    /* Increment 4: one speculative round per bank, created on first use.  The
     * leader keys its rounds by scheduler slot and names the BANK in every
     * round frame; the bank is the identity this rank keys by. */
    pulsar_spec_round **rounds;
    uint32_t n_rounds;
    int *accepted;          /* the accept walk's output, unread here */
    int accepted_cap;
};

static pulsar_tp_worker_slot *worker_find(pulsar_engine *e, uint64_t id) {
    for (uint32_t i = 0; i < e->tp_worker_n; i++)
        if (e->tp_worker_slots[i].id == id) return &e->tp_worker_slots[i];
    return NULL;
}

static pulsar_tp_worker_slot *worker_add(pulsar_engine *e, pulsar_session *s) {
    if (e->tp_worker_n == e->tp_worker_cap) {
        e->tp_worker_cap = e->tp_worker_cap ? e->tp_worker_cap * 2u : 8u;
        e->tp_worker_slots = (pulsar_tp_worker_slot *)xrealloc(
            e->tp_worker_slots, (size_t)e->tp_worker_cap * sizeof(*e->tp_worker_slots));
    }
    pulsar_tp_worker_slot *slot = &e->tp_worker_slots[e->tp_worker_n++];
    memset(slot, 0, sizeof(*slot));
    slot->id = s->tp_session_id;
    slot->s = s;
    return slot;
}

static void worker_drop(pulsar_engine *e, pulsar_tp_worker_slot *slot) {
    for (uint32_t b = 0; b < slot->n_rounds; b++) pulsar_spec_round_free(slot->rounds[b]);
    free(slot->rounds);
    free(slot->accepted);
    slot->s->destroy();
    free(slot->logits);
    *slot = e->tp_worker_slots[e->tp_worker_n - 1u];
    e->tp_worker_n--;
}

/* The decode output buffer this rank writes but nobody reads: the vocab
 * all-gather already left every rank holding the full logits, and sampling is
 * the leader's.  Sized in rows of the engine's logits width. */
static float *worker_logits(pulsar_engine *e, pulsar_tp_worker_slot *slot, uint32_t rows,
                            int *cap_floats) {
    const uint32_t width = (uint32_t)e->logits_width();
    if (rows > slot->logits_rows) {
        slot->logits = (float *)xrealloc(slot->logits, (size_t)rows * width * sizeof(float));
        slot->logits_rows = rows;
    }
    *cap_floats = (int)(slot->logits_rows * width);
    return slot->logits;
}

/* The round for `bank` on this rank, created on first use. */
static pulsar_spec_round *worker_round(pulsar_tp_worker_slot *slot, int bank) {
    if (bank < 0) return NULL;
    if ((uint32_t)bank >= slot->n_rounds) {
        const uint32_t want = (uint32_t)bank + 1u;
        slot->rounds = (pulsar_spec_round **)xrealloc(slot->rounds, (size_t)want * sizeof(*slot->rounds));
        for (uint32_t b = slot->n_rounds; b < want; b++) slot->rounds[b] = NULL;
        slot->n_rounds = want;
    }
    if (!slot->rounds[bank]) slot->rounds[bank] = pulsar_spec_round_new();
    return slot->rounds[bank];
}

static int *worker_accepted(pulsar_tp_worker_slot *slot, int cap) {
    if (cap < 1) cap = 1;
    if (cap > slot->accepted_cap) {
        slot->accepted = (int *)xrealloc(slot->accepted, (size_t)cap * sizeof(int));
        slot->accepted_cap = cap;
    }
    return slot->accepted;
}

/* The worker's own file for a snapshot the leader named: <spill_dir>/tp-<key>.
 * A worker with no spill directory cannot mirror a spill and refuses it. */
static int worker_spill_path(pulsar_engine *e, const char *key, char *out, size_t outlen) {
    if (!e->tp_spill_dir || !key || !key[0]) return 0;
    for (const char *p = key; *p; p++) if (*p == '/' || *p == '\\') return 0;   /* a key, not a path */
    const int w = snprintf(out, outlen, "%s/tp-%s", e->tp_spill_dir, key);
    return w > 0 && (size_t)w < outlen;
}

bool pulsar_engine_is_tp_worker(const pulsar_engine *e) {
    return e && e->tp && pulsar_tp_rank(e->tp) != 0;
}

bool pulsar_engine_is_tp(const pulsar_engine *e) {
    return e && e->tp;
}

struct pulsar_tp *pulsar_engine_tp(const pulsar_engine *e) {
    return e ? e->tp : NULL;
}

/* The two refusals every acked frame shares.  Returns 1 when the frame must be
 * refused (err filled), 0 when the body may run. */
static int worker_refused(pulsar_engine *e, const pulsar_tp_command *c, const char *op,
                          pulsar_tp_worker_slot **slot, char *err, size_t errlen) {
    *slot = NULL;
    if (pulsar_tp_failed(e->tp)) {
        snprintf(err, errlen, "tp: this rank marked the pair failed earlier; refusing to apply %s", op);
        return 1;
    }
    *slot = worker_find(e, c->session_id);
    if (!*slot) {
        snprintf(err, errlen,
                 "tp: the leader mirrored %s for session %llu, which this rank never created; "
                 "the ranks' session order diverged", op, (unsigned long long)c->session_id);
        return 1;
    }
    return 0;
}

/* A MODEL-RUNNING frame this rank refused or failed: the leader is running
 * the step (its exchanges wait on this rank's rows), so the row lane is
 * aborted -- the leader's spinning kernels exit and its step refuses at once,
 * instead of waiting out the transport timeout (L241 4g-2).  The pair is
 * failed from here on, which is what a rank leaving lockstep means. */
static void worker_abort_step(pulsar_engine *e, const char *op, const char *why) {
    char msg[200];
    snprintf(msg, sizeof(msg), "the worker could not run %s: %s", op, why);
    pulsar_tp_row_lane_abort(e->tp, msg);
}

static int worker_ack(pulsar_engine *e, uint64_t sid, int rc, char *err, size_t errlen) {
    if (pulsar_tp_send_command_ack(e->tp, sid, rc) != 0) return 1;
    snprintf(err, errlen, "tp: could not ack the leader (control channel gone)");
    return -1;
}

/* The ack of a logits-producing operation (eval, batch decode, mixed batch):
 * on success it carries the digest of THIS rank's assembled logits, which the
 * leader compares against its own (the cross-rank identity check, L243).  A
 * failure acks its status the plain way -- there are no logits to digest. */
static int worker_ack_logits(pulsar_engine *e, uint64_t sid, int rc, const float *logits,
                             uint32_t n_rows, char *err, size_t errlen) {
    if (rc != 0) return worker_ack(e, sid, rc, err, errlen);
    const uint64_t digest = pulsar_tp_logits_digest(logits, n_rows, (uint32_t)e->logits_width());
    if (pulsar_tp_send_command_ack_digest(e->tp, sid, 0, digest) != 0) return 1;
    snprintf(err, errlen, "tp: could not ack the leader (control channel gone)");
    return -1;
}

static void worker_rows(const pulsar_tp_command *c, pulsar_multiseq_req *rows) {
    for (uint32_t i = 0; i < c->n_items; i++) {
        rows[i].bank  = (uint32_t)c->items[i].bank;
        rows[i].pos   = c->items[i].pos;
        rows[i].token = c->items[i].token;
    }
}

int pulsar_tp_worker_dispatch(pulsar_engine *e, const pulsar_tp_command *c, char *err, size_t errlen) {
    pulsar_tp *tp = e->tp;
    pulsar_tp_worker_slot *slot = NULL;
    char ferr[512];
    ferr[0] = '\0';
    switch (c->type) {
    case PULSAR_TP_FRAME_STOP:
        return 0;

    case PULSAR_TP_FRAME_SESSION_CREATE: {
        int rc = 1;
        if (pulsar_tp_failed(tp)) {
            snprintf(ferr, sizeof(ferr), "tp: this rank marked the pair failed earlier; refusing session create");
        } else if (c->seq == 0 || c->seq > PULSAR_MSEQ_MAX) {
            snprintf(ferr, sizeof(ferr), "tp: session create carries bank pool %llu (want 1..%u)",
                     (unsigned long long)c->seq, (unsigned)PULSAR_MSEQ_MAX);
        } else {
            /* The leader's pool size is the authority (its server sized it at
             * startup); set it for this create exactly as the server does. */
            gpu_graph_bank_pool_set((uint32_t)c->seq);
            pulsar_session *s = NULL;
            rc = pulsar_session::create(&s, e, c->value);
            if (rc != 0 || !s) {
                snprintf(ferr, sizeof(ferr), "tp: session create (ctx %d) failed on this rank", c->value);
                rc = 1;
            } else if (s->tp_session_id != c->session_id) {
                /* Both ranks hand the ordinal out in creation order; a mismatch
                 * means the leader created a session this rank did not see, or
                 * the reverse.  Nothing after it could be trusted. */
                snprintf(ferr, sizeof(ferr),
                         "tp: the leader created session %llu but this rank's create ordinal is %llu; "
                         "the ranks' session order diverged",
                         (unsigned long long)c->session_id, (unsigned long long)s->tp_session_id);
                s->destroy();
                rc = 1;
            } else {
                worker_add(e, s);
            }
        }
        if (rc != 0) {
            fprintf(stderr, "pulsar: %s\n", ferr);
            pulsar_tp_mark_failed(tp);
        }
        return worker_ack(e, c->session_id, rc, err, errlen);
    }

    case PULSAR_TP_FRAME_SESSION_DESTROY:
        slot = worker_find(e, c->session_id);
        if (!slot) {
            pulsar_tp_mirror_fail_void(tp, "session destroy", "the session was never created on this rank");
            return 1;
        }
        worker_drop(e, slot);
        return 1;

    case PULSAR_TP_FRAME_SYNC:
    case PULSAR_TP_FRAME_SYNC_MM: {
        /* SYNC_MM carries the images too: this rank's own tower encodes the
         * same bytes at the same start positions the leader's did. */
        int rc = 1;
        if (!worker_refused(e, c, "sync", &slot, ferr, sizeof(ferr))) {
            pulsar_tokens borrowed;
            borrowed.v = c->tokens;
            borrowed.len = (int)c->n_tokens;
            borrowed.cap = (int)c->n_tokens;
            rc = slot->s->sync(&borrowed, c->n_images ? c->images : NULL, (int)c->n_images, ferr, sizeof(ferr));
        }
        /* INTERRUPTED is the leader's chunk verdict (v15), taken at the same
         * boundary on both ranks: an outcome, not a refusal. */
        if (rc != 0 && rc != PULSAR_SESSION_SYNC_INTERRUPTED) {
            fprintf(stderr, "pulsar: tp worker: sync refused: %s\n", ferr);
            worker_abort_step(e, "sync", ferr);
        }
        return worker_ack(e, c->session_id, rc, err, errlen);
    }

    case PULSAR_TP_FRAME_EVAL: {
        int rc = 1;
        if (!worker_refused(e, c, "eval", &slot, ferr, sizeof(ferr))) {
            const uint64_t pos = (uint64_t)slot->s->checkpoint.len;
            if (c->seq != pos) {
                snprintf(ferr, sizeof(ferr),
                         "tp: the leader evaluated at position %llu but this rank is at %llu; "
                         "the ranks are out of lockstep, refusing to decode",
                         (unsigned long long)c->seq, (unsigned long long)pos);
            } else {
                const double t0 = pulsar_tp_now_sec();   /* TEMPORARY instrument */
                rc = slot->s->eval(c->value, ferr, sizeof(ferr));
                pulsar_tp_timing_add(PULSAR_TP_TSITE_STEP, PULSAR_TP_TPH_XCHG, pulsar_tp_now_sec() - t0, 0);
            }
        }
        if (rc != 0) {
            fprintf(stderr, "pulsar: tp worker: eval refused: %s\n", ferr);
            worker_abort_step(e, "eval", ferr);
        }
        return worker_ack_logits(e, c->session_id, rc, rc == 0 ? slot->s->logits : NULL, 1u,
                                 err, errlen);
    }

    case PULSAR_TP_FRAME_EVAL_BATCH:
    case PULSAR_TP_FRAME_MIXED_BATCH: {
        const bool mixed = c->type == PULSAR_TP_FRAME_MIXED_BATCH;
        const char *op = mixed ? "mixed batch" : "batch decode";
        int rc = 1;
        float *logits = NULL;
        uint32_t out_rows = 0;   /* the rows the step headed: n_items, or the mixed step's count */
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            pulsar_multiseq_req *rows = (pulsar_multiseq_req *)xmalloc((size_t)c->n_items * sizeof(*rows));
            worker_rows(c, rows);
            int cap = 0;
            logits = worker_logits(e, slot, c->n_items, &cap);
            if (mixed) {
                rc = slot->s->decode_mixed(rows, c->n_items, logits, cap, &out_rows,
                                           (uint32_t)c->value, ferr, sizeof(ferr));
            } else {
                rc = slot->s->decode_multiseq(rows, c->n_items, logits, cap, ferr, sizeof(ferr));
                out_rows = c->n_items;
            }
            free(rows);
        }
        if (rc != 0) {
            fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
            worker_abort_step(e, op, ferr);
            return worker_ack(e, c->session_id, rc, err, errlen);
        }
        /* The digest of what the step PRODUCED (argmax / compact / logits rows),
         * the same helper the leader's collect uses. */
        if (pulsar_tp_send_command_ack_digest(e->tp, c->session_id, 0,
                                              pulsar_session_batch_digest(slot->s, logits, out_rows)) != 0)
            return 1;
        snprintf(err, errlen, "tp: could not ack the leader (control channel gone)");
        return -1;
    }

    case PULSAR_TP_FRAME_REWIND:
    case PULSAR_TP_FRAME_INVALIDATE: {
        const char *op = c->type == PULSAR_TP_FRAME_REWIND ? "rewind" : "invalidate";
        if (worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, op, ferr);
            return 1;
        }
        if (c->type == PULSAR_TP_FRAME_REWIND) slot->s->rewind(c->value);
        else                                   slot->s->invalidate();
        return 1;
    }

    case PULSAR_TP_FRAME_BANK_STATE_SAVE:
        if (worker_refused(e, c, "bank state save", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "bank state save", ferr);
            return 1;
        }
        slot->s->bank_state_save((uint32_t)c->value);
        return 1;

    case PULSAR_TP_FRAME_BANK_STATE_RESTORE:
    case PULSAR_TP_FRAME_BANK_REPOINT: {
        /* A VERDICT frame: the ack carries this rank's result and the leader
         * requires every rank's to agree with its own.  A negative status is
         * this rank's refusal, never a verdict. */
        const bool restore = c->type == PULSAR_TP_FRAME_BANK_STATE_RESTORE;
        const char *op = restore ? "bank state restore" : "bank repoint";
        int status = -1;
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            status = restore ? (slot->s->bank_state_restore((uint32_t)c->value) ? 0 : 1)
                             : slot->s->bank_repoint((uint32_t)c->value);
            if (status < 0) status = 1;
        } else {
            fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    case PULSAR_TP_FRAME_BANK_FORK:
    case PULSAR_TP_FRAME_BANK_FORK_PARTIAL: {
        const bool partial = c->type == PULSAR_TP_FRAME_BANK_FORK_PARTIAL;
        const char *op = partial ? "partial bank fork" : "bank fork";
        int status = -1;
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            status = partial
                ? slot->s->bank_fork_partial((uint32_t)c->bank_src, (uint32_t)c->bank_dst,
                                             c->tokens, (int)c->n_tokens, c->n_cached)
                : slot->s->bank_fork((uint32_t)c->bank_src, (uint32_t)c->bank_dst,
                                     c->tokens, (int)c->n_tokens, c->n_cached);
            if (status < 0) status = PULSAR_FORK_EINVAL;   /* a verdict is never negative */
        } else {
            fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    case PULSAR_TP_FRAME_REWRITE_FROM_COMMON: {
        /* A verdict frame whose enum includes -1: the wire status is result + 1
         * so that a negative wire status stays this rank's refusal. */
        int status = -1;
        if (!worker_refused(e, c, "rewrite from common", &slot, ferr, sizeof(ferr))) {
            pulsar_tokens borrowed;
            borrowed.v = c->tokens;
            borrowed.len = (int)c->n_tokens;
            borrowed.cap = (int)c->n_tokens;
            const pulsar_session_rewrite_result rr =
                slot->s->rewrite_from_common(&borrowed, c->value, ferr, sizeof(ferr));
            status = (int)rr + 1;
            if (status < 0) status = 0;   /* an unknown negative result reads as ERROR */
        } else {
            fprintf(stderr, "pulsar: tp worker: rewrite from common refused: %s\n", ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    case PULSAR_TP_FRAME_NOTE_COMMITTED:
        if (worker_refused(e, c, "note committed tokens", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "note committed tokens", ferr);
            return 1;
        }
        slot->s->note_committed_tokens(c->tokens, (int)c->n_tokens);
        return 1;

    case PULSAR_TP_FRAME_SET_LOGITS: {
        int status = -1;
        if (!worker_refused(e, c, "set logits", &slot, ferr, sizeof(ferr))) {
            status = slot->s->set_logits(c->logits, (int)c->n_logits) != 0 ? 1 : 0;
        } else {
            fprintf(stderr, "pulsar: tp worker: set logits refused: %s\n", ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    /* ---- the speculative round family (increment 4).  The rng each call
     * consumes arrived on the frame; the logits block is this rank's own from
     * the mirrored forward (identical after the vocab all-gather). ---- */
    case PULSAR_TP_FRAME_SPEC_NEXT_BASE: {
        int status = -1;
        if (!worker_refused(e, c, "spec_next_base", &slot, ferr, sizeof(ferr))) {
            uint64_t rng = c->spec.rng;
            const int first = pulsar_session_spec_next_base_local(slot->s, c->spec.temperature, c->spec.top_k,
                                                                  c->spec.top_p, c->spec.min_p, &rng);
            status = first + 1;
            if (status < 0) status = 0;
        } else fprintf(stderr, "pulsar: tp worker: spec_next_base refused: %s\n", ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SPEC_ROUND_BEGIN: {
        int status = -1;
        if (!worker_refused(e, c, "spec_round_begin", &slot, ferr, sizeof(ferr))) {
            pulsar_spec_round *r = worker_round(slot, c->spec.bank);
            const int rc = r ? pulsar_session_spec_round_begin_local(slot->s, r, c->spec.i0, c->spec.i1, c->spec.i2,
                                                                     c->spec.temperature, c->spec.top_k, c->spec.top_p,
                                                                     c->spec.min_p, ferr, sizeof(ferr)) : -1;
            status = rc == 0 ? 0 : 1;
            if (rc != 0) fprintf(stderr, "pulsar: tp worker: spec_round_begin failed: %s\n", ferr);
        } else fprintf(stderr, "pulsar: tp worker: spec_round_begin refused: %s\n", ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SPEC_ARM_CAPTURE:
        if (worker_refused(e, c, "spec_arm_capture", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "spec_arm_capture", ferr);
            return 1;
        }
        pulsar_session_spec_arm_capture_local(slot->s, (uint32_t)c->spec.i0);
        return 1;
    case PULSAR_TP_FRAME_SPEC_ROUND_END: {
        int status = -1;
        if (!worker_refused(e, c, "spec_round_end", &slot, ferr, sizeof(ferr))) {
            pulsar_spec_round *r = worker_round(slot, c->spec.bank);
            const uint32_t width = (uint32_t)e->logits_width();
            if (!r || !slot->logits || (uint64_t)(c->spec.i3 + 1) * width > (uint64_t)slot->logits_rows * width) {
                snprintf(ferr, sizeof(ferr), "tp: spec_round_end for bank %d has no round or no forward block on this rank", c->spec.bank);
                fprintf(stderr, "pulsar: tp worker: %s\n", ferr);
            } else {
                uint64_t rng = c->spec.rng;
                int *acc = worker_accepted(slot, c->spec.i2);
                const int na = pulsar_session_spec_round_end_local(slot->s, r, c->spec.i0, c->spec.i1,
                                                                   c->spec.temperature, c->spec.top_k, c->spec.top_p,
                                                                   c->spec.min_p, &rng, slot->logits, (uint32_t)c->spec.i3,
                                                                   acc, c->spec.i2, ferr, sizeof(ferr));
                status = na + 1;
                if (status < 0) status = 0;
                if (na < 0) fprintf(stderr, "pulsar: tp worker: spec_round_end failed: %s\n", ferr);
            }
        } else fprintf(stderr, "pulsar: tp worker: spec_round_end refused: %s\n", ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SPEC_ROUND_ABORT: {
        if (worker_refused(e, c, "spec_round_abort", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "spec_round_abort", ferr);
            return 1;
        }
        pulsar_spec_round *r = worker_round(slot, c->spec.bank);
        if (r) pulsar_session_spec_round_abort_local(slot->s, r);
        return 1;
    }
    case PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH: {
        int status = -1;
        if (!worker_refused(e, c, "spec_redraft_batch", &slot, ferr, sizeof(ferr))) {
            const uint32_t n = c->spec.count;
            pulsar_spec_round **rounds = (pulsar_spec_round **)xmalloc((n ? n : 1u) * sizeof(*rounds));
            uint64_t *states = (uint64_t *)xmalloc((n ? n : 1u) * sizeof(*states));
            uint64_t **rngs = (uint64_t **)xmalloc((n ? n : 1u) * sizeof(*rngs));
            int ok = 1;
            for (uint32_t i = 0; i < n; i++) {
                rounds[i] = worker_round(slot, (int)c->spec_banks[i]);
                states[i] = c->spec_rngs[i];
                rngs[i] = &states[i];
                if (!rounds[i]) ok = 0;
            }
            const int rc = ok ? pulsar_session_spec_redraft_batch_local(slot->s, rounds, c->spec_banks, rngs, (int)n,
                                                                        ferr, sizeof(ferr)) : -1;
            free(rounds); free(states); free(rngs);
            status = rc == 0 ? 0 : 1;
            if (rc != 0) fprintf(stderr, "pulsar: tp worker: spec_redraft_batch failed: %s\n", ferr);
        } else fprintf(stderr, "pulsar: tp worker: spec_redraft_batch refused: %s\n", ferr);
        if (status != 0) worker_abort_step(e, "spec_redraft_batch", ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT: {
        if (worker_refused(e, c, "spec_redraft_commit", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "spec_redraft_commit", ferr);
            return 1;
        }
        pulsar_spec_round *r = worker_round(slot, c->spec.bank);
        if (r) pulsar_session_spec_redraft_commit_local(slot->s, r);
        return 1;
    }
    case PULSAR_TP_FRAME_GENERATE_SPECULATIVE: {
        /* The CLI's whole loop as ONE frame: with the leader's rng and the
         * same state the member's accept walk is deterministic, so the
         * verdict is the token count. */
        int status = -1;
        if (!worker_refused(e, c, "generate_speculative", &slot, ferr, sizeof(ferr))) {
            uint64_t rng = c->spec.rng;
            int *acc = worker_accepted(slot, c->spec.i2);
            const int n = slot->s->generate_speculative(c->spec.temperature, c->spec.top_k, c->spec.top_p, c->spec.min_p,
                                                        &rng, c->spec.i0, c->spec.i1, acc, c->spec.i2, ferr, sizeof(ferr));
            status = n + 1;
            if (status < 0) status = 0;
            if (n < 0) {
                fprintf(stderr, "pulsar: tp worker: generate_speculative failed: %s\n", ferr);
                worker_abort_step(e, "generate_speculative", ferr);
            } else {
                /* The run's last row is this rank's assembled logits after the
                 * whole loop; the positive verdict carries its digest (L243). */
                const uint64_t digest = pulsar_tp_logits_digest(slot->s->logits, 1u,
                                                                (uint32_t)e->logits_width());
                if (pulsar_tp_send_command_ack_digest(e->tp, c->session_id, status, digest) != 0)
                    return 1;
                snprintf(err, errlen, "tp: could not ack the leader (control channel gone)");
                return -1;
            }
        } else {
            fprintf(stderr, "pulsar: tp worker: generate_speculative refused: %s\n", ferr);
            worker_abort_step(e, "generate_speculative", ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    /* ---- the eviction guard's spill path (increment 6) ---- */
    case PULSAR_TP_FRAME_BANK_FREE_PHYSICAL:
    case PULSAR_TP_FRAME_BANK_ALLOC_PHYSICAL: {
        const bool freeing = c->type == PULSAR_TP_FRAME_BANK_FREE_PHYSICAL;
        const char *op = freeing ? "bank free physical" : "bank alloc physical";
        int status = -1;
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            const bool okb = freeing ? slot->s->bank_free_physical((uint32_t)c->value)
                                     : slot->s->bank_alloc_physical((uint32_t)c->value);
            status = okb ? 0 : 1;
        } else fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_BANK_KV_SAVE:
    case PULSAR_TP_FRAME_BANK_KV_LOAD: {
        const bool load = c->type == PULSAR_TP_FRAME_BANK_KV_LOAD;
        const char *op = load ? "bank kv load" : "bank kv save";
        int status = -1;
        char path[4600];
        if (worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        } else if (!worker_spill_path(e, c->spill_key, path, sizeof(path))) {
            /* No spill directory on this rank, or a key that is not a key:
             * the leader's own save succeeded, so this MUST read as a split
             * verdict, not a quiet success. */
            fprintf(stderr, "pulsar: tp worker: %s refused: no spill directory on this rank "
                            "(pulsar_engine_options.tp_spill_dir) for key '%s'\n", op, c->spill_key);
            status = 1;
        } else if (load) {
            FILE *fp = fopen(path, "rb");
            if (!fp) {
                fprintf(stderr, "pulsar: tp worker: %s: cannot open %s\n", op, path);
                status = 1;
            } else {
                status = slot->s->bank_kv_load((uint32_t)c->value, fp, ferr, sizeof(ferr)) == 0 ? 0 : 1;
                fclose(fp);
                if (status) fprintf(stderr, "pulsar: tp worker: %s failed: %s\n", op, ferr);
            }
        } else {
            /* The server's own recipe: write a temp beside the target, fsync,
             * rename, so a crash never leaves a torn snapshot under the key. */
            char tmp[4700];
            snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
            FILE *fp = fopen(tmp, "wb");
            if (!fp) {
                fprintf(stderr, "pulsar: tp worker: %s: cannot create %s\n", op, tmp);
                status = 1;
            } else {
                const int rc = slot->s->bank_kv_save((uint32_t)c->value, fp, ferr, sizeof(ferr));
                const bool synced = rc == 0 && fflush(fp) == 0 && fsync(fileno(fp)) == 0;
                const int fc = fclose(fp);
                if (!synced || fc != 0 || rename(tmp, path) != 0) {
                    remove(tmp);
                    fprintf(stderr, "pulsar: tp worker: %s failed: %s\n", op, rc ? ferr : "flush/rename");
                    status = 1;
                } else {
                    status = 0;
                }
            }
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

    default:
        snprintf(err, errlen, "tp: the leader sent frame type %d, which a worker does not apply", (int)c->type);
        pulsar_tp_mark_failed(tp);
        return -1;
    }
}

int pulsar_tp_worker_run(pulsar_engine *e, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!pulsar_engine_is_tp_worker(e)) {
        if (err) snprintf(err, errlen, "tp: this rank is not a worker (rank 0 or no pair); nothing to run");
        return 1;
    }
    fprintf(stderr, "pulsar: TP worker rank %d/%u: driving sessions from the leader's frames\n",
            pulsar_tp_rank(e->tp), pulsar_tp_n_ranks(e->tp));
    int rc = 0;
    for (;;) {
        pulsar_tp_command c;
        memset(&c, 0, sizeof(c));
        char rerr[512];
        rerr[0] = '\0';
        if (pulsar_tp_recv_command(e->tp, &c, rerr, sizeof(rerr)) == 0) {
            if (err) snprintf(err, errlen, "%s", rerr);
            rc = 1;
            break;
        }
        char derr[512];
        derr[0] = '\0';
        const int d = pulsar_tp_worker_dispatch(e, &c, derr, sizeof(derr));
        pulsar_tp_command_free(&c);
        if (d == 0) break;
        if (d < 0) {
            if (err) snprintf(err, errlen, "%s", derr);
            rc = 1;
            break;
        }
    }
    /* Whatever the leader left open dies with the loop: a worker has no other
     * owner for a session. */
    while (e->tp_worker_n) worker_drop(e, &e->tp_worker_slots[e->tp_worker_n - 1u]);
    free(e->tp_worker_slots);
    e->tp_worker_slots = NULL;
    e->tp_worker_cap = 0;
    fprintf(stderr, "pulsar: TP worker rank %d: %s\n", pulsar_tp_rank(e->tp),
            rc == 0 ? "stopped by the leader" : (err && err[0] ? err : "stopped on a failure"));
    return rc;
}

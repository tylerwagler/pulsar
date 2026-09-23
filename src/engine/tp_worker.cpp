#include "pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"

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
    uint64_t rng;           /* the leader's speculation stream, once shipped */
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

bool pulsar_engine_is_tp_worker(const pulsar_engine *e) {
    return e && e->tp && pulsar_tp_rank(e->tp) != 0;
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

static int worker_ack(pulsar_engine *e, uint64_t sid, int rc, char *err, size_t errlen) {
    if (pulsar_tp_send_command_ack(e->tp, sid, rc) != 0) return 1;
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
        } else {
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

    case PULSAR_TP_FRAME_SYNC: {
        int rc = 1;
        if (!worker_refused(e, c, "sync", &slot, ferr, sizeof(ferr))) {
            pulsar_tokens borrowed;
            borrowed.v = c->tokens;
            borrowed.len = (int)c->n_tokens;
            borrowed.cap = (int)c->n_tokens;
            rc = slot->s->sync(&borrowed, NULL, 0, ferr, sizeof(ferr));
        }
        if (rc != 0) fprintf(stderr, "pulsar: tp worker: sync refused: %s\n", ferr);
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
                rc = slot->s->eval(c->value, ferr, sizeof(ferr));
            }
        }
        if (rc != 0) fprintf(stderr, "pulsar: tp worker: eval refused: %s\n", ferr);
        return worker_ack(e, c->session_id, rc, err, errlen);
    }

    case PULSAR_TP_FRAME_EVAL_BATCH:
    case PULSAR_TP_FRAME_MIXED_BATCH: {
        const bool mixed = c->type == PULSAR_TP_FRAME_MIXED_BATCH;
        const char *op = mixed ? "mixed batch" : "batch decode";
        int rc = 1;
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            pulsar_multiseq_req *rows = (pulsar_multiseq_req *)xmalloc((size_t)c->n_items * sizeof(*rows));
            worker_rows(c, rows);
            int cap = 0;
            float *logits = worker_logits(e, slot, c->n_items, &cap);
            if (mixed) {
                uint32_t out_rows = 0;
                rc = slot->s->decode_mixed(rows, c->n_items, logits, cap, &out_rows,
                                           (uint32_t)c->value, ferr, sizeof(ferr));
            } else {
                rc = slot->s->decode_multiseq(rows, c->n_items, logits, cap, ferr, sizeof(ferr));
            }
            free(rows);
        }
        if (rc != 0) fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        return worker_ack(e, c->session_id, rc, err, errlen);
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

    case PULSAR_TP_FRAME_RNG_STATE:
        if (worker_refused(e, c, "rng sync", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "rng sync", ferr);
            return 1;
        }
        /* The leader's stream becomes this rank's for the next speculative
         * round; the round frames (a later increment) draw from it. */
        slot->rng = c->seq;
        slot->s->spec_rng_synced = true;
        return 1;

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

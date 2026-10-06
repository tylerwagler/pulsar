#include "pulsar_engine_internal.h"
#include "tp/pulsar_tp.h"
#include "pulsar_writeback.h"

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---------------------------------------------------------------------------
 * Slice 4e, the WORKER RECEIVE LOOP (L238).
 *
 * A rank above 0 drives nothing of its own: it never opens a listener and
 * never reads a prompt.  It runs this one loop -- receive a frame from the
 * leader, look the session up in a registry keyed by the CREATE ORDINAL,
 * apply the operation through the family's operation (the leader's own dispatch, L266), ack -- until the leader
 * sends STOP or the transport dies.  "Only the leader's arguments are
 * authoritative" is therefore literally true: a worker has no arguments.
 *
 * The ordinal is the key because both ranks hand it out in the same order
 * (pulsar_session::create, ++tp_session_seq); a frame naming a session this
 * rank never created is a divergence and is refused by name.
 *
 * Acked frames (create, sync, eval, batched and mixed decode, bank restore /
 * repoint) always answer,
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

/* L264 S4e: this rank's copy of a disk KV segment the leader's store names. */
static bool worker_segment_path(pulsar_engine *e, const char *key, char *out, size_t outlen) {
    if (!e->tp_kv_dir || !pulsar_tp_segment_key_ok(key)) return false;
    const int w = snprintf(out, outlen, "%s/%.40s.tpseg", e->tp_kv_dir, key);
    return w > 0 && (size_t)w < outlen;
}

static bool worker_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char buf[4096];
    const size_t n = strlen(path);
    if (n >= sizeof(buf)) return false;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\0') {
            const char saved = buf[i];
            buf[i] = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
            buf[i] = saved;
        }
    }
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* The installed bank's segment [G_prev, G), durably: tmp + fsync + rename, and
 * out of the page cache (L261). */
static bool worker_segment_save(pulsar_session *s, const char *path, int G_prev, int G, char *err, size_t errlen) {
    char tmp[4700];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        snprintf(err, errlen, "cannot create %.400s: %s", tmp, strerror(errno));
        return false;
    }
    err[0] = '\0';
    bool ok = s->save_segment(fp, (uint32_t)G_prev, (uint32_t)G, err, errlen) == 0 &&
              fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    if (ok) pulsar_writeback_drop_file(fp);
    const int saved_errno = errno;
    if (fclose(fp) != 0) ok = false;
    if (ok && rename(tmp, path) != 0) ok = false;
    if (!ok) {
        remove(tmp);
        if (!err[0]) snprintf(err, errlen, "writing %.400s failed: %s", path, strerror(saved_errno ? saved_errno : errno));
    }
    return ok;
}

/* Load this rank's copy and reach the state the leader states: the same span
 * end and the same history digest. */
static bool worker_segment_load(pulsar_session *s, const char *path, const pulsar_tp_segment_command *cmd,
                                char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "no copy %.400s: %s", path, strerror(errno));
        return false;
    }
    struct stat st;
    uint32_t G = 0;
    err[0] = '\0';
    bool ok = fstat(fileno(fp), &st) == 0 &&
              s->load_segment(fp, (uint64_t)st.st_size, cmd->last != 0, &G, err, errlen) == 0;
    fclose(fp);
    if (ok && ((int)G != cmd->G || pulsar_session_checkpoint_digest(s) != cmd->digest)) {
        snprintf(err, errlen, "the copy loaded to %u (digest %016llx); the leader stands at %d (digest %016llx)",
                 G, (unsigned long long)pulsar_session_checkpoint_digest(s), cmd->G,
                 (unsigned long long)cmd->digest);
        ok = false;
    }
    if (!ok && !err[0]) snprintf(err, errlen, "loading %.400s failed", path);
    return ok;
}

/* Bring-up: copies the leader's store does not name can never be loaded (the
 * leader decides every load) -- reclaim them, and any abandoned temp file. */
static void worker_segment_reconcile(pulsar_engine *e, const char *keys, uint32_t n_keys) {
    if (!e->tp_kv_dir) return;
    DIR *d = opendir(e->tp_kv_dir);
    if (!d) return;
    int kept = 0, removed = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *name = de->d_name;
        const size_t len = strlen(name);
        bool keep = false;
        if (len == 46 && !strcmp(name + 40, ".tpseg")) {
            for (uint32_t i = 0; i < n_keys && !keep; i++) keep = !memcmp(keys + (size_t)i * 40u, name, 40);
        } else if (!strstr(name, ".tpseg.tmp.")) {
            continue;   /* not ours */
        }
        if (keep) { kept++; continue; }
        char path[4600];
        snprintf(path, sizeof(path), "%s/%s", e->tp_kv_dir, name);
        if (unlink(path) == 0) removed++;
    }
    closedir(d);
    fprintf(stderr, "pulsar: tp worker: kv segments reconciled: %d kept, %d removed (the leader holds %u)\n",
            kept, removed, n_keys);
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
            rc = e->family->session->sync(slot->s, &borrowed, c->n_images ? c->images : NULL, (int)c->n_images, ferr,
                                         sizeof(ferr));
        }
        /* INTERRUPTED is the leader's chunk verdict (v15), taken at the same
         * boundary on both ranks: an outcome, not a refusal. */
        if (rc != 0 && rc != PULSAR_SESSION_SYNC_INTERRUPTED) {
            fprintf(stderr, "pulsar: tp worker: sync refused: %s\n", ferr);
            worker_abort_step(e, "sync", ferr);
        }
        return worker_ack(e, c->session_id, rc, err, errlen);
    }

    case PULSAR_TP_FRAME_SYNC_CHECK: {
        /* L250: the leader's cached position + prefix digest, sent before it
         * computes anything; this rank answers whether its own state matches.
         * Nothing is running yet, so a mismatch is a plain refusal (the leader
         * then invalidates every rank) -- no step to abort, the pair stays up. */
        int status = 1;
        if (!worker_refused(e, c, "sync check", &slot, ferr, sizeof(ferr))) {
            const int pos = slot->s->checkpoint.len;
            const uint64_t digest = pulsar_session_checkpoint_digest(slot->s);
            if (pos == c->value && digest == c->seq) {
                status = 0;
            } else {
                fprintf(stderr, "pulsar: tp worker: sync check: this rank's cached state differs "
                                "from the leader's -- leader pos %d digest %016llx, this rank pos %d "
                                "digest %016llx; refusing the sync (L250)\n",
                        c->value, (unsigned long long)c->seq, pos, (unsigned long long)digest);
            }
        } else {
            fprintf(stderr, "pulsar: tp worker: sync check refused: %s\n", ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
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
                rc = e->family->session->eval(slot->s, c->value, ferr, sizeof(ferr));
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
                rc = e->family->session->decode_mixed(slot->s, rows, c->n_items, logits, cap, &out_rows,
                                                      (uint32_t)c->value, ferr, sizeof(ferr));
            } else {
                rc = e->family->session->decode_multiseq(slot->s, rows, c->n_items, logits, cap, ferr, sizeof(ferr));
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

    case PULSAR_TP_FRAME_FUSED_BATCH: {
        int rc = 1;
        float *logits = NULL;
        uint32_t out_rows = 0;
        const pulsar_fused_shape *shape = &c->fused;
        if (!worker_refused(e, c, "fused batch", &slot, ferr, sizeof(ferr))) {
            pulsar_multiseq_req *rows = (pulsar_multiseq_req *)xmalloc((size_t)c->n_items * sizeof(*rows));
            worker_rows(c, rows);
            /* The block holds the decode rows and the headed runs, not every
             * row: a prompt chunk's rows are not headed. */
            uint32_t heads = 0;
            for (uint32_t r = 0; r < shape->n_pf && r < (uint32_t)sizeof(shape->head_last); r++)
                heads += shape->head_last[r] ? 1u : 0u;
            int cap = 0;
            logits = worker_logits(e, slot, shape->n_dec + heads, &cap);
            rc = slot->s->decode_fused(rows, c->n_items, shape, logits, cap, &out_rows, ferr, sizeof(ferr));
            free(rows);
        }
        if (rc != 0) {
            fprintf(stderr, "pulsar: tp worker: fused batch refused: %s\n", ferr);
            worker_abort_step(e, "fused batch", ferr);
            return worker_ack(e, c->session_id, rc, err, errlen);
        }
        if (pulsar_tp_send_command_ack_digest(e->tp, c->session_id, 0,
                                              pulsar_session_fused_digest(slot->s, logits, shape->n_dec,
                                                                          out_rows - shape->n_dec)) != 0)
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
        else                                   e->family->session->invalidate(slot->s);
        return 1;
    }

    case PULSAR_TP_FRAME_BANK_STATE_SAVE:
        if (worker_refused(e, c, "bank state save", &slot, ferr, sizeof(ferr))) {
            pulsar_tp_mirror_fail_void(tp, "bank state save", ferr);
            return 1;
        }
        if (FAMILY_BANKS(slot->s)) FAMILY_BANKS(slot->s)->save(slot->s, (uint32_t)c->value);
        else                       slot->s->bank_state_save((uint32_t)c->value);
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
            const bool restored = restore && (FAMILY_BANKS(slot->s) ? FAMILY_BANKS(slot->s)->restore(slot->s, (uint32_t)c->value)
                                                                    : slot->s->bank_state_restore((uint32_t)c->value));
            status = restore ? (restored ? 0 : 1)
                             : slot->s->bank_repoint((uint32_t)c->value);
            if (status < 0) status = 1;
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
        if (FAMILY_BANKS(slot->s)) FAMILY_BANKS(slot->s)->note_committed(slot->s, c->tokens, (int)c->n_tokens);
        else                       slot->s->note_committed_tokens(c->tokens, (int)c->n_tokens);
        return 1;

    case PULSAR_TP_FRAME_NOTE_PREFILLED: {
        int status = -1;
        if (!worker_refused(e, c, "note prefilled", &slot, ferr, sizeof(ferr))) {
            status = slot->s->note_prefilled(c->tokens, (int)c->n_tokens, c->value) != 0 ? 1 : 0;
        } else {
            fprintf(stderr, "pulsar: tp worker: note prefilled refused: %s\n", ferr);
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }

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
    /* L260: the batch phases.  Each record is one bank's step; the verdict is
     * the phase's outcome fingerprint over every step (0 = refused here). */
    case PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH:
    case PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH:
    case PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT_BATCH: {
        const char *op = c->type == PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH ? "spec_assemble_batch"
                       : c->type == PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH ? "spec_round_end_batch"
                                                                         : "spec_redraft_commit_batch";
        int status = 0;
        if (!worker_refused(e, c, op, &slot, ferr, sizeof(ferr))) {
            const int n = (int)c->spec.count;
            pulsar_spec_step steps[PULSAR_TP_SPEC_STEPS_MAX];
            uint64_t rngs[PULSAR_TP_SPEC_STEPS_MAX];
            int cap_total = 0;
            for (int i = 0; i < n; i++) cap_total += c->spec_steps[i].i2 > 0 ? c->spec_steps[i].i2 : 0;
            int *acc = worker_accepted(slot, cap_total);
            int ok = 1;
            for (int i = 0, off = 0; i < n; i++) {
                const pulsar_tp_spec_command *r = &c->spec_steps[i];
                pulsar_spec_step *st = &steps[i];
                memset(st, 0, sizeof(*st));
                st->bank = (uint32_t)r->bank;
                st->round = worker_round(slot, r->bank);
                st->temperature = r->temperature; st->top_k = r->top_k;
                st->top_p = r->top_p; st->min_p = r->min_p;
                rngs[i] = r->rng;
                st->rng = &rngs[i];
                st->max_tokens = r->i0;
                st->first_token = r->i0;
                st->k_alloc = r->i1;
                st->accepted_cap = r->i2;
                st->accepted = acc + off;
                st->row0 = (uint32_t)r->i3;
                off += r->i2 > 0 ? r->i2 : 0;
                if (!st->round) ok = 0;
                if (c->type == PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH &&
                    (!slot->logits || r->i3 < 0 ||
                     (uint64_t)r->i3 + pulsar_spec_round_n_rows(st->round) > (uint64_t)slot->logits_rows))
                    ok = 0;
            }
            if (!ok) {
                snprintf(ferr, sizeof(ferr), "tp: %s names a bank with no round or rows past this rank's forward block", op);
                fprintf(stderr, "pulsar: tp worker: %s\n", ferr);
            } else if (c->type == PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH) {
                uint32_t rows = 0;
                pulsar_session_spec_assemble_batch_local(slot->s, steps, n, c->spec.i0, (uint32_t)c->spec.i1,
                                                         NULL, &rows);
                status = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_ASSEMBLE, steps, n, rows);
            } else if (c->type == PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH) {
                /* v24: the leader's measured spec cost, the one every rank
                 * prices this round's quench from (L263). */
                pulsar_engine_spec_cost_set(e, c->spec.i1, c->spec.i2, c->spec.i3 != 0);
                pulsar_session_spec_round_end_batch_local(slot->s, steps, n, c->spec.i0, slot->logits);
                status = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_ROUND_END, steps, n, 0u);
            } else {
                pulsar_session_spec_redraft_commit_batch_local(slot->s, steps, n);
                status = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_REDRAFT_COMMIT, steps, n, 0u);
            }
            for (int i = 0; i < n; i++)
                if (steps[i].status == PULSAR_SPEC_STEP_FAILED || steps[i].status == PULSAR_SPEC_STEP_RESTORE_FAILED)
                    fprintf(stderr, "pulsar: tp worker: %s bank %u: status %d %s\n", op, steps[i].bank,
                            steps[i].status, steps[i].err);
        } else fprintf(stderr, "pulsar: tp worker: %s refused: %s\n", op, ferr);
        return worker_ack(e, c->session_id, status, err, errlen);
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

    /* ---- the eviction guard's physical-bank pair (increment 6) ---- */
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
    /* ---- L264 S4e: the disk KV cache's segment chains, one copy per rank ---- */
    case PULSAR_TP_FRAME_SEGMENT_SAVE: {
        /* A failed store is a MISS, never a divergence: the leader skips the
         * segment on every rank and the pair stays up. */
        int status = 1;
        char path[4600];
        if (worker_refused(e, c, "kv segment save", &slot, ferr, sizeof(ferr))) {
            status = -1;   /* a refusal (the ranks diverged), not a miss */
            fprintf(stderr, "pulsar: tp worker: kv segment save refused: %s\n", ferr);
        } else if (!worker_segment_path(e, c->segment.key, path, sizeof(path))) {
            fprintf(stderr, "pulsar: tp worker: kv segment save refused: no segment directory on this rank "
                            "(pulsar_engine_options.tp_kv_dir)\n");
        } else if (!worker_segment_save(slot->s, path, c->segment.G_prev, c->segment.G, ferr, sizeof(ferr))) {
            fprintf(stderr, "pulsar: tp worker: kv segment save failed: %s\n", ferr);
        } else {
            status = 0;
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SEGMENT_LOAD: {
        /* The leader loaded first and states the result; a copy that cannot
         * reach it is a MISS -- the leader invalidates every rank and drops the
         * chain. */
        int status = 1;
        char path[4600];
        if (worker_refused(e, c, "kv segment load", &slot, ferr, sizeof(ferr))) {
            status = -1;   /* a refusal (the ranks diverged), not a miss */
            fprintf(stderr, "pulsar: tp worker: kv segment load refused: %s\n", ferr);
        } else if (!worker_segment_path(e, c->segment.key, path, sizeof(path))) {
            fprintf(stderr, "pulsar: tp worker: kv segment load refused: no segment directory on this rank\n");
        } else if (!worker_segment_load(slot->s, path, &c->segment, ferr, sizeof(ferr))) {
            fprintf(stderr, "pulsar: tp worker: kv segment load miss (%.40s): %s\n", c->segment.key, ferr);
        } else {
            status = 0;
        }
        return worker_ack(e, c->session_id, status, err, errlen);
    }
    case PULSAR_TP_FRAME_SEGMENT_DROP: {
        /* void: a copy the leader's store no longer names. */
        char path[4600];
        if (worker_segment_path(e, c->segment.key, path, sizeof(path))) (void)unlink(path);
        return 1;
    }
    case PULSAR_TP_FRAME_SEGMENT_RECONCILE:
        worker_segment_reconcile(e, c->keys ? c->keys : "", c->n_keys);
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
    if (e->tp_kv_dir && !worker_mkdir_p(e->tp_kv_dir)) {
        fprintf(stderr, "pulsar: TP worker: cannot create the segment directory %s (%s); mirrored "
                        "disk KV segments will be misses on this rank\n", e->tp_kv_dir, strerror(errno));
    }
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

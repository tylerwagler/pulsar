#include "pulsar_engine_internal.h"
#include "../tp/pulsar_tp.h"
#include <unistd.h>



void pulsar_linux_graph_backend_set_oom_score(pulsar_backend backend) {
#if defined(__linux__)
    static bool attempted = false;
    if (attempted) return;
    attempted = true;

    const int score = 1000;
    FILE *fp = fopen("/proc/self/oom_score_adj", "w");
    if (!fp) {
        fprintf(stderr,
                "pulsar: failed to set Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    if (fprintf(fp, "%d\n", score) < 0) {
        const int err = errno;
        fclose(fp);
        fprintf(stderr,
                "pulsar: failed to write Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(err));
        return;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr,
                "pulsar: failed to close Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    fprintf(stderr,
            "pulsar: Linux %s backend set oom_score_adj=%d\n",
            pulsar_backend_name(backend),
            score);
#else
    (void)backend;
#endif
}



bool pulsar_think_mode_enabled(pulsar_think_mode mode) {
    return mode != PULSAR_THINK_NONE;
}



bool pulsar_think_mode_valid(pulsar_think_mode mode) {
    return mode == PULSAR_THINK_NONE ||
           (mode >= PULSAR_THINK_EFFORT_MIN && mode <= PULSAR_THINK_EFFORT_MAX);
}



/* One row per effort 0..100: [0] the name / "" prefix for thinking-off, the
 * rest the decimal effort and its rendered line.  Built once, read forever;
 * every caller keeps a const char* into it. */
namespace {
struct think_effort_row {
    char name[4];      /* "1".."100" */
    char prefix[160];  /* PULSAR_REASONING_EFFORT_HEAD N PULSAR_REASONING_EFFORT_TAIL */
};
struct think_effort_table {
    think_effort_row row[PULSAR_THINK_EFFORT_MAX + 1];
    think_effort_table() {
        memset(row, 0, sizeof row);
        for (int n = PULSAR_THINK_EFFORT_MIN; n <= PULSAR_THINK_EFFORT_MAX; n++) {
            snprintf(row[n].name, sizeof row[n].name, "%d", n);
            const int w = snprintf(row[n].prefix, sizeof row[n].prefix,
                                   PULSAR_REASONING_EFFORT_HEAD "%d" PULSAR_REASONING_EFFORT_TAIL, n);
            if (w <= 0 || (size_t)w >= sizeof row[n].prefix) pulsar_die("reasoning-effort line does not fit its row");
        }
    }
};
const think_effort_table &think_efforts() {
    static const think_effort_table t;
    return t;
}
}



const char *pulsar_think_mode_name(pulsar_think_mode mode) {
    switch (mode) {
    case PULSAR_THINK_NONE: return "none";
    case PULSAR_THINK_LOW:  return "low";
    case PULSAR_THINK_HIGH: return "high";
    case PULSAR_THINK_MAX:  return "max";
    default: break;
    }
    if (!pulsar_think_mode_valid(mode)) pulsar_die("pulsar_think_mode_name: not a thinking mode");
    return think_efforts().row[mode].name;
}



const char *pulsar_think_effort_prefix(pulsar_think_mode mode) {
    if (mode == PULSAR_THINK_NONE) return "";
    if (!pulsar_think_mode_valid(mode)) pulsar_die("pulsar_think_effort_prefix: not a thinking mode");
    return think_efforts().row[mode].prefix;
}



bool pulsar_think_effort_v4_valid(pulsar_think_mode mode) {
    return mode == PULSAR_THINK_NONE || mode == PULSAR_THINK_LOW ||
           mode == PULSAR_THINK_HIGH || mode == PULSAR_THINK_MAX;
}



const char *pulsar_think_effort_prefix_family(pulsar_think_mode mode, bool v41) {
    if (v41) return pulsar_think_effort_prefix(mode);
    switch (mode) {
    case PULSAR_THINK_NONE:
    case PULSAR_THINK_LOW:  return "";
    case PULSAR_THINK_HIGH: return PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX;
    case PULSAR_THINK_MAX:  return PULSAR_V4_REASONING_EFFORT_MAX_PREFIX;
    default: break;
    }
    pulsar_die("pulsar_think_effort_prefix_family: the V4 encoder has three effort levels "
               "(low, high, max); the API refuses any other before rendering");
    return "";
}



size_t pulsar_think_effort_prefix_len(const char *s) {
    if (!s) return 0;
    /* The V4 (0731) texts first: they start with the same head and no digit. */
    if (!strncmp(s, PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX, sizeof(PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX) - 1))
        return sizeof(PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX) - 1;
    if (!strncmp(s, PULSAR_V4_REASONING_EFFORT_MAX_PREFIX, sizeof(PULSAR_V4_REASONING_EFFORT_MAX_PREFIX) - 1))
        return sizeof(PULSAR_V4_REASONING_EFFORT_MAX_PREFIX) - 1;
    const size_t head = sizeof(PULSAR_REASONING_EFFORT_HEAD) - 1;
    if (strncmp(s, PULSAR_REASONING_EFFORT_HEAD, head) != 0) return 0;
    const char *p = s + head;
    int n = 0, digits = 0;
    while (*p >= '0' && *p <= '9' && digits < 3) { n = n * 10 + (*p - '0'); p++; digits++; }
    if (digits == 0 || n < PULSAR_THINK_EFFORT_MIN || n > PULSAR_THINK_EFFORT_MAX) return 0;
    const size_t tail = sizeof(PULSAR_REASONING_EFFORT_TAIL) - 1;
    if (strncmp(p, PULSAR_REASONING_EFFORT_TAIL, tail) != 0) return 0;
    return (size_t)(p - s) + tail;
}



void pulsar_release_instance_lock(void) {
    if (g_pulsar_lock_fd >= 0) {
        close(g_pulsar_lock_fd);
        g_pulsar_lock_fd = -1;
    }
}



/* Refuse to start a second pulsar/ds4 process.  The model can map tens of GiB,
 * so a stale accidental second run is more dangerous than a normal CLI error. */
void pulsar_acquire_instance_lock(void) {
    const char *path = getenv("PULSAR_LOCK_FILE");
    /* The default lock path stays "/tmp/ds4.lock" DELIBERATELY after the
     * Pulsar rebrand: an old ds4-server and a new pulsar binary must contend
     * on the SAME lock during the transition — two live instances would OOM
     * the GB10. Do not rename this path (or the PULSAR_* env names). */
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "pulsar: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "pulsar: another pulsar/ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "pulsar: another pulsar/ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "pulsar: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "pulsar: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_pulsar_lock_fd = fd;
    atexit(pulsar_release_instance_lock);
}



/* =========================================================================
 * Public free-function facades (C++ port).
 * =========================================================================
 *
 * The whole pulsar_session_* / pulsar_engine_* verb family now lives on the
 * classes in pulsar_engine_internal.h; the public API declared in pulsar.h
 * keeps its exact free-function signatures as one-line forwarders defined
 * here.  Server/CLI/agent/lib and the tests/ gate harnesses keep calling
 * these; engine internals call the members directly.
 *
 * Null-object tolerance: most of the original bodies accepted a NULL object
 * (returning a constant).  Calling a member through a NULL pointer is UB, so
 * each facade keeps that guard HERE, returning the same constant the old
 * body's null path returned.  (The handful of err-writing entry points do
 * not duplicate the err-message formatting for the never-exercised s==NULL
 * case — every live call site holds a valid session; all other argument
 * validation still runs unchanged inside the member body.) */

int pulsar_engine_open(pulsar_engine **out, const pulsar_engine_options *opt) { return pulsar_engine::open(out, opt); }
void pulsar_engine_close(pulsar_engine *e) { if (e) e->destroy(); }
void pulsar_engine_summary(pulsar_engine *e) { e->summary(); }
int pulsar_engine_vocab_size(pulsar_engine *e) { return e ? e->vocab_size() : 0; }
int pulsar_engine_logits_width(const pulsar_engine *e) { return e ? e->logits_width() : 0; }
const char *pulsar_engine_model_name(pulsar_engine *e) { return e->model_name(); }
/* The loaded shape IS the authority (pulsar_select_shape_from_metadata sets it
 * once at load); `e` is taken so a caller must hold the engine it is asking
 * about, exactly like the other engine facts.  One fact, one name: the chat
 * TEMPLATE family, which is what the renderer forks on. */
bool pulsar_engine_chat_v41(const pulsar_engine *e) {
    if (!e) return true;   /* no engine: the compile-time default profile */
    return g_pulsar_shape.variant != PULSAR_VARIANT_V4;
}



pulsar_think_mode pulsar_engine_think_default(const pulsar_engine *e) {
    return pulsar_engine_chat_v41(e) ? PULSAR_THINK_DEFAULT : PULSAR_THINK_LOW;
}
void pulsar_engine_spec_metrics(pulsar_engine *e, pulsar_spec_metrics *out) { if (e) { e->spec_metrics(out); } else if (out) { memset(out, 0, sizeof(*out)); } }
int pulsar_engine_model_id(pulsar_engine *e) { return e ? e->model_id() : (int)PULSAR_MODEL_VARIANT; }
bool pulsar_engine_is_pruned(pulsar_engine *e) { return e ? e->is_pruned() : false; }
uint64_t pulsar_engine_session_cost_bytes(pulsar_engine *e, int ctx_size) { return e ? e->session_cost_bytes(ctx_size) : 0; }
uint64_t pulsar_engine_session_cost_bytes_banked(pulsar_engine *e, int ctx_size, int n_banks) { return e ? e->session_cost_bytes_banked(ctx_size, n_banks) : 0; }
uint32_t pulsar_engine_bank_pool(int *pinned_by_env) { if (pinned_by_env) *pinned_by_env = gpu_graph_bank_pool_env_pinned(); return gpu_graph_bank_pool_n(); }
void pulsar_engine_set_bank_pool(uint32_t n_banks) { gpu_graph_bank_pool_set(n_banks); }
uint64_t pulsar_engine_demand_paged_bytes_per_bank(pulsar_engine *e, int ctx_size) { return e ? e->demand_paged_bytes_per_bank(ctx_size) : 0; }
uint64_t pulsar_engine_weights_resident_bytes(pulsar_engine *e) { return e ? e->weights_resident_bytes() : 0; }
int pulsar_engine_generate_argmax(pulsar_engine *e, const pulsar_tokens *prompt,
                               int n_predict, int ctx_size,
                               pulsar_token_emit_fn emit,
                               pulsar_generation_done_fn done,
                               void *emit_ud,
                               pulsar_session_progress_fn progress,
                               void *progress_ud) { return e->generate_argmax(prompt, n_predict, ctx_size, emit, done, emit_ud, progress, progress_ud); }
int pulsar_engine_collect_imatrix(pulsar_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) { return e ? e->collect_imatrix(dataset_path, output_path, ctx_size, max_prompts, max_tokens) : 1; }
void pulsar_engine_dump_tokens(pulsar_engine *e, const pulsar_tokens *tokens) { e->dump_tokens(tokens); }
int pulsar_engine_routed_quant_bits(pulsar_engine *e) { return e ? e->routed_quant_bits() : 0; }
bool pulsar_engine_has_dspark(pulsar_engine *e) { return e && e->has_dspark(); }

void pulsar_session_set_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_progress(fn, ud); }
void pulsar_session_set_display_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_display_progress(fn, ud); }
void pulsar_session_set_cancel(pulsar_session *s, pulsar_session_cancel_fn fn, void *ud) { if (s) s->set_cancel(fn, ud); }
uint64_t pulsar_session_touched_kv_bytes(const pulsar_session *s) { return s ? s->touched_kv_bytes() : 0; }
bool pulsar_session_bank_is_evicted(const pulsar_session *s, uint32_t bank) { return s ? s->bank_is_evicted(bank) : false; }
uint64_t pulsar_session_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) { return s ? s->bank_touched_kv_bytes(bank) : 0; }
uint64_t pulsar_session_quantum_growth_bytes_per_bank(pulsar_session *s, uint32_t q) { return s->quantum_growth_bytes_per_bank(q); }
/* The bank wrappers are defined with the mirror below (increment 2). */
bool pulsar_session_bank_fork_pinned(const pulsar_session *s, uint32_t bank) { return s ? s->bank_fork_pinned(bank) : false; }

int pulsar_session_bank_fork_partial_feasible(pulsar_session *s, uint32_t src, int n_cached) { return s ? s->bank_fork_partial_feasible(src, n_cached) : PULSAR_FORK_EINVAL; }

/* ---------------------------------------------------------------------------
 * Slice 4e: lockstep mirroring of the session's input -- the LEADER half.
 *
 * Rank 0 drives the pair: every public session operation below ships a frame
 * to the workers, runs its own half, and collects one ack per peer.  A rank
 * above 0 drives nothing -- it runs pulsar_tp_worker_run (tp_worker.cpp), the
 * receive loop that applies the leader's frames to a session registry keyed
 * by the create ordinal.  A worker rank reaching one of these wrappers is
 * therefore a driver bug (its arguments are not authoritative and never were),
 * and is refused by name rather than mirrored.
 *
 * The mirror lives at this, the public API boundary, and not on the member
 * sync()/eval(): those are re-entered from inside a running operation (the
 * image stitch at sync's resume path, rewrite_from_common, the speculative
 * walk), and a second mirrored frame there would be a second, unbalanced half
 * of an operation the peer is not expecting.
 * ------------------------------------------------------------------------ */

bool pulsar_session_is_mirrored(const pulsar_session *s) {
    return s && s->engine && s->engine->tp && s->tp_session_id != 0;
}

/** The pair this session mirrors onto, or NULL when nothing should be mirrored
 * (pair off, or a session the engine handed out before the transport existed). */
static pulsar_tp *tp_mirror_target(pulsar_session *s) {
    return pulsar_session_is_mirrored(s) ? s->engine->tp : NULL;
}

/** Refusal shared by the mirrored operations: the pair is armed but its
 * transport is already dead, so no frame can be trusted in either direction. */
static int tp_mirror_dead(pulsar_tp *tp, char *err, size_t errlen) {
    if (!pulsar_tp_failed(tp)) return 0;
    if (err) snprintf(err, errlen,
                      "tp: the pair's transport failed earlier in this run; refusing to mirror this session");
    return 1;
}

/** A worker rank has no arguments of its own to mirror: its sessions are driven
 * by the leader's frames through pulsar_tp_worker_run.  A driver that calls a
 * session operation on a worker is the same-driver model this slice retired,
 * and it is refused before any frame moves. */
/* Defined with the bank wrappers below; shared by every verdict operation. */
static int tp_mirror_bank_verdict(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                  int own, int divergence_rc);

static int tp_mirror_worker_drives_nothing(pulsar_tp *tp, const char *operation,
                                           char *err, size_t errlen) {
    if (pulsar_tp_rank(tp) == 0) return 0;
    if (err) snprintf(err, errlen,
                      "tp: rank %d is a worker; it does not drive %s -- it runs pulsar_tp_worker_run",
                      pulsar_tp_rank(tp), operation);
    return 1;
}

/** The collector every mirrored operation ends with on the leader: one ack per
 * peer.  It is read even when `body_rc` failed locally, because an unread ack
 * would be consumed by the NEXT operation, shifting every later frame by one. */
static int tp_mirror_leader_ack(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                int body_rc, char *err, size_t errlen) {
    char peer_err[256];
    peer_err[0] = '\0';
    const int peer_ok = pulsar_tp_wait_command_ack(tp, s->tp_session_id, operation,
                                                   peer_err, sizeof(peer_err));
    if (body_rc != 0) return body_rc;
    if (!peer_ok) {
        if (err) snprintf(err, errlen, "tp: a worker failed the mirrored %s: %s",
                          operation, peer_err);
        return 1;
    }
    return 0;
}

/** Settle a pipelined eval's identity check (pulsar_session_eval defers it,
 * L241 4g-2): before logits VALUES leave the engine (copy, logprobs) and
 * before the session ends.  Token decisions (sample, argmax) do not settle --
 * the next eval does, before the token it drew is returned to its caller.
 * With no `err` the refusal is printed.  1 = nothing pending or it matched. */
static int tp_mirror_settle(pulsar_session *s, char *err, size_t errlen) {
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp || pulsar_tp_rank(tp) != 0) return 1;
    char why[512];
    why[0] = '\0';
    if (pulsar_tp_settle_deferred_ack(tp, why, sizeof(why))) return 1;
    if (err) snprintf(err, errlen, "%s", why);
    else fprintf(stderr, "pulsar: %s\n", why);
    return 0;
}

int pulsar_session_settle(pulsar_session *s, char *err, size_t errlen) {
    return s && tp_mirror_settle(s, err, errlen) ? 0 : 1;
}

/** The collector for the operations that PRODUCE logits (eval, batch decode,
 * mixed batch): the peers' acks carry the digest of their assembled logits and
 * must equal the leader's own -- the cross-rank identity check (L243).  When the
 * leader's own body failed there is nothing to compare; the acks are drained in
 * either shape so the next frame is not shifted, and the local failure is the
 * result.  `n_rows` rows of the engine's logits width, exactly the rows the body
 * wrote. */
static int tp_mirror_leader_ack_logits(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                       int body_rc, const float *logits, uint32_t n_rows,
                                       char *err, size_t errlen) {
    if (body_rc != 0) {
        pulsar_tp_drain_command_acks(tp);
        return body_rc;
    }
    if (!logits && n_rows > 0) {
        pulsar_tp_drain_command_acks(tp);
        if (err) snprintf(err, errlen, "tp: the mirrored %s produced %u rows but no logits buffer to "
                          "check them by", operation, n_rows);
        return 1;
    }
    const uint64_t own = pulsar_tp_logits_digest(logits, n_rows, (uint32_t)s->engine->logits_width());
    char peer_err[256];
    peer_err[0] = '\0';
    if (!pulsar_tp_wait_command_ack_digest(tp, s->tp_session_id, operation, own,
                                           peer_err, sizeof(peer_err))) {
        if (err) snprintf(err, errlen, "tp: a worker failed the mirrored %s: %s",
                          operation, peer_err);
        return 1;
    }
    return 0;
}

/** The failure report a `void` operation can make: it has no error channel, so
 * the pair is marked failed -- every later mirrored operation then refuses
 * through tp_mirror_dead on the leader, and through the worker loop's failed
 * check on a worker -- and the reason is printed here, named, while it is still
 * the newest line on stderr.  Shared with tp_worker.cpp. */
void pulsar_tp_mirror_fail_void(pulsar_tp *tp, const char *operation, const char *why) {
    pulsar_tp_mark_failed(tp);
    fprintf(stderr, "pulsar: tp: the mirrored %s failed (%s); the pair is marked failed\n",
            operation, why ? why : "no reason given");
}

/** The void operations the leader mirrors FIRE-AND-FORGET: destroy, rewind and
 * invalidate.  A `void` caller has nowhere to put a peer's refusal, and several
 * callers are the server's cache and scheduler, whose timing follows LOCAL
 * memory state -- so no ack is collected; the worker loop's checks are the
 * divergence alarm, and the next ACKED operation carries the refusal back. */
static int tp_mirror_void_send(pulsar_tp *tp, uint64_t session_id, const char *name, int sent) {
    if (sent != 0) return 1;
    pulsar_tp_mirror_fail_void(tp, name, "the frame could not be shipped");
    (void)session_id;
    return 0;
}

int pulsar_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    /* A worker rank creates sessions only from the leader's frames (the loop
     * calls the member directly); a driver creating one here is refused before
     * the ordinal is consumed, so the two ranks' ordinals stay aligned. */
    if (e && e->tp && pulsar_tp_rank(e->tp) != 0) {
        fprintf(stderr, "pulsar: tp: rank %d is a worker; it does not create sessions -- "
                        "it runs pulsar_tp_worker_run\n", pulsar_tp_rank(e->tp));
        if (out) *out = NULL;
        return 1;
    }
    const int rc = pulsar_session::create(out, e, ctx_size);
    if (rc != 0 || !out || !*out) return rc;
    pulsar_session *s = *out;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return 0;
    /* Slice 4e: the session's mirror id is its create ordinal, so both ranks
     * already agree on it by construction; what the frame adds is the CHECK --
     * the leader announces the id and the context size it created, and a
     * worker whose own create produced a different ordinal refuses HERE, at
     * the earliest point.  A refused create frees the session on the leader
     * and leaves *out NULL: a session the pair did not agree on must not reach
     * a caller.  This function has no error buffer, so the reason goes to
     * stderr and the return code carries it. */
    char err[256];
    err[0] = '\0';
    if (pulsar_tp_send_session_create(tp, s->tp_session_id, ctx_size) == 0) {
        pulsar_tp_mirror_fail_void(tp, "session create", "the frame could not be shipped");
        s->destroy();
        *out = NULL;
        return 1;
    }
    if (!pulsar_tp_wait_command_ack(tp, s->tp_session_id, "session create", err, sizeof(err))) {
        fprintf(stderr, "pulsar: tp: %s\n", err);
        s->destroy();
        *out = NULL;
        return 1;
    }
    return 0;
}

void pulsar_session_free(pulsar_session *s) {
    if (!s) return;
    pulsar_tp *tp = tp_mirror_target(s);
    if (tp && pulsar_tp_rank(tp) == 0) {
        (void)tp_mirror_settle(s, NULL, 0);   /* a refusal is printed; the pair is marked failed */
        /* Fire-and-forget like rewind: the worker loop drops its registry
         * entry, and an unknown id there marks the pair failed. */
        (void)tp_mirror_void_send(tp, s->tp_session_id, "session destroy",
                                  pulsar_tp_send_session_destroy(tp, s->tp_session_id));
    }
    s->destroy();
}

/** The wire rows of a batched step: the engine's own row contract plus the
 * session.  Shared by the batched decode and the mixed step so the two frames
 * cannot encode a row differently. */
static pulsar_tp_batch_item *tp_mirror_rows(const pulsar_session *s, const pulsar_multiseq_req *reqs,
                                            uint32_t n) {
    pulsar_tp_batch_item *items = (pulsar_tp_batch_item *)xmalloc((size_t)n * sizeof(*items));
    for (uint32_t i = 0; i < n; i++) {
        items[i].session_id = s->tp_session_id;
        items[i].bank       = (int32_t)reqs[i].bank;
        items[i].pos        = reqs[i].pos;
        items[i].token      = reqs[i].token;
        items[i].reserved   = 0;
    }
    return items;
}

int pulsar_session_sync(pulsar_session *s, const pulsar_tokens *prompt, char *err, size_t errlen) {
    return pulsar_session_sync_mm(s, prompt, NULL, 0, err, errlen);
}
int pulsar_session_sync_mm(pulsar_session *s, const pulsar_tokens *prompt,
                           const pulsar_image_ref *images, int n_images, char *err, size_t errlen) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->sync(prompt, images, n_images, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "sync", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    /* The leader's arguments ARE the operation, so an empty prompt is a caller bug. */
    if (!prompt || prompt->len <= 0) {
        if (err) snprintf(err, errlen, "tp: the prompt is empty; refusing to mirror it");
        return 1;
    }
    if (n_images < 0 || (n_images > 0 && !images)) {
        if (err) snprintf(err, errlen, "tp: an image request with no images; refusing");
        return 1;
    }
    /* Ship first, run second, so both ranks prefill together instead of the
     * worker waiting out the leader's whole sync.  Images ride the frame
     * (increment 7): the tokens already carry the expanded sentinel blocks,
     * and every rank's own replicated tower encodes the same bytes. */
    const int sent = n_images > 0
        ? pulsar_tp_send_sync_mm(tp, s->tp_session_id, prompt->v, (uint32_t)prompt->len, images, (uint32_t)n_images)
        : pulsar_tp_send_sync(tp, s->tp_session_id, prompt->v, (uint32_t)prompt->len);
    if (sent == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the prompt%s to the workers",
                          n_images > 0 ? " and its images" : "");
        return 1;
    }
    return tp_mirror_leader_ack(s, tp, "sync",
                                s->sync(prompt, images, n_images, err, errlen),
                                err, errlen);
}
int pulsar_expand_image_placeholders(pulsar_engine *e, const pulsar_tokens *prompt,
                                     pulsar_image_ref *images, int n_images,
                                     pulsar_tokens *out, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!e || !prompt || !out || n_images < 0 || (n_images > 0 && !images)) {
        if (err) snprintf(err, errlen, "image request is missing its prompt, images, or output buffer");
        return 0;
    }
    /* The tower's absence is a client-visible condition (400), not an engine
     * crash, so it is checked HERE rather than left to the graph. */
    if (n_images > 0 && !e->vision_ready) {
        if (err) snprintf(err, errlen, "this model has no vision tower bound; it cannot accept images");
        return 0;
    }
    const int placeholder = e->vocab.image_id;
    if (placeholder < 0) {
        if (err) snprintf(err, errlen,
                          "the tokenizer has no \"%s\" image placeholder token", PULSAR_IMAGE_PLACEHOLDER);
        return 0;
    }
    /* Count first so a mismatch reports both numbers (the expander reports the
     * same condition, but only to stderr, where an HTTP client cannot see it). */
    int seen = 0;
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] == placeholder) seen++;
    }
    if (seen != n_images) {
        if (err) snprintf(err, errlen,
                          "the prompt carries %d image placeholder(s) but the request has %d image(s)",
                          seen, n_images);
        return 0;
    }

    pulsar_vision_args args;
    args.patch_size       = (int)PULSAR_VISION_PATCH;
    args.downsample_ratio = (int)PULSAR_VISION_DOWNSAMPLE;
    args.max_n_token      = (int)PULSAR_VISION_MAX_N_TOKEN;
    args.min_pixels       = (int)PULSAR_VISION_MIN_PIXELS;
    args.max_wh_ratio     = PULSAR_VISION_MAX_WH_RATIO;

    pulsar_vision_prepared *preps = NULL;
    int *starts = NULL;
    if (n_images > 0) {
        preps = (pulsar_vision_prepared *)xmalloc((size_t)n_images * sizeof(preps[0]));
        memset(preps, 0, (size_t)n_images * sizeof(preps[0]));
        starts = (int *)xmalloc((size_t)n_images * sizeof(starts[0]));
    }
    /* The one producer of sentinel blocks.  It also decodes+preprocesses each
     * image to learn its span; the mm prefill re-decodes from the same bytes and
     * args, so the two agree by construction.  `preps` is only needed for the
     * geometry here and is released before the model runs. */
    const int ok = vision_expand_image_placeholders(out, prompt, placeholder,
                                                    images, n_images, &args,
                                                    (int)PULSAR_N_VOCAB, preps, starts);
    if (ok) {
        for (int i = 0; i < n_images; i++) images[i].start_pos = starts[i];
    } else if (err) {
        snprintf(err, errlen,
                 "an image could not be decoded or is not one the vision tower accepts");
    }
    for (int i = 0; i < n_images; i++) vision_prepared_free(&preps[i]);
    free(preps);
    free(starts);
    if (!ok) pulsar_tokens_free(out);
    return ok;
}
pulsar_session_rewrite_result pulsar_session_rewrite_from_common(pulsar_session *s, const pulsar_tokens *prompt, int common, char *err, size_t errlen) {
    if (!s) return PULSAR_SESSION_REWRITE_ERROR;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->rewrite_from_common(prompt, common, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "rewrite from common", err, errlen) ||
        tp_mirror_dead(tp, err, errlen)) return PULSAR_SESSION_REWRITE_ERROR;
    if (!prompt || prompt->len < 0 || (prompt->len > 0 && !prompt->v)) return PULSAR_SESSION_REWRITE_ERROR;
    if (pulsar_tp_send_rewrite_from_common(tp, s->tp_session_id, prompt->v,
                                           (uint32_t)prompt->len, common) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the rewrite to the workers");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    /* The verdict rides the wire as result + 1 (ERROR is -1 and a negative
     * wire status is a worker refusal). */
    const pulsar_session_rewrite_result own = s->rewrite_from_common(prompt, common, err, errlen);
    const int agreed = tp_mirror_bank_verdict(s, tp, "rewrite from common", (int)own + 1, -1);
    return agreed < 0 ? PULSAR_SESSION_REWRITE_ERROR : (pulsar_session_rewrite_result)(agreed - 1);
}
int pulsar_session_common_prefix(pulsar_session *s, const pulsar_tokens *prompt) { return s->common_prefix(prompt); }
void pulsar_session_prefix_match(pulsar_session *s, const pulsar_tokens *prompt, pulsar_prefix_match *out) { if (s) { s->prefix_match(prompt, out); } else if (out) { out->live_cut = 0; out->prompt_cut = 0; out->seamed = false; } }
int pulsar_session_argmax(pulsar_session *s) { return s->argmax(); }
int pulsar_session_argmax_excluding(pulsar_session *s, int excluded_id) { return s ? s->argmax_excluding(excluded_id) : -1; }
int pulsar_session_sample(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) { return s->sample(temperature, top_k, top_p, min_p, rng); }
int pulsar_session_top_logprobs(pulsar_session *s, pulsar_token_score *out, int k) { return s && tp_mirror_settle(s, NULL, 0) ? s->top_logprobs(out, k) : 0; }
int pulsar_session_token_logprob(pulsar_session *s, int token, pulsar_token_score *out) { return s && tp_mirror_settle(s, NULL, 0) ? s->token_logprob(token, out) : 0; }
int pulsar_session_copy_logits(pulsar_session *s, float *out, int cap) { return s && tp_mirror_settle(s, NULL, 0) ? s->copy_logits(out, cap) : 0; }
int pulsar_session_set_logits(pulsar_session *s, const float *logits, int n) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->set_logits(logits, n);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "set logits", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return 1;
    }
    if (!logits || n <= 0) return 1;
    /* The vector itself rides the frame: every rank already holds the same
     * full logits after the vocab all-gather, but the worker's copy lives in
     * its loop's scratch, not in the session, and exactness is the contract. */
    if (pulsar_tp_send_set_logits(tp, s->tp_session_id, logits, (uint32_t)n) == 0) {
        pulsar_tp_mirror_fail_void(tp, "set logits", "the frame could not be shipped");
        return 1;
    }
    return tp_mirror_bank_verdict(s, tp, "set logits", s->set_logits(logits, n) != 0 ? 1 : 0, 1);
}
int pulsar_session_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->eval(token, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "eval", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    /* The frame's seq is this session's decode position -- the number of tokens
     * whose KV the graph holds -- so the worker can do more than trust the
     * leader's token: it refuses when the two ranks are not at the same place.
     * Without that check a rank that fell behind would decode the right token
     * at the wrong position and produce confident nonsense. */
    const uint64_t pos = (uint64_t)s->checkpoint.len;
    static double t_prev_end = 0.0;   /* TEMPORARY instrument */
    const double t_send = pulsar_tp_now_sec();
    if (t_prev_end > 0.0) pulsar_tp_timing_add(PULSAR_TP_TSITE_STEP, PULSAR_TP_TPH_HOST, t_send - t_prev_end, 0);
    if (pulsar_tp_send_eval(tp, s->tp_session_id, pos, token) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the token to the workers");
        return 1;
    }
    /* PIPELINED identity check (L241 4g-2): the previous eval's digest ack is
     * settled here, after this step's body -- it has been in the socket since
     * the worker finished that step -- and this step's own digest is DEFERRED
     * instead of waited on, so the leader samples and ships the next token
     * while the worker is still acking this one.  No token leaves unchecked:
     * the one drawn from these logits is returned by the NEXT eval, which
     * settles them first; logits values settle before they are copied out
     * (tp_mirror_settle).  The body runs even when the settle failed, so both
     * ranks finish this step and its ack is drained, not left in the socket. */
    const int body_rc = s->eval(token, err, errlen);
    t_prev_end = pulsar_tp_now_sec();
    pulsar_tp_timing_add(PULSAR_TP_TSITE_STEP, PULSAR_TP_TPH_XCHG, t_prev_end - t_send, 0);
    char why[512];
    why[0] = '\0';
    const int settled = pulsar_tp_settle_deferred_ack(tp, why, sizeof(why));
    if (body_rc != 0 || !settled) {
        pulsar_tp_drain_command_acks(tp);
        if (body_rc != 0) return body_rc;
        if (err) snprintf(err, errlen, "%s (found at the eval of position %llu)", why,
                          (unsigned long long)pos);
        return 1;
    }
    if (!s->logits) {
        pulsar_tp_drain_command_acks(tp);
        if (err) snprintf(err, errlen, "tp: the mirrored eval produced no logits to check");
        return 1;
    }
    char what[96];
    snprintf(what, sizeof(what), "eval at position %llu", (unsigned long long)pos);
    if (!pulsar_tp_defer_command_ack_digest(tp, s->tp_session_id, what,
                                            pulsar_tp_logits_digest(s->logits, 1u,
                                                                    (uint32_t)s->engine->logits_width()))) {
        pulsar_tp_drain_command_acks(tp);
        if (err) snprintf(err, errlen, "tp: could not defer the eval's identity check");
        return 1;
    }
    return 0;
}
int pulsar_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, float *logits, int logits_cap, char *err, size_t errlen) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->decode_multiseq(reqs, n, logits, logits_cap, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "the batched decode", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    if (!reqs || n == 0) {
        if (err) snprintf(err, errlen, "tp: refusing to mirror an empty batch");
        return 1;
    }
    pulsar_tp_batch_item *items = tp_mirror_rows(s, reqs, n);
    const int sent = pulsar_tp_send_eval_batch(tp, items, n);
    free(items);
    if (sent == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the batch to the workers");
        return 1;
    }
    return tp_mirror_leader_ack_logits(s, tp, "batch decode",
                                       s->decode_multiseq(reqs, n, logits, logits_cap, err, errlen),
                                       logits, n, err, errlen);
}
int pulsar_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits, int logits_cap, uint32_t *out_n_rows, uint32_t max_head_runs, char *err, size_t errlen) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->decode_mixed(reqs, n_rows, logits, logits_cap, out_n_rows,
                                    max_head_runs, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "the mixed step", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    if (!reqs || n_rows == 0) {
        if (err) snprintf(err, errlen, "tp: refusing to mirror an empty batch");
        return 1;
    }
    /* `max_head_runs` is the caller's head policy and rides the frame: a worker
     * has no caller of its own to take it from, and both ranks must head the
     * same rows.  `out_n_rows` is output and stays local on each rank. */
    pulsar_tp_batch_item *items = tp_mirror_rows(s, reqs, n_rows);
    const int sent = pulsar_tp_send_mixed_batch(tp, items, n_rows, max_head_runs);
    free(items);
    if (sent == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the mixed batch to the workers");
        return 1;
    }
    const int body_rc = s->decode_mixed(reqs, n_rows, logits, logits_cap,
                                        out_n_rows, max_head_runs, err, errlen);
    return tp_mirror_leader_ack_logits(s, tp, "mixed batch", body_rc, logits,
                                       out_n_rows ? *out_n_rows : 0u, err, errlen);
}
int pulsar_session_bank_count(pulsar_session *s) { return s ? s->bank_count() : 0; }
/* ---- The bank surface (increment 2).  Bank SELECTION already rode the
 * decode rows; these mirror the leader scheduler's bank DECISIONS -- save,
 * restore, repoint, fork, partial fork -- so both ranks' pools hold the same
 * state.  The leader's decision is the authority and the ordinal names the
 * target.  save is void and fire-and-forget; the others return a verdict the
 * ranks must AGREE on: the same inputs on the same state give the same
 * result, so a split verdict is a divergence (marked failed, refused), not a
 * vote. */
static int tp_mirror_bank_verdict(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                  int own, int divergence_rc) {
    char err[256];
    err[0] = '\0';
    int peers = 0;
    if (!pulsar_tp_wait_command_status(tp, s->tp_session_id, operation, &peers, err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, operation, err);
        return divergence_rc;
    }
    if (peers != own) {
        snprintf(err, sizeof(err), "this rank's verdict is %d but the workers agree on %d", own, peers);
        pulsar_tp_mirror_fail_void(tp, operation, err);
        return divergence_rc;
    }
    return own;
}

/** The verdict collect for a verdict operation that also ends on logits
 * (generate_speculative): a POSITIVE own verdict carries this rank's digest of
 * `n_rows` rows of `logits` and every peer's positive verdict must match it
 * (the cross-rank identity check, L243).  An own verdict of 0 (the run failed
 * here) collects plainly; a peer that ran anyway answers with a digest, which
 * the plain collect refuses as a shape mismatch -- a split either way. */
static int tp_mirror_bank_verdict_logits(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                         int own, int divergence_rc, const float *logits,
                                         uint32_t n_rows) {
    if (own <= 0) return tp_mirror_bank_verdict(s, tp, operation, own, divergence_rc);
    char err[256];
    err[0] = '\0';
    int peers = 0;
    const uint64_t digest = pulsar_tp_logits_digest(logits, n_rows, (uint32_t)s->engine->logits_width());
    if (!pulsar_tp_wait_command_status_digest(tp, s->tp_session_id, operation, &peers, digest,
                                              err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, operation, err);
        return divergence_rc;
    }
    if (peers != own) {
        snprintf(err, sizeof(err), "this rank's verdict is %d but the workers agree on %d", own, peers);
        pulsar_tp_mirror_fail_void(tp, operation, err);
        return divergence_rc;
    }
    return own;
}
int pulsar_session_bank_repoint(pulsar_session *s, uint32_t bank) {
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->bank_repoint(bank);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank repoint", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return 1;
    }
    if (pulsar_tp_send_bank_repoint(tp, s->tp_session_id, bank) == 0) {
        pulsar_tp_mirror_fail_void(tp, "bank repoint", "the frame could not be shipped");
        return 1;
    }
    return tp_mirror_bank_verdict(s, tp, "bank repoint", s->bank_repoint(bank), 1);
}
void pulsar_session_bank_state_save(pulsar_session *s, uint32_t bank) {
    if (!s) return;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { s->bank_state_save(bank); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank state save", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "bank state save", err);
        return;
    }
    if (!tp_mirror_void_send(tp, s->tp_session_id, "bank state save",
                             pulsar_tp_send_bank_state_save(tp, s->tp_session_id, bank))) return;
    s->bank_state_save(bank);
}
bool pulsar_session_bank_state_restore(pulsar_session *s, uint32_t bank) {
    if (!s) return false;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->bank_state_restore(bank);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank state restore", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return false;
    }
    if (pulsar_tp_send_bank_state_restore(tp, s->tp_session_id, bank) == 0) {
        pulsar_tp_mirror_fail_void(tp, "bank state restore", "the frame could not be shipped");
        return false;
    }
    const int own = s->bank_state_restore(bank) ? 0 : 1;
    return tp_mirror_bank_verdict(s, tp, "bank state restore", own, 1) == 0;
}
static int tp_mirror_bank_fork(pulsar_session *s, int partial, uint32_t src, uint32_t dst,
                               const int *tokens, int n_tokens, int n_cached) {
    const char *operation = partial ? "partial bank fork" : "bank fork";
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) {
        return partial ? s->bank_fork_partial(src, dst, tokens, n_tokens, n_cached)
                       : s->bank_fork(src, dst, tokens, n_tokens, n_cached);
    }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, operation, err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return PULSAR_FORK_EINVAL;
    }
    if (n_tokens < 0 || (n_tokens > 0 && !tokens)) return PULSAR_FORK_EINVAL;
    if (pulsar_tp_send_bank_fork(tp, partial, s->tp_session_id, src, dst,
                                 tokens, (uint32_t)n_tokens, n_cached) == 0) {
        pulsar_tp_mirror_fail_void(tp, operation, "the frame could not be shipped");
        return PULSAR_FORK_EINVAL;
    }
    const int own = partial ? s->bank_fork_partial(src, dst, tokens, n_tokens, n_cached)
                            : s->bank_fork(src, dst, tokens, n_tokens, n_cached);
    return tp_mirror_bank_verdict(s, tp, operation, own, PULSAR_FORK_EINVAL);
}
int pulsar_session_bank_fork(pulsar_session *s, uint32_t src, uint32_t dst, const int *tokens, int n_tokens, int n_cached) {
    return s ? tp_mirror_bank_fork(s, 0, src, dst, tokens, n_tokens, n_cached) : 1;
}
/* ---- The eviction guard's spill path (increment 6).  KV is replicated per
 * rank, so a spill is per rank to its own disk.  The snapshot's identity is
 * the KEY of the file the leader was handed: its basename with the server's
 * ".tmp.<pid>" suffix stripped, read back from the descriptor, because the
 * public API takes a FILE* and the server renames the temp into place after
 * the save.  The worker mirrors the snapshot under its own spill directory
 * by that key, so a later load names the same snapshot on every rank. */
static int tp_spill_key(FILE *fp, char *out, size_t outlen) {
    if (!fp) return 0;
    char link[64], target[4096];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fileno(fp));
    const ssize_t n = readlink(link, target, sizeof(target) - 1u);
    if (n <= 0) return 0;
    target[n] = '\0';
    const char *base = strrchr(target, '/');
    base = base ? base + 1 : target;
    if (!base[0]) return 0;
    /* strip a trailing ".tmp.<digits>" */
    size_t len = strlen(base);
    const char *t = strstr(base, ".tmp.");
    while (t) {
        const char *d = t + 5;
        size_t k = 0;
        while (d[k] >= '0' && d[k] <= '9') k++;
        if (k > 0 && d[k] == '\0') { len = (size_t)(t - base); break; }
        t = strstr(t + 1, ".tmp.");
    }
    if (len == 0 || len >= outlen) return 0;
    memcpy(out, base, len);
    out[len] = '\0';
    return 1;
}
static bool tp_mirror_bank_physical(pulsar_session *s, int freeing, uint32_t bank) {
    const char *operation = freeing ? "bank free physical" : "bank alloc physical";
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return freeing ? s->bank_free_physical(bank) : s->bank_alloc_physical(bank);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, operation, err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return false;
    }
    const int sent = freeing ? pulsar_tp_send_bank_free_physical(tp, s->tp_session_id, bank)
                             : pulsar_tp_send_bank_alloc_physical(tp, s->tp_session_id, bank);
    if (sent == 0) {
        pulsar_tp_mirror_fail_void(tp, operation, "the frame could not be shipped");
        return false;
    }
    const int own = (freeing ? s->bank_free_physical(bank) : s->bank_alloc_physical(bank)) ? 0 : 1;
    return tp_mirror_bank_verdict(s, tp, operation, own, 1) == 0;
}
bool pulsar_session_bank_free_physical(pulsar_session *s, uint32_t bank) {
    return s ? tp_mirror_bank_physical(s, 1, bank) : false;
}
bool pulsar_session_bank_alloc_physical(pulsar_session *s, uint32_t bank) {
    return s ? tp_mirror_bank_physical(s, 0, bank) : false;
}
static int tp_mirror_bank_kv(pulsar_session *s, int load, uint32_t bank, FILE *fp, char *err, size_t errlen) {
    const char *operation = load ? "bank kv load" : "bank kv save";
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return load ? s->bank_kv_load(bank, fp, err, errlen) : s->bank_kv_save(bank, fp, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, operation, err, errlen) ||
        tp_mirror_dead(tp, err, errlen)) return 1;
    char key[256];
    if (!tp_spill_key(fp, key, sizeof(key))) {
        if (err) snprintf(err, errlen, "tp: %s: the snapshot file has no usable key (not a named file?)", operation);
        return 1;
    }
    if (pulsar_tp_send_bank_kv(tp, load, s->tp_session_id, bank, key) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror the %s to the workers", operation);
        return 1;
    }
    const int own = (load ? s->bank_kv_load(bank, fp, err, errlen) : s->bank_kv_save(bank, fp, err, errlen)) == 0 ? 0 : 1;
    return tp_mirror_bank_verdict(s, tp, operation, own, 1);
}
int pulsar_session_bank_kv_save(pulsar_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen) {
    return s ? tp_mirror_bank_kv(s, 0, bank, fp, err, errlen) : 1;
}
int pulsar_session_bank_kv_load(pulsar_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen) {
    return s ? tp_mirror_bank_kv(s, 1, bank, fp, err, errlen) : 1;
}
int pulsar_session_bank_fork_partial(pulsar_session *s, uint32_t src, uint32_t dst, const int *tokens, int n_tokens, int n_cached) {
    return s ? tp_mirror_bank_fork(s, 1, src, dst, tokens, n_tokens, n_cached) : PULSAR_FORK_EINVAL;
}
int pulsar_session_bank_pos(pulsar_session *s, uint32_t bank) { return s->bank_pos(bank); }
int pulsar_session_bank_spec_depth(pulsar_session *s, uint32_t bank) { return s->bank_spec_depth(bank); }
const pulsar_tokens *pulsar_session_bank_tokens(pulsar_session *s, uint32_t bank) { return s->bank_tokens(bank); }
int pulsar_session_bank_common_prefix(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt) { return s->bank_common_prefix(bank, prompt); }
void pulsar_session_bank_prefix_match(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, pulsar_prefix_match *out) { if (s) { s->bank_prefix_match(bank, prompt, out); } else if (out) { out->live_cut = 0; out->prompt_cut = 0; out->seamed = false; } }
void pulsar_session_note_committed_tokens(pulsar_session *s, const int *toks, int n) {
    if (!s) return;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { s->note_committed_tokens(toks, n); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "note committed tokens", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "note committed tokens", err);
        return;
    }
    if (n < 0 || (n > 0 && !toks)) return;
    if (!tp_mirror_void_send(tp, s->tp_session_id, "note committed tokens",
                             pulsar_tp_send_note_committed(tp, s->tp_session_id, toks, (uint32_t)n))) return;
    s->note_committed_tokens(toks, n);
}
/* ---- The speculative round family (increment 4).  Each public entry point
 * ships a frame carrying every input the call reads INCLUDING the rng state
 * as it stands before the call, runs the local implementation, and requires
 * the workers' agreed verdict to equal its own.  With identical inputs on
 * identical state -- same logits after the vocab all-gather, same drafts,
 * same rng -- the walk is a deterministic function, so a split verdict is a
 * divergence.  Verdicts that can be -1 ride the wire as value + 1. */
static pulsar_tp_spec_command tp_spec_cmd(const pulsar_session *s, int bank) {
    pulsar_tp_spec_command c;
    memset(&c, 0, sizeof(c));
    c.session_id = s->tp_session_id;
    c.bank = bank;
    return c;
}
static int tp_spec_live_bank(const pulsar_session *s) {
    return (int)gpu_graph_cur_bank(&s->graph);
}
/* Returns 1 when the operation may run locally without mirroring (pair off),
 * 0 when it was refused (err filled), 2 when it must be mirrored. */
static int tp_spec_route(pulsar_session *s, const char *operation, pulsar_tp **tp_out,
                         char *err, size_t errlen) {
    *tp_out = tp_mirror_target(s);
    if (!*tp_out) return 1;
    if (tp_mirror_worker_drives_nothing(*tp_out, operation, err, errlen)) return 0;
    if (tp_mirror_dead(*tp_out, err, errlen)) return 0;
    return 2;
}

int pulsar_session_generate_speculative(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng, int max_tokens, int eos_token, int *accepted, int accepted_cap, char *err, size_t errlen) {
    if (!s) return 0;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "generate_speculative", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return s->generate_speculative(temperature, top_k, top_p, min_p, rng, max_tokens, eos_token, accepted, accepted_cap, err, errlen);
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = max_tokens; c.i1 = eos_token; c.i2 = accepted_cap;
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_GENERATE_SPECULATIVE, &c, NULL, NULL) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror generate_speculative to the workers");
        return -1;
    }
    const int own = s->generate_speculative(temperature, top_k, top_p, min_p, rng, max_tokens, eos_token, accepted, accepted_cap, err, errlen);
    const int agreed = tp_mirror_bank_verdict_logits(s, tp, "generate_speculative", own + 1, -1,
                                                     s->logits, 1u);
    return agreed < 0 ? -1 : own;
}
int pulsar_session_spec_next_base(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s) return -1;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_next_base", &tp, err, sizeof(err));
    if (route == 0) { fprintf(stderr, "pulsar: %s\n", err); return -1; }
    if (route == 1) return pulsar_session_spec_next_base_local(s, temperature, top_k, top_p, min_p, rng);
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_NEXT_BASE, &c, NULL, NULL) == 0) {
        pulsar_tp_mirror_fail_void(tp, "spec_next_base", "the frame could not be shipped");
        return -1;
    }
    const int own = pulsar_session_spec_next_base_local(s, temperature, top_k, top_p, min_p, rng);
    const int agreed = tp_mirror_bank_verdict(s, tp, "spec_next_base", own + 1, -1);
    return agreed < 0 ? -1 : own;
}
int pulsar_session_spec_round_begin(pulsar_session *s, pulsar_spec_round *r, int first_token, int max_tokens, int accepted_cap, float temperature, int top_k, float top_p, float min_p, char *err, size_t errlen) {
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_begin", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_spec_round_begin_local(s, r, first_token, max_tokens, accepted_cap, temperature, top_k, top_p, min_p, err, errlen);
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = first_token; c.i1 = max_tokens; c.i2 = accepted_cap;
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    if (pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_BEGIN, &c, NULL, NULL) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror spec_round_begin to the workers");
        return -1;
    }
    const int own = pulsar_session_spec_round_begin_local(s, r, first_token, max_tokens, accepted_cap, temperature, top_k, top_p, min_p, err, errlen);
    return tp_mirror_bank_verdict(s, tp, "spec_round_begin", own == 0 ? 0 : 1, -1) < 0 ? -1 : own;
}
int pulsar_session_spec_round_end(pulsar_session *s, pulsar_spec_round *r, int first_token, int eos_token, float temperature, int top_k, float top_p, float min_p, uint64_t *rng, const float *rows, uint32_t row0, int *accepted, int accepted_cap, char *err, size_t errlen) {
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_end", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_spec_round_end_local(s, r, first_token, eos_token, temperature, top_k, top_p, min_p, rng, rows, row0, accepted, accepted_cap, err, errlen);
    /* The logits block is NOT shipped: every rank holds the same block from
     * its own mirrored forward (the vocab all-gather made them identical), so
     * only row0 crosses. */
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = first_token; c.i1 = eos_token; c.i2 = accepted_cap; c.i3 = (int32_t)row0;
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_END, &c, NULL, NULL) == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror spec_round_end to the workers");
        return -1;
    }
    const int own = pulsar_session_spec_round_end_local(s, r, first_token, eos_token, temperature, top_k, top_p, min_p, rng, rows, row0, accepted, accepted_cap, err, errlen);
    const int agreed = tp_mirror_bank_verdict(s, tp, "spec_round_end", own + 1, -1);
    return agreed < 0 ? -1 : own;
}
void pulsar_session_spec_round_abort(pulsar_session *s, pulsar_spec_round *r) {
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_abort", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_round_abort", err); return; }
    if (route == 1) { pulsar_session_spec_round_abort_local(s, r); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    if (!tp_mirror_void_send(tp, s->tp_session_id, "spec_round_abort",
                             pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_ABORT, &c, NULL, NULL))) return;
    pulsar_session_spec_round_abort_local(s, r);
}
void pulsar_session_spec_arm_capture(pulsar_session *s, uint32_t n_rows) {
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_arm_capture", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_arm_capture", err); return; }
    if (route == 1) { pulsar_session_spec_arm_capture_local(s, n_rows); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = (int32_t)n_rows;
    if (!tp_mirror_void_send(tp, s->tp_session_id, "spec_arm_capture",
                             pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ARM_CAPTURE, &c, NULL, NULL))) return;
    pulsar_session_spec_arm_capture_local(s, n_rows);
}
int pulsar_session_spec_redraft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n, char *err, size_t errlen) {
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_redraft_batch", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_spec_redraft_batch_local(s, rounds, banks, rngs, n, err, errlen);
    if (n < 0 || (n > 0 && (!banks || !rngs))) return -1;
    uint64_t *states = (uint64_t *)xmalloc(((size_t)n ? (size_t)n : 1u) * sizeof(uint64_t));
    for (int i = 0; i < n; i++) states[i] = rngs[i] ? *rngs[i] : 0;
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.count = (uint32_t)n;
    const int sent = pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH, &c, banks, states);
    free(states);
    if (sent == 0) {
        if (err) snprintf(err, errlen, "tp: could not mirror spec_redraft_batch to the workers");
        return -1;
    }
    const int own = pulsar_session_spec_redraft_batch_local(s, rounds, banks, rngs, n, err, errlen);
    return tp_mirror_bank_verdict(s, tp, "spec_redraft_batch", own == 0 ? 0 : 1, -1) < 0 ? -1 : own;
}
void pulsar_session_spec_redraft_commit(pulsar_session *s, pulsar_spec_round *r) {
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_redraft_commit", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_redraft_commit", err); return; }
    if (route == 1) { pulsar_session_spec_redraft_commit_local(s, r); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    if (!tp_mirror_void_send(tp, s->tp_session_id, "spec_redraft_commit",
                             pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT, &c, NULL, NULL))) return;
    pulsar_session_spec_redraft_commit_local(s, r);
}
int pulsar_session_eval_speculative_block(pulsar_session *s, int first_token, int max_tokens, int eos_token, int *accepted, int accepted_cap, char *err, size_t errlen) { return s ? s->eval_speculative_block(first_token, max_tokens, eos_token, accepted, accepted_cap, err, errlen) : 0; }
void pulsar_session_invalidate(pulsar_session *s) {
    /* The pair-off shape is untouched: one call, no TP code on the live path. */
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { s->invalidate(); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "invalidate", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "invalidate", err);
        return;
    }
    if (!tp_mirror_void_send(tp, s->tp_session_id, "invalidate",
                             pulsar_tp_send_invalidate(tp, s->tp_session_id))) return;
    s->invalidate();
}
void pulsar_session_rewind(pulsar_session *s, int pos) {
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { s->rewind(pos); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "rewind", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "rewind", err);
        return;
    }
    if (!tp_mirror_void_send(tp, s->tp_session_id, "rewind",
                             pulsar_tp_send_rewind(tp, s->tp_session_id, pos))) return;
    s->rewind(pos);
}
int pulsar_session_pos(pulsar_session *s) { return s->pos(); }
int pulsar_session_ctx(pulsar_session *s) { return s->ctx(); }
uint32_t pulsar_session_prefill_quantum_min_suffix(const pulsar_session *s) { return s ? s->prefill_quantum_min_suffix() : 0; }
const pulsar_tokens *pulsar_session_tokens(pulsar_session *s) { return s ? s->tokens() : NULL; }
uint64_t pulsar_session_payload_bytes(pulsar_session *s) { return s ? s->payload_bytes() : 0; }
int pulsar_session_stage_payload(pulsar_session *s, pulsar_session_payload_file *out, const char *stage_dir, char *err, size_t errlen) { return s ? s->stage_payload(out, stage_dir, err, errlen) : 1; }
int pulsar_session_save_payload(pulsar_session *s, FILE *fp, char *err, size_t errlen) { return s ? s->save_payload(fp, err, errlen) : 1; }
int pulsar_session_load_payload(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) { return s ? s->load_payload(fp, payload_bytes, err, errlen) : 1; }
int pulsar_session_save_snapshot(pulsar_session *s, pulsar_session_snapshot *snap, char *err, size_t errlen) { return s ? s->save_snapshot(snap, err, errlen) : 1; }
int pulsar_session_load_snapshot(pulsar_session *s, const pulsar_session_snapshot *snap, char *err, size_t errlen) { return s ? s->load_snapshot(snap, err, errlen) : 1; }


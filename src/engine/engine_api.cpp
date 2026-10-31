/** @file engine_api.cpp
 *  @brief The public pulsar.h engine / session API: the free-function facade over pulsar_engine and pulsar_session. */
#include "pulsar_engine_internal.h"
#include "spec_internal.h"
#include "spec_block.h"
#include "pulsar_nvtx.h"
#include "../tp/pulsar_tp.h"
#include "../lib/pulsar_segstore.h"
#include <unistd.h>
#include <string>



/** Make this process the kernel OOM killer's first choice (oom_score_adj 1000), once per process; Linux only, a
 *  failure is printed and ignored.
 *  @param backend the backend, named in the messages */
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
/** One reasoning-effort level's strings (an entry of think_effort_table). */
struct think_effort_row {
    /// the effort as decimal text, "1".."100"; "" in row 0
    char name[4];      /* "1".."100" */
    /// the rendered effort line (PULSAR_REASONING_EFFORT_HEAD N PULSAR_REASONING_EFFORT_TAIL); "" in row 0
    char prefix[160];  /* PULSAR_REASONING_EFFORT_HEAD N PULSAR_REASONING_EFFORT_TAIL */
};
/** Every effort level's strings, indexed by effort; built once on first use, immutable after. */
struct think_effort_table {
    /// [effort] the level's strings; rows below PULSAR_THINK_EFFORT_MIN stay zeroed
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
/** The process-wide effort table (a function-local static: built on first call, thread-safe init).
 *  @return the table; callers keep pointers into it for the process lifetime */
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



/** Close the process's instance lock file (a no-op when none is held). */
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

/* A C API entry whose feature the loaded family does not declare (family.h
 * PULSAR_FAMILY_CAP_*) refuses here, by name, before any family state is
 * touched; the caller gets the entry's own failure value.  DeepSeek declares
 * every cap, so on its engines this is one always-taken branch per call. */
/** Refuse (return `ret`) when session `s`'s family does not declare capability `cap`; `op` names the entry.  A NULL
 *  session passes through to the entry's own NULL handling. */
#define PULSAR_FAMILY_REQUIRES_S(s, cap, op, ret) \
    do { if ((s) && !pulsar_family_require((s)->engine, (cap), (op))) return ret; } while (0)
/** PULSAR_FAMILY_REQUIRES_S for an engine `e`. */
#define PULSAR_FAMILY_REQUIRES_E(e, cap, op, ret) \
    do { if ((e) && !pulsar_family_require((e), (cap), (op))) return ret; } while (0)
/* The session's bank pool (family.h pulsar_family_bank_ops): every family has one, DeepSeek's its graph's. */
/** The session's family bank pool.  @param s the session  @return the family's bank ops */
static inline const pulsar_family_bank_ops *session_banks(const pulsar_session *s) { return s->engine->family->banks; }


int pulsar_engine_open(pulsar_engine **out, const pulsar_engine_options *opt) { return pulsar_engine::open(out, opt); }
int pulsar_engine_open_group(pulsar_engine **out, const pulsar_engine_options *opts, uint32_t n) {
    return pulsar_engine::open_group(out, opts, n);
}
void pulsar_engine_close(pulsar_engine *e) { if (e) e->destroy(); }
void pulsar_engine_summary(pulsar_engine *e) { e->summary(); }
int pulsar_engine_vocab_size(pulsar_engine *e) { return e ? e->vocab_size() : 0; }
int pulsar_engine_logits_width(const pulsar_engine *e) { return e ? e->logits_width() : 0; }
const char *pulsar_engine_model_name(pulsar_engine *e) { return e->model_name(); }
const char *pulsar_engine_served_model_id(const pulsar_engine *e) { return e->family->served_model_id(e); }
/* The loaded shape IS the authority (pulsar_select_shape_from_metadata sets it
 * once at load); `e` is taken so a caller must hold the engine it is asking
 * about, exactly like the other engine facts.  One fact, one name: the chat
 * TEMPLATE family, which is what the renderer forks on. */
bool pulsar_engine_chat_v41(const pulsar_engine *e) {
    if (!e) return true;   /* no engine: the compile-time default profile */
    return e->family->chat_format(e) == PULSAR_CHAT_DS4_V41;
}
pulsar_family_id pulsar_engine_family(const pulsar_engine *e) {
    return e ? e->family->id : PULSAR_FAMILY_ID_DEEPSEEK4;
}
pulsar_chat_format pulsar_engine_chat_format(const pulsar_engine *e) {
    return e ? e->family->chat_format(e) : PULSAR_CHAT_DS4_V41;
}
const char *pulsar_engine_family_name(const pulsar_engine *e) {
    return e ? e->family->name : PULSAR_FAMILY_DEEPSEEK4.name;
}



/* The think-mode axis (none, 1..100) is DeepSeek's encoder's own; every other family's template renders thinking on
 * at its default effort (PULSAR_THINK_DEFAULT, the enabled marker) or off, its own effort levels being the server's
 * effort map's (L295 E6: Inkling; Qwen since L251). */
/// @param e the engine, or NULL (the compile-time default profile)
/// @return true when the family's chat format is DeepSeek V4 / V4.1
static bool think_axis_is_deepseek(const pulsar_engine *e) {
    if (!e) return true;   /* no engine: the compile-time default profile */
    const pulsar_chat_format f = e->family->chat_format(e);
    return f == PULSAR_CHAT_DS4_V4 || f == PULSAR_CHAT_DS4_V41;
}
pulsar_think_mode pulsar_engine_think_default(const pulsar_engine *e) {
    /* on at the template's default effort (pulsar_encode_chat_prompt; the server's marker too) */
    if (!think_axis_is_deepseek(e)) return PULSAR_THINK_DEFAULT;
    return pulsar_engine_chat_v41(e) ? PULSAR_THINK_DEFAULT : PULSAR_THINK_LOW;
}
bool pulsar_engine_think_mode_supported(const pulsar_engine *e, pulsar_think_mode mode, char *why, size_t n) {
    if (!pulsar_think_mode_valid(mode)) {
        if (why) snprintf(why, n, "%d is not a thinking effort (none, or 1..100)", (int)mode);
        return false;
    }
    if (!think_axis_is_deepseek(e)) {
        if (mode == PULSAR_THINK_NONE || mode == PULSAR_THINK_DEFAULT) return true;
        if (why) snprintf(why, n, "the %s template renders thinking on (at its default effort) or off -- no effort level",
                          e->family->name);
        return false;
    }
    if (pulsar_engine_chat_v41(e) || pulsar_think_effort_v4_valid(mode)) return true;
    if (why) snprintf(why, n, "the V4 (0731) encoder has three levels -- low, high, max");
    return false;
}
void pulsar_engine_spec_metrics(pulsar_engine *e, pulsar_spec_metrics *out) { if (e) { e->spec_metrics(out); } else if (out) { memset(out, 0, sizeof(*out)); } }
int pulsar_engine_model_id(pulsar_engine *e) { return e ? e->model_id() : (int)PULSAR_MODEL_VARIANT; }
bool pulsar_engine_is_pruned(pulsar_engine *e) { return e ? e->is_pruned() : false; }
uint64_t pulsar_engine_session_cost_bytes(pulsar_engine *e, int ctx_size) { return e ? e->session_cost_bytes(ctx_size) : 0; }
uint64_t pulsar_engine_session_cost_bytes_banked(pulsar_engine *e, int ctx_size, int n_banks) { return e ? e->session_cost_bytes_banked(ctx_size, n_banks) : 0; }
uint32_t pulsar_engine_bank_pool(int *pinned_by_env) { if (pinned_by_env) *pinned_by_env = gpu_graph_bank_pool_env_pinned(); return gpu_graph_bank_pool_n(); }
void pulsar_engine_set_bank_pool(uint32_t n_banks) { gpu_graph_bank_pool_set(n_banks); }
KvBytes pulsar_engine_demand_paged_bytes_per_bank(pulsar_engine *e, int ctx_size) { PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_BANKS, "the demand-paged bank price", KvBytes{}); return e ? e->family->banks->demand_paged_bytes(e, ctx_size) : KvBytes{}; }
uint64_t pulsar_engine_weights_resident_bytes(pulsar_engine *e) { return e ? e->weights_resident_bytes() : 0; }
int pulsar_engine_collect_imatrix(pulsar_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) { PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_IMATRIX, "imatrix collection", 1); return e ? e->collect_imatrix(dataset_path, output_path, ctx_size, max_prompts, max_tokens) : 1; }
void pulsar_engine_dump_tokens(pulsar_engine *e, const pulsar_tokens *tokens) { PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_CHAT, "token dumps (no tokenizer)", (void)0); if (e->family->tokenizer) e->family->tokenizer->dump(e, stdout, tokens); }
int pulsar_engine_routed_quant_bits(pulsar_engine *e) { return e ? e->routed_quant_bits() : 0; }
uint32_t pulsar_engine_kv_identity(pulsar_engine *e) {
    return pulsar_segstore_identity((uint32_t)pulsar_engine_model_id(e), (uint32_t)pulsar_engine_routed_quant_bits(e),
                                    e ? e->family->rope_id(e) : 0u);
}
bool pulsar_engine_has_spec_rounds(const pulsar_engine *e) { return e && e->drafter_ops && e->family->spec; }
bool pulsar_engine_can_rewind(const pulsar_engine *e) { return e && (e->family->caps & PULSAR_FAMILY_CAP_REWIND) != 0; }
bool pulsar_engine_has_fused_step(const pulsar_engine *e) { return e && e->family->session->decode_fused; }
uint32_t pulsar_engine_fused_heads_max(const pulsar_engine *e) {
    return pulsar_engine_has_fused_step(e) ? e->family->session->fused_heads_max : 0u;
}
bool pulsar_engine_has_mixed_prefill(const pulsar_engine *e) {
    return e && (e->family->caps & PULSAR_FAMILY_CAP_MIXED_PREFILL) != 0;
}
bool pulsar_engine_has_snapshots(const pulsar_engine *e) { return e && (e->family->caps & PULSAR_FAMILY_CAP_PAYLOAD) != 0; }
bool pulsar_engine_has_vision(const pulsar_engine *e) { return e && e->family->vision && e->vision_ready; }
pulsar_drafter_kind pulsar_engine_drafter(pulsar_engine *e) { return e ? e->family->drafter(e) : PULSAR_DRAFTER_NONE; }

/* L321: the encoder entries -- the family's (pulsar_family_encoder), behind PULSAR_FAMILY_CAP_ENCODE */
/** The family's encoder when the family declares PULSAR_FAMILY_CAP_ENCODE.  @param e the engine, or NULL
 *  @return the encoder, or NULL */
static const pulsar_family_encoder *encoder_of(const pulsar_engine *e) {
    return e && (e->family->caps & PULSAR_FAMILY_CAP_ENCODE) ? e->family->encoder : NULL;
}
pulsar_encoder_kind pulsar_engine_encoder(const pulsar_engine *e) {
    const pulsar_family_encoder *x = encoder_of(e);
    return x ? x->kind(e) : PULSAR_ENCODER_NONE;
}
uint32_t pulsar_engine_encode_dim(const pulsar_engine *e) {
    const pulsar_family_encoder *x = encoder_of(e);
    return x ? x->dim(e) : 0;
}
uint32_t pulsar_engine_encode_max_tokens(const pulsar_engine *e) {
    const pulsar_family_encoder *x = encoder_of(e);
    return x ? x->max_tokens(e) : 0;
}
bool pulsar_engine_encode_render(pulsar_engine *e, pulsar_encode_role role, const char *text, const char *doc,
                                 pulsar_tokens *out, char *err, size_t errlen) {
    PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_ENCODE, "encoding (a decoder family)", false);
    if (!e || !out) {
        if (err) snprintf(err, errlen, "no engine or no output");
        return false;
    }
    return e->family->encoder->render(e, role, text, doc, out, err, errlen);
}
bool pulsar_engine_is_parser(const pulsar_engine *e) {
    return e && (e->family->caps & PULSAR_FAMILY_CAP_PARSE) && e->family->parser;
}
int pulsar_engine_parse_page(pulsar_engine *e, const uint8_t *image, size_t len, const pulsar_parse_options *opt,
                             pulsar_tokens *ids, char **text, pulsar_parse_stats *stats, char *err, size_t errlen) {
    PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_PARSE, "page parsing (not a parser family)", -1);
    if (!e || !image || !ids || !text) {
        if (err) snprintf(err, errlen, "no engine, image or output");
        return -1;
    }
    return e->family->parser->parse(e, image, len, opt, ids, text, stats, err, errlen);
}
int pulsar_engine_encode(pulsar_engine *e, const pulsar_tokens *inputs, uint32_t n, float *out, char *err,
                         size_t errlen) {
    PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_ENCODE, "encoding (a decoder family)", -1);
    if (!e || (n && (!inputs || !out))) {
        if (err) snprintf(err, errlen, "no engine, inputs or output");
        return -1;
    }
    return e->family->encoder->encode(e, inputs, n, out, err, errlen);
}

void pulsar_session_set_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_progress(fn, ud); }
void pulsar_session_set_display_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_display_progress(fn, ud); }
void pulsar_session_set_cancel(pulsar_session *s, pulsar_session_cancel_fn fn, void *ud) { if (s) s->set_cancel(fn, ud); }
void pulsar_session_set_prefill_ahead(pulsar_session *s, const pulsar_tokens *prompt) { if (s) s->prefill_ahead = prompt; }
uint32_t pulsar_session_prompt_read_ahead(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, int run_end) {
    return run_end >= 0 ? pulsar_session_read_ahead(s, bank, prompt, (uint32_t)run_end) : 0u;
}
bool pulsar_engine_read_ahead_stats(pulsar_engine *e, pulsar_read_ahead_stats *out) {
    const pulsar_driver_ops *ops = e ? e->family->driver : NULL;
    if (!ops || !ops->read_ahead) {
        memset(out, 0, sizeof *out);
        return false;
    }
    pulsar_engram_read_ahead_stats(ops->read_ahead_pool(e), out);
    return true;
}
KvBytes pulsar_session_touched_kv_bytes(const pulsar_session *s) {
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "the touched-KV count", KvBytes{});
    if (!s) return KvBytes{};
    const pulsar_family_bank_ops *ops = session_banks(s);
    pulsar_session *ms = const_cast<pulsar_session *>(s);
    KvBytes bytes{};
    for (int b = 0, n = ops->count(ms); b < n; b++) bytes += ops->touched_kv_bytes(ms, (uint32_t)b);
    return bytes;
}
bool pulsar_session_bank_is_evicted(const pulsar_session *s, uint32_t bank) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank residency", false); return s ? session_banks(s)->is_evicted(s, bank) : false; }
KvBytes pulsar_session_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "the touched-KV count", KvBytes{}); return s ? session_banks(s)->touched_kv_bytes(s, bank) : KvBytes{}; }
KvBytes pulsar_session_quantum_growth_bytes_per_bank(pulsar_session *s, KvRows q) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "the bank growth price", KvBytes{}); return session_banks(s)->growth_bytes(s, q); }
/* The bank wrappers are defined with the mirror below (increment 2). */


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
 * image stitch at sync's resume path, the speculative
 * walk), and a second mirrored frame there would be a second, unbalanced half
 * of an operation the peer is not expecting.
 * ------------------------------------------------------------------------ */

bool pulsar_session_is_mirrored(const pulsar_session *s) {
    PULSAR_NVTX_FN();
    return s && s->engine && s->engine->tp && s->tp_session_id != 0;
}

uint64_t pulsar_session_checkpoint_digest(const pulsar_session *s) {
    uint64_t h = 1469598103934665603ull;
    if (!s) return h;
    const int n = s->checkpoint.len;
    const uint64_t len = (uint64_t)(n > 0 ? n : 0);
    for (int b = 0; b < 8; b++) { h ^= (len >> (8 * b)) & 0xffu; h *= 1099511628211ull; }
    for (int i = 0; i < n; i++) {
        const uint32_t t = (uint32_t)s->checkpoint.v[i];
        for (int b = 0; b < 4; b++) { h ^= (t >> (8 * b)) & 0xffu; h *= 1099511628211ull; }
    }
    return h;
}

/** The pair this session mirrors onto, or NULL when nothing should be mirrored
 * (pair off, or a session the engine handed out before the transport existed). */
/// @param s the session
/// @return the engine's transport, or NULL
static pulsar_tp *tp_mirror_target(pulsar_session *s) {
    return pulsar_session_is_mirrored(s) ? s->engine->tp : NULL;
}

/** Refusal shared by the mirrored operations: the pair is armed but its
 * transport is already dead, so no frame can be trusted in either direction. */
/// @param tp     the transport
/// @param err    receives the refusal when non-NULL
/// @param errlen bytes of `err`
/// @return 1 = refused (the transport failed), 0 = go on
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
/// @param s             the session
/// @param tp            the transport
/// @param operation     names the operation
/// @param own           the leader's own verdict
/// @param divergence_rc the result when a peer's verdict differs
/// @return the agreed verdict, or divergence_rc
static int tp_mirror_bank_verdict(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                  int own, int divergence_rc);

/// @param tp        the transport
/// @param operation names the operation
/// @param err       receives the refusal when non-NULL
/// @param errlen    bytes of `err`
/// @return 1 = refused (this rank is a worker), 0 = go on
static int tp_mirror_worker_drives_nothing(pulsar_tp *tp, const char *operation,
                                           char *err, size_t errlen) {
    PULSAR_NVTX("tp: wait worker verdict");
    if (pulsar_tp_rank(tp) == 0) return 0;
    if (err) snprintf(err, errlen,
                      "tp: rank %d is a worker; it does not drive %s -- it runs pulsar_tp_worker_run",
                      pulsar_tp_rank(tp), operation);
    return 1;
}

/** The collector every mirrored operation ends with on the leader: one ack per
 * peer.  It is read even when `body_rc` failed locally, because an unread ack
 * would be consumed by the NEXT operation, shifting every later frame by one. */
/// @param s         the session
/// @param tp        the transport
/// @param operation names the operation
/// @param body_rc   the leader's own result
/// @param err       receives a peer's failure when non-NULL
/// @param errlen    bytes of `err`
/// @return body_rc when it failed, else 0, or 1 when a peer failed
static int tp_mirror_leader_ack(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                int body_rc, char *err, size_t errlen) {
    char peer_err[256];
    peer_err[0] = '\0';
    /* An INTERRUPTED sync stopped at a chunk verdict both ranks took: the
     * worker answers at once, nothing to rescue. */
    if (body_rc != 0 && body_rc != PULSAR_SESSION_SYNC_INTERRUPTED) pulsar_tp_own_step_failed(tp, operation);
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

/** The cross-rank identity digest of a BATCHED step (decode_multiseq / decode_mixed): of the rows it headed, on
 * the device where they stay (L302; pulsar_gpu_rows_digest) -- every rank holds the same rows after the vocab
 * all-gather.  0 when the device digest failed (the ranks then disagree, loudly). */
/// @param s      the session
/// @param n_rows rows the step headed (0 digests to a fixed constant)
/// @return the digest
uint64_t pulsar_session_batch_digest(pulsar_session *s, uint32_t n_rows) {
    uint64_t d = 0;
    return n_rows == 0 ? 0x9E3779B97F4A7C15ull : s->rows_digest(s->dev.step_rows, 0, n_rows, &d) ? d : 0u;
}

uint64_t pulsar_session_fused_digest(pulsar_session *s, uint32_t n_dec, uint32_t n_heads) {
    uint64_t dec = 0, heads = 0x9E3779B97F4A7C15ull;
    if (n_dec && !s->rows_digest(s->dev.step_rows, 0, n_dec, &dec)) return 0;
    if (n_heads && !s->rows_digest(s->dev.step_rows, n_dec, n_heads, &heads)) return 0;
    return dec * 0x100000001b3ull ^ heads;
}

/* The live row's digest (an eval's, a speculative run's last row). */
/** The digest of the session's live row, on the device.  @param s the session  @return the digest, 0 on failure */
static uint64_t live_digest(pulsar_session *s) {
    uint64_t d = 0;
    return s->rows_digest(s->dev.live, 0, 1, &d) ? d : 0u;
}

/** Settle a pipelined eval's identity check (pulsar_session_eval defers it,
 * L241 4g-2): before logits VALUES leave the engine (copy, logprobs) and
 * before the session ends.  Token decisions (sample, argmax) do not settle --
 * the next eval does, before the token it drew is returned to its caller.
 * With no `err` the refusal is printed.  1 = nothing pending or it matched. */
/// @param s      the session
/// @param err    receives the refusal (NULL = printed)
/// @param errlen bytes of `err`
/// @return 1 when nothing was pending or it matched, else 0
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
    PULSAR_NVTX_FN();
    return s && tp_mirror_settle(s, err, errlen) ? 0 : 1;
}

/** The collector for the BATCHED steps (batch decode, mixed batch): the peers'
 * acks carry the digest of what their step produced (pulsar_session_batch_digest)
 * and must equal the leader's own -- the cross-rank identity check (L243).  The
 * eval pipelines its own check (pulsar_session_eval).  When the leader's own
 * body failed there is nothing to compare; the acks are drained in either shape
 * so the next frame is not shifted, and the local failure is the result. */
/// @param s         the session
/// @param tp        the transport
/// @param operation names the operation
/// @param body_rc   the leader's own result
/// @param n_rows    rows the step headed (digested)
/// @param err       receives a peer's failure when non-NULL
/// @param errlen    bytes of `err`
/// @return body_rc when it failed, else 0, or 1 when a peer failed or disagreed
static int tp_mirror_leader_ack_logits(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                       int body_rc, uint32_t n_rows, char *err, size_t errlen) {
    if (body_rc != 0) {
        pulsar_tp_drain_command_acks(tp);
        return body_rc;
    }
    const uint64_t own = pulsar_session_batch_digest(s, n_rows);
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
/// @param tp        the transport (marked failed)
/// @param operation names the operation
/// @param why       the reason (NULL = "no reason given")
void pulsar_tp_mirror_fail_void(pulsar_tp *tp, const char *operation, const char *why) {
    pulsar_tp_mark_failed(tp);
    fprintf(stderr, "pulsar: tp: the mirrored %s failed (%s); the pair is marked failed\n",
            operation, why ? why : "no reason given");
}

/** The ONE verdict on a leader's frame send, for every mirrored operation.  A
 * frame that did not ship -- to any peer, or only part of it -- leaves the
 * workers out of lockstep with this rank (a broadcast can fail after an
 * earlier peer took the frame, or mid-frame), so the pair is marked failed
 * here and every later mirrored operation refuses through tp_mirror_dead.
 * With an `err` buffer the reason goes there; without one it is printed by
 * pulsar_tp_mirror_fail_void.  1 = sent.
 *
 * The void operations the leader mirrors FIRE-AND-FORGET (destroy, rewind,
 * invalidate, save, abort, ...) come through here too: a `void` caller has
 * nowhere to put a peer's refusal, and several callers are the server's cache
 * and scheduler, whose timing follows LOCAL memory state -- so no ack is
 * collected; the worker loop's checks are the divergence alarm, and the next
 * ACKED operation carries the refusal back. */
/// @param tp        the transport
/// @param operation names the operation
/// @param sent      the send's result (nonzero = shipped)
/// @param err       receives the refusal (NULL = printed and the pair failed)
/// @param errlen    bytes of `err`
/// @return 1 when sent, else 0 (the pair marked failed)
static int tp_mirror_sent(pulsar_tp *tp, const char *operation, int sent,
                          char *err, size_t errlen) {
    if (sent != 0) return 1;
    if (!err) {
        pulsar_tp_mirror_fail_void(tp, operation, "the frame could not be shipped");
        return 0;
    }
    pulsar_tp_mark_failed(tp);
    snprintf(err, errlen, "tp: the mirrored %s could not be shipped to the workers; "
             "the pair is marked failed", operation);
    return 0;
}

int pulsar_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    PULSAR_NVTX_FN();
    /* A worker rank creates sessions only from the leader's frames (the loop
     * calls the member directly); a driver creating one here is refused before
     * the ordinal is consumed, so the two ranks' ordinals stay aligned. */
    if (e && e->tp && pulsar_tp_rank(e->tp) != 0) {
        fprintf(stderr, "pulsar: tp: rank %d is a worker; it does not create sessions -- "
                        "it runs pulsar_tp_worker_run\n", pulsar_tp_rank(e->tp));
        if (out) *out = NULL;
        return 1;
    }
    /* L284: a context past the model's trained positions runs positions it never saw -- a silently degraded
     * answer, not an error -- so it is refused here, once, for every family */
    const uint64_t trained = e ? e->family->trained_context(e) : 0;
    if (e && ctx_size > 0 && (uint64_t)ctx_size > trained) {
        fprintf(stderr, "pulsar: a %d-token context is past %s's trained %" PRIu64 " positions -- refusing "
                        "(the rope that runs is the container's rope_parameters, or --rope-yarn F over a native "
                        "one: Qwen's model card serves 1048576 at --rope-yarn 4)\n", ctx_size, e->family->name, trained);
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
    if (!tp_mirror_sent(tp, "session create",
                        pulsar_tp_send_session_create(tp, s->tp_session_id, ctx_size, gpu_graph_bank_pool_n()),
                        NULL, 0)) {
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
    PULSAR_NVTX_FN();
    if (!s) return;
    pulsar_tp *tp = tp_mirror_target(s);
    if (tp && pulsar_tp_rank(tp) == 0) {
        (void)tp_mirror_settle(s, NULL, 0);   /* a refusal is printed; the pair is marked failed */
        /* Fire-and-forget like rewind: the worker loop drops its registry
         * entry, and an unknown id there marks the pair failed. */
        (void)tp_mirror_sent(tp, "session destroy",
                             pulsar_tp_send_session_destroy(tp, s->tp_session_id), NULL, 0);
    }
    s->destroy();
}

/** The wire rows of a batched step: the engine's own row contract plus the
 * session.  Shared by the batched decode and the mixed step so the two frames
 * cannot encode a row differently. */
/// @param s    the session (its mirror id)
/// @param reqs [n] the rows
/// @param n    rows
/// @return the wire items, xmalloc'd (the caller frees them)
static pulsar_tp_batch_item *tp_mirror_rows(const pulsar_session *s, const pulsar_multiseq_req *reqs,
                                            uint32_t n) {
    pulsar_tp_batch_item *items = (pulsar_tp_batch_item *)xmalloc((size_t)n * sizeof(*items));
    for (uint32_t i = 0; i < n; i++) {
        items[i].session_id = s->tp_session_id;
        items[i].bank       = (int32_t)reqs[i].bank;
        items[i].pos        = reqs[i].pos.raw();   /* the wire's int32 */
        items[i].token      = reqs[i].token;
        items[i].reserved   = 0;
    }
    return items;
}

int pulsar_session_family_sync(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                               int n_images, char *err, size_t errlen) {
    /* a new request: what a truncated generation left in flight (the carry, the pendings -- their position
     * stamps cannot see a rebuild that lands on the same length) and a latched quench are the previous
     * request's.  Before L284 only DeepSeek's sync dropped them, and a Qwen bank whose quench latched served
     * every later request plain (the per-bank shadow kept the latch). */
    spec_lookahead_reset(s);
    return pulsar_session_sync_default(s, prompt, images, n_images, s->engine->family->session->sync, err, errlen);
}
int pulsar_session_family_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    const pulsar_family_session_ops *ops = s->engine->family->session;
    int rc = 1;
    if (s->checkpoint.len >= s->ctx_size) {
        if (err) snprintf(err, errlen, "eval: the context is full (%d of %d tokens)", s->checkpoint.len, s->ctx_size);
    } else if (!pulsar_session_token_is_id(s, token, err, errlen)) {
        /* L188: a refused sample (-1) must fail the request, not be clamped to token 0 by the embed kernel */
    } else if (!ops->sync->state_agrees(s)) {
        /* the bank's state is the authority (the sync's rule): a batched step, a multiseq step or a rewind moved
         * it off the view, and a row decoded here would land at the wrong position or attend another state */
        if (err) snprintf(err, errlen, "eval: bank %u's state is not the session's %d-token history; re-sync the "
                                       "session first", pulsar_session_live_bank(s), s->checkpoint.len);
    } else if (ops->eval_row(s, token, err, errlen) != 0) {
        s->checkpoint_valid = false;   /* the row's state is no position's */
    } else {
        token_vec_push(&s->checkpoint, token);
        s->logits_stale = false;
        rc = 0;
    }
    /* a token evaluated outside the speculative round (tool injection, a quenched request's plain step)
     * advances the state past any in-flight carry */
    s->spec.spec_carry_valid = false;
    return rc;
}
void pulsar_session_family_invalidate(pulsar_session *s) {
    if (s->engine->family->session->invalidate) s->engine->family->session->invalidate(s);
    pulsar_session_read_ahead_cancel(s, pulsar_session_live_bank(s));   /* L329 (b2): its prompt is gone */
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->logits_stale = true;
    s->live_images.n = 0;      /* the forgotten history's image records go with it (a run from 0 reads them) */
    spec_lookahead_reset(s);   /* the history the lookahead was conditioned on is gone */
}
void pulsar_session_note_committed(pulsar_session *s, const int *toks, int n) {
    if (!toks || n <= 0) return;
    for (int i = 0; i < n; i++) token_vec_push(&s->checkpoint, toks[i]);
    /* the rows were decoded into the batched lane's buffer, not s->logits; whether the view now describes the
     * state is the state's to say */
    s->logits_stale = true;
    s->checkpoint_valid = s->engine->family->session->sync->state_agrees(s);
}

int pulsar_session_sync(pulsar_session *s, const pulsar_tokens *prompt, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    return pulsar_session_sync_mm(s, prompt, NULL, 0, err, errlen);
}
int pulsar_session_sync_mm(pulsar_session *s, const pulsar_tokens *prompt,
                           const pulsar_image_ref *images, int n_images, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return pulsar_session_family_sync(s, prompt, images, n_images, err, errlen);
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
    /* An image the prefill cannot serve is refused HERE, before anything is
     * mirrored: the worker would refuse the same sync inside it, and a refused
     * mirrored sync fails the pair (2026-10-02: an image at ~205k tokens took
     * the pair down instead of failing its request). */
    if (n_images > 0) {
        char verr[384];
        if (!pulsar_image_spans_fit(s->engine, prompt->v, prompt->len, images, n_images, s->prefill_cap, NULL, verr,
                                    sizeof(verr))) {
            if (err) snprintf(err, errlen, "%s", verr);
            return 1;
        }
    }
    /* L250: agree on the starting state BEFORE anything runs.  The sync below
     * is ship-first-run-second: the leader starts computing at once and its
     * exchanges wait on the workers, so a worker that refused INSIDE the sync
     * would leave the leader spinning.  An unmirrored state change (the disk
     * KV restore) made exactly that: the leader resumed from 20,480 restored
     * tokens, the worker from 0, and the pair deadlocked mid-prefill.  Here
     * the leader states its cached position + prefix digest and waits for the
     * verdict; on a divergence both ranks are invalidated (a mirrored
     * operation, so they end in the SAME empty state) and this sync is
     * refused by name -- the pair stays up and the next request prefills cold. */
    {
        const int pos = s->checkpoint.len;
        const uint64_t digest = pulsar_session_checkpoint_digest(s);
        if (!tp_mirror_sent(tp, "sync check",
                            pulsar_tp_send_sync_check(tp, s->tp_session_id, pos, digest),
                            err, errlen)) return 1;
        int peers = 0;
        char perr[256];
        perr[0] = '\0';
        if (!pulsar_tp_wait_command_status(tp, s->tp_session_id, "sync check", &peers,
                                           perr, sizeof(perr))) {
            pulsar_tp_mirror_fail_void(tp, "sync check", perr);
            if (err) snprintf(err, errlen, "tp: the sync check could not be collected: %s", perr);
            return 1;
        }
        if (peers != 0) {
            fprintf(stderr, "pulsar: tp: sync check: a worker's cached state differs from this "
                            "rank's (leader pos %d digest %016llx); invalidating the session on "
                            "every rank and refusing this sync (L250)\n",
                    pos, (unsigned long long)digest);
            pulsar_session_invalidate(s);
            if (err) snprintf(err, errlen,
                              "tp: the ranks' cached state diverged before this sync (leader pos %d); "
                              "the session was invalidated on every rank -- retry prefills cold", pos);
            return 1;
        }
    }
    /* Ship first, run second, so both ranks prefill together instead of the
     * worker waiting out the leader's whole sync.  Images ride the frame
     * (increment 7): the tokens already carry the expanded sentinel blocks,
     * and every rank's own replicated tower encodes the same bytes. */
    const int sent = n_images > 0
        ? pulsar_tp_send_sync_mm(tp, s->tp_session_id, prompt->v, (uint32_t)prompt->len, images, (uint32_t)n_images)
        : pulsar_tp_send_sync(tp, s->tp_session_id, prompt->v, (uint32_t)prompt->len);
    if (!tp_mirror_sent(tp, n_images > 0 ? "sync with images" : "sync", sent, err, errlen)) return 1;
    s->tp_in_sync = true;
    const int body_rc = pulsar_session_family_sync(s, prompt, images, n_images, err, errlen);
    s->tp_in_sync = false;
    return tp_mirror_leader_ack(s, tp, "sync", body_rc, err, errlen);
}
int pulsar_image_block_starts(pulsar_engine *e, const pulsar_tokens *tokens, int len, int *starts, int cap) {
    if (!e || !tokens || len < 0 || len > tokens->len) return -1;
    int n = 0;
    for (int i = 0; i < len; ) {
        if (!pulsar_image_is_sentinel(e, tokens->v[i])) { i++; continue; }   /* L281: the family's geometry */
        int blk = 0;
        if (!pulsar_image_block_extent(e, tokens->v, len, i, &blk)) return -1;
        if (n < cap && starts) starts[n] = i;
        n++;
        i += blk;
    }
    return n;
}

char *pulsar_history_text(pulsar_engine *e, const pulsar_tokens *tokens, size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!e || !tokens) return NULL;
    std::string out;
    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i], blk = 1;
        if (pulsar_image_is_sentinel(e, id)) {   /* the family's geometry: the block stands for its placeholder */
            if (!pulsar_image_block_extent(e, tokens->v, tokens->len, i, &blk) || blk <= 0) return NULL;
            id = e->family->vision->placeholder_id(e);
        }
        size_t n = 0;
        char *piece = pulsar_token_text(e, id, &n);
        out.append(piece, n);
        free(piece);
        i += blk - 1;
    }
    if (out_len) *out_len = out.size();
    char *r = (char *)xmalloc(out.size() + 1);
    memcpy(r, out.data(), out.size());
    r[out.size()] = '\0';
    return r;
}

uint64_t pulsar_image_hash(const pulsar_image_ref *img) { return pulsar_image_content_hash(img); }

int pulsar_session_image_hashes(pulsar_session *s, uint64_t *hashes, int cap) {
    if (!s) return 0;
    for (uint32_t i = 0; i < s->live_images.n && (int)i < cap; i++) hashes[i] = s->live_images.b[i].content;
    return (int)s->live_images.n;
}

TokPos pulsar_session_persist_end(pulsar_session *s) {
    if (!s || !s->checkpoint_valid) return TokPos{0};
    /* the core's walk answers -1 for malformed blocks: it enters as none() */
    return TokPos{pulsar_image_persist_end(s->engine->family->vision, s->engine, s->checkpoint.v, s->checkpoint.len,
                                           &s->live_images)};
}

int pulsar_expand_image_placeholders(pulsar_engine *e, const pulsar_tokens *prompt, int from,
                                     pulsar_image_ref *images, int n_images,
                                     pulsar_tokens *out, char *err, size_t errlen) {
    PULSAR_FAMILY_REQUIRES_E(e, PULSAR_FAMILY_CAP_VISION, "image placeholders", 0);
    if (err && errlen) err[0] = '\0';
    if (!e || !prompt || !out || n_images < 0 || (n_images > 0 && !images)) {
        if (err) snprintf(err, errlen, "image request is missing its prompt, images, or output buffer");
        return 0;
    }
    /* L268: the family's front, through the core's walk (image_front.cpp) */
    const bool ok = pulsar_image_expand(e, prompt, from, images, n_images, out, err, errlen);
    if (!ok) pulsar_tokens_free(out);
    return ok ? 1 : 0;
}
TokCount pulsar_session_common_prefix(pulsar_session *s, const pulsar_tokens *prompt) { return TokCount{s->common_prefix(prompt)}; }
void pulsar_session_prefix_match(pulsar_session *s, const pulsar_tokens *prompt, pulsar_prefix_match *out) { if (s) { s->prefix_match(prompt, out); } else if (out) { out->live_cut = TokCount{0}; out->prompt_cut = PromptToks{0}; out->seamed = false; } }
int pulsar_session_argmax(pulsar_session *s) { return s->argmax(); }
int pulsar_session_argmax_excluding(pulsar_session *s, int excluded_id) { return s ? s->argmax_excluding(excluded_id) : -1; }
int pulsar_session_sample(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) { return s->sample(temperature, top_k, top_p, min_p, rng); }
int pulsar_session_draw(pulsar_session *s, const pulsar_row_draw *d) { return s && d ? s->draw(d) : PULSAR_SAMPLE_REFUSED; }
int pulsar_session_draw_rows(pulsar_session *s, const uint32_t *rows, const pulsar_row_draw *d, uint32_t n, int *tokens) {
    return s && rows && d && tokens && n ? s->draw_step_rows(rows, d, n, tokens) : 1;
}
int pulsar_session_step_row_to_bank(pulsar_session *s, uint32_t row, uint32_t bank) {
    return s && s->step_row_to_bank(row, bank) ? 0 : 1;
}
int pulsar_session_step_row_logprobs(pulsar_session *s, uint32_t row, int token, pulsar_token_score *chosen,
                                     int *have_chosen, pulsar_token_score *top, int k, int *n_top) {
    return s && chosen && have_chosen && n_top && (top || k == 0) ? s->step_row_logprobs(row, token, chosen, have_chosen, top, k, n_top) : 1;
}
int pulsar_session_top_logprobs(pulsar_session *s, pulsar_token_score *out, int k) { return s && tp_mirror_settle(s, NULL, 0) ? s->top_logprobs(out, k) : 0; }
int pulsar_session_token_logprob(pulsar_session *s, int token, pulsar_token_score *out) { return s && tp_mirror_settle(s, NULL, 0) ? s->token_logprob(token, out) : 0; }
int pulsar_session_copy_logits(pulsar_session *s, float *out, int cap) { return s && tp_mirror_settle(s, NULL, 0) ? s->copy_logits(out, cap) : 0; }
int pulsar_session_set_logits(pulsar_session *s, const float *logits, int n) {
    PULSAR_NVTX_FN();
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
    if (!tp_mirror_sent(tp, "set logits",
                        pulsar_tp_send_set_logits(tp, s->tp_session_id, logits, (uint32_t)n),
                        NULL, 0)) return 1;
    return tp_mirror_bank_verdict(s, tp, "set logits", s->set_logits(logits, n) != 0 ? 1 : 0, 1);
}
int pulsar_session_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return pulsar_session_family_eval(s, token, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "eval", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    /* The frame's seq is this session's decode position -- the number of tokens
     * whose KV the graph holds -- so the worker can do more than trust the
     * leader's token: it refuses when the two ranks are not at the same place.
     * Without that check a rank that fell behind would decode the right token
     * at the wrong position and produce confident nonsense. */
    const uint64_t pos = (uint64_t)s->checkpoint.len;
    if (!tp_mirror_sent(tp, "eval", pulsar_tp_send_eval(tp, s->tp_session_id, pos, token),
                        err, errlen)) return 1;
    /* PIPELINED identity check (L241 4g-2): the previous eval's digest ack is
     * settled here, after this step's body -- it has been in the socket since
     * the worker finished that step -- and this step's own digest is DEFERRED
     * instead of waited on, so the leader samples and ships the next token
     * while the worker is still acking this one.  No token leaves unchecked:
     * the one drawn from these logits is returned by the NEXT eval, which
     * settles them first; logits values settle before they are copied out
     * (tp_mirror_settle).  The body runs even when the settle failed, so both
     * ranks finish this step and its ack is drained, not left in the socket. */
    const int body_rc = pulsar_session_family_eval(s, token, err, errlen);
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
    char what[96];
    snprintf(what, sizeof(what), "eval at position %llu", (unsigned long long)pos);
    if (!pulsar_tp_defer_command_ack_digest(tp, s->tp_session_id, what, live_digest(s))) {
        pulsar_tp_drain_command_acks(tp);
        if (err) snprintf(err, errlen, "tp: could not defer the eval's identity check");
        return 1;
    }
    return 0;
}
int pulsar_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, float *logits, int logits_cap, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->engine->family->session->decode_multiseq(s, reqs, n, logits, logits_cap, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "the batched decode", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    if (!reqs || n == 0) {
        if (err) snprintf(err, errlen, "tp: refusing to mirror an empty batch");
        return 1;
    }
    pulsar_tp_batch_item *items = tp_mirror_rows(s, reqs, n);
    const int sent = pulsar_tp_send_eval_batch(tp, items, n);
    free(items);
    if (!tp_mirror_sent(tp, "batch decode", sent, err, errlen)) return 1;
    return tp_mirror_leader_ack_logits(s, tp, "batch decode",
                                       s->engine->family->session->decode_multiseq(s, reqs, n, logits, logits_cap, err, errlen),
                                       n, err, errlen);
}
int pulsar_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits, int logits_cap, uint32_t *out_n_rows, uint32_t max_head_runs, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->engine->family->session->decode_mixed(s, reqs, n_rows, logits, logits_cap, out_n_rows,
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
    if (!tp_mirror_sent(tp, "mixed batch", sent, err, errlen)) return 1;
    /* The rows the step headed are what the workers digest, whether or not this caller asked for the count
     * (the server's decode-only lane passes NULL): the leader digests the same rows (L266 -- it digested
     * none, so every NULL-count mixed batch on a pair read as the ranks' logits differing). */
    uint32_t headed = 0;
    uint32_t *rows_out = out_n_rows ? out_n_rows : &headed;
    const int body_rc = s->engine->family->session->decode_mixed(s, reqs, n_rows, logits, logits_cap,
                                        rows_out, max_head_runs, err, errlen);
    return tp_mirror_leader_ack_logits(s, tp, "mixed batch", body_rc, *rows_out, err, errlen);
}
int pulsar_session_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                                const pulsar_fused_shape *shape, float *logits, int logits_cap,
                                uint32_t *out_n_rows, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    /* a family without the op has no fused step (its prompt rows take the classic sync) */
    if (!pulsar_engine_has_fused_step(s->engine)) {
        if (err) snprintf(err, errlen, "%s: no fused step", s->engine->family->name);
        return 1;
    }
    pulsar_tp *tp = tp_mirror_target(s);
    const auto fused = pulsar_session_fused_local;
    if (!tp) return fused(s, reqs, n_rows, shape, logits, logits_cap, out_n_rows, err, errlen);
    if (tp_mirror_worker_drives_nothing(tp, "the fused step", err, errlen)) return 1;
    if (tp_mirror_dead(tp, err, errlen)) return 1;
    if (!reqs || n_rows == 0 || !shape) {
        if (err) snprintf(err, errlen, "tp: refusing to mirror an empty fused step");
        return 1;
    }
    /* The shape rides the frame: a worker has no caller to take the decode /
     * prefill split and the head flags from, and both ranks must head the same
     * rows.  `out_n_rows` is output and stays local on each rank. */
    pulsar_tp_batch_item *items = tp_mirror_rows(s, reqs, n_rows);
    const int sent = pulsar_tp_send_fused_batch(tp, items, n_rows, shape);
    free(items);
    if (!tp_mirror_sent(tp, "fused batch", sent, err, errlen)) return 1;
    uint32_t got = 0;
    const int body_rc = fused(s, reqs, n_rows, shape, logits, logits_cap, &got, err, errlen);
    if (out_n_rows) *out_n_rows = got;
    if (body_rc != 0) {
        pulsar_tp_drain_command_acks(tp);
        return body_rc;
    }
    char peer_err[256];
    peer_err[0] = '\0';
    if (!pulsar_tp_wait_command_ack_digest(tp, s->tp_session_id, "fused batch",
                                           pulsar_session_fused_digest(s, shape->n_dec, got - shape->n_dec),
                                           peer_err, sizeof(peer_err))) {
        if (err) snprintf(err, errlen, "tp: a worker failed the mirrored fused batch: %s", peer_err);
        return 1;
    }
    return 0;
}
int pulsar_session_bank_count(pulsar_session *s) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "the bank pool", 0); return s ? session_banks(s)->count(s) : 0; }
/* ---- The bank surface (increment 2).  Bank SELECTION already rode the
 * decode rows; these mirror the leader scheduler's bank DECISIONS -- save,
 * restore, repoint -- so both ranks' pools hold the same
 * state.  The leader's decision is the authority and the ordinal names the
 * target.  save is void and fire-and-forget; the others return a verdict the
 * ranks must AGREE on: the same inputs on the same state give the same
 * result, so a split verdict is a divergence (marked failed, refused), not a
 * vote. */
static int tp_mirror_bank_verdict(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                  int own, int divergence_rc) {
    PULSAR_NVTX("tp: wait worker verdict");
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
 * its live row and every peer's positive verdict must match it
 * (the cross-rank identity check, L243).  An own verdict of 0 (the run failed
 * here) collects plainly; a peer that ran anyway answers with a digest, which
 * the plain collect refuses as a shape mismatch -- a split either way. */
/// @param s             the session
/// @param tp            the transport
/// @param operation     names the operation
/// @param own           the leader's own verdict (> 0 carries the live row's digest)
/// @param divergence_rc the result when the ranks split
/// @return the agreed verdict, or divergence_rc (the pair marked failed)
static int tp_mirror_bank_verdict_logits(pulsar_session *s, pulsar_tp *tp, const char *operation,
                                         int own, int divergence_rc) {
    PULSAR_NVTX("tp: wait worker verdict");
    if (own <= 0) return tp_mirror_bank_verdict(s, tp, operation, own, divergence_rc);
    char err[256];
    err[0] = '\0';
    int peers = 0;
    const uint64_t digest = live_digest(s);
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
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank repoint", 1);
    if (!s) return 1;
    const pulsar_family_bank_ops *ops = session_banks(s);
    if (!ops->repoint) return pulsar_session_bank_state_restore(s, bank) ? 0 : 1;   /* its repoint is a restore */
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return ops->repoint(s, bank);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank repoint", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return 1;
    }
    if (!tp_mirror_sent(tp, "bank repoint", pulsar_tp_send_bank_repoint(tp, s->tp_session_id, bank),
                        NULL, 0)) return 1;
    return tp_mirror_bank_verdict(s, tp, "bank repoint", ops->repoint(s, bank), 1);
}
void pulsar_session_bank_state_save(pulsar_session *s, uint32_t bank) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank state save", (void)0);
    if (!s) return;
    /* the family's save, mirrored (L266: a bank-pool family's once returned before the frame, so a Qwen worker
     * never followed it) */
    auto local = [&]() { session_banks(s)->save(s, bank); };
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { local(); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank state save", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "bank state save", err);
        return;
    }
    if (!tp_mirror_sent(tp, "bank state save",
                        pulsar_tp_send_bank_state_save(tp, s->tp_session_id, bank), NULL, 0)) return;
    local();
}
bool pulsar_session_bank_state_restore(pulsar_session *s, uint32_t bank) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank state restore", false);
    if (!s) return false;
    auto local = [&]() { return session_banks(s)->restore(s, bank); };
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return local();
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "bank state restore", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return false;
    }
    if (!tp_mirror_sent(tp, "bank state restore",
                        pulsar_tp_send_bank_state_restore(tp, s->tp_session_id, bank),
                        NULL, 0)) return false;
    const int own = local() ? 0 : 1;
    return tp_mirror_bank_verdict(s, tp, "bank state restore", own, 1) == 0;
}
/* ---- The eviction guard's physical-bank pair (increment 6): verdicts, each
 * rank frees or allocates its own replicated bank. */
/** Free or re-back a bank's physical KV on this rank, mirrored as a verdict on a TP leader.
 *  @param s       the session
 *  @param freeing nonzero = free_physical, 0 = alloc_physical
 *  @param bank    the bank
 *  @return true when done (on a pair: on every rank) */
static bool tp_mirror_bank_physical(pulsar_session *s, int freeing, uint32_t bank) {
    const char *operation = freeing ? "bank free physical" : "bank alloc physical";
    const pulsar_family_bank_ops *ops = session_banks(s);
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return freeing ? ops->free_physical(s, bank) : ops->alloc_physical(s, bank);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, operation, err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return false;
    }
    const int sent = freeing ? pulsar_tp_send_bank_free_physical(tp, s->tp_session_id, bank)
                             : pulsar_tp_send_bank_alloc_physical(tp, s->tp_session_id, bank);
    if (!tp_mirror_sent(tp, operation, sent, NULL, 0)) return false;
    const int own = (freeing ? ops->free_physical(s, bank) : ops->alloc_physical(s, bank)) ? 0 : 1;
    return tp_mirror_bank_verdict(s, tp, operation, own, 1) == 0;
}
bool pulsar_session_bank_free_physical(pulsar_session *s, uint32_t bank) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank residency", false);
    return s ? tp_mirror_bank_physical(s, 1, bank) : false;
}
bool pulsar_session_bank_alloc_physical(pulsar_session *s, uint32_t bank) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank residency", false);
    return s ? tp_mirror_bank_physical(s, 0, bank) : false;
}
TokPos pulsar_bank_pos_of(uint32_t bank, uint32_t n_banks, const pulsar_tokens *history) {
    if (bank >= n_banks) {
        static bool said = false;   /* once (rule 9): the caller keeps its none() every time */
        if (!said) fprintf(stderr, "pulsar: bank_pos of bank %u, outside the pool of %u -- refusing\n", bank, n_banks);
        said = true;
        return TokPos::none();
    }
    return history ? TokPos{history->len} : TokPos::none();
}
TokPos pulsar_session_bank_pos(pulsar_session *s, uint32_t bank) {
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "per-bank state", TokPos::none());
    if (!s) return TokPos::none();
    return pulsar_bank_pos_of(bank, (uint32_t)session_banks(s)->count(s), pulsar_bank_history(s, bank));
}
/* The session's grid checkpoints: the family's store (L266). */
pulsar_ckpt_store *pulsar_session_kv_store(pulsar_session *s) {
    return s ? session_banks(s)->kv_store(s) : NULL;
}
uint32_t pulsar_session_live_bank(pulsar_session *s) {
    return session_banks(s)->live(s);
}
uint32_t pulsar_session_resume_point(pulsar_session *s, uint32_t bank, int common, int prompt_len) {
    if (!s || common <= 0 || prompt_len <= 0) return 0;
    uint32_t limit = (uint32_t)common;
    const int pf = pulsar_session_bank_prefill_frontier(s, bank).raw();   /* L334: the body stays raw (P2b) */
    const uint32_t frontier = pf < 0 ? 0u : (uint32_t)pf;
    if (limit > frontier) limit = frontier;
    if (limit > (uint32_t)prompt_len - 1u) limit = (uint32_t)prompt_len - 1u;
    const pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    if (!st || !st->ops) return 0u;
    const uint32_t G = limit ? pulsar_ckpt_best(st, bank, limit) : 0u;
    /* L289 / L311: the bank's records (its prompt end, its message endpoint) -- the prefill's own state at an exact
     * T the prompt carries (T within the match and the prefilled history).  A record AT the prompt's end resumes
     * there with the logits its prefill left, no row evaluated, every family.  One below it resumes from T and
     * prefills the rest only on a split-invariant family, where that IS the cold prefill's bytes; elsewhere a longer
     * prompt resumes from the grid (only a grid cut is the cold prefill's, L183). */
    uint32_t rlimit = (uint32_t)common;
    if (rlimit > frontier) rlimit = frontier;
    if (rlimit > (uint32_t)prompt_len) rlimit = (uint32_t)prompt_len;
    const pulsar_tokens *h = pulsar_bank_history(s, bank);
    const pulsar_ckpt_record *r =
        h ? pulsar_ckpt_record_best(st, bank, h->v, (uint32_t)h->len, rlimit, (uint32_t)prompt_len) : NULL;
    if (r && r->T > G && (r->T == (uint32_t)prompt_len || st->ops->split_invariant)) return r->T;
    return G;
}

const pulsar_ckpt_record *pulsar_session_resume_record(pulsar_session *s, uint32_t bank, uint32_t T, int prompt_len) {
    const pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const pulsar_tokens *h = pulsar_bank_history(s, bank);
    /* below the prompt's end a grid checkpoint at T is the restore; AT it only a record (its logits) serves */
    if (!st || !st->ops || !h || T == 0u || (T < (uint32_t)prompt_len && pulsar_ckpt_best(st, bank, T) == T))
        return NULL;
    const pulsar_ckpt_record *r = pulsar_ckpt_record_best(st, bank, h->v, (uint32_t)h->len, T, (uint32_t)prompt_len);
    return r && r->T == T ? r : NULL;
}
bool pulsar_session_bank_continues(pulsar_session *s, uint32_t bank, int len) {
    const pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    if (!st || !st->ops || len < 0) return false;
    return pulsar_session_bank_prefill_frontier(s, bank).raw() >= len &&   /* L334: raw body (P2b) */
           (st->ops->split_invariant || (uint32_t)len % st->ops->resume_grid == 0u) &&
           !pulsar_session_bank_comp_stale(s, bank);
}
uint32_t pulsar_session_resume_grid(const pulsar_session *s) {
    const pulsar_ckpt_store *st = pulsar_session_kv_store(const_cast<pulsar_session *>(s));
    return st && st->ops ? st->ops->resume_grid : 0u;
}
bool pulsar_session_split_invariant(const pulsar_session *s) {
    const pulsar_ckpt_store *st = pulsar_session_kv_store(const_cast<pulsar_session *>(s));
    return st && st->ops && st->ops->split_invariant;
}
TokPos pulsar_session_bank_prefill_frontier(pulsar_session *s, uint32_t bank) {
    return TokPos{s ? (int)session_banks(s)->prefill_frontier(s, bank) : 0}; }
int pulsar_session_bank_spec_depth(pulsar_session *s, uint32_t bank) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", 0); return pulsar_spec_bank_depth(s, bank); }
bool pulsar_session_bank_comp_stale(pulsar_session *s, uint32_t bank) {
    const pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    return st && st->ops && st->ops->stale(st->state, bank);
}
const pulsar_tokens *pulsar_session_bank_tokens(pulsar_session *s, uint32_t bank) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "per-bank state", NULL); return pulsar_bank_history(s, bank); }
/* The bank's committed history, the family's reader or DeepSeek's (the prefix readers below are
 * ONE body over it; L272 P0 folded the two copies). */
/** The bank's committed history (pulsar_bank_history).  @param s the session  @param bank the bank
 *  @return the tokens, or NULL */
static const pulsar_tokens *session_bank_history(pulsar_session *s, uint32_t bank) {
    return pulsar_bank_history(s, bank);
}
TokCount pulsar_session_bank_common_prefix(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "per-bank state", TokCount{0});
    return TokCount{pulsar_tokens_common_prefix(session_bank_history(s, bank), prompt)}; }
/* L115: the prefix-reuse authority against one bank's committed history. */
void pulsar_session_bank_prefix_match(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, pulsar_prefix_match *out) { if (s && !pulsar_family_require(s->engine, PULSAR_FAMILY_CAP_BANKS, "per-bank state")) s = NULL;
    if (!out) return;
    out->live_cut = TokCount{0}; out->prompt_cut = PromptToks{0}; out->seamed = false;
    if (!s) return;
    const pulsar_tokens *t = session_bank_history(s, bank);
    if (t && prompt) pulsar_tokens_prefix_match(s->engine, t->v, t->len, prompt->v, prompt->len, out); }
int pulsar_session_note_prefilled(pulsar_session *s, const int *toks, int n, int head) {
    PULSAR_NVTX_FN();
    if (!s) return 1;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->note_prefilled(toks, n, head);
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "note prefilled", err, sizeof(err)) ||
        tp_mirror_dead(tp, err, sizeof(err))) {
        fprintf(stderr, "pulsar: %s\n", err);
        return 1;
    }
    if (n <= 0 || !toks) return 1;
    /* The head row is named by index: each rank reads its OWN fused step's
     * block, whose digest the step's ack already matched across ranks. */
    if (!tp_mirror_sent(tp, "note prefilled",
                        pulsar_tp_send_note_prefilled(tp, s->tp_session_id, toks, (uint32_t)n, head),
                        NULL, 0)) return 1;
    return tp_mirror_bank_verdict(s, tp, "note prefilled", s->note_prefilled(toks, n, head), 1);
}
void pulsar_session_note_committed_tokens(pulsar_session *s, const int *toks, int n) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "per-bank state", (void)0);
    if (!s) return;
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { pulsar_session_note_committed(s, toks, n); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "note committed tokens", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "note committed tokens", err);
        return;
    }
    if (n < 0 || (n > 0 && !toks)) return;
    if (!tp_mirror_sent(tp, "note committed tokens",
                        pulsar_tp_send_note_committed(tp, s->tp_session_id, toks, (uint32_t)n), NULL, 0)) return;
    pulsar_session_note_committed(s, toks, n);
}
/* ---- The speculative round family (increment 4).  Each public entry point
 * ships a frame carrying every input the call reads INCLUDING the rng state
 * as it stands before the call, runs the local implementation, and requires
 * the workers' agreed verdict to equal its own.  With identical inputs on
 * identical state -- same logits after the vocab all-gather, same drafts,
 * same rng -- the walk is a deterministic function, so a split verdict is a
 * divergence.  Verdicts that can be -1 ride the wire as value + 1. */
/** A speculative frame's command for the session and bank, every other field zero.
 *  @param s    the session (its mirror id)
 *  @param bank the bank
 *  @return the command */
static pulsar_tp_spec_command tp_spec_cmd(const pulsar_session *s, int bank) {
    pulsar_tp_spec_command c;
    memset(&c, 0, sizeof(c));
    c.session_id = s->tp_session_id;
    c.bank = bank;
    return c;
}
/* the bank a mirrored speculative command runs on: the family's live bank (L272 P6: this read DeepSeek's graph,
 * so a Qwen pair's commands named bank 0 whichever bank was live) */
/** @param s the session  @return the live bank */
static int tp_spec_live_bank(const pulsar_session *s) {
    return (int)pulsar_session_live_bank(const_cast<pulsar_session *>(s));
}
/* Returns 1 when the operation may run locally without mirroring (pair off),
 * 0 when it was refused (err filled), 2 when it must be mirrored. */
/// @param s         the session
/// @param operation names the operation
/// @param tp_out    receives the transport (NULL = pair off)
/// @param err       receives the refusal
/// @param errlen    bytes of `err`
/// @return 1 run locally, 0 refused, 2 mirror it
static int tp_spec_route(pulsar_session *s, const char *operation, pulsar_tp **tp_out,
                         char *err, size_t errlen) {
    *tp_out = tp_mirror_target(s);
    if (!*tp_out) return 1;
    if (tp_mirror_worker_drives_nothing(*tp_out, operation, err, errlen)) return 0;
    if (tp_mirror_dead(*tp_out, err, errlen)) return 0;
    return 2;
}

int pulsar_session_generate_speculative(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng, int max_tokens, int *accepted, int accepted_cap, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    /* the round API's single lane (session_spec.cpp), whichever family and drafter are loaded (L272 P1);
     * mirrored the same way on a pair */
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1);
    if (!s) return 0;
    auto local = [&]() {
        return s->generate_speculative(temperature, top_k, top_p, min_p, rng, max_tokens,
                                       accepted, accepted_cap, err, errlen);
    };
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "generate_speculative", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return local();
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = max_tokens; c.i2 = accepted_cap;   /* i1: unused since L284 (the stop set is the engine's) */
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (!tp_mirror_sent(tp, "generate_speculative", pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_GENERATE_SPECULATIVE, &c, NULL, NULL),
                        err, errlen)) return -1;
    const int own = local();
    if (own < 0) pulsar_tp_own_step_failed(tp, "generate_speculative");
    const int agreed = tp_mirror_bank_verdict_logits(s, tp, "generate_speculative", own + 1, -1);
    return agreed < 0 ? -1 : own;
}
int pulsar_session_spec_next_base(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1);
    if (!s) return -1;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_next_base", &tp, err, sizeof(err));
    if (route == 0) { fprintf(stderr, "pulsar: %s\n", err); return -1; }
    if (route == 1) return pulsar_session_spec_next_base_local(s, temperature, top_k, top_p, min_p, NULL, rng);
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (!tp_mirror_sent(tp, "spec_next_base",
                        pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_NEXT_BASE, &c, NULL, NULL),
                        NULL, 0)) return -1;
    const int own = pulsar_session_spec_next_base_local(s, temperature, top_k, top_p, min_p, NULL, rng);
    const int agreed = tp_mirror_bank_verdict(s, tp, "spec_next_base", own + 1, -1);
    return agreed < 0 ? -1 : own;
}
int pulsar_session_spec_round_begin(pulsar_session *s, pulsar_spec_round *r, int first_token, int max_tokens, int accepted_cap, float temperature, int top_k, float top_p, float min_p, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1);
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_begin", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_spec_round_begin_local(s, r, first_token, max_tokens, accepted_cap, temperature, top_k, top_p, min_p, err, errlen);
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = first_token; c.i1 = max_tokens; c.i2 = accepted_cap;
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    if (!tp_mirror_sent(tp, "spec_round_begin", pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_BEGIN, &c, NULL, NULL),
                        err, errlen)) return -1;
    const int own = pulsar_session_spec_round_begin_local(s, r, first_token, max_tokens, accepted_cap, temperature, top_k, top_p, min_p, err, errlen);
    return tp_mirror_bank_verdict(s, tp, "spec_round_begin", own == 0 ? 0 : 1, -1) < 0 ? -1 : own;
}
int pulsar_session_spec_round_end(pulsar_session *s, pulsar_spec_round *r, int first_token, float temperature, int top_k, float top_p, float min_p, uint64_t *rng, uint32_t row0, int *accepted, int accepted_cap, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1);
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_end", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_spec_round_end_local(s, r, first_token, temperature, top_k, top_p, min_p, 0u, NULL, 0u, rng, row0, accepted, accepted_cap, err, errlen);
    /* The rows are NOT shipped: every rank holds the same rows on its device from its own mirrored forward (the vocab
     * all-gather made them identical), so only row0 crosses. */
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = first_token; c.i2 = accepted_cap; c.i3 = (int32_t)row0;   /* i1: unused since L284 */
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.rng = rng ? *rng : 0;
    if (!tp_mirror_sent(tp, "spec_round_end", pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_END, &c, NULL, NULL),
                        err, errlen)) return -1;
    const int own = pulsar_session_spec_round_end_local(s, r, first_token, temperature, top_k, top_p, min_p, 0u, NULL, 0u, rng, row0, accepted, accepted_cap, err, errlen);
    const int agreed = tp_mirror_bank_verdict(s, tp, "spec_round_end", own + 1, -1);
    return agreed < 0 ? -1 : own;
}
void pulsar_session_spec_round_abort(pulsar_session *s, pulsar_spec_round *r) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", (void)0);
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_round_abort", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_round_abort", err); return; }
    if (route == 1) { pulsar_session_spec_round_abort_local(s, r); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    if (!tp_mirror_sent(tp, "spec_round_abort",
                        pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ROUND_ABORT, &c, NULL, NULL), NULL, 0)) return;
    pulsar_session_spec_round_abort_local(s, r);
}
void pulsar_session_spec_arm_capture(pulsar_session *s, uint32_t n_rows) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", (void)0);
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_arm_capture", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_arm_capture", err); return; }
    if (route == 1) { pulsar_session_spec_arm_capture_local(s, n_rows); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = (int32_t)n_rows;
    if (!tp_mirror_sent(tp, "spec_arm_capture",
                        pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_ARM_CAPTURE, &c, NULL, NULL), NULL, 0)) return;
    pulsar_session_spec_arm_capture_local(s, n_rows);
}
/* L315: the verify curve's profile on a pair -- every rank runs it in step from the frame, the leader keeps its
 * timings (verify_curve.cpp).  Verdict: 0 ran, 1 failed. */
int pulsar_session_verify_profile(pulsar_session *s, uint32_t rows_max, double *base, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s) return -1;
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "verify_profile", &tp, err, errlen);
    if (route == 0) return -1;
    if (route == 1) return pulsar_session_verify_profile_local(s, rows_max, base, err, errlen) ? 0 : -1;
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.i0 = (int32_t)rows_max;
    c.i1 = base ? 1 : 0;   /* L306: timed (no cached curve) or the warmup's walk alone -- every rank runs the same reps */
    if (!tp_mirror_sent(tp, "verify_profile",
                        pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_VERIFY_PROFILE, &c, NULL, NULL), err, errlen))
        return -1;
    const bool own = pulsar_session_verify_profile_local(s, rows_max, base, err, errlen);
    if (!own) pulsar_tp_own_step_failed(tp, "verify_profile");
    return tp_mirror_bank_verdict(s, tp, "verify_profile", own ? 0 : 1, -1) == 0 ? 0 : -1;
}
int pulsar_session_spec_redraft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1);
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
    if (!tp_mirror_sent(tp, "spec_redraft_batch", sent, err, errlen)) return -1;
    const int own = pulsar_session_spec_redraft_batch_local(s, rounds, banks, rngs, n, err, errlen);
    if (own != 0) pulsar_tp_own_step_failed(tp, "spec_redraft_batch");
    return tp_mirror_bank_verdict(s, tp, "spec_redraft_batch", own == 0 ? 0 : 1, -1) < 0 ? -1 : own;
}
void pulsar_session_spec_redraft_commit(pulsar_session *s, pulsar_spec_round *r) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", (void)0);
    if (!s) return;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, "spec_redraft_commit", &tp, err, sizeof(err));
    if (route == 0) { pulsar_tp_mirror_fail_void(tp, "spec_redraft_commit", err); return; }
    if (route == 1) { pulsar_session_spec_redraft_commit_local(s, r); return; }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    if (!tp_mirror_sent(tp, "spec_redraft_commit",
                        pulsar_tp_send_spec(tp, PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT, &c, NULL, NULL), NULL, 0)) return;
    pulsar_session_spec_redraft_commit_local(s, r);
}
/* ---- L260: the batched lane's per-bank bookkeeping as ONE frame per phase.
 * The frame carries every step's inputs (its rng as it stands BEFORE the
 * phase); both ranks run the same local batch on the same state and the
 * verdict is the phase's outcome fingerprint, so any bank's split -- a
 * status, a base token, a row, an accepted token, a frontier -- is one
 * divergence, refused. */
/// @param s          the session
/// @param tp         the transport
/// @param frame_type the phase's frame type
/// @param operation  names the phase
/// @param steps      [n] the per-bank steps
/// @param n          steps, <= PULSAR_TP_SPEC_STEPS_MAX
/// @param h1         per-frame header field i1 (the phase's own meaning)
/// @param h2         per-frame header field i2
/// @param h3         per-frame header field i3
/// @return 1 when sent, 0 when refused or not shipped
static int tp_spec_steps_mirror(pulsar_session *s, pulsar_tp *tp, uint32_t frame_type,
                                const char *operation, const pulsar_spec_step *steps, int n,
                                int32_t h1, int32_t h2, int32_t h3) {
    if (n < 0 || (uint32_t)n > PULSAR_TP_SPEC_STEPS_MAX) {
        fprintf(stderr, "pulsar: tp: %s carries %d steps (at most %u)\n", operation, n,
                (unsigned)PULSAR_TP_SPEC_STEPS_MAX);
        return 0;
    }
    pulsar_tp_spec_command c = tp_spec_cmd(s, tp_spec_live_bank(s));
    c.count = (uint32_t)n;
    /* i0: unused since L284 (the stop set is the engine's, not the frame's) */
    c.i1 = h1; c.i2 = h2; c.i3 = h3;   /* per frame: assemble's row budget; round_end's spec cost (v24) */
    pulsar_tp_spec_command recs[PULSAR_TP_SPEC_STEPS_MAX];
    for (int i = 0; i < n; i++) {
        const pulsar_spec_step *st = &steps[i];
        pulsar_tp_spec_command *r = &recs[i];
        *r = tp_spec_cmd(s, (int)st->bank);
        r->temperature = st->temperature; r->top_k = st->top_k;
        r->top_p = st->top_p; r->min_p = st->min_p;
        r->rng = st->rng ? *st->rng : 0;
        r->i0 = frame_type == PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH ? st->max_tokens : st->first_token;
        r->i1 = st->k_alloc;
        r->i2 = st->accepted_cap;
        r->i3 = (int32_t)st->row0;
        r->greedy_rows = st->greedy_rows;
    }
    return tp_mirror_sent(tp, operation, pulsar_tp_send_spec_steps(tp, frame_type, &c, recs), NULL, 0);
}
/** Mark every step failed with "the pair refused <operation>".  @param steps [n] the steps  @param n steps
 *  @param operation names the phase  @return -1, always */
static int tp_spec_steps_fail(pulsar_spec_step *steps, int n, const char *operation) {
    for (int i = 0; i < n; i++) {
        steps[i].status = PULSAR_SPEC_STEP_FAILED;
        snprintf(steps[i].err, sizeof steps[i].err, "tp: the pair refused %s", operation);
    }
    return -1;
}
/* L303: token masks do not cross the pair (the frames carry no mask rows), so a constrained step on a pair fails by
 * name rather than letting the ranks draw from different rows. */
/** Does any step carry a token mask?  @param steps [n] the steps  @param n steps  @return true when one does */
static bool tp_spec_steps_masked(const pulsar_spec_step *steps, int n) {
    for (int i = 0; i < n; i++)
        if (steps[i].masks) return true;
    return false;
}
/** Fail every step: a constrained step does not cross the pair.  @param steps [n] the steps  @param n steps
 *  @return -1, always */
static int tp_spec_steps_refuse_masks(pulsar_spec_step *steps, int n) {
    for (int i = 0; i < n; i++) {
        steps[i].status = PULSAR_SPEC_STEP_FAILED;
        snprintf(steps[i].err, sizeof steps[i].err, "tp: constrained decoding does not cross the pair (L303)");
    }
    return -1;
}
int pulsar_session_spec_assemble_batch(pulsar_session *s, pulsar_spec_step *steps, int n,
                                       uint32_t row_budget, pulsar_multiseq_req *reqs, uint32_t *n_rows_out) {
    PULSAR_NVTX_FN();
    const char *operation = "spec_assemble_batch";
    *n_rows_out = 0;
    if (!s || n < 0 || (n > 0 && (!steps || !reqs))) return -1;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, operation, &tp, err, sizeof(err));
    if (route == 0) { fprintf(stderr, "pulsar: %s\n", err); return tp_spec_steps_fail(steps, n, operation); }
    if (route == 1) {
        pulsar_session_spec_assemble_batch_local(s, steps, n, row_budget, reqs, n_rows_out);
        return 0;
    }
    if (tp_spec_steps_masked(steps, n)) return tp_spec_steps_refuse_masks(steps, n);
    if (!tp_spec_steps_mirror(s, tp, PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH, operation, steps, n,
                              (int32_t)row_budget, 0, 0)) return tp_spec_steps_fail(steps, n, operation);
    pulsar_session_spec_assemble_batch_local(s, steps, n, row_budget, reqs, n_rows_out);
    const int own = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_ASSEMBLE, steps, n, *n_rows_out);
    if (tp_mirror_bank_verdict(s, tp, operation, own, -1) < 0) {
        *n_rows_out = 0;
        return tp_spec_steps_fail(steps, n, operation);
    }
    return 0;
}
int pulsar_session_spec_round_end_batch(pulsar_session *s, pulsar_spec_step *steps, int n) {
    PULSAR_NVTX_FN();
    const char *operation = "spec_round_end_batch";
    if (!s || n < 0 || (n > 0 && !steps)) return -1;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, operation, &tp, err, sizeof(err));
    if (route == 0) { fprintf(stderr, "pulsar: %s\n", err); return tp_spec_steps_fail(steps, n, operation); }
    if (route == 1) {
        pulsar_session_spec_round_end_batch_local(s, steps, n);
        return 0;
    }
    /* The rows are NOT shipped: every rank holds the same rows on its device from
     * its own mirrored forward, so only each step's row0 crosses.  The
     * leader's measured spec cost rides the header (v24): the guard every
     * rank's round_end prices the quench with is computed from these same
     * integers, so a quench latches on every rank or on none. */
    if (tp_spec_steps_masked(steps, n)) return tp_spec_steps_refuse_masks(steps, n);
    const pulsar_lane_cost_fit *cost = &s->engine->lane_cost[PULSAR_LANE_SPEC];
    if (!tp_spec_steps_mirror(s, tp, PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH, operation, steps, n,
                              cost->flat_us, cost->row_us, cost->valid ? 1 : 0))
        return tp_spec_steps_fail(steps, n, operation);
    pulsar_session_spec_round_end_batch_local(s, steps, n);
    const int own = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_ROUND_END, steps, n, 0u);
    return tp_mirror_bank_verdict(s, tp, operation, own, -1) < 0 ? tp_spec_steps_fail(steps, n, operation) : 0;
}
int pulsar_session_spec_redraft_commit_batch(pulsar_session *s, pulsar_spec_step *steps, int n) {
    PULSAR_NVTX_FN();
    const char *operation = "spec_redraft_commit_batch";
    if (!s || n < 0 || (n > 0 && !steps)) return -1;
    char err[256];
    pulsar_tp *tp = NULL;
    const int route = tp_spec_route(s, operation, &tp, err, sizeof(err));
    if (route == 0) { fprintf(stderr, "pulsar: %s\n", err); return tp_spec_steps_fail(steps, n, operation); }
    if (route == 1) {
        pulsar_session_spec_redraft_commit_batch_local(s, steps, n);
        return 0;
    }
    if (!tp_spec_steps_mirror(s, tp, PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT_BATCH, operation, steps, n,
                              0, 0, 0)) return tp_spec_steps_fail(steps, n, operation);
    pulsar_session_spec_redraft_commit_batch_local(s, steps, n);
    const int own = pulsar_spec_steps_verdict(PULSAR_SPEC_PHASE_REDRAFT_COMMIT, steps, n, 0u);
    return tp_mirror_bank_verdict(s, tp, operation, own, -1) < 0 ? tp_spec_steps_fail(steps, n, operation) : 0;
}
int pulsar_session_eval_speculative_block(pulsar_session *s, int first_token, int max_tokens, int *accepted, int accepted_cap, char *err, size_t errlen) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SPEC, "speculative decoding", -1); return s ? s->eval_speculative_block(first_token, max_tokens, accepted, accepted_cap, err, errlen) : 0; }
void pulsar_session_invalidate(pulsar_session *s) {
    PULSAR_NVTX_FN();
    /* The pair-off shape is untouched: one call, no TP code on the live path. */
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { pulsar_session_family_invalidate(s); return; }
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "invalidate", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "invalidate", err);
        return;
    }
    if (!tp_mirror_sent(tp, "invalidate",
                        pulsar_tp_send_invalidate(tp, s->tp_session_id), NULL, 0)) return;
    pulsar_session_family_invalidate(s);
}
void pulsar_engine_lane_cost_observe(pulsar_engine *e, pulsar_decode_lane lane, uint32_t rows, double ms) {
    if (!e || (unsigned)lane >= (unsigned)PULSAR_LANE_COUNT) return;
    pulsar_lane_cost_fit *f = &e->lane_cost[lane];
    lane_cost_fit_observe(f, rows, ms);
    if (!f->valid) return;
    /* Rule 5: the fit announces itself when it first arms and, at most once a
     * window, whenever a term has moved by half from what was last announced
     * -- the number the quench and the lane choice price with is in the log,
     * not inferred. */
    const bool first = f->ann_flat_us == 0;
    const bool moved = !first && f->n - f->ann_n >= 256u &&
                       (abs(f->flat_us - f->ann_flat_us) * 2 > f->ann_flat_us ||
                        abs(f->row_us - f->ann_row_us) * 2 > f->ann_row_us);
    if (first || moved) {
        f->ann_flat_us = f->flat_us;
        f->ann_row_us = f->row_us;
        f->ann_n = f->n;
        fprintf(stderr, "pulsar: %s lane cost measured: step = %.1f + %.2f x rows ms (%u steps%s)\n",
                lane == PULSAR_LANE_SPEC ? "spec" : "plain", (double)f->flat_us / 1000.0,
                (double)f->row_us / 1000.0, f->n, first ? "" : ", moved");
    }
}
pulsar_lane_cost pulsar_engine_lane_cost(const pulsar_engine *e, pulsar_decode_lane lane) {
    if (!e || (unsigned)lane >= (unsigned)PULSAR_LANE_COUNT) {
        pulsar_lane_cost c;
        memset(&c, 0, sizeof c);
        return c;
    }
    return lane_cost_fit_view(&e->lane_cost[lane]);
}
void pulsar_engine_spec_calibration(const pulsar_engine *e, pulsar_spec_calibration *out) {
    if (e) *out = e->spec_calibration;
    else memset(out, 0, sizeof *out);
}
const char *pulsar_engine_drafter_name(const pulsar_engine *e) {
    if (!e || !e->drafter_ops) return "";
    /* a block drafter is named by its container's kind (the lane's ops are shared: "block") */
    if (e->block_drafter && e->drafter_ops == &e->block_drafter->ops && e->block_drafter->kind)
        return e->block_drafter->kind;
    return e->drafter_ops->name ? e->drafter_ops->name : "";
}
void pulsar_engine_spec_cost_set(pulsar_engine *e, int32_t flat_us, int32_t row_us, bool valid) {
    if (!e) return;
    pulsar_lane_cost_fit *f = &e->lane_cost[PULSAR_LANE_SPEC];
    f->flat_us = flat_us;
    f->row_us = row_us;
    f->valid = valid && flat_us > 0 && row_us > 0;
}
TokPos pulsar_session_checkpoint_best(pulsar_session *s, TokPos limit) {
    if (!s || limit.is_none() || limit <= TokPos{0}) return TokPos{0};
    const pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    return TokPos{st && st->ops ? (int)pulsar_ckpt_best(st, pulsar_session_live_bank(s), (uint32_t)limit.raw()) : 0};
}
pulsar_bank_resume pulsar_session_bank_resume_at(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt) {
    const pulsar_bank_resume none = {TokPos{0}, PromptToks{0}};
    if (!s || !prompt || prompt->len <= 0) return none;
    const pulsar_tokens *t = pulsar_bank_history(s, bank);
    if (!t) return none;
    /* every family's sync runs on the token-seam stitch (L115/L284): the bank's tokens up to the deepest shared
     * byte boundary, the prompt after it */
    pulsar_prefix_match m;
    pulsar_tokens_prefix_match(s->engine, t->v, t->len, prompt->v, prompt->len, &m);
    /* an extension (or the same prompt) of a history the sync continues where it stands */
    if (m.live_cut == TokCount{t->len} && pulsar_session_bank_continues(s, bank, m.live_cut.raw()))
        return {TokPos::after(m.live_cut), m.prompt_cut};
    /* else the one resume rule over the stitched prompt (G <= the cut: below it the stitched prompt is the bank's
     * own tokens) */
    const uint32_t G = pulsar_session_resume_point(s, bank, m.live_cut.raw(),
                                                   pulsar_prefix_match_stitched_len(&m, PromptToks{prompt->len}).raw());
    const TokPos at{(int)G};
    /* L334: at the cut the state serves the match's prompt side; below it, what the bank's tokens to G hold of the
     * request (pulsar_tokens_prompt_held_with) -- G itself when no seam lies below it, fewer when a re-spelled stretch
     * does (G was returned as the prompt-side count unconverted before) */
    if (at < TokPos::after(m.live_cut))
        return {at, pulsar_tokens_prompt_held_with(pulsar_engine_token_text_fn, s->engine, t->v, at, prompt->v, prompt->len)};
    return {at, m.prompt_cut};
}
/* L313: where `src`'s published state serves `prompt` -- the one resume rule (pulsar_session_resume_point, the rule a
 * bank's own resume reads) on src: its deepest grid checkpoint within the match and its prefill-only history (one short
 * of the prompt), or a deeper record by the rule's own terms -- and the record (NULL for a grid point); `*len` is the
 * prompt's length in src's currency (the stitched prompt's).  0 when none.
 * L322: the match is the token-seam one (pulsar_tokens_prefix_match), as the route and the bank's own resume read it,
 * not the exact-id common prefix: a history loaded from a byte-only disk match (another spelling of the same bytes)
 * parts from every later prompt by id at that seam, and the id rule capped every adopt of it there (8192, 2026-10-24,
 * Claude Code at 178-193k).  The adopted view is src's tokens to T, and the sync re-stitches it against the prompt, so
 * T must be a boundary the two spellings share: a T inside a re-spelled stretch is no adopt point (the stitch would
 * stand short of T, below every checkpoint the adopt leaves, and prefill from 0). */
/// @param s      the session
/// @param src    the bank adopted from
/// @param prompt the request's tokens
/// @param rec    receives the record at T, or NULL for a grid checkpoint
/// @param len    receives the prompt's length in src's currency
/// @return T, the adopt point in tokens, or 0 when none
static uint32_t bank_adopt_point(pulsar_session *s, uint32_t src, const pulsar_tokens *prompt,
                                 const pulsar_ckpt_record **rec, int *len) {
    *rec = NULL;
    *len = prompt->len;
    const pulsar_tokens *h = pulsar_bank_history(s, src);
    if (!h || prompt->len <= 0) return 0u;
    pulsar_prefix_match m;
    pulsar_tokens_prefix_match(s->engine, h->v, h->len, prompt->v, prompt->len, &m);
    *len = pulsar_prefix_match_stitched_len(&m, PromptToks{prompt->len}).raw();
    const uint32_t T = pulsar_session_resume_point(s, src, m.live_cut.raw(), *len);
    if (T == 0u) return 0u;
    if (m.seamed) {
        pulsar_prefix_match at;
        pulsar_tokens_prefix_match(s->engine, h->v, (int)T, prompt->v, prompt->len, &at);
        if (at.live_cut != TokCount{(int)T}) return 0u;
    }
    *rec = pulsar_session_resume_record(s, src, T, *len);
    return T;
}

pulsar_bank_resume pulsar_session_bank_adopt_score(pulsar_session *s, uint32_t src, const pulsar_tokens *prompt) {
    const pulsar_bank_resume none = {TokPos{0}, PromptToks{0}};
    if (!s || !prompt || src >= (uint32_t)pulsar_session_bank_count(s) || src == pulsar_session_live_bank(s))
        return none;
    /* a pair does not mirror the adopt (pulsar_session_bank_adopt refuses it) */
    if (tp_mirror_target(s)) return none;
    const pulsar_ckpt_record *r = NULL;
    int len = 0;
    const TokPos T{(int)bank_adopt_point(s, src, prompt, &r, &len)};
    /* L334: T is on src's history; the depth an adopt is ranked by is what src's tokens to T hold of the request */
    return {T, pulsar_session_bank_prompt_held(s, src, prompt, T)};
}

PromptToks pulsar_session_bank_prompt_held(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, TokPos at) {
    if (!s || !prompt || at.is_none() || at <= TokPos{0}) return PromptToks{0};
    const pulsar_tokens *h = pulsar_bank_history(s, bank);
    if (!h || at > TokPos{h->len}) return PromptToks{0};
    return pulsar_tokens_prompt_held_with(pulsar_engine_token_text_fn, s->engine, h->v, at, prompt->v, prompt->len);
}

pulsar_bank_adopt_result pulsar_session_bank_adopt(pulsar_session *s, uint32_t src, const pulsar_tokens *prompt,
                                                   char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    const pulsar_bank_adopt_result refused = {PULSAR_ADOPT_REFUSED, TokPos::none()};
    const pulsar_bank_adopt_result none = {PULSAR_ADOPT_NONE, TokPos{0}};
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "bank adopt", refused);
    if (!s || !prompt) {
        if (err) snprintf(err, errlen, "bank adopt: no session or prompt");
        return refused;
    }
    if (tp_mirror_target(s)) {
        if (err) snprintf(err, errlen, "bank adopt: a tensor-parallel engine does not mirror it");
        return refused;
    }
    const uint32_t dst = pulsar_session_live_bank(s);
    if (src >= (uint32_t)pulsar_session_bank_count(s) || src == dst) {
        if (err) snprintf(err, errlen, "bank adopt: bank %u is not another bank of the pool", src);
        return refused;
    }
    const pulsar_ckpt_record *r = NULL;
    int len = 0;   /* the prompt in src's currency: a prompt end is the WHOLE (stitched) prompt */
    const uint32_t T = bank_adopt_point(s, src, prompt, &r, &len);
    if (T == 0u) return none;
    const pulsar_tokens *h = pulsar_bank_history(s, src);
    /* the records and checkpoints keep no image identity across banks: a history with an image block stays its own */
    if (pulsar_image_block_starts(s->engine, h, (int)T, NULL, 0) != 0) return none;
    pulsar_tokens view = {0};
    for (uint32_t i = 0; i < T; i++) pulsar_tokens_push(&view, h->v[i]);   /* h is src's carry: copy before moving */
    pulsar_ckpt_store *st = pulsar_session_kv_store(s);
    const bool at_end = r && r->logits && T == (uint32_t)len;
    if (at_end) memcpy(s->logits, r->logits, (size_t)st->end_width * sizeof(float));
    s->checkpoint_valid = false;   /* the state moves below */
    if (!pulsar_ckpt_adopt(st, dst, src, T, r) || (at_end && !s->live_from_host(s->logits))) {
        pulsar_tokens_free(&view);
        if (err) snprintf(err, errlen, "bank adopt: bank %u's state at %u did not copy into bank %u", src, T, dst);
        return refused;
    }
    s->checkpoint.len = 0;
    for (uint32_t i = 0; i < T; i++) token_vec_push(&s->checkpoint, view.v[i]);
    pulsar_tokens_free(&view);
    s->checkpoint_valid = true;
    s->prefill_frontier = (int)T;
    s->logits_stale = !at_end;
    s->live_images.n = 0;
    spec_lookahead_reset(s);   /* the bank's pending drafts were its old history's */
    fprintf(stderr, "pulsar: bank %u adopts bank %u's %s at %u (prompt %d)\n", dst, src,
            !r ? "grid checkpoint" : r->kind == PULSAR_CKPT_REC_END ? "prompt end" : "message endpoint", T, prompt->len);
    return {PULSAR_ADOPT_ADOPTED, TokPos{(int)T}};
}

int pulsar_session_restore_checkpoint(pulsar_session *s, TokPos G, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    if (!s || G.is_none() || G <= TokPos{0}) {
        if (err) snprintf(err, errlen, "restore checkpoint: no session or position %d", G.raw());
        return 1;
    }
    /* L272 B1: a family without a standalone restore restores its grid checkpoints inside sync (the resume,
     * L266 step 5); before this gate the entry read DeepSeek's zeroed store and refused with "holds no
     * checkpoint". */
    const pulsar_family_bank_ops *ops = session_banks(s);
    if (!ops->restore_checkpoint) {
        if (err) snprintf(err, errlen, "restore checkpoint: %s restores a grid checkpoint only as a sync resumes from it",
                 s->engine->family->name);
        return 1;
    }
    if (tp_mirror_target(s)) {
        if (err) snprintf(err, errlen, "restore checkpoint: a tensor-parallel engine does not mirror it");
        return 1;
    }
    if (!ops->restore_checkpoint(s, (uint32_t)G.raw())) {
        if (err) snprintf(err, errlen, "restore checkpoint: the installed bank holds no checkpoint at %d", G.raw());
        return 1;
    }
    return 0;
}
void pulsar_session_rewind(pulsar_session *s, TokPos pos) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_REWIND, "rewind", (void)0);
    if (!s) return;
    const pulsar_family_bank_ops *ops = session_banks(s);
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) { ops->rewind(s, pos.raw()); return; }   /* L334: the family's rewind stays raw (P2b) */
    char err[256];
    if (tp_mirror_worker_drives_nothing(tp, "rewind", err, sizeof(err))) {
        pulsar_tp_mirror_fail_void(tp, "rewind", err);
        return;
    }
    if (!tp_mirror_sent(tp, "rewind",
                        pulsar_tp_send_rewind(tp, s->tp_session_id, pos.raw()), NULL, 0)) return;
    ops->rewind(s, pos.raw());
}
TokPos pulsar_session_pos(pulsar_session *s) { return TokPos{s->pos()}; }
TokCount pulsar_session_ctx(pulsar_session *s) { return TokCount{s->ctx()}; }
/* Multi-session serving: is interrupting a sync at a chunk boundary (the cancel hook) and re-issuing it bit-identical
 * to letting it run?  Every family's prefill is the core loop over one cut rule (prefill_loop.cpp), a pure function
 * of the chunk's start, and a stop is honoured only where the resume continues exactly (L281 (b)), so the resumed
 * walk cuts where the cold one did -- provided a resumed walk's chunks are as wide as the cold pass's (cap <=
 * resumed_cap) and the cold pass's cap-grid ends are aligned ones (cap % align == 0).  0 = never interrupt;
 * otherwise any positive suffix resumes exactly (L131: no single-token tail path). */
uint32_t pulsar_session_prefill_quantum_min_suffix(const pulsar_session *s) {
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_BANKS, "the prefill quantum", 0);
    if (!s) return 0;
    pulsar_prefill_shape sh = {};
    s->engine->family->session->sync->prefill_shape(const_cast<pulsar_session *>(s), &sh);
    return sh.cap != 0u && sh.align != 0u && sh.cap <= sh.resumed_cap && sh.cap % sh.align == 0u ? 1u : 0u;
}
const pulsar_tokens *pulsar_session_tokens(pulsar_session *s) { return s ? s->tokens() : NULL; }
uint64_t pulsar_session_payload_bytes(pulsar_session *s) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_PAYLOAD, "session payloads", 0); return s ? s->payload_bytes() : 0; }
int pulsar_session_save_payload(pulsar_session *s, FILE *fp, char *err, size_t errlen) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_PAYLOAD, "session payloads", 1); return s ? s->save_payload(fp, err, errlen) : 1; }
int pulsar_session_load_payload(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_PAYLOAD, "session payloads", 1); return s ? s->load_payload(fp, payload_bytes, err, errlen) : 1; }
/* L334: a segment's span [G_prev, G): G_prev a position (none() is none), G past it. */
/// @param G_prev the span's start
/// @param G      the span's end
/// @return true when both are positions and G > G_prev >= 0
static bool segment_span_ok(TokPos G_prev, TokPos G) {
    return !G_prev.is_none() && !G.is_none() && G_prev >= TokPos{0} && G > G_prev;
}
uint64_t pulsar_session_segment_bytes(pulsar_session *s, TokPos G_prev, TokPos G, uint64_t *state_bytes) {
    if (state_bytes) *state_bytes = 0;
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SEGMENTS, "disk KV segments", 0);
    return s && segment_span_ok(G_prev, G) ? s->segment_bytes((uint32_t)G_prev.raw(), (uint32_t)G.raw(), state_bytes)
                                          : 0;
}
/* L264 S4e: on a TP group every rank keeps its own copy of each segment (KV is
 * replicated per rank), named by the leader's store key.  Both operations are
 * mirrored VERDICTS whose disagreement is a MISS on every rank, never a
 * divergence: the pair stays up and the conversation prefills.  Neither may run
 * inside a mirrored prefill -- the workers read only chunk verdicts there. */
/// @param s         the session
/// @param tp        the transport
/// @param operation names the operation
/// @param key       the leader's store key
/// @param err       receives the refusal
/// @param errlen    bytes of `err`
/// @return 1 = refused, 0 = go on
static int tp_segment_refused(pulsar_session *s, pulsar_tp *tp, const char *operation, const char *key,
                              char *err, size_t errlen) {
    if (tp_mirror_worker_drives_nothing(tp, operation, err, errlen) || tp_mirror_dead(tp, err, errlen)) return 1;
    if (s->tp_in_sync) {
        if (err) snprintf(err, errlen, "tp: %s inside a mirrored prefill is not mirrored (the workers are inside the "
                              "prefill)", operation);
        return 1;
    }
    if (!pulsar_tp_segment_key_ok(key)) {
        if (err) snprintf(err, errlen, "tp: %s: '%.40s' is not a segment key", operation, key ? key : "(null)");
        return 1;
    }
    return 0;
}
/** A segment frame's command for the session and key, every other field zero.  @param s the session
 *  @param key the store key (40 characters, copied)  @return the command */
static pulsar_tp_segment_command tp_segment_command(const pulsar_session *s, const char *key) {
    pulsar_tp_segment_command c;
    memset(&c, 0, sizeof(c));
    c.session_id = s->tp_session_id;
    memcpy(c.key, key, 40);
    return c;
}
void pulsar_engine_segment_dropped(pulsar_engine *e, const char *key) {
    if (!e || !e->tp || pulsar_tp_rank(e->tp) != 0 || pulsar_tp_failed(e->tp) || !pulsar_tp_segment_key_ok(key)) return;
    pulsar_tp_segment_command c;
    memset(&c, 0, sizeof(c));
    memcpy(c.key, key, 40);
    /* void: a lost drop leaves an orphan copy no leader key names -- never
     * loaded, reclaimed by the next bring-up's reconcile. */
    if (!pulsar_tp_send_segment(e->tp, PULSAR_TP_FRAME_SEGMENT_DROP, &c))
        pulsar_tp_mirror_fail_void(e->tp, "kv segment drop", "the frame could not be shipped");
}
int pulsar_engine_segment_reconcile(pulsar_engine *e, const char *keys, int n_keys) {
    if (!e || !e->tp || pulsar_tp_rank(e->tp) != 0 || pulsar_tp_failed(e->tp)) return 0;
    if (n_keys < 0 || (n_keys > 0 && !keys)) return 1;
    for (int i = 0; i < n_keys; i++)
        if (!pulsar_tp_segment_key_ok(keys + (size_t)i * 40u)) return 1;
    return pulsar_tp_send_segment_reconcile(e->tp, keys, (uint32_t)n_keys) ? 0 : 1;
}
bool pulsar_session_in_mirrored_sync(const pulsar_session *s) { return s && s->tp_in_sync; }
int pulsar_session_save_segment(pulsar_session *s, FILE *fp, TokPos G_prev, TokPos G, const char *key,
                                char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SEGMENTS, "disk KV segments", 1);
    if (!s || !segment_span_ok(G_prev, G)) {
        if (err) snprintf(err, errlen, "save segment: bad span [%d, %d)", G_prev.raw(), G.raw());
        return 1;
    }
    pulsar_tp *tp = tp_mirror_target(s);
    if (!tp) return s->save_segment(fp, (uint32_t)G_prev.raw(), (uint32_t)G.raw(), err, errlen);
    if (tp_segment_refused(s, tp, "kv segment save", key, err, errlen)) return 1;
    /* Ship first, write second: each rank writes its own copy of the SAME state
     * (nothing else moves on the control plane until the verdict is in), so the
     * writes overlap instead of running back to back. */
    pulsar_tp_segment_command c = tp_segment_command(s, key);
    c.G_prev = G_prev.raw();   /* the wire's int32 */
    c.G = G.raw();
    if (!tp_mirror_sent(tp, "kv segment save", pulsar_tp_send_segment(tp, PULSAR_TP_FRAME_SEGMENT_SAVE, &c),
                        err, errlen)) return 1;
    const int own = s->save_segment(fp, (uint32_t)G_prev.raw(), (uint32_t)G.raw(), err, errlen);
    int peers = 0;
    char perr[256];
    perr[0] = '\0';
    if (!pulsar_tp_wait_command_status(tp, s->tp_session_id, "kv segment save", &peers, perr, sizeof(perr))) {
        pulsar_tp_mirror_fail_void(tp, "kv segment save", perr);
        if (err) snprintf(err, errlen, "tp: kv segment save: the workers' verdict could not be collected: %s", perr);
        return 1;
    }
    if (own != 0) {
        if (peers == 0) pulsar_engine_segment_dropped(s->engine, key);   /* they stored, this rank did not */
        return 1;                                                         /* err: this rank's own reason */
    }
    if (peers != 0) {
        if (err) snprintf(err, errlen, "tp: a worker could not store its copy of segment %.40s (its log names the "
                              "reason); the segment is skipped on every rank", key);
        return 1;
    }
    return 0;
}
int pulsar_session_load_segment(pulsar_session *s, FILE *fp, uint64_t bytes, bool last, TokPos *G_out,
                                const char *key, char *err, size_t errlen) {
    PULSAR_NVTX_FN();
    PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_SEGMENTS, "disk KV segments", 1);
    if (G_out) *G_out = TokPos{0};
    if (!s) {
        if (err) snprintf(err, errlen, "load segment: no session");
        return 1;
    }
    pulsar_tp *tp = tp_mirror_target(s);
    if (tp && tp_segment_refused(s, tp, "kv segment load", key, err, errlen)) return 1;
    uint32_t G = 0;
    /* Load HERE first: the workers must reach the state this rank actually
     * ended in, so the frame carries the result, not the intent.  A local
     * failure never ships it -- the caller's mirrored invalidate brings every
     * rank to the same empty state. */
    const int own = s->load_segment(fp, bytes, last, &G, err, errlen);
    if (own != 0 || !tp) {
        if (G_out) *G_out = TokPos{(int)G};
        return own;
    }
    pulsar_tp_segment_command c = tp_segment_command(s, key);
    c.G = (int32_t)G;
    c.last = last ? 1 : 0;
    c.digest = pulsar_session_checkpoint_digest(s);
    if (!tp_mirror_sent(tp, "kv segment load", pulsar_tp_send_segment(tp, PULSAR_TP_FRAME_SEGMENT_LOAD, &c),
                        err, errlen)) return 1;
    int peers = 0;
    char perr[256];
    perr[0] = '\0';
    if (!pulsar_tp_wait_command_status(tp, s->tp_session_id, "kv segment load", &peers, perr, sizeof(perr))) {
        pulsar_tp_mirror_fail_void(tp, "kv segment load", perr);
        if (err) snprintf(err, errlen, "tp: kv segment load: the workers' verdict could not be collected: %s", perr);
        return 1;
    }
    if (peers != 0) {
        /* A MISS: the copy is missing or reached another state.  Nothing is
         * computed on it -- the caller invalidates every rank and drops the
         * chain. */
        if (err) snprintf(err, errlen, "tp: a worker has no matching copy of segment %.40s (its log names why) -- "
                              "a miss on every rank", key);
        return 1;
    }
    if (G_out) *G_out = TokPos{(int)G};
    return 0;
}
int pulsar_session_save_snapshot(pulsar_session *s, pulsar_session_snapshot *snap, char *err, size_t errlen) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_PAYLOAD, "session snapshots", 1); return s ? s->save_snapshot(snap, err, errlen) : 1; }
int pulsar_session_load_snapshot(pulsar_session *s, const pulsar_session_snapshot *snap, char *err, size_t errlen) { PULSAR_FAMILY_REQUIRES_S(s, PULSAR_FAMILY_CAP_PAYLOAD, "session snapshots", 1); return s ? s->load_snapshot(snap, err, errlen) : 1; }


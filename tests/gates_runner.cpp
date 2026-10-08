/* The model-dependent gates in ONE process (L163).
 *
 * `make gates` used to run every model gate as its own program, and every
 * program streamed the 86 GB model into the device cache: 28 opens at 18.6 s
 * each, a third of the battery, bounded by the NVMe the weights come from
 * (the page cache cannot hold the file beside an 86 GB device copy on a 121 GB
 * box, so each new process re-reads it).  Here the gates are functions
 * (tests/gate_entry.h: each source builds standalone as before, or into this
 * binary with -DPULSAR_GATE_RUNNER -DGATE_ENTRY=gate_<stem>_main) and the
 * engine they open is the broker's below -- one open per engine
 * CONFIGURATION, kept alive across every gate that asks for the same one.
 *
 * Order is part of the design, not a preference:
 *   - the instance lock allows one live engine per process, so configurations
 *     run as groups and a change of configuration closes the live engine;
 *   - the prefill/reference gate scrubs every PULSAR_* variable from the
 *     environment before it opens (numerics hygiene), so it and the other
 *     drafter-off configurations run LAST;
 *   - the GPU-only kernel tests, the unit tests (their own engine dance is part
 *     of what they assert) and the CLI chat smoke stay separate `make` targets.
 *
 * Every gate keeps its argv convention, its assertions and its prints; the
 * runner supplies exactly the arguments and the one-knob environment the
 * Makefile targets supplied, then unsets that knob.  Bank-pool size is set
 * through pulsar_engine_set_bank_pool before each gate (the engine parses
 * PULSAR_MSEQ_BANKS once per process, so the variable cannot do it here).
 *
 * A family host (L278).  MODEL is the primary model, the one the fixture blobs
 * on the command line belong to; each `--model PATH` adds a hosted model of any
 * family, run after it through the same table.  Every gate declares what it
 * needs of the model (gate_spec::needs); the runner reads each model's
 * capabilities through the public queries and a gate whose need is missing is
 * SKIPPED BY NAME ("SKIP <gate> [<family>]: needs <what>"), never run to fail
 * on a capability and never passed silently.
 *
 * usage: tests/gates_runner MODEL --prefill-baseline BLOB --prefill-ref SHORT
 *          --decode-baseline BLOB --decode-ref SHORT [ANCHORS] [--model PATH [ANCHORS]]...
 *          [--only=a,b | --only a,b] [--except=a,b] [--shape]
 *   ANCHORS = [--ref-dir DIR] [--ref-tol TOL] [--kl-story FILE] [--kl-code FILE]
 *             [--story-known-high D,..] [--story-known-flip D,..] [--code-known-high D,..] [--code-known-flip D,..]
 * The reference anchors are a MODEL's data: ANCHORS after `--model X` are X's, before any --model the primary's.
 * Each model's documented depths (a known-high KL row, a known argmax flip) come from the Makefile beside its
 * capture, never from this file.
 * Exit 0 only when every gate passed.  Prints a per-gate time table (all of
 * them, slowest first) and the number of engine opens.  --shape adds the work
 * shape each gate actually ran (prefill chunks, step calls/rows, deepest
 * position), read from the graph funnels; it changes nothing else.  --only /
 * --except select sub-gates by name for the iteration tier (`make gates-dev`):
 * an unknown name is refused, the selection is announced, and a selection that
 * leaves nothing to run FAILS.
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PULSAR_GATE_RUNNER 1
#define GATE_ENTRY gates_runner_has_its_own_main
#include "gate_entry.h"
#include "gate_util.h"

/* One entry per gate source, named by the Makefile from the file stem. */
int gate_multiseq_frontier_gate_main(int, char **);
int gate_rewind_frontier_gate_main(int, char **);
int gate_mseq_rewind_probe_main(int, char **);
int gate_token_seam_gate_main(int, char **);
int gate_multiseq_decode_gate_main(int, char **);
int gate_bank_spec_gate_main(int, char **);
int gate_dspark_batch_gate_main(int, char **);
int gate_accounting_gate_main(int, char **);
int gate_bank_evict_restore_gate_main(int, char **);
int gate_algo_stability_gate_main(int, char **);
int gate_mixed_prefill_gate_main(int, char **);
int gate_mixed_neutrality_gate_main(int, char **);
int gate_spec_sampling_gate_main(int, char **);
int gate_mseq_short_ctx_probe_main(int, char **);
int gate_comp_state_gate_main(int, char **);
int gate_chunk_neutrality_gate_main(int, char **);
int gate_prefill_bitexact_gate_main(int, char **);
int gate_session_payload_gate_main(int, char **);
int gate_tp_head_split_gate_main(int, char **);
int gate_decode_reference_gate_main(int, char **);
int gate_session_contract_gate_main(int, char **);
int gate_chat_decode_smoke_main(int, char **);

/* ---- the engine broker ------------------------------------------------- */

static pulsar_engine *g_live;
static pulsar_engine_options g_live_opt;
static char g_live_model[4096];
static int g_engine_opens;

static bool same_str(const char *a, const char *b) {
    if (!a || !b) return a == b;
    return strcmp(a, b) == 0;
}

/* prefill_chunk 0 IS PULSAR_PREFILL_CHUNK_DEFAULT (main() refuses to run with
 * PULSAR_CUDA_PREFILL_CHUNK set, so the engine cannot resolve 0 to anything
 * else), so a gate that leaves it 0 and one that pins the production grid ask
 * for the same engine -- comparing the raw field cost one model load per
 * battery (the drafter-off multiseq gate vs the prefill gates, L194). */
static uint32_t prefill_chunk_resolved(uint32_t v) { return v ? v : PULSAR_PREFILL_CHUNK_DEFAULT; }

/* The fields a gate may set.  Anything else non-zero is a configuration this
 * runner does not know how to share, so it is treated as different. */
static bool same_config(const pulsar_engine_options *a, const pulsar_engine_options *b) {
    return same_str(a->model_path, b->model_path) &&
           a->dspark_disable == b->dspark_disable &&
           same_str(a->expert_overlay, b->expert_overlay) &&
           a->backend == b->backend &&
           prefill_chunk_resolved(a->prefill_chunk) == prefill_chunk_resolved(b->prefill_chunk) &&
           a->spec_depth == b->spec_depth && a->spec_tau == b->spec_tau &&
           same_str(a->directional_steering_file, b->directional_steering_file) &&
           a->directional_steering_attn == b->directional_steering_attn &&
           a->directional_steering_ffn == b->directional_steering_ffn &&
           a->inspect_only == b->inspect_only &&
           a->tp_role == b->tp_role && a->tp_port == b->tp_port &&
           a->tp_rank == b->tp_rank && a->tp_nranks == b->tp_nranks &&
           same_str(a->tp_peer, b->tp_peer) && same_str(a->tp_peers, b->tp_peers);
}

int gate_engine_open(pulsar_engine **e, const pulsar_engine_options *want) {
    if (!e || !want || !want->model_path) return 1;
    /* The group, if any, joins the gate's options here (gate_entry.h): the
     * broker compares and opens the configuration the process will actually
     * run, so two gates asking for the same engine on the same group share it. */
    pulsar_engine_options o = *want;
    if (gate_tp_options_from_env(&o)) return 1;
    const pulsar_engine_options *opt = &o;
    if (g_live && same_config(&g_live_opt, opt)) {
        *e = g_live;
        return 0;
    }
    if (g_live) {
        fprintf(stderr, "gates_runner: engine configuration changes -- closing the live engine\n");
        pulsar_engine_close(g_live);
        g_live = NULL;
    }
    pulsar_engine *fresh = NULL;
    const int rc = pulsar_engine_open(&fresh, opt);
    if (rc != 0) return rc;
    gate_tp_worker_or_continue(fresh);   /* a worker rank never returns from here */
    g_engine_opens++;
    g_live = fresh;
    g_live_opt = *opt;
    snprintf(g_live_model, sizeof g_live_model, "%s", opt->model_path);
    g_live_opt.model_path = g_live_model;   /* the gate's argv pointer outlives nothing */
    *e = fresh;
    return 0;
}

void gate_engine_close(pulsar_engine *e) {
    /* The broker owns the engine; a gate closing it is a request, not an act.
     * A gate handing back an engine that is not the live one is a bug worth
     * knowing about. */
    if (e && e != g_live)
        fprintf(stderr, "gates_runner: a gate closed an engine the broker does not hold\n");
}

/* ---- the gate table ---------------------------------------------------- */

typedef int (*gate_entry_fn)(int, char **);

/* What a gate needs of the model it runs on (L278).  Each is read from the opened model through a public query,
 * never from its family's name -- except NEED_DS4_INTERNALS, which is what it says: the gate reads DeepSeek V4's
 * session graph through pulsar_engine_internal.h / gate_fixture.h, so it is that family's mechanism gate.  A gate
 * moves off it by being rewritten on the public API (as session-contract and row-neutrality are). */
enum {
    NEED_DS4_INTERNALS = 1u << 0,   /* DeepSeek V4's session graph */
    NEED_TOKENIZER     = 1u << 1,   /* chat text: pulsar_engine_has_tokenizer */
    NEED_SPEC          = 1u << 2,   /* speculative rounds: pulsar_engine_has_spec_rounds */
    NEED_DSPARK        = 1u << 3,   /* the DSpark drafter and its depth knob: pulsar_engine_drafter */
    NEED_REWIND        = 1u << 4,   /* pulsar_session_rewind: pulsar_engine_can_rewind */
    NEED_FIXTURES      = 1u << 5,   /* the blobs on the command line, which are the PRIMARY model's */
};
#define DS4 NEED_DS4_INTERNALS

static const char *need_name(uint32_t bit) {
    switch (bit) {
    case NEED_DS4_INTERNALS: return "DeepSeek V4's session graph (a DeepSeek mechanism gate)";
    case NEED_TOKENIZER:     return "a tokenizer and chat renderer";
    case NEED_SPEC:          return "speculative rounds";
    case NEED_DSPARK:        return "the DSpark drafter";
    case NEED_REWIND:        return "session rewind";
    case NEED_FIXTURES:      return "the command line's fixture blobs (captured from the primary model)";
    }
    return "?";
}

typedef struct {
    const char   *name;      /* the make target name it replaces */
    gate_entry_fn entry;
    uint32_t      banks;     /* bank pool the gate needs (1 = classic layout) */
    const char   *env_name;  /* the one knob the Makefile target exported, or NULL */
    const char   *env_val;
    const char   *args[12];  /* argv after the model path, NULL-terminated */
    uint32_t      needs;     /* NEED_* the model must have, or the gate is skipped by name */
} gate_spec;

/* A model's command-line data: its path and its reference anchors (see the header). */
typedef struct {
    const char *path, *ref_dir, *ref_tol, *kl_story, *kl_code;
    const char *story_high, *story_flip, *code_high, *code_flip;
} model_args;

/* Append `flag value` to a spec's argv when the value is given. */
static void spec_append(gate_spec *g, const char *flag, const char *value) {
    if (!value || !value[0]) return;
    int n = 0;
    while (g->args[n]) n++;
    if (n + 2 >= (int)(sizeof g->args / sizeof g->args[0])) {
        fprintf(stderr, "gates_runner: %s: no argv room for %s\n", g->name, flag);
        return;
    }
    g->args[n++] = flag;
    g->args[n++] = value;
    g->args[n] = NULL;
}

/* The model the table is running on: its path, its family's name, what it has. */
typedef struct {
    const char *path;
    char        family[64];
    uint32_t    has;         /* NEED_* bits this model satisfies */
    bool        primary;
} hosted_model;


typedef struct {
    char name[128];   /* the gate's name; a hosted (non-primary) model's results carry " [<family>]" */
    int rc;
    bool skipped;
    double secs;
    pulsar_gate_shape shape;
    uint64_t drafts;
} gate_result;

static bool g_shape_report;

static int run_gate(const gate_spec *g, const hosted_model *hm, gate_result *out) {
    memset(out, 0, sizeof *out);
    snprintf(out->name, sizeof out->name, hm->primary ? "%s" : "%s [%s]", g->name, hm->family);
    const uint32_t missing = g->needs & ~hm->has;
    if (missing) {
        const uint32_t bit = missing & (0u - missing);
        printf("\n  SKIP  %s [%s]: needs %s\n", g->name, hm->family, need_name(bit));
        fflush(stdout);
        out->skipped = true;
        return 0;
    }
    char *argv[20];
    int argc = 0;
    argv[argc++] = (char *)g->name;
    argv[argc++] = (char *)hm->path;
    for (int i = 0; g->args[i] && argc < 19; i++) argv[argc++] = (char *)g->args[i];
    argv[argc] = NULL;

    printf("\n\033[1m=== %s ===\033[0m\n", out->name);
    for (int i = 1; i < argc; i++) printf("%s%s", i > 1 ? " " : "  argv:", argv[i]);
    printf("%s%s%s\n", g->env_name ? "  " : "", g->env_name ? g->env_name : "",
           g->env_name ? "=set" : "");
    fflush(stdout);

    if (g->env_name) setenv(g->env_name, g->env_val, 1);
    pulsar_engine_set_bank_pool(g->banks);
    const int opens_before = g_engine_opens;
    pulsar_gate_shape_reset();   /* only this gate's work is counted, even on a reused engine */
    /* Spec rounds are a fact the engine already carries (the /metrics counters),
     * not a second copy: the spec lane's draft/verify forwards do not ride the
     * decode funnel, so a gate like bank-spec shows few step_calls and its real
     * repetition would be invisible without this. */
    pulsar_spec_metrics m0;
    const bool have_m0 = g_live != NULL;
    if (have_m0) pulsar_engine_spec_metrics(g_live, &m0);
    const double t0 = gate_now_s();
    const int rc = g->entry(argc, argv);
    const double secs = gate_now_s() - t0;
    pulsar_gate_shape shape1;
    pulsar_gate_shape_read(&shape1);
    uint64_t drafts = 0;
    if (g_live) {
        pulsar_spec_metrics m1;
        pulsar_engine_spec_metrics(g_live, &m1);
        /* A gate that opened its own engine starts those counters at 0. */
        drafts = (g_engine_opens > opens_before || !have_m0) ? m1.num_drafts
                                                             : m1.num_drafts - m0.num_drafts;
    }
    if (g->env_name) unsetenv(g->env_name);
    fflush(stdout); fflush(stderr);
    printf("--- %s: %s (rc=%d, %.1f s, engine %s)", out->name, rc == 0 ? "PASS" : "FAIL", rc, secs,
           g_engine_opens > opens_before ? "opened" : "reused");
    if (g_shape_report)
        printf("  [prefill %llu/%llutok, steps %llu/%llurows, draftrounds %llu, maxpos %llu]",
               (unsigned long long)shape1.prefill_calls,
               (unsigned long long)shape1.prefill_tokens,
               (unsigned long long)shape1.step_calls,
               (unsigned long long)shape1.step_rows,
               (unsigned long long)drafts,
               (unsigned long long)shape1.max_pos);
    printf("\n");
    fflush(stdout);
    out->rc = rc; out->secs = secs;
    out->shape = shape1;
    out->drafts = drafts;
    return rc;
}

/* Open MODEL on the default configuration (the first gate of the default group asks for the same one, so the
 * broker reuses it) and read what it has.  The fixture blobs are the primary's. */
static int host_model(hosted_model *hm, const char *path, bool primary) {
    memset(hm, 0, sizeof *hm);
    hm->path = path;
    hm->primary = primary;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = path;
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "gates_runner: %s did not open\n", path);
        return 1;
    }
    snprintf(hm->family, sizeof hm->family, "%s", pulsar_engine_family_name(e));
    if (pulsar_engine_family(e) == PULSAR_FAMILY_ID_DEEPSEEK4) hm->has |= NEED_DS4_INTERNALS;
    if (pulsar_engine_has_tokenizer(e)) hm->has |= NEED_TOKENIZER;
    if (pulsar_engine_has_spec_rounds(e)) hm->has |= NEED_SPEC;
    if (pulsar_engine_drafter(e) == PULSAR_DRAFTER_DSPARK) hm->has |= NEED_DSPARK;
    if (pulsar_engine_can_rewind(e)) hm->has |= NEED_REWIND;
    if (primary) hm->has |= NEED_FIXTURES;
    printf("\ngates_runner: hosting %s (%s%s):%s%s%s%s%s\n", path, hm->family, primary ? ", primary" : "",
           hm->has & NEED_TOKENIZER ? " tokenizer" : "", hm->has & NEED_SPEC ? " spec" : "",
           hm->has & NEED_DSPARK ? " dspark" : "", hm->has & NEED_REWIND ? " rewind" : "",
           hm->has & NEED_DS4_INTERNALS ? " ds4-internals" : "");
    fflush(stdout);
    return 0;
}

static int cmp_secs_desc(const void *a, const void *b) {
    const double da = ((const gate_result *)a)->secs, db = ((const gate_result *)b)->secs;
    return da < db ? 1 : da > db ? -1 : 0;
}

static bool only_wants(const char *only, const char *name) {
    if (!only) return true;
    const size_t n = strlen(name);
    for (const char *p = only; *p;) {
        const char *q = strchr(p, ',');
        const size_t len = q ? (size_t)(q - p) : strlen(p);
        if (len == n && strncmp(p, name, n) == 0) return true;
        if (!q) break;
        p = q + 1;
    }
    return false;
}

/* ---- selection (--only / --except) --------------------------------------
 * The iteration tier (make gates-dev) drives these.  Three rules, all of them
 * fail-closed:
 *   - an UNKNOWN name is refused (exit 2), never ignored: a typo must not look
 *     like a tier that ran and passed;
 *   - the remaining sub-gates are announced (selected and skipped) so the log
 *     says what the run covered;
 *   - a selection that leaves NO sub-gate to run FAILS the runner, because a
 *     tier that ran nothing is not a green tier.
 * --only also keeps its historical space form (`--only a,b`); `=` is accepted
 * for both. */
static const char *g_only, *g_except;

static bool name_in_list(const char *list, const char *name) {
    return list && only_wants(list, name);
}

static bool gate_selected(const char *name) {
    if (g_only && !name_in_list(g_only, name)) return false;
    if (g_except && name_in_list(g_except, name)) return false;
    return true;
}

/* Refuse any name in `list` that is not a sub-gate the runner knows. */
static int validate_names(const char *list, const char *what,
                          const char *const *known, int n_known) {
    if (!list) return 0;
    for (const char *p = list; *p;) {
        const char *q = strchr(p, ',');
        const size_t len = q ? (size_t)(q - p) : strlen(p);
        bool found = false;
        for (int i = 0; i < n_known && !found; i++)
            found = strlen(known[i]) == len && strncmp(known[i], p, len) == 0;
        if (!found) {
            fprintf(stderr, "gates_runner: %s names '%.*s', which is not a sub-gate; known names:\n", what,
                    (int)len, p);
            for (int i = 0; i < n_known; i++) fprintf(stderr, "    %s\n", known[i]);
            return 1;
        }
        if (!q) break;
        p = q + 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    /* The engine resolves prefill_chunk 0 through this variable; the runner's
     * engine-sharing rule (same_config) assumes the default grid.  Refuse
     * rather than share an engine opened on some other chunk. */
    if (getenv("PULSAR_CUDA_PREFILL_CHUNK")) {
        fprintf(stderr, "gates_runner: PULSAR_CUDA_PREFILL_CHUNK is set; the gate configurations assume the "
                        "default prefill grid -- unset it and rerun\n");
        return 2;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL --prefill-baseline BLOB --prefill-ref SHORT [--ref-dir DIR] "
                        "[--model PATH [ANCHORS]]... [--only=a,b] [--except=a,b] [--shape]  (see the header)\n", argv[0]);
        return 2;
    }
    const char *prefill_baseline = NULL, *prefill_ref = NULL;
    const char *decode_baseline = NULL, *decode_ref = NULL;
    enum { HOSTED_MAX = 4 };
    model_args models[1 + HOSTED_MAX], ignored;
    memset(models, 0, sizeof models);
    memset(&ignored, 0, sizeof ignored);
    int n_models = 1;
    models[0].path = argv[1];
    model_args *cur = &models[0];
    for (int i = 2; i < argc; ) {
        if (!strcmp(argv[i], "--shape")) { g_shape_report = true; i += 1; continue; }
        /* `--only=X` / `--except=X` take their value inline; the historical
         * `--only X` / `--except X` forms take the next argv. */
        if (!strncmp(argv[i], "--only=", 7)) { g_only = argv[i] + 7; i += 1; continue; }
        if (!strncmp(argv[i], "--except=", 9)) { g_except = argv[i] + 9; i += 1; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "gates_runner: option %s needs a value\n", argv[i]); return 2; }
        const char *v = argv[i + 1];
        if (!strcmp(argv[i], "--prefill-baseline")) prefill_baseline = v;
        else if (!strcmp(argv[i], "--prefill-ref")) prefill_ref = v;
        else if (!strcmp(argv[i], "--decode-baseline")) decode_baseline = v;
        else if (!strcmp(argv[i], "--decode-ref")) decode_ref = v;
        else if (!strcmp(argv[i], "--model")) {
            if (!v[0]) {
                cur = &ignored;   /* an empty value hosts nothing (a make default), and its anchors go with it */
            } else if (n_models == 1 + HOSTED_MAX) {
                fprintf(stderr, "gates_runner: at most %d --model\n", HOSTED_MAX);
                return 2;
            } else {
                cur = &models[n_models++];
                cur->path = v;
            }
        }
        else if (!strcmp(argv[i], "--ref-dir")) cur->ref_dir = v[0] ? v : NULL;
        else if (!strcmp(argv[i], "--ref-tol")) cur->ref_tol = v;
        else if (!strcmp(argv[i], "--kl-story")) cur->kl_story = v;
        else if (!strcmp(argv[i], "--kl-code")) cur->kl_code = v;
        else if (!strcmp(argv[i], "--story-known-high")) cur->story_high = v;
        else if (!strcmp(argv[i], "--story-known-flip")) cur->story_flip = v;
        else if (!strcmp(argv[i], "--code-known-high")) cur->code_high = v;
        else if (!strcmp(argv[i], "--code-known-flip")) cur->code_flip = v;
        else if (!strcmp(argv[i], "--only")) g_only = v;
        else if (!strcmp(argv[i], "--except")) g_except = v;
        else { fprintf(stderr, "gates_runner: unknown option %s\n", argv[i]); return 2; }
        i += 2;
    }
    if (!prefill_baseline || !prefill_ref || !decode_baseline || !decode_ref) {
        fprintf(stderr, "gates_runner: --prefill-baseline/--prefill-ref and --decode-baseline/--decode-ref "
                        "are required (the byte gates refuse to run without their blobs)\n");
        return 2;
    }

    /* Configuration A: the drafter-merged model, default options.  The spec
     * oracle runs 1250 trajectories per mode (L163: half the historical 2500;
     * alpha converges in a few hundred rounds, the chi-square keeps its
     * critical values and loses sensitivity accordingly). */
    const gate_spec group_default[] = {
        {"cuda-frontier-gate",        gate_multiseq_frontier_gate_main, 3, NULL, NULL, {NULL}, DS4},
        {"cuda-rewind-gate",          gate_rewind_frontier_gate_main,   1, NULL, NULL, {NULL}, DS4 | NEED_REWIND},
        {"cuda-mseq-rewind-gate",     gate_mseq_rewind_probe_main,      4, NULL, NULL, {NULL}, DS4 | NEED_REWIND},
        {"cuda-seam-gate",            gate_token_seam_gate_main,        3, NULL, NULL, {NULL}, NEED_TOKENIZER},
        {"cuda-multiseq-gate",        gate_multiseq_decode_gate_main,   3, NULL, NULL, {"3", "64", NULL}, NEED_TOKENIZER},
        {"cuda-bank-spec-gate",       gate_bank_spec_gate_main,         2, NULL, NULL, {"32", NULL}, DS4 | NEED_SPEC},
        {"cuda-dspark-batch-gate",    gate_dspark_batch_gate_main,      3, NULL, NULL, {"8", "0", NULL}, DS4 | NEED_DSPARK},
        {"cuda-accounting-gate",      gate_accounting_gate_main,        2, NULL, NULL, {NULL}, DS4},
        {"cuda-evict-restore-gate",   gate_bank_evict_restore_gate_main, 2, NULL, NULL, {NULL}, DS4},
        {"cuda-algo-stability-gate",  gate_algo_stability_gate_main,   16, NULL, NULL, {NULL}, NEED_TOKENIZER},
        /* L175: the same 1..16 width sweep with bank 0 at ~2200 tokens, so the
         * indexed lane is engaged while the row-count-keyed dispatches vary. */
        {"cuda-algo-stability-gate-deep", gate_algo_stability_gate_main, 16, NULL, NULL, {"deep", NULL}, NEED_TOKENIZER},
        {"cuda-mixed-prefill-gate",   gate_mixed_prefill_gate_main,     2, NULL, NULL, {NULL}, DS4},
        {"cuda-mixed-neutrality-gate", gate_mixed_neutrality_gate_main, 3, "PULSAR_GATE_ROWS_FATAL", "5,5 8,8", {NULL}, DS4},
        {"cuda-mixed-neutrality-gate-wide", gate_mixed_neutrality_gate_main, 13, "PULSAR_GATE_NDEC", "12", {NULL}, DS4},
        {"cuda-spec-sampling-gate",   gate_spec_sampling_gate_main,    16, NULL, NULL, {"0.95", "0", "1250", NULL}, NEED_SPEC | NEED_TOKENIZER},
        {"cuda-row-neutrality-gate",  gate_mseq_short_ctx_probe_main,   2, NULL, NULL, {NULL}, NEED_TOKENIZER},
        /* L170: the same probe past 2048 tokens, where the top-k selection is
         * engaged and the indexed fold order is what 1-row and N-row steps
         * must share (filler 1100 -> ~2217 tokens).  Mutation-validated: the
         * n_tokens > 1 sort condition fails it on all 129280 logits. */
        {"cuda-row-neutrality-gate-deep", gate_mseq_short_ctx_probe_main, 2, NULL, NULL, {"1100", NULL}, NEED_TOKENIZER},
        /* L175: the same at ~10k tokens (n_comp ~2600, the pow2<4096> ranking
         * bucket): the 1-row indexer scorer vs the N-row tier (L173) and the
         * ranking kernel past the 2048 boundary, on logits. */
        {"cuda-row-neutrality-gate-deeper", gate_mseq_short_ctx_probe_main, 2, NULL, NULL, {"4200", NULL}, NEED_TOKENIZER},
        /* L168: the ratio-4 compressor state after an unaligned whole-prompt
         * prefill has the decode store's layout (complete group at 0..3,
         * partial rows at 4 + phase) and its content, at r = 1, 2, 3. */
        {"cuda-comp-state-gate",      gate_comp_state_gate_main,        1, NULL, NULL, {NULL}, DS4},
        /* L175: the same assertions on the banked layout (per-layer caches are
         * bank views with a pool, owning allocations without one). */
        {"cuda-comp-state-gate-banked", gate_comp_state_gate_main,      2, NULL, NULL, {NULL}, DS4},
        /* L220: the SAVE -> LOAD round trip (comp-cache bytes + same-token
         * logits, then the v10 digest on a one-byte flip).  Same default
         * configuration as this group, so it reuses the broker's engine
         * instead of paying its own 92 GB load (the Makefile's last
         * not-yet-folded target).  1 bank: the classic payload layout. */
        {"cuda-session-payload-gate", gate_session_payload_gate_main,    1, NULL, NULL, {NULL}, DS4},
        /* L241 (slice 4g): a rank's attn_q_b / attn_output_a row slices are
         * byte-identical pieces of the whole projection on both GEMM arms. */
        /* L272 P5 / L278: the session contract every family meets (chunks C1-C5, banks B1-B6), public API only --
         * it runs on every hosted model, on the default configuration's engine. */
        {"session-contract-gate",     gate_session_contract_gate_main,  4, NULL, NULL, {NULL}, 0},
        /* L278 contract 6: one greedy chat answer, graded on content -- every hosted model's semantic smoke */
        {"chat-smoke-gate",           gate_chat_decode_smoke_main,      1, NULL, NULL, {NULL}, NEED_TOKENIZER},
        {"cuda-tp-head-split-gate",   gate_tp_head_split_gate_main,     1, NULL, NULL, {NULL}, DS4},
    };
    /* Configuration D: drafter depth 1 (the gate sets spec_depth). */
    const gate_spec group_depth1[] = {
        {"cuda-dspark-batch-gate-depth1", gate_dspark_batch_gate_main, 3, NULL, NULL, {"6", "1", NULL}, DS4 | NEED_DSPARK},
        /* L177: the widest admissible speculative step -- 3 banks x (1 + 4) = 15
         * rows, one under PULSAR_SPEC_ROW_BUDGET -- batched == serialized. */
        {"cuda-dspark-batch-gate-depth4", gate_dspark_batch_gate_main, 3, NULL, NULL, {"6", "4", NULL}, DS4 | NEED_DSPARK},
        /* L260: one redraft group of 10 banks (> 8) and 20 drafter rows (> 16), graded,
         * plus the byte-exact markov bank identity; depth 2 keeps the 30 verify rows inside the slab */
        {"cuda-dspark-batch-gate-wide", gate_dspark_batch_gate_main, 10, NULL, NULL, {"4", "2", "10", NULL}, DS4 | NEED_DSPARK},
    };
    /* Configuration B: drafter off -- the prefill gates below pin the chunk
     * to the default grid, so they share this engine (four opens per battery,
     * not five: A, D depth 1, D depth 4, B). */
    const gate_spec group_nodspark[] = {
        {"cuda-multiseq-gate-nodspark", gate_multiseq_decode_gate_main, 2, "PULSAR_GATE_NO_DSPARK", "1", {"2", "64", NULL}, NEED_TOKENIZER},
    };
    const gate_spec prefill = {"cuda-prefill-gate", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                               {"--check", prefill_baseline, prefill_ref, NULL}, NEED_FIXTURES};
    /* L181: one classic decode step after each unaligned prefill, byte-compared
     * -- the state class (compressor state, ring, seed) the prefill gate is
     * blind to.  Same binary, same env scrub, so it runs beside the prefill gate. */
    const gate_spec prefill_decode = {"cuda-prefill-decode-gate", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                                      {"--check-decode", decode_baseline, decode_ref, NULL}, NEED_FIXTURES};
    /* L183: the same tokens under four chunkings (cold, 6-row first chunk, two
     * mid-size chunks, off-grid warm resume) give byte-identical frontier and
     * next-step logits.  Same engine config as the prefill gate (chunk 4096,
     * drafter off), so it runs beside it. */
    const gate_spec chunk_neutrality = {"cuda-chunk-neutrality-gate", gate_chunk_neutrality_gate_main, 1, NULL, NULL,
                                        {NULL}, NEED_TOKENIZER};
    /* The reference gates are built per model, from its anchors (the hosted-model loop below); their names are
     * fixed here for the selection. */
    static const char *const ref_names[4] = {"cuda-reference-gate-story", "cuda-reference-gate-code",
                                             "cuda-decode-reference-gate-story", "cuda-decode-reference-gate-code"};

    /* The known sub-gate set, in run order.  The validator and the selection
     * report both read it, so --only/--except can never name a gate that does
     * not exist here.  A name that is expected but missing is a bug in the
     * caller's list, and it is refused rather than silently skipped. */
    const char *known[64];
    int n_known = 0;
#define KNOWN(spec) do { if (n_known < (int)(sizeof known / sizeof known[0])) known[n_known++] = (spec).name; } while (0)
    for (size_t i = 0; i < sizeof group_default / sizeof group_default[0]; i++) KNOWN(group_default[i]);
    for (size_t i = 0; i < sizeof group_depth1 / sizeof group_depth1[0]; i++) KNOWN(group_depth1[i]);
    for (size_t i = 0; i < sizeof group_nodspark / sizeof group_nodspark[0]; i++) KNOWN(group_nodspark[i]);
    KNOWN(prefill); KNOWN(prefill_decode); KNOWN(chunk_neutrality);
#undef KNOWN
    for (int i = 0; i < 4; i++) known[n_known++] = ref_names[i];
    if (validate_names(g_only, "--only", known, n_known) ||
        validate_names(g_except, "--except", known, n_known)) return 2;
    if (g_only || g_except) {
        printf("gates_runner: selection");
        if (g_only) printf(" --only=%s", g_only);
        if (g_except) printf(" --except=%s", g_except);
        printf("\n  selected (%d):", n_known);
        int n_sel = 0;
        for (int i = 0; i < n_known; i++) if (gate_selected(known[i])) { printf(" %s", known[i]); n_sel++; }
        printf("\n  skipped (%d):", n_known - n_sel);
        for (int i = 0; i < n_known; i++) if (!gate_selected(known[i])) printf(" %s", known[i]);
        printf("\n");
        bool any_ref = false;
        for (int i = 0; i < 4; i++) any_ref = any_ref || gate_selected(ref_names[i]);
        if (any_ref && !models[0].ref_dir)
            printf("  note: cuda-reference-gate-* / cuda-decode-reference-gate-* selected but the primary has no "
                   "--ref-dir (PULSAR_REF_DIR) -- they will SKIP\n");
        fflush(stdout);
    }

    enum { RESULTS_MAX = 192 };
    static gate_result results[RESULTS_MAX];
    int n_results = 0, rc_all = 0;
    const double suite0 = gate_now_s();
    hosted_model hm;
#define RUN(spec) do { if (gate_selected((spec).name) && n_results < RESULTS_MAX) { \
        if (run_gate(&(spec), &hm, &results[n_results++]) != 0) rc_all = 1; } } while (0)

    /* The primary model runs the whole table, its configurations in the order the header gives; each hosted model
     * after it runs the same table, where its needs allow -- one engine open per hosted model (its gates share the
     * default configuration's engine; a gate on another configuration of it would open its own). */
    for (int h = 0; h < n_models; h++) {
        const bool primary = h == 0;
        const model_args *ma = &models[h];
        if (host_model(&hm, ma->path, primary) != 0) { rc_all = 1; continue; }

        /* This model's reference gates, from its anchors.  Absent blobs SKIP LOUDLY: a gate that passes without its
         * fixture grades nothing.  The KL budgets grade direction and are passed only when present. */
        char story_ref[4096] = "", story_tok[4096] = "", code_ref[4096] = "", code_tok[4096] = "";
        bool have_ref = false;
        if (ma->ref_dir) {
            snprintf(story_ref, sizeof story_ref, "%s/story.ref.bin", ma->ref_dir);
            snprintf(story_tok, sizeof story_tok, "%s/story.tokens.bin", ma->ref_dir);
            snprintf(code_ref, sizeof code_ref, "%s/code.ref.bin", ma->ref_dir);
            snprintf(code_tok, sizeof code_tok, "%s/code.tokens.bin", ma->ref_dir);
            have_ref = access(story_ref, R_OK) == 0;
        }
        const char *tol = ma->ref_tol ? ma->ref_tol : "1e-4";
        gate_spec ref_story = {ref_names[0], gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                               {"--check-reference", story_ref, story_tok, tol, NULL}, 0};
        spec_append(&ref_story, "--known-high", ma->story_high);
        spec_append(&ref_story, "--known-flip", ma->story_flip);
        if (ma->kl_story && access(ma->kl_story, R_OK) == 0) spec_append(&ref_story, "--kl-baseline", ma->kl_story);
        gate_spec ref_code = {ref_names[1], gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                              {"--check-reference", code_ref, code_tok, tol, NULL}, 0};
        spec_append(&ref_code, "--known-high", ma->code_high);
        spec_append(&ref_code, "--known-flip", ma->code_flip);
        if (ma->kl_code && access(ma->kl_code, R_OK) == 0) spec_append(&ref_code, "--kl-baseline", ma->kl_code);
        /* L262: decode rows at every served width (1..32) against the same blobs -- the fidelity grade for
         * width-dependent decode arithmetic.  A blob's near-tie row is the prefill reference gate's known flip too. */
        gate_spec dref_story = {ref_names[2], gate_decode_reference_gate_main, 2, NULL, NULL,
                                {story_ref, story_tok, NULL}, DS4};
        spec_append(&dref_story, "--known-flip", ma->story_flip);
        gate_spec dref_code = {ref_names[3], gate_decode_reference_gate_main, 2, NULL, NULL,
                               {code_ref, code_tok, NULL}, DS4};
        spec_append(&dref_code, "--known-flip", ma->code_flip);

        for (size_t i = 0; i < sizeof group_default / sizeof group_default[0]; i++) RUN(group_default[i]);
        /* L278 (battery time): the decode-reference gates open the drafter-on, chunk-4096 configuration -- the default
         * group's -- so they run on its engine here rather than after the drafter-off prefill / reference gates, where
         * they cost a full model reload. */
        if (have_ref) {
            RUN(dref_story);
            RUN(dref_code);
        }
        for (size_t i = 0; i < sizeof group_depth1 / sizeof group_depth1[0]; i++) RUN(group_depth1[i]);
        for (size_t i = 0; i < sizeof group_nodspark / sizeof group_nodspark[0]; i++) RUN(group_nodspark[i]);
        RUN(prefill); RUN(prefill_decode); RUN(chunk_neutrality);
        bool ref_selected = false;
        for (int i = 0; i < 4; i++) ref_selected = ref_selected || gate_selected(ref_names[i]);
        if (have_ref) {
            RUN(ref_story);
            RUN(ref_code);
        } else if (ma->ref_dir && ref_selected) {
            /* The caller ASKED for the reference grade (--ref-dir was passed) and the
             * blob is not readable: that is a misconfiguration, not "not
             * configured", and it must not leave the battery green.  This is how
             * the landing script's PULSAR_REF_DIR pointed at a directory that did
             * not exist while every run still printed ALL GATES PASS -- the grade
             * was silently absent.  Fail. */
            printf("\n  FAIL  cuda-reference-gate [%s]: --ref-dir '%s' has no readable %s\n"
                   "        (blobs live outside the repo; stage them or pass an empty --ref-dir)\n",
                   hm.family, ma->ref_dir, story_ref);
            rc_all = 1;
        } else if (ref_selected) {
            printf("\n  SKIP  cuda-reference-gate, cuda-decode-reference-gate [%s]: no --ref-dir for this model\n"
                   "        (blobs live outside the repo; without them these gates grade nothing)\n", hm.family);
        }
    }
#undef RUN

    int n_ran = 0;
    for (int i = 0; i < n_results; i++) n_ran += !results[i].skipped;
    if (n_ran == 0) {
        printf("\n  FAIL  the selection ran no sub-gate -- a tier that runs nothing "
               "is not a pass\n");
        rc_all = 1;
    }
    if (g_live) { pulsar_engine_close(g_live); g_live = NULL; }

    printf("\n===================== RUNNER SUMMARY =====================\n");
    for (int i = 0; i < n_results; i++)
        if (results[i].rc == 0 && !results[i].skipped) printf("  PASS  %s\n", results[i].name);
    int n_skipped = 0;
    for (int i = 0; i < n_results; i++) n_skipped += results[i].skipped;
    if (n_skipped) {
        /* one line per hosted model: which gates its capabilities skipped (each was said by name above) */
        printf("  SKIP  %d gate(s) whose needs the model lacks:", n_skipped);
        for (int i = 0; i < n_results; i++)
            if (results[i].skipped) printf(" %s;", results[i].name);
        printf("\n");
    }
    for (int i = 0; i < n_results; i++)
        if (results[i].rc != 0) printf("  FAIL  %s (rc=%d)\n", results[i].name, results[i].rc);
    /* the timing tables list what ran */
    int n_timed = 0;
    for (int i = 0; i < n_results; i++)
        if (!results[i].skipped) results[n_timed++] = results[i];
    n_results = n_timed;
    qsort(results, (size_t)n_results, sizeof results[0], cmp_secs_desc);
    printf("\n  seconds per gate (slowest first):\n");
    for (int i = 0; i < n_results; i++) printf("    %6.0f  %s\n", results[i].secs, results[i].name);
    if (g_shape_report) {
        /* PHASE-0 work shape: what each gate actually ran, so the cut list is
         * argued from prefill chunks / step rows / depth instead of its name.
         * prefill = gpu_graph_prefill_layer_major calls and tokens (chunks and
         * L195 state-only warm-ups); steps = gpu_graph_decode_multiseq_batch
         * calls and rows (decode tokens, mixed K-row runs, verify batches);
         * maxpos = the deepest position the gate reached. */
        printf("\n  work shape per gate (slowest first):\n");
        printf("    %6s  %8s %9s  %8s %9s  %6s  %7s  %s\n",
               "secs", "prefill", "tok", "steps", "rows", "drafts", "maxpos", "gate");
        for (int i = 0; i < n_results; i++)
            printf("    %6.0f  %8llu %9llu  %8llu %9llu  %6llu  %7llu  %s\n",
                   results[i].secs,
                   (unsigned long long)results[i].shape.prefill_calls,
                   (unsigned long long)results[i].shape.prefill_tokens,
                   (unsigned long long)results[i].shape.step_calls,
                   (unsigned long long)results[i].shape.step_rows,
                   (unsigned long long)results[i].drafts,
                   (unsigned long long)results[i].shape.max_pos,
                   results[i].name);
    }
    printf("\n  %d gates in %.0f s, %d engine open(s)\n", n_results, gate_now_s() - suite0, g_engine_opens);
    printf(rc_all == 0 ? "RUNNER GATES: PASS\n" : "RUNNER GATES: FAIL\n");
    return rc_all;
}

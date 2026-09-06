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
 * usage: tests/gates_runner MODEL --prefill-baseline BLOB --prefill-ref SHORT
 *          [--ref-dir DIR] [--ref-tol TOL] [--kl-story FILE] [--kl-code FILE]
 *          [--only name,name,...]
 * Exit 0 only when every gate passed.  Prints a per-gate time table (all of
 * them, slowest first) and the number of engine opens.
 */
#include "pulsar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PULSAR_GATE_RUNNER 1
#define GATE_ENTRY gates_runner_has_its_own_main
#include "gate_entry.h"

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
int gate_bank_fork_gate_main(int, char **);
int gate_algo_stability_gate_main(int, char **);
int gate_mixed_prefill_gate_main(int, char **);
int gate_mixed_neutrality_gate_main(int, char **);
int gate_spec_sampling_gate_main(int, char **);
int gate_mseq_short_ctx_probe_main(int, char **);
int gate_comp_state_gate_main(int, char **);
int gate_chunk_neutrality_gate_main(int, char **);
int gate_prefill_bitexact_gate_main(int, char **);

/* ---- the engine broker ------------------------------------------------- */

static pulsar_engine *g_live;
static pulsar_engine_options g_live_opt;
static char g_live_model[4096];
static int g_engine_opens;

static bool same_str(const char *a, const char *b) {
    if (!a || !b) return a == b;
    return strcmp(a, b) == 0;
}

/* prefill_chunk 0 IS PULSAR_PREFILL_CHUNK_DEFAULT (the runner scrubs
 * PULSAR_CUDA_PREFILL_CHUNK), so a gate that leaves it 0 and one that pins the
 * production grid ask for the same engine -- comparing the raw field cost one
 * model load per battery (the drafter-off multiseq gate vs the prefill gates,
 * L194). */
static uint32_t prefill_chunk_resolved(uint32_t v) { return v ? v : PULSAR_PREFILL_CHUNK_DEFAULT; }

/* The fields a gate may set.  Anything else non-zero is a configuration this
 * runner does not know how to share, so it is treated as different. */
static bool same_config(const pulsar_engine_options *a, const pulsar_engine_options *b) {
    return same_str(a->model_path, b->model_path) &&
           a->dspark_disable == b->dspark_disable &&
           same_str(a->expert_overlay, b->expert_overlay) &&
           a->backend == b->backend &&
           prefill_chunk_resolved(a->prefill_chunk) == prefill_chunk_resolved(b->prefill_chunk) &&
           a->dspark_draft_tokens == b->dspark_draft_tokens &&
           same_str(a->directional_steering_file, b->directional_steering_file) &&
           a->directional_steering_attn == b->directional_steering_attn &&
           a->directional_steering_ffn == b->directional_steering_ffn &&
           a->inspect_only == b->inspect_only &&
           a->tp_role == b->tp_role && a->tp_port == b->tp_port &&
           same_str(a->tp_peer, b->tp_peer);
}

int gate_engine_open(pulsar_engine **e, const pulsar_engine_options *opt) {
    if (!e || !opt || !opt->model_path) return 1;
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

typedef struct {
    const char   *name;      /* the make target name it replaces */
    gate_entry_fn entry;
    uint32_t      banks;     /* bank pool the gate needs (1 = classic layout) */
    const char   *env_name;  /* the one knob the Makefile target exported, or NULL */
    const char   *env_val;
    const char   *args[12];  /* argv after the model path, NULL-terminated */
} gate_spec;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { const char *name; int rc; double secs; } gate_result;

static int run_gate(const gate_spec *g, const char *model, gate_result *out) {
    char *argv[20];
    int argc = 0;
    argv[argc++] = (char *)g->name;
    argv[argc++] = (char *)model;
    for (int i = 0; g->args[i] && argc < 19; i++) argv[argc++] = (char *)g->args[i];
    argv[argc] = NULL;

    printf("\n\033[1m=== %s ===\033[0m\n", g->name);
    for (int i = 1; i < argc; i++) printf("%s%s", i > 1 ? " " : "  argv:", argv[i]);
    printf("%s%s%s\n", g->env_name ? "  " : "", g->env_name ? g->env_name : "",
           g->env_name ? "=set" : "");
    fflush(stdout);

    if (g->env_name) setenv(g->env_name, g->env_val, 1);
    pulsar_engine_set_bank_pool(g->banks);
    const int opens_before = g_engine_opens;
    const double t0 = now_s();
    const int rc = g->entry(argc, argv);
    const double secs = now_s() - t0;
    if (g->env_name) unsetenv(g->env_name);
    fflush(stdout); fflush(stderr);
    printf("--- %s: %s (rc=%d, %.1f s, engine %s)\n", g->name, rc == 0 ? "PASS" : "FAIL", rc, secs,
           g_engine_opens > opens_before ? "opened" : "reused");
    fflush(stdout);
    out->name = g->name; out->rc = rc; out->secs = secs;
    return rc;
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL --prefill-baseline BLOB --prefill-ref SHORT [--ref-dir DIR] "
                        "[--ref-tol TOL] [--kl-story FILE] [--kl-code FILE] [--only a,b]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *prefill_baseline = NULL, *prefill_ref = NULL, *ref_dir = NULL, *ref_tol = "1e-4";
    const char *decode_baseline = NULL, *decode_ref = NULL;
    const char *kl_story = NULL, *kl_code = NULL, *only = NULL;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--prefill-baseline")) prefill_baseline = argv[i + 1];
        else if (!strcmp(argv[i], "--prefill-ref")) prefill_ref = argv[i + 1];
        else if (!strcmp(argv[i], "--decode-baseline")) decode_baseline = argv[i + 1];
        else if (!strcmp(argv[i], "--decode-ref")) decode_ref = argv[i + 1];
        else if (!strcmp(argv[i], "--ref-dir")) ref_dir = argv[i + 1][0] ? argv[i + 1] : NULL;
        else if (!strcmp(argv[i], "--ref-tol")) ref_tol = argv[i + 1];
        else if (!strcmp(argv[i], "--kl-story")) kl_story = argv[i + 1];
        else if (!strcmp(argv[i], "--kl-code")) kl_code = argv[i + 1];
        else if (!strcmp(argv[i], "--only")) only = argv[i + 1];
        else { fprintf(stderr, "gates_runner: unknown option %s\n", argv[i]); return 2; }
    }
    if (!prefill_baseline || !prefill_ref || !decode_baseline || !decode_ref) {
        fprintf(stderr, "gates_runner: --prefill-baseline/--prefill-ref and --decode-baseline/--decode-ref "
                        "are required (the byte gates refuse to run without their blobs)\n");
        return 2;
    }

    /* Reference-gate file names, built once.  Absent blobs SKIP LOUDLY, as the
     * Makefile target did: a gate that passes without its fixture grades
     * nothing. */
    char story_ref[4096], story_tok[4096], code_ref[4096], code_tok[4096];
    bool have_ref = false;
    if (ref_dir) {
        snprintf(story_ref, sizeof story_ref, "%s/story.ref.bin", ref_dir);
        snprintf(story_tok, sizeof story_tok, "%s/story.tokens.bin", ref_dir);
        snprintf(code_ref, sizeof code_ref, "%s/code.ref.bin", ref_dir);
        snprintf(code_tok, sizeof code_tok, "%s/code.tokens.bin", ref_dir);
        have_ref = access(story_ref, R_OK) == 0;
    }
    const bool kl_story_ok = kl_story && access(kl_story, R_OK) == 0;
    const bool kl_code_ok = kl_code && access(kl_code, R_OK) == 0;

    /* Configuration A: the drafter-merged model, default options.  The spec
     * oracle runs 1250 trajectories per mode (L163: half the historical 2500;
     * alpha converges in a few hundred rounds, the chi-square keeps its
     * critical values and loses sensitivity accordingly). */
    const gate_spec group_default[] = {
        {"cuda-frontier-gate",        gate_multiseq_frontier_gate_main, 3, NULL, NULL, {NULL}},
        {"cuda-rewind-gate",          gate_rewind_frontier_gate_main,   1, NULL, NULL, {NULL}},
        {"cuda-mseq-rewind-gate",     gate_mseq_rewind_probe_main,      1, NULL, NULL, {NULL}},
        {"cuda-seam-gate",            gate_token_seam_gate_main,        3, NULL, NULL, {NULL}},
        {"cuda-multiseq-gate",        gate_multiseq_decode_gate_main,   3, NULL, NULL, {"3", "512", NULL}},
        {"cuda-bank-spec-gate",       gate_bank_spec_gate_main,         2, NULL, NULL, {"128", NULL}},
        {"cuda-dspark-batch-gate",    gate_dspark_batch_gate_main,      3, NULL, NULL, {"8", "0", NULL}},
        {"cuda-accounting-gate",      gate_accounting_gate_main,        2, NULL, NULL, {NULL}},
        {"cuda-evict-restore-gate",   gate_bank_evict_restore_gate_main, 2, NULL, NULL, {NULL}},
        {"cuda-fork-gate",            gate_bank_fork_gate_main,         3, NULL, NULL, {NULL}},
        {"cuda-algo-stability-gate",  gate_algo_stability_gate_main,   16, NULL, NULL, {NULL}},
        /* L175: the same 1..16 width sweep with bank 0 at ~2200 tokens, so the
         * indexed lane is engaged while the row-count-keyed dispatches vary. */
        {"cuda-algo-stability-gate-deep", gate_algo_stability_gate_main, 16, NULL, NULL, {"deep", NULL}},
        {"cuda-mixed-prefill-gate",   gate_mixed_prefill_gate_main,     2, NULL, NULL, {NULL}},
        {"cuda-mixed-neutrality-gate", gate_mixed_neutrality_gate_main, 3, "PULSAR_GATE_ROWS_FATAL", "5,5 8,8", {NULL}},
        {"cuda-mixed-neutrality-gate-wide", gate_mixed_neutrality_gate_main, 13, "PULSAR_GATE_NDEC", "12", {NULL}},
        {"cuda-spec-sampling-gate",   gate_spec_sampling_gate_main,    16, NULL, NULL, {"0.95", "0", "1250", NULL}},
        {"cuda-row-neutrality-gate",  gate_mseq_short_ctx_probe_main,   2, NULL, NULL, {NULL}},
        /* L170: the same probe past 2048 tokens, where the top-k selection is
         * engaged and the indexed fold order is what 1-row and N-row steps
         * must share (filler 1100 -> ~2217 tokens).  Mutation-validated: the
         * n_tokens > 1 sort condition fails it on all 129280 logits. */
        {"cuda-row-neutrality-gate-deep", gate_mseq_short_ctx_probe_main, 2, NULL, NULL, {"1100", NULL}},
        /* L175: the same at ~10k tokens (n_comp ~2600, the pow2<4096> ranking
         * bucket): the 1-row indexer scorer vs the N-row tier (L173) and the
         * ranking kernel past the 2048 boundary, on logits. */
        {"cuda-row-neutrality-gate-deeper", gate_mseq_short_ctx_probe_main, 2, NULL, NULL, {"4200", NULL}},
        /* L168: the ratio-4 compressor state after an unaligned whole-prompt
         * prefill has the decode store's layout (complete group at 0..3,
         * partial rows at 4 + phase) and its content, at r = 1, 2, 3. */
        {"cuda-comp-state-gate",      gate_comp_state_gate_main,        1, NULL, NULL, {NULL}},
        /* L175: the same assertions on the banked layout (per-layer caches are
         * bank views with a pool, owning allocations without one). */
        {"cuda-comp-state-gate-banked", gate_comp_state_gate_main,      2, NULL, NULL, {NULL}},
    };
    /* Configuration D: drafter depth 1 (the gate sets dspark_draft_tokens). */
    const gate_spec group_depth1[] = {
        {"cuda-dspark-batch-gate-depth1", gate_dspark_batch_gate_main, 3, NULL, NULL, {"6", "1", NULL}},
        /* L177: the widest admissible speculative step -- 3 banks x (1 + 4) = 15
         * rows, one under PULSAR_SPEC_ROW_BUDGET -- batched == serialized. */
        {"cuda-dspark-batch-gate-depth4", gate_dspark_batch_gate_main, 3, NULL, NULL, {"6", "4", NULL}},
    };
    /* Configuration B: drafter off -- the prefill gates below pin the chunk
     * to the default grid, so they share this engine (four opens per battery,
     * not five: A, D depth 1, D depth 4, B). */
    const gate_spec group_nodspark[] = {
        {"cuda-multiseq-gate-nodspark", gate_multiseq_decode_gate_main, 2, "PULSAR_GATE_NO_DSPARK", "1", {"2", "64", NULL}},
    };
    const gate_spec prefill = {"cuda-prefill-gate", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                               {"--check", prefill_baseline, prefill_ref, NULL}};
    /* L181: one classic decode step after each unaligned prefill, byte-compared
     * -- the state class (compressor state, ring, seed) the prefill gate is
     * blind to.  Same binary, same env scrub, so it runs beside the prefill gate. */
    const gate_spec prefill_decode = {"cuda-prefill-decode-gate", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                                      {"--check-decode", decode_baseline, decode_ref, NULL}};
    /* L183: the same tokens under four chunkings (cold, 6-row first chunk, two
     * mid-size chunks, off-grid warm resume) give byte-identical frontier and
     * next-step logits.  Same engine config as the prefill gate (chunk 4096,
     * drafter off), so it runs beside it. */
    const gate_spec chunk_neutrality = {"cuda-chunk-neutrality-gate", gate_chunk_neutrality_gate_main, 1, NULL, NULL, {NULL}};
    const gate_spec ref_story = {"cuda-reference-gate-story", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                                 {"--check-reference", story_ref, story_tok, ref_tol, "--known-high", "512,30464",
                                  NULL}};
    const gate_spec ref_code = {"cuda-reference-gate-code", gate_prefill_bitexact_gate_main, 1, NULL, NULL,
                                {"--check-reference", code_ref, code_tok, ref_tol, "--known-high", "3840", NULL}};

    gate_result results[64];
    int n_results = 0, rc_all = 0;
    const double suite0 = now_s();
#define RUN(spec) do { if (only_wants(only, (spec).name)) { \
        if (run_gate(&(spec), model, &results[n_results++]) != 0) rc_all = 1; } } while (0)

    for (size_t i = 0; i < sizeof group_default / sizeof group_default[0]; i++) RUN(group_default[i]);
    for (size_t i = 0; i < sizeof group_depth1 / sizeof group_depth1[0]; i++) RUN(group_depth1[i]);
    for (size_t i = 0; i < sizeof group_nodspark / sizeof group_nodspark[0]; i++) RUN(group_nodspark[i]);
    RUN(prefill); RUN(prefill_decode); RUN(chunk_neutrality);
    if (have_ref) {
        /* --known-flip and --kl-baseline are appended per blob: the story blob
         * carries a documented argmax flip at 512; the KL budgets grade
         * direction and are passed only when present. */
        gate_spec s = ref_story;
        int n = 6;
        s.args[n++] = "--known-flip"; s.args[n++] = "512";
        if (kl_story_ok) { s.args[n++] = "--kl-baseline"; s.args[n++] = kl_story; }
        s.args[n] = NULL;
        RUN(s);
        gate_spec c = ref_code;
        n = 6;
        if (kl_code_ok) { c.args[n++] = "--kl-baseline"; c.args[n++] = kl_code; }
        c.args[n] = NULL;
        RUN(c);
    } else {
        printf("\n  SKIP  cuda-reference-gate: set PULSAR_REF_DIR to the reference-capture dir\n"
               "        (blobs live outside the repo; without them this gate grades nothing)\n");
    }
#undef RUN

    if (g_live) { pulsar_engine_close(g_live); g_live = NULL; }

    printf("\n===================== RUNNER SUMMARY =====================\n");
    for (int i = 0; i < n_results; i++)
        if (results[i].rc == 0) printf("  PASS  %s\n", results[i].name);
    for (int i = 0; i < n_results; i++)
        if (results[i].rc != 0) printf("  FAIL  %s (rc=%d)\n", results[i].name, results[i].rc);
    qsort(results, (size_t)n_results, sizeof results[0], cmp_secs_desc);
    printf("\n  seconds per gate (slowest first):\n");
    for (int i = 0; i < n_results; i++) printf("    %6.0f  %s\n", results[i].secs, results[i].name);
    printf("\n  %d gates in %.0f s, %d engine open(s)\n", n_results, now_s() - suite0, g_engine_opens);
    printf(rc_all == 0 ? "RUNNER GATES: PASS\n" : "RUNNER GATES: FAIL\n");
    return rc_all;
}

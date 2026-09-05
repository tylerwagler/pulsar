#include "pulsar_engine_internal.h"

/* Confidence-scheduled draft trim threshold.  Defaults to tau=0.25.  At the
 * v0.2.2 default draft depth 3 the 2026-07-17 tau sweep found tau barely moves
 * GREEDY throughput (only 3 positions to trim: full range within 1-3% and the
 * peak wanders inside noise), but tau=0.25 clearly wins under T=1.0 SAMPLING
 * (+25% structured, +10% prose vs verify-all), where the low-confidence tail is
 * real.  The old "optimal trim loosens with depth" was a k=5 artifact — the
 * real driver is acceptance rate, not depth, and it washes out at k=3.  The
 * trim is a SCHEDULE knob: it decides which drafts are verified, never a
 * verified row's numerics -- verify rows are DECODE rows and every decode row
 * takes the M-independent kernels whatever the batch width (row kind chooses
 * the arm, L167), so a row that survives the trim computes the same bytes at
 * any width (cuda-mixed-neutrality-gate GATE 5/5R assert it row by row, 1..16
 * rows).  The "narrowing the verify batch shifts float accumulation ~1 ULP"
 * this comment carried described the pre-L167 dispatch, where a 5..16-row
 * verify batch took cuBLASLt by row count.  PULSAR_DSPARK_CONF_SCHED=<tau> overrides; "0"/"off"
 * disables (verify all n_draft) -- tools/confhead sets it; it is a named
 * exception in docs/ENGINEERING-RULES.md.  Adaptive tau is not worth building
 * at k=3 (payoff ~2-6%, mostly captured by 0.25). */
static float dspark_conf_sched_tau(void) {
    static float cached = -1.0f;
    if (cached < 0.0f) {
        const char *cs = getenv("PULSAR_DSPARK_CONF_SCHED");
        if (!cs || !cs[0]) cached = 0.25f;
        else if (!strcmp(cs, "off") || !strcmp(cs, "false")) cached = 0.0f;
        else {
            float v = (float)atof(cs);
            cached = v > 0.0f ? v : 0.0f;
        }
    }
    return cached;
}

/* --- L107 adaptive draft depth -------------------------------------------
 * The 2026-08-25 depth sweep measured opposite optima per regime (prose 2,
 * structured 5; the shipped static 3 loses ~5%/~9.5% respectively), and the
 * conf-head calibration run measured the head monotone in both regimes with
 * conf>=0.9 -> 1.000 realized accept. Post-draft conf-sched trimming cannot
 * capture this (draft cost is paid before the trim; the sweep ran WITH the
 * trimmer on), so depth itself moves: +/-1 per round in spec_round_end.
 *   UP:   the whole drafted chain was verified AND accepted (commit == depth,
 *         which implies the trimmer kept everything) and the tail position's
 *         confidence clears SPEC_DEPTH_CONF_UP (calibrated >=0.82 accept) --
 *         the drafter was not the bottleneck this round, so probe deeper.
 *   DOWN: less than half the drafted depth converted (2*commit < depth) --
 *         drafting work is outrunning acceptance, back off.
 * Bounds [SPEC_DEPTH_MIN, SPEC_DEPTH_MAX] are the sweep's measured range;
 * depth 6 lost on BOTH regimes, so probing past it is priced as pure waste.
 * Distribution-preserving by construction (verification is exact at any
 * depth); NOT byte-identical on greedy prose -- verify-batch width shifts
 * accumulation ~1 ULP, the same known-flip class as conf-sched itself. */
#define SPEC_DEPTH_MIN PULSAR_SPEC_DEPTH_MIN
#define SPEC_DEPTH_MAX PULSAR_SPEC_DEPTH_MAX
#define SPEC_DEPTH_CONF_UP 0.70f
static uint32_t spec_cur_depth(const pulsar_session *s) {
    int d = s->spec.spec_adaptive_depth;
    if (d <= 0) d = s->engine->dspark_draft_tokens;
    if (d < 1) d = 1;
    if (d > 16) d = 16;
    return (uint32_t)d;
}

/* PULSAR_DSPARK_DUMP set: the offline dump wants the refined ids on the host
 * at draft time (the immediate harvest path) and the drafter's f32 rows
 * (main_x, target_h) written -- gpu_decode keeps those rows only when this
 * says so.  Read once per process. */
int gpu_graph_spec_dump_active(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *a = getenv("PULSAR_DSPARK_DUMP");
        cached = (a && a[0]) ? 1 : 0;
    }
    return cached;
}

/* --- Terminal yield-quench controller (spec-decode Item 4) ---------------
 * Controller design after the Entrpi ds4 yield quench (v0.1.1, MIT): per
 * request, every fused spec step accrues debt = breakeven yield minus
 * realized yield; when the request has provably lost more than a small
 * budget of plain tokens to speculation AND its recent yield is still below
 * breakeven, speculation turns off for the REMAINDER of that request
 * (terminal; a new request re-arms).
 *
 * Breakeven derivation (calibrated 2026-07-17 against per-step
 * PULSAR_DSPARK_STATS traces of the production v5mx serving path — 2585 steps
 * across prose/structured x greedy/T1.0, least-squares, resid rms 7.2 ms;
 * offline method after Entrpi's dspark_trace_replay, tool:
 * temp/quench/quench_replay.py):
 *   fused spec step   ~= FLAT + ROW * n_batch   milliseconds
 *     FLAT: pooled fit 101.6 ms, SHIPPED 95.7 ms — see the greedy-fit
 *           paragraph below for why the lower bound is the one compiled in
 *           (batched projections + shared + drafter dense/markov + sampled-q
 *           readback/dist walk — flat in verify rows; the old 53.6 ms nsys
 *           figure was the draft=3 build before temperature-matched drafting)
 *     ROW:  pooled fit 19.15 ms, SHIPPED 18.37 ms (marginal verify row —
 *           bandwidth-bound; matches the 2026-07-09 nsys 19.7 ms/row audit)
 *   plain decode token = PLAIN(pos), piecewise-linear through the measured
 *     served-plain depth table (2026-07-15, medians of 3): 59.7 ms @0.3k,
 *     67.3 @2.3k, 68.7 @9.3k, 74.5 @38k. Depth-dependence matters: spec step
 *     cost is ~flat in depth while plain slows, so a scalar PLAIN would
 *     overprice speculation exactly in the deep cells where it wins most
 *     (e.g. sweep-prose greedy @2.3k, +11.7%).
 * The breakeven yield of a step that verified K drafts at frontier pos is
 *   guard = (FLAT + ROW*(1+K)) / PLAIN(pos)     [plain tokens]
 * ~2.9 at full depth shallow, ~2.0 for a draft-only step. A request whose
 * committed tokens/step run below guard would have been faster plain.
 * Charging the ACTUAL n_batch (post conf-sched trim) rather than a scalar
 * guard prices exactly the steps the trimmer already shortened; committed
 * likewise is the post-trim realized yield. Measured operating points:
 * t2x prose y~2.05 vs guard ~2.8 (loses -> quench), structured y~5.1-5.3 vs
 * guard ~3.5 (wins by >1.5 -> never quenches).
 *
 * FLAT/ROW are the greedy-prose fit (the LOWER bound across the four
 * calibrated cells; the sampled cells fit ~6-11 ms higher FLAT). Deliberate:
 * underpricing the spec step biases against quenching, which keeps the
 * borderline deep greedy cells (sweep prose @2.3k: y~3.2 vs guard 3.06,
 * wins +11.7% measured) strictly on the no-quench side of the model.
 *
 * WARMUP: the first steps of every request are drafter pipeline fill —
 * n_batch ramps 1->2->3 with near-zero commits while the pendings build and
 * the prompt window seeds (a one-time ~40 ms not in the step model). That is
 * a fixed startup cost every request pays, not evidence about proposal
 * quality — charging it was measured (2026-07-17, first Load-2 gate run) to
 * book ~4.9 debt by step 6 at 2.3k ctx and spuriously quench the WINNING
 * sweep-prose cell (16.6 -> 14.8 t/s). The controller therefore ignores the
 * first PULSAR_QUENCH_WARMUP steps entirely, and MINEV=8 (Entrpi's replay
 * default) delays the verdict until the EWMA reflects steady state.
 * Re-validated offline over all 17 Load-1 traces + deep synthetic steady
 * states: losers (shallow y~2.05, deep y~2.5) fire at tokens ~11-25;
 * winners (struct, deep y>=3.2) never fire.
 *
 * Debt is deliberately NOT clamped below (matches Entrpi's shipped default):
 * unclamped, debt is exactly the request's NET plain-token-equivalents lost,
 * so the quench fires iff the request is genuinely >= BUDGET behind plain —
 * banked credit is real measured savings being spent, not optimism. Entrpi
 * measured the zero-clamped variant false-quenching long bursty winners, and
 * our offline replay selftest reproduces the same false quench for any
 * finite credit cap on a net-positive bursty request. Budget = 4 plain-token
 * equivalents (~250 ms): large enough that per-step yield variance on a
 * winning request cannot cross it (compounded with the EWMA and MINEV
 * conditions), small enough that a 400-token losing request recovers nearly
 * all of the loss.
 *
 * The controller reads only (commit, n_batch) — counts, never wall-clock —
 * so for a fixed token stream the quench point is deterministic. Constants
 * are compile-time (no hot-path env reads; project rule). */
/* REFIT 2026-07-21 (FLAT 95.7 -> 57.0; ROW unchanged). The 2026-07-17 fit went
 * stale: the fused step got ~30-40 ms cheaper (41 commits + the mxfp8head ->
 * type-40/MXFP8_LT model swap) while plain decode did not, so the shipped line
 * OVER-priced the step at EVERY width -- +57% at n_batch=1, +29% at 3, +14-15%
 * at 6..8. Guard = step/plain, so over-pricing raises the break-even, which
 * biases TOWARD quenching: the exact false-quench the design comment says it
 * deliberately biased against. Measured live at 2.7k/n_batch=6, yield 2.964 sat
 * below the shipped guard 3.057 (quench) but above the true 2.732 (spec was
 * ~8% faster than plain).
 *
 * Basis: 7591 steady-state steps over six cells (greedy/T1.0 x prose/structured
 * x 2.7k/9.4k/38k), pooled resid rms 5.88 ms (original fit: 7.2). Per-cell fits
 * are excellent (rms 0.08-2.4 ms), so ONE line is still the right model.
 *
 * Why 57.0 and why ROW stays 18.37:
 *  - Residual structure is by ENGINE DRAFT DEPTH, not temperature or context.
 *    FLAT rises ~3.3 ms per configured draft position (drafting work, which is
 *    NOT in the verify batch) while ROW falls to compensate: draft-2 fits
 *    FLAT 53.3/ROW 20.95, draft-7 fits FLAT 81.0/ROW 16.97. Since
 *    dspark_draft_tokens is fixed per engine and only conf-sched trim moves
 *    n_batch, the WITHIN-engine line is the one the controller actually rides.
 *    57.0 targets the shipped --dspark-draft 3 engine (~56.6); deeper drafts are
 *    then UNDER-priced, which biases against quenching = the safe direction.
 *  - Pooled ROW (20.3) is inflated by smearing cross-engine drafting cost into
 *    the slope. Keeping the lower shipped 18.37 lands the line 3-6% BELOW
 *    measured at every width 2..8, restoring the intended underprice margin.
 *  - Greedy remains the MINIMUM cell (T1.0 costs +3.5..12.4 ms at equal width),
 *    so a greedy-derived FLAT stays the conservative choice, as originally
 *    intended.
 * Depth enters as a small additive FLAT term only (~0.17 ms/1k, +6 ms over 35k)
 * and ROW is depth-invariant, so nothing depth-aware belongs in the step term.
 * NOTE the old comment's premise "spec step cost is ~flat in depth while plain
 * slows" is NOT what was measured: the step's depth slope (0.17 ms/1k) is close
 * to plain's (0.20). The guard ratio still improves with depth, but because the
 * step is 2-3x larger, not because it is flat. */
#define PULSAR_QUENCH_FLAT_MS    57.0f
#define PULSAR_QUENCH_ROW_MS     18.37f
#define PULSAR_QUENCH_ALPHA      0.125f   /* EWMA weight (Entrpi default) */
#define PULSAR_QUENCH_WARMUP     3u      /* ramp steps charged to no one (below) */
#define PULSAR_QUENCH_MINEV      8u      /* min spec steps before quench */
#define PULSAR_QUENCH_BUDGET     4.0f    /* plain-token equivalents */

/* Served plain-decode ms/token vs request depth: piecewise-linear through the
 * measured depth table (re-measured 2026-07-21 — see the block below for the
 * data and for why the 2026-07-15 values were retired). The bottom clamp
 * over-estimates plain and biases AGAINST quenching (conservative for the
 * no-spurious-quench gates). Plain ms/token keeps rising ~linearly with KV
 * depth, so beyond the last anchor we project that segment's slope rather than
 * flat-lining — a flat clamp under-estimates plain, which inflates the guard
 * and biases TOWARD quenching exactly where spec advantage is already marginal.
 * The projection is bounded at ~256k (PULSAR_QUENCH_PLAIN_CAP_POS), well past the
 * measured range, rather than extrapolating without limit. Deterministic
 * (constants only) so the quench point stays reproducible for a fixed stream. */
/* RE-MEASURED 2026-07-21 (medians of 3, run-to-run spread 0.01-0.11%), matching
 * the ORIGINAL instrument: the server's own `decoding ... avg=` line over a
 * 256-token greedy generation, `--no-dspark`, prose. Confirmed the shipped py
 * was exactly 1000/{16.74,14.85,14.55,13.43} from the 2026-07-15 prose table.
 *
 *   depth    old table   measured   ratio
 *     426      60.18       56.18    0.933
 *    2348      67.31       63.05    0.937
 *    9141      68.67       65.78    0.958
 *   38147      74.53       69.98    0.939
 *  100362      87.10*      80.24    0.921      (*extrapolated, now MEASURED)
 *
 * The table was uniformly ~6% HIGH — plain decode got ~6% faster since
 * 2026-07-15, alongside the fused spec step. The 100k anchor is new and real;
 * `-c` confound controlled (38k reads 70.12 at -c 131072 vs 70.00 at -c 40960,
 * +0.17%).
 *
 * A 2026-07-21 spot check that reported 85.89 ms at 38k (and an alarming
 * 0.675 ms/1k slope) was a MEASUREMENT ARTIFACT, reconstructed to 0.03 ms:
 * it used 128-vs-384 marginal differencing WITHOUT `--no-kv-disk`, so the two
 * legs took different disk-KV hits and prefilled 30011 vs 31163 tokens —
 * 70.20 true marginal + 12.55 unequal prefill + 3.17 client overhead = 85.92.
 * The real 9.1k->38k slope is 0.145 ms/1k, so the beyond-38k extrapolation was
 * if anything slightly too STEEP, never "3x too shallow". LESSON: when
 * differencing two runs to cancel prefill, DISABLE the disk KV cache or the
 * cancellation silently fails.
 *
 * DIRECTION (correcting the earlier note): the FLAT staleness and this table's
 * staleness did NOT compound — they partially CANCELLED. FLAT was over-priced
 * (guard too high) while plain was over-estimated (guard too low). Fixing FLAT
 * alone left the guard ~6-9% too LOW; this correction restores it (+7.0% @2.3k,
 * +6.4% @38k, +8.6% @100k, +11.9% @256k at n_batch=6).
 *
 * The measured curve is mildly convex (0.145 -> 0.165 ms/1k), so the linear
 * projection past 100k slightly UNDER-estimates plain, which inflates the guard
 * — the conservative direction this design already prefers, and now anchored on
 * a real 100k point instead of a 38k one. Past 100k is UNMEASURED: the 256k cap
 * (~105.9 ms) is a bounded guess. Also unmeasured: whether structured/tool
 * output shifts plain ms/token at depth (all five anchors are prose). */
#define PULSAR_QUENCH_PLAIN_CAP_POS 256000.0f
static float spec_quench_plain_ms(int pos) {
    static const float px[5] = { 300.0f, 2300.0f, 9300.0f, 38000.0f, 100000.0f };
    static const float py[5] = { 55.7f, 62.9f, 65.8f, 70.0f, 80.2f };
    const float p = (float)pos;
    if (p <= px[0]) return py[0];
    for (int i = 1; i < 5; i++)
        if (p <= px[i])
            return py[i - 1] + (py[i] - py[i - 1]) * (p - px[i - 1]) /
                                   (px[i] - px[i - 1]);
    /* pos > 100000: extend the last segment's slope, capped at the 256k value. */
    const float slope = (py[4] - py[3]) / (px[4] - px[3]);
    const float q = p < PULSAR_QUENCH_PLAIN_CAP_POS ? p : PULSAR_QUENCH_PLAIN_CAP_POS;
    return py[4] + slope * (q - px[4]);
}

static float spec_quench_guard(uint32_t n_batch, int pos) {
    return (PULSAR_QUENCH_FLAT_MS + PULSAR_QUENCH_ROW_MS * (float)n_batch) /
           spec_quench_plain_ms(pos);
}

/* Re-arm at request boundaries (the same sites that drop the carry and
 * pendings). All-zero == armed, matching the xcalloc'd session. */
void spec_quench_reset(pulsar_session *s) {
    s->spec.spec_quench_debt = 0.0f;
    s->spec.spec_quench_ewma = 0.0f;
    s->spec.spec_quench_steps = 0;
    s->spec.spec_quenched = false;
}

static void spec_frontier_free(pulsar_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}



/* Build the batched-copy descriptor tables for the frontier snapshot/restore
 * copy sets. All source/destination tensors are fixed allocations, so the
 * tables are built once and replayed with one kernel launch per direction
 * (~126 cudaMemcpy launches per snapshot, again per restore, before them).
 *
 * This is the ONE copy path (L190 D5).  prepare() rejects only impossible
 * states -- a NULL or undersized tensor, a byte count that is not a multiple
 * of 16 (every tensor here is a 256 B-aligned allocation) -- and device
 * allocation failure; on either the snapshot FAILS, loudly.  The per-tensor
 * cudaMemcpy loop it used to fall back to was a second implementation
 * selected exactly when something was wrong, announced once, and never
 * exercised by a gate; it is deleted.  Returns false with the flag left
 * clear, so the next snapshot retries a transient allocation failure.
 *
 * ⚠ THE FLAG IS SET ON SUCCESS, NOT ON ENTRY.  It used to be set here, before
 * either prepare() below was attempted, so a single transient failure latched
 * the degraded state forever with nothing logged and no way back.  prepare()
 * only builds descriptor tables, so retrying is cheap. */
static bool spec_frontier_copy_tables_init(pulsar_gpu_graph *g) {
    if (g->spec_frontier_copy_init) return true;
    pulsar_gpu_tensor *dst[PULSAR_MAX_LAYER * 4];
    pulsar_gpu_tensor *src[PULSAR_MAX_LAYER * 4];
    uint64_t bytes[PULSAR_MAX_LAYER * 4];
    uint32_t n = 0;
    uint64_t mx = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = pulsar_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        dst[n] = g->spec_attn_state_kv[il];    src[n] = g->layer_attn_state_kv[il];    bytes[n++] = ab;
        dst[n] = g->spec_attn_state_score[il]; src[n] = g->layer_attn_state_score[il]; bytes[n++] = ab;
        if (ab > mx) mx = ab;
        if (ratio == 4) {
            const uint64_t ib = pulsar_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            dst[n] = g->spec_index_state_kv[il];    src[n] = g->layer_index_state_kv[il];    bytes[n++] = ib;
            dst[n] = g->spec_index_state_score[il]; src[n] = g->layer_index_state_score[il]; bytes[n++] = ib;
            if (ib > mx) mx = ib;
        }
    }
    if (n == 0) {
        /* No compressed layers: there is genuinely nothing to copy (copy_n
         * stays 0 and the run is skipped). Done, not degraded. */
        g->spec_frontier_copy_init = 1;
        return true;
    }
    void *snap = pulsar_gpu_batched_copy_prepare(dst, src, bytes, n);
    /* restore = the same set with src/dst swapped */
    void *restore = pulsar_gpu_batched_copy_prepare(src, dst, bytes, n);
    if (!snap || !restore) {
        /* Take neither: the two directions must be the same copy set. */
        pulsar_gpu_batched_copy_free(snap);
        pulsar_gpu_batched_copy_free(restore);
        fprintf(stderr,
                "pulsar: spec frontier batched-copy prepare failed (%u descriptors, max %llu B) "
                "-- refusing the snapshot (no per-tensor copy loop; L190)\n",
                n, (unsigned long long)mx);
        return false;                /* init stays 0 -- the next call retries */
    }
    g->spec_snap_copies = snap;
    g->spec_restore_copies = restore;
    g->spec_frontier_copy_n = n;
    g->spec_frontier_copy_max_bytes = mx;
    g->spec_frontier_copy_init = 1;
    return true;
}

static bool spec_frontier_snapshot(pulsar_spec_frontier *f, pulsar_session *s) {
    memset(f, 0, sizeof(*f));
    pulsar_gpu_graph *g = &s->graph;
    if (!spec_frontier_copy_tables_init(g)) return false;

    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        f->n_comp[il] = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
        f->n_index_comp[il] = gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il);
    }
    if (ok && g->spec_frontier_copy_n)
        ok = pulsar_gpu_batched_copy_run(g->spec_snap_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}



static bool spec_frontier_restore(pulsar_spec_frontier *f, pulsar_session *s) {
    pulsar_gpu_graph *g = &s->graph;
    /* The tables cache the CURRENT bank's state pointers and
     * gpu_graph_bank_repoint drops them, so in the batched lane a round's
     * restore usually finds them gone: the server switches banks between
     * round_begin (snapshot) and round_end, and switches back before calling
     * here.  Rebuild for the bank that is live now -- the round's own -- the
     * same path the snapshot took.  Until L190 this case silently ran the
     * per-tensor loop, i.e. with two or more banks live the production restore
     * WAS the fallback. */
    if (!spec_frontier_copy_tables_init(g)) return false;
    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il) = f->n_comp[il];
        gpu_graph_n_index_comp(g, gpu_graph_cur_bank(g), il) = f->n_index_comp[il];
    }
    if (ok && g->spec_frontier_copy_n)
        ok = pulsar_gpu_batched_copy_run(g->spec_restore_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    return ok;
}


/* Diagnostic: dump the DSpark drafter's per-step inputs (target_h[3], main_x)
 * and pre-markov base logits (spec_logits row 0) so an off-box reference forward
 * can be diffed against ds4 to localize acceptance loss. Enabled by
 * PULSAR_DSPARK_DUMP=<path>; caps at PULSAR_DSPARK_DUMP_STEPS (default 8) records.
 * Record layout (little-endian): pos i32, first_token i32, then f32 arrays
 * target_h[0..2] (PULSAR_N_EMBD each), main_x (PULSAR_N_EMBD), base0 (PULSAR_N_VOCAB). */
static void dspark_dump_step(pulsar_gpu_graph *g, int pos, int first_token,
                             const int32_t *refined_ids, int n_draft) {
    /* Read once (no-hot-path-flags): this is called every spec step, and both
     * switches are fixed at start.  The common case is "not dumping", which
     * must cost a single cached load, not two getenv lookups. */
    static const char *path = NULL;
    static int max_steps = 8;
    static int env_read = 0;
    if (!env_read) {
        env_read = 1;
        const char *p = getenv("PULSAR_DSPARK_DUMP");
        path = (p && p[0]) ? p : NULL;
        const char *lim = getenv("PULSAR_DSPARK_DUMP_STEPS");
        if (lim && lim[0]) max_steps = atoi(lim);
    }
    if (!path) return;
    static int dumped = 0;
    if (dumped >= max_steps) return;

    const uint64_t hcw = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    float *emb = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(float));
    float *voc = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    float *hc = (float *)xmalloc((size_t)hcw * sizeof(float));
    FILE *f = fopen(path, dumped == 0 ? "wb" : "ab");
    if (!f) { free(emb); free(voc); free(hc); return; }
    /* Lean mode (PULSAR_DSPARK_DUMP_LEAN=1): confidence-head training records only —
     * hdr + refined_ids + the post-hc_head hidden rows (batch_ffn_cur) that the
     * engine's confidence kernel consumes (Step 5c), ~64 KB/step at draft=4
     * instead of ~1 MB. Bulk-collectable for the drafter-retune Phase 0. */
    static int lean = -1;
    if (lean < 0) lean = getenv("PULSAR_DSPARK_DUMP_LEAN") != NULL;
    if (lean) {
        int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
        for (int p = 0; p < n_draft; p++) {
            memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
            (void)pulsar_gpu_tensor_read(g->batch_ffn_cur, (uint64_t)p * PULSAR_N_EMBD * 4,
                                      emb, (uint64_t)PULSAR_N_EMBD * 4);
            fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
        }
        fclose(f);
        dumped++;
        if ((dumped & 1023) == 1)
            fprintf(stderr, "pulsar: dspark lean dump step %d pos=%d -> %s\n", dumped, pos, path);
        free(emb); free(voc); free(hc);
        return;
    }
    /* Record: pos, tok, n_draft, refined_ids[0..n_draft], target_h[3], main_x,
     * base0(vocab), then cur_hc for each of the n_draft block positions. n_draft
     * is fixed per run -> fixed record size. Used to validate offline whether the
     * DSpark confidence head predicts per-position acceptance on our requant. */
    int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
    for (int i = 0; i < 3; i++) {
        memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
        (void)pulsar_gpu_tensor_read(g->dspark_target_h[i], 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
        fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    }
    memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
    (void)pulsar_gpu_tensor_read(g->dspark_main_x, 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
    fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    memset(voc, 0, (size_t)PULSAR_N_VOCAB * sizeof(float));
    (void)gpu_graph_read_spec_logits_row(g, 0, voc);
    fwrite(voc, sizeof(float), PULSAR_N_VOCAB, f);
    /* block-2 output hidden (pre-hc_head) for each draft position [HC, EMBD]. */
    for (int p = 0; p < n_draft; p++) {
        memset(hc, 0, (size_t)hcw * sizeof(float));
        (void)pulsar_read_hc_carrier_f32(g->batch_cur_hc, (uint64_t)p * hcw, hc, hcw);
        fwrite(hc, sizeof(float), (size_t)hcw, f);
    }
    fclose(f);
    dumped++;
    fprintf(stderr, "pulsar: dspark dump step %d pos=%d tok=%d n_draft=%d -> %s\n",
            dumped, pos, first_token, n_draft, path);
    free(emb); free(voc); free(hc);
}

/* Fused-loop helper: make batch row `row`'s captured anchor hidden the current
 * drafter conditioning (target_h -> main_x) and seed one drafter-KV row from it.
 * Mirrors the reference invariant "drafter KV row j = f(hidden at position j)". */
static bool dspark_seed_from_batch_row(pulsar_session *s, uint32_t row) {
    pulsar_gpu_graph *g = &s->graph;
    pulsar_engine *e = s->engine;
    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h_batch[i] || !g->dspark_target_h[i]) return false;
        if (!pulsar_gpu_tensor_copy(g->dspark_target_h[i], 0,
                                 g->dspark_target_h_batch[i],
                                 (uint64_t)row * PULSAR_N_EMBD * sizeof(float),
                                 (uint64_t)PULSAR_N_EMBD * sizeof(float))) return false;
    }
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights)) return false;
    /* A failed seed has rolled the drafter rings back; the caller refuses the
     * round (L167) instead of drafting from an unseeded window. */
    return gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);
}

/* Fused DSpark loop (P2, PULSAR_DSPARK_FUSED=1): ONE batched target forward per
 * step instead of Step-1 decode + separate verify. The forward runs over
 * [first_token, pending_drafts...] (drafts made LAST step -- EAGLE pipeline
 * inversion), so position 0 is the base decode, positions 1..K verify the
 * drafts, and the batched anchor-hidden capture gives the drafter its
 * conditioning at whatever position ends up last-accepted. Drafting for the
 * NEXT step then conditions on the hidden that PRODUCED the next base token
 * (matching the reference generate.py forward_spec dataflow; the deleted
 * per-round loop conditioned on the hidden AFTER re-evaluating the base token
 * -- a one-position train/inference mismatch this path removed).
 * Greedy-only (generate.cpp gates on temperature<=0).
 * Partial/zero accepts restore the frontier and replay the committed prefix
 * (Stage A; the Stage-B transactional state removes the replay). */

/* plan-34 inc 6 (step 2a): the accept walk, extracted PURE from the fused
 * loop. Consumes logit rows through `read_row` (row index is round-local:
 * 0..K-1 are the draft rows), decides the accepted prefix and -- for the
 * sampled rule -- draws the rejecting position's residual carry. Touches NO
 * session state beyond the sampling scratch; this is the function the
 * batched lane calls once per bank over a SHARED forward's rows, with
 * read_row pointing into that batch's ALL_ROWS logits at the bank's offset.
 * The rules are verbatim from the fused loop (greedy argmax match; sampled
 * p/q with the draft-time-params q rebuild -- see the walk comment there). */
typedef bool (*spec_row_read_fn)(void *ud, uint32_t row, float *out);

/* L149 phase 2: build row `row`'s target distribution from the compact
 * prefilter block when the request is in the sparse min-p contract and the
 * row's candidate set fit the cap. false = caller reads the full row. */
static bool spec_compact_dist(const int32_t *compact, uint32_t row,
                              float temperature, int top_k, float top_p, float min_p,
                              pulsar_sample_scratch *scratch, pulsar_sample_dist *out) {
    if (!compact || row >= PULSAR_SPEC_LOGITS_ROWS ||
        !(temperature > 0.0f && top_k <= 0 && top_p == 1.0f &&
          min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && min_p <= 1.0f))
        return false;
    const int32_t *h = compact + (size_t)row * PULSAR_DSPARK_PREFILTER_ROW_I32;
    const uint32_t n = (uint32_t)h[0];
    if (n == 0 || n > PULSAR_DSPARK_PREFILTER_CAP) return false;
    float max_logit;
    memcpy(&max_logit, &h[2], sizeof(max_logit));
    return pulsar_sample_dist_build_prefiltered(h + 3,
                                                (const float *)(h + 3 + PULSAR_DSPARK_PREFILTER_CAP),
                                                n, max_logit, temperature, min_p,
                                                scratch, out) != 0;
}

static int spec_accept_walk(pulsar_session *s,
                            spec_row_read_fn read_row, void *read_ud,
                            const int32_t *compact, uint32_t row0,
                            const int *row_tops,
                            const int32_t *pend, uint32_t K, bool pend_sampled,
                            float temperature, int top_k, float top_p, float min_p,
                            uint64_t *rng, int *out_carry_tok) {
    int commit = 0;
    int carry_tok = -1;
    if (temperature <= 0.0f || K == 0) {
        while (commit < (int)K && row_tops[commit] == (int)pend[commit]) commit++;
    } else {
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
        float *row_logits = s->spec_row_scratch;
        while (commit < (int)K) {
            pulsar_sample_dist dist;
            if (!spec_compact_dist(compact, row0 + (uint32_t)commit, temperature, top_k,
                                   top_p, min_p, &s->sample_scratch, &dist)) {
                if (!read_row(read_ud, (uint32_t)commit, row_logits) ||
                    !pulsar_sample_dist_build(row_logits, PULSAR_N_VOCAB, temperature, top_k,
                                              top_p, min_p, &s->sample_scratch, &dist)) {
                    *out_carry_tok = -1;
                    return -1;
                }
            }
            const bool accepted_row = pend_sampled
                ? pulsar_sample_dist_accept_pq(&dist, (int)pend[commit],
                                            s->spec.dspark_pending_q[commit], rng)
                : pulsar_sample_dist_accept(&dist, (int)pend[commit], rng);
            if (accepted_row) {
                pulsar_sample_dist_free(&dist);
                commit++;
                continue;
            }
            if (pend_sampled) {
                pulsar_sample_dist qd;
                const uint32_t qn = s->spec.dspark_pending_qn[commit];
                if (qn > 0 && qn <= PULSAR_DSPARK_QDIST_CAP) {
                    /* L149: q_X exactly as built at draft time (drafting loop) */
                    memset(&qd, 0, sizeof(qd));
                    qd.n = qn;
                    qd.ids = (int *)xmalloc((size_t)qn * sizeof(int));
                    qd.probs = (float *)xmalloc((size_t)qn * sizeof(float));
                    memcpy(qd.ids, s->spec.dspark_pending_qids[commit], (size_t)qn * sizeof(int));
                    memcpy(qd.probs, s->spec.dspark_pending_qprobs[commit],
                           (size_t)qn * sizeof(float));
                } else if (!pulsar_sample_dist_build(s->dspark_pending_qrows +
                                                         (size_t)commit * PULSAR_N_VOCAB,
                                                     PULSAR_N_VOCAB, s->spec.dspark_pending_temp,
                                                     s->spec.dspark_pending_top_k,
                                                     s->spec.dspark_pending_top_p,
                                                     s->spec.dspark_pending_min_p,
                                                     &s->sample_scratch, &qd)) {
                    /* the stored q row refused: the proposal this draft was
                     * drawn from cannot be rebuilt, so the walk fails */
                    pulsar_sample_dist_free(&dist);
                    *out_carry_tok = -1;
                    return -1;
                }
                carry_tok = pulsar_sample_dist_draw_residual(&dist, &qd,
                                                          &s->sample_scratch, rng);
                pulsar_sample_dist_free(&qd);
            } else {
                carry_tok = pulsar_sample_dist_draw_excluding(&dist, (int)pend[commit], rng);
            }
            pulsar_sample_dist_free(&dist);
            break;
        }
    }
    *out_carry_tok = carry_tok;
    return commit;
}

/* Classic row source: the live graph's spec_logits rows. */
static bool spec_row_read_classic(void *ud, uint32_t row, float *out) {
    return gpu_graph_read_spec_logits_row((pulsar_gpu_graph *)ud, row, out);
}

/* inc-6 W5: the redraft block, extracted from the fused loop verbatim
 * (the no-draft guard stays with the caller -- it owns hit_eos/eos_token).
 * Best-effort: any failure returns 0 pendings and the step is still a
 * success (the original early-return contract). Recomputes the drafter
 * locals internally so the batched lane can call it per bank under repoint
 * with that bank's carry as next_base; n_batch/commit feed only the
 * diagnostic dumps. Returns `keep`, the confidence-trimmed pending count it
 * stashed (the caller's step_ms diagnostic reads it). W2 threads the bank's
 * batch-row offset into the spec_logits views and seed/draft-forward calls
 * in here. */
static uint32_t spec_round_redraft(pulsar_session *s, int next_base,
                                   bool main_x_ready,
                                   float temperature, int top_k, float top_p,
                                   float min_p, uint64_t *rng) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    const void *dmap = e->dspark_model.map;
    const uint64_t dsize = e->dspark_model.size;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    uint32_t n_draft = spec_cur_depth(s);   /* L107: session depth, not the static engine width */
    if (n_draft > 16u) n_draft = 16u;
    if (n_draft == 0u) return 0;
    /* Draft forward + markov refine (the reference forward_spec's Steps 3-5).
     * NOTE: no seed here -- the committed positions' rows were seeded above
     * (row j = f(h_j)); next_base's own row is seeded NEXT step when it is
     * processed as batch position 0.
     *
     * main_x is NOT re-projected here.  It used to be, "for clarity", after the
     * seeding loop had already left it at exactly this value — 3 sync copies, a
     * gemv and a norm per step for a value we already had.  The invariant is
     * checked at main_x_ready above, and seed_draft_kv only reads main_x, so
     * nothing between there and here can disturb it. */
    if (!main_x_ready)
        return 0;   /* drafting is best-effort; the step already succeeded */
    int32_t draft_ids[16];
    draft_ids[0] = (int32_t)next_base;
    for (uint32_t i = 1; i < n_draft; i++) draft_ids[i] = PULSAR_DSPARK_NOISE_TOKEN_ID;
    if (!gpu_graph_dspark_draft_forward(g, &e->model, &e->weights,
                                        &e->dspark_model, &e->dspark_weights,
                                        g->spec_logits, draft_ids, n_draft))
        return 0;

    pulsar_gpu_tensor *dspark_logits = g->dspark_markov_logits;   /* persistent scratch */
    if (!dspark_logits || pulsar_gpu_tensor_bytes(dspark_logits) < vocab_bytes) return 0;
    int32_t refined[17];
    refined[0] = (int32_t)next_base;
    /* Temperature-matched draft sampling: draw each draft from the refined
     * logits filtered at the REQUEST's params (q) instead of taking the
     * drafter's argmax, so the verify walk can use min(1, p/q) — whose
     * acceptance is not capped at p(mode).
     *
     * temperature <= 0 keeps the argmax path untouched: no readback, no
     * dist_build, and — critically — no rng draw, so the greedy token stream
     * stays byte-identical (dist_build would collapse to a point mass, but
     * pulsar_sample_dist_draw would still consume an rng word and shift the
     * stream). It is also the fast path we do not want to slow down.
     *
     * The proposal rule follows the temperature alone.  p is built over
     * PULSAR_N_VOCAB target logits and q over the drafter's vocab, and those
     * are the same space by construction: dspark_weights_validate_layout dies
     * at load on a support model whose vocab differs.  A `vocab_size ==
     * PULSAR_N_VOCAB` clause used to sit here and read as a live fallback to
     * argmax proposals; it could never be false past load (L176). */
    const bool sample_drafts = temperature > 0.0f;
    if (sample_drafts) {
        /* n_draft rows, not the whole PULSAR_SPEC_LOGITS_ROWS slab: the depth is
         * bounded per engine, so this is ~n_draft x 0.5 MB per session rather
         * than the slab's ~16.5 MB. Grow-only, so a depth change is still safe. */
        const uint32_t need = n_draft * PULSAR_N_VOCAB;
        if (s->dspark_pending_qrows_cap < need) {
            free(s->dspark_pending_qrows);
            s->dspark_pending_qrows = (float *)xmalloc((size_t)need * sizeof(float));
            s->dspark_pending_qrows_cap = need;
        }
    }
    /* spec_logits rows are PULSAR_N_VOCAB wide (allocated PULSAR_SPEC_LOGITS_ROWS*PULSAR_N_VOCAB,
     * written and read elsewhere at that stride via gpu_graph_read_spec_logits_row).
     * Stride the row view by the TARGET vocab, not the drafter's vocab_size:
     * they are equal on the shipped drafter (markov head 129280 == N_VOCAB), so
     * this is bit-exact today, but a drafter with a smaller vocab would make
     * pos>=1 read into the middle of row 0.  vocab_size stays the LENGTH the
     * markov step consumes. */
    const uint64_t spec_row_bytes = (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    bool draft_ok = true;
    /* L108 P2: with no host consumer at draft time (greedy, no diagnostics),
     * launch the chain + conf scoring and DEFER the readback to the next
     * consumer (pulsar_session_spec_chain_harvest) -- the caller's token
     * emission then overlaps the drafter's GPU time. */
    const bool defer_harvest = !sample_drafts && !gpu_graph_spec_dump_active();
    if (!sample_drafts) {
        /* L108 P1: the greedy walk chains ON DEVICE.  The old loop did a
         * blocking 8-byte read per position purely to hand the next step a
         * token id that already lived in device memory -- ~depth syncs per
         * round, the single largest host-serialization line in the P1 trace.
         * Seed ids[0], launch the whole chain, read all ids back ONCE.  Same
         * kernels, same launch order, same arithmetic: byte-exact (the only
         * behavioural delta is that the chain clamps an out-of-vocab id
         * in-kernel where the loop refused host-side -- unreachable either
         * way, ids are argmaxes over the vocab).  The SAMPLED path keeps the
         * loop below: its chain routes through a host rng draw per position. */
        draft_ok =
            pulsar_gpu_tensor_write(g->dspark_refined_ids, 0, &refined[0],
                                    sizeof(int32_t)) &&
            pulsar_gpu_dspark_markov_chain_model(dspark_logits,
                                              g->dspark_refined_ids,
                                              g->spec_logits, spec_row_bytes,
                                              dmap, dsize,
                                              w->markov_w1->abs_offset,
                                              w->markov_w2->abs_offset,
                                              n_draft, vocab_size, embed_dim,
                                              w->markov_w1->type == PULSAR_TENSOR_BF16,
                                              w->markov_w2->type == PULSAR_TENSOR_BF16) &&
            (defer_harvest ||
             pulsar_gpu_tensor_read(g->dspark_refined_ids, sizeof(int32_t),
                                    &refined[1], (uint64_t)n_draft * sizeof(int32_t)));
    } else
    for (uint32_t pos = 0; pos < n_draft && draft_ok; pos++) {
        pulsar_gpu_tensor *base_row = pulsar_gpu_tensor_view(
            g->spec_logits, (uint64_t)pos * spec_row_bytes, vocab_bytes);
        draft_ok = base_row &&
            pulsar_gpu_dspark_markov_step_model(dspark_logits, &refined[pos + 1],
                                             base_row, dmap, dsize,
                                             w->markov_w1->abs_offset,
                                             w->markov_w2->abs_offset,
                                             refined[pos], vocab_size, embed_dim,
                                             w->markov_w1->type == PULSAR_TENSOR_BF16,
                                             w->markov_w2->type == PULSAR_TENSOR_BF16);
        pulsar_gpu_tensor_free(base_row);
        if (!draft_ok || !sample_drafts) continue;
        /* Build this position's proposal q BEFORE the next markov step
         * overwrites the single-row scratch, and keep it: the residual needs
         * the q of whichever position rejects, which is not known until verify.
         *
         * L149: the production shape (top_k 0, top_p 1, min_p 0.05) needs only
         * the candidates above the min-p floor, and the min-p cutoff is
         * division-free (tokenizer.cpp), so the device prefilter hands back
         * the few survivors instead of the 517 KB row -- the read shrinks to
         * one small block and the host skips the 129k-expf normaliser pass
         * that idled the GPU ~630 us per draft position. The built q is stored
         * for the residual (dspark_pending_qn), so the walk skips the rebuild
         * too. Any other shape, or a candidate set wider than the compact
         * block, reads the full row; a device failure or a refused candidate
         * block ends the draft (L190 D3). */
        pulsar_sample_dist q;
        bool q_built = false;
        s->spec.dspark_pending_qn[pos] = 0;
        if (top_k <= 0 && top_p == 1.0f && min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN &&
            min_p <= 1.0f && vocab_size <= PULSAR_SAMPLE_SPARSE_VOCAB_MAX &&
            g->dspark_prefilter_sel) {
            /* floor in logit units, a hair below T*ln(min_p): the host
             * comparison decides membership, this only bounds the read */
            const float delta = (float)((double)temperature * (log((double)min_p) - 1e-3));
            int32_t sel[PULSAR_DSPARK_PREFILTER_ROW_I32];
            /* The shape is in the sparse contract, so from here a device
             * failure is a failure: the prefilter launch, its readback, a row
             * with no finite logit (n_sel == 0) or a candidate block the host
             * build refuses all end the draft loudly.  A row with MORE
             * candidates above the floor than the compact block holds is the
             * one genuine ineligibility and reads the full row below (L190
             * D3, the L174 class: a device error used to read as the slower
             * path). */
            if (!pulsar_gpu_minp_prefilter_rows(g->dspark_prefilter_sel, dspark_logits, 0, 1u,
                                                vocab_size, vocab_size, delta,
                                                PULSAR_DSPARK_PREFILTER_CAP) ||
                !pulsar_gpu_tensor_read(g->dspark_prefilter_sel, 0, sel, sizeof(sel))) {
                fprintf(stderr, "pulsar: dspark min-p prefilter failed at draft position %u -- "
                                "no drafts this round (no full-row fallback; L190)\n", pos);
                draft_ok = false;
                break;
            }
            const uint32_t n_sel = (uint32_t)sel[0];
            if (n_sel <= PULSAR_DSPARK_PREFILTER_CAP) {
                float max_logit;
                memcpy(&max_logit, &sel[2], sizeof(max_logit));
                if (n_sel == 0 ||
                    !pulsar_sample_dist_build_prefiltered(
                        sel + 3, (const float *)(sel + 3 + PULSAR_DSPARK_PREFILTER_CAP),
                        n_sel, max_logit, temperature, min_p, &s->sample_scratch, &q)) {
                    fprintf(stderr, "pulsar: dspark min-p prefilter handed %u candidates at draft "
                                    "position %u and the build refused them -- no drafts this "
                                    "round (L190)\n", n_sel, pos);
                    draft_ok = false;
                    break;
                }
                q_built = true;
            }
            if (q_built && q.n <= PULSAR_DSPARK_QDIST_CAP) {
                s->spec.dspark_pending_qn[pos] = q.n;
                memcpy(s->spec.dspark_pending_qids[pos], q.ids, (size_t)q.n * sizeof(int32_t));
                memcpy(s->spec.dspark_pending_qprobs[pos], q.probs, (size_t)q.n * sizeof(float));
            } else if (q_built) {
                /* too wide to store: keep q, but the residual will need the row */
                float *qrow = s->dspark_pending_qrows + (size_t)pos * PULSAR_N_VOCAB;
                if (!pulsar_gpu_tensor_read(dspark_logits, 0, qrow, vocab_bytes)) {
                    pulsar_sample_dist_free(&q);
                    draft_ok = false;
                    break;
                }
            }
        }
        if (!q_built) {
            float *qrow = s->dspark_pending_qrows + (size_t)pos * PULSAR_N_VOCAB;
            if (!pulsar_gpu_tensor_read(dspark_logits, 0, qrow, vocab_bytes) ||
                !pulsar_sample_dist_build(qrow, PULSAR_N_VOCAB, temperature, top_k, top_p, min_p,
                                          &s->sample_scratch, &q)) {
                draft_ok = false;
                break;
            }
        }
        const int drawn = pulsar_sample_dist_draw(&q, rng);
        refined[pos + 1] = (int32_t)drawn;   /* the chain continues SAMPLED */
        s->spec.dspark_pending_q[pos] = pulsar_sample_dist_prob(&q, drawn);
        /* Diagnostic: how much proposal entropy is there actually? The whole
         * premise of temperature-matched drafting is that q is a DISTRIBUTION.
         * If q.n == 1 (or q(top) ~ 1) the draw is the argmax, min(1,p/q)
         * degenerates to the deterministic rule, and Item 1 is a no-op. */
        if (dspark_stats)
            fprintf(stderr, "DSPARK_Q pos=%u q_n=%u q_top=%.4f q_drawn=%.4f "
                            "drawn_is_argmax=%d\n",
                    pos, q.n, (double)q.probs[0],
                    (double)s->spec.dspark_pending_q[pos], drawn == q.ids[0]);
        pulsar_sample_dist_free(&q);
    }
    if (!draft_ok) return 0;

    /* Offline-validation / confidence-training dump (same hook as the legacy
     * loop, which the production fused path previously never reached). Emitted
     * after markov refine, while batch_ffn_cur still holds the post-hc_head
     * hidden rows the confidence kernel consumes. */
    dspark_dump_step(g, (int)s->checkpoint.len, (int)next_base, refined, (int)n_draft);

    /* Confidence-scheduled pending length (P1 head; keep the confident prefix). */
    uint32_t keep = n_draft;
    float conf[16];
    bool have_conf = false;
    bool conf_deferred = false;
    {
        const float tau = dspark_conf_sched_tau();
        if (tau > 0.0f) {
            /* Persistent graph-owned scratch (n_draft is clamped to 16 above):
             * the alloc/free pair here ran every fused spec step and each
             * cudaMalloc/cudaFree serializes the device, which is exactly the
             * pattern already retired in gpu_decode's dspark projection. */
            pulsar_gpu_tensor *conf_dev = g->dspark_conf_scores;
            pulsar_gpu_tensor *tok_dev = g->dspark_conf_tokens;
            const bool scored =
                conf_dev && tok_dev &&
                (defer_harvest
                     /* device-to-device: the ids are already in the chain
                      * array; a host write here would force the read this
                      * path exists to avoid */
                     ? pulsar_gpu_tensor_copy(tok_dev, 0, g->dspark_refined_ids, 0,
                                              (uint64_t)n_draft * sizeof(int32_t)) != 0
                     : pulsar_gpu_tensor_write(tok_dev, 0, refined,
                                               (uint64_t)n_draft * sizeof(int32_t)) != 0) &&
                pulsar_gpu_dspark_confidence_score_model(conf_dev, g->batch_ffn_cur, tok_dev,
                                                      dmap, dsize,
                                                      w->markov_w1->abs_offset,
                                                      w->confidence_proj->abs_offset,
                                                      n_draft, PULSAR_N_EMBD, embed_dim, vocab_size,
                                                      w->markov_w1->type == PULSAR_TENSOR_BF16,
                                                      w->confidence_proj->type == PULSAR_TENSOR_BF16) &&
                (defer_harvest ||
                 pulsar_gpu_tensor_read(conf_dev, 0, conf, (uint64_t)n_draft * sizeof(float)));
            if (!scored) {
                /* L190 D4: with tau > 0 the trim is scheduled; a scoring or
                 * readback failure used to keep the untrimmed chain and feed
                 * -1 to the depth controller without a word.  The round
                 * drafts nothing instead, and says so. */
                fprintf(stderr, "pulsar: dspark confidence scoring failed for %u drafts -- "
                                "no drafts this round (the tau trim is not skipped silently; L190)\n",
                        n_draft);
                return 0;
            }
            if (defer_harvest) {
                conf_deferred = true;   /* harvest reads + trims later */
            } else {
                have_conf = true;
                uint32_t k = 0;
                while (k < n_draft && conf[k] >= tau) k++;
                keep = k;   /* 0 pending = next step is a plain n=1 forward */
            }
        }
    }
    s->spec.dspark_pending_base = (int32_t)next_base;
    if (defer_harvest) {
        s->spec.dspark_n_pending = 0;   /* harvest sets the real count */
        s->spec.dspark_chain_unharvested = true;
        s->spec.dspark_chain_conf = conf_deferred;
        s->spec.dspark_chain_n = n_draft;
    } else {
        s->spec.dspark_n_pending = keep;
        for (uint32_t i = 0; i < keep; i++) s->spec.dspark_pending[i] = refined[i + 1];
    }
    /* The proposal rule and the exact params these drafts were sampled under.
     * Stamped unconditionally, in the same straight-line block as
     * dspark_n_pending above — this is the only site that makes pendings
     * non-zero, so a populated dspark_pending_q[]/qrows pool (filled in the
     * drafting loop above, under sample_drafts) always has its params alongside
     * it. Three consumers: the verify walk picks its accept rule from the flag,
     * next step's params guard compares against the params, and — load-bearing —
     * the residual rebuilds q from the qrows under THESE params, so the stored
     * accept denominator and the residual describe one proposal. */
    s->spec.dspark_pending_sampled = sample_drafts;
    s->spec.dspark_pending_pos = (int32_t)s->checkpoint.len;
    s->spec.dspark_pending_temp = temperature;
    s->spec.dspark_pending_top_k = top_k;
    s->spec.dspark_pending_top_p = top_p;
    s->spec.dspark_pending_min_p = min_p;
    for (uint32_t i = 0; i < keep; i++)   /* L107: controller reads these in round_end */
        s->spec.dspark_pending_conf[i] = have_conf ? conf[i] : -1.0f;

    return keep;
}

/* L108 P2: lazy completion of a device-chained greedy draft. Mirrors the
 * immediate path's semantics: a failed ids or confidence readback drops the
 * chain (0 pendings, the next step is a plain forward) and says so -- never
 * an untrimmed keep with -1 conf fed to the controller (L190 D4).  Merge
 * marker for L107: when the
 * adaptive-depth controller lands, its unconditional dspark_pending_conf
 * store must be replicated here. */
void pulsar_session_spec_chain_harvest(pulsar_session *s) {
    if (!s->spec.dspark_chain_unharvested) return;
    s->spec.dspark_chain_unharvested = false;
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t n_draft = s->spec.dspark_chain_n;
    if (n_draft == 0 || n_draft > 16u) return;
    int32_t ids[17];
    if (!g->dspark_refined_ids ||
        !pulsar_gpu_tensor_read(g->dspark_refined_ids, sizeof(int32_t), ids + 1,
                                (uint64_t)n_draft * sizeof(int32_t))) {
        fprintf(stderr, "pulsar: dspark chain harvest: draft ids readback failed -- "
                        "the chain is dropped (0 pendings)\n");
        s->spec.dspark_n_pending = 0;
        return;
    }
    uint32_t keep = n_draft;
    float conf[16];
    const bool have_conf = s->spec.dspark_chain_conf;
    if (have_conf) {
        if (!pulsar_gpu_tensor_read(g->dspark_conf_scores, 0, conf,
                                    (uint64_t)n_draft * sizeof(float))) {
            /* L190 D4: this used to keep the untrimmed chain and feed -1 to
             * the depth controller without a word; a chain whose confidence
             * cannot be read is dropped, and said so. */
            fprintf(stderr, "pulsar: dspark chain harvest: confidence readback failed -- "
                            "the chain is dropped (0 pendings)\n");
            s->spec.dspark_n_pending = 0;
            return;
        }
        const float tau = dspark_conf_sched_tau();
        if (tau > 0.0f) {
            uint32_t k = 0;
            while (k < n_draft && conf[k] >= tau) k++;
            keep = k;
        }
    }
    s->spec.dspark_n_pending = keep;
    for (uint32_t i = 0; i < keep; i++) {
        s->spec.dspark_pending[i] = ids[i + 1];
        /* L107 merge: the adaptive-depth controller reads the verified
         * chain's conf at round_end (pend_conf via round assembly) -- the
         * deferred path must store it here, exactly as the immediate path's
         * unconditional store does at draft time. */
        s->spec.dspark_pending_conf[i] = have_conf ? conf[i] : -1.0f;
    }
}

/** inc-6: one speculative ROUND's state, threaded between round_begin (rows
 * assembled, frontier snapshotted, checkpoint pushed), the verify forward
 * (classic in the fused loop; the SHARED decode_mixed ALL_ROWS forward in
 * the batched lane), and round_end (walk, state, emit, redraft). */
/* L150: what a bank's redraft needs, recorded by round_end in the batched lane
 * (which defers drafting), consumed by pulsar_session_spec_redraft_batch. */
typedef struct {
    bool     valid;         ///< round_end reached the drafting point with main_x ready
    int32_t  next_base;     ///< the carry: batch position 0 of the next round
    uint32_t n_draft;       ///< the bank's depth at round_end (spec_cur_depth), <= 16
    uint32_t n_batch;
    int      commit;
    float    temperature, top_p, min_p;
    int      top_k;
    /* results, filled by the batched redraft; committed into the bank's shadow
     * by pulsar_session_spec_redraft_commit under the server's bank switch */
    bool     done;
    /** L190 C2: the "past the drafter batch cap" line was said for this round
     * object; survives spec_round_begin's reset (like qrows) so it is said
     * once per affected round, not once per process or once per step. */
    bool     cap_said;
    bool     sample_drafts;
    uint32_t keep;
    bool     have_conf;
    int32_t  refined[17];
    float    conf[16];
    float    q_drawn[16];                                  ///< q(pend[i]) per position (sampled)
    uint32_t qn[16];                                       ///< stored proposal dist size (0 = row fallback)
    int32_t  qids[16][PULSAR_DSPARK_QDIST_CAP];
    float    qprobs[16][PULSAR_DSPARK_QDIST_CAP];
    float   *qrows;                                        ///< [16][PULSAR_N_VOCAB] fallback rows, lazily owned
} spec_redraft_req;

typedef struct pulsar_spec_round {
    uint32_t K;             ///< draft depth this round: tokens the drafter proposed
    uint32_t n_batch;       ///< rows the verify forward evaluates (K + the base row)
    int saved_len;          ///< checkpoint length before the round, to trim back to on rejection
    bool pend_sampled;      ///< the pendings below were sampled rather than left from a prior round
    int32_t pend[16];       ///< the drafted token ids, in draft order
    float pend_conf[16];    ///< confidence head's score per draft position
    int row_tops[16];       ///< the target's argmax per verify row; the accept comparison
    /** Frontier snapshot taken before the drafter touched anything, so a
     * partial accept can roll every layer's compressed state back exactly. */
    pulsar_spec_frontier frontier;
    spec_redraft_req redraft;   ///< L150: batched-lane deferred redraft (see the typedef)
} pulsar_spec_round;

/* Assemble the round: load/guard the pendings, seed the prompt window if
 * fresh, snapshot the frontier, push first_token + pendings onto the
 * checkpoint. Body verbatim from the fused loop; the locals it declared are
 * now reference-bound round fields so the batched lane can run one round
 * per bank around a shared forward. Returns 0 or -1 (snapshot failure,
 * session poisoned) exactly as before. */
static int spec_round_begin(pulsar_session *s, int first_token,
                            int max_tokens, int accepted_cap,
                            float temperature, int top_k, float top_p, float min_p,
                            pulsar_spec_round *r,
                            char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    if (first_token < 0 || (uint32_t)first_token >= PULSAR_N_VOCAB) {
        /* -1 is what pulsar_session_spec_next_base returns when the sampler
         * refused the row; the base token feeds the embedding gather, so a
         * non-id must stop here. */
        snprintf(err, errlen, "spec round: base token %d is not a vocab id", first_token);
        return -1;
    }
    {
        /* the round is reused across rounds; its redraft record's lazily-owned
         * fallback rows (L150) survive the reset, everything else is cleared */
        float *qrows = r->redraft.qrows;
        const bool cap_said = r->redraft.cap_said;
        memset(r, 0, sizeof(*r));
        r->redraft.qrows = qrows;
        r->redraft.cap_said = cap_said;
    }
    /* L149 phase 2: accumulate, for the step this round will ride, whether
     * every round is in the sparse min-p contract and the most permissive
     * floor among them (pulsar_session_spec_arm_capture arms from this). A
     * round that begins here but sits the step out only makes the result more
     * conservative (ok=false) or the superset wider (lower floor): both safe. */
    {
        const bool in_contract = temperature > 0.0f && top_k <= 0 && top_p == 1.0f &&
                                 min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && min_p <= 1.0f &&
                                 PULSAR_N_VOCAB <= PULSAR_SAMPLE_SPARSE_VOCAB_MAX;
        const float d = in_contract
            ? (float)((double)temperature * (log((double)min_p) - 1e-3)) : 0.0f;
        if (g->spec_compact_acc_n == 0) {
            g->spec_compact_acc_ok = in_contract;
            g->spec_compact_acc_delta = d;
        } else {
            g->spec_compact_acc_ok = g->spec_compact_acc_ok && in_contract;
            if (d < g->spec_compact_acc_delta) g->spec_compact_acc_delta = d;
        }
        g->spec_compact_acc_n++;
    }
    uint32_t &K = r->K;
    uint32_t &n_batch = r->n_batch;
    int &saved_len = r->saved_len;
    bool &pend_sampled = r->pend_sampled;
    int32_t (&pend)[16] = r->pend;
    float (&pend_conf)[16] = r->pend_conf;
    pulsar_spec_frontier &frontier = r->frontier;
    (void)pend_conf; (void)pend_sampled; (void)n_batch;
    /* Pending drafts continue from the greedy base we predicted last step; if
     * the caller committed something else (tool injection, sampling change),
     * they are stale. */
    pulsar_session_spec_chain_harvest(s);   /* L108 P2 */
    K = s->spec.dspark_n_pending;
    if (K > 16u) K = 16u;
    if (K && s->spec.dspark_pending_base != (int32_t)first_token) K = 0;
    /* Position guard — ACCEPTANCE, not exactness. The drafts were conditioned on
     * the old position; dspark_pending_base is a token VALUE, so a plain eval
     * that advances the session (tool injection, </think> recovery) followed by
     * a first_token that happens to collide with it would otherwise resurrect
     * them. Dropping them is a throughput choice: a draft conditioned on the
     * wrong position is near-worthless and would just burn a verify row. It is
     * NOT what keeps the output exact — both accept rules are proposal-agnostic
     * (see the walk below), so q_oldpos is still exactly the distribution the
     * draft was drawn from and the rule still yields p at the new position.
     * Mirrors carry_pos_match. (checkpoint.len is still the drafting-step value
     * here: this runs before the first_token push below.) */
    if (K && s->spec.dspark_pending_pos != (int32_t)s->checkpoint.len) K = 0;
    /* Params guard — also acceptance, not exactness. Drafts drawn under params X
     * and verified under params Y are still verified EXACTLY: the residual
     * rebuilds q under the stored params X (see the walk below), so the accept
     * denominator and the residual name one proposal q_X, and the p/q rule
     * returns exactly p_Y for any q. What a param change costs is acceptance —
     * q_X is a poor proposal for p_Y — so we drop them and draft afresh.
     * Mirrors the spec_carry_* params guard in pulsar_session_generate_speculative.
     *
     * The same holds for argmax drafts (dspark_pending_sampled == false), which
     * carry no q: a temp<=0 -> temp>0 change cannot misroute them, because the
     * walk picks its rule from pend_sampled, not from the live temperature, so
     * they still meet the deterministic rule — itself exact for an arbitrary
     * proposal. The guard just keeps a badly-matched proposal from wasting a
     * row. */
    const bool pending_params_match =
        s->spec.dspark_pending_temp == temperature && s->spec.dspark_pending_top_k == top_k &&
        s->spec.dspark_pending_top_p == top_p && s->spec.dspark_pending_min_p == min_p;
    if (K && !pending_params_match) K = 0;
    /* Proposal rule the pendings were drafted under; the verify walk must apply
     * the matching rule (see dspark_pending_sampled). */
    pend_sampled = K ? s->spec.dspark_pending_sampled : false;
    if ((int)K > accepted_cap - 1) K = accepted_cap > 1 ? (uint32_t)(accepted_cap - 1) : 0;
    if ((int)K > max_tokens - 1) K = max_tokens > 1 ? (uint32_t)(max_tokens - 1) : 0;
    for (uint32_t i = 0; i < K; i++) pend[i] = s->spec.dspark_pending[i];
    /* Conf carried unconditionally (L107 controller). */
    for (uint32_t i = 0; i < K; i++) pend_conf[i] = s->spec.dspark_pending_conf[i];
    pulsar_spec_drop_pendings(&s->spec);
    s->spec.spec_carry_valid = false;
    n_batch = 1u + K;

    /* Prompt-window seeding (one-time per prompt): fresh drafter state + a
     * captured prompt window -> replay the last <=128 prompt positions into
     * the drafter's context-KV ring, exactly as the reference prefills it.
     * Without this the window starts empty (or, before the invalidate fix,
     * stale from the previous request), and the drafter is near-useless
     * without a valid window (masked-window eval: 4.7% vs 86% top-1). */
    if (g->dspark_n_raw[0] == 0 && g->dspark_prompt_n > 0 && g->dspark_prompt_h[0]) {
        const uint32_t win = PULSAR_DSPARK_DRAFT_WINDOW;
        uint32_t avail = g->dspark_prompt_n - g->dspark_prompt_lo;
        const uint32_t take = avail < win ? avail : win;
        const uint32_t first = g->dspark_prompt_n - take;
        bool seed_ok = true;
        uint32_t j = 0;
        for (; seed_ok && j < take; j++) {
            const uint32_t slot = (first + j) % win;
            for (int i = 0; seed_ok && i < 3; i++) {
                seed_ok = pulsar_gpu_tensor_copy(g->dspark_target_h[i], 0,
                                              g->dspark_prompt_h[i],
                                              (uint64_t)slot * PULSAR_N_EMBD * sizeof(float),
                                              (uint64_t)PULSAR_N_EMBD * sizeof(float)) != 0;
            }
            if (seed_ok)
                seed_ok = gpu_graph_dspark_project_main_x(g, &e->dspark_model, w) &&
                          gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, w, 1);
        }
        if (!seed_ok) {
            /* Refuse the round rather than draft from a half-seeded window
             * (L167; a failed projection used to be skipped over and a failed
             * seed warned once and drafted unseeded).  The target state is
             * untouched and the window stays pending for the next attempt. */
            snprintf(err, errlen, "DSpark prompt-window seed failed after %u of %u rows", j - 1u, take);
            return -1;
        }
        if (dspark_stats)
            fprintf(stderr, "pulsar: dspark prompt-window seeded %u rows (prompt_n=%u)\n",
                    take, g->dspark_prompt_n);
        g->dspark_prompt_n = 0;   /* consumed; commits take over from here */
    }

    memset(&frontier, 0, sizeof(frontier));
    if (!spec_frontier_snapshot(&frontier, s)) {
        snprintf(err, errlen, "DSpark fused frontier snapshot failed");
        s->checkpoint_valid = false;
        return -1;
    }

    saved_len = s->checkpoint.len;
    token_vec_push(&s->checkpoint, first_token);
    for (uint32_t i = 0; i < K; i++) token_vec_push(&s->checkpoint, (int)pend[i]);
    return 0;
}

/* Finish the round over an already-run verify forward: accept walk, EOS
 * clamp, metrics, quench, logits refresh, carry, state seed/rollback, emit,
 * pendings redraft. Body verbatim from the fused loop with three seams for
 * the batched lane: `read_row`/`read_ud` name the forward's logit rows
 * (classic: spec_logits; batched: the shared ALL_ROWS block at the bank's
 * offset), and `row0` is this round's first row in the forward's
 * capture/comp-save buffers (classic: 0). Owns r->frontier's lifetime on
 * every path. Returns tokens emitted, or -1 (session poisoned). */
static int spec_round_end(pulsar_session *s, pulsar_spec_round *r,
                          int first_token, int eos_token,
                          float temperature, int top_k, float top_p, float min_p,
                          uint64_t *rng,
                          spec_row_read_fn read_row, void *read_ud, uint32_t row0,
                          const int32_t *compact,
                          bool defer_redraft,
                          double t0,
                          const int32_t *forced_truth,
                          int *accepted, int accepted_cap,
                          char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    int n_accept = 0;
    const uint32_t K = r->K;
    const uint32_t n_batch = r->n_batch;
    const int saved_len = r->saved_len;
    const bool pend_sampled = r->pend_sampled;
    int32_t (&pend)[16] = r->pend;
    float (&pend_conf)[16] = r->pend_conf;
    int (&row_tops)[16] = r->row_tops;
    pulsar_spec_frontier &frontier = r->frontier;
    (void)pend_conf; (void)t0; (void)dspark_stats;
    /* Accept the longest prefix the target agrees with. Greedy: row i's
     * argmax must equal pend[i]. Sampled: exact speculative sampling under the
     * request's FILTERED target distribution p_i, with the rule matched to how
     * the draft was PROPOSED:
     *   - sampled proposal (the production path at temperature > 0): pend[i]
     *     was drawn from a temperature-matched q_i, so accept w.p.
     *     min(1, p_i/q_i) and on rejection draw the residual (p_i - q_i)+.
     *     Acceptance is NOT capped at p_i(mode).
     *   - argmax proposal (temperature == 0): the deterministic rule -- accept
     *     w.p. p_i(pend[i]), residual p_i with pend[i] excluded. Capped at
     *     p_i(mode).
     * The rejected row's replacement becomes the carry token. All three paths
     * yield the exact per-token target distribution. */
    int carry_tok = -1;
    int commit_rc;
    if (forced_truth) {
        /* TEACHER-FORCED end (gate-facing, L182): the walk is replaced by
         * equality against the true continuation -- draft i is accepted iff it
         * IS truth[i], the carry is the first true token the drafts did not
         * cover, and no draw is made.  The context the redraft conditions on
         * is the corpus, so the drafter's next proposal is a deterministic
         * function of (model, drafter, context).  Everything after this point
         * -- controllers, metrics, the trim, the redraft -- runs as in
         * production. */
        commit_rc = 0;
        while ((uint32_t)commit_rc < K && pend[commit_rc] == forced_truth[commit_rc]) commit_rc++;
        carry_tok = (int)forced_truth[commit_rc];
    } else {
        commit_rc = spec_accept_walk(s, read_row, read_ud, compact, row0,
                                     row_tops, pend, K, pend_sampled,
                                     temperature, top_k, top_p, min_p,
                                     rng, &carry_tok);
    }
    if (commit_rc < 0) {
        s->checkpoint.len = saved_len;
        (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "DSpark sampled-accept logits readback failed");
        s->checkpoint_valid = false;
        return -1;
    }
    int commit = commit_rc;

    /* Ghost-token guard: never commit drafts PAST an accepted EOS. The
     * emission loop below stops at EOS, but the checkpoint trim uses commit —
     * without this clamp the bank history keeps invisible tokens after the
     * stop: the exact-frontier warm gate misses next turn, and the
     * thinking-live continuation embeds the ghosts into the next prompt. The
     * EOS row itself stays committed (it was evaluated in the batch, matching
     * the batched lane's convention); hit_eos below invalidates the carry, so
     * nothing ever samples from a post-EOS row. */
    if ((int)first_token == (int)eos_token) {
        commit = 0;
    } else {
        for (int i = 0; i < commit; i++) {
            if (pend[i] == (int32_t)eos_token) { commit = i + 1; break; }
        }
    }

    /* Prometheus /metrics spec-decode counters (server /metrics endpoint). The
     * base token is always emitted; K drafts were verified this step and the
     * accepted prefix is [0,commit). num_drafts counts draft rounds only. */
    e->spec_gen_tokens += 1u + (uint64_t)commit;
    s->spec.spec_gen_tokens += 1u + (uint64_t)commit;
    if (K > 0) {
        e->spec_draft_tokens += K;
        e->spec_accepted_tokens += (uint64_t)commit;
        e->spec_num_drafts += 1u;
        for (int i = 0; i < commit && i < 16; i++) e->spec_accepted_per_pos[i]++;
        s->spec.spec_draft_tokens += K;
        s->spec.spec_accepted_tokens += (uint64_t)commit;
        s->spec.spec_num_drafts += 1u;
    }

    /* L107 adaptive draft depth (constants + rationale at spec_cur_depth).
     * Runs BEFORE the redraft below so the next chain is drafted at the new
     * depth. commit == depth implies the trimmer kept the whole chain AND the
     * target accepted all of it (commit <= K <= depth always). A tail conf of
     * -1 (head didn't run, e.g. conf-sched disabled) passes the UP check: the
     * full-accept signal alone then drives the climb. Counts-only decision,
     * deterministic for a fixed stream, same property as yield-quench. */
    if (K > 0) {
        const uint32_t depth = spec_cur_depth(s);
        if (s->spec.spec_depth_climb_cooldown) s->spec.spec_depth_climb_cooldown--;
        if (s->spec.spec_depth_rounds_since_up < 255u) s->spec.spec_depth_rounds_since_up++;
        int next = (int)depth;
        if (2u * (uint32_t)commit < depth) {
            /* v2 down-veto: the A/B trajectory showed single bad rounds at
             * depth 5 with tail conf ~0.98 knocking depth down and costing
             * ~2 t/s until the climb back. When the calibrated head still
             * believed in the chain (tail >= 0.90 -> measured 1.000-accept
             * band), forgive ONE down-signal; a second consecutive one backs
             * off regardless. Prose tails run ~0.5-0.7, so its immediate
             * back-off is untouched. */
            /* v3: the veto applies ONLY at depth 5 -- the measured
             * structured optimum it exists to protect. Below 5 it is what
             * regressed prose in the v2 A/B (prose has occasional >=0.90
             * tails and each forgiven round pays a deep draft that converts
             * nothing); at 6 it delays the return to 5, and 6 is never
             * optimal (sweep: depth 6 lost on BOTH regimes). */
            if (depth == 5u &&
                pend_conf[K - 1] >= 0.90f && !s->spec.spec_depth_down_forgiven) {
                s->spec.spec_depth_down_forgiven = true;
            } else {
                s->spec.spec_depth_down_forgiven = false;
                /* v5: only a FAILED EXCURSION (down within 2 rounds of the
                 * last up) triggers the cooldown. */
                if (s->spec.spec_depth_rounds_since_up <= 2u)
                    s->spec.spec_depth_climb_cooldown = 8u;
                next--;
            }
        } else if ((uint32_t)commit == depth &&
                   s->spec.spec_depth_climb_cooldown == 0u &&
                   (pend_conf[depth - 1] >= SPEC_DEPTH_CONF_UP ||
                    pend_conf[depth - 1] < 0.0f)) {
            s->spec.spec_depth_down_forgiven = false;
            s->spec.spec_depth_rounds_since_up = 0u;
            next++;
        } else {
            s->spec.spec_depth_down_forgiven = false;
        }
        if (next < SPEC_DEPTH_MIN) next = SPEC_DEPTH_MIN;
        if (next > SPEC_DEPTH_MAX) next = SPEC_DEPTH_MAX;
        if (dspark_stats && (uint32_t)next != depth)
            fprintf(stderr, "pulsar: adaptive-k depth %u -> %d (commit=%d K=%u tail=%.2f)\n",
                    depth, next, commit, K, (double)pend_conf[K - 1]);
        s->spec.spec_adaptive_depth = next;
    }

    /* Yield-quench controller update (see the constants block up top). Uses
     * the ACTUAL verify width n_batch (post conf-sched trim last step) and the
     * ACTUAL committed yield 1+commit — counts only, so the decision is
     * deterministic for a fixed stream. Once latched, generate_speculative
     * routes this request's remaining tokens down the plain-decode path and
     * the drafting block below is skipped; both paths sample the exact target
     * distribution, so quenching changes speed, never marginals. */
    if (!s->spec.spec_quenched) {
        s->spec.spec_quench_steps++;
        bool fire = false;
        if (s->spec.spec_quench_steps > PULSAR_QUENCH_WARMUP) {
            const float margin = (1.0f + (float)commit) -
                                 spec_quench_guard(n_batch, saved_len);
            s->spec.spec_quench_ewma = (1.0f - PULSAR_QUENCH_ALPHA) * s->spec.spec_quench_ewma +
                                  PULSAR_QUENCH_ALPHA * margin;
            s->spec.spec_quench_debt -= margin;   /* unclamped: NET tokens lost */
            fire = s->spec.spec_quench_steps >= PULSAR_QUENCH_MINEV &&
                   s->spec.spec_quench_ewma < 0.0f &&
                   s->spec.spec_quench_debt > PULSAR_QUENCH_BUDGET;
        }
        if (fire) {
            s->spec.spec_quenched = true;
            fprintf(stderr,
                    "pulsar: dspark yield-quench pos=%d steps=%u debt=%.2f ewma=%.2f "
                    "-> plain decode for request remainder%s\n",
                    saved_len + 1 + commit, s->spec.spec_quench_steps,
                    (double)s->spec.spec_quench_debt, (double)s->spec.spec_quench_ewma,
                    "");
        }
    }

    /* Refresh s->logits to the last committed position's distribution. */
    if (!read_row(read_ud, (uint32_t)commit, s->logits)) {
        s->checkpoint.len = saved_len;
        (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "DSpark fused logits readback failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Finalize the carry token — the next base, drawn from the refreshed
     * s->logits (= row[commit]) when the walk did not already draw a residual:
     * greedy -> argmax (the old next_base); sampled full-accept -> the bonus
     * draw from the last accepted row's distribution. */
    if (carry_tok < 0) {
        if (temperature <= 0.0f) {
            carry_tok = sample_argmax(s->logits, PULSAR_N_VOCAB);
        } else {
            pulsar_sample_dist bonus;
            if (pulsar_sample_dist_build(s->logits, PULSAR_N_VOCAB, temperature, top_k,
                                         top_p, min_p, &s->sample_scratch, &bonus)) {
                carry_tok = pulsar_sample_dist_draw(&bonus, rng);
                pulsar_sample_dist_free(&bonus);
            }
        }
        if (carry_tok < 0) {
            /* the committed row has no drawable distribution (the sampler said
             * why, once): no carry can be honest, so the round fails */
            s->checkpoint.len = saved_len;
            (void)spec_frontier_restore(&frontier, s);
            spec_frontier_free(&frontier);
            snprintf(err, errlen, "DSpark fused: the sampler refused the carry row");
            s->checkpoint_valid = false;
            return -1;
        }
    }

    bool ok_state = true;
    /* Every path below leaves main_x projected from the LAST committed hidden:
     * the two seeding loops go through dspark_seed_from_batch_row, which
     * projects it itself, and the replay loop projects per token.  The only way
     * it can be stale is a projection failure in the replay path, which is
     * tolerated there — so track that instead of re-projecting to find out. */
    bool main_x_ready = true;
    if (commit == (int)K) {
        /* Full accept: the batch advanced the target state by exactly the
         * committed tokens. Seed drafter rows for every committed position from
         * its own captured hidden (row j = f(h_j)); the last copy leaves
         * target_h = the drafting hidden. */
        s->checkpoint.len = saved_len + 1 + commit;
        for (int m = 0; ok_state && m <= commit && m < (int)n_batch; m++)
            ok_state = dspark_seed_from_batch_row(s, row0 + (uint32_t)m);
    } else {
        /* Partial/zero accept: target state includes rejected positions.
         * Stage A: restore the pre-batch frontier and replay the committed
         * prefix (first_token + accepted drafts). The decode path re-captures
         * each position's hidden; seed per position, as the reference does.
         * (Stage B replaces this replay with transactional state rollback.) */
        s->checkpoint.len = saved_len;
        ok_state = spec_frontier_restore(&frontier, s);
        /* Stage B is the only rollback.  PULSAR_DSPARK_REPLAY used to restore
         * the Stage A transformer replay; it had no setter anywhere, so that
         * arm was unreachable and is deleted. */
        if (ok_state) {
            /* Stage B: transformer-free rollback. Roll only the recurrent
             * compressor/indexer pool state forward through the committed
             * prefix from the projections saved during the verify batch
             * (bit-identical: same update kernels, same rows, same order).
             * Raw KV + comp-cache rows are position-addressed and already
             * correct; counters are set by formula; s->logits was already
             * read from the fused batch's last committed row; drafter rows
             * seed from the batch capture. */
            ok_state = gpu_graph_dspark_compressor_rollforward(g, &e->model, &e->weights,
                                                               (uint32_t)saved_len,
                                                               (uint32_t)(1 + commit),
                                                               row0);
            if (ok_state) {
                s->checkpoint.len = saved_len + 1 + commit;
                for (int m = 0; ok_state && m <= commit; m++)
                    ok_state = dspark_seed_from_batch_row(s, row0 + (uint32_t)m);
            }
        }
    }
    spec_frontier_free(&frontier);
    if (!ok_state) {
        snprintf(err, errlen, "DSpark fused state update failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Emit first_token + accepted drafts. */
    accepted[n_accept++] = first_token;
    bool hit_eos = first_token == eos_token;
    for (int i = 0; i < commit && n_accept < accepted_cap && !hit_eos; i++) {
        accepted[n_accept++] = (int)pend[i];
        if (pend[i] == (int32_t)eos_token) hit_eos = true;
    }

    /* L155 guard: the committed frontier (saved_len + 1 + commit) must equal
     * what the caller receives.  EOS is already handled -- the ghost-token
     * guard above clamps commit at an accepted EOS, so no post-EOS draft is
     * ever committed (I first read this as a live leak; it is not).  The one
     * way the two can still differ is accepted_cap binding below 1 + commit,
     * which no current caller does (the server passes its array size).  Enforce
     * the invariant anyway, where both counts are known: rewind() clamps the
     * compressor frontier, replays the straddled ratio-4 group when the ring
     * covers it, and drops the carry and drafter window, which is right -- the
     * carry was conditioned on positions that no longer exist.  The server's
     * tripwire (server_sched.cpp) checks the same equality after every round. */
    const int exposed_end = saved_len + n_accept;
    const bool trimmed = exposed_end < s->checkpoint.len;
    if (trimmed) s->rewind(exposed_end);

    /* The carry IS the next base (already correctly distributed). Persist it
     * so the next generate_speculative call forwards it as batch position 0;
     * pre-draft the NEXT block conditioned on it. */
    const int next_base = carry_tok;
    s->spec.spec_carry_token = (int32_t)carry_tok;
    s->spec.spec_carry_valid = !hit_eos && !trimmed;
    s->spec.spec_carry_pos = (int32_t)s->checkpoint.len;
    s->spec.spec_carry_temp = temperature;
    s->spec.spec_carry_top_k = top_k;
    s->spec.spec_carry_top_p = top_p;
    s->spec.spec_carry_min_p = min_p;
    uint32_t n_draft = spec_cur_depth(s);   /* L107: session depth, not the static engine width */
    if (n_draft > 16u) n_draft = 16u;
    if (hit_eos || trimmed || next_base == eos_token || n_draft == 0 || s->spec.spec_quenched) {
        /* Quenched: don't draft the next chain — the carry persisted above is
         * still the correctly-distributed next base, which the next
         * generate_speculative call consumes before routing plain.  (After an
         * L155 trim there is no carry: the generation ended at the cap.) */
        if (dspark_stats && t0 > 0.0)
            fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d nodraft step_ms=%.1f\n",
                    n_batch, commit, (now_sec() - t0) * 1000.0);
        return n_accept;
    }

    if (defer_redraft) {
        /* L150: the batched lane drafts every bank in ONE pass after all the
         * walks (pulsar_session_spec_redraft_batch). Record what this bank's
         * redraft needs and leave the shadow with no pendings: a bank the batch
         * cannot serve simply takes a plain n=1 step next round. */
        pulsar_spec_drop_pendings(&s->spec);
        spec_redraft_req *q = &r->redraft;
        q->valid = main_x_ready;
        q->done = false;
        q->next_base = (int32_t)next_base;
        q->n_draft = n_draft;
        q->n_batch = n_batch;
        q->commit = commit;
        q->temperature = temperature;
        q->top_k = top_k;
        q->top_p = top_p;
        q->min_p = min_p;
        return n_accept;
    }
    const uint32_t keep = spec_round_redraft(s, next_base, main_x_ready,
                                             temperature, top_k, top_p, min_p, rng);
    /* t0 == 0 marks a lane whose forward ran outside this round (the batched
     * lane) -- step_ms would print the epoch, so skip the line there. */
    if (dspark_stats && t0 > 0.0)
        fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d pend=%u step_ms=%.1f\n",
                n_batch, commit, keep, (now_sec() - t0) * 1000.0);
    return n_accept;
}

static int pulsar_session_eval_speculative_fused(pulsar_session *s, int first_token,
                                              int max_tokens, int eos_token,
                                              float temperature, int top_k,
                                              float top_p, float min_p,
                                              uint64_t *rng,
                                              int *accepted, int accepted_cap,
                                              char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    const double t0 = dspark_stats ? now_sec() : 0.0;

    pulsar_spec_round r;
    {
        const int brc = spec_round_begin(s, first_token, max_tokens, accepted_cap,
                                         temperature, top_k, top_p, min_p, &r,
                                         err, errlen);
        if (brc != 0) return brc;
    }

    /* ONE batched forward: base decode + draft verify + anchor capture. */
    g->dspark_capture_batch_n = r.n_batch;
    g->spec_comp_save_n = r.n_batch;   /* Stage-B: save per-position comp projections */
    /* L149 phase 2: this lane's step carries exactly the round begun above */
    g->spec_compact_armed = g->spec_compact_acc_n > 0 && g->spec_compact_acc_ok;
    g->spec_compact_delta = g->spec_compact_acc_delta;
    g->spec_compact_acc_n = 0;
    g->spec_compact_acc_ok = false;
    bool ok = gpu_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                           &s->checkpoint,
                                           (uint32_t)r.saved_len, r.n_batch,
                                           r.K ? r.row_tops : NULL, NULL);
    g->dspark_capture_batch_n = 0;
    g->spec_comp_save_n = 0;
    if (ok && g->spec_compact_armed) {
        if (!gpu_graph_spec_compact_read(g, 0u, r.n_batch)) {
            fprintf(stderr, "pulsar: spec compact readback failed for %u rows -- refusing (L174)\n", r.n_batch);
            ok = false;
        }
    } else {
        g->spec_compact_rows = 0;
    }
    g->spec_compact_armed = false;
    if (!ok) {
        s->checkpoint.len = r.saved_len;
        (void)spec_frontier_restore(&r.frontier, s);
        spec_frontier_free(&r.frontier);
        snprintf(err, errlen, "DSpark fused verify failed");
        s->checkpoint_valid = false;
        return -1;
    }

    return spec_round_end(s, &r, first_token, eos_token,
                          temperature, top_k, top_p, min_p, rng,
                          spec_row_read_classic, g, 0u,
                          g->spec_compact_rows >= r.n_batch ? g->spec_compact_host : NULL,
                          false, t0, NULL, accepted, accepted_cap, err, errlen);
}

/* Speculative generation that OWNS sampling: draws the base token from the
 * request's filtered distribution (or forwards the carry left by the previous
 * call), runs the fused draft/verify step with exact sampled acceptance, and
 * leaves the next correctly-distributed base as the carry. temperature <= 0
 * degenerates to the greedy argmax-equality path (byte-identical to the old
 * eval_speculative_block behavior). Returns the number of tokens emitted. */
int pulsar_session::generate_speculative(float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens, int eos_token,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval: the spec loop
     * decodes and emits against the graph's scalar frontier counters, which
     * hold a cross-bank superset after a multiseq step. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative generate after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return 0;
    }
    int first;
    const bool carry_params_match =
        s->spec.spec_carry_temp == temperature && s->spec.spec_carry_top_k == top_k &&
        s->spec.spec_carry_top_p == top_p && s->spec.spec_carry_min_p == min_p;
    /* the carry is only valid at the exact position it was drawn at; any
     * session advance outside this path (sync, plain eval, tool injection)
     * means s->logits no longer matches the carry's source distribution */
    const bool carry_pos_match = s->spec.spec_carry_pos == (int32_t)s->checkpoint.len;
    if (s->spec.spec_carry_valid && carry_params_match && carry_pos_match) {
        first = (int)s->spec.spec_carry_token;
        s->spec.spec_carry_valid = false;
    } else {
        /* no carry, or params changed mid-stream (e.g. tool-call payloads
         * force greedy): redraw from the current distribution */
        s->spec.spec_carry_valid = false;
        first = sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k,
                                   top_p, min_p, rng, &s->sample_scratch);
        if (first < 0) {
            snprintf(err, errlen, "the sampler refused the live logits row");
            return -1;
        }
    }
    if (first == eos_token) {
        /* never forward EOS through the target (matches the old caller loops,
         * which broke before eval) */
        accepted[0] = first;
        return 1;
    }
    /* Yield-quenched requests run plain for their remainder — the same route
     * as a drafterless engine, chosen per request. The carry consumed above is
     * already correctly distributed, so this is a pure speed decision. */
    if (!s->engine->has_dspark() || s->spec.spec_quenched) {
        if (s->eval(first, err, errlen) != 0) return -1;
        accepted[0] = first;
        return 1;
    }
    return pulsar_session_eval_speculative_fused(s, first, max_tokens, eos_token,
                                              temperature, top_k, top_p, min_p, rng,
                                              accepted, accepted_cap, err, errlen);
}



int pulsar_session::eval_speculative_block(int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval (which the no-dspark
     * fallback below would otherwise hit one frame deeper). */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative block eval after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return -1;
    }
    if (!s->engine->has_dspark() || s->spec.spec_quenched) {
        if (s->eval(first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    /* The fused loop -- one batched forward/step plus transactional no-replay
     * rollback -- is the only loop.  It measured 16.4 vs 15.2 t/s against the
     * old Step1+verify loop with byte-identical deterministic output, and that
     * legacy loop is DELETED along with PULSAR_DSPARK_LEGACY_LOOP, which had no
     * setter anywhere and so could never have restored it. */
    /* an externally chosen first_token invalidates any pending carry */
    s->spec.spec_carry_valid = false;
    return pulsar_session_eval_speculative_fused(s, first_token, max_tokens, eos_token,
                                                 0.0f, 0, 1.0f, 0.0f, NULL,
                                                 accepted, accepted_cap, err, errlen);
}

/* ---- plan-34 inc 6: the batched-lane round API (pulsar.h) --------------- */

pulsar_spec_round *pulsar_spec_round_new(void) {
    pulsar_spec_round *r = (pulsar_spec_round *)xmalloc(sizeof(pulsar_spec_round));
    memset(r, 0, sizeof(*r));
    return r;
}

void pulsar_spec_round_free(pulsar_spec_round *r) {
    if (!r) return;
    free(r->redraft.qrows);   /* L150: the lazily-owned fallback rows */
    free(r);
}

/* The carry-or-sample head of generate_speculative, verbatim: forward the
 * carry when it is valid at these params and this position, else draw fresh
 * from the live logits. The caller owns the EOS short-circuit. */
int pulsar_session_spec_next_base(pulsar_session *s, float temperature,
                               int top_k, float top_p, float min_p,
                               uint64_t *rng) {
    int first;
    const bool carry_params_match =
        s->spec.spec_carry_temp == temperature && s->spec.spec_carry_top_k == top_k &&
        s->spec.spec_carry_top_p == top_p && s->spec.spec_carry_min_p == min_p;
    const bool carry_pos_match = s->spec.spec_carry_pos == (int32_t)s->checkpoint.len;
    if (s->spec.spec_carry_valid && carry_params_match && carry_pos_match) {
        first = (int)s->spec.spec_carry_token;
        s->spec.spec_carry_valid = false;
    } else {
        s->spec.spec_carry_valid = false;
        first = sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k,
                                   top_p, min_p, rng, &s->sample_scratch);
    }
    return first;
}

int pulsar_session_spec_round_begin(pulsar_session *s, pulsar_spec_round *r,
                                 int first_token, int max_tokens, int accepted_cap,
                                 float temperature, int top_k, float top_p,
                                 float min_p, char *err, size_t errlen) {
    return spec_round_begin(s, first_token, max_tokens, accepted_cap,
                            temperature, top_k, top_p, min_p, r, err, errlen);
}

uint32_t pulsar_spec_round_n_rows(const pulsar_spec_round *r) {
    return r->n_batch;
}

uint32_t pulsar_spec_round_fill_reqs(const pulsar_spec_round *r, uint32_t bank,
                                  int first_token, pulsar_multiseq_req *out) {
    for (uint32_t i = 0; i < r->n_batch; i++) {
        out[i].bank = bank;
        out[i].pos = r->saved_len + (int32_t)i;
        out[i].token = i == 0 ? first_token : (int)r->pend[i - 1];
    }
    return r->n_batch;
}

/** Row source over the server-held ALL_ROWS logits block.
 *
 * The batched lane runs one spec round per bank around ONE shared forward, so
 * a round reads its rows out of the shared block at an offset rather than
 * owning a logits buffer of its own. */
typedef struct {
    const float *rows;  ///< borrowed: the shared ALL_ROWS logits block
    uint32_t row0;      ///< first row belonging to this round
} spec_block_rows;

static bool spec_row_read_block(void *ud, uint32_t row, float *out) {
    const spec_block_rows *b = (const spec_block_rows *)ud;
    memcpy(out, b->rows + ((size_t)b->row0 + row) * PULSAR_N_VOCAB,
           (size_t)PULSAR_N_VOCAB * sizeof(float));
    return true;
}

/* L149 phase 2: row source over this step's device rows at a bank's offset
 * (the compact path never brought the full block to the host). */
typedef struct {
    pulsar_gpu_graph *g;
    uint32_t row0;
} spec_dev_rows;

static bool spec_row_read_dev(void *ud, uint32_t row, float *out) {
    const spec_dev_rows *d = (const spec_dev_rows *)ud;
    return gpu_graph_read_spec_logits_row(d->g, d->row0 + row, out);
}

static int spec_round_end_block(pulsar_session *s, pulsar_spec_round *r,
                                int first_token, int eos_token,
                                float temperature, int top_k, float top_p,
                                float min_p, uint64_t *rng,
                                const float *rows, uint32_t row0,
                                const int32_t *forced_truth,
                                int *accepted, int accepted_cap,
                                char *err, size_t errlen) {
    pulsar_gpu_graph *g = &s->graph;
    /* L149 phase 2: the step read the compact block instead of `rows`. Row
     * argmaxes come from its headers (the host tie rule, computed on device);
     * the walk builds from candidates; any row that needs its full logits
     * (no finite max, overflow, the s->logits refresh) is read from the
     * device, where this step's rows still sit. */
    if (g->spec_compact_rows >= row0 + r->n_batch && g->spec_compact_host) {
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
        for (uint32_t i = 0; i < r->K && i < 16u; i++) {
            const int32_t *h = g->spec_compact_host +
                               (size_t)(row0 + i) * PULSAR_DSPARK_PREFILTER_ROW_I32;
            if (h[1] >= 0) {
                r->row_tops[i] = h[1];
            } else {
                if (!gpu_graph_read_spec_logits_row(g, row0 + i, s->spec_row_scratch)) {
                    snprintf(err, errlen, "spec compact: row %u readback failed", row0 + i);
                    return -1;
                }
                r->row_tops[i] = (int)sample_argmax(s->spec_row_scratch, PULSAR_N_VOCAB);
            }
        }
        spec_dev_rows src = { g, row0 };
        return spec_round_end(s, r, first_token, eos_token,
                              temperature, top_k, top_p, min_p, rng,
                              spec_row_read_dev, &src, row0, g->spec_compact_host,
                              true /* L150: redraft deferred to the batch */,
                              0.0 /* t0: step_ms diagnostic reads 0 in this lane */,
                              forced_truth, accepted, accepted_cap, err, errlen);
    }
    /* Greedy walk consumes per-row argmaxes; the classic forward computes
     * them on-device, the shared block computes them here. Draft i is judged
     * against round-local row i (the row that PREDICTS it). */
    for (uint32_t i = 0; i < r->K && i < 16u; i++) {
        r->row_tops[i] = (int)sample_argmax(
                rows + ((size_t)row0 + i) * PULSAR_N_VOCAB, PULSAR_N_VOCAB);
    }
    spec_block_rows src = { rows, row0 };
    return spec_round_end(s, r, first_token, eos_token,
                          temperature, top_k, top_p, min_p, rng,
                          spec_row_read_block, &src, row0, NULL,
                          true /* L150: redraft deferred to the batch */,
                          0.0 /* t0: step_ms diagnostic reads 0 in this lane */,
                          forced_truth, accepted, accepted_cap, err, errlen);
}

int pulsar_session_spec_round_end(pulsar_session *s, pulsar_spec_round *r,
                               int first_token, int eos_token,
                               float temperature, int top_k, float top_p,
                               float min_p, uint64_t *rng,
                               const float *rows, uint32_t row0,
                               int *accepted, int accepted_cap,
                               char *err, size_t errlen) {
    return spec_round_end_block(s, r, first_token, eos_token, temperature, top_k, top_p,
                                min_p, rng, rows, row0, NULL, accepted, accepted_cap,
                                err, errlen);
}

int pulsar_session_spec_round_end_forced(pulsar_session *s, pulsar_spec_round *r,
                                      int first_token, int eos_token,
                                      float temperature, int top_k, float top_p,
                                      float min_p, uint64_t *rng,
                                      const float *rows, uint32_t row0,
                                      const int32_t *truth,
                                      int *accepted, int accepted_cap,
                                      char *err, size_t errlen) {
    if (!truth) {
        snprintf(err, errlen, "spec round_end_forced: no truth");
        return -1;
    }
    return spec_round_end_block(s, r, first_token, eos_token, temperature, top_k, top_p,
                                min_p, rng, rows, row0, truth, accepted, accepted_cap,
                                err, errlen);
}


static bool redraft_qrows_reserve(spec_redraft_req *q) {
    if (!q->qrows) q->qrows = (float *)xmalloc((size_t)16 * PULSAR_N_VOCAB * sizeof(float));
    return q->qrows != NULL;
}

/* L150: one redraft group -- the banks order[0..n_sel) whose rows fit the
 * M-neutral row budget together. Greedy banks come first in `order` (the
 * caller sorted them), so within the group they are the prefix. */
static int spec_redraft_group(pulsar_session *s, pulsar_spec_round **rounds,
                              const uint32_t *banks, uint64_t **rngs,
                              const int *order, int n_sel,
                              char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    const void *dmap = e->dspark_model.map;
    const uint64_t dsize = e->dspark_model.size;
    int n_g = 0;
    for (int j = 0; j < n_sel; j++) if (!rounds[order[j]]->redraft.sample_drafts) n_g++;
    const int n_s = n_sel - n_g;
    /* rows */
    int32_t draft_ids[PULSAR_SPEC_LOGITS_ROWS];
    uint32_t row_bank[PULSAR_SPEC_LOGITS_ROWS];
    uint32_t bank_n_raw[PULSAR_MSEQ_MAX][3];
    uint32_t bank_n_draft[PULSAR_MSEQ_MAX];
    int32_t base_row[PULSAR_DSPARK_BANKS_MAX] = {0};
    uint32_t n_rows = 0, max_draft_g = 0, max_draft_s = 0;
    for (int j = 0; j < n_sel; j++) {
        spec_redraft_req *q = &rounds[order[j]]->redraft;
        const uint32_t bank = banks[order[j]];
        if (bank >= g->banks.n_banks || n_rows + q->n_draft > PULSAR_SPEC_LOGITS_ROWS) {
            snprintf(err, errlen, "redraft batch: rows exceed the drafter budget");
            return -1;
        }
        base_row[j] = (int32_t)n_rows;
        for (uint32_t li = 0; li < 3; li++) bank_n_raw[bank][li] = g->ms_dspark_n_raw[bank][li];
        bank_n_draft[bank] = q->n_draft;
        for (uint32_t k = 0; k < q->n_draft; k++) {
            draft_ids[n_rows] = k == 0 ? q->next_base : PULSAR_DSPARK_NOISE_TOKEN_ID;
            row_bank[n_rows] = bank;
            n_rows++;
        }
        if (j < n_g) { if (q->n_draft > max_draft_g) max_draft_g = q->n_draft; }
        else         { if (q->n_draft > max_draft_s) max_draft_s = q->n_draft; }
        q->refined[0] = q->next_base;
        for (int k = 1; k < 17; k++) q->refined[k] = 0;
        for (int k = 0; k < 16; k++) { q->qn[k] = 0; q->q_drawn[k] = 0.0f; q->conf[k] = -1.0f; }
        q->have_conf = false;
        q->keep = q->n_draft;
    }
    /* the chain reads base rows (base_row[b] + pos) for pos < the group's max
     * depth: a shallower bank's extra positions read the next bank's rows or
     * the slab tail -- harmless, discarded -- but must stay inside the block */
    if ((uint32_t)base_row[n_sel - 1] + (n_g == n_sel ? max_draft_g : max_draft_s) >
        PULSAR_SPEC_LOGITS_ROWS) {
        snprintf(err, errlen, "redraft batch: chain rows exceed the block");
        return -1;
    }

    if (!gpu_graph_dspark_draft_forward_banks(g, &e->model, &e->weights, &e->dspark_model, w,
                                              g->spec_logits, draft_ids, n_rows,
                                              (uint32_t)n_sel, row_bank, bank_n_raw, bank_n_draft)) {
        snprintf(err, errlen, "redraft batch: draft forward failed");
        return -1;
    }
    if (pulsar_gpu_tensor_bytes(g->dspark_markov_logits) < (uint64_t)n_sel * vocab_bytes) {
        snprintf(err, errlen, "redraft batch: markov scratch too small");
        return -1;
    }
    /* device meta: base rows [0, MAX), sampled prev tokens [MAX, 2 MAX) */
    int32_t meta[2 * PULSAR_DSPARK_BANKS_MAX];
    memset(meta, 0, sizeof(meta));
    for (int j = 0; j < n_sel; j++) meta[j] = base_row[j];
    /* seed the chain feed: ids[j][0] = next_base */
    int32_t ids_seed[PULSAR_DSPARK_BANKS_MAX * 17];
    memset(ids_seed, 0, sizeof(ids_seed));
    for (int j = 0; j < n_sel; j++) ids_seed[j * 17] = rounds[order[j]]->redraft.next_base;
    if (!pulsar_gpu_tensor_write(g->dspark_bank_meta, 0, meta, sizeof(meta)) ||
        !pulsar_gpu_tensor_write(g->dspark_refined_ids, 0, ids_seed,
                                 (uint64_t)n_sel * 17 * sizeof(int32_t))) {
        snprintf(err, errlen, "redraft batch: meta upload failed");
        return -1;
    }
    const uint64_t spec_row_bytes = (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    const int w1_bf16 = w->markov_w1->type == PULSAR_TENSOR_BF16;
    const int w2_bf16 = w->markov_w2->type == PULSAR_TENSOR_BF16;

    /* greedy banks: the whole chain on device, ids read back once */
    if (n_g > 0) {
        if (!pulsar_gpu_dspark_markov_chain_banks_model(
                g->dspark_markov_logits, g->dspark_refined_ids, 17u,
                g->spec_logits, spec_row_bytes, g->dspark_bank_meta,
                dmap, dsize, w->markov_w1->abs_offset, w->markov_w2->abs_offset,
                (uint32_t)n_g, max_draft_g, vocab_size, embed_dim, w1_bf16, w2_bf16)) {
            snprintf(err, errlen, "redraft batch: markov chain failed");
            return -1;
        }
    }
    /* sampled banks: one step per position across banks, host draws between */
    if (n_s > 0) {
        pulsar_gpu_tensor *refined_s = pulsar_gpu_tensor_view(
            g->dspark_markov_logits, (uint64_t)n_g * vocab_bytes, (uint64_t)n_s * vocab_bytes);
        pulsar_gpu_tensor *ids_s = pulsar_gpu_tensor_view(
            g->dspark_refined_ids, (uint64_t)n_g * 17 * sizeof(int32_t), (uint64_t)n_s * 17 * sizeof(int32_t));
        pulsar_gpu_tensor *base_s = pulsar_gpu_tensor_view(
            g->dspark_bank_meta, (uint64_t)n_g * sizeof(int32_t), (uint64_t)n_s * sizeof(int32_t));
        pulsar_gpu_tensor *prev_s = pulsar_gpu_tensor_view(
            g->dspark_bank_meta, (uint64_t)PULSAR_DSPARK_BANKS_MAX * sizeof(int32_t),
            (uint64_t)n_s * sizeof(int32_t));
        bool ok = refined_s && ids_s && base_s && prev_s;
        /* the most permissive floor across the sampled banks: a superset for each */
        float delta = 0.0f;
        bool all_sparse = true;
        for (int j = n_g; j < n_sel && ok; j++) {
            const spec_redraft_req *q = &rounds[order[j]]->redraft;
            const bool sparse = q->top_k <= 0 && q->top_p == 1.0f &&
                                q->min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && q->min_p <= 1.0f &&
                                vocab_size <= PULSAR_SAMPLE_SPARSE_VOCAB_MAX;
            if (!sparse) { all_sparse = false; continue; }
            const float d = (float)((double)q->temperature * (log((double)q->min_p) - 1e-3));
            if (j == n_g || d < delta) delta = d;
        }
        int32_t *sel = ok ? (int32_t *)xmalloc((size_t)n_s * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t)) : NULL;
        int32_t prev[PULSAR_DSPARK_BANKS_MAX];
        for (uint32_t pos = 0; ok && pos < max_draft_s; pos++) {
            for (int j = n_g; j < n_sel; j++) prev[j - n_g] = rounds[order[j]]->redraft.refined[pos];
            ok = pulsar_gpu_tensor_write(prev_s, 0, prev, (uint64_t)n_s * sizeof(int32_t)) &&
                 pulsar_gpu_dspark_markov_step_banks_model(
                     refined_s, ids_s, 17u, g->spec_logits, spec_row_bytes, base_s, prev_s,
                     dmap, dsize, w->markov_w1->abs_offset, w->markov_w2->abs_offset,
                     (uint32_t)n_s, pos, vocab_size, embed_dim, w1_bf16, w2_bf16);
            if (ok && all_sparse &&
                !(pulsar_gpu_minp_prefilter_rows(g->dspark_prefilter_sel, refined_s, 0,
                                                 (uint32_t)n_s, vocab_size, vocab_size,
                                                 delta, PULSAR_DSPARK_PREFILTER_CAP) &&
                  pulsar_gpu_tensor_read(g->dspark_prefilter_sel, 0, sel,
                                         (uint64_t)n_s * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t)))) {
                /* every sampled round here is in the sparse contract: a device
                 * failure is a failure, not a full-row read (L190 D3) */
                fprintf(stderr, "pulsar: dspark batched min-p prefilter failed at draft position %u "
                                "-- refusing the batch (no full-row fallback; L190)\n", pos);
                ok = false;
            }
            for (int j = n_g; j < n_sel && ok; j++) {
                spec_redraft_req *q = &rounds[order[j]]->redraft;
                if (pos >= q->n_draft) continue;
                pulsar_sample_dist qd;
                bool built = false;
                if (all_sparse) {
                    const int32_t *h = sel + (size_t)(j - n_g) * PULSAR_DSPARK_PREFILTER_ROW_I32;
                    const uint32_t n_c = (uint32_t)h[0];
                    /* n_c > cap: more candidates above the floor than the
                     * compact block holds -- the one genuine ineligibility;
                     * the full row is read below.  Anything else the build
                     * refuses is a broken device result. */
                    if (n_c <= PULSAR_DSPARK_PREFILTER_CAP) {
                        float max_logit;
                        memcpy(&max_logit, &h[2], sizeof(max_logit));
                        if (n_c == 0 ||
                            !pulsar_sample_dist_build_prefiltered(
                                h + 3, (const float *)(h + 3 + PULSAR_DSPARK_PREFILTER_CAP), n_c,
                                max_logit, q->temperature, q->min_p, &s->sample_scratch, &qd)) {
                            fprintf(stderr, "pulsar: dspark batched min-p prefilter handed %u "
                                            "candidates at draft position %u and the build refused "
                                            "them -- refusing the batch (L190)\n", n_c, pos);
                            ok = false;
                            break;
                        }
                        built = true;
                    }
                }
                if (built && qd.n <= PULSAR_DSPARK_QDIST_CAP) {
                    q->qn[pos] = qd.n;
                    memcpy(q->qids[pos], qd.ids, (size_t)qd.n * sizeof(int32_t));
                    memcpy(q->qprobs[pos], qd.probs, (size_t)qd.n * sizeof(float));
                } else {
                    /* full row: the residual will need it, and the build may too */
                    if (!redraft_qrows_reserve(q)) { ok = false; break; }
                    float *row = q->qrows + (size_t)pos * PULSAR_N_VOCAB;
                    if (!pulsar_gpu_tensor_read(refined_s, (uint64_t)(j - n_g) * vocab_bytes, row, vocab_bytes)) {
                        if (built) pulsar_sample_dist_free(&qd);
                        ok = false; break;
                    }
                    if (!built &&
                        !pulsar_sample_dist_build(row, PULSAR_N_VOCAB, q->temperature, q->top_k,
                                                  q->top_p, q->min_p, &s->sample_scratch, &qd)) {
                        ok = false; break;
                    }
                    q->qn[pos] = 0;
                }
                const int drawn = pulsar_sample_dist_draw(&qd, rngs[order[j]]);
                q->refined[pos + 1] = (int32_t)drawn;
                q->q_drawn[pos] = pulsar_sample_dist_prob(&qd, drawn);
                pulsar_sample_dist_free(&qd);
            }
        }
        free(sel);
        pulsar_gpu_tensor_free(refined_s);
        pulsar_gpu_tensor_free(ids_s);
        pulsar_gpu_tensor_free(base_s);
        pulsar_gpu_tensor_free(prev_s);
        if (!ok) {
            snprintf(err, errlen, "redraft batch: sampled markov loop failed");
            return -1;
        }
    }
    /* greedy ids come back in one read */
    if (n_g > 0) {
        int32_t ids_all[PULSAR_DSPARK_BANKS_MAX * 17];
        if (!pulsar_gpu_tensor_read(g->dspark_refined_ids, 0, ids_all, (uint64_t)n_sel * 17 * sizeof(int32_t))) {
            snprintf(err, errlen, "redraft batch: ids readback failed");
            return -1;
        }
        for (int j = 0; j < n_sel; j++) {
            spec_redraft_req *q = &rounds[order[j]]->redraft;
            for (uint32_t k = 1; k <= q->n_draft; k++) {
                if (j < n_g) q->refined[k] = ids_all[j * 17 + k];
            }
        }
    }
    /* confidence over every row: tok row (b,k) = refined_b[k], as the
     * single-bank path feeds it (tok_dev <- refined[0..n_draft)) */
    const float tau = dspark_conf_sched_tau();
    if (tau > 0.0f) {
        int32_t toks[PULSAR_SPEC_LOGITS_ROWS];
        float confs[PULSAR_SPEC_LOGITS_ROWS];
        for (int j = 0; j < n_sel; j++) {
            const spec_redraft_req *q = &rounds[order[j]]->redraft;
            for (uint32_t k = 0; k < q->n_draft; k++) toks[base_row[j] + k] = q->refined[k];
        }
        if (pulsar_gpu_tensor_bytes(g->dspark_conf_tokens) < (uint64_t)n_rows * sizeof(int32_t) ||
            pulsar_gpu_tensor_bytes(g->dspark_conf_scores) < (uint64_t)n_rows * sizeof(float) ||
            !pulsar_gpu_tensor_write(g->dspark_conf_tokens, 0, toks, (uint64_t)n_rows * sizeof(int32_t)) ||
            !pulsar_gpu_dspark_confidence_score_model(g->dspark_conf_scores, g->batch_ffn_cur,
                                                     g->dspark_conf_tokens, dmap, dsize,
                                                     w->markov_w1->abs_offset,
                                                     w->confidence_proj->abs_offset,
                                                     n_rows, PULSAR_N_EMBD, embed_dim, vocab_size,
                                                     w1_bf16,
                                                     w->confidence_proj->type == PULSAR_TENSOR_BF16) ||
            !pulsar_gpu_tensor_read(g->dspark_conf_scores, 0, confs, (uint64_t)n_rows * sizeof(float))) {
            snprintf(err, errlen, "redraft batch: confidence failed");
            return -1;
        }
        for (int j = 0; j < n_sel; j++) {
            spec_redraft_req *q = &rounds[order[j]]->redraft;
            q->have_conf = true;
            for (uint32_t k = 0; k < q->n_draft; k++) q->conf[k] = confs[base_row[j] + k];
            if (tau > 0.0f) {
                uint32_t k = 0;
                while (k < q->n_draft && q->conf[k] >= tau) k++;
                q->keep = k;
            }
        }
    }
    return 0;
}

/* =====================================================================
 * L150: batched redraft -- one drafter pass for every bank of the tick.
 *
 * The batched lane's round_end records each bank's redraft request and
 * defers. This runs the draft forward over all requesting banks' rows at once
 * (gpu_graph_dspark_draft_forward_banks: rope/store/visibility positions per
 * row, bank-major rings), the markov refine with one weight stream per
 * position across banks, and the confidence head over every row, then leaves
 * each bank's results in its round. It NEVER switches banks -- it reads the
 * banks' ring counters from the saved carries (g->ms_dspark_n_raw) and the
 * rings from the slabs -- so the server's bank_switch bookkeeping stays the
 * single authority; pulsar_session_spec_redraft_commit stamps a bank's shadow
 * under the server's switch.
 *
 * Byte-exact per bank with the serialized single-bank redraft (the kernels are
 * exact by construction; each bank draws from its own rng in the same order);
 * the greedy identity gate pins it. Diagnostics of the single-bank path
 * (DSPARK_Q lines, the on-policy and ring dumps, dspark_dump_step) are not
 * emitted here: they are dev instruments of the classic lane. */
int pulsar_session_spec_redraft_batch(pulsar_session *s, pulsar_spec_round **rounds,
                                      const uint32_t *banks, uint64_t **rngs, int n,
                                      char *err, size_t errlen) {
    pulsar_gpu_graph *g = &s->graph;
    if (!rounds || !banks || !rngs || n <= 0) return 0;
    if (!g->dspark_markov_logits || !g->dspark_refined_ids ||
        !g->dspark_bank_meta || !g->dspark_conf_scores || !g->dspark_conf_tokens ||
        !g->dspark_prefilter_sel || g->banks.n_banks == 0) {
        snprintf(err, errlen, "redraft batch: drafter scratch missing");
        return -1;
    }
    /* greedy banks first, then sampled, so each group is contiguous in rows
     * and in the bank arrays */
    int order[PULSAR_DSPARK_BANKS_MAX];
    int n_sel = 0, n_g = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n && n_sel < (int)PULSAR_DSPARK_BANKS_MAX; i++) {
            spec_redraft_req *q = &rounds[i]->redraft;
            if (!q->valid || q->n_draft == 0) continue;
            const bool sampled = q->temperature > 0.0f;   /* vocab pinned at load; see the drafting loop */
            if ((pass == 0) != !sampled) continue;
            q->sample_drafts = sampled;
            order[n_sel++] = i;
            if (!sampled) n_g++;
        }
    /* The drafter's batch buffers hold PULSAR_DSPARK_BANKS_MAX banks; the
     * scheduler can pass more live rounds under an operator PULSAR_MSEQ_BANKS
     * pin (auto-size stops at 8).  A round past the cap never gets done = true
     * and takes base-only steps.  That is a per-ROUND condition, so it is said
     * once per affected round object (the flag survives round_begin's reset,
     * like qrows) -- not once per process, which after the first hid every
     * later request it hit (L177 -> L190 C2). */
    for (int i = 0; i < n; i++) {
        spec_redraft_req *q = &rounds[i]->redraft;
        if (!q->valid || q->n_draft == 0 || q->cap_said) continue;
        bool picked = false;
        for (int j = 0; j < n_sel && !picked; j++) picked = order[j] == i;
        if (picked) continue;
        q->cap_said = true;
        fprintf(stderr, "pulsar: redraft batch: %d live rounds but the drafter batch holds %u banks -- "
                        "round %d drafts nothing while that many are live (pin fewer banks)\n",
                n, (unsigned)PULSAR_DSPARK_BANKS_MAX, i);
    }
    if (n_sel == 0) return 0;
    (void)n_g;

    /* groups: consecutive banks whose rows fit the widest decode batch the
     * M-independent kernels take (PULSAR_GPU_MNEUTRAL_ROWS_MAX; the forward
     * declares its rows decode), at most PULSAR_DSPARK_BANKS_MAX banks each;
     * production (3 banks x <=4 drafts) is one group. A bank whose depth alone
     * exceeds the cap is refused. */
    for (int gs = 0; gs < n_sel;) {
        int ge = gs;
        uint32_t rows = 0;
        while (ge < n_sel && ge - gs < (int)PULSAR_DSPARK_BANKS_MAX &&
               rows + rounds[order[ge]]->redraft.n_draft <= PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
            rows += rounds[order[ge]]->redraft.n_draft;
            ge++;
        }
        if (ge == gs) {
            snprintf(err, errlen, "redraft batch: a bank's depth exceeds the row budget");
            return -1;
        }
        const int rc = spec_redraft_group(s, rounds, banks, rngs, order + gs, ge - gs, err, errlen);
        if (rc != 0) return rc;
        gs = ge;
    }
    for (int j = 0; j < n_sel; j++) rounds[order[j]]->redraft.done = true;
    return 0;
}

/* L150: stamp the CURRENT bank's shadow from its batched redraft result. The
 * caller has switched to the round's bank and saves it afterwards. Mirrors the
 * tail of spec_round_redraft field for field; the greedy identity gate is what
 * keeps the two in step. */
void pulsar_session_spec_redraft_commit(pulsar_session *s, pulsar_spec_round *r) {
    spec_redraft_req *q = &r->redraft;
    if (!q->valid || !q->done) return;
    s->spec.dspark_pending_base = q->next_base;
    s->spec.dspark_chain_unharvested = false;
    s->spec.dspark_n_pending = q->keep;
    for (uint32_t i = 0; i < q->keep; i++) s->spec.dspark_pending[i] = q->refined[i + 1];
    s->spec.dspark_pending_sampled = q->sample_drafts;
    s->spec.dspark_pending_pos = (int32_t)s->checkpoint.len;
    s->spec.dspark_pending_temp = q->temperature;
    s->spec.dspark_pending_top_k = q->top_k;
    s->spec.dspark_pending_top_p = q->top_p;
    s->spec.dspark_pending_min_p = q->min_p;
    for (uint32_t i = 0; i < q->keep; i++)
        s->spec.dspark_pending_conf[i] = q->have_conf ? q->conf[i] : -1.0f;
    if (q->sample_drafts) {
        bool need_rows = false;
        for (uint32_t i = 0; i < q->n_draft; i++) {
            s->spec.dspark_pending_q[i] = q->q_drawn[i];
            s->spec.dspark_pending_qn[i] = q->qn[i];
            if (q->qn[i] > 0) {
                memcpy(s->spec.dspark_pending_qids[i], q->qids[i], (size_t)q->qn[i] * sizeof(int32_t));
                memcpy(s->spec.dspark_pending_qprobs[i], q->qprobs[i], (size_t)q->qn[i] * sizeof(float));
            } else {
                need_rows = true;
            }
        }
        if (need_rows && q->qrows) {
            const uint32_t need = q->n_draft * PULSAR_N_VOCAB;
            if (s->dspark_pending_qrows_cap < need) {
                free(s->dspark_pending_qrows);
                s->dspark_pending_qrows = (float *)xmalloc((size_t)need * sizeof(float));
                s->dspark_pending_qrows_cap = need;
            }
            for (uint32_t i = 0; i < q->n_draft; i++)
                if (q->qn[i] == 0)
                    memcpy(s->dspark_pending_qrows + (size_t)i * PULSAR_N_VOCAB,
                           q->qrows + (size_t)i * PULSAR_N_VOCAB,
                           (size_t)PULSAR_N_VOCAB * sizeof(float));
        }
    }
    q->done = false;
    q->valid = false;
}

int pulsar_session_spec_redraft_peek(const pulsar_spec_round *r, int32_t ids[17], float conf[16],
                                     uint32_t *n_draft, uint32_t *keep, int *sampled) {
    if (!r || !r->redraft.valid || !r->redraft.done) return 0;
    const spec_redraft_req *q = &r->redraft;
    if (ids) memcpy(ids, q->refined, sizeof(q->refined));
    if (conf) memcpy(conf, q->conf, sizeof(q->conf));
    if (n_draft) *n_draft = q->n_draft;
    if (keep) *keep = q->keep;
    if (sampled) *sampled = q->sample_drafts ? 1 : 0;
    return 1;
}

uint32_t pulsar_session_bank_pending_confs(const pulsar_session *s, uint32_t bank,
                                        float out[16]) {
    /* L108 P2: the live session's pending count may still be an
     * in-flight chain; complete it before peeking. Logically a lazy
     * read-completion, not observable-state mutation. */
    pulsar_session_spec_chain_harvest((pulsar_session *)s);

    if (!s || !s->bank_carry || bank >= s->bank_carry_n) return 0;
    const pulsar_bank_carry *c = &s->bank_carry[bank];
    if (!c->valid) return 0;
    uint32_t n = c->spec.dspark_n_pending;
    if (n > 16u) n = 16u;
    for (uint32_t i = 0; i < n; i++) out[i] = c->spec.dspark_pending_conf[i];
    return n;
}

int pulsar_spec_round_saved_len(const pulsar_spec_round *r) {
    return r ? r->saved_len : -1;
}

uint32_t pulsar_session_spec_next_rows_max(const pulsar_session *s) {
    /* Upper bound on the next round's n_batch: begin's guards (base-token,
     * position, params, caps) only ever TRIM K from dspark_n_pending, so
     * 1 + pending is safe to budget against before consuming anything. */
    uint32_t k = s->spec.dspark_n_pending;
    if (k > 16u) k = 16u;
    return 1u + k;
}

void pulsar_session_spec_arm_capture(pulsar_session *s, uint32_t n_rows) {
    pulsar_gpu_graph *g = &s->graph;
    g->dspark_capture_batch_n = n_rows;
    g->spec_comp_save_n = n_rows;
    /* L149 phase 2: arm (n_rows > 0) the compact verify read from the rounds
     * begun since the last step; disarm and reset the accumulator after it. */
    if (n_rows > 0) {
        g->spec_compact_armed = g->spec_compact_acc_n > 0 && g->spec_compact_acc_ok;
        g->spec_compact_delta = g->spec_compact_acc_delta;
    } else {
        g->spec_compact_armed = false;
        g->spec_compact_acc_n = 0;
        g->spec_compact_acc_ok = false;
    }
}

void pulsar_session_spec_round_abort(pulsar_session *s, pulsar_spec_round *r) {
    /* Mirror the fused loop's forward-failure branch: drop the pushed rows,
     * restore the frontier, release the snapshot. */
    s->checkpoint.len = r->saved_len;
    (void)spec_frontier_restore(&r->frontier, s);
    spec_frontier_free(&r->frontier);
    s->checkpoint_valid = false;
}

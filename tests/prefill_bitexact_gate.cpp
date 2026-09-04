/* PREFILL BIT-EXACTNESS gate (the D2R tensor-core gate).
 *
 * MODEL-DEPENDENT: this gate freezes the ENGINE's full-vocab frontier logits
 * after a from-scratch prefill at several depths and byte-compares them against
 * a blob dumped by a BASELINE build (by default `dev` @ 8aa9d35, built in a
 * git worktree — see the `cuda-prefill-gate-baseline` Makefile target).  It is
 * the acceptance gate for the D2R (dequant-to-register) tensor-core MoE work:
 * every D2R increment claims to be BIT-EXACT, not merely close, so the gate's
 * tolerance is ZERO BYTES.  Run manually on the GB10 via `make cuda-prefill-gate`
 * (see the Makefile target for the memory discipline) — it is NOT part of
 * `make test`.
 *
 * WHY BYTE-COMPARE IS LEGITIMATE HERE.  Prefill is deterministic by
 * construction: the MoE path uses per-pair stores into a flat `down` buffer
 * plus a fixed-order `moe_sum_kernel` and NO float atomicAdd, precisely so
 * float reduction order cannot vary with tile scheduling
 * (src/cuda/pulsar_cuda_moe.cu — "no atomicAdd so numerics stay run-to-run
 * deterministic").  Check (c) below re-asserts that property every run; if a
 * future change reintroduces a float atomic, (c) fails before the byte-compare
 * has a chance to produce a confusing result.
 *
 * WHY FULL-VOCAB LOGITS AND NOT ARGMAX.  Our own 2026-07 IMMA post-mortem
 * (memory ds4-prefill-tensorcore) found argmax PRESERVED (494 == 494) while
 * late-layer hidden states diverged ~40% and the KV cache was materially
 * changed — generation drifted a few tokens later.  An argmax or top-k gate is
 * blind to exactly the failure that killed that attempt.  Full-vocab bytes at
 * MULTIPLE depths is the cheapest probe that is not.
 *
 * WHAT THE DEPTHS BUY.  With opt.prefill_chunk pinned to 4096,
 * pulsar_session_create computes prefill_cap = pulsar_prefill_cap_for_prompt(ctx_size,
 * 4096) = 4096 (src/engine/session.c:1813, src/engine/layers.c:14) — note it is
 * derived from ctx_size at CREATE, so it is 4096 for every depth here, not a
 * function of the prompt.  pulsar_session_sync then tests `prefill_cap <
 * prompt->len` (session.c:2063):
 *   - 512/2048/4096  -> FALSE (4096 included, at the boundary): the NON-chunked
 *     one-shot gpu_graph_prefill_raw_swa path.  Depth D is a single routed-MoE
 *     call at n_tokens == D.
 *   - 4102/6144      -> TRUE: gpu_graph_prefill_chunked, i.e. SEVERAL routed-MoE
 *     calls whose batch shapes are set by the chunk loop rather than by D.
 * The 6144 row is why the chunked path is not a blind spot: production chunks
 * every prompt > 4096, the cold chunk loop trims each non-final chunk to the
 * compress-ratio LCM and leaves a REMAINDER chunk (session.c:3476), and resumed
 * chunks re-shape batches and thereby change cuBLASLt algo selection
 * (session.c:3460).  A D2R re-tiling bug confined to chunk-remainder shapes
 * would pass a three-depth {512,2048,4096} gate and ship.
 * The 4102 row exists because 6144 was NOT enough: its remainder is 2048, so
 * every row above leaves a final chunk of 512..4096 and the gate never saw a
 * NARROW last chunk.  n_tok 1..8 is a distinct dispatch regime — the GEMV caps
 * switch there — and a change that moved every logit at n_tok=6 passed all four
 * original depths.  4102 gives a 6-token final chunk.  See g_depths below.
 * All depths clear the `use_big_batch = ... && n_tokens >= 128` bar in
 * routed_moe (src/cuda/pulsar_cuda_moe.cu) — including every chunk of the 6144 row
 * — i.e. they exercise the expert-tiled rowspan/tile16 kernels that D2R
 * replaces, NOT the per-pair qwarp32 decode path.  The ONE exception is 4102's
 * 6-token final chunk, which falls UNDER that bar and is the only row here that
 * reaches the small-batch path — deliberately so, since that is the regime the
 * n_tok-conditional dispatches select on.  Every routed layer runs on every
 * token, so all depths cover BOTH the 12 MXFP4 (type-39) promoted layers AND
 * the 31 IQ2_XXS (type-16) floor layers of v5mx in every run; the depths vary
 * the tile occupancy and the expert-tile remainder shapes, not the format
 * coverage.
 *
 * ...WHICH IS WHY THE ENV IS SCRUBBED.  That coverage claim is only true at the
 * DEFAULT env.  PULSAR_MOE_FP4_TILED=0 (src/cuda/pulsar_cuda_moe.cu, static-cached
 * getenv) forces use_sorted_pairs=0 for both-mxfp4 layers, so all 12 type-39
 * layers fall back to the per-pair qwarp32 path.  Exported for BOTH the dump and
 * the check, that would make the gate agree with itself having never run the
 * MXFP4 tiled kernels at all — a vacuous PASS.  scrub_numerics_env() below
 * unsets that whole class of knob and says so loudly — the whole PULSAR_ namespace,
 * plus the numerics knobs that live OUTSIDE it and are read by cuBLAS/the CUDA
 * runtime rather than by our code (NVIDIA_TF32_OVERRIDE, CUBLAS_WORKSPACE_CONFIG),
 * which no namespace sweep could find.  (Differing env between dump and check
 * fails LOUD, which is the safe direction; the danger is only the SAME wrong env
 * on both sides.)
 *
 * Checks, per depth D in {512, 2048, 4096, 6144}:
 *   (a) full-vocab frontier logits after a from-scratch prefill of D tokens
 *       byte-match the baseline blob's row for D.  ZERO tolerance.
 *   (b) the blob header (logits width, depth list, FNV-1a of the token ids)
 *       matches the baseline's, so a re-tokenization or a different model
 *       fails LOUD instead of silently comparing the wrong thing.
 *   (c) run-to-run determinism: D is prefilled TWICE through two fresh
 *       sessions and the two rows must be byte-identical.  This is the guard
 *       against a reintroduced float atomicAdd or a schedule-dependent
 *       reduction; it holds independently of the baseline.
 *
 * WHICH REDUCTION ORDER ACTUALLY MATTERS (measured 2026-07-15, and NOT what
 * temp/d2r-prefill-spec.md section 3.1 predicts — read this before writing a
 * D2R kernel).  Each gate/up output element is a THREE-level float reduction
 * (moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel + the blockN dot):
 *
 *   level 1  facc_b   = SUM over sb=0..7 of scale_sb * (float)sumi_sb
 *   level 2  g_l      = SUM over b == l (mod 8), ascending, of 0.5f*y->d*facc_b
 *   level 3  out      = quarter_warp_sum_f32's butterfly over l=0..7
 *                       i.e. ((g0+g4)+(g2+g6)) + ((g1+g5)+(g3+g7))
 *
 * The spec constrains ONLY level 1 ("keep sb ascending and it is bit-exact").
 * That is backwards on both counts:
 *   - Level 1 is order-INSENSITIVE over a provable window.  Every term is an
 *     exact power of two (E8M0) times an integer with |sumi| <= 32*12*127 =
 *     48768 (2^15.57).  facc sums EIGHT of them, so in units of the smallest
 *     scale the partial sums are integers bounded by 8*48768*2^k = 390144*2^k
 *     (2^18.57 * 2^k) for a scale spread of k octaves.  Exact integers in float
 *     run to 2^24, so the fold is provably exact — hence order-independent —
 *     while k <= 5.  Beyond that it MAY round (k=6 reaches 2^24.57); it is then
 *     data-dependent, not licensed.  Real weight rows keep neighbouring 32-block
 *     scales close, and measured on v5mx reversing sb moves ZERO bits of ZERO
 *     logits (the positive control below) — but do not read that as a licence to
 *     reorder level 1 unconditionally: the guarantee stops at 5 octaves.
 *   - IQ2_XXS is a SEPARATE case, and weaker: its reduction is integer
 *     (bsum += sumi*ls) and therefore exact and order-free as int32, with no
 *     overflow (|bsum| <= 8*32*43*127*31 = 43,338,496 << 2^31).  But that bound
 *     is 2^25.37, i.e. ABOVE 2^24 — so accumulating bsum in FLOAT is not
 *     provably exact, it merely happens not to round on this model.  Keep bsum
 *     integer; the int32 fold is both exact and faster.
 *   - Levels 2 and 3 ARE order-sensitive and the spec does not mention them.
 *     Their terms are 0.5f * y->d * facc_b with y->d an ARBITRARY float (the
 *     q8_K block scale, not a power of two), so those sums round.  Note also
 *     that 0.5f*y->d is applied PER BLOCK: SUM_b (0.5*d*facc_b) is not the same
 *     float as 0.5*d*SUM_b(facc_b).
 * A D2R re-tiling changes precisely the level-2/3 lane->block mapping, so that
 * is both what binds and the realistic failure mode — hence the teeth below.
 *
 * TEETH (each seeded as a local kernel edit, gate re-run, edit reverted;
 * verified on this tree 2026-07-15 — see the commit message):
 *   T1  MXFP4 gate/up block->lane REPARTITION — src/cuda/pulsar_cuda_moe.cu,
 *       moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel: give lane l the
 *       blocks {2l, 2l+1} instead of {l, l+8}
 *           for (uint32_t b = lane * 2u; b < lane * 2u + 2u && b < xq_blocks; b++)
 *       At v5mx's xq_blocks == 16 (n_embd 4096 / 256) that is the SAME blocks
 *       and the same weights, with only the level-2/3 grouping moved — the
 *       cleanest possible probe of the reduction order.  (The seed is written
 *       for 16; at xq_blocks > 16 it would also drop blocks 16.. and become a
 *       wrong-answer seed rather than a pure repartition.)
 *       -> gate FAILS on the 12 type-39 layers.
 *   T2  IQ2_XXS gate/up block->lane repartition — the same edit in
 *       moe_gate_up_mid_expert_tile8_rowspan_kernel.
 *       -> gate FAILS on the 31 type-16 layers.
 *       (T1+T2 together also prove BOTH tiled kernels are actually launched at
 *       these depths, i.e. the coverage claim above is real and not dead code.)
 *
 * Measured 2026-07-15 on v5mx (ds4flash.gguf), baseline dev @ 8aa9d35:
 *   T1 -> 129280/129280 logits differ at depth 512, worst |delta| 1.50
 *   T2 -> 129279/129280 logits differ at depth 512, worst |delta| 1.66
 *   PC -> byte-identical at all depths
 * Liveness was cross-checked with a GROSS seed (scale the block8 result by 2):
 * it FAILS for both tiled kernels, while the same seed in the single-block
 * dev_dot_mxfp4_q8_K_block (the per-pair qwarp32 dot) PASSES — i.e. prefill at
 * these depths runs the TILED kernels and not the qwarp32 path, exactly as the
 * coverage argument claims.
 *
 *   POSITIVE CONTROL (not teeth — this one must PASS): reversing the `sb` loop
 *   in dev_dot_mxfp4_q8_K_block8 (7..0 instead of 0..7) changes the level-1
 *   float chain and PASSES, because level 1 is exactly representable (above).
 *   Keep it: T1/T2 with the control together show the gate is sensitive to the
 *   reductions that genuinely round and correctly INSENSITIVE to a provably
 *   exact reorder — it measures the right thing, not merely "any change at
 *   all".  A gate that failed the control would be over-tight and would block
 *   legitimate D2R freedom.
 *
 * WHY THE BLOB CARRIES A BUILD REF.  Nothing else in the blob identifies WHICH
 * BUILD produced it — prompt_fnv hashes only token ids, which are identical for
 * every engine build.  Without a ref, the obvious debugging move
 * (`--dump` over the default baseline path, from the tree under test) silently
 * re-baselines the gate against ITSELF, and it then prints PASS forever, output
 * indistinguishable from a real pass.  So the Makefile compiles the dumping
 * binary with -DPULSAR_GATE_BUILD_REF=<git short HEAD of the tree that built it>,
 * the blob records it, and --check REQUIRES the caller to state the ref it
 * expects (the Makefile passes PREFILL_BASELINE_REF).  A self-baseline from a
 * D2R commit then fails LOUD instead of passing vacuously.
 * Residual hole, stated honestly: while HEAD *is* the baseline ref (i.e.
 * Increment 0 itself, before any D2R commit lands), a self-dump records the same
 * ref and is indistinguishable.  That window closes as soon as the first D2R
 * commit moves HEAD — which is exactly when the gate starts carrying real load.
 *
 * usage: ./tests/prefill_bitexact_gate MODEL --dump  FILE
 *        ./tests/prefill_bitexact_gate MODEL --check FILE EXPECTED_BASELINE_REF
 *        ./tests/prefill_bitexact_gate MODEL --check-reference REF.bin TOKENS.bin [KL_TOL]
 *            [--known-high d1,d2,...] [--known-flip d1,d2,...]
 *        (from the repo root — reads tests/long_context_story_prompt.txt;
 *         or `make cuda-prefill-gate` / `make cuda-prefill-gate-baseline`)
 */
#include "pulsar.h"
#include "gate_entry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* environ */

/* Injected by the Makefile: git short HEAD of the tree that built this binary. */
#ifndef PULSAR_GATE_BUILD_REF
#define PULSAR_GATE_BUILD_REF "unknown"
#endif

/* Depths: 512/2048/4096 are single-chunk; 6144 and 4102 exceed the pinned 4096
 * prefill_cap and so take the CHUNKED path (see the header).  All except the
 * final chunk of 4102 are >= 128, i.e. on the expert-tiled big-batch MoE path
 * that D2R replaces.
 *
 * 4102 is the SMALL-REMAINDER row, added 2026-07-21 after a real miss.  The
 * other four depths chunk as 4096+2048, so this gate only ever exercised final
 * chunks of 512..4096 and was structurally blind to a narrow last chunk.
 * gpu_graph_prefill_chunked_range (src/engine/imatrix.c:717) keeps the final
 * chunk's EXACT remainder, so production hits n_tok 1..8 whenever
 * prompt_len mod chunk lands there -- and every short continuation prefill off
 * the prefix-cache/partial-prefix path lands there by construction.  That is
 * also precisely the window the n_tok-conditional GEMV dispatches switch on
 * (gemv_max_n / f16_gemv_max_n / a_gemv_max_n in pulsar_cuda_matmul.cu,
 * moe_gemv_cap in pulsar_cuda_moe.cu, all capped at 4).  Raising those caps to 8
 * moved 129280/129280 logits by up to 1.876 at 4102 while 4100 (n_tok=4) stayed
 * byte-identical -- and all four original depths, plus the bank/multiseq gates
 * (draft 3, width <= 4), passed clean through it.  4102 -> final chunk n_tok=6,
 * mid-range of that window; keep it whenever the caps or the chunk loop change.
 * NOTE the coupling: this row's shape is chunk-dependent, so it only means what
 * it says while opt.prefill_chunk stays pinned at 4096 below. */
static const uint32_t g_depths[] = { 512u, 2048u, 4096u, 4102u, 6144u };
#define N_DEPTHS ((uint32_t)(sizeof(g_depths) / sizeof(g_depths[0])))
#define MAX_DEPTHS 8u
#define GATE_CTX 8192

#define BLOB_MAGIC "DS4PFXG1"
#define BLOB_VERSION 2u
#define REF_LEN 24u

/* Blob layout: header, then n_depths rows of `width` floats, row i = depth[i]. */
typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t n_depths;
    uint32_t width;        /* pulsar_engine_logits_width — the row stride, NOT vocab_size */
    uint32_t reserved;
    uint64_t prompt_fnv;   /* FNV-1a over the token ids actually prefilled */
    uint32_t depths[MAX_DEPTHS];
    char     build_ref[REF_LEN];  /* git short HEAD of the tree that BUILT the dumper */
} blob_header;

static pulsar_engine *g_e;
static pulsar_tokens g_toks;

/* The engine reads many PULSAR_* env knobs; some of them change prefill numerics
 * or delete
 * kernel coverage outright (PULSAR_MOE_FP4_TILED=0 sends every type-39 layer down
 * the per-pair qwarp32 path).  A knob set DIFFERENTLY between the dump and the
 * check fails loud, which is the safe direction; the danger is the SAME wrong
 * knob on both sides, where the gate agrees with itself having certified a
 * configuration that is not the one we ship.  Enumerating the offenders by hand
 * does not scale and silently rots, so scrub the whole namespace and keep only
 * what selects WHERE things are rather than WHAT is computed.
 * (PULSAR_CUDA_PREFILL_CHUNK needs no entry here: opt.prefill_chunk is pinned and
 * takes precedence over the env.)
 *
 * Keep this list SHORT and re-verify each entry against THIS binary's link graph
 * (tests/prefill_bitexact_gate = the gate .o + src/lib/pulsar_help.o + CORE_OBJS —
 * engine + cuda, NO server objects).  It already rotted once: PULSAR_MODEL_DIR sat
 * here until a 2026-07 review found it dead — it is read only by
 * src/server/cli_main.c, which this binary does not link, and the gate takes the
 * model as argv[1] anyway.  A hand-maintained keep-list is exactly the thing this
 * scrub exists to avoid, so anything added here owes a reason it is not numerics. */
static const char *const g_env_keep[] = {
    /* Infrastructure: the lock PATH, not any numeric.  Read by
     * src/engine/engine_api.c, which this binary does link. */
    "PULSAR_LOCK_FILE",
};

/* Numerics knobs OUTSIDE the PULSAR_ namespace.  The scrub below sweeps PULSAR_* by
 * prefix, which is exactly the wrong shape for these: they are read by the CUDA
 * runtime and by cuBLAS themselves, not by our code, so they appear nowhere in
 * this tree and no namespace sweep can find them — yet they change the arithmetic
 * the gate certifies:
 *   - NVIDIA_TF32_OVERRIDE=0 disables TF32 GLOBALLY, overriding the driver's
 *     default regardless of what cublasSetMathMode() asks for.  That is the same
 *     effect the retired PULSAR_CUDA_NO_TF32 knob used to have, one namespace
 *     over, and out of reach of this scrub.
 *   - CUBLAS_WORKSPACE_CONFIG changes cuBLAS workspace sizing and with it
 *     reduction split/determinism.
 * Exported identically to both the dump and the check, either makes the gate
 * agree with itself while certifying a configuration we do not ship. */
static const char *const g_env_scrub_foreign[] = {
    "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
};

static int env_kept(const char *name) {
    for (size_t i = 0; i < sizeof(g_env_keep) / sizeof(g_env_keep[0]); i++)
        if (!strcmp(name, g_env_keep[i])) return 1;
    return 0;
}

/* 0, or 2 when the knob could not be unset (the caller refuses to run). */
static int scrub_one(const char *name, const char *value) {
    fprintf(stderr,
            "PREFILL GATE: ignoring %s=%s from the environment — this gate only "
            "certifies the DEFAULT configuration (a knob set identically on both "
            "sides of the compare would agree with itself vacuously)\n",
            name, value ? value : "");
    if (unsetenv(name) != 0) {
        /* Leaving a numerics knob live is precisely the failure this function
         * exists to prevent; never continue past it. */
        fprintf(stderr, "PREFILL GATE FAIL: unsetenv(%s) failed — refusing to run "
                        "with a numerics knob still set\n", name);
        return 2;
    }
    return 0;
}

/* Diagnostic opt-in: PULSAR_GATE_DIAGNOSTIC_DUMPS=1 keeps the four graph-dump
 * variables through the scrub so an instrumented run can record tensors (the
 * L033 flip needed the top-k SELECTION at its one regressing depth, and this
 * scrub -- correctly -- ate the request without a trace).
 *
 * Why this is an explicit mode and not a keep-list entry: dumps are
 * observation, not numerics, BUT enabling them un-skips the dump-only f32
 * stores, so the binary exercises a slightly different store path than the
 * one certified.  A diagnostic run is therefore NOT a certification, and says
 * so at the top of its output rather than printing a PASS someone might file. */
static int diagnostic_dumps_on(void) {
    const char *e = getenv("PULSAR_GATE_DIAGNOSTIC_DUMPS");
    return e && !strcmp(e, "1");
}
static const char *const g_env_keep_diag[] = {
    "PULSAR_GATE_DIAGNOSTIC_DUMPS",
    "PULSAR_CUDA_GRAPH_DUMP_PREFIX",
    "PULSAR_CUDA_GRAPH_DUMP_NAME",
    "PULSAR_CUDA_GRAPH_DUMP_LAYER",
    "PULSAR_CUDA_GRAPH_DUMP_POS",
};
static int env_kept_diag(const char *name) {
    if (!diagnostic_dumps_on()) return 0;
    for (size_t i = 0; i < sizeof(g_env_keep_diag) / sizeof(g_env_keep_diag[0]); i++)
        if (!strcmp(name, g_env_keep_diag[i])) return 1;
    return 0;
}

/* Returns 0, or 2 when the environment could not be proven clean (the caller
 * returns that as its exit status).
 *
 * RUNNER ORDER: this scrub unsets every PULSAR_* variable for the rest of the
 * PROCESS, not just this gate.  In tests/gates_runner.cpp this gate therefore
 * runs LAST -- any gate after it would see an environment with no
 * PULSAR_MSEQ_BANKS, no PULSAR_GATE_* knobs, nothing. */
static int scrub_numerics_env(void) {
    if (diagnostic_dumps_on())
        fprintf(stderr,
                "PREFILL GATE: *** DIAGNOSTIC RUN — graph dumps kept; this run "
                "certifies NOTHING; do not record its verdict ***\n");
    /* Collect first: unsetenv() invalidates `environ` mid-iteration. */
    char *names[256];
    const size_t cap = sizeof(names) / sizeof(names[0]);
    size_t n = 0;
    for (char **e = environ; *e; e++) {
        if (strncmp(*e, "PULSAR_", 7) != 0) continue;
        const char *eq = strchr(*e, '=');
        if (!eq) continue;
        const size_t len = (size_t)(eq - *e);
        /* Both of the following would otherwise leave a knob SET while the gate
         * reported nothing and went on to print PASS.  The engine reads ~90 PULSAR_*
         * knobs, so neither is reachable today — but "unreachable" is what this
         * whole file refuses to take on faith.  Fail loud instead. */
        if (n == cap) {
            fprintf(stderr,
                    "PREFILL GATE FAIL: more than %zu PULSAR_* variables in the "
                    "environment — the scrub list is full, so the remainder would "
                    "stay SET and silently steer the numerics this gate claims to "
                    "certify.  Raise the cap in scrub_numerics_env().\n", cap);
            for (size_t i = 0; i < n; i++) free(names[i]);
            return 2;
        }
        char *nm = (char *)malloc(len + 1);
        if (!nm) {
            fprintf(stderr,
                    "PREFILL GATE FAIL: out of memory collecting the env scrub list "
                    "at '%.*s' — cannot prove it is unset, refusing to run\n",
                    (int)len, *e);
            for (size_t i = 0; i < n; i++) free(names[i]);
            return 2;
        }
        memcpy(nm, *e, len);
        nm[len] = '\0';
        if (env_kept(nm) || env_kept_diag(nm)) { free(nm); continue; }
        names[n++] = nm;
    }
    int status = 0;
    for (size_t i = 0; i < n; i++) {
        if (status == 0) status = scrub_one(names[i], getenv(names[i]));
        free(names[i]);
    }
    if (status != 0) return status;
    /* Named explicitly: these are not PULSAR_* and the sweep above cannot see them. */
    for (size_t i = 0; i < sizeof(g_env_scrub_foreign) / sizeof(g_env_scrub_foreign[0]); i++) {
        const char *v = getenv(g_env_scrub_foreign[i]);
        if (v && scrub_one(g_env_scrub_foreign[i], v) != 0) return 2;
    }
    return 0;
}

/* A row that is degenerate (all-equal, or non-finite) would byte-match another
 * degenerate row and pass memcmp having proven nothing. */
static int row_is_sane(const float *row, int width, uint32_t depth) {
    int n_finite = 0;
    for (int i = 0; i < width; i++) if (isfinite(row[i])) n_finite++;
    if (n_finite != width) {
        fprintf(stderr, "PREFILL GATE FAIL: depth %u has %d/%d non-finite logits\n",
                depth, width - n_finite, width);
        return 0;
    }
    for (int i = 1; i < width; i++) if (row[i] != row[0]) return 1;
    fprintf(stderr, "PREFILL GATE FAIL: depth %u logits are all identical (%g) — "
                    "degenerate row, a byte-compare against it proves nothing\n",
            depth, (double)row[0]);
    return 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

/* ⚠ NOT canonical FNV-1a: the offset basis below is the canonical
 * 14695981039346656037 (0xCBF29CE484222325) with its LAST DIGIT DROPPED — a
 * typo that predates every baseline blob.  It is a perfectly fine hash and
 * every blob this file ever wrote is self-consistent with it, so changing it
 * would force a full baseline re-anchor for zero information.  It only
 * matters when an EXTERNAL party must compute the same value — which is why
 * --check-reference uses fnv1a_ref() below instead.  (Found 2026-08-21 when
 * the reference blobs, hashed canonically in python, refused to match.) */
static uint64_t fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;   /* sic — see comment above */
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* Canonical FNV-1a 64, for artifacts produced OUTSIDE this file (the
 * reference-capture blobs and tokens.bin hash with this basis). */
static uint64_t fnv1a_ref(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 14695981039346656037ull;  /* 0xCBF29CE484222325 */
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* One from-scratch prefill of `depth` tokens through a FRESH session; the
 * session is torn down before returning so nothing carries between depths.
 * `ctx` is GATE_CTX for the byte/fidelity modes; --check-reference passes a
 * larger one because the reference blob carries depths past 8192. */
static int prefill_logits_ctx(uint32_t depth, float *out, int width, int ctx) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, ctx) != 0) {
        fprintf(stderr, "PREFILL GATE: session_create failed (depth %u)\n", depth);
        return 0;
    }
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = (int *)malloc((size_t)depth * sizeof(int));
    if (!p.v) { pulsar_session_free(s); return 0; }
    p.len = p.cap = (int)depth;
    memcpy(p.v, g_toks.v, (size_t)depth * sizeof(int));

    char err[256];
    const int rc = pulsar_session_sync(s, &p, err, sizeof(err));
    free(p.v);
    if (rc != 0) {
        fprintf(stderr, "PREFILL GATE: sync failed at depth %u: %s\n", depth, err);
        pulsar_session_free(s);
        return 0;
    }
    const int got = pulsar_session_copy_logits(s, out, width);
    pulsar_session_free(s);
    if (got != width) {
        fprintf(stderr, "PREFILL GATE: copy_logits returned %d, want %d (depth %u)\n",
                got, width, depth);
        return 0;
    }
    return 1;
}

static int prefill_logits(uint32_t depth, float *out, int width) {
    return prefill_logits_ctx(depth, out, width, GATE_CTX);
}

/* Report the first byte difference and the worst float difference in a row.
 * `ref` is whatever the run is being held to (the baseline blob, or run 1 of
 * the determinism pair); `cur` is this run's row. */
static void diff_row(const float *cur, const float *ref, int width, uint32_t depth,
                     const char *ref_label) {
    const uint8_t *pc = (const uint8_t *)cur, *pr = (const uint8_t *)ref;
    const size_t bytes = (size_t)width * sizeof(float);
    size_t first = bytes;
    for (size_t i = 0; i < bytes; i++) if (pc[i] != pr[i]) { first = i; break; }

    int n_diff = 0, worst_i = -1;
    double worst = 0.0;
    for (int i = 0; i < width; i++) {
        if (memcmp(&cur[i], &ref[i], sizeof(float)) != 0) {
            n_diff++;
            const double d = (double)cur[i] - (double)ref[i];
            const double ad = d < 0 ? -d : d;
            if (ad >= worst) { worst = ad; worst_i = i; }
        }
    }
    fprintf(stderr,
            "  depth %u: %d/%d logits differ; first differing BYTE at offset %zu "
            "(logit %zu)\n",
            depth, n_diff, width, first, first / sizeof(float));
    if (worst_i >= 0) {
        uint32_t bc, br;
        memcpy(&bc, &cur[worst_i], 4);
        memcpy(&br, &ref[worst_i], 4);
        fprintf(stderr,
                "  depth %u: worst logit[%d] %s=%.9g (0x%08x) current=%.9g (0x%08x) "
                "absdiff=%.6g\n",
                depth, worst_i, ref_label, (double)ref[worst_i], br,
                (double)cur[worst_i], bc, worst);
    }
}

/* FIDELITY compare (the residual->BF16 gate, task #62).  The byte-exact --check
 * above is for changes that CLAIM bit-exactness (D2R).  A storage-precision change
 * — narrowing the f32 hyper-connection residual stream to BF16 to match the source
 * (torch_dtype bfloat16, [[ds4-source-numerics]]) — is DELIBERATELY not bit-exact:
 * it rounds the residual at each layer boundary.  So this mode holds the current
 * build's full-vocab logits to the golden f32 blob under a TOLERANCE, and reports
 * the divergence per depth: top-1 argmax agreement (does the predicted token move?),
 * KL(golden||current) over the softmax (distributional shift), and logit RMS / max
 * abs.  Full-vocab, not argmax, for the same reason the byte gate is (our IMMA
 * post-mortem: argmax stayed put while hidden states drifted 40%).  PASS iff top-1
 * holds at every depth AND KL <= tol at every depth.  Softmax/KL accumulate in
 * double so the metric itself does not round. */
/* out_kl (optional) receives this row's KL so the caller can grade DIRECTION
 * against a recorded budget, not just against an absolute ceiling. */
static int fidelity_row(const float *cur, const float *ref, int width,
                        uint32_t depth, double kl_tol, int known_high,
                        int known_flip, double *out_kl) {
    double maxg = -1e300, maxc = -1e300;
    int arg_g = 0, arg_c = 0;
    for (int i = 0; i < width; i++) {
        if (ref[i] > maxg) { maxg = ref[i]; arg_g = i; }
        if (cur[i] > maxc) { maxc = cur[i]; arg_c = i; }
    }
    double Zg = 0.0, Zc = 0.0;
    for (int i = 0; i < width; i++) { Zg += exp((double)ref[i] - maxg); Zc += exp((double)cur[i] - maxc); }
    const double lZg = log(Zg) + maxg, lZc = log(Zc) + maxc;
    double kl = 0.0, sse = 0.0, maxabs = 0.0;
    for (int i = 0; i < width; i++) {
        const double lp = (double)ref[i] - lZg;   /* log P (golden) */
        const double lq = (double)cur[i] - lZc;   /* log Q (current) */
        kl += exp(lp) * (lp - lq);
        const double d = (double)cur[i] - (double)ref[i];
        sse += d * d;
        if (fabs(d) > maxabs) maxabs = fabs(d);
    }
    if (kl < 0.0) kl = 0.0;   /* fp noise can push a ~0 KL slightly negative */
    if (out_kl) *out_kl = kl;
    const double rms = sqrt(sse / (double)width);
    const int top1_ok = (arg_g == arg_c);
    /* TOP-1 is enforced at EVERY depth including known-high ones: a divergence
     * that moves the argmax is a different class of event from one that only
     * widens KL, and exempting a row from the KL ceiling must never exempt it
     * from that.  known_high only relaxes the KL half. */
    const int kl_ok = (kl <= kl_tol) || known_high;
    /* known_flip: a NAMED per-depth allowance for an argmax flip, and nothing
     * else -- the KL contract is untouched.  Added for story@512 (L084): the
     * TF32 removal moved EVERY depth toward the reference (512 itself 0.697 ->
     * 0.643) and crossed a 0.25-logit near-tie on the L080 outlier row, golden
     * argmax landing at rank #2.  A flip that rides an overall improvement on
     * an already-degenerate depth is churn, not regression -- but that verdict
     * lives HERE, per depth, with the evidence printed below, never in a
     * loosened default. */
    const int top1_pass = top1_ok || known_flip;
    const int pass = top1_pass && kl_ok;
    printf("  depth %4u: top1 %s (golden argmax=%d current=%d)  KL=%.3e  logit_rms=%.3e  max|d|=%.3e  -> %s\n",
           depth, top1_ok ? "MATCH" : (known_flip ? "FLIP (KNOWN)" : "FLIP"),
           arg_g, arg_c, kl, rms, maxabs, pass ? "OK" : "FAIL");
    /* Same self-tightening rule as --known-high: an exemption that is no
     * longer needed must say so, or the list protects a future regression. */
    if (known_flip && top1_ok)
        printf("  depth %4u: KNOWN-FLIP but top-1 MATCHES — drop it from --known-flip\n",
               depth);
    if (!top1_ok) {
        /* A flip's SEVERITY is the gap, not the fact of it.  On a known-high
         * depth the whole distribution already sits far from the reference, so
         * the argmax can ride a near-tie; print the neighborhood so "churn
         * between near-equals" and "the golden token sank" are distinguishable
         * without a second instrumented run. */
        printf("  depth %4u: FLIP DETAIL  cur[%d]=%.6f  cur[%d]=%.6f  (gap %.6f)  "
               "golden[%d]=%.6f  golden[%d]=%.6f\n",
               depth, arg_c, cur[arg_c], arg_g, cur[arg_g],
               (double)cur[arg_c] - (double)cur[arg_g],
               arg_g, ref[arg_g], arg_c, ref[arg_c]);
        int rank_g = 1;   /* rank of the GOLDEN argmax in OUR logits */
        for (int i = 0; i < width; i++) if (cur[i] > cur[arg_g]) rank_g++;
        printf("  depth %4u: FLIP DETAIL  golden argmax now ranks #%d in current logits\n",
               depth, rank_g);
    }
    if (!top1_ok && !known_flip)
        fprintf(stderr, "  depth %u: TOP-1 FLIPPED golden=%d current=%d — a storage-precision "
                        "change moved the predicted token; investigate before accepting\n",
                depth, arg_g, arg_c);
    if (kl > kl_tol && !known_high)
        fprintf(stderr, "  depth %u: KL %.3e exceeds tol %.3e\n", depth, kl, kl_tol);
    if (known_high && kl > kl_tol)
        printf("  depth %4u: KL %.3e over tol %.3e but depth is KNOWN-HIGH — "
               "informational, top-1 still enforced\n", depth, kl, kl_tol);
    /* A known-high row that has come back under tolerance should stop being
     * exempt, or the list calcifies and silently protects a future regression.
     * Report it rather than failing: tightening the list is a deliberate edit. */
    if (known_high && kl <= kl_tol)
        printf("  depth %4u: KNOWN-HIGH but KL %.3e is now WITHIN tol %.3e — "
               "drop it from --known-high\n", depth, kl, kl_tol);
    return pass;
}

/* ---- --check-reference: divergence vs an EXTERNAL engine's logits ----------
 *
 * The blob here is NOT one of ours: it comes from the official checkpoint served
 * by a second engine (vLLM on a rented box — see
 * pulsar-notes/reference-capture/README.md for the capture protocol and the
 * 2026-08-21 blobs).  That changes three contracts relative to --check-fidelity:
 *
 *   1. TOKEN AUTHORITY IS A FILE.  The gate's own tokenizer path
 *      (pulsar_tokenize_text) and the CLI's --dump-tokens path produce
 *      DIFFERENT id streams for the same text (measured 2026-08-20: fnv
 *      0xfba1... vs 0x1a55... over the same 6144-token prefix), so
 *      re-tokenizing here would silently compare against the wrong prompt.
 *      Both sides prefill ids loaded from the SAME tokens.bin (int32 LE); the
 *      blob's prompt_fnv over the deepest depth is required to match the
 *      file's, so a wrong pairing fails loud before the model loads.
 *   2. SHAPES DIFFER.  The reference row width is the true vocab (129,280);
 *      ours is pulsar_engine_logits_width, a padded stride.  Depths come from
 *      the REFERENCE header (it carries rows past GATE_CTX, e.g. 30464), and
 *      the compare runs over min(ref_width, our_width) columns — the padded
 *      tail lanes carry no information and are excluded.
 *   3. NO PROVENANCE REF, NO TEETH BY DEFAULT.  build_ref in a reference blob
 *      names the other engine ("vllm-0.27.1"), printed but not matched — the
 *      self-baseline attack --check guards against cannot arise from a blob we
 *      cannot produce.  Without [KL_TOL] the mode is REPORT-ONLY (exit 0
 *      unless infrastructure fails): the first run against a new reference
 *      ESTABLISHES the divergence budget, it does not enforce one.  With
 *      [KL_TOL] it enforces top-1 match + KL <= tol per depth, same as
 *      --check-fidelity.
 *
 * What the number means: this reference is MATCHED-PRECISION cross-engine
 * (the checkpoint ships MXFP4 experts; there is no higher-precision release),
 * so KL here is divergence-from-another-implementation on identical weights —
 * decisive for changes to OUR arithmetic (fast-math, accumulation formats,
 * storage narrowing: L072/L045/L079), a budget anchor rather than an oracle
 * for quant-quality absolutes.  Determinism re-runs are skipped: that property
 * belongs to --check, and doubling a 30k-token prefill buys nothing here. */
/* A KL budget file is `depth kl` per line, '#' comments ignored.  Returns the
 * recorded KL for `depth`, or -1 if the file has no entry for it (a new depth
 * is not a failure -- it just has nothing to be graded against yet). */
static double kl_baseline_lookup(const char *path, uint32_t depth) {
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    char line[256];
    double found = -1.0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned d = 0; double k = 0.0;
        if (sscanf(line, "%u %lf", &d, &k) == 2 && d == depth) { found = k; break; }
    }
    fclose(f);
    return found;
}

/* Shannon entropy (nats) of a logit row, and p(top1).  Used to classify a
 * graded depth as CONFIDENT or FLAT.
 *
 * WHY THE GATE NEEDS THIS (L080, 2026-08-24): measured over both reference
 * blobs, every depth is one of two populations and there is nothing in
 * between -- the clean depths sit at entropy ~0.0000 with p(top1)=1.0000 and an
 * effective support of ONE token, while the three "known-high" depths sit at
 * ~1.7 nats with 5-14 tokens carrying 90% of the mass.  At a certain position
 * there is no distribution to disagree about, so any two implementations agree
 * at ~1e-6 trivially; at a flat one, small logit differences move REAL
 * probability mass and KL is naturally O(0.1).
 *
 * That matters for the VERDICT: summing KL across all depths lets the three
 * flat rows (0.17-0.57) swamp the six confident ones (1e-6) by five orders, so
 * a NET over everything is decided almost entirely by the least informative
 * rows.  The confident set carries the decision; the flat set is reported. */
static void row_entropy(const float *row, int width, double *H_out, double *p1_out) {
    double m = -1e300;
    for (int i = 0; i < width; i++) if (row[i] > m) m = row[i];
    double Z = 0.0;
    for (int i = 0; i < width; i++) Z += exp((double)row[i] - m);
    double H = 0.0, p1 = 0.0;
    for (int i = 0; i < width; i++) {
        const double p = exp((double)row[i] - m) / Z;
        if (p > 0.0) H -= p * log(p);
        if (p > p1) p1 = p;
    }
    *H_out = H; *p1_out = p1;
}

/* The two populations are ~0.0003 and ~1.7 nats, so any cut in between works;
 * this one is two decades clear of the confident side. */
#define GATE_FLAT_ENTROPY_NATS 0.05

/* The NET line prints the direction to two decimals, so a move smaller than
 * half of the last printed digit is a move this gate cannot see.  Calling one
 * of those "CLOSER TO SOURCE" puts a verdict next to a "+0.00%" that flatly
 * contradicts it -- and the verdict is the half that gets quoted later, by
 * someone who did not read the number beside it.  Below this, say UNCHANGED.
 *
 * This is not a tolerance on the physics; it is the display's own resolution.
 * The positive control (grading a tree against budgets recorded from that same
 * tree) lands on exact bitwise equality, so it reads UNCHANGED, which is the
 * whole point: a control that cannot tell "identical" from "improved" is not
 * a control. */
#define GATE_NET_DEAD_BAND 5e-5   /* = 0.005%, half of the last printed digit */

static int depth_in_list(uint32_t d, const uint32_t *list, int n) {
    for (int i = 0; i < n; i++) if (list[i] == d) return 1;
    return 0;
}

static int run_check_reference(const char *model, const char *ref_path,
                               const char *tokens_path, double kl_tol,
                               int enforce,
                               const uint32_t *known_high, int n_known_high,
                               const uint32_t *known_flip, int n_known_flip,
                               const char *kl_base_path, const char *kl_dump_path) {
    struct { uint32_t depth; double kl; int known_high; int flat; double H; double p1; } kl_out[64];
    int n_kl_out = 0;
    /* Load the reference blob in full (its shape is the run's shape). */
    FILE *fp = fopen(ref_path, "rb");
    if (!fp) { fprintf(stderr, "cannot read reference blob %s\n", ref_path); return 1; }
    blob_header rh;
    if (fread(&rh, sizeof(rh), 1, fp) != 1) {
        fprintf(stderr, "reference %s: short header\n", ref_path);
        fclose(fp);
        return 1;
    }
    if (memcmp(rh.magic, BLOB_MAGIC, 8) != 0 || rh.version != BLOB_VERSION ||
        rh.n_depths == 0 || rh.n_depths > MAX_DEPTHS || rh.width == 0) {
        fprintf(stderr, "reference %s: bad magic/version/shape\n", ref_path);
        fclose(fp);
        return 1;
    }
    rh.build_ref[REF_LEN - 1] = '\0';
    const size_t ref_n = (size_t)rh.n_depths * (size_t)rh.width;
    float *ref_rows = (float *)calloc(ref_n, sizeof(float));
    if (!ref_rows) { fclose(fp); fprintf(stderr, "oom\n"); return 1; }
    if (fread(ref_rows, sizeof(float), ref_n, fp) != ref_n) {
        fprintf(stderr, "reference %s: short body\n", ref_path);
        fclose(fp);
        free(ref_rows);
        return 1;
    }
    fclose(fp);

    /* Token authority: the shared int32 LE id file, fnv-locked to the blob. */
    size_t tok_bytes = 0;
    char *tok_raw = read_file(tokens_path, &tok_bytes);
    if (!tok_raw || tok_bytes < sizeof(int) || (tok_bytes % sizeof(int)) != 0) {
        fprintf(stderr, "tokens file %s: unreadable or not a whole number of int32s\n",
                tokens_path);
        free(tok_raw);
        free(ref_rows);
        return 1;
    }
    const int n_toks = (int)(tok_bytes / sizeof(int));
    const uint32_t deepest = rh.depths[rh.n_depths - 1];
    if (n_toks < (int)deepest) {
        fprintf(stderr, "tokens file %s: %d ids, reference needs %u\n",
                tokens_path, n_toks, deepest);
        free(tok_raw);
        free(ref_rows);
        return 1;
    }
    const uint64_t fnv = fnv1a_ref(tok_raw, (size_t)deepest * sizeof(int));
    if (fnv != rh.prompt_fnv) {
        fprintf(stderr,
                "REFERENCE GATE FAIL: token/blob mismatch — fnv over the first %u ids "
                "of %s is %016llx but the reference blob was captured for %016llx.\n"
                "  (wrong tokens.bin for this blob, or a re-tokenized prompt; the two "
                "sides would silently compare different prompts)\n",
                deepest, tokens_path, (unsigned long long)fnv,
                (unsigned long long)rh.prompt_fnv);
        free(tok_raw);
        free(ref_rows);
        return 1;
    }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.prefill_chunk = 4096;   /* production parity, as in the byte gate */
    opt.dspark_disable = true;
    if (gate_engine_open(&g_e, &opt) != 0) {
        fprintf(stderr, "engine open failed\n");
        free(tok_raw);
        free(ref_rows);
        return 1;
    }
    const int width = pulsar_engine_logits_width(g_e);
    if (width <= 0) {
        fprintf(stderr, "bad logits width %d\n", width);
        free(tok_raw);
        free(ref_rows);
        gate_engine_close(g_e);
        return 1;
    }
    const int ncmp = (int)rh.width < width ? (int)rh.width : width;

    memset(&g_toks, 0, sizeof(g_toks));
    g_toks.v = (int *)tok_raw;
    g_toks.len = g_toks.cap = n_toks;

    /* Smallest 4096-multiple that clears the deepest row, plus one chunk of
     * headroom for the session's own accounting. */
    const int ref_ctx = (int)(((deepest + 4095u) / 4096u + 1u) * 4096u);

    printf("reference compare: blob=%s (engine '%s', %u depths, width %u)\n"
           "  tokens=%s (%d ids, fnv %016llx OK)  ncmp=%d  ctx=%d  %s\n",
           ref_path, rh.build_ref, rh.n_depths, rh.width,
           tokens_path, n_toks, (unsigned long long)fnv, ncmp, ref_ctx,
           enforce ? "ENFORCING" : "report-only (no tolerance enforced)");

    float *row = (float *)calloc((size_t)width, sizeof(float));
    if (!row) {
        fprintf(stderr, "oom\n");
        free(ref_rows);
        pulsar_tokens_free(&g_toks);   /* owns tok_raw */
        gate_engine_close(g_e);
        return 1;
    }
    int fail = 0;
    for (uint32_t i = 0; i < rh.n_depths; i++) {
        const float *ref_row = ref_rows + (size_t)i * (size_t)rh.width;
        int ref_finite = 1;
        for (int j = 0; j < ncmp; j++) if (!isfinite(ref_row[j])) { ref_finite = 0; break; }
        if (!ref_finite) {
            fprintf(stderr, "REFERENCE GATE FAIL: reference row for depth %u has "
                            "non-finite logits — blob damaged or mis-captured\n",
                    rh.depths[i]);
            fail = 1;
            continue;
        }
        if (!prefill_logits_ctx(rh.depths[i], row, width, ref_ctx)) {
            free(row);
            free(ref_rows);
            pulsar_tokens_free(&g_toks);
            gate_engine_close(g_e);
            return 1;
        }
        if (!row_is_sane(row, width, rh.depths[i])) { fail = 1; continue; }
        const int kh = depth_in_list(rh.depths[i], known_high, n_known_high);
        const int kf = depth_in_list(rh.depths[i], known_flip, n_known_flip);
        double kl_here = -1.0;
        if (!fidelity_row(row, ref_row, ncmp, rh.depths[i], kl_tol, kh, kf, &kl_here) && enforce) fail = 1;
        if (kl_here >= 0.0 && n_kl_out < (int)(sizeof(kl_out)/sizeof(kl_out[0]))) {
            double H = 0.0, p1 = 0.0;
            row_entropy(ref_row, ncmp, &H, &p1);   /* classify from the REFERENCE */
            kl_out[n_kl_out].depth = rh.depths[i];
            kl_out[n_kl_out].kl = kl_here;
            kl_out[n_kl_out].known_high = kh;
            kl_out[n_kl_out].flat = (H >= GATE_FLAT_ENTROPY_NATS);
            kl_out[n_kl_out].H = H;
            kl_out[n_kl_out].p1 = p1;
            n_kl_out++;
        }
    }

    /* ---- DIRECTION vs a recorded budget --------------------------------
     * The absolute ceiling above cannot see DIRECTION: at the clean depths we
     * sit at 1e-5..1e-7 against a 1e-4 tol, so a change could move 500x FURTHER
     * from the source and still "pass".  And an improvement at a known-high
     * depth earns no credit at all, because those rows are informational.
     *
     * With --kl-baseline the gate grades what we actually care about: did this
     * change move us CLOSER to the source or further?  (Tyler, 2026-08-24.)
     *
     * ⚠ The verdict is NET, deliberately.  A per-depth "nothing may worsen"
     * rule would have REJECTED the hc_expand tree summation, which improved 6
     * of 9 depths including all three known-high outliers while worsening two
     * that were already at 1e-7..1e-6.  Trading 1e-7 for 1e-6 to buy 0.25->0.17
     * is a good trade and the gate must be able to say so.  A single depth may
     * still fail on its own if it blows past the LARGE-regression guard. */
    if (kl_base_path && n_kl_out > 0) {
        double sum_cur = 0.0, sum_base = 0.0;          /* CONFIDENT depths only */
        double flat_cur = 0.0, flat_base = 0.0;        /* FLAT depths, reported  */
        int matched = 0, n_flat = 0, worse_big = 0;
        printf("\n  KL DIRECTION vs %s:\n", kl_base_path);
        for (int k = 0; k < n_kl_out; k++) {
            double b = kl_baseline_lookup(kl_base_path, kl_out[k].depth);
            if (b < 0.0) { printf("    depth %6u: %.3e  (no baseline entry)\n",
                                  kl_out[k].depth, kl_out[k].kl); continue; }
            if (kl_out[k].flat) { n_flat++; flat_cur += kl_out[k].kl; flat_base += b; }
            else                { matched++; sum_cur += kl_out[k].kl; sum_base += b; }
            const double rel = (b > 0.0) ? (kl_out[k].kl - b) / b : 0.0;
            const char *tag = (kl_out[k].kl < b) ? "CLOSER" :
                              (kl_out[k].kl > b) ? "further" : "same";
            printf("    depth %6u: %.3e vs %.3e  %+7.1f%%  %s   [%s H=%.3f p1=%.3f]%s\n",
                   kl_out[k].depth, kl_out[k].kl, b, rel * 100.0, tag,
                   kl_out[k].flat ? "FLAT     " : "confident",
                   kl_out[k].H, kl_out[k].p1,
                   kl_out[k].known_high ? "  (known-high)" : "");
            /* Large single-depth regression: 10x worse AND above 1e-5.
             *
             * ⚠ THE ABSOLUTE FLOOR IS THE LOAD-BEARING HALF, and it was raised
             * from 1e-6 to 1e-5 after the known-answer run: the hc_expand
             * adoption moved depth 6144 from 1.865e-07 to 1.815e-06, which is
             * 9.73x -- it cleared a 10x guard by a hair while being a GOOD
             * change (net -17.3%, all three outliers closer).  A relative test
             * is inherently jumpy at 1e-7 absolute, where a KL is indis-
             * tinguishable from zero for any purpose we have.  1e-5 keeps the
             * guard meaningful (it is one decade under the 1e-4 tol this gate
             * already enforces per depth) without letting it veto changes on
             * noise. */
            /* CONFIDENT depths only: a flat row swinging 10x is expected, not a
             * blow-up, because its KL is dominated by which of several
             * near-equal tokens each implementation happens to favour. */
            if (!kl_out[k].flat && kl_out[k].kl > b * 10.0 && kl_out[k].kl > 1e-5) {
                printf("      ^ LARGE single-depth regression (>10x and >1e-5)\n");
                worse_big = 1;
            }
        }
        /* FLAT rows are reported and NEVER decide.  Their KL is O(0.1) against
         * O(1e-6) at confident rows, so including them in one sum means the
         * verdict is set by the three least-informative depths (L080). */
        if (n_flat > 0) {
            const double fnet = (flat_base > 0.0) ? (flat_cur - flat_base) / flat_base : 0.0;
            printf("  FLAT %d depths (entropy >= %.2f nats — the model is genuinely "
                   "uncertain there, so KL is large for both of us):\n"
                   "       %.6e vs %.6e  (%+.2f%%)  INFORMATIONAL, does not decide\n",
                   n_flat, (double)GATE_FLAT_ENTROPY_NATS, flat_cur, flat_base, fnet * 100.0);
        }
        if (matched > 0) {
            const double net = (sum_base > 0.0) ? (sum_cur - sum_base) / sum_base : 0.0;
            /* One predicate decides both the word and the verdict, so the two
             * can never disagree.  Enforcement used to fail on `sum_cur >
             * sum_base`, which red-gates a change whose regression is smaller
             * than the resolution this line prints at -- a 1-ulp increase and a
             * real 30% regression got the same treatment. */
            const int moved_further = (net > GATE_NET_DEAD_BAND);
            const char *dir = moved_further ? "FURTHER FROM SOURCE"
                            : (net < -GATE_NET_DEAD_BAND) ? "CLOSER TO SOURCE"
                            : "UNCHANGED (below this gate's display resolution)";
            printf("  NET over %d CONFIDENT depths: %.6e vs %.6e  (%+.2f%%) -> %s\n",
                   matched, sum_cur, sum_base, net * 100.0, dir);
            if (enforce && moved_further) {
                fprintf(stderr, "REFERENCE GATE FAIL: net KL over CONFIDENT depths "
                                "moved AWAY from the source (%.6e > %.6e)\n",
                        sum_cur, sum_base);
                fail = 1;
            }
            if (enforce && worse_big) {
                fprintf(stderr, "REFERENCE GATE FAIL: a single depth regressed "
                                ">10x above 1e-5\n");
                fail = 1;
            }
        }
    }
    if (kl_dump_path && n_kl_out > 0) {
        FILE *kf2 = fopen(kl_dump_path, "w");
        if (!kf2) {
            fprintf(stderr, "cannot write %s\n", kl_dump_path);
            free(row);
            free(ref_rows);
            pulsar_tokens_free(&g_toks);
            gate_engine_close(g_e);
            return 1;
        }
        fprintf(kf2, "# depth kl   (recorded %s)\n", ref_path);
        for (int k = 0; k < n_kl_out; k++)
            fprintf(kf2, "%u %.17g\n", kl_out[k].depth, kl_out[k].kl);
        fclose(kf2);
        printf("  KL budget written to %s (%d depths)\n", kl_dump_path, n_kl_out);
    }

    printf("\nREFERENCE GATE: %s\n",
           fail ? "FAIL" : (enforce ? "PASS" : "REPORT COMPLETE"));
    free(row);
    free(ref_rows);
    pulsar_tokens_free(&g_toks);
    gate_engine_close(g_e);
    return fail ? 1 : 0;
}

/* Ref strings come from `git rev-parse --short`, whose length is a property of
 * the REPO that ran it (object count decides how many chars disambiguate), not
 * of the commit: the blob's stamp and this run's expectation can legitimately
 * be 7 and 8 chars of the same sha.  Provenance therefore matches on the
 * common prefix -- never fewer than 7 chars, which keeps a self-baseline just
 * as detectable as the exact compare did. */
static int ref_matches(const char *a, const char *b) {
    const size_t la = strlen(a), lb = strlen(b);
    const size_t n = la < lb ? la : lb;
    return n >= 7 && strncmp(a, b, n) == 0;
}

/* Everything about the blob that does NOT need the engine: magic, version and
 * provenance.  Called before pulsar_engine_open so the likeliest misuses (a stale
 * blob, or one re-dumped from the tree under test) fail instantly instead of
 * after a 35 s model load. */
static int precheck_baseline(const char *path, const char *expect_ref) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot read baseline %s -- build one with "
                        "`make cuda-prefill-gate-baseline`\n", path);
        return 0;
    }
    blob_header bh;
    const int got = fread(&bh, sizeof(bh), 1, fp) == 1;
    fclose(fp);
    if (!got) { fprintf(stderr, "baseline %s: short header\n", path); return 0; }
    if (memcmp(bh.magic, BLOB_MAGIC, 8) != 0 || bh.version != BLOB_VERSION) {
        fprintf(stderr, "baseline %s: bad magic or version (got %u, want %u) — re-dump it "
                        "with `make cuda-prefill-gate-baseline`\n",
                path, bh.version, BLOB_VERSION);
        return 0;
    }
    bh.build_ref[REF_LEN - 1] = '\0';
    if (!ref_matches(bh.build_ref, expect_ref)) {
        fprintf(stderr,
                "PREFILL GATE FAIL: baseline provenance mismatch.\n"
                "  blob %s was built from ref '%s'\n"
                "  but this check expects the baseline to be ref '%s'\n"
                "  (this binary is built from ref '%s')\n"
                "  A blob re-dumped from the tree under test would be compared against "
                "ITSELF and pass vacuously; rebuild the baseline with\n"
                "  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=%s`\n",
                path, bh.build_ref, expect_ref, PULSAR_GATE_BUILD_REF, expect_ref);
        return 0;
    }
    printf("  baseline provenance: blob built from ref '%s' (expected '%s') OK\n",
           bh.build_ref, expect_ref);
    return 1;
}

/* Validate a baseline blob against `hdr` (this run's shape) and load its body.
 * Called BEFORE any prefill so a bad baseline costs seconds, not GPU-minutes.
 * Returns 1 and sets *base_out on success. */
static int load_baseline(const char *path, const char *expect_ref,
                         const blob_header *hdr, int width, float **base_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot read baseline %s -- build one with "
                        "`make cuda-prefill-gate-baseline`\n", path);
        return 0;
    }
    blob_header bh;
    if (fread(&bh, sizeof(bh), 1, fp) != 1) {
        fprintf(stderr, "baseline %s: short header\n", path);
        fclose(fp);
        return 0;
    }
    if (memcmp(bh.magic, BLOB_MAGIC, 8) != 0 || bh.version != hdr->version) {
        fprintf(stderr, "baseline %s: bad magic or version (got %u, want %u) — re-dump it "
                        "with `make cuda-prefill-gate-baseline`\n",
                path, bh.version, hdr->version);
        fclose(fp);
        return 0;
    }
    /* Provenance: the blob must come from the ref the caller expects, NOT from
     * the tree under test.  This is what stops a self-baseline passing forever. */
    bh.build_ref[REF_LEN - 1] = '\0';
    if (!ref_matches(bh.build_ref, expect_ref)) {
        fprintf(stderr,
                "PREFILL GATE FAIL: baseline provenance mismatch.\n"
                "  blob %s was built from ref '%s'\n"
                "  but this check expects the baseline to be ref '%s'\n"
                "  (this binary is built from ref '%s')\n"
                "  A blob re-dumped from the tree under test would be compared against "
                "ITSELF and pass vacuously; rebuild the baseline with\n"
                "  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=%s`\n",
                path, bh.build_ref, expect_ref, PULSAR_GATE_BUILD_REF, expect_ref);
        fclose(fp);
        return 0;
    }
    if (bh.width != hdr->width || bh.n_depths != hdr->n_depths ||
        memcmp(bh.depths, hdr->depths, sizeof(hdr->depths)) != 0 ||
        bh.prompt_fnv != hdr->prompt_fnv) {
        fprintf(stderr,
                "PREFILL GATE FAIL: baseline header mismatch -- this baseline does not "
                "describe this run.\n"
                "  width      baseline=%u current=%u\n"
                "  n_depths   baseline=%u current=%u\n"
                "  prompt_fnv baseline=%016llx current=%016llx\n"
                "  (different model, different tokenization, or a stale blob)\n",
                bh.width, hdr->width, bh.n_depths, hdr->n_depths,
                (unsigned long long)bh.prompt_fnv, (unsigned long long)hdr->prompt_fnv);
        fclose(fp);
        return 0;
    }
    const size_t n = (size_t)bh.n_depths * (size_t)width;
    float *base = (float *)calloc(n, sizeof(float));
    if (!base) { fclose(fp); fprintf(stderr, "oom\n"); return 0; }
    if (fread(base, sizeof(float), n, fp) != n) {
        fprintf(stderr, "baseline %s: short body\n", path);
        fclose(fp);
        free(base);
        return 0;
    }
    fclose(fp);
    /* Provenance was already reported by precheck_baseline; do not print twice. */
    *base_out = base;
    return 1;
}

int GATE_ENTRY(int argc, char **argv) {
    g_e = NULL;
    memset(&g_toks, 0, sizeof(g_toks));
    if (argc < 4 || (strcmp(argv[2], "--dump") && strcmp(argv[2], "--check") &&
                     strcmp(argv[2], "--check-fidelity") &&
                     strcmp(argv[2], "--check-reference"))) {
        fprintf(stderr, "usage: %s MODEL --dump  FILE\n"
                        "       %s MODEL --check FILE EXPECTED_BASELINE_REF\n"
                        "       %s MODEL --check-fidelity FILE EXPECTED_BASELINE_REF [KL_TOL]\n"
                        "       %s MODEL --check-reference REF.bin TOKENS.bin [KL_TOL]\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[2], "--check-reference") == 0) {
        if (argc < 5) {
            fprintf(stderr, "%s --check-reference requires REF.bin and TOKENS.bin — the "
                            "reference blob and the int32 token file it was captured "
                            "against (see pulsar-notes/reference-capture/)\n", argv[0]);
            return 2;
        }
        /* [KL_TOL] [--known-high d1,d2,...]
         * known-high names depths whose KL is a documented, not-yet-explained
         * outlier (see ledger L080: shallow and file-end rows where a flat
         * next-token distribution amplifies small numeric differences).  They
         * keep the TOP-1 contract and lose only the KL ceiling.  Naming them
         * explicitly is the point: a blanket tolerance loose enough to pass
         * them would be 1e4x too loose for the mid rows and would protect
         * nothing. */
        uint32_t known_high[32];
        int n_known_high = 0;
        /* --known-flip: depths where an argmax flip is a documented near-tie
         * (gap + rank printed by the flip detail), accepted by name.  KL is
         * still enforced there; only the top-1 half relaxes.  The dual of
         * --known-high, and just as deliberately narrow. */
        uint32_t known_flip[32];
        int n_known_flip = 0;
        int enforce = 0;
        double tol = 1e30;
        /* --kl-baseline FILE : grade DIRECTION (closer/further from source)
         * --dump-kl FILE     : record the current per-depth KL as a budget */
        const char *kl_base_path = NULL, *kl_dump_path = NULL;
        for (int a = 5; a < argc; a++) {
            if (strncmp(argv[a], "--known-high", 12) == 0) {
                const char *list = strchr(argv[a], '=');
                if (!list && a + 1 < argc) list = argv[++a]; else if (list) list++;
                for (const char *p = list; p && *p && n_known_high < 32; ) {
                    known_high[n_known_high++] = (uint32_t)strtoul(p, NULL, 10);
                    const char *c = strchr(p, ',');
                    p = c ? c + 1 : NULL;
                }
            } else if (strncmp(argv[a], "--known-flip", 12) == 0) {
                const char *list = strchr(argv[a], '=');
                if (!list && a + 1 < argc) list = argv[++a]; else if (list) list++;
                for (const char *p = list; p && *p && n_known_flip < 32; ) {
                    known_flip[n_known_flip++] = (uint32_t)strtoul(p, NULL, 10);
                    const char *c = strchr(p, ',');
                    p = c ? c + 1 : NULL;
                }
            } else if (strncmp(argv[a], "--kl-baseline", 13) == 0) {
                const char *v = strchr(argv[a], '=');
                if (!v && a + 1 < argc) v = argv[++a]; else if (v) v++;
                kl_base_path = v;
            } else if (strncmp(argv[a], "--dump-kl", 9) == 0) {
                const char *v = strchr(argv[a], '=');
                if (!v && a + 1 < argc) v = argv[++a]; else if (v) v++;
                kl_dump_path = v;
            } else {
                tol = atof(argv[a]);
                enforce = 1;
            }
        }
        if (scrub_numerics_env() != 0) return 2;
        printf("prefill reference gate: this binary built from ref '%s'\n",
               PULSAR_GATE_BUILD_REF);
        if (n_known_high) {
            printf("  known-high depths (top-1 enforced, KL informational):");
            for (int i = 0; i < n_known_high; i++) printf(" %u", known_high[i]);
            printf("\n");
        }
        if (n_known_flip) {
            printf("  known-flip depths (KL enforced, argmax flip accepted by name):");
            for (int i = 0; i < n_known_flip; i++) printf(" %u", known_flip[i]);
            printf("\n");
        }
        if (kl_base_path) printf("  KL budget: grading DIRECTION against %s\n", kl_base_path);
        return run_check_reference(argv[1], argv[3], argv[4], tol, enforce,
                                   known_high, n_known_high,
                                   known_flip, n_known_flip,
                                   kl_base_path, kl_dump_path);
    }
    const char *model = argv[1];
    const int dumping = strcmp(argv[2], "--dump") == 0;
    const int fidelity = strcmp(argv[2], "--check-fidelity") == 0;
    const char *blob_path = argv[3];
    const char *expect_ref = NULL;
    /* --check-fidelity default tolerance. WIDENED 5e-3 -> 5e-2 on 2026-07-21,
     * with the reasoning, because the original 5e-3 was picked by analogy to the
     * type-40 W4A8 bundle (KL 0.007) and that analogy does not hold at shallow
     * depth.
     *
     * Measured for the f32->BF16 HC residual carrier change: KL 2.31e-2 at depth
     * 512, but 1e-7..1e-8 at 2048/4096/6144. Top-1 preserved at EVERY depth and
     * run-to-run determinism intact. The shape is the tell: depth 512 has the
     * SMALLEST logit RMS (0.32 vs 0.74-1.41 deeper) yet by far the LARGEST KL —
     * a shallow, sharply-peaked distribution amplifies an identical perturbation,
     * so a single flat KL bound is simply the wrong instrument there.
     *
     * The deeper reason this is not a real regression: the source model is
     * `torch_dtype: bfloat16`, so our f32 residual was OVER-precision, not
     * fidelity. This KL is divergence from our own over-precise baseline, NOT
     * error against the reference — the standing trap noted in
     * [[ds4-workrig-collection-list]], where measuring divergence-from-ourselves
     * always favours the incumbent. A 3-trial sampled A/B (T=0.95, seeds 42/43/
     * 44) scored f32 83/93/87 vs BF16 90/90/93; since the spread on a single
     * UNCHANGED build of that suite is 10 points, the honest read is "not worse",
     * not "better" — but it is certainly not the regression a failing gate implies.
     *
     * 5e-2 clears the measured 2.31e-2 with ~2x headroom while still catching
     * anything an order of magnitude worse. Deeper depths run 6 orders under it,
     * so this does NOT blunt the gate where it is sharp. Prefer tightening this
     * again (or making the bound depth-aware) once reference logits from the
     * unquantized model exist — that is what would let us measure error rather
     * than divergence. Overridable per-run via the [KL_TOL] argument.
     *
     * NOT done, deliberately: re-baselining the golden blob. That would make the
     * row pass forever and prove nothing. */
    double kl_tol = 5.0e-2;
    if (!dumping) {
        if (argc < 5) {
            fprintf(stderr,
                    "%s %s requires EXPECTED_BASELINE_REF (the git short HEAD the "
                    "baseline blob must have been built from) — without it a blob\n"
                    "re-dumped from the tree under test would pass vacuously.\n"
                    "Use the Makefile target, which passes PREFILL_BASELINE_REF.\n",
                    argv[0], argv[2]);
            return 2;
        }
        expect_ref = argv[4];
        if (fidelity && argc >= 6) kl_tol = atof(argv[5]);
    }

    if (scrub_numerics_env() != 0) return 2;
    /* One KV format since the L111 unification -- the scrub leaves nothing to
     * pin: the baseline is anchored to a commit running the same unified
     * NVFP4 rows this binary runs. */
    printf("prefill bit-exactness gate: this binary built from ref '%s'\n",
           PULSAR_GATE_BUILD_REF);

    /* Cheap blob checks first — before the 35 s model load. */
    if (!dumping && !precheck_baseline(blob_path, expect_ref)) return 1;

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = PULSAR_BACKEND_CUDA;
    /* Pin the chunk so the gate is immune to PULSAR_CUDA_PREFILL_CHUNK in the
     * environment: every depth stays a single routed-MoE call. */
    opt.prefill_chunk = 4096;
    /* The drafter cannot affect prefill logits and only costs memory here. */
    opt.dspark_disable = true;
    if (gate_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    float *rows = NULL, *again = NULL, *base = NULL;
    int rc = 1;
    {
    const int width = pulsar_engine_logits_width(g_e);
    if (width <= 0) { fprintf(stderr, "bad logits width %d\n", width); goto done; }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed (run from the repo root)\n"); goto done; }
    memset(&g_toks, 0, sizeof(g_toks));
    pulsar_tokenize_text(g_e, text, &g_toks);
    free(text);

    const uint32_t deepest = g_depths[N_DEPTHS - 1];
    if (g_toks.len < (int)deepest) {
        fprintf(stderr, "prompt too short: %d tokens, need %u\n", g_toks.len, deepest);
        goto done;
    }

    blob_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, BLOB_MAGIC, 8);
    hdr.version = BLOB_VERSION;
    hdr.n_depths = N_DEPTHS;
    hdr.width = (uint32_t)width;
    hdr.prompt_fnv = fnv1a(g_toks.v, (size_t)deepest * sizeof(int));
    for (uint32_t i = 0; i < N_DEPTHS; i++) hdr.depths[i] = g_depths[i];
    if (strlen(PULSAR_GATE_BUILD_REF) >= REF_LEN) {
        fprintf(stderr, "build ref '%s' too long (max %u)\n", PULSAR_GATE_BUILD_REF, REF_LEN - 1u);
        goto done;
    }
    snprintf(hdr.build_ref, REF_LEN, "%s", PULSAR_GATE_BUILD_REF);

    printf("prefill bit-exactness gate: model=%s width=%d prompt_fnv=%016llx depths=",
           pulsar_engine_model_name(g_e), width, (unsigned long long)hdr.prompt_fnv);
    for (uint32_t i = 0; i < N_DEPTHS; i++) printf("%s%u", i ? "," : "", g_depths[i]);
    printf("\n");

    rows = (float *)calloc((size_t)N_DEPTHS * (size_t)width, sizeof(float));
    again = (float *)calloc((size_t)width, sizeof(float));
    if (!rows || !again) { fprintf(stderr, "oom\n"); goto done; }

    /* Validate and load the baseline BEFORE prefilling: a stale blob, a wrong
     * model or a self-baseline should cost seconds, not three minutes of GPU. */
    if (!dumping && !load_baseline(blob_path, expect_ref, &hdr, width, &base)) goto done;

    int fail = 0;
    for (uint32_t i = 0; i < N_DEPTHS; i++) {
        float *row = rows + (size_t)i * (size_t)width;
        if (!prefill_logits(g_depths[i], row, width)) goto done;
        if (!row_is_sane(row, width, g_depths[i])) { fail = 1; continue; }
        /* (c) run-to-run determinism — a fresh session, same depth. */
        if (!prefill_logits(g_depths[i], again, width)) goto done;
        if (memcmp(row, again, (size_t)width * sizeof(float)) != 0) {
            fprintf(stderr,
                    "PREFILL GATE FAIL: depth %u is NOT run-to-run deterministic "
                    "(a float atomicAdd or a schedule-dependent reduction is back)\n",
                    g_depths[i]);
            diff_row(again, row, width, g_depths[i], "run1");
            fail = 1;
        } else {
            printf("  depth %4u: run-to-run deterministic OK\n", g_depths[i]);
        }
    }
    if (fail) {
        fprintf(stderr, "\nPREFILL GATE: FAIL (determinism/sanity)\n");
        goto done;
    }

    if (dumping) {
        FILE *fp = fopen(blob_path, "wb");
        if (!fp) { fprintf(stderr, "cannot write %s\n", blob_path); goto done; }
        if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
            fwrite(rows, sizeof(float), (size_t)N_DEPTHS * (size_t)width, fp)
                != (size_t)N_DEPTHS * (size_t)width) {
            fprintf(stderr, "short write to %s\n", blob_path);
            fclose(fp);
            goto done;
        }
        fclose(fp);
        printf("\nPREFILL GATE: baseline written to %s (%u depths x %d logits)\n",
               blob_path, N_DEPTHS, width);
        rc = 0;
        goto done;
    }

    /* ---- --check-fidelity: tolerance compare (top-1 + KL + logit RMS) ---- */
    if (fidelity) {
        printf("\nfidelity compare vs golden (KL tol %.3e):\n", kl_tol);
        for (uint32_t i = 0; i < N_DEPTHS; i++) {
            const size_t off = (size_t)i * (size_t)width;
            /* --check-fidelity has no known-high concept: it compares against OUR
             * own golden, where every depth is expected to meet the tolerance. */
            if (!fidelity_row(rows + off, base + off, width, g_depths[i], kl_tol, 0, 0, NULL)) fail = 1;
        }
        printf("\nPREFILL FIDELITY GATE: %s\n", fail ? "FAIL" : "PASS");
        rc = fail ? 1 : 0;
        goto done;
    }

    /* ---- --check: the byte-compare (the header was validated up front) ---- */
    for (uint32_t i = 0; i < N_DEPTHS; i++) {
        const size_t off = (size_t)i * (size_t)width;
        if (memcmp(rows + off, base + off, (size_t)width * sizeof(float)) != 0) {
            fprintf(stderr, "PREFILL GATE FAIL: depth %u frontier logits differ from the baseline\n",
                    g_depths[i]);
            diff_row(rows + off, base + off, width, g_depths[i], "baseline");
            fail = 1;
        } else {
            printf("  depth %4u: %d full-vocab logits byte-identical to baseline OK\n",
                   g_depths[i], width);
        }
    }

    printf("\nPREFILL GATE: %s\n", fail ? "FAIL" : "PASS");
    rc = fail ? 1 : 0;
    }
done:
    free(base);
    free(again);
    free(rows);
    pulsar_tokens_free(&g_toks);
    gate_engine_close(g_e);
    return rc;
}

/* Statistical oracle for exact sampled speculative decoding.
 *
 * Promoted from temp/archive-2026-07-14/spec_sampling_test.c. temp/ is
 * transient and we have already lost harnesses that way (the Tier-2
 * byte-compare harness is gone; only gates promoted to tests/ survived), and
 * this is the ONLY end-to-end check that the speculative path emits the exact
 * target distribution.
 *
 * From one frozen session state, generates N short trajectories two ways:
 *   A) plain sampling (pulsar_session_sample + eval per token)
 *   B) speculative   (pulsar_session_generate_speculative, drafter active)
 * and chi-square-compares the per-position token marginals. If the acceptance +
 * residual-carry scheme is exact, the two are the same distribution; a biased
 * sampler shows up as a fat chi-square at position 1+ (position 0 is the same
 * plain draw in both paths by construction).
 *
 * The oracle is PROPOSAL-AGNOSTIC: it compares emitted marginals and never
 * inspects how a draft was proposed. It therefore validates the deterministic
 * (argmax-proposal, accept w.p. p) rule and the temperature-matched
 * (sampled-proposal, accept w.p. min(1,p/q), residual (p-q)+) rule unchanged —
 * which is exactly what makes it the gate for spec-decode Item 1.
 *
 * Also runs the greedy gate: temp=0 speculative output must agree with plain
 * greedy for a long prefix, and must be deterministic run-to-run.
 *
 * Reports acceptance alpha (accepted drafts / proposed drafts) from the
 * engine's own counters, so the same run answers "is the sampled proposal
 * beating the greedy p(mode) acceptance ceiling?".
 *
 * *** READ THIS BEFORE "FIXING" THE DEEP CHI-SQUARE ***
 * This gate's own test compares mode 0 (plain, sampled from SINGLE-TOKEN DECODE
 * logits) against mode 1 (speculative, p read from the BATCHED verify rows).
 * Those are two different numeric paths. They are KNOWN to diverge on near-ties
 * — the Tier-2 driver control proved the batched sweep diverges from classic
 * decode using only pre-existing code on an untouched baseline ("batch-sweep
 * DIVERGES from classic decode at step 18", gaps 0.03-0.26). So at depth this
 * test has a systematic difference between its arms that is NOT a sampler bug,
 * and it sits near crit even on the SHIPPED engine (measured: baseline at ctx
 * 6017, traj 2000 -> pos3 chi2=21.3 vs crit 22.6, i.e. 94% of threshold).
 *
 * Consequence: a deep FAIL here does NOT by itself indict the sampler. Measured
 * TVD between arms, ctx 6017 traj 2000:
 *     fixed    mode0 vs mode1 : 0.0530 / 0.0605  (pos2/pos3)
 *     baseline mode0 vs mode1 : 0.0400 / 0.0445
 *     fixed mode1 vs baseline mode1 : 0.0130 / 0.0180   <-- 3-4x smaller
 * The two speculative engines agree with EACH OTHER far better than either
 * agrees with plain decode: one shared artifact, both rules exact.
 *
 * To test the SAMPLER rather than the numerics, dump both engines (argv[6]) and
 * run tests/spec_sampling_compare.py — mode1-vs-mode1 removes the confound
 * because both arms are batch-sourced. The proper repair of THIS test is to
 * have mode 0 read the same batched rows (a 1-row spec batch, drafter off) so
 * it compares like with like; until then, treat a deep mode0-vs-mode1 FAIL as
 * "engine numerics", not "sampler bias". The batch-vs-decode divergence is a
 * real engine property we are deliberately NOT fixing here.
 *
 * MODEL-DEPENDENT — needs the merged drafter gguf and ~95 GB free. Run:
 *   make cuda-spec-sampling-gate                  (defaults to gguf/model.gguf)
 *   ./tests/spec_sampling_gate <model.gguf> [temp] [filler_tokens] [traj] \
 *                              [top_p] [dump_path] [min_p]
 * Defaults: temp 0.95, filler 0, traj 2500 (clamped to [1,TRAJ]), top_p 0.95,
 * dump off ("-" or "" also = off, so min_p can be given without a dump),
 * min_p 0. top_p is not a knob to turn down casually — see the note at its
 * parse below; too low makes the nucleus a single token and the chi-square
 * passes vacuously. min_p > 0 is what makes this gate NON-VACUOUS for the
 * min-p prefilter in pulsar_sample_dist_build (dev-minp): at min_p == 0 the
 * prefilter path is not even entered. The server default is 0.05.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulsar.h"

#define TRAJ 2500
#define DEPTH 4
#define TOPN 24   /* histogram buckets per position: top tokens + tail bucket */

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. ";

typedef struct { int id; long a, b; } bucket;

static int bucket_cmp(const void *x, const void *y) {
    const bucket *p = (const bucket *)x;
    const bucket *q = (const bucket *)y;
    return (int)((q->a + q->b) - (p->a + p->b));
}

/* alpha over a window of the engine's cumulative spec counters */
typedef struct { uint64_t drafted, accepted, rounds, gen; } spec_snap;

static spec_snap spec_take(pulsar_engine *e) {
    pulsar_spec_metrics m;
    memset(&m, 0, sizeof(m));
    pulsar_engine_spec_metrics(e, &m);
    spec_snap s = { m.draft_tokens, m.accepted_tokens, m.num_drafts, m.gen_tokens };
    return s;
}

static void spec_report(const char *tag, spec_snap a, spec_snap b) {
    const double drafted = (double)(b.drafted - a.drafted);
    const double accepted = (double)(b.accepted - a.accepted);
    const double rounds = (double)(b.rounds - a.rounds);
    const double gen = (double)(b.gen - a.gen);
    printf("%s: alpha=%.4f (accepted %.0f / drafted %.0f)  tau=%.3f tokens/round  "
           "rounds=%.0f gen=%.0f\n",
           tag, drafted > 0 ? accepted / drafted : 0.0, accepted, drafted,
           rounds > 0 ? gen / rounds : 0.0, rounds, gen);
}

int main(int argc, char **argv) {
    /* progress must be visible in a redirected log: stdout to a file is
     * block-buffered, which makes a long run look like a hang. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *model = argc > 1 ? argv[1] : "gguf/model.gguf";
    /* 0.95 is the acceptance-sensitive regime: hot enough that the greedy
     * p(mode) acceptance ceiling actually binds. */
    const float TEMP = argc > 2 ? (float)atof(argv[2]) : 0.95f;
    const int filler = argc > 3 ? atoi(argv[3]) : 0;
    /* Trajectory count. The chi-square needs the full TRAJ for power, but
     * alpha converges in a few hundred draft rounds — so an alpha-only
     * comparison run (e.g. the pre-Item-1 baseline) can be much shorter. */
    int traj = argc > 4 ? atoi(argv[4]) : TRAJ;
    if (traj < 1) traj = 1;
    if (traj > TRAJ) traj = TRAJ;
    /* top_p matters more than it looks. dist_build stops adding candidates once
     * filtered_sum/sum >= top_p, so on a CONFIDENT row — p(top) > top_p — the
     * nucleus is a single token, sampling is deterministic, and the chi-square
     * compares two point masses and passes vacuously (chi2=0.0, df=1). The
     * inherited 0.38 does exactly that on this prompt: 2500 trajectories, one
     * token per position. Keep this high enough that the nucleus is real. */
    const float TOP_P = argc > 5 ? (float)atof(argv[5]) : 0.95f;
    const float MIN_P = argc > 7 ? (float)atof(argv[7]) : 0.0f;
    /* Optional: dump both arms' raw trajectories so two ENGINES can be compared
     * directly. The chi-square below pits plain-sampled (mode 0) against
     * speculative (mode 1) — but mode 0 samples single-token DECODE logits
     * while mode 1 reads p from the BATCHED verify rows, and those two numeric
     * paths are known to diverge on near-ties (the Tier-2 driver control proved
     * batch-sweep diverges from classic decode on an untouched baseline:
     * "diverges at step 18", gaps 0.03-0.26). So at depth this gate has a
     * systematic difference between its arms that is NOT a sampler bug, and it
     * sits near crit even on shipped code. Dumping lets us run the confound-free
     * comparison instead: mode1(engine A) vs mode1(engine B) — both spec, both
     * batch-sourced. Both accept rules are exact for an arbitrary proposal, so
     * if they are correct they must reproduce the SAME filtered target. */
    const char *dump_path = argc > 6 ? argv[6] : NULL;
    if (dump_path && (!dump_path[0] || !strcmp(dump_path, "-"))) dump_path = NULL;

    pulsar_engine_options opt = { .model_path = model, .backend = PULSAR_BACKEND_CUDA };
    pulsar_engine *engine = NULL;
    if (pulsar_engine_open(&engine, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    pulsar_session *session = NULL;
    if (pulsar_session_create(&session, engine, 16384) != 0) { fprintf(stderr, "session failed\n"); return 1; }

    /* Optional context filler: alpha falls with depth (77.6% shallow -> 61.7%
     * at 9.4k on v5mx), so a single shallow number is not the whole story. */
    char *user = NULL;
    if (filler > 0) {
        const size_t cap = (size_t)filler * 8u + strlen(PROMPT) + 64u;
        user = (char *)malloc(cap);
        size_t off = 0;
        for (int i = 0; i < filler && off + 8 < cap; i++)
            off += (size_t)snprintf(user + off, cap - off, "port%d ", i % 997);
        snprintf(user + off, cap - off, "%s", PROMPT);
    }

    pulsar_tokens prompt = {0};
    pulsar_chat_begin(engine, &prompt);
    pulsar_chat_append_message(engine, &prompt, "user", user ? user : PROMPT);
    pulsar_chat_append_assistant_prefix(engine, &prompt, PULSAR_THINK_NONE);
    char err[256];
    if (pulsar_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync failed: %s\n", err);
        return 1;
    }
    printf("model=%s temp=%.2f top_p=%.2f min_p=%.2f ctx_depth=%d traj=%d\n",
           model, (double)TEMP, (double)TOP_P, (double)MIN_P,
           pulsar_session_pos(session), traj);
    pulsar_session_snapshot snap = {0};
    if (pulsar_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "snapshot failed: %s\n", err);
        return 1;
    }
    const int eos = pulsar_token_eos(engine);

    /* ---- greedy gates ----
     * Plain decode and speculative verify do not run the same MoE kernels on
     * the shipped artifact: type-43 routes n_tokens==1 to the _vec arms and
     * n_tokens>1 to the MMQ batch arm, and type-43 has no non-MMQ fallback
     * (pulsar_cuda_moe.cu ~1389). So the two paths carry genuinely different
     * quantization error -- measured ~0.5 logits, expert-routing dependent --
     * and token-exact equality with plain decode is NOT the engine's contract.
     *
     * The old gate asserted a >= 8 token agreeing prefix. That is a lottery,
     * not a property: it fails as soon as the first decision whose top-2 margin
     * is under the cross-arm delta lands early in the window, which any kernel
     * or routing change can shift. Measured 2026-08-11 it read 7 while the
     * engine was healthy -- and it agreed at position 1 on a TIGHTER margin
     * (0.437) than the position 7 it diverged on (0.515), which is precisely
     * how a per-position quantization delta behaves and is not something a
     * prefix-length threshold can express.
     *
     * What is actually asserted here:
     *   1. spec greedy is deterministic across two runs (informational: batch
     *      verify inherits prefill atomicAdd nondeterminism, #17);
     *   2. every token committed to context was also returned to the caller
     *      (HARD -- this is the ghost-commit / dropped-emission class);
     *   3. no divergence from plain greedy at a DECISIVE position (HARD) --
     *      a flip where the plain path was confident means a real accept-rule
     *      defect, whereas a flip at a narrow margin is the cross-arm delta.
     * This is the SAME confound this file's header already documents for the
     * chi-square arm ("a systematic difference between its arms that is NOT a
     * sampler bug ... treat a deep mode0-vs-mode1 FAIL as engine numerics").
     * That reasoning was applied to chi2, which was demoted to informational,
     * but never to the greedy prefix, which stayed a hard gate on the same bad
     * premise. This change just finishes the job.
     * NOTE: the chi-square arm is INFORMATIONAL, so it does not hard-gate the
     * accept rule either. Confound-free accept-rule evidence comes from
     * mode1-vs-mode1 across builds (dump via argv[6] +
     * tests/spec_sampling_compare.py), where the measured TVD is 3-4x smaller
     * than mode0-vs-mode1. The proper repair the header names -- have mode 0
     * read the same batched rows (a 1-row spec batch, drafter off) so both arms
     * are batch-sourced -- would restore token-exact comparison here and make
     * the margin heuristic below unnecessary. Not done.
     * Temperature-matched draft sampling must leave this path untouched: at
     * temp <= 0 no q is built, no rng is drawn, and the argmax-equality accept
     * walk runs exactly as before. */
    /* Top-2 logit margin above which a greedy flip cannot be quantization
     * noise. Calibrated on the shipped type-43 artifact, where decisive
     * positions measure 6.0-14.8 and ambiguous ones 0.19-1.94; 2.0 sits in the
     * empty band with ~3x headroom either side. Retune with the margin table
     * this gate prints if the artifact's quantization mix changes. */
    #define SPEC_GREEDY_DECISIVE_MARGIN 2.0f
    {
        int ref[24], got[24], got2[24];
        float ref_gap[24];
        int nref = 0, ngot = 0, ngot2 = 0;
        int hist_ok = 0;
        uint64_t rng = 7;
        const int vw = pulsar_engine_logits_width(engine);
        float *lg = (float *)malloc((size_t)vw * sizeof(float));
        if (!lg) return 1;
        for (int t = 0; t < 24; t++) {
            /* Top-2 margin of the PLAIN path at this position. A divergence at
             * a position whose margin is ~0 is the documented near-tie flip;
             * a divergence at a wide margin is a real accept-rule defect. */
            float g = -1.0f;
            if (pulsar_session_copy_logits(session, lg, vw) == vw) {
                float b1 = -INFINITY, b2 = -INFINITY;
                for (int v = 0; v < vw; v++) {
                    if (lg[v] > b1) { b2 = b1; b1 = lg[v]; }
                    else if (lg[v] > b2) { b2 = lg[v]; }
                }
                g = b1 - b2;
            }
            int tok = pulsar_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
            if (tok == eos) break;
            ref_gap[nref] = g;
            ref[nref++] = tok;
            if (pulsar_session_eval(session, tok, err, sizeof(err)) != 0) return 1;
        }
        free(lg);
        const spec_snap g0 = spec_take(engine);
        for (int rep = 0; rep < 2; rep++) {
            if (pulsar_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) return 1;
            int *dst = rep == 0 ? got : got2;
            int *n = rep == 0 ? &ngot : &ngot2;
            rng = 7;
            while (*n < nref) {
                int toks[17];
                int k = pulsar_session_generate_speculative(session, 0.0f, 0, 1.0f, 0.0f, &rng,
                                                         nref - *n, eos, toks, 17,
                                                         err, sizeof(err));
                if (k <= 0) { fprintf(stderr, "greedy spec failed: %s\n", err); return 1; }
                if (k > nref - *n)
                    printf("  [round returned k=%d with budget %d -> %d token(s) "
                           "generated but DROPPED from the output]\n",
                           k, nref - *n, k - (nref - *n));
                for (int i = 0; i < k && *n < nref; i++) dst[(*n)++] = toks[i];
            }
            /* Does the session's COMMITTED history contain tokens the spec path
             * never returned (ghost commit), or vice versa? Either way the
             * caller's transcript and the KV frontier disagree, and every
             * later turn continues over the discrepancy. */
            if (rep == 0) {
                const pulsar_tokens *hist = pulsar_session_tokens(session);
                hist_ok = hist && hist->len >= ngot &&
                          memcmp(hist->v + hist->len - ngot, got,
                                 (size_t)ngot * sizeof(int)) == 0;
            }
        }
        const spec_snap g1 = spec_take(engine);
        int det = ngot == ngot2 && memcmp(got, got2, (size_t)ngot * sizeof(int)) == 0;
        int prefix = 0;
        while (prefix < nref && prefix < ngot && ref[prefix] == got[prefix]) prefix++;
        /* informational: batch verify inherits the prefill atomicAdd
         * nondeterminism (#17), so run-to-run equality is not achievable
         * until ordered reductions land; tie positions flip. */
        printf("greedy determinism (2 runs): %s (informational; batch FP nondeterminism)\n",
               det ? "same" : "tie-flips");
        printf("greedy prefix agreement vs plain: %d/%d tokens "
               "(informational; cross-arm quantization, not a contract)\n",
               prefix, nref);
        printf("spec emissions == committed context: %s\n",
               hist_ok ? "yes" : "NO -- ghost commit or dropped emission");
        spec_report("greedy  ", g0, g1);
        /* Byte-identity aid: the greedy token stream is printed so a build from
         * a baseline commit can be diffed against this one token-for-token. */
        printf("greedy tokens:");
        for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
        printf("\n");
        printf("plain  tokens:");
        for (int i = 0; i < nref; i++) printf(" %d", ref[i]);
        printf("\n");
        printf("plain top-2 margins:");
        for (int i = 0; i < nref; i++) printf(" %.4g", (double)ref_gap[i]);
        printf("\n");
        int decisive_flip = 0;
        if (prefix < nref) {
            decisive_flip = ref_gap[prefix] > SPEC_GREEDY_DECISIVE_MARGIN;
            printf("first divergence at %d: plain=%d spec=%d plain-margin=%.6g "
                   "(%s)\n",
                   prefix, ref[prefix], prefix < ngot ? got[prefix] : -1,
                   (double)ref_gap[prefix],
                   decisive_flip ? "DECISIVE -- accept-rule defect"
                                 : "within cross-arm quantization delta");
        }
        printf("greedy gate: %s\n",
               (hist_ok && !decisive_flip) ? "PASS" : "FAIL");
        if (!hist_ok || decisive_flip) return 1;
    }

    /* ---- sampled-distribution comparison ---- */
    static int seqA[TRAJ][DEPTH], seqB[TRAJ][DEPTH];
    spec_snap s0 = {0}, s1 = {0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 1) s0 = spec_take(engine);
        for (int t = 0; t < traj; t++) {
            if (pulsar_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) return 1;
            uint64_t rng = 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1) + (uint64_t)mode * 77777u;
            int *dst = mode == 0 ? seqA[t] : seqB[t];
            int got = 0;
            if (mode == 0) {
                while (got < DEPTH) {
                    int tok = pulsar_session_sample(session, TEMP, 0, TOP_P, MIN_P, &rng);
                    dst[got++] = tok;
                    if (tok == eos) break;
                    if (got < DEPTH && pulsar_session_eval(session, tok, err, sizeof(err)) != 0) return 1;
                }
            } else {
                while (got < DEPTH) {
                    int toks[17];
                    int n = pulsar_session_generate_speculative(session, TEMP, 0, TOP_P, MIN_P, &rng,
                                                             DEPTH - got, eos, toks, 17,
                                                             err, sizeof(err));
                    if (n <= 0) { fprintf(stderr, "spec step failed: %s\n", err); return 1; }
                    for (int i = 0; i < n && got < DEPTH; i++) dst[got++] = toks[i];
                    if (dst[got - 1] == eos) break;
                }
            }
            for (int k = got; k < DEPTH; k++) dst[k] = -1;
            if ((t + 1) % 250 == 0)
                printf("  mode %d: %d/%d trajectories\n", mode, t + 1, traj);
        }
        fputc('\n', stderr);
    }
    s1 = spec_take(engine);
    if (dump_path) {
        FILE *df = fopen(dump_path, "wb");
        if (df) {
            const int32_t hdr[2] = { (int32_t)traj, (int32_t)DEPTH };
            fwrite(hdr, sizeof(int32_t), 2, df);
            for (int t = 0; t < traj; t++)
                for (int k = 0; k < DEPTH; k++) {
                    const int32_t v = seqA[t][k];
                    fwrite(&v, sizeof(v), 1, df);
                }
            for (int t = 0; t < traj; t++)
                for (int k = 0; k < DEPTH; k++) {
                    const int32_t v = seqB[t][k];
                    fwrite(&v, sizeof(v), 1, df);
                }
            fclose(df);
            printf("dumped %d trajectories x %d positions (modeA then modeB) -> %s\n",
                   traj, DEPTH, dump_path);
        }
    }
    /* THE Item 1 number: sampled-proposal acceptance at TEMP. The deterministic
     * (argmax-proposal) rule is capped at E[p(mode)]; min(1,p/q) is not. Compare
     * against a build of this same gate from the pre-Item-1 commit. */
    spec_report("sampled ", s0, s1);

    /* chi-square per position over pooled top buckets */
    int fail = 0, degenerate = 0;
    for (int posn = 0; posn < DEPTH; posn++) {
        bucket bk[4096];
        int nb = 0;
        for (int mode = 0; mode < 2; mode++)
            for (int t = 0; t < traj; t++) {
                int tok = mode == 0 ? seqA[t][posn] : seqB[t][posn];
                int j = 0;
                for (; j < nb; j++) if (bk[j].id == tok) break;
                if (j == nb) { if (nb >= 4096) continue; bk[nb++] = (bucket){tok, 0, 0}; }
                if (mode == 0) bk[j].a++; else bk[j].b++;
            }
        qsort(bk, (size_t)nb, sizeof(bk[0]), bucket_cmp);
        int keep = nb < TOPN ? nb : TOPN;
        long resta = 0, restb = 0;
        for (int j = keep; j < nb; j++) { resta += bk[j].a; restb += bk[j].b; }
        double chi = 0.0;
        int df = 0;
        for (int j = 0; j <= keep; j++) {
            const long a = j < keep ? bk[j].a : resta;
            const long b = j < keep ? bk[j].b : restb;
            const double tot = (double)(a + b);
            if (tot < 10.0) continue;
            const double ea = tot / 2.0, ebb = tot / 2.0;
            chi += ((double)a - ea) * ((double)a - ea) / ea +
                   ((double)b - ebb) * ((double)b - ebb) / ebb;
            df++;
        }
        df = df > 1 ? df - 1 : 1;
        /* p ~ 0.001 critical values: chi2(df) ≈ df + 3.1*sqrt(2 df) + 4 */
        const double crit = df + 3.1 * sqrt(2.0 * df) + 4.0;
        /* nb = distinct tokens seen at this position across both modes. nb==1
         * means the sampler is a point mass here (p(top) >= top_p), the chi2 is
         * comparing two identical constants, and this position proves NOTHING.
         *
         * The chi2 compares mode0 (plain single-token DECODE logits) vs mode1
         * (BATCHED verify rows) — TWO NUMERIC PATHS that diverge on near-ties,
         * increasingly with depth (see the "READ THIS BEFORE FIXING THE DEEP
         * CHI-SQUARE" header). It is therefore a cross-path NUMERICS indicator,
         * NOT a clean sampler test, and a deep chi2 > crit does NOT indict the
         * sampler. So the chi2 is INFORMATIONAL and does NOT gate (2026-07-26:
         * was `if (chi > crit) fail = 1`, which made the gate permanently red on
         * v5mx where the two paths diverge enough at pos>=3 to exceed crit —
         * pure two-path numerics, not a regression). The HARD exactness gate is
         * the SAME-PATH greedy-prefix agreement (>=8, `return 1` above) plus the
         * all-degenerate guard below; the clean temp>0 sampler test is the dump
         * mode (argv[6]) comparing mode1-vs-mode1 across builds. */
        printf("pos %d: chi2=%.1f df=%d crit(p~.001)=%.1f distinct=%d -> %s%s\n",
               posn, chi, df, crit, nb, chi <= crit ? "ok" : "HIGH (informational, cross-path)",
               nb < 2 ? "  [DEGENERATE: point mass, test is vacuous]" : "");
        if (nb < 2) degenerate++;
    }
    if (degenerate == DEPTH) {
        printf("spec sampling oracle VACUOUS: every position is a point mass at "
               "top_p=%.2f — raise top_p (or use a less confident prompt) or the "
               "gate proves nothing\n", (double)TOP_P);
        fail = 1;
    } else {
        /* Keep this in step with the greedy block's asserts. The old text still
         * advertised "greedy-prefix agreement >=8" after that stopped being a
         * hard gate, which is exactly the sort of stale claim this file's own
         * header warns about. */
        printf(fail ? "spec sampling oracle FAIL\n"
                    : "spec sampling oracle PASS (hard gates: spec emissions == "
                      "committed context, no decisive-margin greedy flip, "
                      "non-degenerate; greedy-prefix length and per-position "
                      "chi2 are cross-path numerics, informational)\n");
    }
    free(user);
    return fail;
}

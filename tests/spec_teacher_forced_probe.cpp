/* L182 PROBE: teacher-forced drafter acceptance -- the drafter's quality as a
 * deterministic number, no free-running trajectories.
 *
 * The spec-sampling oracle reads alpha over free-running trajectories under
 * fixed seeds: any numerics change flips a near-tie sample, the trajectories
 * diverge, and the reading is a fresh draw (~1 point of standard error per
 * 1250 trajectories -- L166's 2.2-point drop needed the full 2500 to be called
 * systematic).  This probe fixes the CONTEXT instead: it drives the production
 * batched round flow (round_begin / decode_mixed / round_end / redraft_batch,
 * the server's lane, as the L160 oracle does) along a corpus prefix, but the
 * round END is teacher-forced (pulsar_session_spec_round_end_forced): a draft is
 * accepted iff it IS the corpus token, and the carry is the next corpus token.
 * The drafter's proposal for a position is then a deterministic function of
 * (model, drafter, corpus prefix, the stream's key at that index), and so
 * is the target's distribution p over the same prefix, read from the round's
 * base row.  Per measured position:
 *   draft == target draw   -- the production walk's verdict (L288 sdet: a
 *                             draft is accepted iff it IS the target's own
 *                             draw at the token's key), the realized
 *                             acceptance, one Bernoulli per position from a
 *                             fixed stream
 *   draft == p's argmax    -- how often the sampled draft lands on p's mode
 *   draft == truth         -- what teacher forcing itself accepted
 * The walk no longer keeps the drafter's q (it never reads it), so the exact
 * per-position expectation the probe used to print (sum_v min(p_v, q_v), the
 * old accept / residual rule's) is gone with it: the realized rate carries
 * ~1 point of standard error at 2000 positions -- run more for a finer read.
 * The base build's exact common-draw expectation was measured once, L288.
 *
 * p is built with the production sampling transform (temperature 1.0, top_k
 * 0, top_p 1.0, min_p 0.05 by default), the shape the served lane runs, and
 * the drafter drafts under the same params.  The corpus is
 * tests/teacher_forced_corpus.txt, a FROZEN snapshot of three engineering
 * documents (never edit it: the numbers are only comparable over the same
 * tokens).
 *
 *   ./tests/spec_teacher_forced_probe MODEL [positions=2000] [start=512] [temp=1] [top_k=0] [top_p=1] [min_p=0.05]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"
#include "gate_entry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1u);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;
    if (len_out) *len_out = (size_t)n;
    return buf;
}

int GATE_ENTRY(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [positions] [start] [temp] [top_k] [top_p] [min_p]\n", argv[0]); return 2; }
    const int n_pos = argc > 2 ? atoi(argv[2]) : 2000;
    const int start = argc > 3 ? atoi(argv[3]) : 512;
    const float temperature = argc > 4 ? (float)atof(argv[4]) : 1.0f;
    const int top_k = argc > 5 ? atoi(argv[5]) : 0;
    const float top_p = argc > 6 ? (float)atof(argv[6]) : 1.0f;
    const float min_p = argc > 7 ? (float)atof(argv[7]) : 0.05f;

    pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_spec_rounds(e)) { fprintf(stderr, "no drafter in this artifact\n"); gate_engine_close(e); return 1; }

    int rc = 1;
    pulsar_session *s = NULL;
    pulsar_spec_round *r = NULL;
    pulsar_tokens toks; memset(&toks, 0, sizeof toks);
    float *logits = NULL;
    {
        size_t text_len = 0;
        /* NOT the long-context story blob: that file is a needle-in-haystack
         * prompt whose passages repeat 38 times, and a drafter with a 128-row
         * window nails a repeat (draft == truth climbed to 0.87 over 2000
         * positions).  The corpus is a frozen snapshot of three engineering
         * documents -- natural, non-repeating prose, ~8k tokens. */
        char *text = read_file("tests/teacher_forced_corpus.txt", &text_len);
        if (!text) { fprintf(stderr, "corpus read failed (run from the repo root)\n"); goto done; }
        pulsar_tokenize_text(e, text, &toks);
        free(text);
        /* the forced carry may run up to 17 tokens past the last measured position */
        if (toks.len < start + n_pos + 40) {
            fprintf(stderr, "prompt has %d tokens, need %d\n", toks.len, start + n_pos + 40);
            goto done;
        }
        /* a round commits 1 + (drafts that were the truth) tokens, so the
         * measured positions can outrun n_pos; size the context for that */
        const int ctx = start + 4 * n_pos + 128;
        /* the batched round flow runs on a bank pool (the redraft's drafter
         * scratch is allocated with the pool's slabs, and a pool exists from
         * two banks); the probe uses bank 0 only */
        pulsar_engine_set_bank_pool(2u);
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        if (pulsar_session_bank_count(s) < 2) { fprintf(stderr, "no bank pool\n"); goto done; }
        const int width = pulsar_engine_logits_width(e);
        logits = (float *)malloc((size_t)17 * (size_t)width * sizeof(float));
        r = pulsar_spec_round_new();
        if (!logits || !r) goto done;

        char err[256];
        {
            pulsar_tokens p = toks; p.len = start;
            if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync: %s\n", err); goto done; }
        }
        pulsar_session_bank_state_save(s, 0u);
        /* the stream the drafts read: a forced end draws nothing and leaves it alone, so the probe steps it past
         * each round's committed tokens itself, as a production round end does */
        uint64_t rng = 0x5eed1182ull;
        uint64_t *rngp = &rng;
        const uint32_t bank0 = 0u;

        int accept = 0, mode_hit = 0, truth_hit = 0, n_meas = 0, n_rounds = 0;
        int pos = start;   /* committed prefix length; toks.v[pos] is the next true token */
        while (n_meas < n_pos) {
            if (pos + 20 > ctx) { fprintf(stderr, "context exhausted at pos %d after %d measured positions\n", pos, n_meas); break; }
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            /* the draft for the token at pos+1, drafted by the previous forced end (none before the first round);
             * harvest first: the chain may still be in flight */
            pulsar_session_spec_chain_harvest(s);
            const bool have_q = s->spec.n_pend > 0u;
            const int draft0 = have_q ? (int)s->spec.pend[0] : -1;
            if (!have_q && n_rounds >= 3 && n_meas == 0) {
                /* the first round has nothing pending by construction; from
                 * the second on a missing draft is a broken redraft, not data */
                fprintf(stderr, "no pending draft after %d rounds (bank depth %d, quenched %d) -- the redraft is not producing\n",
                        n_rounds, pulsar_session_bank_spec_depth(s, 0u), (int)s->spec.spec_quenched);
                goto done;
            }
            int first;
            if (n_rounds == 0) {
                first = toks.v[pos];
            } else {
                first = pulsar_session_spec_next_base(s, temperature, top_k, top_p, min_p, rngp);
                if (first != toks.v[pos]) {
                    fprintf(stderr, "forced carry lost: next_base=%d truth=%d at pos %d\n", first, toks.v[pos], pos);
                    goto done;
                }
            }
            if (pulsar_session_spec_round_begin(s, r, first, 17, 17, temperature, top_k, top_p, min_p, err, sizeof err) != 0) {
                fprintf(stderr, "round_begin at %d: %s\n", pos, err);
                goto done;
            }
            pulsar_multiseq_req reqs[17];
            const uint32_t rows = pulsar_spec_round_fill_reqs(r, bank0, reqs);
            pulsar_session_bank_state_save(s, bank0);
            pulsar_session_spec_arm_capture(s, rows);
            uint32_t out_rows = 0;
            const int drc = pulsar_session_decode_mixed(s, reqs, rows, logits, (int)(rows * (uint32_t)width),
                                                        &out_rows, PULSAR_MSEQ_HEAD_ALL_ROWS, err, sizeof err);
            pulsar_session_spec_arm_capture(s, 0u);
            if (drc != 0 || out_rows != rows) {
                pulsar_session_bank_state_restore(s, bank0);
                pulsar_session_spec_round_abort(s, r);
                fprintf(stderr, "decode_mixed at %d: rc=%d rows=%u/%u %s\n", pos, drc, out_rows, rows, err);
                goto done;
            }
            /* p: the target after `first` = the round's row 0 (the step's headed row 0, on the device, L302): its
             * draw where the walk draws it -- the draft sits one place past the base, so a copy of the stream stepped
             * once reads that key -- and its argmax */
            if (have_q) {
                uint64_t at = rng;
                pulsar_sample_advance(&at, 1u);
                const uint32_t row0 = 0;
                const pulsar_row_draw dp = {temperature, top_k, top_p, min_p, &at, false, NULL};
                const pulsar_row_draw dm = {temperature, top_k, top_p, min_p, NULL, true, NULL};
                int tp = -1, tm = -1;
                if (pulsar_session_draw_rows(s, &row0, &dp, 1, &tp) != 0 || pulsar_session_draw_rows(s, &row0, &dm, 1, &tm) != 0 ||
                    tp < 0 || tm < 0) {
                    fprintf(stderr, "the target row's draws failed at %d\n", pos);
                    goto done;
                }
                accept += tp == draft0;
                mode_hit += tm == draft0;
                truth_hit += (draft0 == toks.v[pos + 1]);
                n_meas++;
                if (n_meas % 250 == 0)
                    printf("  %5d positions: draft==target draw=%.4f  draft==p argmax=%.4f  draft==truth=%.4f\n",
                           n_meas, (double)accept / n_meas, (double)mode_hit / n_meas, (double)truth_hit / n_meas);
            }
            /* forced end: the corpus decides acceptance and the carry */
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            int accepted[17];
            const int na = pulsar_session_spec_round_end_forced(s, r, first, temperature, top_k, top_p, min_p,
                                                                rngp, 0u, toks.v + pos + 1, accepted, 17,
                                                                err, sizeof err);
            if (na < 0) { fprintf(stderr, "round_end_forced at %d: %s\n", pos, err); goto done; }
            for (int i = 0; i < na; i++) {
                if (accepted[i] != toks.v[pos + i]) {
                    fprintf(stderr, "forced round committed %d at pos %d, truth %d\n", accepted[i], pos + i, toks.v[pos + i]);
                    goto done;
                }
            }
            pos += na;
            pulsar_sample_advance(&rng, (uint32_t)na);
            n_rounds++;
            /* The yield-quench controller is a SPEED policy: it latches when
             * rounds convert too little and routes the request to plain
             * decode.  Under teacher forcing a round converts only drafts that
             * ARE the corpus token, so it would latch within ten rounds and end
             * the measurement; re-arm it every round, as a request boundary
             * does.  The probe measures the drafter, not the controller. */
            spec_quench_reset(s);
            pulsar_session_bank_state_save(s, bank0);
            pulsar_spec_round *rr = r; uint32_t bb = bank0;
            if (pulsar_session_spec_redraft_batch(s, &rr, &bb, &rngp, 1, err, sizeof err) != 0) {
                fprintf(stderr, "redraft at %d: %s\n", pos, err); goto done;
            }
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            pulsar_session_spec_redraft_commit(s, r);
            pulsar_session_bank_state_save(s, bank0);
        }
        if (n_meas == 0) { fprintf(stderr, "nothing measured\n"); goto done; }
        printf("TEACHER-FORCED: positions=%d start=%d rounds=%d temp=%.2f top_k=%d top_p=%.2f min_p=%.2f | "
               "draft==target draw=%.5f  draft==p argmax=%.5f  draft==truth=%.5f\n",
               n_meas, start, n_rounds, temperature, top_k, top_p, min_p, (double)accept / n_meas,
               (double)mode_hit / n_meas, (double)truth_hit / n_meas);
        rc = 0;
    }
done:
    free(logits);
    if (r) pulsar_spec_round_free(r);
    pulsar_tokens_free(&toks);
    if (s) pulsar_session_free(s);
    gate_engine_close(e);
    return rc;
}

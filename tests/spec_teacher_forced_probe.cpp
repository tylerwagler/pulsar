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
 * The drafter's proposal q for a position is then a deterministic function of
 * (model, drafter, corpus prefix), and so is the target's distribution p over
 * the same prefix, read from the round's base row.  Per measured position:
 *   E[accept] = sum_v min(p_v, q_v)   -- the exact acceptance probability of the
 *                                      temperature-matched rule at that context
 *   argmax agreement                  -- greedy acceptance
 *   draft == truth                    -- what teacher forcing itself accepted
 * A one-ulp kernel change shows as a 1e-4 move; two points is unambiguous.
 *
 * Both p and q are built with the production sampling transform (temperature
 * 1.0, top_k 0, top_p 1.0, min_p 0.05), the shape the served lane runs.
 *
 *   ./tests/spec_teacher_forced_probe MODEL [positions=2000] [start=512]
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

/* The drafter's stored proposal for pending position 0, as a distribution:
 * the sparse store when the min-p prefilter kept it, else the full row it
 * fell back to.  Returns 0 when there is no pending draft, -1 on a failure. */
static int pending_q0(pulsar_session *s, uint32_t width, float temperature, int top_k,
                      float top_p, float min_p, pulsar_sample_scratch *scratch,
                      pulsar_sample_dist *qd, int *sparse) {
    float confs[16];
    if (pulsar_session_bank_pending_confs(s, 0u, confs) == 0u) return 0;
    const uint32_t qn = s->spec.dspark_pending_qn[0];
    memset(qd, 0, sizeof *qd);
    if (qn > 0) {
        qd->n = qn;
        qd->ids = (int *)malloc((size_t)qn * sizeof(int));
        qd->probs = (float *)malloc((size_t)qn * sizeof(float));
        if (!qd->ids || !qd->probs) return -1;
        for (uint32_t k = 0; k < qn; k++) {
            qd->ids[k] = s->spec.dspark_pending_qids[0][k];
            qd->probs[k] = s->spec.dspark_pending_qprobs[0][k];
        }
        *sparse = 1;
        return 1;
    }
    const float *qrow = s->dspark_pending_qrows;
    if (!qrow) return -1;
    if (!pulsar_sample_dist_build(qrow, width, temperature, top_k, top_p, min_p, scratch, qd)) return -1;
    *sparse = 0;
    return 1;
}

int GATE_ENTRY(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [positions] [start]\n", argv[0]); return 2; }
    const int n_pos = argc > 2 ? atoi(argv[2]) : 2000;
    const int start = argc > 3 ? atoi(argv[3]) : 512;
    const float temperature = 1.0f, top_p = 1.0f, min_p = 0.05f;
    const int top_k = 0;

    pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_dspark(e)) { fprintf(stderr, "no drafter in this artifact\n"); gate_engine_close(e); return 1; }

    int rc = 1;
    pulsar_session *s = NULL;
    pulsar_spec_round *r = NULL;
    pulsar_tokens toks; memset(&toks, 0, sizeof toks);
    float *logits = NULL;
    pulsar_sample_scratch scratch; memset(&scratch, 0, sizeof scratch);
    {
        size_t text_len = 0;
        char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
        if (!text) { fprintf(stderr, "prompt file read failed (run from the repo root)\n"); goto done; }
        pulsar_tokenize_text(e, text, &toks);
        free(text);
        /* the forced carry may run up to 17 tokens past the last measured position */
        if (toks.len < start + n_pos + 40) {
            fprintf(stderr, "prompt has %d tokens, need %d\n", toks.len, start + n_pos + 40);
            goto done;
        }
        const int ctx = start + n_pos + 64;
        /* the batched round flow runs on a bank pool (the redraft's drafter
         * scratch is allocated with it); one bank is the whole probe */
        pulsar_engine_set_bank_pool(1u);
        if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        if (pulsar_session_bank_count(s) < 1) { fprintf(stderr, "no bank pool\n"); goto done; }
        const int width = pulsar_engine_logits_width(e);
        const int eos = pulsar_token_eos(e);
        logits = (float *)malloc((size_t)17 * (size_t)width * sizeof(float));
        r = pulsar_spec_round_new();
        if (!logits || !r) goto done;

        char err[256];
        {
            pulsar_tokens p = toks; p.len = start;
            if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync: %s\n", err); goto done; }
        }
        pulsar_session_bank_state_save(s, 0u);
        uint64_t rng = 0x5eed1182ull;   /* never drawn from: every end is forced */
        uint64_t *rngp = &rng;
        const uint32_t bank0 = 0u;

        double sum_acc = 0.0, sum_q_at_pmode = 0.0;
        int agree = 0, greedy_hit = 0, n_meas = 0, n_sparse = 0, n_rounds = 0;
        int pos = start;   /* committed prefix length; toks.v[pos] is the next true token */
        while (n_meas < n_pos) {
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            /* q for the token at pos+1, drafted by the previous forced end (none before the first round) */
            pulsar_sample_dist qd; int sparse = 0;
            const int have_q = pending_q0(s, (uint32_t)width, temperature, top_k, top_p, min_p, &scratch, &qd, &sparse);
            if (have_q < 0) { fprintf(stderr, "q read failed at pos %d\n", pos); goto done; }
            const int draft0 = have_q ? (int)s->spec.dspark_pending[0] : -1;
            int first;
            if (n_rounds == 0) {
                first = toks.v[pos];
            } else {
                first = pulsar_session_spec_next_base(s, temperature, top_k, top_p, min_p, rngp);
                if (first != toks.v[pos]) {
                    fprintf(stderr, "forced carry lost: next_base=%d truth=%d at pos %d\n", first, toks.v[pos], pos);
                    if (have_q) pulsar_sample_dist_free(&qd);
                    goto done;
                }
            }
            if (pulsar_session_spec_round_begin(s, r, first, 17, 17, temperature, top_k, top_p, min_p, err, sizeof err) != 0) {
                fprintf(stderr, "round_begin at %d: %s\n", pos, err);
                if (have_q) pulsar_sample_dist_free(&qd);
                goto done;
            }
            pulsar_multiseq_req reqs[17];
            const uint32_t rows = pulsar_spec_round_fill_reqs(r, bank0, first, reqs);
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
                if (have_q) pulsar_sample_dist_free(&qd);
                goto done;
            }
            /* p: the target after `first`, row 0 of the block, production transform */
            if (have_q) {
                pulsar_sample_dist pd; memset(&pd, 0, sizeof pd);
                if (!pulsar_sample_dist_build(logits, (uint32_t)width, temperature, top_k, top_p, min_p, &scratch, &pd)) {
                    fprintf(stderr, "p build failed at %d\n", pos);
                    pulsar_sample_dist_free(&qd);
                    goto done;
                }
                int p_mode = -1; float p_mode_v = -1.0f;
                for (uint32_t k = 0; k < pd.n; k++) if (pd.probs[k] > p_mode_v) { p_mode_v = pd.probs[k]; p_mode = pd.ids[k]; }
                int q_mode = -1; float q_mode_v = -1.0f;
                double acc = 0.0, q_at_pmode = 0.0;
                for (uint32_t k = 0; k < qd.n; k++) {
                    const float qv = qd.probs[k], pv = pulsar_sample_dist_prob(&pd, qd.ids[k]);
                    acc += (double)(qv < pv ? qv : pv);
                    if (qv > q_mode_v) { q_mode_v = qv; q_mode = qd.ids[k]; }
                    if (qd.ids[k] == p_mode) q_at_pmode = qv;
                }
                pulsar_sample_dist_free(&pd);
                pulsar_sample_dist_free(&qd);
                sum_acc += acc; sum_q_at_pmode += q_at_pmode;
                agree += (q_mode == p_mode);
                greedy_hit += (draft0 == toks.v[pos + 1]);
                n_sparse += sparse;
                n_meas++;
                if (n_meas % 250 == 0)
                    printf("  %5d positions: E[accept]=%.5f  argmax agreement=%.4f  draft==truth=%.4f\n",
                           n_meas, sum_acc / n_meas, (double)agree / n_meas, (double)greedy_hit / n_meas);
            }
            /* forced end: the corpus decides acceptance and the carry */
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            int accepted[17];
            const int na = pulsar_session_spec_round_end_forced(s, r, first, eos, temperature, top_k, top_p, min_p,
                                                                rngp, logits, 0u, toks.v + pos + 1, accepted, 17,
                                                                err, sizeof err);
            if (na < 0) { fprintf(stderr, "round_end_forced at %d: %s\n", pos, err); goto done; }
            for (int i = 0; i < na; i++) {
                if (accepted[i] != toks.v[pos + i]) {
                    fprintf(stderr, "forced round committed %d at pos %d, truth %d\n", accepted[i], pos + i, toks.v[pos + i]);
                    goto done;
                }
            }
            pos += na;
            n_rounds++;
            pulsar_session_bank_state_save(s, bank0);
            pulsar_spec_round *rr = r; uint32_t bb = bank0;
            if (pulsar_session_spec_redraft_batch(s, &rr, &bb, &rngp, 1, err, sizeof err) != 0) {
                fprintf(stderr, "redraft at %d: %s\n", pos, err); goto done;
            }
            if (!pulsar_session_bank_state_restore(s, bank0)) { fprintf(stderr, "bank restore failed\n"); goto done; }
            pulsar_session_spec_redraft_commit(s, r);
            pulsar_session_bank_state_save(s, bank0);
        }
        printf("TEACHER-FORCED: positions=%d start=%d rounds=%d temp=%.2f min_p=%.2f | E[accept]=%.6f  "
               "argmax agreement=%.5f  draft==truth=%.5f  mean q(p-mode)=%.5f  (sparse q at %d positions)\n",
               n_meas, start, n_rounds, temperature, min_p, sum_acc / n_meas, (double)agree / n_meas,
               (double)greedy_hit / n_meas, sum_q_at_pmode / n_meas, n_sparse);
        rc = 0;
    }
done:
    pulsar_sample_scratch_free(&scratch);
    free(logits);
    if (r) pulsar_spec_round_free(r);
    pulsar_tokens_free(&toks);
    if (s) pulsar_session_free(s);
    gate_engine_close(e);
    return rc;
}

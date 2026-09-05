/* L182 PROBE: teacher-forced drafter acceptance -- the drafter's quality as a
 * deterministic number, no free-running trajectories.
 *
 * The spec-sampling oracle reads alpha over free-running trajectories under
 * fixed seeds: any numerics change flips a near-tie sample, the trajectories
 * diverge, and the reading is a fresh draw (~1 point of standard error per
 * 1250 trajectories -- L166's 2.2-point drop needed the full 2500 to be called
 * systematic).  This probe fixes the CONTEXT instead: walk a corpus prefix
 * token by token, and at each position take the drafter's proposal
 * distribution q and the target's distribution p over the SAME context, then
 * accumulate the exact expected acceptance under the temperature-matched rule,
 * sum_v min(p_v, q_v), plus argmax agreement.  Both p and q are deterministic
 * functions of (model, drafter, context), so a one-ulp kernel change shows as
 * a 1e-4 move and a two-point move is unambiguous.
 *
 * Mechanics per position i (true token t_i = corpus[P0 + i]):
 *   round_begin(first = t_i, depth 1)  -- the round evaluates t_i and one draft;
 *   p = row 0 of the round's spec logits (the target after t_i), built with the
 *       production sampling transform (temperature 1.0, top_k 0, top_p 1.0,
 *       min_p 0.05);
 *   q = the drafter's stored proposal distribution for draft position 0;
 *   round_abort; eval(t_i) to advance the committed context.
 * Two target forwards per position; ~2000 positions in a few minutes.
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
        if (toks.len < start + n_pos + 2) {
            fprintf(stderr, "prompt has %d tokens, need %d\n", toks.len, start + n_pos + 2);
            goto done;
        }
        if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session failed\n"); goto done; }
        const int width = pulsar_engine_logits_width(e);
        logits = (float *)malloc((size_t)width * sizeof(float));
        r = pulsar_spec_round_new();
        if (!logits || !r) goto done;

        /* committed prefix */
        char err[256];
        {
            pulsar_tokens p = toks; p.len = start;
            if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync: %s\n", err); goto done; }
        }
        double sum_acc = 0.0, sum_qmax_p = 0.0;
        int agree = 0, n_done = 0, n_qrow = 0;
        for (int i = 0; i < n_pos; i++) {
            const int t = toks.v[start + i];
            /* the round evaluates t and drafts ONE token after it */
            if (pulsar_session_spec_round_begin(s, r, t, 1, 17, temperature, top_k, top_p, min_p, err, sizeof err) != 0) {
                fprintf(stderr, "round_begin at %d: %s\n", start + i, err); goto done;
            }
            /* p: the target after t, from the round's row 0 */
            if (!gpu_graph_read_spec_logits_row(&s->graph, 0u, logits)) { fprintf(stderr, "spec logits row 0 read failed\n"); goto done; }
            pulsar_sample_dist pd; memset(&pd, 0, sizeof pd);
            if (!pulsar_sample_dist_build(logits, (uint32_t)width, temperature, top_k, top_p, min_p, &scratch, &pd)) {
                fprintf(stderr, "p build failed at %d\n", start + i); goto done;
            }
            /* q: the drafter's stored proposal distribution for draft position 0 */
            const uint32_t qn = s->spec.dspark_pending_qn[0];
            double acc = 0.0, q_at_pmode = 0.0;
            int p_mode = -1; float p_mode_v = -1.0f;
            for (uint32_t k = 0; k < pd.n; k++) if (pd.probs[k] > p_mode_v) { p_mode_v = pd.probs[k]; p_mode = pd.ids[k]; }
            int q_mode = -1; float q_mode_v = -1.0f;
            if (qn > 0) {
                for (uint32_t k = 0; k < qn; k++) {
                    const int id = s->spec.dspark_pending_qids[0][k];
                    const float qv = s->spec.dspark_pending_qprobs[0][k];
                    const float pv = pulsar_sample_dist_prob(&pd, id);
                    acc += (double)(qv < pv ? qv : pv);
                    if (qv > q_mode_v) { q_mode_v = qv; q_mode = id; }
                    if (id == p_mode) q_at_pmode = qv;
                }
            } else {
                /* q too wide for the sparse store: rebuild from the drafter's full logits row */
                const float *qrow = s->dspark_pending_qrows;
                pulsar_sample_dist qd; memset(&qd, 0, sizeof qd);
                if (!qrow || !pulsar_sample_dist_build(qrow, (uint32_t)width, temperature, top_k, top_p, min_p, &scratch, &qd)) {
                    fprintf(stderr, "q build failed at %d\n", start + i); pulsar_sample_dist_free(&pd); goto done;
                }
                for (uint32_t k = 0; k < qd.n; k++) {
                    const float qv = qd.probs[k], pv = pulsar_sample_dist_prob(&pd, qd.ids[k]);
                    acc += (double)(qv < pv ? qv : pv);
                    if (qv > q_mode_v) { q_mode_v = qv; q_mode = qd.ids[k]; }
                    if (qd.ids[k] == p_mode) q_at_pmode = qv;
                }
                pulsar_sample_dist_free(&qd);
                n_qrow++;
            }
            pulsar_sample_dist_free(&pd);
            pulsar_session_spec_round_abort(s, r);
            sum_acc += acc; sum_qmax_p += q_at_pmode; agree += (q_mode == p_mode);
            n_done++;
            /* advance the committed context by the true token */
            if (pulsar_session_eval(s, t, err, sizeof err) != 0) { fprintf(stderr, "eval at %d: %s\n", start + i, err); goto done; }
            if ((i + 1) % 250 == 0)
                printf("  %5d positions: E[accept]=%.5f  argmax agreement=%.4f\n", i + 1, sum_acc / n_done, (double)agree / n_done);
        }
        printf("TEACHER-FORCED: positions=%d start=%d temp=%.2f min_p=%.2f | E[accept]=%.6f  argmax agreement=%.5f  "
               "mean q(p-mode)=%.5f  (q rebuilt from full row at %d positions)\n",
               n_done, start, temperature, min_p, sum_acc / n_done, (double)agree / n_done, sum_qmax_p / n_done, n_qrow);
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

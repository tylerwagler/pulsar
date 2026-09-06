#include "dsq_internal.h"

/* ============================================================
 * ds4-spark: MSE oracle (--mse-probe). For each routed expert tensor, sample
 * experts, round-trip the FP4-dequantized source through each candidate format
 * (quantize -> dequantize), and report imatrix-weighted relative error per
 * (tensor, format). Emits mse.json for prisma_alloc.py --mse. The FP4 source is
 * ground truth: this is the error each format ADDS on top of the source, which
 * is exactly what the bit-allocation knapsack trades. */
/* Probe ONLY what we serve: the IQ2_XXS/Q2_K combo. FP8_E4M3 and MXFP4 are
 * lossless relative to the FP4/FP8 source so they don't need MSE probing. */
static const ds4q_type PROBE_CANDS[] = {
    DS4Q_TYPE_IQ2_XXS, DS4Q_TYPE_Q2_K,
};
static const char *PROBE_NAMES[] = { "IQ2_XXS","Q2_K" };
#define PROBE_NCAND ((int)(sizeof(PROBE_CANDS)/sizeof(PROBE_CANDS[0])))

static void probe_dequant(ds4q_type t, const void *blocks, float *out, int64_t n) {
    switch (t) {
    case DS4Q_TYPE_IQ2_XXS: ds4q_dequantize_iq2_xxs(blocks, out, n); break;
    case DS4Q_TYPE_Q2_K:    ds4q_dequantize_q2_k(blocks, out, n); break;
    default: die("probe: unsupported candidate type");
    }
}

static void probe_one_tensor(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                             const imatrix_store *im, int n_experts, int sample,
                             double mse_out[PROBE_NCAND], double *energy_out,
                             double *raw_out) {
    expert_tensor e = parse_expert_tensor(gguf_name);
    const char *wid = expert_part_name(e.part);
    const int64_t ncols = tmpl->ne[0], nrows = tmpl->ne[1];
    double wse[PROBE_NCAND]; for (int c = 0; c < PROBE_NCAND; c++) wse[c] = 0.0;
    double wsum = 0.0, raw_sum = 0.0;   /* FULL (all-expert) energy: Σimat·src², Σsrc² */
    double wsum_smp = 0.0;              /* sampled energy, to normalize the sampled MSE */
    /* Sensitivity (wsum) is summed over ALL experts because per-expert imatrix
     * importance is heavy-tailed — sampling it makes the per-layer sensitivity
     * unstable. MSE is a flat relative ratio, so it stays sampled (every mse_step). */
    int mse_step = n_experts / sample; if (mse_step < 1) mse_step = 1;
    for (int xid = 0; xid < n_experts; xid++) {
        char prefix[256], wn[320], sn[320];
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d.%s", e.layer, xid, wid);
        snprintf(wn, sizeof(wn), "%s.weight", prefix);
        snprintf(sn, sizeof(sn), "%s.scale", prefix);
        st_value w = db_read(db, wn), s = db_read(db, sn);
        int64_t n = 0;
        float *f32 = dequant_fp4_weight(&w, &s, &n);
        const char *names[3] = { gguf_name, wn, NULL };
        const float *imat = imatrix_find(im, names, 2, ncols, xid, n_experts);
        /* The probe's error is imatrix-weighted by definition; a tensor the
         * collector did not cover has no unweighted stand-in. */
        if (!imat) {
            fprintf(stderr, "error: --mse-probe: no imatrix entry for %s (expert %d)\n", wn, xid);
            exit(1);
        }
        /* energy over EVERY expert */
        double en = 0.0, rawe = 0.0;
        for (int64_t r = 0; r < nrows; r++) {
            const float *xr = f32 + (size_t)r * ncols;
            for (int64_t col = 0; col < ncols; col++) {
                double wc = imat[col], sq = (double)xr[col] * (double)xr[col];
                en += wc * sq; rawe += sq;
            }
        }
        wsum += en; raw_sum += rawe;
        /* MSE round-trip only on sampled experts (expensive) */
        if (xid % mse_step == 0) {
            float *rt = xmalloc((size_t)n * sizeof(float));
            for (int c = 0; c < PROBE_NCAND; c++) {
                byte_buf q = f32_to_type(f32, n, PROBE_CANDS[c], ncols, imat, wn);
                probe_dequant(PROBE_CANDS[c], q.data, rt, n);
                double se = 0.0;
                for (int64_t r = 0; r < nrows; r++) {
                    const float *xr = f32 + (size_t)r * ncols, *rr = rt + (size_t)r * ncols;
                    for (int64_t col = 0; col < ncols; col++) {
                        double wc = imat[col], d = (double)xr[col] - (double)rr[col];
                        se += wc * d * d;
                    }
                }
                wse[c] += se;
                free(q.data);
            }
            free(rt);
            wsum_smp += en;
        }
        free(f32); st_value_free(&w); st_value_free(&s);
    }
    for (int c = 0; c < PROBE_NCAND; c++) mse_out[c] = wsum_smp > 0 ? wse[c] / wsum_smp : 0.0;
    /* wsum = imatrix-weighted source energy = principled per-tensor sensitivity.
     * dKL(tensor,fmt) = wsum * mse_rel = absolute imatrix-weighted output error,
     * which the allocator sums and minimizes under the byte budget. */
    if (energy_out) *energy_out = wsum;
    if (raw_out) *raw_out = raw_sum;   /* weight-only energy: weight- vs imat-driven sensitivity */
}

typedef struct {
    st_db *db; const gguf_file *tmpl; const imatrix_store *im;
    int n_experts, sample;
    const int *work; int n_work;
    double (*mse)[PROBE_NCAND];
    double *energy;
    double *raw;
    int next, done; pthread_mutex_t lock;
} probe_job;

static void *probe_worker(void *arg) {
    probe_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int wi = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (wi >= j->n_work) break;
        int ti = j->work[wi];
        probe_one_tensor(j->db, j->tmpl->tensors[ti].name, &j->tmpl->tensors[ti],
                         j->im, j->n_experts, j->sample, j->mse[wi], &j->energy[wi],
                         &j->raw[wi]);
        pthread_mutex_lock(&j->lock);
        int d = ++j->done;
        fprintf(stderr, "mse-probe: %d/%d %s\n", d, j->n_work, j->tmpl->tensors[ti].name);
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

void run_mse_probe(st_db *db, const gguf_file *tmpl, const imatrix_store *im,
                          int n_experts, int n_threads, int sample, const char *out_path) {
    if (sample <= 0) sample = 8;
    for (int c = 0; c < PROBE_NCAND; c++) ds4q_quantize_init(PROBE_CANDS[c]);
    int *work = xmalloc((size_t)tmpl->n_tensors * sizeof(int));
    int n_work = 0;
    for (uint64_t i = 0; i < tmpl->n_tensors; i++)
        if (parse_expert_tensor(tmpl->tensors[i].name).is_expert)
            work[n_work++] = (int)i;
    fprintf(stderr, "mse-probe: %d routed expert tensors, sample=%d experts, %d candidates\n",
            n_work, sample, PROBE_NCAND);
    double (*mse)[PROBE_NCAND] = xmalloc((size_t)n_work * sizeof(*mse));
    double *energy = xmalloc((size_t)n_work * sizeof(double));
    double *raw = xmalloc((size_t)n_work * sizeof(double));
    probe_job job = { .db = db, .tmpl = tmpl, .im = im, .n_experts = n_experts,
                      .sample = sample, .work = work, .n_work = n_work,
                      .mse = mse, .energy = energy, .raw = raw };
    pthread_mutex_init(&job.lock, NULL);
    int wc = n_threads > 0 ? n_threads : 8;
    if (wc > n_work) wc = n_work;
    if (wc < 1) wc = 1;
    pthread_t *th = xcalloc((size_t)wc, sizeof(th[0]));
    for (int i = 1; i < wc; i++) pthread_create(&th[i], NULL, probe_worker, &job);
    probe_worker(&job);
    for (int i = 1; i < wc; i++) pthread_join(th[i], NULL);
    pthread_mutex_destroy(&job.lock); free(th);
    FILE *fp = fopen(out_path, "w");
    if (!fp) die_errno("open mse-probe out", out_path);
    fputc('{', fp);
    for (int wi = 0; wi < n_work; wi++) {
        fprintf(fp, "%s\"%s\":{", wi ? "," : "", tmpl->tensors[work[wi]].name);
        for (int c = 0; c < PROBE_NCAND; c++)
            fprintf(fp, "%s\"%s\":%.8e", c ? "," : "", PROBE_NAMES[c], mse[wi][c]);
        fputc('}', fp);
    }
    fputc('}', fp);
    fclose(fp);
    fprintf(stderr, "mse-probe: wrote %s\n", out_path);
    /* sidecar sensitivity file <out>.sens.json: wsum (imatrix-weighted source
     * energy) per tensor, for prisma_alloc.py --sens. */
    char sens_path[4096];
    { const char *dot = strrchr(out_path, '.');
      size_t base = (dot && strcmp(dot, ".json") == 0) ? (size_t)(dot - out_path)
                                                       : strlen(out_path);
      snprintf(sens_path, sizeof(sens_path), "%.*s.sens.json", (int)base, out_path); }
    FILE *sp = fopen(sens_path, "w");
    if (!sp) die_errno("open mse-probe sens out", sens_path);
    fputc('{', sp);
    for (int wi = 0; wi < n_work; wi++)
        fprintf(sp, "%s\"%s\":%.8e", wi ? "," : "",
                tmpl->tensors[work[wi]].name, energy[wi]);
    fputc('}', sp);
    fclose(sp);
    fprintf(stderr, "mse-probe: wrote %s\n", sens_path);
    /* diagnostic sidecar <out>.diag.json: wsum (imat-weighted) and raw (Σsrc²)
     * per tensor, to distinguish weight-driven vs imatrix-driven sensitivity. */
    char diag_path[4096];
    { const char *dot = strrchr(out_path, '.');
      size_t base = (dot && strcmp(dot, ".json") == 0) ? (size_t)(dot - out_path)
                                                       : strlen(out_path);
      snprintf(diag_path, sizeof(diag_path), "%.*s.diag.json", (int)base, out_path); }
    FILE *dp = fopen(diag_path, "w");
    if (dp) {
        fputc('{', dp);
        for (int wi = 0; wi < n_work; wi++)
            fprintf(dp, "%s\"%s\":{\"wsum\":%.8e,\"raw\":%.8e}", wi ? "," : "",
                    tmpl->tensors[work[wi]].name, energy[wi], raw[wi]);
        fputc('}', dp);
        fclose(dp);
        fprintf(stderr, "mse-probe: wrote %s\n", diag_path);
    }
    free(work); free(mse); free(energy); free(raw);
}


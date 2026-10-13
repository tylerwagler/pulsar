/* L291: the KV-format walk -- teacher-forced windows at depth, one engine load for every window.
 *
 * The short grade (L288 head_short.py) resets rows per document and scores their first <= 2048 tokens; a KV
 * cache's error compounds with context, so this scores WINDOWS that END at a chosen depth: each manifest line
 * is one document (int32 token ids) cut at `depth`, its first depth - window tokens prefilled through sync()
 * (production prefill, which writes the cache in the armed K/V format), then `window` positions teacher-forced
 * through eval() -- the decode lane on top of that cache.  Every window is its own session (rows reset).
 *
 *   qwen_kv_walk <container> <manifest> dump  <dir>              the reference arm: full-vocab f32 logits per
 *                                                                window -> dir/w<i>.f32, per-position NLL -> stdout
 *   qwen_kv_walk <container> <manifest> score <dir> <out.tsv>    any other arm against dir's logits
 *
 * manifest: one window per line, "<tokens.bin> <depth> <window>".  out.tsv: one line per scored position --
 * window, depth, position, KL(ref || arm), ref argmax == arm argmax, NLL of the true token under ref and under
 * the arm (nats).  tools/qwen/kv_grade.py turns a set of them into the per-depth table with block-bootstrap CIs.
 * The K/V format is the engine's (PULSAR_QWEN_KV, announced at load); this tool never chooses it. */
#include "pulsar.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static bool slurp_ids(const char *path, std::vector<int32_t> *ids) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4) { fclose(f); return false; }
    ids->resize((size_t)n / 4);
    const bool ok = fread(ids->data(), 4, ids->size(), f) == ids->size();
    fclose(f);
    return ok;
}

/* log-softmax normaliser and argmax of one logits row */
static void row_stats(const float *z, int W, double *lse, int *am) {
    double mx = z[0];
    int a = 0;
    for (int v = 1; v < W; v++) if (z[v] > mx) { mx = z[v]; a = v; }
    double s = 0.0;
    for (int v = 0; v < W; v++) s += exp((double)z[v] - mx);
    *lse = mx + log(s);
    *am = a;
}

int main(int argc, char **argv) {
    const bool dump = argc == 5 && !strcmp(argv[3], "dump");
    const bool score = argc == 6 && !strcmp(argv[3], "score");
    if (!dump && !score) {
        fprintf(stderr, "usage: %s <container> <manifest> dump <dir> | score <dir> <out.tsv>\n", argv[0]);
        return 2;
    }
    struct Win { std::string path; int depth, window; };
    std::vector<Win> wins;
    {
        FILE *mf = fopen(argv[2], "r");
        if (!mf) { fprintf(stderr, "qwen-kv-walk: cannot open manifest %s\n", argv[2]); return 2; }
        char p[1024];
        int d, w;
        while (fscanf(mf, "%1023s %d %d", p, &d, &w) == 3) wins.push_back({p, d, w});
        fclose(mf);
    }
    if (wins.empty()) { fprintf(stderr, "qwen-kv-walk: empty manifest\n"); return 2; }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.tp_rank = -1;
    opt.dspark_disable = true;   /* teacher-forced: no drafter state */
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "qwen-kv-walk: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);
    std::vector<float> z((size_t)W), ref((size_t)W);
    FILE *out = score ? fopen(argv[5], "w") : stdout;
    if (!out) { fprintf(stderr, "qwen-kv-walk: cannot write %s\n", argv[5]); return 2; }
    fprintf(out, "win\tdepth\tpos\tkl\ttop1\tnll_ref\tnll_arm\n");

    for (size_t wi = 0; wi < wins.size(); wi++) {
        const Win &w = wins[wi];
        std::vector<int32_t> ids;
        if (!slurp_ids(w.path.c_str(), &ids) || (int)ids.size() < w.depth || w.window <= 0 || w.window >= w.depth) {
            fprintf(stderr, "qwen-kv-walk: window %zu: %s cannot give depth %d window %d\n", wi, w.path.c_str(), w.depth,
                    w.window);
            return 2;
        }
        char fpath[1100];
        snprintf(fpath, sizeof fpath, "%s/w%zu.f32", argv[4], wi);
        FILE *lf = fopen(fpath, dump ? "wb" : "rb");
        if (!lf) { fprintf(stderr, "qwen-kv-walk: cannot open %s\n", fpath); return 2; }
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, w.depth + 16) != 0) { fprintf(stderr, "qwen-kv-walk: session refused\n"); return 2; }
        const int P = w.depth - w.window;
        char err[256] = "";
        const double t0 = now_s();
        pulsar_tokens pre = {ids.data(), P, P};
        if (pulsar_session_sync(s, &pre, err, sizeof(err)) != 0) { fprintf(stderr, "qwen-kv-walk: sync: %s\n", err); return 1; }
        const double t1 = now_s();
        double kl_sum = 0.0, dnll_sum = 0.0;
        int agree = 0;
        for (int i = 0; i < w.window; i++) {
            if (i > 0 && pulsar_session_eval(s, ids[(size_t)P + i - 1], err, sizeof(err)) != 0) {
                fprintf(stderr, "qwen-kv-walk: eval at %d: %s\n", P + i, err);
                return 1;
            }
            if (pulsar_session_copy_logits(s, z.data(), W) != W) { fprintf(stderr, "qwen-kv-walk: copy_logits\n"); return 1; }
            const int truth = ids[(size_t)P + i];
            double lse;
            int am;
            row_stats(z.data(), W, &lse, &am);
            const double nll = lse - (double)z[(size_t)truth];
            if (dump) {
                if (fwrite(z.data(), 4, (size_t)W, lf) != (size_t)W) { fprintf(stderr, "qwen-kv-walk: short write\n"); return 1; }
                fprintf(out, "%zu\t%d\t%d\t0\t1\t%.9g\t%.9g\n", wi, w.depth, P + i, nll, nll);
                continue;
            }
            if (fread(ref.data(), 4, (size_t)W, lf) != (size_t)W) { fprintf(stderr, "qwen-kv-walk: %s is short\n", fpath); return 1; }
            double rlse;
            int ram;
            row_stats(ref.data(), W, &rlse, &ram);
            double kl = 0.0;
            for (int v = 0; v < W; v++) {
                const double lr = (double)ref[(size_t)v] - rlse;
                kl += exp(lr) * (lr - ((double)z[(size_t)v] - lse));
            }
            const double rnll = rlse - (double)ref[(size_t)truth];
            fprintf(out, "%zu\t%d\t%d\t%.9g\t%d\t%.9g\t%.9g\n", wi, w.depth, P + i, kl, ram == am, rnll, nll);
            kl_sum += kl;
            dnll_sum += nll - rnll;
            agree += ram == am;
        }
        const double t2 = now_s();
        fclose(lf);
        pulsar_session_free(s);
        fprintf(stderr, "qwen-kv-walk: window %zu (%s, depth %d, %d scored): prefill %d tok %.1f s (%.0f tok/s), walk %.1f s",
                wi, w.path.c_str(), w.depth, w.window, P, t1 - t0, P / (t1 - t0), t2 - t1);
        if (score) fprintf(stderr, "; KL %.4g, top-1 %.4f, dNLL %+.4f", kl_sum / w.window, (double)agree / w.window,
                           dnll_sum / w.window);
        fprintf(stderr, "\n");
        fflush(out);
    }
    if (score) fclose(out);
    return 0;
}

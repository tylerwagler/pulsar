/* imatrix_qwen.cpp -- the Qwen4-exp family's importance-matrix collection (L284 P15; family.h pulsar_family_imatrix).
 *
 * The dataset walk, the caps and the llama.cpp `.dat` frame are the core's (session.cpp, imatrix.cpp).  What this
 * file adds is the family's half: how a prompt is run through the forward with its linears observed.
 *
 *   - the run: one session, each prompt invalidated and synced whole (no shared prefix is skipped), so every
 *     token passes every layer exactly once;
 *   - the observation: the trunk's ops note each linear's INPUT rows through the core's front doors
 *     (family_qwen_s4.cpp: step_linear, the shared expert, the routed MoE) into the tap below -- the same
 *     bf16 rows the kernels read, so the statistics are of the exact inputs the weights see.  The routed
 *     down's input is the fold's bf16 SwiGLU rows with the route weight folded in, as DeepSeek's collector
 *     reads batch_routed_mid.
 *
 * An entry is the checkpoint's own tensor name and sum(x[col]^2) / rows per input column; an expert-stacked
 * tensor carries `n_expert` such vectors.  Nothing about the forward changes: the tap only reads. */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"
#include <map>
#include <string>
#include <vector>

struct imatrix_acc {
    std::string name;
    uint32_t n_expert, n_col;
    std::vector<double> sum2;      ///< [n_expert][n_col]
    std::vector<uint32_t> count;   ///< [n_expert]
};

struct pulsar_imatrix_tap {
    std::vector<imatrix_acc> accs;
    std::map<std::string, size_t> index;
    uint64_t routes = 0;           ///< (token, expert) routing decisions observed
    std::vector<uint16_t> row_buf;
    std::vector<int32_t> sel_buf;
    bool failed = false;           ///< a device read failed: the collection is not to be written
};

static imatrix_acc *tap_acc(pulsar_imatrix_tap *t, const pulsar_tensor *w, uint32_t n_expert, uint32_t n_col) {
    const std::string name(w->name.ptr, w->name.len);
    auto it = t->index.find(name);
    if (it == t->index.end()) {
        it = t->index.emplace(name, t->accs.size()).first;
        t->accs.push_back({name, n_expert, n_col, std::vector<double>((size_t)n_expert * n_col, 0.0),
                           std::vector<uint32_t>(n_expert, 0u)});
    }
    imatrix_acc *a = &t->accs[it->second];
    if (a->n_expert != n_expert || a->n_col != n_col) {
        const std::string msg = "imatrix: " + name + " observed at two shapes";
        pulsar_die(msg.c_str());
    }
    return a;
}

static float bf16_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static bool tap_read(pulsar_imatrix_tap *t, const void *dev, void *host, uint64_t bytes) {
    if (pulsar_gpu_device_read(dev, host, bytes)) return true;
    t->failed = true;
    return false;
}

static void add_row_sq(imatrix_acc *a, uint32_t expert, const uint16_t *row) {
    double *dst = a->sum2.data() + (size_t)expert * a->n_col;
    for (uint32_t i = 0; i < a->n_col; i++) {
        const double v = bf16_f32(row[i]);
        dst[i] += v * v;
    }
    a->count[expert]++;
}

void pulsar_imatrix_note_dense(pulsar_imatrix_tap *tap, const pulsar_tensor *w, const uint16_t *x_dev, uint32_t rows,
                               uint32_t cols) {
    tap->row_buf.resize((size_t)rows * cols);
    if (!tap_read(tap, x_dev, tap->row_buf.data(), (uint64_t)rows * cols * sizeof(uint16_t))) return;
    imatrix_acc *a = tap_acc(tap, w, 1, cols);
    for (uint32_t r = 0; r < rows; r++) add_row_sq(a, 0, tap->row_buf.data() + (size_t)r * cols);
}

void pulsar_imatrix_note_routed_in(pulsar_imatrix_tap *tap, const pulsar_tensor *gate, const pulsar_tensor *up,
                                   const uint16_t *x_dev, const int32_t *sel_dev, uint32_t rows, uint32_t k,
                                   uint32_t n_expert, uint32_t cols) {
    tap->row_buf.resize((size_t)rows * cols);
    tap->sel_buf.resize((size_t)rows * k);
    if (!tap_read(tap, x_dev, tap->row_buf.data(), (uint64_t)rows * cols * sizeof(uint16_t)) ||
        !tap_read(tap, sel_dev, tap->sel_buf.data(), (uint64_t)rows * k * sizeof(int32_t)))
        return;
    imatrix_acc *a[2] = {tap_acc(tap, gate, n_expert, cols), up ? tap_acc(tap, up, n_expert, cols) : NULL};
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t j = 0; j < k; j++) {
            const int32_t ex = tap->sel_buf[(size_t)r * k + j];
            if (ex < 0 || (uint32_t)ex >= n_expert) continue;
            for (imatrix_acc *x : a)
                if (x) add_row_sq(x, (uint32_t)ex, tap->row_buf.data() + (size_t)r * cols);
            tap->routes++;
        }
}

void pulsar_imatrix_note_routed_mid(pulsar_imatrix_tap *tap, const pulsar_tensor *down, const uint16_t *mid_dev,
                                    const int32_t *sel_dev, uint32_t rows, uint32_t k, uint32_t n_expert,
                                    uint32_t mid) {
    if (!mid_dev) {
        tap->failed = true;
        return;
    }
    tap->row_buf.resize((size_t)rows * k * mid);
    tap->sel_buf.resize((size_t)rows * k);
    if (!tap_read(tap, mid_dev, tap->row_buf.data(), (uint64_t)rows * k * mid * sizeof(uint16_t)) ||
        !tap_read(tap, sel_dev, tap->sel_buf.data(), (uint64_t)rows * k * sizeof(int32_t)))
        return;
    imatrix_acc *a = tap_acc(tap, down, n_expert, mid);
    for (uint32_t p = 0; p < rows * k; p++) {
        const int32_t ex = tap->sel_buf[p];
        if (ex < 0 || (uint32_t)ex >= n_expert) continue;
        add_row_sq(a, (uint32_t)ex, tap->row_buf.data() + (size_t)p * mid);
    }
}

/* ---- the family's collection ----------------------------------------------------------------------------- */

struct qwen_imatrix_run {
    pulsar_imatrix_tap tap;
    pulsar_session *session;
    std::string dataset;
    uint32_t prompts;
};

static void *qwen_imatrix_begin(pulsar_engine *e, const char *dataset_path, int ctx_size) {
    if (e->tp) {
        fprintf(stderr, "pulsar: imatrix collection on %s runs on one rank; this engine is tensor-parallel -- refusing\n",
                e->family->name);
        return NULL;
    }
    qwen_imatrix_run *r = new qwen_imatrix_run;
    r->dataset = dataset_path;
    r->prompts = 0;
    r->session = NULL;
    if (pulsar_session_create(&r->session, e, ctx_size) != 0) {
        fprintf(stderr, "pulsar: failed to create the imatrix session\n");
        delete r;
        return NULL;
    }
    e->imatrix_tap = &r->tap;
    fprintf(stderr, "pulsar: collecting the linear inputs' imatrix from %s (%s, ctx=%d)\n", dataset_path,
            e->family->name, ctx_size);
    return r;
}

static bool qwen_imatrix_prompt(pulsar_engine *, void *c, const pulsar_tokens *tokens) {
    qwen_imatrix_run *r = (qwen_imatrix_run *)c;
    char err[256] = "";
    pulsar_session_invalidate(r->session);   /* a prompt's shared prefix is observed, never skipped */
    if (pulsar_session_sync(r->session, tokens, err, sizeof err) != 0) {
        fprintf(stderr, "pulsar: imatrix prefill: %s\n", err);
        return false;
    }
    r->prompts++;
    return !r->tap.failed;
}

static uint64_t qwen_imatrix_routes(const void *c) { return ((const qwen_imatrix_run *)c)->tap.routes; }

static bool qwen_imatrix_save(pulsar_engine *, void *c, const char *path) {
    qwen_imatrix_run *r = (qwen_imatrix_run *)c;
    if (r->tap.failed) {
        fprintf(stderr, "pulsar: imatrix: a device read failed during the collection -- not writing %s\n", path);
        return false;
    }
    FILE *fp = imatrix_dat_open(path, (int32_t)r->tap.accs.size());
    if (!fp) return false;
    for (const imatrix_acc &a : r->tap.accs) {
        std::vector<float> sum2(a.sum2.begin(), a.sum2.end());
        imatrix_write_entry(fp, a.name.c_str(), sum2.data(), a.count.data(), a.n_expert, a.n_col);
    }
    return imatrix_dat_close(fp, path, (int32_t)r->prompts, r->dataset.c_str());
}

static void qwen_imatrix_end(pulsar_engine *e, void *c) {
    qwen_imatrix_run *r = (qwen_imatrix_run *)c;
    e->imatrix_tap = NULL;
    pulsar_session_free(r->session);
    delete r;
}

const pulsar_family_imatrix k_qwen_imatrix = {
    /* .begin  = */ qwen_imatrix_begin,
    /* .prompt = */ qwen_imatrix_prompt,
    /* .routes = */ qwen_imatrix_routes,
    /* .save   = */ qwen_imatrix_save,
    /* .end    = */ qwen_imatrix_end,
};

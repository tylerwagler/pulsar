/* nemotron_enc_forward.cpp -- the Nemotron-3.5 encoders' forward and their encoder entries (nemotron_enc.h section 3).
 *
 * The family declares; the shared blocks compute.  One encode call packs its inputs into forwards of at most
 * PULSAR_NEMOTRON_ENC_ROWS rows (and NE_SEQS_MAX inputs), each input one bidirectional run from position 0:
 *
 *   embed      E5   pulsar_head_embed_launch: the gather (no embedding norm) -> the residual h (bf16)
 *   layer 0         xn = rmsnorm(h) * attn_norm                                  row ops (pulsar_cuda_rowops.h)
 *   per layer, the linears in row chunks of NE_CHUNK (the prompt arms: a row's bytes do not depend on its chunk)
 *     q | k | v = bf16(xn W_qkv)  (one multi launch)                             rows front door (linear.cpp)
 *     a = GQA(q, k, v): every row of an input attends every row of it (F1's block run, no pool), YaRN rope from the
 *         per-pair table built here, scale 1/sqrt(head_dim), no norm / bias / gate         F1 (pulsar_cuda_attn_gqa.h)
 *     h += bf16(a W_o);  xn = rmsnorm(h) * ffn_norm                               row ops (one fused launch)
 *     g | u = bf16(xn W_gate | W_up);  s = bf16(silu(g) * u)                      rows front door; the SwiGLU block
 *     h += bf16(s W_down);  xn = rmsnorm(h) * (the next layer's attn_norm, or output_norm after the last)
 *   pool       pooled[i] = the f32 mean of input i's xn rows                     (pulsar_cuda_pool.h)
 *   finish     Embed: pooled / max(|pooled|, 1e-12);  Rerank: score . pooled     host, f32 out (double sums)
 *
 * transformers' Ministral3 (bf16) is the order: each sublayer x += f(RMSNorm(x)) with the residual bf16, each add and
 * each norm one rounding.  Where pulsar rounds differently, it is said here and graded (rule 7; retriever/ grades the
 * vectors against the f32 CPU reference): the f32 linears' one bf16 rounding (= a bf16 GEMM's output); silu(g) * u in
 * f32 rounded once (vLLM's silu_and_mul; transformers rounds silu(g) to bf16 first); the mean pooled in f32 from the
 * bf16 rows (the reference pools its own last hidden state in f32).
 *
 * THE ROPE: transformers' _compute_yarn_parameters -- the correction range in double (Python), the frequencies, the
 * ramp and the blend in f32 (torch) -- evaluated once on the host; the kernel's angle is pos * inv_freq in f32, cos and
 * sin rounded to bf16 (F1's rope contract).  The attention factor is exactly 1 (the shape pins mscale =
 * mscale_all_dim = 1), and the llama-4 q scale is exactly 1 below position 16384 (the input cap). */
#include "nemotron_enc.h"
#include "lib/bpe_tokenizer.h"
#include "lib/pulsar_yarn.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_blockdraft.h"   /* the SwiGLU block (family-neutral) */
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_pool.h"
#include "cuda/pulsar_cuda_qwen.h"         /* the rows front door's launchers (family-neutral) */
#include "cuda/pulsar_cuda_rowops.h"

#include <math.h>
#include <pthread.h>

#include <string>
#include <vector>

namespace {

const char *const ARCH = PULSAR_NEMOTRON_ENC_ARCH;
/** Rows a linear launch carries (the f32 staging is sized by it). */
constexpr uint32_t NE_CHUNK = 2048;
/** Inputs one forward carries (the pooled rows are sized by it). */
constexpr uint32_t NE_SEQS_MAX = 256;
/** The U+200C the template writes after '[' and '<' in client text. */
const char ZWNJ[] = "\xe2\x80\x8c";

/* The forward's scratch, the PROCESS's (L321): every encoder engine of a process (pulsar_engine_open_group) runs on
 * one GPU, one call at a time, so they share one set of buffers -- each sized to the largest member's need -- and the
 * lock that serialises their calls. */
enum ne_buf { B_IDS, B_H, B_XN, B_Q, B_K, B_V, B_A, B_G, B_U, B_S, B_DELTA, B_STAGE, B_LWS, B_AWS, B_POOLED, B_COUNT };

struct ne_scratch {
    uint32_t refs;                       ///< the engines sharing it
    pulsar_gpu_tensor *t[B_COUNT];       ///< ids i32 [rows]; h, xn bf16 [rows][n_embd]; q, k, v, a bf16 [rows][heads *
                                         ///< head_dim]; g, u, s bf16 [NE_CHUNK][n_ff]; delta bf16 [NE_CHUNK][n_embd];
                                         ///< stage f32 [NE_CHUNK][2 n_ff] (a multi launch's staging, the widest: gate
                                         ///< | up); the linears' workspace at NE_CHUNK rows; the attention's at `rows`;
                                         ///< pooled f32 [NE_SEQS_MAX][n_embd]
    uint64_t bytes[B_COUNT];
    pthread_mutex_t mu;                  ///< one encode at a time, across the engines
};

ne_scratch *g_scratch;

}  // namespace

struct pulsar_nemotron_enc_run {
    pulsar_gqa_layer attn;
    uint32_t rows;                 ///< rows one forward carries: the shape's input cap
    pulsar_gpu_tensor *rope;       ///< f32 [head_dim / 2]: the YaRN inverse frequencies
    ne_scratch *sc;                ///< the process's scratch (g_scratch)
};

namespace {

pulsar_nemotron_enc_model *W(const pulsar_engine *e) { return (pulsar_nemotron_enc_model *)e->family_state; }

template <class T = void>
T *dp(const pulsar_gpu_tensor *t) { return (T *)pulsar_gpu_tensor_device_ptr(t); }

bool fail(char *err, size_t errlen, const char *what) {
    if (err) snprintf(err, errlen, "%s: %s", ARCH, what);
    return false;
}

/* transformers' YaRN inverse frequencies for the pinned shape (see the file comment) */
std::vector<float> yarn_inv_freq(const pulsar_nemotron_enc_shape &s) {
    const uint32_t dim = s.head_dim, half = dim / 2u;
    double low = 0, high = 0;
    pulsar_yarn_corr_range<double>(dim, s.rope_orig_ctx, (double)s.rope_theta, (double)s.rope_beta_fast,
                                   (double)s.rope_beta_slow, &low, &high);
    std::vector<float> inv(half);
    for (uint32_t i = 0; i < half; i++) {
        const float pf = powf(s.rope_theta, (float)(2u * i) / (float)dim);   /* torch: base ** (arange f32 / dim) */
        const float extrap = 1.0f / pf, interp = 1.0f / (s.rope_factor * pf);
        const float w = pulsar_yarn_extrap_weight<float>((float)low, (float)high, (float)i);
        inv[i] = pulsar_yarn_blend<float>(interp, extrap, w);
    }
    return inv;
}

/* n (1..3) linears of the same bf16 input x [rows][in] -> their bf16 rows yb[j] [rows][out[j]], on the prompt arms */
bool linears(const pulsar_engine *e, const pulsar_nemotron_enc_run *r, int n, const pulsar_tensor *const *t, int in,
             const int *out, const char *const *what, const uint16_t *x, uint32_t rows, uint16_t *const *yb) {
    pulsar_rows_linear l[3];
    const pulsar_rows_linear *lp[3];
    float *st[3];
    float *stage = dp<float>(r->sc->t[B_STAGE]);
    for (int j = 0; j < n; j++) {
        if (!pulsar_linear_rows_ref(&e->model, t[j], in, out[j], true, what[j], &l[j])) return false;
        lp[j] = &l[j];
        st[j] = stage;
        stage += (size_t)rows * (size_t)out[j];
    }
    return pulsar_rows_linear_multi_bf16_launch(lp, n, x, (int)rows, st, yb, dp(r->sc->t[B_LWS]), r->sc->bytes[B_LWS],
                                                0) == 0;
}

/* One forward over inputs [0, n): their pooled rows into the scratch's B_POOLED.  `rows` = the inputs' total length. */
bool forward(pulsar_engine *e, const pulsar_tokens *in, uint32_t n, uint32_t rows, char *err, size_t errlen) {
    pulsar_nemotron_enc_model *w = W(e);
    pulsar_nemotron_enc_run *r = w->run;
    const pulsar_nemotron_enc_shape &s = w->shape;
    const uint32_t E = s.n_embd, Q = s.n_head * s.head_dim, KV = s.n_head_kv * s.head_dim, F = s.n_ff;
    const pulsar_model *m = &e->model;
    auto wp = [&](const pulsar_tensor *t, const char *what) {
        return (const uint16_t *)pulsar_weight_device_ptr(m, t, what);
    };
    /* the ids, and each row's input and position */
    std::vector<int32_t> ids(rows);
    std::vector<uint32_t> row_seq(rows), row_pos(rows);
    std::vector<pulsar_gqa_seq> seqs(n);   /* no pool: a block run from position 0 stores nothing */
    for (uint32_t i = 0, o = 0; i < n; i++)
        for (int t = 0; t < in[i].len; t++, o++) {
            ids[o] = in[i].v[t];
            row_seq[o] = i;
            row_pos[o] = (uint32_t)t;
        }
    if (!pulsar_gpu_tensor_write(r->sc->t[B_IDS], 0, ids.data(), (uint64_t)rows * 4u))
        return fail(err, errlen, "the ids upload failed");
    pulsar_head_decl hd = {};
    hd.n_embd = E;
    hd.n_logits = s.n_vocab;
    hd.n_real = s.n_vocab;
    hd.logit_scale = 1.0f;
    pulsar_gpu_tensor *const *T = r->sc->t;
    uint16_t *h = dp<uint16_t>(T[B_H]), *xn = dp<uint16_t>(T[B_XN]), *q = dp<uint16_t>(T[B_Q]), *k = dp<uint16_t>(T[B_K]),
             *v = dp<uint16_t>(T[B_V]), *a = dp<uint16_t>(T[B_A]), *g = dp<uint16_t>(T[B_G]), *u = dp<uint16_t>(T[B_U]),
             *sw = dp<uint16_t>(T[B_S]), *dl = dp<uint16_t>(T[B_DELTA]);
    const uint16_t *table = wp(w->embed, "nemotron_enc embed");
    if (!table || pulsar_head_embed_launch(&hd, table, s.n_vocab, dp<int32_t>(r->sc->t[B_IDS]), rows, h, E, 0) != 0)
        return fail(err, errlen, "the embedding refused");
    if (pulsar_rows_rmsnorm_launch(h, E, rows, E, wp(w->layer[0].attn_norm, "nemotron_enc attn_norm"), s.rms_eps, xn,
                                   E, 0) != 0)
        return fail(err, errlen, "layer 0's attention norm refused");
    pulsar_gqa_io io = {};
    io.q = {q, Q, s.head_dim};
    io.k = {k, KV, s.head_dim};
    io.v = {v, KV, s.head_dim};
    io.out_bf16 = a;
    io.ld_out = Q;
    io.workspace = dp(r->sc->t[B_AWS]);
    io.workspace_bytes = r->sc->bytes[B_AWS];
    for (uint32_t il = 0; il < s.n_layer; il++) {
        const pulsar_nemotron_enc_layer &L = w->layer[il];
        const uint16_t *next = il + 1 < s.n_layer ? wp(w->layer[il + 1].attn_norm, "nemotron_enc attn_norm")
                                                  : wp(w->output_norm, "nemotron_enc output_norm");
        const uint16_t *fnorm = wp(L.ffn_norm, "nemotron_enc ffn_norm");
        if (!next || !fnorm) return fail(err, errlen, "a norm has no device copy");
        for (uint32_t r0 = 0; r0 < rows; r0 += NE_CHUNK) {
            const uint32_t c = rows - r0 < NE_CHUNK ? rows - r0 : NE_CHUNK;
            const pulsar_tensor *t[3] = {L.attn_q, L.attn_k, L.attn_v};
            const int out[3] = {(int)Q, (int)KV, (int)KV};
            const char *what[3] = {"nemotron_enc q", "nemotron_enc k", "nemotron_enc v"};
            uint16_t *yb[3] = {q + (size_t)r0 * Q, k + (size_t)r0 * KV, v + (size_t)r0 * KV};
            if (!linears(e, r, 3, t, (int)E, out, what, xn + (size_t)r0 * E, c, yb))
                return fail(err, errlen, "the q / k / v projections refused");
        }
        if (pulsar_gqa_forward(&r->attn, seqs.data(), n, row_seq.data(), row_pos.data(), rows, &io, 0) != 0)
            return fail(err, errlen, "the attention refused");
        for (uint32_t r0 = 0; r0 < rows; r0 += NE_CHUNK) {
            const uint32_t c = rows - r0 < NE_CHUNK ? rows - r0 : NE_CHUNK;
            uint16_t *hc = h + (size_t)r0 * E, *xc = xn + (size_t)r0 * E;
            const pulsar_tensor *to[1] = {L.attn_o};
            const int oo[1] = {(int)E};
            const char *wo[1] = {"nemotron_enc o"};
            uint16_t *yo[1] = {dl};
            if (!linears(e, r, 1, to, (int)Q, oo, wo, a + (size_t)r0 * Q, c, yo) ||
                pulsar_rows_add_launch(hc, E, dl, E, c, E, fnorm, s.rms_eps, xc, E, 0) != 0)
                return fail(err, errlen, "the attention's output refused");
            const pulsar_tensor *tf[2] = {L.ffn_gate, L.ffn_up};
            const int of[2] = {(int)F, (int)F};
            const char *wf[2] = {"nemotron_enc gate", "nemotron_enc up"};
            uint16_t *yf[2] = {g, u};
            const pulsar_tensor *td[1] = {L.ffn_down};
            const char *wd[1] = {"nemotron_enc down"};
            if (!linears(e, r, 2, tf, (int)E, of, wf, xc, c, yf) ||
                pulsar_blockdraft_swiglu_launch(g, F, u, F, c, F, sw, F, 0) != 0 ||
                !linears(e, r, 1, td, (int)F, oo, wd, sw, c, yo) ||
                pulsar_rows_add_launch(hc, E, dl, E, c, E, next, s.rms_eps, xc, E, 0) != 0)
                return fail(err, errlen, "the MLP refused");
        }
    }
    for (uint32_t i = 0, o = 0; i < n; o += (uint32_t)in[i].len, i++)
        if (pulsar_rows_mean_f32_launch(xn + (size_t)o * E, E, (uint32_t)in[i].len, E,
                                        dp<float>(T[B_POOLED]) + (size_t)i * E, 0) != 0)
            return fail(err, errlen, "the pool refused");
    return true;
}

/* ---- the entries ------------------------------------------------------------------------------------------------- */

pulsar_encoder_kind kind(const pulsar_engine *e) { return W(e)->shape.kind; }
uint32_t dim(const pulsar_engine *e) { return W(e)->shape.kind == PULSAR_ENCODER_EMBED ? W(e)->shape.n_embd : 1u; }
uint32_t max_tokens(const pulsar_engine *e) { return W(e)->shape.rope_orig_ctx; }

/* client text, as the template writes it: a U+200C after every '[' and '<' -- and a span over it */
void mark(std::string *out, std::vector<pulsar_text_span> *spans, const char *text) {
    const uint32_t lo = (uint32_t)out->size();
    for (const char *p = text; *p; p++) {
        out->push_back(*p);
        if (*p == '[' || *p == '<') out->append(ZWNJ);
    }
    spans->push_back({lo, (uint32_t)out->size()});
}

bool render(pulsar_engine *e, pulsar_encode_role role, const char *text, const char *doc, pulsar_tokens *out,
            char *err, size_t errlen) {
    const pulsar_encoder_kind k = W(e)->shape.kind;
    std::string t;
    std::vector<pulsar_text_span> spans;
    if (!text || (role == PULSAR_ENCODE_PAIR && !doc)) return fail(err, errlen, "an input with no text");
    if (k == PULSAR_ENCODER_EMBED && (role == PULSAR_ENCODE_QUERY || role == PULSAR_ENCODE_PASSAGE)) {
        t = role == PULSAR_ENCODE_QUERY ? "query: " : "passage: ";   /* the system prefix, rstrip(" ") + " " */
        mark(&t, &spans, text);
    } else if (k == PULSAR_ENCODER_RERANK && role == PULSAR_ENCODE_PAIR) {
        t = "query: ";
        mark(&t, &spans, text);
        t += "\n\npassage: ";
        mark(&t, &spans, doc);
    } else {
        return fail(err, errlen, k == PULSAR_ENCODER_EMBED ? "an embedder takes a query or a passage, not a pair"
                                                           : "a reranker takes a (query, document) pair");
    }
    std::vector<int> ids;
    char why[256] = "";
    if (!bpe_tokenizer_encode(W(e)->tok, t.data(), t.size(), spans.data(), (uint32_t)spans.size(), &ids, why,
                              sizeof why)) {
        if (err) snprintf(err, errlen, "%s: the tokenizer refused the input: %s", ARCH, why);
        return false;
    }
    for (int id : ids) pulsar_tokens_push(out, id);
    return true;
}

int encode(pulsar_engine *e, const pulsar_tokens *in, uint32_t n, float *out, char *err, size_t errlen) {
    pulsar_nemotron_enc_model *w = W(e);
    pulsar_nemotron_enc_run *r = w->run;
    const pulsar_nemotron_enc_shape &s = w->shape;
    const uint32_t E = s.n_embd, D = dim(e);
    for (uint32_t i = 0; i < n; i++)
        if (in[i].len <= 0 || (uint32_t)in[i].len > r->rows) {
            if (err) snprintf(err, errlen, "%s: input %u has %d tokens (1 .. %u)", ARCH, i, in[i].len, r->rows);
            return -1;
        }
    pthread_mutex_lock(&r->sc->mu);
    std::vector<float> pooled((size_t)NE_SEQS_MAX * E);
    bool ok = true;
    /* inputs in order, greedily packed into forwards of at most r->rows rows and NE_SEQS_MAX inputs */
    for (uint32_t i0 = 0; ok && i0 < n;) {
        uint32_t i1 = i0, rows = 0;
        while (i1 < n && i1 - i0 < NE_SEQS_MAX && rows + (uint32_t)in[i1].len <= r->rows) rows += (uint32_t)in[i1++].len;
        ok = forward(e, in + i0, i1 - i0, rows, err, errlen) &&
             (pulsar_gpu_tensor_read(r->sc->t[B_POOLED], 0, pooled.data(), (uint64_t)(i1 - i0) * E * 4u) ||
              fail(err, errlen, "the pooled rows' read failed"));
        for (uint32_t i = i0; ok && i < i1; i++) {
            const float *p = &pooled[(size_t)(i - i0) * E];
            float *o = out + (size_t)i * D;
            if (s.kind == PULSAR_ENCODER_EMBED) {   /* F.normalize: x / max(|x|, 1e-12) */
                double ss = 0;
                for (uint32_t c = 0; c < E; c++) ss += (double)p[c] * p[c];
                const float nrm = fmaxf((float)sqrt(ss), 1e-12f);
                for (uint32_t c = 0; c < E; c++) o[c] = p[c] / nrm;
            } else {
                double dot = 0;
                for (uint32_t c = 0; c < E; c++) dot += (double)p[c] * w->score_f32[c];
                o[0] = (float)dot;
            }
        }
        i0 = i1;
    }
    pthread_mutex_unlock(&r->sc->mu);
    return ok ? 0 : -1;
}

}  // namespace

const pulsar_family_encoder PULSAR_NEMOTRON_ENC_ENCODER = {
    /* .kind       = */ kind,
    /* .dim        = */ dim,
    /* .max_tokens = */ max_tokens,
    /* .render     = */ render,
    /* .encode     = */ encode,
};

void pulsar_nemotron_enc_run_free(pulsar_nemotron_enc_model *w) {
    pulsar_nemotron_enc_run *r = w ? w->run : NULL;
    if (!r) return;
    pulsar_gpu_tensor_free(r->rope);
    if (r->sc && --r->sc->refs == 0) {   /* the last encoder of the process: its scratch goes */
        for (pulsar_gpu_tensor *t : r->sc->t) pulsar_gpu_tensor_free(t);
        pthread_mutex_destroy(&r->sc->mu);
        delete r->sc;
        g_scratch = NULL;
    }
    delete r;
    w->run = NULL;
}

bool pulsar_nemotron_enc_run_create(pulsar_engine *e) {
    pulsar_nemotron_enc_model *w = W(e);
    const pulsar_nemotron_enc_shape &s = w->shape;
    const uint64_t E = s.n_embd, Q = (uint64_t)s.n_head * s.head_dim, KV = (uint64_t)s.n_head_kv * s.head_dim,
                   F = s.n_ff, R = s.rope_orig_ctx, C = NE_CHUNK;
    pulsar_nemotron_enc_run *r = new pulsar_nemotron_enc_run();
    w->run = r;   /* unload frees it, whatever this verdict */
    r->rows = s.rope_orig_ctx;
    pulsar_gqa_layer &A = r->attn;
    A.n_q = s.n_head;
    A.n_kv = s.n_head_kv;
    A.head_dim = s.head_dim;
    A.kv_fmt = PULSAR_GQA_KV_BF16;
    A.span = PULSAR_GQA_FULL;
    A.scale = 1.0f / sqrtf((float)s.head_dim);
    A.rope_theta = s.rope_theta;
    A.block = 1;   /* every row of a run attends the whole run; nothing is stored */
    const std::vector<float> inv = yarn_inv_freq(s);
    r->rope = pulsar_gpu_tensor_alloc(inv.size() * sizeof(float));
    if (!r->rope || !pulsar_gpu_tensor_write(r->rope, 0, inv.data(), inv.size() * sizeof(float))) {
        fprintf(stderr, "pulsar: %s: the rope table did not reach the device -- refusing\n", ARCH);
        return false;
    }
    A.rope_inv_freq = dp<float>(r->rope);
    /* the linears' workspace: the most any of the layer's launches takes at a chunk */
    uint64_t lws = 0;
    for (uint32_t il = 0; il < s.n_layer; il++) {
        const pulsar_nemotron_enc_layer &L = w->layer[il];
        struct { int n; const pulsar_tensor *t[3]; int in; int out[3]; } groups[] = {
            {3, {L.attn_q, L.attn_k, L.attn_v}, (int)E, {(int)Q, (int)KV, (int)KV}},
            {1, {L.attn_o}, (int)Q, {(int)E}},
            {2, {L.ffn_gate, L.ffn_up}, (int)E, {(int)F, (int)F}},
            {1, {L.ffn_down}, (int)F, {(int)E}},
        };
        for (const auto &gr : groups) {
            pulsar_rows_linear l[3];
            const pulsar_rows_linear *lp[3];
            for (int j = 0; j < gr.n; j++) {
                if (!pulsar_linear_rows_ref(&e->model, gr.t[j], gr.in, gr.out[j], true, "nemotron_enc linear", &l[j]))
                    return false;
                lp[j] = &l[j];
            }
            const uint64_t b = pulsar_rows_linear_multi_workspace_bytes(lp, gr.n, (int)C);
            if (b > lws) lws = b;
        }
    }
    /* this engine's need of every buffer; the process's scratch grows to it (the engines open as one group, before
     * any of them encodes, so nothing reads a buffer while it is replaced) */
    uint64_t need[B_COUNT] = {};
    need[B_IDS] = R * 4u;
    need[B_H] = need[B_XN] = R * E * 2u;
    need[B_Q] = need[B_A] = R * Q * 2u;
    need[B_K] = need[B_V] = R * KV * 2u;
    need[B_G] = need[B_U] = need[B_S] = C * F * 2u;
    need[B_DELTA] = C * E * 2u;
    need[B_STAGE] = C * 2u * F * 4u;
    need[B_LWS] = lws ? lws : 256u;
    need[B_AWS] = pulsar_gqa_workspace_bytes(&A, (uint32_t)R, (uint32_t)R);
    need[B_POOLED] = (uint64_t)NE_SEQS_MAX * E * 4u;
    if (!g_scratch) {
        g_scratch = new ne_scratch();
        pthread_mutex_init(&g_scratch->mu, NULL);
    }
    r->sc = g_scratch;
    r->sc->refs++;
    uint64_t grown = 0, total = 0;
    for (int b = 0; b < B_COUNT; b++) {
        if (need[b] > r->sc->bytes[b]) {
            pulsar_gpu_tensor_free(r->sc->t[b]);
            r->sc->t[b] = pulsar_gpu_tensor_alloc(need[b]);
            r->sc->bytes[b] = r->sc->t[b] ? need[b] : 0;
            if (!r->sc->t[b]) {
                fprintf(stderr, "pulsar: %s: the forward's scratch did not fit the device -- refusing\n", ARCH);
                return false;
            }
            grown += need[b];
        }
        total += r->sc->bytes[b];
    }
    const double mib = 1.0 / 1048576.0;
    fprintf(stderr, "pulsar: %s: forward scratch %.0f MiB, shared by the process's %u encoder(s) (%.0f MiB allocated "
                    "for this one; rows %llu, linears at %llu rows, attention workspace %.0f MiB)\n", ARCH,
            (double)total * mib, r->sc->refs, (double)grown * mib, (unsigned long long)R, (unsigned long long)C,
            (double)r->sc->bytes[B_AWS] * mib);
    return true;
}

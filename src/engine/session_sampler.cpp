/* session_sampler.cpp -- a session's draws, on the device (L302).
 *
 * The logits a session produces stay where its forward wrote them: the live row (the distribution of the token
 * after the checkpoint) in dev.live, a batched step's headed rows in the family's slab (dev.step_rows).  Every
 * draw, argmax and logprob is THE sampler's (src/cuda/pulsar_cuda_sampler.cu), launched over those rows; what
 * comes back is a few bytes a row.  The host mirror of the live row (pulsar_session::logits) is read back only
 * for a host reader -- a copy out, the payload, the TP digest -- through live_host().
 *
 * One rule for every row, and the key a token draws at is pulsar_sample_key(stream, its index): the plain draw
 * reads the stream's next key and steps it once; the speculative walk reads key (base, j + 1) for its row j
 * (session_spec.cpp).  So a seeded request is one text whichever lane decodes which token (L288 sdet). */
#include "pulsar_engine_internal.h"

static uint32_t mask_words(const pulsar_session *s) { return ((uint32_t)s->engine->logits_width() + 31u) / 32u; }

/* dev.out: the sample outs, the walk outs, then a digest word */
static uint64_t dev_digest_off() {
    return ((uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * (sizeof(pulsar_gpu_sample_out) + sizeof(pulsar_gpu_spec_walk_out)) +
            7u) & ~7ull;
}

/* The device buffers, allocated with the session (session_build: priced with it). */
bool pulsar_session_dev_alloc(pulsar_session *s, uint32_t n_banks) {
    const uint32_t W = (uint32_t)s->engine->logits_width();
    pulsar_session_dev_rows &d = s->dev;
    d.n_banks = n_banks ? n_banks : 1u;
    d.live = pulsar_gpu_tensor_alloc((uint64_t)W * sizeof(float));
    d.bank_rows = pulsar_gpu_tensor_alloc((uint64_t)d.n_banks * W * sizeof(float));
    d.scratch = pulsar_gpu_tensor_alloc(pulsar_gpu_sample_scratch_bytes(W));
    d.out = pulsar_gpu_tensor_alloc(dev_digest_off() + sizeof(uint64_t));
    d.mask = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_GPU_SAMPLE_ROWS_MAX * mask_words(s) * sizeof(uint32_t));
    d.lp = pulsar_gpu_tensor_alloc(PULSAR_GPU_SAMPLE_ROWS_MAX * sizeof(pulsar_gpu_logprob_row));
    d.chain_out = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_GPU_CHAINS_MAX * PULSAR_GPU_CHAIN_SLOTS_MAX * sizeof(int32_t));
    d.host_stale = false;
    d.step_rows = NULL;
    d.step_n = 0;
    if (d.live && d.bank_rows && d.scratch && d.out && d.mask && d.lp && d.chain_out) return true;
    fprintf(stderr, "pulsar: session: the device sampler's buffers could not be allocated\n");
    pulsar_session_dev_free(s);
    return false;
}

void pulsar_session_dev_free(pulsar_session *s) {
    pulsar_session_dev_rows &d = s->dev;
    pulsar_gpu_tensor_free(d.live);
    pulsar_gpu_tensor_free(d.bank_rows);
    pulsar_gpu_tensor_free(d.scratch);
    pulsar_gpu_tensor_free(d.out);
    pulsar_gpu_tensor_free(d.mask);
    pulsar_gpu_tensor_free(d.lp);
    pulsar_gpu_tensor_free(d.chain_out);
    memset(&d, 0, sizeof(d));
}

/* ---- the live row ------------------------------------------------------------------------------------------- */

bool pulsar_session::live_from_rows(const pulsar_gpu_tensor *rows, uint32_t row) {
    const uint64_t rb = (uint64_t)engine->logits_width() * sizeof(float);
    if (!rows || !pulsar_gpu_tensor_copy(dev.live, 0, rows, (uint64_t)row * rb, rb)) {
        fprintf(stderr, "pulsar: session: the live row could not be taken from row %u of the step\n", row);
        return false;
    }
    dev.host_stale = true;
    return true;
}

bool pulsar_session::live_from_host(const float *row) {
    const uint64_t rb = (uint64_t)engine->logits_width() * sizeof(float);
    if (row != logits) memcpy(logits, row, rb);
    if (!pulsar_gpu_tensor_write(dev.live, 0, logits, rb)) return false;
    dev.host_stale = false;
    return true;
}

const float *pulsar_session::live_host() {
    if (dev.host_stale) {
        if (!pulsar_gpu_tensor_read(dev.live, 0, logits, (uint64_t)engine->logits_width() * sizeof(float))) {
            fprintf(stderr, "pulsar: session: the live row's readback failed\n");
            return NULL;
        }
        dev.host_stale = false;
    }
    return logits;
}

bool pulsar_session::live_save_bank(uint32_t bank) {
    const uint64_t rb = (uint64_t)engine->logits_width() * sizeof(float);
    return bank < dev.n_banks && pulsar_gpu_tensor_copy(dev.bank_rows, (uint64_t)bank * rb, dev.live, 0, rb);
}

bool pulsar_session::step_row_to_bank(uint32_t row, uint32_t bank) {
    const uint64_t rb = (uint64_t)engine->logits_width() * sizeof(float);
    if (!dev.step_rows || row >= dev.step_n || bank >= dev.n_banks ||
        !pulsar_gpu_tensor_copy(dev.bank_rows, (uint64_t)bank * rb, dev.step_rows, (uint64_t)row * rb, rb)) {
        fprintf(stderr, "pulsar: session: row %u of the step could not become bank %u's row\n", row, bank);
        return false;
    }
    return true;
}

bool pulsar_session::live_restore_bank(uint32_t bank) {
    const uint64_t rb = (uint64_t)engine->logits_width() * sizeof(float);
    if (bank >= dev.n_banks || !pulsar_gpu_tensor_copy(dev.live, 0, dev.bank_rows, (uint64_t)bank * rb, rb))
        return false;
    dev.host_stale = true;
    return true;
}

/* ---- draws ---------------------------------------------------------------------------------------------------- */

/* THE launch every draw goes through: n draws over `rows` (draw k reads row row_index[k], W wide) by reqs, the masks
 * (per draw, NULL = none) uploaded first; the outs land in dev.out.  false (said) on a bad launch -- a refused ROW is
 * its out's token -1 with its status, not a failure. */
static bool draw_launch(pulsar_session *s, const pulsar_gpu_tensor *rows, uint32_t W, const pulsar_gpu_tensor *id_map,
                        const uint32_t *row_index, uint32_t n, pulsar_gpu_sample_req *reqs,
                        const uint32_t *const *masks) {
    pulsar_session_dev_rows &dev = s->dev;
    const uint32_t words = (W + 31u) / 32u;
    if (W > (uint32_t)s->engine->logits_width()) {
        fprintf(stderr, "pulsar: session: a draw over rows %u wide, past the sampler's %d -- refusing\n", W,
                s->engine->logits_width());
        return false;
    }
    if (!rows || n == 0 || n > PULSAR_GPU_SAMPLE_ROWS_MAX) {
        fprintf(stderr, "pulsar: session: a draw of %u rows (at most %u) -- refusing\n", n, PULSAR_GPU_SAMPLE_ROWS_MAX);
        return false;
    }
    uint32_t n_mask = 0, n_rows = 0;
    for (uint32_t k = 0; k < n; k++) {
        reqs[k].mask_row = PULSAR_GPU_SAMPLE_NO_MASK;
        reqs[k].row = row_index[k];
        n_rows = row_index[k] + 1u > n_rows ? row_index[k] + 1u : n_rows;
        if (masks && masks[k]) {
            if (!pulsar_gpu_tensor_write(dev.mask, (uint64_t)n_mask * words * sizeof(uint32_t), masks[k],
                                         (uint64_t)words * sizeof(uint32_t)))
                return false;
            reqs[k].mask_row = n_mask++;
        }
    }
    if (!pulsar_gpu_sample_rows(rows, 0, W, W, n_rows, n, reqs, dev.mask, words, id_map, dev.scratch, dev.out, 0,
                                NULL, 0)) {
        fprintf(stderr, "pulsar: session: the device sampler failed over %u rows\n", n);
        return false;
    }
    return true;
}

bool pulsar_session::draw_rows_dev(const pulsar_gpu_tensor *rows, const uint32_t *row_index, uint32_t n,
                                   pulsar_gpu_sample_req *reqs) {
    return draw_launch(this, rows, (uint32_t)engine->logits_width(), NULL, row_index, n, reqs, NULL);
}

bool pulsar_session::draw_rows_width(const pulsar_gpu_tensor *rows, uint32_t width, const pulsar_gpu_tensor *id_map,
                                     const uint32_t *row_index, uint32_t n, pulsar_gpu_sample_req *reqs,
                                     pulsar_gpu_sample_out *out) {
    if (!draw_launch(this, rows, width, id_map, row_index, n, reqs, NULL)) return false;
    if (!pulsar_gpu_tensor_read(dev.out, 0, out, (uint64_t)n * sizeof(*out))) {
        fprintf(stderr, "pulsar: session: the device sampler's readback failed\n");
        return false;
    }
    return true;
}

bool pulsar_session::draw_rows(const pulsar_gpu_tensor *rows, const uint32_t *row_index, uint32_t n,
                               pulsar_gpu_sample_req *reqs, const uint32_t *const *masks, pulsar_gpu_sample_out *out) {
    if (!draw_launch(this, rows, (uint32_t)engine->logits_width(), NULL, row_index, n, reqs, masks)) return false;
    if (!pulsar_gpu_tensor_read(dev.out, 0, out, (uint64_t)n * sizeof(*out))) {
        fprintf(stderr, "pulsar: session: the device sampler's readback failed\n");
        return false;
    }
    return true;
}

/* a block drafter's sampled chains, every slot's draw on the device (pulsar_gpu_sample_chains): one launch, one
 * readback of n * slots tokens */
bool pulsar_session::draw_chains(const pulsar_gpu_tensor *cand, uint64_t cand_off, const pulsar_gpu_tensor *scores,
                                 uint64_t scores_off, uint32_t slots, uint32_t topk, uint32_t vocab,
                                 const pulsar_gpu_chain_req *reqs, uint32_t n, int32_t *tokens) {
    return pulsar_gpu_sample_chains(cand, cand_off, scores, scores_off, slots, topk, vocab, reqs, n, dev.chain_out,
                                    0) &&
           pulsar_gpu_tensor_read(dev.chain_out, 0, tokens, (uint64_t)n * slots * sizeof(int32_t));
}

static void draw_req(const pulsar_row_draw *d, pulsar_gpu_sample_req *q) {
    memset(q, 0, sizeof(*q));
    q->temperature = d->temperature;
    q->top_k = d->top_k;
    q->top_p = d->top_p;
    q->min_p = d->min_p;
    q->key = d->rng ? pulsar_sample_key(*d->rng, 0u) : 0u;
    q->flags = d->greedy ? PULSAR_GPU_SAMPLE_GREEDY : 0u;
    q->probe = -1;
}

/* rows k of the step (or the live row): each draw reads its stream's next key and steps it past the token */
static int session_draw(pulsar_session *s, const pulsar_gpu_tensor *rows, const uint32_t *row_index,
                        const pulsar_row_draw *d, uint32_t n, int *tokens) {
    pulsar_gpu_sample_req reqs[PULSAR_GPU_SAMPLE_ROWS_MAX];
    pulsar_gpu_sample_out out[PULSAR_GPU_SAMPLE_ROWS_MAX];
    const uint32_t *masks[PULSAR_GPU_SAMPLE_ROWS_MAX];
    if (n == 0 || n > PULSAR_GPU_SAMPLE_ROWS_MAX) return 1;
    for (uint32_t k = 0; k < n; k++) {
        if (!d[k].rng && !d[k].greedy && d[k].temperature > 0.0f) return 1;   /* a sampled draw needs its stream */
        draw_req(&d[k], &reqs[k]);
        masks[k] = d[k].mask;
    }
    if (!s->draw_rows(rows, row_index, n, reqs, masks, out)) return 1;
    for (uint32_t k = 0; k < n; k++) {
        tokens[k] = out[k].token;
        if (out[k].token >= 0 && d[k].rng) pulsar_sample_advance(d[k].rng, 1u);
    }
    return 0;
}

int pulsar_session::draw(const pulsar_row_draw *d) {
    const uint32_t zero = 0;
    int tok = PULSAR_SAMPLE_REFUSED;
    return session_draw(this, dev.live, &zero, d, 1, &tok) == 0 ? tok : PULSAR_SAMPLE_REFUSED;
}

int pulsar_session::sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!rng) return PULSAR_SAMPLE_REFUSED;
    const pulsar_row_draw d = {temperature, top_k, top_p, min_p, rng, false, NULL};
    return draw(&d);
}

int pulsar_session::argmax() {
    const pulsar_row_draw d = {0.0f, 0, 1.0f, 0.0f, NULL, true, NULL};
    return draw(&d);
}

/* the argmax with one id masked out (a forced continuation kept off EOS) */
int pulsar_session::argmax_excluding(int excluded_id) {
    const uint32_t W = (uint32_t)engine->logits_width();
    std::vector<uint32_t> m(mask_words(this), ~0u);
    if (excluded_id >= 0 && (uint32_t)excluded_id < W) m[(uint32_t)excluded_id >> 5] &= ~(1u << (excluded_id & 31));
    const pulsar_row_draw d = {0.0f, 0, 1.0f, 0.0f, NULL, true, m.data()};
    return draw(&d);
}

int pulsar_session::draw_step_rows(const uint32_t *rows, const pulsar_row_draw *d, uint32_t n, int *tokens) {
    for (uint32_t k = 0; k < n; k++)
        if (!dev.step_rows || rows[k] >= dev.step_n) {
            fprintf(stderr, "pulsar: session: a draw of row %u of a step that headed %u -- refusing\n", rows[k],
                    dev.step_rows ? dev.step_n : 0u);
            return 1;
        }
    return session_draw(this, dev.step_rows, rows, d, n, tokens);
}

/* ---- logprobs ----------------------------------------------------------------------------------------------- */

/* One row's logprobs (the scored token when token >= 0, the top k): the host scorers' bytes (pulsar_cuda_sampler.cu). */
static bool session_logprobs(pulsar_session *s, const pulsar_gpu_tensor *rows, uint32_t row, int token, int k,
                             pulsar_gpu_logprob_row *L) {
    const uint32_t W = (uint32_t)s->engine->logits_width();
    /* the scored token rides the sampler's out block, where a lane's own draw would sit */
    const pulsar_gpu_sample_out t = {token, 0u, 0.0f, 0u};
    if (!pulsar_gpu_tensor_write(s->dev.out, 0, &t, sizeof(t)) ||
        !pulsar_gpu_logprobs_rows(rows, (uint64_t)row * W * sizeof(float), W, W, 1, NULL, 0, NULL,
                                  token >= 0 ? s->dev.out : NULL, 0, sizeof(t) / sizeof(int32_t), (uint32_t)k,
                                  s->dev.lp) ||
        !pulsar_gpu_tensor_read(s->dev.lp, 0, L, sizeof(*L))) {
        fprintf(stderr, "pulsar: session: the device logprobs failed\n");
        return false;
    }
    return true;
}

static int clip_k(int k) { return k < 0 ? 0 : k > (int)PULSAR_GPU_LOGPROB_TOPK_MAX ? (int)PULSAR_GPU_LOGPROB_TOPK_MAX : k; }
static pulsar_token_score score(const pulsar_gpu_token_score &t) { return pulsar_token_score{t.id, t.logit, t.logprob}; }

/* the host scorers' contracts: top fills k entries (k clipped to the width; -1 / NEG_INF past the finite ones) and
 * returns that count, 0 for a row with no finite value; the token's 1, or 0 with `out` untouched */
static void fill_top(const pulsar_gpu_logprob_row &L, pulsar_token_score *top, int k) {
    for (int i = 0; i < k; i++)
        top[i] = (uint32_t)i < L.n_top ? score(L.top[i]) : pulsar_token_score{-1, PULSAR_NEG_INF, PULSAR_NEG_INF};
}

int pulsar_session::top_logprobs(pulsar_token_score *out, int k) {
    if (!out || k <= 0) return 0;
    k = clip_k(k);
    pulsar_gpu_logprob_row L;
    if (!session_logprobs(this, dev.live, 0, -1, k, &L)) L.n_top = 0;
    fill_top(L, out, k);
    return (int)L.n_top;
}

int pulsar_session::token_logprob(int token, pulsar_token_score *out) {
    if (!out || token < 0 || token >= engine->logits_width()) return 0;
    pulsar_gpu_logprob_row L;
    if (!session_logprobs(this, dev.live, 0, token, 0, &L) || !L.have_token) return 0;
    *out = score(L.token);
    return 1;
}

int pulsar_session::step_row_logprobs(uint32_t row, int token, pulsar_token_score *chosen, int *have_chosen,
                                      pulsar_token_score *top, int k, int *n_top) {
    if (!dev.step_rows || row >= dev.step_n) return 1;
    k = clip_k(k);
    pulsar_gpu_logprob_row L;
    if (!session_logprobs(this, dev.step_rows, row, token, k, &L)) return 1;
    *have_chosen = token >= 0 && token < engine->logits_width() ? (int)L.have_token : 0;
    if (*have_chosen) *chosen = score(L.token);
    if (top) fill_top(L, top, k);
    *n_top = (int)L.n_top;
    return 0;
}

bool pulsar_session::rows_digest(const pulsar_gpu_tensor *rows, uint32_t row0, uint32_t n, uint64_t *out) {
    const uint32_t W = (uint32_t)engine->logits_width();
    if (!rows || !pulsar_gpu_rows_digest(rows, (uint64_t)row0 * W * sizeof(float), n, W, dev.out, dev_digest_off()) ||
        !pulsar_gpu_tensor_read(dev.out, dev_digest_off(), out, sizeof(*out))) {
        fprintf(stderr, "pulsar: session: the rows' digest failed\n");
        return false;
    }
    return true;
}

int pulsar_session::copy_logits(float *out, int cap) {
    const int W = engine->logits_width();
    if (!out || cap < W) return 0;
    const float *h = live_host();
    if (!h) return 0;
    memcpy(out, h, (size_t)W * sizeof(out[0]));
    return W;
}

int pulsar_session::set_logits(const float *row, int n) {
    if (!row || n != engine->logits_width()) return 1;
    return live_from_host(row) ? 0 : 1;
}

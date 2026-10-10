/* Engram on the DeepSeek forward (L242 slice 5, L269 W2): the load-time bind of the
 * artifact's Engram layers and the graph's per-step staging and layer op.  The host
 * half it builds on -- the hash, the row tables and the pread pool -- is engram.cpp,
 * which the host-only Engram tests compile on its own. */

#include "pulsar_engine_internal.h"

#include <sys/stat.h>

/* ---- slice 5: Engram on the DeepSeek forward (L269 W2) -----------------------
 *
 * engram_bind reads what the artifact carries (tools/container/kv.py
 * _engram_layout_triples): the blocks, their row counts, the hash layout, and
 * each row table's file name beside the container.  The `wkv` / q / k tensors
 * keep their HF names (names.py: the engine binds them by those or not at all),
 * so a block is found under its SOURCE layer (pulsar_layer_source) -- the same
 * index the row table's header and the hash multipliers were built for. */

#define ENGRAM_HEAD_DIM 256u
#define ENGRAM_IN       (PULSAR_ENGRAM_N_COLS * ENGRAM_HEAD_DIM)   /* 24 x 256 = 6144: the `wkv` input */

static bool engram_refuse(const char *what) {
    fprintf(stderr, "pulsar: engram: %s -- refusing\n", what);
    return false;
}

/* An integer array key of exactly `n` entries into `out` (as int64; the element
 * type is any of the integer codes the container writes). */
static bool engram_int_array(const pulsar_model *m, const char *key, int64_t *out, uint64_t n) {
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.len != n) {
        fprintf(stderr, "pulsar: engram: %s is missing or does not hold %llu entries\n", key, (unsigned long long)n);
        return false;
    }
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < n; i++) {
        switch (arr.type) {
        case PULSAR_META_UINT32: { uint32_t v; if (!cursor_u32(&c, &v)) return false; out[i] = v; break; }
        case PULSAR_META_INT32:  { int32_t v; if (!cursor_read(&c, &v, 4)) return false; out[i] = v; break; }
        case PULSAR_META_UINT64: { uint64_t v; if (!cursor_u64(&c, &v) || v > (uint64_t)INT64_MAX) return false;
                                   out[i] = (int64_t)v; break; }
        case PULSAR_META_INT64:  { int64_t v; if (!cursor_read(&c, &v, 8)) return false; out[i] = v; break; }
        default:
            fprintf(stderr, "pulsar: engram: %s is not an integer array\n", key);
            return false;
        }
    }
    return true;
}

static uint64_t engram_meta_len(const pulsar_model *m, const char *key) {
    pulsar_array_ref arr;
    return model_get_array(m, key, &arr) ? arr.len : 0;
}

static float engram_elt_f32(const pulsar_tensor *t, const unsigned char *p, uint64_t i) {
    if (t->type == PULSAR_TENSOR_F32) { float v; memcpy(&v, p + i * 4, 4); return v; }
    uint16_t b; memcpy(&b, p + i * 2, 2);
    const uint32_t u = (uint32_t)b << 16;
    float v; memcpy(&v, &u, 4);
    return v;
}

bool engram_bind(pulsar_weights *w, const pulsar_model *m, const char *model_path) {
    w->engram = NULL;
    const uint64_t n = engram_meta_len(m, "deepseek4.engram.layers");
    if (n == 0) return true;   /* no Engram in this model */
    if (n > PULSAR_ENGRAM_MAX_LAYERS) return engram_refuse("deepseek4.engram.layers names more layers than V4.1 has");
    if (!engram_meta_len(m, "deepseek4.engram.token_map"))
        return engram_refuse("the artifact declares Engram layers but carries no hash layout (deepseek4.engram.token_map;"
                             " build it with tools/container build.py --engram-layout) -- V4.1 without Engram is not "
                             "the model");
    pulsar_engram_model *em = (pulsar_engram_model *)xcalloc(1, sizeof *em);
    em->n_layers = (uint32_t)n;
    w->engram = em;
    int64_t blocks[PULSAR_ENGRAM_MAX_LAYERS], rows[PULSAR_ENGRAM_MAX_LAYERS];
    uint32_t cv = 0, pad = 0;
    if (!engram_int_array(m, "deepseek4.engram.layers", blocks, n) ||
        !engram_int_array(m, "deepseek4.engram.n_rows", rows, n) ||
        !model_get_u32(m, "deepseek4.engram.compressed_vocab", &cv) ||
        !model_get_u32(m, "deepseek4.engram.pad_compressed_id", &pad))
        return engram_refuse("the Engram layout keys are incomplete");
    const uint64_t n_vocab = engram_meta_len(m, "deepseek4.engram.token_map");
    if (n_vocab != PULSAR_N_VOCAB) return engram_refuse("the Engram token map does not cover the vocabulary");
    if (cv == 0 || pad >= cv) return engram_refuse("the compressed vocab or its pad id is out of range");
    int64_t *tmp = (int64_t *)xmalloc(n_vocab * sizeof(int64_t));
    em->token_map = (int32_t *)xmalloc(n_vocab * sizeof(int32_t));
    em->multipliers = (int64_t *)xmalloc(n * PULSAR_ENGRAM_MAX_NGRAM * sizeof(int64_t));
    em->primes = (uint32_t *)xmalloc(n * PULSAR_ENGRAM_N_COLS * sizeof(uint32_t));
    em->offsets = (uint64_t *)xmalloc(n * PULSAR_ENGRAM_N_COLS * sizeof(uint64_t));
    bool ok = engram_int_array(m, "deepseek4.engram.token_map", tmp, n_vocab);
    for (uint64_t i = 0; ok && i < n_vocab; i++) {
        if (tmp[i] < 0 || (uint64_t)tmp[i] >= cv) ok = engram_refuse("a token map entry is outside the compressed vocab");
        else em->token_map[i] = (int32_t)tmp[i];
    }
    ok = ok && engram_int_array(m, "deepseek4.engram.multipliers", em->multipliers, n * PULSAR_ENGRAM_MAX_NGRAM) &&
         engram_int_array(m, "deepseek4.engram.primes", tmp, n * PULSAR_ENGRAM_N_COLS);
    for (uint64_t i = 0; ok && i < n * PULSAR_ENGRAM_N_COLS; i++) {
        if (tmp[i] <= 0 || tmp[i] > (int64_t)UINT32_MAX) ok = engram_refuse("a bucket prime is out of range");
        else em->primes[i] = (uint32_t)tmp[i];
    }
    ok = ok && engram_int_array(m, "deepseek4.engram.offsets", tmp, n * PULSAR_ENGRAM_N_COLS);
    for (uint64_t i = 0; ok && i < n * PULSAR_ENGRAM_N_COLS; i++) em->offsets[i] = (uint64_t)tmp[i];
    free(tmp);
    if (!ok) return false;
    /* the layout's own invariants (pulsar_engram_hash_pos): odd multipliers below INT64_MAX / compressed_vocab,
     * every bucket inside its table */
    for (uint64_t l = 0; l < n; l++) {
        for (uint32_t i = 0; i < PULSAR_ENGRAM_MAX_NGRAM; i++) {
            const int64_t mu = em->multipliers[l * PULSAR_ENGRAM_MAX_NGRAM + i];
            if (mu <= 0 || (mu & 1) == 0 || mu > INT64_MAX / (int64_t)cv)
                return engram_refuse("a hash multiplier breaks the layout's no-overflow invariant");
        }
        for (uint32_t c = 0; c < PULSAR_ENGRAM_N_COLS; c++) {
            const uint64_t i = l * PULSAR_ENGRAM_N_COLS + c;
            if (rows[l] <= 0 || em->offsets[i] + em->primes[i] > (uint64_t)rows[l])
                return engram_refuse("a hash bucket runs past its table");
        }
        em->num_embeddings[l] = (uint64_t)rows[l];
    }
    em->layout.n_vocab = (uint32_t)n_vocab;
    em->layout.compressed_vocab = cv;
    em->layout.pad_compressed_id = pad;
    em->layout.n_layers = (uint32_t)n;
    em->layout.token_map = em->token_map;
    em->layout.multipliers = em->multipliers;
    em->layout.primes = em->primes;
    em->layout.offsets = em->offsets;
    em->layout.num_embeddings = em->num_embeddings;

    /* the container's directory: the row tables live beside it (kv.py names them) */
    char dir[4096];
    struct stat mst;
    const char *mp = model_path ? model_path : "";
    if (stat(mp, &mst) == 0 && S_ISDIR(mst.st_mode)) {
        snprintf(dir, sizeof dir, "%s/", mp);
    } else {
        const char *slash = strrchr(mp, '/');
        snprintf(dir, sizeof dir, "%.*s", slash ? (int)(slash - mp + 1) : 0, mp);
    }
    for (uint64_t l = 0; l < n; l++) {
        const uint32_t il = (uint32_t)blocks[l];
        if (blocks[l] < 0 || il >= PULSAR_N_LAYER || (l && blocks[l] <= blocks[l - 1]))
            return engram_refuse("deepseek4.engram.layers is not ascending blocks of this model");
        const uint32_t src = pulsar_layer_source(il);
        em->block[l] = il;
        char key[96], name[128], path[4096 + 256];
        pulsar_str file = {NULL, 0};
        snprintf(key, sizeof key, "deepseek4.engram.rows_file.%llu", (unsigned long long)l);
        if (!model_get_string(m, key, &file)) return engram_refuse("a row table is not named (deepseek4.engram.rows_file.N)");
        snprintf(path, sizeof path, "%s%.*s", dir, (int)file.len, file.ptr);
        if (!pulsar_engram_table_open(&em->table[l], path, src, em->num_embeddings[l]))
            return engram_refuse("a row table does not open against this model");
        pulsar_layer_weights *L = &w->layer[il];
        snprintf(name, sizeof name, "layers.%u.engram.wkv.weight", src);
        L->engram_wkv = model_find_tensor(m, name);
        snprintf(name, sizeof name, "layers.%u.engram.q_weight", src);
        const pulsar_tensor *q = model_find_tensor(m, name);
        snprintf(name, sizeof name, "layers.%u.engram.k_weight", src);
        const pulsar_tensor *k = model_find_tensor(m, name);
        const pulsar_tensor *wkv = L->engram_wkv;
        const uint64_t out = (uint64_t)(PULSAR_N_HC + 1u) * PULSAR_N_EMBD;
        if (!wkv || wkv->type != PULSAR_TENSOR_MXFP8_LT || wkv->ndim != 2 || wkv->dim[0] != ENGRAM_IN ||
            wkv->dim[1] != out)
            return engram_refuse("a block's `wkv` is missing or not mxfp8_lt [24*256 -> (n_hc+1)*dim]");
        for (const pulsar_tensor *t : { q, k })
            if (!t || (t->type != PULSAR_TENSOR_F32 && t->type != PULSAR_TENSOR_BF16) || t->ndim != 2 ||
                t->dim[0] != PULSAR_N_EMBD || t->dim[1] != PULSAR_N_HC)
                return engram_refuse("a block's q_weight / k_weight is missing or not [n_hc][dim] f32 / bf16");
        if (tensor_map_base(m, wkv) != tensor_map_base(m, L->attn_norm))
            return engram_refuse("a block's `wkv` is not in its layer's mapping");
        L->engram_slot = (uint32_t)l;
    }
    em->io = pulsar_engram_io_create(PULSAR_ENGRAM_IO_THREADS);
    if (!em->io) return engram_refuse("the row reader pool did not start");
    for (uint64_t l = 0; l < n; l++)
        fprintf(stderr, "pulsar: engram: block %u (source layer %u): %s, %llu rows, %u pread threads\n", em->block[l],
                pulsar_layer_source(em->block[l]), em->table[l].path, (unsigned long long)em->table[l].n_rows,
                PULSAR_ENGRAM_IO_THREADS);
    return true;
}

bool engram_upload(pulsar_weights *w, const pulsar_model *m) {
    pulsar_engram_model *em = w->engram;
    if (!em) return true;
    const uint64_t hd = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    float *qk = (float *)xmalloc(hd * sizeof(float));
    bool ok = true;
    for (uint32_t l = 0; ok && l < em->n_layers; l++) {
        const uint32_t src = pulsar_layer_source(em->block[l]);
        char name[128];
        snprintf(name, sizeof name, "layers.%u.engram.q_weight", src);
        const pulsar_tensor *q = model_find_tensor(m, name);
        snprintf(name, sizeof name, "layers.%u.engram.k_weight", src);
        const pulsar_tensor *k = model_find_tensor(m, name);
        const unsigned char *qp = (const unsigned char *)tensor_map_base(m, q) + q->abs_offset;
        const unsigned char *kp = (const unsigned char *)tensor_map_base(m, k) + k->abs_offset;
        /* the reference's `q_weight.float() * k_weight.float()` -- only ever used as the product */
        for (uint64_t i = 0; i < hd; i++) qk[i] = engram_elt_f32(q, qp, i) * engram_elt_f32(k, kp, i);
        em->qk[l] = pulsar_gpu_tensor_alloc(hd * sizeof(float));
        ok = em->qk[l] && pulsar_gpu_tensor_write(em->qk[l], 0, qk, hd * sizeof(float));
    }
    free(qk);
    return ok || engram_refuse("could not upload the q*k gate weights");
}

void engram_model_free(pulsar_weights *w) {
    pulsar_engram_model *em = w ? w->engram : NULL;
    if (!em) return;
    pulsar_engram_io_destroy(em->io);
    for (uint32_t l = 0; l < PULSAR_ENGRAM_MAX_LAYERS; l++) {
        pulsar_engram_table_close(&em->table[l]);
        pulsar_gpu_tensor_free(em->qk[l]);
    }
    free(em->token_map);
    free(em->multipliers);
    free(em->primes);
    free(em->offsets);
    free(em);
    w->engram = NULL;
}


/* ---- the graph's half: staging, the ring, the layer op ---------------------- */

bool gpu_graph_engram_alloc(pulsar_gpu_graph *g, const pulsar_weights *w, uint32_t cap, uint32_t n_banks) {
    memset(&g->engram, 0, sizeof g->engram);
    if (!w->engram) return true;
    const pulsar_engram_model *em = w->engram;
    g->engram.model = em;
    g->engram.cap = cap;
    g->engram.n_banks = n_banks ? n_banks : 1u;
    const uint64_t rec = (uint64_t)PULSAR_ENGRAM_N_COLS * PULSAR_ENGRAM_ROW_BYTES;
    g->engram.rows = pulsar_gpu_tensor_alloc((uint64_t)cap * rec);
    g->engram.x_key = pulsar_gpu_tensor_alloc((uint64_t)cap * ENGRAM_IN * sizeof(float));
    g->engram.kv = pulsar_gpu_tensor_alloc((uint64_t)cap * (PULSAR_N_HC + 1u) * PULSAR_N_EMBD * sizeof(float));
    for (uint32_t l = 0; l < em->n_layers; l++) {
        g->engram.host_rows[l] = (unsigned char *)xmalloc((size_t)cap * rec);
        g->engram.ids[l] = (uint64_t *)xmalloc((size_t)cap * PULSAR_ENGRAM_N_COLS * sizeof(uint64_t));
    }
    const size_t ring = (size_t)g->engram.n_banks * PULSAR_ENGRAM_RING;
    g->engram.ring_tok = (int32_t *)xmalloc(ring * sizeof(int32_t));
    g->engram.ring_pos = (uint32_t *)xmalloc(ring * sizeof(uint32_t));
    for (size_t i = 0; i < ring; i++) g->engram.ring_pos[i] = UINT32_MAX;
    return g->engram.rows && g->engram.x_key && g->engram.kv;
}

static void engram_drain(pulsar_gpu_graph *g) {
    for (uint32_t l = 0; l < PULSAR_ENGRAM_MAX_LAYERS; l++) {
        if (g->engram.pending[l]) (void)pulsar_engram_gather_wait(g->engram.pending[l]);
        g->engram.pending[l] = NULL;
    }
    g->engram.pending_n = 0;
}

void gpu_graph_engram_release(pulsar_gpu_graph *g) {
    engram_drain(g);
    pulsar_gpu_tensor_free(g->engram.rows);
    pulsar_gpu_tensor_free(g->engram.x_key);
    pulsar_gpu_tensor_free(g->engram.kv);
    for (uint32_t l = 0; l < PULSAR_ENGRAM_MAX_LAYERS; l++) {
        free(g->engram.host_rows[l]);
        free(g->engram.ids[l]);
    }
    free(g->engram.ring_tok);
    free(g->engram.ring_pos);
    memset(&g->engram, 0, sizeof g->engram);
}

void gpu_graph_engram_forget(pulsar_gpu_graph *g, uint32_t bank) {
    if (!g->engram.model || bank >= g->engram.n_banks) return;
    for (uint32_t i = 0; i < PULSAR_ENGRAM_RING; i++) g->engram.ring_pos[(size_t)bank * PULSAR_ENGRAM_RING + i] = UINT32_MAX;
}

static void engram_ring_put(pulsar_gpu_graph *g, uint32_t bank, uint32_t pos, int32_t tok) {
    const size_t i = (size_t)bank * PULSAR_ENGRAM_RING + (pos & (PULSAR_ENGRAM_RING - 1u));
    g->engram.ring_tok[i] = tok;
    g->engram.ring_pos[i] = pos;
}

bool gpu_graph_engram_stage(pulsar_gpu_graph *g, const int *tok, const int32_t *pos, const int32_t *bank,
                            uint32_t n, const int *hist, uint32_t hist_len) {
    const pulsar_engram_model *em = g->engram.model;
    if (!em) return true;
    engram_drain(g);   /* a step that failed before its Engram layer left its gathers in flight */
    if (n == 0 || n > g->engram.cap) return engram_refuse("a step's rows do not fit the Engram scratch");
    const uint32_t cur = gpu_graph_cur_bank(g);
    if (hist) {   /* the prompt is authoritative for the positions before the step */
        const uint32_t b = bank ? (uint32_t)bank[0] : cur;
        if (b >= g->engram.n_banks || pos[0] < 0 || (uint32_t)pos[0] > hist_len)
            return engram_refuse("the step's history is out of range");
        const uint32_t p0 = (uint32_t)pos[0];
        for (uint32_t q = p0 > PULSAR_ENGRAM_MAX_NGRAM - 1u ? p0 - (PULSAR_ENGRAM_MAX_NGRAM - 1u) : 0u; q < p0; q++)
            engram_ring_put(g, b, q, hist[q]);
    }
    for (uint32_t r = 0; r < n; r++) {
        const uint32_t b = bank ? (uint32_t)bank[r] : cur;
        if (b >= g->engram.n_banks || pos[r] < 0) return engram_refuse("a row's bank or position is out of range");
        const uint32_t p = (uint32_t)pos[r];
        if (tok[r] < 0 || (uint32_t)tok[r] >= em->layout.n_vocab)
            return engram_refuse("a token outside the text vocabulary reached the n-gram hash (image spans are not "
                                 "an Engram input on this layout)");
        engram_ring_put(g, b, p, tok[r]);
        /* the n-gram ending at p: oldest first, as much as the sequence has (pulsar_engram_hash_pos pads the rest) */
        int32_t win[PULSAR_ENGRAM_MAX_NGRAM];
        const uint32_t back = p < PULSAR_ENGRAM_MAX_NGRAM - 1u ? p : PULSAR_ENGRAM_MAX_NGRAM - 1u;
        for (uint32_t i = 0; i <= back; i++) {
            const uint32_t q = p - back + i;
            const size_t at = (size_t)b * PULSAR_ENGRAM_RING + (q & (PULSAR_ENGRAM_RING - 1u));
            if (g->engram.ring_pos[at] != q) {
                fprintf(stderr, "pulsar: engram: bank %u position %u: the token at %u is not in the history ring "
                                "(that slot holds position %u) -- refusing the step\n", b, p, q,
                        g->engram.ring_pos[at]);
                return false;
            }
            win[i] = g->engram.ring_tok[at];
        }
        for (uint32_t l = 0; l < em->n_layers; l++) {
            uint32_t cols[PULSAR_ENGRAM_N_COLS];
            pulsar_engram_hash_pos(&em->layout, l, win, back + 1u, back, cols);
            for (uint32_t c = 0; c < PULSAR_ENGRAM_N_COLS; c++)
                g->engram.ids[l][(size_t)r * PULSAR_ENGRAM_N_COLS + c] = cols[c];
        }
    }
    for (uint32_t l = 0; l < em->n_layers; l++) {
        g->engram.pending[l] = pulsar_engram_gather_start(em->io, &em->table[l], g->engram.ids[l],
                                                          n * PULSAR_ENGRAM_N_COLS, g->engram.host_rows[l]);
        if (!g->engram.pending[l]) {
            engram_drain(g);
            return engram_refuse("a row gather did not start");
        }
    }
    g->engram.pending_n = n;
    return true;
}

bool gpu_graph_engram_apply(pulsar_gpu_graph *g, const pulsar_model *model, const pulsar_layer_weights *layer,
                            uint32_t il, uint32_t n_tokens) {
    if (!layer->engram_wkv) return true;
    const pulsar_engram_model *em = g->engram.model;
    const uint32_t l = layer->engram_slot;
    if (!em || l >= em->n_layers || em->block[l] != il || !g->engram.pending[l] || g->engram.pending_n != n_tokens) {
        fprintf(stderr, "pulsar: engram: block %u has no staged gather for this %u-row step -- refusing\n", il,
                n_tokens);
        return false;
    }
    const int got = pulsar_engram_gather_wait(g->engram.pending[l]);
    g->engram.pending[l] = NULL;
    if (!got) return engram_refuse("a row gather failed (message above)");
    static bool announced[PULSAR_MAX_LAYER];
    if (!announced[il]) {
        announced[il] = true;
        fprintf(stderr, "pulsar: engram: block %u runs rows -> MX slot -> wkv -> gate + add (first step %u rows)\n",
                il, n_tokens);
    }
    const uint64_t rec = (uint64_t)PULSAR_ENGRAM_N_COLS * PULSAR_ENGRAM_ROW_BYTES;
    bool ok = pulsar_gpu_tensor_write(g->engram.rows, 0, g->engram.host_rows[l], (uint64_t)n_tokens * rec) != 0;
    ok = ok && pulsar_gpu_engram_rows_emit(g->engram.x_key, g->engram.rows, n_tokens) != 0;
    ok = ok && pulsar_linear_slot(g->engram.kv, model, layer->engram_wkv, ENGRAM_IN, 0,
                                  (uint64_t)(PULSAR_N_HC + 1u) * PULSAR_N_EMBD, g->engram.x_key, n_tokens);
    pulsar_gpu_act_slot_drop(g->engram.x_key);
    ok = ok && pulsar_gpu_engram_gate_add(g->batch_cur_hc, g->engram.kv, em->qk[l], NULL, n_tokens, PULSAR_N_HC,
                                          PULSAR_N_EMBD, PULSAR_RMS_EPS) != 0;
    if (!ok) fprintf(stderr, "pulsar: engram: block %u device op failed\n", il);
    return ok;
}

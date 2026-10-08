/* spec_dspark.cpp -- DeepSeek V4's verify hooks and the DSpark drafter behind the speculation round
 * API (L272 P1 S2).  The bodies are session_spec.cpp's as they stood at 8e89f5b0, moved here verbatim
 * behind pulsar_spec_target_ops / pulsar_drafter_ops (spec_ops.h): the frontier snapshot and restore,
 * the Stage-B roll-forward, the capture and readback arming, the row readers over the graph's device
 * rows, the prompt-window seed, the drafter's row absorb, the drafting pass both lanes share (noise-token
 * forward, markov chain, min-p prefilter, confidence head, the stop rule) and the deferred-chain harvest.  Nothing
 * here is reached except through the two tables at the end. */
#include "pulsar_engine_internal.h"
#include "pulsar_nvtx.h"
#include "spec_depth.h"
#include "spec_internal.h"

/** Snapshot of every layer's compressed-row frontier, taken before a
 * speculative block so a rejected draft can be rolled back exactly. */
typedef struct {
    uint32_t n_comp[PULSAR_MAX_LAYER];        ///< compressed rows per kv source
} pulsar_spec_frontier;

/* PULSAR_DSPARK_DUMP set: the offline dump wants the refined ids on the host
 * at draft time (the immediate harvest path) and the drafter's f32 rows
 * (main_x, target_h) written -- gpu_decode keeps those rows only when this
 * says so.  Read once per process. */
int gpu_graph_spec_dump_active(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *a = getenv("PULSAR_DSPARK_DUMP");
        cached = (a && a[0]) ? 1 : 0;
    }
    return cached;
}

/* Diagnostic: dump the DSpark drafter's per-step inputs (target_h[3], main_x)
 * and pre-markov base logits (spec_logits row 0) so an off-box reference forward
 * can be diffed against ds4 to localize acceptance loss. Enabled by
 * PULSAR_DSPARK_DUMP=<path>; caps at PULSAR_DSPARK_DUMP_STEPS (default 8) records.
 * Record layout (little-endian): pos i32, first_token i32, then f32 arrays
 * target_h[0..2] (PULSAR_N_EMBD each), main_x (PULSAR_N_EMBD), base0 (PULSAR_N_VOCAB). */
static void dspark_dump_step(pulsar_gpu_graph *g, int pos, int first_token,
                             const int32_t *refined_ids, int n_draft) {
    /* Read once (no-hot-path-flags): this is called every spec step, and both
     * switches are fixed at start.  The common case is "not dumping", which
     * must cost a single cached load, not two getenv lookups. */
    static const char *path = NULL;
    static int max_steps = 8;
    static int env_read = 0;
    if (!env_read) {
        env_read = 1;
        const char *p = getenv("PULSAR_DSPARK_DUMP");
        path = (p && p[0]) ? p : NULL;
        const char *lim = getenv("PULSAR_DSPARK_DUMP_STEPS");
        if (lim && lim[0]) max_steps = atoi(lim);
    }
    if (!path) return;
    static int dumped = 0;
    if (dumped >= max_steps) return;

    const uint64_t hcw = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    float *emb = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(float));
    float *voc = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    float *hc = (float *)xmalloc((size_t)hcw * sizeof(float));
    FILE *f = fopen(path, dumped == 0 ? "wb" : "ab");
    if (!f) { free(emb); free(voc); free(hc); return; }
    /* Lean mode (PULSAR_DSPARK_DUMP_LEAN=1): confidence-head training records only —
     * hdr + refined_ids + the post-hc_head hidden rows (batch_ffn_cur) that the
     * engine's confidence kernel consumes (Step 5c), ~64 KB/step at draft=4
     * instead of ~1 MB. Bulk-collectable for the drafter-retune Phase 0. */
    static int lean = -1;
    if (lean < 0) lean = getenv("PULSAR_DSPARK_DUMP_LEAN") != NULL;
    if (lean) {
        int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
        for (int p = 0; p < n_draft; p++) {
            memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
            (void)pulsar_gpu_tensor_read(g->batch_ffn_cur, (uint64_t)p * PULSAR_N_EMBD * 4,
                                      emb, (uint64_t)PULSAR_N_EMBD * 4);
            fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
        }
        fclose(f);
        dumped++;
        if ((dumped & 1023) == 1)
            fprintf(stderr, "pulsar: dspark lean dump step %d pos=%d -> %s\n", dumped, pos, path);
        free(emb); free(voc); free(hc);
        return;
    }
    /* Record: pos, tok, n_draft, refined_ids[0..n_draft], target_h[3], main_x,
     * base0(vocab), then cur_hc for each of the n_draft block positions. n_draft
     * is fixed per run -> fixed record size. Used to validate offline whether the
     * DSpark confidence head predicts per-position acceptance on our requant. */
    int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
    for (int i = 0; i < 3; i++) {
        memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
        (void)pulsar_gpu_tensor_read(g->dspark_target_h[i], 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
        fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    }
    memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
    (void)pulsar_gpu_tensor_read(g->dspark_main_x, 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
    fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    memset(voc, 0, (size_t)PULSAR_N_VOCAB * sizeof(float));
    (void)gpu_graph_read_spec_logits_row(g, 0, voc);
    fwrite(voc, sizeof(float), PULSAR_N_VOCAB, f);
    /* block-2 output hidden (pre-hc_head) for each draft position [HC, EMBD]. */
    for (int p = 0; p < n_draft; p++) {
        memset(hc, 0, (size_t)hcw * sizeof(float));
        (void)pulsar_read_hc_carrier_f32(g->batch_cur_hc, (uint64_t)p * hcw, hc, hcw);
        fwrite(hc, sizeof(float), (size_t)hcw, f);
    }
    fclose(f);
    dumped++;
    fprintf(stderr, "pulsar: dspark dump step %d pos=%d tok=%d n_draft=%d -> %s\n",
            dumped, pos, first_token, n_draft, path);
    free(emb); free(voc); free(hc);
}

static void spec_frontier_free(pulsar_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}

/* Build the batched-copy descriptor tables for the frontier snapshot/restore
 * copy sets. All source/destination tensors are fixed allocations, so the
 * tables are built once and replayed with one kernel launch per direction
 * (~126 cudaMemcpy launches per snapshot, again per restore, before them).
 *
 * This is the ONE copy path (L190 D5).  prepare() rejects only impossible
 * states -- a NULL or undersized tensor, a byte count that is not a multiple
 * of 16 (every tensor here is a 256 B-aligned allocation) -- and device
 * allocation failure; on either the snapshot FAILS, loudly.  The per-tensor
 * cudaMemcpy loop it used to fall back to was a second implementation
 * selected exactly when something was wrong, announced once, and never
 * exercised by a gate; it is deleted.  Returns false with the flag left
 * clear, so the next snapshot retries a transient allocation failure.
 *
 * ⚠ THE FLAG IS SET ON SUCCESS, NOT ON ENTRY.  It used to be set here, before
 * either prepare() below was attempted, so a single transient failure latched
 * the degraded state forever with nothing logged and no way back.  prepare()
 * only builds descriptor tables, so retrying is cheap. */
static bool spec_frontier_copy_tables_init(pulsar_gpu_graph *g) {
    if (g->spec_frontier_copy_init) return true;
    pulsar_gpu_tensor *dst[PULSAR_MAX_LAYER * 4];
    pulsar_gpu_tensor *src[PULSAR_MAX_LAYER * 4];
    uint64_t bytes[PULSAR_MAX_LAYER * 4];
    uint32_t n = 0;
    uint64_t mx = 0;
    /* CSA2 (L218): the only recurrent state is the ratio-2 kv sources' pending
     * group (kv + score); ratio-1 sources and member layers carry none.
     * V4 has a SECOND such lane: the indexer's own compressor, which exists
     * exactly where the profile says so (s119 found this copy set missing it;
     * a rejected round then rolled the attention lane back and left the
     * indexer's carry where the rejected draft had put it). */
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (!gpu_graph_layer_has_comp_state(il)) continue;
        const uint64_t ab = pulsar_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        dst[n] = g->spec_attn_state_kv[il];    src[n] = g->layer_attn_state_kv[il];    bytes[n++] = ab;
        dst[n] = g->spec_attn_state_score[il]; src[n] = g->layer_attn_state_score[il]; bytes[n++] = ab;
        if (ab > mx) mx = ab;
        if (g->layer_index_state_kv[il]) {
            const uint64_t ib = pulsar_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            dst[n] = g->spec_index_state_kv[il];
            src[n] = g->layer_index_state_kv[il];
            bytes[n++] = ib;
            dst[n] = g->spec_index_state_score[il];
            src[n] = g->layer_index_state_score[il];
            bytes[n++] = ib;
            if (ib > mx) mx = ib;
        }
    }
    if (n == 0) {
        /* No stateful compressor: there is genuinely nothing to copy (copy_n
         * stays 0 and the run is skipped). Done, not degraded. */
        g->spec_frontier_copy_init = 1;
        return true;
    }
    void *snap = pulsar_gpu_batched_copy_prepare(dst, src, bytes, n);
    /* restore = the same set with src/dst swapped */
    void *restore = pulsar_gpu_batched_copy_prepare(src, dst, bytes, n);
    if (!snap || !restore) {
        /* Take neither: the two directions must be the same copy set. */
        pulsar_gpu_batched_copy_free(snap);
        pulsar_gpu_batched_copy_free(restore);
        fprintf(stderr,
                "pulsar: spec frontier batched-copy prepare failed (%u descriptors, max %llu B) "
                "-- refusing the snapshot (no per-tensor copy loop; L190)\n",
                n, (unsigned long long)mx);
        return false;                /* init stays 0 -- the next call retries */
    }
    g->spec_snap_copies = snap;
    g->spec_restore_copies = restore;
    g->spec_frontier_copy_n = n;
    g->spec_frontier_copy_max_bytes = mx;
    g->spec_frontier_copy_init = 1;
    return true;
}

static bool spec_frontier_snapshot(pulsar_spec_frontier *f, pulsar_session *s) {
    memset(f, 0, sizeof(*f));
    pulsar_gpu_graph *g = s->graph;
    if (!spec_frontier_copy_tables_init(g)) return false;

    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        f->n_comp[il] = gpu_graph_n_comp(g, gpu_graph_cur_bank(g), il);
    }
    if (ok && g->spec_frontier_copy_n)
        ok = pulsar_gpu_batched_copy_run(g->spec_snap_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}

static bool spec_frontier_restore(pulsar_spec_frontier *f, pulsar_session *s) {
    pulsar_gpu_graph *g = s->graph;
    /* The tables cache the CURRENT bank's state pointers and
     * gpu_graph_bank_repoint drops them, so in the batched lane a round's
     * restore usually finds them gone: the server switches banks between
     * round_begin (snapshot) and round_end, and switches back before calling
     * here.  Rebuild for the bank that is live now -- the round's own -- the
     * same path the snapshot took.  Until L190 this case silently ran the
     * per-tensor loop, i.e. with two or more banks live the production restore
     * WAS the fallback. */
    if (!spec_frontier_copy_tables_init(g)) return false;
    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        gpu_graph_set_n_comp(g, gpu_graph_cur_bank(g), il, f->n_comp[il]);
    }
    if (ok && g->spec_frontier_copy_n)
        ok = pulsar_gpu_batched_copy_run(g->spec_restore_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    return ok;
}

/* Fused-loop helper: make batch row `row`'s captured anchor hidden the current
 * drafter conditioning (target_h -> main_x) and seed one drafter-KV row from it.
 * Mirrors the reference invariant "drafter KV row j = f(hidden at position j)". */
static bool dspark_absorb(pulsar_session *s, uint32_t row, int32_t /*next: DSpark conditions on the row's hidden alone*/) {
    pulsar_gpu_graph *g = s->graph;
    pulsar_engine *e = s->engine;
    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h_batch[i] || !g->dspark_target_h[i]) return false;
        /* Async: project_main_x below reads these on the same (per-thread)
         * stream; nothing on the host reads them before the sync that already
         * bounds the step. */
        if (!pulsar_gpu_tensor_copy_async(g->dspark_target_h[i], 0,
                                 g->dspark_target_h_batch[i],
                                 (uint64_t)row * PULSAR_N_EMBD * sizeof(float),
                                 (uint64_t)PULSAR_N_EMBD * sizeof(float))) return false;
    }
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights)) return false;
    /* A failed seed has rolled the drafter rings back; the caller refuses the
     * round (L167) instead of drafting from an unseeded window. */
    return gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);
}

/* Classic row source: the live graph's spec_logits rows. */
static bool spec_row_read_classic(void *ud, uint32_t row, float *out) {
    return gpu_graph_read_spec_logits_row((pulsar_gpu_graph *)ud, row, out);
}

static bool spec_row_read_dev(void *ud, uint32_t row, float *out) {
    const pulsar_spec_rows *d = (const pulsar_spec_rows *)ud;
    return gpu_graph_read_spec_logits_row((pulsar_gpu_graph *)d->target.g, d->target.row0 + row, out);
}

/* The mapping holding a drafter tensor.  The drafter is NOT one file: its three
 * MTP layers are three separate shards, so a single `dspark_model.map` (which is
 * the TARGET's first shard, because the drafter is aliased to the target by
 * value) resolves every drafter weight against the wrong bytes.  Each tensor
 * names its own mapping, exactly as in the target. */
static inline const void *dspark_map(const pulsar_engine *e, const pulsar_tensor *t) {
    return tensor_map_base(&e->dspark_model, t);
}

static inline uint64_t dspark_map_size(const pulsar_engine *e, const pulsar_tensor *t) {
    return tensor_map_size(&e->dspark_model, t);
}

/* The drafting pass (L150; the single lane's draft since L284): one group of records qs[0..n_sel), greedy
 * ones first (the caller sorted them), whose rows fit the drafter forward together.  The draft forward over
 * every record's rows -- `banks` NULL is the installed bank's classic forward (the single lane, one record);
 * else banks[j] is record j's bank and the forward takes the banked arm over the slabs, no bank switch --
 * the markov refine with one weight stream per position across records (greedy: the whole chain on
 * device; sampled: one step per position with the host draws between, each record from its own rng in
 * position order), the confidence head over every row and the stop rule (pulsar_spec_conf_keep).
 *
 * `defer` (the single greedy lane with no host consumer at draft time, L108 P2): launch the chain and the
 * confidence scoring and DO NOT read them back -- dspark_harvest reads them at the next consumer, so the
 * caller's token emission overlaps the drafter's GPU time; the record then carries no ids, no confidences
 * and keep = n_draft. */
static int dspark_draft_group(pulsar_session *s, spec_redraft_req *const *qs, const uint32_t *banks,
                              uint64_t *const *rngs, int n_sel, bool defer, char *err, size_t errlen) {
    PULSAR_NVTX("redraft group");
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    int n_g = 0;
    for (int j = 0; j < n_sel; j++) if (!qs[j]->sample_drafts) n_g++;
    const int n_s = n_sel - n_g;
    /* rows */
    int32_t draft_ids[PULSAR_DSPARK_DRAFT_ROWS_MAX];
    uint32_t row_bank[PULSAR_DSPARK_DRAFT_ROWS_MAX];
    uint32_t bank_n_raw[PULSAR_MSEQ_MAX][3];
    uint32_t bank_n_draft[PULSAR_MSEQ_MAX];
    int32_t base_row[PULSAR_DSPARK_BANKS_MAX] = {0};
    uint32_t n_rows = 0, max_draft_g = 0, max_draft_s = 0;
    for (int j = 0; j < n_sel; j++) {
        spec_redraft_req *q = qs[j];
        if ((banks && banks[j] >= g->banks.n_banks) || n_rows + q->n_draft > PULSAR_DSPARK_DRAFT_ROWS_MAX) {
            snprintf(err, errlen, "dspark draft: rows exceed the drafter budget");
            return -1;
        }
        base_row[j] = (int32_t)n_rows;
        if (banks) {
            for (uint32_t li = 0; li < 3; li++) bank_n_raw[banks[j]][li] = g->ms_dspark_n_raw[banks[j]][li];
            bank_n_draft[banks[j]] = q->n_draft;
        }
        for (uint32_t k = 0; k < q->n_draft; k++) {
            /* the noise-token forward: row 0 the carry, the rest DSpark's noise id */
            draft_ids[n_rows] = k == 0 ? q->next_base : PULSAR_DSPARK_NOISE_TOKEN_ID;
            row_bank[n_rows] = banks ? banks[j] : 0u;
            n_rows++;
        }
        if (j < n_g) { if (q->n_draft > max_draft_g) max_draft_g = q->n_draft; }
        else         { if (q->n_draft > max_draft_s) max_draft_s = q->n_draft; }
        q->refined[0] = q->next_base;
        for (int k = 1; k < 17; k++) q->refined[k] = 0;
        for (int k = 0; k < 16; k++) { q->qn[k] = 0; q->q_drawn[k] = 0.0f; q->conf[k] = -1.0f; }
        q->have_conf = false;
        q->keep = q->n_draft;
    }
    /* the chain reads base rows (base_row[b] + pos) for pos < the group's max
     * depth: a shallower bank's extra positions read the next bank's rows or
     * the slab tail -- harmless, discarded -- but must stay inside the block */
    if ((uint32_t)base_row[n_sel - 1] + (n_g == n_sel ? max_draft_g : max_draft_s) >
        PULSAR_SPEC_LOGITS_ALLOC_ROWS) {
        snprintf(err, errlen, "dspark draft: chain rows exceed the block");
        return -1;
    }

    const bool fwd = banks
        ? gpu_graph_dspark_draft_forward_banks(g, &e->model, &e->weights, &e->dspark_model, w, g->spec_logits,
                                               draft_ids, n_rows, (uint32_t)n_sel, row_bank, bank_n_raw,
                                               bank_n_draft)
        : gpu_graph_dspark_draft_forward(g, &e->model, &e->weights, &e->dspark_model, w, g->spec_logits,
                                         draft_ids, n_rows);
    if (!fwd) {
        snprintf(err, errlen, "dspark draft: draft forward failed");
        return -1;
    }
    if (pulsar_gpu_tensor_bytes(g->dspark_markov_logits) < (uint64_t)n_sel * vocab_bytes) {
        snprintf(err, errlen, "dspark draft: markov scratch too small");
        return -1;
    }
    /* device meta: base rows [0, MAX), sampled prev tokens [MAX, 2 MAX) */
    int32_t meta[2 * PULSAR_DSPARK_BANKS_MAX];
    memset(meta, 0, sizeof(meta));
    for (int j = 0; j < n_sel; j++) meta[j] = base_row[j];
    /* seed the chain feed: ids[j][0] = next_base */
    int32_t ids_seed[PULSAR_DSPARK_BANKS_MAX * 17];
    memset(ids_seed, 0, sizeof(ids_seed));
    for (int j = 0; j < n_sel; j++) ids_seed[j * 17] = qs[j]->next_base;
    if (!pulsar_gpu_tensor_write(g->dspark_bank_meta, 0, meta, sizeof(meta)) ||
        !pulsar_gpu_tensor_write(g->dspark_refined_ids, 0, ids_seed,
                                 (uint64_t)n_sel * 17 * sizeof(int32_t))) {
        snprintf(err, errlen, "dspark draft: meta upload failed");
        return -1;
    }
    /* spec_logits rows are PULSAR_N_VOCAB wide: the row stride is the TARGET vocab, vocab_size the LENGTH the
     * markov step consumes (equal on the shipped drafter; dspark_weights_validate_layout pins it at load) */
    const uint64_t spec_row_bytes = (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    const int w1_bf16 = w->markov_w1->type == PULSAR_TENSOR_BF16;
    const int w2_fmt = pulsar_markov_w2_fmt(w->markov_w2->type);

    /* greedy records: the whole chain on device (L108 P1), ids read back once */
    if (n_g > 0) {
        if (!pulsar_gpu_dspark_markov_chain_banks_model(
                g->dspark_markov_logits, g->dspark_refined_ids, 17u,
                g->spec_logits, spec_row_bytes, g->dspark_bank_meta,
                dspark_map(e, w->markov_w1), dspark_map_size(e, w->markov_w1), w->markov_w1->abs_offset, w->markov_w2->abs_offset,
                (uint32_t)n_g, max_draft_g, vocab_size, embed_dim, w1_bf16, w2_fmt)) {
            snprintf(err, errlen, "dspark draft: markov chain failed");
            return -1;
        }
    }
    /* sampled records: temperature-matched drafting -- each draft drawn from the refined logits filtered at
     * the REQUEST's params (q), so the walk's min(1, p/q) is not capped at p(mode).  One step per position
     * across records, the host draws between.  L149: in the sparse min-p contract the device prefilter hands
     * back the few candidates above the floor instead of the 517 KB row; a row with more candidates than the
     * compact block holds, or any other shape, reads the full row; a device failure or a refused candidate
     * block refuses the pass (L190 D3: no full-row fallback). */
    if (n_s > 0) {
        pulsar_gpu_tensor *refined_s = pulsar_gpu_tensor_view(
            g->dspark_markov_logits, (uint64_t)n_g * vocab_bytes, (uint64_t)n_s * vocab_bytes);
        pulsar_gpu_tensor *ids_s = pulsar_gpu_tensor_view(
            g->dspark_refined_ids, (uint64_t)n_g * 17 * sizeof(int32_t), (uint64_t)n_s * 17 * sizeof(int32_t));
        pulsar_gpu_tensor *base_s = pulsar_gpu_tensor_view(
            g->dspark_bank_meta, (uint64_t)n_g * sizeof(int32_t), (uint64_t)n_s * sizeof(int32_t));
        pulsar_gpu_tensor *prev_s = pulsar_gpu_tensor_view(
            g->dspark_bank_meta, (uint64_t)PULSAR_DSPARK_BANKS_MAX * sizeof(int32_t),
            (uint64_t)n_s * sizeof(int32_t));
        bool ok = refined_s && ids_s && base_s && prev_s;
        /* the most permissive floor across the sampled records: a superset for each */
        float delta = 0.0f;
        bool all_sparse = true;
        for (int j = n_g; j < n_sel && ok; j++) {
            const spec_redraft_req *q = qs[j];
            const bool sparse = q->top_k <= 0 && q->top_p == 1.0f &&
                                q->min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && q->min_p <= 1.0f &&
                                vocab_size <= PULSAR_SAMPLE_SPARSE_VOCAB_MAX;
            if (!sparse) { all_sparse = false; continue; }
            /* floor in logit units, a hair below T*ln(min_p): the host build decides membership, this
             * only bounds the read */
            const float d = (float)((double)q->temperature * (log((double)q->min_p) - 1e-3));
            if (j == n_g || d < delta) delta = d;
        }
        int32_t *sel = ok ? (int32_t *)xmalloc((size_t)n_s * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t)) : NULL;
        int32_t prev[PULSAR_DSPARK_BANKS_MAX];
        for (uint32_t pos = 0; ok && pos < max_draft_s; pos++) {
            for (int j = n_g; j < n_sel; j++) prev[j - n_g] = qs[j]->refined[pos];
            ok = pulsar_gpu_tensor_write(prev_s, 0, prev, (uint64_t)n_s * sizeof(int32_t)) &&
                 pulsar_gpu_dspark_markov_step_banks_model(
                     refined_s, ids_s, 17u, g->spec_logits, spec_row_bytes, base_s, prev_s,
                     dspark_map(e, w->markov_w1), dspark_map_size(e, w->markov_w1), w->markov_w1->abs_offset, w->markov_w2->abs_offset,
                     (uint32_t)n_s, pos, vocab_size, embed_dim, w1_bf16, w2_fmt);
            if (ok && all_sparse &&
                !(pulsar_gpu_minp_prefilter_rows(g->dspark_prefilter_sel, refined_s, 0,
                                                 (uint32_t)n_s, vocab_size, vocab_size,
                                                 delta, PULSAR_DSPARK_PREFILTER_CAP) &&
                  pulsar_gpu_tensor_read(g->dspark_prefilter_sel, 0, sel,
                                         (uint64_t)n_s * PULSAR_DSPARK_PREFILTER_ROW_I32 * sizeof(int32_t)))) {
                fprintf(stderr, "pulsar: dspark min-p prefilter failed at draft position %u "
                                "-- refusing the pass (no full-row fallback; L190)\n", pos);
                ok = false;
            }
            for (int j = n_g; j < n_sel && ok; j++) {
                spec_redraft_req *q = qs[j];
                if (pos >= q->n_draft) continue;
                pulsar_sample_dist qd;
                bool built = false;
                if (all_sparse) {
                    const int32_t *h = sel + (size_t)(j - n_g) * PULSAR_DSPARK_PREFILTER_ROW_I32;
                    const uint32_t n_c = (uint32_t)h[0];
                    /* n_c > cap: more candidates above the floor than the
                     * compact block holds -- the one genuine ineligibility;
                     * the full row is read below.  Anything else the build
                     * refuses is a broken device result. */
                    if (n_c <= PULSAR_DSPARK_PREFILTER_CAP) {
                        float max_logit;
                        memcpy(&max_logit, &h[2], sizeof(max_logit));
                        if (n_c == 0 ||
                            !pulsar_sample_dist_build_prefiltered(
                                h + 3, (const float *)(h + 3 + PULSAR_DSPARK_PREFILTER_CAP), n_c,
                                max_logit, q->temperature, q->min_p, &s->sample_scratch, &qd)) {
                            fprintf(stderr, "pulsar: dspark min-p prefilter handed %u "
                                            "candidates at draft position %u and the build refused "
                                            "them -- refusing the pass (L190)\n", n_c, pos);
                            ok = false;
                            break;
                        }
                        built = true;
                    }
                }
                /* the full row: the build needs it when nothing was prefiltered, the walk when q is too
                 * wide for the compact store */
                const bool need_row = !built || !pulsar_spec_q_compact(qd.n);
                if (need_row) {
                    if (!pulsar_spec_redraft_qrows_reserve(q, spec_vocab(s))) {
                        if (built) pulsar_sample_dist_free(&qd);
                        ok = false; break;
                    }
                    float *row = q->qrows + (size_t)pos * spec_vocab(s);
                    if (!pulsar_gpu_tensor_read(refined_s, (uint64_t)(j - n_g) * vocab_bytes, row, vocab_bytes)) {
                        if (built) pulsar_sample_dist_free(&qd);
                        ok = false; break;
                    }
                    if (!built &&
                        !pulsar_sample_dist_build(row, spec_vocab(s), q->temperature, q->top_k,
                                                  q->top_p, q->min_p, &s->sample_scratch, &qd)) {
                        ok = false; break;
                    }
                }
                if (built) (void)pulsar_spec_q_record(q, pos, &qd);   /* a row-built q stays a row */
                const int drawn = pulsar_sample_dist_draw(&qd, rngs[j]);
                q->refined[pos + 1] = (int32_t)drawn;   /* the chain continues SAMPLED */
                q->q_drawn[pos] = pulsar_sample_dist_prob(&qd, drawn);
                pulsar_sample_dist_free(&qd);
            }
        }
        free(sel);
        pulsar_gpu_tensor_free(refined_s);
        pulsar_gpu_tensor_free(ids_s);
        pulsar_gpu_tensor_free(base_s);
        pulsar_gpu_tensor_free(prev_s);
        if (!ok) {
            snprintf(err, errlen, "dspark draft: sampled markov loop failed");
            return -1;
        }
    }
    /* greedy ids come back in one read */
    if (n_g > 0 && !defer) {
        int32_t ids_all[PULSAR_DSPARK_BANKS_MAX * 17];
        if (!pulsar_gpu_tensor_read(g->dspark_refined_ids, 0, ids_all, (uint64_t)n_sel * 17 * sizeof(int32_t))) {
            snprintf(err, errlen, "dspark draft: ids readback failed");
            return -1;
        }
        for (int j = 0; j < n_g; j++)
            for (uint32_t k = 1; k <= qs[j]->n_draft; k++) qs[j]->refined[k] = ids_all[j * 17 + k];
    }
    /* the offline-validation / confidence-training dump (single lane): after the markov refine, while
     * batch_ffn_cur still holds the post-hc_head rows the confidence kernel consumes */
    if (!banks && !defer) dspark_dump_step(g, (int)s->checkpoint.len, (int)qs[0]->next_base, qs[0]->refined,
                                           (int)qs[0]->n_draft);
    /* the confidence head over every row: tok row (j, k) = refined_j[k] -- the draft BEFORE position k, so
     * conf[k] is scored without the draft it judges (the stop is a stopping time) -- then the stop rule */
    const float tau = pulsar_spec_tau(e);
    if (tau > 0.0f) {
        int32_t toks[PULSAR_DSPARK_DRAFT_ROWS_MAX];
        float confs[PULSAR_DSPARK_DRAFT_ROWS_MAX];
        for (int j = 0; j < n_sel; j++)
            for (uint32_t k = 0; k < qs[j]->n_draft; k++) toks[base_row[j] + k] = qs[j]->refined[k];
        const bool scored =
            pulsar_gpu_tensor_bytes(g->dspark_conf_tokens) >= (uint64_t)n_rows * sizeof(int32_t) &&
            pulsar_gpu_tensor_bytes(g->dspark_conf_scores) >= (uint64_t)n_rows * sizeof(float) &&
            (defer
                 /* device-to-device: the ids are already in the chain array; a host write here would
                  * force the read this path exists to avoid */
                 ? pulsar_gpu_tensor_copy(g->dspark_conf_tokens, 0, g->dspark_refined_ids, 0,
                                          (uint64_t)n_rows * sizeof(int32_t)) != 0
                 : pulsar_gpu_tensor_write(g->dspark_conf_tokens, 0, toks, (uint64_t)n_rows * sizeof(int32_t)) != 0) &&
            pulsar_gpu_dspark_confidence_score_model(g->dspark_conf_scores, g->batch_ffn_cur,
                                                     g->dspark_conf_tokens, dspark_map(e, w->markov_w1), dspark_map_size(e, w->markov_w1),
                                                     w->markov_w1->abs_offset,
                                                     w->confidence_proj->abs_offset,
                                                     n_rows, PULSAR_N_EMBD, embed_dim, vocab_size,
                                                     w1_bf16,
                                                     w->confidence_proj->type == PULSAR_TENSOR_BF16) &&
            (defer || pulsar_gpu_tensor_read(g->dspark_conf_scores, 0, confs, (uint64_t)n_rows * sizeof(float)));
        if (!scored) {
            /* L190 D4: the stop is scheduled; a scoring failure drafts nothing, loudly */
            snprintf(err, errlen, "dspark draft: confidence scoring failed (the stop is not skipped silently)");
            return -1;
        }
        for (int j = 0; j < n_sel; j++) {
            spec_redraft_req *q = qs[j];
            q->have_conf = true;
            if (defer) continue;   /* dspark_harvest reads and trims */
            for (uint32_t k = 0; k < q->n_draft; k++) q->conf[k] = confs[base_row[j] + k];
            q->keep = pulsar_spec_conf_keep(q->conf, q->n_draft, tau);
        }
    }
    return 0;
}

/* The single lane's draft: one record through the drafting pass, stamped into the installed bank's
 * pendings -- or, deferred (L108 P2), the in-flight chain marked for dspark_harvest. */
static uint32_t dspark_draft(pulsar_session *s, int next_base, bool, float temperature, int top_k, float top_p,
                             float min_p, uint64_t *rng) {
    spec_redraft_req rec;
    memset(&rec, 0, sizeof rec);
    pulsar_spec_draft_req_init(s, &rec, next_base, temperature, top_k, top_p, min_p);
    /* greedy with no host consumer at draft time (no dump) defers the readback */
    const bool defer = !rec.sample_drafts && !gpu_graph_spec_dump_active();
    spec_redraft_req *qs[1] = {&rec};
    uint64_t *rngs[1] = {rng};
    char err[200];
    uint32_t keep = 0;
    if (dspark_draft_group(s, qs, NULL, rngs, 1, defer, err, sizeof err) != 0) {
        fprintf(stderr, "pulsar: %s -- no drafts this round\n", err);
        pulsar_spec_drop_pendings(&s->spec);
    } else {
        if (defer) rec.keep = 0;   /* the harvest sets the real count */
        pulsar_spec_redraft_stamp(s, &rec);
        if (defer) {
            s->spec.dspark_chain_unharvested = true;
            s->spec.dspark_chain_conf = rec.have_conf;
            s->spec.dspark_chain_n = rec.n_draft;
        }
        keep = rec.keep;
    }
    free(rec.qrows);
    return keep;
}

/* L108 P2: lazy completion of a deferred greedy chain: the ids and the confidences read back, the stop rule
 * applied.  A failed readback drops the chain (0 pendings, the next step is a plain forward) and says so --
 * never an untrimmed keep with -1 conf fed to the controller (L190 D4). */
static void dspark_harvest(pulsar_session *s) {
    if (!s->spec.dspark_chain_unharvested) return;
    s->spec.dspark_chain_unharvested = false;
    pulsar_gpu_graph *g = s->graph;
    const uint32_t n_draft = s->spec.dspark_chain_n;
    if (n_draft == 0 || n_draft > 16u) return;
    int32_t ids[17];
    float conf[16];
    const bool have_conf = s->spec.dspark_chain_conf;
    if (!g->dspark_refined_ids ||
        !pulsar_gpu_tensor_read(g->dspark_refined_ids, sizeof(int32_t), ids + 1,
                                (uint64_t)n_draft * sizeof(int32_t)) ||
        (have_conf && !pulsar_gpu_tensor_read(g->dspark_conf_scores, 0, conf, (uint64_t)n_draft * sizeof(float)))) {
        fprintf(stderr, "pulsar: dspark chain harvest: the draft %s readback failed -- the chain is dropped "
                        "(0 pendings)\n", have_conf ? "ids or confidence" : "ids");
        s->spec.n_pend = 0;
        return;
    }
    const uint32_t keep = have_conf ? pulsar_spec_conf_keep(conf, n_draft, pulsar_spec_tau(s->engine)) : n_draft;
    s->spec.n_pend = keep;
    for (uint32_t i = 0; i < keep; i++) {
        s->spec.pend[i] = ids[i + 1];
        /* the adaptive controller reads the verified chain's conf at round_end */
        s->spec.pend_conf[i] = have_conf ? conf[i] : -1.0f;
    }
}

/* L149 phase 2 / L219: accumulate, for the step this round will ride, whether every round is in the
 * sparse min-p contract and the most permissive floor among them (the compact readback), and whether
 * every round is greedy (the per-row argmax readback); ds4_arm_verify arms from these.  A round that
 * begins here but sits the step out only makes the result more conservative (ok=false) or the superset
 * wider (lower floor): both safe. */
static void ds4_round_note(pulsar_session *s, float temperature, int top_k, float top_p, float min_p) {
    pulsar_gpu_graph *g = s->graph;
    const bool in_contract = temperature > 0.0f && top_k <= 0 && top_p == 1.0f &&
                             min_p >= PULSAR_SAMPLE_SPARSE_MINP_MIN && min_p <= 1.0f &&
                             spec_vocab(s) <= PULSAR_SAMPLE_SPARSE_VOCAB_MAX;
    const float d = in_contract
        ? (float)((double)temperature * (log((double)min_p) - 1e-3)) : 0.0f;
    if (g->spec_compact_acc_n == 0) {
        g->spec_compact_acc_ok = in_contract;
        g->spec_compact_acc_delta = d;
    } else {
        g->spec_compact_acc_ok = g->spec_compact_acc_ok && in_contract;
        if (d < g->spec_compact_acc_delta) g->spec_compact_acc_delta = d;
    }
    g->spec_compact_acc_n++;
    /* L219: the greedy arm.  temperature <= 0 is the only condition: the
     * walk then consults row argmaxes and nothing else, which a device
     * argmax readback can supply without the full rows. */
    const bool greedy = temperature <= 0.0f;
    if (g->spec_argmax_acc_n == 0) g->spec_argmax_acc_ok = greedy;
    else                           g->spec_argmax_acc_ok = g->spec_argmax_acc_ok && greedy;
    g->spec_argmax_acc_n++;
}

static void ds4_arm_verify(pulsar_session *s, uint32_t n_rows) {
    pulsar_gpu_graph *g = s->graph;
    g->dspark_capture_batch_n = n_rows;
    g->spec_comp_save_n = n_rows;
    /* L149 phase 2: arm (n_rows > 0) the compact verify read from the rounds
     * begun since the last step; disarm and reset the accumulator after it. */
    if (n_rows > 0) {
        g->spec_compact_armed = g->spec_compact_acc_n > 0 && g->spec_compact_acc_ok;
        g->spec_compact_delta = g->spec_compact_acc_delta;
        g->spec_argmax_armed = !g->spec_compact_armed &&
                               g->spec_argmax_acc_n > 0 && g->spec_argmax_acc_ok;
    } else {
        g->spec_compact_armed = false;
        g->spec_compact_acc_n = 0;
        g->spec_compact_acc_ok = false;
        /* The argmax rows stay live for the per-bank round ends that follow the
         * step; only the arming and the accumulator retire here. */
        g->spec_argmax_armed = false;
        g->spec_argmax_acc_n = 0;
        g->spec_argmax_acc_ok = false;
    }
}

/* The single lane's verify: ONE batched target forward over the round's rows with the drafter's anchor
 * capture and the Stage-B state saves armed; the device computes the row argmaxes; the compact readback
 * runs when the round is in the sparse contract.  The rows stay on the device (spec_row_read_classic). */
static bool ds4_verify_single(pulsar_session *s, pulsar_spec_round *r, pulsar_spec_rows *out,
                              char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = s->graph;
    /* ONE batched forward: base decode + draft verify + anchor capture. */
    g->dspark_capture_batch_n = r->n_batch;
    g->spec_comp_save_n = r->n_batch;   /* Stage-B: save per-position comp projections */
    /* L149 phase 2: this lane's step carries exactly the round begun above */
    g->spec_compact_armed = g->spec_compact_acc_n > 0 && g->spec_compact_acc_ok;
    g->spec_compact_delta = g->spec_compact_acc_delta;
    g->spec_compact_acc_n = 0;
    g->spec_compact_acc_ok = false;
    /* The fused lane already has device row argmaxes from
     * gpu_graph_verify_suffix_tops, so it never arms the greedy readback --
     * but it must retire the accumulator this round fed. */
    g->spec_argmax_armed = false;
    g->spec_argmax_acc_n = 0;
    g->spec_argmax_acc_ok = false;
    bool ok = gpu_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                           &s->checkpoint,
                                           (uint32_t)r->saved_len, r->n_batch,
                                           r->K ? r->row_tops : NULL, NULL);
    g->dspark_capture_batch_n = 0;
    g->spec_comp_save_n = 0;
    if (ok && g->spec_compact_armed) {
        if (!gpu_graph_spec_compact_read(g, 0u, r->n_batch)) {
            fprintf(stderr, "pulsar: spec compact readback failed for %u rows -- refusing (L174)\n", r->n_batch);
            ok = false;
        }
    } else {
        g->spec_compact_rows = 0;
    }
    g->spec_compact_armed = false;
    if (!ok) {
        snprintf(err, errlen, "DSpark fused verify failed");
        return false;
    }
    out->read = spec_row_read_classic;
    out->ud = g;
    out->compact = g->spec_compact_rows >= r->n_batch ? g->spec_compact_host : NULL;
    return true;

}

/* The batched lane: this round's rows of the last shared forward, from whichever readback the step
 * armed -- the compact candidate block, the per-row argmaxes, or the caller's full block. */
static bool ds4_verify_rows(pulsar_session *s, pulsar_spec_round *r, const float *rows, uint32_t row0,
                            pulsar_spec_rows *out, char *err, size_t errlen) {
    pulsar_gpu_graph *g = s->graph;
    /* L149 phase 2: the step read the compact block instead of `rows`. Row
     * argmaxes come from its headers (the host tie rule, computed on device);
     * the walk builds from candidates; any row that needs its full logits
     * (no finite max, overflow, the s->logits refresh) is read from the
     * device, where this step's rows still sit. */
    if (g->spec_compact_rows >= row0 + r->n_batch && g->spec_compact_host) {
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)spec_vocab(s) * sizeof(float));
        for (uint32_t i = 0; i < r->K && i < 16u; i++) {
            const int32_t *h = g->spec_compact_host +
                               (size_t)(row0 + i) * PULSAR_DSPARK_PREFILTER_ROW_I32;
            if (h[1] >= 0) {
                r->row_tops[i] = h[1];
            } else {
                if (!gpu_graph_read_spec_logits_row(g, row0 + i, s->spec_row_scratch)) {
                    snprintf(err, errlen, "spec compact: row %u readback failed", row0 + i);
                    return false;
                }
                r->row_tops[i] = (int)sample_argmax(s->spec_row_scratch, spec_vocab(s));
            }
        }
        out->read = spec_row_read_dev;
        out->ud = out;
        out->target.g = g;
        out->target.row0 = row0;
        out->compact = g->spec_compact_host;
        return true;
    }
    /* L219: the greedy readback replaced the full rows with per-row argmaxes.
     * The walk needs only these; the one full row the s->logits refresh wants
     * is read from the device on demand. */
    if (g->spec_argmax_rows >= row0 + r->n_batch && g->spec_argmax_host) {
        for (uint32_t i = 0; i < r->K && i < 16u; i++)
            r->row_tops[i] = g->spec_argmax_host[row0 + i];
        out->read = spec_row_read_dev;
        out->ud = out;
        out->target.g = g;
        out->target.row0 = row0;
        out->compact = NULL;
        return true;
    }
    /* Greedy walk consumes per-row argmaxes; the classic forward computes
     * them on-device, the shared block computes them here. Draft i is judged
     * against round-local row i (the row that PREDICTS it). */
    for (uint32_t i = 0; i < r->K && i < 16u; i++) {
        r->row_tops[i] = (int)sample_argmax(
                rows + ((size_t)row0 + i) * spec_vocab(s), spec_vocab(s));
    }
    out->read = pulsar_spec_row_read_block;
    out->ud = out;
    out->block.rows = rows;
    out->block.row0 = row0;
    out->block.vocab = spec_vocab(s);
    out->compact = NULL;
    return true;
}

/* Prompt-window seeding (one-time per prompt): fresh drafter state + a
 * captured prompt window -> replay the last <=128 prompt positions into
 * the drafter's context-KV ring, exactly as the reference prefills it.
 * Without this the window starts empty (or, before the invalidate fix,
 * stale from the previous request), and the drafter is near-useless
 * without a valid window (masked-window eval: 4.7% vs 86% top-1). */
static int dspark_prime(pulsar_session *s, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    if (!(g->dspark_n_raw[0] == 0 && g->dspark_prompt_n > 0 && g->dspark_prompt_h[0])) return 0;
    const uint32_t win = PULSAR_DSPARK_DRAFT_WINDOW;
    uint32_t avail = g->dspark_prompt_n - g->dspark_prompt_lo;
    const uint32_t take = avail < win ? avail : win;
    const uint32_t first = g->dspark_prompt_n - take;
    bool seed_ok = true;
    uint32_t j = 0;
    for (; seed_ok && j < take; j++) {
        const uint32_t slot = (first + j) % win;
        for (int i = 0; seed_ok && i < 3; i++) {
            seed_ok = pulsar_gpu_tensor_copy(g->dspark_target_h[i], 0,
                                          g->dspark_prompt_h[i],
                                          (uint64_t)slot * PULSAR_N_EMBD * sizeof(float),
                                          (uint64_t)PULSAR_N_EMBD * sizeof(float)) != 0;
        }
        if (seed_ok)
            seed_ok = gpu_graph_dspark_project_main_x(g, &e->dspark_model, w) &&
                      gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, w, 1);
    }
    if (!seed_ok) {
        /* Refuse the round rather than draft from a half-seeded window
         * (L167; a failed projection used to be skipped over and a failed
         * seed warned once and drafted unseeded).  The target state is
         * untouched and the window stays pending for the next attempt. */
        snprintf(err, errlen, "DSpark prompt-window seed failed after %u of %u rows", j - 1u, take);
        return -1;
    }
    if (dspark_stats)
        fprintf(stderr, "pulsar: dspark prompt-window seeded %u rows (prompt_n=%u)\n",
                take, g->dspark_prompt_n);
    g->dspark_prompt_n = 0;   /* consumed; commits take over from here */
    return 0;
}


static int dspark_draft_batch(pulsar_session *s, pulsar_spec_round **rounds,
                                      const uint32_t *banks, uint64_t **rngs, int n,
                                      char *err, size_t errlen) {
    pulsar_gpu_graph *g = s->graph;
    if (!rounds || !banks || !rngs || n <= 0) return 0;
    if (!g->dspark_markov_logits || !g->dspark_refined_ids ||
        !g->dspark_bank_meta || !g->dspark_conf_scores || !g->dspark_conf_tokens ||
        !g->dspark_prefilter_sel || g->banks.n_banks == 0) {
        snprintf(err, errlen, "redraft batch: drafter scratch missing");
        return -1;
    }
    /* Every live round drafts (L260): the groups below split them into
     * drafter passes of at most PULSAR_DSPARK_BANKS_MAX banks (one markov
     * launch per position carries them all, L262), each pass reusing the batch buffers after the
     * previous one's results are read back.  Selection used to stop at
     * PULSAR_DSPARK_BANKS_MAX, so past 8 live banks the rest took base-only
     * steps -- at c10, two streams drafted nothing on every round. */
    if (n > (int)PULSAR_MSEQ_MAX) {
        snprintf(err, errlen, "redraft batch: %d rounds, more than the %u-bank pool", n,
                 (unsigned)PULSAR_MSEQ_MAX);
        return -1;
    }
    /* greedy rounds first, then sampled, so each group is contiguous in rows
     * and in the bank arrays */
    spec_redraft_req *qs[PULSAR_MSEQ_MAX];
    uint32_t bk[PULSAR_MSEQ_MAX];
    uint64_t *rg[PULSAR_MSEQ_MAX];
    int n_sel = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n; i++) {
            spec_redraft_req *q = &rounds[i]->redraft;
            if (!q->valid || q->n_draft == 0 || (pass == 0) != !q->sample_drafts) continue;
            qs[n_sel] = q;
            bk[n_sel] = banks[i];
            rg[n_sel] = rngs[i];
            n_sel++;
        }
    if (n_sel == 0) return 0;

    /* groups: consecutive banks whose rows fit the drafter forward
     * (PULSAR_DSPARK_DRAFT_ROWS_MAX; the forward declares them decode rows
     * under its own cap), at most PULSAR_DSPARK_BANKS_MAX banks each -- L260:
     * a full 16-bank pool at depth <= 4 is ONE group, so the drafter's head,
     * dense and markov weights are read once per step (16 rows / 8 banks made
     * it three groups at c16, each re-reading them: ~1/6 of the step).  A bank
     * whose depth alone exceeds the cap is refused. */
    for (int gs = 0; gs < n_sel;) {
        int ge = gs;
        uint32_t rows = 0;
        while (ge < n_sel && ge - gs < (int)PULSAR_DSPARK_BANKS_MAX &&
               rows + qs[ge]->n_draft <= PULSAR_DSPARK_DRAFT_ROWS_MAX) {
            rows += qs[ge]->n_draft;
            ge++;
        }
        if (ge == gs) {
            snprintf(err, errlen, "redraft batch: a bank's depth exceeds the row budget");
            return -1;
        }
        const int rc = dspark_draft_group(s, qs + gs, bk + gs, rg + gs, ge - gs, false, err, errlen);
        if (rc != 0) return rc;
        gs = ge;
    }
    for (int j = 0; j < n_sel; j++) qs[j]->done = true;
    return 0;
}

/* ---- the target hooks ------------------------------------------------------------------------- */

static bool ds4_spec_snapshot(pulsar_session *s, pulsar_spec_round *r, char *err, size_t errlen) {
    pulsar_spec_frontier *f = (pulsar_spec_frontier *)r->snap;
    if (!f) f = (pulsar_spec_frontier *)xmalloc(sizeof *f);
    r->snap = f;
    if (spec_frontier_snapshot(f, s)) return true;
    snprintf(err, errlen, "DSpark fused frontier snapshot failed");
    return false;
}

static bool ds4_spec_restore(pulsar_session *s, pulsar_spec_round *r) {
    return r->snap && spec_frontier_restore((pulsar_spec_frontier *)r->snap, s);
}

static void ds4_spec_release(pulsar_session *, pulsar_spec_round *r) {
    if (r->snap) spec_frontier_free((pulsar_spec_frontier *)r->snap);
}

/* A partial accept: restore the pre-verify frontier, then Stage B -- roll only the recurrent
 * compressor / indexer pool state forward through the committed prefix from the projections saved
 * during the verify batch (bit-identical: same update kernels, same rows, same order).  Raw KV and
 * comp-cache rows are position-addressed and already correct; counters are set by formula; the
 * drafter's rows absorb from the batch capture afterwards (the core). */
static bool ds4_commit(pulsar_session *s, pulsar_spec_round *r, uint32_t commit, uint32_t row0) {
    pulsar_engine *e = s->engine;
    /* a full accept: the verify advanced the target state by exactly the committed tokens */
    if (commit == r->K) return true;
    if (!ds4_spec_restore(s, r)) return false;
    return gpu_graph_dspark_compressor_rollforward(s->graph, &e->model, &e->weights,
                                                   (uint32_t)r->saved_len, 1u + commit, row0);
}

/* The L155 trim: rewind() clamps the compressor frontier and drops the carry and the drafter window
 * (the carry was conditioned on positions that no longer exist). */
static void ds4_cut(pulsar_session *s, int pos) { s->rewind(pos); }

/* gpu_graph_decode_multiseq_batch records the step's form on every step: exactly one of the two row counts
 * nonzero, or both zero for full rows -- the per-row argmaxes (L219) or the compact min-p candidates (L149). */
static bool ds4_readback(pulsar_session *s, const void **rows, uint32_t *n_rows, uint32_t *width) {
    const pulsar_gpu_graph *g = s->graph;
    if (g->spec_argmax_rows > 0) {
        *rows = g->spec_argmax_host; *n_rows = g->spec_argmax_rows; *width = 1u;
        return true;
    }
    if (g->spec_compact_rows > 0) {
        *rows = g->spec_compact_host; *n_rows = g->spec_compact_rows; *width = (uint32_t)PULSAR_DSPARK_PREFILTER_ROW_I32;
        return true;
    }
    return false;
}

const pulsar_spec_target_ops k_ds4_spec_target = {
    /* .snapshot      = */ ds4_spec_snapshot,
    /* .restore       = */ ds4_spec_restore,
    /* .release       = */ ds4_spec_release,
    /* .round_note    = */ ds4_round_note,
    /* .arm_verify    = */ ds4_arm_verify,
    /* .verify_single = */ ds4_verify_single,
    /* .verify_rows   = */ ds4_verify_rows,
    /* .commit        = */ ds4_commit,
    /* .cut           = */ ds4_cut,
    /* .readback      = */ ds4_readback,
};

/* ---- the DSpark drafter -------------------------------------------------------------------------- */

static bool dspark_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *,
                                 uint32_t n) {
    pulsar_engine *e = s->engine;
    return gpu_graph_dspark_seed_rows_banked(s->graph, &e->dspark_model, &e->dspark_weights, rows, banks, n);
}

/* The L107 adaptive depth, DSpark's numbers (the rule and its v2..v5 history: spec_depth.h).  Bounds 2..5 are
 * the 2026-08-25 sweep's measured range (prose peaked at 2, structured at 5); 5 is the drafter's TRAINED BLOCK
 * (0731 metadata: stages=3 block=5) -- position 6 is out of distribution and depth 6 lost on BOTH regimes, so
 * even transient excursions there are purchased losses; re-tune only with a drafter retrained at a wider block
 * (L092).  UP needs a tail confidence >= 0.70 (calibrated >= 0.82 accept); the v3 veto
 * only at depth 5 with a >= 0.90 tail; the v4/v5 8-round cooldown after a down within 2 rounds of an up. */
static const pulsar_spec_depth_policy k_dspark_depth = {2, 5, 0.70f, 5, 0.90f, 8u, 2u};

/* DSpark's numbers.
 * Depth 3 to start: the v5mx optimum (2026-07-17 k-sweep at the tau 0.25 stop, quench disarmed).  k = 3 beat
 * k = 5 by +15% structured to +32% prose served decode; the drafter forward is autoregressive, so its cost
 * scales with the chain ON TOP of the verify rows -- ms / accepted token stays flat ~41-46 ms across k, depth
 * never amortizes, shallower wins (the prior 5 was a compact-model figure).  The controller moves it from
 * there.  Depth cap 16: the drafter's row arrays.
 * The stop at 0.55: under the shared rule (the first draft under tau is the chain's LAST, still verified;
 * L284) 0.55 matched what the old rule (that draft dropped) did at 0.25 -- single-stream decode, 3 reps
 * median, code greedy / chat greedy / chat T0.7: 31.72 / 30.11 / 31.74 tok/s vs 31.16 / 30.18 / 32.34 (0.40:
 * 31.34 / 30.00 / 31.44; 0.25: 31.08 / 29.97 / 30.61).  The 2026-07-17 sweep under the old rule: tau barely
 * moves GREEDY throughput at depth 3, but clearly wins under T = 1.0 SAMPLING (+25% structured, +10% prose vs
 * verify-all), where the low-confidence tail is real.  The stop is a SCHEDULE knob: it decides which drafts
 * are verified, never a verified row's numerics (verify rows are decode rows; cuda-mixed-neutrality-gate
 * GATE 5/5R) -- a named exception in docs/ENGINEERING-RULES.md; tools/confhead sets --spec-tau off to collect
 * unbiased labels. */
const pulsar_drafter_ops k_dspark_drafter = {
    /* .name          = */ "DSpark",
    /* .depth         = */ 3u,
    /* .depth_max     = */ 16u,
    /* .tau           = */ 0.55f,
    /* .adapt         = */ &k_dspark_depth,
    /* .prime         = */ dspark_prime,
    /* .absorb        = */ dspark_absorb,
    /* .absorb_banked = */ dspark_absorb_banked,
    /* .draft         = */ dspark_draft,
    /* .harvest       = */ dspark_harvest,
    /* .draft_batch   = */ dspark_draft_batch,
};

#include "pulsar_engine_internal.h"

/* Batched multi-session decode over the session's bank pool — see the pulsar.h
 * declaration for the caller contract and gpu_graph_decode_multiseq_batch
 * (imatrix.cpp) for the step mechanics.
 *
 * The session's own single-bank bookkeeping is not merely un-advanced, it is
 * INVALIDATED on success: a multiseq step leaves the scalar frontier
 * counters holding a cross-bank superset (never any single bank's truth) and
 * advances a bank's KV past s->checkpoint.
 *
 * TWO separate guards are needed, because they cover different callers:
 *
 *   checkpoint_valid = false  stops pulsar_session_sync from taking its
 *       prefix-resume path (it gates on checkpoint_valid alone), forcing the
 *       rebuild path — which resets the graph's prefill state and so
 *       re-establishes per-bank truth.
 *
 *   mseq_dirty = true  stops pulsar_session_eval.  checkpoint_valid does NOT
 *       cover it: eval never reads checkpoint_valid, it just decodes at
 *       s->checkpoint.len on whatever bank is installed.
 *
 *       ⚠ The MECHANISM here has been rewritten twice; the HAZARD has not
 *       changed.  This used to describe eval calling gpu_graph_eval_token_raw_swa
 *       and reading a cross-bank SUPERSET as its emit row — writing the
 *       compressor row there and attending over a previous tenant's bytes.
 *       Both halves of that are now gone: L130 made eval a 1-row batch on its
 *       own bank, and stage 1b deleted the scalar superset (gpu_graph_n_comp
 *       reads ms_n_comp[cur_bank] directly).  What remains is that a multiseq
 *       step which touched OTHER banks leaves this session's per-bank carry
 *       needing re-establishment, so eval fails loud while dirty rather than
 *       decoding against state it cannot vouch for.
 *
 * The caller owns per-bank histories and must re-establish per-bank state
 * explicitly to resume classic work on a bank: a fresh pulsar_session_sync
 * (rebuild path) clears both flags. */
#define PULSAR_MULTISEQ_ERR(...) do { \
        if (err && errlen) snprintf(err, errlen, __VA_ARGS__); \
    } while (0)

/* The step every batched entry below runs once its own arguments are checked: the requests split into the
 * driver's descriptor (positions / seq ids / tokens, heap scratch sized to n_rows -- no [PULSAR_MSEQ_MAX]
 * stack ceiling; the driver still bounds rows and banks itself), gpu_graph_decode_multiseq_batch, and the
 * one reading of its rc.  `what` names the step in the error.  0 = ran, 1 = rejected (recoverable),
 * -1 = failed mid-sweep (session state fatal). */
static int ds4_batch_step(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits,
                          uint32_t *out_n_rows, uint32_t max_head_runs, const pulsar_fused_shape *shape,
                          const char *what, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    int32_t *pos = (int32_t *)xmalloc((size_t)n_rows * sizeof(*pos));
    int32_t *bank = (int32_t *)xmalloc((size_t)n_rows * sizeof(*bank));
    int *tokens = (int *)xmalloc((size_t)n_rows * sizeof(*tokens));
    for (uint32_t k = 0; k < n_rows; k++) {
        pos[k]    = reqs[k].pos;
        bank[k]   = (int32_t)reqs[k].bank;
        tokens[k] = reqs[k].token;
    }
    const int rc = gpu_graph_decode_multiseq_batch(s->graph, &e->model, &e->weights, tokens, pos, bank,
                                                   n_rows, logits, out_n_rows, max_head_runs,
                                                   /*capture_cur=*/false, shape);
    /* The batch call consumed the descriptor synchronously (tokens copied to a stack row, pos/seq uploaded
     * to device in step_begin); the host scratch is dead now regardless of rc. */
    free(pos); free(bank); free(tokens);
    if (rc == 0) {
        /* Recoverable: the driver rejected before arming the step, so nothing
         * was mutated — the upload writes ahead of it touch scratch only, and
         * every gpu_graph_multiseq_step_begin rejection point precedes its
         * first scalar write (the superset refresh) and its cur-bank capture.
         * The classic view is therefore still true: leave both flags alone so
         * the caller can fix the batch and retry, or fall back to classic. */
        PULSAR_MULTISEQ_ERR("%s rejected (recoverable; reason on stderr)", what);
        return 1;
    }
    /* The step was armed: the scalar counters hold a superset on success, and
     * on a fatal mid-sweep failure their state is unknown.  Both leave the
     * classic single-bank view stale (see above); only the diagnosis differs. */
    s->checkpoint_valid = false;
    s->mseq_dirty = true;
    s->spec.spec_carry_valid = false;
    if (rc == 1) return 0;
    PULSAR_MULTISEQ_ERR("%s failed mid-sweep (session state fatal)", what);
    return -1;
}

int pulsar_session::decode_multiseq(const pulsar_multiseq_req *reqs,
                                uint32_t n, float *logits, int logits_cap,
                                char *err, size_t errlen) {
    auto *s = this;
    if (!s || !reqs || !logits || n == 0 || n > PULSAR_MSEQ_MAX) {
        PULSAR_MULTISEQ_ERR("multiseq decode: bad args (n=%u)", n);
        return 1;
    }
    /* The engine writes n rows of PULSAR_N_VOCAB floats; without a capacity the
     * readback would overflow a caller buffer sized from a different vocab
     * notion (pulsar_engine_vocab_size is the tokenizer table length, which the
     * loader never checks against the shape profile's n_vocab). */
    if (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n * PULSAR_N_VOCAB) {
        PULSAR_MULTISEQ_ERR("multiseq decode: logits capacity %d < %u rows x %u",
                         logits_cap, n, (unsigned)PULSAR_N_VOCAB);
        return 1;
    }
    /* decode_multiseq is decode-only (1 row per bank), so n_runs == n; no
     * out-param needed (the caller reads n logit rows). max_head_runs = 0 (all). */
    return ds4_batch_step(s, reqs, n, logits, NULL, 0u, NULL, "multiseq decode step", err, errlen);
}

/* plan-34 phase-2 increment 1 — mixed prefill+decode descriptor entry.
 *
 * BYTE-IDENTICAL to pulsar_session_decode_multiseq for a decode-only (1-row-per-bank)
 * batch: the same step (ds4_batch_step) with up to prefill_cap rows.  The kernel path
 * (gpu_graph_decode_multiseq_batch + step_begin) still bounds the current ROW count to
 * PULSAR_MSEQ_MAX and keeps PULSAR_MSEQ_MAX as the BANK-count bound. */
int pulsar_session::decode_mixed(const pulsar_multiseq_req *reqs,
                             uint32_t n_rows, float *logits, int logits_cap,
                             uint32_t *out_n_rows, uint32_t max_head_runs,
                             char *err, size_t errlen) {
    auto *s = this;
    if (out_n_rows) *out_n_rows = 0;
    if (!s || !reqs || !logits || n_rows == 0 ||
        n_rows > s->graph->prefill_cap) {
        PULSAR_MULTISEQ_ERR("mixed decode: bad args (n_rows=%u prefill_cap=%u)",
                      n_rows, s ? s->graph->prefill_cap : 0u);
        return 1;
    }
    /* plan-34 inc 3: the engine emits ONE logit row per BANK RUN (last-of-run),
     * not one per token-row, so the caller need only size `logits` for n_runs =
     * the number of contiguous per-bank runs (<= PULSAR_MSEQ_MAX). Compute it here
     * for the capacity check; the engine returns it via out_n_rows. */
    uint32_t n_runs = 0;
    for (uint32_t k = 0; k < n_rows; k++)
        if (k + 1 == n_rows || reqs[k + 1].bank != reqs[k].bank) n_runs++;
    /* Inc 6 (ALL_ROWS): one logit row PER BATCH ROW, for the batched spec
     * verify's accept walk.  The step may carry at most PULSAR_SPEC_ROW_BUDGET
     * rows -- the M-neutral range, so every row is a decode row on the
     * width-neutral arm and no session's numerics depend on the batch width
     * (L177).  Above it the step is refused by name, never run on the
     * tensor-core arm the row-kind inference would otherwise choose. */
    const uint32_t head_rows =
        (max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS) ? n_rows : n_runs;
    if (max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS && n_rows > PULSAR_SPEC_ROW_BUDGET) {
        PULSAR_MULTISEQ_ERR("mixed decode: a speculative verify step carries at most %u rows "
                         "(the M-neutral range; n_rows=%u) -- refusing",
                      (unsigned)PULSAR_SPEC_ROW_BUDGET, n_rows);
        return 1;
    }
    if (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)head_rows * PULSAR_N_VOCAB) {
        PULSAR_MULTISEQ_ERR("mixed decode: logits capacity %d < %u rows x %u",
                      logits_cap, head_rows, (unsigned)PULSAR_N_VOCAB);
        return 1;
    }
    return ds4_batch_step(s, reqs, n_rows, logits, out_n_rows, max_head_runs, NULL, "mixed decode step", err,
                          errlen);
}

int pulsar_session::decode_fused(const pulsar_multiseq_req *reqs, uint32_t n_rows,
                                 const pulsar_fused_shape *shape, float *logits, int logits_cap,
                                 uint32_t *out_n_rows, char *err, size_t errlen) {
    auto *s = this;
    if (out_n_rows) *out_n_rows = 0;
    if (!reqs || !shape || !logits || n_rows == 0 || n_rows > s->graph->prefill_cap) {
        PULSAR_MULTISEQ_ERR("fused step: bad args (n_rows=%u prefill_cap=%u)", n_rows, s->graph->prefill_cap);
        return 1;
    }
    const uint32_t heads = pulsar_fused_shape_heads(shape);
    if (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)(shape->n_dec + heads) * PULSAR_N_VOCAB) {
        PULSAR_MULTISEQ_ERR("fused step: logits capacity %d < %u rows x %u", logits_cap, shape->n_dec + heads,
                         (unsigned)PULSAR_N_VOCAB);
        return 1;
    }
    return ds4_batch_step(s, reqs, n_rows, logits, out_n_rows, 0u, shape, "fused step", err, errlen);
}
#undef PULSAR_MULTISEQ_ERR

int pulsar_session_fused_local(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                               const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                               char *err, size_t errlen) {
    s->fused_logits = nullptr;
    s->fused_n_dec = s->fused_heads = 0;
    const int rc = s->engine->family->session->decode_fused(s, reqs, n_rows, shape, logits, logits_cap, out_n_rows,
                                                            err, errlen);
    if (rc == 0) {
        s->fused_logits = logits;
        s->fused_n_dec = shape->n_dec;
        s->fused_heads = pulsar_fused_shape_heads(shape);
    }
    return rc;
}

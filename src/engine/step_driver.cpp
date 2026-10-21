/* step_driver.cpp -- the core row-step driver (step_driver.h, L295 F4).
 *
 * Lifted from Qwen's step driver (family_qwen.cpp, L251 .. L289), where it was generic in shape and Qwen's only by
 * where it lived: the row layout and its checks, the forward over the layer plan through the family's blocks, the
 * head list, the drafter's absorb, each bank's position and prefill frontier, and the speculation round's verify
 * target.  The family's facts it reads are pulsar_driver_ops (its blocks), the engine's layer plan, and the
 * session's declared state (state_kinds.h). */
#include "pulsar_engine_internal.h"
#include "spec_mtp.h"
#include "spec_block.h"
#include "tp/pulsar_tp.h"

static const pulsar_driver_ops *DOPS(const pulsar_engine *e) { return e->family->driver; }
static const char *ARCH(const pulsar_engine *e) { return e->family->arch; }

pulsar_driver_heads pulsar_driver_heads_span(uint32_t row0, uint32_t n) {
    pulsar_driver_heads h;
    h.n = n;
    for (uint32_t i = 0; i < n && i < PULSAR_DRIVER_HEADS_MAX; i++) h.row[i] = row0 + i;
    return h;
}

const char *pulsar_driver_first_missing(const pulsar_driver_ops *ops, const pulsar_layer_plan *plan,
                                        uint32_t *at_layer) {
    *at_layer = UINT32_MAX;
    if (!ops->embed) return "embed";
    for (uint32_t il = 0; il < plan->n_layer; il++) {
        if (plan->kind[il] >= PULSAR_LAYER_KIND_COUNT || !ops->block[plan->kind[il]]) {
            *at_layer = il;
            return "block";
        }
    }
    return ops->head ? NULL : "head";
}

/* L266 step 7: under TP a rank's block leaves a PARTIAL output in y; the group's sum is the block's.  One exchange,
 * f32 [rows][width], in exchange slot `slot`.  One GPU: nothing to do. */
bool pulsar_driver_tp_allreduce(const pulsar_driver_step *st, uint32_t slot, pulsar_gpu_tensor *y, uint32_t width,
                                const char *what) {
    pulsar_driver_state *d = st->buf;
    if (!d->tp) return true;
    const pulsar_tp_rows x = {d->tp, d->tp_slab_dev, d->tp_bulk_dev, d->tp_ticket, &d->tp_seq, width};
    pulsar_gpu_tensor *v = pulsar_gpu_tensor_view(y, 0, (uint64_t)st->n_rows * width * sizeof(float));
    const bool ok = v && pulsar_tp_allreduce_rows(&x, slot, st->n_rows, v, NULL, what);
    pulsar_gpu_tensor_free(v);
    return ok;
}

/* ---- the session ---------------------------------------------------------------------------------------------- */

static void driver_bufs_free(pulsar_driver_state *d) {
    pulsar_gpu_tensor_free(d->logits);
    pulsar_gpu_tensor_free(d->row_pos);
    pulsar_gpu_tensor_free(d->row_bank);
    pulsar_gpu_tensor_free(d->run_first_dev);
    for (uint32_t a = 0; a < PULSAR_DRIVER_SCRATCH_MAX; a++) pulsar_gpu_tensor_free(d->scratch[a]);
    pulsar_gpu_tensor_free(d->tp_ticket);
    pulsar_gpu_tensor_free(d->tp_vocab_own);
    pulsar_mtp_lane_free(d->mtp);
    d->mtp = NULL;
    pulsar_block_lane_free(d->block);
    d->block = NULL;
}

/* The family's state, the driver's buffers and the declared state, built into a session the core allocated (L272
 * P2: the core sets ctx_size, the prefill cap and the logits row, and measures the bytes this allocates).  The same
 * code prices it: the core runs the create under pulsar_gpu_tensor_dry_begin (session_cost_bytes_banked). */
int pulsar_driver_session_create(pulsar_session *s, uint32_t n_banks) {
    pulsar_engine *e = s->engine;
    const pulsar_driver_ops *ops = DOPS(e);
    uint32_t at = 0;
    if (const char *miss = pulsar_driver_first_missing(ops, &e->plan, &at)) {
        char where[48] = "";
        if (at != UINT32_MAX) snprintf(where, sizeof(where), " for layer %u (%s)", at,
                                       pulsar_layer_kind_name(e->plan.kind[at]));
        fprintf(stderr, "pulsar: %s: the family supplies no %s%s -- refusing the session\n", ARCH(e), miss, where);
        return 1;
    }
    pulsar_driver_state *d = ops->create(s, n_banks);
    if (!d) return 1;
    s->driver = d;
    const bool verifies = ops->verifies(s);
    const uint64_t W = e->logits_width(), rows = s->prefill_cap;
    d->max_rows = s->prefill_cap;
    d->logits   = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_DRIVER_HEADS_MAX * W * sizeof(float));
    d->row_pos  = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
    d->row_bank = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
    bool ok = d->logits && d->row_pos && d->row_bank;
    if (ok && verifies) {
        d->run_first_dev = pulsar_gpu_tensor_alloc((uint64_t)(PULSAR_DRIVER_HEADS_MAX + 1u) * sizeof(int32_t));
        ok = d->run_first_dev != NULL;
    }
    /* L296 N9: the chained-MTP drafter's lane when the family declares modules -- before the declared state, whose
     * fresh banks set the lane's counters (pulsar_mtp_drafter_at) */
    if (ok && ops->mtp && ops->mtp->modules(e) > 0) ok = (d->mtp = pulsar_mtp_lane_create(s, ops->mtp, n_banks,
                                                                                           d->max_rows)) != NULL;
    /* L298: the block drafter's lane (its tap rows and its op arena) when the engine loaded one */
    if (ok && e->block_drafter) ok = (d->block = pulsar_block_lane_create(s, n_banks, d->max_rows)) != NULL;
    /* the declared state (its verify capture when the session verifies), every bank fresh */
    if (ok) ok = (s->states = pulsar_state_create(e, &e->state_plan, d, n_banks, (uint32_t)s->ctx_size,
                                                  verifies ? PULSAR_STATE_VERIFY_ROWS : 0u)) != NULL;
    for (uint32_t a = 0; ok && a < ops->n_scratch && a < PULSAR_DRIVER_SCRATCH_MAX; a++) {
        const uint64_t b = ops->scratch_bytes(a, d->max_rows, s->states->ctx);
        if (b) ok = (d->scratch[a] = pulsar_gpu_tensor_alloc(b)) != NULL;
    }
    /* L266 step 7: under TP, the row all-reduce's stage ticket (one u32) and the vocab gather's own slice (its row
     * lane's; sized whatever the transport, so the price holds) -- allocated here so the dry run prices them */
    if (ok && e->model.tp_n_ranks > 1) {
        uint32_t nl = 0, ne = 0, nv = 0;
        e->family->tp_shape(e, &nl, &ne, &nv);
        const uint64_t vb = pulsar_tp_vocab_own_bytes_for((uint32_t)e->model.tp_n_ranks, (uint64_t)ne * sizeof(float),
                                                          nv, PULSAR_DRIVER_HEADS_MAX);
        ok = (d->tp_ticket = pulsar_gpu_tensor_alloc(sizeof(uint32_t))) != NULL &&
             (d->tp_vocab_own = pulsar_gpu_tensor_alloc(vb)) != NULL;
    }
    if (ok) s->states->ckpt.artifact = ops->artifact(e);
    if (ok && e->tp) {   /* L266 step 7: the engine's lanes; the ticket is the session's (priced with it), zeroed here */
        const uint32_t zero = 0;
        d->tp = e->tp;
        d->tp_slab_dev = e->tp_slab_dev;
        d->tp_bulk_dev = e->tp_bulk_dev;
        ok = d->tp_ticket && pulsar_gpu_tensor_write(d->tp_ticket, 0, &zero, sizeof(zero));
        if (!ok) fprintf(stderr, "pulsar: %s: the TP stage ticket is missing\n", ARCH(e));
    }
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session allocation failed (%u banks x %d tokens)\n", ARCH(e), n_banks,
                s->ctx_size);
        pulsar_driver_session_destroy(s);
        return 1;
    }
    if (!pulsar_gpu_tensor_dry_active()) {   /* a pricing run allocates nothing */
        const pulsar_state_plan *p = s->states->plan;
        double fixed = 0, per_token = 0;
        for (uint32_t k = 0; k < p->n_slots; k++)
            for (uint32_t id = 0; id < p->n_decl; id++) {
                if (!(p->slot[k] >> id & 1u)) continue;
                if (p->decl[id].kind == PULSAR_STATE_POOL) per_token += (double)p->decl[id].bytes / p->decl[id].tokens;
                else fixed += (double)pulsar_state_decl_bank_bytes(&p->decl[id]);
            }
        fprintf(stderr, "pulsar: %s session: %u bank(s) x %d tokens, %u-row steps "
                        "(%.1f MiB fixed per bank + %.1f KiB per token)\n",
                ARCH(e), n_banks, s->ctx_size, s->prefill_cap, fixed / 1048576.0, per_token / 1024.0);
    }
    return 0;
}

void pulsar_driver_session_destroy(pulsar_session *s) {
    /* L296 perf: a captured decode step bakes this session's buffers (its key names them): drop every graph before
     * they are freed and another allocation can take their addresses */
    pulsar_gpu_seg_reset();
    pulsar_state_destroy(s->states);
    s->states = NULL;
    if (s->driver) driver_bufs_free(s->driver);
    DOPS(s->engine)->destroy(s);
    s->driver = NULL;
}

/* ---- the forward ---------------------------------------------------------------------------------------------- */

struct driver_image_ud {
    const pulsar_driver_ops *ops;
    const pulsar_driver_step *st;
};

static bool driver_image_rows(void *ud, const uint16_t *rows, uint32_t n_rows, uint32_t row0, uint32_t n_tokens) {
    const driver_image_ud *u = (const driver_image_ud *)ud;
    return u->ops->image_rows(u->st, rows, n_rows, row0, n_tokens);
}

/* L296 perf: a captured step's key -- everything host-side its launches depend on: the session's buffers (the
 * driver state; a destroyed session drops every graph, pulsar_driver_session_destroy), the step's kind (a decode step
 * or a pure verify), the row count, the head list, and the family's own key (pulsar_driver_ops::stage).  Tagged in the
 * top bits so it never meets the output head segment's structured keys (gpu_graph_output_head_seg_key) in the one
 * segment table. */
static uint64_t driver_graph_key(const pulsar_driver_state *d, bool verify, uint32_t n_rows,
                                 const pulsar_driver_heads &heads, uint64_t family_key) {
    auto mix = [](uint64_t h, uint64_t v) {
        uint64_t z = h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    uint64_t h = mix(0x44525647ull /* "DRVG" */, (uint64_t)(uintptr_t)d);
    h = mix(h, verify ? 2u : 1u);
    h = mix(h, n_rows);
    h = mix(h, heads.n);
    for (uint32_t i = 0; i < heads.n; i++) h = mix(h, heads.row[i]);
    h = mix(h, family_key);
    return (h >> 4) | (0xDull << 60);
}

/* rule 5: each graphed lane (the decode step, the verify step) says once that it ran (the counters are
 * pulsar_gpu_seg_counts) */
static void driver_graph_announce(const char *arch, bool verify, const char *what) {
    static bool seen_[2][2] = {};
    bool &seen = seen_[verify ? 1 : 0][what[0] == 'c' ? 0 : 1];
    if (seen) return;
    seen = true;
    fprintf(stderr, "pulsar: %s: L296 %s step graph %s (one captured graph per %s shape)\n", arch,
            verify ? "verify" : "decode", what, verify ? "verify" : "decode");
}

/* One step of the trunk: the plan, in its order.  DECODE: one row per bank; PREFILL: runs of one bank each -- a
 * verify's [0, n_verify), then the fused step's prompt runs.  The `heads` rows are headed into the driver's slab
 * (d->logits, the logits width a row), which becomes the session's step rows (L302: draws read them there); a
 * caller that wants them on the host too passes `logits_out`.  n_verify > 0 arms the declared state's verify capture
 * over the verify rows. */
static bool driver_forward(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                           const int32_t *bank, uint32_t n_rows, const pulsar_driver_heads &heads, float *logits_out,
                           uint32_t n_verify = 0) {
    pulsar_engine *e = s->engine;
    const pulsar_driver_ops *ops = DOPS(e);
    pulsar_driver_state *d = s->driver;
    pulsar_state_set *ss = s->states;
    if (n_rows == 0 || n_rows > d->max_rows || heads.n > PULSAR_DRIVER_HEADS_MAX) {
        fprintf(stderr, "pulsar: %s: a step of %u rows heading %u (at most %u rows, %u heads) -- refusing\n",
                ARCH(e), n_rows, heads.n, d->max_rows, PULSAR_DRIVER_HEADS_MAX);
        return false;
    }
    pulsar_driver_step &st = *ops->trunk(s);
    st.model = &e->model;
    st.plan = &e->plan;
    st.states = ss;
    st.buf = d;
    st.mode = mode;
    st.n_rows = n_rows;
    st.tokens = tokens;
    st.pos = pos;
    st.bank = bank;
    st.verify = n_verify > 0;
    st.n_dec = pulsar_step_n_dec(mode, n_verify, n_rows);
    st.tap = e->imatrix_tap;
    std::vector<uint32_t> rope;
    st.rope = ops->rope ? ops->rope(s, pos, bank, n_rows, &rope) : NULL;
    if (st.verify && (mode != PULSAR_STEP_PREFILL || n_verify > n_rows || n_verify > PULSAR_DRIVER_HEADS_MAX ||
                      !ss->verify_rows)) {
        fprintf(stderr, "pulsar: %s: a verify step of %u rows is outside the capture -- refusing\n", ARCH(e),
                n_verify);
        return false;
    }
    if (st.verify && n_verify > PULSAR_GPU_MNEUTRAL_ROWS_MAX) {
        static bool announced = false;   /* rule 5: the wide verify says once that it ran */
        if (!announced) {
            announced = true;
            fprintf(stderr, "pulsar: %s: L288 C1 wide verify ran: %u rows, the decode arms in blocks of %u (every row "
                            "a one-token decode's bytes)\n", ARCH(e), n_verify, PULSAR_GPU_MNEUTRAL_ROWS_MAX);
        }
    }
    /* L272 P1 S4 / L284 #2: a PREFILL step's rows as runs -- one bank each, consecutive positions, a bank in one
     * run.  The verify rows [0, n_verify) are one run a bank (every row headed); a run never crosses n_verify; the
     * rows after it are the fused step's prompt runs, at most PULSAR_FUSED_PF_MAX of them. */
    uint32_t run_first[PULSAR_DRIVER_HEADS_MAX + PULSAR_FUSED_PF_MAX + 1] = {0};
    uint32_t n_runs = 0;
    if (mode == PULSAR_STEP_PREFILL) {
        n_runs = 1;
        uint32_t n_prompt = n_verify == 0 ? 1u : 0u;   /* the runs from n_verify on */
        for (uint32_t r = 1; r < n_rows; r++) {
            if (bank[r] == bank[r - 1] && r != n_verify) {
                if (pos[r] == pos[r - 1] + 1) continue;
                fprintf(stderr, "pulsar: %s: prefill row %u (bank %d pos %d) does not follow its run -- refusing\n",
                        ARCH(e), r, bank[r], pos[r]);
                return false;
            }
            bool seen = false;
            for (uint32_t k = 0; k < n_runs; k++) seen |= bank[run_first[k]] == bank[r];
            if (r >= n_verify) n_prompt++;
            if (seen || n_prompt > PULSAR_FUSED_PF_MAX) {
                fprintf(stderr, "pulsar: %s: prefill row %u (bank %d) starts a run %s -- refusing\n", ARCH(e), r,
                        bank[r], seen ? "of a bank the step already carries" : "past the step's run bound");
                return false;
            }
            run_first[n_runs++] = r;
        }
        run_first[n_runs] = n_rows;
    }
    st.n_runs = n_runs;
    st.run_first = n_runs ? run_first : NULL;
    const uint32_t n_vruns = st.verify ? pulsar_step_verify_runs(&st) : 0u;
    /* the verify's runs, and each run's host state before the step: a commit finds its bank's run */
    if (st.verify) pulsar_state_verify_begin(ss, n_vruns, run_first, bank, pos);
    /* L296 N9: the verify rows' tokens, which the MTP lane's absorb pairs a bank's still-parked row with */
    if (st.verify && d->mtp) pulsar_mtp_note_verify(d->mtp, tokens, n_verify);
    /* L289: enqueued, not waited for -- the step's uploads never drain the stream */
    bool ok = pulsar_gpu_tensor_write_async(d->row_pos, 0, pos, (uint64_t)n_rows * sizeof(int32_t)) != 0 &&
              pulsar_gpu_tensor_write_async(d->row_bank, 0, bank, (uint64_t)n_rows * sizeof(int32_t)) != 0;
    int32_t rf[PULSAR_DRIVER_HEADS_MAX + 1];
    if (ok && n_vruns > 1) {   /* a ragged verify's runs (the recurrent ops read them on the device) */
        for (uint32_t k = 0; k <= n_vruns; k++) rf[k] = (int32_t)run_first[k];
        ok = pulsar_gpu_tensor_write_async(d->run_first_dev, 0, rf, (uint64_t)(n_vruns + 1u) * sizeof(int32_t)) != 0;
    }
    /* L296 perf: the family's host half (every upload the launches read), and what its launches depend on */
    uint64_t family_key = 0;
    if (ok && ops->stage) ok = ops->stage(&st, &family_key);
    if (ok) ok = pulsar_gpu_begin_commands() != 0;
    /* each verify run's open blocks, before any op of the step writes them (state_kinds.h STAGE) */
    if (ok && st.verify) ok = pulsar_state_verify_capture(ss);
    /* the step's launches: embed, the image rows of a prompt chunk, the plan's blocks, the head list */
    auto launches = [&]() -> bool {
        bool lok = ops->embed(&st);
        /* L268: a sync's prefill chunk takes the image rows of the blocks it owns over their pads' embeddings (HF
         * masked_scatter), through the family's writer */
        if (lok && mode == PULSAR_STEP_PREFILL && s->sync_images && n_runs == 1 && bank[0] == (int32_t)ss->live_bank) {
            if (!ops->image_rows) {
                fprintf(stderr, "pulsar: %s: a prompt chunk with image blocks, and the family takes no image rows\n",
                        ARCH(e));
                lok = false;
            } else {
                driver_image_ud u = {ops, &st};
                lok = pulsar_image_merge_chunk(e, s->sync_prompt->v, s->sync_prompt->len, s->sync_images,
                                               s->sync_n_images, (uint32_t)pos[0], n_rows, driver_image_rows, &u);
            }
        }
        for (uint32_t il = 0; lok && il < e->plan.n_layer; il++) {
            lok = ops->block[e->plan.kind[il]](&st, il);
            /* the diagnostic tap after layer il (PULSAR_CUDA_GRAPH_DUMP_PREFIX armed; the family's dump re-checks
             * its own arming, so an unarmed run does no work) */
            if (lok && ops->dump && gpu_graph_debug_dump_enabled()) ops->dump(&st, il);
        }
        /* the head list into its logits rows (L284 #2): a span of consecutive decode rows is one call (a verify
         * heads every row), a prompt row is its own call -- the one-row head its classic chunk makes */
        for (uint32_t i = 0, j; lok && i < heads.n; i = j) {
            for (j = i + 1; j < heads.n && heads.row[i] < st.n_dec && heads.row[j] == heads.row[j - 1] + 1u &&
                            heads.row[j] < st.n_dec; j++) {}
            lok = ops->head(&st, heads.row[i], j - i, i);
        }
        return lok;
    };
    /* L296 perf: a DECODE step of a staged family runs its launches as one captured graph per key (the session's
     * buffers, the step's kind, the rows, the head list, the family's key): the L119 segment cache captures a key's
     * 12th use and replays it after.  L296 fullspec: so does a PURE verify step (every row a verify row -- a
     * speculative round's; its capture of the recurrent state is the driver's, before the launches), keyed by the
     * family on its run structure too.  Eager while a diagnostic tap or an importance-matrix observer reads the step
     * from the host, and under TP (the row exchange waits on the host). */
    const bool graphable = mode == PULSAR_STEP_DECODE ? !st.verify : st.verify && n_verify == n_rows;
    const uint64_t gkey = ok && graphable && family_key && !st.tap && !d->tp && !gpu_graph_debug_dump_enabled()
                              ? driver_graph_key(d, st.verify, n_rows, heads, family_key)
                              : 0;
    if (ok && gkey) {
        const int seg = pulsar_gpu_seg_enter(gkey);
        if (seg == 2) {
            driver_graph_announce(ARCH(e), st.verify, "replayed");
        } else if (seg == 1) {
            const bool body = launches();
            /* a failed capture never ran its work: the launches again, eager (a real refusal refuses again) */
            if (pulsar_gpu_seg_exit(gkey, body ? 1 : 0)) driver_graph_announce(ARCH(e), st.verify, "captured");
            else ok = launches();
        } else {
            ok = launches();
        }
    } else if (ok) {
        ok = launches();
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    /* the family's step end: settles what a failed step left in flight and, on a completed step, its own checks
     * (rule 9) */
    ok = ops->step_end(&st, ok);
    if (!ok) {
        /* the step may have advanced any of its banks' state before it failed: those banks start over, and the
         * live bank's checkpoint no longer describes its state */
        for (uint32_t r = 0; r < n_rows; r++)
            if (!pulsar_state_reset_bank(ss, (uint32_t)bank[r]))
                fprintf(stderr, "pulsar: %s: could not clear bank %d after a failed step\n", ARCH(e), bank[r]);
        s->checkpoint_valid = false;
    }
    s->dev.step_rows = ok ? d->logits : NULL;
    s->dev.step_n = ok ? heads.n : 0u;
    if (ok && heads.n && logits_out)
        ok = pulsar_gpu_tensor_read(d->logits, 0, logits_out, (uint64_t)heads.n * e->logits_width() * sizeof(float)) != 0;
    return ok;
}

/* ---- the sync's prefill ----------------------------------------------------------------------------------------- */

/* One prefill chunk of the live bank (pulsar_prefill_chunk_fn): rows [pos0, pos0 + rows) of the prompt through the
 * forward (the drafter's lane absorbs them too), the bank's position advanced, and -- while the prefill CONTINUES the
 * bank's prefill-only history (L266 step 5) -- that history extended. */
struct driver_prefill_ctx {
    int32_t *pos, *bank;
    uint32_t live;
    bool canonical;
};

static bool driver_prefill_chunk(pulsar_session *s, const pulsar_tokens *prompt, uint32_t pos0, uint32_t rows,
                                 bool last, void *ud) {
    driver_prefill_ctx *c = (driver_prefill_ctx *)ud;
    const pulsar_driver_ops *ops = DOPS(s->engine);
    for (uint32_t r = 0; r < rows; r++) c->pos[r] = (int32_t)(pos0 + r);
    bool ok = driver_forward(s, PULSAR_STEP_PREFILL, prompt->v + pos0, c->pos, c->bank, rows,
                                    pulsar_driver_heads_span(rows - 1u, last ? 1u : 0u), NULL);
    if (ok && last) ok = s->live_from_rows(s->driver->logits, 0);
    if (ok && ops->absorb)
        ok = ops->absorb(s, PULSAR_STEP_PREFILL, prompt->v + pos0, c->pos, c->bank, rows, 0, false);
    if (ok) pulsar_state_set_pos(s->states, c->live, pos0 + rows);
    if (ok && c->canonical) s->states->prefill_pos[c->live] = pos0 + rows;
    return ok;
}

/* Prefill the live bank from `start` through the core loop (prefill_loop.cpp, L272 P2): prefill_cap chunks,
 * progress, the cancel hook at every chunk boundary, and the shared checkpoint ladder (P13: a capture at every grid
 * chunk end while the prefill continues the bank's prefill-only history, the last chunk cut at the prompt's last
 * grid point).  A cut changes no byte on a split-invariant family (every prompt chunk takes the prefill arms,
 * session_contract_gate C1), which is also why an interrupted sync resumes exactly.  Returns the loop's 0 /
 * PULSAR_SESSION_SYNC_INTERRUPTED / 1. */
int pulsar_driver_prefill(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start) {
    pulsar_state_set *ss = s->states;
    driver_prefill_ctx c;
    c.live = ss->live_bank;
    c.canonical = ss->prefill_pos[c.live] == start && !ss->frontier_stale[c.live];
    c.pos = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    c.bank = (int32_t *)xmalloc((size_t)s->prefill_cap * sizeof(int32_t));
    for (uint32_t r = 0; r < s->prefill_cap; r++) c.bank[r] = (int32_t)c.live;
    const int rc = pulsar_prefill_loop(s, prompt, start, driver_prefill_chunk, &c);
    free(c.pos);
    free(c.bank);
    return rc;
}

/* The family's sync is the core's (sync_driver.cpp, L272 P2) over these: the bank's position is the declared state's
 * authority, a reset clears the bank's state, the prefill is pulsar_driver_prefill, and the shape is the session's
 * cap with no alignment (a cut anywhere is the cold bytes: the plan's split_invariant). */
bool pulsar_driver_sync_state_agrees(pulsar_session *s) {
    return s->states->bank_pos[s->states->live_bank] == (uint32_t)s->checkpoint.len;
}

bool pulsar_driver_sync_reset_bank(pulsar_session *s) {
    return pulsar_state_reset_bank(s->states, s->states->live_bank);
}

void pulsar_driver_prefill_shape(pulsar_session *s, pulsar_prefill_shape *out) {
    out->cap = s->prefill_cap;
    out->resumed_cap = s->prefill_cap;
    out->align = 1u;
}

/* ---- decode ----------------------------------------------------------------------------------------------------- */

/* The row (the core's eval checked the context, the id, and the bank standing at the view's end): the trunk's decode
 * step, the drafter lane's absorb, and the bank's position advanced. */
int pulsar_driver_eval_row(pulsar_session *s, int token, char *err, size_t errlen) {
    const pulsar_driver_ops *ops = DOPS(s->engine);
    const uint32_t live = s->states->live_bank;
    const int32_t tok = token, pos = s->checkpoint.len, bank = (int32_t)live;
    if (!driver_forward(s, PULSAR_STEP_DECODE, &tok, &pos, &bank, 1, pulsar_driver_heads_span(0, 1), NULL) ||
        !s->live_from_rows(s->driver->logits, 0)) {
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", ARCH(s->engine));
        return 1;
    }
    if (ops->absorb && !ops->absorb(s, PULSAR_STEP_DECODE, &tok, &pos, &bank, 1, 0, true)) {
        if (err) snprintf(err, errlen, "%s: the drafter's step refused (see the log)", ARCH(s->engine));
        return 1;
    }
    pulsar_state_set_pos(s->states, live, (uint32_t)pos + 1u);
    return 0;
}

int pulsar_driver_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, float *logits,
                                  int logits_cap, char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_state_set *ss = s->states;
    const uint32_t nv = e->logits_width();
    if (!reqs || n == 0 || n > PULSAR_DRIVER_HEADS_MAX || n > s->driver->max_rows ||
        (logits && (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n * nv))) {
        if (err) snprintf(err, errlen, "%s: batched decode of %u rows refused (bad args)", ARCH(e), n);
        return 1;
    }
    int32_t tok[PULSAR_DRIVER_HEADS_MAX], pos[PULSAR_DRIVER_HEADS_MAX], bank[PULSAR_DRIVER_HEADS_MAX];
    for (uint32_t i = 0; i < n; i++) {
        /* One row per bank, at that bank's next position: a recurrent state advances exactly one token per row. */
        if (reqs[i].bank >= ss->n_banks || (uint32_t)reqs[i].pos != ss->bank_pos[reqs[i].bank]) {
            if (err) snprintf(err, errlen, "%s: row %u (bank %u pos %d) is not its bank's next position", ARCH(e), i,
                              reqs[i].bank, reqs[i].pos);
            return 1;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (reqs[j].bank == reqs[i].bank) {
                if (err) snprintf(err, errlen, "%s: bank %u has two rows in one decode step", ARCH(e), reqs[i].bank);
                return 1;
            }
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bank[i] = (int32_t)reqs[i].bank;
    }
    const pulsar_driver_ops *ops = DOPS(e);
    if (!driver_forward(s, PULSAR_STEP_DECODE, tok, pos, bank, n, pulsar_driver_heads_span(0, n), logits) ||
        (ops->absorb && !ops->absorb(s, PULSAR_STEP_DECODE, tok, pos, bank, n, 0, false))) {
        /* the failed step reset the banks it touched (pulsar_driver_forward): fatal for these rows */
        if (err) snprintf(err, errlen, "%s: decode refused (see the log for the op)", ARCH(e));
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        pulsar_state_set_pos(ss, reqs[i].bank, ss->bank_pos[reqs[i].bank] + 1u);
        if (reqs[i].bank == ss->live_bank) {   /* the live bank moved past the host view */
            s->checkpoint_valid = false;
            s->logits_stale = true;
        }
    }
    return 0;
}

/* A verify's rows reqs[0, n): one run a bank, each run starting at its bank's next position and each further row
 * following the one before -- the speculation lane's verify, alone (decode_mixed) or in front of a fused step's
 * prompt run.  Fills tok / pos / bk; false (nothing moved) names the first row that is not. */
static bool driver_verify_rows(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, int32_t *tok,
                               int32_t *pos, int32_t *bk, char *err, size_t errlen) {
    const pulsar_state_set *ss = s->states;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t b = reqs[i].bank;
        const bool run_start = i == 0 || reqs[i - 1].bank != b;
        const uint32_t want = run_start ? (b < ss->n_banks ? ss->bank_pos[b] : 0u) : (uint32_t)reqs[i - 1].pos + 1u;
        if (b >= ss->n_banks || (uint32_t)reqs[i].pos != want) {
            if (err) snprintf(err, errlen, "%s: verify row %u (bank %u pos %d) is not its run's next position",
                              ARCH(s->engine), i, b, reqs[i].pos);
            return false;
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bk[i] = (int32_t)b;
    }
    return true;
}

/* The server's batched lane: its plain decode is this entry with one row per bank and no prefill runs
 * (max_head_runs 0), which is exactly the batched decode.  A step that carries a PREFILL run (a bank with more than
 * one row) is refused before anything moves: a driver family declares no PULSAR_FAMILY_CAP_MIXED_PREFILL, so the
 * server's mixed lane never sends one -- its prompts ride the fused step or the sync. */
int pulsar_driver_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits,
                               int logits_cap, uint32_t *out_n_rows, uint32_t max_head_runs, char *err,
                               size_t errlen) {
    pulsar_engine *e = s->engine;
    if (out_n_rows) *out_n_rows = 0;
    if (max_head_runs == PULSAR_MSEQ_HEAD_ALL_ROWS) {
        /* L272 P1 S3/S4: the speculation lane's verify -- one run per bank, each [p, p + k] at its bank's next
         * position, every row headed into the caller's block, the declared state's capture armed (the target's
         * commit rolls each bank's rejected rows back, pulsar_state_commit).  At most PULSAR_DRIVER_HEADS_MAX rows,
         * every kernel at its decode arm (L288 C1: in 16-row blocks past 16), so each row's logits are the bytes a
         * one-token decode gives. */
        const uint32_t nv = e->logits_width();
        if (!reqs || n_rows == 0 || n_rows > PULSAR_DRIVER_HEADS_MAX || !s->states->verify_rows ||
            (logits && (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n_rows * nv))) {
            if (err) snprintf(err, errlen, "%s: a verify step of %u rows refused (bad args, over %u rows, or no "
                              "verify capture)", ARCH(e), n_rows, PULSAR_DRIVER_HEADS_MAX);
            return 1;
        }
        int32_t tok[PULSAR_DRIVER_HEADS_MAX], pos[PULSAR_DRIVER_HEADS_MAX], bk[PULSAR_DRIVER_HEADS_MAX];
        if (!driver_verify_rows(s, reqs, n_rows, tok, pos, bk, err, errlen)) return 1;
        if (!driver_forward(s, PULSAR_STEP_PREFILL, tok, pos, bk, n_rows,
                                   pulsar_driver_heads_span(0, n_rows), logits, n_rows)) {
            if (err) snprintf(err, errlen, "%s: the verify step refused (see the log for the op)", ARCH(e));
            return -1;
        }
        if (out_n_rows) *out_n_rows = n_rows;
        return 0;
    }
    for (uint32_t i = 0; reqs && i < n_rows; i++)
        for (uint32_t j = 0; j < i; j++)
            if (reqs[j].bank == reqs[i].bank) {
                if (err) snprintf(err, errlen, "%s: a fused prefill run (bank %u, %u+ rows) is not implemented; "
                                  "prefill goes through sync", ARCH(e), reqs[i].bank, 2u);
                return 1;
            }
    const int rc = pulsar_driver_decode_multiseq(s, reqs, n_rows, logits, logits_cap, err, errlen);
    if (rc == 0 && out_n_rows) *out_n_rows = n_rows;
    return rc;
}

/* L284 #2: the fused step (contract in pulsar.h) -- a verify's rows [0, n_dec) and the shape's n_pf prompt runs
 * [n_dec, n_rows) in one forward.  The verify rows are exactly the speculation lane's verify (decode_mixed with every
 * row headed): headed in place, their state captured for the commit's rollback, their bank positions left for the
 * commit to move.  Each prompt run is exactly a sync's prefill chunk of the same rows on its bank, at any width: a
 * run from position 0 resets its bank first (as a cold sync does), the drafter's lane absorbs it, the bank's position
 * moves past it, and so does its prefill-only history when the run continues it.  A run with head_last set has its
 * last row headed into the next logits row after the verify's, in run order.  The record of each chunk in its bank's
 * history is the core's pulsar_session_note_prefilled.  1 = refused before anything moved, -1 = the step failed with
 * state moved (its banks' state is fatal). */
int pulsar_driver_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                               const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                               char *err, size_t errlen) {
    if (out_n_rows) *out_n_rows = 0;
    pulsar_engine *e = s->engine;
    const pulsar_driver_ops *ops = DOPS(e);
    pulsar_state_set *ss = s->states;
    const uint32_t nv = e->logits_width();
    if (!reqs || !shape || n_rows == 0 || n_rows > s->driver->max_rows) {
        if (err) snprintf(err, errlen, "%s: fused step: bad args (%u rows, at most %u)", ARCH(e), n_rows,
                          s->driver->max_rows);
        return 1;
    }
    const uint32_t n_dec = shape->n_dec, heads = pulsar_fused_shape_heads(shape);
    if (n_dec >= n_rows || shape->n_pf == 0 || shape->n_pf > PULSAR_FUSED_PF_MAX) {
        if (err) snprintf(err, errlen, "%s: a fused step is a verify and 1..%u prompt runs (n_dec %u of %u rows, %u "
                          "prompt runs)", ARCH(e), PULSAR_FUSED_PF_MAX, n_dec, n_rows, shape->n_pf);
        return 1;
    }
    if (n_dec > PULSAR_DRIVER_HEADS_MAX || (n_dec > 0 && !ss->verify_rows) ||
        n_dec + heads > e->family->session->fused_heads_max ||
        (logits && (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)(n_dec + heads) * nv))) {
        if (err) snprintf(err, errlen, "%s: fused step: %u verify rows (at most %u, with a verify capture) + %u "
                          "headed (at most %u rows headed), logits capacity %d", ARCH(e), n_dec,
                          PULSAR_DRIVER_HEADS_MAX, heads, e->family->session->fused_heads_max, logits_cap);
        return 1;
    }
    std::vector<int32_t> tok(n_rows), pos(n_rows), bk(n_rows);
    if (!driver_verify_rows(s, reqs, n_dec, tok.data(), pos.data(), bk.data(), err, errlen)) return 1;
    /* the prompt runs: each on a bank no other row of the step is on, consecutive positions, text rows */
    uint32_t first[PULSAR_FUSED_PF_MAX + 1];
    uint32_t n_pf = 0;
    const pulsar_family_vision *vis = e->family->vision;
    for (uint32_t i = n_dec; i < n_rows; i++) {
        const uint32_t b = reqs[i].bank;
        const bool starts = i == n_dec || b != reqs[i - 1].bank;
        bool taken = false;
        for (uint32_t j = 0; starts && j < i; j++) taken |= reqs[j].bank == b;
        if (starts && n_pf < PULSAR_FUSED_PF_MAX) first[n_pf] = i;
        n_pf += starts ? 1u : 0u;
        const char *why = n_pf > shape->n_pf                                       ? "starts a run past the shape's"
                          : taken                                                 ? "is on a bank the step already carries"
                          : !starts && reqs[i].pos != reqs[i - 1].pos + 1         ? "does not follow its run"
                          : !pulsar_session_token_is_id(s, reqs[i].token, NULL, 0) ? "is not a token id"
                          : vis && vis->is_sentinel(e, reqs[i].token)               ? "is an image block's (a sync's)"
                                                                                  : NULL;
        if (why) {
            if (err) snprintf(err, errlen, "%s: fused prompt row %u (bank %u pos %d) %s", ARCH(e), i, b, reqs[i].pos,
                              why);
            return 1;
        }
        tok[i] = reqs[i].token;
        pos[i] = reqs[i].pos;
        bk[i] = (int32_t)b;
    }
    if (n_pf != shape->n_pf) {
        if (err) snprintf(err, errlen, "%s: fused step: %u prompt runs, the shape says %u", ARCH(e), n_pf, shape->n_pf);
        return 1;
    }
    first[n_pf] = n_rows;
    /* each run at its bank's next position or from 0 (a run from 0 must not inherit a dead conversation's image
     * records: its rope would read them) */
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank, m = first[r + 1] - first[r];
        const int32_t p0 = reqs[first[r]].pos;
        const pulsar_image_identity *img = b < ss->n_banks ? pulsar_session_bank_images(s, b, ss->live_bank) : NULL;
        const char *why = b >= ss->n_banks                            ? "is outside the pool"
                          : p0 != 0 && (uint32_t)p0 != ss->bank_pos[b] ? "holds a different position"
                          : (uint64_t)p0 + m > ss->ctx                 ? "has no room"
                          : p0 == 0 && img && img->n                   ? "still holds image blocks (invalidate it first)"
                                                                       : NULL;
        if (why) {
            if (err) snprintf(err, errlen, "%s: fused prompt run %u on bank %u at %d + %u rows: the bank %s", ARCH(e),
                              r, b, p0, m, why);
            return 1;
        }
    }
    /* a run from 0 starts its bank over, as a cold sync does; whether each run continues its bank's prefill-only
     * history is read before the step moves it */
    bool canonical[PULSAR_FUSED_PF_MAX];
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank;
        if (reqs[first[r]].pos == 0 && !pulsar_state_reset_bank(ss, b)) {
            if (err) snprintf(err, errlen, "%s: fused step: bank %u could not be reset", ARCH(e), b);
            return -1;
        }
        canonical[r] = ss->prefill_pos[b] == (uint32_t)reqs[first[r]].pos && !ss->frontier_stale[b];
    }
    pulsar_driver_heads hl = pulsar_driver_heads_span(0, n_dec);
    for (uint32_t r = 0; r < n_pf; r++)
        if (shape->head_last[r]) hl.row[hl.n++] = first[r + 1] - 1u;
    bool ok = driver_forward(s, PULSAR_STEP_PREFILL, tok.data(), pos.data(), bk.data(), n_rows, hl, logits,
                                    n_dec);
    for (uint32_t r = 0; ok && ops->absorb && r < n_pf; r++)   /* each run's drafter rows, as its sync chunk's absorb */
        ok = ops->absorb(s, PULSAR_STEP_PREFILL, tok.data() + first[r], pos.data() + first[r], bk.data() + first[r],
                         first[r + 1] - first[r], first[r], false);
    if (!ok) {
        if (err) snprintf(err, errlen, "%s: the fused step refused (see the log for the op)", ARCH(e));
        return -1;
    }
    for (uint32_t r = 0; r < n_pf; r++) {
        const uint32_t b = reqs[first[r]].bank, end = (uint32_t)reqs[first[r]].pos + (first[r + 1] - first[r]);
        pulsar_state_set_pos(ss, b, end);
        if (canonical[r]) ss->prefill_pos[b] = end;
        if (b == ss->live_bank) {   /* the live bank moved past the host view */
            s->checkpoint_valid = false;
            s->logits_stale = true;
        }
    }
    if (out_n_rows) *out_n_rows = hl.n;
    return 0;
}

/* ---- the speculation round's verify target (spec_ops.h; L272 P1 S3) ------------------------------------------------
 *   verify   the trunk runs [base, d_1 .. d_k] at p .. p + k as ONE PREFILL step of the bank with every row headed
 *            and the declared state's capture armed (the batched lane reaches the same step through decode_mixed
 *            with heads on every row).  k + 1 <= PULSAR_DRIVER_HEADS_MAX, so every kernel takes its decode arm (in 16-row
 *            blocks past 16, L288 C1) and each row's logits are the bytes a one-token decode would give.
 *   commit   the bank's declared state is set to the last kept row's (pulsar_state_commit), its position counter
 *            moves, the logits are the kept row's.
 *   cut      the L155 trim: a recurrent state has no rewind, so the bank's view is invalidated and its next sync
 *            prefills it (a grid checkpoint at or below the cut would do). */

/* The single lane's verify: the trunk over the round's rows, every row headed into the driver's slab (the session's
 * step rows: the walk reads them on the device, L302). */
static bool driver_verify_single(pulsar_session *s, pulsar_spec_round *r, char *err, size_t errlen) {
    const uint32_t R = r->n_batch;
    if (R > PULSAR_DRIVER_HEADS_MAX) {
        snprintf(err, errlen, "%s: a verify step of %u rows is outside the capture", ARCH(s->engine), R);
        return false;
    }
    /* the batched lane's verify (decode_mixed, every row headed) over this round's rows alone */
    pulsar_multiseq_req reqs[PULSAR_DRIVER_HEADS_MAX];
    pulsar_spec_round_fill_reqs(r, s->states->live_bank, reqs);
    return s->engine->family->session->decode_mixed(s, reqs, R, NULL, 0, NULL, PULSAR_MSEQ_HEAD_ALL_ROWS, err,
                                                    errlen) == 0;
}

/* The walk kept `commit` drafts: the installed bank's run of the last verify back to its last kept row (rows
 * 1 + commit .. of the run undone; the commit finds the bank's run in the capture, wherever it sat in the step),
 * the bank's position counter there, the kept row's logits fresh. */
static bool driver_spec_commit(pulsar_session *s, pulsar_spec_round *r, uint32_t commit, uint32_t) {
    pulsar_state_set *ss = s->states;
    pulsar_multiseq_req reqs[PULSAR_DRIVER_HEADS_MAX];
    int32_t tok[PULSAR_DRIVER_HEADS_MAX], pos[PULSAR_DRIVER_HEADS_MAX], bank[PULSAR_DRIVER_HEADS_MAX];
    if (r->n_batch > PULSAR_DRIVER_HEADS_MAX) return false;
    pulsar_spec_round_fill_reqs(r, ss->live_bank, reqs);
    if (!driver_verify_rows(s, reqs, r->n_batch, tok, pos, bank, NULL, 0)) return false;
    const uint32_t keep = 1u + commit;
    if (!pulsar_state_commit(ss, ss->live_bank, tok, keep)) return false;
    pulsar_state_set_pos(ss, ss->live_bank, (uint32_t)r->saved_len + keep);
    s->logits_stale = false;
    return true;
}

static void driver_spec_cut(pulsar_session *s, int) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->logits_stale = true;
}

const pulsar_spec_target_ops PULSAR_DRIVER_SPEC_TARGET = {
    /* .snapshot      = */ NULL,   /* the verify step captures its own undo state (the declared state's capture) */
    /* .restore       = */ NULL,
    /* .release       = */ NULL,
    /* .arm_verify    = */ NULL,   /* a heads-on-every-row decode_mixed IS the verify */
    /* .verify_single = */ driver_verify_single,
    /* .commit        = */ driver_spec_commit,
    /* .cut           = */ driver_spec_cut,
};

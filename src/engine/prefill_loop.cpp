/* prefill_loop.cpp -- L272 P2: the family-neutral prefill loop.
 *
 * A sync's prefill is the same walk for every family: ONE cut rule (pulsar_prefill_plan_next_end), a grid
 * checkpoint wherever a chunk lands on the grid in a prefill's state (P13: pulsar_ckpt_landed, the one capture
 * rule), the session's view advanced as each chunk lands, the progress hooks told, and the cancel hook polled at
 * every chunk boundary where a resume is exact -- which is how the server yields a long prefill to other slots and
 * honours a client that hung up (gen_prefill_cancel_cb).  The family supplies only the chunk's forward and its
 * prefill shape (pulsar_sync_ops::prefill_shape).  Before L272 Qwen's prefill had its own loop with no progress and
 * no cancel; before L284 DeepSeek's sync had its own planner (imatrix.cpp) with the same rules in another order.
 *
 * The walk (pulsar_prefill_walk_run) is shared with DeepSeek's session-less range prefill (the imatrix collector,
 * the gates' classic suffix): the cut is the plan's, the effects are each caller's hooks.  The events are
 * prefill_chunk / prefill_display at the start and after each chunk. */
#include "pulsar_engine_internal.h"

void pulsar_prefill_plan_init(pulsar_prefill_plan *p, const pulsar_prefill_shape *shape, uint32_t start,
                              const pulsar_ckpt_store *store, const pulsar_engine *e, const pulsar_tokens *prompt,
                              const pulsar_image_ref *images, int n_images) {
    p->grid_cap = shape->cap;
    p->chunk_cap = start != 0u && shape->resumed_cap < shape->cap ? shape->resumed_cap : shape->cap;
    p->align = shape->align;
    p->store = store;
    p->e = e;
    p->prompt = prompt;
    p->images = n_images > 0 ? images : NULL;
    p->n_images = n_images > 0 ? n_images : 0;
    p->endpoint = 0u;
}

uint32_t pulsar_prompt_endpoint(pulsar_engine *e, const pulsar_ckpt_store *store, const pulsar_tokens *prompt) {
    if (!e || !store || !store->ops || !store->ops->split_invariant || !prompt || prompt->len < 2) return 0u;
    pulsar_turn_markers m;
    if (!pulsar_chat_turn_markers(e, &m)) return 0u;
    for (int i = prompt->len - 1; i > 0; i--)
        if (pulsar_turn_marker_at(&m, prompt->v, prompt->len, i) != 0)
            return (uint32_t)i >= store->ops->min_checkpoint(store->state) ? (uint32_t)i : 0u;
    return 0u;
}

/* L268: the image block a cut at `cut` would split -- [bs, be) with bs < cut < be -- or false.  The FIRST such
 * block in request order (the cut rule adjusts block by block in that order). */
static bool prefill_plan_block_across(const pulsar_prefill_plan *p, uint32_t cut, uint32_t *bs, uint32_t *be) {
    for (int i = 0; i < p->n_images; i++) {
        int len = 0;
        const int st = p->images[i].start_pos;
        if (!pulsar_image_block_extent(p->e, p->prompt->v, p->prompt->len, st, &len)) continue;
        if ((uint32_t)st < cut && (uint32_t)(st + len) > cut) {
            *bs = (uint32_t)st;
            *be = (uint32_t)(st + len);
            return true;
        }
    }
    return false;
}

uint32_t pulsar_prefill_plan_next_end(const pulsar_prefill_plan *p, uint32_t pos0, uint32_t end) {
    const uint32_t remaining = end - pos0;
    /* snap to the absolute grid of the cap after any unaligned start: a resume (start != 0) lands on the cold
     * prefill's boundaries, and so does the chunk after an image cut below -- on the cold pass too, which is what
     * keeps a resumed image prefill the cold one's chunk for chunk */
    uint32_t local_cap = p->chunk_cap;
    const uint32_t to_boundary = pulsar_prefill_to_boundary(pos0, p->grid_cap);
    if (to_boundary < local_cap) local_cap = to_boundary;
    uint32_t chunk = remaining < local_cap ? remaining : local_cap;
    /* every NON-final chunk end on the alignment (DeepSeek: the compress ratios' LCM -- one unaligned boundary makes
     * every later chunk's start unaligned and each takes the per-token compressor path; 1 = none).  The final chunk
     * keeps its exact remainder; an unaligned START pays for its first chunk only, which still ends aligned. */
    if (chunk < remaining && p->align > 1u) {
        const uint32_t aligned_end = (pos0 + chunk) / p->align * p->align;
        if (aligned_end > pos0) chunk = aligned_end - pos0;
    }
    /* L268: an image block is merged whole by the chunk that owns it, so a cut never splits one: it moves to the
     * block's start, or -- the block starts this chunk (it fits one: pulsar_image_spans_fit) -- past its end.
     * Positions only, so the cold pass and any resume over the same prompt cut alike. */
    for (int i = 0; i < p->n_images; i++) {
        int len = 0;
        const int st = p->images[i].start_pos;
        if (!pulsar_image_block_extent(p->e, p->prompt->v, p->prompt->len, st, &len)) continue;
        const uint32_t bs = (uint32_t)st, be = (uint32_t)(st + len), ce = pos0 + chunk;
        if (bs < ce && ce < be) chunk = bs > pos0 ? bs - pos0 : be - pos0;
    }
    /* L264 / P13: the final chunk stops at the last grid point inside it (pulsar_ckpt_final_cut), so the prefill
     * leaves the checkpoint the next turn resumes from -- not when that point falls inside an image block */
    if (pos0 + chunk == end) {
        const uint32_t cut = pulsar_ckpt_final_cut(p->store, pos0, end);
        uint32_t bs = 0, be = 0;
        if (!prefill_plan_block_across(p, cut, &bs, &be)) chunk = cut - pos0;
    }
    /* L311: and a chunk stops at the prompt's message endpoint inside it, so the walk lands there and records it (a
     * split-invariant family's only: any cut is its cold prefill's bytes) -- after the grid cut, which the walk
     * still makes (the disk segments and the grid resume end there) */
    if (p->endpoint > pos0 && p->endpoint < pos0 + chunk) {
        uint32_t bs = 0, be = 0;
        if (!prefill_plan_block_across(p, p->endpoint, &bs, &be)) chunk = p->endpoint - pos0;
    }
    return pos0 + chunk;
}

int pulsar_prefill_walk_run(const pulsar_prefill_walk *w, uint32_t start, uint32_t end) {
    for (uint32_t pos0 = start; pos0 < end;) {
        if (w->stop && w->stop(w->ud)) {
            (void)pulsar_gpu_synchronize();   /* L188: drain before handing the stream to whatever runs next */
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
        const uint32_t chunk_end = pulsar_prefill_plan_next_end(w->plan, pos0, end);
        if (chunk_end <= pos0 || chunk_end > end) {
            fprintf(stderr, "pulsar: prefill walk: a chunk from %u cut at %u (prompt ends at %u)\n", pos0, chunk_end, end);
            return 1;
        }
        if (!w->chunk(w->ud, pos0, chunk_end - pos0, chunk_end == end)) return 1;
        if (w->landed && !w->landed(w->ud, chunk_end)) return 1;
        pos0 = chunk_end;
        if (pos0 < end && w->stop && w->stop(w->ud)) {
            (void)pulsar_gpu_synchronize();
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    }
    return 0;
}

/* ---- L329 (b2): the prompt read-ahead ---------------------------------------
 *
 * A Qwen prefill chunk's PLE gather (16 rows a token, 65,536 for a 4096-token chunk, from the ~96 GB ple-l1.rows on
 * the NVMe) is issued at the chunk's embed and waited at layer 1; layer 0 hides ~60 ms of it.  The table does not fit
 * the page cache beside the model, so a production prefill's rows are mostly cold, and the cold remainder sat on the
 * critical path: +94..152 ms at 4k, +1.09..1.19 s at 32k (~5 % of TTFT; L329 window #476).  But the prompt is
 * tokenized whole before the prefill starts, so the NEXT chunk's rows are known while this one computes: read them
 * then, and that chunk's gather finds them cached.
 *
 * The rule is the core's, for every prefill path that gathers -- this walk (solo prefill, a cold-store or L329 (a)
 * shared-prefix cut, which reads ahead along the whole request: pulsar_session_set_prefill_ahead) and the fused
 * lane's prompt runs (the server calls pulsar_session_prompt_read_ahead per round): one chunk past the rows in
 * flight, one record per bank so a run that advances by less than a chunk queues each row once.  The family supplies
 * the rows (pulsar_driver_ops::read_ahead) and the pool reads them on its background class, below every gather
 * (engram.cpp).  A hint: the gather reads every byte itself, so no byte changes with it on or off.
 *
 * Bounded (window #477/#478: systemd-oomd's pressure kill took the job's whole unit during gates-dev): it runs only
 * for a prompt a server is serving (the walk's license is pulsar_session_set_prefill_ahead; nothing else -- a gate,
 * the CLI, imatrix -- ever reads ahead), the pool holds at most PULSAR_ENGRAM_AHEAD_PENDING_MAX unread rows (~1 GiB of
 * cache), and an arm is skipped while MemAvailable is under PULSAR_READ_AHEAD_MEM_HEADROOM plus that cap. */

static uint64_t read_ahead_hash(const int *v, uint32_t from, uint32_t to) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = from >= 2u ? from - 2u : 0u; i < to; i++) {
        h ^= (uint32_t)v[i];
        h *= 1099511628211ull;
    }
    return h ^ ((uint64_t)from << 32) ^ to;
}

bool pulsar_read_ahead_plan(const pulsar_read_ahead_mark *mark, const int *prompt, uint32_t len, uint32_t run_end,
                            uint32_t window, uint32_t *from, uint32_t *to, bool *drop, pulsar_read_ahead_mark *next) {
    *from = *to = run_end;
    *drop = false;
    if (!prompt || window == 0u || run_end >= len) return false;
    const uint32_t end = len - run_end < window ? len : run_end + window;
    /* the record holds rows still ahead of the run for these very tokens: queue only past it */
    const bool live = mark && mark->from < mark->to && mark->from <= run_end && run_end < mark->to && mark->to <= len &&
                      read_ahead_hash(prompt, mark->from, mark->to) == mark->hash;
    *drop = mark && mark->from < mark->to && !live;
    const uint32_t f = live ? mark->to : run_end;
    if (f >= end || (end < len && end - f < window / 2u)) return false;
    *from = f;
    *to = end;
    next->from = run_end;
    next->to = end;
    next->hash = read_ahead_hash(prompt, run_end, end);
    return true;
}

bool pulsar_read_ahead_admit(uint64_t mem_available) {
    /* the headroom plus the most the pool can hold unread, a page a row (a cold row ~ one page of cache) */
    return mem_available != 0 && mem_available > PULSAR_READ_AHEAD_MEM_HEADROOM + PULSAR_ENGRAM_AHEAD_PENDING_MAX * 4096ull;
}

/* MemAvailable from /proc/meminfo in bytes, 0 when unreadable (the admit then fails closed).  Read once per arm --
 * a chunk, or a fused round that queues -- never per row or per step. */
uint64_t pulsar_read_ahead_mem_available(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof(line), fp))
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    fclose(fp);
    return (uint64_t)kib * 1024ull;
}

static uint64_t read_ahead_tag(const pulsar_session *s, uint32_t bank) {
    return ((uint64_t)(uintptr_t)s << 16) ^ bank;   /* banks < 2^16; one tag per session and bank */
}

uint32_t pulsar_session_read_ahead(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, uint32_t run_end) {
    const pulsar_driver_ops *ops = s ? s->engine->family->driver : NULL;
    if (!ops || !ops->read_ahead || !prompt || bank >= 0x10000u) return 0;
    if (bank >= s->n_ahead) {
        const uint32_t n = bank + 1u;
        s->ahead = (pulsar_read_ahead_mark *)xrealloc(s->ahead, (size_t)n * sizeof(*s->ahead));
        memset(s->ahead + s->n_ahead, 0, (size_t)(n - s->n_ahead) * sizeof(*s->ahead));
        s->n_ahead = n;
    }
    pulsar_read_ahead_mark *m = &s->ahead[bank], next = {};
    uint32_t from = 0, to = 0;
    bool drop = false;
    const bool go = pulsar_read_ahead_plan(m, prompt->v, (uint32_t)prompt->len, run_end, s->prefill_cap, &from, &to,
                                           &drop, &next);
    const uint64_t tag = read_ahead_tag(s, bank);
    if (drop) {
        (void)pulsar_engram_read_ahead_cancel(ops->read_ahead_pool(s->engine), tag);
        *m = {};
    }
    if (!go) return 0;
    if (!pulsar_read_ahead_admit(pulsar_read_ahead_mem_available())) {
        pulsar_engram_read_ahead_skipped(ops->read_ahead_pool(s->engine));
        static bool said = false;   /* rule 5: the stand-down, once */
        if (!said) {
            said = true;
            fprintf(stderr, "pulsar: L329 (b2) read-ahead stood down: MemAvailable under %llu GiB + the arm's pages "
                            "(counted: pulsar:ple_read_ahead_rows_total{outcome=\"skipped\"})\n",
                    (unsigned long long)(PULSAR_READ_AHEAD_MEM_HEADROOM >> 30));
        }
        return 0;
    }
    const uint32_t rows = ops->read_ahead(s->engine, tag, prompt->v, from, to);
    if (rows == 0) return 0;   /* off (said at load), at the pool's cap (counted), or a token the gather refuses */
    *m = next;
    static bool announced = false;   /* rule 5: the path taken, once */
    if (!announced) {
        announced = true;
        fprintf(stderr, "pulsar: L329 (b2) read-ahead ran: bank %u, tokens [%u, %u) of %d (%u rows) queued for the "
                        "background readers while the rows to %u are in flight\n",
                bank, from, to, prompt->len, rows, run_end);
    }
    return to - from;
}

void pulsar_session_read_ahead_cancel(pulsar_session *s, uint32_t bank) {
    const pulsar_driver_ops *ops = s ? s->engine->family->driver : NULL;
    if (!ops || !ops->read_ahead || bank >= s->n_ahead || s->ahead[bank].from == s->ahead[bank].to) return;
    (void)pulsar_engram_read_ahead_cancel(ops->read_ahead_pool(s->engine), read_ahead_tag(s, bank));
    s->ahead[bank] = {};
}

static void prefill_loop_progress(pulsar_session *s, int current, int total) {
    if (s->progress) s->progress(s->progress_ud, "prefill_chunk", current, total);
    if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_display", current, total);
}

/* The session loop's hooks over the walk: the view advanced and the shared capture rule applied as each chunk
 * lands, the session's cancel hook as the stop -- polled only where a resume is exact. */
struct prefill_loop_ctx {
    pulsar_session *s;
    const pulsar_tokens *prompt;
    uint32_t bank;
    pulsar_ckpt_store *ckpt;
    pulsar_prefill_chunk_fn chunk;
    void *ud;
    uint32_t endpoint;   ///< L311: the plan's message endpoint the walk records (0 = none)
    /** L281 (b): the walk stands where a resume is exact -- its start, or a chunk end the model's prefill
     *  reproduces (anywhere when split_invariant, else on the resume grid; a chunk never ends inside an image
     *  block).  A stop is honoured only there: an interrupt at an image block's off-grid end would resume from the
     *  grid point below it, inside the block, i.e. from 0. */
    bool at_resume_point;
};

/* L327: each chunk is also an observation of what a prompt chunk costs (verify_curve.cpp): its wall, the stream
 * drained first by the step's own drain (end_commands: a no-op wait where the family's chunk already ended in one, as
 * the step driver's does; a flag it reads fails the chunk, as at any step's end) -- not a chunk of a sync that merges
 * images, whose encoder the fit does not model. */
static bool prefill_loop_chunk(void *ud, uint32_t pos0, uint32_t rows, bool last) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    /* L329 (b2): the next chunk's rows, read while this one computes (queued first; this chunk's own gather, issued
     * at its embed, goes ahead of them) -- for a SERVED prompt only: the one a server named for this sync
     * (pulsar_session_set_prefill_ahead); a sync nobody licensed (a gate, the CLI, imatrix) passes NULL, reads nothing */
    (void)pulsar_session_read_ahead(c->s, c->bank, c->s->prefill_ahead, pos0 + rows);
    const double t0 = now_sec();
    if (!c->chunk(c->s, c->prompt, pos0, rows, last, c->ud) || pulsar_gpu_end_commands() == 0) return false;
    if (c->s->sync_n_images == 0)
        pulsar_engine_prefill_chunk_observe(c->s->engine, rows, pos0, (now_sec() - t0) * 1e3);
    return true;
}

static bool prefill_loop_landed(void *ud, uint32_t chunk_end) {
    prefill_loop_ctx *c = (prefill_loop_ctx *)ud;
    pulsar_session *s = c->s;
    for (int i = s->checkpoint.len; i < (int)chunk_end; i++) token_vec_push(&s->checkpoint, c->prompt->v[i]);
    s->checkpoint_valid = true;
    if (!pulsar_ckpt_landed(c->ckpt, c->bank, chunk_end)) return false;
    /* L311: the walk stands at the prompt's message endpoint -- record it (said when it cannot be) */
    if (chunk_end == c->endpoint)
        (void)pulsar_ckpt_record_capture(c->ckpt, c->bank, PULSAR_CKPT_REC_ENDPOINT, chunk_end, c->prompt->v, NULL, 0u);
    c->at_resume_point = c->ckpt->ops->split_invariant || chunk_end % c->ckpt->ops->resume_grid == 0u;
    prefill_loop_progress(s, (int)chunk_end, c->prompt->len);
    return true;
}

static bool prefill_loop_stop(void *ud) {
    const prefill_loop_ctx *c = (const prefill_loop_ctx *)ud;
    return c->at_resume_point && pulsar_session_cancelled(c->s);
}

int pulsar_prefill_loop(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start, pulsar_prefill_chunk_fn chunk,
                        void *ud) {
    const uint32_t end = (uint32_t)prompt->len;
    pulsar_prefill_shape shape = {};
    s->engine->family->session->sync->prefill_shape(s, &shape);
    pulsar_ckpt_store *ckpt = pulsar_session_kv_store(s);
    if (shape.cap == 0u || shape.align == 0u || start >= end || !ckpt || !ckpt->ops) {
        fprintf(stderr, "pulsar: prefill loop: nothing to prefill (start %u, end %u, cap %u)\n", start, end, shape.cap);
        return 1;
    }
    pulsar_prefill_plan plan;
    pulsar_prefill_plan_init(&plan, &shape, start, ckpt, s->engine, prompt, s->sync_images, s->sync_n_images);
    /* L311: a text history's message endpoint past the walk's start is cut at and recorded (the records keep no
     * image identity, as the prompt end below) */
    if (s->sync_n_images == 0 && s->live_images.n == 0) {
        const uint32_t E = pulsar_prompt_endpoint(s->engine, ckpt, prompt);
        plan.endpoint = E > start ? E : 0u;
    }
    if (plan.n_images > 0) {
        /* every block must fit one chunk of THIS walk (a resume's chunks may be narrower than the cold pass's) */
        char verr[384];
        if (!pulsar_image_spans_fit(s->engine, prompt->v, prompt->len, plan.images, plan.n_images, plan.chunk_cap,
                                    NULL, verr, sizeof(verr))) {
            fprintf(stderr, "pulsar: %s\n", verr);
            return 1;
        }
        /* L268: a start inside an image block would re-evaluate its merged rows without their image */
        uint32_t bs = 0, be = 0;
        if (start > 0 && prefill_plan_block_across(&plan, start, &bs, &be)) {
            fprintf(stderr, "pulsar: prefill loop: start %u is inside image block [%u, %u) -- refusing\n", start, bs, be);
            return 1;
        }
    }
    /* the view below `start` is the session's already; it grows with the prompt from here */
    s->checkpoint.len = 0;
    for (uint32_t i = 0; i < start; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
    s->checkpoint_valid = start > 0u;
    prefill_loop_progress(s, (int)start, prompt->len);
    prefill_loop_ctx c = { s, prompt, pulsar_session_live_bank(s), ckpt, chunk, ud, plan.endpoint, true };
    const pulsar_prefill_walk w = { &plan, prefill_loop_chunk, prefill_loop_landed, prefill_loop_stop, &c };
    const int rc = pulsar_prefill_walk_run(&w, start, end);
    /* the logits are the prompt's next-token row only when the final chunk ran (a stop leaves the view
     * at a boundary, a failure leaves it at the last chunk that landed) */
    s->logits_stale = rc != 0;
    if (rc == 1) s->checkpoint_valid = false;
    /* L289: the prompt end is recorded -- the state, open rows and logits this prefill left at `end` -- so the
     * same prompt resumes here without evaluating a row.  A text history only: a KV holding an image block is
     * matched by its image identity, which the record does not keep (such a prompt resumes from the grid). */
    if (rc == 0 && s->sync_n_images == 0 && s->live_images.n == 0) {
        const float *live = s->live_host();   /* L302: the record keeps the row on the host */
        if (live)
            (void)pulsar_ckpt_record_capture(ckpt, c.bank, PULSAR_CKPT_REC_END, end, prompt->v, live,
                                             (uint32_t)pulsar_engine_logits_width(s->engine));
    }
    return rc;
}

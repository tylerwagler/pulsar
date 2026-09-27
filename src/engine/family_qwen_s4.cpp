/* family_qwen_s4.cpp -- the Qwen4-exp ops stream S4 owns (L251): embed, the PLE
 * injection, the gated-residual read / write, the MoE block and the head
 * (mixer + lm_head); their scratch; and at load, the admission of every tensor
 * those ops read and the PLE row file.
 *
 * Each op resolves its weights to device pointers (the engine's model-range
 * cache) and calls the S4 launchers in src/cuda/pulsar_cuda_qwen.h -- the same
 * launchers tests/qwen_s4_gate.cu grades against the double references and
 * tools/qwen/s4_xcheck.py against transformers' modules.  Formats (family_qwen.h
 * section 4): streams bf16, x bf16 + its armed E4M3 slot, y f32.
 *
 * The PLE's n-gram rows are hashed on the host from the step's tokens (the ids
 * are known before the forward) and read from the row file by the Engram pread
 * pool: embed hashes and ISSUES the gather, the PLE op at layer 1 waits for it,
 * so the reads overlap layer 0. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "qwen_ngram.h"
#include "cuda/pulsar_cuda_qwen.h"

#include <sys/stat.h>

struct pulsar_qwen_ple_io {
    pulsar_engram_table table;
    pulsar_engram_io *io;
    pulsar_qwen_ngram_layout layout;
    unsigned char *rows;            ///< host [cap][16][320]: the gathered rows, head order
    uint64_t *ids;                  ///< host [cap][16]
    uint32_t cap;                   ///< tokens the host buffers hold
    pulsar_engram_gather *pending;  ///< the gather embed issued, waited on by the PLE op
    uint32_t pending_n;             ///< its tokens
};

namespace {

constexpr uint64_t kGrInjStride = (uint64_t)PULSAR_QWEN_HC * sizeof(float);   /* inj per row */

const pulsar_qwen_layer_weights &layer_w(const pulsar_qwen_step *st, uint32_t il) { return st->w->layer[il]; }

const void *wptr(const pulsar_qwen_step *st, const pulsar_tensor *t, const char *what) {
    const void *p = pulsar_qwen_weight_ptr(tensor_map_base(st->model, t), t->abs_offset, t->bytes, what);
    if (!p) fprintf(stderr, "pulsar: %s: no device copy of %.*s (%s) -- refusing\n", PULSAR_QWEN_ARCH,
                    (int)t->name.len, t->name.ptr, what);
    return p;
}

void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

/* ---- scratch layouts: one function each, used for sizing AND carving -------- */

uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }

struct ple_scratch { uint64_t emb, row_seq, row_j, seq_first, seq_rows, seq_bank, ws, ws_bytes, total; };
ple_scratch ple_layout(uint32_t rows) {
    ple_scratch p{};
    uint64_t o = 0;
    p.emb = o;       o += a256((uint64_t)rows * PULSAR_QWEN_HIDDEN * 2u);
    p.row_seq = o;   o += a256((uint64_t)rows * 4u);
    p.row_j = o;     o += a256((uint64_t)rows * 4u);
    p.seq_first = o; o += a256((uint64_t)rows * 4u);
    p.seq_rows = o;  o += a256((uint64_t)rows * 4u);
    p.seq_bank = o;  o += a256((uint64_t)rows * 4u);
    p.ws = o;
    p.ws_bytes = pulsar_qwen_ple_workspace_bytes((int)rows);
    p.total = o + p.ws_bytes;
    return p;
}
struct gr_scratch { uint64_t ws, ws_bytes, inj[2], total; };
gr_scratch gr_layout(uint32_t rows) {
    gr_scratch g{};
    g.ws = 0;
    g.ws_bytes = a256(pulsar_qwen_gr_workspace_bytes((int)rows));
    g.inj[0] = g.ws_bytes;
    g.inj[1] = g.inj[0] + a256((uint64_t)rows * kGrInjStride);
    g.total = g.inj[1] + a256((uint64_t)rows * kGrInjStride);
    return g;
}
struct moe_scratch { uint64_t nf, ws, ws_bytes, total; };
moe_scratch moe_layout(uint32_t rows) {
    moe_scratch m{};
    m.nf = 0;
    m.ws = 256;
    m.ws_bytes = pulsar_qwen_moe_workspace_bytes((int)rows);
    m.total = m.ws + m.ws_bytes;
    return m;
}
struct head_scratch { uint64_t ws, ws_bytes, xkey, total; };
head_scratch head_layout(void) {
    head_scratch h{};
    h.ws = 0;
    h.ws_bytes = a256(pulsar_qwen_gr_workspace_bytes((int)PULSAR_QWEN_HEAD_ROWS_MAX));
    h.xkey = h.ws_bytes;   /* the key of the head's bf16 activation slot: [16][2560] f32-sized */
    h.total = h.xkey + (uint64_t)PULSAR_QWEN_HEAD_ROWS_MAX * PULSAR_QWEN_HIDDEN * sizeof(float);
    return h;
}

/* A view of an op's scratch at [off, off + bytes) -- a tensor key for the
 * activation-cache calls and a write target; freed by the caller. */
pulsar_gpu_tensor *scratch_view(const pulsar_qwen_step *st, pulsar_qwen_op_id op, uint64_t off, uint64_t bytes) {
    pulsar_gpu_tensor *s = st->st->scratch[op];
    return s ? pulsar_gpu_tensor_view(s, off, bytes) : NULL;
}

bool fail(const char *what) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the step\n", PULSAR_QWEN_ARCH, what);
    return false;
}

/* The GR weights of one site, as the launcher takes them. */
bool gr_dev(const pulsar_qwen_step *st, const pulsar_qwen_gr_weights &g, pulsar_qwen_gr_dev *d) {
    const uint64_t R = st->shape->n_hc_lowrank, HC = pulsar_qwen_hc_dim(st->shape);
    const uint8_t *down = (const uint8_t *)wptr(st, g.mix_down, "qwen GR W_down");
    const uint8_t *up = (const uint8_t *)wptr(st, g.mix_up, "qwen GR W_up");
    d->norm_w = (const uint16_t *)wptr(st, g.hc_norm, "qwen GR hc_norm");
    d->inject = g.inject ? (const uint16_t *)wptr(st, g.inject, "qwen GR inject") : NULL;
    if (!down || !up || !d->norm_w || (g.inject && !d->inject)) return false;
    /* mxfp8_lt: E4M3 [out][in] then the swizzled E8M0 plane; bf16: [out][in] (the
     * mixer in the graded recipe).  Admission made both the same format. */
    const bool w8 = g.mix_down->type == PULSAR_TENSOR_MXFP8_LT;
    d->down = {down, w8 ? down + R * HC : NULL, (int)R, (int)HC};
    d->up = {up, w8 ? up + HC * R : NULL, (int)HC, (int)R};
    return true;
}

bool linear_dev(const pulsar_qwen_step *st, const pulsar_tensor *t, int in, int out, const char *what,
                pulsar_qwen_linear *l) {
    l->w = wptr(st, t, what);
    l->k2 = exl3_type_k2(t->type);
    l->in = in;
    l->out = out;
    return l->w != NULL;
}

} // namespace

/* ======================================================================== */
/* load                                                                      */

static bool admit(const pulsar_tensor *t, bool ok, const char *want) {
    if (ok) return true;
    fprintf(stderr, "pulsar: %s: tensor %.*s is %s; the S4 op that reads it takes %s -- refusing\n",
            PULSAR_QWEN_ARCH, (int)t->name.len, t->name.ptr, tensor_type_name(t->type), want);
    return false;
}
static bool admit_bf16(const pulsar_tensor *t) { return admit(t, t->type == PULSAR_TENSOR_BF16, "bf16"); }
static bool admit_mx8(const pulsar_tensor *t) { return admit(t, t->type == PULSAR_TENSOR_MXFP8_LT, "mxfp8_lt"); }
static bool admit_exl3(const pulsar_tensor *t, int arm, const char *want) {
    const int k2 = exl3_type_k2(t->type);
    return admit(t, k2 != 0 && exl3_arm_has_rate(arm, k2), want);
}
/* the low-rank pair: MXFP8 (the per-layer sites) or BF16 (the mixer, as the
 * graded recipe stores it) -- both arms of the GR read, one format per pair */
static bool admit_gr(const pulsar_qwen_gr_weights &g) {
    const bool pair_bf16 = g.mix_down->type == PULSAR_TENSOR_BF16 && g.mix_up->type == PULSAR_TENSOR_BF16;
    bool ok = admit_bf16(g.hc_norm);
    if (!pair_bf16) ok &= admit_mx8(g.mix_down) & admit_mx8(g.mix_up);
    if (g.inject) ok &= admit_bf16(g.inject);
    return ok;
}

bool pulsar_qwen_s4_load(pulsar_engine *e, const pulsar_engine_options *opt) {
    pulsar_qwen_weights *w = e->qwen_weights;
    const pulsar_qwen_shape *s = &g_qwen_shape;
    bool ok = admit_bf16(w->token_embd) & admit_bf16(w->output) & admit_gr(w->mixer);
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        const pulsar_qwen_layer_weights &L = w->layer[il];
        ok &= admit_gr(L.gr_attn) & admit_gr(L.gr_mlp);
        ok &= admit_bf16(L.moe_router) & admit_bf16(L.sh_gate_scalar);
        ok &= admit_exl3(L.moe_gate_up, EXL3_ARM_GATE_UP_FUSED, "exl3m_k4 / exl3m_k5 (the fused gate_up arm)");
        ok &= admit_exl3(L.moe_down, EXL3_ARM_DOWN, "an exl3m rate the routed down arm reads");
        ok &= admit_exl3(L.sh_gate, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
              admit_exl3(L.sh_up, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
              admit_exl3(L.sh_down, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads");
        if (L.ple_key) {
            ok &= admit_exl3(L.ple_key, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
                  admit_exl3(L.ple_value, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads");
            ok &= admit_bf16(L.ple_norm_key) & admit_bf16(L.ple_norm_query) & admit_bf16(L.ple_norm_conv) &
                  admit_bf16(L.ple_conv);
        }
    }
    if (!ok) return false;

    /* the PLE row file, beside the container */
    pulsar_str file = {NULL, 0};
    uint64_t n_rows = 0;
    if (!model_get_string(&e->model, "pulsar.ple_rows.file", &file) ||
        !model_get_u64_compat(&e->model, "pulsar.ple_rows.n_rows", &n_rows)) {
        fprintf(stderr, "pulsar: %s: the container names no PLE row file (pulsar.ple_rows.file / .n_rows) -- "
                        "refusing\n", PULSAR_QWEN_ARCH);
        return false;
    }
    char path[4096];
    struct stat mst;
    const char *mp = opt->model_path ? opt->model_path : "";
    if (stat(mp, &mst) == 0 && S_ISDIR(mst.st_mode)) {
        snprintf(path, sizeof(path), "%s/%.*s", mp, (int)file.len, file.ptr);
    } else {
        const char *slash = strrchr(mp, '/');
        snprintf(path, sizeof(path), "%.*s%.*s", slash ? (int)(slash - mp + 1) : 0, mp, (int)file.len, file.ptr);
    }
    pulsar_qwen_ple_io *io = (pulsar_qwen_ple_io *)xcalloc(1, sizeof(*io));
    pulsar_qwen_ngram_layout &Ly = io->layout;
    for (uint32_t i = 0; i < PULSAR_QWEN_NGRAM_ORDER; i++) Ly.mult[i] = (int64_t)w->ple_multipliers[i];
    for (uint32_t h = 0; h < PULSAR_QWEN_NGRAM_COLS; h++) {
        Ly.prime[h] = w->ple_head_vocab[h];
        Ly.offset[h] = w->ple_head_offset[h];
    }
    Ly.eos = (int32_t)s->eos_id;
    Ly.vocab = s->n_vocab;
    Ly.n_rows = n_rows;
    if (s->ngram_size != PULSAR_QWEN_NGRAM_ORDER || (s->ngram_size - 1u) * s->ngram_heads != PULSAR_QWEN_NGRAM_COLS ||
        !pulsar_qwen_ngram_layout_check(&Ly) || !pulsar_qwen_ngram_table_open(&io->table, path, s->ple_layer, &Ly)) {
        fprintf(stderr, "pulsar: %s: the PLE row file %s does not open against this model -- refusing\n",
                PULSAR_QWEN_ARCH, path);
        free(io);
        return false;
    }
    io->io = pulsar_engram_io_create(PULSAR_ENGRAM_IO_THREADS);
    if (!io->io) {
        pulsar_engram_table_close(&io->table);
        free(io);
        return false;
    }
    w->ple_io = io;
    fprintf(stderr, "pulsar: L251 qwen PLE rows: %s (%llu rows x %u B, %u per token, %u pread threads); S4 tensors "
                    "admitted (experts fused gate_up + down EXL3, dense EXL3, GR low-rank MXFP8, head bf16)\n",
            path, (unsigned long long)n_rows, PULSAR_QWEN_NGRAM_ROW_BYTES, PULSAR_QWEN_NGRAM_COLS,
            PULSAR_ENGRAM_IO_THREADS);
    return true;
}

void pulsar_qwen_s4_unload(pulsar_qwen_weights *w) {
    pulsar_qwen_ple_io *io = w ? w->ple_io : NULL;
    if (!io) return;
    if (io->pending) (void)pulsar_engram_gather_wait(io->pending);
    pulsar_engram_io_destroy(io->io);
    pulsar_engram_table_close(&io->table);
    free(io->rows);
    free(io->ids);
    free(io);
    w->ple_io = NULL;
}

uint64_t pulsar_qwen_s4_scratch_bytes(pulsar_qwen_op_id op, const pulsar_qwen_shape *, uint32_t max_rows) {
    switch (op) {
    case PULSAR_QWEN_OP_EMBED:   return (uint64_t)max_rows * sizeof(int32_t);
    case PULSAR_QWEN_OP_PLE:     return ple_layout(max_rows).total;
    case PULSAR_QWEN_OP_GR_READ: return gr_layout(max_rows).total;
    case PULSAR_QWEN_OP_MOE:     return moe_layout(max_rows).total;
    case PULSAR_QWEN_OP_HEAD:    return head_layout().total;
    default:                     return 0;   /* GR write reads GR read's inj; GDN / QSA are S2's / S3's */
    }
}

/* ======================================================================== */
/* the ops                                                                   */

bool pulsar_qwen_s4_embed(const pulsar_qwen_step *st) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    for (uint32_t r = 0; r < n; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= s->n_vocab) return fail("a token id outside the vocabulary");
    /* the MoE non-finite flag starts every step clear (read at pulsar_qwen_s4_step_end) */
    pulsar_gpu_tensor *moe_sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    const uint32_t zero = 0;
    if (!moe_sc || !pulsar_gpu_tensor_write(moe_sc, moe_layout(st->st->max_rows).nf, &zero, sizeof(zero)))
        return fail("could not clear the MoE non-finite flag");
    pulsar_gpu_tensor *tok = st->st->scratch[PULSAR_QWEN_OP_EMBED];
    const pulsar_tensor *te = st->w->token_embd;
    const uint16_t *table = (const uint16_t *)wptr(st, te, "qwen embed_tokens");
    if (!tok || !table || !pulsar_gpu_tensor_write(tok, 0, st->tokens, (uint64_t)n * sizeof(int32_t))) return false;
    if (pulsar_qwen_embed_launch(table, (const int32_t *)dptr(tok), (int)n, (int)s->n_vocab,
                                 (uint16_t *)dptr(st->st->streams), 0))
        return false;

    /* the PLE rows of these tokens: hash on the host, issue the gather now */
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (!io) return fail("no PLE row file is open");
    if (io->pending) return fail("a previous step's PLE gather was never consumed");
    if (io->cap < n) {
        free(io->rows);
        free(io->ids);
        io->rows = (unsigned char *)xmalloc((size_t)n * PULSAR_QWEN_NGRAM_COLS * PULSAR_QWEN_NGRAM_ROW_BYTES);
        io->ids = (uint64_t *)xmalloc((size_t)n * PULSAR_QWEN_NGRAM_COLS * sizeof(uint64_t));
        io->cap = n;
    }
    const uint32_t nc = s->ngram_size - 1u;
    for (uint32_t r = 0; r < n; r++) {
        int32_t *ctx_b = st->st->ngram_ctx + (size_t)st->bank[r] * nc;   /* oldest first */
        pulsar_qwen_ngram_ctx ctx = {{ctx_b[0], ctx_b[1]}};
        pulsar_qwen_ngram_rows(&io->layout, &ctx, &st->tokens[r], 1, io->ids + (size_t)r * PULSAR_QWEN_NGRAM_COLS);
        ctx_b[0] = ctx.prev[0];
        ctx_b[1] = ctx.prev[1];
    }
    io->pending = pulsar_engram_gather_start(io->io, &io->table, io->ids, n * PULSAR_QWEN_NGRAM_COLS, io->rows);
    io->pending_n = n;
    return io->pending != NULL || fail("the PLE gather did not start");
}

bool pulsar_qwen_s4_step_end(const pulsar_qwen_step *st, bool ok) {
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (io && io->pending) {                 /* a step that failed before layer 1 */
        (void)pulsar_engram_gather_wait(io->pending);
        io->pending = NULL;
    }
    if (!ok) return false;
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    uint32_t nf = 0;
    if (!sc || !pulsar_gpu_tensor_read(sc, moe_layout(st->st->max_rows).nf, &nf, sizeof(nf)))
        return fail("could not read the MoE non-finite flag");
    if (nf) {
        fprintf(stderr, "pulsar: %s: the MoE output of layer %u went non-finite (code 0x%08x) -- refusing the step\n",
                PULSAR_QWEN_ARCH, nf & 0xffffffu, nf);
        return false;
    }
    return true;
}

bool pulsar_qwen_s4_ple(const pulsar_qwen_step *st, uint32_t il) {
    const uint32_t n = st->n_rows;
    pulsar_qwen_ple_io *io = st->w->ple_io;
    if (!io || !io->pending || io->pending_n != n) return fail("the PLE op found no gather for this step");
    const int got = pulsar_engram_gather_wait(io->pending);
    io->pending = NULL;
    if (!got) return fail("the PLE row gather failed (message above)");

    const ple_scratch p = ple_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_PLE];
    if (!sc) return fail("no PLE scratch");
    /* the rows, and where each row sits: DECODE = one row per bank, PREFILL = one bank's run */
    const bool decode = st->mode == PULSAR_QWEN_STEP_DECODE;
    const uint32_t n_seq = decode ? n : 1u;
    int32_t *map = (int32_t *)xmalloc((size_t)n * 5u * sizeof(int32_t));
    int32_t *row_seq = map, *row_j = map + n, *seq_first = map + 2 * n, *seq_rows = map + 3 * n, *seq_bank = map + 4 * n;
    for (uint32_t r = 0; r < n; r++) { row_seq[r] = decode ? (int32_t)r : 0; row_j[r] = decode ? 0 : (int32_t)r; }
    for (uint32_t q = 0; q < n_seq; q++) {
        seq_first[q] = decode ? (int32_t)q : 0;
        seq_rows[q] = decode ? 1 : (int32_t)n;
        seq_bank[q] = st->bank[decode ? q : 0];
    }
    bool ok = pulsar_gpu_tensor_write(sc, p.emb, io->rows, (uint64_t)n * PULSAR_QWEN_HIDDEN * 2u) &&
              pulsar_gpu_tensor_write(sc, p.row_seq, row_seq, (uint64_t)n * 4u) &&
              pulsar_gpu_tensor_write(sc, p.row_j, row_j, (uint64_t)n * 4u) &&
              pulsar_gpu_tensor_write(sc, p.seq_first, seq_first, (uint64_t)n_seq * 4u) &&
              pulsar_gpu_tensor_write(sc, p.seq_rows, seq_rows, (uint64_t)n_seq * 4u) &&
              pulsar_gpu_tensor_write(sc, p.seq_bank, seq_bank, (uint64_t)n_seq * 4u);
    free(map);
    if (!ok) return fail("could not stage the PLE rows");

    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const int H = (int)st->shape->n_embd, HC = (int)pulsar_qwen_hc_dim(st->shape);
    pulsar_qwen_ple_dev w;
    ok = linear_dev(st, L.ple_key, H, HC, "qwen PLE key_proj", &w.key_proj) &&
         linear_dev(st, L.ple_value, H, H, "qwen PLE value_proj", &w.value_proj);
    w.norm_key = (const uint16_t *)wptr(st, L.ple_norm_key, "qwen PLE norm_key");
    w.norm_query = (const uint16_t *)wptr(st, L.ple_norm_query, "qwen PLE norm_query");
    w.norm_conv = (const uint16_t *)wptr(st, L.ple_norm_conv, "qwen PLE norm_conv");
    w.conv_w = (const uint16_t *)wptr(st, L.ple_conv, "qwen PLE conv1d");
    if (!ok || !w.norm_key || !w.norm_query || !w.norm_conv || !w.conv_w) return false;
    uint8_t *base = (uint8_t *)dptr(sc);
    const pulsar_qwen_rows rows = {(const int32_t *)(base + p.row_seq), (const int32_t *)(base + p.row_j),
                                   (const int32_t *)(base + p.seq_first), (const int32_t *)(base + p.seq_rows),
                                   (const int32_t *)(base + p.seq_bank), (int)n_seq};
    return pulsar_qwen_ple_launch(&w, (const uint16_t *)(base + p.emb), (uint16_t *)dptr(st->st->streams), (int)n,
                                  &rows, (float *)dptr(st->st->layer[il].ple_conv), base + p.ws, p.ws_bytes, 0) == 0;
}

bool pulsar_qwen_s4_gr_read(const pulsar_qwen_step *st, uint32_t il, pulsar_qwen_gr_side side) {
    const uint32_t n = st->n_rows;
    const int H = (int)st->shape->n_embd;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    pulsar_qwen_gr_dev w;
    if (!gr_dev(st, side == PULSAR_QWEN_GR_ATTN ? L.gr_attn : L.gr_mlp, &w)) return false;
    const gr_scratch g = gr_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GR_READ];
    if (!sc) return fail("no GR scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    /* the block input's A8 encoding goes to x's activation-cache slot (family_qwen.h s4) */
    void *xq = NULL, *xsf = NULL;
    int kbp = 0;
    pulsar_gpu_tensor *x = st->st->x;
    if (!pulsar_gpu_mxfp8_act_cache_e4m3_slot(x, n, (uint64_t)H, &xq, &xsf, &kbp)) return false;
    const pulsar_qwen_slot slot = {(uint8_t *)xq, (uint8_t *)xsf, kbp};
    if (pulsar_qwen_gr_read_launch(&w, (const uint16_t *)dptr(st->st->streams), (int)n, (uint16_t *)dptr(x), &slot,
                                   (float *)(base + g.inj[side]), base + g.ws, g.ws_bytes, 0))
        return false;
    pulsar_gpu_mxfp8_act_cache_arm(x, n, (uint64_t)H);
    pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    pulsar_gpu_mxfp8_act_cache_note_f32_skipped(n);   /* x holds the bf16 row, not f32 */
    return true;
}

bool pulsar_qwen_s4_gr_write(const pulsar_qwen_step *st, uint32_t, pulsar_qwen_gr_side side) {
    const gr_scratch g = gr_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GR_READ];
    if (!sc) return fail("no GR scratch");
    return pulsar_qwen_gr_write_launch((uint16_t *)dptr(st->st->streams), (const float *)dptr(st->st->y),
                                       (const float *)((uint8_t *)dptr(sc) + g.inj[side]), (int)st->n_rows, 0) == 0;
}

bool pulsar_qwen_s4_moe(const pulsar_qwen_step *st, uint32_t il) {
    const uint32_t n = st->n_rows;
    const pulsar_qwen_shape *s = st->shape;
    const int H = (int)s->n_embd, MID = (int)s->n_ff_exp, SMID = (int)s->n_ff_shexp;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    pulsar_qwen_moe_dev w;
    w.router_w = (const uint16_t *)wptr(st, L.moe_router, "qwen router");
    w.shared_gate_w = (const uint16_t *)wptr(st, L.sh_gate_scalar, "qwen shared_expert_gate");
    /* the two expert stacks: [trellis | suh | svh] per expert (exl3_expert_layout) */
    uint64_t tgu = 0, sc_gu = 0, stride_gu = 0, td = 0, sc_d = 0, stride_d = 0;
    w.k2_gate_up = exl3_type_k2(L.moe_gate_up->type);
    w.k2_down = exl3_type_k2(L.moe_down->type);
    if (!exl3_expert_layout((uint64_t)H, 2ull * MID, w.k2_gate_up, &tgu, &sc_gu, &stride_gu) ||
        !exl3_expert_layout((uint64_t)MID, (uint64_t)H, w.k2_down, &td, &sc_d, &stride_d) ||
        L.moe_gate_up->bytes != stride_gu * s->n_expert || L.moe_down->bytes != stride_d * s->n_expert)
        return fail("an expert stack's bytes are not n_expert EXL3 slices");
    const void *gu = wptr(st, L.moe_gate_up, "qwen experts gate_up"), *dn = wptr(st, L.moe_down, "qwen experts down");
    if (!gu || !dn || !w.router_w || !w.shared_gate_w) return false;
    w.gate_up_table = pulsar_qwen_expert_table(gu, s->n_expert, stride_gu, tgu);
    w.down_table = pulsar_qwen_expert_table(dn, s->n_expert, stride_d, td);
    if (!w.gate_up_table || !w.down_table) return fail("no EXL3 expert table");
    if (!linear_dev(st, L.sh_gate, H, SMID, "qwen shared gate_proj", &w.shared_gate) ||
        !linear_dev(st, L.sh_up, H, SMID, "qwen shared up_proj", &w.shared_up) ||
        !linear_dev(st, L.sh_down, SMID, H, "qwen shared down_proj", &w.shared_down))
        return false;
    const void *xq = NULL, *xsf = NULL;
    int kbp = 0;
    if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(st->st->x, n, (uint64_t)H, &xq, &xsf, &kbp))
        return fail("the MoE found no E4M3 slot for its input (the GR read is its producer)");
    const pulsar_qwen_slot x = {(uint8_t *)xq, (uint8_t *)xsf, kbp};
    const moe_scratch m = moe_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    if (!sc) return fail("no MoE scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    return pulsar_qwen_moe_launch(&w, (const uint16_t *)dptr(st->st->x), &x, (int)n, (float *)dptr(st->st->y),
                                  base + m.ws, m.ws_bytes, (uint32_t *)(base + m.nf), 0x51000000u | il, 0) == 0;
}

bool pulsar_qwen_s4_head(const pulsar_qwen_step *st, uint32_t row0, uint32_t n) {
    const pulsar_qwen_shape *s = st->shape;
    const int H = (int)s->n_embd;
    if (n == 0 || n > PULSAR_QWEN_HEAD_ROWS_MAX || row0 + n > st->n_rows) return fail("head rows out of range");
    pulsar_qwen_gr_dev w;
    if (!gr_dev(st, st->w->mixer, &w)) return false;
    const head_scratch h = head_layout();
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_HEAD];
    if (!sc) return fail("no head scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    /* the mixer's bf16 row IS the head GEMM's bf16 activation plane: written
     * straight into the slot of a key tensor, noted, read by the bf16 GEMM */
    pulsar_gpu_tensor *xkey = scratch_view(st, PULSAR_QWEN_OP_HEAD, h.xkey, (uint64_t)n * H * sizeof(float));
    void *xb = NULL;
    bool ok = xkey && pulsar_gpu_bf16_act_slot(xkey, n, (uint64_t)H, &xb);
    const uint16_t *streams = (const uint16_t *)dptr(st->st->streams) + (size_t)row0 * pulsar_qwen_hc_dim(s);
    ok = ok && pulsar_qwen_gr_read_launch(&w, streams, (int)n, (uint16_t *)xb, NULL, NULL, base + h.ws, h.ws_bytes, 0) == 0;
    if (ok) {
        pulsar_gpu_bf16_act_note(xkey, n, (uint64_t)H);
        const pulsar_decode_rows_scope rows(n);   /* the head's rows are decode rows: the M-independent arms */
        const pulsar_tensor *out = st->w->output;
        ok = rows.ok() && pulsar_gpu_matmul_bf16_tensor(st->st->logits, tensor_map_base(st->model, out),
                                                        tensor_map_size(st->model, out), out->abs_offset,
                                                        (uint64_t)H, s->n_vocab, xkey, n) != 0;
    }
    if (xkey) {
        pulsar_gpu_act_slot_drop(xkey);
        pulsar_gpu_tensor_free(xkey);
    }
    return ok || fail("the head (mixer + lm_head) failed");
}

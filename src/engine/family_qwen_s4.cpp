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
#include "cuda/pulsar_cuda_gdn.h"

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
    /* mxfp8_lt: E4M3 [out][in] then the swizzled E8M0 plane (k2 stays 0, which
     * is how the launcher knows).  EXL3 carries its scales inside the slice. */
    l->sf = (t->type == PULSAR_TENSOR_MXFP8_LT) ? (const uint8_t *)l->w + (uint64_t)out * in : NULL;
    return l->w != NULL;
}

/* L251 S2: the Gated DeltaNet op's scratch.  qkv/z/a/b hold the four f32
 * projections; `ws` is pulsar_gdn_forward's own scratch; a8/a8_sf are the A8
 * output slot the kernel emits and out_proj then reads -- sized with the
 * host-side geometry helpers, because pulsar_cuda_mx.cuh cannot be included
 * here (g++ does not know __host__/__device__).  `lin_ws` is ONE dense-arm
 * workspace shared by the five launches (they serialize on the stream). */
struct gdn_scratch { uint64_t qkv, z, a, b, ws, ws_bytes, a8, a8_sf, obf16, lin_ws, lin_ws_bytes, total; };

uint64_t gdn_lin_ws(const pulsar_qwen_shape *s, uint32_t rows) {
    const int H = (int)s->n_embd, CD = (int)pulsar_qwen_gdn_conv_dim(s);
    const int VT = (int)pulsar_qwen_gdn_v_total(s), NV = (int)s->gdn_n_v_head;
    const int dims[5][2] = {{H, CD}, {H, VT}, {H, NV}, {H, NV}, {VT, H}};
    uint64_t m = 0;
    for (int i = 0; i < 5; i++) {
        pulsar_qwen_linear l{};
        l.in = dims[i][0]; l.out = dims[i][1];
        const uint64_t b = a256(pulsar_qwen_linear_workspace_bytes(&l, (int)rows));
        if (b > m) m = b;
    }
    return m;
}

gdn_scratch gdn_layout(const pulsar_qwen_shape *s, uint32_t rows) {
    const uint64_t CD = pulsar_qwen_gdn_conv_dim(s), VT = pulsar_qwen_gdn_v_total(s), NV = s->gdn_n_v_head;
    gdn_scratch g{};
    uint64_t o = 0;
    g.qkv = o; o += a256((uint64_t)rows * CD * sizeof(float));
    g.z   = o; o += a256((uint64_t)rows * VT * sizeof(float));
    g.a   = o; o += a256((uint64_t)rows * NV * sizeof(float));
    g.b   = o; o += a256((uint64_t)rows * NV * sizeof(float));
    g.ws  = o; g.ws_bytes = a256((uint64_t)pulsar_gdn_scratch_bytes((int)rows)); o += g.ws_bytes;
    g.a8    = o; o += a256((uint64_t)rows * VT);                                          /* E4M3 codes */
    g.a8_sf = o; o += a256(pulsar_gpu_mx_sf_slab_bytes((int)rows, pulsar_gpu_mx_kbp((int)VT)));
    g.lin_ws = o; g.lin_ws_bytes = gdn_lin_ws(s, rows); o += g.lin_ws_bytes;
        /* L251 / ac69748f: the GDN output's bf16 row -- what the out_proj reads.  bf16 is 2 bytes
     * per element where the A8 slot was 1, so this take is twice the width of a8. */
    g.obf16 = o; o += a256((uint64_t)rows * VT * sizeof(uint16_t));
g.total = o;
    return g;
}

/* L251 S3: the QSA op's scratch.  qg/k/v/idx hold the four f32 projections (the
 * op hands the kernel VIEWS of them, because pulsar_qsa_io wants tensors and the
 * scratch IS one); a8/a8_sf are the o_proj A8 slot the kernel emits; ws is
 * pulsar_gpu_qsa_forward's workspace; lin_ws is the shared dense workspace. */
struct qsa_scratch { uint64_t qg, k, v, idx, a8, a8_sf, obf16, ws, ws_bytes, lin_ws, lin_ws_bytes, total; };

uint64_t qsa_lin_ws(const pulsar_qwen_shape *s, uint32_t rows) {
    const int H = (int)s->n_embd;
    const int dims[5][2] = {{H, PULSAR_QSA_Q_IN}, {H, PULSAR_QSA_KV_IN}, {H, PULSAR_QSA_KV_IN},
                            {H, PULSAR_QSA_IDX_IN}, {PULSAR_QSA_OUT_DIM, H}};
    uint64_t m = 0;
    for (int i = 0; i < 5; i++) {
        pulsar_qwen_linear l{};
        l.in = dims[i][0]; l.out = dims[i][1];
        const uint64_t b = a256(pulsar_qwen_linear_workspace_bytes(&l, (int)rows));
        if (b > m) m = b;
    }
    return m;
}

qsa_scratch qsa_layout(const pulsar_qwen_shape *s, uint32_t rows, uint32_t ctx) {
    qsa_scratch g{};
    uint64_t o = 0;
    g.qg  = o; o += a256((uint64_t)rows * PULSAR_QSA_Q_IN * sizeof(float));
    g.k   = o; o += a256((uint64_t)rows * PULSAR_QSA_KV_IN * sizeof(float));
    g.v   = o; o += a256((uint64_t)rows * PULSAR_QSA_KV_IN * sizeof(float));
    g.idx = o; o += a256((uint64_t)rows * PULSAR_QSA_IDX_IN * sizeof(float));
    g.a8    = o; o += a256((uint64_t)rows * PULSAR_QSA_OUT_DIM);
    g.a8_sf = o; o += a256(pulsar_gpu_mx_sf_slab_bytes((int)rows, pulsar_gpu_mx_kbp(PULSAR_QSA_OUT_DIM)));
    g.ws = o; g.ws_bytes = a256((uint64_t)pulsar_gpu_qsa_workspace_bytes(rows, ctx)); o += g.ws_bytes;
    g.lin_ws = o; g.lin_ws_bytes = qsa_lin_ws(s, rows); o += g.lin_ws_bytes;
        /* L251 / ac69748f: the o_proj's bf16 row.  bf16 is 2 bytes where the A8 slot is 1, so this
     * take is twice the width of a8. */
    g.obf16 = o; o += a256((uint64_t)rows * PULSAR_QSA_OUT_DIM * sizeof(uint16_t));
g.total = o;
    (void)s;
    return g;
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

/* a layer's MoE: the router, the routed experts (fused gate_up, or the MTP layer's gate + up pair), the
 * shared expert */
static bool admit_moe(const pulsar_qwen_layer_weights &L) {
    bool ok = admit_bf16(L.moe_router) & admit_bf16(L.sh_gate_scalar);
    if (L.moe_gate_up)
        ok &= admit_exl3(L.moe_gate_up, EXL3_ARM_GATE_UP_FUSED, "exl3m_k4 / exl3m_k5 (the fused gate_up arm)");
    else
        ok &= admit_exl3(L.moe_gate, EXL3_ARM_PAIR, "an exl3m rate the gate / up pair arm reads") &
              admit_exl3(L.moe_up, EXL3_ARM_PAIR, "an exl3m rate the gate / up pair arm reads") &
              admit(L.moe_up, L.moe_up->type == L.moe_gate->type, "the gate slice's rate (the pair shares one)");
    ok &= admit_exl3(L.moe_down, EXL3_ARM_DOWN, "an exl3m rate the routed down arm reads");
    ok &= admit_exl3(L.sh_gate, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
          admit_exl3(L.sh_up, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
          admit_exl3(L.sh_down, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads");
    return ok;
}

bool pulsar_qwen_s4_load(pulsar_engine *e, const pulsar_engine_options *opt) {
    pulsar_qwen_weights *w = e->qwen_weights;
    const pulsar_qwen_shape *s = &g_qwen_shape;
    bool ok = admit_bf16(w->token_embd) & admit_bf16(w->output) & admit_gr(w->mixer);
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        const pulsar_qwen_layer_weights &L = w->layer[il];
        ok &= admit_gr(L.gr_attn) & admit_gr(L.gr_mlp);
        ok &= admit_moe(L);
        if (L.ple_key) {
            ok &= admit_exl3(L.ple_key, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads") &
                  admit_exl3(L.ple_value, EXL3_ARM_DENSE, "an exl3m rate the dense arm reads");
            ok &= admit_bf16(L.ple_norm_key) & admit_bf16(L.ple_norm_query) & admit_bf16(L.ple_norm_conv) &
                  admit_bf16(L.ple_conv);
        }
    }
    if (w->mtp.present) {
        /* L251 MTP: the layer's GR sites and MoE as the trunk's (split experts); the head side's
         * norms bf16 and its two fc Linears mxfp8_lt (the sidecar recipe); the mixer's pair as the
         * trunk mixer's rule allows */
        const pulsar_qwen_layer_weights &L = w->layer[e->plan.n_layer];
        ok &= admit_gr(L.gr_attn) & admit_gr(L.gr_mlp) & admit_moe(L) & admit_gr(w->mtp.mixer);
        ok &= admit_bf16(w->mtp.norm_embd) & admit_bf16(w->mtp.norm_hidden) &
              admit_mx8(w->mtp.fc_embd) & admit_mx8(w->mtp.fc_hidden);
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
    if (w && w->head_mx) {
        pulsar_gpu_tensor_free(w->head_mx);
        w->head_mx = NULL;
    }
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

uint64_t pulsar_qwen_s4_scratch_bytes(pulsar_qwen_op_id op, const pulsar_qwen_shape *s, uint32_t max_rows, uint32_t ctx) {
    switch (op) {
    case PULSAR_QWEN_OP_EMBED:   return (uint64_t)max_rows * sizeof(int32_t);
    case PULSAR_QWEN_OP_PLE:     return ple_layout(max_rows).total;
    case PULSAR_QWEN_OP_GR_READ: return gr_layout(max_rows).total;
    case PULSAR_QWEN_OP_MOE:     return moe_layout(max_rows).total;
    case PULSAR_QWEN_OP_HEAD:    return head_layout().total;
    case PULSAR_QWEN_OP_GDN:     return gdn_layout(s, max_rows).total;
    case PULSAR_QWEN_OP_QSA:     return qsa_layout(s, max_rows, pulsar_qwen_qsa_cap(s, ctx)).total;
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
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    pulsar_qwen_gr_dev w;
    if (!gr_dev(st, side == PULSAR_QWEN_GR_ATTN ? L.gr_attn : L.gr_mlp, &w)) return false;
    const gr_scratch g = gr_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GR_READ];
    if (!sc) return fail("no GR scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    /* L251 / ac69748f: the block input is bf16 and there is no E4M3 activation slot in this family, so the
     * read emits the bf16 row and nothing else -- no slot, no arming, no notes.  `nullptr` is the read's
     * documented "no slot" form, so the mxfp8 W_down weights are still read by the same arm. */
    pulsar_gpu_tensor *x = st->st->x;
    if (pulsar_qwen_gr_read_launch(&w, (const uint16_t *)dptr(st->st->streams), (int)n, (uint16_t *)dptr(x),
                                   (float *)(base + g.inj[side]), base + g.ws, g.ws_bytes, 0))
        return false;
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
    pulsar_qwen_moe_dev w{};
    w.router_w = (const uint16_t *)wptr(st, L.moe_router, "qwen router");
    w.shared_gate_w = (const uint16_t *)wptr(st, L.sh_gate_scalar, "qwen shared_expert_gate");
    /* the expert stacks: [trellis | suh | svh] per expert (exl3_expert_layout).  The trunk's gate_up is
     * ONE fused stack (2 MID outputs); the MTP layer's gate and up are two stacks of MID (split). */
    const bool split = L.moe_gate_up == NULL;
    const pulsar_tensor *first = split ? L.moe_gate : L.moe_gate_up;
    const uint64_t first_out = split ? (uint64_t)MID : 2ull * MID;
    uint64_t tgu = 0, sc_gu = 0, stride_gu = 0, td = 0, sc_d = 0, stride_d = 0;
    w.k2_gate_up = exl3_type_k2(first->type);
    w.k2_down = exl3_type_k2(L.moe_down->type);
    if (!exl3_expert_layout((uint64_t)H, first_out, w.k2_gate_up, &tgu, &sc_gu, &stride_gu) ||
        !exl3_expert_layout((uint64_t)MID, (uint64_t)H, w.k2_down, &td, &sc_d, &stride_d) ||
        first->bytes != stride_gu * s->n_expert || L.moe_down->bytes != stride_d * s->n_expert ||
        (split && L.moe_up->bytes != stride_gu * s->n_expert))
        return fail("an expert stack's bytes are not n_expert EXL3 slices");
    const void *gu = wptr(st, first, split ? "qwen experts gate" : "qwen experts gate_up");
    const void *up = split ? wptr(st, L.moe_up, "qwen experts up") : NULL;
    const void *dn = wptr(st, L.moe_down, "qwen experts down");
    if (!gu || (split && !up) || !dn || !w.router_w || !w.shared_gate_w) return false;
    if (split) {
        w.gate_table = pulsar_qwen_expert_table(gu, s->n_expert, stride_gu, tgu);
        w.up_table = pulsar_qwen_expert_table(up, s->n_expert, stride_gu, tgu);
    } else {
        w.gate_up_table = pulsar_qwen_expert_table(gu, s->n_expert, stride_gu, tgu);
    }
    w.down_table = pulsar_qwen_expert_table(dn, s->n_expert, stride_d, td);
    if ((split ? !w.gate_table || !w.up_table : !w.gate_up_table) || !w.down_table)
        return fail("no EXL3 expert table");
    if (!linear_dev(st, L.sh_gate, H, SMID, "qwen shared gate_proj", &w.shared_gate) ||
        !linear_dev(st, L.sh_up, H, SMID, "qwen shared up_proj", &w.shared_up) ||
        !linear_dev(st, L.sh_down, SMID, H, "qwen shared down_proj", &w.shared_down))
        return false;
    /* L251 / ac69748f: there is no E4M3 activation slot in this family.  The MoE reads the block
     * input's bf16 row directly -- the routed arm by ids_src1 and the shared expert as a plain row --
     * so unlike the GDN and QSA ops there is nothing here to fetch, arm or encode. */
    const moe_scratch m = moe_layout(st->st->max_rows);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_MOE];
    if (!sc) return fail("no MoE scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    return pulsar_qwen_moe_launch(&w, (const uint16_t *)dptr(st->st->x), (int)n, (float *)dptr(st->st->y),
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
    ok = ok && pulsar_qwen_gr_read_launch(&w, streams, (int)n, (uint16_t *)xb, NULL, base + h.ws, h.ws_bytes, 0) == 0;
    if (ok) {
        /* L251: the lm_head in MXFP8, made once on the device from the container's bf16 head (the
         * producers' own encoder) -- half the bytes of the bf16 GEMV that was 5.1 ms of a 33 ms
         * token -- through the W8A16 arms (the GEMV at decode widths). */
        pulsar_qwen_weights *wm = const_cast<pulsar_qwen_weights *>(st->w);
        if (!wm->head_mx) {
            const uint16_t *hb = (const uint16_t *)wptr(st, st->w->output, "qwen lm_head");
            const uint64_t bytes = pulsar_qwen_mxfp8_bytes((int)s->n_vocab, H);
            wm->head_mx = hb && bytes ? pulsar_gpu_tensor_alloc(bytes) : NULL;
            ok = wm->head_mx && pulsar_qwen_bf16_to_mxfp8(hb, (int)s->n_vocab, H, dptr(wm->head_mx), 0) == 0;
            if (ok) fprintf(stderr, "pulsar: %s: lm_head as MXFP8 (%.2f GB, from bf16 %.2f GB)\n", PULSAR_QWEN_ARCH,
                            (double)bytes / 1e9, (double)s->n_vocab * H * 2 / 1e9);
            else if (wm->head_mx) { pulsar_gpu_tensor_free(wm->head_mx); wm->head_mx = NULL; }
        }
        if (ok) {
            const uint8_t *hq = (const uint8_t *)dptr(wm->head_mx);
            const pulsar_qwen_lowrank l = {hq, hq + (uint64_t)s->n_vocab * H, (int)s->n_vocab, H};
            ok = pulsar_qwen_mxfp8_linear_launch(&l, (const uint16_t *)xb, (int)n, (float *)dptr(st->st->logits),
                                                 NULL, 0, 0) == 0;
        }
    }
    if (xkey) {
        pulsar_gpu_act_slot_drop(xkey);
        pulsar_gpu_tensor_free(xkey);
    }
    return ok || fail("the head (mixer + lm_head) failed");
}

/* ======================================================================== */
/* L251 S2's op: the Gated DeltaNet block.  On the integration branch it lives
 * in this TU because it needs the weight / scratch helpers above (wptr,
 * linear_dev, dptr, fail, gdn_layout); split into family_qwen_s2.cpp when S2
 * rebases on the family interface.  The kernel is src/cuda/pulsar_cuda_gdn.cu.
 *
 * x (bf16 + its armed E4M3 slot) -> in_proj_qkv / _z / _a / _b through the EXL3
 * dense arm -> pulsar_gdn_forward (the recurrence; it EMITS the A8 slot) ->
 * out_proj -> y f32.  The four kernel weights are bf16, the container's
 * storage; the kernel widens them (rule 3). */
bool pulsar_qwen_s2_gdn(const pulsar_qwen_step *st, uint32_t il) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    const int H = (int)s->n_embd, CD = (int)pulsar_qwen_gdn_conv_dim(s);
    const int VT = (int)pulsar_qwen_gdn_v_total(s), NV = (int)s->gdn_n_v_head;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const pulsar_tensor *gw4[4] = {L.gdn_conv, L.gdn_a_log, L.gdn_dt_bias, L.gdn_norm};
    for (int i = 0; i < 4; i++)
        if (!admit(gw4[i], gw4[i]->type == PULSAR_TENSOR_BF16, "bf16 (the GDN kernel's weight dtype)")) return false;
    pulsar_qwen_linear qkv, zz, aa, bb, out;
    if (!linear_dev(st, L.gdn_in_qkv, H,  CD, "qwen GDN in_proj_qkv", &qkv) ||
        !linear_dev(st, L.gdn_in_z,   H,  VT, "qwen GDN in_proj_z",   &zz)  ||
        !linear_dev(st, L.gdn_in_a,   H,  NV, "qwen GDN in_proj_a",   &aa)  ||
        !linear_dev(st, L.gdn_in_b,   H,  NV, "qwen GDN in_proj_b",   &bb)  ||
        !linear_dev(st, L.gdn_out,    VT, H,  "qwen GDN out_proj",    &out)) return false;
    /* L251 / ac69748f: the block input is bf16 and there is no E4M3 activation slot in this family, so
     * these projections read the bf16 row its producer emitted (rule 3). */
    const uint16_t *xin = (const uint16_t *)dptr(st->st->x);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_GDN];
    if (!sc) return fail("no GDN scratch");
    uint8_t *base = (uint8_t *)dptr(sc);
    const gdn_scratch g = gdn_layout(s, st->st->max_rows);
    void *linws = base + g.lin_ws;
    if (pulsar_qwen_linear_launch(&qkv, xin, (int)n, (float *)(base + g.qkv), linws, g.lin_ws_bytes, 0) != 0 ||
        pulsar_qwen_linear_launch(&zz,  xin, (int)n, (float *)(base + g.z),   linws, g.lin_ws_bytes, 0) != 0 ||
        pulsar_qwen_linear_launch(&aa,  xin, (int)n, (float *)(base + g.a),   linws, g.lin_ws_bytes, 0) != 0 ||
        pulsar_qwen_linear_launch(&bb,  xin, (int)n, (float *)(base + g.b),   linws, g.lin_ws_bytes, 0) != 0)
        return fail("a GDN projection launch failed");
    pulsar_gdn_weights gw;
    gw.conv_w  = (const uint16_t *)wptr(st, L.gdn_conv,    "qwen GDN conv1d");
    gw.A_log   = (const uint16_t *)wptr(st, L.gdn_a_log,   "qwen GDN A_log");
    gw.dt_bias = (const uint16_t *)wptr(st, L.gdn_dt_bias, "qwen GDN dt_bias");
    gw.norm_w  = (const uint16_t *)wptr(st, L.gdn_norm,    "qwen GDN norm");
    if (!gw.conv_w || !gw.A_log || !gw.dt_bias || !gw.norm_w) return false;
    pulsar_gdn_call c{};
    const bool prefill = st->mode == PULSAR_QWEN_STEP_PREFILL;
    c.n_seq = prefill ? 1 : (int)n;                 /* DECODE: one row per bank; PREFILL: one sequence */
    c.seq_rows = prefill ? (int)n : 1;
    c.row_slot = (const int32_t *)dptr(st->st->row_bank);
    c.conv_state = (float *)dptr(st->st->layer[il].gdn_conv);
    c.rec_state  = (float *)dptr(st->st->layer[il].gdn_state);
    c.qkv = (const float *)(base + g.qkv); c.ld_qkv = CD;
    c.z   = (const float *)(base + g.z);   c.ld_z   = VT;
    c.a   = (const float *)(base + g.a);   c.ld_a   = NV;
    c.b   = (const float *)(base + g.b);   c.ld_b   = NV;
    c.scratch = base + g.ws; c.scratch_bytes = g.ws_bytes;
    c.out_f32 = NULL;                               /* the bf16 row is the consumer's input, not f32 */
    c.out_bf16 = base + g.obf16;                    /* L251 / ac69748f: no E4M3 slot in this family */
    if (pulsar_gdn_forward(&gw, &c, 0) != 0) return fail("pulsar_gdn_forward failed");
    return pulsar_qwen_linear_launch(&out, (const uint16_t *)(base + g.obf16), (int)n,
                                     (float *)dptr(st->st->y), linws, g.lin_ws_bytes, 0) == 0 ||
           fail("the GDN out_proj launch failed");
}

/* ======================================================================== */
/* L251 S3's op: the Qwen full-attention + QSA layer.  On the integration branch
 * it lives in this TU for the same reason the gdn op does; split into
 * family_qwen_s3.cpp when S3 rebases.  The kernel is src/cuda/pulsar_cuda_qsa.cu.
 *
 * x (bf16 + its armed E4M3 slot) -> q_proj / k_proj / v_proj / index_qk_proj
 * through the EXL3 dense arm -> pulsar_gpu_qsa_forward (appends K/V + the pooled
 * block keys to the bank's caches and EMITS the A8 slot) -> o_proj -> y f32.
 * The four norms are bf16 (the recipe's format; the kernel widens). */
bool pulsar_qwen_s3_qsa(const pulsar_qwen_step *st, uint32_t il) {
    const pulsar_qwen_shape *s = st->shape;
    const uint32_t n = st->n_rows;
    const int H = (int)s->n_embd;
    const pulsar_qwen_layer_weights &L = layer_w(st, il);
    const pulsar_tensor *nrm[4] = {L.attn_q_norm, L.attn_k_norm, L.idx_q_norm, L.idx_k_norm};
    for (int i = 0; i < 4; i++)
        if (!admit(nrm[i], nrm[i]->type == PULSAR_TENSOR_BF16, "bf16 (the QSA kernel's norm dtype)")) return false;
    pulsar_qwen_linear q, k, v, ix, out;
    if (!linear_dev(st, L.attn_q, H, PULSAR_QSA_Q_IN,   "qwen QSA q_proj", &q) ||
        !linear_dev(st, L.attn_k, H, PULSAR_QSA_KV_IN,  "qwen QSA k_proj", &k) ||
        !linear_dev(st, L.attn_v, H, PULSAR_QSA_KV_IN,  "qwen QSA v_proj", &v) ||
        !linear_dev(st, L.idx_qk, H, PULSAR_QSA_IDX_IN, "qwen QSA index_qk_proj", &ix) ||
        !linear_dev(st, L.attn_o, PULSAR_QSA_OUT_DIM, H, "qwen QSA o_proj", &out)) return false;
    /* L251 / ac69748f: bf16, as above -- the QSA projections read the block input directly. */
    const uint16_t *xin = (const uint16_t *)dptr(st->st->x);
    pulsar_gpu_tensor *sc = st->st->scratch[PULSAR_QWEN_OP_QSA];
    if (!sc) return fail("no QSA scratch");
    const qsa_scratch g = qsa_layout(s, st->st->max_rows, st->st->ctx);
    void *linws = (uint8_t *)dptr(sc) + g.lin_ws;

    /* pulsar_qsa_io wants tensors for the four projections; the scratch IS a
     * tensor, so hand it views (and free them on every exit). */
    pulsar_gpu_tensor *vq = scratch_view(st, PULSAR_QWEN_OP_QSA, g.qg, (uint64_t)n * PULSAR_QSA_Q_IN * 4);
    pulsar_gpu_tensor *vk = scratch_view(st, PULSAR_QWEN_OP_QSA, g.k, (uint64_t)n * PULSAR_QSA_KV_IN * 4);
    pulsar_gpu_tensor *vv = scratch_view(st, PULSAR_QWEN_OP_QSA, g.v, (uint64_t)n * PULSAR_QSA_KV_IN * 4);
    pulsar_gpu_tensor *vi = scratch_view(st, PULSAR_QWEN_OP_QSA, g.idx, (uint64_t)n * PULSAR_QSA_IDX_IN * 4);
    pulsar_gpu_tensor *views[64 * 3 + 4];
    int nv = 0;
    if (vq) views[nv++] = vq;
    if (vk) views[nv++] = vk;
    if (vv) views[nv++] = vv;
    if (vi) views[nv++] = vi;
    auto drop = [&]() { for (int i = 0; i < nv; i++) pulsar_gpu_tensor_free(views[i]); };

    const uint32_t nb = st->st->n_banks;
    if (nv != 4 || nb == 0 || nb > 64) { drop(); return fail("the QSA op needs 4 projection views and 1..64 banks"); }
    bool ok = pulsar_qwen_linear_launch(&q,  xin, (int)n, (float *)dptr(vq), linws, g.lin_ws_bytes, 0) == 0 &&
              pulsar_qwen_linear_launch(&k,  xin, (int)n, (float *)dptr(vk), linws, g.lin_ws_bytes, 0) == 0 &&
              pulsar_qwen_linear_launch(&v,  xin, (int)n, (float *)dptr(vv), linws, g.lin_ws_bytes, 0) == 0 &&
              pulsar_qwen_linear_launch(&ix, xin, (int)n, (float *)dptr(vi), linws, g.lin_ws_bytes, 0) == 0;
    if (!ok) { drop(); return fail("a QSA projection launch failed"); }

    /* the per-bank cache views: bank-major, at the sizes family_qwen.h owns */
    const uint64_t kvb = pulsar_qwen_kv_row_bytes(s);
    const uint64_t ikb = pulsar_qwen_index_row_bytes(s);
    const uint64_t itb = pulsar_qwen_index_tail_bytes(s);
    const uint64_t nblk = ((uint64_t)st->st->ctx + s->idx_block - 1u) / s->idx_block;
    pulsar_qsa_seq seqs[64];
    for (uint32_t b = 0; ok && b < nb; b++) {
        seqs[b].kv    = pulsar_gpu_tensor_view(st->st->layer[il].kv,       b * st->st->ctx * kvb, st->st->ctx * kvb);
        seqs[b].bkey  = pulsar_gpu_tensor_view(st->st->layer[il].idx_keys, b * nblk * ikb, nblk * ikb);
        seqs[b].stage = pulsar_gpu_tensor_view(st->st->layer[il].idx_tail, b * itb, itb);
        seqs[b].cap   = st->st->ctx;
        views[nv++] = seqs[b].kv; views[nv++] = seqs[b].bkey; views[nv++] = seqs[b].stage;
        ok = seqs[b].kv && seqs[b].bkey && seqs[b].stage;
    }
    if (!ok) { drop(); return fail("a QSA bank cache view failed"); }

    const pulsar_qsa_layer layer = {(const uint16_t *)wptr(st, L.attn_q_norm,  "qwen QSA q_norm"),
                                    (const uint16_t *)wptr(st, L.attn_k_norm,  "qwen QSA k_norm"),
                                    (const uint16_t *)wptr(st, L.idx_q_norm,   "qwen QSA idx_q_norm"),
                                    (const uint16_t *)wptr(st, L.idx_k_norm,   "qwen QSA idx_k_norm")};
    pulsar_qsa_io io{};
    io.qg = vq; io.k = vk; io.v = vv; io.idx = vi;
    io.out_bf16 = (uint8_t *)dptr(sc) + g.obf16;      /* L251 / ac69748f: bf16, not the A8 slot */
    io.out_e4m3 = NULL;
    io.out_scale = (uint8_t *)dptr(sc) + g.a8_sf;
    io.out_sf_pitch = pulsar_gpu_mx_kbp(PULSAR_QSA_OUT_DIM);
    io.tap_out_f32 = NULL; io.tap_sel = NULL;              /* the lane, not a gate */
    /* row_seq / row_pos are HOST arrays -- pulsar_gpu_qsa_forward walks them on
     * the host to build its row list.  The step already carries them as `bank` /
     * `pos`.  (The GDN call's row_slot is the opposite: a DEVICE pointer.  Passing
     * the device row_bank here segfaulted the host walk, 18:56.) */
    const int rc = ok ? pulsar_gpu_qsa_forward(&layer, seqs, nb,
                                               (const uint32_t *)st->bank,
                                               (const uint32_t *)st->pos, n, &io,
                                               scratch_view(st, PULSAR_QWEN_OP_QSA, g.ws, g.ws_bytes)) : -1;
    drop();
    if (rc != 1) return fail("pulsar_gpu_qsa_forward refused the step");
    return pulsar_qwen_linear_launch(&out, (const uint16_t *)((uint8_t *)dptr(sc) + g.obf16), (int)n,
                                     (float *)dptr(st->st->y), linws, g.lin_ws_bytes, 0) == 0 ||
           fail("the QSA o_proj launch failed");
}

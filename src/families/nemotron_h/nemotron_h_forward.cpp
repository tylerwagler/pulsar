/* nemotron_h_forward.cpp -- the Nemotron-H forward on the core row-step driver (L296 N3-N7; step_driver.h).
 *
 * The family declares; the core runs.  What is Nemotron-H's here is only the composition -- which shared blocks a
 * layer runs, in which order, on which state, with which declaration.  Every kernel is a shared block's:
 *
 *   embed      E5   pulsar_head_embed_launch: the gather (no embedding norm) -> the residual h (bf16)
 *   per layer  (pulsar_layer_desc: a Mamba-2 or an attention mixer, the MoE or no FFN -- one HF block each)
 *       xn = rmsnorm(h) * mixer_norm  (layer 0; then the previous layer's last add) row ops (pulsar_cuda_rowops.h)
 *     Mamba-2:  p = bf16(xn W_in) = [z | x B C | dt]                              rows front door (linear.cpp)
 *               y = Mamba2(p) (conv + bias + SiLU, selective scan, D skip,         N2 (pulsar_cuda_mamba2.h; the
 *                   gated grouped RMSNorm)                                         SSM and conv state: F3's)
 *               delta = bf16(y W_out)
 *     attention: q | k | v = bf16(xn W_qkv)  (one multi launch)                   rows front door
 *               a = GQA(q, k, v)  (FULL span, 1/sqrt(head_dim), no positional     F1 (the KV pool: F3's)
 *                   encoding, no norm / bias / gate)
 *               delta = bf16(a W_o)
 *       h += delta  [; xn = rmsnorm(h) * ffn_norm  when the layer has the MoE]    row ops (one fused launch)
 *     MoE:      router: f32 logits of xn, sigmoid, top-22 of 512 by score +        F2 (the 512 / 22 instance)
 *                   the selection bias, renormalised by the sum x 5.0
 *               f = latent_out(bf16(sum_k w_k relu2_expert_k(bf16(latent_in xn))))  the rows MoE front door: the
 *                   + shared(xn), the shared expert relu^2 on the full width          latent wrapper + relu^2 (N4)
 *               h += bf16(f)                                                      row ops
 *       the layer's last add also writes xn = rmsnorm(h) * the next reader's norm (the next layer's mixer norm, or
 *       output_norm after the last layer) -- one launch (L296 perf)
 *   head       xn -> the head linear (untied, every id real) -> E5 finish (nothing to scale)
 *
 * vLLM's NemotronHForCausalLM (models/nemotron_h.py) is the order: each HF block is `x += mixer(RMSNorm(x))`, the
 * residual bf16, each add and each norm one rounding (vLLM's fused add_rmsnorm).  pulsar groups a mixer with the MoE
 * block after it into one layer (nemotron_h_shape.cpp's plan), which changes nothing in the arithmetic: the MoE's
 * pre-norm is the fused add's output.  Where pulsar rounds differently, it is said at the site and graded (rule 7):
 *   - the f32 linears' one bf16 rounding (= a bf16 GEMM's output);
 *   - the router's route scale 5.0 on each weight (F2's SUM renormalisation) where vLLM scales the bf16 routed sum
 *     (`apply_routed_scale_to_output`), and the weight folded into each expert's relu^2 row before its bf16 rounding
 *     (F7's fold) where vLLM multiplies the down's output;
 *   - the routed sum kept f32 into one bf16 rounding before the back projection (vLLM: the fused-MoE output bf16, x 5
 *     in bf16); the shared expert added to the back projection's output in f32 before the one bf16 rounding of the
 *     block's delta (vLLM adds two bf16 tensors).
 *
 * ROW KINDS (L284 #2): every arm-sensitive launch (the linears, the routed experts) runs rows [0, n_dec) on the decode
 * arms and [n_dec, n_rows) on the prompt arms; the router, the row ops, the Mamba-2 block and F1 are one arithmetic at
 * any row count.  So a prompt cut anywhere is the cold prefill's bytes (the plan is split-invariant), and a decode row
 * is the same bytes alone or batched.
 *
 * STATE (state_kinds.h; nemotron_h.h PULSAR_NEMOTRON_H_STATE_*): a Mamba-2 layer's SSM state (FIXED, a verify's DELTA
 * per row, committed by pulsar_mamba2_commit) and its conv tail (FIXED, SNAPSHOT per row); an attention layer's K|V
 * pool (POOL); the drafter slot's K|V pool (POOL: the MTP module's attention) and each bank's parked conditioning row
 * (FIXED), written by the chained-MTP lane (N9, spec_mtp.h).  The core allocates, prices, pages, walks and commits it;
 * nothing here copies state.
 *
 * THE MTP MODULE (N9): one module step is this file's composition on its own residual stream (mtp_forward, below); the
 * core's chained-MTP lane decides which rows a step carries and runs the module recursively for a chain. */
#include "nemotron_h.h"
#include "engine/spec_mtp.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_mamba2.h"
#include "cuda/pulsar_cuda_qwen.h"     /* the rows front door's launchers + the rows MoE block (family-neutral) */
#include "cuda/pulsar_cuda_router.h"
#include "cuda/pulsar_cuda_rowops.h"

namespace {

const char *const ARCH = PULSAR_NEMOTRON_H_ARCH;

/* The resume grid (kv_state.h): the pools are one token a row and every prompt chunk takes the prompt arms at any row
 * count, so a cut anywhere is the cold bytes; 128 keeps the slots on the server's cadence. */
constexpr uint32_t kResumeGrid = 128u;
/* A checkpoint walks every Mamba-2 layer's FIXED state (40 x (4 MiB SSM + 60 KiB conv) = 162 MiB on the full model) */
constexpr uint32_t kCkptSlots = 4u;

/* The op arenas (pulsar_driver_ops::scratch_bytes) */
enum nh_arena { ARENA_ROWS = 0, ARENA_MIX, ARENA_FFN, ARENA_HEAD, ARENA_COUNT };

uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }
void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

bool fail(const char *what) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the step\n", ARCH, what);
    return false;
}
bool failf(const char *what, uint32_t il) {
    fprintf(stderr, "pulsar: %s: %s (layer %u) -- refusing the step\n", ARCH, what, il);
    return false;
}

/* ---- the declarations every op reads ------------------------------------------------------------------------------
 * The scratch is laid out from the shape before any session exists (pulsar_driver_ops::scratch_bytes has no engine),
 * and the geometry IS the profile's: nemotron_h_shape.cpp pins every geometry field to PULSAR_NEMOTRON_H_SHAPE_SUPER. */
const pulsar_nemotron_h_shape &G() { return PULSAR_NEMOTRON_H_SHAPE_SUPER; }

/* N2: the Mamba-2 block's geometry -- heads x head_dim, d_state, the B / C groups, the conv kernel, the gated norm's
 * group (d_inner / groups: vLLM's mamba_mixer2 group_size) and eps */
pulsar_mamba2_geom mamba_geom(const pulsar_nemotron_h_shape &s) {
    pulsar_mamba2_geom g{};
    g.n_heads = s.ssm_n_head;
    g.head_dim = s.ssm_head_dim;
    g.d_state = s.ssm_state;
    g.n_groups = s.ssm_groups;
    g.conv_kernel = s.conv_k;
    g.norm_group = s.ssm_n_head * s.ssm_head_dim / s.ssm_groups;
    g.norm_eps = s.rms_eps;
    return g;
}
/* the in_proj row [z | x B C | dt]: d_inner, the conv's channels, heads */
struct mamba_dims { uint32_t di, cc, nh, in; };
mamba_dims mamba_of(const pulsar_nemotron_h_shape &s) {
    const pulsar_mamba2_geom g = mamba_geom(s);
    mamba_dims d;
    d.di = s.ssm_n_head * s.ssm_head_dim;
    d.cc = pulsar_mamba2_conv_channels(&g);
    d.nh = s.ssm_n_head;
    d.in = d.di + d.cc + d.nh;
    return d;
}

/* N5, F1's declaration of an attention layer: FULL span, n_q / n_kv / head_dim from the shape, the score scale
 * 1/sqrt(head_dim), no positional encoding (NoPE: nothing to prepare), no q/k norm, no relative bias, no temperature,
 * no output gate -- vLLM's NemotronHAttention */
pulsar_gqa_layer gqa_decl(const pulsar_nemotron_h_shape &s) {
    pulsar_gqa_layer L{};
    L.n_q = s.n_head;
    L.n_kv = s.n_head_kv;
    L.head_dim = s.head_dim;
    L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = PULSAR_GQA_FULL;
    L.window = 0;
    L.scale = (float)(1.0 / sqrt((double)s.head_dim));
    L.qk_norm = 0;
    L.rel_extent = 0;
    L.temp_floor = 0.0f;
    L.temp_alpha = 0.0f;
    L.out_gate = 0;
    return L;
}

/* N4, F2: the router as the shared router's declaration -- sigmoid scores, the selection bias picks the top-k (the
 * weights stay unbiased), the picks renormalised by their sum (no floor) x routed_scaling_factor; f32 logits (vLLM's
 * gate GEMM runs fp32), f32 weights; one shared expert, UNGATED (no gate row: scale 1). */
pulsar_router_cfg router_cfg(const pulsar_nemotron_h_shape &s) {
    pulsar_router_cfg c{};
    c.n_expert = (int)s.n_expert;
    c.top_k = (int)s.n_expert_used;
    c.n_shared = 1;
    c.shared_gate = PULSAR_ROUTER_SHARED_NONE;
    c.score = PULSAR_ROUTER_SCORE_SIGMOID;
    c.renorm = PULSAR_ROUTER_RENORM_SUM;
    c.logits_bf16 = false;
    c.weights_bf16 = false;
    c.sum_floor = 0.0f;
    c.route_scale = s.route_scale;
    return c;
}

/* N4: the latent MoE block's shape -- the routed experts on the latent (latent -> n_ff_exp -> latent), one shared
 * expert on the hidden (hidden -> n_ff_shared -> hidden) */
pulsar_moe_block_shape moe_shape(const pulsar_nemotron_h_shape &s) {
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_moe_block_shape b{};
    b.n_expert = rc.n_expert;
    b.top_k = rc.top_k;
    b.n_logit_rows = pulsar_router_logit_rows(&rc);
    b.hidden = (int)s.n_embd;
    b.expert_mid = (int)s.n_ff_exp;
    b.n_shared = rc.n_shared;
    b.shared_mid = (int)s.n_ff_shared;
    b.latent = (int)s.latent;
    return b;
}

/* The dense-arm workspace of n linears of one input (a multi launch) or of one linear in -> out */
uint64_t multi_ws(int in, int n, const int *out, uint32_t rows) {
    pulsar_rows_linear l[4] = {};
    const pulsar_rows_linear *lp[4];
    for (int j = 0; j < n; j++) {
        l[j].in = in;
        l[j].out = out[j];
        lp[j] = &l[j];
    }
    return a256(pulsar_rows_linear_multi_workspace_bytes(lp, n, (int)rows));
}
uint64_t lin_ws(int in, int out, uint32_t rows) {
    pulsar_rows_linear l{};
    l.in = in;
    l.out = out;
    return a256(pulsar_rows_linear_workspace_bytes(&l, (int)rows));
}
uint64_t umax(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* ---- scratch layouts: one function each, for sizing AND carving -------------------------------------------------- */

/* ROWS: the step's tables -- the token ids, the Mamba-2 calls' sequence tables (the verify rows' captured call, then
 * the rest), the MoE non-finite flag */
struct rows_scratch { uint64_t ids, first_a, slot_a, first_b, slot_b, nf, total; };
rows_scratch rows_layout(uint32_t R) {
    rows_scratch r{};
    uint64_t o = 0;
    r.ids = o;     o += a256((uint64_t)R * 4u);
    r.first_a = o; o += a256((uint64_t)(R + 1u) * 4u);
    r.slot_a = o;  o += a256((uint64_t)R * 4u);
    r.first_b = o; o += a256((uint64_t)(R + 1u) * 4u);
    r.slot_b = o;  o += a256((uint64_t)R * 4u);
    r.nf = o;      o += 256u;
    r.total = o;
    return r;
}

/* MIX: the mixer's buffers -- a layer runs ONE mixer, so the Mamba-2 and the attention views share the region:
 *   pf  f32 [R][in]      the in_proj's output (Mamba) / q | k | v at their offsets (attention)
 *   pb  bf16 [R][in]     its bf16 rows (N9: the MTP module's [enorm(embed) | hnorm(cond)] rows before its layer)
 *   yb  bf16 [R][d_inner | n_q head_dim]   the block's output (the out / o projection's input)
 *   of  f32 [R][n_embd]  the out / o projection's output;  db bf16 [R][n_embd] its bf16 rows (the delta)
 *   m2  the Mamba-2 block's scratch;  lin the dense arm's workspace;  gqa F1's workspace */
struct mix_scratch { uint64_t pf, pb, yb, of, db, m2, m2_bytes, lin, lin_bytes, gqa, gqa_bytes, total; };
mix_scratch mix_layout(uint32_t R, uint32_t ctx) {
    const pulsar_nemotron_h_shape &s = G();
    const mamba_dims md = mamba_of(s);
    const uint64_t E = s.n_embd, Q = (uint64_t)s.n_head * s.head_dim, KV = (uint64_t)s.n_head_kv * s.head_dim;
    const uint64_t P = umax(md.in, Q + 2u * KV), Y = umax(md.di, Q);
    mix_scratch t{};
    uint64_t o = 0;
    t.pf = o; o += a256((uint64_t)R * P * 4u);
    t.pb = o; o += a256((uint64_t)R * P * 2u);
    t.yb = o; o += a256((uint64_t)R * Y * 2u);
    t.of = o; o += a256((uint64_t)R * E * 4u);
    t.db = o; o += a256((uint64_t)R * E * 2u);
    const pulsar_mamba2_geom g = mamba_geom(s);
    t.m2 = o; t.m2_bytes = a256(pulsar_mamba2_scratch_bytes(&g, R)); o += t.m2_bytes;
    const int qkv[3] = {(int)Q, (int)KV, (int)KV};
    t.lin = o;
    t.lin_bytes = umax(umax(lin_ws((int)E, (int)md.in, R), lin_ws((int)md.di, (int)E, R)),
                       umax(umax(multi_ws((int)E, 3, qkv, R), lin_ws((int)Q, (int)E, R)), lin_ws(2 * (int)E, (int)E, R)));
    o += t.lin_bytes;
    const pulsar_gqa_layer gd = gqa_decl(s);
    t.gqa = o; t.gqa_bytes = a256(pulsar_gqa_workspace_bytes(&gd, R, ctx ? ctx : 1u)); o += t.gqa_bytes;
    t.total = o;
    return t;
}

/* FFN: the block's f32 output, its bf16 row, the MoE block's one carved workspace (router, latent, routed, shared) */
struct ffn_scratch { uint64_t fo, db, u, moe_bytes, total; };
ffn_scratch ffn_layout(uint32_t R) {
    const pulsar_nemotron_h_shape &s = G();
    const uint64_t E = s.n_embd;
    ffn_scratch f{};
    uint64_t o = 0;
    f.fo = o; o += a256((uint64_t)R * E * 4u);
    f.db = o; o += a256((uint64_t)R * E * 2u);
    const pulsar_moe_block_shape bs = moe_shape(s);
    f.u = o; f.moe_bytes = a256(pulsar_moe_block_workspace_bytes(&bs, (int)R)); o += f.moe_bytes;
    f.total = o;
    return f;
}

/* HEAD: the head linear's workspace (at most PULSAR_DRIVER_HEADS_MAX rows; its input is xn, the final norm) */
struct head_scratch { uint64_t lin, lin_bytes, total; };
head_scratch head_layout() {
    const pulsar_nemotron_h_shape &s = G();
    head_scratch h{};
    h.lin = 0;
    h.lin_bytes = lin_ws((int)s.n_embd, (int)s.n_vocab, PULSAR_DRIVER_HEADS_MAX);
    h.total = h.lin + h.lin_bytes;
    return h;
}

uint64_t scratch_bytes(uint32_t arena, uint32_t max_rows, uint32_t ctx) {
    switch (arena) {
    case ARENA_ROWS: return rows_layout(max_rows).total;
    case ARENA_MIX:  return mix_layout(max_rows, ctx).total;
    case ARENA_FFN:  return ffn_layout(max_rows).total;
    case ARENA_HEAD: return head_layout().total;
    }
    return 0;
}

/* ---- the session's state and the step ------------------------------------------------------------------------- */

struct nh_step;
/* The family's session state (the driver's buffers are its base): the residual stream and the normed rows the
 * blocks read, the trunk's step object, and the step's host tables (rebuilt by every stage). */
struct nh_state : pulsar_driver_state {
    pulsar_gpu_tensor *h;     /* [max_rows][n_embd] bf16: the residual stream */
    pulsar_gpu_tensor *xn;    /* [max_rows][n_embd] bf16: the RMSNorm of h a block reads */
    nh_step *trunk;
    /* N9, the MTP module's step (when the session drafts): its own residual stream (the trunk's must outlive the
     * verify, whose rows the absorb reads), its output rows (final_layernorm: the head's input and the next chain
     * step's conditioning), the draft pick's workspace */
    nh_step *mtp;
    pulsar_gpu_tensor *mh;    /* [max_rows][n_embd] bf16 */
    pulsar_gpu_tensor *mo;    /* [max_rows][n_embd] bf16 */
    pulsar_gpu_tensor *mtop;  /* pulsar_qwen_draft_top_ws_bytes */
    /* the step's sequences: one a decode row, one a prefill run (a verify's runs, then the prompt runs) */
    uint32_t n_seq;
    uint32_t n_seq_cap;       /* [0, n_seq_cap) are the verify's runs (their Mamba-2 state captured), rows [0, rows_cap) */
    uint32_t rows_cap;
    uint32_t *seq_first;      /* host [max_rows + 1] */
    int32_t *seq_bank;        /* host [max_rows] */
    uint32_t *row_seq;        /* host [max_rows]: F1's */
    uint32_t *row_pos;        /* host [max_rows]: F1's */
    int32_t *tab_a, *tab_b;   /* host [2 (max_rows + 1)]: the Mamba-2 calls' seq_first then seq_slot, as uploaded */
    uint32_t max_run_a, max_run_b;
    pulsar_gqa_seq *seqs;     /* host [max_rows]: F1's sequences for the layer being planned */
    /* L296 perf: every attention layer's F1 call, planned by the step's host half (stage) and launched by the block
     * from the uploaded descriptors -- the step's launches read no host value but the step's shape */
    uint32_t n_attn;              /* attention layers in the plan; call n_attn is the MTP module's (N9, drafting) */
    uint32_t *attn_slot;          /* host [n_layer]: the layer's attention ordinal (UINT32_MAX = none) */
    pulsar_gqa_call *attn_call;   /* host [n_calls]: the calls' plans */
    uint64_t *attn_off;           /* host [n_calls]: each call's descriptors' offset in attn_desc(_host) */
    uint8_t *attn_desc_host;      /* host [n_calls * attn_desc_max]: the descriptors, staged */
    pulsar_gpu_tensor *attn_desc; /* device, the same layout */
    uint64_t attn_desc_max;       /* a256(pulsar_gqa_desc_bytes(max_rows)) */
};

struct nh_step : pulsar_driver_step {
    const pulsar_nemotron_h_model *w;
    nh_state *st;
    pulsar_gpu_tensor *res;                  /* the residual stream the step runs on: the trunk's h, the MTP's mh */
    const pulsar_nemotron_h_block *module;   /* N9: the MTP module's layer (every il of the step is its slot), or NULL */
};

const nh_step *S(const pulsar_driver_step *st) { return static_cast<const nh_step *>(st); }
nh_state *ST(pulsar_session *s) { return static_cast<nh_state *>(s->driver); }
const pulsar_nemotron_h_model *W(const pulsar_engine *e) { return (const pulsar_nemotron_h_model *)e->family_state; }
/* layer il's weights: the trunk's, or the MTP module's on its step */
const pulsar_nemotron_h_block &LW(const nh_step *st, uint32_t il) { return st->module ? *st->module : st->w->layer[il]; }

void state_free(nh_state *st) {
    if (!st) return;
    pulsar_gpu_tensor_free(st->h);
    pulsar_gpu_tensor_free(st->xn);
    pulsar_gpu_tensor_free(st->mh);
    pulsar_gpu_tensor_free(st->mo);
    pulsar_gpu_tensor_free(st->mtop);
    free(st->mtp);
    free(st->seq_first);
    free(st->seq_bank);
    free(st->row_seq);
    free(st->row_pos);
    free(st->tab_a);
    free(st->tab_b);
    free(st->seqs);
    free(st->attn_slot);
    free(st->attn_call);
    free(st->attn_off);
    free(st->attn_desc_host);
    pulsar_gpu_tensor_free(st->attn_desc);
    free(st->trunk);
    free(st);
}

pulsar_driver_state *create(pulsar_session *s, uint32_t n_banks) {
    const pulsar_engine *e = s->engine;
    const pulsar_nemotron_h_shape &sh = W(e)->shape;
    nh_state *st = (nh_state *)xcalloc(1, sizeof(*st));
    const uint64_t R = s->prefill_cap;
    st->h = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->xn = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->seq_first = (uint32_t *)xmalloc((R + 1u) * sizeof(uint32_t));
    st->seq_bank = (int32_t *)xmalloc(R * sizeof(int32_t));
    st->row_seq = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->row_pos = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->tab_a = (int32_t *)xmalloc(2u * (R + 1u) * sizeof(int32_t));
    st->tab_b = (int32_t *)xmalloc(2u * (R + 1u) * sizeof(int32_t));
    st->seqs = (pulsar_gqa_seq *)xcalloc(R, sizeof(pulsar_gqa_seq));
    st->trunk = (nh_step *)xcalloc(1, sizeof(nh_step));
    st->trunk->w = W(e);
    st->trunk->st = st;
    st->trunk->res = st->h;
    /* L296 perf: the attention calls -- the trunk's attention layers, then (N9, when the session drafts) the MTP
     * module's at ordinal n_attn; each call's descriptors at its own offset of one device region */
    st->attn_slot = (uint32_t *)xmalloc((size_t)e->plan.n_layer * sizeof(uint32_t));
    for (uint32_t il = 0; il < e->plan.n_layer; il++)
        st->attn_slot[il] = e->plan.desc[il].mixer == PULSAR_MIXER_ATTN_FULL ? st->n_attn++ : UINT32_MAX;
    const uint32_t n_calls = st->n_attn + (W(e)->drafting ? 1u : 0u);
    st->attn_desc_max = a256(pulsar_gqa_desc_bytes((uint32_t)R));
    st->attn_call = (pulsar_gqa_call *)xcalloc(n_calls ? n_calls : 1u, sizeof(pulsar_gqa_call));
    st->attn_off = (uint64_t *)xcalloc(n_calls ? n_calls : 1u, sizeof(uint64_t));
    st->attn_desc_host = (uint8_t *)xmalloc(n_calls ? n_calls * st->attn_desc_max : 1u);
    st->attn_desc = n_calls ? pulsar_gpu_tensor_alloc(n_calls * st->attn_desc_max) : NULL;
    bool ok = st->h && st->xn && (!n_calls || st->attn_desc);
    if (ok && W(e)->drafting) {
        st->mh = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
        st->mo = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
        st->mtop = pulsar_gpu_tensor_alloc(pulsar_qwen_draft_top_ws_bytes());
        st->mtp = (nh_step *)xcalloc(1, sizeof(nh_step));
        st->mtp->w = W(e);
        st->mtp->st = st;
        st->mtp->res = st->mh;
        st->mtp->module = &W(e)->mtp.block;
        ok = st->mh && st->mo && st->mtop;
    }
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %d tokens)\n", ARCH, n_banks,
                s->ctx_size);
        state_free(st);
        return NULL;
    }
    return st;
}

void destroy(pulsar_session *s) { state_free(ST(s)); }

/* the declared state's verify capture exists with the drafter (N9: the MTP module, when the session drafts) */
bool verifies(const pulsar_session *s) { return W(s->engine)->drafting; }

pulsar_driver_step *trunk(pulsar_session *s) { return ST(s)->trunk; }

uint64_t artifact(const pulsar_engine *e) { return W(e)->artifact_digest; }

const void *wp(const nh_step *st, const pulsar_tensor *t, const char *what) {
    return pulsar_weight_device_ptr(st->model, t, what);
}

/* ---- row kinds ----------------------------------------------------------------------------------------------------
 * L284 #2: rows [lo, hi) of an arm-sensitive launch, cut at n_dec -- the decode rows first (prompt = false), then the
 * prompt rows; an empty segment is skipped.  `launch(prompt, r0, m)` offsets its row pointers by r0. */
template <class Launch>
bool arm_segments(const nh_step *st, uint32_t lo, uint32_t hi, Launch &&launch) {
    const uint32_t cut = st->n_dec < lo ? lo : st->n_dec > hi ? hi : st->n_dec;
    return (cut == lo || launch(false, lo, cut - lo)) && (cut == hi || launch(true, cut, hi - cut));
}

/* n linears (1..3) of the same bf16 input x [n_rows][in] -> y[j] f32 [n_rows][out[j]] (row stride out[j]), each
 * row-kind segment through its arm (one multi launch) */
bool step_linears(const nh_step *st, int n, const pulsar_tensor *const *t, int in, const int *out,
                  const char *const *what, const uint16_t *x, float *const *y, void *ws, uint64_t ws_bytes) {
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_linear l[3];
        const pulsar_rows_linear *lp[3];
        float *yr[3];
        for (int j = 0; j < n; j++) {
            if (!pulsar_linear_rows_ref(st->model, t[j], in, out[j], prompt, what[j], &l[j])) return false;
            lp[j] = &l[j];
            yr[j] = y[j] + (size_t)r0 * (size_t)out[j];
        }
        return pulsar_rows_linear_multi_launch(lp, n, x + (size_t)r0 * in, (int)m, yr, ws, ws_bytes, 0) == 0;
    });
}

/* one linear's f32 output -> its bf16 rows (the producer-stage encode: the one bf16 rounding a bf16 GEMM's output
 * has) */
bool emit(const float *y, uint16_t *yb, uint32_t n, uint32_t width) {
    return pulsar_rows_emit_bf16_launch(y, width, n, width, NULL, PULSAR_ROWS_SCALE_NONE, yb, width, 0) == 0;
}

/* ---- the step's tables (stage) ---------------------------------------------------------------------------------- */

/* The step's sequences: a DECODE row is one; a PREFILL run is one (a verify's runs first, rows [0, n_dec)).  The
 * Mamba-2 block runs two calls when the step verifies -- the verify rows with their per-row captures, then the prompt
 * rows without (the captures hold PULSAR_STATE_VERIFY_ROWS rows) -- so each call's table is uploaded on its own. */
bool build_tables(const nh_step *st) {
    nh_state *q = st->st;
    const uint32_t n = st->n_rows;
    uint32_t ns = 0;
    if (st->mode == PULSAR_STEP_DECODE) {
        for (uint32_t r = 0; r < n; r++) {
            q->seq_first[ns] = r;
            q->seq_bank[ns++] = st->bank[r];
        }
    } else {
        for (uint32_t k = 0; k < st->n_runs; k++) {
            q->seq_first[ns] = st->run_first[k];
            q->seq_bank[ns++] = st->bank[st->run_first[k]];
        }
    }
    q->seq_first[ns] = n;
    q->n_seq = ns;
    for (uint32_t k = 0; k < ns; k++)
        for (uint32_t r = q->seq_first[k]; r < q->seq_first[k + 1]; r++) {
            q->row_seq[r] = k;
            q->row_pos[r] = (uint32_t)st->pos[r];
        }
    q->n_seq_cap = st->verify ? pulsar_step_verify_runs(st) : 0u;
    q->rows_cap = q->seq_first[q->n_seq_cap];
    /* call A: the captured sequences [0, n_seq_cap); call B: the rest, rows from rows_cap */
    const uint32_t na = q->n_seq_cap, nb = ns - na;
    q->max_run_a = q->max_run_b = 0;
    for (uint32_t k = 0; k <= na; k++) q->tab_a[k] = (int32_t)q->seq_first[k];
    for (uint32_t k = 0; k < na; k++) {
        q->tab_a[na + 1 + k] = q->seq_bank[k];
        const uint32_t m = q->seq_first[k + 1] - q->seq_first[k];
        if (m > q->max_run_a) q->max_run_a = m;
    }
    for (uint32_t k = 0; k <= nb; k++) q->tab_b[k] = (int32_t)(q->seq_first[na + k] - q->rows_cap);
    for (uint32_t k = 0; k < nb; k++) {
        q->tab_b[nb + 1 + k] = q->seq_bank[na + k];
        const uint32_t m = q->seq_first[na + k + 1] - q->seq_first[na + k];
        if (m > q->max_run_b) q->max_run_b = m;
    }
    pulsar_gpu_tensor *rt = q->scratch[ARENA_ROWS];
    const rows_scratch L = rows_layout(q->max_rows);
    static const uint32_t zero = 0;   /* L289: an enqueued write reads it later */
    bool ok = rt != NULL;
    ok = ok && pulsar_gpu_tensor_write_async(rt, L.ids, st->tokens, (uint64_t)n * 4u) != 0;
    ok = ok && pulsar_gpu_tensor_write_async(rt, L.nf, &zero, sizeof(zero)) != 0;
    if (ok && na) {
        ok = pulsar_gpu_tensor_write_async(rt, L.first_a, q->tab_a, (uint64_t)(na + 1u) * 4u) != 0 &&
             pulsar_gpu_tensor_write_async(rt, L.slot_a, q->tab_a + na + 1, (uint64_t)na * 4u) != 0;
    }
    if (ok && nb) {
        ok = pulsar_gpu_tensor_write_async(rt, L.first_b, q->tab_b, (uint64_t)(nb + 1u) * 4u) != 0 &&
             pulsar_gpu_tensor_write_async(rt, L.slot_b, q->tab_b + nb + 1, (uint64_t)nb * 4u) != 0;
    }
    return ok || fail("could not upload the step's tables");
}

/* N6, E5: the head's declaration -- no embedding norm, no logit scale, every one of the n_vocab ids real */
pulsar_head_decl head_decl(const pulsar_nemotron_h_shape &s) {
    pulsar_head_decl hd{};
    hd.n_embd = s.n_embd;
    hd.embed_norm_w = NULL;
    hd.embed_norm_eps = 0.0f;
    hd.n_logits = s.n_vocab;
    hd.n_real = s.n_vocab;
    hd.logit_scale = 1.0f;
    return hd;
}

/* L296 perf: every attention layer's F1 call planned (the step's sequences on the layer's pools), the descriptors
 * staged at this step's stride and uploaded in one write.  Bounded at the power of two (>= one 256-key chunk) above
 * the step's last position, capped at the pools' context: the launches then depend on the bound, not the positions
 * (pulsar_gqa_plan), and a decode graph is recaptured once per doubling of the context. */
uint32_t attn_bound(const nh_step *st) {
    const nh_state *q = st->st;
    uint32_t last = 0;
    for (uint32_t r = 0; r < st->n_rows; r++) last = q->row_pos[r] > last ? q->row_pos[r] : last;
    uint32_t bound = 256u;
    while (bound <= last && bound < st->states->ctx) bound *= 2u;
    return bound > st->states->ctx ? st->states->ctx : bound;
}

/* call k planned: slot il's pools (a trunk layer's, or the MTP module's drafter slot) under the step's sequences, its
 * descriptors staged at off (stride bytes) */
bool plan_call(const nh_step *st, uint32_t il, uint32_t k, uint32_t bound, uint64_t off, uint64_t stride) {
    nh_state *q = st->st;
    const pulsar_gqa_layer G1 = gqa_decl(st->w->shape);
    const uint64_t tb = pulsar_gqa_kv_token_bytes(&G1);
    for (uint32_t j = 0; j < q->n_seq; j++) {
        pulsar_gpu_tensor *pool = pulsar_state_pool(st->states, il, PULSAR_NEMOTRON_H_STATE_KV, (uint32_t)q->seq_bank[j]);
        if (!pool) return failf("an attention layer's pool is evicted", il);
        pulsar_gqa_seq &sq = q->seqs[j];
        sq.kv = dptr(pool);
        sq.bytes = pulsar_gpu_tensor_bytes(pool);
        sq.cap = (uint32_t)(sq.bytes / tb);
    }
    q->attn_off[k] = off;
    return pulsar_gqa_plan(&G1, q->seqs, q->n_seq, q->row_seq, q->row_pos, st->n_rows, bound, q->attn_desc_host + off,
                           stride, &q->attn_call[k]) == 0 ||
           failf("the attention's plan refused", il);
}

bool plan_attention(const nh_step *st, uint32_t *bound_out) {
    nh_state *q = st->st;
    *bound_out = 0;
    if (!q->n_attn) return true;
    const uint32_t bound = attn_bound(st);
    const uint64_t stride = a256(pulsar_gqa_desc_bytes(st->n_rows));
    for (uint32_t il = 0; il < st->plan->n_layer; il++) {
        const uint32_t k = q->attn_slot[il];
        if (k != UINT32_MAX && !plan_call(st, il, k, bound, (uint64_t)k * stride, stride)) return false;
    }
    *bound_out = bound;
    return pulsar_gpu_tensor_write_async(q->attn_desc, 0, q->attn_desc_host, (uint64_t)q->n_attn * stride) != 0 ||
           fail("could not upload the attention descriptors");
}

/* N9, the MTP module step's host half: its tables, and its one attention call (ordinal n_attn, past the trunk's
 * descriptors, so a captured trunk step's never move) -- the module's step runs eager */
bool mtp_stage(const nh_step *st, uint32_t il) {
    nh_state *q = st->st;
    for (uint32_t r = 0; r < st->n_rows; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= st->w->shape.n_vocab)
            return fail("a token id outside the vocabulary");
    const uint64_t off = (uint64_t)q->n_attn * q->attn_desc_max;
    return build_tables(st) && plan_call(st, il, q->n_attn, attn_bound(st), off, q->attn_desc_max) &&
           (pulsar_gpu_tensor_write_async(q->attn_desc, off, q->attn_desc_host + off, q->attn_desc_max) != 0 ||
            fail("could not upload the MTP attention's descriptors"));
}

/* L296 perf, the step's host half (pulsar_driver_ops::stage): the token ids checked, the step's tables and every
 * attention call's descriptors uploaded.  What the launches then depend on beyond the driver's key: the attention
 * bound (the row count and the head list are the driver's; a decode row is one sequence of one row, so the tables'
 * shapes follow from the row count). */
bool stage(const pulsar_driver_step *st0, uint64_t *graph_key) {
    const nh_step *st = S(st0);
    const pulsar_nemotron_h_shape &s = st->w->shape;
    *graph_key = 0;
    for (uint32_t r = 0; r < st->n_rows; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= s.n_vocab) return fail("a token id outside the vocabulary");
    uint32_t bound = 0;
    if (!build_tables(st) || !plan_attention(st, &bound)) return false;
    uint64_t key = 0x4E480000ull | ((uint64_t)bound << 20) | 1u;
    /* L296 fullspec: a verify step's launches also read its run structure -- the row-kind cut, the Mamba-2 calls'
     * sequence counts, row splits and longest runs, and every attention call's shape -- so they are its key too (a
     * decode step's follow from its row count: one sequence a row) */
    if (st->verify) {
        const nh_state *q = st->st;
        auto mix = [](uint64_t h, uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
            return h ^ (h >> 27);
        };
        key = mix(mix(mix(key, st->n_dec), mix(q->n_seq, q->n_seq_cap)), mix(q->rows_cap, mix(q->max_run_a, q->max_run_b)));
        for (uint32_t k = 0; k < q->n_attn; k++) {
            const pulsar_gqa_call &c = q->attn_call[k];
            key = mix(key, mix(mix(c.n_rows, c.n_tiles), mix(mix(c.grid_chunks, c.ctx_bound), (uint64_t)c.fold)));
        }
        key |= 1u;
    }
    *graph_key = key;
    return true;
}

/* tokens -> h: the gather from the ids stage uploaded (launch-only) */
bool embed(const pulsar_driver_step *st0) {
    const nh_step *st = S(st0);
    const pulsar_nemotron_h_shape &s = st->w->shape;
    const pulsar_head_decl hd = head_decl(s);
    const uint16_t *table = (const uint16_t *)wp(st, st->w->embed, "nemotron_h embed");
    const rows_scratch L = rows_layout(st->st->max_rows);
    if (!table) return false;
    return pulsar_head_embed_launch(&hd, table, s.n_vocab,
                                    (const int32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + L.ids), st->n_rows,
                                    (uint16_t *)dptr(st->res), s.n_embd, 0) == 0 ||
           fail("the embedding refused");
}

/* ---- the diagnostic taps (PULSAR_CUDA_GRAPH_DUMP_PREFIX: what the block grade reads) ------------------------------ */

/* a bf16 [n_rows][width] region of an arena, widened to f32 by the dump (named `name`, layer il) */
void dump_rows(const nh_step *st, const char *name, pulsar_gpu_tensor *arena, uint64_t off, uint32_t width,
               uint32_t il) {
    if (!gpu_graph_debug_dump_enabled()) return;
    const uint64_t n = (uint64_t)st->n_rows * width;
    pulsar_gpu_tensor *v = arena ? pulsar_gpu_tensor_view(arena, off, n * 2u) : NULL;
    if (v) gpu_graph_debug_dump_hc_tensor(name, v, n, il, (uint32_t)st->pos[st->n_rows - 1]);
    pulsar_gpu_tensor_free(v);
}

/* ---- the mixers ---------------------------------------------------------------------------------------------------- */

/* N2: the Mamba-2 mixer of xn into the delta db (bf16 [n][n_embd]): in_proj -> pulsar_mamba2_forward (the verify rows'
 * call with the SSM DELTA and conv SNAPSHOT captures, then the rest) -> out_proj */
bool mamba(const nh_step *st, uint32_t il, uint8_t *base, const mix_scratch &M) {
    nh_state *q = st->st;
    const pulsar_nemotron_h_shape &s = st->w->shape;
    const pulsar_nemotron_h_block &L = LW(st, il);
    const mamba_dims md = mamba_of(s);
    const uint32_t n = st->n_rows, E = s.n_embd;
    float *pf = (float *)(base + M.pf), *of = (float *)(base + M.of);
    uint16_t *pb = (uint16_t *)(base + M.pb), *yb = (uint16_t *)(base + M.yb), *db = (uint16_t *)(base + M.db);
    const uint16_t *xn = (const uint16_t *)dptr(q->xn);
    {
        const pulsar_tensor *t[1] = {L.ssm_in};
        const int outs[1] = {(int)md.in};
        const char *what[1] = {"nemotron_h ssm_in"};
        float *y[1] = {pf};
        if (!step_linears(st, 1, t, (int)E, outs, what, xn, y, base + M.lin, M.lin_bytes) || !emit(pf, pb, n, md.in))
            return failf("the Mamba-2 in_proj refused", il);
    }
    pulsar_mamba2_layer ml{};
    ml.geom = mamba_geom(s);
    ml.conv_w = (const uint16_t *)wp(st, L.ssm_conv, "nemotron_h ssm conv");
    ml.conv_b = (const uint16_t *)wp(st, L.ssm_conv_bias, "nemotron_h ssm conv bias");
    ml.A_log = (const uint16_t *)wp(st, L.ssm_a_log, "nemotron_h ssm A_log");
    ml.dt_bias = (const uint16_t *)wp(st, L.ssm_dt_bias, "nemotron_h ssm dt_bias");
    ml.D = (const uint16_t *)wp(st, L.ssm_d, "nemotron_h ssm D");
    ml.norm_w = (const uint16_t *)wp(st, L.ssm_out_norm, "nemotron_h ssm out norm");
    if (!ml.conv_w || !ml.conv_b || !ml.A_log || !ml.dt_bias || !ml.D || !ml.norm_w) return false;
    pulsar_gpu_tensor *ts = pulsar_state_tensor(st->states, il, PULSAR_NEMOTRON_H_STATE_SSM);
    pulsar_gpu_tensor *tc = pulsar_state_tensor(st->states, il, PULSAR_NEMOTRON_H_STATE_CONV);
    if (!ts || !tc) return failf("no Mamba-2 state", il);
    const rows_scratch R = rows_layout(q->max_rows);
    uint8_t *rt = (uint8_t *)dptr(q->scratch[ARENA_ROWS]);
    for (int call = 0; call < 2; call++) {
        const bool cap = call == 0;
        const uint32_t r0 = cap ? 0u : q->rows_cap, r1 = cap ? q->rows_cap : n;
        if (r1 == r0) continue;
        pulsar_mamba2_call c{};
        c.n_seqs = cap ? q->n_seq_cap : q->n_seq - q->n_seq_cap;
        c.n_rows = r1 - r0;
        c.max_seq_rows = cap ? q->max_run_a : q->max_run_b;
        c.seq_first = (const int32_t *)(rt + (cap ? R.first_a : R.first_b));
        c.seq_slot = (const int32_t *)(rt + (cap ? R.slot_a : R.slot_b));
        /* the in_proj row [z | x B C | dt]: strided column ranges */
        const uint16_t *row = pb + (size_t)r0 * md.in;
        c.z = row;
        c.ld_z = md.in;
        c.xbc = row + md.di;
        c.ld_xbc = md.in;
        c.dt = row + md.di + md.cc;
        c.ld_dt = md.in;
        c.conv_state = (uint16_t *)dptr(tc);
        c.ssm_state = (float *)dptr(ts);
        if (cap) {
            c.conv_capture = (uint16_t *)pulsar_state_capture_rows(st->states, il, PULSAR_NEMOTRON_H_STATE_CONV);
            c.ssm_capture = (float *)pulsar_state_capture_rows(st->states, il, PULSAR_NEMOTRON_H_STATE_SSM);
            if (!c.conv_capture || !c.ssm_capture) return failf("a verify step with no Mamba-2 capture", il);
        }
        c.scratch = base + M.m2;
        c.scratch_bytes = M.m2_bytes;
        c.out = yb + (size_t)r0 * md.di;
        c.ld_out = md.di;
        if (pulsar_mamba2_forward(&ml, &c, 0) != 0) return failf("the Mamba-2 block refused", il);
    }
    const pulsar_tensor *t[1] = {L.ssm_out};
    const int outs[1] = {(int)E};
    const char *what[1] = {"nemotron_h ssm_out"};
    float *y[1] = {of};
    if (!step_linears(st, 1, t, (int)md.di, outs, what, yb, y, base + M.lin, M.lin_bytes) || !emit(of, db, n, E))
        return failf("the Mamba-2 out_proj refused", il);
    return true;
}

/* N5: the attention mixer of xn into the delta db: q | k | v (one multi launch), F1 on the layer's pool, o_proj */
bool attention(const nh_step *st, uint32_t il, uint8_t *base, const mix_scratch &M) {
    nh_state *q = st->st;
    const pulsar_nemotron_h_shape &s = st->w->shape;
    const pulsar_nemotron_h_block &L = LW(st, il);
    const uint32_t n = st->n_rows, E = s.n_embd, Q = s.n_head * s.head_dim, KV = s.n_head_kv * s.head_dim;
    float *qf = (float *)(base + M.pf), *kf = qf + (size_t)q->max_rows * Q, *vf = kf + (size_t)q->max_rows * KV;
    uint16_t *qb = (uint16_t *)(base + M.pb), *kb = qb + (size_t)q->max_rows * Q, *vb = kb + (size_t)q->max_rows * KV;
    uint16_t *ab = (uint16_t *)(base + M.yb), *db = (uint16_t *)(base + M.db);
    float *of = (float *)(base + M.of);
    const uint16_t *xn = (const uint16_t *)dptr(q->xn);
    {
        const pulsar_tensor *t[3] = {L.attn_q, L.attn_k, L.attn_v};
        const int outs[3] = {(int)Q, (int)KV, (int)KV};
        const char *what[3] = {"nemotron_h q_proj", "nemotron_h k_proj", "nemotron_h v_proj"};
        float *y[3] = {qf, kf, vf};
        if (!step_linears(st, 3, t, (int)E, outs, what, xn, y, base + M.lin, M.lin_bytes))
            return failf("the q / k / v projections refused", il);
        if (!emit(qf, qb, n, Q) || !emit(kf, kb, n, KV) || !emit(vf, vb, n, KV))
            return failf("the q / k / v bf16 emit refused", il);
    }
    const pulsar_gqa_layer G1 = gqa_decl(s);
    const uint32_t slot = st->module ? q->n_attn : q->attn_slot[il];   /* N9: the MTP module's call is n_attn */
    if (slot == UINT32_MAX || q->attn_call[slot].n_rows != n) return failf("an attention layer the step did not plan", il);
    pulsar_gqa_io io{};
    io.q = {qb, Q, s.head_dim};
    io.k = {kb, KV, s.head_dim};
    io.v = {vb, KV, s.head_dim};
    io.out_bf16 = ab;
    io.ld_out = Q;
    io.workspace = base + M.gqa;
    io.workspace_bytes = M.gqa_bytes;
    if (pulsar_gqa_launch(&G1, &q->attn_call[slot], (const uint8_t *)dptr(q->attn_desc) + q->attn_off[slot],
                          &io, 0) != 0)
        return failf("the attention refused", il);
    const pulsar_tensor *t[1] = {L.attn_o};
    const int outs[1] = {(int)E};
    const char *what[1] = {"nemotron_h o_proj"};
    float *y[1] = {of};
    if (!step_linears(st, 1, t, (int)Q, outs, what, ab, y, base + M.lin, M.lin_bytes) || !emit(of, db, n, E))
        return failf("the o_proj refused", il);
    return true;
}

/* ---- the MoE ------------------------------------------------------------------------------------------------------ */

/* N4: the latent MoE of xn (the ffn_norm of h) into fo (f32): F2's router (512 / 22), the routed relu^2 experts on the
 * latent through the front door's latent wrapper, then the ungated relu^2 shared expert on the hidden added */
bool moe(const nh_step *st, uint32_t il, float *fo, uint8_t *base, const ffn_scratch &F) {
    const pulsar_nemotron_h_shape &s = st->w->shape;
    const pulsar_nemotron_h_block &L = LW(st, il);
    const uint32_t n = st->n_rows;
    const int E = (int)s.n_embd, Lt = (int)s.latent, SM = (int)s.n_ff_shared, K = (int)s.n_expert_used;
    const uint16_t *xn = (const uint16_t *)dptr(st->st->xn);
    const pulsar_moe_block_shape bs = moe_shape(s);
    pulsar_moe_block_parts parts;
    if (pulsar_moe_block_carve(&bs, (int)n, base + F.u, F.moe_bytes, &parts) != 0) return failf("the MoE workspace", il);
    /* the router: f32 logits over the routed rows (no gate rows), then the select */
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_router_layer rl{};
    rl.bias = (const float *)wp(st, L.router_select_bias, "nemotron_h router bias");
    const uint16_t *rw = (const uint16_t *)wp(st, L.router, "nemotron_h router");
    if (!rw || !rl.bias ||
        pulsar_router_logits_launch(xn, (int)n, E, rw, bs.n_logit_rows, NULL, 0, parts.logits, 0) != 0 ||
        pulsar_router_select_launch(&rc, &rl, parts.logits, NULL, (int)n, parts.sel, parts.wts, NULL, NULL, 0) != 0)
        return failf("the router refused", il);
    /* the routed experts through the core's front door (moe.cpp): up-only relu^2 on the latent, wrapped in the two
     * latent projections */
    const rows_scratch R = rows_layout(st->st->max_rows);
    pulsar_moe_rows_call c{};
    c.m = st->model;
    c.gate = NULL;
    c.up = L.exp_up;
    c.down = L.exp_down;
    c.act = PULSAR_FFN_ACT_RELU2;
    c.latent_in = L.latent_in;
    c.latent_out = L.latent_out;
    c.lat_ws = parts.lat_lin;
    c.lat_ws_bytes = parts.lat_lin_bytes;
    c.top_k = (uint32_t)K;
    c.ws = parts.routed;
    c.ws_bytes = parts.routed_bytes;
    c.nf_flag = (uint32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + R.nf);
    c.nf_code = 0x4E000000u | il;
    if (!arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            c.selected = parts.sel + (size_t)r0 * K;
            c.weights = parts.wts + (size_t)r0 * K;
            c.x_bf16 = xn + (size_t)r0 * E;
            c.n_rows = (int)rows;
            c.out = fo + (size_t)r0 * E;
            c.lat_f = parts.lat_f + (size_t)r0 * Lt;
            c.lat_x = parts.lat_x + (size_t)r0 * Lt;
            c.lat_y = parts.lat_y + (size_t)r0 * Lt;
            c.prompt = prompt;
            return pulsar_moe_routed_rows(&c);
        }))
        return failf("the routed experts refused", il);
    /* the shared expert: out += down(relu(up xn)^2), ungated */
    const bool ok = arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
        pulsar_rows_moe_shared q{};
        q.act = PULSAR_FFN_ACT_RELU2;
        q.scale = NULL;
        q.yu = parts.yu;
        q.ys = parts.ys;
        q.h = parts.h;
        q.ws = parts.lin;
        q.ws_bytes = parts.lin_bytes;
        return pulsar_linear_rows_ref(st->model, L.sh_up, E, SM, prompt, "nemotron_h shared up", &q.up) &&
               pulsar_linear_rows_ref(st->model, L.sh_down, SM, E, prompt, "nemotron_h shared down", &q.down) &&
               pulsar_rows_moe_shared_launch(&q, xn + (size_t)r0 * E, (int)rows, fo + (size_t)r0 * E, 0) == 0;
    });
    return ok || failf("the shared expert refused", il);
}

/* ---- the layer --------------------------------------------------------------------------------------------------- */

/* layer il's mixer pre-norm weight (a trunk layer) */
const uint16_t *mixer_norm(const nh_step *st, uint32_t il) {
    const pulsar_nemotron_h_block &L = st->w->layer[il];
    return (const uint16_t *)wp(st, st->plan->desc[il].mixer == PULSAR_MIXER_MAMBA2 ? L.ssm_pre_norm : L.attn_norm,
                                "nemotron_h mixer norm");
}
/* L296 perf: the norm whatever reads h after trunk layer il applies -- layer il + 1's mixer pre-norm, or the final
 * norm the head reads after the last layer.  Layer il's last add writes it into xn in the same launch. */
const uint16_t *next_norm(const nh_step *st, uint32_t il) {
    if (il + 1u < st->plan->n_layer) return mixer_norm(st, il + 1u);
    return (const uint16_t *)wp(st, st->w->output_norm, "nemotron_h norm_f");
}

/* One layer on the step's residual stream: the mixer's HF block, then the MoE's when it has one, each
 * `h += sublayer(rmsnorm(h))` -- a trunk layer as its description says, or the MTP module's {attention, MoE} (N9).
 * `pre`: the mixer's pre-norm when the layer norms its own input (trunk layer 0, the MTP module), NULL when the
 * previous layer's last add wrote it into xn (L296 perf).  The layer's last add also writes rmsnorm(h) * `nn` into
 * `nout` -- what reads h next (vLLM's fused add_rmsnorm; the norm of the rounded sum, rowops: the bytes of the add
 * then the norm). */
bool compose(const nh_step *st, uint32_t il, bool is_mamba, bool is_moe, const uint16_t *pre, const uint16_t *nn,
             uint16_t *nout) {
    nh_state *q = st->st;
    const pulsar_nemotron_h_shape &s = st->w->shape;
    const pulsar_nemotron_h_block &L = LW(st, il);
    const uint32_t n = st->n_rows, E = s.n_embd;
    uint16_t *h = (uint16_t *)dptr(st->res), *xn = (uint16_t *)dptr(q->xn);
    if (!nn || !nout) return failf("the next pre-norm's weight is missing", il);
    /* the mixer's HF block */
    if (pre && pulsar_rows_rmsnorm_launch(h, E, n, E, pre, s.rms_eps, xn, E, 0) != 0)
        return failf("the mixer norm refused", il);
    const mix_scratch M = mix_layout(q->max_rows, st->states->ctx);
    uint8_t *mbase = (uint8_t *)dptr(q->scratch[ARENA_MIX]);
    if (!(is_mamba ? mamba(st, il, mbase, M) : attention(st, il, mbase, M))) return false;
    dump_rows(st, "nemotron_mix", q->scratch[ARENA_MIX], M.db, E, il);
    /* h += delta, and the pre-norm of the new h in the same launch: the MoE's, else the next reader's */
    const uint16_t *fn = is_moe ? (const uint16_t *)wp(st, L.ffn_norm, "nemotron_h ffn norm") : nn;
    if (!fn || pulsar_rows_add_launch(h, E, (const uint16_t *)(mbase + M.db), E, n, E, fn, s.rms_eps, is_moe ? xn : nout, E,
                                    0) != 0)
        return failf("the mixer's residual add refused", il);
    if (!is_moe) return true;
    dump_rows(st, "nemotron_mid", st->res, 0, E, il);
    /* the MoE's HF block */
    const ffn_scratch F = ffn_layout(q->max_rows);
    uint8_t *fbase = (uint8_t *)dptr(q->scratch[ARENA_FFN]);
    float *fo = (float *)(fbase + F.fo);
    uint16_t *db = (uint16_t *)(fbase + F.db);
    if (!moe(st, il, fo, fbase, F)) return false;
    if (!emit(fo, db, n, E)) return failf("the MoE bf16 emit refused", il);
    dump_rows(st, "nemotron_ffn", q->scratch[ARENA_FFN], F.db, E, il);
    return pulsar_rows_add_launch(h, E, db, E, n, E, nn, s.rms_eps, nout, E, 0) == 0 ||
           failf("the MoE's residual add refused", il);
}

/* block[PULSAR_LAYER_BLOCKS]: one layer as its description says; its pre-norm is the previous layer's last add's
 * (L296 perf: next_norm), layer 0 norms the embedding */
bool block(const pulsar_driver_step *st0, uint32_t il) {
    const nh_step *st = S(st0);
    const pulsar_layer_desc &d = st0->plan->desc[il];
    const bool is_mamba = d.mixer == PULSAR_MIXER_MAMBA2, is_moe = d.ffn == PULSAR_FFN_MOE;
    if ((!is_mamba && d.mixer != PULSAR_MIXER_ATTN_FULL) || (!is_moe && d.ffn != PULSAR_FFN_NONE))
        return failf("a layer description that is not {Mamba-2 | full attention} x {MoE | none}", il);
    const uint16_t *pre = il == 0 ? mixer_norm(st, 0) : NULL;
    if (il == 0 && !pre) return failf("the mixer norm's weight is missing", il);
    return compose(st, il, is_mamba, is_moe, pre, next_norm(st, il), (uint16_t *)dptr(st->st->xn));
}

/* n normed bf16 rows xb -> logits rows [out0, out0 + n) of the driver's slab: the head linear (the decode arm at every
 * call: a row's logits never depend on the rows beside it), E5's finish (no scale, no padding: nothing launches).
 * The trunk's head, and the MTP module's -- the target's own (the checkpoint has no MTP head; vLLM shares it). */
bool head_rows(const nh_step *st, const uint16_t *xb, uint32_t n, uint32_t out0) {
    const pulsar_nemotron_h_shape &s = st->w->shape;
    if (n == 0 || out0 + n > PULSAR_DRIVER_HEADS_MAX) return fail("head rows out of range");
    const head_scratch H = head_layout();
    uint8_t *base = (uint8_t *)dptr(st->st->scratch[ARENA_HEAD]);
    pulsar_rows_linear l{};
    float *logits = (float *)dptr(st->buf->logits) + (size_t)out0 * s.n_vocab;
    if (!pulsar_linear_rows_ref(st->model, st->w->head, (int)s.n_embd, (int)s.n_vocab, false, "nemotron_h lm_head", &l) ||
        pulsar_rows_linear_launch(&l, xb, (int)n, logits, base + H.lin, H.lin_bytes, 0) != 0)
        return fail("the head linear refused");
    const pulsar_head_decl hd = head_decl(s);
    return pulsar_head_logits_finish_launch(&hd, logits, n, s.n_vocab, 0) == 0 || fail("the logits finish refused");
}

/* step rows [row0, row0 + n) -> logits rows [out0, out0 + n): the head over the final norm of h (the last layer's
 * last add wrote it into xn, next_norm) */
bool head(const pulsar_driver_step *st0, uint32_t row0, uint32_t n, uint32_t out0) {
    const nh_step *st = S(st0);
    if (n == 0 || row0 + n > st->n_rows) return fail("head rows out of range");
    return head_rows(st, (const uint16_t *)dptr(st->st->xn) + (size_t)row0 * st->w->shape.n_embd, n, out0);
}

/* the step's own checks (rule 9): the routed experts' non-finite flag */
bool step_end(const pulsar_driver_step *st0, bool ok) {
    if (!ok) return false;
    const nh_step *st = S(st0);
    uint32_t nf = 0;
    if (!pulsar_gpu_tensor_read(st->st->scratch[ARENA_ROWS], rows_layout(st->st->max_rows).nf, &nf, sizeof(nf)))
        return fail("could not read the MoE non-finite flag");
    if (nf) {
        fprintf(stderr, "pulsar: %s: the MoE output of layer %u went non-finite (code 0x%08x) -- refusing the step\n",
                ARCH, nf & 0xffffffu, nf);
        return false;
    }
    return true;
}

/* the diagnostic tap: the residual stream after layer il, the whole step's rows (bf16 widened to f32), named
 * "nemotron_h" -- what the block grade reads against the reference's hidden_states (the layer's last HF block) */
void dump(const pulsar_driver_step *st0, uint32_t il) {
    const nh_step *st = S(st0);
    gpu_graph_debug_dump_hc_tensor("nemotron_h", st->res, (uint64_t)st->n_rows * st->w->shape.n_embd, il,
                                   (uint32_t)st->pos[st->n_rows - 1]);
}

/* N3: the SSM state's DELTA commit -- the kept rows of a verify run applied with the forward's own operations
 * (pulsar_mamba2_commit: bit-identical to a forward over them).  The geometry is the profile's (the F3 hook has no
 * layer argument; every Mamba-2 layer is one geometry). */
bool ssm_commit(void *state, uint32_t bank, const void *rows, uint32_t keep) {
    const pulsar_mamba2_geom g = mamba_geom(G());
    return pulsar_mamba2_commit(&g, (float *)state, bank, (const float *)rows, keep, 0) == 0;
}

/* ---- N9: the MTP module, declared to the chained-MTP lane (spec_mtp.h) ---------------------------------------------
 * vLLM's NemotronHMultiTokenPredictor (models/nemotron_h_mtp.py), one module ({attention, MoE}) applied recursively:
 *     x = eh_proj([enorm(embed(t_{p+1})) | hnorm(c_p)])         (embed FIRST: nemotron_h_mtp.py:106-113)
 *     x += attn(rmsnorm(x));  x += moe(rmsnorm(x))              the trunk's composition, the module's own weights,
 *                                                                its K|V on the drafter's slot
 *     o = final_layernorm(x);  logits = lm_head(o)              the target's head (shared)
 * The conditioning row c_p is the target's hidden vLLM hands the proposer -- norm_f of the trunk's residual at p, the
 * head's input (nemotron_h.py:658) -- and deeper in a chain the module's own output o (the proposer feeds the draft
 * model's returned hidden back).  Every rounding is the trunk's at the same op: each norm, the eh_proj's one bf16
 * rounding, the residual adds. */

uint32_t mtp_modules(const pulsar_engine *e) { return W(e)->drafting ? 1u : 0u; }
uint64_t mtp_cond_bytes(const pulsar_engine *e) { return (uint64_t)W(e)->shape.n_embd * 2u; }
uint32_t mtp_head_width(const pulsar_engine *e) { return W(e)->shape.n_vocab; }

/* the last trunk step's rows [row0, row0 + n): norm_f of the residual (the target hidden) into dst at off */
bool mtp_trunk_rows(pulsar_session *s, uint32_t row0, uint32_t n, pulsar_gpu_tensor *dst, uint64_t off) {
    const nh_state *q = ST(s);
    const pulsar_nemotron_h_model *w = W(s->engine);
    const uint32_t E = w->shape.n_embd;
    const uint16_t *on = (const uint16_t *)pulsar_weight_device_ptr(&s->engine->model, w->output_norm, "nemotron_h norm_f");
    if (!on || row0 + n > q->max_rows || off + (uint64_t)n * E * 2u > pulsar_gpu_tensor_bytes(dst))
        return fail("the MTP conditioning rows are out of range");
    return pulsar_rows_rmsnorm_launch((const uint16_t *)dptr(q->h) + (size_t)row0 * E, E, n, E, on, w->shape.rms_eps,
                                      (uint16_t *)((uint8_t *)dptr(dst) + off), E, 0) == 0 ||
           fail("the MTP conditioning norm refused");
}

pulsar_gpu_tensor *mtp_out_rows(pulsar_session *s) { return ST(s)->mo; }

/* One module step (pulsar_mtp_ops::forward): the rows' runs (as the driver builds a step's), the module's input side,
 * its {attention, MoE} on the drafter slot, the out norm, the headed rows through the head and the device pick. */
bool mtp_forward(pulsar_session *s, const pulsar_mtp_step *m) {
    pulsar_engine *e = s->engine;
    nh_state *q = ST(s);
    const pulsar_nemotron_h_model *w = W(e);
    const pulsar_nemotron_h_shape &sh = w->shape;
    const uint32_t n = m->n, E = sh.n_embd, V = sh.n_vocab;
    if (!q->mtp || m->module != 0 || n == 0 || n > q->max_rows || m->head_n > n ||
        m->head_n > PULSAR_DRIVER_HEADS_MAX || (m->pick && m->head_n == 0) ||
        (m->decode_rows && (m->mode != PULSAR_STEP_PREFILL || n > PULSAR_DRIVER_HEADS_MAX)))
        return fail("an MTP step outside the module's rows (one module; its heads within the driver's)");
    /* the runs: a prompt step is one bank's consecutive rows; decode rows of a PREFILL step one run a bank */
    uint32_t run_first[PULSAR_DRIVER_HEADS_MAX + 1] = {0};
    uint32_t n_runs = 0;
    if (m->mode == PULSAR_STEP_PREFILL) {
        bool runs_ok = true;
        for (uint32_t r = 0; runs_ok && r < n; r++) {
            if (r > 0 && m->bank[r] == m->bank[r - 1]) {
                runs_ok = m->pos[r] == m->pos[r - 1] + 1;
                continue;
            }
            for (uint32_t k = 0; k < n_runs; k++) runs_ok &= m->bank[run_first[k]] != m->bank[r];
            runs_ok = runs_ok && (m->decode_rows ? n_runs < PULSAR_DRIVER_HEADS_MAX : n_runs == 0);
            if (runs_ok) run_first[n_runs++] = r;
        }
        if (!runs_ok) return fail("an MTP step's rows are not one run a bank");
        run_first[n_runs] = n;
    }
    nh_step &st = *q->mtp;
    st.model = &e->model;
    st.plan = &e->plan;
    st.states = s->states;
    st.buf = q;
    st.mode = m->mode;
    st.n_rows = n;
    st.tokens = m->tokens;
    st.pos = m->pos;
    st.bank = m->bank;
    st.rope = NULL;
    st.verify = false;
    st.n_runs = n_runs;
    st.run_first = n_runs ? run_first : NULL;
    st.n_dec = m->decode_rows ? n : pulsar_step_n_dec(m->mode, 0u, n);
    st.tap = NULL;
    const uint32_t il = e->state_plan.n_trunk + m->module;   /* the module's drafter slot */
    const pulsar_nemotron_h_draft &D = w->mtp;
    const mix_scratch M = mix_layout(q->max_rows, s->states->ctx);
    uint8_t *mbase = (uint8_t *)dptr(q->scratch[ARENA_MIX]);
    uint16_t *mh = (uint16_t *)dptr(q->mh), *mo = (uint16_t *)dptr(q->mo), *cat = (uint16_t *)(mbase + M.pb);
    float *of = (float *)(mbase + M.of);
    const uint16_t *en = (const uint16_t *)wp(&st, D.embed_norm, "nemotron_h mtp enorm");
    const uint16_t *hn = (const uint16_t *)wp(&st, D.hidden_norm, "nemotron_h mtp hnorm");
    const uint16_t *fn = (const uint16_t *)wp(&st, D.out_norm, "nemotron_h mtp final_layernorm");
    const uint16_t *an = (const uint16_t *)wp(&st, D.block.attn_norm, "nemotron_h mtp attn norm");
    bool ok = en && hn && fn && an && mtp_stage(&st, il) && pulsar_gpu_begin_commands() != 0;
    /* the token gather into mh, then [enorm(embed) | hnorm(cond)] -> eh_proj -> the residual */
    if (ok) ok = embed(&st);
    if (ok)
        ok = (pulsar_rows_rmsnorm_launch(mh, E, n, E, en, sh.rms_eps, cat, 2u * E, 0) == 0 &&
              pulsar_rows_rmsnorm_launch((const uint16_t *)pulsar_gpu_tensor_device_ptr(m->cond), E, n, E, hn, sh.rms_eps, cat + E, 2u * E,
                                         0) == 0) ||
             fail("the MTP input norms refused");
    if (ok) {
        const pulsar_tensor *t[1] = {D.in_proj};
        const int outs[1] = {(int)E};
        const char *what[1] = {"nemotron_h mtp eh_proj"};
        float *y[1] = {of};
        ok = (step_linears(&st, 1, t, 2 * (int)E, outs, what, cat, y, mbase + M.lin, M.lin_bytes) && emit(of, mh, n, E)) ||
             fail("the MTP eh_proj refused");
    }
    /* the module's {attention, MoE}; its last add writes final_layernorm(x) into mo */
    if (ok) ok = compose(&st, il, false, true, an, fn, mo);
    if (ok && gpu_graph_debug_dump_enabled()) dump_rows(&st, "nemotron_mtp", q->mo, 0, E, il);
    if (ok && m->head_n) ok = head_rows(&st, mo, m->head_n, 0);
    /* L289: the picks reduced on the device, behind the head (every headed row in one launch) */
    if (ok && m->pick)
        ok = pulsar_qwen_draft_top_launch((const float *)dptr(q->logits), (int)V, (int)m->head_n, dptr(q->mtop), 0) == 0 ||
             fail("the MTP draft pick refused");
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    ok = step_end(&st, ok);
    if (ok && m->pick) {
        pulsar_qwen_draft_top rec[PULSAR_DRIVER_HEADS_MAX];
        ok = pulsar_gpu_tensor_read(q->mtop, 0, rec, m->head_n * sizeof rec[0]) != 0 || fail("could not read the pick");
        for (uint32_t r = 0; ok && r < m->head_n; r++) {
            if (rec[r].nan || rec[r].index < 0 || (uint32_t)rec[r].index >= V) return fail("the MTP head's row holds a NaN");
            m->pick[r] = {rec[r].index, rec[r].conf};
        }
    }
    if (ok && m->head_n && m->logits)
        ok = pulsar_gpu_tensor_read(q->logits, 0, m->logits, (uint64_t)m->head_n * V * sizeof(float)) != 0 ||
             fail("could not read the MTP logits");
    return ok;
}

/* the sampled proposal q of a headed row: the full vocabulary under the request's params (the head is the target's) */
bool mtp_dist(pulsar_session *s, const float *row, float temperature, int top_k, float top_p, float min_p,
              pulsar_sample_dist *qd) {
    return row && pulsar_sample_dist_build(row, W(s->engine)->shape.n_vocab, temperature, top_k, top_p, min_p,
                                           &s->sample_scratch, qd);
}

const pulsar_mtp_ops k_mtp_ops = {
    /* .pend_decl     = */ PULSAR_NEMOTRON_H_STATE_MTP_PEND,
    /* .modules       = */ mtp_modules,
    /* .cond_bytes    = */ mtp_cond_bytes,
    /* .head_width    = */ mtp_head_width,
    /* .trunk_rows    = */ mtp_trunk_rows,
    /* .out_rows      = */ mtp_out_rows,
    /* .forward       = */ mtp_forward,
    /* .dist          = */ mtp_dist,
    /* .stage_save    = */ NULL,   /* the module's state is its K|V pool: position-addressed, nothing to undo */
    /* .stage_restore = */ NULL,
    /* .prefetch      = */ NULL,
    /* .row_exact     = */ true,   /* the trunk's blocks at their decode arms (M3: the drafter slot byte-exact) */
};

}  // namespace

/* ---- N3: the state declaration ------------------------------------------------------------------------------------ */

void pulsar_nemotron_h_state_plan(pulsar_engine *e) {
    const pulsar_nemotron_h_shape *s = &W(e)->shape;
    const pulsar_layer_plan *lp = &e->plan;
    pulsar_state_plan *p = &e->state_plan;
    *p = pulsar_state_plan{};
    p->name = "nemotron_h";
    p->resume_grid = kResumeGrid;
    p->split_invariant = true;
    p->ckpt_slots = kCkptSlots;
    const pulsar_mamba2_geom g = mamba_geom(*s);
    const pulsar_gqa_layer gd = gqa_decl(*s);
    pulsar_state_decl d[PULSAR_NEMOTRON_H_STATE_COUNT] = {};
    d[PULSAR_NEMOTRON_H_STATE_SSM] = {"ssm", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_DELTA, false, false, 0,
                                      pulsar_mamba2_ssm_state_floats(&g) * sizeof(float),
                                      pulsar_mamba2_upd_floats(&g) * sizeof(float), 0, ssm_commit, NULL, NULL};
    d[PULSAR_NEMOTRON_H_STATE_CONV] = {"ssm_conv", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_SNAPSHOT, false, false, 0,
                                       pulsar_mamba2_conv_state_elems(&g) * sizeof(uint16_t), 0, 0, NULL, NULL, NULL};
    d[PULSAR_NEMOTRON_H_STATE_KV] = {"kv", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1u,
                                     pulsar_gqa_kv_token_bytes(&gd), 0, 0, NULL, NULL, NULL};
    d[PULSAR_NEMOTRON_H_STATE_MTP_PEND] = {"mtp_pend", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_NONE, false, false, 0,
                                           mtp_cond_bytes(e), 0, 0, NULL, NULL, NULL};
    for (uint32_t id = 0; id < PULSAR_NEMOTRON_H_STATE_COUNT; id++) (void)pulsar_state_plan_add(p, d[id]);
    for (uint32_t il = 0; il < lp->n_layer; il++)
        p->slot[il] = lp->desc[il].mixer == PULSAR_MIXER_MAMBA2
                          ? 1u << PULSAR_NEMOTRON_H_STATE_SSM | 1u << PULSAR_NEMOTRON_H_STATE_CONV
                          : 1u << PULSAR_NEMOTRON_H_STATE_KV;
    p->n_trunk = lp->n_layer;
    p->n_slots = lp->n_layer;
    /* N9, the drafter's slot when the session drafts: the MTP module's attention K|V (one geometry with the trunk's)
     * and each bank's parked conditioning row; the chained-MTP lane's hooks keep its counters with the walk */
    if (W(e)->drafting) {
        p->slot[p->n_slots++] = 1u << PULSAR_NEMOTRON_H_STATE_KV | 1u << PULSAR_NEMOTRON_H_STATE_MTP_PEND;
        p->drafter_at = pulsar_mtp_drafter_at;
        p->drafter_frontier = pulsar_mtp_drafter_frontier;
    }
    p->session = 0;
}

/* The drafter's numbers, measured on the full K4 container (L296 fullspec, served single stream T0 over code / prose /
 * chat / tool-call / edit, tok/s over all five): a chain may run 7 deep and no adaptive controller moves that cap --
 * the stop rule (tau 0.7) and the priced depth (session_spec.cpp, the measured lane costs) end every chain sooner when
 * a further draft would not pay.  59.1 tok/s vs the card's 3..5 under the shared controller 57.3 (tool calls 72.1 ->
 * 80.1, edit 77.8 -> 83.4, code 52.2 -> 53.3, prose / chat flat); a cap of 3 55.1, 5 57.6, 8 / 10 / 12 58.6 / 58.6 /
 * 59.3; the controller at 3..7 / 2..6 / 4..8 58.3-58.9; tau 0.5 / 0.6 / 0.8 / 0.85 / off all lower.  Plain decode
 * 28.1. */
const pulsar_drafter_ops PULSAR_NEMOTRON_H_DRAFTER =
    PULSAR_MTP_DRAFTER_OPS("MTP", 7u, PULSAR_MTP_DRAFT_MAX, 0.7f, NULL);

const pulsar_driver_ops PULSAR_NEMOTRON_H_DRIVER = {
    /* .create        = */ create,
    /* .destroy       = */ destroy,
    /* .verifies      = */ verifies,
    /* .trunk         = */ trunk,
    /* .scratch_bytes = */ scratch_bytes,
    /* .n_scratch     = */ ARENA_COUNT,
    /* .embed         = */ embed,
    /* .block         = */ {NULL, NULL, NULL, NULL, block},   /* by pulsar_layer_kind: BLOCKS */
    /* .head          = */ head,
    /* .step_end      = */ step_end,
    /* .absorb        = */ pulsar_mtp_absorb,   /* N9: the chained-MTP lane's lockstep (nothing without a lane) */
    /* .mtp           = */ &k_mtp_ops,          /* N9: the MTP module (spec_mtp.h) */
    /* .rope          = */ NULL,   /* no positional encoding anywhere (NoPE) */
    /* .image_rows    = */ NULL,   /* images: N11 */
    /* .dump          = */ dump,
    /* .artifact      = */ artifact,
    /* .stage         = */ stage,   /* L296 perf: the host half; the ops are launch-only (a decode step graphs) */
};
static_assert(PULSAR_LAYER_BLOCKS == 4 && PULSAR_LAYER_KIND_COUNT == 5, "the block table's slot is the kind's value");

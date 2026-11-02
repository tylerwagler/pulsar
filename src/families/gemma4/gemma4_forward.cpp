/* gemma4_forward.cpp -- the Gemma 4 forward on the core row-step driver (L338 E4; step_driver.h).
 *
 * The family declares; the core runs.  What is Gemma 4's here is only the composition -- which shared blocks a layer
 * runs, in which order, on which state, with which declaration.  Every kernel is a shared block's:
 *
 *   embed      E5   pulsar_head_embed_launch: the gather times bf16(sqrt(hidden)) = 53 (embed_scale) -> h (bf16)
 *   per layer  (pulsar_layer_desc: window or full attention, PULSAR_FFN_DENSE_MOE)
 *       xa = rmsnorm(h) * input_layernorm      (layer 0; then the previous layer's last add-scale)       row ops
 *     attention: q | k [| v] = bf16(xa W)       (one multi launch; a global layer has no v_proj: K = V)  rows front door
 *               v = rmsnorm_noweight(v_proj out, or k_proj out on a global layer), per KV head      row ops (ones w)
 *               a = GQA(q, k, v): qk RMSNorm (weights), rope (window: NeoX 1e4; global: the proportional table,
 *                   64 of 256 pairs at 1e6^(-2p/512), the rest 0), scale 1, window 1024 / full     F1 (the wide heads)
 *               delta = bf16(a W_o);  dn = rmsnorm(delta) * post_attention_layernorm
 *               h += dn  [; xd = rmsnorm(h) * pre_feedforward_layernorm, the same launch]           row ops
 *     FFN:      xm = rmsnorm(h) * pre_feedforward_layernorm_2;  xr = rmsnorm(h) * router.scale       row ops
 *               d1 = rmsnorm(bf16(down(gelu_tanh(gate xd) * up xd))) * post_feedforward_layernorm_1  the rows shared
 *                   (the dense MLP: the MoE block's shared-expert machinery, its own input and norm)  producer, GeGLU
 *               router: bf16 logits of xr x hidden^-1/2 (logit_scale), softmax, top-8 of 128, sum
 *                   renormalised, x per_expert_scale (expert_scale)                                  F2
 *               d2 = rmsnorm(bf16(sum_k w_k down_k(gelu_tanh(gate_k xm) * up_k xm))) * post_ffn_ln_2  rows MoE front
 *                                                                                                     door, EXL3 pair
 *               s = rmsnorm(bf16(d1 + d2)) * post_feedforward_layernorm                             row ops
 *               h = bf16(bf16(h + s) * layer_scalar)  [; xa = rmsnorm(h) * the next reader's norm]  add-scale
 *   head       xa (output_norm) -> the pack's EXL3 head -> E5 finish: soft-cap 30 (tanh(x / 30) * 30)
 *
 * HF's Gemma4TextDecoderLayer (modeling_gemma4.py) is the order; every sublayer output, norm and add rounds to bf16
 * where HF's bf16 tensors do.  Where pulsar rounds differently, it is said here and graded (rule 7):
 *   - the f32 linears' one bf16 rounding (= a bf16 GEMM's output);
 *   - the router input: rmsnorm(r) * scale in one rounding (HF rounds the norm, then the product, then times
 *     hidden^-1/2 in bf16); hidden^-1/2 moves to the f32 logits (W (x c) = c (W x)), the logits then rounded to bf16
 *     (HF's bf16 projection output);
 *   - the routed weight folded into each expert's GeGLU row before its bf16 rounding (F7's fold), the routed sum kept
 *     f32 into one bf16 rounding; HF multiplies each bf16 expert output by its f32 weight and index-adds in bf16.
 *
 * ROW KINDS (L284 #2): every arm-sensitive launch (the linears, the routed experts) runs rows [0, n_dec) on the decode
 * arms and [n_dec, n_rows) on the prompt arms; the router, the row ops and F1 are one arithmetic at any row count.
 *
 * STATE (state_kinds.h; gemma4.h PULSAR_GEMMA4_STATE_*): a window layer's K|V ring (RING of W - 1 + verify rows
 * slots), a global layer's K|V pool (POOL).  The core allocates, prices, pages, walks and commits it. */
/** @file gemma4_forward.cpp
 *  @brief The Gemma 4 forward on the core row-step driver: attention (window / global, K = V), the dense GeGLU MLP
 *  beside the routed experts, the head, and the state declaration. */
#include "gemma4.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_qwen.h"     /* the rows front door's launchers + the rows MoE block (family-neutral) */
#include "cuda/pulsar_cuda_router.h"
#include "cuda/pulsar_cuda_rowops.h"
#include "cuda/pulsar_cuda_f16_rows.h"   /* L338: the runtime LoRA (pulsar_rows_lora_launch) */

#include <math.h>

namespace {

/// The architecture name every refusal is prefixed with.
const char *const ARCH = PULSAR_GEMMA4_ARCH;

/// The state plan's resume grid, in tokens (pulsar_state_plan::resume_grid; the full pools are one token a row).
constexpr uint32_t kResumeGrid = 128u;
/* A checkpoint walks every window layer's ring whole: 25 x 1055 slots x 8 KiB = 211 MiB a checkpoint on the model */
/// The state plan's checkpoint slots (pulsar_state_plan::ckpt_slots).
constexpr uint32_t kCkptSlots = 2u;

/** The driver's scratch arenas, an index into pulsar_driver_state::scratch; layouts rows_layout / mix_layout /
 *  ffn_layout / head_layout; ARENA_COUNT is pulsar_driver_ops::n_scratch. */
enum g4_arena { ARENA_ROWS = 0, ARENA_MIX, ARENA_FFN, ARENA_HEAD, ARENA_COUNT };

/** @param x  a byte count
 *  @return x rounded up to a multiple of 256 (every arena region's alignment) */
uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }
/** @param t  a device tensor
 *  @return its device base address */
void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }
/** @param a  a value
 *  @param b  a value
 *  @return the larger */
uint64_t umax(uint64_t a, uint64_t b) { return a > b ? a : b; }

/** Say a step refusal.
 *  @param what  the reason
 *  @return always false */
bool fail(const char *what) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the step\n", ARCH, what);
    return false;
}
/** Say a step refusal at a layer.
 *  @param what  the reason
 *  @param il    the layer
 *  @return always false */
bool failf(const char *what, uint32_t il) {
    fprintf(stderr, "pulsar: %s: %s (layer %u) -- refusing the step\n", ARCH, what, il);
    return false;
}

/* ---- the declarations every op reads ------------------------------------------------------------------------------
 * The scratch is laid out from the shape before any session exists (pulsar_driver_ops::scratch_bytes has no engine);
 * the geometry IS the profile's (gemma4_shape.cpp pins every geometry field to PULSAR_GEMMA4_SHAPE_26B_A4B). */
/** @return the pinned profile, the geometry every scratch layout is sized from */
const pulsar_gemma4_shape &G() { return PULSAR_GEMMA4_SHAPE_26B_A4B; }

/** The bf16 value nearest x (round to nearest even), as a float: what torch's `.to(bfloat16)` of an f32 gives.
 *  @param x  an f32 value
 *  @return bf16(x) widened back */
float bf16_round_host(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

/* F1's declaration of a layer: window (8 kv x 256, NeoX rope 1e4, a 1024-key window) or global (2 kv x 512, the
 * proportional rope's table, full); q / k RMS-normed with the layer's weights (eps 1e-6), scale 1 (HF's
 * self.scaling), no bias, temperature or gate.  The weights and the table are device pointers (NULL in a plan-only
 * declaration: the plan reads the geometry, never the weights). */
/** @param s      the shape
 *  @param full   a global layer
 *  @param qw     the layer's q_norm weight (bf16 [head_dim], device), or NULL
 *  @param kw     the layer's k_norm weight, or NULL
 *  @param table  the proportional rope's per-pair table (device f32 [256]) for a global layer, else NULL
 *  @return F1's declaration of the layer */
pulsar_gqa_layer gqa_decl(const pulsar_gemma4_shape &s, bool full, const uint16_t *qw, const uint16_t *kw,
                          const float *table) {
    pulsar_gqa_layer L{};
    L.n_q = s.n_head;
    L.n_kv = full ? s.n_head_kv : s.swa_n_head_kv;
    L.head_dim = full ? s.head_dim : s.swa_head_dim;
    L.kv_fmt = PULSAR_GQA_KV_BF16;
    L.span = full ? PULSAR_GQA_FULL : PULSAR_GQA_WINDOW;
    L.window = full ? 0u : s.window;
    L.scale = 1.0f;
    L.qk_norm = 1;
    L.norm_eps = s.rms_eps;
    L.q_norm_w = qw;
    L.k_norm_w = kw;
    L.rope_theta = full ? s.rope_theta : s.swa_rope_theta;
    L.rope_inv_freq = full ? table : NULL;
    return L;
}

/* F2: softmax over the bf16 logits of the router input times hidden^-1/2, top-8 of 128, renormalised by the sum (no
 * floor), each pick times its expert's per_expert_scale (the layer's expert_scale) -- HF's Gemma4TextRouter */
/** @param s  the shape
 *  @return F2's router configuration for every layer */
pulsar_router_cfg router_cfg(const pulsar_gemma4_shape &s) {
    pulsar_router_cfg c{};
    c.n_expert = (int)s.n_expert;
    c.top_k = (int)s.n_expert_used;
    c.n_shared = 0;
    c.shared_gate = PULSAR_ROUTER_SHARED_NONE;
    c.score = PULSAR_ROUTER_SCORE_SOFTMAX;
    c.renorm = PULSAR_ROUTER_RENORM_SUM;
    c.logits_bf16 = true;
    c.weights_bf16 = false;
    c.sum_floor = 0.0f;
    c.route_scale = 1.0f;
    c.logit_scale = (float)(1.0 / sqrt((double)s.n_embd));
    return c;
}

/* The rows MoE block's shape: the routed experts at their stored width, and ONE "shared expert" whose buffers carry the
 * dense MLP (its own input xd and its own output norm: it is never added into the routed sum) */
/** @param s  the shape
 *  @return the rows MoE block's shape (its workspace is sized and carved by it) */
pulsar_moe_block_shape moe_shape(const pulsar_gemma4_shape &s) {
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_moe_block_shape b{};
    b.n_expert = rc.n_expert;
    b.top_k = rc.top_k;
    b.n_logit_rows = pulsar_router_logit_rows(&rc);
    b.hidden = (int)s.n_embd;
    b.expert_mid = (int)pulsar_gemma4_ff_exp_stored(&s);
    b.n_shared = 1;
    b.shared_mid = (int)pulsar_gemma4_ff_stored(&s);
    b.latent = 0;
    return b;
}

/* E5: no embedding norm, the embedding times bf16(sqrt(hidden)) (HF: the f32 sqrt cast to the weight dtype -- 53),
 * no logit scale, every id real, the soft-cap */
/** @param s  the shape
 *  @return E5's head declaration (the embedding gather and the logits finish read it) */
pulsar_head_decl head_decl(const pulsar_gemma4_shape &s) {
    pulsar_head_decl hd{};
    hd.n_embd = s.n_embd;
    hd.n_logits = s.n_vocab;
    hd.n_real = s.n_vocab;
    hd.logit_scale = 1.0f;
    hd.softcap = s.softcap;
    hd.embed_scale = bf16_round_host(sqrtf((float)s.n_embd));
    return hd;
}

/* The dense-arm workspace of n linears of one input (a multi launch) or of one linear in -> out */
/** @param in    the shared input width
 *  @param n     the linears, 1..3
 *  @param out   their output widths
 *  @param rows  the row count
 *  @return the multi launch's workspace bytes, 256-aligned */
uint64_t multi_ws(int in, int n, const int *out, uint32_t rows) {
    pulsar_rows_linear l[3] = {};
    const pulsar_rows_linear *lp[3];
    for (int j = 0; j < n; j++) {
        l[j].in = in;
        l[j].out = out[j];
        lp[j] = &l[j];
    }
    return a256(pulsar_rows_linear_multi_workspace_bytes(lp, n, (int)rows));
}
/** @param in    the input width
 *  @param out   the output width
 *  @param rows  the row count
 *  @return one linear's workspace bytes, 256-aligned */
uint64_t lin_ws(int in, int out, uint32_t rows) {
    pulsar_rows_linear l{};
    l.in = in;
    l.out = out;
    return a256(pulsar_rows_linear_workspace_bytes(&l, (int)rows));
}

/* ---- scratch layouts: one function each, for sizing AND carving -------------------------------------------------- */

/** The ROWS arena: int32 [R] token ids, the routed experts' non-finite flag (zeroed by every stage), u8 [R] the rows
 *  the runtime adapter applies to (L338 P2: uploaded by a stage whose banks run it). */
struct rows_scratch { uint64_t ids, nf, ad, total; };
/** @param R  the step's row capacity (max_rows)
 *  @return the ROWS arena's offsets and size */
rows_scratch rows_layout(uint32_t R) {
    rows_scratch r{};
    uint64_t o = 0;
    r.ids = o; o += a256((uint64_t)R * 4u);
    r.nf = o;  o += 256u;
    r.ad = o;  o += a256(R);
    r.total = o;
    return r;
}

/** The MIX arena (attention), sized for the wider of the two layer kinds:
 *  qf / kf / vf f32 and qb / kb / vb bf16 the projections (kb also v's source on a global layer), vn bf16 the
 *  unweighted-normed v, ab bf16 the attention's output, of f32 / db bf16 the o_proj's, dn bf16 its post-norm; the
 *  dense arm's workspace; F1's workspace (the wider declaration's). */
struct mix_scratch { uint64_t qf, kf, vf, qb, kb, vb, vn, ab, of, db, dn, lora, lora_bytes, lin, lin_bytes, gqa, gqa_bytes, total; };
/** @param R    the step's row capacity (max_rows)
 *  @param ctx  the pools' capacity in tokens (F1's workspace; 0 = sized for 1)
 *  @return the MIX arena's offsets and size */
mix_scratch mix_layout(uint32_t R, uint32_t ctx) {
    const pulsar_gemma4_shape &s = G();
    const uint64_t E = s.n_embd;
    const uint64_t Qw = (uint64_t)s.n_head * s.swa_head_dim, KVw = (uint64_t)s.swa_n_head_kv * s.swa_head_dim;
    const uint64_t Qf = (uint64_t)s.n_head * s.head_dim, KVf = (uint64_t)s.n_head_kv * s.head_dim;
    const uint64_t Q = umax(Qw, Qf), KV = umax(KVw, KVf);
    mix_scratch t{};
    uint64_t o = 0;
    t.qf = o; o += a256((uint64_t)R * Q * 4u);
    t.kf = o; o += a256((uint64_t)R * KV * 4u);
    t.vf = o; o += a256((uint64_t)R * KV * 4u);
    t.qb = o; o += a256((uint64_t)R * Q * 2u);
    t.kb = o; o += a256((uint64_t)R * KV * 2u);
    t.vb = o; o += a256((uint64_t)R * KV * 2u);
    t.vn = o; o += a256((uint64_t)R * KV * 2u);
    t.ab = o; o += a256((uint64_t)R * Q * 2u);
    t.of = o; o += a256((uint64_t)R * E * 4u);
    t.db = o; o += a256((uint64_t)R * E * 2u);
    t.dn = o; o += a256((uint64_t)R * E * 2u);
    t.lora = o;   /* L338: the runtime LoRA's workspace, its widest projection (q / o of a global layer) */
    t.lora_bytes = pulsar_rows_lora_workspace_bytes((int)R, (int)umax(E, Q), PULSAR_GEMMA4_LORA_RANK_MAX, (int)umax(E, Q));
    o += a256(t.lora_bytes);
    const int qkv[3] = {(int)Qw, (int)KVw, (int)KVw}, qk[2] = {(int)Qf, (int)KVf};
    t.lin = o;
    t.lin_bytes = umax(umax(multi_ws((int)E, 3, qkv, R), multi_ws((int)E, 2, qk, R)),
                       umax(lin_ws((int)Qw, (int)E, R), lin_ws((int)Qf, (int)E, R)));
    o += t.lin_bytes;
    const pulsar_gqa_layer gw = gqa_decl(s, false, NULL, NULL, NULL), gf = gqa_decl(s, true, NULL, NULL, NULL);
    const uint32_t c = ctx ? ctx : 1u;
    t.gqa = o;
    t.gqa_bytes = a256(umax(pulsar_gqa_workspace_bytes(&gw, R, c), pulsar_gqa_workspace_bytes(&gf, R, c)));
    o += t.gqa_bytes;
    t.total = o;
    return t;
}

/** The FFN arena: xd / xm / xr bf16 the three normed inputs (dense MLP, experts, router); fo f32 the routed sum; e1 /
 *  e2 bf16 the dense MLP's and the experts' rows; d1 / d2 bf16 their post-norms; then the MoE block's carved
 *  workspace (router, routed, the dense MLP's buffers as the block's one shared expert). */
struct ffn_scratch { uint64_t xd, xm, xr, fo, e1, e2, d1, d2, lora, lora_bytes, u, moe_bytes, total; };
/** @param R  the step's row capacity (max_rows)
 *  @return the FFN arena's offsets and size */
ffn_scratch ffn_layout(uint32_t R) {
    const pulsar_gemma4_shape &s = G();
    const uint64_t E = s.n_embd;
    ffn_scratch f{};
    uint64_t o = 0;
    for (uint64_t *x : {&f.xd, &f.xm, &f.xr}) { *x = o; o += a256((uint64_t)R * E * 2u); }
    f.fo = o; o += a256((uint64_t)R * E * 4u);
    for (uint64_t *x : {&f.e1, &f.e2, &f.d1, &f.d2}) { *x = o; o += a256((uint64_t)R * E * 2u); }
    f.lora = o;   /* L338: the dense MLP's LoRA workspace (E -> n_ff -> E) */
    f.lora_bytes = pulsar_rows_lora_workspace_bytes((int)R, (int)umax(E, s.n_ff), PULSAR_GEMMA4_LORA_RANK_MAX,
                                                    (int)umax(E, s.n_ff));
    o += a256(f.lora_bytes);
    const pulsar_moe_block_shape bs = moe_shape(s);
    f.u = o; f.moe_bytes = a256(pulsar_moe_block_workspace_bytes(&bs, (int)R)); o += f.moe_bytes;
    f.total = o;
    return f;
}

/** The HEAD arena: the head linear's workspace at PULSAR_DRIVER_HEADS_MAX rows. */
struct head_scratch { uint64_t lin, lin_bytes, total; };
/** @return the HEAD arena's offsets and size */
head_scratch head_layout() {
    const pulsar_gemma4_shape &s = G();
    head_scratch h{};
    h.lin = 0;
    h.lin_bytes = lin_ws((int)s.n_embd, (int)s.n_vocab, PULSAR_DRIVER_HEADS_MAX);
    h.total = h.lin + h.lin_bytes;
    return h;
}

/** pulsar_driver_ops::scratch_bytes.
 *  @param arena     a g4_arena
 *  @param max_rows  the step's row capacity
 *  @param ctx       the pools' capacity in tokens
 *  @return the arena's bytes (0 for an arena out of range) */
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

struct g4_step;
/** The family's session state on the driver's (pulsar_driver_state is its base): built by create, freed by
 *  state_free.  The residual stream, the normed rows the next block reads, the device constants the blocks read (the
 *  proportional rope's table, the ones weight of the unweighted norms, every layer's per_expert_scale in f32), the
 *  trunk's step object, and the step's host tables and attention plans (rebuilt by every stage). */
struct g4_state : pulsar_driver_state {
    /// bf16 [max_rows][n_embd]: the residual stream
    pulsar_gpu_tensor *h;
    /// bf16 [max_rows][n_embd]: the normed rows the next layer's attention (or the head) reads
    pulsar_gpu_tensor *xa;
    /// f32 [head_dim / 2]: the global layers' proportional rope table (pulsar_gqa_layer::rope_inv_freq)
    pulsar_gpu_tensor *rope_tab;
    /// bf16 [head_dim] of 1.0: the weight of the unweighted RMSNorms (v_norm) -- bit-exact (x * rstd * 1)
    pulsar_gpu_tensor *ones;
    /// f32 [n_layer][n_expert]: each layer's per_expert_scale (pulsar_router_layer::expert_scale)
    pulsar_gpu_tensor *escale;
    /// The trunk's step object (pulsar_driver_ops::trunk); owned
    g4_step *trunk;
    /// The step's sequences (build_tables): one a decode row, one a prefill run
    uint32_t n_seq;
    /// Host [max_rows + 1]: each sequence's first row, then n_rows
    uint32_t *seq_first;
    /// Host [max_rows]: each sequence's bank
    int32_t *seq_bank;
    /// Host [max_rows]: each row's sequence (F1's)
    uint32_t *row_seq;
    /// Host [max_rows]: each row's position (F1's)
    uint32_t *row_pos;
    /// Host [max_rows]: F1's sequences for the call being planned
    pulsar_gqa_seq *seqs;
    /// Host [n_layer]: each layer's attention call's plan (every layer attends)
    pulsar_gqa_call *attn_call;
    /// Host [n_layer]: each call's descriptors' byte offset in attn_desc(_host)
    uint64_t *attn_off;
    /// Host [n_layer * attn_desc_max]: the descriptors, staged
    uint8_t *attn_desc_host;
    /// Device, attn_desc_host's layout
    pulsar_gpu_tensor *attn_desc;
    /// One call's descriptor bytes at max_rows, 256-aligned
    uint64_t attn_desc_max;
    /// Host [max_rows]: L338 P2 -- 1 where the row's bank runs the runtime adapter (stage builds it; rows_scratch::ad)
    uint8_t *row_adapter;
    /// The staged step has rows on adapter banks: the LoRAs launch (stage sets it; part of the graph key)
    bool step_adapter;
    /// bf16 [n_embd]: L338 P2 -- the final-norm hidden of the last row the head read (the decision head's input)
    pulsar_gpu_tensor *last_hidden;
    /// The head has run since create: last_hidden holds a row
    bool headed;
};

/** A step object: the driver's step (its base) plus the family's view -- the weights and the session state. */
struct g4_step : pulsar_driver_step {
    /// The family's weights and shape
    const pulsar_gemma4_model *w;
    /// The session state this step runs on
    g4_state *st;
};

/** @param st  a step this family's driver ops were handed
 *  @return it as the family's step */
const g4_step *S(const pulsar_driver_step *st) { return static_cast<const g4_step *>(st); }
/** @param s  a session of this family
 *  @return its family state */
g4_state *ST(pulsar_session *s) { return static_cast<g4_state *>(s->driver); }
/** @param e  an engine this family loaded
 *  @return its pulsar_gemma4_model */
const pulsar_gemma4_model *W(const pulsar_engine *e) { return (const pulsar_gemma4_model *)e->family_state; }

/** Free the family state and everything it owns.
 *  @param st  the state (NULL = nothing) */
void state_free(g4_state *st) {
    if (!st) return;
    for (pulsar_gpu_tensor *t : {st->h, st->xa, st->rope_tab, st->ones, st->escale, st->attn_desc, st->last_hidden})
        pulsar_gpu_tensor_free(t);
    free(st->seq_first);
    free(st->seq_bank);
    free(st->row_seq);
    free(st->row_pos);
    free(st->seqs);
    free(st->attn_call);
    free(st->attn_off);
    free(st->attn_desc_host);
    free(st->row_adapter);
    free(st->trunk);
    free(st);
}

/* The device constants: the proportional rope's table (HF's _compute_proportional_rope_parameters: 64 of the 256 pairs
 * at 1 / theta^(2p / 512), the rest 0 -- computed in double, rounded once to f32), the ones weight, the f32
 * per_expert_scale of every layer (the bf16 weights widened, exact). */
/** @param e   the engine (its model and weights)
 *  @param st  the state whose constants are written
 *  @return true = uploaded */
bool constants(const pulsar_engine *e, g4_state *st) {
    const pulsar_gemma4_model *w = W(e);
    const pulsar_gemma4_shape &s = w->shape;
    const uint32_t P = s.head_dim / 2u, live = (uint32_t)(s.rope_partial * (float)s.head_dim / 2.0f);
    std::vector<float> tab(P, 0.0f);
    for (uint32_t p = 0; p < live; p++) tab[p] = (float)(1.0 / pow((double)s.rope_theta, (double)(2u * p) / (double)s.head_dim));
    std::vector<uint16_t> ones(s.head_dim, (uint16_t)0x3f80u);
    std::vector<float> es((size_t)s.n_layer * s.n_expert);
    for (uint32_t il = 0; il < s.n_layer; il++) {
        const uint16_t *src = (const uint16_t *)tensor_data(&e->model, w->layer[il].router_expert_scale);
        if (!src) return false;
        for (uint32_t x = 0; x < s.n_expert; x++) {
            const uint32_t u = (uint32_t)src[x] << 16;
            memcpy(&es[(size_t)il * s.n_expert + x], &u, 4);
        }
    }
    return pulsar_gpu_tensor_write(st->rope_tab, 0, tab.data(), (uint64_t)P * 4u) &&
           pulsar_gpu_tensor_write(st->ones, 0, ones.data(), (uint64_t)s.head_dim * 2u) &&
           pulsar_gpu_tensor_write(st->escale, 0, es.data(), (uint64_t)es.size() * 4u);
}

/** pulsar_driver_ops::create: the residual and normed rows, the device constants, the host tables, the attention
 *  descriptors (one call a layer: every layer attends).
 *  @param s        the session (prefill_cap rows are the step's capacity)
 *  @param n_banks  the session's banks (named in the failure message only)
 *  @return the state, or NULL (said, nothing left allocated) */
pulsar_driver_state *create(pulsar_session *s, uint32_t n_banks) {
    const pulsar_engine *e = s->engine;
    const pulsar_gemma4_shape &sh = W(e)->shape;
    g4_state *st = (g4_state *)xcalloc(1, sizeof(*st));
    const uint64_t R = s->prefill_cap, NL = e->plan.n_layer;
    st->h = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->xa = pulsar_gpu_tensor_alloc(R * sh.n_embd * 2u);
    st->rope_tab = pulsar_gpu_tensor_alloc((uint64_t)sh.head_dim / 2u * 4u);
    st->ones = pulsar_gpu_tensor_alloc((uint64_t)sh.head_dim * 2u);
    st->escale = pulsar_gpu_tensor_alloc(NL * sh.n_expert * 4u);
    st->seq_first = (uint32_t *)xmalloc((R + 1u) * sizeof(uint32_t));
    st->seq_bank = (int32_t *)xmalloc(R * sizeof(int32_t));
    st->row_seq = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->row_pos = (uint32_t *)xmalloc(R * sizeof(uint32_t));
    st->row_adapter = (uint8_t *)xcalloc(R, 1);
    st->seqs = (pulsar_gqa_seq *)xcalloc(R, sizeof(pulsar_gqa_seq));
    st->trunk = (g4_step *)xcalloc(1, sizeof(g4_step));
    st->trunk->w = W(e);
    st->trunk->st = st;
    st->attn_desc_max = a256(pulsar_gqa_desc_bytes((uint32_t)R));
    st->attn_call = (pulsar_gqa_call *)xcalloc(NL, sizeof(pulsar_gqa_call));
    st->attn_off = (uint64_t *)xcalloc(NL, sizeof(uint64_t));
    st->attn_desc_host = (uint8_t *)xmalloc(NL * st->attn_desc_max);
    st->attn_desc = pulsar_gpu_tensor_alloc(NL * st->attn_desc_max);
    st->last_hidden = pulsar_gpu_tensor_alloc((uint64_t)sh.n_embd * 2u);
    const bool ok = st->h && st->xa && st->rope_tab && st->ones && st->escale && st->attn_desc && st->last_hidden &&
                    constants(e, st);
    if (!ok) {
        fprintf(stderr, "pulsar: %s: session state allocation failed (%u banks x %d tokens)\n", ARCH, n_banks,
                s->ctx_size);
        state_free(st);
        return NULL;
    }
    return st;
}

/** pulsar_driver_ops::destroy.
 *  @param s  the session; its family state is freed */
void destroy(pulsar_session *s) { state_free(ST(s)); }

/** pulsar_driver_ops::verifies: no drafter yet (L338 P3), so no verify capture.
 *  @return false */
bool verifies(const pulsar_session *) { return false; }

/** pulsar_driver_ops::trunk.
 *  @param s  the session
 *  @return the trunk's step object */
pulsar_driver_step *trunk(pulsar_session *s) { return ST(s)->trunk; }

/** pulsar_driver_ops::adapter (L338 P2).
 *  @param e  the engine
 *  @return the container carries the runtime adapter */
bool has_adapter(const pulsar_engine *e) { return W(e)->adapter; }

/** pulsar_driver_ops::artifact.
 *  @param e  the engine
 *  @return the artifact digest load computed */
uint64_t artifact(const pulsar_engine *e) { return W(e)->artifact_digest; }

/** @param st    the step (its model)
 *  @param t     a weight
 *  @param what  its name, for a refusal
 *  @return its device copy, or NULL */
const void *wp(const g4_step *st, const pulsar_tensor *t, const char *what) {
    return pulsar_weight_device_ptr(st->model, t, what);
}

/* ---- row kinds ----------------------------------------------------------------------------------------------------
 * L284 #2: rows [lo, hi) of an arm-sensitive launch, cut at n_dec -- the decode rows first (prompt = false), then the
 * prompt rows; an empty segment is skipped.  `launch(prompt, r0, m)` offsets its row pointers by r0. */
/** @tparam Launch  a callable bool(bool prompt, uint32_t r0, uint32_t m)
 *  @param st      the step (its n_dec)
 *  @param lo      the first row
 *  @param hi      one past the last row
 *  @param launch  runs one segment
 *  @return true when every non-empty segment launched */
template <class Launch>
bool arm_segments(const g4_step *st, uint32_t lo, uint32_t hi, Launch &&launch) {
    const uint32_t cut = st->n_dec < lo ? lo : st->n_dec > hi ? hi : st->n_dec;
    return (cut == lo || launch(false, lo, cut - lo)) && (cut == hi || launch(true, cut, hi - cut));
}

/* n linears (1..3) of the same bf16 input x [n_rows][in] -> their bf16 rows yb[j] [n_rows][out[j]], each row-kind
 * segment through its arm (one multi launch), the producer's one bf16 rounding */
/** @param st        the step
 *  @param n         the linears, 1..3
 *  @param t         the n weights
 *  @param in        the input width
 *  @param out       the n output widths
 *  @param what      the n names a refusal reports
 *  @param x         the bf16 input rows, [n_rows][in]
 *  @param y         the n f32 stagings, [n_rows][out[j]]
 *  @param yb        the n bf16 outputs, [n_rows][out[j]]
 *  @param ws        the dense arm's workspace
 *  @param ws_bytes  its bytes
 *  @return true = launched */
bool step_linears(const g4_step *st, int n, const pulsar_tensor *const *t, int in, const int *out,
                  const char *const *what, const uint16_t *x, float *const *y, uint16_t *const *yb, void *ws,
                  uint64_t ws_bytes) {
    return arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        pulsar_rows_linear l[3];
        const pulsar_rows_linear *lp[3];
        float *yr[3];
        uint16_t *br[3];
        for (int j = 0; j < n; j++) {
            if (!pulsar_linear_rows_ref(st->model, t[j], in, out[j], prompt, what[j], &l[j])) return false;
            lp[j] = &l[j];
            yr[j] = y[j] + (size_t)r0 * (size_t)out[j];
            br[j] = yb[j] + (size_t)r0 * (size_t)out[j];
        }
        return pulsar_rows_linear_multi_bf16_launch(lp, n, x + (size_t)r0 * in, (int)m, yr, br, ws, ws_bytes, 0) == 0;
    });
}

/* one f32 output -> its bf16 rows (the producer-stage encode) */
/** @param y      the f32 rows, [n][width]
 *  @param yb     their bf16 rows, [n][width]
 *  @param n      the row count
 *  @param width  the row width
 *  @return true = launched */
bool emit(const float *y, uint16_t *yb, uint32_t n, uint32_t width) {
    return pulsar_rows_emit_bf16_launch(y, width, n, width, NULL, PULSAR_ROWS_SCALE_NONE, yb, width, 0) == 0;
}

/* x -> y = rmsnorm(x) * w, n rows of width (the shape's eps) */
/** @param x  bf16 rows
 *  @param w  the bf16 norm weight
 *  @param y  bf16 output rows
 *  @param n  the row count
 *  @param width  the row width
 *  @return true = launched */
bool norm(const uint16_t *x, const uint16_t *w, uint16_t *y, uint32_t n, uint32_t width) {
    return w && pulsar_rows_rmsnorm_launch(x, width, n, width, w, G().rms_eps, y, width, 0) == 0;
}

/* ---- the step's tables (stage) ---------------------------------------------------------------------------------- */

/* The step's sequences: a DECODE row is one, a PREFILL run is one; the token ids and the non-finite flag uploaded */
/** @param st  the step
 *  @return true = the tables are built and their uploads enqueued; false = refused (said) */
bool build_tables(const g4_step *st) {
    g4_state *q = st->st;
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
    pulsar_gpu_tensor *rt = q->scratch[ARENA_ROWS];
    const rows_scratch L = rows_layout(q->max_rows);
    static const uint32_t zero = 0;   /* L289: an enqueued write reads it later */
    const bool ok = rt != NULL && pulsar_gpu_tensor_write_async(rt, L.ids, st->tokens, (uint64_t)n * 4u) != 0 &&
                    pulsar_gpu_tensor_write_async(rt, L.nf, &zero, sizeof(zero)) != 0;
    return ok || fail("could not upload the step's tables");
}

/* The attention bound: the power of two (>= one 256-key chunk) above the step's last position, capped at the pools'
 * context -- the launches then depend on the bound, not the positions (pulsar_gqa_plan) */
/** @param st  the step
 *  @return the attention bound in tokens */
uint32_t attn_bound(const g4_step *st) {
    const g4_state *q = st->st;
    uint32_t last = 0;
    for (uint32_t r = 0; r < st->n_rows; r++) last = q->row_pos[r] > last ? q->row_pos[r] : last;
    uint32_t bound = 256u;
    while (bound <= last && bound < st->states->ctx) bound *= 2u;
    return bound > st->states->ctx ? st->states->ctx : bound;
}

/* layer il's F1 call planned: the step's sequences on the layer's ring (a window layer: bank b's slots of the decl's
 * bank-major tensor) or pool (a global layer), its descriptors staged at off */
/** @param st      the step
 *  @param il      the layer
 *  @param bound   the attention bound in tokens
 *  @param off     the descriptors' byte offset in attn_desc_host
 *  @param stride  one call's descriptor bytes
 *  @return true = planned; false = a pool is evicted or the plan refused (said) */
bool plan_call(const g4_step *st, uint32_t il, uint32_t bound, uint64_t off, uint64_t stride) {
    g4_state *q = st->st;
    const bool full = pulsar_gemma4_is_full(&st->w->shape, il);
    const pulsar_gemma4_block &L = st->w->layer[il];
    const pulsar_gqa_layer G1 = gqa_decl(st->w->shape, full, (const uint16_t *)wp(st, L.attn_q_norm, "gemma4 q_norm"),
                                         (const uint16_t *)wp(st, L.attn_k_norm, "gemma4 k_norm"),
                                         (const float *)dptr(q->rope_tab));
    const uint64_t tb = pulsar_gqa_kv_token_bytes(&G1);
    pulsar_gpu_tensor *ring = full ? NULL : pulsar_state_tensor(st->states, il, PULSAR_GEMMA4_STATE_KV_WIN);
    const uint64_t ring_bank = full ? 0 : pulsar_state_decl_bank_bytes(&st->states->plan->decl[PULSAR_GEMMA4_STATE_KV_WIN]);
    if (!full && !ring) return failf("a window layer's ring is missing", il);
    for (uint32_t j = 0; j < q->n_seq; j++) {
        pulsar_gqa_seq &sq = q->seqs[j];
        const uint32_t bank = (uint32_t)q->seq_bank[j];
        if (full) {
            pulsar_gpu_tensor *pool = pulsar_state_pool(st->states, il, PULSAR_GEMMA4_STATE_KV_FULL, bank);
            if (!pool) return failf("a global layer's pool is evicted", il);
            sq.kv = dptr(pool);
            sq.bytes = pulsar_gpu_tensor_bytes(pool);
        } else {
            sq.kv = (uint8_t *)dptr(ring) + (uint64_t)bank * ring_bank;
            sq.bytes = ring_bank;
        }
        sq.cap = (uint32_t)(sq.bytes / tb);
    }
    q->attn_off[il] = off;
    return pulsar_gqa_plan(&G1, q->seqs, q->n_seq, q->row_seq, q->row_pos, st->n_rows, bound, q->attn_desc_host + off,
                           stride, &q->attn_call[il]) == 0 ||
           failf("the attention's plan refused", il);
}

/** pulsar_driver_ops::stage: the token ids checked, the step's tables and every layer's attention call staged and
 *  uploaded; the ops are launch-only after it.
 *  @param st0        the trunk's step
 *  @param graph_key  receives the key a captured graph of the step is reused under (0 on refusal)
 *  @return true = staged; false = refused (said) */
bool stage(const pulsar_driver_step *st0, uint64_t *graph_key) {
    const g4_step *st = S(st0);
    g4_state *q = st->st;
    *graph_key = 0;
    for (uint32_t r = 0; r < st->n_rows; r++)
        if (st->tokens[r] < 0 || (uint32_t)st->tokens[r] >= st->w->shape.n_vocab) return fail("a token id outside the vocabulary");
    if (!build_tables(st)) return false;
    const uint32_t bound = attn_bound(st);
    const uint64_t stride = a256(pulsar_gqa_desc_bytes(st->n_rows));
    for (uint32_t il = 0; il < st->plan->n_layer; il++)
        if (!plan_call(st, il, bound, (uint64_t)il * stride, stride)) return false;
    if (pulsar_gpu_tensor_write_async(q->attn_desc, 0, q->attn_desc_host, (uint64_t)st->plan->n_layer * stride) == 0)
        return fail("could not upload the attention descriptors");
    /* L338 P2: the rows whose bank runs the runtime adapter (the LoRAs launch on them only) */
    q->step_adapter = false;
    for (uint32_t r = 0; r < st->n_rows; r++) {
        q->row_adapter[r] = q->bank_adapter && q->bank_adapter[st->bank[r]] ? 1u : 0u;
        q->step_adapter |= q->row_adapter[r] != 0;
    }
    if (q->step_adapter && pulsar_gpu_tensor_write_async(q->scratch[ARENA_ROWS], rows_layout(q->max_rows).ad,
                                                         q->row_adapter, st->n_rows) == 0)
        return fail("could not upload the adapter rows");
    /* the key: the "G4" tag in bits 48..63, bit 47 the LoRAs launched (L338 P2: which rows they touch is the mask's,
     * read at replay), the bound in bits 1..32, bit 0 set (a key is nonzero); a decode step's shapes follow from the
     * row count (one sequence a row) */
    if (bound > st->states->ctx) return fail("an attention bound past the pools' context");
    *graph_key = (0x4734ull << 48) | ((uint64_t)q->step_adapter << 47) | ((uint64_t)bound << 1) | 1u;
    return true;
}

/* tokens -> h: the scaled gather from the ids stage uploaded */
/** pulsar_driver_ops::embed.
 *  @param st0  the step
 *  @return true = launched; false = refused (said) */
bool embed(const pulsar_driver_step *st0) {
    const g4_step *st = S(st0);
    const pulsar_gemma4_shape &s = st->w->shape;
    const pulsar_head_decl hd = head_decl(s);
    const uint16_t *table = (const uint16_t *)wp(st, st->w->embed, "gemma4 embed");
    const rows_scratch L = rows_layout(st->st->max_rows);
    if (!table) return false;
    return pulsar_head_embed_launch(&hd, table, s.n_vocab,
                                    (const int32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + L.ids), st->n_rows,
                                    (uint16_t *)dptr(st->st->h), s.n_embd, 0) == 0 ||
           fail("the embedding refused");
}

/* ---- the runtime adapter (L338 P2) ---------------------------------------------------------------------------------- */

/* one projection's LoRA on its f32 output, on the step's adapter rows (rows_scratch::ad) */
/** @param st        the step (its model, adapter rows and row kinds)
 *  @param l         the projection's pair
 *  @param x         the projection's bf16 input rows
 *  @param ld_x      their pitch
 *  @param in        the LoRA's input width
 *  @param y         the f32 output rows
 *  @param ld_y      their pitch
 *  @param out       the LoRA's output width
 *  @param ws        the LoRA's workspace
 *  @param ws_bytes  its bytes
 *  @return true = launched (each row-kind segment on its arm) */
bool lora_rows(const g4_step *st, const pulsar_gemma4_lora &l, const uint16_t *x, uint64_t ld_x, uint32_t in, float *y,
               uint64_t ld_y, uint32_t out, void *ws, uint64_t ws_bytes) {
    pulsar_rows_lora d{};
    d.A = wp(st, l.a, "gemma4 lora_A");
    d.B = wp(st, l.b, "gemma4 lora_B");
    d.r = st->w->lora_r;
    d.in = in;
    d.out = out;
    d.scale = st->w->lora_scale;
    const uint8_t *rows = (const uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + rows_layout(st->st->max_rows).ad;
    return d.A && d.B && arm_segments(st, 0, st->n_rows, [&](bool prompt, uint32_t r0, uint32_t m) {
        return pulsar_rows_lora_launch(x + (size_t)r0 * ld_x, ld_x, (int)m, &d, y + (size_t)r0 * ld_y, ld_y, rows + r0,
                                       prompt, ws, ws_bytes, 0) == 0;
    });
}

/* ---- the attention ------------------------------------------------------------------------------------------------- */

/* xa -> M.dn: q | k [| v], v's unweighted norm (v_proj's, or k_proj's on a global layer: K = V), F1, o_proj, its
 * post-norm */
/** @param st    the step
 *  @param il    the layer
 *  @param base  the MIX arena's device base
 *  @param M     the MIX arena's layout
 *  @return true = launched (the normed delta in M.dn); false = refused (said) */
bool attention(const g4_step *st, uint32_t il, uint8_t *base, const mix_scratch &M) {
    g4_state *q = st->st;
    const pulsar_gemma4_shape &s = st->w->shape;
    const pulsar_gemma4_block &L = st->w->layer[il];
    const bool full = pulsar_gemma4_is_full(&s, il);
    const uint32_t n = st->n_rows, E = s.n_embd, hd = full ? s.head_dim : s.swa_head_dim;
    const uint32_t nkv = full ? s.n_head_kv : s.swa_n_head_kv, Q = s.n_head * hd, KV = nkv * hd;
    float *qf = (float *)(base + M.qf), *kf = (float *)(base + M.kf), *vf = (float *)(base + M.vf);
    uint16_t *qb = (uint16_t *)(base + M.qb), *kb = (uint16_t *)(base + M.kb), *vb = (uint16_t *)(base + M.vb);
    uint16_t *vn = (uint16_t *)(base + M.vn), *ab = (uint16_t *)(base + M.ab), *db = (uint16_t *)(base + M.db);
    const uint16_t *xa = (const uint16_t *)dptr(q->xa);
    {
        const pulsar_tensor *t[3] = {L.attn_q, L.attn_k, L.attn_v};
        const int outs[3] = {(int)Q, (int)KV, (int)KV};
        const char *what[3] = {"gemma4 q_proj", "gemma4 k_proj", "gemma4 v_proj"};
        float *y[3] = {qf, kf, vf};
        uint16_t *yb16[3] = {qb, kb, vb};
        if (!step_linears(st, full ? 2 : 3, t, (int)E, outs, what, xa, y, yb16, base + M.lin, M.lin_bytes))
            return failf("the q / k / v projections refused", il);
        /* L338 P2: the adapter's LoRAs on the f32 projections, then their bf16 rows again (a plain row re-rounds to the
         * same bits); on a global layer k's carries into v (K = V: v is k_proj's output) */
        if (q->step_adapter) {
            const pulsar_gemma4_lora *lp[3] = {&L.lora_q, &L.lora_k, &L.lora_v};
            for (int j = 0; j < (full ? 2 : 3); j++)
                if (!lora_rows(st, *lp[j], xa, E, E, y[j], (uint64_t)outs[j], (uint32_t)outs[j], base + M.lora,
                               M.lora_bytes) ||
                    !emit(y[j], yb16[j], n, (uint32_t)outs[j]))
                    return failf("the q / k / v LoRAs refused", il);
        }
    }
    /* v = v_norm(v_pre) per KV head, no weight (the ones vector: bit-exact); v_pre is k_proj's output on a global
     * layer (HF: value_states = key_states before k_norm when attention_k_eq_v) */
    if (!norm(full ? kb : vb, (const uint16_t *)dptr(q->ones), vn, n * nkv, hd)) return failf("the v norm refused", il);
    const pulsar_gqa_layer G1 = gqa_decl(s, full, (const uint16_t *)wp(st, L.attn_q_norm, "gemma4 q_norm"),
                                         (const uint16_t *)wp(st, L.attn_k_norm, "gemma4 k_norm"),
                                         (const float *)dptr(q->rope_tab));
    if (q->attn_call[il].n_rows != n) return failf("an attention layer the step did not plan", il);
    pulsar_gqa_io io{};
    io.q = {qb, Q, hd};
    io.k = {kb, KV, hd};
    io.v = {vn, KV, hd};
    io.out_bf16 = ab;
    io.ld_out = Q;
    io.workspace = base + M.gqa;
    io.workspace_bytes = M.gqa_bytes;
    if (pulsar_gqa_launch(&G1, &q->attn_call[il], (const uint8_t *)dptr(q->attn_desc) + q->attn_off[il], &io, 0) != 0)
        return failf("the attention refused", il);
    const pulsar_tensor *t[1] = {L.attn_o};
    const int outs[1] = {(int)E};
    const char *what[1] = {"gemma4 o_proj"};
    float *y[1] = {(float *)(base + M.of)};
    uint16_t *yb16[1] = {db};
    if (!step_linears(st, 1, t, (int)Q, outs, what, ab, y, yb16, base + M.lin, M.lin_bytes))
        return failf("the o_proj refused", il);
    if (q->step_adapter && (!lora_rows(st, L.lora_o, ab, Q, Q, y[0], E, E, base + M.lora, M.lora_bytes) ||
                            !emit(y[0], db, n, E)))   /* L338 P2 */
        return failf("the o_proj LoRA refused", il);
    return norm(db, (const uint16_t *)wp(st, L.attn_post_norm, "gemma4 post_attention_layernorm"),
                (uint16_t *)(base + M.dn), n, E) ||
           failf("the post-attention norm refused", il);
}

/* ---- the FFN: the dense MLP and the routed experts side by side ----------------------------------------------------- */

/* F.xd (the dense MLP's input, written by the attention's add) and h -> F.d1 + F.d2 summed and normed into F.e1:
 * the dense GeGLU MLP and its post-norm, the router on xr, the routed GeGLU experts on xm and their post-norm, the sum
 * and its norm */
/** @param st    the step
 *  @param il    the layer
 *  @param base  the FFN arena's device base
 *  @param F     the FFN arena's layout
 *  @return true = launched (post_feedforward_layernorm(d1 + d2) in F.e1); false = refused (said) */
bool ffn(const g4_step *st, uint32_t il, uint8_t *base, const ffn_scratch &F) {
    const pulsar_gemma4_shape &s = st->w->shape;
    const pulsar_gemma4_block &L = st->w->layer[il];
    const uint32_t n = st->n_rows, E = s.n_embd, K = s.n_expert_used;
    const int Fd = (int)pulsar_gemma4_ff_stored(&s);
    const uint16_t *h = (const uint16_t *)dptr(st->st->h);
    uint16_t *xd = (uint16_t *)(base + F.xd), *xm = (uint16_t *)(base + F.xm), *xr = (uint16_t *)(base + F.xr);
    uint16_t *e1 = (uint16_t *)(base + F.e1), *e2 = (uint16_t *)(base + F.e2);
    uint16_t *d1 = (uint16_t *)(base + F.d1), *d2 = (uint16_t *)(base + F.d2);
    float *fo = (float *)(base + F.fo);
    /* the experts' and the router's inputs (the dense MLP's, xd, came with the attention's add) */
    if (!norm(h, (const uint16_t *)wp(st, L.moe_pre_norm, "gemma4 pre_feedforward_layernorm_2"), xm, n, E) ||
        !norm(h, (const uint16_t *)wp(st, L.router_scale, "gemma4 router.scale"), xr, n, E))
        return failf("the FFN input norms refused", il);
    const pulsar_moe_block_shape bs = moe_shape(s);
    pulsar_moe_block_parts parts;
    if (pulsar_moe_block_carve(&bs, (int)n, base + F.u, F.moe_bytes, &parts) != 0) return failf("the MoE workspace", il);
    /* the dense MLP: down(gelu_tanh(gate xd) * up xd), per row-kind segment, through the block's shared-expert buffers */
    /* L338 P2: the adapter's LoRAs on the dense MLP's three Linears (the shared producer adds them to its f32 outputs) */
    const uint8_t *ad_rows = (const uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + rows_layout(st->st->max_rows).ad;
    pulsar_rows_lora lg{}, lu{}, ld{};
    if (st->st->step_adapter) {
        const pulsar_gemma4_lora *src[3] = {&L.lora_gate, &L.lora_up, &L.lora_down};
        pulsar_rows_lora *dst[3] = {&lg, &lu, &ld};
        for (int j = 0; j < 3; j++) {
            dst[j]->A = wp(st, src[j]->a, "gemma4 mlp lora_A");
            dst[j]->B = wp(st, src[j]->b, "gemma4 mlp lora_B");
            dst[j]->r = st->w->lora_r;
            dst[j]->in = j < 2 ? E : s.n_ff;
            dst[j]->out = j < 2 ? s.n_ff : E;
            dst[j]->scale = st->w->lora_scale;
            if (!dst[j]->A || !dst[j]->B) return failf("the MLP LoRA's weights are missing", il);
        }
    }
    const bool dense_ok = arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
        pulsar_rows_moe_shared qd{};
        if (st->st->step_adapter) {
            qd.lora_gate = &lg;
            qd.lora_up = &lu;
            qd.lora_down = &ld;
            qd.lora_rows = ad_rows + r0;
            qd.lora_ws = base + F.lora;
            qd.lora_ws_bytes = F.lora_bytes;
        }
        qd.act = PULSAR_FFN_ACT_GEGLU_TANH;
        qd.scale = NULL;
        qd.yg = parts.yg + (size_t)r0 * Fd;
        qd.yu = parts.yu + (size_t)r0 * Fd;
        qd.ys = parts.ys + (size_t)r0 * E;
        qd.h = parts.h + (size_t)r0 * Fd;
        qd.ws = parts.lin;
        qd.ws_bytes = parts.lin_bytes;
        return pulsar_linear_rows_ref(st->model, L.ffn_gate, (int)E, Fd, prompt, "gemma4 mlp gate", &qd.gate) &&
               pulsar_linear_rows_ref(st->model, L.ffn_up, (int)E, Fd, prompt, "gemma4 mlp up", &qd.up) &&
               pulsar_linear_rows_ref(st->model, L.ffn_down, Fd, (int)E, prompt, "gemma4 mlp down", &qd.down) &&
               pulsar_rows_moe_shared_ys_launch(&qd, xd + (size_t)r0 * E, (int)rows, 0) == 0;
    });
    if (!dense_ok || !emit(parts.ys, e1, n, E) ||
        !norm(e1, (const uint16_t *)wp(st, L.ffn_post_norm_1, "gemma4 post_feedforward_layernorm_1"), d1, n, E))
        return failf("the dense MLP refused", il);
    /* the router: bf16 logits of xr (x hidden^-1/2), softmax, top-8, renormalised, x per_expert_scale */
    const pulsar_router_cfg rc = router_cfg(s);
    pulsar_router_layer rl{};
    rl.expert_scale = (const float *)dptr(st->st->escale) + (size_t)il * s.n_expert;
    const uint16_t *rw = (const uint16_t *)wp(st, L.router, "gemma4 router");
    if (!rw || pulsar_router_logits_launch(xr, (int)n, (int)E, rw, bs.n_logit_rows, NULL, 0, parts.logits, NULL) != 0 ||
        pulsar_router_select_launch(&rc, &rl, parts.logits, NULL, (int)n, parts.sel, parts.wts, NULL, NULL, NULL) != 0)
        return failf("the router refused", il);
    /* the routed GeGLU experts on xm through the core's front door (moe.cpp): the EXL3 gate + up pair and down */
    const rows_scratch R = rows_layout(st->st->max_rows);
    pulsar_moe_rows_call c{};
    c.m = st->model;
    c.gate = L.exp_gate;
    c.up = L.exp_up;
    c.down = L.exp_down;
    c.act = PULSAR_FFN_ACT_GEGLU_TANH;
    c.top_k = K;
    c.ws = parts.routed;
    c.ws_bytes = parts.routed_bytes;
    c.nf_flag = (uint32_t *)((uint8_t *)dptr(st->st->scratch[ARENA_ROWS]) + R.nf);
    c.nf_code = 0x47000000u | il;
    if (!arm_segments(st, 0, n, [&](bool prompt, uint32_t r0, uint32_t rows) {
            c.selected = parts.sel + (size_t)r0 * K;
            c.weights = parts.wts + (size_t)r0 * K;
            c.x_bf16 = xm + (size_t)r0 * E;
            c.n_rows = (int)rows;
            c.out = fo + (size_t)r0 * E;
            c.prompt = prompt;
            return pulsar_moe_routed_rows(&c);
        }))
        return failf("the routed experts refused", il);
    if (!emit(fo, e2, n, E) ||
        !norm(e2, (const uint16_t *)wp(st, L.ffn_post_norm_2, "gemma4 post_feedforward_layernorm_2"), d2, n, E))
        return failf("the routed experts' norm refused", il);
    /* d1 += d2 (one rounding), and post_feedforward_layernorm of the sum into e1 */
    const uint16_t *pn = (const uint16_t *)wp(st, L.ffn_post_norm, "gemma4 post_feedforward_layernorm");
    return (pn && pulsar_rows_add_launch(d1, E, d2, E, n, E, pn, s.rms_eps, e1, E, 0) == 0) ||
           failf("the FFN sum's norm refused", il);
}

/* ---- the layer --------------------------------------------------------------------------------------------------- */

/* the norm whatever reads h after layer il applies: layer il + 1's input_layernorm, or output_norm (the head's) */
/** @param st  the step
 *  @param il  a layer
 *  @return the device weight, or NULL */
const uint16_t *next_norm(const g4_step *st, uint32_t il) {
    if (il + 1u < st->plan->n_layer)
        return (const uint16_t *)wp(st, st->w->layer[il + 1u].attn_norm, "gemma4 input_layernorm");
    return (const uint16_t *)wp(st, st->w->output_norm, "gemma4 norm");
}

/** pulsar_driver_ops::block[PULSAR_LAYER_BLOCKS]: one layer -- attention, then the FFN, each added to h; the last add
 *  scales h by layer_scalar and writes the next reader's norm into xa (layer 0 norms the embedding first).
 *  @param st0  the trunk's step
 *  @param il   the layer
 *  @return true = launched; false = refused (said) */
bool block(const pulsar_driver_step *st0, uint32_t il) {
    const g4_step *st = S(st0);
    g4_state *q = st->st;
    const pulsar_layer_desc &d = st0->plan->desc[il];
    if ((d.mixer != PULSAR_MIXER_ATTN_FULL && d.mixer != PULSAR_MIXER_ATTN_WINDOW) || d.ffn != PULSAR_FFN_DENSE_MOE)
        return failf("a layer description that is not {window | full attention} x {dense MLP + MoE}", il);
    const pulsar_gemma4_shape &s = st->w->shape;
    const pulsar_gemma4_block &L = st->w->layer[il];
    const uint32_t n = st->n_rows, E = s.n_embd;
    uint16_t *h = (uint16_t *)dptr(q->h), *xa = (uint16_t *)dptr(q->xa);
    if (il == 0 && !norm(h, (const uint16_t *)wp(st, L.attn_norm, "gemma4 input_layernorm"), xa, n, E))
        return failf("the input norm refused", il);
    const mix_scratch M = mix_layout(q->max_rows, st->states->ctx);
    uint8_t *mbase = (uint8_t *)dptr(q->scratch[ARENA_MIX]);
    if (!attention(st, il, mbase, M)) return false;
    /* h += post_attention_layernorm(delta), and the dense MLP's pre-norm of the new h in the same launch */
    const ffn_scratch F = ffn_layout(q->max_rows);
    uint8_t *fbase = (uint8_t *)dptr(q->scratch[ARENA_FFN]);
    const uint16_t *pre = (const uint16_t *)wp(st, L.ffn_pre_norm, "gemma4 pre_feedforward_layernorm");
    if (!pre || pulsar_rows_add_launch(h, E, (const uint16_t *)(mbase + M.dn), E, n, E, pre, s.rms_eps,
                                       (uint16_t *)(fbase + F.xd), E, 0) != 0)
        return failf("the attention's residual add refused", il);
    if (!ffn(st, il, fbase, F)) return false;
    /* h = bf16(bf16(h + ffn) * layer_scalar), and the next reader's norm into xa */
    const uint16_t *nn = next_norm(st, il), *ls = (const uint16_t *)wp(st, L.layer_scalar, "gemma4 layer_scalar");
    return (nn && ls && pulsar_rows_add_scale_launch(h, E, (const uint16_t *)(fbase + F.e1), E, n, E, ls, nn, s.rms_eps,
                                                     xa, E, 0) == 0) ||
           failf("the layer's scaled residual add refused", il);
}

/** pulsar_driver_ops::head: step rows [row0, row0 + n) of xa (output_norm of h) -> the EXL3 head (the decode arm at
 *  every call) -> logits rows [out0, out0 + n), soft-capped.
 *  @param st0   the trunk's step
 *  @param row0  the first step row headed
 *  @param n     the row count
 *  @param out0  the first logits row written
 *  @return true = launched; false = refused (said) */
bool head(const pulsar_driver_step *st0, uint32_t row0, uint32_t n, uint32_t out0) {
    const g4_step *st = S(st0);
    const pulsar_gemma4_shape &s = st->w->shape;
    if (n == 0 || row0 + n > st->n_rows || out0 + n > PULSAR_DRIVER_HEADS_MAX) return fail("head rows out of range");
    const head_scratch H = head_layout();
    uint8_t *base = (uint8_t *)dptr(st->st->scratch[ARENA_HEAD]);
    pulsar_rows_linear l{};
    float *logits = (float *)dptr(st->buf->logits) + (size_t)out0 * s.n_vocab;
    const uint16_t *xb = (const uint16_t *)dptr(st->st->xa) + (size_t)row0 * s.n_embd;
    if (!pulsar_linear_rows_ref(st->model, st->w->head, (int)s.n_embd, (int)s.n_vocab, false, "gemma4 lm_head", &l) ||
        pulsar_rows_linear_launch(&l, xb, (int)n, logits, base + H.lin, H.lin_bytes, 0) != 0)
        return fail("the head linear refused");
    const pulsar_head_decl hd = head_decl(s);
    if (pulsar_head_logits_finish_launch(&hd, logits, n, s.n_vocab, 0) != 0) return fail("the logits finish refused");
    /* L338 P2: the last headed row's final-norm hidden, kept for the decision head (pulsar_session_copy_hidden) */
    const uint64_t rb = (uint64_t)s.n_embd * 2u;
    if (pulsar_gpu_tensor_copy_async(st->st->last_hidden, 0, st->st->xa, (uint64_t)(row0 + n - 1) * rb, rb) == 0)
        return fail("could not keep the last row's hidden");
    st->st->headed = true;
    return true;
}

/** pulsar_driver_ops::hidden (L338 P2): the last headed row's final-norm hidden, widened from bf16.
 *  @param s      the session
 *  @param out    receives n_embd floats
 *  @param cap    floats at out
 *  @return n_embd, or 0 when nothing was headed or cap is short (said) */
uint32_t hidden(pulsar_session *s, float *out, uint32_t cap) {
    g4_state *q = ST(s);
    const uint32_t E = W(s->engine)->shape.n_embd;
    if (!q->headed || cap < E) {
        fprintf(stderr, "pulsar: %s: hidden readout: %s -- refusing\n", ARCH,
                !q->headed ? "no row headed yet" : "the buffer is shorter than the hidden width");
        return 0;
    }
    std::vector<uint16_t> h(E);
    if (pulsar_gpu_synchronize() == 0 || !pulsar_gpu_tensor_read(q->last_hidden, 0, h.data(), (uint64_t)E * 2u)) {
        fail("could not read the last row's hidden");
        return 0;
    }
    for (uint32_t i = 0; i < E; i++) {
        const uint32_t u = (uint32_t)h[i] << 16;
        memcpy(&out[i], &u, 4);
    }
    return E;
}

/** pulsar_driver_ops::step_end: the routed experts' non-finite flag (rule 9).
 *  @param st0  the step
 *  @param ok   the forward's verdict
 *  @return ok and no layer's routed output went non-finite (said) */
bool step_end(const pulsar_driver_step *st0, bool ok) {
    if (!ok) return false;
    const g4_step *st = S(st0);
    uint32_t nf = 0;
    if (!pulsar_gpu_tensor_read(st->st->scratch[ARENA_ROWS], rows_layout(st->st->max_rows).nf, &nf, sizeof(nf)))
        return fail("could not read the MoE non-finite flag");
    if (nf) {
        fprintf(stderr, "pulsar: %s: the routed output of layer %u went non-finite (code 0x%08x) -- refusing the step\n",
                ARCH, nf & 0xffffffu, nf);
        return false;
    }
    return true;
}

/** pulsar_driver_ops::dump: the residual stream after layer il (bf16 widened by the dump), named "gemma4" -- what the
 *  parity grade reads against the reference's hidden_states.
 *  @param st0  the step
 *  @param il   the layer just run */
void dump(const pulsar_driver_step *st0, uint32_t il) {
    const g4_step *st = S(st0);
    gpu_graph_debug_dump_hc_tensor("gemma4", st->st->h, (uint64_t)st->n_rows * st->w->shape.n_embd, il,
                                   (uint32_t)st->pos[st->n_rows - 1]);
}

}  // namespace

/* ---- the state declaration --------------------------------------------------------------------------------------- */

/** @param e  the engine: its shape and plan are read; state_plan is written */
void pulsar_gemma4_state_plan(pulsar_engine *e) {
    const pulsar_gemma4_shape *s = &W(e)->shape;
    const pulsar_layer_plan *lp = &e->plan;
    pulsar_state_plan *p = &e->state_plan;
    *p = pulsar_state_plan{};
    p->name = "gemma4";
    p->resume_grid = kResumeGrid;
    p->split_invariant = true;
    p->ckpt_slots = kCkptSlots;
    const pulsar_gqa_layer gw = gqa_decl(*s, false, NULL, NULL, NULL), gf = gqa_decl(*s, true, NULL, NULL, NULL);
    pulsar_state_decl d[PULSAR_GEMMA4_STATE_COUNT] = {};
    d[PULSAR_GEMMA4_STATE_KV_WIN] = {"kv_win", PULSAR_STATE_RING, PULSAR_STATE_CAPTURE_NONE, false, false, s->window,
                                     pulsar_gqa_kv_token_bytes(&gw), 0, 0, NULL, NULL, NULL};
    d[PULSAR_GEMMA4_STATE_KV_FULL] = {"kv_full", PULSAR_STATE_POOL, PULSAR_STATE_CAPTURE_NONE, false, false, 1u,
                                      pulsar_gqa_kv_token_bytes(&gf), 0, 0, NULL, NULL, NULL};
    for (uint32_t id = 0; id < PULSAR_GEMMA4_STATE_COUNT; id++) (void)pulsar_state_plan_add(p, d[id]);
    for (uint32_t il = 0; il < lp->n_layer; il++)
        p->slot[il] = 1u << (pulsar_gemma4_is_full(s, il) ? PULSAR_GEMMA4_STATE_KV_FULL : PULSAR_GEMMA4_STATE_KV_WIN);
    p->n_trunk = lp->n_layer;
    p->n_slots = lp->n_layer;
    p->session = 0;
}

const pulsar_driver_ops PULSAR_GEMMA4_DRIVER = {
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
    /* .absorb        = */ NULL,   /* no drafter lane yet (L338 P3: DFlash) */
    /* .mtp           = */ NULL,
    /* .rope          = */ NULL,   /* text positions */
    /* .image_rows    = */ NULL,   /* text only */
    /* .dump          = */ dump,
    /* .artifact      = */ artifact,
    /* .stage         = */ stage,   /* the host half; the ops are launch-only (a decode step graphs) */
    /* .read_ahead    = */ NULL,
    /* .read_ahead_pool = */ NULL,
    /* .adapter       = */ has_adapter,   /* L338 P2: the JEV LoRA, per bank */
    /* .hidden        = */ hidden,        /* L338 P2: the decision head's input */
};
static_assert(PULSAR_LAYER_BLOCKS == 4 && PULSAR_LAYER_KIND_COUNT == 5, "the block table's slot is the kind's value");

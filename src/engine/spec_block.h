/* spec_block.h -- the BLOCK drafter (L298, L299): a DFlash-family drafter as ONE platform piece, behind the speculation
 * round API (spec_ops.h pulsar_drafter_ops) and the core row-step driver (step_driver.h).  Two drafters run on it today:
 * NVIDIA's LiLiCorr for Nemotron 3.5 Super (L298) and z-lab's DFlash2 for Qwen3.8-Flash-Next (incoai's, L299).  A
 * family declares only where its trunk's hidden states are tapped (pulsar_block_drafter::tap_layer / tap_sub, from the
 * drafter's target block ids) and writes the tap rows there with its own op (pulsar_block_tap_rows: Nemotron a row
 * copy of its residual, Qwen the mean of its hyper-connection streams); everything else -- the declaration, the
 * weights' binding from the container, the drafter's state, the context absorb, the block forward, the chain -- is
 * here.
 *
 * WHAT A BLOCK DRAFTER IS (vLLM qwen3_dflash.py / qwen3_dflash2.py / lilicorr.py, z-lab dflash/model.py):
 *   context   the trunk's residual stream after each of n_taps target blocks, concatenated (n_taps x hidden) ->
 *             fc -> c (bf16), cn = rmsnorm(c) * hidden_norm; every drafter layer l keeps K | V of cn over the context:
 *             K = rope(rmsnorm_head(cn Wk_l) * k_norm_l), V = cn Wv_l (a sliding-window RING per layer: the drafter's
 *             state, state_kinds.h).  A position enters the context once the trunk has COMMITTED it (a prompt row, a
 *             decode row, a verify's kept rows: never a rejected one), so the drafter's state at T is a function of
 *             the trunk's rows < T -- what the grid checkpoints and the payloads walk.
 *   draft     ONE forward over a block of `block` rows at positions P .. P + block - 1 (P the bank's next position):
 *             row 0 the carry (the round's next base token), rows 1 .. the mask token, embedded by the TARGET's
 *             input embedding.  Each layer (Qwen3-style, pre-norm, the residual in bf16):
 *                 x = rmsnorm(h) * in_norm;  (cx, d) = conv_a.prepare(x);  a = attn(cx);  h += conv_a.finish(a, d)
 *                 x = rmsnorm(h) * post_norm;  (cx, d) = conv_m.prepare(x);  m = mlp(cx);  h += conv_m.finish(m, d)
 *             attn: GQA with per-head q / k RMSNorm and NeoX rope, BIDIRECTIONAL inside the block and the window over
 *             the context ring (the shared GQA block's `block` mode: its rows are never stored); mlp: SwiGLU;
 *             conv: the grouped dynamic convolution (pulsar_cuda_blockdraft.h), its per-token taps projected from
 *             the sublayer's input (kernel_projection: hidden -> 2 sides x taps x groups).  Then o = rmsnorm(h) *
 *             norm, the drafter's head (its copy of the target's lm_head, over the whole vocabulary or a subset of
 *             it -- head_ids) over rows 1 .. block - 1, and per slot the top-k ids with their log-softmax values.
 *   chain     the declaration's CHAIN HEAD (pulsar_block_head_kind), each a producer of the same three things -- per
 *             slot the candidates, a [prev][candidate] score table (slot 0: one row, broadcast), the head's own
 *             greedy walk -- so the walk below is one for both:
 *               LILICORR  the LiLiCorr candidate head (pulsar_cuda_lilicorr.h): the slots' top-8 candidates re-scored
 *                         jointly with the anchor (cn of the last committed position)
 *               SELECTOR  the CandidateSelector (pulsar_cuda_candsel.h): each slot's top-16 re-scored against the
 *                         token chosen at the slot before through two rank-256 codebooks
 *             then a walk over the scores -- the head's own greedy walk (greedy), or slot by slot a draw from the
 *             realized row's candidates under the request's params at the draft's key (sampled: vLLM's candidate
 *             sampler's proposal).
 *
 * THE DECLARATION (the container's): `pulsar.drafter` names the KIND ("lilicorr", "dflash2"), and the kind's config
 * sits verbatim under `<kind>.` (the builder writes the drafter's HF config there, tools/container); one table here
 * maps each kind's key names to the geometry, the taps, the chain head and its stop rule's tau (spec_block.cpp
 * k_kinds) -- the one reader for every kind.
 *
 * THE ARMS (rule 8, Tyler 2026-10-16 "cutlass / cublaslt / b12x before our own kernels"): every Linear through the
 * rows front door (pulsar_linear_rows_ref: BF16 rows kernel / cuBLASLt for prompt rows, NVFP4 W4A16 rows kernel for
 * the nvfp4 weights); the attention and the context store are the shared GQA block's; the norms and adds the row ops';
 * the convolution, SwiGLU, the taps and the top-k the small kernels of pulsar_cuda_blockdraft.h; the chain heads
 * their own modules.
 *
 * Include after pulsar_engine_internal.h. */
#ifndef PULSAR_SPEC_BLOCK_H
#define PULSAR_SPEC_BLOCK_H

#include "spec_internal.h"
#include "role_bind.h"

#define PULSAR_BLOCK_MAX_LAYERS 8u
#define PULSAR_BLOCK_MAX_TAPS 8u
/** The widest block: its drafts (block - 1) fit the round record's 16 and the verify width. */
#define PULSAR_BLOCK_SIZE_MAX 16u

/** Where in a trunk layer a tap reads the residual stream: after its mixer's residual add, or after its FFN's. */
enum pulsar_block_tap_sub : uint32_t { PULSAR_BLOCK_TAP_MIXER = 0, PULSAR_BLOCK_TAP_FFN = 1 };

/** The chain head a drafter's declaration ends in (the header's CHAIN). */
enum pulsar_block_head_kind : uint32_t { PULSAR_BLOCK_HEAD_LILICORR = 0, PULSAR_BLOCK_HEAD_SELECTOR = 1 };

/** One drafter layer's tensors (bind block `drafter`, layer l). */
typedef struct {
    pulsar_tensor *in_norm, *post_norm;              ///< [hidden]
    pulsar_tensor *q, *k, *v, *o;                    ///< hidden -> n_head hd | n_kv hd | n_kv hd, n_head hd -> hidden
    pulsar_tensor *q_norm, *k_norm;                  ///< [head_dim]
    pulsar_tensor *ac_base, *ac_proj;                ///< the attention conv: base [2][taps][hidden], hidden -> 2 taps groups
    pulsar_tensor *mc_base, *mc_proj;                ///< the MLP conv
    pulsar_tensor *gate, *up, *down;                 ///< hidden -> ff, ff -> hidden
} pulsar_block_layer;

/** The LiLiCorr candidate head's tensors (bind block `lilicorr`; lattice layer i = layer i), one per field of
 *  pulsar_lilicorr_head (src/cuda/pulsar_cuda_lilicorr.h).  `_b` = the Linear's bias. */
typedef struct {
    pulsar_tensor *token_proj, *token_proj_b, *pass_proj, *pass_proj_b, *context_proj, *context_proj_b;
    pulsar_tensor *feature_norm_w, *feature_norm_b, *feature_up_w, *feature_up_b, *feature_down, *feature_down_b;
    pulsar_tensor *slot_embedding, *rank_embedding, *relative_slot_bias, *same_slot_bias;
    pulsar_tensor *output_norm, *anchor_norm, *factor_in, *factor_in_b;
    pulsar_tensor *out_head, *out_head_b, *in_head, *in_head_b, *anchor_out_head, *anchor_out_head_b;
    struct {
        pulsar_tensor *attn_norm, *attn_in, *attn_in_b, *attn_out, *attn_out_b;
        pulsar_tensor *mlp_norm, *mlp_up, *mlp_up_b, *mlp_down, *mlp_down_b;
    } layer[2];
    float vector_eps, logit_scale;
} pulsar_block_head;

/** The CandidateSelector's tensors (bind block `selector`), pulsar_cuda_candsel.h. */
typedef struct {
    pulsar_tensor *proj;         ///< hidden_projection: hidden -> rank (a Linear)
    pulsar_tensor *pred, *succ;  ///< the predecessor / successor codebooks, bf16 [vocab rows of rank]
    uint32_t rank;
} pulsar_block_selector;

/** The loaded block drafter (pulsar_engine::block_drafter): its declaration, the family's tap points, its weights,
 *  its state ids. */
typedef struct pulsar_block_drafter {
    const char *kind;            ///< the container's pulsar.drafter ("lilicorr", "dflash2")
    pulsar_block_head_kind head_kind;
    /* the declaration (the container's <kind>.* keys) */
    uint32_t hidden, n_layer, n_head, n_kv, head_dim, ff, vocab;
    uint32_t block;              ///< rows a draft pass carries: the carry + block - 1 mask rows
    int32_t mask_id;
    uint32_t window;             ///< the sliding window over the context (keys a row sees, itself included)
    uint32_t conv_taps, conv_group;
    uint32_t topk;               ///< candidates per slot
    float eps, rope_theta;
    uint32_t n_taps;
    uint32_t tap_block[PULSAR_BLOCK_MAX_TAPS];   ///< the target blocks (in the target's own numbering) tapped AFTER
    /* the family's: tap j reads the residual after trunk layer tap_layer[j]'s sublayer tap_sub[j] */
    uint32_t tap_layer[PULSAR_BLOCK_MAX_TAPS];
    uint32_t tap_sub[PULSAR_BLOCK_MAX_TAPS];
    /* the weights (bind block `drafter`) */
    pulsar_tensor *fc;           ///< n_taps hidden -> hidden
    pulsar_tensor *hidden_norm;  ///< the context norm
    pulsar_tensor *out_norm;     ///< the final norm
    pulsar_tensor *head;         ///< hidden -> head_rows (the drafter's own lm_head)
    /** The head's rows' vocabulary ids, I32 [head_rows] ascending, when the head is a SUBSET of the vocabulary (a
     *  draft vocabulary: the cheaper read); NULL = the head is the whole vocabulary (head_rows == vocab). */
    pulsar_tensor *head_ids;
    uint32_t head_rows;
    pulsar_block_layer layer[PULSAR_BLOCK_MAX_LAYERS];
    pulsar_block_head head_lc;   ///< the LiLiCorr candidate head (bind block `lilicorr`), head_kind LILICORR
    pulsar_block_selector sel;   ///< the CandidateSelector (bind block `selector`), head_kind SELECTOR
    const pulsar_tensor *embed;  ///< the TARGET's input embedding, bf16 [vocab rows of hidden] (the family's)
    /* the state plan's ids (pulsar_block_drafter_plan); decl_anchor only for a head that reads the anchor row
     * (LILICORR), else UINT32_MAX */
    uint32_t decl_kv, decl_anchor;
    /** The round API's drafter for this declaration (pulsar_block_drafter_read: PULSAR_BLOCK_DRAFTER_OPS at a chain of
     *  the block's block - 1 slots) -- the engine's drafter_ops points here. */
    pulsar_drafter_ops ops;
} pulsar_block_drafter;

/** The block drafter kind the container holds: its pulsar.drafter when that names a kind this lane implements
 *  ("lilicorr", "dflash2"), "" when it holds none (no key, or an empty one), NULL refused (said, counted: another
 *  name, or not a string). */
const char *pulsar_block_drafter_kind_of(const pulsar_model *m);
/** Read the declaration of the kind the container holds (its <kind>.* keys, the kind table) and check every number
 *  the blocks implement.  false = refused (said, counted: pulsar_load_refuse). */
bool pulsar_block_drafter_read(const pulsar_model *m, pulsar_block_drafter *d);
/** Bind block `drafter`'s tensors and the chain head's (block `lilicorr` or `selector`) through the role binder (the
 *  linears admitted by the format registry at bf16 rows).  false = refused. */
bool pulsar_block_drafter_bind(pulsar_role_binder *b, pulsar_block_drafter *d);
/** The blocks a drafter of `kind` occupies in a container (`drafter` and its chain head's), NULL-terminated: what an
 *  open that does not select it declares unloaded. */
const char *const *pulsar_block_drafter_blocks(const char *kind);
/** Declare the drafter's state in the family's plan (after the trunk's slots): a RING of K | V rows per drafter layer
 *  (the context; window `window`) and, on the first drafter slot, each bank's anchor row when the chain head reads it
 *  (FIXED: cn of the last committed position, LILICORR's input). */
void pulsar_block_drafter_plan(pulsar_engine *e, pulsar_state_plan *p);
/** The announce line's words for the loaded drafter. */
void pulsar_block_drafter_describe(const pulsar_block_drafter *d, char *out, size_t cap);

/** The round API's drafter (spec_ops.h): the functions; a loaded drafter's numbers are its own copy's
 *  (pulsar_block_drafter::ops). */
extern const pulsar_drafter_ops PULSAR_BLOCK_DRAFTER_OPS;

/* ---- the lane: the per-session buffers (pulsar_driver_state::block) -------------------------------------------- */
struct pulsar_block_lane;
pulsar_block_lane *pulsar_block_lane_create(pulsar_session *s, uint32_t n_banks, uint32_t max_rows);
void pulsar_block_lane_free(pulsar_block_lane *l);

/** The tap rows' destination: the next tap (from *j) reading at trunk layer il's sublayer `sub` -- its columns of
 *  the lane's tap rows (bf16, row r at the result + r * *ld_dst), *j moved past it -- or NULL when no further tap reads
 *  there (or no lane; *refused set when the step is wider than the tap rows, said).  pulsar_block_tap_rows walks it. */
uint16_t *pulsar_block_tap_dst(pulsar_driver_state *d, uint32_t il, uint32_t sub, uint32_t n, uint32_t *j,
                               uint64_t *ld_dst, bool *refused);
/** The family's trunk forward, after trunk layer il's sublayer `sub` wrote the step's n rows: `emit(dst, ld_dst)`
 *  writes them as the drafter's tap (bf16, hidden wide, row r at dst + r * ld_dst) for every tap reading there -- the
 *  family's own op (a row copy of its residual, the mean of its streams).  Launch-only (fixed device pointers: a
 *  captured step replays it).  No lane or no tap there: nothing.  false = refused (said). */
template <class Emit>
bool pulsar_block_tap_rows(pulsar_driver_state *d, uint32_t il, uint32_t sub, uint32_t n, Emit emit) {
    uint32_t j = 0;
    uint64_t ld = 0;
    bool refused = false;
    while (uint16_t *dst = pulsar_block_tap_dst(d, il, sub, n, &j, &ld, &refused))
        if (!emit(dst, ld)) return false;
    return !refused;
}
/** ... with the tap a row copy of the residual h (bf16 rows [n][ld]): Nemotron-H's. */
bool pulsar_block_tap(pulsar_driver_state *d, uint32_t il, uint32_t sub, const uint16_t *h, uint64_t ld, uint32_t n);

/** The driver's absorb (pulsar_driver_ops::absorb) for a family whose drafter is a block drafter or a chained MTP:
 *  the session's lane's lockstep (spec_mtp.h pulsar_mtp_absorb, or the block lane's context absorb of the step's
 *  rows), nothing without a lane. */
bool pulsar_drafter_lane_absorb(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                                const int32_t *bank, uint32_t n, uint32_t row0, bool eval);

#endif /* PULSAR_SPEC_BLOCK_H */

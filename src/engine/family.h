/* family.h -- the model-family interface (L251 S1).
 *
 * A FAMILY is one model architecture the engine serves end to end: DeepSeek V4
 * (0731, Vision-Exp, V4.1 -- one family, two shape profiles) and Qwen4-exp
 * (Qwen3.8-Flash-Next).  The family is chosen ONCE, at pulsar_engine::open,
 * from the artifact's `general.architecture` key (pulsar_family_for_model);
 * an architecture no family claims is refused by name.  There is no runtime
 * flag, no fallback between families, and no per-layer or per-token test of
 * which family is loaded: generic code reaches the family through
 * `pulsar_engine::family` at STEP granularity (one indirect call per session
 * operation), and a family's own forward walks its own layer plan.
 *
 * What a family provides:
 *   - identity: the id, the architecture string, the chat/tokenizer format,
 *     the drafter kind, and the cache-compatibility model id;
 *   - load: config validation (the artifact's metadata against the family's
 *     known shape; refuse on any mismatch), the layer plan, the weight binding;
 *   - the layer plan: one pulsar_layer_kind per layer, THE authority for which
 *     op runs at layer il (DeepSeek: one kind -- its per-layer attention mode
 *     stays in pulsar_attn_layout, which is that fact's authority);
 *   - session state: create/destroy/price (DeepSeek: the pulsar_gpu_graph
 *     with its SWA rings, compressed KV, compressor and indexer frontiers and
 *     bank slabs; Qwen: DeltaNet recurrent + conv state per GDN layer, FP8 KV
 *     for the QSA layers, the PLE conv state -- src/engine/family_qwen.h);
 *   - the forward at session-operation granularity (sync, eval, batched and
 *     mixed decode, invalidate);
 *   - capabilities: every session feature beyond that core (banks, fork,
 *     speculation, disk KV payloads, rewind, vision, TP, imatrix, chat) is a
 *     CAP bit.  The C API refuses an operation whose cap the loaded family
 *     does not declare, by name, before any family state is touched (rule 9).
 *
 * DeepSeek's implementation is the engine as it stood at L250 (family_deepseek.cpp
 * points at it; the numerics are untouched).  Qwen's is family_qwen.cpp; its
 * per-layer ops are the pulsar_qwen_ops table (family_qwen.h) that streams
 * S2-S4 implement. */
#ifndef PULSAR_FAMILY_H
#define PULSAR_FAMILY_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "pulsar.h"
#include "spec_ops.h"

/* Included from pulsar_engine_internal.h, after pulsar_model; not a standalone
 * header. */

/** What a layer IS.  Values are stable (they are printed and may be stored). */
typedef enum : uint8_t {
    PULSAR_LAYER_NONE      = 0,  ///< no layer (a plan slot past n_layer)
    PULSAR_LAYER_DS4_BLOCK = 1,  ///< DeepSeek V4 block: HC mix, MLA attention (SWA + compressed/indexed per pulsar_attn_layout), MoE
    PULSAR_LAYER_QWEN_GDN  = 2,  ///< Qwen4-exp `linear_attention`: Gated DeltaNet mixer + gated residual + MoE
    PULSAR_LAYER_QWEN_QSA  = 3,  ///< Qwen4-exp `full_attention`: QSA (gated GQA + block indexer) mixer + gated residual + MoE
} pulsar_layer_kind;

/** Printable name of a layer kind ("ds4_block", "qwen_gdn", "qwen_qsa"). */
const char *pulsar_layer_kind_name(pulsar_layer_kind k);

/** The widest plan any family builds.  DeepSeek V4 Pro reserved 61 layers
 * (PULSAR_MAX_LAYER); Qwen4-exp has 48.  Checked against PULSAR_MAX_LAYER in
 * pulsar_engine_internal.h. */
#define PULSAR_FAMILY_MAX_LAYER 64u

/** L272 P4b: a rank's tensor-parallel plan (tp_slice.cpp; opaque to the families, which add to it). */
typedef struct pulsar_tp_plan pulsar_tp_plan;

/** A family's IMAGE FRONT (L281 geometry, L268 the rest).  The core owns the image path
 *  (image_identity.cpp, image_front.cpp): the placeholder expansion, the block fit, the reuse licence, the
 *  identity a session holds, the chunk merge and its tower-output cache.  The family supplies only what is its
 *  own -- where blocks are, what its renderer writes, how one image becomes block ids, and its tower.  NULL on a
 *  family that serves no images: no id is a sentinel and an image request is refused by name.  Whether the loaded
 *  artifact carries a bound tower is `pulsar_engine::vision_ready`, set by the family's load. */
typedef struct pulsar_family_vision {
    bool (*is_sentinel)(const pulsar_engine *e, int32_t id);
    /** true and *len = the block's length when `ids[start..]` begins an image block */
    bool (*block_extent)(const pulsar_engine *e, const int32_t *ids, int n, int start, int *len);
    /** The text a chat renderer writes where one image goes; it carries the placeholder token once. */
    const char *placeholder_text;
    /** The placeholder token the expansion replaces, one per image, or -1 when the tokenizer has none. */
    int (*placeholder_id)(const pulsar_engine *e);
    /** Append image `img`'s block ids to `out` -- the block begins at out->len.  false + `err` when the image
     *  cannot be decoded or is not one the tower accepts. */
    bool (*expand)(const pulsar_engine *e, const pulsar_image_ref *img, pulsar_tokens *out, char *err,
                   size_t errlen);
    /** bf16 elements in one block row (the language model's embedding width). */
    uint32_t (*row_width)(const pulsar_engine *e);
    /** The `block_len` rows image `img` puts at its block `ids[0..block_len)`, into `out`.  `tower` is the image's
     *  tower output (n_tower rows, position-independent) when the core's cache holds it; NULL = run the tower and
     *  hand its output to the core through *tower_out / *n_tower_out (malloc'd; the core caches and frees it). */
    bool (*block_rows)(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *ids, int block_len,
                       const uint16_t *tower, int n_tower, uint16_t **tower_out, int *n_tower_out, uint16_t *out,
                       char *err, size_t errlen);
    /** Optional (NULL = blocks have no 2D layout): the block's grid in rows, *h x *w = its length -- what
     *  multi-axis rope positions follow (pulsar_image_rope3).  false = the image cannot be read. */
    bool (*grid)(const pulsar_engine *e, const pulsar_image_ref *img, uint32_t *h, uint32_t *w);
} pulsar_family_vision;

/** L272 P4c: how a family's forward hands its linears their activation -- the backend's MX slot (DeepSeek) or raw
 *  bf16 rows (Qwen).  The core chooses by it: the dense / MoE arm with the format, and a TP slice's operation (a
 *  slot reader reads slices REGISTERED with the backend, a rows reader slices BUILT and kept on the device). */
typedef enum { PULSAR_ACT_KIND_SLOT = 0, PULSAR_ACT_KIND_ROWS = 1 } pulsar_act_kind;

/** The layer plan: built once by the family's load from the artifact, then
 * read-only.  kind[il] for il < n_layer is never PULSAR_LAYER_NONE. */
typedef struct {
    uint32_t n_layer;
    pulsar_layer_kind kind[PULSAR_FAMILY_MAX_LAYER];
} pulsar_layer_plan;

/** Count the layers of one kind. */
uint32_t pulsar_layer_plan_count(const pulsar_layer_plan *p, pulsar_layer_kind k);

/** Session features beyond the core forward.  A C API entry point whose
 * feature the loaded family does not declare refuses by name
 * (pulsar_family_require). */
enum : uint32_t {
    PULSAR_FAMILY_CAP_BANKS   = 1u << 0,  ///< a bank pool > 1: repoint, fork, physical residency, bank state save/restore, per-bank KV spill
    PULSAR_FAMILY_CAP_SPEC    = 1u << 1,  ///< speculative decoding: the family provides pulsar_family::spec (the verify hooks) for whichever drafter is loaded
    PULSAR_FAMILY_CAP_PAYLOAD = 1u << 2,  ///< disk-KV payloads and snapshots (save/load/stage/mirror)
    PULSAR_FAMILY_CAP_REWIND  = 1u << 3,  ///< rewind / rewrite-from-common to an earlier position of a live session
    PULSAR_FAMILY_CAP_VISION  = 1u << 4,  ///< image spans in a prompt
    PULSAR_FAMILY_CAP_TP      = 1u << 5,  ///< tensor parallelism across a pair / mesh
    PULSAR_FAMILY_CAP_IMATRIX = 1u << 6,  ///< importance-matrix collection
    PULSAR_FAMILY_CAP_CHAT    = 1u << 7,  ///< tokenizer + chat renderer (a family without it cannot take text)
    PULSAR_FAMILY_CAP_GENERATE = 1u << 8, ///< pulsar_engine_generate_argmax (the session-less whole-graph path)
    PULSAR_FAMILY_CAP_SEGMENTS = 1u << 9, ///< disk KV segments: span files of the grid checkpoint store (L264; Qwen L266)
    PULSAR_FAMILY_CAP_MIXED_PREFILL = 1u << 10, ///< decode_mixed carries prefill runs beside its decode rows (the plain lane's mixed quantum); without it a prompt rides only the fused step
};

/** The session operations every family implements.  The C API
 * (engine_api.cpp) calls these at step granularity; the TP mirror wraps the
 * same calls on a family with PULSAR_FAMILY_CAP_TP. */
typedef struct {
    /** Build the family's state into a session the core allocated (L272 P2: engine, ctx_size, prefill_cap
     * and the logits row are set; the core measures the GPU bytes this allocates as resident_bytes and
     * assigns the TP mirror id).  0 on success; on failure the family frees what it built. */
    int (*create)(pulsar_session *s);
    /** Free the family's state; the core frees the session's own (the view, the carry, the sampler and
     * speculation scratch, the logits) and the session itself. */
    void (*destroy)(pulsar_session *s);
    /** GPU bytes create() takes at (ctx_size, n_banks): the allocation run dry,
     * so the price and the allocation are one function.  0 = cannot create. */
    uint64_t (*cost_bytes)(pulsar_engine *e, int ctx_size, int n_banks);
    /** Make the session's state hold exactly `prompt` (prefill what is new). */
    int (*sync)(pulsar_session *s, const pulsar_tokens *prompt,
                const pulsar_image_ref *images, int n_images, char *err, size_t errlen);
    /** Append one token and decode the next logits row. */
    int (*eval)(pulsar_session *s, int token, char *err, size_t errlen);
    /** One decode row per request (each on its own bank). */
    int (*decode_multiseq)(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n,
                           float *logits, int logits_cap, char *err, size_t errlen);
    /** Decode rows and prefill runs in one step. */
    int (*decode_mixed)(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                        float *logits, int logits_cap, uint32_t *out_n_rows,
                        uint32_t max_head_runs, char *err, size_t errlen);
    /** Forget the session's state (the next sync prefills cold). */
    void (*invalidate)(pulsar_session *s);
    /** The fused step (pulsar_session_decode_fused, contract in pulsar.h): decode rows and prompt runs in one
     *  forward, each run's last row headed on request.  NULL = the family has none; pulsar_engine_has_fused_step
     *  is this op's presence.  The core calls it through pulsar_session_fused_local, which keeps the session's
     *  record of the step's block (fused_logits) for every family. */
    int (*decode_fused)(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                        const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                        char *err, size_t errlen);
    /** The most logits rows one decode_fused step heads -- its verify rows plus its headed prompt runs (L284 #2:
     *  the ONE authority the server sizes a round's verify budget by, pulsar_engine_fused_heads_max).  0 with no
     *  decode_fused. */
    uint32_t fused_heads_max;
} pulsar_family_session_ops;

/** A family's tokenizer and chat front (L272 P2): what the public tokenizer entries (tokenizer.cpp) do on
 * this family's model.  NULL on the family = no tokenizer (the entries end the process by name; the front
 * ends refuse such an engine at startup through pulsar_engine_has_tokenizer).  Before L272 the entries
 * branched on the Qwen tokenizer's pointer at every site. */
typedef struct pulsar_family_tokenizer {
    /** Raw text, all of it client data (no added token matches). */
    void (*encode_text)(pulsar_engine *e, const char *text, pulsar_tokens *out);
    /** A rendered chat: added tokens match except inside the client-data `spans` (NULL / 0 = none). */
    void (*encode_rendered)(pulsar_engine *e, const char *text, const pulsar_text_span *spans, uint32_t n_spans,
                            pulsar_tokens *out);
    /** One system + one user message, rendered and tokenized the family's way (the CLI's one-shot). */
    void (*encode_chat_prompt)(pulsar_engine *e, const char *system, const char *prompt, pulsar_think_mode think_mode,
                               pulsar_tokens *out);
    bool (*is_stop)(pulsar_engine *e, int token);
    /** The stop id a caller that needs ONE uses (pulsar_token_is_stop tests them all). */
    int (*eos)(pulsar_engine *e);
    /** `token`'s bytes, malloc'd and NUL-terminated (empty for an id outside the table). */
    char *(*token_text)(pulsar_engine *e, int token, size_t *len);
    int (*think_close)(pulsar_engine *e);
    /** The turn markers the server's prefix anchors scan for; false = unknown (said once). */
    bool (*turn_markers)(pulsar_engine *e, pulsar_turn_markers *out);
    /** --dump-tokens: the ids, then one line per token. */
    void (*dump)(pulsar_engine *e, FILE *fp, const pulsar_tokens *tokens);
    /** The family's chat is DeepSeek's marker template, which the incremental entries
     *  (pulsar_chat_begin / _append_lead_in / _append_message / _append_assistant_prefix) build; a family
     *  that renders its chat whole (Qwen: qwen_chat_render) refuses them by name. */
    bool incremental_ds4_template;
} pulsar_family_tokenizer;

/** A family's own bank pool (L251): the server's per-bank bookkeeping for a
 * family whose banks are not DeepSeek's graph pool.  NULL (DeepSeek) = the
 * pulsar_session members in session_banks.cpp.  When set, engine_api.cpp's bank
 * entries call these, and the operations a family cannot do -- forks, per-bank
 * KV spill, physical eviction -- refuse with each entry's own failure value
 * (the server has a path for every one of them).  The demand-paged KV accounting
 * (L270) is the same contract as DeepSeek's (admission prices the touched share,
 * the 2b guard keeps touched KV under budget). */
typedef struct {
    int (*count)(pulsar_session *s);
    /** The live host view into bank's carry -- the core's pulsar_bank_carry_save_view (L272 P2) plus any
     *  device side of the family's own; the history readers are the core's (pulsar_bank_history). */
    void (*save)(pulsar_session *s, uint32_t bank);
    /** Make `bank` live: the core's pulsar_bank_carry_restore_view, and what a bank with no carry means.
     *  false = refused. */
    bool (*restore)(pulsar_session *s, uint32_t bank);
    /** Tokens the batched lane fed that the host view has not recorded yet. */
    void (*note_committed)(pulsar_session *s, const int *toks, int n);
    /** L266: the family's grid checkpoints (kv_state.h), keyed by bank like DeepSeek's graph.ckpt. */
    struct pulsar_ckpt_store *(*kv_store)(pulsar_session *s);
    /** L266: where `bank`'s prefill-only history ends (the deepest point a checkpoint may sit). */
    uint32_t (*prefill_frontier)(pulsar_session *s, uint32_t bank);
    /** L266: the bank sync / eval run on (the store's walk reads and writes this one). */
    uint32_t (*live)(pulsar_session *s);
    /** L270: the demand-paged (physical-on-touch) KV bytes ONE bank reserves at ctx_size -- the share
     * admission prices as it is touched, not up front. */
    uint64_t (*demand_paged_bytes)(pulsar_engine *e, int ctx_size);
    /** L270: `bank`'s touched demand-paged KV: what its pages physically hold. */
    uint64_t (*touched_kv_bytes)(pulsar_session *s, uint32_t bank);
    /** L270: the most one bank's touched KV can grow over a decode quantum of q tokens. */
    uint64_t (*growth_bytes)(pulsar_session *s, uint32_t q);
} pulsar_family_bank_ops;

/** One model family.  Instances are static and const; pulsar_engine::family
 * points at one for the engine's lifetime. */
struct pulsar_family {
    pulsar_family_id id;
    const char *arch;              ///< the artifact's `general.architecture` value this family claims
    const char *name;              ///< human-readable family name
    uint32_t caps;                 ///< PULSAR_FAMILY_CAP_* bits
    /** Everything between model_open and the GPU: tokenizer, config
     * validation, the layer plan (e->plan), weight binding.  Exits or returns
     * false (after printing) on any mismatch. */
    bool (*load)(pulsar_engine *e, const pulsar_engine_options *opt);
    /** After the GPU is up (and the TP transport, on a TP family): the
     * family's device-side registrations and its boot announce lines. */
    bool (*after_gpu)(pulsar_engine *e);
    /** Engine facts that depend on the loaded shape. */
    uint32_t (*logits_width)(const pulsar_engine *e);
    const char *(*model_name)(const pulsar_engine *e);
    /** The id the API serves the model as (`model` in responses and /v1/models). */
    const char *(*served_model_id)(const pulsar_engine *e);
    pulsar_chat_format (*chat_format)(const pulsar_engine *e);
    int (*model_id)(const pulsar_engine *e);
    /** The longest context the loaded model's positions were trained for, with whatever position scaling the
     *  family implements (DeepSeek: YaRN's original context x factor; Qwen: max_position_embeddings until L280's
     *  YaRN).  pulsar_session_create refuses a larger one (L284: past it a session ran untrained positions). */
    uint64_t (*trained_context)(const pulsar_engine *e);
    /** The tensor-parallel transport's shape (L266): exchange slots per step (layers), the row width an
     *  all-reduce moves, and the vocab -- the TP identity and the slab are sized by it. */
    void (*tp_shape)(const pulsar_engine *e, uint32_t *n_layer, uint32_t *n_embd, uint32_t *n_vocab);
    /** The speculative drafter the OPENED artifact carries (its weights loaded), or NONE (L272 B15:
     *  the one answer pulsar_engine_drafter gives; the server's lanes key on it). */
    pulsar_drafter_kind (*drafter)(pulsar_engine *e);
    /** Bits per weight of the routed experts -- the artifact's dominant quantisation, the quant field
     *  of the disk-KV (pulsar_segstore_identity) and TP identities.  EXL3 is its own value space:
     *  20 + the highest rate present in half bits (24 = K2 .. 36 = K8), so an EXL3 build never shares
     *  a store with the IQ2 (2) or MXFP4 (4) tier of the same model id.  0 = no routed experts. */
    int (*quant_bits)(pulsar_engine *e);
    /** The family's verify hooks for the speculation round API (spec_ops.h), or NULL = no drafter can
     *  run against this family's forward (L272 P1; a family without them does not declare CAP_SPEC). */
    const pulsar_spec_target_ops *spec;
    const pulsar_family_session_ops *session;
    /** The tokenizer and chat front (L272 P2), or NULL = none. */
    const pulsar_family_tokenizer *tokenizer;
    /** L272 P4b: DECLARES this rank's tensor-parallel slices of the loaded model into `plan` (the rank and group
     *  are the model's, e->model.tp_rank / tp_n_ranks): which tensors split, along which axis, the rank's ranges.
     *  The core chooses each operation by the tensor's format and runs it (tp_slice.cpp).  NULL = no TP. */
    bool (*tp_slices)(pulsar_engine *e, pulsar_tp_plan *plan);
    /** How the forward hands its linears their activation (pulsar_act_kind): what the core's arms and the TP
     *  plan's operations are chosen for. */
    pulsar_act_kind act_kind;
    /** L281: the family's image geometry (image_identity.cpp), or NULL = no images. */
    const pulsar_family_vision *vision;
    /** NULL = the DeepSeek graph pool's members (session_banks.cpp). */
    const pulsar_family_bank_ops *banks;
};

extern const pulsar_family PULSAR_FAMILY_DEEPSEEK4;
extern const pulsar_family PULSAR_FAMILY_QWEN4_EXP;

/** The family that claims the model's `general.architecture`.  Prints the
 * architecture and the known ones and returns NULL when none does; a missing
 * key is refused the same way. */
const pulsar_family *pulsar_family_for_model(const pulsar_model *m);

/** True when the engine's family declares `cap`.  Otherwise prints, once per
 * (op) call site, "<family> does not implement <op>" and returns false: the
 * caller refuses with its own error value.  A NULL engine passes (the caller's
 * own NULL check decides). */
bool pulsar_family_require(const pulsar_engine *e, uint32_t cap, const char *op);

#endif /* PULSAR_FAMILY_H */

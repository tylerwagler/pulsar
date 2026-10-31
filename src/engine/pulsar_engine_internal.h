/** @file pulsar_engine_internal.h
 *  @brief Internal shared declarations for the engine sources. */
/* pulsar_engine_internal.h — internal shared declarations for the engine sources.
 * Produced by the multi-TU split of ds4.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_ENGINE_INTERNAL_H
#define PULSAR_ENGINE_INTERNAL_H

/** =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 tensor layouts, CPU reference kernels, the whole-model GPU
 * graph driver, and tokenizer wiring.  Model shape selection is intentionally
 * narrow: validation accepts the known Flash and Pro layouts and fails early
 * for anything else.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until GPU wraps slices of the mapping as
 * no-copy GPU buffers.
 */

#include "tp/pulsar_tp.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <functional>

#include "pulsar.h"

#include "pulsar_gpu.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// Finite stand-in for -infinity in logits and scores (-1e30f).
#define PULSAR_NEG_INF (-1.0e30f)
/* RMSNorm epsilon is PER MODEL and must not be a shared "default": it is 1e-6
 * for 0731 and 1e-20 for V4.1.  The L218 work redefined this macro in place
 * (leaving "0731 was 1e-6" in a comment) rather than naming the second value,
 * which is how a per-model constant hides inside a shared one.  Each profile
 * names its own. */
/// RMSNorm epsilon of DeepSeek V4 (0731).
#define PULSAR_V4_RMS_EPS      ( 1.0e-6f)
/// RMSNorm epsilon of DeepSeek V4.1.
#define PULSAR_V41_RMS_EPS     ( 1.0e-20f)
/// Hyper-connection epsilon the shape profiles use.
#define PULSAR_DEFAULT_HC_EPS  ( 1.0e-6f)
/// SwiGLU clamp of the routed experts the shape profiles use.
#define PULSAR_DEFAULT_SWIGLU_CLAMP_EXP    (10.0f)
/// RoPE frequency base of the dense (uncompressed) layers, the profiles' value.
#define PULSAR_DEFAULT_ROPE_FREQ_BASE      (10000.0f)
/// YaRN scale factor the shape profiles use.
#define PULSAR_DEFAULT_ROPE_SCALE_FACTOR   (16.0f)
/// YaRN beta_fast the shape profiles use.
#define PULSAR_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
/// YaRN beta_slow the shape profiles use.
#define PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
/// RoPE frequency base of the compressed layers, the profiles' value.
#define PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
/// YaRN original context length, in tokens, the shape profiles use.
#define PULSAR_DEFAULT_ROPE_ORIG_CTX       UINT64_C(65536)

/** The V4.1 reasoning-effort line, byte-identical to the reference encoder
 * (encoding.py REASONING_EFFORT_TEMPLATE): the head, the decimal effort, the
 * tail.  Rendered before the system message whenever thinking is on; there
 * is no level that renders nothing (the 0731 "low adds nothing" scheme went
 * with 0731, L218). */
#define PULSAR_REASONING_EFFORT_HEAD "Reasoning Effort: "
/// The tail of the V4.1 reasoning-effort line, after the decimal effort.
#define PULSAR_REASONING_EFFORT_TAIL " (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

/** The V4 (0731) effort prefixes, byte-identical to the 0731 reference encoder
 * (encoding_dsv4.py REASONING_EFFORT_PROMPTS).  The 0731 release restructured
 * the levels: "low" (its default) adds nothing, "high" carries the text that
 * was "max" in the original release, and "max" gained a new stronger text.
 * They left the tree with the V4.1 numeric line (981090f7) and came back when
 * the served 0731 model was found rendering V4.1's line (L239). */
static const char PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// The V4 (0731) "max" effort prefix (see the block above PULSAR_V4_REASONING_EFFORT_HIGH_PREFIX).
static const char PULSAR_V4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Beyond maximum \xe2\x80\x94 exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";


#if defined(__GNUC__) || defined(__clang__)
/// Marks a definition that may be unused, silencing the warning (empty off GCC / clang).
#define PULSAR_MAYBE_UNUSED __attribute__((unused))
#else
/// Empty where the unused attribute is not available.
#define PULSAR_MAYBE_UNUSED
#endif

/** ---- shared macros ---- */



#define PULSAR_MODEL_SHAPE_NAME          (g_pulsar_shape.name)
/// The loaded shape profile's pulsar_shape::variant (g_pulsar_shape).
#define PULSAR_MODEL_VARIANT             (g_pulsar_shape.variant)
/// The loaded shape profile's pulsar_shape::n_layer (g_pulsar_shape).
#define PULSAR_N_LAYER                   (g_pulsar_shape.n_layer)
/// The loaded shape profile's pulsar_shape::n_embd (g_pulsar_shape).
#define PULSAR_N_EMBD                    (g_pulsar_shape.n_embd)
/// The loaded shape profile's pulsar_shape::n_vocab (g_pulsar_shape).
#define PULSAR_N_VOCAB                   (g_pulsar_shape.n_vocab)
/// The loaded shape profile's pulsar_shape::n_head (g_pulsar_shape).
#define PULSAR_N_HEAD                    (g_pulsar_shape.n_head)
/// The loaded shape profile's pulsar_shape::n_head_kv (g_pulsar_shape).
#define PULSAR_N_HEAD_KV                 (g_pulsar_shape.n_head_kv)
/// The loaded shape profile's pulsar_shape::n_head_dim (g_pulsar_shape).
#define PULSAR_N_HEAD_DIM                (g_pulsar_shape.n_head_dim)
/// The loaded shape profile's pulsar_shape::n_value_dim (g_pulsar_shape).
#define PULSAR_N_VALUE_DIM               (g_pulsar_shape.n_value_dim)
/// The loaded shape profile's pulsar_shape::n_rot (g_pulsar_shape).
#define PULSAR_N_ROT                     (g_pulsar_shape.n_rot)
/// The loaded shape profile's pulsar_shape::n_out_group (g_pulsar_shape).
#define PULSAR_N_OUT_GROUP               (g_pulsar_shape.n_out_group)
/// The loaded shape profile's pulsar_shape::n_lora_q (g_pulsar_shape).
#define PULSAR_N_LORA_Q                  (g_pulsar_shape.n_lora_q)
/// The loaded shape profile's pulsar_shape::n_lora_o (g_pulsar_shape).
#define PULSAR_N_LORA_O                  (g_pulsar_shape.n_lora_o)
/// The loaded shape profile's pulsar_shape::n_expert (g_pulsar_shape).
#define PULSAR_N_EXPERT                  (g_pulsar_shape.n_expert)
/// The loaded shape profile's pulsar_shape::n_expert_used (g_pulsar_shape).
#define PULSAR_N_EXPERT_USED             (g_pulsar_shape.n_expert_used)
/// The loaded shape profile's pulsar_shape::n_dspark_expert (g_pulsar_shape).
#define PULSAR_N_DSPARK_EXPERT           (g_pulsar_shape.n_dspark_expert)
/// The loaded shape profile's pulsar_shape::n_dspark_expert_used (g_pulsar_shape).
#define PULSAR_N_DSPARK_EXPERT_USED      (g_pulsar_shape.n_dspark_expert_used)
/// The loaded shape profile's pulsar_shape::n_expert_shared (g_pulsar_shape).
#define PULSAR_N_EXPERT_SHARED           (g_pulsar_shape.n_expert_shared)
/// The loaded shape profile's pulsar_shape::n_hash_layer (g_pulsar_shape).
#define PULSAR_N_HASH_LAYER              (g_pulsar_shape.n_hash_layer)
/// The loaded shape profile's pulsar_shape::q_head_norm (g_pulsar_shape).
#define PULSAR_Q_HEAD_NORM               (g_pulsar_shape.q_head_norm)
/// The loaded shape profile's pulsar_shape::chat_system_marker (g_pulsar_shape).
#define PULSAR_CHAT_SYSTEM_MARKER        (g_pulsar_shape.chat_system_marker)
/// The loaded shape profile's pulsar_shape::n_ff_exp (g_pulsar_shape).
#define PULSAR_N_FF_EXP                  (g_pulsar_shape.n_ff_exp)
/// The loaded shape profile's pulsar_shape::n_swa (g_pulsar_shape).
#define PULSAR_N_SWA                     (g_pulsar_shape.n_swa)
/// The loaded shape profile's pulsar_shape::n_indexer_head (g_pulsar_shape).
#define PULSAR_N_INDEXER_HEAD            (g_pulsar_shape.n_indexer_head)
/// The loaded shape profile's pulsar_shape::n_indexer_head_dim (g_pulsar_shape).
#define PULSAR_N_INDEXER_HEAD_DIM        (g_pulsar_shape.n_indexer_head_dim)
/// The loaded shape profile's pulsar_shape::n_indexer_top_k (g_pulsar_shape).
#define PULSAR_N_INDEXER_TOP_K           (g_pulsar_shape.n_indexer_top_k)
/// The loaded shape profile's pulsar_shape::candidate_block_size (g_pulsar_shape).
#define PULSAR_CANDIDATE_BLOCK_SIZE      (g_pulsar_shape.candidate_block_size)
/// The loaded shape profile's pulsar_shape::candidate_topk_blocks (g_pulsar_shape).
#define PULSAR_CANDIDATE_TOPK_BLOCKS     (g_pulsar_shape.candidate_topk_blocks)
/// The loaded shape profile's pulsar_shape::n_hc (g_pulsar_shape).
#define PULSAR_N_HC                      (g_pulsar_shape.n_hc)
/// The loaded shape profile's pulsar_shape::n_hc_sinkhorn_iter (g_pulsar_shape).
#define PULSAR_N_HC_SINKHORN_ITER        (g_pulsar_shape.n_hc_sinkhorn_iter)
/// The loaded shape profile's pulsar_shape::rms_eps (g_pulsar_shape).
#define PULSAR_RMS_EPS                   (g_pulsar_shape.rms_eps)
/// The loaded shape profile's pulsar_shape::hc_eps (g_pulsar_shape).
#define PULSAR_HC_EPS                    (g_pulsar_shape.hc_eps)
/// The loaded shape profile's pulsar_shape::expert_weight_scale (g_pulsar_shape).
#define PULSAR_EXPERT_WEIGHT_SCALE       (g_pulsar_shape.expert_weight_scale)
/// The loaded shape profile's pulsar_shape::swiglu_clamp_exp (g_pulsar_shape).
#define PULSAR_SWIGLU_CLAMP_EXP          (g_pulsar_shape.swiglu_clamp_exp)
/// The loaded shape profile's pulsar_shape::rope_freq_base (g_pulsar_shape).
#define PULSAR_ROPE_FREQ_BASE            (g_pulsar_shape.rope_freq_base)
/// The loaded shape profile's pulsar_shape::rope_scale_factor (g_pulsar_shape).
#define PULSAR_ROPE_SCALE_FACTOR         (g_pulsar_shape.rope_scale_factor)
/// The loaded shape profile's pulsar_shape::rope_yarn_beta_fast (g_pulsar_shape).
#define PULSAR_ROPE_YARN_BETA_FAST       (g_pulsar_shape.rope_yarn_beta_fast)
/// The loaded shape profile's pulsar_shape::rope_yarn_beta_slow (g_pulsar_shape).
#define PULSAR_ROPE_YARN_BETA_SLOW       (g_pulsar_shape.rope_yarn_beta_slow)
/// The loaded shape profile's pulsar_shape::compress_rope_freq_base (g_pulsar_shape).
#define PULSAR_COMPRESS_ROPE_FREQ_BASE   (g_pulsar_shape.compress_rope_freq_base)
/// The loaded shape profile's pulsar_shape::rope_orig_ctx (g_pulsar_shape).
#define PULSAR_ROPE_ORIG_CTX             (g_pulsar_shape.rope_orig_ctx)

/** =========================================================================
 * The 2-bit routed-expert block size.
 * =========================================================================
 *
 * QK_K is the super-block width (256 weights) of the one llama.cpp-derived
 * quant layout the engine still reads: IQ2_XXS (block_iq2_xxs below), stored
 * as the iq2_xxs_mmq_k container layout for DeepSeek's routed experts.  The
 * Q2_K and Q8_K blocks this section once described, and the GGUF container
 * they came in, are gone; model.cpp recognises a GGUF file only to refuse it
 * by name (PULSAR_GGUF_MAGIC).
 */
#define QK_K 256


/// Compile-time assertion: declares the array typedef `name`, ill-formed when `cond` is false.
#define PULSAR_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]


/** =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

#define PULSAR_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
/// Most dimensions a pulsar_tensor carries.
#define PULSAR_MAX_DIMS   8




/* KV row geometry is asked for by KIND through pulsar_kv_row_bytes(), which
 * reads the loaded profile's kv_row_style.  There is deliberately no
 * per-format macro left: a caller must not be able to name WINDOW/MAIN when the
 * loaded model stores one unified row.  See pulsar_kv_row_style. */


/** =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated GPU tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

#define PULSAR_SESSION_IO_CHUNK (8u * 1024u * 1024u)
/// The DSpark drafter's context window, in positions (its rings are indexed pos % this).
#define PULSAR_DSPARK_DRAFT_WINDOW 128u
/// The token id DSpark feeds at every draft position after the first.
#define PULSAR_DSPARK_NOISE_TOKEN_ID 128799

/* ---- shared types ---- */

/** =========================================================================
 * DeepSeek V4 Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator select one of the known model
 * profiles below.  Arrays reserve the maximum Pro dimensions; hot loops read
 * the active profile after GGUF validation.
 */

enum {
    PULSAR_MAX_LAYER            = 61,
    /* A source set can name any layer.  V4.1 shares (4 kv / 8 index sources),
     * but 0731 compresses every layer >= 2 on its own, so its set is the whole
     * backbone.  plans/96-SPIKE0-results.md S2. */
    PULSAR_MAX_ATTN_SOURCE      = PULSAR_MAX_LAYER,
    PULSAR_NO_LAYER             = 0xFFFFFFFFu,  ///< "no such layer" in the attention layout table
    PULSAR_MAX_EMBD             = 7168,
    PULSAR_MAX_VOCAB            = 129280,
    PULSAR_MAX_HEAD             = 128,
    PULSAR_MAX_HEAD_KV          = 1,
    PULSAR_MAX_HEAD_DIM         = 512,
    PULSAR_MAX_VALUE_DIM        = 512,
    PULSAR_MAX_ROT              = 64,
    PULSAR_MAX_OUT_GROUP        = 16,
    PULSAR_MAX_LORA_Q           = 1536,
    PULSAR_MAX_LORA_O           = 1024,
    PULSAR_MAX_EXPERT           = 384,
    PULSAR_MAX_EXPERT_USED      = 6,
    PULSAR_MAX_EXPERT_SHARED    = 1,
    PULSAR_MAX_FF_EXP           = 3072,
    PULSAR_MAX_SWA              = 128,
    PULSAR_MAX_INDEXER_HEAD     = 32,
    PULSAR_MAX_INDEXER_HEAD_DIM = 128,
    PULSAR_MAX_INDEXER_TOP_K    = 1024,
    PULSAR_MAX_HC               = 4,
    PULSAR_MAX_HC_SINKHORN_ITER = 20,
};
/* L188: the routed-MoE non-finite flag packs layer_index + 1 into 8 bits (pulsar_cuda_moe.cu) */
static_assert(PULSAR_MAX_LAYER < 255, "the non-finite flag's layer field is 8 bits");

/** The two architectures this engine serves.  They are adjacent revisions of
 * one family -- same attention lineage, same MoE, byte-identical tokenizer --
 * and everything that differs between them is selected from the profile.  This
 * enum is the ONE model-identity dispatch, and it is read at load time only
 * (pulsar_select_shape_from_metadata, expected_layer_compress_ratio); nothing
 * in the per-layer hot path may consult it.  See
 * plans/96-two-profiles-one-engine.md and plans/96-SPIKE0-results.md. */
typedef enum {
    PULSAR_VARIANT_V4  = 0,  /**< DeepSeek-V4-Flash (0731): 43 layers, 4/128 CSA/HCA, 3 hash layers */
    PULSAR_VARIANT_V41 = 1,  /**< DeepSeek-V4.1-Flash (L218): 40 layers, CSA2 sharing, no hash layers */
} pulsar_variant;

/* pulsar_kv_row_style and pulsar_kv_row_kind are declared in src/pulsar_gpu.h:
 * both sides of the seam need them (the engine sizes, the CUDA module packs),
 * so the definition lives on the shared header and neither side owns a copy. */

/** The model's architectural constants, resolved once at load and then treated
 * as compile-time-ish truth by the graph and kernels.
 *
 * Sizes here drive every buffer the engine allocates, so a mismatch against the
 * GGUF is a load-time failure rather than a runtime surprise. Note n_vocab is
 * the LOGITS row width -- the tokenizer table length lives on pulsar_vocab and
 * the two are not required to agree (see pulsar_engine_logits_width()). */
typedef struct {
    const char *name;          ///< human-readable profile name
    pulsar_variant variant;    ///< architecture variant selector
    uint32_t n_layer;          ///< transformer layers
    uint32_t n_embd;           ///< residual/embedding width
    uint32_t n_vocab;          ///< LOGITS row width; size logits buffers with this
    uint32_t n_head;           ///< query heads
    uint32_t n_head_kv;        ///< key/value heads (< n_head when grouped)
    uint32_t n_head_dim;       ///< per-head key/query dimension
    uint32_t n_value_dim;      ///< per-head value dimension (may differ from n_head_dim)
    uint32_t n_rot;            ///< dimensions covered by rotary embedding
    uint32_t n_out_group;      ///< output-head grouping factor
    uint32_t n_lora_q;         ///< rank of the low-rank query path (attn_q_a/q_b)
    uint32_t n_lora_o;         ///< rank of the low-rank attention-output path
    uint32_t n_expert;         ///< routed experts per MoE layer
    uint32_t n_expert_used;    ///< experts activated per token (top-k routing)
    uint32_t n_expert_shared;  ///< always-on shared experts
    uint32_t n_hash_layer;     ///< leading layers routed by token id, not by the gate (0731: 3; V4.1: 0)
    uint32_t n_dspark_expert;      ///< routed experts per DSpark drafter layer (V4.1: 128)
    uint32_t n_dspark_expert_used; ///< experts a drafter token activates (V4.1: 3)
    uint32_t n_ff_exp;         ///< per-expert FFN hidden width
    uint32_t n_swa;            ///< sliding-window attention span
    uint32_t n_indexer_head;      ///< indexer tower heads
    uint32_t n_indexer_head_dim;  ///< indexer per-head dimension
    uint32_t n_indexer_top_k;     ///< compressed rows the indexer selects per query
    /** CSA2 (L218): the layers that compress their own KV and publish it (with
     * the index keys derived from the same latent) to every later layer up to
     * the next source; the layers that run their own indexer and publish its
     * top-k the same way; and the one indexer whose block-max pool pre-filters
     * every later indexer (-1: none).  Ascending and inside the backbone.
     *
     * The two sets are NOT the same size in general: V4.1's are equal, but
     * 0731 has 41 kv sources against 21 index sources, and the 20 layers that
     * compress without an indexer are PULSAR_ATTN_FULL_UNINDEXED.  A kv source
     * is therefore NOT necessarily an index source -- derive the mode and ask
     * pulsar_attn_reads_index rather than assuming the sets coincide. */
    /// Number of kv-source layers in kv_source_layer[].
    uint32_t n_kv_source;
    /// The kv-source layer indices, ascending; first n_kv_source entries valid.
    uint32_t kv_source_layer[PULSAR_MAX_ATTN_SOURCE];
    /// Number of index-source layers in index_source_layer[].
    uint32_t n_index_source;
    /// The index-source layer indices (layers running their own indexer), ascending; first n_index_source valid.
    uint32_t index_source_layer[PULSAR_MAX_ATTN_SOURCE];
    /// The layer whose indexer's block-max pool pre-filters every later indexer; -1 = none.
    int32_t  candidate_source_layer;
    uint32_t candidate_topk_blocks;  ///< compressed-position blocks the candidate pool keeps per query
    uint32_t candidate_block_size;   ///< compressed positions per candidate block
    uint32_t n_hc;                ///< hyper-connection streams
    uint32_t n_hc_sinkhorn_iter;  ///< Sinkhorn normalisation iterations in the HC mix
    /** Attention weight families, read at bind only.
     * compressor_ape: the compressors carry an absolute-position embedding
     *   (0731 does; V4.1's are plain projections).
     * indexer_own_compressor: the indexer compresses its own index key
     *   (0731, indexer_compressor_*) instead of projecting it from the kv
     *   source's latent (V4.1, indexer_k / indexer_k_norm). */
    /// The compressors carry an absolute-position embedding (see the block above).
    bool compressor_ape;
    /// The indexer compresses its own index key (0731) rather than projecting it from the kv source (V4.1).
    bool indexer_own_compressor;
    /** The reference's Attention.forward normalises Q PER HEAD before the tail
     * rope (`q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`), on the
     * wq_b output.  0731 does; V4.1 does not -- its q_norm is an RMSNorm on the
     * low-rank latent, applied before wq_b, with no per-head pass.  Both are
     * "the reference", so this is a profile fact, not a choice.  The branch
     * carried only V4.1's arm and deleted 0731's, which left V4's Q roughly 30x
     * too small (dev's Qcur max 15.1 vs the branch's 0.51 for one prompt) and
     * drove layer 0's attention to NaN.  Restored from dev per PLAN 96 s5. */
    bool q_head_norm;
    /** The chat template writes a `<｜System｜>` MARKER before the lead-in system
     * region (V4.1's format).  0731's does not: dev -- the engine that served
     * 0731 -- writes the system text straight after BOS.  A profile fact
     * because the template is the model family's, and because the difference is
     * exactly one token in the prompt: with it, a 5-token prompt rendered to 15
     * tokens where dev renders 14, and the V4 model then answered a different
     * question ("In the world of" where dev says "The capital of France is
     * Paris.").  plans/96-... s9. */
    bool chat_system_marker;
    /** The head computes its OWN HC coefficients (0731: the output_hc_* and
     * dspark.2.hc_head_* mixes) instead of collapsing with the pre its last FFN
     * handed on (V4.1).  A profile fact rather than "are the tensors there",
     * because the requiredness has to be known when they are ABSENT: a 0731
     * artifact missing its mix would otherwise collapse with V4.1's arithmetic
     * and produce a wrong head without failing.  plans/96-... s9/S1. */
    bool hc_head_mix;
    /** Where the DSpark drafter's anchor hidden is captured inside a layer.
     * The reference appends `h.mean(dim=2)` at `target_layer_ids` BEFORE
     * `h = layer(h)` for V4.1 and AFTER the layer for 0731, so the drafter is
     * conditioned on the layer's INPUT in one case and its OUTPUT in the other.
     * False = capture at layer entry (V4.1); true = capture after the layer's
     * HC swap (0731).  Getting this wrong feeds the drafter a structurally
     * different hidden (measured 20-64% off) and walks its proposals off dev. */
    bool dspark_anchor_after;
    /** The image span this profile's tower feeds.  True (Vision-Exp, the V4
     * profile): compressor-aligned pads, the N-layout and a learned image_pad row
     * (vision.cpp vision_build_image_block) -- the one layout the image path
     * implements.  False (V4.1): reading-order rows, no pad tensor, and its own
     * preprocessing limits (inference/image_processor.py: 1024 tokens, 544^2
     * minimum pixels, no aspect cap) -- the tower binds and is validated without
     * image_pad, and the image path is refused until that layout exists. */
    bool image_span_padded;
    /** The KV row family (see pulsar_kv_row_style).  Every row-geometry
     * question goes through pulsar_kv_row_bytes(), which reads this -- so no
     * caller names a format and the geometry cannot disagree with the packer. */
    pulsar_kv_row_style kv_row_style;
    float rms_eps;             ///< epsilon for the transformer RMSNorms
    float hc_eps;              ///< epsilon for the HC normalisation
    float expert_weight_scale; ///< scale applied to routed-expert gate weights
    float swiglu_clamp_exp;    ///< clamp on the SwiGLU exponent (overflow guard)
    float rope_freq_base;      ///< RoPE base frequency
    float rope_scale_factor;   ///< RoPE frequency scaling (context extension)
    float rope_yarn_beta_fast; ///< YaRN fast-interpolation beta
    float rope_yarn_beta_slow; ///< YaRN slow-interpolation beta
    float compress_rope_freq_base;  ///< RoPE base used inside the KV compressor
    uint64_t rope_orig_ctx;    ///< context length the RoPE settings were trained at
} pulsar_shape;

/** The compressor ROW width: `coff*head_dim`.  One quantity, three users --
 * the kv/gate projection rows (batch_comp_kv/sc), the indexer-visible latent's
 * source, and the state lane, which is `coff*ratio` such rows.  Naming it once
 * is what stops a layer's buffer and the kernels that fill it from disagreeing
 * about whether the projection is split.
 *
 * `coff` is the SAME one that sets the compressor projection width
 * (pulsar_compress_coff), so the state and the weights cannot disagree about it
 * -- and they must not, because with overlap the projection is split: pooled
 * column c comes from the previous group's first half and column c+d from the
 * current group's second, so the state's two halves hold different things.
 *
 * V4.1 never overlaps (coff is 1 for every ratio it uses: 0, 1 and 2), so its
 * row is head_dim and its state `ratio` such rows, exactly as before.  V4's
 * ratio-4 layers are the ones that need the doubled form; see the transcription
 * in tests/compressor_pool_test.cpp for what the halves mean.
 *
 * head_dim is a parameter rather than read from the shape global so this can be
 * used before that global is declared (and from a host test). */
/// @param ratio     the layer's compression ratio (tokens per compressed row)
/// @param head_dim  the head dimension
/// @return the compressor state row width, in elements (coff * head_dim).
static inline uint32_t pulsar_comp_row_width(uint32_t ratio, uint32_t head_dim) {
    return pulsar_compress_coff(ratio) * head_dim;
}
/// Rows of a compressor's pending-group state at `ratio`: ratio rows, doubled where groups overlap.
/// @param ratio  the layer's compression ratio
/// @return the state's row count.
static inline uint32_t pulsar_comp_state_rows(uint32_t ratio) {
    return pulsar_compress_coff(ratio) * ratio;
}

/** ---- Engram n-gram hashing (L218 phase 0c / V4.1 phase 4) ------------------
 *
 * Engram is NEW IN V4.1 -- 0731 has no such layers (PLAN 95's delta table lists
 * it as "none" for 0731).  V4.1 adds it at layers 1 and 14 as ~203 GB (189 GiB)
 * of FP8 tables that are DISK-RESIDENT BY DESIGN: per layer an `embed`
 * [384M, 256] fp8 with a [384M, 8] scale plane, a `wkv` [25600, 6144] fp8 and
 * `q_weight`/`k_weight` [4, 5120] bf16.  That is far too large to bind into the
 * 121 GiB the box has, so the tables are side files beside the container (the
 * `wkv` / q / k ride in it under their HF names), opened by engram_bind and read
 * per step (L242 slice 5, below); a V4.1 artifact without them refuses to load.
 *
 * What makes a table this size viable at all is that its ADDRESSES ARE KNOWN
 * BEFORE THE FORWARD: the hash below depends only on token ids, so the rows a
 * token will need can be prefetched from NVMe while earlier layers compute.  The
 * hash runs on the HOST for exactly that reason.  (A drafter's ids are known one
 * step early too, which is where verify-time prefetch would come from.)
 *
 * The tables are addressed by an n-gram hash of the COMPRESSED token ids:
 * (max_ngram-1) x n_heads = 24 bucket rows per token per layer, one column per
 * (n-gram order, hash head).
 *
 * The layout (compressed token map, the per-layer primes, the bucket offsets and
 * the hash multipliers) is emitted by the artifact tooling and carried in the
 * manifest; gate-baseline/l218-v41/engram_hash.py is the reference that produced
 * it and gate-baseline/l218-v41/engram-layout-v41.json is its output.  The
 * compressed vocab is 99,092 and the bucket layout reproduces both table sizes
 * to the row (384,006,168 / 384,016,682).
 *
 * The hash is INTEGER arithmetic on non-negative values, and the bit patterns
 * are the point: `rolling` is an XOR, not a sum.  What makes the arithmetic
 * safe is that every multiplier is strictly positive and every compressed id is
 * in [0, compressed_vocab), so no product has its sign bit set and an XOR of
 * non-negative int64 is itself non-negative.  The layout guarantees the size
 * half of that (`mult = 2v+1` with `v < (INT64_MAX / compressed_vocab) / 2`, so
 * `tok * mult < INT64_MAX`); pulsar_engram_hash_pos enforces the range half.  A
 * plain `%` is therefore exact -- there is no floor-versus-truncate divergence
 * to defend against, and a `rolling` that came out negative would mean the
 * layout is corrupt, which is a refusal, not a different modulo. */
#define PULSAR_ENGRAM_MAX_NGRAM  4u
/// Hash heads per n-gram order.
#define PULSAR_ENGRAM_N_HEADS    8u
/** (n-gram orders 2..max) x heads: the columns one token contributes. */
#define PULSAR_ENGRAM_N_COLS     ((PULSAR_ENGRAM_MAX_NGRAM - 1u) * PULSAR_ENGRAM_N_HEADS)
/// Most Engram layers an artifact may carry.
#define PULSAR_ENGRAM_MAX_LAYERS 2u

/** Everything the hash needs, as the artifact carries it.  All pointers are
 * length-indexed as marked; none are owned here. */
typedef struct {
    uint32_t n_vocab;            ///< length of token_map (the model's token ids)
    uint32_t compressed_vocab;   ///< every token_map value is in [0, compressed_vocab)
    uint32_t pad_compressed_id;  ///< the id used for positions before the sequence starts
    uint32_t n_layers;           ///< engram layers (2)
    const int32_t  *token_map;      ///< [n_vocab] token id -> compressed id
    const int64_t  *multipliers;    ///< [n_layers][max_ngram] odd hash multipliers
    const uint32_t *primes;         ///< [n_layers][max_ngram-1][n_heads] bucket moduli
    const uint64_t *offsets;        ///< [n_layers][n_cols] bucket base per column
    const uint64_t *num_embeddings; ///< [n_layers] rows in each layer's table
} pulsar_engram_layout;

/** Write layer `layer`'s PULSAR_ENGRAM_N_COLS bucket rows for the token at
 * `pos` into `cols`, all in [0, num_embeddings[layer]).  Mirrors engram.py's
 * `NgramHashState.forward`. */
/// @param L      the hash layout
/// @param layer  Engram layer index, < L->n_layers
/// @param ids    the sequence's token ids
/// @param n_ids  entries in `ids`
/// @param pos    the position to hash, < n_ids (the n-gram reads back from it, padding before 0)
/// @param cols   out: PULSAR_ENGRAM_N_COLS row indices
void pulsar_engram_hash_pos(const pulsar_engram_layout *L, uint32_t layer,
                            const int32_t *ids, uint32_t n_ids, uint32_t pos,
                            uint32_t *cols);

/** ---- Engram TABLE (L242): the disk-resident rows and how they are fetched -----
 *
 * One table per Engram layer, a row file written by tools/engram/engram_rows.c:
 * a 64-byte header ("PENGRAM1", layer, n_rows, 264 / 256 / 8) then n_rows
 * records of 264 bytes -- the checkpoint's 256 E4M3 values followed by their 8
 * E8M0 block scales, interleaved so one read fetches a row whole.  The record
 * IS the device path's input (pulsar_gpu_engram_rows_emit), byte for byte.
 *
 * The file lives on the serving box's NVMe (189 GiB for both layers) and is
 * READ, never mapped: a token needs 24 rows of 264 bytes at addresses the hash
 * knows before the forward, and 48 random reads per token are what an NVMe
 * does well and what a 94 GiB mapping under page-cache pressure does badly.
 * A pool of pread threads (the box has no liburing) serves gathers; a gather
 * is issued ahead of the layer that needs it and waited on there, which is the
 * whole point of hashing on the host.  Open asserts the header against the
 * layout's row count; a mismatch is a wrong table, refused.
 *
 * The same pool and table serve Qwen3.8-Flash-Next's PLE n-gram table (L251,
 * src/engine/qwen_ngram.h): 320M rows of 160 bf16 held as 128 row-contiguous
 * tensors spread over the checkpoint's shard files.  So a table is a list of
 * PARTS -- (file, byte offset of the part's row 0) -- of `rows_per_part` rows
 * each (the last may be shorter), with one record size; an Engram row file is
 * the one-part case (offset 64, 264-byte records). */
#define PULSAR_ENGRAM_ROW_BYTES 264u
/// Bytes of an Engram row file's header (its rows start at this offset).
#define PULSAR_ENGRAM_HDR_BYTES 64u
/** The row-gather pool's threads.  A cold row is one NVMe read (~150 us) and a pread blocks its thread, so the
 *  pool is the gather's queue depth: 16 left a Qwen verify's ~72 rows four to eight reads deep while the GPU
 *  waited at the PLE layer; 64 keeps a decode or verify gather one or two deep (qwen-perf, 2026-10-11). */
#define PULSAR_ENGRAM_IO_THREADS 64u

/** One row-contiguous range of a table: rows [i * rows_per_part, ...) at `base`. */
typedef struct {
    int fd;                 ///< shared by the parts that live in one file
    uint64_t base;          ///< byte offset of the part's first row
} pulsar_engram_part;

/** One Engram layer's row table on disk: the file ranges its rows live in, read row by row through the
 *  pread pool (pulsar_engram_io).  Opened by pulsar_engram_table_open; owns `parts` and `path`. */
typedef struct pulsar_engram_table {
    pulsar_engram_part *parts;  ///< n_parts ranges, owned; NULL when closed
    /// Entries in parts[].
    uint32_t n_parts;
    uint64_t rows_per_part;     ///< rows in every part but possibly the last
    uint32_t row_bytes;         ///< bytes per row record (and per row of a gather's dst)
    uint32_t layer;             ///< the model layer this table serves
    uint64_t n_rows;            ///< rows in the table, == the layout's row count
    char *path;                 ///< owned copy of the first file's path, for messages
} pulsar_engram_table;

/** A gather in flight: `rows` row ids -> `dst` (n_rows x row_bytes, caller-owned,
 * must outlive the wait).  Completion is observed with pulsar_engram_gather_wait,
 * which returns 1 when every row landed and 0 with the failure printed. */
typedef struct pulsar_engram_gather pulsar_engram_gather;

/** The pool.  One per engine; every table's gathers go through it. */
typedef struct pulsar_engram_io pulsar_engram_io;

/// Open one Engram row file (a PENGRAM1 header, then its rows) as a one-part table.
/// @param t                 the table to fill (zeroed first)
/// @param path              the row file
/// @param layer             the model layer the table serves
/// @param n_rows_expected   rows the layout says the table holds; a mismatch is refused
/// @return 1 = open, 0 = refused (said).
int  pulsar_engram_table_open(pulsar_engram_table *t, const char *path, uint32_t layer,
                              uint64_t n_rows_expected);
/** A table of `n_parts` row-contiguous parts in existing files (a part's file
 * must hold `rows_per_part` rows -- fewer for the last -- of `row_bytes` at
 * `bases[i]`; checked against the file size).  Paths may repeat: each distinct
 * file is opened once.  1 = open. */
/// @param t              the table to fill (zeroed first)
/// @param layer          the model layer the table serves
/// @param n_parts        parts, == ceil(n_rows / rows_per_part)
/// @param paths          n_parts file paths
/// @param bases          n_parts byte offsets of each part's first row
/// @param rows_per_part  rows in every part but possibly the last
/// @param row_bytes      bytes per row record
/// @param n_rows         rows in the table
/// @return 1 = open, 0 = refused (said).
int  pulsar_engram_table_open_parts(pulsar_engram_table *t, uint32_t layer, uint32_t n_parts,
                                    const char *const *paths, const uint64_t *bases,
                                    uint64_t rows_per_part, uint32_t row_bytes, uint64_t n_rows);
/// Close a table's files (each shared fd once) and free its parts; leaves `t` zeroed.  NULL is a no-op.
/// @param t  the table
void pulsar_engram_table_close(pulsar_engram_table *t);
/// Start the row-gather pool.
/// @param n_threads  reader threads (PULSAR_ENGRAM_IO_THREADS in the engine)
/// @return the pool (free with pulsar_engram_io_destroy), or NULL for 0 threads or a thread that would not start.
pulsar_engram_io *pulsar_engram_io_create(uint32_t n_threads);
/// Stop and join the pool's threads and free it, dropping any queued read-ahead.  NULL is a no-op.
/// @param io  the pool
void pulsar_engram_io_destroy(pulsar_engram_io *io);
/** Issue: the pool starts reading immediately; returns NULL only on a bad argument. */
/// @param io      the pool
/// @param t       the table to read
/// @param rows    n_rows row indices (borrowed until the wait)
/// @param n_rows  rows to read
/// @param dst     n_rows * t->row_bytes bytes, row k at dst + k * row_bytes; must outlive the wait
/// @return the handle to pass to pulsar_engram_gather_wait.
pulsar_engram_gather *pulsar_engram_gather_start(pulsar_engram_io *io, const pulsar_engram_table *t,
                                                 const uint64_t *rows, uint32_t n_rows,
                                                 unsigned char *dst);
/** Block until the gather is complete; frees the handle.  1 = every row read. */
/// @param g  a handle from pulsar_engram_gather_start (NULL returns 0)
/// @return 1 when every row was read, 0 with the failure printed.
int  pulsar_engram_gather_wait(pulsar_engram_gather *g);
/** L289: warm the page cache with rows a gather will read soon (the pool reads them into its own buffer, and
 *  nothing waits on it): the later gather reads the same bytes, faster.  A hint -- it changes no bytes and a
 *  failed read is the later gather's to report. */
/// @param io      the pool
/// @param t       the table
/// @param rows    n_rows row indices (copied)
/// @param n_rows  rows to warm
void pulsar_engram_prefetch(pulsar_engram_io *io, const pulsar_engram_table *t, const uint64_t *rows, uint32_t n_rows);
/** L329 (b2): the pool's BACKGROUND readers -- at most this many read ahead at once, and only while no gather or
 *  prefetch is queued or being read (engram.cpp has the priority rule).  16 cold reads in flight (~150 us each on the
 *  NVMe) read a 4096-token chunk's 65,536 rows in ~0.6 s, inside the ~2.4 s that chunk computes (L329 window #476),
 *  and leave 48 of the 64 threads idle for the next decode gather. */
#define PULSAR_ENGRAM_AHEAD_READERS 16u
/** L329 (b2): the most read-ahead rows a pool holds queued and unread -- four 4096-token chunks of Qwen PLE rows
 *  (16 a token).  A cold row costs about one 4 KB page of cache (0.94 distinct rows per page, L329 (b)), so the
 *  read-ahead never has more than ~1 GiB of reads outstanding; a read-ahead that would pass it is refused whole. */
#define PULSAR_ENGRAM_AHEAD_PENDING_MAX (4ull * 4096ull * 16ull)
/** L329 (b2): MemAvailable must exceed this plus the pool's unread cap (a page a row) for a read-ahead to be queued (else it is
 *  skipped and counted): under memory pressure the page cache it would fill is the reclaim the box is already
 *  doing, so the read-ahead stands down rather than add to it.  8 GiB = twice the server's live floor
 *  (PULSAR_SERVER_MEM_FLOOR_BYTES) and well above where systemd-oomd's pressure kill and earlyoom act. */
#define PULSAR_READ_AHEAD_MEM_HEADROOM (8ull << 30)
/** L329 (b2): warm the page cache with rows a LATER gather will read -- the next prompt chunk's -- on the pool's
 *  background class: below every gather and prefetch, at most PULSAR_ENGRAM_AHEAD_READERS readers, each row into a
 *  throwaway buffer.  A hint: it changes no byte any gather returns.  `tag` names the issuer for the cancel.  Returns
 *  the rows queued (0 on a bad argument). */
/// @param io      the pool
/// @param t       the table
/// @param rows    n_rows row indices (copied)
/// @param n_rows  rows to read ahead
/// @param tag     the issuer, for pulsar_engram_read_ahead_cancel
/// @return the rows queued (0 on a bad argument).
uint32_t pulsar_engram_read_ahead(pulsar_engram_io *io, const pulsar_engram_table *t, const uint64_t *rows,
                                  uint32_t n_rows, uint64_t tag);
/** L329 (b2): count a read-ahead not queued for memory pressure (pulsar_read_ahead_admit). */
/// @param io  the pool (NULL is a no-op)
void pulsar_engram_read_ahead_skipped(pulsar_engram_io *io);
/** L329 (b2): drop the rows of `tag`'s read-aheads no reader has taken yet (rows in flight finish).  Returns them. */
/// @param io   the pool (NULL returns 0)
/// @param tag  the issuer whose read-aheads to drop
/// @return the rows dropped.
uint32_t pulsar_engram_read_ahead_cancel(pulsar_engram_io *io, uint64_t tag);
/** L329 (b2): the background class's counters since the pool started (pulsar.h pulsar_read_ahead_stats). */
/// @param io   the pool (NULL: `out` zeroed)
/// @param out  receives the counters, in-flight readers and unread pending rows
void pulsar_engram_read_ahead_stats(pulsar_engram_io *io, pulsar_read_ahead_stats *out);

/** ---- Engram on the DeepSeek forward (L242 slice 5; L269 W2) ------------------
 *
 * The reference (inference/model.py Transformer.forward) runs, before block i's
 * own forward and before the drafter's anchor capture, for every Engram layer:
 *     h = engram(h, hashes[:, :, layer_hash_index], mask)
 * hashes from the n-gram of COMPRESSED ids ending at each position (engram.py
 * NgramHashState).  Here: the artifact carries the hash layout and names the row
 * tables (tools/container/kv.py; deepseek4.engram.*), engram_bind opens them at
 * load; every forward stages its rows' hashes and starts the gathers before the
 * first layer (gpu_graph_engram_stage), and the Engram layer waits on its gather
 * and runs rows -> MX slot -> `wkv` -> gate + add on the HC copies
 * (gpu_graph_engram_apply), all in engram_forward.cpp.  The device path is slices 3-4's, gated by
 * cuda-engram-gate. */
typedef struct pulsar_engram_model {
    uint32_t n_layers;                               ///< Engram layers in this artifact (V4.1: 2; a fixture may keep fewer)
    uint32_t block[PULSAR_ENGRAM_MAX_LAYERS];        ///< the block each serves (the artifact's index)
    pulsar_engram_layout layout;                     ///< the hash, over the arrays below (owned)
    /// Owned storage behind layout.token_map: [layout.n_vocab] token id -> compressed id.
    int32_t *token_map;
    /// Owned storage behind layout.multipliers: [n_layers][PULSAR_ENGRAM_MAX_NGRAM] odd hash multipliers.
    int64_t *multipliers;
    /// Owned storage behind layout.primes: [n_layers][PULSAR_ENGRAM_N_COLS] bucket moduli.
    uint32_t *primes;
    /// Owned storage behind layout.offsets: [n_layers][PULSAR_ENGRAM_N_COLS] bucket base row per column.
    uint64_t *offsets;
    /// Storage behind layout.num_embeddings: rows in each layer's table.
    uint64_t num_embeddings[PULSAR_ENGRAM_MAX_LAYERS];
    /// Each Engram layer's row table, opened at bind.
    pulsar_engram_table table[PULSAR_ENGRAM_MAX_LAYERS];
    pulsar_engram_io *io;                            ///< the pread pool every table's gathers go through
    pulsar_gpu_tensor *qk[PULSAR_ENGRAM_MAX_LAYERS]; ///< f32 [n_hc][dim]: q_weight * k_weight, resident
} pulsar_engram_model;

/** IQ2_XXS weight block: 2-bit quants addressed through a shared codebook.
 *
 * `qs` is not raw quants -- it packs indices INTO a fixed grid of 8-value
 * patterns plus the sign bits, which is how the format reaches ~2.06 bits per
 * weight. Decoding needs the grid table, not just these bytes. */
typedef struct {
    uint16_t d;                  ///< block scale, f16
    uint16_t qs[QK_K / 8];       ///< packed codebook indices and sign bits
} block_iq2_xxs;


/** A borrowed string slice: pointer plus length, NOT NUL-terminated.
 *
 * GGUF strings are length-prefixed and live inside the mapping, so copying
 * them to make them NUL-terminated would mean allocating for every metadata
 * key in the file. */
typedef struct {
    const char *ptr;  ///< first byte; points into the mapping, not owned
    uint64_t len;     ///< length in bytes
} pulsar_str;

/// The engine's internal name for a growable token array (pulsar.h pulsar_tokens).
typedef pulsar_tokens token_vec;

/** Bounds-checked sequential reader over a byte buffer, used to parse the GGUF
 * header without trusting its length fields.
 *
 * Every read bounds-checks against `size` before advancing `pos`, and returns
 * false rather than reading out of range. Each read reports its OWN result --
 * a failure does not disable the cursor, so callers must check every call
 * (weights.cpp does). What is first-wins is the MESSAGE: set_error() only
 * writes `error` when it is still empty, so the text names the first failure
 * and the offset it happened at, not the most recent one. */
typedef struct {
    const uint8_t *base;   ///< start of the buffer being read
    uint64_t size;         ///< buffer length; the bound every read is checked against
    uint64_t pos;          ///< current read offset
    char error[256];       ///< first failure message; empty while the cursor is healthy
} pulsar_cursor;


/** =========================================================================
 * Model Mapping and Metadata Values.
 * =========================================================================
 *
 * The loader maps every shard once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the payloads into private
 * structures.
 *
 * The metadata VALUE codes below are Pulsar's own.  They used to be ggml's,
 * which was the last place a dead project's numbering survived: a safetensors
 * checkpoint stores its values with a TYPE NAME (a pulsar.kv entry is
 * {"key":..,"type":"u32","value":..}), so that name is the only authority and
 * these numbers never leave this process.  They are payload-agnostic -- the
 * tensor LAYOUTS are a separate vocabulary entirely (PULSAR_TENSOR_* in
 * pulsar_gpu.h).
 */

enum {
    PULSAR_META_UINT8   = 0,
    PULSAR_META_INT8    = 1,
    PULSAR_META_UINT16  = 2,
    PULSAR_META_INT16   = 3,
    PULSAR_META_UINT32  = 4,
    PULSAR_META_INT32   = 5,
    PULSAR_META_FLOAT32 = 6,
    PULSAR_META_BOOL    = 7,
    PULSAR_META_STRING  = 8,
    PULSAR_META_ARRAY   = 9,
    PULSAR_META_UINT64  = 10,
    PULSAR_META_INT64   = 11,
    PULSAR_META_FLOAT64 = 12,
};

/* The tensor layout vocabulary is ONE spelling, in pulsar_gpu.h: the CUDA MoE
 * dispatch keys on the same numbers, and the engine's tensor `type` field
 * holds one of them.  Each layout's byte model lives with the container
 * declaration that must account for it exactly (st_bytes_for / 
 * cutlass_mxfp4_expert_layout), not in a table indexed by a foreign id. */


/** The drafter markov_w2 table's storage, derived from its GGUF type -- the one
 * place the type -> kernel-arm mapping is spelled (L213).  Any other type is a
 * refusal, not a default: the loader's dims contract already rejected it, so
 * reaching here with one is a bug. */
/// @param msg  the message, printed as "pulsar: <msg>" before exit(1)
void pulsar_die(const char *msg);   /* declared in full further down; needed here */
/// The drafter markov_w2 table's kernel arm for its storage type (see the comment above pulsar_die).
/// @param type  the tensor's PULSAR_TENSOR_* type: F32, BF16 or FP8_E4M3_SOA_K
/// @return a PULSAR_MARKOV_W2_* format; any other type dies.
static inline int pulsar_markov_w2_fmt(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_F32:           return PULSAR_MARKOV_W2_F32;
    case PULSAR_TENSOR_BF16:          return PULSAR_MARKOV_W2_BF16;
    case PULSAR_TENSOR_FP8_E4M3_SOA_K: return PULSAR_MARKOV_W2_MXFP8;
    default: pulsar_die("markov_w2: unsupported storage type"); return -1;
    }
}

/** One GGUF metadata entry, held as a key plus an OFFSET rather than a parsed
 * value: values vary in type and length, and most are never read, so parsing
 * is deferred to whoever actually asks for the key. */
typedef struct {
    pulsar_str key;     ///< metadata key, borrowed from the mapping
    uint32_t type;      ///< PULSAR_META_* code of the value
    uint64_t value_pos; ///< byte offset of the value within the file
} pulsar_kv;

/** One entry of the tensor directory: where a tensor lives and how to read
 * it. Describes bytes inside a mapped shard; owns nothing. */
typedef struct {
    pulsar_str name;      ///< the engine's canonical tensor name
    uint32_t ndim;        ///< number of used entries in dim[]
    uint64_t dim[PULSAR_MAX_DIMS];  ///< extents, fastest-varying first
    uint32_t type;        ///< storage type (see the PULSAR_TENSOR_* family)
    uint64_t rel_offset;  ///< offset from the file's tensor-data section start
    uint64_t abs_offset;  ///< offset from the start of the mapping -- what the kernels take
    uint64_t elements;    ///< product of dim[0..ndim)
    uint64_t bytes;       ///< on-disk size, quantisation included
    /** Set only when this entry was swapped in from an overlay GGUF
     * (--expert-overlay): the payload lives at ext_map + abs_offset inside
     * the overlay file's mapping instead of the owning model's map. */
    const uint8_t *ext_map;
    uint64_t ext_size;   ///< size of that overlay mapping, for the bounds check
    /** L295 E1: the builder's BIND RECORD for this entry -- its pulsar.tensors / pulsar.experts `role`, `block`,
     *  `layer` and `index` -- what a role-driven binder (role_bind.h) binds by instead of the name.  Each part is
     *  absent (empty / has_* false) when the record does not carry it: DeepSeek's and Qwen's containers write none
     *  (their binders go by name; Qwen's expert records carry a layer and block, unread). */
    /// The bind record's `role` (empty when absent).
    pulsar_str bind_role;
    /// The bind record's `block` (empty when absent).
    pulsar_str bind_block;
    int32_t bind_layer;   ///< valid when has_bind_layer
    int32_t bind_index;   ///< valid when has_bind_index
    /** @var pulsar_tensor::has_bind_index
     *  bind_index is valid (the record carries an `index`). */
    /// bind_layer is valid (the record carries a `layer`).
    bool has_bind_layer, has_bind_index;
} pulsar_tensor;

/** A memory-mapped GGUF file plus its parsed directory.
 *
 * The mapping stays live for the engine's lifetime and every weight pointer in
 * pulsar_layer_weights is a borrowed view into it -- nothing copies tensor data
 * to the host. */
typedef struct {
    int fd;                 ///< open file descriptor backing the mapping
    const uint8_t *map;     ///< base of the read-only mapping
    /** The PRIMARY mapping's length.  This is the whole file for a one-file
     * model, and ONE SHARD for a safetensors checkpoint -- so it is NOT the
     * model's footprint.  Anything accounting for the model's memory must use
     * mapped_bytes; `size` is only for code that has already resolved which
     * mapping it means (tensor_map_size falls back to it). */
    uint64_t size;
    /** Every shard's length summed: the checkpoint's mapped footprint.  This is
     * what the admission budget, --inspect and the load banner report; before
     * it existed all three disagreed, and the budget was the one that mattered
     * (it read one shard, 0.88 GiB, and over-stated the budget by ~85 GiB). */
    uint64_t mapped_bytes;
    /** Slice 4f (L237): the rank this process loads the model FOR and the TP
     * group size (rank 0 of 1 when the pair is off).  Set by the engine from
     * its options BEFORE the tensor staging pass, because staging IS
     * residency on GB10: under TP the stored routed-expert stacks are not
     * staged (each rank builds its half of every expert at open) and the
     * admission budget charges the same bytes.  The transport, created after
     * the load, is asserted to come up as this same rank. */
    int tp_rank;
    /// The TP group size (1 when the pair is off).
    uint32_t tp_n_ranks;
    /** The residency rule under TP (L272 P4b): tp_unstaged[i] marks tensor i as never staged -- a stored tensor
     *  the rank's plan replaces (a slice built at open, a routed stack's half, the other rank's experts).  Set
     *  by pulsar_tp_plan_build; NULL = nothing unstaged. */
    uint8_t *tp_unstaged;
    /** L298: unloaded[i] marks tensor i as not part of what this engine serves -- a drafter the open did not select
     *  (pulsar_role_binder_unload): never bound, never staged, never priced.  NULL = every tensor is served. */
    uint8_t *unloaded;
    /** L272 P4b: the engine's TP plan (owned by the engine; tp_slice.cpp), for the forward's lookup of a built
     *  slice (pulsar_tp_built_ptr).  NULL on one GPU. */
    struct pulsar_tp_plan *tp_plan;

    uint32_t version;       ///< GGUF format version
    uint64_t n_kv;          ///< metadata key/value pair count
    uint64_t n_tensors;     ///< tensor directory entry count
    uint64_t alignment;     ///< tensor data alignment declared by the file
    uint64_t tensor_data_pos;   ///< byte offset where the tensor data section starts
    uint64_t max_tensor_bytes;  ///< largest single tensor, for staging-buffer sizing

    pulsar_kv *kv;             ///< parsed metadata pairs, n_kv entries
    pulsar_tensor *tensors;    ///< parsed tensor directory, n_tensors entries

    /* ---- safetensors container (n_shards > 0) ------------------------------
     * A safetensors model is one shard per layer -- several mmapped files --
     * rather than one mapping, so each tensor selects its own shard through the
     * existing ext_map/ext_size fields, the same mechanism --expert-overlay
     * already uses.  The GGUF path leaves all of these zero and keeps using the
     * single fd/map/size above.
     *
     * kv_base is where cursor_at() reads metadata values from.  A GGUF file
     * keeps them inline in its mapping; a safetensors file carries them as JSON
     * text, which safetensors_open() re-encodes once into one GGUF-typed blob
     * so every existing kv consumer is unchanged. */
    uint64_t n_shards;              ///< number of mmapped shards, 0 for GGUF
    int *shard_fd;                  ///< one fd per shard
    const uint8_t **shard_map;      ///< one mapping base per shard
    uint64_t *shard_size;           ///< one mapping length per shard
    const uint8_t *kv_base;         ///< metadata value buffer (NULL => map)
    uint64_t kv_size;               ///< length of that buffer
    /** L301: the container's identity -- FNV-1a over each shard's name and safetensors header (the declared
     *  layout: every tensor's name, dtype, shape and offsets, and the metadata), in shard order.  What a cache of
     *  measurements taken on this container (the verify cost curve, verify_curve.cpp) is keyed by. */
    uint64_t container_digest;
} pulsar_model;

/* The model-family interface (L251 S1): the family descriptor, the layer plan,
 * and the Qwen4-exp family's shape/weights/state/op contract. */
#include "family.h"
/* L295 F4: the core row-step driver's types (a driver family's session state and step extend them). */
#include "step_driver.h"
#include "family_qwen.h"
static_assert(PULSAR_FAMILY_MAX_LAYER >= PULSAR_MAX_LAYER,
              "a layer plan must hold every layer a DeepSeek profile can have");

/* The DeepSeek family's entries (family_deepseek.cpp points at them; bodies in
 * session.cpp, where pulsar_engine::open and pulsar_session::create kept them
 * until L251). */
/// DeepSeek's family load: vocabulary, weight binding and drafter for an engine whose model is mapped.
/// @param e    the engine being opened
/// @param opt  its options (an external --drafter is refused)
/// @return false (said) on a refusal.
bool pulsar_ds4_family_load(pulsar_engine *e, const pulsar_engine_options *opt);
/// DeepSeek's after-GPU step of open: the MoE-tier boot line, the Engram upload and the TP group range.
/// @param e  the engine, GPU up
/// @return false (said) on a failure.
bool pulsar_ds4_family_after_gpu(pulsar_engine *e);
/* L272 P4b: DeepSeek's TP slices for this rank (the family's tp_slices op) */
/// @param e     the engine
/// @param plan  receives this rank's slices
/// @return false (said) when the rank's groups or a slice is refused.
bool pulsar_ds4_tp_slices(pulsar_engine *e, pulsar_tp_plan *plan);
/// DeepSeek's session create: allocate the GPU graph (KV, scratch, bank pool) for `s->ctx_size`.
/// @param s        the session being created (engine, ctx_size, prefill_cap set)
/// @param n_banks  banks in the pool (1 = no pool)
/// @return 0 on success, 1 on failure (said).
int pulsar_ds4_session_create(pulsar_session *s, uint32_t n_banks);
/// DeepSeek's session destroy: free (or, for a priced dry session, release) the graph.  No graph: a no-op.
/// @param s  the session
void pulsar_ds4_session_destroy(pulsar_session *s);

/** A GGUF metadata array left UNPARSED: its type, length, and where its
 * elements start. Reading an array means walking the file from `data_pos`, and
 * most arrays are never read at all. */
typedef struct {
    uint32_t type;      ///< GGUF type code of the elements
    uint64_t len;       ///< element count
    uint64_t data_pos;  ///< byte offset of the first element
} pulsar_array_ref;

/** Half-open byte range [off, end) of the model mapping that an accelerator
 * must have resident. Used to prefetch/pin exactly the spans a step touches. */
typedef struct {
    /** The mapping this span lives in.  A GGUF is one mapping, but a
     * safetensors checkpoint is one per layer, so a span has to name its own
     * base and length or the accelerator would cache layer 5's bytes at
     * layer 6's offset. */
    const uint8_t *base;
    uint64_t map_size;  ///< that mapping's length, for the bounds check
    uint64_t off;       ///< first byte, offset into that mapping
    uint64_t end;       ///< one past the last byte
} accelerator_tensor_span;

/** Every weight tensor for ONE transformer layer, resolved from the GGUF at
 * load time. Pointers are borrowed views into the memory-mapped model and stay
 * valid for the engine's lifetime; a NULL member means the layer does not have
 * that tensor (e.g. the compressor set exists only on compressing layers, and
 * the routed-expert set only on MoE layers).
 *
 * Layout mirrors the DeepSeek-V4-Flash block: an HC (hyper-connection) mix
 * around each of the two sublayers, attention with a low-rank Q path and a
 * shared KV projection, a KV COMPRESSOR that emits one pooled row every
 * `ratio` positions, a lighter INDEXER tower that scores which compressed rows
 * to attend to, and a mixture-of-experts FFN with a always-on shared expert. */
typedef struct {
    pulsar_tensor *hc_attn_fn;       ///< HC mix weight feeding the attention sublayer
    pulsar_tensor *hc_attn_scale;    ///< HC per-channel scale, attention side
    pulsar_tensor *hc_attn_base;     ///< HC per-channel base/offset, attention side
    pulsar_tensor *attn_norm;        ///< RMSNorm weight before attention
    pulsar_tensor *attn_q_a;         ///< query down-projection (low-rank A factor)
    pulsar_tensor *attn_q_a_norm;    ///< RMSNorm on the low-rank query latent
    pulsar_tensor *attn_q_b;         ///< query up-projection (low-rank B factor) to head space
    pulsar_tensor *attn_kv;          ///< fused key/value projection
    pulsar_tensor *attn_kv_a_norm;   ///< RMSNorm on the KV latent
    pulsar_tensor *attn_sinks;       ///< per-head attention sink logits (always-attendable slots)
    pulsar_tensor *attn_output_a;    ///< attention output down-projection (A factor)
    pulsar_tensor *attn_output_b;    ///< attention output up-projection (B factor) back to embedding
    pulsar_tensor *attn_compressor_ape;   ///< compressor absolute-position embedding; 0731 compressed layers only
    pulsar_tensor *attn_compressor_kv;    ///< compressor KV projection [n_embd -> head_dim]; every kv source
    pulsar_tensor *attn_compressor_gate;  ///< compressor softmax gate over the group's rows; kv sources of ratio > 1
    pulsar_tensor *attn_compressor_norm;  ///< RMSNorm on the pooled latent; every kv source
    pulsar_tensor *indexer_attn_q_b;      ///< indexer query up-projection (its own head space); indexed layers
    pulsar_tensor *indexer_proj;          ///< indexer per-head score weights; indexed layers
    pulsar_tensor *indexer_k;             ///< index key from the kv source's latent; V4.1 indexed layers
    pulsar_tensor *indexer_k_norm;        ///< RMSNorm on that index key; V4.1 indexed layers
    /** The indexer's OWN compressor (0731 indexed layers).  0731 does not
     * derive the index key from the kv source's latent -- it compresses the
     * index key itself, so these replace indexer_k / indexer_k_norm.  Which
     * family a profile uses is pulsar_shape::indexer_own_compressor. */
    /// Absolute-position embedding of the indexer's compressor.
    pulsar_tensor *indexer_compressor_ape;
    /// The indexer compressor's kv projection.
    pulsar_tensor *indexer_compressor_kv;
    /// The indexer compressor's softmax gate over the group's rows.
    pulsar_tensor *indexer_compressor_gate;
    /// RMSNorm on the indexer compressor's pooled key.
    pulsar_tensor *indexer_compressor_norm;
    pulsar_tensor *hc_ffn_fn;        ///< HC mix weight feeding the FFN sublayer
    pulsar_tensor *hc_ffn_scale;     ///< HC per-channel scale, FFN side
    pulsar_tensor *hc_ffn_base;      ///< HC per-channel base/offset, FFN side
    pulsar_tensor *ffn_norm;         ///< RMSNorm weight before the FFN
    pulsar_tensor *ffn_gate_inp;     ///< router projection producing per-expert logits
    pulsar_tensor *ffn_exp_probs_b;  ///< router bias added to the expert probabilities
    /** 0731's token-id -> expert-id table (I32 [n_expert_used, n_vocab]) for the
     * leading pulsar_shape::n_hash_layer layers, which route by token id instead
     * of by a top-k over the gate logits.  V4.1 ships none: NULL here means the
     * layer routes by the gate. */
    pulsar_tensor *ffn_gate_tid2eid;
    /** This layer's router width, the experts a token activates, and the experts
     * physically present in its stacks (REAP keep count on the target; the
     * full width on the drafter).  Set at bind; the FFN encoder reads THESE, so
     * a drafter layer routes with its own 128 / top-3 and never with the
     * target's 384 / top-6 (L218 audit risk #2). */
    /// Router width: routed experts the gate scores.
    uint32_t n_expert;
    /// Experts a token activates (top-k).
    uint32_t n_expert_used;
    /// Experts physically present in this layer's stacks (REAP keep count on the target).
    uint32_t n_expert_present;
    pulsar_tensor *ffn_exp_probs_b_vl; ///< image-token router bias (Vision-Exp only): the router adds THIS instead of ffn_exp_probs_b for tokens whose id is >= vocab_size
    pulsar_tensor *ffn_gate_exps;    ///< ROUTED experts, gate projection (expert-major)
    pulsar_tensor *ffn_up_exps;      ///< routed experts, up projection
    pulsar_tensor *ffn_down_exps;    ///< routed experts, down projection
    pulsar_tensor *ffn_gate_shexp;   ///< SHARED expert gate projection (runs for every token)
    pulsar_tensor *ffn_up_shexp;     ///< shared expert up projection
    pulsar_tensor *ffn_down_shexp;   ///< shared expert down projection
    /** Engram (V4.1 layers 1 and 14; engram_bind): the `wkv` projection of the
     * gathered rows (mxfp8_lt [24*256 -> (n_hc+1)*dim]) and the layer's slot in
     * pulsar_engram_model.  NULL on every other layer and on drafter blocks. */
    /// The `wkv` projection of the gathered Engram rows; NULL on non-Engram layers.
    pulsar_tensor *engram_wkv;
    /// This layer's index into pulsar_engram_model's per-layer arrays (meaningful only when engram_wkv is set).
    uint32_t engram_slot;
} pulsar_layer_weights;

/** Every weight tensor of the target model: the per-layer stacks plus the
 * embedding and output head. Tensors point into the model mapping; the struct
 * owns none of them. */
typedef struct {
    pulsar_tensor *token_embd;       ///< token embedding table
    pulsar_tensor *output_norm;      ///< final RMSNorm before the vocab projection
    pulsar_tensor *output;           ///< vocab projection (the output head)
    /** The HC HEAD MIX: the coefficients the final collapse uses.  0731 ships
     * these and its head computes its own coefficients; V4.1 ships none and
     * collapses with the pre its last FFN handed on (pulsar_shape::hc_head_mix
     * says which).  BOUND ONCE, not per layer -- the head is one tensor set.
     * plans/96-two-profiles-one-engine.md s9/S1. */
    pulsar_tensor *output_hc_fn;     ///< head mix weight [n_hc * n_embd -> n_hc]
    pulsar_tensor *output_hc_scale;  ///< head mix row scale
    pulsar_tensor *output_hc_base;   ///< head mix per-stream bias
    pulsar_layer_weights layer[PULSAR_MAX_LAYER];  ///< per-layer weight stacks
    /** Engram's tables, layout and pool (engram_bind); NULL for a model with no
     * Engram layer.  Owned: engram_model_free. */
    pulsar_engram_model *engram;
} pulsar_weights;

/** Open the artifact's Engram layers (deepseek4.engram.*): the hash layout, the
 * row tables beside the container (`model_path` names it), the `wkv` / q / k
 * tensors bound into `w->layer[block]` and `w->engram`.  An artifact that
 * declares Engram layers but carries no layout or no table refuses -- V4.1
 * without Engram is not the model.  No Engram layers -> w->engram stays NULL and
 * the forward is unchanged.  Returns false with the reason printed. */
/// @param w           the weights; w->engram is set when the artifact carries Engram layers
/// @param m           the model
/// @param model_path  the container's path; the row tables sit beside it
/// @return true (also with no Engram layers), false with the reason printed.
bool engram_bind(pulsar_weights *w, const pulsar_model *m, const char *model_path);
/** The device half of engram_bind (the q*k products), once the GPU is up. */
/// @param w  the weights (no Engram: a no-op)
/// @param m  the model
/// @return false when an upload fails.
bool engram_upload(pulsar_weights *w, const pulsar_model *m);
/// Free w->engram: the pool, the tables, the device q*k products and the layout arrays.  NULL-safe.
/// @param w  the weights
void engram_model_free(pulsar_weights *w);

/** DSpark drafter weights: the small model that proposes tokens for the target
 * to verify. Shipped inside the same GGUF as `dspark.*` tensors, so a drafter
 * is present or absent per artifact rather than per run.
 *
 * It reads the TARGET's hidden states at three anchor layers
 * (target_layer_ids) rather than running its own full stack -- which is why the
 * graph captures those hiddens during the target forward. */
typedef struct {
    pulsar_tensor *main_proj;   ///< projects captured target hiddens into the drafter's width
    pulsar_tensor *main_norm;   ///< RMSNorm on that projection
    pulsar_layer_weights layer[3];  ///< the drafter's own three transformer layers
    pulsar_tensor *markov_w1;   ///< Markov head, first projection (cheap next-token prior)
    pulsar_tensor *markov_w2;   ///< Markov head, second projection
    pulsar_tensor *confidence_proj;  ///< confidence head: scores how likely a draft is to be accepted, feeding the adaptive-depth controller
    pulsar_tensor *final_norm;       ///< RMSNorm before the drafter's vocab projection
    uint32_t embed_dim;              ///< drafter hidden width
    uint32_t vocab_size;             ///< drafter output width; must match the target's logits width
    uint32_t target_layer_ids[3];    ///< TARGET layer indices whose hiddens the drafter consumes
    /** The DRAFTER's HC head mix (dspark.2.hc_head_*): 0731 ships it and the
     * drafter computes its own head coefficients; V4.1 ships none.  Bound once
     * from block 2 -- the block whose hidden feeds the head (L216). */
    pulsar_tensor *hc_head_fn;       ///< drafter head mix weight
    pulsar_tensor *hc_head_scale;    ///< drafter head mix row scale
    pulsar_tensor *hc_head_base;     ///< drafter head mix per-stream bias
} pulsar_dspark_weights;

/** DeepSeek Vision-Exp tower weights.  The tower's SHAPE constants
 * (PULSAR_VISION_*) and the CUDA-facing offset contract live in pulsar_gpu.h:
 * the CUDA TUs cannot see this header, and every kernel takes (map, size,
 * offset) rather than engine types.  This struct is the engine-side binding;
 * vision_offsets_from_weights() flattens it for the kernels.
 *
 * The reference is the checkpoint's own inference/vision.py: RMSNorm(1e-6),
 * fused QKV, 2D RoPE with 16 frequencies per axis over the 64-wide head,
 * bias-free SwiGLU, bidirectional attention; and inference/model.py's
 * merge_image_embeddings, which permutes the aligner output into the text span
 * with image_start/pad/newline/end filling the non-IMAGE slots. */
typedef struct {
    pulsar_tensor *patch_proj;   ///< patch embedding (3*patch^2 -> dim), with bias
    /// Bias of patch_proj.
    pulsar_tensor *patch_bias;
    pulsar_tensor *norm;         ///< final RMSNorm over the tower output
    pulsar_tensor *aligner_w1;   ///< aligner 1st projection (dim*ratio^2 -> text dim)
    /// Bias of aligner_w1.
    pulsar_tensor *aligner_b1;
    pulsar_tensor *aligner_w2;   ///< aligner 2nd projection (text dim -> text dim)
    /// Bias of aligner_w2.
    pulsar_tensor *aligner_b2;
    pulsar_tensor *image_start;  ///< text-space embedding for an IMAGE_START slot
    /// Text-space embedding for an IMAGE_END slot.
    pulsar_tensor *image_end;
    /// Text-space embedding for an IMAGE_NEWLINE slot.
    pulsar_tensor *image_newline;
    /// Text-space embedding for an IMAGE_PAD slot.
    pulsar_tensor *image_pad;
    /** @var pulsar_vision_weights::block
     *  The tower's per-layer weights; the first n_layers entries are bound. */
    struct {
        pulsar_tensor *norm1;    ///< pre-attention RMSNorm
        pulsar_tensor *wqkv;     ///< fused QKV (dim -> 3*dim), with bias
        pulsar_tensor *wqkv_bias;
        pulsar_tensor *wo;       ///< attention output projection (dim -> dim), with bias
        pulsar_tensor *wo_bias;
        pulsar_tensor *norm2;    ///< pre-MLP RMSNorm
        pulsar_tensor *w1;       ///< SwiGLU gate+up (dim -> 2*inter), NO bias
        pulsar_tensor *w2;       ///< SwiGLU down (inter -> dim), NO bias
    } block[PULSAR_VISION_LAYERS];
    uint32_t n_layers;           ///< bound layers; equals PULSAR_VISION_LAYERS or the tower is absent
} pulsar_vision_weights;

/* THE WHOLE CPU Q8_0 SURFACE WAS HERE, and it is gone (2026-08-18).
 *
 * Six ctx structs (matvec_q8_0_ctx, _pair_ctx, _grouped_ctx,
 * matmul_q8_0_batch_ctx, _pair_batch_ctx, _grouped_batch_ctx),
 * quantize_mid_pairs_ctx, and nine declarations.  matvec_q8_0 was the only one
 * of them with a definition reachable from anywhere, and it had ZERO callers --
 * its comment still said "used heavily in decode", which stopped being true when
 * decode moved to the GPU.  The other five function declarations
 * (matvec_q8_0_pair_prequant, matvec_q8_0_grouped_rows, matmul_q8_0_batch,
 * matmul_q8_0_pair_batch, matmul_q8_0_grouped_batch) had no definition AT ALL:
 * their bodies were deleted at some earlier point and the prototypes outlived
 * them, which is why a grep for "q8" kept finding a CPU int8 path that could not
 * run.  block_q8_K and block_q2_K followed 2026-09-04 (L159 inc 3): nothing
 * read them; quant_formats.cpp only asserted their sizes, and went too. */

/* =========================================================================
 * KV Cache and Compressors.
 * =========================================================================
 *
 * Maintains raw SWA KV rows, optional compressed KV rows, the indexer mask
 * for ratio-4 layers, and a reusable decode scratch arena so token
 * generation does not allocate in the hot loop.
 */

/* =========================================================================
 * GPU Release Graph State.
 * =========================================================================
 *
 * The release GPU executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

/* Tier-2 multi-session bank pool (compile-time bound on co-scheduled
 * sessions; the runtime co-schedule cap is a later, smaller number).
 *
 * Multi-sequence batched-decode KV banking design (per-row positions[]/
 * seq_id[] descriptors over fixed per-bank KV slabs) adapted from the
 * MIT-licensed Entrpi/ds4 fork (https://github.com/Entrpi/ds4, v0.2,
 * c71a49ac9316db02eaa6322dee2c919e6de1e792).  Reimplemented from scratch
 * against this engine's packed MXFP8/MXFP4 KV layout; no Entrpi code was
 * copied. */
/** Raised 8 -> 16 (2026-08-10): banks are WARM-STATE slots, not decode
 * streams — decode throughput saturates ~4 concurrent streams, but every
 * bank beyond that keeps another conversation's KV warm between turns
 * instead of evicting it.  imatrix's run-head structure documents <= 16 as
 * its bound; the dense-step row cap (PULSAR_GPU_MNEUTRAL_ROWS_MAX) bounds
 * it, and the build refuses the drift below. */
#define PULSAR_MSEQ_MAX 16u
/** Every decode row of a batched step must fit the row cap's M-independent
 * kernels, or rows past the cap silently take a batch-shape-dependent GEMM
 * (this exact drift happened once: the caps were written when MSEQ_MAX was 8
 * and did not follow it to 16). The build refuses the drift now. */
static_assert(PULSAR_MSEQ_MAX <= PULSAR_GPU_MNEUTRAL_ROWS_MAX,
              "PULSAR_MSEQ_MAX exceeds the dense-step row cap; extend the "
              "NT kernel instantiations in pulsar_cuda_matmul.cu and the MoE "
              "boundary in pulsar_cuda_moe.cu, then raise "
              "PULSAR_GPU_MNEUTRAL_ROWS_MAX in pulsar_gpu.h");


/* L264/L265: the KV state model's model-neutral half; DeepSeek V4's answer is kv_state_ds4.cpp. */
#include "kv_state.h"
/* L295 F3: declared session state -- a family's state kinds, the core's alloc / paging / kv-state / capture. */
#include "state_kinds.h"
uint32_t pulsar_layer_compress_ratio(uint32_t il);   /* the frontier hook below needs it */

/** Declares the rows of every GEMM / MoE call issued inside its scope as
 * DECODE rows (pulsar_gpu_matmul_set_batch_decode_rows, pulsar_gpu.h): they
 * take the M-independent arms whatever the batch width.  Lanes that own decode
 * rows open one at their entry -- the classic verify block, the drafter's
 * forwards and seeds, the one-row output head; the batched step sets the
 * count itself in gpu_graph_multiseq_step_begin -- and the destructor restores
 * the caller's count, so a scope opened mid-step (a seed inside the fused spec
 * loop) leaves the step's declaration intact.  `ok()` is false when the setter
 * refused (n past PULSAR_GPU_MNEUTRAL_ROWS_MAX); the caller refuses too. */
class pulsar_decode_rows_scope {
public:
    /// Declare `n` decode rows for the scope, saving the caller's count; check ok() after.
    /// @param n    decode rows the scope's GEMM / MoE calls carry
    /// @param cap  the lane's row cap the setter checks `n` against
    explicit pulsar_decode_rows_scope(uint32_t n, uint32_t cap = PULSAR_GPU_MNEUTRAL_ROWS_MAX)
        : saved_(pulsar_gpu_matmul_batch_decode_rows()),
          ok_(pulsar_gpu_matmul_set_batch_decode_rows_capped((int)n, (int)cap) != 0) {}
    ~pulsar_decode_rows_scope() {
        /* restoring an accepted value (it passed its own lane's cap) */
        (void)pulsar_gpu_matmul_set_batch_decode_rows_capped(saved_, (int)PULSAR_DSPARK_DRAFT_ROWS_MAX);
    }
    /// @return false when the setter refused `n` (past the cap); the caller must refuse too.
    bool ok() const { return ok_; }
    pulsar_decode_rows_scope(const pulsar_decode_rows_scope &) = delete;
    pulsar_decode_rows_scope &operator=(const pulsar_decode_rows_scope &) = delete;
private:
    /// The caller's decode-row count, restored by the destructor.
    int  saved_;
    /// The setter accepted the scope's count.
    bool ok_;
};

/** Fixed per-bank KV slabs: per layer, one contiguous allocation per cache
 * kind, bank-major, stride = one bank's single-session capacity.  When the
 * pool is enabled (n_banks >= 2), the graph's per-layer cache pointers
 * (layer_raw_cache[il] etc.) are VIEWS into these slabs — a single-bank view
 * means every existing single-session code path (prefill, decode, snapshot,
 * spec) runs unmodified against that bank; the batched decode kernels address
 * other banks with per-row seq_id[t]*cap offsets over the whole slab.
 *
 * The ctx-scaled comp/index slabs are cudaMallocManaged: on GB10 unified
 * memory that is the demand-paged analog of the reference's cuMemAddressReserve
 * VMM scheme (physical pages materialize on first touch, address math is
 * byte-identical to an eager slab), so short sessions do not pay resident
 * memory for worst-case padding.  Raw rings and compressor state lanes are
 * eager (fixed floor).  n_banks == 0 means the pool is disabled and the graph
 * owns plain single-session cache tensors. */
typedef struct {
    uint32_t n_banks;   ///< pool size; 0 = disabled and the graph owns plain single-session tensors
    uint32_t cur_bank;  ///< bank the installed views currently address (0 when the pool is disabled)
    uint64_t raw_bank_bytes;                     ///< one bank's raw ring: raw_cap * WINDOW row bytes
    /** CSA2 (L218): the compressed pool, the index-K pool and the compressor
     * state lane exist only at a kv SOURCE layer's index; every other layer
     * reads its source's through pulsar_layer_attn_layout(il)->kv_source. */
    uint64_t comp_bank_bytes[PULSAR_MAX_LAYER];  ///< per kv source, one bank's compressed pool: layer_comp_cap * comp row bytes
    uint64_t index_bank_bytes[PULSAR_MAX_LAYER]; ///< per kv source, one bank's index-K pool: layer_comp_cap * index row bytes
    uint64_t astate_bank_bytes[PULSAR_MAX_LAYER];///< per ratio>1 kv source, one bank's compressor state lane (ratio rows x head_dim f32); 0 at ratio 1
    /** V4 ONLY (`pulsar_shape::indexer_own_compressor`): 0731's indexer compresses
     * its OWN key, so it keeps a SECOND recurrent lane per bank at the indexer's
     * head dim.  Same two authorities as the attention lane, the indexer's width;
     * 0 where the profile has no such lane (V4.1 derives its index key from the
     * latent and keeps none).  A banked V4 graph could not be built at all until
     * this lane existed -- handing every bank ONE lane would have had them share
     * the indexer's carry, which is a wrong answer rather than a crash (L218). */
    uint64_t istate_bank_bytes[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *raw[PULSAR_MAX_LAYER];    ///< per layer, the bank-major raw KV ring slab
    /** Tier-2 task #55 (increment 2a): the ctx-scaled comp/index caches are now
     * ONE cudaMallocManaged allocation PER BANK (comp[il][bank]) instead of one
     * n_banks*bank_bytes slab — so the increment-2 eviction guard can cudaFree a
     * single idle bank's physical directly (the only reclaim primitive that
     * returns memory on GB10; Step-1). The stride stays UNIFORM 1M (comp_bank_bytes
     * is per-layer, bank-independent) — this is NOT B-full (no variable caps). The
     * seq_id-scattered batched-decode READ kernels can no longer address a bank as
     * base + seq_id*comp_cap across one slab, so they take a per-bank BASE-POINTER
     * table (comp_bases[il]/index_bases[il]): a device array of the n_banks
     * comp[il][*]->ptr, indexed by seq_id[t]. NULL when the pool is disabled
     * (single-session paths use the repointed view). */
    pulsar_gpu_tensor *comp[PULSAR_MAX_LAYER][PULSAR_MSEQ_MAX];   ///< compressed KV, ONE managed allocation per (kv source, bank) so a single idle bank can be freed
    pulsar_gpu_tensor *index[PULSAR_MAX_LAYER][PULSAR_MSEQ_MAX];  ///< index-K cache, same per-(kv source, bank) shape
    pulsar_gpu_tensor *comp_bases[PULSAR_MAX_LAYER];  ///< device array of the n_banks comp[S][*] pointers, indexed by seq_id[t]; NULL when the pool is disabled
    pulsar_gpu_tensor *index_bases[PULSAR_MAX_LAYER]; ///< device array of the n_banks index[S][*] pointers, indexed by seq_id[t]
    pulsar_gpu_tensor *askv[PULSAR_MAX_LAYER];  ///< compressor state lane, KV half (ratio>1 kv sources)
    pulsar_gpu_tensor *assc[PULSAR_MAX_LAYER];  ///< compressor state lane, score half
    pulsar_gpu_tensor *iskv[PULSAR_MAX_LAYER];  ///< V4 only: indexer compressor state lane, KV half
    pulsar_gpu_tensor *issc[PULSAR_MAX_LAYER];  ///< V4 only: indexer compressor state lane, score half
    /* Tier-2 Option F: per-bank DSpark drafter context ring, bank-major
     * (~6.75 MB/bank: raw 0.75 + prompt 6).  Allocated in
     * gpu_graph_init_dspark_target only when the pool is enabled AND the
     * drafter is loaded; the graph's dspark_raw_cache[i]/dspark_prompt_h[i]
     * become bank views into these, swapped by gpu_graph_bank_repoint so the
     * spec path transparently uses the active bank's ring.  NULL otherwise. */
    /** plan-34 inc 6: per-bank SPEC FRONTIER SNAPSHOT lanes (same shapes as
     * askv/assc). The batched spec round snapshots EVERY decode
     * bank before the shared verify forward, so the single-set spec_* buffers
     * cannot hold them all; under banks the graph's spec_attn_state_*
     * become bank views into these, re-sliced by gpu_graph_bank_repoint
     * exactly like the live-state views (repoint already drops the baked
     * batched-copy tables, so the snapshot fast path re-prepares per bank).
     * NULL when the pool is spec-less. */
    pulsar_gpu_tensor *spec_askv[PULSAR_MAX_LAYER];  ///< spec frontier snapshot, compressor state KV; NULL when the pool is spec-less
    pulsar_gpu_tensor *spec_assc[PULSAR_MAX_LAYER];  ///< spec frontier snapshot, compressor state score
    pulsar_gpu_tensor *spec_iskv[PULSAR_MAX_LAYER];  ///< spec frontier snapshot, indexer compressor state KV; V4 only
    pulsar_gpu_tensor *spec_issc[PULSAR_MAX_LAYER];  ///< spec frontier snapshot, indexer compressor state score
    uint64_t dspark_raw_bank_bytes;      ///< one bank's drafter raw ring: DRAFT_WINDOW * WINDOW row (528 B)
    uint64_t dspark_prompt_bank_bytes;   ///< one bank's drafter prompt ring: DRAFT_WINDOW * n_embd * f32
    pulsar_gpu_tensor *dspark_raw[3];       ///< per draft layer, bank-major drafter raw ring; NULL without a pool or drafter
    pulsar_gpu_tensor *dspark_prompt[3];    ///< per draft layer, bank-major drafter prompt-hidden ring
    /** L260: each PARKED bank's spec-frontier batched-copy tables (the graph's spec_snap_copies /
     *  spec_restore_copies and their counts while that bank is installed).  The tables address the bank's state-lane
     *  views, whose slabs are allocated once at pool build and never move, so a bank switch parks the outgoing
     *  bank's tables here and installs the incoming bank's -- freeing them was a device-synchronizing cudaFree pair
     *  on every switch, rebuilt with cudaMalloc at the next snapshot.  The installed bank's slot is NULL (its
     *  tables live in the graph fields); NULL elsewhere = not built yet. */
    void *spec_snap_copies[PULSAR_MSEQ_MAX];
    /// Per bank: its parked spec -> layer batched-copy handle (the graph's spec_restore_copies).
    void *spec_restore_copies[PULSAR_MSEQ_MAX];
    /// Per bank: the parked tables' tensor count (the graph's spec_frontier_copy_n).
    uint32_t spec_frontier_copy_n[PULSAR_MSEQ_MAX];
    /// Per bank: the parked tables' largest single copy, in bytes (the graph's spec_frontier_copy_max_bytes).
    uint64_t spec_frontier_copy_max_bytes[PULSAR_MSEQ_MAX];
    /// Per bank: the parked spec_frontier_copy_init (0 unbuilt, 1 built).
    int spec_frontier_copy_init[PULSAR_MSEQ_MAX];
} pulsar_bank_slabs;

/** Every device buffer one session needs, plus the host bookkeeping that says
 * what is in them.
 *
 * NOT a graph in the framework sense -- there is no node list and nothing is
 * traversed. It is a fixed set of named allocations reused in place by every
 * layer, which is why the code below is verbose but predictable: each pointer
 * names an actual DS4 stage rather than a slot in a generic arena.
 *
 * Three things live here that are easy to mistake for each other:
 *
 *  - DEVICE BUFFERS, holding the tensors themselves.
 *  - HOST FRONTIER COUNTERS (the ms_* per-bank arrays), which say how much of
 *    each buffer is live. These are bookkeeping the multiseq driver owns;
 *    nothing on the device reads them. Read the compressed frontier through
 *    gpu_graph_n_comp(), never by reaching into the
 *    array -- stage 1b deleted the scalar twins precisely so there is one
 *    store to get wrong.
 *  - BANK VIEWS. When a pool is active the per-layer cache pointers are views
 *    into ::pulsar_bank_slabs rather than owned allocations, re-pointed by
 *    gpu_graph_bank_repoint. Freeing a view would free another bank's rows.
 *
 * During a multiseq step the scalar counters become cross-bank SUPERSETS
 * rather than any one bank's frontier -- see the multiseq block below, which
 * is the part to read before touching decode state.
 */

/** The images a prefill must merge, bound to the engine whose family front encodes them.  Lives
 * on the graph (pulsar_gpu_graph::vision_req) as a BORROW for the duration of one
 * prefill, the same way `prompt` is borrowed: the owner sets it before entering
 * the prefill and clears it on every exit.  Declared here because the graph
 * carries the pointer. */
typedef struct {
    /// The prefill's images, borrowed from the caller; n_images entries.
    const pulsar_image_ref *images;
    /// Entries in images[].
    int                     n_images;
    /// The engine whose family front encodes the images (borrowed).
    const pulsar_engine    *engine;
} pulsar_vision_request;

/** A session's GPU graph: every device buffer one session needs and the host bookkeeping over them (see the
 *  block comment above pulsar_vision_request).  The DeepSeek family's device state, one per session (pulsar_session::graph). */
typedef struct {
    /** One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    pulsar_gpu_tensor *cur_hc;    ///< the live HC residual carrier: token enters here, walks all layers
    pulsar_gpu_tensor *hc_split;  ///< per-stream split of the mix, before recombination
    pulsar_gpu_tensor *hc_post;   ///< HC state leaving the sublayer
    pulsar_gpu_tensor *hc_comb;   ///< recombined HC streams written back to cur_hc
    pulsar_gpu_tensor *attn_norm; ///< RMSNorm output feeding the projections
    pulsar_gpu_tensor *kv;        ///< KV latent after its RMSNorm; the row stored into the ring

    /** Per-layer KV.  Every layer keeps its own sliding-window ring.  The
     * compressed pool, the index-K pool and the compressor state live at the
     * kv SOURCE layer's index only (CSA2, L218): a member layer addresses
     * them through pulsar_layer_attn_layout(il)->kv_source, and the state is
     * the ratio-2 sources' one pending group-first projection (ratio 1 keeps
     * no state).  The state must be snapshotted with the row counter whenever
     * a checkpoint is saved or partially rewound. */
    pulsar_gpu_tensor *layer_raw_cache[PULSAR_MAX_LAYER];        ///< per layer, the sliding-window raw KV ring (NVFP4, 384 B/row)
    pulsar_gpu_tensor *layer_attn_comp_cache[PULSAR_MAX_LAYER];  ///< per kv source, pooled compressed rows (one per `ratio` positions)
    pulsar_gpu_tensor *layer_attn_state_kv[PULSAR_MAX_LAYER];    ///< per ratio>1 kv source, compressor accumulator KV half: the group being built (ratio rows x head_dim f32)
    pulsar_gpu_tensor *layer_attn_state_score[PULSAR_MAX_LAYER]; ///< compressor accumulator, score half
    pulsar_gpu_tensor *layer_index_comp_cache[PULSAR_MAX_LAYER]; ///< per kv source, index-K rows derived from the emitted latent
    /** V4 ONLY: the indexer's OWN compressor accumulator (`indexer_own_compressor`),
     * one lane per indexed ratio>1 source.  V4 builds `Compressor(..., head_dim=
     * PULSAR_N_INDEXER_HEAD_DIM, rotate=True)` INSIDE the Indexer, so the index
     * key is pooled from its own weights and is not the kv source's latent the
     * way V4.1's is; the lane is therefore a SECOND lane, sized by the same two
     * authorities at the indexer's head dim (`pulsar_comp_row_width(ratio, 128)`
     * x `pulsar_comp_state_rows(ratio)` = 8x256 floats at ratio 4).  NULL on a
     * V4.1 artifact: nothing would ever write it. */
    pulsar_gpu_tensor *layer_index_state_kv[PULSAR_MAX_LAYER];    ///< indexer compressor accumulator, KV half
    pulsar_gpu_tensor *layer_index_state_score[PULSAR_MAX_LAYER]; ///< indexer compressor accumulator, score half

    /** Speculative decoding scratch.  The drafter is allowed to mutate graph
     * state only if the target verifier can either commit it or restore the
     * saved frontiers. */
    pulsar_gpu_tensor *spec_attn_state_kv[PULSAR_MAX_LAYER];     ///< saved compressor state KV, per ratio>1 kv source
    pulsar_gpu_tensor *spec_attn_state_score[PULSAR_MAX_LAYER];  ///< saved compressor state score
    /** V4 only: the SPEC FRONTIER twin of the indexer-compressor state lane.  The
     * batched frontier copy set must cover every recurrent lane a rejected round
     * can move, and this one exists only where the indexer compresses its own key
     * (s119 found it missing; s123 made a banked V4 graph buildable at all). */
    pulsar_gpu_tensor *spec_index_state_kv[PULSAR_MAX_LAYER];    ///< saved indexer compressor state KV, per indexed ratio>1 source (V4)
    pulsar_gpu_tensor *spec_index_state_score[PULSAR_MAX_LAYER]; ///< saved indexer compressor state score
    /** Batched-copy descriptor tables for the frontier snapshot (layer->spec)
     * and restore (spec->layer) copy sets: one kernel launch instead of ~126
     * cudaMemcpy calls per direction. Built lazily on first snapshot; NULL
     * handle falls back to the per-tensor copy loop. */
    void *spec_snap_copies;      ///< baked batched-copy handle, layer -> spec direction
    void *spec_restore_copies;   ///< baked batched-copy handle, spec -> layer direction
    uint32_t spec_frontier_copy_n;          ///< tensors in each copy set
    uint64_t spec_frontier_copy_max_bytes;  ///< largest single copy, for staging
    int spec_frontier_copy_init;            ///< 0 unbuilt (a failed prepare leaves it 0 and the next snapshot retries), 1 built; see spec_frontier_copy_tables_init
    /** Shared multi-row logits slab (16 rows x n_vocab f32), written by every
     * batched multi-row output head: the DSpark draft/verify passes,
     * gpu_graph_verify_suffix, and the Tier-2 batched multi-session
     * decode driver.  Despite the "spec_" name it is NOT speculation-owned —
     * gpu_graph_alloc_raw_cap allocates it unconditionally so the batched
     * paths work with speculation disabled. */
    pulsar_gpu_tensor *spec_logits;
    /** STAGE 1b: the scalar frontier counters are GONE. They were a second copy
     * of ms_n_comp[cur_bank][il] kept in sync by hand, and L133 was the bill.
     * Use gpu_graph_n_comp(). */
    uint32_t raw_cap;
    /** Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /** Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[PULSAR_MAX_LAYER];
    uint32_t attn_comp_stage_cap;  ///< rows the attention compressor staging buffer can hold

    /** Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    pulsar_gpu_tensor *attn_comp_stage;  ///< staging the attention compressor writes through
    /** f32 staging used only when PULSAR_IDX_FP4 is on: the compressor emits new
     * indexer rows here (comp-cap rows, same row indices as the cache), and
     * the QAT+pack step stores them MXKV-FP4-packed into the persistent
     * layer_index_comp_cache.  Also reused for session-save dequant and
     * session-load repack. */
    pulsar_gpu_tensor *idx_comp_stage;
    pulsar_gpu_tensor *indexer_scores;   ///< indexer relevance score per compressed row (one <= slice-token span)
    /** CSA2 (L218): the top-k selection is SHARED down the layer sweep -- an
     * index source writes rows [t] for every batch row t of the step (absolute
     * batch row, not span-relative), and every REUSE layer up to the next
     * index source attends with them unchanged.  [prefill_cap][top_k] u32. */
    pulsar_gpu_tensor *comp_selected;
    /** CSA2 (L218): the candidate pool the candidate source (layer 20) publishes
     * per batch row and every later index source scores inside: one bit per
     * block of candidate_block_size compressed positions, cand_mask_words u32
     * per row, [prefill_cap] rows.  cand_bscore is the block-max scratch of one
     * span ([slice][n_blocks] f32). */
    pulsar_gpu_tensor *cand_mask;
    /// Block-max scratch of one span, [slice][n_blocks] f32.
    pulsar_gpu_tensor *cand_bscore;
    /// u32 words per row of cand_mask (one bit per candidate block).
    uint32_t           cand_mask_words;
    pulsar_gpu_tensor *ffn_norm;         ///< RMSNorm output feeding the FFN
    pulsar_gpu_tensor *output_embd;      ///< collapsed embedding-width vector
    pulsar_gpu_tensor *output_norm;      ///< final RMSNorm before the vocab projection

    /** DSpark target hidden capture buffers */
    pulsar_gpu_tensor *dspark_target_h[3];  ///< target hiddens captured at the three anchor layers
    pulsar_gpu_tensor *dspark_main_x;       ///< target-model input the drafter conditions on
    uint32_t dspark_target_layer_ids[3];    ///< which target layers the anchors are taken from
    /** Bulk prefill anchor-hidden capture for drafter retraining
     * (PULSAR_DSPARK_PREFILL_DUMP): per-chunk [prefill_cap, N_EMBD] buffers, one
     * per anchor layer. dspark_bulk_n is armed to the chunk's token count by
     * the prefill path and cleared by the drain; 0 everywhere else. */
    pulsar_gpu_tensor *dspark_bulk_h[3];  ///< per-chunk anchor hiddens, one buffer per anchor layer
    uint32_t dspark_bulk_n;               ///< tokens armed for capture this chunk; 0 = off
    uint32_t dspark_bulk_row0;            ///< L260: first batch row the bulk capture reads (a fused step's prefill rows follow its decode rows)
    /* plan-92 P0 teacher dump (PULSAR_DISTILL_DUMP, env-once at graph alloc):
     * per-position teacher top-64 ids/logits + tail logsumexp for the chunk,
     * filled by the all-rows head sweep after the prefill layer loop and
     * drained alongside dspark_bulk_h. NULL when the mode is off. */
    /// Teacher top-64 ids per position, [prefill_cap, 64] i32; NULL when the dump is off.
    pulsar_gpu_tensor *distill_top_ids;    /* [prefill_cap, 64] i32 */
    /// Teacher top-64 logits per position, [prefill_cap, 64] f16 bits.
    pulsar_gpu_tensor *distill_top_vals;   /* [prefill_cap, 64] f16 bits */
    /// Tail logsumexp per position, [prefill_cap] f16 bits.
    pulsar_gpu_tensor *distill_tail_lse;   /* [prefill_cap] f16 bits */
    /// [1] i32: count of top-64 verify misses.
    pulsar_gpu_tensor *distill_inexact;    /* [1] i32: top-64 verify misses */
    /** Prompt-window capture for drafter seeding: the anchor hiddens of the
     * last <=128 prompt positions, kept as a position%128 ring so the fused
     * loop can seed the drafter's context window at generation start (the
     * reference prefills this window; an empty or stale window collapses
     * drafter acceptance). dspark_prompt_n counts captured prompt positions. */
    pulsar_gpu_tensor *dspark_prompt_h[3];  ///< prompt-window anchor hiddens, as a position%128 ring
    uint32_t dspark_prompt_n;  ///< positions captured: ring valid for [lo, n)
    uint32_t dspark_prompt_lo;  ///< oldest position the ring still holds
    /** Fused spec loop (P2): per-position anchor hiddens captured during the
     * verify batch — [spec cap, N_EMBD] per anchor layer. dspark_capture_batch_n
     * != 0 arms the capture in gpu_graph_encode_layer_batch for that many
     * positions; 0 = off (prefill and plain decode unaffected). */
    pulsar_gpu_tensor *dspark_target_h_batch[3];  ///< per-position anchor hiddens from the verify batch
    uint32_t dspark_capture_batch_n;              ///< positions to capture; 0 = off
    /** Fused spec loop Stage B (no-replay rollback): per-position compressor
     * projections saved during the verify batch, so a partial accept can roll
     * the recurrent pool state forward from the frontier snapshot WITHOUT
     * replaying the transformer (the pool update kernels re-run from these
     * exact rows -> bit-identical state). [17 rows x width] per ratio>1 kv
     * source; the index-K rows derive from the emitted latent, so these saves
     * cover them too. spec_comp_save_n arms the save (0 = off). */
    pulsar_gpu_tensor *spec_comp_kv_save[PULSAR_MAX_LAYER];   ///< saved compressor KV projections a rejected draft must not keep
    pulsar_gpu_tensor *spec_comp_sc_save[PULSAR_MAX_LAYER];   ///< saved compressor score projections
    /** V4 ONLY: the INDEXER compressor's own projections, saved beside the
     * attention compressor's.  Its lane is a second recurrent state (see
     * layer_index_state_kv) and a rejected draft has to roll it back too --
     * for V4.1 there is no such lane, so these stay NULL.  Same row indexing
     * and the same `spec_comp_save_n` arming as the pair above; the ape fold
     * the reference does before the store happens in the rollforward, exactly
     * as it does on the live path. */
    pulsar_gpu_tensor *spec_icomp_kv_save[PULSAR_MAX_LAYER];  ///< indexer-compressor KV projections (V4)
    pulsar_gpu_tensor *spec_icomp_sc_save[PULSAR_MAX_LAYER];  ///< indexer-compressor score projections (V4)
    pulsar_gpu_tensor *spec_comp_scratch_row;   ///< emit sink during roll-forward: absorbs writes that must not land in the real cache
    uint32_t spec_comp_save_n;                  ///< arms the save; 0 = off
    /** Persistent drafter scratch (was per-call cudaMalloc/cudaFree churn --
     * cudaFree device-syncs, and the fused loop projects/seeds up to 5x/step). */
    pulsar_gpu_tensor *dspark_concat;  ///< [3*N_EMBD] target_h concat
    pulsar_gpu_tensor *dspark_proj_out;  ///< [N_EMBD] pre-norm projection
    /** Confidence scoring scratch, persistent for the same reason as the two
     * above: it is touched once per fused spec step and n_draft is clamped to
     * 16, so a per-step alloc/free pair bought nothing but device
     * serialization. */
    pulsar_gpu_tensor *dspark_conf_scores;  ///< [16] f32 per-draft confidence
    pulsar_gpu_tensor *dspark_conf_tokens;  ///< [16] i32 refined draft ids
    pulsar_gpu_tensor *dspark_embed_tokens;  ///< [16] i32 draft ids for the embed upload (L104 fix B: was a cudaMalloc/free PER DRAFTER FORWARD in gpu_decode)
    pulsar_gpu_tensor *dspark_refined_ids;  ///< [17] i32: L108 P1 device-chained greedy walk -- [0] seeded with the base token, reduce pos p writes the winner to [p+1]
    pulsar_gpu_tensor *dspark_bank_meta;      ///< L150: [2 x PULSAR_DSPARK_BANKS_MAX] i32: per-bank base row, per-bank sampled prev token
    pulsar_gpu_tensor *dspark_row_meta;       ///< L150: [7 x PULSAR_SPEC_LOGITS_ROWS] i32 per-row rope/visibility positions per layer + bank id for the banked drafter forward
    pulsar_gpu_tensor *dspark_seed_kv;  ///< [HEAD_DIM] seed kv scratch
    pulsar_gpu_tensor *dspark_seed_norm;  ///< [HEAD_DIM]
    pulsar_gpu_tensor *dspark_seed_rot;  ///< [HEAD_DIM]
    pulsar_gpu_tensor *dspark_markov_logits;  ///< [N_VOCAB] markov refine scratch

    /** DSpark draft KV raw caches (one per draft layer, window=128) */
    pulsar_gpu_tensor *dspark_raw_cache[3];  ///< the drafter's raw KV ring, one per draft layer
    uint32_t dspark_n_raw[3];                ///< positions held in each ring

    uint32_t prefill_cap;  ///< maximum rows one prefill chunk may carry; sizes the batch tensors
    uint32_t raw_window;   ///< positions the raw (uncompressed) KV ring retains per layer

    /** Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    pulsar_gpu_tensor *prefill_tokens;
    /** L216 image-span visibility for the CURRENT prefill chunk: `prefill_cap`
     * int32 left counts followed by `prefill_cap` int32 right counts (the
     * reference's get_image_visible).  Uploaded once per chunk by
     * gpu_graph_upload_vision_visible; vision_visible_tokens is 0 unless THIS
     * chunk carries sentinel ids, which is what keeps every text prefill on the
     * NULL path it took before. */
    pulsar_gpu_tensor *vision_visible;
    /// Sentinel (image) tokens in the current chunk; 0 = text-only chunk, vision_visible unused.
    uint32_t           vision_visible_tokens;
    pulsar_gpu_tensor *batch_cur_hc;                ///< batched twin: HC residual carrier
    pulsar_gpu_tensor *batch_next_hc;               ///< batched twin: HC residual for the next layer (swapped with batch_cur_hc each layer)
    pulsar_gpu_tensor *batch_flat_hc;               ///< batched twin: HC streams flattened for the mix GEMV
    pulsar_gpu_tensor *batch_hc_mix;                ///< batched twin: HC mix projection output
    pulsar_gpu_tensor *batch_hc_split;              ///< batched twin: per-stream split of the mix
    /** Single-pass mHC (L218): the `pre` collapse weights handed from one
     * sublayer to the next, [prefill_cap][n_hc] f32 -- born one-hot with the
     * residual stream (embedding expansion), rewritten by every sublayer's
     * split with its own pre, read by the output head after the last FFN.
     * Row-indexed like batch_cur_hc: whatever moves an hc row moves its pre. */
    pulsar_gpu_tensor *batch_hc_pre;
    /** 0731's HC head-mix scratch, one row's worth: the norm+mix projection and
     * the sigmoid'd coefficients the shared collapse consumes.  V4.1 allocates
     * them and never reads them -- its collapse reads batch_hc_pre.
     * plans/96-two-profiles-one-engine.md s17. */
    pulsar_gpu_tensor *output_pre;                  ///< [n_hc] norm+mix output
    pulsar_gpu_tensor *output_weights;              ///< [n_hc] sigmoid'd coefficients
    pulsar_gpu_tensor *batch_attn_cur;              ///< batched twin: attention sublayer input
    pulsar_gpu_tensor *batch_attn_norm;             ///< batched twin: RMSNorm output feeding the projections
    pulsar_gpu_tensor *batch_qr;                    ///< batched twin: low-rank query latent
    pulsar_gpu_tensor *batch_qr_norm;               ///< batched twin: normalised query latent
    pulsar_gpu_tensor *batch_q;                     ///< batched twin: queries in head space
    /** L037 lever 3: when q_prep_active, batch_q holds RAW head projections
     * for the current layer and every attention call this chunk passes
     * &q_prep so the f16 kernel fuses norm+rope into its Q load -- the only
     * attention consumer since L166; there is no standalone q_prep kernel.
     * Set per layer at the Q-path norm decision in gpu_prefill. */
    pulsar_gpu_q_prep q_prep;
    int q_prep_active;  ///< the fused norm+rope Q path is armed for this layer
    /** Set by the imatrix collector around its prefill: it reads the f32
     * ffn_norm rows on the host, so the norm keeps them; every other consumer
     * reads the producer's E4M3 and the rows are not stored. */
    int imatrix_f32_rows;
    pulsar_gpu_tensor *batch_kv_raw;                ///< batched twin: fused KV projection output, pre-norm
    pulsar_gpu_tensor *batch_kv;                    ///< batched twin: KV latent after its RMSNorm
    /** The chunk's KV in WINDOW rows -- what attention actually reads.
     * batch_kv above stays f32 because norm/rope/fp8-quantize are in-place
     * elementwise passes over it, which is f32-as-scratch and is what torch does
     * too (compute wide, store narrow). What was wrong until 2026-08-17 was f32
     * as the multiply OPERAND: attention read the staging buffer directly, so
     * the chunk's own KV was attended at 4 bytes/element while every later
     * chunk read the same rows out of the packed ring at 384 B/row. */
    pulsar_gpu_tensor *batch_kv_pack;
    pulsar_gpu_tensor *batch_comp_kv;               ///< batched twin: compressed KV rows produced this chunk
    pulsar_gpu_tensor *batch_comp_sc;               ///< batched twin: compressed score rows produced this chunk
    /** V4's indexer-own compressor runs a SECOND compression over the same rows
     * at the indexer's head dim (`indexer_compressor_kv/gate`), so it needs its
     * own pair of projection rows.  V4.1 projects its index key from the kv
     * source's latent and never fills these; they are still allocated, like
     * batch_comp_kv, because a V4.1 graph must not be shaped differently from a
     * V4 one.  Width authority: pulsar_comp_row_width(ratio,
     * PULSAR_N_INDEXER_HEAD_DIM). */
    pulsar_gpu_tensor *batch_index_comp_kv;         ///< batched twin: V4 indexer-compressor KV rows
    pulsar_gpu_tensor *batch_index_comp_sc;         ///< batched twin: V4 indexer-compressor score rows
    /** Scratch for the ratio-4 compressor state rebuild's tail re-projection
     * (<= 8 rows x comp width, both halves).  Its own buffer because the
     * rebuild used to write into batch_comp_kv/_sc rows 0..n_tail-1 while the
     * rewind projection-ring deposit still had to read that chunk's rows
     * n_tokens-8..n_tokens-1 from the same buffer -- for chunks of 5..11
     * tokens the ring received tail tokens filed under head positions (L171). */
    pulsar_gpu_tensor *batch_indexer_q;  ///< f32 rope staging, producer-internal (L090.4)
    pulsar_gpu_tensor *batch_indexer_qp;  ///< packed E2M1 Q rows -- what the scorers read
    pulsar_gpu_tensor *batch_indexer_weights;  ///< batched twin: per-head indexer mixing weights
    pulsar_gpu_tensor *batch_heads;                 ///< batched twin: per-head attention output
    pulsar_gpu_tensor *batch_attn_low;              ///< batched twin: attention output through the low-rank 'a' projection
    pulsar_gpu_tensor *batch_attn_out;              ///< batched twin: attention output at embedding width
    pulsar_gpu_tensor *batch_after_attn_hc;         ///< batched twin: HC residual after the attention sublayer
    pulsar_gpu_tensor *batch_ffn_cur;               ///< batched twin: FFN sublayer input
    pulsar_gpu_tensor *batch_ffn_norm;              ///< batched twin: RMSNorm output feeding the FFN
    pulsar_gpu_tensor *batch_shared_gate;           ///< batched twin: shared expert, gate branch
    pulsar_gpu_tensor *batch_shared_up;             ///< batched twin: shared expert, up branch
    pulsar_gpu_tensor *batch_shared_mid;            ///< batched twin: shared expert, SwiGLU product
    pulsar_gpu_tensor *batch_shared_out;            ///< batched twin: shared expert, down output
    pulsar_gpu_tensor *batch_router_logits;         ///< batched twin: per-expert routing logits
    pulsar_gpu_tensor *batch_router_probs;          ///< batched twin: routing probabilities
    pulsar_gpu_tensor *batch_router_selected;       ///< batched twin: chosen expert ids
    pulsar_gpu_tensor *batch_router_weights;        ///< batched twin: per-expert mixing weights
    pulsar_gpu_tensor *batch_routed_up;             ///< batched twin: routed experts, up branch
    pulsar_gpu_tensor *batch_routed_mid;            ///< batched twin: routed experts, SwiGLU product
    pulsar_gpu_tensor *batch_routed_down;           ///< batched twin: routed experts, per-expert down output
    pulsar_gpu_tensor *batch_routed_out;            ///< batched twin: routed experts pooled by mixing weight
    pulsar_gpu_tensor *batch_ffn_out;               ///< batched twin: shared + routed FFN output
    pulsar_gpu_tensor *directional_steering_dirs;  ///< steering direction vectors; NULL when steering is off
    float directional_steering_attn_scale;         ///< strength applied at the attention sublayer
    float directional_steering_ffn_scale;          ///< strength applied at the FFN sublayer

    /** Tier-2 bank pool (see pulsar_bank_slabs above).  banks.n_banks == 0 keeps
     * the classic single-session layout; >= 2 makes the per-layer cache
     * pointers bank views into the slabs. */
    pulsar_bank_slabs banks;


    /** Tier-2 banked multiseq step state (increment 2 — per-bank compressor
     * frontiers).  The authoritative per-bank compressed-row counters are
     * ms_n_comp (indexed by TRUE bank id, never a packed
     * row ordinal); they are HOST bookkeeping owned by the multiseq driver,
     * and gpu_graph_bank_repoint swaps device views only.
     *
     * ⚠ STAGE 1b (1a0bd1a) DELETED THE SCALAR TWINS. There is no
     * layer_n_comp any more: gpu_graph_n_comp() resolves to
     * ms_n_comp[cur_bank][il], and a
     * session with no pool is simply bank 0, so there is no "classic case"
     * left to special-case or to hand off at a boundary. That is the fix for
     * the class that produced L133, where a correctness fix landed on one of
     * two copies and the second could not be seen; the divergence is now
     * unrepresentable rather than merely repaired.
     *
     * The two ALIGNED-chunk writes that used to publish a CROSS-BANK SUPERSET
     * are consequently guarded !mseq — with the scalar gone such a write would
     * land on cur_bank's real row and clobber it — and the banked arms publish
     * per bank instead.  The batched emit loop writes each emitted row
     * into seq_id[t]'s bank at that bank's frontier and bumps ONLY that
     * bank's ms counter; per-row raw-ring state needs no bookkeeping at all
     * (the ring is position-indexed: slot = pos % raw_cap per bank).
     *
     * ms_positions/ms_seq_id are the host mirrors the emit loop reads;
     * batch_positions/batch_seq_id the device arrays the kernels read.
     * All four are lazily allocated (prefill_cap entries) on the first
     * multiseq step; NULL in production single-session serving. */
    uint32_t ms_n_comp[PULSAR_MSEQ_MAX][PULSAR_MAX_LAYER];        ///< compressed rows per (bank, kv source): the authoritative frontier for the KV pool AND the index-K pool (one emit writes both)
    /** High-water of ms_n_comp per (bank, kv source) since the bank's comp/index
     * physical was last allocated: the rows whose demand-paged pages are
     * RESIDENT.  A rewind, an invalidate, an eviction to an empty conversation
     * or a speculative rollback lowers the frontier but frees no page, so the
     * frontier undercounts memory; this does not.  Raised only by
     * gpu_graph_set_n_comp, cleared only by gpu_graph_bank_free_physical (the
     * one path that returns the pages). */
    uint32_t ms_comp_hw[PULSAR_MSEQ_MAX][PULSAR_MAX_LAYER];
    /** The step's cross-bank compressed-row superset per layer, max over the
     * step's rows of (pos + 1) / ratio: computed ONCE in step_begin (where it
     * is checked against layer_comp_cap) and read by the layer encode as the
     * comp operand bound of every attention/indexer launch.  It used to be
     * re-derived from ms_positions at the encode with no cap check (L178). */
    uint32_t batch_comp_sup[PULSAR_MAX_LAYER];
    /** Tier-2 Option F: per-bank DSpark drafter-ring frontier counters (the
     * device rings themselves are banked slabs, pulsar_bank_slabs.dspark_*).
     * Captured/installed alongside ms_n_comp so each bank keeps a WARM drafter
     * window under N=2 spec-time-slice — the whole point of Option F. */
    uint32_t ms_dspark_n_raw[PULSAR_MSEQ_MAX][3];   ///< per-bank raw-ring fill for each of the drafter's 3 rings
    /** L264: the bank's recurrent compressor lanes and raw window do not
     * describe its frontier -- a rewind landed somewhere other than 0 or a grid
     * checkpoint.  Set only by pulsar_session::rewind; cleared only where the
     * state at the frontier is re-established whole (a checkpoint restore, a
     * reset to position 0).  Every compressor store on a stale bank refuses
     * (gpu_graph_csa2_produce), eval refuses up front, and the server keeps a
     * stale bank out of fused rounds until a sync restores it. */
    bool ms_comp_state_stale[PULSAR_MSEQ_MAX];
    uint32_t ms_dspark_prompt_n[PULSAR_MSEQ_MAX];   ///< drafter prompt-window length held by the bank
    uint32_t ms_dspark_prompt_lo[PULSAR_MSEQ_MAX];  ///< first position of that window

    /** L264 GRID CHECKPOINTS (kv_state.h, checkpoint.cpp; what a slot holds:
     * kv_state_ds4.cpp).  A checkpoint at G is valid while the bank's compressed
     * frontier reaches G: gpu_graph_set_n_comp lowers it with the frontier, and
     * replacing a bank's rows wholesale drops all of its checkpoints. */
    pulsar_ckpt_store ckpt;

    int32_t *ms_positions;                ///< HOST mirror: KV position of each row in the step
    int32_t *ms_seq_id;                   ///< HOST mirror: owning bank of each row in the step
    pulsar_gpu_tensor *batch_positions;   ///< DEVICE copy of ms_positions, read by the kernels
    pulsar_gpu_tensor *batch_seq_id;      ///< DEVICE copy of ms_seq_id, read by the kernels
    /** A multiseq step is in flight. Armed by gpu_graph_multiseq_step_begin for
     * EVERY batched step -- one row or sixteen -- which is what makes the
     * batched lane the only decode lane. While armed, the aligned-chunk
     * frontier publishes are guarded off (see the multiseq block above): with
     * the scalar twins deleted in stage 1b there is no superset slot to write,
     * and the banked arms publish per bank instead. */
    bool batch_multiseq;
    uint32_t batch_multiseq_rows;         ///< rows in the current step
    /** L260: the first PREFILL row of the current step -- rows below it are the
     *  decode/verify runs, rows from it on are prompt runs (step_begin's declared
     *  or inferred split; == batch_multiseq_rows when the step has none).  The
     *  compressor takes its batched arm per prompt run from here. */
    uint32_t batch_multiseq_pf_row0;
    /** Borrowed for ONE prefill: the images to merge and the tower to encode them
     * with, or NULL for the text-only path.  Set and cleared by the prefill's
     * owner (the family sync's prefill, ds4_sync_prefill); nothing else may leave it set. */
    const pulsar_vision_request *vision_req;
    /** Borrowed TP transport for the owning session's engine (slice 4b), set
     * from the engine at graph init; NULL when the pair is not armed.  The
     * prefill big-gate call sites read this rather than threading the engine
     * through every prefill entry point. */
    struct pulsar_tp *tp;
    /** Monotonic prefill big-gate exchange counter (slice 4b), incremented by
     * gpu_graph_tp_allreduce_rows on its big-gate path: once per layer per prefill chunk
     * for the attention output and once for the FFN (4g-2).  Every rank advances it in the same
     * order from the same starting value, so the exchange seq stays in
     * lockstep (the transport uses it as a desync guard). */
    uint64_t tp_prefill_seq;
    /** The registered slab's device mapping, borrowed from the engine at graph
     * init beside `tp` -- what the row-lane kernels (4g-2) address. */
    void *tp_slab_dev;
    /** The bulk lane's buffer, device mapping (v14), borrowed the same way. */
    void *tp_bulk_dev;
    /** The vocab gather's own-slice scratch (4g-2): this rank's packed head
     * slice, PULSAR_SPEC_LOGITS_ROWS rows at the widest range, rounded up to
     * whole row-lane messages so the last chunk's stage reads inside it.
     * Allocated at graph init on a row-lane pair; NULL otherwise. */
    pulsar_gpu_tensor *tp_vocab_own;
    /** The attention-input gather's packed own payload (4g-3): every row-split
     * projection's block for one decode/verify step, PULSAR_TP_BATCH_MAX_ROWS
     * row-lane messages.  Allocated beside tp_vocab_own. */
    pulsar_gpu_tensor *tp_ain_own;
    /** The row lane's stage+publish ticket (one zeroed device u32): the
     * stage kernel's blocks count themselves in on it and the last one
     * publishes the descriptor and re-zeroes it.  Per graph, so two graphs
     * on two streams never share one.  Allocated beside tp_vocab_own. */
    pulsar_gpu_tensor *tp_stage_ticket;
    /** The key the engine registered this rank's K-half weights under (4g-2
     * row-parallel splits: the shared expert's down projection), borrowed at
     * graph init; resolved as (key, parent tensor's abs_offset). */
    const void *tp_kslice_key;
    /* (key offset: pulsar_tp_kslice_key_offset below) */
    /** Monotonic vocab all-gather counter (slice 4d), incremented once per eval
     * by tp_vocab_split's host lane (a row-lane pair gathers on the stream and
     * is sequenced by the lane's own message counter instead).  Every rank advances it
     * the same number of times in the same order, so the gather's seq stays in
     * lockstep -- the transport's desync guard keys on it, which is also what
     * catches a lane that ran a different number of heads on the two ranks. */
    uint64_t tp_vocab_seq;
    /** Slice 4g: the engine's owned output-group range, borrowed at graph init
     * beside `tp` (see pulsar_engine::tp_group_lo).  The attention block runs
     * its heads, its grouped 'a' projection and its `low` gather on it. */
    uint32_t tp_group_lo, tp_group_hi;
    /** @var pulsar_gpu_graph::tp_group_hi
     *  One past the last owned output group (the range is [tp_group_lo, tp_group_hi)). */
    /** Engram (L242 slice 5, engram_forward.cpp): the step's staged hashes and gathers,
     * the device scratch the Engram layer runs in, and the token history the
     * hashes read.  Allocated only when the weights carry Engram layers.
     *
     * The history is a per-bank RING of the last PULSAR_ENGRAM_RING positions'
     * token ids, each tagged with its position: every row a forward stages writes
     * its token at its position (a prefill also writes the three positions before
     * its chunk from the prompt, which is authoritative), and a row's n-gram reads
     * the three positions before it BY TAG -- a tag that does not match (a bank
     * restored past the ring, a first row that is not where its bank's history
     * ends) refuses the step rather than hash a stale token.  Rejected drafts leave
     * their tokens at positions the next committed row overwrites before any row
     * reads them, so a rewind needs no undo. */
    struct {
        const pulsar_engram_model *model;    ///< borrowed from the weights; NULL = no Engram
        pulsar_gpu_tensor *rows;             ///< [cap][24][264] the gathered records, uploaded per layer
        pulsar_gpu_tensor *x_key;            ///< f32 [cap][24*256]: the MX slot's key (never written)
        pulsar_gpu_tensor *kv;               ///< f32 [cap][(n_hc+1)*dim]: the `wkv` output
        unsigned char *host_rows[PULSAR_ENGRAM_MAX_LAYERS];   ///< [cap][24][264] gather targets
        uint64_t *ids[PULSAR_ENGRAM_MAX_LAYERS];              ///< [cap][24] the staged row ids
        pulsar_engram_gather *pending[PULSAR_ENGRAM_MAX_LAYERS];
        uint32_t pending_n;                  ///< rows of the staged step (0 = none)
        uint32_t cap, n_banks;
        int32_t *ring_tok;                   ///< [n_banks][PULSAR_ENGRAM_RING]
        uint32_t *ring_pos;                  ///< [n_banks][PULSAR_ENGRAM_RING], UINT32_MAX = empty
    } engram;
} pulsar_gpu_graph;

/** Positions of token history each bank's Engram ring keeps (a power of two
 * past the deepest rewind a staged step reads across: a verify block). */
#define PULSAR_ENGRAM_RING 64u
/** Stage the Engram hashes of a step's rows and start their gathers: row r is
 * token tok[r] at position pos[r] of bank bank[r] (NULL bank = the graph's
 * current bank for every row).  `hist` (may be NULL) is the sequence the rows
 * continue, authoritative for the positions before pos[0].  No Engram -> a
 * no-op.  false = refused, reason printed (a token outside the layout, history
 * the ring does not hold). */
/// @param g         the graph
/// @param tok       n token ids
/// @param pos       n positions (tokens)
/// @param bank      n bank ids, or NULL = the current bank for every row
/// @param n         the step's rows (<= the Engram scratch's cap)
/// @param hist      the sequence the rows continue (may be NULL)
/// @param hist_len  entries in `hist`
/// @return true (also with no Engram), false = refused, reason printed.
bool gpu_graph_engram_stage(pulsar_gpu_graph *g, const int *tok, const int32_t *pos, const int32_t *bank,
                            uint32_t n, const int *hist, uint32_t hist_len);
/** The Engram layer's op at the top of block `layer` (before the drafter anchor
 * capture, as the reference): wait its gather, rows -> MX slot -> `wkv` -> gate
 * + gated add on g->batch_cur_hc.  A block with no Engram is a no-op. */
/// @param g         the graph, its gathers staged for this step
/// @param model     the model
/// @param layer     the block's weights (engram_wkv NULL: a no-op)
/// @param il        the block index
/// @param n_tokens  the step's rows; must match the staged step
/// @return false (said) when no gather was staged for this step or the op failed.
bool gpu_graph_engram_apply(pulsar_gpu_graph *g, const pulsar_model *model, const pulsar_layer_weights *layer,
                            uint32_t il, uint32_t n_tokens);
/// Allocate the graph's Engram scratch and history rings; no Engram in `w`: nothing to allocate.
/// @param g        the graph
/// @param w        the weights (w->engram NULL: a no-op)
/// @param cap      most rows one step stages
/// @param n_banks  banks with a history ring (0 is taken as 1)
/// @return false when a device allocation failed.
bool gpu_graph_engram_alloc(pulsar_gpu_graph *g, const pulsar_weights *w, uint32_t cap, uint32_t n_banks);
/** Empty `bank`'s Engram history ring (a restore or a reset replaced its state). */
/// @param g     the graph
/// @param bank  the bank (out of range: a no-op)
void gpu_graph_engram_forget(pulsar_gpu_graph *g, uint32_t bank);
/// Drain any in-flight gathers and free the graph's Engram scratch.
/// @param g  the graph
void gpu_graph_engram_release(pulsar_gpu_graph *g);

/* ONE-STATE-MODEL stage 1a — the compressor frontier has ONE accessor.
 *
 * The frontier is currently stored twice: the scalars below and the per-bank
 * ms_n_comp[bank][il]. Nothing enforces that they agree, and L133 was the bill
 * for that -- L120's fix clamped the scalars while the served path validates
 * the per-bank copy, so a production bug closed on 08-27 still reproduced on
 * 08-30. plans/ONE-STATE-MODEL.md is the collapse.
 *
 * This step changes NO behaviour: the bodies still return the scalars. Its
 * whole purpose is that stage 1b then flips the representation in ONE place
 * rather than at 71 call sites, on state that decides which KV rows attention
 * reads.
 *
 * References, not get/set pairs: call sites use ++, =, and comparisons, and
 * rewriting each of those by hand is exactly the kind of mechanical edit that
 * introduces the bug this refactor exists to prevent. */
/** STAGE 1b: the per-bank slots are now the ONLY storage. The scalars are gone.
 * A session with no pool allocated is simply bank 0 -- ms_n_comp is a
 * fixed-size struct member, always present, and gpu_graph_bank_raw_pool()
 * already falls back to the classic tensors for bank 0, so there is no
 * "classic" case left to special-case. */
/// @param g  the graph
/// @return the bank the installed device views address (0 without a pool).
static inline uint32_t gpu_graph_cur_bank(const pulsar_gpu_graph *g) {
    return g->banks.n_banks ? g->banks.cur_bank : 0u;
}
/* THE BANK IS EXPLICIT.  These used to take (g, il) and resolve through
 * gpu_graph_cur_bank(), which reads `banks.cur_bank` -- documented as "bank the
 * installed views currently address".  That is a DEVICE VIEW BINDING, set by
 * gpu_graph_bank_repoint() and by nothing else.  It is not a session identity.
 *
 * The two coincide for classic single-session work, and diverge exactly where
 * it matters: during a batched step spanning several banks, the views are bound
 * to ONE of them, so an unqualified "what is the frontier?" resolves to
 * whichever bank was repointed last -- an artifact of setup order, not a
 * property of the step.  L139 is that divergence: a co-scheduled step left
 * cur_bank on the prefill bank (2) where a decode-only step left it on the last
 * decode bank (1), and the same read returned a different bank's row.
 *
 * So the caller names the bank.  Classic paths pass gpu_graph_cur_bank(g) --
 * still correct there, and now visibly a CHOICE rather than a default.  Batched
 * paths pass the row's seq_id.  A site that cannot name a bank is a site that
 * did not know whose frontier it was reading. */
/// A bank's compressed frontier at layer `il`: its compressed rows.
/// @param g     the graph
/// @param bank  the bank, named by the caller
/// @param il    the layer
/// @return the row count, ms_n_comp[bank][il].
static inline uint32_t gpu_graph_n_comp(const pulsar_gpu_graph *g, uint32_t bank, uint32_t il) {
    return g->ms_n_comp[bank][il];
}
/** The one writer of a bank's compressed frontier: sets ms_n_comp and raises the
 * bank's resident high-water (ms_comp_hw) with it, so no write can move the
 * frontier past rows the accounting has not counted. */
/// @param g     the graph
/// @param bank  the bank
/// @param il    the layer
/// @param rows  the new compressed row count
static inline void gpu_graph_set_n_comp(pulsar_gpu_graph *g, uint32_t bank, uint32_t il, uint32_t rows) {
    /* L264: rows below a checkpoint's frontier may be rewritten once the counter
     * drops under them, so the checkpoint goes with them -- here, where every
     * rewind, cut and invalidate already passes.  The intact rows [0, rows) serve
     * every position up to rows * R + R - 1 (a position's open group is its lanes'):
     * a grid checkpoint (a multiple of R) is dropped exactly as at rows * R, and a
     * prompt end inside the group (L289) survives a round's rollback to it. */
    if (rows < g->ms_n_comp[bank][il]) {
        const uint32_t R = pulsar_layer_compress_ratio(il);
        pulsar_ckpt_frontier_lowered(&g->ckpt, bank, rows * R + R - 1u);
    }
    g->ms_n_comp[bank][il] = rows;
    if (rows > g->ms_comp_hw[bank][il]) g->ms_comp_hw[bank][il] = rows;
}

/** =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During GPU prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */
typedef struct {
    float *gate_up_sum2;   ///< running sum of SQUARED activations per [layer][expert][hidden], gate/up inputs
    float *down_sum2;      ///< running sum of squared activations per [layer][expert][ffn], down inputs
    uint32_t gate_up_count[PULSAR_MAX_LAYER][PULSAR_MAX_EXPERT];  ///< rows accumulated into gate_up_sum2, the divisor for the mean
    uint32_t down_count[PULSAR_MAX_LAYER][PULSAR_MAX_EXPERT];     ///< rows accumulated into down_sum2
    float *ffn_norm_buf;   ///< host copy of batch_ffn_norm for the chunk being observed
    float *routed_mid_buf; ///< host copy of batch_routed_mid (post route-weighting)
    int   *selected_buf;   ///< host copy of batch_router_selected: which expert each row went to
    float *sq_tmp;         ///< scratch for the per-row squaring pass
    uint32_t cap_tokens;   ///< rows the host buffers can hold, i.e. the prefill chunk width
    uint64_t observed_routes; ///< (token, expert) routing decisions accumulated
    uint32_t chunks;          ///< prefill chunks processed
    const char *dataset_path; ///< calibration corpus being read
} pulsar_imatrix_collector;

/// C-style name for struct pulsar_vocab.
typedef struct pulsar_vocab pulsar_vocab;

/** =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

/** One slot in a ::str_i32_table. */
typedef struct {
    pulsar_str key;  ///< the key, borrowed from the mapping; valid only while `used`
    int value;       ///< the mapped value
    bool used;       ///< the slot is occupied (open addressing needs this, not a NULL key)
} str_i32_entry;

/** Open-addressed string to int32 map, used for the vocabulary lookup.
 *
 * Keys are borrowed ::pulsar_str slices into the model mapping rather than
 * copies, so building the table over a 100k-entry vocabulary costs no string
 * allocation at all. */
typedef struct {
    str_i32_entry *entry;  ///< the slot array
    uint64_t cap;          ///< slots allocated; always a power of two
    uint64_t used;         ///< slots occupied; the load factor numerator
} str_i32_table;

struct owned_str;  ///< forward decl: bpe_rank() param; full def appears later

/** Byte-level BPE vocabulary and the special ids the chat template needs.
 *
 * ⚠ `n_vocab` here is the TOKENIZER TABLE LENGTH and is not required to equal
 * pulsar_shape::n_vocab, which is the logits row width. Size logits buffers
 * with the shape's value (pulsar_engine_logits_width()), never this one. */
struct pulsar_vocab {
    pulsar_str *token;     ///< id -> token bytes, n_vocab entries
    int n_vocab;           ///< tokenizer table length (NOT the logits width)
    int bos_id;            ///< beginning-of-sequence token
    int eos_id;            ///< end-of-sequence token
    int user_id;           ///< chat role marker: user turn
    int assistant_id;      ///< chat role marker: assistant turn
    int system_id;         ///< chat role marker: system text (V4.1 leads a thinking conversation with it)
    int think_start_id;    ///< opens a reasoning span
    int think_end_id;      ///< closes a reasoning span
    int dsml_id;           ///< DSML tool-call marker
    int image_id;          ///< PULSAR_IMAGE_PLACEHOLDER's id, or -1 when the artifact has no such token
    str_i32_table token_to_id;  ///< token bytes -> id, for the BPE merge loop
    str_i32_table merge_rank;   ///< BPE merge priority; lower rank merges first

    /** ---- methods (C++ port): 1:1 mirror of the vocab verb family in
     * tokenizer.cpp; bodies keep the auto *vocab = this alias, logic verbatim.
     * Names kept as-is (none carry the pulsar_vocab type-name prefix). ---- */
    /** Merge priority of the pair (a, b); lower merges first. @return rank, or a
     * sentinel above every real rank when the pair is not in the merge table. */
    /// @param a  left symbol of the pair
    /// @param b  right symbol of the pair
    int bpe_rank(const owned_str *a, const owned_str *b) const;
    /** Emit one pre-tokenized piece as ids, running the BPE merge loop over it. */
    /// @param raw_piece  the piece's bytes (borrowed)
    /// @param out        ids are appended
    void bpe_emit_piece(pulsar_str raw_piece, token_vec *out) const;
    /** Tokenize plain text: pre-tokenize, then BPE-merge each piece. */
    /// @param text  NUL-terminated plain text; special-token strings are NOT recognised here
    /// @param out   ids are appended
    void bpe_tokenize_text(const char *text, token_vec *out) const;
    /** Exact-match lookup of a REQUIRED token's id.  A missing token is a fatal
     * load error (exits), so only callers whose token every served artifact
     * carries may use it. @return the id. */
    /// @param text  the token's exact bytes, NUL-terminated
    int vocab_lookup(const char *text) const;
    /** Exact-match lookup that returns -1 when the token is absent -- the
     * OPTIONAL counterpart of vocab_lookup, which exits on a missing token. */
    /// @param text  the token's exact bytes, NUL-terminated
    /// @return the id, or -1 when the vocabulary has no such token.
    int vocab_find(const char *text) const;
    /** Populate the vocabulary from the model's GGUF metadata. */
    /// @param model  the loaded model; token strings stay borrowed from its mapping
    void vocab_load(const pulsar_model *model);
    /** Release the vocabulary's owned tables. */
    void vocab_free();
    /** Longest special-token match at `p`. @return true and set token/len on a hit. */
    /// @param p      NUL-terminated text to match at
    /// @param token  out: the special token's id, on a hit
    /// @param len    out: bytes the match covers, on a hit
    bool special_token_at(const char *p, int *token, size_t *len) const;
    /** Tokenize `n` bytes at `p`, honouring special tokens found inside the span. */
    /// @param p    span start (need not be NUL-terminated)
    /// @param n    span length in bytes; 0 emits nothing
    /// @param out  ids are appended
    void tokenize_span(const char *p, size_t n, token_vec *out) const;
    /** Tokenize an ALREADY-RENDERED chat string -- the caller has applied the
     * template, so role markers appear as literal special tokens. */
    /// @param text  the rendered chat string; NULL is treated as ""
    /// @param out   ids are appended
    void tokenize_rendered_chat_vocab(const char *text, token_vec *out) const;
    /** L223: tokenize_rendered_chat_vocab(), but bytes inside any of `spans` are plain text: no special-token
     *  matching there, so client-typed role markers become ordinary tokens.
     *  @param text     the rendered chat string; NULL is treated as ""
     *  @param spans    half-open byte ranges [lo, hi) into `text`, ascending and disjoint; clamped to its end
     *  @param n_spans  entries in `spans`
     *  @param out      ids are appended */
    void tokenize_rendered_chat_spans_vocab(const char *text, const pulsar_text_span *spans,
                                          uint32_t n_spans, token_vec *out) const;
};

/** The loaded model and everything derived from it.
 *
 * One engine owns the weights; MANY sessions share it. Everything here is
 * immutable after open() except the cumulative metrics counters -- which is
 * what makes concurrent sessions safe against a single engine. */
/* L263 / L284: a decode lane's self-measured step cost (pulsar.h pulsar_lane_cost
 * is the read-only view), one structure for both lanes.  Exponentially weighted
 * least squares of a step's wall time on the rows its forward carried:
 * step_ms = flat + row * rows. */
typedef struct {
    /// EW sum of weights.
    double w, sx, sy, sxx, sxy;  ///< EW sums: weight, rows, ms, rows^2, rows*ms
    /** @var pulsar_lane_cost_fit::sx
     *  EW sum of rows. */
    /** @var pulsar_lane_cost_fit::sy
     *  EW sum of step ms. */
    /** @var pulsar_lane_cost_fit::sxx
     *  EW sum of rows^2. */
    uint32_t n;                  ///< steps observed
    /// The fit's flat term (per step), in microseconds.
    int32_t flat_us, row_us;     ///< the fit, in microseconds, when `valid`
    bool valid;                  ///< enough evidence (lane_cost_fit_observe says what)
    /// flat_us as last announced (0: never).
    int32_t ann_flat_us, ann_row_us;  ///< the last announced terms (0: never)
    uint32_t ann_n;                   ///< `n` at that announcement
} pulsar_lane_cost_fit;

/* L304 / L327: what a prompt chunk costs (verify_curve.cpp has the model): ONE exponentially weighted least-squares
 * fit over every prompt chunk the engine runs -- solo prefill chunks (the prefill loop's own) and fused ones (beside a
 * round's verify rows: the server's report less the verify curve's forward of those rows) -- on
 * x = [solo, fused, prompt rows, prompt attention pairs]: a fixed term per chunk kind, the rows and pairs shared. */
enum { PULSAR_CHUNK_SOLO = 0, PULSAR_CHUNK_FUSED = 1, PULSAR_CHUNK_KINDS = 2 };
/// Dimension of the prompt-chunk cost fit's feature vector x = [solo, fused, rows, pairs].
#define PULSAR_PREFILL_FIT_DIM 4
/** L304 / L327: the engine's prompt-chunk cost model (see the comment above PULSAR_CHUNK_SOLO); one per engine
 *  (pulsar_engine::prefill_cost), fitted and persisted by verify_curve.cpp. */
typedef struct {
    double m[PULSAR_PREFILL_FIT_DIM][PULSAR_PREFILL_FIT_DIM];   ///< EW sums of x x^T (pairs in M)
    double v[PULSAR_PREFILL_FIT_DIM];                           ///< EW sums of x y
    uint32_t n;                                                  ///< chunks observed
    uint32_t n_kind[PULSAR_CHUNK_KINDS];                         ///< ... of each kind
    bool valid[PULSAR_CHUNK_KINDS];       ///< a chunk of that kind may be priced (its fixed term + the shared ones)
    double fixed_ms[PULSAR_CHUNK_KINDS];  ///< each kind's fixed term
    /// Cost per prompt row, ms.
    double alpha_ms, beta_ms;             ///< per prompt row, per attention pair
    bool ann_kind[PULSAR_CHUNK_KINDS];    ///< rule 5: the kind's terms were announced
    uint32_t ann_n;                       ///< `n` at the last announcement
    /// alpha_ms as last announced.
    double ann_alpha, ann_beta;
    /** @var pulsar_prefill_cost_fit::ann_beta
     *  beta_ms as last announced. */
    /** L327: the file the fit persists to beside the verify curve, named by the curve's key ("" = none) */
    char cache_path[4096];
    /// The verify curve's key, written into (and checked against) the persisted fit file.
    uint64_t cache_key;
    uint32_t saved_n;                     ///< `n` at the last write
} pulsar_prefill_cost_fit;

/** One loaded model and everything the engine derives from it: family, weights, tokenizer, GPU state, the TP
 *  transport and the speculation counters.  Opened by pulsar_engine::open / open_group, freed by destroy();
 *  sessions (pulsar_session) borrow it. */
struct pulsar_engine {
    /** The model family, chosen once at open from `general.architecture`
     * (pulsar_family_for_model) and never switched; generic code reaches the
     * family's session operations through it (family.h). */
    const pulsar_family *family;
    /** The family's layer plan, built by family->load: THE authority for which
     * op runs at layer il.  DeepSeek: n_layer x PULSAR_LAYER_DS4_BLOCK. */
    pulsar_layer_plan plan;
    /** L295 F3: the family's state declaration (state_kinds.h), built by family->load for a family whose sessions
     *  keep declared state (pulsar_session::states); unused (n_decl 0) on DeepSeek. */
    pulsar_state_plan state_plan;
    /** L295 E1: the loaded family's own state, opaque to the core -- built by family->load, released by
     *  family->unload at destroy (the plugin contract's opaque family state; DeepSeek and Qwen still keep typed
     *  members below).  NULL = the family keeps none here. */
    void *family_state;
    /** The Qwen4-exp family's bound weights; NULL on a DeepSeek engine. */
    pulsar_qwen_weights *qwen_weights;
    /** L284 P15: the Qwen importance-matrix collection's observer while one runs (imatrix_qwen.cpp), else NULL --
     *  the trunk's steps note their linears' input rows into it. */
    struct pulsar_imatrix_tap *imatrix_tap;
    /** The Qwen4-exp family's tokenizer (L251 S5; the shared engine src/lib/bpe_tokenizer.h in its HF
     * configuration), built at open from the checkpoint's own tokenizer.json + generation_config.json; NULL on a DeepSeek engine.  When set,
     * the engine's tokenizer entries (tokenizer.cpp) dispatch to it instead of `vocab`. */
    struct bpe_tokenizer *qwen_tok;
    pulsar_model model;         ///< the target model's mapping and directory
    pulsar_model dspark_model;  ///< drafter mapping; a distinct file only when dspark_external
    pulsar_vocab vocab;         ///< tokenizer tables and special ids
    pulsar_weights weights;     ///< resolved target tensors, per layer
    pulsar_dspark_weights dspark_weights;  ///< resolved drafter tensors
    pulsar_backend backend;     ///< CPU or CUDA
    int spec_depth;             ///< configured starting draft depth (options / PULSAR_SPEC_DEPTH); 0 = the drafter's
    float spec_tau;             ///< configured draft stop threshold (options / PULSAR_SPEC_TAU); 0 = the drafter's, < 0 = none
    /** L263 / L284: each decode lane's measured step cost (pulsar_engine_lane_cost), indexed by
     * pulsar_decode_lane.  Written by the leader's observations (the server's step loops, or the
     * single lane off TP) or, for the spec lane on a TP worker, by the leader's values riding
     * SPEC_ROUND_END_BATCH; the spec lane's is read by the quench guard and the allocator, both by
     * the server's lane choice. */
    pulsar_lane_cost_fit lane_cost[PULSAR_LANE_COUNT];
    /** L288 (tf-gap pass 2): a drafter's lockstep step, measured the same way -- its wall on the chains it
     * carried (one row a chain), fed by a drafter that drafts token by token (the MTP chain) off TP.  What one
     * more step costs the round (pulsar_spec_chain_step). */
    pulsar_lane_cost_fit draft_step_cost;
    /** L301: the spec lane's verify cost curve (verify_curve.cpp; pulsar_session_verify_curve_open installs it,
     *  owned), the one authority every verify pick reads.  NULL = none: the row budget alone admits drafts. */
    struct pulsar_verify_curve *verify_curve;
    /** L301: the drafters' confidence calibration per draft position (pulsar_spec_calibration_note). */
    pulsar_spec_calibration spec_calibration;
    /** L304 / L327: what a prompt chunk costs, solo or fused -- what the server's round selection prices a riding
     *  chunk by (L312, pulsar_engine_fused_prompt_ms) and the disk tier a recompute (L310, pulsar_engine_prefill_ms). */
    pulsar_prefill_cost_fit prefill_cost;
    char *directional_steering_file;   ///< steering-vector file path, or NULL
    float *directional_steering_dirs;  ///< loaded steering directions, or NULL
    float directional_steering_attn_scale;  ///< steering strength on the attention stream
    float directional_steering_ffn_scale;   ///< steering strength on the FFN stream
    uint32_t prefill_chunk;     ///< tokens per prefill chunk
    /** TP state (slices 4b..4f).  Non-NULL only when a TP group was actually
     * configured (tp_role != 0 or tp_peers).  The slab is the host-pinned,
     * GPU-visible registered block pulsar_tp_gpu_slab_alloc_hostpin hands to
     * pulsar_tp_attach_slab. */
    struct pulsar_tp *tp;       ///< transport handle, or NULL when off
    char *tp_kv_dir;            ///< a worker's segment copies (pulsar_engine_options.tp_kv_dir), owned, or NULL
    struct pulsar_tp_plan *tp_plan;   ///< L272 P4b: this rank's slices (tp_slice.cpp), or NULL on one GPU
    uint64_t tp_built_bytes;    ///< 4g-2: device bytes this rank BUILT at open (DeepSeek: its routed-expert half-stacks; Qwen: its dense slices and expert halves) -- resident weights the model's staged count never sees
    void *tp_slab_base;         ///< registered slab base (host-pinned), or NULL
    void *tp_slab_dev;          ///< the slab's device mapping (row-lane kernels), or NULL
    void *tp_bulk_base;         ///< the bulk lane's buffer (host-pinned, v14), or NULL
    void *tp_bulk_dev;          ///< its device mapping (bulk stage/combine), or NULL
    /// Size of the bulk lane's buffer (tp_bulk_base), in bytes; 0 when there is none.
    uint64_t tp_bulk_bytes;
    size_t tp_slab_bytes;       ///< slab size in bytes
    /** Slice 4g (L241): the attention OUTPUT GROUPS this rank owns,
     * [tp_group_lo, tp_group_hi) of PULSAR_N_OUT_GROUP, from the range
     * authority (pulsar_tp_owned_range) at open -- the same rule as the
     * routed experts and the vocab.  The heads a rank computes are exactly
     * these groups' heads; the attn_q_b / attn_output_a row slices it
     * registered at open are exactly these rows.  [0, n_out_group) on one box. */
    uint32_t tp_group_lo, tp_group_hi;
    /** @var pulsar_engine::tp_group_hi
     *  One past the last owned output group. */
    /** Slice 4e: the next session ordinal, handed out by pulsar_session::create
     * as the session's mirror id.  It is an ordinal rather than a random or
     * leader-assigned id because the SAME driver opens the same sessions in the
     * same order on every rank, so both ranks agree without a wire round trip
     * on the create path (which is not itself mirrored).  A frame whose id does
     * not match the receiving session therefore means the drivers diverged, and
     * the receiver fails loud instead of mirroring into the wrong session. */
    uint64_t tp_session_seq;
    /** Slice 4e (L238): a WORKER rank's session registry, keyed by the create
     * ordinal the leader names in every frame; owned and driven only by
     * pulsar_tp_worker_run (tp_worker.cpp).  Empty on the leader. */
    struct pulsar_tp_worker_slot *tp_worker_slots;
    /// Live entries in tp_worker_slots.
    uint32_t tp_worker_n;
    /// Allocated entries in tp_worker_slots (grows by doubling).
    uint32_t tp_worker_cap;
    bool gpu_ready;             ///< CUDA backend initialised and weights resident
    bool live;                  ///< L321: counted among the process's live engines (holds the instance)
    bool dspark_ready;          ///< a usable drafter is loaded; false disables speculation
    const struct pulsar_drafter_ops *drafter_ops;   ///< L272 P1: the loaded drafter behind the round API (spec_ops.h), or NULL = none
    /** L298: the loaded block drafter (spec_block.h: its declaration, tap points, weights, state ids), or NULL -- set
     *  by the family's load when the container holds one and the open selects it; the family's unload frees it. */
    struct pulsar_block_drafter *block_drafter;
    bool dspark_external;       ///< drafter came from its OWN GGUF (separate map/fd), not the target's
    pulsar_vision_weights vision_weights;  ///< resolved ViT tower/aligner tensors (Vision-Exp artifacts)
    bool vision_ready;          ///< the artifact carries a bound, layout-validated vision tower
    pulsar_model overlay_model; ///< donor GGUF for --expert-overlay, if any
    bool overlay_ready;         ///< overlay tensors resolved and swapped in
    /** Prometheus /metrics spec-decode counters (server /metrics endpoint via
     * pulsar_engine_spec_metrics). Incremented from the DSpark fused verify loop;
     * monotonic. GPU decode submission is single-threaded, so plain uint64 is
     * adequate for these monitoring counters. */
    uint64_t spec_accepted_tokens;  ///< accepted draft tokens
    uint64_t spec_draft_tokens;   ///< proposed/verified draft tokens (engine-cumulative)
    uint64_t spec_num_drafts;     ///< draft rounds, i.e. verify steps carrying drafts
    uint64_t spec_gen_tokens;     ///< tokens emitted by the speculative loop
    uint64_t spec_accepted_per_pos[16];  ///< accepted count per draft position
    uint64_t spec_verified_per_pos[16];  ///< rounds that VERIFIED position i (the trim kept
                                         ///< it); the honest denominator for per-position
                                         ///< acceptance under the L107 adaptive depth
    uint64_t spec_copy_probes;    ///< L308: copy chains drafted as a stale estimate's probe while it stood under tau

    /** ---- methods (C++ port): 1:1 mirror of the pulsar_engine_* verb family.
     * The public API in pulsar.h stays the free-function facade (defined in
     * engine_api.cpp); engine internals call these members directly.  Members
     * stay public and the struct stays trivially constructible: lifetime is
     * managed exactly as before via open()/destroy() (xcalloc/free), NOT
     * constructors/destructors. */
    /// Open one engine: a group of one (open_group(out, opt, 1)).
    /// @param out  receives the engine
    /// @param opt  the engine options
    /// @return 0 on success, as open_group().
    static int open(pulsar_engine **out, const pulsar_engine_options *opt);
    /** L321: n engines in one process (pulsar.h pulsar_engine_open_group). */
    /// @param out   n slots; out[i] receives opts[i]'s engine
    /// @param opts  n option sets, one per engine
    /// @param n     engines in the group
    /// @return 0, or 1 with every out[i] NULL (the reason said on stderr).
    static int open_group(pulsar_engine **out, const pulsar_engine_options *opts, uint32_t n);
    void destroy();  ///< was pulsar_engine_close
    /** Print a human-readable model summary (shape, quantisation, memory) to
     * stderr. */
    void summary();
    /** Tokenizer table length. NOT the logits width -- see logits_width(). */
    /// @return the table length; 0 for a family without a tokenizer (refused on stderr).
    int vocab_size();
    /** Logits row width (the shape profile's n_vocab). Size every logits buffer
     * with THIS. @return the row stride, in floats. */
    int logits_width() const;
    /** Model name from the GGUF metadata. */
    /// @return the family's model name (owned by the engine).
    const char *model_name();
    /** Copy out the engine-cumulative speculative-decode counters. */
    /// @param out  zeroed, then filled; NULL is a no-op
    void spec_metrics(pulsar_spec_metrics *out);
    /** Stable id for this model, for cache keys and metrics labels. */
    /// @return the family's model id.
    int model_id();
    /** True when the artifact is a REAP-pruned expert set rather than the full
     * model -- the two have different expert counts and cannot share caches. */
    /// @return true when any layer's expert count differs from PULSAR_N_EXPERT.
    bool is_pruned() const;
    /** GPU bytes pulsar_session::create takes at `ctx_size` with the current
     * bank pool: the allocation code run dry, so the price and the allocation
     * are one function.  The number admission control must use.
     * @return bytes, or 0 if no session could be created. */
    /// @param ctx_size  context size to price, in tokens
    uint64_t session_cost_bytes(int ctx_size);
    /** session_cost_bytes() for an explicit bank-pool size, so the server can
     * evaluate the (banks, ctx) fit table before committing to one.
     * @param ctx_size context size to price
     * @param n_banks  >= 1; 1 is the classic single-session layout
     * @param managed_bytes optional: the demand-paged (cudaMallocManaged) subset, 0 when none was created
     * @return bytes, or 0 if no session could be created. */
    uint64_t session_cost_bytes_banked(int ctx_size, int n_banks, uint64_t *managed_bytes = NULL);
    /** Demand-paged (not reserved) bytes ONE bank actually materialises at
     * `ctx_size` -- the overcommit figure, below the reserved capacity. */
    /// @param ctx_size  context size, in tokens
    /// @return the bytes; zero without the GPU graph backend, before the GPU is ready, or for ctx_size <= 0.
    KvBytes demand_paged_bytes_per_bank(int ctx_size);
    /** Resident weight bytes, excluding per-session state. */
    /// @return bytes; 0 for a NULL engine.
    uint64_t weights_resident_bytes();
    /** Run the dataset through the model accumulating per-tensor activation
     * magnitudes, and write the importance matrix used to steer quantisation.
     * @return 0 on success. */
    /// @param dataset_path  text file of prompts, separated by `===== DS4_IMATRIX_PROMPT` marker lines
    /// @param output_path   where the importance matrix is written
    /// @param ctx_size      per-prompt token cap (<= 0: 32768); longer prompts are truncated
    /// @param max_prompts   stop after this many prompts (<= 0: no cap)
    /// @param max_tokens    total token budget across prompts (<= 0: unlimited)
    int collect_imatrix(const char *dataset_path, const char *output_path,
                        int ctx_size, int max_prompts, int max_tokens);
    /** Bits per weight of the ROUTED expert tensors (the artifact's dominant
     * quantisation), for reporting and tier selection. */
    /// @return bits per weight, from the family (0 for a NULL engine).
    int routed_quant_bits();
    /** True when a usable drafter is loaded and speculation can run. */
    /// @return dspark_ready (false for a NULL engine).
    bool has_dspark();
};

/** A string this struct OWNS, as opposed to ::pulsar_str which borrows.
 * Not NUL-terminated either -- the length is authoritative. */
typedef struct owned_str {
    char *ptr;     ///< the bytes, owned
    uint64_t len;  ///< length in bytes
} owned_str;



/** The per-conversation SPECULATIVE / DSpark host shadow, factored into one
 * named aggregate so it can be saved and restored WHOLESALE.  It is embedded
 * BY VALUE in both pulsar_session (the live state) and pulsar_bank_carry (the
 * per-bank Tier-2 shadow), and pulsar_session_bank_state_save/_restore copy it
 * with a single struct assignment.  That is the whole point: the old
 * field-by-field mirror meant every new field had to be added at three sites
 * (struct, save, restore) and forgetting one silently handed a bank ANOTHER
 * conversation's speculative state.  Add new spec/DSpark scalars HERE and
 * both paths pick them up for free.
 *
 * Deliberately EXCLUDED: heap-backed state (checkpoint, logits) -- those
 * need deep copies with per-side ownership, so they stay as explicit members
 * of each struct. */
typedef struct pulsar_spec_carry_state {
    /** Fused DSpark loop (P2): drafts produced LAST step from the last-accepted
     * position's hidden, pending verification in THIS step's single batched
     * forward (EAGLE pipeline inversion). 0 pending = next step is a plain
     * n=1 forward. Invalidated on rewind/invalidate. */
    int32_t pend[16];
    /** L108 P2: a device-chained greedy draft was LAUNCHED but its ids/conf
     * have not been read back yet.  The read happens lazily ("harvest") at
     * the next consumer -- round assembly, the bank conf peek, or a bank
     * save -- so token emission/streaming overlaps the drafter's GPU time.
     * Every site that DROPS pendings must also drop the flag (the
     * pulsar_spec_drop_pendings helper below is the single authority), or a
     * later harvest would resurrect pendings the reset meant to kill. */
    bool dspark_chain_unharvested;
    bool dspark_chain_conf;      ///< the confidence head ran for the in-flight chain
    uint32_t dspark_chain_n;     ///< drafted depth of the in-flight chain
    uint32_t n_pend;   ///< drafts proposed and awaiting verification
    /** The base token the pending drafts continue from (predicted greedy next).
     * If the caller's next first_token differs (non-greedy interruption, tool
     * injection), the pending drafts are stale and dropped. */
    int32_t pend_base;
    /** checkpoint.len the drafts were produced at — an ACCEPTANCE guard, not an
     * exactness guard. The base-token check above is a VALUE check, not an
     * identity check: a plain pulsar_session_eval (tool injection, think-tag
     * recovery) advances the session and clears the carry but leaves the
     * pendings, so a later first_token that merely COLLIDES with
     * pend_base would resurrect drafts conditioned on a different
     * position.
     *
     * Do NOT read this as an exactness guard and do NOT delete it on the grounds
     * that it isn't one. Both accept rules are PROPOSAL-AGNOSTIC: they are exact
     * for an arbitrary proposal and never read where it came from. q_oldpos IS
     * the distribution the stale draft was actually drawn from, so p_newpos/q_oldpos
     * is a perfectly valid ratio and the rule still emits exactly p_newpos — the
     * draft simply won't get accepted very often. That was true of the
     * deterministic rule (which is why staleness was benign before Item 1) and it
     * stays true under p/q. The cost of staleness is throughput; we drop stale
     * pendings because a draft conditioned on the wrong position is a wasted
     * verify row. Mirrors spec_carry_pos. */
    int32_t pend_pos;
    /** Speculative-sampling carry: the next base token, already drawn from the
     * request's filtered distribution (bonus draw on full accept, residual
     * draw on rejection) but NOT yet forwarded through the target. The next
     * generate_speculative call forwards it as batch position 0. Invalidated
     * with the pendings on rewind/invalidate/sync. */
    int32_t spec_carry_token;  ///< the drawn-but-unemitted base token
    bool spec_carry_valid;     ///< false once the carry has been consumed or voided
    /** checkpoint.len the carry was drawn at; any session advance outside the
     * speculative path (sync, plain eval) moves it and voids the carry */
    int32_t spec_carry_pos;
    /** sampling params the carry was drawn under; a param change between calls
     * drops the carry and redraws from s->logits (exact: the carry was never
     * emitted or forwarded) */
    float spec_carry_temp;   ///< temperature the carry was drawn under
    float spec_carry_top_p;  ///< top-p the carry was drawn under
    float spec_carry_min_p;  ///< min-p the carry was drawn under
    int spec_carry_top_k;    ///< top-k the carry was drawn under; a change voids the carry
    /** Confidence-head score per pending draft, carried draft->verify. Stored
     * UNCONDITIONALLY (-1 when the head didn't run): the L107 adaptive-depth
     * controller reads the verified chain's tail confidence in round_end. */
    float   pend_conf[16];
    /** L288: the pendings are a copy chain (pulsar_spec_copy_chain), not the drafter's: their verdict feeds the
     * copy acceptance (copy_accept). */
    bool    pend_copy;
    /** L288: this request's copy drafts' measured chance to be accepted once the one before was (an EMA over its
     * verified copy chains, weight 0.1 a round; 0 = none yet) -- the confidence its copy drafts carry.  A request
     * boundary clears it (spec_lookahead_reset).  The one copy estimate: pulsar_spec_copy_price reads it,
     * pulsar_spec_copy_observe moves it (session_spec.cpp). */
    float   copy_accept;
    /** L308: tokens the request has committed since copy_accept's last verdict (saturating).  An estimate this old
     * (PULSAR_SPEC_COPY_REPROBE) is stale: the next copy chain is a probe, and its verdict re-seeds the estimate. */
    uint32_t copy_unseen;
    /** L107 adaptive draft depth: the session's CURRENT draft depth, moved
     * +/-1 per round by the controller in spec_round_end from the realized
     * accept count and the verified tail confidence. 0 = uninitialized (first
     * draft reads pulsar_spec_depth_start, which is thereby the STARTING
     * depth, not a fixed width). Persists across requests in a session on
     * purpose: a client's workload regime usually does too. */
    int spec_adaptive_depth;  ///< bounds: the drafter's adaptive policy (pulsar_drafter_ops::adapt)
    bool spec_depth_down_forgiven;  ///< L107 v2: one down-signal was vetoed on a still-confident tail; a second consecutive one backs off regardless
    uint8_t spec_depth_rounds_since_up;  ///< L107 v5: rounds since the last UP, saturating at 255; a down within 2 of an up is a FAILED EXCURSION and triggers the cooldown; a down after a sustained ride carries no penalty (v4's blanket cooldown cost ~0.9 t/s on BOTH server workloads by suppressing profitable climbs).
    uint8_t spec_depth_climb_cooldown;  ///< L107 v4: rounds remaining in which UP is suppressed after a failed excursion. Raw-completion prose oscillated 2->3->4-> crash forever (29 transitions/192 tok, -14%): each failed excursion burns a deep round, and tail conf does NOT separate good climbs from bad (a 0.93 tail climbed into commit=0). Cooldown makes excursions rare after they fail; structured's downs are rare (and v3-forgiven rounds are not downs), so its climb is untouched.
    /** --- Temperature-matched draft sampling (spec-decode Item 1; L288 sdet) ---
     * At temperature > 0 the drafts above are DRAWN from a temperature-matched
     * proposal q (the drafter's refined logits filtered at the request's
     * params) at each draft's own key in the request's stream -- the key the
     * target's draw for that token reads, so a draft is the target's token
     * with probability at least (1 - TV) / (1 + TV) of q and p.
     * The verify walk emits the target's draw and needs nothing of q.
     *
     * The sampling params the pendings were drawn under: drafts drawn under X
     * and verified under Y are still exact (the walk emits the target's draw
     * under Y), but they rarely match it, so the verify guard drops them
     * rather than spend rows on them.  Mirrors the spec_carry_* params guard. */
    float pend_temp;   ///< temperature the pending drafts were proposed under
    float pend_top_p;  ///< top-p the drafts were proposed under
    float pend_min_p;  ///< min-p the drafts were proposed under
    int   pend_top_k;  ///< top-k the drafts were proposed under; the verify guard drops mismatched drafts
    /** --- Terminal yield-quench controller (spec-decode Item 4) ---
     * Per-request cumulative-regret gate: each fused spec step charges
     * debt += guard(n_batch) - tokens_committed (the breakeven yield minus the
     * realized yield, in plain-token equivalents), and once the request has
     * demonstrably lost more than PULSAR_QUENCH_BUDGET plain tokens to
     * speculation with its recent yield still below breakeven, spec_quenched
     * latches and the request decodes PLAIN for its remainder (terminal;
     * re-armed at the request boundaries: sync/invalidate/rewind/load — the
     * same sites that drop the carry and pendings). All-zero is the armed
     * state, so xcalloc'd sessions start armed and spec_quench_reset is a
     * plain zeroing. Controller design after Entrpi ds4 v0.1.1 (MIT).
     *
     * The guard prices the step from the engine's MEASURED cost (L263,
     * pulsar_engine::lane_cost[PULSAR_LANE_SPEC]) -- the quench point moves with the machine,
     * not with a fixed stream; both paths sample the exact target
     * distribution, so only speed is at stake.  Multiseq note: this state belongs to the classic single-request flow;
     * the dormant multi-bank driver would need per-bank copies (not wired —
     * generate_speculative already refuses when mseq_dirty). */
    float spec_quench_debt;  ///< cumulative plain-token-equivalents lost
    float spec_quench_ewma;     ///< EWMA of the per-step margin (realized yield - breakeven guard)
    int spec_round_banks;       ///< banks sharing the round being ended (the flat cost's divisor); 1 in the single lane
    uint32_t spec_quench_steps; ///< fused spec steps taken by this request
    bool spec_quenched;         ///< LATCHED: speculation disabled for the request's remainder
    /** Per-SESSION mirror of the engine's cumulative DSpark counters. The engine
     * copies are global (Prometheus /metrics, cross-request); these let the
     * server compute a per-RESPONSE accept-rate/tokens-per-step by snapshotting
     * at decode start and diffing at finish, which the global copies cannot give
     * because decode quanta from concurrent sessions interleave on the single
     * worker. Incremented alongside the engine counters in the fused verify
     * loop; monotonic since session open (never reset per request). */
    uint64_t spec_accepted_tokens;
    uint64_t spec_draft_tokens;  ///< per-carry draft tokens proposed
    uint64_t spec_num_drafts;    ///< per-carry draft rounds
    uint64_t spec_gen_tokens;    ///< per-carry tokens emitted
} pulsar_spec_carry_state;

/** Drop pendings AND any unharvested in-flight chain (L108 P2). The single
 * authority for every "pendings are stale" reset -- setting n_pend
 * to 0 by hand while a chain is in flight leaves a flag that would resurrect
 * the stale ids at the next harvest. */
/// @param sp  the speculative carry whose pendings go
static inline void pulsar_spec_drop_pendings(pulsar_spec_carry_state *sp) {
    sp->n_pend = 0;
    sp->pend_copy = false;
    sp->dspark_chain_unharvested = false;
}

/** L108 P2: read back an in-flight device-chained draft (ids + conf), apply
 * the conf-sched trim, and populate the pendings. Idempotent; no-op when no
 * chain is in flight. Must run before anything consumes n_pend /
 * pend[], and before a bank save copies spec state (banks share
 * the session's graph tensors, so a saved unharvested flag would harvest
 * another bank's chain). Defined in session_spec.cpp. */
void pulsar_session_spec_chain_harvest(pulsar_session *s);

/** L281: one image block a session's KV holds -- its rows [start, end) and the hash of the image bytes merged there
 *  (pulsar_image_content_hash).  Block ids encode only geometry, so the record is what says WHICH image. */
typedef struct {
    /// First KV row of the block.
    uint32_t start, end;
    /** @var pulsar_image_block::end
     *  One past the block's last KV row. */
    /// pulsar_image_content_hash of the image merged into the block.
    uint64_t content;
    /** L268: the block's 2D layout in rows (the family's pulsar_family_vision::grid, e.g. Qwen's merged patch grid,
     *  rows = grid_h * grid_w) -- what multi-axis rope positions follow; 0 x 0 = a block with no 2D layout. */
    uint32_t grid_h, grid_w;
    /** @var pulsar_image_block::grid_w
     *  The block's 2D layout width, in rows (see grid_h). */
} pulsar_image_block;
/// Most image blocks one KV records (pulsar_image_identity).
#define PULSAR_IMAGE_BLOCKS_MAX 64
/** The blocks a KV holds, in position order (image_identity.cpp). */
typedef struct {
    /// Valid entries in b[].
    uint32_t n;
    /// The block records, in position order.
    pulsar_image_block b[PULSAR_IMAGE_BLOCKS_MAX];
} pulsar_image_identity;
/// FNV-1a 64 of an image's encoded bytes (seeded with their length): the identity a block record carries.
/// @param img  the image (NULL or no bytes hashes as empty)
/// @return the hash; never 0 (0 maps to 1).
uint64_t pulsar_image_content_hash(const pulsar_image_ref *img);
/** The family's geometry (pulsar_family::vision); false / no sentinel on a family without images. */
/// @param e   the engine
/// @param id  a token id
/// @return true when `id` is the family's image-block sentinel.
bool pulsar_image_is_sentinel(const pulsar_engine *e, int32_t id);
/// The image block that begins at ids[start], by the family's geometry.
/// @param e      the engine
/// @param ids    the token ids
/// @param n      entries in `ids`
/// @param start  where the block should begin
/// @param len    out: the block's length in tokens (0 when none)
/// @return true when a block of length > 0 begins there; false on a family without images.
bool pulsar_image_block_extent(const pulsar_engine *e, const int32_t *ids, int n, int start, int *len);
/** The records of `images` (in prompt order, each naming its block in `ids` by the geometry `v` -- the family's,
 *  e->family->vision; `e` is the hook's argument) whose blocks end at or below `limit`.  false = an image names no
 *  block, blocks out of order, or more than PULSAR_IMAGE_BLOCKS_MAX (said). */
/// @param v         the family's vision geometry
/// @param e         the engine (the hooks' argument)
/// @param ids       the prompt's token ids
/// @param n         entries in `ids`
/// @param images    the request's images, in prompt order, start_pos set
/// @param n_images  entries in `images`
/// @param limit     only blocks ending at or below this position are recorded
/// @param out       receives the records
/// @return false (said) on a bad image list.
bool pulsar_image_identity_build(const pulsar_family_vision *v, const pulsar_engine *e, const int32_t *ids, int n,
                                 const pulsar_image_ref *images, int n_images, uint32_t limit,
                                 pulsar_image_identity *out);
/** Keep the records of the blocks that end at or below `pos` (a rewind's survivors). */
/// @param id   the records
/// @param pos  the new history end, in tokens
void pulsar_image_identity_trim(pulsar_image_identity *id, uint32_t pos);
/// Two KVs' image records are the same: every block's rows, content hash and grid.
/// @param a  one record set
/// @param b  the other
/// @return true when equal.
bool pulsar_image_identity_equal(const pulsar_image_identity *a, const pulsar_image_identity *b);
/** The exclusive end of the last block, 0 when none. */
/// @param id  the records
/// @return the exclusive end, in KV rows.
uint32_t pulsar_image_identity_end(const pulsar_image_identity *id);
/** L335: the blocks lying wholly inside rows [lo, hi): b[first, first + n) -- one run of the array, because the
 *  records are in position order and disjoint, which this checks (a segment's save writes the run as it stands). */
/// @param id     the records
/// @param lo     the span's first row
/// @param hi     one past its last row
/// @param first  receives the first such block's index (0 when none)
/// @param n      receives how many there are
/// @return false when the records are not in position order, overlap, or hold an empty block.
bool pulsar_image_identity_span(const pulsar_image_identity *id, uint32_t lo, uint32_t hi, uint32_t *first,
                                uint32_t *n);
/** L268: the interleaved multi-axis rope position (T, H, W) of KV row `row` after the blocks `id` records (HF
 *  get_rope_index; a block without a grid is text).  Text rows have T = H = W; with no gridded block before them
 *  that is the row itself. */
/// @param id   the blocks before `row`
/// @param row  the KV row
/// @param out  receives (T, H, W)
void pulsar_image_rope3(const pulsar_image_identity *id, uint32_t row, uint32_t out[3]);
/** How much of `ids` a disk chain may hold: everything, or up to the first image block `id` does not record
 *  (its rows' image is unknown).  -1 = a malformed block. */
/// @param v    the family's vision geometry (NULL: no images, all of `ids`)
/// @param e    the engine
/// @param ids  the history's token ids
/// @param n    entries in `ids`
/// @param id   the image records the history's KV holds
/// @return the tokens a disk chain may hold, or -1.
int pulsar_image_persist_end(const pulsar_family_vision *v, const pulsar_engine *e, const int32_t *ids, int n,
                             const pulsar_image_identity *id);

/* ---- L268: the image path every family shares (image_front.cpp) ---- */
/** Replace each placeholder token in `prompt` at or past `from` by its image's block (appended to `out`, which the
 *  caller owns and passes empty) and set each image's start_pos; [0, from) is copied unchanged (a continuation's
 *  held history).  false + `err` (a client condition): no tower bound, no placeholder token, a count mismatch, an
 *  image the family cannot prepare. */
/// @param e         the engine
/// @param prompt    the tokenized prompt, one placeholder token per image
/// @param from      tokens before this are copied unchanged
/// @param images    the request's images; each one's start_pos is set
/// @param n_images  entries in `images`
/// @param out       receives the expanded ids (passed empty)
/// @param err       failure message buffer
/// @param errlen    its size
/// @return false + `err` on a client condition.
bool pulsar_image_expand(const pulsar_engine *e, const pulsar_tokens *prompt, int from, pulsar_image_ref *images,
                         int n_images, pulsar_tokens *out, char *err, size_t errlen);
/** Every image names a block the prompt carries and every block fits one prefill chunk of `chunk_cap` rows;
 *  *end_out = the exclusive end of the last block.  The ONE statement of the rule (sync, TP preflight, planner). */
/// @param e          the engine
/// @param ids        the expanded prompt ids
/// @param n          entries in `ids`
/// @param images     the request's images, start_pos set
/// @param n_images   entries in `images`
/// @param chunk_cap  the prefill chunk width, in rows
/// @param end_out    out: the exclusive end of the last block
/// @param err        failure message buffer
/// @param errlen     its size
/// @return false + `err` when a block is missing or wider than a chunk.
bool pulsar_image_spans_fit(const pulsar_engine *e, const int32_t *ids, int n, const pulsar_image_ref *images,
                            int n_images, uint32_t chunk_cap, int *end_out, char *err, size_t errlen);
/** false + `err` when a prompt with no images carries a sentinel or placeholder id. */
/// @param e       the engine
/// @param prompt  the text-only prompt
/// @param err     failure message buffer
/// @param errlen  its size
/// @return true when the prompt is clean.
bool pulsar_image_refuse_orphans(const pulsar_engine *e, const pulsar_tokens *prompt, char *err, size_t errlen);
/** The `block_len` rows image `img` puts at its block `ids[0..block_len)` (row_width bf16 each, into `out`): the
 *  family's assembly over its tower output, the tower run at most once per image per process (the cache). */
/// @param e          the engine
/// @param img        the image
/// @param ids        the block's ids
/// @param block_len  the block's length, in tokens / rows
/// @param out        receives block_len rows of the family's row width, bf16
/// @param cache_hit  out (may be NULL): the tower output came from the process cache
/// @param err        failure message buffer
/// @param errlen     its size
/// @return false + `err` on a failure.
bool pulsar_image_block_rows(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *ids, int block_len,
                             uint16_t *out, bool *cache_hit, char *err, size_t errlen);
/** Write `n_rows` rows of `width` bf16 into a [token][stream][width] carrier at chunk row `row0` (of `n_tokens`),
 *  each replicated into every one of `n_streams` streams -- the hyper-connection expansion both families' models do
 *  after the embedding (DeepSeek's HC, Qwen's 4 streams).  false, without writing, when it does not fit. */
/// @param carrier    the device carrier, [n_tokens][n_streams][width] bf16
/// @param rows       n_rows host rows of `width` bf16
/// @param n_rows     rows to write
/// @param row0       the chunk row the first lands at
/// @param n_tokens   the chunk's rows
/// @param width      row width, in elements
/// @param n_streams  streams each row is replicated into
/// @return true when written.
bool pulsar_image_write_stream_rows(pulsar_gpu_tensor *carrier, const uint16_t *rows, uint32_t n_rows, uint32_t row0,
                                    uint32_t n_tokens, uint32_t width, uint32_t n_streams);
/** The image records of `bank`'s KV: the live bank's (the request's during a sync, whose rows are being written),
 *  another bank's carry, or NULL (no carry: no images). */
/// @param s          the session
/// @param bank       the bank asked about
/// @param live_bank  the installed bank
/// @return the records (borrowed from the session), or NULL.
const pulsar_image_identity *pulsar_session_bank_images(const pulsar_session *s, uint32_t bank, uint32_t live_bank);
/** Where a family's prefill puts block rows: `n_rows` rows at chunk row `row0` of a chunk of `n_tokens`. */
typedef bool (*pulsar_image_row_writer)(void *ud, const uint16_t *rows, uint32_t n_rows, uint32_t row0,
                                        uint32_t n_tokens);
/** The rows of every image block inside the chunk [pos0, pos0+n_tokens) of `ids`, through `write` (a block another
 *  chunk owns is skipped: the planner never splits one).  false = a block is missing or failed (said). */
/// @param e         the engine
/// @param ids       the prompt's ids
/// @param n_ids     entries in `ids`
/// @param images    the request's images, start_pos set
/// @param n_images  entries in `images` (0: nothing to do, true)
/// @param pos0      the chunk's first position
/// @param n_tokens  the chunk's rows
/// @param write     the family's row writer
/// @param ud        passed to `write`
/// @return true when every block in the chunk was written.
bool pulsar_image_merge_chunk(const pulsar_engine *e, const int32_t *ids, int n_ids, const pulsar_image_ref *images,
                              int n_images, uint32_t pos0, uint32_t n_tokens, pulsar_image_row_writer write,
                              void *ud);
/** The reuse licence for an image request against a session's live history (L226, L261, L281). */
typedef struct {
    uint32_t live_len;   ///< the live tokens (0 = none valid)
    uint32_t common;     ///< the prompt's common prefix with them
    bool extends_live;   ///< the prompt extends the live tokens -- the only case the licence decides
    bool straddles;      ///< an image block crosses the common prefix
    bool keep;           ///< extends_live, nothing straddles, and the held images' records are exactly the live ones
    int n_held, n_new;   ///< images inside / after the common prefix
    /** @var pulsar_image_licence::n_held
     *  Images inside the common prefix. */
    uint32_t held_end;   ///< the exclusive end of the last held block: a resume must restore at or above it
} pulsar_image_licence;
/// Decide the reuse licence of an image request against `s`'s live history.
/// @param s         the session
/// @param prompt    the expanded prompt
/// @param images    the request's images, start_pos set
/// @param n_images  entries in `images`
/// @param out       receives the licence (zeroed first)
void pulsar_image_licence_decide(const pulsar_session *s, const pulsar_tokens *prompt,
                                 const pulsar_image_ref *images, int n_images, pulsar_image_licence *out);

/* ---- L268: Qwen3.8-Flash-Next's own part of the image path (vision_qwen.cpp) ----
 * Preprocessing, from the checkpoint's preprocessor_config.json (HF Qwen2VLImageProcessor). */
/// Qwen vision patch side, in pixels.
#define PULSAR_QWEN_VISION_PATCH      16u
/// Qwen vision spatial merge: patches per side merged into one token (2x2).
#define PULSAR_QWEN_VISION_MERGE      2u
/// Qwen vision temporal patch size: frames per patch.
#define PULSAR_QWEN_VISION_TEMPORAL   2u
/// smart_resize lower bound on the resized area, in pixels.
#define PULSAR_QWEN_VISION_MIN_PIXELS 65536u      /* size.shortest_edge */
/// smart_resize upper bound on the resized area, in pixels.
#define PULSAR_QWEN_VISION_MAX_PIXELS 16777216u   /* size.longest_edge */
/// Per-channel normalisation mean (after rescale to [0, 1]).
#define PULSAR_QWEN_VISION_MEAN       0.5
/// Per-channel normalisation standard deviation.
#define PULSAR_QWEN_VISION_STD        0.5
/** One image's patches as the tower takes them: `rows` = grid_h * grid_w patches in 2x2 merge-block order, `cols`
 *  = 3 * temporal * patch^2 float32 values each (channel, frame, y, x). */
typedef struct {
    /// rows x cols float32 values, owned (qwen_vision_pixels_free).
    float *values;
    /// Patches: grid_h * grid_w, in 2x2 merge-block order.
    int rows, cols;
    /** @var qwen_vision_pixels::cols
     *  Values per patch: 3 * temporal * patch^2. */
    int grid_h, grid_w;        ///< the patch grid (the resized image / patch)
    /** @var qwen_vision_pixels::grid_h
     *  Patch-grid height (resized_h / patch). */
    int resized_h, resized_w;  ///< smart_resize's size
    /** @var qwen_vision_pixels::resized_h
     *  smart_resize's height, in pixels. */
} qwen_vision_pixels;
/** HF's smart_resize: the nearest multiple of patch x merge (Python's half-to-even), scaled into
 *  [min_pixels, max_pixels].  false + `err` past an aspect ratio of 200. */
/// @param h       source height, pixels
/// @param w       source width, pixels
/// @param h_out   out: resized height
/// @param w_out   out: resized width
/// @param err     failure message buffer
/// @param errlen  its size
/// @return false + `err` past the aspect-ratio limit.
bool qwen_vision_smart_resize(int h, int w, int *h_out, int *w_out, char *err, size_t errlen);
/** Decode, resize, rescale, normalise and patchify one image exactly as HF's Qwen2VLImageProcessorPil. */
/// @param bytes   the encoded image
/// @param len     its length, in bytes
/// @param out     receives the patches (free with qwen_vision_pixels_free)
/// @param err     failure message buffer
/// @param errlen  its size
/// @return false + `err` on an image that does not decode or resize.
bool qwen_vision_preprocess(const uint8_t *bytes, size_t len, qwen_vision_pixels *out, char *err, size_t errlen);
void qwen_vision_pixels_free(qwen_vision_pixels *p);
/** L268: Qwen's image front (vision_qwen.cpp) -- pulsar_family::vision of PULSAR_FAMILY_QWEN4_EXP. */
extern const pulsar_family_vision PULSAR_QWEN_IMAGE_FRONT;
/** Bind the tower's `model.visual.*` tensors (bf16, the tower's dims) when the artifact carries them; *present
 *  says whether it does.  false = it carries a tower that does not bind (said). */
/// @param m        the model
/// @param w        receives the tower's weights
/// @param present  out: the artifact carries a tower
/// @return true (also with no tower), false when a carried tower does not bind.
bool qwen_vision_bind(const pulsar_model *m, pulsar_qwen_vision_weights_dev *w, bool *present);
/** Preprocess and encode one image: *rows = malloc'd (*n_rows x PULSAR_QWEN_VISION_OUT) bf16, the rows its
 *  image_pad block takes, in merge-block raster order.  `dbg` (NULL in production) receives a malloc'd 3 x
 *  n_patches x PULSAR_QWEN_VISION_DIM stage dump (pulsar_cuda_qwen_vision_forward). */
/// @param w          the bound tower
/// @param bytes      the encoded image
/// @param len        its length, in bytes
/// @param rows       out: the malloc'd rows (caller frees)
/// @param n_rows     out: their count
/// @param dbg        out (NULL in production): the malloc'd stage dump
/// @param n_patches  out: the image's patches
/// @param err        failure message buffer
/// @param errlen     its size
/// @return false + `err` on a failure (*rows NULL).
bool qwen_vision_encode(const pulsar_qwen_vision_weights_dev *w, const uint8_t *bytes, size_t len, uint16_t **rows,
                        int *n_rows, uint16_t **dbg, int *n_patches, char *err, size_t errlen);

/** Tier-2 PATH A: per-bank host carry for the unified bank model.  The shared
 * pool-session's HOST per-conversation state (checkpoint token history, host
 * logits, and the whole DSpark fused-loop / spec-carry shadow) is single-
 * instance on pulsar_session, so time-slicing classic/spec work across banks in
 * one graph must save the leaving bank's carry and restore the entering bank's.
 * gpu_graph_bank_repoint swaps the DEVICE views; this covers the HOST half the
 * engine header (gpu_graph_bank_repoint contract) delegates to the caller.
 * The three heap members are owned deep copies; every other field is a scalar
 * mirror of the identically-named pulsar_session field; the speculative/DSpark
 * scalars are carried wholesale as one embedded pulsar_spec_carry_state. */
/** L302: a session's device sampling state (session_sampler.cpp): the live row -- the authority; the host mirror
 *  pulsar_session::logits is read back only on demand -- each bank's saved live row, the sampler's scratch, and the
 *  rows the last batched step headed into the family's slab. */
typedef struct pulsar_session_dev_rows {
    pulsar_gpu_tensor *live;        ///< [W] f32: the live row
    bool host_stale;                ///< pulsar_session::logits is behind `live`
    pulsar_gpu_tensor *bank_rows;   ///< [n_banks][W] f32: each bank's live row while another bank is installed
    /// Banks bank_rows holds a row for.
    uint32_t n_banks;
    pulsar_gpu_tensor *scratch;     ///< the sampler's and the logprob passes' (the larger of their scratch bytes)
    pulsar_gpu_tensor *out;         ///< PULSAR_GPU_SAMPLE_ROWS_MAX sample outs, then as many walk outs
    pulsar_gpu_tensor *mask;        ///< [PULSAR_GPU_SAMPLE_ROWS_MAX][ceil(W/32)] u32 token masks
    pulsar_gpu_tensor *lp;          ///< [PULSAR_GPU_SAMPLE_ROWS_MAX] logprob rows
    pulsar_gpu_tensor *chain_out;   ///< [PULSAR_GPU_CHAINS_MAX][PULSAR_GPU_CHAIN_SLOTS_MAX] i32: draw_chains' tokens
    /** the rows the last batched step (decode_multiseq / _mixed / _fused, a verify) headed: logits row k at
     *  step_rows + k * W floats, k < step_n -- valid until the session's next step */
    const pulsar_gpu_tensor *step_rows;
    /// Rows valid at step_rows.
    uint32_t step_n;
} pulsar_session_dev_rows;
/// Allocate the session's device sampling state (pulsar_session_dev_rows) for `n_banks` banks.
/// @param s        the session (its engine sets the row width)
/// @param n_banks  banks with a saved row (0 is taken as 1)
/// @return false when an allocation failed.
bool pulsar_session_dev_alloc(pulsar_session *s, uint32_t n_banks);
/// Free the session's device sampling state.
/// @param s  the session
void pulsar_session_dev_free(pulsar_session *s);

/** One bank's saved HOST state while another bank is installed (see the block comment above
 *  pulsar_session_dev_rows: owned deep copies plus scalar mirrors of pulsar_session fields). */
typedef struct pulsar_bank_carry {
    bool      valid;  ///< has this bank's state ever been saved
    /** heap-backed (owned): */
    token_vec checkpoint;  ///< deep copy of s->checkpoint
    /** scalar mirrors: */
    bool      checkpoint_valid;
    bool      logits_stale;     ///< mirror of pulsar_session::logits_stale
    int       prefill_frontier;  ///< L195: mirror of pulsar_session::prefill_frontier
    /** L226 / L281: mirror of pulsar_session::live_images -- the image identity travels with the bank, exactly like
     * the checkpoint it describes, so a bank switch can never pair one conversation's blocks with another's KV. */
    pulsar_image_identity live_images;
    /** Whole speculative/DSpark shadow, mirrored by value (single assignment in
     * save/restore).  NOTE: pulsar_session.mseq_dirty is deliberately NOT carried:
     * it is a property of the GRAPH's scalar frontier counters, not of a bank's
     * conversation, and _restore re-establishes per-bank frontier truth via
     * gpu_graph_bank_counters_install and then clears it unconditionally.  The
     * old mirror field was write-only (saved, never read) and has been dropped. */
    pulsar_spec_carry_state spec;
    void free_one();  ///< was bank_carry_free_one
} pulsar_bank_carry;


/** One conversation's state: the KV it owns, the tokens that produced it, and
 * the host bookkeeping that must stay in step with both.
 *
 * Sessions are NOT thread-safe individually; concurrency comes from running
 * several banks inside one session (see pulsar_bank_slabs), which is why the
 * server owns a single pool session rather than one session per request.
 *
 * The central invariant is that `checkpoint` describes exactly the tokens whose
 * KV rows the graph holds for the CURRENT bank. Every operation that can break
 * that -- sync, rewind, a multiseq step, a bank switch -- either restores it or
 * sets a flag that makes the next classic call fail loud. */
/* Slice 4e: the one condition "this session is mirrored onto a TP pair".
 * Defined in engine_api.cpp beside the mirror itself and declared here because
 * the refusals for operations that are NOT mirrored yet live in other files
 * (session_spec.cpp); the condition has a single definition so it cannot drift
 * between them. */
/// @param s  the session (NULL: false)
/// @return true when the engine has a TP transport and the session a mirror id.
bool pulsar_session_is_mirrored(const pulsar_session *s);

/** L250: identity of the session's cached state as a mirrored sync starts from
 *  it -- FNV-1a over the checkpoint length and tokens.  The leader ships it on
 *  SYNC_CHECK; a worker compares its own.  O(checkpoint) per sync. */
/// @param s  the session (NULL: the FNV offset basis)
/// @return the digest.
uint64_t pulsar_session_checkpoint_digest(const pulsar_session *s);

/** Slice 4e (L238 increment 4): the speculative round family's LOCAL
 * implementations (session_spec.cpp).  The public pulsar_session_spec_*
 * entry points (engine_api.cpp) mirror onto the pair and call these; the
 * worker loop calls these directly.  Every rng a call consumes rides its
 * frame, so the two ranks draw from the same state by construction. */
/// The carry-or-sample base draw (pulsar_session_spec_next_base): the carried token when its sampling knobs and
/// position match, else a fresh draw through `mask`.
/// @param s            the session (installed bank)
/// @param temperature  sampling temperature
/// @param top_k        top-k cutoff
/// @param top_p        nucleus cutoff
/// @param min_p        relative probability floor
/// @param mask         token mask for a fresh draw, or NULL
/// @param rng          the request's sampler stream (read, not stepped)
/// @return the base token, or PULSAR_SAMPLE_REFUSED.
int pulsar_session_spec_next_base_local(pulsar_session *s, float temperature, int top_k,
                                        float top_p, float min_p, const uint32_t *mask, uint64_t *rng);
/// Begin a round on the installed bank (pulsar_session_spec_round_begin): guards, frontier snapshot, the base and
/// its drafts pushed into `r`.
/// @param s             the session
/// @param r             the round, filled
/// @param first_token   the round's base token
/// @param max_tokens    the most tokens the round may emit
/// @param accepted_cap  the caller's accepted[] size
/// @param temperature   request temperature (<= 0: greedy)
/// @param top_k         request top-k
/// @param top_p         request top-p
/// @param min_p         request min-p
/// @param err           failure message buffer
/// @param errlen        its size
/// @return 0, or -1 with `err`.
int pulsar_session_spec_round_begin_local(pulsar_session *s, pulsar_spec_round *r, int first_token,
                                          int max_tokens, int accepted_cap, float temperature,
                                          int top_k, float top_p, float min_p, char *err, size_t errlen);
/// Finish the round against the last shared forward's rows (pulsar_session_spec_round_end).
/// @param s             the session
/// @param r             the begun round
/// @param first_token   its base token
/// @param temperature   request temperature
/// @param top_k         request top-k
/// @param top_p         request top-p
/// @param min_p         request min-p
/// @param greedy_rows   bit i set = verify row i draws greedy
/// @param masks         per verify row token masks, or NULL
/// @param mask_rows     rows in `masks`
/// @param rng           the request's sampler stream, stepped past every emitted token
/// @param row0          the round's first headed row in the step
/// @param accepted      receives the emitted tokens
/// @param accepted_cap  its capacity
/// @param err           failure message buffer
/// @param errlen        its size
/// @return the tokens emitted, or -1 (session poisoned).
int pulsar_session_spec_round_end_local(pulsar_session *s, pulsar_spec_round *r, int first_token,
                                        float temperature, int top_k, float top_p,
                                        float min_p, uint32_t greedy_rows,
                                        const uint32_t *masks, uint32_t mask_rows, uint64_t *rng, uint32_t row0,
                                        int *accepted, int accepted_cap, char *err, size_t errlen);
/// Abort a begun round whose forward failed: drop its frontier snapshot and unverified rows.
/// @param s  the session
/// @param r  the round
void pulsar_session_spec_round_abort_local(pulsar_session *s, pulsar_spec_round *r);
/// Arm (n_rows > 0) or disarm (0) the drafter capture for the next batched forward (pulsar_session_spec_arm_capture).
/// @param s       the session
/// @param n_rows  rows the next forward captures
void pulsar_session_spec_arm_capture_local(pulsar_session *s, uint32_t n_rows);
/** L301 / L315: the verify curve's profile on the installed bank, every 1..rows_max-row verify forward timed into
 *  base[1..rows_max] (verify_curve.cpp has the procedure); `base` NULL runs the walk untimed (L306: the startup
 *  warmup's catalog of the verify widths, a cached curve).  The _local form is this rank's own run -- a pair's
 *  worker runs it from the VERIFY_PROFILE frame; pulsar_session_verify_profile is the leader's, every rank in step
 *  (one rank: the local run).  false / -1 with `err`. */
/// @param s         the session (installed bank)
/// @param rows_max  the widest verify forward profiled, in rows
/// @param base      receives base[1..rows_max] in ms, or NULL for an untimed walk
/// @param err       failure message buffer
/// @param errlen    its size
/// @return false with `err`.
bool pulsar_session_verify_profile_local(pulsar_session *s, uint32_t rows_max, double *base, char *err,
                                         size_t errlen);
/// The leader's verify profile, every rank in step (one rank: the local run).
/// @param s         the session
/// @param rows_max  the widest verify forward profiled, in rows
/// @param base      receives base[1..rows_max] in ms, or NULL
/// @param err       failure message buffer
/// @param errlen    its size
/// @return 0, or -1 with `err`.
int pulsar_session_verify_profile(pulsar_session *s, uint32_t rows_max, double *base, char *err, size_t errlen);
/// One drafter pass over every requesting bank's deferred redraft (pulsar_session_spec_redraft_batch).
/// @param s       the session
/// @param rounds  n live rounds
/// @param banks   their banks
/// @param rngs    each bank's sampler stream (read, never stepped)
/// @param n       entries in each array (<= 0: nothing to do)
/// @param err     failure message buffer
/// @param errlen  its size
/// @return 0, or -1 with `err` on a device failure.
int pulsar_session_spec_redraft_batch_local(pulsar_session *s, pulsar_spec_round **rounds,
                                            const uint32_t *banks, uint64_t **rngs, int n,
                                            char *err, size_t errlen);
/// Stamp the installed bank's shadow with its round's batched redraft (pulsar_session_spec_redraft_commit).
/// @param s  the session
/// @param r  the round
void pulsar_session_spec_redraft_commit_local(pulsar_session *s, pulsar_spec_round *r);
/** L260: the batch phases' local implementations (the steps' contract is on
 * pulsar_session_spec_assemble_batch in pulsar.h); reqs may be NULL (the
 * worker's rows ride the mixed-batch frame). */
/// @param s           the session
/// @param steps       n steps, statuses written
/// @param n           steps
/// @param row_budget  rows the shared forward holds
/// @param reqs        receives the rows (row_budget entries), or NULL
/// @param n_rows_out  out: rows written
void pulsar_session_spec_assemble_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n,
                                              uint32_t row_budget,
                                              pulsar_multiseq_req *reqs, uint32_t *n_rows_out);
/// The round-end phase over every step (pulsar_session_spec_round_end_batch), locally.
/// @param s      the session
/// @param steps  n steps from the assemble phase
/// @param n      steps
void pulsar_session_spec_round_end_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n);
/// The redraft-commit phase over every step (pulsar_session_spec_redraft_commit_batch), locally.
/// @param s      the session
/// @param steps  n steps
/// @param n      steps
void pulsar_session_spec_redraft_commit_batch_local(pulsar_session *s, pulsar_spec_step *steps, int n);
/** L260: a batch phase's verdict -- a positive fingerprint of what the phase
 * decided for every step (statuses, base tokens and rows; frontiers and
 * accepted tokens), identical on ranks that agree.  n_rows: assemble's total
 * rows, 0 for the other phases. */
typedef enum {
    PULSAR_SPEC_PHASE_ASSEMBLE = 1,
    PULSAR_SPEC_PHASE_ROUND_END = 2,
    PULSAR_SPEC_PHASE_REDRAFT_COMMIT = 3,
} pulsar_spec_phase;
/// @param phase   the phase whose outcome is fingerprinted
/// @param steps   n steps
/// @param n       steps
/// @param n_rows  assemble's total rows; 0 for the other phases
/// @return a positive 30-bit fingerprint.
int pulsar_spec_steps_verdict(pulsar_spec_phase phase, const pulsar_spec_step *steps, int n,
                              uint32_t n_rows);

/** Slice 4e (L238): the failure report a void mirrored operation can make --
 * marks the pair failed and prints the reason once.  Defined in engine_api.cpp,
 * shared with the worker loop. */
void pulsar_tp_mirror_fail_void(struct pulsar_tp *tp, const char *operation, const char *why);
/** The worker loop's per-frame body, exposed for the mirror test (which feeds
 * it frames for sessions that do not exist and asserts the refusals).  Returns
 * 1 to continue, 0 on STOP, -1 on a failure that ends the loop (err filled). */
/// @param e       the worker's engine
/// @param c       one received frame
/// @param err     failure message buffer
/// @param errlen  its size
/// @return 1 continue, 0 STOP, -1 failure (err filled).
int pulsar_tp_worker_dispatch(pulsar_engine *e, const pulsar_tp_command *c,
                              char *err, size_t errlen);


/** One conversation's state against an engine: the KV the family's device state holds (graph or driver state),
 *  the tokens it describes (`checkpoint`), the live logits row, the bank pool's carries and the speculation state.
 *  Created by pulsar_session::create, freed by destroy(); borrows its engine. */
struct pulsar_session {
    pulsar_engine *engine;    ///< borrowed; the engine outlives every session
    /** Slice 4e: this session's mirror id, or 0 when the pair is not armed (or
     * the engine handed this session out before the transport existed).  Every
     * mirrored frame carries it; see tp_session_seq above for why it is the
     * create ordinal and why a mismatch is refused. */
    uint64_t tp_session_id;
    /** L260: the batched round end's deferred drafter seed.  While `active`
     *  (pulsar_session_spec_round_end_batch_local), each bank's round end records
     *  its committed capture rows here instead of seeding them one GEMV row at a
     *  time; the batch then seeds every bank in one pass
     *  (gpu_graph_dspark_seed_rows_banked). */
    struct {
        bool active;
        uint32_t n;
        uint32_t src_row[PULSAR_SPEC_LOGITS_ROWS + 1];
        uint32_t bank[PULSAR_SPEC_LOGITS_ROWS + 1];
        int32_t next_tok[PULSAR_SPEC_LOGITS_ROWS + 1];   ///< the token after each row (a drafter that pairs rows with their successor, L272 P1)
    } seed_defer;
    pulsar_gpu_graph *graph;  ///< L272 P6: the DeepSeek family's device state (KV, scratch, bank views), owned by its create / destroy; NULL on any other family
    /** L295 F4: the core row-step driver's buffers (step_driver.h) -- the base of the family's own session state
     *  (Qwen: pulsar_qwen_state) on a family that runs on the driver; NULL on a DeepSeek session. */
    pulsar_driver_state *driver;
    /** L295 F3: the session's declared state (state_kinds.h) -- every bank's per-layer state, positions and grid
     *  checkpoints, built from the engine's state_plan by the family's create; NULL on a DeepSeek session. */
    pulsar_state_set *states;
    token_vec checkpoint;     ///< tokens whose KV the graph currently holds, current bank
    /** L302: the live row -- the distribution of the token after `checkpoint` -- lives ON THE DEVICE (dev.live) and
     *  every draw, argmax and logprob reads it there (session_sampler.cpp).  `logits` is its host mirror, read back
     *  only when a host reader asks for it (live_host(): the payload, a bank carry on the host, a copy out). */
    float *logits;            ///< the live row's host mirror, pulsar_engine_logits_width() floats (live_host())
    /// The device sampling state: the live row (the authority), bank rows, sampler scratch.
    pulsar_session_dev_rows dev;
    pulsar_session_progress_fn progress;   ///< durable progress callback: fires at real checkpoint boundaries
    void *progress_ud;                     ///< user data for progress
    pulsar_session_progress_fn display_progress;  ///< UI-only progress; may fire MID-chunk, so not a checkpoint boundary
    void *display_progress_ud;             ///< user data for display_progress
    pulsar_session_cancel_fn cancel;       ///< cooperative cancellation hook, polled at safe points only
    void *cancel_ud;                       ///< user data for cancel
    /** L329 (b2): the prompt a sync's read-ahead follows (pulsar_session_set_prefill_ahead; borrowed for one sync),
     *  NULL = the sync's own */
    const pulsar_tokens *prefill_ahead;
    /** L329 (b2): each bank's read-ahead record (pulsar_read_ahead_plan), [n_ahead], grown on first use */
    struct pulsar_read_ahead_mark *ahead;
    /// Entries allocated in `ahead`.
    uint32_t n_ahead;
    /** The disk-KV staging buffer: PULSAR_SESSION_IO_CHUNK pinned bytes, allocated once at create (a live session
     *  only; a priced dry one has none).  Every segment and kv-state payload streams through it, so a persist or a
     *  restore allocates no host memory and its device reads are DMA into pinned pages, not the driver's pageable
     *  staging (the 2026-10-29 production hang: a persist at ~190k context under ~5 GB MemAvailable). */
    uint8_t *io_buf;
    uint32_t prefill_cap;                  ///< max tokens per prefill chunk for this session
    int ctx_size;                          ///< allocated context length, in tokens
    bool checkpoint_valid;                 ///< false when `checkpoint` no longer describes the graph's KV (forces a rebuild on the next sync)
    /** L264: s->logits do NOT describe the next token after `checkpoint` -- the
     *  bank was cut (rewind) or put at a grid checkpoint (restore, a segment
     *  chain's load), which moves the KV but not the logits.  Set by every cut,
     *  cleared by a prefill or eval that writes them; a sync whose prompt needs
     *  no new rows re-evaluates the last one while it is set.  Fails safe: a
     *  writer that forgets to clear it costs one row, never a wrong sample.
     *  One flag for every family (L272 P2: Qwen's logits_fresh folded into it). */
    bool logits_stale;
    /** L264 S4e: the leader is inside a mirrored sync -- the workers read only
     *  chunk verdicts until it returns, so no other mirrored frame may ship
     *  (a disk KV store from the prefill's progress callback). */
    bool tp_in_sync;
    /** L281: the image blocks inside `checkpoint`, one record each (image_identity.cpp).  The blocks' TOKEN IDS
     * encode only their geometry, never the pixels, so a client that swaps an image for a different one of the same
     * size produces an identical token prefix -- the records are what keep such a request from reusing KV rows
     * computed from the other image.  rewind() keeps the records of the blocks that survive it. */
    pulsar_image_identity live_images;
    /** L268: the images of the sync in flight on the core driver (sync_driver.cpp), BORROWED for its prefill so a
     *  family's chunk forward merges the blocks it owns (pulsar_image_merge_chunk); NULL / 0 outside a sync. */
    const pulsar_image_ref *sync_images;
    /// Entries in sync_images.
    int sync_n_images;
    const pulsar_tokens *sync_prompt;        ///< ... the prompt their blocks sit in
    pulsar_image_identity sync_identity;     ///< ... and every block's record (the rows being written, for rope)
    int resume_origin;                     ///< L194 instrument: the position the last sync's resume started evaluating from (a grid point, 0 = cold from the start), -1 when the sync did not resume
    /** L260 fusion: the last successful fused step's logits block (the caller's
     *  buffer), its decode-row count and its headed rows -- what
     *  pulsar_session_note_prefilled's `head` indexes.  NULL when the last fused
     *  step failed or none ran. */
    bool fused_valid = false;
    /// Decode rows of the last fused step.
    uint32_t fused_n_dec = 0;
    /// Rows the last fused step headed.
    uint32_t fused_heads = 0;
    int prefill_frontier;                  ///< L195: the last position a PREFILL wrote for this checkpoint (decode advances the checkpoint, not this); the resume grid point is derived from min(checkpoint, this); clamped by rewind, carried per bank and in the payload
    /** A multiseq step has run and this session's per-bank state is no longer
     * re-establishable by bookkeeping alone.
     *
     * ⚠ The MECHANISM changed in stage 1b, the HAZARD did not. It used to be
     * that the scalar frontier counters held a cross-bank superset and a
     * classic entry decoding against them would emit at the superset index and
     * attend over a previous tenant's bytes. Those scalars are gone
     * (gpu_graph_n_comp() reads ms_n_comp[cur_bank] directly), so that exact
     * shape is unrepresentable — but a step that touched OTHER banks still
     * leaves this session's notion of which bank is live, and the rest of the
     * per-bank carry, needing a re-establish. Keep the guard.  checkpoint_valid
     * does NOT cover this: pulsar_session_eval never reads it.  Set on every
     * decode_multiseq path that armed a step; cleared only where per-bank
     * device state is legitimately re-established (pulsar_session_sync's rebuild
     * path, via gpu_graph_reset_prefill_state zeroing the counters). */
    bool mseq_dirty;
    /** GPU bytes this session's create actually allocated (tensor-allocator
     * delta across pulsar_session_create); the server ledger commits this. */
    uint64_t resident_bytes;
    /** Live speculative/DSpark shadow; see pulsar_spec_carry_state. */
    pulsar_spec_carry_state spec;
    /** Tier-2 PATH A: per-bank host carry, one entry per pool bank.  Lazily
     * allocated on the first pulsar_session_bank_state_save; NULL / bank_carry_n==0
     * when the pool is disabled (single-session use never touches it). */
    pulsar_bank_carry *bank_carry;
    uint32_t bank_carry_n;   ///< carries allocated; 0 when the pool is disabled

    /** ---- methods (C++ port): 1:1 mirror of the pulsar_session_* verb family.
     * The public API in pulsar.h stays the free-function facade (defined in
     * engine_api.cpp); engine internals call these members directly.  Members
     * stay public and the struct stays trivially constructible: lifetime is
     * managed exactly as before via create()/destroy() (xcalloc/free), NOT
     * constructors/destructors.
     * NOTE: pulsar_session_prefill_cap and pulsar_session_resident_bytes stay
     * free functions — members would collide with the same-named data members.
     * pulsar_session_snapshot_free does not take a
     * session and stays free. */
    /// Create a session against `e`. Behind pulsar_session_create().
    /// @param out       receives the session (free with destroy())
    /// @param e         the engine; borrowed, must outlive the session
    /// @param ctx_size  context length to allocate, in tokens (> 0)
    /// @return 0 on success, 1 on bad arguments, an engine without a ready GPU backend, or a failed allocation.
    static int create(pulsar_session **out, pulsar_engine *e, int ctx_size);
    /** Tear down the session and release its GPU allocations. Behind pulsar_session_free(). */
    void destroy();
    /** Install the durable progress callback. Behind pulsar_session_set_progress(). */
    /// @param fn  the callback (NULL clears it)
    /// @param ud  user data passed to `fn`
    void set_progress(pulsar_session_progress_fn fn, void *ud);
    /** Install the UI-only progress callback -- may fire mid-chunk, so it is NOT
     * a durable KV checkpoint boundary. Behind pulsar_session_set_display_progress(). */
    /// @param fn  the callback (NULL clears it)
    /// @param ud  user data passed to `fn`
    void set_display_progress(pulsar_session_progress_fn fn, void *ud);
    /** Install the cooperative cancellation hook, checked only at safe
     * boundaries. Behind pulsar_session_set_cancel(). */
    /// @param fn  the hook (NULL clears it)
    /// @param ud  user data passed to `fn`
    void set_cancel(pulsar_session_cancel_fn fn, void *ud);
    /** Longest common TOKEN prefix between the session's history and `prompt`.
     * For the byte-level, seam-aware answer use prefix_match(). */
    /// @param prompt  the tokens to compare with the installed bank's history
    /// @return the shared prefix length, in tokens; 0 when the checkpoint is not valid.
    int common_prefix(const pulsar_tokens *prompt);
    /** L115: the prefix-reuse authority (see pulsar.h). */
    /// @param prompt  the prompt to match against the session's history
    /// @param out     receives the verdict (pulsar.h pulsar_prefix_match)
    void prefix_match(const pulsar_tokens *prompt, pulsar_prefix_match *out);
    /** L302 (session_sampler.cpp): the live row from row `row` of a step's device rows / from a host row; the host
     *  mirror, read back if the device moved past it (NULL on a device failure); each bank's row kept on the device
     *  across a bank switch. */
    /// @param rows  a step's device rows, logits-width f32 each
    /// @param row   the row to take
    /// @return false (said) when the copy fails; the host mirror is then stale.
    bool live_from_rows(const pulsar_gpu_tensor *rows, uint32_t row);
    /// The live row from a host row (copied into `logits`, then uploaded).
    /// @param row  logits-width floats (may be `logits` itself)
    /// @return false when the upload fails.
    bool live_from_host(const float *row);
    /// The live row's host mirror, read back first when the device moved past it.
    /// @return `logits`, or NULL when the readback fails.
    const float *live_host();
    /// Copy the live row into `bank`'s saved row on the device.
    /// @param bank  bank index, < dev.n_banks
    /// @return false for an out-of-range bank or a failed copy.
    bool live_save_bank(uint32_t bank);
    /// Make `bank`'s saved row the live row (on the device; the host mirror becomes stale).
    /// @param bank  bank index, < dev.n_banks
    /// @return false for an out-of-range bank or a failed copy.
    bool live_restore_bank(uint32_t bank);
    /** Row `row` of the last batched step becomes `bank`'s saved live row (pulsar_session_step_row_to_bank). */
    /// @param row   headed row of the last batched step, < dev.step_n
    /// @param bank  bank index, < dev.n_banks
    /// @return false (said) on a bad row / bank or a failed copy.
    bool step_row_to_bank(uint32_t row, uint32_t bank);
    /** THE launch: rows row_index[0 .. n) of `rows` by reqs (mask_row is set here from `masks`, NULL
     *  = none), out[k] from the device.  false (said) on a bad launch or a device failure. */
    /// @param rows       the device rows, logits-width f32 each
    /// @param row_index  n row indices into `rows`, one per draw
    /// @param n          draws, 1 .. PULSAR_GPU_SAMPLE_ROWS_MAX
    /// @param reqs       n sampler requests; `row` and `mask_row` are written here
    /// @param masks      per draw, a token mask (NULL entry = none) or NULL for no masks at all
    /// @param out        n sampler outs, read back from the device
    /// @return false (said) on a bad launch or a device failure; a refused ROW is its out's token -1, not a failure.
    bool draw_rows(const pulsar_gpu_tensor *rows, const uint32_t *row_index, uint32_t n, pulsar_gpu_sample_req *reqs,
                   const uint32_t *const *masks, pulsar_gpu_sample_out *out);
    /** draw_rows over rows `width` wide (<= the logits width) whose index i is token id_map[i] (NULL = i): a
     *  drafter's head rows, no mask. */
    /// @param rows       the device rows, `width` f32 each
    /// @param width      row width, <= the logits width
    /// @param id_map     device map row index -> token id, or NULL for the identity
    /// @param row_index  n row indices into `rows`
    /// @param n          draws, 1 .. PULSAR_GPU_SAMPLE_ROWS_MAX
    /// @param reqs       n sampler requests
    /// @param out        n sampler outs, read back from the device
    /// @return false (said) on a bad launch or a device failure.
    bool draw_rows_width(const pulsar_gpu_tensor *rows, uint32_t width, const pulsar_gpu_tensor *id_map,
                         const uint32_t *row_index, uint32_t n, pulsar_gpu_sample_req *reqs, pulsar_gpu_sample_out *out);
    /** One draw over a row of `width` made from n candidates (ids, values; every other entry -inf): a drafter whose
     *  head scores a few candidates a slot.  The token, or PULSAR_SAMPLE_REFUSED. */
    /// @param cand        device candidate ids
    /// @param cand_off    byte offset of the first candidate in `cand`
    /// @param scores      device candidate scores
    /// @param scores_off  byte offset of the first score in `scores`
    /// @param slots       draws per chain
    /// @param topk        candidates per slot
    /// @param vocab       the row width the candidates index
    /// @param reqs        n chain requests
    /// @param n           chains
    /// @param tokens      receives n * slots tokens (read back from the device)
    /// @return false when the launch or the readback fails.
    bool draw_chains(const pulsar_gpu_tensor *cand, uint64_t cand_off, const pulsar_gpu_tensor *scores,
                     uint64_t scores_off, uint32_t slots, uint32_t topk, uint32_t vocab,
                     const pulsar_gpu_chain_req *reqs, uint32_t n, int32_t *tokens);
    /** The launch alone (masks as draw_rows'), the outs left in dev.out (a walk reads them there). */
    /// @param rows       the device rows, logits-width f32 each
    /// @param row_index  n row indices into `rows`
    /// @param n          draws, 1 .. PULSAR_GPU_SAMPLE_ROWS_MAX
    /// @param reqs       n sampler requests
    /// @param masks      as draw_rows'
    /// @return false (said) on a bad launch.
    bool draw_rows_dev(const pulsar_gpu_tensor *rows, const uint32_t *row_index, uint32_t n,
                       pulsar_gpu_sample_req *reqs, const uint32_t *const *masks);
    /** The rows' identity digest, on the device (pulsar_gpu_rows_digest): rows [row0, row0 + n) of `rows`. */
    /// @param rows  the device rows, logits-width f32 each
    /// @param row0  first row
    /// @param n     rows to digest
    /// @param out   receives the digest
    /// @return false (said) on a failure.
    bool rows_digest(const pulsar_gpu_tensor *rows, uint32_t row0, uint32_t n, uint64_t *out);
    /** One draw from the live row (pulsar_session_draw). */
    /// @param d  the draw's knobs, stream and mask; d->rng is stepped past a drawn token
    /// @return the token, or PULSAR_SAMPLE_REFUSED.
    int draw(const pulsar_row_draw *d);
    /** Draws from rows of the last batched step (pulsar_session_draw_rows). */
    /// @param rows    n headed-row indices of the last batched step, each < dev.step_n
    /// @param d       n draws, one per row
    /// @param n       draws, 1 .. PULSAR_GPU_SAMPLE_ROWS_MAX
    /// @param tokens  receives n tokens (PULSAR_SAMPLE_REFUSED for a degenerate row)
    /// @return 0, or 1 when nothing was drawn.
    int draw_step_rows(const uint32_t *rows, const pulsar_row_draw *d, uint32_t n, int *tokens);
    /** Logprobs of a row of the last batched step (pulsar_session_step_row_logprobs). */
    /// @param row          headed row of the last batched step, < dev.step_n
    /// @param token        the token to score (< 0: none)
    /// @param chosen       receives the scored token's logprob when *have_chosen
    /// @param have_chosen  out: 1 when `token` was scored, else 0
    /// @param top          receives the top k (NULL: not wanted)
    /// @param k            top candidates wanted, clipped to [0, PULSAR_GPU_LOGPROB_TOPK_MAX]
    /// @param n_top        out: finite entries in `top`
    /// @return 0, or 1 on a bad row or a device failure.
    int step_row_logprobs(uint32_t row, int token, pulsar_token_score *chosen, int *have_chosen,
                          pulsar_token_score *top, int k, int *n_top);
    /** Highest-scoring token in the session's current logits row. */
    /// @return the token, or PULSAR_SAMPLE_REFUSED (-1) for a degenerate row.
    int argmax();
    /** argmax() ignoring one id -- used to keep a forced continuation off EOS. */
    /// @param excluded_id  the id masked out (ignored when outside the row)
    /// @return the token, or PULSAR_SAMPLE_REFUSED.
    int argmax_excluding(int excluded_id);
    /** Sample from the current logits with the given knobs. `rng` is advanced. */
    /// @param temperature  sampling temperature
    /// @param top_k        top-k cutoff
    /// @param top_p        nucleus cutoff
    /// @param min_p        relative probability floor
    /// @param rng          the request's sampler stream (required)
    /// @return the token, or PULSAR_SAMPLE_REFUSED (also for a NULL `rng`).
    int sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
    /** Fill `out` with the k highest-logprob candidates. @return count written. */
    /// @param out  receives k entries (clipped to PULSAR_GPU_LOGPROB_TOPK_MAX); past the finite ones id -1, NEG_INF
    /// @param k    candidates wanted
    int top_logprobs(pulsar_token_score *out, int k);
    /** Score one specific token from the current logits. */
    /// @param token  the token to score
    /// @param out    receives the score; untouched on 0
    /// @return 1 when scored, 0 for a bad token, a non-finite row or a device failure.
    int token_logprob(int token, pulsar_token_score *out);
    /** Copy the logits row out. Size `cap` with pulsar_engine_logits_width(),
     * NOT with the tokenizer's vocab size. */
    /// @param out  destination
    /// @param cap  its capacity, in floats
    /// @return floats written (the logits width), or 0 when `cap` is short or the readback fails.
    int copy_logits(float *out, int cap);
    /** Overwrite the session's logits row (replay/testing). */
    /// @param logits  the new row
    /// @param n       its length; must equal the logits width
    /// @return 0, or 1 on a wrong length or a failed upload.
    int set_logits(const float *logits, int n);
    /** Decode ONE row per request across `n` banks in a single batched step.
     * Each row lands at its own bank's frontier.
     * @param reqs        one entry per participating bank
     * @param n           entries in `reqs`
     * @param logits      receives one row per request, in `reqs` order
     * @param logits_cap  the TOTAL float capacity of `logits` (all rows: >= n x the logits width)
     * @param err         failure message buffer
     * @param errlen      its size
     * @return 0 on success. */
    int decode_multiseq(const pulsar_multiseq_req *reqs, uint32_t n,
                        float *logits, int logits_cap, char *err, size_t errlen);
    /** The general batched step: rows may belong to different banks AND carry
     * different row counts, so one call can mix decode rows with a prefill
     * chunk.
     * @param reqs           the rows to evaluate, any mix of banks and lengths
     * @param n_rows         entries in `reqs`
     * @param logits         receives the output-head rows
     * @param logits_cap     the TOTAL float capacity of `logits` (all rows: >= the headed rows x the logits width)
     * @param out_n_rows     rows actually produced
     * @param max_head_runs  caps how many separate output-head runs the step performs
     * @param err            failure message buffer
     * @param errlen         its size
     * @return 0 on success. */
    int decode_mixed(const pulsar_multiseq_req *reqs, uint32_t n_rows,
                     float *logits, int logits_cap, uint32_t *out_n_rows,
                     uint32_t max_head_runs, char *err, size_t errlen);
    /** L260 fusion: pulsar_session_decode_fused's local body (contract in pulsar.h). */
    /// @param reqs        the step's rows: n_dec decode rows, then the prefill runs
    /// @param n_rows      entries in `reqs`, <= the session's prefill_cap
    /// @param shape       decode rows, prefill runs and which runs are headed
    /// @param logits      receives (n_dec + headed runs) rows, or NULL
    /// @param logits_cap  the TOTAL float capacity of `logits` (all rows: >= (n_dec + headed runs) x the logits width)
    /// @param out_n_rows  out: rows produced
    /// @param err         failure message buffer
    /// @param errlen      its size
    /// @return 0 on success, 1 on failure (err filled).
    int decode_fused(const pulsar_multiseq_req *reqs, uint32_t n_rows,
                     const pulsar_fused_shape *shape, float *logits, int logits_cap,
                     uint32_t *out_n_rows, char *err, size_t errlen);
    /** Release the host-side per-bank carry (checkpoints, logits, pendings). */
    void bank_carry_free();
    /** L260 fusion: pulsar_session_note_prefilled's local body. */
    /// @param toks  the chunk's tokens
    /// @param n     their count (> 0)
    /// @param head  index of the headed run of the last fused step that finishes the prompt, or -1 on an intermediate chunk
    /// @return 0, or 1 (refused, said).
    int note_prefilled(const int *toks, int n, int head);
    /** Generate with the drafter: propose a block, verify it against the target
     * in one pass, and commit the accepted prefix.
     * @param temperature   sampling temperature; 0 selects argmax
     * @param top_k         top-k cutoff
     * @param top_p         nucleus cutoff
     * @param min_p         relative probability floor
     * @param rng           sampler state; advanced by this call
     * @param max_tokens    cap on tokens committed
     * (a stop -- the family's whole set, pulsar_token_is_stop -- ends the block)
     * @param accepted      receives the committed token ids
     * @param accepted_cap  its capacity
     * @param err           failure message buffer
     * @param errlen        its size
     * @return tokens committed. */
    int generate_speculative(float temperature, int top_k, float top_p, float min_p,
                             uint64_t *rng, int max_tokens,
                             int *accepted, int accepted_cap, char *err, size_t errlen);
    /** Speculative generation seeded with a known `first_token` -- the forced-
     * continuation form, where the caller has already chosen the opening token.
     * @param first_token   the opening token, committed as-is
     * @param max_tokens    cap on tokens committed
     * (a stop -- the family's whole set, pulsar_token_is_stop -- ends the block)
     * @param accepted      receives the committed token ids
     * @param accepted_cap  its capacity
     * @param err           failure message buffer
     * @param errlen        its size
     * @return tokens committed. */
    int eval_speculative_block(int first_token, int max_tokens,
                               int *accepted, int accepted_cap, char *err, size_t errlen);
    /** Undo back to `pos` tokens: trim the checkpoint, drop speculative state,
     * and clamp every compressing layer's frontier to pos/ratio.
     *
     * The clamp is unconditional and is what keeps the next step admissible.
     * A second, best-effort half restores the VALUES of re-emitted compressed
     * rows from the projection ring -- only when the ring covers the rewound
     * span, which on the served (multiseq) path it does not. */
    /// @param pos  the new history length, in tokens
    void rewind(int pos);
    /** L264: put the installed bank at its grid checkpoint G (pulsar_ckpt_restore)
     * and trim the host history to match.  The next sync evaluates from G, which
     * is a prefill grid point, so the result is the cold prefill's byte for byte.
     * False -- and nothing changed -- when the bank holds no checkpoint at G or the
     * history is shorter than G. */
    /// @param G  the grid position (tokens) whose checkpoint to restore
    /// @return true when restored; false with nothing changed.
    bool restore_checkpoint(uint32_t G);
    /** The host half shared by rewind and restore_checkpoint: the history ends at
     * pos, so everything that described positions above it goes. */
    /// @param pos  the new history length, in tokens
    void trim_history(int pos);
    /** Committed token count for the current bank. */
    /// @return the history length, in tokens.
    int pos();
    /** Allocated context length, in tokens. */
    /// @return ctx_size.
    int ctx();
    /** Borrowed view of the committed token history. Do not free. */
    /// @return the view; valid until the history next changes.
    const pulsar_tokens *tokens();
    /** Serialized size of this session's payload, in bytes. */
    /// @return the byte count save_payload() writes.
    uint64_t payload_bytes();
    /** Write the payload to an open stream. @return 0 on success. */
    /// @param fp      the open output stream
    /// @param err     failure message buffer
    /// @param errlen  its size
    int save_payload(FILE *fp, char *err, size_t errlen);
    /** Read a payload back, replacing this session's state. Refuses a payload
     * whose version or shape does not match this build. @return 0 on success. */
    /// @param fp             the open input stream, at the payload
    /// @param payload_bytes  the payload's length, in bytes
    /// @param err            failure message buffer
    /// @param errlen         its size
    int load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
    /** L264 S4 disk segments (pulsar.h, pulsar_session_save_segment). */
    /// @param G_prev       the segment's first grid position (tokens; may be 0)
    /// @param G            the segment's end grid position
    /// @param state_bytes  out (may be NULL): the trailing state part's bytes, 0 = rows only
    /// @return the bytes save_segment() writes, or 0 when [G_prev, G) is not a segment span.
    uint64_t segment_bytes(uint32_t G_prev, uint32_t G, uint64_t *state_bytes);
    /// Write the segment [G_prev, G) to `fp` (pulsar_session_save_segment's local body).
    /// @param fp      the open output stream
    /// @param G_prev  the segment's first grid position (tokens)
    /// @param G       its end grid position
    /// @param err     failure message buffer
    /// @param errlen  its size
    /// @return 0 on success, 1 on a refusal or I/O failure (err filled).
    int save_segment(FILE *fp, uint32_t G_prev, uint32_t G, char *err, size_t errlen);
    /// Read one segment into the installed bank (pulsar_session_load_segment's local body).
    /// @param fp      the open input stream, at the segment
    /// @param bytes   the segment's length
    /// @param last    the chain's last segment: only it restores the state at its G
    /// @param G_out   out: the segment's end grid position (0 on failure)
    /// @param err     failure message buffer
    /// @param errlen  its size
    /// @return 0 on success, 1 on a refusal or I/O failure (err filled).
    int load_segment(FILE *fp, uint64_t bytes, bool last, uint32_t *G_out, char *err, size_t errlen);
    /** Capture the session into an owned in-memory blob. @return 0 on success. */
    /// @param snap    receives the owned blob (pulsar_session_snapshot_free)
    /// @param err     failure message buffer
    /// @param errlen  its size
    int save_snapshot(pulsar_session_snapshot *snap, char *err, size_t errlen);
    /** Restore the session from a snapshot taken by save_snapshot().
     * @return 0 on success. */
    /// @param snap    a snapshot from save_snapshot()
    /// @param err     failure message buffer
    /// @param errlen  its size
    int load_snapshot(const pulsar_session_snapshot *snap, char *err, size_t errlen);
};

/** ---- helpers shared across the session_*.cpp TUs ----
 * payload_set_err (session_payload.cpp) is the payload/bank-KV error stamper. */
/// @param err     the caller's message buffer (may be NULL)
/// @param errlen  its size
/// @param msg     the message
void payload_set_err(char *err, size_t errlen, const char *msg);
/** A request boundary (session_spec.cpp): the speculative lookahead -- the carry token, the pre-drafted
 *  pendings -- belongs to the previous request's distribution and goes, and the terminal yield quench
 *  re-arms.  The core runs it for EVERY family wherever a request begins or the history the lookahead was
 *  conditioned on is replaced: a sync and an invalidate (pulsar_session_family_sync / _invalidate), a fused
 *  prompt chunk (note_prefilled), a rewind, a payload load.  Before L284 Qwen's sync and invalidate never ran
 *  it, and the per-bank shadow kept a latched quench for every later request on that bank. */
/// @param s  the session (installed bank)
void spec_lookahead_reset(pulsar_session *s);
/** The quench half alone: re-arm the terminal yield quench (the teacher-forced probe re-arms it every round). */
/// @param s  the session (installed bank)
void spec_quench_reset(pulsar_session *s);

/** How one layer's attention reaches beyond its 128-token window (CSA2, L218).
 *
 * WINDOW:  compress_ratio 0; the sliding window is all there is (layers 0-1,
 *          and every drafter layer).
 * FULL:    a kv source.  Runs the compressor over its own input, writes the
 *          shared compressed-KV rows and the index-K rows derived from the same
 *          latent, runs its own indexer and publishes the top-k.
 * REINDEX: an index source that is not a kv source.  Reads its kv source's
 *          compressed KV and index K, scores them with its own indexer weights
 *          (inside the candidate pool when one is published), publishes top-k.
 * REUSE:   neither.  Reads the kv source's rows and the index source's top-k
 *          unchanged; owns nothing beyond its window ring and its q path.
 * FULL_UNINDEXED: a kv source that runs NO indexer -- 0731's ratio-128 (HCA)
 *          layers.  It runs the compressor over its own input and publishes the
 *          compressed-KV rows, but publishes no index K and no top-k, so its
 *          attention reads its own compressed cache unindexed.  The artifact
 *          agrees: such a layer carries attn_compressor_* and no indexer.*
 *          tensor at all.  Reachable only from a profile whose index source set
 *          is narrower than its kv source set (0731: 21 index of 41 kv).
 *
 * Ask the predicates below rather than comparing against PULSAR_ATTN_FULL:
 * a bare comparison silently MISSES the new mode, which skips a layer's
 * compressor without a word. */
typedef enum {
    PULSAR_ATTN_WINDOW = 0,
    PULSAR_ATTN_FULL,
    PULSAR_ATTN_REINDEX,
    PULSAR_ATTN_REUSE,
    PULSAR_ATTN_FULL_UNINDEXED,
} pulsar_attn_mode;

/** Runs the compressor and publishes compressed KV for its group. */
/// @param m  the layer's mode
/// @return true for FULL and FULL_UNINDEXED.
static inline bool pulsar_attn_owns_kv(pulsar_attn_mode m) {
    return m == PULSAR_ATTN_FULL || m == PULSAR_ATTN_FULL_UNINDEXED;
}

/** Runs an indexer: query projection, scoring, and a published top-k. */
/// @param m  the layer's mode
/// @return true for FULL and REINDEX.
static inline bool pulsar_attn_runs_indexer(pulsar_attn_mode m) {
    return m == PULSAR_ATTN_FULL || m == PULSAR_ATTN_REINDEX;
}

/** Depends on an index source's top-k: FULL and REINDEX publish one, REUSE
 * consumes one it did not compute.  FULL_UNINDEXED is the single compressed mode
 * that does not -- it has no indexer, so it has no top-k and no index source to
 * read.
 *
 * This is NOT pulsar_attn_runs_indexer: that asks whether the indexer KERNEL
 * runs, and REUSE -- the mode that reads a top-k it never computed -- answers no
 * to it.  Anything asking "does this layer's behaviour depend on its index
 * source?" wants this one. */
/** The mode's name, for instruments and refusals.  One authority: a test that
 * spells the modes out again is a second list that can drift from this one. */
/// @param m  the mode
/// @return its name (static), "?" for a value outside the enum.
static inline const char *pulsar_attn_mode_name(pulsar_attn_mode m) {
    switch (m) {
    case PULSAR_ATTN_WINDOW:         return "WINDOW";
    case PULSAR_ATTN_FULL:           return "FULL";
    case PULSAR_ATTN_REINDEX:        return "REINDEX";
    case PULSAR_ATTN_REUSE:          return "REUSE";
    case PULSAR_ATTN_FULL_UNINDEXED: return "FULL_UNINDEXED";
    }
    return "?";
}

/// Whether the layer's behaviour depends on an index source's top-k (see the comment above pulsar_attn_mode_name).
/// @param m  the layer's mode
/// @return true for FULL, REINDEX and REUSE.
static inline bool pulsar_attn_reads_index(pulsar_attn_mode m) {
    return m == PULSAR_ATTN_FULL || m == PULSAR_ATTN_REINDEX || m == PULSAR_ATTN_REUSE;
}

/** One layer's row of the attention layout table.  Derived once at load from
 * the artifact's compress_ratios / kv_source_layers / index_source_layers /
 * candidate_source_layer by pulsar_attn_layout_install(), which is the only
 * writer; every cache-ownership and weight-binding decision reads THIS. */
typedef struct {
    uint32_t ratio;         ///< tokens per compressed row; 0 = window only
    uint32_t kv_source;     ///< layer whose compressed KV + index K this layer reads (itself when FULL; PULSAR_NO_LAYER at ratio 0)
    uint32_t index_source;  ///< layer whose top-k this layer attends with (itself when FULL/REINDEX; PULSAR_NO_LAYER at ratio 0)
    /// How this layer's attention reads compressed rows (FULL / REINDEX / REUSE / ...).
    pulsar_attn_mode mode;
    bool candidate_source;  ///< this layer's indexer publishes the candidate block mask
    bool uses_candidates;   ///< this layer's indexer scores only inside the published mask
} pulsar_layer_attn;

/** ---- shared globals ---- */

/** The shape profiles this engine serves: the two are selected at load from the
 * artifact's own metadata.  plans/96-two-profiles-one-engine.md. */
extern const pulsar_shape PULSAR_SHAPE_V4;
/// The DeepSeek V4.1 Flash profile (shape_profiles.cpp).
extern const pulsar_shape PULSAR_SHAPE_V41;
/// The loaded model's profile, set once at load (pulsar_select_shape_from_metadata); every PULSAR_N_* macro reads it.
extern pulsar_shape g_pulsar_shape;
/** REAP ds4-compact-v1: per-layer count of physically-present routed experts.
 * 0 means "not set" -> falls back to n_expert (the un-pruned default). The
 * router/bias tensors stay padded to n_expert (256); only the expert weight
 * tensors are dense-trimmed to this count. Read from reap.layer.keep_count. */
extern uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER];
/// The process's instance-lock file descriptor, -1 when not held (engine_api.cpp acquire / release).
extern int g_pulsar_lock_fd;

/** ---- shared functions ---- */

/// Whether a backend runs the GPU graph.
/// @param backend  the backend
/// @return true for PULSAR_BACKEND_CUDA.
bool pulsar_backend_uses_graph(pulsar_backend backend);
void pulsar_die(const char *msg);
/** Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape.
 */
/// @param il  the layer, < PULSAR_N_LAYER (else dies)
/// @return the layer's compression ratio (tokens per compressed row; 0 = window only).
uint32_t pulsar_layer_compress_ratio(uint32_t il);
/** The layer's row of the CSA2 attention layout table (ratio, sources, mode). */
/// @param il  the layer
/// @return the row (static storage).
const pulsar_layer_attn *pulsar_layer_attn_layout(uint32_t il);
/** Build the attention layout table from a per-layer ratio array and the
 * source sets, asserting the shape profile's expectation and the CSA2
 * invariants (sources ascending and inside the backbone, a layer's ratio equals
 * the ratio of every source it READS, the candidate source is an index source).
 * The loader calls it with the artifact's metadata; a unit test with the
 * profile's own sets.  Dies on any violation. */
/// @param ratios            per-layer compression ratios, PULSAR_N_LAYER entries
/// @param kv_sources        the kv-source layers
/// @param n_kv              entries in kv_sources
/// @param index_sources     the index-source layers
/// @param n_index           entries in index_sources
/// @param candidate_source  the candidate-source layer, -1 = none
void pulsar_attn_layout_install(const uint32_t *ratios,
                                const uint32_t *kv_sources, uint32_t n_kv,
                                const uint32_t *index_sources, uint32_t n_index,
                                int32_t candidate_source);
/** A LAYER-SUBSET FIXTURE (L269 W2, model_layout.cpp): the artifact keeps `n`
 * ascending layers of the profile just selected (`pulsar.fixture.source_layers`),
 * renumbered densely.  Cuts g_pulsar_shape to them (n_layer, the source sets and
 * the candidate in block indices) and announces it; n == 0 is the identity.
 * Returns false (message printed) on a malformed map. */
/// @param source_layers  the kept profile layers, ascending
/// @param n              entries (0: the identity)
/// @return false (said) on a malformed map.
bool pulsar_layer_subset_install(const uint32_t *source_layers, uint32_t n);
/** The profile's layer that block `il` of the loaded artifact is: il itself, or
 * the fixture's map.  Everything a layer index means in the source model (its
 * ratio, its Engram row table) is asked of this. */
/// @param il  the artifact's block index
/// @return the profile layer index.
uint32_t pulsar_layer_source(uint32_t il);
/** True when the loaded artifact is a layer-subset fixture. */
/// @return true for a layer-subset fixture.
bool pulsar_layer_subset(void);

/** CSA2 (L218) ownership.  The compressed pool, the index-K pool, the
 * compressor state and the frontier a layer reads all live at its kv SOURCE's
 * index; a layer that is its own source (mode FULL) writes them.  Every
 * per-layer cache array in pulsar_gpu_graph / pulsar_bank_slabs is indexed by
 * the source, so a consumer resolves through these and never through `il`. */
/* pulsar_kv_row_kind is declared in src/pulsar_gpu.h (see above). */

/** The row geometry of the LOADED model, in bytes, asked by KIND.
 *
 * CSA2 (V4.1): RING 528 (WINDOW), COMP 288 (MAIN), INDEX 68.
 * UNIFIED (0731): one 384 B NVFP4 row in every non-index buffer; INDEX 68.
 *
 * plans/96-two-profiles-one-engine.md s10. */
/// @param kind  RING, COMP or INDEX
/// @return the row's bytes under the loaded profile.
static inline uint64_t pulsar_kv_row_bytes(pulsar_kv_row_kind kind) {
    /* The index-K row is the same FP4 row in both families. */
    if (kind == PULSAR_KV_ROW_INDEX) {
        return (uint64_t)PULSAR_MXKV_FP4_ROWBYTES((uint64_t)PULSAR_N_INDEXER_HEAD_DIM);
    }
    if (g_pulsar_shape.kv_row_style == PULSAR_KV_ROWS_UNIFIED) {
        return (uint64_t)PULSAR_ATTN_PACK_ROWBYTES((uint64_t)PULSAR_N_HEAD_DIM);
    }
    return kind == PULSAR_KV_ROW_RING
               ? (uint64_t)PULSAR_WINKV_ROWBYTES((uint64_t)PULSAR_N_HEAD_DIM)
               : (uint64_t)PULSAR_MAINKV_ROWBYTES((uint64_t)PULSAR_N_HEAD_DIM);
}

/* The L218 row-format refactor replaced these two macros with
 * pulsar_kv_row_bytes() above, because the row family became profile-dependent --
 * the right shape, since UNIFIED (0731) and CSA2 (V4.1) no longer share a width.
 * But 19 call sites across tests/ still NAME the old macros, so the test runner
 * stopped building and took every gate that depends on it down with it.
 *
 * Keep the names working with the semantics the engine itself uses for the same
 * two rows (session_banks.cpp:350): the attention row is the COMP row and the
 * index row is the FP4 index row.  A caller that needs the raw RING row should
 * call pulsar_kv_row_bytes(PULSAR_KV_ROW_RING) directly. */
/// Legacy name: bytes of a compressed attention (COMP) KV row.
#define PULSAR_ENGINE_MAINKV_ROWBYTES  (pulsar_kv_row_bytes(PULSAR_KV_ROW_COMP))
/// Legacy name: bytes of an FP4 index-key (INDEX) row.
#define PULSAR_ENGINE_IDXFP4_ROWBYTES  (pulsar_kv_row_bytes(PULSAR_KV_ROW_INDEX))
/// Legacy name: bytes of a raw sliding-window (RING) KV row.
#define PULSAR_ENGINE_WINKV_ROWBYTES   (pulsar_kv_row_bytes(PULSAR_KV_ROW_RING))

/// The layer whose caches layer `il` reads (see the CSA2 ownership comment above pulsar_kv_row_bytes).
/// @param il  the layer
/// @return its kv source (PULSAR_NO_LAYER at ratio 0).
static inline uint32_t gpu_graph_kv_source(uint32_t il) {    return pulsar_layer_attn_layout(il)->kv_source;
}
/// @param il  the layer
/// @return true when the layer compresses and publishes its own KV (FULL / FULL_UNINDEXED).
static inline bool gpu_graph_layer_is_kv_source(uint32_t il) {
    return pulsar_attn_owns_kv(pulsar_layer_attn_layout(il)->mode);
}
/** A kv source of ratio > 1 keeps a pending group in the state lane; ratio 1
 * emits a row per token and keeps none. */
/// @param il  the layer
/// @return true for a kv source of ratio > 1.
static inline bool gpu_graph_layer_has_comp_state(uint32_t il) {
    const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
    return pulsar_attn_owns_kv(a->mode) && a->ratio > 1u;
}
/** A layer owns an index-K pool exactly when it is a kv source that RUNS an
 * indexer.  0731's ratio-128 HCA layers are kv sources that run none, so they
 * publish no index pool, own no index base table and store no index rows -- the
 * single authority every index-pool consumer asks (the allocator, the sizing
 * estimate, the bank snapshot). */
/// @param il  the layer
/// @return true for a kv source that runs an indexer.
static inline bool gpu_graph_layer_has_index_pool(uint32_t il) {
    const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
    return pulsar_attn_owns_kv(a->mode) && pulsar_attn_runs_indexer(a->mode);
}
/** Physically-present routed-expert count for a layer. For an un-pruned model
 * (or any layer whose keep_count was not set) this is the full n_expert; for a
 * REAP ds4-compact-v1 model the pruned layers report their dense survivor
 * count. Only the expert *weight* tensors are trimmed to this; the router and
 * bias stay padded to n_expert.
 */
/// @param il  the layer
/// @return the expert count present in its stacks.
uint32_t pulsar_layer_n_expert(uint32_t il);
/// Print "pulsar: <what> '<path>': strerror(errno)" and exit(1).
/// @param what  the failed operation
/// @param path  the file it was on
void pulsar_die_errno(const char *what, const char *path);
/// @param s  a borrowed string view
/// @param z  a NUL-terminated string
/// @return true when the bytes are equal.
bool pulsar_streq(pulsar_str s, const char *z);
/// @param a  a string view
/// @param b  another
/// @return true when the bytes are equal.
bool pulsar_str_eq(pulsar_str a, pulsar_str b);
/// FNV-1a 64 over `len` bytes.
/// @param ptr  the bytes
/// @param len  their count
/// @return the hash.
uint64_t hash_bytes(const void *ptr, uint64_t len);
/// calloc that dies on failure.
/// @param n     elements
/// @param size  bytes each
/// @return the zeroed allocation (free with free()).
void *xcalloc(size_t n, size_t size);
/// malloc that dies on failure.
/// @param size  bytes
/// @return the allocation (free with free()).
void *xmalloc(size_t size);
/// strdup through xmalloc (dies on failure).
/// @param s  the string
/// @return the owned copy (free with free()).
char *pulsar_strdup(const char *s);
/// realloc that dies on failure.
/// @param ptr   the block (may be NULL)
/// @param size  the new size, bytes
/// @return the reallocated block.
void *xrealloc(void *ptr, size_t size);
/// @return CLOCK_MONOTONIC time, in seconds.
double now_sec(void);

/// Fold one step into a lane's exponentially weighted cost fit and refit.
/// @param f     the fit (NULL is a no-op)
/// @param rows  rows the step's forward carried (0 is ignored)
/// @param ms    the step's wall time (non-positive is ignored)
void lane_cost_fit_observe(pulsar_lane_cost_fit *f, uint32_t rows, double ms);
/** L301: the verify curve (verify_curve.cpp): a curve over the profiled forward base_ms[1..rows_max] (NULL past
 *  the landing buffers), its release; the pick over survivals sorted descending -- the drafts a round of n base
 *  rows verifies within cap_rows, `extra_ms` added to every candidate round, *ratio its tokens per ms (c NULL: as
 *  many as the rows hold, ratio 0); and the lockstep chains' continuation on it (pulsar_spec_chain_step's rule with
 *  the drafter's step cost and the row cap given).  The engine-free forms the scheduler tests drive. */
/// @param rows_max  rows profiled, 1 .. PULSAR_SPEC_LOGITS_ROWS
/// @param base_ms   base_ms[1..rows_max], the forward's ms at each row count (copied)
/// @return the curve (free with pulsar_verify_curve_free), or NULL for a bad rows_max.
struct pulsar_verify_curve *pulsar_verify_curve_new(uint32_t rows_max, const double *base_ms);
/// Free a curve (NULL is fine).
/// @param c  the curve
void pulsar_verify_curve_free(struct pulsar_verify_curve *c);
/// The drafts a round of `n` base rows verifies (see the comment above pulsar_verify_curve_new).
/// @param c         the curve, or NULL (as many as the rows hold)
/// @param n         base rows
/// @param surv      m draft survival probabilities, sorted descending
/// @param m         drafts offered
/// @param cap_rows  the row cap (clamped to the curve's rows_max)
/// @param extra_ms  ms added to every candidate round
/// @param ratio     out: the pick's tokens per ms (0 without a curve)
/// @return drafts to verify.
uint32_t pulsar_verify_curve_pick(const struct pulsar_verify_curve *c, uint32_t n, const float *surv, uint32_t m,
                                  uint32_t cap_rows, double extra_ms, double *ratio);
/// The lockstep chains' continuation on the curve (pulsar_spec_chain_step's rule).
/// @param c          the curve
/// @param step       the drafter's step cost
/// @param cap        the row cap
/// @param n_chains   chains (each holds a base row)
/// @param drafted    survivals of the drafts already made
/// @param n_drafted  entries in `drafted`
/// @param run        each live chain's survival so far
/// @param n_live     live chains
/// @param spent_ms   drafting time already spent this round
/// @param go         out: whether another lockstep step pays
/// @return the drafts the verify should carry.
uint32_t pulsar_verify_chain_pick(const struct pulsar_verify_curve *c, const pulsar_lane_cost *step, uint32_t cap,
                                  uint32_t n_chains, const float *drafted, uint32_t n_drafted, const float *run,
                                  uint32_t n_live, double spent_ms, bool *go);
/** L327: a solo prefill chunk of q rows over the c already computed took `ms` (the prefill loop's own report; the
 *  fused chunks come through pulsar_engine_fused_cost_observe). */
/// @param e   the engine (a TP pair's ranks fit nothing)
/// @param q   the chunk's rows, in tokens
/// @param c   context before it, in tokens
/// @param ms  its wall time
void pulsar_engine_prefill_chunk_observe(pulsar_engine *e, uint32_t q, uint32_t c, double ms);
/** L327: the fit's persistence beside the verify curve -- the file a fit of `key` is kept in under `cache_dir`, read
 *  into *f (false, *f untouched, when absent, of another key or version, or unreadable) and written (tmp + rename;
 *  said once when it cannot be).  The engine-free forms the tests drive. */
/// @param cache_dir  the verify curve's cache directory
/// @param key        the curve's key
/// @param out        receives `<cache_dir>/<key hex>.fit`
/// @param cap        its size
/// @return false without a directory or when the path does not fit.
bool pulsar_prefill_fit_path(const char *cache_dir, uint64_t key, char *out, size_t cap);
/// @param path  the fit file
/// @param key   the key it must carry
/// @param f     receives the fit
/// @return true when read.
bool pulsar_prefill_fit_read(const char *path, uint64_t key, pulsar_prefill_cost_fit *f);
/// @param path  the fit file
/// @param key   the key it carries
/// @param f     the fit
/// @return true when written.
bool pulsar_prefill_fit_write(const char *path, uint64_t key, const pulsar_prefill_cost_fit *f);
/** The read-only view of one fit (its terms, count and weighted centre). */
/// @param f  the fit
/// @return the view.
pulsar_lane_cost lane_cost_fit_view(const pulsar_lane_cost_fit *f);
/** A TP worker takes the leader's terms as they rode the wire. */
/// @param e        the worker's engine (NULL is a no-op)
/// @param flat_us  the leader's flat term, us
/// @param row_us   its per-row term, us
/// @param valid    the leader's fit is valid (also needs both terms > 0)
void pulsar_engine_spec_cost_set(pulsar_engine *e, int32_t flat_us, int32_t row_us, bool valid);
/// Write n raw floats to `path`.
/// @param path  the file (created / truncated)
/// @param data  the floats
/// @param n     their count
/// @return false (said) on an I/O failure.
bool write_f32_binary_file(const char *path, const float *data, uint64_t n);
/// Read exactly n raw floats from `path`; the file's size must be n * 4.
/// @param path  the file
/// @param data  receives the floats
/// @param n     their count
/// @return false (said) on a size mismatch or I/O failure.
bool read_f32_binary_file(const char *path, float *data, uint64_t n);
/// Read n bytes at the cursor and advance it (pulsar::Cursor::read).
/// @param c    the cursor
/// @param dst  receives the bytes
/// @param n    bytes
/// @return false past the blob's end.
bool cursor_read(pulsar_cursor *c, void *dst, uint64_t n);
/// Read a little-endian u32 and advance.
/// @param c  the cursor
/// @param v  receives the value
/// @return false past the blob's end.
bool cursor_u32(pulsar_cursor *c, uint32_t *v);
/// Read a little-endian u64 and advance.
/// @param c  the cursor
/// @param v  receives the value
/// @return false past the blob's end.
bool cursor_u64(pulsar_cursor *c, uint64_t *v);
/// Read a length-prefixed string (borrowed view into the blob) and advance.
/// @param c  the cursor
/// @param s  receives the view
/// @return false past the blob's end.
bool cursor_string(pulsar_cursor *c, pulsar_str *s);
/// @param value      the value
/// @param alignment  the alignment (> 0)
/// @return `value` rounded up to a multiple of `alignment`.
uint64_t align_up(uint64_t value, uint64_t alignment);
/** The name of a tensor LAYOUT (PULSAR_TENSOR_*), for messages.  It is the same
 * string a container declares, so a diagnostic and a declaration cannot drift. */
/// @param type  a PULSAR_TENSOR_* id
/// @return its name (static), "unknown" for an id outside the table.
const char *tensor_type_name(uint32_t type);
/** The layout id a container's declared name denotes, or -1 if the name is not
 * a layout this engine reads.  The ONE name -> id authority: the safetensors
 * reader resolves every declaration through this rather than restating it. */
/// @param name  a declared layout name
/// @return the PULSAR_TENSOR_* id, or -1.
int tensor_type_from_name(const char *name);
/// One expert's byte layout in a PULSAR_TENSOR_CUTLASS_MXFP4 stack (expert-major).
/// @param k           input width
/// @param n           output width
/// @param data_bytes  out: packed FP4 data, n * k / 2
/// @param sf_bytes    out: scale factors, over k and n padded to 128
/// @param stride      out: bytes per expert (data + scales)
void cutlass_mxfp4_expert_layout(uint64_t k, uint64_t n,
                                  uint64_t *data_bytes, uint64_t *sf_bytes,
                                  uint64_t *stride);
/// A cursor at `pos` in the model's metadata value blob (the synthesized one for safetensors, else the mapping).
/// @param m    the model
/// @param pos  byte offset
/// @return the cursor.
pulsar_cursor cursor_at(const pulsar_model *m, uint64_t pos);
/// Read metadata `key` as a UINT32.
/// @param m    the model
/// @param key  the key
/// @param out  receives the value
/// @return false when absent or of another type.
bool model_get_u32(const pulsar_model *m, const char *key, uint32_t *out);
/// Read metadata `key` as a STRING (a view into the blob).
/// @param m    the model
/// @param key  the key
/// @param out  receives the view
/// @return false when absent or of another type.
bool model_get_string(const pulsar_model *m, const char *key, pulsar_str *out);
/// Read metadata `key` as a UINT64, accepting a UINT32.
/// @param m    the model
/// @param key  the key
/// @param out  receives the value
/// @return false when absent or of another type.
bool model_get_u64_compat(const pulsar_model *m, const char *key, uint64_t *out);
/// Read metadata `key` as a FLOAT32, accepting a FLOAT64.
/// @param m    the model
/// @param key  the key
/// @param out  receives the value
/// @return false when absent or of another type.
bool model_get_f32_compat(const pulsar_model *m, const char *key, float *out);
/// Read metadata `key` as a BOOL.
/// @param m    the model
/// @param key  the key
/// @param out  receives the value
/// @return false when absent or of another type.
bool model_get_bool(const pulsar_model *m, const char *key, bool *out);
/// Locate metadata array `key`: its element type, length and data offset (left unparsed).
/// @param m    the model
/// @param key  the key
/// @param out  receives the reference
/// @return false when absent or not an array.
bool model_get_array(const pulsar_model *m, const char *key, pulsar_array_ref *out);
/** Give the checkpoint's host pages back after load: MADV_DONTNEED on every
 * shard mapping (the mapping stays valid -- a later host read refaults the page
 * from disk) and POSIX_FADV_DONTNEED on its file (drops the page cache).  Only
 * when the device reads no weight through the host mapping
 * (pulsar_gpu_model_reads_host_pages) and the graph backend owns the weights;
 * the caller decides.  @return the resident bytes released. */
/// @param m  the model (NULL returns 0)
uint64_t pulsar_model_release_host_pages(pulsar_model *m);
/// Unmap and close everything the model mapped and free its directory.  NULL is a no-op.
/// @param m  the model
void model_close(pulsar_model *m);
/** Open and map the GGUF once.  The GPU path needs a shared mapping for
 * no-copy GPU buffers; tokenizer/inspection opens use a private read-only
 * mapping instead.
 */
/** false = refused (said and counted); model_close releases whatever the refused open mapped. */
/// @param m            the model to fill
/// @param path         the GGUF file (a directory goes to safetensors_open)
/// @param gpu_mapping  map shared for the GPU path; false = private read-only (tokenizer / inspect)
/// @return true when open.
bool model_open(pulsar_model *m, const char *path, bool gpu_mapping);

/** Open a safetensors checkpoint DIRECTORY: mmap every shard, re-encode the
 * JSON __metadata__ into the one GGUF-typed KV blob the rest of the engine
 * already reads, and synthesize the same pulsar_tensor directory the GGUF path
 * builds -- including ONE stacked tensor per routed-expert projection, because
 * a projection's per-expert tensors are contiguous and gap-free.  That is why
 * nothing downstream of model_open changes. */
/** false = refused (said and counted: pulsar_load_refuse); the model holds what was mapped -- model_close it. */
/// @param m            the model to fill
/// @param path         the checkpoint directory
/// @param gpu_mapping  as model_open's
/// @return true when open.
bool safetensors_open(pulsar_model *m, const char *path, bool gpu_mapping);
/// Print the model's metadata summary (name, architecture, shape) to stderr.
/// @param m  the model
void model_summary(const pulsar_model *m);
/// Find a tensor by its canonical name.
/// @param m     the model
/// @param name  the name
/// @return the directory entry (owned by the model), or NULL.
pulsar_tensor *model_find_tensor(const pulsar_model *m, const char *name);
/** L299: every tensor whose name starts with `prefix` is declared NOT LOADED (pulsar_model::unloaded: never bound,
 *  staged or priced) -- the drafter an open did not select, in a container that binds it by name (Qwen's mtp.*;
 *  role-recorded blocks use pulsar_role_binder_unload).  Returns the tensors so marked. */
/// @param m       the model
/// @param prefix  the name prefix
/// @return the tensors marked.
uint64_t model_unload_prefix(pulsar_model *m, const char *prefix);
/// Stage the model's tensors to the device (merged spans of the mapping) and register its MXFP8 weights.
/// @param backend       the backend (not CUDA: nothing to do, true)
/// @param m             the model
/// @param span_offsets  [span_count] filter starts
/// @param span_sizes    [span_count] filter lengths
/// @param span_count    filter spans (0 = every tensor)
/// @param skip_prefix   tensors with this name prefix are skipped, or NULL
/// @return false (said) on a bad model, an out-of-range tensor or a failed device copy.
bool accelerator_cache_model_tensors(pulsar_backend backend,
                                            const pulsar_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count,
                                            const char *skip_prefix);
/** L241 4g-2: bytes of stored routed-expert stacks this rank does NOT stage --
 * every `*_exps.weight` stack under TP, where each rank serves its half of every
 * expert from per-rank stacks built at open; 0 when the pair is off.  The
 * admission budget subtracts it from mapped_bytes. */
/// @param m  the model (NULL / one rank: 0)
/// @return the bytes.
uint64_t pulsar_model_unstaged_expert_bytes(const pulsar_model *m);
/** Return the in-place tensor payload inside the mapped GGUF (or inside the
 * overlay file's mapping for --expert-overlay swapped tensors).
 */
/// @param m  the model
/// @param t  the tensor
/// @return its first payload byte in the owning mapping (borrowed).
const void *tensor_data(const pulsar_model *m, const pulsar_tensor *t);
/// --expert-overlay: repoint every 3-D `*_exps.*` tensor of `base` under `prefix` at the overlay's same-named tensor.
/// @param base     the model whose directory entries are swapped
/// @param overlay  the overlay model (must outlive `base`'s use of it)
/// @param prefix   the name prefix the swap applies under
/// @return tensors swapped; 0 on a missing or mismatched overlay tensor (refused and counted).
uint32_t model_apply_expert_overlay(pulsar_model *base, const pulsar_model *overlay,
                                    const char *prefix);
/// Pre-populate device access for the overlay-swapped tensors' spans.
/// @param backend  the backend (not CUDA: nothing to do, true)
/// @param base     the model whose entries were swapped
/// @param overlay  the overlay
/// @return false when a span could not be cached.
bool accelerator_prepare_expert_overlay(pulsar_backend backend,
                                        const pulsar_model *base,
                                        const pulsar_model *overlay);

/** Mapping that owns a tensor's payload: the overlay file's map for
 * --expert-overlay swapped tensors, the base model's map otherwise. */
/* TOTAL on a NULL tensor, deliberately: several call sites pass an OPTIONAL
 * tensor (the router bias on a hash-routed layer, the image-token bias) whose
 * offset they likewise guard with a `? : 0`, so a null check here is what keeps
 * the pair (map, offset) safe to compute unconditionally.  Dereferencing a NULL
 * `t` here is a segfault on layer 0 of any artifact that omits the bias, which
 * is every shipped one. */
/// @param m  the model
/// @param t  the tensor, or NULL
/// @return the overlay's map for a swapped tensor, else m->map.
static inline const void *tensor_map_base(const pulsar_model *m, const pulsar_tensor *t) {
    return (t && t->ext_map) ? (const void *)t->ext_map : (const void *)m->map;
}
/// The length of tensor_map_base()'s mapping (NULL-tolerant the same way).
/// @param m  the model
/// @param t  the tensor, or NULL
/// @return bytes.
static inline uint64_t tensor_map_size(const pulsar_model *m, const pulsar_tensor *t) {
    return (t && t->ext_map) ? t->ext_size : m->size;
}
/// Read a REQUIRED UINT32 metadata key.
/// @param m    the model
/// @param key  the key
/// @return the value; 0 when missing (said and counted as a load refusal).
uint32_t required_u32(const pulsar_model *m, const char *key);
/// Bytes of one QK_K-blocked row of a routed-expert stack (dies when dim[0] is not a multiple of QK_K).
/// @param t  the stack
/// @return the row's bytes.
PULSAR_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const pulsar_tensor *t);
/** One routed side's byte model from (type, in, out): per-expert stride and the
 *  consumer's "row bytes" (row stride, or the plane split point for the
 *  CUTLASS and EXL3 layouts).  False for a type that is not a routed-expert
 *  layout or a shape the layout refuses. */
/// @param type          the stack's PULSAR_TENSOR_* type
/// @param k             input width
/// @param n             output width
/// @param expert_bytes  out: per-expert stride
/// @param row_bytes     out: the consumer's row bytes
/// @return false for a type / shape the layout refuses.
bool routed_expert_side_layout(uint32_t type, uint64_t k, uint64_t n,
                               uint64_t *expert_bytes, uint64_t *row_bytes);
/// @param w  the weights (NULL: false)
/// @return true when both output_norm and output are bound.
bool weights_have_output_head(const pulsar_weights *w);
/// @param w  the weights (NULL: NULL)
/// @return the first layer with its required tensors bound, or NULL.
const pulsar_layer_weights *weights_first_bound_layer(const pulsar_weights *w);
/** Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp.
 */
/// @param m  the model
void config_validate_model(const pulsar_model *m);
/** Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers.
 */
/// @param w  the weights to fill
/// @param m  the model
void weights_bind(pulsar_weights *w, const pulsar_model *m);
/// Bind the DSpark drafter's tensors (after the whole-artifact scans); a refusal is counted (pulsar_load_refuse).
/// @param w  the drafter weights to fill (zeroed first)
/// @param m  the drafter's model
void dspark_weights_bind(pulsar_dspark_weights *w, const pulsar_model *m);
/** The whole-artifact scans EVERY family's bind runs (L272 B7): refuse a tensor type no reader
 * takes, an E8M0 scale byte of 0xFF (NaN) and a non-finite EXL3 scale. */
/// @param m  the model
void weights_reject_unsupported_types(const pulsar_model *m);
/// The E8M0 half of the scans above: refuse an MX scale byte of 0xFF (NaN) in any tensor.
/// @param m  the model
void weights_reject_bad_e8m0(const pulsar_model *m);

/* ---- L272 P0: the session readers both families share (engine_api.cpp) ------------------------
 * The one answer each, over the family's bank ops or DeepSeek's graph pool: a family's sync and the
 * C API read the same store, the same live bank and the same resume rule. */
/** The session's grid checkpoint store: the family's own (L266, Qwen) or DeepSeek's graph pool's. */
/// @param s  the session
/// @return the store (borrowed), or NULL.
struct pulsar_ckpt_store *pulsar_session_kv_store(pulsar_session *s);
/** The bank the session's sync and eval run on. */
/// @param s  the session
/// @return the live bank (the family's bank ops).
uint32_t pulsar_session_live_bank(pulsar_session *s);
/** L266 step 5: where a sync of `prompt_len` tokens sharing `common` with `bank`'s history resumes --
 * the deepest grid checkpoint within the shared prefix, the bank's prefill-only history and one
 * token short of the prompt (the last row must be evaluated for the logits); 0 = prefill from 0.
 * L289 / L311: or a deeper record of the bank's within the prefix and the prefill-only history -- at the
 * prompt's end with its logits (every family), or below it on a split-invariant family. */
/// @param s           the session
/// @param bank        the bank
/// @param common      tokens the prompt shares with the bank's history
/// @param prompt_len  the prompt's length, in tokens
/// @return the resume position, in tokens (0 = from the start).
uint32_t pulsar_session_resume_point(pulsar_session *s, uint32_t bank, int common, int prompt_len);
/** L311: the record a resume at T (pulsar_session_resume_point's answer) restores -- NULL when T, below the prompt's
 *  end, is a grid checkpoint of the bank's (or 0, or no record of its is at T). */
/// @param s           the session
/// @param bank        the bank
/// @param T           the resume position
/// @param prompt_len  the prompt's length, in tokens
/// @return the record (borrowed from the store), or NULL.
const pulsar_ckpt_record *pulsar_session_resume_record(pulsar_session *s, uint32_t bank, uint32_t T, int prompt_len);
/** L284: whether a sync continues `bank`'s history where its first `len` tokens end, every family's one rule:
 * those tokens are all prefill rows (none a decode step's), the cut is one the model's prefill reproduces
 * (anywhere when pulsar_kv_state_ops::split_invariant, else on the resume grid) and the bank's state is not
 * stale.  Otherwise the sync resumes from pulsar_session_resume_point. */
/// @param s     the session
/// @param bank  the bank
/// @param len   tokens of the bank's history the sync keeps
/// @return true when the sync can continue there.
bool pulsar_session_bank_continues(pulsar_session *s, uint32_t bank, int len);
/** L188: the id check every eval runs before the embed kernel can clamp a refused sample (-1) to
 * token 0.  false with `err` filled when `token` is not a vocab id. */
/// @param s       the session
/// @param token   the token about to be evaluated
/// @param err     failure message buffer
/// @param errlen  its size
/// @return true for an id in [0, the tokenizer table length).
bool pulsar_session_token_is_id(const pulsar_session *s, int token, char *err, size_t errlen);
/** Tokens `t` (NULL = none) and `p` share from the start. */
/// @param t  one token array, or NULL
/// @param p  another, or NULL
/// @return the shared prefix length, in tokens (0 when either is NULL).
static inline int pulsar_tokens_common_prefix(const pulsar_tokens *t, const pulsar_tokens *p) {
    if (!t || !p) return 0;
    const int n = t->len < p->len ? t->len : p->len;
    int i = 0;
    while (i < n && t->v[i] == p->v[i]) i++;
    return i;
}
/** Bind + layout-validate the Vision-Exp tower.  Returns false (and leaves the
 * struct zeroed) when the artifact carries no `vision.patch_embed.proj.weight`,
 * so a text-only artifact is not an error; a PRESENT tower with any wrong dims,
 * type or missing tensor refuses loudly. */
/// @param w  the tower weights to fill
/// @param m  the model
/// @return true when a tower is bound.
bool vision_weights_bind(pulsar_vision_weights *w, const pulsar_model *m);

/** The families' tokenizer tables (L272 P2: tokenizer.cpp, tokenizer_qwen.cpp). */
extern const pulsar_family_tokenizer k_ds4_tokenizer;
/** DeepSeek's bank pool: its graph's (session_banks.cpp). */
extern const pulsar_family_bank_ops k_ds4_bank_ops;
/// The Qwen4-exp family's tokenizer table (tokenizer_qwen.cpp).
extern const pulsar_family_tokenizer k_qwen_tokenizer;
/** One token's bytes for --dump-tokens: UTF-8 verbatim, the usual escapes, other bytes as backslash-x-NN. */
/// @param fp  the stream
/// @param s   the token's bytes
/// @param n   their count
void pulsar_dump_piece_quoted(FILE *fp, const char *s, size_t n);

/** L272: the loader's one failure policy (family.cpp): count a refusal (said by the caller where it was
 *  found); the family's load checks the count at each stage boundary and fails cleanly. */
void pulsar_load_refuse(void);
/// @return refusals counted since the last reset.
uint32_t pulsar_load_refusals(void);
/// Zero the refusal count (a family's load starts with it).
void pulsar_load_refusals_reset(void);

/** L272 P4b: the tensor-parallel plan (tp_slice.cpp).  A family declares its rank's slices (pulsar_family::tp_slices)
 *  as a tensor, an axis and ranges; the core chooses the operation by the tensor's format, derives the residency
 *  rule from it, and runs it after the GPU is up -- or, in record mode, writes one canonical line per slice to `f`
 *  instead (tests/tp_plan_test.cpp). */
typedef enum {
    PULSAR_TP_AXIS_OUT = 0,       ///< a linear's output rows (dim[1]); a plain tensor's outermost dim
    PULSAR_TP_AXIS_IN = 1,        ///< a linear's input (dim[0]): the rank's output is a partial the all-reduce sums
    PULSAR_TP_AXIS_EXPERTS = 2,   ///< whole experts of a stack (dim[2])
} pulsar_tp_axis;
/** The operation a slice takes (tp_slice.cpp plan_op: by the tensor's format, the axis and the family's act_kind). */
typedef enum {
    PULSAR_TP_OP_NONE = 0,
    PULSAR_TP_OP_FP8_ROWS = 1u << 0,
    PULSAR_TP_OP_FP8_K = 1u << 1,
    PULSAR_TP_OP_EXL3_COLS = 1u << 2,
    PULSAR_TP_OP_EXL3_ROWS = 1u << 3,
    PULSAR_TP_OP_VIEW = 1u << 4,
    PULSAR_TP_OP_GATHER = 1u << 5,
    PULSAR_TP_OP_MXFP4_HALF = 1u << 6,
    PULSAR_TP_OP_EXPERTS = 1u << 7,
    PULSAR_TP_OP_EXL3_HALF = 1u << 8,   ///< L269 W1: the EXL3 twin of MXFP4_HALF (a routed stack, slot reader)
} pulsar_tp_op;
/// Most ranges one pulsar_tp_slice carries.
#define PULSAR_TP_SLICE_RANGES 3
/** One slice of a rank's TP plan: a tensor, the axis it is cut along and the kept ranges (pulsar_tp_plan_add). */
typedef struct {
    pulsar_model *m;              ///< the model whose mapping holds the tensor
    /// The tensor sliced (borrowed from `m`'s directory).
    const pulsar_tensor *t;
    /// The axis the ranges index.
    pulsar_tp_axis axis;
    uint32_t n;                   ///< ranges (several only for a gather: Qwen's q | k | v channels)
    /// Each range's first index along `axis`; first n entries valid.
    uint64_t lo[PULSAR_TP_SLICE_RANGES], hi[PULSAR_TP_SLICE_RANGES];
    /** @var pulsar_tp_slice::hi
     *  Each range's end along `axis` (exclusive). */
} pulsar_tp_slice;
/// Declare one slice of the rank's plan (a family's tp_slices op calls this).
/// @param p     the plan
/// @param m     the model whose mapping holds `t`
/// @param t     the tensor
/// @param axis  the axis the ranges index
/// @param n     ranges, 1 .. PULSAR_TP_SLICE_RANGES
/// @param lo    n range starts
/// @param hi    n range ends (exclusive)
/// @return false (said) for no tensor or a bad range count.
bool pulsar_tp_plan_add(pulsar_tp_plan *p, pulsar_model *m, const pulsar_tensor *t, pulsar_tp_axis axis, uint32_t n,
                        const uint64_t *lo, const uint64_t *hi);
/// pulsar_tp_plan_add with one range.
/// @param p     the plan
/// @param m     the model whose mapping holds `t`
/// @param t     the tensor
/// @param axis  the axis
/// @param lo    the range start
/// @param hi    the range end (exclusive)
/// @return as pulsar_tp_plan_add.
bool pulsar_tp_plan_add1(pulsar_tp_plan *p, pulsar_model *m, const pulsar_tensor *t, pulsar_tp_axis axis, uint64_t lo,
                         uint64_t hi);
/** At open, after the family's load (before the inspect-only exit): the family declares, the core chooses each
 *  operation and marks what the rank never stages.  Refuses an operation the format has not, or the forward reads not. */
/// @param e  the engine
/// @return false (said) on a refusal.
bool pulsar_tp_plan_build(pulsar_engine *e);
/** After the GPU is up: every slice, in the declared order. */
/// @param e  the engine (no plan: true)
/// @return false (said) when a slice could not be built.
bool pulsar_tp_plan_run(pulsar_engine *e);
/// Free the engine's plan and every device slice it built; detaches it from the models.  No plan: a no-op.
/// @param e  the engine
void pulsar_tp_plan_free(pulsar_engine *e);
/** The device copy of a host-built slice of `t`, or NULL = `t` has none (the forward reads the stored tensor). */
/// @param m  the model
/// @param t  the tensor
/// @return the device pointer, or NULL.
const void *pulsar_tp_built_ptr(const pulsar_model *m, const pulsar_tensor *t);
/** What the rank's plan does to `t`: its operation and first range, and the key its registered slices resolve
 *  under (the engine).  false = the plan does not slice `t` (or there is no plan: one GPU). */
/// @param m    the model
/// @param t    the tensor
/// @param op   out: the operation
/// @param lo   out: the first range's start
/// @param hi   out: the first range's end (exclusive)
/// @param key  out: the key its registered slices resolve under
/// @return true when the plan slices `t`.
bool pulsar_tp_slice_of(const pulsar_model *m, const pulsar_tensor *t, pulsar_tp_op *op, uint64_t *lo, uint64_t *hi,
                        const void **key);
/// Enter record mode: plan builds write one canonical line per slice to `f` instead of building (tests/tp_plan_test.cpp).
/// @param f  the record stream
void pulsar_tp_record_begin(FILE *f);
/// Leave record mode.
void pulsar_tp_record_end(void);

/** L272 P4: a tensor's ROLE in the forward, declared by its family (weight_format.cpp). */
typedef enum {
    PULSAR_ROLE_DENSE = 0,             ///< a linear through the family's dense path
    PULSAR_ROLE_EXPERT_GATE_UP = 1,    ///< a routed expert's gate or up stack (a split pair)
    PULSAR_ROLE_EXPERT_DOWN = 2,       ///< a routed expert's down stack
    PULSAR_ROLE_EXPERT_GATE_UP_FUSED = 3,   ///< one [in -> 2 mid] gate | up stack
    PULSAR_ROLE_SHARED_EXPERT = 4,     ///< a shared expert's projection inside the MoE launcher
} pulsar_weight_role;
/** The activation a linear reads, as its producer emitted it (L272 P4c: the names say how it is REFERENCED -- the
 *  MX slot is the backend's activation cache, keyed by the f32 buffer the producer wrote; rows are a raw pointer). */
typedef enum {
    PULSAR_ACT_SLOT_BF16 = 0,   ///< the slot's bf16 plane (DeepSeek's plain F32 / BF16 weights, cuBLAS)
    PULSAR_ACT_ROWS_BF16 = 1,   ///< raw bf16 rows (Qwen)
    PULSAR_ACT_SLOT_E4M3 = 2,   ///< the slot's E4M3 MX plane (DeepSeek's MXFP8 dense and routed experts)
    PULSAR_ACT_ROWS_F16 = 3,    ///< raw f16 rows (L321: an fp16 tensor-core consumer's operand -- the Turing lane)
    PULSAR_ACT_COUNT = 4,
} pulsar_act_format;
/// Bit mask of one pulsar_act_format `a`, for activation-set arguments.
#define PULSAR_ACTS(a) (1u << (a))
/** L272 P4c: the kernel arm a dense linear takes for (stored format, activation) -- admission (pulsar_format_serves)
 *  and the launcher (linear.cpp) read this one table. */
typedef enum {
    PULSAR_DENSE_ARM_NONE = 0,
    PULSAR_DENSE_ARM_F32_PLANE,    ///< f32 weight (converted once to bf16), the slot's bf16 plane (cuBLAS)
    PULSAR_DENSE_ARM_BF16_PLANE,   ///< bf16 weight, the slot's bf16 plane (cuBLAS)
    PULSAR_DENSE_ARM_MXFP8_SLOT,   ///< mxfp8_lt, the slot's E4M3 (cuBLASLt)
    PULSAR_DENSE_ARM_MXFP8_ROWS,   ///< mxfp8_lt, raw bf16 rows (W8A16 split-K GEMV / MMA)
    PULSAR_DENSE_ARM_EXL3_ROWS,    ///< EXL3 at a dense-arm rate, raw bf16 rows
    PULSAR_DENSE_ARM_BF16_ROWS,    ///< bf16 weight, raw bf16 rows (L295 F7: the tensor-core rows kernel / cuBLASLt)
    PULSAR_DENSE_ARM_NVFP4_ROWS,   ///< nvfp4 weight, raw bf16 rows (L298: the W4A16 rows kernel, every row kind)
    PULSAR_DENSE_ARM_F16_ROWS,     ///< f16 weight, raw bf16 rows (L321: the fp16 tensor-core arm, every row kind, sm_75+)
} pulsar_dense_arm;
/// The dense arm for a stored format at an activation (the table above).
/// @param type  the weight's PULSAR_TENSOR_* type
/// @param act   the activation's format
/// @return the arm, or PULSAR_DENSE_ARM_NONE.
pulsar_dense_arm pulsar_dense_arm_for(uint32_t type, pulsar_act_format act);
/** L272 P4c: the dense linear's front door (linear.cpp).  A weight's device pointer: the rank's TP slice, else the
 *  mapped range (NULL = said). */
/// @param m     the model
/// @param t     the weight
/// @param what  the caller's name for it, for the refusal
/// @return the device pointer, or NULL (said).
const void *pulsar_weight_device_ptr(const pulsar_model *m, const pulsar_tensor *t, const char *what);
/** out [n_tok][row_hi - row_lo] = rows [row_lo, row_hi) of w [in_dim -> dim[1]] times the activation armed in the
 *  backend's MX slot for `x` -- the arm by w's format (E4M3 for mxfp8_lt, the bf16 plane for bf16 / f32). */
/// @param out     the output rows, f32
/// @param m       the model
/// @param w       the weight
/// @param in_dim  the input width
/// @param row_lo  first output row of `w`
/// @param row_hi  one past the last output row
/// @param x       the input the backend's MX slot was armed for
/// @param n_tok   rows
/// @return false (said) when the format has no slot arm or the launch failed.
bool pulsar_linear_slot(pulsar_gpu_tensor *out, const pulsar_model *m, const pulsar_tensor *w, uint64_t in_dim,
                        uint64_t row_lo, uint64_t row_hi, const pulsar_gpu_tensor *x, uint64_t n_tok);
/** The bf16-rows launcher's reference to t (pulsar_rows_linear_launch, and the composite ops that take one): the
 *  arm by t's format, refused by name when the table has none. */
struct pulsar_rows_linear;
/// @param m       the model
/// @param t       the weight
/// @param in      input width
/// @param out     output width
/// @param prompt  a PROMPT chunk (the prefill arms at every row count)
/// @param what    the caller's name for it, for the refusal
/// @param l       receives the reference
/// @return false (said) when t's format has no raw-rows arm.
bool pulsar_linear_rows_ref(const pulsar_model *m, const pulsar_tensor *t, int in, int out, bool prompt,
                            const char *what, struct pulsar_rows_linear *l);
/** L272 P4c: the routed MoE's front door (moe.cpp): the arm for (gate / up format, down format, activation) -- `up`
 *  NULL = a fused gate_up stack in `gate` -- from the format registry. */
typedef enum {
    PULSAR_MOE_ARM_NONE = 0,
    PULSAR_MOE_ARM_SLOT,         ///< the backend's routed dispatcher over the E4M3 slot (CUTLASS MXFP4 / IQ2 / EXL3 / mixed)
    PULSAR_MOE_ARM_ROWS_FUSED,   ///< EXL3 fused gate_up + down over raw bf16 rows
    PULSAR_MOE_ARM_ROWS_PAIR,    ///< EXL3 gate + up pair + down over raw bf16 rows
    PULSAR_MOE_ARM_ROWS_BF16,    ///< BF16 gate + up + down stacks over raw bf16 rows (L295 F7)
    PULSAR_MOE_ARM_ROWS_UP,      ///< EXL3 up (no gate) + down over raw bf16 rows: ungated relu^2 experts (L296)
} pulsar_moe_arm;
/// @param gate  the gate stack (or fused gate_up)
/// @param up    the up stack, or NULL for a fused gate_up
/// @param down  the down stack
/// @param act   the activation's format
/// @return the arm, or PULSAR_MOE_ARM_NONE.
pulsar_moe_arm pulsar_moe_arm_for(const pulsar_tensor *gate, const pulsar_tensor *up, const pulsar_tensor *down,
                                  pulsar_act_format act);
/** L296 N4: the arm of UP-ONLY experts (pulsar_ffn_act RELU2: an up and a down stack, no gate) at activation `act`:
 *  BF16 up + down at raw bf16 rows (the BF16 routed arm's relu^2, L295 F7) -> ROWS_BF16; EXL3 up + down at raw bf16
 *  rows (the EXL3 routed front door's up-only form, L296) -> ROWS_UP; else NONE.  The one authority the forward's
 *  front door and a family's load both ask. */
/// @param up    the up stack
/// @param down  the down stack
/// @param act   the activation's format
/// @return ROWS_BF16, ROWS_UP or NONE.
pulsar_moe_arm pulsar_moe_arm_up_only(const pulsar_tensor *up, const pulsar_tensor *down, pulsar_act_format act);
/** The routed part over the MX slot: out = the selected experts' weighted SwiGLU FFN of the activation armed for
 *  `x`; the up / mid / experts buffers are the backend's scratch.  A stack the rank's plan halved reads its halves. */
typedef struct {
    /// The routed sum, [n_tokens][out width] f32.
    pulsar_gpu_tensor *out, *up_out, *mid_out, *experts_out;
    /** @var pulsar_moe_slot_call::up_out
     *  Backend scratch: the gate / up projections. */
    /** @var pulsar_moe_slot_call::mid_out
     *  Backend scratch: the SwiGLU mid rows. */
    /** @var pulsar_moe_slot_call::experts_out
     *  Backend scratch: the per-pick expert outputs before the sum. */
    /// The model whose mapping holds the stacks.
    const pulsar_model *m;
    /// The routed gate stack.
    const pulsar_tensor *gate, *up, *down;
    /** @var pulsar_moe_slot_call::up
     *  The routed up stack. */
    /** @var pulsar_moe_slot_call::down
     *  The routed down stack. */
    /// The router's picks, [n_tokens][n_expert_used] expert ids.
    const pulsar_gpu_tensor *selected, *weights;
    /** @var pulsar_moe_slot_call::weights
     *  The picks' route weights, [n_tokens][n_expert_used]. */
    /// Experts present in the stacks.
    uint32_t n_expert_present, n_expert_used;
    /** @var pulsar_moe_slot_call::n_expert_used
     *  Picks per token. */
    /// SwiGLU clamp: gate clamped above, up to [-clamp, clamp]; <= 1e-6 = no clamp.
    float clamp;
    /// The input rows, whose armed E4M3 slot the experts read.
    const pulsar_gpu_tensor *x;
    /// The layer index.
    uint32_t layer, n_tokens;
    /** @var pulsar_moe_slot_call::n_tokens
     *  Rows in the call. */
} pulsar_moe_slot_call;
/// Run the routed part over the MX slot (pulsar_moe_slot_call).
/// @param c  the call
/// @return false (said) when the stacks have no slot arm, a TP slice has none, or the launch failed.
bool pulsar_moe_routed_slot(const pulsar_moe_slot_call *c);
/** The routed part over raw bf16 rows (pulsar_rows_moe_routed_launch): `selected` is localised in place under
 *  expert parallelism (the rank's plan's range of whole experts). */
struct pulsar_tessera_proj;   /* src/cuda/mmq/pulsar_tessera.h (L255) */
struct pulsar_rows_moe_shared; /* src/cuda/pulsar_cuda_qwen.h (task #44) */
/** The routed MoE over raw bf16 rows (pulsar_moe_routed_rows): the stacks or Tessera planes, the picks, the rows and
 *  the workspace; the fields map onto pulsar_rows_moe (src/cuda/pulsar_cuda_qwen.h). */
typedef struct {
    /// The model whose mapping holds the stacks.
    const pulsar_model *m;
    const pulsar_tensor *gate, *up, *down;   ///< up NULL = gate is the fused gate_up stack; gate NULL = up-only (RELU2)
    /** @var pulsar_moe_rows_call::gate
     *  The routed gate stack (or the fused gate_up stack when `up` is NULL; NULL for up-only experts). */
    /** @var pulsar_moe_rows_call::up
     *  The routed up stack (NULL: `gate` is the fused gate_up stack). */
    /** L295 F7: the experts' activation, a pulsar_ffn_act (src/cuda/pulsar_cuda_qwen.h): 0 SWIGLU (gate + up);
     *  RELU2 runs the BF16 arm's up-only stacks (gate NULL) */
    int act;
    /** L296 N4: a LATENT MoE (both or neither): the routed experts run on latent_in's output width --
     *  x_lat = bf16(latent_in x), the routed sum over x_lat (f32), out = latent_out(bf16(that sum)) -- each projection
     *  through the dense front door (pulsar_linear_rows_ref) at the call's row kind.  The buffers are the rows MoE
     *  block's latent parts (pulsar_cuda_qwen.h pulsar_moe_block_parts lat_*): lat_f f32 [n_rows][latent], lat_x /
     *  lat_y bf16 [n_rows][latent], the projections' workspace.  No riding shared expert (it reads the full width). */
    const pulsar_tensor *latent_in, *latent_out;
    /** @var pulsar_moe_rows_call::latent_out
     *  The latent MoE's output projection, latent -> hidden (NULL: no latent MoE). */
    /// Latent routed sum, f32 [n_rows][latent].
    float *lat_f;
    /// bf16 [n_rows][latent]: x_lat, the routed experts' input.
    uint16_t *lat_x, *lat_y;
    /** @var pulsar_moe_rows_call::lat_y
     *  bf16 [n_rows][latent]: the routed sum rounded to bf16, latent_out's input. */
    /// The latent projections' workspace.
    void *lat_ws;
    /// Size of lat_ws, in bytes.
    size_t lat_ws_bytes;
    /** L255: the layer's routed experts as Tessera planes instead -- the three kernel-ready stacks the family
     *  built from them (family_qwen.h PULSAR_QWEN_TESS_*), or NULL; then gate / up / down are unread. */
    const struct pulsar_tessera_proj *tessera_gate, *tessera_up, *tessera_down;
    /** @var pulsar_moe_rows_call::tessera_up
     *  Tessera up planes (see tessera_gate). */
    /** @var pulsar_moe_rows_call::tessera_down
     *  Tessera down planes (see tessera_gate). */
    /// The picks, [n_rows][top_k] expert ids; localised in place under expert parallelism.
    int32_t *selected;
    /// The picks' route weights, [n_rows][top_k].
    const float *weights;
    uint32_t top_k;                          ///< picks per row (selected / weights are [n_rows][top_k])
    /** task #44: [n_shared] EXL3 shared experts riding the routed launches (EXL3 routed arm only), or NULL */
    const struct pulsar_rows_moe_shared *shared;
    /// Entries in `shared` (0 when NULL).
    int n_shared;
    /// The input rows, bf16 [n_rows][hidden].
    const uint16_t *x_bf16;
    /// Rows in the call.
    int n_rows;
    /// The routed sum, f32 [n_rows][hidden].
    float *out;
    /// The launch's workspace (pulsar_rows_moe_routed_workspace_bytes).
    void *ws;
    /// Size of ws, in bytes.
    size_t ws_bytes;
    /// Device flag a non-finite output records nf_code into (first writer wins).
    uint32_t *nf_flag;
    /// The code recorded in *nf_flag.
    uint32_t nf_code;
    /// A PROMPT chunk (the prefill GEMM at every row count); false: decode / verify rows.
    bool prompt;
    void *stream;   ///< L296 dec2: the stream every launch goes on, a pulsar_rows_stream (NULL: the per-thread default)
} pulsar_moe_rows_call;
/// Run the routed part over raw bf16 rows (pulsar_moe_rows_call).
/// @param c  the call
/// @return false (said) when the stacks have no rows arm or the launch failed.
bool pulsar_moe_routed_rows(const pulsar_moe_rows_call *c);
/** Whether stored format `type` has a kernel for `role` at activation `act` -- the one table (weight_format.cpp). */
/// @param type  a PULSAR_TENSOR_* type
/// @param role  the tensor's role
/// @param act   the activation
/// @return true when a kernel exists.
bool pulsar_format_serves(uint32_t type, pulsar_weight_role role, pulsar_act_format act);
/** Admission by role: t's format serves `role` at one of the activations in the mask `acts`; a refusal names the
 *  formats that would (pulsar_tensor_admit's report). */
/// @param t      the tensor
/// @param owner  the family's name, for the message
/// @param role   the tensor's role
/// @param acts   PULSAR_ACTS() mask of the activations it may be read at
/// @return false (said) when no activation in `acts` serves it.
bool pulsar_tensor_admit_role(const pulsar_tensor *t, const char *owner, pulsar_weight_role role, uint32_t acts);
/** The MoE launchers' pairing rules for a layer's stacks (`up` NULL for a fused gate_up). */
/// @param gate   the gate stack (or fused gate_up)
/// @param up     the up stack, or NULL
/// @param down   the down stack
/// @param act    the activation
/// @param owner  the family's name, for the message
/// @return false (said) when the stacks do not pair.
bool pulsar_format_moe_combo(const pulsar_tensor *gate, const pulsar_tensor *up, const pulsar_tensor *down,
                             pulsar_act_format act, const char *owner);

/** L272 P4a: the mechanics of a family's weight binder (tensor_bind.cpp) -- report the same way for every
 *  family and return the verdict; the binder keeps its failure policy.  `owner` names the family in the
 *  message.  The required tensor `name`, or NULL (said). */
/// @param m      the model
/// @param owner  the family's name, for the message
/// @param name   the tensor's name
/// @return the tensor (owned by the model), or NULL.
pulsar_tensor *pulsar_tensor_bind(const pulsar_model *m, const char *owner, const char *name);
/** `t` has `nd` dims equal to d0, d1, d2 (ne order); false = said, with both shapes. */
/// @param t      the tensor
/// @param owner  the family's name, for the message
/// @param nd     dims expected
/// @param d0     dim[0] (fastest-varying)
/// @param d1     dim[1] (when nd >= 2)
/// @param d2     dim[2] (when nd == 3)
/// @return true when the shape matches.
bool pulsar_tensor_dims(const pulsar_tensor *t, const char *owner, uint32_t nd, uint64_t d0, uint64_t d1 = 0,
                        uint64_t d2 = 0);
/** `ok` is whether `t`'s format is one its reading op takes (`want` names them); false = said. */
/// @param t      the tensor
/// @param owner  the family's name, for the message
/// @param ok     the caller's verdict on t's format
/// @param want   the formats the reading op takes, for the message
/// @return `ok`.
bool pulsar_tensor_admit(const pulsar_tensor *t, const char *owner, bool ok, const char *want);

/** L272 P2: the bank carry is the core's (session_banks.cpp).  Save the live host view -- checkpoint,
 *  logits, the flags, the prefill frontier, the image identity, the speculative shadow -- into `bank`'s
 *  carry; host only. */
/// @param s     the session
/// @param bank  the bank whose carry receives the view
void pulsar_bank_carry_save_view(pulsar_session *s, uint32_t bank);
/** pulsar_token_text_fn over the engine's tokenizer -- the seam walks' vocabulary (session.cpp's prefix match,
 *  engine_api.cpp's prompt-held reads): one function for both. */
/// @param ud    the engine (pulsar_engine *)
/// @param token the token id
/// @param len   receives the text's length in bytes
/// @return a malloc'd copy of the token's text (the caller frees it), or NULL (no tokenizer: pulsar_token_text)
char *pulsar_engine_token_text_fn(void *ud, int token, size_t *len);
/** Bring `bank`'s saved view back into the host view, all or nothing.  false = nothing valid was saved, or the
 *  bank's live row did not restore (said) -- either way the host view is untouched, and the family decides what a
 *  bank with no carry means. */
/// @param s     the session
/// @param bank  the bank
/// @return true when the whole saved view was installed.
bool pulsar_bank_carry_restore_view(pulsar_session *s, uint32_t bank);
/** L322: the host view of a bank nothing was saved for -- empty and INVALID, so its first sync resets the bank and
 *  prefills cold (never continues another bank's history: the host view the caller held describes the bank it was
 *  synced on).  Every family's restore of such a bank installs it. */
/// @param s  the session
void pulsar_bank_carry_fresh_view(pulsar_session *s);
/** A bank's committed history: the live view for the live bank, else its carry.  NULL = none. */
/// @param s     the session
/// @param bank  the bank
/// @return the history (borrowed), or NULL.
const pulsar_tokens *pulsar_bank_history(pulsar_session *s, uint32_t bank);
/** L334: pulsar_session_bank_pos's rule over what the pool knows: none() for a bank outside a pool of `n_banks`
 *  (refused by name, logged once) or one that holds no history (`history` NULL), else the history's length.
 *  @param bank the bank asked about
 *  @param n_banks the pool's banks
 *  @param history the bank's history (pulsar_bank_history) */
/// @return the bank's position (its history's length), or none().
TokPos pulsar_bank_pos_of(uint32_t bank, uint32_t n_banks, const pulsar_tokens *history);

/** The session's cooperative cancel hook, polled at a prefill chunk boundary (session.cpp).  A mirrored
 *  session stops only together: the leader decides and ships the verdict, every worker reads it, so every
 *  rank must poll at the same boundaries. */
/// @param s  the session (NULL: false)
/// @return true when the sync must stop at this boundary.
bool pulsar_session_cancelled(pulsar_session *s);

/** L115 token-seam rescue, the family-neutral half (sync_driver.cpp, L284): the prompt re-spells, with canonical
 *  ids, bytes the live view holds with its sampled ids.  `tokens` is live[0..live_cut) + prompt[prompt_cut..] --
 *  the live history up to the deepest shared byte boundary, the prompt after it -- and `placed` the request's
 *  images re-placed on it (L226/L273).  How the session reaches `live_cut` is the family's: a rewind, or the
 *  resume rule from a grid checkpoint at or below it. */
struct pulsar_seam_stitch {
    enum { IMAGES_MAX = 64 };
    pulsar_tokens tokens = {};                ///< owned; freed with the stitch
    pulsar_image_ref placed[IMAGES_MAX] = {};  ///< the request's images, start_pos on `tokens`
    TokCount live_cut{0};                     ///< live tokens the stitch keeps
    PromptToks prompt_cut{0};                 ///< prompt tokens the kept live tokens re-spell
    pulsar_seam_stitch() = default;
    pulsar_seam_stitch(const pulsar_seam_stitch &) = delete;
    pulsar_seam_stitch &operator=(const pulsar_seam_stitch &) = delete;
    ~pulsar_seam_stitch() { free(tokens.v); }
};
/** Stitch `prompt` onto the live view when the byte match keeps more than `past` live tokens.  false = no
 *  stitch: the view is not valid, the match keeps `past` or fewer, or (said) the stitched prompt's image blocks
 *  are not the request's images (a block cut by the seam, a count that differs). */
/// @param s         the session
/// @param prompt    the request's prompt
/// @param images    the request's images, start_pos on `prompt`
/// @param n_images  entries in `images`
/// @param past      live tokens the plain match already keeps
/// @param out       receives the stitch
/// @return true when stitched.
bool pulsar_session_seam_stitch(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                                int n_images, int past, pulsar_seam_stitch *out);

/** A family's prefill shape (L284): the chunk grid, how wide a resumed walk's chunks may be, and the alignment of
 *  every non-final chunk end -- the inputs of the one cut rule (pulsar_prefill_plan_next_end).  Qwen: {prefill cap,
 *  prefill cap, 1}; DeepSeek: {prefill cap, raw window cap, the compress ratios' LCM} (gpu_graph_prefill_shape). */
typedef struct {
    uint32_t cap;           ///< chunk ends snap to the absolute multiples of it (the cold pass's chunk width)
    uint32_t resumed_cap;   ///< a walk that starts past 0 takes chunks of at most min(cap, this)
    uint32_t align;         ///< a non-final chunk end rounds down to a multiple of it (1 = none)
} pulsar_prefill_shape;

/** What a family supplies for the core's sync (sync_driver.cpp, L272 P2; every family since L284). */
typedef struct pulsar_sync_ops {
    const char *name;                                  ///< the family's name in messages
    /** The live bank's state holds exactly what the session's view (checkpoint) says. */
    bool (*state_agrees)(pulsar_session *s);
    /** Clear the live bank to position 0 (a cold prefill follows). */
    bool (*reset_bank)(pulsar_session *s);
    /** Prefill `prompt` on the live bank from `start` -- the core loop's 0 /
     *  PULSAR_SESSION_SYNC_INTERRUPTED / 1 (pulsar_prefill_loop). */
    int (*prefill)(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start);
    /** The session's prefill shape (the core loop and the prefill quantum read it). */
    void (*prefill_shape)(pulsar_session *s, pulsar_prefill_shape *out);
} pulsar_sync_ops;
/** The core's sync: continue the view, else resume from the shared prefix's deepest grid checkpoint, else reset and
 *  prefill from 0 -- interruptibly (sync_driver.cpp, L272 P2), with the request's images under the core's licence
 *  (L268; n_images 0 = text).  Returns 0, PULSAR_SESSION_SYNC_INTERRUPTED, or 1 with `err`. */
/// @param s         the session
/// @param prompt    the prompt the live bank must hold
/// @param images    the request's images (start_pos on `prompt`)
/// @param n_images  entries in `images` (0 = text)
/// @param ops       the family's sync ops
/// @param err       failure message buffer
/// @param errlen    its size
/// @return 0, PULSAR_SESSION_SYNC_INTERRUPTED, or 1.
int pulsar_session_sync_default(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                                int n_images, const pulsar_sync_ops *ops, char *err, size_t errlen);

/** The family's sync / eval / invalidate as the core runs them (engine_api.cpp, L284) -- every rank, mirrored
 *  or not, and every in-engine caller: what a family's op does to its state, plus what the session does for
 *  every family around it.  A sync begins a request and an invalidate forgets the history, so both drop the
 *  speculative lookahead and re-arm the quench (spec_lookahead_reset); an eval commits a token chosen outside
 *  the speculative round, so the carry no longer follows the state. */
/// @param s         the session
/// @param prompt    the prompt
/// @param images    the request's images
/// @param n_images  entries in `images`
/// @param err       failure message buffer
/// @param errlen    its size
/// @return as pulsar_session_sync_default.
int pulsar_session_family_sync(pulsar_session *s, const pulsar_tokens *prompt, const pulsar_image_ref *images,
                               int n_images, char *err, size_t errlen);
/// Evaluate one token on the live bank through the family's eval (refuses a full context, a non-id, a stale state).
/// @param s       the session
/// @param token   the token
/// @param err     failure message buffer
/// @param errlen  its size
/// @return 0, or 1 with `err`.
int pulsar_session_family_eval(pulsar_session *s, int token, char *err, size_t errlen);
/// Forget the live view's history (the family's invalidate, the read-ahead cancel, the lookahead reset).
/// @param s  the session
void pulsar_session_family_invalidate(pulsar_session *s);
/** Tokens the batched lane fed (decoded into its own buffer) that the view has not recorded yet -- every family's
 *  one rule (L284): the view grows, its logits are stale, and it is valid exactly when the state agrees with it. */
/// @param s     the session
/// @param toks  the committed tokens
/// @param n     their count (<= 0: nothing)
void pulsar_session_note_committed(pulsar_session *s, const int *toks, int n);

/** THE cut rule (prefill_loop.cpp, L284): a walk's chunk ends, as a pure function of the chunk's start, the
 *  prompt's end and these fixed inputs.  In order: snap to the absolute grid of `grid_cap` (at most `chunk_cap`
 *  rows); round a non-final end down to `align`; move an end that would split an image block to the block's start
 *  (or past its end when the block starts the chunk); stop at the prompt's message endpoint inside the chunk (L311);
 *  then cut the final chunk at the prompt's last resume-grid point (pulsar_ckpt_final_cut) unless that lands inside a
 *  block. */
typedef struct {
    /// The absolute grid ends snap to, in tokens.
    uint32_t grid_cap, chunk_cap, align;
    /** @var pulsar_prefill_plan::chunk_cap
     *  The most rows one chunk carries. */
    /** @var pulsar_prefill_plan::align
     *  A non-final chunk end is rounded down to a multiple of this. */
    const pulsar_ckpt_store *store;          ///< the resume grid the final cut lands on
    const pulsar_engine *e;                  ///< the image geometry (pulsar_image_block_extent)
    /// The prompt being walked (borrowed).
    const pulsar_tokens *prompt;
    const pulsar_image_ref *images;          ///< the request's blocks (NULL / 0 = text)
    /// Entries in images.
    int n_images;
    uint32_t endpoint;                       ///< L311: the message endpoint a chunk stops at (0 = none; the walk's)
} pulsar_prefill_plan;
/** L311: the prompt's MESSAGE ENDPOINT -- where its last turn marker begins (pulsar_turn_marker_at): the history of
 *  complete messages before the generation header, the point an agent's side request (the same history plus a
 *  message of its own) and a re-rendered generation header branch from.  Only a split-invariant family's (any cut
 *  is its cold prefill's bytes, so the cut the record needs moves no byte); 0 for another family, a prompt with no
 *  marker past position 0, or one below the store's smallest checkpoint. */
/// @param e the engine (its family's split-invariance and turn markers)
/// @param store the checkpoint store (its smallest grid point)
/// @param prompt the prompt
/// @return the endpoint position, or 0
uint32_t pulsar_prompt_endpoint(pulsar_engine *e, const pulsar_ckpt_store *store, const pulsar_tokens *prompt);
/** The plan of a walk from `start` (chunk_cap = the shape's resumed width when start != 0). */
/// @param[out] p the plan
/// @param shape the family's prefill shape
/// @param start the walk's first position
/// @param store the resume grid the final cut lands on
/// @param e the engine (image geometry)
/// @param prompt the prompt (borrowed)
/// @param images the request's images (NULL = text)
/// @param n_images their count
void pulsar_prefill_plan_init(pulsar_prefill_plan *p, const pulsar_prefill_shape *shape, uint32_t start,
                              const pulsar_ckpt_store *store, const pulsar_engine *e, const pulsar_tokens *prompt,
                              const pulsar_image_ref *images, int n_images);
/** The end of the chunk that starts at `pos0`, in (pos0, end]. */
/// @param p the plan
/// @param pos0 the chunk's first position
/// @param end the walk's end
/// @return the chunk's end
uint32_t pulsar_prefill_plan_next_end(const pulsar_prefill_plan *p, uint32_t pos0, uint32_t end);

/** The prefill walk every chunked prefill runs (prefill_loop.cpp, L272 P2): the order -- poll the stop hook, cut the
 *  chunk by the plan, run it, land it, poll again -- with the effects as hooks.  `chunk` runs rows [pos0, pos0 +
 *  rows) (`last`: it ends the prompt); `landed` (NULL = none) takes the effects of a chunk that ended at `chunk_end`
 *  (captures, the view, progress); `stop` (NULL = never) is polled before every chunk and after every chunk but the
 *  last. */
struct pulsar_prefill_walk {
    /// The cut rule the walk follows.
    const pulsar_prefill_plan *plan;
    /// Run rows [pos0, pos0 + rows); `last` = the chunk ends the prompt.  false aborts the walk.
    bool (*chunk)(void *ud, uint32_t pos0, uint32_t rows, bool last);
    /// Take the effects of a chunk that ended at chunk_end (NULL = none).
    bool (*landed)(void *ud, uint32_t chunk_end);
    /// Polled before every chunk and after every chunk but the last; true stops the walk (NULL = never).
    bool (*stop)(void *ud);
    /// User data passed to every hook.
    void *ud;
};
/** L329 (b2): what a bank's prompt read-ahead has queued -- the rows of prompt[from, to), decided by the tokens
 *  [from - 2, to) (the n-gram reaches two back) whose FNV-1a is `hash`.  from == to = nothing queued. */
struct pulsar_read_ahead_mark {
    /// First queued prompt position.
    uint32_t from, to;
    /** @var pulsar_read_ahead_mark::to
     *  One past the last queued position (from == to: nothing queued). */
    /// FNV-1a of tokens [from - 2, to).
    uint64_t hash;
};
/** L329 (b2): THE read-ahead rule (prefill_loop.cpp), a pure function of the bank's record and the call: a prefill
 *  has rows to `run_end` of prompt[0, len) in flight; the rows to read ahead are [*from, *to) within [run_end,
 *  run_end + window) (window = the prefill chunk): not those `mark` already queued for the same tokens, and nothing
 *  when that adds less than window / 2 short of the prompt's end.  *drop: the record is for other tokens (or
 *  behind the run) -- its queued rows are dropped first.  Returns true when [*from, *to) is to be queued; the caller
 *  then records *next as the bank's mark.  Pure, so the host test drives it (pulsar_test --read-ahead). */
/// @param mark the bank's record of what is queued
/// @param prompt the prompt's ids
/// @param len its length
/// @param run_end the end of the rows the prefill has in flight
/// @param window the prefill chunk, in tokens
/// @param[out] from first position to read ahead
/// @param[out] to one past the last
/// @param[out] drop the record is stale: cancel its queued rows first
/// @param[out] next the record to keep once [from, to) is queued
/// @return true when [*from, *to) is to be queued
bool pulsar_read_ahead_plan(const pulsar_read_ahead_mark *mark, const int *prompt, uint32_t len, uint32_t run_end,
                            uint32_t window, uint32_t *from, uint32_t *to, bool *drop, pulsar_read_ahead_mark *next);
/** L329 (b2): may a read-ahead be queued with MemAvailable at `mem_available` bytes (0 = unreadable: fail closed)?
 *  Only above PULSAR_READ_AHEAD_MEM_HEADROOM plus the pool's whole unread cap at a page a row (~9 GiB).  Pure
 *  (pulsar_test --read-ahead). */
/// @param mem_available MemAvailable in bytes (0 = unreadable)
/// @return true when a read-ahead may be queued
bool pulsar_read_ahead_admit(uint64_t mem_available);
/** L329 (b2): MemAvailable from /proc/meminfo in bytes, 0 when unreadable (read once per arm, never per row). */
/// @return bytes, or 0
uint64_t pulsar_read_ahead_mem_available(void);
/** L329 (b2): the session's read-ahead for `bank` (pulsar.h pulsar_session_prompt_read_ahead, which is this) and its
 *  cancel -- the bank's queued rows dropped, its record cleared (pulsar_session_family_invalidate, the walk).  It
 *  runs ONLY for a served prompt: `prompt` is the one a server named (the fused lane's call, or the sync's
 *  pulsar_session_set_prefill_ahead); NULL -- every sync nobody licensed (a CLI, a gate, imatrix, a tool sync) --
 *  reads nothing ahead. */
/// @param s the session
/// @param bank the bank (< 2^16)
/// @param prompt the served prompt, or NULL (nothing is read ahead)
/// @param run_end the end of the rows in flight
/// @return rows queued (0: nothing to do, refused, stood down for memory, or off)
uint32_t pulsar_session_read_ahead(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt, uint32_t run_end);
void pulsar_session_read_ahead_cancel(pulsar_session *s, uint32_t bank);
/** Run the walk over [start, end): 0 when it reached `end`, PULSAR_SESSION_SYNC_INTERRUPTED when `stop`
 *  said so at a chunk boundary (the device drained), 1 when a hook failed or a cut was out of range. */
/// @param w the walk (its plan and hooks)
/// @param start first position
/// @param end one past the last
/// @return 0, PULSAR_SESSION_SYNC_INTERRUPTED, or 1
int pulsar_prefill_walk_run(const pulsar_prefill_walk *w, uint32_t start, uint32_t end);
/** Rows from `pos0` to the next ABSOLUTE multiple of `cap` (P13, every walk's chunk rule): a walk that starts
 *  off the cap grid -- a resume, a continuation, the chunk after an image cut -- lands on the cold prefill's
 *  chunk ends, which sit on the resume grid where the walk captures (pulsar_ckpt_landed). */
/// @param pos0 the chunk's first position
/// @param cap the chunk cap (> 0)
/// @return rows, 1..cap
static inline uint32_t pulsar_prefill_to_boundary(uint32_t pos0, uint32_t cap) {
    return cap - pos0 % cap;
}

/** One prefill chunk of the family's forward (L272 P2): rows [pos0, pos0 + rows) of `prompt` on the
 *  live bank; `last` heads the final row into s->logits.  false = the chunk failed (logged). */
typedef bool (*pulsar_prefill_chunk_fn)(pulsar_session *s, const pulsar_tokens *prompt, uint32_t pos0,
                                        uint32_t rows, bool last, void *ud);

/** The family-neutral prefill loop (prefill_loop.cpp, L272 P2): `prompt` from `start` on the session's live bank
 *  and store, cut by the one rule over the family's prefill shape and the sync's borrowed images, every chunk end
 *  handed to the shared capture rule (pulsar_ckpt_landed).  After each chunk the session's view advances (checkpoint
 *  = the prompt so far) and the progress hooks hear prefill_chunk / prefill_display; the cancel hook is polled
 *  before the first chunk and after every chunk that ends where a resume is exact.  Returns 0 when the prompt is in,
 *  PULSAR_SESSION_SYNC_INTERRUPTED when the hook stopped it at a chunk boundary (the view stands there, the logits
 *  stale), 1 when a chunk failed. */
/// @param s the session
/// @param prompt the prompt
/// @param start first position to prefill
/// @param chunk the family's chunk forward
/// @param ud passed to `chunk`
/// @return 0, PULSAR_SESSION_SYNC_INTERRUPTED, or 1
int pulsar_prefill_loop(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start, pulsar_prefill_chunk_fn chunk,
                        void *ud);

/* L295 F4: the core row-step driver's session operations (step_driver.cpp) -- a driver family's session ops and sync
 * are these, built by the two macros below (the sync carries the family's name for its messages). */
int pulsar_driver_session_create(pulsar_session *s, uint32_t n_banks);
void pulsar_driver_session_destroy(pulsar_session *s);
bool pulsar_driver_sync_state_agrees(pulsar_session *s);
bool pulsar_driver_sync_reset_bank(pulsar_session *s);
int pulsar_driver_prefill(pulsar_session *s, const pulsar_tokens *prompt, uint32_t start);
void pulsar_driver_prefill_shape(pulsar_session *s, pulsar_prefill_shape *out);
int pulsar_driver_eval_row(pulsar_session *s, int token, char *err, size_t errlen);
int pulsar_driver_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, float *logits,
                                  int logits_cap, char *err, size_t errlen);
int pulsar_driver_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits,
                               int logits_cap, uint32_t *out_n_rows, uint32_t max_head_runs, char *err, size_t errlen);
int pulsar_driver_decode_fused(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                               const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                               char *err, size_t errlen);
/** A driver family's sync ops (`NAME` in its messages). */
#define PULSAR_DRIVER_SYNC_OPS(NAME)                                                                                \
    { NAME, pulsar_driver_sync_state_agrees, pulsar_driver_sync_reset_bank, pulsar_driver_prefill,                  \
      pulsar_driver_prefill_shape }
/** A driver family's session ops over its sync ops `SYNC`.  invalidate is NULL: the core's forgets the HOST view;
 *  the device state is left alone -- the next sync of that bank finds the view invalid and resets the bank before
 *  it prefills, so a stale state is never decoded, and the server's invalidate (a bank provision, an eviction, a
 *  stop string mid-batch) cannot wipe OTHER banks' conversations.  The fused step heads the driver's slab. */
#define PULSAR_DRIVER_SESSION_OPS(SYNC)                                                                             \
    { pulsar_driver_session_create, pulsar_driver_session_destroy, SYNC, pulsar_driver_eval_row,                     \
      pulsar_driver_decode_multiseq, pulsar_driver_decode_mixed, NULL, pulsar_driver_decode_fused,                   \
      PULSAR_DRIVER_HEADS_MAX }

/* L216 image-layout math: a port of the checkpoint's inference/image_processor.py.
 * Pure functions of the image dimensions and the block's position in the prompt,
 * so they are graded directly against the reference by tests/vision_layout_gate.cpp
 * (goldens generated by running image_processor.py itself).  `types` are the
 * sentinel roles IMAGE_START/PAD/IMAGE/NEWLINE/END = 0..4; their token ids are
 * vocab_size + type (out-of-vocab, per the reference). */
/** The aligner grid and sentinel-block length of an image at a given size (vision_grid_tokens). */
typedef struct {
    /// Aligner grid height (text rows).
    int n_llm_h, n_llm_w;   ///< aligner grid, i.e. text rows/cols the image occupies
    int num_tokens;         ///< sentinel-block length incl. row padding and the compressor pad
} pulsar_vision_grid;
/** The size an image is resized to and the grid it then occupies (vision_solve_resize_ratio / vision_safe_resize). */
typedef struct {
    /// Aligner grid height after the fit.
    int n_llm_h, n_llm_w;   ///< aligner grid after the fit
    /// Patch-aligned pixel height the image is resized to.
    int best_height, best_width;  ///< patch-aligned pixel size the image is resized to
    int num_tokens;         ///< sentinel-block length at that size
} pulsar_vision_resize;
pulsar_vision_grid vision_grid_tokens(int best_height, int best_width,
                                      int patch_size, int downsample_ratio);
pulsar_vision_resize vision_solve_resize_ratio(int height, int width,
                                               int patch_size, int downsample_ratio,
                                               int max_n_token);
/** Shrink until the grid fits `max_n_token` minus the compressor pad.  Returns 0
 * when the reference's `max_w > 1` assert would fire (caller refuses the image). */
/// @param height source height
/// @param width source width
/// @param best_height starting canvas height
/// @param best_width starting canvas width
/// @param patch_size ViT patch side
/// @param downsample_ratio the aligner's merge factor
/// @param max_n_token the block's token budget
/// @param[out] out the fitted size and grid
/// @return 1, or 0
int vision_safe_resize(int height, int width, int best_height, int best_width,
                       int patch_size, int downsample_ratio, int max_n_token,
                       pulsar_vision_resize *out);
/** Build the sentinel block for a grid at `start_pos`.  Returns the types count
 * (<= types_cap) or -1 when a buffer is too small; `perm` receives
 * n_llm_h*n_llm_w aligner-row indices for the IMAGE slots. */
/// @param n_llm_h text-grid height
/// @param n_llm_w text-grid width
/// @param start_pos the block's first prompt position
/// @param[out] types_out [types_cap] roles
/// @param types_cap its capacity
/// @param[out] perm_out [perm_cap] aligner-row indices
/// @param perm_cap its capacity
/// @return roles written, or -1
int vision_build_image_block(int n_llm_h, int n_llm_w, int start_pos,
                             int *types_out, int types_cap,
                             int *perm_out, int perm_cap);
/** Run the bound tower over one image's patches (n_h*n_w*3*PATCH*PATCH bf16
 * values) and write (out_rows, PULSAR_N_EMBD) bf16 embeddings.  Returns 0 on
 * refusal.  Needs a GPU and a bound tower. */
/// @param w the tower's weights
/// @param m the model (their mapping)
/// @param patches the image's patches, bf16
/// @param n_h ViT patch-grid height
/// @param n_w ViT patch-grid width
/// @param[out] out [out_rows][PULSAR_N_EMBD] bf16
/// @param out_cap capacity of out, in elements
/// @param[out] out_rows rows written
/// @param[out] dbg optional stage dump, (2 + dbg_blocks) rows of n_h*n_w*PULSAR_VISION_DIM bf16 (NULL = none)
/// @param dbg_blocks ViT blocks the dump captures
/// @return 1, or 0 on refusal
int vision_forward(const pulsar_vision_weights *w, const pulsar_model *m,
                   const uint16_t *patches, int n_h, int n_w,
                   uint16_t *out, int out_cap, int *out_rows,
                   uint16_t *dbg, uint32_t dbg_blocks);

/** The vision config the preprocessing reads (mirrors the reference's args). */
typedef struct {
    /// ViT patch side, in pixels.
    int   patch_size;
    /// The aligner's merge factor (ViT patches per text row / column).
    int   downsample_ratio;
    /// The most tokens an image's sentinel block may take.
    int   max_n_token;
    /// The reference's min_pixels.
    int   min_pixels;
    float max_wh_ratio;   ///< <= 0 means the reference's None (no clamp, no stretch)
} pulsar_vision_args;
/** What one preprocessed image produced. */
typedef struct {
    /// ViT patch-grid height.
    int n_vit_h, n_vit_w;         ///< ViT patch grid
    /// Text-grid height after the aligner merge.
    int n_llm_h, n_llm_w;         ///< text grid after the aligner merge
    /// Width of the resized / padded canvas, in pixels.
    int best_width, best_height;  ///< the resized/padded canvas the patches came from
} pulsar_vision_image;
/** Pixel half of image_processor.load_image(): decode-free.  Runs Pillow's
 * bicubic resample, ImageOps.contain/pad (127-grey), the (x/255-0.5)/0.5
 * normalisation and the patchify, all bit-exact with the reference.  `rgb` is
 * width*height*3 bytes; `patch_out` receives n_vit_h*n_vit_w*3*patch^2 bf16
 * values (as bit patterns).  Returns 0 on refusal (grid that the reference's
 * assert would reject, or a cap too small). */
/// @param rgb the decoded image
/// @param width its width
/// @param height its height
/// @param args the processor's arguments
/// @param[out] patch_out the patches
/// @param patch_cap capacity of patch_out, in values
/// @param[out] out the grid produced
/// @return 1, or 0
int vision_preprocess_rgb(const uint8_t *rgb, int width, int height,
                          const pulsar_vision_args *args,
                          uint16_t *patch_out, size_t patch_cap,
                          pulsar_vision_image *out);
/** The grid a decoded image WILL produce, without touching its pixels: the one
 * place the geometry lives, so a caller can size its patch buffer first (the
 * canvas can be larger than the input, because min_pixels upscales). */
/// @param width source width
/// @param height source height
/// @param args the processor's arguments
/// @param[out] out the grid
/// @return 1, or 0 when no grid fits
int vision_image_grid(int width, int height, const pulsar_vision_args *args,
                      pulsar_vision_image *out);
/** Decode image BYTES to 8-bit RGB.  PNG (libpng) and JPEG (libjpeg-turbo, with
 * Pillow's settings, which are libjpeg's defaults) only -- the caller fails
 * loudly on 0 rather than guessing at a format.  `*rgb_out` is malloc'd and the
 * caller owns it.  CMYK/YCCK JPEG is refused: Pillow keeps those in CMYK and its
 * own .convert("RGB") is a different transform from libjpeg's. */
/** L321: the process's GPU phase lock.  The engines of a group (pulsar_engine_open_group) share one GPU, the fixed
 *  GEMM's shapes and workspace and the encoders' scratch, so a single-call family (an encoder, a parser) runs its GPU
 *  work in PHASES -- a packed encode forward, one ViT block, a page's prefill, one decode step -- each one a
 *  pulsar_group_gpu_phase: the lock held, the device synchronised before it goes.  Phases of different engines
 *  interleave; a phase is never shared, so whatever a phase leaves for the next lives in its engine's own buffers. */
void pulsar_group_gpu_lock(void);
void pulsar_group_gpu_unlock(void);
/** One GPU phase (see above): takes the group lock on construction; synchronises the device and releases the
 *  lock on destruction.  Not copyable. */
struct pulsar_group_gpu_phase {
    pulsar_group_gpu_phase() { pulsar_group_gpu_lock(); }
    ~pulsar_group_gpu_phase() {
        (void)pulsar_gpu_synchronize();   /* a failure surfaces at the phase's own next launch or read */
        pulsar_group_gpu_unlock();
    }
    pulsar_group_gpu_phase(const pulsar_group_gpu_phase &) = delete;
    pulsar_group_gpu_phase &operator=(const pulsar_group_gpu_phase &) = delete;
};

/** Pillow's resample filters the image paths use (L321: BILINEAR for Nemotron Parse). */
typedef enum { PULSAR_PIL_BICUBIC = 0, PULSAR_PIL_BILINEAR = 1 } pulsar_pil_filter;
/** Pillow's Image.resize(BICUBIC | BILINEAR) of packed RGB8 (ImagingResample, fixed-point, horizontal then vertical),
 *  bit for bit -- every family's resize whose reference is Pillow (DeepSeek's load_image, Qwen's PIL processor,
 *  Parse's).  Returns a malloc'd dst_w x dst_h image (a copy when the size is unchanged), NULL on failure. */
/// @param src RGB8 [src_h][src_w]
/// @param src_w source width
/// @param src_h source height
/// @param dst_w destination width
/// @param dst_h destination height
/// @param filter the resample filter
/// @return the resized image (caller frees), or NULL
uint8_t *vision_pil_resize_rgb(const uint8_t *src, int src_w, int src_h, int dst_w, int dst_h, pulsar_pil_filter filter);
int vision_decode_rgb(const uint8_t *bytes, size_t len,
                      uint8_t **rgb_out, int *w_out, int *h_out);

/** One image READY for the model: the reference's per-image half of
 * prepare_vl_inputs.  `span_ids` are the sentinel token ids (vocab_size + type),
 * `span_types` the roles, `perm` the aligner-row order for the IMAGE slots, and
 * `patches` the ViT input.  All buffers are malloc'd; free with
 * vision_prepared_free(). */
typedef struct {
    uint16_t *patches;      ///< n_patches * 3 * PATCH * PATCH bf16
    int32_t  *span_ids;     ///< span_len ids: vocab_size + type
    int32_t  *span_types;   ///< span_len roles (IMAGE_START..IMAGE_END)
    int32_t  *perm;         ///< n_perm aligner-row indices for the IMAGE slots
    /// ViT patches in `patches`.
    int n_patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, span_len, n_perm;
    /** @var pulsar_vision_prepared::n_vit_h
     *  ViT patch-grid height. */
    /** @var pulsar_vision_prepared::n_vit_w
     *  ViT patch-grid width. */
    /** @var pulsar_vision_prepared::n_llm_h
     *  Text-grid height after the aligner merge. */
    /** @var pulsar_vision_prepared::n_llm_w
     *  Text-grid width after the aligner merge. */
    /** @var pulsar_vision_prepared::span_len
     *  Entries in span_ids / span_types: the sentinel block's tokens. */
    /** @var pulsar_vision_prepared::n_perm
     *  Entries in perm. */
} pulsar_vision_prepared;
/** Decode + preprocess + build the sentinel span for ONE image at `start_pos`
 * (its token index in the prompt).  `vocab_size` is the text model's n_vocab,
 * which the sentinel ids are offset by.  Returns 0 on refusal. */
/// @param bytes the encoded image
/// @param len its length
/// @param args the processor's arguments
/// @param start_pos the block's first prompt position
/// @param vocab_size the text model's n_vocab
/// @param[out] out the prepared image (free with vision_prepared_free)
/// @return 1, or 0
int vision_prepare_image(const uint8_t *bytes, size_t len, const pulsar_vision_args *args,
                         int start_pos, int vocab_size, pulsar_vision_prepared *out);
void vision_prepared_free(pulsar_vision_prepared *p);
/** The reference's prepare_vl_inputs(): expand every image placeholder in `in`
 * into that image's sentinel block in `out` (which the caller owns and must have
 * emptied).  `out`'s ids become `vocab_size + role` in build_image_block's order,
 * and `starts[k]` receives image k's BLOCK start -- the value
 * pulsar_image_ref::start_pos must carry.  `preps[k]` receives the prepared image
 * for the later merge.  Refuses when the placeholder count and the image count
 * disagree, or when an image cannot be prepared. */
/** L268: DeepSeek's pulsar_family_vision::expand -- prepare `img` where its block begins (out->len) and append
 *  the block's sentinel ids (`vocab_size + role`, in build_image_block's N-layout order). */
/// @param img the image
/// @param[in,out] out ids appended
/// @param err message buffer
/// @param errlen its size
/// @return false + err when the image cannot be prepared
bool vision_ds4_expand(const pulsar_image_ref *img, pulsar_tokens *out, char *err, size_t errlen);

/* The reference's sentinel ROLES (image_processor: IMAGE_START, IMAGE_PAD,
 * IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)).  A prompt slot belonging to an
 * image block carries `vocab_size + role`, so these are the ONLY ids at or above
 * vocab_size the engine will accept -- which is what lets the prefill token
 * upload distinguish a sentinel from a bad id. */
/// Sentinel role: the block's opening slot.
#define PULSAR_VISION_ROLE_IMAGE_START 0
/// Sentinel role: a padding slot.
#define PULSAR_VISION_ROLE_IMAGE_PAD   1
/// Sentinel role: an image-embedding slot (an aligner row).
#define PULSAR_VISION_ROLE_IMAGE       2
/// Sentinel role: the end of a grid row.
#define PULSAR_VISION_ROLE_NEWLINE     3
/// Sentinel role: the block's closing slot.
#define PULSAR_VISION_ROLE_IMAGE_END   4

/** The embedding width the VISION GOLDENS were written at.  The golden blobs
 * carry the prepared image block as bf16 rows, and those rows are the width of
 * the REFERENCE checkpoint they came from (Vision-Exp / 0731, 4096) -- a
 * property of the FILE, not of whatever profile the reading binary happens to
 * compile its default in.  A gate that sizes that read with PULSAR_N_EMBD
 * desyncs by 25% in a V4.1-default build (5120) and reads garbage from then on;
 * that is exactly what the first merged-tree battery did. */
#define PULSAR_VISION_GOLDEN_N_EMBD 4096u

/** The image sentinel BLOCK beginning at `start_pos` (the reference's
 * ImageInput.start: the block's first slot, which is a compressor pad, with the
 * IMAGE_START sentinel a few slots in).  `*len_out` receives the BLOCK length --
 * what merge_image_embeddings writes and what the chunk planner must not split.
 * Returns 0 if the ids there are not such a block.  The ONE place the sentinel
 * roles are resolved for a scan. */
/// @param ids the prompt
/// @param n its length
/// @param n_vocab the sentinel offset
/// @param start_pos where to look
/// @param[out] len_out the block's length
/// @return 1, or 0 when no block starts there
int vision_span_extent(const int32_t *ids, int n, int n_vocab, int start_pos, int *len_out);
/** The reference's get_image_visible(): per-token visible counts to the
 * left/right within each [IMAGE_START, IMAGE_END] span.  Pure integer function
 * of the token ids, so it is graded directly against the reference by
 * tests/vision_visible_gate.cpp.  Prefill only -- an image span must arrive in
 * one chunk. */
/// @param ids the tokens
/// @param n their count
/// @param n_vocab the sentinel offset
/// @param max_image_tokens the widest span
/// @param[out] left [n] visible tokens to the left
/// @param[out] right [n] visible tokens to the right
void vision_image_visible(const int32_t *ids, int n, int n_vocab, int max_image_tokens,
                          int32_t *left, int32_t *right);
/** The reference's `width = min(seqlen, window_size + max_image_tokens)`: the
 * column count of the matrix vision_window_topk_visible() writes. */
/// @param n sequence length
/// @param window_size the attention window
/// @param max_image_tokens the widest span
/// @return columns
int vision_visible_width(int n, int window_size, int max_image_tokens);
/** The reference's get_window_topk_idxs_visible(): the window index matrix,
 * widened per query so a token inside an image span reaches the whole span.
 * `out` receives n * vision_visible_width(...) int32 values. */
/// @param window_size the attention window
/// @param n sequence length
/// @param left [n] from vision_image_visible
/// @param right [n] from vision_image_visible
/// @param max_image_tokens the widest span
/// @param[out] out [n][vision_visible_width] indices
void vision_window_topk_visible(int window_size, int n, const int32_t *left,
                                const int32_t *right, int max_image_tokens,
                                int32_t *out);
/** The reference's merge_image_embeddings() for ONE prepared image: run the
 * tower over its patches and scatter the aligner rows (in `perm` order) into the
 * IMAGE slots, filling every other slot with its type's learned vector.  `out`
 * receives span_len * PULSAR_N_EMBD bf16 values. */
/// @param w the vision weights
/// @param m the model (their mapping)
/// @param prep the prepared image
/// @param[out] out the span rows
/// @param out_cap capacity of out, in elements
/// @param[out] out_len rows written
/// @return 1, or 0
int vision_merge_span(const pulsar_vision_weights *w, const pulsar_model *m,
                      const pulsar_vision_prepared *prep,
                      uint16_t *out, int out_cap, int *out_len);
/** DeepSeek's image-preprocessing args (the checkpoint's policy constants and the tower dims). */
/// @param[out] a the arguments
void vision_ds4_args(pulsar_vision_args *a);
/** L268: DeepSeek's pulsar_family_vision::block_rows -- prepare `img` at its block's position, take the aligner
 *  rows from `tower` (the core's cache) or run the tower (handing its rows back through *tower_out), and assemble
 *  the block (the merge_image_embeddings scatter vision_merge_span does). */
/// @param w the vision weights
/// @param m the model
/// @param img the image
/// @param block_len the block's rows
/// @param tower cached aligner rows, or NULL
/// @param n_tower their count
/// @param[out] tower_out a fresh tower run's rows (caller owns)
/// @param[out] n_tower_out their count
/// @param[out] out [block_len][PULSAR_N_EMBD] bf16
/// @param err message buffer
/// @param errlen its size
/// @return false + err on a refusal
bool vision_ds4_block_rows(const pulsar_vision_weights *w, const pulsar_model *m, const pulsar_image_ref *img,
                           int block_len, const uint16_t *tower, int n_tower, uint16_t **tower_out,
                           int *n_tower_out, uint16_t *out, char *err, size_t errlen);
/** Dense layers and compressed layers use different RoPE bases. */
/// @param il a layer
/// @return its rope frequency base
float layer_rope_freq_base(uint32_t il);
float layer_rope_freq_scale(uint32_t il);
/// @param x a value
/// @return x * sigmoid(x)
float silu(float x);
/// CPU SwiGLU with the clamp.
/// @param[out] out [n]
/// @param gate [n]
/// @param up [n]
/// @param n elements
/// @param clamp the swiglu clamp
void swiglu(float *out, const float *gate, const float *up, uint64_t n, float clamp);
uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk);
uint64_t argmax_f32(const float *x, uint64_t n);
/** Release every GPU tensor owned by the whole-model graph runtime. */
/// @param g the graph
void gpu_graph_free(pulsar_gpu_graph *g);
void gpu_graph_release(pulsar_gpu_graph *g);   /* tensors only; no segment-graph reset */
bool gpu_tensor_fill_f32(pulsar_gpu_tensor *t, float v, uint64_t n);
bool gpu_graph_load_directional_steering(
        pulsar_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale);
bool gpu_graph_directional_steering_attn_enabled(const pulsar_gpu_graph *g);
bool gpu_graph_directional_steering_ffn_enabled(const pulsar_gpu_graph *g);
bool gpu_graph_apply_directional_steering_attn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);
bool gpu_graph_apply_directional_steering_ffn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);
uint64_t gpu_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out);
pulsar_gpu_tensor *gpu_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes);
/** True when PULSAR_CUDA_GRAPH_DUMP_PREFIX is set (cached). Graph allocation
 * uses this to skip buffers that exist only to be dumped. */
/// @return whether the dump prefix is set
bool gpu_graph_debug_dump_enabled(void);
const char *gpu_graph_debug_dump_prefix(void);
bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos);
/** The predicate every f32 store-skip must use: true when EITHER observer (the
 * dump or the range sweep) will read the bytes.  See the definition's comment
 * -- the sweep reads through dump_tensor's own early branch, so debug_wants
 * alone is not the question.  _any() is the coarse, per-name-less twin. */
/// @param name the tensor's dump name
/// @param il the layer
/// @param pos the position
/// @return true when an observer reads the bytes
bool gpu_graph_f32_store_observed(const char *name, uint32_t il, uint32_t pos);
bool gpu_graph_f32_store_observed_any(void);
/** PULSAR_DSPARK_DUMP is set (parsed once): the drafter's f32 rows the dump
 * reads are stored; otherwise the drafter emits E4M3 only. */
/// @return nonzero when PULSAR_DSPARK_DUMP is set
int  gpu_graph_spec_dump_active(void);
void gpu_graph_debug_dump_hc_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_q_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos);
bool gpu_graph_needs_ffn_out(const pulsar_gpu_graph *g, uint32_t il, uint32_t pos);
bool gpu_graph_ensure_batch_ffn_out(pulsar_gpu_graph *g);
bool gpu_graph_alloc_raw_cap(
        pulsar_gpu_graph *g,
        const pulsar_weights     *weights,
        const pulsar_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        uint32_t                n_banks,
        bool                    enable_spec);
/** Bank-pool size a live gpu_graph_alloc_raw_cap is given: PULSAR_MSEQ_BANKS
 * parsed once (clamped to [1, PULSAR_MSEQ_MAX]; 1 = pool disabled), or the value
 * a caller set.  The one owner of the number (L159 inc 5). */
/// @return banks
uint32_t gpu_graph_bank_pool_n(void);
void     gpu_graph_bank_pool_set(uint32_t n);
int      gpu_graph_bank_pool_env_pinned(void);   /* 1 when the operator's variable set it */
/** Re-install the graph's per-layer cache views onto `bank` (pool mode only).
 * Contract: call only between fully synchronized forwards — the previous
 * bank's enqueued work must be complete, because the graph pointers change
 * under every subsequent launch.  This swaps DEVICE views only: the host
 * per-session state (ring fill, positions, spec-shadow contents) is the
 * caller's to save/restore per bank.  The compressed frontier is NOT in that
 * list any more — ms_n_comp is indexed by bank, and repoint sets cur_bank, so
 * the accessors follow automatically (stage 1b).  On
 * failure the views may be mixed-bank — treat the graph as dead. */
/// @param g the graph
/// @param bank the bank
/// @return false on failure (the graph is then dead)
bool gpu_graph_bank_repoint(pulsar_gpu_graph *g, uint32_t bank);
/** Effective pool size for banked kernel launches: banks.n_banks, or 1 when
 * the pool is disabled (the classic tensors act as bank 0). */
/// @param g the graph
/// @return banks, >= 1
uint32_t gpu_graph_bank_pool_count(const pulsar_gpu_graph *g);
/** Tier-2 overcommit (task #55): demand-paged comp+index VA bytes for ONE bank at
 * a context (the overcommit-reserved, physical-on-touch part).  See gpu_diag.cpp. */
/// @param ctx_size the context, tokens
/// @return bytes
KvBytes gpu_graph_demand_paged_bytes_per_bank(uint32_t ctx_size);
/** Compressed rows a layer of compress ratio `ratio` (non-zero) holds at
 * ctx_size: the one capacity formula, read by gpu_graph_compute_dims (the
 * allocator's per-layer caps) and the KV sizing in steering.cpp. */
/// @param ctx_size the context, tokens
/// @param ratio the compress ratio (non-zero)
/// @return compressed rows
static inline uint32_t gpu_graph_comp_cap(uint32_t ctx_size, uint32_t ratio) {
    return ctx_size / ratio + 2u;
}
/** One bank's comp + index cache bytes at ctx_size in their stored row formats
 * (steering.cpp); the demand-paged term of the overcommit split. */
/// @param ctx_size the context, tokens
/// @return bytes
uint64_t gpu_graph_comp_index_bytes_for_context(uint32_t ctx_size);
/** One bank's raw SWA ring across every layer at raw_cap rows, in the stored
 * WINDOW / MAIN row formats (steering.cpp): the eager term of the KV
 * sizing, the same bytes gpu_graph_bank_slabs_alloc lays out per layer. */
/// @param raw_cap the ring's rows
/// @return bytes
uint64_t gpu_graph_raw_ring_bytes_for_context(uint32_t raw_cap);
/** The deepest compressed pool any layer holds at ctx_size (the smallest
 * non-zero compress ratio's gpu_graph_comp_cap): the row count the
 * indexer_scores scratch is sized by and the one the boot line reports as
 * compressed_kv_rows.  ctx_size itself when no layer compresses. */
/// @param ctx_size the context, tokens
/// @return rows
uint32_t gpu_graph_comp_cap_max(uint32_t ctx_size);
/** Exact touched (physically resident) demand-paged comp/index KV of ONE bank,
 * from its resident high-water (ms_comp_hw), not its frontier: pages above a
 * rewound or evicted frontier stay resident until the bank's physical is freed.
 * The increment-2b guard uses this for the per-bank Δ projection and the
 * victim tie-break; the server's provisioning reads it to prefer a bank whose
 * pages are already resident.
 */
/// @param g the graph
/// @param bank the bank
/// @return bytes
KvBytes gpu_graph_bank_touched_kv_bytes(const pulsar_gpu_graph *g, uint32_t bank);
/** Tier-2 task #55 increment 2b — CONSERVATIVE per-bank comp/index growth over
 * `rows` token rows: Σ_kv-sources ceil(rows/ratio)·(comp_row + index_row) (one
 * emit writes both rows; the ceil rounds up, the safe side).  Position-
 * independent, so total Δ = n_live_growing_banks × this.  DeepSeek's one
 * conversion of token rows to bytes (L333); its per-layer compressed rows never
 * leave it.
 */
/// @param rows token rows
/// @return bytes
KvBytes gpu_graph_quantum_growth_bytes_per_bank(KvRows rows);
/** Tier-2 task #55 increment 2b — per-bank physical evict/restore reclaim
 * primitives (direct cudaFree / cudaMallocManaged of one bank's split comp/index
 * + base-table rebuild). See gpu_diag.cpp. */
/// @param g the graph
/// @param bank the bank
/// @return false on a bad bank or a failed free
bool gpu_graph_bank_free_physical(pulsar_gpu_graph *g, uint32_t bank);
/** Tier-2 task #55 increment 2b — RESTORE alloc primitive. Reallocate ONE evicted
 * bank's comp/index physical (fresh cudaMallocManaged: VA reserved, physical on
 * touch) and rebuild its base-table entries to the new pointers. The caller then
 * reloads the bank's KV into these (the server from its segment chain). Idempotent: a
 * slab already present is left untouched. Returns false on OOM.
 */
/// @param g the graph
/// @param bank the bank
/// @return false on OOM
bool gpu_graph_bank_alloc_physical(pulsar_gpu_graph *g, uint32_t bank);
/** True when `bank`'s comp/index physical is currently freed (evicted) — the
 * server checks this before restoring on a returning request.
 */
/// @param g the graph
/// @param bank the bank
/// @return true when freed
bool gpu_graph_bank_is_evicted(const pulsar_gpu_graph *g, uint32_t bank);

/** Whole-pool cache tensors for banked kernel operands: the bank slab when
 * the pool is enabled, else the classic single-session tensor (== bank 0).
 * NULL for layers without that cache kind. */
/// @param g the graph
/// @param il the layer
/// @return the tensor (borrowed), or NULL
pulsar_gpu_tensor *gpu_graph_bank_raw_pool(pulsar_gpu_graph *g, uint32_t il);
/** Nominal comp/index operand for the batched attention/indexer wrappers. With
 * per-bank split allocations there is no single slab; the batched (descriptor)
 * path addresses banks through the base-pointer table (below), so this returns
 * the lowest RESIDENT bank's allocation as the nominal typed operand (every
 * bank's is the same size, which drives the wrappers' buffer-size validation;
 * L328: bank 0 can be spilled like any idle bank). Pool disabled → the classic
 * tensor.
 */
/// @param g the graph
/// @param il the layer
/// @return the tensor (borrowed), or NULL
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_pool(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_pool(pulsar_gpu_graph *g, uint32_t il);
/** Per-bank comp/index base-pointer tables (device arrays of n_banks pointers,
 * indexed by seq_id) the batched READ kernels use in place of base +
 * seq_id*comp_cap over one slab. NULL when the pool is disabled. */
/// @param g the graph
/// @param il the layer
/// @return the table (borrowed), or NULL
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_bases(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_bases(pulsar_gpu_graph *g, uint32_t il);
/** Fresh single-bank views for the batched emit path (caller frees; when the
 * pool is disabled, bank must be 0 and the view wraps the classic tensor).
 * kind: the per-(bank,layer) comp caches and compressor state lanes. */
/// @param g the graph
/// @param il the layer
/// @param bank the bank
/// @return the view (caller frees), or NULL
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
/** V4 only (the profile owns an indexer compressor): the same two views for the
 * indexer's own state lane.  NULL where the profile keeps no such lane. */
/// @param g the graph
/// @param il the layer
/// @param bank the bank
/// @return the view (caller frees), or NULL
pulsar_gpu_tensor *gpu_graph_bank_index_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
/** Host state hand-off for the fields that still have scalar twins.
 *
 * ⚠ THE COMPRESSED FRONTIER NO LONGER RIDES THIS. Stage 1b deleted
 * layer_n_comp, so install's frontier loop is empty and capture's half is
 * gone: ms_n_comp is indexed by bank and the accessors follow cur_bank on
 * their own. What these still carry is the drafter ring state (dspark_n_raw,
 * dspark_prompt_lo/n) — the twins stage 2 is meant to collapse.
 *
 * Capture after per-bank work so those arrays reflect the bank; install before
 * per-bank work resumes so the scalars are that bank's again. */
/// @param g the graph
/// @param bank the bank whose arrays receive the scalars
void gpu_graph_bank_counters_capture(pulsar_gpu_graph *g, uint32_t bank);
void gpu_graph_bank_counters_install(pulsar_gpu_graph *g, uint32_t bank);

/* Tier-2 PATH A host-carry primitive (see pulsar_bank_carry).  save copies the
 * session's live HOST per-conversation state into bank's shadow AND captures
 * the graph frontier counters (gpu_graph_bank_counters_capture).  restore
 * repoints the device views to bank, installs its frontier counters, copies
 * the shadow back into the session, and clears mseq_dirty (the cheap
 * no-re-prefill resume: counters_install re-establishes per-bank truth, which
 * is exactly what mseq_dirty guards).  Call save on the leaving bank before,
 * and restore on the entering bank after, a bank switch — both only between
 * fully synchronized forwards (the gpu_graph_bank_repoint contract).  restore
 * returns false only on a bad bank id or OOM growing an owned buffer. */
/* pulsar_session_bank_state_save/restore + pulsar_session_bank_count/repoint are the
 * public server-facing API (declared in pulsar.h). */
/* pulsar_session::bank_carry_free() is the member form of the old
 * pulsar_session_bank_carry_free free function (internal-only, no facade). */
/** Arm one banked multiseq batched step over n_rows packed rows: pos[t] is
 * row t's absolute position, seq[t] its TRUE bank id.  Writes the host
 * mirrors + device descriptor arrays (lazily allocated), verifies the
 * DRIVER CONTRACT (each batched bank's ms frontier is position-true —
 * ms_n_comp == first_pos/ratio — i.e. no mid-prefill bank is co-scheduled),
 * and refreshes the scalar superset counters ONCE (the step's emit-inclusive
 * bound, max over rows of (pos+1)/ratio).  capture_cur first captures the
 * current bank's scalars into its ms row (single-session diagnostic use).
 * Constraint (fail-loud): each bank's rows form ONE contiguous run with
 * consecutive positions inside the run (pos ascending by 1), and every run
 * starts at a position > 0 (position-0 rows rejected — admission prefill is
 * classic single-bank in v1).  Banks may sit at unrelated positions: every
 * upstream batch stage is per-row-position driven (RoPE variants take the
 * batch_positions device array; NULL degenerates to pos0+t).  Every rejection
 * prints the reason.  Disarm + self-check with gpu_graph_multiseq_step_end
 * after the layer sweep (it validates every batched bank's frontier advanced
 * to its position-derived value and the superset equals max over banks). */
/* n_dec_declared: the step's leading decode-row count when the caller knows it
 * (a fused step, L260); < 0 infers it from the layout as before. */
/// @param g the graph
/// @param pos [n_rows] absolute positions
/// @param seq [n_rows] true bank ids
/// @param n_rows rows
/// @param capture_cur capture the current bank's scalars first
/// @param n_dec_declared the step's leading decode rows (< 0: inferred)
/// @return false (printed) on a layout the contract refuses
bool gpu_graph_multiseq_step_begin(pulsar_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur, int32_t n_dec_declared);
bool gpu_graph_multiseq_step_end(pulsar_gpu_graph *g);
/** Tier-2 batched multi-session decode: one token per live bank through ONE
 * weight sweep (see the definition comment in imatrix.cpp for the full driver
 * contract).  logits out = [n_active * PULSAR_N_VOCAB], row k = bank[k].
 * Returns 1 ok / 0 recoverable rejection (nothing mutated) / -1 fatal (armed
 * sweep or head failed — session state untrusted). */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param tokens [n_active] tokens
/// @param pos [n_active] positions
/// @param bank [n_active] banks
/// @param n_active rows
/// @param[out] logits headed rows, or NULL
/// @param[out] out_n_rows rows headed, or NULL
/// @param max_head_runs prompt runs headed at most
/// @param capture_cur capture the current bank's scalars first
/// @param fused the fused step's shape, or NULL
/// @return 1 ok, 0 recoverable rejection (nothing mutated), -1 fatal
int gpu_graph_decode_multiseq_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const int             *tokens,
        const int32_t         *pos,
        const int32_t         *bank,
        uint32_t               n_active,
        float                 *logits,
        uint32_t              *out_n_rows,
        uint32_t               max_head_runs,
        bool                   capture_cur,
        const pulsar_fused_shape *fused);   /* L260: NULL = the plain mixed contract */

/** Work shape of everything the process has run through the two graph funnels:
 * gpu_graph_prefill_layer_major (one call per prefill chunk, plus the L195
 * state-only warm-up passes) and gpu_graph_decode_multiseq_batch (one call per
 * classic decode step, per mixed-entry K-row run, and per speculative verify
 * batch).  tests/gates_runner.cpp --shape snapshots this around each gate to
 * say whether a gate's cost is DEPTH (max_pos), REPETITION (step_calls) or
 * SETUP (prefill_calls/prefill_tokens) instead of guessing from its name.
 *
 * Process-global, not per-graph: the graph carries no back-pointer to the
 * engine, and one process runs one battery.  The reader resets them per gate
 * (pulsar_gate_shape_reset) so a reused engine's earlier work is not charged
 * to a later gate.  Counters only -- no branch, no allocation. */
typedef struct {
    uint64_t prefill_calls;   /**< gpu_graph_prefill_layer_major calls */
    uint64_t prefill_tokens;  /**< tokens covered by those calls (n_tokens summed) */
    uint64_t step_calls;      /**< gpu_graph_decode_multiseq_batch calls */
    uint64_t step_rows;       /**< rows across those calls (n_active summed) */
    uint64_t max_pos;         /**< deepest position reached, +1 (a token count) */
} pulsar_gate_shape;
void pulsar_gate_shape_read(pulsar_gate_shape *out);
void pulsar_gate_shape_reset(void);

bool gpu_graph_init_dspark_target(pulsar_gpu_graph *g, const uint32_t target_layer_ids[3]);
uint32_t gpu_graph_raw_span_for_batch(
        const pulsar_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens);
uint32_t gpu_graph_raw_start_for_span(
        const pulsar_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw);
bool gpu_graph_env_flag(const char *name, int *cache);
/** Prefill score slice, in rows: the prefill [indexer score -> top-k -> indexed
 * attention] sequence runs in <= slice-token spans so indexer_scores (the one
 * ctx-scaling f32 work buffer with a token dimension) is allocated with slice
 * rows instead of prefill_cap.  512, one number one place (the
 * PULSAR_PREFILL_SLICE env override went with L159 inc 4).
 */
/// @return rows (512)
uint32_t gpu_graph_prefill_slice(void);
/** Comp-cache row stride in bytes for the active storage format (pack-aware). */
/** The output head for ONE row of the sweep-final stream: batch_cur_hc row
 * `row` collapsed with batch_hc_pre row `row` (the last FFN's pre), normed,
 * projected into `out`.  Slice 4d: the projection covers the vocab RANGE
 * [vocab_lo, vocab_lo + vocab_dim); the single-box caller passes
 * (0, N_VOCAB, its row), which is the whole head. */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param row the stream row
/// @param vocab_lo first vocab id
/// @param vocab_dim vocab ids projected
/// @param[out] out the logits row
/// @return false on a failed launch
bool gpu_graph_encode_output_head(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row,
        uint32_t               vocab_lo,
        uint64_t               vocab_dim,
        pulsar_gpu_tensor      *out);
/** The output head for rows [row0, row0 + n_tokens) of the sweep-final stream
 * into `out` rows [0, n_tokens), covering the vocab range
 * [vocab_lo, vocab_lo + vocab_dim).  The single-box caller passes
 * (0, N_VOCAB, g->spec_logits). */
/** L292: the output head's graph-capture key -- the carrier parity, the head's row count, and its decode
 *  prefix (pulsar_gpu_matmul_decode_prefix: which rows take the decode arms).  Every host-side fact the
 *  captured launches depend on is a field; n_tokens <= 16, so each fits its byte. */
/// @param hc_parity the carrier parity
/// @param n_tokens head rows (<= 16)
/// @param decode_prefix rows taking the decode arms
/// @return the key
uint64_t gpu_graph_output_head_seg_key(bool hc_parity, uint32_t n_tokens, uint32_t decode_prefix);
bool gpu_graph_encode_output_head_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row0,
        uint32_t               n_tokens,
        uint32_t               vocab_lo,
        uint64_t               vocab_dim,
        pulsar_gpu_tensor      *out);
/** THE one place the output head knows about TP (slice 4d).  With the group off
 * each is exactly the call above with the whole range -- same bytes, same
 * captures.  With the group armed this rank projects only ITS vocab range into
 * a slice, stages it to host, all-gathers every rank's range (concatenation in
 * rank order) and writes the assembled full logits back into `out`, so every
 * rank holds the same full vector and samples independently: no leader-only
 * decision and no token broadcast.  `out` must be the full [rows, N_VOCAB]
 * destination; the slice reuses its head as scratch. */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param row the stream row
/// @param[out] out the full logits
/// @return false on a failed launch or gather
bool gpu_graph_encode_output_head_row_tp(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row,
        pulsar_gpu_tensor      *out);
bool gpu_graph_encode_output_head_batch_tp(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row0,
        uint32_t               n_tokens,
        pulsar_gpu_tensor      *out);
bool gpu_graph_encode_dspark_output_head_batch(
        pulsar_gpu_graph            *g,
        const pulsar_model          *dspark_model,
        const pulsar_dspark_weights *dw,
        const pulsar_model          *base_model,
        const pulsar_weights        *bw,
        uint32_t                  n_tokens,
        uint64_t                  vocab_dim);
bool gpu_graph_dspark_project_main_x(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w);
/** Seed n_rows drafter-KV rows from main_x.  false = a stage failed and the
 * three ring counters were rolled back; the caller refuses its spec round. */
/// @param g the graph
/// @param dspark_model the drafter's model
/// @param w its weights
/// @param n_rows rows
/// @return false when a stage failed (ring counters rolled back)
bool gpu_graph_dspark_seed_draft_kv(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        uint32_t                 n_rows);
/** L260: seed the drafter rings of SEVERAL banks from verify-capture rows in one
 *  pass: row t (capture row src_rows[t]) seeds bank row_bank[t]'s three rings at
 *  that bank's next position; a bank's rows are contiguous and in commit order.
 *  The projections run over all rows at once (chunks of at most the exact decode
 *  range, so each row is bit-identical to the one-row seed), the stores go
 *  through the bank-major slabs, and each bank's ring counters advance by its
 *  rows only when every chunk succeeded.  false = nothing advanced. */
/// @param g the graph
/// @param dspark_model the drafter's model
/// @param w its weights
/// @param src_rows [n_rows] capture rows
/// @param row_bank [n_rows] banks
/// @param n_rows rows
/// @return false when a chunk failed (nothing advanced)
bool gpu_graph_dspark_seed_rows_banked(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        const uint32_t            *src_rows,
        const uint32_t            *row_bank,
        uint32_t                 n_rows);
bool gpu_graph_dspark_draft_forward(
        pulsar_gpu_graph          *g,
        const pulsar_model         *base_model,
        const pulsar_weights       *base_weights,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        pulsar_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft);
/** L150: the same forward over the rows of n_banks banks at once (contract at
 * the definition); n_banks == 1 with NULL arrays == the single-bank forward. */
/// @param g the graph
/// @param base_model the target model
/// @param base_weights its weights
/// @param dspark_model the drafter's model
/// @param w its weights
/// @param[out] base_logits_out logits, or NULL
/// @param draft_ids [n_rows] draft tokens
/// @param n_rows rows
/// @param n_banks banks
/// @param row_bank [n_rows] each row's bank, or NULL (one bank)
/// @param bank_n_raw [n_banks] each bank's three ring counts, or NULL
/// @param bank_n_draft [n_banks] each bank's draft rows, or NULL
/// @return false on a failed stage
bool gpu_graph_dspark_draft_forward_banks(
        pulsar_gpu_graph          *g,
        const pulsar_model         *base_model,
        const pulsar_weights       *base_weights,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        pulsar_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_rows,
        uint32_t                n_banks,
        const uint32_t          *row_bank,
        const uint32_t         (*bank_n_raw)[3],
        const uint32_t          *bank_n_draft);
/* The pair's all-reduce of `n_tokens` n_embd-wide f32 rows of `t` (gpu_prefill.cpp):
 * the row lane at decode/verify width, the bulk lane above it, the host big gate
 * on transports without either.  `addend` (optional) is folded in first and
 * zeroed.  A no-op returning true when the graph has no pair. */
/** What a row all-reduce borrows (tp_rows.cpp, L266): the transport, the registered slab's and bulk
 *  buffer's device mappings, the stage ticket (one zeroed device u32), the exchange seq both ranks advance
 *  in lockstep, the row width in floats. */
typedef struct {
    /// The transport.
    pulsar_tp *tp;
    /// The registered slab's device mapping.
    void *slab_dev;
    /// The bulk buffer's device mapping.
    void *bulk_dev;
    /// The stage ticket: one zeroed device u32.
    pulsar_gpu_tensor *ticket;
    /// The exchange seq both ranks advance in lockstep.
    uint64_t *seq;
    /// Row width, in floats.
    uint32_t n_embd;
} pulsar_tp_rows;
/** Sum t [n_rows][n_embd] (plus `addend`, then zeroed, when given) across the group, in place; slot = the
 *  exchange's slab layer.  No transport: a no-op. */
/// @param x the transport and buffers
/// @param slot the exchange's slab layer
/// @param n_rows rows
/// @param[in,out] t [n_rows][n_embd] f32
/// @param addend folded in first and zeroed, or NULL
/// @param what the caller, for messages
/// @return false on a transport failure
bool pulsar_tp_allreduce_rows(const pulsar_tp_rows *x, uint32_t slot, uint32_t n_rows, pulsar_gpu_tensor *t,
                              pulsar_gpu_tensor *addend, const char *what);
/** What a vocab gather borrows (tp_rows.cpp, L266): the transport, the slab's device mapping, the stage ticket,
 *  the own-slice scratch (pulsar_tp_vocab_own_bytes; NULL off a row-lane pair), the gather's seq, the vocab
 *  width and the most rows one gather carries. */
typedef struct {
    /// The transport.
    pulsar_tp *tp;
    /// The slab's device mapping.
    void *slab_dev;
    /// The stage ticket.
    pulsar_gpu_tensor *ticket;
    /// Own-slice scratch (pulsar_tp_vocab_own_bytes); NULL off a row-lane pair.
    pulsar_gpu_tensor *vocab_own;
    /// The gather's seq.
    uint64_t *seq;
    /// Vocab width, in logits.
    uint32_t n_vocab;
    /// The most rows one gather carries.
    uint32_t max_rows;
} pulsar_tp_vocab;
/** head(lo, width, dst): this rank's logits for vocab [lo, lo + width), row r at dst + r * width. */
typedef std::function<bool(uint32_t lo, uint32_t width, pulsar_gpu_tensor *dst)> pulsar_tp_head_fn;
/** Bytes of the vocab gather's own-slice scratch for max_rows rows (0 off a row-lane pair); _for: the same
 *  from the group size and the transport's vector width (a caller that sizes before the transport exists). */
/// @param tp the transport
/// @param n_vocab vocab width
/// @param max_rows rows
/// @return bytes
uint64_t pulsar_tp_vocab_own_bytes(pulsar_tp *tp, uint32_t n_vocab, uint32_t max_rows);
uint64_t pulsar_tp_vocab_own_bytes_for(uint32_t n_ranks, uint64_t vec_bytes, uint32_t n_vocab, uint32_t max_rows);
/** Every rank's vocab range gathered into out [n_rows][n_vocab] (rank order). */
/// @param x the transport and buffers
/// @param n_rows rows (<= max_rows)
/// @param head this rank's head
/// @param[out] out the full logits
/// @return false on a head or transport failure
bool pulsar_tp_vocab_gather(const pulsar_tp_vocab *x, uint32_t n_rows, const pulsar_tp_head_fn &head,
                            pulsar_gpu_tensor *out);
bool gpu_graph_tp_allreduce_rows(pulsar_gpu_graph *g, uint32_t il, uint32_t n_tokens,
                                 pulsar_gpu_tensor *t, pulsar_gpu_tensor *addend,
                                 const char *what);
pulsar_gpu_tensor *gpu_graph_tensor_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
pulsar_gpu_tensor *gpu_graph_hc_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/** Q buffers stride by PULSAR_Q_ELT_SIZE (L045) -- use this for batch_q/q, not
 * the generic float-strided helper above. */
/// @param base the tensor
/// @param row the row
/// @param row_values values per row
/// @return the view (caller frees), or NULL
pulsar_gpu_tensor *gpu_graph_q_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/** heads buffers stride by PULSAR_HEADS_ELT_SIZE (L033) -- use this for
 * batch_heads/heads, not the generic float-strided helper above. */
/// @param base the tensor
/// @param row the row
/// @param row_values values per row
/// @return the view (caller frees), or NULL
pulsar_gpu_tensor *gpu_graph_heads_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/** Read an HC residual carrier (BF16 storage; task #62) into an f32 host buffer,
 * expanding each sample. Dev-only (parity self-test + env-gated DSpark dumps). */
/// @param t a Q buffer (f16 storage)
/// @param off_elems first element
/// @param[out] out [n] f32
/// @param n elements
/// @return nonzero on success
int pulsar_read_q_f32(const pulsar_gpu_tensor *t, uint64_t off_elems,
                      float *out, uint64_t n);
int pulsar_read_hc_carrier_f32(const pulsar_gpu_tensor *t, uint64_t off_elems,
                            float *out, uint64_t n);
bool gpu_graph_upload_prompt_tokens(
        pulsar_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens);
/** Where the residual stream is born: prompt[pos0 .. pos0+n_tokens) embedded
 * into g->batch_cur_hc (hc copies) and g->batch_hc_pre set to the identity
 * pre-mix for those rows. */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param prompt the prompt
/// @param pos0 first position
/// @param n_tokens rows
/// @return false on a failed launch
bool gpu_graph_upload_prompt_embeddings_hc(
        pulsar_gpu_graph     *g,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens);
/** Scatter a merged image span (vision_merge_span's n_rows * PULSAR_N_EMBD bf16
 * rows) into the HC carrier at `row0`, replicating each row across all
 * PULSAR_N_HC streams the way the reference's post-merge HC expansion does.
 * Returns false, without writing, if the span does not fit the carrier. */
/// @param out_hc the HC carrier
/// @param rows [n_rows][PULSAR_N_EMBD] bf16
/// @param n_rows rows
/// @param row0 first carrier row
/// @param n_tokens the carrier's rows
/// @return false, without writing, when the span does not fit
bool gpu_graph_write_vision_span(pulsar_gpu_tensor *out_hc, const uint16_t *rows,
                                 uint32_t n_rows, uint32_t row0, uint32_t n_tokens);
/** Merge every image block that lies inside [pos0, pos0 + n_tokens) into the HC
 * carrier: the core's chunk merge (pulsar_image_merge_chunk) through DeepSeek's
 * HC-replicating writer.  `ids` is the whole prompt (the block extent comes from
 * the family geometry, not from the image), so a request whose prompt does not
 * carry the block it claims is refused.  Returns false on any refusal; true (a
 * no-op) when `vr` is NULL. */
/// @param out_hc the HC carrier
/// @param ids the whole prompt
/// @param n_ids its length
/// @param vr the borrowed image request, or NULL
/// @param pos0 the chunk's first position
/// @param n_tokens its rows
/// @return false on any refusal; true (a no-op) when vr is NULL
bool gpu_graph_merge_image_spans(pulsar_gpu_tensor *out_hc, const int32_t *ids, int n_ids,
                                 const pulsar_vision_request *vr, uint32_t pos0, uint32_t n_tokens);
/** Compute THIS chunk's image-span visibility and upload it to
 * g->vision_visible, once per chunk, before any layer's attention runs.  A
 * chunk with no sentinel id (every text chunk, and every chunk of an image
 * request that is not the first) leaves g->vision_visible_tokens == 0, which is
 * what keeps the text path bit-identical: the attention entries then receive
 * NULL pointers.  Refuses (returns false) a visibility span that cannot be
 * addressed in one pass rather than clipping it.  A no-op when the graph has no
 * borrowed image request. */
/// @param g the graph
/// @param ids the prompt
/// @param n_ids its length
/// @param start the chunk's first position
/// @param n_tokens its rows
/// @return false when a span cannot be addressed in one pass
bool gpu_graph_upload_vision_visible(pulsar_gpu_graph *g, const int32_t *ids,
                                     int n_ids, uint32_t start, uint32_t n_tokens);

bool gpu_graph_warmup_prefill_kernels(
        pulsar_gpu_graph   *g,
        const pulsar_model   *model,
        const pulsar_weights *weights,
        uint32_t           n_tokens);
bool gpu_graph_encode_layer_attention_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
bool gpu_graph_encode_layer_ffn_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
bool gpu_graph_encode_layer_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
/** save_row0 (inc 6, W2): the first row of THIS session's positions within
 * the verify forward's comp-save buffers. Classic single-bank rounds pass 0;
 * the batched lane passes the bank's row offset in the shared batch. */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param pos0 first position
/// @param n_positions positions
/// @param save_row0 the session's first row in the comp-save buffers
/// @return false (printed) on a failed stage
bool gpu_graph_dspark_compressor_rollforward(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        const pulsar_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions,
        uint32_t          save_row0);
bool imatrix_collector_init(pulsar_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path);
void imatrix_collector_free(pulsar_imatrix_collector *c);
bool imatrix_collector_save(
        const pulsar_imatrix_collector *c,
        const pulsar_weights           *weights,
        const char                  *path);
/** The llama.cpp legacy `.dat` writer every family's collection shares (imatrix.cpp): open + entry count, one
 *  entry per tensor (`n_expert` vectors of `n_col` means; a never-observed expert writes 1.0), then close with the
 *  chunk count and the dataset's name. */
/// @param path output path
/// @param n_entries entries that follow
/// @return the file, or NULL (printed)
FILE *imatrix_dat_open(const char *path, int32_t n_entries);
void imatrix_write_entry(FILE *fp, const char *name, const float *sum2, const uint32_t *counts, uint32_t n_expert,
                         uint32_t n_col);
bool imatrix_dat_close(FILE *fp, const char *path, int32_t chunks, const char *dataset_path);
/// DeepSeek's importance-matrix collection (imatrix.cpp).
extern const pulsar_family_imatrix k_ds4_imatrix;     // imatrix.cpp
/// The Qwen4-exp family's importance-matrix collection (imatrix_qwen.cpp).
extern const pulsar_family_imatrix k_qwen_imatrix;    // imatrix_qwen.cpp
bool gpu_graph_reset_prefill_state(pulsar_gpu_graph *g);
bool gpu_graph_prefill_layer_major(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        pulsar_gpu_tensor     *head_out,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud);
/** DeepSeek's prefill shape (L284): {prefill cap, raw window cap, the compress ratios' LCM} -- the session sync's
 *  (pulsar_sync_ops::prefill_shape) and the range prefill's one authority. */
/// @param g the graph
/// @return the shape
pulsar_prefill_shape gpu_graph_prefill_shape(const pulsar_gpu_graph *g);
/** Prefill prompt[start, start + n_tokens) on the graph's installed bank outside a session -- the imatrix collector
 *  and the gates' classic suffix -- by the one cut rule over gpu_graph_prefill_shape (text only), capturing grid
 *  checkpoints by the shared rule.  `logits` (NULL = none) receives the last row. */
/// @param g the graph
/// @param model the model
/// @param weights the weights
/// @param prompt the prompt
/// @param start first position
/// @param n_tokens rows
/// @param[out] head_out the last row's logits, or NULL
/// @param imatrix collector, or NULL
/// @return false on a failed chunk
bool gpu_graph_prefill_chunked_range(pulsar_gpu_graph *g, const pulsar_model *model, const pulsar_weights *weights,
                                     const token_vec *prompt, uint32_t start, uint32_t n_tokens,
                                     pulsar_gpu_tensor *head_out, pulsar_imatrix_collector *imatrix);
bool gpu_graph_verify_suffix(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens);
/** Reset a bank's compressor state lanes to the canonical empty group (kv 0,
 *  score -INF): their state at any even position. */
/// @param g the graph
/// @param bank the bank
/// @return false on a failed write
bool gpu_graph_compressor_state_reset(pulsar_gpu_graph *g, uint32_t bank);
/** Pick a raw SWA cache size for GPU.  During batched prefill it must cover
 * the previous window plus the current ubatch.
 */
/// @param ctx_size the context, tokens
/// @param prefill_cap the prefill chunk's rows
/// @return ring rows
uint32_t gpu_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap);
uint32_t gpu_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk);
void token_vec_push(token_vec *tv, int token);
void token_vec_free(token_vec *tv);
void dump_tokens_fp(FILE *fp, const pulsar_vocab *vocab, const token_vec *tokens);
/** The bytes `token` decodes to -- the GPT-2 byte alphabet reversed, literal
 * specials verbatim, an out-of-range id empty.  The ONE detokenizer:
 * pulsar_token_text and dump_tokens_fp both call it.  Malloc'd,
 * NUL-terminated; *len (optional) receives the byte count. */
/// @param vocab the vocabulary
/// @param token the id
/// @param[out] len byte count, or NULL
/// @return the bytes (caller frees)
char *vocab_token_text(const pulsar_vocab *vocab, int token, size_t *len);
/* The identity digest of a batched step's headed rows, on the device (engine_api.cpp; the TP mirror's check). */
uint64_t pulsar_session_batch_digest(pulsar_session *s, uint32_t n_rows);
/** L260 fusion: a fused step's digest -- its decode rows', then its `n_heads` headed rows'. */
/// @param s the session
/// @param n_dec decode rows
/// @param n_heads headed rows
/// @return the digest
uint64_t pulsar_session_fused_digest(pulsar_session *s, uint32_t n_dec, uint32_t n_heads);
/** L284 #2: the prompt runs a fused step heads (head_last[r] set, r < n_pf) -- THE count every reader of a
 *  shape uses. */
/// @param sh the shape
/// @return headed prompt runs
static inline uint32_t pulsar_fused_shape_heads(const pulsar_fused_shape *sh) {
    uint32_t h = 0;
    for (uint32_t r = 0; r < sh->n_pf && r < (uint32_t)sizeof(sh->head_last); r++) h += sh->head_last[r] ? 1u : 0u;
    return h;
}
/** L284 #2: the fused step on THIS rank (session_multiseq.cpp) -- the family's decode_fused op with the session's
 *  bookkeeping around it, for every family: the last fused step's block (fused_logits / fused_n_dec /
 *  fused_heads) is cleared before the op and recorded when it succeeds.  The C entry (both ranks' leader side)
 *  and the TP worker call this, never the op directly.  The caller has checked pulsar_engine_has_fused_step. */
/// @param s the session
/// @param reqs [n_rows] requests
/// @param n_rows rows
/// @param shape the step's shape
/// @param[out] logits the headed rows
/// @param logits_cap the TOTAL float capacity of `logits` (all rows: >= the headed rows x the logits width)
/// @param[out] out_n_rows rows headed
/// @param err message buffer
/// @param errlen its size
/// @return the family op's (0; 1 refused; -1 failed with state moved)
int pulsar_session_fused_local(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                               const pulsar_fused_shape *shape, float *logits, int logits_cap, uint32_t *out_n_rows,
                               char *err, size_t errlen);
/** L288 sdet: the key of the token `ahead` places past the stream's next undrawn one -- a function of the request's
 *  seed and the token's index only, read without consuming anything.  Every lane draws a token from its own row at
 *  this key, the plain samplers and the speculative walk alike, so a seeded stream's tokens do not depend on the
 *  lane or the drafts that produced them. */
/// @param rng the stream's state
/// @param ahead tokens past the next undrawn one
/// @return the key
uint64_t pulsar_sample_key(uint64_t rng, uint32_t ahead);
/** Step the stream past `n` committed tokens. */
/// @param[in,out] rng the stream's state
/// @param n committed tokens
void pulsar_sample_advance(uint64_t *rng, uint32_t n);

void pulsar_linux_graph_backend_set_oom_score(pulsar_backend backend);
void pulsar_release_instance_lock(void);
/** Refuse to start a second pulsar/ds4 process.  The model can map tens of GiB,
 * so a stale accidental second run is more dangerous than a normal CLI error.
 */
void pulsar_acquire_instance_lock(void);

/** ---- shared inline helpers ---- */

/** =========================================================================
 * Scalar Conversion and Small Host Helpers.
 * =========================================================================
 *
 * Host-side helpers the engine's TUs share: the f16 -> f32 scalar conversion
 * (the host reads of f16 tensors and diagnostics), the drafter's starting
 * depth, and the TP slice-registry key offsets.  There is no CPU inference
 * backend and no host quantized dot product; every model format is computed
 * on the GPU.
 */

/// @param h binary16 bits
/// @return the value
static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}


/** The depth a fresh session drafts at: the configured spec_depth or the drafter's, clamped to the drafter's
 *  depth_max; 0 = no drafter (session_spec.cpp). */
/// @param e the engine
/// @return the depth
uint32_t pulsar_spec_depth_start(const pulsar_engine *e);
/* The K-half registry key's offset for one tensor (L241 4g-2): the tensor
 * object's address, unique per tensor per engine -- abs_offsets repeat across
 * safetensors shards.  One authority for registration and lookup. */
/// The K-half registry key's offset for one tensor: its address.
/// @param t the tensor
/// @return the offset
static inline uint64_t pulsar_tp_kslice_key_offset(const pulsar_tensor *t) {
    return (uint64_t)(uintptr_t)t;
}

/* The expert tensor-parallel half-stack key's offset for one routed stack
 * (L241 4g-2): the tensor's index in its model's tensor table, 4 GiB apart so
 * no two half-stacks' ranges overlap under the one engine key.  The drafter
 * aliases the target's table, so indices are unique across both.  One
 * authority for registration (open) and lookup (the FFN encoder). */
/// The expert half-stack key's offset: (index + 1) << 32.
/// @param m the model
/// @param t one of its tensors
/// @return the offset
static inline uint64_t pulsar_tp_expert_half_offset(const pulsar_model *m, const pulsar_tensor *t) {
    return ((uint64_t)(t - m->tensors) + 1u) << 32;
}

#endif /* PULSAR_ENGINE_INTERNAL_H */

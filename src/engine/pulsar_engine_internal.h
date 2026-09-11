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

#include "pulsar.h"

#include "pulsar_gpu.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PULSAR_NEG_INF (-1.0e30f)
#define PULSAR_DEFAULT_RMS_EPS ( 1.0e-20f)   /* V4.1 (L218); 0731 was 1e-6 */
#define PULSAR_DEFAULT_HC_EPS  ( 1.0e-6f)
#define PULSAR_DEFAULT_SWIGLU_CLAMP_EXP    (10.0f)
#define PULSAR_DEFAULT_ROPE_FREQ_BASE      (10000.0f)
#define PULSAR_DEFAULT_ROPE_SCALE_FACTOR   (16.0f)
#define PULSAR_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
#define PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
#define PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define PULSAR_DEFAULT_ROPE_ORIG_CTX       UINT64_C(65536)

/** The V4.1 reasoning-effort line, byte-identical to the reference encoder
 * (encoding.py REASONING_EFFORT_TEMPLATE): the head, the decimal effort, the
 * tail.  Rendered before the system message whenever thinking is on; there
 * is no level that renders nothing (the 0731 "low adds nothing" scheme went
 * with 0731, L218). */
#define PULSAR_REASONING_EFFORT_HEAD "Reasoning Effort: "
#define PULSAR_REASONING_EFFORT_TAIL " (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


#if defined(__GNUC__) || defined(__clang__)
#define PULSAR_MAYBE_UNUSED __attribute__((unused))
#else
#define PULSAR_MAYBE_UNUSED
#endif

/** ---- shared macros ---- */



#define PULSAR_MODEL_SHAPE_NAME          (g_pulsar_shape.name)
#define PULSAR_MODEL_VARIANT             (g_pulsar_shape.variant)
#define PULSAR_N_LAYER                   (g_pulsar_shape.n_layer)
#define PULSAR_N_EMBD                    (g_pulsar_shape.n_embd)
#define PULSAR_N_VOCAB                   (g_pulsar_shape.n_vocab)
#define PULSAR_N_HEAD                    (g_pulsar_shape.n_head)
#define PULSAR_N_HEAD_KV                 (g_pulsar_shape.n_head_kv)
#define PULSAR_N_HEAD_DIM                (g_pulsar_shape.n_head_dim)
#define PULSAR_N_VALUE_DIM               (g_pulsar_shape.n_value_dim)
#define PULSAR_N_ROT                     (g_pulsar_shape.n_rot)
#define PULSAR_N_OUT_GROUP               (g_pulsar_shape.n_out_group)
#define PULSAR_N_LORA_Q                  (g_pulsar_shape.n_lora_q)
#define PULSAR_N_LORA_O                  (g_pulsar_shape.n_lora_o)
#define PULSAR_N_EXPERT                  (g_pulsar_shape.n_expert)
#define PULSAR_N_EXPERT_USED             (g_pulsar_shape.n_expert_used)
#define PULSAR_N_DSPARK_EXPERT           (g_pulsar_shape.n_dspark_expert)
#define PULSAR_N_DSPARK_EXPERT_USED      (g_pulsar_shape.n_dspark_expert_used)
#define PULSAR_N_EXPERT_SHARED           (g_pulsar_shape.n_expert_shared)
#define PULSAR_N_FF_EXP                  (g_pulsar_shape.n_ff_exp)
#define PULSAR_N_SWA                     (g_pulsar_shape.n_swa)
#define PULSAR_N_INDEXER_HEAD            (g_pulsar_shape.n_indexer_head)
#define PULSAR_N_INDEXER_HEAD_DIM        (g_pulsar_shape.n_indexer_head_dim)
#define PULSAR_N_INDEXER_TOP_K           (g_pulsar_shape.n_indexer_top_k)
#define PULSAR_CANDIDATE_BLOCK_SIZE      (g_pulsar_shape.candidate_block_size)
#define PULSAR_CANDIDATE_TOPK_BLOCKS     (g_pulsar_shape.candidate_topk_blocks)
#define PULSAR_N_HC                      (g_pulsar_shape.n_hc)
#define PULSAR_N_HC_SINKHORN_ITER        (g_pulsar_shape.n_hc_sinkhorn_iter)
#define PULSAR_RMS_EPS                   (g_pulsar_shape.rms_eps)
#define PULSAR_HC_EPS                    (g_pulsar_shape.hc_eps)
#define PULSAR_EXPERT_WEIGHT_SCALE       (g_pulsar_shape.expert_weight_scale)
#define PULSAR_SWIGLU_CLAMP_EXP          (g_pulsar_shape.swiglu_clamp_exp)
#define PULSAR_ROPE_FREQ_BASE            (g_pulsar_shape.rope_freq_base)
#define PULSAR_ROPE_SCALE_FACTOR         (g_pulsar_shape.rope_scale_factor)
#define PULSAR_ROPE_YARN_BETA_FAST       (g_pulsar_shape.rope_yarn_beta_fast)
#define PULSAR_ROPE_YARN_BETA_SLOW       (g_pulsar_shape.rope_yarn_beta_slow)
#define PULSAR_COMPRESS_ROPE_FREQ_BASE   (g_pulsar_shape.compress_rope_freq_base)
#define PULSAR_ROPE_ORIG_CTX             (g_pulsar_shape.rope_orig_ctx)

/** =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - IQ2_XXS routed gate/up experts
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256


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
#define PULSAR_MAX_DIMS   8




/** MXKV FP4 row bytes for the indexer compressed cache: the shared
 * definition at this model's indexer head_dim (68 B at 128). */
#define PULSAR_ENGINE_IDXFP4_ROWBYTES \
    ((uint64_t)PULSAR_MXKV_FP4_ROWBYTES((uint64_t)PULSAR_N_INDEXER_HEAD_DIM))

/** The two KV rows (src/pulsar_gpu.h, L218) at this model's head_dim: WINDOW
 * rows (528 B) fill every sliding-window ring -- raw, drafter, the current
 * chunk's pack buffer; MAIN rows (288 B) fill a kv source's compressed pool.
 * Neither has a conversion path from 0731's 384 B row -- stale payloads and
 * bank snapshots refuse by version and stride. */
#define PULSAR_ENGINE_WINKV_ROWBYTES  PULSAR_WINKV_ROWBYTES((uint64_t)PULSAR_N_HEAD_DIM)
#define PULSAR_ENGINE_MAINKV_ROWBYTES PULSAR_MAINKV_ROWBYTES((uint64_t)PULSAR_N_HEAD_DIM)


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
#define PULSAR_DSPARK_DRAFT_WINDOW 128u
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
    PULSAR_MAX_ATTN_SOURCE      = 8,   ///< CSA2 source layers of one kind (V4.1: 4 kv, 8 index)
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

typedef enum {
    PULSAR_VARIANT_FLASH = 0,
} pulsar_variant;

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
     * every later indexer (-1: none).  Ascending, and every kv source is also
     * an index source -- pulsar_attn_layout_install asserts both. */
    uint32_t n_kv_source;
    uint32_t kv_source_layer[PULSAR_MAX_ATTN_SOURCE];
    uint32_t n_index_source;
    uint32_t index_source_layer[PULSAR_MAX_ATTN_SOURCE];
    int32_t  candidate_source_layer;
    uint32_t candidate_topk_blocks;  ///< compressed-position blocks the candidate pool keeps per query
    uint32_t candidate_block_size;   ///< compressed positions per candidate block
    uint32_t n_hc;                ///< hyper-connection streams
    uint32_t n_hc_sinkhorn_iter;  ///< Sinkhorn normalisation iterations in the HC mix
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
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

/** One row of the GGUF tensor-type table: how a type is named and how its
 * bytes divide into blocks. Everything that sizes or strides a tensor goes
 * through this rather than restating the arithmetic. */
typedef struct {
    const char *name;      ///< type name as written in the file
    uint32_t block_elems;  ///< elements per quantisation block
    uint32_t block_bytes;  ///< bytes per block; the two give bytes-per-element
} gguf_type_info;

enum {
    PULSAR_TENSOR_F32      = 0,
    PULSAR_TENSOR_F16      = 1,
    PULSAR_TENSOR_Q8_0     = 8,
    PULSAR_TENSOR_Q2_K     = 10,
    PULSAR_TENSOR_IQ2_XXS  = 16,
    PULSAR_TENSOR_I32      = 26,
    PULSAR_TENSOR_BF16     = 30,
    PULSAR_TENSOR_FP8_E4M3 = 38,
    PULSAR_TENSOR_FP4_E2M1 = 39,
    /* CUTLASS block-scaled MXFP4: expert-major ColumnMajor E2M1 data blob
     * followed by a swizzled E8M0 SF blob, per expert. Byte size is NOT a
     * uniform per-element rate (see cutlass_mxfp4_expert_bytes()) -- the
     * gguf_types[] table entry for this type exists only so tensor_type()
     * recognizes it; real per-expert offsets come from that helper, not
     * from the table's block_elems/block_bytes. */
    PULSAR_TENSOR_CUTLASS_MXFP4 = PULSAR_GPU_TENSOR_CUTLASS_MXFP4,   /* one spelling: pulsar_gpu.h */
    /* MXFP8_LT: the pre-store of PULSAR_TENSOR_FP8_E4M3 (type 38). Identical E4M3
     * weights and E8M0 block scales, but stored in the EXACT device-side layout
     * the runtime otherwise builds at first use: de-interleaved [in,out]
     * col-major E4M3 data immediately followed by the mx_sfoff()-swizzled E8M0
     * scale. On detection the FP8 matmul skips the per-weight cudaMalloc+convert
     * and points cuBLASLt straight at the mmap (g_model_device_base+offset),
     * freeing the ~6.4 GiB double-store. For 128-aligned shapes (out%128==0,
     * (in/32)%4==0 -- true for every shipped weight) the total byte size equals
     * the type-38 size exactly, so it shares the {32,33} table entry. */
    PULSAR_TENSOR_MXFP8_LT = 41,
    /* Pre-stored SoA twin of IQ2_XXS (16): identical 66 B/block content with
     * the q and d planes split so the code stream is load-aligned (block_iq2_xxs
     * puts qs[] at offset 2, forcing 2-byte LDG.E.U16 loads).  Byte size, dims
     * and row size are UNCHANGED -- a pure permutation, exactly as MXFP8_LT (41)
     * is to FP8_E4M3 (38).  Layout spec: ds4q_iq2_xxs_soa_repack() in
     * gguf-tools/quants_common.c; device reader: dev_iq2_soa_planes(). */
    PULSAR_TENSOR_IQ2_XXS_SOA = 42,
    /* Pre-stored MMQ *aligned-SoA* twin of IQ2_XXS (16).  Same 66 B/block
     * content as type 16 and type 42, but permuted into the layout the
     * vendored llama.cpp MMQ adapter's kernels read directly:
     *
     *   [ __half d[nblk] ][ pad to 64B ][ uint2 qs[nblk*8] ]
     *
     * i.e. the d plane FIRST and the code plane 64B-aligned.  This is NOT the
     * same permutation as PULSAR_TENSOR_IQ2_XXS_SOA (42), which puts the q
     * plane first with no padding -- the two layouts are not interchangeable
     * and each has its own readers.  Byte size, dims and row size are
     * UNCHANGED (align_up(nblk*2,64) + nblk*64 == nblk*66 whenever
     * nblk % 32 == 0, which holds for every shipped expert stack), so it
     * shares type 16's {256, 66} accounting and mmaps through the generic
     * path, exactly as 42 and MXFP8_LT (41) do.
     *
     * Producer: gguf-tools/repack_iq2_mmq.py builds this aligned layout OFFLINE
     * (the weight server's --repack-iq2-aligned). There is no on-device repack
     * twin any more -- the old ds4_repack_iq2_aligned_device()/ds4_repack.cu was
     * removed; the device side reads the stored artifact DIRECTLY through the
     * MMQ SoA consumers below, so the layout invariant is now repack_iq2_mmq.py
     * <-> those SoA loaders. The size rule is in src/cuda/mmq/ds4_mmq.h (its
     * ds4_mmq_iq2_xxs_aligned_bytes oracle was removed unused, L066 step 2).
     * Consumers: ds4_mmq_iq2_xxs_moe_pair_soa (gate/up) and
     * ds4_mmq_iq2_xxs_moe_soa (down).
     *
     * Storing this layout in the GGUF replaces the runtime repack cache: that
     * cache is capacity-bound (~22.9 GiB budget vs ~35 GB to hold all 90+ IQ2
     * stacks), so it covered only part of the model and made the first prefill
     * frontier absorb the repack.  Pre-storing costs zero model growth. */
    PULSAR_TENSOR_IQ2_XXS_MMQ_K = PULSAR_GPU_TENSOR_IQ2_XXS_MMQ_K,
    PULSAR_TENSOR_FP8_E4M3_SOA_K = PULSAR_GPU_TENSOR_FP8_E4M3_SOA_K, /* one spelling: pulsar_gpu.h */       /* one spelling: pulsar_gpu.h */
};

/** The drafter markov_w2 table's storage, derived from its GGUF type -- the one
 * place the type -> kernel-arm mapping is spelled (L213).  Any other type is a
 * refusal, not a default: the loader's dims contract already rejected it, so
 * reaching here with one is a bug. */
void pulsar_die(const char *msg);   /* declared in full further down; needed here */
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
    uint32_t type;      ///< GGUF type code of the value
    uint64_t value_pos; ///< byte offset of the value within the file
} pulsar_kv;

/** THE accept set for gpu_graph_matmul_plain_tensor -- ONE definition.
 *
 * This lived as three parallel lists: the dispatcher's arms, the load
 * validator's accept set, and a decode-time predicate, each with a comment
 * telling the next person to keep it in step with the other two. They drifted
 * exactly as you would expect -- the validator's comment said "the four arms"
 * while there were three -- and the failure mode is nasty: a type accepted by
 * the validator but missing an arm passes load and dies at runtime on a tensor
 * the artifact was told was fine.
 *
 * The dispatcher still switches, because it must MAP a type to an arm. What it
 * may not do is disagree about membership, so it asserts against this instead
 * of restating it. */
static inline bool pulsar_weight_is_plain_or_mxfp8(uint32_t type) {
    return type == PULSAR_TENSOR_BF16 ||
           type == PULSAR_TENSOR_F32 ||
           type == PULSAR_TENSOR_FP8_E4M3 ||
           type == PULSAR_TENSOR_MXFP8_LT;
}

/** One entry of the GGUF tensor directory: where a tensor lives and how to
 * read it. Describes bytes inside the model mapping; owns nothing. */
typedef struct {
    pulsar_str name;      ///< tensor name as it appears in the GGUF
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
} pulsar_tensor;

/** A memory-mapped GGUF file plus its parsed directory.
 *
 * The mapping stays live for the engine's lifetime and every weight pointer in
 * pulsar_layer_weights is a borrowed view into it -- nothing copies tensor data
 * to the host. */
typedef struct {
    int fd;                 ///< open file descriptor backing the mapping
    const uint8_t *map;     ///< base of the read-only mapping
    uint64_t size;          ///< mapped bytes

    uint32_t version;       ///< GGUF format version
    uint64_t n_kv;          ///< metadata key/value pair count
    uint64_t n_tensors;     ///< tensor directory entry count
    uint64_t alignment;     ///< tensor data alignment declared by the file
    uint64_t tensor_data_pos;   ///< byte offset where the tensor data section starts
    uint64_t max_tensor_bytes;  ///< largest single tensor, for staging-buffer sizing

    pulsar_kv *kv;             ///< parsed metadata pairs, n_kv entries
    pulsar_tensor *tensors;    ///< parsed tensor directory, n_tensors entries
} pulsar_model;

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
    uint64_t off;  ///< first byte, offset into the model mapping
    uint64_t end;  ///< one past the last byte
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
    pulsar_tensor *attn_compressor_kv;    ///< compressor KV projection [n_embd -> head_dim]; FULL layers
    pulsar_tensor *attn_compressor_gate;  ///< compressor softmax gate over the group's rows; FULL layers of ratio > 1
    pulsar_tensor *attn_compressor_norm;  ///< RMSNorm on the pooled latent; FULL layers
    pulsar_tensor *indexer_attn_q_b;      ///< indexer query up-projection (its own head space); FULL + REINDEX layers
    pulsar_tensor *indexer_proj;          ///< indexer per-head score weights; FULL + REINDEX layers
    pulsar_tensor *indexer_k;             ///< index key from the compressor latent [head_dim -> indexer_head_dim]; FULL layers
    pulsar_tensor *indexer_k_norm;        ///< RMSNorm on the index key; FULL layers
    pulsar_tensor *hc_ffn_fn;        ///< HC mix weight feeding the FFN sublayer
    pulsar_tensor *hc_ffn_scale;     ///< HC per-channel scale, FFN side
    pulsar_tensor *hc_ffn_base;      ///< HC per-channel base/offset, FFN side
    pulsar_tensor *ffn_norm;         ///< RMSNorm weight before the FFN
    pulsar_tensor *ffn_gate_inp;     ///< router projection producing per-expert logits
    pulsar_tensor *ffn_exp_probs_b;  ///< router bias added to the expert probabilities
    /** This layer's router width, the experts a token activates, and the experts
     * physically present in its stacks (REAP keep count on the target; the
     * full width on the drafter).  Set at bind; the FFN encoder reads THESE, so
     * a drafter layer routes with its own 128 / top-3 and never with the
     * target's 384 / top-6 (L218 audit risk #2). */
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_present;
    pulsar_tensor *ffn_gate_exps;    ///< ROUTED experts, gate projection (expert-major)
    pulsar_tensor *ffn_up_exps;      ///< routed experts, up projection
    pulsar_tensor *ffn_down_exps;    ///< routed experts, down projection
    pulsar_tensor *ffn_gate_shexp;   ///< SHARED expert gate projection (runs for every token)
    pulsar_tensor *ffn_up_shexp;     ///< shared expert up projection
    pulsar_tensor *ffn_down_shexp;   ///< shared expert down projection
} pulsar_layer_weights;

/** Every weight tensor of the target model: the per-layer stacks plus the
 * embedding and output head. Tensors point into the model mapping; the struct
 * owns none of them. */
typedef struct {
    pulsar_tensor *token_embd;       ///< token embedding table
    pulsar_tensor *output_norm;      ///< final RMSNorm before the vocab projection
    pulsar_tensor *output;           ///< vocab projection (the output head)
    pulsar_layer_weights layer[PULSAR_MAX_LAYER];  ///< per-layer weight stacks
} pulsar_weights;

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
} pulsar_dspark_weights;

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
    explicit pulsar_decode_rows_scope(uint32_t n)
        : saved_(pulsar_gpu_matmul_batch_decode_rows()),
          ok_(pulsar_gpu_matmul_set_batch_decode_rows((int)n) != 0) {}
    ~pulsar_decode_rows_scope() {
        (void)pulsar_gpu_matmul_set_batch_decode_rows(saved_);   /* restoring an accepted value */
    }
    bool ok() const { return ok_; }
    pulsar_decode_rows_scope(const pulsar_decode_rows_scope &) = delete;
    pulsar_decode_rows_scope &operator=(const pulsar_decode_rows_scope &) = delete;
private:
    int  saved_;
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
    uint64_t dspark_raw_bank_bytes;      ///< one bank's drafter raw ring: DRAFT_WINDOW * WINDOW row (528 B)
    uint64_t dspark_prompt_bank_bytes;   ///< one bank's drafter prompt ring: DRAFT_WINDOW * n_embd * f32
    pulsar_gpu_tensor *dspark_raw[3];       ///< per draft layer, bank-major drafter raw ring; NULL without a pool or drafter
    pulsar_gpu_tensor *dspark_prompt[3];    ///< per draft layer, bank-major drafter prompt-hidden ring
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

    /** Speculative decoding scratch.  The drafter is allowed to mutate graph
     * state only if the target verifier can either commit it or restore the
     * saved frontiers. */
    pulsar_gpu_tensor *spec_attn_state_kv[PULSAR_MAX_LAYER];     ///< saved compressor state KV, per ratio>1 kv source
    pulsar_gpu_tensor *spec_attn_state_score[PULSAR_MAX_LAYER];  ///< saved compressor state score, per ratio>1 kv source
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
     * gpu_graph_verify_suffix_tops, and the Tier-2 batched multi-session
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
    pulsar_gpu_tensor *cand_bscore;
    uint32_t           cand_mask_words;
    pulsar_gpu_tensor *ffn_norm;         ///< RMSNorm output feeding the FFN
    pulsar_gpu_tensor *output_embd;      ///< collapsed embedding-width vector
    pulsar_gpu_tensor *output_norm;      ///< final RMSNorm before the vocab projection
    pulsar_gpu_tensor *logits;           ///< vocab logits row (width = pulsar_shape::n_vocab)

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
    /* plan-92 P0 teacher dump (PULSAR_DISTILL_DUMP, env-once at graph alloc):
     * per-position teacher top-64 ids/logits + tail logsumexp for the chunk,
     * filled by the all-rows head sweep after the prefill layer loop and
     * drained alongside dspark_bulk_h. NULL when the mode is off. */
    pulsar_gpu_tensor *distill_top_ids;    /* [prefill_cap, 64] i32 */
    pulsar_gpu_tensor *distill_top_vals;   /* [prefill_cap, 64] f16 bits */
    pulsar_gpu_tensor *distill_tail_lse;   /* [prefill_cap] f16 bits */
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
    pulsar_gpu_tensor *dspark_prefilter_sel;  ///< L149: [16 x PULSAR_DSPARK_PREFILTER_ROW_I32] i32 min-p prefilter output rows
    pulsar_gpu_tensor *dspark_bank_meta;      ///< L150: [2 x PULSAR_DSPARK_BANKS_MAX] i32: per-bank base row, per-bank sampled prev token
    pulsar_gpu_tensor *dspark_row_meta;       ///< L150: [7 x PULSAR_SPEC_LOGITS_ROWS] i32 per-row rope/visibility positions per layer + bank id for the banked drafter forward
    /** L149 phase 2: compact verify rows. spec_round_begin accumulates, over
     * the rounds begun since the last step, whether EVERY one is in the sparse
     * min-p contract and the most permissive floor among them;
     * pulsar_session_spec_arm_capture arms the step from that. An armed
     * ALL_ROWS head then runs the min-p prefilter over its rows and reads the
     * compact block into spec_compact_host INSTEAD of the full logits, and the
     * accept walk builds each target distribution from the row's candidates
     * (a device read of that one row is the per-row fallback). A more negative
     * floor only widens the candidate superset, so the min over rounds is safe
     * for every round. */
    bool     spec_compact_acc_ok;
    uint32_t spec_compact_acc_n;
    float    spec_compact_acc_delta;
    bool     spec_compact_armed;
    float    spec_compact_delta;
    int32_t *spec_compact_host;   ///< PULSAR_SPEC_LOGITS_ROWS x PULSAR_DSPARK_PREFILTER_ROW_I32, owned
    uint32_t spec_compact_rows;   ///< rows [0, spec_compact_rows) hold this step's compact output (0 = none)
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
    /** Scratch for the ratio-4 compressor state rebuild's tail re-projection
     * (<= 8 rows x comp width, both halves).  Its own buffer because the
     * rebuild used to write into batch_comp_kv/_sc rows 0..n_tail-1 while the
     * rewind projection-ring deposit still had to read that chunk's rows
     * n_tokens-8..n_tokens-1 from the same buffer -- for chunks of 5..11
     * tokens the ring received tail tokens filed under head positions (L171). */
    pulsar_gpu_tensor *comp_tail_kv;
    pulsar_gpu_tensor *comp_tail_sc;
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
    /** L218: the last verify round's compressor-projection save span, per
     * bank -- positions [pos0, pos0 + rows) sit at save rows [row0, row0 + rows)
     * of spec_comp_kv/sc_save.  A rewind to an odd position inside the round
     * rebuilds the ratio-2 pending slot from them (pulsar_session::rewind);
     * rows == 0 means no round has saved for this bank. */
    uint32_t ms_spec_save_pos0[PULSAR_MSEQ_MAX];
    uint32_t ms_spec_save_row0[PULSAR_MSEQ_MAX];
    uint32_t ms_spec_save_rows[PULSAR_MSEQ_MAX];
    /** L218: the bank's ratio-2 pending groups do not describe its position
     * (a rewind landed on an odd position outside the saved span).  Only a
     * store at a group boundary makes the state consistent again; a store at
     * any other position on a stale bank refuses (gpu_graph_csa2_produce). */
    bool ms_comp_state_stale[PULSAR_MSEQ_MAX];
    uint32_t ms_dspark_prompt_n[PULSAR_MSEQ_MAX];   ///< drafter prompt-window length held by the bank
    uint32_t ms_dspark_prompt_lo[PULSAR_MSEQ_MAX];  ///< first position of that window
    /** Tier-2 PATH-A partial-prefix KV-reuse (plan-33). fork_pin[bank] is a
     * transient eviction pin so the guard's victim picker cannot free_physical
     * a source bank mid-clone (plan-33 anti-corruption guarantee).  Zero-
     * initialised with the graph. */
    uint8_t  fork_pin[PULSAR_MSEQ_MAX];      ///< transient eviction pin: the guard must not free a bank being cloned

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
} pulsar_gpu_graph;

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
static inline uint32_t &gpu_graph_n_comp(pulsar_gpu_graph *g, uint32_t bank, uint32_t il) {
    return g->ms_n_comp[bank][il];
}
static inline uint32_t gpu_graph_n_comp(const pulsar_gpu_graph *g, uint32_t bank, uint32_t il) {
    return g->ms_n_comp[bank][il];
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
    str_i32_table token_to_id;  ///< token bytes -> id, for the BPE merge loop
    str_i32_table merge_rank;   ///< BPE merge priority; lower rank merges first

    /** ---- methods (C++ port): 1:1 mirror of the vocab verb family in
     * tokenizer.cpp; bodies keep the auto *vocab = this alias, logic verbatim.
     * Names kept as-is (none carry the pulsar_vocab type-name prefix). ---- */
    /** Merge priority of the pair (a, b); lower merges first. @return rank, or a
     * sentinel above every real rank when the pair is not in the merge table. */
    int bpe_rank(const owned_str *a, const owned_str *b) const;
    /** Emit one pre-tokenized piece as ids, running the BPE merge loop over it. */
    void bpe_emit_piece(pulsar_str raw_piece, token_vec *out) const;
    /** Tokenize plain text: pre-tokenize, then BPE-merge each piece. */
    void bpe_tokenize_text(const char *text, token_vec *out) const;
    /** Exact-match lookup of a token's id. @return the id, or -1 if absent. */
    int vocab_lookup(const char *text) const;
    /** Populate the vocabulary from the model's GGUF metadata. */
    void vocab_load(const pulsar_model *model);
    /** Release the vocabulary's owned tables. */
    void vocab_free();
    /** Longest special-token match at `p`. @return true and set token/len on a hit. */
    bool special_token_at(const char *p, int *token, size_t *len) const;
    /** Tokenize `n` bytes at `p`, honouring special tokens found inside the span. */
    void tokenize_span(const char *p, size_t n, token_vec *out) const;
    /** Tokenize an ALREADY-RENDERED chat string -- the caller has applied the
     * template, so role markers appear as literal special tokens. */
    void tokenize_rendered_chat_vocab(const char *text, token_vec *out) const;
    /** Tokenize tool-result content, which may contain sequences that must not be
     * interpreted as chat control tokens. */
    void bpe_tokenize_tool_result_text(const char *content, token_vec *out);
    /** Debug: print ids with their decoded bytes to stderr. */
    void dump_tokens(const token_vec *tokens) const;
};

/** The loaded model and everything derived from it.
 *
 * One engine owns the weights; MANY sessions share it. Everything here is
 * immutable after open() except the cumulative metrics counters -- which is
 * what makes concurrent sessions safe against a single engine. */
struct pulsar_engine {
    pulsar_model model;         ///< the target model's mapping and directory
    pulsar_model dspark_model;  ///< drafter mapping; a distinct file only when dspark_external
    pulsar_vocab vocab;         ///< tokenizer tables and special ids
    pulsar_weights weights;     ///< resolved target tensors, per layer
    pulsar_dspark_weights dspark_weights;  ///< resolved drafter tensors
    pulsar_backend backend;     ///< CPU or CUDA
    int dspark_draft_tokens;    ///< configured draft depth k
    char *directional_steering_file;   ///< steering-vector file path, or NULL
    float *directional_steering_dirs;  ///< loaded steering directions, or NULL
    float directional_steering_attn_scale;  ///< steering strength on the attention stream
    float directional_steering_ffn_scale;   ///< steering strength on the FFN stream
    uint32_t prefill_chunk;     ///< tokens per prefill chunk
    bool gpu_ready;             ///< CUDA backend initialised and weights resident
    bool dspark_ready;          ///< a usable drafter is loaded; false disables speculation
    bool dspark_external;       ///< drafter came from its OWN GGUF (separate map/fd), not the target's
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

    /** ---- methods (C++ port): 1:1 mirror of the pulsar_engine_* verb family.
     * The public API in pulsar.h stays the free-function facade (defined in
     * engine_api.cpp); engine internals call these members directly.  Members
     * stay public and the struct stays trivially constructible: lifetime is
     * managed exactly as before via open()/destroy() (xcalloc/free), NOT
     * constructors/destructors.
     * NOTE: pulsar_engine_dspark_draft_tokens stays a free function — a member
     * would collide with the data member of the same name. */
    static int open(pulsar_engine **out, const pulsar_engine_options *opt);
    void destroy();  ///< was pulsar_engine_close
    /** Print a human-readable model summary (shape, quantisation, memory) to
     * stderr. */
    void summary();
    /** Tokenizer table length. NOT the logits width -- see logits_width(). */
    int vocab_size();
    /** Logits row width (the shape profile's n_vocab). Size every logits buffer
     * with THIS. @return the row stride, in floats. */
    int logits_width() const;
    /** Model name from the GGUF metadata. */
    const char *model_name();
    /** Copy out the engine-cumulative speculative-decode counters. */
    void spec_metrics(pulsar_spec_metrics *out);
    /** Stable id for this model, for cache keys and metrics labels. */
    int model_id();
    /** True when the artifact is a REAP-pruned expert set rather than the full
     * model -- the two have different expert counts and cannot share caches. */
    bool is_pruned() const;
    /** GPU bytes pulsar_session::create takes at `ctx_size` with the current
     * bank pool: the allocation code run dry, so the price and the allocation
     * are one function.  The number admission control must use.
     * @return bytes, or 0 if no session could be created. */
    uint64_t session_cost_bytes(int ctx_size);
    /** session_cost_bytes() for an explicit bank-pool size, so the server can
     * evaluate the (banks, ctx) fit table before committing to one.
     * @param ctx_size context size to price
     * @param n_banks  >= 1; 1 is the classic single-session layout
     * @return bytes, or 0 if no session could be created. */
    uint64_t session_cost_bytes_banked(int ctx_size, int n_banks);
    /** Demand-paged (not reserved) bytes ONE bank actually materialises at
     * `ctx_size` -- the overcommit figure, below the reserved capacity. */
    uint64_t demand_paged_bytes_per_bank(int ctx_size);
    /** Resident weight bytes, excluding per-session state. */
    uint64_t weights_resident_bytes();
    /** One-shot greedy generation: prefill `prompt`, then decode argmax until
     * `n_predict` tokens or EOS, delivering each through `emit`. The
     * self-contained path the CLI and the diagnostics use, with no session
     * management for the caller to do. @return 0 on success. */
    int generate_argmax(const pulsar_tokens *prompt,
                        int n_predict, int ctx_size,
                        pulsar_token_emit_fn emit,
                        pulsar_generation_done_fn done,
                        void *emit_ud,
                        pulsar_session_progress_fn progress,
                        void *progress_ud);
    /** Run the dataset through the model accumulating per-tensor activation
     * magnitudes, and write the importance matrix used to steer quantisation.
     * @return 0 on success. */
    int collect_imatrix(const char *dataset_path, const char *output_path,
                        int ctx_size, int max_prompts, int max_tokens);
    /** Debug: print ids with decoded text to stderr. */
    void dump_tokens(const pulsar_tokens *tokens);
    /** Bits per weight of the ROUTED expert tensors (the artifact's dominant
     * quantisation), for reporting and tier selection. */
    int routed_quant_bits();
    /** True when a usable drafter is loaded and speculation can run. */
    bool has_dspark();
};

/** A string this struct OWNS, as opposed to ::pulsar_str which borrows.
 * Not NUL-terminated either -- the length is authoritative. */
typedef struct owned_str {
    char *ptr;     ///< the bytes, owned
    uint64_t len;  ///< length in bytes
} owned_str;

/** One token under consideration by the sampler. Carries both the raw logit
 * and the normalised probability because the filters need different ones --
 * top-k sorts on the logit, min-p compares probabilities. */
typedef struct {
    int id;       ///< token id
    float logit;  ///< raw logit
    float prob;   ///< probability after softmax over the candidate set
} sample_candidate;

/** Reusable working set for pulsar_sample_dist_build's full-vocab (top_k <= 0)
 * path, which sorts all n_vocab candidates. Zero-initialize before first use;
 * grows on demand and is reused across calls, so the sampled speculative walk
 * does not malloc/free ~1.5 MB per accepted position. Free with
 * pulsar_sample_scratch_free.
 *
 * Caller-owned and NOT shared: each concurrent session runs its own sampled
 * acceptance walk, so this lives on pulsar_session, never on pulsar_engine. */
typedef struct {
    sample_candidate *cand;  ///< sorted candidates
    uint64_t *keys;  ///< packed (sort key << 32 | id)
    uint64_t *tmp;  ///< radix ping-pong buffer
    /** Gather target for the min-p prefilter path: survivors are collected
     * into `cand` in ascending-id order (probs computed once, alongside the
     * full-vocab sum), then gathered here in descending sort order. A second
     * buffer because the gather cannot run in place and the degenerate
     * all-equal-logits case keeps every candidate (m == cap). */
    sample_candidate *cand2;
    uint32_t cap;  ///< elements reserved in each
    /** Dense token->q(prob) map for pulsar_sample_dist_draw_residual, which needs
     * q(x) for each x in p's support: the linear pulsar_sample_dist_prob scan
     * would make that O(|p|*|q|) — 1.6e10 at full vocab. Sized by token id
     * (NOT by `cap`, which is top_k on the preselect path while ids still run
     * to n_vocab), and INVARIANT: all-zero on entry and on exit. The residual
     * draw scatters q in, reads, then re-zeros only q's own ids, so the clear
     * is O(|q|) rather than a full-vocab memset. */
    float *qmap;
    uint32_t qmap_cap;  ///< entries allocated in qmap; sized by n_vocab, not by `cap`
} pulsar_sample_scratch;

void pulsar_sample_scratch_free(pulsar_sample_scratch *s);


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
 * Deliberately EXCLUDED: heap-backed state (checkpoint, logits,
 * dspark_pending_qrows) — those need deep copies with per-side ownership, so
 * they stay as explicit members of each struct. */
/** L149: device min-p prefilter (pulsar_gpu_minp_prefilter_rows) output row:
 * i32 [0] candidate count, [1] max id, [2] max logit bits, then
 * PULSAR_DSPARK_PREFILTER_CAP ids and PULSAR_DSPARK_PREFILTER_CAP f32 logits
 * (ascending id). A count above the cap means "too many survivors": the host
 * reads the full row instead. Rows are stored back-to-back. */
#define PULSAR_DSPARK_PREFILTER_CAP 2048u
#define PULSAR_DSPARK_PREFILTER_ROW_I32 (3u + 2u * PULSAR_DSPARK_PREFILTER_CAP)
/** L149: widest proposal distribution stored per pending draft position. */
#define PULSAR_DSPARK_QDIST_CAP 256u

typedef struct pulsar_spec_carry_state {
    /** Fused DSpark loop (P2): drafts produced LAST step from the last-accepted
     * position's hidden, pending verification in THIS step's single batched
     * forward (EAGLE pipeline inversion). 0 pending = next step is a plain
     * n=1 forward. Invalidated on rewind/invalidate. */
    int32_t dspark_pending[16];
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
    uint32_t dspark_n_pending;   ///< drafts proposed and awaiting verification
    /** The base token the pending drafts continue from (predicted greedy next).
     * If the caller's next first_token differs (non-greedy interruption, tool
     * injection), the pending drafts are stale and dropped. */
    int32_t dspark_pending_base;
    /** checkpoint.len the drafts were produced at — an ACCEPTANCE guard, not an
     * exactness guard. The base-token check above is a VALUE check, not an
     * identity check: a plain pulsar_session_eval (tool injection, think-tag
     * recovery) advances the session and clears the carry but leaves the
     * pendings, so a later first_token that merely COLLIDES with
     * dspark_pending_base would resurrect drafts conditioned on a different
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
    int32_t dspark_pending_pos;
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
    float   dspark_pending_conf[16];
    /** L107 adaptive draft depth: the session's CURRENT draft depth, moved
     * +/-1 per round by the controller in spec_round_end from the realized
     * accept count and the verified tail confidence. 0 = uninitialized (first
     * draft reads the engine's --dspark-draft value, which is thereby the
     * STARTING depth, not a fixed width). Persists across requests in a
     * session on purpose: a client's workload regime usually does too. */
    int spec_adaptive_depth;  ///< bounds: PULSAR_SPEC_DEPTH_{MIN,MAX} below the struct
    bool spec_depth_down_forgiven;  ///< L107 v2: one down-signal was vetoed on a still-confident tail; a second consecutive one backs off regardless
    uint8_t spec_depth_rounds_since_up;  ///< L107 v5: rounds since the last UP, saturating at 255; a down within 2 of an up is a FAILED EXCURSION and triggers the cooldown; a down after a sustained ride carries no penalty (v4's blanket cooldown cost ~0.9 t/s on BOTH server workloads by suppressing profitable climbs).
    uint8_t spec_depth_climb_cooldown;  ///< L107 v4: rounds remaining in which UP is suppressed after a failed excursion. Raw-completion prose oscillated 2->3->4-> crash forever (29 transitions/192 tok, -14%): each failed excursion burns a deep round, and tail conf does NOT separate good climbs from bad (a 0.93 tail climbed into commit=0). Cooldown makes excursions rare after they fail; structured's downs are rare (and v3-forgiven rounds are not downs), so its climb is untouched.
    /** --- Temperature-matched draft sampling (spec-decode Item 1) ---
     * At temperature > 0 the drafts above are DRAWN from a temperature-matched
     * proposal q (the drafter's refined logits filtered at the request's
     * params) rather than taken as the drafter's argmax. Verification then uses
     * the standard sampled-proposal rule — accept w.p. min(1, p/q), else draw
     * the residual (p-q)+ — which is not capped at p(mode) the way the
     * deterministic-proposal rule is.
     *
     * `dspark_pending_sampled` records which rule the pendings were PROPOSED
     * under, so verify applies the matching rule: false => argmax proposal =>
     * the deterministic rule (accept w.p. p(x), residual p-excluding). The two
     * rules are not interchangeable; applying p/q to an argmax proposal (or
     * vice versa) silently breaks exactness. */
    bool dspark_pending_sampled;
    /** q(pend[i]) at draft time — the accept denominator. */
    float dspark_pending_q[16];
    /** L149: the proposal distribution q_i exactly as BUILT at draft time,
     * kept for the residual draw. dspark_pending_qn[i] == 0 means "not
     * stored": rebuild it from the full row in dspark_pending_qrows under the
     * pending params, as before. When stored it IS that rebuild (same inputs,
     * same params, deterministic build), so the walk skips both the 517 KB
     * row read and the rebuild. Carried by value with the rest of the shadow. */
    uint32_t dspark_pending_qn[16];
    int32_t  dspark_pending_qids[16][PULSAR_DSPARK_QDIST_CAP];
    float    dspark_pending_qprobs[16][PULSAR_DSPARK_QDIST_CAP];
    /** The sampling params the pendings were sampled under. TWO consumers, and
     * they are different in kind:
     *   1) EXACTNESS (load-bearing): the verify walk rebuilds the rejecting
     *      position's q from dspark_pending_qrows using THESE params, so the
     *      stored accept denominator dspark_pending_q[i] and the residual's q
     *      name the same proposal q_X by construction. This is why the rebuild
     *      must never be fed the live request params.
     *   2) THROUGHPUT (the guard in the verify path): drafts sampled under X and
     *      verified under Y are still exact — the rule is proposal-agnostic and
     *      returns exactly p_Y for any q — but q_X is a badly-matched proposal
     *      for p_Y, so acceptance craters. The guard drops them to avoid wasting
     *      verify rows, not to avoid bias.
     * Mirrors the spec_carry_* params guard. Greedy never needed this. */
    float dspark_pending_temp;   ///< temperature the pending drafts were proposed under
    float dspark_pending_top_p;  ///< top-p the drafts were proposed under
    float dspark_pending_min_p;  ///< min-p the drafts were proposed under
    int   dspark_pending_top_k;  ///< top-k the drafts were proposed under; the verify guard drops mismatched drafts
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
     * The controller reads only (commit, n_batch) — counts, never wall-clock —
     * so the quench decision is deterministic run-to-run for a fixed stream.
     * Multiseq note: this state belongs to the classic single-request flow;
     * the dormant multi-bank driver would need per-bank copies (not wired —
     * generate_speculative already refuses when mseq_dirty). */
    float spec_quench_debt;  ///< cumulative plain-token-equivalents lost
    float spec_quench_ewma;     ///< EWMA of the per-step margin (realized yield - breakeven guard)
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
 * authority for every "pendings are stale" reset -- setting dspark_n_pending
 * to 0 by hand while a chain is in flight leaves a flag that would resurrect
 * the stale ids at the next harvest. */
static inline void pulsar_spec_drop_pendings(pulsar_spec_carry_state *sp) {
    sp->dspark_n_pending = 0;
    sp->dspark_chain_unharvested = false;
}

/** L108 P2: read back an in-flight device-chained draft (ids + conf), apply
 * the conf-sched trim, and populate the pendings. Idempotent; no-op when no
 * chain is in flight. Must run before anything consumes dspark_n_pending /
 * dspark_pending[], and before a bank save copies spec state (banks share
 * the session's graph tensors, so a saved unharvested flag would harvest
 * another bank's chain). Defined in session_spec.cpp. */
void pulsar_session_spec_chain_harvest(pulsar_session *s);

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
typedef struct pulsar_bank_carry {
    bool      valid;  ///< has this bank's state ever been saved
    /** heap-backed (owned): */
    token_vec checkpoint;  ///< deep copy of s->checkpoint
    float    *logits;  ///< PULSAR_N_VOCAB floats, owned
    float    *dspark_pending_qrows;  ///< dspark_pending_qrows_cap floats, owned
    uint32_t  dspark_pending_qrows_cap;  ///< floats allocated in dspark_pending_qrows
    /** scalar mirrors: */
    bool      checkpoint_valid;
    int       prefill_frontier;  ///< L195: mirror of pulsar_session::prefill_frontier
    /** Whole speculative/DSpark shadow, mirrored by value (single assignment in
     * save/restore).  NOTE: pulsar_session.mseq_dirty is deliberately NOT carried:
     * it is a property of the GRAPH's scalar frontier counters, not of a bank's
     * conversation, and _restore re-establishes per-bank frontier truth via
     * gpu_graph_bank_counters_install and then clears it unconditionally.  The
     * old mirror field was write-only (saved, never read) and has been dropped. */
    pulsar_spec_carry_state spec;
    void copy(const pulsar_bank_carry *sc);  ///< was bank_carry_copy
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
struct pulsar_session {
    pulsar_engine *engine;    ///< borrowed; the engine outlives every session
    pulsar_gpu_graph graph;   ///< this session's device state (KV, scratch, bank views)
    token_vec checkpoint;     ///< tokens whose KV the graph currently holds, current bank
    float *logits;            ///< last decoded row, pulsar_engine_logits_width() floats
    /** Reused working set for the sampled speculative acceptance walk's
     * full-vocab distribution builds (one per accepted position). Per-session,
     * never shared: concurrent sessions each run their own walk. */
    pulsar_sample_scratch sample_scratch;
    /** Reusable PULSAR_N_VOCAB-float staging row for the speculative paths'
     * spec_logits readbacks.  ~517 KB, i.e. above glibc's mmap threshold, so a
     * per-step malloc/free would mmap/munmap and re-fault it every accepted
     * position — allocate once per session instead.  Deliberately NOT drawn
     * from sample_scratch: sample_scratch_reserve frees the whole struct when
     * it grows, which would dangle a row held across a pulsar_sample_dist_build.
     * Only ever live inside one speculative eval call (the fused and block
     * paths never overlap: block tail-calls fused). */
    float *spec_row_scratch;
    pulsar_session_progress_fn progress;   ///< durable progress callback: fires at real checkpoint boundaries
    void *progress_ud;                     ///< user data for progress
    pulsar_session_progress_fn display_progress;  ///< UI-only progress; may fire MID-chunk, so not a checkpoint boundary
    void *display_progress_ud;             ///< user data for display_progress
    pulsar_session_cancel_fn cancel;       ///< cooperative cancellation hook, polled at safe points only
    void *cancel_ud;                       ///< user data for cancel
    uint32_t prefill_cap;                  ///< max tokens per prefill chunk for this session
    int ctx_size;                          ///< allocated context length, in tokens
    bool checkpoint_valid;                 ///< false when `checkpoint` no longer describes the graph's KV (forces a rebuild on the next sync)
    int resume_origin;                     ///< L194 instrument: the position the last sync's resume started evaluating from (a grid point, 0 = cold from the start), -1 when the sync did not resume
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
    /** The refined-logits row each pending draft was sampled from, 16 rows of
     * PULSAR_N_VOCAB floats, allocated lazily on first sampled draft. The residual
     * needs the FULL q, but only for the single rejected position — unknown
     * until verify — so every position's row is persisted and the one that
     * rejects rebuilds its q via pulsar_sample_dist_build.
     *
     * Rebuilding from these PERSISTED logits is bit-identical to the draft-time
     * q: dist_build is a pure function of (logits, params), and the rebuild is
     * handed BOTH persisted halves — this row and dspark_pending_temp/top_k/
     * top_p/min_p below. Feeding it either half from live state is the trap: the
     * live drafter state has advanced, and the live REQUEST params may differ
     * from the draft-time ones, which would leave the stored accept denominator
     * dspark_pending_q[i] (computed under the draft-time params) and the
     * residual's q describing two different proposals inside one rule.
     *
     * Device-first note (Item 2): storing the logits ROW + params rather than a
     * materialized nucleus is deliberate — the GPU accept kernel wants exactly
     * this, so it swaps this host pool for a resident device buffer instead of
     * reshaping the format. */
    float *dspark_pending_qrows;
    uint32_t dspark_pending_qrows_cap;  ///< floats reserved
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
     * pulsar_session_rewrite_requires_rebuild / _write_staged_payload /
     * _payload_file_free / _snapshot_free do not take a session and stay free. */
    static int create(pulsar_session **out, pulsar_engine *e, int ctx_size);
    /** Tear down the session and release its GPU allocations. Behind pulsar_session_free(). */
    void destroy();
    /** Install the durable progress callback. Behind pulsar_session_set_progress(). */
    void set_progress(pulsar_session_progress_fn fn, void *ud);
    /** Install the UI-only progress callback -- may fire mid-chunk, so it is NOT
     * a durable KV checkpoint boundary. Behind pulsar_session_set_display_progress(). */
    void set_display_progress(pulsar_session_progress_fn fn, void *ud);
    /** Install the cooperative cancellation hook, checked only at safe
     * boundaries. Behind pulsar_session_set_cancel(). */
    void set_cancel(pulsar_session_cancel_fn fn, void *ud);
    /** Copy out the cumulative speculative-decode counters. */
    /** Resident KV bytes actually touched by the CURRENT bank -- the demand-paged
     * figure, which is below the reserved capacity on a short session. */
    uint64_t touched_kv_bytes() const;

    /* ---- Tier-2 bank pool: physical residency, spill, fork ---------------- */

    /** Release one idle bank's ctx-scaled physical pages (cudaFree on its managed
     * comp/index allocations). The only reclaim primitive that actually returns
     * memory on GB10. The bank keeps its logical identity and can be re-armed
     * with bank_alloc_physical(). @return false if the bank is live or pinned. */
    bool bank_free_physical(uint32_t bank);
    /** Re-allocate physical for a previously freed bank, before installing it. */
    bool bank_alloc_physical(uint32_t bank);
    /** True when the bank's physical has been released and its KV must be
     * reloaded from disk before use. */
    bool bank_is_evicted(uint32_t bank) const;
    /** touched_kv_bytes() for an arbitrary bank, live or idle. */
    uint64_t bank_touched_kv_bytes(uint32_t bank);
    /** Write one bank's KV to `fp` bit-identically (the spill half of the
     * eviction guard). @return 0 on success. */
    int bank_kv_save(uint32_t bank, FILE *fp, char *err, size_t errlen);
    /** Reload a spilled bank's KV from `fp`, byte-for-byte. @return 0 on success. */
    int bank_kv_load(uint32_t bank, FILE *fp, char *err, size_t errlen);
    /** Extra bytes ONE bank would demand-page in if the context grew to quantum
     * `q` -- the admission question "can this session take another step" priced
     * before committing to it. */
    uint64_t quantum_growth_bytes_per_bank(uint32_t q);
    /** FULL-prefix fork: clone `src`'s committed KV into `dst` and continue
     * there, leaving the trunk intact for other siblings. This is what makes a
     * branching conversation cheap -- the shared history is copied, not
     * recomputed.
     * @param src       bank to clone FROM; left intact
     * @param dst       bank to clone INTO
     * @param tokens    the full prompt the forked bank should hold
     * @param n_tokens  its length
     * @param n_cached  how many of `tokens` are already committed in src
     * @return tokens reused, negative on failure. */
    int bank_fork(uint32_t src, uint32_t dst, const int *tokens, int n_tokens, int n_cached);
    /** Is `bank` pinned against eviction because a fork is cloning from it?
     * The guard's victim picker must not free physical pages mid-clone. */
    bool bank_fork_pinned(uint32_t bank) const;
    /** PARTIAL-prefix fork: clone only the shared prefix, cut at a ratio-4
     * boundary, and replay the rest.
     * @param src       bank to clone FROM
     * @param dst       bank to clone INTO
     * @param tokens    the full prompt the forked bank should hold
     * @param n_tokens  its length
     * @param n_cached  shared prefix length to cut at
     * @return tokens reused, negative on failure. */
    int bank_fork_partial(uint32_t src, uint32_t dst, const int *tokens, int n_tokens, int n_cached);
    /** Would a partial fork from `src` at `n_cached` reuse enough to be worth
     * it? @return the reusable token count, 0 when a cold prefill is better. */
    int bank_fork_partial_feasible(uint32_t src, int n_cached);
    /** Bring the session's KV in line with `prompt`: reuse the common prefix and
     * evaluate the rest. The main prefill entry point. @return 0 on success. */
    int sync(const pulsar_tokens *prompt, char *err, size_t errlen);
    /** Rewrite the session to `prompt` given an already-computed `common`
     * prefix length, rather than re-deriving it. */
    pulsar_session_rewrite_result rewrite_from_common(const pulsar_tokens *prompt, int common,
                                                      char *err, size_t errlen);
    /** Longest common TOKEN prefix between the session's history and `prompt`.
     * For the byte-level, seam-aware answer use prefix_match(). */
    int common_prefix(const pulsar_tokens *prompt);
    /** L115: the prefix-reuse authority (see pulsar.h). */
    void prefix_match(const pulsar_tokens *prompt, pulsar_prefix_match *out);
    /** Highest-scoring token in the session's current logits row. */
    int argmax();
    /** argmax() ignoring one id -- used to keep a forced continuation off EOS. */
    int argmax_excluding(int excluded_id);
    /** Sample from the current logits with the given knobs. `rng` is advanced. */
    int sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
    /** Fill `out` with the k highest-logprob candidates. @return count written. */
    int top_logprobs(pulsar_token_score *out, int k);
    /** Score one specific token from the current logits. */
    int token_logprob(int token, pulsar_token_score *out);
    /** Copy the logits row out. Size `cap` with pulsar_engine_logits_width(),
     * NOT with the tokenizer's vocab size. */
    int copy_logits(float *out, int cap);
    /** Overwrite the session's logits row (replay/testing). */
    int set_logits(const float *logits, int n);
    /** Decode ONE token at the session's current position and update the logits.
     *
     * Since L130 this runs the same batched step the server uses, as a 1-row
     * batch on the session's own bank -- so there is no separate single-token
     * kernel path any more. Fails loud when the session's per-bank state is
     * stale (see mseq_dirty) rather than decoding against another bank's rows.
     * @return 0 on success. */
    int eval(int token, char *err, size_t errlen);
    /** Decode ONE row per request across `n` banks in a single batched step.
     * Each row lands at its own bank's frontier.
     * @param reqs        one entry per participating bank
     * @param n           entries in `reqs`
     * @param logits      receives one row per request, in `reqs` order
     * @param logits_cap  floats per row
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
     * @param logits_cap     floats per row
     * @param out_n_rows     rows actually produced
     * @param max_head_runs  caps how many separate output-head runs the step performs
     * @param err            failure message buffer
     * @param errlen         its size
     * @return 0 on success. */
    int decode_mixed(const pulsar_multiseq_req *reqs, uint32_t n_rows,
                     float *logits, int logits_cap, uint32_t *out_n_rows,
                     uint32_t max_head_runs, char *err, size_t errlen);
    /** Release the host-side per-bank carry (checkpoints, logits, pendings). */
    void bank_carry_free();
    /** Number of banks in the pool; 1 when the pool is disabled. */
    int bank_count();
    /** Point the graph's device views at `bank` and set cur_bank. Does NOT move
     * host state -- bank_state_restore() is the full hand-off. */
    int bank_repoint(uint32_t bank);
    /** Publish the live host+frontier state into `bank`'s slots. Callers pass the
     * bank that is currently installed; the server does this when switching AWAY,
     * which is what keeps idle banks' carry readable. */
    void bank_state_save(uint32_t bank);
    /** Install `bank`: repoint device views, then restore its host carry. Clears
     * the multiseq-poison flag. @return false if the bank cannot be installed. */
    bool bank_state_restore(uint32_t bank);
    /** Committed token count for `bank`, live or idle. */
    int bank_pos(uint32_t bank);
    /** Adaptive draft depth for `bank`, read from the live state or its carry. */
    int bank_spec_depth(uint32_t bank);
    /** Borrowed view of `bank`'s committed token history. Do not free. */
    const pulsar_tokens *bank_tokens(uint32_t bank);
    /** Longest common token prefix between `bank`'s history and `prompt`. */
    int bank_common_prefix(uint32_t bank, const pulsar_tokens *prompt);
    /** Byte-level prefix match against `bank` -- the seam-aware form that reports
     * live-side and prompt-side cuts separately (see pulsar_prefix_match). */
    void bank_prefix_match(uint32_t bank, const pulsar_tokens *prompt, pulsar_prefix_match *out);
    /** Append tokens to the session's checkpoint WITHOUT decoding them: for
     * callers that committed rows through a batched step and must now bring the
     * host history back in line with the KV. */
    void note_committed_tokens(const int *toks, int n);
    /** Generate with the drafter: propose a block, verify it against the target
     * in one pass, and commit the accepted prefix.
     * @param temperature   sampling temperature; 0 selects argmax
     * @param top_k         top-k cutoff
     * @param top_p         nucleus cutoff
     * @param min_p         relative probability floor
     * @param rng           sampler state; advanced by this call
     * @param max_tokens    cap on tokens committed
     * @param eos_token     stop once this is committed
     * @param accepted      receives the committed token ids
     * @param accepted_cap  its capacity
     * @param err           failure message buffer
     * @param errlen        its size
     * @return tokens committed. */
    int generate_speculative(float temperature, int top_k, float top_p, float min_p,
                             uint64_t *rng, int max_tokens, int eos_token,
                             int *accepted, int accepted_cap, char *err, size_t errlen);
    /** Speculative generation seeded with a known `first_token` -- the forced-
     * continuation form, where the caller has already chosen the opening token.
     * @param first_token   the opening token, committed as-is
     * @param max_tokens    cap on tokens committed
     * @param eos_token     stop once this is committed
     * @param accepted      receives the committed token ids
     * @param accepted_cap  its capacity
     * @param err           failure message buffer
     * @param errlen        its size
     * @return tokens committed. */
    int eval_speculative_block(int first_token, int max_tokens, int eos_token,
                               int *accepted, int accepted_cap, char *err, size_t errlen);
    /** Discard the conversation: clear the checkpoint, drop spec carry and
     * pendings, and disarm the rewind rings so a NEXT conversation can never
     * restore this one's rows. The graph keeps its allocations. */
    void invalidate();
    /** Undo back to `pos` tokens: trim the checkpoint, drop speculative state,
     * and clamp every compressing layer's frontier to pos/ratio.
     *
     * The clamp is unconditional and is what keeps the next step admissible.
     * A second, best-effort half restores the VALUES of re-emitted compressed
     * rows from the projection ring -- only when the ring covers the rewound
     * span, which on the served (multiseq) path it does not. */
    void rewind(int pos);
    /** Committed token count for the current bank. */
    int pos();
    /** Allocated context length, in tokens. */
    int ctx();
    /** Smallest suffix worth prefilling as its own chunk: below this the fixed
     * per-chunk cost dominates and the caller should extend instead. */
    uint32_t prefill_quantum_min_suffix() const;
    /** Borrowed view of the committed token history. Do not free. */
    const pulsar_tokens *tokens();
    /** Serialized size of this session's payload, in bytes. */
    uint64_t payload_bytes();
    /** Write the payload to a fresh file under `stage_dir` and report where it
     * landed -- the disk-KV store's write path. @return 0 on success. */
    int stage_payload(pulsar_session_payload_file *out, const char *stage_dir,
                      char *err, size_t errlen);
    /** Write the payload to an open stream. @return 0 on success. */
    int save_payload(FILE *fp, char *err, size_t errlen);
    /** Read a payload back, replacing this session's state. Refuses a payload
     * whose version or shape does not match this build. @return 0 on success. */
    int load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
    /** Capture the session into an owned in-memory blob. @return 0 on success. */
    int save_snapshot(pulsar_session_snapshot *snap, char *err, size_t errlen);
    /** Restore the session from a snapshot taken by save_snapshot().
     * @return 0 on success. */
    int load_snapshot(const pulsar_session_snapshot *snap, char *err, size_t errlen);
};

/** Snapshot of every layer's compressed-row frontier, taken before a
 * speculative block so a rejected draft can be rolled back exactly. */
typedef struct {
    uint32_t n_comp[PULSAR_MAX_LAYER];        ///< compressed rows per kv source
} pulsar_spec_frontier;

/** Userdata wrapping a caller's progress callback during sync().
 *
 * The engine reports absolute positions; the wrapper holds the session and
 * prompt so it can hand the caller's own callback the context it needs without
 * the engine having to know about it. */
typedef struct {
    pulsar_session *session;            ///< session being synced
    const pulsar_tokens *prompt;        ///< prompt being synced to
    pulsar_session_progress_fn user;    ///< the caller's callback
    void *user_ud;                      ///< the caller's userdata
} pulsar_sync_progress;

/** ---- helpers shared across the session_*.cpp TUs ----
 * payload_set_err (session_payload.cpp) is the payload/bank-KV error stamper;
 * spec_quench_reset (session_spec.cpp) re-arms the terminal yield quench at
 * request boundaries (sync/invalidate/rewind/load_payload). */
void payload_set_err(char *err, size_t errlen, const char *msg);
/** Re-arm at request boundaries (the same sites that drop the carry and
 * pendings). All-zero == armed, matching the xcalloc'd session.
 */
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
 *          unchanged; owns nothing beyond its window ring and its q path. */
typedef enum {
    PULSAR_ATTN_WINDOW = 0,
    PULSAR_ATTN_FULL,
    PULSAR_ATTN_REINDEX,
    PULSAR_ATTN_REUSE,
} pulsar_attn_mode;

/** One layer's row of the attention layout table.  Derived once at load from
 * the artifact's compress_ratios / kv_source_layers / index_source_layers /
 * candidate_source_layer by pulsar_attn_layout_install(), which is the only
 * writer; every cache-ownership and weight-binding decision reads THIS. */
typedef struct {
    uint32_t ratio;         ///< tokens per compressed row; 0 = window only
    uint32_t kv_source;     ///< layer whose compressed KV + index K this layer reads (itself when FULL; PULSAR_NO_LAYER at ratio 0)
    uint32_t index_source;  ///< layer whose top-k this layer attends with (itself when FULL/REINDEX; PULSAR_NO_LAYER at ratio 0)
    pulsar_attn_mode mode;
    bool candidate_source;  ///< this layer's indexer publishes the candidate block mask
    bool uses_candidates;   ///< this layer's indexer scores only inside the published mask
} pulsar_layer_attn;

/** ---- shared globals ---- */

extern const pulsar_shape PULSAR_SHAPE_FLASH;
extern pulsar_shape g_pulsar_shape;
/** REAP ds4-compact-v1: per-layer count of physically-present routed experts.
 * 0 means "not set" -> falls back to n_expert (the un-pruned default). The
 * router/bias tensors stay padded to n_expert (256); only the expert weight
 * tensors are dense-trimmed to this count. Read from reap.layer.keep_count. */
extern uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER];
extern int g_pulsar_lock_fd;

/** ---- shared functions ---- */

bool pulsar_backend_uses_graph(pulsar_backend backend);
void pulsar_die(const char *msg);
/** Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape.
 */
uint32_t pulsar_layer_compress_ratio(uint32_t il);
/** The layer's row of the CSA2 attention layout table (ratio, sources, mode). */
const pulsar_layer_attn *pulsar_layer_attn_layout(uint32_t il);
/** Build the attention layout table from a per-layer ratio array and the
 * source sets, asserting the shape profile's expectation and the CSA2
 * invariants (sources ascending and inside the backbone, a member's ratio
 * equals its source's, every kv source is an index source, the candidate
 * source is an index source).  The loader calls it with the artifact's
 * metadata; a unit test with the profile's own sets.  Dies on any violation. */
void pulsar_attn_layout_install(const uint32_t *ratios,
                                const uint32_t *kv_sources, uint32_t n_kv,
                                const uint32_t *index_sources, uint32_t n_index,
                                int32_t candidate_source);

/** CSA2 (L218) ownership.  The compressed pool, the index-K pool, the
 * compressor state and the frontier a layer reads all live at its kv SOURCE's
 * index; a layer that is its own source (mode FULL) writes them.  Every
 * per-layer cache array in pulsar_gpu_graph / pulsar_bank_slabs is indexed by
 * the source, so a consumer resolves through these and never through `il`. */
static inline uint32_t gpu_graph_kv_source(uint32_t il) {
    return pulsar_layer_attn_layout(il)->kv_source;
}
static inline bool gpu_graph_layer_is_kv_source(uint32_t il) {
    return pulsar_layer_attn_layout(il)->mode == PULSAR_ATTN_FULL;
}
/** A kv source of ratio > 1 keeps a pending group in the state lane; ratio 1
 * emits a row per token and keeps none. */
static inline bool gpu_graph_layer_has_comp_state(uint32_t il) {
    const pulsar_layer_attn *a = pulsar_layer_attn_layout(il);
    return a->mode == PULSAR_ATTN_FULL && a->ratio > 1u;
}
/** Physically-present routed-expert count for a layer. For an un-pruned model
 * (or any layer whose keep_count was not set) this is the full n_expert; for a
 * REAP ds4-compact-v1 model the pruned layers report their dense survivor
 * count. Only the expert *weight* tensors are trimmed to this; the router and
 * bias stay padded to n_expert.
 */
uint32_t pulsar_layer_n_expert(uint32_t il);
void pulsar_die_errno(const char *what, const char *path);
bool pulsar_streq(pulsar_str s, const char *z);
bool pulsar_str_eq(pulsar_str a, pulsar_str b);
uint64_t hash_bytes(const void *ptr, uint64_t len);
void *xcalloc(size_t n, size_t size);
void *xmalloc(size_t size);
char *pulsar_strdup(const char *s);
void *xrealloc(void *ptr, size_t size);
double now_sec(void);
bool write_f32_binary_file(const char *path, const float *data, uint64_t n);
bool read_f32_binary_file(const char *path, float *data, uint64_t n);
void cursor_error(pulsar_cursor *c, const char *msg);
bool cursor_read(pulsar_cursor *c, void *dst, uint64_t n);
bool cursor_skip(pulsar_cursor *c, uint64_t n);
bool cursor_u32(pulsar_cursor *c, uint32_t *v);
bool cursor_u64(pulsar_cursor *c, uint64_t *v);
bool cursor_string(pulsar_cursor *c, pulsar_str *s);
uint64_t align_up(uint64_t value, uint64_t alignment);
const gguf_type_info *tensor_type(uint32_t type);
const char *tensor_type_name(uint32_t type);
void cutlass_mxfp4_expert_layout(uint64_t k, uint64_t n,
                                  uint64_t *data_bytes, uint64_t *sf_bytes,
                                  uint64_t *stride);
pulsar_cursor cursor_at(const pulsar_model *m, uint64_t pos);
bool model_get_u32(const pulsar_model *m, const char *key, uint32_t *out);
bool model_get_u64_compat(const pulsar_model *m, const char *key, uint64_t *out);
bool model_get_f32_compat(const pulsar_model *m, const char *key, float *out);
bool model_get_bool(const pulsar_model *m, const char *key, bool *out);
bool model_get_array(const pulsar_model *m, const char *key, pulsar_array_ref *out);
void model_close(pulsar_model *m);
/** Open and map the GGUF once.  The GPU path needs a shared mapping for
 * no-copy GPU buffers; tokenizer/inspection opens use a private read-only
 * mapping instead.
 */
void model_open(pulsar_model *m, const char *path, bool gpu_mapping);
void model_summary(const pulsar_model *m);
pulsar_tensor *model_find_tensor(const pulsar_model *m, const char *name);
bool accelerator_cache_model_tensors(pulsar_backend backend,
                                            const pulsar_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count,
                                            const char *skip_prefix);
/** Return the in-place tensor payload inside the mapped GGUF (or inside the
 * overlay file's mapping for --expert-overlay swapped tensors).
 */
const void *tensor_data(const pulsar_model *m, const pulsar_tensor *t);
uint32_t model_apply_expert_overlay(pulsar_model *base, const pulsar_model *overlay,
                                    const char *prefix);
bool accelerator_prepare_expert_overlay(pulsar_backend backend,
                                        const pulsar_model *base,
                                        const pulsar_model *overlay);

/** Mapping that owns a tensor's payload: the overlay file's map for
 * --expert-overlay swapped tensors, the base model's map otherwise. */
static inline const void *tensor_map_base(const pulsar_model *m, const pulsar_tensor *t) {
    return t->ext_map ? (const void *)t->ext_map : (const void *)m->map;
}
static inline uint64_t tensor_map_size(const pulsar_model *m, const pulsar_tensor *t) {
    return t->ext_map ? t->ext_size : m->size;
}
uint32_t required_u32(const pulsar_model *m, const char *key);
PULSAR_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const pulsar_tensor *t);
bool routed_expert_gate_down_layout(
        const pulsar_tensor *gate,
        const pulsar_tensor *down,
        uint64_t         *gate_expert_bytes,
        uint64_t         *gate_row_bytes,
        uint64_t         *down_expert_bytes,
        uint64_t         *down_row_bytes);
bool weights_have_output_head(const pulsar_weights *w);
const pulsar_layer_weights *weights_first_bound_layer(const pulsar_weights *w);
/** Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp.
 */
void config_validate_model(const pulsar_model *m);
/** Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers.
 */
void weights_bind(pulsar_weights *w, const pulsar_model *m);
void dspark_weights_bind(pulsar_dspark_weights *w, const pulsar_model *m);
void weights_free(pulsar_weights *w);
/** Dense layers and compressed layers use different RoPE bases. */
float layer_rope_freq_base(uint32_t il);
float layer_rope_freq_scale(uint32_t il);
float silu(float x);
void swiglu(float *out, const float *gate, const float *up, uint64_t n, float clamp);
uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk);
uint64_t argmax_f32(const float *x, uint64_t n);
/** Release every GPU tensor owned by the whole-model graph runtime. */
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
bool gpu_graph_debug_dump_enabled(void);
const char *gpu_graph_debug_dump_prefix(void);
bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos);
/** The predicate every f32 store-skip must use: true when EITHER observer (the
 * dump or the range sweep) will read the bytes.  See the definition's comment
 * -- the sweep reads through dump_tensor's own early branch, so debug_wants
 * alone is not the question.  _any() is the coarse, per-name-less twin. */
bool gpu_graph_f32_store_observed(const char *name, uint32_t il, uint32_t pos);
bool gpu_graph_f32_store_observed_any(void);
/** PULSAR_DSPARK_DUMP is set (parsed once): the drafter's f32 rows the dump
 * reads are stored; otherwise the drafter emits E4M3 only. */
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
bool gpu_graph_bank_repoint(pulsar_gpu_graph *g, uint32_t bank);
/** Effective pool size for banked kernel launches: banks.n_banks, or 1 when
 * the pool is disabled (the classic tensors act as bank 0). */
uint32_t gpu_graph_bank_pool_count(const pulsar_gpu_graph *g);
/** Tier-2 overcommit (task #55): demand-paged comp+index VA bytes for ONE bank at
 * a context (the overcommit-reserved, physical-on-touch part); and the EXACT
 * touched (physically resident) demand-paged KV summed over the whole pool from
 * the per-bank compressor frontier. See the definitions in gpu_diag.cpp. */
uint64_t gpu_graph_demand_paged_bytes_per_bank(uint32_t ctx_size);
uint64_t gpu_graph_touched_kv_bytes(const pulsar_gpu_graph *g);
/** Compressed rows a layer of compress ratio `ratio` (non-zero) holds at
 * ctx_size: the one capacity formula, read by gpu_graph_compute_dims (the
 * allocator's per-layer caps) and the KV sizing in steering.cpp. */
static inline uint32_t gpu_graph_comp_cap(uint32_t ctx_size, uint32_t ratio) {
    return ctx_size / ratio + 2u;
}
/** One bank's comp + index cache bytes at ctx_size in their stored row formats
 * (steering.cpp); the demand-paged term of the overcommit split. */
uint64_t gpu_graph_comp_index_bytes_for_context(uint32_t ctx_size);
/** One bank's raw SWA ring across every layer at raw_cap rows, in the stored
 * WINDOW / MAIN row formats (steering.cpp): the eager term of the KV
 * sizing, the same bytes gpu_graph_bank_slabs_alloc lays out per layer. */
uint64_t gpu_graph_raw_ring_bytes_for_context(uint32_t raw_cap);
/** The deepest compressed pool any layer holds at ctx_size (the smallest
 * non-zero compress ratio's gpu_graph_comp_cap): the row count the
 * indexer_scores scratch is sized by and the one the boot line reports as
 * compressed_kv_rows.  ctx_size itself when no layer compresses. */
uint32_t gpu_graph_comp_cap_max(uint32_t ctx_size);
/** Exact touched (physically resident) demand-paged comp/index KV of ONE bank,
 * from its compressor frontier.  Every bank -- live or idle -- reads
 * ms_n_comp[bank] now; stage 1b removed the separate live-scalar case.  The increment-2b guard uses this for the
 * per-bank Δ projection and the smallest-frontier victim tie-break.
 */
uint64_t gpu_graph_bank_touched_kv_bytes(const pulsar_gpu_graph *g, uint32_t bank);
/** Tier-2 task #55 increment 2b — CONSERVATIVE per-bank comp/index growth over one
 * decode quantum of `q` tokens: Σ_layers( ceil(q/ratio)·comp_row + q·index_row ).
 * The index term charges q (not ceil(q/ratio)) rows — a deliberate over-estimate
 * so the guard fires EARLY (safe side). Position-independent, so total Δ =
 * n_live_growing_banks × this.
 */
uint64_t gpu_graph_quantum_growth_bytes_per_bank(uint32_t q);
/** Tier-2 task #55 increment 2b — per-bank physical evict/restore reclaim
 * primitives (direct cudaFree / cudaMallocManaged of one bank's split comp/index
 * + base-table rebuild). See gpu_diag.cpp. */
bool gpu_graph_bank_free_physical(pulsar_gpu_graph *g, uint32_t bank);
/** Tier-2 task #55 increment 2b — RESTORE alloc primitive. Reallocate ONE evicted
 * bank's comp/index physical (fresh cudaMallocManaged: VA reserved, physical on
 * touch) and rebuild its base-table entries to the new pointers. The caller then
 * reloads the bank's KV (H2D from the disk snapshot) into these. Idempotent: a
 * slab already present is left untouched. Returns false on OOM.
 */
bool gpu_graph_bank_alloc_physical(pulsar_gpu_graph *g, uint32_t bank);
/** True when `bank`'s comp/index physical is currently freed (evicted) — the
 * server checks this before restoring on a returning request.
 */
bool gpu_graph_bank_is_evicted(const pulsar_gpu_graph *g, uint32_t bank);
/** Tier-2 PATH-A full-prefix fork (plan-33 inc A): D2D clone src bank's committed
 * KV into dst + mirror frontier counters. Caller validates + pins src first. */
bool gpu_graph_bank_fork_copy(pulsar_gpu_graph *g, uint32_t src, uint32_t dst);
/** plan-33 inc C: partial-cut fork + boundary machinery (gpu_diag.cpp). */
uint32_t pulsar_partial_fork_base_align(void);
bool gpu_graph_bank_fork_copy_cut(pulsar_gpu_graph *g, uint32_t src, uint32_t dst,
                                  uint32_t R, uint32_t src_len);
/** Whole-pool cache tensors for banked kernel operands: the bank slab when
 * the pool is enabled, else the classic single-session tensor (== bank 0).
 * NULL for layers without that cache kind. */
pulsar_gpu_tensor *gpu_graph_bank_raw_pool(pulsar_gpu_graph *g, uint32_t il);
/** Nominal comp/index operand for the batched attention/indexer wrappers. With
 * per-bank split allocations there is no single slab; the batched (descriptor)
 * path addresses banks through the base-pointer table (below), so this returns
 * bank 0's allocation as the nominal typed operand (its per-bank size drives the
 * wrappers' buffer-size validation). Pool disabled → the classic tensor.
 */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_pool(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_pool(pulsar_gpu_graph *g, uint32_t il);
/** Per-bank comp/index base-pointer tables (device arrays of n_banks pointers,
 * indexed by seq_id) the batched READ kernels use in place of base +
 * seq_id*comp_cap over one slab. NULL when the pool is disabled. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_bases(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_bases(pulsar_gpu_graph *g, uint32_t il);
/** Fresh single-bank views for the batched emit path (caller frees; when the
 * pool is disabled, bank must be 0 and the view wraps the classic tensor).
 * kind: the per-(bank,layer) comp caches and compressor state lanes. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
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
bool gpu_graph_multiseq_step_begin(pulsar_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur);
bool gpu_graph_multiseq_step_end(pulsar_gpu_graph *g);
/** Tier-2 batched multi-session decode: one token per live bank through ONE
 * weight sweep (see the definition comment in imatrix.cpp for the full driver
 * contract).  logits out = [n_active * PULSAR_N_VOCAB], row k = bank[k].
 * Returns 1 ok / 0 recoverable rejection (nothing mutated) / -1 fatal (armed
 * sweep or head failed — session state untrusted). */
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
        bool                   capture_cur);
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
uint32_t gpu_graph_prefill_slice(void);
/** Comp-cache row stride in bytes for the active storage format (pack-aware). */
/** The output head for ONE row of the sweep-final stream: batch_cur_hc row
 * `row` collapsed with batch_hc_pre row `row` (the last FFN's pre), normed,
 * projected into g->logits. */
bool gpu_graph_encode_output_head(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row,
        uint64_t               vocab_dim);
/** The output head for rows [row0, row0 + n_tokens) of the sweep-final stream
 * into g->spec_logits rows [0, n_tokens). */
bool gpu_graph_encode_output_head_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               row0,
        uint32_t               n_tokens,
        uint64_t               vocab_dim);
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
bool gpu_graph_dspark_seed_draft_kv(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
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
bool gpu_graph_matmul_plain_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);
bool gpu_graph_matmul_mxfp8_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);
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
pulsar_gpu_tensor *gpu_graph_q_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/** heads buffers stride by PULSAR_HEADS_ELT_SIZE (L033) -- use this for
 * batch_heads/heads, not the generic float-strided helper above. */
pulsar_gpu_tensor *gpu_graph_heads_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/** Read an HC residual carrier (BF16 storage; task #62) into an f32 host buffer,
 * expanding each sample. Dev-only (parity self-test + env-gated DSpark dumps). */
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
bool gpu_graph_upload_prompt_embeddings_hc(
        pulsar_gpu_graph     *g,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens);
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
bool gpu_graph_reset_prefill_state(pulsar_gpu_graph *g);
bool gpu_graph_prefill_layer_major(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud);
bool gpu_graph_prefill_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_prefill_chunked_range(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_prefill_chunked(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_verify_suffix_tops(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        int                   *row_tops,
        float                 *row_logits);
bool gpu_graph_read_spec_logits_row(pulsar_gpu_graph *g, uint32_t row, float *logits);

/** L195/L218: the RESUME GRID.  A continuation of a checkpoint is a cold
 *  prefill from G = the last multiple of PULSAR_RESUME_GRID at or below the
 *  session's PREFILL frontier (the last position a prefill wrote -- decode rows
 *  are the decode kernels' and can never equal a cold prefill's, so a resume
 *  recomputes the tokens generated since): rewind the bank to G, then prefill
 *  [G, N) as the cold prefill would.  Nothing is saved and nothing is warmed
 *  up: at any even position the ratio-2 compressors hold no pending group and
 *  the ratio-1 compressor holds no state at all, so the state at G IS the cold
 *  prefill's (0731's ratio-4 two-group window needed a 32-token state-only
 *  warm-up here; V4.1 has no such window).  128 is a multiple of 32, the
 *  period of the one remaining chunk-mate mechanism, the HC-mix GEMM's
 *  dependence on a row's offset within the call (censuses 14/15, 2026-09-06,
 *  at n_embd 4096; RE-CENSUS at 5120 before moving this). */
#define PULSAR_RESUME_GRID 128u
static_assert(PULSAR_RESUME_GRID % 2u == 0u, "a grid point must be a complete ratio-2 group");

/** Reset a bank's compressor state lanes to the canonical empty group (kv 0,
 *  score -INF): their state at any even position. */
bool gpu_graph_compressor_state_reset(pulsar_gpu_graph *g, uint32_t bank);
/** L218: make a bank's compressor state describe position `pos` after a
 *  rewind.  At a group boundary that is the empty group.  Inside a group the
 *  pending slots are re-stored from the last verify round's saved projections
 *  when they cover the group's committed positions; otherwise the state is
 *  reset and the bank marked stale (see ms_comp_state_stale).  Returns false
 *  only on a device failure. */
bool gpu_graph_compressor_state_rewind(pulsar_gpu_graph *g, uint32_t bank, uint32_t pos);
/** L149 phase 2: run the min-p prefilter (floor g->spec_compact_delta) over
 * spec_logits rows [row0, row0+n_rows) and read the compact block into
 * g->spec_compact_host at those row offsets; sets g->spec_compact_rows to
 * row0+n_rows on success, 0 on failure. Blocking read (one small copy). */
bool gpu_graph_spec_compact_read(pulsar_gpu_graph *g, uint32_t row0, uint32_t n_rows);
/** Pick a raw SWA cache size for GPU.  During batched prefill it must cover
 * the previous window plus the current ubatch.
 */
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
char *vocab_token_text(const pulsar_vocab *vocab, int token, size_t *len);
/** THE row-max rule: the first finite value seeds, lowest id wins a tie.
 * @return the argmax id, or -1 when the row has no finite value. */
int sample_argmax(const float *logits, uint32_t n_vocab);
/** The candidate distribution a sampler draws from, after filtering. */
typedef struct {
    int *ids;      ///< candidate token ids
    float *probs;  ///< renormalized over the filtered nucleus
    uint32_t n;    ///< candidates present in both arrays
} pulsar_sample_dist;

/** THE authority for the sampled candidate set (temperature, top-k, top-p,
 * min-p) and its order: every lane -- the plain samplers below and the
 * speculative accept walk -- draws from this object, so one rng state yields
 * one token whichever lane runs (L186).  `scratch` is required (non-NULL) and
 * must outlive nothing: it is pure working memory, reusable across calls and
 * independent of `out`.
 * @return 1 with `out` filled; 0 with `out` zeroed for a row no distribution
 * can be drawn from (no finite logit, or a non-positive / non-finite
 * candidate mass) -- said once on stderr. */
int pulsar_sample_dist_build(const float *logits, uint32_t n_vocab,
                          float temperature, int top_k, float top_p, float min_p,
                          pulsar_sample_scratch *scratch, pulsar_sample_dist *out);
/** L149: smallest min_p the device-prefiltered build accepts. Below it the
 * full path's top_p == 1.0f check is no longer provably redundant (it needs
 * min_p / filtered_sum > 2 ulp(1.0f) with filtered_sum <= PULSAR_N_VOCAB);
 * the production default is 0.05. */
#define PULSAR_SAMPLE_SPARSE_MINP_MIN 0.02f
/** ... and the widest vocab that bound was derived for (the caller refuses the
 * sparse path above it; the shape's vocab is a runtime value). */
#define PULSAR_SAMPLE_SPARSE_VOCAB_MAX 131072u
static_assert(PULSAR_SAMPLE_SPARSE_MINP_MIN > 2.0f * 5.9604645e-8f * (float)PULSAR_SAMPLE_SPARSE_VOCAB_MAX,
              "sparse min-p floor must dominate the top_p==1 rounding band over the vocab");
/** Byte-identical to pulsar_sample_dist_build(row, PULSAR_N_VOCAB, temperature, 0,
 * 1.0f, min_p) when fed the device prefilter's candidates (contract at the
 * definition). Returns 0, `out` untouched, for anything outside it. */
int pulsar_sample_dist_build_prefiltered(const int32_t *ids, const float *vals, uint32_t n_cand,
                                         float max_logit, float temperature, float min_p,
                                         pulsar_sample_scratch *scratch, pulsar_sample_dist *out);
void pulsar_sample_dist_free(pulsar_sample_dist *d);
float pulsar_sample_dist_prob(const pulsar_sample_dist *d, int token);
int pulsar_sample_dist_accept(const pulsar_sample_dist *d, int token, uint64_t *rng);
int pulsar_sample_dist_draw(const pulsar_sample_dist *d, uint64_t *rng);
int pulsar_sample_dist_draw_excluding(const pulsar_sample_dist *d, int excluded, uint64_t *rng);
/** Sampled-proposal speculative rule (the deterministic-proposal pair above is
 * pulsar_sample_dist_accept / _draw_excluding). `token` was drawn from a proposal
 * q; `q` is q(token). Accepts with probability min(1, p(token)/q(token)).
 * Never accepts a token with p(token) <= 0. Consumes no rng when the outcome
 * is certain (p >= q), matching pulsar_sample_dist_accept's p >= 1 fast path —
 * which is what keeps the temperature<=0 path byte-identical. */
int pulsar_sample_dist_accept_pq(const pulsar_sample_dist *p, int token, float q, uint64_t *rng);
/** The matching residual: draw from (p-q)+ normalized. Every token it can
 * return has p(token) > 0 AND strictly positive residual mass; if the total
 * residual mass is <= 0 it falls back to a plain draw from p. `scratch` is
 * working memory (see pulsar_sample_scratch::qmap); it must not alias p or q. */
int pulsar_sample_dist_draw_residual(const pulsar_sample_dist *p, const pulsar_sample_dist *q,
                                  pulsar_sample_scratch *scratch, uint64_t *rng);

/** The plain per-token sampler: pulsar_sample_dist_build -> pulsar_sample_dist_draw,
 * nothing else (greedy takes the point mass without an rng word).  `scratch`
 * is optional reusable working memory; pass the calling session's
 * sample_scratch, or NULL for a call-local one (pulsar_sample_logits, which
 * has no session) -- the same path, ~5 MB of malloc/free per call.
 * @return the token, or -1 when the build refused the row. */
int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        pulsar_sample_scratch *scratch);
int generate_gpu_graph_raw_swa(
        const pulsar_model   * model,
        const pulsar_vocab   * vocab,
        const pulsar_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        uint32_t            prefill_chunk,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        pulsar_token_emit_fn   emit,
        pulsar_generation_done_fn done,
        void              * emit_ud,
        pulsar_session_progress_fn progress,
        void              * progress_ud);
void pulsar_linux_graph_backend_set_oom_score(pulsar_backend backend);
void pulsar_release_instance_lock(void);
/** Refuse to start a second pulsar/ds4 process.  The model can map tens of GiB,
 * so a stale accidental second run is more dangerous than a normal CLI error.
 */
void pulsar_acquire_instance_lock(void);

/** ---- shared inline helpers ---- */

/** =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions are the CPU reference math used by the C backend and by
 * GPU diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

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


/** L107 adaptive draft depth bounds (controller in session_spec.cpp). MAX is
 * the drafter's TRAINED BLOCK (0731 DSpark metadata: stages=3 block=5):
 * position 6 is out of distribution, and the sweep measured depth 6 DOMINATED
 * everywhere -- accepted/step falls (3.31 -> 3.20 structured) while drafting
 * cost jumps, so even transient controller excursions there are purchased
 * losses (Tyler's catch, 2026-08-25 evening; the earlier ceiling of 6 was a
 * "probe step" rationale that predates knowing the block width). Re-tune MAX
 * only with a drafter retrained at a wider block (L092). The /metrics
 * max_draft reports at least MAX so the per-position waterfall covers every
 * position the controller can reach. */
enum { PULSAR_SPEC_DEPTH_MIN = 2, PULSAR_SPEC_DEPTH_MAX = 5 };
#endif /* PULSAR_ENGINE_INTERNAL_H */

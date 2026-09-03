#ifndef PULSAR_GPU_H
#define PULSAR_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* =========================================================================
 * THE BACKEND SEAM.
 * =========================================================================
 *
 * This header is the ONLY interface between the engine and a GPU backend.
 * The contract, mechanically enforced by `make seam-check`:
 *
 *   - No code outside src/cuda/ may include a CUDA header, name a CUDA
 *     runtime/driver/cuBLAS symbol, or use kernel-launch syntax. The engine
 *     talks exclusively in pulsar_gpu_* calls and pulsar_gpu_tensor handles.
 *   - A future backend (re-adding one means a new src/<backend>/ implementing
 *     these functions) must be selectable via pulsar_backend without engine
 *     changes.
 *   - Engine files may read backend TUNING knobs from the environment under
 *     the PULSAR_CUDA_* env NAMES (ops scripts use them), but the questions
 *     they answer in engine code must stay backend-neutral ("is fusion
 *     disabled", "dump wanted", "prefill chunk override").
 *   - Backend-branded log lines (e.g. "CUDA loading model tensors") are
 *     emitted only by the backend TUs about themselves.
 */

/** Hyper-connection (HC) residual-stream storage precision (task #62).
 * The source model runs a BF16 residual (config torch_dtype: bfloat16 — see
 * ds4-source-numerics); our HC carriers were f32, i.e. 2x the precision AND
 * bandwidth of the source with no fidelity gain. We narrow the STORAGE of the
 * six swap-coupled HC residual carriers (cur_hc/after_attn_hc/after_ffn_hc and
 * their batched twins) to BF16, while every kernel keeps accumulating in f32 —
 * exactly what torch does (bf16 storage, f32 math). This macro is the element
 * size of a stored HC residual sample, shared by the CUDA kernels (typed via
 * pulsar_hc_t in pulsar_cuda_internal.h) and the C host stride/offset math.
 *
 * There WAS a PULSAR_HC_F32 compile flag restoring f32 carriers. It existed to
 * prove the storage-narrowing plumbing was a pure no-op before the BF16 flip,
 * that flip shipped, and nothing has set it since -- no target, no gate, no
 * test. Removed 2026-08-17 with the rest of the callerless switches. */
#define PULSAR_HC_ELT_SIZE 2u
#define PULSAR_HC_ELT_FMT  PULSAR_ELT_BF16   /* pairs with pulsar_hc_t; bridge assert CUDA-side */

/** Stored element size of batch_heads / heads -- the attention output, and the
 * largest UNCONDITIONAL f32 activation store in the engine (~512 MiB at 4096
 * prefill; 21.5 GiB of writes per prefill, every layer, every arm).
 *
 * FLIPPED TO 2u (bf16) 2026-08-24, Tyler-accepted, after the plumbing landed
 * inert (increments 1-7, SASS-proven no-ops) exactly as the Q flip did (L045).
 * Every producer and consumer goes through pulsar_heads_t / heads_load /
 * heads_store; the re-anchor ceremony (budgets + baseline at 5d45142) recorded
 * the move.  This paragraph used to say "STILL 4 IN THIS INCREMENT" -- the
 * colonoscopy found that stale claim sitting fifteen lines above the 2u it
 * contradicted, at the single most load-bearing comment in the flip (L106 K4).
 *
 * WHY BF16 AND NOT F16, when the attention tier packs Q to __half anyway:
 * measured, not assumed.  The 2026-08-23 three-way grade re-scored under the
 * confident/flat split (L080) gives, against the B300 source, on the depths
 * where the model is actually certain:
 *     f32 -> f16  : +7.23% FURTHER from source
 *     f32 -> bf16 : -29.67% CLOSER
 * They point in OPPOSITE directions; the old all-depths view called both
 * "neutral" because the two flat rows masked the difference.  bf16 is the
 * source's own residual dtype (ds4-source-numerics), f16 is a dtype the source
 * has nowhere -- so this narrowing is a FIDELITY IMPROVEMENT, not a tradeoff.
 *
 * ⚠ IT IS NOT A SPEED CHANGE.  The engine is latency-bound and attn_f16 sits at
 * ~22% of memory throughput (engine-sol-sweep-2026-08-24.md), so removing store
 * bytes cannot speed it up.  The gains are fidelity and a 512 -> 256 MiB
 * allocation.  Do not sell it as throughput. */
#define PULSAR_HEADS_ELT_SIZE 2u
#define PULSAR_HEADS_ELT_FMT  PULSAR_ELT_BF16   /* pairs with pulsar_heads_t; bridge assert CUDA-side */


/** Q activation storage precision (L045).  batch_q is the largest activation in
 * the model -- pc * n_head * head_dim, 512 MiB at a 4096-token prefill -- and
 * it exists largely to be read back and narrowed: the shipped attention tier
 * packs Q to __half for HMMA on the very next instruction.  Storing it f32 is
 * over-precision against a source whose activations are narrower still.
 *
 * Same shape as PULSAR_HC_ELT_SIZE above: this macro is the stored element
 * size for the host stride/offset math, pulsar_q_t is the matching CUDA type,
 * and the kernels load to f32 and accumulate in f32 regardless -- only the
 * STORAGE narrows.  Both Q buffers (batch_q and the single-token q) use it, so
 * there is exactly ONE Q element type in the engine rather than two sharing
 * kernels.
 *
 * ⚠ NOT bit-exact when narrowed, unlike the HC flip: under q_prep_active
 * batch_q holds UNNORMALISED q_b output and the per-head sum of squares is
 * taken from it, so narrowing moves the inputs to a reduction rather than just
 * its storage.  Measured neutral against the cross-engine reference (L045
 * stage 1); it is cuda-reference-gate that certifies this, not the byte-exact
 * prefill gate. */
#define PULSAR_Q_ELT_SIZE 2u
#define PULSAR_Q_ELT_FMT  PULSAR_ELT_F16   /* pairs with pulsar_q_t; bridge assert CUDA-side */

/** Shared-expert gate/up staging element size (L033 increment 2).  Same
 * contract as the two above: this is the STORED width the alloc, the byte
 * budget, and any host stride math must agree on; the producer (the mxfp8
 * GEMM) and the consumer (the swiglu fold) both derive their kernel type from
 * the tensor's esz at runtime, never from this macro.  f16 today. */
#define PULSAR_SHARED_ACT_ELT_SIZE 2u
#define PULSAR_SHARED_ACT_ELT_FMT  PULSAR_ELT_F16   /* f16 staging; runtime-esz consumers */

/** spec_logits row capacity.  The multi-row logits slab is sized to the
 * deepest speculative verify / multiseq head the engine ever emits
 * (PULSAR_MSEQ_MAX), NOT to prefill_cap -- guards on row indices must check
 * against THIS, or the 16 lives only in a comment. */
#define PULSAR_SPEC_LOGITS_ROWS 32u   /* L117 2026-08-27: 16 -> 32. The 16-row
 * ceiling squeezed per-bank draft depth at c3+ (c4: K~3 vs solo K~8, the
 * measured sublinear c4 scaling); the ROWCOST table says marginal row cost
 * is ~8-11 ms with no cliff, so a 32-row slab (+8.3 MB logits) lets the
 * ranked allocator keep K near its survival optimum at c4. Every consumer
 * derives from THIS constant (slab alloc, driver reject, lane arrays,
 * dspark batch-capture buffer) -- audited 2026-08-27, rows/L117.md. */



/** =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct pulsar_gpu_tensor pulsar_gpu_tensor;

int pulsar_gpu_init(void);
void pulsar_gpu_cleanup(void);

/** Running total of live tensor-alloc bytes (owned allocations only, views
 * excluded).  Snapshot around a session create to measure its true resident
 * cost; the server ledger commits that actual. */
uint64_t pulsar_gpu_tensor_alloc_bytes_current(void);
/** Dry-run allocation on the calling thread: after dry_begin, the two tensor
 * allocators return placeholders and total the requested bytes instead of
 * touching the device; every device operation on a placeholder is a no-op
 * that reports success.  dry_end returns the totals (all bytes; the
 * cudaMallocManaged subset) and ends the run.  The session pricer runs the
 * real allocation code this way -- the price of a session and its allocation
 * are one piece of code. */
void     pulsar_gpu_tensor_dry_begin(void);
void     pulsar_gpu_tensor_dry_end(uint64_t *bytes, uint64_t *managed_bytes);
int      pulsar_gpu_tensor_dry_active(void);
pulsar_gpu_tensor *pulsar_gpu_tensor_alloc(uint64_t bytes);
/* n_elems * esz bytes, with esz recorded on the tensor.  Use this for any
 * buffer whose elements are not f32; consumers then derive the type from
 * the tensor instead of being handed a flag that can disagree with it. */
/** Element FORMAT of a tensor (L106 K15, Tyler: option A).  An element SIZE
 * cannot distinguish __half from __nv_bfloat16 (both 2 B) nor E4M3 from int8
 * from raw bytes (all 1 B), and this engine's dominant historical defect class
 * is clean-compiling byte reinterpretation -- including in the widening
 * diagnostics that fidelity decisions are read from.  The tag makes wrong
 * dispatch IMPOSSIBLE rather than merely disciplined:
 *   - every alloc_elt names its format from the same authority that names its
 *     width (the *_ELT_FMT macro beside each *_ELT_SIZE);
 *   - views/subviews inherit it;
 *   - a widening reader dispatches on it and REFUSES what it cannot widen.
 * PULSAR_ELT_BYTES marks packed/opaque rows and integer payloads: sized,
 * never widenable. */
typedef enum pulsar_elt_fmt {
    PULSAR_ELT_F32  = 0,   /* 0 so a zeroed struct reads as the f32 default,
                              exactly as esz==0 does */
    PULSAR_ELT_F16  = 1,
    PULSAR_ELT_BF16 = 2,
    PULSAR_ELT_BYTES = 3,
} pulsar_elt_fmt;

pulsar_gpu_tensor *pulsar_gpu_tensor_alloc_elt(uint64_t n_elems, uint32_t esz,
                                               pulsar_elt_fmt fmt);
pulsar_gpu_tensor *pulsar_gpu_tensor_alloc_managed(uint64_t bytes);
pulsar_gpu_tensor *pulsar_gpu_tensor_view(const pulsar_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void pulsar_gpu_tensor_free(pulsar_gpu_tensor *tensor);
uint64_t pulsar_gpu_tensor_bytes(const pulsar_gpu_tensor *tensor);
/** Raw device pointer without a synchronize (for building device pointer tables). */
void *pulsar_gpu_tensor_device_ptr(const pulsar_gpu_tensor *tensor);
int pulsar_gpu_tensor_fill_f32(pulsar_gpu_tensor *tensor, float value, uint64_t count);
int pulsar_gpu_tensor_write(pulsar_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);

/** Fill a Q buffer from host f32, converting to the stored Q element type.
 * Host callers cannot see pulsar_q_t (it is CUDA-only), so a plain
 * tensor_write of floats into a narrowed Q buffer is both an overrun and a
 * reinterpretation -- and it compiles.  `n` counts ELEMENTS, not bytes. */
int pulsar_gpu_tensor_write_q_f32(pulsar_gpu_tensor *tensor, uint64_t off_elems,
                                  const float *src, uint64_t n);
int pulsar_gpu_tensor_read(const pulsar_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int pulsar_gpu_tensor_copy(pulsar_gpu_tensor *dst, uint64_t dst_offset,
                          const pulsar_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
/** Same copy on the per-thread stream, WITHOUT blocking the host. Only for
 * destinations consumed by a later kernel on that same stream -- stream order
 * covers those. Anything the host reads back, or that crosses streams, keeps
 * the blocking form above. */
int pulsar_gpu_tensor_copy_async(pulsar_gpu_tensor *dst, uint64_t dst_offset,
                                 const pulsar_gpu_tensor *src, uint64_t src_offset,
                                 uint64_t bytes);

/** Batched D2D copy: prepare a device-side descriptor table over fixed tensor
 * allocations once (whole-tensor copies, byte counts multiples of 16; returns
 * NULL on any violation), then replay all copies with one kernel launch.
 * max_bytes is the largest descriptor's byte count (grid sizing). Built for the
 * spec-frontier snapshot/restore paths (~126 tiny per-layer copies per step). */
void *pulsar_gpu_batched_copy_prepare(pulsar_gpu_tensor **dst, pulsar_gpu_tensor **src,
                                   const uint64_t *bytes, uint32_t n);
int pulsar_gpu_batched_copy_run(void *handle, uint32_t n_descs, uint64_t max_bytes);
void pulsar_gpu_batched_copy_free(void *handle);

/** Command-tape bracket. There is NO CUDA-graph capture behind this: the
 * decode graph tape was measured at a +0.6% ceiling on GB10 (decode is
 * 98.9% GPU-busy, so replay has nothing to reclaim) and removed --
 * begin_commands returns 1 unconditionally, end_commands synchronizes.
 * The pair survives because callers bracket their encode regions with it.
 * An earlier comment here described a capture pair and a
 * PULSAR_CUDA_NO_GRAPHS opt-out, neither of which exists anywhere in the
 * tree; upstream ds4 independently measured the same ~0.5% and turned
 * graphs off by default on integrated Blackwell. */
int pulsar_gpu_begin_commands(void);
int pulsar_gpu_end_commands(void);
int pulsar_gpu_synchronize(void);
/** Record a completion marker on the per-thread stream (L142).  Returns a
 *  handle, or -1 if the runtime could not record one; pulsar_gpu_marker_done
 *  reports a -1 handle as complete, so a poller never stalls on it. */
int pulsar_gpu_marker_record(void);
/** 1 when every op enqueued before the marker has completed (or the handle is
 *  -1 / invalid), 0 while the GPU is still working towards it.  Never blocks. */
int pulsar_gpu_marker_done(int marker);

/** L119 segment capture-or-replay for round-invariant decode-sweep stretches
 * (plan 119 in pulsar-notes; supersedes the removed whole-sweep tape the
 * comment above describes — the unified batch lane measures 92% GPU-busy
 * unprofiled, and these reclaim its inter-launch gaps). seg_enter: 0 = run
 * the body eagerly, 1 = capture armed (run body, then seg_exit), 2 = cached
 * graph replayed (skip the body). seg_exit returning 0 means the captured
 * work NEVER RAN — re-run the body eagerly. Always on (L119 verdict: the
 * head segment ships default-on; the per-layer FFN segments measured a net
 * loss on GB10 and are deleted). Worker thread only. */
int pulsar_gpu_seg_enter(uint64_t key);
int pulsar_gpu_seg_exit(uint64_t key, int body_ok);
void pulsar_gpu_seg_reset(void);   /* REQUIRED on gpu-graph teardown: cached
                                    * execs bake that graph's device pointers */
void pulsar_gpu_seg_note_device_free(void); /* REQUIRED at every grow-realloc
                                    * scratch free: baked pointers go stale */

int pulsar_gpu_set_model_map(const void *model_map, uint64_t model_size);
int pulsar_gpu_set_model_fd(int fd);
int pulsar_gpu_set_model_fd_for_map(int fd, const void *model_map);
int pulsar_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int pulsar_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int pulsar_gpu_cache_external_range(const void *host_base_key, int fd, uint64_t offset, uint64_t bytes, const char *label);
int pulsar_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void pulsar_gpu_print_memory_report(const char *label);
/** cudaMemGetInfo passthrough (0/0 on failure) for diagnostics/samplers. */
void pulsar_gpu_mem_info(uint64_t *free_out, uint64_t *total_out);

/** =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int pulsar_gpu_embed_tokens_hc_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int pulsar_gpu_indexer_score_one_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);


/** Banked (multi-session) mode: positions/seq_id are per-row int32 device
 * arrays (row t's absolute position and TRUE bank id), comp_cap the per-bank
 * compressed-row stride, n_banks the pool size; the comp cache operand is
 * the whole bank pool.  Per-row visible count = (qpos+1)/ratio (the engine's
 * emit-before-read rule); rows past it (and dead rows, seq_id out of pool)
 * score -INF.  Scalar n_comp = cross-bank superset (scan bound + scores-row
 * stride only).  NULL/NULL/0/1 = classic single-cache behavior bit-exactly.
 * Banked multi-token rows run the generic kernel (the WMMA tier stays
 * single-bank); banked n_tokens==1 keeps the direct-one fast tier so the
 * scan is bit-identical to classic single-token decode.
 * L121: the engine now splits banked multi-token spans into same-bank
 * consecutive-position runs and feeds each through the run entry below
 * (block-scaled MXFP4 tier); this generic path remains the fallback for
 * non-conforming spans. */
int pulsar_gpu_indexer_scores_decode_batch_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *index_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks);

/** L121: score ONE same-bank consecutive-position run of a banked decode span
 * through the block-scaled MXFP4 tier.  All tensor views are positioned at
 * the run's first row; bank_index_comp is that bank's own comp slab view.
 * n_comp is the scores stride / scan bound (step-top superset), not the
 * bank's frontier -- the tier's position-derived causal mask bounds every
 * comp read at (run_pos0+run_n)/ratio and writes -INF into the masked tail
 * of each scores row.  Reassociation-class vs the generic kernel (MMA
 * accumulation): NOT bit-identical, same fidelity class the non-banked
 * prefill spans already ship (suite-v1 KL + top-k overlap evidence).
 * Requires n_head == 64, head_dim == 128.  Returns 0 on refusal/failure. */
int pulsar_gpu_indexer_scores_decode_run_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *bank_index_comp,
        uint32_t                n_comp,
        uint32_t                run_n,
        uint32_t                run_pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

/** Does the backend's PREFILL attention read PULSAR_ATTN_PACK comp rows
 * natively?  When it does, the engine hands it the packed cache directly and
 * skips dequantising into the f32 shadow -- 584 B/row instead of 2048, on the
 * rows that dominate the tile, plus one whole pass removed.  Bit-exact either
 * way: packed rows decode to exactly the values the f32 cache would hold.
 * Backend-neutral question; the answer is a property of the backend's kernels,
 * not of any particular one. */
int pulsar_gpu_attention_prefill_reads_packed_comp(void);

/* fp16 tensor-core attention (m16n8k16, f32 accumulate).  Raw pointers: a leaf
 * kernel behind the attention launchers, which do the tensor-level checking.
 * Returns 0 on refusal or failure.  Requires head_dim == 512 and n_head a
 * multiple of 16.  Operand format chosen by measurement, not preference --
 * the operand study is in src/cuda/pulsar_cuda_attn_f16.cu's header, and
 * docs/engine-perf-map.md.  (It used to cite tests/attn_precision_fidelity.cc,
 * deleted in a71e346 -- L106 K8 -- when its dump-format producer left the
 * tree.) */
/** Q-prep descriptor for the fused norm+rope Q load (L037 lever 3). Non-NULL
 * means `q` holds RAW projections: the consumer must apply the per-head RMS
 * norm and tail rope itself, bit-exactly matching head_rms_norm_rope_tail
 * (the fp16 kernel fuses it into its Q fragment build; a non-f16 path applies
 * the standalone kernel first and continues as if q_prep were NULL). NULL
 * means q is already normed+roped -- every decode caller, and the fallback.
 * Carries only the launch-invariant shape; pos0/positions stay the wrapper
 * params they already are. */
typedef struct {
    float eps;             ///< RMS-norm epsilon
    uint32_t n_rot;        ///< rotary dimensions at the head's tail
    uint32_t n_ctx_orig;   ///< context length the RoPE settings were trained at
    float freq_base;       ///< RoPE base frequency
    float freq_scale;      ///< linear frequency scaling (position interpolation)
    float ext_factor;      ///< YaRN extrapolation mix; 0 disables it
    float attn_factor;     ///< YaRN attention temperature correction
    float beta_fast;       ///< YaRN ramp start, in rotations per context
    float beta_slow;       ///< YaRN ramp end, in rotations per context
} pulsar_gpu_q_prep;

/** True when the fp16 attention tier will take eligible batches (the device
 * has the MMA path; the env opt-out was retired 2026-09-02) -- the engine uses
 * it to decide whether to defer Q norm+rope to the kernel or run the
 * standalone kernel as before. */
int pulsar_gpu_attn_f16_tier_on(void);

/** Opaque packed-row carriers (L092).  The packed caches (384-B NVFP4
 * ATTN_PACK rows, see pulsar_cuda_internal.h; MXKV-FP4 indexer rows) used to
 * travel as `const float *` -- a carrier type
 * that described nothing, so one direct raw_kv[i] anywhere was defect ten and
 * compiled clean.  Deliberately INCOMPLETE types: indexing or arithmetic is a
 * compile error, so every read goes through the format's accessor (or an
 * explicit byte-level cast at a row-granular copy).  Pass tensor->ptr. */
typedef struct pulsar_attn_pack_s pulsar_attn_pack_t;
typedef struct pulsar_mxkv_pack_s pulsar_mxkv_pack_t;

int pulsar_gpu_attention_f16_prefill_mx(
        /* q: stored Q, PULSAR_Q_ELT_SIZE bytes per element.  Opaque here so the
         * public header does not pull in cuda_fp16.h; the concrete type is
         * pulsar_q_t in pulsar_cuda_internal.h.  Pass tensor->ptr.
         *
         * heads: stored attention output, PULSAR_HEADS_ELT_SIZE bytes per
         * element, opaque for the same reason (concrete type pulsar_heads_t).
         * It was `float *` until L033; if you are adding a caller, pass
         * tensor->ptr and do NOT assume f32. */
        void *heads, const float *sinks, const void *q,
        const pulsar_attn_pack_t *raw_kv, const pulsar_attn_pack_t *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp,
        uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope,
        uint32_t gact_tok0, uint32_t gact_ntok,
        /* positions: int32 [n_tokens] DEVICE array or NULL.  Dense mode uses it
         * for the fused Q rope only (positions[t] instead of t); the row plan is
         * the batch's own causal window either way. */
        const int *positions,
        const pulsar_gpu_q_prep *q_prep);

int pulsar_gpu_attention_f16_prefill(
        /* heads: stored attention output, PULSAR_HEADS_ELT_SIZE bytes per
         * element, opaque for the same reason as q below (concrete type
         * pulsar_heads_t).  It was `float *` until L033.
         *
         * This wrapper has NO caller in the engine -- it exists so
         * tests/attn_f16_kernel_test.cu can drive the real kernel through a
         * narrow signature.  That is exactly why this declaration went stale
         * when its definition changed and the engine still built clean: no
         * production translation unit references the symbol, so nothing but
         * the gate could catch the mismatch.  Keep it in step with the
         * definition in pulsar_cuda_attn_f16.cu by hand. */
        void                    *heads,
        const float             *sinks,
        /* q: stored Q, PULSAR_Q_ELT_SIZE bytes/element; opaque here so this header
         * need not include cuda_fp16.h.  Pass tensor->ptr. */
        const void              *q,
        const pulsar_attn_pack_t *raw_kv,
        const pulsar_attn_pack_t *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep);

/** fp16 tensor-core attention, INDEXED: raw rows come from a ring buffer and
 * compressed rows are a top-k selection (topk != NULL) or the visible prefix
 * (topk == NULL, the decode-batch sweep).  This is the ONE decode attention
 * kernel (L166): every row count, every context length, causal and
 * non-causal.  Banked descriptors (positions/seq_id/comp_bank_ptrs;
 * all-or-nothing) are served; comp rows are ATTN_PACK rows, always -- bank
 * isolation gated by tests/attn_f16_banked_test.cu.  Returns 0 on refusal or
 * failure. */
int pulsar_gpu_attention_f16_indexed(
        /* heads: stored attention output, PULSAR_HEADS_ELT_SIZE bytes/element;
         * opaque here for the same reason as q below (L033). Pass tensor->ptr. */
        void                    *heads,
        const float             *sinks,
        /* q: stored Q, PULSAR_Q_ELT_SIZE bytes/element; opaque here so this header
         * need not include cuda_fp16.h.  Pass tensor->ptr. */
        const void              *q,
        const pulsar_attn_pack_t *raw_kv,
        const pulsar_attn_pack_t *comp_kv,
        const int               *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const int               *positions,
        const int               *seq_id,
        const void * const      *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        /* Raw visibility rule: 0 = causal (a row sees ring rows at or before
         * its own position), nonzero = every ring row up to the last one
         * (the drafter's raw-window forward).  Only WHICH rows are visible
         * changes; compressed-row visibility and the fold are the same. */
        uint32_t                non_causal,
        const pulsar_gpu_q_prep *q_prep);

/** Block-scaled indexer scorer (SM120 mxf8f6f4 MMA over the stored MXFP4 rows).
 * Raw pointers, not tensors: it is a leaf kernel behind indexer_scores_launch,
 * which does the tensor-level bounds checking.  Returns 0 on refusal or
 * failure; the caller checks the shape conditions itself so a 0 is always a
 * real failure.  Requires n_head == 64 and head_dim == 128. */
int pulsar_gpu_indexer_scores_mxfp4(
        float                   *scores,
        const pulsar_mxkv_pack_t *q,
        const float             *weights,
        const pulsar_mxkv_pack_t *comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        int                     causal);


int pulsar_gpu_indexer_topk_tensor(
        pulsar_gpu_tensor       *selected,
        const pulsar_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/** GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int pulsar_gpu_argmax_tensor(
        pulsar_gpu_tensor       *out_idx,
        const pulsar_gpu_tensor *logits,
        uint32_t                n_vocab);


/** =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

/** The MXFP8 matmul workhorse: `n_tok` activation rows against a registered
 * MXFP8 weight living in the model mapping.
 *
 * The output element type is read from `out` ITSELF (see
 * pulsar_gpu_tensor_alloc_elt) rather than passed. That applies to every arm --
 * cuBLASLt and the mmvq/NT kernels alike -- because a buffer written from two
 * widths (a prefill chunk vs the drafter's n_draft) must not end up holding two
 * element types.
 *
 * @param out            destination rows; its element type selects the store width
 * @param model_map      base of the model mapping the weight lives in
 * @param model_size     its size; the bound the weight span is checked against
 * @param weight_offset  byte offset of the weight within that mapping
 * @param in_dim         input width (K)
 * @param out_dim        output width (N)
 * @param x              activation rows
 * @param n_tok          rows to multiply (M)
 * @return 0 on success.
 */
int pulsar_gpu_matmul_mxfp8_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);

/** Register one MXFP8 workhorse weight (attn_kv/q, attn_output, shared experts,
 * output head) by offset so the matmul above executes it; done once at load. */
void pulsar_gpu_register_fp8_weight(uint64_t weight_offset);

/** Mark an already-fp8-registered offset as a pre-stored MXFP8_LT weight: the
 * device layout (de-interleaved E4M3 data + swizzled E8M0 scale) is already in
 * the mmap, so the matmul resolver skips the cudaMalloc+convert and points
 * cuBLASLt directly at g_model_device_base+offset. Done once at load. */
void pulsar_gpu_register_fp8_lt_weight(uint64_t weight_offset);

/** Batched-prefill activation quantization cache.
 *
 * One normalized activation feeds several block-scaled MXFP8 projections per
 * layer; the per-GEMM activation quantization is a pure function of the buffer
 * and its shape, so it only has to run once.  Arm the cache immediately after
 * the activation is written (arming invalidates any earlier contents, which is
 * what makes a later hit safe even though the buffer pointer is reused every
 * layer); disarm when the activation is dead.  A backend without the
 * optimization may implement both as no-ops.  Purely a traffic optimization:
 * results are bit-identical either way.
 */
void pulsar_gpu_mxfp8_act_cache_arm(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);
void pulsar_gpu_mxfp8_act_cache_disarm(void);


/** The widest DECODE batch the M-independent kernels take (the nt GEMV
 * instantiations in pulsar_cuda_matmul.cu, the small-batch expert FFN GEMV
 * and per-expert projection in pulsar_cuda_moe.cu enumerate up to it).  It
 * bounds pulsar_gpu_matmul_set_batch_decode_rows, and the PULSAR_MSEQ_MAX
 * static_assert in the engine keeps the bank count inside it: a decode row
 * past this width would have no M-independent arm, so growing PULSAR_MSEQ_MAX
 * past it FAILS THE BUILD.  It chooses no arm: row KIND does (below). */
#define PULSAR_GPU_MNEUTRAL_ROWS_MAX 16u

/** ROW KIND.  `n` is the number of leading DECODE rows in the batch being
 * encoded -- a fact about the rows, declared by the lane that owns them, and
 * the ONE thing every dense GEMM and MoE dispatcher reads to choose its arm:
 *   - decode rows (n > 0, n >= the call's n_tok) take the M-INDEPENDENT arms
 *     (one-row GEMV at 1, nt / small-batch FFN / per-expert projection at
 *     2..PULSAR_GPU_MNEUTRAL_ROWS_MAX), so a decode row's bytes depend on
 *     neither its batchmates nor the batch width;
 *   - prefill rows (n == 0) take the TENSOR-CORE arms (cuBLAS(Lt), grouped
 *     CUTLASS) at ANY n_tok, one row included -- the arm the B300 reference
 *     computes prefill rows with;
 *   - a mixed batch (0 < n < n_tok) is laid out [decode rows 0..n) then one
 *     prefill run [n..n_tok); the dispatchers split there and recurse with n
 *     on the prefix and 0 on the suffix.
 * Row COUNT chooses nothing (L167: it had been a proxy for kind, with a mode
 * flag as tie-break).  Lanes that own decode rows declare them at their entry
 * and restore on exit (pulsar_decode_rows_scope in the engine): the batched
 * step, the classic verify block, the drafter's forwards and seeds, the
 * one-row output head.  Prefill declares nothing.  The prefill encoder's
 * f32-store skips read it too: the split's offset views key no slot, so the
 * skips apply only while no decode prefix is in flight.
 * Returns 0 and refuses when n exceeds PULSAR_GPU_MNEUTRAL_ROWS_MAX. */
int pulsar_gpu_matmul_set_batch_decode_rows(int n);
int pulsar_gpu_matmul_batch_decode_rows(void);

int pulsar_gpu_matmul_bf16_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);



int pulsar_gpu_matmul_f32_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);

/** Read n ELEMENTS of t to the host as f32, whatever width they are stored at.
 *
 * pulsar_gpu_tensor_read copies BYTES, so a caller that wants floats has to
 * know the stored width -- and host code cannot see it (the esz field lives on
 * the struct in pulsar_cuda_internal.h).  Every such caller therefore assumed
 * f32, which is correct until it isn't: reinterpreting a narrowed buffer is
 * type-legal, silent, and produces a dump that looks like data.  This is the
 * one entry point diagnostics should use.  Returns 0 on failure. */
int pulsar_gpu_tensor_read_f32(const pulsar_gpu_tensor *t, uint64_t elem_off,
                               float *out, uint64_t n_elems);

/* Reserve the activation cache's E4M3 slots and hand back both device pointers
 * plus the scale pitch, so a producer can emit the MX encoding from its own
 * epilogue and the separate quantize pass disappears.  Returns 0 on failure. */
/** Producer-side BF16 activation slot (L086 T3): reserve the bf16 copy's
 * storage for (x, n_tok, in_dim) so an epilogue can write it, then note() to
 * mark it valid once the kernel succeeded.  The bf16 GEMM core reads this plane
 * (any row window of it, so offset views and mixed-batch suffixes hit); there is
 * NO convert-on-miss -- a bf16-weight GEMM whose activation has no producer-
 * emitted plane refuses (L159).  pulsar_gpu_act_slot_drop forgets every plane
 * keyed on a buffer that is about to be freed. */
int pulsar_gpu_bf16_act_slot(const pulsar_gpu_tensor *x,
                             uint64_t n_tok, uint64_t in_dim, void **xb_out);
void pulsar_gpu_bf16_act_note(const pulsar_gpu_tensor *x,
                              uint64_t n_tok, uint64_t in_dim);
void pulsar_gpu_act_slot_drop(const pulsar_gpu_tensor *x);

int pulsar_gpu_mxfp8_act_cache_e4m3_slot(const pulsar_gpu_tensor *x,
                                         uint64_t n_tok, uint64_t in_dim,
                                         void **data_out, void **scale_out,
                                         int *sf_pitch);

/** GROUPED activation slots for the attn-output "a" projection (batch_heads).
 * Reserves per-group E4M3 data plus a per-group swizzled E8M0 scale slab and
 * zeroes the scales, so the attention epilogue and rope_tail can emit the
 * encoding between them and the GEMM's quantize pass disappears.  `scale_slab`
 * returns the per-group stride the producers must index with.  Returns 0 on
 * failure; note() only after BOTH producers have run. */
int pulsar_gpu_mxfp8_gact_slot(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                               uint32_t n_groups, uint64_t group_dim,
                               void **data_out, void **scale_out,
                               int *sf_pitch, uint64_t *scale_slab);
void pulsar_gpu_mxfp8_gact_note(void);
void pulsar_gpu_mxfp8_gact_disarm(void);

/** L158: the drafter's concat of three target hidden states, emitted as E4M3
 * straight into the activation slot (producer-side A8; the f32 concat is never
 * written).  slot_data/slot_scale/sf_pitch come from
 * pulsar_gpu_mxfp8_act_cache_e4m3_slot for the concat tensor at (1 row,
 * 3*n_embd); the caller notes the encoding current and the f32 store skipped.
 * @return 1 on success, 0 on a bad argument or launch failure. */
int pulsar_gpu_dspark_concat3_e4m3(void *slot_data, void *slot_scale, int sf_pitch,
                                   const pulsar_gpu_tensor *h0, const pulsar_gpu_tensor *h1,
                                   const pulsar_gpu_tensor *h2, uint32_t n_embd);

/** L158: give an OFFSET ROW VIEW of an encoded activation its own slot, filled
 * from the producer's encoding (byte copy + scale re-base; no quantise).  Used
 * by the mixed-batch prefix splits for their prefill suffix views so every
 * consumer finds the encoding by its ordinary lookup.  0 = no valid encoding
 * on the full activation (the caller refuses). */
int pulsar_gpu_mxfp8_act_cache_window(const pulsar_gpu_tensor *x_full, uint64_t row0, uint64_t rows,
                                      uint64_t in_dim, const pulsar_gpu_tensor *x_view);

/** L158: PRODUCER-side encode of an f32 activation the caller itself produced
 * and owns -- for tests and tools that synthesise an activation.  Arms the
 * slot for (x, n_tok, in_dim), encodes with the engine's quantiser, notes it
 * current.  Engine producers emit from their epilogues and must not call this;
 * no consumer quantises any more. @return 1 on success. */
int pulsar_gpu_mxfp8_act_cache_encode_f32(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);
/** bf16 twin for a synthesised activation (probes/tests): fills x's bf16 plane
 * by RNE from the f32 rows and notes it.  No engine producer calls this. */
int pulsar_gpu_bf16_act_encode_f32(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);

/** L158: PRODUCER-side grouped encode of the attention output heads for an
 * attention producer without an E4M3 epilogue (the drafter's raw batch
 * attention).  Call right after the inverse rope tail and before the attn-out
 * projection; pulsar_gpu_mxfp8_gact_disarm after it. @return 1 on success. */
int pulsar_gpu_mxfp8_gact_emit_heads(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                                     uint32_t n_groups, uint64_t group_dim);

/** Declare the E4M3 encoding current after a producer filled those slots. */
void pulsar_gpu_mxfp8_act_cache_note_mxfp8(void);

/* Record that the producer ALSO skipped this buffer's f32 store, so the f32
 * bytes are stale.  Call only after note_mxfp8(), and only from a producer that
 * actually emitted the encoding -- the two must be gated on the SAME predicate,
 * or the GEMM reads a store that was never written and returns a well-formed
 * wrong answer.  Every f32-reading arm of the mxfp8 family checks this and
 * fails loudly rather than run. */
/** keep_from: rows below it were skipped; rows >= keep it.  A full skip
 * passes the arming n_tok. */
void pulsar_gpu_mxfp8_act_cache_note_f32_skipped(uint32_t keep_from);

/** Hand back the E4M3 encoding this buffer already carries, or 0 if the cache
 * holds none for (ptr, n_tok, in_dim). Lets a consumer that would otherwise
 * quantize the f32 copy reuse the producer's encoding instead -- the routed-MoE
 * gather is the one that matters, since it re-encoded every gathered row.
 * `scale` is indexed with the mx_sfoff swizzle over `kbp` blocks per row. */
int pulsar_gpu_mxfp8_act_cache_get_e4m3(const pulsar_gpu_tensor *x,
                                        uint64_t n_tok,
                                        uint64_t in_dim,
                                        const void **data,
                                        const void **scale,
                                        int *kbp);

/** Same lookup keyed on the raw device pointer, for callers that only ever held
 * one (the routed-MoE path takes float* activations, not tensors). */
int pulsar_gpu_mxfp8_act_cache_get_e4m3_ptr(const void *ptr,
                                            uint64_t n_tok,
                                            uint64_t in_dim,
                                            const void **data,
                                            const void **scale,
                                            int *kbp);


int pulsar_gpu_rms_norm_plain_rows_tensor(
        pulsar_gpu_tensor       *out,
        /** Optional BF16 copy of the normalised rows (L086 T3), from
         *  pulsar_gpu_bf16_act_slot().  NULL emits nothing.  This buffer's
         *  consumer is a BF16 GEMM -- pulsar_gpu_matmul_f32_tensor resolves a
         *  bf16 weight and runs the shared bf16 core -- so emitting from the
         *  epilogue DELETES its convert pass rather than relocating it. */
        void                    *out_b,
        const pulsar_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps,
        /** L157: nonzero skips the f32 rows entirely (requires out_b).  Legal
         *  only when every consumer reads the bf16 copy -- the caller must also
         *  declare it with pulsar_gpu_act_note_f32_skipped_for() so a
         *  consumer that misses the bf16 slot refuses instead of converting
         *  unwritten bytes.  At hc_dim the f32 rows are 40% of this kernel's
         *  traffic (64 of 160 KB per row). */
        int                     skip_f32);

/** L157: declare that rows [keep_from, n_tok) of x's f32 store were skipped,
 * for a buffer that was never arm()ed (bf16 slot only), keyed like the slot. */
void pulsar_gpu_act_note_f32_skipped_for(const pulsar_gpu_tensor *x, uint64_t n_tok,
                                         uint64_t in_dim, uint32_t keep_from);

/** As below, but also emits the E4M3 + ue8m0 encoding into the activation-cache
 * slots, so a GEMV consuming this norm multiplies in the source's format.
 * NULL slots give the plain behaviour.
 *
 * @param out            destination, f32
 * @param x              input rows
 * @param model_map      base of the model mapping the norm weight lives in
 * @param model_size     its size; the bound the weight span is checked against
 * @param weight_offset  byte offset of the norm weight within that mapping
 * @param n              row width
 * @param eps            RMS epsilon
 * @param out_q          E4M3 activation-cache slot, or NULL for plain behaviour
 * @param out_sf         matching ue8m0 scale slot, or NULL
 * @param out_kbp        scale-table stride: k-blocks per row
 * @param out_b          bf16 activation-cache plane (pulsar_gpu_bf16_act_slot) for a
 *                       bf16-weight consumer, or NULL.  L159: the bf16 GEMM core
 *                       reads this plane and never converts f32 itself.
 * @param w_bf16         1 when this norm weight is stored bf16 (source format)
 *                       rather than f32. Storage only -- the value is promoted
 *                       to f32 before it multiplies, so an f32 tensor stays
 *                       bit-exact. Pass the TENSOR's type; never assume, the
 *                       drafter and the main model can differ.
 * @return 0 on success.
 */
int pulsar_gpu_rms_norm_weight_mx_tensor(
        pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps,
        void *out_q, void *out_sf, int out_kbp,
        void *out_b,
        int w_bf16);

int pulsar_gpu_rms_norm_weight_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps,
        int                     w_bf16);

/** L158: the rows twin of pulsar_gpu_rms_norm_weight_mx_tensor -- `rows`
 * independent rows normalised with the same weight, each row's E4M3 encoding
 * and UE8M0 block scales emitted into out_q/out_sf (pass NULLs for f32 only; `out` may be NULL when out_q or out_b is set -- no f32 row is stored then).
 * Used by the drafter's per-row norms so its GEMVs read a slot, not f32.
 * n must be a multiple of 256 when out_q is set (fails loudly otherwise).
 * @return 0 on success. */
int pulsar_gpu_rms_norm_weight_rows_mx_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x,
                                              const void *model_map, uint64_t model_size,
                                              uint64_t weight_offset, uint32_t n, uint32_t rows, float eps,
                                              void *out_q, void *out_sf, int out_kbp, void *out_b, int w_bf16);

int pulsar_gpu_rms_norm_weight_rows_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps,
        void                   *out_b,     ///< bf16 plane for a bf16-weight consumer, or NULL (L159)
        int                     w_bf16);

/** As below, but the Q half's E4M3 + E8M0 block-scale encoding is emitted from
 * the norm's own epilogue into the activation-cache slots, so the MXFP8
 * attn_q_b GEMM never runs a separate quantize pass over batch_qr_norm.  Pass
 * NULL slots for the plain behaviour.  Bit-exact: same value, same rounding the
 * standalone quantiser would have applied.
 *
 * Normalises the Q and KV halves in ONE launch -- they share the row loop, so
 * splitting them would read the same rows twice.
 *
 * @param q_out             normalised Q rows
 * @param q                 Q input rows
 * @param model_map         base of the model mapping the norm weights live in
 * @param model_size        its size; the bound both weight spans are checked against
 * @param q_weight_offset   byte offset of the Q norm weight
 * @param q_n               Q row width
 * @param kv_out            normalised KV rows
 * @param kv                KV input rows
 * @param kv_weight_offset  byte offset of the KV norm weight
 * @param kv_n              KV row width
 * @param rows              rows to normalise
 * @param eps               RMS epsilon
 * @param q_out_q           E4M3 activation-cache slot for Q, or NULL for plain behaviour
 * @param q_out_sf          matching ue8m0 scale slot, or NULL
 * @param q_out_kbp         scale-table stride: k-blocks per Q row
 * @param q_w_bf16          1 when the Q norm weight is stored bf16 rather than f32
 * @param kv_w_bf16         1 when the KV norm weight is stored bf16 rather than f32
 * @param q_skip_f32        Drop q_out's f32 store, leaving the E4M3 emission as
 *                          the buffer's ONLY content. Requires q_out_q. The
 *                          caller must arm the cache and call note_mxfp8() +
 *                          note_f32_skipped() straight after, and must NOT set
 *                          this while a debug dump of that tensor is active --
 *                          the dump reads f32 and would silently show stale
 *                          bytes.
 * @return 0 on success.
 */
int pulsar_gpu_dsv4_qkv_rms_norm_rows_mx_tensor(
        pulsar_gpu_tensor       *q_out,
        const pulsar_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        pulsar_gpu_tensor       *kv_out,
        const pulsar_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps,
        void                   *q_out_q,
        void                   *q_out_sf,
        int                     q_out_kbp,
        int                     q_w_bf16,
        int                     kv_w_bf16,
        int                     q_skip_f32);


/** positions (both RoPE entries below): optional int32 [n_tok] DEVICE array of
 * per-row absolute positions for multi-session banked batches (rows of
 * different sessions sit at unrelated positions).  NULL keeps the classic
 * consecutive pos0+t rule bit-exactly — the multiseq degeneracy invariant.
 * The launcher bounds-checks the array's SIZE; its VALUES are the caller's
 * contract (they are device-side, and a per-step D2H scan to validate them
 * would put host-visible work on the per-token path).  Values are used as
 * uint32 rotation positions: a negative entry rotates at a garbage angle
 * rather than faulting.  gpu_graph_multiseq_step_begin is the host-side
 * validator that every position is > 0 before any launch sees the array. */
/** Per-head RMS norm followed by the tail rope rotation, in one launch.
 *
 * The Q element type is read from `x` itself rather than passed: the RMS
 * reduction and the rotation stay in f32 either way, and only the STORES
 * narrow.
 *
 * @param x           queries, normed and rotated in place
 * @param n_tok       rows
 * @param n_head      heads per row
 * @param head_dim    per-head width
 * @param n_rot       rotary dimensions at the head's tail
 * @param pos0        absolute position of row 0 (ignored when `positions` is given)
 * @param n_ctx_orig  context length the RoPE settings were trained at
 * @param inverse     rotate backwards (used when replaying a rewind)
 * @param freq_base   RoPE base frequency
 * @param freq_scale  linear frequency scaling
 * @param ext_factor  YaRN extrapolation mix; 0 disables
 * @param attn_factor YaRN attention temperature correction
 * @param beta_fast   YaRN ramp start
 * @param beta_slow   YaRN ramp end
 * @param eps         RMS epsilon
 * @param positions   optional per-row absolute positions; see the note above
 * @return 0 on success.
 */
int pulsar_gpu_head_rms_norm_rope_tail_tensor(
        pulsar_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps,
        const pulsar_gpu_tensor *positions);

/* PULSAR_ATTN_PACK storage -- since the L111 unification (2026-08-27), ONE
 * row format serves EVERY KV buffer: the raw SWA ring, the compressed pool,
 * the drafter's ring, the MTP cache and prefill's current-chunk rows are all
 * the 384 B NVFP4 row
 *   [n_nope/2 e2m1 nibble bytes][n_nope/16 E4M3 scale codes][f32 row scale]
 *   [n_rot bf16 rope]
 * (224+28+4+128 at head_dim 512 / n_rot 64).  The nope dims are a LOSSY
 * re-quantization of the model's QAT e4m3 values, shipped on the measured
 * L111 verdict (net KL closer to the vLLM source than the retired e4m3 row,
 * drafter acceptance at/above it, -31%% KV reservation); the rope tail is
 * bf16 verbatim.  The e4m3 584 B row, its quantize recipe and every
 * backward-compat decode arm are GONE -- old session payloads and bank
 * snapshots refuse loudly and re-prefill; there is deliberately no
 * conversion loader (an FP4 re-encode misrounds ~33%% of blocks; bytes are
 * the values).  Requires n_rot == 64 and (head_dim - n_rot) %% 16 == 0. */
/** THE KV ROW GEOMETRY -- one definition, used by the kernels that index the
 * rows and by the engine that sizes the caches and payload spans (L159 inc 5).
 * It was defined twice, once per side of this seam, kept in step by a comment
 * and a start-up check that asked the backend for its number; both sides now
 * read these macros, and the check is a tautology that no longer exists.
 *
 * Packed attention KV row (NVFP4, L111 unification): per row
 *   [n_nope/2 E2M1 nibbles][n_nope/16 E4M3 block scales][f32 row scale]
 *   [n_rot bf16 rope]  = 384 B at head_dim 512 / n_rot 64.
 * Requires n_rot == PULSAR_ATTN_PACK_NROT and (head_dim - n_rot) a multiple
 * of PULSAR_KV4_NV_BLOCK; the graph alloc refuses any other shape.  Quantise
 * EXACTLY ONCE (attn_pack_store_kernel); every later move is a byte move, and
 * there is no conversion path from any other row format.  Bumping this layout
 * MUST bump PULSAR_SESSION_PAYLOAD_VERSION and PULSAR_BANK_KV_VERSION. */
#define PULSAR_ATTN_PACK_NROT 64u
#define PULSAR_ATTN_PACK_NOPE_ALIGN 64u   /* the kernels walk the nope dims in 64-wide lanes */
#define PULSAR_ATTN_PACK_NOPE(HD) ((HD) - PULSAR_ATTN_PACK_NROT)
#define PULSAR_ATTN_PACK_NIB(HD)  (PULSAR_ATTN_PACK_NOPE(HD) / 2u)
#define PULSAR_KV4_NV_BLOCK      16u
#define PULSAR_KV4_NV_NBLK(HD)   (PULSAR_ATTN_PACK_NOPE(HD) / PULSAR_KV4_NV_BLOCK)
#define PULSAR_ATTN_PACK_ROWBYTES(HD) \
    ((uint64_t)PULSAR_ATTN_PACK_NIB(HD) + PULSAR_KV4_NV_NBLK(HD) + 4u + \
     (uint64_t)PULSAR_ATTN_PACK_NROT * 2u)
/** Microscaling compressed-KV row (the indexer's FP4 cache): one E8M0 scale
 * byte per 32 elements, [HD/2 E2M1 nibble bytes][NBLK scale bytes]
 * (HD=128 -> 68 B/row).  CUTLASS-consumable layout; the GEMM re-tiles the
 * scales into its swizzled SF layout at use time. */
#define PULSAR_MXKV_BLOCK 32u
#define PULSAR_MXKV_NBLK(HD) (((HD) + PULSAR_MXKV_BLOCK - 1u) / PULSAR_MXKV_BLOCK)
#define PULSAR_MXKV_FP4_ROWBYTES(HD) (((HD) + 1u) / 2u + PULSAR_MXKV_NBLK(HD))
/** Accessors over the two macros for callers that hold head_dim as a runtime
 * value (tests, the engine's comp-row helper).  Same expression, one authority. */
uint64_t pulsar_gpu_attn_pack_rowbytes(uint32_t head_dim);
uint64_t pulsar_gpu_mxkv_fp4_rowbytes(uint32_t head_dim);

/** Quantise `n_rows` KV rows to E2M1 and store them packed.
 *
 * THE single E2M1 quantize of KV in the engine. Re-encoding already-packed FP4
 * misrounds ~33% of blocks (norm_kv.cu), so every later move of these rows --
 * scatter, fork, evict/restore, session save/load -- must be a BYTE move.
 *
 * @param x         f32 staging rows to quantise; mutated in place when keep_f32
 * @param packed    destination for the packed rows
 * @param out_row0  first row of `packed` to write
 * @param n_rows    rows to quantise
 * @param head_dim  per-head width
 * @param n_rot     rotary dimensions at the head's tail
 * @param keep_f32  write the dequantised values back into the f32 staging.
 *                  OBSERVER-ONLY -- consumers read the packed rows. Pass
 *                  gpu_graph_f32_store_observed_any() (L094).
 * @return 0 on success.
 */
int pulsar_gpu_attn_pack_quantize_store_tensor(
        pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim,
        uint32_t          n_rot,
        bool              keep_f32);

/** Fused rope + QAT for the indexer q projection: one launch replacing the
 * rope_tail + indexer_qat pair over the same tensor; bit-exact vs that
 * sequence (shared rotation device fn, same QAT body, same order). */
int pulsar_gpu_dsv4_indexer_rope_qat_tensor(
        pulsar_gpu_tensor *x,          /* f32 rope staging, mutated in place */
        pulsar_gpu_tensor *packed,     /* MXKV FP4 rows out -- the only Q output */
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions);

/** QAT-roundtrip n_rows f32 rows of x in place AND store them MXKV-FP4-packed
 * into `packed` at rows [out_row0, out_row0+n_rows).  The f32 result in x is
 * bit-identical to the fused rope+QAT entry above.
 *
 * @param x         f32 indexer rows, round-tripped in place
 * @param packed    destination for the MXKV-FP4 rows
 * @param out_row0  first row of `packed` to write
 * @param n_rows    rows to process
 * @param head_dim  per-head width
 * @param keep_f32  write the dequantised values back into the f32 staging.
 *                  OBSERVER-ONLY -- consumers read the packed rows. Pass
 *                  gpu_graph_f32_store_observed_any() (L094).
 * @return 0 on success.
 */
int pulsar_gpu_dsv4_indexer_qat_pack_tensor(
        pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim,
        bool              keep_f32);

/* Tell the indexer score kernels the indexer compressed cache is stored
 * MXKV-FP4-packed (68 B/row at head_dim 128) instead of f32. */

/* Every KV buffer is PULSAR_ATTN_PACK rows -- 584 B at head_dim 512 -- so the
 * attention readers and raw-KV writers below take NO format parameter.
 *
 * They carried one (`raw_f16`, later `raw_pack`) until 2026-08-17. Three storage
 * formats died with it: __half for the main ring, f32-holding-f16-values for the
 * drafter's, and the f32 batch buffer prefill handed attention for the current
 * chunk. None of the three is a format the source model uses. */

/** As below, but also emits the grouped E4M3 encoding for the MX blocks this
 * kernel rewrites -- head dims [head_dim - n_rot, head_dim).  It is the second
 * half of the attn-output "a" activation: the fp16 attention epilogue emits
 * [0, head_dim - n_rot) and this emits the rest, because this kernel rewrites
 * that range IN PLACE after attention has already run.  Pass NULL slots for
 * the plain behaviour. */
int pulsar_gpu_rope_tail_mx_tensor(
        pulsar_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        const pulsar_gpu_tensor *positions,
        void             *gact_data,
        void             *gact_scale,
        int               gact_kbp,
        uint32_t          gact_slab,
        uint32_t          n_groups);

int pulsar_gpu_rope_tail_tensor(
        pulsar_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        const pulsar_gpu_tensor *positions);

/** Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * pulsar_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int pulsar_gpu_store_raw_kv_tensor(
        pulsar_gpu_tensor       *raw_cache,
        const pulsar_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

/** Banked mode (positions/seq_id non-NULL): row t stores to bank seq_id[t]'s
 * ring at slot positions[t] %% raw_cap over the whole pool (raw_cache = the
 * bank slab, byte-bounded by n_banks); pos0 is ignored.  Dead rows (seq_id
 * out of pool) store nothing.  NULL/NULL/1 = classic consecutive store. */
int pulsar_gpu_store_raw_kv_batch_tensor(
        pulsar_gpu_tensor       *raw_cache,
        const pulsar_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                n_banks);

/** Same scatter, but the source rows are ALREADY PULSAR_ATTN_PACK.  Prefer this
 * wherever the caller has already packed the batch: re-quantising a buffer that
 * has been round-tripped is the ~5%-misround pattern the norm_kv header warns
 * about, and a byte copy makes the ring agree with what attention read by
 * construction instead of by argument. */
int pulsar_gpu_store_raw_kv_batch_packed_tensor(
        pulsar_gpu_tensor       *raw_cache,
        const pulsar_gpu_tensor *packed,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                n_banks);

/** =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int pulsar_gpu_compressor_update_tensor(
        const pulsar_gpu_tensor *kv_cur,
        const pulsar_gpu_tensor *sc_cur,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        pulsar_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int pulsar_gpu_compressor_store_batch_tensor(
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

/** L120 value-half: the ratio-4 two-group window shift as a standalone entry
 * (the emit path runs it inside compressor_update).  The rewind-time window
 * replay re-runs store+shift over committed projections. */
int pulsar_gpu_compressor_shift_ratio4_tensor(
        pulsar_gpu_tensor *state_kv,
        pulsar_gpu_tensor *state_score,
        uint32_t           head_dim);

int pulsar_gpu_compressor_prefill_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int pulsar_gpu_compressor_prefill_state_ratio4_tensor(
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv_tail,
        const pulsar_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

/** As below, but the fp16 tier additionally emits the grouped E4M3 encoding of
 * batch_heads for the attn-output "a" projection.  *mx_out is set to 1 ONLY if
 * that tier actually ran and slots were supplied -- any other tier leaves it 0,
 * and the caller must not note() the activation cache in that case.  Note that
 * rope_tail rewrites the rope tail afterwards and must emit those blocks too;
 * both halves are required before the encoding is complete. */
int pulsar_gpu_attention_prefill_raw_heads_mx_tensor(
        pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window,
        uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab,
        uint32_t n_groups, uint32_t n_nope, int *mx_out,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep);

int pulsar_gpu_attention_prefill_raw_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep);

/** Batched decode attention.  The trailing descriptor quad enables multi-
 * session banked mode: positions/seq_id are int32 [n_tokens] DEVICE arrays
 * (row t's absolute query position and TRUE bank id), comp_cap is the
 * per-bank compressed-row stride and n_banks the pool size; the raw/comp KV
 * operands are then whole bank pools addressed at seq_id*raw_cap /
 * seq_id*comp_cap, with the raw window, ring start, and visible compressed
 * count derived per row from the position.  Scalar n_raw/raw_start are
 * ignored AND unvalidated in banked mode (pass 0); raw_cap must still be
 * the true per-bank ring capacity.  The per-row visible compressed count is
 * (qpos+1)/ratio — identical to the engine's classic single-session decode,
 * which emits a step's compressed row BEFORE attention, so at an emit step
 * (qpos ≡ ratio-1 mod ratio) the row sees the row emitted that same step.
 * DRIVER CONTRACT: in banked mode every bank's compressed rows for the
 * current step — including same-step emits — must be written before the
 * attention launch.  Scalar n_comp is the cross-bank superset clamp, a
 * SAFETY bound only: if it bites (a bank lagging behind its own frontier,
 * e.g. mid-prefill), the row reads fewer rows instead of garbage but its
 * output DIVERGES from single-session — never co-schedule such a bank.
 * seq_id must carry TRUE bank ids in [0, n_banks): a row whose id is out
 * of range (stale slot, -1 sentinel) reads nothing and gets zero head
 * outputs — fail-visible, never a wild read.  Banked mode requires a
 * nonzero window <= 256 (the kernel's per-row raw scratch bound); banked
 * argument rejections return 0 and print the reason to stderr.  Pass
 * NULL/NULL/0/1 for the classic single-cache behavior — bit-exact.
 * non_causal selects the raw visibility rule: 0 = a row sees ring rows at or
 * before its own position; nonzero = every ring row up to the last one (the
 * drafter's raw-window forward, whose queries all see the whole draft).
 * Both rules run on the same fp16 tensor-core kernel (L166); the flag
 * changes which rows are visible, never how a visible row is folded. */
int pulsar_gpu_attention_decode_raw_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep);

int pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep);

int pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        const pulsar_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep);

int pulsar_gpu_attention_prefill_static_mixed_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        /* gact_*: grouped E4M3 emission for the attn-output "a" GEMM, same
         * contract as the raw _mx entry: NULL = no emission; *mx_out set to 1
         * ONLY when the fp16 tier ran and wrote the encoding.  The mixed tier
         * is the per-layer traffic carrier, so this is where the emission
         * matters (the raw entry runs twice a prefill, this one per layer). */
        void                    *gact_data,
        void                    *gact_scale,
        int                      gact_kbp,
        uint32_t                 gact_slab,
        uint32_t                 n_groups,
        uint32_t                 n_nope,
        int                     *mx_out,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep);


int pulsar_gpu_attention_output_batch_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *heads,
        uint32_t                n_tokens);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/MXFP4 routed experts.
 */

/** SwiGLU over gate/up.  The result's E4M3 + E8M0 block-scale encoding is
 * emitted from the epilogue into the activation-cache slots, so the shared_down
 * GEMM never runs a separate quantize pass over the mid tensor.  `mid_dim` is
 * the row width (the launch is flat over n = rows * mid_dim, so the MX row/col
 * must be recovered by division).  NULL slots give the plain behaviour.
 *
 * `skip_f32` additionally drops the f32 store of `out`, leaving the encoding as
 * the buffer's ONLY content -- the dead-store half of the same one-two that
 * took the f16 side from +2.5% to +6.6% (see d967327).  Requires out_q: without
 * an encoding the kernel would write nothing at all, so that combination is
 * refused rather than silently downgraded.  The caller must arm the cache and
 * call both note_mxfp8() and note_f32_skipped() straight after. */
int pulsar_gpu_swiglu_mx_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *gate,
        const pulsar_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight,
        void                   *out_q,
        void                   *out_sf,
        int                     out_kbp,
        uint32_t                mid_dim,
        int                     skip_f32);

int pulsar_gpu_add_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *a,
        const pulsar_gpu_tensor *b,
        uint32_t                n);

int pulsar_gpu_directional_steering_project_tensor(
        pulsar_gpu_tensor       *x,
        const pulsar_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int pulsar_gpu_router_select_batch_tensor(
        pulsar_gpu_tensor       *selected,
        pulsar_gpu_tensor       *weights,
        pulsar_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const pulsar_gpu_tensor *logits,
        const pulsar_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);


int pulsar_gpu_routed_moe_batch_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *up,
        pulsar_gpu_tensor       *mid,
        pulsar_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const pulsar_gpu_tensor *selected,
        const pulsar_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const pulsar_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens);


/** Small-batch (n_tokens 2..4) rich-expert FFN over the packed CUTLASS MXFP4 weights:
 * direct GEMVs, one launch for gate+up+SwiGLU and one for down, per (token, slot)
 * slots, no sort, no host readback.  The activation is the producer's E4M3
 * (act_q/act_sf/act_kbp, act_sf in the pulsar_mx_sfoff SWIZZLE with act_kbp blocks per
 * row); without it, or with a pitch that does not match this call's geometry, the call
 * refuses.  The gate/up GEMV emits mid as E4M3 + ue8m0 from its epilogue and the down
 * GEMV reads that pair -- no f32 mid exists.  down_out gets one pre-weighted FFN result
 * per slot at [slot*out_dim]; the caller sums the n_expert slices per token (moe_sum).
 * selected/rweights are the device [n_tokens,n_expert] routing outputs.  Returns 0 on
 * success. */
int pulsar_cutlass_expert_ffn_gemv_small(
        float          *down_out,
        const int32_t  *selected,
        const float    *rweights,
        const uint8_t  *gate_w,
        const uint8_t  *up_w,
        const uint8_t  *down_w,
        uint64_t        gate_stride,
        uint64_t        gate_data_bytes,
        uint64_t        down_stride,
        uint64_t        down_data_bytes,
        float           clamp,
        int             n_tokens,
        int             n_expert,
        unsigned        n_total_expert,
        int             in_dim,
        int             mid_dim,
        int             out_dim,
        const void     *act_q,
        const void     *act_sf,
        int             act_kbp);

/** Grouped (ptr-array) MXFP4 prefill FFN: runs EVERY active expert's gate/up/down as a single
 * blockscaled grouped GEMM launch each -- replacing the per-expert host loop + blocking offsets
 * readback in routed_moe_launch_cutlass. Per-group problem shapes, A/B/D + SFA/SFB pointer arrays
 * and SF-layouts are built on device from `counts`/`padded_offsets`; there is no host readback.
 *
 * The caller must gather each expert's tokens to 128-ROW-PADDED offsets (padded_offsets[e], a
 * multiple of 128) into x_gathered[padded_total,in_dim] + w_gathered[padded_total], PRE-ZEROING
 * padding rows, because the SM120 SF atom spans 128 rows and each group's SF must start on a
 * 128-row boundary. Output ffn_out[padded_total,out_dim] holds one pre-weighted result per padded
 * row; the caller scatters the real rows into the flat down buffer (via the same padded map),
 * then moe_sum reduces. `padded_total` is a host upper bound (multiple of 128); inactive experts
 * carry M=0 and contribute nothing. `scratch` must be >= the *_scratch_bytes value. No host sync. */
size_t pulsar_cutlass_grouped_moe_scratch_bytes(
        int padded_total, int n_total_expert, int in_dim, int mid_dim, int out_dim);
int pulsar_cutlass_grouped_moe(
        float          *ffn_out,
        const float    *x_gathered,
        const float    *w_gathered,
        const uint8_t  *gate_w,
        const uint8_t  *up_w,
        const uint8_t  *down_w,
        uint64_t        gate_stride,
        uint64_t        gate_data_bytes,
        uint64_t        down_stride,
        uint64_t        down_data_bytes,
        float           clamp,
        int             n_total_expert,
        int             in_dim,
        int             mid_dim,
        int             out_dim,
        const uint32_t *counts,
        const uint32_t *padded_offsets,
        int             padded_total,
        uint8_t        *scratch,
        size_t          scratch_bytes,
        /* Optional pre-encoded activation. The norm that produced this tensor
         * already emitted E4M3 + ue8m0 into the activation cache, in the same
         * scheme this GEMM's packer uses (verified bit-identical, payload and
         * scale byte). Pass the cache's data/scale plus a row->source-token map
         * and the gather moves 1-byte codes straight into the A operand instead
         * of copying f32 and re-encoding it: a quarter of the activation traffic
         * and one fewer pass over [padded_total x in_dim].
         * act_sf is read through the ENGINE's mx_sfoff swizzle and written
         * through the CUTLASS tile atom -- only the location differs, never the
         * byte. row_src_tok[R] < 0 marks a padding row (zero payload + scale).
         * Pass act_q = NULL to take the f32 pack path, in which case
         * x_gathered must be populated and its padding rows pre-zeroed. */
        const void     *act_q,
        const void     *act_sf,
        int             act_kbp,
        const int32_t  *row_src_tok);

/** Single-projection W4A8 GEMM for MIXED type-40 + iq2/q2k layers. Computes out[T,out_dim] =
 * x[T,in_dim] . W[out_dim,in_dim]^T for ONE expert's type-40 CUTLASS weight (data at W_d, swizzled
 * SFB at W_sf), packing x to E4M3 dynamic block-scaled activations -- bit-identical to a single
 * projection of the uniform grouped path. Caller gathers x contiguously (T = tokens for that
 * expert) and sizes scratch once via pulsar_cutlass_proj_scratch_bytes(). No allocation, no sync. */
size_t pulsar_cutlass_proj_scratch_bytes(int T, int in_dim, int out_dim);

/** Grouped single-projection W4A8 GEMM for MIXED layers -- one device-built ptr-array grouped GEMM
 * over 128-padded gathered activations: out[padded_total,out_dim] = x_gathered . W^T for every
 * active expert (W_base+e*W_stride data, +W_data_bytes swizzled SFB). No host readback, no per-expert
 * sync; bit-identical to the per-expert single-proj path (same pack + gather order + GEMM). Padding
 * rows must be pre-zeroed. Caller sizes scratch once via pulsar_cutlass_grouped_proj_scratch_bytes().
 * reuse_packed_a: skip the E4M3 activation pack and consume the encoding a PREVIOUS call left in
 * this same scratch -- valid only when x_gathered, padded_total, in_dim and scratch are identical
 * to that call's (the gate/up pair over one gathered activation). Ends the case-A double-encode. */
size_t pulsar_cutlass_grouped_proj_scratch_bytes(int padded_total, int n_total_expert, int in_dim, int out_dim);
int pulsar_cutlass_grouped_proj(float *out, const float *x_gathered,
        const uint8_t *W_base, uint64_t W_stride, uint64_t W_data_bytes,
        int n_total_expert, int in_dim, int out_dim,
        const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
        uint8_t *scratch, size_t scratch_bytes, int reuse_packed_a,
        /* Producer handover (L089), same contract as pulsar_cutlass_grouped_moe:
         * act_q/act_sf are the E4M3 + ue8m0 the producing norm already emitted
         * (act_sf in the pulsar_mx_sfoff SWIZZLE), and row_src_tok[R] says which
         * token row padded row R came from, < 0 marking a padding row that gets a
         * zero payload and scale.  When all three are supplied, the f32
         * x_gathered is NOT READ.  Pass NULL/NULL/0/NULL to pack from it. */
        const void *act_q, const void *act_sf, int act_kbp,
        const int32_t *row_src_tok);

/** Single-projection W4A8 GEMV for MIXED type-40 layers at decode/small-batch (n<=4): lean fp4-weight
 * GEMV with E4M3-roundtripped f32 activations (same function as the prefill grouped GEMM), one launch
 * over all (token,expert) slots, no per-expert loop/host sync. mid/down_out are pair-layout f32. */
int pulsar_cutlass_gemv_gateup(float *mid, const int32_t *selected, const float *rweights,
        const uint8_t *gate_w, const uint8_t *up_w, uint64_t gate_stride, uint64_t gate_data_bytes,
        float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim,
    const void *act_q, const void *act_sf, int act_kbp);
/** L158 inc 5: mid arrives as the MoE stage's E4M3 encoding (mid_q/mid_sf in the
 * VEC32 swizzle at pitch mid_kbp, rows = (token, slot) pairs); no f32 mid. */
int pulsar_cutlass_gemv_down(float *down_out, const int32_t *selected,
        const uint8_t *down_w, uint64_t down_stride, uint64_t down_data_bytes,
        int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim,
        const void *mid_q, const void *mid_sf, int mid_kbp);


/** =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */


int pulsar_gpu_hc_weighted_sum_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);


/** Same, but also emits norm_out's E4M3 + swizzled E8M0 encoding (norm_out_q,
 * norm_out_sf, pitch norm_out_kbp) from the same registers that produce the f32
 * norm_out, so the MXFP8 consumers do not need a separate quantize pass over
 * the tensor.  Bit-exact: fmaxf is exact and max is order-independent, so any
 * reduction order yields the same amax, shared exponent and E4M3 bytes.
 * norm_out_q/sf may be NULL, in which case only the f32 norm_out is written.
 *
 * NOTE the _f16 in the name is a MISNOMER kept for call-site stability: this
 * emitted __half until the F16 weights were retired (2026-08-16) and it has
 * emitted E4M3 ever since.  There is no f16 anything on this path -- the
 * shipped artifact contains ZERO F16 tensors.  See ledger L079.
 *
 * @param out                 the HC collapse output
 * @param norm_out            the f32 normalised rows
 * @param norm_out_q          E4M3 activation-cache slot for norm_out; NULL writes only f32
 * @param norm_out_sf         matching ue8m0 scale slot
 * @param norm_out_kbp        scale-table stride: k-blocks per row
 * @param norm_out_b          bf16 copy of norm_out (act-cache xb slot); NULL = skip (L086 T3)
 * @param norm_f32_keep_from  f32 rows BELOW this index are dead stores and are
 *                            skipped; 0 stores all. Pass nonzero ONLY with
 *                            note_f32_skipped(), so the skipped-store hazard
 *                            check refuses instead of reading unwritten bytes.
 * @param split               per-stream split of the mix
 * @param mix                 the HC mix projection output
 * @param residual_hc         the HC residual being collapsed
 * @param model_map           base of the model mapping the weights live in
 * @param model_size          its size; the bound every weight span is checked against
 * @param scale_offset        byte offset of the HC per-channel scale
 * @param base_offset         byte offset of the HC per-channel base
 * @param norm_weight_offset  byte offset of the norm weight
 * @param n_rows              rows to process
 * @param n_embd              embedding width
 * @param n_hc                HC stream count
 * @param sinkhorn_iters      Sinkhorn normalisation iterations for the collapse weights
 * @param eps                 collapse epsilon
 * @param norm_eps            RMS epsilon for the norm half
 * @param norm_w_bf16         1 when the norm weight is stored bf16 rather than f32
 * @return 0 on success.
 */
int pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *norm_out,
        void                    *norm_out_q,
        void                    *norm_out_sf,
        int                      norm_out_kbp,
        void                    *norm_out_b,
        uint32_t                 norm_f32_keep_from,
        pulsar_gpu_tensor       *split,
        const pulsar_gpu_tensor *mix,
        const pulsar_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_rows,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps,
        /** Storage of norm_w (attn_norm / ffn_norm): 1 = bf16, 0 = f32. */
        int                     norm_w_bf16);


/** Fused plain-RMSNorm + HC-mix GEMV (decode, n_tok == 1).  Byte-identical to
 * a one-row plain RMSNorm (pulsar_gpu_rms_norm_plain_rows_tensor) followed by the
 * matmul for `w_type`; see the kernel
 * comment in pulsar_cuda_hc_router.cu for the order argument.  `x` is an HC
 * residual CARRIER (pulsar_hc_t storage, PULSAR_HC_ELT_SIZE bytes/sample), not f32.
 *
 * @param out            destination, f32
 * @param model_map      base of the model mapping the mix weight lives in
 * @param model_size     its size; the bound the weight span is checked against
 * @param weight_offset  byte offset of the mix weight within that mapping
 * @param in_dim         input width (K)
 * @param out_dim        output width (N)
 * @param x              the HC residual carrier to norm and mix
 * @param eps            RMS epsilon
 * @param w_type         ds4 tensor type of the mix weight: 1 F16, 30 BF16,
 *                       0 F32. Templated rather than F16-gated -- the fusion is
 *                       about avoiding a scratch round trip, not about the
 *                       weight being 2 bytes.
 * @return 0 on success.
 */
int pulsar_gpu_hc_norm_mix_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        float                   eps,
        uint32_t                w_type);

int pulsar_gpu_output_hc_weights_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int pulsar_gpu_hc_expand_split_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *block_out,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int pulsar_gpu_hc_expand_add_split_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *block_out,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/** L149: min-p candidate prefilter over F32 logits rows. For each of n_rows
 * rows (row r at logits + logits_offset_bytes + r*row_stride_elems floats,
 * n_vocab wide) writes to out row r (int32, stride 3 + 2*cap):
 *   [0] count of finite logits v with v >= max + delta (ALL of them counted,
 *       even past cap), [1] id of the row max (first finite maximum, lowest id
 *       on ties -- matches the host scan), [2] the max logit's float bits,
 *   [3 .. 3+cap) the candidate ids in ascending id order,
 *   [3+cap .. 3+2cap) their logits as float bits, same order.
 * When count > cap the candidate arrays are not written. A row with no finite
 * logit reports count 0. `delta` is the caller's floor below the max in logit
 * units (T * ln(min_p) minus a margin); the compare is a single float add
 * and compare, reproducible on the host. Non-blocking; runs on the engine
 * stream. Returns 0 on bad arguments or launch failure. */
int pulsar_gpu_minp_prefilter_rows(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *logits,
        uint64_t                logits_offset_bytes,
        uint32_t                n_rows,
        uint32_t                row_stride_elems,
        uint32_t                n_vocab,
        float                   delta,
        uint32_t                cap);

/** L150: bank-batched markov refine. One launch per draft position serves
 * n_banks banks (one markov_w2 stream shared across them); byte-exact per
 * bank with the single-bank kernels. Layouts: refined_logits [n_banks][vocab]
 * f32; ids_dev [n_banks][ids_stride] i32 with [b][0] seeded by the caller
 * (chain) and winners written to [b][pos+1]; the bank's base logits row
 * for position pos is base_logits row (base_row_dev[b] + pos), rows
 * base_row_stride_bytes apart. The chain runs positions 0..n_draft-1 with the
 * device feed; the single step takes the previous token per bank from prev_dev
 * ([n_banks] i32, caller-written -- the sampled path's host draw). */
#define PULSAR_DSPARK_BANKS_MAX 8u
int pulsar_gpu_dspark_markov_chain_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t n_draft, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_bf16);
int pulsar_gpu_dspark_markov_step_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev, const pulsar_gpu_tensor *prev_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t pos, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_bf16);

/** DSpark Markov + confidence heads */

int pulsar_gpu_dspark_markov_chain_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_draft, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_bf16);
int pulsar_gpu_dspark_markov_step_model(
        pulsar_gpu_tensor       *refined_logits,
        int32_t               *refined_id_dst,
        const pulsar_gpu_tensor *base_logits,
        const void             *dspark_model_map,
        uint64_t                dspark_model_size,
        uint64_t                markov_w1_offset,
        uint64_t                markov_w2_offset,
        int32_t                prev_token,
        uint32_t               vocab_size,
        uint32_t               embed_dim,
        /* Storage of markov_w1 and markov_w2: 1 = bf16 (source format), 0 =
         * f32.  Separate flags because they are separate tensors.  The step
         * streams all of markov_w2 at one FMA per element, so its width sets
         * the kernel's runtime, not just its footprint. */
        int                    w1_bf16,
        int                    w2_bf16);

int pulsar_gpu_dspark_hc_mean_reduce_batch(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *hc_batch,
        uint32_t               n_embd,
        uint32_t               n_hc,
        uint32_t               n_tokens);

/* plan-92 P0: per-row teacher top-64 ids + f16 logits + f16 tail logsumexp
 * over a block of logits rows, written at row offset row0 of the chunk-sized
 * output tensors. `inexact` counts (never observed, verified per row) cases
 * where the candidate union missed the exact top-64. Collection-rate code. */
int pulsar_gpu_distill_top64_tensor(
        const pulsar_gpu_tensor *logits_rows,
        uint32_t               n_rows,
        uint32_t               vocab,
        pulsar_gpu_tensor       *top_ids,
        pulsar_gpu_tensor       *top_vals,
        pulsar_gpu_tensor       *tail_lse,
        pulsar_gpu_tensor       *inexact,
        uint32_t               row0);

/** DSpark confidence head: per block position, confidence that the draft is
 * accepted. hidden = post-hc_head drafter hidden [n_positions, hidden_dim];
 * token_ids = block token per position; markov_w1/proj resolved from the dspark
 * model map. Drives confidence-scheduled verification (sizing the draft length). */
int pulsar_gpu_dspark_confidence_score_model(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *hidden,
        const pulsar_gpu_tensor *token_ids,
        const void             *dspark_model_map,
        uint64_t                dspark_model_size,
        uint64_t                markov_w1_offset,
        uint64_t                proj_offset,
        uint32_t                n_positions,
        uint32_t                hidden_dim,
        uint32_t                embed_dim,
        uint32_t                vocab_size,
        /* Storage of markov_w1 and proj respectively -- they are separate
         * tensors and nothing forces them to agree, so do not collapse these
         * into one flag. */
        int                     w1_bf16,
        int                     proj_bf16);


/** ===========================================================================
 * Wrong-arch build trap (checked at startup by pulsar_gpu_init).
 *
 * A plain `make` rebuilds stale CUDA objects WITHOUT CUDA_ARCH=sm_120f, so
 * they compile for nvcc's default arch (sm_75 on CUDA 13).  The binary still
 * links and loads on the GB10, then dies mid-run with an opaque device assert
 * the first time a kernel needing sm_120f features launches — and `make test`
 * cannot catch it.  Every nvcc-compiled TU that includes this header (all of
 * src/cuda/ does, directly or via pulsar_cuda_internal.h) registers the
 * __CUDA_ARCH_LIST__ it was compiled with; pulsar_gpu_init() verifies each TU
 * carries code for the device's SM family and aborts startup with an
 * actionable message instead.  Per-TU registration also catches MIXED builds
 * (one stale object rebuilt default-arch by a plain `make`).  Host C compiles
 * never define __CUDA_ARCH_LIST__, so this block is nvcc-only; the `inline`
 * list head merges into one copy per linked binary, so any subset of objects
 * links (no central symbol required).
 */
#if defined(__cplusplus) && defined(__CUDA_ARCH_LIST__)
typedef struct pulsar_tu_archs {
    const char            *file;    /* __BASE_FILE__ of the registering TU */
    const int             *archs;   /* __CUDA_ARCH_LIST__ entries, e.g. 1200 */
    int                     n_archs;
    struct pulsar_tu_archs   *next;
} pulsar_tu_archs;
inline pulsar_tu_archs *pulsar_tu_archs_head = nullptr;
namespace {
const int pulsar_tu_arch_list_[] = { __CUDA_ARCH_LIST__ };
pulsar_tu_archs pulsar_tu_archs_rec_ = {
    __BASE_FILE__, pulsar_tu_arch_list_,
    (int)(sizeof(pulsar_tu_arch_list_) / sizeof(pulsar_tu_arch_list_[0])), nullptr
};
__attribute__((constructor)) void pulsar_tu_archs_register_(void) {
    pulsar_tu_archs_rec_.next = pulsar_tu_archs_head;
    pulsar_tu_archs_head = &pulsar_tu_archs_rec_;
}
}
#endif

#endif

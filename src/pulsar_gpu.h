/** @file
 *  The backend seam: the only interface between the engine and a GPU backend (tensors, streams,
 *  kernels, the device sampler, TP row lane, vision towers and QSA attention). */
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
/** The pulsar_elt_fmt of a stored HC residual sample (bf16), paired with PULSAR_HC_ELT_SIZE. */
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
/** The pulsar_elt_fmt of the stored attention output (bf16), paired with PULSAR_HEADS_ELT_SIZE. */
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
/** The pulsar_elt_fmt of the stored Q activation (f16), paired with PULSAR_Q_ELT_SIZE. */
#define PULSAR_Q_ELT_FMT  PULSAR_ELT_F16   /* pairs with pulsar_q_t; bridge assert CUDA-side */

/** Shared-expert gate/up staging element size (L033 increment 2).  Same
 * contract as the two above: this is the STORED width the alloc, the byte
 * budget, and any host stride math must agree on; the producer (the mxfp8
 * GEMM) and the consumer (the swiglu fold) both derive their kernel type from
 * the tensor's esz at runtime, never from this macro.  f16 today. */
#define PULSAR_SHARED_ACT_ELT_SIZE 2u
/** The pulsar_elt_fmt of the shared-expert gate/up staging (f16), paired with PULSAR_SHARED_ACT_ELT_SIZE. */
#define PULSAR_SHARED_ACT_ELT_FMT  PULSAR_ELT_F16   /* f16 staging; runtime-esz consumers */

/** The batched speculative lane's per-forward ROW BUDGET: the most rows one
 * shared verify forward may carry (each co-scheduled session's committed row
 * plus its draft rows).  It is the M-neutral range, PULSAR_GPU_MNEUTRAL_ROWS_MAX:
 * at or under it every row is a decode row on the width-neutral GEMV arm, so
 * a session's verify numerics do not depend on who else is drafting.  L117
 * (2026-08-27) raised the budget to 32 with the slab; past 16 the step's
 * row-kind inference found no single-row runs, declared zero decode rows, and
 * every row ran on the tensor-core arm -- a bank's numerics then depended on
 * the batch width (Tyler, 2026-09-05: budget back to 16; L177).  The server's
 * allocator fits draft depths to this and the ALL_ROWS head refuses above it.
 * 2026-09-30 (Tyler: batch invariance is not a requirement -- "if we get a
 * slightly different token when we batch versus when we don't; who cares?"):
 * back to the slab's 32 rows; a step past 16 runs its verify rows on the
 * tensor-core arms, graded by the reference gate like the TP split-K. */
#define PULSAR_SPEC_ROW_BUDGET PULSAR_SPEC_LOGITS_ROWS

/** spec_logits row CAPACITY (>= PULSAR_SPEC_ROW_BUDGET, asserted in imatrix.cpp):
 * the slab the batched lane's head writes,
 * NOT prefill_cap.  The budget the lane admits against is PULSAR_SPEC_ROW_BUDGET
 * above (server_sched.cpp fits draft depths to it, session_multiseq.cpp's ALL_ROWS
 * head refuses above it); a plain PULSAR_MSEQ_MAX-row step must fit
 * (static_assert PULSAR_MSEQ_MAX <= PULSAR_SPEC_LOGITS_ROWS, imatrix.cpp).
 * spec_logits and dspark_row_meta
 * are sized by this one constant (gpu_diag.cpp) -- guards on row indices
 * must check against THIS, or the number lives only in a comment. */
#define PULSAR_SPEC_LOGITS_ROWS 32u   /* L117 2026-08-27: 16 -> 32. The 16-row
 * ceiling squeezed per-bank draft depth at c3+ (c4: K~3 vs solo K~8, the
 * measured sublinear c4 scaling); the row cost (measured live since L263,
 * pulsar_engine_lane_cost; L214's refit put it at 7.17 ms) has no cliff, so a 32-row slab (+8.3 MB logits) lets the
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

/** @brief Select device 0 and check it can run pulsar (sm_75 or newer, and the archs this build carries).
 *  @return 1 on success, 0 (said on stderr) when the device is unusable */
int pulsar_gpu_init(void);
/** L321: can THIS device run a weight layout (a PULSAR_TENSOR_* id)?  1 yes; 0 no, with *need (if not NULL) naming
 *  what the layout's kernels require.  The engine asks it for every layout a model holds, after pulsar_gpu_init and
 *  before any weight reaches the GPU, so an older card refuses a model by name at load instead of tripping a kernel. */
/** @param layout a PULSAR_TENSOR_* weight layout id
 *  @param need   receives a static string naming the requirement on a 0 return; may be NULL
 *  @return 1 when the device can run the layout, else 0 */
int pulsar_gpu_layout_supported(uint32_t layout, const char **need);
/** @brief Synchronize the device, clear the routed-expert flags and weight caches and free the model arenas
 *  (process teardown, or before another engine opens in the same process). */
void pulsar_gpu_cleanup(void);

/** Running total of live tensor-alloc bytes (owned allocations only, views
 * excluded).  Snapshot around a session create to measure its true resident
 * cost; the server ledger commits that actual. */
/** @return live owned tensor bytes (cudaMalloc and managed), process-wide */
uint64_t pulsar_gpu_tensor_alloc_bytes_current(void);
/** Dry-run allocation on the calling thread: after dry_begin, the two tensor
 * allocators return placeholders and total the requested bytes instead of
 * touching the device; every device operation on a placeholder is a no-op
 * that reports success.  dry_end returns the totals (all bytes; the
 * cudaMallocManaged subset) and ends the run.  The session pricer runs the
 * real allocation code this way -- the price of a session and its allocation
 * are one piece of code. */
/** @brief Start a dry-run allocation on the calling thread (see above). */
void     pulsar_gpu_tensor_dry_begin(void);
/** @brief End the calling thread's dry run.
 *  @param bytes         receives every byte requested during the run
 *  @param managed_bytes receives the cudaMallocManaged subset */
void     pulsar_gpu_tensor_dry_end(uint64_t *bytes, uint64_t *managed_bytes);
/** @return nonzero while the calling thread is in a dry run */
int      pulsar_gpu_tensor_dry_active(void);
/** @brief Allocate an owned device tensor (cudaMalloc; f32 elements by default).
 *  @param bytes size in bytes (0 allocates 1)
 *  @return the tensor (free with pulsar_gpu_tensor_free), or NULL on failure; a placeholder during a dry run */
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

/** @brief Allocate n_elems * esz bytes with the element size and format recorded on the tensor.
 *  @param n_elems element count
 *  @param esz     element size in bytes (0 = 4); must match `fmt` (4 for F32, 2 for F16/BF16, any for BYTES)
 *  @param fmt     element format
 *  @return the tensor, or NULL on failure or a fmt/esz mismatch (said) */
pulsar_gpu_tensor *pulsar_gpu_tensor_alloc_elt(uint64_t n_elems, uint32_t esz,
                                               pulsar_elt_fmt fmt);
/** @brief Allocate an owned tensor in managed (demand-paged) memory: the large-KV-cache class
 *  (pulsar_gpu_should_use_managed_kv_cache).
 *  @param bytes size in bytes (0 allocates 1)
 *  @return the tensor, or NULL on failure */
pulsar_gpu_tensor *pulsar_gpu_tensor_alloc_managed(uint64_t bytes);
/** @brief A non-owning view of [offset, offset + bytes) of `base`; it inherits the element size and format.
 *  Freeing the view (pulsar_gpu_tensor_free) frees only the handle; `base` must outlive it.
 *  @param base   the tensor viewed
 *  @param offset byte offset into base
 *  @param bytes  view size in bytes
 *  @return the view, or NULL when the range leaves `base` */
pulsar_gpu_tensor *pulsar_gpu_tensor_view(const pulsar_gpu_tensor *base, uint64_t offset, uint64_t bytes);
/** @brief Free a tensor handle and, when it owns its memory, the memory.  NULL is a no-op.
 *  @param tensor the tensor or view */
void pulsar_gpu_tensor_free(pulsar_gpu_tensor *tensor);
/** @param tensor the tensor (may be NULL)
 *  @return its size in bytes; 0 for NULL */
uint64_t pulsar_gpu_tensor_bytes(const pulsar_gpu_tensor *tensor);
/** Raw device pointer without a synchronize (for building device pointer tables). */
/** @param tensor the tensor (may be NULL)
 *  @return its device pointer, or NULL; never for reading contents on the host */
void *pulsar_gpu_tensor_device_ptr(const pulsar_gpu_tensor *tensor);
/** @brief Fill the first `count` f32 elements with `value` (enqueued on the per-thread stream).
 *  @param tensor the tensor
 *  @param value  the fill value
 *  @param count  elements, not bytes; must fit the tensor
 *  @return 1 on success, 0 on a bad range or launch failure */
int pulsar_gpu_tensor_fill_f32(pulsar_gpu_tensor *tensor, float value, uint64_t count);
/** @brief Blocking host -> device copy (cudaMemcpy) into the tensor.
 *  @param tensor destination
 *  @param offset byte offset into the tensor
 *  @param data   host source
 *  @param bytes  byte count; the range must fit the tensor
 *  @return 1 on success, 0 on a bad range or copy failure */
int pulsar_gpu_tensor_write(pulsar_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
/** L289: the same write ENQUEUED on the per-thread stream (cudaMemcpyAsync H2D): the host does not wait for the
 *  GPU work queued before it, so a write in the middle of a step does not drain the step.  From pageable memory
 *  the driver stages `data` before returning; from pinned memory (pulsar_gpu_host_alloc) the copy reads `data`
 *  when the stream reaches it, so the caller keeps those bytes until the stream passes it (the step's drain). */
/** @param tensor destination
 *  @param offset byte offset into the tensor
 *  @param data   host source (see above for how long it must live)
 *  @param bytes  byte count; the range must fit the tensor
 *  @return 1 on success (enqueued), 0 on a bad range or enqueue failure */
int pulsar_gpu_tensor_write_async(pulsar_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
/** L289: page-locked host memory (cudaMallocHost) for buffers a step uploads with pulsar_gpu_tensor_write_async;
 *  NULL on failure.  pulsar_gpu_host_free(NULL) is a no-op. */
/** @param bytes size in bytes; 0 returns NULL
 *  @return pinned host memory (free with pulsar_gpu_host_free), or NULL */
void *pulsar_gpu_host_alloc(uint64_t bytes);
/** @brief Free memory from pulsar_gpu_host_alloc.
 *  @param p the buffer, or NULL */
void pulsar_gpu_host_free(void *p);

/** Fill a Q buffer from host f32, converting to the stored Q element type.
 * Host callers cannot see pulsar_q_t (it is CUDA-only), so a plain
 * tensor_write of floats into a narrowed Q buffer is both an overrun and a
 * reinterpretation -- and it compiles.  `n` counts ELEMENTS, not bytes. */
/** @param tensor    a Q buffer (PULSAR_Q_ELT_SIZE elements)
 *  @param off_elems destination offset, elements
 *  @param src       host f32 source, n elements
 *  @param n         element count
 *  @return 1 on success, 0 on a bad range or copy failure */
int pulsar_gpu_tensor_write_q_f32(pulsar_gpu_tensor *tensor, uint64_t off_elems,
                                  const float *src, uint64_t n);
/** @brief Blocking device -> host copy (cudaMemcpy, so after the work queued before it).
 *  @param tensor source
 *  @param offset byte offset into the tensor
 *  @param data   host destination
 *  @param bytes  byte count; the range must fit the tensor
 *  @return 1 on success, 0 on a bad range or copy failure */
int pulsar_gpu_tensor_read(const pulsar_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
/** Copy `bytes` from a raw DEVICE pointer (a slice of a scratch tensor a kernel wrote) to the host, after the
 *  work queued before it -- the importance-matrix collection's observation (L284 P15).  1 on success. */
/** @param device raw device source
 *  @param host   host destination
 *  @param bytes  byte count
 *  @return 1 on success, 0 on failure */
int pulsar_gpu_device_read(const void *device, void *host, uint64_t bytes);
/** @brief Blocking device -> device copy.
 *  @param dst        destination tensor
 *  @param dst_offset byte offset into dst
 *  @param src        source tensor
 *  @param src_offset byte offset into src
 *  @param bytes      byte count; both ranges must fit
 *  @return 1 on success, 0 on a bad range or copy failure */
int pulsar_gpu_tensor_copy(pulsar_gpu_tensor *dst, uint64_t dst_offset,
                          const pulsar_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
/** Same copy on the per-thread stream, WITHOUT blocking the host. Only for
 * destinations consumed by a later kernel on that same stream -- stream order
 * covers those. Anything the host reads back, or that crosses streams, keeps
 * the blocking form above. */
/** @param dst        destination tensor
 *  @param dst_offset byte offset into dst
 *  @param src        source tensor
 *  @param src_offset byte offset into src
 *  @param bytes      byte count; both ranges must fit
 *  @return 1 on success (enqueued), 0 on a bad range or enqueue failure */
int pulsar_gpu_tensor_copy_async(pulsar_gpu_tensor *dst, uint64_t dst_offset,
                                 const pulsar_gpu_tensor *src, uint64_t src_offset,
                                 uint64_t bytes);

/** Batched D2D copy: prepare a device-side descriptor table over fixed tensor
 * allocations once (whole-tensor copies, byte counts multiples of 16; returns
 * NULL on any violation), then replay all copies with one kernel launch.
 * max_bytes is the largest descriptor's byte count (grid sizing). Built for the
 * spec-frontier snapshot/restore paths (~126 tiny per-layer copies per step). */
/** @param dst   n destination tensors
 *  @param src   n source tensors
 *  @param bytes n byte counts, each a nonzero multiple of 16 that fits both tensors
 *  @param n     copies
 *  @return a device descriptor table (free with pulsar_gpu_batched_copy_free), or NULL */
void *pulsar_gpu_batched_copy_prepare(pulsar_gpu_tensor **dst, pulsar_gpu_tensor **src,
                                   const uint64_t *bytes, uint32_t n);
/** @brief Replay the first n_descs copies of a prepared table in one launch on the per-thread stream.
 *  @param handle    from pulsar_gpu_batched_copy_prepare
 *  @param n_descs   copies to run (<= the prepared n)
 *  @param max_bytes the largest copy's byte count (grid sizing)
 *  @return 1 on success, 0 on a NULL handle, n_descs 0 or launch failure */
int pulsar_gpu_batched_copy_run(void *handle, uint32_t n_descs, uint64_t max_bytes);
/** @brief Free a descriptor table.
 *  @param handle from pulsar_gpu_batched_copy_prepare, or NULL */
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
/** @return always 1 */
int pulsar_gpu_begin_commands(void);
/** @brief Drain the per-thread stream, then read the TP error word and the routed experts' non-finite and
 *  route-bounds flags (each set flag fails the step by name).
 *  @return 1 when the stream drained clean, 0 on a CUDA error or a set flag */
int pulsar_gpu_end_commands(void);
/** @brief cudaDeviceSynchronize, consuming the routed-expert flags as pulsar_gpu_end_commands does (the failure and
 *  cancel paths drain here).
 *  @return 1 on success, 0 on a CUDA error or a set flag */
int pulsar_gpu_synchronize(void);
/** L302: the host<->device synchronisations the process has paid, by kind (monotonic; every blocking copy and every
 *  drain counts once) -- what the sync-budget gate (tests/sync_budget_gate.cpp) holds a round of each lane to. */
typedef struct pulsar_gpu_sync_stats {
    uint64_t stream_syncs;   ///< pulsar_gpu_end_commands: the stream drained
    uint64_t device_syncs;   ///< pulsar_gpu_synchronize
    uint64_t d2h;            ///< blocking device -> host copies (tensor reads, the step-end flag reads)
    /** Bytes moved by the blocking device -> host copies counted in d2h. */
    uint64_t d2h_bytes;
    uint64_t h2d;            ///< blocking host -> device copies (tensor writes, flag clears)
} pulsar_gpu_sync_stats;
/** @brief Snapshot the process's synchronisation counters.
 *  @param out receives them */
void pulsar_gpu_sync_stats_get(pulsar_gpu_sync_stats *out);
/** Record a completion marker on the per-thread stream (L142).  Returns a
 *  handle, or -1 if the runtime could not record one; pulsar_gpu_marker_done
 *  reports a -1 handle as complete, so a poller never stalls on it. */
/** @return a marker handle, or -1 */
int pulsar_gpu_marker_record(void);
/** 1 when every op enqueued before the marker has completed (or the handle is
 *  -1 / invalid), 0 while the GPU is still working towards it.  Never blocks. */
/** @param marker a handle from pulsar_gpu_marker_record
 *  @return 1 when complete (or invalid), 0 while pending */
int pulsar_gpu_marker_done(int marker);

/** L119 / L306 segment graphs: the CUDA-graph cache of a step's round-invariant stretch, keyed by everything host-side
 *  its launches depend on (the contract, the two kinds of key and the invalidation: pulsar_cuda_runtime.cu).
 *  seg_enter: 2 = a captured graph replayed (skip the body), 1 = capture armed (run the body, then seg_exit), 0 = run
 *  the body eagerly (a warm use, key 0, the cache off, or nested in a capture), -1 = REFUSED (this key's capture or
 *  replay failed: the step fails -- L306, no eager fallback).  seg_exit: 1 = the captured work ran, 0 = nothing ran
 *  (the body refused, or the capture failed): the step refuses.  Worker thread only. */
/** @param key the stretch's host-side key; 0 = run eagerly, uncached
 *  @return 2 replayed, 1 capture armed, 0 run eagerly, -1 refused (see above) */
int pulsar_gpu_seg_enter(uint64_t key);
/** @brief Close a capture armed by pulsar_gpu_seg_enter: instantiate, launch and cache it.
 *  @param key     the key passed to seg_enter
 *  @param body_ok nonzero when the body enqueued its work without refusing
 *  @return 1 when the captured work ran, 0 when nothing ran (the step refuses) */
int pulsar_gpu_seg_exit(uint64_t key, int body_ok);
/** @brief Drop every cached segment graph (they bake device pointers); deferred to seg_exit mid-capture. */
void pulsar_gpu_seg_reset(void);   /* REQUIRED on gpu-graph teardown: cached
                                    * execs bake that graph's device pointers */
/** @brief Note a scratch free: every cached graph is dropped (deferred to seg_exit mid-capture); catalog keys
 *  recapture on use. */
void pulsar_gpu_seg_note_device_free(void); /* REQUIRED at every grow-realloc
                                    * scratch free: baked pointers go stale */
/** Process-lifetime segment counters: graphs replayed and captures begun (L292: a gate proves it ran the replay
 *  lane, or the eager one, by their movement across its steps). */
/** @param replays  receives graphs replayed
 *  @param captures receives captures begun */
void pulsar_gpu_seg_counts(uint64_t *replays, uint64_t *captures);
/** L306: every segment counter (process lifetime; graphs and graph_bytes are what is held now). */
typedef struct pulsar_gpu_seg_stats {
    uint64_t replays;         ///< graphs replayed
    uint64_t captures;        ///< captures begun
    uint64_t live_captures;   ///< of them, outside the warmup's window: a live step paid the capture
    uint64_t misses;          ///< uses of a graphed key that did not replay (its warm uses and its capture)
    uint64_t catalog_keys;    ///< shapes the warmup declared (pulsar_gpu_seg_catalog_open .. _close)
    uint64_t graphs;          ///< instantiated graphs held
    uint64_t graph_bytes;     ///< the device memory their instantiation took (cudaMemGetInfo deltas)
    uint64_t evictions;       ///< live graphs dropped for room (the table never runs a shape eager when full)
    uint64_t regrows;         ///< scratch regrows outside the warmup that dropped every graph
    uint64_t dead;            ///< keys whose capture or replay failed (each refuses from then on)
    uint64_t refusals;        ///< uses refused
} pulsar_gpu_seg_stats;
/** @brief Snapshot every segment counter.
 *  @param out receives them */
void pulsar_gpu_seg_stats_get(pulsar_gpu_seg_stats *out);
/** L306: the startup warmup's declaration window.  Every key a step uses between open and close is a CATALOG key:
 *  pinned, its first use eager (its scratch warms), its second captured -- so the shapes the warmup walks are
 *  captured before the listener opens.  Close announces what was declared, captured and what it cost, and returns
 *  how many declared shapes FAILED their capture or refused under it (the server refuses to start on any). */
/** @brief Open the warmup's declaration window (see above). */
void pulsar_gpu_seg_catalog_open(void);
/** @return declared shapes whose capture failed or refused (0 = the server may start) */
uint64_t pulsar_gpu_seg_catalog_close(void);
/** L296 perf, instruments only: 0 runs every segment eager (the cache dropped, nothing captured or replayed) -- the
 *  eager reference a gate compares the graphed lane against (nemotron-fwd-gate X7); 1 restores.  Returns the previous
 *  setting.  The engine never calls it. */
/** @param on 0 = every segment eager, 1 = the cache on
 *  @return the previous setting */
int pulsar_gpu_seg_set_enabled(int on);

/** L307 scratch poisoning, a diagnostic (gates; PULSAR_SCRATCH_POISON=1 arms it at start): armed, every step's entry
 *  fills the calling thread's CUDA scratch slots and the step's own disposable regions with a cycling byte pattern
 *  (0xFF, 0x00, 0x7F) before it uploads anything, so a step that reads scratch it did not write this step is caught
 *  by an output that differs from the unarmed run's (scratch-poison-gate).  _set returns the previous setting; _step
 *  is the step-entry hook (1 = done or unarmed, 0 = refused: called inside a capture, or the fill failed); _counts =
 *  steps poisoned and bytes filled. */
/** @param on nonzero arms poisoning
 *  @return the previous setting */
int pulsar_gpu_scratch_poison_set(int on);
/** @brief The step-entry poison hook (see above).
 *  @param regions the step's own disposable regions
 *  @param n       regions
 *  @return 1 = done or unarmed, 0 = refused */
int pulsar_gpu_scratch_poison_step(pulsar_gpu_tensor *const *regions, uint32_t n);
/** @param steps receives steps poisoned
 *  @param bytes receives bytes filled */
void pulsar_gpu_scratch_poison_counts(uint64_t *steps, uint64_t *bytes);

/** @brief Make the model mapping device-readable: an integrated GPU host-registers it (read-only); a discrete card
 *  stages spans into device memory on use, through the fd of pulsar_gpu_set_model_fd_for_map.
 *  @param model_map  the host mapping of the model file
 *  @param model_size its size in bytes
 *  @return 1 on success, 0 when it cannot be served (said) */
int pulsar_gpu_set_model_map(const void *model_map, uint64_t model_size);
/** @brief Register the file descriptor behind a mapping, for staged span reads (an O_DIRECT twin is opened when the
 *  filesystem allows).  The fd stays the caller's.
 *  @param fd        the open model file, or -1 to clear
 *  @param model_map the mapping it backs (the key)
 *  @return 1 on success, 0 when the mapping table is full */
int pulsar_gpu_set_model_fd_for_map(int fd, const void *model_map);
/** @brief Same as pulsar_gpu_set_model_map; map_offset, map_size and max_tensor_bytes are ignored.
 *  @param model_map        the host mapping
 *  @param model_size       its size in bytes
 *  @param map_offset       unused
 *  @param map_size         unused
 *  @param max_tensor_bytes unused
 *  @return as pulsar_gpu_set_model_map */
int pulsar_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
/** @brief Make one span of the model device-resident now (startup), instead of on first use.
 *  @param model_map  the model mapping (NULL or bytes 0: nothing to do, 1)
 *  @param model_size the mapping's size in bytes
 *  @param offset     the span's byte offset
 *  @param bytes      the span's size in bytes
 *  @param label      a name for diagnostics, or NULL
 *  @return 1 when the span is device-ready, 0 on a bad range or failure */
int pulsar_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
/** @brief Device-cache a span of a second GGUF (the --expert-overlay donor): pread from `fd` into device memory,
 *  registered under `host_base_key` so lookups resolve it like a model span.  Startup only, synchronous.
 *  @param host_base_key the donor's host mapping (the lookup key)
 *  @param fd            the donor file (read, not kept)
 *  @param offset        the span's byte offset in the file
 *  @param bytes         the span's size in bytes
 *  @param label         a name for diagnostics, or NULL
 *  @return 1 on success, 0 on failure or when it would pass the weight-cache limit (said) */
int pulsar_gpu_cache_external_range(const void *host_base_key, int fd, uint64_t offset, uint64_t bytes, const char *label);
/** @brief Whether the KV cache should be managed memory: always at >= 8 GiB, else when an >= 8 GiB context would
 *  leave free memory under the reserve (a quarter of total, clamped to 8..40 GiB).
 *  @param kv_cache_bytes the KV cache's size in bytes
 *  @param context_bytes  the whole context's device footprint in bytes
 *  @return 1 = allocate it managed, 0 = device memory */
int pulsar_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
/** @brief Print the device's free / total memory to stderr.
 *  @param label prefix for the line, or NULL */
void pulsar_gpu_print_memory_report(const char *label);
/** cudaMemGetInfo passthrough (0/0 on failure) for diagnostics/samplers. */
/** @param free_out  receives free device bytes
 *  @param total_out receives total device bytes */
void pulsar_gpu_mem_info(uint64_t *free_out, uint64_t *total_out);

/** =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

/** Pulsar's tensor layouts -- the ONE storage vocabulary, and Pulsar's OWN
 * numbers.
 *
 * A payload's storage is declared BY NAME in the container ("native" plus its
 * dtype, "mxfp8_lt", "iq2_xxs_mmq_k", "cutlass_mxfp4", "fp8_e4m3_soa_k",
 * "exl3m_k2" / "exl3m_k2h" / "exl3m_k3" / "exl3m_k4" / "exl3m_k5", "nvfp4") and
 * resolved to one of these ids exactly once, at load (st_layout_type).  The
 * numbers used to be ggml's, which meant the engine's storage vocabulary came
 * from a project it no longer speaks to -- and types it can only REFUSE (q8_0,
 * q2_k, iq2_xxs, f16, fp4_e2m1, iq2_xxs_soa) sat in the same id space, so
 * "supported" was an accident of arithmetic on a dead numbering.
 *
 * A tag is spelled once and the CUDA MoE dispatch chooses a kernel -- and an
 * arithmetic -- by these numbers (pulsar_cuda_moe.cu). */
enum {
    PULSAR_TENSOR_F32 = 0,
    PULSAR_TENSOR_I32 = 1,
    PULSAR_TENSOR_BF16 = 2,
    /* The PRE-STORED MXFP8: E4M3 weights and E8M0 scales in the exact
     * device-side layout the runtime otherwise builds at first use --
     * de-interleaved [in,out] col-major E4M3 followed by the
     * mx_sfoff()-swizzled E8M0 scale.  The FP8 matmul points cuBLASLt straight
     * at the mapping instead of materialising a second resident copy.  This is
     * the ONLY MXFP8 storage: the legacy interleaved form is refused by name at
     * load, so there is no plain-MXFP8 id to carry. */
    PULSAR_TENSOR_MXFP8_LT = 3,
    /* IQ2_XXS routed experts, k-major: (expert, k, code-word, row) order, so the
     * D2R GEMM reads a warp's 16 rows as 128 contiguous bytes and needs no
     * shared transpose. */
    PULSAR_TENSOR_IQ2_XXS_MMQ_K = 4,
    /* MXFP4 routed experts in the CUTLASS block-scaled grouped-GEMM layout:
     * expert-major ColumnMajor E2M1 data blob followed by a swizzled E8M0
     * scale-factor blob, per expert.  NOT a uniform per-element byte rate (see
     * cutlass_mxfp4_expert_layout()). */
    PULSAR_TENSOR_CUTLASS_MXFP4 = 5,
    /* MXFP8 SoA, k-major: E4M3 + E8M0-per-32 content split into two planes --
     * E8M0 scales [rows][cols/32], then E4M3 payload [rows][cols] -- so a
     * lane's payload load is one aligned 4-byte word where an interleaved
     * 33-byte block forced five byte loads.  The drafter's markov_w2 is the
     * consumer. */
    PULSAR_TENSOR_FP8_E4M3_SOA_K = 6,
    /* EXL3 trellis-coded routed experts (L245), the mul1 codebook pinned by
     * the name.  Per expert one self-contained slice: the tile stream verbatim
     * in exllamav3's (kt, nt, word) order, then the fp16 scale vectors
     * suh[in] | svh[out]; stride, the TP spans and the two-plane table all
     * follow from (dims, rate) -- exl3_expert_layout() in exl3_trellis.h is
     * the one byte model.  One id per rate because the rate is not
     * recoverable from a stack's dims. */
    PULSAR_TENSOR_EXL3M_K2 = 7,          /* K = 2   */
    PULSAR_TENSOR_EXL3M_K2H = 8,         /* K = 2.5 */
    PULSAR_TENSOR_EXL3M_K3 = 9,          /* K = 3   */
    /* L251: Qwen3.8-Flash-Next's rates -- routed experts (fused gate_up + down)
     * at K = 4, every EXL3-able dense Linear at K = 5 (the U-e4-d5 recipe).
     * Same byte model; a dense Linear is the one-slice case. */
    PULSAR_TENSOR_EXL3M_K4 = 10,         /* K = 4   */
    PULSAR_TENSOR_EXL3M_K5 = 11,         /* K = 5   */
    /* L266: turboderp's Qwen3.8-Flash-Next packs -- 4.05 bpw carries K = 6
     * dense, shared and head tensors; 6.05 K = 6 experts and K = 8 dense and
     * shared.  Same byte model. */
    PULSAR_TENSOR_EXL3M_K6 = 12,         /* K = 6   */
    PULSAR_TENSOR_EXL3M_K8 = 13,         /* K = 8   */
    /* L298: NVFP4 W4A16 (ModelOpt W4A16_NVFP4) -- one [in, out] Linear as e2m1 codes, the 128x4-tiled E4M3
     * scale plane (one per 16 inputs) and the f32 global scale; pulsar_nvfp4_layout() in
     * src/cuda/pulsar_cuda_nvfp4.h is the one byte model. */
    PULSAR_TENSOR_NVFP4 = 14,
    /* L321: IEEE f16, the Turing lane's dense format (no bf16 tensor cores below sm_80): a Linear [in, out] the f16
     * rows arm multiplies on the fp16 tensor cores (pulsar_cuda_f16_rows.h), or a table read as f16. */
    PULSAR_TENSOR_F16 = 15,
    PULSAR_TENSOR_TYPE_COUNT
};

/** Storage of the drafter's markov_w2 table (one spelling; the engine derives
 * it from the tensor's GGUF type in pulsar_markov_w2_fmt(), the CUDA launchers
 * dispatch a kernel arm on it).  Every arm is k-major (L213). */
enum {
    PULSAR_MARKOV_W2_F32   = 0,   /* GGUF type 0 */
    PULSAR_MARKOV_W2_BF16  = 1,   /* GGUF type 30 */
    PULSAR_MARKOV_W2_MXFP8 = 2,   /* GGUF type 46: E8M0 scale plane [E][V/32], then E4M3 payload plane [E][V] -- the shipped table */
};

#if defined(__CUDACC__)
/** `__host__ __device__` under nvcc, empty elsewhere: for helpers shared by host code and kernels. */
#define PULSAR_GPU_HD __host__ __device__   /* the helper is called from kernels too */
#else
/** Empty outside nvcc (the host-only definition). */
#define PULSAR_GPU_HD

#endif

/** @brief Seed the HC residual: each token's bf16 embedding row, copied into all n_hc streams.
 *  @param out_hc        [n_tokens][n_hc][n_embd] HC residual (PULSAR_HC_ELT_SIZE elements)
 *  @param tokens        int32 [n_tokens] token ids
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset byte offset of the bf16 [n_vocab][n_embd] embedding table
 *  @param n_vocab       table rows
 *  @param n_tokens      tokens
 *  @param n_embd        embedding width
 *  @param n_hc          HC streams
 *  @return 1 on success, 0 on a bad size or launch failure */
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

/** @brief Score one query row against every compressed indexer row (non-causal), on the MXFP4 tier.
 *  @param scores     f32 [n_comp] out
 *  @param q          the packed MXFP4 query, n_head rows of PULSAR_MXKV_FP4_ROWBYTES(head_dim)
 *  @param weights    f32 [n_head] per-head weights
 *  @param index_comp packed MXFP4 compressed indexer rows, [n_comp]
 *  @param n_comp     compressed rows
 *  @param n_head     indexer heads (a count the tier tiles)
 *  @param head_dim   indexer head width (128)
 *  @param scale      score scale
 *  @return 1 on success, 0 on refusal (said) or failure */
int pulsar_gpu_indexer_score_one_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);


/** Score a causal prefill/decode span through the block-scaled MXFP4 tier:
 * row t sees compressed rows < (pos0 + t + 1) / ratio, the rest -INF.  Single
 * cache operand.  The engine scores a BANKED span as same-bank consecutive-
 * position runs through pulsar_gpu_indexer_scores_decode_run_tensor below
 * (and refuses a span whose runs are not consecutive); the descriptor
 * (positions/seq_id/bank-table) arm this entry once carried, and the generic
 * per-(comp,row) kernel behind it, served no engine shape and are gone
 * (L176).  n_head must be 64 and head_dim 128 -- the tier's shape; anything
 * else is refused by name. */
/** @param scores     f32 [n_tokens][n_comp] out (-inf past each row's reach)
 *  @param q          packed MXFP4 queries, [n_tokens][n_head]
 *  @param weights    f32 [n_tokens][n_head] per-head weights
 *  @param index_comp packed MXFP4 compressed rows, [n_comp]
 *  @param n_comp     compressed rows (the scores stride)
 *  @param n_tokens   query rows
 *  @param pos0       position of row 0
 *  @param n_head     indexer heads
 *  @param head_dim   indexer head width (128)
 *  @param ratio      compression ratio (tokens per compressed row); nonzero
 *  @param scale      score scale
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        float                   scale);

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
/** @param scores          f32 [run_n][n_comp] out
 *  @param q               packed MXFP4 queries, [run_n][n_head]
 *  @param weights         f32 [run_n][n_head]
 *  @param bank_index_comp the bank's packed MXFP4 compressed rows
 *  @param n_comp          the scores stride / scan bound (not the bank's frontier)
 *  @param run_n           rows in the run
 *  @param run_pos0        position of the run's first row
 *  @param n_head          indexer heads
 *  @param head_dim        indexer head width (128)
 *  @param ratio           compression ratio; nonzero
 *  @param scale           score scale
 *  @return 1 on success, 0 on refusal (said) or failure */
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

/** Does the backend's PREFILL attention read MAIN comp rows natively?  When
 * it does, the engine hands it the packed cache directly and skips
 * dequantising into the f32 shadow -- PULSAR_MAINKV_ROWBYTES (288 B/row at
 * head_dim 512) instead of 2048, on the rows that dominate the tile, plus one
 * whole pass removed.  Bit-exact either
 * way: packed rows decode to exactly the values the f32 cache would hold.
 * Backend-neutral question; the answer is a property of the backend's kernels,
 * not of any particular one. */
/** @return 1 when it does (today: whenever the fp16 tier is on), else 0 */
int pulsar_gpu_attention_prefill_reads_packed_comp(void);

/* fp16 tensor-core attention (m16n8k16, f32 accumulate).  Raw pointers: a leaf
 * kernel behind the attention launchers, which do the tensor-level checking.
 * Returns 0 on refusal or failure.  Requires head_dim == 512 (AF16_DIM) and
 * n_head a multiple of 32 (AF16_HPB, the heads one block owns; every launcher
 * refuses n_head % AF16_HPB).  Operand format chosen by measurement, not preference --
 * the operand study is in src/cuda/pulsar_cuda_attn_f16.cu's header, and
 * docs/engine-perf-map.md.  (It used to cite tests/attn_precision_fidelity.cc,
 * deleted in a71e346 -- L106 K8 -- when its dump-format producer left the
 * tree.) */
/** Q-prep descriptor for the fused rope Q load (L037 lever 3). Non-NULL
 * means `q` holds RAW projections: the fp16 kernel applies the tail rope
 * itself at its Q fragment build (the same rotation pulsar_gpu_rope_tail_tensor
 * applies, then the reference's bf16 rounding).  NULL means q is already
 * roped -- the Qcur dump arm.  L218: V4.1 has no per-head RMS on Q, so the
 * prologue is rope only.  Carries only the launch-invariant shape;
 * pos0/positions stay the wrapper params they already are. */
typedef struct {
    /** Per-head RMS epsilon for the fused prologue, or 0 for a profile whose
     * reference has no per-head Q norm.  0731's Attention.forward normalises Q
     * per head before the tail rope (`q *= rsqrt(q.square().mean(-1) + eps)`);
     * V4.1's does not.  Nonzero both arms the scale and selects the reference's
     * Q path (fp32 rotation, no extra narrowing of the rotated pair). */
    float eps;
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
/** @return 1 when the device has the fp16 MMA tier, else 0 */
int pulsar_gpu_attn_f16_tier_on(void);

/** Opaque packed-row carriers (L092).  The packed caches (WINDOW / MAIN KV
 * rows, see the format block below; MXKV-FP4 indexer rows) used to
 * travel as `const float *` -- a carrier type
 * that described nothing, so one direct raw_kv[i] anywhere was defect ten and
 * compiled clean.  Deliberately INCOMPLETE types: indexing or arithmetic is a
 * compile error, so every read goes through the format's accessor (or an
 * explicit byte-level cast at a row-granular copy).  Pass tensor->ptr. */
typedef struct pulsar_winkv_row_s pulsar_winkv_row_t;    /* WINDOW rows (the ring) */
/** Opaque carrier of MAIN (compressed) KV rows; see above. */
typedef struct pulsar_mainkv_row_s pulsar_mainkv_row_t;  /* MAIN rows (a source's pool) */
/** Opaque carrier of packed MXFP4 indexer rows (PULSAR_MXKV_FP4_ROWBYTES each); see above. */
typedef struct pulsar_mxkv_pack_s pulsar_mxkv_pack_t;

/** @brief Dense-batch prefill attention on the fp16 tier: each query folds its causal raw window and the visible
 *  compressed prefix, optionally emitting the grouped E4M3 operand of the output "a" GEMM.
 *  @param heads     stored attention output, [n_tokens][n_head][head_dim] (see below)
 *  @param sinks     f32 [n_head] per-head attention-sink logits
 *  @param q         stored Q, [n_tokens][n_head][head_dim]; raw projections when q_prep is set
 *  @param raw_kv    the batch's WINDOW rows, [n_tokens]
 *  @param comp_kv   MAIN rows, [n_comp]; may be NULL when n_comp is 0
 *  @param n_tokens  query rows (nonzero)
 *  @param n_comp    compressed rows
 *  @param window    raw rows a query sees (its sliding window)
 *  @param ratio     compression ratio (tokens per compressed row)
 *  @param n_head    heads, a multiple of 32
 *  @param head_dim  512
 *  @param gact_data   grouped E4M3 slot for the attention-output "a" GEMM (pulsar_gpu_mxfp8_gact_slot), or NULL = no emission
 *  @param gact_scale  its swizzled E8M0 scale slab
 *  @param gact_kbp    the slot's scale pitch (pulsar_gpu_mx_kbp)
 *  @param gact_slab   bytes of one group's scale slab
 *  @param n_groups    output groups; n_head % n_groups == 0
 *  @param n_nope      head dims [0, n_nope) this epilogue encodes (a multiple of 32; the rope tail owns the rest)
 *  @param gact_tok0   this launch's first row in the slot (batch-absolute)
 *  @param gact_ntok   the slot's row count (the stride the GEMM reads with)
 *  @param positions see below
 *  @param q_prep    the fused rope descriptor, or NULL when q is already roped
 *  @param vis_left  see below
 *  @param vis_right see below
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        const pulsar_winkv_row_t *raw_kv, const pulsar_mainkv_row_t *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp,
        uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope,
        uint32_t gact_tok0, uint32_t gact_ntok,
        /* positions: int32 [n_tokens] DEVICE array or NULL.  Dense mode uses it
         * for the fused Q rope only (positions[t] instead of t); the row plan is
         * the batch's own causal window either way. */
        const int *positions,
        const pulsar_gpu_q_prep *q_prep,
        /* Image-span visibility: int32 [n_tokens] DEVICE arrays of the
         * reference's get_image_visible() counts, or NULL/NULL (the text-only
         * path, bit-identical to the pre-L216 kernel).  Non-NULL makes an image
         * query see [q - vis_left[q], q + vis_right[q]] -- a FORWARD reach the
         * plain causal window cannot express.  A half pair or a visibility
         * launch mixed with non_causal is refused. */
        const int *vis_left,
        const int *vis_right);

/** @brief pulsar_gpu_attention_f16_prefill_mx without the MX emission (the kernel test's narrow entry).
 *  @param heads     stored attention output
 *  @param sinks     f32 [n_head] attention-sink logits
 *  @param q         stored Q
 *  @param raw_kv    WINDOW rows, [n_tokens]
 *  @param comp_kv   MAIN rows, [n_comp]
 *  @param n_tokens  query rows
 *  @param n_comp    compressed rows
 *  @param window    raw rows a query sees
 *  @param ratio     compression ratio
 *  @param n_head    heads, a multiple of 32
 *  @param head_dim  512
 *  @param q_prep    fused rope descriptor, or NULL
 *  @param vis_left  image-span visibility, or NULL
 *  @param vis_right image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal or failure */
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
        const pulsar_winkv_row_t *raw_kv,
        const pulsar_mainkv_row_t *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep,
        const int               *vis_left,
        const int               *vis_right);

/** fp16 tensor-core attention, INDEXED: raw rows come from a ring buffer and
 * compressed rows are a top-k selection (topk != NULL) or the visible prefix
 * (topk == NULL, the decode-batch sweep).  This is the ONE decode attention
 * kernel (L166): every row count, every context length, causal and
 * non-causal.  Banked descriptors are served: positions and seq_id are checked
 * as a PAIR (one without the other is refused); comp_bank_ptrs is an optional
 * per-bank base table -- NULL under descriptors means the scalar comp base +
 * seq_id*comp_cap, bit-identical to a contiguous pool.  Comp rows are
 * MAIN rows, always -- bank
 * isolation gated by tests/attn_f16_banked_test.cu.  Returns 0 on refusal or
 * failure. */
/** @param heads          stored attention output, [n_tokens][n_head][head_dim]
 *  @param sinks          f32 [n_head] attention-sink logits
 *  @param q              stored Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv         the WINDOW ring, raw_cap rows
 *  @param comp_kv        MAIN rows (the pool, or bank 0's base); may be NULL when n_comp is 0
 *  @param topk           int32 [n_tokens][top_k] selected compressed rows, or NULL = the visible prefix
 *  @param n_tokens       query rows (nonzero)
 *  @param pos0           position of row 0 (without descriptors)
 *  @param n_raw          ring rows holding data
 *  @param raw_cap        ring capacity, rows (nonzero)
 *  @param raw_start      ring index of the oldest row
 *  @param n_comp         compressed rows visible / addressable
 *  @param top_k          entries per topk row (nonzero when topk is set)
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads, a multiple of 32
 *  @param head_dim       512
 *  @param positions      int32 [n_tokens] device positions (banked descriptors), or NULL; paired with seq_id
 *  @param seq_id         int32 [n_tokens] device bank ids, or NULL
 *  @param comp_bank_ptrs device table of per-bank MAIN-row bases, [n_banks], or NULL
 *  @param comp_cap       per-bank compressed-row stride (descriptors with n_comp > 0)
 *  @param n_banks        banks addressed (nonzero under descriptors)
 *  @param non_causal     see below
 *  @param q_prep         fused rope descriptor, or NULL
 *  @param vis_left       see below
 *  @param vis_right      see below
 *  @param gact_data   grouped E4M3 slot for the attention-output "a" GEMM (pulsar_gpu_mxfp8_gact_slot), or NULL = no emission
 *  @param gact_scale  its swizzled E8M0 scale slab
 *  @param gact_kbp    the slot's scale pitch (pulsar_gpu_mx_kbp)
 *  @param gact_slab   bytes of one group's scale slab
 *  @param n_groups    output groups; n_head % n_groups == 0
 *  @param n_nope      head dims [0, n_nope) this epilogue encodes (a multiple of 32; the rope tail owns the rest)
 *  @param gact_tok0   this launch's first row in the slot (batch-absolute)
 *  @param gact_ntok   the slot's row count (the stride the GEMM reads with)
 *  @return 1 on success, 0 on refusal (said) or failure */
int pulsar_gpu_attention_f16_indexed(
        /* heads: stored attention output, PULSAR_HEADS_ELT_SIZE bytes/element;
         * opaque here for the same reason as q below (L033). Pass tensor->ptr. */
        void                    *heads,
        const float             *sinks,
        /* q: stored Q, PULSAR_Q_ELT_SIZE bytes/element; opaque here so this header
         * need not include cuda_fp16.h.  Pass tensor->ptr. */
        const void              *q,
        const pulsar_winkv_row_t *raw_kv,
        const pulsar_mainkv_row_t *comp_kv,
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
        const pulsar_gpu_q_prep *q_prep,
        /* Image-span visibility: see pulsar_gpu_attention_f16_prefill_mx.  The
         * arrays are indexed by this launch's own token axis (like positions),
         * so a caller launching a sub-span offsets them by its first row. */
        const int               *vis_left,
        const int               *vis_right,
        /* Grouped E4M3 for the attn-output 'a' GEMM (pulsar_gpu_mxfp8_gact_slot), or
         * NULL: the epilogue encodes the nope blocks of every row it stores --
         * the rows past the launch's split-K decode rows -- at batch row
         * gact_tok0 + t of a gact_ntok-row slab (L260). */
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab,
        uint32_t n_groups, uint32_t n_nope, uint32_t gact_tok0, uint32_t gact_ntok);

/** Block-scaled indexer scorer (SM120 mxf8f6f4 MMA over the stored MXFP4 rows).
 * Raw pointers, not tensors: it is a leaf kernel behind indexer_scores_launch,
 * which does the tensor-level bounds checking.  Returns 0 on refusal or
 * failure; the caller checks the shape conditions itself so a 0 is always a
 * real failure.  Requires n_head == 64 and head_dim == 128. */
/** @param scores   f32 [n_tokens][n_comp] out
 *  @param q        packed MXFP4 queries, [n_tokens][n_head]
 *  @param weights  f32 [n_tokens][n_head] per-head weights
 *  @param comp     packed MXFP4 compressed rows, [n_comp]
 *  @param n_comp   compressed rows
 *  @param n_tokens query rows
 *  @param pos0     position of row 0
 *  @param n_head   indexer heads
 *  @param head_dim 128
 *  @param ratio    compression ratio (read when causal)
 *  @param scale    score scale
 *  @param causal   nonzero: row t sees rows < (pos0 + t + 1) / ratio, the rest -inf
 *  @return 1 on success, 0 on refusal or failure */
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


/** @brief Per row, the indices of the top_k highest indexer scores (the compressed rows attention reads).
 *  @param selected uint32 [n_tokens][top_k] out
 *  @param scores   f32 [n_tokens][n_comp]
 *  @param n_comp   scores per row
 *  @param n_tokens rows
 *  @param top_k    selections per row, 1 .. n_comp
 *  @return 1 on success, 0 on refusal (said) or failure */
int pulsar_gpu_indexer_topk_tensor(
        pulsar_gpu_tensor       *selected,
        const pulsar_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/** CSA2 candidate pool (L218): level one of the two-level top-k, the
 * reference's select_candidate_blocks.  From the candidate source's indexer
 * scores ([n_rows][n_comp] f32, -inf beyond each row's reach) keep the
 * `topk_blocks` highest-scoring blocks of `block_size` compressed positions
 * per row -- the row's newest block pinned, unreachable blocks dropped -- as
 * a bitmask of `mask_words` u32 per row (bit b = block b).  `bscore_scratch`
 * holds [n_rows][n_blocks] f32 during the call.  vis per row is
 * (pos + 1) / ratio with pos = pos0 + row or positions[row]. */
/** @param n_comp_cap the most compressed rows a row can reach
 *  @param block_size compressed positions per block
 *  @return u32 words of one row's block mask (0 when block_size is 0) */
uint32_t pulsar_gpu_candidate_mask_words(uint32_t n_comp_cap, uint32_t block_size);
/** @param mask           u32 [n_rows][mask_words] out
 *  @param bscore_scratch f32 [n_rows][n_blocks] scratch
 *  @param scores         f32 [n_rows][n_comp] indexer scores
 *  @param n_comp         scores per row
 *  @param n_rows         rows
 *  @param mask_words     pulsar_gpu_candidate_mask_words
 *  @param block_size     compressed positions per block
 *  @param topk_blocks    blocks kept per row
 *  @param pos0           position of row 0 (when positions is NULL)
 *  @param ratio          compression ratio
 *  @param positions      int32 [n_rows] positions, or NULL
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_gpu_candidate_blocks_tensor(pulsar_gpu_tensor *mask, pulsar_gpu_tensor *bscore_scratch,
                                       const pulsar_gpu_tensor *scores,
                                       uint32_t n_comp, uint32_t n_rows, uint32_t mask_words,
                                       uint32_t block_size, uint32_t topk_blocks,
                                       uint32_t pos0, uint32_t ratio, const pulsar_gpu_tensor *positions);
/** Level two: -inf into every score whose block is outside the row's mask,
 * before that layer's top-k. */
/** @param scores     f32 [n_rows][n_comp], masked in place
 *  @param mask       u32 [n_rows][mask_words] from pulsar_gpu_candidate_blocks_tensor
 *  @param n_comp     scores per row
 *  @param n_rows     rows
 *  @param mask_words words per mask row
 *  @param block_size compressed positions per block
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_gpu_candidate_mask_scores_tensor(pulsar_gpu_tensor *scores, const pulsar_gpu_tensor *mask,
                                            uint32_t n_comp, uint32_t n_rows, uint32_t mask_words, uint32_t block_size);

/** GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
/** @param out_idx int32 [1] out
 *  @param logits  f32 [n_vocab]
 *  @param n_vocab logits (nonzero)
 *  @return 1 on success, 0 on a bad size or launch failure */
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
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`).
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
/** @param model_map     the mapping the weight lives in
 *  @param weight_offset its byte offset (the weight's key) */
void pulsar_gpu_register_fp8_weight(const void *model_map, uint64_t weight_offset);

/** Mark an already-fp8-registered offset as a pre-stored MXFP8_LT weight: the
 * device layout (de-interleaved E4M3 data + swizzled E8M0 scale) is already in
 * the mmap, so the matmul resolver skips the cudaMalloc+convert and points
 * cuBLASLt directly at g_model_device_base+offset. Done once at load. */
/** @param model_map     the mapping the weight lives in
 *  @param weight_offset its byte offset, already registered with pulsar_gpu_register_fp8_weight */
void pulsar_gpu_register_fp8_lt_weight(const void *model_map, uint64_t weight_offset);
/* Slice 4g (L241): register the OUTPUT-ROW slice [row_lo, row_hi) of a
 * pre-stored MXFP8_LT weight as a weight of its own, addressed by its data
 * offset `parent_offset + row_lo * in_dim` with out_dim = row_hi - row_lo.
 * row_lo must sit on a 128-row scale band.  Every arm that resolves weights by
 * offset then consumes the slice unchanged.  Returns 1, or 0 with the reason
 * printed. */
/** @brief Register an output-row slice of a pre-stored MXFP8_LT weight as a weight of its own (see above).
 *  @param model_map     the mapping
 *  @param parent_offset the full weight's data offset
 *  @param in_dim        input width (K)
 *  @param out_full      the full weight's output rows
 *  @param row_lo        first output row, on a 128-row scale band
 *  @param row_hi        one past the last output row (<= out_full)
 *  @return 1, or 0 with the reason printed */
int pulsar_gpu_register_fp8_lt_row_slice(const void *model_map, uint64_t parent_offset,
                                         uint64_t in_dim, uint64_t out_full,
                                         uint64_t row_lo, uint64_t row_hi);
/* L241 4g-2 expert tensor-parallel: build this rank's half of every expert of
 * one routed cutlass_mxfp4 stack from the host mapping (rows [lo,hi) for
 * gate/up, input columns [lo,hi) for down when k_half) and register it as a
 * model range under (key_map, key_offset); src/dst geometries are the
 * cutlass_mxfp4_expert_layout of the full and half shapes.  Returns 1, or 0
 * with the reason printed. */
/** @brief Build and register this rank's half of a routed cutlass_mxfp4 expert stack (see above).
 *  @param key_map    the mapping the half is registered under (its key)
 *  @param key_offset the key offset
 *  @param model_map  the host mapping holding the full stack
 *  @param src_offset the full stack's byte offset
 *  @param n_expert   experts in the stack
 *  @param k          the full input width
 *  @param n          the full output width
 *  @param k_half     nonzero: split the input columns (down); 0: the output rows (gate/up)
 *  @param lo         first row / column of this rank's half
 *  @param hi         one past the last
 *  @param src_stride bytes per expert in the full layout
 *  @param src_data   bytes of an expert's data blob in the full layout (its scale blob follows)
 *  @param dst_stride bytes per expert in the half layout
 *  @param dst_data   bytes of an expert's data blob in the half layout
 *  @return 1, or 0 with the reason printed */
int pulsar_gpu_register_mxfp4_expert_half(const void *key_map, uint64_t key_offset,
                                          const void *model_map, uint64_t src_offset,
                                          uint32_t n_expert, uint64_t k, uint64_t n, int k_half,
                                          uint64_t lo, uint64_t hi,
                                          uint64_t src_stride, uint64_t src_data,
                                          uint64_t dst_stride, uint64_t dst_data);
/* L269 W1: the EXL3 twin -- this rank's half of every expert of one routed
 * EXL3 stack (in = k, out = n, rate k2): output columns [lo,hi) for gate/up,
 * input rows [lo,hi) for down when k_half.  The cut is exl3_expert_cut's (the
 * one byte model; the half of every expert is exl3_expert_layout of the half
 * shape); registered as a model range under (key_map, key_offset) like the
 * MXFP4 half.  Returns 1, or 0 with the reason printed. */
/** @brief Build and register this rank's half of a routed EXL3 expert stack (see above).
 *  @param key_map    the mapping the half is registered under
 *  @param key_offset the key offset
 *  @param model_map  the host mapping holding the full stack
 *  @param src_offset the full stack's byte offset
 *  @param n_expert   experts in the stack
 *  @param k          input width
 *  @param n          output width
 *  @param k2         the EXL3 rate code (exl3_trellis.h)
 *  @param k_half     nonzero: split the input rows (down); 0: the output columns (gate/up)
 *  @param lo         first row / column of this rank's half
 *  @param hi         one past the last
 *  @return 1, or 0 with the reason printed */
int pulsar_gpu_register_exl3_expert_half(const void *key_map, uint64_t key_offset,
                                         const void *model_map, uint64_t src_offset,
                                         uint32_t n_expert, uint64_t k, uint64_t n, int k2, int k_half,
                                         uint64_t lo, uint64_t hi);
/* L241 4g-2: register the INPUT-COLUMN half [k_lo, k_hi) of a pre-stored
 * MXFP8_LT weight (a row-parallel TP split: the rank's share of the
 * reduction).  Repacked once into device buffers the backend owns (freed with
 * the weight cache) and registered under (key_map, key_offset), in_dim =
 * k_hi - k_lo; the GEMMs then resolve it by that key like any weight, with a
 * model_size covering key_offset.  Bounds must be 128-aligned.  Returns 1, or
 * 0 with the reason printed. */
/** @brief Register the input-column half [k_lo, k_hi) of a pre-stored MXFP8_LT weight (see above).
 *  @param model_map     the mapping holding the full weight
 *  @param parent_offset the full weight's data offset
 *  @param in_full       the full input width
 *  @param out_dim       output width
 *  @param k_lo          first input column (128-aligned)
 *  @param k_hi          one past the last (128-aligned)
 *  @param key_map       the mapping the slice is registered under
 *  @param key_offset    the key offset
 *  @return 1, or 0 with the reason printed */
int pulsar_gpu_register_fp8_lt_kslice(const void *model_map, uint64_t parent_offset,
                                      uint64_t in_full, uint64_t out_dim,
                                      uint64_t k_lo, uint64_t k_hi,
                                      const void *key_map, uint64_t key_offset);
/* L242: a pre-stored MXFP8_LT weight already resident in device buffers (data
 * plane, then the swizzled scale plane as separate tensors) enters the resolved
 * cache under (map_key, offset); the GEMMs then find it by offset.  For gates
 * that carry a weight outside any model map. @return 1 on success. */
/** @brief Register a device-resident MXFP8_LT weight (see above).
 *  @param map_key the mapping key it is found under
 *  @param offset  the key offset
 *  @param in_dim  input width (K)
 *  @param out_dim output width (N)
 *  @param data    the E4M3 data plane (borrowed; must outlive the registration)
 *  @param scale   the swizzled E8M0 scale plane (borrowed)
 *  @return 1 on success, 0 on failure */
int pulsar_gpu_register_fp8_lt_weight_resident(const void *map_key, uint64_t offset,
                                               uint64_t in_dim, uint64_t out_dim,
                                               const pulsar_gpu_tensor *data,
                                               const pulsar_gpu_tensor *scale);
/* The producer step for an f32 activation that an MXFP8 GEMM will read: emit
 * its E4M3 encoding into the activation cache slot and arm it (the pure
 * per-block quantiser, bit-identical however it is invoked).  This is what the
 * attention-out stage does for `low` before stage 'b'; a gate that fabricates
 * an input does the same rather than asking the GEMM for a quantise pass it no
 * longer has (L158).  dim must be a multiple of 256.  @return 1 on success. */
/** @brief Encode an f32 activation into its MXFP8 slot and arm it (see above).
 *  @param x        f32 [n_tokens][dim] activation (the slot's key)
 *  @param n_tokens rows
 *  @param dim      row width, a multiple of 256
 *  @return 1 on success, 0 on failure */
int pulsar_gpu_mxfp8_act_emit_f32(pulsar_gpu_tensor *x, uint32_t n_tokens, uint64_t dim);

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
/** @param x      the activation just written (the slot's key)
 *  @param n_tok  its rows
 *  @param in_dim its row width */
void pulsar_gpu_mxfp8_act_cache_arm(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);
/** @brief Disarm the cache: the armed activation is dead. */
void pulsar_gpu_mxfp8_act_cache_disarm(void);


/** The widest DECODE batch the decode arms take (the small-batch expert FFN
 * GEMV and per-expert projection in pulsar_cuda_moe.cu enumerate up to it; the
 * dense GEMMs switch to cuBLASLt past pulsar_gpu_matmul_decode_exact_rows()).  It
 * bounds pulsar_gpu_matmul_set_batch_decode_rows, and the PULSAR_MSEQ_MAX
 * static_assert in the engine keeps the bank count inside it: a decode row
 * past this width would have no M-independent arm, so growing PULSAR_MSEQ_MAX
 * past it FAILS THE BUILD.  It chooses no arm: row KIND does (below). */
#define PULSAR_GPU_MNEUTRAL_ROWS_MAX 16u

/** ROW KIND.  `n` is the number of leading DECODE rows in the batch being
 * encoded -- a fact about the rows, declared by the lane that owns them, and
 * the ONE thing every dense GEMM and MoE dispatcher reads to choose its arm:
 *   - decode rows (n > 0, n >= the call's n_tok) take the decode arms: the
 *     one-row GEMV at 1; the dense nt GEMVs at 2..pulsar_gpu_matmul_decode_exact_rows()
 *     and cuBLASLt at the call's own row count above it (L260, Tyler 2026-10-01:
 *     the fastest arm per width, graded by KL rather than byte-gated); the
 *     small-batch FFN / per-expert projection at 2..PULSAR_GPU_MNEUTRAL_ROWS_MAX.
 *     A decode row's bytes never depend on its batchmates' VALUES, and within
 *     the exact range not on the batch width either;
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
/** Does any kernel read the model through its HOST mapping (a host-registered
 * whole model or range)?  When it does not, every weight byte the device reads
 * lives in device memory, and the engine may release the checkpoint's host
 * pages after load (pulsar_model_release_host_pages). */
/** @return 1 when some kernel reads host-mapped model pages, else 0 */
int pulsar_gpu_model_reads_host_pages(void);

/** @brief Declare the calling thread's leading decode rows (ROW KIND, above), capped at PULSAR_GPU_MNEUTRAL_ROWS_MAX.
 *  @param n decode rows; <= 0 = none (every row prefill)
 *  @return 1, or 0 (said) when n exceeds the cap */
int pulsar_gpu_matmul_set_batch_decode_rows(int n);
/** The same declaration with an explicit row cap: the drafter forward declares
 *  up to PULSAR_DSPARK_DRAFT_ROWS_MAX decode rows (every decode arm takes any
 *  width past the exact range); everything else stays at the M-neutral cap. */
/** @param n   decode rows; <= 0 = none
 *  @param cap the lane's row cap
 *  @return 1, or 0 (said) when n > cap */
int pulsar_gpu_matmul_set_batch_decode_rows_capped(int n, int cap);
/** @return the calling thread's declared decode rows (0 = none) */
int pulsar_gpu_matmul_batch_decode_rows(void);
/** The row-kind fact a dense call of `n_tok` rows chooses its arms by (L167): its effective decode prefix,
 *  min(declared decode rows, n_tok) -- 0 = every row prefill (tensor-core arms), n_tok = every row decode
 *  (M-independent arms), between = split there.  Two calls with the same n_tok and the same prefix take the
 *  same arms; a different prefix is different arithmetic.  Anything that caches an encoded call (a captured
 *  CUDA graph) keys on this (L292). */
/** @param n_tok the call's rows
 *  @return min(declared decode rows, n_tok) */
uint32_t pulsar_gpu_matmul_decode_prefix(uint32_t n_tok);
/** L260: the widest decode step whose rows are still BYTE-IDENTICAL to the same
 *  rows at any narrower width (and alone).  Wider decode steps take width-
 *  dependent arithmetic (Tyler 2026-10-01: the fastest arm per width), graded,
 *  not byte-gated -- the gates read this rather than restating the threshold. */
/** @return that width, rows */
int pulsar_gpu_matmul_decode_exact_rows(void);

/** @brief out = x . W^T for a bf16 weight W [out_dim][in_dim] in the model mapping (the shared bf16 core; reads the
 *  activation's producer-emitted bf16 plane).
 *  @param out           f32 [n_tok][out_dim]
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset the weight's byte offset
 *  @param in_dim        input width (K)
 *  @param out_dim       output width (N)
 *  @param x             activation rows [n_tok][in_dim]
 *  @param n_tok         rows (M)
 *  @return 1 on success, 0 on a bad shape, a missing bf16 plane or launch failure */
int pulsar_gpu_matmul_bf16_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);



/** @brief out = x . W^T for an f32-stored weight [out_dim][in_dim]: converted once to bf16 and run on the shared
 *  bf16 core (its source math is bf16).
 *  @param out           f32 [n_tok][out_dim]
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset the weight's byte offset
 *  @param in_dim        input width (K)
 *  @param out_dim       output width (N)
 *  @param x             activation rows [n_tok][in_dim]
 *  @param n_tok         rows (M)
 *  @return 1 on success, 0 on a bad shape or failure */
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
/** @param t        source tensor (f32, f16 or bf16 elements; a BYTES tensor refuses)
 *  @param elem_off first element
 *  @param out      host f32 [n_elems]
 *  @param n_elems  elements
 *  @return 1 on success, 0 on failure */
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
/** @param x      the activation (the slot's key)
 *  @param n_tok  rows
 *  @param in_dim row width
 *  @param xb_out receives the bf16 [n_tok][in_dim] device plane to write
 *  @return 1 on success, 0 when no slot or storage is available */
int pulsar_gpu_bf16_act_slot(const pulsar_gpu_tensor *x,
                             uint64_t n_tok, uint64_t in_dim, void **xb_out);
/** @brief Mark the bf16 plane for (x, n_tok, in_dim) valid (after its producer kernel succeeded).
 *  @param x      the activation
 *  @param n_tok  rows
 *  @param in_dim row width */
void pulsar_gpu_bf16_act_note(const pulsar_gpu_tensor *x,
                              uint64_t n_tok, uint64_t in_dim);
/** @brief Forget every activation plane keyed on `x` (call before freeing it).
 *  @param x the buffer */
void pulsar_gpu_act_slot_drop(const pulsar_gpu_tensor *x);

/** @brief Reserve the E4M3 slot for (x, n_tok, in_dim) and zero its scale slab, for a producer epilogue to fill.
 *  @param x         the activation (the slot's key)
 *  @param n_tok     rows
 *  @param in_dim    row width, a nonzero multiple of 32
 *  @param data_out  receives the E4M3 [n_tok][in_dim] device plane
 *  @param scale_out receives the swizzled E8M0 scale slab
 *  @param sf_pitch  receives the scale pitch (pulsar_gpu_mx_kbp(in_dim))
 *  @return 1 on success, 0 on failure (said) */
int pulsar_gpu_mxfp8_act_cache_e4m3_slot(const pulsar_gpu_tensor *x,
                                         uint64_t n_tok, uint64_t in_dim,
                                         void **data_out, void **scale_out,
                                         int *sf_pitch);

/** The MX slab geometry, host-side.  A producer that hands a kernel a RAW A8
 *  output slot (pulsar_gdn_call::out_e4m3 / out_scale / out_kbp) sizes its
 *  scratch with these instead of spelling the geometry: `in_dim` -> the blocks
 *  per row (padded to 4), and `rows` x that -> the E8M0 scale-slab bytes.
 *  One authority: both delegate to src/cuda/pulsar_cuda_mx.cuh, which an engine
 *  TU (compiled by g++) cannot include. */
/** @param in_dim row width
 *  @return MX blocks (32 values) per row, padded to 4 */
int pulsar_gpu_mx_kbp(int in_dim);
/** @param rows rows
 *  @param kbp  pulsar_gpu_mx_kbp of the row width
 *  @return bytes of the swizzled E8M0 scale slab */
size_t pulsar_gpu_mx_sf_slab_bytes(int rows, int kbp);

/** GROUPED activation slots for the attn-output "a" projection (batch_heads).
 * Reserves per-group E4M3 data plus a per-group swizzled E8M0 scale slab and
 * zeroes the scales, so the attention epilogue and rope_tail can emit the
 * encoding between them and the GEMM's quantize pass disappears.  `scale_slab`
 * returns the per-group stride the producers must index with.  Returns 0 on
 * failure; note() only after BOTH producers have run. */
/** @param heads      the attention output (the slot's key)
 *  @param n_tokens   rows
 *  @param n_groups   output groups
 *  @param group_dim  width of one group (heads per group x head_dim), a multiple of 32
 *  @param data_out   receives the E4M3 [n_groups][n_tokens][group_dim] plane
 *  @param scale_out  receives the scale slabs
 *  @param sf_pitch   receives the scale pitch
 *  @param scale_slab receives one group's scale-slab stride, bytes
 *  @return 1 on success, 0 on failure */
int pulsar_gpu_mxfp8_gact_slot(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                               uint32_t n_groups, uint64_t group_dim,
                               void **data_out, void **scale_out,
                               int *sf_pitch, uint64_t *scale_slab);
/** @brief Mark the grouped slot's encoding current (after both producers ran). */
void pulsar_gpu_mxfp8_gact_note(void);
/** @brief Disarm the grouped slot. */
void pulsar_gpu_mxfp8_gact_disarm(void);

/** L158: the drafter's concat of three target hidden states, emitted as E4M3
 * straight into the activation slot (producer-side A8; the f32 concat is never
 * written).  slot_data/slot_scale/sf_pitch come from
 * pulsar_gpu_mxfp8_act_cache_e4m3_slot for the concat tensor at (1 row,
 * 3*n_embd); the caller notes the encoding current and the f32 store skipped.
 * @return 1 on success, 0 on a bad argument or launch failure. */
/** @param slot_data  E4M3 slot data
 *  @param slot_scale its scale slab
 *  @param sf_pitch   its scale pitch
 *  @param h0         first hidden state, f32 [n_embd]
 *  @param h1         second
 *  @param h2         third
 *  @param n_embd     width of each */
int pulsar_gpu_dspark_concat3_e4m3(void *slot_data, void *slot_scale, int sf_pitch,
                                   const pulsar_gpu_tensor *h0, const pulsar_gpu_tensor *h1,
                                   const pulsar_gpu_tensor *h2, uint32_t n_embd);
/** The most rows one batched drafter seed chunk carries (scratch capacity; the
 *  chunk is also bounded by pulsar_gpu_matmul_decode_exact_rows()). */
#define PULSAR_DSPARK_SEED_ROWS 16u
/** L260: n_rows concat rows at once -- slot row r is [h0|h1|h2] of source row
 *  src_rows[r] of the [rows, n_embd] tensors h0/h1/h2 (the verify capture). */
/** @param slot_data  E4M3 slot data, [n_rows][3 n_embd]
 *  @param slot_scale its scale slab
 *  @param sf_pitch   its scale pitch
 *  @param h0         f32 [rows][n_embd]
 *  @param h1         f32 [rows][n_embd]
 *  @param h2         f32 [rows][n_embd]
 *  @param src_rows   host [n_rows] source row per slot row
 *  @param n_rows     rows (<= PULSAR_DSPARK_SEED_ROWS)
 *  @param n_embd     width of each source
 *  @return 1 on success, 0 on a bad argument or launch failure */
int pulsar_gpu_dspark_concat3_e4m3_rows(void *slot_data, void *slot_scale, int sf_pitch,
                                        const pulsar_gpu_tensor *h0, const pulsar_gpu_tensor *h1,
                                        const pulsar_gpu_tensor *h2, const uint32_t *src_rows,
                                        uint32_t n_rows, uint32_t n_embd);

/** L158: give an OFFSET ROW VIEW of an encoded activation its own slot, filled
 * from the producer's encoding (byte copy + scale re-base; no quantise).  Used
 * by the mixed-batch prefix splits for their prefill suffix views so every
 * consumer finds the encoding by its ordinary lookup.  0 = no valid encoding
 * on the full activation (the caller refuses). */
/** @param x_full the encoded full activation
 *  @param row0   the view's first row
 *  @param rows   the view's rows
 *  @param in_dim row width
 *  @param x_view the view (its new slot's key)
 *  @return 1 on success, 0 (see above) */
int pulsar_gpu_mxfp8_act_cache_window(const pulsar_gpu_tensor *x_full, uint64_t row0, uint64_t rows,
                                      uint64_t in_dim, const pulsar_gpu_tensor *x_view);

/** L158: PRODUCER-side encode of an f32 activation the caller itself produced
 * and owns -- for tests and tools that synthesise an activation.  Arms the
 * slot for (x, n_tok, in_dim), encodes with the engine's quantiser, notes it
 * current.  Engine producers emit from their epilogues and must not call this;
 * no consumer quantises any more. @return 1 on success. */
/** @param x      f32 [n_tok][in_dim]
 *  @param n_tok  rows
 *  @param in_dim row width */
int pulsar_gpu_mxfp8_act_cache_encode_f32(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);

/** L158: PRODUCER-side grouped encode of the attention output heads for an
 * attention producer without an E4M3 epilogue (the drafter's raw batch
 * attention).  Call right after the inverse rope tail and before the attn-out
 * projection; pulsar_gpu_mxfp8_gact_disarm after it. @return 1 on success. */
/** @param heads     attention output [n_tokens][n_groups * group_dim]
 *  @param n_tokens  rows
 *  @param n_groups  output groups
 *  @param group_dim width of one group */
int pulsar_gpu_mxfp8_gact_emit_heads(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                                     uint32_t n_groups, uint64_t group_dim);
/** L260: encode rows [row0, row0 + n_rows) of the n_tokens-row heads into the
 *  grouped slot ALREADY armed for them (pulsar_gpu_mxfp8_gact_slot: an attention
 *  epilogue and the rope tail filled the other rows) -- the split-K decode rows,
 *  whose combine has no epilogue.  Run after the rope tail; refuses when no slot
 *  is armed for exactly these heads.  Does not note. */
/** @param heads     attention output, the armed slot's key
 *  @param n_tokens  its rows
 *  @param row0      first row to encode
 *  @param n_rows    rows to encode
 *  @param n_groups  output groups
 *  @param group_dim width of one group
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_gpu_mxfp8_gact_emit_heads_rows(const pulsar_gpu_tensor *heads, uint32_t n_tokens,
                                          uint32_t row0, uint32_t n_rows,
                                          uint32_t n_groups, uint64_t group_dim);

/** Declare the E4M3 encoding current after a producer filled those slots. */
void pulsar_gpu_mxfp8_act_cache_note_mxfp8(void);

/* ---- Engram (L242, V4.1) ------------------------------------------------------
 * The gathered table rows -- [n_tok][24][264 B]: 256 E4M3 values then their 8
 * E8M0 block scales, the row file's record -- become the `wkv` GEMM's activation
 * AS THEY ARE: emitted into the MXFP8 slot of `x_key` (an f32 [n_tok][6144]
 * tensor that is the slot's KEY and is never written), armed, noted current and
 * f32-skipped.  The `wkv` GEMM is the ordinary pulsar_gpu_matmul_mxfp8_tensor on
 * `x_key`.  @return 1 on success. */
/** @brief Emit the gathered Engram rows as the wkv GEMM's MXFP8 activation (see above).
 *  @param x_key f32 [n_tok][6144]: the slot's key, never written
 *  @param rows  the gathered rows, [n_tok][24] records of 264 B
 *  @param n_tok tokens */
/** @return 1 on success, 0 on failure */
int pulsar_gpu_engram_rows_emit(const pulsar_gpu_tensor *x_key, const pulsar_gpu_tensor *rows,
                                uint32_t n_tok);
/* The gate and the gated add on the bf16 hyper-connection copies
 * hc[n_tok][n_hc][dim]: kv is the `wkv` output [n_tok][(n_hc+1)*dim] f32 (n_hc
 * keys then one value), qk_weight the f32 [n_hc][dim] product q_weight*k_weight,
 * dead an optional [n_tok] byte mask (1 = image span, copies untouched), eps the
 * model's norm eps.  Math in f32, one bf16 rounding on the store. */
/** @brief The Engram gate and gated add (see above).
 *  @param hc        bf16 [n_tok][n_hc][dim], updated in place
 *  @param kv        f32 [n_tok][(n_hc + 1) * dim]: n_hc keys then one value
 *  @param qk_weight f32 [n_hc][dim]: q_weight * k_weight
 *  @param dead      byte [n_tok] mask (1 = leave the row untouched), or NULL
 *  @param n_tok     tokens
 *  @param n_hc      HC streams
 *  @param dim       row width
 *  @param eps       the model's norm epsilon
 *  @return 1 on success, 0 on failure */
int pulsar_gpu_engram_gate_add(pulsar_gpu_tensor *hc, const pulsar_gpu_tensor *kv,
                               const pulsar_gpu_tensor *qk_weight, const pulsar_gpu_tensor *dead,
                               uint32_t n_tok, uint32_t n_hc, uint32_t dim, float eps);

/* Record that the producer ALSO skipped this buffer's f32 store, so the f32
 * bytes are stale.  Call only after note_mxfp8(), and only from a producer that
 * actually emitted the encoding -- the two must be gated on the SAME predicate,
 * or the GEMM reads a store that was never written and returns a well-formed
 * wrong answer.  Every f32-reading arm of the mxfp8 family checks this and
 * fails loudly rather than run. */
/** keep_from: rows below it were skipped; rows >= keep it.  A full skip
 * passes the arming n_tok. */
/** @param keep_from first row whose f32 store was kept */
void pulsar_gpu_mxfp8_act_cache_note_f32_skipped(uint32_t keep_from);

/** Hand back the E4M3 encoding this buffer already carries, or 0 if the cache
 * holds none for (ptr, n_tok, in_dim). Lets a consumer that would otherwise
 * quantize the f32 copy reuse the producer's encoding instead -- the routed-MoE
 * gather is the one that matters, since it re-encoded every gathered row.
 * `scale` is indexed with the mx_sfoff swizzle over `kbp` blocks per row. */
/** @param x      the activation
 *  @param n_tok  rows
 *  @param in_dim row width
 *  @param data   receives the E4M3 plane
 *  @param scale  receives the swizzled scale slab
 *  @param kbp    receives the blocks per row
 *  @return 1 when an encoding is current, else 0 */
int pulsar_gpu_mxfp8_act_cache_get_e4m3(const pulsar_gpu_tensor *x,
                                        uint64_t n_tok,
                                        uint64_t in_dim,
                                        const void **data,
                                        const void **scale,
                                        int *kbp);

/** L246: decode the E4M3 slot armed for (x, n_tok, in_dim) into `dst` as f32
 * [n_tok][in_dim] -- the collection-mode reader of an activation the producers
 * keep only as E4M3 (the routed SwiGLU leaf).  Refuses when no slot is armed.
 * Not a serving-path call. */
/** @param x      the armed activation (the slot's key)
 *  @param n_tok  rows
 *  @param in_dim row width
 *  @param dst    f32 [n_tok][in_dim] out
 *  @return 1 on success, 0 when no slot is armed or on failure */
int pulsar_gpu_mxfp8_act_cache_decode_f32(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim,
                                          pulsar_gpu_tensor *dst);

/** Same lookup keyed on the raw device pointer, for callers that only ever held
 * one (the routed-MoE path takes float* activations, not tensors). */
/** @param ptr    the activation's device pointer
 *  @param n_tok  rows
 *  @param in_dim row width
 *  @param data   receives the E4M3 plane
 *  @param scale  receives the swizzled scale slab
 *  @param kbp    receives the blocks per row
 *  @return 1 when an encoding is current, else 0 */
int pulsar_gpu_mxfp8_act_cache_get_e4m3_ptr(const void *ptr,
                                            uint64_t n_tok,
                                            uint64_t in_dim,
                                            const void **data,
                                            const void **scale,
                                            int *kbp);


/** @brief RMS-normalise `rows` HC-residual rows with no learned weight (x * rsqrt(mean(x^2) + eps)).
 *  @param out      f32 [rows][n] (unwritten when skip_f32)
 *  @param out_b    see below
 *  @param x        the rows, HC residual elements (PULSAR_HC_ELT_SIZE)
 *  @param n        row width
 *  @param rows     rows
 *  @param eps      RMS epsilon
 *  @param skip_f32 see below
 *  @return 1 on success, 0 on a bad size, skip_f32 without out_b, or launch failure */
int pulsar_gpu_rms_norm_plain_rows_tensor(
        pulsar_gpu_tensor       *out,
        /* Optional BF16 copy of the normalised rows (L086 T3), from
         *  pulsar_gpu_bf16_act_slot().  NULL emits nothing.  This buffer's
         *  consumer is a BF16 GEMM -- pulsar_gpu_matmul_f32_tensor resolves a
         *  bf16 weight and runs the shared bf16 core -- so emitting from the
         *  epilogue DELETES its convert pass rather than relocating it. */
        void                    *out_b,
        const pulsar_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps,
        /* L157: nonzero skips the f32 rows entirely (requires out_b).  Legal
         *  only when every consumer reads the bf16 copy -- the caller must also
         *  declare it with pulsar_gpu_act_note_f32_skipped_for() so a
         *  consumer that misses the bf16 slot refuses instead of converting
         *  unwritten bytes.  At hc_dim the f32 rows are 40% of this kernel's
         *  traffic (64 of 160 KB per row). */
        int                     skip_f32);

/** L157: declare that rows [keep_from, n_tok) of x's f32 store were skipped,
 * for a buffer that was never arm()ed (bf16 slot only), keyed like the slot. */
/** @param x         the buffer
 *  @param n_tok     its rows
 *  @param in_dim    its row width
 *  @param keep_from first row whose f32 store was kept */
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
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`).
 */
int pulsar_gpu_rms_norm_weight_mx_tensor(
        pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps,
        void *out_q, void *out_sf, int out_kbp,
        void *out_b,
        int w_bf16);

/** @brief One row of RMS norm with a learned weight: out = x * rsqrt(mean(x^2) + eps) * w.
 *  @param out           f32 [n]
 *  @param x             f32 [n]
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset byte offset of the norm weight [n]
 *  @param n             row width
 *  @param eps           RMS epsilon
 *  @param w_bf16        1 when the weight is stored bf16, 0 for f32
 *  @return 1 on success, 0 on a bad shape or launch failure */
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
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`). */
/** @param out           f32 [rows][n], or NULL (see above)
 *  @param x             f32 [rows][n]
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset byte offset of the norm weight [n]
 *  @param n             row width
 *  @param rows          rows
 *  @param eps           RMS epsilon
 *  @param out_q         E4M3 slot, or NULL
 *  @param out_sf        its scale slab, or NULL
 *  @param out_kbp       its scale pitch
 *  @param out_b         bf16 plane, or NULL
 *  @param w_bf16        1 when the weight is stored bf16 */
int pulsar_gpu_rms_norm_weight_rows_mx_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x,
                                              const void *model_map, uint64_t model_size,
                                              uint64_t weight_offset, uint32_t n, uint32_t rows, float eps,
                                              void *out_q, void *out_sf, int out_kbp, void *out_b, int w_bf16);

/** @brief `rows` rows of RMS norm with one learned weight.
 *  @param out           f32 [rows][n]
 *  @param x             f32 [rows][n]
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset byte offset of the norm weight [n]
 *  @param n             row width
 *  @param rows          rows
 *  @param eps           RMS epsilon
 *  @param out_b         bf16 plane for a bf16-weight consumer, or NULL
 *  @param w_bf16        1 when the weight is stored bf16
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_rms_norm_weight_rows_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps,
        void                   *out_b,     // bf16 plane for a bf16-weight consumer, or NULL (L159)
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
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`).
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

/** THE TWO KV ROWS (L218, DeepSeek-V4.1).  Each holds, byte for byte, what
 * the reference stores after its in-place quantise (dequantised bf16 values),
 * so a reader multiplying code by scale in fp32 reproduces the reference's
 * tensor exactly.
 *
 *   WINDOW row -- every layer's sliding-window ring (and the drafter's):
 *     [head_dim E4M3][head_dim / 32 E8M0]              = 528 B at head_dim 512
 *     act_quant(kv, 32): amax floored 1e-4, scale 2^ceil(log2(amax / 448)).
 *   MAIN row -- a kv source's compressed pool:
 *     [head_dim / 2 E2M1 nibbles, low nibble first][head_dim / 16 E4M3]
 *                                                      = 288 B at head_dim 512
 *     fp4_act_quant(latent, 16, e4m3 scales): amax floored 6 * 2^-9, scale
 *     e4m3(amax / 6).  E2M1 x E4M3 products fit bf16, so the stored value is
 *     the reference's bf16 exactly.
 *
 * Quantise EXACTLY ONCE (pulsar_cuda_kvrows.cu); every later move is a byte
 * move, and there is no conversion path from any other row format (0731's
 * unified 384 B NVFP4 row refuses through the payload / segment versions).
 * Bumping either layout MUST bump PULSAR_SESSION_KV_PAYLOAD_VERSION and
 * PULSAR_SESSION_SEGMENT_VERSION. */
#define PULSAR_WINKV_BLOCK  32u
/** Values per E4M3 scale in a MAIN row. */
#define PULSAR_MAINKV_BLOCK 16u
/** Bytes of one WINDOW row at head_dim HD. */
#define PULSAR_WINKV_ROWBYTES(HD)  ((uint64_t)(HD) + (HD) / PULSAR_WINKV_BLOCK)
/** Bytes of one MAIN row at head_dim HD. */
#define PULSAR_MAINKV_ROWBYTES(HD) ((uint64_t)(HD) / 2u + (HD) / PULSAR_MAINKV_BLOCK)

/** The KV row family a model stores, and the buffer a geometry question is
 * about.  These live HERE because both sides of the seam need them: the engine
 * sizes caches and payload spans from the profile, and the CUDA module packs
 * the rows.  One definition, two readers.
 *
 * CSA2:    V4.1.  WINDOW rows in every sliding-window ring, MAIN rows in a kv
 *          source's compressed pool, plus an FP4 index-K row.
 * UNIFIED: 0731.  One NVFP4 row for every non-index buffer, plus the same index
 *          row.  plans/96-two-profiles-one-engine.md s2 axis 2. */
typedef enum {
    PULSAR_KV_ROWS_CSA2    = 0,
    PULSAR_KV_ROWS_UNIFIED = 1,
} pulsar_kv_row_style;
/** Which KV buffer a row-geometry question (pulsar_gpu_kv_row_bytes) is about. */

typedef enum {
    PULSAR_KV_ROW_RING = 0,   /**< a sliding-window ring row */
    PULSAR_KV_ROW_COMP,       /**< a compressed-pool row */
    PULSAR_KV_ROW_INDEX,      /**< an index-K pool row */
} pulsar_kv_row_kind;

/** Push the loaded profile's row family into the CUDA module, once at load.
 * The profile stays the authority; this is the seam's copy of it, and it is how
 * the packers below learn which family to write without every call site
 * carrying it. */
/** @param style the loaded profile's row family */
void pulsar_gpu_set_kv_row_style(pulsar_kv_row_style style);

/** Where the HC collapse takes its coefficients, pushed once at load from the
 * profile (`pulsar_shape::hc_head_mix`).
 *
 * 0731's head COMPUTES its own coefficients, so a sublayer collapses with the
 * split it just derived (`sp`); V4.1's collapses with the `pre` its previous
 * sublayer handed on.  The two are different functions of the same inputs, and
 * the branch applied V4.1's to both: on V4 every layer's attention collapsed
 * with the wrong coefficients, which showed up as one bf16 ULP in
 * `attn_norm` at layer 0 and grew to a 0.48 logit correlation by the head.
 * The same flag also decides the two bf16 round trips the reference's
 * `hc_pre`/norm narrowing implies for V4.1's arithmetic and 0731's does not:
 * that profile's kernel, which served 0731, has neither.
 *
 * @param on true = the head computes them (0731); false = carry the pre (V4.1). */
void pulsar_gpu_set_hc_head_mix(bool on);
/** @return the row family pushed by pulsar_gpu_set_kv_row_style (CSA2 until set) */
pulsar_kv_row_style pulsar_gpu_kv_row_style(void);

/** The row geometry, in bytes, asked by KIND -- the CUDA side's reader.  head_dim
 * is a parameter, as everywhere else in this module.  The index-K row is not
 * packed here (the engine carries its stride), so asking for it refuses. */
/** @param kind     RING or COMP (INDEX refuses)
 *  @param head_dim head width
 *  @return bytes per row of that kind in the loaded family, or 0 for INDEX (said) */
uint64_t pulsar_gpu_kv_row_bytes(pulsar_kv_row_kind kind, uint32_t head_dim);

/** 0731's unified NVFP4 attention KV row (the L111 format).  Restored for the
 * two-profile engine: V4.1 replaced it, 0731 stores it in EVERY non-index
 * buffer (raw ring, comp pool, drafter ring, MTP cache, current chunk).
 *
 *   [n_nope/2 E2M1 nibbles][n_nope/16 E4M3 scale codes][f32 row scale]
 *   [n_rot bf16 rope]   = 384 B at head_dim 512 / n_rot 64
 *
 * Requires n_rot == PULSAR_ATTN_PACK_NROT and (head_dim - n_rot) a multiple of
 * PULSAR_KV4_NV_BLOCK.  Quantise EXACTLY ONCE; every later move is a byte move.
 *
 * The packer is src/cuda/pulsar_cuda_attnpack.cu; the oracle is
 * tests/attn_pack_fixture.h (host, run by `make attn-pack-fixture-check`) and
 * tests/kv4_pack_gate.cpp (device vs host).  plans/96-two-profiles-one-engine.md
 * s11/s12. */
#define PULSAR_ATTN_PACK_NROT 64u
/** Alignment of the unified row's non-rope width (the kernels' 64-wide lanes). */
#define PULSAR_ATTN_PACK_NOPE_ALIGN 64u   /* the kernels walk the nope dims in 64-wide lanes */
/** Non-rope dims of a unified row at head_dim HD. */
#define PULSAR_ATTN_PACK_NOPE(HD) ((HD) - PULSAR_ATTN_PACK_NROT)
/** Bytes of a unified row's E2M1 nibbles. */
#define PULSAR_ATTN_PACK_NIB(HD)  (PULSAR_ATTN_PACK_NOPE(HD) / 2u)
/** Values per E4M3 scale code in a unified row. */
#define PULSAR_KV4_NV_BLOCK      16u
/** E4M3 scale codes in a unified row. */
#define PULSAR_KV4_NV_NBLK(HD)   (PULSAR_ATTN_PACK_NOPE(HD) / PULSAR_KV4_NV_BLOCK)
/** Bytes of one unified NVFP4 row at head_dim HD (384 at 512). */
#define PULSAR_ATTN_PACK_ROWBYTES(HD) \
    ((uint64_t)PULSAR_ATTN_PACK_NIB(HD) + PULSAR_KV4_NV_NBLK(HD) + 4u + \
     (uint64_t)PULSAR_ATTN_PACK_NROT * 2u)
/** Microscaling compressed-KV row (the indexer's FP4 cache): one E8M0 scale
 * byte per 32 elements, [HD/2 E2M1 nibble bytes][NBLK scale bytes]
 * (HD=128 -> 68 B/row).  CUTLASS-consumable layout; the GEMM re-tiles the
 * scales into its swizzled SF layout at use time. */
#define PULSAR_MXKV_BLOCK 32u
/** E8M0 scale bytes of an MXKV-FP4 row. */
#define PULSAR_MXKV_NBLK(HD) (((HD) + PULSAR_MXKV_BLOCK - 1u) / PULSAR_MXKV_BLOCK)
/** Bytes of one MXKV-FP4 row at head_dim HD (68 at 128). */
#define PULSAR_MXKV_FP4_ROWBYTES(HD) (((HD) + 1u) / 2u + PULSAR_MXKV_NBLK(HD))

/** The compressor's width multiplier at a given compress ratio: 0731's ratio-4
 * (CSA) compressor pools two OVERLAPPING groups and emits 2 * head_dim, while
 * its ratio-128 (HCA) compressor emits head_dim; V4.1's CSA2 ratios (1 and 2)
 * are both 1.  One authority for that width -- the binder's shape expectations
 * and the kernels read this rather than keeping their own copy. */
/** @param ratio the compress ratio
 *  @return 2 at ratio 4 (the overlapping CSA compressor), else 1 */
static inline PULSAR_GPU_HD uint32_t pulsar_compress_coff(uint32_t ratio) { return ratio == 4u ? 2u : 1u; }

/** The window-row packer: quantise `n_rows` f32 rows of `src` to WINDOW rows
 * (see the format block) and store them.  raw_cap == 0: consecutive rows of
 * `packed` from out_row0 (the batch pack buffer).  raw_cap != 0: a ring
 * scatter, destination row = seq_id[r] * raw_cap + pos % raw_cap with pos =
 * positions[r] or out_row0 + r; a row whose seq_id lies outside n_banks stores
 * nothing.  `x` (optional) receives the dequantised values -- OBSERVER-ONLY,
 * pass NULL unless gpu_graph_f32_store_observed_any().
 * @return nonzero on success, 0 on a bad shape or a failed launch. */
/** @param x         f32 dequantised copy out, or NULL (observer-only)
 *  @param src       f32 [n_rows][head_dim]
 *  @param packed    WINDOW rows out (the batch buffer, or the bank ring pool)
 *  @param out_row0  first destination row, or the position of row 0 in ring mode
 *  @param n_rows    rows
 *  @param head_dim  head width
 *  @param positions int32 [n_rows] positions (ring mode), or NULL
 *  @param seq_id    int32 [n_rows] bank ids (ring mode), or NULL
 *  @param n_banks   banks in the pool
 *  @param raw_cap   ring capacity, rows; 0 = consecutive rows */
int pulsar_gpu_winkv_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                 uint32_t out_row0, uint32_t n_rows, uint32_t head_dim,
                                 const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                                 uint32_t n_banks, uint32_t raw_cap);
/** The main-row packer: quantise `n_rows` f32 rows of `src` to MAIN rows and
 * store them at `packed` rows [out_row0, out_row0 + n_rows).  `x` as above. */
/** @param x        f32 dequantised copy out, or NULL (observer-only)
 *  @param src      f32 [n_rows][head_dim]
 *  @param packed   MAIN rows out
 *  @param out_row0 first destination row
 *  @param n_rows   rows
 *  @param head_dim head width
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_mainkv_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                  uint32_t out_row0, uint32_t n_rows, uint32_t head_dim);

/** 0731's unified NVFP4 row packer (src/cuda/pulsar_cuda_attnpack.cu): quantise
 * `n_rows` f32 rows of `src` and store them at `packed` rows
 * [out_row0, out_row0 + n_rows).  `x`, when non-NULL, receives the DECODED
 * values -- observer-only, pass NULL unless gpu_graph_f32_store_observed_any().
 * Public because tests/attn_pack_gate.cpp grades it; the ENGINE reaches it
 * through the dispatchers below, never by name. */
/** @param x        f32 decoded copy out, or NULL (observer-only)
 *  @param src      f32 [n_rows][head_dim]
 *  @param packed   unified rows out
 *  @param out_row0 first destination row
 *  @param n_rows   rows
 *  @param head_dim head width
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_attn_pack_store_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src,
                                      pulsar_gpu_tensor *packed, uint32_t out_row0,
                                      uint32_t n_rows, uint32_t head_dim);

/** ---- the row-family DISPATCHERS: the engine-facing pack surface ----------
 *
 * Callers ask to pack a RING row or a COMP row.  WHICH format that is comes
 * from the loaded profile (pulsar_kv_row_style): V4.1 has two distinct formats
 * (the two packers above), 0731 has one unified NVFP4 row.  The engine must
 * call these rather than a family's packer, so that no site can be left packing
 * the wrong format for the loaded model -- which is how a new family silently
 * mis-sizes a buffer instead of failing.  The family packers stay public for
 * the gates that test a format on purpose.
 *
 * Signatures mirror the two they dispatch to. */
/** @brief Pack ring rows in the loaded family's format (arguments as pulsar_gpu_winkv_pack_tensor).
 *  @param x         f32 decoded copy out, or NULL
 *  @param src       f32 [n_rows][head_dim]
 *  @param packed    ring rows out
 *  @param out_row0  first row / position of row 0
 *  @param n_rows    rows
 *  @param head_dim  head width
 *  @param positions int32 [n_rows] positions, or NULL
 *  @param seq_id    int32 [n_rows] bank ids, or NULL
 *  @param n_banks   banks in the pool
 *  @param raw_cap   ring capacity; 0 = consecutive rows
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_kv_ring_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                   uint32_t out_row0, uint32_t n_rows, uint32_t head_dim,
                                   const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id,
                                   uint32_t n_banks, uint32_t raw_cap);
/** @brief Pack compressed-pool rows in the loaded family's format.
 *  @param x        f32 decoded copy out, or NULL
 *  @param src      f32 [n_rows][head_dim]
 *  @param packed   comp rows out
 *  @param out_row0 first destination row
 *  @param n_rows   rows
 *  @param head_dim head width
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_kv_comp_pack_tensor(pulsar_gpu_tensor *x, const pulsar_gpu_tensor *src, pulsar_gpu_tensor *packed,
                                   uint32_t out_row0, uint32_t n_rows, uint32_t head_dim);

/** Fused rope + FP4 pack for the indexer q projection (L218: the reference's
 * fp4_act_quant with E8M0 scales per 32 on the bf16-rounded row; 0731's
 * Hadamard is gone): one launch replacing a rope_tail + pack pair over the
 * same tensor, bit-exact vs that sequence (shared rotation device fn, same
 * pack body, same order). */
/** @param x         f32 [n_tok][n_head][head_dim] staging, roped in place
 *  @param packed    MXKV-FP4 rows out, [n_tok][n_head]
 *  @param n_tok     rows
 *  @param n_head    indexer heads
 *  @param head_dim  indexer head width
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param pos0        position of row 0 (when positions is NULL)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param inverse     rotate by the negated angle (undo the rope)
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param positions   int32 [n_tok] device positions, or NULL (see the positions note above)
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_indexer_rope_fp4_pack_tensor(
        pulsar_gpu_tensor *x,          /* f32 rope staging, mutated in place */
        pulsar_gpu_tensor *packed,     /* MXKV FP4 rows out -- the only Q output */
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions);

/** Pack n_rows f32 rows of x MXKV-FP4 into `packed` at rows [out_row0,
 * out_row0+n_rows) -- the same quant the fused rope entry above applies.
 *
 * @param x         f32 indexer rows, round-tripped in place
 * @param packed    destination for the MXKV-FP4 rows
 * @param out_row0  first row of `packed` to write
 * @param n_rows    rows to process
 * @param head_dim  per-head width
 * @param keep_f32  write the dequantised values back into the f32 staging.
 *                  OBSERVER-ONLY -- consumers read the packed rows. Pass
 *                  gpu_graph_f32_store_observed_any() (L094).
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`).
 */
int pulsar_gpu_indexer_fp4_pack_tensor(
        pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim,
        bool              keep_f32);

/** V4's indexer key: the reference's `rotate=True` compressor runs
 * `rotate_activation(x) = hadamard_transform(x, d**-0.5)` -- the natural-order
 * 128-point Walsh-Hadamard transform scaled by 1/sqrt(128) -- before the same
 * fp4 quant the entry above applies.  L218 deleted the rotation with the 0731
 * checkpoint; this is its restore, and its ORDER and NORMALISATION are pinned
 * by tests/indexer_hadamard_kernel_test rather than argued.
 *
 * Unreachable for a V4.1 artifact: `g_pulsar_shape.indexer_own_compressor` is
 * false there and V4.1's index key carries no rotation.
 *
 * @return nonzero on success, 0 on a bad shape or a failed launch. */
/** @param x        f32 [n_rows][head_dim] index-key rows
 *  @param packed   MXKV-FP4 rows out
 *  @param out_row0 first row of packed to write
 *  @param n_rows   rows
 *  @param head_dim 128
 *  @param keep_f32 write the rotated, dequantised values back into x (observer-only) */
int pulsar_gpu_dsv4_indexer_qat_pack_tensor(
        pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim,
        bool              keep_f32);

/** V4's indexer q: rope the tail, then the rotation+pack above, one launch.
 * Indexer.forward does `apply_rotary_emb(q[..., -rd:]); q = rotate_activation(q)`
 * and every consumer reads the PACKED row, so there is no f32 q to observe. */
/** @param x          f32 [n_tok][n_head][head_dim] staging, modified in place
 *  @param packed     MXKV-FP4 rows out, [n_tok][n_head]
 *  @param n_tok      rows
 *  @param n_head     indexer heads
 *  @param head_dim   128
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param pos0        position of row 0 (when positions is NULL)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param inverse     rotate by the negated angle (undo the rope)
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param positions   int32 [n_tok] device positions, or NULL (see the positions note above)
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_dsv4_indexer_rope_qat_tensor(
        pulsar_gpu_tensor *x,
        pulsar_gpu_tensor *packed,
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

/** V4's indexer key, end to end: the ape fold, the gated pooling + RMSNorm, the
 * rope at each group's position, then the rotation and fp4 pack above.  This is
 * `Compressor(args, ratio, head_dim = PULSAR_N_INDEXER_HEAD_DIM, rotate=True)`
 * as the Indexer uses it -- a SECOND compressor with its own weights, its own
 * state lane and its own output pool, not the kv source's latent (which is what
 * V4.1's index key is).
 *
 * Every step is an existing kernel; the ORDER is the whole content.  `sc` is
 * modified in place (the ape fold is the reference's `score += self.ape`).
 *
 * @return nonzero on success, 0 on bad operands or a failed launch. */
/**
 *  @param packed      MXKV-FP4 index-K rows out (the indexer's own pool)
 *  @param latent      f32 pooled-row scratch
 *  @param state_kv    the compressor's pending-group kv lane
 *  @param state_score the compressor's pending-group score lane
 *  @param sc          f32 score projections, ape-folded in place
 *  @param kv          f32 kv projections
 *  @param model_map   the model mapping
 *  @param model_size  its size in bytes
 *  @param ape_offset  byte offset of the ape table [ratio][coff * head_dim]
 *  @param ape_type    PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param norm_offset byte offset of the compressor norm weight [head_dim]
 *  @param norm_type   PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param out_row0    first row of packed to write
 *  @param head_dim    indexer head width
 *  @param ratio       compress ratio
 *  @param pos0        position of the first token
 *  @param n_tokens    tokens
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param rms_eps     RMS epsilon of the pooled-row norm */
int pulsar_gpu_indexer_compressor_prefill_tensor(
        pulsar_gpu_tensor       *packed,
        pulsar_gpu_tensor       *latent,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        pulsar_gpu_tensor       *sc,
        const pulsar_gpu_tensor *kv,
        const void              *model_map,
        uint64_t                 model_size,
        uint64_t                 ape_offset,
        uint32_t                 ape_type,
        uint64_t                 norm_offset,
        uint32_t                 norm_type,
        uint32_t                 out_row0,
        uint32_t                 head_dim,
        uint32_t                 ratio,
        uint32_t                 pos0,
        uint32_t                 n_tokens,
        uint32_t                 n_rot,
        uint32_t                 n_ctx_orig,
        float                    freq_base,
        float                    freq_scale,
        float                    ext_factor,
        float                    attn_factor,
        float                    beta_fast,
        float                    beta_slow,
        float                    rms_eps);

/** The decode-phase twin of the composite above: one token's row, through the
 * indexer's OWN compressor, emitting the group's index-K row when that token
 * closes a group.  `emitted` is set exactly when a row was packed (0 on success
 * without one, so the caller advances its frontier on `emitted`, not on the
 * return value).
 *
 * `sc` is the token's score row and `ape` the compressor's absolute-position
 * embedding ([ratio][coff*head_dim]); the ape fold happens HERE, before the
 * store, because that is where the reference's decode branch does it
 * (`score += self.ape[start_pos % ratio]`).  The pooled row is roped at the
 * group's FIRST position and then rotated and fp4-packed, matching
 * pulsar_gpu_indexer_compressor_prefill_tensor row for row.
 *
 * @return nonzero on success, 0 on bad operands or a failed launch. */
/**
 *  @param packed      MXKV-FP4 index-K rows out (the indexer's own pool)
 *  @param latent      f32 pooled-row scratch
 *  @param state_kv    the compressor's pending-group kv lane
 *  @param state_score the compressor's pending-group score lane
 *  @param sc          f32 [1] score row, ape-folded in place
 *  @param kv          f32 [1] kv row
 *  @param model_map   the model mapping
 *  @param model_size  its size in bytes
 *  @param ape_offset  byte offset of the ape table [ratio][coff * head_dim]
 *  @param ape_type    PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param norm_offset byte offset of the compressor norm weight [head_dim]
 *  @param norm_type   PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param out_row     row of packed to write when a group closes
 *  @param head_dim    indexer head width
 *  @param ratio       compress ratio
 *  @param pos         the token's position
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param rms_eps     RMS epsilon of the pooled-row norm
 *  @param emitted     set to 1 when a row was packed, else 0 */
int pulsar_gpu_indexer_compressor_update_tensor(
        pulsar_gpu_tensor       *packed,
        pulsar_gpu_tensor       *latent,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        pulsar_gpu_tensor       *sc,
        const pulsar_gpu_tensor *kv,
        const void              *model_map,
        uint64_t                 model_size,
        uint64_t                 ape_offset,
        uint32_t                 ape_type,
        uint64_t                 norm_offset,
        uint32_t                 norm_type,
        uint32_t                 out_row,
        uint32_t                 head_dim,
        uint32_t                 ratio,
        uint32_t                 pos,
        uint32_t                 n_rot,
        uint32_t                 n_ctx_orig,
        float                    freq_base,
        float                    freq_scale,
        float                    ext_factor,
        float                    attn_factor,
        float                    beta_fast,
        float                    beta_slow,
        float                    rms_eps,
        int                     *emitted);

/** As below, but also emits the grouped E4M3 encoding for the MX blocks this
 * kernel rewrites -- head dims [head_dim - n_rot, head_dim).  It is the second
 * half of the attn-output "a" activation: the fp16 attention epilogue emits
 * [0, head_dim - n_rot) and this emits the rest, because this kernel rewrites
 * that range IN PLACE after attention has already run.  Pass NULL slots for
 * the plain behaviour. */
/** @param x           [n_tok][n_head][head_dim] heads, rotated in place
 *  @param n_tok       rows
 *  @param n_head      heads
 *  @param head_dim    head width
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param pos0        position of row 0 (when positions is NULL)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param inverse     rotate by the negated angle (undo the rope)
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param positions   int32 [n_tok] device positions, or NULL (see the positions note above)
 *  @param gact_data   grouped E4M3 slot (pulsar_gpu_mxfp8_gact_slot), or NULL
 *  @param gact_scale  its scale slab
 *  @param gact_kbp    its scale pitch
 *  @param gact_slab   bytes of one group's scale slab
 *  @param n_groups    output groups
 *  @param gact_tok0   this launch's first row in the slot
 *  @param gact_ntok   the slot's row count
 *  @return 1 on success, 0 on a bad shape or launch failure */
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
        uint32_t          n_groups,
        uint32_t gact_tok0, uint32_t gact_ntok);

/** The reference's per-head Q norm and tail rope, fused: RMS-normalise each
 * head of `x` in f32 (`x *= rsqrt(mean(x^2) + eps)`, no learned weight) and
 * rotate its tail in place, rounding back into `x`'s element type -- 0731's
 * Attention.forward, and bit-exact with the fp16 attention kernel's fused
 * prologue (attn_f16.cu replicates this reduction operation for operation).
 *
 * `x` is [n_tok][n_head][head_dim] and may be f32 or f16; the element type is
 * derived from the buffer, never passed.  Unreachable for a V4.1 artifact:
 * `g_pulsar_shape.q_head_norm` is false there. */
/** @param x           [n_tok][n_head][head_dim], f32 or f16, in place
 *  @param n_tok       rows
 *  @param n_head      heads
 *  @param head_dim    head width
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param pos0        position of row 0 (when positions is NULL)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param inverse     rotate by the negated angle (undo the rope)
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param eps         per-head RMS epsilon
 *  @param positions   int32 [n_tok] device positions, or NULL
 *  @return 1 on success, 0 on a bad shape or launch failure */
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

/** @brief Rotate the last n_rot dims of every head in place (DS4's tail-only RoPE, YaRN-capable).
 *  @param x           [n_tok][n_head][head_dim] (element type from the tensor)
 *  @param n_tok       rows
 *  @param n_head      heads
 *  @param head_dim    head width
 *  @param n_rot       rotary dimensions at each head's tail
 *  @param pos0        position of row 0 (when positions is NULL)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param inverse     rotate by the negated angle (undo the rope)
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix; 0 disables it
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start, rotations
 *  @param beta_slow   YaRN ramp end, rotations
 *  @param positions   int32 [n_tok] device positions, or NULL (see the positions note above)
 *  @return 1 on success, 0 on a bad shape or launch failure */
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

/** CSA2 (L218): rotate `n_rows` single-head f32 rows standing for positions
 * pos0, pos0 + pos_stride, ... (a kv source's compressed rows, one per group
 * of `ratio` tokens, each rotated at its group's first position). */
/** @param x           f32 [n_rows][head_dim], rotated in place
 *  @param n_rows      rows
 *  @param head_dim    head width
 *  @param n_rot       rotary dimensions at the tail
 *  @param pos0        position of row 0
 *  @param pos_stride  positions between rows (the ratio)
 *  @param n_ctx_orig  the context the RoPE settings were trained at
 *  @param freq_base   RoPE base frequency
 *  @param freq_scale  linear frequency scaling
 *  @param ext_factor  YaRN extrapolation mix
 *  @param attn_factor YaRN attention-temperature correction
 *  @param beta_fast   YaRN ramp start
 *  @param beta_slow   YaRN ramp end
 *  @return 1 on success, 0 on a bad shape or launch failure */
int pulsar_gpu_rope_tail_strided_tensor(
        pulsar_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          pos_stride,
        uint32_t          n_ctx_orig,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/** Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * pulsar_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
/** @param raw_cache the ring, raw_cap rows in the loaded family's ring format
 *  @param kv        f32 [head_dim] row to pack
 *  @param raw_cap   ring capacity, rows
 *  @param row       ring slot (< raw_cap)
 *  @param head_dim  head width
 *  @return 1 on success, 0 on a bad size or launch failure */
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
/** @param raw_cache the ring (or the bank pool in banked mode)
 *  @param kv        f32 [n_tokens][head_dim] rows to pack
 *  @param raw_cap   per-bank ring capacity, rows
 *  @param pos0      position of row 0 (classic mode)
 *  @param n_tokens  rows
 *  @param head_dim  head width
 *  @param positions int32 [n_tokens] positions, or NULL
 *  @param seq_id    int32 [n_tokens] bank ids, or NULL
 *  @param n_banks   banks in the pool (1 classic)
 *  @return 1 on success, 0 on a bad size or launch failure */
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

/** Same scatter, but the source rows are ALREADY WINDOW rows.  Prefer this
 * wherever the caller has already packed the batch: re-quantising a buffer that
 * has been round-tripped is the ~5%-misround pattern the norm_kv header warns
 * about, and a byte copy makes the ring agree with what attention read by
 * construction instead of by argument. */
/** @param raw_cache the ring (or the bank pool)
 *  @param packed    [n_tokens] WINDOW rows, already packed
 *  @param raw_cap   per-bank ring capacity, rows
 *  @param pos0      position of row 0 (classic mode)
 *  @param n_tokens  rows
 *  @param head_dim  head width
 *  @param positions int32 [n_tokens] positions, or NULL
 *  @param seq_id    int32 [n_tokens] bank ids, or NULL
 *  @param n_banks   banks in the pool (1 classic)
 *  @return 1 on success, 0 on a bad size or launch failure */
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

/** Add each position's within-group ape row to its score projection:
 * `sc[t] += ape[(pos0 + t) %% ratio]`, width = pulsar_compress_coff(ratio) *
 * head_dim.  The reference adds the ape at four different sites (both prefill
 * stashes, the batched pool and the decode add) but always as
 * `ape[position %% ratio]`, so folding it in once here is exactly equivalent and
 * keeps the compressor kernels from each re-deriving it.  V4.1 has no ape
 * (compressor_ape is false), so this is called only where the weights exist. */
/** @param sc         [n_tokens][width] f32 score rows, in place
 *  @param model_map  the model mapping
 *  @param model_size its size in bytes
 *  @param ape_offset byte offset of the ape table [ratio][width]
 *  @param ape_type   PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param width      coff * head_dim
 *  @param ratio      compress ratio
 *  @param pos0       position of row 0
 *  @param n_tokens   rows
 *  @return 1 on success, 0 on bad operands or launch failure */
int pulsar_gpu_csa2_comp_ape_add_tensor(
        pulsar_gpu_tensor       *sc,           /* [n_tokens][coff * head_dim] f32, in/out */
        const void              *model_map,    /* the model's mapped weights */
        uint64_t                 model_size,
        uint64_t                 ape_offset,   /* the ape table inside that map */
        uint32_t                 ape_type,     /* PULSAR_TENSOR_F32 or _BF16 */
        uint32_t                 width,
        uint32_t                 ratio,
        uint32_t                 pos0,
        uint32_t                 n_tokens);

/** CSA2 (L218) compressor, the reference's Compressor.forward as two entries.
 *
 * A kv source of ratio > 1 pools the `ratio` tokens of a group into ONE latent:
 * softmax over the group's score projections (fp32, the reference's order --
 * weights normalised first, then the weighted sum), rounded to bf16, RMS-normed
 * against attn_compressor_norm (fp32 math, the weighted result rounded to
 * bf16).  A source of ratio 1 is the projection rounded to bf16 then normed.
 * Both write the latent PRE-RoPE as fp32 rows holding bf16-exact values: the
 * index-key projection reads it unrotated, the attention row pack rotates it.
 *
 * The state lane (ratio > 1 only) holds the pending group's kv / score rows at
 * slot pos %% ratio, empty = kv 0 / score -inf; a complete group is consumed at
 * its emit and the next group's stores overwrite every slot before the next
 * emit reads them, so at any position that is a multiple of `ratio` the state
 * is canonically empty.
 *
 * prefill: kv/sc hold n_tokens rows starting at position pos0, which must be a
 * multiple of ratio (an unaligned continuation takes the per-position entry
 * until it aligns).  Emits n_tokens / ratio latent rows and leaves the trailing
 * n_tokens %% ratio rows in the state; at ratio 1 the state tensors may be NULL.
 * update: one position; stores its row and, when it completes a group, emits
 * that group's latent (sets *emitted).  At ratio 1 every position emits. */
/** @param latent      see the inline notes
 *  @param latent_b    see the inline notes
 *  @param kv          f32 [n_tokens][coff * head_dim] kv projections
 *  @param sc          f32 [n_tokens][coff * head_dim] score projections (ignored at ratio 1)
 *  @param state_kv    pending-group kv lane (NULL at ratio 1)
 *  @param state_score pending-group score lane (NULL at ratio 1)
 *  @param model_map   the model mapping
 *  @param model_size  its size in bytes
 *  @param norm_offset byte offset of attn_compressor_norm [head_dim]
 *  @param norm_type   PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param head_dim    head width
 *  @param ratio       compress ratio
 *  @param pos0        position of the first token (a multiple of ratio)
 *  @param n_tokens    tokens
 *  @param rms_eps     RMS epsilon
 *  @return 1 on success, 0 on bad operands or launch failure */
int pulsar_gpu_csa2_compressor_prefill_tensor(
        pulsar_gpu_tensor       *latent,       /* [n_tokens / ratio][head_dim] f32 out */
        void                    *latent_b,     /* the rows' bf16 plane (pulsar_gpu_bf16_act_slot), or NULL */
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,           /* ignored at ratio 1 */
        pulsar_gpu_tensor       *state_kv,     /* NULL at ratio 1 */
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                   rms_eps);

/** @brief One position of the CSA2 compressor (see above).
 *  @param latent      [1][head_dim] f32 out, written only when *emitted
 *  @param latent_b    its bf16 plane, or NULL
 *  @param kv_cur      f32 kv row of this position
 *  @param sc_cur      f32 score row of this position
 *  @param state_kv    pending-group kv lane
 *  @param state_score pending-group score lane
 *  @param model_map   the model mapping
 *  @param model_size  its size in bytes
 *  @param norm_offset byte offset of the compressor norm [head_dim]
 *  @param norm_type   PULSAR_TENSOR_F32 or PULSAR_TENSOR_BF16
 *  @param head_dim    head width
 *  @param ratio       compress ratio
 *  @param pos         the position
 *  @param rms_eps     RMS epsilon
 *  @param emitted     set to 1 when a latent row was emitted, else 0
 *  @return 1 on success, 0 on bad operands or launch failure */
int pulsar_gpu_csa2_compressor_update_tensor(
        pulsar_gpu_tensor       *latent,       /* [1][head_dim] f32 out, written only when *emitted */
        void                    *latent_b,     /* its bf16 plane, or NULL (written with it) */
        const pulsar_gpu_tensor *kv_cur,
        const pulsar_gpu_tensor *sc_cur,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        float                   rms_eps,
        int                    *emitted);

/** Store one position's kv / score projections into the state slot pos %% ratio
 * without pooling (the rewind path rebuilding a pending group from the verify
 * saves).  ratio > 1 only. */
/** @param kv_row      f32 kv row
 *  @param sc_row      f32 score row
 *  @param state_kv    pending-group kv lane
 *  @param state_score pending-group score lane
 *  @param head_dim    head width
 *  @param ratio       compress ratio (> 1)
 *  @param pos         the position (slot pos % ratio)
 *  @return 1 on success, 0 on bad operands or launch failure */
int pulsar_gpu_csa2_compressor_store_tensor(
        const pulsar_gpu_tensor *kv_row,
        const pulsar_gpu_tensor *sc_row,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos);

/** The overlapping compressor's (coff 2) end-of-group SHIFT, on its own: move
 * the current half (rows [ratio, coff*ratio)) down into the carry half and clear
 * what it left behind.  The emit does this after pooling; the L120 rewind replay
 * (pulsar_session::rewind) does it to rebuild the carry a resume needs, which is
 * the only other writer of the lane.  Refuses (0) on a non-overlapping ratio --
 * there is no carry half to shift into. */
/** @param state_kv    pending-group kv lane
 *  @param state_score pending-group score lane
 *  @param head_dim    head width
 *  @param ratio       compress ratio (coff 2 only)
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_gpu_csa2_compressor_shift_tensor(
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        uint32_t                head_dim,
        uint32_t                ratio);

/** As below, but the fp16 tier additionally emits the grouped E4M3 encoding of
 * batch_heads for the attn-output "a" projection.  *mx_out is set to 1 ONLY if
 * that tier actually ran and slots were supplied -- any other tier leaves it 0,
 * and the caller must not note() the activation cache in that case.  Note that
 * rope_tail rewrites the rope tail afterwards and must emit those blocks too;
 * both halves are required before the encoding is complete. */
/**
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv       the chunk's WINDOW rows, [n_tokens]
 *  @param n_tokens     rows
 *  @param window       raw rows a query sees
 *  @param n_head       heads
 *  @param head_dim     head width
 *  @param gact_data   grouped E4M3 slot (pulsar_gpu_mxfp8_gact_slot), or NULL
 *  @param gact_scale  its scale slab
 *  @param gact_kbp    its scale pitch
 *  @param gact_slab   bytes of one group's scale slab
 *  @param n_groups    output groups
 *  @param n_nope       head dims the epilogue encodes
 *  @param mx_out       set to 1 only when the emission ran
 *  @param positions    int32 [n_tokens] positions (fused rope), or NULL
 *  @param q_prep       fused rope descriptor, or NULL
 *  @param vis_left     image-span visibility, or NULL
 *  @param vis_right    image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_gpu_attention_prefill_raw_heads_mx_tensor(
        pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window,
        uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab,
        uint32_t n_groups, uint32_t n_nope, int *mx_out,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep,
        /* Image-span visibility (L216): int32 [n_tokens] DEVICE tensor slices of
         * get_image_visible()'s left/right counts, or NULL/NULL for a text
         * chunk.  Non-NULL gives an image query a FORWARD reach the causal
         * window cannot express; see the f16 entry for the exact range. */
        const pulsar_gpu_tensor *vis_left, const pulsar_gpu_tensor *vis_right);

/** @brief pulsar_gpu_attention_prefill_raw_heads_mx_tensor without the MX emission.
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv       the chunk's WINDOW rows
 *  @param n_tokens     rows
 *  @param window       raw rows a query sees
 *  @param n_head       heads
 *  @param head_dim     head width
 *  @param positions    positions, or NULL
 *  @param q_prep       fused rope descriptor, or NULL
 *  @param vis_left     image-span visibility, or NULL
 *  @param vis_right    image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal or failure */
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
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep,
        const pulsar_gpu_tensor *vis_left, const pulsar_gpu_tensor *vis_right);

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
 * changes which rows are visible, never how a visible row is folded.
 * vis_left/vis_right (L283): a prefill chunk past position 0 runs this launcher
 * too, and an image span it owns keeps the visibility the zero-prefix
 * launchers give it -- int32 [n_tokens] DEVICE slices of get_image_visible()'s
 * counts, or NULL/NULL for text and for every decode row (see
 * pulsar_gpu_attention_prefill_raw_heads_mx_tensor). */
/**
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv       the WINDOW ring (the bank pool in banked mode)
 *  @param n_tokens     rows
 *  @param pos0         position of row 0 (classic mode)
 *  @param n_raw        ring rows holding data (classic mode; 0 banked)
 *  @param raw_cap      per-bank ring capacity, rows
 *  @param raw_start    ring index of the oldest row (classic mode; 0 banked)
 *  @param window       raw rows a query sees
 *  @param n_head       heads
 *  @param head_dim     head width
 *  @param non_causal   see above
 *  @param positions    int32 [n_tokens] positions, or NULL
 *  @param seq_id       int32 [n_tokens] bank ids, or NULL
 *  @param comp_cap     per-bank compressed-row stride
 *  @param n_banks      banks (1 classic)
 *  @param q_prep       fused rope descriptor, or NULL
 *  @param vis_left     image-span visibility, or NULL
 *  @param vis_right    image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        const pulsar_gpu_q_prep *q_prep,
        const pulsar_gpu_tensor *vis_left,
        const pulsar_gpu_tensor *vis_right);

/** @brief Batched decode attention over the raw ring and the visible compressed prefix (banked or classic).
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv         the WINDOW ring (the bank pool when banked)
 *  @param comp_kv        MAIN rows (the pool, or bank 0's base when banked)
 *  @param n_tokens       rows
 *  @param pos0           position of row 0 (classic mode)
 *  @param n_raw          ring rows holding data (classic mode)
 *  @param raw_cap        per-bank ring capacity, rows
 *  @param raw_start      ring index of the oldest row (classic mode)
 *  @param n_comp         compressed rows visible (the cross-bank superset clamp when banked)
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads
 *  @param head_dim       head width
 *  @param non_causal     0 = causal raw visibility, nonzero = the whole ring
 *  @param positions      int32 [n_tokens] positions, or NULL (classic)
 *  @param seq_id         int32 [n_tokens] bank ids, or NULL
 *  @param comp_bank_ptrs device table of per-bank MAIN-row bases, or NULL = comp_kv + seq_id * comp_cap
 *  @param comp_cap       per-bank compressed-row stride
 *  @param n_banks        banks (1 classic)
 *  @param q_prep         fused rope descriptor, or NULL
 *  @return 1 on success, 0 on refusal (said) or failure */
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
/** L260: the same launch with the attn-output 'a' GEMM's grouped E4M3 emitted by
 *  the attention epilogue for the rows past the step's split-K decode rows
 *  (the arms below emit nothing; see pulsar_gpu_attention_f16_indexed). */
/**
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv         the WINDOW ring (the bank pool when banked)
 *  @param comp_kv        MAIN rows (the pool, or bank 0's base when banked)
 *  @param n_tokens       rows
 *  @param pos0           position of row 0 (classic mode)
 *  @param n_raw          ring rows holding data (classic mode)
 *  @param raw_cap        per-bank ring capacity, rows
 *  @param raw_start      ring index of the oldest row (classic mode)
 *  @param n_comp         compressed rows visible (the cross-bank superset clamp when banked)
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads
 *  @param head_dim       head width
 *  @param non_causal     0 = causal raw visibility, nonzero = the whole ring
 *  @param positions      int32 [n_tokens] positions, or NULL (classic)
 *  @param seq_id         int32 [n_tokens] bank ids, or NULL
 *  @param comp_bank_ptrs device table of per-bank MAIN-row bases, or NULL = comp_kv + seq_id * comp_cap
 *  @param comp_cap       per-bank compressed-row stride
 *  @param n_banks        banks (1 classic)
 *  @param q_prep         fused rope descriptor, or NULL
 *  @param vis_left       image-span visibility, or NULL (text)
 *  @param vis_right      image-span visibility, or NULL
 *  @param gact_data      grouped E4M3 slot, or NULL = no emission
 *  @param gact_scale     its scale slab
 *  @param gact_kbp       its scale pitch
 *  @param gact_slab      bytes of one group's scale slab
 *  @param n_groups       output groups
 *  @param n_nope         head dims the epilogue encodes
 *  @return 1 on success, 0 on refusal (said) or failure */
int pulsar_gpu_attention_decode_mixed_batch_heads_mx_tensor(
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
        const pulsar_gpu_q_prep *q_prep,
        /* L283: the chunk's image-span visibility, as for the raw batch launcher above (NULL/NULL: text) */
        const pulsar_gpu_tensor *vis_left,
        const pulsar_gpu_tensor *vis_right,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab,
        uint32_t n_groups, uint32_t n_nope);

/** @brief Batched attention over the raw ring and a top-k selection of compressed rows.
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv         the WINDOW ring (the bank pool when banked)
 *  @param comp_kv        MAIN rows (the pool, or bank 0's base when banked)
 *  @param topk           int32 [n_tokens][top_k] selected compressed rows
 *  @param n_tokens       rows
 *  @param pos0           position of row 0 (classic mode)
 *  @param n_raw          ring rows holding data (classic mode)
 *  @param raw_cap        per-bank ring capacity, rows
 *  @param raw_start      ring index of the oldest row (classic mode)
 *  @param n_comp         compressed rows visible (the cross-bank superset clamp when banked)
 *  @param top_k          entries per topk row
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads
 *  @param head_dim       head width
 *  @param positions      int32 [n_tokens] positions, or NULL (classic)
 *  @param seq_id         int32 [n_tokens] bank ids, or NULL
 *  @param comp_bank_ptrs device table of per-bank MAIN-row bases, or NULL = comp_kv + seq_id * comp_cap
 *  @param comp_cap       per-bank compressed-row stride
 *  @param n_banks        banks (1 classic)
 *  @param q_prep         fused rope descriptor, or NULL
 *  @param vis_left       image-span visibility, or NULL (text)
 *  @param vis_right      image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        const pulsar_gpu_q_prep *q_prep,
        const pulsar_gpu_tensor *vis_left,
        const pulsar_gpu_tensor *vis_right);
/** L260: the same launch emitting the grouped E4M3 for its rows past the split-K
 *  decode rows, at batch row gact_tok0 + t of a gact_ntok-row slab (a span of a
 *  wider step passes its first row). */
/**
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv         the WINDOW ring (the bank pool when banked)
 *  @param comp_kv        MAIN rows (the pool, or bank 0's base when banked)
 *  @param topk           int32 [n_tokens][top_k] selected compressed rows
 *  @param n_tokens       rows
 *  @param pos0           position of row 0 (classic mode)
 *  @param n_raw          ring rows holding data (classic mode)
 *  @param raw_cap        per-bank ring capacity, rows
 *  @param raw_start      ring index of the oldest row (classic mode)
 *  @param n_comp         compressed rows visible (the cross-bank superset clamp when banked)
 *  @param top_k          entries per topk row
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads
 *  @param head_dim       head width
 *  @param positions      int32 [n_tokens] positions, or NULL (classic)
 *  @param seq_id         int32 [n_tokens] bank ids, or NULL
 *  @param comp_bank_ptrs device table of per-bank MAIN-row bases, or NULL = comp_kv + seq_id * comp_cap
 *  @param comp_cap       per-bank compressed-row stride
 *  @param n_banks        banks (1 classic)
 *  @param q_prep         fused rope descriptor, or NULL
 *  @param vis_left       image-span visibility, or NULL (text)
 *  @param vis_right      image-span visibility, or NULL
 *  @param gact_data      grouped E4M3 slot, or NULL = no emission
 *  @param gact_scale     its scale slab
 *  @param gact_kbp       its scale pitch
 *  @param gact_slab      bytes of one group's scale slab
 *  @param n_groups       output groups
 *  @param n_nope         head dims the epilogue encodes
 *  @param gact_tok0      this launch's first row in the slot
 *  @param gact_ntok      the slot's row count
 *  @return 1 on success, 0 on refusal (said) or failure */
int pulsar_gpu_attention_indexed_mixed_batch_heads_mx_tensor(
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
        const pulsar_gpu_q_prep *q_prep,
        const pulsar_gpu_tensor *vis_left,
        const pulsar_gpu_tensor *vis_right,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab,
        uint32_t n_groups, uint32_t n_nope, uint32_t gact_tok0, uint32_t gact_ntok);

/** @brief Prefill attention of a chunk from position 0: the causal raw window plus the visible compressed prefix.
 *  @param heads        stored attention output, [n_tokens][n_head][head_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param sinks_offset byte offset of the f32 [n_head] attention-sink logits
 *  @param q            Q, [n_tokens][n_head][head_dim]
 *  @param raw_kv       the chunk's WINDOW rows, [n_tokens]
 *  @param comp_kv      MAIN rows, [n_comp]
 *  @param gact_data      grouped E4M3 slot, or NULL = no emission
 *  @param gact_scale     its scale slab
 *  @param gact_kbp       its scale pitch
 *  @param gact_slab      bytes of one group's scale slab
 *  @param n_groups       output groups
 *  @param n_nope         head dims the epilogue encodes
 *  @param mx_out       set to 1 only when the emission ran
 *  @param n_tokens     rows
 *  @param n_comp       compressed rows
 *  @param window         raw rows a query sees
 *  @param ratio          compression ratio
 *  @param n_head         heads
 *  @param head_dim       head width
 *  @param q_prep       fused rope descriptor, or NULL
 *  @param vis_left       image-span visibility, or NULL (text)
 *  @param vis_right      image-span visibility, or NULL
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        const pulsar_gpu_q_prep *q_prep,
        const pulsar_gpu_tensor *vis_left,
        const pulsar_gpu_tensor *vis_right);


/** @brief Attention output stage "a": the grouped low-rank projection low[g] = heads[g] . out_a[g]^T per group.
 *  @param low          f32 [n_tokens][n_groups * rank] out
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param out_a_offset byte offset of the (rank's slice of the) out_a weight
 *  @param group_dim    input width of one group (heads per group x head_dim)
 *  @param rank         output width of one group
 *  @param n_groups     groups (on a TP rank, the owned ones)
 *  @param heads        the attention output
 *  @param n_tokens     rows
 *  @return 1 on success, 0 on a bad shape or failure */
int pulsar_gpu_attention_output_a_tensor(
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const pulsar_gpu_tensor *heads,
        uint32_t                n_tokens);
/* Stage "b" over the FULL low_dim; on a TP rank the caller gathers `low`
 * across the group between the two stages (slice 4g). */
/** @param out          f32 [n_tokens][out_dim]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param out_b_offset byte offset of the out_b weight
 *  @param low_dim      input width (every group's rank, gathered)
 *  @param out_dim      output width
 *  @param low          stage "a"'s output (its MXFP8 encoding is emitted here)
 *  @param n_tokens     rows
 *  @return 1 on success, 0 on a bad shape or failure */
int pulsar_gpu_attention_output_b_tensor(
        pulsar_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_b_offset,
        uint64_t                low_dim,
        uint64_t                out_dim,
        pulsar_gpu_tensor       *low,
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
/** @param out      f32 [n] out = silu(g) * u * weight (unwritten when skip_f32)
 *  @param gate     gate activations, f32 or f16 [n] (same element type as up)
 *  @param up       up activations [n]
 *  @param n        elements (rows * mid_dim; a multiple of 32 when emitting)
 *  @param clamp    > 1e-6: clamp g above at clamp and u to [-clamp, clamp]; else no clamp
 *  @param weight   scale on every element
 *  @param out_q    E4M3 slot, or NULL
 *  @param out_sf   its scale slab
 *  @param out_kbp  its scale pitch
 *  @param mid_dim  row width (a multiple of 32 when emitting)
 *  @param skip_f32 see above
 *  @return 1 on success, 0 on refusal (said) or launch failure */
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

/** @brief out = a + b over n f32 elements.
 *  @param out f32 [n]
 *  @param a   f32 [n]
 *  @param b   f32 [n]
 *  @param n   elements
 *  @return 1 on success, 0 on a bad size or launch failure */
int pulsar_gpu_add_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *a,
        const pulsar_gpu_tensor *b,
        uint32_t                n);

/** @brief Directional steering: per row, x -= scale * (x . d) * d with d = directions[layer].
 *  @param x          f32 [rows][width], in place
 *  @param directions f32 [n_layers][width] steering directions
 *  @param layer      the layer's direction row
 *  @param width      row width
 *  @param rows       rows
 *  @param scale      steering scale (0 refuses)
 *  @return 1 on success, 0 on a bad argument or launch failure */
int pulsar_gpu_directional_steering_project_tensor(
        pulsar_gpu_tensor       *x,
        const pulsar_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/** DeepSeek's router binding (L295 F2): resolves the layer's correction bias, image-slot bias (bias_vl) and hash
 *  table (tid2eid) from the model map and routes n_tokens rows of logits through the shared router
 *  (src/cuda/pulsar_cuda_router.h) with DeepSeek's declared cfg.  Nonzero on success. */
/** @param selected            int32 [n_tokens][n_expert_used] chosen experts
 *  @param weights             f32 [n_tokens][n_expert_used] their weights
 *  @param probs               f32 [n_tokens][n_expert] scores out, or NULL
 *  @param model_map           the model mapping
 *  @param model_size          its size in bytes
 *  @param bias_offset         byte offset of the f32 [n_expert] correction bias
 *  @param has_bias            the layer has a correction bias (offset 0 is a real offset)
 *  @param tid2eid_offset      byte offset of the hash-routing table
 *  @param tid2eid_rows        its rows (the vocab size); 0 = no table (top-k routing)
 *  @param logits              f32 [n_tokens][n_expert] router logits
 *  @param tokens              int32 [n_tokens] token ids (required by the hash arm)
 *  @param n_expert            routed experts
 *  @param n_expert_used       experts per token
 *  @param expert_weight_scale route scale on the weights
 *  @param n_tokens            rows
 *  @param vl_bias_offset      byte offset of the image-slot bias (bias_vl)
 *  @param n_vocab             vocabulary size (ids at or above it are image slots)
 *  @param has_vl_bias         the layer has bias_vl
 *  @return 1 on success, 0 on a bad shape or failure */
int pulsar_gpu_router_select_batch_tensor(
        pulsar_gpu_tensor       *selected,
        pulsar_gpu_tensor       *weights,
        pulsar_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        bool                    has_bias,
        uint64_t                tid2eid_offset,
        uint32_t                tid2eid_rows,
        const pulsar_gpu_tensor *logits,
        const pulsar_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens,
        uint64_t                vl_bias_offset,
        uint32_t                n_vocab,
        bool                    has_vl_bias);


/** @brief The routed-expert FFN for n_tokens rows: each row's selected experts' gate/up/SwiGLU/down, weighted and
 *  summed into out.  Every tensor is f32 (the lane refuses a narrowed one).
 *  @param out               f32 [n_tokens][out_dim]
 *  @param up                f32 scratch
 *  @param mid               f32 scratch
 *  @param experts           f32 per-slot down scratch (the definition names it `down`)
 *  @param model_map         the model mapping
 *  @param model_size        its size in bytes
 *  @param gate_offset       byte offset of the gate stack
 *  @param up_offset         byte offset of the up stack
 *  @param down_offset       byte offset of the down stack
 *  @param gate_type         PULSAR_TENSOR_* layout of gate/up
 *  @param down_type         PULSAR_TENSOR_* layout of down
 *  @param gate_expert_bytes bytes per expert in the gate/up stacks
 *  @param gate_row_bytes    bytes per row of a gate/up expert
 *  @param down_expert_bytes bytes per expert in the down stack
 *  @param down_row_bytes    bytes per row of a down expert
 *  @param expert_in_dim     expert input width
 *  @param expert_mid_dim    expert intermediate width (the half width under expert TP)
 *  @param out_dim           output width
 *  @param selected          int32 [n_tokens][n_expert] chosen experts
 *  @param weights           f32 [n_tokens][n_expert] their weights
 *  @param n_total_expert    experts in the stack
 *  @param n_expert          experts per token
 *  @param clamp             SwiGLU clamp (pulsar_gpu_swiglu_mx_tensor)
 *  @param x                 f32 [n_tokens][expert_in_dim] input
 *  @param layer_index       the layer (diagnostics, the non-finite flag)
 *  @param n_tokens          rows
 *  @param expert_split      nonzero: the stacks are this rank's expert halves (cutlass_mxfp4 or EXL3 only)
 *  @return 1 on success, 0 on refusal (said) or failure */
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
        uint32_t                n_tokens,
        uint32_t                expert_split);


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
/** @param down_out        f32 [n_tokens * n_expert][out_dim] pre-weighted per-slot results
 *  @param selected        device int32 [n_tokens][n_expert]
 *  @param rweights        device f32 [n_tokens][n_expert]
 *  @param gate_tab        device table of per-expert gate bytes
 *  @param up_tab          device table of per-expert up bytes
 *  @param down_tab        device table of per-expert down bytes
 *  @param gate_stride     bytes per gate/up expert
 *  @param gate_data_bytes bytes of a gate/up expert's data blob (scales follow)
 *  @param down_stride     bytes per down expert
 *  @param down_data_bytes bytes of a down expert's data blob
 *  @param clamp           SwiGLU clamp
 *  @param n_tokens        rows (2..4)
 *  @param n_expert        experts per token
 *  @param n_total_expert  experts in the stack
 *  @param in_dim          input width (a multiple of 256)
 *  @param mid_dim         intermediate width (a multiple of 128)
 *  @param out_dim         output width (a multiple of 8)
 *  @param act_q           the activation's E4M3 plane
 *  @param act_sf          its swizzled scales
 *  @param act_kbp         its scale pitch
 *  @return 0 on success, nonzero on refusal (CUTLASS-side convention, inverted from pulsar_gpu_*) */
int pulsar_cutlass_expert_ffn_gemv_small(
        float          *down_out,
        const int32_t  *selected,
        const float    *rweights,
        /* PLAN 94 phase 1: per-expert bytes, one device array per projection. */
        const uint8_t *const *gate_tab,
        const uint8_t *const *up_tab,
        const uint8_t *const *down_tab,
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
/** @param padded_total   gathered rows (a multiple of 128)
 *  @param n_total_expert experts in the stack
 *  @param in_dim         input width
 *  @param mid_dim        intermediate width
 *  @param out_dim        output width
 *  @return scratch bytes pulsar_cutlass_grouped_moe needs */
size_t pulsar_cutlass_grouped_moe_scratch_bytes(
        int padded_total, int n_total_expert, int in_dim, int mid_dim, int out_dim);
/** @param ffn_out         f32 [padded_total][out_dim] pre-weighted results
 *  @param x_gathered      f32 [padded_total][in_dim] gathered input (read only when act_q is NULL)
 *  @param w_gathered      f32 [padded_total] routing weight per gathered row
 *  @param gate_tab        device table of per-expert gate bytes
 *  @param up_tab          device table of per-expert up bytes
 *  @param down_tab        device table of per-expert down bytes
 *  @param gate_data_bytes bytes of a gate/up expert's data blob
 *  @param down_data_bytes bytes of a down expert's data blob
 *  @param clamp           SwiGLU clamp
 *  @param n_total_expert  experts in the stack
 *  @param in_dim          input width
 *  @param mid_dim         intermediate width
 *  @param out_dim         output width
 *  @param counts          device uint32 [n_total_expert] rows per expert
 *  @param padded_offsets  device uint32 [n_total_expert] 128-padded first row per expert
 *  @param padded_total    host upper bound on gathered rows (a multiple of 128)
 *  @param scratch         device scratch
 *  @param scratch_bytes   its size (>= pulsar_cutlass_grouped_moe_scratch_bytes)
 *  @param act_q           see below
 *  @param act_sf          see below
 *  @param act_kbp         the activation's scale pitch
 *  @param row_src_tok     device int32 [padded_total] source token per row, < 0 = padding
 *  @return 0 on success, nonzero on failure */
int pulsar_cutlass_grouped_moe(
        float          *ffn_out,
        const float    *x_gathered,
        const float    *w_gathered,
        /* PLAN 94 phase 1: one device array per projection holding expert e's
         * bytes (scale factors at +data_bytes).  The ADDRESS is data now, not
         * `base + e*stride`, so a slot cache can move an expert without any
         * consumer knowing -- see mxfp4_expert_table(). */
        const uint8_t *const *gate_tab,
        const uint8_t *const *up_tab,
        const uint8_t *const *down_tab,
        uint64_t        gate_data_bytes,
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
/** @param T       rows
 *  @param in_dim  input width
 *  @param out_dim output width
 *  @return scratch bytes of the single-projection GEMM */
size_t pulsar_cutlass_proj_scratch_bytes(int T, int in_dim, int out_dim);

/** Grouped single-projection W4A8 GEMM for MIXED layers -- one device-built ptr-array grouped GEMM
 * over 128-padded gathered activations: out[padded_total,out_dim] = x_gathered . W^T for every
 * active expert (W_base+e*W_stride data, +W_data_bytes swizzled SFB). No host readback, no per-expert
 * sync; bit-identical to the per-expert single-proj path (same pack + gather order + GEMM). Padding
 * rows must be pre-zeroed. Caller sizes scratch once via pulsar_cutlass_grouped_proj_scratch_bytes().
 * reuse_packed_a: skip the E4M3 activation pack and consume the encoding a PREVIOUS call left in
 * this same scratch -- valid only when x_gathered, padded_total, in_dim and scratch are identical
 * to that call's (the gate/up pair over one gathered activation). Ends the case-A double-encode. */
/** @param padded_total   gathered rows (a multiple of 128)
 *  @param n_total_expert experts in the stack
 *  @param in_dim         input width
 *  @param out_dim        output width
 *  @return scratch bytes pulsar_cutlass_grouped_proj needs */
size_t pulsar_cutlass_grouped_proj_scratch_bytes(int padded_total, int n_total_expert, int in_dim, int out_dim);
/** @param out             f32 [padded_total][out_dim]
 *  @param x_gathered      f32 [padded_total][in_dim] (not read when act_q is supplied)
 *  @param W_tab           device table of per-expert weight bytes
 *  @param W_data_bytes    bytes of an expert's data blob (swizzled SFB follows)
 *  @param n_total_expert  experts in the stack
 *  @param in_dim          input width
 *  @param out_dim         output width
 *  @param counts          device rows per expert
 *  @param padded_offsets  device 128-padded first row per expert
 *  @param padded_total    host upper bound on gathered rows
 *  @param scratch         device scratch
 *  @param scratch_bytes   its size
 *  @param reuse_packed_a  see above
 *  @param act_q           producer E4M3 plane, or NULL
 *  @param act_sf          its swizzled scales, or NULL
 *  @param act_kbp         its scale pitch
 *  @param row_src_tok     device source token per row, < 0 = padding, or NULL
 *  @return 0 on success, nonzero on failure */
int pulsar_cutlass_grouped_proj(float *out, const float *x_gathered,
        const uint8_t *const *W_tab, uint64_t W_data_bytes,   /* per-expert bytes (PLAN 94 phase 1) */
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
 * over all (token,expert) slots, no per-expert loop/host sync. mid/down_out are pair-layout f32.
 *
 * L219: when emit_q is non-NULL the SwiGLU epilogue writes the mid E4M3 + E8M0
 * into that slot (rows = (token, slot) pairs, emit_sf's swizzle pitch is
 * emit_kbp) and `mid` is NOT written; mid_dim must then be a multiple of 32.
 * Pass NULL/NULL/0 for the historical f32 output. */
/** @param mid             f32 [n_tokens * n_expert][mid_dim] out (unwritten when emit_q is set)
 *  @param selected        device int32 [n_tokens][n_expert]
 *  @param rweights        device f32 [n_tokens][n_expert]
 *  @param gate_tab        device per-expert gate bytes
 *  @param up_tab          device per-expert up bytes
 *  @param gate_stride     bytes per expert
 *  @param gate_data_bytes bytes of an expert's data blob
 *  @param clamp           SwiGLU clamp
 *  @param n_tokens        rows (<= 4)
 *  @param n_expert        experts per token
 *  @param n_total_expert  experts in the stack
 *  @param in_dim          input width
 *  @param mid_dim         intermediate width
 *  @param act_q           the activation's E4M3 plane
 *  @param act_sf          its swizzled scales
 *  @param act_kbp         its scale pitch
 *  @param emit_q          E4M3 slot for mid, or NULL
 *  @param emit_sf         its scales
 *  @param emit_kbp        its scale pitch
 *  @return 0 on success, nonzero on failure */
int pulsar_cutlass_gemv_gateup(float *mid, const int32_t *selected, const float *rweights,
        const uint8_t *const *gate_tab, const uint8_t *const *up_tab,   /* PLAN 94 phase 1 */
        uint64_t gate_stride, uint64_t gate_data_bytes,
        float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim,
    const void *act_q, const void *act_sf, int act_kbp,
    void *emit_q, void *emit_sf, int emit_kbp);
/** L158 inc 5: mid arrives as the MoE stage's E4M3 encoding (mid_q/mid_sf in the
 * VEC32 swizzle at pitch mid_kbp, rows = (token, slot) pairs); no f32 mid. */
/** @param down_out        f32 [n_tokens * n_expert][out_dim] per-slot results
 *  @param selected        device int32 [n_tokens][n_expert]
 *  @param down_tab        device per-expert down bytes
 *  @param down_stride     bytes per expert
 *  @param down_data_bytes bytes of an expert's data blob
 *  @param n_tokens        rows
 *  @param n_expert        experts per token
 *  @param n_total_expert  experts in the stack
 *  @param mid_dim         input width
 *  @param out_dim         output width
 *  @param mid_q           mid's E4M3 plane
 *  @param mid_sf          its swizzled scales
 *  @param mid_kbp         its scale pitch
 *  @return 0 on success, nonzero on failure */
int pulsar_cutlass_gemv_down(float *down_out, const int32_t *selected,
        const uint8_t *const *down_tab,   /* PLAN 94 phase 1 */
        uint64_t down_stride, uint64_t down_data_bytes,
        int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim,
        const void *mid_q, const void *mid_sf, int mid_kbp);


/** =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */


/** @brief Collapse the HC streams: out[t] = sum_s weights[t][s] * residual_hc[t][s] (rows from out's size).
 *  @param out         f32 [n_tokens][n_embd]
 *  @param residual_hc HC residual [n_tokens][n_hc][n_embd]
 *  @param weights     f32 [n_tokens][n_hc]
 *  @param n_embd      embedding width
 *  @param n_hc        HC streams
 *  @return 1 on success, 0 on a bad size or launch failure */
int pulsar_gpu_hc_weighted_sum_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

/** The HC HEAD MIX (0731): the two steps that produce the coefficients the
 * collapse above consumes.  V4.1 has neither -- it collapses with the pre its
 * last FFN handed on -- and no profile but 0731's binds the weights, so these
 * are unreachable for V4.1.  plans/96-two-profiles-one-engine.md s17. */
/** @param out           f32 [out_dim] mix
 *  @param model_map     the model mapping
 *  @param model_size    its size in bytes
 *  @param weight_offset byte offset of the mix weight [out_dim][in_dim]
 *  @param in_dim        input width
 *  @param out_dim       output width
 *  @param x             one HC residual row [in_dim] (HC elements)
 *  @param eps           RMS epsilon
 *  @param w_type        PULSAR_TENSOR_BF16 or PULSAR_TENSOR_F32
 *  @return 1 on success, 0 on a bad shape or weight type */
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
/** @brief out = sigmoid(pre * scale + base[h]) + eps per stream (rows from out's size).
 *  @param out          f32 [n_tokens][n_hc]
 *  @param pre          f32 [n_tokens][n_hc]
 *  @param model_map    the model mapping
 *  @param model_size   its size in bytes
 *  @param scale_offset byte offset of the one f32 scale
 *  @param base_offset  byte offset of the f32 [n_hc] base
 *  @param n_hc         HC streams
 *  @param eps          added after the sigmoid
 *  @return 1 on success, 0 on a bad size or launch failure */
int pulsar_gpu_output_hc_weights_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);


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
 * @param split               per-stream split of the mix: this sublayer's pre / post / comb
 * @param pre_carry           [n_rows][n_hc] f32: IN the pre handed to this sublayer (the
 *                            previous sublayer's, or the identity before layer 0), OUT this
 *                            sublayer's pre for the next collapse -- single-pass mHC (L218)
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
 * @return nonzero on success, 0 on a bad shape or a failed launch (every caller tests `!= 0`).
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
        pulsar_gpu_tensor       *pre_carry,
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
        /* Storage of norm_w (attn_norm / ffn_norm): 1 = bf16, 0 = f32. */
        int                     norm_w_bf16);


/** make_identity_pre_mix (L218): rows [0, n_rows) of `pre` become the one-hot
 * (1, 0, ...) that layer 0's attention collapses with.  Called where the HC
 * residual stream is born (the embedding expansion), once per sweep. */
/** @param pre    f32 [n_rows][n_hc]
 *  @param n_rows rows
 *  @param n_hc   HC streams
 *  @return 1 on success, 0 on a bad size or launch failure */
int pulsar_gpu_hc_pre_identity_tensor(pulsar_gpu_tensor *pre, uint32_t n_rows, uint32_t n_hc);
/** @brief Expand a sublayer's output back into the HC streams with the split's post / comb coefficients.
 *  @param out_hc      HC residual out [n_tokens][n_hc][n_embd] (rows from its size)
 *  @param block_out   f32 [n_tokens][n_embd] sublayer output
 *  @param residual_hc the HC residual the sublayer collapsed
 *  @param split       f32 per-row split [pre | post | comb] (2 n_hc + n_hc^2 per row)
 *  @param n_embd      embedding width
 *  @param n_hc        HC streams
 *  @return 1 on success, 0 on a bad argument or launch failure */
int pulsar_gpu_hc_expand_split_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *block_out,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/** @brief pulsar_gpu_hc_expand_split_tensor with block_out + block_add as the sublayer output.
 *  @param out_hc      HC residual out
 *  @param block_out   f32 [n_tokens][n_embd]
 *  @param block_add   f32 [n_tokens][n_embd] added to block_out
 *  @param residual_hc the HC residual the sublayer collapsed
 *  @param split       f32 per-row split [pre | post | comb]
 *  @param n_embd      embedding width
 *  @param n_hc        HC streams
 *  @return 1 on success, 0 on a bad argument or launch failure */
int pulsar_gpu_hc_expand_add_split_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *block_out,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* ---- L302: THE sampler, on the device (pulsar_cuda_sampler.cu has the rule) ------------------------------------ */

/** Rows one sampler launch takes: a verify's PULSAR_SPEC_ROW_BUDGET rows plus a full accept's carry row. */
#define PULSAR_GPU_SAMPLE_ROWS_MAX 33u
/** Global scratch slots for rows whose candidate set outgrows the row kernel's shared store. */
#define PULSAR_GPU_SAMPLE_SLOTS 4u
/** A row with no mask (pulsar_gpu_sample_req::mask_row). */
#define PULSAR_GPU_SAMPLE_NO_MASK 0xffffffffu
/** pulsar_gpu_sample_req::flags: draw this row's argmax whatever its temperature (L296 toolsample positions). */
#define PULSAR_GPU_SAMPLE_GREEDY 1u

/** A row's status: OK, or why no token was drawn (the host sampler's three refusals). */
enum {
    PULSAR_GPU_SAMPLE_OK = 0,
    PULSAR_GPU_SAMPLE_REFUSED_KNOB = 1,    ///< a sampling knob is not finite
    PULSAR_GPU_SAMPLE_REFUSED_EMPTY = 2,   ///< no finite logit (after the mask)
    PULSAR_GPU_SAMPLE_REFUSED_MASS = 3,    ///< the candidate mass is not a positive finite number
};

/** One row's draw: the request's knobs as it carries them (the sampler clamps them as the host's always did), the
 *  key (pulsar_sample_key at the token's index), the flags, and the row of the token mask it reads. */
typedef struct pulsar_gpu_sample_req {
    /** Softmax temperature; <= 0 = greedy (the argmax).  Not finite: the row refuses (REFUSED_KNOB) unless greedy. */
    float temperature;
    /** Nucleus mass; outside (0, 1] = 1 (off). */
    float top_p;
    /** Min-p floor relative to the best token's probability; < 0 = 0 (off). */
    float min_p;
    /** Top-k cut; <= 0 = off, clamped to 1024. */
    int32_t top_k;
    /** The draw's random key: pulsar_sample_key at the token's index in the request's stream. */
    uint64_t key;
    /** PULSAR_GPU_SAMPLE_* flag bits (PULSAR_GPU_SAMPLE_GREEDY). */
    uint32_t flags;
    uint32_t mask_row;   ///< row of the launch's mask (bit i of word i/32 set = token i allowed), or PULSAR_GPU_SAMPLE_NO_MASK
    int32_t probe;       ///< a token whose emitted probability the row reports (a verify row's draft; L301), or -1
    uint32_t row;        ///< the logits row this draw reads (row `row` of the launch's rows)
} pulsar_gpu_sample_req;

/** What the sampler writes per row. */
typedef struct pulsar_gpu_sample_out {
    int32_t token;     ///< -1 = refused
    uint32_t status;   ///< PULSAR_GPU_SAMPLE_*
    float probe_p;     ///< the probe's probability in the emitted distribution (0 when not emitted; greedy: 1 for the argmax)
    /** Padding to 16 bytes; carries nothing. */
    uint32_t pad_;
} pulsar_gpu_sample_out;

/** Bytes of the scratch tensor a session gives every sampler launch over rows of n_vocab. */
/** @param n_vocab logits row width
 *  @return scratch bytes */
uint64_t pulsar_gpu_sample_scratch_bytes(uint32_t n_vocab);

/** n_rows draws: reqs[k] reads the row at logits + logits_offset_bytes + reqs[k].row * row_stride floats, n_vocab
 *  wide, and out receives, from out_offset_bytes, n_rows pulsar_gpu_sample_out in reqs order.  `logits_rows` bounds
 *  every reqs[k].row.  `mask` (NULL = none) is
 *  uint32 [rows][mask_words]: a cleared bit makes that token's logit -inf before every transform -- L303's
 *  structured-output hook, L272's tool-name mask; no mask = the unmasked bytes exactly.  `dist` (NULL in production)
 *  receives the emitted distribution per row as uint32 [n, ids[dist_cap], prob bits[dist_cap]] for the equality
 *  gate.  `id_map` (NULL = identity) is int32 [n_vocab]: row index i is token id_map[i] -- a drafter's draft
 *  vocabulary (the candidates still order by row index on ties; the race hashes, the probe and the token are the
 *  mapped ids).  Enqueued on the caller's stream; 0 (said) on a bad launch. */
/** @param logits              f32 logits rows
 *  @param logits_offset_bytes byte offset of row 0
 *  @param row_stride          floats between rows
 *  @param n_vocab             row width
 *  @param logits_rows         rows available (bounds every reqs[k].row)
 *  @param n_rows              draws (<= PULSAR_GPU_SAMPLE_ROWS_MAX)
 *  @param reqs                host [n_rows] requests
 *  @param mask                uint32 [rows][mask_words] token masks, or NULL
 *  @param mask_words          words per mask row
 *  @param id_map              int32 [n_vocab] row index -> token id, or NULL = identity
 *  @param scratch             pulsar_gpu_sample_scratch_bytes of scratch
 *  @param out                 receives n_rows pulsar_gpu_sample_out
 *  @param out_offset_bytes    byte offset into out
 *  @param dist                emitted-distribution dump for the equality gate, or NULL
 *  @param dist_cap            entries per dist row
 *  @return 1 on success (enqueued), 0 on a bad launch (said) */
int pulsar_gpu_sample_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                           uint32_t n_vocab, uint32_t logits_rows, uint32_t n_rows, const pulsar_gpu_sample_req *reqs,
                           const pulsar_gpu_tensor *mask, uint32_t mask_words, const pulsar_gpu_tensor *id_map,
                           pulsar_gpu_tensor *scratch, pulsar_gpu_tensor *out, uint64_t out_offset_bytes,
                           pulsar_gpu_tensor *dist, uint32_t dist_cap);

/** A verify run for the device walk: the sampler rows [row0, row0 + k] of one bank (the draw after the base and
 *  after each of its k drafts) and the drafts. */
typedef struct pulsar_gpu_spec_run {
    /** The run's first sampler row (the draw after the base token), an index into the launch's out rows. */
    uint32_t row0;
    /** Drafts in the run (<= 16); the run reads rows [row0, row0 + k]. */
    uint32_t k;
    uint32_t greedy_rows;   ///< bit j: row j drew greedy (its calibration label is the acceptance itself)
    /** The draft tokens, pend[0 .. k). */
    int32_t pend[16];
} pulsar_gpu_spec_run;
/** The walk's verdict for a run: drafts kept, the carry token (-1 when a row refused), the refusing row's status,
 *  and L301's calibration labels for the drafts it reached: label[j] = 1/0 (greedy row: argmax == draft) or the
 *  target's emitted probability of draft j (sampled row), hit[j] = the row drew draft j. */
typedef struct pulsar_gpu_spec_walk_out {
    /** Drafts kept (0 .. k). */
    uint32_t commit;
    /** The token the walk's stopping row drew (a rejection's draw, or a full accept's row k); -1 when a row refused. */
    int32_t carry;
    /** PULSAR_GPU_SAMPLE_OK, or the refusing row's PULSAR_GPU_SAMPLE_REFUSED_*. */
    uint32_t status;
    /** Entries of label / hit written: the drafts the walk reached. */
    uint32_t n_label;
    /** Per reached draft j: 1/0 (greedy row: argmax == draft) or the target's emitted probability of draft j. */
    float label[16];
    /** Per reached draft j: 1 = the row drew draft j. */
    uint8_t hit[16];
} pulsar_gpu_spec_walk_out;
/** The speculative walk over a sampler launch's rows (pulsar_gpu_sample_rows' out at rows_offset_bytes): a draft is
 *  kept iff its row drew it.  out receives n_runs pulsar_gpu_spec_walk_out at out_offset_bytes. */
/** @param rows              the sampler's out (pulsar_gpu_sample_out rows)
 *  @param rows_offset_bytes byte offset of row 0
 *  @param n_rows            rows available
 *  @param runs              host [n_runs] verify runs
 *  @param n_runs            runs
 *  @param out               receives n_runs pulsar_gpu_spec_walk_out
 *  @param out_offset_bytes  byte offset into out
 *  @return 1 on success (enqueued), 0 on a bad launch */
int pulsar_gpu_spec_walk(const pulsar_gpu_tensor *rows, uint64_t rows_offset_bytes, uint32_t n_rows,
                         const pulsar_gpu_spec_run *runs, uint32_t n_runs, pulsar_gpu_tensor *out,
                         uint64_t out_offset_bytes);

/** The logprob outputs of a row (the host's pulsar_token_score, field for field). */
#define PULSAR_GPU_LOGPROB_TOPK_MAX 20u
/** The -inf stand-in a logprob row writes for a non-finite logit or an empty top slot. */
#define PULSAR_GPU_LOGPROB_NEG_INF (-1.0e30f)
/** One token's score in a logprob row (the host's pulsar_token_score). */
typedef struct pulsar_gpu_token_score {
    /** Token id; -1 in a top slot past the row's finite candidates. */
    int32_t id;
    /** The raw logit (after the mask); PULSAR_GPU_LOGPROB_NEG_INF in an empty slot. */
    float logit;
    /** log-softmax of the logit over the row; PULSAR_GPU_LOGPROB_NEG_INF for a non-finite logit or an empty slot. */
    float logprob;
} pulsar_gpu_token_score;
/** What pulsar_gpu_logprobs_rows writes per row. */
typedef struct pulsar_gpu_logprob_row {
    uint32_t n_top;        ///< entries in top (0: the row has no finite logit)
    uint32_t have_token;   ///< token holds the scored token's score
    /** The scored token's score; valid when have_token. */
    pulsar_gpu_token_score token;
    /** The row's best tokens, best first; entries [0, n_top) are real. */
    pulsar_gpu_token_score top[PULSAR_GPU_LOGPROB_TOPK_MAX];
} pulsar_gpu_logprob_row;
/** Logprobs of n_rows rows: the scored token of row r (tokens at tokens_offset_bytes + r * token_stride int32s;
 *  NULL = none) and the top_k (<= PULSAR_GPU_LOGPROB_TOPK_MAX) best, as the host scorers did (frozen in
 *  tests/sampler_host_ref.cpp).  mask_rows (NULL = unmasked) names each row's mask row. */
/** @param logits              f32 logits rows
 *  @param logits_offset_bytes byte offset of row 0
 *  @param row_stride          floats between rows
 *  @param n_vocab             row width
 *  @param n_rows              rows
 *  @param mask                uint32 token masks, or NULL
 *  @param mask_words          words per mask row
 *  @param mask_rows           host [n_rows] mask row per row, or NULL = unmasked
 *  @param tokens              int32 scored tokens, or NULL
 *  @param tokens_offset_bytes byte offset of row 0's token
 *  @param token_stride        int32s between rows' tokens
 *  @param top_k               best tokens per row (<= PULSAR_GPU_LOGPROB_TOPK_MAX)
 *  @param scratch             pulsar_gpu_logprobs_scratch_bytes of scratch
 *  @param out                 receives n_rows pulsar_gpu_logprob_row
 *  @return 1 on success, 0 on a bad launch */
int pulsar_gpu_logprobs_rows(const pulsar_gpu_tensor *logits, uint64_t logits_offset_bytes, uint32_t row_stride,
                             uint32_t n_vocab, uint32_t n_rows, const pulsar_gpu_tensor *mask, uint32_t mask_words,
                             const uint32_t *mask_rows, const pulsar_gpu_tensor *tokens, uint64_t tokens_offset_bytes,
                             uint32_t token_stride, uint32_t top_k, pulsar_gpu_tensor *scratch,
                             pulsar_gpu_tensor *out);
/** Bytes of the scratch a logprob launch over rows of n_vocab takes (any row count). */
/** @param n_vocab logits row width
 *  @return scratch bytes */
uint64_t pulsar_gpu_logprobs_scratch_bytes(uint32_t n_vocab);

/** A block drafter's sampled chains (L298 rows, L302 on the device): chain c walks slots 0..n-1, slot j's row the
 *  candidates cand[c][j][0..topk) scored by scores[c][j][prev][0..topk) (prev: the slot before's pick, 0 at slot 0)
 *  -- a sparse row of `vocab` (every other id -inf, a repeated id keeps its last score, an id outside the vocabulary
 *  none) drawn by THE sampler under the chain's knobs at key[j]; the pick is the first candidate equal to the draw.
 *  out receives tokens [n_chains][slots] (int32; -1 from a refused slot on, the walk stops there). */
#define PULSAR_GPU_CHAIN_SLOTS_MAX 16u
/** The most candidates per chain slot. */
#define PULSAR_GPU_CHAIN_TOPK_MAX 64u
/** The most chains one launch walks. */
#define PULSAR_GPU_CHAINS_MAX 16u
/** One chain's draw request for pulsar_gpu_sample_chains. */
typedef struct pulsar_gpu_chain_req {
    float temperature, top_p, min_p;
    /** Top-k cut (the pulsar_gpu_sample_req rules, as are temperature / top_p / min_p). */
    int32_t top_k;
    /** The chain's index in cand / scores. */
    uint32_t chain;   /* the chain's index in cand / scores */
    /** Slots to walk (<= slots, <= PULSAR_GPU_CHAIN_SLOTS_MAX). */
    uint32_t n;       /* slots to walk (<= slots) */
    /** Per slot j: the draw's random key (pulsar_sample_key at the token's index). */
    uint64_t key[PULSAR_GPU_CHAIN_SLOTS_MAX];
} pulsar_gpu_chain_req;
/** @var pulsar_gpu_chain_req::temperature
 *  Sampler temperature (pulsar_gpu_sample_req rules). */
/** @var pulsar_gpu_chain_req::top_p
 *  Sampler nucleus mass (pulsar_gpu_sample_req rules). */
/** @var pulsar_gpu_chain_req::min_p
 *  Sampler min-p floor (pulsar_gpu_sample_req rules). */
/** @brief Walk a block drafter's sampled chains on the device (see above).
 *  @param cand                int32 candidates [chains][slots][topk]
 *  @param cand_offset_bytes   byte offset into cand
 *  @param scores              f32 scores [chains][slots][topk][topk]
 *  @param scores_offset_bytes byte offset into scores
 *  @param slots               slots per chain (<= PULSAR_GPU_CHAIN_SLOTS_MAX)
 *  @param topk                candidates per slot (<= PULSAR_GPU_CHAIN_TOPK_MAX)
 *  @param vocab               vocabulary width of the sparse rows
 *  @param reqs                host [n_chains] requests
 *  @param n_chains            chains (<= PULSAR_GPU_CHAINS_MAX)
 *  @param out                 int32 tokens [n_chains][slots] out
 *  @param out_offset_bytes    byte offset into out
 *  @return 1 on success, 0 on a bad launch */
int pulsar_gpu_sample_chains(const pulsar_gpu_tensor *cand, uint64_t cand_offset_bytes, const pulsar_gpu_tensor *scores,
                             uint64_t scores_offset_bytes, uint32_t slots, uint32_t topk, uint32_t vocab,
                             const pulsar_gpu_chain_req *reqs, uint32_t n_chains, pulsar_gpu_tensor *out,
                             uint64_t out_offset_bytes);

/** A deterministic 64-bit digest of n_rows f32 rows of `width` (8-byte aligned) into out at out_offset_bytes: the
 *  TP mirror's cross-rank identity check over rows that stay on the device (L243, L302). */
/** @param rows             f32 rows
 *  @param offset_bytes     byte offset of row 0 (8-byte aligned)
 *  @param n_rows           rows
 *  @param width            floats per row
 *  @param out              receives the uint64 digest
 *  @param out_offset_bytes byte offset into out
 *  @return 1 on success, 0 on a bad launch */
int pulsar_gpu_rows_digest(const pulsar_gpu_tensor *rows, uint64_t offset_bytes, uint32_t n_rows, uint32_t width,
                           pulsar_gpu_tensor *out, uint64_t out_offset_bytes);


/** L150: bank-batched markov refine. One launch per draft position serves
 * n_banks banks (one markov_w2 stream shared across them); byte-exact per
 * bank with the single-bank kernels. Layouts: refined_logits [n_banks][vocab]
 * f32; ids_dev [n_banks][ids_stride] i32 with [b][0] seeded by the caller
 * (chain) and winners written to [b][pos+1]; the bank's base logits row
 * for position pos is base_logits row (base_row_dev[b] + pos), rows
 * base_row_stride_bytes apart. The chain runs positions 0..n_draft-1 with the
 * device feed; the single step takes the previous token per bank from prev_dev
 * ([n_banks] i32, caller-written -- the sampled path's host draw). */
#define PULSAR_DSPARK_BANKS_MAX 16u
/** L260: the most drafter rows one redraft forward carries (every bank of a
 *  16-bank pool at depth 4 -- ONE group per step at c16, so the drafter's head,
 *  dense and markov weights are read once per step, not once per 8-bank /
 *  16-row group).  Its own decode-rows bound (pulsar_decode_rows_scope with
 *  this cap); PULSAR_GPU_MNEUTRAL_ROWS_MAX keeps inferring row kinds. */
#define PULSAR_DSPARK_DRAFT_ROWS_MAX 64u
/** w2 steps the banked markov kernel loads ahead of their products (L262);
 *  the drafter's embed_dim must be a multiple. */
#define PULSAR_DSPARK_MARKOV_PREFETCH 8u
/** spec_logits rows: the verify slab and the drafter forward both write it. */
#define PULSAR_SPEC_LOGITS_ALLOC_ROWS \
    (PULSAR_DSPARK_DRAFT_ROWS_MAX > PULSAR_SPEC_LOGITS_ROWS ? PULSAR_DSPARK_DRAFT_ROWS_MAX : PULSAR_SPEC_LOGITS_ROWS)
/** @brief The greedy Markov refine chain over n_draft positions for every bank, fed on the device (see above).
 *  @param refined_logits   f32 [n_banks][vocab_size] refined logits out
 *  @param ids_dev          int32 [n_banks][ids_stride] chain ids ([b][0] seeded by the caller; winners at [b][pos + 1])
 *  @param ids_stride       ids_dev row stride, elements
 *  @param base_logits      the drafter's base logits rows
 *  @param base_row_stride_bytes bytes between base_logits rows
 *  @param base_row_dev     device int32 [n_banks] first base row per bank
 *  @param dspark_model_map  the drafter's mapping
 *  @param dspark_model_size its size in bytes
 *  @param markov_w1_offset  byte offset of markov_w1 [vocab_size][embed_dim]
 *  @param markov_w2_offset  byte offset of markov_w2
 *  @param n_banks           banks (<= PULSAR_DSPARK_BANKS_MAX)
 *  @param n_draft           positions to chain
 *  @param vocab_size        vocabulary size
 *  @param embed_dim         Markov embedding width (a multiple of PULSAR_DSPARK_MARKOV_PREFETCH)
 *  @param w1_bf16           1 = markov_w1 stored bf16, 0 = f32
 *  @param w2_fmt            PULSAR_MARKOV_W2_* storage of markov_w2
 *  @return 1 on success, 0 on a bad argument or failure */
int pulsar_gpu_dspark_markov_chain_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t n_draft, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_fmt);
/** @brief One Markov refine position for every bank, previous tokens from prev_dev.
 *  @param refined_logits   f32 [n_banks][vocab_size] refined logits out
 *  @param ids_dev          int32 [n_banks][ids_stride] chain ids ([b][0] seeded by the caller; winners at [b][pos + 1])
 *  @param ids_stride       ids_dev row stride, elements
 *  @param base_logits      the drafter's base logits rows
 *  @param base_row_stride_bytes bytes between base_logits rows
 *  @param base_row_dev     device int32 [n_banks] first base row per bank
 *  @param prev_dev         device int32 [n_banks] previous token per bank
 *  @param dspark_model_map  the drafter's mapping
 *  @param dspark_model_size its size in bytes
 *  @param markov_w1_offset  byte offset of markov_w1 [vocab_size][embed_dim]
 *  @param markov_w2_offset  byte offset of markov_w2
 *  @param n_banks           banks (<= PULSAR_DSPARK_BANKS_MAX)
 *  @param pos               the draft position
 *  @param vocab_size        vocabulary size
 *  @param embed_dim         Markov embedding width (a multiple of PULSAR_DSPARK_MARKOV_PREFETCH)
 *  @param w1_bf16           1 = markov_w1 stored bf16, 0 = f32
 *  @param w2_fmt            PULSAR_MARKOV_W2_* storage of markov_w2
 *  @return 1 on success, 0 on a bad argument or failure */
int pulsar_gpu_dspark_markov_step_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev, const pulsar_gpu_tensor *prev_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t pos, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_fmt);

/** DSpark Markov + confidence heads */

/** @brief One Markov refine step for one bank; reads the refined argmax back (blocking).
 *  @param refined_logits    f32 [vocab_size] refined logits out
 *  @param refined_id_dst    host: receives the refined argmax
 *  @param base_logits       f32 [vocab_size] base logits
 *  @param dspark_model_map  the drafter's mapping
 *  @param dspark_model_size its size in bytes
 *  @param markov_w1_offset  byte offset of markov_w1
 *  @param markov_w2_offset  byte offset of markov_w2
 *  @param prev_token        the previous token (< vocab_size)
 *  @param vocab_size        vocabulary size
 *  @param embed_dim         Markov embedding width (<= 1024)
 *  @param w1_bf16           see below
 *  @param w2_fmt            see below
 *  @return 1 on success, 0 on a bad argument or failure */
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
        /* markov_w1: 1 = bf16 (source format), 0 = f32.  markov_w2: a
         * PULSAR_MARKOV_W2_* format (the two are separate tensors and get
         * separate descriptors).  The step streams all of markov_w2 at one
         * FMA per element, so its storage sets the kernel's runtime, not
         * just its footprint. */
        int                    w1_bf16,
        int                    w2_fmt);

/** @brief out[p] = the mean over the n_hc streams of hc_batch[p] (the drafter's anchor hidden per position).
 *  @param out      f32 [n_tokens][n_embd]
 *  @param hc_batch HC residual [n_tokens][n_hc][n_embd]
 *  @param n_embd   embedding width
 *  @param n_hc     HC streams
 *  @param n_tokens rows
 *  @return 1 on success, 0 on a bad size or launch failure */
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
/** @param logits_rows f32 [n_rows][vocab]
 *  @param n_rows      rows
 *  @param vocab       row width
 *  @param top_ids     int32 [..][64] out, written at rows [row0, row0 + n_rows)
 *  @param top_vals    f16 [..][64] logits out
 *  @param tail_lse    f16 [..] log-sum-exp of the tail out
 *  @param inexact     int32 counter of rows whose candidate union missed the exact top-64
 *  @param row0        first output row
 *  @return 1 on success, 0 on a bad size or launch failure */
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
/** @param scores            f32 [n_positions] confidences out
 *  @param hidden            f32 [n_positions][hidden_dim]
 *  @param token_ids         int32 [n_positions] block tokens
 *  @param dspark_model_map  the drafter's mapping
 *  @param dspark_model_size its size in bytes
 *  @param markov_w1_offset  byte offset of markov_w1 [vocab_size][embed_dim]
 *  @param proj_offset       byte offset of the projection [hidden_dim + embed_dim]
 *  @param n_positions       positions
 *  @param hidden_dim        hidden width
 *  @param embed_dim         Markov embedding width
 *  @param vocab_size        vocabulary size
 *  @param w1_bf16           1 = markov_w1 stored bf16
 *  @param proj_bf16         1 = proj stored bf16
 *  @return 1 on success, 0 on a bad argument or launch failure */
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

/* ---------------------------------------------------------------------------
 * L216: the Vision-Exp tower's shape and its CUDA-facing weight contract.
 *
 * These live HERE, not in the engine's internal header, because the CUDA TUs
 * cannot see engine-internal types (pulsar_cuda_internal.h pulls in only
 * cuda/cub headers) and every kernel in the tree takes (map, size, offset) for
 * the same reason.  The engine builds the offsets from pulsar_vision_weights;
 * the kernels read them against the mapped model base.
 *
 * The artifact carries the vision TENSORS but no vision metadata (the template
 * writes only the text-side config keys), so these compiled values are the
 * authority.  They match the checkpoint's config.json (vision_n_layers 32,
 * vision_dim 1024, vision_n_heads 16, vision_inter_dim 2816,
 * vision_patch_size 14, vision_downsample_ratio 3, vision_rope_theta 10000) and
 * predict every `vision.*` tensor's dims. */
#define PULSAR_VISION_LAYERS      32u
/** DeepSeek vision tower width (vision_dim). */
#define PULSAR_VISION_DIM         1024u
/** Attention heads per vision block. */
#define PULSAR_VISION_HEADS       16u
/** Vision MLP intermediate width. */
#define PULSAR_VISION_INTER       2816u
/** Patch side, pixels. */
#define PULSAR_VISION_PATCH       14u
/** The aligner's downsample ratio: R x R patches become one text-model row. */
#define PULSAR_VISION_DOWNSAMPLE  3u
/** The tower's 2-D RoPE base. */
#define PULSAR_VISION_ROPE_THETA  10000.0f
/* The reference's image-preprocessing POLICY (config.json: vision_max_n_token,
 * vision_min_pixels, vision_max_wh_ratio).  Unlike PATCH/DOWNSAMPLE these are
 * not derivable from any tensor shape -- they bound how large an image the model
 * will accept -- and the artifact carries no metadata for them: every scalar key
 * in the GGUF is deepseek4.<...> or dspark.<...>, with no vision.<...> scalar (the tower
 * arrives as tensors only).  They are therefore constants of THIS checkpoint,
 * exactly as the tower dims above are.  A future artifact that changes them
 * needs them in the GGUF metadata, not a second constant here. */
/** The most visual tokens one image may become (vision_max_n_token). */
#define PULSAR_VISION_MAX_N_TOKEN  384
/** The fewest pixels an image is resized to (vision_min_pixels). */
#define PULSAR_VISION_MIN_PIXELS   147456
/** The widest aspect ratio accepted (vision_max_wh_ratio). */
#define PULSAR_VISION_MAX_WH_RATIO 8.0f

/** One vision-tower tensor as a file offset into the model mapping.  The CUDA
 * forward reads every weight through these; the engine fills them from the
 * bound pulsar_vision_weights. */
typedef struct {
    uint64_t patch_proj, patch_bias, norm;
    uint64_t aligner_w1, aligner_b1, aligner_w2, aligner_b2;
    /** Per transformer block: its norm1, attn.wqkv (+ bias), attn.wo (+ bias), norm2 and MLP w1 / w2 tensors. */
    struct {
        uint64_t norm1, wqkv, wqkv_bias, wo, wo_bias, norm2, w1, w2;
    } block[PULSAR_VISION_LAYERS];
    uint32_t n_layers;              ///< must be PULSAR_VISION_LAYERS
    uint32_t text_dim;              ///< the aligner's output width (the text model's n_embd)
} pulsar_vision_offsets;
/** @var pulsar_vision_offsets::patch_proj
 *  vision.patch_embed.proj.weight: the patch embedding. */
/** @var pulsar_vision_offsets::patch_bias
 *  vision.patch_embed.proj.bias. */
/** @var pulsar_vision_offsets::norm
 *  vision.norm.weight: the RMSNorm after the last block. */
/** @var pulsar_vision_offsets::aligner_w1
 *  aligner.w1.weight: the aligner's first linear (a DOWNSAMPLE^2 patch group to text_dim, then GELU). */
/** @var pulsar_vision_offsets::aligner_b1
 *  aligner.w1.bias. */
/** @var pulsar_vision_offsets::aligner_w2
 *  aligner.w2.weight: the aligner's second linear. */
/** @var pulsar_vision_offsets::aligner_b2
 *  aligner.w2.bias. */

/** ViT + aligner over ONE image's patches: `patches` is
 * (n_h*n_w, 3, PATCH, PATCH) bf16, the result is (out_rows, text_dim) bf16
 * written to `out`.  Returns 0 on any refusal.
 *
 * `dbg` is an OPTIONAL instrument (NULL in production): when set it receives
 * (2 + dbg_blocks) rows of n_h*n_w*PULSAR_VISION_DIM bf16 -- patch_embed,
 * blocks 0..dbg_blocks-1, then the final norm -- which is how
 * tests/vision_tower_gate.cpp localises a mismatch instead of guessing. */
/** @param o          the tower's tensor offsets
 *  @param map        the model mapping
 *  @param map_size   its size in bytes
 *  @param patches    HOST bf16 patches, (n_h * n_w, 3, PATCH, PATCH)
 *  @param n_h        patch rows
 *  @param n_w        patch columns
 *  @param out        HOST bf16 out, (out_rows, text_dim)
 *  @param out_cap    capacity of out, elements
 *  @param out_rows   receives the rows written (ceil(n_h / R) * ceil(n_w / R))
 *  @param dbg        HOST debug rows, or NULL
 *  @param dbg_blocks blocks dumped into dbg
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_cuda_vision_forward(const pulsar_vision_offsets *o,
                               const void *map, uint64_t map_size,
                               const uint16_t *patches, int n_h, int n_w,
                               uint16_t *out, int out_cap, int *out_rows,
                               uint16_t *dbg, uint32_t dbg_blocks);

/** L268: the image towers' GEMM (both families' towers, pulsar_cuda_vision.cu): C (f32, row-major [m x n], ldc) =
 *  A (bf16 row-major [m x k], lda) . op(B), B bf16 row-major -- [n x k] with op(B) = B^T when b_trans, else [k x n].
 *  f32 accumulate on cuBLASLt.  Returns 0 on refusal (said). */
/** @param c       device f32 C
 *  @param ldc     C's leading dimension
 *  @param a       device bf16 A
 *  @param lda     A's leading dimension
 *  @param b       device bf16 B
 *  @param ldb     B's leading dimension
 *  @param b_trans nonzero: B is [n x k] and op(B) = B^T
 *  @param m       rows of C
 *  @param n       columns of C
 *  @param k       the reduction
 *  @return 1 on success, 0 on refusal (said) */
int pulsar_cuda_vision_gemm(float *c, int ldc, const uint16_t *a, int lda, const uint16_t *b, int ldb, int b_trans,
                            int m, int n, int k);

/* L268: Qwen3.8-Flash-Next's vision tower (config.json vision_config; every `model.visual.*` tensor's dims follow).
 * The weights are bf16 and read through the model's mapping, as DeepSeek's tower's are. */
/** Qwen vision blocks. */
#define PULSAR_QWEN_VISION_LAYERS     27u
/** Qwen vision tower width. */
#define PULSAR_QWEN_VISION_DIM        1152u
/** Attention heads per Qwen vision block. */
#define PULSAR_QWEN_VISION_HEADS      16u
/** Qwen vision MLP intermediate width. */
#define PULSAR_QWEN_VISION_INTER      4304u
/** Values per input patch (3 channels x 2 frames x 16 x 16). */
#define PULSAR_QWEN_VISION_PATCH_IN   1536u     /* 3 channels x 2 frames x 16 x 16 */
/** Side of the learned position grid (48 x 48 = 2304 entries). */
#define PULSAR_QWEN_VISION_POS_SIDE   48u       /* num_position_embeddings 2304 = 48 x 48 */
/** The merger's output width: the language model's hidden size. */
#define PULSAR_QWEN_VISION_OUT        2560u     /* out_hidden_size: the language model's width */
/** The Qwen tower's 2-D RoPE base. */
#define PULSAR_QWEN_VISION_ROPE_THETA 10000.0f
/** One Qwen vision block's weights: bf16 pointers into the model's mapping (device-readable; not owned), each the
 *  `model.visual.blocks.<l>.*` tensor named below.  Linear shapes are (in, out). */
typedef struct {
    const void *norm1_w, *norm1_b, *qkv_w, *qkv_b, *proj_w, *proj_b;
    const void *norm2_w, *norm2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} pulsar_qwen_vision_block_dev;
/** @var pulsar_qwen_vision_block_dev::norm1_w
 *  norm1.weight, DIM (the LayerNorm before attention). */
/** @var pulsar_qwen_vision_block_dev::norm1_b
 *  norm1.bias, DIM. */
/** @var pulsar_qwen_vision_block_dev::qkv_w
 *  attn.qkv.weight, (DIM, 3 DIM). */
/** @var pulsar_qwen_vision_block_dev::qkv_b
 *  attn.qkv.bias, 3 DIM. */
/** @var pulsar_qwen_vision_block_dev::proj_w
 *  attn.proj.weight, (DIM, DIM). */
/** @var pulsar_qwen_vision_block_dev::proj_b
 *  attn.proj.bias, DIM. */
/** @var pulsar_qwen_vision_block_dev::norm2_w
 *  norm2.weight, DIM (the LayerNorm before the MLP). */
/** @var pulsar_qwen_vision_block_dev::norm2_b
 *  norm2.bias, DIM. */
/** @var pulsar_qwen_vision_block_dev::fc1_w
 *  mlp.linear_fc1.weight, (DIM, INTER). */
/** @var pulsar_qwen_vision_block_dev::fc1_b
 *  mlp.linear_fc1.bias, INTER. */
/** @var pulsar_qwen_vision_block_dev::fc2_w
 *  mlp.linear_fc2.weight, (INTER, DIM). */
/** @var pulsar_qwen_vision_block_dev::fc2_b
 *  mlp.linear_fc2.bias, DIM. */
/** The Qwen vision tower's weights (vision_qwen.cpp binds them): bf16 pointers into the model's mapping
 *  (device-readable; not owned), each the `model.visual.*` tensor named below.  Linear shapes are (in, out). */
typedef struct {
    const void *patch_w, *patch_b, *pos_embed;
    /** The PULSAR_QWEN_VISION_LAYERS blocks, in order. */
    pulsar_qwen_vision_block_dev block[PULSAR_QWEN_VISION_LAYERS];
    const void *merger_norm_w, *merger_norm_b, *merger_fc1_w, *merger_fc1_b, *merger_fc2_w, *merger_fc2_b;
} pulsar_qwen_vision_weights_dev;
/** @var pulsar_qwen_vision_weights_dev::patch_w
 *  patch_embed.proj.weight: the patch projection, PATCH_IN (3 channels, 2 frames, 16 x 16) to DIM. */
/** @var pulsar_qwen_vision_weights_dev::patch_b
 *  patch_embed.proj.bias, DIM. */
/** @var pulsar_qwen_vision_weights_dev::pos_embed
 *  pos_embed.weight: the learned position table, POS_SIDE^2 rows of DIM. */
/** @var pulsar_qwen_vision_weights_dev::merger_norm_w
 *  merger.norm.weight, DIM. */
/** @var pulsar_qwen_vision_weights_dev::merger_norm_b
 *  merger.norm.bias, DIM. */
/** @var pulsar_qwen_vision_weights_dev::merger_fc1_w
 *  merger.linear_fc1.weight, (4 DIM, 4 DIM): over a 2x2 merge block's concatenated rows. */
/** @var pulsar_qwen_vision_weights_dev::merger_fc1_b
 *  merger.linear_fc1.bias, 4 DIM. */
/** @var pulsar_qwen_vision_weights_dev::merger_fc2_w
 *  merger.linear_fc2.weight, (4 DIM, OUT). */
/** @var pulsar_qwen_vision_weights_dev::merger_fc2_b
 *  merger.linear_fc2.bias, OUT. */
/** Tower + merger over ONE image: `patches` (n_tok x PATCH_IN bf16, 2x2 merge-block order), `pos` (n_tok x 2
 *  int32: the patch's row and column), the learned-position taps (n_tok x 4 int32 table rows, n_tok x 4 f32
 *  weights).  Writes n_tok/4 x OUT bf16 rows.  `dbg` (NULL in production) receives 3 x n_tok x DIM bf16: the
 *  blocks' input, block 0's output, the last block's output.  Returns 0 on any refusal. */
/** @param w          the bound tower weights
 *  @param patches    HOST bf16 [n_tok][PATCH_IN]
 *  @param pos        HOST int32 [n_tok][2] patch row and column
 *  @param interp_idx HOST int32 [n_tok][4] position-table rows
 *  @param interp_w   HOST f32 [n_tok][4] their weights
 *  @param n_tok      patches (a multiple of 4)
 *  @param out        HOST bf16 [n_tok / 4][OUT]
 *  @param out_cap    capacity of out, elements
 *  @param dbg        HOST debug rows, or NULL
 *  @return 1 on success, 0 on refusal or failure */
int pulsar_cuda_qwen_vision_forward(const pulsar_qwen_vision_weights_dev *w, const uint16_t *patches,
                                    const int32_t *pos, const int32_t *interp_idx, const float *interp_w,
                                    int n_tok, uint16_t *out, int out_cap, uint16_t *dbg);

/* Tensor-parallel row lane, GPU half (L241 4g-2; src/cuda/pulsar_cuda_tp.cu).
 * One exchange = stage+publish + combine on the calling thread's stream;
 * nothing here waits on the host.  `slab_dev` is the registered slab's device
 * mapping; offsets, the ring size and the exchange/message numbers come from
 * the transport (pulsar_tp_row_lane_begin / pulsar_tp_row_lane_layout).
 *
 * Stage+publish: `rows` rows of src -> the slab's out-slots, and the last
 * block to finish publishes the descriptor {exch, first_msg, rows} at
 * `desc_dev` (id last, release/system).  A non-NULL `addend` is folded in
 * first: src[i] = src[i] + addend[i] (the value staged), addend[i] = 0.
 * `ticket` is a zeroed device u32 the kernel leaves zeroed; one per stream. */
/** @brief Row lane stage + publish (see above).
 *  @param src        f32 [rows][vec_bytes / 4] this rank's partial (the addend folded in)
 *  @param addend     f32 rows folded into src then zeroed, or NULL
 *  @param slab_dev   the registered slab's device mapping
 *  @param out_off    byte offset of the slab's out-slots
 *  @param vec_bytes  bytes per message slot (one row)
 *  @param first_msg  the exchange's first message number
 *  @param n_slots    slots in the ring
 *  @param rows       rows (messages)
 *  @param desc_dev   the descriptor the last block publishes
 *  @param exch       this exchange's number (nonzero)
 *  @param ticket     a zeroed device u32, one per stream
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_stage_publish(pulsar_gpu_tensor *src, pulsar_gpu_tensor *addend,
                                void *slab_dev, uint64_t out_off, uint64_t vec_bytes,
                                uint64_t first_msg, uint32_t n_slots, uint32_t rows,
                                void *desc_dev, uint64_t exch, pulsar_gpu_tensor *ticket);
/* The row lane's stream wait (L269-perf): resolves cuStreamWaitValue64 and
 * checks the device has 64-bit stream memory operations.  Called once at the
 * lane's bring-up; 0 with `err` filled = the lane cannot run here. */
/** @param err    receives the reason on a 0 return
 *  @param errlen its capacity
 *  @return 1 when the lane's stream wait is available, else 0 */
int pulsar_gpu_tp_wait_init(char *err, size_t errlen);
/* dst[r][*] = dst[r][*] + peer[r][*]; the stream first waits (front end, no SM
 * held) for done >= exch.  A set error word (`err_dev`: the lane failed and
 * released the wait) leaves dst untouched -- every combine below does the same. */
/** @param dst        f32 [rows][vec_bytes / 4], += the peer's rows
 *  @param slab_dev   the registered slab's device mapping
 *  @param in_off     byte offset of the slab's in-slots
 *  @param vec_bytes  bytes per message slot (a multiple of 4)
 *  @param first_msg  the exchange's first message number
 *  @param n_slots    slots in the ring
 *  @param rows       rows
 *  @param done_dev   device u64 the proxy advances as exchanges complete
 *  @param exch       this exchange's number (the stream waits for done >= exch)
 *  @param err_dev    the lane's device error word (set = leave dst untouched)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_combine_sum(pulsar_gpu_tensor *dst, const void *slab_dev,
                              uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                              uint32_t n_slots, uint32_t rows, const void *done_dev,
                              uint64_t exch, const void *err_dev);
/* The vocab gather's peer half: waits for done >= exch, then scatters this
 * chunk (`msgs` messages holding payload elements elem0..) of the peer's
 * packed [rows][width] slice to dst[r * pitch + col0 + c]. */
/** @param dst        f32 [rows][pitch]
 *  @param slab_dev   the registered slab's device mapping
 *  @param in_off     byte offset of the slab's in-slots
 *  @param vec_bytes  bytes per message slot (a multiple of 4)
 *  @param first_msg  the exchange's first message number
 *  @param n_slots    slots in the ring
 *  @param msgs       messages in this chunk
 *  @param elem0      first payload element the chunk holds
 *  @param rows       rows of the slice
 *  @param width      the slice's columns
 *  @param pitch      dst row stride, elements
 *  @param col0       first dst column of the slice (col0 + width <= pitch)
 *  @param done_dev   device u64 the proxy advances as exchanges complete
 *  @param exch       this exchange's number (the stream waits for done >= exch)
 *  @param err_dev    the lane's device error word (set = leave dst untouched)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_combine_scatter(pulsar_gpu_tensor *dst, const void *slab_dev,
                                  uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                                  uint32_t n_slots, uint32_t msgs, uint64_t elem0,
                                  uint32_t rows, uint64_t width, uint64_t pitch, uint64_t col0,
                                  const void *done_dev, uint64_t exch, const void *err_dev);
/* The own half: packed [rows][width] src -> dst[r * pitch + col0 + c]. */
/** @param dst   f32 [rows][pitch]
 *  @param src   f32 packed [rows][width]
 *  @param rows  rows
 *  @param width columns
 *  @param pitch dst row stride, elements
 *  @param col0  first dst column (col0 + width <= pitch)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_scatter_cols(pulsar_gpu_tensor *dst, const pulsar_gpu_tensor *src,
                               uint32_t rows, uint64_t width, uint64_t pitch, uint64_t col0);
/* The row lane's error word (host view of the slab's), read at every stream
 * drain (pulsar_gpu_end_commands / pulsar_gpu_synchronize) beside the routed
 * experts' flags: a set word fails the step there, by name, before any logits
 * the drained work produced are read back.  NULL disarms. */
/** @param word the host view of the slab's error word, or NULL to disarm */
void pulsar_gpu_tp_err_word_set(const volatile uint32_t *word);
/* A ROW-SPLIT GATHER over several projections at once (L241 4g-3): each rank
 * computed its row half of every block into one packed payload -- block k is
 * [rows][width_k] at element elem0_k -- which crossed as one row-lane exchange.
 * After the exchange this scatters BOTH halves of every block into the full
 * destination rows: dst_k[r * pitch_k + own_col0_k + c] = own payload,
 * dst_k[r * pitch_k + peer_col0_k + c] = the peer's (the same layout, in the
 * row-lane in-slots).  Copies only.  Behind the stream wait like the sums. */
/** The most blocks one gather (pulsar_gpu_tp_combine_gather_blocks) carries. */
#define PULSAR_TP_GATHER_BLOCKS_MAX 8
/** One projection's block in a row-split gather (pulsar_gpu_tp_combine_gather_blocks).  Every count is in
 *  elements, not bytes. */
typedef struct {
    /** The block's full destination rows, [rows][pitch]; both halves are written into it. */
    pulsar_gpu_tensor *dst;
    uint64_t elem0, width, pitch, own_col0, peer_col0;
} pulsar_tp_gather_block;
/** @var pulsar_tp_gather_block::elem0
 *  Element offset of this block's packed [rows][width] half in the exchange payload (own and peer alike). */
/** @var pulsar_tp_gather_block::width
 *  Columns of one rank's half of the block. */
/** @var pulsar_tp_gather_block::pitch
 *  Row stride of dst, elements. */
/** @var pulsar_tp_gather_block::own_col0
 *  First dst column of this rank's half. */
/** @var pulsar_tp_gather_block::peer_col0
 *  First dst column of the peer's half. */
/** @param blocks     host [n_blocks]; their elem0 must run contiguously from 0 (each = the previous + rows * width)
 *  @param n_blocks   blocks (1 .. PULSAR_TP_GATHER_BLOCKS_MAX)
 *  @param rows       rows of every block
 *  @param own        f32: this rank's packed payload
 *  @param slab_dev   the registered slab's device mapping
 *  @param in_off     byte offset of the slab's in-slots
 *  @param vec_bytes  bytes per message slot (a multiple of 4)
 *  @param first_msg  the exchange's first message number
 *  @param n_slots    slots in the ring
 *  @param done_dev   device u64 the proxy advances as exchanges complete
 *  @param exch       this exchange's number (the stream waits for done >= exch)
 *  @param err_dev    the lane's device error word (set = leave dst untouched)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_combine_gather_blocks(const pulsar_tp_gather_block *blocks, uint32_t n_blocks,
                                        uint32_t rows, const pulsar_gpu_tensor *own,
                                        const void *slab_dev, uint64_t in_off, uint64_t vec_bytes,
                                        uint64_t first_msg, uint32_t n_slots, const void *done_dev,
                                        uint64_t exch, const void *err_dev);
/* The BULK lane (prefill-sized exchanges, v14; pulsar_tp.h): stage `bytes` of
 * `src` from `off` into the mapped bulk out-region with an SM kernel -- plus
 * the addend at the same offset when there is one, which is then zeroed --
 * publish the descriptor {exch, bytes, word2 = bulk flag | buffer}, and combine
 * dst[off..] = own + the peer's rows (own = the staged out-region, peer = the
 * mapped receive buffer) once done >= exch. */
/** @param src     f32 partial
 *  @param addend  f32 folded in at the same offset then zeroed, or NULL
 *  @param off     byte offset into src (and addend), a multiple of 4
 *  @param out_dev the mapped bulk out-region
 *  @param bytes   bytes to stage (a nonzero multiple of 4)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_bulk_stage(const pulsar_gpu_tensor *src, pulsar_gpu_tensor *addend, uint64_t off,
                             void *out_dev, uint64_t bytes);
/** @param desc_dev the bulk descriptor
 *  @param exch     this exchange's number (nonzero)
 *  @param bytes    bytes staged (nonzero)
 *  @param word2    the bulk flag | receive buffer word
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_publish_bulk(void *desc_dev, uint64_t exch, uint64_t bytes, uint64_t word2);
/** @param dst      f32 destination
 *  @param dst_off  byte offset into dst, a multiple of 4
 *  @param own_dev  this rank's staged partial (the out-region)
 *  @param peer_dev the mapped receive buffer
 *  @param bytes    bytes (a nonzero multiple of 4)
 *  @param done_dev   device u64 the proxy advances as exchanges complete
 *  @param exch       this exchange's number (the stream waits for done >= exch)
 *  @param err_dev    the lane's device error word (set = leave dst untouched)
 *  @return 1 on success (enqueued), 0 on a bad argument or launch failure */
int pulsar_gpu_tp_bulk_combine_sum(pulsar_gpu_tensor *dst, uint64_t dst_off, const void *own_dev,
                                   const void *peer_dev, uint64_t bytes, const void *done_dev,
                                   uint64_t exch, const void *err_dev);

/* ---- Qwen3.8-Flash-Next full attention + QSA (L251 stream S3) ----------------
 * One of the 12 full-attention layers (3, 7, ..., 47) between its Linears: the
 * caller runs q_proj / k_proj / v_proj / index_qk_proj (f32 outputs) and o_proj
 * (which reads the E4M3 slot this emits).  Semantics are transformers
 * `qwen4_exp` (Qwen4ExpTextAttention + Qwen4ExpTextQSAIndexer), text positions
 * only (the three mRoPE rows equal, so interleaved mRoPE is plain RoPE):
 *   q_proj row = 24 heads x [256 query | 256 sigmoid output gate] (per head)
 *   q/k RMSNorm per head (weight applied as 1 + w), partial RoPE on dims 0..63
 *   (rotate_half pairs (i, i+32), angle = f32(pos) * f32 inv_freq[i], cos and sin
 *   times the rope's mscale -- 1 natively, YaRN's attention factor under L280), GQA 12:1
 *   indexer: 4 q heads x 128 + 1 raw key x 128; keys mean-pooled over aligned
 *   blocks of 4 BEFORE the norm, then RMSNorm, then RoPE at the block start;
 *   score(block) = sum_h relu(q_h . k_block); a query at position p sees
 *   n = p + 1 tokens, nb = n / 4 complete blocks: nb <= 512 attends to every
 *   token; else the top 512 blocks (ties: lower block index) + the n % 4 tail.
 *   (HF divides the score by sqrt(128): a positive constant, ranking unchanged.)
 *
 * KV cache per sequence per layer (the caller allocates; S1 owns the policy):
 *   kv    cap x 1056 B per token: K h0 | K h1 | V h0 | V h1 (256 E4M3 each,
 *         post-norm post-RoPE K, raw V), then 4 x 8 E8M0 scales in the same
 *         order -- one power-of-two scale per 32 elements (the MX rule of
 *         pulsar_cuda_mx.cuh, the Engram row's scale granularity)
 *   bkey  cap/4 x 128 bf16: each COMPLETE block's pooled, normed, roped key --
 *         what the scorer reads; raw per-token indexer keys are not kept
 *   stage 4 x 128 f32: the raw indexer keys of the open (incomplete) block
 * KV/token/layer = 1056 + 64 (bkey) B; 12 layers = 13.1 KiB/token. */
enum {
    PULSAR_QSA_N_HEAD     = 24,
    PULSAR_QSA_N_KV       = 2,
    PULSAR_QSA_HEAD_DIM   = 256,
    PULSAR_QSA_ROT_DIM    = 64,
    PULSAR_QSA_Q_IN       = 12288,   /* q_proj width: 24 x (256 + 256) */
    PULSAR_QSA_KV_IN      = 512,     /* k_proj / v_proj width */
    PULSAR_QSA_OUT_DIM    = 6144,    /* o_proj input */
    PULSAR_QSA_IDX_HEADS  = 4,
    PULSAR_QSA_IDX_DIM    = 128,
    PULSAR_QSA_IDX_IN     = 640,     /* index_qk_proj width: 4 x 128 q + 128 k */
    PULSAR_QSA_BLOCK      = 4,       /* indexer_compress_ratio */
    PULSAR_QSA_TOP_BLOCKS = 512,     /* indexer_budget / indexer_compress_ratio */
    PULSAR_QSA_KV_TOKEN_BYTES = 1056,
    PULSAR_QSA_BKEY_BYTES = 256,
    PULSAR_QSA_STAGE_BYTES = 2048
};
/** The QSA layers' native RoPE base. */
#define PULSAR_QSA_ROPE_THETA 10000000.0
/** The QSA q / k / indexer RMSNorm epsilon. */
#define PULSAR_QSA_RMS_EPS    1e-6f

/** L280: YaRN over the native rope (transformers `rope_type: yarn`): `factor` > 1 times the context the rope was
 *  trained at, `orig_ctx` (original_max_position_embeddings), with the ramp bounded by beta_fast / beta_slow
 *  rotations (transformers' defaults 32 / 1).  The math is lib/pulsar_yarn.h's, shared with DeepSeek's rope. */
typedef struct {
    double   factor;               /**< double, as transformers reads the config's number */
    /** The context the rope was trained at, tokens (original_max_position_embeddings); 0 refuses. */
    uint32_t orig_ctx;
    double   beta_fast, beta_slow;
} pulsar_qsa_yarn;
/** @var pulsar_qsa_yarn::beta_fast
 *  The ramp's high-frequency bound, rotations (transformers' default 32). */
/** @var pulsar_qsa_yarn::beta_slow
 *  The ramp's low-frequency bound, rotations (transformers' default 1). */

/** The QSA rope of a loaded model: the f32 inverse frequency of each rotary pair and the factor on cos and sin
 *  (YaRN's attention factor; exactly 1 for the native rope).  pulsar_qsa_rope_init is its one producer, run once
 *  per load; every pulsar_gpu_qsa_forward call names it (pulsar_qsa_io::rope). */
typedef struct {
    /** Per rotary pair i (dims i, i + 32): the inverse frequency, radians per position. */
    float inv_freq[PULSAR_QSA_ROT_DIM / 2];
    /** The factor on cos and sin: YaRN's attention factor, exactly 1 for the native rope. */
    float mscale;
} pulsar_qsa_rope;

/** Fill `out` for the native rope (`yarn` NULL: inv_freq[i] = 1 / theta^(2i/64), mscale 1) or for YaRN, as
 *  transformers' _compute_yarn_parameters defines it -- every step evaluated in double and each result rounded
 *  once to f32 (tests/qsa_yarn_test grades the table element-wise against transformers).  Refuses (returns 0, says
 *  why) a factor <= 1, an orig_ctx of 0, or a ramp whose range is empty. */
/** @param out  the table out
 *  @param yarn YaRN settings, or NULL for the native rope
 *  @return 1 on success, 0 on refusal (said) */
int pulsar_qsa_rope_init(pulsar_qsa_rope *out, const pulsar_qsa_yarn *yarn);

/** One sequence's cache for ONE attention layer (layout above). */
typedef struct {
    /** The KV records, cap x pulsar_qsa_kv_token_bytes per token (E4M3 K and V plus E8M0 scales; caller-owned). */
    pulsar_gpu_tensor *kv;
    /** cap / 4 x 128 bf16: each complete indexer block's pooled, normed, roped key (caller-owned). */
    pulsar_gpu_tensor *bkey;
    /** 4 x 128 f32: the raw indexer keys of the open (incomplete) block (caller-owned). */
    pulsar_gpu_tensor *stage;
    uint32_t cap;              /**< token capacity, a multiple of 4 */
} pulsar_qsa_seq;

/** The layer's four norm weights, f32, as stored in the checkpoint (w, not 1 + w). */
typedef struct {
    /* Raw device pointers, f32 -- the shape S2's pulsar_gdn_weights uses, so a
     * family op can pass wptr()-resolved weights straight in (an engine op
     * cannot name pulsar_gpu_tensor's fields, and has no tensor to wrap).  The
     * binder guarantees the widths: 256 / 256 / 128 / 128. */
    const uint16_t *q_norm;      /**< [256] bf16, as stored (w, not 1 + w) */
    const uint16_t *k_norm;      /**< [256] bf16 */
    const uint16_t *idx_q_norm;  /**< [128] bf16 */
    const uint16_t *idx_k_norm;  /**< [128] bf16 */
} pulsar_qsa_layer;

/** One pulsar_gpu_qsa_forward call's activations: the projections' outputs in, o_proj's input slot out, plus the
 *  rope, the tensor-parallel degree and optional gate taps.  Every pointer is borrowed. */
typedef struct {
    const pulsar_gpu_tensor *qg;    /**< f32 [n_rows][12288] q_proj output */
    const pulsar_gpu_tensor *k;     /**< f32 [n_rows][512] k_proj output */
    const pulsar_gpu_tensor *v;     /**< f32 [n_rows][512] v_proj output */
    const pulsar_gpu_tensor *idx;   /**< f32 [n_rows][640] index_qk_proj output */
    void *out_bf16;                 /**< o_proj bf16 [n_rows][6144] -- what the Qwen o_proj reads
                                     *   (L251 / ac69748f: no E4M3 activation slot) */
    void *out_e4m3;                 /**< o_proj A8 slot: E4M3 [n_rows][6144] ... */
    void *out_scale;                /**< ... and its swizzled E8M0 slab (zeroed by the slot) */
    int   out_sf_pitch;             /**< the slot's KBp */
    /** Observation taps for gates; NULL in the lane.  out_f32 = the gated
     * attention output the slot encodes; sel = each row's selected blocks,
     * ascending, or all 0xffffffff for a row that attends to every token. */
    pulsar_gpu_tensor *tap_out_f32; /**< f32 [n_rows][6144] */
    pulsar_gpu_tensor *tap_sel;     /**< u32 [n_rows][512] */
    /** L266 step 7: the tensor-parallel degree (0 / 1 = the whole layer).  At 2 the call is ONE rank's
     *  KV head and its 12 query heads: qg [n_rows][6144], k / v [n_rows][256], KV records of
     *  pulsar_qsa_kv_token_bytes(2) per token, an o_proj input [n_rows][3072]; the indexer whole. */
    int tp_ranks;
    /** L268: interleaved multi-axis RoPE.  NULL = text positions (the row's position on every axis -- what the
     *  three mRoPE axes are without images).  Else HOST [n_rows][6]: the row's (T, H, W) and the (T, H, W) of the
     *  first token of its indexer block (pos - 3; read when the row completes the block -- that token may be an
     *  earlier call's).  Rotary pair j takes H when j % 3 == 1, W when j % 3 == 2, else T (HF's
     *  apply_interleaved_mrope over mrope_section [11, 11, 10]).  The KV row is still `row_pos`. */
    const uint32_t *row_rope;
    /** L280: the model's rope (pulsar_qsa_rope_init); required -- a call without one refuses. */
    const pulsar_qsa_rope *rope;
} pulsar_qsa_io;

/** Bytes of one token's KV record for one rank at the tensor-parallel degree tp_ranks (0 / 1: 1056). */
/** @param tp_ranks the tensor-parallel degree (0 / 1 / 2)
 *  @return bytes per token of one rank's KV record */
uint32_t pulsar_qsa_kv_token_bytes(int tp_ranks);

/** Workspace for a call of up to n_rows rows whose positions are < max_ctx. */
/** @param n_rows  rows per call
 *  @param max_ctx bound on every position
 *  @return workspace bytes */
uint64_t pulsar_gpu_qsa_workspace_bytes(uint32_t n_rows, uint32_t max_ctx);

/** Run the layer's attention for n_rows rows.  Row r is the token at position
 * row_pos[r] of sequence row_seq[r].  Each sequence's rows form ONE contiguous
 * run of consecutive positions, and the run starts at the sequence's frontier
 * (every earlier position was written by an earlier call).  The same rows give
 * the same bytes whatever else is in the batch and however a prompt is chunked
 * (decode == prefill; tests/qsa_attn_gate).  Refuses (returns 0, says why) on
 * any violated precondition. */
/** @param layer     the layer's norm weights
 *  @param seqs      host [n_seqs] per-sequence caches
 *  @param n_seqs    sequences
 *  @param row_seq   host [n_rows] sequence of each row
 *  @param row_pos   host [n_rows] position of each row
 *  @param n_rows    rows
 *  @param io        the call's activations
 *  @param workspace pulsar_gpu_qsa_workspace_bytes of scratch
 *  @return 1 on success, 0 on refusal (said) */
int pulsar_gpu_qsa_forward(const pulsar_qsa_layer *layer,
                           const pulsar_qsa_seq *seqs, uint32_t n_seqs,
                           const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows,
                           const pulsar_qsa_io *io, pulsar_gpu_tensor *workspace);

#endif

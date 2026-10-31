/** @file
 *  Tensor-parallel row-lane kernels: the GPU half of the pair's asynchronous exchange over the mapped slab. */
/* Tensor-parallel row-lane kernels (L241 4g-2): the GPU half of the pair's
 * asynchronous decode/verify exchange.
 *
 * The slab is host-pinned and GPU-mapped (pulsar_tp_gpu.h); on GB10 it is
 * coherent with the CPU, which runs the transport's proxy thread.  Per
 * exchange the stream carries a kernel, a wait and a kernel, and the engine
 * thread never waits:
 *
 *   stage    this rank's rows -> the slab's out-slots (message seq t lives in
 *            slot (t-1) % n_slots, the transport's ring rule); the LAST block
 *            to finish publishes the descriptor {exchange id, first message,
 *            rows}, the id written LAST, release/system scope -- the proxy's
 *            go signal
 *   wait     a STREAM wait (cuStreamWaitValue64, done >= exch) -- the GPU's
 *            front end holds the stream until the proxy reports the exchange
 *            complete (the peer's rows landed and our sends retired).  No SM
 *            is held while the peer works: the combine kernels spun on the
 *            done word until L269-perf, which on a shared GPU (the loopback
 *            pair: two processes, time-sliced) kept the waiting rank's context
 *            resident while the rank it waited for could not run.
 *   combine  own + peer, one add per element, own first -- the all-reduce's
 *            arithmetic, bit-identical to the host lane it replaced.
 *
 * A failed lane -- the proxy's transport timeout or error, or an abort
 * (pulsar_tp_row_lane_abort) -- latches the error word and then releases every
 * wait (pulsar_tp.cpp tp_row_err_latch); each combine reads the word first and
 * leaves its destination untouched when it is set.  The engine refuses at its
 * next host sync (pulsar_gpu_end_commands / pulsar_tp_row_lane_check), so a
 * lost peer fails closed, loudly, instead of hanging the device or returning
 * numbers from a half exchange.
 */

#include "pulsar_cuda_internal.h"

#include <cuda.h>
#include <stdint.h>

/// st.release.sys of one u64: every earlier write of this thread is visible system-wide before it.
/// @param p device (slab-mapped) address
/// @param v the value
static __device__ __forceinline__ void tp_st_release_sys(uint64_t *p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

/* The combines run behind the stream wait; a set error word means the wait
 * was RELEASED by a failed lane, not satisfied by the peer's rows, so the
 * combine leaves its destination alone.  Block-uniform. */
/// @param err the lane's mapped error word
/// @return true when the lane failed (the wait was released, not satisfied): leave the destination alone
static __device__ __forceinline__ bool tp_lane_failed(const uint32_t *err) {
    return *(const volatile uint32_t *)err != 0u;
}

/* Stage + publish in ONE launch.  Every block copies its rows into the slab's
 * out-slots; with ADD the element staged is src + addend -- the same single
 * f32 add, same operand order, as the engine's add_kernel it replaces (both
 * TUs build with the same NVCCFLAGS) -- written back to src as the combine's
 * own operand, and the addend is zeroed, the fill the caller's consumers
 * expect.  Then a last-block-done ticket: each block's thread 0, after the
 * block barrier (which orders every thread's slab stores before it), fences
 * at system scope and takes a ticket; the block that draws the last one
 * fences again (acquiring every other block's fenced writes through the
 * ticket's RMW chain), resets the ticket for the next exchange on this
 * stream, and publishes exactly what the old <<<1,1>>> publish kernel did:
 * desc[1], desc[2], then desc[0] = exch with st.release.sys.  The proxy sees
 * the same descriptor sequence over the same slab bytes. */
/** Stage + publish.  Grid tp_grid(rows * vec_floats), block 256, no smem; grid-stride over the elements.
 *  @tparam ADD fold `addend` in first (src = src + addend, addend = 0)
 *  @param src        device f32 rows [rows][vec_floats]; with ADD, rewritten with the staged sum
 *  @param addend     device f32 rows like src (ADD only), zeroed
 *  @param slab       device mapping of the slab
 *  @param out_off    byte offset of the out-slots in the slab
 *  @param vec_floats floats a message (one row)
 *  @param first_msg  the first row's message sequence number (slot (seq - 1) % n_slots)
 *  @param n_slots    the slot ring's size
 *  @param rows       rows to stage
 *  @param desc       the descriptor words: [0] exchange id (written last), [1] first_msg, [2] rows
 *  @param exch       the exchange id (nonzero)
 *  @param ticket     device u32 last-block ticket, zero on entry and on exit */
template <bool ADD>
static __global__ void tp_stage_publish_kernel(float *src, float *addend, uint8_t *slab,
                                               uint64_t out_off, uint64_t vec_floats,
                                               uint64_t first_msg, uint32_t n_slots,
                                               uint32_t rows, uint64_t *desc, uint64_t exch,
                                               unsigned int *ticket) {
    const uint64_t n = (uint64_t)rows * vec_floats;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = i / vec_floats, j = i - r * vec_floats;
        const uint64_t slot = (first_msg + r - 1u) % n_slots;
        float *dst = (float *)(slab + out_off) + slot * vec_floats;
        float v = src[i];
        if constexpr (ADD) {
            v = v + addend[i];
            src[i] = v;
            addend[i] = 0.0f;
        }
        dst[j] = v;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    __threadfence_system();
    if (atomicAdd(ticket, 1u) != gridDim.x - 1u) return;
    __threadfence_system();
    *ticket = 0u;
    desc[1] = first_msg;
    desc[2] = rows;
    tp_st_release_sys(&desc[0], exch);
}

/** dst[r][j] = dst[r][j] + peer[r][j] (own first).  Grid tp_grid(rows * vec_floats), block 256; runs behind the
 *  stream wait and does nothing when the lane failed.
 *  @param dst        device f32 rows [rows][vec_floats], this rank's partial (in / out)
 *  @param slab       device mapping of the slab
 *  @param in_off     byte offset of the in-slots in the slab
 *  @param vec_floats floats a message (one row)
 *  @param first_msg  the first row's message sequence number
 *  @param n_slots    the slot ring's size
 *  @param rows       rows to combine
 *  @param err        the lane's mapped error word */
static __global__ void tp_combine_sum_kernel(float *dst, const uint8_t *slab,
                                             uint64_t in_off, uint64_t vec_floats,
                                             uint64_t first_msg, uint32_t n_slots,
                                             uint32_t rows, const uint32_t *err) {
    if (tp_lane_failed(err)) return;
    const uint64_t n = (uint64_t)rows * vec_floats;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = i / vec_floats, j = i - r * vec_floats;
        const uint64_t slot = (first_msg + r - 1u) % n_slots;
        const float *peer = (const float *)(slab + in_off) + slot * vec_floats;
        dst[i] = dst[i] + __ldcv(peer + j);
    }
}

/* The vocab gather's two halves (4g-2).  A head slice travels as a packed
 * payload -- row r of this rank's `width` logits at r * width -- cut into
 * vec-sized messages.  Each chunk's combine runs behind the stream wait like the
 * sum does, then scatters payload element e (chunk start elem0 + i) to row
 * e / width, column col0 + e % width of the pitched destination; the own
 * slice is scattered the same way with no wait.  Copies only: the assembled
 * rows hold exactly the bytes the heads wrote. */
/** The vocab gather's peer half: payload element e = elem0 + i of `msgs` messages to dst[e / width][col0 + e %
 *  width].  Grid tp_grid(msgs * vec_floats), block 256; behind the stream wait; nothing when the lane failed.
 *  @param dst        device f32 destination rows, pitch `pitch` floats
 *  @param slab       device mapping of the slab
 *  @param in_off     byte offset of the in-slots in the slab
 *  @param vec_floats floats a message
 *  @param first_msg  the chunk's first message sequence number
 *  @param n_slots    the slot ring's size
 *  @param msgs       messages in this chunk
 *  @param elem0      payload element the chunk starts at
 *  @param n_elem     payload elements in all (rows * width); the tail of the last message is ignored
 *  @param width      the peer's columns a row
 *  @param pitch      destination row pitch, in floats
 *  @param col0       the peer's first destination column
 *  @param err        the lane's mapped error word */
static __global__ void tp_combine_scatter_kernel(float *dst, const uint8_t *slab,
                                                 uint64_t in_off, uint64_t vec_floats,
                                                 uint64_t first_msg, uint32_t n_slots,
                                                 uint32_t msgs, uint64_t elem0,
                                                 uint64_t n_elem, uint64_t width,
                                                 uint64_t pitch, uint64_t col0,
                                                 const uint32_t *err) {
    if (tp_lane_failed(err)) return;
    const uint64_t n = (uint64_t)msgs * vec_floats;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t e = elem0 + i;
        if (e >= n_elem) break;
        const uint64_t m = i / vec_floats, j = i - m * vec_floats;
        const uint64_t slot = (first_msg + m - 1u) % n_slots;
        const float *peer = (const float *)(slab + in_off) + slot * vec_floats;
        const uint64_t r = e / width;
        dst[r * pitch + col0 + (e - r * width)] = __ldcv(peer + j);
    }
}

/* The row-split gather's combine (4g-3): after the wait, payload element i
 * belongs to the block whose [elem0, elem0 + rows*width) holds it; both the
 * own copy (from the packed own payload) and the peer's (from the in-slot the
 * row lane landed it in) are written to their columns of the full rows. */
/// The row-split gather's blocks, by value for tp_combine_gather_blocks_kernel (n of PULSAR_TP_GATHER_BLOCKS_MAX).
typedef struct {
    /// per block: the full f32 output rows (device)
    float *dst[PULSAR_TP_GATHER_BLOCKS_MAX];
    /// per block: its first payload element; elem0[n] = the payload's total elements
    uint64_t elem0[PULSAR_TP_GATHER_BLOCKS_MAX + 1];
    /// per block: columns each rank owns, per row
    uint64_t width[PULSAR_TP_GATHER_BLOCKS_MAX], pitch[PULSAR_TP_GATHER_BLOCKS_MAX];
    /** @var pitch
     *  per block: the output row pitch, in floats */
    /// per block: the column this rank's own columns start at
    uint64_t own_col0[PULSAR_TP_GATHER_BLOCKS_MAX], peer_col0[PULSAR_TP_GATHER_BLOCKS_MAX];
    /** @var peer_col0
     *  per block: the column the peer's columns start at */
    /// blocks in use
    uint32_t n;
} tp_gather_blocks_t;

/** The row-split gather's combine: both halves of every block into its destination rows.  Grid tp_grid(total
 *  payload elements), block 256; behind the stream wait; nothing when the lane failed.
 *  @param B          the blocks (by value)
 *  @param own        device f32 own packed payload [B.elem0[B.n]]
 *  @param slab       device mapping of the slab
 *  @param in_off     byte offset of the in-slots in the slab
 *  @param vec_floats floats a message
 *  @param first_msg  the payload's first message sequence number
 *  @param n_slots    the slot ring's size
 *  @param err        the lane's mapped error word */
static __global__ void tp_combine_gather_blocks_kernel(tp_gather_blocks_t B, const float *own,
                                                       const uint8_t *slab, uint64_t in_off,
                                                       uint64_t vec_floats, uint64_t first_msg,
                                                       uint32_t n_slots, const uint32_t *err) {
    if (tp_lane_failed(err)) return;
    const uint64_t n = B.elem0[B.n];
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t b = 0;
        while (b + 1u < B.n && i >= B.elem0[b + 1u]) b++;
        const uint64_t e = i - B.elem0[b];
        const uint64_t r = e / B.width[b], c = e - r * B.width[b];
        const uint64_t m = i / vec_floats, j = i - m * vec_floats;
        const uint64_t slot = (first_msg + m - 1u) % n_slots;
        const float *peer = (const float *)(slab + in_off) + slot * vec_floats;
        float *row = B.dst[b] + r * B.pitch[b];
        row[B.own_col0[b] + c] = own[i];
        row[B.peer_col0[b] + c] = __ldcv(peer + j);
    }
}

/** The own half of a gather: packed src [rows][width] to dst[r * pitch + col0 + c].  Grid tp_grid(n_elem), block 256.
 *  @param dst    device f32 destination rows
 *  @param src    device f32 packed rows
 *  @param n_elem rows * width
 *  @param width  columns a row in src
 *  @param pitch  destination row pitch, in floats
 *  @param col0   first destination column */
static __global__ void tp_scatter_cols_kernel(float *dst, const float *src, uint64_t n_elem,
                                              uint64_t width, uint64_t pitch, uint64_t col0) {
    for (uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; e < n_elem;
         e += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = e / width;
        dst[r * pitch + col0 + (e - r * width)] = src[e];
    }
}

/* The bulk lane's publish: word 1 = bytes, word 2 = flag | receive buffer. */
/** <<<1, 1>>>: desc[1] = bytes, desc[2] = word2, then desc[0] = exch with st.release.sys after a system fence.
 *  @param desc  the descriptor words (slab-mapped)
 *  @param exch  the exchange id (nonzero)
 *  @param bytes the exchange's payload bytes
 *  @param word2 the bulk flag | the receive buffer */
static __global__ void tp_publish_bulk_kernel(uint64_t *desc, uint64_t exch, uint64_t bytes,
                                              uint64_t word2) {
    __threadfence_system();
    desc[1] = bytes;
    desc[2] = word2;
    tp_st_release_sys(&desc[0], exch);
}

/* The bulk lane's stage (L260): this rank's partial written by SMs straight
 * into the mapped out-region -- src + addend when there is an addend (the
 * engine's add_kernel arithmetic), which is then zeroed, exactly what the row
 * lane's stage kernel does.  On GB10 an SM store into the registered buffer
 * runs at device-memory speed (0.31 ms per 32 MiB) where the copy engine took
 * 0.56 ms, and the fold no longer costs an add pass + a fill pass of its own. */
/** Bulk stage: out[i] = src[i] (+ addend[i], then addend[i] = 0).  Grid tp_bulk_grid(n), block 256.
 *  @tparam ADD fold the addend in
 *  @param out    the mapped bulk out-region (f32)
 *  @param src    device f32 source
 *  @param addend device f32 addend (ADD only), zeroed
 *  @param n      elements */
template <bool ADD>
static __global__ void tp_bulk_stage_kernel(float *out, const float *src, float *addend, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        if (ADD) {
            out[i] = src[i] + addend[i];
            addend[i] = 0.0f;
        } else {
            out[i] = src[i];
        }
    }
}

/* The bulk lane's combine: dst[i] = own[i] + peer[i] behind the stream wait (the proxy reported
 * the exchange done), own being this rank's staged partial in the out-region --
 * the all-reduce's own + peer, one add per element, bit-identical to the host
 * big gate it replaces (acc[i] += peer[i]). */
/** Bulk combine: dst[i] = own[i] + peer[i].  Grid tp_bulk_grid(n), block 256; behind the stream wait; nothing when
 *  the lane failed.
 *  @param dst  device f32 destination
 *  @param own  this rank's staged partial (the out-region)
 *  @param peer the peer's rows (the mapped receive buffer)
 *  @param n    elements
 *  @param err  the lane's mapped error word */
static __global__ void tp_bulk_combine_sum_kernel(float *dst, const float *own, const float *peer,
                                                  uint64_t n, const uint32_t *err) {
    if (tp_lane_failed(err)) return;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] = own[i] + __ldcv(peer + i);
}

/// @param n elements
/// @return blocks of 256 threads for the bulk kernels: ceil(n / 256), capped at 2048
static unsigned tp_bulk_grid(uint64_t n) {
    const uint64_t b = (n + 255u) / 256u;
    return (unsigned)(b > 2048u ? 2048u : b);
}

/// @param n elements
/// @return blocks of 256 threads for the row-lane kernels: ceil(n / 256) in 1 .. 256
static unsigned tp_grid(uint64_t n) {
    const uint64_t b = (n + 255u) / 256u;
    return (unsigned)(b > 256u ? 256u : (b ? b : 1u));
}

/* cuStreamWaitValue64, resolved once through the runtime (the engine links no
 * libcuda).  pulsar_gpu_tp_wait_init runs at the row lane's bring-up and
 * refuses a device without 64-bit stream memory operations; the lane has no
 * other way to wait. */
/// cuStreamWaitValue64's signature.
typedef CUresult (*tp_wait_value_fn)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
/// cuDeviceGetAttribute's signature.
typedef CUresult (*tp_dev_attr_fn)(int *, CUdevice_attribute, CUdevice);
/// cuStreamWaitValue64, resolved by pulsar_gpu_tp_wait_init; NULL until then.
static tp_wait_value_fn g_tp_wait_value = NULL;

/** Resolves cuStreamWaitValue64 through the runtime and checks the device has 64-bit stream memory operations.
 *  @param err    out: the reason, on refusal
 *  @param errlen bytes at err
 *  @return 1 ready (also when already resolved), 0 the lane cannot run here */
int pulsar_gpu_tp_wait_init(char *err, size_t errlen) {
    if (g_tp_wait_value) return 1;
    void *wv = NULL, *ga = NULL;
    cudaDriverEntryPointQueryResult q1 = cudaDriverEntryPointSymbolNotFound, q2 = q1;
    int dev = 0, can64 = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDriverEntryPointByVersion("cuStreamWaitValue64", &wv, 12000, cudaEnableDefault, &q1) != cudaSuccess ||
        cudaGetDriverEntryPointByVersion("cuDeviceGetAttribute", &ga, 12000, cudaEnableDefault, &q2) != cudaSuccess ||
        q1 != cudaDriverEntryPointSuccess || q2 != cudaDriverEntryPointSuccess || !wv || !ga) {
        snprintf(err, errlen, "tp row lane: the driver exports no cuStreamWaitValue64 / cuDeviceGetAttribute");
        return 0;
    }
    if (((tp_dev_attr_fn)ga)(&can64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, (CUdevice)dev) !=
            CUDA_SUCCESS || !can64) {
        snprintf(err, errlen, "tp row lane: device %d has no 64-bit stream memory operations "
                              "(the lane's stream wait)", dev);
        return 0;
    }
    g_tp_wait_value = (tp_wait_value_fn)wv;
    return 1;
}

/* Hold this thread's stream until done >= exch (GEQ compares cyclically, as
 * int64(done - exch) >= 0: the release bit a failed lane ORs in satisfies it). */
/** @param done_dev the lane's device-mapped done word
 *  @param exch     the exchange id to wait for
 *  @return 1 the wait was queued on the per-thread stream, 0 refused (named on stderr) */
static int tp_stream_wait_done(const void *done_dev, uint64_t exch) {
    if (!g_tp_wait_value) {
        fprintf(stderr, "pulsar: tp row lane stream wait used before pulsar_gpu_tp_wait_init -- refusing\n");
        return 0;
    }
    const CUresult r = g_tp_wait_value(CU_STREAM_PER_THREAD, (CUdeviceptr)(uintptr_t)done_dev, exch,
                                       CU_STREAM_WAIT_VALUE_GEQ);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "pulsar: tp row lane stream wait (exchange %llu) failed: CUresult %d\n",
                (unsigned long long)exch, (int)r);
        return 0;
    }
    return 1;
}

/** Stage + publish one row-lane exchange (contract in pulsar_gpu.h).
 *  @param src       f32 rows [rows][vec_bytes / 4] (rewritten with the sum when addend is set)
 *  @param addend    f32 rows folded in first and zeroed, or NULL
 *  @param slab_dev  device mapping of the slab
 *  @param out_off   byte offset of the out-slots
 *  @param vec_bytes bytes a message (one row)
 *  @param first_msg the first row's message sequence number
 *  @param n_slots   the slot ring's size
 *  @param rows      rows to stage
 *  @param desc_dev  the descriptor words
 *  @param exch      the exchange id (nonzero)
 *  @param ticket    a zeroed device u32, one per stream
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_stage_publish(pulsar_gpu_tensor *src, pulsar_gpu_tensor *addend,
                                void *slab_dev, uint64_t out_off, uint64_t vec_bytes,
                                uint64_t first_msg, uint32_t n_slots, uint32_t rows,
                                void *desc_dev, uint64_t exch, pulsar_gpu_tensor *ticket) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!src || !slab_dev || !desc_dev || exch == 0 || !ticket ||
        ticket->bytes < sizeof(unsigned int) || rows == 0 || n_slots == 0 || vf == 0 ||
        (uint64_t)rows * vec_bytes > src->bytes ||
        (addend && (uint64_t)rows * vec_bytes > addend->bytes)) return 0;
    const unsigned grid = tp_grid((uint64_t)rows * vf);
    if (addend)
        tp_stage_publish_kernel<true><<<grid, 256>>>(
            (float *)src->ptr, (float *)addend->ptr, (uint8_t *)slab_dev, out_off, vf, first_msg,
            n_slots, rows, (uint64_t *)desc_dev, exch, (unsigned int *)ticket->ptr);
    else
        tp_stage_publish_kernel<false><<<grid, 256>>>(
            (float *)src->ptr, NULL, (uint8_t *)slab_dev, out_off, vf, first_msg,
            n_slots, rows, (uint64_t *)desc_dev, exch, (unsigned int *)ticket->ptr);
    return cuda_ok(cudaGetLastError(), "tp stage+publish launch");
}

/** Waits for done >= exch on the stream, then dst += peer rows.
 *  @param dst       f32 rows [rows][vec_bytes / 4]
 *  @param slab_dev  device mapping of the slab
 *  @param in_off    byte offset of the in-slots
 *  @param vec_bytes bytes a message
 *  @param first_msg the first row's message sequence number
 *  @param n_slots   the slot ring's size
 *  @param rows      rows to combine
 *  @param done_dev  the device-mapped done word
 *  @param exch      the exchange id
 *  @param err_dev   the device-mapped error word
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_combine_sum(pulsar_gpu_tensor *dst, const void *slab_dev,
                              uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                              uint32_t n_slots, uint32_t rows, const void *done_dev,
                              uint64_t exch, const void *err_dev) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!dst || !slab_dev || !done_dev || !err_dev || rows == 0 || n_slots == 0 || vf == 0 ||
        (uint64_t)rows * vec_bytes > dst->bytes) return 0;
    if (!tp_stream_wait_done(done_dev, exch)) return 0;
    tp_combine_sum_kernel<<<tp_grid((uint64_t)rows * vf), 256>>>(
        (float *)dst->ptr, (const uint8_t *)slab_dev, in_off, vf, first_msg, n_slots, rows,
        (const uint32_t *)err_dev);
    return cuda_ok(cudaGetLastError(), "tp combine sum launch");
}

/** Waits for done >= exch, then scatters this chunk of the peer's packed [rows][width] slice into dst.
 *  @param dst       f32 destination [rows][pitch]
 *  @param slab_dev  device mapping of the slab
 *  @param in_off    byte offset of the in-slots
 *  @param vec_bytes bytes a message
 *  @param first_msg the chunk's first message sequence number
 *  @param n_slots   the slot ring's size
 *  @param msgs      messages in the chunk
 *  @param elem0     payload element the chunk starts at (< rows * width)
 *  @param rows      payload rows
 *  @param width     the peer's columns a row
 *  @param pitch     destination row pitch, in floats
 *  @param col0      the peer's first destination column (col0 + width <= pitch)
 *  @param done_dev  the device-mapped done word
 *  @param exch      the exchange id
 *  @param err_dev   the device-mapped error word
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_combine_scatter(pulsar_gpu_tensor *dst, const void *slab_dev,
                                  uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                                  uint32_t n_slots, uint32_t msgs, uint64_t elem0,
                                  uint32_t rows, uint64_t width, uint64_t pitch, uint64_t col0,
                                  const void *done_dev, uint64_t exch, const void *err_dev) {
    const uint64_t vf = vec_bytes / sizeof(float);
    const uint64_t n_elem = (uint64_t)rows * width;
    if (!dst || !slab_dev || !done_dev || !err_dev || msgs == 0 || n_slots == 0 || vf == 0 ||
        width == 0 || col0 + width > pitch || elem0 >= n_elem ||
        (uint64_t)rows * pitch * sizeof(float) > dst->bytes) return 0;
    if (!tp_stream_wait_done(done_dev, exch)) return 0;
    tp_combine_scatter_kernel<<<tp_grid((uint64_t)msgs * vf), 256>>>(
        (float *)dst->ptr, (const uint8_t *)slab_dev, in_off, vf, first_msg, n_slots, msgs,
        elem0, n_elem, width, pitch, col0, (const uint32_t *)err_dev);
    return cuda_ok(cudaGetLastError(), "tp combine scatter launch");
}

/** The own half: packed src [rows][width] to dst[r * pitch + col0 + c], no wait.
 *  @param dst   f32 destination [rows][pitch]
 *  @param src   f32 packed rows
 *  @param rows  rows
 *  @param width columns a row
 *  @param pitch destination row pitch, in floats
 *  @param col0  first destination column (col0 + width <= pitch)
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_scatter_cols(pulsar_gpu_tensor *dst, const pulsar_gpu_tensor *src,
                               uint32_t rows, uint64_t width, uint64_t pitch, uint64_t col0) {
    const uint64_t n_elem = (uint64_t)rows * width;
    if (!dst || !src || rows == 0 || width == 0 || col0 + width > pitch ||
        n_elem * sizeof(float) > src->bytes ||
        (uint64_t)rows * pitch * sizeof(float) > dst->bytes) return 0;
    tp_scatter_cols_kernel<<<tp_grid(n_elem), 256>>>(
        (float *)dst->ptr, (const float *)src->ptr, n_elem, width, pitch, col0);
    return cuda_ok(cudaGetLastError(), "tp scatter cols launch");
}

/** Bulk stage: `bytes` of src (+ addend, then zeroed) at byte offset `off` into the mapped out-region.
 *  @param src     f32 source tensor
 *  @param addend  f32 addend tensor read at the same offset, or NULL
 *  @param off     byte offset into src / addend (a multiple of 4)
 *  @param out_dev the mapped bulk out-region
 *  @param bytes   bytes to stage (a nonzero multiple of 4)
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_bulk_stage(const pulsar_gpu_tensor *src, pulsar_gpu_tensor *addend, uint64_t off,
                             void *out_dev, uint64_t bytes) {
    if (!src || !out_dev || bytes == 0 || bytes % sizeof(float) != 0 || off % sizeof(float) != 0 ||
        off > src->bytes || bytes > src->bytes - off ||
        (addend && (off > addend->bytes || bytes > addend->bytes - off))) return 0;
    const uint64_t n = bytes / sizeof(float);
    const float *s = (const float *)((const uint8_t *)src->ptr + off);
    if (addend)
        tp_bulk_stage_kernel<true><<<tp_bulk_grid(n), 256>>>(
            (float *)out_dev, s, (float *)((uint8_t *)addend->ptr + off), n);
    else
        tp_bulk_stage_kernel<false><<<tp_bulk_grid(n), 256>>>((float *)out_dev, s, NULL, n);
    return cuda_ok(cudaGetLastError(), "tp bulk stage launch");
}

/** Publishes the bulk descriptor {exch, bytes, word2}.
 *  @param desc_dev the descriptor words
 *  @param exch     the exchange id (nonzero)
 *  @param bytes    the payload bytes (nonzero)
 *  @param word2    the bulk flag | the receive buffer
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_publish_bulk(void *desc_dev, uint64_t exch, uint64_t bytes, uint64_t word2) {
    if (!desc_dev || exch == 0 || bytes == 0) return 0;
    tp_publish_bulk_kernel<<<1, 1>>>((uint64_t *)desc_dev, exch, bytes, word2);
    return cuda_ok(cudaGetLastError(), "tp publish bulk launch");
}

/** Waits for done >= exch, then dst[dst_off..] = own + peer.
 *  @param dst      f32 destination tensor
 *  @param dst_off  byte offset into dst (a multiple of 4)
 *  @param own_dev  this rank's staged partial (the out-region)
 *  @param peer_dev the peer's rows (the mapped receive buffer)
 *  @param bytes    bytes to combine (a nonzero multiple of 4)
 *  @param done_dev the device-mapped done word
 *  @param exch     the exchange id
 *  @param err_dev  the device-mapped error word
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_bulk_combine_sum(pulsar_gpu_tensor *dst, uint64_t dst_off, const void *own_dev,
                                   const void *peer_dev, uint64_t bytes, const void *done_dev,
                                   uint64_t exch, const void *err_dev) {
    if (!dst || !own_dev || !peer_dev || !done_dev || !err_dev || bytes == 0 ||
        bytes % sizeof(float) != 0 || dst_off % sizeof(float) != 0 || dst_off > dst->bytes ||
        bytes > dst->bytes - dst_off) return 0;
    const uint64_t n = bytes / sizeof(float);
    if (!tp_stream_wait_done(done_dev, exch)) return 0;
    tp_bulk_combine_sum_kernel<<<tp_bulk_grid(n), 256>>>((float *)((uint8_t *)dst->ptr + dst_off),
                                                         (const float *)own_dev,
                                                         (const float *)peer_dev, n,
                                                         (const uint32_t *)err_dev);
    return cuda_ok(cudaGetLastError(), "tp bulk combine sum launch");
}

/** Waits for done >= exch, then scatters both halves of every gather block into its destination rows.
 *  @param blocks    the blocks (contiguous payload: block k's elem0 = the sum of the earlier rows * width)
 *  @param n_blocks  blocks, 1 .. PULSAR_TP_GATHER_BLOCKS_MAX
 *  @param rows      rows of every block
 *  @param own       f32 own packed payload
 *  @param slab_dev  device mapping of the slab
 *  @param in_off    byte offset of the in-slots
 *  @param vec_bytes bytes a message
 *  @param first_msg the payload's first message sequence number
 *  @param n_slots   the slot ring's size
 *  @param done_dev  the device-mapped done word
 *  @param exch      the exchange id
 *  @param err_dev   the device-mapped error word
 *  @return 1 launched, 0 refused or failed */
int pulsar_gpu_tp_combine_gather_blocks(const pulsar_tp_gather_block *blocks, uint32_t n_blocks,
                                        uint32_t rows, const pulsar_gpu_tensor *own,
                                        const void *slab_dev, uint64_t in_off, uint64_t vec_bytes,
                                        uint64_t first_msg, uint32_t n_slots, const void *done_dev,
                                        uint64_t exch, const void *err_dev) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!blocks || n_blocks == 0 || n_blocks > PULSAR_TP_GATHER_BLOCKS_MAX || rows == 0 || !own ||
        !slab_dev || !done_dev || !err_dev || vf == 0 || n_slots == 0) return 0;
    tp_gather_blocks_t B;
    memset(&B, 0, sizeof(B));
    uint64_t e0 = 0;
    for (uint32_t k = 0; k < n_blocks; k++) {
        const pulsar_tp_gather_block *bk = &blocks[k];
        if (!bk->dst || bk->width == 0 || bk->elem0 != e0 ||
            bk->own_col0 + bk->width > bk->pitch || bk->peer_col0 + bk->width > bk->pitch ||
            (uint64_t)rows * bk->pitch * sizeof(float) > bk->dst->bytes) return 0;
        B.dst[k] = (float *)bk->dst->ptr;
        B.elem0[k] = bk->elem0;
        B.width[k] = bk->width;
        B.pitch[k] = bk->pitch;
        B.own_col0[k] = bk->own_col0;
        B.peer_col0[k] = bk->peer_col0;
        e0 += (uint64_t)rows * bk->width;
    }
    B.elem0[n_blocks] = e0;
    B.n = n_blocks;
    if (e0 * sizeof(float) > own->bytes) return 0;
    if (!tp_stream_wait_done(done_dev, exch)) return 0;
    tp_combine_gather_blocks_kernel<<<tp_grid(e0), 256>>>(B, (const float *)own->ptr,
                                                          (const uint8_t *)slab_dev, in_off, vf,
                                                          first_msg, n_slots, (const uint32_t *)err_dev);
    return cuda_ok(cudaGetLastError(), "tp combine gather blocks launch");
}

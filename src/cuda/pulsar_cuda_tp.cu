/* Tensor-parallel row-lane kernels (L241 4g-2): the GPU half of the pair's
 * asynchronous decode/verify exchange.
 *
 * The slab is host-pinned and GPU-mapped (pulsar_tp_gpu.h); on GB10 it is
 * coherent with the CPU, which runs the transport's proxy thread.  Per
 * exchange the stream carries two kernels, and the engine thread never
 * waits:
 *
 *   stage    this rank's rows -> the slab's out-slots (message seq t lives in
 *            slot (t-1) % n_slots, the transport's ring rule); the LAST block
 *            to finish publishes the descriptor {exchange id, first message,
 *            rows}, the id written LAST, release/system scope -- the proxy's
 *            go signal
 *   combine  spin on the done word until the proxy reports the exchange
 *            complete (the peer's rows landed and our sends retired), then
 *            own + peer, one add per element, own first -- the all-reduce's
 *            arithmetic, bit-identical to the host lane it replaced.
 *
 * A spin that outlives the transport's timeout writes the error word and
 * leaves the destination untouched; the engine refuses at its next host
 * sync (pulsar_tp_row_lane_check), so a lost peer fails closed, loudly,
 * instead of hanging the device or returning numbers from a half exchange.
 */

#include "pulsar_cuda_internal.h"

#include <stdint.h>

static __device__ __forceinline__ uint64_t tp_ld_acquire_sys(const uint64_t *p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ void tp_st_release_sys(uint64_t *p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

static __device__ __forceinline__ uint64_t tp_globaltimer_ns(void) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/* Block-wide wait for done >= exch.  Returns false when the error word is
 * set -- by an earlier kernel's timeout, or by the host on a row-lane abort
 * (pulsar_tp_row_lane_abort: the peer will not run this step) -- or when the
 * timeout expires first, which latches it. */
static __device__ bool tp_wait_done(const uint64_t *done, uint64_t exch,
                                    uint32_t *err, uint64_t timeout_ns) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t t0 = tp_globaltimer_ns();
        ok = 1;
        while (tp_ld_acquire_sys(done) < exch) {
            if (*(volatile uint32_t *)err != 0u) {
                ok = 0;
                break;
            }
            if (tp_globaltimer_ns() - t0 > timeout_ns) {
                atomicExch(err, 1u);
                ok = 0;
                break;
            }
            __nanosleep(200);
        }
    }
    __syncthreads();
    return ok != 0;
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

static __global__ void tp_combine_sum_kernel(float *dst, const uint8_t *slab,
                                             uint64_t in_off, uint64_t vec_floats,
                                             uint64_t first_msg, uint32_t n_slots,
                                             uint32_t rows, const uint64_t *done,
                                             uint64_t exch, uint32_t *err,
                                             uint64_t timeout_ns) {
    if (!tp_wait_done(done, exch, err, timeout_ns)) return;
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
 * vec-sized messages.  Each chunk's combine waits on the done word like the
 * sum does, then scatters payload element e (chunk start elem0 + i) to row
 * e / width, column col0 + e % width of the pitched destination; the own
 * slice is scattered the same way with no wait.  Copies only: the assembled
 * rows hold exactly the bytes the heads wrote. */
static __global__ void tp_combine_scatter_kernel(float *dst, const uint8_t *slab,
                                                 uint64_t in_off, uint64_t vec_floats,
                                                 uint64_t first_msg, uint32_t n_slots,
                                                 uint32_t msgs, uint64_t elem0,
                                                 uint64_t n_elem, uint64_t width,
                                                 uint64_t pitch, uint64_t col0,
                                                 const uint64_t *done, uint64_t exch,
                                                 uint32_t *err, uint64_t timeout_ns) {
    if (!tp_wait_done(done, exch, err, timeout_ns)) return;
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

static __global__ void tp_scatter_cols_kernel(float *dst, const float *src, uint64_t n_elem,
                                              uint64_t width, uint64_t pitch, uint64_t col0) {
    for (uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; e < n_elem;
         e += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = e / width;
        dst[r * pitch + col0 + (e - r * width)] = src[e];
    }
}

/* The bulk lane's publish: word 1 = bytes, word 2 = flag | receive buffer. */
static __global__ void tp_publish_bulk_kernel(uint64_t *desc, uint64_t exch, uint64_t bytes,
                                              uint64_t word2) {
    __threadfence_system();
    desc[1] = bytes;
    desc[2] = word2;
    tp_st_release_sys(&desc[0], exch);
}

/* The bulk lane's combine: dst[i] = dst[i] + peer[i] once the proxy reports
 * the exchange done -- the all-reduce's own + peer, one add per element,
 * bit-identical to the host big gate it replaces (acc[i] += peer[i]). */
static __global__ void tp_bulk_combine_sum_kernel(float *dst, const float *peer, uint64_t n,
                                                  const uint64_t *done, uint64_t exch,
                                                  uint32_t *err, uint64_t timeout_ns) {
    if (!tp_wait_done(done, exch, err, timeout_ns)) return;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] = dst[i] + __ldcv(peer + i);
}

static unsigned tp_grid(uint64_t n) {
    const uint64_t b = (n + 255u) / 256u;
    return (unsigned)(b > 256u ? 256u : (b ? b : 1u));
}

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

int pulsar_gpu_tp_combine_sum(pulsar_gpu_tensor *dst, const void *slab_dev,
                              uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                              uint32_t n_slots, uint32_t rows, const void *done_dev,
                              uint64_t exch, void *err_dev, uint64_t timeout_ns) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!dst || !slab_dev || !done_dev || !err_dev || rows == 0 || n_slots == 0 || vf == 0 ||
        (uint64_t)rows * vec_bytes > dst->bytes) return 0;
    tp_combine_sum_kernel<<<tp_grid((uint64_t)rows * vf), 256>>>(
        (float *)dst->ptr, (const uint8_t *)slab_dev, in_off, vf, first_msg, n_slots, rows,
        (const uint64_t *)done_dev, exch, (uint32_t *)err_dev, timeout_ns);
    return cuda_ok(cudaGetLastError(), "tp combine sum launch");
}

int pulsar_gpu_tp_combine_scatter(pulsar_gpu_tensor *dst, const void *slab_dev,
                                  uint64_t in_off, uint64_t vec_bytes, uint64_t first_msg,
                                  uint32_t n_slots, uint32_t msgs, uint64_t elem0,
                                  uint32_t rows, uint64_t width, uint64_t pitch, uint64_t col0,
                                  const void *done_dev, uint64_t exch, void *err_dev,
                                  uint64_t timeout_ns) {
    const uint64_t vf = vec_bytes / sizeof(float);
    const uint64_t n_elem = (uint64_t)rows * width;
    if (!dst || !slab_dev || !done_dev || !err_dev || msgs == 0 || n_slots == 0 || vf == 0 ||
        width == 0 || col0 + width > pitch || elem0 >= n_elem ||
        (uint64_t)rows * pitch * sizeof(float) > dst->bytes) return 0;
    tp_combine_scatter_kernel<<<tp_grid((uint64_t)msgs * vf), 256>>>(
        (float *)dst->ptr, (const uint8_t *)slab_dev, in_off, vf, first_msg, n_slots, msgs,
        elem0, n_elem, width, pitch, col0, (const uint64_t *)done_dev, exch,
        (uint32_t *)err_dev, timeout_ns);
    return cuda_ok(cudaGetLastError(), "tp combine scatter launch");
}

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

int pulsar_gpu_tp_bulk_stage(const pulsar_gpu_tensor *src, uint64_t src_off, void *dst_dev,
                             uint64_t bytes) {
    if (!src || !dst_dev || bytes == 0 || src_off > src->bytes || bytes > src->bytes - src_off) return 0;
    return cuda_ok(cudaMemcpyAsync(dst_dev, (const uint8_t *)src->ptr + src_off, (size_t)bytes,
                                   cudaMemcpyDefault, cudaStreamPerThread),
                   "tp bulk stage");
}

int pulsar_gpu_tp_publish_bulk(void *desc_dev, uint64_t exch, uint64_t bytes, uint64_t word2) {
    if (!desc_dev || exch == 0 || bytes == 0) return 0;
    tp_publish_bulk_kernel<<<1, 1>>>((uint64_t *)desc_dev, exch, bytes, word2);
    return cuda_ok(cudaGetLastError(), "tp publish bulk launch");
}

int pulsar_gpu_tp_bulk_combine_sum(pulsar_gpu_tensor *dst, uint64_t dst_off, const void *peer_dev,
                                   uint64_t bytes, const void *done_dev, uint64_t exch,
                                   void *err_dev, uint64_t timeout_ns) {
    if (!dst || !peer_dev || !done_dev || !err_dev || bytes == 0 || bytes % sizeof(float) != 0 ||
        dst_off % sizeof(float) != 0 || dst_off > dst->bytes || bytes > dst->bytes - dst_off) return 0;
    const uint64_t n = bytes / sizeof(float);
    const uint64_t b = (n + 255u) / 256u;
    const unsigned grid = (unsigned)(b > 2048u ? 2048u : b);
    tp_bulk_combine_sum_kernel<<<grid, 256>>>((float *)((uint8_t *)dst->ptr + dst_off),
                                              (const float *)peer_dev, n,
                                              (const uint64_t *)done_dev, exch,
                                              (uint32_t *)err_dev, timeout_ns);
    return cuda_ok(cudaGetLastError(), "tp bulk combine sum launch");
}

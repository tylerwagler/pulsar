/* Tensor-parallel row-lane kernels (L241 4g-2): the GPU half of the pair's
 * asynchronous decode/verify exchange.
 *
 * The slab is host-pinned and GPU-mapped (pulsar_tp_gpu.h); on GB10 it is
 * coherent with the CPU, which runs the transport's proxy thread.  Per
 * exchange the stream carries three kernels, and the engine thread never
 * waits:
 *
 *   stage    this rank's rows -> the slab's out-slots (message seq t lives in
 *            slot (t-1) % n_slots, the transport's ring rule)
 *   publish  the descriptor {exchange id, first message, rows}; the id is
 *            written LAST, release/system scope -- the proxy's go signal
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

static __global__ void tp_stage_rows_kernel(const float *src, uint8_t *slab,
                                            uint64_t out_off, uint64_t vec_floats,
                                            uint64_t first_msg, uint32_t n_slots,
                                            uint32_t rows) {
    const uint64_t n = (uint64_t)rows * vec_floats;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = i / vec_floats, j = i - r * vec_floats;
        const uint64_t slot = (first_msg + r - 1u) % n_slots;
        float *dst = (float *)(slab + out_off) + slot * vec_floats;
        dst[j] = src[i];
    }
    /* Each thread's own writes to the mapped slab are ordered before the
     * publish kernel's descriptor at system scope. */
    __threadfence_system();
}

static __global__ void tp_publish_kernel(uint64_t *desc, uint64_t exch,
                                         uint64_t first_msg, uint64_t rows) {
    __threadfence_system();
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

static unsigned tp_grid(uint64_t n) {
    const uint64_t b = (n + 255u) / 256u;
    return (unsigned)(b > 256u ? 256u : (b ? b : 1u));
}

int pulsar_gpu_tp_stage_rows(const pulsar_gpu_tensor *src, void *slab_dev,
                             uint64_t out_off, uint64_t vec_bytes, uint64_t first_msg,
                             uint32_t n_slots, uint32_t rows) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!src || !slab_dev || rows == 0 || n_slots == 0 || vf == 0 ||
        (uint64_t)rows * vec_bytes > src->bytes) return 0;
    tp_stage_rows_kernel<<<tp_grid((uint64_t)rows * vf), 256>>>(
        (const float *)src->ptr, (uint8_t *)slab_dev, out_off, vf, first_msg, n_slots, rows);
    return cuda_ok(cudaGetLastError(), "tp stage rows launch");
}

int pulsar_gpu_tp_publish(void *desc_dev, uint64_t exch, uint64_t first_msg, uint32_t rows) {
    if (!desc_dev || exch == 0) return 0;
    tp_publish_kernel<<<1, 1>>>((uint64_t *)desc_dev, exch, first_msg, rows);
    return cuda_ok(cudaGetLastError(), "tp publish launch");
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

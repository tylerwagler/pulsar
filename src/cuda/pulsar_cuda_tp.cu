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
 *            the SAME arithmetic the host lane did: own + peer (FFN partials,
 *            one add per element, own first) or placement of each rank's
 *            slice at its group offset (attention `low`).  Bit-identical.
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

/* Block-wide wait for done >= exch.  Returns false (and latches the error
 * word) when the timeout expires first. */
static __device__ bool tp_wait_done(const uint64_t *done, uint64_t exch,
                                    uint32_t *err, uint64_t timeout_ns) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t t0 = tp_globaltimer_ns();
        ok = 1;
        while (tp_ld_acquire_sys(done) < exch) {
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

static __global__ void tp_combine_gather_kernel(float *dst, const uint8_t *slab,
                                                uint64_t out_off, uint64_t in_off,
                                                uint64_t vec_floats, uint64_t first_msg,
                                                uint32_t n_slots, uint32_t rows,
                                                uint64_t self_off, uint64_t peer_off,
                                                uint64_t full_floats, const uint64_t *done,
                                                uint64_t exch, uint32_t *err,
                                                uint64_t timeout_ns) {
    if (!tp_wait_done(done, exch, err, timeout_ns)) return;
    const uint64_t n = (uint64_t)rows * vec_floats;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = i / vec_floats, j = i - r * vec_floats;
        const uint64_t slot = (first_msg + r - 1u) % n_slots;
        const float *own  = (const float *)(slab + out_off) + slot * vec_floats;
        const float *peer = (const float *)(slab + in_off) + slot * vec_floats;
        dst[r * full_floats + self_off + j] = __ldcv(own + j);
        dst[r * full_floats + peer_off + j] = __ldcv(peer + j);
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

int pulsar_gpu_tp_combine_gather(pulsar_gpu_tensor *dst, const void *slab_dev,
                                 uint64_t out_off, uint64_t in_off, uint64_t vec_bytes,
                                 uint64_t first_msg, uint32_t n_slots, uint32_t rows,
                                 uint64_t self_off_floats, uint64_t peer_off_floats,
                                 uint64_t full_floats, const void *done_dev, uint64_t exch,
                                 void *err_dev, uint64_t timeout_ns) {
    const uint64_t vf = vec_bytes / sizeof(float);
    if (!dst || !slab_dev || !done_dev || !err_dev || rows == 0 || n_slots == 0 || vf == 0 ||
        self_off_floats + vf > full_floats || peer_off_floats + vf > full_floats ||
        (uint64_t)rows * full_floats * sizeof(float) > dst->bytes) return 0;
    tp_combine_gather_kernel<<<tp_grid((uint64_t)rows * vf), 256>>>(
        (float *)dst->ptr, (const uint8_t *)slab_dev, out_off, in_off, vf, first_msg,
        n_slots, rows, self_off_floats, peer_off_floats, full_floats,
        (const uint64_t *)done_dev, exch, (uint32_t *)err_dev, timeout_ns);
    return cuda_ok(cudaGetLastError(), "tp combine gather launch");
}

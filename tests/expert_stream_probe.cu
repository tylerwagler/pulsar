/* EXPERT STREAM PROBE (L199/L200 candidate #3, picked up for L210).
 *
 * The IQ2 decode expert GEMV reads ~60% of DRAM roofline at ONE row (L208: the
 * gate/up bytes take 4.8 ms and the kernel takes 7.8).  L200's follow-up list
 * left three candidates: (1) the per-iteration __syncthreads(), (2) a deeper
 * weight prefetch (MOOT -- L202 deleted the staged ring), (3) DRAM page locality
 * of the per-expert stride, "which no kernel tuning would fix".  The test that
 * separates them is a streaming microbenchmark: does the ADDRESS ORDER the
 * kernel issues cost anything against a straight line on the same bytes?
 *
 * It allocates one real IQ2 gate/up tensor's worth of experts (256 planes x
 * 2,162,688 B -- the artifact's own per-expert size, measured from its tensor
 * table) and streams the bytes a LAYER-STACK of routed sets moves (43 layers x
 * the set), so every timed pass is ~1-4 GB and DRAM-bound rather than L2-bound.
 * Four orders on the same counted bytes:
 *
 *   linear        a straight ascending sweep (many blocks grid-strided: the
 *                 widest possible spread of open DRAM rows)
 *   linear+bar    the same with a __syncthreads() per 2 KB per block (candidate 1)
 *   planes        the routed experts in ASCENDING expert order, one plane at a
 *                 time -- the kernel's order (its grid is expert-sorted)
 *   interleave    the same planes round-robin at 2 KB -- the page-thrash pattern
 *                 a token-major (unsorted) grid would issue, i.e. candidate 3
 *
 * Comparisons: linear vs planes (does the per-expert stride cost anything?),
 * linear vs linear+bar (is the barrier the limiter?  L201 says no), planes vs
 * interleave (what would a non-expert-sorted grid cost?).
 *
 * Every kernel COUNTS the bytes it actually read (one atomicAdd per block, so the
 * counter cannot drift from the addressing) and the report divides by that count.
 * Model-free and standalone: it needs the GPU and nothing else.
 *
 *   ./tests/expert_stream_probe [experts] [layers] [reps]
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "expert-stream: %s failed: %s\n", #x, cudaGetErrorString(e_)); \
        exit(2); \
    } \
} while (0)

/* The artifact's own numbers (type 44 IQ2_XXS_MMQ_K, dims [4096,2048,256]:
 * 553,648,128 B / 256 experts). */
static const size_t kPlaneBytes = 2162688;

/* Routed sets: slots per token at 1/3/6 rows, and the unique experts the route
 * census measured after 23/30/44% duplicate slots. */
static const int kSlotRows[3] = {1, 3, 6};
static const int kUnique[3] = {6, 13, 20};

static const int kThreads = 256;
static const int kBlocks = 512;

/* Exact: each thread reports the bytes IT read; a warp reduces and one lane per
 * warp does the atomic, so the counter cannot drift from the addressing. */
__device__ __forceinline__ void count_block(unsigned long long *bytes_out, size_t n) {
    for (int off = 16; off > 0; off >>= 1)
        n += (size_t)__shfl_down_sync(0xffffffffu, (unsigned long long)n, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(bytes_out, (unsigned long long)n);
}

/* Keeps the loads: never true, but the compiler cannot prove it. */
__device__ __forceinline__ void keep(uint32_t acc, uint32_t *out) {
    if (acc == 0x9e3779b9u) out[0] = acc;
}

__global__ void stream_linear(const uint4 *buf, size_t n_vec, unsigned long long *moved,
                              uint32_t *out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = 0;
    uint32_t acc = 0;
    for (; i < n_vec; i += stride) {
        const uint4 v = buf[i];
        acc += v.x + v.y + v.z + v.w;
        n += sizeof(uint4);
    }
    keep(acc, out);
    count_block(moved, n);
}

/* Candidate 1, in its STRONGEST form: a barrier on EVERY block iteration (each
 * iteration moves 256 threads x 16 B = 4 KB), which bounds warp spread to one
 * iteration exactly as the kernel's per-iteration __syncthreads() does.  The
 * barrier is unconditional -- a thread-divergent __syncthreads() is undefined
 * (the first version of this probe hung on it). */
__global__ void stream_linear_barrier(const uint4 *buf, size_t n_vec, unsigned long long *moved,
                                      uint32_t *out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t n = 0;
    uint32_t acc = 0;
    for (size_t base = (size_t)blockIdx.x * blockDim.x; base < n_vec; base += stride) {
        const size_t i = base + threadIdx.x;
        if (i < n_vec) {
            const uint4 v = buf[i];
            acc += v.x + v.y + v.z + v.w;
            n += sizeof(uint4);
        }
        __syncthreads();
    }
    keep(acc, out);
    count_block(moved, n);
}

/* One plane at a time, in the order given: the expert-sorted grid.  gridDim.y
 * walks the routed list; gridDim.x covers one plane. */
__global__ void stream_planes(const uint4 *buf, const int *planes, int n_planes,
                              size_t plane_vec, unsigned long long *moved, uint32_t *out) {
    size_t n = 0;
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int p = blockIdx.y; p < n_planes; p += gridDim.y) {
        const uint4 *base = buf + (size_t)planes[p] * plane_vec;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < plane_vec; i += stride) {
            const uint4 v = base[i];
            acc += v.x + v.y + v.z + v.w;
            n += sizeof(uint4);
        }
    }
    keep(acc, out);
    count_block(moved, n);
}

/* The same planes round-robin at kChunk granularity: what a token-major
 * (unsorted) grid would issue.  Each block walks every plane, so the plane
 * index advances per chunk; the byte count is the same class as stream_planes
 * only if the per-plane coverage divides evenly, which is why the counter -- not
 * arithmetic -- is the authority. */
__global__ void stream_interleaved(const uint4 *buf, const int *planes, int n_planes,
                                   size_t plane_vec, unsigned long long *moved, uint32_t *out,
                                   size_t chunk_cap) {
    size_t n = 0;
    uint32_t acc = 0;
    /* kThreads uint4 per chunk (4 KB), so every thread in a chunk reads a
     * DISTINCT address: a narrower chunk would make the block's upper half
     * re-read the next chunk and count L2 hits as traffic. */
    const size_t per_block = kThreads;
    size_t n_chunks = (plane_vec + per_block - 1) / per_block;
    if (chunk_cap && n_chunks > chunk_cap) n_chunks = chunk_cap;
    for (size_t c = (size_t)blockIdx.x; c < n_chunks; c += gridDim.x) {
        for (int p = 0; p < n_planes; p++) {
            const uint4 *base = buf + (size_t)planes[p] * plane_vec;
            const size_t i = c * per_block + threadIdx.x;
            if (i < plane_vec) {
                const uint4 v = base[i];
                acc += v.x + v.y + v.z + v.w;
                n += sizeof(uint4);
            }
        }
    }
    keep(acc, out);
    count_block(moved, n);
}

/* A plane list for one "layer": `count` distinct experts, deterministic but
 * spread across the whole buffer (so no pass is L2-resident). */
/* Indices are spread over a span >= the largest list, so no pass is L2-resident;
 * `span` is the buffer's plane count, not the artifact's expert count. */
static void build_plane_list(int *out, int count, int layer, int span) {
    for (int i = 0; i < count; i++) out[i] = (layer * 37 + i * 13 + (i * i) % 7) % span;
}

static float time_kernel(int which, const uint4 *buf, size_t n_vec, const int *d_planes,
                         int n_planes, size_t plane_vec, int reps,
                         unsigned long long *d_moved, double *gb_out, uint32_t *d_out,
                         size_t chunk_cap) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaMemset(d_moved, 0, sizeof(*d_moved)));
    CUDA_CHECK(cudaEventRecord(a));
    for (int r = 0; r < reps; r++) {
        if (which == 0)      stream_linear<<<kBlocks, kThreads>>>(buf, n_vec, d_moved, d_out);
        else if (which == 1) stream_linear_barrier<<<kBlocks, kThreads>>>(buf, n_vec, d_moved, d_out);
        else if (which == 2) stream_planes<<<dim3(kBlocks, n_planes), kThreads>>>(
                                 buf, d_planes, n_planes, plane_vec, d_moved, d_out);
        else                 stream_interleaved<<<dim3(kBlocks, 1), kThreads>>>(
                                 buf, d_planes, n_planes, plane_vec, d_moved, d_out, chunk_cap);
    }
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    unsigned long long moved = 0;
    CUDA_CHECK(cudaMemcpy(&moved, d_moved, sizeof(moved), cudaMemcpyDeviceToHost));
    *gb_out = (double)moved / 1e9;
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms;
}

int main(int argc, char **argv) {
    int experts = argc > 1 ? atoi(argv[1]) : 256;
    int layers = argc > 2 ? atoi(argv[2]) : 8;
    int reps = argc > 3 ? atoi(argv[3]) : 2;
    if (experts < 64) experts = 64;

    /* The buffer must cover the largest plane LIST (so the linear control can
     * walk the same byte count), not just `experts` planes. */
    const size_t max_planes = (size_t)kUnique[2] * 2 * (size_t)layers;
    const size_t buf_planes = max_planes > (size_t)experts ? max_planes : (size_t)experts;
    const size_t buf_bytes = buf_planes * kPlaneBytes;
    uint4 *buf = NULL;
    CUDA_CHECK(cudaMalloc(&buf, buf_bytes));
    CUDA_CHECK(cudaMemset(buf, 0x5a, buf_bytes));
    unsigned long long *d_moved = NULL;
    CUDA_CHECK(cudaMalloc(&d_moved, sizeof(*d_moved)));
    uint32_t *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(*d_out)));
    int *d_planes = NULL;
    /* up to kUnique[2] * 2 planes per layer */
    CUDA_CHECK(cudaMalloc(&d_planes, sizeof(int) * (size_t)kUnique[2] * 2 * (size_t)layers));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("expert-stream: %s, L2 %.1f MiB; %d expert planes of %zu B (%.2f GiB); "
           "%d layers per pass, %d reps\n", prop.name, prop.l2CacheSize / 1048576.0,
           experts, kPlaneBytes, buf_bytes / 1073741824.0, layers, reps);
    printf("\n%-10s %-14s %10s %10s %10s %12s\n", "routed", "pattern", "GB moved",
           "ms/pass", "GB/s", "ms/1-row");
    printf("(interleave walks ~256 MB per pass -- enough to leave L2 -- so its GB/s is "
           "comparable; its total differs)\n");

    for (int k = 0; k < 3; k++) {
        const int uniq = kUnique[k];
        const int n_planes = uniq * 2 * layers;      /* gate+up, per layer */
        /* The linear controls walk the SAME byte count over the buffer. */
        const size_t lin_vec = (size_t)n_planes * kPlaneBytes / sizeof(uint4);
        int *h_planes = (int *)malloc(sizeof(int) * (size_t)n_planes);
        for (int l = 0; l < layers; l++) {
            int one[64];   /* kUnique[2] * 2 == 40 */
            build_plane_list(one, uniq * 2, l, (int)buf_planes);
            for (int i = 0; i < uniq * 2; i++) h_planes[l * uniq * 2 + i] = one[i];
        }
        CUDA_CHECK(cudaMemcpy(d_planes, h_planes, sizeof(int) * (size_t)n_planes,
                              cudaMemcpyHostToDevice));

        const char *names[4] = {"linear", "linear+bar", "planes", "interleave"};
        /* The interleaved order must still exceed L2 or it measures L2, so cap it
         * by BYTES (~256 MB per pass), not by chunks. */
        const size_t iv_cap = (size_t)(256u << 20) / ((size_t)n_planes * (size_t)kThreads * sizeof(uint4));
        const size_t iv_cap_use = iv_cap ? iv_cap : 1;
        const double row_ms = (double)kPlaneBytes * 6 * 2 * layers / 1e6;   /* bytes -> MB, for scale */
        for (int w = 0; w < 4; w++) {
            double gb = 0;
            const float ms = time_kernel(w, buf, lin_vec, d_planes, n_planes,
                                         kPlaneBytes / sizeof(uint4), reps, d_moved, &gb, d_out,
                                         w == 3 ? iv_cap_use : (size_t)0);
            const double per_pass_gb = gb / reps;
            const double gbs = per_pass_gb / (ms / reps / 1000.0);
            printf("%d row%-3d %-14s %10.3f %10.3f %10.1f %12.3f\n",
                   kSlotRows[k], k, names[w], per_pass_gb, ms / reps, gbs, row_ms);
            fflush(stdout);
        }
        free(h_planes);
    }
    CUDA_CHECK(cudaFree(d_planes));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_moved));
    CUDA_CHECK(cudaFree(buf));
    return 0;
}

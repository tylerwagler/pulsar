#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"   /* L158: pulsar_mx_emit_block for the producer-side E4M3 concat */


/*
 * Each block reduces the argmax over its slice of the vocabulary and writes
 * the partial winner to block_best_id/val[blockIdx.x].  The kernel does NOT
 * reduce across blocks: dspark_markov_reduce_kernel merges the grid_dim
 * partials on device.  Ties break toward the lowest id -- the strided loop
 * visits a thread's ids in ascending order with a strict-'>' update, and the
 * shared-memory tree keeps the lower-index side as the incumbent.
 */
/* W1BF16/W2BF16 select the storage of markov_w1 and markov_w2 (pulsar_w_load).
 * They are SEPARATE tensors and nothing in the format forces them to agree, so
 * they get separate flags even though every artifact so far converts them
 * together -- a single shared flag silently misreads one of them the first time
 * that stops being true.
 *
 * This kernel streams the WHOLE of markov_w2 -- every vocab row, every step --
 * at one multiply-add per element, so it is pure bandwidth: halving the element
 * width halves its runtime. That, not the 132 MB, is why the storage matters. */
template <bool W1BF16, bool W2BF16>
__global__ static void dspark_markov_step_kernel(
        float *refined_logits,
        int32_t *block_best_id,
        float *block_best_val,
        const float *base_logits,
        const void *markov_w1,
        const void *markov_w2,
        const int32_t *prev_token_ptr,  /* L108 P1: device-fed so the walk
                                         * chains without a host round-trip;
                                         * broadcast read, cached */
        uint32_t vocab_size,
        uint32_t embed_dim) {
    int32_t prev_token = *prev_token_ptr;
    if (prev_token < 0 || (uint32_t)prev_token >= vocab_size) prev_token = 0;
    const uint64_t embed_base = (uint64_t)prev_token * embed_dim;
    float best_val = -INFINITY;
    int32_t best_id = 0;

    for (uint32_t v = threadIdx.x + blockIdx.x * blockDim.x; v < vocab_size;
         v += blockDim.x * gridDim.x) {
        float dot = 0.0f;
        const uint64_t w2_base = (uint64_t)v * embed_dim;
        for (uint32_t i = 0; i < embed_dim; i++)
            dot += pulsar_w_load_f32_or_bf16<W2BF16>(markov_w2, w2_base + i) *
                   pulsar_w_load_f32_or_bf16<W1BF16>(markov_w1, embed_base + i);
        float val = base_logits[v] + dot;
        refined_logits[v] = val;
        if (val > best_val) { best_val = val; best_id = (int32_t)v; }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            /* strict '>': the lower-index side stays the incumbent on a tie */
            const float nb = best_vals[tid + stride];
            if (nb > best_vals[tid]) {
                best_vals[tid] = nb; best_ids[tid] = best_ids[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_id[blockIdx.x] = best_ids[0];
        block_best_val[blockIdx.x] = best_vals[0];
    }
}

/*
 * Merge the per-block argmax partials to the global argmax ON DEVICE.  The
 * single-step entry reads back only the winning id (4 bytes, one copy); the
 * chained walk writes it straight into the device id feed.
 *
 * Tie-break: blocks map to ascending contiguous vocab ranges and the
 * per-block reduction breaks ties toward the lowest id.  Each thread here
 * scans its strided subset of blocks in ascending order with the same
 * strict-'>' merge, and the shared-memory tree keeps the lower-index side as
 * the incumbent, so the lowest-id global argmax is preserved exactly --
 * matching a sequential argmax over the vocab.
 */
__global__ static void dspark_markov_reduce_kernel(
        int32_t *dst,               /* winner id (L108 P1: points into the
                                     * chain's device id array, so the next
                                     * step kernel reads it directly) */
        const int32_t *ids,
        const float *vals,
        uint32_t n_blocks) {
    float best_val = -INFINITY;
    int32_t best_id = 0;
    for (uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const float nb = vals[b];
        if (nb > best_val) { best_val = nb; best_id = ids[b]; }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float nb = best_vals[tid + stride];
            if (nb > best_vals[tid]) {
                best_vals[tid] = nb; best_ids[tid] = best_ids[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *dst = best_ids[0];
}

static int dspark_markov_reduce_blocks(const pulsar_gpu_tensor *id_dev,
                                        const pulsar_gpu_tensor *val_dev,
                                        pulsar_gpu_tensor *out_dev, /* 1 x int32 */
                                        uint32_t grid_dim,
                                        int32_t *refined_id_dst) {
    dspark_markov_reduce_kernel<<<1, 256>>>(
        (int32_t *)out_dev->ptr,
        (const int32_t *)id_dev->ptr,
        (const float *)val_dev->ptr,
        grid_dim);
    if (cudaGetLastError() != cudaSuccess) return 0;
    int32_t out;
    if (!pulsar_gpu_tensor_read(out_dev, 0, &out, sizeof(out))) return 0;
    *refined_id_dst = out;
    return 1;
}


int pulsar_gpu_dspark_markov_step_model(
        pulsar_gpu_tensor *refined_logits,
        int32_t *refined_id_dst,
        const pulsar_gpu_tensor *base_logits,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t markov_w2_offset,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim,
        int w1_bf16,
        int w2_bf16) {
    if (!refined_logits || !refined_id_dst || !base_logits || !dspark_model_map)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (base_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if ((uint64_t)prev_token >= vocab_size) return 0;

    /* L108 P1: the kernel now takes the previous id by DEVICE pointer (so the
     * chained greedy walk needs no host round-trip). This single-step entry
     * (the sampled path, which routes the chain through a host rng draw)
     * stages its host id into a persistent 1-int slot -- one 4-byte H2D per
     * position, noise next to that path's per-position 517 KB qrow read. */
    static thread_local pulsar_gpu_tensor *staged_id = NULL;
    if (!staged_id) staged_id = pulsar_gpu_tensor_alloc(sizeof(int32_t));
    if (!staged_id) return 0;
    if (!pulsar_gpu_tensor_write(staged_id, 0, &prev_token, sizeof(int32_t)))
        return 0;

    /* Sized by the STORAGE width, not sizeof(float): a bf16 markov head is half
     * these bytes and checking it against the f32 size would reject a valid
     * tensor near the end of the file. */
    const uint64_t w1_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w1_bf16);
    const uint64_t w2_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w2_bf16);
    if (markov_w1_offset > dspark_model_size ||
        w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (markov_w2_offset > dspark_model_size ||
        w2_bytes > dspark_model_size - markov_w2_offset) return 0;

    const void *w1 = cuda_model_range_ptr(
        dspark_model_map, markov_w1_offset, w1_bytes, "dspark_markov_w1");
    const void *w2 = cuda_model_range_ptr(
        dspark_model_map, markov_w2_offset, w2_bytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) return 0;

    /* Persistent reduce buffers, grown on demand: grid_dim is fixed for a given
     * vocab, and this runs once per draft position per spec step -- per-call
     * cudaMalloc/cudaFree pairs were 2 device-serializing allocs each. Grouped
     * into one struct (Pulsar C++ port, plan-70 TU dspark) -- same alloc sizes,
     * free order, and grow-on-demand logic as the prior loose statics, so the
     * launches are byte-identical.
     *
     * ⚠ THREAD_LOCAL, NOT STATIC, AND THAT IS THE WHOLE POINT.  As a process
     * global this carried three separate hazards the moment a second thread
     * submitted:
     *   - the grow path is a read-modify-write on shared state, so two threads
     *     both seeing grid_dim > rb.cap both free and both allocate: one set of
     *     pointers is double-freed and the other leaks;
     *   - a thread growing the buffers frees storage another thread's launched
     *     kernel may still be reading;
     *   - and worst because it does not crash: with NO growth at all, two
     *     threads share the same reduction scratch and silently overwrite each
     *     other's intermediates, i.e. wrong drafter output, not a fault.
     * The old comment held this together with the words "Single submission
     * thread" -- true today (cli_main.cpp creates exactly one worker), but an
     * assumption, not a guarantee, and invisible from any call site.
     *
     * A mutex would be the wrong fix: guarding only the grow leaves the third
     * hazard, and guarding the whole reduction serialises the drafter across
     * sessions. Removing the sharing costs ~8 KB per submitting thread
     * (grid_dim ~505 at vocab 129280) and makes the property structural.
     * Same reason and same shape as the thread_local activation cache in
     * pulsar_cuda_matmul.cu. */
    struct DsparkReduceBufs {
        pulsar_gpu_tensor *id, *val, *out;
        uint32_t cap;
    };
    static thread_local DsparkReduceBufs rb = {};
    if (grid_dim > rb.cap) {
        pulsar_gpu_tensor_free(rb.id);
        pulsar_gpu_tensor_free(rb.val);
        pulsar_gpu_tensor_free(rb.out);
        rb.id  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        rb.val = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        rb.out = pulsar_gpu_tensor_alloc(sizeof(int32_t));
        rb.cap = (rb.id && rb.val && rb.out) ? grid_dim : 0;
    }
    if (!rb.id || !rb.val || !rb.out) return 0;

#define PULSAR_MARKOV_LAUNCH(A, B)                          \
    dspark_markov_step_kernel<A, B><<<grid_dim, block_dim>>>( \
        (float *)refined_logits->ptr,                         \
        (int32_t *)rb.id->ptr,                                \
        (float *)rb.val->ptr,                                 \
        (const float *)base_logits->ptr,                      \
        w1, w2, (const int32_t *)staged_id->ptr, vocab_size, embed_dim)
    if (w1_bf16 && w2_bf16)   PULSAR_MARKOV_LAUNCH(true, true);
    else if (w1_bf16)         PULSAR_MARKOV_LAUNCH(true, false);
    else if (w2_bf16)         PULSAR_MARKOV_LAUNCH(false, true);
    else                      PULSAR_MARKOV_LAUNCH(false, false);
#undef PULSAR_MARKOV_LAUNCH

    int rc = 0;
    if (cudaGetLastError() == cudaSuccess)
        rc = dspark_markov_reduce_blocks(rb.id, rb.val, rb.out, grid_dim,
                                         refined_id_dst);
    return rc;
}


/** Per-block argmax partials the chained markov walk reduces from.  Kept
 * device-side for the whole chain so the walk needs no host round trip. */
struct DsparkReduceBufsChain {
    pulsar_gpu_tensor *id,    ///< per-block winner token id
                      *val;   ///< per-block winner score
    uint32_t cap;             ///< blocks the buffers can hold
};

/* L108 P1: device-chained greedy markov walk.  Launches the whole n_draft
 * refine sequence with the token feed in device memory: step kernel pos p
 * reads ids[p], its reduce writes the winner to ids[p+1], which the next step
 * kernel reads directly.  ids[0] must be pre-seeded by the caller.  NO
 * device-to-host read happens here -- the caller reads ids once after the
 * chain (that read is the round's one drafter sync).  refined_logits is a single [vocab] f32 scratch row reused
 * by every position: nothing in the greedy chain reads it back, and reuse is
 * stream-ordered.  Same kernels, same launch order, same arithmetic as the
 * single-step path -- byte-exact by construction (any gate red is a bug).
 * The SAMPLED path cannot use this: its chain routes through a host rng draw
 * per position (pulsar_sample_dist_draw), so it keeps the single-step entry
 * above (P1b would need device-side dist build + draw with rng parity). */
int pulsar_gpu_dspark_markov_chain_model(
        pulsar_gpu_tensor *refined_logits,      /* [vocab] f32 scratch */
        pulsar_gpu_tensor *ids_dev,             /* [n_draft+1] i32, [0] seeded */
        const pulsar_gpu_tensor *base_logits,   /* [n_draft, row_stride] f32 */
        uint64_t base_row_stride_bytes,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t markov_w2_offset,
        uint32_t n_draft,
        uint32_t vocab_size,
        uint32_t embed_dim,
        int w1_bf16,
        int w2_bf16) {
    if (!refined_logits || !ids_dev || !base_logits ||
        !dspark_model_map || n_draft == 0 || n_draft > 16)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (ids_dev->bytes < ((uint64_t)n_draft + 1) * sizeof(int32_t)) return 0;
    if (base_row_stride_bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (base_logits->bytes < (uint64_t)(n_draft - 1) * base_row_stride_bytes +
                             (uint64_t)vocab_size * sizeof(float)) return 0;

    const uint64_t w1_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w1_bf16);
    const uint64_t w2_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w2_bf16);
    if (markov_w1_offset > dspark_model_size ||
        w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (markov_w2_offset > dspark_model_size ||
        w2_bytes > dspark_model_size - markov_w2_offset) return 0;
    const void *w1 = cuda_model_range_ptr(
        dspark_model_map, markov_w1_offset, w1_bytes, "dspark_markov_w1");
    const void *w2 = cuda_model_range_ptr(
        dspark_model_map, markov_w2_offset, w2_bytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) return 0;
    static thread_local DsparkReduceBufsChain crb = {};
    if (grid_dim > crb.cap) {
        pulsar_gpu_tensor_free(crb.id);
        pulsar_gpu_tensor_free(crb.val);
        crb.id  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        crb.val = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        crb.cap = (crb.id && crb.val) ? grid_dim : 0;
    }
    if (!crb.id || !crb.val) return 0;

    int32_t *ids = (int32_t *)ids_dev->ptr;
    for (uint32_t pos = 0; pos < n_draft; pos++) {
        const float *base_row =
            (const float *)((const char *)base_logits->ptr +
                            (uint64_t)pos * base_row_stride_bytes);
#define PULSAR_MARKOV_CHAIN_LAUNCH(A, B)                        \
        dspark_markov_step_kernel<A, B><<<grid_dim, block_dim>>>( \
            (float *)refined_logits->ptr,                         \
            (int32_t *)crb.id->ptr,                               \
            (float *)crb.val->ptr,                                \
            base_row, w1, w2, ids + pos, vocab_size, embed_dim)
        if (w1_bf16 && w2_bf16)   PULSAR_MARKOV_CHAIN_LAUNCH(true, true);
        else if (w1_bf16)         PULSAR_MARKOV_CHAIN_LAUNCH(true, false);
        else if (w2_bf16)         PULSAR_MARKOV_CHAIN_LAUNCH(false, true);
        else                      PULSAR_MARKOV_CHAIN_LAUNCH(false, false);
#undef PULSAR_MARKOV_CHAIN_LAUNCH
        dspark_markov_reduce_kernel<<<1, 256>>>(
            ids + pos + 1, (const int32_t *)crb.id->ptr, (const float *)crb.val->ptr,
            grid_dim);
    }
    return cudaGetLastError() == cudaSuccess;
}


/* DSpark confidence head (model.py DSparkConfidenceHead): per block position,
 * scores[p] = sigmoid( proj . cat(hidden[p], markov_embed[token_ids[p]]) ), where
 * hidden is the post-hc_head drafter hidden and markov_embed is a row of markov_w1
 * gathered by token id. Used to size the verify budget (confidence-scheduled
 * verification) -- higher confidence => the draft is more likely accepted. */
template <bool W1BF16, bool PROJBF16>
__global__ static void dspark_confidence_score_kernel(
        float *scores,
        const float *hidden,        /* [n_positions, hidden_dim] */
        const int32_t *token_ids,   /* [n_positions] */
        const void *markov_w1,      /* [vocab, embed_dim], f32 or bf16 */
        const void *proj,           /* [hidden_dim + embed_dim], f32 or bf16 */
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size) {
    uint32_t p = blockIdx.x;
    if (p >= n_positions) return;
    int32_t t = token_ids[p];
    if (t < 0 || (uint32_t)t >= vocab_size) t = 0;
    const float *hp = hidden + (uint64_t)p * hidden_dim;
    const uint64_t emb_base = (uint64_t)t * embed_dim;
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        dot += hp[i] * pulsar_w_load_f32_or_bf16<PROJBF16>(proj, i);
    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x)
        dot += pulsar_w_load_f32_or_bf16<W1BF16>(markov_w1, emb_base + i) *
               pulsar_w_load_f32_or_bf16<PROJBF16>(proj, hidden_dim + i);
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[p] = 1.0f / (1.0f + expf(-partial[0]));
}


int pulsar_gpu_dspark_confidence_score_model(
        pulsar_gpu_tensor *scores,
        const pulsar_gpu_tensor *hidden,
        const pulsar_gpu_tensor *token_ids,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t proj_offset,
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size,
        int w1_bf16, int proj_bf16) {
    if (!scores || !hidden || !token_ids || !dspark_model_map) return 0;
    if (n_positions == 0 || hidden_dim == 0 || embed_dim == 0 || vocab_size == 0) return 0;
    if (scores->bytes < (uint64_t)n_positions * sizeof(float)) return 0;
    if (hidden->bytes < (uint64_t)n_positions * hidden_dim * sizeof(float)) return 0;
    if (token_ids->bytes < (uint64_t)n_positions * sizeof(int32_t)) return 0;
    /* Each sized by its OWN storage width -- w1 and proj are separate tensors. */
    const uint64_t w1_bytes = (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w1_bf16);
    const uint64_t proj_bytes = (uint64_t)(hidden_dim + embed_dim) * pulsar_w_elt_bytes(proj_bf16);
    if (markov_w1_offset > dspark_model_size || w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (proj_offset > dspark_model_size || proj_bytes > dspark_model_size - proj_offset) return 0;
    const void *w1 = cuda_model_range_ptr(dspark_model_map, markov_w1_offset, w1_bytes, "dspark_conf_w1");
    const void *proj = cuda_model_range_ptr(dspark_model_map, proj_offset, proj_bytes, "dspark_conf_proj");
    if (!w1 || !proj) return 0;
#define PULSAR_CONF_LAUNCH(A, B)                                                    \
    dspark_confidence_score_kernel<A, B><<<n_positions, 256>>>(                     \
        (float *)scores->ptr, (const float *)hidden->ptr, (const int32_t *)token_ids->ptr, \
        w1, proj, n_positions, hidden_dim, embed_dim, vocab_size)
    if (w1_bf16 && proj_bf16)        PULSAR_CONF_LAUNCH(true, true);
    else if (w1_bf16)                PULSAR_CONF_LAUNCH(true, false);
    else if (proj_bf16)              PULSAR_CONF_LAUNCH(false, true);
    else                             PULSAR_CONF_LAUNCH(false, false);
#undef PULSAR_CONF_LAUNCH
    return cuda_ok(cudaGetLastError(), "dspark confidence score");
}



/* Batched variant over a [n_tokens, n_hc, n_embd] HC tensor: out[p] = mean over
 * hc of in[p]. Captures the drafter's anchor-layer hidden for EVERY position of
 * a spec verify batch, so the fused loop can pick the last-accepted position's
 * hidden without replaying. */
__global__ static void dspark_hc_mean_reduce_batch_kernel(
        float *out,
        const pulsar_hc_t *hc_batch,   /* HC residual carrier (BF16); task #62 */
        uint32_t n_embd,
        uint32_t n_hc) {
    const uint32_t p = blockIdx.y;
    const pulsar_hc_t *in = hc_batch + (uint64_t)p * n_hc * n_embd;
    float *op = out + (uint64_t)p * n_embd;
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += pulsar_hc_load(in, (uint64_t)hc * n_embd + d);
        op[d] = sum / (float)n_hc;
    }
}


int pulsar_gpu_dspark_hc_mean_reduce_batch(
        pulsar_gpu_tensor *out,
        const pulsar_gpu_tensor *hc_batch,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens) {
    if (!out || !hc_batch || n_embd == 0 || n_hc == 0 || n_tokens == 0) return 0;
    if (out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (hc_batch->bytes < (uint64_t)n_tokens * n_hc * n_embd * PULSAR_HC_ELT_SIZE) return 0;   /* carrier */

    const uint32_t block_dim = 256;
    dim3 grid((n_embd + block_dim - 1) / block_dim, n_tokens, 1);
    dspark_hc_mean_reduce_batch_kernel<<<grid, block_dim>>>(
        (float *)out->ptr,
        (const pulsar_hc_t *)hc_batch->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce batch");
}


/* =====================================================================
 * L149: min-p candidate prefilter (contract in pulsar_gpu.h).
 *
 * The sampled speculative path used to read the drafter's whole 517 KB
 * logits row back per draft position and run the host sampler over it; the
 * 129k host expf calls behind that read idled the GPU ~630 us per position
 * (production profile, rows/L149.md). The host min-p cutoff is now
 * division-free (tokenizer.cpp), so the sampler needs only the candidates
 * at or above the floor: this kernel finds the row max exactly as the host
 * scan does (first finite maximum, lowest id on ties) and emits every finite
 * logit >= max + delta in ascending id order -- a superset of the survivors
 * that pulsar_sample_dist_build_prefiltered trims with the host's own
 * comparison. No expf on the device: membership is one float add and one
 * compare, reproducible bit-for-bit on the host (tests/minp_prefilter_gate).
 *
 * One block per row, 1024 threads: a strided max reduce, then a contiguous
 * chunk per thread so a block-wide scan places each thread's candidates at
 * their index-ordered offsets. */
__global__ static void minp_prefilter_rows_kernel(const float *__restrict__ logits,
                                                  uint32_t row_stride, uint32_t n_vocab,
                                                  float delta, uint32_t cap,
                                                  int32_t *__restrict__ out, uint32_t out_stride) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];
    __shared__ uint32_t sm_cnt[THREADS];
    const float *x = logits + (size_t)blockIdx.x * row_stride;
    int32_t *o = out + (size_t)blockIdx.x * out_stride;
    const uint32_t tid = threadIdx.x;

    /* 1. Row max: first finite maximum, lowest id on ties (the host's pass 1). */
    float lv = 0.0f;
    int32_t li = -1;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = x[i];
        if (!isfinite(v)) continue;
        if (li < 0 || v > lv) { lv = v; li = (int32_t)i; }
    }
    sm_val[tid] = lv;
    sm_idx[tid] = li;
    __syncthreads();
    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            const bool take_right = ir >= 0 && (il < 0 || vr > vl || (vr == vl && ir < il));
            if (take_right) { sm_val[tid] = vr; sm_idx[tid] = ir; }
        }
        __syncthreads();
    }
    const float mx = sm_val[0];
    const int32_t mi = sm_idx[0];
    if (mi < 0) {
        if (tid == 0) { o[0] = 0; o[1] = -1; o[2] = 0; }
        return;
    }
    const float thr = mx + delta;

    /* 2. Count per contiguous chunk, scan, then place -- ascending id overall. */
    const uint32_t chunk = (n_vocab + THREADS - 1u) / THREADS;
    const uint32_t lo = tid * chunk;
    const uint32_t hi = min(lo + chunk, n_vocab);
    uint32_t cnt = 0;
    for (uint32_t i = lo; i < hi; i++) {
        const float v = x[i];
        if (isfinite(v) && v >= thr) cnt++;
    }
    sm_cnt[tid] = cnt;
    __syncthreads();
    for (uint32_t off = 1; off < THREADS; off <<= 1) {
        const uint32_t t = tid >= off ? sm_cnt[tid - off] : 0u;
        __syncthreads();
        sm_cnt[tid] += t;
        __syncthreads();
    }
    const uint32_t total = sm_cnt[THREADS - 1];
    if (tid == 0) { o[0] = (int32_t)total; o[1] = mi; o[2] = __float_as_int(mx); }
    if (total > cap) return;
    uint32_t pos = sm_cnt[tid] - cnt;
    int32_t *ids = o + 3;
    float *vals = reinterpret_cast<float *>(o + 3 + cap);
    for (uint32_t i = lo; i < hi; i++) {
        const float v = x[i];
        if (isfinite(v) && v >= thr) {
            ids[pos] = (int32_t)i;
            vals[pos] = v;
            pos++;
        }
    }
}

int pulsar_gpu_minp_prefilter_rows(
        pulsar_gpu_tensor       *out,
        const pulsar_gpu_tensor *logits,
        uint64_t                logits_offset_bytes,
        uint32_t                n_rows,
        uint32_t                row_stride_elems,
        uint32_t                n_vocab,
        float                   delta,
        uint32_t                cap) {
    const uint64_t out_stride = 3ull + 2ull * cap;
    if (!out || !logits || n_rows == 0 || n_rows > 1024u || n_vocab == 0 || cap == 0 ||
        row_stride_elems < n_vocab || !(delta <= 0.0f) ||
        logits_offset_bytes > logits->bytes ||
        (uint64_t)(n_rows - 1) * row_stride_elems * sizeof(float) + (uint64_t)n_vocab * sizeof(float) >
            logits->bytes - logits_offset_bytes ||
        out->bytes < (uint64_t)n_rows * out_stride * sizeof(int32_t)) {
        return 0;
    }
    minp_prefilter_rows_kernel<<<n_rows, 1024>>>(
        (const float *)((const char *)logits->ptr + logits_offset_bytes),
        row_stride_elems, n_vocab, delta, cap,
        (int32_t *)out->ptr, (uint32_t)out_stride);
    return cuda_ok(cudaGetLastError(), "minp prefilter launch");
}

/* --- plan-92 P0: per-row teacher top-64 + tail logsumexp ---------------------
 * One block per logits row. Selection: each thread keeps a local top-8 over
 * its strided slice (~505 elems at N_VOCAB/256), the 2048 candidates go to
 * smem, and 64 rounds of block argmax pick the winners. Exactness is VERIFIED
 * (a slice holding >8 of the global top-64 would drop one): a final count of
 * elements above the selected floor bumps `inexact` if it exceeds 64 -- with
 * real logits this never fires, and the counter makes "never" checkable.
 * tail_lse = log(sumexp(all) - sumexp(top64)) + max, so the training loss can
 * renormalize the non-top mass exactly. Collection-rate code: clarity over
 * throughput. */
__global__ static void distill_top64_lse_kernel(
        const float *logits,      /* [n_rows, vocab] */
        uint32_t vocab,
        int32_t *top_ids,         /* [cap, 64], written at row0+row */
        uint16_t *top_vals,       /* f16 bits */
        uint16_t *tail_lse,       /* f16 bits */
        int32_t *inexact,
        uint32_t row0) {
    constexpr int kThreads = 256, kLocal = 8, kTop = 64;
    const uint32_t row = blockIdx.x;
    const float *x = logits + (uint64_t)row * vocab;
    const int tid = threadIdx.x;

    __shared__ float s_red[kThreads];
    /* pass 1: row max */
    float mx = -INFINITY;
    for (uint32_t i = tid; i < vocab; i += kThreads) mx = fmaxf(mx, x[i]);
    s_red[tid] = mx; __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] = fmaxf(s_red[tid], s_red[tid + s]);
        __syncthreads();
    }
    const float rmax = s_red[0]; __syncthreads();

    /* pass 2: local top-8 + sumexp */
    float lv[kLocal]; int li[kLocal];
    for (int j = 0; j < kLocal; j++) { lv[j] = -INFINITY; li[j] = -1; }
    float sumexp = 0.0f;
    for (uint32_t i = tid; i < vocab; i += kThreads) {
        const float v = x[i];
        sumexp += __expf(v - rmax);
        if (v > lv[kLocal - 1]) {
            int j = kLocal - 1;
            while (j > 0 && v > lv[j - 1]) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--; }
            lv[j] = v; li[j] = (int)i;
        }
    }
    s_red[tid] = sumexp; __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    const float total_sumexp = s_red[0]; __syncthreads();

    __shared__ float c_val[kThreads * kLocal];
    __shared__ int   c_id[kThreads * kLocal];
    for (int j = 0; j < kLocal; j++) {
        c_val[tid * kLocal + j] = lv[j];
        c_id[tid * kLocal + j]  = li[j];
    }
    __syncthreads();

    /* 64 rounds of block argmax over the 2048 candidates */
    __shared__ int s_arg[kThreads];
    __shared__ float s_sel_floor;
    __shared__ float s_top_sumexp;
    if (tid == 0) s_top_sumexp = 0.0f;
    for (int t = 0; t < kTop; t++) {
        float bv = -INFINITY; int bi = -1;
        for (int j = tid; j < kThreads * kLocal; j += kThreads) {
            if (c_val[j] > bv) { bv = c_val[j]; bi = j; }
        }
        s_red[tid] = bv; s_arg[tid] = bi; __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s && s_red[tid + s] > s_red[tid]) {
                s_red[tid] = s_red[tid + s]; s_arg[tid] = s_arg[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int w = s_arg[0];
            const float wv = c_val[w];
            top_ids[(uint64_t)(row0 + row) * kTop + t] = c_id[w];
            top_vals[(uint64_t)(row0 + row) * kTop + t] = __half_as_ushort(__float2half(wv));
            s_top_sumexp += __expf(wv - rmax);
            c_val[w] = -INFINITY;
            if (t == kTop - 1) s_sel_floor = wv;
        }
        __syncthreads();
    }

    /* verify: exact top-64 requires <= 64 elements above the floor */
    uint32_t above = 0;
    for (uint32_t i = tid; i < vocab; i += kThreads) above += x[i] > s_sel_floor ? 1u : 0u;
    s_red[tid] = (float)above; __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        if ((int)s_red[0] > kTop) atomicAdd(inexact, 1);
        const float tail = total_sumexp - s_top_sumexp;
        const float lse = tail > 0.0f ? logf(tail) + rmax : -INFINITY;
        tail_lse[row0 + row] = __half_as_ushort(__float2half(lse));
    }
}

int pulsar_gpu_distill_top64_tensor(
        const pulsar_gpu_tensor *logits_rows,
        uint32_t n_rows,
        uint32_t vocab,
        pulsar_gpu_tensor *top_ids,
        pulsar_gpu_tensor *top_vals,
        pulsar_gpu_tensor *tail_lse,
        pulsar_gpu_tensor *inexact,
        uint32_t row0) {
    if (!logits_rows || !top_ids || !top_vals || !tail_lse || !inexact ||
        n_rows == 0 || vocab == 0)
        return 0;
    if (logits_rows->bytes < (uint64_t)n_rows * vocab * sizeof(float)) return 0;
    if (top_ids->bytes < (uint64_t)(row0 + n_rows) * 64 * sizeof(int32_t)) return 0;
    if (top_vals->bytes < (uint64_t)(row0 + n_rows) * 64 * sizeof(uint16_t)) return 0;
    if (tail_lse->bytes < (uint64_t)(row0 + n_rows) * sizeof(uint16_t)) return 0;
    distill_top64_lse_kernel<<<n_rows, 256>>>(
        (const float *)logits_rows->ptr, vocab,
        (int32_t *)top_ids->ptr, (uint16_t *)top_vals->ptr,
        (uint16_t *)tail_lse->ptr, (int32_t *)inexact->ptr, row0);
    return cuda_ok(cudaGetLastError(), "distill top64 lse");
}


/* =====================================================================
 * L150: bank-batched markov refine.
 *
 * The single-bank kernels above stream all of markov_w2 (129,280 x 256 bf16,
 * 66 MB) once per draft position PER BANK; the served lane ran three banks'
 * drafter passes back to back, so a tick paid 3 x n_draft streams. These
 * kernels take N banks per launch: each thread loads a w2 row element once
 * and dots it against N w1 rows (one per bank's previous token), so one
 * stream serves every bank.
 *
 * BYTE-EXACT with the single-bank kernels by construction: per bank the dot
 * accumulates the same products in the same i order into its own
 * accumulator, the value is base + dot exactly as before, the per-block argmax
 * uses the same strict-'>' update over the same block -> vocab mapping, and
 * the reduce is the same merge. The greedy identity gate (rows/L150.md) pins
 * it; any red there is a bug, not tolerance.
 *
 * Layouts: refined_logits [n_banks][vocab]; per-block partials
 * [n_banks][gridDim.x]; the bank's base row is base_logits[(base_row[b] +
 * base_row_add) * base_row_stride]; the previous token for bank b is
 * prev[b * prev_stride]. n_banks <= PULSAR_DSPARK_BANKS_MAX. */
template <bool W1BF16, bool W2BF16>
__global__ static void dspark_markov_step_banks_kernel(
        float *__restrict__ refined_logits,
        int32_t *__restrict__ block_best_id,
        float *__restrict__ block_best_val,
        const float *__restrict__ base_logits,
        const int32_t *__restrict__ base_row,
        uint32_t base_row_add,
        uint64_t base_row_stride,
        const void *markov_w1,
        const void *markov_w2,
        const int32_t *__restrict__ prev,
        uint32_t prev_stride,
        uint32_t n_banks,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    enum { MAXB = PULSAR_DSPARK_BANKS_MAX };
    uint64_t embed_base[MAXB];
    const float *base[MAXB];
    float best_val[MAXB];
    int32_t best_id[MAXB];
    for (uint32_t b = 0; b < MAXB; b++) {
        embed_base[b] = 0; base[b] = base_logits;
        best_val[b] = -INFINITY; best_id[b] = 0;
        if (b < n_banks) {
            int32_t pt = prev[(uint64_t)b * prev_stride];
            if (pt < 0 || (uint32_t)pt >= vocab_size) pt = 0;
            embed_base[b] = (uint64_t)pt * embed_dim;
            base[b] = base_logits + ((uint64_t)base_row[b] + base_row_add) * base_row_stride;
        }
    }
    for (uint32_t v = threadIdx.x + blockIdx.x * blockDim.x; v < vocab_size;
         v += blockDim.x * gridDim.x) {
        float dot[MAXB];
        for (uint32_t b = 0; b < MAXB; b++) dot[b] = 0.0f;
        const uint64_t w2_base = (uint64_t)v * embed_dim;
        for (uint32_t i = 0; i < embed_dim; i++) {
            const float w2v = pulsar_w_load_f32_or_bf16<W2BF16>(markov_w2, w2_base + i);
            for (uint32_t b = 0; b < n_banks; b++)
                dot[b] += w2v * pulsar_w_load_f32_or_bf16<W1BF16>(markov_w1, embed_base[b] + i);
        }
        for (uint32_t b = 0; b < n_banks; b++) {
            const float val = base[b][v] + dot[b];
            refined_logits[(uint64_t)b * vocab_size + v] = val;
            if (val > best_val[b]) { best_val[b] = val; best_id[b] = (int32_t)v; }
        }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    const uint32_t tid = threadIdx.x;
    for (uint32_t b = 0; b < n_banks; b++) {
        best_vals[tid] = best_val[b];
        best_ids[tid] = best_id[b];
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float nb = best_vals[tid + stride];
                if (nb > best_vals[tid]) {
                    best_vals[tid] = nb; best_ids[tid] = best_ids[tid + stride];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const uint64_t o = (uint64_t)b * gridDim.x + blockIdx.x;
            block_best_id[o] = best_ids[0];
            block_best_val[o] = best_vals[0];
        }
        __syncthreads();
    }
}

/* One block per bank: the same merge as dspark_markov_reduce_kernel over that
 * bank's partials, the winner written to dst[b * dst_stride]. */
__global__ static void dspark_markov_reduce_banks_kernel(
        int32_t *__restrict__ dst,
        uint32_t dst_stride,
        const int32_t *__restrict__ ids,
        const float *__restrict__ vals,
        uint32_t n_blocks) {
    const uint32_t b = blockIdx.x;
    ids += (uint64_t)b * n_blocks; vals += (uint64_t)b * n_blocks;
    float best_val = -INFINITY;
    int32_t best_id = 0;
    for (uint32_t k = threadIdx.x; k < n_blocks; k += blockDim.x) {
        const float nb = vals[k];
        if (nb > best_val) { best_val = nb; best_id = ids[k]; }
    }
    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float nb = best_vals[tid + stride];
            if (nb > best_vals[tid]) {
                best_vals[tid] = nb; best_ids[tid] = best_ids[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) dst[(uint64_t)b * dst_stride] = best_ids[0];
}

/* Per-bank partials for the banked kernels; grown on demand, thread_local for
 * the same reasons as DsparkReduceBufs above. */
struct DsparkReduceBufsBanks {
    pulsar_gpu_tensor *id, *val;
    uint32_t cap;   /* n_banks * grid_dim slots */
};

static bool dspark_markov_banks_args_ok(
        const pulsar_gpu_tensor *refined_logits, const pulsar_gpu_tensor *ids_dev,
        uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev, uint32_t n_banks, uint32_t n_draft,
        uint32_t vocab_size, uint32_t embed_dim) {
    if (!refined_logits || !ids_dev || !base_logits || !base_row_dev) return false;
    if (n_banks == 0 || n_banks > PULSAR_DSPARK_BANKS_MAX || n_draft == 0 || n_draft > 16) return false;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return false;
    if (ids_stride < n_draft + 1) return false;
    if (refined_logits->bytes < (uint64_t)n_banks * vocab_size * sizeof(float)) return false;
    if (ids_dev->bytes < (uint64_t)n_banks * ids_stride * sizeof(int32_t)) return false;
    if (base_row_dev->bytes < (uint64_t)n_banks * sizeof(int32_t)) return false;
    if (base_row_stride_bytes < (uint64_t)vocab_size * sizeof(float) ||
        base_row_stride_bytes % sizeof(float) != 0) return false;
    /* the caller's base rows are bounded by its own row budget; the kernel
     * indexes base_logits by (base_row[b] + pos) * stride and trusts it */
    return true;
}

static int dspark_markov_banks_launch(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev,
        uint32_t ids_stride, const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev, const void *dspark_model_map,
        uint64_t dspark_model_size, uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t pos_first, uint32_t pos_count,
        const pulsar_gpu_tensor *prev_override, /* NULL: chain feed ids[b][pos]; else [n_banks] i32 */
        uint32_t vocab_size, uint32_t embed_dim, int w1_bf16, int w2_bf16) {
    const uint64_t w1_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w1_bf16);
    const uint64_t w2_bytes =
        (uint64_t)vocab_size * embed_dim * pulsar_w_elt_bytes(w2_bf16);
    if (markov_w1_offset > dspark_model_size ||
        w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (markov_w2_offset > dspark_model_size ||
        w2_bytes > dspark_model_size - markov_w2_offset) return 0;
    const void *w1 = cuda_model_range_ptr(
        dspark_model_map, markov_w1_offset, w1_bytes, "dspark_markov_w1");
    const void *w2 = cuda_model_range_ptr(
        dspark_model_map, markov_w2_offset, w2_bytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) return 0;
    static thread_local DsparkReduceBufsBanks bb = {};
    const uint32_t need = n_banks * grid_dim;
    if (need > bb.cap) {
        pulsar_gpu_tensor_free(bb.id);
        pulsar_gpu_tensor_free(bb.val);
        const uint32_t cap = PULSAR_DSPARK_BANKS_MAX * grid_dim;
        bb.id  = pulsar_gpu_tensor_alloc((uint64_t)cap * sizeof(int32_t));
        bb.val = pulsar_gpu_tensor_alloc((uint64_t)cap * sizeof(float));
        bb.cap = (bb.id && bb.val) ? cap : 0;
    }
    if (!bb.id || !bb.val) return 0;

    int32_t *ids = (int32_t *)ids_dev->ptr;
    for (uint32_t pos = pos_first; pos < pos_first + pos_count; pos++) {
        const int32_t *prev = prev_override ? (const int32_t *)prev_override->ptr : ids + pos;
        const uint32_t prev_stride = prev_override ? 1u : ids_stride;
#define PULSAR_MARKOV_BANKS_LAUNCH(A, B)                                    \
        dspark_markov_step_banks_kernel<A, B><<<grid_dim, block_dim>>>(       \
            (float *)refined_logits->ptr,                                     \
            (int32_t *)bb.id->ptr, (float *)bb.val->ptr,                      \
            (const float *)base_logits->ptr, (const int32_t *)base_row_dev->ptr, \
            pos, base_row_stride_bytes / sizeof(float),                       \
            w1, w2, prev, prev_stride, n_banks, vocab_size, embed_dim)
        if (w1_bf16 && w2_bf16)   PULSAR_MARKOV_BANKS_LAUNCH(true, true);
        else if (w1_bf16)         PULSAR_MARKOV_BANKS_LAUNCH(true, false);
        else if (w2_bf16)         PULSAR_MARKOV_BANKS_LAUNCH(false, true);
        else                      PULSAR_MARKOV_BANKS_LAUNCH(false, false);
#undef PULSAR_MARKOV_BANKS_LAUNCH
        dspark_markov_reduce_banks_kernel<<<n_banks, 256>>>(
            ids + pos + 1, ids_stride,
            (const int32_t *)bb.id->ptr, (const float *)bb.val->ptr,
            grid_dim);
    }
    return cudaGetLastError() == cudaSuccess;
}

int pulsar_gpu_dspark_markov_chain_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t n_draft, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_bf16) {
    if (!dspark_markov_banks_args_ok(refined_logits, ids_dev, ids_stride, base_logits,
                                     base_row_stride_bytes, base_row_dev, n_banks, n_draft,
                                     vocab_size, embed_dim))
        return 0;
    return dspark_markov_banks_launch(refined_logits, ids_dev, ids_stride, base_logits,
                                      base_row_stride_bytes, base_row_dev, dspark_model_map,
                                      dspark_model_size, markov_w1_offset, markov_w2_offset,
                                      n_banks, 0u, n_draft, NULL, vocab_size, embed_dim,
                                      w1_bf16, w2_bf16);
}

int pulsar_gpu_dspark_markov_step_banks_model(
        pulsar_gpu_tensor *refined_logits, pulsar_gpu_tensor *ids_dev, uint32_t ids_stride,
        const pulsar_gpu_tensor *base_logits, uint64_t base_row_stride_bytes,
        const pulsar_gpu_tensor *base_row_dev, const pulsar_gpu_tensor *prev_dev,
        const void *dspark_model_map, uint64_t dspark_model_size,
        uint64_t markov_w1_offset, uint64_t markov_w2_offset,
        uint32_t n_banks, uint32_t pos, uint32_t vocab_size, uint32_t embed_dim,
        int w1_bf16, int w2_bf16) {
    if (!prev_dev || prev_dev->bytes < (uint64_t)n_banks * sizeof(int32_t)) return 0;
    if (!dspark_markov_banks_args_ok(refined_logits, ids_dev, ids_stride, base_logits,
                                     base_row_stride_bytes, base_row_dev, n_banks, pos + 1,
                                     vocab_size, embed_dim))
        return 0;
    return dspark_markov_banks_launch(refined_logits, ids_dev, ids_stride, base_logits,
                                      base_row_stride_bytes, base_row_dev, dspark_model_map,
                                      dspark_model_size, markov_w1_offset, markov_w2_offset,
                                      n_banks, pos, 1u, prev_dev, vocab_size, embed_dim,
                                      w1_bf16, w2_bf16);
}


/* ---- L158 (2026-09-03): the drafter's concat of three target hidden states,
 * emitted as E4M3 at the PRODUCER -----------------------------------------
 *
 * The concat [h0 | h1 | h2] (3 x N_EMBD f32) fed main_proj through the dense
 * GEMV as an f32 activation for the whole A8 campaign: three tensor copies
 * built the f32 row, nothing armed a slot, and the GEMV fell to the f32
 * kernel deleted in L158 -- the one W8A32 projection left in the served lane,
 * on the drafter.  A8 is producer-side: this kernel writes the E4M3 data and
 * UE8M0 block scales straight into the activation slot with the shared
 * pulsar_mx_emit_block (same encoder, same bytes every other producer emits),
 * one warp per 32-block; N_EMBD is a multiple of 32, so no block straddles two
 * sources.  The f32 concat is never written -- the caller declares the f32
 * store skipped and the GEMV reads the slot. */
__global__ static void dspark_concat3_e4m3_kernel(const float *h0, const float *h1, const float *h2,
                                                  uint32_t E, __nv_fp8_e4m3 *data,
                                                  unsigned char *scale, int KBp) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = warp * 32u + lane;
    if (col >= 3u * E) return;
    const uint32_t src = col / E, off = col - src * E;
    const float v = src == 0 ? h0[off] : src == 1 ? h1[off] : h2[off];
    pulsar_mx_emit_block(v, col, 0u, 3u * E, KBp, data, scale);
}

int pulsar_gpu_dspark_concat3_e4m3(void *slot_data, void *slot_scale, int sf_pitch,
                                   const pulsar_gpu_tensor *h0, const pulsar_gpu_tensor *h1,
                                   const pulsar_gpu_tensor *h2, uint32_t n_embd) {
    if (!slot_data || !slot_scale || !h0 || !h1 || !h2 || n_embd == 0 || n_embd % 32u != 0) return 0;
    const uint64_t need = (uint64_t)n_embd * sizeof(float);
    if (h0->bytes < need || h1->bytes < need || h2->bytes < need) return 0;
    const uint32_t warps = 3u * n_embd / 32u;
    dspark_concat3_e4m3_kernel<<<(warps * 32u + 255u) / 256u, 256>>>(
        (const float *)h0->ptr, (const float *)h1->ptr, (const float *)h2->ptr, n_embd,
        (__nv_fp8_e4m3 *)slot_data, (unsigned char *)slot_scale, sf_pitch);
    return cuda_ok(cudaGetLastError(), "dspark concat3 e4m3");
}

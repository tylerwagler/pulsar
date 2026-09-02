#include "pulsar_cuda_internal.h"


/*
 * Each block reduces the TOP-2 over its slice of the vocabulary and writes the
 * partial winner + runner-up to block_best_id/val[blockIdx.x] and
 * block_second_id/val[blockIdx.x].  The kernel does NOT reduce across blocks —
 * the host caller reads the grid_dim partials back and merges to the global
 * top-2.  (An earlier version wrote only element 0 into a 4-byte buffer, which
 * both overran the allocation and returned block 0's argmax over just the first
 * blockDim entries.)
 *
 * DTree Phase 0: the top-1 (best) track is BYTE-IDENTICAL to the prior
 * argmax-only kernel — every best_* update below is unchanged and the added
 * second_* track never feeds back into it — so refined_id_dst is bit-exact with
 * or without the runner-up requested.  The runner-up (drafter #2) is used only
 * when the host passes a non-NULL refined_id2_dst (measurement path).
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
        int32_t *block_second_id,
        float *block_second_val,
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
    float second_val = -INFINITY;
    int32_t second_id = 0;

    for (uint32_t v = threadIdx.x + blockIdx.x * blockDim.x; v < vocab_size;
         v += blockDim.x * gridDim.x) {
        float dot = 0.0f;
        const uint64_t w2_base = (uint64_t)v * embed_dim;
        for (uint32_t i = 0; i < embed_dim; i++)
            dot += pulsar_w_load_f32_or_bf16<W2BF16>(markov_w2, w2_base + i) *
                   pulsar_w_load_f32_or_bf16<W1BF16>(markov_w1, embed_base + i);
        float val = base_logits[v] + dot;
        refined_logits[v] = val;
        if (val > best_val) { second_val = best_val; second_id = best_id;
                              best_val = val; best_id = (int32_t)v; }
        else if (val > second_val) { second_val = val; second_id = (int32_t)v; }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    __shared__ float sec_vals[256];
    __shared__ int32_t sec_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    sec_vals[tid] = second_val;
    sec_ids[tid] = second_id;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            /* Merge two sorted (best>=sec) pairs into the top-2 at tid. The
             * best branch is exactly the original strict-'>' update, so best_*
             * evolves identically to the argmax-only kernel. */
            const float ob = best_vals[tid];        const int32_t oi = best_ids[tid];
            const float os = sec_vals[tid];         const int32_t oj = sec_ids[tid];
            const float nb = best_vals[tid + stride]; const int32_t ni = best_ids[tid + stride];
            const float ns = sec_vals[tid + stride]; const int32_t nj = sec_ids[tid + stride];
            if (nb > ob) {
                best_vals[tid] = nb; best_ids[tid] = ni;
                if (ob > ns) { sec_vals[tid] = ob; sec_ids[tid] = oi; }
                else         { sec_vals[tid] = ns; sec_ids[tid] = nj; }
            } else if (nb > os) {
                sec_vals[tid] = nb; sec_ids[tid] = ni;
            }
            (void)oj;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_id[blockIdx.x] = best_ids[0];
        block_best_val[blockIdx.x] = best_vals[0];
        block_second_id[blockIdx.x] = sec_ids[0];
        block_second_val[blockIdx.x] = sec_vals[0];
    }
}

/*
 * Merge the per-block top-2 partials to the global top-2 ON DEVICE, then read
 * back only the two winning ids (8 bytes, one copy).  The old path read all
 * four partial arrays to the host (4 device-serializing D2H copies per draft
 * position, 5 positions per spec step at production depth) and merged there.
 *
 * Tie-break: blocks map to ascending contiguous vocab ranges and the
 * per-block reduction breaks ties toward the lowest id.  Each thread here
 * scans its strided subset of blocks in ascending order with the same
 * strict-'>' merge, and the shared-memory tree keeps the lower-index side as
 * the incumbent, so the lowest-id global argmax is preserved exactly —
 * matching a sequential argmax over the vocab (verified logic mirror of the
 * removed host merge).  The best track is exactly the original argmax merge;
 * out_ids[1] (runner-up) feeds the optional refined_id2 path.
 */
__global__ static void dspark_markov_reduce_kernel(
        int32_t *dst_top1,          /* winner id (L108 P1: points into the
                                     * chain's device id array, so the next
                                     * step kernel reads it directly) */
        int32_t *dst_top2,          /* runner-up id, or NULL */
        const int32_t *ids,
        const float *vals,
        const int32_t *ids2,
        const float *vals2,
        uint32_t n_blocks) {
    float best_val = -INFINITY;
    int32_t best_id = 0;
    float sec_val = -INFINITY;
    int32_t sec_id = 0;
    for (uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const float nb = vals[b];  const int32_t ni = ids[b];
        const float ns = vals2[b]; const int32_t nj = ids2[b];
        if (nb > best_val) {
            if (best_val > ns) { sec_val = best_val; sec_id = best_id; }
            else               { sec_val = ns;       sec_id = nj; }
            best_val = nb; best_id = ni;
        } else if (nb > sec_val) {
            sec_val = nb; sec_id = ni;
        }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    __shared__ float sec_vals[256];
    __shared__ int32_t sec_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    sec_vals[tid] = sec_val;
    sec_ids[tid] = sec_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float ob = best_vals[tid];          const int32_t oi = best_ids[tid];
            const float os = sec_vals[tid];
            const float nb = best_vals[tid + stride]; const int32_t ni = best_ids[tid + stride];
            const float ns = sec_vals[tid + stride];  const int32_t nj = sec_ids[tid + stride];
            if (nb > ob) {
                best_vals[tid] = nb; best_ids[tid] = ni;
                if (ob > ns) { sec_vals[tid] = ob; sec_ids[tid] = oi; }
                else         { sec_vals[tid] = ns; sec_ids[tid] = nj; }
            } else if (nb > os) {
                sec_vals[tid] = nb; sec_ids[tid] = ni;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *dst_top1 = best_ids[0];
        if (dst_top2) *dst_top2 = sec_ids[0];
    }
}

static int dspark_markov_reduce_blocks(const pulsar_gpu_tensor *id_dev,
                                        const pulsar_gpu_tensor *val_dev,
                                        const pulsar_gpu_tensor *id2_dev,
                                        const pulsar_gpu_tensor *val2_dev,
                                        pulsar_gpu_tensor *out_dev, /* 2 x int32 */
                                        uint32_t grid_dim,
                                        int32_t *refined_id_dst,
                                        int32_t *refined_id2_dst) {
    dspark_markov_reduce_kernel<<<1, 256>>>(
        (int32_t *)out_dev->ptr,
        (int32_t *)out_dev->ptr + 1,
        (const int32_t *)id_dev->ptr,
        (const float *)val_dev->ptr,
        (const int32_t *)id2_dev->ptr,
        (const float *)val2_dev->ptr,
        grid_dim);
    if (cudaGetLastError() != cudaSuccess) return 0;
    int32_t out[2];
    if (!pulsar_gpu_tensor_read(out_dev, 0, out, sizeof(out))) return 0;
    *refined_id_dst = out[0];
    if (refined_id2_dst) *refined_id2_dst = out[1];
    return 1;
}


int pulsar_gpu_dspark_markov_step_model(
        pulsar_gpu_tensor *refined_logits,
        int32_t *refined_id_dst,
        int32_t *refined_id2_dst,
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
        pulsar_gpu_tensor *id, *val, *id2, *val2, *out;
        uint32_t cap;
    };
    static thread_local DsparkReduceBufs rb = {};
    if (grid_dim > rb.cap) {
        pulsar_gpu_tensor_free(rb.id);
        pulsar_gpu_tensor_free(rb.val);
        pulsar_gpu_tensor_free(rb.id2);
        pulsar_gpu_tensor_free(rb.val2);
        pulsar_gpu_tensor_free(rb.out);
        rb.id   = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        rb.val  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        rb.id2  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        rb.val2 = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        rb.out  = pulsar_gpu_tensor_alloc(2u * sizeof(int32_t));
        rb.cap  = (rb.id && rb.val && rb.id2 && rb.val2 && rb.out) ? grid_dim : 0;
    }
    if (!rb.id || !rb.val || !rb.id2 || !rb.val2 || !rb.out) return 0;

#define PULSAR_MARKOV_LAUNCH(A, B)                          \
    dspark_markov_step_kernel<A, B><<<grid_dim, block_dim>>>( \
        (float *)refined_logits->ptr,                         \
        (int32_t *)rb.id->ptr,                                \
        (float *)rb.val->ptr,                                 \
        (int32_t *)rb.id2->ptr,                               \
        (float *)rb.val2->ptr,                                \
        (const float *)base_logits->ptr,                      \
        w1, w2, (const int32_t *)staged_id->ptr, vocab_size, embed_dim)
    if (w1_bf16 && w2_bf16)   PULSAR_MARKOV_LAUNCH(true, true);
    else if (w1_bf16)         PULSAR_MARKOV_LAUNCH(true, false);
    else if (w2_bf16)         PULSAR_MARKOV_LAUNCH(false, true);
    else                      PULSAR_MARKOV_LAUNCH(false, false);
#undef PULSAR_MARKOV_LAUNCH

    int rc = 0;
    if (cudaGetLastError() == cudaSuccess)
        rc = dspark_markov_reduce_blocks(rb.id, rb.val, rb.id2, rb.val2,
                                         rb.out, grid_dim,
                                         refined_id_dst, refined_id2_dst);
    return rc;
}


/** Device buffers the chained markov walk reduces into.
 *
 * Two winners per position, not one: `id`/`val` carry the argmax the chain
 * itself follows, and `id2`/`val2` the runner-up, which is what the DTree
 * branch and the measurement path consume. Kept device-side for the whole
 * chain so the walk needs no host round trip. */
struct DsparkReduceBufsChain {
    pulsar_gpu_tensor *id,    ///< winner token id per position; the chain's own feed
                      *val,   ///< winner score per position
                      *id2,   ///< runner-up token id per position
                      *val2;  ///< runner-up score per position
    uint32_t cap;             ///< positions the buffers can hold
};

/* L108 P1: device-chained greedy markov walk.  Launches the whole n_draft
 * refine sequence with the token feed in device memory: step kernel pos p
 * reads ids[p], its reduce writes the winner to ids[p+1] (and the runner-up
 * to ids2[p+1]), which the next step kernel reads directly.  ids[0] must be
 * pre-seeded by the caller.  NO device-to-host read happens here -- the
 * caller reads ids/ids2 once after the chain (that read is the round's one
 * drafter sync).  refined_logits is a single [vocab] f32 scratch row reused
 * by every position: nothing in the greedy chain reads it back, and reuse is
 * stream-ordered.  Same kernels, same launch order, same arithmetic as the
 * single-step path -- byte-exact by construction (any gate red is a bug).
 * The SAMPLED path cannot use this: its chain routes through a host rng draw
 * per position (pulsar_sample_dist_draw), so it keeps the single-step entry
 * above (P1b would need device-side dist build + draw with rng parity). */
int pulsar_gpu_dspark_markov_chain_model(
        pulsar_gpu_tensor *refined_logits,      /* [vocab] f32 scratch */
        pulsar_gpu_tensor *ids_dev,             /* [n_draft+1] i32, [0] seeded */
        pulsar_gpu_tensor *ids2_dev,            /* [n_draft+1] i32 runner-ups */
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
    if (!refined_logits || !ids_dev || !ids2_dev || !base_logits ||
        !dspark_model_map || n_draft == 0 || n_draft > 16)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (ids_dev->bytes  < ((uint64_t)n_draft + 1) * sizeof(int32_t)) return 0;
    if (ids2_dev->bytes < ((uint64_t)n_draft + 1) * sizeof(int32_t)) return 0;
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
        pulsar_gpu_tensor_free(crb.id2);
        pulsar_gpu_tensor_free(crb.val2);
        crb.id   = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        crb.val  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        crb.id2  = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        crb.val2 = pulsar_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        crb.cap  = (crb.id && crb.val && crb.id2 && crb.val2) ? grid_dim : 0;
    }
    if (!crb.id || !crb.val || !crb.id2 || !crb.val2) return 0;

    int32_t *ids = (int32_t *)ids_dev->ptr;
    int32_t *ids2 = (int32_t *)ids2_dev->ptr;
    for (uint32_t pos = 0; pos < n_draft; pos++) {
        const float *base_row =
            (const float *)((const char *)base_logits->ptr +
                            (uint64_t)pos * base_row_stride_bytes);
#define PULSAR_MARKOV_CHAIN_LAUNCH(A, B)                        \
        dspark_markov_step_kernel<A, B><<<grid_dim, block_dim>>>( \
            (float *)refined_logits->ptr,                         \
            (int32_t *)crb.id->ptr,                               \
            (float *)crb.val->ptr,                                \
            (int32_t *)crb.id2->ptr,                              \
            (float *)crb.val2->ptr,                               \
            base_row, w1, w2, ids + pos, vocab_size, embed_dim)
        if (w1_bf16 && w2_bf16)   PULSAR_MARKOV_CHAIN_LAUNCH(true, true);
        else if (w1_bf16)         PULSAR_MARKOV_CHAIN_LAUNCH(true, false);
        else if (w2_bf16)         PULSAR_MARKOV_CHAIN_LAUNCH(false, true);
        else                      PULSAR_MARKOV_CHAIN_LAUNCH(false, false);
#undef PULSAR_MARKOV_CHAIN_LAUNCH
        dspark_markov_reduce_kernel<<<1, 256>>>(
            ids + pos + 1, ids2 + pos + 1,
            (const int32_t *)crb.id->ptr,  (const float *)crb.val->ptr,
            (const int32_t *)crb.id2->ptr, (const float *)crb.val2->ptr,
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


/* after_ffn_hc is an HC residual CARRIER (BF16 storage; task #62) — load via
 * pulsar_hc_load, accumulate in f32. Output is a plain f32 embedding row. */
__global__ static void dspark_hc_mean_reduce_kernel(
        float *out,
        const pulsar_hc_t *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += pulsar_hc_load(after_ffn_hc, (uint64_t)hc * n_embd + d);
        out[d] = sum / (float)n_hc;
    }
}

int pulsar_gpu_dspark_hc_mean_reduce(
        pulsar_gpu_tensor *out,
        const pulsar_gpu_tensor *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!out || !after_ffn_hc || n_embd == 0 || n_hc == 0) return 0;
    if (out->bytes < (uint64_t)n_embd * sizeof(float)) return 0;
    if (after_ffn_hc->bytes < (uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE) return 0;   /* carrier */

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (n_embd + block_dim - 1) / block_dim;

    dspark_hc_mean_reduce_kernel<<<grid_dim, block_dim>>>(
        (float *)out->ptr,
        (const pulsar_hc_t *)after_ffn_hc->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce");
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

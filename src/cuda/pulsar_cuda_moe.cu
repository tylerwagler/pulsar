#include "pulsar_cuda_internal.h"
#ifdef PULSAR_HAVE_MMQ
#include "mmq/ds4_mmq.h"     /* vendored llama.cpp MMQ adapter -- see mmq/VENDOR.md */

/* Every MoE tier below that gathers or stages from the raw f32 activation must
 * refuse if that buffer's f32 stores were skipped (L089): a gather reads
 * arbitrary rows, so ANY absent row is a silent wrong answer.  The corrupting
 * first flight of the ffn store-skip came through exactly these reads. */
#define PULSAR_MOE_F32_GUARD(xptr, ntok, indim, tag)                             \
    do {                                                                          \
        if (pulsar_gpu_act_f32_first_present_row((xptr), (ntok), (indim))) {      \
            fprintf(stderr, "pulsar: %s would gather from a SKIPPED f32 store "   \
                            "(n_tok=%u in_dim=%u) -- refusing\n",                \
                    (tag), (unsigned)(ntok), (unsigned)(indim));                  \
            return 0;                                                             \
        }                                                                         \
    } while (0)
#endif











































































__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}



/* Exclusive prefix sum over the per-expert counts.  This ran as <<<1,1>>>: one
 * CUDA thread walking 256 experts serially, on every CUTLASS layer of every
 * step, with two more launches either side of it.  Now a single block scans in
 * shared memory.
 *
 * The results are IDENTICAL, not merely equivalent: these are uint32 counts and
 * integer addition is associative and exact, so reassociating the sum cannot
 * move a value.  That is why this is safe to change under a bit-exactness
 * regime where the same edit on floats would not be.
 *
 * Chunked so expert_count > blockDim still works; the model's 256 fits one
 * pass. */
__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
    __shared__ uint32_t sh[256];
    __shared__ uint32_t base;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) base = 0u;
    __syncthreads();

    for (uint32_t chunk = 0; chunk < expert_count; chunk += 256u) {
        const uint32_t e = chunk + tid;
        const uint32_t v = (e < expert_count) ? counts[e] : 0u;
        sh[tid] = v;
        __syncthreads();
        for (uint32_t off = 1u; off < 256u; off <<= 1) {
            const uint32_t add = (tid >= off) ? sh[tid - off] : 0u;
            __syncthreads();
            sh[tid] += add;
            __syncthreads();
        }
        if (e < expert_count) {
            const uint32_t excl = base + sh[tid] - v;   /* inclusive -> exclusive */
            offsets[e] = excl;
            cursors[e] = excl;
        }
        __syncthreads();
        if (tid == 255u) base += sh[255];
        __syncthreads();
    }
    if (tid == 0) offsets[expert_count] = base;
}



__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}















































__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}









/* ---- CUTLASS MXFP4 (type 40) grouped-expert dispatch. --------------------------------------
 * Unlike the per-expert paths above, which process each (token,expert) pair independently on
 * device, CUTLASS's dense GemmUniversal interface needs one contiguous [T_e,in_dim] activation
 * matrix per expert with T_e known host-side at launch time. So this path: (1) sorts tokens by
 * expert (it builds its own sorted pairs; the routed path's build was deleted), (2) reads the
 * resulting per-expert offsets back to
 * host (the one sync point this path has that the others don't -- n_total_expert is small, so
 * this is a bounded single readback per CUTLASS layer per forward pass, not per expert), then
 * (3) for each active expert: gather its rows into a contiguous scratch buffer, run the CUTLASS
 * FFN, scatter the result into a [n_tokens*n_expert,out_dim] flat buffer at the pair's slot --
 * reusing `down` as that flat buffer and moe_sum_kernel for the final per-token reduction is
 * exactly the existing convention other paths use, just fed from CUTLASS instead of dp4a. */

__global__ static void moe_cutlass_gather_kernel(
        float *x_gathered,
        float *w_gathered,
        const float *x,
        const float *weights,
        const uint32_t *sorted_pairs,
        uint32_t pair_offset,
        uint32_t count,
        uint32_t n_expert,
        uint32_t in_dim) {
    uint32_t i = blockIdx.x;
    if (i >= count) return;
    uint32_t pair = sorted_pairs[pair_offset + i];
    uint32_t tok = pair / n_expert;
    const float *src = x + (uint64_t)tok * in_dim;
    float *dst = x_gathered + (uint64_t)i * in_dim;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) dst[k] = src[k];
    if (threadIdx.x == 0) w_gathered[i] = weights[pair];
}



__global__ static void moe_cutlass_scatter_kernel(
        float *down_flat,
        const float *ffn_out,
        const uint32_t *sorted_pairs,
        uint32_t pair_offset,
        uint32_t count,
        uint32_t out_dim) {
    uint32_t i = blockIdx.x;
    if (i >= count) return;
    uint32_t pair = sorted_pairs[pair_offset + i];
    const float *src = ffn_out + (uint64_t)i * out_dim;
    float *dst = down_flat + (uint64_t)pair * out_dim;
    for (uint32_t k = threadIdx.x; k < out_dim; k += blockDim.x) dst[k] = src[k];
}



static uint64_t cutlass_moe_align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/* gate_stride/gate_data_bytes and down_stride/down_data_bytes come from
 * routed_expert_gate_down_layout()'s CUTLASS_MXFP4 branch: *_stride is the full per-expert
 * [data+SF] block size, *_data_bytes is where the SF blob starts within that block (the
 * "row_bytes" parameter slot, repurposed -- see that function's comment in weights.cpp). */
/* L106 K12 (audited, WON'T-DO): this per-expert loop gathers f32 rows and
 * lets the cutlass pack re-derive the E4M3 the act cache already holds -- a
 * duplicate encode its grouped sibling (:607) and the mixed-40 path avoid.
 * Taking the handover here means threading act_q/act_sf through the whole
 * per-expert pulsar_cutlass_expert_ffn API for a path that is an ANNOUNCED
 * ~4x-slower fallback (fires only when the grouped GEMM fails).  Bytes are
 * identical either way (encoders verified byte-identical); the waste is real
 * but only on a path whose firing is itself the alarm.  Revisit only if the
 * "per-expert loop" warning is ever observed in production. */
static int routed_moe_launch_cutlass(
        pulsar_gpu_tensor *out,
        pulsar_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_stride,
        uint64_t gate_data_bytes,
        uint64_t down_stride,
        uint64_t down_data_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const pulsar_gpu_tensor *selected,
        const pulsar_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const pulsar_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_cutlass_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_cutlass_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_cutlass_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);

    /* Safe upper bound for the per-expert FFN scratch: one expert could receive every token
     * in this batch. Sizing at n_tokens (rather than the true, smaller max observed count)
     * means everything fits in ONE cuda_tmp_alloc call -- cuda_tmp_alloc is a single global
     * buffer that may cudaFree-and-regrow on a size increase, so a second call after reading
     * back real per-expert counts could invalidate the sorted_pairs region we still need. */
    const uint32_t T_max = n_tokens;
    const uint64_t gather_x_bytes = (uint64_t)T_max * expert_in_dim * sizeof(float);
    const uint64_t gather_w_bytes = (uint64_t)T_max * sizeof(float);
    const uint64_t ffn_out_bytes = (uint64_t)T_max * out_dim * sizeof(float);
    const uint64_t ffn_scratch_bytes = pulsar_cutlass_expert_ffn_scratch_bytes(
            (int)T_max, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim);

    /* Eight buffers out of one reservation.  This used to be eight hand-written
     * align_up steps followed by eight pointer reconstructions; the arena does
     * the same bump in one place and refuses (latching) rather than handing back
     * a slice that overlaps its neighbour.  Sizing is unchanged: take() aligns
     * each slice's START, which lands on exactly the offsets the old end-aligned
     * chain produced, because the first slice began at 0. */
    const uint64_t align = 256;
    const uint64_t total_scratch =
        cutlass_moe_align_up(counts_bytes, align) +
        cutlass_moe_align_up(offsets_bytes, align) +
        cutlass_moe_align_up(cursors_bytes, align) +
        cutlass_moe_align_up(sorted_bytes, align) +
        cutlass_moe_align_up(gather_x_bytes, align) +
        cutlass_moe_align_up(gather_w_bytes, align) +
        cutlass_moe_align_up(ffn_out_bytes, align) +
        cutlass_moe_align_up(ffn_scratch_bytes, align);

    cuda_arena ar;
    if (!cuda_arena_begin(&ar, total_scratch, "routed_moe cutlass")) return 0;
    uint32_t *counts       = (uint32_t *)cuda_arena_take(&ar, counts_bytes, align);
    uint32_t *offsets      = (uint32_t *)cuda_arena_take(&ar, offsets_bytes, align);
    uint32_t *cursors      = (uint32_t *)cuda_arena_take(&ar, cursors_bytes, align);
    uint32_t *sorted_pairs = (uint32_t *)cuda_arena_take(&ar, sorted_bytes, align);
    float    *x_gathered   = (float *)cuda_arena_take(&ar, gather_x_bytes, align);
    float    *w_gathered   = (float *)cuda_arena_take(&ar, gather_w_bytes, align);
    float    *ffn_out      = (float *)cuda_arena_take(&ar, ffn_out_bytes, align);
    uint8_t  *ffn_scratch  = (uint8_t *)cuda_arena_take(&ar, ffn_scratch_bytes, align);
    /* One check covers all eight: take() latches on the first refusal, so a
     * partial success cannot produce aliased pointers. */
    if (!ffn_scratch) return 0;

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    /* L119: Async (capture-legal); this per-expert path is a fallback but must
     * not break a capture that reaches it. Its D2H offsets readback below
     * remains capture-illegal by design — grouped is the captured path. */
    int ok = cuda_ok(cudaMemsetAsync(counts, 0, counts_bytes), "routed_moe_cutlass counts clear");
    if (ok) {
        moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass count launch");
    }
    if (ok) {
        moe_prefix_sorted_pairs_kernel<<<1, 256>>>(offsets, cursors, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass prefix launch");
    }
    if (ok) {
        moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass scatter launch");
    }
    if (!ok) return 0;

    std::vector<uint32_t> h_offsets((size_t)n_total_expert + 1);
    if (!cuda_ok(cudaMemcpy(h_offsets.data(), offsets, offsets_bytes, cudaMemcpyDeviceToHost),
                "routed_moe_cutlass offsets readback")) {
        return 0;
    }

    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t e_offset = h_offsets[e];
        const uint32_t e_count = h_offsets[e + 1] - e_offset;
        if (e_count == 0) continue;

        const uint8_t *Wg_d = (const uint8_t *)gate_w + (uint64_t)e * gate_stride;
        const uint8_t *Wg_sf = Wg_d + gate_data_bytes;
        const uint8_t *Wu_d = (const uint8_t *)up_w + (uint64_t)e * gate_stride;
        const uint8_t *Wu_sf = Wu_d + gate_data_bytes;
        const uint8_t *Wd_d = (const uint8_t *)down_w + (uint64_t)e * down_stride;
        const uint8_t *Wd_sf = Wd_d + down_data_bytes;

        /* One expert can receive MORE than n_tokens pairs when routing carries
         * duplicates (e.g. tid2eid -1 "dropped" entries all clamp to expert 0),
         * but the gather/ffn scratch is sized for T_max = n_tokens rows -- so
         * run the expert in <=T_max-row slices. Every row is computed
         * independently (per-row GEMM dot products, per-row swiglu), so slicing
         * is bit-identical to a single full-count pass; with well-formed
         * routing (count <= n_tokens) this loop body runs exactly once with
         * the same arguments as before. */
        for (uint32_t done = 0; done < e_count; done += T_max) {
            const uint32_t pair_offset = e_offset + done;
            const uint32_t count = (e_count - done < T_max) ? (e_count - done) : T_max;

            PULSAR_MOE_F32_GUARD(x->ptr, n_tokens, expert_in_dim, "moe per-expert gather");
            moe_cutlass_gather_kernel<<<count, 256>>>(x_gathered, w_gathered,
                    (const float *)x->ptr, (const float *)weights->ptr,
                    sorted_pairs, pair_offset, count, n_expert, expert_in_dim);
            if (!cuda_ok(cudaGetLastError(), "routed_moe_cutlass gather launch")) return 0;

            const int rc = pulsar_cutlass_expert_ffn_scratch(ffn_out, x_gathered,
                    Wg_d, Wg_sf, Wu_d, Wu_sf, Wd_d, Wd_sf,
                    w_gathered, clamp,
                    (int)count, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                    ffn_scratch, ffn_scratch_bytes);
            if (rc != 0) return 0;

            moe_cutlass_scatter_kernel<<<count, 256>>>((float *)down->ptr, ffn_out,
                    sorted_pairs, pair_offset, count, out_dim);
            if (!cuda_ok(cudaGetLastError(), "routed_moe_cutlass scatter launch")) return 0;
        }
    }

    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "routed_moe_cutlass sum launch");
}



/* ---- Grouped MXFP4 prefill path (single ptr-array grouped GEMM per gate/up/down; no host
 * readback, no host sync). Mirrors routed_moe_launch_cutlass but replaces the per-expert loop
 * with pulsar_cutlass_grouped_moe. Experts' tokens are gathered to 128-ROW-PADDED offsets so each
 * group's block-scaled SF starts on a 128-row atom boundary (see pulsar_cutlass_grouped_moe). ---- */

/* padded_offsets[e] = sum_{e'<e} ceil(counts[e']/128)*128 (per-group 128-row-padded row base). */
__global__ static void moe_padded_offsets_kernel(uint32_t *padded_off, const uint32_t *counts, uint32_t n_total) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t run = 0;
    for (uint32_t e = 0; e < n_total; e++) {
        padded_off[e] = run;
        run += (counts[e] + 127u) / 128u * 128u;
    }
}

/* One block per sorted-pair slot s: locate its expert e (offsets[e] <= s < offsets[e+1]), place
 * the token's activation row at padded row padded_off[e] + (s - offsets[e]), record the routing
 * weight and the originating pair index (padded_pair[R]). The grouped path also records the
 * INVERSE map (pair_slot[pair] = R) so its reduction reads the padded GEMM output directly;
 * the mixed path passes pair_slot = NULL and still scatters through padded_pair. */
/* row_src_tok[R] records which token row R came from, so a consumer that already
 * has that token encoded (the CUTLASS path reads the activation cache's E4M3)
 * can gather the codes itself.  When it does, x_gathered is NULL and the f32 row
 * copy below -- the expensive half of this kernel -- is skipped entirely. */
__global__ static void moe_padded_gather_kernel(
        float *x_gathered, float *w_gathered, int32_t *padded_pair, int32_t *row_src_tok,
        int32_t *pair_slot,
        const float *x, const float *weights,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *padded_off,
        uint32_t pair_count, uint32_t n_total, uint32_t n_expert, uint32_t in_dim) {
    uint32_t s = blockIdx.x;
    if (s >= pair_count) return;
    /* upper_bound(offsets, s) - 1 over [0,n_total) */
    uint32_t lo = 0, hi = n_total;
    while (lo < hi) {
        uint32_t midx = (lo + hi) >> 1;
        if (offsets[midx] <= s) lo = midx + 1; else hi = midx;
    }
    uint32_t e = lo - 1u;
    uint32_t i = s - offsets[e];
    uint32_t R = padded_off[e] + i;
    uint32_t pair = sorted_pairs[s];
    uint32_t tok = pair / n_expert;
    if (x_gathered) {
        const float *src = x + (uint64_t)tok * in_dim;
        float *dst = x_gathered + (uint64_t)R * in_dim;
        for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) dst[k] = src[k];
    }
    if (threadIdx.x == 0) {
        w_gathered[R] = weights[pair];
        padded_pair[R] = (int32_t)pair;
        if (pair_slot) pair_slot[pair] = (int32_t)R;
        if (row_src_tok) row_src_tok[R] = (int32_t)tok;
    }
}

/* Per-token reduction reading the grouped GEMM's 128-row-padded output
 * DIRECTLY through the pair->padded-row map, in the same ascending-expert
 * order as moe_sum_kernel: identical values, identical addition order,
 * bit-exact. This is what lets the grouped path skip the scatter's full
 * round trip of the flat down buffer (a write + re-read of
 * n_tokens*n_expert*out_dim f32 per layer). Rows a pair never claimed carry
 * -1, matching the padded_pair<0 convention, and contribute nothing. */
__global__ static void moe_sum_padded_kernel(
        float *out, const float *ffn_out, const int32_t *pair_slot,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const int32_t R = pair_slot[(uint64_t)tok * n_expert + e];
        if (R >= 0) acc += ffn_out[(uint64_t)R * out_dim + row];
    }
    out[gid] = acc;
}

/* One block per padded row R: scatter the pre-weighted FFN result back to the flat down buffer at
 * its originating pair (padding rows carry padded_pair<0 and are skipped). */
__global__ static void moe_padded_scatter_kernel(
        float *down_flat, const float *ffn_out, const int32_t *padded_pair,
        uint32_t padded_total, uint32_t out_dim) {
    uint32_t R = blockIdx.x;
    if (R >= padded_total) return;
    int32_t pair = padded_pair[R];
    if (pair < 0) return;
    const float *src = ffn_out + (uint64_t)R * out_dim;
    float *dst = down_flat + (uint64_t)pair * out_dim;
    for (uint32_t k = threadIdx.x; k < out_dim; k += blockDim.x) dst[k] = src[k];
}

static int routed_moe_launch_cutlass_grouped(
        pulsar_gpu_tensor *out,
        pulsar_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_stride,
        uint64_t gate_data_bytes,
        uint64_t down_stride,
        uint64_t down_data_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const pulsar_gpu_tensor *selected,
        const pulsar_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const pulsar_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_grouped_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_grouped_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_grouped_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    /* Host upper bound on padded rows: every active expert adds <=127 padding; round to 128. */
    const uint64_t padded_upper = (((uint64_t)pair_count + 128ull * n_total_expert + 127ull) / 128ull) * 128ull;

    /* The norm that produced x already emitted its E4M3 + ue8m0 encoding. When
     * it is still cached we hand it to the grouped GEMM and gather the 1-byte
     * codes, which drops both the f32 row copy here and the re-encode there --
     * the same values, encoded once instead of twice. A miss is not an error;
     * it just means we take the f32 path we always took. */
    const void *act_q = NULL, *act_sf = NULL;
    int act_kbp = 0;
    const bool cached_act = pulsar_gpu_mxfp8_act_cache_get_e4m3(x, n_tokens, expert_in_dim,
                                                                &act_q, &act_sf, &act_kbp) != 0;

    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
    const uint64_t padoff_bytes = counts_bytes;
    /* No f32 gather buffer at all on the cached path -- nothing reads it. */
    const uint64_t xg_bytes = cached_act ? 0 : padded_upper * expert_in_dim * sizeof(float);
    const uint64_t wg_bytes = padded_upper * sizeof(float);
    const uint64_t ppair_bytes = padded_upper * sizeof(int32_t);
    const uint64_t pslot_bytes = (uint64_t)pair_count * sizeof(int32_t);
    const uint64_t rsrc_bytes = cached_act ? padded_upper * sizeof(int32_t) : 0;
    const uint64_t ffn_bytes = padded_upper * out_dim * sizeof(float);
    const uint64_t grp_bytes = pulsar_cutlass_grouped_moe_scratch_bytes(
            (int)padded_upper, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim);

    const uint64_t align = 256;
    /* Eleven buffers from one reservation.  Two are used only on one side of
     * `cached_act`, but BOTH are still taken so the layout is independent of
     * that flag -- the old code reserved space for both unconditionally and only
     * the POINTERS were conditional, and keeping that property means this
     * conversion cannot change what any kernel sees. */
    const uint64_t total_scratch =
        cutlass_moe_align_up(counts_bytes, align) +
        cutlass_moe_align_up(offsets_bytes, align) +
        cutlass_moe_align_up(cursors_bytes, align) +
        cutlass_moe_align_up(sorted_bytes, align) +
        cutlass_moe_align_up(padoff_bytes, align) +
        cutlass_moe_align_up(xg_bytes, align) +
        cutlass_moe_align_up(wg_bytes, align) +
        cutlass_moe_align_up(ppair_bytes, align) +
        cutlass_moe_align_up(pslot_bytes, align) +
        cutlass_moe_align_up(rsrc_bytes, align) +
        cutlass_moe_align_up(ffn_bytes, align) +
        cutlass_moe_align_up(grp_bytes, align);

    cuda_arena ar;
    if (!cuda_arena_begin(&ar, total_scratch, "routed_moe grouped")) return 0;
    uint32_t *counts       = (uint32_t *)cuda_arena_take(&ar, counts_bytes, align);
    uint32_t *offsets      = (uint32_t *)cuda_arena_take(&ar, offsets_bytes, align);
    uint32_t *cursors      = (uint32_t *)cuda_arena_take(&ar, cursors_bytes, align);
    uint32_t *sorted_pairs = (uint32_t *)cuda_arena_take(&ar, sorted_bytes, align);
    uint32_t *padded_off   = (uint32_t *)cuda_arena_take(&ar, padoff_bytes, align);
    float    *x_gathered   = (float *)cuda_arena_take(&ar, xg_bytes, align);
    float    *w_gathered   = (float *)cuda_arena_take(&ar, wg_bytes, align);
    int32_t  *padded_pair  = (int32_t *)cuda_arena_take(&ar, ppair_bytes, align);
    int32_t  *pair_slot    = (int32_t *)cuda_arena_take(&ar, pslot_bytes, align);
    int32_t  *row_src_tok  = (int32_t *)cuda_arena_take(&ar, rsrc_bytes, align);
    float    *ffn_out      = (float *)cuda_arena_take(&ar, ffn_bytes, align);
    uint8_t  *grp_scratch  = (uint8_t *)cuda_arena_take(&ar, grp_bytes, align);
    if (!grp_scratch) return 0;   /* take() latches: one check covers all twelve */
    /* The two flag-dependent operands, exactly as before. */
    if (cached_act) x_gathered = NULL;
    else            row_src_tok = NULL;

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    /* L119: Async — the sync form is a stream-capture violation and this path
     * is reached by captured decode segments at rows > 8 (same stream, same
     * ordering; behavior unchanged). */
    int ok = cuda_ok(cudaMemsetAsync(counts, 0, counts_bytes), "moe_grouped counts clear");
    /* padding rows must be zeroed (pack sees clean data) and unmapped (padded_pair = -1). */
    if (ok && x_gathered) ok = cuda_ok(cudaMemsetAsync(x_gathered, 0, xg_bytes), "moe_grouped xg clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(w_gathered, 0, wg_bytes), "moe_grouped wg clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(padded_pair, 0xFF, ppair_bytes), "moe_grouped ppair clear");
    /* -1: a pair the gather never claims must read as "no row" in the padded
     * sum, not as padded row 0. Every real pair is written by the gather. */
    if (ok) ok = cuda_ok(cudaMemsetAsync(pair_slot, 0xFF, pslot_bytes), "moe_grouped pslot clear");
    /* -1 everywhere: rows the gather never visits ARE the padding rows, and the
     * E4M3 gather zero-fills exactly those (what the pre-zeroed f32 rows gave). */
    if (ok && row_src_tok) ok = cuda_ok(cudaMemsetAsync(row_src_tok, 0xFF, rsrc_bytes),
                                        "moe_grouped rsrc clear");
    if (ok) {
        moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped count launch");
    }
    if (ok) {
        moe_prefix_sorted_pairs_kernel<<<1, 256>>>(offsets, cursors, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped prefix launch");
    }
    if (ok) {
        moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped scatter launch");
    }
    if (ok) {
        moe_padded_offsets_kernel<<<1, 1>>>(padded_off, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped padded offsets launch");
    }
    if (ok) {
        if (x_gathered)
            PULSAR_MOE_F32_GUARD(x->ptr, n_tokens, expert_in_dim, "moe grouped uncached gather");
        moe_padded_gather_kernel<<<pair_count, 256>>>(x_gathered, w_gathered, padded_pair, row_src_tok,
                pair_slot,
                (const float *)x->ptr, (const float *)weights->ptr,
                sorted_pairs, offsets, padded_off, pair_count, n_total_expert, n_expert, expert_in_dim);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped gather launch");
    }
    if (ok) {
        int rc = pulsar_cutlass_grouped_moe(ffn_out, x_gathered, w_gathered,
                (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                gate_stride, gate_data_bytes, down_stride, down_data_bytes,
                clamp, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                counts, padded_off, (int)padded_upper, grp_scratch, grp_bytes,
                act_q, act_sf, act_kbp, row_src_tok);
        if (rc != 0) return 0;
    }
    if (!ok) return 0;

    /* Reduce straight from the padded GEMM output through pair_slot -- the
     * flat `down` buffer is NOT touched on this path (its scatter + re-read
     * used to move 2*n_tokens*n_expert*out_dim f32 per layer for nothing).
     * The per-expert fallback and the MMQ arms still write `down`. */
    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_padded_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, ffn_out, pair_slot, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "moe_grouped padded sum launch");
}

/* Grouped single-launch path, with the legacy per-expert loop kept as the
 * fallback when it fails. Same result, bit-exact. */
static int routed_moe_launch_cutlass_dispatch(
        pulsar_gpu_tensor *out, pulsar_gpu_tensor *down, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_stride, uint64_t gate_data_bytes, uint64_t down_stride, uint64_t down_data_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const pulsar_gpu_tensor *selected, const pulsar_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const pulsar_gpu_tensor *x, uint32_t n_tokens) {
    {
        int rc = routed_moe_launch_cutlass_grouped(out, down, model_map, model_size,
                gate_offset, up_offset, down_offset, gate_stride, gate_data_bytes,
                down_stride, down_data_bytes, expert_in_dim, expert_mid_dim, out_dim,
                selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
        {
            static int glog = -1;
            if (glog < 0) glog = getenv("PULSAR_MOE_GROUPED_LOG") != NULL ? 1 : 0;
            static int logged = 0;
            if (glog && !logged) { logged = 1;
                fprintf(stderr, "pulsar: moe grouped path rc=%d n_tok=%u n_total=%u n_exp=%u -> %s\n",
                        rc, n_tokens, n_total_expert, n_expert, rc ? "USED grouped" : "FELL BACK to per-expert"); }
        }
        if (rc) return rc;   /* any failure falls through to the legacy loop (safety net) */
    }
    /* Reached only when the grouped path FAILED -> the ~4x slower per-expert
     * loop.  Announce once, unconditionally, so this slow tier is never silent. */
    {
        static int fb_logged = 0;
        if (!fb_logged) { fb_logged = 1;
            fprintf(stderr,
                    "pulsar: WARNING MoE grouped GEMM failed -> per-expert loop "
                    "(~4x slower prefill)\n");
        }
    }
    return routed_moe_launch_cutlass(out, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_stride, gate_data_bytes,
            down_stride, down_data_bytes, expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
}



/* ---- MIXED type-40 (CUTLASS W4A8) + type-43 (MMQ) layer dispatch. --------------------------------
 * Some layers have ONE projection as CUTLASS MXFP4 (type 40) and the other as IQ2_XXS_MMQ (type 43);
 * the artifact has 7 such layers (4x gate=40/down=43, 3x gate=43/down=40). The grouped path needs
 * all-three-same-type and CUTLASS cannot read the MMQ layout (or vice versa), so we run each
 * projection in its native precision and compose in f32. The CUTLASS side runs as a GROUPED
 * single-projection GEMM over 128-row-padded gathered activations (pulsar_cutlass_grouped_proj) --
 * device-built ptr-array, NO host readback, NO per-expert sync -- the same machinery the uniform
 * grouped path uses. The MMQ side stays in the per-(token,expert) pair layout; padded gather/scatter
 * bridge the two:
 *   Case A (gate/up type-40, down type-43): padded-gather x -> grouped W4A8 gate & up GEMMs -> swiglu
 *     (+routing weight) on the gathered rows -> padded-scatter mid into pair layout -> MMQ down per
 *     pair (takes mid as f32, does its own q8_1) -> moe_sum.
 *   Case B (gate/up type-43, down type-40): MMQ gate/up per pair (fused swiglu) -> padded-gather mid
 *     -> grouped W4A8 down GEMM -> padded-scatter into pair layout -> moe_sum.
 * The grouped CUTLASS projection is bit-identical to the per-expert single-proj path it replaces
 * (same pack + same g_build_arrays gather order + same GEMM, just batched). No cross-contamination:
 * each projection reads only its own weight/SF and its own activation quant.
 *
 * Both MMQ arms fail closed if MMQ declines: the dp4a kernels that used to back them up are gone
 * with types 16/42/39/10, and no other reader understands the aligned layout. */
__global__ static void moe_swiglu_gathered_kernel(
        float *mid_g, const float *gate_g, const float *up_g, const float *w_g,
        float clamp, uint32_t mid_dim, uint32_t n_rows) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * mid_dim;
    if (i >= n) return;
    float g = gate_g[i], u = up_g[i];
    if (clamp > 1.0e-6f) { if (g > clamp) g = clamp; if (u > clamp) u = clamp; if (u < -clamp) u = -clamp; }
    mid_g[i] = (g / (1.0f + expf(-g))) * u * w_g[i / mid_dim];   /* SwiGLU order matches the other MoE gate/up paths */
}

/* 128-padded gather of a PAIR-LAYOUT f32 buffer (src[pair*dim]) into the grouped-GEMM activation
 * layout -- the mid-side companion to moe_padded_gather_kernel (which gathers per-TOKEN x). One block
 * per sorted-pair slot s: locate its expert e, place the row at padded_off[e]+(s-offsets[e]), record
 * the originating pair for the later padded scatter. Used to feed the CUTLASS down GEMM in case B. */
__global__ static void moe_padded_gather_pairflat_kernel(
        float *dst, int32_t *padded_pair, const float *src_pairflat,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *padded_off,
        uint32_t pair_count, uint32_t n_total, uint32_t dim) {
    uint32_t s = blockIdx.x;
    if (s >= pair_count) return;
    uint32_t lo = 0, hi = n_total;
    while (lo < hi) { uint32_t m = (lo + hi) >> 1; if (offsets[m] <= s) lo = m + 1; else hi = m; }
    uint32_t e = lo - 1u;
    uint32_t R = padded_off[e] + (s - offsets[e]);
    uint32_t pair = sorted_pairs[s];
    const float *src = src_pairflat + (uint64_t)pair * dim;
    float *d = dst + (uint64_t)R * dim;
    for (uint32_t k = threadIdx.x; k < dim; k += blockDim.x) d[k] = src[k];
    if (threadIdx.x == 0) padded_pair[R] = (int32_t)pair;
}

#ifdef PULSAR_HAVE_MMQ
/* both defined below, next to each other */
static int routed_moe_try_mmq_gate_up(float *mid_out, float *gate_scratch,
                                      const char *gate_w, const char *up_w,
                                      const float *x_f32, const int32_t *selected_ptr,
                                      const float *weights_ptr, uint32_t gate_type,
                                      uint32_t expert_in_dim, uint32_t expert_mid_dim,
                                      uint32_t n_total_expert, uint32_t n_expert,
                                      uint32_t n_tokens, float clamp);
static int routed_moe_try_mmq_down(float *down_out, const float *mid_f32,
                                   const char *down_w, const int32_t *selected_ptr,
                                   uint32_t down_type, uint32_t expert_mid_dim,
                                   uint32_t out_dim, uint32_t n_total_expert,
                                   uint64_t pairs);
#endif

static int routed_moe_launch_mixed40(
        pulsar_gpu_tensor *out, pulsar_gpu_tensor *up,
        pulsar_gpu_tensor *mid, pulsar_gpu_tensor *down,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const pulsar_gpu_tensor *selected, const pulsar_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const pulsar_gpu_tensor *x, uint32_t n_tokens) {
    const int caseA = (gate_type == 40u);       /* gate/up type-40, down MMQ-43 */
    const int caseB = (down_type == 40u);       /* gate/up MMQ-43, down type-40 */
    if (caseA == caseB) return 0;               /* exactly one side must be cutlass */
    /* ...and the OTHER side must be type 43, the only non-cutlass expert format
     * left.  This is where the artifact's 7 mixed layers land (4x gate=40/
     * down=43, 3x gate=43/down=40); the dp4a types that used to be admitted
     * here -- IQ2_XXS (16), IQ2_XXS_SOA (42), Q2_K (10), FP4_E2M1 (39) -- are
     * gone along with their kernels.
     *
     * Stated as an equality rather than a list of rejections on purpose: the
     * dispatch below is a chain keyed on a type tag, and every silent misread
     * this file has produced came from such a chain ACCEPTING an unknown tag and
     * reading its bytes as the last arm's format.  An allowlist cannot do that. */
    if ((caseA ? down_type : gate_type) != 43u) return 0;
    if (!out || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0) return 0;
    if (mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float)) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    /* Host upper bound on padded rows: #active experts <= min(n_total, pair_count), each adds <=127
     * padding; round to 128. Using min() (not n_total) keeps the gathered buffer / memset / pack
     * TIGHT at decode (pair_count=6..8 -> ~1k rows, not 128*256=32k) so the grouped path is cheap for
     * small batches too; at prefill min()==n_total so it matches the uniform grouped path's bound. */
    const uint64_t nact = (n_total_expert < pair_count) ? (uint64_t)n_total_expert : (uint64_t)pair_count;
    const uint64_t padded_upper = (((uint64_t)pair_count + 128ull * nact + 127ull) / 128ull) * 128ull;
    /* the CUTLASS side's inner (K) dim: in_dim for gate/up (A), mid_dim for down (B) */
    const uint32_t cut_k = caseA ? expert_in_dim : expert_mid_dim;
    /* Small batches (decode n=1, spec-verify n<=4) use the PER-EXPERT single-proj CUTLASS path: the
     * 128-padded grouped buffer would be almost all padding (few tokens spread over many experts),
     * so its memset/pack/grouped-launch overhead dominates. Big batches (prefill) use the grouped,
     * no-host-sync path. `rows` sizes the gather/proj buffers for whichever path is active. */
    /* plan-34 inc 2: a batched multiseq/mixed step forces the M-INDEPENDENT
     * per-token (non-grouped) path across the whole row range (up to
     * PULSAR_GPU_MNEUTRAL_ROWS_MAX, static_assert-tied to PULSAR_MSEQ_MAX):
     * the grouped path's per-expert group sizes depend on the batch composition,
     * so a co-scheduled decode bank's expert output would shift with the batch
     * width. The non-grouped CUTLASS proj uses fixed compile-time tiles (per-row
     * output independent of M) and buffers are sized by n_tokens, so any armed
     * width is safe on the same argument that covered 5..8.
     * Classic prefill (never armed) keeps the grouped, no-host-sync path at >4. */
    const int use_grouped = pulsar_gpu_matmul_batch_mneutral()
            ? (n_tokens > PULSAR_GPU_MNEUTRAL_ROWS_MAX) : (n_tokens > 4u);
    const uint64_t rows = use_grouped ? padded_upper : (uint64_t)n_tokens;

    /* model weight bases */
    const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_total > model_size - gate_offset || gate_total > model_size - up_offset ||
        down_total > model_size - down_offset) return 0;
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total, "moe_mix_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total, "moe_mix_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total, "moe_mix_down");
    if (!gate_w || !up_w || !down_w) return 0;

    /* one scratch alloc: sort/pad structures + 128-padded gather buffers + grouped-proj scratch + quant */
    const uint64_t A = 256;
    const uint64_t counts_b  = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_b = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_b = counts_b;
    const uint64_t sorted_b  = (uint64_t)pair_count * sizeof(uint32_t);
    const uint64_t padoff_b  = counts_b;
    const uint64_t xg_b   = rows * cut_k * sizeof(float);              /* gathered x(A)/mid(B) */
    const uint64_t wg_b   = rows * sizeof(float);                      /* routing weights (A) */
    const uint64_t ppair_b= rows * sizeof(int32_t);
    const uint64_t gg_b   = caseA ? rows * expert_mid_dim * sizeof(float) : 0;  /* gate_g (A) */
    const uint64_t ug_b   = caseA ? rows * expert_mid_dim * sizeof(float) : 0;  /* up_g   (A) */
    const uint64_t mg_b   = caseA ? rows * expert_mid_dim * sizeof(float) : 0;  /* mid_g  (A) */
    const uint64_t og_b   = caseB ? rows * out_dim * sizeof(float) : 0;         /* out_g  (B) */
    /* row_src_tok: only the caseA GROUPED arm gathers x, so only it can take the
     * producer handover.  caseB's grouped_proj consumes mid_g, which the GPU
     * just computed and no norm ever encoded. */
    const uint64_t rsrc_b = (use_grouped && caseA) ? rows * sizeof(int32_t) : 0;
    const uint64_t proj_b = use_grouped
        ? (caseA ? pulsar_cutlass_grouped_proj_scratch_bytes((int)padded_upper, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim)
                 : pulsar_cutlass_grouped_proj_scratch_bytes((int)padded_upper, (int)n_total_expert, (int)expert_mid_dim, (int)out_dim))
        : (caseA ? pulsar_cutlass_proj_scratch_bytes((int)n_tokens, (int)expert_in_dim, (int)expert_mid_dim)
                 : pulsar_cutlass_proj_scratch_bytes((int)n_tokens, (int)expert_mid_dim, (int)out_dim));

    uint64_t off = 0;
    /* Twelve buffers from one reservation.  The last four (gate/up/mid/out
     * gathers) are consumed further down; they are TAKEN here with the rest so
     * every slice comes from one bump sequence rather than being reconstructed
     * from a saved offset at the point of use -- which is the pattern that lets
     * a stale offset survive a layout change. */
    const uint64_t total_scratch =
        cutlass_moe_align_up(counts_b, A)  + cutlass_moe_align_up(offsets_b, A) +
        cutlass_moe_align_up(cursors_b, A) + cutlass_moe_align_up(sorted_b, A) +
        cutlass_moe_align_up(padoff_b, A)  + cutlass_moe_align_up(xg_b, A) +
        cutlass_moe_align_up(wg_b, A)      + cutlass_moe_align_up(ppair_b, A) +
        cutlass_moe_align_up(gg_b, A)      + cutlass_moe_align_up(ug_b, A) +
        cutlass_moe_align_up(mg_b, A)      + cutlass_moe_align_up(og_b, A) +
        cutlass_moe_align_up(rsrc_b, A)    + cutlass_moe_align_up(proj_b, A);

    cuda_arena ar;
    if (!cuda_arena_begin(&ar, total_scratch, "routed_moe mixed40")) return 0;
    uint32_t *counts       = (uint32_t *)cuda_arena_take(&ar, counts_b, A);
    uint32_t *offsets      = (uint32_t *)cuda_arena_take(&ar, offsets_b, A);
    uint32_t *cursors      = (uint32_t *)cuda_arena_take(&ar, cursors_b, A);
    uint32_t *sorted_pairs = (uint32_t *)cuda_arena_take(&ar, sorted_b, A);
    uint32_t *padded_off   = (uint32_t *)cuda_arena_take(&ar, padoff_b, A);
    float    *x_gathered   = (float *)cuda_arena_take(&ar, xg_b, A);
    float    *w_gathered   = (float *)cuda_arena_take(&ar, wg_b, A);
    int32_t  *padded_pair  = (int32_t *)cuda_arena_take(&ar, ppair_b, A);
    float    *gate_g_buf   = (float *)cuda_arena_take(&ar, gg_b, A);
    float    *up_g_buf     = (float *)cuda_arena_take(&ar, ug_b, A);
    float    *mid_g_buf    = (float *)cuda_arena_take(&ar, mg_b, A);
    float    *out_g_buf    = (float *)cuda_arena_take(&ar, og_b, A);
    int32_t  *row_src_tok  = (int32_t *)cuda_arena_take(&ar, rsrc_b, A);
    uint8_t  *proj_scratch = (uint8_t *)cuda_arena_take(&ar, proj_b, A);
    if (!proj_scratch) return 0;  /* take() latches: one check covers all thirteen */
    float *mid_flat = (float *)mid->ptr;        /* pair-layout mid accumulator */
    float *down_flat = (float *)down->ptr;      /* pair-layout down accumulator */

    /* padding rows: zeroed (pack sees clean data) + unmapped (padded_pair=-1).
     * L119: Async like its four siblings — the synchronous form is a CUDA
     * stream-capture violation and this clear runs on every decode round's
     * mixed-type layers (same stream, same ordering; behavior unchanged). */
    int ok = cuda_ok(cudaMemsetAsync(counts, 0, counts_b), "mixed40 counts clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(x_gathered, 0, xg_b), "mixed40 xg clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(padded_pair, 0xFF, ppair_b), "mixed40 ppair clear");
    /* -1 everywhere: rows the gather never visits ARE the padding rows, and the
     * E4M3 gather zero-fills exactly those (what pre-zeroed f32 rows gave). */
    if (ok && rsrc_b) ok = cuda_ok(cudaMemsetAsync(row_src_tok, 0xFF, rsrc_b), "mixed40 rsrc clear");
    if (ok && caseA) ok = cuda_ok(cudaMemsetAsync(w_gathered, 0, wg_b), "mixed40 wg clear");
    if (ok) { moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
              ok = cuda_ok(cudaGetLastError(), "mixed40 count"); }
    /* PULSAR_MOE_OVERLAP: expert-set overlap probe (read-only diagnostic).
     *
     * Three spec levers died on "wider spec batches don't amortize on our
     * bandwidth-bound decode" (BlockV #1, DTree #33, HybridLC #74). The ledger
     * blames verify running the prefill batch path, but that path is the
     * bandwidth-FAVOURABLE one -- it reads each dense weight once for all rows.
     * The likelier wall is MoE expert-set growth: M rows route to up to M*top_k
     * DISTINCT experts, and routed experts dominate model bytes.
     *
     * This measures it directly. `counts[e]` already holds rows-per-expert, so
     * SUM = pair_count = rows * top_k, and UNION = #{e : counts[e] > 0}. If
     * union is far below sum, drafts already share experts and the ceiling for
     * "expert-coherent drafting" is low; if union tracks sum, each extra verify
     * row really does drag in its own expert weights and there is headroom.
     *
     * Costs a sync + copy, so it is OFF unless asked for and never on a
     * measured path. Changes no computation. */
    static int overlap_env = -1;
    if (overlap_env < 0) overlap_env = getenv("PULSAR_MOE_OVERLAP") != NULL;
    if (ok && overlap_env && n_tokens > 0) {
        uint32_t *h_counts = (uint32_t *)malloc((size_t)n_total_expert * sizeof(uint32_t));
        if (h_counts) {
            if (cudaMemcpy(h_counts, counts,
                           (size_t)n_total_expert * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                uint32_t distinct = 0, maxc = 0;
                uint64_t total = 0;
                for (uint32_t e = 0; e < n_total_expert; e++) {
                    if (h_counts[e]) { distinct++; total += h_counts[e]; }
                    if (h_counts[e] > maxc) maxc = h_counts[e];
                }
                /* saturation = distinct / min(sum, n_total_expert): 1.0 means
                 * every routed slot pulled a different expert (no sharing). */
                const uint32_t ceil_distinct =
                        (uint64_t)pair_count < (uint64_t)n_total_expert
                        ? pair_count : n_total_expert;
                fprintf(stderr,
                        "pulsar: MOE-OVERLAP rows=%u top_k=%u sum=%u "
                        "distinct=%u of %u experts (saturation %.3f of the "
                        "%u reachable) max_rows_per_expert=%u\n",
                        n_tokens, n_expert, pair_count, distinct,
                        n_total_expert,
                        ceil_distinct ? (double)distinct / (double)ceil_distinct : 0.0,
                        ceil_distinct, maxc);
            }
            free(h_counts);
        }
    }
    if (ok) { moe_prefix_sorted_pairs_kernel<<<1, 256>>>(offsets, cursors, counts, n_total_expert);
              ok = cuda_ok(cudaGetLastError(), "mixed40 prefix"); }
    if (ok) { moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
              ok = cuda_ok(cudaGetLastError(), "mixed40 scatter"); }
    if (ok) { moe_padded_offsets_kernel<<<1, 1>>>(padded_off, counts, n_total_expert);
              ok = cuda_ok(cudaGetLastError(), "mixed40 padded offsets"); }
    if (!ok) return 0;

    if (caseA) {
        /* Phase 1: gate & up W4A8 -> swiglu -> mid (pair layout). */
        float *gate_g = gate_g_buf;
        float *up_g   = up_g_buf;
        float *mid_g  = mid_g_buf;
        if (use_grouped) {
            /* grouped: padded-gather x -> grouped GEMMs -> swiglu(padded) -> padded-scatter. */
            /* HANDOVER (L089 item a): grouped_proj now HAS a pre-encoded entry,
             * so the mixed-type layers stop being the one tier that gathers raw
             * f32.  On a hit the f32 x is never gathered or read; the E4M3 is
             * permuted straight into the CUTLASS layout, riding the row
             * permutation that had to happen anyway.  A miss keeps the f32
             * gather, under the guard. */
            const void *mq = NULL, *msf = NULL; int mkbp = 0;
            if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(x, n_tokens, expert_in_dim, &mq, &msf, &mkbp)) {
                mq = NULL; msf = NULL; mkbp = 0;
            }
            const int mixed_handover = (mq && msf && row_src_tok);
            if (!mixed_handover)
                PULSAR_MOE_F32_GUARD(x->ptr, n_tokens, expert_in_dim, "moe mixed40 gather (handover miss)");
            moe_padded_gather_kernel<<<pair_count, 256>>>(
                    mixed_handover ? NULL : x_gathered, w_gathered, padded_pair,
                    mixed_handover ? row_src_tok : NULL,
                    NULL /* pair_slot: this path scatters, no inverse map */,
                    (const float *)x->ptr, (const float *)weights->ptr,
                    sorted_pairs, offsets, padded_off, pair_count, n_total_expert, n_expert, expert_in_dim);
            ok = cuda_ok(cudaGetLastError(), "mixed40A gather");
            if (ok && pulsar_cutlass_grouped_proj(gate_g, x_gathered, (const uint8_t *)gate_w,
                    gate_expert_bytes, gate_row_bytes, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim,
                    counts, padded_off, (int)padded_upper, proj_scratch, proj_b, 0,
                    mixed_handover ? mq : NULL, mixed_handover ? msf : NULL,
                    mixed_handover ? mkbp : 0, mixed_handover ? row_src_tok : NULL) != 0) ok = 0;
            /* reuse_packed_a: same x_gathered, same scratch, same layout -- the
             * up leg consumes the gate leg's E4M3 encoding instead of packing
             * the identical values a second time. */
            if (ok && pulsar_cutlass_grouped_proj(up_g, x_gathered, (const uint8_t *)up_w,
                    gate_expert_bytes, gate_row_bytes, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim,
                    counts, padded_off, (int)padded_upper, proj_scratch, proj_b, 1,
                    mixed_handover ? mq : NULL, mixed_handover ? msf : NULL,
                    mixed_handover ? mkbp : 0, mixed_handover ? row_src_tok : NULL) != 0) ok = 0;
            if (ok) {
                uint64_t n = padded_upper * expert_mid_dim;
                moe_swiglu_gathered_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>(
                        mid_g, gate_g, up_g, w_gathered, clamp, expert_mid_dim, (uint32_t)padded_upper);
                ok = cuda_ok(cudaGetLastError(), "mixed40A swiglu");
            }
            if (ok) {
                moe_padded_scatter_kernel<<<(uint32_t)padded_upper, 256>>>(mid_flat, mid_g, padded_pair,
                        (uint32_t)padded_upper, expert_mid_dim);
                ok = cuda_ok(cudaGetLastError(), "mixed40A mid scatter");
            }
        } else {
            /* decode/verify (n<=4): lean W4A8 GEMV -> mid_flat (fused swiglu+routing weight), pair
             * layout, ONE launch over all slots -- no gather/scatter, no host sync, no TC underfill. */
            (void)gate_g; (void)up_g; (void)mid_g;
            /* Hand over the producing norm's E4M3 if it is still cached, exactly
             * as the grouped path above does. Without it the GEMV re-derives the
             * same encoding and dequantises it straight back to f32 -- the same
             * values, encoded twice. A miss just takes the f32 path. */
            const void *gq = NULL, *gsf = NULL;
            int gkbp = 0;
            if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(x, n_tokens, expert_in_dim, &gq, &gsf, &gkbp)) {
                gq = NULL; gsf = NULL; gkbp = 0;
            }
            /* The FIFTH f32 reader the census missed (found by the depth-4102
             * chunk-boundary failure): the GEMV fallback round-trips from raw
             * f32 when the handover misses.  Same rule as the other four. */
            if (!gq)
                PULSAR_MOE_F32_GUARD(x->ptr, n_tokens, expert_in_dim, "moe GEMV staging (handover miss)");
            if (pulsar_cutlass_gemv_gateup(mid_flat, (const float *)x->ptr, selected_ptr, (const float *)weights->ptr,
                    (const uint8_t *)gate_w, (const uint8_t *)up_w, gate_expert_bytes, gate_row_bytes,
                    clamp, (int)n_tokens, (int)n_expert, n_total_expert, (int)expert_in_dim, (int)expert_mid_dim,
                    gq, gsf, gkbp) != 0) ok = 0;
        }
        /* Phase 2: type-43 down against a type-40 gate/up (4 of the artifact's
         * layers).  MMQ takes mid_flat as f32 directly and does its own q8_1,
         * which is why no midq quantize remains here. */
        int mmq_down_done = 0;
#ifdef PULSAR_HAVE_MMQ
        if (ok) {
            mmq_down_done = routed_moe_try_mmq_down(down_flat, mid_flat, down_w, selected_ptr,
                                                    down_type, expert_mid_dim, out_dim,
                                                    n_total_expert, (uint64_t)pair_count);
            if (mmq_down_done) ok = cuda_ok(cudaGetLastError(), "mixed40A mmq down");
        }
#endif
        /* Fail closed: the dp4a chain that used to follow is gone with Q2_K and
         * type 16, so a declined MMQ has nothing left to fall through to. */
        if (ok && !mmq_down_done) {
            fprintf(stderr, "pulsar: mixed40 type-43 down but MMQ declined and no fallback exists\n");
            ok = 0;
        }
    } else {
        /* Case B. Phase 1: MMQ gate/up (fused swiglu) -> mid_flat.  The Q8_K
         * quantize of x that used to run here fed the dp4a kernels; MMQ takes x
         * as f32 and does its own q8_1, so it was pure waste once they went. */
        /* DELIBERATE ALIAS: reuse the gather buffer (K=mid_dim=cut_k).  Kept as an
         * explicit alias of x_gathered rather than a separate arena slice -- giving
         * it its own slice would silently double this scratch and change behaviour. */
        float *mid_g = x_gathered;
        float *out_g = out_g_buf;
        /* Type-43 gate/up against a type-40 down (3 of the artifact's layers).
         * `up` is pair-sized and was only read by the deleted qwarp32 branch, so
         * it serves as MMQ's raw gate buffer -- no arena change needed. */
        int mmq_gu_done = 0;
#ifdef PULSAR_HAVE_MMQ
        if (ok) {
            mmq_gu_done = routed_moe_try_mmq_gate_up(
                mid_flat, (float *)up->ptr, gate_w, up_w,
                (const float *)x->ptr, selected_ptr, (const float *)weights->ptr,
                gate_type, expert_in_dim, expert_mid_dim,
                n_total_expert, n_expert, n_tokens, clamp);
            if (mmq_gu_done) ok = cuda_ok(cudaGetLastError(), "mixed40B mmq gate/up");
        }
#endif
        /* Fail closed: the dp4a chain that used to follow is gone with type 16,
         * so a declined MMQ has nothing left to fall through to. */
        if (ok && !mmq_gu_done) {
            fprintf(stderr, "pulsar: mixed40 type-43 gate/up but MMQ declined and no fallback exists\n");
            ok = 0;
        }
        /* Phase 2: mid -> W4A8 down -> down_flat[pair]. */
        if (ok && use_grouped) {
            /* grouped: padded-gather mid -> grouped down GEMM -> padded-scatter. */
            moe_padded_gather_pairflat_kernel<<<pair_count, 256>>>(mid_g, padded_pair, mid_flat,
                    sorted_pairs, offsets, padded_off, pair_count, n_total_expert, expert_mid_dim);
            ok = cuda_ok(cudaGetLastError(), "mixed40B mid gather");
            /* No handover on the down leg: its input is mid_g, which the GPU
             * just computed and no producing norm ever encoded. */
            if (ok && pulsar_cutlass_grouped_proj(out_g, mid_g, (const uint8_t *)down_w,
                    down_expert_bytes, down_row_bytes, (int)n_total_expert, (int)expert_mid_dim, (int)out_dim,
                    counts, padded_off, (int)padded_upper, proj_scratch, proj_b, 0,
                    NULL, NULL, 0, NULL) != 0) ok = 0;
            if (ok) {
                moe_padded_scatter_kernel<<<(uint32_t)padded_upper, 256>>>(down_flat, out_g, padded_pair,
                        (uint32_t)padded_upper, out_dim);
                ok = cuda_ok(cudaGetLastError(), "mixed40B down scatter");
            }
        } else if (ok) {
            /* decode/verify (n<=4): lean W4A8 GEMV -> down_flat, pair layout, ONE launch over all
             * slots (routing weight already applied by the dp4a gate/up swiglu into mid_flat). */
            (void)mid_g; (void)out_g;
            if (pulsar_cutlass_gemv_down(down_flat, mid_flat, selected_ptr,
                    (const uint8_t *)down_w, down_expert_bytes, down_row_bytes,
                    (int)n_tokens, (int)n_expert, n_total_expert, (int)expert_mid_dim, (int)out_dim) != 0) ok = 0;
        }
    }
    if (!ok) return 0;

    uint64_t n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>((float *)out->ptr, down_flat, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "mixed40 sum");
}



#ifdef PULSAR_HAVE_MMQ
/* ---- MMQ (llama.cpp INT8 tensor-core) routed gate/up -----------------------
 * Plan 41b.  Measured on GB10, same box/model/prompt/metric: the shipped dp4a
 * the since-removed per-expert tile8 rowspan kernel cost 73.20 ms/layer at a
 * 4096-token prefill; Palaferri's stock mul_mat_q<IQ2_XXS> costs 29.99 ms/layer
 * (2.44x) and Entrpi's bespoke fused D2R 18.01 ms (4.07x).  This routes our
 * gate/up through the vendored stock MMQ.
 *
 * NOT BIT-EXACT vs the dp4a path by construction: the integer dots are exact
 * but the per-block float scale-fold ORDER differs.  Gated at shape time only
 * (never per token/layer, see [[no-hot-path-flags]]).
 *
 * MMQ emits RAW gate and up; our dp4a kernel fuses clamp + SwiGLU + the routing
 * weight into mid.  moe_mmq_swiglu_fold_kernel reproduces that tail exactly:
 *   off  = pair * mid_dim + row,  pair = tok * n_expert + slot
 *   so weights[tok * n_expert + slot] == weights[pair].
 */
/* The runtime aligned-IQ2 repack cache is GONE (2026-08-15).  It existed to
 * lazily repack raw IQ2_XXS (type 16) into the MMQ aligned layout under a
 * memory budget.  The loader now accepts only IQ2_XXS_MMQ (43) and
 * CUTLASS_MXFP4 (40) for routed experts (weights.cpp tensor_is_routed_expert
 * _type), and 43 is ALREADY aligned in the gguf -- so every tensor took the
 * prealigned branch and the cache never repacked anything.  With it go the
 * 256-entry table, the budget probe, and the raw-forever pinning. */

/* The ds4_cuda_q8_fold_take_q81 stub stood here.  The vendored adapter called it
 * from its q8-fold vec path to ask for pre-quantized canonical q8_1 codes;
 * pulsar never had such a registry, so the stub always answered "no fold" and
 * existed purely to satisfy the link.  That path went with the mmvq half in
 * ledger L066 step 2, so the stub has no caller and no reason to exist. */

/* ONE definition of the per-element math, shared by the scalar and vector
 * kernels so they cannot drift.  clamp semantics match the swiglu gate/up
 * path above. */
__device__ __forceinline__ static float moe_fold_elem(float g, float u, float wv, float clamp) {
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    return (g / (1.0f + expf(-g))) * u * wv;
}

__global__ static void moe_mmq_swiglu_fold_kernel(
        float *mid_out,
        const float *gate_raw,
        const float *up_raw,
        const float *weights,
        uint64_t total,
        uint32_t expert_mid_dim,
        float clamp) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mid_out[idx] = moe_fold_elem(gate_raw[idx], up_raw[idx],
                                 weights[idx / (uint64_t)expert_mid_dim], clamp);
}

/* Vector-4 twin (L128).  ncu on the decode path measured this kernel at 9.7%
 * SM throughput with 93% occupancy and a long-scoreboard stall ratio of
 * 111.9 -- fully resident and doing almost nothing but waiting on memory.
 * It is a pure streaming op (two loads, one store per output), so one
 * element per thread wastes three quarters of every transaction and pays a
 * 64-bit divide per element.  BIT-EXACT: identical per-element math, no
 * reduction, only the access granularity and thread mapping change. */
__global__ static void moe_mmq_swiglu_fold_v4_kernel(
        float4 *mid_out,
        const float4 *gate_raw,
        const float4 *up_raw,
        const float *weights,
        uint64_t quads,
        uint32_t expert_mid_dim_q,
        float clamp) {
    const uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= quads) return;
    const float4 g = gate_raw[q], u = up_raw[q];
    const float wv = weights[q / (uint64_t)expert_mid_dim_q];
    float4 o;
    o.x = moe_fold_elem(g.x, u.x, wv, clamp);
    o.y = moe_fold_elem(g.y, u.y, wv, clamp);
    o.z = moe_fold_elem(g.z, u.z, wv, clamp);
    o.w = moe_fold_elem(g.w, u.w, wv, clamp);
    mid_out[q] = o;
}

/* Returns 1 when MMQ produced `mid`, 0 to fall through to the dp4a path. */
static int routed_moe_try_mmq_gate_up(
        float *mid_out,
        float *gate_scratch,
        const char *gate_w,
        const char *up_w,
        const float *x_f32,
        const int32_t *selected_ptr,
        const float *weights_ptr,
        uint32_t gate_type,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t n_tokens,
        float clamp) {
    /* 16 = raw IQ2_XXS block stream (aligned lazily via the cache below).
     * 43 = IQ2_XXS_MMQ, already stored aligned IN THE GGUF -- no cache, no
     *      repack, no budget, and every tensor gets the fast path instead of
     *      whichever ~58 won a 22.9 GiB cache.
     * 42 (our Phase-0 SoA) is a DIFFERENT layout and is not MMQ-consumable. */
    if (gate_type != 43u) return 0;   /* PULSAR_TENSOR_IQ2_XXS_MMQ, the only one */
    /* One-shot: the adapter allocates its persistent q8_1 scratch here.  Without
     * it every call falls back to per-call async allocation, which cost 6x
     * (78 vs 477 t/s) the first time this was wired.  One-shot, not per-token. */
    static int mmq_ready = -1;
    if (mmq_ready < 0) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        mmq_ready = (ds4_mmq_init(dev) == 0) ? 1 : 0;
        fprintf(stderr, "pulsar: MMQ routed gate/up %s\n", mmq_ready ? "ready" : "unavailable");
    }
    if (!mmq_ready) return 0;
    /* should_use takes a GGML type; both 16 and our 43 are IQ2_XXS data, and 43
     * is not a ggml type at all -- passing it through returned false and
     * silently declined every pre-aligned layer. */
    if (!ds4_mmq_should_use(16, (int64_t)n_tokens, (int64_t)n_total_expert)) return 0;

    /* Why the artifact stores IQ2 pre-aligned, recorded because the code that
     * proved it is gone: block_iq2_xxs is 66 B with qs[] at offset 2, so a raw
     * stream is 2-byte aligned and nvcc emits LDG.E.U16.  MMQ is heavily
     * weight-read-bound, so at the production shape raw measured 44.70 vs
     * aligned 18.65 ms/pair-call = 2.40x, against a one-time repack of
     * 4.81 ms/tensor.  That settled it: the gguf now ships type 43 aligned, the
     * runtime repack cache is deleted, and tests/mmq_align_bench.cu with it --
     * a closed question does not need a live A/B keeping the losing arm
     * compiled. */
    /* Type 43 is stored aligned IN THE GGUF, so the weight pointer IS the SoA
     * entry.  There is no raw arm and no repack. */
    float *gate_raw = gate_scratch;   /* both are n_tokens*n_expert*mid f32 */
    float *up_raw = mid_out;          /* folded in place below */
    /* x_f32 is ffn_norm, whose fused norm already emitted its E4M3 + ue8m0 into
     * the activation cache. Hand that over and the input staging gathers the
     * codes instead of encoding them again from f32. A miss just means the
     * staging encodes, which is what it did before -- same bytes either way. */
    const void *act_q = NULL, *act_sf = NULL;
    int act_kbp = 0;
    if (!pulsar_gpu_mxfp8_act_cache_get_e4m3_ptr(x_f32, n_tokens, expert_in_dim,
                                                 &act_q, &act_sf, &act_kbp)) {
        act_q = NULL; act_sf = NULL; act_kbp = 0;
    }
    if (!act_q)
        PULSAR_MOE_F32_GUARD(x_f32, n_tokens, expert_in_dim, "moe MMQ staging (handover miss)");
    const int rc = ds4_mmq_iq2_xxs_moe_pair_soa(
        gate_w, up_w, x_f32, selected_ptr, gate_raw, up_raw,
        (int)expert_mid_dim, (int)expert_in_dim,
        (int)n_tokens, (int)n_total_expert,
        (int)n_expert, cudaStreamPerThread,
        act_q, act_sf, act_kbp);
    if (rc != 0) return 0;
    const uint64_t total = (uint64_t)n_tokens * n_expert * expert_mid_dim;
    const uint32_t threads = 256u;
    /* L128: four elements per thread when the shape and alignment allow --
     * see the v4 kernel's note.  mid_out and gate_scratch are whole
     * allocations here (256 B aligned), but the check is cheap and keeps the
     * fast path honest if a caller ever hands in an offset view. */
    const bool v4 = (expert_mid_dim % 4u) == 0u && (total % 4u) == 0u &&
                    ((uintptr_t)mid_out % 16u) == 0u &&
                    ((uintptr_t)gate_raw % 16u) == 0u &&
                    ((uintptr_t)up_raw % 16u) == 0u;
    if (v4) {
        const uint64_t quads = total / 4u;
        const uint64_t blocks = (quads + threads - 1u) / threads;
        moe_mmq_swiglu_fold_v4_kernel<<<(unsigned)blocks, threads>>>(
            (float4 *)mid_out, (const float4 *)gate_raw, (const float4 *)up_raw,
            weights_ptr, quads, expert_mid_dim / 4u, clamp);
    } else {
        const uint64_t blocks = (total + threads - 1u) / threads;
        moe_mmq_swiglu_fold_kernel<<<(unsigned)blocks, threads>>>(
            mid_out, gate_raw, up_raw, weights_ptr, total, expert_mid_dim, clamp);
    }
    return 1;
}
/* Routed DOWN through MMQ.  Upstream's down rode inside a Q2_K-specific fused
 * kernel, useless for v5mx whose down is IQ2 -- and that whole Q2_K expert
 * family has since been deleted, because the loader accepts only types 43 and
 * 40 for routed experts (weights.cpp: tensor_is_routed_expert_type), so no
 * Q2_K expert tensor can reach the engine at all.
 *
 * But down does not need a fused kernel here: `mid` is ALREADY materialized f32
 * per (token,slot), and `selected` is ALREADY the flat per-pair expert id
 * (pair = tok*n_expert + slot).  So the generic MoE entry fits exactly if each
 * pair is treated as its own row using exactly one expert:
 *     n_tokens'      = n_tokens * n_expert     (one row per pair)
 *     n_expert_used' = 1
 * Output is per-pair, which is what the existing fixed-order moe_sum_kernel
 * already consumes -- so the bit-exact reduction is untouched.  This also skips
 * the mid -> midq q8_K quantize entirely (MMQ does its own q8_1).
 */
static int routed_moe_try_mmq_down(
        float *down_out,
        const float *mid_f32,
        const char *down_w,
        const int32_t *selected_ptr,
        uint32_t down_type,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_total_expert,
        uint64_t pairs) {
    if (down_type != 43u) return 0;   /* PULSAR_TENSOR_IQ2_XXS_MMQ, the only one */
    if (pairs > (uint64_t)INT32_MAX) return 0;
    if (!ds4_mmq_should_use(16, (int64_t)pairs, (int64_t)n_total_expert)) return 0;
    /* Pre-aligned in the gguf: take the SoA entry directly.  This is the case the
     * runtime cache could never afford -- letting down compete for a 22.9 GiB
     * budget just starved late layers of gate/up (measured 566.46 vs 590.77).
     * With the layout on disk there is no budget to compete for. */
    return ds4_mmq_iq2_xxs_moe_soa(down_w, mid_f32, selected_ptr, down_out,
                                   (int)out_dim, (int)expert_mid_dim,
                                   (int)pairs, (int)n_total_expert, 1,
                                   cudaStreamPerThread) == 0;
}
#endif /* PULSAR_HAVE_MMQ */

static int routed_moe_launch(
        pulsar_gpu_tensor *out,
        pulsar_gpu_tensor *up,
        pulsar_gpu_tensor *mid,
        pulsar_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const pulsar_gpu_tensor *selected,
        const pulsar_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const pulsar_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens) {
    if (!out || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    /* This function is now IQ2_XXS_MMQ (43) on BOTH sides, and nothing else.
     *
     * The three dispatchers above it split the space first: 40/40 goes to
     * routed_moe_launch_cutlass, exactly-one-40 to routed_moe_launch_mixed40,
     * and what is left arrives here.  That used to mean any pairing of
     * {IQ2_XXS 16, IQ2_XXS_SOA 42, Q2_K 10, FP4_E2M1 39, 43}; a scan of the
     * shipped artifact found only 40 and 43 among its expert tensors, none of
     * the other four is ever synthesised at load, and their kernels have been
     * deleted.  So the door is a simple equality and everything downstream is
     * MMQ-only -- no layout traits, no d-plane offsets, no dp4a fallback.
     *
     * Anything else is refused rather than reinterpreted: a dispatch chain that
     * accepts an unknown tag does not fail, it reads the bytes as some other
     * format and returns plausible garbage. */
    if (gate_type != 43u || down_type != 43u) return 0;
#ifndef PULSAR_HAVE_MMQ
    return 0;   /* no MMQ build -> type 43 is unreadable */
#endif
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    /* No Q8_K scratch is carved out of `down`/`gate` any more: it existed only
     * to feed the dp4a readers, and MMQ takes x and mid as f32 and does its own
     * q8_1.  The scratch-size guard that used to wrap this body went with it --
     * it could only ever have failed into an f32 fallback that no longer
     * exists.  `down` is still the per-pair output buffer, sized by the caller
     * checks above. */
    {
        /* Read once (no-hot-path-flags): this runs on every MoE launch, i.e.
         * every layer of every decode token, for a switch fixed at start.
         * Same cached-static idiom the rest of this file already uses. */
        static int profile_moe_env = -1;
        if (profile_moe_env < 0) profile_moe_env = getenv("PULSAR_CUDA_MOE_PROFILE") != NULL;
        const uint32_t profile_moe = (uint32_t)profile_moe_env;
        /* Six boundaries, not seven. The seventh bracketed the sorted-pair /
         * expert-tile build, which was deleted when nothing was found to read
         * it -- but its cudaEventRecord went with the kernels while the
         * teardown kept calling cudaEventElapsedTime on it. That returns
         * cudaErrorInvalidResourceHandle, and because the result was discarded
         * with (void) it became the sticky last error, which the NEXT launch
         * check reported as "CUDA swiglu launch failed: invalid resource
         * handle". So turning the profiler on killed the run one layer after
         * the first profiled MoE -- a diagnostic that destroyed the thing it
         * was measuring, and printed xq=0.000 sort=0.000 (the two intervals
         * touching the dead event) as its only warning. */
        cudaEvent_t prof_ev[5] = {NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 5u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        /* One release configuration (the tuned production values from the old
         * PULSAR_CUDA_MOE_* env matrix): expert-major sorted pair tiles, 16-row
         * down tiles + row-span kernels for big prefill batches, LUT gate for
         * decode, direct down-sum for the 6-expert decode case.  (This said
         * "atomic down accumulation" until this commit; there is no atomicAdd
         * on any down path -- every output element is written by exactly one
         * thread, per-pair stores plus a fixed-order moe_sum_kernel, which is
         * precisely why these stages are bit-exact and re-tileable.) */
        const uint32_t pair_count = n_tokens * n_expert;
        /* EVERY batch size takes the batch arm, n_tokens==1 included.  This was
         * `n_tokens > 1u`, and that comparison was the last size-thresholded
         * activation-format split in the engine: n>=2 reached the D2R arm with
         * E4M3 operands, while n==1 went to the _vec arm, which stages
         * `block_q8_1` and runs integer dp4a against the IQ2 codebook.  q8_1 is
         * not a preference there -- dp4a is an integer instruction and E4M3
         * cannot feed it -- so the split was a KERNEL difference wearing a
         * format's clothes, and the exposure was plain decode and the drafter's
         * own forwards (speculative verify batches are n>1 and already ran
         * E4M3).
         *
         * ds4_mmq.cu:492 already refuses exactly this on the prefill down path
         * by name -- "one activation format, every batch size" -- having been
         * written after a size-thresholded split hid a missing conversion for
         * two days.  This was the surviving instance of the thing that comment
         * forbids, and Tyler's call (2026-08-17) is that the format wins:
         * "We want one activation format everywhere."
         *
         * ⚠ IT COSTS 9.4% OF DECODE, MEASURED, AND THAT IS ACCEPTED, NOT
         * UNKNOWN: 17.47 -> 15.82 t/s, routed_moe 0.473 -> 0.584 ms/layer.
         * Cause is tile occupancy, not arithmetic -- the batch path's kNTile is
         * 64 token-expert pairs wide and decode presents 6, so the N dimension
         * runs at 9.4% occupancy.  Recovering it is a separate perf row (L056
         * option B), NOT a precondition for this line; the fidelity decision
         * ships first and the speed follows it.  Prefill is a clean control at
         * 957.08 -> 957.00 t/s: only the n==1 path moved. */
        /* The sorted-pair and expert-tile structures USED to be built here: a
         * scratch allocation plus six kernel launches (count, prefix, scatter,
         * two tile builders, and an 8/16-row pair) on every layer of every
         * prefill chunk.  Nothing read them.  MMQ takes selected_ptr directly,
         * and the only surviving use of the result was `sorted_pairs != NULL`
         * as a stand-in for `n_tokens > 1`, a distinction this path no longer
         * makes at all.
         * The file said so itself: "not read by MMQ itself; they survive as
         * the shape signal the batch arms are gated on".  A shape signal does
         * not need six kernels to compute.
         * (The CUTLASS dispatchers above build their own and DO read them.) */
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        /* MMQ is now the ONLY reader here: this function is type-43-on-both-
         * sides, and the dp4a chains that used to follow each arm are gone with
         * the types that fed them.  So "MMQ declined" is terminal -- there is
         * nothing to fall through to, which is exactly what the two fail-closed
         * arms below say.  What stays enforced at load is the binder's rule that
         * gate and up share a type: the fused gate+up kernels read ONE layout. */
        int mmq_done = 0;
#ifdef PULSAR_HAVE_MMQ
        if (ok) {
            mmq_done = routed_moe_try_mmq_gate_up(
                (float *)mid->ptr, (float *)up->ptr,
                gate_w, up_w, (const float *)x->ptr, selected_ptr,
                (const float *)weights->ptr, gate_type,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert, n_tokens, clamp);
            if (mmq_done) ok = cuda_ok(cudaGetLastError(), "routed_moe mmq gate/up");
        }
        /* Fail closed, and note WHICH failure this now is.  A _vec fallback used
         * to sit here for n_tokens==1.  It cannot come back: it stages q8_1, so
         * reinstating it as a decline-handler would mean a declined batch launch
         * silently changes the activation format instead of failing -- the split
         * removed above, restored by the back door and visible only as a quality
         * regression.  One format, or an error. */
        if (ok && !mmq_done) {
            fprintf(stderr, "pulsar: type-43 gate/up but MMQ declined and no fallback exists\n");
            ok = 0;
        }
#else
        (void)mmq_done;
        return 0;   /* no MMQ build -> type 43 is unreadable; rejected at the door */
#endif
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        int mmq_down_done = 0;
#ifdef PULSAR_HAVE_MMQ
        /* Shape-time, once per launch.  MMQ consumes mid as f32 and does its own
         * q8_1, which is why there is no mid -> midq q8_K quantize left here. */
        if (ok) {
            mmq_down_done = routed_moe_try_mmq_down((float *)down->ptr,
                                                    (const float *)mid->ptr,
                                                    down_w, selected_ptr,
                                                    down_type, expert_mid_dim, out_dim,
                                                    n_total_expert,
                                                    (uint64_t)n_tokens * n_expert);
            if (mmq_down_done) ok = cuda_ok(cudaGetLastError(), "routed_moe mmq down");
        }
        /* Same fail-closed rule as gate/up: the down _vec arm is gone with the
         * threshold that was its only reader.  The reduction below is unaffected
         * -- it was already owed unconditionally, because every surviving arm
         * writes per-pair. */
        if (ok && !mmq_down_done) {
            fprintf(stderr, "pulsar: type-43 down but MMQ declined and no fallback exists\n");
            ok = 0;
        }
#endif
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        /* The MMQ arms write PER-PAIR results, so the fixed-order reduction is
         * always owed; the arms above are fail-closed, so reaching here with ok
         * set means one of them ran. */
        if (ok) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[4]) {
            (void)cudaEventRecord(prof_ev[4], 0);
            if (cudaEventSynchronize(prof_ev[4]) == cudaSuccess) {
                float ms_pre = 0.0f, ms_gate = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                /* Checked, not discarded: swallowing these with (void) is what
                 * let a dead event poison the next launch's error check. */
                cudaError_t te = cudaSuccess;
                te = cudaEventElapsedTime(&ms_pre,   prof_ev[0], prof_ev[1]) != cudaSuccess ? cudaGetLastError() : te;
                te = cudaEventElapsedTime(&ms_gate,  prof_ev[1], prof_ev[2]) != cudaSuccess ? cudaGetLastError() : te;
                te = cudaEventElapsedTime(&ms_down,  prof_ev[2], prof_ev[3]) != cudaSuccess ? cudaGetLastError() : te;
                te = cudaEventElapsedTime(&ms_sum,   prof_ev[3], prof_ev[4]) != cudaSuccess ? cudaGetLastError() : te;
                te = cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[4]) != cudaSuccess ? cudaGetLastError() : te;
                if (te != cudaSuccess) {
                    fprintf(stderr, "pulsar: MoE profile timing unavailable (%s) -- "
                                    "numbers below are incomplete\n", cudaGetErrorString(te));
                }
                fprintf(stderr,
                        "pulsar: CUDA MoE profile tokens=%u pairs=%u pre=%.3f gateup=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_pre, ms_gate, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 5u; i++) (void)cudaEventDestroy(prof_ev[i]);
            /* Clear anything the teardown itself raised, so a diagnostic can
             * never be mistaken for a kernel failure by the caller's check. */
            (void)cudaGetLastError();
        }
        return ok;
    }

    return ok;
}



int pulsar_gpu_routed_moe_one_tensor(pulsar_gpu_tensor *out, pulsar_gpu_tensor *up, pulsar_gpu_tensor *mid, pulsar_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const pulsar_gpu_tensor *selected, const pulsar_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const pulsar_gpu_tensor *x, uint32_t layer_index) {
    /* Nothing below this line consults esz: every scratch/output cast in the
     * MoE lane is (float *) over a tensor the graph allocates f32.  That was
     * TRUE-BY-ACCIDENT rather than checked (types sweep 2026-08-22): if any
     * MoE tensor is ever narrowed, ten casts become defect nine at once,
     * silently.  So the whole lane's f32 assumption is enforced ONCE, here at
     * its two entry points. */
    if (pulsar_tensor_esz(out) != sizeof(float) || pulsar_tensor_esz(up) != sizeof(float) ||
        pulsar_tensor_esz(mid) != sizeof(float) || pulsar_tensor_esz(down) != sizeof(float) ||
        pulsar_tensor_esz(weights) != sizeof(float) || pulsar_tensor_esz(x) != sizeof(float)) {
        fprintf(stderr, "pulsar: routed MoE lane is f32-only; a narrowed tensor reached it\n");
        return 0;
    }
    if (gate_type == 40u && down_type == 40u) {
        /* Decode (n=1) takes the same direct fp4 GEMV as small verify batches:
         * 4 launches with no host round-trip, vs the grouped path's BLOCKING
         * per-layer offsets readback -- required for CUDA graph capture of the
         * decode tape, and computes the exact same function (see the batch
         * path's comment; bit-exact oracle in temp/fp4gemv_test.cu covers
         * n_tokens>=1). */
        if (            mid && mid->ptr && down && down->ptr && out && out->ptr &&
            selected && selected->ptr && weights && weights->ptr && x && x->ptr &&
            mid->bytes >= (uint64_t)n_expert * expert_mid_dim * sizeof(float) &&
            down->bytes >= (uint64_t)n_expert * out_dim * sizeof(float) &&
            out->bytes >= (uint64_t)out_dim * sizeof(float) &&
            selected->bytes >= (uint64_t)n_expert * sizeof(int32_t) &&
            weights->bytes >= (uint64_t)n_expert * sizeof(float) &&
            x->bytes >= (uint64_t)expert_in_dim * sizeof(float)) {
            const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
            const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
            const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total, "moe_fp4_gemv_gate");
            const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total, "moe_fp4_gemv_up");
            const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total, "moe_fp4_gemv_down");
            /* Handover first (L089): when the producing norm already emitted
             * this x as E4M3, gemv_small reads THOSE bytes and never
             * dereferences the f32 -- which REMOVES the sixth reader instead of
             * guarding it, and is what makes lifting the n<=8 ffn store-skip
             * possible.  The guard now covers only the miss path, which packs.
             * Same shape as the gemv_gateup handover above. */
            const void *hq = NULL, *hsf = NULL;
            int hkbp = 0;
            if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(x, 1u, expert_in_dim, &hq, &hsf, &hkbp)) {
                hq = NULL; hsf = NULL; hkbp = 0;
            }
            if (!hq)
                PULSAR_MOE_F32_GUARD(x->ptr, 1u, expert_in_dim, "moe small-batch FFN GEMV (decode, handover miss)");
            if (gate_w && up_w && down_w &&
                pulsar_cutlass_expert_ffn_gemv_small(
                        (float *)down->ptr, (float *)mid->ptr, (const float *)x->ptr,
                        (const int32_t *)selected->ptr, (const float *)weights->ptr,
                        (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        clamp, 1, (int)n_expert, n_total_expert,
                        (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                        hq, hsf, hkbp) == 0) {
                moe_sum_kernel<<<(uint32_t)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, 1u);
                if (cuda_ok(cudaGetLastError(), "moe fp4 gemv sum")) return 1;
            }
            /* any failure: fall through to the grouped CUTLASS path */
        }
        /* type-40 decode bypassed the fp4 GEMV fast path -> the per-expert
         * CUTLASS loop.  Announce once, unconditionally, so the slow decode
         * tier is not silent. */
        {
            static int gemv_dec_logged = 0;
            if (!gemv_dec_logged) { gemv_dec_logged = 1;
                fprintf(stderr,
                        "pulsar: WARNING MoE fp4 GEMV decode path not taken -> per-expert loop (slower decode)\n");
            }
        }
        return routed_moe_launch_cutlass(out, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, gate_row_bytes,
                                         down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x, 1);
    }
    if ((gate_type == 40u) != (down_type == 40u)) {
        /* MIXED type-40 + type-43: per-projection dispatch. Fail-closed --
         * routed_moe_launch is 43/43 only and cannot read the type-40 swizzle,
         * so a mixed layer must never fall through to it. */
        return routed_moe_launch_mixed40(out, up, mid, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset, gate_type, down_type,
                                         gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x, 1);
    }
    return routed_moe_launch(out, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1);
}


/* plan-34 inc 4: row-offset sub-view of a per-token MoE tensor for the two-pass
 * split (byte offset into a row-major [n_tokens x stride] buffer). Reads ptr/bytes
 * only; owner=0 (never freed). NULL/empty tensors pass through as empty. */
static inline pulsar_gpu_tensor moe_subrow(const pulsar_gpu_tensor *t, uint64_t off_bytes) {
    pulsar_gpu_tensor s; s.ptr = NULL; s.bytes = 0; s.owner = 0;
    if (t && t->ptr) {
        s.ptr = (char *)t->ptr + off_bytes;
        s.bytes = t->bytes > off_bytes ? t->bytes - off_bytes : 0;
    }
    return s;
}

static int routed_moe_batch_impl(pulsar_gpu_tensor *out, pulsar_gpu_tensor *up, pulsar_gpu_tensor *mid, pulsar_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const pulsar_gpu_tensor *selected, const pulsar_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const pulsar_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    /* plan-34 inc 4 — MoE TWO-PASS split of a fused mixed step. Row layout is
     * [decode rows 0..n_dec) then one K-row prefill run [n_dec..n_tokens). The MoE
     * is strictly per-row (selected/weights/x/out addressed by token), so a decode
     * row's expert output is invariant to co-scheduled rows ONLY if it takes the
     * M-independent per-token path — but the grouped GEMM's per-expert group sizes
     * depend on the whole batch. So: run the PER-TOKEN path over the decode prefix
     * (n_tokens=n_dec<=PULSAR_GPU_MNEUTRAL_ROWS_MAX keeps the GEMV/non-grouped
     * dispatch) and the GROUPED path over the prefill suffix, offsetting every
     * per-token buffer by n_dec rows.
     *   - Pass 1 keeps the flag = n_dec (nonzero) so the capped dispatch picks the
     *     per-token path for any armed n_dec; n_tokens==n_dec => it does NOT re-split.
     *   - Pass 2 clears the flag so it (a) does not re-split and (b) is BYTE-
     *     IDENTICAL to an inc-3 pure-prefill MoE of the same K rows (grouped path,
     *     which reads no flag). The flag is restored before returning so the rest of
     *     the layer/step still splits its dense GEMMs. */
    {
        const uint32_t n_dec = (uint32_t)pulsar_gpu_matmul_batch_mneutral();
        if (n_dec > 0 && n_dec < n_tokens && x && selected && weights && out) {
            const uint64_t f = sizeof(float), i4 = sizeof(int32_t);
            pulsar_gpu_tensor x_s    = moe_subrow(x,        (uint64_t)n_dec * expert_in_dim * f);
            pulsar_gpu_tensor sel_s  = moe_subrow(selected, (uint64_t)n_dec * n_expert * i4);
            pulsar_gpu_tensor w_s    = moe_subrow(weights,  (uint64_t)n_dec * n_expert * f);
            pulsar_gpu_tensor out_s  = moe_subrow(out,      (uint64_t)n_dec * out_dim * f);
            pulsar_gpu_tensor up_s   = moe_subrow(up,       (uint64_t)n_dec * n_expert * expert_mid_dim * f);
            pulsar_gpu_tensor mid_s  = moe_subrow(mid,      (uint64_t)n_dec * n_expert * expert_mid_dim * f);
            pulsar_gpu_tensor down_s = moe_subrow(down,     (uint64_t)n_dec * n_expert * out_dim * f);
            const int r1 = routed_moe_batch_impl(out, up, mid, down, model_map, model_size,
                    gate_offset, up_offset, down_offset, gate_type, down_type,
                    gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                    expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                    n_total_expert, n_expert, clamp, x, layer_index, n_dec);
            pulsar_gpu_matmul_set_batch_mneutral(0);
            const int r2 = routed_moe_batch_impl(&out_s, &up_s, &mid_s, &down_s,
                    model_map, model_size, gate_offset, up_offset, down_offset, gate_type, down_type,
                    gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                    expert_in_dim, expert_mid_dim, out_dim, &sel_s, &w_s,
                    n_total_expert, n_expert, clamp, &x_s, layer_index, n_tokens - n_dec);
            pulsar_gpu_matmul_set_batch_mneutral((int)n_dec);
            return (r1 && r2) ? 1 : 0;
        }
    }
    {
        static int entry_log = -1;
        if (entry_log < 0) entry_log = getenv("PULSAR_MOE_PATH_LOG") != NULL ? 200 : 0;
        if (entry_log > 0) { entry_log--;
            fprintf(stderr, "pulsar: moe_batch ENTRY layer=%u gate_type=%u down_type=%u n_tok=%u\n",
                    layer_index, gate_type, down_type, n_tokens); }
    }
    if (gate_type == 40u && down_type == 40u) {
        /* Small batches (spec-decode verify, n_tokens 2..4): direct fp4 GEMV over the
         * packed expert weights -- 4 launches per layer (act-roundtrip + gate/up+swiglu
         * + act-roundtrip + down) vs the grouped path's BLOCKING per-layer offsets
         * readback + ~5 launches per active expert. Computes the exact same function
         * as the grouped GEMMs (weights read in the verified row-major CUTLASS B
         * layout; activations round-tripped through the same fp4 quantizer) -- proven
         * bit-exact vs a quant-exact CPU oracle (temp/fp4gemv_test.cu) and clean-text
         * on-model. Measured verify(3) step 118.2 -> 116.5 ms; the bigger win is
         * removing the per-rich-layer host sync from the verify path (CUDA-graph
         * prerequisite). n_tokens==1 (decode) keeps the grouped path unchanged.
         */
        static int path_log = -1;
        if (path_log < 0) path_log = getenv("PULSAR_MOE_PATH_LOG") != NULL ? 400 : 0;
        if (path_log > 0) {
            path_log--;
            fprintf(stderr,
                    "pulsar: moe40 layer=%u n_tok=%u n_total=%u stride=%llu split=%llu mid=%d down=%d midb=%llu need=%llu\n",
                    layer_index, n_tokens, n_total_expert,
                    (unsigned long long)gate_expert_bytes, (unsigned long long)gate_row_bytes,
                    mid && mid->ptr ? 1 : 0, down && down->ptr ? 1 : 0,
                    (unsigned long long)(mid ? mid->bytes : 0),
                    (unsigned long long)((uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float)));
        }
        /* inc 2: raise the M-independent GEMV cap to PULSAR_MSEQ_MAX (8) for a batched
         * step so 5..8 keep the per-token path instead of the grouped GEMM. */
        /* 2026-07-21: the un-armed default was 4, so spec-verify at --dspark-draft 5
         * (w=6) fell off the per-token GEMV onto the grouped expert GEMM with its
         * blocking per-layer offsets readback. Both arms are 8 now; the 5..8 path
         * is the same kernel already used under mneutral.
         * VERIFIED bit-exact in isolation vs the cap-4 build at verify widths 6
         * and 8 (byte-identical logits + token stream, 3 fresh loads per arm), and
         * width 4 is untouched. Median verify: w=6 258.0 -> 180.7 ms (-30.0%),
         * w=8 290.4 -> 213.9 ms (-26.4%); layers_encode 213 -> 130 ms. The three
         * sibling caps in pulsar_cuda_matmul.cu were tried alongside this and are NOT
         * bit-exact -- they stay at 4; do not re-couple them to this one.
         * L117 2026-08-26: floor lowered 2 -> 1. The >=2 floor predated any
         * M=1 caller (classic decode owns single-token MoE elsewhere), but the
         * batched lanes DO reach here with one row — a lane-2 group that
         * shrinks to one slot, a 1-bank quenched spec round, and the row-cost
         * probe — and fell onto the grouped-CUTLASS prefill machinery
         * (pack + 3 grouped GEMMs + gather + f32 swiglu per type-40 layer,
         * ~25-30 ms/sweep of fixed cost for one token; nsys diff in
         * pulsar-notes rows/L117.md). GEMV vs grouped was byte-identical at
         * widths 6/8 (above); M=1 verified the same way (probe logits hash,
         * see the row). */
        const uint32_t moe_gemv_cap = 8u;
        if (n_tokens >= 1u && n_tokens <= moe_gemv_cap &&
            mid && mid->ptr && down && down->ptr && out && out->ptr &&
            selected && selected->ptr && weights && weights->ptr && x && x->ptr &&
            mid->bytes >= (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) &&
            down->bytes >= (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) &&
            out->bytes >= (uint64_t)n_tokens * out_dim * sizeof(float) &&
            selected->bytes >= (uint64_t)n_tokens * n_expert * sizeof(int32_t) &&
            weights->bytes >= (uint64_t)n_tokens * n_expert * sizeof(float) &&
            x->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(float)) {
            const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
            const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
            const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total, "moe_fp4_gemv_gate");
            const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total, "moe_fp4_gemv_up");
            const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total, "moe_fp4_gemv_down");
            /* Handover first (L089): when the producing norm already emitted
             * this x as E4M3, gemv_small reads THOSE bytes and never
             * dereferences the f32 -- which REMOVES the sixth reader instead of
             * guarding it, and is what makes lifting the n<=8 ffn store-skip
             * possible.  The guard now covers only the miss path, which packs.
             * Same shape as the gemv_gateup handover above. */
            const void *hq = NULL, *hsf = NULL;
            int hkbp = 0;
            if (!pulsar_gpu_mxfp8_act_cache_get_e4m3(x, n_tokens, expert_in_dim, &hq, &hsf, &hkbp)) {
                hq = NULL; hsf = NULL; hkbp = 0;
            }
            if (!hq)
                PULSAR_MOE_F32_GUARD(x->ptr, n_tokens, expert_in_dim, "moe small-batch FFN GEMV (handover miss)");
            if (gate_w && up_w && down_w &&
                pulsar_cutlass_expert_ffn_gemv_small(
                        (float *)down->ptr, (float *)mid->ptr, (const float *)x->ptr,
                        (const int32_t *)selected->ptr, (const float *)weights->ptr,
                        (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        clamp, (int)n_tokens, (int)n_expert, n_total_expert,
                        (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                        hq, hsf, hkbp) == 0) {
                const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
                moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
                        (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
                if (cuda_ok(cudaGetLastError(), "moe fp4 gemv sum")) {
                    if (path_log > 0) fprintf(stderr, "pulsar: moe40 layer=%u -> gemv\n", layer_index);
                    return 1;
                }
            }
            /* any failure: fall through to the grouped CUTLASS path */
        }
        if (path_log > 0) fprintf(stderr, "pulsar: moe40 layer=%u -> grouped\n", layer_index);
        /* A small verify batch (n_tokens 2..4) that was ELIGIBLE for the fp4
         * GEMV fast path but fell through is a genuine (if minor) silent
         * fallback -- announce once, unconditionally.  n_tokens>4 (prefill)
         * routing to the grouped dispatch is the intended fast path, not a
         * fallback, so it is deliberately NOT flagged here. */
        {
            static int gemv_batch_logged = 0;
            if (n_tokens >= 1u && n_tokens <= 4u && !gemv_batch_logged) {
                gemv_batch_logged = 1;
                fprintf(stderr,
                        "pulsar: WARNING MoE fp4 GEMV verify(%u) path not taken -> grouped dispatch\n",
                        n_tokens);
            }
        }
        return routed_moe_launch_cutlass_dispatch(out, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, gate_row_bytes,
                                         down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x,
                                         n_tokens);
    }
    if ((gate_type == 40u) != (down_type == 40u)) {
        /* MIXED type-40 + type-43: per-projection dispatch. Fail-closed. */
        return routed_moe_launch_mixed40(out, up, mid, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset, gate_type, down_type,
                                         gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
    }
    return routed_moe_launch(out, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens);
}

/* ---- Per-format MoE prefill timing (PULSAR_MOE_TIME=1). Buckets each batch MoE call's GPU time
 * by (gate_type,down_type) via a CUDA-event pair around the impl, and dumps a table at exit.
 * The event sync perturbs host wall-time but the measured per-call GPU interval is accurate;
 * this is a diagnostic-only path (one getenv read, no per-token flag branching in production). */
struct moe_time_bucket { uint32_t gate_type, down_type; double ms; uint64_t calls; };
static moe_time_bucket g_moe_time[64];
static int g_moe_time_n = 0;
static void moe_time_dump(void){
    if (g_moe_time_n == 0) return;
    double tot = 0; for (int i = 0; i < g_moe_time_n; i++) tot += g_moe_time[i].ms;
    fprintf(stderr, "\n=== PULSAR_MOE_TIME per-format MoE batch GPU time (total %.2f ms over all calls) ===\n", tot);
    for (int i = 0; i < g_moe_time_n; i++) {
        moe_time_bucket *b = &g_moe_time[i];
        fprintf(stderr, "  gate_type=%2u down_type=%2u : %9.2f ms  (%5.1f%%)  calls=%llu  avg=%.3f ms\n",
                b->gate_type, b->down_type, b->ms, tot > 0 ? 100.0 * b->ms / tot : 0.0,
                (unsigned long long)b->calls, b->calls ? b->ms / (double)b->calls : 0.0);
    }
    fprintf(stderr, "=== end PULSAR_MOE_TIME ===\n");
}
static void moe_time_accum(uint32_t gt, uint32_t dt, double ms){
    for (int i = 0; i < g_moe_time_n; i++)
        if (g_moe_time[i].gate_type == gt && g_moe_time[i].down_type == dt) {
            g_moe_time[i].ms += ms; g_moe_time[i].calls++; return;
        }
    if (g_moe_time_n < 64) {
        static int registered = 0;
        if (!registered) { registered = 1; atexit(moe_time_dump); }
        g_moe_time[g_moe_time_n].gate_type = gt; g_moe_time[g_moe_time_n].down_type = dt;
        g_moe_time[g_moe_time_n].ms = ms; g_moe_time[g_moe_time_n].calls = 1; g_moe_time_n++;
    }
}

int pulsar_gpu_routed_moe_batch_tensor(pulsar_gpu_tensor *out, pulsar_gpu_tensor *up, pulsar_gpu_tensor *mid, pulsar_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const pulsar_gpu_tensor *selected, const pulsar_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const pulsar_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    /* Nothing below this line consults esz: every scratch/output cast in the
     * MoE lane is (float *) over a tensor the graph allocates f32.  That was
     * TRUE-BY-ACCIDENT rather than checked (types sweep 2026-08-22): if any
     * MoE tensor is ever narrowed, ten casts become defect nine at once,
     * silently.  So the whole lane's f32 assumption is enforced ONCE, here at
     * its two entry points. */
    if (pulsar_tensor_esz(out) != sizeof(float) || pulsar_tensor_esz(up) != sizeof(float) ||
        pulsar_tensor_esz(mid) != sizeof(float) || pulsar_tensor_esz(down) != sizeof(float) ||
        pulsar_tensor_esz(weights) != sizeof(float) || pulsar_tensor_esz(x) != sizeof(float)) {
        fprintf(stderr, "pulsar: routed MoE lane is f32-only; a narrowed tensor reached it\n");
        return 0;
    }
    static int time_moe = -1;
    if (time_moe < 0) time_moe = getenv("PULSAR_MOE_TIME") != NULL ? 1 : 0;
    if (!time_moe) {
        return routed_moe_batch_impl(out, up, mid, down, model_map, model_size,
                gate_offset, up_offset, down_offset, gate_type, down_type,
                gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                n_total_expert, n_expert, clamp, x, layer_index, n_tokens);
    }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    int r = routed_moe_batch_impl(out, up, mid, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_type, down_type,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, clamp, x, layer_index, n_tokens);
    cudaEventRecord(e, 0);
    if (cudaEventSynchronize(e) == cudaSuccess) {
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        moe_time_accum(gate_type, down_type, (double)ms);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return r;
}


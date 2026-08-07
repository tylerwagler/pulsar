
/* pulsar (plan 41b): SINGLE-tensor IQ2_XXS D2R, for a routed DOWN whose tensor
 * is IQ2 rather than Q2_K (our v5mx).  Upstream has a Q2_K single-tensor D2R
 * (down_q2k_d2r_kernel) and an IQ2 PAIR D2R, but no IQ2 single -- so an IQ2
 * down fell back to stock mul_mat_q at 33.24 ms/layer against Entrpi's 9.78 ms
 * Q2_K D2R.  That was the entire remaining gap after alignment.
 *
 * No new kernel is needed.  gateup_iq2_d2r_pair_kernel already separates the
 * two projections along blockIdx.z ("leg"), selecting
 *     W_soa = leg == 0 ? gate_soa : up_soa
 *     out   = leg == 0 ? out_gate : out_up
 * so launching the SAME kernel with gridDim.z = 1 pins leg to 0 and computes
 * exactly one tensor.  up_soa/out_up are passed as the gate pointers and are
 * never dereferenced.
 *
 * Appended to cuda/mmq/ds4_mmq_d2r.cu; declared in ds4_mmq_d2r.cuh.
 */
int ds4_mmq_iq2_xxs_moe_d2r_single_launch(const void *W_soa,
                                          int64_t soa_blocks,
                                          const void *q8,
                                          const int32_t *ids_dst,
                                          const int32_t *expert_bounds,
                                          float *out,
                                          int M,
                                          int K,
                                          int64_t ne_get_rows,
                                          int n_experts,
                                          void *worklist_scratch,
                                          size_t worklist_scratch_bytes,
                                          cudaStream_t stream) {
    (void)soa_blocks;
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_single_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return -1;
    }
    if (!W_soa || !q8 || !ids_dst || !expert_bounds || !out) {
        return -1;
    }
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0 || ne_get_rows <= 0) {
        return -1;
    }

    const int64_t capacity64 = d2r_worklist_capacity(ne_get_rows, n_experts, kNTile);
    if (capacity64 <= 0 || capacity64 > 0xFFFF) {
        return -1;
    }
    const size_t need = d2r_worklist_bytes(capacity64);
    if (!worklist_scratch || worklist_scratch_bytes < need) {
        fprintf(stderr, "%s: worklist scratch too small\n", tag);
        return -1;
    }
    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* z = 1: leg is pinned to 0, so only W_soa / out are ever touched. */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 1);
    const dim3 block(32, kWarps, 1);
    gateup_iq2_d2r_pair_kernel<<<grid, block, 0, stream>>>(
        W_soa, W_soa, (const block_q8_1_mmq *)q8, ids_dst, expert_bounds, work, n_items,
        out, out, M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

/* Does FlashInfer's SM120 sparse-MLA prefill kernel beat ours at OUR shape?
 *
 * Measured before any integration, because the integration is expensive
 * (repack our 528 B/row MXFP8 compressed KV into their 584 B/token packed
 * format, repage it into 64-row blocks, convert Q/out f32<->bf16) and the only
 * published number for this kernel on this model+arch is +1-4% prefill in vLLM.
 * If it does not clearly beat 6.917 ms here, none of that work is worth doing.
 *
 * No torch.  The kernels take raw pointers -- sparse_mla_prefill_dispatch() is
 * plain C++ -- so only csrc/sparse_mla_sm120_prefill.cu plus the
 * include/flashinfer/attention/sparse_mla_sm120 headers are needed; the torch
 * binding TU is deliberately not compiled.
 *
 * OUR REFERENCE, from the since-deleted tests/attn_indexed_bench.cu (a71e346)
 * calibrated to the engine ramp
 * at 3.9%, at the plateau (sub-batch 4+, causal depth >= 2048):
 *     attention_indexed_mixed_heads8_online_kernel<8,16>  6.917 ms / 512 tokens
 *     (the f32 indexed kernel of the time, deleted in L166; the live kernel
 *     is attn_f16_kernel in pulsar_cuda_attn_f16.cu)
 *
 * Shape (verified against the gguf and the engine trace):
 *     num_heads 64, head_dim 512 (DSV4: d_qk == d_v == 512), topk 512,
 *     512 tokens per launch, page_block_size 64, per-head sinks.
 * (64, 512) is in FlashInfer's own DSV4 dispatch set, so this is a supported
 * configuration rather than an off-label one.
 *
 * NOT MODELLED HERE, and it matters for interpreting the result: our kernel
 * also folds a 128-token raw window into the same online softmax.  FlashInfer
 * expresses that with extra_KV_cache/extra_indices (dispatch_dsv4_dual); this
 * bench runs the single path first to get a clean upper bound on the sparse
 * part alone.  If the single path is not already faster, the dual path -- which
 * does strictly more work -- will not be either.
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_120a -std=c++17 \
 *     -I/home/claude/fi/include -o /tmp/fibench \
 *     tests/flashinfer_sparse_mla_bench.cu /home/claude/fi/sparse_mla_sm120_prefill.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

using bf16 = __nv_bfloat16;

bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads,
                                 int topk, int page_block_size, int topk_extra,
                                 int extra_page_block_size, const bf16* Q, const uint8_t* KV_cache,
                                 const int32_t* indices, const uint8_t* extra_KV_cache,
                                 const int32_t* extra_indices, bf16* output, float* out_lse,
                                 float sm_scale, int num_tokens, size_t stride_kv_block,
                                 size_t stride_kv_block_extra, const float* attn_sink,
                                 const int* topk_length, const int* extra_topk_length,
                                 cudaStream_t stream);

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static const double kOursMs = 6.917;   /* our kernel, same shape, plateau */

int main(int argc, char **argv) {
    const int num_tokens = argc > 1 ? atoi(argv[1]) : 512;
    const int iters      = argc > 2 ? atoi(argv[2]) : 5;
    const int num_heads  = 64;
    const int topk       = 512;
    const int d          = 512;          /* DSV4: d_qk == d_v == 512 */
    const int pbs        = 64;           /* _DECODE_DSV4_PAGE_BLOCK_SIZE */
    const int bpt        = 584;          /* _BPT_DSV4 bytes per token */
    const int n_pages    = 2048;         /* enough to index into */

    CK(cudaSetDevice(0));

    bf16 *Q=nullptr, *out=nullptr; uint8_t *kv=nullptr; int32_t *idx=nullptr;
    float *lse=nullptr, *sinks=nullptr;
    const size_t q_el   = (size_t)num_tokens * num_heads * d;
    const size_t out_el = (size_t)num_tokens * num_heads * d;
    const size_t kv_bytes = (size_t)n_pages * pbs * bpt;
    const size_t idx_el = (size_t)num_tokens * topk;

    CK(cudaMalloc(&Q,     q_el   * sizeof(bf16)));
    CK(cudaMalloc(&out,   out_el * sizeof(bf16)));
    CK(cudaMalloc(&kv,    kv_bytes));
    CK(cudaMalloc(&idx,   idx_el * sizeof(int32_t)));
    CK(cudaMalloc(&lse,   (size_t)num_tokens * num_heads * sizeof(float)));
    CK(cudaMalloc(&sinks, num_heads * sizeof(float)));

    printf("shape tokens=%d heads=%d d=%d topk=%d page_block=%d bytes/token=%d\n",
           num_tokens, num_heads, d, topk, pbs, bpt);
    printf("alloc Q %.2f MiB  KV %.1f MiB  indices %.1f MiB\n",
           q_el*sizeof(bf16)/1048576.0, kv_bytes/1048576.0, idx_el*4/1048576.0);

    std::mt19937 rng(1234);
    {
        std::vector<uint8_t> h(1u<<20);
        for (auto &v : h) v = (uint8_t)rng();
        for (size_t o=0;o<kv_bytes;o+=h.size())
            CK(cudaMemcpy(kv+o, h.data(), std::min(h.size(), kv_bytes-o), cudaMemcpyHostToDevice));
        std::vector<bf16> hq(1u<<18);
        std::uniform_real_distribution<float> uf(-1.f,1.f);
        for (auto &v : hq) v = __float2bfloat16(uf(rng));
        for (size_t o=0;o<q_el;o+=hq.size())
            CK(cudaMemcpy(Q+o, hq.data(), std::min(hq.size(), q_el-o)*sizeof(bf16),
                          cudaMemcpyHostToDevice));
        std::vector<float> hs(num_heads);
        for (auto &v : hs) v = uf(rng);
        CK(cudaMemcpy(sinks, hs.data(), hs.size()*sizeof(float), cudaMemcpyHostToDevice));
        /* indexer output: topk distinct token indices per query, ascending --
         * the same shape our indexer emits. */
        std::vector<int32_t> hi(idx_el);
        std::uniform_int_distribution<int> ui(0, n_pages*pbs - 1);
        for (int t=0;t<num_tokens;t++){
            for (int i=0;i<topk;i++) hi[(size_t)t*topk+i] = ui(rng);
            std::sort(hi.begin()+(size_t)t*topk, hi.begin()+(size_t)(t+1)*topk);
        }
        CK(cudaMemcpy(idx, hi.data(), hi.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    auto run = [&]() {
        return sparse_mla_prefill_dispatch(
            ModelType::DSV4, num_heads, topk, pbs, /*topk_extra=*/0,
            /*extra_page_block_size=*/0, Q, kv, idx, /*extra_KV=*/nullptr,
            /*extra_indices=*/nullptr, out, lse, 1.0f/32.0f, num_tokens,
            /*stride_kv_block=*/(size_t)pbs*bpt, /*stride_kv_block_extra=*/0,
            sinks, /*topk_length=*/nullptr, /*extra_topk_length=*/nullptr, 0);
    };

    if (!run()) {
        printf("\n  sparse_mla_prefill_dispatch REFUSED this configuration.\n"
               "  (num_heads=%d, topk=%d, page_block=%d, DSV4) is not dispatchable\n"
               "  in this build -- that is itself the answer for the shape.\n",
               num_heads, topk, pbs);
        return 2;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best=1e30;
    for (int i=0;i<iters;i++){
        CK(cudaEventRecord(a)); run(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms=0.f; CK(cudaEventElapsedTime(&ms,a,b));
        if (ms<best) best=ms;
    }
    printf("\n  flashinfer  %8.3f ms/launch (sparse only, no window)\n", best);
    printf("  ours        %8.3f ms/launch (sparse + 128-token window fused)\n", kOursMs);
    printf("  ratio       %8.3fx %s\n", kOursMs/best,
           best < kOursMs ? "(flashinfer faster)" : "(ours faster)");
    printf("\n  NOTE: flashinfer here does strictly LESS work -- no raw window.\n"
           "  Treat this as an upper bound on what the port could win.\n");
    return 0;
}

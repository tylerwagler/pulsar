// SPDX-License-Identifier: MIT
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// The module contains dense and routed kernels for Q8_0, Q2_K, Q4_K, and
// IQ2_XXS, plus the GB10 aligned-SoA D2R and fused MoE epilogues. Numerical
// parity and long-context coverage live in tests/cuda_long_context_smoke.cpp.

#include "ds4_mmq.h"

#include "ds4_cuda_env.cuh"
#include "../pulsar_cuda_scratch.h"
#include "ds4_act_block.cuh"
#include "mmid.cuh"
#include "ds4_mmid.cuh"
#include "ds4_mmq_d2r.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_MMQ_HAS_NVTX 1
#endif
#endif
#ifndef DS4_MMQ_HAS_NVTX
#define DS4_MMQ_HAS_NVTX 0
#endif

// Upstream deleted get_mmq_x_max_host in the post-5c0e946 restructure: the tile
// width J is now an INPUT to ggml_cuda_mmq_get_config rather than something the
// host derives.  We only ever used it to size the q8_1 scratch, and the bound
// that matters is the largest J the launcher can pick, which is the top of its
// search loop -- `for (int J = 8; J <= 128 ...)` in mmq.cuh.  128 also matches
// what the old helper returned on Turing-MMA hardware, so the scratch is the
// same size as before.  Deliberately an upper bound: under-sizing this buffer
// would be an overflow, over-sizing costs a few KiB once. (L008)
// ⚠ 128 HERE WAS A BUG (fixed 2026-08-14): it undersized the q8_1 scratch and
// produced 129280/129280 NON-FINITE logits at prefill depth 4102 (a 4096 chunk
// plus a 6-token tail).  I had reasoned that 128 was "the top of the launcher's
// J search loop" -- that is the TILING loop (`for J = 8; J <= 128`), not the
// bound that sizes this buffer.  The real bound is
// ggml_cuda_mmq_get_J_max = min(ne11, 512) rounded down to a multiple of 8, so
// J reaches 512 and the quantize kernel wrote past the end.
//
// 512 is that hard cap, so it can never undersize regardless of type, cc or
// ne11.  Upstream calls the accessor per-tensor to allocate exactly; we
// deliberately take the constant instead -- the over-allocation is bounded by
// 512*sizeof(block_mx_act_mmq) once per call, and a constant cannot be wrong the
// next time upstream changes how J is chosen.
static constexpr int ds4_mmq_x_max() { return 512; }

// Reservation arithmetic for the scratch arena: each take() 256-aligns, so a
// reservation must budget the ALIGNED size of every slice it will hand out.
static constexpr size_t ds4_mmq_a256(size_t b) { return (b + 255u) & ~(size_t)255u; }


static uint64_t ds4_mmq_nvtx_payload(uint32_t first, uint32_t second) {
    return ((uint64_t)first << 32) | second;
}

class ds4_mmq_nvtx_scope {
public:
    ds4_mmq_nvtx_scope(const char *name, uint64_t payload, bool enabled)
        : active_(enabled) {
#if DS4_MMQ_HAS_NVTX
        if (active_) {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attr.payload.ullValue = payload;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = name;
            (void)nvtxRangePushEx(&attr);
        }
#else
        (void)name;
        (void)payload;
        active_ = false;
#endif
    }

    ~ds4_mmq_nvtx_scope() {
#if DS4_MMQ_HAS_NVTX
        if (active_) (void)nvtxRangePop();
#endif
    }

    ds4_mmq_nvtx_scope(const ds4_mmq_nvtx_scope &) = delete;
    ds4_mmq_nvtx_scope &operator=(const ds4_mmq_nvtx_scope &) = delete;

private:
    bool active_;
};

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

// Task #29 parked a persistent Q8_1 scratch buffer here, behind
// DS4_CUDA_MMQ_Q81_PERSISTENT, to test one hypothesis: that
// ggml_cuda_pool_alloc records a cudaMallocAsync node into the captured layer
// graph, so at replay the alloc returns a different address while the matvec's
// pointer argument stays baked in from capture time -- which would make a
// captured decode disagree with an eager one on identical inputs.  The
// experiment was never adopted and the env was never set by anything, so the
// buffer could not be allocated and its consumer branch could not be taken;
// removed 2026-08-17.  The hypothesis is kept because it is worth knowing, and
// it is still UNTESTED -- do not read this comment as a negative result.



/* DS4_MMQ_D2R and DS4_MMQ_D2R_IQ2 are GONE (2026-08-15).  Both were
 * default-ON kill switches back to the mul_mat_q SoA-tile path, and a
 * repo-wide grep found ZERO setters -- no test, no script, no Makefile, no
 * caller.  Per [[no-hot-path-flags]] an opt-out with no real user is deleted
 * along with the arm it selected; a comment is not a user.  D2R is simply what
 * IQ2 experts do. */

// Blanket output zeroing on the dense/MoE-down/pair GEMM entries.  Added by
// 82b2622 as belt-and-suspenders while root-causing the cont BOS spam; the
// actual roots were fixed in the same commit (stream-K fixup write_back goes
// dense + tmp_fixup zeroed + ncols_max=ne_get_rows), after which every
// element a consumer reads is stored by the GEMM itself and the zeroing was
// ~1.0 s/12k-admission of pure memset tax.  REMOVED (2026-07-09 gated
// increment: L42 deep tensors BIT-IDENTICAL with/without, same-boot ABBA
// 641.5 -> 678 tok/s @12k, gsm8k 119/120 / mbpp 36/40 / canary=[]).


extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    /* The AMD arms are gone (L066 step 3, 2026-08-18).  Upstream branched here
     * on CDNA / CDNA3 / RDNA3 / RDNA3_0 / RDNA3_5 via amd_mfma_available and
     * amd_wmma_available; this engine runs on GB10 and nothing else, and `cc`
     * is read from the DEVICE, so those arms were unreachable at runtime rather
     * than merely uncompiled.  They were also the majority of what common.cuh
     * still had to supply -- eight CC macros and two predicates for branches
     * that cannot be taken -- which is what kept a 1,669-line vendored header
     * alive.  The NVIDIA behaviour below is unchanged, including the pre-Turing
     * fp16/DP4A knee, which is still reachable on an older NVIDIA part. */
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch used
// by the dense and routed entry points.
namespace {

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds4_mmq_sanitize_f32(float *p, uint64_t n, cudaStream_t stream) {
    if (!p || n == 0) return;
    ds4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

} // anonymous namespace






// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ggml_cuda_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. quantize_mmq_q8_1_cuda with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds4_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifact; when non-null the mmq
         * kernel loads tiles from it directly and W is ignored (see mmq_args). */
        const char    * x_soa      = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): false skips the whole-buffer nonfinite pass; only valid
         * when every consumer sanitizes at read (the routed-MoE swiglu/sum
         * kernels do). */
        bool            sanitize_out = true) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    /* The vendored ggml pool is gone (L066): scratch now comes from the MMQ
     * arena slot, which is a single persistent reservation rather than a
     * per-call alloc.  That also retires task #22's ds4_pool_set_stream --
     * there are no pool ops left to order against the kernels, because there is
     * no allocation on the stream at all. */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);

    /* Every buffer this call needs, sized up front, taken from ONE arena on the
     * MMQ scratch slot.  Sizing has to be hoisted above the first take because
     * an arena reserves once; that is the only thing the arena asks of a caller
     * that the vendored pool did not, and it buys a single reservation per call
     * instead of five pool round-trips.  All five are live simultaneously --
     * D2R is mandatory below (the non-D2R arm returns -1), so nothing here is
     * reserved for a path that will not run. */
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_mx_act_mmq) / DS4_ACT_BLOCK_VALS +
        ds4_mmq_x_max() * sizeof(block_mx_act_mmq);
    const size_t w_bytes =
        ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
    if (w_bytes == 0) {
        fprintf(stderr, "%s: D2R scratch sizing failed (rows=%lld experts=%d)\n",
                tag, (long long)ne_get_rows, n_experts);
        return -1;
    }
    const size_t ids_bytes = (size_t)ne_get_rows * sizeof(int32_t);
    const size_t eb_bytes  = (size_t)(n_experts + 1) * sizeof(int32_t);

    // 1. Build the expert-major work map.
    cuda_arena ar;
    if (!cuda_arena_begin_slot(&ar, CUDA_SCRATCH_MMQ,
                               ds4_mmq_a256(ids_bytes) * 2 + ds4_mmq_a256(eb_bytes) +
                               ds4_mmq_a256(nbytes_src1_q8_1) + w_bytes, tag)) {
        fprintf(stderr, "%s: scratch reservation failed\n", tag);
        return -1;
    }
    int32_t *ids_src1      = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    int32_t *ids_dst       = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    int32_t *expert_bounds = (int32_t *)cuda_arena_take(&ar, eb_bytes, 256);
    char    *src1_e4m3_p   = (char *)cuda_arena_take(&ar, nbytes_src1_q8_1, 256);
    char    *d2r_work      = (char *)cuda_arena_take(&ar, w_bytes, 256);
    if (!d2r_work) return -1;   /* take() latches: one check covers all five */

    // Task #22 root-cause fix: mm_ids_helper COMPACTS - it only writes ids_src1
    // entries for in-range router ids and drops invalid ones (the router's NaN
    // path emits -1 by design), so with any dropped id the tail of ids_src1
    // stays unwritten pool memory.  quantize_mmq_q8_1's grid covers all
    // ne_get_rows rows and gathers x rows via ids_src1[i1] unconditionally
    // (quantize.cu:304), so a stale/garbage tail entry becomes a wild OOB read
    // (the intermittent batched-draft illegal access; B200 memcheck-convicted).
    // Zero both id maps so unwritten tail slots gather/scatter row 0 instead:
    // those lanes' output is never consumed (the mmq write-back loop is
    // expert_bounds-bounded), the cost is a few KB of memset on-stream.
    cudaMemsetAsync(ids_src1, 0, ids_bytes, stream);
    cudaMemsetAsync(ids_dst,  0, ids_bytes, stream);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    // The smem mm_ids_helper uses n_tokens * 4 bytes of dynamic shared memory;
    // the down matmul reaches here with n_tokens = assignments (6x the forward
    // width), so 8192-row prefill chunks pass 48384 "tokens" > cap.  The cap
    // check now lives in ds4_launch_mm_ids_helper (ds4_mmid.cu), which past the
    // cap dispatches the bit-identical two-pass global variant instead of
    // asserting — refusing here used to throw the WHOLE MoE block (including
    // gate/up mmq work) onto the legacy expert-tile fallback, the W8192 prefill
    // cliff.  
    ds4_launch_mm_ids_helper(
        ids, ids_src1, ids_dst, expert_bounds,
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize the activation into E4M3.
    /* ONE decision, made once, BEFORE staging: does the D2R E4M3 path take this
     * call?  Staging then follows the decision, instead of writing both formats
     * on every call and letting the consumer pick.  IQ2 experts are E4M3-only
     * now -- there is no int8 arm -- so q8_1 is staged solely for the generic
     * MMQ fallback: another quant type, or D2R declining at launch. */
    static int d2r_iq2s_cc = -1;
    static int d2r_iq2s_avail = 0;
    if (d2r_iq2s_cc != cc) {
        d2r_iq2s_cc = cc;
        d2r_iq2s_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
    }
    const bool d2r_iq2 = (type == GGML_TYPE_IQ2_XXS && x_soa != nullptr &&
                          K % 256 == 0 && d2r_iq2s_avail != 0);


    // S1.1a fix (same as the dense path): the mmq Y buffer is over-allocated for the
    // kernel's tail-tile reads and ne_get_rows columns need not fill the final mmq
    // column tile, but quantize only writes the valid columns.  The mmq kernel
    // (mmq.cuh:3528) unconditionally loads the full tile, reading the never-written
    // tail from stale pool memory -> allocator-perturbation-dependent garbage in the
    // (write_back-masked) tail lanes -> non-deterministic batched-forward output.
    // Zero it so the masked-out tail is a deterministic zero.
    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    /* d2r_iq2 is a REQUIREMENT here, not a preference.  This function is
     * instantiated for exactly one type (GGML_TYPE_IQ2_XXS, from
     * ds4_mmq_iq2_xxs_moe_soa), so "another quant type could arrive and want
     * q8_1" describes no caller.  Every way d2r_iq2 can be false is a defect:
     * a null SoA pointer, K not a multiple of 256, or an architecture below
     * Ampere -- none of which a working configuration produces.
     *
     * The q8_1 staging that used to sit here covered those cases by silently
     * running this GEMM on int8 affine activations while every other expert
     * GEMM ran E4M3.  That is a FORMAT divergence, and a size-thresholded
     * version of exactly it is what once hid the down conversion for two days.
     * Fail closed instead: one activation format, every batch size, or an
     * error that says which precondition broke. */
    if (!d2r_iq2) {
        fprintf(stderr,
                "%s: D2R E4M3 preconditions not met (soa=%p K=%d K%%256=%d avail=%d) -- "
                "refusing to run these expert activations in another format\n",
                tag, (const void *)x_soa, (int)K, (int)(K % 256), d2r_iq2s_avail);
        return -1;
    }
    cudaMemsetAsync(src1_e4m3_p, 0, nbytes_src1_q8_1, stream);
    ds4_quantize_mmq_e4m3_cuda(
        X_f32, ids_src1, (void *)src1_e4m3_p,
        /*ne00=*/K, s11_src, s12_src, s13_src,
        /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
        /*n_expert_used=*/0, /*scatter=*/false, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: ds4_quantize_mmq_e4m3_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.


    /* pulsar (plan 41b): IQ2_XXS twin of the Q2_K block below.  Our v5mx down
     * tensors are IQ2, not Q2_K, so without this an aligned IQ2 down still ran
     * stock mul_mat_q.  Same gating, same scratch contract; the launcher pins
     * the pair kernel's leg to 0 so it computes one tensor. */
    /* ⚠ NO d2r_min_cols() HERE, deliberately.  It used to read
     * `ne_get_rows >= d2r_min_cols()`, which sent every batch under 1024 rows to
     * the generic MMQ kernel below -- i.e. the SAME GEMM ran E4M3 activations on
     * long prompts and int8 q8_1 on short ones, chosen by batch size.  That is
     * the A8 invariant broken by a threshold, and it hid the down conversion
     * completely: a 26-token prompt gives 156 rows, so the arm never engaged and
     * two days of cross-arm measurements came back bit-identical.
     * One path, one activation format, every batch size. */
    {
        const int rc = ds4_mmq_iq2_xxs_moe_d2r_single_launch(
            x_soa, soa_blocks,
            src1_e4m3_p,
            ids_dst, expert_bounds,
            out_f32, M, K, ne_get_rows, n_experts, d2r_work, w_bytes,
            stream);
        /* Every rc != 0 is a bad pointer, a bad shape, or undersized scratch --
         * a caller defect, never a shape this path legitimately hands back. */
        if (rc != 0) {
            fprintf(stderr, "%s: D2R launch declined (rc=%d, rows=%lld) -- no fallback\n",
                    tag, rc, (long long)ne_get_rows);
            return -1;
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: D2R launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }
    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

// Produce the weighted SwiGLU rows in their canonical pair-major order. The
// proven upstream quantizer below gathers them through the already available
// ids_dst map, so gate/up and down share one expert-major schedule without a
// second mm_ids_helper.

// Paired MoE: one helper + one quantize covers both weights.  See the
// header comment on ds4_mmq_iq2_xxs_moe_pair for motivation.  Internal
// structure mirrors ds4_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template <ggml_type type>
int ds4_mmq_moe_pair_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifacts for W_a / W_b (same
         * shape, so one block count); see ds4_mmq_moe_impl. */
        const char    * xa_soa     = NULL,
        const char    * xb_soa     = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): see ds4_mmq_moe_impl. */
        bool            sanitize_out = true,
        /* Optional: the E4M3 encoding of X_f32 that its producing norm already
         * emitted, plus the ue8m0 plane and its blocks-per-row pitch. When
         * present the staging gathers these bytes instead of re-encoding the
         * f32 -- same output, a quarter of the read. NULL = encode from f32. */
        const void    * act_q      = NULL,
        const void    * act_sf     = NULL,
        int             act_kbp    = 0) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const bool nvtx_prefill = false;
    ds4_mmq_nvtx_scope fused_scope(
            "ds4/prefill/moe/mmq_fused",
            ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_expert_used),
            nvtx_prefill);

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    /* The vendored ggml pool is gone (L066): scratch now comes from the MMQ
     * arena slot, which is a single persistent reservation rather than a
     * per-call alloc.  That also retires task #22's ds4_pool_set_stream --
     * there are no pool ops left to order against the kernels, because there is
     * no allocation on the stream at all. */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);

    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_mx_act_mmq) / DS4_ACT_BLOCK_VALS +
        ds4_mmq_x_max() * sizeof(block_mx_act_mmq);
    /* One arena on the MMQ scratch slot for every buffer this call needs; see
     * the note in ds4_mmq_moe_impl.  D2R is mandatory on both arms below, so
     * nothing reserved here is for a path that will not run. */
    const size_t d2r_work_bytes =
        ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
    const size_t ids_bytes = (size_t)ne_get_rows * sizeof(int32_t);
    const size_t eb_bytes  = (size_t)(n_experts + 1) * sizeof(int32_t);
    cuda_arena ar;
    if (!cuda_arena_begin_slot(&ar, CUDA_SCRATCH_MMQ,
                               ds4_mmq_a256(ids_bytes) * 2 + ds4_mmq_a256(eb_bytes) +
                               ds4_mmq_a256(nbytes_src1_q8_1) +
                               ds4_mmq_a256(d2r_work_bytes), tag)) {
        fprintf(stderr, "%s: scratch reservation failed\n", tag);
        return -1;
    }
    int32_t *ids_src1 = nullptr;
    int32_t *ids_dst = nullptr;
    int32_t *expert_bounds = nullptr;

    ids_src1      = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    ids_dst       = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    expert_bounds = (int32_t *)cuda_arena_take(&ar, eb_bytes, 256);
    if (!expert_bounds) return -1;   /* take() latches */

    const int si1  = n_expert_used;
    const int sis1 = 1;

    // Cap handling lives in ds4_launch_mm_ids_helper (see ds4_mmq_moe_impl).
    cudaError_t err = cudaSuccess;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/expert_map",
                ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_experts),
                nvtx_prefill);
        // Task #22 root-cause fix (same as ds4_mmq_moe_impl): zero the id maps
        // so entries dropped by mm_ids_helper never expose stale pool memory.
        cudaMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
        cudaMemsetAsync(ids_dst,  0, ne_get_rows * sizeof(int32_t), stream);
        ds4_launch_mm_ids_helper(
            ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11,
            si1, sis1, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
            return -2;
        }
    }
    /* The fused target-prefill path receives a true top-k assignment: one
     * token cannot select the same expert twice, so no expert bucket can
     * exceed n_tokens rows. Keep the conservative gathered-row bound for all
     * generic MMQ callers, including DSpark/MTP. */

    /* The materialized path stream-frees gate/up Q8_1 before allocating the
     * down Q8_1. The direct path needs both simultaneously, but writes down
     * Q8_1 into caller-owned gate scratch instead of growing the CUDA pool. */
    {
    char *src1_e4m3 = (char *)cuda_arena_take(&ar, nbytes_src1_q8_1, 256);

    /* ONE decision, made once, BEFORE staging.  IQ2 experts are E4M3-only now;
     * q8_1 exists solely for the generic MMQ fallback (another quant type, or
     * D2R declining at launch) and is staged lazily below, so the common IQ2
     * path never writes a format it does not use. */
    static int d2r_iq2_avail_cc = -1;
    static int d2r_iq2_avail = 0;
    if (d2r_iq2_avail_cc != cc) {
        d2r_iq2_avail_cc = cc;
        d2r_iq2_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
    }
    const bool d2r_iq2 = (type == GGML_TYPE_IQ2_XXS && xa_soa != nullptr &&
                          xb_soa != nullptr && K % 256 == 0 &&
                          d2r_iq2_avail != 0);

    // S1.1a fix (same as the dense/moe paths): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.cuh:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/input_quant",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)K),
                nvtx_prefill);
        /* E4M3 or nothing -- same requirement as the single-tensor impl. */
        if (!d2r_iq2) {
            fprintf(stderr,
                    "%s: D2R E4M3 preconditions not met (soa=%p K=%d K%%256=%d avail=%d) -- "
                    "refusing to run these expert activations in another format\n",
                    tag, (const void *)xa_soa, (int)K, (int)(K % 256), d2r_iq2_avail);
            return -1;
        }
        if (!src1_e4m3) return -1;   /* take() latches */
        cudaMemsetAsync(src1_e4m3, 0, nbytes_src1_q8_1, stream);
        /* Prefer the producer's own encoding when the caller handed one over:
         * identical bytes, but a 1-byte read per element instead of 4 and no
         * encode at all.  Falling back to encoding from f32 is a COST choice,
         * not a format one -- both paths emit the same E4M3. */
        /* One-shot, like the other tier announcements: which way the expert
         * activations got their E4M3 is provenance worth seeing once, and it is
         * the difference between "this made no difference" and "this never
         * ran". */
        static int staging_announced = 0;
        if (!staging_announced) {
            staging_announced = 1;
            fprintf(stderr, "pulsar: MoE gate/up E4M3 staging = %s\n",
                    (act_q && act_sf) ? "gathered from the producer's encoding"
                                      : "encoded from f32 (no cached encoding)");
        }
        if (act_q && act_sf) {
            ds4_gather_mmq_e4m3_cuda(
                act_q, act_sf, act_kbp, ids_src1, (void *)src1_e4m3,
                /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
                /*n_expert_used=*/0, /*scatter=*/false, stream);
        } else {
            ds4_quantize_mmq_e4m3_cuda(
                X_f32, ids_src1, (void *)src1_e4m3,
                /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
                /*n_expert_used=*/0, /*scatter=*/false, stream);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: ds4_quantize_mmq_e4m3_cuda failed: %s\n", tag, cudaGetErrorString(err));
            return -3;
        }
    }



    bool gate_up_done = false;
    /* No d2r_min_cols() -- see the note on the single-tensor guard in
     * ds4_mmq_moe_impl.  A size threshold here means one activation format for
     * long prompts and another for short ones. */
    if (d2r_iq2) {
        {
            char *d2r_work = (char *)cuda_arena_take(&ar, d2r_work_bytes, 256);
            if (d2r_work) {
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/iq2_gate_up_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                        nvtx_prefill);
                const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
                        xa_soa, xb_soa, soa_blocks,
                        src1_e4m3, ids_dst,
                        expert_bounds, out_a, out_b, M, K, ne_get_rows, n_experts,
                        d2r_work, d2r_work_bytes, stream);
                if (d2r_rc == 0) {
                    gate_up_done = true;
                }
            }
        }
    }

    /* No q8_1 fallback: see the requirement note in ds4_mmq_moe_impl. D2R
     * declining is a defect -- bad SoA pointer, K not a 256-multiple, undersized
     * scratch -- not a shape this path legitimately returns, and covering it by
     * silently switching gate/up to int8 affine activations is the exact
     * divergence that once hid the down conversion. */
    if (!gate_up_done) {
        fprintf(stderr,
                "%s: D2R gate/up declined (rows=%lld) -- no fallback\n",
                tag, (long long)ne_get_rows);
        return -1;
    }
    }

    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds4_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

} // anonymous namespace





/* ds4 (P4 Inc3): mmq MoE over the aligned row-pair-SoA Q2_K artifact
 * (weight server --repack-q2k-aligned) -- no raw-layout weights and no
 * derepack scratch involved; the mul_mat_q tile loader reads the SoA
 * sections directly (load_tiles_q2_K_soa, bit-identical tiles). */






/* THE WHOLE "direct gate/up" FUSED PATH IS GONE (2026-08-15).
 * ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa was an exported entry point with no
 * callers, and the only thing that ever set direct_gateup_q8 true.  A path trace
 * confirmed its launcher never executed.  Because the field was ALSO the gate on
 * the E4M3 staging, it silently disabled the MXFP8 arm and made every cross-arm
 * measurement come back bit-identical -- a dead flag guarding an unreachable
 * branch, costing more than the branch was ever worth.
 * Removed with it: the field, its four `if` branches, the caller-owned scratch
 * members, ds4_mmq_iq2_xxs_moe_d2r_fused_launch and the fused SwiGLU kernel.
 * gate/up now has exactly one shape: D2R pair launch, else generic MMQ. */

/* ds4 (P4 Inc3): paired mmq MoE over the aligned-SoA IQ2_XXS gate/up
 * artifacts (weight server --repack-iq2-aligned); same contract as
 * ds4_mmq_q2_K_moe_soa. */
extern "C" int ds4_mmq_iq2_xxs_moe_pair_soa(
        const void * Wa_soa, const void * Wb_soa,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream,
        const void * act_q, const void * act_sf, int act_kbp) {
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_moe_pair_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t nblk = (int64_t)n_experts * (int64_t)M * (int64_t)(K/256);
    /* sanitize_out=false: see ds4_mmq_q2_K_moe_soa. */
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_soa", Wa_soa, Wb_soa, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream,
        (const char *)Wa_soa, (const char *)Wb_soa, nblk,
        /*sanitize_out=*/false, act_q, act_sf, act_kbp);
}


/* REMOVED 2026-08-18 (ledger L066, extraction step 2): the mmvq-backed half.
 *
 * It held ds4_mmq_q8_0_aligned_dense_vec, its q8_0_aligned_dense_vec_kernel,
 * ds4_mmq_vec_dot_q8_1, ds4_mmq_iq2_xxs_aligned_bytes, and the explicit
 * mul_mat_q_case<Q8_0/IQ2_XXS/Q4_K> instantiations.  Every one had ZERO
 * undefined references anywhere in the object set -- defined, linked, and
 * called by nothing.
 *
 * That is not an accident of this audit.  The only path that could still have
 * reached mul_mat_q_case makes D2R a hard REQUIREMENT and fails closed rather
 * than falling back to q8_1 ("one activation format, every batch size, or an
 * error that says which precondition broke"), so once the routed experts moved
 * to E4M3 the vendored kernels became unreachable by construction.
 *
 * The instantiation comment claimed these "drag in the load_tiles_<type> +
 * vec_dot_<type> device functions ... so the .o objects contain everything
 * needed to link against the public C entries".  True when written; the public
 * C entries stopped going through mul_mat_q_case, and the comment did not
 * notice.  It is the reason the vendored kernel family looked load-bearing.
 *
 * This also drops the last consumer of ds4_cuda_q8_fold_take_q81, the
 * producer-emitted-q8_1 registry callback, which is removed above. */

/* pulsar (plan 41b): IQ2_XXS single-tensor MoE over the aligned-SoA artifact.
 * Upstream has q2_K_moe_soa (single) and iq2_xxs_moe_pair_soa (pair) but no
 * IQ2 SINGLE soa entry; a routed DOWN whose tensor is IQ2 rather than Q2_K
 * (our v5mx) needs exactly that.  Body mirrors ds4_mmq_q2_K_moe_soa with the
 * IQ2 block count from ds4_mmq_iq2_xxs_moe_pair_soa (blocks, not row-pairs),
 * and keeps sanitize_out=true so semantics match the raw entry it replaces. */
extern "C" int ds4_mmq_iq2_xxs_moe_soa(
        const void * W_soa, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_moe_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t nblk = (int64_t)n_experts * (int64_t)M * (int64_t)(K/256);
    return ds4_mmq_moe_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_moe_soa", W_soa, X, ids, out,
                                               M, K, n_tokens, n_experts, n_expert_used, stream,
                                               (const char *)W_soa, nblk,
                                               /*sanitize_out=*/true);
}

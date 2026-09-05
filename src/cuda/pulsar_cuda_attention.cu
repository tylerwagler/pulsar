#include "pulsar_cuda_internal.h"

/* Attention dispatch.  Every attention launch in this engine -- prefill raw
 * window, prefill static-mixed, batched decode (causal and the drafter's
 * non-causal raw window), indexed top-k -- goes to the ONE fp16 tensor-core
 * kernel in pulsar_cuda_attn_f16.cu (L161/L164/L166).  This file holds the
 * tensor-level argument checks, the shape refusals and the top-k sort; it
 * launches no attention kernel of its own.  The f32 kernels that used to live
 * here (prefill raw, prefill mixed, indexed heads8-online, decode heads8-online
 * + split-KV merge) were deleted in L166 once their last shapes -- n_tokens ==
 * 1 and the non-causal drafter window -- moved to the fp16 kernel; each had
 * been a second numerics for a shape the fp16 kernel also served.
 *
 * Every KV operand is PULSAR_ATTN_PACK rows -- 384 B NVFP4 at head_dim 512:
 * e2m1 nibbles under per-16 e4m3 scales and an f32 row scale, rope dims bf16
 * -- so ONE decoder (attn_comp_pack_ld / attn_comp_row_ld4 in
 * pulsar_cuda_internal.h) serves the sliding-window ring, the compressed pool,
 * the drafter's ring and the current chunk alike (L111).  The row is LOSSY vs
 * the f32 pipeline; what makes it shippable is the measured L111 verdict.  All
 * pack formats are pointer-compatible, so a wrong operand reads rows at the
 * wrong stride -- out of bounds, NaN, and a clean compile -- which is why the
 * launchers below check byte bounds at the packed stride. */

/* positions/seq_id/comp_cap -- the descriptor contract of the fp16 kernel's
 * ring mode (pulsar_cuda_attn_f16.cu), validated by the launchers below: per-row
 * multi-session banking (design adapted from the MIT-licensed Entrpi/ds4
 * fork, v0.2 c71a49a — see pulsar_bank_slabs in the engine; reimplemented, no
 * code copied).  positions[t] is row t's absolute query position, seq_id[t]
 * its TRUE bank id (never a packed row ordinal); raw window, ring start and
 * the visible compressed count are derived per row from the position because
 * the raw ring is position-indexed (slot = pos % raw_cap per bank) and
 * compression closes one ratio group every `ratio` positions.  Banked rows
 * read the raw ring at seq_id*raw_cap and compressed rows at seq_id*comp_cap
 * offsets; the scalar n_comp becomes a cross-bank superset used ONLY as a
 * clamp/scratch bound, never to address into a specific bank.  The per-row
 * visible count is (qpos+1)/ratio — the SAME rule the engine's classic
 * single-session decode follows, because the engine emits a step's
 * compressed row BEFORE attention (gpu_decode.cpp: the bank's ms_n_comp entry
 * is incremented before the attention launch reads it), so at an emit step
 * (qpos ≡ ratio-1 mod ratio) the row attends to the compressed row emitted
 * that same step.  DRIVER CONTRACT (banked mode): every bank's compressed
 * rows for the current step — including same-step emits — must be written
 * before the attention launch; the scalar n_comp superset clamp is a safety
 * bound only.  If the clamp ever bites, the row reads fewer rows than
 * single-session would (fail-safe, not garbage) and its output DIVERGES
 * from classic — that is the mid-prefill-bank case the driver must never
 * co-schedule.  positions == NULL && seq_id == NULL degenerates to the
 * classic single-cache scalar path bit-exactly. */

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

int pulsar_gpu_attention_prefill_raw_heads_mx_tensor(pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const pulsar_gpu_tensor *q, const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope, int *mx_out,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep) {
    if (mx_out) *mx_out = 0;
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_HEADS_ELT_SIZE ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < (uint64_t)n_tokens * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* One kernel for every row count (L166): the fp16 tensor-core tier is the
     * raw-window prefill arm at n_tokens == 1 as at 4096.  A 1-token chunk
     * (a prompt of 4096k+1 tokens, a one-token prompt, a one-token suffix)
     * used to take an f32 per-(token,head) kernel -- the L161 class of split,
     * on the prefill side.  Shapes the tier does not serve have no kernel and
     * are refused by name; a 0 from the launcher is a real failure, reported
     * rather than demoted.  Under banked descriptors the kernel ropes Q at
     * positions[t] (the dense row plan is unchanged: a zero-prefix batch's
     * raw rows are its own tokens). */
    if (head_dim != 512u) {
        fprintf(stderr, "pulsar: prefill raw attention: head_dim %u has no kernel (only 512 is built)\n", head_dim);
        return 0;
    }
    if ((n_head % 32u) != 0u) {
        fprintf(stderr, "pulsar: prefill raw attention: n_head %u has no kernel "
                        "(the fp16 tensor-core tier tiles 32 heads per block)\n", n_head);
        return 0;
    }
    if (!pulsar_gpu_attn_f16_tier_on()) {
        fprintf(stderr, "pulsar: prefill raw attention: this device has no fp16 tensor-core tier "
                        "(mma.m16n8k16 needs sm_80+); no other kernel is built\n");
        return 0;
    }
    /* One-shot branch report.  This is the RAW entry; pulsar-bench showed real
     * prefill never reaches the mixed launch, so this is where production
     * prefill attention is actually served. */
    static int raw_path_reported = 0;
    if (!raw_path_reported) {
        raw_path_reported = 1;
        fprintf(stderr,
                "pulsar: ATTN-RAW n_tokens=%u head_dim=%u window=%u -> fp16 tensor-core tier "
                "(fp16 operands, f32 accumulate; every row count)\n",
                n_tokens, head_dim, window);
    }
    /* Only the fp16 tier emits the grouped E4M3 encoding.  If it declines,
     * *mx_out stays unset and the launcher refuses: the "a" GEMM has no
     * quantiser of its own, and a half-written encoding would be a wrong
     * answer. */
    if (pulsar_gpu_attention_f16_prefill_mx(
            (pulsar_heads_t *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
            (const pulsar_attn_pack_t *)raw_kv->ptr, NULL,
            n_tokens, 0u, window, 1u, n_head, head_dim,
            gact_data, gact_scale, gact_kbp, gact_slab, n_groups, n_nope,
            0u, n_tokens,
            positions ? (const int *)positions->ptr : NULL, q_prep)) {
        if (mx_out && gact_data) *mx_out = 1;
        return 1;
    }
    fprintf(stderr, "pulsar: fp16 prefill raw attention FAILED (n_tokens=%u n_head=%u "
                    "window=%u); refusing to fall through\n", n_tokens, n_head, window);
    return 0;
}

static int attention_decode_batch_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                non_causal,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    /* Descriptor (banked) mode: both per-row arrays or neither; the KV
     * operands are whole bank pools, so byte bounds scale by n_banks and the
     * uint32 row ABI (seq*cap + local) must not overflow.  The scalar
     * n_raw/raw_start are IGNORED and NOT validated in this mode (pass 0):
     * the raw span and ring start are per-row derived in the kernels from
     * positions[t].  raw_cap must still be the true per-bank ring capacity
     * (it addresses the ring and sizes the byte bound); scalar n_comp is the
     * cross-bank superset clamp (comp_cap is the per-bank row stride, so it
     * must cover it).  All rejections here are fail-loud: a silent 0 return
     * looks like a generic launch failure to the driver. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (n_comp != 0 && comp_cap < n_comp) ||
         /* Per-row derivation clamps the raw span to `window`, and the
          * kernels' raw_rows scratch holds PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP
          * rows.  A zero (unbounded) or over-cap window would be silently
          * truncated to the OLDEST rows — fail loud instead. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked decode attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    const uint32_t kernel_n_banks = descr ? n_banks : 1u;
    /* comp_kv is the per-bank comp cache. With split allocations + a base-pointer
     * table (increment 2a) each bank is its OWN comp_cap-row allocation, and
     * comp_kv is bank 0's — so the byte bound is per-bank (comp_cap), not
     * n_banks*comp_cap. Without a table (legacy contiguous), it spans all banks. */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_HEADS_ELT_SIZE ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < kv_banks * raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0 ||
        (n_comp && comp_kv->bytes < comp_rows_min *
         PULSAR_ATTN_PACK_ROWBYTES(head_dim)) ||
        false) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) {
        fprintf(stderr, "pulsar: decode attention: %u compressed rows with compression ratio 0 -- "
                        "the visible compressed count is (pos+1)/ratio; refusing\n", n_comp);
        return 0;
    }
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    /* Per-bank comp base-pointer table (descriptor mode only; NULL → the kernel's
     * scalar base + seq_id*comp_cap fallback, bit-identical to the contiguous pool). */
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* ONE decode attention kernel (L161, L164, L166): the fp16 tensor-core
     * kernel serves every row count, every context length and both
     * visibility rules.  It tiles over heads, so one query row is a natural
     * input; it walks every compressed row, so it has no scratch that could
     * run out; and non_causal is a flag it carries (the drafter's raw-window
     * forward: a query sees every raw row up to the last one, not only rows
     * at or before its own position -- pulsar_cuda_attn_f16.cu, the ring
     * preamble).  Three selectors once split one conversation's attention
     * numerics without a shape change -- the row count (n_tokens > 1 chose
     * the tier; all 129280 head logits differed, rows/L161.md), the context
     * length (an f32 kernel's score scratch was checked before the tier
     * choice, rows/L164.md) and the visibility rule (the f32 kernel survived
     * as the non-causal arm, rows/L166.md).  All three are gone with the f32
     * kernel.  What the fp16 kernel does not serve has no kernel here and is
     * refused by name below; a 0 from the launcher is a real failure and is
     * reported, not demoted. */
    if (head_dim != 512u) {
        fprintf(stderr, "pulsar: decode attention: head_dim %u has no kernel (only 512 is built)\n", head_dim);
        return 0;
    }
    if ((n_head % 32u) != 0u) {
        fprintf(stderr, "pulsar: decode attention: n_head %u has no kernel "
                        "(the fp16 tensor-core tier tiles 32 heads per block)\n", n_head);
        return 0;
    }
    if (!pulsar_gpu_attention_prefill_reads_packed_comp()) {
        fprintf(stderr, "pulsar: decode attention: this device has no fp16 tensor-core tier "
                        "(mma.m16n8k16 needs sm_80+); no other kernel is built\n");
        return 0;
    }
    static int announced_dc = 0;
    if (!announced_dc) {
        announced_dc = 1;
        fprintf(stderr, "pulsar: decode attention = fp16 tensor-core tier "
                        "(every row count, causal and non-causal)\n");
    }
    if (pulsar_gpu_attention_f16_indexed(
            (pulsar_heads_t *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
            (const pulsar_attn_pack_t *)raw_kv->ptr,
            n_comp ? (const pulsar_attn_pack_t *)comp_kv->ptr : (const pulsar_attn_pack_t *)raw_kv->ptr,
            NULL /* no topk: visible-prefix sweep */,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
            0u, window, ratio, n_head, head_dim,
            (const int *)positions_ptr, (const int *)seq_id_ptr,
            comp_bank_ptrs_ptr, comp_cap, kernel_n_banks, non_causal, q_prep))
        return 1;
    fprintf(stderr, "pulsar: fp16 decode attention FAILED (n_tokens=%u n_head=%u "
                    "n_comp=%u non_causal=%u); refusing to fall through\n",
            n_tokens, n_head, n_comp, non_causal);
    return 0;
}

int pulsar_gpu_attention_prefill_raw_heads_tensor(pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const pulsar_gpu_tensor *q, const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep) {
    return pulsar_gpu_attention_prefill_raw_heads_mx_tensor(heads, model_map, model_size, sinks_offset,
                                                            q, raw_kv, n_tokens, window, n_head, head_dim, NULL, NULL, 0, 0u, 0u, 0u, NULL,
                                                            positions, q_prep);
}

int pulsar_gpu_attention_decode_raw_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, non_causal, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim,
                                      positions, seq_id, NULL /* raw path: no comp */, comp_cap, n_banks,
                                      q_prep);
}

int pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, non_causal,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim,
                                      positions, seq_id, comp_bank_ptrs, comp_cap, n_banks,
                                      q_prep);
}

int pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        const pulsar_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    /* Descriptor (banked) mode: same contract as attention_decode_batch_launch
     * (scalar n_raw/raw_start ignored and unvalidated, raw_cap must be the true
     * per-bank ring capacity, rejections fail-loud).  Banked rows take the same
     * fp16 kernel as scalar rows (its ring preamble derives the per-row plan
     * from the descriptors); there is no single-bank-only arm. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 || ratio == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         comp_cap < n_comp ||
         /* raw_rows scratch bound — see attention_decode_batch_launch. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked indexed attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u ratio=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window, ratio);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    /* Per-bank split: comp_kv is bank 0's comp_cap-row allocation when the base-
     * pointer table is present (see attention_decode_batch_launch). */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_HEADS_ELT_SIZE ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < kv_banks * raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0 ||
        comp_kv->bytes < comp_rows_min * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) {
        fprintf(stderr, "pulsar: indexed attention: top_k %u > 512 has no kernel -- refusing\n", top_k);
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    /* comp_selected is WRITTEN as uint32_t by the indexer; read here as
     * int32_t.  Same width, and every value is a row index < n_comp << 2^31,
     * so the signedness restatement cannot change a bit -- noted so the next
     * format change treats the pair as one fact. */
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    /* The indexed kernel takes topk[] in whatever order it is given, masks each
     * id outside the row's visible prefix, and folds rows through an online
     * softmax that is correct for ANY permutation; but the fold is
     * order-dependent in floating point, so the ORDER of topk[] is part of the
     * arithmetic.  Sort by id at EVERY row count (L170): the fold order is then
     * a function of the SET selected, not of which n_comp-bucketed ranking
     * kernel (1024 / pow2 / CUB) produced it and not of the batch width.  It
     * used to sort only n_tokens > 1, and that was a live 1-row-vs-N-row split
     * of the L161 class: at 2217 tokens of context (selection engaged) a 1-row
     * decode step and a 2-row step differed on all 129280 logits (max |dlogit|
     * 2.81); cuda-row-neutrality-gate-deep holds that shape.  The B300
     * reference is indifferent (prefill never takes a one-row indexed step).
     * top_k != 512 is a shape the sort kernel does not take; those ids fold in
     * ranking order for every row count, one order still. */
    if (top_k == 512u) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) {
            fprintf(stderr, "pulsar: indexed attention: %llu-byte scratch for the top-k id sort unavailable "
                            "(n_tokens=%u) -- refusing\n", (unsigned long long)sort_bytes, n_tokens);
            return 0;
        }
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    /* One kernel for every row count (L166): the fp16 tensor-core tier is the
     * indexed arm at n_tokens == 1 too -- the per-token indexed prefill loop
     * and a 1-token indexed chunk used to take an f32 heads8-online kernel
     * (the L161 class of split: 1-row and N-row indexed attention were two
     * numerics).  Banked descriptors and ATTN_PACK comp rows ride the tier,
     * each behind its own gate: bank isolation is proven algebraically
     * (tests/attn_f16_banked_test.cu -- a wrong-bank read is plausible
     * attention, not an error, so it needs a test that cannot be fooled), and
     * packed rows decode through the one shared attn_comp_pack_ld.  Shapes the
     * tier does not serve are refused by name. */
    if (head_dim != 512u) {
        fprintf(stderr, "pulsar: indexed attention: head_dim %u has no kernel (only 512 is built)\n", head_dim);
        return 0;
    }
    if ((n_head % 32u) != 0u) {
        fprintf(stderr, "pulsar: indexed attention: n_head %u has no kernel "
                        "(the fp16 tensor-core tier tiles 32 heads per block)\n", n_head);
        return 0;
    }
    if (!pulsar_gpu_attention_prefill_reads_packed_comp()) {
        fprintf(stderr, "pulsar: indexed attention: this device has no fp16 tensor-core tier "
                        "(mma.m16n8k16 needs sm_80+); no other kernel is built\n");
        return 0;
    }
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "pulsar: indexed attention = fp16 tensor-core tier (every row count)\n");
    }
    if (pulsar_gpu_attention_f16_indexed(
            (pulsar_heads_t *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
            (const pulsar_attn_pack_t *)raw_kv->ptr, (const pulsar_attn_pack_t *)comp_kv->ptr,
            (const int *)topk_ptr, n_tokens, pos0, n_raw, raw_cap,
            raw_start, n_comp, top_k, window, ratio, n_head, head_dim, (const int *)positions_ptr,
            (const int *)seq_id_ptr, comp_bank_ptrs_ptr,
            comp_cap, descr ? n_banks : 1u, 0u /* causal */, q_prep))
        return 1;
    fprintf(stderr, "pulsar: fp16 indexed attention FAILED (n_tokens=%u n_head=%u n_comp=%u "
                    "top_k=%u); refusing to fall through\n", n_tokens, n_head, n_comp, top_k);
    return 0;
}

static int attention_prefill_mixed_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        void                    *gact_data,
        void                    *gact_scale,
        int                      gact_kbp,
        uint32_t                 gact_slab,
        uint32_t                 n_groups,
        uint32_t                 n_nope,
        int                     *mx_out,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep) {
    if (mx_out) *mx_out = 0;   /* set only by a successful fp16-tier emission */
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_HEADS_ELT_SIZE ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < (uint64_t)n_tokens * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        /* Pack-aware, like the three sibling launches.  A guard that hard-codes
         * the f32 row stride demands 2048 B from a 384 B packed pool, fails,
         * and returns 0 -- which is "did not encode", not an error, so the
         * graph silently does not run.  ->bytes is just a number, so the
         * mismatch is type-legal and compiles clean. */
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_KV4_NV_BLOCK) != 0 ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp *
         PULSAR_ATTN_PACK_ROWBYTES(head_dim)) ||
        false) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* One kernel for every row count (L166) -- see the twin in the raw-window
     * launcher.  This is the site that carries the traffic: the raw-window one
     * runs twice a prefill, this one runs per layer. */
    if (head_dim != 512u) {
        fprintf(stderr, "pulsar: prefill mixed attention: head_dim %u has no kernel (only 512 is built)\n", head_dim);
        return 0;
    }
    if ((n_head % 32u) != 0u) {
        fprintf(stderr, "pulsar: prefill mixed attention: n_head %u has no kernel "
                        "(the fp16 tensor-core tier tiles 32 heads per block)\n", n_head);
        return 0;
    }
    if (!pulsar_gpu_attn_f16_tier_on()) {
        fprintf(stderr, "pulsar: prefill mixed attention: this device has no fp16 tensor-core tier "
                        "(mma.m16n8k16 needs sm_80+); no other kernel is built\n");
        return 0;
    }
    /* One-shot: which shape actually serves this workload.  Guessing at this
     * has been wrong twice; print it rather than infer it. */
    static int mixed_path_reported = 0;
    if (!mixed_path_reported) {
        mixed_path_reported = 1;
        fprintf(stderr,
                "pulsar: ATTN-MIXED n_tokens=%u n_comp=%u -> fp16 tensor-core tier "
                "(fp16 operands, f32 accumulate; every row count)\n",
                n_tokens, n_comp);
    }
    /* Same emission contract as the raw-window twin: only the fp16 tier may
     * write the grouped encoding, and on a decline *mx_out stays unset and
     * the launcher refuses rather than reading a half-written slab. */
    if (pulsar_gpu_attention_f16_prefill_mx(
            (pulsar_heads_t *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
            (const pulsar_attn_pack_t *)raw_kv->ptr,
            n_comp ? (const pulsar_attn_pack_t *)comp_kv->ptr : NULL,
            n_tokens, n_comp, window, ratio, n_head, head_dim,
            gact_data, gact_scale, gact_kbp,
            gact_slab, n_groups, n_nope,
            0u, n_tokens, NULL /* dense zero-prefix batch: rope at t */, q_prep)) {
        if (mx_out && gact_data) *mx_out = 1;
        return 1;
    }
    fprintf(stderr, "pulsar: fp16 prefill mixed attention FAILED (n_tokens=%u n_head=%u "
                    "n_comp=%u); refusing to fall through\n", n_tokens, n_head, n_comp);
    return 0;
}

int pulsar_gpu_attention_prefill_static_mixed_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        void                    *gact_data,
        void                    *gact_scale,
        int                      gact_kbp,
        uint32_t                 gact_slab,
        uint32_t                 n_groups,
        uint32_t                 n_nope,
        int                     *mx_out,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv,
                                       gact_data, gact_scale, gact_kbp,
                                       gact_slab, n_groups, n_nope, mx_out,
                                       n_tokens,
                                       n_comp, window, ratio, n_head, head_dim,
                                       q_prep);
}


# TP engine split — design decisions (slices 4+, Phase 2/4)

Decision record for how the two Spark ranks split the model, grounded in the
vLLM DSv4 TP/EP study (private: `pulsar-notes/vllm-ds4-tp-ep-study-2026-08-26.md`
— evidence and sources; this file is the public record the code implements).
Port decisions: ledger L102, plan 102, docs/tensor-parallel-port.md.

## The split model (two ranks = one EP group)

1. **Routed experts: whole-expert-per-rank.** vLLM is EP-first for MoE — routed
   experts are never split inside an expert; 256 experts land 128 per rank and
   tokens are dispatched between ranks every MoE layer. This is exactly
   upstream ds4's default ("one contiguous half of the routed experts").
   **Payoff:** halves per-node routed-expert weight bytes AND halves per-token
   active-expert GEMM FLOPs at batch 1 (each node computes each token only for
   its ~3 of the 6 active experts). Per-token dispatch bytes are unchanged
   (a function of routing, not the split).

2. **Attention + KV cache: replicated per rank, never sharded.** vLLM keeps the
   latent MLA projections (`W_DKV`, `kv_a`, `W_KR`, `W_DQ`) and the whole
   compressed KV cache local to every TP rank; each rank attends over the full
   sequence on its local head slice with its own local cache copy. **No
   cross-node KV traffic during attention.** Our attn-pack KV layout stays
   replicated on both boxes.

3. **Attention head split (Phase 4) is a re-scope.** The real mechanism is
   NOT "split the attention weights": it is heading the *per-head*
   up-projections (`W_UK/W_UV` → kv_b, `W_UQ` → q_b) as sharded-by-head
   ColumnParallel GEMMs, with `W_O` a RowParallel all-reduce per layer, while
   Q/KV/rope latents stay replicated. That per-head + output GEMM surface is
   where the ~2.82 GB/token attention byte cut lives. Do not spend Phase-4
   effort on sharding the replicated latent side.

4. **Vocab/output head: split.** Worker ships its logits half on the control
   socket (frames already ported in slice 3a).

5. **Shared expert, norms, router, embeddings: replicated** (upstream default).

## Transport consequence — RDMA is mandatory for decode TP

At batch-1 on V4-Flash, per token ≈ 1 dispatch (≈8 KiB bf16/rank-pair) + 1
combine (≤ topk×hidden, 24-48 KiB when remote experts) + 1 all-reduce per MoE
layer ≈ **2-4 small messages/layer × 43 layers**. At 10-30 µs RoCE RTT that is
≲ 4 ms/token inside the ~11 ms/token budget at 90 t/s — viable. At TCP
fallback RTT (100-1000 µs) the same pattern is 14-140 ms/token — **dead**.
So: TCP stays a bring-up/unit-test path only (why the loopback tests exist);
the pair's direct 200G RoCE DAC is load-bearing. Our f32 gate payload is
exactly 2× the bf16 dispatch (f32 partials, never quantized on the wire).

## Not transferable (keep our fine-grained per-gate design)

Bulk collectives/NCCL-all-reduce-rings (8+ rank shaped), DP/mega-batch
pipelines, CUDA-graph capture + handle-caching amortization, EPLB/weight-
shuffle orchestration, MTP/spec-decode machinery, and vLLM's DSpark
c4a/c128a compressor + lightning-indexer are overhead at 2 ranks × batch 1,
not structure. DeepEP's low-latency decode mode is the same per-layer
dispatch+combine shape as ours (RDMA internode) — the closest external
calibration, minus its handle caching.

## Constants cross-check (verify at binder time)

43 layers, hidden 4096, 64 heads × 512 (+64 rope), q/o_lora_rank 1024,
256 routed + 1 shared expert, top-6, moe_intermediate 2048, vocab 129280,
1M context, 284B total / ~13B activated, first 3 layers hash-routed, fp4
expert dtype. Should match our shape profile; any mismatch is a bind-time
bug, not a design change.

## Slice-4 sequencing (what is desk-testable vs GPU-gated)

- 4a. Identity extraction from a loaded engine (host code, GPU to run).
- 4b. CUDA gate machinery on the engine worker thread — big_gate first
      (prefill), per-layer gates (decode). GPU-gated; the hard chunk.
- 4c. Ownership-aware routed-MoE kernels (skip peer-owned experts, emit the
      f32 partial). GPU-gated.
- 4d. Vocab head split on the logits path (frames ported; engine-side wiring).
- 4e. Phase-3 lockstep over our session surface (banks/warm-fork, multiseq,
      mixed, spec rounds).

## Open items for bring-up
- Slab must be host-coherent + GPU-visible on GB10 (unified memory; register
  once via ibv_reg_mr) — negative path already exercised.
- Two-cable RDMA: pick device explicitly (`PULSAR_TP_RDMA_DEV`); bench 1-link
  vs 2-link.
- Caveat recorded, not re-litigated: community reports the shipped
  "NVFP4"-dtype DS-V4-Flash artifact is actually FP8 under the covers; and
  V4-Flash official docs are thin (mechanics drawn from V3 + V4 tech report +
  vLLM source).

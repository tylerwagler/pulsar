# TP engine split — design decisions (slices 4+, Phase 2/4)

Decision record for how the two Spark ranks split the model, grounded in the
vLLM DSv4 TP/EP study (private: `pulsar-notes/vllm-ds4-tp-ep-study-2026-08-26.md`
— evidence and sources; this file is the public record the code implements).
Port decisions: ledger L102, plan 102, docs/tensor-parallel-port.md.

Merged with the `tensor_parallel` branch into `dev` 2026-09-04 (the branch is
history; this file lives on dev). Slice-4 status below is current as of that
merge — the remaining 4b-CUDA/4c/4d/4e chunks are GPU/pair-gated.

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

- 4a. **DONE 2026-08-26** — DS-default identity builder
      `pulsar_tp_identity_init_defaults` (src/tp) + host tests. Engine open
      still fails loudly (see the guard) until 4b, per the fail-loud rule.
      **Binder cross-check PASSED:** `shape_profiles.cpp` Flash {43L, 4096h,
      129280v, 512 head-dim, 64 rot, 256+1 exp, 6 used} matches the study's
      V4-Flash constants exactly.
- 4b. **DONE → dev (host half only, by design; merged 2026-09-04)** — gate
      scheduler `pulsar_tp_sched.{h,cpp}`: drives the transport per-exchange
      over the DS per-layer ATTN→FFN order and the prefill big-gate-per-layer
      path, with the two CUDA-touching steps (.cu write/read partial) funneled
      through hook pointers → every scheduling/lockstep rule is host-tested
      (tp_sched_test: 2 decode tokens x 86 gates + prefill chunk, symmetric
      on both ranks over the TCP loopback).  Also landed in the merge: the
      host-pinned slab allocator (`pulsar_tp_gpu_slab_alloc_hostpin/free_hostpin`,
      src/tp/pulsar_tp_gpu.cpp — `cudaHostRegister` verdict).  The `.cu`
      wrapper implementations deliberately ship with their engine callers
      (no dead code) — see the hook-targets inventory note in port.md.
- 4b. **Prefill big-gate arm (engine-wired, compile-verified 2026-09-21).**
      `pulsar_engine::open` now stands up the pair when `--tp-role` + `--tp-arm
      prefill`: builds the identity from the resolved shape, `pulsar_tp_create`
      (leader listens / worker dials), allocates the host-pinned GPU-visible
      slab and `pulsar_tp_attach_slab`; teardown on `destroy()`.  The prefill
      path (gpu_prefill.cpp `tp_prefill_big_gate`, called from
      `gpu_graph_encode_layer_ffn_batch` before the HC expansion) exchanges one
      big gate per layer per chunk on the layer's routed contribution and folds
      the peer's partial in.  Compile-only so far (no runtime) — the `.cu`
      gate kernels for the DECODE per-layer arm are still unwired and
      `tp_role != 0` with no `--tp-arm` still fails loudly (rule 4).  The
      per-rank partial is ownership-aware only once 4c lands.
- **N-way transport (2026-09-22).** The transport is no longer two-rank-only:
      `pulsar_tp_create_mesh` brings up **n** ranks in a full mesh
      (`--tp-rank`/`--tp-nranks`/`--tp-peers`; every rank connects to every
      other, rank-ordered dial/accept) and `pulsar_tp_allreduce_sum` combines
      the routed partial across all n ranks (all-gather + local sum in
      canonical ascending-rank order so every rank bits the same full value).
      The n=2 legacy `--tp-role`/`--tp-peer` pair path and its RDMA are
      unchanged.  Perimeter of this increment: decode/batch gates and the
      command plane fail loudly for n>2 (rule 9) until their n-way slices;
      per-peer RDMA is pair-gated; and the sum remains n× the full-model value
      until 4c ownership makes each rank emit its owned 1/n partial.
- 4c. Ownership-aware routed-MoE kernels (skip peer-owned experts, emit the
      f32 partial). GPU-gated.
- 4d. Vocab head split on the logits path (frames ported; engine-side wiring).
- 4e. Phase-3 lockstep over our session surface (banks/warm-fork, multiseq,
      mixed, spec rounds).

## Open items for bring-up
- **Slab (resolved on-pair 2026-09-02, allocator merged → dev):**
  `cudaMallocManaged` does NOT register with `ibv_reg_mr` on GB10 (EFAULT).
  Host-pinned (`malloc` + `cudaHostRegister`) registers and the probe passes
  over RDMA — `pulsar_tp_gpu_slab_alloc_hostpin` (src/tp/pulsar_tp_gpu.cpp,
  on dev) is the allocator slice 4b-CUDA uses; kernels still write it directly
  over unified memory, no D2H/H2D bounce. **GPUDirect RDMA is unavailable on this platform**
  (tests/tp_dmabuf_probe: attrs 110/116 = 0, CUDA-13 GPURDMA flag rejected,
  dma-buf import EINVAL) — so host-pinned is final, not a stopgap. Recheck the
  probe if the nvidia driver ever advertises `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`.
- Two-cable RDMA: pick device explicitly (`PULSAR_TP_RDMA_DEV`); bench 1-link
  vs 2-link.
- Caveat recorded, not re-litigated: community reports the shipped
  "NVFP4"-dtype DS-V4-Flash artifact is actually FP8 under the covers; and
  V4-Flash official docs are thin (mechanics drawn from V3 + V4 tech report +
  vLLM source).

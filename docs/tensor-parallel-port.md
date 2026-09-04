# Tensor Parallelism (TP) — port seed

Status of the two-Spark TP effort, kept in-repo as the decision record. The
`tensor_parallel` branch was retired into `dev` 2026-09-04 (zero commits absent
from dev; nothing unique lost) — `dev` is the single active line, so read
`origin/dev` for any engine analysis. All TP code lives only inside dev's
history.
Source of truth for scope decisions: `~/Projects/pulsar-notes/plans/102-tensor-parallel-two-sparks.md`
(private) + ledger L102. This file is the public, in-tree pointer that code
comments can cite (like the Lnnn tags elsewhere).

## Decision (Tyler, 2026-08-26)

- **TP pays on dual DGX Sparks.** The 90+ t/s full-weight TP2 decode on the work
  pair (stock vLLM, `-tp 2`, DSpark spec, direct 200G NDR RoCE) is the existence
  proof. The plan's Phase-0 bandwidth-vs-latency gate is WAVED for this decision.
- **This is a forward-port, not a restore.** Our fork point (80ebbc3) predates
  upstream's TP; upstream's two-machine TP is Metal-only, so the GPU half is new
  CUDA work and only the transport (ds4_tp.c ~2.2k lines) lifts.
- **Interconnect is characterized (read-only, 2026-08-26, re-verified 2026-09-02):**
  the pair has TWO direct-attach 200G NDR RoCE wires: `rocep1s0f1`
  (192.168.0.12/.13 — the NCCL/vLLM wire) and `roceP2p1s0f1`
  (192.168.9.12/.13/30, MTU 9000 — the TP bring-up wire, separate from NCCL),
  RS-FEC, 0 CRC errors; no `nvidia_peermem` loaded yet TP2 runs at speed
  (GB10 unified-memory registerable-slabs thesis holds). Production pair is
  off-limits (read-only, ~10 GB free) — engine bring-up needs a spare box or
  an approved window; a co-tenant transport/probe run can pin
  `PULSAR_TP_RDMA_DEV=roceP2p1s0f1` and stay on the 9.x wire.
  **TWO QSFP cables are attached** (all four CX-7 ports LINK_UP; `rocep1s0f0`
  and `roceP2p1s0f0` carry no IP); the bench must measure 1-link vs 2-link
  (documented Spark-pair results: ~13.5 GB/s single NIC vs ~24.5 GB/s
  merged). The transport keeps RDMA device selection explicit so a later
  multi-link merge stays possible.

## Scope locked

Lift adapts (transport): `ds4_tp.{c,h}` → `src/tp/pulsar_tp.{cpp,h}` (C++, our
naming/error/log conventions, `pulsar_*` prefix; no `.c` TUs in this tree).

New CUDA work (no upstream CUDA version exists):
- Gate machinery: `pulsar_gpu_tp_{init,gate_encode,batch,big,kick,wait}` — stream/event
  choreography on the engine worker thread, register-once slab, must not break the
  CUDA-graph tiers.
- Ownership-aware routed-MoE kernels (our MXFP4/MXFP8_LT layout; skip peer-owned
  experts, emit partials).
- Vocab-split output head (CUDA twin of upstream's Metal version).
- Lockstep mirroring over OUR session surface (banks/warm-fork, multiseq, mixed,
  spec rounds, logprobs) — the underestimated cost, ~3x upstream's op set.

Deferred (last): attention head split — where most of the byte payoff lives but
highest risk; interacts with MLA, attn-pack KV, indexer.

## Design rules (non-negotiable)

1. **Every CUDA call stays on the engine worker thread.** The transport *may*
   block on a service thread; the slab recv-completion may never call CUDA off
   the worker. Otherwise the per-context `g_cublas` globals race returns.
2. **Slab registered once at init** (`ibv_reg_mr` on a host-coherent, GPU-visible
   cuda allocation on GB10); partials stay f32 on the wire, never quantized.
3. **TP output is NOT byte-exact vs single-box** (partials summed in a new order).
   Fidelity bar = reference-graded (`--check-reference` against the B300 logits),
   NOT our own baseline. `cuda-reference-gate` must never be graded while it prints
   SKIP (require `PULSAR_REF_DIR`).
4. **Fail loudly, not degraded.** Until a feature arm is wired, requesting it via
   the CLI must produce one clear error, per the plan's fail-not-trap rule (L028).

## Sequence (each increment compiles, no dead code left behind)

1. **Option/identity surface** — TP CLI parse + validate, `pulsar_tp_options`,
   `pulsar_tp_identity`; requesting TP before wiring errors clearly. Host-testable.
   **DONE → dev** (fail-loud open guard in place).
2. **Slab layout + pure arithmetic** — `pulsar_tp_slab_bytes`/offsets, the two-rank
   hello/identity exchange types, unit-tested (no sockets, no CUDA).
   **DONE → dev**.
3. **Transport alone** — import `pulsar_tp.{cpp,h}`, socket/RDMA bring-up + gate
   exchanges, exercised by a standalone two-thread (or two-host) loopback test.
   Still NO callers in the engine. **DONE → dev** (TCP + dlopen-libibverbs RDMA,
   loopback host suite; bench 1-link vs 2-link awaits the pair window).
4. **Prefill TP (Phase 1)** — write the CUDA `big_gate` path only, split routed
   experts, prove on prefill. First engine-visible TP. **= slice 4b-CUDA; OPEN,
   GPU-gated** (see slice-4 sequencing in docs/tensor-parallel-split.md).
5. **Decode gates + vocab head (Phase 2)** — per-layer gates, ownership-aware MoE,
   vocab-split output head. **= slices 4b-CUDA/4c/4d; OPEN, GPU-gated.**
6. **Session lockstep (Phase 3)** — mirror banks/warm-fork, multiseq, mixed, spec.
   **= slice 4e; OPEN, GPU-gated.**
7. **Attention head split (Phase 4)** — deferred; only after 1-6 prove transport.

Exit criteria per phase: numeric/gated on a TP pair, reference-graded where the
summation order changes. Single-box behavior must remain bit-identical (options
default off, zero TP code on the live path).

## 4b-CUDA hook targets (engine inventory, verified on dev 2026-09-04)

Where the gate-scheduler `write_partial`/`read_partial` hooks bind. Names are
stable; re-verify line numbers if they drift before the pair window.

- **Layer boundaries (decode + prefill).** Every layer runs through
  `gpu_graph_encode_layer_batch` (src/engine/gpu_prefill.cpp:2668), which chains
  `gpu_graph_encode_layer_attention_batch` then `gpu_graph_encode_layer_ffn_batch`
  and finally swaps the HC twins. The two gate points per layer are just after
  the ATTN call (output of attention) and just after the FFN call, before the
  cur/next swap (output of the layer). Decode goes through the multiseq graph
  tier (`gpu_graph_decode_multiseq_batch`, session.cpp) — a gate's recv is a
  host-side wait and cannot live inside a captured graph, so the machinery must
  straddle per-layer graph launches (this interaction is part of the 4b work).
- **Hidden carrier (what a partial is read from / added to).** The batched HC
  twins `g->batch_cur_hc` / `g->batch_next_hc`
  (pulsar_engine_internal.h:1013-1014), f32
  `[n_tokens, PULSAR_N_HC * PULSAR_N_EMBD]`, row = `batch_index * hc_dim`.
  The outgoing layer-il residual is the twin that pre-swap holds the layer's
  output — exact phase/offset (which twin, pre- vs post-swap) and where the
  peer partial lands is pinned at bind time.
- **Routed MoE output (slice 4c).** `pulsar_gpu_routed_moe_batch_tensor`
  (src/cuda/pulsar_cuda_moe.cu:1465) emits `out` f32 `[n_tokens, N_EMBD]` from
  the top-k `selected`/`weights` per token; the FFN accumulation at
  gpu_prefill.cpp:2620 sums `batch_routed_out` + `batch_shared_out` into
  `batch_ffn_out` (pulsar_engine_internal.h:1054-1063). Ownership-aware kernels
  take a contiguous 128-expert half per rank and emit an f32 partial instead of
  the pooled sum.
- **Vocab head (4d).** `gpu_graph_encode_output_head` (src/engine/gpu_decode.cpp:936),
  `output_embd` in, `logits` f32 N_VOCAB out; both bf16-matmul and MXFP8 arms
  (gpu_decode.cpp:1001-1011); per-token in decode, last row of the chunk in
  prefill.
- **Threading (design rule 1 holds on dev).** One GPU-capable thread: pthread at
  cli_main.cpp:1030 → `worker_main` (server_sched.cpp:2514) →
  `worker_{batched,mixed,spec}_batched_quantum` → `pulsar_session_decode_mixed`.
  `g_cublas` is created and bound to the per-thread default stream
  (pulsar_cuda_runtime.cu:916-920); kernels launch on `cudaStreamPerThread`
  (`--default-stream per-thread` build), events via
  `pulsar_gpu_marker_record`/`_done` (pulsar_cuda_runtime.cu:1594-1615), full
  sync only via `pulsar_gpu_synchronize`.
- **Already in place.** CLI parse for `--tp-role` (cli_main.cpp:445,
  pulsar_cli.cpp) and the fail-loud engine-open guard (session.cpp:296 —
  refuses `tp_role != 0` until 4b-CUDA lands). No other engine TP path exists;
  everything else in this section is new wiring.

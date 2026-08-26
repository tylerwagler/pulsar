# Tensor Parallelism (TP) — port seed

Status of the two-Spark TP effort, kept in-repo as the branch's decision record.
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
- **Interconnect is characterized (read-only, 2026-08-26):** the pair is wired
  direct-attach 200G NDR RoCE on `rocep1s0f1` (192.168.0.12/.13), RS-FEC, 0 CRC
  errors; no `nvidia_peermem` loaded yet TP2 runs at speed (GB10 unified-memory
  registerable-slabs thesis holds). Production pair is off-limits (read-only,
  ~10 GB free) — engine bring-up needs a spare box. **TWO QSFP cables are
  attached** (all four CX-7 ports LINK_UP); the bench must measure 1-link vs
  2-link (documented Spark-pair results: ~13.5 GB/s single NIC vs ~24.5 GB/s
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
2. **Slab layout + pure arithmetic** — `pulsar_tp_slab_bytes`/offsets, the two-rank
   hello/identity exchange types, unit-tested (no sockets, no CUDA).
3. **Transport alone** — import `pulsar_tp.{cpp,h}`, socket/RDMA bring-up + gate
   exchanges, exercised by a standalone two-thread (or two-host) loopback test.
   Still NO callers in the engine.
4. **Prefill TP (Phase 1)** — write the CUDA `big_gate` path only, split routed
   experts, prove on prefill. First engine-visible TP.
5. **Decode gates + vocab head (Phase 2)** — per-layer gates, ownership-aware MoE,
   vocab-split output head.
6. **Session lockstep (Phase 3)** — mirror banks/warm-fork, multiseq, mixed, spec.
7. **Attention head split (Phase 4)** — deferred; only after 1-6 prove transport.

Exit criteria per phase: numeric/gated on a TP pair, reference-graded where the
summation order changes. Single-box behavior must remain bit-identical (options
default off, zero TP code on the live path).

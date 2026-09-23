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
4b. **N-way transport (2026-09-22)** — the TP transport is generalized from a
   two-rank pair to an **n-rank full mesh**: `pulsar_tp_create_mesh` (rank-ordered
   dial/accept, `--tp-rank`/`--tp-nranks`/`--tp-peers`) and
   `pulsar_tp_allreduce_sum` (all-gather + local sum in canonical ascending-rank
   order).  The engine is no longer hard-coded to 2 GPUs (1..n).  Perimeter:
   decode/batch gates and the command plane fail loudly for n>2; per-peer RDMA
   is pair-gated; n=2 legacy pair (with RDMA) is unchanged.
5. **Decode gates + vocab head (Phase 2)** — per-layer gates, ownership-aware MoE,
   vocab-split output head. **= slices 4b-CUDA/4c/4d; OPEN, GPU-gated.**
6. **Session lockstep (Phase 3)** — mirror banks/warm-fork, multiseq, mixed, spec.
   **= slice 4e; IN PROGRESS.** Increment 1 (the prompt mirror on sync) landed
   2026-09-23; see the section below for the model it fixes.

### 4e: how the worker is driven (decided 2026-09-23)

Both ranks run the **same driver with the same arguments** (the CLI/server, told
apart by `--tp-role`); there is no worker-side receive loop and no session
registry. Only the **leader's arguments are authoritative**: the leader ships the
operation on the command plane and then waits for one ack per peer, and the worker
blocks for that frame and runs on what it received. A worker whose own driver
disagreed — a stale prompt, a truncated request — therefore cannot desync the
pair, because its own arguments are never read.

- The mirror lives at the **public API boundary** (`pulsar_session_sync_mm` and
  `pulsar_session_eval` in `engine_api.cpp`), *not* on the member
  `pulsar_session::sync`/`eval`. Those members are re-entered from inside a
  running operation (the image stitch on sync's resume path, `rewrite_from_common`,
  the speculative walk); a mirrored frame there would be a second, unbalanced half
  of an operation the peer is not expecting.
- The id every frame carries is the **create ordinal** (`pulsar_session::tp_session_id`,
  handed out by `pulsar_session::create` as `++tp_session_seq`). Both ranks agree
  on it by construction, with no wire round trip on the (unmirrored) create path.
  A frame whose id does not match the receiving session means the ranks' drivers
  diverged, and the worker refuses loudly rather than mirroring another session's
  tokens into this one.
- The leader collects the ack even when its own half failed: an ack left unread
  would be consumed by the *next* operation, shifting every later frame by one.
- `pulsar_tp_wait_command_ack` returns **1 on success, 0 on failure** (failure
  leaves the message in `err`), the inverse of the `tp_send_*` convention.
- Increment 2 (2026-09-23) mirrors **eval**.  The frame's `seq` is the session's
  decode position (`checkpoint.len`, the number of tokens whose KV the graph
  holds), so the worker does more than trust the leader's token: it refuses when
  the two ranks are not at the same position.  That is the one corruption the
  mirror alone cannot catch -- the right token decoded at the wrong place.
- **Session create is deliberately not mirrored.**  A worker cannot check the
  leader's create frame until it reaches the same create, and blocking there
  would turn a driver divergence -- the case the create ordinal exists to catch
  -- into a hang instead of the clean refusal the first mirrored operation
  already gives.
- `tests/tp_mesh_test.cpp` now rides the two frames a mirrored session actually
  uses (a 16-token SYNC array, then an EVAL position+token), with one ack per
  frame per peer, at n=2..5 over real RDMA.  Its gating is mutation-proven: a
  wrong expected position exits nonzero instead of printing and passing.
- Increment 3 (2026-09-23) mirrors **rewind and invalidate** -- and they are the
  one pair of mirrored operations that collects **no ack**, for two reasons that
  agree.  A `void` caller has nowhere to put a peer's refusal, and a leader that
  waited would hang on the first operation the peer's driver did not happen to
  make -- a live risk, because of the 49 `pulsar_session_rewind`/`_invalidate`
  call sites only 8 can run with a pair armed (src/server and src/cli; the
  kvstore and agent tools never arm TP), and several of those 8 are the server's
  cache and scheduler (`kv_cache.cpp:664`, `server_sched.cpp:318/2375`), whose
  timing follows LOCAL memory state rather than the request stream.  The frame
  is therefore fire-and-forget, and the worker's frame-type check is the
  divergence alarm: an unexpected frame marks the pair failed and prints, so the
  next *acked* operation carries the refusal to the leader instead of the pair
  hanging on it.  `tp_mesh_test` rides the REWIND+INVALIDATE pair and then an
  acked sync under a different session id, which proves the silence (a stray ack
  is read by that collect and fails it; verified by mutation).
- A worker must never skip a frame it was sent.  The `tp_mirror_dead`
  early-return is therefore **leader-only**: a worker that returned there would
  leave the leader blocked in `wait_command_ack` forever, and a dead transport
  failing its `recv` is the right ending while a hang is the wrong one.
- **The engine mirror layer has no on-box test, and it cost a real bug.**  All
  seven `pulsar_tp_send_*`/`recv_command` calls in the first three increments had
  the transport's convention backwards (nonzero is SUCCESS), so the leader
  reported "could not mirror the prompt" on every successful send and gave up,
  and the worker read a failed ack as delivered.  The mesh test could not catch
  it -- it drives the transport directly and never enters the engine wrappers --
  and a two-process engine pair on one box is impossible (earlyoom; see the
  bringup doc).  The earlier slices are what identified it: gpu_prefill's
  allreduce and gpu_decode's vocab all-gather both test `... != 0` for success.
- **The engine mirror layer now has its own on-box test** (`make tp-mirror-test`,
  `tests/tp_mirror_test.cpp`), which is what the convention bug above showed was
  missing.  It fabricates a session -- a zeroed `pulsar_engine` carrying a real
  mesh transport, and a zeroed `pulsar_session` whose `tp_session_id` is NOT the
  id the leader mirrors under -- and asserts the three refusal paths that return
  *before* the wrapper reaches the graph: worker session-id mismatch, worker
  frame-type mismatch, and a leader on a dead transport.  No model, no weights.
  Mutating the worker's recv check back to the broken convention fails it (empty
  error, then the leader hangs and the target's `timeout` turns that into a
  nonzero exit -- which is why the target is wrapped in `timeout`: a missing ack
  HANGS the leader, and a hang is not a test result).
  It still does not cover the wrappers' SUCCESS path, which calls `sync`/`eval`
  on the real graph; that needs two Sparks.
- Increment 4 (round 6) mirrors **`pulsar_session_decode_multiseq`** -- the
  decode path the production server actually uses (plain `eval` is the CLI
  path), so before this a server pair was not mirroring its real workload.
  It rides `FRAME_EVAL_BATCH`, whose payload had to be defined first: the item
  was `{session_id, token, reserved}` and could not carry a row's **bank** and
  **position**, which is why nothing had ever been written against it.  It is
  now `{session_id, bank, pos, token, reserved}` -- the engine's own row
  contract plus the session.  Frame *numbers* are still never reused; only that
  payload changed, once, while it had no users.
  The worker checks the row COUNT and refuses on a mismatch instead of warning
  (unlike a single token, whose value cannot resize anything): the caller sized
  `logits` for ITS OWN n, so decoding the leader's different count would decode
  into a buffer shaped for someone else's batch.  Both ranks still assemble full
  logits from the 4d vocab all-gather, so no logits cross the wire.
  Coverage: `tp_mesh_test` now rides a 3-row batch with distinct bank/pos/token
  at n=2..5 (mutating one field's expectation fails it), and `tp_mirror_test`
  adds the wrapper's batch refusal.
  **Not covered:** the wrapper's row *reconstruction* runs only after the frame
  checks pass, which on a fabricated session would reach the graph -- so it
  needs two Sparks like the rest of the success path.
- **Still open:** `decode_mixed` (its `FRAME_MIXED_BATCH` is now the only
  remaining frame with no user), banks, warm-fork, spec.
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

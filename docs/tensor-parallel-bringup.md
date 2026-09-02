# TP bring-up runbook (2026-08-26) — the first GPU session, scripted

Purpose: turn the first box session with the work pair (or any spare Spark pair)
into *running a pre-made harness*, not discovery. Every step names the command
and the "pass" line. The transport, scheduler, and tests here compile and the
host tests are green on a GPU-less box — what this runbook validates is the
runtime layer that only exists on real hardware.

⚠ **Prerequisite / constraint (from 2026-08-26): the work pair
(`ca1070wk30007/008`) is PRODUCTION and read-only — do not run anything heavy
or traffic-generating on it without an approved maintenance window, and do not
build/run our engine next to the live vLLM (~10 GB free). This runbook targets
a spare pair OR the work pair during an approved window.**

## 0. Baseline — build + host tests (any box, ~2 min)

```sh
make tp-core-test tp-transport-test tp-sched-test     # all three must print ok
make tests/tp_slab_gpu_probe                          # compile-check at sm_120f
```
If a session needs a full engine build first: `make cuda-spark`.

## 1. Interconnect sanity (read-only, ~1 min) — both boxes

```sh
rdma link; ibv_devinfo -l                 # 4 HCAs; all four ports ACTIVE/LINK_UP
ip -br addr         # 192.168.0.12/.13 (NCCL wire) + 192.168.9.12/.13 (TP wire)
                    # + 192.168.1.12/.13 (mgmt)
cat /sys/class/net/enP2p1s0f1np1/mtu      # 9000 on the TP wire
```
Pass: the `rocep1s0f1` port carrying `192.168.0.x` AND `roceP2p1s0f1` carrying
`192.168.9.x` are both ACTIVE on both sides.

> **TP wire (verified 2026-09-02 on ca1070wk30007/008):** the second 200G cable is
> `enP2p1s0f1np1` / `roceP2p1s0f1` / `192.168.9.12-.13/30`, MTU 9000, IPv4 GID
> present (high index — the fallback scan picks it up; `PULSAR_TP_RDMA_GID_INDEX`
> not needed). It is **separate from the NCCL wire** (`enp1s0f1np1` /
> `rocep1s0f1` / `192.168.0.x`), so a co-tenant run dials the 9.x address and
> pins `PULSAR_TP_RDMA_DEV=roceP2p1s0f1` on BOTH ranks (tp-pair-bringup.sh
> threads it through every leg) and never touches the wire vLLM/NCCL uses.
> `rocep1s0f0` and `roceP2p1s0f0` are LINK_UP but carry no IP — ignore them.

## 2. Cross-host transport over TCP (fail-safe first, ~2 min)

```sh
# Box A (leader):
./tests/tp_transport_test remote-leader 0.0.0.0 5588
# Box B (worker):
./tests/tp_transport_test remote-worker <BOX_A_NIC_IP> 5588
```
Connection uses the first usable RDMA-capable pair; to FORCE the TCP path for
this baseline, unset verbs availability is not practical — instead treat step 2
as "whatever transport auto-selects; assert which one" and note it. Pass: both
sides print `transport=tcp|rdma` and `... ok`.

## 3. Cross-host transport over real RDMA (the point of the slice) — ~2 min

```sh
PULSAR_TP_EXPECT_RDMA=1 ./tests/tp_transport_test remote-leader 0.0.0.0 5589   # Box A
PULSAR_TP_EXPECT_RDMA=1 ./tests/tp_transport_test remote-worker <BOX_A_IP> 5589  # Box B
```
Pass: both print `transport=rdma` + `... ok` (hello, attach, gate/batch/big,
lockstep, all frames over verbs). Any `transport=tcp` here is a FAIL — record
the `rdma link`/`ibv_devinfo` state and continue anyway.

## 4. GPU-visible slab + RDMA registration (the GB10 unified-memory thesis) — ~3 min

```sh
./tests/tp_slab_gpu_probe leader 0.0.0.0 5599     # Box A
PULSAR_TP_EXPECT_RDMA=1 ./tests/tp_slab_gpu_probe worker <BOX_A_IP> 5599  # Box B
```
Pass: both print `slab attached, GPU-visible (...), mr=registered` and
`tp_slab_probe: rank N ok (GPU slab: gates, batch, big; RDMA)`.
**This is the single most important number for the port**: if `cudaMallocManaged`
memory registers with `ibv_reg_mr` and moves 16 KiB partials over RoCE, the
plan's "register-once GPU-visible slabs, no D2H/H2D bounce" holds. If it fails,
capture the `cudaGetErrorString`/`strerror` now — it changes the slab design.

> **VERDICT (pair-verified 2026-09-02):** `cudaMallocManaged` does **not**
> register — `ibv_reg_mr(14091976) : Bad address` on both ranks (device pages
> aren't HCA-pinnable). The host-pinned path (`malloc` + `cudaHostRegister` +
> `ibv_reg_mr`) **does** register, and the full probe passes over real RDMA on
> the 9.x wire (`mr=registered`, gates/batch/big both ranks). The engine slab
> (4c) must use the **host-pinned registered slab** — GB10 unified memory still
> lets kernels write it directly (no D2H/H2D bounce). The probe defaults to
> host-pinned; `PULSAR_TP_SLAB_MANAGED=1` reproduces the failed managed attempt.
> **GPUDirect RDMA is NOT available on this platform** (tests/tp_dmabuf_probe,
> verified on-pair): `GPU_DIRECT_RDMA_SUPPORTED=0` and
> `GDR_WITH_VMM_SUPPORTED=0`, `cuMemCreate` rejects the CUDA-13 GPURDMA flag,
> and the exported dma-buf is refused by `ibv_reg_dmabuf_mr` (EINVAL) — the
> driver cannot produce GDR-capable memory no matter what. `nvidia-peermem`
> (stale DKMS left-over, not in `dkms status`, fails to insert) could not
> change that anyway. Host-pinned is therefore the final slab design; rerun
> `tests/tp_dmabuf_probe` after any nvidia driver update to recheck attr 110.

## 5. RDMA link bench — 1-link vs 2-link (supports the two-cable note) — ~5 min

Run the PLAN 102 probe's RDMA halves (not on production without a window):
```sh
# per link: ib_send_lat -d <hca> -s 16384 -n 5000 -F, then -d <hca2> ...
# record 16 KiB round-trip for each of the two cabled links; expect ~10-30 us.
# PULSAR_TP_RDMA_DEV=<hca> pins the transport's device per rank for the A/B.
```
Pass: numbers recorded into `pulsar-notes` (1-link vs 2-link; merged 2-link if
later implemented). This closes Phase 0a Q2 (~RTT) and the two-cable question.

> **RESULT (pair-verified 2026-09-02, perftest 6.28 both sides):**
> 16 KiB send latency, 5000 iters:
> - 9.x TP wire (`roceP2p1s0f1`, off vLLM): avg **4.64 µs**, typical 4.58, min
>   4.28, max 9.87, stdev 0.08, p99 5.07, p99.9 5.62.
> - 0.x NCCL wire (`rocep1s0f1`, co-tenant with vLLM): avg **4.77 µs**, typical
>   4.78, min 4.51, max 12.79, stdev 0.07, p99 4.94, p99.9 5.19.
> Both links are ~same latency (well under the transport's 10-30 µs budget) —
> the two-cable question is answered on latency: symmetric.  The 13.5 vs
> 24.5 GB/s single-vs-merged bandwidth difference (port doc) is throughput,
> not latency.  Requires MATCHING perftest on both ranks: box A (DOCA repo,
> 26.01.5-1) and box B (Ubuntu, 24.01.0) failed the negotiation handshake
> ("Failed to exchange negotiation parameters") until B was aligned up to the
> Mellanox DOCA 26.01.5-1 build (repo + key: `doca.list` signed-by
> `/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub`).

## 6. (After 4c) first live engine pair — prefill-TP before decode-TP

Per `docs/tensor-parallel-split.md` §Slice-4 sequencing: build `make cuda-spark`
with TP wired (4c), start leader `pulsar-server --tp-role leader --tp-port 5590`
and worker `... --tp-role worker --tp-peer <leader> --tp-port 5590` on the pair,
then run the prefill big-gate path first (plan §2.4: gates amortize per chunk),
grade reference-fidelity per the port rules.

## Rollback

Single-box behavior is untouched by design (`tp_role` defaults 0; the guard is
fail-loud until 4c). Restore = run without `--tp-role`. Nothing on this branch
touches the production instance.

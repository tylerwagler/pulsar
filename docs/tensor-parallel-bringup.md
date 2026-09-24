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

⚠ **A single-box two-process loopback is NOT a substitute (measured
2026-09-22).** Both ranks stand up fine over TCP loopback — `PULSAR_LOCK_FILE`
scopes the instance lock per rank — but with the served 85.90 GiB container the
box dies in MODEL LOAD on both sides (`rc=137`, no TP line ever printed):
earlyoom killed the leader with `shmem-rss 94087056 kB` while `file-rss` was
only 22 MB. So the model pages are SHARED (one physical copy) and the kill was
earlyoom's heuristic summing rss across processes — **not** a proven physical
shortfall, and not evidence that a smaller checkpoint is required. What makes it
fire is that the engine marks itself `oom_score_adj=1000`
(`engine_api.cpp`), i.e. first in line; two such processes on a 121 GB box is a
fight with an OOM daemon we do not own, so do not run it there. A loopback pair
proves nothing about TP and must not be read as a TP failure; the transport
ITSELF is exercised on one box by `tests/tp_mesh_test` (n=2 and n=3 over TCP
loopback, no model).**

## 0. Baseline — build + host tests (any box, ~2 min)

```sh
make tp-core-test tp-transport-test tp-sched-test     # all three must print ok
make tests/tp_slab_gpu_probe                          # compile-check at sm_120f
```
If a session needs a full engine build first: `make cuda-spark`.

> **Native probe build on the pair (verified 2026-09-23):** `tests/tp_slab_gpu_probe`
> links with the engine-generic `CUDA_LDLIBS` (`-lcublas -lcublasLt -lpng -ljpeg`)
> though it needs only `-lcudart`. A stock Spark image has no `libpng`/`libjpeg`
> dev packages, so `tools/tp-pair-bringup.sh`'s native build fails at link with
> `cannot find -lpng`. Fix: `sudo apt-get install -y libpng-dev libjpeg-dev` on
> both ranks (Ubuntu 24.04). The pair's nvcc is `/usr/local/cuda/bin/nvcc`
> (CUDA 13.3, sbsa) — not on the ssh PATH, but the Makefile's absolute
> `CUDA_HOME` default finds it.

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

> **Re-verified 2026-09-23 on the current n-rank transport (`work/tp-owned-moe`,
> 1c22835d):** both ranks `slab attached, GPU-visible (14091976 bytes,
> host-pinned (pulsar_tp_gpu)), mr=registered` and `tp_slab_probe: rank N ok
> (GPU slab: gates, batch, big; RDMA)` over `mlx5_3` / 192.168.9.x, gid index 5.
>
> **Co-tenant with vLLM is a hard block for this step, not a "go light" caution.**
> With the production vLLM worker resident (~101 GiB of the 121.6 GiB GB10
> unified pool), a fresh process cannot initialize a CUDA context at all:
> `cudaSetDevice(0)` returns `out of memory`, `cudaMemGetInfo` reports 0/0, and
> a 64 KiB `cudaHostRegister` (plain or Mapped), `cudaMalloc(1 MiB)` and
> `cudaMallocManaged(1 MiB)` all fail — independent of slab size. `free -g`
> showing ~10-13 GiB "available" is host-side accounting and does not mean
> GPU-allocatable memory. Steps 1/2/3/5 are verbs-only and run fine co-tenant
> on the 9.x wire; step 4 and everything after it require vLLM stopped.

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

> **Re-run 2026-09-23** (same perftest 6.28 both sides, `-d mlx5_3 -s 16384
> -n 5000 -F`, 9.x wire, gid index 5): avg **4.75 µs**, typical 4.69, min 4.44,
> stdev 0.04-0.06, p99 5.17, p99.9 5.51-5.66. Matches 2026-09-02 within noise.

## 6. (After 4c) first live engine pair — prefill-TP before decode-TP

Per `docs/tensor-parallel-split.md` §Slice-4 sequencing: build `make cuda-spark`
with TP wired (4c), then stand the group up. **Use the n-way mesh flags, not the
legacy two-rank pair** — the target is n Sparks:

```sh
# rank r of n, on each host (h0..hn-1 are the hosts' addresses in rank order):
pulsar -m <checkpoint> --tp-rank r --tp-nranks n \
       --tp-peers "h0:5590,h1:5590,..." --tp-port 5590 \
       -p "<prompt>" --temp 0 --nothink -n 32 --dump-logprobs rank$r.lp.json
```

The checkpoint is the full-fidelity MXFP4 build (`ElytronAI/DeepSeek-v4-Flash`,
168 GB; ~83 GiB per rank at n=2 with slice 4f) -- the one-box IQ2 build refuses
TP at layer 0 by design.  A worker rank needs no prompt: it runs the receive
loop (slice 4e, L238) and exits when the leader stops it, so its exit code is
the loop's verdict.  `tools/tp-pair-engine-grade.sh` runs exactly that over
ssh, in rank order, and grades it (`PULSAR_TP_HOSTS="h0 h1 [h2 ...]" ./tools/tp-pair-engine-grade.sh`;
`PULSAR_TP_DRYRUN=1` prints the plan). Its two legs:

- **LEG A — the one that matters, and it needs NO reference.** Every rank must
  assemble BYTE-IDENTICAL logits on every frame. That IS slice 4d's contract:
  each rank computes only its own vocab range, the group all-gathers, and every
  rank assembles the full vector — so all ranks must agree. Since 4e a worker
  never writes logprobs, so the check is in the engine (L243, protocol v11):
  the ack of every logits-producing frame (eval, batch decode, mixed batch,
  and the positive verdict of the CLI's one-frame `generate_speculative`)
  carries the worker's 64-bit digest of its assembled vector; the leader
  digests its own, refuses the frame by name on a mismatch (and marks the
  group failed), and prints the tally at close:
  `pulsar: tp: cross-rank logits identity: M/N worker frames matched`. The
  grading tool reads that line from rank 0's stderr and passes only when the
  line exists, N > 0 and M = N. A wrong range partition, a wrong gather or a
  wrong assembly shows up as a refusal naming the rank and the frame. This is
  the instrument that proves the vocab split ran the lane — the production
  lane, on every frame, not a side file.
- **DSpark stays ON in this run.**  Speculation on a pair is lockstep because
  the CLI's `generate_speculative` synchronizes the rng with the leader before
  its first draw (4e, 2026-09-23); every accept test and carry then follows the
  leader's by construction.  Byte-identical logprobs across ranks with the
  drafter live is therefore part of what LEG A proves.
- **LEG B — optional, tolerance only.** Set `PULSAR_TP_BASELINE=<single-box
  logprobs.json>` to grade rank 0 against a single-box run. TP is NOT
  byte-exact (partials are summed in a new order), so this is a report
  (`greedy-token disagreements`, `worst |logprob delta|`), never an equality
  assert — rule 3.

- **LEG C -- the pair's fidelity instrument (L240).**  For the MXFP4 artifact no
  single GB10 can produce LEG B's baseline, so the B300 reference is the grade:
  `PULSAR_TP_REF_DIR=<capture dir>` runs `tests/prefill_bitexact_gate
  --check-reference` THROUGH the group (rank 0 grades the story and code blobs
  at every recorded depth, ranks > 0 run the receive loop).  A gate joins a
  group through its environment -- `PULSAR_TP_RANK/NRANKS/PEERS/PORT`, read
  once at `gate_engine_open` (tests/gate_entry.h) -- so every gate is
  TP-capable without per-gate flags.  Pass the capture's documented outlier
  depths (`PULSAR_TP_REF_KNOWN_HIGH_STORY` etc.) as the battery's runner spec
  has them; the gate binary lives at `PULSAR_TP_REF_BIN` on each host.

Rule 3 still holds on one box: `cuda-reference-gate` must never be graded while
it prints SKIP -- since L240 the battery points `PULSAR_REF_DIR` at the staged
Vision-Exp capture by default, so a SKIP now means the override was set empty.

> **FIRST LIVE PAIR — VERDICT (2026-09-23, ca1070wk30007 = rank 0 / 30008 = rank 1,
> dev `70c02339` + the fixes on `work/tp-pair-live`, artifact
> `ElytronAI/DeepSeek-v4-Flash` rev `fc589c1a`, 48 shards = 167,979,453,688 B
> staged on both ranks; vLLM stopped):** the engine loads across both Sparks.
> Each rank staged **83.12 GiB** of owned tensors in 187 spans (~17.9 s;
> 73.31 GiB of peer-owned experts not resident), 43/43 layers grouped-CUTLASS
> MXFP4, `rank r/2 mesh connected (1 peers), transport=rdma` on `mlx5_3`
> (gid 5, 9.x wire), `TP rank r/2 armed (prefill big-gate)`, attention output
> groups [0,4)/[4,8) = heads [0,32)/[32,64) registered. Rank 0 generated 32
> greedy tokens for the script's default prompt — *"A pointer is a variable
> that stores a memory address, and it can be reassigned to point to different
> locations, whereas an array is a fixed-size block of"* (mean top-1 logprob
> −0.128); rank 1 ran the receive loop and reported `stopped by the leader`;
> both rc=0, zero warnings. Two runs produced byte-identical `rank0.lp.json`
> (75,015 B, sha256 `5d20a7b9…`). `tools/tp-pair-engine-grade.sh` now launches
> and grades this unattended (`PULSAR_TP_RDMA_DEV=mlx5_3`,
> `PULSAR_TP_ADDRS="192.168.9.12 192.168.9.13"`).
>
> Three defects stood between the runbook and this result, all fixed on the
> branch: `--tp-rank 0` was refused by both binaries (strictly-positive int
> parse); the grading script's relative workdir default broke its own second
> `cd`; and its rank launch stayed attached to the ssh channel, so ssh returned
> only when the engine exited — rank 0 sat in `accept()` and rank 1 was never
> launched. A missing rank stderr is now a scan failure, not "clean".
>
> **LEG A under slice 4e (resolved 2026-09-23 evening, L243).** The first live
> run's LEG A compared `rank<r>.lp.json` files, which a worker rank no longer
> writes (it runs the receive loop and is stopped by the leader before any
> head), so the leg could only fail. The check now lives in the engine: every
> logits-producing frame's ack carries the worker's digest of its assembled
> logits, the leader compares it with its own on every frame, refuses by name
> on a mismatch, and prints `cross-rank logits identity: M/N worker frames
> matched` at close; the grading tool reads that line. Protocol v11 — both
> ranks must run the same binary (the hello refuses a version mismatch). Not
> yet run on the pair: the next pair session (2026-09-24) is the first to
> exercise it. What N will read: the CLI with DSpark on (the default, and the
> grading tool's run) issues its whole generation as ONE
> `generate_speculative` frame, whose positive verdict carries the digest of
> the run's last row — so expect `1/1`; with `--no-dspark` every token is an
> eval frame and N is the token count; the served lane's mixed-batch steps are
> checked one frame per step. The speculative ROUND frames the server uses
> (round begin/end, redraft) do not carry a digest yet — that is the next
> slice of this instrument.

> **Pair on dev `ece8a459` (2026-09-24): graded PASS, first numbers.** LEG A via
> the engine's tally (L243): 32/32 frames byte-identical. 128-token runs, ctx
> 4096, 22-token prompt: DSpark on **15.57 t/s** generation (57 frames, ~2.2
> tokens/round); drafter off, greedy **10.75 t/s** (116 frames); prefill ~40 t/s
> (prompt too short to mean anything). The greedy and DSpark outputs are
> byte-identical (592 B) -- speculation is lossless on the pair. Against the
> one-box no-spec ceiling (24.2 t/s) decode TP costs ~2.3x today: every layer's
> attention gather and FFN big gate stage through host memory
> (`tp_attn_gather_low`: tensor read -> all-gather -> tensor write); that is the
> collective-latency work L241 lists (4g-2). One CLI gap closed on the way:
> `--no-dspark --temp 0` dispatched to the raw whole-graph path, which builds
> its own graph with no transport and refused at layer 0 with a message about
> the graph's owned groups; a TP engine now rides the session lane for every
> one-shot generation and `pulsar_engine_generate_argmax` refuses under TP by
> name. `CUDA host registration skipped: operation not supported` at load is
> the mmap'd checkpoint not being HCA/GPU-registrable on GB10 (the same class
> as the step-4 managed-memory verdict); the loader falls back to its local
> copy path and the run is unaffected.

> **LEG C -- REFERENCE GATE PASS THROUGH THE PAIR (2026-09-24, dev `ece8a459` +
> `work/tp-raw-path-refuse`, capture = the Vision-Exp B300 blobs from
> pulsar-notes `gate-baseline/ref-vexp`, story.ref.bin 27fec79e / code.ref.bin
> 1b987bf8, tol 1e-4, anchors story known-high 512,30464 / known-flip 30464,
> code known-high 3840).** Story: 2048 KL 1.45e-6, 4096 4.38e-6, 4102 1.61e-7,
> 6144 5.34e-9, 30464 **top-1 MATCH** KL 1.89e-2 (known-high, informational);
> code: 512 4.34e-7, 2048 1.53e-8, 3840 1.71e-2 (known-high). Top-1 matched at
> every enforced depth, both blobs PASS, rank 1 `TP worker loop ended rc=0`.
> Against the served one-box IQ2 budgets (3.16e-6 / 6.91e-6 / 5.33e-6 / 7.30e-8 /
> 0.349; 1.88e-6 / 5.18e-7 / 0.191) the pair grades closer to the source at
> EVERY depth, and the IQ2's 30464 flip does not flip here -- the gate says
> "drop it from --known-flip" for this artifact's anchors. Prefill at real shape
> the same morning: the ~35k-token story prompt through 9 chunks at 368.57 t/s,
> 16/16 needle assignments recalled, 17/17 frames byte-identical.
>
> Two things the run needed that the handoff did not say: the gate reads the
> reference blob on EVERY rank before opening the engine ("cannot read
> reference blob" on a worker without it), so `PULSAR_TP_REF_DIR` must be
> readable on every host; and `tests/prefill_bitexact_gate`'s environment
> scrub kept the four TP-group variables but not `PULSAR_TP_RDMA_DEV` /
> `PULSAR_TP_RDMA_GID_INDEX`, so the gate's rank auto-picked the IP-less
> `mlx5_0` (link-local GID, no peer) while the engine beside it rode
> `mlx5_3`; both are on the keep-list now (transport addressing, not numerics).

## 7. Arrival skew — clock lock + the pinning/spin A/B (`work/tp-skew`, L241 4g-2 follow-up)

Decode does 86 row-lane exchanges per token; each costs 21-43 µs against a
~5 µs wire (step 5), the same on both ranks — so most of it is the two ranks
ARRIVING at different times, not the wire.  `work/tp-skew` attacks the host
side of that (small-message latency traced to CPU idle-state wake, LPI exit
231/433 µs) in three commits, plus a fourth on top that measures it and never
lands:

| commit | what it changes | how it announces itself |
|---|---|---|
| pinning | row-lane proxy + launch thread on two discovered big cores; proxy timerslack 1 ns | `pulsar-tp: rank R proxy pinned to cpuN, launch thread cpuM reserved (cpu_capacity V/MAX, ...)`, then `pulsar-tp: rank R launch thread pinned to cpuM` — or `core pinning skipped: ...` |
| proxy spin | the proxy backs off to `usleep(50)` only after 100 ms without an exchange (was 2 ms) | the instrument counts backoffs |
| worker wait | the worker spins 5 ms on the control socket before `poll(-1)` | the instrument counts `poll(-1)` fallbacks |
| **TEMPORARY INSTRUMENT** (tip) | per-exchange timestamps in the proxy; histogram printed per rank at close | `pulsar-tp: TEMPORARY tp-skew instrument rank R: ...` |

**Lock both GPUs to one clock first.**  The two hosts do not run the same
clock (one is thermally capped), and a slower GPU is a late rank by
construction — pinning cannot fix that.  On EACH host:

```sh
nvidia-smi -q -d SUPPORTED_CLOCKS | head -30          # what -lgc may take
# during a decode run, watch both hosts; the capped one sets the target:
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_event_reasons.active \
           --format=csv -lms 500
sudo nvidia-smi -lgc <mhz>,<mhz>   # SAME value on both hosts, at or below the capped host's sustained clock
nvidia-smi --query-gpu=clocks.sm --format=csv   # verify it holds UNDER LOAD (watch during a run)
# ... the A/B ...
sudo nvidia-smi -rgc               # RESTORE on both hosts when done -- and after any abort
```

If `-lgc` is refused on GB10 ("not supported" / "insufficient permissions"),
say so in the row, record both hosts' `clocks.sm` during every run, and run
unlocked — the A/B is then only valid while the two clocks read alike.

**Binaries** — three, on each host, from their own worktrees (never reset a
tree someone is using); the engine prints its build sha and `PULSAR_TP_SHA`
asserts it:

| name | rev | what |
|---|---|---|
| `dev` | `05d162d3` | the baseline |
| `skew` | `origin/work/tp-skew~1` | the three commits, WITHOUT the instrument — every timing comes from here |
| `instr` | `origin/work/tp-skew` | `skew` + the instrument — histogram only, never a t/s |

Build each with the host's usual `make cuda-spark` (zero warnings), copy the
binary to `~/pulsar-<name>` and its short sha to `~/pulsar-<name>.sha`.

**The A/B** — greedy (`--no-dspark`) and DSpark (default), 128 tokens, ctx
4096, the grading tool's default prompt, 3 reps, dev and skew INTERLEAVED so
drift hits both.  The comparison is ALWAYS against the `dev` arm measured in
the same session, under the same clock lock, never against a number from
another day.  For scale only: dev `05d162d3` on the pair measured greedy
~27.2-27.45 t/s and DSpark ~38 t/s through the CLI (37.95 through
pulsar-server) (pulsar-notes OPEN-REGISTER, 2026-09-24 15:25 and 21:05).  The
15.57 / 10.75 t/s above were the first pair run, before the 4g-2 series.

```sh
export PULSAR_TP_HOSTS="ca1070wk30007 ca1070wk30008" PULSAR_TP_ADDRS="192.168.9.12 192.168.9.13"
export PULSAR_TP_RDMA_DEV=mlx5_3 PULSAR_TP_TOKENS=128 PULSAR_TP_CTX=4096
H=($PULSAR_TP_HOSTS); mkdir -p ab
for rep in 1 2 3; do for arm in dev skew; do for mode in spec greedy; do
  extra=; [ $mode = greedy ] && extra=--no-dspark
  PULSAR_TP_BIN="\$HOME/pulsar-$arm" PULSAR_TP_EXTRA_ARGS="$extra" \
  PULSAR_TP_SHA=$(ssh ${H[0]} cat pulsar-$arm.sha) \
    ./tools/tp-pair-engine-grade.sh > ab/$arm-$mode-$rep.log 2>&1; echo "$arm $mode $rep rc=$?"
  for r in 0 1; do ssh ${H[$r]} cat tp-pair-grade/rank$r.err > ab/$arm-$mode-$rep.rank$r.err; done
done; done; done
grep -H "generation:" ab/*.rank0.err                               # prefill: X t/s, generation: Y t/s
grep -H "pulsar-tp: rank .* pinned\|pinning skipped" ab/skew-*.err  # the pin lines, BOTH ranks
```

Every run must grade PASS (the LEG A tally M/N with M = N), and greedy and
DSpark outputs stay byte-identical: the branch moves no numeric path.  Report
per arm and mode the three generation t/s and their median; a win is a median
outside the other arm's range.  A skew run whose rank*.err lacks the two pin
lines did not run the lane being measured.  Then ONE rep of each mode on
`instr` (`PULSAR_TP_BIN="\$HOME/pulsar-instr"`) and keep both ranks' block:

```sh
grep -A8 "TEMPORARY tp-skew instrument" ab/instr-*.rank*.err
```

Reading it: `post->arrive` is how long this rank's proxy waited, after posting
its rows, for the peer's rows (the wire is ~5 µs of it); `peer-first` counts
exchanges whose peer rows were ALREADY there at the first poll after this
rank's post — THIS rank arrived late.  The late rank shows the large
`peer-first` and the early rank the fat histogram tail; equal tails and few
`peer-first` on both ranks mean a common cause (the proxy or the wire), not
skew.  `after-backoff` > 0 inside a stream means the proxy was asleep when an
exchange came; worker `poll(-1) fallbacks` near its command count means the
5 ms spin is shorter than the leader's turnaround.  Restore the clocks
(`sudo nvidia-smi -rgc`, both hosts) before leaving.

## Rollback

Single-box behavior is untouched by design (`tp_role` defaults 0; the guard is
fail-loud until 4c). Restore = run without `--tp-role`. Nothing on this branch
touches the production instance.

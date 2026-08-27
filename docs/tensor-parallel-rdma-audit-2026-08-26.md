# TP RDMA/scheduler static audit — 2026-08-26

Static review of the parts of `src/tp/pulsar_tp.cpp` that have never run at
runtime (no GPU / no RDMA device on the dev box): the RDMA gate path, the
scheduler thread model, and the exchange-counter/schedule consistency. Same
shape as the 2026-08-13 engine review — file:line findings with a disposition
(FIXED / RUNBOOK / NOTE). Re-verify at runtime on the pair; this is the
"watch these first" list for the bring-up session.

## Findings

### F1 — [HIGH, FIXED] RDMA device auto-pick can pick the wrong cable
`tp_rdma_open` auto-picked the first device with an ACTIVE port per rank.
With **two** live cables (the pair has all four CX-7 ports LINK_UP), the two
ranks could each pick a different link and never connect.
**Fix:** devices are now name-sorted for a deterministic ordinal on both
hosts, and a multi-HCA box prints a "pin the SAME cable on both ranks with
PULSAR_TP_RDMA_DEV=<hca>" warning when the env is unset. The runbook step 3
must set `PULSAR_TP_RDMA_DEV` identically on both ranks (physical cable
identity cannot be inferred from inside the process).

### F2 — [HIGH, FIXED] GID scan was Mac/Thunderbolt-only
The GID scan accepted only IPv4-mapped `::ffff:` GIDs — upstream's two-Mac
assumption. Linux RoCEv2 (CX-7) commonly exposes the usable GID at a
non-zero index (the pair's NCCL notes use gid index 3). A miss made
`tp_rdma_open` return "no IPv4-mapped GID" and `create()` fail **the whole
pair** (not a graceful TCP demotion — see F3).
**Fix:** `PULSAR_TP_RDMA_GID_INDEX=<n>` override, plus a fallback scan that
accepts the first non-link-local (non-`fe80`) GID when no IPv4-mapped one
exists. Runtime confirmation on the pair still required (expect index 3 on
the CX-7).

### F3 — [MEDIUM, RUNBOOK] RDMA bring-up failure is fatal, not TCP
Once the hello advertises `rdma_ok`, a later `tp_rdma_open` failure aborts
`create()` (by design — silent TCP after advertising verbs would be a
mystery). Practical trap: a verbs-present but misconfigured port (no active
port, GID miss) hard-fails. The error already lists tried devices/states.
Bring-up must pre-check `rdma link` ACTIVE on the exact device named by
`PULSAR_TP_RDMA_DEV`, and keep `PULSAR_TP_EXPECT_RDMA` off until F1/F2 are
confirmed.

### F4 — [MEDIUM, NOTE] the >16 KiB chunked-gate path is latent
At n_embd 4096, a gate partial is exactly one 16 KiB message, so the
"2 chunked messages" branch (final-chunk-carries-seq wr_id) never fires for
our model. Registration also REJECTS vec_bytes > 32 KiB (2 x MAX_MSG), so a
future wider-embd model (n_embd > 8192) would fail RDMA at register time.
Not a bug today; flag for any wider-embd model and to exercise the chunked
gate at some point.

### F5 — [LOW, NOTE] post_lock held across the full CQ-drain wait
`gate_exchange` holds `post_lock` while it blocks (up to the timeout) polling
completions. Correct for the engine's single gate thread (port rule #1); a
second gate thread would serialize/block. Documented, not changed.

### F6 — [LOW, NOTE] teardown skips dlclose/mutex-destroy
`tp_rdma_close` frees QP/MR/CQ/PD/ctx but not the dlopen handle or the
mutex. One-time per process; intentionally left (dlclose on libibverbs state
can be fragile). Null-guarded correctly — the TCP-only destructor path never
calls into a zeroed api (verified).

### F7 — [INFO, VERIFIED] exchange-counter/schedule consistency
The scheduler's monotonic `e` with the DS identity schedule satisfies the
RDMA gauge (`slot == tp_gate_slot(e)` → `(e-1) % 86`): token 2 starts at
e=87 → slot 0 again, so `tp_gate_slot` and the per-gate slot stay in lock.
Any engine caller must hand the scheduler-aligned `e` — the assert fires
loudly if not. Covered by tp_sched_test beyond e=86.

### F8 — [INFO, PAIR-CONFIRM] relies on in-order UC completions
The recv watermark + in-slot reuse across tokens assumes in-order QP
completions (sound for RC/UC). The pair stress pass should run the remote
harness at a high gate count and confirm the watermark never regresses.

## Disposition
FIXED: F1, F2 (host-compiled; runtime to confirm on the pair).
RUNBOOK: F3 (+ runbook step 3 pinning).
NOTE/LATER: F4, F5, F6. VERIFIED: F7. CONFIRM-ON-PAIR: F8.

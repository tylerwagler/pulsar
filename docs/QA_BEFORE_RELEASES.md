# QA Before Releases

This is the release gate for Pulsar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: CUDA graph inference,
disk KV cache, server APIs, and the agent TUI/tool state
machine.

This fork is CUDA-only and targets the DGX Spark (GB10, ~128 GB unified
memory); all release testing happens on that host.  Do not run multiple huge
model processes at the same time.  Record the commit, GGUF file, context size,
and any non-default flags for every manual run.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the release binary:
  `make clean && make cuda-spark` (this builds **only `pulsar-server`** — the
  shipped release).  Build the development tools by name with the same arch:
  `make pulsar pulsar-agent pulsar-bench pulsar-eval CUDA_ARCH=sm_120f`.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./pulsar --help`, `./pulsar-server --help`, and `./pulsar-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test` (also the first entry of `make gates`, as `unit-test-gate`,
  since 2026-09-03 -- a battery run covers it, as `pulsar_test --gating` since
  2026-10-14: every test but the informational `logprob-vectors`).
- Run the CUDA smoke regression:
  `make cuda-regression`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./pulsar_test --logprob-vectors`
  (the local golden-vector leg was retired 2026-09-02, L156; the reference
  gate `make cuda-reference-gate` with `PULSAR_REF_DIR` set is the fidelity
  oracle).
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./pulsar_test --server`.
- Build and run the agent unit tests: `make pulsar_agent_test CUDA_ARCH=sm_120f
  && ./pulsar_agent_test` (they are not part of `make test`).
- Run `./pulsar-eval --self-test-extractors`.

### CUDA gates ledger

**Run them with one command:**

```
make gates FRONTIER_MODEL=/srv/models/<checkpoint-dir>
```

It runs every gate below, continues past failures so one break does not hide
the rest, prints a PASS/FAIL summary, and exits non-zero if any failed.

> ⚠ **Do not hand-run these individually and tick them off.** That is what this
> checklist used to say, and on 2026-08-14 a sweep found the prefill gate
> rotted (its baseline ref predated type-43, so it could not run *at all*), two
> attention gates uncompilable since an API parameter was removed that morning,
> and a real production bug — fp16 attention breaking mixed-batch prefill —
> that had shipped six days earlier.  **None of it was visible to the product
> build.**  A gate nobody invokes is a gate that silently stops compiling.
>
> Two rules that follow:
> * After removing or changing any engine API parameter, run `make gates`.
>   Gates are separate targets; the product build will not tell you.
> * Re-baseline `PREFILL_BASELINE_REF` whenever a numerics change **ships**, and
>   only then.  Anchoring it before a deliberate precision change makes the gate
>   fail forever, and a gate that always fails is one you learn to ignore.

The individual targets, all needing the GB10 and (except the first two) the
model.  Record pass/fail against the release commit:

- `make cuda-regression` — modelless kernel smokes.
- `make cuda-attn-gates` — fp16 attention kernel oracle (dense, compressed,
  indexed, decode-batch, one-row), banked cross-session KV-leak isolation
  (modelless).
- `make cuda-prefill-gate` — full-vocab frontier byte-compare against the
  `PREFILL_BASELINE_REF` blob.  **The baseline ref must postdate type-43**
  (the aligned-MMQ pre-store): a pre-type-43 baseline build cannot load the
  shipped artifact and the gate cannot certify the release.  Rebuild the blob
  with `make cuda-prefill-gate-baseline` after bumping the ref.
- `make cuda-evict-restore-gate` — bank evict/restore bit-identity.
- `make cuda-rewind-gate` — compressor frontier position-truth across ghost
  rewinds (L120): counters equal pos/ratio at every mod-4 residue and the
  ratio-128 boundary, plus the value leg (a ghost rewind must not contaminate
  re-emitted comp rows) and the ratio-128 undo walk (L124).
- `make cuda-mseq-rewind-gate` — a rewind must clamp BOTH frontier
  representations. L120 clamped the scalars; the served path validates the next
  multiseq step against the PER-BANK slots (ms_n_comp[bank]), which rewind did
  not touch, so a rewound bank decoded again with no bank switch-away between
  was rejected with production's own "frontier not position-true". Reproduces
  that deterministically and asserts both copies agree.
- `make cuda-seam-gate` — token-boundary seams keep the live KV (L115).
  Discovers a seam pair from the model's own vocabulary, then pins the byte
  walk, the sync rescue, the shorter-echo shape, and a fork across seams.
  Needs `PULSAR_MSEQ_BANKS>=2` for its fork leg (the target sets it).
- `make cuda-frontier-gate`, `make cuda-multiseq-gate`,
  `make cuda-multiseq-gate-nodspark` — multiseq isolation and throughput.
- `make cuda-bank-spec-gate`, `make cuda-accounting-gate`,
  `make cuda-algo-stability-gate` — bank/spec interaction, admission
  accounting exactness, cuBLASLt M-independence.
- `make cuda-mixed-prefill-gate`, `make cuda-mixed-neutrality-gate` — mixed
  decode+prefill batching neutrality.
- `make cuda-spec-sampling-gate` — speculative-sampling chi-square exactness.

**The battery runs the production drafters (L305, 2026-10-21).**  Qwen's hosted model and every Qwen gate default
(`QWEN_GATE_MODEL`, `DFLASH2_GATE_MODEL`) are the served pack, `/srv/models/qwen38fn-dflash2` (DFlash2, the EXL3 K5
head, no MTP module); Nemotron's drafter gate is LiLiCorr's, DeepSeek's is DSpark's.  **MTP is a dormant lane: not
release-blocking, tested on MTP-path changes only.**  `mtp-spec-gate` is out of `make gates` (its runner entry is
gone; the target builds and runs by hand on `MTP_GATE_MODEL`, the Nemotron mini), and `make gates-dev` runs it when a
touched path is an MTP one (`*mtp*`, the family MTP bindings in `family_qwen`, `nemotron_h_family`, `step_driver`).
`sync-budget-gate` runs each hosted model's own drafter, so its Qwen leg is DFlash2's.  The one Qwen leg left on the
MTP pack is `tp-plan-gate`'s (manual; `TP_QWEN_PLAN_MODEL`): the served pack has no TP plan (a block drafter under
TP and an EXL3 head refuse) and Qwen is not a pair model.

### Battery wall time

The one home for what `make gates` costs and why (Phase 5 #39: back under ~35 min).  The battery prints
its own table at the end -- `seconds per gate` in the runner summary, and each section against its budget
below in the GATE SUMMARY -- so every battery log is a fresh measurement; this section records the levers.

**The budget (L317, 2026-10-21).**  `make gates` ends with each section's seconds against this table
(`tests/battery_budget.py` reads it from here -- the one place the budgets are written) and flags, never
fails, a section over its budget or a section with no row.  A row that adds battery time says so: it moves
its section's budget here, in its own commit, and its ledger row says why.  Sections: each serial target;
the runner per hosted model (`cuda-runner-gate [<family>]`, its engine opens included); the live legs per
model (`live-legs [<model>]`, its server loads included); the host-only gates, which overlap the runner, as
their longest; and the suite total (the recipe's wall, the prerequisite build outside it, ~100 s).
*Provisional:* the section rows are estimates from #422; the first battery at the L317 sha re-sets them from its
own table.  The total is the target (L317: about 50 min with the live legs), not the sum of the rows.

<!-- battery-budget -->
| section | budget s | holds |
|---|---:|---|
| `unit-test-gate` | 175 | `pulsar_test --gating`: three model loads |
| `agent-test-gate` | 10 | host |
| `cuda-regression` | 70 | ten model-free kernel gates, concurrent |
| `cuda-kv-rows-pack-gate` | 10 | |
| `cuda-sampler-device-gate` | 30 | |
| `cuda-chat-smoke-gate` | 35 | the CLI's own load |
| `cuda-attn-gates` | 45 | |
| `cuda-attn-pack-gate` | 5 | |
| `indexer-hadamard-kernel-check` | 5 | |
| `cuda-prefill-gate-cutlass-mxfp4` | 5 | skips loudly off-box |
| `qwen-family-gate-device` | 60 | zero-weight containers |
| `state-kinds-gate` | 5 | |
| `vision-qwen-tower-gate` | 10 | |
| `tessera-kernel-gate` | 10 | |
| `blockdraft-gate` | 5 | |
| `nvfp4-rows-gate` | 10 | |
| `lilicorr-head-gate` | 10 | |
| `cuda-runner-gate [DeepSeek V4]` | 1150 | the whole table, 2 engine opens |
| `cuda-runner-gate [Qwen4-exp]` | 720 | the whole table where its needs allow, 2 opens; block-spec (DFlash2) |
| `cuda-runner-gate [Nemotron-H]` | 300 | block-spec (LiLiCorr), scratch-poison, bank-adopt; 1 open |
| `live-legs [deepseek]` | 600 | constrained, server-live, abandon: warm + hooks instances |
| `live-legs [qwen]` | 1500 | the stall check, abandon, constrained, server-live, prefix-turns x2, preempt: warm + hooks + fresh |
| `live-legs [nemotron]` | 240 | opencode-turns: warm + fresh |
| `host-only gates (longest)` | 90 | overlap the runner |
| `suite total` | 3000 | the target: <= ~50 min, the live legs included |
<!-- /battery-budget -->

**2026-10-21 (L317, the battery diet; #389 / #422 before it: suite 3,329 / 3,219 s + ~45 min of live
legs, ~1 h 40 min).**  No leg lost its owner: each runs in the battery, or on a gates-dev path trigger, or
on an engine it now shares.

| leg | before | now |
|---|---|---|
| constrained-live, server-live, server-abandon (DeepSeek, Qwen) | own servers after `make gates`, 4 loads a model | `live-legs`, the last section of `make gates`: the model's shared server; constrained's isolation on its hooks instance |
| L312 stall check | `~/flare-work` script + `~/l312w/tbt2.py`, own server (262k ctx, 16 banks, no disk) | `tests/stall_live.py`, Qwen's shared server, first (production's shape: 262k ctx, 16 banks, 64 GB disk tier) |
| prefix-turns reply / side (Qwen) | 2 servers each (warm, cold) | shared server; the reference on Qwen's one fresh instance (0 cached tokens asserted); L316's instrument (thinking echoed, one prompt asserted) |
| preempt-exact (Qwen) | 4 servers (probe, count, preempt, alone) | probe + counts on the shared server, the preemption on the hooks instance (budget override), A alone on the fresh instance |
| opencode-turns (Nemotron) | 2 servers | shared + fresh (Nemotron has one leg) |
| lilicorr-spec-gate, dflash2-spec-gate | own process + load each, 256 tokens (389 + 307 s) | runner `block-spec-gate` on the hosted Nemotron (LiLiCorr) and Qwen (DFlash2), 128 tokens; the make targets at 256 are gates-dev's drafter trigger |
| scratch-poison LiLiCorr leg | by hand / gates-dev | runner `scratch-poison-gate` on the hosted Nemotron |
| bank-adopt-gate-{ds,qwen,nemotron} | by hand | runner `bank-adopt-gate` on all three hosted models (Nemotron's: the LiLiCorr pack, was the mini) |
| cuda-chunk-neutrality-gate (382 + 212 s) | 16 schedules + X, both families | battery: A + one schedule per class (grid, endpoint, finishing, disk), seeded by the sha; the past-ring class (K, M, P: the 4,600-step decode) and the rest: `cuda-chunk-neutrality-gate-full`, gates-dev on prefill / chunk / kv_state / checkpoint / session / segment-store paths |
| row-neutrality -deep / -deeper, algo-stability -deep | battery, both families | gates-dev entries of the runner (attention and kernel-dispatch paths); the battery keeps the short probe and the short sweep |

- *The runner hosts three models, one at a time* (each closed before the next opens): DeepSeek, Qwen
  (`qwen38fn-dflash2`, the production pack), Nemotron (`nemotron35-k4-lilicorr-pulsar`).  `--gates=a,b`
  after a `--model` restricts a hosted model to its own legs; a runner entry marked DEV runs only when
  `--only` names it (gates-dev), and every run lists them by name.  A hosted model none of whose entries the
  selection takes is not opened.
- *The live legs* run at each model's production flags (`tests/live_legs.py` MODELS, from `~/pulsar-serve/*.env`:
  all PULSAR_MSEQ_BANKS=16; DeepSeek 1M ctx; Qwen 262k ctx, 64 GB disk tier; Nemotron 1M ctx, LiLiCorr, 64 GB disk
  tier).  `tests/live_legs.py`, `make live-legs`, `LIVE_LEGS=a,b` for a subset: per model the
  shared server, then one instance with every server-wide test hook its legs need (only if one does), then
  one fresh instance for every reuse == cold reference (only if one does) -- never two engines at once.
  Each leg keeps its checks and its PASS/FAIL; each phase names the instance it ran on; each leg's
  conversations open with text no other leg sends, and every fresh-instance reference must report 0 cached
  prompt tokens.  Each script still runs alone by its old command line, on instances of its own.
- *The mutation spot check* (`make battery-mutants`, a GPU job): tests/gates_runner linked against one
  mutated TU per moved leg (`tests/runner_mutants.py`, plus L307's scratch-poison mutant), each run with the
  selection that must catch it.

**Before (2026-10-13, 1b33c1d4, sparky `~/qland2-battery-1013-2134`):** 3,324 s wall, of which 3,240 s
suite: the runner 2,629 s (52 sub-gates, 9 engine opens), the serial GPU targets in front of it 611 s.
The slowest sub-gates:

| s | gate | where it went |
|---:|---|---|
| 392 | cuda-decode-reference-gate-story | a model reload, then 48 prefills from 0 (6 depths x 4 widths x wide/serial; 380k tokens) |
| 380 | cuda-chunk-neutrality-gate | 17 schedules from 0; schedule K's 4,600 single-row decode steps are ~230 s of it |
| 262 | cuda-regression (make target) | ten model-free kernel gates, one after another |
| 231 | unit-test-gate (make target) | the pulsar_test compile (~40 s) and FIVE model loads (shared, logprob-vectors, tensor-equivalence x2, the image tests re-opening the shared one) |
| 225 | cuda-chunk-neutrality-gate [Qwen4-exp] | as above on the hosted Qwen |
| 171 / 150 | cuda-spec-sampling-gate [Qwen4-exp] / DeepSeek | 1,250 trajectories per mode (L163) |
| 107 | cuda-accounting-gate | prefill to 64k in three levels |
| 78 | cuda-head-graph-replay-gate | three 120-step runs (eager, graphs, eager) |
| 70 | cuda-prefill-gate | the byte gate's five depths, cold |
| 67 | cuda-decode-reference-gate-code | as the story gate, 3 depths |
| 66 / 43 | cuda-row-neutrality-gate-deeper [Qwen4-exp] / DeepSeek | ~10k-token probe |
| 65 | qwen-family-gate-device (make target) | seven zero-weight containers, the loader refusals, one device session |
| 49 / 46 | session-contract-gate [Qwen4-exp] / DeepSeek | |
| 47, 30, 30 | cuda-dspark-batch-gate-wide / -depth4 / -depth1 | each a full model reload for its draft depth (~22 s of each) |
| 45 / 41 | cuda-reference-gate-story [Qwen4-exp] / DeepSeek | |
| 42, 29 | cuda-multiseq-gate-nodspark [Qwen4-exp] / DeepSeek | the drafter-off engine's open |
| 20 | cuda-tp-head-split-gate | its own model load (drafter off), and one more for the gates after it |

**2026-10-14 (work/battery-trim).  No release-blocking gate left the battery; one informational test did
(`pulsar_test --logprob-vectors`, below):**
- *Engine opens 9 -> 4* (two per hosted model).  The draft depth is a session-start field the broker sets
  on the live engine (`spec_depth_shareable`, tests/gates_runner.cpp), so the three dspark depth legs run on
  the default engine; `cuda-tp-head-split-gate` moved from the end of the default group to the end of the
  drafter-off group whose configuration it has, so it no longer costs its own load AND the default
  engine's reload for the decode-reference gates.
- *Decode-reference gates: one cold prefill per depth.*  Every other run restores the grid checkpoint it
  left and prefills the rest -- the cold computation (L264) -- announced per depth
  (`[resume] depth D: ... resumed from grid checkpoint G`) and ASSERTED: the depth's widest wide run is
  computed on the cold prefill and again over a resumed one, and must be byte-identical (it is, at
  every depth of both blobs).
- *cuda-regression runs its ten model-free gates concurrently* (`make -O`: each one's output stays together).
  They are host-bound -- measured alone 2026-10-14: exl3_moe_prefill 114 s (113 s of it single-threaded user
  time; its W_hat decode now runs on 16 threads -- the same bytes), qsa_attn 46, qwen_s4 36, exl3_moe_epilogue
  30, exl3_dense 25, the other five 7 together; 258 s in series.  None grades a time.
- *unit-test-gate runs `pulsar_test --gating`*: every test but the informational `logprob-vectors` (it
  can never fail the suite; `make test` still runs it), and the image tests moved in front of the two
  own-engine tests so they reuse the shared engine.  Five model loads -> three.  pulsar_test, pulsar,
  pulsar_agent_test and the cuda-regression binaries are `gates` prerequisites, built in the one parallel
  make before the timed loop.
- **Measured per piece** (2026-10-14, a targeted run on sparky, not a battery: the changed runner gates on
  both hosted models, cuda-regression and unit-test-gate): decode-reference story 392 -> 93 s, code 67 -> 31;
  dspark-batch depth1 30 -> 8, depth4 30 -> 9, wide 47 -> 25; tp-head-split 20 -> 0; cuda-regression 262 -> 77;
  unit-test-gate 231 -> 167 (its compile now in the prerequisite build); 4 engine opens for two hosted models.
- **Predicted battery: ~2,670 s (44 min)**, from 3,324: the runner ~2,210 s, the serial front ~370 s, the
  build ~90 s.  The first battery after this lands measures it (its GATE SUMMARY).

**Not cut, and what each would take** (each removes coverage or changes a gate's premise -- a decision,
not hygiene):
- cuda-chunk-neutrality-gate schedule K's 4,600 single-row decode steps past the raw ring (~230 s DeepSeek;
  on Qwen schedule P owns it, ~140 s): no compared row reads those rows, so 16 teacher-forced rows a step would
  do -- but neither family's session supports it today.  Tried 2026-10-14 (the server's fold: carry saved,
  `decode_mixed` ALL_ROWS steps, carry restored, `pulsar_session_note_committed_tokens`): on DeepSeek's
  classic (unpooled) session the noted view does not agree with the state, so the payload K shares refuses
  ("no valid checkpoint to snapshot"); on Qwen an ALL_ROWS step is a verify step that does not advance the
  bank, so the next step is refused ("not its run's next position").  An engine API for a teacher-forced
  multi-row decode on a classic session is the lever (~-300 s across both families).
- cuda-accounting-gate's 64k peak (107 s): levels 8k/16k/32k would halve it.
- cuda-head-graph-replay-gate's run C (26 s): it only proves eager determinism, the premise of B == A.
- The spec-sampling oracles' 1,250 trajectories per mode (L163 already halved them).

**Outside `make gates` (2026-10-13; L317 moved the live legs into it):** the landing scripts' extra legs (`~/*-battery.sh` after `make gates`:
`session-contract-gate-qwen`, `cuda-chunk-neutrality-gate` and `cuda-spec-sampling-gate` on
`$(QWEN_GATE_MODEL)`, ~12 min on 2026-10-13) repeat what the runner already runs on the hosted Qwen
(`session-contract-gate [Qwen4-exp]`, `cuda-chunk-neutrality-gate [Qwen4-exp]`,
`cuda-spec-sampling-gate [Qwen4-exp]`, the last at the battery's 1,250 trajectories).  Drop them; keep the
vision, image-smoke and live / abandon server legs, which the runner does not cover.  The live legs are
`make gates`' last section since L317, `live-legs`: constrained-live, server-live, the abandon leg, the L312
stall check, (L316) `prefix-turns-live-gate` -- an agent loop's in-loop and re-sent turns == a fresh server's on
the production Qwen pack, release-blocking (prefix-cache checks are must-pass) --, prefix-turns' side shape,
preempt-exact, (L318) resume-deepest and opencode-turns.

### The GPU queue (gpuq)

Sparky's GPU is shared through `gpuq` (`tools/gpuq/`, installed as `~/bin/gpuq`, daemon `gpuq.service`, a
systemd user unit; spool `~/.gpuq/`).  It replaces `tools/gpu-run.sh` and `tools/gpu-lock-status.sh` (#43)
and a bare `flock -o ~/gpu.lock`, which is not FIFO (seven waiters ran in random order on 2026-10-14), knows
no job classes and no memory, and shows little.

    gpuq submit --class short --timeout 20m --wait -- make gates-dev PATHS=src/engine/session.cpp
    gpuq submit --class quiet --timeout 75m --wait -- ~/qland3-battery.sh <sha>     # a battery
    gpuq submit --class long --timeout 20h --mem 40G --min-hold 1h --yield-grace 90m \
         --label inkling-k3 -- bash ~/inkling/runs/k3_slice.sh                      # prints the job id
    gpuq status [--json]    gpuq log [-f] ID    gpuq cancel ID

- **Classes, in priority order.**  `quiet` > `short` > `long`; FIFO within a class; one job at a time,
  except `--shared` short jobs (below).
  `short`: timeout at most 20m (refused otherwise; default 20m).  `quiet`: the exclusive box for batteries
  (default timeout 2h) -- it starts when nothing runs, nothing is admitted while it runs, and the daemon
  creates `~/BATTERY_RUNNING` before it and removes it after.  `long` (default timeout 24h): preemptible.
  Once `--min-hold` (default 5m) has passed and a quiet or short job waits, the daemon sends SIGUSR1 to the
  job's process group, once.  The job exits **75 (EX_TEMPFAIL)** at its next checkpoint: re-queued with the
  same cmd, `GPUQ_ATTEMPT` + 1, and it resumes when no quiet or short job waits.  Any other exit is final.
  Not exited within `--yield-grace` (default 20m): logged once, and the daemon keeps waiting -- it never
  kills for a yield.  SIGUSR1's default action terminates a process, so submit as `long` only a job that
  handles it.  `gpuq status` shows "yield requested at" from the SIGUSR1 to the job's exit.
- **The job** runs in its cwd (`--cwd`, default the submitter's), with the submitter's environment plus
  `GPUQ_JOB_ID`, `GPUQ_CLASS`, `GPUQ_ATTEMPT`, `GPUQ_RESOURCES`; stdout+stderr to `~/.gpuq/logs/ID.log`
  (`--wait` streams it).  It gets no lock fd.  Timeout is wall time per attempt: SIGTERM to the process
  group, SIGKILL 30 s later.  On any exit every process still carrying the attempt's environment token (a
  daemonized child included) is killed before the lock is released.
- **Admission.**  Head of the queue, no foreign `~/BATTERY_RUNNING` (a legacy battery), `~/gpu.lock` taken,
  then `sync` + drop_caches (once per admission; the page cache masks GB10 unified-memory pressure;
  `GPUQ_DROP_CACHES=0` in the unit turns it off), then MemAvailable >= `--mem` (default 8G short, 16G
  otherwise) + a 6G margin.  A head job short of memory waits holding the lock.
- **Shared jobs** (`submit --class short --shared`): "timing-insensitive and safe to co-run".  Refused on
  `quiet` and `long`.  Rules, in the daemon's order:
  - A shared job starts only from the **head** of the queue (FIFO stays the rule: an exclusive job is never
    overtaken).  An exclusive head -- a non-shared short, a quiet, a long -- waits until the running shared
    group drains, and no new shared job starts past it, so it cannot be starved.
  - It never starts beside a non-shared job, nor while a quiet job waits (a waiting quiet job is the head) or
    runs, nor while a foreign `~/BATTERY_RUNNING` exists.
  - At most `GPUQ_MAX_SHARED` (default 3; `Environment=` in the unit) run at once.
  - Memory, measured fresh at each admission: MemAvailable >= `--mem` + 6G, plus the `--mem` of every group
    member started under 60 s ago (it may not have allocated yet).  drop_caches runs only on an admission
    onto an idle resource; a job joining a running group skips it (it would disturb the running jobs) and
    the daemon logs "drop_caches skipped".
  - The lock: the first job of a group starts a **lock holder** process (`gpuq.py _hold`, pid in
    `~/.gpuq/holders.json`) that keeps `~/gpu.lock` until the daemon sees the group drained, so legacy
    `flock` callers serialize with the whole group; shared shims get no lock fd.  The holder survives a
    daemon restart like the shims, and the next daemon re-adopts it.
  - A waiting shared job is a waiter for a running long job's yield, exactly like any short job.
  - `gpuq status` lists every running job (`shared` tagged) and the group's lock holder.  `status --json`:
    `resources.<res>.running` is a **list** (was one object or null); `running_first` is the old
    single-object shape (or null) for existing readers; `queue` is unchanged, entries gain `shared`.

  **Migration for callers:** model-free kernel gates and host-side tests add `--shared --mem 8G`.  Anything
  loading a 60+ GB model, any perf / timing measurement, and batteries stay exclusive (no `--shared`).
- **Exit codes of `--wait`:** the job's own; 124 timeout; 125 cancelled (also when the waiter itself gets
  INT/TERM/HUP -- the job is tied to its waiter, like a `flock` child); 120 lost (its shim vanished while
  the daemon was down); 127/126 cmd not found / not executable; 2 usage.  `gpuq status` exits 1 on an
  anomaly: an orphaned lock (a dead holder, the lock kept by an inherited fd) or a GPU process that is
  neither the running job's nor under a legacy holder.
- **Legacy callers.**  While a job runs its shim (a shared group: its holder) holds `~/gpu.lock` (flock),
  so `flock -o ~/gpu.lock` callers still serialize with gpuq; a legacy holder makes gpuq wait (it joins the kernel's queue like any
  flock waiter, so it is not ahead of legacy waiters), and `gpuq status` shows it as LEGACY HOLDER with the
  legacy waiters.  Migrate: a battery script's `flock -o ~/gpu.lock -c 'timeout 5400 ~/x-battery.sh'`
  becomes `gpuq submit --class quiet --timeout 90m --wait -- ~/x-battery.sh` (gpuq owns
  `~/BATTERY_RUNNING` now; a script's own `touch` + `trap 'rm -f'` is harmless and can go.  A marker gpuq did
  not create -- a legacy battery's -- blocks every admission until it is removed); an agent's
  `flock -o ~/gpu.lock -c 'timeout 1200 cmd'` becomes `gpuq submit --class short --timeout 20m --wait -- cmd`.
- **Robustness.**  The daemon restarts without touching jobs (`KillMode=process`): each attempt runs under
  a shim that holds the lock and writes the outcome, and the next daemon re-adopts live shims, finalizes
  those that finished while it was down, and marks the rest lost (killing what is left of them).  Every
  state file is written atomically; every scheduling decision is in `~/.gpuq/gpuq.log` and the journal.
  The user unit needs `loginctl enable-linger claude` to outlive the last login session.
- **Tests.**  `python3 tools/gpuq/test_gpuq.py -v` on Sparky (scratch in `~/gpuq-test/`): a real daemon per
  test in test mode (`GPUQ_TEST=1` with its own `GPUQ_HOME`, `GPUQ_LOCK`, `GPUQ_MARKER`, a fake
  `GPUQ_MEMINFO` and drop_caches command), dummy jobs, no GPU, ~75 s.  Install: `tools/gpuq/install.sh`.
- **Spool format, version 1, and the v2 plan.**  Every job record, outcome and the daemon state carry
  `"version": 1`, and every job names its resources (`"resources": ["sparky:gpu0"]`, from the hostname;
  `--gpus gpu0` or `--gpus sparky:gpu0`).  The lock and the memory pool are per resource; v1 owns one
  resource per host and refuses any other by name.  v2 (not built): a coordinator reads several hosts'
  spools over ssh (`~/.gpuq/jobs/*.json`, `state.json`) and adds gang jobs that need every listed resource
  at once (the two-Spark pair: `--gpus sparky:gpu0,spark2:gpu0`), taking each host's lock before starting.


## 3. Flash Inference Path

Use the served Flash checkpoint that 128 GB users run.

- One-shot CLI:
  `./pulsar -m /srv/models/<checkpoint-dir> --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./pulsar --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `pulsar-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.
- Run a longer prompt that exercises routed experts past a few thousand tokens.

## 4. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./pulsar-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 5. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.
- `curl /metrics` and confirm the Prometheus page renders: spec-decode
  acceptance, token totals, request latency, per-slot generation phase, pool
  churn and admission pressure counters all present and moving after a request.
- Reasoning effort: send Anthropic requests at `low`, `high`, and `max` effort
  and confirm the reported `prompt_tokens` reflects the three distinct effort
  prefixes (and that `xhigh` maps to `max`).
- Anthropic server tools: send a Claude-Code-style request advertising
  `web_search_20250305` next to an ordinary tool and confirm the server-tool
  entry is dropped at parse (tool log line), the ordinary tool is rendered,
  and the model never emits a dead search call. The server executes no
  Anthropic server tools; the router in front of it owns web search.

## 6. pulsar-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Read/search/edit/write tools:
  create a temp project, ask for edits, verify old/new and `[upto]` anchored
  edits fail safely on ambiguous matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask pulsar-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 7. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target enough to verify URL, resume, Hugging Face CLI/curl
  behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.
- Run the container audit on every shipped checkpoint (`tools/container`, L247): every
  tensor declared in the layout its kernel reads, no plain twin of a pre-formatted
  layout, the expert stacks contiguous.  (The GGUF-era `audit_artifact_types.py` was
  archived 2026-09-24 at tag `archive/gguf-tooling-2026-09-24`.)

## 8. Performance

- Run `pulsar-bench` on the release machine and compare with tracked CSV baselines.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

### Client-side serving gates

These talk HTTP to an already-running `pulsar-server`, because they measure what
a client experiences rather than what the engine reports about itself.  Start the
server with `--no-kv-disk` for all three: a warm disk checkpoint skips prefill
outright, so TTFT and wall-clock t/s from a warm run cannot be compared against a
cold one.  Each writes a machine-readable JSON record — keep it with the release
notes, since a number with no run record is not a baseline.

- `make decode-floor-gate` — five workloads (Python, Rust, TypeScript, CUDA, Go)
  at 512 tokens, concurrency 1.  **Scored on the worst workload, not the mean**:
  averaging one bad prompt shape against four healthy ones produces a number that
  still looks fine while the regression ships.  Fails (exit 1) when the worst
  workload drops more than `--tolerance` (default 10%) below the recorded floor.

  The floor is ours, captured with `make decode-floor-baseline` on a known-good
  build — not a competitor's published figure, which was measured on a different
  harness, depth and sampler and is therefore not a floor.  Re-record it
  deliberately when a change is *expected* to move decode, and say so in the
  release notes; never re-record it to make a red gate go green.

- `make sse-decode-bench` — decode rate measured between SSE delta arrivals,
  reported **alongside** the wall-clock rate rather than replacing it.  When the
  two disagree the gap is the finding: a slot starved behind another job's
  prefill shows a healthy inter-delta rate and a poor wall-clock one.

- `make context-coherence-probe` — plants three facts at the beginning, middle
  and end of a long context and asks for them back in free prose, with no grammar
  mask or constrained decoding (under a mask a wrong answer still comes out
  structurally perfect, so the probe cannot fail in the way that matters).  A
  checksum question requiring all three values at once guards against partial
  latching.  Fails (exit 1) if any depth misses a fact or the sum.

## 9. Release Sign-off

Do not sign off until:

- `make test` and `make cuda-regression` passed on the GB10 host.
- The CUDA gates ledger (section 2) passed against the release commit and the
  shipped artifact, with a post-type-43 prefill-gate baseline.
- The Flash inference path (CLI, thinking modes, long-context recall) was
  exercised.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- `make decode-floor-gate` passed against the recorded floor, and its JSON run
  record is filed with the release notes.
- Any skipped item is written down with the reason.

# L264 on the TP2 pair: test kit

L264 replaced the KV snapshot machinery with **grid checkpoints** (a resume restores the deepest
checkpoint at or below the prefill frontier) and the disk KV cache with **segment chains**
(`src/lib/pulsar_segstore`, `src/lib/pulsar_kvchain`). On a TP group every worker keeps its own
copy of each segment the leader's store names. The engine mirrors save and load as verdicts,
and any disagreement is a *miss* on every rank, never a divergence. Drops and a bring-up
reconcile keep the copies in step. This is **protocol v23**: an L264 build pairs only with an
L264 build.

Hosts: head (rank 0) `ca1070wk30008` (TP wire 192.168.9.13); worker (rank 1) `ca1070wk30007`
(192.168.9.12). Launch recipe as `notes/l260/scripts/run-build.sh`: worker first, `mlx5_3` and
the second rail `mlx5_1` on both.

## 1. Build the same tree on both nodes

```sh
# on each node, same path
git -C ~/pulsar-l264 fetch && git -C ~/pulsar-l264 checkout --detach <sha>   # or a git archive of <sha>
make -C ~/pulsar-l264 -j20 CUDA_ARCH=sm_120f pulsar-server 2>&1 | grep -E "error|warning"   # must print nothing
```

## 2. Run the kit on the head

```sh
~/pulsar-l264/tools/tp2-l264/run.sh ~/pulsar-l264
```

It stops `pulsar-tp` / `pulsar-tp-exp` on both nodes and runs everything under the transient
unit `pulsar-tp-exp` with a fresh disk KV dir (`$KV`, default `~/l264-tp2-kv`) on both nodes.
It leaves the pair stopped. Restart production afterwards with `sudo systemctl start pulsar-tp`
on both nodes. The whole run takes about 25 minutes, mostly the three model loads. Logs go to
`~/l264-tp2-<date>/`, and the last block of `run.log` is the verdict.

You can override any of these through the environment: `WORKER`, `PEERS`, `MODEL`, `RDMA_DEV`,
`RDMA_DEV2`, `KV`, `GUARD_BUDGET` (bytes), `OUT`.

## What each verdict proves

| Verdict | Step | Proves |
|---|---|---|
| `preflight` | same protocol (23) and the same engine source digest on both nodes | the pair can form |
| `host tests` | `tp_mirror_test`, `tp_transport_test` (TCP), `tp_mesh_test` | the frames: SEGMENT_SAVE/LOAD/DROP/RECONCILE encode, decode, refuse |
| `phase1 (resume)` | A cold 6.5k-token turn; B, the next turn echoed **without reasoning** (the L264 client shape); C, B resent exactly; D, turn 3 | B resumes from a grid checkpoint (cached ≥ 90 % of A), not cold from 0; an exact resend re-evaluates its last row and gives the same greedy reply |
| `identity runN` | the engine's `cross-rank logits identity M/N` at shutdown | every mirrored step produced the same logits on both ranks |
| `alarms runN` | no `diverg` / `marked failed` / `refusing to apply` lines | no split verdicts |
| `reconcile` | an orphan `.tpseg` planted on the worker before restart is gone after bring-up | SEGMENT_RECONCILE |
| `copies` | worker `.tpseg` count = leader `.seg` count | every stored segment was mirrored |
| `phase2 (disk restore)` | E: turn 3 after the restart; F: four concurrent new conversations | the chain loads on **both** ranks (cached > 50 %) and the reply is identical to before the restart; mirrored stores under concurrency |
| `miss` / `miss seen` | every worker copy truncated, then turn 3 resent | a worker miss is a clean miss: the pair stays up, the chain is dropped, and the reply is unchanged (cold prefill) |
| `guard` / `guard restores` | relaunched with `PULSAR_SERVER_KV_BUDGET_OVERRIDE`; six ~10k conversations, each resent | the eviction guard spills banks to the segment chain and restores them on both ranks, and the replies are unchanged |

## Reading a failure

- **The pair won't form**: the preflight caught a protocol or tree mismatch, or the bring-up
  failed. Check `head-boot*.log` / `worker-boot*.log` for the hello's refusal.
- **`phase1` B cached is low**: the resume didn't find a checkpoint. In `head-run1.log` look for
  `resume at N on bank B: no grid checkpoint at or below G -- prefilling the prompt from 0`
  (engine stderr). The single-Spark gates cover this path (chunk-neutrality schedules L–P, X), so
  a TP-only failure points at the mirrored sync (`SYNC_CHECK` refusing, or the ranks capturing
  different checkpoints).
- **`copies` differ**: grep both logs for `kv segment save`. A worker `save failed` is a miss: the
  leader skipped that segment on every rank. The count can then differ legitimately only if the
  leader's store evicted something, which it doesn't at these sizes.
- **`phase2` E cached is 0**: the chain didn't load. Look for `kv cache segment ... failed to
  load` on the head and `kv segment load miss` on the worker. A miss names its reason (no copy,
  or a different state: G or digest).
- **A `marked failed` line anywhere**: a split verdict or a lost frame. The line before it names
  the operation. A segment frame inside a mirrored prefill is refused by name (`inside a mirrored
  prefill is not mirrored`). The server skips its "continued" stores there, so seeing that refusal
  is a bug.
- **Reasoning drift under concurrency** (`[WARN] H resend`): expected. Batched decode is not
  bitwise deterministic. That's a deliberate speed tradeoff: a row's logits depend on what it is
  batched with, so a greedy near-tie can tip differently between runs. The guard leg decodes two
  banks at a time, so it grades the final answer and only reports a reasoning difference. The
  sequential legs (phase1/phase2) decode alone and grade the whole output.
- **A hang (`/health` ok, nothing moves)**: that's L258's wedge class, not L264. Take
  `gdb -p <pid> -batch -ex "thread apply all bt"` on both ranks before stopping.

## Code map

| Piece | Where |
|---|---|
| Frames 52–55, `pulsar_tp_segment_command` | `src/tp/pulsar_tp.h` |
| Leader: mirrored `save_segment` / `load_segment`, drop, reconcile | `src/engine/engine_api.cpp` (`pulsar_session_save_segment` …) |
| Worker: `<tp_kv_dir>/<key>.tpseg` save/load/drop/reconcile | `src/engine/tp_worker.cpp` (`worker_segment_*`) |
| Store hook → drop frames; bring-up reconcile | `src/server/cli_main.cpp` (`kv_segment_removed`) |
| No stores inside a mirrored prefill | `kv_cache_persist` → `pulsar_session_in_mirrored_sync` |

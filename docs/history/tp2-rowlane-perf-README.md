> **Historical, not maintained** (moved to docs/history/ on 2026-10-30): the L269-perf TP row-lane decode A/B, about 2026-10-11 (was tools/tp2-rowlane-perf/README.md; the scripts stay there). The code at the tip of `dev` is the authority.

# Row-lane decode A/B on the pair (L269-perf)

The TP row lane's combine kernels used to **spin on the done word** until the proxy reported the peer's rows. On
the dev GB10's loopback (both ranks on one GPU, two processes, time-sliced) a spinning kernel keeps its context resident
while the rank it waits for cannot run, so every exchange cost about one GPU timeslice (~2 ms):

| the dev GB10 loopback, 256 greedy tokens | lane off (pre-v26) | lane on, spin (fe8ab143) | lane on, stream wait (this build) |
|---|---|---|---|
| V4 7-layer fixture (vexp-tp-fixture) | 19.96 ms/token | 60.59 | **18.25** |
| V4.1 7-layer fixture (v41-tp1-fixture-mx) | 35.42 | 65.16 | **35.21** |

Logprobs byte-identical across all three builds on both fixtures. The new build waits with
`cuStreamWaitValue64` on the done word (the GPU front end holds the stream; no SM is held) and runs the combine after
it. Decode on the pair already ran the spinning lane (V4 0731 has 43 layers, so the lane was on); there each rank has
its own GPU, so time-slicing does not apply. A one-process probe on GB10 puts the stream wait at **+2.7 us per
exchange** over the spin (6.2 vs 3.5 us round trip). V4 0731 runs ~131 exchanges per decode step, so expect about
**+0.35 ms per step (~0.7% at c1)** on the pair. This kit measures it.

## 1. Build both trees on both nodes, at the same paths

```sh
# A = what the pair serves today (or fe8ab143, the v26 lane with the spin); B = work/rowlane-perf
for s in <A sha>:pulsar-rl-a <B sha>:pulsar-rl-b; do
  sha=${s%%:*}; dir=~/${s##*:}
  git -C $dir fetch && git -C $dir checkout --detach $sha                     # clone first if missing
  make -C $dir -j20 CUDA_ARCH=sm_120f CUTLASS_DIR=<pinned cutlass> pulsar pulsar-server 2>&1 | grep -E "error|warning"   # nothing
done
```

A and B may speak different TP protocols (v25 / v26); each arm runs both ranks from the same tree, and the preflight
checks that per arm.

## 2. Models on both nodes, at the same paths

| Model | Default (env) | Needed |
|---|---|---|
| DeepSeek-v4-Flash 0731 (production) | `~/models/DeepSeek-v4-Flash` (`MODEL=`) | yes |
| V4 Vision-Exp 7-layer MXFP4 fixture, 27 GB | `~/models/vexp-tp-fixture` (`V4FIX=`) | optional: the dev GB10 `/path/to/models/vexp-tp-fixture` |
| V4.1 7-layer fixture, MXFP4 layer 20 | `~/models/v41-tp1-fixture-mx` (`V41FIX=`) | optional, see below |

A fixture missing on either node skips its leg by name. **V4.1 on the pair:** there is no V4.1 artifact on the pair
(L269). The fixture is the dev GB10 `/path/to/models/v41-tp1-fixture-mx`: 9 shards, ~57 GB once copied with `cp -L` (eight are
symlinks into `/path/to/models/v41-tp1-fixture`), plus its two Engram tables `engram-l1.rows` / `engram-l14.rows`,
~94 GB each (`/path/to/models/engram-v41`). Put the tables on local NVMe and keep the two `engram-*.rows` links pointing
at them: over NFS the Engram reads dominate the decode (~3 tok/s, L269). That is ~245 GB per node. If that is too
much to carry, run without it. The V4 fixture covers the same lane at the same 7-layer shape.

## 3. Run on the head (pair-b), as the serving user

```sh
~/pulsar-rl-b/tools/tp2-rowlane-perf/run.sh ~/pulsar-rl-a ~/pulsar-rl-b
```

This stops the pair (`pulsar-tp` on both nodes). Then, for each of `ROUNDS` (2) rounds, it serves the production model
from A then B (B then A on even rounds). Each serve gets one 32-token warm request, then `STREAMS` (`1 2 4`)
concurrent greedy completions of `TG` (256) tokens on `PP` (~512)-token prompts. The prompts are fixed and identical
across builds. Then it runs each fixture through the TP=2 CLI, 256 greedy tokens per build. Expect about 30-40 min.
Nothing needs the network.

The pair is left stopped. Afterwards, run `sudo systemctl start pulsar-tp` on both nodes.

## 4. What it prints (paste the last block back)

```
| streams | A decode tok/s | B decode tok/s | B/A | A per-stream | B per-stream | A step ms (p90) | B step ms (p90) | tok/step A/B |
| fixture | A lane | A ms/token | B lane | B ms/token |
================ ROW-LANE A/B VERDICT (~/rowlane-ab-<date>) ================
```

- **decode tok/s:** all streams' tokens over the window where every stream decodes.
- **step ms:** the gap between engine steps as one stream sees it. Chunks within 2 ms are one step.
- **tok/step:** above 1 when drafts are accepted.

The rounds are averaged, and the per-round line shows the spread. The verdicts check four things: each serve's
cross-rank logits identity, no divergence or refusal alarms, each fixture's identity, and fixture logprobs A == B
byte for byte. The lane does not change arithmetic: the same combine in the same order.

**Reading it:**
- **B/A >= ~0.99 at every stream count:** the wait is right on the pair too. Land it.
- **B clearly slower (a few % at c1):** the stream wait's dispatch latency costs more on the pair than the probe
  says. Bring the table back. The fix then is to cut the wait's dispatch cost, not to keep a second (spinning) path.
- **Fixture rows:** with A older than v26, A runs both fixtures with the lane OFF. That is the "lane on vs off"
  answer for the pair.

Logs and raw JSON: `~/rowlane-ab-<date>/` (`results.jsonl`, `table.txt`, `head-*.log` / `worker-*.log`, `cli-*.log`).
The plumbing is shared with the other pair kits (`tools/tp2/lib.sh`).

# L266 on the TP2 pair: test kit

L266 serves **Qwen3.8-Flash-Next on turboderp's EXL3 packs** in pulsar, and step 7 splits it across the pair:

| Part | How it is split |
|---|---|
| GDN | Each rank holds 8 key heads and 24 value heads |
| QSA | Each rank holds one KV head and its 12 query heads; the indexer is whole on both ranks |
| Routed experts | Expert-parallel: rank r holds experts [256r, 256r+256) |
| Shared expert | Rank 0 computes it |
| Head | Split by vocab and gathered |

- **Exchanges:** y is all-reduced after every mixer and every MoE.
- **One-host validation:** everything the kit checks was first validated on one GB10 with both ranks as processes (TCP and RDMA loopback; L266 notes). The pair adds two real GPUs and the 6.05 pack, which does not fit one Spark (about 47 GB per rank on the pair).
- **Protocol:** unchanged, v23. A pair runs only the same tree.

Hosts and launch recipe: as `tools/tp2-l264/README.md`. The plumbing is shared in `tools/tp2/lib.sh`.

## 1. Build the same tree on both nodes

```sh
git -C ~/pulsar-l266 fetch && git -C ~/pulsar-l266 checkout --detach <sha>
make -C ~/pulsar-l266 -j20 CUDA_ARCH=sm_120f pulsar-server tests/qwen_generate 2>&1 | grep -E "error|warning"   # nothing
```

## 2. The two containers on both nodes, at the same paths

Built on sparky from turboderp's snapshots plus the HF BF16 checkpoint
(`tools/container/build.py --recipe format-maps/qwen38fn-turboderp-{4.05,6.05}.json`).

| Container | Size | n-gram rows |
|---|---|---|
| `/srv/models/qwen38fn-td405` | 67 GB | `ple-l1.rows`, 102 GB |
| `/srv/models/qwen38fn-td605` | 98 GB | the same file, hard-linked |

Copy each container as a whole (`rsync -aH`, so the rows file stays one inode), to `~/models/` on each node, or
set `Q405` / `Q605`.

## 3. Run the kit on the head

```sh
~/pulsar-l266/tools/tp2-l266/run.sh ~/pulsar-l266
```

| Step | Proves |
|---|---|
| preflight | One protocol and one source tree on both nodes; both containers whole on both |
| host tests | The TP transport, mirror and mesh host tests |
| answers 4.05 | Three chat answers through the pair (391 / Tokyo / a Fibonacci `def`) |
| resume 4.05 | L264's resume smoke across a stripped-reasoning echo: Qwen's grid checkpoints mirrored on the pair |
| identity / alarms | Every decode frame's logits equal on both ranks; no divergence or refusal |
| TP=2 == TP=1 | `qwen_generate` greedy, 32 tokens of the bundled code probe, on the pair == on one node |
| answers 6.05 | The 6.05 pack on the pair: the three answers, with the end-to-end tok/s in `qa-605.log` |

## Reading a failure

- **TP=2 != TP=1:** run the two `qwen_generate` logs side by side. With
  `PULSAR_CUDA_GRAPH_DUMP_PREFIX` and `_POS=199` the per-layer streams can be compared. On one GB10 the difference is
  1e-3 after layer 0, growing to about 0.2 by layer 47, and the tokens still agreed. A divergence at a near-tie
  is possible in principle; a divergence at layer 0 is a bug.
- **Identity FAIL:** the line names the operation. The mixed batch's row count was one such bug (fixed in
  67b00051).

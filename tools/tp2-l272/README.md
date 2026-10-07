# L272 on the TP2 pair: test kit

L272 made the model families plugins. Two of its changes can only be fully proven on two real GPUs:

- **P4b, the TP plan.** Each family *declares* its rank's slices; the core picks the operation by format (fp8 row/K
  registration, EXL3 gathers, MXFP4 expert halves, expert ranges) and the plan is the residency rule.
- **P4c, the front doors.** One dense linear door and one routed-MoE door; under TP a rank reads what its plan says
  (DeepSeek: the MXFP4 halves and the shared-down K slice under the engine key; Qwen: its range of whole experts).

Both are bit-exact by construction -- the same slices, the same kernels in the same order. So the kit grades the
new build **byte for byte against a reference build from before P4** on the same greedy requests, plus the
engine's own per-frame cross-rank identity. Protocol v24, unchanged by L272: the reference pairs with the same frames.

Hosts and launch recipe: as `tools/tp2-l264/README.md`. The plumbing is shared in `tools/tp2/lib.sh`.

## 1. Build both trees on both nodes, at the same paths

```sh
for s in <new sha>:pulsar-l272 eee6d95f:pulsar-l272-ref; do
  sha=${s%%:*}; dir=~/${s##*:}
  git -C $dir fetch && git -C $dir checkout --detach $sha     # clone first if missing
  make -C $dir -j20 CUDA_ARCH=sm_120f CUTLASS_DIR=<pinned cutlass> pulsar-server tests/qwen_generate tests/tp_plan_test \
      2>&1 | grep -E "error|warning"                          # nothing
done
```

`eee6d95f` is the last commit before P4b. The reference is optional: without it the A/B legs say SKIPPED and the rest
still grades.

## 2. Models on both nodes, at the same paths

| Model | Default path | Why |
|---|---|---|
| DeepSeek-v4-Flash (CUTLASS MXFP4 experts) | `~/models/DeepSeek-v4-Flash` (`DS=`) | DeepSeek TP builds the expert halves from MXFP4 stacks |
| Qwen td405 | `~/models/qwen38fn-td405` (`Q405=`) | fits one Spark: TP=2 == TP=1 |
| Qwen td605 | `~/models/qwen38fn-td605` (`Q605=`) | the pair-only pack |

## 3. Run on the head

```sh
~/pulsar-l272/tools/tp2-l272/run.sh ~/pulsar-l272 ~/pulsar-l272-ref
```

About 45 minutes with a reference (six model loads), 25 without. Logs: `~/l272-tp2-<date>/`; the last block of `run.log`
is the verdict. The pair is left stopped: `sudo systemctl start pulsar-tp` on both nodes afterwards.

| Verdict | Proves |
|---|---|
| `preflight` (per build), `models` | one protocol and one tree per build on both nodes; the models present on both |
| `host tests` | the TP transport / mirror / mesh host tests |
| `DeepSeek TP plan == golden` | the pair's own copy of the model plans, for both ranks, exactly the committed `tests/tp-plan-golden/` (no GPU) |
| `answers ds-new` / `identity ds-new` / `alarms ds-new` | DeepSeek served through the plan and both doors on two GPUs: the three answers, every frame's logits equal on both ranks, no divergence |
| `A/B ds-new == ds-ref` | **the P4b/P4c proof for DeepSeek**: the three greedy replies, reasoning included, byte-identical to the pre-P4 build |
| `Qwen TP=2 == TP=1` | the Qwen split (expert ranges from the plan, built dense slices) is the one-GPU model, 32 greedy tokens |
| `answers / identity / alarms q605-new`, `A/B q605-new == q605-ref` | the pair-only pack through the plan, byte-identical to the pre-P4 build |

## Reading a failure

- **`DeepSeek TP plan` differs:** the pair's model copy is not the artifact the golden was recorded from (offsets in
  the diff), or the plan changed. Compare `sha256sum model.safetensors.index.json` with sparky's copy first.
- **`A/B` differs but `identity` passes:** both ranks agree, the build changed the numbers. Run the two `qa-*.json`
  side by side: the first differing token's request names the path (a dense projection: the dense door; an MoE layer:
  the routed door; the shared expert under TP: the K slice). P4c's single-box proof is the battery, so a TP-only
  difference points at what a rank reads from its plan (`pulsar_tp_slice_of`).
- **`identity` fails:** the ranks disagree -- a slice registered on one rank and not the other, or a different plan.
  The `TP plan:` line each rank prints at open must be the same apart from the rank.
- **A bring-up refusal naming a slice** (`TP plan: no operation ...`): the model's format has no operation for that
  family's reader. That is the plan working; the model is not servable under TP as built.

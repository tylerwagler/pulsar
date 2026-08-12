# Reproducing a serving artifact from source weights

Written 2026-08-12 after auditing whether the shipped artifact could actually be
rebuilt. It could not, cleanly: the format map that decides every tensor's type
was an untracked JSON in a scratch directory, and the imatrix that drives every
quantization decision was a 450 MB file whose only backup nobody could find
because the NAS copy has a different name. Both are now pinned below.

## ✅ VERIFIED 2026-08-12 — this recipe has been executed end to end

A full rebuild from the source weights was run and compared against the served
artifact `v5mx4-0731-ltdraft.gguf`:

    tensor data, both 92,490,470,016 bytes
      rebuild b4c4ac7c47463b4046215975aa904603c0b63789c3e5cbb68e026413b3350523
      served  b4c4ac7c47463b4046215975aa904603c0b63789c3e5cbb68e026413b3350523

**Byte-identical.** Same 1406 tensors, same 69 KVs, same type census
(F32 536 / MXFP8_LT 370 / F16 359 / IQ2_XXS_MMQ 91 / CUTLASS_MXFP4 47 / I32 3),
audit gate PASS. The ONLY difference in the whole 92.5 GB file is the
`quantize.imatrix.file` KV, which records the imatrix's PATH — `../v5mx2-build/
routed-moe-ds4-1p5m.dat` on the served copy vs the NAS path here. That 46-byte
string difference pushes `data_pos` one 32-byte alignment unit, which is the
entire 32-byte file-size delta.

**Fixed 2026-08-12.** The GGUF now records `quantize.imatrix.sha256` — the
imatrix's content hash — and no longer records its path at all. A path says
where someone's disk was; a hash says which imatrix it was. The KV is now a
fixed 64 hex chars, so it is identical on every machine and the one remaining
metadata divergence (and the 32-byte `data_pos` shift it caused) is gone.

SHA-256 is implemented in `quantize/dsq_sha256.c` to keep the quantizer
dependency-free. It self-tests against the FIPS 180-4 vectors on first use,
including a multi-block case, because a silently wrong hash would stamp a
confident and meaningless identity into every artifact. It was also checked
against this document's independently recorded value for the real 450 MB
imatrix — `02a7c78c…` — which it reproduces exactly.

Note the ordering: artifacts built before this change (including the first
collapsed-pipeline validation run) still carry `quantize.imatrix.file`. Compare
their metadata accordingly.

**Executable form: `gguf-tools/build/rebuild_stages_3_7.sh`** — stages 3-7 with
`set -euo pipefail`. Prefer it over the prose below; a script cannot silently
omit a stage, and four of the defects listed under §5 were exactly that.

---

## 1. Inputs — what you need before you start

| input | location | identity |
|---|---|---|
| HF source checkpoint | `/mnt/pve1-models/dsv4-flash-0731/` (NAS; **not** on any local disk) | 225 shards |
| imatrix | `/mnt/pve1-models/ds4-quant-archive/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` | sha256 `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e`, 450,892,648 B |
| format map (v5mx4, **the shipped one**) | `gguf-tools/prisma/v5mx4-format-map.json` | 345 MXFP8_LT / 38 CUTLASS_MXFP4 / 91 IQ2_XXS |
| REAP survivor map | `gguf-tools/reap/reap25-lcb50-survivors.json` | — |
| drafter type pins | `gguf-tools/dspark_type_flags.txt` | 25 mxfp8_lt / 44 f32 / 9 cutlass_mxfp4 |
| imatrix calib corpus | `gguf-tools/imatrix/dataset/` | `calib-diverse-ds4-v1` |
| eval corpus (scoring only) | `gguf-tools/quality-testing/data/` | 100 cases |
| **a known-good existing GGUF** | see the bootstrap warning below | tokenizer source |

The imatrix filename on the NAS does not contain the word "imatrix". That is the
whole reason it looked missing during the audit. It is byte-identical to what was
sitting in `~/Projects/AI/temp/imatrix.dat`; prefer the NAS copy and verify the
hash above.

### Tokenizer: the bootstrap is now broken (2026-08-12)

`build_main_template.py` still defaults to `--splice-tokenizer-from <existing ds4
GGUF>`, copying `tokenizer.*` KVs verbatim from a prior artifact. That made a
strict from-source rebuild impossible: each artifact needed an earlier one.

**`gguf-tools/tokenizer/build_tokenizer_kvs.py` replaces it.** It derives all
eleven KVs from `tokenizer.json` + `tokenizer_config.json`, verified
**byte-identical** to the live artifact (including the 2.3 MB tokens and 2.4 MB
merges arrays):

    python3 gguf-tools/tokenizer/build_tokenizer_kvs.py \
        --hf /mnt/pve1-models/dsv4-flash-0731 --out KVDIR \
        [--verify-against REFDIR]

Three inputs are **not derivable** from any HF file and live in the repo as
explicit constants — they are our decisions, not borrowed artifact bytes:

| input | where | why not derivable |
|---|---|---|
| `tokenizer.ggml.pre` = `joyai-llm` | `build_tokenizer_kvs.py` | llama.cpp pretokenizer id; a classification |
| chat template | `gguf-tools/tokenizer/chat_template.jinja` | checkpoint ships the format as Python (`encoding/encoding_dsv4.py`), not Jinja |
| 6 `USER_DEFINED` tokens | `build_tokenizer_kvs.py` | hand-curated; HF's `special` flag gives 1230/53, not the required 1277/6 |

That last one is not cosmetic. `<think>`, `</think>` and the DSML markers must be
USER_DEFINED (4), not CONTROL (3), because CONTROL tokens are skipped during
detokenization — mark them wrong and the model's reasoning output and tool calls
are silently swallowed, with every other gate still green.

**Always run `--verify-against` when the tokenizer path changes.** A tokenizer
that is merely close yields an artifact that loads, generates, and mis-tokenizes;
nothing else here catches that.

⚠ **Still to do:** wire this into `build_main_template.py` so splicing is no
longer the default. Needs a full template build to verify; not yet run. Until
then, if you do splice: never proceed past a template build whose log does not
show >0 spliced tokenizer KVs and an MB-scale file size.

## 2. Pipeline

**Reproducing an artifact does NOT re-run the probe/cost/allocator stages.** Those
MEASURE, and re-measuring yields a new map — a different (also valid) exercise.
A reproduction uses the pinned `v5mx4-format-map.json` from §1. Re-derivation is:

    HF checkpoint -> sensitivity probe -> cost stage -> prisma_alloc.py -> map
    (CACHE_HEADROOM_GB=90 REQUIRED for probe AND cost, or autoscale takes an
     ~86 GB layer cache and earlyoom kills the run, exit 137)

The build proper, given a map — run it with
`gguf-tools/build/rebuild_collapsed.sh`:

    1. build_main_template.py --hf DIR --out T.gguf --tokenizer-from-hf      5 MB
         --reap-survivors SURV     -> REAP-SHAPED template (expert dim 192,
                                      router/bias still 256)
    2. deepseek4-quantize --hf DIR --template T.gguf --out COMPACT          81.5 GB
         --format-map MAP-MMQ --reap-survivors SURV [--imatrix ...] --threads N
         -> pruned AND pre-formatted in one pass
    3. build_dspark_template.py     (drafter, from the checkpoint's mtp.* block)

...and that is the whole build. Step 2 takes `--dspark-template` too, so one
quantizer pass emits main + drafter, pruned and pre-formatted, straight to the
92.5 GB artifact.

**Three stages collapsed into the quantizer on 2026-08-12** (see §Collapse
below): `trim_reap.py` → `--reap-survivors`, `merge_dspark_gguf.py` →
`--dspark-template`, and `repack_iq2_mmq.py` → a format map naming
`IQ2_XXS_MMQ`. The 102 GB full-256 intermediate is gone entirely.

**Cost.** ~92.5 GB written for a 92.5 GB artifact (1.0x amplification, was
4.1x). There are no full-size intermediates left, so nothing has to be deleted
mid-run. The old staged script (`build/rebuild_stages_3_7.sh`) is superseded and
kept only as history.

**Every stage here fails silently if skipped.** That is why they were collapsed
rather than documented harder: the drafter shipped 0.43 GiB of double-store for
months because a repack was never run, and skipping the REAP stage produces an
unpruned artifact ~16 GB too large with nothing going red. Prefer fewer stages
over better discipline.

### Collapse: why the order was forced

REAP had to collapse **first**, and that is not a preference:

- `trim_reap.py` is a post-quantize transplant that re-slices whole tensors, so
  it needs block geometry for every type in the file. Pointing the old pipeline
  at an MMQ format map died with `KeyError: 43` in `tbytes()` — the repack stage
  ran *after* trim, but trim had to parse the type trim itself never emits.
- With REAP inside the quantizer there is no post-pass left to trip over, so
  MMQ became emittable in the same step. `trim_reap.py` now refuses type 43
  outright rather than growing a block-geometry entry that would re-enable the
  broken ordering.

Both collapses were verified byte-exact against the shipped v5mx4 artifact
*before* replacing the staged script, using `--compare-tensor`:

| check | tensor | result |
|---|---|---|
| REAP router permutation | `blk.5.ffn_gate_inp.weight` (f16) | byte-identical |
| REAP bias sentinels | `blk.5.exp_probs_b.bias` (f32) | byte-identical |
| REAP policy-1 control | `blk.0.ffn_gate_inp.weight` | byte-identical (untouched) |
| REAP expert remap | `blk.5.ffn_gate_exps.weight` (cutlass_mxfp4, 855 MB) | byte-identical |
| REAP + imatrix-by-original-id | `blk.12.ffn_down_exps.weight` (iq2_xxs, 415 MB) | byte-identical |
| MMQ collapse | `blk.12.ffn_down_exps.weight` type 43 | == `repack(type 39)` |
| merge collapse, drafter experts | `dspark.0.ffn_gate_exps.weight` (1.14 GB) | byte-identical |
| merge collapse, drafter dense | `dspark.main_norm.weight`, `dspark.0.hc_attn_fn.weight` | byte-identical |
| merge collapse, main unaffected | `blk.5.ffn_gate_inp.weight` | byte-identical |

The fully-collapsed plan also reproduces the shipped artifact's shape exactly
before a single byte is quantized: 1406 tensors and `approx_file_bytes`
92,495,809,696, both matching the served copy.

**The drafter carries its own `dspark.N.ffn_*_exps` stacks**, and those N
collide with main layer indices. Handing the REAP survivor map down to them
would trim them against an unrelated layer's policy. It is harmless *today*
only because dspark's layers 0-2 land on the three policy-1 (untouched) layers
— luck, not design — so `generate_tensor` cuts REAP off by name for anything
under `dspark.`. Without that, changing the survivor map would silently corrupt
the drafter's experts.

Layers 5 and 12 were chosen because their survivor lists deviate from identity
at slot 0 and 1 respectively and run out to source experts 254/255 — an identity
bug cannot pass there by luck. The negative control matters as much as the
checks: re-running the router comparison **without** `--reap-survivors` fails at
byte 0 with 2,038,329 mismatches, which is what proves the passing runs are
testing anything at all.

**`fnv1a64_bytes()` in `dsq_gguf_io.c` is not standard FNV-1a** — its offset
basis is `1469598103934665603`, one digit short of the real
`14695981039346656037`. It is a self-consistent checksum used only for
diagnostics, and `--compare-tensor`'s OK/FAIL verdict comes from an actual
bytewise comparison rather than the hash, so no result is affected. It does mean
these hashes cannot be cross-checked against any external FNV implementation
without reproducing the typo.

`CACHE_HEADROOM_GB=90` must be set explicitly for the probe and cost stages or
autoscale grabs an ~86 GB layer cache and earlyoom kills the run (exit 137).

The DSpark drafter always ships **inside** the checkpoint as the `mtp.*` weight
block (4,705 tensors) — there is no separate drafter repo or file. Check the
weight index, not filenames.

**Drafter-only fixes do not need a rebuild.** `merge_dspark_gguf.py` preserves the
main region byte-for-byte, so `unmerge_dspark_gguf.py` + fix + re-merge is far
cheaper than re-quantizing (a from-scratch rebuild once crawled at ~38 min/layer).

## 3. Gates before anything is served

Run all of these. Each one has caught a real defect:

- **Tensor census** — 0 mismatches against the expected manifest.
- **Tokenizer KVs** — >0 spliced, MB-scale template (see the bootstrap warning).
- **Drafter type parity** — diff against `dspark_type_flags.txt` must be 0. A
  drafter built with plain quantizer defaults once came out 60 tensors different
  and refused to load (`dspark.main_norm.weight has type bf16, expected f32`).
- **`gguf-tools/audit_artifact_types.py MODEL.gguf`** — fails if any tensor ships
  a plain type that has a pre-formatted twin (16 -> 42/43, 38 -> 41, 39 -> 40).
  These are never *incorrect*, so nothing else catches them; the drafter shipped
  0.429 GiB of avoidable double-store this way for months.
- **Generation smoke** — load, generate, check a known answer. Do not leave a
  model swap before a health check AND a real generation have returned.

## 5. What executing it caught that reading it did not

Every one of these was present in the first written version of this document and
invisible until the pipeline was actually run. They are listed because they are
the argument for the script over the prose.

1. **The REAP stage was missing entirely.** Following the doc produced an
   unpruned 256-expert artifact ~16 GB too large, with no gate to say so.
2. **The drafter type flags were attached to the wrong tool.**
   `dspark_type_flags.txt` holds `--tensor-type` overrides for the QUANTIZER;
   the doc hung them on `build_dspark_template.py`, which does not take them.
   Building the drafter with default types is the 2026-08-01
   `has type bf16, expected f32` load failure.
3. **The Python environment never migrated off the decommissioned LXC.**
   `trim_reap.py` died on `ModuleNotFoundError: numpy` at the first stage. The
   tools need `/home/claude/.venvs/pulsar-quant`, not system python3.
4. **Disk sequencing needed two deletion points, not one.** Dropping only the
   102 GB intermediate still left compact + drafter + merged + final = 277 GB on
   a 296 GB disk, which hit ENOSPC partway through the final copy.

## 4. Known reproduction gaps

- No one has completed a full source-to-artifact rebuild against this document.
  Until that happens, §2 is reconstructed rather than verified.
- The tokenizer bootstrap (§1) means "from source weights alone" is not strictly
  achievable today. Removing it needs an HF-tokenizer converter.
- `~/Projects/ds4-quant/` (the decommissioned LXC rescue: legacy monolith, quant
  recipes, mse baselines) is a git repo with **no remote**, on one disk. Its
  `gguf-tools/` is a 2026-07-02 snapshot and must not be used to build —
  see [`ds4-quant-lxc`] in memory and the MANIFEST in that repo.

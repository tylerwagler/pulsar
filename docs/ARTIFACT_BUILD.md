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

**Fix worth making:** record the imatrix sha256 instead of its path. A path says
where someone's disk was; a hash says which imatrix it was, and would make the
artifact byte-reproducible rather than byte-reproducible-modulo-one-KV.

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

The build proper, given a map:

    1. build_main_template.py --hf DIR --out T.gguf --tokenizer-from-hf     5 MB
    2. deepseek4-quantize --hf DIR --template T.gguf --format-map MAP       102 GB
         [--imatrix ...] --threads N      -> FULL 256-expert intermediate
    3. reap/trim_reap.py --oracle INTERMEDIATE --out COMPACT                 86 GB
         stamps reap25-lcb50-survivors.json: expert tensors dense-trimmed to
         the survivors, router/bias stay PADDED to 256 with pruned bias slots
         set to -1e30 so they can never win top-k (see §3 and the tool header)
    4. build_dspark_template.py     (drafter, from the checkpoint's mtp.* block)
         with the exact-name --tensor-type overrides in dspark_type_flags.txt
    5. merge_dspark_gguf.py         (main bytes preserved verbatim)           92.5 GB
    6. repack_iq2_mmq.py            (IQ2_XXS 16 -> IQ2_XXS_MMQ 43)            92.5 GB

**Step 3 was missing from the first version of this document** — found on
2026-08-12 by actually running the pipeline. REAP is a POST-quantize transplant,
not part of quantization. Following the old text produced an unpruned 256-expert
artifact ~16 GB too large, with no gate to say so.

**Cost.** ~383 GB written for a 92.5 GB artifact (4.1x amplification), three
full-size intermediates, and a disk peak that does not fit unless the 102 GB
intermediate is deleted immediately after step 3. Steps 3, 5 and 6 are each a
full-artifact copy; step 6 is a pure permutation the quantizer cannot emit
directly only because `DS4Q_TYPE_*` stops at 42 while the engine's
`PULSAR_TENSOR_IQ2_XXS_MMQ` is 43. Collapsing 3/5/6 into the quantizer would take
this to template -> quantize; each collapse is byte-exactly verifiable against a
reference artifact.

**Every stage here fails silently if skipped.** Missing step 6 costs 2.4x on the
MMQ path with nothing going red; the drafter shipped 0.43 GiB of double-store for
months because a repack was never run. Prefer fewer stages over better discipline.

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

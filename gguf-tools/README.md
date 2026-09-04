# DS4 GGUF Tools

This directory contains the offline tools used to build and evaluate DeepSeek
V4 Flash GGUF files for `ds4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the DS4 scalar quant formats:
  `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ds4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the DS4 runtime:

```sh
./pulsar \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this DS4 quantizer.  Generic GGUF
tools need DS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `pulsar --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the DS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Repack IQ2_XXS To IQ2_XXS_MMQ (type 43)

Type 43 (`iq2_xxs_mmq`) is a load-aligned twin of `IQ2_XXS`: the same
66 B/block content, permuted so the vendored llama.cpp MMQ kernels can read it
with full-width aligned loads — `d` plane first (2 B/block), pad to 64 B, then
a 64 B-aligned `q` plane.  (An earlier twin, type 42 `iq2_xxs_soa`, fed the
dp4a MoE kernels; both the kernels and its repack scripts have been removed,
and the engine rejects type 42 at load.)

Type 43 is the layout `repack_iq2_mmq.py` builds offline, so storing it in
the GGUF is what lets MMQ read routed experts with full-width aligned loads
instead of `LDG.E.U16` pairs.  Microbenched at the production shape (K=4096,
M=2048, 192 experts, 6 used, 4096 tokens): raw blocks 44.704 vs aligned SoA
18.654 ms/pair-call, a 2.40x.

The artifact is exactly the same size as the raw stream
(`align_up(nblk*2,64) + nblk*64 == nblk*66` whenever `nblk % 32 == 0`, which
holds for every shipped expert stack), so the repack costs **zero** model
growth and leaves every tensor's data offset untouched.  That is what replaces
the runtime repack cache, which was capacity-bound (~22.9 GiB budget against
~35 GB to align all 90+ stacks) and made the first prefill frontier pay for the
repack.

```sh
# Standalone repack for an existing type-16 GGUF. The main artifact rebuild
# no longer calls this: deepseek4-quantize emits type 43 directly from the
# format map (see build/rebuild_collapsed.sh).
python3 gguf-tools/repack_iq2_mmq.py in.gguf out.gguf
python3 gguf-tools/verify_iq2_mmq_model.py in.gguf out.gguf manifest.txt
```

`repack_iq2_mmq.py` converts every 3-D `IQ2_XXS` routed-expert tensor by
default; `--match` narrows it to tensors whose names contain one of a
comma-separated list of substrings, and gate and up must be converted together
(the fused gate+up kernels read ONE layout, so the binder rejects a mixed
pair at load; `down` is independent).  It asserts per tensor that the
aligned size equals the raw size and refuses to grow the file — that assertion
is the load-bearing check that the dims were read correctly.

`verify_iq2_mmq_model.py` checks the header (same size, same offsets, only
16 -> 43) and byte-compares every region the repack was not supposed to touch,
then writes a manifest of the converted spans.  There is no GPU-side
verifier: the raw-block kernels the SoA kernels were once checked against went
with the runtime repack cache, so the comparison has nothing to compare.

**Measured caveat, and it is a real one** (taken while the raw-block kernels
still existed).  On this adapter `ds4_mmq_iq2_xxs_moe_soa(aligned)` is *bit-identical*
to `ds4_mmq_iq2_xxs_moe(raw)` and to `ds4_mmq_iq2_xxs_moe_pair(raw)`, but
`ds4_mmq_iq2_xxs_moe_pair_soa(aligned)` is **not** — it differs from all three
by up to 1.43e-06 absolute on ~74% of outputs at the production shape.  Since
the single-tensor SoA kernel reproduces the raw answer exactly from the *same*
aligned bytes, the artifact is correct and the gap lives in `moe_pair_soa`'s own
accumulation schedule.  Note what that implies for the runtime cache it
replaces: that cache routed whichever gate/up tensors happened to fit through
`moe_pair_soa` and the rest through `moe_pair`, so the current build's logits
already depend on which tensors won the cache.  Pre-storing type 43 makes the
choice uniform and deterministic instead.

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Q4 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

Useful checks before writing a full model:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real DS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 gguf-tools/quality-testing/collect_official.py
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf gguf-tools/quality-testing/data/manifest.tsv /tmp/model.tsv 4096
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

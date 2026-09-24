# safetensors_lane -- the checkpoint the engine serves

This is the tool that emits the served artifact: a per-layer-shard safetensors
checkpoint in the source model's HF scheme, with every payload copied VERBATIM
under a declared layout id.

```sh
python3 safetensors_lane.py plan   --gguf MODEL.gguf
python3 safetensors_lane.py emit   --gguf MODEL.gguf --out DIR --all
python3 safetensors_lane.py verify --gguf MODEL.gguf --out DIR --all
python3 safetensors_lane.py audit  --out DIR
```

`emit --all` writes one shard per layer plus the vision tower, the head and
each drafter layer (`model-000NN-of-000MM.safetensors`; 48 for the 43-layer
0731 artifacts, 45 for the 40-layer V4.1) plus `model.safetensors.index.json`.
`verify` is the stage-1 gate and `audit` is the directory-level one; both exit
non-zero on a finding.

**EXL3 experts (L245).** `plan`/`emit`/`verify` take `--exl3-experts DIR
[--exl3-layers 5,18-22]`: the routed experts of those blk layers are copied
VERBATIM from an EXL3 checkpoint's HF shards (exllamav3's format; the public
Mia-AiLab V4.1 build or our own convert output) instead of the GGUF, one
contiguous `[trellis | suh | svh]` slice per expert under the layout the
trellis width names (`exl3m_k2` 32 words, `exl3m_k2h` 40, `exl3m_k3` 48; the
mul1 codebook is required, an `mcg` tensor is refused).  The byte model is
`src/engine/exl3_trellis.h`'s `exl3_expert_layout`, and the engine re-derives
`expert_bytes` from it at load and refuses a stack that disagrees.  `verify`
checks those experts against the EXL3 shards, not the GGUF.  A partial
download holding just the layers under test is enough (`--exl3-layers`).

## The contract, and why each part is not negotiable

**Payloads are verbatim; the layout is declared.** Every engine payload is a
KERNEL READ PATTERN, not a container choice -- `iq2_xxs_mmq_k` alone measures
218 vs 88 GB/s -- and there is no load-time rebuilder for any of them (the
aligned-IQ2 repack cache was deleted 2026-08-15). So they are written as `U8`
and each shard declares, per tensor, the `layout`, the logical dims in `dims_ne`
(ne order: `dim[0]` is the input dimension) and the engine's `gguf_name`. Only a
tensor with no engine-specific ordering keeps a native dtype.

**The expert geometry is part of the contract, not a hint.** The kernels index
experts as `base + xid*expert_bytes` from ONE pointer, so a projection's
per-expert tensors must be back-to-back with no inter-expert padding:
`offset(E) == offset(0) + E*expert_bytes`, asserted per projection by `verify`.
`pulsar.experts` declares `n_experts`/`expert_bytes`/`contiguity` for each.

**The data buffer is gap-free and alignment comes from ORDER.** safetensors
requires the buffer to tile `[0, end)` with no gaps; the reference reader
rejects a tensor whose offset is not exactly the previous tensor's end
(`invalid offset`). The first version of this writer padded *between* tensors to
force 32-byte alignment -- every one of our own gates passed, and the reference
library rejected all 48 shards. So tensors are placed by DESCENDING alignment
requirement: each then lands on its own alignment for free, and the sort is
stable so a projection's per-expert run stays adjacent.

**`pulsar.expert_dtype` is deliberately not `"fp4"`.** prismaquant's
`declared_fp4_expert_dtype` would turn on its MXFP4 nibble decode for any 2-D
int8/uint8 expert tensor -- which is exactly what our `U8` IQ2 superblocks look
like.

## Where the names come from

`../quantize/dsq_names.c`, in THIS tree, **parsed rather than imported** so a
second copy cannot drift. `plan` audits coverage before anything is written,
because a name that maps to nothing is a hole in the checkpoint and finding it
after writing 92 GB costs a rebuild. Both `emit` and `verify` refuse to proceed
on an unmapped tensor.

## What is verified

Against `v5-vexp-full256-iq2-t46.gguf` (the source), all 48 shards:

* **`emit` reproduces the delivered artifact byte for byte** -- every shard's
  sha256 matches `/srv/models/vexp-safetensors/`, which is the strongest
  statement available: it covers payloads, ordering, alignment, the header and
  the whole `__metadata__` at once.
* **`verify`** checks each shard against the GGUF DIRECTLY (not against any
  intermediate dump): structure (32-byte header, exact size, gap-free), each
  tensor's alignment for its OWN requirement, every non-expert payload
  byte-identical, and each projection's 256 per-expert payloads reassembled
  byte-identical to the GGUF's stacked tensor.
* **`audit`** checks the 32-byte header rule, the required metadata keys, and
  the index closed BOTH ways: nothing indexed that its shard does not declare,
  and nothing on disk without a correct index entry.

## Status, honestly

The engine reads this container and nothing else (`src/engine/model.cpp`
detects a GGUF only to refuse it by name). This tool is what emits it.

**The quantizer still writes an intermediate GGUF.** `deepseek4-quantize`
(`../quantize/`) takes an HF source plus a GGUF *template* and writes a GGUF;
this lane then re-containers it. Making the quantizer emit shards directly is
the remaining piece, and what it needs is one thing: a GGUF-KV -> JSON encoder,
so the template's architecture and tokenizer metadata can be written as
`pulsar.kv` instead of copied as a GGUF KV block. Everything else it needs
already exists -- the tensor plan (`output_context`), the payload generator
(`generate_tensor`), and the names (`dsq_names.c`).

Until then this lane is also what makes the delivered artifact **reproducible
from a clean clone**, which it was not while these scripts lived outside the
repo.

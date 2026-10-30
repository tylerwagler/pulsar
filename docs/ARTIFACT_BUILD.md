# Building a serving artifact from the HF checkpoint

The engine serves a safetensors container whose header declares every tensor's
storage layout.  `tools/container/build.py` builds that container directly from
the HF checkpoint(s): no intermediate format, no GGUF (the GGUF-era pipeline is
archived at tag `archive/gguf-tooling-2026-09-24`; L247 replaced it).  The
builder is pure Python (numpy for the codecs) and runs anywhere the checkpoints
are mounted; it never needs a GPU.

The container's tensor names are the HF checkpoint's names.  The engine reads
four metadata keys (`src/engine/safetensors.cpp`): `pulsar.kv`
(the model hyper-parameters and the tokenizer, on the first shard),
`pulsar.kv_arch` (the architecture keys, on every shard, checked against the
first shard's copy), `pulsar.tensors` (per dense tensor: `layout`, `dims_ne`,
`gguf_name`) and `pulsar.experts` (per routed-expert stack: layout,
`expert_bytes`, count).  It binds tensors by `gguf_name` (a historical
spelling of the engine's binding name; nothing GGUF is read).

Families (`build.FAMILIES`, by the checkpoint's `model_type`): DeepSeek V4
Flash / Vision-Exp and V4.1 (`deepseek.py`), Qwen3.8-Flash-Next (`qwen.py`,
`qwen4_exp`), Inkling (`inkling.py`), Nemotron-H / Nemotron 3.5 Super
(`nemotron_h.py`), the Nemotron-3.5 Embed / Rerank encoders
(`nemotron_enc.py`) and Nemotron-3.5 Parse (`nemotron_parse.py`).  Every
family but DeepSeek takes its recipe from
[tools/container/format-maps/](../tools/container/format-maps/README.md).

## Inputs

| input | what it supplies |
|---|---|
| `--hf DIR` | the HF checkpoint (source `hf`): `config.json`, the shards, the tokenizer files.  Every dense tensor, the tower, the drafter, and the routed experts a recipe row reads from it (DeepSeek: the QAT FP4 source, I8 nibbles + E8M0 scale). |
| `--recipe JSON` | a `pulsar.recipe.v1` (`tools/container/recipe.py`): every emitted tensor's format and source, by ROLE (optionally a block and a layer set) or by HF-name pattern; exactly one row per tensor, no dead rows.  Required for every family but DeepSeek (`format-maps/*.json`).  DeepSeek: optional; the default is generated from `policy.py` and the EXL3 source (`build.py recipe` prints it -- edit it to choose which layers take EXL3 experts). |
| `--exl3 DIR` | an exllamav3 EXL3 checkpoint (source `exl3`; MIT; the MiaAI V4.1 quantization is public).  DeepSeek's default recipe takes the routed experts of every main layer it holds (`layers.N.ffn.experts.E.{w1,w2,w3}.{trellis,suh,svh}`) at the rate the trellis holds; a recipe row may also name the drafter's (`block: mtp`).  The rate is the recipe's, the source must match, and the engine arm the role runs must read it (`exl3_rates.admit` == `exl3_arm_has_rate`). |
| `--exl3-experts DIR` | a second exllamav3 checkpoint (source `exl3_experts`; the default source of a routed expert's EXL3 rows when given). |
| `--tessera DIR` | a Tessera bundle (source `tessera`; `<stack>.pt` unit-blob files).  The format and the producer-table slot exist; the producer is not on dev and refuses by name. |
| `--drafter KIND DIR` | a separate block-drafter checkpoint written into the container as shard `drafter` (`drafters.py`): `lilicorr` on Nemotron-H (NVIDIA's LiLiCorr, ModelOpt NVFP4), `dflash2` on Qwen3.8-Flash-Next (z-lab's DFlash2, BF16). |
| `--ple-rows MANIFEST` | Qwen3.8-Flash-Next: the PLE n-gram row file's manifest (`tools/container/ple_rows.py build`).  The row file (~102 GB) is served from local NVMe beside the container. |
| `--layers L,L,...` | DeepSeek: a layer-subset fixture keeping these source layers, renumbered densely (test fixtures). |
| `--engram-layout JSON` | DeepSeek V4.1: the Engram hash layout (`tools/engram/engram_layout.py`). |
| `--reap-map JSON` | a REAP expert-survivor map written into `pulsar.kv`. |
| `--mxfp8-scale rederive\|verbatim` | how an FP8 dense weight's per-32 E8M0 scale is formed: `rederive` (per-32 amax; byte-identical to the archived C codec and to the served Vision-Exp artifact) or `verbatim` (the source's 128-block scale broadcast).  Default `rederive`. |
| `--tokenizer DIR` | tokenizer files if not beside the checkpoint. |

## Commands

```
PY=python3                                                   # any python with numpy
$PY tools/container/build.py recipe --hf $HF [--exl3 $EXL3] [--recipe R]           # the pulsar.recipe.v1 the build runs (DeepSeek: the generated default)
$PY tools/container/build.py plan   --hf $HF [--exl3 $EXL3] [--dump F]             # seconds: shards, per-shard layout census, total bytes;
                                                                                   #  --dump: every header + source descriptor (the byte-identity instrument)
$PY tools/container/build.py emit   --hf $HF [--exl3 $EXL3] --out $DIR --all       # or --shard layers.7 / vision / top / mtp.2
$PY tools/container/build.py verify --hf $HF [--exl3 $EXL3] --out $DIR --all       # re-produces every payload from the SOURCE and compares
$PY tools/container/build.py audit  --out $DIR                                     # declarations and entries closed both ways
```

`emit` streams: the header is fixed from the byte model before any payload is
read, every entry is copied (native spans, EXL3 ranges) or produced straight
into the file, and `--all` writes `model.safetensors.index.json` at the end.
Emitting one shard at a time and moving it (`rsync` over ssh, never a stat on an
in-flight NFS copy) is the way to build onto a remote model store.

## The decisions (DeepSeek's default recipe: `tools/container/policy.py`)

The recipe decides every format (L279).  DeepSeek's default recipe is today's
policy, checked against the checkpoints' own shard headers: five decisions,
everything else keeps the source dtype (`native`: `bf16`, `f32`, `i32`).  A
DeepSeek recipe may restate these for a non-expert tensor, never change them
(the loader binds nothing else); routed experts take `cutlass_mxfp4` or EXL3.

1. FP8 (E4M3) 2-D dense weights -> `mxfp8_lt`, the `.scale` companion folds in.
2. Routed experts -> the expert SOURCE's layout: the QAT FP4 source -> `cutlass_mxfp4`; an EXL3 trellis names its own rate by its words-per-tile (32/40/48 -> `exl3m_k2/k2h/k3`).
3. The drafter's `markov_w2` -> `fp8_e4m3_soa_k`, transposed k-major (L213; the one lossy-by-design row).
4. `ffn.gate.tid2eid` I64 -> `i32` (the one dtype narrowing; every value is checked to fit).
5. The declared shape is what the container holds: `[1, n]` -> `[n]` (the confidence head), `markov_w2` -> `[256, V]`; nothing else reshapes.

A source dtype the engine has no layout for refuses.  There is no fallback
format anywhere in the builder.

## Modules

| file | role | proven by |
|---|---|---|
| `hf_source.py` | header-only readers: `HFCheckpoint` (`names/shape/dtype/span/raw/copy_src`, `X#i` = slice i of a stacked tensor, `config` with `text_config` merged), `Exl3Checkpoint` (`names`, `linear(key, k, n, rates)` with every refusal the format allows; the family names the key -- `names.exl3_expert_key` / `exl3_layers` for DeepSeek) | used by every test below |
| `names.py` | DeepSeek's naming table: HF name -> container entry (`Mapped`: container name, `gguf_name`, role, family, shard, expert/part, `emit`); `ModelShape.from_config`; the shard plan; the EXL3 expert keys | `test_names.py` |
| `policy.py` | DeepSeek's default formats (`layout_for`) and declared shapes (`declared_shape`) | `test_names.py` |
| `deepseek.py` / `qwen.py` / `inkling.py` / `nemotron_h.py` / `nemotron_enc.py` / `nemotron_parse.py` | the family modules (registered by `model_type`, `build.FAMILIES`): the source view (`source`; Inkling's names each projection its fused w13 / expert stacks hold), naming, shard order, what the loader admits, the default recipe, expert grouping, the arch KV.  Inkling also declares its shape (refusing a config or tensor that disagrees) and writes each tensor's bind role into its record | `test_<family>.py` |
| `drafters.py` | a separate block-drafter checkpoint as shard `drafter` (`--drafter lilicorr\|dflash2 DIR`) | `test_nemotron_h.py`, `test_qwen.py` |
| `ple_rows.py` | Qwen's PLE n-gram table as one disk row file (`build` / `verify`) | its `verify` subcommand |
| `tokenizer/core.py` | the `tokenizer.*` kv entries, one core for every family | `test_kv.py` and the family tests |
| `recipe.py` | `pulsar.recipe.v1`: rows by role / block / layers or name pattern; exactly one row per tensor, no dead rows | `test_deepseek.py`, `test_qwen.py`, `test_inkling.py` |
| `kv.py` | the `pulsar.kv` block from `config.json` + tokenizer files, in the engine's spelling | `test_kv.py` |
| `producers.py` | the codecs, bytes in / bytes out (`mxfp8_lt`, `mxfp8_lt_from_bf16`, `cutlass_mxfp4`, `nvfp4`, `nvfp4_from_bf16`, `fp8_e4m3_soa_k_from_bf16`, `f16_from_f32`, `i64_to_i32`, `rows_gather`, `i32_values`; `rows_strided`, a source view's verbatim rows), the ONE producer table `PRODUCER_FOR` ((source kind, format) -> producer), `bytes_for` (== `st_bytes_for` / `routed_expert_side_layout`) | `test_producers.py` |
| `entries.py` | one entry's payload through the producer table (dense, a routed expert, an EXL3 Linear's ranges) | the synthetic goldens |
| `exl3_rates.py` | the EXL3 rate table and the engine's arms (`exl3_arm_has_rate`) by role | `test_exl3_rates.py` (in host-checks) |
| `build.py` | the one plan; recipe / plan / emit / verify / audit | the shard comparison below; `test_deepseek.py` / `test_qwen.py` / `test_inkling.py` GOLDEN SHA-256s |

## The instrument

The oracle was the then-served Vision-Exp artifact (the container published as
`ElytronAI/DeepSeek-v4-Flash`), which the archived lane built from the same HF checkpoint.  The builder must
reproduce it, and the comparison is entry-for-entry, not file-for-file: payload
bytes by tensor name, declarations (dtype/shape), and the three `pulsar.*`
blocks.  Result on 2026-09-24 (L247): every one of the 36,909 dense entries and
every expert stack byte-identical, with three header differences decided
against the old spelling because the engine is the authority for each:

- native layouts are declared by their dtype name (`bf16`/`f32`/`i32`), the loader's own name table; the lane wrote `native`;
- the seven GGUF bookkeeping keys (`general.file_type`, `general.quantization_version`, `deepseek4.expert_gating_func`, `quantize.imatrix.*`) are not written -- nothing in `src/` reads them;
- entry order inside a shard follows the HF walk; the loader parses entries in any order.

The header comparison is not optional: it found two declarations the engine
would have refused at load (the confidence head's rank, the k-major
`markov_w2` dims) while every payload byte already matched.  `test_names.py`
grades `dims_ne` against the served artifact for that reason.

The tests need numpy; `make container-tests` runs every `test_*.py` (it is
part of `make host-checks`).

## What the artifact directory holds

`model-NNNNN-of-MMMMM.safetensors` (shard 1 = the vision tower, then one per
layer, `top`, one per drafter layer), `model.safetensors.index.json`, and
`config.json` copied from the source for provenance.  The engine needs only
the shards: the tokenizer rides in `pulsar.kv`.

## Known limits

- V4.1's tower lacks `image_pad`; the V4.1 shape profile declares an unpadded image span and the engine's image path refuses it (`shape_profiles.cpp`, `session.cpp`).  The builder emits what the checkpoint has.
- Engram (V4.1) row tables stay side files (L242); `wkv/q/k` are ordinary layer-shard tensors.
- No IQ2 producer: the type-44 permutation lived in the archived `repack_iq2_mmq.py`; a recipe row naming IQ2 refuses.  The GGUF-era `format-maps/v5mx4-format-map.json` (91 IQ2 rows) is no longer read by anything (its re-key path left with `--format-map`, L279); building IQ2 or retiring the map is L279's step 7.
- DeepSeek EXL3 dense refuses by name: the DeepSeek loader admits its dense linears at the MX slot only (weights.cpp `tensor_expect_mxfp8`); the EXL3 dense arm reads bf16 rows.

# tools/container — the direct builder: HF checkpoint(s) → pulsar's safetensors container

No GGUF anywhere (L247; the GGUF-era tooling is archived at tag `archive/gguf-tooling-2026-09-24`).  The engine
reads four metadata keys from the container -- `pulsar.kv`, `pulsar.tensors`, `pulsar.experts`, and
`pulsar.kv_arch` (every shard's copy of the architecture keys, checked against the primary shard; L298)
(`src/engine/safetensors.cpp`) -- and binds tensors by the `gguf_name` they carry (the name is a historical
spelling; nothing GGUF is read).  The container's tensor names ARE the HF checkpoint's names; the format producers
are pure bytes-in / bytes-out codecs.  The driver is `build.py`; how to run it and how a build is proved is
[docs/ARTIFACT_BUILD.md](../../docs/ARTIFACT_BUILD.md).

Families (build.py `FAMILIES`, keyed by `config.json` `model_type`): `deepseek_v4` / `deepseek_v41`
(`deepseek.py`), `qwen4_exp` (`qwen.py`), Inkling (`inkling.py`), Nemotron-H (`nemotron_h.py`), the Nemotron-3.5
retrieval encoders (`nemotron_enc.py`, L321) and Nemotron-3.5 Parse (`nemotron_parse.py`, L321).  Recipes live in
[format-maps/](format-maps/README.md).  The Python unit tests are `test_*.py` beside the modules
(`make container-tests`).

## Module contract (the interfaces the modules are written against)

```
hf_source.py                            # the source readers, family-neutral, header-only until bytes are asked for
  class HFCheckpoint(hf_dir)            # .names(), .shape(n), .dtype(n), .span(n), .raw(n) -> bytes, .copy_src(n),
                                        #  .config (text_config merged), .generation_config; `X#i` = slice i of stack X
  class Exl3Checkpoint(dir)             # .names(), .linear(key, k, n, rates) -> (ranges, words); the family names the key
  class TesseraBundle(dir)              # .names(): <stack>.pt unit-blob files (its producer is not on dev; see below)

recipe.py                               # pulsar.recipe.v1 (module docstring): every family's recipe
  class Recipe                          # .load(path, settings); .resolve(mapped) -> {name: (format, source)}:
                                        #  exactly one row per emitted tensor, no dead rows; rows by role (+ block, layers)
                                        #  or HF-name pattern; formats = layouts | native | tessera | ple_rows / kv / omit
  def rows_for(mapped, assign) -> rows  # the fewest role rows that say a {name: (format, source)} (default recipes)

names.py                                # DeepSeek's naming table
  @dataclass Mapped:                    # shared by every family's table
      container_name: str               # the file key = the HF name (experts: one U8 per expert-projection)
      gguf_name: str                    # what the engine binds: blk.N.*, dspark.N.*, token_embd.weight, ...
      family: str                       # 'top' | 'layer' | 'vision' | 'mtp' | 'expert'
      shard: str                        # 'vision' | 'layers.N' | 'top' | 'mtp.N'
      layer: int | None; expert: int | None; part: str | None   # part in {'w1','w2','w3'}
      is_scale: bool                    # 'X.scale' pairs with 'X.weight' (never emitted on its own)
      emit: bool                        # False = a drop rule (consumed, never written)
      role: str                         # the recipe's vocabulary: dense | shared_expert | expert_gate / _up / _down |
                                        #  router | route_table | norm | embed | head | draft_head | vision | other
  def map_hf(hf_name: str, shape: ModelShape) -> Mapped | None      # None = not a model tensor (refuse, never skip silently)
  ModelShape = dataclass(n_layer, n_mtp, has_vision, v41: bool, n_hash: int,     # ModelShape.from_config(config.json)
                         keep: tuple | None, drafter: bool)  # .subset(keep, anchors): a layer-subset fixture (--layers)
  def shard_order(shape) -> [str]; shard_file(shape, shard) -> str               # the shard plan
  def exl3_expert_key(block, layer, e, part); exl3_layers(names)                 # DeepSeek's EXL3 keys

policy.py                               # DeepSeek's default formats (the five decisions); a PolicyError is a refusal
  def layout_for(m: Mapped, dtype: str, shape: list[int]) -> str  # native dtypes stay native; F8_E4M3 2-D -> mxfp8_lt;
                                        #  FP4 experts -> cutlass_mxfp4; markov_w2 -> fp8_e4m3_soa_k; tid2eid -> i32
  def declared_shape(m: Mapped, shape: list[int]) -> list[int]   # decision 5: what the container HOLDS; dims_ne = reversed

deepseek.py / qwen.py / inkling.py      # the family modules (build.FAMILIES, by config.json model_type): META, SETTINGS,
                                        #  EXPERT_STACKS, GATE_UP_PARTS, source (the HF view the plan reads), shape,
                                        #  shard_order, map_hf, declared_shape, admit (what the loader binds),
                                        #  default_recipe, group_* / stack_families, gate_up_key, build_kv.
                                        #  inkling.Source names the projections Inkling's fused w13 / expert stacks hold
                                        #  (the EXL3 pack's names) and checks names, dtypes, shapes against the declared
                                        #  Shape both ways; Mapped.bind carries the role table into pulsar.tensors
nemotron_enc.py / nemotron_parse.py      # the Nemotron-3.5 Embed / Rerank encoders and the Parse model (L321)
nemotron_h.py                           # Nemotron-H (the Omni checkpoint); with_drafter(hf, kind, dir) is build.py's
                                        #  --drafter lilicorr DIR (L298), through drafters.py; the recipe omits the MTP
drafters.py                             # the block-drafter input, every family's (L298 / L299): with_drafter(hf, kind,
                                        #  dir, Target) -> a view with the drafter's tensors as `drafter.<HF name>` in
                                        #  shard `drafter`, bind blocks drafter + the chain head's (lilicorr | selector),
                                        #  kv pulsar.drafter + the drafter config verbatim under `<kind>.`; every config
                                        #  key written or ignored-with-a-reason, the rest refused.  lilicorr: NVIDIA's
                                        #  ModelOpt NVFP4 checkpoint (Nemotron-H); dflash2: z-lab's BF16 DFlash2 (qwen.py
                                        #  --drafter dflash2 DIR), its head the target lm_head's draft-vocabulary rows
                                        #  (rows_gather + their ids, i32_values) and its Linears nvfp4_from_bf16 by recipe

producers.py                            # every producer: bytes in, bytes out, no I/O; each gated byte-identical
  PRODUCER_FOR[(source kind, format)]   # THE producer table; producer_for(kind, fmt, what) refuses a missing pair by name
  spec(producer, inputs, **args); produce(desc, source)          # a plan entry's serializable producer descriptor
  def mxfp8_lt(w_fp8, scale_e8m0, out, inp, block, mode) -> bytes   # 'rederive' (archived codec) | 'verbatim'
  def mxfp8_lt_from_bf16(w_bf16, out, inp) -> bytes
  def cutlass_mxfp4(w_i8, scale_e8m0, out, inp) -> bytes            # per-expert [data | SF]
  def nvfp4_from_bf16(w_bf16, out, inp) -> bytes    # L299: our NVFP4 (global amax/2688, per-16 E4M3 scale by
                                                     #  least squared error over the max-based code and 4 below)
  def nvfp4(w_u8, scale_e4m3, scale2_f32, out, inp) -> bytes        # ModelOpt NVFP4 Linear: [nibbles | 128x4-tiled
                                                                     #  E4M3 SF | f32 scale_2 + 28 zero B]; in % 64
  def fp8_e4m3_soa_k_from_bf16(w_bf16, rows, cols) -> bytes          # markov_w2, k-major
  def i64_to_i32(raw) -> bytes                                       # tid2eid; refuses a value outside int32
  def tessera_planes(...)                                            # the (tessera, tessera) slot: refuses (producer not on dev)
  def bytes_for(layout, dims_ne, dtype=None) -> int                  # == st_bytes_for / routed_expert_side_layout

ple_rows.py                             # qwen4_exp: the PLE n-gram table as one disk row file (`build` | `verify`);
                                        #  build.py --ple-rows MANIFEST names it in the container
tokenizer/core.py                       # the tokenizer.* kv entries, one core for every family (L279 step 6)
entries.py                              # one entry's payload through the producer table (dense, expert, EXL3 ranges)
exl3_rates.py                           # K2 (layout <-> rate), ARMS == exl3_arm_has_rate, ROLE_ARM, admit(role, layout)

kv.py
  def build_kv(hf, tokenizer_dir, reap_map=None, keep=None, engram_layout=None) -> list[dict]
      # [{'key','type','value'}] in the engine's spelling: types u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool string array;
      # arrays as {"__array__": elem_type, "n": N, "v": [...]}; every f32 rounded through struct.pack('<f') first.
      # Emits the keys the engine CONSUMES (src/engine/weights.cpp and the family loaders;
      # tokenizer.ggml.tokens / merges; drafter deepseek_v4_dspark.embedding_length + dspark.target_layer_ids.N).

build.py                                # the driver: `recipe` | `plan` | `emit` | `verify` | `audit` (docs/ARTIFACT_BUILD.md)
  plan(hf, fam, ctx) -> (shape, order, files, shards, consumed)   # the ONE plan, every family
      # per shard: entries (byte ranges or a producer spec), pulsar.tensors / pulsar.experts, pulsar.kv on shard 1
  write_dump(...)                       # plan --dump: every shard's exact header + every entry's source descriptor
  write_shard(path, entries, meta, hf)  # streamed: header fixed from the byte model, payloads copied/produced into the file
  write_index(out_dir)                  # model.safetensors.index.json from the shard headers
  cmd_verify                            # vs the SOURCE: native spans byte-equal, producers re-run and compared, EXL3 verbatim
  cmd_audit                             # structure + declarations closed both ways
```

## History

The builder replaced the GGUF pipeline in L247 (2026-09-24): its first instrument reproduced the served
DeepSeek-V4-Flash Vision-Exp artifact entry-for-entry (docs/ARTIFACT_BUILD.md), then the V4.1 EXL3 artifact,
the vision tower, the drafters and Engram followed.  L279 made it one plan, one producer table and one recipe
schema (`pulsar.recipe.v1`) for every family.

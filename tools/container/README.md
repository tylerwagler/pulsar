# tools/container — the direct builder: HF checkpoint(s) → pulsar's safetensors container

No GGUF anywhere (Tyler, 2026-09-24; L247).  The engine reads exactly three
metadata keys from the container -- `pulsar.kv`, `pulsar.tensors`,
`pulsar.experts` (`src/engine/safetensors.cpp`) -- and binds tensors by the
`gguf_name` they carry.  The container's tensor names ARE the HF checkpoint's
names (DeepSeek's native namespace); the format producers are pure
bytes-in / bytes-out codecs.  This directory is the driver and the writer
that replace the header-only GGUF template + `deepseek4-quantize` + the lane's
GGUF reader.  Read `pulsar-notes/research/direct-container-builder-2026-09-24.md`
for the file:line evidence behind every statement here.

## Module contract (the interfaces the modules are written against)

```
hf_source.py                            # the source readers, family-neutral, header-only until bytes are asked for
  class HFCheckpoint(hf_dir)            # .names(), .shape(n), .dtype(n), .span(n), .raw(n) -> bytes,
                                        #  .config (text_config merged), .generation_config
  class Exl3Checkpoint(dir)             # .names(), .linear(key, k, n, rates) -> (ranges, words); the family names the key
  class TesseraBundle(dir)              # .names(): <stack>.pt unit-blob files (the producer lands with L255)

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

deepseek.py / qwen.py                   # the family modules (build.FAMILIES, by config.json model_type): META, SETTINGS,
                                        #  EXPERT_STACKS, GATE_UP_PARTS, shape, shard_order, map_hf, declared_shape,
                                        #  admit (what the loader binds), default_recipe, group_* / stack_families, build_kv

producers.py                            # every producer: bytes in, bytes out, no I/O; each gated byte-identical
  PRODUCER_FOR[(source kind, format)]   # THE producer table; producer_for(kind, fmt, what) refuses a missing pair by name
  spec(producer, inputs, **args); produce(desc, source)          # a plan entry's serializable producer descriptor
  def mxfp8_lt(w_fp8, scale_e8m0, out, inp, block, mode) -> bytes   # 'rederive' (archived codec) | 'verbatim'
  def mxfp8_lt_from_bf16(w_bf16, out, inp) -> bytes
  def cutlass_mxfp4(w_i8, scale_e8m0, out, inp) -> bytes            # per-expert [data | SF]
  def fp8_e4m3_soa_k_from_bf16(w_bf16, rows, cols) -> bytes          # markov_w2, k-major
  def i64_to_i32(raw) -> bytes                                       # tid2eid; refuses a value outside int32
  def tessera_planes(...)                                            # the (tessera, tessera) slot: refuses until L255
  def bytes_for(layout, dims_ne, dtype=None) -> int                  # == st_bytes_for / routed_expert_side_layout

entries.py                              # one entry's payload through the producer table (dense, expert, EXL3 ranges)
exl3_rates.py                           # K2 (layout <-> rate), ARMS == exl3_arm_has_rate, ROLE_ARM, admit(role, layout)

kv.py
  def build_kv(hf: HFCheckpoint, tokenizer_dir: str, reap_map: str | None) -> list[dict]
      # [{'key','type','value'}] in the engine's spelling: types u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool string array;
      # arrays as {"__array__": elem_type, "n": N, "v": [...]}; every f32 rounded through struct.pack('<f') first.
      # Emits the keys the engine CONSUMES (weights.cpp:941-965, 1021-1063, 777-813, 874-882, 838-858, 1072;
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

## Order of work (L247)

1. Text stack: names + policy + kv + producers {native, mxfp8_lt, cutlass_mxfp4, exl3} + emit + verify.
   DONE 2026-09-24: the served Vision-Exp artifact is reproduced entry-for-entry (see the instrument in
   docs/ARTIFACT_BUILD.md); the V4.1 artifact with EXL3 experts is graded against
   `/rpool/models/v41-exl3-safetensors` (the last lane emission).
2. Tower: native passthrough of `vision.*`, `aligner.*`, `image_*` -- DONE (shard 1); V4.1 lacks `image_pad`.
3. Drafter: `mtp.*`, the V4.1 `markov_head.{embed,head}` rename, k-major `fp8_e4m3_soa_k` -- DONE.
4. Engram: `wkv/q/k` as ordinary layer-shard tensors; row tables side files named by a kv (with L242 slice 5).
5. Delete: `deepseek4-quantize`, both template builders, `gguf_hdr.py`, the GGUF tools, the lane's GGUF
   reader; `quants_*.c` shrink to the codecs the builder still calls; docs/ARTIFACT_BUILD.md rewritten.

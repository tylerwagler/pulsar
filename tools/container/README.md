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
hf_source.py
  class HFCheckpoint(hf_dir)            # shard headers only; .names(), .shape(n), .dtype(n), .raw(n) -> bytes,
                                        #  .config (text_config merged), .generation_config
  class Exl3Checkpoint(dir)             # moved from the lane: .layers(), .expert(layer, e, part, k, n) -> (ranges, words)

names.py
  @dataclass Mapped:
      container_name: str               # the file key = the HF name (experts: one U8 per expert-projection)
      gguf_name: str                    # what the engine binds: blk.N.*, dspark.N.*, token_embd.weight, ...
      family: str                       # 'top' | 'layer' | 'vision' | 'mtp' | 'expert'
      shard: str                        # 'vision' | 'layers.N' | 'top' | 'mtp.N'
      layer: int | None; expert: int | None; part: str | None   # part in {'w1','w2','w3'}
      is_scale: bool                    # 'X.scale' pairs with 'X.weight' (never emitted on its own)
  def map_hf(hf_name: str, shape: ModelShape) -> Mapped | None      # None = not a model tensor (refuse, never skip silently)
  ModelShape = dataclass(n_layer, n_mtp, has_vision, v41: bool, n_hash: int)     # ModelShape.from_config(config.json)
  def shard_order(shape) -> [str]; shard_file(shape, shard) -> str               # the shard plan

policy.py                               # the five decisions (module docstring); a PolicyError is a refusal
  def layout_for(m: Mapped, dtype: str, shape: list[int], overrides: dict) -> str
      # 'bf16' | 'f32' | 'i32' | 'mxfp8_lt' | 'cutlass_mxfp4' | 'exl3m_k2' | 'exl3m_k2h' | 'exl3m_k3' | 'fp8_e4m3_soa_k'
      # native dtypes stay native; F8_E4M3 2-D dense -> mxfp8_lt; routed experts -> the expert source's layout;
      # markov_w2 -> fp8_e4m3_soa_k; tid2eid I64 -> i32.  overrides: {container-name pattern: layout}.
  def declared_shape(m: Mapped, shape: list[int]) -> list[int]
      # decision 5: what the container HOLDS ([1, n] -> [n]; markov_w2 -> [cols, rows]); dims_ne = reversed
  def rekey_format_map(gguf_map: dict, shape: ModelShape) -> dict   # a prisma map's GGUF names -> container names

producers.py                            # every producer: bytes in, bytes out, no I/O; each gated byte-identical
  def mxfp8_lt(w_fp8: bytes, scale_e8m0: bytes, out: int, inp: int, block: int, mode: str) -> bytes
      # mode 'verbatim' (source block scale broadcast per 32-group) or 'rederive' (today's codec: per-32 amax)
  def cutlass_mxfp4(w_i8: bytes, scale_e8m0: bytes, out: int, inp: int) -> bytes      # per-expert [data | SF]
  def fp8_e4m3_soa_k_from_bf16(w_bf16: bytes, rows: int, cols: int) -> bytes           # markov_w2, k-major
  def i64_to_i32(raw: bytes) -> bytes                                                    # tid2eid; refuses a value outside int32
  def native(raw: bytes) -> bytes
  def bytes_for(layout: str, dims_ne: list[int], dtype: str | None = None) -> int        # == st_bytes_for / routed_expert_side_layout

kv.py
  def build_kv(hf: HFCheckpoint, tokenizer_dir: str, reap_map: str | None) -> list[dict]
      # [{'key','type','value'}] in the engine's spelling: types u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool string array;
      # arrays as {"__array__": elem_type, "n": N, "v": [...]}; every f32 rounded through struct.pack('<f') first.
      # Emits the keys the engine CONSUMES (weights.cpp:941-965, 1021-1063, 777-813, 874-882, 838-858, 1072;
      # tokenizer.ggml.tokens / merges; drafter deepseek_v4_dspark.embedding_length + dspark.target_layer_ids.N).

build.py                                # the driver: `plan` | `emit` | `verify` | `audit` (docs/ARTIFACT_BUILD.md)
  plan(hf, exl3, layers, overrides, mxfp8_mode, tokenizer_dir, reap_map) -> (shape, order, files, shards, consumed)
      # per shard: entries (source ranges or a producer closure), pulsar.tensors / pulsar.experts, pulsar.kv on shard 1
  write_shard(path, entries, meta)      # streamed: header fixed from the byte model, payloads copied/produced into the file
  write_index(out_dir)                  # model.safetensors.index.json from the shard headers
  cmd_verify                            # vs the HF SOURCE: native spans byte-equal, producers re-run and compared, EXL3 verbatim
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

#!/usr/bin/env python3
"""build_dspark_template.py — generate DSpark draft GGUF template (raw binary).

The gguf Python library doesn't support FP8_E4M3 (38) or MXFP4 (39) types, so
this writes the GGUF binary directly.  Output has tensor info entries (zero
weight data) for all DSpark tensors under the dspark.* namespace.

Feed to quantizer (cutlass_mxfp4 = CUTLASS type-40 routed experts, the
tensor-core layout the engine's DSpark path loads; byte-lossless from the QAT
E2M1+E8M0 source):
  gguf-tools/deepseek4-quantize \\
    --hf HF_DIR --template dspark_template.gguf --out dspark.gguf \\
    --experts cutlass_mxfp4 $(cat gguf-tools/dspark_type_flags.txt)
Then splice into the main model with gguf-tools/merge_dspark_gguf.py.

The dspark_type_flags.txt overrides are REQUIRED, but as of 2026-08-15 they no
longer force f32 on the norms and gates. The engine reads bf16 for those now, so
the pin was the wrong half: this template already emitted type 30 natively and
the flags were overriding it back to f32 -- which cost 132 MB on the markov head
alone, and the markov step streams all of markov_w2 every draft position, so it
cost bandwidth too. The overrides that remain are the ones that carry real
information: mxfp8_lt attention and shared experts, cutlass_mxfp4 routed
experts, and f32 for the tensors that genuinely ARE f32 upstream (attn_sinks,
hc_*_base/fn/scale).

Which tensors are which was read from the checkpoint's own mtp.* dtypes, not
assumed -- our dspark.* IS the source's mtp module.

A drafter built with bf16 norms will NOT load on an engine older than that
change; it fails with "has type bf16, expected f32". Gate every drafter build on
a type-parity diff against a known-good drafter before merging, and expect
exactly the 20 bf16 flips if you are comparing across that boundary.
"""
import json, os, struct, sys

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types
VAL_UINT8   = 0
VAL_INT8    = 1
VAL_UINT16  = 2
VAL_INT16   = 3
VAL_UINT32  = 4
VAL_INT32   = 5
VAL_FLOAT32 = 6
VAL_BOOL    = 7
VAL_STRING  = 8
VAL_ARRAY   = 9
VAL_UINT64  = 10
VAL_INT64   = 11
VAL_FLOAT64 = 12

class Buf:
    def __init__(self):
        self.data = bytearray()
    def u8(self, v): self.data += struct.pack('<B', v)
    def u16(self, v): self.data += struct.pack('<H', v)
    def u32(self, v): self.data += struct.pack('<I', v)
    def u64(self, v): self.data += struct.pack('<Q', v)
    def f32(self, v): self.data += struct.pack('<f', v)
    def string(self, s):
        self.u64(len(s))
        self.data += s.encode('utf-8')
    def raw(self, b): self.data += b
    def pad_to(self, align):
        while len(self.data) % align:
            self.u8(0)
    def tell(self): return len(self.data)

def write_gguf(path, kvs, tensors, alignment=GGUF_DEFAULT_ALIGNMENT):
    """Write a minimal GGUF file with tensor info only (no data)."""
    buf = Buf()
    buf.raw(GGUF_MAGIC)       # magic
    buf.u32(GGUF_VERSION)     # version
    buf.u64(len(tensors))     # tensor count
    buf.u64(len(kvs))         # KV count

    # KV records
    for key, typ, val in kvs:
        buf.string(key)
        buf.u32(typ)
        if typ == VAL_STRING:
            buf.string(val)
        elif typ == VAL_UINT32:
            buf.u32(val)
        elif typ == VAL_UINT64:
            buf.u64(val)
        elif typ == VAL_FLOAT32:
            buf.f32(val)
        elif typ == VAL_BOOL:
            buf.u8(1 if val else 0)
        elif typ == VAL_ARRAY:
            elem_typ, items = val[0], val[1]
            buf.u32(elem_typ)
            buf.u64(len(items))
            for item in items:
                if elem_typ == VAL_STRING:
                    buf.string(item)
                elif elem_typ == VAL_UINT32:
                    buf.u32(item)
        else:
            raise ValueError(f'unsupported KV type {typ}')

    # Tensor info records (all offsets 0 — template with no data)
    for name, ndim, shape, ggml_type in tensors:
        buf.string(name)
        buf.u32(ndim)
        for d in shape:
            buf.u64(d)
        buf.u32(ggml_type)
        buf.u64(0)     # offset

    buf.pad_to(alignment)

    with open(path, 'wb') as f:
        f.write(buf.data)

def make_kvs(cfg):
    E = cfg['hidden_size']
    H = cfg['num_attention_heads']
    K = cfg['num_key_value_heads']
    D = cfg['head_dim']
    R = cfg['n_routed_experts']
    F = cfg['moe_intermediate_size']
    eps = cfg.get('rms_norm_eps', 1e-6)
    # The 3 draft sublayers target the last 3 main-model layers; the engine's
    # dspark_weights_bind() reads these KVs and merge_dspark_gguf requires
    # them (a merged artifact without them fails to bind the drafter).
    L = cfg['num_hidden_layers']
    kvs = [
        ('general.architecture', VAL_STRING, 'deepseek_v4_dspark'),
        ('deepseek_v4_dspark.embedding_length', VAL_UINT32, E),
        ('deepseek_v4_dspark.block_count', VAL_UINT32, 0),
        ('deepseek_v4_dspark.head_count', VAL_UINT32, H),
        ('deepseek_v4_dspark.head_count_kv', VAL_UINT32, K),
        ('deepseek_v4_dspark.key_length', VAL_UINT32, D),
        ('deepseek_v4_dspark.expert_count', VAL_UINT32, R),
        ('deepseek_v4_dspark.expert_feed_forward_length', VAL_UINT32, F),
        ('deepseek_v4_dspark.layer_norm_rms_eps', VAL_FLOAT32, eps),
        ('dspark.target_layer_ids.0', VAL_UINT32, L - 3),
        ('dspark.target_layer_ids.1', VAL_UINT32, L - 2),
        ('dspark.target_layer_ids.2', VAL_UINT32, L - 1),
        ('general.file_type', VAL_UINT32, 1),
        ('general.quantization_version', VAL_UINT32, 2),
    ]
    return kvs

# GGML type IDs matching ds4's ds4q_type / DS4_TENSOR_*
FP8_E4M3 = 38
MXFP4 = 39
# The DSpark workhorse weights (attn q/kv/output, shared experts, main_proj)
# ship pre-swizzled as MXFP8_LT (type 41) so the engine mmaps the layout
# instead of rebuilding it at first use. deepseek4-quantize packs it natively;
# same E4M3 weights + E8M0 scales as FP8_E4M3, just the device layout.
MXFP8_LT = 41

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.hf, 'config.json')))
    kvs = make_kvs(cfg)

    # Build tensor list with correct names
    E = cfg['hidden_size']
    F = cfg['moe_intermediate_size']
    R = cfg['n_routed_experts']
    H = cfg['num_attention_heads']
    D = cfg['head_dim']
    Q = cfg.get('q_lora_rank', 1024)
    O = cfg.get('o_lora_rank', 1024)
    OG = cfg.get('o_groups', 8)
    V = cfg['vocab_size']
    NHC = cfg.get('hc_mult', 4)
    HC_MIX = 24  # draft decoder layers use hc_mix_dim=24 (not 4)

    tensors = []

    def add(name, ndim, shape, gtype):
        tensors.append((name, ndim, shape, gtype))

    def gguf(hf_r, hf_c=None, hf_d=None):
        if hf_d is not None: return (hf_d, hf_c, hf_r)
        if hf_c is not None: return (hf_c, hf_r)
        return (hf_r,)

    add('dspark.main_proj.weight', 2, gguf(E, 3*E), MXFP8_LT)
    add('dspark.main_norm.weight', 1, gguf(E), 30)

    LAYER_MAP = [
        ('attn_norm.weight',        gguf(E),      30),
        ('ffn_norm.weight',         gguf(E),      30),
        ('hc_attn_base.weight',     gguf(HC_MIX),     0),
        ('hc_attn_fn.weight',       gguf(HC_MIX, NHC*E), 1),
        ('hc_attn_scale.weight',    gguf(3),       0),
        ('hc_ffn_base.weight',      gguf(HC_MIX),     0),
        ('hc_ffn_fn.weight',        gguf(HC_MIX, NHC*E), 1),
        ('hc_ffn_scale.weight',     gguf(3),       0),
        ('attn_sinks.weight',       gguf(H),       0),
        ('attn_q_a.weight',         gguf(Q, E),    MXFP8_LT),
        ('attn_q_a_norm.weight',    gguf(Q),       30),
        ('attn_q_b.weight',         gguf(H*D, Q),  MXFP8_LT),
        ('attn_kv.weight',          gguf(D, E),    MXFP8_LT),
        ('attn_kv_a_norm.weight',   gguf(D),       30),
        ('attn_output_a.weight',    gguf(O*OG, E), MXFP8_LT),
        ('attn_output_b.weight',    gguf(E, O*OG), MXFP8_LT),
        ('ffn_gate_inp.weight',     gguf(R, E),    30),
        ('ffn_gate_shexp.weight',   gguf(F, E),    MXFP8_LT),
        ('ffn_up_shexp.weight',     gguf(F, E),    MXFP8_LT),
        ('ffn_down_shexp.weight',   gguf(E, F),    MXFP8_LT),
    ]

    for li in range(3):
        for suffix, shape, gtype in LAYER_MAP:
            add(f'dspark.{li}.{suffix}', 2 if len(shape) > 1 else 1, shape, gtype)
        add(f'dspark.{li}.ffn_gate_exps.weight', 3, gguf(R, F, E), MXFP4)
        add(f'dspark.{li}.ffn_up_exps.weight',   3, gguf(R, F, E), MXFP4)
        add(f'dspark.{li}.ffn_down_exps.weight', 3, gguf(R, E, F), MXFP4)

    add('dspark.2.markov_head.markov_w1.weight', 2, gguf(V, 256), 30)
    add('dspark.2.markov_head.markov_w2.weight', 2, gguf(V, 256), 30)
    add('dspark.2.confidence_head.proj.weight',  1, (E + 256,), 30)
    add('dspark.2.hc_head_base.weight',  1, gguf(NHC), 0)
    add('dspark.2.hc_head_fn.weight',    2, gguf(NHC, NHC*E), 0)
    add('dspark.2.hc_head_scale.weight', 1, gguf(1), 0)
    add('dspark.2.norm.weight',          1, gguf(E), 30)

    write_gguf(a.out, kvs, tensors)
    print(f'Wrote {a.out}: {len(tensors)} tensors, {len(kvs)} KVs', file=sys.stderr)

if __name__ == '__main__':
    main()

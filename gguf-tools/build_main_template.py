#!/usr/bin/env python3
"""build_main_template.py — P0 stage 2, derived purely from the HF checkpoint.

Earlier version of this script derived its tensor manifest by stripping the
data out of one of our own already-quantized GGUFs (oracle-zeroq8-99gb.gguf).
That was circular: that GGUF's own tensor list was itself originally produced
using antirez's template.gguf, so copying it forward didn't remove anything --
it just moved which file the same information was copied from.

This version derives the ENTIRE tensor manifest (names, real shapes, types)
and ALL deepseek4.*/general.* architecture metadata directly from the HF
checkpoint itself:
  - non-expert tensor shapes/dtypes: read from the real safetensors shard
    headers (ground truth, not guessed/formula-derived)
  - per-layer heterogeneity (which layers have an indexer, a compressor, the
    tid2eid hash table): read from which HF tensors actually exist per layer,
    not hardcoded index lists
  - routed-expert combined shapes: derived from config.json (hidden_size,
    moe_intermediate_size, n_routed_experts), matching the same [in,out,R]
    stacking convention gguf-tools/build_dspark_template.py already uses
  - architecture metadata (block_count, rope params, compress_ratios, hyper-
    connection config, etc.): read directly from config.json fields
  - generation KVs (sampling top_p/temp): read from generation_config.json
  - HF<->ds4 tensor-name mapping: lifted verbatim from the top_map[]/
    layer_map[] tables in deepseek4-quantize.c -- OUR OWN already-written,
    already-working mapping, not a copy of any GGUF's structure

SCOPED OUT (deliberately, not silently): tokenizer.ggml.* and
tokenizer.chat_template. That's DeepSeek's own published tokenizer, not an
antirez architectural/precision choice, so it isn't part of the disputed
claim -- but converting a raw HF tokenizer.json (byte-level BPE vocab +
merges + special-token typing) correctly is its own nontrivial task. Until
that's built, splice those specific KV keys in from an existing valid ds4
GGUF (any one will do -- they're identical regardless of source, since it's
just the model's tokenizer, not a layout choice).

Usage:
  gguf-tools/build_main_template.py --hf HF_DIR --out main_template.gguf
"""
import argparse, hashlib, json, os, struct, sys

GGUF_MAGIC = b'GGUF'

# ds4 GGUF tensor type ids (gguf.c's gguf_types[] table).
F32, F16 = 0, 1
BF16 = 30
FP8_E4M3 = 38
MXFP4 = 39
I32 = 26

# GGUF KV value type ids.
VAL_UINT32, VAL_FLOAT32, VAL_BOOL, VAL_STRING, VAL_ARRAY, VAL_UINT64 = 4, 6, 7, 8, 9, 10

# ---------------------------------------------------------------------------
# HF <-> ds4 tensor-name mapping, copied verbatim from deepseek4-quantize.c's
# top_map[] (lines 932-939) and layer_map[] (lines 941-974).
# ---------------------------------------------------------------------------
TOP_MAP = {
    'token_embd.weight':      'embed.weight',
    'output_norm.weight':     'norm.weight',
    'output.weight':          'head.weight',
    'output_hc_base.weight':  'hc_head_base',
    'output_hc_fn.weight':    'hc_head_fn',
    'output_hc_scale.weight': 'hc_head_scale',
}

LAYER_MAP = {
    'hc_attn_base.weight':            'hc_attn_base',
    'hc_attn_fn.weight':              'hc_attn_fn',
    'hc_attn_scale.weight':           'hc_attn_scale',
    'hc_ffn_base.weight':             'hc_ffn_base',
    'hc_ffn_fn.weight':               'hc_ffn_fn',
    'hc_ffn_scale.weight':            'hc_ffn_scale',
    'attn_sinks.weight':              'attn.attn_sink',
    'attn_q_a.weight':                'attn.wq_a.weight',
    'attn_q_b.weight':                'attn.wq_b.weight',
    'attn_q_a_norm.weight':           'attn.q_norm.weight',
    'attn_kv.weight':                 'attn.wkv.weight',
    'attn_kv_a_norm.weight':          'attn.kv_norm.weight',
    'attn_output_a.weight':           'attn.wo_a.weight',
    'attn_output_b.weight':           'attn.wo_b.weight',
    'attn_compressor_ape.weight':     'attn.compressor.ape',
    'attn_compressor_kv.weight':      'attn.compressor.wkv.weight',
    'attn_compressor_gate.weight':    'attn.compressor.wgate.weight',
    'attn_compressor_norm.weight':    'attn.compressor.norm.weight',
    'indexer.attn_q_b.weight':        'attn.indexer.wq_b.weight',
    'indexer.proj.weight':            'attn.indexer.weights_proj.weight',
    'indexer_compressor_ape.weight':  'attn.indexer.compressor.ape',
    'indexer_compressor_kv.weight':   'attn.indexer.compressor.wkv.weight',
    'indexer_compressor_gate.weight': 'attn.indexer.compressor.wgate.weight',
    'indexer_compressor_norm.weight': 'attn.indexer.compressor.norm.weight',
    'attn_norm.weight':               'attn_norm.weight',
    'ffn_norm.weight':                'ffn_norm.weight',
    'ffn_gate_shexp.weight':          'ffn.shared_experts.w1.weight',
    'ffn_up_shexp.weight':            'ffn.shared_experts.w3.weight',
    'ffn_down_shexp.weight':          'ffn.shared_experts.w2.weight',
    'ffn_gate_inp.weight':            'ffn.gate.weight',
    'exp_probs_b.bias':               'ffn.gate.bias',
    'ffn_gate_tid2eid.weight':        'ffn.gate.tid2eid',
}

# Suffixes present on every one of the num_hidden_layers main-model layers.
ALWAYS_LAYER_SUFFIXES = [
    'attn_norm.weight', 'ffn_norm.weight',
    'hc_attn_base.weight', 'hc_attn_fn.weight', 'hc_attn_scale.weight',
    'hc_ffn_base.weight', 'hc_ffn_fn.weight', 'hc_ffn_scale.weight',
    'attn_sinks.weight', 'attn_q_a.weight', 'attn_q_a_norm.weight',
    'attn_q_b.weight', 'attn_kv.weight', 'attn_kv_a_norm.weight',
    'attn_output_a.weight', 'attn_output_b.weight',
    'ffn_gate_inp.weight', 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight',
    'ffn_down_shexp.weight',
]

# Suffixes present only on some layers -- presence is checked against the
# real HF tensor list per layer, never hardcoded to specific layer indices.
CONDITIONAL_LAYER_SUFFIXES = [
    'exp_probs_b.bias', 'ffn_gate_tid2eid.weight',
    'attn_compressor_ape.weight', 'attn_compressor_kv.weight',
    'attn_compressor_gate.weight', 'attn_compressor_norm.weight',
    'indexer.attn_q_b.weight', 'indexer.proj.weight',
    'indexer_compressor_ape.weight', 'indexer_compressor_kv.weight',
    'indexer_compressor_gate.weight', 'indexer_compressor_norm.weight',
]

# Per-tensor-group template type policy (the default type used when the
# quantizer isn't given an explicit --dense/--attention/--experts/etc
# override; norms are additionally a hard ds4 requirement -- see
# weights.c's mtp_weights_validate_layout-style F32 checks). NOT a native-
# HF-dtype passthrough: several groups take a specific ds4 type regardless of
# their HF dtype -- confirmed against real HF shard dtypes.
#
# 2026-08-15: every group below is now the format the checkpoint actually uses,
# checked against the real shard headers rather than config.json's torch_dtype
# (a default, not a per-tensor promise). Getting there took two passes:
#
#   - the families that ARE bf16 upstream were stored f16. bf16 -> f16 is lossy
#     where it counts: f16 has 5 exponent bits against bf16's 8, so it flushes
#     anything under 5.96e-8 to zero, and these tensors run to 2.7e-15.
#   - the families that are F32 upstream were ALSO stored f16, and briefly moved
#     to bf16 here, which fixed the flushing but still sat below source. They
#     are F32 now -- see F32_SOURCE.
#   - indexer.attn_q_b is f8_e4m3 upstream and belongs in neither group.
#
# So BF16_GROUP means exactly one thing again: bf16 upstream, bf16 here,
# lossless.
DENSE_FP8 = {
    'attn_q_a.weight', 'attn_q_b.weight', 'attn_kv.weight',
    'attn_output_a.weight', 'attn_output_b.weight',
    'ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight',
}
# BF16 upstream -> BF16 here (was F16 until 2026-08-15; see the note above).
# BF16 upstream -> BF16 here. Lossless copy, nothing to weigh.
BF16_GROUP = {
    'ffn_gate_inp.weight',
    'attn_compressor_kv.weight', 'attn_compressor_gate.weight',
    'indexer.proj.weight',
    'indexer_compressor_kv.weight', 'indexer_compressor_gate.weight',
}

# F32 upstream -> F32 here. These were f16, which was actively destructive:
# nothing in them approaches f16's 65504 ceiling (max |w| is 5.1) but the small
# end is dense, values run to 2.7e-15, and f16 flushes everything under 5.96e-8
# to exactly zero --
#
#     layers.0.hc_attn_fn   11.07% of nonzero weights -> 0, worst row loses 28%
#     hc_head_fn             5.73%                    -> 0
#     layers.0.hc_ffn_fn     0.71%                    -> 0
#
# relative error 1.0 on each: not approximated, deleted. bf16 would stop that
# (f32's exponent range, min normal 1.18e-38) and was the interim choice, but it
# is still narrower than the source. F32 is what the checkpoint holds, costs
# +70.6 MB over bf16 across all five families, and needs no new engine path --
# the plain matmul already has an F32 arm and the compressor kernels already
# take ape_type 0 = F32.
#
# The two *_ape families in particular MUST NOT be bf16: the compressor kernels
# validate ape_type as 0 (F32) or 1 (F16) and reject 30 outright, so a bf16 ape
# fails at run time.
F32_SOURCE = {
    'attn_compressor_ape.weight', 'indexer_compressor_ape.weight',
    'hc_attn_fn.weight', 'hc_ffn_fn.weight', 'output_hc_fn.weight',
}

# NOT in the group: indexer.attn_q_b.weight is F8_E4M3 upstream (21 tensors).
# f16 already represents every e4m3 value exactly -- 10 mantissa bits against
# e4m3's 3, and f16's range covers e4m3's -- so storing it F16 costs nothing in
# fidelity, only 2x the bytes. BF16 would be exact too, so moving it buys
# nothing. The real fix is to keep it fp8 end to end, which is an A8-campaign
# item and an engine change, not a template type flip.


# 1-D tensors the checkpoint holds in BF16. The rest of the 1-D set --
# hc_*_base, hc_*_scale, attn_sinks, exp_probs_b -- really is f32 upstream and
# stays F32, so this cannot be a blanket rule on ndim.
#
# Storing these f32 was never a fidelity question (f32 represents every bf16
# value exactly); it just spent 0.95 MB carrying nothing. They are cheap to
# move only because the engine reads bf16 norm weights directly -- widening
# them back to f32 at load would cost more VRAM than the file saves.
NORM_BF16 = {
    'attn_norm.weight', 'ffn_norm.weight',
    'attn_q_a_norm.weight', 'attn_kv_a_norm.weight',
    'attn_compressor_norm.weight', 'indexer_compressor_norm.weight',
    'output_norm.weight',
}


def suffix_type(ds4_name, ndim):
    suffix = ds4_name.split('.', 2)[-1] if ds4_name.startswith('blk.') else ds4_name
    if ndim <= 1:
        return BF16 if suffix in NORM_BF16 else F32
    if suffix in DENSE_FP8:
        return FP8_E4M3
    if suffix in BF16_GROUP:
        return BF16
    if suffix in F32_SOURCE:
        return F32
    if suffix == 'ffn_gate_tid2eid.weight':
        return I32
    if suffix == 'indexer.attn_q_b.weight':
        # F8_E4M3 upstream. Held F16 until 2026-08-15, which was lossless (f16
        # represents every e4m3 code exactly) but cost 352 MB to store 181 MB of
        # information -- the single largest above-source item in the model.
        # ds4's FP8_E4M3 is MXFP8: E4M3 + a per-32 E8M0 scale. The source's
        # scales are 128x128 blocks and 128 is a multiple of 32, so every per-32
        # group lies wholly inside one source block and inherits its scale --
        # the mantissas copy verbatim and the round trip is exact.
        return FP8_E4M3
    if ds4_name == 'token_embd.weight':
        # BF16 upstream, and BF16 here since 2026-08-15. This one is FREE: f16
        # and bf16 are both 2 bytes, so the 1059 MB table is exactly the same
        # size either way -- f16 was simply a lossy copy of a bf16 source for no
        # saving at all. The engine reads either (weights.cpp
        # tensor_expect_f32_or_bf16_or_f16 + the templated embed kernels).
        return BF16
    if ds4_name == 'output.weight':
        return BF16
    raise ValueError(f'no type policy for {ds4_name!r} (ndim={ndim})')


# ---------------------------------------------------------------------------
# HF checkpoint reader: real shapes/dtypes from shard headers, no guessing.
# ---------------------------------------------------------------------------
class HFCheckpoint:
    def __init__(self, hf_dir):
        self.dir = hf_dir
        self.config = json.load(open(os.path.join(hf_dir, 'config.json')))
        self.index = json.load(open(os.path.join(hf_dir, 'model.safetensors.index.json')))
        self.weight_map = self.index['weight_map']
        self._shard_hdr_cache = {}
        gc_path = os.path.join(hf_dir, 'generation_config.json')
        self.generation_config = json.load(open(gc_path)) if os.path.exists(gc_path) else {}

    def has(self, name):
        return name in self.weight_map

    def _shard_header(self, shard_file):
        if shard_file not in self._shard_hdr_cache:
            with open(os.path.join(self.dir, shard_file), 'rb') as f:
                n, = struct.unpack('<Q', f.read(8))
                self._shard_hdr_cache[shard_file] = json.loads(f.read(n))
        return self._shard_hdr_cache[shard_file]

    def shape(self, name):
        """Real on-disk HF shape (row-major [out, in, ...])."""
        shard = self.weight_map[name]
        return self._shard_header(shard)[name]['shape']

    def dtype(self, name):
        """On-disk safetensors dtype string, e.g. 'BF16', 'F8_E4M3', 'I8'.

        This is the CONTAINER dtype, which is not always the logical format:
        the routed experts are declared fp4 by config.json but ride in an I8
        container two-per-byte alongside an F8_E8M0 scale.
        """
        shard = self.weight_map[name]
        return self._shard_header(shard)[name]['dtype']

    def raw(self, name):
        """The tensor's bytes exactly as stored, no dtype interpretation.

        Row-major, which is the same byte order ds4 wants -- ne_reversed()
        reverses the DIMENSION LIST for the GGUF convention, it does not
        transpose data -- so these bytes can be written straight into a
        matching ds4 tensor region.
        """
        shard = self.weight_map[name]
        hdr = self._shard_header(shard)
        start, end = hdr[name]['data_offsets']
        with open(os.path.join(self.dir, shard), 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            f.seek(8 + n + start)
            buf = f.read(end - start)
        if len(buf) != end - start:
            raise SystemExit(f'{name}: short read from {shard}')
        return buf


def ne_reversed(hf_shape):
    """HF stores row-major [out, in]; GGUF ne[] is the reverse."""
    return list(reversed(hf_shape))


# ---------------------------------------------------------------------------
# Tensor manifest enumeration.
# ---------------------------------------------------------------------------
def build_tensor_list(ckpt, keep=None):
    """keep: optional per-layer routed-expert count (the REAP survivor count).
    When given, pruned layers declare their DENSE survivor count in the expert
    tensors' trailing dim, so the quantizer fills an already-pruned manifest
    instead of a downstream tool re-slicing a full-256 artifact. Router and bias
    tensors stay padded to the full n_expert -- that is the ds4-compact-v1
    convention the CUDA router kernels depend on."""
    cfg = ckpt.config
    L = cfg['num_hidden_layers']
    E = cfg['hidden_size']
    F = cfg['moe_intermediate_size']
    R = cfg['n_routed_experts']
    if keep is not None and len(keep) < L:
        raise SystemExit(f'survivor map covers {len(keep)} layers, model has {L}')
    tensors = []  # (ds4_name, ne, type)

    for ds4_name, hf_name in TOP_MAP.items():
        if not ckpt.has(hf_name):
            raise SystemExit(f'expected top-level HF tensor missing: {hf_name}')
        ne = ne_reversed(ckpt.shape(hf_name))
        tensors.append((ds4_name, ne, suffix_type(ds4_name, len(ne))))

    for layer in range(L):
        for suffix in ALWAYS_LAYER_SUFFIXES:
            hf_name = f'layers.{layer}.{LAYER_MAP[suffix]}'
            if not ckpt.has(hf_name):
                raise SystemExit(f'expected HF tensor missing: {hf_name}')
            ne = ne_reversed(ckpt.shape(hf_name))
            tensors.append((f'blk.{layer}.{suffix}', ne, suffix_type(suffix, len(ne))))

        for suffix in CONDITIONAL_LAYER_SUFFIXES:
            hf_name = f'layers.{layer}.{LAYER_MAP[suffix]}'
            if not ckpt.has(hf_name):
                continue
            ne = ne_reversed(ckpt.shape(hf_name))
            tensors.append((f'blk.{layer}.{suffix}', ne, suffix_type(suffix, len(ne))))

        # Routed experts: combined [in,out,R] stack, shape from config (the
        # per-expert HF tensors are individually MXFP4-packed, not a single
        # combined tensor -- same convention as build_dspark_template.py).
        Rl = keep[layer] if keep is not None else R
        tensors.append((f'blk.{layer}.ffn_gate_exps.weight', [E, F, Rl], MXFP4))
        tensors.append((f'blk.{layer}.ffn_up_exps.weight',   [E, F, Rl], MXFP4))
        tensors.append((f'blk.{layer}.ffn_down_exps.weight', [F, E, Rl], MXFP4))

    return tensors


# ---------------------------------------------------------------------------
# deepseek4.*/general.* metadata, derived from config.json + generation_config.json.
# ---------------------------------------------------------------------------
def build_kvs(ckpt, reap=None, reap_sha=None):
    cfg = ckpt.config
    gen = ckpt.generation_config
    L = cfg['num_hidden_layers']
    rope = cfg['rope_scaling']

    kvs = [
        ('general.architecture', VAL_STRING, 'deepseek4'),
        ('general.type', VAL_STRING, 'model'),
        ('general.name', VAL_STRING, 'DeepSeek V4 Flash'),
        ('general.file_type', VAL_UINT32, 19),
        ('general.quantization_version', VAL_UINT32, 2),
        ('deepseek4.block_count', VAL_UINT32, L),
        ('deepseek4.context_length', VAL_UINT32, cfg['max_position_embeddings']),
        ('deepseek4.embedding_length', VAL_UINT32, cfg['hidden_size']),
        ('deepseek4.attention.head_count', VAL_UINT32, cfg['num_attention_heads']),
        ('deepseek4.attention.head_count_kv', VAL_UINT32, cfg['num_key_value_heads']),
        ('deepseek4.rope.scaling.type', VAL_STRING, rope['type']),
        ('deepseek4.rope.scaling.factor', VAL_FLOAT32, float(rope['factor'])),
        ('deepseek4.rope.scaling.original_context_length', VAL_UINT32, rope['original_max_position_embeddings']),
        ('deepseek4.rope.scaling.yarn_beta_fast', VAL_FLOAT32, float(rope['beta_fast'])),
        ('deepseek4.rope.scaling.yarn_beta_slow', VAL_FLOAT32, float(rope['beta_slow'])),
        ('deepseek4.rope.freq_base', VAL_FLOAT32, float(cfg['rope_theta'])),
        ('deepseek4.attention.layer_norm_rms_epsilon', VAL_FLOAT32, float(cfg['rms_norm_eps'])),
        ('deepseek4.expert_used_count', VAL_UINT32, cfg['num_experts_per_tok']),
        # NOTE: not read by any code in this repo (grepped weights.c/gguf.c/
        # deepseek4-quantize.c -- no hits); vestigial GGUF bookkeeping.
        # 4 is the only value observed for scoring_func="sqrtsoftplus"; no
        # independent enum derivation available.
        ('deepseek4.expert_gating_func', VAL_UINT32, 4),
        ('deepseek4.attention.key_length', VAL_UINT32, cfg['head_dim']),
        ('deepseek4.attention.value_length', VAL_UINT32, cfg['head_dim']),
        ('deepseek4.vocab_size', VAL_UINT32, cfg['vocab_size']),
        ('deepseek4.rope.dimension_count', VAL_UINT32, cfg['qk_rope_head_dim']),
        ('deepseek4.attention.q_lora_rank', VAL_UINT32, cfg['q_lora_rank']),
        ('deepseek4.attention.output_lora_rank', VAL_UINT32, cfg['o_lora_rank']),
        ('deepseek4.attention.output_group_count', VAL_UINT32, cfg['o_groups']),
        ('deepseek4.attention.compress_ratios', VAL_ARRAY, (VAL_UINT32, cfg['compress_ratios'][:L])),
        ('deepseek4.attention.compress_rope_freq_base', VAL_FLOAT32, float(cfg['compress_rope_theta'])),
        ('deepseek4.expert_feed_forward_length', VAL_UINT32, cfg['moe_intermediate_size']),
        ('deepseek4.expert_count', VAL_UINT32, cfg['n_routed_experts']),
        ('deepseek4.expert_shared_count', VAL_UINT32, cfg['n_shared_experts']),
        ('deepseek4.expert_weights_scale', VAL_FLOAT32, float(cfg['routed_scaling_factor'])),
        ('deepseek4.hash_layer_count', VAL_UINT32, cfg['num_hash_layers']),
        ('deepseek4.expert_weights_norm', VAL_BOOL, bool(cfg['norm_topk_prob'])),
        ('deepseek4.swiglu_clamp_exp', VAL_ARRAY, (VAL_FLOAT32, [float(cfg['swiglu_limit'])] * L)),
        ('deepseek4.attention.sliding_window', VAL_UINT32, cfg['sliding_window']),
        ('deepseek4.attention.indexer.head_count', VAL_UINT32, cfg['index_n_heads']),
        ('deepseek4.attention.indexer.key_length', VAL_UINT32, cfg['index_head_dim']),
        ('deepseek4.attention.indexer.top_k', VAL_UINT32, cfg['index_topk']),
        ('deepseek4.nextn_predict_layers', VAL_UINT32, cfg['num_nextn_predict_layers']),
        ('deepseek4.hyper_connection.count', VAL_UINT32, cfg['hc_mult']),
        ('deepseek4.hyper_connection.sinkhorn_iterations', VAL_UINT32, cfg['hc_sinkhorn_iters']),
        ('deepseek4.hyper_connection.epsilon', VAL_FLOAT32, float(cfg['hc_eps'])),
        ('general.sampling.top_p', VAL_FLOAT32, float(gen.get('top_p', 1.0))),
        ('general.sampling.temp', VAL_FLOAT32, float(gen.get('temperature', 1.0))),
    ]
    if reap is not None:
        # The engine validates these at load (validate_reap_metadata in
        # weights.cpp): the arrays must cover every layer and each entry must
        # be in [1, n_expert], or it dies. policy 1 = hash-routed layers that
        # keep all 256 (they route by a fixed tid2eid table and structurally
        # cannot be pruned); policy 2 = biased top-k layers trimmed to
        # keep_count, with the pruned bias slots set to -1e30 so they can never
        # win selection.
        kvs += [
            ('reap.enabled', VAL_BOOL, True),
            ('reap.layout', VAL_STRING, reap['layout']),
            ('reap.layer.expert_count', VAL_ARRAY, (VAL_UINT32, [int(x) for x in reap['expert_count']])),
            ('reap.layer.keep_count', VAL_ARRAY, (VAL_UINT32, [int(x) for x in reap['keep_count']])),
            ('reap.layer.policy', VAL_ARRAY, (VAL_UINT32, [int(x) for x in reap['policy']])),
        ]
        # Binds this template to the EXACT survivor map it was shaped by. The
        # arrays above only carry per-layer counts and policies, so two maps
        # that keep the same NUMBER of experts per layer -- but different ones
        # -- produce byte-identical templates and pass every shape check, while
        # routing every token to a different expert. Nothing else in the build
        # would notice. The quantizer refuses a map whose hash does not match.
        if reap_sha:
            kvs.append(('reap.survivors.sha256', VAL_STRING, reap_sha))
    return kvs


# ---------------------------------------------------------------------------
# GGUF binary writer (hand-rolled: the `gguf` python package doesn't support
# FP8_E4M3(38)/MXFP4(39), same reason build_dspark_template.py rolls its own).
# ---------------------------------------------------------------------------
class Buf:
    def __init__(self):
        self.data = bytearray()

    def u32(self, v): self.data += struct.pack('<I', v)
    def u64(self, v): self.data += struct.pack('<Q', v)
    def f32(self, v): self.data += struct.pack('<f', v)
    def u8(self, v): self.data += struct.pack('<B', v)

    def string(self, s):
        b = s.encode('utf-8')
        self.u64(len(b))
        self.data += b

    def kv_value(self, typ, val):
        if typ == VAL_STRING:
            self.string(val)
        elif typ == VAL_UINT32:
            self.u32(val)
        elif typ == VAL_FLOAT32:
            self.f32(val)
        elif typ == VAL_BOOL:
            self.u8(1 if val else 0)
        elif typ == VAL_ARRAY:
            elem_typ, items = val
            self.u32(elem_typ)
            self.u64(len(items))
            for item in items:
                self.kv_value(elem_typ, item)
        else:
            raise ValueError(f'unsupported KV type {typ}')

    def pad_to(self, align):
        while len(self.data) % align:
            self.u8(0)


def splice_tokenizer_kvs(src_gguf_path):
    """Read tokenizer.ggml.* + tokenizer.chat_template KV entries verbatim
    (raw bytes) from an existing valid ds4 GGUF. Deliberately scoped-out
    piece (see module docstring) -- this is DeepSeek's own tokenizer data,
    not an antirez layout choice, so borrowing the bytes isn't the same
    category of dependency this script otherwise removes."""
    f = open(src_gguf_path, 'rb')
    assert f.read(4) == GGUF_MAGIC
    struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))

    def read_string():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n)

    def read_value(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            return read_string()
        if t == 9:
            et, = struct.unpack('<I', f.read(4))
            n, = struct.unpack('<Q', f.read(8))
            return [read_value(et) for _ in range(n)]
        return f.read(sz[t])

    out = []
    for _ in range(n_kv):
        key_start = f.tell()
        key = read_string()
        t, = struct.unpack('<I', f.read(4))
        val_start_after_type = f.tell()
        read_value(t)
        val_end = f.tell()
        if key.startswith(b'tokenizer.'):
            f.seek(key_start)
            raw = f.read(val_end - key_start)
            out.append(raw)
    f.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True, help='HF checkpoint directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--splice-tokenizer-from', help='existing ds4 GGUF to copy tokenizer.* KVs from (legacy bootstrap; prefer --tokenizer-from-hf)')
    ap.add_argument('--tokenizer-from-hf', action='store_true',
                    help='derive tokenizer.* KVs from the HF checkpoint (no prior GGUF needed)')
    ap.add_argument('--tokenizer-template', default=None,
                    help='chat template for --tokenizer-from-hf (default: gguf-tools/tokenizer/chat_template.jinja)')
    ap.add_argument('--reap-survivors', metavar='JSON',
                    help='REAP survivor map; declares pruned expert dims and emits the reap.* KVs')
    a = ap.parse_args()

    ckpt = HFCheckpoint(a.hf)
    reap = None
    reap_sha = None
    if a.reap_survivors:
        with open(a.reap_survivors, 'rb') as f:
            reap_sha = hashlib.sha256(f.read()).hexdigest()
        with open(a.reap_survivors, encoding='utf-8') as f:
            reap = json.load(f)
        for k in ('layout', 'expert_count', 'keep_count', 'policy', 'survivors'):
            if k not in reap:
                raise SystemExit(f'{a.reap_survivors}: survivor map is missing "{k}"')
    tensors = build_tensor_list(ckpt, keep=reap['keep_count'] if reap else None)
    kvs = build_kvs(ckpt, reap=reap, reap_sha=reap_sha)

    if a.tokenizer_from_hf:
        # Derive the tokenizer KVs from the checkpoint instead of copying them
        # out of a prior artifact. Same raw-blob format and same order, so the
        # concatenation below is unchanged. Verified byte-identical to a spliced
        # block by gguf-tools/tokenizer/build_tokenizer_kvs.py --verify-against.
        tokdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
        sys.path.insert(0, tokdir)
        from build_tokenizer_kvs import build as build_tokenizer_kvs
        tmpl = a.tokenizer_template or os.path.join(tokdir, 'chat_template.jinja')
        tok_raw_kvs = list(build_tokenizer_kvs(a.hf, tmpl).values())
    elif a.splice_tokenizer_from:
        tok_raw_kvs = splice_tokenizer_kvs(a.splice_tokenizer_from)
    else:
        tok_raw_kvs = []

    buf = Buf()
    for key, typ, val in kvs:
        buf.string(key)
        buf.u32(typ)
        buf.kv_value(typ, val)
    kv_bytes = bytes(buf.data) + b''.join(tok_raw_kvs)
    n_kv = len(kvs) + len(tok_raw_kvs)

    tbuf = Buf()
    for name, ne, ttype in tensors:
        tbuf.string(name)
        tbuf.u32(len(ne))
        for d in ne:
            tbuf.u64(d)
        tbuf.u32(ttype)
        tbuf.u64(0)  # offset: template carries no tensor data

    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack('<I', 3)
    out += struct.pack('<Q', len(tensors))
    out += struct.pack('<Q', n_kv)
    out += kv_bytes
    out += bytes(tbuf.data)

    with open(a.out, 'wb') as f:
        f.write(bytes(out))
    print(f'Wrote {a.out}: {len(tensors)} tensors, {n_kv} KV pairs '
          f'({len(tok_raw_kvs)} spliced tokenizer KVs), {len(out)} bytes',
          file=sys.stderr)


if __name__ == '__main__':
    main()

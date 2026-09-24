"""kv.py -- the container's `pulsar.kv` block, straight from the HF checkpoint (L247).

    build_kv(hf, tokenizer_dir, reap_map=None) -> [{'key', 'type', 'value'}, ...]

`hf` is an HFCheckpoint (`.config`, `.generation_config`) or a plain HF
directory.  The block is what src/engine/safetensors.cpp parses
(`meta_code_for_type_name`, `blob_array`): types in the engine's spelling
(`u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool string array`), arrays as
`{"__array__": elem, "n": N, "v": [...]}`, every f32 rounded through
`struct.pack('<f')` before it becomes a Python float, so `json.dumps` writes the
repr of the value the engine will read (rms_eps 1e-20 -> 9.999999682655225e-21).

What is emitted, and why, in the served block's order
(tools/container/test_kv.py grades it entry-for-entry against the served artifact):

  * the deepseek4.* keys config_validate_model / validate_attention_layout_metadata /
    validate_swiglu_clamp_metadata read (src/engine/weights.cpp), from config.json
    (`text_config` when the checkpoint nests it, as V4.1 does);
  * `hash_layer_count` only when the config declares `num_hash_layers` (Vision-Exp);
    the CSA2 sharing keys only when it declares `kv_source_layer_ids` (V4.1) -- the
    engine treats both as "declared -> must agree with the profile, absent -> none";
  * `general.architecture` / `general.name` (the engine's load summary,
    src/engine/model.cpp) and the rest of the served block's record-only keys
    that have an HF source: `general.type`, `context_length`, `rope.scaling.type`,
    `nextn_predict_layers`, `general.sampling.*`;
  * the reap.* keys when a survivor map is given (validate_reap_metadata);
  * the eleven tokenizer.* entries (tools/container/tokenizer);
  * the drafter's `deepseek_v4_dspark.embedding_length` + `dspark.target_layer_ids.N`
    (dspark_weights_bind reads exactly these four; the drafter's shape is pinned
    against its tensors, weights.cpp:1555-1569);
  * `deepseek4.engram.{layers,n_rows}` when the config declares Engram layers
    (V4.1), naming the side row tables so L242 slice 5 can refuse a mismatch.

NOT emitted, because the direct builder has no source for them: the GGUF
bookkeeping numbers (`general.file_type`, `general.quantization_version`,
`deepseek4.expert_gating_func`) and the quantizer's provenance
(`quantize.imatrix.*`).  Nothing in src/ reads any of them.
"""
import hashlib
import json
import os
import struct

from tokenizer.build_tokenizer_kvs import build as build_tokenizer_kvs

INT_RANGE = {
    'u8': (0, 2**8 - 1), 'i8': (-2**7, 2**7 - 1),
    'u16': (0, 2**16 - 1), 'i16': (-2**15, 2**15 - 1),
    'u32': (0, 2**32 - 1), 'i32': (-2**31, 2**31 - 1),
    'u64': (0, 2**64 - 1), 'i64': (-2**63, 2**63 - 1),
}

# general.name from the checkpoint's model_type (HFCheckpoint may hand us the
# merged text_config, whose spelling carries a `_text` suffix).
FAMILY_NAME = {
    'deepseek_v4': 'DeepSeek V4 Flash',
    'deepseek_v41': 'DeepSeek V4.1 Flash',
}


def f32(x):
    """The float the engine will read: x rounded to binary32."""
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def _scalar(typ, value, key):
    if typ in INT_RANGE:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f'{key}: {typ} wants an int, got {value!r}')
        lo, hi = INT_RANGE[typ]
        if not lo <= value <= hi:
            raise SystemExit(f'{key}: {value} is outside {typ}')
        return value
    if typ == 'f32':
        return f32(value)
    if typ == 'f64':
        return float(value)
    if typ == 'bool':
        if not isinstance(value, bool):
            raise SystemExit(f'{key}: bool wants a bool, got {value!r}')
        return value
    if typ == 'string':
        if not isinstance(value, str):
            raise SystemExit(f'{key}: string wants a str, got {type(value).__name__}')
        return value
    raise SystemExit(f'{key}: unknown metadata type {typ!r}')


def entry(key, typ, value):
    """One (key, type, value) triple -> the JSON entry the engine parses.

    Array triples carry (elem_type, values)."""
    if typ == 'array':
        elem, values = value
        return {'key': key, 'type': 'array',
                'value': {'__array__': elem, 'n': len(values),
                          'v': [_scalar(elem, v, f'{key}[{i}]') for i, v in enumerate(values)]}}
    return {'key': key, 'type': typ, 'value': _scalar(typ, value, key)}


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class _Source:
    """config.json + generation_config.json, from an HFCheckpoint or a directory."""

    def __init__(self, hf):
        if isinstance(hf, (str, os.PathLike)):
            top = _load_json(os.path.join(hf, 'config.json'))
            gen_path = os.path.join(hf, 'generation_config.json')
            gen = _load_json(gen_path) if os.path.exists(gen_path) else {}
        else:
            top, gen = hf.config, hf.generation_config
        # V4.1 nests the language model under text_config; Vision-Exp is flat.
        self.text = top['text_config'] if 'text_config' in top else top
        self.model_type = top['model_type']
        self.generation = gen


def _one_of(d, keys, what):
    present = [k for k in keys if k in d]
    if len(present) != 1:
        raise SystemExit(f'{what}: expected exactly one of {keys}, found {present}')
    return d[present[0]]


def _family_name(model_type):
    base = model_type[:-len('_text')] if model_type.endswith('_text') else model_type
    if base not in FAMILY_NAME:
        raise SystemExit(f'config.json model_type {model_type!r} is not a family this builder names '
                         f'({sorted(FAMILY_NAME)})')
    return FAMILY_NAME[base]


def _compress_ratios(cfg, n_layer):
    """config.json lists one ratio per main layer AND per next-n (MTP) layer;
    the main block carries the main layers' (the drafter owns the tail)."""
    v = list(cfg['compress_ratios'])
    n_mtp = cfg['num_nextn_predict_layers']
    if len(v) != n_layer + n_mtp:
        raise SystemExit(f'config.json compress_ratios has {len(v)} entries for '
                         f'{n_layer} layers + {n_mtp} next-n layers')
    return v[:n_layer]


def _reap_triples(reap_map):
    with open(reap_map, 'rb') as f:
        raw = f.read()
    reap = json.loads(raw.decode('utf-8'))
    for k in ('layout', 'expert_count', 'keep_count', 'policy', 'survivors'):
        if k not in reap:
            raise SystemExit(f'{reap_map}: survivor map is missing "{k}"')
    # The engine validates these at load (validate_reap_metadata): keep_count
    # covers every layer and each entry is in [1, n_expert].  The sha binds the
    # container to the EXACT survivor map: two maps keeping the same NUMBER of
    # experts per layer but different ones would pass every shape check.
    return [
        ('reap.enabled', 'bool', True),
        ('reap.layout', 'string', reap['layout']),
        ('reap.layer.expert_count', 'array', ('u32', [int(x) for x in reap['expert_count']])),
        ('reap.layer.keep_count', 'array', ('u32', [int(x) for x in reap['keep_count']])),
        ('reap.layer.policy', 'array', ('u32', [int(x) for x in reap['policy']])),
        ('reap.survivors.sha256', 'string', hashlib.sha256(raw).hexdigest()),
    ]


def build_kv(hf, tokenizer_dir, reap_map=None):
    src = _Source(hf)
    cfg = src.text
    gen = src.generation
    L = cfg['num_hidden_layers']
    rope = cfg['rope_scaling']

    kvs = [
        ('general.architecture', 'string', 'deepseek4'),
        ('general.type', 'string', 'model'),
        ('general.name', 'string', _family_name(src.model_type)),
        ('deepseek4.block_count', 'u32', L),
        ('deepseek4.context_length', 'u32', cfg['max_position_embeddings']),
        ('deepseek4.embedding_length', 'u32', cfg['hidden_size']),
        ('deepseek4.attention.head_count', 'u32', cfg['num_attention_heads']),
        ('deepseek4.attention.head_count_kv', 'u32', cfg['num_key_value_heads']),
        # Vision-Exp spells it `type`, V4.1 `rope_type`.
        ('deepseek4.rope.scaling.type', 'string', _one_of(rope, ('rope_type', 'type'), 'rope_scaling')),
        ('deepseek4.rope.scaling.factor', 'f32', rope['factor']),
        ('deepseek4.rope.scaling.original_context_length', 'u32', rope['original_max_position_embeddings']),
        ('deepseek4.rope.scaling.yarn_beta_fast', 'f32', rope['beta_fast']),
        ('deepseek4.rope.scaling.yarn_beta_slow', 'f32', rope['beta_slow']),
        ('deepseek4.rope.freq_base', 'f32', cfg['rope_theta']),
        ('deepseek4.attention.layer_norm_rms_epsilon', 'f32', cfg['rms_norm_eps']),
        ('deepseek4.expert_used_count', 'u32', cfg['num_experts_per_tok']),
        ('deepseek4.attention.key_length', 'u32', cfg['head_dim']),
        ('deepseek4.attention.value_length', 'u32', cfg['head_dim']),
        ('deepseek4.vocab_size', 'u32', cfg['vocab_size']),
        ('deepseek4.rope.dimension_count', 'u32', cfg['qk_rope_head_dim']),
        ('deepseek4.attention.q_lora_rank', 'u32', cfg['q_lora_rank']),
        ('deepseek4.attention.output_lora_rank', 'u32', cfg['o_lora_rank']),
        ('deepseek4.attention.output_group_count', 'u32', cfg['o_groups']),
        ('deepseek4.attention.compress_ratios', 'array', ('u32', _compress_ratios(cfg, L))),
        ('deepseek4.attention.compress_rope_freq_base', 'f32', cfg['compress_rope_theta']),
        ('deepseek4.expert_feed_forward_length', 'u32', cfg['moe_intermediate_size']),
        ('deepseek4.expert_count', 'u32', cfg['n_routed_experts']),
        ('deepseek4.expert_shared_count', 'u32', cfg['n_shared_experts']),
        ('deepseek4.expert_weights_scale', 'f32', cfg['routed_scaling_factor']),
    ]
    # Hash-routed layers: declared by Vision-Exp/0731 (3), absent from V4.1.
    # weights.cpp:1072 -- present must agree with the profile, absent means none.
    if 'num_hash_layers' in cfg:
        kvs.append(('deepseek4.hash_layer_count', 'u32', cfg['num_hash_layers']))
    kvs += [
        ('deepseek4.expert_weights_norm', 'bool', bool(cfg['norm_topk_prob'])),
        ('deepseek4.swiglu_clamp_exp', 'array', ('f32', [cfg['swiglu_limit']] * L)),
        ('deepseek4.attention.sliding_window', 'u32', cfg['sliding_window']),
        ('deepseek4.attention.indexer.head_count', 'u32', cfg['index_n_heads']),
        ('deepseek4.attention.indexer.key_length', 'u32', cfg['index_head_dim']),
        ('deepseek4.attention.indexer.top_k', 'u32', cfg['index_topk']),
        ('deepseek4.nextn_predict_layers', 'u32', cfg['num_nextn_predict_layers']),
    ]
    # CSA2 sharing (V4.1): which layers write the shared caches and the shared
    # top-k, and the candidate pool.  All five or none -- the engine derives the
    # set from the profile when they are absent (validate_attention_layout_metadata).
    csa2 = ('kv_source_layer_ids', 'index_source_layer_ids', 'candidate_source_layer_id',
            'candidate_topk_blocks', 'candidate_block_size')
    have = [k for k in csa2 if k in cfg]
    if have and len(have) != len(csa2):
        raise SystemExit(f'config.json declares only {have} of the CSA2 sharing keys {csa2}')
    if have:
        kvs += [
            ('deepseek4.attention.kv_source_layers', 'array', ('u32', list(cfg['kv_source_layer_ids']))),
            ('deepseek4.attention.index_source_layers', 'array', ('u32', list(cfg['index_source_layer_ids']))),
            ('deepseek4.attention.candidate_source_layer', 'u32', cfg['candidate_source_layer_id']),
            ('deepseek4.attention.candidate_topk_blocks', 'u32', cfg['candidate_topk_blocks']),
            ('deepseek4.attention.candidate_block_size', 'u32', cfg['candidate_block_size']),
        ]
    kvs += [
        ('deepseek4.hyper_connection.count', 'u32', cfg['hc_mult']),
        ('deepseek4.hyper_connection.sinkhorn_iterations', 'u32', cfg['hc_sinkhorn_iters']),
        ('deepseek4.hyper_connection.epsilon', 'f32', cfg['hc_eps']),
        # transformers' GenerationConfig defaults when the checkpoint ships no
        # generation_config.json (V4.1); record-only either way.
        ('general.sampling.top_p', 'f32', gen.get('top_p', 1.0)),
        ('general.sampling.temp', 'f32', gen.get('temperature', 1.0)),
    ]
    if reap_map is not None:
        kvs += _reap_triples(reap_map)

    kvs += build_tokenizer_kvs(tokenizer_dir)

    # The drafter: the anchor layers whose INPUT hiddens it conditions on.  No
    # derivation from the layer count -- a drafter trained on other anchors
    # would run and draft garbage (dspark_weights_bind).
    targets = list(cfg['dspark_target_layer_ids'])
    if len(targets) != 3:
        raise SystemExit(f'dspark_target_layer_ids must name 3 layers, got {targets}')
    kvs.append(('deepseek_v4_dspark.embedding_length', 'u32', cfg['hidden_size']))
    kvs += [(f'dspark.target_layer_ids.{i}', 'u32', t) for i, t in enumerate(targets)]

    # Engram (V4.1): the row tables are side files, named here so the loader can
    # refuse a table whose row count is not the checkpoint's (design s4, L242 slice 5).
    if 'engram_layer_ids' in cfg:
        layers = list(cfg['engram_layer_ids'])
        n_rows = list(cfg['engram_num_embeddings'])
        if len(layers) != len(n_rows):
            raise SystemExit(f'engram_layer_ids {layers} and engram_num_embeddings {n_rows} differ in length')
        kvs += [
            ('deepseek4.engram.layers', 'array', ('u32', layers)),
            ('deepseek4.engram.n_rows', 'array', ('u64', n_rows)),
        ]

    out = [entry(k, t, v) for k, t, v in kvs]
    keys = [e['key'] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit('duplicate kv key: ' + ', '.join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out

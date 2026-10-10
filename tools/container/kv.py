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
  * the eleven tokenizer.* entries (tokenizer/core.py, with DeepSeek's TOKENIZER settings below);
  * the drafter's `deepseek_v4_dspark.embedding_length` + `dspark.target_layer_ids.N`
    (dspark_weights_bind reads exactly these four; the drafter's shape is pinned
    against its tensors, weights.cpp:1555-1569);
  * `deepseek4.engram.{layers,n_rows}` when the config declares Engram layers
    (V4.1), naming the side row tables so L242 slice 5 can refuse a mismatch;
    with `build.py --engram-layout` also the hash layout the engine hashes with
    (`deepseek4.engram.{compressed_vocab,pad_compressed_id,token_map,multipliers,
    primes,offsets}`) and each table's file name beside the container
    (`deepseek4.engram.rows_file.N`; engine engram_forward.cpp engram_bind);
  * for a LAYER-SUBSET FIXTURE (`build.py --layers`, names.py drop rule 4) the
    per-layer keys cut to the kept layers, every layer index in block numbers,
    the drafter's keys only when its anchors are kept, and the map itself,
    `pulsar.fixture.source_layers` (engine model_layout.cpp).

NOT emitted, because the direct builder has no source for them: the GGUF
bookkeeping numbers (`general.file_type`, `general.quantization_version`,
`deepseek4.expert_gating_func`) and the quantizer's provenance
(`quantize.imatrix.*`).  Nothing in src/ reads any of them.
"""
import hashlib
import json
import os
import struct

from tokenizer import core as TOK

# DeepSeek's tokenizer settings (the walk is tokenizer/core.py's).  Nine of the eleven tokenizer.* entries derive
# mechanically from tokenizer.json + tokenizer_config.json; THREE DO NOT, so they live here as explicit, documented
# repo-side constants -- OUR decisions, not borrowed artifact bytes:
#   1. PRE_TOKENIZER_ID -- a llama.cpp-side pretokenizer identifier, not present in (or derivable from) any HF file.
#   2. tokenizer/chat_template.jinja -- the checkpoint ships its chat format as PYTHON (encoding/encoding_dsv4.py),
#      not Jinja; our template is a translation of it, a source artifact in its own right.
#   3. USER_DEFINED_TOKENS -- hand-curated; the HF `special` flag does NOT reproduce it (1230/53 there vs the
#      required 1277/6).  CONTROL tokens are skipped during detokenization; these six must survive as literal text
#      because the engine's own parsers consume them (the reasoning split reads <think>/</think>, the DSML tool
#      grammar the dsml markers).  Every other added token (special=False ones like <|fim_hole|> included) is
#      CONTROL.
PRE_TOKENIZER_ID = "joyai-llm"
USER_DEFINED_TOKENS = frozenset({
    "<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:", "<｜/table>｜",
})
TOKENIZER = TOK.Settings(
    added_type=lambda a: TOK.TT_USER_DEFINED if a["content"] in USER_DEFINED_TOKENS else TOK.TT_CONTROL,
    template=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer", "chat_template.jinja"),
    pre=PRE_TOKENIZER_ID)

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


def text_config(hf):
    """The language model's config (V4.1 nests it under text_config)."""
    return _Source(hf).text


def _fixture_map(keep, cfg):
    """A LAYER-SUBSET FIXTURE (build.py --layers; names.py drop rule 4): source layer keep[i] is the container's block
    i.  Returns (block_of, sub): block_of(source layer) -> block or None; sub(per-layer list) -> the kept entries.  The
    builder refuses what the loader would (model_layout.cpp pulsar_attn_layout_install): a kept compressed layer whose
    kv / index source -- the last one at or below it in the source model -- is not kept, which would silently re-bind
    the layer to an earlier cache."""
    if keep is None:
        return (lambda l: l), (lambda v: list(v))
    pos = {l: i for i, l in enumerate(keep)}

    def latest(l, srcs):
        below = [x for x in srcs if x <= l]
        return below[-1] if below else None
    ratios = list(cfg['compress_ratios'])
    for l in keep:
        if not ratios[l]:
            continue
        for what in ('kv_source_layer_ids', 'index_source_layer_ids'):
            src = latest(l, list(cfg.get(what, ())))
            if what in cfg and src not in pos:
                raise SystemExit(f'--layers: layer {l} reads {what[:-10].replace("_", " ")} {src}, which the fixture '
                                 'drops -- keep it (a reader of a dropped source would re-bind to an earlier cache)')
    return pos.get, (lambda v: [v[l] for l in keep])


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


def _engram_layout_triples(path, cfg, rows):
    """The Engram hash layout (tools/engram/engram_layout.py) for the kept Engram layers `rows` (indices into the
    checkpoint's engram_layer_ids), checked against config.json, plus each layer's row table by name (the side file
    tools/engram/engram_rows.c writes, beside the container).  The engine hashes with exactly these numbers
    (pulsar_engram_hash_pos) -- it never re-derives them."""
    lay = _load_json(path)
    for k, ck in (('layer_ids', 'engram_layer_ids'), ('num_embeddings', 'engram_num_embeddings'),
                  ('compressed_vocab_size', 'engram_compressed_vocab_size'), ('max_ngram_size', 'engram_max_ngram_size'),
                  ('n_heads', 'engram_n_heads'), ('head_dim', 'engram_head_dim')):
        if lay[k] != cfg[ck]:
            raise SystemExit(f'{path}: {k} {lay[k]} is not config.json {ck} {cfg[ck]}')
    if len(lay['token_map']) != cfg['vocab_size']:
        raise SystemExit(f'{path}: token_map has {len(lay["token_map"])} entries for vocab {cfg["vocab_size"]}')

    def flat(xs):
        return [v for x in xs for v in (flat(x) if isinstance(x, list) else [x])]
    out = [
        ('deepseek4.engram.compressed_vocab', 'u32', lay['compressed_vocab_size']),
        ('deepseek4.engram.pad_compressed_id', 'u32', lay['pad_compressed_id']),
        ('deepseek4.engram.token_map', 'array', ('i32', list(lay['token_map']))),
        ('deepseek4.engram.multipliers', 'array', ('i64', flat([lay['multipliers'][r] for r in rows]))),
        ('deepseek4.engram.primes', 'array', ('u32', flat([lay['primes'][r] for r in rows]))),
        ('deepseek4.engram.offsets', 'array', ('u64', flat([lay['offsets'][r] for r in rows]))),
    ]
    out += [(f'deepseek4.engram.rows_file.{i}', 'string', f'engram-l{lay["layer_ids"][r]}.rows')
            for i, r in enumerate(rows)]
    return out


def build_kv(hf, tokenizer_dir, reap_map=None, keep=None, engram_layout=None):
    src = _Source(hf)
    cfg = src.text
    gen = src.generation
    L = cfg['num_hidden_layers']
    rope = cfg['rope_scaling']
    block_of, sub = _fixture_map(keep, cfg)
    n_block = len(keep) if keep is not None else L

    kvs = [
        ('general.architecture', 'string', 'deepseek4'),
        ('general.type', 'string', 'model'),
        ('general.name', 'string', _family_name(src.model_type)),
        ('deepseek4.block_count', 'u32', n_block),
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
        ('deepseek4.attention.compress_ratios', 'array', ('u32', sub(_compress_ratios(cfg, L)))),
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
        ('deepseek4.swiglu_clamp_exp', 'array', ('f32', [cfg['swiglu_limit']] * n_block)),
        ('deepseek4.attention.sliding_window', 'u32', cfg['sliding_window']),
        ('deepseek4.attention.indexer.head_count', 'u32', cfg['index_n_heads']),
        ('deepseek4.attention.indexer.key_length', 'u32', cfg['index_head_dim']),
        ('deepseek4.attention.indexer.top_k', 'u32', cfg['index_topk']),
    ]
    # a fixture whose drafter anchors are not all kept carries no drafter (names.py drop rule 4)
    targets = list(cfg['dspark_target_layer_ids'])
    if len(targets) != 3:
        raise SystemExit(f'dspark_target_layer_ids must name 3 layers, got {targets}')
    drafter = all(block_of(t) is not None for t in targets)
    kvs.append(('deepseek4.nextn_predict_layers', 'u32', cfg['num_nextn_predict_layers'] if drafter else 0))
    # CSA2 sharing (V4.1): which layers write the shared caches and the shared
    # top-k, and the candidate pool.  All five or none -- the engine derives the
    # set from the profile when they are absent (validate_attention_layout_metadata).
    csa2 = ('kv_source_layer_ids', 'index_source_layer_ids', 'candidate_source_layer_id',
            'candidate_topk_blocks', 'candidate_block_size')
    have = [k for k in csa2 if k in cfg]
    if have and len(have) != len(csa2):
        raise SystemExit(f'config.json declares only {have} of the CSA2 sharing keys {csa2}')
    if have:
        def kept(srcs):
            return [block_of(x) for x in srcs if block_of(x) is not None]
        cand = block_of(cfg['candidate_source_layer_id'])
        if cand is None and keep is not None and keep[-1] > cfg['candidate_source_layer_id']:
            raise SystemExit(f'--layers: the candidate source {cfg["candidate_source_layer_id"]} is dropped but later '
                             'layers are kept -- keep it')
        # a fixture of window layers only keeps no source: the keys go (the engine reads no empty arrays, and
        # cuts the profile's sets to the same nothing)
        kvs += [(k, 'array', ('u32', kept(cfg[c]))) for k, c in
                (('deepseek4.attention.kv_source_layers', 'kv_source_layer_ids'),
                 ('deepseek4.attention.index_source_layers', 'index_source_layer_ids')) if kept(cfg[c])]
        if cand is not None:
            kvs.append(('deepseek4.attention.candidate_source_layer', 'u32', cand))
        kvs += [
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

    kvs += TOK.kvs(tokenizer_dir, TOKENIZER)

    # The drafter: the anchor layers whose INPUT hiddens it conditions on.  No
    # derivation from the layer count -- a drafter trained on other anchors
    # would run and draft garbage (dspark_weights_bind).
    if drafter:
        kvs.append(('deepseek_v4_dspark.embedding_length', 'u32', cfg['hidden_size']))
        kvs += [(f'dspark.target_layer_ids.{i}', 'u32', block_of(t)) for i, t in enumerate(targets)]

    # Engram (V4.1): the row tables are side files, named here so the loader can
    # refuse a table whose row count is not the checkpoint's (design s4, L242 slice 5).
    if 'engram_layer_ids' in cfg:
        if len(cfg['engram_layer_ids']) != len(cfg['engram_num_embeddings']):
            raise SystemExit(f'engram_layer_ids {cfg["engram_layer_ids"]} and engram_num_embeddings '
                             f'{cfg["engram_num_embeddings"]} differ in length')
        rows = [r for r, l in enumerate(cfg['engram_layer_ids']) if block_of(l) is not None]
    if 'engram_layer_ids' in cfg and rows:          # a fixture may keep none (the engine reads no empty arrays)
        kvs += [
            ('deepseek4.engram.layers', 'array', ('u32', [block_of(cfg['engram_layer_ids'][r]) for r in rows])),
            ('deepseek4.engram.n_rows', 'array', ('u64', [cfg['engram_num_embeddings'][r] for r in rows])),
        ]
        if engram_layout is not None:
            kvs += _engram_layout_triples(engram_layout, cfg, rows)
    elif engram_layout is not None:
        raise SystemExit('--engram-layout: this build carries no Engram layer')
    if keep is not None:
        kvs.append(('pulsar.fixture.source_layers', 'array', ('u32', list(keep))))

    out = [entry(k, t, v) for k, t, v in kvs]
    keys = [e['key'] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit('duplicate kv key: ' + ', '.join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out

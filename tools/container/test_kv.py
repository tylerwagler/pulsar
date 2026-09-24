#!/usr/bin/env python3
"""test_kv.py -- build_kv reproduces the served pulsar.kv block entry for entry.

Plain python, no GPU.  Two legs:

  A. The served source-precision repack (SERVED) carries `pulsar.kv` in one
     shard's `__metadata__`.  build_kv on the HF checkpoint it was built from
     (VEXP) must reproduce that array ENTRY FOR ENTRY -- same keys in the same
     order, same type strings, same values (arrays element-wise, strings
     byte-equal, floats equal after the f32 round) -- once the served block's
     UNREPRODUCIBLE keys are removed.  That set is named exactly: every member
     must be in the served block, and none may be in the built one.
  B. build_kv on the V4.1 checkpoint (V41): every key weights.cpp requires is
     present with the type the reader asks for, hash_layer_count is absent, and
     the V4.1-specific values are printed.

Exit status is non-zero on any difference; the first differences are printed.
"""
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv import build_kv, f32  # noqa: E402

SERVED = '/mnt/models/DeepSeek-v4-Flash'
VEXP = '/mnt/models/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/*/'
V41 = '/mnt/models/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/'

# Served keys the direct builder has no source for: GGUF bookkeeping numbers and
# the quantizer's provenance.  Nothing in src/ reads any of them (kv.py docstring).
UNREPRODUCIBLE = (
    'general.file_type',
    'general.quantization_version',
    'deepseek4.expert_gating_func',
    'quantize.imatrix.sha256',
    'quantize.imatrix.entries_count',
    'quantize.imatrix.dataset',
    'quantize.imatrix.chunks_count',
)

# What the engine reads, with the type each reader asks for
# (src/engine/weights.cpp config_validate_model, validate_attention_layout_metadata,
# validate_swiglu_clamp_metadata, dspark_weights_bind; src/engine/tokenizer.cpp vocab_load).
ENGINE_REQUIRED = {
    'deepseek4.block_count': 'u32',
    'deepseek4.embedding_length': 'u32',
    'deepseek4.vocab_size': 'u32',
    'deepseek4.attention.head_count': 'u32',
    'deepseek4.attention.head_count_kv': 'u32',
    'deepseek4.attention.key_length': 'u32',
    'deepseek4.attention.value_length': 'u32',
    'deepseek4.rope.dimension_count': 'u32',
    'deepseek4.attention.q_lora_rank': 'u32',
    'deepseek4.attention.output_lora_rank': 'u32',
    'deepseek4.attention.output_group_count': 'u32',
    'deepseek4.expert_count': 'u32',
    'deepseek4.expert_used_count': 'u32',
    'deepseek4.expert_feed_forward_length': 'u32',
    'deepseek4.expert_shared_count': 'u32',
    'deepseek4.attention.sliding_window': 'u32',
    'deepseek4.attention.indexer.head_count': 'u32',
    'deepseek4.attention.indexer.key_length': 'u32',
    'deepseek4.attention.indexer.top_k': 'u32',
    'deepseek4.hyper_connection.count': 'u32',
    'deepseek4.hyper_connection.sinkhorn_iterations': 'u32',
    'deepseek4.attention.compress_ratios': 'array:u32',
    'deepseek4.swiglu_clamp_exp': 'array:f32',
    'deepseek4.rope.freq_base': 'f32',
    'deepseek4.attention.compress_rope_freq_base': 'f32',
    'deepseek4.expert_weights_scale': 'f32',
    'deepseek4.attention.layer_norm_rms_epsilon': 'f32',
    'deepseek4.hyper_connection.epsilon': 'f32',
    'deepseek4.expert_weights_norm': 'bool',
    'tokenizer.ggml.tokens': 'array:string',
    'tokenizer.ggml.merges': 'array:string',
    'deepseek_v4_dspark.embedding_length': 'u32',
    'dspark.target_layer_ids.0': 'u32',
    'dspark.target_layer_ids.1': 'u32',
    'dspark.target_layer_ids.2': 'u32',
}
# Read when present; present must agree with the profile.
ENGINE_OPTIONAL = {
    'deepseek4.rope.scaling.original_context_length': 'u32',   # u64-or-u32 reader
    'deepseek4.rope.scaling.factor': 'f32',
    'deepseek4.rope.scaling.yarn_beta_fast': 'f32',
    'deepseek4.rope.scaling.yarn_beta_slow': 'f32',
    'deepseek4.hash_layer_count': 'u32',
    'deepseek4.attention.kv_source_layers': 'array:u32',
    'deepseek4.attention.index_source_layers': 'array:u32',
    'deepseek4.attention.candidate_source_layer': 'u32',
    'deepseek4.attention.candidate_topk_blocks': 'u32',
    'deepseek4.attention.candidate_block_size': 'u32',
    'reap.enabled': 'bool',
    'reap.layer.keep_count': 'array:u32',
    'general.name': 'string',          # load summary only (model.cpp)
    'general.architecture': 'string',
    'deepseek4.context_length': 'u32',  # summary reads u64 -> blank today (design s2)
}
# V4.1 must carry the CSA2 sharing set (its profile has sources and a candidate pool).
V41_REQUIRED_OPTIONAL = (
    'deepseek4.attention.kv_source_layers', 'deepseek4.attention.index_source_layers',
    'deepseek4.attention.candidate_source_layer', 'deepseek4.attention.candidate_topk_blocks',
    'deepseek4.attention.candidate_block_size',
)


def one_dir(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise SystemExit(f'{pattern}: expected exactly one snapshot, found {hits}')
    return hits[0]


def served_kv(model_dir):
    carriers = []
    for p in sorted(glob.glob(os.path.join(model_dir, '*.safetensors'))):
        with open(p, 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            hdr = json.loads(f.read(n))
        md = hdr.get('__metadata__') or {}
        if 'pulsar.kv' in md:
            carriers.append((p, json.loads(md['pulsar.kv'])))
    if len(carriers) != 1:
        raise SystemExit(f'{model_dir}: expected exactly one shard carrying pulsar.kv, found {[c[0] for c in carriers]}')
    return carriers[0]


def type_of(e):
    return f"array:{e['value']['__array__']}" if e['type'] == 'array' else e['type']


def short(v):
    s = repr(v)
    return s if len(s) <= 100 else s[:100] + '...'


def diff_entry(i, a, b):
    """First difference between served entry a and built entry b, or None."""
    if a['key'] != b['key']:
        return f'[{i}] key: served {a["key"]!r}, built {b["key"]!r}'
    if a['type'] != b['type']:
        return f'[{i}] {a["key"]}: type served {a["type"]}, built {b["type"]}'
    if a['type'] == 'array':
        av, bv = a['value'], b['value']
        if av['__array__'] != bv['__array__']:
            return f'[{i}] {a["key"]}: elem type served {av["__array__"]}, built {bv["__array__"]}'
        if av['n'] != bv['n'] or len(av['v']) != av['n'] or len(bv['v']) != bv['n']:
            return f'[{i}] {a["key"]}: n served {av["n"]}/{len(av["v"])}, built {bv["n"]}/{len(bv["v"])}'
        for j, (x, y) in enumerate(zip(av['v'], bv['v'])):
            d = diff_value(f'[{i}] {a["key"]}[{j}]', av['__array__'], x, y)
            if d:
                return d
        return None
    return diff_value(f'[{i}] {a["key"]}', a['type'], a['value'], b['value'])


def diff_value(where, typ, x, y):
    """Served x vs built y.  An f32 must ALREADY be the binary32 double on both
    sides -- that double's repr is what json.dumps writes and the engine reads --
    so the round is asserted, never applied before comparing."""
    if typ == 'f32':
        if x != f32(x):
            return f'{where}: served value {x!r} is not an f32 (artifact bug)'
        if y != f32(y):
            return f'{where}: built value {y!r} is not rounded to f32 (repr trap)'
    if type(x) is not type(y) or x != y:
        return f'{where}: served {short(x)}, built {short(y)}'
    return None


def leg_a():
    shard, served = served_kv(SERVED)
    vexp = one_dir(VEXP)
    print(f'LEG A: {shard} ({len(served)} entries) vs build_kv({vexp})')
    built = build_kv(vexp, vexp)
    served_keys = [e['key'] for e in served]
    built_keys = [e['key'] for e in built]
    bad = []
    for k in UNREPRODUCIBLE:
        if k not in served_keys:
            bad.append(f'UNREPRODUCIBLE names {k}, which the served block does not carry')
        if k in built_keys:
            bad.append(f'{k} is named unreproducible but build_kv emitted it')
    expected = [e for e in served if e['key'] not in UNREPRODUCIBLE]
    if len(expected) != len(built):
        bad.append(f'length: served-minus-unreproducible {len(expected)}, built {len(built)}; '
                   f'served-only {sorted(set(k for k in served_keys if k not in UNREPRODUCIBLE) - set(built_keys))}, '
                   f'built-only {sorted(set(built_keys) - set(served_keys))}')
    for i, (a, b) in enumerate(zip(expected, built)):
        d = diff_entry(i, a, b)
        if d:
            bad.append(d)
            if len(bad) >= 10:
                break
    bad += check_types(built, ENGINE_REQUIRED, 'Vision-Exp')
    if built_keys.count('deepseek4.hash_layer_count') != 1 or \
            next(e for e in built if e['key'] == 'deepseek4.hash_layer_count')['value'] != 3:
        bad.append('Vision-Exp: hash_layer_count must be the config\'s num_hash_layers (3)')
    # The whole point of the f32 round: the JSON text must carry the served repr.
    text = json.dumps(built, ensure_ascii=False)
    for probe in ('9.999999682655225e-21', '9.999999974752427e-07'):
        if probe not in text:
            bad.append(f'json.dumps(built) lacks the f32 repr {probe}')
    print(f'  served {len(served)} = {len(expected)} reproducible + {len(UNREPRODUCIBLE)} unreproducible '
          f'({", ".join(UNREPRODUCIBLE)}); built {len(built)}')
    if bad:
        print('  FAIL, first differences:')
        for d in bad:
            print('   ', d)
        return False
    print(f'  PASS: {len(built)} entries identical, key order identical, types identical')
    return True


def check_types(block, required, label):
    by_key = {e['key']: e for e in block}
    bad = []
    for k, t in required.items():
        if k not in by_key:
            bad.append(f'{label}: missing required key {k} ({t})')
        elif type_of(by_key[k]) != t:
            bad.append(f'{label}: {k} is {type_of(by_key[k])}, reader wants {t}')
    for k, t in ENGINE_OPTIONAL.items():
        if k in by_key and type_of(by_key[k]) != t:
            bad.append(f'{label}: {k} is {type_of(by_key[k])}, reader wants {t}')
    return bad


def leg_b():
    v41 = one_dir(V41)
    print(f'LEG B: build_kv({v41})')
    block = build_kv(v41, v41)
    by_key = {e['key']: e for e in block}
    bad = check_types(block, ENGINE_REQUIRED, 'V4.1')
    for k in V41_REQUIRED_OPTIONAL:
        if k not in by_key:
            bad.append(f'V4.1: missing CSA2 key {k}')
    if 'deepseek4.hash_layer_count' in by_key:
        bad.append('V4.1: hash_layer_count emitted, but V4.1 has no hash-routed layers')
    cfg = json.load(open(os.path.join(v41, 'config.json')))['text_config']
    print(f'  {len(block)} entries; keys:')
    for e in block:
        v = e['value']
        if e['type'] == 'array':
            shown = f"array<{v['__array__']}> n={v['n']} {short(v['v'])}"
        elif e['type'] == 'string':
            shown = short(v)
        else:
            shown = repr(v)
        print(f"    {e['key']:<52} {e['type']:<6} {shown}")
    print('  V4.1 specifics:')
    for k in ('deepseek4.block_count', 'deepseek4.embedding_length', 'deepseek4.expert_count',
              'deepseek4.expert_used_count', 'deepseek4.expert_feed_forward_length',
              'deepseek4.attention.q_lora_rank', 'deepseek4.attention.indexer.head_count',
              'deepseek4.attention.layer_norm_rms_epsilon', 'deepseek4.attention.compress_ratios',
              'deepseek4.attention.kv_source_layers', 'deepseek4.attention.index_source_layers',
              'deepseek4.attention.candidate_source_layer', 'deepseek4.attention.candidate_topk_blocks',
              'deepseek4.attention.candidate_block_size', 'deepseek4.engram.layers', 'deepseek4.engram.n_rows',
              'deepseek_v4_dspark.embedding_length', 'dspark.target_layer_ids.0',
              'dspark.target_layer_ids.1', 'dspark.target_layer_ids.2', 'general.sampling.top_p'):
        e = by_key.get(k)
        print(f'    {k:<52} {"<absent>" if e is None else short(e["value"])}')
    print('  drafter shape in config.json (pinned against the drafter tensors by the engine, not emitted):')
    for k in ('dspark_n_routed_experts', 'dspark_num_experts_per_tok', 'dspark_block_size',
              'dspark_noise_token_id', 'dspark_markov_rank'):
        print(f'    {k:<52} {cfg.get(k)}')
    print('  engram config.json (side files; only layers + n_rows are named in the block):')
    for k in sorted(k for k in cfg if k.startswith('engram_')):
        print(f'    {k:<52} {cfg[k]}')
    if bad:
        print('  FAIL:')
        for d in bad:
            print('   ', d)
        return False
    print(f'  PASS: every key weights.cpp/tokenizer.cpp requires is present with the reader\'s type')
    return True


def main():
    ok_a = leg_a()
    ok_b = leg_b()
    print('RESULT:', 'PASS' if ok_a and ok_b else 'FAIL')
    return 0 if ok_a and ok_b else 1


if __name__ == '__main__':
    sys.exit(main())

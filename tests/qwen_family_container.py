#!/usr/bin/env python3
"""qwen_family_container.py -- a zero-weight qwen4_exp container for the family
gate (L251 S1).

Reads ONLY the headers and config of a Hugging Face Qwen3.8-Flash-Next
checkpoint (no weight bytes; the three int64 PLE buffers are 280 bytes and are
read) and writes one SPARSE safetensors file carrying the container contract
the engine's Qwen family binds (src/engine/family_qwen.h section 2):

  general.architecture = "qwen4_exp", every text_config key verbatim under
  "qwen4_exp." (nested dicts flattened with '.'), the PLE int64 buffers as u64
  arrays, and every text-model tensor under its HF name as gguf_name, bf16,
  dims in ne order.  The data region is a hole: nothing reads it at --inspect.

The metadata entries go through tools/container/kv.py's `entry` -- the shipping
encoder -- so this file cannot drift from what the builder writes.

--mutate makes one fault the loader must refuse by name:
  arch        general.architecture is an unknown family
  shape       hidden_size is 2048
  tensor      one layer's in_proj_qkv is dropped
  layer-type  layer_types[5] is "sliding_attention"
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'container'))
from kv import entry  # noqa: E402

TEXT = 'model.language_model.'
PLE_BUFFERS = {
    'ple_embedding.layer_multipliers': 'ple_layer_multipliers',
    'ple_embedding.ngram_heads_vocab_sizes': 'ple_ngram_heads_vocab_sizes',
    'ple_embedding.ngram_heads_offsets': 'ple_ngram_heads_offsets',
}


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def kv_entries(key, value, out):
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            kv_entries(f'{key}.{k}', v, out)
    elif isinstance(value, bool):
        out.append(entry(key, 'bool', value))
    elif isinstance(value, int):
        out.append(entry(key, 'u32' if 0 <= value <= 0xFFFFFFFF else 'u64' if value >= 0 else 'i64', value))
    elif isinstance(value, float):
        out.append(entry(key, 'f32', value))
    elif isinstance(value, str):
        out.append(entry(key, 'string', value))
    elif isinstance(value, list):
        if not value:
            return
        if all(isinstance(v, str) for v in value):
            out.append(entry(key, 'array', ('string', value)))
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            out.append(entry(key, 'array', ('u64' if max(value) > 0xFFFFFFFF else 'u32', value)))
        else:
            raise SystemExit(f'{key}: mixed list {value!r}')
    else:
        raise SystemExit(f'{key}: unsupported config value {value!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('hf_dir')
    ap.add_argument('out')
    ap.add_argument('--mutate', choices=['none', 'arch', 'shape', 'tensor', 'layer-type'], default='none')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.hf_dir, 'config.json')))
    text = dict(cfg['text_config'])
    if a.mutate == 'shape':
        text['hidden_size'] = 2048
    if a.mutate == 'layer-type':
        text['layer_types'] = list(text['layer_types'])
        text['layer_types'][5] = 'sliding_attention'
    kvs = [entry('general.architecture', 'string', 'qwen9_bogus' if a.mutate == 'arch' else 'qwen4_exp'),
           entry('general.name', 'string', 'Qwen3.8-Flash-Next (zero-weight family gate container)')]
    for k, v in text.items():
        kv_entries(f'qwen4_exp.{k}', v, kvs)

    index = json.load(open(os.path.join(a.hf_dir, 'model.safetensors.index.json')))['weight_map']
    headers = {}
    tensors = {}
    for name, shard in sorted(index.items()):
        if shard not in headers:
            headers[shard] = read_header(os.path.join(a.hf_dir, shard))
        hdr, base = headers[shard]
        info = hdr[name]
        if not (name.startswith(TEXT) or name == 'lm_head.weight'):
            continue                                  # vision tower, MTP: not in this family's bind yet
        if '.ngram_embedding.' in name:
            continue                                  # the n-gram tables are disk tables, never container tensors
        buf = next((k for k in PLE_BUFFERS if name.endswith(k)), None)
        if buf:
            off0, off1 = info['data_offsets']
            with open(os.path.join(a.hf_dir, shard), 'rb') as f:
                f.seek(base + off0)
                raw = f.read(off1 - off0)
            vals = list(struct.unpack(f'<{len(raw) // 8}q', raw))
            kvs.append(entry(f'qwen4_exp.{PLE_BUFFERS[buf]}', 'array', ('u64', vals)))
            continue
        if info['dtype'] != 'BF16':
            raise SystemExit(f'{name}: dtype {info["dtype"]} (only BF16 tensors are expected here)')
        if a.mutate == 'tensor' and name == TEXT + 'layers.4.linear_attn.in_proj_qkv.weight':
            continue
        tensors[name] = list(info['shape'])

    meta_tensors = {n: {'layout': 'bf16', 'gguf_name': n, 'dims_ne': list(reversed(s))} for n, s in tensors.items()}
    header = {'__metadata__': {'pulsar.kv': json.dumps(kvs), 'pulsar.tensors': json.dumps(meta_tensors)}}
    off = 0
    for n, s in tensors.items():
        nbytes = 2
        for d in s:
            nbytes *= d
        header[n] = {'dtype': 'BF16', 'shape': s, 'data_offsets': [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    blob += b' ' * (-len(blob) % 8)
    with open(a.out, 'wb') as f:
        f.write(struct.pack('<Q', len(blob)))
        f.write(blob)
        f.truncate(8 + len(blob) + off)            # the data region is a hole
    print(f'{a.out}: {len(tensors)} tensors, {len(kvs)} metadata keys, {off / 2**30:.1f} GiB sparse ({a.mutate})')


if __name__ == '__main__':
    main()

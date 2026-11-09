#!/usr/bin/env python3
"""container_edit.py -- reading and editing a loaded container's pulsar.tensors records and pulsar.kv, for a family gate's mutants."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader_contract as LC  # noqa: E402  (the header edit)


def records(shards):
    """[(shard index, container name, record)] of every pulsar.tensors entry"""
    out = []
    for i, s in enumerate(shards):
        for cname, rec in json.loads(s['header'].get('__metadata__', {}).get('pulsar.tensors', '{}')).items():
            out.append((i, cname, rec))
    return out


def find(shards, role, layer=None, index=None):
    for i, cname, rec in records(shards):
        if rec.get('role') == role and (layer is None or rec.get('layer') == layer) and \
                (index is None or rec.get('index') == index):
            return i, cname, rec
    sys.exit(f'container_edit: the container has no {role} (layer {layer}, index {index}) to mutate')


def kv_of(shards):
    for s in shards:
        kv = s['header'].get('__metadata__', {}).get('pulsar.kv')
        if kv:
            return {e['key']: e for e in json.loads(kv)}
    sys.exit('container_edit: no pulsar.kv')


def edit_kv(shards, fn):
    for s in shards:
        for key in ('pulsar.kv', 'pulsar.kv_arch'):
            if key in s['header'].get('__metadata__', {}):
                LC.edit_meta_json(s, key, fn)


def set_kv(shards, key, typ, value):
    def fn(kv):
        out = [e for e in kv if e['key'] != key]
        if value is not None:
            out.append({'key': key, 'type': typ, 'value': value})
        return out
    edit_kv(shards, fn)


def edit_record(shards, i, cname, fn):
    LC.edit_meta_json(shards[i], 'pulsar.tensors', lambda t: {k: (fn(dict(v)) if k == cname else v) for k, v in t.items()})

#!/usr/bin/env python3
"""loader_contract.py -- the loader contract, for every family (L278 contract 2).

A family's loader must REFUSE a broken artifact by name, at a stage boundary, never exit the process and never
load it (L272 f31f4188, one failure policy).  This gate proves it on each family's real served container without
reading a weight byte:

  1. a HEADER CLONE: every shard's safetensors header (tensor table + the pulsar metadata) copied into a sparse
     file of the same size -- the data region is a hole -- plus the small side files (tokenizer.json, configs) and
     a sparse clone of any large side file (Qwen's PLE row file keeps its header, the rows are a hole);
  2. the clean clone must open (tests/loader_open: pulsar_engine_open inspect-only): the clone IS the artifact as
     the loader sees it;
  3. each MUTANT -- the clean clone with one fault -- must be REFUSED: the open RETURNS an error (loader_open prints
     its marker after the call comes back; a missing marker is a loader that exited the process) AND the log names
     the fault (the key, or the tensor by its engine or container name).

The mutations are the family-neutral faults every loader must catch:

  arch          general.architecture names no family
  kv            a required metadata key is missing
  tensor        a required tensor is missing
  dtype         a packed tensor's header dtype disagrees with its layout
  shape         a required 2-D tensor is transposed (same bytes, wrong dims: only the binder can see it)
  native-dtype  a plain tensor is stored in a dtype no reader takes
  format        a routed expert stack's layout is one no arm reads (the format registry's admission, L272 P4)

The targets (which tensor, which key, which stack) are the family's row in FAMILIES below -- the only family
knowledge here.  A family adds a row; a family-only fault (Qwen's PLE row file) is an extra in its row.

  loader_contract.py --open tests/loader_open --work DIR MODEL [MODEL ...]
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

# Each family's fixture row: the faults' targets, by the ENGINE's name (gguf_name), and the refusal text each fault
# must produce.  `expect` strings are what the loader says today; a reworded refusal changes them here, on purpose.
FAMILIES = {
    'deepseek4': {
        'tensor': 'blk.1.attn_q_a.weight',          # a packed (mxfp8_lt) projection every layer has
        'plain': 'blk.1.ffn_gate_inp.weight',       # a 2-D plain tensor: the router, bf16 [4096, 256]
        'kv': 'deepseek4.block_count',
        'stack': 'blk.3.ffn_gate_exps.weight',
    },
    'qwen4_exp': {
        'tensor': 'model.language_model.layers.1.linear_attn.in_proj_qkv.weight',
        'plain': 'model.language_model.layers.1.mlp.gate.weight',   # the router, bf16 [2560, 512]
        'kv': 'qwen4_exp.num_hidden_layers',
        'stack': 'model.language_model.layers.3.mlp.experts.gate_up_proj',
    },
}
MUTATIONS = ('arch', 'kv', 'tensor', 'dtype', 'shape', 'native-dtype', 'format')
REFUSAL_WORDS = ('refus', 'missing', 'declares', 'expected', 'unsupported', 'disagrees', 'not ', 'unknown', 'no arm', 'want ',
                 'does not')
SMALL_SIDE_FILE = 64 << 20      # side files up to this size are copied; larger ones are cloned sparse
SPARSE_PREFIX = 1 << 20         # the bytes a sparse side-file clone keeps (its header)


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_shard(path, header, data_bytes, align):
    """the header (padded with spaces so the data region starts aligned, as the builder writes it) and a hole for
    the data region -- the same file size as the source"""
    raw = json.dumps(header, separators=(',', ':')).encode()
    pad = (-(8 + len(raw))) % align
    raw += b' ' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + data_bytes)


def load_container(src):
    shards = sorted(x for x in os.listdir(src) if x.endswith('.safetensors'))
    if not shards:
        sys.exit(f'loader_contract: {src} holds no .safetensors shards')
    out = []
    for name in shards:
        path = os.path.join(src, name)
        header, data_off = read_header(path)
        out.append({'name': name, 'header': header, 'data_bytes': os.path.getsize(path) - data_off})
    return out


def family_of(shards):
    for s in shards:
        kv = s['header'].get('__metadata__', {}).get('pulsar.kv')
        if kv:
            for e in json.loads(kv):
                if e['key'] == 'general.architecture':
                    return e['value']
    sys.exit('loader_contract: no general.architecture in any shard\'s pulsar.kv')


def find_tensor(shards, gguf_name):
    """(shard index, container name) of the tensor the engine calls gguf_name"""
    for i, s in enumerate(shards):
        tensors = json.loads(s['header'].get('__metadata__', {}).get('pulsar.tensors', '{}'))
        for cname, d in tensors.items():
            if d.get('gguf_name') == gguf_name:
                return i, cname
    return None, None


def edit_meta_json(shard, key, fn):
    meta = shard['header']['__metadata__']
    meta[key] = json.dumps(fn(json.loads(meta[key])), separators=(',', ':'))


def mutate(shards, family, row, what):
    """apply one fault in place; returns (a one-line description, the names a refusal of it may say)"""
    if what == 'arch':
        for s in shards:
            for key in ('pulsar.kv', 'pulsar.kv_arch'):
                if key in s['header'].get('__metadata__', {}):
                    edit_meta_json(s, key, lambda kv: [dict(e, value='nonesuch') if e['key'] == 'general.architecture'
                                                       else e for e in kv])
        return 'general.architecture = nonesuch', ['nonesuch']
    if what == 'kv':
        hit = False
        for s in shards:
            for key in ('pulsar.kv', 'pulsar.kv_arch'):
                if key in s['header'].get('__metadata__', {}):
                    before = json.loads(s['header']['__metadata__'][key])
                    after = [e for e in before if e['key'] != row['kv']]
                    hit |= len(after) != len(before)
                    s['header']['__metadata__'][key] = json.dumps(after, separators=(',', ':'))
        if not hit:
            sys.exit(f'loader_contract: {family} has no metadata key {row["kv"]} to drop')
        return f'{row["kv"]} dropped', [row['kv']]
    if what in ('tensor', 'dtype', 'shape', 'native-dtype'):
        target = row['plain'] if what in ('shape', 'native-dtype') else row['tensor']
        i, cname = find_tensor(shards, target)
        if i is None:
            sys.exit(f'loader_contract: {family} has no tensor {target}')
        s = shards[i]
        names = [target, cname]
        if what == 'tensor':
            del s['header'][cname]
            edit_meta_json(s, 'pulsar.tensors', lambda t: {k: v for k, v in t.items() if k != cname})
            return f'{target} dropped', names
        if what == 'shape':
            # transposed in both tables: the same element count and bytes, so the file is consistent -- only the
            # binder, checking the dims its reader needs, can refuse it
            edit_meta_json(s, 'pulsar.tensors', lambda t: {k: (dict(v, dims_ne=list(reversed(v['dims_ne'])))
                                                              if k == cname else v) for k, v in t.items()})
            s['header'][cname]['shape'] = list(reversed(s['header'][cname]['shape']))
            return f'{target} transposed', names
        s['header'][cname]['dtype'] = 'F64'
        return f'{target} stored as F64', names
    if what == 'format':
        stack = row['stack']
        for s in shards:
            meta = s['header'].get('__metadata__', {})
            recs = json.loads(meta.get('pulsar.experts', '[]'))
            if any(r.get('gguf_name') == stack for r in recs):
                edit_meta_json(s, 'pulsar.experts', lambda rs: [dict(r, layout='bf16') if r.get('gguf_name') == stack
                                                                else r for r in rs])
                return f'{stack} layout bf16 (expert record)', [stack]
        i, cname = find_tensor(shards, stack)
        if i is None:
            sys.exit(f'loader_contract: {family} has no expert stack {stack}')
        edit_meta_json(shards[i], 'pulsar.tensors', lambda t: {k: (dict(v, layout='bf16') if k == cname else v)
                                                              for k, v in t.items()})
        return f'{stack} layout bf16', [stack, cname]
    sys.exit(f'loader_contract: unknown mutation {what}')


def write_clone(src, shards, out):
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for s in shards:
        align = int(s['header'].get('__metadata__', {}).get('pulsar.alignment', '32'))
        write_shard(os.path.join(out, s['name']), s['header'], s['data_bytes'], align)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if name.endswith('.safetensors') or not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        if size <= SMALL_SIDE_FILE:
            shutil.copyfile(p, os.path.join(out, name))
        else:
            with open(p, 'rb') as f:
                head = f.read(SPARSE_PREFIX)
            with open(os.path.join(out, name), 'wb') as f:
                f.write(head)
                f.truncate(size)


def open_model(probe, model, lock):
    """(returned: the open came back, rc, the log, seconds)"""
    env = dict(os.environ, PULSAR_LOCK_FILE=lock)
    t0 = time.time()
    p = subprocess.run([probe, model], capture_output=True, text=True, env=env, timeout=600)
    log = p.stdout + p.stderr
    return 'LOADER_OPEN_RETURNED' in log, p.returncode, log, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--open', required=True, help='tests/loader_open')
    ap.add_argument('--work', required=True, help='a scratch directory for the clones (sparse: ~no disk)')
    ap.add_argument('models', nargs='+')
    a = ap.parse_args()
    fails = 0
    lock = os.path.join(a.work, 'inspect.lock')
    os.makedirs(a.work, exist_ok=True)
    for src in a.models:
        shards = load_container(src)
        family = family_of(shards)
        row = FAMILIES.get(family)
        if not row:
            print(f'  FAIL  {src}: family {family} has no row in FAMILIES -- the loader contract cannot run for it')
            fails += 1
            continue
        tag = os.path.basename(os.path.normpath(src))
        clean = os.path.join(a.work, f'{tag}-clean')
        write_clone(src, shards, clean)
        returned, rc, log, dt = open_model(a.open, clean, lock)
        if not returned or rc != 0:
            print(f'  FAIL  {family} {tag}: the clean header clone does not open (rc {rc}, {dt:.1f} s):')
            print('\n'.join('        ' + x for x in log.strip().splitlines()[-6:]))
            fails += 1
            continue
        print(f'  ok    {family} {tag}: the clean header clone opens ({dt:.1f} s)')
        for what in MUTATIONS:
            m = load_container(src)
            desc, names = mutate(m, family, row, what)
            out = os.path.join(a.work, f'{tag}-{what}')
            write_clone(src, m, out)
            returned, rc, log, dt = open_model(a.open, out, lock)
            said = [x for x in log.splitlines() if any(n in x for n in names) and any(w in x for w in REFUSAL_WORDS)]
            if returned and rc == 0:
                print(f'  FAIL  {family} {what}: {desc} -- OPENED (the loader must refuse it)')
                fails += 1
            elif not returned:
                why = said[0].strip()[:150] if said else 'no named reason'
                print(f'  FAIL  {family} {what}: {desc} -- the loader EXITED the process (rc {rc}) instead of '
                      f'refusing: {why}')
                fails += 1
            elif not said:
                print(f'  FAIL  {family} {what}: {desc} -- refused (rc {rc}) without naming {names[0]!r}; it said:')
                print('\n'.join('        ' + x for x in log.strip().splitlines()[-4:]))
                fails += 1
            else:
                print(f'  ok    {family} {what}: {desc} -- refused by name ({dt:.1f} s): {said[0].strip()[:150]}')
            shutil.rmtree(out)
        shutil.rmtree(clean)
    print('LOADER-CONTRACT GATE ' + ('PASS' if fails == 0 else f'FAIL ({fails})'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

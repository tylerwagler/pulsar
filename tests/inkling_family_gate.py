#!/usr/bin/env python3
"""inkling_family_gate.py -- L295 E1: the Inkling family's load, on a built container.

  inkling_family_gate.py --probe tests/inkling_family_gate --open tests/loader_open --work DIR MODEL [--refuse MODEL ...]

1. The REAL container: tests/inkling_family_gate opens it inspect-only and checks every claim of the load (the plan,
   every tensor bound by its bind record, every byte priced, the slots, the facts, the session refusals).
2. Its HEADER CLONE (tests/loader_contract.py's: the headers and metadata, the data region a hole): the same probe
   -- the load reads no weight byte it does not need, so the clone IS the artifact as the loader sees it.
3. Each Inkling MUTANT of the clone must be REFUSED BY NAME, the open returning (tests/loader_open): a missing tensor,
   a wrong shape, an unknown switch, a flipped switch, a geometry the profile does not have, a bad plan list, a
   tensor with no bind record, an undeclared tensor, two tensors with one record, a linear in a format no arm reads,
   a drafter the config does not describe.  (The family-neutral faults are loader_contract.py's `inkling` row.)
4. --refuse MODEL: a container the family must refuse as a whole (the BF16 build: its routed stacks have no arm at
   bf16 rows until L295 F7) -- the open returns an error and names why.

The targets are the container's own names (the builder's: the HF / EXL3-pack names), chosen from the clone's
records by bind role -- the only Inkling knowledge here."""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader_contract as LC  # noqa: E402  (the header clone, the open probe)


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
    sys.exit(f'inkling_family_gate: the container has no {role} (layer {layer}, index {index}) to mutate')


def kv_of(shards):
    for s in shards:
        kv = s['header'].get('__metadata__', {}).get('pulsar.kv')
        if kv:
            return {e['key']: e for e in json.loads(kv)}
    sys.exit('inkling_family_gate: no pulsar.kv')


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


def mutate(shards, what):
    """apply one fault; returns (description, [strings the refusal must name])"""
    kv = kv_of(shards)
    n_layer = kv['inkling.num_hidden_layers']['value']
    n_dense = kv['inkling.dense_mlp_idx']['value']
    moe = n_dense            # the first MoE layer
    if what == 'missing':
        i, cname, rec = find(shards, 'attn_rel_basis', layer=n_layer - 1)
        del shards[i]['header'][cname]
        LC.edit_meta_json(shards[i], 'pulsar.tensors', lambda t: {k: v for k, v in t.items() if k != cname})
        return f'{cname} dropped', ['attn_rel_basis']
    if what == 'shape':
        i, cname, rec = find(shards, 'attn_rel_basis', layer=0)
        edit_record(shards, i, cname, lambda r: dict(r, dims_ne=list(reversed(r['dims_ne']))))
        shards[i]['header'][cname]['shape'] = list(reversed(shards[i]['header'][cname]['shape']))
        return f'{cname} transposed', [rec['gguf_name']]
    if what == 'switch-unknown':
        set_kv(shards, 'inkling.use_rope', 'bool', True)
        return 'inkling.use_rope = true added', ['inkling.use_rope']
    if what == 'switch-flipped':
        set_kv(shards, 'inkling.use_sconv', 'bool', False)
        return 'inkling.use_sconv = false', ['inkling.use_sconv']
    if what == 'geometry':
        set_kv(shards, 'inkling.d_rel', 'u32', 32)
        return 'inkling.d_rel = 32', ['inkling.d_rel']
    if what == 'plan':
        set_kv(shards, 'inkling.local_layer_ids', 'array', {'__array__': 'u32', 'n': 2, 'v': [1, 0]})
        return 'inkling.local_layer_ids = [1, 0]', ['inkling.local_layer_ids']
    if what == 'no-role':
        i, cname, rec = find(shards, 'output_norm')
        edit_record(shards, i, cname, lambda r: {k: v for k, v in r.items() if k != 'role'})
        return f'{cname}: bind role removed', [rec['gguf_name']]
    if what == 'undeclared':
        i, cname, rec = find(shards, 'shared_gate', layer=moe, index=0)
        edit_record(shards, i, cname, lambda r: dict(r, index=7))
        return f'{cname}: shared_gate index 0 -> 7', ['declares no such tensor']
    if what == 'duplicate':
        i, cname, rec = find(shards, 'attn_norm', layer=0)
        edit_record(shards, i, cname, lambda r: dict(r, role='ffn_norm'))
        return f'{cname}: role attn_norm -> ffn_norm (layer 0 then has two)', ['one bind record']
    if what == 'format':
        i, cname, rec = find(shards, 'attn_k', layer=0)
        nbytes = shards[i]['header'][cname]['data_offsets'][1] - shards[i]['header'][cname]['data_offsets'][0]
        if nbytes % 4:
            sys.exit(f'inkling_family_gate: {cname} is {nbytes} bytes, not a whole number of i32')
        edit_record(shards, i, cname, lambda r: dict(r, layout='i32', dims_ne=[nbytes // 4]))
        shards[i]['header'][cname].update(dtype='I32', shape=[nbytes // 4])
        return f'{cname} relabelled i32 [{nbytes // 4}]', [rec['gguf_name']]
    if what == 'drafter':
        set_kv(shards, 'pulsar.mtp_present', 'bool', True)
        return 'pulsar.mtp_present = true without an mtp config', ['pulsar.mtp_present']
    sys.exit(f'inkling_family_gate: unknown mutation {what}')


MUTATIONS = ('missing', 'shape', 'switch-unknown', 'switch-flipped', 'geometry', 'plan', 'no-role', 'undeclared',
             'duplicate', 'format', 'drafter')


def probe(binary, model, lock, tag):
    returned, rc, log, dt = LC.open_model(binary, model, lock)
    ok = rc == 0 and 'INKLING-FAMILY PROBE PASS' in log
    print(f'  {"ok  " if ok else "FAIL"}  {tag}: the probe {"passes" if ok else "fails"} ({dt:.1f} s)')
    lines = [x for x in log.splitlines() if x.startswith('  ') or 'inkling:' in x]
    for x in (lines if not ok else [x for x in lines if 'tensors bound' in x or 'priced' in x]):
        print('        ' + x.strip()[:200])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--probe', required=True, help='tests/inkling_family_gate')
    ap.add_argument('--open', required=True, help='tests/loader_open')
    ap.add_argument('--work', required=True, help='a scratch directory for the clones (sparse: ~no disk)')
    ap.add_argument('--refuse', action='append', default=[], help='a container the family must refuse whole')
    ap.add_argument('model')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    lock = os.path.join(a.work, 'inspect.lock')
    fails = 0
    src = a.model
    shards = LC.load_container(src)
    if LC.family_of(shards) != 'inkling':
        sys.exit(f'inkling_family_gate: {src} is not an Inkling container')
    fails += not probe(a.probe, src, lock, f'{os.path.basename(os.path.normpath(src))} (the container)')
    clean = os.path.join(a.work, 'inkling-clean')
    LC.write_clone(src, shards, clean)
    fails += not probe(a.probe, clean, lock, 'its header clone')
    for what in MUTATIONS:
        m = LC.load_container(src)
        desc, names = mutate(m, what)
        out = os.path.join(a.work, f'inkling-{what}')
        LC.write_clone(src, m, out)
        returned, rc, log, dt = LC.open_model(a.open, out, lock)
        said = [x for x in log.splitlines() if any(n in x for n in names) and any(w in x for w in LC.REFUSAL_WORDS)]
        if returned and rc == 0:
            print(f'  FAIL  {what}: {desc} -- OPENED (the family must refuse it)')
            fails += 1
        elif not returned:
            print(f'  FAIL  {what}: {desc} -- the loader EXITED the process (rc {rc}) instead of refusing')
            fails += 1
        elif not said:
            print(f'  FAIL  {what}: {desc} -- refused without naming {names[0]!r}; it said:')
            print('\n'.join('        ' + x for x in log.strip().splitlines()[-5:]))
            fails += 1
        else:
            print(f'  ok    {what}: {desc} -- refused by name: {said[0].strip()[:170]}')
        shutil.rmtree(out)
    shutil.rmtree(clean)
    for r in a.refuse:
        returned, rc, log, dt = LC.open_model(a.open, r, lock)
        said = [x for x in log.splitlines() if 'refus' in x or 'refuses' in x]
        ok = returned and rc != 0 and said
        print(f'  {"ok  " if ok else "FAIL"}  refuse {os.path.basename(os.path.normpath(r))}: '
              f'{"refused" if ok else "NOT refused by name"} ({dt:.1f} s): {said[0].strip()[:170] if said else ""}')
        fails += not ok
    print('INKLING-FAMILY GATE ' + ('PASS' if fails == 0 else f'FAIL ({fails})'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

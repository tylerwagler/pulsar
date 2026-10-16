#!/usr/bin/env python3
"""nemotron_family_gate.py -- L296 N1: the Nemotron-H family's load, on a built container.

  nemotron_family_gate.py --probe tests/nemotron_family_gate --open tests/loader_open --work DIR MODEL

1. The REAL container: tests/nemotron_family_gate opens it inspect-only and checks every claim of the load (the plan,
   every tensor bound by its bind record, every byte priced, the slots, the facts, the tokenizer and chat front, the
   session refusals).
2. Its HEADER CLONE (tests/loader_contract.py's: the headers and metadata, the data region a hole): the same probe.
3. Each Nemotron-H MUTANT of the clone must be REFUSED BY NAME, the open returning (tests/loader_open): a missing
   tensor, a wrong shape, an unknown switch, a flipped switch, a geometry the profile does not have, a block-kind list
   the grouping refuses, a block kind no block implements, a block list the tensors contradict, a dt clamp, a tower
   the declaration does not hold, a drafter the config does not describe, a tensor with no bind record, an undeclared
   tensor, two tensors with one record, a linear in a format no arm reads.  (The family-neutral faults are
   loader_contract.py's `nemotron_h` row.)

The targets are the container's own names, chosen from the clone's records by bind role -- the only Nemotron-H
knowledge here."""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader_contract as LC  # noqa: E402  (the header clone, the open probe)
import inkling_family_gate as IG  # noqa: E402  (the record / kv editors)


def mutate(shards, what):
    """apply one fault; returns (description, [strings the refusal must name])"""
    kv = IG.kv_of(shards)
    blocks = kv['nemotron_h.layers_block_type']['value']['v']
    sa = lambda v: {'__array__': 'string', 'n': len(v), 'v': v}   # noqa: E731
    if what == 'missing':
        i, cname, rec = IG.find(shards, 'ssm_d')
        del shards[i]['header'][cname]
        LC.edit_meta_json(shards[i], 'pulsar.tensors', lambda t: {k: v for k, v in t.items() if k != cname})
        return f'{cname} dropped', ['ssm_d']
    if what == 'shape':
        i, cname, rec = IG.find(shards, 'ssm_conv', layer=0)
        IG.edit_record(shards, i, cname, lambda r: dict(r, dims_ne=list(reversed(r['dims_ne']))))
        shards[i]['header'][cname]['shape'] = list(reversed(shards[i]['header'][cname]['shape']))
        return f'{cname} transposed', [rec['gguf_name']]
    if what == 'switch-unknown':
        IG.set_kv(shards, 'nemotron_h.use_rope', 'bool', True)
        return 'nemotron_h.use_rope = true added', ['nemotron_h.use_rope']
    if what == 'switch-flipped':
        IG.set_kv(shards, 'nemotron_h.mlp_hidden_act', 'string', 'silu')
        return 'nemotron_h.mlp_hidden_act = silu', ['nemotron_h.mlp_hidden_act']
    if what == 'geometry':
        IG.set_kv(shards, 'nemotron_h.ssm_state_size', 'u32', 64)
        return 'nemotron_h.ssm_state_size = 64', ['nemotron_h.ssm_state_size']
    if what == 'plan':
        IG.set_kv(shards, 'nemotron_h.layers_block_type', 'array', sa(['moe'] + blocks[1:]))
        return 'layers_block_type[0] = moe (a moe with no mixer before it)', ['nemotron_h.layers_block_type']
    if what == 'plan-kind':
        IG.set_kv(shards, 'nemotron_h.layers_block_type', 'array', sa(blocks[:-1] + ['mlp']))
        return 'layers_block_type[-1] = mlp', ['nemotron_h.layers_block_type']
    if what == 'plan-tensors':
        IG.set_kv(shards, 'nemotron_h.layers_block_type', 'array', sa(blocks + ['mamba']))
        return 'layers_block_type gains a mamba block the container has no tensors for', ['ssm_in']
    if what == 'dt-clamp':
        IG.set_kv(shards, 'nemotron_h.time_step_limit.min', 'f32', 0.001)
        return 'nemotron_h.time_step_limit.min = 0.001', ['nemotron_h.time_step_limit.min']
    if what == 'vision':
        IG.set_kv(shards, 'pulsar.vision_present', 'bool', True)
        return 'pulsar.vision_present = true', ['pulsar.vision_present']
    if what == 'drafter':
        IG.set_kv(shards, 'nemotron_h.num_nextn_predict_layers', 'u32', 0)
        return 'num_nextn_predict_layers = 0 with the drafter written', ['pulsar.mtp_present']
    if what == 'no-role':
        i, cname, rec = IG.find(shards, 'output_norm')
        IG.edit_record(shards, i, cname, lambda r: {k: v for k, v in r.items() if k != 'role'})
        return f'{cname}: bind role removed', [rec['gguf_name']]
    if what == 'undeclared':
        i, cname, rec = IG.find(shards, 'shared_up', index=0)
        IG.edit_record(shards, i, cname, lambda r: dict(r, index=7))
        return f'{cname}: shared_up index 0 -> 7', ['declares no such tensor']
    if what == 'duplicate':
        i, cname, rec = IG.find(shards, 'attn_norm')
        IG.edit_record(shards, i, cname, lambda r: dict(r, role='ffn_norm'))
        return f'{cname}: role attn_norm -> ffn_norm (its layer then has two)', ['one bind record']
    if what == 'format':
        i, cname, rec = IG.find(shards, 'attn_k')
        nbytes = shards[i]['header'][cname]['data_offsets'][1] - shards[i]['header'][cname]['data_offsets'][0]
        IG.edit_record(shards, i, cname, lambda r: dict(r, layout='i32', dims_ne=[nbytes // 4]))
        shards[i]['header'][cname].update(dtype='I32', shape=[nbytes // 4])
        return f'{cname} relabelled i32 [{nbytes // 4}]', [rec['gguf_name']]
    sys.exit(f'nemotron_family_gate: unknown mutation {what}')


MUTATIONS = ('missing', 'shape', 'switch-unknown', 'switch-flipped', 'geometry', 'plan', 'plan-kind', 'plan-tensors',
             'dt-clamp', 'vision', 'drafter', 'no-role', 'undeclared', 'duplicate', 'format')


def probe(binary, model, lock, tag):
    returned, rc, log, dt = LC.open_model(binary, model, lock)
    ok = rc == 0 and 'NEMOTRON-FAMILY PROBE PASS' in log
    print(f'  {"ok  " if ok else "FAIL"}  {tag}: the probe {"passes" if ok else "fails"} ({dt:.1f} s)')
    lines = [x for x in log.splitlines() if x.startswith('  ') or 'nemotron_h:' in x]
    for x in (lines if not ok else [x for x in lines if 'tensors bound' in x or 'priced' in x or 'plan:' in x]):
        print('        ' + x.strip()[:200])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--probe', required=True, help='tests/nemotron_family_gate')
    ap.add_argument('--open', required=True, help='tests/loader_open')
    ap.add_argument('--work', required=True, help='a scratch directory for the clones (sparse: ~no disk)')
    ap.add_argument('model')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    lock = os.path.join(a.work, 'inspect.lock')
    fails = 0
    src = a.model
    shards = LC.load_container(src)
    if LC.family_of(shards) != 'nemotron_h':
        sys.exit(f'nemotron_family_gate: {src} is not a Nemotron-H container')
    fails += not probe(a.probe, src, lock, f'{os.path.basename(os.path.normpath(src))} (the container)')
    clean = os.path.join(a.work, 'nemotron-clean')
    LC.write_clone(src, shards, clean)
    fails += not probe(a.probe, clean, lock, 'its header clone')
    for what in MUTATIONS:
        m = LC.load_container(src)
        desc, names = mutate(m, what)
        out = os.path.join(a.work, f'nemotron-{what}')
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
    print('NEMOTRON-FAMILY GATE ' + ('PASS' if fails == 0 else f'FAIL ({fails})'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

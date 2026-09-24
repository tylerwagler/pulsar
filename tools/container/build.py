#!/usr/bin/env python3
"""The direct container builder: HF checkpoint(s) -> pulsar's safetensors container.

    plan    --hf DIR [--exl3 DIR [--exl3-layers 5,18-22]] [--format-map JSON]
    emit    ... --out DIR (--shard S | --all) [--mxfp8-scale rederive|verbatim]
    verify  ... --out DIR (--shard S | --all)        # against the HF SOURCE
    audit   --out DIR                                # structure + index closed both ways

No GGUF anywhere (L247).  The engine reads three metadata keys -- `pulsar.kv`,
`pulsar.tensors`, `pulsar.experts` -- and binds tensors by the `gguf_name`
they carry; the container's tensor names ARE the HF names.  Every payload is
written in the layout its kernel reads, produced by `producers.py` (pure
bytes -> bytes) or copied verbatim; nothing is converted at load.

The alignment contract (from the lane it replaces): safetensors requires a
gap-free data buffer, so alignment comes from ORDER -- entries are placed by
descending alignment requirement (32 / 4 / 2 / 1 from their own byte count)
and a projection's per-expert run stays adjacent, back to back, so the kernels'
`base + xid * expert_bytes` holds.  A shard is streamed: the header is fixed
first from the byte model, then each entry's bytes are produced or copied
straight into the file.
"""
import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_source import HFCheckpoint, Exl3Checkpoint, EXL3_LAYOUT, exl3_expert_bytes  # noqa: E402
import names as N          # noqa: E402
import policy as P         # noqa: E402
import producers as PR     # noqa: E402
import kv as KV            # noqa: E402

ALIGN = 32
NATIVE_DTYPES = {'bf16': 'BF16', 'f32': 'F32', 'i32': 'I32'}
PARTS = ('w1', 'w3', 'w2')


def align_for(nbytes):
    for a in (32, 4, 2):
        if nbytes % a == 0:
            return a
    return 1


def shard_order(shape):
    """names.py owns the plan (vision first and primary, one per layer, top, one per drafter layer)."""
    order = N.shard_order(shape)
    return order, {s: N.shard_file(shape, s) for s in order}


def parse_layers(spec):
    out = set()
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def model_shape(hf):
    """From config.json alone (names.py's rule): the checkpoint's word, not a guess from its names."""
    return N.ModelShape.from_config(hf.config['top_level'])


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def plan(hf, exl3, exl3_layers, overrides, mxfp8_mode, tokenizer_dir, reap_map):
    shape = model_shape(hf)
    order, files = shard_order(shape)
    shards = {s: {'entries': [], 'tensors': {}, 'experts': []} for s in order}
    hf_names = hf.names()
    experts = {}        # (shard, layer, part) -> {e: (weight_name, scale_name)}
    consumed = 0        # names the source carries and the engine never binds (names.py: emit=False)
    for name in hf_names:
        m = N.map_hf(name, shape)
        if m is None:
            raise SystemExit(f'{name}: not a tensor this builder maps -- refusing (names.py)')
        if not m.emit:
            consumed += 1
            continue
        if m.family == 'expert':
            key = (m.shard, m.layer, m.part)
            slot = experts.setdefault(key, {}).setdefault(m.expert, {})
            slot['scale' if m.is_scale else 'weight'] = name
            slot['mapped'] = m
            continue
        if m.is_scale:
            continue                      # folded into its weight's producer
        if m.shard not in shards:
            raise SystemExit(f'{name}: shard {m.shard} outside the plan ({len(order)} shards)')
        dtype = hf.dtype(name)
        hshape = hf.shape(name)                        # the SOURCE shape (producers take it)
        dshape = P.declared_shape(m, hshape)           # what the container holds (policy decision 5)
        layout = P.layout_for(m, dtype, hshape, overrides)
        dims_ne = list(reversed(dshape))
        entry = {'name': m.container_name, 'layout': layout, 'gguf_name': m.gguf_name}
        if layout in NATIVE_DTYPES and dtype == NATIVE_DTYPES[layout]:
            path, off, n = hf.span(name)
            entry.update(dtype=NATIVE_DTYPES[layout], shape=list(dshape), nbytes=n,
                         src=('ranges', [(path, off, n)]))
        elif layout == 'i32' and dtype == 'I64':
            # the ONE narrowing: the routing table ffn.gate.tid2eid is I64 in the
            # checkpoint and I32 in the engine (tensor_expect_layout); values must fit
            n_el = 1
            for d in hshape:
                n_el *= d
            entry.update(dtype='I32', shape=list(dshape), nbytes=4 * n_el,
                         src=('produce', (lambda w=name: PR.i64_to_i32(hf.raw(w)))))
        elif layout == 'mxfp8_lt':
            scale = name[:-len('.weight')] + '.scale'
            if not hf.has(scale):
                raise SystemExit(f'{name}: mxfp8_lt needs {scale}')
            out, inp = hshape
            srows = hf.shape(scale)[0]
            block = out // srows
            nbytes = PR.bytes_for(layout, dims_ne)
            entry.update(dtype='U8', shape=[nbytes], nbytes=nbytes,
                         src=('produce', (lambda w=name, s=scale, o=out, i=inp, b=block:
                                          PR.mxfp8_lt(hf.raw(w), hf.raw(s), o, i, b, mxfp8_mode))))
        elif layout == 'fp8_e4m3_soa_k':
            rows, cols = hshape
            nbytes = PR.bytes_for(layout, dims_ne)
            entry.update(dtype='U8', shape=[nbytes], nbytes=nbytes,
                         src=('produce', (lambda w=name, r=rows, c=cols:
                                          PR.fp8_e4m3_soa_k_from_bf16(hf.raw(w), r, c))))
        else:
            raise SystemExit(f'{name}: layout {layout} has no dense producer')
        shards[m.shard]['entries'].append(entry)
        shards[m.shard]['tensors'][m.container_name] = {'layout': layout, 'dims_ne': dims_ne,
                                                        'gguf_name': m.gguf_name}

    # the routed expert families: one U8 per expert-projection, contiguous
    for (shard, layer, part), slots in sorted(experts.items()):
        n_exp = max(slots) + 1
        if sorted(slots) != list(range(n_exp)):
            raise SystemExit(f'{shard} {part}: experts not contiguous 0..{n_exp - 1}')
        m0 = slots[0]['mapped']
        wname0 = slots[0]['weight']
        hshape = hf.shape(wname0)                      # I8 [out, in/2] for the FP4 source
        dtype = hf.dtype(wname0)
        out = hshape[0]
        inp = hshape[1] * 2 if dtype == 'I8' else hshape[1]
        dims_ne = [inp, out]
        use_exl3 = exl3 is not None and shard.startswith('layers.') and layer in exl3_layers
        if use_exl3:
            words = None
            per = []
            for e in range(n_exp):
                ranges, w = exl3.expert(layer, e, part, inp, out)
                if words is None:
                    words = w
                elif w != words:
                    raise SystemExit(f'layers.{layer}.ffn.experts.{e}.{part}: {w} words per tile, expert 0 has {words}')
                per.append(ranges)
            layout = EXL3_LAYOUT[words]
            trellis, scales = exl3_expert_bytes(inp, out, words)
            eb = trellis + scales
        else:
            layout = P.layout_for(m0, dtype, hshape, overrides)
            if layout != 'cutlass_mxfp4':
                raise SystemExit(f'{shard} {part}: routed layout {layout} has no producer here')
            eb = PR.bytes_for(layout, dims_ne)
        if PR.bytes_for(layout, dims_ne) != eb:
            raise SystemExit(f'{shard} {part}: byte model disagrees ({PR.bytes_for(layout, dims_ne)} vs {eb})')
        gguf_name = m0.gguf_name
        shards[shard]['experts'].append({'gguf_name': gguf_name, 'part': part, 'n_experts': n_exp,
                                         'expert_bytes': eb, 'layout': layout, 'contiguous': True,
                                         'dims_per_expert_ne': dims_ne})
        for e in range(n_exp):
            m = slots[e]['mapped']
            entry = {'name': m.container_name, 'layout': layout, 'gguf_name': gguf_name,
                     'dtype': 'U8', 'shape': [eb], 'nbytes': eb}
            if use_exl3:
                entry['src'] = ('ranges', per[e])
            else:
                wn, sn = slots[e]['weight'], slots[e].get('scale')
                if not sn:
                    raise SystemExit(f'{wn}: cutlass_mxfp4 needs its .scale')
                entry['src'] = ('produce', (lambda w=wn, s=sn, o=out, i=inp: PR.cutlass_mxfp4(hf.raw(w), hf.raw(s), o, i)))
            shards[shard]['entries'].append(entry)

    kvs = KV.build_kv(hf, tokenizer_dir, reap_map)
    kv_arch = [k for k in kvs if not k['key'].startswith('tokenizer.')]
    for s in order:
        p = shards[s]
        exp_layouts = {e['layout'] for e in p['experts']}
        gu = {}
        for e in p['experts']:
            if e['part'] in ('w1', 'w3'):
                gu.setdefault(e['gguf_name'].split('.')[1], set()).add(e['layout'])
        for lay, ls in gu.items():
            if len(ls) > 1:
                raise SystemExit(f'{s}: layer {lay} gate/up layouts differ: {sorted(ls)}')
        p['meta'] = {
            'format': 'pt',
            'pulsar.format': 'pulsar-safetensors-v1',
            'pulsar.alignment': str(ALIGN),
            'pulsar.shard': files[s],
            'pulsar.shard_key': s,
            'pulsar.n_shards': str(len(order)),
            'pulsar.primary': files['vision'],
            'pulsar.tensors': json.dumps(p['tensors'], separators=(',', ':'), sort_keys=True),
            'pulsar.experts': json.dumps(p['experts'], separators=(',', ':')),
            'pulsar.kv_arch': json.dumps(kv_arch, separators=(',', ':')),
            'pulsar.expert_dtype': ('none' if not exp_layouts else
                                    'mxfp4_cutlass' if exp_layouts <= {'cutlass_mxfp4'} else
                                    'exl3' if all(l.startswith('exl3m_') for l in exp_layouts) else
                                    'mixed'),
        }
        if s == 'vision':
            p['meta']['pulsar.kv'] = json.dumps(kvs, separators=(',', ':'))
    return shape, order, files, shards, consumed


# ---------------------------------------------------------------------------
# the writer: streamed
# ---------------------------------------------------------------------------
def write_shard(path, entries, meta):
    sized = sorted(entries, key=lambda e: -align_for(e['nbytes']))   # stable: expert runs stay adjacent
    header, cursor = {}, 0
    for e in sized:
        a = align_for(e['nbytes'])
        if cursor % a:
            raise SystemExit(f'{e["name"]}: offset {cursor} violates {a}-byte alignment')
        header[e['name']] = {'dtype': e['dtype'], 'shape': e['shape'], 'data_offsets': [cursor, cursor + e['nbytes']]}
        cursor += e['nbytes']
    hj = json.dumps({**header, '__metadata__': meta}, separators=(',', ':'), sort_keys=True).encode()
    hj += b' ' * (-(8 + len(hj)) % ALIGN)
    handles = {}
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        for e in sized:
            kind, src = e['src']
            if kind == 'ranges':
                n_written = 0
                for sp, off, n in src:
                    fh = handles.get(sp)
                    if fh is None:
                        fh = handles[sp] = open(sp, 'rb')
                    fh.seek(off)
                    left = n
                    while left:
                        b = fh.read(min(left, 64 << 20))
                        if not b:
                            raise SystemExit(f'{e["name"]}: short read from {sp}')
                        f.write(b)
                        left -= len(b)
                    n_written += n
                if n_written != e['nbytes']:
                    raise SystemExit(f'{e["name"]}: ranges give {n_written} bytes, the model says {e["nbytes"]}')
            else:
                b = src()
                if len(b) != e['nbytes']:
                    raise SystemExit(f'{e["name"]}: producer gave {len(b)} bytes, the model says {e["nbytes"]}')
                f.write(b)
    for fh in handles.values():
        fh.close()
    return header, cursor


def read_shard(path):
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        hdr = json.loads(f.read(n))
        meta = hdr.pop('__metadata__', {})
        return hdr, meta, 8 + n, os.path.getsize(path)


def write_index(out_dir):
    wm, total = {}, 0
    for f in sorted(os.listdir(out_dir)):
        if not (f.startswith('model-') and f.endswith('.safetensors')):
            continue
        hdr, _meta, _o, size = read_shard(os.path.join(out_dir, f))
        for k in hdr:
            wm[k] = f
        total += size
    json.dump({'metadata': {'total_size': total}, 'weight_map': dict(sorted(wm.items()))},
              open(os.path.join(out_dir, 'model.safetensors.index.json'), 'w'), indent=0, sort_keys=True)
    return len(wm), total


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def load_sources(args):
    hf = HFCheckpoint(args.hf)
    exl3 = Exl3Checkpoint(args.exl3) if args.exl3 else None
    layers = set()
    if exl3:
        have = set(exl3.layers())
        layers = parse_layers(args.exl3_layers) if args.exl3_layers else have
        missing = sorted(layers - have)
        if missing:
            raise SystemExit(f'{args.exl3}: layers {missing} requested but not present (have {sorted(have)})')
    overrides = P.rekey_format_map(json.load(open(args.format_map)), model_shape(hf)) if args.format_map else {}
    return hf, exl3, layers, overrides


def cmd_plan(args):
    hf, exl3, layers, overrides = load_sources(args)
    shape, order, files, shards, consumed = plan(hf, exl3, layers, overrides, args.mxfp8_scale, args.tokenizer or args.hf, args.reap_map)
    print(f'model: {shape}; shards {len(order)}; source tensors consumed but not written: {consumed}')
    tot = 0
    for s in order:
        p = shards[s]
        b = sum(e['nbytes'] for e in p['entries'])
        tot += b
        lay = {}
        for e in p['entries']:
            lay[e['layout']] = lay.get(e['layout'], 0) + 1
        print(f'  {files[s]}  {s:10s} {len(p["entries"]):6d} tensors  {len(p["experts"]):2d} families  {b / 1e9:8.3f} GB  {dict(sorted(lay.items()))}')
    print(f'total {tot / 1e9:.2f} GB')
    if exl3:
        print(f'exl3 experts from {exl3.dir}: layers {sorted(layers)}')
    return 0


def cmd_emit(args):
    hf, exl3, layers, overrides = load_sources(args)
    shape, order, files, shards, consumed = plan(hf, exl3, layers, overrides, args.mxfp8_scale, args.tokenizer or args.hf, args.reap_map)
    todo = order if args.all else [args.shard]
    os.makedirs(args.out, exist_ok=True)
    import time
    for s in todo:
        if s not in shards:
            raise SystemExit(f'unknown shard {s}; one of {order}')
        t0 = time.time()
        header, total = write_shard(os.path.join(args.out, files[s]), shards[s]['entries'], shards[s]['meta'])
        print(f'{files[s]}  {s:10s} {len(header):6d} tensors {total / 1e9:8.3f} GB  {time.time() - t0:6.1f} s', flush=True)
    if args.all:
        n, total = write_index(args.out)
        print(f'index: {n} tensors, total_size {total / 1e9:.2f} GB')
    return 0


def cmd_verify(args):
    """Against the HF SOURCE, not an intermediate: structure and alignment, every
    native span byte-equal, every produced payload re-produced and compared,
    every EXL3 slice equal to its source ranges, the declared byte model equal
    to the span, expert families contiguous."""
    hf, exl3, layers, overrides = load_sources(args)
    shape, order, files, shards, consumed = plan(hf, exl3, layers, overrides, args.mxfp8_scale, args.tokenizer or args.hf, args.reap_map)
    todo = order if args.all else [args.shard]
    failing = 0
    for s in todo:
        path = os.path.join(args.out, files[s])
        hdr, meta, buf_off, size = read_shard(path)
        end = max((h['data_offsets'][1] for h in hdr.values()), default=0)
        offs = sorted((h['data_offsets'][0], h['data_offsets'][1]) for h in hdr.values())
        gaps = sum(1 for a, b in zip(offs, offs[1:]) if a[1] != b[0])
        structure = (buf_off % ALIGN == 0) and (size == buf_off + end) and (not offs or offs[0][0] == 0) and gaps == 0
        want = {e['name']: e for e in shards[s]['entries']}
        n_ok = n_bad = unexpected = missing = misaligned = 0
        decl = json.loads(meta.get('pulsar.tensors', '{}'))
        experts = json.loads(meta.get('pulsar.experts', '[]'))
        with open(path, 'rb') as f:
            for name, h in hdr.items():
                e = want.get(name)
                if e is None:
                    unexpected += 1
                    print(f'  UNEXPECTED {name}')
                    continue
                o0, o1 = h['data_offsets']
                if (o1 - o0) != e['nbytes']:
                    n_bad += 1
                    print(f'  SPAN {name}: {o1 - o0} vs model {e["nbytes"]}')
                    continue
                if e['layout'] not in NATIVE_DTYPES and o0 % ALIGN:
                    misaligned += 1
                f.seek(buf_off + o0)
                got = f.read(o1 - o0)
                kind, src = e['src']
                if kind == 'ranges':
                    exp = b''
                    for sp, off, n in src:
                        with open(sp, 'rb') as g:
                            g.seek(off)
                            exp += g.read(n)
                else:
                    exp = src()
                if got == exp:
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f'  BYTES DIFFER {name} ({e["layout"]})')
        for name in want:
            if name not in hdr:
                missing += 1
                print(f'  MISSING {name}')
        # declarations: byte model + contiguity
        decl_bad = 0
        for name, d in decl.items():
            h = hdr.get(name)
            if not h or PR.bytes_for(d['layout'], d['dims_ne']) != h['data_offsets'][1] - h['data_offsets'][0]:
                decl_bad += 1
                print(f'  DECL {name}: byte model vs span')
        contig_bad = 0
        for fam in experts:
            gname = fam['gguf_name']
            ns, lay = ('layers', gname.split('.')[1]) if gname.startswith('blk.') else ('mtp', gname.split('.')[1])
            first = None
            for e in range(fam['n_experts']):
                h = hdr.get(f'{ns}.{lay}.ffn.experts.{e}.{fam["part"]}.weight')
                if not h:
                    contig_bad += 1
                    break
                o0 = h['data_offsets'][0]
                if e == 0:
                    first = o0
                elif o0 != first + e * fam['expert_bytes']:
                    contig_bad += 1
                    break
            if PR.bytes_for(fam['layout'], fam['dims_per_expert_ne']) != fam['expert_bytes']:
                contig_bad += 1
        ok = structure and not misaligned and n_bad == 0 and unexpected == 0 and missing == 0 and decl_bad == 0 and contig_bad == 0
        failing += 0 if ok else 1
        print(f'{files[s]}  {s:10s} structure={structure} misaligned={misaligned} tensors {n_ok}/{n_ok + n_bad} '
              f'unexpected={unexpected} missing={missing} decl_bad={decl_bad} contiguity_bad={contig_bad} -> {"PASS" if ok else "FAIL"}', flush=True)
    print(f'shards checked: {len(todo)}  failing: {failing}')
    return 1 if failing else 0


def cmd_audit(args):
    """Directory-level: structure, the three metadata keys present, the index closed both ways."""
    shards = sorted(f for f in os.listdir(args.out) if f.startswith('model-') and f.endswith('.safetensors'))
    n_idx, total = write_index(args.out)
    idx = json.load(open(os.path.join(args.out, 'model.safetensors.index.json')))
    bad, nod, declared, kv_shards = [], [], {}, 0
    for f in shards:
        hdr, md, buf_off, size = read_shard(os.path.join(args.out, f))
        end = max((h['data_offsets'][1] for h in hdr.values()), default=0)
        if buf_off % ALIGN or size != buf_off + end:
            bad.append(f)
        for k in ('pulsar.format', 'pulsar.tensors', 'pulsar.experts'):
            if k not in md:
                nod.append((f, k))
        if 'pulsar.kv' in md:
            kv_shards += 1
        names_ = set(hdr) | set(json.loads(md.get('pulsar.tensors', '{}')))
        for e in json.loads(md.get('pulsar.experts', '[]')):
            ns = 'layers' if e['gguf_name'].startswith('blk.') else 'mtp'
            lay = e['gguf_name'].split('.')[1]
            names_ |= {f'{ns}.{lay}.ffn.experts.{x}.{e["part"]}.weight' for x in range(e['n_experts'])}
        declared[f] = names_
    mismatch = sum(1 for n, s in idx['weight_map'].items() if n not in declared.get(s, set()))
    unindexed = sum(1 for s, ns in declared.items() for n in ns if idx['weight_map'].get(n) != s)
    print(f'shards: {len(shards)}  total_size: {total / 1e9:.2f} GB  index entries: {n_idx}  pulsar.kv on {kv_shards} shard(s)')
    print(f'structurally bad: {len(bad)} {bad[:3]}')
    print(f'metadata problems: {len(nod)} {nod[:3]}')
    print(f'indexed but not declared by its shard: {mismatch}')
    print(f'declared with no/wrong index entry: {unindexed}')
    ok = not bad and not nod and kv_shards == 1 and mismatch == 0 and unindexed == 0
    print('AUDIT', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('plan', cmd_plan), ('emit', cmd_emit), ('verify', cmd_verify), ('audit', cmd_audit)):
        p = sub.add_parser(name)
        if name != 'audit':
            p.add_argument('--hf', required=True, help='the HF checkpoint directory (config.json + shards)')
            p.add_argument('--exl3', metavar='DIR', help='source the routed experts of --exl3-layers from this EXL3 checkpoint')
            p.add_argument('--exl3-layers', metavar='SPEC', help='e.g. 5,18-22 (default: every layer the EXL3 checkpoint holds)')
            p.add_argument('--format-map', metavar='JSON', help='per-tensor layout overrides (tools/container/format-maps)')
            p.add_argument('--mxfp8-scale', choices=('rederive', 'verbatim'), default='rederive',
                           help='FP8 dense -> mxfp8_lt: re-derive the per-32 E8M0 (byte-identical to the archived codec) or broadcast the source block scale')
            p.add_argument('--tokenizer', metavar='DIR', help='tokenizer files (default: the HF dir)')
            p.add_argument('--reap-map', metavar='JSON')
        if name != 'plan':
            p.add_argument('--out', required=True)
        if name in ('emit', 'verify'):
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument('--shard')
            g.add_argument('--all', action='store_true')
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())

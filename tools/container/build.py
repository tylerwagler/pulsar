#!/usr/bin/env python3
"""The direct container builder: HF checkpoint(s) -> pulsar's safetensors container.

    recipe  --hf DIR [--exl3 DIR] [--recipe JSON]    # print the pulsar.recipe.v1 the build runs (DeepSeek's default
                                                     #  is generated from policy.py + the EXL3 source)
    plan    --hf DIR [--exl3 DIR] [--exl3-experts DIR] [--tessera DIR] [--recipe JSON] [--dump FILE]
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
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_source import HFCheckpoint, Exl3Checkpoint, TesseraBundle  # noqa: E402
import exl3_rates          # noqa: E402
import producers as PR     # noqa: E402
import recipe as R         # noqa: E402
import qwen as Q           # noqa: E402
import deepseek as D       # noqa: E402
import entries as EN       # noqa: E402

ALIGN = 32
NATIVE_DTYPES = {'bf16': 'BF16', 'f32': 'F32', 'i32': 'I32'}


def align_for(nbytes):
    for a in (32, 4, 2):
        if nbytes % a == 0:
            return a
    return 1


# ---------------------------------------------------------------------------
# the plan: ONE walk for every family; the family module supplies only what is its own (deepseek.py, qwen.py)
# ---------------------------------------------------------------------------
def shard_file(order, shard):
    return f'model-{order.index(shard) + 1:05d}-of-{len(order):05d}.safetensors'


def expert_dtype(layouts):
    """pulsar.expert_dtype: the one vocabulary for a shard's routed families."""
    if not layouts:
        return 'none'
    if layouts <= {'cutlass_mxfp4'}:
        return 'mxfp4_cutlass'
    return 'exl3' if all(x.startswith('exl3m_') for x in layouts) else 'mixed'


def add_families(shard, families):
    """Routed families -> one U8 entry per expert-projection, contiguous in expert order, plus the pulsar.experts
    record (the engine's stride is expert_bytes)."""
    for f in families:
        if f['layout'] in exl3_rates.K2:
            exl3_rates.admit(f['role'], f['layout'], f['gguf_name'])
        eb = PR.bytes_for(f['layout'], [f['inp'], f['n']])
        shard['experts'].append({'gguf_name': f['gguf_name'], 'part': f['part'], 'n_experts': len(f['srcs']),
                                 'expert_bytes': eb, 'layout': f['layout'], 'contiguous': True,
                                 'dims_per_expert_ne': [f['inp'], f['n']], **f['extras']})
        for name, src in zip(f['entry_names'], f['srcs']):
            shard['entries'].append({'name': name, 'layout': f['layout'], 'gguf_name': f['gguf_name'],
                                     'dtype': 'U8', 'shape': [eb], 'nbytes': eb, 'src': src})


def dense_entry(hf, fam, m, name, fmt, ctx):
    """One dense entry: an EXL3 Linear's ranges, or the producer table's payload for (source, layout)."""
    layout, src = fmt
    hshape = hf.shape(name)                        # the SOURCE shape (producers take it)
    dshape = fam.declared_shape(m, hshape)         # what the container holds (the family's quirks)
    dims_ne = list(reversed(dshape))
    if layout == 'native':
        if hf.dtype(name) not in PR.NATIVE_LAYOUT:
            raise SystemExit(f'{name}: native format, but the engine has no layout for a {hf.dtype(name)} source')
        layout = PR.NATIVE_LAYOUT[hf.dtype(name)]
    entry = {'name': m.container_name, 'layout': layout, 'gguf_name': m.gguf_name}
    if layout in exl3_rates.K2:
        exl3_rates.admit(m.role, layout, name)
        if hf.dtype(name) != 'BF16' or len(hshape) != 2 or not name.endswith('.weight'):
            raise SystemExit(f'{name}: EXL3 dense Linear from a BF16 [out, in] .weight, got {hf.dtype(name)} {hshape}')
        out, inp = hshape
        ranges, nbytes = EN.exl3_ranges(ctx.sources[src], name[:-len('.weight')], layout, inp, out)
        entry.update(dtype='U8', shape=[nbytes], nbytes=nbytes, src=('ranges', ranges))
    elif layout == 'tessera':
        PR.produce(PR.spec(PR.producer_for('tessera', layout, name), []), ctx.sources[src])
    else:
        entry.update(EN.dense(hf, name, layout, dshape, ctx.mxfp8_mode))
    return entry, dims_ne


def resolve(hf, fam, mapped, ctx):
    """name -> (format, source name) for every emitted tensor: the recipe's row, admitted by the family's loader,
    its source defaulted (hf; exl3 for the EXL3 formats -- a routed expert's from --exl3-experts when given; tessera
    for tessera) and present.  Every source the build was given must be read by some row."""
    recipe = ctx.recipe or fam.default_recipe(hf, mapped, ctx)
    ctx.recipe = recipe
    if recipe.model_type != hf.config['top_level']['model_type']:
        raise SystemExit(f'{recipe.path}: a recipe for {recipe.model_type!r}; the checkpoint is '
                         f'{hf.config["top_level"]["model_type"]!r}')
    out, used = {}, {'hf'}
    for name, (fmt, src) in recipe.resolve(mapped).items():
        m = mapped[name]
        fam.admit(hf, m, fmt, name)
        if fmt in R.CONSUMED:
            out[name] = (fmt, None)
            continue
        if src is None:
            src = ('tessera' if fmt == 'tessera' else
                   'exl3_experts' if fmt in exl3_rates.K2 and m.role.startswith('expert_') and ctx.sources.get('exl3_experts')
                   else 'exl3' if fmt in exl3_rates.K2 else 'hf')
        if ctx.sources.get(src) is None:
            raise SystemExit(f'{name}: the recipe reads {fmt} from source {src}; pass --{src.replace("_", "-")}')
        if (src in ('exl3', 'exl3_experts')) != (fmt in exl3_rates.K2) or (src == 'tessera') != (fmt == 'tessera'):
            raise SystemExit(f'{name}: format {fmt} is not read from source {src}')
        used.add(src)
        out[name] = (fmt, src)
    idle = sorted(k for k, v in ctx.sources.items() if v is not None and k not in used)
    if idle:
        raise SystemExit(f'--{idle[0].replace("_", "-")}: given, but no row of {recipe.path} reads it')
    return out


def plan(hf, fam, ctx):
    shape = ctx.shape = fam.shape(hf, ctx)
    order = fam.shard_order(shape)
    files = {s: shard_file(order, s) for s in order}
    shards = {s: {'entries': [], 'tensors': {}, 'experts': []} for s in order}
    hf_names = hf.names()
    mapped = {}
    for name in hf_names:
        m = fam.map_hf(name, shape)
        if m is None:
            raise SystemExit(f'{name}: not a tensor the {fam.__name__} naming table maps -- refusing')
        mapped[name] = m
    fmt = resolve(hf, fam, mapped, ctx)
    groups = {}         # per-expert tensors, grouped by the family (fam.group_key) and planned after the walk
    consumed = 0        # names the source carries and the container does not (drop rules, consumed formats)
    for name in hf_names:
        m = mapped[name]
        if not m.emit:
            consumed += 1
            continue
        if m.is_scale:
            continue                      # folded into its weight's producer
        if fmt[name][0] in R.CONSUMED:
            consumed += 1
            continue
        if m.shard not in shards:
            raise SystemExit(f'{name}: shard {m.shard} outside the plan ({len(order)} shards)')
        if m.role.startswith('expert_'):
            if fam.EXPERT_STACKS:         # the HF tensor is the whole stack: its families, in walk order
                add_families(shards[m.shard], fam.stack_families(hf, m, fmt[name], ctx))
            else:
                groups.setdefault(fam.group_key(m), {})[m.expert] = (name, m)
            continue
        entry, dims_ne = dense_entry(hf, fam, m, name, fmt[name], ctx)
        shards[m.shard]['entries'].append(entry)
        shards[m.shard]['tensors'][m.container_name] = {'layout': entry['layout'], 'dims_ne': dims_ne,
                                                        'gguf_name': m.gguf_name}
    for key, slots in sorted(groups.items()):
        add_families(shards[key[0]], fam.group_families(hf, key, slots, fmt, ctx))

    if ctx.mxfp8_mode != 'rederive' and not any(
            e['src'][0] == 'produce' and e['src'][1]['producer'] == 'mxfp8_lt' for p in shards.values() for e in p['entries']):
        raise SystemExit(f'--mxfp8-scale {ctx.mxfp8_mode}: no entry of this build is an FP8-sourced mxfp8_lt')
    kvs = fam.build_kv(hf, ctx)
    kv_arch = [k for k in kvs if not k['key'].startswith('tokenizer.')]
    for s in order:
        p = shards[s]
        gu = {}
        for e in p['experts']:
            if e['part'] in fam.GATE_UP_PARTS:
                gu.setdefault(e['gguf_name'].rsplit('.', 1)[0] if fam.EXPERT_STACKS else e['gguf_name'].split('.')[1],
                              set()).add(e['layout'])
        for lay, ls in gu.items():
            if len(ls) > 1:
                raise SystemExit(f'{s}: layer {lay} gate/up layouts differ: {sorted(ls)}')
        p['meta'] = {
            'format': 'pt',
            'pulsar.format': 'pulsar-safetensors-v1',
            **fam.META,
            'pulsar.alignment': str(ALIGN),
            'pulsar.shard': files[s],
            'pulsar.shard_key': s,
            'pulsar.n_shards': str(len(order)),
            'pulsar.primary': files['vision'],
            'pulsar.tensors': json.dumps(p['tensors'], separators=(',', ':'), sort_keys=True),
            'pulsar.experts': json.dumps(p['experts'], separators=(',', ':')),
            'pulsar.kv_arch': json.dumps(kv_arch, separators=(',', ':')),
            'pulsar.expert_dtype': expert_dtype({e['layout'] for e in p['experts']}),
        }
        if s == 'vision':
            p['meta']['pulsar.kv'] = json.dumps(kvs, separators=(',', ':'))
    return shape, order, files, shards, consumed


# ---------------------------------------------------------------------------
# the writer: streamed
# ---------------------------------------------------------------------------
def shard_layout(entries, meta):
    """The entries in file order and the exact header bytes (length prefix excluded) -- fixed before any payload."""
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
    return sized, header, cursor, hj


def source_bytes(e, hf):
    """An entry's payload from its source: the byte ranges concatenated, or its producer run."""
    kind, src = e['src']
    if kind == 'produce':
        return PR.produce(src, hf)
    out = b''
    for sp, off, n in src:
        with open(sp, 'rb') as g:
            g.seek(off)
            out += g.read(n)
    return out


def write_dump(path, shape, order, files, shards, consumed):
    """plan --dump: one JSON line for the model, then per shard (file order) its exact header and one line per entry
    with the entry's source descriptor -- everything a build writes except the payload bytes themselves."""
    with open(path, 'w') as f:
        f.write(json.dumps({'model': str(shape), 'consumed': consumed, 'shards': order}) + '\n')
        for s in order:
            sized, _header, _total, hj = shard_layout(shards[s]['entries'], shards[s]['meta'])
            f.write(json.dumps({'shard': s, 'file': files[s], 'header': hj.decode()}) + '\n')
            for e in sized:
                kind, src = e['src']
                d = ['ranges', [list(r) for r in src]] if kind == 'ranges' else ['produce', src]
                f.write(json.dumps({'name': e['name'], 'src': d}, sort_keys=True) + '\n')


def write_shard(path, entries, meta, hf):
    sized, header, cursor, hj = shard_layout(entries, meta)
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
                b = PR.produce(src, hf)
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
# The family modules, registered by the checkpoint's config.json model_type -- the one authority for which naming
# table applies.  deepseek.py: V4 Flash / Vision-Exp and V4.1; qwen.py: qwen4_exp.
FAMILIES = {'deepseek_v4': D, 'deepseek_v41': D, 'qwen4_exp': Q}


def family_of(hf):
    mt = hf.config['top_level'].get('model_type')
    if mt not in FAMILIES:
        raise SystemExit(f'config.json model_type {mt!r}: no family module registered ({sorted(FAMILIES)})')
    return FAMILIES[mt]


def context(args):
    """The family the checkpoint's model_type names and the build's context: its sources, its recipe (None = the
    family's default), the options its producers and kv read."""
    hf = hf_of(args)
    fam = family_of(hf)
    sources = {'hf': hf, 'exl3': Exl3Checkpoint(args.exl3) if args.exl3 else None,
               'exl3_experts': Exl3Checkpoint(args.exl3_experts) if args.exl3_experts else None,
               'tessera': TesseraBundle(args.tessera) if args.tessera else None}
    rec = R.Recipe.load(args.recipe, fam.SETTINGS) if args.recipe else None
    if rec is not None and fam.SETTINGS:
        fam.check_recipe(rec)
    ctx = argparse.Namespace(sources=sources, recipe=rec, mxfp8_mode=args.mxfp8_scale,
                             tokenizer_dir=args.tokenizer or args.hf, reap_map=args.reap_map, ple_rows=args.ple_rows)
    return hf, fam, ctx


def make_plan(args):
    hf, fam, ctx = context(args)
    return plan(hf, fam, ctx)


def cmd_recipe(args):
    """Print the pulsar.recipe.v1 the build runs: the one given, or the family's default for these sources."""
    hf, fam, ctx = context(args)
    if ctx.recipe is None:
        shape = fam.shape(hf, ctx)
        ctx.recipe = fam.default_recipe(hf, {n: fam.map_hf(n, shape) for n in hf.names()}, ctx)
    print(ctx.recipe.to_json())
    return 0


def cmd_plan(args):
    shape, order, files, shards, consumed = make_plan(args)
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
    if args.dump:
        write_dump(args.dump, shape, order, files, shards, consumed)
        print(f'plan dump: {args.dump}')
    print('sources: ' + ', '.join(f'{k} {getattr(args, k)}' for k in ('exl3', 'exl3_experts', 'tessera')
                                  if getattr(args, k)))
    return 0


def cmd_emit(args):
    shape, order, files, shards, consumed = make_plan(args)
    todo = order if args.all else [args.shard]
    os.makedirs(args.out, exist_ok=True)
    import time
    for s in todo:
        if s not in shards:
            raise SystemExit(f'unknown shard {s}; one of {order}')
        t0 = time.time()
        header, total = write_shard(os.path.join(args.out, files[s]), shards[s]['entries'], shards[s]['meta'],
                                    hf_of(args))
        print(f'{files[s]}  {s:10s} {len(header):6d} tensors {total / 1e9:8.3f} GB  {time.time() - t0:6.1f} s', flush=True)
    if args.all:
        n, total = write_index(args.out)
        print(f'index: {n} tensors, total_size {total / 1e9:.2f} GB')
    return 0


def cmd_verify(args):
    """Against the HF SOURCE, not an intermediate: structure and alignment, every
    native span byte-equal, every produced payload re-produced and compared,
    every EXL3 slice equal to its source ranges, the declared byte model equal
    to the span, expert families contiguous.  --roundtrip adds the fidelity
    half: every BF16-sourced mxfp8_lt entry decoded and held to the format's
    own rounding bound against its BF16 source (qwen4_exp)."""
    shape, order, files, shards, consumed = make_plan(args)
    todo = order if args.all else [args.shard]
    failing = 0
    rt_worst = {}
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
                exp = source_bytes(e, hf_of(args))
                if got == exp:
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f'  BYTES DIFFER {name} ({e["layout"]})')
                kind, src = e['src']
                if args.roundtrip and kind == 'produce' and src['producer'] == 'mxfp8_lt_from_bf16':
                    over, rel = mxfp8_roundtrip(hf_of(args), src['inputs'][0], got)
                    rt_worst[name] = rel
                    if over:
                        n_bad += 1
                        print(f'  ROUNDTRIP {name}: {over} elements outside the E4M3 rounding bound')
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
            if 'entry_name' in fam:           # qwen4_exp: the family names its own entries
                entry = fam['entry_name']
            else:                             # DeepSeek: the engine's derivation (safetensors.cpp)
                ns, lay = ('layers', gname.split('.')[1]) if gname.startswith('blk.') else ('mtp', gname.split('.')[1])
                entry = f'{ns}.{lay}.ffn.experts.{{e}}.{fam["part"]}.weight'
            first = None
            for e in range(fam['n_experts']):
                h = hdr.get(entry.replace('{e}', str(e)))
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
    if args.roundtrip:
        if rt_worst:
            w = max(rt_worst, key=rt_worst.get)
            print(f'roundtrip: {len(rt_worst)} mxfp8_lt entries decoded within the E4M3 bound; '
                  f'worst relative Frobenius error {rt_worst[w]:.3e} ({w})')
        else:
            print('roundtrip: no BF16-sourced mxfp8_lt entries in the checked shards')
    print(f'shards checked: {len(todo)}  failing: {failing}')
    return 1 if failing else 0


_HF = {}


def hf_of(args):
    if args.hf not in _HF:
        _HF[args.hf] = HFCheckpoint(args.hf)
    return _HF[args.hf]


def mxfp8_roundtrip(hf, src_name, payload):
    """Decode an mxfp8_lt payload and hold every element to E4M3's rounding bound against the BF16 source:
    with x = w / 2^s (s the group's E8M0 exponent), |q - w| <= 2^(floor(log2|x|) - 4) * 2^s for |x| >= 2^-6
    (half the 3-bit mantissa spacing), <= 2^-10 * 2^s below it (half the subnormal step).  Returns (elements
    outside the bound, relative Frobenius error)."""
    import numpy as np
    out, inp = hf.shape(src_name)
    w = (np.frombuffer(hf.raw(src_name), dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(out, inp)
    q = PR.mxfp8_lt_decode(payload, out, inp)
    kb = inp // 32
    s = PR.unswizzle_sf(np.frombuffer(payload, dtype=np.uint8, offset=out * inp), out, kb).astype(np.int32) - 127
    s = np.repeat(s, 32, axis=1)
    x = np.abs(np.ldexp(w, -s))
    e = np.floor(np.log2(np.where(x > 0, x, 1.0))).astype(np.int32)
    bound = np.where(x >= 2.0 ** -6, np.ldexp(1.0, e - 4 + s), np.ldexp(1.0, s - 10))
    over = int((np.abs(q.astype(np.float64) - w) > bound).sum())
    rel = float(np.linalg.norm(q.astype(np.float64) - w) / max(np.linalg.norm(w.astype(np.float64)), 1e-30))
    return over, rel


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
            if 'entry_name' in e:
                names_ |= {e['entry_name'].replace('{e}', str(x)) for x in range(e['n_experts'])}
                continue
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
    for name, fn in (('recipe', cmd_recipe), ('plan', cmd_plan), ('emit', cmd_emit), ('verify', cmd_verify),
                     ('audit', cmd_audit)):
        p = sub.add_parser(name)
        if name != 'audit':
            p.add_argument('--hf', required=True, help='the HF checkpoint directory (config.json + shards)')
            p.add_argument('--exl3', metavar='DIR', help='an exllamav3 checkpoint: the source `exl3` recipe rows read')
            p.add_argument('--exl3-experts', metavar='DIR', help='a second exllamav3 checkpoint: source `exl3_experts` '
                           '(the default source of a routed expert\'s EXL3 rows when given)')
            p.add_argument('--tessera', metavar='DIR', help='a Tessera bundle: source `tessera` (the producer lands with L255)')
            p.add_argument('--recipe', metavar='JSON', help='a pulsar.recipe.v1 (recipe.py; format-maps/*.json).  '
                           'DeepSeek default: generated from policy.py + the EXL3 source (`recipe` prints it)')
            p.add_argument('--mxfp8-scale', choices=('rederive', 'verbatim'), default='rederive',
                           help='FP8 dense -> mxfp8_lt: re-derive the per-32 E8M0 (byte-identical to the archived codec) or broadcast the source block scale')
            p.add_argument('--tokenizer', metavar='DIR', help='tokenizer files (default: the HF dir)')
            p.add_argument('--reap-map', metavar='JSON')
            p.add_argument('--ple-rows', metavar='MANIFEST', help='qwen4_exp: the PLE row file manifest (ple_rows.py build)')
        if name == 'plan':
            p.add_argument('--dump', metavar='FILE', help='write every shard\'s exact header and every entry\'s source '
                           'descriptor (no payload bytes): two plans that dump the same write the same bytes')
        if name == 'verify':
            p.add_argument('--roundtrip', action='store_true',
                           help='also decode every BF16-sourced mxfp8_lt entry and hold it to the E4M3 rounding bound')
        if name not in ('plan', 'recipe'):
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

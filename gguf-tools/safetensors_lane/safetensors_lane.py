#!/usr/bin/env python3
"""Emit the served model as a per-layer-shard safetensors checkpoint.

WHAT THIS IS. Every payload in our checkpoint is a KERNEL READ PATTERN, not a
container choice -- iq2_xxs_mmq_k alone is 218 vs 88 GB/s -- and the engine has
no load-time rebuilder for any of them (the aligned-IQ2 repack cache was deleted
2026-08-15).  So the emitter copies payloads VERBATIM under a declared layout id
and never converts; the declarations are what make the file readable.

THE CONTRACT, fixed by the performance requirements rather than by taste:

  * PAYLOAD VERBATIM, layout declared.  Engine layouts are written as `U8` with
    their logical dims in `pulsar.tensors` (`layout`, `dims_ne`, `gguf_name`);
    only tensors with no engine-specific ordering keep a native dtype.

  * GAP-FREE BUFFER, ALIGNMENT BY ORDER.  safetensors requires the data buffer
    to tile [0, end) with no gaps -- the reference reader rejects any tensor
    whose offset is not exactly the previous end ("invalid offset").  The first
    version of this writer padded BETWEEN tensors to force 32-byte alignment and
    every one of our own gates passed; the reference library rejected all 48
    shards.  Alignment therefore comes from ORDER: place tensors by descending
    alignment requirement and each lands on its own alignment for free.  The
    sort is stable, so a projection's per-expert run stays adjacent.

  * CONTIGUOUS EXPERT BLOCKS.  The kernels index experts as
    base + xid*expert_bytes from ONE pointer, so a projection's per-expert
    tensors must be back-to-back with zero inter-expert padding:
    offset(E) == offset(0) + E*expert_bytes, asserted per projection.  Both
    expert strides happen to be 32-multiples (IQ2 2,162,688 B, MXFP4
    4,456,448 B), which is what makes that possible.

WHERE THE NAMES COME FROM.  `quantize/dsq_names.c` in THIS tree -- parsed, not
imported, so a copy cannot drift.  `verify` and `emit` both refuse to write a
tensor they cannot name, because an unnamed tensor is a hole in the checkpoint
and finding it after writing 92 GB costs a rebuild.

Subcommands:
    plan    --gguf G            print the shard plan and audit name coverage
    emit    --gguf G --out DIR [--shard S | --all]   write shard(s)
    verify  --gguf G --out DIR [--shard S | --all]   byte-exact gate vs the GGUF
    audit   --out DIR           structure + 32B alignment + index closure

  plan / emit / verify take `--exl3-experts DIR [--exl3-layers 5,18-22]`: the
  routed experts of those layers (default: every blk layer the EXL3 checkpoint
  holds) are sourced from an EXL3 checkpoint (exllamav3's HF shards -- the
  public Mia-AiLab V4.1 build, or our own convert output) instead of the GGUF,
  as one contiguous [trellis | suh | svh] slice per expert under the layout
  `exl3m_k2|k2h|k3` the trellis width names (L245).  The bytes are copied
  VERBATIM from the EXL3 shards; the mul1 codebook is required, the rate is
  read off the trellis' last dim, and gate/up must share one rate.  `verify`
  reassembles those experts from the EXL3 shards.  The shard plan (one shard
  per layer) follows the GGUF's own layer count, not a constant.

The `verify` gate is the stage-1 gate: it checks the file against the GGUF
DIRECTLY (not against any intermediate dump), including reassembling each
projection's 256 per-expert payloads and comparing the concatenation with the
GGUF's stacked tensor.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DSQ_NAMES = os.path.join(REPO, 'gguf-tools', 'quantize', 'dsq_names.c')
sys.path.insert(0, os.path.join(REPO, 'gguf-tools'))

ALIGN = 32

# GGUF tensor type -> the layout id we declare.  These are the trade's names,
# not ggml's: the engine resolves a declaration through its own layout table.
BLOB = {40: 'cutlass_mxfp4', 41: 'mxfp8_lt', 44: 'iq2_xxs_mmq_k', 46: 'fp8_e4m3_soa_k'}
NATIVE = {0: 'F32', 26: 'I32', 30: 'BF16'}
NATIVE_BYTES = {'F32': 4, 'I32': 4, 'BF16': 2}

EXP = re.compile(r'^(blk|dspark)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
EXP_PART = {'gate': 'w1', 'up': 'w3', 'down': 'w2'}
EXP_HF = re.compile(r'^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$')

SHARD_ORDER = None      # set by set_shard_plan() from the GGUF's own layer counts
SHARD_FILE = None


def set_shard_plan(tensors):
    """One shard per layer, plus the vision tower, the head/norm and each drafter
    layer: the counts come from the tensor names (0731: 43 + 3 -> 48 shards;
    V4.1: 40 + 3 -> 45), never from a constant."""
    global SHARD_ORDER, SHARD_FILE
    n_layers = n_mtp = 0
    for t in tensors:
        m = re.match(r'^(blk|dspark)\.(\d+)\.', t['name'])
        if not m:
            continue
        if m.group(1) == 'blk':
            n_layers = max(n_layers, int(m.group(2)) + 1)
        else:
            n_mtp = max(n_mtp, int(m.group(2)) + 1)
    if n_layers == 0:
        raise SystemExit('no blk.N tensors in the GGUF -- cannot lay out shards')
    SHARD_ORDER = (['vision'] + [f'layers.{i}' for i in range(n_layers)] + ['top']
                   + [f'mtp.{i}' for i in range(n_mtp)])
    n = len(SHARD_ORDER)
    SHARD_FILE = {s: f'model-{i:05d}-of-{n:05d}.safetensors' for i, s in enumerate(SHARD_ORDER, 1)}


# --------------------------------------------------------------------------
# The EXL3 expert source (L245): exllamav3's HF shards, read by header, no
# torch.  Mirrors src/engine/exl3_trellis.h: words per 16x16 tile name the
# rate (16K, or 16K+8 for the half-integer rates), and one expert-projection
# is [trellis | suh | svh] with trellis (in/16)(out/16) tiles x words x 2 B
# and the two fp16 scale vectors -- the engine refuses a stack whose declared
# expert_bytes disagrees with that model, so this and the header are checked
# against each other at every load.
# --------------------------------------------------------------------------
EXL3_LAYOUT = {32: 'exl3m_k2', 40: 'exl3m_k2h', 48: 'exl3m_k3'}


def exl3_expert_bytes(k, n, words):
    if k % 128 or n % 128:
        raise SystemExit(f'exl3: dims ({k}, {n}) are not multiples of 128')
    trellis = (k // 16) * (n // 16) * words * 2
    return trellis, (k + n) * 2


class Exl3Checkpoint:
    """name -> (shard path, absolute byte offset, byte count, dtype, shape) from
    every model-*.safetensors header in the directory (a partial download of the
    layers under test is enough; the index is not required)."""

    def __init__(self, hf_dir):
        self.dir = hf_dir
        self.entries = {}
        shards = sorted(f for f in os.listdir(hf_dir)
                        if f.startswith('model-') and f.endswith('.safetensors'))
        if not shards:
            raise SystemExit(f'{hf_dir}: no model-*.safetensors shards')
        for f in shards:
            path = os.path.join(hf_dir, f)
            with open(path, 'rb') as fh:
                (n,) = struct.unpack('<Q', fh.read(8))
                hdr = json.loads(fh.read(n))
            hdr.pop('__metadata__', None)
            for name, h in hdr.items():
                o0, o1 = h['data_offsets']
                self.entries[name] = (path, 8 + n + o0, o1 - o0, h['dtype'], h['shape'])

    def layers(self):
        return sorted({int(m.group(1)) for m in
                       (re.match(r'^layers\.(\d+)\.ffn\.experts\.0\.w1\.trellis$', k)
                        for k in self.entries) if m})

    def expert(self, layer, e, part, k, n):
        """The three source ranges of one expert-projection and the words per
        tile, after every refusal the format allows."""
        key = f'layers.{layer}.ffn.experts.{e}.{part}'
        if f'{key}.mcg' in self.entries:
            raise SystemExit(f'{key}: mcg codebook -- pulsar reads the mul1 codebook only')
        for sub in ('trellis', 'suh', 'svh', 'mul1'):
            if f'{key}.{sub}' not in self.entries:
                raise SystemExit(f'{key}.{sub}: missing from the EXL3 checkpoint')
        tp, to, tn, tdt, tsh = self.entries[f'{key}.trellis']
        up, uo, un, udt, ush = self.entries[f'{key}.suh']
        vp, vo, vn, vdt, vsh = self.entries[f'{key}.svh']
        if tdt != 'I16' or len(tsh) != 3 or tsh[0] != k // 16 or tsh[1] != n // 16:
            raise SystemExit(f'{key}.trellis: dtype {tdt} shape {tsh}, expected I16 [{k // 16}, {n // 16}, words]')
        words = tsh[2]
        if words not in EXL3_LAYOUT:
            raise SystemExit(f'{key}.trellis: {words} words per tile is not a rate pulsar reads '
                             f'({sorted(EXL3_LAYOUT)})')
        if udt != 'F16' or ush != [k] or vdt != 'F16' or vsh != [n]:
            raise SystemExit(f'{key}: suh {udt}{ush} / svh {vdt}{vsh}, expected F16 [{k}] / F16 [{n}]')
        trellis, scales = exl3_expert_bytes(k, n, words)
        if tn != trellis or un + vn != scales:
            raise SystemExit(f'{key}: {tn} + {un} + {vn} bytes on disk, the layout says {trellis} + {scales}')
        return [(tp, to, tn), (up, uo, un), (vp, vo, vn)], words


# --------------------------------------------------------------------------
# The producer's byte model.
#
# The engine checks every declaration against the file's own span
# (st_bytes_for in safetensors.cpp), so this must agree with it exactly; a
# disagreement is caught as a sha mismatch by `verify`, and independently by the
# GGUF-offset self-check in scan_gguf() below.  ne order throughout: dim[0] is
# the INPUT/columns dimension, which is how the engine reads it.
# --------------------------------------------------------------------------
def tensor_bytes(ttype, dims):
    if ttype in NATIVE:
        n = 1
        for d in dims:
            n *= d
        return n * NATIVE_BYTES[NATIVE[ttype]]
    if ttype == 44:                                   # iq2_xxs_mmq_k
        n = 1
        for d in dims:
            n *= d
        if n % 256:
            raise SystemExit(f'iq2_xxs_mmq_k: {dims} is not a multiple of 256')
        return n // 256 * 66
    if ttype in (41, 46):                             # mxfp8_lt / fp8_e4m3_soa_k
        cols, rows = dims[0], dims[1]
        if cols % 32:
            raise SystemExit(f'mxfp8: input dim {cols} is not a multiple of 32')
        if ttype == 41:
            kb_pad = (cols // 32 + 3) // 4 * 4
            return rows * cols + (rows + 127) // 128 * 128 * kb_pad
        return rows * cols + rows * (cols // 32)
    if ttype == 40:                                   # cutlass_mxfp4, per expert
        k, n = dims[0], dims[1]
        k_pad, n_pad = (k + 127) // 128 * 128, (n + 127) // 128 * 128
        return (n * k // 2 + (n_pad // 32) * k_pad) * dims[2]
    raise SystemExit(f'no byte model for tensor type {ttype}')


def align_for(nbytes):
    """The strongest alignment a tensor's own size permits."""
    for a in (32, 4, 2):
        if nbytes % a == 0:
            return a
    return 1


# --------------------------------------------------------------------------
# The GGUF source: header (via the shared in-tree scanner) plus the KV block.
# --------------------------------------------------------------------------
def _read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')


_SCALARS = {0: ('<B', 'u8'), 1: ('<b', 'i8'), 2: ('<H', 'u16'), 3: ('<h', 'i16'),
            4: ('<I', 'u32'), 5: ('<i', 'i32'), 6: ('<f', 'f32'), 7: ('<B', 'bool'),
            10: ('<Q', 'u64'), 11: ('<q', 'i64'), 12: ('<d', 'f64')}


def _read_value(f, t):
    """A typed GGUF KV value.  Types are named, because that is what the
    checkpoint stores and what the engine's meta_code_for_type_name() reads --
    no ggml NUMBER leaves this function."""
    if t in _SCALARS:
        fmt, name = _SCALARS[t]
        v = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        return name, bool(v) if name == 'bool' else v
    if t == 8:
        return 'string', _read_str(f)
    if t == 9:
        (et,) = struct.unpack('<I', f.read(4))
        (n,) = struct.unpack('<Q', f.read(8))
        vals = []
        ename = None
        for _ in range(n):
            ename, v = _read_value(f, et)
            vals.append(v)
        return 'array', {'__array__': ename, 'n': n, 'v': vals}
    raise SystemExit(f'unknown GGUF KV value type {t}')


def read_kvs(path):
    """Every KV in FILE ORDER, as {'key','type','value'} -- the shape the
    checkpoint stores.  Order matters: it is what makes the emitted header
    reproducible."""
    out = []
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit(f'{path}: not a GGUF')
        struct.unpack('<I', f.read(4))
        struct.unpack('<Q', f.read(8))
        (n_kv,) = struct.unpack('<Q', f.read(8))
        for _ in range(n_kv):
            key = _read_str(f)
            (vt,) = struct.unpack('<I', f.read(4))
            tname, value = _read_value(f, vt)
            out.append({'key': key, 'type': tname, 'value': value})
    return out


def scan_gguf(path):
    """-> (tensors, data_start).  Each tensor carries name/dims/type/off/bytes,
    with `bytes` computed from the byte model and SELF-CHECKED against the next
    tensor's offset: a wrong byte model would desynchronise the copy."""
    from gguf_hdr import scan
    ts, _align, data_start = scan(path)
    size = os.path.getsize(path)
    out = []
    for i, t in enumerate(ts):
        nb = tensor_bytes(t['type'], t['dims'])
        nxt = ts[i + 1]['offset'] if i + 1 < len(ts) else (size - data_start)
        pad = nxt - t['offset'] - nb
        if not (0 <= pad < ALIGN):
            raise SystemExit(
                f'byte model disagrees with the file at {t["name"]}: model says '
                f'{nb} B, the next tensor starts {nxt - t["offset"]} B later '
                f'(padding {pad} is not in [0,{ALIGN}))')
        out.append({'name': t['name'], 'dims': list(t['dims']), 'type': t['type'],
                    'off': t['offset'], 'bytes': nb})
    return out, data_start


# --------------------------------------------------------------------------
# Names: parsed out of THIS tree's dsq_names.c.
# --------------------------------------------------------------------------
def _parse_map(src, table):
    m = re.search(rf'static const name_map {table}\[\] = \{{(.*?)\n\}};', src, re.S)
    if not m:
        raise SystemExit(f'table {table} not found in {DSQ_NAMES}')
    return dict(re.findall(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}', m.group(1)))


VISION_PREFIX = ('vision.', 'aligner.')
VISION_TOP = {'image_start', 'image_end', 'image_newline', 'image_pad'}


class Names:
    def __init__(self):
        src = open(DSQ_NAMES).read()
        self.top = _parse_map(src, 'top_map')
        self.layer = _parse_map(src, 'layer_map')

    def resolve(self, gguf):
        """-> (hf_name, shard) or (None, why).  Mirrors hf_name_for_regular()
        plus the shard split: one shard per layer, and the vision tower, the
        head/norm, and each drafter layer of their own."""
        if gguf in self.top:
            return self.top[gguf], 'top'
        m = EXP.match(gguf)
        if m:
            ns = 'layers' if m.group(1) == 'blk' else 'mtp'
            return (f'{ns}.{int(m.group(2))}.ffn.experts.{{E}}.{EXP_PART[m.group(3)]}.weight',
                    f'{ns}.{int(m.group(2))}')
        m = re.match(r'^(blk|dspark)\.(\d+)\.(.+)$', gguf)
        if m:
            suffix = m.group(3)
            if suffix in self.layer:
                ns = 'layers' if m.group(1) == 'blk' else 'mtp'
                return f'{ns}.{int(m.group(2))}.{self.layer[suffix]}', f'{ns}.{int(m.group(2))}'
            return None, f'no layer_map entry for suffix {suffix!r}'
        if gguf.startswith(VISION_PREFIX) or gguf in VISION_TOP:
            return gguf, 'vision'
        m = re.match(r'^dspark\.(.+)$', gguf)
        if m:
            rest = m.group(1)
            hf = self.layer.get(rest, rest)
            return f'mtp.0.{hf}', 'mtp.0'
        return None, 'unclassified'


# --------------------------------------------------------------------------
# Declarations: what the engine reads out of each shard's __metadata__.
# --------------------------------------------------------------------------
def build_declarations(tensors, names, kvs, exl3=None, exl3_layers=None):
    """-> (plan, index, config-pieces).

    plan[shard] = {'entries': [...], 'tensors': {hf: decl},
                   'experts': [decl], 'meta': {...}}
    """
    kv_arch = [k for k in kvs if not k['key'].startswith('tokenizer.')]
    plan = {}
    for s in SHARD_ORDER:
        plan[s] = {'entries': [], 'tensors': {}, 'experts': [], 'index': {}}

    for t in tensors:
        hf, shard = names.resolve(t['name'])
        if hf is None:
            raise SystemExit(f'{t["name"]}: {shard}')
        p = plan[shard]
        m = EXP.match(t['name'])
        if m:
            ns = 'layers' if m.group(1) == 'blk' else 'mtp'
            n_exp = t['dims'][2]
            eb = t['bytes'] // n_exp
            part = EXP_PART[m.group(3)]
            layer = int(m.group(2))
            layout = BLOB[t['type']]
            per_expert = None
            if exl3 is not None and ns == 'layers' and layer in exl3_layers:
                k, n = t['dims'][0], t['dims'][1]
                per_expert, words = [], None
                for e in range(n_exp):
                    ranges, w = exl3.expert(layer, e, part, k, n)
                    if words is None:
                        words = w
                    elif w != words:
                        raise SystemExit(f'layers.{layer}.ffn.experts.{e}.{part}: {w} words per tile, '
                                         f'expert 0 has {words} -- one rate per family')
                    per_expert.append(ranges)
                trellis, scales = exl3_expert_bytes(k, n, words)
                eb, layout = trellis + scales, EXL3_LAYOUT[words]
            p['experts'].append({
                'gguf_name': t['name'], 'part': part, 'n_experts': n_exp,
                'expert_bytes': eb, 'layout': layout, 'contiguous': True,
                # ne order, NOT the HF reversal: one convention everywhere so a
                # reader never has to know which key holds which orientation
                'dims_per_expert_ne': list(t['dims'][:2])})
            for e in range(n_exp):
                name = f'{ns}.{layer}.ffn.experts.{e}.{part}.weight'
                ranges = (per_expert[e] if per_expert is not None
                          else [(None, t['off'] + e * eb, eb)])
                p['entries'].append({'name': name, 'dtype': 'U8', 'shape': [eb],
                                     'ranges': ranges, 'layout': layout})
                p['index'][name] = shard
            continue
        layout = BLOB.get(t['type'], 'native')
        if layout == 'native':
            dtype = NATIVE[t['type']]
            shape = list(reversed(t['dims']))
            ne = 1
            for d in shape:
                ne *= d
            if ne * NATIVE_BYTES[dtype] != t['bytes']:
                raise SystemExit(f'{t["name"]}: native shape does not account for its bytes')
        else:
            dtype, shape = 'U8', [t['bytes']]
        p['entries'].append({'name': hf, 'dtype': dtype, 'shape': shape,
                             'ranges': [(None, t['off'], t['bytes'])], 'layout': layout})
        p['tensors'][hf] = {'layout': layout, 'dims_ne': t['dims'],
                            'gguf_name': t['name']}
        p['index'][hf] = shard

    for shard, p in plan.items():
        exp_layouts = {e['layout'] for e in p['experts']}
        exl3_layouts = {l for l in exp_layouts if l.startswith('exl3m_')}
        # gate/up must share one rate per layer (the fused arm decodes them
        # together); the engine refuses the mix at load, the lane refuses it here
        for lay in {e['gguf_name'].split('.')[1] for e in p['experts']}:
            gu = {e['layout'] for e in p['experts']
                  if e['gguf_name'].split('.')[1] == lay and e['part'] in ('w1', 'w3')}
            if len(gu) > 1:
                raise SystemExit(f'{shard}: layer {lay} gate/up layouts differ: {sorted(gu)}')
        md = {
            'format': 'pt',                     # the near-universal HF convention
            'pulsar.format': 'pulsar-safetensors-v1',
            'pulsar.alignment': str(ALIGN),
            'pulsar.shard': SHARD_FILE[shard],
            'pulsar.shard_key': shard,
            'pulsar.n_shards': str(len(SHARD_ORDER)),
            'pulsar.primary': SHARD_FILE['vision'],
            'pulsar.tensors': json.dumps(p['tensors'], separators=(',', ':'), sort_keys=True),
            'pulsar.experts': json.dumps(p['experts'], separators=(',', ':')),
            'pulsar.kv_arch': json.dumps(kv_arch, separators=(',', ':')),
            # deliberately not "fp4": prismaquant's declared_fp4_expert_dtype
            # would turn on its MXFP4 nibble decode for any 2-D int8/uint8 expert
            # tensor, which is exactly what our U8 IQ2 superblocks look like.
            'pulsar.expert_dtype': ('none' if not exp_layouts else
                                    'iq2_xxs' if exp_layouts <= {'iq2_xxs_mmq_k'} else
                                    'mxfp4_cutlass' if exp_layouts <= {'cutlass_mxfp4'} else
                                    'exl3' if exp_layouts == exl3_layouts else
                                    'mixed'),
        }
        if shard == 'vision':
            md['pulsar.kv'] = json.dumps(kvs, separators=(',', ':'))
        p['meta'] = md
    return plan


def write_shard(path, entries, meta, gguf_fh, data_start):
    """Write one shard.  See the module docstring for the alignment contract.
    A range is (source, offset, bytes): source None reads the GGUF's data
    section, a path reads that file at an absolute offset (the EXL3 shards)."""
    sized = [(e['name'], e['dtype'], e['shape'], e['ranges'], e['layout'],
              sum(n for _, _, n in e['ranges'])) for e in entries]
    sized.sort(key=lambda e: -align_for(e[5]))          # stable: expert runs stay adjacent

    header, blobs, cursor, hist, handles = {}, [], 0, {}, {}
    for name, dtype, shape, ranges, layout, need in sized:
        a = align_for(need)
        if cursor % a:
            raise SystemExit(f'{name}: offset {cursor} violates {a}-byte alignment')
        hist[a] = hist.get(a, 0) + 1
        start = cursor
        for src, off, n in ranges:
            if src is None:
                fh, base = gguf_fh, data_start
            else:
                if src not in handles:
                    handles[src] = open(src, 'rb')
                fh, base = handles[src], 0
            fh.seek(base + off)
            b = fh.read(n)
            if len(b) != n:
                raise SystemExit(f'{name}: short read from {src or "the GGUF"}')
            blobs.append(b)
        cursor += need
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [start, cursor]}

    # The reference writer serialises a BTreeMap: sorted keys, no spaces, and
    # __metadata__ as an ordinary key in that same map (it sorts first, '_' <
    # letters).  Matching it is what keeps our output reproducible byte for byte.
    hj = json.dumps({**header, '__metadata__': meta},
                    separators=(',', ':'), sort_keys=True).encode()
    hj += b' ' * (-(8 + len(hj)) % ALIGN)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    for fh in handles.values():
        fh.close()
    return header, cursor, hist


def read_shard(path):
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        hdr = json.loads(f.read(n))
        meta = hdr.pop('__metadata__', {})
        return hdr, meta, 8 + n, os.path.getsize(path)


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------
def load(gguf):
    tensors, data_start = scan_gguf(gguf)
    set_shard_plan(tensors)
    names = Names()
    kvs = read_kvs(gguf)
    return tensors, data_start, names, kvs


def parse_layers(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def exl3_source(args):
    """-> (Exl3Checkpoint, layer set) or (None, None)."""
    d = getattr(args, 'exl3_experts', None)
    if not d:
        return None, None
    ck = Exl3Checkpoint(d)
    have = set(ck.layers())
    if not have:
        raise SystemExit(f'{d}: no layers.N.ffn.experts.0.w1.trellis in any shard')
    want = parse_layers(args.exl3_layers) if args.exl3_layers else have
    missing = sorted(want - have)
    if missing:
        raise SystemExit(f'{d}: layers {missing} requested but not present (have {sorted(have)})')
    return ck, want


def plan_for(args):
    tensors, data_start, names, kvs = load(args.gguf)
    if getattr(args, 'shard', None) and args.shard not in SHARD_ORDER:
        raise SystemExit(f'unknown shard {args.shard!r}; one of {SHARD_ORDER}')
    exl3, layers = exl3_source(args)
    plan = build_declarations(tensors, names, kvs, exl3, layers)
    return tensors, data_start, names, kvs, plan, exl3, layers


def cmd_plan(args):
    tensors, _ds, names, _kvs, plan, exl3, layers = plan_for(args)
    holes = 0
    per_shard = {s: (len(p['entries']), len(p['experts'])) for s, p in plan.items()}
    for t in tensors:
        hf, shard = names.resolve(t['name'])
        if hf is None:
            holes += 1
            print(f'  HOLE {t["name"]}: {shard}')
    print(f'gguf tensors: {len(tensors)}')
    print(f'shards: {len(SHARD_ORDER)}   unmapped: {holes}')
    if exl3 is not None:
        fams = [e for p in plan.values() for e in p['experts'] if e['layout'].startswith('exl3m_')]
        print(f'exl3 experts from {exl3.dir}: layers {sorted(layers)}, {len(fams)} families, '
              f'{sum(e["n_experts"] * e["expert_bytes"] for e in fams) / 1e9:.2f} GB')
    for s in SHARD_ORDER:
        e, x = per_shard[s]
        print(f'  {SHARD_FILE[s]}  {s:10s} {e:6d} tensors  {x:3d} expert families')
    return 1 if holes else 0


def cmd_emit(args):
    _tensors, data_start, _names, _kvs, plan, _exl3, _layers = plan_for(args)
    shards = SHARD_ORDER if args.all else [args.shard]
    os.makedirs(args.out, exist_ok=True)
    with open(args.gguf, 'rb') as fh:
        for s in shards:
            p = plan[s]
            path = os.path.join(args.out, SHARD_FILE[s])
            header, total, hist = write_shard(path, p['entries'], p['meta'], fh, data_start)
            print(f'{SHARD_FILE[s]}  {s:10s} {len(p["entries"]):6d} tensors '
                  f'{total/1e9:7.3f} GB  align={dict(sorted(hist.items(), reverse=True))}')
    return 0


def cmd_verify(args):
    """Stage-1 gate: is the shard a LOSSLESS re-containerization of the GGUF?

    Checks the file against the GGUF directly, not against any intermediate:
    structure (gap-free, 32 B header, exact size), alignment per tensor's own
    requirement, non-expert payloads byte-identical, and each projection's 256
    per-expert payloads reassembled byte-identical to the GGUF's stacked tensor
    -- the property the kernels' base + xid*expert_bytes arithmetic depends on.
    """
    tensors, data_start, names, _kvs, plan, _exl3, _layers = plan_for(args)
    by_name = {t['name']: t for t in tensors}
    man = {t['name']: names.resolve(t['name']) for t in tensors}
    shards = SHARD_ORDER if args.all else [args.shard]
    bad_total = 0
    for s in shards:
        path = os.path.join(args.out, SHARD_FILE[s])
        hdr, meta, buf_off, size = read_shard(path)
        # a shard may hold metadata and no tensors (V4.1 has no vision tower,
        # so its primary shard carries only the kv block): end = 0, no offsets
        end = max((h['data_offsets'][1] for h in hdr.values()), default=0)
        offs = sorted((h['data_offsets'][0], h['data_offsets'][1]) for h in hdr.values())
        gaps = sum(1 for a, b in zip(offs, offs[1:]) if a[1] != b[0])
        structure = (buf_off % ALIGN == 0) and (size == buf_off + end) and (not offs or offs[0][0] == 0) and gaps == 0

        decl = json.loads(meta['pulsar.tensors'])
        mis_align = []
        for name, h in hdr.items():
            o = h['data_offsets'][0]
            layout = decl.get(name, {}).get('layout')
            engine = (layout is not None and layout != 'native') or EXP_HF.match(name)
            if engine and o % ALIGN:
                mis_align.append(name)
            if o % {'U8': 1, 'BF16': 2, 'F32': 4, 'I32': 4, 'I64': 8}.get(h['dtype'], 1):
                mis_align.append(name)

        g = open(args.gguf, 'rb')
        sf = open(path, 'rb')

        def gguf_bytes(t):
            g.seek(data_start + t['off'])
            return g.read(t['bytes'])

        def st_bytes(h):
            o0, o1 = h['data_offsets']
            sf.seek(buf_off + o0)
            return sf.read(o1 - o0)

        rev = {hf: gn for gn, (hf, sh) in man.items() if sh == s and not EXP.match(gn)}
        n_ok = n_bad = unexpected = 0
        for name, h in hdr.items():
            if EXP_HF.match(name):
                continue
            gn = rev.get(name)
            if gn is None:
                unexpected += 1
                print(f'  UNEXPECTED {name}')
                continue
            if hashlib.sha256(gguf_bytes(by_name[gn])).digest() == \
               hashlib.sha256(st_bytes(h)).digest():
                n_ok += 1
            else:
                n_bad += 1
                print(f'  BYTES DIFFER {name} ({gn})')
        e_ok = e_bad = x_ok = x_bad = 0
        exl3_entries = {e['name']: e for e in plan[s]['entries']
                        if e['layout'].startswith('exl3m_')}
        src_handles = {}
        for gn, (hf_t, sh) in man.items():
            if sh != s:
                continue
            m = EXP.match(gn)
            if not m:
                continue
            t = by_name[gn]
            part = EXP_PART[m.group(3)]
            ns = 'layers' if m.group(1) == 'blk' else 'mtp'
            n_exp = t['dims'][2]
            first = f'{ns}.{int(m.group(2))}.ffn.experts.0.{part}.weight'
            if first in exl3_entries:
                # each expert against ITS source ranges: [trellis | suh | svh]
                # verbatim from the EXL3 shards, then the declared expert_bytes
                bad = 0
                for e in range(n_exp):
                    ent = exl3_entries[f'{ns}.{int(m.group(2))}.ffn.experts.{e}.{part}.weight']
                    want = b''
                    for src, off, n in ent['ranges']:
                        if src not in src_handles:
                            src_handles[src] = open(src, 'rb')
                        src_handles[src].seek(off)
                        want += src_handles[src].read(n)
                    got = st_bytes(hdr[ent['name']])
                    if got != want or len(got) != ent['shape'][0]:
                        bad += 1
                if bad:
                    x_bad += 1
                    print(f'  EXL3 EXPERT BYTES DIFFER {gn} ({bad} experts)')
                else:
                    x_ok += 1
                continue
            eb = t['bytes'] // n_exp
            cat = b''.join(st_bytes(hdr[f'{ns}.{int(m.group(2))}.ffn.experts.{e}.{part}.weight'])
                           for e in range(n_exp))
            if cat == gguf_bytes(t):
                e_ok += 1
            else:
                e_bad += 1
                print(f'  EXPERT REASSEMBLY DIFFERS {gn}')
        for fh in src_handles.values():
            fh.close()
        # expert contiguity: offset(E) == offset(0) + E*expert_bytes
        proj = {}
        for name, h in hdr.items():
            m = EXP_HF.match(name)
            if m:
                proj.setdefault((m.group(1), m.group(2), m.group(4)), []).append((int(m.group(3)), h))
        contig_bad = 0
        for k, v in proj.items():
            v.sort()
            o0 = v[0][1]['data_offsets'][0]
            eb = v[0][1]['data_offsets'][1] - o0
            for e, h in v:
                if h['data_offsets'][0] != o0 + e * eb:
                    contig_bad += 1
                    print(f'  CONTIGUITY BROKEN {k}')
                    break

        ok = (structure and not mis_align and n_bad == 0 and e_bad == 0 and x_bad == 0
              and unexpected == 0 and contig_bad == 0)
        bad_total += 0 if ok else 1
        print(f'{SHARD_FILE[s]}  {s:10s} structure={structure} misaligned={len(mis_align)} '
              f'non-expert {n_ok}/{n_ok+n_bad} experts {e_ok}/{e_ok+e_bad} '
              f'exl3 {x_ok}/{x_ok+x_bad} contiguity_bad={contig_bad} -> {"PASS" if ok else "FAIL"}')
        g.close()
        sf.close()
    print(f'shards checked: {len(shards)}  failing: {bad_total}')
    return 1 if bad_total else 0


def cmd_audit(args):
    """Directory-level audit: structure, and the index closed BOTH ways."""
    shards = sorted(f for f in os.listdir(args.out)
                    if f.startswith('model-') and f.endswith('.safetensors'))
    idx_path = os.path.join(args.out, 'model.safetensors.index.json')
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {'weight_map': {}}
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in shards)
    if idx.get('metadata'):
        idx['metadata']['total_size'] = total
        json.dump(idx, open(idx_path, 'w'), indent=0, sort_keys=True)
    bad, nod, declared = [], [], {}
    for f in shards:
        p = os.path.join(args.out, f)
        with open(p, 'rb') as fh:
            (n,) = struct.unpack('<Q', fh.read(8))
            hdr = json.loads(fh.read(n))
        md = hdr.pop('__metadata__', {})
        end = max((h['data_offsets'][1] for h in hdr.values()), default=0)   # a tensor-less shard is legal
        if (8 + n) % ALIGN or os.path.getsize(p) != 8 + n + end:
            bad.append(f)
        for k in ('format', 'pulsar.format', 'pulsar.alignment', 'pulsar.tensors',
                  'pulsar.experts', 'pulsar.kv_arch', 'pulsar.expert_dtype'):
            if k not in md:
                nod.append((f, k))
        if md.get('format') != 'pt':
            nod.append((f, 'format!=pt'))
        names = set(hdr) | set(json.loads(md['pulsar.tensors']))
        for e in json.loads(md['pulsar.experts']):
            ns = 'layers' if e['gguf_name'].startswith('blk.') else 'mtp'
            lay = e['gguf_name'].split('.')[1]
            names |= {f'{ns}.{lay}.ffn.experts.{x}.{e["part"]}.weight'
                      for x in range(e['n_experts'])}
        declared[f] = names
    mismatch = sum(1 for n, s in idx['weight_map'].items() if n not in declared.get(s, set()))
    unindexed = sum(1 for s, ns in declared.items() for n in ns
                    if idx['weight_map'].get(n) != s)
    print(f'shards: {len(shards)}  total_size: {total/1e9:.2f} GB  index entries: {len(idx["weight_map"])}')
    print(f'structurally bad: {len(bad)} {bad[:3]}')
    print(f'metadata problems: {len(nod)} {nod[:3]}')
    print(f'indexed but not declared by its shard: {mismatch}')
    print(f'on disk with no/wrong index entry: {unindexed}')
    print(f'index shard files missing: {sorted(set(idx["weight_map"].values()) - set(shards))}')
    return 1 if bad or nod or mismatch or unindexed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('plan', cmd_plan), ('emit', cmd_emit),
                     ('verify', cmd_verify), ('audit', cmd_audit)):
        p = sub.add_parser(name)
        if name != 'audit':
            p.add_argument('--gguf', required=True)
        p.add_argument('--out', required=(name in ('emit', 'verify', 'audit')))
        if name in ('emit', 'verify'):
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument('--shard')
            g.add_argument('--all', action='store_true')
        if name != 'audit':
            p.add_argument('--exl3-experts', metavar='DIR',
                           help='source the routed experts from this EXL3 checkpoint (L245)')
            p.add_argument('--exl3-layers', metavar='SPEC',
                           help='which blk layers take EXL3 experts, e.g. 5,18-22 (default: all present)')
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())

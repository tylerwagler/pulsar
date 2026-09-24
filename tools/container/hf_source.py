#!/usr/bin/env python3
"""The builder's sources: an HF safetensors checkpoint read by shard header (no
torch, no dtype interpretation -- `raw` hands back the stored bytes), and an
exllamav3 EXL3 checkpoint whose routed experts become pulsar's `exl3m_*` slices.

Both are header-only until a tensor's bytes are asked for, so planning a
200 GB artifact costs seconds and streams the bytes once at emit time.
"""
import glob
import json
import os
import struct


class HFCheckpoint:
    """`config` is the language-model config (V4.1 nests it under `text_config`;
    the top level keeps the vision config and the quantization recipe, which
    stay reachable as `config['top_level']` / `config['quantization_config']`)."""

    def __init__(self, hf_dir):
        self.dir = hf_dir
        top = json.load(open(os.path.join(hf_dir, 'config.json')))
        self.config = dict(top['text_config']) if 'text_config' in top else dict(top)
        self.config['quantization_config'] = top.get('quantization_config')
        self.config['image_token_id'] = top.get('image_token_id')
        self.config['top_level'] = top
        idx_path = os.path.join(hf_dir, 'model.safetensors.index.json')
        if os.path.exists(idx_path):
            self.weight_map = json.load(open(idx_path))['weight_map']
        else:
            # a partial download: index every shard present by header
            self.weight_map = {}
            for p in sorted(glob.glob(os.path.join(hf_dir, 'model-*.safetensors'))):
                for name in self._shard_header_of(p):
                    self.weight_map[name] = os.path.basename(p)
        self._hdr = {}
        gc_path = os.path.join(hf_dir, 'generation_config.json')
        self.generation_config = json.load(open(gc_path)) if os.path.exists(gc_path) else {}

    # -- headers ----------------------------------------------------------
    def _shard_header_of(self, path):
        with open(path, 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            hdr = json.loads(f.read(n))
        hdr.pop('__metadata__', None)
        return hdr

    def _shard_header(self, shard_file):
        if shard_file not in self._hdr:
            self._hdr[shard_file] = self._shard_header_of(os.path.join(self.dir, shard_file))
        return self._hdr[shard_file]

    def names(self):
        return sorted(self.weight_map)

    def has(self, name):
        return name in self.weight_map

    def shape(self, name):
        """Real on-disk shape, row-major ([out, in, ...])."""
        return self._shard_header(self.weight_map[name])[name]['shape']

    def dtype(self, name):
        """The CONTAINER dtype ('BF16', 'F8_E4M3', 'F8_E8M0', 'I8', 'I16', 'F16', 'I32', 'F32'),
        not the logical format: the routed experts are FP4 declared by config.json
        but ride in an I8 container two per byte beside an F8_E8M0 scale."""
        return self._shard_header(self.weight_map[name])[name]['dtype']

    def span(self, name):
        """(absolute path, absolute byte offset, byte count) -- a copy range."""
        shard = self.weight_map[name]
        h = self._shard_header(shard)
        start, end = h[name]['data_offsets']
        path = os.path.join(self.dir, shard)
        with open(path, 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
        return path, 8 + n + start, end - start

    def raw(self, name):
        """The tensor's bytes exactly as stored."""
        path, off, n = self.span(name)
        with open(path, 'rb') as f:
            f.seek(off)
            buf = f.read(n)
        if len(buf) != n:
            raise SystemExit(f'{name}: short read from {path}')
        return buf


# ---------------------------------------------------------------------------
# The EXL3 expert source (L245): exllamav3's HF shards, read by header.  Mirrors
# src/engine/exl3_trellis.h: words per 16x16 tile name the rate (16K, or 16K+8
# for the half-integer rates), and one expert-projection is [trellis | suh | svh]
# with trellis (in/16)(out/16) tiles x words x 2 B and the two fp16 scale
# vectors -- the engine refuses a stack whose declared expert_bytes disagrees
# with that model, so this and the header are checked against each other at
# every load.
# ---------------------------------------------------------------------------
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
        import re
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

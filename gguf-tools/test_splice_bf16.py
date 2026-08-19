#!/usr/bin/env python3
"""End-to-end test for splice_bf16_plain.py on a synthetic artifact.

Builds a small GGUF carrying REAL ds4 tensor names (so the tool maps them to
real source tensors and reads the real shards) plus a decoy tensor that is not
in BF16_GROUP, runs the splice, and checks four things:

  1. every targeted tensor is retyped F16 -> BF16
  2. its bytes equal the source, converted the same way the quantizer would
  3. the decoy is untouched -- same type, same bytes
  4. nothing outside the tensor data regions moved (the header is the same size
     and every data offset is unchanged)

(3) and (4) are the ones worth having. The splice writes at absolute offsets,
so an arithmetic slip there does not fail loudly -- it silently corrupts a
neighbouring tensor, which is exactly the failure a 92 GB run would hide.

Usage: test_splice_bf16.py [--hf /mnt/models/dsv4-flash-0731]
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_main_template import HFCheckpoint, ne_reversed  # noqa: E402
from gguf_hdr import scan  # noqa: E402
from splice_bf16_plain import bf16_bytes, hf_name_for  # noqa: E402

F16, BF16 = 1, 30
ALIGN = 32

# (ds4 name, is_target). The decoy must NOT be in BF16_GROUP. All targets are
# BF16-upstream group members -- the F32-narrowing arm was deleted along with
# the hc_*_fn fixtures that exercised it (those tensors moved to the F32
# group under the source-format policy, so the splicer rightly skips them).
# ffn_gate_inp is the only group member present in the checkpoint's top-level
# weight map (the compressor/indexer members live in the QAT sidecar), so the
# post-decoy ordering case uses layer 1's copy of it.
CASES = [
    ('blk.0.ffn_gate_inp.weight', True),   # BF16 source -> lossless copy
    ('blk.0.attn_norm.weight', False),     # decoy, stays put
    ('blk.1.ffn_gate_inp.weight', True),   # target AFTER the decoy
]


def _str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def build_gguf(path, entries):
    """entries: list of (name, dims, type, payload_bytes)."""
    hdr = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(entries)) + struct.pack('<Q', 1)
    hdr += _str('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', ALIGN)
    off, infos, blobs = 0, b'', []
    for name, dims, ttype, payload in entries:
        infos += _str(name) + struct.pack('<I', len(dims))
        infos += b''.join(struct.pack('<Q', d) for d in dims)
        infos += struct.pack('<I', ttype) + struct.pack('<Q', off)
        blobs.append(payload)
        n = len(payload)
        off += n + ((ALIGN - (n % ALIGN)) % ALIGN)
    body = hdr + infos
    pad = (ALIGN - (len(body) % ALIGN)) % ALIGN
    with open(path, 'wb') as f:
        f.write(body + b'\0' * pad)
        for p in blobs:
            f.write(p)
            f.write(b'\0' * ((ALIGN - (len(p) % ALIGN)) % ALIGN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', default='/mnt/models/dsv4-flash-0731')
    a = ap.parse_args()
    ckpt = HFCheckpoint(a.hf)

    entries, expect = [], {}
    for name, is_target in CASES:
        hf = hf_name_for(name)
        if hf is None or not ckpt.has(hf):
            raise SystemExit(f'test setup: no source for {name} ({hf})')
        dims = ne_reversed(ckpt.shape(hf))
        ne = 1
        for d in dims:
            ne *= d
        # Seed the artifact with a recognisable pattern rather than the real
        # F16: the test asserts the target is OVERWRITTEN and the decoy is NOT,
        # and a distinctive filler makes both directions unambiguous.
        payload = (b'\xAB\xCD' * ne)
        entries.append((name, dims, F16, payload))
        expect[name] = dict(is_target=is_target, hf=hf, ne=ne, orig=payload,
                            src_dtype=ckpt.dtype(hf))

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'in.gguf')
        dst = os.path.join(td, 'out.gguf')
        build_gguf(src, entries)
        before, _, dstart_before = scan(src)

        r = subprocess.run([sys.executable, os.path.join(HERE, 'splice_bf16_plain.py'),
                            src, dst, '--hf', a.hf], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit('splice failed')

        after, _, dstart_after = scan(dst)
        bmap = {t['name']: t for t in before}
        amap = {t['name']: t for t in after}
        fails = []

        if dstart_before != dstart_after:
            fails.append(f'data_start moved {dstart_before} -> {dstart_after}')
        if os.path.getsize(src) != os.path.getsize(dst):
            fails.append('file size changed')

        with open(dst, 'rb') as f:
            for name, e in expect.items():
                b, af = bmap[name], amap[name]
                if b['offset'] != af['offset']:
                    fails.append(f'{name}: offset moved {b["offset"]} -> {af["offset"]}')
                f.seek(dstart_after + af['offset'])
                got = f.read(e['ne'] * 2)
                if e['is_target']:
                    if af['type'] != BF16:
                        fails.append(f'{name}: type {af["type"]}, expected BF16')
                    want = bf16_bytes(ckpt, e['hf'], e['src_dtype'])
                    if got != want:
                        fails.append(f'{name}: data != source-converted')
                    if got == e['orig']:
                        fails.append(f'{name}: still the filler, not spliced')
                else:
                    if af['type'] != F16:
                        fails.append(f'{name}: DECOY retyped to {af["type"]}')
                    if got != e['orig']:
                        fails.append(f'{name}: DECOY data was overwritten')

    for msg in fails:
        print('FAIL:', msg)
    if fails:
        raise SystemExit(1)
    ntgt = sum(1 for e in expect.values() if e['is_target'])
    print(f'PASS: {ntgt} tensors retyped + rewritten from source, '
          f'{len(expect) - ntgt} decoy untouched, all offsets and file size unchanged')


if __name__ == '__main__':
    main()

"""Check that a REAP-compacted artifact's ROUTER matches its own declaration.

WHY THIS EXISTS. Under ds4-compact-v1 the expert weight tensors are trimmed to
the survivors and densely renumbered, so slot j holds original expert
survivors[L][j]. The router (ffn_gate_inp) and its bias (exp_probs_b) stay
padded to the full n_expert, with survivor rows moved down to 0..keep-1 and the
tail neutralised:
    router : zero rows        (moved-down survivors, zeroed tail)
    bias   : -1e30 sentinel   (what actually makes a pruned slot unselectable)

If the router is written in ORIGINAL expert order instead, every routed token
fetches a different expert's weights than the router scored -- and NOTHING
FAILS. Every index is in range, no shape mismatch, no NaN, and the model still
emits fluent text. It just quietly loses most of what it knows.

That is not hypothetical: v5mx4-0731-srcfmt-v1 shipped that way on 40 of its 43
layers. The artifact declared reap.enabled=1, keep_count=192 and policy=2 --
everything needed to catch it -- and nothing checked the data against the
declaration. This does, from the artifact alone: no HF source, no survivor map,
no GPU, no numpy, a few seconds. Cheap enough to run before anything loads
76 GiB of weights.

Usage:  python3 audit_reap_router.py MODEL [MODEL2 ...]
        MODEL may be a GGUF file or a pulsar safetensors checkpoint DIRECTORY.
Exit 0 if every artifact passes, 1 otherwise.

CONTAINERS.  The tensors this audit reads are GGUF-layout: it treats the router
as raw f32/f16/bf16 rows at a file offset.  A pulsar safetensors checkpoint
stores the router in one of the engine's declared layouts (mxfp8_lt here), so
those reads do not apply.  The DECLARATION is container-independent, and a
checkpoint that does not declare reap.enabled was never compacted, so there is
nothing to check and the audit says so.  A safetensors checkpoint that DOES
declare reap.enabled is REFUSED: this script has no layout-aware reader to
certify it, and passing it unseen is the exact failure this file exists to
prevent.
"""
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..'), '/tmp'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gguf_hdr import scan     # noqa: E402

SENTINEL_MAX = -1e29          # anything at or below this counts as the -1e30 mask
F32, F16, BF16 = 0, 1, 30


def _read_kvs(path, want):
    """Minimal scalar/array KV reader -- gguf_hdr.scan deliberately skips values."""
    out = {}
    sc = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4),
          5: ('<i', 4), 6: ('<f', 4), 7: ('<B', 1), 10: ('<Q', 8), 11: ('<q', 8),
          12: ('<d', 8)}

    def rstr(f):
        (n,) = struct.unpack('<Q', f.read(8))
        return f.read(n).decode('utf-8', 'replace')

    def val(f, t):
        if t in sc:
            fmt, n = sc[t]
            return struct.unpack(fmt, f.read(n))[0]
        if t == 8:
            return rstr(f)
        if t == 9:
            (et,) = struct.unpack('<I', f.read(4))
            (n,) = struct.unpack('<Q', f.read(8))
            return [val(f, et) for _ in range(n)]
        raise SystemExit('unknown GGUF value type %d' % t)

    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit('%s: not a GGUF' % path)
        struct.unpack('<I', f.read(4))
        struct.unpack('<Q', f.read(8))
        (n_kv,) = struct.unpack('<Q', f.read(8))
        for _ in range(n_kv):
            k = rstr(f)
            (vt,) = struct.unpack('<I', f.read(4))
            v = val(f, vt)
            if k in want:
                out[k] = v
    return out


def _zero_rows(path, t, data_start):
    """Which router rows are all-zero.

    Tested on the RAW BYTES rather than decoded floats: the padded tail is
    written by a calloc'd buffer, so it is +0.0 in every storage type, and an
    all-zero byte run is exactly that in F16, BF16 and F32 alike. That keeps
    this dependency-free and makes it storage-agnostic -- it does not need to
    know how the router happens to be quantised this month.
    """
    ncols, nrows = t['dims'][0], t['dims'][1]
    width = 4 if t['type'] == F32 else 2
    if t['type'] not in (F32, F16, BF16):
        raise SystemExit('router type %d is not a plain float type' % t['type'])
    row_bytes = ncols * width
    zeros = bytes(row_bytes)
    out = []
    with open(path, 'rb') as f:
        f.seek(t['offset'] + data_start)
        for _ in range(nrows):
            out.append(f.read(row_bytes) == zeros)
    return out


def _f32_vec(path, t, data_start):
    n = 1
    for d in t['dims']:
        n *= d
    with open(path, 'rb') as f:
        f.seek(t['offset'] + data_start)
        return struct.unpack('<%df' % n, f.read(n * 4))


REAP_KVS = ('reap.enabled', 'reap.layer.keep_count',
            'reap.layer.expert_count', 'reap.layer.policy')


def _safetensors_kvs(path, want):
    """The declared keys of a pulsar safetensors checkpoint.

    Read from the SAME place the engine reads them -- the `pulsar.kv` blob (a
    GGUF-typed KV array) in the primary shard's __metadata__ -- rather than
    inventing a second metadata source that could drift from it.  Loudly dies if
    no shard carries one, because "this is not a checkpoint" and "this
    checkpoint declares nothing" must not look alike.
    """
    import glob
    for p in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        with open(p, 'rb') as f:
            head = f.read(8)
            if len(head) < 8:
                continue
            (n,) = struct.unpack('<Q', head)
            hdr = json.loads(f.read(n))
        blob = (hdr.get('__metadata__') or {}).get('pulsar.kv')
        if not blob:
            continue
        out = {}
        for e in json.loads(blob):
            if e.get('key') in want:
                out[e['key']] = e.get('value')
        return out
    raise SystemExit('%s: no shard carries pulsar.kv -- not a pulsar checkpoint' % path)


def audit(path):
    name = os.path.basename(path.rstrip('/'))
    if os.path.isdir(path):
        kv = _safetensors_kvs(path, REAP_KVS)
        if not kv.get('reap.enabled'):
            print('%s: reap not enabled -- nothing to check' % name)
            return True
        print('%s: FAIL -- this checkpoint declares reap.enabled, but this audit '
              'reads GGUF-layout tensors and the safetensors router is stored in a '
              'declared layout. It cannot certify this artifact, so it refuses '
              'rather than passing it unseen.' % name)
        return False
    kv = _read_kvs(path, set(REAP_KVS))
    if not kv.get('reap.enabled'):
        print('%s: reap not enabled -- nothing to check' % name)
        return True
    keep = kv['reap.layer.keep_count']
    policy = kv['reap.layer.policy']
    ts, _align, ds = scan(path)
    by = {t['name']: t for t in ts}

    bad = []
    checked = 0
    for L, (k, pol) in enumerate(zip(keep, policy)):
        rname = 'blk.%d.ffn_gate_inp.weight' % L
        if rname not in by:
            continue
        t = by[rname]
        n_expert = t['dims'][1]
        if pol != 2 or k >= n_expert:
            continue                      # unpruned layer: no padding expected
        checked += 1
        zero = _zero_rows(path, t, ds)
        n_tail = sum(1 for z in zero[k:] if z)
        n_head = sum(1 for z in zero[:k] if z)
        if n_tail != n_expert - k or n_head != 0:
            bad.append('layer %2d: expected %d zero rows at %d..%d, found %d '
                       '(%d zero rows inside 0..%d) -- router is very likely in '
                       'SOURCE expert order, not survivor order'
                       % (L, n_expert - k, k, n_expert - 1, n_tail, n_head, k - 1))
        bname = 'blk.%d.exp_probs_b.bias' % L
        if bname in by:
            b = _f32_vec(path, by[bname], ds)
            worst = max(b[k:]) if len(b) > k else None
            if worst is not None and worst > SENTINEL_MAX:
                bad.append('layer %2d: exp_probs_b tail is not the -1e30 sentinel '
                           '(max %.3g) -- pruned slots are selectable' % (L, worst))

    if bad:
        print('%s: FAIL (%d findings over %d pruned layers)' % (name, len(bad), checked))
        for m in bad[:8]:
            print('   ' + m)
        if len(bad) > 8:
            print('   ... and %d more' % (len(bad) - 8))
        return False
    print('%s: OK (%d pruned layers checked)' % (name, checked))
    return True


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(0 if all([audit(p) for p in args]) else 1)

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

Usage:  python3 audit_reap_router.py MODEL_DIR [MODEL_DIR2 ...]
        MODEL_DIR is a pulsar safetensors checkpoint DIRECTORY.
Exit 0 if every artifact passes, 1 otherwise.

The row-level check below used to read raw f32/f16/bf16 router rows out of a
GGUF; that tooling was archived 2026-09-24 (tag archive/gguf-tooling-2026-09-24,
L247).  A pulsar checkpoint stores the router in one of the engine's declared
layouts (mxfp8_lt here), which this script has no reader for.  So: a checkpoint
that does not declare reap.enabled was never compacted and passes with a note;
one that DOES declare it is REFUSED -- certifying it needs a layout-aware
reader, and passing it unseen is the exact failure this file exists to prevent.
"""
import json
import os
import struct
import sys



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
    print('%s: not a directory -- this audit reads pulsar checkpoints only (GGUF tooling archived, L247)' % name)
    return False


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(0 if all([audit(p) for p in args]) else 1)

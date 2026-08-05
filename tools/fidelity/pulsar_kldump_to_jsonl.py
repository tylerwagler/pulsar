#!/usr/bin/env python3
"""Convert a pulsar --kl-ref-dump (DS4KLRF1 full-vocab logits) plus its
.tokens.json sidecar into the ds4-fidelity-jsonl-v1 common format, for
cross-rig comparison with a vLLM capture via fidelity_compare.py.

(Exact full-vocab KL between two PULSAR variants should use the engine's own
--kl-score instead; this converter exists for the truncated cross-rig lane.)

Usage: pulsar_kldump_to_jsonl.py DUMP.klrf --name long-mixed --out OUT.jsonl \
         [--topk 20] [--tag pulsar-v5mx4]
"""
import argparse
import json
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--name", required=True, help="suite document name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tag", default="pulsar")
    a = ap.parse_args()

    tokens = json.load(open(a.dump + ".tokens.json"))
    with open(a.dump, "rb") as f:
        magic = f.read(8)
        if magic != b"DS4KLRF1":
            raise SystemExit(f"{a.dump}: not a DS4KLRF1 dump")
        vocab, stride, prefix_len, n_pos = struct.unpack("<4I", f.read(16))
        with open(a.out, "w") as out:
            out.write(json.dumps({
                "format": "ds4-fidelity-jsonl-v1",
                "source": a.tag,
                "model": a.tag,
                "file": a.name,
                "topk": a.topk,
                "tokens": tokens,
            }) + "\n")
            for m in range(n_pos):
                logits = np.frombuffer(f.read(4 * vocab), dtype=np.float32)
                if logits.size != vocab:
                    raise SystemExit(f"truncated dump at position {m}")
                x = logits.astype(np.float64)
                lse = np.logaddexp.reduce(x)
                pos = prefix_len + m * stride
                tok = tokens[pos] if pos < len(tokens) else None
                idx = np.argpartition(-x, a.topk)[:a.topk]
                idx = idx[np.argsort(-x[idx])]
                topk = [[int(t), round(float(x[t] - lse), 6)] for t in idx]
                lp = round(float(x[tok] - lse), 6) if tok is not None else None
                out.write(json.dumps({
                    "pos": pos,
                    "token": tok,
                    "logprob": lp,
                    "topk": topk,
                }) + "\n")
    print(f"wrote {a.out}: {n_pos} positions (stride {stride}, "
          f"prefix {prefix_len}, vocab {vocab})")


if __name__ == "__main__":
    main()

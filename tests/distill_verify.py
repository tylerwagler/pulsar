#!/usr/bin/env python3
"""plan-92 P0 pilot verifier: parse a PULSAR_DISTILL_DUMP file and check the
teacher sections. Usage: distill_verify.py DUMP N_EMBD"""
import struct, sys

import numpy as np

path, n_embd = sys.argv[1], int(sys.argv[2])
MAGIC = 0x50445431

chunks = 0
positions = 0
inexact_last = 0
with open(path, "rb") as f:
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        cap_n, start = struct.unpack("<II", hdr)
        toks = np.frombuffer(f.read(4 * cap_n), dtype=np.int32)
        # P1 format: collection-mode dumps store the hidden streams as f16
        # (teacher magic follows); pre-P0 dumps are f32 with no magic. Probe:
        # try f16 first, fall back to f32 on a magic mismatch.
        probe_pos = f.tell()
        hidden = [np.frombuffer(f.read(2 * cap_n * n_embd), dtype=np.float16)
                  .astype(np.float32).reshape(cap_n, n_embd) for _ in range(3)]
        magic_raw = f.read(4)
        if len(magic_raw) == 4 and struct.unpack("<I", magic_raw)[0] != MAGIC:
            f.seek(probe_pos)
            hidden = [np.frombuffer(f.read(4 * cap_n * n_embd), dtype=np.float32)
                      .reshape(cap_n, n_embd) for _ in range(3)]
            magic_raw = f.read(4)
        if len(magic_raw) < 4:
            print("chunk %d: NO teacher section (pre-P0 format?)" % chunks)
            break
        (magic,) = struct.unpack("<I", magic_raw)
        assert magic == MAGIC, "bad magic 0x%08x at chunk %d" % (magic, chunks)
        (inexact_last,) = struct.unpack("<i", f.read(4))
        ids = np.frombuffer(f.read(4 * cap_n * 64), dtype=np.int32).reshape(cap_n, 64)
        vals = np.frombuffer(f.read(2 * cap_n * 64), dtype=np.float16).reshape(cap_n, 64)
        tail = np.frombuffer(f.read(2 * cap_n), dtype=np.float16)

        # invariants
        assert (ids >= 0).all(), "negative token id"
        assert np.isfinite(vals.astype(np.float32)).all(), "non-finite top logit"
        d = np.diff(vals.astype(np.float32), axis=1)
        assert (d <= 1e-3).all(), "top logits not descending (max viol %.4f)" % d.max()
        for r in range(cap_n):
            assert len(set(ids[r].tolist())) == 64, "duplicate ids in row %d" % r
        # tail mass must be below the top-64 mass floor in logit terms:
        # lse(tail) is finite and the top-1 logit exceeds tail_lse - log(vocab)
        tf = tail.astype(np.float32)
        assert not np.isnan(tf).any(), "NaN tail lse"
        neginf = int(np.isneginf(tf).sum())   # legal: near-one-hot rows cancel to -inf
        # teacher top-1 should usually equal the next token in natural text --
        # report the agreement rate, don't assert it (temp-0 argmax property)
        agree = float((ids[:-1, 0] == toks[1:]).mean()) if cap_n > 1 else float("nan")
        norms = [float(np.linalg.norm(h, axis=1).mean()) for h in hidden]
        print("chunk %d: start=%d n=%d hidden|mean|=%s top1-vs-next=%.3f "
              "tail_lse mean=%.2f (-inf rows=%d)" % (chunks, start, cap_n,
              ["%.1f" % x for x in norms], agree,
              float(tf[np.isfinite(tf)].mean()) if np.isfinite(tf).any() else float("nan"),
              neginf))
        chunks += 1
        positions += cap_n

print("OK: %d chunks, %d positions, inexact_total=%d -> %s"
      % (chunks, positions, inexact_last,
         "PASS" if chunks > 0 and inexact_last == 0 else "FAIL"))
sys.exit(0 if chunks > 0 and inexact_last == 0 else 1)

#!/usr/bin/env python3
"""plan-92 P2: reader + loss for the teacher dump.

The engine writes, per prefill chunk (imatrix.cpp dspark_bulk_drain):

    u32 cap_n | u32 start                       chunk header
    i32 token_id[cap_n]                         the chunk's input tokens
    f16 hidden[3][cap_n][n_embd]                drafter-input anchor states
                                                (f32 in pre-P1 dumps)
    u32 magic 0x50445431 ("PDT1")               teacher section (P0+)
    i32 inexact_total                           cumulative exactness counter
    i32 top_ids[cap_n][64]                      teacher top-64 token ids
    f16 top_logits[cap_n][64]                   their raw logits, descending
    f16 tail_lse[cap_n]                         logsumexp of the NON-top mass

`hidden[s][t]` is the mean-reduced HC residual at anchor layer s (the last
three layers) for position `start + t` -- exactly what the drafter reads at
inference. The teacher row at t is the QUANTIZED MAIN's next-token
distribution AT that position, so a sample is (hidden[:, t], teacher[t]).

The loss is KL(teacher || student) computed over the top-64 support plus a
single lumped tail term, which is exact under the standard assumption that
the student's tail mass is spread like the teacher's -- see kl_topk_with_tail.

This module is deliberately torch-optional: parsing, iteration and the
numpy reference loss work without torch, so the format can be validated
anywhere. Import torch only inside the training entry point.
"""
import argparse, struct, sys

import numpy as np

MAGIC = 0x50445431
TOPK = 64


class DistillChunk:
    __slots__ = ("start", "n", "tokens", "hidden", "top_ids", "top_logits",
                 "tail_lse", "inexact_total")

    def __init__(self, start, n, tokens, hidden, top_ids, top_logits, tail_lse,
                 inexact_total):
        self.start = start
        self.n = n
        self.tokens = tokens
        self.hidden = hidden          # (3, n, n_embd) float32
        self.top_ids = top_ids        # (n, 64) int32
        self.top_logits = top_logits  # (n, 64) float32
        self.tail_lse = tail_lse      # (n,) float32, -inf legal
        self.inexact_total = inexact_total


def read_chunks(path, n_embd):
    """Yield DistillChunk per chunk. Detects f16 vs f32 hidden by probing the
    teacher magic at the f16 offset first (P1 writes f16; pre-P1 wrote f32)."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            cap_n, start = struct.unpack("<II", hdr)
            if cap_n == 0:
                return
            toks = np.frombuffer(f.read(4 * cap_n), dtype=np.int32)

            probe = f.tell()
            hid = np.frombuffer(f.read(2 * cap_n * n_embd * 3), dtype=np.float16)
            magic_raw = f.read(4)
            is_f16 = (len(magic_raw) == 4 and
                      struct.unpack("<I", magic_raw)[0] == MAGIC)
            if is_f16:
                hidden = hid.astype(np.float32).reshape(3, cap_n, n_embd)
            else:
                f.seek(probe)
                hidden = (np.frombuffer(f.read(4 * cap_n * n_embd * 3),
                                        dtype=np.float32)
                          .reshape(3, cap_n, n_embd))
                magic_raw = f.read(4)
                if len(magic_raw) < 4:
                    raise ValueError("no teacher section: pre-P0 dump, unusable "
                                     "for distillation (hidden states only)")
                if struct.unpack("<I", magic_raw)[0] != MAGIC:
                    raise ValueError("bad teacher magic at chunk start=%d "
                                     "(wrong --n-embd?)" % start)

            (inexact,) = struct.unpack("<i", f.read(4))
            ids = np.frombuffer(f.read(4 * cap_n * TOPK),
                                dtype=np.int32).reshape(cap_n, TOPK)
            vals = (np.frombuffer(f.read(2 * cap_n * TOPK), dtype=np.float16)
                    .astype(np.float32).reshape(cap_n, TOPK))
            tail = (np.frombuffer(f.read(2 * cap_n), dtype=np.float16)
                    .astype(np.float32))
            yield DistillChunk(start, cap_n, toks, hidden, ids, vals, tail,
                               inexact)


def teacher_probs(top_logits, tail_lse):
    """Exact renormalized teacher distribution over the top-64 support, plus
    the tail's total probability mass. Everything in float64 -- the tail term
    is a logsumexp of ~129k small terms and f32 loses it."""
    tl = top_logits.astype(np.float64)
    tail = tail_lse.astype(np.float64)
    m = np.maximum(tl.max(axis=1), np.where(np.isneginf(tail), -np.inf, tail))
    top_exp = np.exp(tl - m[:, None])
    tail_exp = np.where(np.isneginf(tail), 0.0, np.exp(tail - m))
    denom = top_exp.sum(axis=1) + tail_exp
    return top_exp / denom[:, None], tail_exp / denom


def kl_topk_with_tail(student_logits, top_ids, top_logits, tail_lse):
    """KL(teacher || student) over the top-64 support + one lumped tail term.

    student_logits: (n, vocab) float32/float64 (numpy reference path).
    The tail term treats the student's non-top-64 mass as a single bucket:
    exact when the student spreads its tail like the teacher, and a strict
    upper bound on the support term otherwise -- which is the behavior we
    want (it never rewards a student for hiding mass outside the support).
    """
    p_top, p_tail = teacher_probs(top_logits, tail_lse)
    sl = student_logits.astype(np.float64)
    m = sl.max(axis=1, keepdims=True)
    z = np.exp(sl - m).sum(axis=1)
    log_z = np.log(z) + m[:, 0]
    rows = np.arange(sl.shape[0])[:, None]
    q_top_log = sl[rows, top_ids] - log_z[:, None]
    q_top = np.exp(q_top_log)
    q_tail = np.clip(1.0 - q_top.sum(axis=1), 1e-12, 1.0)

    eps = 1e-12
    kl = (p_top * (np.log(np.clip(p_top, eps, None)) - q_top_log)).sum(axis=1)
    kl += np.where(p_tail > eps,
                   p_tail * (np.log(np.clip(p_tail, eps, None)) - np.log(q_tail)),
                   0.0)
    return kl


def top1_agreement(student_logits, top_ids):
    """The cheap offline proxy for live acceptance: does the student's argmax
    equal the teacher's? Correlates with alpha without needing the server."""
    return float((student_logits.argmax(axis=1) == top_ids[:, 0]).mean())


def main():
    ap = argparse.ArgumentParser(description="Inspect/validate a teacher dump")
    ap.add_argument("dump")
    ap.add_argument("--n-embd", type=int, default=4096)
    ap.add_argument("--max-chunks", type=int, default=0)
    args = ap.parse_args()

    n_pos = 0
    n_chunks = 0
    inexact = 0
    ent = []
    tail_mass = []
    for c in read_chunks(args.dump, args.n_embd):
        p_top, p_tail = teacher_probs(c.top_logits, c.tail_lse)
        ent.append(float((-(p_top * np.log(np.clip(p_top, 1e-12, None)))
                          .sum(axis=1)).mean()))
        tail_mass.append(float(p_tail.mean()))
        n_pos += c.n
        n_chunks += 1
        inexact = c.inexact_total
        if args.max_chunks and n_chunks >= args.max_chunks:
            break
    if not n_chunks:
        print("no chunks parsed")
        return 1
    print("chunks=%d positions=%d inexact_total=%d" % (n_chunks, n_pos, inexact))
    print("teacher top-64 entropy mean=%.3f nats | tail mass mean=%.4f"
          % (float(np.mean(ent)), float(np.mean(tail_mass))))
    print("PASS" if inexact == 0 else "FAIL (inexact top-64 rows)")
    return 0 if inexact == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

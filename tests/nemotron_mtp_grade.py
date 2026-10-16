#!/usr/bin/env python3
"""The Nemotron-H MTP drafter's first-step grade on the BF16 mini (L296 N9; rule 7).

  nemotron_mtp_grade.py prep  REF_PT OUT_DIR
      OUT_DIR/ids.bin: the reference capture's prompt as int32 ids (tests/mtp_spec_gate probe's input)
  nemotron_mtp_grade.py grade MTP_REF_PT CAP_DIR [--bf16 MTP_REF_BF16_PT] [--trunk REF_PT]
      grades what `mtp_spec_gate probe` left in CAP_DIR (mtp_decode.f32, mtp_prefill.f32: the module's logits at every
      position, row p = (the trunk at p, token p + 1), teacher-forced by the engine's own trunk; mtp_ids.i32 when the
      drafter's head is a draft head over a subset of the vocabulary -- then every distribution is over it) against the fp32
      reference module (tools/nemotron/capture_mtp_ref.py on capture_ref.py's fp32 trunk): rel-fro, KL(ref || ours),
      top-1 agreement.  --bf16: transformers' own bf16 module on its own bf16 trunk, graded the same way -- the bf16
      rounding floor the engine is read against.  --trunk: the fp32 trunk capture (capture_ref.py), for what a draft is
      FOR: how often the module's first draft is the trunk's own next token (row p's argmax vs the trunk's argmax at
      p + 1) -- the teacher-forced draft-1 acceptance on this fixture, the reference's and ours.

PASS: on both engine paths, KL mean <= 2 x the bf16 floor's and top-1 within 2 points of it (the floor graded the same
way); decode vs prefill reported.  Needs torch (~/inkling/.venv on Sparky).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def rel_fro(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def kl_top1(ref, ours):
    """KL(ref || ours) mean / max over rows and top-1 agreement, in row chunks (the rows are 131072 wide)"""
    kls, top = [], []
    for i in range(0, ref.shape[0], 64):
        p, q = ref[i:i + 64].double(), ours[i:i + 64].double()
        lp, lq = F.log_softmax(p, -1), F.log_softmax(q, -1)
        kls.append((lp.exp() * (lp - lq)).sum(-1))
        top.append(p.argmax(-1) == q.argmax(-1))
    kl = torch.cat(kls)
    return kl.mean().item(), kl.max().item(), torch.cat(top).double().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "grade"])
    ap.add_argument("ref")
    ap.add_argument("dir")
    ap.add_argument("--bf16")
    ap.add_argument("--trunk")
    a = ap.parse_args()
    ref = torch.load(a.ref, weights_only=False)
    if a.mode == "prep":
        os.makedirs(a.dir, exist_ok=True)
        ref["input_ids"][0].to(torch.int32).numpy().tofile(os.path.join(a.dir, "ids.bin"))
        print(f"wrote {a.dir}/ids.bin ({ref['input_ids'].shape[1]} ids)")
        return 0
    R = ref["logits"][0].float()   # [T - 1, V]
    # the drafter's head scores a subset of the vocabulary (L296 draft-cost: mtp_ids.i32, the draft head's ids): the
    # grade is of what the drafter computes -- every distribution over those ids (the references' logits restricted to
    # them, renormalised by the softmax), a pick mapped back to its id
    ids_path = os.path.join(a.dir, "mtp_ids.i32")
    ids = torch.from_numpy(np.fromfile(ids_path, dtype=np.int32)).long() if os.path.exists(ids_path) else None
    if ids is not None:
        R = R[:, ids]
        print(f"the drafter's head: {ids.numel()} of {ref['logits'].shape[-1]} ids (the draft vocabulary); every "
              f"distribution below is over them")
    V = R.shape[1]
    rows = {}
    for name in ("decode", "prefill"):
        path = os.path.join(a.dir, f"mtp_{name}.f32")
        x = torch.from_numpy(np.fromfile(path, dtype=np.float32))
        if x.numel() % V:
            raise SystemExit(f"{path}: {x.numel()} floats is not whole rows of {V}")
        rows[name] = x.view(-1, V)
    n = min(r.shape[0] for r in rows.values())
    R = R[:n]
    print(f"MTP first step, {n} positions (row p: the module on (trunk p, token p + 1)); reference fp32 "
          f"({ref['meta']['transformers']})")
    res = {}
    for name, x in rows.items():
        x = x[:n]
        km, kx, t1 = kl_top1(R, x)
        res[name] = (km, t1)
        print(f"  ours {name:7s} vs fp32: rel-fro {rel_fro(x, R):.3e}  KL mean {km:.3e} max {kx:.3e}  top-1 {100 * t1:.2f}%")
    km, kx, t1 = kl_top1(rows["prefill"][:n], rows["decode"][:n])
    print(f"  ours decode vs prefill:  rel-fro {rel_fro(rows['decode'][:n], rows['prefill'][:n]):.3e}  KL mean {km:.3e} "
          f"max {kx:.3e}  top-1 {100 * t1:.2f}%")
    ok = True
    if a.bf16:
        B = torch.load(a.bf16, weights_only=False)["logits"][0].float()[:n]
        if ids is not None:
            B = B[:, ids]
        fkm, fkx, ft1 = kl_top1(R, B)
        print(f"  tf bf16 (floor) vs fp32: rel-fro {rel_fro(B, R):.3e}  KL mean {fkm:.3e} max {fkx:.3e}  "
              f"top-1 {100 * ft1:.2f}%")
        for name, x in rows.items():
            km, kx, t1 = kl_top1(B, x[:n])
            print(f"  ours {name:7s} vs tf bf16: KL mean {km:.3e} max {kx:.3e}  top-1 {100 * t1:.2f}%")
            good = res[name][0] <= 2.0 * fkm and res[name][1] >= ft1 - 0.02
            ok = ok and good
            print(f"  {'ok  ' if good else 'FAIL'}  {name}: KL {res[name][0]:.3e} <= 2 x floor {2 * fkm:.3e}, top-1 "
                  f"{100 * res[name][1]:.2f}% >= floor - 2 = {100 * ft1 - 2:.2f}%")
    if a.trunk:
        T = torch.load(a.trunk, weights_only=False)
        tl = T["logits_dt0"][0]   # [T, V]: the trunk's next-token logits at every position
        nxt = tl[1:n + 1].argmax(-1)   # the trunk's greedy token at p + 2 (its logits at p + 1)
        tok = T["input_ids"][0]

        def pick(x):   # a row's draft: its argmax, as a vocabulary id
            p = x.argmax(-1)
            return ids[p] if ids is not None else p
        print(f"  draft-1 acceptance proxy (teacher-forced): the module's argmax at p == the trunk's argmax at p + 1:")
        print(f"    reference fp32 {100 * (pick(R) == nxt).double().mean().item():.2f}%   "
              + "   ".join(f"ours {k} {100 * (pick(v[:n]) == nxt).double().mean().item():.2f}%" for k, v in rows.items()))
        m = min(n, tok.shape[0] - 2)   # rows whose token p + 2 is in the prompt
        print(f"    (the actual token p + 2: reference {100 * (pick(R[:m]) == tok[2:m + 2]).double().mean().item():.2f}%, "
              f"the trunk's own top-1 on it {100 * (nxt[:m] == tok[2:m + 2]).double().mean().item():.2f}%)")
    print("MTP GRADE PASS" if ok else "MTP GRADE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

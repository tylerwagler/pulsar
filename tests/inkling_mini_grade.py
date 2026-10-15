#!/usr/bin/env python3
"""The Inkling forward's layer grade on the 3-layer mini (L294 / L295; rule 7).

  inkling_mini_grade.py prep  REF_PT OUT_DIR
      OUT_DIR/ids.bin: the reference capture's prompt as int32 ids (tests/inkling_fwd_gate's input)
  inkling_mini_grade.py grade REF_PT MINI_DIR CAP_DIR [--exl3 EXL3_PT]
      grades what `inkling_fwd_gate capture` left in CAP_DIR -- the residual stream after every layer (the driver's
      dump tap, prefix CAP_DIR/d: the prefill's one chunk, and every decode step's row) and the logits of the decode
      path at every position (decode_logits.f32) and of the prefill's last row (prefill_last.f32) -- against
        the transformers fp32 capture REF_PT (L294: input_ids, hidden_states[0..n] = the normed embedding, then each
        layer's output, the last one after the final norm; logits over the real vocabulary, /16 applied), and
        the exllamav3 forward of the same EXL3 pack (EXL3_PT, tests/inkling_exl3_capture.py): its layers' outputs and
        its final-normed rows, with the logits of the SAME bf16 head the container carries -- apples to apples.
      MINI_DIR is the bf16 mini checkpoint: the final norm's weight and eps (the engine's last dump is the residual
      before it; the grade applies it, rounding to bf16 as the head's input).

Metrics: rel-fro = ||ours - ref||_F / ||ref||_F; logits: mean / max KL(ref || ours) over the real vocabulary, top-1
agreement, rel-fro.  Needs torch + safetensors (the quant lane's ~/inkling/.venv on Sparky).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

E = 4096


def rel_fro(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def kl_top1(ref_logits, ours):
    lp, lq = F.log_softmax(ref_logits.double(), -1), F.log_softmax(ours.double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (ref_logits.argmax(-1) == ours.argmax(-1)).double().mean().item()
    return kl.mean().item(), kl.max().item(), top1


def bf16(x):
    return x.to(torch.bfloat16).float()


def final_norm(h, w, eps):
    """vLLM's rmsnorm on the bf16 residual: one rounding to bf16 (the head's input)"""
    h = h.float()
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return bf16(y)


def load_dumps(cap, T, n_layer):
    """per layer, the prefill's rows [T][E] (its chunks: every multi-row dump, a file of k rows at pos p holding
    positions p - k + 1 .. p) and the decode path's (the one-row dumps, one a position)"""
    pre, dec = [], []
    for il in range(n_layer):
        prows, drows = {}, {}
        for f in glob.glob(os.path.join(cap, f"d_inkling_h-{il}_pos*.bin")):
            p = int(re.search(r"_pos(\d+)(?:_c\d+)?\.bin$", f).group(1))
            a = np.fromfile(f, dtype=np.float32).reshape(-1, E)
            if a.shape[0] == 1:
                drows[p] = a[0]
            else:
                for i in range(a.shape[0]):
                    prows[p - a.shape[0] + 1 + i] = a[i]
        pre.append(torch.from_numpy(np.stack([prows[p] for p in range(T)])) if len(prows) == T else None)
        dec.append(torch.from_numpy(np.stack([drows[p] for p in range(T)])) if len(drows) == T else None)
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "grade"])
    ap.add_argument("ref")
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--exl3", default=None)
    args = ap.parse_args()
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    ids = ref["input_ids"][0].to(torch.int32)
    if args.mode == "prep":
        os.makedirs(args.a, exist_ok=True)
        ids.numpy().tofile(os.path.join(args.a, "ids.bin"))
        print(f"prep: {ids.numel()} ids -> {args.a}/ids.bin")
        return
    mini, cap = args.a, args.b
    cfg = json.load(open(os.path.join(mini, "config.json")))["text_config"]
    eps, n_layer = cfg["rms_norm_eps"], cfg["num_hidden_layers"]
    with safe_open(os.path.join(mini, "globals.safetensors"), "pt") as f:
        w_norm = f.get_tensor("model.llm.norm.weight")
    T = ids.numel()
    hs = [h[0].float() for h in ref["hidden_states"]]
    logits_ref = ref["logits"][0].float()
    nv = logits_ref.shape[-1]
    pre, dec = load_dumps(cap, T, n_layer)
    ex = torch.load(args.exl3, map_location="cpu", weights_only=False) if args.exl3 else None

    print(f"grade: {T} tokens, {n_layer} layers, real vocab {nv}")
    print(f"{'':28s} {'vs BF16 ref (fp32)':>20s} {'vs exllamav3 K3':>18s} {'exllamav3 vs ref':>18s}")

    def row(name, ours, il_ref, ex_t):
        r = rel_fro(ours, il_ref) if ours is not None else float("nan")
        e = rel_fro(ours, ex_t) if (ours is not None and ex_t is not None) else float("nan")
        x = rel_fro(ex_t, il_ref) if ex_t is not None else float("nan")
        print(f"  {name:26s} {r:20.3e} {e:18.3e} {x:18.3e}")

    for il in range(n_layer):
        last = il == n_layer - 1
        ex_t = None
        if ex is not None:
            ex_t = ex["final"].float() if last else ex["hs"][il].float()
        for path, t in (("prefill", pre[il]), ("decode", dec[il])):
            if t is None:
                print(f"  layer {il} {path}: no dump")
                continue
            ours = final_norm(t, w_norm, eps) if last else t
            row(f"layer {il}{'+norm' if last else ''} ({path})", ours, hs[il + 1], ex_t)
    if pre[0] is not None and dec[0] is not None:
        for il in range(n_layer):
            print(f"  layer {il}: decode vs prefill rows rel-fro {rel_fro(dec[il], pre[il]):.3e}, "
                  f"bytes equal {bool(torch.equal(dec[il], pre[il]))}")

    W = os.path.getsize(os.path.join(cap, "decode_logits.f32")) // (4 * T)
    lg = torch.from_numpy(np.fromfile(os.path.join(cap, "decode_logits.f32"), dtype=np.float32).reshape(T, W))
    pad_ok = bool(torch.isinf(lg[:, nv:]).all()) if W > nv else True
    lg = lg[:, :nv]
    kl_m, kl_x, top1 = kl_top1(logits_ref, lg)
    print(f"  logits (decode path, all {T} positions) vs BF16 ref: rel-fro {rel_fro(lg, logits_ref):.3e}, KL mean "
          f"{kl_m:.4e} max {kl_x:.4e}, top-1 {100 * top1:.2f}%; padding -inf {pad_ok}")
    if ex is not None:
        el = ex["logits"].float()[:, :nv]
        kl_m, kl_x, top1 = kl_top1(el, lg)
        print(f"  logits vs exllamav3 K3 (bf16 head):           rel-fro {rel_fro(lg, el):.3e}, KL mean {kl_m:.4e} max "
              f"{kl_x:.4e}, top-1 {100 * top1:.2f}%")
        kl_m, kl_x, top1 = kl_top1(logits_ref, el)
        print(f"  exllamav3 K3 vs BF16 ref (the quantization gap): rel-fro {rel_fro(el, logits_ref):.3e}, KL mean "
              f"{kl_m:.4e} max {kl_x:.4e}, top-1 {100 * top1:.2f}%")
    pl = torch.from_numpy(np.fromfile(os.path.join(cap, "prefill_last.f32"), dtype=np.float32))[:nv]
    print(f"  prefill last row vs ref: rel-fro {rel_fro(pl, logits_ref[-1]):.3e}, argmax {pl.argmax().item()} / ref "
          f"{logits_ref[-1].argmax().item()}; vs the decode path's last row: rel-fro {rel_fro(pl, lg[-1]):.3e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""refkernels.py against independent statements of what the V4.1 reference kernels compute.

    python tools/exl3quant/test_refkernels.py [--ref /path/to/DeepSeek-V4.1-Flash]

With --ref, the e2m1 weight decode is also checked against the reference repo's own
convert.py (cast_e2m1fn_to_e4m3fn, an independent decode) on a real expert tensor.
"""

import argparse
import importlib.util
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refkernels as rk  # noqa: E402

FAILS = []


def check(cond, what):
    print(("PASS  " if cond else "FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def e2m1_rounding():
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    x = torch.tensor(grid + [-g for g in grid])
    check(torch.equal(rk._round_e2m1(x), x), "e2m1: grid points are fixed")
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])  # to the even code
    check(torch.equal(rk._round_e2m1(ties), want) and torch.equal(rk._round_e2m1(-ties), -want),
          "e2m1: midpoints round to the even code, sign-symmetric")
    near = torch.tensor([0.26, 0.74, 1.26, 1.74, 2.51, 3.49, 5.01])
    check(torch.equal(rk._round_e2m1(near), torch.tensor([0.5, 0.5, 1.5, 1.5, 3.0, 3.0, 6.0])),
          "e2m1: off-midpoint values round to nearest")


def act_quant():
    torch.manual_seed(0)
    x = (torch.randn(64, 256) * torch.logspace(-3, 3, 64)[:, None]).bfloat16()
    q, s = rk.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    sf = s.float()
    amax = x.float().unflatten(-1, (-1, 32)).abs().amax(-1).clamp_min(1e-4)
    log2 = torch.log2(sf)
    check(torch.equal(log2, log2.round()), "act_quant ue8m0: scales are powers of two")
    check(bool((amax / 448 <= sf).all() and (amax / 448 > sf / 2).all()),
          "act_quant ue8m0: the smallest power of two >= amax/448")
    y = x.clone()
    rk.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    deq = (q.float().unflatten(-1, (-1, 32)) * sf.unsqueeze(-1)).flatten(-2)
    check(torch.equal(y.float(), deq), "act_quant: in-place form = dequantized codes (exact in bf16)")
    rel = ((y.float() - x.float()).abs() / x.float().abs().clamp_min(1e-30)).median().item()
    check(rel < 0.07, f"act_quant: median relative error {rel:.4f} within e4m3's 3-bit mantissa")


def fp4_act_quant():
    torch.manual_seed(1)
    x = torch.randn(8, 512).bfloat16()
    y = x.clone()
    rk.fp4_act_quant(y, 16, True, scale_dtype=torch.float8_e4m3fn)
    blocks = y.float().unflatten(-1, (-1, 16))
    amax = x.float().unflatten(-1, (-1, 16)).abs().amax(-1, keepdim=True)
    s = (amax / 6).to(torch.float8_e4m3fn).float()
    codes = blocks / s
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    on_grid = (codes.abs().unsqueeze(-1) - grid).abs().amin(-1) < 1e-5
    check(bool(on_grid.all()), "fp4_act_quant e4m3/16: every value is an e2m1 code times the block's e4m3 scale")
    z = torch.zeros(2, 64).bfloat16()
    rk.fp4_act_quant(z, 32, True)
    check(bool((z == 0).all()), "fp4_act_quant e8m0/32: an all-zero block stays zero")


def gemms():
    torch.manual_seed(2)
    a = torch.randn(16, 256).bfloat16()
    q, s = rk.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    w = torch.randn(96, 256)
    wq, ws = rk.act_quant(w.bfloat16(), 32, "ue8m0", torch.float8_e8m0fnu)  # per-row scales stand in
    # fp8_gemm takes weight scales per (32-row block, 32-col block): use each block's max
    ws_blocks = ws.float().view(3, 32, 8).amax(1)
    wq2 = (wq.float().view(3, 32, 8, 32) * ws.float().view(3, 32, 8, 1) / ws_blocks.view(3, 1, 8, 1)).view(96, 256)
    wq2 = wq2.clamp(-448, 448).to(torch.float8_e4m3fn)
    out = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        c = rk.fp8_gemm(q, s, wq2, ws_blocks.to(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
    finally:
        torch.set_default_dtype(out)
    ad = (q.double().view(16, 8, 32) * s.double().view(16, 8, 1)).view(16, 256)
    wd = (wq2.double().view(3, 32, 8, 32) * ws_blocks.double().view(3, 1, 8, 1)).view(96, 256)
    ref = (ad @ wd.T).to(torch.bfloat16)
    ulps = ((c.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-6)).max().item()
    check(c.dtype == torch.bfloat16 and ulps < 2 ** -7, f"fp8_gemm: = fp64 product of the dequantized operands to bf16 rounding ({ulps:.2e})")


def sparse_attn():
    torch.manual_seed(3)
    b, m, h, d, n = 2, 12, 4, 64, 20
    q = torch.randn(b, m, h, d).bfloat16()
    kv = torch.randn(b, n, d).bfloat16()
    sink = torch.randn(h)
    idx = torch.full((b, m, 8), -1, dtype=torch.int32)
    for i in range(m):
        sel = torch.randperm(n)[: min(8, i)]  # row 0 empty
        idx[:, i, : len(sel)] = sel.int()
    o = rk.sparse_attn(q, kv, sink, idx, 0.125)
    ref = torch.zeros(b, m, h, d)
    for bi in range(b):
        for i in range(m):
            sel = idx[bi, i][idx[bi, i] >= 0].long()
            if len(sel) == 0:
                continue
            sc = (q[bi, i].double() @ kv[bi, sel].double().T) * 0.125  # [h, k]
            logits = torch.cat([sc, sink.double()[:, None]], dim=1)
            p = logits.softmax(-1)[:, :-1]
            ref[bi, i] = (p @ kv[bi, sel].double()).float()
    err = (o.float() - ref).abs().max().item()
    check(err < 2e-2, f"sparse_attn: = softmax with a sink over the gathered rows ({err:.2e})")
    check(bool((o[:, 0] == 0).all()), "sparse_attn: a row with no valid index is zero")


def sinkhorn():
    torch.manual_seed(4)
    mixes = torch.randn(3, 5, 24)
    pre, post, comb = rk.hc_split_sinkhorn(mixes, torch.rand(3), torch.randn(24), 4, 20, 1e-6)
    check(pre.shape == (3, 5, 4) and post.shape == (3, 5, 4) and comb.shape == (3, 5, 4, 4), "hc_split_sinkhorn: shapes")
    check(bool((post > 0).all() and (post < 2).all()), "hc_split_sinkhorn: post in (0, 2)")
    cols = comb.sum(-2)
    rows = comb.sum(-1)
    check(bool(((cols - 1).abs() < 1e-4).all() and ((rows - 1).abs() < 1e-3).all()),
          "hc_split_sinkhorn: comb doubly stochastic (column-normalized last)")


def fp4_vs_convert(ref_dir):
    spec = importlib.util.spec_from_file_location("v41_convert", os.path.join(ref_dir, "inference", "convert.py"))
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    from safetensors import safe_open
    wm = json.load(open(os.path.join(ref_dir, "model.safetensors.index.json")))["weight_map"]
    name = "layers.3.ffn.experts.17.w1"
    with safe_open(os.path.join(ref_dir, wm[name + ".weight"]), framework="pt") as f:
        w, s = f.get_tensor(name + ".weight"), f.get_tensor(name + ".scale")
    ours = rk.dequant_fp4(w.view(torch.float4_e2m1fn_x2), s)
    e4, s8 = conv.cast_e2m1fn_to_e4m3fn(w, s)  # the reference repo's lossless e2m1 -> e4m3 cast
    theirs = (e4.float().view(e4.size(0) // 32, 32, e4.size(1) // 32, 32) * s8.float()[:, None, :, None]).view_as(e4)
    check(torch.equal(ours, theirs.float()), f"dequant_fp4 = convert.py's e2m1 decode on {name} ({tuple(ours.shape)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="V4.1 snapshot (enables the convert.py cross-check)")
    a = ap.parse_args()
    e2m1_rounding()
    act_quant()
    fp4_act_quant()
    gemms()
    sparse_attn()
    sinkhorn()
    if a.ref:
        fp4_vs_convert(a.ref)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

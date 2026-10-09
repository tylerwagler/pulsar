#!/usr/bin/env python3
"""Grade one quantized block's routed experts against the FP4 source on HELD-OUT activations -- a sanity grade
of the quantizer's output, not the model grade (that is end-to-end KL / perplexity, L269 W5).

For every expert of the block and every K whose shard v41_stream.py wrote, the EXL3 weight is decoded by
exllamav3's own reconstruction (LinearEXL3.get_weight_tensor -- the decode pulsar's host header is graded
byte-exact against, tests/exl3_dequant_gate.cpp) and compared with the source weight (refkernels.dequant_fp4):

  rel_fro     ||W_q - W|| / ||W||                                   weight-space, no activations
  gate / up   ||X (W_q - W)^T|| / ||X W^T||, X = the fp8 GEMM input of the held-out tokens ROUTED to the expert
  down        the same over the expert's source mid activation silu(gate) * up of those tokens
  moe_out     || sum_k w_k E_k^q(x) - sum_k w_k E_k(x) || / || sum_k w_k E_k(x) ||   over every held-out token:
              the block's routed-expert output, every projection quantized, the routes and weights the source
              gate chose (the shared expert is not included; it stays MXFP8)

    python tools/exl3quant/grade_layer.py --ref $V41 --exllamav3 ~/exllamav3 --run $OUT --block layers.20
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refkernels  # noqa: E402
from v41_stream import Checkpoint, gemm_input  # noqa: E402


def stats(v):
    v = sorted(v)
    n = len(v)
    return {"median": v[n // 2], "mean": sum(v) / n, "p90": v[min(n - 1, int(0.9 * n))], "max": v[-1], "n": n}


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--exllamav3", required=True)
    ap.add_argument("--run", required=True, help="v41_stream.py's --out (exl3-kK/, holdout/)")
    ap.add_argument("--block", required=True, help="layers.L or mtp.S")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--swiglu-limit", type=float, default=None, help="default: config.json's")
    ap.add_argument("--experts", type=int, default=None,
                    help="DRY RUN ONLY: grade the first N experts (a partial block); moe_out then sums only their "
                         "contributions")
    ap.add_argument("--out", default=None, help="JSON report (default: <run>/grade-<block>.json)")
    a = ap.parse_args()
    sys.path.insert(0, a.exllamav3)
    from exllamav3.modules.quant.exl3 import LinearEXL3

    dev = torch.device(a.device)
    comp, idx = a.block.split(".")
    name = f"{'layer' if comp == 'layers' else 'mtp'}{int(idx):02d}"
    cfg = json.load(open(os.path.join(a.ref, "config.json")))["text_config"]
    limit = cfg["swiglu_limit"] if a.swiglu_limit is None else a.swiglu_limit
    n_exp = cfg["n_routed_experts"] if comp == "layers" else cfg["dspark_n_routed_experts"]
    if a.experts is not None:
        n_exp = min(n_exp, a.experts)

    hold = safe_open(os.path.join(a.run, "holdout", f"{name}.safetensors"), framework="pt")
    x = hold.get_tensor("x").to(dev)
    route_i, route_w = hold.get_tensor("indices").to(dev).long(), hold.get_tensor("weights").to(dev)
    xq = gemm_input(x)                                   # what every expert GEMM multiplies
    shards = sorted(glob.glob(os.path.join(a.run, "exl3-k*", f"model-{name}.safetensors")))
    if not shards:
        raise SystemExit(f"{a.run}: no exl3-k*/model-{name}.safetensors")
    ckpt = Checkpoint(a.ref)

    def src(e, p):  # (out, in) fp32
        k = f"{a.block}.ffn.experts.{e}.{p}"
        return refkernels.dequant_fp4(ckpt.get(f"{k}.weight").view(torch.float4_e2m1fn_x2).to(dev),
                                      ckpt.get(f"{k}.scale").to(dev))

    def mid(xin, w1, w3):
        g = F.linear(xin.bfloat16(), w1.bfloat16()).float()
        u = F.linear(xin.bfloat16(), w3.bfloat16()).float()
        if limit > 0:
            u, g = u.clamp(-limit, limit), g.clamp(max=limit)
        return gemm_input((F.silu(g) * u).bfloat16())

    report = {"block": a.block, "holdout_tokens": x.size(0), "experts": n_exp, "by_k": {}}
    q_files = {int(os.path.basename(os.path.dirname(s))[len("exl3-k"):]): safe_open(s, framework="pt", device="cpu")
               for s in shards}
    y_src = torch.zeros(x.size(0), x.size(1), dtype=torch.float32, device=dev)
    y_q = {K: torch.zeros_like(y_src) for K in q_files}
    per = {K: {"rel_fro": {p: [] for p in ("w1", "w3", "w2")}, "func": {p: [] for p in ("w1", "w3", "w2")}}
           for K in q_files}
    for e in range(n_exp):
        W = {p: src(e, p) for p in ("w1", "w3", "w2")}
        hit = (route_i == e)
        tok = hit.any(dim=1).nonzero().flatten()
        wt = (route_w * hit).sum(dim=1)[tok]
        xs = xq[tok]
        h_src = mid(xs, W["w1"], W["w3"])
        o_src = F.linear(h_src.bfloat16(), W["w2"].bfloat16()).float()
        y_src[tok] += wt[:, None] * o_src
        for K, f in q_files.items():
            Q = {}
            for p in ("w1", "w3", "w2"):
                k = f"{a.block}.ffn.experts.{e}.{p}"
                t = {s: f.get_tensor(f"{k}.{s}").to(dev) for s in ("trellis", "suh", "svh", "mul1")}
                o, i = W[p].shape
                Q[p] = LinearEXL3(None, i, o, **t).get_weight_tensor().float().T
                per[K]["rel_fro"][p].append(rel(Q[p], W[p]))
            if tok.numel():
                per[K]["func"]["w1"].append(rel(xs @ Q["w1"].T, xs @ W["w1"].T))
                per[K]["func"]["w3"].append(rel(xs @ Q["w3"].T, xs @ W["w3"].T))
                per[K]["func"]["w2"].append(rel(h_src @ Q["w2"].T, h_src @ W["w2"].T))
                h_q = mid(xs, Q["w1"], Q["w3"])
                y_q[K][tok] += wt[:, None] * F.linear(h_q.bfloat16(), Q["w2"].bfloat16()).float()
    for K in sorted(q_files):
        proxy_path = os.path.join(a.run, f"exl3-k{K}", f"proxy-{name}.json")
        proxy = json.load(open(proxy_path)) if os.path.exists(proxy_path) else {}
        r = {"rel_fro": {p: stats(v) for p, v in per[K]["rel_fro"].items()},
             "func_routed": {p: stats(v) for p, v in per[K]["func"].items() if v},
             "moe_out_rel": rel(y_q[K], y_src),
             "proxy_err": {p: stats([v for k, v in proxy.items() if k.endswith(p)]) for p in ("w1", "w3", "w2")}
             if proxy else None}
        report["by_k"][K] = r
        print(f"{a.block} K{K}: moe_out rel {r['moe_out_rel']:.4f} | "
              + " | ".join(f"{p} fro {r['rel_fro'][p]['median']:.4f} func {r['func_routed'][p]['median']:.4f}"
                           + (f" proxy {r['proxy_err'][p]['median']:.5f}" if proxy else "")
                           for p in ("w1", "w3", "w2")), flush=True)
    out = a.out or os.path.join(a.run, f"grade-{name}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

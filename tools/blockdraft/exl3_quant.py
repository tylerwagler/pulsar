#!/usr/bin/env python3
"""exl3_quant.py -- our own EXL3 quantisation of a BF16 block drafter (L300): every Linear of the DFlash2 drafter and
its draft-vocabulary head, at K = 4 / 5 / 6, with Hessians from the drafter's own inputs on the engine's taps.

Two steps, because the second needs a GPU (exllamav3's trellis search, ext.quantize_tiles, is CUDA only):

  hessians   (CPU)  runs the BF16 drafter (ref.py's arithmetic) over grade sources (fmtgrade.py: a PULSAR_BLOCK_DUMP
                    and / or L292 tap fixtures) and accumulates H = X^T X of every Linear's input as the served lane
                    feeds it: fc and each layer's k / v over every context row (the absorb), every block Linear and the
                    head over the block rows of a draft at every known position (stride --stride).  -> one .pt file
  quantize   (GPU)  exllamav3's quantize_exl3 per Linear with its Hessian (the convert defaults: mul1 codebook, sigma_reg
                    0.025, output scales auto, seed = the Linear's index), the head's out padded to 128 as exllamav3's
                    Linear pads it -> dflash2_k{K}.safetensors in exllamav3's names (X.trellis / X.suh / X.svh /
                    X.mul1, lm_head.* the draft-vocabulary rows), which fmtgrade.py reads as exl3:FILE for the Linears
                    and the head

    python3 tools/blockdraft/exl3_quant.py hessians --drafter /mnt/models/qwen38fn-dflash2 --target-hf HF \
        --container CONTAINER (--dump DIR | --l292 NAME:TAPSDIR:IDS:PLEN ...) --out H.pt
    PYTHONPATH=<exllamav3 checkout> python3 tools/blockdraft/exl3_quant.py quantize --drafter ... --target-hf HF \
        --container CONTAINER --hessians H.pt --k 4,5,6 --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fmtgrade as FG  # noqa: E402
import ref as R  # noqa: E402
import weights as WS  # noqa: E402

torch.set_grad_enabled(False)
LAYER_LINEARS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
                 "acp": "attention_conv.kernel_projection", "mcp": "mlp_conv.kernel_projection",
                 "g": "mlp.gate_proj", "u": "mlp.up_proj", "d": "mlp.down_proj"}


def linear_names(D):
    """id(W) -> the HF Linear name, for every Linear the format grade quantises (the selector projection is not)"""
    names = {id(D.fc[0]): "fc", id(D.head[0]): "lm_head"}
    for l, Lw in enumerate(D.lay):
        for k, n in LAYER_LINEARS.items():
            names[id(Lw[k][0])] = f"layers.{l}.{n}"
    return names


class Capture(R.Drafter):
    """ref.Drafter whose Linears accumulate X^T X of their inputs (f64 sums kept in f32 per call, as exllamav3)."""

    def arm(self):
        self.names = linear_names(self)
        self.H = {}
        self.count = {}

    def lin(self, x, w):
        n = self.names.get(id(w[0])) if hasattr(self, "names") else None
        if n is not None:
            xf = x.float().reshape(-1, x.shape[-1])
            if n == "lm_head":
                xf = xf[1:] if xf.shape[0] == self.block else xf   # the head serves the draft rows only
            if n not in self.H:
                self.H[n] = torch.zeros(xf.shape[1], xf.shape[1], dtype=torch.float32)
                self.count[n] = 0
            self.H[n].addmm_(xf.T, xf)
            self.count[n] += xf.shape[0]
        return R.Drafter.lin(x, w)


def cmd_hessians(a):
    cache = WS.Cache()
    src = WS.Arm("bf16", "bf16", "bf16", a.drafter, a.target_hf, a.container, cache)
    D = Capture(a.drafter, None, torch.device("cpu"), src=src)
    D.arm()
    sources = []
    B = a.block
    if a.dump:
        dm = FG.Dump(a.dump)
        B = dm.B
        FG.log("dump:", dm.summary())
        for e in dm.epochs:
            if e["passes"]:
                e["label"] = f"bank{e['bank']}.e{e['id']}"
                e["rows"] = (lambda e_: (lambda lo, hi: dm.rows(e_, lo, hi)))(e)
                e["spans"] = [s for s in dm.spans(e) if s[1] > s[0]]
                sources.append(e)
    for spec in a.l292 or []:
        sources.append(FG.taps_fixture(spec, B))
    if not sources:
        raise SystemExit("hessians: --dump DIR and / or --l292 NAME:TAPSDIR:IDS:PROMPT_LEN")
    for e in sources:
        spans = e["spans"]
        hi = max(s[1] for s in spans)
        lo = max(0, min(s[0] for s in spans) - (D.W - 1) - (B - 1))
        taps, have = e["rows"](lo, hi)
        t0 = time.time()
        kvs = None
        for i in range(0, hi - lo, 1024):   # the absorb: fc, then k / v of every context row
            t = R.u16(taps[i:i + 1024], D.dev)
            _fc, cn = D.context(t)
            part = D.ctx_kv(cn, torch.arange(lo + i, lo + i + t.shape[0]))
            kvs = part if kvs is None else [(torch.cat([k0, k1]), torch.cat([v0, v1]))
                                            for (k0, v0), (k1, v1) in zip(kvs, part)]
        miss = np.flatnonzero(~have)
        n = 0
        for s0, s1 in spans:                 # a draft at every stride-th known position
            for p in range(s0, s1, a.stride):
                lo_p = max(0, p - (D.W - 1) - (B - 1))
                if ((miss >= lo_p - lo) & (miss < p - lo)).any():
                    continue
                kv = [(k[lo_p - lo:p - lo], v[lo_p - lo:p - lo]) for k, v in kvs]
                D.forward(None, torch.arange(lo_p, p), p, e["S"][p], kv=kv)
                n += 1
        FG.log(f"source {e['label']}: {hi - lo} context rows, {n} drafts in {time.time() - t0:.0f} s")
    torch.save(dict(H=D.H, count=D.count, sources=[e["label"] for e in sources], block=B), a.out)
    FG.log(f"wrote {a.out}: " + ", ".join(f"{k} {D.count[k]}" for k in sorted(D.count)[:6]) + " ...")


def bf16_weight(src: WS.Arm, hf):
    """the Linear's BF16 source as f32 [in][out_padded] (exllamav3's orientation; out padded to 128 with zeros)"""
    W = src.cache.linear("bf16", hf + ".weight", src)[0]
    out, inp = W.shape
    pad = WS.PR._rup(out, 128) - out
    Wt = W.T.contiguous()
    if pad:
        Wt = torch.cat([Wt, torch.zeros(inp, pad)], dim=1)
    return Wt


def cmd_quantize(a):
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3   # needs exllamav3's CUDA extension
    from safetensors.torch import save_file
    if not torch.cuda.is_available():
        raise SystemExit("quantize: exllamav3's trellis search runs on a GPU (no CUDA device here)")
    dev = torch.device("cuda:0")
    hs = torch.load(a.hessians)
    cache = WS.Cache()
    src = WS.Arm("bf16", "bf16", "bf16", a.drafter, a.target_hf, a.container, cache)
    cfg = json.load(open(os.path.join(a.drafter, "config.json")))
    names = ["fc"] + [f"layers.{l}.{n}" for l in range(cfg["num_hidden_layers"]) for n in LAYER_LINEARS.values()] \
        + ["lm_head"]
    missing = [n for n in names if n not in hs["H"]]
    if missing:
        raise SystemExit(f"{a.hessians} has no Hessian for {missing[:4]}")
    os.makedirs(a.out_dir, exist_ok=True)
    for K in [int(x) for x in a.k.split(",")]:
        path = os.path.join(a.out_dir, f"dflash2_k{K}.safetensors")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists -- refusing")
        tensors, report = {}, {}
        t0 = time.time()
        for idx, n in enumerate(names):
            W = bf16_weight(src, n).to(dev)
            H = hs["H"][n].to(dev).clone()
            H_data = {"H": H, "first_key": n, "count": hs["count"][n], "finalized": False,
                      "num_total": hs["count"][n] * H.shape[0], "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
                      "device": dev}
            qa = {"seed": idx, "K": K, "devices": [0], "device_ratios": None, "apply_out_scales": None,
                  "debug_dir": os.path.join(a.out_dir, "debug"), "mul1": True}
            _wq, proxy_err, out = quantize_exl3(W, H_data, qa, False)
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    tensors[f"{n}.{k}"] = v.contiguous().cpu()
            report[n] = dict(proxy_err=float(proxy_err), q_fallback=bool(qa.get("q_fallback", False)))
            print(f"  K{K} {n:40s} proxy err {float(proxy_err):.6f}{' FALLBACK' if qa.get('q_fallback') else ''}",
                  flush=True)
        save_file(tensors, path, metadata={"l300": json.dumps(dict(K=K, hessians=os.path.abspath(a.hessians),
                                                                   sources=hs["sources"], report=report))})
        FG.log(f"K{K}: wrote {path} in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["hessians", "quantize"])
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target-hf", required=True)
    ap.add_argument("--container", required=True, help="the served container (the draft vocabulary's ids)")
    ap.add_argument("--dump")
    ap.add_argument("--l292", action="append")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1, help="hessians: a draft at every stride-th known position")
    ap.add_argument("--hessians")
    ap.add_argument("--k", default="4,5,6")
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "hessians":
        if not a.out:
            raise SystemExit("hessians: --out H.pt")
        cmd_hessians(a)
    else:
        if not a.hessians or not a.out_dir:
            raise SystemExit("quantize: --hessians H.pt --out-dir DIR")
        cmd_quantize(a)


if __name__ == "__main__":
    main()

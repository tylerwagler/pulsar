#!/usr/bin/env python3
"""Go / no-go for DeepSeek's inference/ stack on this GPU (the V4.1 kit's on-start probe, notes/reference-capture/
v41/onstart.sh, as a file): the reference's own act_quant + fp8 linear at M = 1 / 512 / 2048 must be bit-identical
on rerun, NaN-free, and within 1e-2 relative of an fp32 product.  Rentals #4/#5 (B200, sm_100, tilelang 0.1.8)
compiled fine and returned NaN rows and different bits on every call for M > 1: every prefill from those boxes was
garbage.  Exit 1 = do not capture on this host with this stack.

    CUDA_VISIBLE_DEVICES=6 python probe_fp8.py --inference $SRC/inference
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inference", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.inference)
    import model as M
    from kernel import act_quant
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    print("probe on", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    K, N = 5120, 1280
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    ws = torch.full(((N + 31) // 32, (K + 31) // 32), 1.0, device="cuda").to(torch.float8_e8m0fnu)
    W = torch.nn.Parameter(w, requires_grad=False)
    W.scale = ws
    wd = w.float() * ws.float().repeat_interleave(32, 0)[:N].repeat_interleave(32, 1)[:, :K]
    bad = False
    for rows in (1, 512, 2048):
        x = torch.randn(1, rows, K, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            y1 = M.linear(x, W)
            torch.cuda.synchronize()
            y2 = M.linear(x, W)
            torch.cuda.synchronize()
            xq, sc = act_quant(x, 32, M.scale_fmt, M.scale_dtype)
            xd = xq.float() * sc.float().reshape(1, rows, -1).repeat_interleave(32, -1)[..., :K]
            yref = (xd.reshape(-1, K) @ wd.T).reshape(1, rows, N)
        nan = int(torch.isnan(y1.float()).any(-1).sum())
        same = torch.equal(y1, y2)
        rel = ((y1.float() - yref).abs() / (yref.abs() + 1e-2)).nan_to_num(9).max().item()
        print(f"  fp8 linear M={rows}: rerun identical {same}, NaN rows {nan}, max rel {rel:.2e}", flush=True)
        bad |= (not same) or nan > 0 or rel > 1e-2
    print("NO-GO" if bad else "GO", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

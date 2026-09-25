#!/usr/bin/env python3
"""v41_stream.quantize_layer on two real V4.1 experts, on a CUDA device.

    python tools/exl3quant/test_quantize.py --ref /path/to/DeepSeek-V4.1-Flash --exllamav3 ~/exllamav3 \\
        [--layer 20] [--device cuda:0]

Takes experts 0 and 1 of one layer from the checkpoint, builds their Hessians the way
ExpertHessians does from synthetic FFN inputs, quantizes at K=2 and K=3, and checks:
  * every tensor passes tools/container's Exl3Checkpoint reader (the builder's --exl3 input);
  * proxy errors are finite and fall from K=2 to K=3;
  * exllamav3's own reconstruction of each trellis is close to the source weight.
"""

import argparse
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "container"))
import refkernels  # noqa: E402
import v41_stream as vs  # noqa: E402

FAILS = []


def check(cond, what):
    print(("PASS  " if cond else "FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class TwoExperts(torch.nn.Module):
    """The slice of MoE that ExpertHessians and quantize_layer read."""

    def __init__(self, experts, dim):
        super().__init__()
        self.experts = torch.nn.ModuleList(experts)
        self.dim = dim
        self.n_routed_experts = len(experts)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--exllamav3", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    sys.path.insert(0, a.exllamav3)
    torch.set_default_dtype(torch.bfloat16)

    ref = vs.load_reference(a.ref)
    args = vs.model_args(ref, a.ref, 1, 128)
    ckpt = vs.Checkpoint(a.ref)
    experts = []
    for e in range(2):
        with torch.device(dev):
            ex = ref.Expert(args.dim, args.moe_inter_dim, dtype=torch.float4_e2m1fn_x2, swiglu_limit=args.swiglu_limit)
        pre = f"layers.{a.layer}.ffn.experts.{e}."
        sd = {}
        for w in ("w1", "w2", "w3"):
            sd[f"{w}.weight"] = ckpt.get(pre + w + ".weight").view(torch.float4_e2m1fn_x2)
            sd[f"{w}.scale"] = ckpt.get(pre + w + ".scale")
        vs.load_into(ex, sd, pre)
        experts.append(ex)
    moe = TwoExperts(experts, args.dim)

    hess = vs.ExpertHessians(moe, dev, flush_rows=4096)
    torch.manual_seed(0)
    for _ in range(3):  # 3 x 4096 rows, rank well above 5120 is not needed for the plumbing
        hess.observe((torch.randn(4096, args.dim, device=dev) * 0.5).bfloat16())
    hess.flush()
    check(hess.count == 3 * 4096, f"ExpertHessians: {hess.count} rows counted")

    import hf_source

    with tempfile.TemporaryDirectory() as tmp:
        prox = {}
        for K in (2, 3):
            out = os.path.join(tmp, f"k{K}")
            prox[K] = vs.quantize_layer(hess, a.layer, K, [dev.index or 0], out, os.path.join(tmp, "debug"))
            check(len(prox[K]) == 6 and all(v == v and v < 1 for v in prox[K].values()),
                  f"K={K}: 6 tensors, proxy_err finite < 1 (median {sorted(prox[K].values())[3]:.4f})")
            ck = hf_source.Exl3Checkpoint(out)
            ok = True
            for e in range(2):
                for w, (k, n) in (("w1", (args.dim, args.moe_inter_dim)), ("w3", (args.dim, args.moe_inter_dim)),
                                  ("w2", (args.moe_inter_dim, args.dim))):
                    _, words = ck.expert(a.layer, e, w, k, n)
                    ok &= words == 16 * K
            check(ok, f"K={K}: the container builder's Exl3Checkpoint reads every projection at {16 * K} words/tile")
        check(all(prox[3][k] < prox[2][k] for k in prox[2]), "proxy_err falls from K=2 to K=3 on every tensor")

        # exllamav3's own reconstruction of one K=3 tensor (LinearEXL3.get_weight_tensor) vs the source
        from safetensors.torch import load_file
        from exllamav3.modules.quant.exl3 import LinearEXL3
        t = load_file(os.path.join(tmp, "k3", f"model-layer{a.layer:02d}.safetensors"))
        key = f"layers.{a.layer}.ffn.experts.0.w1"
        lin = LinearEXL3(None, args.dim, args.moe_inter_dim,
                         **{s: t[f"{key}.{s}"].to(dev) for s in ("trellis", "suh", "svh", "mul1")})
        w_q = lin.get_weight_tensor().float()
        w_src = refkernels.dequant_fp4(experts[0].w1.weight, experts[0].w1.scale).T
        rel = ((w_q - w_src).norm() / w_src.norm()).item()
        check(rel < 0.25, f"K=3 {key}: exllamav3 reconstruction within {rel:.3f} relative Frobenius of the source")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

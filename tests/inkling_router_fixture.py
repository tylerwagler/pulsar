#!/usr/bin/env python3
"""L295 F2: the Inkling-Small router's answer on REAL router weights, for tests/router_gate.cu.

The oracle is the reference's own code, not a re-derivation: transformers' InklingTopkRouter (sigmoid scores, the
selection bias, top-k, log-sigmoid renormalisation over the k routed logits and the S sink logits, x route_scale x
global_scale) is instantiated, loaded with a layer's gate tensors from the 3-layer mini checkpoint
(/mnt/models/inkling-small-mini, made by the quantization session from Inkling-Small layers 0 / 2 / 5) and run in
fp32 on bf16 rows.  The rows approximate the MoE input: the mini model's fp32 reference hidden state entering that
layer (inkling-small-mini-ref/ref_fp32.pt), RMS-normed by the layer's mlp_norm and rounded to bf16 -- the attention
delta is omitted, so they are realistic in scale and direction, not the exact activations.  Only the router is under
test: it sees the same bf16 rows the fixture records.

Each record (little-endian):
  b"INKRTR01", int32 layer, T, H, R (logit rows = routed + sinks), E (routed), K, S, f32 route_scale, f32 global_scale,
  x bf16 [T][H], W bf16 [R][H], bias f32 [E],
  logits f32 [T][R] (fp32 F.linear of the bf16 values), idx int32 [T][K] (the router's top-k, torch order),
  w f32 [T][K], gammas f32 [T][S]

usage: inkling_router_fixture.py MINI_DIR REF_PT OUT.bin   (needs torch + transformers >= 5.14, CPU only)
"""
import json
import struct
import sys
import types

import torch
from safetensors import safe_open
from transformers.models.inkling.modeling_inkling import InklingTopkRouter

if len(sys.argv) != 4:
    sys.exit(__doc__)
mini, ref_pt, out_path = sys.argv[1:4]
torch.manual_seed(0)
cfg = json.load(open(f"{mini}/config.json"))["text_config"]
E, S, K = cfg["n_routed_experts"], cfg["n_shared_experts"], cfg["num_experts_per_tok"]
H, route_scale = cfg["hidden_size"], float(cfg["route_scale"])
moe_layers = [i for i in range(cfg["num_hidden_layers"]) if i >= cfg["dense_mlp_idx"]]
hidden = torch.load(ref_pt, map_location="cpu", weights_only=False)["hidden_states"]

with open(out_path, "wb") as out:
    for layer in moe_layers:
        pre = f"model.llm.layers.{layer}"
        with safe_open(f"{mini}/layer-{layer}.safetensors", "pt") as f:
            W = f.get_tensor(f"{pre}.mlp.gate.weight")            # bf16 [E + S][H]
            bias = f.get_tensor(f"{pre}.mlp.gate.bias").float()     # [E]
            gscale = f.get_tensor(f"{pre}.mlp.gate.global_scale").float()
            norm_w = f.get_tensor(f"{pre}.mlp_norm.weight").float()
        h = hidden[layer][0]                                         # [tokens][H] f32, the layer's input
        rows = torch.cat([torch.arange(0, 48), torch.arange(48, h.shape[0], 61)])
        hn = h[rows] * torch.rsqrt(h[rows].pow(2).mean(-1, keepdim=True) + cfg["rms_norm_eps"]) * norm_w
        x = hn.to(torch.bfloat16)
        conf = types.SimpleNamespace(n_routed_experts=E, n_shared_experts=S, hidden_size=H, route_scale=route_scale,
                                     num_experts_per_tok=K)
        router = InklingTopkRouter(conf).float()
        with torch.no_grad():
            router.weight.copy_(W.float())
            router.global_scale.copy_(gscale)
            router.e_score_correction_bias.copy_(bias)
            xf = x.float()
            logits = torch.nn.functional.linear(xf, W.float())
            _, w, idx, gam = router(xf)
        T, R = x.shape[0], W.shape[0]
        out.write(b"INKRTR01")
        out.write(struct.pack("<7i2f", layer, T, H, R, E, K, S, route_scale, float(gscale[0])))
        out.write(x.view(torch.int16).numpy().tobytes())
        out.write(W.view(torch.int16).numpy().tobytes())
        out.write(bias.numpy().astype("<f4").tobytes())
        out.write(logits.numpy().astype("<f4").tobytes())
        out.write(idx.to(torch.int32).numpy().astype("<i4").tobytes())
        out.write(w.float().numpy().astype("<f4").tobytes())
        out.write(gam.float().numpy().astype("<f4").tobytes())
        print(f"layer {layer}: {T} rows, global_scale {float(gscale[0]):.6g}, weights sum "
              f"{float((w.sum(-1) + gam.sum(-1)).mean()):.6g} (route_scale x global_scale "
              f"{route_scale * float(gscale[0]):.6g})")
print(f"wrote {out_path}")

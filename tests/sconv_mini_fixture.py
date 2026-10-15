#!/usr/bin/env python3
"""Short-convolution test vectors for the SConv block (L295 E3) from the Inkling-Small
mini checkpoint (3 layers, source layers 0 / 2 / 5; made by the L294 quant session).

Layer 0 (a window layer with the dense MLP) is rebuilt in fp32 from the transformers
fp32 capture's layer input (ref_fp32.pt hidden_states[0], 1300 tokens), following
transformers' InklingDecoderLayer:

    k, v   = k_sconv(x Wk), v_sconv(x Wv)        x = input RMSNorm(h0)
    h1     = h0 + attn_sconv(o_proj(attention))
    h2     = h1 + mlp_sconv(mlp(post-attention RMSNorm(h1)))

and h2 is checked against the capture's hidden_states[1] (so the four streams' inputs
are the model's own, and the convolution's placement is confirmed).  For each stream
(k, v, attn, mlp) it writes

    NAME.in.bin     bf16 [1300][C]   the convolution's input, rounded to bf16
    NAME.w.bin      bf16 [C][4]      the checkpoint's depthwise weight as stored
    NAME.ref32.bin  f32  [1300][C]   transformers' InklingShortConvolution (fp32 conv1d
                                     + residual) of the bf16 input

tests/sconv_gate --fixture DIR grades the block against ref32.  CPU only, about a minute.

Usage: sconv_mini_fixture.py MINI_DIR REF_PT OUT_DIR
"""
import json
import os
import sys

import torch
from safetensors import safe_open


def rmsnorm(x, w, eps):
    # transformers InklingRMSNorm: weight * (x * rsqrt(mean(x^2) + eps)).to(dtype), in f32
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def sconv(x, w):
    # transformers InklingShortConvolution on a whole prompt: fp32 depthwise causal conv + residual
    xf = x.float()
    K = w.shape[-1]
    y = torch.nn.functional.conv1d(xf.t()[None], w.float(), padding=K - 1, groups=xf.shape[1])[0, :, : xf.shape[0]].t()
    return y + xf


def attention(q_raw, k, v, r, qn, kn, P, eps, window):
    # transformers InklingAttention after the projections and the K / V convolutions, fp32, a window layer
    # (no log-length temperature); the same math as tests/gqa_mini_fixture.py (L295 F1)
    T = q_raw.shape[0]
    nq, nkv, hd, drel = 32, 8, 128, 16
    q = rmsnorm(q_raw.view(T, nq, hd), qn, eps).transpose(0, 1)
    k = rmsnorm(k.view(T, nkv, hd), kn, eps).transpose(0, 1)
    v = v.view(T, nkv, hd).transpose(0, 1)
    rel = r.view(T, nq, drel)
    extent = P.shape[1]
    pos = torch.arange(T)
    rl = (rel @ P).transpose(0, 1)
    dist = pos[:, None] - pos[None, :]
    idx = dist.clamp(0, extent - 1).expand(nq, -1, -1)
    bias = rl.gather(-1, idx).masked_fill((dist < 0) | (dist >= extent), 0.0)
    G = nq // nkv
    k = k.repeat_interleave(G, 0)
    v = v.repeat_interleave(G, 0)
    s = torch.matmul(q, k.transpose(1, 2)) * (1.0 / hd) + bias
    s = s.masked_fill((dist < 0) | (dist >= window), float("-inf"))
    p = torch.softmax(s, dim=-1, dtype=torch.float32)
    return torch.matmul(p, v).transpose(0, 1).reshape(T, nq * hd)


def main():
    mini, ref_pt, out = sys.argv[1:4]
    cfg = json.load(open(os.path.join(mini, "config.json")))["text_config"]
    eps = cfg["rms_norm_eps"]
    assert 0 in cfg["local_layer_ids"] and cfg["sconv_kernel_size"] == 4
    ref = torch.load(ref_pt, map_location="cpu", weights_only=False)
    torch.set_num_threads(16)
    f = safe_open(os.path.join(mini, "layer-0.safetensors"), "pt")
    W = lambda n: f.get_tensor(f"model.llm.layers.0.{n}")
    Wf = lambda n: W(n).float()
    h0 = ref["hidden_states"][0][0].float()
    x = rmsnorm(h0, Wf("attn_norm.weight"), eps)
    streams = {}
    streams["k"] = (x @ Wf("attn.wk_dv.weight").t(), W("attn.k_sconv.weight"))
    streams["v"] = (x @ Wf("attn.wv_dv.weight").t(), W("attn.v_sconv.weight"))
    k = sconv(*streams["k"])
    v = sconv(*streams["v"])
    a = attention(x @ Wf("attn.wq_du.weight").t(), k, v, x @ Wf("attn.wr_du.weight").t(), Wf("attn.q_norm.weight"),
                  Wf("attn.k_norm.weight"), Wf("attn.rel_logits_proj.proj"), eps, cfg["sliding_window_size"])
    streams["attn"] = (a @ Wf("attn.wo_ud.weight").t(), W("attn_sconv.weight"))
    h1 = h0 + sconv(*streams["attn"])
    y = rmsnorm(h1, Wf("mlp_norm.weight"), eps)
    w13 = Wf("mlp.w13_dn.weight")          # gate / up rows interleaved (transformers' Interleave(dim=0) + Chunk)
    m = torch.nn.functional.silu(y @ w13[0::2].t()) * (y @ w13[1::2].t())
    streams["mlp"] = ((m @ Wf("mlp.w2_md.weight").t()) * Wf("mlp.global_scale"), W("mlp_sconv.weight"))
    h2 = h1 + sconv(*streams["mlp"])
    want = ref["hidden_states"][1][0].float()
    rel = ((h2 - want).norm() / want.norm()).item()
    print(f"layer 0 rebuilt in fp32 vs the capture's hidden_states[1]: rel fro {rel:.3e}")
    if rel > 1e-4:
        sys.exit("the rebuilt layer does not match the capture -- the streams would not be the model's")
    os.makedirs(out, exist_ok=True)
    for name, (xin, w) in streams.items():
        xb = xin.to(torch.bfloat16)
        r32 = sconv(xb, w)
        xb.contiguous().view(torch.int16).numpy().tofile(os.path.join(out, f"{name}.in.bin"))
        w.reshape(w.shape[0], -1).contiguous().view(torch.int16).numpy().tofile(os.path.join(out, f"{name}.w.bin"))
        r32.contiguous().numpy().tofile(os.path.join(out, f"{name}.ref32.bin"))
        print(f"{name:4s}: {xb.shape[0]} x {xb.shape[1]}, rms(in) {xin.pow(2).mean().sqrt().item():.4f}, "
              f"rms(out) {r32.pow(2).mean().sqrt().item():.4f}, rms(conv part) {(r32 - xb.float()).pow(2).mean().sqrt().item():.4f}")


if __name__ == "__main__":
    main()

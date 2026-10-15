#!/usr/bin/env python3
"""Attention-only test vectors for the GQA block (L295 F1) from the Inkling-Small
mini checkpoint (3 layers, source layers 0 / 2 / 5; made by the L294 quant session).

For a window layer (mini 0) and the global layer (mini 2) it takes the layer's input
from the transformers fp32 reference capture (ref_fp32.pt: hidden_states[i] is layer
i's input, 1300 tokens), runs what precedes the attention block in fp32 -- the
input RMSNorm, the q / k / v / r projections, the K / V short convolutions (with
their residual) -- and writes, per layer:

  in.bin      bf16 [1300][q 4096 | k 1024 | v 1024 | r 512]   the block's inputs
  qn.bin kn.bin  bf16 [128]   P.bin bf16 [16][extent]         the block's weights
  ref32.bin   f32 [1300][4096]  transformers' attention (InklingAttention.forward up to
              o_proj: per-head RMSNorm, relative logits, log-length tau, eager softmax)
              computed in fp32 from the bf16 inputs above
  ref16.bin   f32 [1300][4096]  the same in bf16 (the model's dtype), widened
  meta.txt    the layer's declaration

tests/gqa_attn_gate --fixture DIR grades the block against both.  CPU only.

Usage: gqa_mini_fixture.py MINI_DIR REF_PT OUT_DIR
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
    # InklingShortConvolution on a whole prompt: fp32 depthwise causal conv (kernel K) + residual
    xf = x.float()
    K = w.shape[-1]
    y = torch.nn.functional.conv1d(xf.t()[None], w.float(), padding=K - 1, groups=xf.shape[1])[0, :, : xf.shape[0]].t()
    return (y + xf).to(x.dtype)


def attention(q_raw, k_raw, v_raw, r, qn, kn, P, eps, window, floor_, alpha, dtype):
    T = q_raw.shape[0]
    nq, nkv, hd, drel = 32, 8, 128, 16
    q = rmsnorm(q_raw.to(dtype).view(T, nq, hd), qn.to(dtype), eps).transpose(0, 1)   # [h, T, hd]
    k = rmsnorm(k_raw.to(dtype).view(T, nkv, hd), kn.to(dtype), eps).transpose(0, 1)
    v = v_raw.to(dtype).view(T, nkv, hd).transpose(0, 1)
    rel = r.to(dtype).view(T, nq, drel)
    extent = P.shape[1]
    pos = torch.arange(T)
    rl = (rel @ P.to(dtype)).transpose(0, 1)                         # [h, T, extent]
    dist = pos[:, None] - pos[None, :]
    idx = dist.clamp(0, extent - 1).expand(nq, -1, -1)
    bias = rl.gather(-1, idx).masked_fill((dist < 0) | (dist >= extent), 0.0)
    if floor_:
        tau = 1.0 + alpha * torch.log(((pos + 1).float() / floor_).clamp(min=1.0))
        tau = tau.view(1, -1, 1)
        q = (q.float() * tau).to(q.dtype)
        bias = (bias.float() * tau).to(bias.dtype)
    G = nq // nkv
    k = k.repeat_interleave(G, 0)
    v = v.repeat_interleave(G, 0)
    s = torch.matmul(q, k.transpose(1, 2)) * (1.0 / hd) + bias
    mask = dist < 0
    if window:
        mask = mask | (dist >= window)
    s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(p, v)                                           # [h, T, hd]
    return o.transpose(0, 1).reshape(T, nq * hd).float()


def main():
    mini, ref_pt, out = sys.argv[1:4]
    cfg = json.load(open(os.path.join(mini, "config.json")))["text_config"]
    eps = cfg["rms_norm_eps"]
    ref = torch.load(ref_pt, map_location="cpu", weights_only=False)
    torch.set_num_threads(16)
    for layer in (0, 2):
        local = layer in cfg["local_layer_ids"]
        f = safe_open(os.path.join(mini, f"layer-{layer}.safetensors"), "pt")
        W = lambda n: f.get_tensor(f"model.llm.layers.{layer}.{n}")
        h = ref["hidden_states"][layer][0].float()
        x = rmsnorm(h, W("attn_norm.weight").float(), eps)
        q_raw = x @ W("attn.wq_du.weight").float().t()
        k_raw = sconv(x @ W("attn.wk_dv.weight").float().t(), W("attn.k_sconv.weight"))
        v_raw = sconv(x @ W("attn.wv_dv.weight").float().t(), W("attn.v_sconv.weight"))
        r = x @ W("attn.wr_du.weight").float().t()
        ins = [t.to(torch.bfloat16) for t in (q_raw, k_raw, v_raw, r)]
        qn, kn, P = W("attn.q_norm.weight"), W("attn.k_norm.weight"), W("attn.rel_logits_proj.proj")
        window = cfg["sliding_window_size"] if local else 0
        floor_ = 0 if local else cfg["log_scaling_n_floor"]
        alpha = cfg["log_scaling_alpha"]
        r32 = attention(*[t.float() for t in ins], qn.float(), kn.float(), P.float(), eps, window, floor_, alpha, torch.float32)
        r16 = attention(*ins, qn, kn, P, eps, window, floor_, alpha, torch.bfloat16)
        d = os.path.join(out, f"layer{layer}")
        os.makedirs(d, exist_ok=True)
        torch.cat(ins, dim=1).contiguous().view(torch.int16).numpy().tofile(os.path.join(d, "in.bin"))
        for name, t in (("qn", qn), ("kn", kn), ("P", P)):
            t.contiguous().view(torch.int16).numpy().tofile(os.path.join(d, f"{name}.bin"))
        r32.numpy().tofile(os.path.join(d, "ref32.bin"))
        r16.numpy().tofile(os.path.join(d, "ref16.bin"))
        with open(os.path.join(d, "meta.txt"), "w") as m:
            m.write(f"tokens {h.shape[0]}\nwindow {window}\nextent {P.shape[1]}\nfloor {floor_}\nalpha {alpha}\neps {eps}\n")
        e = ((r16 - r32).norm() / r32.norm()).item()
        print(f"layer {layer} ({'window' if local else 'global'}): {h.shape[0]} tokens, extent {P.shape[1]}, "
              f"rms(ref32) {r32.pow(2).mean().sqrt().item():.4f}, transformers bf16 vs fp32 fro {e:.3e}")


if __name__ == "__main__":
    main()

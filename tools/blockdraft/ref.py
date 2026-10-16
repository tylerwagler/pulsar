#!/usr/bin/env python3
"""ref.py -- the block drafter's reference grade (L298): replay an engine dump (PULSAR_BLOCK_DUMP=<dir>, written by
src/engine/spec_block.cpp) through a PyTorch transcription of the drafter (vLLM's qwen3_dflash.py / qwen3_dflash2.py /
lilicorr.py backbone: the DFlash block forward with the DFlash2 grouped convolution) and grade the engine's draft pass
against it: each row's final hidden (cosine, max |diff|) and its top-k candidates (top-1 agreement, top-k overlap,
log-prob error).

The reference reads the drafter's own ModelOpt checkpoint (W4A16 NVFP4 linears dequantized exactly: e2m1 x e4m3 in
f32, the f32 global scale on the GEMM's output, as the NVFP4 W4A16 GEMM applies it), the target's input embedding from
the target's HF checkpoint, and the engine's dumped context taps (so the target's hidden states are the engine's: the
grade isolates the drafter).  Arithmetic follows vLLM's: bf16 activations between ops, f32 inside each op.

    ~/venvs/tf/bin/python tools/blockdraft/ref.py --dump DIR --drafter /mnt/models/nemotron35-super-lilicorr \
        --target-hf /mnt/models/nemotron35-super-vl-bf16 [--passes N] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

BF16 = torch.bfloat16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def bf(x):
    return x.to(BF16)


class Weights:
    def __init__(self, path, device):
        self.f = safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu")
        self.keys = set(self.f.keys())
        self.device = device

    def get(self, name):
        return self.f.get_tensor(name).to(self.device)

    def linear(self, name):
        """(W f32 [out][in], global scale or None)"""
        if name + ".weight_scale" in self.keys:
            w = self.f.get_tensor(name + ".weight")                      # u8 [out][in/2]
            sc = self.f.get_tensor(name + ".weight_scale").float()       # e4m3 [out][in/16]
            s2 = float(self.f.get_tensor(name + ".weight_scale_2").float())
            lo, hi = (w & 0xF).long(), (w >> 4).long()
            v = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(w.shape[0], -1)
            v = v * sc.repeat_interleave(16, dim=1)
            return v.to(self.device), s2
        return self.f.get_tensor(name + ".weight").float().to(self.device), None


def lin(x, w):
    W, s2 = w
    y = x.float() @ W.T
    if s2 is not None:
        y = y * s2
    return bf(y)


def rms(x, w, eps):
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return bf(bf(xf * r).float() * w.float())


def rope(x, pos, theta):
    """x bf16 [n][heads][d], pos [n]: NeoX, bf16 cos / sin, bf16 products (vLLM's rotary kernel on bf16)"""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
    ang = pos.float()[:, None] * inv[None, :]
    c, s = bf(torch.cos(ang))[:, None, :].float(), bf(torch.sin(ang))[:, None, :].float()
    x0, x1 = x[..., : d // 2].float(), x[..., d // 2:].float()
    r0 = bf(bf(x0 * c).float() - bf(x1 * s).float())
    r1 = bf(bf(x1 * c).float() + bf(x0 * s).float())
    return torch.cat([r0, r1], dim=-1)


def head_norm(x, w, eps):
    return rms(x, w, eps)


def conv(x, delta, base, taps, group, block):
    """x bf16 [R][C], delta bf16 [R][taps][G], base bf16 [taps][C] -> bf16 (Triton's f32 order)"""
    R, C = x.shape
    G = C // group
    d = delta.float().repeat_interleave(group, dim=-1)                 # [R][taps][C]
    co = base.float()[None] + d
    acc = co[:, 0] * x.float()
    pos = torch.arange(R, device=x.device) % block
    for t in range(1, taps):
        sh = torch.zeros_like(x.float())
        sh[t:] = x.float()[:-t]
        acc = acc + co[:, t] * sh * (pos >= t).float()[:, None]
    return bf(acc)


def read_dump(d):
    ctx = []
    with open(os.path.join(d, "ctx.bin"), "rb") as f:
        while True:
            h = f.read(4)
            if not h:
                break
            n = struct.unpack("<I", h)[0]
            pos = np.frombuffer(f.read(4 * n), dtype=np.int32)
            bank = np.frombuffer(f.read(4 * n), dtype=np.int32)
            w = struct.unpack("<I", f.read(4))[0]
            taps = np.frombuffer(f.read(2 * n * w), dtype=np.uint16).reshape(n, w)
            ctx.append((pos, bank, taps))
    passes = []
    with open(os.path.join(d, "draft.bin"), "rb") as f:
        while True:
            h = f.read(16)
            if not h:
                break
            nb, B, E, k = struct.unpack("<4I", h)
            bank = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            carry = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            P = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            o = np.frombuffer(f.read(2 * nb * B * E), dtype=np.uint16).reshape(nb * B, E)
            ids = np.frombuffer(f.read(4 * nb * B * k), dtype=np.int32).reshape(nb * B, k)
            lp = np.frombuffer(f.read(4 * nb * B * k), dtype=np.float32).reshape(nb * B, k)
            passes.append(dict(nb=nb, B=B, E=E, k=k, bank=bank, carry=carry, P=P, o=o, ids=ids, lp=lp))
    return ctx, passes


def u16_to_bf16(a):
    return torch.from_numpy(a.astype(np.int16).copy()).view(BF16)


def target_embed(path, device):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    name = "language_model.backbone.embeddings.weight"
    with safe_open(os.path.join(path, idx[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target-hf", required=True)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_grad_enabled(False)
    cfg = json.load(open(os.path.join(a.drafter, "config.json")))
    dc = cfg["dflash_config"]
    E, L, NQ, NKV, HD = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta, W = cfg["rms_norm_eps"], cfg["rope_theta"], dc["swa_window_size"]
    taps, group, block, mask = dc["conv_kernel_size"], dc["conv_group_size"], cfg["block_size"], dc["mask_token_id"]
    G = E // group
    w = Weights(a.drafter, dev)
    emb = target_embed(a.target_hf, dev)
    ctx, passes = read_dump(a.dump)
    fc = w.linear("fc")
    hn = w.get("hidden_norm.weight")
    lay = []
    for l in range(L):
        p = f"layers.{l}."
        lay.append(dict(
            inn=w.get(p + "input_layernorm.weight"), post=w.get(p + "post_attention_layernorm.weight"),
            q=w.linear(p + "self_attn.q_proj"), k=w.linear(p + "self_attn.k_proj"), v=w.linear(p + "self_attn.v_proj"),
            o=w.linear(p + "self_attn.o_proj"), qn=w.get(p + "self_attn.q_norm.weight"),
            kn=w.get(p + "self_attn.k_norm.weight"),
            acb=w.get(p + "attention_conv.base_kernel"), acp=w.linear(p + "attention_conv.kernel_projection"),
            mcb=w.get(p + "mlp_conv.base_kernel"), mcp=w.linear(p + "mlp_conv.kernel_projection"),
            g=w.linear(p + "mlp.gate_proj"), u=w.linear(p + "mlp.up_proj"), d=w.linear(p + "mlp.down_proj")))
    norm = w.get("norm.weight")
    head = w.linear("lm_head")
    # the context per bank: position -> taps row (the last absorbed wins: a re-prefilled position is rewritten)
    rows = {}
    for pos, bank, t in ctx:
        for i in range(len(pos)):
            rows.setdefault(int(bank[i]), {})[int(pos[i])] = t[i]
    worst = 0.0
    for pi, ps in enumerate(passes[: a.passes]):
        B = ps["B"]
        for bi in range(ps["nb"]):
            b, P, carry = int(ps["bank"][bi]), int(ps["P"][bi]), int(ps["carry"][bi])
            have = rows.get(b, {})
            lo = max(0, P - (W - 1) - (block - 1))
            missing = [x for x in range(lo, P) if x not in have]
            if missing:
                print(f"pass {pi} bank {b}: context positions missing from the dump ({len(missing)}, first {missing[0]})")
                continue
            cpos = torch.arange(lo, P, device=dev)
            tap = u16_to_bf16(np.stack([have[x] for x in range(lo, P)])).to(dev)
            c = lin(tap, fc)
            cn = rms(c, hn, eps)
            qpos = torch.arange(P, P + B, device=dev)
            ids = torch.tensor([carry] + [mask] * (B - 1), device=dev)
            h = emb[ids].to(BF16)
            t = None
            for l in range(L):
                Lw = lay[l]
                if t is None:
                    x = rms(h, Lw["inn"], eps)
                else:
                    h = bf(h.float() + t.float())
                    x = rms(h, Lw["inn"], eps)
                co = lin(x, Lw["acp"]).reshape(B, 2, taps, G)
                cx = conv(x, co[:, 0], Lw["acb"][0], taps, group, block)
                q = head_norm(lin(cx, Lw["q"]).reshape(B, NQ, HD), Lw["qn"], eps)
                kq = head_norm(lin(cx, Lw["k"]).reshape(B, NKV, HD), Lw["kn"], eps)
                vq = lin(cx, Lw["v"]).reshape(B, NKV, HD)
                kc = head_norm(lin(cn, Lw["k"]).reshape(-1, NKV, HD), Lw["kn"], eps)
                vc = lin(cn, Lw["v"]).reshape(-1, NKV, HD)
                q, kq, kc = rope(q, qpos, theta), rope(kq, qpos, theta), rope(kc, cpos, theta)
                K = torch.cat([kc, kq]).float()
                V = torch.cat([vc, vq]).float()
                kp = torch.cat([cpos, qpos])
                rep = NQ // NKV
                Kh, Vh = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
                s = torch.einsum("qhd,khd->hqk", q.float(), Kh) / math.sqrt(HD)
                vis = (qpos[:, None] - kp[None, :]) <= (W - 1)         # the window; bidirectional in the block
                s = s.masked_fill(~vis[None], float("-inf"))
                at = bf(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), Vh)).reshape(B, NQ * HD)
                ob = lin(at, Lw["o"])
                t = conv(ob, co[:, 1], Lw["acb"][1], taps, group, block)
                h = bf(h.float() + t.float())
                x = rms(h, Lw["post"], eps)
                co = lin(x, Lw["mcp"]).reshape(B, 2, taps, G)
                cx = conv(x, co[:, 0], Lw["mcb"][0], taps, group, block)
                m = bf(torch.nn.functional.silu(lin(cx, Lw["g"]).float()) * lin(cx, Lw["u"]).float())
                t = conv(lin(m, Lw["d"]), co[:, 1], Lw["mcb"][1], taps, group, block)
            o = rms(bf(h.float() + t.float()), norm, eps)
            eo = u16_to_bf16(ps["o"][bi * B:(bi + 1) * B]).to(dev)
            cos = torch.nn.functional.cosine_similarity(o.float(), eo.float(), dim=-1)
            md = (o.float() - eo.float()).abs().max().item()
            logits = lin(o, head).float()
            lp = torch.log_softmax(logits, dim=-1)
            tv, ti = torch.topk(lp, ps["k"], dim=-1)
            e_ids = torch.from_numpy(ps["ids"][bi * B:(bi + 1) * B].copy()).to(dev)
            e_lp = torch.from_numpy(ps["lp"][bi * B:(bi + 1) * B].copy()).to(dev)
            top1 = (ti[1:, 0] == e_ids[1:, 0]).float().mean().item()
            ovl = np.mean([len(set(ti[j].tolist()) & set(e_ids[j].tolist())) / ps["k"] for j in range(1, B)])
            dlp = (tv[1:, 0] - e_lp[1:, 0]).abs().max().item()
            worst = max(worst, 1.0 - cos[1:].min().item())
            print(f"pass {pi} bank {b} P {P} carry {carry}: hidden cos min {cos.min().item():.5f} max|d| {md:.4f}; "
                  f"slots top-1 agree {top1:.2f}, top-{ps['k']} overlap {ovl:.2f}, top-1 lp max|d| {dlp:.4f}; "
                  f"ref chain {ti[1:, 0].tolist()} engine {e_ids[1:, 0].tolist()}")
    print(f"worst 1 - cos over the slots: {worst:.6f}")


if __name__ == "__main__":
    main()

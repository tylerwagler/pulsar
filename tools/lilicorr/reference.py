#!/usr/bin/env python3
"""reference.py -- a standalone PyTorch reference of NVIDIA's LiLiCorr block drafter (L298), loading the checkpoint's
model.safetensors directly, graded against vllm_dump.py's goldens.

THE DRAFTER (vLLM d9a543c5 / the serving image's 0.32.0rc1.dev190: models/lilicorr.py on qwen3_dflash.py +
qwen3_dflash2.py's grouped-conv layer; v1/worker/gpu/spec_decode/{dflash,dflash2,lilicorr}/speculator.py):

  context    every target step's rows (prefill chunk, or the verify block): the 5 tapped hidden states concatenated
             [n, 20480] -> fc (NVFP4) -> c [n, 4096].  The rows the target accepted (all but the last num_rejected)
             enter each layer's context KV: k = rope(k_norm(k_proj(hidden_norm(c)))), v = v_proj(hidden_norm(c)), at
             the row's position (no layer is run over the context: DFlash projects it straight into every layer).
  anchor     hidden_norm(c[last accepted row]) (zero when the step accepted no row).
  block      8 query rows at P .. P+7, P = last accepted position + 1: embed(bonus token), then 7 x embed(990), the
             TARGET's input embedding.  5 layers, each
                 h = input_layernorm(x (+ residual))                     (vLLM fused add: residual = bf16(x + res))
                 h, d = attention_conv.prepare(h); h = attn(h); h = attention_conv.finish(h, d)
                 h = post_attention_layernorm(h + residual)
                 h, d = mlp_conv.prepare(h); h = mlp(h); h = mlp_conv.finish(h, d)
             attention: GQA 32 q / 8 kv x 128, q/k RMSNorm, neox RoPE theta 1e6, non-causal over the block plus the
             context window (key position > q - 8192), scale 1/sqrt(128).  The grouped dynamic conv (2 taps, group 16):
             coefficients = kernel_projection(h) [8, 2 sides, 2 taps, 256 groups];
             out[t,c] = (base[side,0,c] + d[t,0,c/16]) * x[t,c] + [t%8 >= 1] (base[side,1,c] + d[t,1,c/16]) * x[t-1,c]
             in f32, one bf16 rounding.  MLP: SwiGLU 12288 (NVFP4).  last = norm(h + residual).
  candidates rows 1..7 of last ("pass hidden") -> the drafter's own NVFP4 LM head -> log_softmax over the full vocab ->
             top-8 per slot (ids, log-probs).
  head       LiLiCorrHead (models/lilicorr.py): the 56-candidate lattice -> scores [7][8][8] (slot 0's rows all equal
             the start scores) -> the selector walk: prev = 0; for s in 0..6: j = first argmax of scores[s][prev][:];
             chain[s] = cand_ids[s][j]; prev = j.

NUMERICS: BF16 tensors as vLLM holds them; every BF16-weight linear is F.linear in bf16 (one rounding); an NVFP4
linear is W = e2m1 x e4m3 (exact in bf16) in an f32 GEMM, times the f32 global scale, plus the bias, one bf16 rounding
(vLLM runs Marlin W4A16: same products, its own accumulation order).  The drafter's KV cache is FP8 E4M3 in the
serving config (kv_cache_dtype fp8_e4m3, scale from the dump's meta); --kv bf16 skips that rounding.

    python3 reference.py check  --golden /work/goldens [--prompts code,prose,...]   stage diffs + chain match per round
    python3 reference.py fixture --golden /work/goldens --out /work/head_fixture    head-kernel fixtures (one per prompt)
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

DRAFT = "/mnt/models/nemotron35-super-lilicorr"
TARGET = "/srv/models/nemotron35-super-vl-nvfp4"
TARGET_EMBED = "language_model.backbone.embeddings.weight"
BF = torch.bfloat16

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def nvfp4_dequant(packed, scale, dev):
    """ModelOpt NVFP4: packed u8 [out, in/2], element 2i in the LOW nibble; scale e4m3 [out, in/16].  Returns the
    bf16 product e2m1 x e4m3 (exact) -- the f32 global scale (weight_scale_2) is applied after the GEMM."""
    p = packed.to(dev)
    lo, hi = (p & 0xF).long(), (p >> 4).long()
    vals = torch.stack([E2M1.to(dev)[lo], E2M1.to(dev)[hi]], dim=-1).reshape(p.shape[0], -1)
    s = scale.to(dev).float().repeat_interleave(16, dim=1)
    w = vals * s
    out = w.to(BF)
    assert torch.equal(out.float(), w), "e2m1 x e4m3 must be exact in bf16"
    return out


class Weights:
    def __init__(self, path=DRAFT, dev="cuda"):
        self.dev = dev
        self.t = {}
        self.s2 = {}
        with safe_open(os.path.join(path, "model.safetensors"), "pt", device="cpu") as f:
            names = list(f.keys())
            for n in names:
                if n.endswith(".weight_scale") or n.endswith(".weight_scale_2") or n.endswith(".input_scale"):
                    continue
                if n.endswith(".weight") and n + "_scale" in names:
                    base = n[: -len(".weight")]
                    self.t[n] = nvfp4_dequant(f.get_tensor(n), f.get_tensor(n + "_scale"), dev)
                    self.s2[n] = float(f.get_tensor(n + "_scale_2").float().item())
                else:
                    self.t[n] = f.get_tensor(n).to(dev)
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        print(f"[reference] {len(self.t)} tensors, {len(self.s2)} NVFP4 linears", flush=True)

    def lin(self, x, name, bias=True):
        """Linear by name: NVFP4 (f32 GEMM of the exact bf16 product, x s2, + bias, one rounding) or BF16 F.linear."""
        w = self.t[name + ".weight"]
        b = self.t.get(name + ".bias") if bias else None
        if name + ".weight" in self.s2:
            y = x.float() @ w.float().t() * self.s2[name + ".weight"]
            if b is not None:
                y = y + b.float()
            return y.to(x.dtype)
        return F.linear(x, w, b)

    def as_f32(self):
        """A copy whose every tensor is f32: the head evaluated without bf16 roundings (the noise floor's yardstick)."""
        c = object.__new__(Weights)
        c.dev, c.s2, c.cfg = self.dev, self.s2, self.cfg
        c.t = {k: (v.float() if v.is_floating_point() else v) for k, v in self.t.items()}
        return c


def rms(x, w, eps=1e-5):
    """vLLM's rms_norm (ir/ops/layernorm.py): f32 normalize, bf16 cast, times the bf16 weight."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(BF) * w


def add_rms(x, res, w, eps=1e-5):
    """vLLM's fused_add_rms_norm: residual = bf16(x + res); normalize the f32 sum."""
    s = x.float() + res.float()
    r = s.to(BF)
    y = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(BF) * w, r


class Rope:
    def __init__(self, dev, dim=128, base=1e6, max_pos=1 << 17):
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        t = torch.arange(max_pos, dtype=torch.float)
        fr = torch.einsum("i,j->ij", t, inv)
        self.cos = fr.cos().to(BF).to(dev)
        self.sin = fr.sin().to(BF).to(dev)

    def __call__(self, x, pos):
        """neox rotation of [n, heads*128] bf16 at positions [n]; bf16 arithmetic per op, as vLLM's kernel."""
        n = x.shape[0]
        x = x.view(n, -1, 128)
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        x1, x2 = x[..., :64], x[..., 64:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).reshape(n, -1)


def grouped_conv(x, delta, base, block=8, group=16):
    """qwen3_dflash2's triton kernel: f32 accumulate of (base + delta) * x over 2 taps within each 8-row block."""
    n, c = x.shape
    taps = base.shape[0]
    xf = x.float()
    g = torch.arange(c, device=x.device) // group
    pos = torch.arange(n, device=x.device) % block
    acc = (base[0].float() + delta[:, 0, :].float()[:, g]) * xf
    for tap in range(1, taps):
        coef = base[tap].float() + delta[:, tap, :].float()[:, g]
        sh = torch.zeros_like(xf)
        sh[tap:] = xf[:-tap]
        acc = acc + torch.where((pos >= tap)[:, None], coef * sh, torch.zeros_like(acc))
    return acc.to(BF)


def fp8_round(x, scale):
    return (x.float() / scale).to(torch.float8_e4m3fn).float().mul(scale).to(BF)


class Drafter:
    """The DFlash backbone with its context KV (one request)."""

    def __init__(self, W, embed, kv="fp8", kv_scales=None, max_pos=1 << 16):
        self.W, self.embed, self.dev = W, embed, W.dev
        cfg = W.cfg
        self.L = cfg["num_hidden_layers"]
        self.win = cfg["dflash_config"]["swa_window_size"]
        self.rope = Rope(self.dev)
        self.kv = kv
        self.kv_scales = kv_scales or {}
        self.K = torch.zeros(self.L, max_pos, 1024, dtype=BF, device=self.dev)
        self.V = torch.zeros_like(self.K)
        self.filled = torch.zeros(max_pos, dtype=torch.bool, device=self.dev)

    def kvq(self, i, k, v):
        if self.kv != "fp8":
            return k, v
        s = self.kv_scales.get(f"L{i}", {"k_scale": 1.0, "v_scale": 1.0})
        return fp8_round(k, s["k_scale"]), fp8_round(v, s["v_scale"])

    def knorm(self, i, k):
        n = k.shape[0]
        return rms(k.view(n, 8, 128), self.W.t[f"layers.{i}.self_attn.k_norm.weight"]).view(n, -1)

    def add_context(self, taps, pos, n_valid):
        """fc the step's rows; store the accepted ones' K/V in every layer.  Returns fc_out (all rows)."""
        c = self.W.lin(taps, "fc", bias=False)
        if n_valid:
            hn = rms(c[:n_valid], self.W.t["hidden_norm.weight"])
            p = pos[:n_valid]
            for i in range(self.L):
                k = F.linear(hn, self.W.t[f"layers.{i}.self_attn.k_proj.weight"])
                v = F.linear(hn, self.W.t[f"layers.{i}.self_attn.v_proj.weight"])
                k = self.rope(self.knorm(i, k), p)
                k, v = self.kvq(i, k, v)
                self.K[i, p], self.V[i, p] = k, v
            self.filled[p] = True
        return c

    def attention(self, i, q, k, v, qpos):
        """block queries over the context window + the block (non-causal), GQA 32/8."""
        P = int(qpos[0])
        lo = max(0, P - self.win + 1)          # the first query's window; later queries mask the left edge
        ck, cv = self.K[i, lo:P], self.V[i, lo:P]
        assert bool(self.filled[lo:P].all()), "context hole"
        kk = torch.cat([ck, k], 0).view(-1, 8, 128)
        vv = torch.cat([cv, v], 0).view(-1, 8, 128)
        kpos = torch.cat([torch.arange(lo, P, device=self.dev), qpos])
        mask = (qpos[:, None] - kpos[None, :]) < self.win            # key > q - win (FlashInfer window_left = win - 1)
        qq = q.view(8, 32, 128).transpose(0, 1)                          # [32, 8, 128]
        kk = kk.repeat_interleave(4, dim=1).transpose(0, 1)              # [32, n, 128]
        vv = vv.repeat_interleave(4, dim=1).transpose(0, 1)
        s = (qq.float() @ kk.float().transpose(-1, -2)) / math.sqrt(128.0)
        s = s.masked_fill(~mask[None], float("-inf"))
        o = torch.softmax(s, -1) @ vv.float()
        return o.transpose(0, 1).reshape(8, 4096).to(BF)

    def conv(self, i, which, h, d, side):
        return grouped_conv(h, d, self.W.t[f"layers.{i}.{which}.base_kernel"][side])

    def conv_prepare(self, i, which, h):
        d = F.linear(h, self.W.t[f"layers.{i}.{which}.kernel_projection.weight"]).view(8, 2, 2, 256)
        return self.conv(i, which, h, d[:, 0], 0), d[:, 1]

    def layer(self, i, x, res, qpos, cap=None):
        t = self.W.t
        pre = f"layers.{i}."
        if res is None:
            res = x
            h = rms(x, t[pre + "input_layernorm.weight"])
        else:
            h, res = add_rms(x, res, t[pre + "input_layernorm.weight"])
        h, d = self.conv_prepare(i, "attention_conv", h)
        q = F.linear(h, t[pre + "self_attn.q_proj.weight"])
        k = F.linear(h, t[pre + "self_attn.k_proj.weight"])
        v = F.linear(h, t[pre + "self_attn.v_proj.weight"])
        q = rms(q.view(8, 32, 128), t[pre + "self_attn.q_norm.weight"]).view(8, -1)
        q = self.rope(q, qpos)
        k = self.rope(self.knorm(i, k), qpos)
        if cap is not None:
            cap[f"L{i}.q"], cap[f"L{i}.k"], cap[f"L{i}.v"] = q, k, v
        kq, vq = self.kvq(i, k, v)
        a = self.attention(i, q, kq, vq, qpos)
        if cap is not None:
            cap[f"L{i}.attn"] = a
        h = F.linear(a, t[pre + "self_attn.o_proj.weight"])
        h = self.conv(i, "attention_conv", h, d, 1)
        h, res = add_rms(h, res, t[pre + "post_attention_layernorm.weight"])
        h, d = self.conv_prepare(i, "mlp_conv", h)
        g = self.W.lin(h, pre + "mlp.gate_proj", bias=False)
        u = self.W.lin(h, pre + "mlp.up_proj", bias=False)
        h = (F.silu(g.float()) * u.float()).to(BF)
        h = self.W.lin(h, pre + "mlp.down_proj", bias=False)
        h = self.conv(i, "mlp_conv", h, d, 1)
        return h, res

    def block(self, q_ids, qpos, cap=None):
        x = self.embed[q_ids]
        res = None
        for i in range(self.L):
            if cap is not None:
                cap[f"L{i}.in_hidden"] = x
            x, res = self.layer(i, x, res, qpos, cap)
            if cap is not None:
                cap[f"L{i}.out_hidden"], cap[f"L{i}.out_residual"] = x, res
        last, _ = add_rms(x, res, self.W.t["norm.weight"])
        return last

    def candidates(self, pass_hidden, k=8):
        logits = self.W.lin(pass_hidden, "lm_head", bias=False)
        lse = torch.logsumexp(logits.float(), -1, keepdim=True)
        v, ids = torch.topk(logits, k, dim=-1)
        return ids, v.float() - lse


class Head:
    """LiLiCorrHead.score/forward (models/lilicorr.py), torch ops in the order vLLM runs them."""

    def __init__(self, W):
        self.W = W
        dc = W.cfg["dflash_config"]
        self.topk, self.H, self.nh = dc["lilicorr_candidate_topk"], dc["lilicorr_hidden_size"], dc["lilicorr_num_heads"]
        self.eps_v, self.scale = dc["lilicorr_vector_eps"], dc["lilicorr_logit_scale"]
        self.block = W.cfg["block_size"]
        t = W.t
        p = "lilicorr."
        ns = self.block - 1
        slot = torch.arange(ns, device=W.dev).repeat_interleave(self.topk)
        rel = slot.view(-1, 1) - slot.view(1, -1)
        bias = t[p + "relative_slot_bias"][:, rel + self.block - 1]
        same = (slot.view(-1, 1) == slot.view(1, -1))
        self.attn_bias = (bias + same.unsqueeze(0).to(bias.dtype) * t[p + "same_slot_bias"].view(-1, 1, 1)).contiguous()
        w = t[p + "factor_input_proj.weight"]
        self.w_self, self.w_anchor, self.w_cross = w[:, :1024].contiguous(), w[:, 1024:2048].contiguous(), w[:, 2048:].contiguous()
        self.edge_w = torch.cat([t[p + "out_head.weight"], t[p + "in_head.weight"]], 0).contiguous()
        self.edge_b = torch.cat([t[p + "out_head.bias"], t[p + "in_head.bias"]], 0).contiguous()
        self.rank_frac = (torch.arange(self.topk, device=W.dev, dtype=torch.float32) / (self.topk - 1)).view(1, 1, -1)
        self.is_top1 = torch.zeros(1, 1, self.topk, device=W.dev)
        self.is_top1[..., 0] = 1.0

    def lin(self, x, n, bias=True):
        return self.W.lin(x, "lilicorr." + n, bias)

    def mlp(self, x, n):
        return self.lin(F.silu(self.lin(x, n + ".up_proj")), n + ".down_proj")

    def norm(self, x, n):
        return F.rms_norm(x, (x.shape[-1],), self.W.t["lilicorr." + n + ".weight"], 1e-5)

    def score(self, tok_emb, logp, pass_hidden, anchor_hidden, anchor_valid, cap=None):
        """tok_emb [B,7,8,4096] bf16, logp [B,7,8] f32, pass_hidden [B,7,4096], anchor_hidden [B,4096], valid [B]."""
        t = self.W.t
        B, ns, k = logp.shape
        tok = self.lin(tok_emb, "token_proj")
        pas = self.lin(pass_hidden, "pass_hidden_proj").unsqueeze(-2)
        lp = logp.float()
        feats = torch.stack([lp, lp.exp(), lp - lp.max(-1, keepdim=True).values, self.rank_frac.expand_as(lp),
                             self.is_top1.expand_as(lp)], -1)
        h = tok + pas
        fn = F.layer_norm(feats.to(tok.dtype), (5,), t["lilicorr.feature_norm.weight"], t["lilicorr.feature_norm.bias"])
        h = h + self.mlp(fn, "feature_mlp")
        h = h + t["lilicorr.slot_embedding"][:, 0, :ns]
        h = h + t["lilicorr.rank_embedding"][:, 0]
        h = h.reshape(B, ns * k, self.H)
        a = self.lin(anchor_hidden, "context_proj")
        a = a * anchor_valid[:, None].to(a.dtype)
        n = ns * k
        bias = self.attn_bias[None, :, :n, :n]
        for li in range(2):
            p = f"layers.{li}."
            x = self.norm(h, p + "attn_norm")
            qkv = F.linear(x, t["lilicorr." + p + "attn.in_proj_weight"], t["lilicorr." + p + "attn.in_proj_bias"])
            q, kk, v = qkv.chunk(3, -1)
            sh = (B, n, self.nh, self.H // self.nh)
            q, kk, v = (z.reshape(sh).transpose(1, 2) for z in (q, kk, v))
            o = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias)
            h = h + self.lin(o.transpose(1, 2).reshape(B, n, self.H), p + "attn.out_proj")
            h = h + self.mlp(self.norm(h, p + "mlp_norm"), p + "mlp")
            if cap is not None:
                cap[f"head.L{li}"] = h
        h = self.norm(h, "output_norm").reshape(B, ns, k, self.H)
        a = self.norm(a, "anchor_norm")
        ar = a[:, None, None, :]
        pre = F.linear(h, self.w_self, t["lilicorr.factor_input_proj.bias"])
        pre = pre + F.linear(ar, self.w_anchor)
        pre = pre + F.linear(h * ar, self.w_cross)
        fh = F.silu(pre)
        edges = F.linear(fh, self.edge_w, self.edge_b)
        out_v, in_v = F.normalize(edges.unflatten(-1, (2, 1024)), dim=-1, eps=self.eps_v).unbind(-2)
        aout = F.normalize(self.lin(a, "anchor_out_head"), dim=-1, eps=self.eps_v)
        start = (aout[:, None, :] * in_v[:, 0]).sum(-1)
        pairs = torch.matmul(out_v[:, :-1], in_v[:, 1:].transpose(-1, -2))
        return start, pairs

    def forward(self, *a, cap=None):
        start, pairs = self.score(*a, cap=cap)
        start = self.scale * start.float()
        pairs = self.scale * pairs.float()
        first = start[:, None, None, :].expand(-1, 1, self.topk, -1)
        return torch.cat([first, pairs], 1), start, pairs


def walk(scores, cand_ids):
    """_selector_walk_kernel at temperature 0: prev = 0, first argmax (tl.max tie-break left)."""
    B, ns, k, _ = scores.shape
    chain = torch.zeros(B, ns, dtype=torch.long)
    rows = torch.zeros(B, ns, dtype=torch.long)
    for b in range(B):
        prev = 0
        for s in range(ns):
            row = scores[b, s, prev].tolist()
            j = max(range(k), key=lambda i: (row[i], -i))
            chain[b, s], rows[b, s] = int(cand_ids[b, s, j]), j
            prev = j
    return chain, rows


def load_embed(dev):
    idx = json.load(open(os.path.join(TARGET, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(TARGET, idx[TARGET_EMBED]), "pt", device="cpu") as f:
        e = f.get_tensor(TARGET_EMBED)
    assert e.dtype == BF, e.dtype
    return e.to(dev)


def md(a, b):
    """max |a - b| relative to max |b| (the stage's scale), so stages of any magnitude compare."""
    b = b.float().to(a.device)
    return float((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-30))


def margin(scores, rows):
    """The walk's smallest decision margin: per slot, best minus second best of the row the walk read."""
    m = float("inf")
    for s in range(scores.shape[0]):
        prev = 0 if s == 0 else int(rows[s - 1])
        v = scores[s, prev].float().sort(descending=True).values
        m = min(m, float(v[0] - v[1]))
    return m


def rounds_of(g):
    return sorted({int(k[1:4]) for k in g})


def check(a):
    dev = a.device
    W = Weights(dev=dev)
    head = Head(W)
    head32 = Head(W.as_f32())
    embed = load_embed(dev)
    total = {"rounds": 0, "chain_head": 0, "chain_full": 0, "slots_cand_ids_equal": 0, "slots": 0}
    for gpath in sorted(glob.glob(os.path.join(a.golden, "*.safetensors"))):
        name = os.path.basename(gpath)[: -len(".safetensors")]
        if a.prompts and name not in a.prompts.split(","):
            continue
        meta = json.load(open(gpath[: -len(".safetensors")] + ".json"))
        g = load_file(gpath, device=dev)
        dr = Drafter(W, embed, kv=a.kv, kv_scales=meta["attn"])
        worst = {}
        n_ok_h = n_ok_f = n_slot = 0
        d_vllm32 = d_ref32 = 0.0
        R = rounds_of(g)
        for r in R:
            G = {k[5:]: v for k, v in g.items() if k.startswith(f"r{r:03d}.")}
            n = G["ctx_taps"].shape[0]
            nv = n - int(G["num_rejected"][0])
            fc = dr.add_context(G["ctx_taps"], G["ctx_pos"], nv)
            stage = {"fc_out": md(fc, G["fc_out"])}
            if nv:
                stage["anchor"] = md(rms(fc[nv - 1], W.t["hidden_norm.weight"]), G["anchor_hidden"])
            cap = {}
            last = dr.block(G["q_ids"], G["q_pos"], cap)
            for key in cap:
                if key in G:
                    stage[key] = md(cap[key], G[key])
            stage["last_hidden"] = md(last, G["last_hidden"])
            # teacher-forced: each layer on vLLM's own inputs (its error alone)
            for i in range(dr.L):
                res = G.get(f"L{i}.in_residual")
                h, rr = dr.layer(i, G[f"L{i}.in_hidden"], res, G["q_pos"])
                stage[f"tf.L{i}.out_hidden"] = md(h, G[f"L{i}.out_hidden"])
                stage[f"tf.L{i}.out_residual"] = md(rr, G[f"L{i}.out_residual"])
            ids, lp = dr.candidates(last[1:])
            tf_ids, tf_lp = dr.candidates(G["pass_hidden"])
            n_slot += int((tf_ids.cpu() == G["cand_ids"].cpu()).all(-1).sum())
            stage["tf.cand_logp"] = md(tf_lp, G["cand_logp"])
            # the head on vLLM's own inputs: the kernel's contract
            args = (G["tok_emb"][None], G["cand_logp"][None], G["pass_hidden"][None], G["anchor_hidden"][None],
                    G["anchor_valid"][:1])
            sc, st, pr = head.forward(*args)
            sc32, _, _ = head32.forward(*(x.float() if x.is_floating_point() else x for x in args))
            stage["head_start"] = md(st[0], G["head_start"] * head.scale)
            stage["scores"] = md(sc[0], G["scores"])
            d_vllm32 = max(d_vllm32, float((G["scores"] - sc32[0]).abs().max()))
            d_ref32 = max(d_ref32, float((sc[0] - sc32[0]).abs().max()))
            ch, _ = walk(sc.cpu(), G["cand_ids"][None].cpu())
            _, vrows = walk(G["scores"][None].cpu(), G["cand_ids"][None].cpu())
            okh = torch.equal(ch[0], G["draft"].cpu())
            n_ok_h += okh
            # the full reference: its own candidates, embeddings, pass rows and anchor
            fa = G["anchor_hidden"] if not nv else rms(fc[nv - 1], W.t["hidden_norm.weight"])
            fv = torch.tensor([bool(nv)], device=dev)
            scf, _, _ = head.forward(embed[ids][None], lp[None], last[1:][None], (fa * fv.to(BF))[None], fv)
            chf, _ = walk(scf.cpu(), ids[None].cpu())
            okf = torch.equal(chf[0], G["draft"].cpu())
            n_ok_f += okf
            for k_, v in stage.items():
                worst[k_] = max(worst.get(k_, 0.0), v)
            if a.verbose or not okh:
                print(f"  {name} r{r:03d} n={n} nv={nv} P={int(G['q_pos'][0])} head_chain={'ok' if okh else 'MISMATCH'} "
                      f"full_chain={'ok' if okf else 'MISMATCH'} vllm_margin={margin(G['scores'].cpu(), vrows[0]):.4f} "
                      f"vllm={G['draft'].tolist()} head={ch[0].tolist()} full={chf[0].tolist()}", flush=True)
                if a.verbose:
                    print("     " + " ".join(f"{k_}={v:.3g}" for k_, v in stage.items()), flush=True)
        print(f"[reference] {name}: {len(R)} rounds  head-chain {n_ok_h}/{len(R)}  full-chain {n_ok_f}/{len(R)}  "
              f"slots with vLLM's exact top-8 order (on vLLM's pass rows) {n_slot}/{7 * len(R)}  (kv {a.kv})", flush=True)
        print(f"   head scores vs an all-f32 head on the same inputs: vLLM {d_vllm32:.4f}, reference {d_ref32:.4f} "
              f"(logits, max abs)", flush=True)
        print("   max |diff| / max |vLLM| per stage: " + " ".join(f"{k_}={v:.2e}" for k_, v in worst.items()),
              flush=True)
        total["rounds"] += len(R)
        total["chain_head"] += n_ok_h
        total["chain_full"] += n_ok_f
        total["slots_cand_ids_equal"] += n_slot
        total["slots"] += 7 * len(R)
    print(f"[reference] TOTAL {json.dumps(total)}", flush=True)
    return 0 if total["chain_head"] == total["rounds"] else 1


def fixture(a):
    """Per prompt <out>/<prompt>.safetensors: the head kernel's inputs for every round (vLLM's), a compact embedding
    table (the rounds' unique candidate rows; cand_local indexes it), vLLM's scores / chain, and this reference's."""
    dev = a.device
    W = Weights(dev=dev)
    head = Head(W)
    embed = load_embed(dev)
    os.makedirs(a.out, exist_ok=True)
    for gpath in sorted(glob.glob(os.path.join(a.golden, "*.safetensors"))):
        name = os.path.basename(gpath)[: -len(".safetensors")]
        g = load_file(gpath, device=dev)
        R = rounds_of(g)
        cat = lambda k: torch.stack([g[f"r{r:03d}.{k}"] for r in R])
        ids, lp, ph = cat("cand_ids"), cat("cand_logp"), cat("pass_hidden")
        ah, av = cat("anchor_hidden"), cat("anchor_valid")[:, 0]
        uniq, local = torch.unique(ids, return_inverse=True)
        assert torch.equal(embed[ids], cat("tok_emb")), "the dump's token_embeddings are the target embedding rows"
        sc, _, _ = head.forward(embed[ids], lp, ph, ah, av)
        ch, rows = walk(sc.cpu(), ids.cpu())
        out = {"cand_ids": ids.int().contiguous(), "cand_local": local.int().contiguous(), "cand_logp": lp.float(),
               "emb_rows": embed[uniq].contiguous(), "pass_hidden": ph.contiguous(), "anchor_hidden": ah.contiguous(),
               "anchor_valid": av.to(torch.uint8), "vllm_scores": cat("scores").float(), "vllm_chain": cat("draft").int(),
               "ref_scores": sc.float().contiguous(), "ref_chain": ch.int(), "ref_rows": rows.int()}
        save_file({k: v.cpu().contiguous() for k, v in out.items()}, os.path.join(a.out, f"{name}.safetensors"))
        print(f"[reference] fixture {name}: {len(R)} rounds, {len(uniq)} unique candidates", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "fixture"])
    p.add_argument("--golden", required=True)
    p.add_argument("--out")
    p.add_argument("--prompts", default="")
    p.add_argument("--kv", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", default="cuda", help="cpu: a fixture without the GPU (the reference chain then carries CPU torch numerics)")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return check(a) if a.mode == "check" else fixture(a)


if __name__ == "__main__":
    sys.exit(main())

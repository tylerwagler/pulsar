#!/usr/bin/env python3
"""L292 DFlash2 acceptance gate: CPU replay of incoai/Qwen3.8-Flash-Next-DFlash2 on pulsar's own taps and greedy replies.

Transcribed from z-lab/dflash model.py @07ebd93 (DFlash2DraftModel, dflash_generate) + the drafter's config.json:
  context    taps [n, 5 x 2560] (HC 4-stream mean of layers 5/15/25/35/45's completed outputs, bf16) -> fc -> hidden_norm
             -> per layer k = rope(k_norm(k_proj(c))), v = v_proj(c) at the row's position (no layer runs over context).
  block      B rows at p .. p+B-1: embed(anchor S[p]), then embed(mask 248077) x (B-1), the TARGET's embedding.
             5 layers: h = h + conv_finish(attn(conv_prepare(input_ln(h)))); h = h + conv_finish(mlp(conv_prepare(
             post_ln(h)))).  Attention 32q/8kv x 128, q/k RMSNorm, neox RoPE theta 1e4, NON-causal (is_causal false),
             sliding window 2048 both ways over context + block.
  head       rows 1..B-1 of norm(h) -> target lm_head -> top-16 -> CandidateSelector walk from the anchor id.
SERVED rounds (greedy, lossless): anchor at p (taps 0..p-1 known), drafts for p+1..p+B-1, accept up to the first
mismatch vs pulsar's own reply S, next anchor p + accepted + 1.

    replay.py run --fixture code [--blocks 8,6,4] [--heads sel,argmax,sel-dv]
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
TAPS = [5, 15, 25, 35, 45]
H, NH, NKV, HD, WIN, MASK, V = 2560, 32, 8, 128, 2048, 248077, 248320
EPS = 1e-6
torch.set_grad_enabled(False)


def bf16_round(x):
    return x.to(torch.bfloat16).float()


def load_taps(d, n):
    """[n, 12800] f32 (bf16 values): per layer, the 4-stream mean of the dumped streams, rounded to bf16 as
    SGLang's .mean(dim=-2) on a bf16 tensor rounds it."""
    feats = []
    for il in TAPS:
        rows = torch.full((n, 4, H), float("nan"))
        for f in os.listdir(d):
            m = re.match(rf"t_qwen_h-{il}_pos(\d+)(?:_c(\d+))?\.bin$", f)
            if not m or int(m[2] or 1) != 1:
                continue
            a = np.fromfile(os.path.join(d, f), dtype=np.float32).reshape(-1, 4, H)
            last = int(m[1])
            rows[last - a.shape[0] + 1:last + 1] = torch.from_numpy(a)
        assert not torch.isnan(rows).any(), f"layer {il}: positions missing"
        assert torch.equal(bf16_round(rows), rows), "streams are bf16 values"
        feats.append(bf16_round(rows.mean(dim=1)))
    return torch.cat(feats, dim=-1)


class Drafter:
    def __init__(self, threads=None):
        if threads:
            torch.set_num_threads(threads)
        self.cfg = json.load(open(os.path.join(HERE, "drafter_config.json")))
        assert self.cfg["dflash_config"]["target_layer_ids"] == TAPS and not self.cfg["is_causal"]
        assert self.cfg["sliding_window"] == WIN and self.cfg["dflash_config"]["mask_token_id"] == MASK
        self.w = {}
        with safe_open(os.path.join(HERE, "drafter.safetensors"), "pt") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k).float()
        mm = lambda p: torch.from_numpy(np.fromfile(os.path.join(HERE, p), dtype=np.uint16).astype(np.int32) << 16
                                        ).view(torch.float32).view(V, H)
        self.embed = mm("embed.bf16")
        self.head = mm("lm_head.bf16")
        theta = self.cfg["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        self.topk = self.cfg["dflash_config"]["selector_top_k"]
        self.gs = self.cfg["dflash_config"]["conv_group_size"]

    def L(self, i, n):
        return self.w[f"layers.{i}.{n}"]

    @staticmethod
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

    def rope(self, x, pos):  # x [.., T, D], pos [T]
        fr = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat([fr, fr], -1)
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def context(self, taps):
        """Per layer the context K [8, n, 128] (normed + roped) and V [8, n, 128] for positions 0..n-1."""
        c = self.rms(taps @ self.w["fc.weight"].T, self.w["hidden_norm.weight"])
        pos = torch.arange(c.shape[0])
        kv = []
        for i in range(5):
            k = (c @ self.L(i, "self_attn.k_proj.weight").T).view(-1, NKV, HD)
            k = self.rms(k, self.L(i, "self_attn.k_norm.weight")).transpose(0, 1)
            v = (c @ self.L(i, "self_attn.v_proj.weight").T).view(-1, NKV, HD).transpose(0, 1)
            kv.append((self.rope(k, pos).contiguous(), v.contiguous()))
        return kv

    def conv(self, x, dyn, base):
        """z-lab _grouped_dynamic_convolve over the block rows: out[t] = sum_o (base[o] + dyn[t, o, group]) * x[t-o]."""
        T = x.shape[0]
        g = H // self.gs
        out = torch.zeros_like(x)
        for o in range(base.shape[0]):
            vals = x if o == 0 else F.pad(x[:-o], (0, 0, o, 0))
            d = dyn[:, o, :].repeat_interleave(self.gs, dim=-1)  # [T, H]
            out = out + base[o] * vals + d * vals
        return out

    def conv_prepare(self, i, which, x):
        kp = self.L(i, f"{which}.kernel_projection.weight")
        base = self.L(i, f"{which}.base_kernel")
        K = base.shape[1]
        dyn = (x @ kp.T).view(x.shape[0], 2, K, H // self.gs)
        return self.conv(x, dyn[:, 0], base[0]), (dyn[:, 1], base[1])

    def block(self, kv, p, ids):
        """The draft forward for anchor position p: ids [B] (anchor + masks) at positions p .. p+B-1; context = rows
        0..p-1 of kv.  Returns norm(h) [B, H]."""
        B = len(ids)
        qpos = torch.arange(p, p + B)
        h = self.embed[torch.tensor(ids)]
        lo = max(0, p - (WIN - 1))               # the first query (p) sees k > p - WIN
        kpos = torch.cat([torch.arange(lo, p), qpos])
        vis = ((qpos[:, None] - kpos[None]) < WIN) & ((kpos[None] - qpos[:, None]) < WIN)
        bias = torch.zeros(vis.shape).masked_fill(~vis, float("-inf"))
        for i in range(5):
            res = h
            x = self.rms(h, self.L(i, "input_layernorm.weight"))
            x, fin = self.conv_prepare(i, "attention_conv", x)
            q = (x @ self.L(i, "self_attn.q_proj.weight").T).view(B, NH, HD)
            q = self.rope(self.rms(q, self.L(i, "self_attn.q_norm.weight")).transpose(0, 1), qpos)
            kb = (x @ self.L(i, "self_attn.k_proj.weight").T).view(B, NKV, HD)
            kb = self.rope(self.rms(kb, self.L(i, "self_attn.k_norm.weight")).transpose(0, 1), qpos)
            vb = (x @ self.L(i, "self_attn.v_proj.weight").T).view(B, NKV, HD).transpose(0, 1)
            kc, vc = kv[i]
            k = torch.cat([kc[:, lo:p], kb], 1).repeat_interleave(NH // NKV, 0)
            v = torch.cat([vc[:, lo:p], vb], 1).repeat_interleave(NH // NKV, 0)
            a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(HD) + bias, -1) @ v
            o = a.transpose(0, 1).reshape(B, NH * HD) @ self.L(i, "self_attn.o_proj.weight").T
            h = res + self.conv(o, *fin)
            res = h
            x = self.rms(h, self.L(i, "post_attention_layernorm.weight"))
            x, fin = self.conv_prepare(i, "mlp_conv", x)
            x = (F.silu(x @ self.L(i, "mlp.gate_proj.weight").T) * (x @ self.L(i, "mlp.up_proj.weight").T)) \
                @ self.L(i, "mlp.down_proj.weight").T
            h = res + self.conv(x, *fin)
        return self.rms(h, self.w["norm.weight"])

    def propose(self, hidden, anchor, vocab_mask=None):
        """Greedy chains for the B-1 draft rows: the selector walk ('sel') and the plain per-row argmax ('argmax');
        vocab_mask restricts the head to a draft vocabulary first."""
        logits = hidden @ self.head.T
        if vocab_mask is not None:
            logits = logits.masked_fill(~vocab_mask, float("-inf"))
        unary, cand = torch.topk(logits, self.topk, dim=-1)
        hp = hidden @ self.w["candidate_selector.hidden_projection.weight"].T
        pc, sc = self.w["candidate_selector.predecessor_codebook"], self.w["candidate_selector.successor_codebook"]
        pred, path, margins = anchor, [], []
        for j in range(hidden.shape[0]):
            s = unary[j] + (sc[cand[j]] @ (pc[pred] * hp[j]))
            top2 = torch.topk(s, 2).values
            idx = int(torch.argmax(s))
            pred = int(cand[j, idx])
            path.append(pred)
            margins.append(float(top2[0] - top2[1]))
        return {"sel": path, "argmax": cand[:, 0].tolist(), "top1p": torch.softmax(logits, -1).max(-1).values.tolist()}


def served(S, P, B, draft_fn):
    """Served greedy rounds over the reply: returns per-round accepted counts (only rounds whose whole block lies
    inside S, so every draft is gradable)."""
    acc = []
    p = P
    while p + B - 1 <= len(S) - 1:
        d = draft_fn(p, B)
        a = 0
        while a < B - 1 and d[a] == S[p + 1 + a]:
            a += 1
        acc.append(a)
        p += a + 1
    return acc


def summary(acc, B):
    R = len(acc)
    tpr = sum(a + 1 for a in acc) / R
    keep = [sum(1 for a in acc if a >= d) / R for d in range(1, B)]
    return {"rounds": R, "tokens_per_round": tpr, "p_kept_ge_d": keep}


def draft_vocab_mask():
    src = open(os.path.join(HERE, "qwen_draft_vocab.inc")).read()
    body = src.split("k_qwen_draft_extra[] = {", 1)[1].split("};", 1)[0]
    extra = [int(x) for x in re.findall(r"\d+", body)][:16384]
    m = torch.zeros(V, dtype=torch.bool)
    m[:65536] = True
    m[torch.tensor(extra)] = True
    m[248044:] = True
    assert int(m.sum()) == 82196, int(m.sum())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--ids", required=True, help="prompt + reply ids (int32)")
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--blocks", default="8,6,4")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = np.fromfile(a.ids, dtype=np.int32).tolist()
    P = a.prompt_len
    D = Drafter(a.threads)
    taps = load_taps(a.taps, len(S))
    kv = D.context(taps)
    dv = draft_vocab_mask()
    cache = {}

    def drafts(p, B, variant):
        key = (p, B)
        if key not in cache:
            hid = D.block(kv, p, [S[p]] + [MASK] * (B - 1))[1:]
            r = D.propose(hid, S[p])
            r["sel-dv"] = D.propose(hid, S[p], dv)["sel"]
            cache[key] = r
        return cache[key][variant]

    res = {"fixture": a.fixture, "prompt_len": P, "reply_len": len(S) - P}
    for B in [int(x) for x in a.blocks.split(",")]:
        for variant in ("sel", "argmax", "sel-dv"):
            acc = served(S, P, B, lambda p, B: drafts(p, B, variant))
            s = summary(acc, B)
            s["acc"] = acc
            res[f"B{B}-{variant}"] = s
            print(f"{a.fixture} B{B} {variant:7s}: {s['rounds']} rounds, {s['tokens_per_round']:.3f} tok/round, "
                  f"P(kept>=d) " + " ".join(f"{x:.3f}" for x in s["p_kept_ge_d"]), flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()

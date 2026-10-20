#!/usr/bin/env python3
"""ref.py -- the block drafter's reference grade (L298): a PyTorch transcription of the drafter (vLLM's
qwen3_dflash.py / qwen3_dflash2.py / lilicorr.py backbone: the DFlash block forward with the DFlash2 grouped
convolution, an fp8_e4m3 KV cache at unit scale as vLLM served it) graded two ways:

  --golden FILE.safetensors   against vLLM's own tensors (the head agent's goldens, tools/lilicorr/reference.py: per
                              round ctx_taps / ctx_pos / num_rejected, fc_out, L{i}.{q,k,v,attn,out_hidden,
                              out_residual}, last_hidden, cand_ids / cand_logp, q_ids / q_pos) -- is THIS transcription
                              the model vLLM runs?
  --dump DIR                  against the engine (PULSAR_BLOCK_DUMP=<dir>, src/engine/spec_block.cpp: every absorbed
                              context row's taps, the first draft passes' final hidden and top-k) -- does the engine
                              compute what the transcription does?  The context is the engine's own taps, so the grade
                              isolates the drafter.

The reference reads the drafter's own ModelOpt checkpoint (W4A16 NVFP4 linears dequantized exactly: e2m1 x e4m3 in
f32, the f32 global scale on the GEMM's output) and the target's input embedding from the target's HF checkpoint.
Arithmetic follows vLLM's: bf16 activations between ops, f32 inside each op.

(L299) DFlash2 (z-lab's DFlash2DraftModel, config architectures [DFlash2DraftModel]; tools/dflash2's replay is the
f32 transcription it was graded against): the same backbone with bf16 K / V (no fp8 cache), rope_parameters' theta,
the head the target's lm_head and the CandidateSelector's walk.  --container DIR reads every drafter tensor and the
target's input embedding from the PULSAR CONTAINER the engine served (nvfp4 entries decoded exactly,
tools/container/producers.nvfp4_decode; the head's draft-vocabulary rows and ids as stored), so the --dump grade
isolates the engine's arithmetic from the container's quantisation; the selector's chain is graded too.

    ~/venvs/tf/bin/python tools/blockdraft/ref.py --drafter /mnt/models/nemotron35-super-lilicorr \
        --target-hf /mnt/models/nemotron35-super-vl-bf16 (--golden F | --dump DIR) [--rounds N] [--device cpu|cuda]
    ~/venvs/tf/bin/python tools/blockdraft/ref.py --drafter /mnt/models/qwen38fn-dflash2 --container CONTAINER \
        --dump DIR [--rounds N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct

import numpy as np
import torch
from safetensors import safe_open

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import producers as PR  # noqa: E402

BF16 = torch.bfloat16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def bf(x):
    return x.to(BF16)


def e4m3(x):
    """a bf16 tensor through the fp8_e4m3 cache at unit scale (satfinite) and back"""
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(BF16)


class Container:
    """A pulsar container's tensors by name (every shard's header; an nvfp4 entry decoded through its pulsar.tensors
    declaration)."""

    def __init__(self, d):
        self.at = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".safetensors"):
                continue
            p = os.path.join(d, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            decl = json.loads(h.pop("__metadata__", {}).get("pulsar.tensors", "{}"))
            for k, v in h.items():
                self.at[k] = (p, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v,
                              decl.get(k, {}))

    def raw(self, name):
        p, off, n, _v, _d = self.at[name]
        with open(p, "rb") as f:
            f.seek(off)
            return f.read(n)

    def has(self, name):
        return name in self.at

    def tensor(self, name):
        """bf16 / i32 as stored ([shape]); an nvfp4 Linear as (W f32 [out][in], None) -- the global scale folded"""
        _p, _o, _n, v, d = self.at[name]
        b = self.raw(name)
        if d.get("layout") == "nvfp4":
            inp, out = d["dims_ne"]
            return torch.from_numpy(PR.nvfp4_decode(b, out, inp).copy())
        if v["dtype"] == "BF16":
            return torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).view(BF16).reshape(v["shape"])
        if v["dtype"] == "I32":
            return torch.from_numpy(np.frombuffer(b, dtype=np.int32).copy()).reshape(v["shape"])
        raise SystemExit(f"{name}: {v['dtype']} {d.get('layout')} is not a layout the reference reads")


class Drafter:
    def __init__(self, path, target_hf, device, container=None):
        self.dev = device
        cfg = json.load(open(os.path.join(path, "config.json")))
        dc = cfg["dflash_config"]
        self.dflash2 = cfg.get("architectures") == ["DFlash2DraftModel"]
        self.E, self.L, self.NQ, self.NKV, self.HD = (cfg["hidden_size"], cfg["num_hidden_layers"],
                                                      cfg["num_attention_heads"], cfg["num_key_value_heads"],
                                                      cfg["head_dim"])
        self.taps, self.group = dc["conv_kernel_size"], dc["conv_group_size"]
        self.mask = dc["mask_token_id"]
        if self.dflash2:   # L299: its keys, bf16 K / V, the container's tensors
            self.eps, self.theta, self.W = cfg["rms_norm_eps"], cfg["rope_parameters"]["rope_theta"], cfg["sliding_window"]
            self.block, self.k, self.kv8 = dc["block_size"], dc["selector_top_k"], False
            if container is None:
                raise SystemExit("--container: the DFlash2 reference reads the drafter as the container stores it")
            self.c = Container(container)
            self.emb = self.c.tensor("model.language_model.embed_tokens.weight").to(device)
        else:
            self.f = safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu")
            self.keys = set(self.f.keys())
            self.eps, self.theta, self.W = cfg["rms_norm_eps"], cfg["rope_theta"], dc["swa_window_size"]
            self.block, self.k, self.kv8 = cfg["block_size"], dc["lilicorr_candidate_topk"], True
            idx = json.load(open(os.path.join(target_hf, "model.safetensors.index.json")))["weight_map"]
            en = "language_model.backbone.embeddings.weight"
            with safe_open(os.path.join(target_hf, idx[en]), framework="pt", device="cpu") as g:
                self.emb = g.get_tensor(en).to(device)
        self.G = self.E // self.group
        self.fc = self.linear("fc")
        self.hn = self.get("hidden_norm.weight")
        self.lay = []
        for l in range(self.L):
            p = f"layers.{l}."
            self.lay.append(dict(
                inn=self.get(p + "input_layernorm.weight"), post=self.get(p + "post_attention_layernorm.weight"),
                q=self.linear(p + "self_attn.q_proj"), k=self.linear(p + "self_attn.k_proj"),
                v=self.linear(p + "self_attn.v_proj"), o=self.linear(p + "self_attn.o_proj"),
                qn=self.get(p + "self_attn.q_norm.weight"), kn=self.get(p + "self_attn.k_norm.weight"),
                acb=self.get(p + "attention_conv.base_kernel"), acp=self.linear(p + "attention_conv.kernel_projection"),
                mcb=self.get(p + "mlp_conv.base_kernel"), mcp=self.linear(p + "mlp_conv.kernel_projection"),
                g=self.linear(p + "mlp.gate_proj"), u=self.linear(p + "mlp.up_proj"),
                d=self.linear(p + "mlp.down_proj")))
        self.norm = self.get("norm.weight")
        self.head = self.linear("lm_head")
        self.head_ids = None
        if self.dflash2:
            if self.c.has("drafter.lm_head.ids"):
                self.head_ids = self.c.tensor("drafter.lm_head.ids").long().to(self.dev)
            self.sel_proj = self.linear("candidate_selector.hidden_projection")
            self.sel_pred = self.get("candidate_selector.predecessor_codebook").float()
            self.sel_succ = self.get("candidate_selector.successor_codebook").float()

    def get(self, name):
        if self.dflash2:
            return self.c.tensor("drafter." + name).to(self.dev)
        return self.f.get_tensor(name).to(self.dev)

    def linear(self, name):
        """(W f32 [out][in], global scale or None)"""
        if self.dflash2:
            t = self.c.tensor("drafter." + name + ".weight")
            return t.float().to(self.dev), None
        if name + ".weight_scale" in self.keys:
            w = self.f.get_tensor(name + ".weight")                      # u8 [out][in/2]
            sc = self.f.get_tensor(name + ".weight_scale").float()       # e4m3 [out][in/16]
            s2 = float(self.f.get_tensor(name + ".weight_scale_2").float())
            lo, hi = (w & 0xF).long(), (w >> 4).long()
            v = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(w.shape[0], -1)
            return (v * sc.repeat_interleave(16, dim=1)).to(self.dev), s2
        return self.f.get_tensor(name + ".weight").float().to(self.dev), None

    # ---- the ops -------------------------------------------------------------------------------------------------
    @staticmethod
    def lin(x, w):
        W, s2 = w
        y = x.float() @ W.T
        return bf(y * s2 if s2 is not None else y)

    def rms(self, x, w):
        xf = x.float()
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return bf(bf(xf * r).float() * w.float())

    def rope(self, x, pos):
        """x bf16 [n][heads][d]: NeoX, bf16 cos / sin, bf16 products (vLLM's rotary kernel on bf16)"""
        d = x.shape[-1]
        inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
        ang = pos.float()[:, None] * inv[None, :]
        c, s = bf(torch.cos(ang))[:, None, :].float(), bf(torch.sin(ang))[:, None, :].float()
        x0, x1 = x[..., : d // 2].float(), x[..., d // 2:].float()
        return torch.cat([bf(bf(x0 * c).float() - bf(x1 * s).float()), bf(bf(x1 * c).float() + bf(x0 * s).float())],
                         dim=-1)

    def conv(self, x, delta, base):
        """x bf16 [R][C], delta bf16 [R][taps][G], base bf16 [taps][C] -> bf16 (Triton's f32 order)"""
        d = delta.float().repeat_interleave(self.group, dim=-1)
        co = base.float()[None] + d
        acc = co[:, 0] * x.float()
        pos = torch.arange(x.shape[0], device=x.device) % self.block
        for t in range(1, self.taps):
            sh = torch.zeros_like(x.float())
            sh[t:] = x.float()[:-t]
            acc = acc + co[:, t] * sh * (pos >= t).float()[:, None]
        return bf(acc)

    def context(self, taps):
        """taps bf16 [n][n_taps hidden] -> (fc_out, cn)"""
        c = self.lin(taps, self.fc)
        return c, self.rms(c, self.hn)

    def forward(self, cn, cpos, P, carry):
        """the block at positions P .. P + block - 1 over the context rows cn (bf16 [n][hidden]) at cpos: per-layer
        intermediates, the final hidden o, the log-softmax rows"""
        B, NQ, NKV, HD, G = self.block, self.NQ, self.NKV, self.HD, self.G
        qpos = torch.arange(P, P + B, device=self.dev)
        ids = torch.tensor([carry] + [self.mask] * (B - 1), device=self.dev)
        h = self.emb[ids].to(BF16)
        t = None
        inter = []
        for l in range(self.L):
            Lw = self.lay[l]
            if t is not None:
                h = bf(h.float() + t.float())
            x = self.rms(h, Lw["inn"])
            co = self.lin(x, Lw["acp"]).reshape(B, 2, self.taps, G)
            cx = self.conv(x, co[:, 0], Lw["acb"][0])
            q = self.rope(self.rms(self.lin(cx, Lw["q"]).reshape(B, NQ, HD), Lw["qn"]), qpos)
            kvf = e4m3 if self.kv8 else (lambda x: x)   # the ring's row format: LiLiCorr E4M3, DFlash2 bf16
            kq = kvf(self.rope(self.rms(self.lin(cx, Lw["k"]).reshape(B, NKV, HD), Lw["kn"]), qpos))
            vq = kvf(self.lin(cx, Lw["v"]).reshape(B, NKV, HD))
            kc = kvf(self.rope(self.rms(self.lin(cn, Lw["k"]).reshape(-1, NKV, HD), Lw["kn"]), cpos))
            vc = kvf(self.lin(cn, Lw["v"]).reshape(-1, NKV, HD))
            K, V = torch.cat([kc, kq]).float(), torch.cat([vc, vq]).float()
            kp = torch.cat([cpos, qpos])
            rep = NQ // NKV
            s = torch.einsum("qhd,khd->hqk", q.float(), K.repeat_interleave(rep, dim=1)) / math.sqrt(HD)
            vis = (qpos[:, None] - kp[None, :]) <= (self.W - 1)     # the window; bidirectional in the block
            s = s.masked_fill(~vis[None], float("-inf"))
            at = bf(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), V.repeat_interleave(rep, dim=1)))
            at = at.reshape(B, NQ * HD)
            t = self.conv(self.lin(at, Lw["o"]), co[:, 1], Lw["acb"][1])
            h = bf(h.float() + t.float())
            x = self.rms(h, Lw["post"])
            co = self.lin(x, Lw["mcp"]).reshape(B, 2, self.taps, G)
            cx = self.conv(x, co[:, 0], Lw["mcb"][0])
            m = bf(torch.nn.functional.silu(self.lin(cx, Lw["g"]).float()) * self.lin(cx, Lw["u"]).float())
            t = self.conv(self.lin(m, Lw["d"]), co[:, 1], Lw["mcb"][1])
            inter.append(dict(q=q.reshape(B, -1), k=kq.reshape(B, -1), v=vq.reshape(B, -1), attn=at, out_hidden=t,
                              out_residual=h))
        o = self.rms(bf(h.float() + t.float()), self.norm)
        lp = torch.log_softmax(self.lin(o, self.head).float(), dim=-1)
        return inter, o, lp

    def select(self, o, lp, carry):
        """(L299) the CandidateSelector's walk over rows 1 .. block - 1: top-k of each slot's head row (ids through the
        head's id map), score = log-prob + succ[c] . (pred[prev] * hidden_projection(o)) -> the chain"""
        tv, ti = torch.topk(lp[1:], self.k, dim=-1)
        ids = self.head_ids[ti] if self.head_ids is not None else ti
        hp = self.lin(o[1:], self.sel_proj).float()
        prev, chain = carry, []
        for s in range(ids.shape[0]):
            sc = tv[s] + self.sel_succ[ids[s]] @ (self.sel_pred[prev] * hp[s])
            j = int(torch.argmax(sc))
            prev = int(ids[s, j])
            chain.append(prev)
        return chain


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def topk_report(lp, ids, elp, k):
    tv, ti = torch.topk(lp, k, dim=-1)
    top1 = (ti[:, 0] == ids[:, 0]).float().mean().item()
    ovl = float(np.mean([len(set(ti[j].tolist()) & set(ids[j].tolist())) / k for j in range(ti.shape[0])]))
    dlp = (tv[:, 0] - elp[:, 0]).abs().max().item()
    return ti, top1, ovl, dlp


def golden(D, path, rounds):
    g = safe_open(path, framework="pt", device="cpu")
    keys = set(g.keys())
    ctx = {}
    agree = n = 0
    for r in range(10000):
        p = f"r{r:03d}."
        if p + "ctx_taps" not in keys or r >= rounds:
            break
        taps, pos = g.get_tensor(p + "ctx_taps").to(D.dev), g.get_tensor(p + "ctx_pos").tolist()
        nrej = int(g.get_tensor(p + "num_rejected")[0])
        keep = len(pos) - nrej
        fc, cn = D.context(taps[:keep])
        e_fc = g.get_tensor(p + "fc_out").to(D.dev)[:keep]
        for i in range(keep):
            ctx[int(pos[i])] = cn[i]
        qpos = g.get_tensor(p + "q_pos").tolist()
        P, carry = int(qpos[0]), int(g.get_tensor(p + "q_ids")[0])
        lo = max(0, P - (D.W - 1) - (D.block - 1))
        cp = [x for x in range(lo, P) if x in ctx]
        cpos = torch.tensor(cp, device=D.dev)
        inter, o, lp = D.forward(torch.stack([ctx[x] for x in cp]), cpos, P, carry)
        line = [f"r{r:03d} P {P} ctx {len(cp)} fc {rel(fc, e_fc):.4f}"]
        for l, it in enumerate(inter):
            line.append(f"L{l} " + " ".join(f"{k} {rel(it[k], g.get_tensor(p + f'L{l}.{k}').to(D.dev)):.3f}"
                                            for k in ("q", "k", "v", "attn", "out_hidden", "out_residual")))
        line.append(f"last {rel(o, g.get_tensor(p + 'last_hidden').to(D.dev)):.4f}")
        ids, elp = g.get_tensor(p + "cand_ids").to(D.dev), g.get_tensor(p + "cand_logp").to(D.dev)
        ti, top1, ovl, dlp = topk_report(lp[1:], ids, elp, D.k)
        agree += int((ti[:, 0] == ids[:, 0]).sum())
        n += ti.shape[0]
        line.append(f"top-1 {top1:.2f} top-{D.k} {ovl:.2f} lp {dlp:.3f}")
        print(" | ".join(line), flush=True)
    print(f"golden: top-1 slot agreement {agree}/{n}")


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
            ctx.append((pos, bank, np.frombuffer(f.read(2 * n * w), dtype=np.uint16).reshape(n, w)))
    passes = []
    with open(os.path.join(d, "draft.bin"), "rb") as f:
        while True:
            h = f.read(16)
            if not h:
                break
            nb, B, E, k = struct.unpack("<4I", h)
            rec = dict(nb=nb, B=B, E=E, k=k)
            rec["bank"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            rec["carry"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            rec["P"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            S = B - 1
            rec["o"] = np.frombuffer(f.read(2 * nb * B * E), dtype=np.uint16).reshape(nb * B, E)
            rec["ids"] = np.frombuffer(f.read(4 * nb * S * k), dtype=np.int32).reshape(nb, S, k)
            rec["lp"] = np.frombuffer(f.read(4 * nb * S * k), dtype=np.float32).reshape(nb, S, k)
            rec["scores"] = np.frombuffer(f.read(4 * nb * S * k * k), dtype=np.float32).reshape(nb, S, k, k)
            rec["chain"] = np.frombuffer(f.read(4 * nb * S), dtype=np.int32).reshape(nb, S)
            passes.append(rec)
    return ctx, passes


def u16(a, dev):
    return torch.from_numpy(a.astype(np.int16).copy()).view(BF16).to(dev)


def dump(D, d, rounds):
    ctx, passes = read_dump(d)
    rows = {}
    for pos, bank, t in ctx:   # the last absorbed wins: a re-prefilled position is rewritten
        for i in range(len(pos)):
            rows.setdefault(int(bank[i]), {})[int(pos[i])] = t[i]
    for pi, ps in enumerate(passes[:rounds]):
        B = ps["B"]
        for bi in range(ps["nb"]):
            b, P, carry = int(ps["bank"][bi]), int(ps["P"][bi]), int(ps["carry"][bi])
            have = rows.get(b, {})
            lo = max(0, P - (D.W - 1) - (B - 1))
            miss = [x for x in range(lo, P) if x not in have]
            if miss:
                print(f"pass {pi} bank {b}: {len(miss)} context positions missing from the dump (first {miss[0]})")
                continue
            _fc, cn = D.context(u16(np.stack([have[x] for x in range(lo, P)]), D.dev))
            _inter, o, lp = D.forward(cn, torch.arange(lo, P, device=D.dev), P, carry)
            eo = u16(ps["o"][bi * B:(bi + 1) * B], D.dev)
            cos = torch.nn.functional.cosine_similarity(o.float(), eo.float(), dim=-1)
            ids = torch.from_numpy(ps["ids"][bi].copy()).to(D.dev)
            elp = torch.from_numpy(ps["lp"][bi].copy()).to(D.dev)
            if D.head_ids is not None:   # the engine's ids are vocabulary ids; the reference's top-k head rows
                lpv = torch.full((lp.shape[0], int(D.head_ids.max()) + 1), float("-inf"), device=D.dev)
                lpv[:, D.head_ids] = lp
            else:
                lpv = lp
            ti, top1, ovl, dlp = topk_report(lpv[1:], ids, elp, ps["k"])
            extra = ""
            if D.dflash2:
                ref_chain = D.select(o, lp, carry)
                same = sum(int(a == b) for a, b in zip(ref_chain, ps["chain"][bi].tolist()))
                extra = f"; selector chain ref {ref_chain} agree {same}/{len(ref_chain)}"
            print(f"pass {pi} bank {b} P {P} carry {carry}: final hidden rel {rel(eo, o):.4f} cos min "
                  f"{cos.min().item():.5f}; slots top-1 agree {top1:.2f}, top-{ps['k']} overlap {ovl:.2f}, top-1 lp "
                  f"max|d| {dlp:.4f}; ref top-1 {ti[:, 0].tolist()} engine top-1 {ids[:, 0].tolist()} engine chain "
                  f"{ps['chain'][bi].tolist()}{extra}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target-hf", help="the target's HF checkpoint (LiLiCorr: its input embedding)")
    ap.add_argument("--container", help="(DFlash2) the pulsar container the engine served")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--golden")
    g.add_argument("--dump")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    D = Drafter(a.drafter, a.target_hf, torch.device(a.device), a.container)
    if a.golden:
        golden(D, a.golden, a.rounds)
    else:
        dump(D, a.dump, a.rounds)


if __name__ == "__main__":
    main()

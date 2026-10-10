#!/usr/bin/env python3
"""L287 step 1: how much functional error does EXL3 prefill on FP8 tensor cores add?  A fidelity screen, no kernel.

The MXFP8 rounding is the engine's own rule (producers.mxfp8_lt_from_bf16 / pulsar_cuda_mx.cuh
pulsar_mx_shared_exp): per 32 along K, E8M0 exponent se = floor(log2 amax) - 7 (the block max lands in
[128, 256), nothing saturates), E4M3 round-to-nearest-even of v * 2^-se.  `mx()` below is asserted
bit-exact against producers.py's encoder at startup (weights) -- the same rule the activation producers use.

Arms, each graded as ||Y_arm - Y_ref|| / ||Y_ref|| with Y_ref = X W_src^T (X = the activation the GEMM sees in the
reference: DeepSeek's served E4M3 x, Qwen's BF16 x):

  a    X  W_q^T                         today's decode (W_q = the EXL3 weight in its original basis)
  b    X  mx(W_q)^T                     reconstruct -> MXFP8 -> cuBLASLt / CUTLASS
  c    mx(X) mx(W_q)^T                  b with the activation also MXFP8 (Qwen: BF16 by contract today)
  x    mx(X) W_q^T                      activation rounding alone
  d    X  mx(W_src)^T                   context: MXFP8 of the source weight itself
  e    H(mx(X_r) mx(W_i)^T) * svh       D2R in the rotated basis: X_r = (X * suh) H128 re-encoded per 32,
                                        W_i the trellis-decoded inner weight; output Hadamard + svh in the epilogue
  eW   H(X_r mx(W_i)^T) * svh           e's weight rounding alone
  eX   H(mx(X_r) W_i^T) * svh           e's activation rounding alone (on DeepSeek a NEW rounding of an E4M3 x)

Errors are pooled over all rows of a tensor family (sqrt(sum ||dY||^2 / sum ||Y_ref||^2)); per-expert medians too.
moe_out: the block's routed-expert output with every projection on the arm, routes/weights from the source router
(grade_layer.py's metric; the SwiGLU output re-served in the arm's activation format).

    fp8_screen.py v41  --ref $V41 --exllamav3 ~/exllamav3 --run /mnt/models/v41-exl3-ours-est/l20 --block layers.20 --out J
    fp8_screen.py qwen --exllamav3 ~/exllamav3 --container /mnt/models/qwen38fn-u-e4-d5 --layers 2 3 22 23 46 47 --out J
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(HERE, "..", "container"))
_spec = importlib.util.spec_from_file_location("pulsar_producers", os.path.join(HERE, "..", "container", "producers.py"))
PR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PR)

ARMS_W = ("a", "b", "d")                    # weight-side arms on the served X
ARMS_X = ("c", "x")                         # arms that also round X in the original basis
ARMS_E = ("e", "eW", "eX")                  # rotated-basis (D2R) arms


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mx(t):
    """MXFP8 round-trip over the last dim (per 32): the engine's encoder rule, returned as fp32 values."""
    shp = t.shape
    b = t.float().reshape(-1, shp[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True)
    _, e = torch.frexp(amax)
    se = torch.where(amax > 0, (e - 1) - 7, torch.full_like(e, -127)).clamp(-127, 127).float()
    q = (b * torch.exp2(-se)).to(torch.float8_e4m3fn).float()
    return (q * torch.exp2(se)).reshape(shp)


def check_mx_rule(w_bf16):
    """mx() must equal producers.mxfp8_lt_from_bf16 decoded by its own inverse, bit for bit."""
    out, inp = w_bf16.shape
    buf = PR.mxfp8_lt_from_bf16(w_bf16.cpu().contiguous().view(torch.int16).numpy().tobytes(), out, inp)
    ref = torch.from_numpy(np.ascontiguousarray(PR.mxfp8_lt_decode(buf, out, inp))).float()
    got = mx(w_bf16.float().cuda()).cpu()
    n_bad = int((got != ref).sum())
    log(f"mx rule check vs producers.mxfp8_lt_from_bf16 on [{out},{inp}]: {n_bad} of {got.numel()} differ")
    if n_bad:
        raise SystemExit("REFUSING: the screen's MXFP8 encoder is not the engine's")


class Had:
    """128-blockwise Hadamard (exllamav3's had_k = had_n = 128; Sylvester, symmetric, orthonormal)."""

    def __init__(self, dev):
        from exllamav3.modules.quant.exl3_lib.quantize import get_hadamard_dt
        self.H = get_hadamard_dt(128, dev, torch.float, 1 / math.sqrt(128))

    def __call__(self, y):                  # rows of y, last dim in 128-blocks
        s = y.shape
        return (y.reshape(-1, s[-1] // 128, 128) @ self.H).reshape(s)


class Proj:
    """One EXL3 projection in both bases, plus its source weight; all fp32 [out, in]."""

    def __init__(self, lin, w_src, had):
        self.had = had
        self.suh, self.svh = lin.suh.float(), lin.svh.float()
        wi = lin.get_inner_weight_tensor().float()                  # [in, out], rotated basis
        self.wi = wi.t().contiguous()                               # [out, in]
        # W = diag(suh) Hk Wi Hn diag(svh)  (LinearEXL3.get_weight_tensor, in fp32)
        w = had(wi.t()).t()                                         # Hk Wi  (H symmetric)
        w = w * self.suh[:, None]
        w = had(w) * self.svh[None, :]
        self.wq = w.t().contiguous()                                # [out, in], original basis
        ex = lin.get_weight_tensor().float().t()                    # exllamav3's own (fp16) composition
        dev_ = float((self.wq - ex).norm() / ex.norm())
        if dev_ > 2e-3:
            raise SystemExit(f"REFUSING: fp32 composition differs from LinearEXL3.get_weight_tensor by {dev_:.2e}")
        self.wsrc = w_src.float()
        self.mx_wq, self.mx_wsrc, self.mx_wi = mx(self.wq), mx(self.wsrc), mx(self.wi)

    def rot(self, x):
        return self.had(x.float() * self.suh)

    def unrot(self, y):
        return self.had(y) * self.svh

    def run(self, x, arm):
        x = x.float()
        if arm == "ref":
            return x @ self.wsrc.t()
        if arm == "a":
            return x @ self.wq.t()
        if arm == "b":
            return x @ self.mx_wq.t()
        if arm == "c":
            return mx(x) @ self.mx_wq.t()
        if arm == "x":
            return mx(x) @ self.wq.t()
        if arm == "d":
            return x @ self.mx_wsrc.t()
        xr = self.rot(x)
        if arm == "e":
            return self.unrot(mx(xr) @ self.mx_wi.t())
        if arm == "eW":
            return self.unrot(xr @ self.mx_wi.t())
        if arm == "eX":
            return self.unrot(mx(xr) @ self.wi.t())
        raise ValueError(arm)


class Acc:
    """Pooled and per-item relative errors per (tensor, arm)."""

    def __init__(self):
        self.num, self.den, self.items = {}, {}, {}

    def add(self, key, arm, y, ref):
        n, d = float(((y - ref) ** 2).sum()), float((ref ** 2).sum())
        self.num[(key, arm)] = self.num.get((key, arm), 0.0) + n
        self.den[(key, arm)] = self.den.get((key, arm), 0.0) + d
        if d > 0:
            self.items.setdefault((key, arm), []).append(math.sqrt(n / d))

    def table(self):
        out = {}
        for (key, arm), n in self.num.items():
            v = sorted(self.items.get((key, arm), [float("nan")]))
            out.setdefault(key, {})[arm] = {"pooled": math.sqrt(n / self.den[(key, arm)]),
                                            "median": v[len(v) // 2], "n": len(v)}
        return out


def summarize(table, arms):
    """Added error of each arm over a (today), relative and in quadrature."""
    for key, r in table.items():
        a = r["a"]["pooled"]
        for arm in arms:
            if arm in r and arm != "a":
                v = r[arm]["pooled"]
                r[arm]["added_rel"] = v / a - 1
                r[arm]["added_quad"] = math.sqrt(max(v * v - a * a, 0.0))
    return table


def print_table(title, table, arms):
    log(title)
    print(f"{'tensor':58s} " + " ".join(f"{x:>8s}" for x in arms) + "   b/a-1   e/a-1  quad(b) quad(e)", flush=True)
    for key in table:
        r = table[key]
        row = " ".join(f"{r[x]['pooled']:8.5f}" if x in r else f"{'-':>8s}" for x in arms)
        ex = "".join(f" {r[x]['added_rel']*100:6.2f}%" if x in r else "       -" for x in ("b", "e"))
        qd = "".join(f" {r[x]['added_quad']:7.4f}" if x in r else "       -" for x in ("b", "e"))
        print(f"{key:58s} {row}  {ex}  {qd}", flush=True)


# ---------------------------------------------------------------------------------------------- DeepSeek V4.1
@torch.inference_mode()
def run_v41(a):
    sys.path.insert(0, a.exllamav3)
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from safetensors import safe_open
    import refkernels
    from v41_stream import Checkpoint, gemm_input

    dev = torch.device("cuda:0")
    had = Had(dev)
    comp, idx = a.block.split(".")
    name = f"{'layer' if comp == 'layers' else 'mtp'}{int(idx):02d}"
    cfg = json.load(open(os.path.join(a.ref, "config.json")))["text_config"]
    limit = cfg["swiglu_limit"]
    n_exp = cfg["n_routed_experts"] if comp == "layers" else cfg["dspark_n_routed_experts"]
    hold = safe_open(os.path.join(a.run, "holdout", f"{name}.safetensors"), framework="pt")
    x = hold.get_tensor("x").to(dev)
    route_i, route_w = hold.get_tensor("indices").to(dev).long(), hold.get_tensor("weights").to(dev)
    xq = gemm_input(x)                                                  # the served E4M3 x (fp32 values)
    q = safe_open(os.path.join(a.run, f"exl3-k{a.k}", f"model-{name}.safetensors"), framework="pt", device="cpu")
    ckpt = Checkpoint(a.ref)
    log(f"{a.block}: {x.size(0)} held-out tokens, {n_exp} experts, K{a.k}, swiglu_limit {limit}")

    def src(e, p):
        k = f"{a.block}.ffn.experts.{e}.{p}"
        return refkernels.dequant_fp4(ckpt.get(f"{k}.weight").view(torch.float4_e2m1fn_x2).to(dev),
                                      ckpt.get(f"{k}.scale").to(dev))

    check_mx_rule(src(0, "w1").bfloat16())

    def swiglu(g, u):
        if limit > 0:
            u, g = u.clamp(-limit, limit), g.clamp(max=limit)
        return gemm_input((F.silu(g) * u).bfloat16())                  # served: E4M3 again

    arms = ("a", "b", "d") + ARMS_E
    acc = Acc()
    y = {arm: torch.zeros(x.size(0), x.size(1), dtype=torch.float32, device=dev) for arm in ("ref",) + arms}
    for e in range(n_exp):
        P = {}
        for p in ("w1", "w3", "w2"):
            k = f"{a.block}.ffn.experts.{e}.{p}"
            t = {s: q.get_tensor(f"{k}.{s}").to(dev) for s in ("trellis", "suh", "svh", "mul1")}
            ws = src(e, p)
            P[p] = Proj(LinearEXL3(None, ws.shape[1], ws.shape[0], **t), ws, had)
        hit = route_i == e
        tok = hit.any(dim=1).nonzero().flatten()
        if not tok.numel():
            continue
        wt = (route_w * hit).sum(dim=1)[tok].float()
        xs = xq[tok]
        h_src = swiglu(P["w1"].run(xs, "ref"), P["w3"].run(xs, "ref"))
        for p, xin in (("w1", xs), ("w3", xs), ("w2", h_src)):
            ref = P[p].run(xin, "ref")
            for arm in arms:
                acc.add(p, arm, P[p].run(xin, arm), ref)
        for arm in ("ref",) + arms:
            h = swiglu(P["w1"].run(xs, arm), P["w3"].run(xs, arm))
            y[arm][tok] += wt[:, None] * P["w2"].run(h, arm)
        if e % 64 == 0:
            log(f"expert {e}: {tok.numel()} routed tokens")
    table = summarize(acc.table(), arms)
    moe = {arm: float((y[arm] - y["ref"]).norm() / y["ref"].norm()) for arm in arms}
    table["moe_out"] = {arm: {"pooled": v} for arm, v in moe.items()}
    summarize({"moe_out": table["moe_out"]}, arms)
    print_table(f"{a.block} K{a.k}: functional rel error per arm (pooled over routed rows)", table, arms)
    json.dump({"block": a.block, "k": a.k, "tokens": x.size(0), "table": table}, open(a.out, "w"), indent=1)
    log(f"-> {a.out}")


# ---------------------------------------------------------------------------------------------- Qwen3.8
@torch.inference_mode()
def run_qwen(a):
    sys.path.insert(0, a.exllamav3)
    sys.path.insert(0, os.path.join(HERE, "..", "allocator"))
    import glue as G
    import container as CT

    M, DEV, PFX = G.M, G.DEV, G.PFX
    had = Had(DEV)
    cfg, W = G.text_config(), G.Weights()
    G.install()
    tok = G.AutoTokenizer.from_pretrained(G.TOK)
    ho, _ = G.heldout(tok)
    ho = ho[:a.rows]                       # the held-out list is round-robin over source classes
    log(f"qwen: {len(ho)} held-out rows, {sum(s.numel() for _, s in ho):,} tokens; target layers {a.layers}")
    c = CT.Container(a.container)
    rotary = M.Qwen4ExpTextRotaryEmbedding(config=cfg).to(DEV)
    emb = W.get(f"{PFX}embed_tokens.weight")
    hs = [emb[ids.to(DEV)].unsqueeze(0).repeat(1, 1, cfg.hc_count) for _, ids in ho]
    del emb
    checked = False
    arms_d = ("a", "b", "c", "x", "d") + ARMS_E
    report = {"rows": len(ho), "tokens": int(sum(s.numel() for _, s in ho)), "layers": {}}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for L in range(max(a.layers) + 1):
        t0 = time.time()
        layer = G.build_layer(cfg, W, L)
        if layer.ple is not None:
            G.gather_ple(layer, ho)
        pre = f"{PFX}layers.{L}."
        cap, moe_x, moe_tk, hooks = {}, [], [], []
        if L in a.layers:
            for mn, mod in layer.named_modules():
                full = f"{pre}{mn}.weight"
                if isinstance(mod, torch.nn.Linear) and full in c.tensors and c.tensors[full][1].startswith("exl3"):
                    cap[mn] = []
                    hooks.append(mod.register_forward_pre_hook(
                        lambda m, args, mn=mn: cap[mn].append(args[0].reshape(-1, args[0].shape[-1])[::a.stride].clone())))
            hooks.append(layer.mlp.register_forward_pre_hook(
                lambda m, args: moe_x.append(args[0].flatten(0, 1).to(torch.bfloat16))))
            hooks.append(layer.mlp.gate.register_forward_hook(lambda m, args, out: moe_tk.append((out[1], out[2]))))
        for j, (_, ids) in enumerate(ho):
            hs[j] = G.run_layer(cfg, rotary, layer, hs[j], ids)
        for h in hooks:
            h.remove()
        if L not in a.layers:
            log(f"layer {L:2d} streamed {time.time() - t0:.0f}s")
            del layer
            torch.cuda.empty_cache()
            continue
        acc = Acc()
        # ---- dense EXL3 tensors (and the shared expert)
        missing = [nm for nm, (_, lay) in c.tensors.items()
                   if nm.startswith(pre) and lay.startswith("exl3") and nm[len(pre):-len(".weight")] not in cap]
        if missing:
            raise SystemExit(f"layer {L}: EXL3 tensors with no captured input: {missing}")
        for mn, xs in cap.items():
            X = torch.cat(xs)
            if X.size(0) > a.dense_rows:
                X = X[torch.randperm(X.size(0), generator=gen)[:a.dense_rows].to(DEV)]
            full = f"{pre}{mn}.weight"
            lay, k, n = c.layout(full)
            ws = layer.get_submodule(mn).weight.data
            if not checked:
                check_mx_rule(ws)
                checked = True
            P = Proj(CT.exl3_linear(c.raw(full), k, n, CT.words(lay)), ws, had)
            ref = P.run(X, "ref")
            for arm in arms_d:
                acc.add(f"L{L}.{mn} ({lay})", arm, P.run(X, arm), ref)
            del P
        # ---- routed experts
        xm = torch.cat(moe_x)
        tw = torch.cat([w for w, _ in moe_tk]).float()
        ti = torch.cat([i for _, i in moe_tk]).long()
        assert ti.shape == (xm.size(0), cfg.num_experts_per_tok), ti.shape
        fams = {}
        for part in ("gate_up_proj", "down_proj"):
            blobs, ent = c.expert_blobs(f"{pre}mlp.experts.{part}")
            fams[part] = (blobs, ent["layout"], ent["dims_per_expert_ne"])
        moe_arms = ("ref",) + arms_d
        y = {arm: torch.zeros(xm.size(0), xm.size(1), dtype=torch.float32, device=DEV) for arm in moe_arms}
        inter = cfg.moe_intermediate_size
        gu_src_all = layer.mlp.experts.gate_up_proj.data
        dn_src_all = layer.mlp.experts.down_proj.data

        def mid(gu, arm):
            g, u = gu[:, :inter], gu[:, inter:]
            return (F.silu(g) * u).bfloat16().float()                    # Qwen serves BF16 between the GEMMs

        for e in range(cfg.num_experts):
            hit = ti == e
            tk = hit.any(dim=1).nonzero().flatten()
            if not tk.numel():
                continue
            wt = (tw * hit).sum(dim=1)[tk]
            P = {}
            for part, ws in (("gate_up_proj", gu_src_all[e]), ("down_proj", dn_src_all[e])):
                blobs, lay, (k, n) = fams[part]
                P[part] = Proj(CT.exl3_linear(blobs[e], k, n, CT.words(lay)), ws, had)
            xs = xm[tk].float()
            h_src = mid(P["gate_up_proj"].run(xs, "ref"), "ref")
            for part, xin in (("gate_up_proj", xs), ("down_proj", h_src)):
                ref = P[part].run(xin, "ref")
                for arm in arms_d:
                    acc.add(f"L{L}.experts.{part} ({fams[part][1]})", arm, P[part].run(xin, arm), ref)
            for arm in moe_arms:
                y[arm][tk] += wt[:, None] * P["down_proj"].run(mid(P["gate_up_proj"].run(xs, arm), arm), arm)
        table = summarize(acc.table(), arms_d)
        mo = {arm: {"pooled": float((y[arm] - y["ref"]).norm() / y["ref"].norm())} for arm in arms_d}
        table[f"L{L}.moe_out (routed)"] = summarize({"m": mo}, arms_d)["m"]
        print_table(f"layer {L} ({cfg.layer_types[L]}): functional rel error per arm", table, arms_d)
        report["layers"][L] = table
        json.dump(report, open(a.out, "w"), indent=1)
        log(f"layer {L:2d} streamed+graded {time.time() - t0:.0f}s")
        del layer, y, xm, cap, P
        torch.cuda.empty_cache()
    log(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="family", required=True)
    v = sub.add_parser("v41")
    v.add_argument("--ref", required=True)
    v.add_argument("--run", required=True)
    v.add_argument("--block", required=True)
    v.add_argument("--k", type=int, default=3)
    qw = sub.add_parser("qwen")
    qw.add_argument("--container", required=True)
    qw.add_argument("--layers", type=int, nargs="+", required=True)
    qw.add_argument("--rows", type=int, default=16, help="held-out rows streamed (round-robin over sources)")
    qw.add_argument("--stride", type=int, default=2, help="dense-input row subsample at capture")
    qw.add_argument("--dense-rows", type=int, default=8192)
    for p in (v, qw):
        p.add_argument("--exllamav3", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    (run_v41 if a.family == "v41" else run_qwen)(a)


if __name__ == "__main__":
    main()

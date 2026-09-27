#!/usr/bin/env python3
"""L251 S4: the Qwen3.8-Flash-Next router, MoE block, Gated Residual and PLE launchers on REAL weights,
against transformers' own `qwen4_exp` modules (and exllamav3's reconstruct for the EXL3 weights).

  router   layer 12's router on the captured MoE inputs (/srv/models/qwen38fn-act-L12/moe_in.pt, 145,172
           rows) vs the source's recorded top-10 (topk.pt): the SET per row must match; every mismatch is
           classified (a bf16 tie at the boundary, or a boundary logit that rounds to a different bf16).
  moe      layer 12's MoE block on a handful of captured rows: the experts they route to and the shared
           expert quantized with exllamav3's quantize_exl3 (real Hessians from the captured rows) and run
           by pulsar's launcher; graded against
             emu    the SAME arithmetic in fp64 from exllamav3's reconstruct (get_inner_weight_tensor) with
                    the A8 encodings where the device encodes -- the weights decode identically, the arm is
                    f32-class;
             recon  the source semantics in fp64 with the reconstructed weights, no A8 -- the quant error;
             hf     transformers' Qwen4ExpTextSparseMoeBlock on the BF16 source -- the whole difference.
  capture  embed + layer 0 + layer 1's PLE and attention GR through the HF modules for a few sequences
           (the stream inputs of the next two checks), to /srv/models/qwen-s4/cap-l1.
  gr       layer 1's attention-site GR (and the top-level mixer on the same streams) with W_down / W_up
           MXFP8 (tools/container/producers.py, the builder's encoder) vs the HF module (bf16) and its
           fp64 twin on the MXFP8-decoded weights; the host double reference for 16 rows.
  ple      layer 1's PLE with key_proj / value_proj at EXL3 (quantize_exl3, H from the captured
           embedding rows) vs the HF module (bf16) and the fp64 twin with reconstructed weights + A8.

  PYTHONPATH=~/exllamav3 ~/venvs/q38/bin/python s4_xcheck.py {router|moe|capture|gr|ple} --pulsar TREE --out DIR
Real-weight GPU work: run it under `flock ~/qwen-gpu.lock`.
"""
import argparse, json, math, os, struct, subprocess, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

SRC = "/srv/models/qwen38fn-bf16"
TOK = "/srv/models/qwen38fn-tok"
CALIB = "/srv/models/calib-qwen38-v1.jsonl"
ACT = "/srv/models/qwen38fn-act-L12"
CAP = "/srv/models/qwen-s4/cap-l1"
PFX = "model.language_model."
DEV = torch.device(os.environ.get("QWEN_S4_DEV", "cuda:0"))   # cpu only for a dry run of the python side
LOGF = None


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n")
        LOGF.flush()


class Weights:
    def __init__(self):
        self.map = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
        self._open = {}

    def f(self, name):
        fn = self.map[name]
        if fn not in self._open:
            self._open[fn] = safe_open(f"{SRC}/{fn}", framework="pt", device="cpu")
        return self._open[fn]

    def get(self, name, device=DEV):
        return self.f(name).get_tensor(name).clone().to(device)   # host clone first: mmap -> CUDA is ~5 MB/s on GB10

    def prefixed(self, prefix, skip=()):
        return {k[len(prefix):]: self.get(k) for k in self.map if k.startswith(prefix) and not any(s in k for s in skip)}


def text_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    cfg = Qwen4ExpConfig.from_pretrained(TOK).text_config
    cfg._attn_implementation = "sdpa"
    cfg._experts_implementation = "eager"
    return cfg


def tofile(t, path):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    t.numpy().tofile(path)


def run_tool(args, tool):
    r = subprocess.run([tool] + [str(a) for a in args], capture_output=True, text=True)
    log(f"tool {' '.join(str(a) for a in args[:3])}: rc={r.returncode} {r.stdout.strip()[-400:]}")
    if r.returncode:
        log("STDERR", r.stderr[-2000:])
        sys.exit(f"tool failed rc={r.returncode}")
    return r.stdout


# ------------------------------------------------------------------ the A8 encoding and the Hadamard, fp64
def e4m3_dec(x):
    """pulsar's producer encoding per 32 (se = floor(log2 amax) - 7, E4M3 RNE, satfinite) -> decoded fp64."""
    shp = x.shape
    g = x.double().reshape(*shp[:-1], shp[-1] // 32, 32)
    amax = g.abs().amax(-1, keepdim=True)
    se = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-300))) - 7, torch.full_like(amax, -127.0)).clamp(-127, 127)
    q = (g * torch.exp2(-se)).float().to(torch.float8_e4m3fn).double()
    return (q * torch.exp2(se)).reshape(shp)


_H128 = None


def had(v):
    """natural-order Sylvester H128 / sqrt(128) over blocks of 128 of the last dim (exl3_had128)"""
    global _H128
    if _H128 is None or _H128.device != v.device:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H128 = (h / math.sqrt(128)).to(v.device)
    shp = v.shape
    return (v.reshape(*shp[:-1], shp[-1] // 128, 128) @ _H128).reshape(shp)


class Exl3Lin:
    """one EXL3 Linear: trellis / suh / svh as quantize_exl3 produced them, W_tr = exllamav3's reconstruct"""

    def __init__(self, t, k, n):
        from exllamav3.modules.quant.exl3 import LinearEXL3
        self.t, self.k, self.n = t, k, n
        lin = LinearEXL3(None, k, n, suh=t["suh"], svh=t["svh"], trellis=t["trellis"], mcg=t.get("mcg"),
                         mul1=t.get("mul1"), out_dtype=torch.float, key="s4")
        self.wtr = lin.get_inner_weight_tensor().double()          # (in, out), the rotated-basis decode
        self.suh = t["suh"].double()
        self.svh = t["svh"].double()

    def __call__(self, x):                                           # y = svh H(W_tr^T H(suh x))
        return self.svh * had(had(self.suh * x) @ self.wtr)

    def dense(self):                                                 # the original-basis W (out, in)
        eye = torch.eye(self.k, dtype=torch.float64, device=self.wtr.device)
        return self(eye).T

    def slice_bytes(self):
        tr = self.t["trellis"].contiguous().cpu()
        assert tr.dtype == torch.int16 and tuple(tr.shape) == (self.k // 16, self.n // 16, tr.shape[2])
        return (tr.numpy().tobytes() + self.t["suh"].contiguous().cpu().numpy().tobytes()
                + self.t["svh"].contiguous().cpu().numpy().tobytes())


def quantize(w_oi, X, K, seed):
    """exllamav3 quantize_exl3 of W [out, in] with H = X^T X (mul1, out scales -- the L251 settings)"""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    X = X.to(DEV).float()
    H = X.T @ X
    H_data = {"H": H, "first_key": "s4", "count": X.shape[0], "finalized": False,
              "num_total": X.shape[0] * X.shape[1], "inf_nan": torch.zeros(2), "device": DEV}
    qa = {"seed": seed, "K": K, "devices": [0], "device_ratios": None, "apply_out_scales": True,
          "debug_dir": None, "mul1": True}
    _wq, perr, t = quantize_exl3(w_oi.to(DEV).float().t().contiguous().clone(), H_data, qa, return_weight_q=True,
                                 progress_str=None, verbose=False)
    return Exl3Lin(t, w_oi.shape[1], w_oi.shape[0]), float(perr)


def relF(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def rows_relF(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1))


# ------------------------------------------------------------------ router
def cmd_router(a, tool):
    W = Weights()
    X = torch.load(f"{ACT}/moe_in.pt")                              # bf16 [N, 2560]
    topk = torch.load(f"{ACT}/topk.pt").long()                      # [N, 10], the source's own router hook
    N = X.shape[0]
    wr = W.get(PFX + "layers.12.mlp.gate.weight", "cpu")
    wsg = W.get(PFX + "layers.12.mlp.shared_expert_gate.weight", "cpu").reshape(-1)
    d = os.path.join(a.out, "router")
    os.makedirs(d, exist_ok=True)
    tofile(X, f"{d}/x.bin"); tofile(wr, f"{d}/wr.bin"); tofile(wsg, f"{d}/wsg.bin")
    run_tool(["router", d, N], tool)
    sel = torch.from_numpy(np.fromfile(f"{d}/sel.bin", dtype=np.int32).reshape(N, 10)).long()
    wts = torch.from_numpy(np.fromfile(f"{d}/wts.bin", dtype=np.float32).reshape(N, 10))
    lg = torch.from_numpy(np.fromfile(f"{d}/logits.bin", dtype=np.float32).reshape(N, 513))
    set_ok = (sel.sort(-1).values == topk.sort(-1).values).all(-1)
    order_ok = (sel == topk).all(-1)
    bad = (~set_ok).nonzero().flatten()
    log(f"router: {N:,} rows; top-10 SET equal on {int(set_ok.sum()):,}, order equal on {int(order_ok.sum()):,}; "
        f"{len(bad)} set mismatches")
    # the source's arithmetic again on this GPU (F.linear in bf16), for the classification and a sanity check
    res = {"rows": N, "set_equal": int(set_ok.sum()), "order_equal": int(order_ok.sum()), "mismatch": []}
    wr_d = wr.to(DEV)
    n_logit_diff, n_hf_redo_bad = 0, 0
    for c0 in range(0, N, 8192):
        xs = X[c0:c0 + 8192].to(DEV)
        hl = F.linear(xs, wr_d)                                      # bf16, as the source
        hp = torch.softmax(hl, dim=-1, dtype=torch.float)
        _, hi = torch.topk(hp, 10, dim=-1)
        n_hf_redo_bad += int((hi.cpu() != topk[c0:c0 + 8192]).any(-1).sum())
        ours = lg[c0:c0 + 8192, :512].to(DEV).to(torch.bfloat16)
        n_logit_diff += int((ours != hl).sum())
    res["hf_recompute_rows_differing_from_capture"] = n_hf_redo_bad
    res["bf16_logits_differing_from_source"] = n_logit_diff
    log(f"router: the source's F.linear recomputed here reproduces the capture on all but {n_hf_redo_bad} rows; "
        f"{n_logit_diff:,} of {N * 512:,} bf16 logits differ from the source's (f32 order vs cuBLAS)")
    for r in bad.tolist():
        xs = X[r:r + 1].to(DEV)
        hl = F.linear(xs, wr_d)[0].float().cpu()
        ours = lg[r, :512].to(torch.bfloat16).float()
        hp = torch.softmax(hl, -1)
        srt = hp.sort(descending=True)
        p10, p11 = float(srt.values[9]), float(srt.values[10])
        e10, e11 = int(srt.indices[9]), int(srt.indices[10])
        only_ours = sorted(set(sel[r].tolist()) - set(topk[r].tolist()))
        only_src = sorted(set(topk[r].tolist()) - set(sel[r].tolist()))
        kind = ("tie in the source's bf16 logits at rank 10/11" if p10 == p11 else
                "boundary logit rounds to a different bf16" if any(float(ours[e]) != float(hl[e]) for e in only_ours + only_src)
                else "UNEXPLAINED")
        m = {"row": r, "ours_only": only_ours, "source_only": only_src, "p10": p10, "p11": p11, "e10": e10, "e11": e11,
             "src_logits": {e: float(hl[e]) for e in only_ours + only_src},
             "our_logits_bf16": {e: float(ours[e]) for e in only_ours + only_src},
             "our_logits_f32": {e: float(lg[r, e]) for e in only_ours + only_src}, "kind": kind}
        res["mismatch"].append(m)
        log(f"  row {r}: ours {only_ours} vs source {only_src}; {kind}; src logits {m['src_logits']} ours bf16 "
            f"{m['our_logits_bf16']} f32 {m['our_logits_f32']}")
    # weights where the order matches: our bf16 weights vs the source's (recomputed)
    ok_rows = order_ok.nonzero().flatten()[:20000]
    xs = X[ok_rows].to(DEV)
    hl = F.linear(xs, wr_d)
    hp = torch.softmax(hl, -1, dtype=torch.float)
    tv, _ = torch.topk(hp, 10, -1)
    tv = (tv / tv.sum(-1, keepdim=True)).to(torch.bfloat16).float().cpu()
    dw = (wts[ok_rows] - tv).abs()
    res["weights_max_abs_diff"] = float(dw.max())
    res["weights_frac_bit_equal"] = float((dw == 0).float().mean())
    log(f"router weights (bf16) on {len(ok_rows):,} order-equal rows: {res['weights_frac_bit_equal'] * 100:.3f}% bit-equal "
        f"to the source's, max |diff| {res['weights_max_abs_diff']:.2e}")
    ties = sum(1 for m in res["mismatch"] if m["kind"].startswith("tie"))
    rnd = sum(1 for m in res["mismatch"] if m["kind"].startswith("boundary"))
    unx = sum(1 for m in res["mismatch"] if m["kind"] == "UNEXPLAINED")
    res["summary"] = {"ties": ties, "rounding": rnd, "unexplained": unx}
    log(f"router mismatches: {ties} ties, {rnd} bf16-rounding flips, {unx} unexplained")
    json.dump(res, open(f"{d}/router.json", "w"), indent=1)
    return unx == 0


# ------------------------------------------------------------------ MoE
def cmd_moe(a, tool):
    cfg = text_config()
    W = Weights()
    X = torch.load(f"{ACT}/moe_in.pt")
    topk = torch.load(f"{ACT}/topk.pt").long()
    N = X.shape[0]
    rows = torch.linspace(0, N - 1, a.rows).long()
    experts = sorted(set(topk[rows].flatten().tolist()))
    log(f"moe: {a.rows} rows, {len(experts)} distinct experts, K gate/up {a.k_expert} down {a.k_expert}, shared {a.k_shared}")
    p = PFX + "layers.12.mlp."
    gu = W.get(p + "experts.gate_up_proj", "cpu")                   # [512, 1280, 2560]
    dn = W.get(p + "experts.down_proj", "cpu")                      # [512, 2560, 640]
    t0 = time.time()
    q = {}
    fit_cap = 4096
    for i, e in enumerate(experts):
        ridx = (topk == e).any(-1).nonzero().flatten()
        ridx = ridx[torch.randperm(len(ridx), generator=torch.Generator().manual_seed(e))[:fit_cap]]
        Xe = X[ridx].to(DEV).float()
        wgu, wd = gu[e].to(DEV), dn[e].to(DEV)                     # gate_up FUSED [1280, 2560], as the container
        mid = F.silu(Xe @ wgu[:640].float().T) * (Xe @ wgu[640:].float().T)
        lgu_, pgu_ = quantize(wgu, Xe, a.k_expert, 1000 + e)
        ld_, pd_ = quantize(wd, mid, a.k_expert, 3000 + e)
        q[e] = (lgu_, ld_)
        if i % 10 == 0:
            log(f"  expert {e}: {len(ridx)} fit rows, proxy err gate_up {pgu_:.4f} down {pd_:.4f} ({time.time() - t0:.0f}s)")
    Xs = X[torch.randperm(N, generator=torch.Generator().manual_seed(7))[:8192]].to(DEV).float()
    sg_w, su_w, sd_w = (W.get(p + f"shared_expert.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
    smid = F.silu(Xs @ sg_w.float().T) * (Xs @ su_w.float().T)
    qsg, _ = quantize(sg_w, Xs, a.k_shared, 11)
    qsu, _ = quantize(su_w, Xs, a.k_shared, 12)
    qsd, _ = quantize(sd_w, smid, a.k_shared, 13)
    log(f"moe: quantized {2 * len(experts) + 3} matrices in {time.time() - t0:.0f}s")

    d = os.path.join(a.out, f"moe-K{a.k_expert}")
    os.makedirs(d, exist_ok=True)
    xr = X[rows]
    tofile(xr, f"{d}/x.bin")
    tofile(W.get(p + "gate.weight", "cpu"), f"{d}/wr.bin")
    tofile(W.get(p + "shared_expert_gate.weight", "cpu").reshape(-1), f"{d}/wsg.bin")
    with open(f"{d}/experts.idx", "wb") as f:
        f.write(struct.pack(f"<I{len(experts)}I", len(experts), *experts))
    for j, name in enumerate(("gate_up", "down")):
        with open(f"{d}/exp_{name}.bin", "wb") as f:
            for e in experts:
                f.write(q[e][j].slice_bytes())
    for lin, name in ((qsg, "gate"), (qsu, "up"), (qsd, "down")):
        open(f"{d}/shared_{name}.bin", "wb").write(lin.slice_bytes())
    k2 = 2 * a.k_expert
    run_tool(["moe", d, a.rows, k2, k2, 2 * a.k_shared, 2 * a.k_shared, 2 * a.k_shared], tool)
    out = torch.from_numpy(np.fromfile(f"{d}/out.bin", dtype=np.float32).reshape(a.rows, 2560)).double()
    sel = torch.from_numpy(np.fromfile(f"{d}/sel.bin", dtype=np.int32).reshape(a.rows, 10)).long()
    wts = torch.from_numpy(np.fromfile(f"{d}/wts.bin", dtype=np.float32).reshape(a.rows, 10)).double()
    sgt = torch.from_numpy(np.fromfile(f"{d}/sgate.bin", dtype=np.float32)).double()
    same_route = bool((sel.sort(-1).values == topk[rows].sort(-1).values).all())

    # emu: the device's arithmetic in fp64 from exllamav3's reconstruct
    x = xr.to(DEV).double()
    x8 = e4m3_dec(x)
    emu = torch.zeros(a.rows, 2560, dtype=torch.float64, device=DEV)
    rec = torch.zeros_like(emu)
    hp = torch.softmax(F.linear(xr.to(DEV), W.get(p + "gate.weight")), -1, dtype=torch.float)   # the source's routing
    src_v, src_i = torch.topk(hp, 10, -1)
    src_v = src_v / src_v.sum(-1, keepdim=True)
    for r in range(a.rows):
        for k in range(10):
            lgu_, ld_ = q[int(sel[r, k])]
            z = lgu_(x8[r:r + 1])
            v = F.silu(z[:, :640]) * z[:, 640:] * float(wts[r, k])
            t = e4m3_dec(had(ld_.suh * v))
            emu[r] += (ld_.svh * had(t @ ld_.wtr))[0]
        for k in range(10):                                         # recon: the source's semantics, no A8
            lgu_, ld_ = q[int(src_i[r, k])]
            z = lgu_(x[r:r + 1])
            rec[r] += (ld_(F.silu(z[:, :640]) * z[:, 640:]) * float(src_v[r, k]))[0]
    hs = qsd(e4m3_dec(F.silu(qsg(x8)) * qsu(x8)))
    emu += sgt.to(DEV)[:, None] * hs
    sgl = torch.sigmoid((x @ W.get(p + "shared_expert_gate.weight").double().T))
    rec += sgl * qsd(F.silu(qsg(x)) * qsu(x))
    # hf: the source module
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    with torch.device("meta"):
        blk = M.Qwen4ExpTextSparseMoeBlock(cfg)
    sd = W.prefixed(p)
    missing, unexpected = blk.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        hf = blk(xr.to(DEV).view(1, a.rows, 2560)).view(a.rows, 2560).double()
    out = out.to(DEV)
    m1 = torch.from_numpy(np.fromfile(f"{d}/out_m1.bin", dtype=np.float32).reshape(a.rows, 2560)).double().to(DEV)
    res = {"K_expert": a.k_expert, "K_shared": a.k_shared, "rows": a.rows, "experts": len(experts),
           "routing_equals_source": same_route,
           "dev_vs_emu": relF(out, emu), "dev_vs_emu_rows_max": float(rows_relF(out, emu).max()),
           "dev_m1_vs_batch_bit_equal": bool((m1 == out).all()),
           "dev_vs_hf": relF(out, hf), "recon_vs_hf": relF(rec, hf), "emu_vs_recon": relF(emu, rec),
           "hf_vs_fp64_source": None}
    log(f"moe K={a.k_expert}: routing == source {same_route}; dev vs emu (exllamav3 reconstruct + the A8 points, fp64) "
        f"{res['dev_vs_emu']:.2e} (worst row {res['dev_vs_emu_rows_max']:.2e}); M=1 rows bit-equal {res['dev_m1_vs_batch_bit_equal']}")
    log(f"moe K={a.k_expert}: dev vs HF bf16 block {res['dev_vs_hf']:.3e}; recon (quant only) vs HF {res['recon_vs_hf']:.3e}; "
        f"A8 alone (emu vs recon) {res['emu_vs_recon']:.3e}")
    json.dump(res, open(f"{d}/moe.json", "w"), indent=1)
    return res["dev_vs_emu"] < 1e-5 and res["dev_m1_vs_batch_bit_equal"]


# ------------------------------------------------------------------ capture for GR / PLE
class DiskNgram(torch.nn.Module):
    """ple_embedding.ngram_embedding with rows read from the checkpoint's 128 shards (research/l251/qwen_capture.py)"""

    def __init__(self, W, layer, n_shards):
        super().__init__()
        self.W, self.layer, self.n = W, layer, n_shards
        self.weight = torch.empty(1, device=DEV)

    def _name(self, s):
        return f"{PFX}layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"

    def forward(self, ids):
        flat = ids.flatten().cpu()
        rows_per = self.W.f(self._name(0)).get_slice(self._name(0)).get_shape()[0]
        out = torch.empty(len(flat), 160, dtype=torch.bfloat16)
        for s in torch.unique(flat // rows_per).tolist():
            sel = (flat // rows_per == s).nonzero().flatten()
            sl = self.W.f(self._name(s)).get_slice(self._name(s))
            for i in sel.tolist():
                r = int(flat[i]) - s * rows_per
                out[i] = sl[r:r + 1][0]
        return out.to(ids.device).view(*ids.shape, 160)


def build_layer(cfg, W, i):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    with torch.device("meta"):
        layer = M.Qwen4ExpTextDecoderLayer(cfg, i)
    sd = W.prefixed(f"{PFX}layers.{i}.", skip=("ngram_embedding.shard_",))
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and set(missing) <= {"ple.ple_embedding.ngram_embedding.weight"}, (missing, unexpected)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = DiskNgram(W, i, cfg.split_ngram_parts)
    return layer.eval()


@torch.no_grad()
def cmd_capture(a, tool):
    from transformers import AutoTokenizer
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    cfg = text_config()
    W = Weights()
    tok = AutoTokenizer.from_pretrained(TOK)
    texts = [json.loads(l)["text"] for l in open(CALIB)]
    seqs = [torch.tensor(tok(texts[i * 37 % len(texts)], add_special_tokens=False)["input_ids"][:a.seq_len])
            for i in range(a.n_seq)]
    emb = W.get(f"{PFX}embed_tokens.weight")
    rotary = M.Qwen4ExpTextRotaryEmbedding(config=cfg).to(DEV)
    l0, l1 = build_layer(cfg, W, 0), build_layer(cfg, W, 1)
    mixer = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    mixer.load_state_dict(W.prefixed(f"{PFX}hyper_connection_mixer."), strict=True)
    mixer = mixer.to(DEV).eval()
    cap = {k: [] for k in ("ids", "l1_in", "emb", "ple_out", "gr_x", "gr_inj", "mix_x", "seqlen")}
    for ids in seqs:
        T = ids.numel()
        h = emb[ids.to(DEV)].unsqueeze(0).repeat(1, 1, cfg.hc_count)
        pid = torch.arange(T, device=DEV).view(1, 1, -1).expand(3, 1, -1)
        pe = rotary(torch.empty(1, T, cfg.hidden_size, device=DEV, dtype=torch.bfloat16), pid)
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril().view(1, 1, T, T)
        h = l0(h, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None,
               ple_input_ids=ids.view(1, -1).to(DEV))
        cap["l1_in"].append(h[0].cpu())
        e = l1.ple.ple_embedding(ids.view(1, -1).to(DEV), None)
        cap["emb"].append(e[0].cpu())
        h2 = h + l1.ple(h, ids.view(1, -1).to(DEV), None)
        cap["ple_out"].append(h2[0].cpu())
        xg, _, inj = l1.attn_hyper_connection(h2)
        cap["gr_x"].append(xg[0].cpu()); cap["gr_inj"].append(inj[0].cpu())
        cap["mix_x"].append(mixer(h2)[0].cpu())
        cap["ids"].append(ids); cap["seqlen"].append(torch.tensor([T]))
    os.makedirs(CAP, exist_ok=True)
    for k, v in cap.items():
        torch.save(torch.cat(v), f"{CAP}/{k}.pt")
    log(f"capture: {a.n_seq} sequences, {sum(s.numel() for s in seqs):,} tokens -> {CAP}")
    return True


# ------------------------------------------------------------------ GR
def mxfp8_lt_bytes(w):
    """W [out, in] (bf16) -> the builder's mxfp8_lt bytes (tools/container/producers.py) and the decoded W"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
    from producers import _quantize_fp8_e4m3_planes, swizzle_sf
    out, inp = w.shape
    sc, codes = _quantize_fp8_e4m3_planes(w.float().cpu().numpy())
    dec = torch.from_numpy(codes.view(np.uint8).copy()).view(torch.float8_e4m3fn).double() \
        .reshape(out, inp // 32, 32) * torch.exp2(torch.from_numpy(sc.astype(np.float64) - 127))[..., None]
    return codes.tobytes() + swizzle_sf(sc, out, inp // 32).tobytes(), dec.reshape(out, inp)


def gr_fp64(h, norm_w, wd, wu, wi, a8):
    """the GR read in fp64 (with the device's A8 points when a8)"""
    x = h.double().view(*h.shape[:-1], 4, 2560)
    xn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).flatten(-2) * (1.0 + norm_w.double())
    xin = e4m3_dec(xn) if a8 else xn
    a_ = F.silu(xin @ wd.T / 4)
    ain = e4m3_dec(a_) if a8 else a_
    g = torch.sigmoid(ain @ wu.T)
    mixed = (g.view(*g.shape[:-1], 4, 2560) * xn.view(*xn.shape[:-1], 4, 2560)).mean(-2)
    inj = 2 * torch.sigmoid(xn @ wi.double().T / 4) if wi is not None else None
    return mixed, inj


def cmd_gr(a, tool):
    W = Weights()
    h = torch.load(f"{CAP}/ple_out.pt")                             # the streams entering layer 1's attention site
    N = h.shape[0]
    ok = True
    for site, pfx, hf_x, hf_inj in (("attn", f"{PFX}layers.1.attn_hyper_connection.", "gr_x", "gr_inj"),
                                    ("mixer", f"{PFX}hyper_connection_mixer.", "mix_x", None)):
        d = os.path.join(a.out, f"gr-{site}")
        os.makedirs(d, exist_ok=True)
        norm = W.get(pfx + "hc_norm.weight", "cpu")
        down_b, wd = mxfp8_lt_bytes(W.get(pfx + "input_mix_weight_down.weight", "cpu"))
        up_b, wu = mxfp8_lt_bytes(W.get(pfx + "input_mix_weight_up.weight", "cpu"))
        tofile(h, f"{d}/streams.bin"); tofile(norm, f"{d}/norm.bin")
        open(f"{d}/down.bin", "wb").write(down_b); open(f"{d}/up.bin", "wb").write(up_b)
        wi = None
        if hf_inj:
            wi = W.get(pfx + "block_inject_weight.weight", "cpu")
            tofile(wi, f"{d}/inject.bin")
        elif os.path.exists(f"{d}/inject.bin"):
            os.remove(f"{d}/inject.bin")
        run_tool(["gr", d, N], tool)
        x = torch.from_numpy(np.fromfile(f"{d}/x.bin", dtype=np.int16).reshape(N, 2560)).view(torch.bfloat16).double()
        inj = torch.from_numpy(np.fromfile(f"{d}/inj.bin", dtype=np.float32).reshape(N, 4)).double()
        refx = torch.from_numpy(np.fromfile(f"{d}/ref_x.bin", dtype=np.float64).reshape(-1, 2560))
        nr = refx.shape[0]
        hfx = torch.load(f"{CAP}/{hf_x}.pt").double()
        e_mx, e_inj = gr_fp64(h.to(DEV), norm.to(DEV), wd.to(DEV), wu.to(DEV), wi.to(DEV) if wi is not None else None, True)
        s_mx, _ = gr_fp64(h.to(DEV), norm.to(DEV), W.get(pfx + "input_mix_weight_down.weight").double(),
                          W.get(pfx + "input_mix_weight_up.weight").double(), None, False)
        res = {"site": site, "rows": N,
               "dev_vs_host_ref16": relF(x[:nr], refx), "dev_vs_emu": relF(x, e_mx.cpu()),
               "dev_vs_hf_bf16": relF(x, hfx), "emu_vs_srcfp64": relF(e_mx, s_mx), "hf_bf16_vs_srcfp64": relF(hfx.to(DEV), s_mx)}
        if hf_inj:
            hfi = torch.load(f"{CAP}/{hf_inj}.pt").double()
            res["inj_dev_vs_emu"] = relF(inj, e_inj.cpu())
            res["inj_dev_vs_hf_bf16"] = relF(inj, hfi)
        log(f"gr {site}: dev vs host ref (16 rows) {res['dev_vs_host_ref16']:.2e}; vs fp64 emu (MXFP8 + A8) {res['dev_vs_emu']:.2e}; "
            f"vs HF bf16 {res['dev_vs_hf_bf16']:.3e}  [MXFP8+A8 alone {res['emu_vs_srcfp64']:.3e}; HF bf16 vs fp64 source "
            f"{res['hf_bf16_vs_srcfp64']:.3e}]" + (f"; inj vs emu {res['inj_dev_vs_emu']:.2e}, vs HF {res['inj_dev_vs_hf_bf16']:.2e}" if hf_inj else ""))
        json.dump(res, open(f"{d}/gr.json", "w"), indent=1)
        ok &= res["dev_vs_host_ref16"] < 4e-3 and res["dev_vs_emu"] < 4e-3
    return ok


# ------------------------------------------------------------------ PLE
def ple_fp64(emb, h, lens, key, value, nk, nq, nc, cw, a8):
    """the PLE injection in fp64; key / value callables on the (A8-encoded) embedding"""
    e = e4m3_dec(emb.double()) if a8 else emb.double()
    k, v = key(e), value(e)
    kk = k.view(-1, 4, 2560)
    kn = kk * torch.rsqrt(kk.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + nk.double().view(4, 2560))
    q = h.double().view(-1, 4, 2560)
    qn = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + nq.double().view(4, 2560))
    g = (kn * qn).sum(-1, keepdim=True) / math.sqrt(2560)
    g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
    gv = torch.sigmoid(g) * v.view(-1, 1, 2560)
    gvn = gv * torch.rsqrt(gv.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + nc.double().view(4, 2560))
    gv, gvn = gv.flatten(-2), gvn.flatten(-2)
    out = torch.empty_like(gv)
    r0 = 0
    w = cw.double().view(10240, 4)
    for L in lens:
        x = F.pad(gvn[r0:r0 + L].T, (9, 0))                           # [C, 9 + L]
        y = sum(w[:, m:m + 1] * x[:, 3 * m:3 * m + L] for m in range(4))
        out[r0:r0 + L] = gv[r0:r0 + L] + F.silu(y.T)
        r0 += L
    return h.double() + out


def cmd_ple(a, tool):
    cfg = text_config()
    W = Weights()
    emb = torch.load(f"{CAP}/emb.pt")
    h = torch.load(f"{CAP}/l1_in.pt")
    hf = torch.load(f"{CAP}/ple_out.pt").double()
    lens = torch.load(f"{CAP}/seqlen.pt").tolist()
    N = emb.shape[0]
    pfx = f"{PFX}layers.1.ple."
    kw, vw = W.get(pfx + "key_proj.weight"), W.get(pfx + "value_proj.weight")
    X = emb.to(DEV).float()
    qk, pk = quantize(kw, X, a.k_dense, 21)
    qv, pv = quantize(vw, X, a.k_dense, 22)
    log(f"ple: key/value quantized at K={a.k_dense} on {N} embedding rows (proxy err {pk:.4f} / {pv:.4f})")
    d = os.path.join(a.out, f"ple-K{a.k_dense}")
    os.makedirs(d, exist_ok=True)
    tofile(emb, f"{d}/emb.bin"); tofile(h, f"{d}/streams.bin")
    with open(f"{d}/seqs.bin", "wb") as f:
        f.write(struct.pack(f"<i{len(lens)}i", len(lens), *lens))
    open(f"{d}/key.bin", "wb").write(qk.slice_bytes()); open(f"{d}/value.bin", "wb").write(qv.slice_bytes())
    nk, nq, nc = (W.get(pfx + f"{n}.weight", "cpu") for n in ("norm_key", "norm_query", "norm_conv"))
    cw = W.get(pfx + "conv1d.weight", "cpu").reshape(10240, 4)
    tofile(nk, f"{d}/norm_key.bin"); tofile(nq, f"{d}/norm_query.bin"); tofile(nc, f"{d}/norm_conv.bin"); tofile(cw, f"{d}/conv.bin")
    run_tool(["ple", d, N, 2 * a.k_dense, 2 * a.k_dense], tool)
    out = torch.from_numpy(np.fromfile(f"{d}/out.bin", dtype=np.int16).reshape(N, 10240)).view(torch.bfloat16).double()
    ref = torch.from_numpy(np.fromfile(f"{d}/ref_out.bin", dtype=np.int16).reshape(-1, 10240)).view(torch.bfloat16).double()
    emu = ple_fp64(emb.to(DEV), h.to(DEV), lens, qk, qv, nk.to(DEV), nq.to(DEV), nc.to(DEV), cw.to(DEV), True).cpu()
    src = ple_fp64(emb.to(DEV), h.to(DEV), lens, lambda e: e @ kw.double().T, lambda e: e @ vw.double().T,
                   nk.to(DEV), nq.to(DEV), nc.to(DEV), cw.to(DEV), False).cpu()
    h0 = h.double()
    delta = lambda y: y - h0                                          # the injection, graded on its own scale
    nr = ref.shape[0]
    res = {"rows": N, "K": a.k_dense,
           "stream_dev_vs_hostref": relF(out[:nr], ref), "inject_dev_vs_emu": relF(delta(out), delta(emu)),
           "inject_dev_vs_hf": relF(delta(out), delta(hf)), "inject_emu_vs_src": relF(delta(emu), delta(src)),
           "inject_hf_vs_src": relF(delta(hf), delta(src)), "stream_dev_vs_hf": relF(out, hf)}
    log(f"ple K={a.k_dense}: injection dev vs fp64 emu (EXL3 reconstruct + A8) {res['inject_dev_vs_emu']:.2e}; vs HF bf16 "
        f"{res['inject_dev_vs_hf']:.3e} [quant+A8 alone {res['inject_emu_vs_src']:.3e}; HF bf16 vs fp64 source "
        f"{res['inject_hf_vs_src']:.3e}]; streams vs host ref {res['stream_dev_vs_hostref']:.2e}")
    json.dump(res, open(f"{d}/ple.json", "w"), indent=1)
    return res["inject_dev_vs_emu"] < 1e-2


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["router", "moe", "capture", "gr", "ple"])
    ap.add_argument("--pulsar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--k-expert", type=int, default=4)
    ap.add_argument("--k-shared", type=int, default=5)
    ap.add_argument("--k-dense", type=int, default=5)
    ap.add_argument("--n-seq", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=384)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    LOGF = open(os.path.join(a.out, "s4_xcheck.log"), "a")
    sha = subprocess.run(["git", "-C", a.pulsar, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", a.pulsar, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    log(f"== {a.cmd} {' '.join(sys.argv[2:])}  pulsar {sha}{' DIRTY' if dirty else ''}  torch {torch.__version__}")
    tool = os.path.join(a.pulsar, "tests/qwen_xcheck")
    ok = {"router": cmd_router, "moe": cmd_moe, "capture": cmd_capture, "gr": cmd_gr, "ple": cmd_ple}[a.cmd](a, tool)
    log(f"== {a.cmd}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

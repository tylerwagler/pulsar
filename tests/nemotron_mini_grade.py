#!/usr/bin/env python3
"""The Nemotron-H forward's block grade on the BF16 mini (L296 N7; rule 7).

  nemotron_mini_grade.py prep  REF_PT OUT_DIR
      OUT_DIR/ids.bin: the reference capture's prompt as int32 ids (tests/nemotron_fwd_gate's input)
  nemotron_mini_grade.py grade REF_PT MINI_DIR CAP_DIR [--bf16 REF_BF16_PT] [--key dt0|clamp]
      grades what `nemotron_fwd_gate capture` left in CAP_DIR against the transformers fp32 capture REF_PT
      (tools/nemotron/capture_ref.py): per HF block, the residual after it and the block's own output (the mixer's /
      the MoE's delta), on the prefill path (the one-chunk prompt) and the decode path (one row a step); the final norm;
      the logits of the decode path at every position and of the prefill's last row -- rel-fro, top-1 agreement and
      KL(ref || ours).  --bf16: transformers' own bf16 run of the same prompt (capture_ref.py --dtype bf16), graded the
      same way against the fp32 reference -- the bf16 rounding floor the engine is read against.  --key: which of the
      reference's two forwards (`dt0`, the default: dt unclamped, the checkpoint's time_step_limit (0, inf) as vLLM
      serves it; `clamp`: transformers' time_step_min clamp in its chunked path).
      MINI_DIR is the HF mini (the final norm's weight and eps: the engine's last dump is the residual before it; the
      grade applies vLLM's rmsnorm, one bf16 rounding, as the head's input).

The engine's taps (PULSAR_CUDA_GRAPH_DUMP_PREFIX = CAP_DIR/d): for pulsar layer il -- nemotron_mix (the mixer's delta),
nemotron_mid (the residual after the mixer, a layer with the MoE), nemotron_ffn (the MoE's delta), nemotron_h (the
residual after the layer).  pulsar layer il groups HF block b (its mixer) and b + 1 (the MoE after it, when it has one).

Metrics: rel-fro = ||ours - ref||_F / ||ref||_F.  Needs torch + safetensors (~/inkling/.venv on Sparky).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def rel_fro(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def kl_top1(ref_logits, ours):
    lp, lq = F.log_softmax(ref_logits.double(), -1), F.log_softmax(ours.double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (ref_logits.argmax(-1) == ours.argmax(-1)).double().mean().item()
    return kl.mean().item(), kl.max().item(), top1


def bf16(x):
    return x.to(torch.bfloat16).float()


def final_norm(h, w, eps):
    """vLLM's rmsnorm on the bf16 residual: one rounding to bf16 (the head's input)"""
    h = h.float()
    return bf16(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def load_tap(cap, name, il, T, E):
    """the prefill's rows [T][E] of one tap (its one multi-row dump) and the decode path's (one-row dumps)"""
    prows, drows = {}, {}
    for f in glob.glob(os.path.join(cap, f"d_{name}-{il}_pos*.bin")):
        p = int(re.search(r"_pos(\d+)(?:_c\d+)?\.bin$", f).group(1))
        a = np.fromfile(f, dtype=np.float32).reshape(-1, E)
        if a.shape[0] == 1:
            drows[p] = a[0]
        else:
            for i in range(a.shape[0]):
                prows[p - a.shape[0] + 1 + i] = a[i]
    pre = torch.from_numpy(np.stack([prows[p] for p in range(T)])) if len(prows) == T else None
    dec = torch.from_numpy(np.stack([drows[p] for p in range(T)])) if len(drows) == T else None
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "grade"])
    ap.add_argument("ref")
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--bf16", default=None)
    ap.add_argument("--key", choices=["dt0", "clamp"], default="dt0")
    args = ap.parse_args()
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    ids = ref["input_ids"][0].to(torch.int32)
    if args.mode == "prep":
        os.makedirs(args.a, exist_ok=True)
        ids.numpy().tofile(os.path.join(args.a, "ids.bin"))
        print(f"prep: {ids.numel()} ids -> {args.a}/ids.bin")
        return
    mini, cap = args.a, args.b
    sfx = "_dt0" if args.key == "dt0" else ""
    cfg = json.load(open(os.path.join(mini, "config.json")))["llm_config"]
    eps, E = cfg["layer_norm_epsilon"], cfg["hidden_size"]
    idx = json.load(open(os.path.join(mini, "model.safetensors.index.json")))["weight_map"]
    key = next(k for k in idx if k.endswith("backbone.norm_f.weight"))
    with safe_open(os.path.join(mini, idx[key]), "pt") as f:
        w_norm = f.get_tensor(key)
    blocks = list(ref["blocks"])
    T = ids.numel()
    hs = [h[0].float() for h in ref["hidden_states" + sfx]]
    mo = [m[0].float() for m in ref["mixer_out" + sfx]]
    fn_ref = ref["final_norm" + sfx][0].float()
    logits_ref = ref["logits" + sfx][0].float()
    tb = torch.load(args.bf16, map_location="cpu", weights_only=False) if args.bf16 else None
    if tb is not None and not torch.equal(tb["input_ids"], ref["input_ids"]):
        raise SystemExit("the bf16 capture's prompt is not the reference's")

    # pulsar's grouping: a mixer and the MoE after it
    layers, b = [], 0
    while b < len(blocks):
        moe = b + 1 < len(blocks) and blocks[b + 1] == "moe"
        layers.append((b, b + 1 if moe else None))
        b += 2 if moe else 1
    print(f"grade ({args.key}): {T} tokens, HF blocks {blocks} -> pulsar layers {layers}")
    print(f"  {'':44s} {'ours (prefill)':>15s} {'ours (decode)':>15s} {'tf bf16':>12s}   decode == prefill")

    def tb_t(key, b):
        return tb[key + sfx][b][0].float() if tb is not None else None

    def row(name, pre, dec, ref_t, tbt):
        r = lambda x: f"{rel_fro(x, ref_t):15.3e}" if x is not None else f"{'no dump':>15s}"
        t = f"{rel_fro(tbt, ref_t):12.3e}" if tbt is not None else f"{'-':>12s}"
        same = "" if pre is None or dec is None else ("BYTES" if torch.equal(pre, dec) else f"{rel_fro(dec, pre):.2e}")
        print(f"  {name:44s} {r(pre)} {r(dec)} {t}   {same}")

    last = None
    for il, (bm, be) in enumerate(layers):
        kind = blocks[bm]
        mix = load_tap(cap, "nemotron_mix", il, T, E)
        after_mix = load_tap(cap, "nemotron_mid" if be is not None else "nemotron_h", il, T, E)
        row(f"L{il} block {bm} {kind:9s} mixer out", *mix, mo[bm], tb_t("mixer_out", bm))
        row(f"L{il} block {bm} {kind:9s} residual", *after_mix, hs[bm], tb_t("hidden_states", bm))
        last = after_mix
        if be is not None:
            ffn = load_tap(cap, "nemotron_ffn", il, T, E)
            after = load_tap(cap, "nemotron_h", il, T, E)
            row(f"L{il} block {be} moe       MoE out", *ffn, mo[be], tb_t("mixer_out", be))
            row(f"L{il} block {be} moe       residual", *after, hs[be], tb_t("hidden_states", be))
            last = after
    fpre = final_norm(last[0], w_norm, eps) if last[0] is not None else None
    fdec = final_norm(last[1], w_norm, eps) if last[1] is not None else None
    row("final norm (norm_f, vLLM's one rounding)", fpre, fdec, fn_ref,
        tb["final_norm" + sfx][0].float() if tb is not None else None)

    W = os.path.getsize(os.path.join(cap, "decode_logits.f32")) // (4 * T)
    lg = torch.from_numpy(np.fromfile(os.path.join(cap, "decode_logits.f32"), dtype=np.float32).reshape(T, W))
    kl_m, kl_x, top1 = kl_top1(logits_ref, lg)
    print(f"  logits (decode path, all {T} positions) vs fp32 ref: rel-fro {rel_fro(lg, logits_ref):.3e}, "
          f"KL mean {kl_m:.4e} max {kl_x:.4e}, top-1 {100 * top1:.2f}%")
    if tb is not None:
        tl = tb["logits" + sfx][0].float()
        kl_m, kl_x, top1 = kl_top1(logits_ref, tl)
        print(f"  transformers bf16 logits vs fp32 ref:              rel-fro {rel_fro(tl, logits_ref):.3e}, "
              f"KL mean {kl_m:.4e} max {kl_x:.4e}, top-1 {100 * top1:.2f}%")
        kl_m, kl_x, top1 = kl_top1(tl, lg)
        print(f"  ours (decode) vs transformers bf16:                rel-fro {rel_fro(lg, tl):.3e}, "
              f"KL mean {kl_m:.4e} max {kl_x:.4e}, top-1 {100 * top1:.2f}%")
    pl = torch.from_numpy(np.fromfile(os.path.join(cap, "prefill_last.f32"), dtype=np.float32))
    print(f"  prefill last row vs ref: rel-fro {rel_fro(pl, logits_ref[-1]):.3e}, argmax {pl.argmax().item()} / ref "
          f"{logits_ref[-1].argmax().item()}; vs the decode path's last row: rel-fro {rel_fro(pl, lg[-1]):.3e}")


if __name__ == "__main__":
    main()

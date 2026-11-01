#!/usr/bin/env python3
"""The Gemma 4 forward's grade (L338 E4c; rule 7): pulsar against exllamav3 on the SAME EXL3 pack, both against the
transformers BF16 reference.

  gemma4_grade.py prep  REF_PT OUT_DIR
      OUT_DIR/<prompt>.bin: each reference prompt as int32 ids (tests/nemotron_fwd_gate capture's input -- the core
      driver's forward gate is family-neutral; prompts under its 64-id floor are skipped)
  gemma4_grade.py exl3  PACK IDS_BIN N_LAST OUT_PREFIX
      the yardstick: exllamav3 (the version that made PACK, importable) runs the same pack on the same ids, no cache:
      OUT_PREFIX_hidden.npy (f16 [n_layer][T][E], every decoder block's output) and OUT_PREFIX_logits.f32 (the last
      N_LAST positions).  The same weights through another engine: pulsar's distance to it is the engine's own, both
      engines' distance to the reference the quantization's.  (aarch64 builds of its extension link -lgcc.)
  gemma4_grade.py grade REF_PT PROMPT CAP_DIR BASE_DIR [--exl3 OUT_PREFIX] [--decode DEC_DIR]
      grades what `nemotron_fwd_gate capture <container> <ids> CAP_DIR` left in CAP_DIR, run with
      PULSAR_CUDA_GRAPH_DUMP_PREFIX=CAP_DIR/d PULSAR_CUDA_GRAPH_DUMP_NAME=gemma4 (no position filter):
        hidden   per layer, the residual after it (the driver's "gemma4" tap): every prompt row, the prefill's chunks
                 walked back from T - 1 (the sync cuts a prompt at its resume grid; each chunk's tap is the first write
                 at its last position) -- against transformers' hidden_states[il + 1] (the last layer: before the final
                 norm, so to the reference only through the logits) and against exllamav3's block output
        logits   every reference-logit position: pulsar's residual through the final norm and BASE_DIR's tied BF16
                 head (soft-capped) -- the trunk's grade, the head apart -- KL(ref || x) mean / median / max, top-1
                 agreement, and the NLL of the prompt's real next tokens (the quality that matters: the BF16 model's
                 own NLL is the floor); the prefill's last row through pulsar's own EXL3 head (prefill_last.f32);
                 when the capture decoded (decode_logits.f32: every position one row a step, the decode arms and the
                 EXL3 head; --decode: from DEC_DIR, a capture with its own taps filtered), those rows the same way

rel-fro = ||x - y||_F / ||y||_F; KL and NLL in nats over the full vocabulary.  Needs torch + safetensors.
Exit 0 always: the grade is read, its numbers recorded in the ledger row (L338).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F


def rel_fro(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def kl_stats(ref, x):
    """KL(ref || x) per row -> mean, median, max; and the top-1 agreement"""
    lr, lx = F.log_softmax(ref.double(), -1), F.log_softmax(x.double(), -1)
    k = (lr.exp() * (lr - lx)).sum(-1)
    return k.mean().item(), k.median().item(), k.max().item(), (ref.argmax(-1) == x.argmax(-1)).double().mean().item()


def read_f32(path, width):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, width))


def prep(a):
    ref = torch.load(a.ref, weights_only=False)
    os.makedirs(a.out, exist_ok=True)
    for name, r in ref.items():
        ids = r["ids"].numpy().astype(np.int32)
        if len(ids) < 64:
            print(f"{name}: {len(ids)} ids, under the forward gate's 64 -- skipped")
            continue
        ids.tofile(os.path.join(a.out, f"{name}.bin"))
        print(f"{name}: {len(ids)} ids -> {name}.bin")


def exl3(a):
    from exllamav3 import Config, Model
    ids = torch.from_numpy(np.fromfile(a.ids, dtype=np.int32).astype(np.int64))[None]
    model = Model.from_config(Config.from_directory(a.pack))
    model.load(progressbar=False)
    caps = {}
    for m in model.modules:   # every decoder block's output: wrap TransformerBlock.forward
        if type(m).__name__ == "TransformerBlock" and ".language_model.layers." in m.key:
            def wrap(orig, il):
                def f(x, params, out_dtype=None):
                    y = orig(x, params, out_dtype)
                    caps[il] = y.detach().float().reshape(-1, y.shape[-1]).cpu()
                    return y
                return f
            m.forward = wrap(m.forward, int(m.key.split(".")[-1]))
    with torch.inference_mode():
        y = model.forward(ids, {"attn_mode": "flash_attn_nc", "last_tokens_only": a.n_last})
    y = y.float().reshape(-1, y.shape[-1]).cpu()
    y.numpy().astype(np.float32).tofile(a.out + "_logits.f32")
    np.save(a.out + "_hidden.npy", torch.stack([caps[i] for i in sorted(caps)]).to(torch.float16).numpy())
    print(f"exllamav3: {ids.shape[1]} ids, {len(caps)} block taps, the last {y.shape[0]} positions' logits -> {a.out}_*")


def first_tap(pre, il, pos):
    """the first write of layer il's tap at pos (the dump numbers each (name, layer, pos) write `_c<k>`), or None"""
    fs = glob.glob(f"{pre}_gemma4-{il}_pos{pos}.bin") + glob.glob(f"{pre}_gemma4-{il}_pos{pos}_c*.bin")
    if not fs:
        return None
    return min(fs, key=lambda f: int(m.group(1)) if (m := re.search(r"_c(\d+)\.bin$", f)) else 0)


def prefill_rows(pre, il, T, E):
    """layer il's residual at every prompt row: the prefill's chunk taps walked back from T - 1"""
    parts, end = [], T - 1
    while end >= 0:
        f = first_tap(pre, il, end)
        if f is None:
            raise SystemExit(f"layer {il}: no tap ends at position {end} (capture without a position filter)")
        h = read_f32(f, E)
        parts.insert(0, h)
        end -= h.shape[0]
    if end != -1:
        raise SystemExit(f"layer {il}: the chunk taps do not tile [0, {T})")
    return torch.cat(parts)


def tensors(base, keys):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    out = []
    for k in keys:
        with safe_open(os.path.join(base, idx[k]), "pt") as f:
            out.append(f.get_tensor(k))
    return out


def grade(a):
    ref = torch.load(a.ref, weights_only=False)[a.prompt]
    hid = ref["hidden"].float()          # [n_layer + 1][T][E]: the embedding, after each layer, the last normed
    L, T, E = hid.shape[0] - 1, hid.shape[1], hid.shape[2]
    pre = os.path.join(a.cap, "d")
    ours = [prefill_rows(pre, il, T, E) for il in range(L)]
    ex = torch.from_numpy(np.load(a.exl3 + "_hidden.npy").astype(np.float32)) if a.exl3 else None
    print(f"{a.prompt}: {T} positions, {L} layers -- residual rel-fro after each layer, every prompt row")
    print(" layer   pulsar-vs-bf16   exl3-vs-bf16   pulsar-vs-exl3")
    for il in range(L):
        b = hid[il + 1] if il < L - 1 else None   # transformers' last entry is the normed residual
        pb = f"{rel_fro(ours[il], b):.5f}" if b is not None else "   --  "
        eb = f"{rel_fro(ex[il], b):.5f}" if ex is not None and b is not None else "   --  "
        pe = f"{rel_fro(ours[il], ex[il]):.5f}" if ex is not None else "   --  "
        print(f" {il:5d}   {pb}          {eb}        {pe}")
    lg = ref["logits"].float()
    n, first = lg.shape[0], ref["logits_first_pos"]
    W, nw = tensors(a.base, ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"])
    h = ours[L - 1][first:first + n]
    x = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.float()).to(torch.bfloat16)
    soft = 30.0   # final_logit_softcapping
    trunk = torch.tanh((x @ W.T).float() / soft) * soft
    tgt = ref["ids"][first + 1:first + n].long()   # the real next token of each graded position but the last
    print(f"logits over {n} positions        KL mean   KL median  KL max    top-1     NLL (real next tokens)")
    rows = [("bf16 reference", lg), ("pulsar (bf16 head)", trunk)]
    dl = os.path.join(a.decode or a.cap, "decode_logits.f32")
    if os.path.exists(dl) and os.path.getsize(dl) >= T * lg.shape[1] * 4:   # the decode path: one row a step
        rows.append(("pulsar decode (EXL3 head)", read_f32(dl, lg.shape[1])[first:first + n]))
    if a.exl3:
        ex_logits = read_f32(a.exl3 + "_logits.f32", lg.shape[1])[-n:]
        rows.append(("exllamav3", ex_logits))
    for name, l in rows:
        m, md, mx, t = kl_stats(lg, l) if name != "bf16 reference" else (0.0, 0.0, 0.0, 1.0)
        nll = F.cross_entropy(l[:-1].double(), tgt, reduction="mean").item()
        print(f" {name:28s}  {m:.5f}   {md:.5f}    {mx:7.3f}   {t:.3f}     {nll:.4f}")
    if a.exl3:
        m, md, mx, t = kl_stats(ex_logits, trunk)
        print(f" {'pulsar vs exllamav3':28s}  {m:.5f}   {md:.5f}    {mx:7.3f}   {t:.3f}")
    last = read_f32(os.path.join(a.cap, "prefill_last.f32"), lg.shape[1])
    m, _, _, t = kl_stats(lg[-1:], last)
    print(f"the prefill's last row through pulsar's EXL3 head: KL(ref || pulsar) {m:.6f}, top-1 {'=' if t else '!='}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("prep"); q.add_argument("ref"); q.add_argument("out")
    q = s.add_parser("exl3"); q.add_argument("pack"); q.add_argument("ids"); q.add_argument("n_last", type=int)
    q.add_argument("out")
    q = s.add_parser("grade"); q.add_argument("ref"); q.add_argument("prompt"); q.add_argument("cap")
    q.add_argument("base"); q.add_argument("--exl3"); q.add_argument("--decode")
    a = p.parse_args()
    {"prep": prep, "exl3": exl3, "grade": grade}[a.cmd](a)


if __name__ == "__main__":
    main()

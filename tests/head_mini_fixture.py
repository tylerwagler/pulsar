#!/usr/bin/env python3
"""Head test vectors for the family head options (L295 E5) from the Inkling-Small mini
checkpoint and its transformers fp32 capture (ref_fp32.pt: input_ids, hidden_states[0]
= the normed embedding, the last hidden state, logits [1300][200058]).

    emb.ids.bin     int32 [T]           the prompt's ids, remapped into emb.table.bin
    emb.table.bin   bf16  [U][n_embd]   the embedding rows those ids use (as stored)
    emb.norm_w.bin  bf16  [n_embd]      model.llm.embed_norm.weight
    emb.ref32.bin   f32   [T][n_embd]   transformers' InklingNormedEmbedding (hidden_states[0])
    head.raw.bin    f32   [R][n_vocab]  the head GEMM with NO scale: final hidden (fp32) x unembed^T, all
                                        201024 rows, for R rows spread over the prompt
    head.ref.bin    f32   [R][n_real]   transformers' logits for those rows (hidden / 16 before lm_head,
                                        sliced to the unpadded vocab)
    meta.txt        tokens, rows, n_embd, n_vocab, n_real, mup, eps

The fixture checks which of the capture's last hidden state (post-norm, or pre-norm then
the final norm) reproduces the reference logits, and says so.  tests/head_gate --fixture
DIR grades the embedding norm and the logits finish (scale, padding) against it.  CPU only.

Usage: head_mini_fixture.py MINI_DIR REF_PT OUT_DIR
"""
import json
import os
import sys

import torch
from safetensors import safe_open

ROWS = 16


def rmsnorm(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))


def main():
    mini, ref_pt, out = sys.argv[1:4]
    cfg = json.load(open(os.path.join(mini, "config.json")))["text_config"]
    eps, mup, n_real = cfg["rms_norm_eps"], cfg["logits_mup_width_multiplier"], cfg["unpadded_vocab_size"]
    assert cfg["use_embed_norm"]
    ref = torch.load(ref_pt, map_location="cpu", weights_only=False)
    torch.set_num_threads(16)
    g = safe_open(os.path.join(mini, "globals.safetensors"), "pt")
    ids = ref["input_ids"][0]
    T = ids.shape[0]
    uniq, inv = torch.unique(ids, return_inverse=True)
    emb = g.get_slice("model.llm.embed.weight")
    table = torch.stack([emb[int(i): int(i) + 1][0] for i in uniq])
    norm_w = g.get_tensor("model.llm.embed_norm.weight")
    os.makedirs(out, exist_ok=True)
    inv.to(torch.int32).numpy().tofile(os.path.join(out, "emb.ids.bin"))
    table.contiguous().view(torch.int16).numpy().tofile(os.path.join(out, "emb.table.bin"))
    norm_w.contiguous().view(torch.int16).numpy().tofile(os.path.join(out, "emb.norm_w.bin"))
    e_ref = ref["hidden_states"][0][0].float().contiguous()
    e_ref.numpy().tofile(os.path.join(out, "emb.ref32.bin"))
    mine = rmsnorm(table[inv].float(), norm_w.float(), eps)
    print(f"embed: {T} ids ({uniq.numel()} distinct); fp32 rebuild vs hidden_states[0]: rel fro "
          f"{((mine - e_ref).norm() / e_ref.norm()).item():.3e}")

    U = g.get_tensor("model.llm.unembed.weight").float()
    n_vocab = U.shape[0]
    rows = torch.linspace(0, T - 1, ROWS).round().long()
    last = ref["hidden_states"][-1][0].float()[rows]
    want = ref["logits"][0][rows].float()
    cand = {"post-norm": last, "pre-norm + final norm": rmsnorm(last, g.get_tensor("model.llm.norm.weight").float(), eps)}
    best = None
    for name, h in cand.items():
        lg = (h / mup) @ U.t()
        e = ((lg[:, :n_real] - want).norm() / want.norm()).item()
        print(f"head: the capture's last hidden state taken as {name}: logits rel fro {e:.3e}")
        if best is None or e < best[1]:
            best = (name, e, h)
    if best[1] > 1e-5:
        sys.exit("neither reading of the last hidden state reproduces the reference logits")
    raw = best[2] @ U.t()   # the head GEMM without the scale, every padded row
    raw.contiguous().numpy().tofile(os.path.join(out, "head.raw.bin"))
    want.contiguous().numpy().tofile(os.path.join(out, "head.ref.bin"))
    with open(os.path.join(out, "meta.txt"), "w") as m:
        m.write(f"tokens {T}\nrows {ROWS}\nn_embd {U.shape[1]}\nn_vocab {n_vocab}\nn_real {n_real}\nmup {mup}\neps {eps}\n")
    print(f"head: {ROWS} rows x {n_vocab} logits ({n_real} real), the last hidden state is {best[0]}; "
          f"pad logits before the mask: max {raw[:, n_real:].max().item():.3f}")


if __name__ == "__main__":
    main()

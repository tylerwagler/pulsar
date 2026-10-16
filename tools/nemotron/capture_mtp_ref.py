#!/usr/bin/env python3
"""capture_mtp_ref.py -- the MTP module's reference forward on the Nemotron mini (L296 N9): the drafter's first step
at every position of capture_ref.py's prompt, in transformers' own modules.

transformers has no MTP forward for Nemotron-H (NemotronHForCausalLM drops `mtp.*` on load), so this builds vLLM's
NemotronHMultiTokenPredictor (models/nemotron_h_mtp.py) from transformers' blocks -- its NemotronHBlock for the
{attention, MoE} pattern (deep copies of the loaded model's own attention and MoE blocks, the checkpoint's
`mtp.layers.*` weights loaded over them), NemotronHRMSNorm for enorm / hnorm / final_layernorm, a bias-free eh_proj:

    x = eh_proj([enorm(embed(t_{p+1})) | hnorm(c_p)])      embed FIRST (nemotron_h_mtp.py:106-113)
    x = x + attn(norm0(x))     causal over the prompt (NoPE)
    x = x + moe(norm1(x))
    logits = lm_head(final_layernorm(x))                   the target's head (the checkpoint has none of its own)

c_p is the target hidden vLLM hands the proposer: norm_f of the trunk's residual at p (capture_ref.py's `final_norm`,
the dt-unclamped `final_norm_dt0` -- vLLM's Mamba semantics, what the engine is graded against).  Row p predicts token
p + 2.  The mapping of the checkpoint's per-expert tensors onto transformers' stacked experts is checked on the
trunk's own MoE block first (its loaded parameters must equal the same mapping of its checkpoint tensors).

    python capture_mtp_ref.py <mini_dir> <ref.pt> <out.pt> [--dtype fp32|bf16]

`--dtype bf16` runs the module in transformers' bf16 on the bf16 capture's own trunk (pass capture_ref.py's
--dtype bf16 output as ref.pt): the comparator that puts a bf16 engine's distance to the fp32 reference in context.

out.pt: input_ids [1, T]; logits [1, T - 1, V] f32 (row p: the module at p); hidden [1, T - 1, H] (final_layernorm's
output, stored as computed); meta.
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mini")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoModel

    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    model = AutoModel.from_pretrained(a.mini, trust_remote_code=True, dtype=dtype, attn_implementation="eager")
    model.eval()
    lm = model.language_model
    cfg = lm.config
    layers = lm.model.layers
    kinds = [l.block_type for l in layers]
    att = next(i for i, k in enumerate(kinds) if "attention" in k and "linear" not in k)
    moe = next(i for i, k in enumerate(kinds) if k == "moe")
    print(f"loaded in {time.time() - t0:.0f} s; blocks {kinds}; attention block {att}, MoE block {moe}", flush=True)

    idx = json.load(open(os.path.join(a.mini, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.mini, f), framework="pt")
        return handles[f].get_tensor(name)

    def experts_of(prefix, n):
        up = torch.stack([get(f"{prefix}.experts.{e}.up_proj.weight") for e in range(n)])
        down = torch.stack([get(f"{prefix}.experts.{e}.down_proj.weight") for e in range(n)])
        return up, down

    n_exp = cfg.n_routed_experts
    # the mapping, checked on the trunk's own MoE block (loaded by transformers' conversion)
    trunk_names = [k for k in idx if k.endswith(f".layers.{moe}.mixer.gate.weight") and ".mtp." not in k]
    if len(trunk_names) != 1:
        raise SystemExit(f"the trunk MoE block {moe}'s gate is not one tensor: {trunk_names}")
    tpre = trunk_names[0][: -len(".gate.weight")]
    up, down = experts_of(tpre, n_exp)
    mx = layers[moe].mixer
    for what, mine, theirs in [("experts.up_proj", up, mx.experts.up_proj), ("experts.down_proj", down, mx.experts.down_proj),
                               ("gate.weight", get(f"{tpre}.gate.weight"), mx.gate.weight),
                               ("gate.e_score_correction_bias", get(f"{tpre}.gate.e_score_correction_bias"),
                                mx.gate.e_score_correction_bias)]:
        if mine.shape != theirs.shape or not torch.equal(mine.to(theirs.dtype), theirs.detach()):
            raise SystemExit(f"the expert mapping is not transformers' on the trunk block {moe}: {what} "
                             f"{tuple(mine.shape)} vs {tuple(theirs.shape)}")
    print(f"mapping checked on the trunk MoE block ({tpre})", flush=True)

    P = "language_model.mtp.layers"
    H = cfg.hidden_size
    eps = cfg.layer_norm_epsilon
    Norm = type(lm.model.norm_f)

    def norm(w):
        n = Norm(H, eps=eps)
        n.weight.data = w.to(dtype)
        return n

    enorm, hnorm = norm(get(f"{P}.0.enorm.weight")), norm(get(f"{P}.0.hnorm.weight"))
    final = norm(get(f"{P}.1.final_layernorm.weight"))
    eh = torch.nn.Linear(2 * H, H, bias=False, dtype=dtype)
    eh.weight.data = get(f"{P}.0.eh_proj.weight").to(dtype)
    b0, b1 = copy.deepcopy(layers[att]), copy.deepcopy(layers[moe])
    b0.norm.weight.data = get(f"{P}.0.norm.weight").to(dtype)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(b0.mixer, p).weight.data = get(f"{P}.0.mixer.{p}.weight").to(dtype)
    b1.norm.weight.data = get(f"{P}.1.norm.weight").to(dtype)
    m1 = b1.mixer
    m1.gate.weight.data = get(f"{P}.1.mixer.gate.weight").to(m1.gate.weight.dtype)
    m1.gate.e_score_correction_bias.data = get(f"{P}.1.mixer.gate.e_score_correction_bias").to(
        m1.gate.e_score_correction_bias.dtype)
    m1.fc1_latent_proj.weight.data = get(f"{P}.1.mixer.fc1_latent_proj.weight").to(dtype)
    m1.fc2_latent_proj.weight.data = get(f"{P}.1.mixer.fc2_latent_proj.weight").to(dtype)
    m1.shared_experts.up_proj.weight.data = get(f"{P}.1.mixer.shared_experts.up_proj.weight").to(dtype)
    m1.shared_experts.down_proj.weight.data = get(f"{P}.1.mixer.shared_experts.down_proj.weight").to(dtype)
    up, down = experts_of(f"{P}.1.mixer", n_exp)
    m1.experts.up_proj.data = up.to(dtype)
    m1.experts.down_proj.data = down.to(dtype)

    ref = torch.load(a.ref, weights_only=False)
    ids = ref["input_ids"]
    T = ids.shape[1]
    cond = ref["final_norm_dt0"].to(dtype)[:, : T - 1]
    with torch.inference_mode():
        emb = lm.model.embeddings(ids[:, 1:T]).to(dtype)
        x = eh(torch.cat([enorm(emb), hnorm(cond)], dim=-1))
        n = T - 1
        mask = torch.full((n, n), float("-inf"), dtype=dtype).triu(1)[None, None]
        x = b0(x, attention_mask=mask)
        x = b1(x)
        o = final(x)
        logits = torch.nn.functional.linear(o, lm.lm_head.weight.to(dtype)).float()   # (bf16 weights in either load)
    torch.save({"input_ids": ids, "logits": logits, "hidden": o,
                "meta": {"transformers": transformers.__version__, "torch": torch.__version__, "dtype": a.dtype,
                         "ref": a.ref, "attention_block": att, "moe_block": moe}}, a.out)
    print(f"wrote {a.out}: logits {tuple(logits.shape)} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())

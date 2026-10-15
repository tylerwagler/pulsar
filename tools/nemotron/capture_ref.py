#!/usr/bin/env python3
"""capture_ref.py -- the fp32 reference forward of the Nemotron mini (L296 N1): transformers (trust_remote_code: the
checkpoint's own NemotronH_Omni_Reasoning_V3 wrapper around transformers' NemotronHForCausalLM), fp32, eager
attention, CPU, on the first T tokens of a text.  The format-neutral correctness fixture the pulsar forward (N2-N7)
grades against, as /mnt/models/inkling-small-mini-ref is Inkling's.

    python capture_ref.py <mini_dir> <out.pt> <text_file> [--tokens 1300]

out.pt (torch.save) holds:
  input_ids      [1, T] int64 (HF tokenizers on the text, no special tokens added -- the template renders no BOS)
  embed          [1, T, H] the embedding gather
  hidden_states  list of [1, T, H]: the residual stream after HF block b (b = 0 .. n_blocks-1), PRE norm_f
  mixer_out      list of [1, T, H]: block b's mixer output (the residual delta: hidden_states[b] - previous)
  final_norm     [1, T, H] norm_f(hidden_states[-1])
  logits         [1, T, V] fp32
  blocks         the mini's layers_block_type
  dt_clamp       the (min, max) transformers clamped dt to in this run
  and the same four tensors under `*_dt0` for the forward with time_step_limit (0, inf): the checkpoint's own
  time_step_limit, which vLLM uses (it ignores the field); transformers 5.19 clamps dt at time_step_min (0.001)
  instead (NemotronHMamba2Mixer.time_step_limit).  Which one pulsar grades against is R1's question; both are here.
  meta           {transformers, torch, text, tokens, dt_min_seen}
"""
import argparse
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mini")
    ap.add_argument("out")
    ap.add_argument("text")
    ap.add_argument("--tokens", type=int, default=1300)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.mini)
    ids = tok(open(a.text, encoding="utf-8").read(), add_special_tokens=False)["input_ids"][:a.tokens]
    if len(ids) != a.tokens:
        raise SystemExit(f"{a.text}: {len(ids)} tokens < {a.tokens}")
    input_ids = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    model, info = AutoModel.from_pretrained(a.mini, trust_remote_code=True, dtype=torch.float32,
                                            attn_implementation="eager", output_loading_info=True)
    missing, unexpected = sorted(info["missing_keys"]), sorted(info["unexpected_keys"])
    print(f"loaded in {time.time() - t0:.0f} s; missing {missing[:8]} ({len(missing)}); unexpected {len(unexpected)} "
          f"(MTP / input conditioner by design: {sorted({'.'.join(k.split('.')[:2]) for k in unexpected})})", flush=True)
    # transformers 5.19 renames mlp1.* to multi_modal_projector.* before the checkpoint's own conversion mapping
    # (mlp1 -> vision_projector.mlp1) can apply, so the image projector loads nothing: vision only, not graded here
    lm_missing = [k for k in missing if not k.startswith("vision_projector.")]
    if lm_missing:
        raise SystemExit(f"missing keys outside the image projector: {lm_missing[:8]}")
    model.eval()
    lm = model.language_model
    blocks = list(lm.config.layers_block_type)
    cap = {}

    def put(key):
        def hook(_m, _inp, out):
            cap[key] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook
    hooks = [lm.model.embeddings.register_forward_hook(put("embed")), lm.model.norm_f.register_forward_hook(put("final_norm"))]
    for b, layer in enumerate(lm.model.layers):
        hooks.append(layer.register_forward_hook(put(f"h{b}")))
        hooks.append(layer.mixer.register_forward_hook(put(f"m{b}")))
    dt_seen = []
    mamba = [layer.mixer for layer in lm.model.layers if hasattr(layer.mixer, "dt_bias")]

    def dt_probe(mixer):
        def hook(_m, inp, out):
            proj = out
            dt = proj[..., -mixer.num_heads:]
            dt_seen.append(torch.nn.functional.softplus(dt + mixer.dt_bias).min().item())
        return hook
    for mx in mamba:
        hooks.append(mx.in_proj.register_forward_hook(dt_probe(mx)))

    def run():
        cap.clear()
        with torch.inference_mode():
            t = time.time()
            logits = lm(input_ids=input_ids).logits.float()
            print(f"forward {time.time() - t:.0f} s", flush=True)
        return {"embed": cap["embed"], "hidden_states": [cap[f"h{b}"] for b in range(len(blocks))],
                "mixer_out": [cap[f"m{b}"] for b in range(len(blocks))], "final_norm": cap["final_norm"],
                "logits": logits}
    clamp = mamba[0].time_step_limit
    ref = run()
    for mx in mamba:
        mx.time_step_limit = (0.0, float("inf"))
    ref0 = run()
    out = {"input_ids": input_ids, **ref, "blocks": blocks, "dt_clamp": tuple(clamp),
           **{f"{k}_dt0": v for k, v in ref0.items()},
           "meta": {"transformers": transformers.__version__, "torch": torch.__version__, "text": a.text,
                    "tokens": a.tokens, "dt_min_seen": min(dt_seen)}}
    same = torch.equal(ref["logits"], ref0["logits"])
    d = (ref["logits"] - ref0["logits"]).abs().max().item()
    print(f"blocks {blocks}; dt clamp {clamp}; min softplus(dt + dt_bias) seen {min(dt_seen):.3e}; "
          f"logits clamp vs (0, inf): {'IDENTICAL' if same else f'max |diff| {d:.3e}'}", flush=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

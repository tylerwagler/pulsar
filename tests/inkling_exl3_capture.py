#!/usr/bin/env python3
"""The exllamav3 forward of an Inkling EXL3 pack, captured for the engine's layer grade (L294 / L295).

  inkling_exl3_capture.py EXL3_DIR MINI_DIR REF_PT OUT_PT [--device cuda:0]

Runs the quant lane's exllamav3 Inkling module (its fork, ~/inkling/exllamav3 on Sparky; the same forward its
parity check graded against the transformers fp32 capture) over the reference capture's prompt, module by module, and
keeps what the engine's grade compares with (tests/inkling_mini_grade.py --exl3):
  hs[i]   each decoder layer's output (the residual stream after it), f32 [T][n_embd]
  final   the last layer's output after the final norm, f32 [T][n_embd]
  logits  final @ U^T / logits_mup_width_multiplier with U the bf16 checkpoint's unembed (MINI_DIR) -- the head the
          pulsar container carries -- not the pack's own quantized head, so the logits differ from the engine's only
          by the forward
"""
import argparse
import json
import os

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exl3_dir")
    ap.add_argument("mini_dir")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from exllamav3 import Config, Model

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    ids = ref["input_ids"]
    cfg = Config.from_directory(args.exl3_dir)
    model = Model.from_config(cfg)
    dev = torch.device(args.device)
    params = {}
    x = model.prepare_inputs(ids, params)
    hs, final = [], None
    for idx, m in enumerate(model.modules):
        m.load(dev)
        x = m.prepare_for_device(x, params)
        x = m.forward(x, params)
        m.unload()
        torch.cuda.empty_cache()
        if model.first_block_idx <= idx < model.last_kv_module_idx:
            hs.append(x[0].float().cpu())
        elif idx == model.last_kv_module_idx + 1:   # the final norm
            final = x[0].float().cpu()
            break
    text = json.load(open(os.path.join(args.mini_dir, "config.json")))["text_config"]
    with safe_open(os.path.join(args.mini_dir, "globals.safetensors"), "pt") as f:
        U = f.get_tensor("model.llm.unembed.weight").float()
    logits = (final.to(torch.bfloat16).float() @ U.t()) / float(text["logits_mup_width_multiplier"])
    torch.save({"hs": hs, "final": final, "logits": logits}, args.out)
    print(f"exl3 capture: {ids.shape[1]} tokens, {len(hs)} layers, logits {tuple(logits.shape)} -> {args.out}")


if __name__ == "__main__":
    main()

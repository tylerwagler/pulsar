#!/usr/bin/env python3
"""Generate Qwen3.8-Flash-Next (qwen4_exp) vision TOWER goldens from HF's own modules (L268).

Loads the checkpoint's `model.visual.*` tensors into transformers' `Qwen4ExpVisionModel`, preprocesses each case's
encoded image with `Qwen2VLImageProcessorPil` (vision-qwen-pixel-gate pins the engine's preprocessing to it bit for
bit, so the gate feeds the same bytes through the engine's), and runs the tower twice -- in bf16 (the reference) and
in fp32 (for the floor), on the GPU: this torch's CPU bf16 Conv3d (2.11, aarch64) is wrong by ~65% relative RMS
against its own fp32 path and an explicit GEMM, so a CPU bf16 run is not a reference.  Recorded per stage: the bf16 reference rows and the reference's OWN bf16-vs-fp32 relative
RMS -- the self-calibrating floor vision-tower-gate already grades DeepSeek's tower against.

Stages: the blocks' input (patch embed + interpolated learned positions), block 0's output, the last block's output,
the merger's output (the rows the language model receives).  The first case dumps all four; the rest only the merger.

    ~/venvs/q38/bin/python tests/vision_qwen_tower_goldens.py <hf-snapshot> > tests/test-vectors/vision-qwen-tower-goldens.bin
"""
import io
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

SNAP = sys.argv[1]
torch.manual_seed(0)
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg = Qwen4ExpConfig.from_pretrained(SNAP)
vcfg = cfg.vision_config
vcfg._attn_implementation = "sdpa"
pcfg = json.load(open(os.path.join(SNAP, "preprocessor_config.json")))
proc = Qwen2VLImageProcessorPil(
    size=pcfg["size"], patch_size=pcfg["patch_size"], temporal_patch_size=pcfg["temporal_patch_size"],
    merge_size=pcfg["merge_size"], image_mean=pcfg["image_mean"], image_std=pcfg["image_std"])

index = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
state = {}
for name, shard in index.items():
    if name.startswith("model.visual."):
        with safe_open(os.path.join(SNAP, shard), "pt") as f:
            state[name[len("model.visual."):]] = f.get_tensor(name)


def build(dtype):
    m = Qwen4ExpVisionModel._from_config(vcfg)
    missing, unexpected = m.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True), None
    return m.to(dtype).to(DEV).eval()


tower = {torch.bfloat16: build(torch.bfloat16), torch.float32: build(torch.float32)}


def run(dtype, pv, grid):
    m = tower[dtype]
    caught = {}
    hooks = [m.blocks[0].register_forward_pre_hook(lambda mod, args, kw: caught.__setitem__("input", args[0] if args else kw["hidden_states"]), with_kwargs=True),
             m.blocks[0].register_forward_hook(lambda mod, a, out: caught.__setitem__("block0", out)),
             m.blocks[-1].register_forward_hook(lambda mod, a, out: caught.__setitem__("blockN", out))]
    with torch.no_grad():
        out = m(torch.from_numpy(pv).to(DEV), grid_thw=grid.to(DEV))
    for h in hooks:
        h.remove()
    caught["merger"] = out.pooler_output
    return {k: (v[0] if isinstance(v, tuple) else v).float().cpu() for k, v in caught.items()}


def make(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx * 3 + yy * 5) % 256)], -1)
    rgb = np.clip(base + rng.integers(-30, 31, size=(h, w, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def bf16_bits(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16).tobytes()


def rel_rms(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))


CASES = [((300, 220, 11), ("input", "block0", "blockN", "merger")),
         ((260, 300, 12), ("merger",))]
out = sys.stdout.buffer
out.write(b"QVT1")
out.write(struct.pack("<I", len(CASES)))
for (w, h, seed), stages in CASES:
    enc = make(w, h, seed)
    feat = proc(images=[Image.open(io.BytesIO(enc))], return_tensors="np")
    pv = feat["pixel_values"].astype(np.float32)
    grid = torch.from_numpy(feat["image_grid_thw"].astype(np.int64))
    ref = run(torch.bfloat16, pv, grid)
    f32 = run(torch.float32, pv, grid)
    out.write(struct.pack("<I", len(enc)))
    out.write(enc)
    out.write(struct.pack("<3I", int(grid[0, 1]), int(grid[0, 2]), len(stages)))
    for s in stages:
        r = ref[s]
        floor = rel_rms(r, f32[s])
        out.write(s.encode().ljust(8, b"\0"))
        out.write(struct.pack("<2If", r.shape[0], r.shape[1], floor))
        out.write(bf16_bits(r))
        print(f"case {w}x{h}: {s} {tuple(r.shape)} bf16-vs-fp32 floor {floor:.3e}", file=sys.stderr)

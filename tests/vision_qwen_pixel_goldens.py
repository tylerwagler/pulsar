#!/usr/bin/env python3
"""Generate Qwen3.8-Flash-Next (qwen4_exp) vision PIXEL goldens from HF's own image processor (L268).

The authority is transformers' `Qwen2VLImageProcessorPil` (the PIL backend HF ships beside the torchvision one; the
checkpoint's preprocessor_config.json names the processor family and its settings).  Its resize is Pillow's
BICUBIC -- the same operation DeepSeek's path already reproduces bit for bit (vision.cpp pil_resize_rgb), so the two
families share it.  (vLLM runs the torchvision backend, whose antialiased bicubic can differ from Pillow's by one
uint8 level at some pixels; this golden pins the PIL path.)

Each case is an encoded image (PNG, or JPEG for the codec), chosen to cover smart_resize's branches: the plain
rounding, the min_pixels upscale, a wide aspect.  Recorded per case: the encoded bytes, the source size, the
resized size, the patch grid, an FNV-1a of every pixel_values float32 (row-major), and a spread of sample rows in
full -- so the C++ gate grades the whole array exactly without the array living in the repo.

    ~/venvs/q38/bin/python tests/vision_qwen_pixel_goldens.py <hf-snapshot> > tests/test-vectors/vision-qwen-pixel-goldens.bin
"""
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

SNAP = sys.argv[1]
cfg = json.load(open(os.path.join(SNAP, "preprocessor_config.json")))
proc = Qwen2VLImageProcessorPil(
    size=cfg["size"], patch_size=cfg["patch_size"], temporal_patch_size=cfg["temporal_patch_size"],
    merge_size=cfg["merge_size"], image_mean=cfg["image_mean"], image_std=cfg["image_std"],
    do_convert_rgb=cfg.get("do_convert_rgb", True))
SAMPLE_ROWS = 12


def fnv1a(b):
    h = 0xCBF29CE484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def make(w, h, seed, fmt):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx + yy) * 7) % 256], -1)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    else:
        Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


CASES = [
    (300, 220, 1, "png"),   # rounds to 288x224 (above min_pixels)
    (96, 64, 2, "png"),     # min_pixels upscale
    (640, 200, 3, "png"),   # wide aspect, rounding
    (257, 255, 4, "jpeg"),  # JPEG codec + rounding at the 256 boundary
]

out = sys.stdout.buffer
out.write(b"QVP1")
out.write(struct.pack("<I", len(CASES)))
for w, h, seed, fmt in CASES:
    enc = make(w, h, seed, fmt)
    img = Image.open(io.BytesIO(enc))
    feat = proc(images=[img], return_tensors="np")
    pv = np.ascontiguousarray(feat["pixel_values"].astype(np.float32))
    t, gh, gw = (int(x) for x in feat["image_grid_thw"][0])
    rows, cols = pv.shape
    assert t == 1 and rows == gh * gw and cols == 3 * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
    picks = sorted(set(int(round(i * (rows - 1) / (SAMPLE_ROWS - 1))) for i in range(SAMPLE_ROWS)))
    out.write(struct.pack("<I", len(enc)))
    out.write(enc)
    out.write(struct.pack("<8I", w, h, gh * cfg["patch_size"], gw * cfg["patch_size"], gh, gw, rows, cols))
    out.write(struct.pack("<Q", fnv1a(pv.tobytes())))
    out.write(struct.pack("<I", len(picks)))
    for r in picks:
        out.write(struct.pack("<I", r))
        out.write(pv[r].tobytes())
    print(f"case {w}x{h} {fmt}: resized {gh * 16}x{gw * 16}, grid {gh}x{gw}, {rows} rows", file=sys.stderr)

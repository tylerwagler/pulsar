#!/usr/bin/env python3
"""Generate vision-MERGE goldens: the reference's merge_image_embeddings output.

This is the whole per-image path end to end -- preprocess -> ViT + aligner ->
scatter into the span (IMAGE slots take the aligner rows in `perm` order, every
other slot takes its type's learned vector) -- which is what the model actually
consumes.  It is the reference's own code, so the engine's `vision_merge_span()`
is graded against the authority rather than against a re-derivation.

Grids are deliberately SMALL (cropped from a real image's patch grid): the span
is the aligner grid plus padding, so this keeps the fixture tiny while still
exercising every sentinel type and the aligner merge.

    python3 tests/vision_merge_goldens.py <snapshot-dir> > tests/test-vectors/vision-merge-goldens.bin
"""
import io
import json
import struct
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

SNAP = sys.argv[1]
sys.path.insert(0, SNAP + "/inference")
import vision as V  # noqa: E402
import image_processor as ip  # noqa: E402


class Args:
    def __init__(self, c):
        for k in ("vision_n_layers", "vision_dim", "vision_n_heads", "vision_inter_dim",
                  "vision_patch_size", "vision_rope_theta", "vision_downsample_ratio",
                  "vision_max_n_token", "vision_min_pixels", "vision_max_wh_ratio",
                  "hidden_size", "vocab_size"):
            setattr(self, k, c[k])
        self.dim = c["hidden_size"]


class Tower(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.vision = V.ViT(a)
        self.aligner = V.Aligner(a)


def load(a):
    index = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]
    want = {k: index[k] for k in index if k.startswith(("vision.", "aligner.", "image_"))}
    per = {}
    for k, sh in want.items():
        per.setdefault(sh, []).append(k)
    ten = {}
    for sh, keys in per.items():
        with safe_open(SNAP + "/" + sh, framework="pt") as f:
            for k in keys:
                ten[k] = f.get_tensor(k)
    t = Tower(a)
    t.vision.load_state_dict({k[7:]: v for k, v in ten.items() if k.startswith("vision.")}, strict=True)
    t.aligner.load_state_dict({k[8:]: v for k, v in ten.items() if k.startswith("aligner.")}, strict=True)
    return t.to(torch.bfloat16).eval(), ten


def realistic_patches(a, nh, nw):
    y, x = np.mgrid[0:384, 0:512]
    r = (x * 255 // 511).astype(np.uint8)
    g = (y * 255 // 383).astype(np.uint8)
    b = (((x // 5) + (y // 7)) % 2 * 200 + 27).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([r, g, b]), "RGB").save(buf, format="PNG")
    patches, nvh, nvw, _, _ = ip.load_image({"data": buf.getvalue()}, a)
    p = a.vision_patch_size
    grid = patches.view(nvh, nvw, 3, p, p)
    return grid[:nh, :nw].reshape(nh * nw, 3, p, p).contiguous()


def bf16_bits(t):
    return t.contiguous().view(torch.uint16).cpu().numpy().astype("<u2").tobytes()


def merge(tower, ten, a, patches, nh, nw, start_pos, dtype):
    types, perm = ip.build_image_block((nh + a.vision_downsample_ratio - 1) // a.vision_downsample_ratio,
                                       (nw + a.vision_downsample_ratio - 1) // a.vision_downsample_ratio,
                                       start_pos)
    params = torch.stack([ten["image_start"], ten["image_pad"], ten["image_pad"],
                          ten["image_newline"], ten["image_end"]]).to(dtype)
    with torch.no_grad():
        embeds = tower.aligner(tower.vision(patches, nh, nw), nh, nw)[perm]
    block = params[types].clone()
    block[types == ip.IMAGE] = embeds.to(dtype)
    return types, perm, block


def main():
    cfg = json.load(open(SNAP + "/config.json"))
    a = Args(cfg)
    tower_bf, ten = load(a)
    tower_fp, _ = load(a)
    tower_fp = tower_fp.to(torch.float32).eval()

    cases = [(4, 6, 0), (4, 6, 3), (7, 5, 1)]
    out = sys.stdout.buffer
    out.write(b"VMG1")
    out.write(struct.pack("<I", len(cases)))
    for (nh, nw, start_pos) in cases:
        patches = realistic_patches(a, nh, nw)
        types, perm, block = merge(tower_bf, ten, a, patches, nh, nw, start_pos, torch.bfloat16)
        _, _, block_fp = merge(tower_fp, ten, a, patches.float(), nh, nw, start_pos, torch.float32)
        d = (block.float() - block_fp).flatten()
        floor = (d.pow(2).sum() / block_fp.flatten().pow(2).sum()).sqrt().item()
        out.write(struct.pack("<iiiiiif", nh, nw, a.vision_patch_size, a.hidden_size,
                              start_pos, len(types), float(floor)))
        out.write(struct.pack("<i", len(perm)))
        out.write(bf16_bits(patches))
        out.write(struct.pack("<%di" % len(types), *[int(t) for t in types.tolist()]))
        out.write(struct.pack("<%di" % len(perm), *[int(p) for p in perm.tolist()]))
        out.write(bf16_bits(block))
        sys.stderr.write("%dx%d start=%d: span %d, perm %d, floor %.3e\n"
                         % (nh, nw, start_pos, len(types), len(perm), floor))


if __name__ == "__main__":
    main()

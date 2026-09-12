#!/usr/bin/env python3
"""Generate whole-image goldens: encoded bytes IN, the reference's merged block OUT.

This is the fixture the ENGINE's per-image entry is graded against.  The other
two image fixtures each carry half of it: vision-span-goldens has the encoded
image bytes and the span layout but not the embeddings, and vision-merge-goldens
has the patches and the embeddings but not the bytes.  Here the bytes are the
input, so a gate can hand them to gpu_graph_merge_image_spans() and make the
engine do its OWN decode -> preprocess -> tower -> merge before the comparison --
which is the point: the parts are graded separately elsewhere, this grades the
lane they form.

The expected block is the reference's own merge_image_embeddings: the ViT +
aligner over image_processor.load_image()'s patches, with IMAGE slots replaced by
the aligner rows in `perm` order and every other slot by its type's learned
vector.  The bf16 floor travels with it (the reference's bf16-vs-fp32 gap), the
same way vision-merge-goldens carries it.

    python3 tests/vision_image_goldens.py <snapshot-dir> > tests/test-vectors/vision-image-goldens.bin
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



def synth_rgb(w, h):
    """A deterministic RGB image: gradients plus a high-frequency checker, so the
    patches exercise the resampler and the tower instead of a flat field."""
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    b = (((x // 3) + (y // 5)) % 2 * 210 + 20).astype(np.uint8)
    return np.dstack([r, g, b])


def main():
    cfg = json.load(open(SNAP + "/config.json"))
    a = Args(cfg)
    vocab = cfg["vocab_size"]
    tower_bf, ten = load(a)
    tower_fp, _ = load(a)
    tower_fp = tower_fp.to(torch.float32).eval()

    # (width, height, format, start_pos): two aspect ratios and both codecs the
    # engine decodes, each with the span at a different place in the prompt.
    cases = [(512, 384, "PNG", 0), (384, 512, "JPEG", 5), (1200, 140, "PNG", 37)]
    out = sys.stdout.buffer
    out.write(b"VIG1")
    out.write(struct.pack("<I", len(cases)))
    for (w, h, fmt, start_pos) in cases:
        buf = io.BytesIO()
        Image.fromarray(synth_rgb(w, h), "RGB").save(buf, format=fmt)
        enc = buf.getvalue()
        patches, n_vh, n_vw, n_lh, n_lw = ip.load_image({"data": enc}, a)
        types, perm, block = merge(tower_bf, ten, a, patches, n_vh, n_vw, start_pos, torch.bfloat16)
        _, _, block_fp = merge(tower_fp, ten, a, patches.float(), n_vh, n_vw, start_pos, torch.float32)
        d = (block.float() - block_fp).flatten()
        floor = (d.pow(2).sum() / block_fp.flatten().pow(2).sum()).sqrt().item()
        # coerce: struct.pack wants exact ints, and these arrive as numpy/torch
        # scalars or as floats from the config JSON
        out.write(struct.pack("<iiiifiiiiifi", int(a.vision_patch_size), int(a.vision_downsample_ratio),
                              int(a.vision_max_n_token), int(a.vision_min_pixels),
                              float(a.vision_max_wh_ratio), int(vocab), int(start_pos),
                              int(n_vh), int(n_vw), int(len(types)), float(floor), int(len(enc))))
        out.write(enc)
        out.write(struct.pack("<%di" % len(types), *[int(t) for t in types.tolist()]))
        out.write(bf16_bits(block))
        sys.stderr.write("%dx%d %s start=%d: vit %dx%d span %d enc %d floor %.3e\n"
                         % (w, h, fmt, start_pos, n_vh, n_vw, len(types), len(enc), floor))


if __name__ == "__main__":
    main()

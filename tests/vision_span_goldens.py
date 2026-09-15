#!/usr/bin/env python3
"""Generate image-SPAN goldens: the reference's per-image half of prepare_vl_inputs.

Runs the checkpoint's `image_processor.load_image` on an encoded image and then
`build_image_block(n_llm_h, n_llm_w, start_pos)`, which is exactly what
`prepare_vl_inputs` does per image -- the sentinel token ids (vocab_size + type),
the roles, and the aligner-row permutation.  `vision_prepare_image()` composes
decode + preprocess + that block, so this grades the composition end to end.

    python3 tests/vision_span_goldens.py <snapshot>/inference <snapshot>/config.json > tests/test-vectors/vision-span-goldens.bin
"""
import io
import json
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
import image_processor as ip  # noqa: E402


class Args:
    def __init__(self, c):
        self.vision_patch_size = c["vision_patch_size"]
        self.vision_downsample_ratio = c["vision_downsample_ratio"]
        self.vision_max_n_token = c["vision_max_n_token"]
        self.vision_min_pixels = c["vision_min_pixels"]
        self.vision_max_wh_ratio = c["vision_max_wh_ratio"]


def synth_rgb(w, h):
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    b = (((x // 3) + (y // 5)) % 2 * 255).astype(np.uint8)
    return np.dstack([r, g, b])


def main():
    cfg = json.load(open(sys.argv[2]))
    args = Args(cfg)
    vocab = cfg["vocab_size"]
    out = sys.stdout.buffer

    cases = []
    for (w, h) in ((512, 384), (384, 512), (1200, 140)):
        buf = io.BytesIO()
        Image.fromarray(synth_rgb(w, h), "RGB").save(buf, format="PNG")
        for start_pos in (0, 1, 3, 17):
            cases.append((w, h, buf.getvalue(), start_pos))

    out.write(b"VSP1")
    out.write(struct.pack("<I", len(cases)))
    for (w, h, enc, start_pos) in cases:
        patches, n_vh, n_vw, n_lh, n_lw = ip.load_image({"data": enc}, args)
        types, perm = ip.build_image_block(n_lh, n_lw, start_pos)
        ids = (vocab + types).tolist()
        types = types.tolist()
        perm = perm.tolist()
        out.write(struct.pack("<iiiiiifi", w, h, args.vision_patch_size,
                              args.vision_downsample_ratio, args.vision_max_n_token,
                              args.vision_min_pixels, float(args.vision_max_wh_ratio), vocab))
        out.write(struct.pack("<iiiiiiI", start_pos, n_vh, n_vw, n_lh, n_lw,
                              len(ids), len(enc)))
        out.write(enc)
        out.write(struct.pack("<%di" % len(ids), *ids))
        out.write(struct.pack("<%di" % len(types), *types))
        out.write(struct.pack("<%di" % len(perm), *perm))
        sys.stderr.write("%dx%d start=%d: vit %dx%d llm %dx%d span %d\n"
                         % (w, h, start_pos, n_vh, n_vw, n_lh, n_lw, len(ids)))


if __name__ == "__main__":
    main()

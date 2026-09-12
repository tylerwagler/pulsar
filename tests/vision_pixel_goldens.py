#!/usr/bin/env python3
"""Generate vision PIXEL goldens from the checkpoint's OWN image_processor.py.

Calls the reference's `load_image` (so the goldens ARE the authority's output)
and, alongside each case's ViT patches, records the DECODED RGB bytes it ran on.
The decoder is therefore out of scope here on purpose: the C++ gate is fed the
same RGB and grades the resample/pad/normalise/patchify, which is the part that
has to match Pillow bit for bit.  Images are PNG (lossless), so any PNG decoder
reproduces these bytes exactly.

    python3 tests/vision_pixel_goldens.py <snapshot>/inference > tests/test-vectors/vision-pixel-goldens.bin
"""

import io
import struct
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, sys.argv[1])
import image_processor as ip  # noqa: E402


class Args:
    vision_patch_size = 14
    vision_downsample_ratio = 3
    vision_max_n_token = 384
    vision_min_pixels = 147456
    vision_max_wh_ratio = 8


def synth_rgb(w, h):
    """Deterministic content with gradients, edges and noise, so a resampler
    error shows up instead of being averaged away by a flat image."""
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    b = (((x // 3) + (y // 5)) % 2 * 255).astype(np.uint8)
    s = ((x * 7 + y * 13) % 97).astype(np.uint8)
    r = np.clip(r.astype(np.int32) + s.astype(np.int32) - 48, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int32) - s.astype(np.int32) + 48, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int32) + (s // 3) - 16, 0, 255).astype(np.uint8)
    return np.dstack([r, g, b])


def main():
    args = Args()
    out = sys.stdout.buffer
    # Coverage: the pad branch in both orientations, plus the max_wh_ratio clamp
    # AND the stretch branch (which needs w >= 8h and no other case reaches).
    cases = [(512, 384), (150, 1200), (1200, 140)]
    out.write(b"VPX1")
    out.write(struct.pack("<I", len(cases)))

    for (w, h) in cases:
        rgb = synth_rgb(w, h)
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="PNG")
        record = {"data": buf.getvalue()}   # bytes are taken as-is by load_image_bytes

        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image(record, args)
        # The decoded RGB the reference resampled -- recomputed the same way it
        # does (PNG, lossless), so the gate can isolate the pixel transform.
        decoded = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
        assert decoded.shape == (h, w, 3), decoded.shape
        assert (decoded == rgb).all(), "PNG round-trip changed the pixels"

        pbits = patches.view(torch.uint16).cpu().numpy().astype("<u2").tobytes()
        p = args.vision_patch_size
        assert len(pbits) == n_vit_h * n_vit_w * 3 * p * p * 2, len(pbits)

        out.write(struct.pack("<iiiiiifiiii", w, h, p, args.vision_downsample_ratio,
                              args.vision_max_n_token, args.vision_min_pixels,
                              float(args.vision_max_wh_ratio),
                              n_vit_h, n_vit_w, n_llm_h, n_llm_w))
        out.write(decoded.tobytes())
        out.write(pbits)
        sys.stderr.write("case %dx%d -> vit %dx%d, llm %dx%d, %d patch bytes\n"
                         % (w, h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, len(pbits)))


if __name__ == "__main__":
    main()

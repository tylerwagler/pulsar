#!/usr/bin/env python3
"""Generate vision CODEC goldens: Pillow's decode of the same bytes we decode.

The reference decodes with Pillow (libpng for PNG, libjpeg-turbo for JPEG), so
the engine links the same two libraries.  This records, per case, the encoded
file AND the RGB Pillow's `.convert("RGB")` produced from it -- which is what
`vision_decode_rgb` must reproduce byte for byte.

Covered on purpose: PNG RGB / RGBA (alpha dropped) / grayscale / palette, and
JPEG colour at two qualities plus grayscale -- the conversions where a decoder
difference would show up first.

    python3 tests/vision_codec_goldens.py > tests/test-vectors/vision-codec-goldens.bin
"""
import io
import struct
import sys

import numpy as np
from PIL import Image

W, H = 64, 48


def synth_rgb(w, h):
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    b = (((x // 3) + (y // 5)) % 2 * 255).astype(np.uint8)
    s = ((x * 7 + y * 13) % 97).astype(np.uint8)
    r = np.clip(r.astype(np.int32) + s.astype(np.int32) - 48, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int32) - s.astype(np.int32) + 48, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int32) + (s // 3) - 16, 0, 255).astype(np.uint8)
    return np.dstack([r, g, b])


def cases():
    rgb = synth_rgb(W, H)
    yield 0, Image.fromarray(rgb, "RGB"), dict(format="PNG")
    rgba = np.dstack([rgb, np.full((H, W), 200, np.uint8)])
    yield 0, Image.fromarray(rgba, "RGBA"), dict(format="PNG")
    gray = np.asarray(Image.fromarray(rgb, "RGB").convert("L"), dtype=np.uint8)
    yield 0, Image.fromarray(gray, "L"), dict(format="PNG")
    pal = Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=64)
    yield 0, pal, dict(format="PNG")
    bigger = synth_rgb(96, 72)
    for q in (90, 75):
        yield 1, Image.fromarray(bigger, "RGB"), dict(format="JPEG", quality=q)
    bg = np.asarray(Image.fromarray(bigger, "RGB").convert("L"), dtype=np.uint8)
    yield 1, Image.fromarray(bg, "L"), dict(format="JPEG", quality=80)
    yield 1, Image.fromarray(synth_rgb(129, 97), "RGB"), dict(format="JPEG", quality=50)


def main():
    out = sys.stdout.buffer
    items = list(cases())
    out.write(b"VCX1")
    out.write(struct.pack("<I", len(items)))
    for fmt, img, save_kw in items:
        buf = io.BytesIO()
        img.save(buf, **save_kw)
        enc = buf.getvalue()
        decoded = Image.open(io.BytesIO(enc)).convert("RGB")
        w, h = decoded.size
        arr = np.asarray(decoded, dtype=np.uint8)
        out.write(struct.pack("<IiiI", fmt, w, h, len(enc)))
        out.write(enc)
        out.write(arr.tobytes())
        sys.stderr.write("%s %dx%d %s -> %d enc bytes, %d rgb bytes\n"
                         % (save_kw["format"], w, h, save_kw, len(enc), arr.size))


if __name__ == "__main__":
    main()

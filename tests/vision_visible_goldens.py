#!/usr/bin/env python3
"""Generate image-visibility goldens from the checkpoint's OWN code.

An image span is the one place the model is bidirectional: a patch must see its
whole image rather than just the causal sliding window.  The reference expresses
that as two pure integer functions in `inference/model.py`:

    get_image_visible(input_ids, vocab_size, max_image_tokens)   -> (left, right)
    get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens)

which this dumps, together with the plain `get_window_topk_idxs` window matrix
for the same sequence.  The gate grades the engine's port against the first two
EXACTLY (they are integer arithmetic -- no tolerance is involved or wanted) and
uses the third as the discriminating check: a sequence with no image must
reproduce the plain window, and a sequence with one must not.

The sequences are built from the reference's own sentinel roles
(`image_processor`: IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END =
0..4) offset by vocab_size, which is how a real span reaches the model.  Spans
longer than the window, longer than max_image_tokens, at the start, at the end,
two of them, and the malformed pair (an END with no START, a START with no END)
are all present, because the reference's cumsum/cummax/cummin expressions have a
defined answer for each and a port that "cleans them up" would diverge.

    python3 tests/vision_visible_goldens.py <snapshot-dir> > tests/test-vectors/vision-visible-goldens.bin
"""
import io
import json
import struct
import sys
import types

import numpy as np
import torch

SNAP = sys.argv[1]

# model.py imports a `kernel` module needing tilelang; the two functions below
# are module-level and never reach it, but the import still has to succeed.
_kernel = types.ModuleType("kernel")
for _n in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn",
           "hc_split_sinkhorn"):
    setattr(_kernel, _n, lambda *a, **k: None)
_kernel.block_size = 128
_kernel.fp4_block_size = 32
sys.modules["kernel"] = _kernel

sys.path.insert(0, SNAP + "/inference")
import model  # noqa: E402
from image_processor import IMAGE, IMAGE_START, IMAGE_END, IMAGE_PAD, IMAGE_NEW_LINE  # noqa: E402

CFG = json.load(open(SNAP + "/config.json"))
N_VOCAB = CFG["vocab_size"]
WINDOW = CFG["sliding_window"]
MAX_IMG = CFG["vision_max_n_token"]

TEXT = 100  # any id below vocab_size: an ordinary text token


def span(n_img, n_pad_lo=3, n_pad_hi=5, grid_w=6, rows=1, close=True):
    """A realistic sentinel span: START, pads, (IMAGE*w + NEWLINE)*rows, pads, END."""
    types = []
    if close is not False:
        types.append(IMAGE_START)
    types += [IMAGE_PAD] * n_pad_lo
    for _ in range(rows):
        types += [IMAGE] * grid_w + [IMAGE_NEW_LINE]
    types += [IMAGE_PAD] * n_pad_hi
    if close:
        types.append(IMAGE_END)
    return [N_VOCAB + t for t in types]


def plain_window(n):
    """The reference's non-visible matrix, padded to the visible width with -1."""
    width = min(n, WINDOW + MAX_IMG)
    m = model.get_window_topk_idxs(WINDOW, 1, n, 0)[0].numpy().astype(np.int32)
    out = np.full((n, width), -1, dtype=np.int32)
    out[:, :m.shape[1]] = m
    return out


def big_span(n_img):
    """One span holding n_img IMAGE slots, laid out in rows of 32."""
    return span(0, n_pad_lo=2, n_pad_hi=3, grid_w=32, rows=(n_img + 31) // 32)


CASES = [
    ("no image at all", [TEXT] * 40),
    ("span short, mid", [TEXT] * 4 + span(0, grid_w=6, rows=3) + [TEXT] * 4),
    ("span longer than window",
     [TEXT] * 4 + span(0, grid_w=32, rows=6) + [TEXT] * 4),          # 192 images > 128
    ("span at the very start", span(0, grid_w=8, rows=2) + [TEXT] * 6),
    ("span at the very end", [TEXT] * 6 + span(0, grid_w=8, rows=2)),
    ("two spans", span(0, grid_w=6, rows=2) + [TEXT] * 3 + span(0, grid_w=6, rows=2)),
    ("span longer than max_image_tokens",
     [TEXT] * 2 + big_span(400) + [TEXT] * 2),                        # 400 > 384
    ("END with no START", [TEXT] * 3 + [N_VOCAB + IMAGE_END] + [TEXT] * 4),
    ("START with no END", [TEXT] * 3 + [N_VOCAB + IMAGE_START] + [N_VOCAB + IMAGE] * 3),
    ("START and END adjacent", [TEXT] * 3 + [N_VOCAB + IMAGE_START, N_VOCAB + IMAGE_END] + [TEXT] * 3),
]


def main():
    out = io.BytesIO()
    out.write(b"VVS1")
    out.write(struct.pack("<5I", len(CASES), WINDOW, N_VOCAB, MAX_IMG, 0))

    for name, ids in CASES:
        n = len(ids)
        ids_t = torch.tensor([ids], dtype=torch.long)
        left_t, right_t = model.get_image_visible(ids_t, N_VOCAB, MAX_IMG)
        # get_window_topk_idxs_visible takes the (bsz, seqlen) tensors straight
        # from get_image_visible -- it calls .clamp on them, so numpy will not do.
        matrix_t = model.get_window_topk_idxs_visible(WINDOW, n, left_t, right_t, MAX_IMG)
        left = left_t[0].numpy().astype(np.int32)
        right = right_t[0].numpy().astype(np.int32)
        matrix = matrix_t[0].numpy().astype(np.int32)
        plain = plain_window(n)
        assert matrix.shape == plain.shape, (matrix.shape, plain.shape)

        out.write(name.encode().ljust(32, b"\0")[:32])
        out.write(struct.pack("<2I", n, matrix.shape[1]))
        out.write(np.asarray(ids, dtype=np.int32).tobytes())
        out.write(left.tobytes())
        out.write(right.tobytes())
        out.write(matrix.tobytes())
        out.write(plain.tobytes())
        print("  %-34s n=%-4d width=%-4d max left/right=%d/%d differs=%s" % (
            name, n, matrix.shape[1], int(left.max()), int(right.max()),
            not np.array_equal(matrix, plain)), file=sys.stderr)

    sys.stdout.buffer.write(out.getvalue())


main()

#!/usr/bin/env python3
"""Generate multi-axis rope position goldens from HF's own Qwen4ExpModel.get_rope_index (L268).

Each case is a token layout -- text runs and image runs (an image run is h' x w' image_pad tokens, the merged grid)
-- and the golden is get_rope_index's (T, H, W) position for every row, called on HF's method itself with only the
config it reads (vision_config.spatial_merge_size).  pulsar_test --lib-image-rope rebuilds the same image records and
requires pulsar_image_rope3 to give the same triple at every row.

    ~/venvs/q38/bin/python tests/image_rope_goldens.py > tests/test-vectors/image-rope-goldens.bin
"""
import struct
import sys
import types

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

MERGE = 2
fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=MERGE)))
fake.get_vision_position_ids = types.MethodType(Qwen4ExpModel.get_vision_position_ids, fake)

# (kind, n) runs: ("t", n_text) or ("i", (h', w')) -- the merged grid
CASES = [
    [("t", 7), ("i", (7, 10)), ("t", 5)],
    [("i", (8, 8)), ("t", 3)],
    [("t", 2), ("i", (3, 9)), ("t", 4), ("i", (12, 2)), ("t", 6)],
    [("t", 30), ("i", (1, 5)), ("t", 1), ("i", (5, 1)), ("t", 9)],
    [("t", 4)],
]

out = sys.stdout.buffer
out.write(b"IRG1")
out.write(struct.pack("<I", len(CASES)))
for runs in CASES:
    types_, grids, blocks = [], [], []
    for kind, v in runs:
        if kind == "t":
            types_ += [0] * v
        else:
            h, w = v
            blocks.append((len(types_), h * w, h, w))
            types_ += [1] * (h * w)
            grids.append([1, h * MERGE, w * MERGE])   # get_rope_index takes the patch grid; it divides by the merge
    L = len(types_)
    ids = torch.zeros(1, L, dtype=torch.long)
    tt = torch.tensor([types_], dtype=torch.int)
    pos, delta = Qwen4ExpModel.get_rope_index(fake, ids, tt, image_grid_thw=torch.tensor(grids) if grids else None)
    pos = pos[:, 0, :].to(torch.int32)
    out.write(struct.pack("<II", L, len(blocks)))
    for b in blocks:
        out.write(struct.pack("<4I", *b))
    out.write(pos.numpy().astype("<i4").tobytes())   # [3][L]: T row, H row, W row
    print(f"case: {L} rows, {len(blocks)} image block(s), rope delta {int(delta)}", file=sys.stderr)

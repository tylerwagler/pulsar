#!/usr/bin/env python3
"""Generate ViT + aligner goldens from the checkpoint's OWN vision.py.

Loads the checkpoint's vision/aligner weights out of the safetensors shards,
instantiates the reference `ViT` and `Aligner` classes from the snapshot's
`inference/vision.py`, and runs them on deterministic synthetic patches.  Dumps
the input patches and several STAGE outputs, so a mismatch in the engine's
forward can be bisected instead of hunted: patch_embed, blocks 0..2, the final
norm, and the aligner.

Bit-exactness is NOT expected and not asked for: the reference attention goes
through `F.scaled_dot_product_attention`, whose kernel reduction order is not
reproducible by a hand-written attention.  The gate therefore reports per-stage
max-abs and relative-RMS error and passes under a documented threshold.

    python3 tests/vision_tower_goldens.py <snapshot-dir> > tests/test-vectors/vision-tower-goldens.bin
"""
import json
import struct
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAP = sys.argv[1]
sys.path.insert(0, SNAP + "/inference")
import vision as V  # noqa: E402

N_STAGE_BLOCKS = 3          # blocks 0..2 are dumped as well as the final stack


class Args:
    def __init__(self, c):
        self.vision_n_layers = c["vision_n_layers"]
        self.vision_dim = c["vision_dim"]
        self.vision_n_heads = c["vision_n_heads"]
        self.vision_inter_dim = c["vision_inter_dim"]
        self.vision_patch_size = c["vision_patch_size"]
        self.vision_rope_theta = c["vision_rope_theta"]
        self.vision_downsample_ratio = c["vision_downsample_ratio"]
        self.dim = c["hidden_size"]


class Tower(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vision = V.ViT(args)
        self.aligner = V.Aligner(args)


def load_tower(args):
    cfg = json.load(open(SNAP + "/config.json"))
    index = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]
    want = {k: index[k] for k in index if k.startswith(("vision.", "aligner."))}
    per_shard = {}
    for k, shard in want.items():
        per_shard.setdefault(shard, []).append(k)
    tensors = {}
    dtypes = set()
    for shard, keys in per_shard.items():
        with safe_open(SNAP + "/" + shard, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                dtypes.add(str(t.dtype))
                tensors[k] = t
    sys.stderr.write("loaded %d vision/aligner tensors, dtypes=%s\n" % (len(tensors), sorted(dtypes)))

    tower = Tower(args)
    vis = {k[len("vision."):]: v for k, v in tensors.items() if k.startswith("vision.")}
    alg = {k[len("aligner."):]: v for k, v in tensors.items() if k.startswith("aligner.")}
    tower.vision.load_state_dict(vis, strict=True)
    tower.aligner.load_state_dict(alg, strict=True)
    return tower


def bf16_bits(t):
    return t.contiguous().view(torch.uint16).cpu().numpy().astype("<u2").tobytes()


def main():
    cfg = json.load(open(SNAP + "/config.json"))
    args = Args(cfg)
    # model.py does exactly this: vision/aligner .to(dtype=model_config.dtype).
    # The checkpoint stores the tower in bf16 and load_state_dict copies into
    # whatever dtype nn.Module created, so the cast is what makes the module the
    # checkpoint's own dtype -- and what the engine has to reproduce.
    tower = load_tower(args).to(torch.bfloat16).eval()
    # The same tower in fp32: the reference's own bf16 output differs from THIS
    # by a measurable amount, and that gap is the noise floor the gate calibrates
    # against.  Without it a tolerance is a number someone picked.
    tower_fp32 = load_tower(args).to(torch.float32).eval()
    p = args.vision_patch_size

    out = sys.stdout.buffer
    cases = [(4, 6), (7, 5), (3, 3)]
    out.write(b"VTX2")
    out.write(struct.pack("<I", len(cases)))

    for (nh, nw) in cases:
        n = nh * nw
        g = torch.Generator().manual_seed(1234 + n)
        patches = torch.randn(n, 3, p, p, generator=g).to(torch.bfloat16) * 0.5

        with torch.no_grad():
            x = tower.vision.patch_embed(patches)
            x0 = x
            cos, sin = V.get_vision_cos_sin(nh, nw, tower.vision.rope_dim, tower.vision.rope_theta)
            stages = []
            for i, block in enumerate(tower.vision.blocks):
                x = block(x, cos, sin)
                if i < N_STAGE_BLOCKS:
                    stages.append(x)
            normed = tower.vision.norm(x)
            aligned = tower.aligner(normed, nh, nw)
            with torch.no_grad():
                xf = tower_fp32.vision.patch_embed(patches.float())
                cosf, sinf = V.get_vision_cos_sin(nh, nw, tower_fp32.vision.rope_dim,
                                                  tower_fp32.vision.rope_theta)
                for block in tower_fp32.vision.blocks:
                    xf = block(xf, cosf, sinf)
                aligned_fp32 = tower_fp32.aligner(tower_fp32.vision.norm(xf), nh, nw)
        d = (aligned.float() - aligned_fp32).flatten()
        floor = (d.pow(2).sum() / aligned_fp32.flatten().pow(2).sum()).sqrt().item()

        n_llm_rows = aligned.shape[0]
        out.write(struct.pack("<iiiiiiifI", nh, nw, p, args.vision_dim, args.vision_n_heads,
                              args.vision_inter_dim, args.vision_downsample_ratio,
                              float(args.vision_rope_theta), N_STAGE_BLOCKS))
        out.write(struct.pack("<i", args.dim))
        out.write(struct.pack("<f", float(floor)))
        out.write(bf16_bits(patches))
        out.write(bf16_bits(x0))
        for s in stages:
            out.write(bf16_bits(s))
        out.write(bf16_bits(normed))
        out.write(bf16_bits(aligned))
        sys.stderr.write("case %dx%d: %d patches, aligner %d rows; bf16-vs-fp32 floor %.3e\n"
                         % (nh, nw, n, n_llm_rows, floor))


if __name__ == "__main__":
    main()

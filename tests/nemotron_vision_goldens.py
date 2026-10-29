#!/usr/bin/env python3
"""L331: Nemotron 3.5 Super VL's image path -- the CPU reference and its fixture, from the checkpoint's OWN code.

Runs on any CPU (no GPU): the checkpoint's `image_processing_nemotron_h_omni.py` (the processor), `modeling_radio.py`
(C-RADIO ViT-H/16) and `modeling_nemotron_h_omni.py` (the vision projector: vision_final_layernorm, pixel shuffle,
mlp1), imported from the checkpoint directory as a package, the vision tensors loaded by name with the checkpoint's own
conversion mapping (modeling_radio.register_radio_conversion_mapping, modeling_nemotron_h_omni's mlp1 renames) spelled
out below and every vision tensor accounted for.  Nothing of the language model is loaded.

Writes tests/test-vectors/nemotron-vision-goldens.bin (little-endian, "NVG1"):
  SIZES   the processor's tile for many image sizes: (w, h) -> (w_patches, h_patches, LLM tokens), from
          NemotronH_Omni_Reasoning_V3ImageProcessor._compute_target_patches at the processor's own budget
          (nemotron-vision-pixel-gate: the engine's tile rule against it, every size)
  PIXELS  encoded PNGs and the processor's pixel_values (f32 [3][H][W], torch's antialiased bicubic on the CPU):
          the bf16 bits of every value for tiles of at most FULL_MAX values (what the tower reads: the projector
          casts pixel_values to the tower's dtype) and a strided sample of the f32 values (the gate's ulp report)
  POS     the CPE position rows (modeling_radio's _interpolate_position_embedding, its own code on a synthetic
          128 x 128 x POS_C table: an LCG's values, so the gate rebuilds the table) for several tiles, every value f32
  TOWER   encoded PNGs and the rows the language model receives (the projector's output, 4096 wide), the tower run in
          FP32 on the CPU (the truth) -- stored as bf16 bits -- WITH vision_final_layernorm (the checkpoint's path; see
          L331 for why it is the only path) and, for case 0, WITHOUT it (information: how far apart the arms are); per
          case the reference's own bf16-vs-fp32 relative RMS (the self-calibrating floor, as vision-qwen-tower-gate) and
          the bf16-storage error; case 0 also the tower's residual after the last block (vit_out: cls/register tokens +
          patches, raster order, f32), first VIT_ROWS rows.

    ~/Projects/AI/temp/l331/venv/bin/python -I tests/nemotron_vision_goldens.py /mnt/models/nemotron35-super-vl-bf16 \
        tests/test-vectors/nemotron-vision-goldens.bin
(transformers prints to stdout, so the fixture is written to the path given, and stdout goes to stderr.)
"""
import copy
import importlib.util
import io
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

CKPT, OUT = sys.argv[1], sys.argv[2]
out = open(OUT, "wb")
sys.stdout = sys.stderr
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 8)
VIT_ROWS = 64
FULL_MAX = 900_000
log = lambda *a: print(*a, file=sys.stderr, flush=True)


def load_package(path, name="nhomni"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"),
                                                  submodule_search_locations=[path])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return {m: importlib.import_module(f"{name}.{m}") for m in
            ("configuration_nemotron_h_omni", "modeling_radio", "modeling_nemotron_h_omni",
             "image_processing_nemotron_h_omni")}


mods = load_package(CKPT)
Cfg = mods["configuration_nemotron_h_omni"].NemotronH_Omni_Reasoning_V3_Config
RadioModel = mods["modeling_radio"].RadioModel
Projector = mods["modeling_nemotron_h_omni"].NemotronH_Omni_Reasoning_V3VisionProjector
Processor = mods["image_processing_nemotron_h_omni"].NemotronH_Omni_Reasoning_V3ImageProcessor

cfg = Cfg.from_pretrained(CKPT, attn_implementation="sdpa")
pcfg = json.load(open(os.path.join(CKPT, "preprocessor_config.json")))
proc = Processor(**{k: pcfg[k] for k in ("norm_mean", "norm_std", "patch_size", "downsample_ratio", "min_num_patches",
                                         "max_num_patches", "max_model_len", "video_target_num_patches",
                                         "video_maintain_aspect_ratio")})

# ---- the vision tensors, by the checkpoint's own conversion mapping ----------------------------------------------
index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
VISION = ("vision_model.", "vision_projector.", "mlp1.")
RADIO_RENAMES = [  # modeling_radio.register_radio_conversion_mapping, in its order
    ("radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"),
    ("radio_model.model.patch_generator.embedder", "embeddings.patch_projection"),
    ("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding"),
    ("radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"),
    ("radio_model.model.blocks", "encoder.layer"),
    ("attn.proj", "attention.output.dense"),
    ("radio_model.input_conditioner", "input_conditioner"),
    ("radio_model.summary_idxs", "summary_idxs"),
]
radio_sd, proj_sd, consumed = {}, {}, set()
for name, shard in sorted(index.items()):
    if not name.startswith(VISION):
        continue
    with safe_open(os.path.join(CKPT, shard), "pt") as f:
        t = f.get_tensor(name)
    consumed.add(name)
    if name.startswith("vision_model."):
        k = name[len("vision_model."):]
        for a, b in RADIO_RENAMES:
            k = k.replace(a, b)
        if ".attn.qkv." in k:   # WeightConverter attn.qkv -> query / key / value, Chunk(dim=0)
            q, kk, v = t.chunk(3, dim=0)
            for part, x in (("query", q), ("key", kk), ("value", v)):
                radio_sd[k.replace("attn.qkv", f"attention.attention.{part}")] = x
        else:
            radio_sd[k] = t
    elif name.startswith("vision_projector."):
        proj_sd[name[len("vision_projector."):]] = t
    else:   # register_nemotron_h_omni_conversion_mapping: mlp1.0 / .1 / .3 -> mlp1.norm / linear1 / linear2
        k = name.replace("mlp1.0.", "mlp1.norm.").replace("mlp1.1.", "mlp1.linear1.").replace("mlp1.3.", "mlp1.linear2.")
        proj_sd[k] = t
log(f"vision tensors: {len(consumed)} consumed ({sum(t.numel() for t in radio_sd.values()) / 1e6:.1f}M tower, "
    f"{sum(t.numel() for t in proj_sd.values()) / 1e6:.1f}M projector)")


def build(dtype, final_norm):
    vm = RadioModel(copy.deepcopy(cfg.vision_config))   # its own config: the dtype below is per model
    vm.make_preprocessor_external()   # as the Omni model does: the processor normalises
    vm.input_conditioner = torch.nn.Identity()
    sd = {k: v for k, v in radio_sd.items() if not k.startswith("input_conditioner.")}
    missing, unexpected = vm.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    assert all("layer_scale" in m for m in missing), missing   # C-RADIO has no LayerScale (identity, value 1.0)
    assert not unexpected, unexpected
    for n, p in vm.named_parameters():
        if "layer_scale" in n:
            assert torch.all(p == 1.0), n
    vm = vm.to(dtype).eval()
    vm.config.dtype = dtype   # the projector casts pixel_values to the tower's config dtype
    pj = Projector(cfg)
    assert pj.vision_final_layernorm is not None   # num_nextn_predict_layers = 1: the checkpoint's path has it
    missing, unexpected = pj.load_state_dict({k: v.to(dtype) for k, v in proj_sd.items()}, strict=True)
    if not final_norm:
        pj.vision_final_layernorm = None
    return vm, pj.to(dtype).eval()


def make_png(kind, w, h):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    if kind == "circle":
        d.ellipse((w * 0.2, h * 0.2, w * 0.8, h * 0.8), fill=(220, 20, 20))
    elif kind == "text":
        d.rectangle((0, 0, w, h), fill=(250, 250, 240))
        d.text((w * 0.25, h * 0.3), "42", fill=(0, 0, 0), font_size=int(h * 0.4))
    elif kind == "shapes":
        d.rectangle((w * 0.05, h * 0.1, w * 0.45, h * 0.9), fill=(30, 60, 200))
        d.polygon([(w * 0.55, h * 0.9), (w * 0.95, h * 0.9), (w * 0.75, h * 0.1)], fill=(240, 210, 20))
    elif kind == "noise":
        rng = np.random.default_rng(w * 7919 + h)
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), (xx * 3 + yy * 5) % 256], -1)
        img = Image.fromarray(np.clip(base + rng.integers(-40, 41, size=(h, w, 3)), 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def processed(png):
    feat = proc(images=[Image.open(io.BytesIO(png))], return_tensors="pt")
    pv = feat["pixel_values"]
    pv = pv[0] if isinstance(pv, (list, tuple)) else pv[0]
    return pv.float(), int(feat["num_tokens"][0])


def rel_rms(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))


def bf16_bits(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype(np.uint16).tobytes()


out.write(b"NVG1")

# ---- SIZES ---------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(331)
sizes = [(1, 1), (16, 16), (32, 32), (48, 48), (512, 512), (511, 513), (640, 480), (1920, 1080), (1080, 1920),
         (4000, 3000), (8000, 8000), (13312, 16), (16, 13312), (10000, 10), (300, 3000), (1, 5000), (2049, 2047),
         (1024, 1024), (1600, 1664), (1847, 1847), (2048, 1536), (3840, 2160), (100, 80), (37, 53)]
sizes += [(int(rng.integers(1, 6000)), int(rng.integers(1, 6000))) for _ in range(76)]
budget = max(min((proc.max_model_len - 4) * proc._downsample_factor ** 2, proc.max_num_patches), proc.min_num_patches)


class _Size:
    def __init__(self, w, h):
        self.width, self.height = w, h


out.write(struct.pack("<II", len(sizes), budget))
for w, h in sizes:
    wp, hp = proc._compute_target_patches(_Size(w, h), budget)
    out.write(struct.pack("<5I", w, h, wp, hp, (wp * hp) // proc._downsample_factor ** 2))
log(f"sizes: {len(sizes)} cases at a budget of {budget} patches")

# ---- PIXELS --------------------------------------------------------------------------------------------------------
PIXELS = [("shapes", 1500, 1100), ("noise", 100, 80), ("shapes", 512, 512), ("text", 3000, 200), ("circle", 2000, 1500),
          ("text", 400, 300), ("shapes", 4000, 3000), ("text", 1000, 9000)]
out.write(struct.pack("<I", len(PIXELS)))
for kind, w, h in PIXELS:
    png = make_png(kind, w, h)
    pv, ntok = processed(png)
    vals = pv.numpy().astype(np.float32).ravel()
    stride = 61
    sample = vals[::stride]
    out.write(struct.pack("<I", len(png)) + png)
    out.write(struct.pack("<4I", pv.shape[1], pv.shape[2], ntok, stride))
    full = vals.size <= FULL_MAX   # the bf16 bits of every value for the small tiles; the sample alone past it
    out.write(struct.pack("<I", int(full)))
    if full:
        out.write(bf16_bits(pv))
    out.write(struct.pack("<I", len(sample)) + sample.tobytes())
    log(f"pixels {kind} {w}x{h}: tile {pv.shape[2]}x{pv.shape[1]} px, {ntok} LLM tokens")

# ---- POS -----------------------------------------------------------------------------------------------------------
POS_C = 8
POS_TILES = [(32, 32), (42, 26), (26, 42), (208, 64), (104, 128), (66, 200), (128, 128), (2, 802)]


def lcg_field(n):
    x, out = 331, np.empty(n, np.float32)
    for i in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out[i] = ((x >> 40) & 0xFFFF) / 65536.0 - 0.5
    return out


class _PE(mods["modeling_radio"].RadioPatchEmbeddings):
    def __init__(self):   # the table alone: _interpolate_position_embedding reads max_rows / max_cols and the table
        torch.nn.Module.__init__(self)
        self.max_rows = self.max_cols = 128
        self.position_embedding = torch.nn.Parameter(
            torch.from_numpy(lcg_field(128 * 128 * POS_C)).reshape(1, 128 * 128, POS_C), requires_grad=False)


pe = _PE()
out.write(struct.pack("<II", POS_C, len(POS_TILES)))
for wp, hp in POS_TILES:
    with torch.no_grad():
        rows = pe._interpolate_position_embedding((hp, wp), torch.float32)[0]   # [hp * wp][C], raster order
    out.write(struct.pack("<2I", wp, hp) + rows.contiguous().numpy().astype(np.float32).tobytes())
log(f"pos: {len(POS_TILES)} tiles over a {POS_C}-channel synthetic table")

# ---- TOWER ---------------------------------------------------------------------------------------------------------
TOWER = [("circle", 512, 512), ("text", 640, 384), ("shapes", 300, 500)]
models = {}
for dt in (torch.float32, torch.bfloat16):
    for fn in (True, False):
        models[(dt, fn)] = build(dt, fn)
out.write(struct.pack("<II", len(TOWER), VIT_ROWS))
for c, (kind, w, h) in enumerate(TOWER):
    png = make_png(kind, w, h)
    pv, ntok = processed(png)
    H, W = pv.shape[1] // 16, pv.shape[2] // 16
    res = {}
    t0 = time.time()
    with torch.no_grad():
        for (dt, fn), (vm, pj) in models.items():
            if not fn and c > 0:
                continue
            res[(dt, fn)] = pj(pv.unsqueeze(0), vm).float()[0]
            if dt == torch.float32 and fn and c == 0:
                vit_out = vm(pv.unsqueeze(0).to(dt)).last_hidden_state.float()[0]
    log(f"tower case {c} ({kind} {w}x{h} -> {W}x{H} patches, {ntok} rows): {time.time() - t0:.1f} s")
    arms = [True] + ([False] if c == 0 else [])
    out.write(struct.pack("<I", len(png)) + png)
    out.write(struct.pack("<4I", H, W, ntok, len(arms)))
    for fn in arms:
        ref = res[(torch.float32, fn)]
        assert ref.shape == (ntok, 4096), ref.shape
        floor = rel_rms(res[(torch.bfloat16, fn)], ref)
        store = rel_rms(ref.to(torch.bfloat16).float(), ref)
        out.write(struct.pack("<I2f", int(fn), floor, store))
        out.write(bf16_bits(ref))
        log(f"  final norm {fn}: bf16-vs-fp32 floor {floor:.3e}, bf16 storage {store:.3e}")
    if c == 0:
        d = rel_rms(res[(torch.float32, False)], res[(torch.float32, True)])
        log(f"  the arms differ by rel RMS {d:.3e} (fp32)")
        v = vit_out[:VIT_ROWS].contiguous().numpy().astype(np.float32)
        out.write(struct.pack("<2I", v.shape[0], v.shape[1]) + v.tobytes())
out.close()

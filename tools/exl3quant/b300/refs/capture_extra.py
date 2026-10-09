#!/usr/bin/env python3
"""Reference captures the kit's ref_capture_v41.py does not make, on DeepSeek's own inference/ stack (V4.1-Flash or
V4-Flash-Vision-Exp), under torchrun (MP = GPUs; rank 0 writes):

  images   prompts with ONE image block at chosen positions (L248 (B) / L283: a late image past the 4096 prefill
           chunk, one straddling it, one straddling the 256 grid, the vision_chunk_gate positions 40 / 256 / 300 /
           2200), each prefilled in ONE pass (the reference's rule: an image span lives in a single chunk).  Per case:
             <case>.tokens.bin   the expanded ids, int32 LE, exactly as the model's prepare_vl_inputs laid them out
                                 (V4.1: every image slot carries image_token_id, the layout is in the token types;
                                 Vision-Exp: image slots are vocab_size + type, its N-layout with pads)
             <case>.ref.bin      DS4PFXG1 v2 (the kit's write_blob, unchanged): fresh-prefill last-row f32 logits at
                                 depths E (the image's last slot), E+1, E+8, E+32 -- E = the block's end
             <case>.json         image start / length / grid / token types, the image file + sha256, per depth argmax
                                 and top-64
           and images/<name>.png, the image bytes every case used.  No engine gate reads these yet (the reference
           readers prefill token ids only); the files carry everything a gate needs to pair our prefill with them.
  depths   text-only rows at --depths for the kit's <slug>.tokens.bin, written exactly as the kit's phase_logits
           writes them (DS4PFXG1 v2 + sidecar): for a model the kit's script cannot load (Vision-Exp's Transformer
           takes no tokenizer).

    torchrun --nproc-per-node 2 capture_extra.py --inference $SRC/inference --ckpt $MPDIR --tokenizer $SRC \\
        --kit bundle/refkit --out refs/out/v41-images --mode images --model deepseek-ai/DeepSeek-V4.1-Flash --revision R
"""
import argparse
import hashlib
import inspect
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

CASES = [  # (name, where the image block starts: an int, or ("straddle", grid) = centred on that boundary)
    ("img-at40", 40), ("img-at256", 256), ("img-at300", 300), ("img-straddle256", ("straddle", 256)),
    ("img-deep2200", 2200), ("img-straddle4096", ("straddle", 4096)), ("img-late6000", 6000),
]
AFTER = (0, 1, 8, 32)  # depth rows at E + these (E = the block's end: the prompt through its last slot)


def synth_png(w, h, seed):
    """A deterministic test image (gradients + blocks + noise): the bytes are saved, so the engine side reads
    the same file."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([255 * x / w, 255 * y / h, 128 + 127 * np.sin(x / 17.0) * np.cos(y / 23.0)], -1)
    for _ in range(12):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        img[y0:y0 + rng.integers(10, 60), x0:x0 + rng.integers(10, 60)] = rng.integers(0, 256, 3)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


class IdsTokenizer:
    """The model's tokenizer, except encode() returns a fixed id list: prepare_vl_inputs then expands the image
    placeholder in OUR id stream (the kit's token bins are the single authority; nothing is re-tokenized)."""

    def __init__(self, tok, ids):
        self._tok, self._ids = tok, ids

    def encode(self, prompt, *a, **k):
        return list(self._ids)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def build_case(IP, args, tok, text, image_id, record, where):
    """(ids, token types or None, the ImageInput) for one image case: the placeholder at `where` of the text stream
    (an int), or centred on a boundary (("straddle", B): the block's length can depend on its start -- Vision-Exp
    pads to a multiple of 4 -- so the start is iterated), followed by max(AFTER)+1 text ids."""
    def expand(start, n_after):
        ids = text[:start] + [image_id] + text[start:start + n_after]
        res = IP.prepare_vl_inputs("", [record], IdsTokenizer(tok, ids), args)
        if len(res) == 3:   # V4.1: (tokens, token_types, images)
            return res[0], res[1], res[2][0]
        return res[0], None, res[1][0]   # Vision-Exp: (tokens, images), the types encoded in the ids
    start = where if isinstance(where, int) else where[1] - 1
    if not isinstance(where, int):
        for _ in range(4):
            start = where[1] - expand(start, 0)[2].types.numel() // 2
    ids, types, img = expand(start, max(AFTER) + 1)
    if not isinstance(where, int) and not img.start < where[1] < img.start + img.types.numel():
        raise SystemExit(f"{where}: block [{img.start}, {img.start + img.types.numel()}) does not straddle")
    return ids, types, img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inference", required=True, help="the snapshot's inference/ (model.py, image_processor.py)")
    ap.add_argument("--ckpt", required=True, help="convert.py --model-parallel MP output")
    ap.add_argument("--tokenizer", required=True, help="the HF snapshot (tokenizer.json)")
    ap.add_argument("--kit", required=True, help="bundle/refkit: ref_capture_v41.py (write_blob, fnv1a64), token bins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("images", "depths"), required=True)
    ap.add_argument("--slugs", default="story,code")
    ap.add_argument("--depths", default="64,128,256")
    ap.add_argument("--text-slug", default="story", help="images: the id stream the image is placed into")
    ap.add_argument("--model", required=True, help="HF repo id (manifest)")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    a = ap.parse_args()

    sys.path.insert(0, a.inference)
    sys.path.insert(0, os.path.join(os.path.dirname(a.inference), "encoding"))
    sys.path.insert(0, a.kit)
    import image_processor as IP
    import model as M
    import tilelang
    from ref_capture_v41 import fnv1a64, write_blob
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    ws, rank, local = (int(os.getenv(k, d)) for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if ws > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    torch.set_default_dtype(torch.bfloat16)
    args = M.ModelArgs(**json.load(open(os.path.join(a.inference, "config.json"))))
    args.max_batch_size, args.max_seq_len = 1, a.max_seq_len
    if hasattr(args, "temperature"):
        args.temperature = 0.0
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with torch.device("cuda"):
        model = M.Transformer(args, tok) if "tokenizer" in inspect.signature(M.Transformer.__init__).parameters \
            else M.Transformer(args)
    load_model(model, os.path.join(a.ckpt, f"model{rank}-mp{ws}.safetensors"))
    torch.set_default_device("cuda")
    fwd_params = inspect.signature(model.forward).parameters
    r0 = rank == 0
    if r0:
        os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    build_ref = f"ds-ref-extra torch{torch.__version__}"[:23]
    manifest = {"model": a.model, "revision": a.revision, "stack": "deepseek inference/ reference (capture_extra.py)",
                "mode": a.mode, "torch": torch.__version__, "tilelang": tilelang.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "mp": ws, "max_seq_len": a.max_seq_len, "temperature": 0.0,
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": {}}

    def say(*m):
        if r0:
            print(*m, flush=True)

    def prefill(ids, images=None, types=None):
        kw = {}
        if images is not None:
            kw["images"] = [images]
            if "token_types" in fwd_params:
                kw["token_types"] = torch.tensor([types], dtype=torch.int64, device="cuda")
        x = torch.tensor([ids], dtype=torch.int64, device="cuda")
        with torch.inference_mode():
            out = model.forward(x, start_pos=0, **kw)
        return out[1][0].float().cpu().numpy()

    def side(row):
        top = np.argsort(-row)[:64]
        return {"argmax": int(top[0]), "top64_ids": top.tolist(), "top64_logits": row[top].tolist()}

    if a.mode == "depths":
        for slug in a.slugs.split(","):
            ids = np.fromfile(os.path.join(a.kit, f"{slug}.tokens.bin"), dtype="<i4")
            depths = [d for d in (int(x) for x in a.depths.split(",")) if d <= len(ids)]
            rows, sc = [], {"depths": {}}
            for d in depths:
                t0 = time.time()
                rows.append(prefill(ids[:d].astype(np.int64).tolist()))
                sc["depths"][str(d)] = dict(side(rows[-1]), seconds=round(time.time() - t0, 1))
                say(f"{slug} depth {d}: argmax {sc['depths'][str(d)]['argmax']}")
            if r0:
                fnv = fnv1a64(ids[:depths[-1]].astype("<i4").tobytes())
                write_blob(os.path.join(a.out, f"{slug}.ref.bin"), rows, rows[0].shape[0], depths, fnv, build_ref)
                ids.tofile(os.path.join(a.out, f"{slug}.tokens.bin"))
                json.dump(dict(sc, fnv=f"0x{fnv:016x}"), open(os.path.join(a.out, f"{slug}.ref.json"), "w"), indent=1)
            manifest["cases"][slug] = {"depths": depths}
    else:
        text = np.fromfile(os.path.join(a.kit, f"{a.text_slug}.tokens.bin"), dtype="<i4").astype(np.int64).tolist()
        # V4.1's prepare_vl_inputs takes the id from the config, Vision-Exp's from the tokenizer (same spelling)
        image_id = getattr(args, "image_token_id", None) or tok.convert_tokens_to_ids("<｜deepseek_image｜>")
        png = synth_png(640, 480, seed=4101)
        png_name = "synth-640x480.png"
        if r0:
            open(os.path.join(a.out, "images", png_name), "wb").write(png)
        record = {"data": png}

        for name, where in CASES:
            ids, types, img = build_case(IP, args, tok, text, image_id, record, where)
            span = img.types.numel()
            end = img.start + span                      # first position after the block
            depths = [end + k for k in AFTER]
            rows, sc = [], {"depths": {}}
            for d in depths:
                t0 = time.time()
                rows.append(prefill(ids[:d], [img], None if types is None else types[:d]))
                sc["depths"][str(d)] = dict(side(rows[-1]), seconds=round(time.time() - t0, 1))
            say(f"{name}: block [{img.start}, {end}) ({span} slots, vit {img.n_vit_h}x{img.n_vit_w}); depths {depths} "
                f"argmax {[sc['depths'][str(d)]['argmax'] for d in depths]}")
            if r0:
                arr = np.asarray(ids[:depths[-1]], dtype="<i4")
                fnv = fnv1a64(arr.tobytes())
                write_blob(os.path.join(a.out, f"{name}.ref.bin"), rows, rows[0].shape[0], depths, fnv, build_ref)
                arr.tofile(os.path.join(a.out, f"{name}.tokens.bin"))
                sc.update({"fnv": f"0x{fnv:016x}", "image": png_name, "image_sha256": hashlib.sha256(png).hexdigest(),
                           "image_start": img.start, "image_slots": span, "vit_grid": [img.n_vit_h, img.n_vit_w],
                           "token_types": img.types.tolist(), "image_token_id": image_id,
                           "text_slug": a.text_slug, "text_before": img.start, "text_after": max(AFTER) + 1})
                json.dump(sc, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
            manifest["cases"][name] = {"image_start": img.start, "image_slots": span, "depths": depths}
    if r0:
        json.dump(manifest, open(os.path.join(a.out, "capture_manifest.json"), "w"), indent=1)
    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

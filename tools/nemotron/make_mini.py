#!/usr/bin/env python3
"""make_mini.py -- the Nemotron 3.5 Super VL mini fixture (L296 N1): a few real layers of the Omni checkpoint sliced
into a small BF16 checkpoint in the SAME HF layout (the Omni wrapper's config.json with its nested llm_config, the
`language_model.` prefix, the vision tensors whole), so the builder (tools/container/nemotron_h.py) and transformers
(trust_remote_code) read it exactly as they read the full model.

    python3 tools/nemotron/make_mini.py <src_dir> <out_dir>

The mini (HF blocks of the source -> the mini's):
    src 0 (mamba) -> 0, src 1 (moe) -> 1, src 6 (mamba) -> 2, src 7 (attention) -> 3, src 8 (moe) -> 4
so layers_block_type = [mamba, moe, mamba, attention, moe] and pulsar's grouping (plan 121 section 1: a mixer and
the MoE after it) gives 3 layers -- {Mamba-2, MoE}, {Mamba-2, no FFN}, {attention, MoE}: every layer kind the full
model has.  The MTP module (language_model.mtp.layers.0 attention + .1 MoE), embeddings, norm_f, lm_head and the
vision tower + projector are copied whole.  The side files (tokenizer, chat template, the remote code) are copied
as shipped.  Pure byte copies (no dtype interpretation); header-only planning, then one pass of pwrites.
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import struct
import sys

LAYER_MAP = {0: 0, 1: 1, 6: 2, 7: 3, 8: 4}
CHUNK = 256 << 20
EXTRA = ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
         "preprocessor_config.json", "processor_config.json", "configuration_nemotron_h_omni.py",
         "configuration_radio.py", "modeling_nemotron_h_omni.py", "modeling_radio.py",
         "image_processing_nemotron_h_omni.py", "processing_nemotron_h_omni.py", "ultra_v3_reasoning_parser.py",
         "__init__.py"]
BACKBONE = re.compile(r"^language_model\.backbone\.layers\.(\d+)\.(.*)$")


def header(path):
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        return n, json.loads(f.read(n))


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(weight_map.values()))
    with cf.ThreadPoolExecutor(8) as ex:
        headers = dict(zip(shards, ex.map(lambda s: header(os.path.join(src, s)), shards)))
    files = {}
    for shard, (hlen, hdr) in headers.items():
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            m = BACKBONE.match(k)
            if m:
                if int(m.group(1)) not in LAYER_MAP:
                    continue
                li = LAYER_MAP[int(m.group(1))]
                nk, fn = f"language_model.backbone.layers.{li}.{m.group(2)}", f"layer-{li}.safetensors"
            elif k.startswith("language_model.mtp."):
                nk, fn = k, "mtp.safetensors"
            elif k.startswith("language_model."):
                nk, fn = k, "globals.safetensors"
            else:
                nk, fn = k, "vision.safetensors"
            s, e = v["data_offsets"]
            files.setdefault(fn, []).append((nk, v["dtype"], v["shape"], shard, 8 + hlen + s, e - s))
    index = {"metadata": {}, "weight_map": {}}
    total = 0
    for fn, items in sorted(files.items()):
        items.sort()
        hdr, pos = {}, 0
        for nk, dt, sh, _, _, n in items:
            hdr[nk] = {"dtype": dt, "shape": sh, "data_offsets": [pos, pos + n]}
            pos += n
            index["weight_map"][nk] = fn
        hdr["__metadata__"] = {"format": "pt"}
        hb = json.dumps(hdr, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        path = os.path.join(out, fn)
        total += pos
        if os.path.exists(path):
            print("have", fn, flush=True)
            continue
        fd = os.open(path + ".part", os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.pwrite(fd, struct.pack("<Q", len(hb)) + hb, 0)
        base = 8 + len(hb)
        jobs = []
        for nk, _, _, shard, off, n in items:
            dst = base + hdr[nk]["data_offsets"][0]
            for c in range(0, n, CHUNK):
                jobs.append((os.path.join(src, shard), off + c, min(CHUNK, n - c), dst + c))

        def run(j):
            with open(j[0], "rb") as f:
                f.seek(j[1])
                buf = f.read(j[2])
            if len(buf) != j[2]:
                raise SystemExit(f"short read {j}")
            os.pwrite(fd, buf, j[3])
            return j[2]
        with cf.ThreadPoolExecutor(8) as ex:
            done = sum(ex.map(run, jobs))
        os.fsync(fd)
        os.close(fd)
        os.rename(path + ".part", path)
        print(f"{fn}: {len(items)} tensors, {done / 1e9:.2f} GB", flush=True)
    index["metadata"]["total_size"] = total
    json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(src, "config.json")))
    types = cfg["llm_config"]["layers_block_type"]
    cfg["llm_config"]["layers_block_type"] = [types[s] for s in sorted(LAYER_MAP)]
    cfg["_mini_source"] = {"source": os.path.abspath(src),
                           "layer_map_src_to_mini": {str(k): v for k, v in sorted(LAYER_MAP.items())}}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in EXTRA:
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    print(f"done: {len(index['weight_map'])} tensors, {total / 1e9:.2f} GB, blocks {cfg['llm_config']['layers_block_type']}",
          flush=True)


if __name__ == "__main__":
    main()

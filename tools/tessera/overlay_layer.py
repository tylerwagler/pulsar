#!/usr/bin/env python3
"""overlay_layer.py -- a Qwen container with ONE layer's routed experts in Tessera planes (L255).

    python3 overlay_layer.py --base CONTAINER_DIR --blobs BLOBS.pt --layer 12 --out OUT_DIR

OUT_DIR is the base container with layer N's routed experts carried as Tessera's value-family planes: every base
file is symlinked except the shard holding that layer's EXL3 expert stacks, which is rewritten WITHOUT them (a
layer declares its experts once -- the engine refuses both forms), and one new shard,
model-tessera-l<N>.safetensors, carries mlp.experts.tessera.<gate|up|down>.<plane> (family_qwen.h
PULSAR_QWEN_TESS_*), prepared by Tessera's own load-time prep (planes.py, the planes tests/tessera_kernel_gate
proves).  BLOBS is a torch.save dict {"gate"/"up"/"down": [unit blob per expert, expert order]} of q256 1024
BF16-grid units.  Needs Tessera + torch + a CUDA device (the prep runs there).

Shards are written the way the container builder writes them: the header padded to a 256-byte boundary, the
data contiguous in the original order, every tensor 16-byte aligned.

Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
"""
import argparse
import json
import os
import struct

import torch

import planes

DTYPE_NAME = {torch.int32: "I32", torch.float32: "F32", torch.bfloat16: "BF16", torch.uint8: "U8"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_shard(path, entries, metadata, host_only=()):
    """entries: [(name, dtype_name, shape, bytes)] in data order.  Every tensor starts 16-byte aligned (the device
    reads), except the names in host_only, which need only their element alignment (4 bytes)."""
    header, off = {}, 0
    for name, dt, shape, data in entries:
        if off % (4 if name in host_only else 16):
            raise SystemExit(f"{path}: {name} would start at {off}, not aligned for its reader")
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
    header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 256)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for _, _, _, data in entries:
            f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prefix = f"model.language_model.layers.{a.layer}.mlp.experts."
    os.makedirs(a.out, exist_ok=False)
    shards = sorted(f for f in os.listdir(a.base) if f.endswith(".safetensors"))
    target = None
    for f in shards:
        hdr, _ = read_header(os.path.join(a.base, f))
        stacks = json.loads(hdr.get("__metadata__", {}).get("pulsar.experts", "[]"))
        if any(e["gguf_name"].startswith(prefix) for e in stacks):
            if target:
                raise SystemExit(f"layer {a.layer}'s experts span {target} and {f}; this tool rewrites one shard")
            target = f
    if not target:
        raise SystemExit(f"no shard of {a.base} declares {prefix}* expert stacks")

    for f in os.listdir(a.base):
        if f != target and f != "SHA256SUMS":   # the sums no longer describe this container
            os.symlink(os.path.abspath(os.path.join(a.base, f)), os.path.join(a.out, f))

    # the rewritten shard: every tensor but the layer's per-expert EXL3 slices, and no stack entry for them
    src = os.path.join(a.base, target)
    hdr, data_start = read_header(src)
    meta = hdr.pop("__metadata__")
    stacks = json.loads(meta.get("pulsar.experts", "[]"))
    dropped = [e for e in stacks if e["gguf_name"].startswith(prefix)]
    meta["pulsar.experts"] = json.dumps([e for e in stacks if not e["gguf_name"].startswith(prefix)],
                                        separators=(",", ":"))
    drop_names = set()
    for e in dropped:
        for i in range(e["n_experts"]):
            drop_names.add(e["entry_name"].replace("{e}", str(i)))
    missing = drop_names - set(hdr)
    if missing:
        raise SystemExit(f"{target}: {len(missing)} expert tensors named by pulsar.experts are absent")
    keep = sorted(((v["data_offsets"][0], k, v) for k, v in hdr.items() if k not in drop_names))
    entries = []
    with open(src, "rb") as f:
        for off0, name, v in keep:
            f.seek(data_start + off0)
            entries.append((name, v["dtype"], v["shape"], f.read(v["data_offsets"][1] - off0)))
    write_shard(os.path.join(a.out, target), entries, meta)
    print(f"{target}: dropped {len(drop_names)} expert tensors ({[e['gguf_name'] for e in dropped]}), kept "
          f"{len(entries)}", flush=True)

    # the Tessera shard
    n_expert = dropped[0]["n_experts"]
    blobs = torch.load(a.blobs)
    if any(len(blobs[k]) != n_expert for k in planes.PROJS):
        raise SystemExit(f"{a.blobs}: {[len(blobs[k]) for k in planes.PROJS]} experts, the layer has {n_expert}")
    pl = planes.stack_planes(planes.fused_stack(blobs, torch.device("cuda")))
    entries, decl, geoms = [], {}, []
    for k in planes.PROJS:
        for p in planes.PLANES:
            t = pl[k][p]
            name = f"{prefix}tessera.{k}.{p}"
            item = (name, DTYPE_NAME[t.dtype], tuple(t.shape), t.view(torch.uint8).numpy().tobytes())
            (geoms if p == "geom" else entries).append(item)   # the 8-byte geoms go last: they break 16B alignment
            decl[name] = {"layout": "native", "gguf_name": name, "dims_ne": list(reversed(t.shape))}
    new = f"model-tessera-l{a.layer}.safetensors"
    write_shard(os.path.join(a.out, new), entries + geoms, host_only={g[0] for g in geoms}, metadata=
                {"format": "pt", "pulsar.format": "pulsar-safetensors-v1", "pulsar.family": meta.get("pulsar.family", ""),
                 "pulsar.shard": new, "pulsar.shard_key": f"layers.{a.layer}.tessera",
                 "pulsar.tensors": json.dumps(decl, separators=(",", ":")),
                 "pulsar.tessera": "Made with Tessera by Robert Tand - https://github.com/RobTand/tessera"})
    print(f"{new}: {len(decl)} planes, {os.path.getsize(os.path.join(a.out, new)) / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()

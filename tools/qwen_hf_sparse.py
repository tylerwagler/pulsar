#!/usr/bin/env python3
"""qwen_hf_sparse.py OUT [--revision R] -- what qwen-family-gate reads of the Qwen3.8-Flash-Next checkpoint, sparsely.

tests/qwen_family_container.py reads the checkpoint's config, tokenizer, index, shard headers and PLE buffers: a few
hundred MB of a ~300 GB repo.  This writes those bytes at their true offsets, each shard a sparse file of its full
size, by HTTP range requests.  Set QWEN_HF_DIR to OUT.  A token, if any, comes from HF_TOKEN or HF_TOKEN_PATH."""
import argparse
import importlib.util
import json
import os
import struct
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--revision", default="de4b8e4d43b917e7706784d8bb445c9af86a3540")
a = ap.parse_args()
BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % a.revision
tok = os.environ.get("HF_TOKEN")
tp = os.environ.get("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
if not tok and os.path.exists(tp):
    tok = open(tp).read().strip()
HDR = {"Authorization": "Bearer " + tok} if tok else {}


def get(name, lo=None, hi=None):
    h = dict(HDR)
    if lo is not None:
        h["Range"] = "bytes=%d-%d" % (lo, hi - 1)
    with urllib.request.urlopen(urllib.request.Request(BASE + name, headers=h), timeout=120) as r:
        return r.read(), r.headers


spec = importlib.util.spec_from_file_location("qfc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "qwen_family_container.py"))
qfc = importlib.util.module_from_spec(spec)
sys.modules["qfc"] = qfc
spec.loader.exec_module(qfc)
if os.path.exists(a.out):
    sys.exit("%s exists: this writes a fresh copy" % a.out)
os.makedirs(a.out)
for f in ("config.json", "generation_config.json", "tokenizer.json", "model.safetensors.index.json"):
    open(os.path.join(a.out, f), "wb").write(get(f)[0])
index = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]
got = 0
for shard in sorted(set(index.values())):
    n = struct.unpack("<Q", get(shard, 0, 8)[0])[0]
    raw, h = get(shard, 0, 8 + n)
    size = int(h["Content-Range"].rsplit("/", 1)[1])
    hdr = json.loads(raw[8:])
    with open(os.path.join(a.out, shard), "wb") as f:
        f.write(raw)
        got += len(raw)
        for name, info in hdr.items():
            if name != "__metadata__" and any(name.endswith(k) for k in qfc.PLE_BUFFERS):
                lo, hi = info["data_offsets"]
                data = get(shard, 8 + n + lo, 8 + n + hi)[0]
                f.seek(8 + n + lo)
                f.write(data)
                got += len(data)
        f.truncate(size)
    print("%s: %d GB sparse" % (shard, size // 10**9), flush=True)
print("%d shards, %.1f MB fetched" % (len(set(index.values())), got / 1e6))

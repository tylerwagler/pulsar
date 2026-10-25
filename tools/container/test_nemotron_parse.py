#!/usr/bin/env python3
"""test_nemotron_parse.py -- NVIDIA Nemotron-3.5 Parse in the artifact builder (nemotron_parse.py, L321), on a synthetic
checkpoint at the REAL geometry (the declaration pins every dimension, so there is no tiny one): one model.safetensors
whose header is the real layout and whose payload is a sparse file -- zeros, ~0 bytes on disk -- except decoder layer 0
and summary_idxs, which carry real bytes.

  * plan --dump: 12 shards (vision, layers.0..9, top); GOLDEN over the dump (every header byte a build writes);
    the records (role + block / layer), the declared shapes (conv1 / conv2 as Linears, the position field 2-D), f16 for
    every Linear and the embedding, f32 for the rest; pulsar.kv's pinned facts; the consumed tensors (the input
    conditioner, summary_idxs, the MTP head)
  * emit --shard layers.0 twice, byte-identical: every Linear == f16(W) as stored ([out, in] = ne [in, out]), every
    norm / bias == its f32 bytes
  * refusals by name: a pinned value (repetition penalty, KV heads), the backbone, summary_idxs, an unknown tensor, a
    missing tensor, a wrong shape, a Linear recipe row that is not f16, --exl3, no recipe
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nemotron_parse as NP  # noqa: E402

FAILS: list[str] = []
# sha256 of the plan dump (the source directory as <hf>); fixed when the family landed (L321)
GOLDEN = {"plan": "62bce3c15d58e7f7d41745e951f7e21e5819f5278f5f1c8631b5d3d0beddaef0"}
RECIPE = os.path.join(HERE, "format-maps", "nemotron_parse-f16.json")


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def configs():
    """config.json / preprocessor_config.json / generation_config.json carrying exactly the pinned values"""
    blocks = {"top": {"model_type": NP.MODEL_TYPE, "architectures": [NP.ARCH]},
              "dec": {}, "enc": {"args": {"model": "vit_so400m_patch16_224", "register_multiple": 10}},
              "pre": {}, "gen": {}}
    for (where, key), v in NP.PINS.items():
        blocks[where][key] = v
    top = blocks["top"]
    top["decoder"], top["encoder"] = blocks["dec"], blocks["enc"]
    return top, blocks["pre"], blocks["gen"]


def tensors():
    """{name: (dtype, shape)}: the declaration's tensors + the consumed ones"""
    t = {n: (dt, shp) for n, (dt, shp, _m, _d) in NP.expected().items()}
    rp = "encoder.model_encoder.radio_model."
    t[rp + "input_conditioner.norm_mean"] = ("F32", [3, 1, 1])
    t[rp + "input_conditioner.norm_std"] = ("F32", [3, 1, 1])
    t[rp + "summary_idxs"] = ("I64", [2])
    t["decoder.extra_heads.0.fc1.weight"] = ("F32", [4096, 1024])
    t["decoder.extra_proj.weight"] = ("F32", [1024, 2048])
    return t


def make_hf(root, rng, edit=None, cfg_edit=None, idx=(0, 1)):
    """the sparse checkpoint; returns {name: bytes} of the tensors given real data"""
    os.makedirs(root, exist_ok=True)
    top, pre, gen = configs()
    if cfg_edit:
        cfg_edit(top, pre, gen)
    for name, obj in (("config.json", top), ("preprocessor_config.json", pre), ("generation_config.json", gen)):
        json.dump(obj, open(os.path.join(root, name), "w"))
    open(os.path.join(root, "tokenizer.json"), "w").write('{"synthetic": "tokenizer.json"}\n')
    t = tensors()
    if edit:
        edit(t)
    size = {"F32": 4, "I64": 8}
    hdr, off = {}, 0
    for n in sorted(t):
        dt, shp = t[n]
        nb = int(np.prod(shp)) * size[dt]
        hdr[n] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + nb]}
        off += nb
    hj = json.dumps(hdr).encode()
    hj += b" " * (-len(hj) % 8)
    base = 8 + len(hj)
    data = {}
    for n in hdr:
        if n.startswith("decoder.layers.0."):
            data[n] = (rng.standard_normal(hdr[n]["shape"]).astype(np.float32) * 0.05).tobytes()
    if "encoder.model_encoder.radio_model.summary_idxs" in hdr:
        data["encoder.model_encoder.radio_model.summary_idxs"] = np.array(idx, dtype="<i8").tobytes()
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj)
        f.truncate(base + off)   # sparse: the payload reads as zeros
        for n, b in data.items():
            f.seek(base + hdr[n]["data_offsets"][0])
            f.write(b)
    return data


def run(*args, expect_fail=False, contains=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py")] + list(args), capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if contains is not None:
        ok = ok and contains in out
    if not ok:
        print("    " + "\n    ".join(out.strip().splitlines()[-4:]))
    return ok, out


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    return h, 8 + n


def entry_bytes(path, name):
    h, base = header(path)
    o0, o1 = h[name]["data_offsets"]
    with open(path, "rb") as g:
        g.seek(base + o0)
        return g.read(o1 - o0)


def main():
    rng = np.random.default_rng(321)
    tmp = tempfile.mkdtemp(prefix="nemotron-parse-builder-test-")
    try:
        hf = os.path.join(tmp, "hf")
        data = make_hf(hf, rng)
        args = ["--hf", hf, "--recipe", RECIPE]

        print("plan")
        dump = os.path.join(tmp, "plan.jsonl")
        ok, out = run("plan", *args, "--dump", dump)
        check(ok and "shards 12" in out, "plan: 12 shards")
        lines = [json.loads(x) for x in open(dump)] if ok else []
        # the dump names the source by path: the temporary directory's, normalised before the digest
        got = hashlib.sha256(open(dump, "rb").read().replace(hf.encode(), b"<hf>")).hexdigest() if ok else ""
        check(got == GOLDEN["plan"], f"GOLDEN plan: {got}")
        model = lines[0] if lines else {}
        check(model.get("shards") == ["vision"] + [f"layers.{i}" for i in range(10)] + ["top"],
              f"shard order {model.get('shards')}")
        check(model.get("consumed") == 5, f"{model.get('consumed')} consumed, not written: the input conditioner's "
                                          f"two, summary_idxs, the MTP head's two")
        heads = {x["shard"]: json.loads(x["header"]) for x in lines if "shard" in x}
        decl, ents = {}, {}
        for s, h in heads.items():
            meta = h.get("__metadata__", {})
            if "pulsar.tensors" in meta:
                decl.update(json.loads(meta["pulsar.tensors"]))
            ents.update({n: v for n, v in h.items() if n != "__metadata__"})
        check(len(ents) == len(NP.expected()), f"{len(ents)} tensors written == the declaration's {len(NP.expected())}")
        vit = NP.VIT
        want = {vit + "blocks.3.attn.qkv.weight": ("F16", [3456, 1152]), vit + "blocks.3.attn.qkv.bias": ("F32", [3456]),
                vit + "patch_generator.pos_embed": ("F32", [16384, 1152]),
                "encoder.conv1.weight": ("F16", [1024, 1152]), "encoder.conv2.weight": ("F16", [1024, 4096]),
                "decoder.layers.0.self_attn.k_proj.weight": ("F16", [512, 1024]),
                "decoder.embed_tokens.weight": ("F16", [72512, 1024]),
                "decoder.layer_norm.bias": ("F32", [1024])}
        for n, (dt, shp) in want.items():
            e = ents.get(n, {})
            check(e.get("dtype") == dt and e.get("shape") == shp, f"{n}: {e.get('dtype')} {e.get('shape')}")
        r = decl.get(vit + "blocks.3.mlp.fc1.weight", {})
        check(r.get("role") == "vit_fc1" and r.get("block") == "vit" and r.get("layer") == 3, f"a ViT record: {r}")
        r = decl.get("decoder.layers.7.fc2.weight", {})
        check(r.get("role") == "ffn_down" and r.get("block") == "layers" and r.get("layer") == 7, f"a decoder record: {r}")
        check(decl.get("encoder.conv2.weight", {}).get("role") == "neck_merge", "conv2: role neck_merge")
        kvm = heads.get("vision", {}).get("__metadata__", {})
        kv = {e["key"]: e["value"] for e in json.loads(kvm.get("pulsar.kv", "[]"))}
        check(kv.get("general.architecture") == "nemotron_parse" and kv.get("nemotron_parse.vit.head_dim") == 72 and
              kv.get("nemotron_parse.dec.kv_heads") == 8 and kv.get("nemotron_parse.gen.prompt") == NP.PROMPT and
              "nemotron_parse.vit.head_dim_stored" not in kv,
              f"pulsar.kv: {len(kv)} facts, the pinned ones as the engine reads them")

        print("emit layers.0")
        o1, o2 = os.path.join(tmp, "out1"), os.path.join(tmp, "out2")
        ok1, _ = run("emit", *args, "--out", o1, "--shard", "layers.0")
        ok2, _ = run("emit", *args, "--out", o2, "--shard", "layers.0")
        f1 = sorted(os.listdir(o1)) if ok1 else []
        check(ok1 and ok2 and len(f1) == 1 and f1 == sorted(os.listdir(o2)) and
              open(os.path.join(o1, f1[0]), "rb").read() == open(os.path.join(o2, f1[0]), "rb").read(),
              f"emit --shard layers.0 twice, byte-identical ({f1})")
        exact = 0
        for n, b in data.items():
            if not n.startswith("decoder.layers.0."):
                continue
            src = np.frombuffer(b, dtype=np.float32)
            dense = NP.DEC_LAYER[n[len("decoder.layers.0."):]][1] == "dense"
            exact += entry_bytes(os.path.join(o1, f1[0]), n) == (src.astype(np.float16) if dense else src).tobytes()
        check(exact == len(NP.DEC_LAYER), f"{exact} / {len(NP.DEC_LAYER)} tensors: Linears f16(W) as stored, the rest f32")

        print("refusals")
        cases = [
            ("repetition penalty 1.2", None, lambda t, p, g: g.update(repetition_penalty=1.2), "repetition_penalty"),
            ("KV heads 16", None, lambda t, p, g: t["decoder"].update(self_attention_kv_heads=16), "self_attention_kv_heads"),
            ("another backbone", None, lambda t, p, g: t["encoder"]["args"].update(model="vit_large_patch16_224"),
             "vit_so400m_patch16_224"),
            ("an unknown tensor", lambda t: t.update({"decoder.layers.0.extra.weight": ("F32", [4])}), None, "does not map"),
            ("a missing tensor", lambda t: t.pop("decoder.layers.4.fc1.bias"), None, "missing"),
            ("a wrong shape", lambda t: t.update({"decoder.layers.2.fc1.weight": ("F32", [4096, 512])}), None,
             "declared shape"),
        ]
        for i, (why, edit, cfg_edit, needle) in enumerate(cases):
            d = os.path.join(tmp, f"bad{i}")
            make_hf(d, rng, edit=edit, cfg_edit=cfg_edit)
            ok, out = run("plan", "--hf", d, "--recipe", RECIPE, expect_fail=True, contains=needle)
            check(ok, f"refused ({why}): {out.strip().splitlines()[-1][:140] if out.strip() else ''}")
            shutil.rmtree(d)
        d = os.path.join(tmp, "badidx")
        make_hf(d, rng, idx=(0, 2))
        ok, out = run("plan", "--hf", d, "--recipe", RECIPE, expect_fail=True, contains="summary_idxs")
        check(ok, f"refused (summary_idxs [0, 2]): {out.strip().splitlines()[-1][:140]}")
        rec = json.load(open(RECIPE))
        bad = copy.deepcopy(rec)
        bad["rows"][0]["format"] = "native"
        bp = os.path.join(tmp, "recipe-native.json")
        json.dump(bad, open(bp, "w"))
        ok, out = run("plan", "--hf", hf, "--recipe", bp, expect_fail=True, contains="not native")
        check(ok, f"refused (a Linear row native): {out.strip().splitlines()[-1][:140]}")
        ok, out = run("plan", "--hf", hf, "--recipe", RECIPE, "--exl3", hf, expect_fail=True, contains="--exl3")
        check(ok, f"refused (--exl3): {out.strip().splitlines()[-1][:140]}")
        ok, out = run("plan", "--hf", hf, expect_fail=True, contains="--recipe")
        check(ok, f"refused (no recipe): {out.strip().splitlines()[-1][:140]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_nemotron_parse: {'FAIL (' + str(len(FAILS)) + ')' if FAILS else 'PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

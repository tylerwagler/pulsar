#!/usr/bin/env python3
"""The DeepSeek family of the direct builder end to end on a SYNTHETIC miniature checkpoint (L279 step 0) -- no
mounts, no GPU, seconds.  The Vision-Exp checkpoint's names and dtypes at toy shapes (hidden 256, 3 layers, one
hash layer, 4 experts, one drafter layer, an Engram layer, a tower stub), an exllamav3-layout EXL3 expert
checkpoint for layers 1 and 2 (K2 / K2.5 + K3), a toy BPE tokenizer.  What it pins:

  * every DeepSeek source kind the builder writes: FP8 + 128x128 and 32x32 block scales -> mxfp8_lt (rederive and
    verbatim), I8 + E8M0 experts -> cutlass_mxfp4, EXL3 experts -> exl3m_k2 / k2h / k3 byte ranges, BF16 markov_w2
    -> fp8_e4m3_soa_k (declared transposed), I64 tid2eid -> i32, the [1, n] confidence head declared rank-1, native
    BF16 / F32 / I32, the three drop rules (hash-layer gate bias, drafter bias_vl, the Engram row table)
  * plan -> emit twice -> byte-identical shards -> verify PASS -> audit PASS, for the default build and for one
    that selects EXL3 layers, overrides formats, re-derives nothing (verbatim) and carries a REAP survivor map
  * GOLDEN: the SHA-256 of every emitted file, fixed on the builder before L279's refactor.  A refactor that moves
    a byte of either build fails here by file name -- the builder's byte-identity instrument (rule 7)
  * refusals: a format the source cannot be written in, an IQ2 row (no producer)
"""
from __future__ import annotations

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
FAILS: list[str] = []
H, V, L, E, FF = 256, 64, 3, 4, 128

# sha256 over (file name, sha256(file)) of every emitted file, in name order; taken at dev bdb62d13 (pre-L279)
GOLDEN = {
    "default": "6ee5371f8591485607e09d2fed533d3d7b0bbc217517f97b2427aec2b364cae2",
    "selected": "cbddfdbb9cfc782109934f490c61a7163f16c0eaaff36670730e44ea097d66fa",
}


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def write_st(path, tensors):
    """tensors: name -> (dtype, shape, bytes)"""
    hdr, off, blobs = {}, 0, []
    for n in sorted(tensors):
        dt, shape, b = tensors[n]
        hdr[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hj = json.dumps(hdr).encode()
    hj += b" " * (-len(hj) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj)
        for b in blobs:
            f.write(b)


def bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def make_hf(root, rng):
    cfg = {"model_type": "deepseek_v4", "num_hidden_layers": L, "num_nextn_predict_layers": 1, "num_hash_layers": 1,
           "vision_n_layers": 1, "hidden_size": H, "vocab_size": V, "max_position_embeddings": 4096,
           "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128, "qk_rope_head_dim": 64,
           "q_lora_rank": 128, "o_lora_rank": 128, "o_groups": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
           "rope_scaling": {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 256,
                            "beta_fast": 32, "beta_slow": 1},
           "compress_ratios": [1, 4, 128, 1], "compress_rope_theta": 160000.0, "num_experts_per_tok": 2,
           "moe_intermediate_size": FF, "n_routed_experts": E, "n_shared_experts": 1, "routed_scaling_factor": 1.5,
           "norm_topk_prob": True, "swiglu_limit": 10.0, "sliding_window": 128, "index_n_heads": 4,
           "index_head_dim": 128, "index_topk": 64, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
           "dspark_target_layer_ids": [0, 1, 2], "engram_layer_ids": [1], "engram_num_embeddings": [100]}
    json.dump(cfg, open(os.path.join(root, "config.json"), "w"))
    json.dump({"top_p": 0.95, "temperature": 0.6}, open(os.path.join(root, "generation_config.json"), "w"))
    t = {}

    def put(name, dt, shape, b):
        t[name] = (dt, list(shape), b)

    def mat(name, *shape, scale=0.05):
        put(name, "BF16", shape, bf16(rng.standard_normal(shape).astype(np.float32) * scale).tobytes())

    def f32(name, *shape):
        put(name, "F32", shape, rng.standard_normal(shape).astype(np.float32).tobytes())

    def fp8(name, out, inp, block=128):
        codes = rng.integers(0, 256, out * inp, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0x3C                        # no NaN codes in a source
        put(name + ".weight", "F8_E4M3", [out, inp], codes.tobytes())
        put(name + ".scale", "F8_E8M0", [out // block, inp // block],
            rng.integers(110, 130, (out // block) * (inp // block), dtype=np.uint8).tobytes())

    def fp4_expert(name, out, inp):
        put(name + ".weight", "I8", [out, inp // 2], rng.integers(-128, 128, out * inp // 2, dtype=np.int8).tobytes())
        put(name + ".scale", "F8_E8M0", [out, inp // 32],
            rng.integers(110, 130, out * inp // 32, dtype=np.uint8).tobytes())

    def block(p, mtp):
        mat(p + "attn_norm.weight", H); mat(p + "ffn_norm.weight", H)
        for hc in ("attn", "ffn"):
            f32(p + f"hc_{hc}_base", 24); f32(p + f"hc_{hc}_fn", 24, 4 * H); f32(p + f"hc_{hc}_scale", 3)
        f32(p + "attn.attn_sink", 2)
        fp8(p + "attn.wq_a", 128, H); fp8(p + "attn.wq_b", H, 128); mat(p + "attn.q_norm.weight", 128)
        fp8(p + "attn.wkv", 128, H); mat(p + "attn.kv_norm.weight", 128)
        fp8(p + "attn.wo_a", 128, H); fp8(p + "attn.wo_b", H, 128)
        mat(p + "ffn.gate.weight", E, H); f32(p + "ffn.gate.bias", E)
        fp8(p + "ffn.shared_experts.w1", FF, H); fp8(p + "ffn.shared_experts.w3", FF, H)
        fp8(p + "ffn.shared_experts.w2", H, FF)
        for e in range(E):
            fp4_expert(p + f"ffn.experts.{e}.w1", FF, H); fp4_expert(p + f"ffn.experts.{e}.w3", FF, H)
            fp4_expert(p + f"ffn.experts.{e}.w2", H, FF)
        if mtp:
            f32(p + "ffn.gate.bias_vl", E)                                   # drop rule 2

    for i in range(L):
        p = f"layers.{i}."
        block(p, False)
        if i == 0:                                                           # the hash layer
            put(p + "ffn.gate.tid2eid", "I64", [V, 2], rng.integers(0, E, V * 2, dtype=np.int64).tobytes())
            f32(p + "ffn.gate.bias_vl", E)
        if i == 1:
            fp8(p + "attn.compressor.wkv", 128, H, block=32)                 # a 32x32-block FP8 source
            f32(p + "attn.compressor.ape", 4, 128); mat(p + "attn.compressor.norm.weight", 128)
            fp8(p + "engram.wkv", 128, H); mat(p + "engram.q_weight", 4, H); mat(p + "engram.k_weight", 4, H)
            mat(p + "engram.embed.weight", 100, 32)                          # drop rule 3 (a side artifact)
    p = "mtp.0."
    block(p, True)
    fp8(p + "main_proj", H, 2 * H); mat(p + "main_norm.weight", H); mat(p + "norm.weight", H)
    mat(p + "markov_head.markov_w1.weight", V, 32); mat(p + "markov_head.markov_w2.weight", 64, V)
    mat(p + "confidence_head.proj.weight", 1, H + 4)
    f32(p + "hc_head_base", 4); f32(p + "hc_head_fn", 4, 4 * H); f32(p + "hc_head_scale", 1)
    mat("embed.weight", V, H); mat("norm.weight", H); mat("head.weight", V, H)
    f32("hc_head_base", 4); f32("hc_head_fn", 4, 4 * H); f32("hc_head_scale", 1)
    mat("vision.blocks.0.attn.qkv.weight", 96, 32); mat("aligner.proj.weight", H, 32)
    mat("image_start", H); mat("image_end", H)
    names = sorted(t)
    third = len(names) // 3
    wm = {}
    for k, part in enumerate((names[:third], names[third:2 * third], names[2 * third:])):
        fn = f"model-{k + 1:05d}-of-00003.safetensors"
        write_st(os.path.join(root, fn), {n: t[n] for n in part})
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    vocab = {f"t{i}": i for i in range(V - 4)}
    added = [{"id": V - 4 + i, "content": c, "special": True} for i, c in
             enumerate(["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<think>", "</think>"])]
    json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": ["t1 t2", "t3 t4"]}, "added_tokens": added},
              open(os.path.join(root, "tokenizer.json"), "w"))
    json.dump({"bos_token": "<｜begin▁of▁sentence｜>", "eos_token": "<｜end▁of▁sentence｜>",
               "pad_token": "<｜end▁of▁sentence｜>", "add_bos_token": True},
              open(os.path.join(root, "tokenizer_config.json"), "w"))
    return t


def make_exl3(root, rng, words):
    """layers.L.ffn.experts.E.wP.{trellis,suh,svh,mul1} for each (layer, part) in `words`."""
    t = {}
    for (layer, part), w in words.items():
        k, n = (FF, H) if part == "w2" else (H, FF)
        for e in range(E):
            key = f"layers.{layer}.ffn.experts.{e}.{part}"
            t[key + ".trellis"] = ("I16", [k // 16, n // 16, w],
                                   rng.integers(-32768, 32767, (k // 16) * (n // 16) * w, dtype=np.int16).tobytes())
            t[key + ".suh"] = ("F16", [k], rng.standard_normal(k).astype(np.float16).tobytes())
            t[key + ".svh"] = ("F16", [n], rng.standard_normal(n).astype(np.float16).tobytes())
            t[key + ".mul1"] = ("I32", [], struct.pack("<I", 0x83DCD12D))
    os.makedirs(root, exist_ok=True)
    write_st(os.path.join(root, "model-00001-of-00001.safetensors"), t)


def run(*args, expect_fail=False, contains=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py")] + list(args), capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if contains is not None:
        ok = ok and contains in out
    return ok, out


def digest(d):
    """sha256 over (name, sha256(bytes)) of every file in `d`, in name order."""
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        h.update(f.encode() + b"\0" + hashlib.sha256(open(os.path.join(d, f), "rb").read()).digest())
    return h.hexdigest()


def build_and_grade(tmp, label, args):
    print(f"{label}:")
    ok, out = run("plan", *args)
    check(ok, "plan")
    if not ok:
        print(out)
        return
    print("    " + "\n    ".join(out.strip().splitlines()[-2:]))
    o1, o2 = os.path.join(tmp, label + "-1"), os.path.join(tmp, label + "-2")
    ok1, out1 = run("emit", *args, "--out", o1, "--all")
    ok2, _ = run("emit", *args, "--out", o2, "--all")
    check(ok1 and ok2, "emit --all twice")
    if not ok1:
        print(out1)
        return
    check(digest(o1) == digest(o2), f"two emits byte-identical ({len(os.listdir(o1))} files)")
    ok, out = run("verify", *args, "--out", o1, "--all")
    check(ok and "failing: 0" in out, "verify --all PASS")
    ok, out = run("audit", "--out", o1)
    check(ok and "AUDIT PASS" in out, "audit PASS")
    got = digest(o1)
    check(got == GOLDEN[label], f"GOLDEN {label}: {got}")
    if got != GOLDEN[label]:
        for f in sorted(os.listdir(o1)):
            print(f"      {hashlib.sha256(open(os.path.join(o1, f), 'rb').read()).hexdigest()}  {f}")
    return o1


def main():
    tmp = tempfile.mkdtemp(prefix="deepseek-builder-test-")
    try:
        hf = os.path.join(tmp, "hf")
        os.makedirs(hf)
        make_hf(hf, np.random.default_rng(279))
        ex = os.path.join(tmp, "exl3")
        make_exl3(ex, np.random.default_rng(280), {(1, "w1"): 32, (1, "w3"): 32, (1, "w2"): 32,
                                                   (2, "w1"): 40, (2, "w3"): 40, (2, "w2"): 48})
        o = build_and_grade(tmp, "default", ["--hf", hf, "--exl3", ex])
        if o:
            with open(os.path.join(o, "model-00002-of-00006.safetensors"), "rb") as f:     # layers.0
                (n,) = struct.unpack("<Q", f.read(8))
                h = json.loads(f.read(n))
            tens = json.loads(h["__metadata__"]["pulsar.tensors"])
            check(tens["layers.0.ffn.gate.tid2eid"]["layout"] == "i32" and "layers.0.ffn.gate.bias" not in h
                  and h["__metadata__"]["pulsar.expert_dtype"] == "mxfp4_cutlass", "layers.0: tid2eid i32, hash-layer bias dropped, FP4 experts")
            with open(os.path.join(o, "model-00004-of-00006.safetensors"), "rb") as f:     # layers.2
                (n,) = struct.unpack("<Q", f.read(8))
                h = json.loads(f.read(n))
            fams = {x["part"]: x["layout"] for x in json.loads(h["__metadata__"]["pulsar.experts"])}
            check(fams == {"w1": "exl3m_k2h", "w3": "exl3m_k2h", "w2": "exl3m_k3"}, f"layers.2 families {fams}")

        fm = os.path.join(tmp, "format-map.json")
        json.dump({"blk.0.ffn_gate_exps.weight": "CUTLASS_MXFP4", "blk.0.attn_q_a.weight": "MXFP8_LT",
                   "dspark.0.ffn_down_exps.weight": "CUTLASS_MXFP4"}, open(fm, "w"))
        reap = os.path.join(tmp, "reap.json")
        json.dump({"layout": "per-layer", "expert_count": [E] * L, "keep_count": [E] * L, "policy": [0] * L,
                   "survivors": [list(range(E))] * L}, open(reap, "w"))
        build_and_grade(tmp, "selected", ["--hf", hf, "--exl3", ex, "--exl3-layers", "2", "--format-map", fm,
                                          "--mxfp8-scale", "verbatim", "--reap-map", reap])

        print("refusals:")
        bad = os.path.join(tmp, "bad-map.json")
        json.dump({"blk.0.attn_norm.weight": "MXFP8_LT"}, open(bad, "w"))
        ok, _ = run("plan", "--hf", hf, "--format-map", bad, expect_fail=True, contains="is written as bf16")
        check(ok, "a bf16 source asked for mxfp8_lt -> refused")
        json.dump({"blk.0.ffn_down_exps.weight": "IQ2_XXS_MMQ"}, open(bad, "w"))
        ok, _ = run("plan", "--hf", hf, "--format-map", bad, expect_fail=True, contains="does not produce")
        check(ok, "an IQ2 expert row -> refused (no producer)")
    finally:
        shutil.rmtree(tmp)
    print(f"test_deepseek: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

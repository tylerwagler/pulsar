#!/usr/bin/env python3
"""The qwen4_exp family of the direct builder (L251 S6), end to end on a SYNTHETIC miniature checkpoint -- no mounts,
no GPU, seconds.  The real checkpoint's names and dtypes at toy shapes (hidden 256, 4 layers, 4 experts, 2 PLE parts),
an exllamav3-layout EXL3 checkpoint of random trellis / suh / svh, a toy BPE tokenizer.  What it pins:

  * mxfp8_lt_from_bf16 vs a scalar reference quantiser (per-32 amax -> E8M0 = floor(log2 amax) - 7, E4M3 RNE) on
    random matrices with zeros, subnormal-range groups and the non-128 shapes Qwen has ([48, 256], [64, 1024],
    [1024, 64]); the decode round trip within E4M3's rounding bound; the swizzled plane padded as bytes_for says
  * the recipe: a tensor no row names, a tensor two rows name, a row that names nothing -> each refused by name
  * plan -> emit twice -> byte-identical shards (the L247 standard) -> verify --roundtrip PASS -> audit PASS
  * EXL3 entries byte-identical to the source's [trellis | suh | svh]; expert families contiguous with entry_name
  * refusals: an EXL3 rate the recipe does not name, a missing EXL3 tensor, a wrong mul1 codebook, a DeepSeek
    option on a Qwen build, a PLE manifest whose head table is not the checkpoint's
  * the PLE row file: build -> verify (sampled + --full) PASS; a flipped byte in the file -> verify FAILS
  * split gate/up (turboderp's form) -> gate_proj / up_proj families when the recipe says split
  * GOLDEN: the SHA-256 of every emitted file of the fused and the split build, fixed on the builder before L279's
    refactor (the byte-identity instrument, rule 7; test_deepseek.py holds the DeepSeek pair)
"""
from __future__ import annotations

import copy
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
import producers as P  # noqa: E402
from test_deepseek import digest  # noqa: E402

FAILS: list[str] = []
# sha256 over (file name, sha256(file)) of every emitted file, in name order (test_deepseek.digest); taken at dev
# bdb62d13 (pre-L279)
GOLDEN = {
    "fused": "a4d6fc36f9d898f229f188d6d2f76bf77e8d55a62da9c2d03df85522837e171b",
    "split": "d818bfd9a3ba47de09a8c1b837be41ed9effa401d89286d277d09667d186cf1a",
}
PFX = "model.language_model."
H, L, E, FF, HC, LR = 256, 4, 4, 128, 4, 64


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


# --------------------------------------------------------------------------- the synthetic checkpoint
def bf16(a):
    f = np.ascontiguousarray(a, dtype=np.float32)
    u = f.view(np.uint32)
    r = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
    return r


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


def make_hf(root, rng):
    types = ["linear_attention"] * 3 + ["full_attention"]
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
           "vision_config": {"depth": 1}, "text_config": {
               "model_type": "qwen4_exp_text", "num_hidden_layers": L, "hidden_size": H, "vocab_size": 64,
               "layer_types": types, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
               "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128, "output_gate_type": "sigmoid",
               "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                   "mrope_interleaved": True, "rope_type": "default"},
               "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_compress_ratio": 4,
               "indexer_budget": 2048, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
               "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
               "mamba_ssm_dtype": "float32", "hc_count": HC, "hc_lowrank": LR, "num_experts": E,
               "num_experts_per_tok": 2, "moe_intermediate_size": FF, "shared_expert_intermediate_size": FF,
               "ple_layer_ids": [2], "ple_embed_dim": H, "ple_conv_kernel_size": 4, "ngram_size": 3,
               "heads_per_ngram": 2, "split_ngram_parts": 2, "eos_token_id": 5}}
    json.dump(cfg, open(os.path.join(root, "config.json"), "w"))
    json.dump({"top_p": 0.95, "top_k": 20, "temperature": 1.0}, open(os.path.join(root, "generation_config.json"), "w"))
    t = {}

    def mat(name, *shape, scale=0.05):
        a = rng.standard_normal(shape).astype(np.float32) * scale
        t[name] = ("BF16", shape, bf16(a).tobytes())

    def i64(name, vals):
        t[name] = ("I64", [len(vals)], struct.pack(f"<{len(vals)}q", *vals))
    mat("lm_head.weight", 64, H)
    mat(PFX + "embed_tokens.weight", 64, H)
    for n in ("hc_norm.weight",):
        mat(PFX + "hyper_connection_mixer." + n, HC * H)
    mat(PFX + "hyper_connection_mixer.input_mix_weight_down.weight", LR, HC * H)
    mat(PFX + "hyper_connection_mixer.input_mix_weight_up.weight", HC * H, LR)
    for i, ty in enumerate(types):
        p = f"{PFX}layers.{i}."
        for hc in ("attn_hyper_connection.", "mlp_hyper_connection."):
            mat(p + hc + "block_inject_weight.weight", HC, HC * H)
            mat(p + hc + "hc_norm.weight", HC * H)
            mat(p + hc + "input_mix_weight_down.weight", LR, HC * H)
            mat(p + hc + "input_mix_weight_up.weight", HC * H, LR)
        if ty == "linear_attention":
            q = p + "linear_attn."
            mat(q + "A_log", 2); mat(q + "dt_bias", 2); mat(q + "norm.weight", 128)
            mat(q + "conv1d.weight", 512, 1, 4)
            mat(q + "in_proj_a.weight", 48, H); mat(q + "in_proj_b.weight", 48, H)
            mat(q + "in_proj_qkv.weight", 512, H); mat(q + "in_proj_z.weight", 256, H); mat(q + "out_proj.weight", H, 256)
        else:
            q = p + "self_attn."
            mat(q + "q_proj.weight", 512, H); mat(q + "k_proj.weight", 128, H); mat(q + "v_proj.weight", 128, H)
            mat(q + "o_proj.weight", H, 256); mat(q + "q_norm.weight", 128); mat(q + "k_norm.weight", 128)
            mat(q + "indexer.index_qk_proj.weight", 640, H)
            mat(q + "indexer.q_layernorm.weight", 128); mat(q + "indexer.k_layernorm.weight", 128)
        m = p + "mlp."
        mat(m + "gate.weight", E, H)
        mat(m + "experts.gate_up_proj", E, 2 * FF, H)
        mat(m + "experts.down_proj", E, H, FF)
        mat(m + "shared_expert.gate_proj.weight", FF, H); mat(m + "shared_expert.up_proj.weight", FF, H)
        mat(m + "shared_expert.down_proj.weight", H, FF); mat(m + "shared_expert_gate.weight", 1, H)
        if i == 1:
            e = p + "ple."
            mat(e + "key_proj.weight", HC * H, H); mat(e + "value_proj.weight", H, H)
            mat(e + "conv1d.weight", HC * H, 1, 4)
            for n in ("norm_conv", "norm_key", "norm_query"):
                mat(e + n + ".weight", HC * H)
            sizes = [101, 103, 107, 109]
            i64(e + "ple_embedding.ngram_heads_vocab_sizes", sizes)
            i64(e + "ple_embedding.ngram_heads_offsets", [0, 101, 204, 311])
            i64(e + "ple_embedding.layer_multipliers", [2 ** 40 + 1, 3 ** 20, 7])
            for s in range(2):
                mat(e + f"ple_embedding.ngram_embedding.shard_{s}.weight", 256, H // 4)   # 512 rows >= 420
    mat("model.visual.blocks.0.attn.qkv.weight", 96, 32)
    mat("model.visual.merger.norm.weight", 32)
    mat("mtp.fc_embedding.weight", H, H)
    names = sorted(t)
    # two shards, an index
    half = len(names) // 2
    wm = {}
    for k, part in enumerate((names[:half], names[half:])):
        fn = f"model-{k + 1:05d}-of-00002.safetensors"
        write_st(os.path.join(root, fn), {n: t[n] for n in part})
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    # a toy BPE tokenizer
    vocab = {f"t{i}": i for i in range(40)}
    added = [{"id": 40 + i, "content": c, "special": sp} for i, (c, sp) in
             enumerate([("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True), ("<think>", False)])]
    json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": ["t1 t2", "t3 t4"]}, "added_tokens": added},
              open(os.path.join(root, "tokenizer.json"), "w"))
    json.dump({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None, "add_bos_token": False,
               "pretokenize_regex": "\\s+"}, open(os.path.join(root, "tokenizer_config.json"), "w"))
    open(os.path.join(root, "chat_template.jinja"), "w").write("{{ messages }}")
    return t


def exl3_parts(rng, k, n, K, mul1=0x83DCD12D):
    return {"trellis": ("I16", [k // 16, n // 16, 16 * K], rng.integers(-32768, 32767, (k // 16) * (n // 16) * 16 * K,
                                                                        dtype=np.int16).tobytes()),
            "suh": ("F16", [k], rng.standard_normal(k).astype(np.float16).tobytes()),
            "svh": ("F16", [n], rng.standard_normal(n).astype(np.float16).tobytes()),
            "mul1": ("I32", [], struct.pack("<I", mul1))}


def make_exl3(root, hf_t, rng, fused=True, dense_K=5, expert_K=4, drop=None, bad_mul1=None):
    t = {}
    dense = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj",
             "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "shared_expert.gate_proj",
             "shared_expert.up_proj", "shared_expert.down_proj", "ple.key_proj", "ple.value_proj")
    for name, (dt, shape, _) in hf_t.items():
        if name.endswith(".weight") and any(name.endswith(d + ".weight") for d in dense):
            out, inp = shape
            base = name[:-len(".weight")]
            for s, v in exl3_parts(rng, inp, out, dense_K, bad_mul1 if base == drop and bad_mul1 else 0x83DCD12D).items():
                t[f"{base}.{s}"] = v
        elif name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
            pre = name[:name.rindex(".") + 1]
            Ee, out, inp = shape
            stack = name[name.rindex(".") + 1:]
            parts = [(stack, out)] if (fused or stack == "down_proj") else [("gate_proj", out // 2), ("up_proj", out // 2)]
            for e in range(Ee):
                for part, n in parts:
                    for s, v in exl3_parts(rng, inp, n, expert_K).items():
                        t[f"{pre}{e}.{part}.{s}"] = v
    if drop and not bad_mul1:
        for s in ("trellis", "suh", "svh", "mul1"):
            t.pop(f"{drop}.{s}", None)
    os.makedirs(root, exist_ok=True)
    write_st(os.path.join(root, "model-L00.safetensors"), t)
    return t


# --------------------------------------------------------------------------- helpers
def run(*args, expect_fail=False, contains=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, args[0])] + list(args[1:]), capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if contains is not None:
        ok = ok and contains in out
    return ok, out


def scalar_mxfp8(w):
    """Independent reference: per 32-group along `in`, s = floor(log2 amax) - 7 (clamped to E8M0's range), then each
    |w| / 2^s to the NEAREST positive E4M3 magnitude by exhaustive search over codes 0x00..0x7E, ties to the even
    code; sign bit = (w < 0), as the codec writes it (a negative value that rounds to zero keeps 0x80)."""
    mag = P.E4M3_VALUE[:0x7F].astype(np.float64)                     # codes 0x00..0x7E, ascending
    out, inp = w.shape
    g = w.astype(np.float64).reshape(out, inp // 32, 32)
    amax = np.abs(g).max(axis=2)
    s = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 7, -127)
    s = np.clip(s, -127, 127).astype(np.int32)
    a = np.abs(g) / np.ldexp(1.0, s)[:, :, None]
    d = np.abs(a[..., None] - mag)                                   # [out, kb, 32, 127]
    best = d.min(axis=-1, keepdims=True)
    cand = d == best
    even = cand & ((np.arange(0x7F) & 1) == 0)
    idx = np.where(even.any(-1), even.argmax(-1), cand.argmax(-1))
    codes = (idx | np.where(g < 0, 0x80, 0)).astype(np.uint8).reshape(out, inp)
    return codes, (s + 127).astype(np.uint8)


def main():
    rng = np.random.default_rng(251)
    print("mxfp8_lt_from_bf16:")
    for (o, i) in ((48, 256), (64, 1024), (1024, 64), (128, 128)):
        a = rng.standard_normal((o, i)).astype(np.float32) * 0.02
        a[0, :32] = 0.0                                   # an all-zero group
        a[1, :32] *= 1e-6                                  # a tiny group
        a[2, 5] = 3.0                                      # an outlier group (most elements underflow to subnormals)
        wb = bf16(a)
        w = (wb.astype(np.uint32) << 16).view(np.float32).reshape(o, i)
        buf = P.mxfp8_lt_from_bf16(wb.tobytes(), o, i)
        check(len(buf) == P.bytes_for("mxfp8_lt", [i, o]), f"[{o}, {i}] {len(buf)} bytes == bytes_for")
        if o * i <= 48 * 256:
            codes, scales = scalar_mxfp8(w)
            got_codes = np.frombuffer(buf, np.uint8, count=o * i).reshape(o, i)
            got_sc = P.unswizzle_sf(np.frombuffer(buf, np.uint8, offset=o * i), o, i // 32)
            check(np.array_equal(got_codes, codes) and np.array_equal(got_sc, scales),
                  f"[{o}, {i}] codes + scales == the scalar reference")
        q = P.mxfp8_lt_decode(buf, o, i)
        s = np.repeat(P.unswizzle_sf(np.frombuffer(buf, np.uint8, offset=o * i), o, i // 32).astype(np.int32) - 127, 32, 1)
        x = np.abs(np.ldexp(w, -s))
        e = np.floor(np.log2(np.where(x > 0, x, 1.0))).astype(np.int32)
        bound = np.where(x >= 2.0 ** -6, np.ldexp(1.0, e - 4 + s), np.ldexp(1.0, s - 10))
        check(int((np.abs(q.astype(np.float64) - w) > bound).sum()) == 0, f"[{o}, {i}] decode within the E4M3 bound")
        sf = np.frombuffer(buf, np.uint8, offset=o * i)
        pad = P.swizzle_sf(np.zeros((o, i // 32), np.uint8) + 1, o, i // 32)
        check(bool(np.all(sf[pad == 0] == 0)), f"[{o}, {i}] swizzle padding is zero")

    tmp = tempfile.mkdtemp(prefix="qwen-builder-test-")
    try:
        hf_dir = os.path.join(tmp, "hf")
        os.makedirs(hf_dir)
        hf_t = make_hf(hf_dir, rng)
        ex = os.path.join(tmp, "exl3")
        make_exl3(ex, hf_t, np.random.default_rng(7))
        recipe = os.path.join(HERE, "format-maps", "qwen38fn-u-e4-d5.json")
        ple_out = os.path.join(tmp, "ple")

        print("PLE row file:")
        ok, out = run("ple_rows.py", "build", "--hf", hf_dir, "--out", ple_out)
        check(ok, "ple_rows build")
        ok, out = run("ple_rows.py", "verify", "--hf", hf_dir, "--out", ple_out, "--sample", "300", "--full")
        check(ok and "VERIFY PASS" in out, "ple_rows verify --full PASS")
        rows_p = os.path.join(ple_out, "ple-l1.rows")
        blob = bytearray(open(rows_p, "rb").read())
        # row 0 of shard 1 == the source's first row of shard_1
        src = hf_t[PFX + "layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight"][2]
        check(bytes(blob[64 + 256 * 128: 64 + 257 * 128]) == src[:128], "global row 256 == shard_1 row 0")
        blob[64 + 1000] ^= 1
        bad = os.path.join(tmp, "ple-bad")
        os.makedirs(bad)
        open(os.path.join(bad, "ple-l1.rows"), "wb").write(bytes(blob))
        shutil.copy(os.path.join(ple_out, "ple-l1.json"), bad)
        ok, out = run("ple_rows.py", "verify", "--hf", hf_dir, "--out", bad, "--sample", "10", "--full", expect_fail=True)
        check(ok and "FAIL" in out, "a flipped byte in the row file -> verify FAILS")
        man = os.path.join(ple_out, "ple-l1.json")

        print("build / emit / verify:")
        common = ["--hf", hf_dir, "--exl3", ex, "--recipe", recipe, "--ple-rows", man]
        ok, out = run("build.py", "plan", *common)
        check(ok, "plan")
        print("    " + "\n    ".join(out.strip().splitlines()[-3:]))
        o1, o2 = os.path.join(tmp, "out1"), os.path.join(tmp, "out2")
        ok1, _ = run("build.py", "emit", *common, "--out", o1, "--all")
        ok2, _ = run("build.py", "emit", *common, "--out", o2, "--all")
        check(ok1 and ok2, "emit --all twice")
        files = sorted(os.listdir(o1))
        same = files == sorted(os.listdir(o2)) and all(
            open(os.path.join(o1, f), "rb").read() == open(os.path.join(o2, f), "rb").read() for f in files)
        check(same and len(files) == L + 3, f"two emits byte-identical ({len(files)} files)")
        ok, out = run("build.py", "verify", *common, "--out", o1, "--all", "--roundtrip")
        check(ok and "failing: 0" in out and "roundtrip:" in out, "verify --all --roundtrip PASS")
        ok, out = run("build.py", "audit", "--out", o1)
        check(ok and "AUDIT PASS" in out, "audit PASS")
        got = digest(o1)
        check(got == GOLDEN["fused"], f"GOLDEN fused: {got}")
        # the instrument must see a single flipped payload byte (mutation: a green verify is not vacuous)
        victim = os.path.join(o2, "model-00002-of-00006.safetensors")
        vb = bytearray(open(victim, "rb").read())
        vb[-1000] ^= 0x10
        open(victim, "wb").write(bytes(vb))
        ok, out = run("build.py", "verify", *common, "--out", o2, "--shard", "layers.0", expect_fail=True,
                      contains="BYTES DIFFER")
        check(ok, "one flipped payload byte in an emitted shard -> verify FAILS by name")
        # the container's own view: kv + families
        with open(os.path.join(o1, "model-00001-of-00006.safetensors"), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(n))["__metadata__"]
        kv = {e["key"]: e["value"] for e in json.loads(meta["pulsar.kv"])}
        check(kv["general.architecture"] == "qwen4_exp" and kv["qwen4_exp.num_hidden_layers"] == L
              and kv["qwen4_exp.ple_layer_ids"]["v"] == [2] and kv["pulsar.ple_rows.layer"] == 1
              and kv["qwen4_exp.rope_parameters.mrope_section"]["v"] == [11, 11, 10]
              and kv["qwen4_exp.ple_ngram_heads_offsets"]["v"] == [0, 101, 204, 311]
              and kv["qwen4_exp.ple_layer_multipliers"]["__array__"] == "u64"
              and kv["pulsar.mtp_present"] is False
              and kv["qwen4_exp.layer_types"]["v"][3] == "full_attention",
              "pulsar.kv: the family, text_config verbatim (flattened), the PLE buffers as u64, the row-file facts")
        with open(os.path.join(o1, "model-00003-of-00006.safetensors"), "rb") as f:   # layers.1
            (n,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(n))
        fams = json.loads(h["__metadata__"]["pulsar.experts"])
        check(sorted(x["part"] for x in fams) == ["down_proj", "gate_up_proj"]
              and all(x["layout"] == "exl3m_k4" and x["layer"] == 1 for x in fams), "layers.1 families: fused gate_up + down, K4")
        tens = json.loads(h["__metadata__"]["pulsar.tensors"])
        check(tens[PFX + "layers.1.linear_attn.in_proj_qkv.weight"]["layout"] == "exl3m_k5"
              and tens[PFX + "layers.1.attn_hyper_connection.input_mix_weight_down.weight"]["layout"] == "mxfp8_lt"
              and tens[PFX + "layers.1.mlp.gate.weight"]["layout"] == "bf16", "per-tensor layouts follow the recipe")
        check(not any(k.startswith("mtp.") or "ngram_embedding" in k for k in h), "no MTP / n-gram rows in the container")

        print("refusals:")
        bad_ex = os.path.join(tmp, "exl3-k3")
        make_exl3(bad_ex, hf_t, np.random.default_rng(7), dense_K=3)
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", bad_ex, "--recipe", recipe, "--ple-rows", man,
                      expect_fail=True, contains="the recipe names exl3m_k5, the EXL3 source holds exl3m_k3")
        check(ok, "dense at K3 (no row names it) -> refused")
        bad_ex = os.path.join(tmp, "exl3-k5-experts")
        make_exl3(bad_ex, hf_t, np.random.default_rng(7), expert_K=5)
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", bad_ex, "--recipe", recipe, "--ple-rows", man,
                      expect_fail=True, contains="the recipe names exl3m_k4, the EXL3 source holds exl3m_k5")
        check(ok, "experts at K5 where the recipe names K4 -> refused")
        drop = PFX + "layers.0.linear_attn.out_proj"
        bad_ex = os.path.join(tmp, "exl3-missing")
        make_exl3(bad_ex, hf_t, np.random.default_rng(7), drop=drop)
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", bad_ex, "--recipe", recipe, "--ple-rows", man,
                      expect_fail=True, contains="missing from the EXL3 checkpoint")
        check(ok, "a missing EXL3 tensor -> refused by name")
        bad_ex = os.path.join(tmp, "exl3-mcg")
        make_exl3(bad_ex, hf_t, np.random.default_rng(7), drop=drop, bad_mul1=0xCBAC1FED)
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", bad_ex, "--recipe", recipe, "--ple-rows", man,
                      expect_fail=True, contains="codebook multiplier")
        check(ok, "a tensor on another codebook -> refused")
        ok, out = run("build.py", "plan", *common, "--mxfp8-scale", "verbatim", expect_fail=True, contains="no entry of this build is an FP8-sourced mxfp8_lt")
        check(ok, "--mxfp8-scale verbatim with no FP8 source (the Qwen checkpoint is BF16) -> refused")
        r = json.load(open(recipe))
        for label, rows, msg in (
                ("unnamed", [x for x in r["rows"] if x[0] != "model.visual.*"], "no row of"),
                ("doubly named", r["rows"] + [["model.visual.merger.*", "bf16"]], "rows of"),
                ("dead row", r["rows"] + [["model.language_model.nothing.*", "bf16"]], "match no checkpoint tensor")):
            rr = dict(r, rows=rows)
            rp = os.path.join(tmp, f"recipe-{label.replace(' ', '-')}.json")
            json.dump(rr, open(rp, "w"))
            ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", ex, "--recipe", rp, "--ple-rows", man,
                          expect_fail=True, contains=msg)
            check(ok, f"recipe with a {label} tensor/row -> refused")
        mm = json.load(open(man))
        mm["head_offsets"] = [0, 101, 204, 312]
        mp = os.path.join(tmp, "ple-l1.json")
        json.dump(mm, open(mp, "w"))
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", ex, "--recipe", recipe, "--ple-rows", mp,
                      expect_fail=True, contains="a different table")
        check(ok, "a PLE manifest with another head table -> refused")

        tb = os.path.join(tmp, "tessera")
        os.makedirs(tb)
        open(os.path.join(tb, PFX + "layers.0.mlp.experts.gate_up_proj.pt"), "wb").write(b"\0")
        rt = dict(r, rows=[[p, "tessera"] if p.endswith("mlp.experts.gate_up_proj") else [p, f] for p, f in r["rows"]])
        rp = os.path.join(tmp, "recipe-tessera.json")
        json.dump(rt, open(rp, "w"))
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", ex, "--tessera", tb, "--recipe", rp, "--ple-rows",
                      man, expect_fail=True, contains="the tessera producer lands with L255")
        check(ok, "routed experts as Tessera planes -> refused by name (the producer is L255's)")
        json.dump(dict(r, model_type="deepseek_v4"), open(rp, "w"))
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", ex, "--recipe", rp, "--ple-rows", man,
                      expect_fail=True, contains="a recipe for 'deepseek_v4'")
        check(ok, "a recipe for another model_type -> refused")

        print("split gate/up (turboderp's layout):")
        ex_s = os.path.join(tmp, "exl3-split")
        make_exl3(ex_s, hf_t, np.random.default_rng(9), fused=False)
        rs = dict(r, expert_gate_up="split")
        rp = os.path.join(tmp, "recipe-split.json")
        json.dump(rs, open(rp, "w"))
        o3 = os.path.join(tmp, "out3")
        ok, out = run("build.py", "emit", "--hf", hf_dir, "--exl3", ex_s, "--recipe", rp, "--ple-rows", man,
                      "--out", o3, "--all")
        check(ok, "emit with split gate/up")
        ok, out = run("build.py", "verify", "--hf", hf_dir, "--exl3", ex_s, "--recipe", rp, "--ple-rows", man,
                      "--out", o3, "--all")
        check(ok and "failing: 0" in out, "verify PASS (split)")
        got = digest(o3)
        check(got == GOLDEN["split"], f"GOLDEN split: {got}")
        with open(os.path.join(o3, "model-00003-of-00006.safetensors"), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(n))
        fams = json.loads(h["__metadata__"]["pulsar.experts"])
        check(sorted(x["part"] for x in fams) == ["down_proj", "gate_proj", "up_proj"], "families gate_proj / up_proj / down_proj")
        ok, out = run("build.py", "plan", "--hf", hf_dir, "--exl3", ex, "--recipe", rp, "--ple-rows", man,
                      expect_fail=True, contains="missing from the EXL3 checkpoint")
        check(ok, "recipe says split, the source is fused -> refused")
    finally:
        shutil.rmtree(tmp)
    print(f"test_qwen: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

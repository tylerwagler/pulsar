#!/usr/bin/env python3
"""test_gemma4.py -- Gemma 4 in the artifact builder (gemma4.py, L338), on a synthetic checkpoint with the real layout
(text_config nested under model_type gemma4, `model.language_model.` prefix, a sliding and a global layer -- K = V, no
v_proj --, the dense MLP beside the fused expert stacks, the router's scales, the per-layer scalar, vision tensors)
and its synthetic exllamav3 pack (every Linear under the checkpoint's names, experts split per expert, every width
padded to 128 with svh 0 on the gate / up pad, as exllamav3 writes it):

  * EXL3 K4 (the head K6, the pack's): plan, emit --all twice byte-identical, verify --all (side files byte-equal), audit; GOLDEN
  * the records: every bind role per layer (block layers, layer N), no attn_v on the global layer, embed /
    output_norm at the top; the dense MLP declared at the stored width ([H -> 256] for a 200-wide MLP); the expert
    families gate / up / down at the stored width (128 for a 100-wide expert), role, layer, block; every Linear's bytes
    == the pack's [trellis | suh | svh]; pulsar.kv: text_config verbatim (both ropes flattened, layer_types)
  * refusals by name: an unknown tensor, a missing tensor, a wrong shape, a switch (attention_k_eq_v false), the
    global rope (partial factor 0.5), a nonzero svh on a padded gate column (dense MLP and expert), a dense-MLP row
    written bf16, the head written bf16, no --exl3
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
sys.path.insert(0, HERE)
from test_qwen import bf16, write_st, exl3_parts  # noqa: E402

FAILS: list[str] = []
# sha256 over (file name, sha256(file)) of every emitted file but the side files; fixed when the family landed (L338)
GOLDEN = {
    "k4": "82672c0483a73b8d0cbf649419beb3e06a6fed774eebdffbb7779677d85e7d9d",
}
H, V, NQ, FF, E, FE = 128, 64, 2, 200, 4, 100
SLIDING, FULL = (1, 128), (1, 256)          # (n_kv, head_dim)
TYPES = ["sliding_attention", "full_attention"]
L = len(TYPES)
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")
P = "model.language_model."


def stored(n):
    return (n + 127) // 128 * 128


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def config():
    text = {"model_type": "gemma4_text", "hidden_activation": "gelu_pytorch_tanh", "attention_k_eq_v": True,
            "enable_moe_block": True, "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False, "tie_word_embeddings": True, "attention_bias": False,
            "use_bidirectional_attention": "vision", "final_logit_softcapping": 30.0, "hidden_size": H,
            "vocab_size": V, "num_hidden_layers": L, "layer_types": list(TYPES), "sliding_window": 16,
            "num_attention_heads": NQ, "num_key_value_heads": SLIDING[0], "head_dim": SLIDING[1],
            "num_global_key_value_heads": FULL[0], "global_head_dim": FULL[1], "intermediate_size": FF,
            "num_experts": E, "top_k_experts": 2, "moe_intermediate_size": FE, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
            "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                                   "rope_type": "proportional"},
                                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}
    return {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": text,
            "vision_config": {"model_type": "gemma4_vision", "hidden_size": 32}}


def make_hf(root, rng, cfg=None, edit=None):
    os.makedirs(root, exist_ok=True)
    json.dump(cfg or config(), open(os.path.join(root, "config.json"), "w"))
    for f in SIDE[:-1]:
        open(os.path.join(root, f), "w").write(f"{{\"synthetic\": \"{f}\"}}\n")
    t = {}

    def mat(name, *shape):
        a = rng.standard_normal(shape).astype(np.float32) * 0.05
        t[name] = ("BF16", list(shape), bf16(a).tobytes())

    for il, typ in enumerate(TYPES):
        q = f"{P}layers.{il}."
        nkv, hd = FULL if typ == "full_attention" else SLIDING
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm_1", "pre_feedforward_layernorm_2", "post_feedforward_layernorm_2",
                  "post_feedforward_layernorm"):
            mat(q + n + ".weight", H)
        mat(q + "self_attn.q_proj.weight", NQ * hd, H); mat(q + "self_attn.k_proj.weight", nkv * hd, H)
        if typ == "sliding_attention":
            mat(q + "self_attn.v_proj.weight", nkv * hd, H)
        mat(q + "self_attn.o_proj.weight", H, NQ * hd)
        mat(q + "self_attn.q_norm.weight", hd); mat(q + "self_attn.k_norm.weight", hd)
        mat(q + "mlp.gate_proj.weight", FF, H); mat(q + "mlp.up_proj.weight", FF, H); mat(q + "mlp.down_proj.weight", H, FF)
        mat(q + "router.proj.weight", E, H); mat(q + "router.scale", H); mat(q + "router.per_expert_scale", E)
        mat(q + "experts.gate_up_proj", E, 2 * FE, H); mat(q + "experts.down_proj", E, H, FE)
        mat(q + "layer_scalar", 1)
    mat(P + "embed_tokens.weight", V, H); mat(P + "norm.weight", H)
    mat("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", 32, 32)
    mat("model.embed_vision.embedding_projection.weight", H, 32)
    if edit:
        edit(t)
    names = sorted(t)
    half = len(names) // 2
    wm = {}
    for fn, part in (("model-00001-of-00002.safetensors", names[:half]), ("model-00002-of-00002.safetensors", names[half:])):
        write_st(os.path.join(root, fn), {n: t[n] for n in part})
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    return t


def pack_linear(t, rng, key, k, n, n_logical=None, K=4, dirty_pad=False):
    """One exllamav3 Linear [k -> n] (both 128-multiples); svh zero on [n_logical, n) as exllamav3 pads gate / up."""
    parts = exl3_parts(rng, k, n, K)
    if n_logical is not None and n_logical < n:
        svh = np.frombuffer(parts["svh"][2], np.float16).copy()
        svh[n_logical:] = 0
        if dirty_pad:
            svh[n_logical] = np.float16(0.5)
        parts["svh"] = ("F16", [n], svh.tobytes())
    for s, v in parts.items():
        t[f"{key}.{s}"] = v


def make_exl3(root, rng, dirty=None):
    """The synthetic pack: every Linear under the checkpoint's names, experts split, widths stored (dirty: a key whose
    first padded svh column is nonzero)."""
    t = {}
    for il, typ in enumerate(TYPES):
        q = f"{P}layers.{il}."
        nkv, hd = FULL if typ == "full_attention" else SLIDING
        pack_linear(t, rng, q + "self_attn.q_proj", H, NQ * hd)
        pack_linear(t, rng, q + "self_attn.k_proj", H, nkv * hd)
        if typ == "sliding_attention":
            pack_linear(t, rng, q + "self_attn.v_proj", H, nkv * hd)
        pack_linear(t, rng, q + "self_attn.o_proj", NQ * hd, H)
        for p in ("gate_proj", "up_proj"):
            pack_linear(t, rng, q + f"mlp.{p}", H, stored(FF), FF, dirty_pad=dirty == q + f"mlp.{p}")
        pack_linear(t, rng, q + "mlp.down_proj", stored(FF), H)
        for e in range(E):
            for p in ("gate_proj", "up_proj"):
                key = q + f"experts.{e}.{p}"
                pack_linear(t, rng, key, H, stored(FE), FE, dirty_pad=dirty == key)
            pack_linear(t, rng, q + f"experts.{e}.down_proj", stored(FE), H)
    pack_linear(t, rng, "lm_head", H, stored(V), K=6)   # the head exllamav3 quantizes from the tied embedding
    os.makedirs(root, exist_ok=True)
    names = sorted(t)
    half = len(names) // 2
    write_st(os.path.join(root, "model-00001-of-00002.safetensors"), {n: t[n] for n in names[:half]})
    write_st(os.path.join(root, "model-00002-of-00002.safetensors"), {n: t[n] for n in names[half:]})
    return t


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


def entry_bytes(d, f, name):
    h, base = header(os.path.join(d, f))
    o0, o1 = h[name]["data_offsets"]
    with open(os.path.join(d, f), "rb") as g:
        g.seek(base + o0)
        return g.read(o1 - o0)


def shard_digest(d):
    h = hashlib.sha256()
    for f in sorted(x for x in os.listdir(d) if x not in SIDE):
        h.update(f.encode() + b"\0" + hashlib.sha256(open(os.path.join(d, f), "rb").read()).digest())
    return h.hexdigest()


def recipe(tmp, label, rows):
    p = os.path.join(tmp, f"recipe-{label}.json")
    json.dump({"recipe": f"test-{label}", "model_type": "gemma4", "rows": rows}, open(p, "w"))
    return p


def build_grade(tmp, label, args, n_files, n_side=len(SIDE)):
    o1, o2 = os.path.join(tmp, f"{label}-1"), os.path.join(tmp, f"{label}-2")
    ok, out = run("plan", *args)
    check(ok, f"{label}: plan")
    if ok:
        print("    " + "\n    ".join(out.strip().splitlines()[:1]))
    ok1, _ = run("emit", *args, "--out", o1, "--all")
    ok2, _ = run("emit", *args, "--out", o2, "--all")
    files = sorted(os.listdir(o1)) if ok1 else []
    same = ok1 and ok2 and files == sorted(os.listdir(o2)) and all(
        open(os.path.join(o1, f), "rb").read() == open(os.path.join(o2, f), "rb").read() for f in files)
    check(same and len(files) == n_files + 1 + n_side,
          f"{label}: emit --all twice, byte-identical ({len(files)} files: shards, index, side files)")
    ok, out = run("verify", *args, "--out", o1, "--all")
    check(ok and "failing: 0" in out and out.count("side file ") == n_side and "-> FAIL" not in out,
          f"{label}: verify --all PASS (the side files byte-equal)")
    ok, out = run("audit", "--out", o1)
    check(ok and "AUDIT PASS" in out, f"{label}: audit PASS")
    return o1


def records(o):
    shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
    decl, names, experts = {}, set(), []
    for f in shards:
        h, _ = header(os.path.join(o, f))
        names |= set(h) - {"__metadata__"}
        decl.update(json.loads(h["__metadata__"]["pulsar.tensors"]))
        experts += json.loads(h["__metadata__"]["pulsar.experts"])
    hv, _ = header(os.path.join(o, shards[0]))
    kv = {e["key"]: e["value"] for e in json.loads(hv["__metadata__"]["pulsar.kv"])}
    return shards, decl, names, experts, kv, hv


JEV_REGEX = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
R, SLOTS = 4, 6


def make_adapter(root, rng, cfg_edit=None, edit=None):
    """A JEV-shaped adapter for the synthetic target: PEFT LoRA (r 4, alpha 8) on q / k / v / o + the dense MLP, a
    [SLOTS, H] head, judge_config + calibration_gold.  Returns {container name: F32 bytes}."""
    os.makedirs(os.path.join(root, "adapter"), exist_ok=True)
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": 8, "target_modules": JEV_REGEX, "bias": "none",
           "use_dora": False, "use_rslora": False, "fan_in_fan_out": False, "lora_dropout": 0.05,
           "base_model_name_or_path": "x", "peft_version": "0.21.0"}
    if cfg_edit:
        cfg_edit(cfg)
    json.dump(cfg, open(os.path.join(root, "adapter", "adapter_config.json"), "w"))
    json.dump({"hidden_size": H, "slots": {"num_slots": SLOTS}, "softcap": 30.0, "readout": {"type": "last_token"},
               "lora": {"r": R, "alpha": 8, "target_modules": JEV_REGEX}},
              open(os.path.join(root, "judge_config.json"), "w"))
    json.dump({"per_kind": {"noul": 1.2}}, open(os.path.join(root, "calibration_gold.json"), "w"))
    t, want = {}, {}

    def f32(src, cname, *shape):
        b = (rng.standard_normal(shape).astype(np.float32) * 0.01).tobytes()
        t[src] = ("F32", list(shape), b)
        want[cname] = b

    for il, typ in enumerate(TYPES):
        nkv, hd = FULL if typ == "full_attention" else SLIDING
        dims = {"self_attn.q_proj": (NQ * hd, H), "self_attn.k_proj": (nkv * hd, H), "self_attn.o_proj": (H, NQ * hd),
                "mlp.gate_proj": (FF, H), "mlp.up_proj": (FF, H), "mlp.down_proj": (H, FF)}
        if typ == "sliding_attention":
            dims["self_attn.v_proj"] = (nkv * hd, H)
        for s, (o, i) in dims.items():
            pre = f"base_model.model.{P}layers.{il}.{s}"
            f32(pre + ".lora_A.weight", f"adapter.layers.{il}.{s}.lora_A.weight", R, i)
            f32(pre + ".lora_B.weight", f"adapter.layers.{il}.{s}.lora_B.weight", o, R)
    if edit:
        edit(t)
    write_st(os.path.join(root, "adapter", "adapter_model.safetensors"), t)
    head = {}
    for n, cname, shape in (("proj.weight", "adapter.head.weight", (SLOTS, H)), ("proj.bias", "adapter.head.bias", (SLOTS,))):
        b = (rng.standard_normal(shape).astype(np.float32)).tobytes()
        head[n] = ("F32", list(shape), b)
        want[cname] = b
    write_st(os.path.join(root, "head.safetensors"), head)
    return want


ROWS_K4 = [{"role": "dense", "format": "exl3m_k4"}, {"role": "expert_gate_up", "format": "exl3m_k4"},
           {"role": "expert_down", "format": "exl3m_k4"}, {"role": "router", "format": "native"},
           {"role": "norm", "format": "native"}, {"role": "embed", "format": "native"},
           {"role": "head", "format": "exl3m_k6"}, {"role": "other", "format": "native"}]


def main():
    rng = np.random.default_rng(338)
    tmp = tempfile.mkdtemp(prefix="gemma4-builder-test-")
    try:
        n_files = 1 + L + 1                 # vision (kv only) + layers + top
        print("EXL3 K4:")
        hf = os.path.join(tmp, "hf")
        t = make_hf(hf, rng)
        pk = os.path.join(tmp, "pack")
        pt = make_exl3(pk, rng)
        rk4 = recipe(tmp, "k4", ROWS_K4)
        o = build_grade(tmp, "k4", ["--hf", hf, "--exl3", pk, "--recipe", rk4], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["k4"], f"GOLDEN k4: {got}")
        shards, decl, names, experts, kv, hv = records(o)

        def d(il, s):
            return decl[f"{P}layers.{il}.{s}"]
        check(d(0, "self_attn.q_proj.weight")["role"] == "attn_q" and d(0, "self_attn.q_proj.weight")["layer"] == 0
              and d(0, "self_attn.q_proj.weight")["block"] == "layers" and d(0, "self_attn.v_proj.weight")["role"] == "attn_v"
              and d(1, "self_attn.k_proj.weight")["dims_ne"] == [H, FULL[0] * FULL[1]]
              and f"{P}layers.1.self_attn.v_proj.weight" not in decl
              and d(1, "self_attn.q_norm.weight")["role"] == "attn_q_norm"
              and d(0, "post_attention_layernorm.weight")["role"] == "attn_post_norm"
              and d(0, "pre_feedforward_layernorm.weight")["role"] == "ffn_pre_norm"
              and d(0, "post_feedforward_layernorm_1.weight")["role"] == "ffn_post_norm_1"
              and d(0, "pre_feedforward_layernorm_2.weight")["role"] == "moe_pre_norm"
              and d(0, "post_feedforward_layernorm_2.weight")["role"] == "ffn_post_norm_2"
              and d(0, "post_feedforward_layernorm.weight")["role"] == "ffn_post_norm"
              and d(0, "router.proj.weight")["role"] == "router" and d(0, "router.scale")["role"] == "router_scale"
              and d(0, "router.per_expert_scale")["role"] == "router_expert_scale"
              and d(1, "layer_scalar")["role"] == "layer_scalar"
              and decl[P + "embed_tokens.weight"]["role"] == "embed" and "block" not in decl[P + "embed_tokens.weight"]
              and decl[P + "norm.weight"]["role"] == "output_norm"
              and decl["lm_head.weight"]["role"] == "head" and decl["lm_head.weight"]["layout"] == "exl3m_k6"
              and decl["lm_head.weight"]["dims_ne"] == [H, V],
              "records: every bind role per layer (block layers, layer N), K = V (no attn_v on the global layer), "
              "embed / output_norm / head (EXL3 K6) at the top")
        check(d(0, "mlp.gate_proj.weight")["dims_ne"] == [H, stored(FF)]
              and d(0, "mlp.up_proj.weight")["dims_ne"] == [H, stored(FF)]
              and d(0, "mlp.down_proj.weight")["dims_ne"] == [stored(FF), H]
              and d(0, "mlp.down_proj.weight")["layout"] == "exl3m_k4",
              f"the dense MLP declared at the stored width ({FF} -> {stored(FF)})")
        fams = {(x["layer"], x["part"]): x for x in experts}
        check(len(experts) == 3 * L
              and all(fams[(il, p)]["dims_per_expert_ne"] == ([H, stored(FE)] if p != "down_proj" else [stored(FE), H])
                      and fams[(il, p)]["n_experts"] == E and fams[(il, p)]["layout"] == "exl3m_k4"
                      and fams[(il, p)]["role"] == {"gate_proj": "expert_gate", "up_proj": "expert_up",
                                                    "down_proj": "expert_down"}[p]
                      and fams[(il, p)]["block"] == "layers"
                      and fams[(il, p)]["entry_name"] == f"{P}layers.{il}.experts.{{e}}.{p}.weight"
                      for il in range(L) for p in ("gate_proj", "up_proj", "down_proj")),
              f"expert families gate / up / down per layer at the stored width ({FE} -> {stored(FE)}), role, block, "
              "entry_name")
        bad, n_lin = 0, 0
        for name in sorted(names):
            if name != "lm_head.weight" and (not name.startswith(P + "layers.") or not (".self_attn." in name and name.endswith("_proj.weight")
                                                          or ".mlp." in name or ".experts." in name)):
                continue
            key = name[:-len(".weight")]
            want = b"".join(pt[f"{key}.{s}"][2] for s in ("trellis", "suh", "svh"))
            f = next(x for x in shards if name in header(os.path.join(o, x))[0])
            bad += entry_bytes(o, f, name) != want
            n_lin += 1
        check(bad == 0 and n_lin == L * (3 + 3 + 3 * E) + 1 + 1,
              f"every Linear's bytes == the pack's [trellis | suh | svh] ({n_lin})")
        check(kv["general.architecture"] == "gemma4" and kv["gemma4.attention_k_eq_v"] is True
              and kv["gemma4.rope_parameters.full_attention.rope_type"] == "proportional"
              and kv["gemma4.rope_parameters.full_attention.partial_rotary_factor"] == 0.25
              and kv["gemma4.rope_parameters.sliding_attention.rope_theta"] == 10000.0
              and kv["gemma4.layer_types"] == {"__array__": "string", "n": L, "v": TYPES}
              and kv["gemma4.final_logit_softcapping"] == 30.0
              and kv["pulsar.vision_present"] is False and [k for k in hv if k != "__metadata__"] == [],
              "pulsar.kv: text_config verbatim (both ropes flattened, layer_types); the primary shard holds no tensor")
        check(not any("vision" in n for n in names), "no vision tensor in the container (consumed by name)")

        print("the JEV adapter (--adapter jev):")
        ad = os.path.join(tmp, "jev")
        aw = make_adapter(ad, np.random.default_rng(9))
        rjev = recipe(tmp, "k4-jev", ROWS_K4 + [{"role": "adapter", "format": "native"}])
        oa = build_grade(tmp, "k4-jev", ["--hf", hf, "--exl3", pk, "--recipe", rjev, "--adapter", "jev", ad], n_files + 1,
                         n_side=len(SIDE) + 3)
        a_shards, a_decl, a_names, _a_exp, a_kv, _ = records(oa)
        last = a_shards[-1]
        lora = [n for n in a_names if n.startswith("adapter.layers.")]
        ok_bytes = all(entry_bytes(oa, last, n) == b for n, b in aw.items())
        check(len(lora) == 2 * (6 * L + 1) and all(n in header(os.path.join(oa, last))[0] for n in aw) and ok_bytes,
              f"the adapter shard holds every LoRA pair ({len(lora)} tensors: no v on the global layer) and the head, "
              "bytes == the source's F32")
        q0 = a_decl["adapter.layers.0.self_attn.q_proj.lora_A.weight"]
        b1 = a_decl["adapter.layers.1.mlp.down_proj.lora_B.weight"]
        check(q0["role"] == "attn_q_lora_a" and q0["layer"] == 0 and q0["block"] == "layers" and q0["layout"] == "f32"
              and q0["dims_ne"] == [H, R] and b1["role"] == "ffn_down_lora_b" and b1["dims_ne"] == [R, H]
              and a_decl["adapter.layers.0.mlp.gate_proj.lora_B.weight"]["dims_ne"] == [R, FF]
              and a_decl["adapter.head.weight"]["role"] == "decision_head"
              and a_decl["adapter.head.bias"]["role"] == "decision_head_bias"
              and "adapter.layers.1.self_attn.v_proj.lora_A.weight" not in a_decl,
              "adapter records: <target>_lora_a / _lora_b per layer (block layers), the logical widths (the MLP at "
              f"{FF}, not {stored(FF)}), decision_head / _bias")
        check(a_kv["pulsar.adapter"] == "jev" and a_kv["pulsar.adapter.lora_r"] == R
              and a_kv["pulsar.adapter.lora_alpha"] == 8.0
              and all(os.path.isfile(os.path.join(oa, f)) for f in ("judge_config.json", "calibration_gold.json",
                                                                     "adapter_config.json")),
              "adapter kv (kind, rank, alpha) and its side files")
        base_same = all(entry_bytes(oa, a_shards[i], n) == entry_bytes(o, shards[i], n)
                        for i in range(len(shards)) for n in header(os.path.join(o, shards[i]))[0] if n != "__metadata__")
        check(base_same, "the base tensors' bytes unchanged beside the adapter")

        def adapter_refused(label, contains, **kw):
            dd = os.path.join(tmp, "badjev-" + label.replace(" ", "-")[:30])
            make_adapter(dd, np.random.default_rng(10), **kw)
            ok, _ = run("plan", "--hf", hf, "--exl3", pk, "--recipe", rjev, "--adapter", "jev", dd, expect_fail=True,
                        contains=contains)
            check(ok, f"refused: adapter {label}")
            shutil.rmtree(dd, ignore_errors=True)

        adapter_refused("DoRA", "use_dora", cfg_edit=lambda c: c.update(use_dora=True))
        adapter_refused("another target set", "target_modules", cfg_edit=lambda c: c.update(target_modules="q_proj"))
        adapter_refused("an unknown config key", "does not know", cfg_edit=lambda c: c.update(new_switch=1))
        x = f"base_model.model.{P}layers.0.self_attn.q_proj.lora_B.weight"
        adapter_refused("a wrong shape", "disagree", edit=lambda t: t.update({x: ("F32", [7, R], bytes(28 * R))}))
        adapter_refused("an extra tensor", "does not map",
                        edit=lambda t: t.update({"base_model.model.extra.weight": ("F32", [1], bytes(4))}))
        ok, _ = run("plan", "--hf", hf, "--exl3", pk, "--recipe", rjev, "--adapter", "other", ad, expect_fail=True,
                    contains="reads ['jev']")
        check(ok, "refused: an adapter kind the builder does not read")

        print("refusals:")

        def refused(label, contains, cfg=None, edit=None, dirty=None, rec=None, exl3=True):
            dd = os.path.join(tmp, "bad-" + label.replace(" ", "-")[:40])
            make_hf(dd, np.random.default_rng(7), cfg=cfg, edit=edit)
            args = ["plan", "--hf", dd, "--recipe", rec or rk4]
            if exl3:
                bp = dd + "-pack"
                make_exl3(bp, np.random.default_rng(8), dirty=dirty)
                args += ["--exl3", bp]
            ok, _ = run(*args, expect_fail=True, contains=contains)
            check(ok, f"refused: {label}")
            shutil.rmtree(dd, ignore_errors=True)
            shutil.rmtree(dd + "-pack", ignore_errors=True)

        def put(name, *shape):
            def e(t):
                t[name] = ("BF16", list(shape), bf16(np.zeros(shape, np.float32)).tobytes())
            return e

        def drop(name):
            def e(t):
                del t[name]
            return e

        refused("an unknown tensor", "does not map", edit=put(P + "layers.0.mlp.extra.weight", 4))
        refused("a missing tensor", "missing", edit=drop(P + "layers.1.router.scale"))
        refused("a wrong shape", "declared shape", edit=put(P + "layers.1.self_attn.k_proj.weight", 2 * FULL[1], H))
        refused("a v_proj on the global layer (K = V)", "does not map",
                edit=put(P + "layers.1.self_attn.v_proj.weight", FULL[0] * FULL[1], H))
        c = config(); c["text_config"]["attention_k_eq_v"] = False
        refused("a switch (attention_k_eq_v false)", "attention_k_eq_v", cfg=c)
        c = config(); c["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.5
        refused("the global rope (partial factor 0.5)", "partial_rotary_factor", cfg=c)
        refused("a nonzero svh on a padded dense-MLP gate column", "padded columns", dirty=P + "layers.1.mlp.up_proj")
        refused("a nonzero svh on a padded expert gate column", "padded columns",
                dirty=P + "layers.0.experts.3.gate_proj")
        rows = [r if r["role"] != "dense" else {"role": "dense", "format": "native"} for r in ROWS_K4]
        refused("the dense MLP written bf16 (its width is the pack's)", "binds this dense tensor",
                rec=recipe(tmp, "mlp-bf16", rows))
        rows = [r if r["role"] != "head" else {"role": "head", "format": "native"} for r in ROWS_K4]
        refused("the head written bf16 (the Turing build runs the pack's)", "binds this head tensor",
                rec=recipe(tmp, "head-bf16", rows))
        refused("no --exl3", "pass --exl3", rec=recipe(tmp, "no-exl3", [
            {"role": r, "format": "native"} for r in ("router", "norm", "embed", "head", "other")]), exl3=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'test_gemma4: PASS' if not FAILS else 'test_gemma4: FAIL (' + str(len(FAILS)) + ')'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

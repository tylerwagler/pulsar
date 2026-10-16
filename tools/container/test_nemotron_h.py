#!/usr/bin/env python3
"""test_nemotron_h.py -- the Nemotron-H family in the artifact builder (nemotron_h.py, L296 N1), on a synthetic Omni
checkpoint (tiny dims, the real layout: llm_config nested, `language_model.` prefix, per-expert up / down, an MTP
module, vision tensors):

  * BF16 build: plan, emit --all twice byte-identical, verify --all (side files byte-equal), audit; GOLDEN
  * the grouping: blocks [mamba, moe, mamba, attention, moe] -> 3 pulsar layers {M,E} {M,-} {*,E}; every record's
    role / block / layer is the PULSAR layer; the conv declared [K, C]; the shared expert index 0; routed families
    up / down bf16 with entry_name + layer + role + block; the drafter's input side and out norm on mtp.0
  * pulsar.kv: llm_config verbatim, time_step_limit as its two pinned facts, mtp / vision presence; no vision
    tensor in the container
  * an EXL3 build of every dense Linear and the shared expert (dense arm), the routed experts BF16: bytes == the
    pack's [trellis | suh | svh], the pack keyed as exllamav3 writes it (no `language_model.` prefix)
  * a full EXL3 build (L296): the up-only routed experts too, from the pack (expert width 128, an EXL3 multiple):
    every expert-projection's bytes == the pack's, its family exl3m_k4; a pack keyed under the HF names refused
  * the drafter omitted whole -> no mtp shard, pulsar.mtp_present false
  * refusals by name: an unknown tensor, a missing tensor, a wrong shape, a switch (relu2 -> silu), a dt clamp
    (time_step_limit [0.001, inf]), a moe with no mixer before it, an `mlp` block, half a drafter omitted, an
    omitted main-model tensor
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
from test_qwen import bf16, write_st, exl3_parts  # noqa: E402

FAILS: list[str] = []
# sha256 over (file name, sha256(file)) of every emitted file but the side files; fixed when the family landed (L296 N1)
GOLDEN = {
    "bf16": "5b1b9a7921dc9fbe143774827d9ee7e60f3637def4fc2dfe390beec4ea4a8289",
}
H, V, NQ, NKV, HD = 128, 64, 4, 2, 32
NH, HDM, N, G, CK = 8, 32, 16, 2, 4          # Mamba: heads, head dim (d_inner 256 = 2 x H), state, groups, conv
X, TOPK, FF, LAT, SFF = 4, 2, 64, 128, 128    # experts, top-k, expert width, latent, shared width
BLOCKS = ["mamba", "moe", "mamba", "attention", "moe"]
DI, CD = NH * HDM, NH * HDM + 2 * G * N
LM, BB, MTPP = "language_model.", "language_model.backbone.", "language_model.mtp."
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def config():
    return {
        "architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni", "eos_token_id": 11,
        "vision_config": {"model_type": "radio", "hidden_size": 32}, "tie_word_embeddings": False,
        "llm_config": {
            "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": H, "vocab_size": V,
            "layers_block_type": list(BLOCKS), "mtp_layers_block_type": ["attention", "moe"],
            "num_nextn_predict_layers": 1, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD,
            "mamba_num_heads": NH, "mamba_head_dim": HDM, "ssm_state_size": N, "n_groups": G, "conv_kernel": CK,
            "expand": 2, "chunk_size": 128, "n_routed_experts": X, "num_experts_per_tok": TOPK,
            "moe_intermediate_size": FF, "intermediate_size": FF, "moe_latent_size": LAT,
            "moe_shared_expert_intermediate_size": SFF, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": True, "routed_scaling_factor": 5.0, "layer_norm_epsilon": 1e-5,
            "attention_bias": False, "mlp_bias": False, "use_bias": False, "mamba_proj_bias": False,
            "use_conv_bias": True, "mlp_hidden_act": "relu2", "mamba_hidden_act": "silu", "sliding_window": None,
            "tie_word_embeddings": False, "residual_in_fp32": False, "moe_shared_expert_overlap": False,
            "time_step_limit": [0.0, {"__float__": "Infinity"}], "time_step_min": 0.001, "time_step_max": 0.1,
            "time_step_floor": 0.0001, "max_position_embeddings": 1048576, "eos_token_id": 11, "bos_token_id": 1,
            "pad_token_id": 0, "mamba_ssm_cache_dtype": "float32", "dtype": "bfloat16"},
    }


def make_hf(root, rng, cfg=None, edit=None):
    os.makedirs(root, exist_ok=True)
    json.dump(cfg or config(), open(os.path.join(root, "config.json"), "w"))
    ff = (cfg or config())["llm_config"]["moe_intermediate_size"]
    for f in SIDE[:-1]:
        open(os.path.join(root, f), "w").write(f"{{\"synthetic\": \"{f}\"}}\n")
    t = {}

    def mat(name, *shape, dt="BF16"):
        a = rng.standard_normal(shape).astype(np.float32) * 0.05
        t[name] = (dt, list(shape), bf16(a).tobytes() if dt == "BF16" else a.tobytes())

    def block(p, kind):
        mat(p + "norm.weight", H)
        m = p + "mixer."
        if kind == "mamba":
            mat(m + "in_proj.weight", DI + CD + NH, H); mat(m + "conv1d.weight", CD, 1, CK); mat(m + "conv1d.bias", CD)
            mat(m + "dt_bias", NH); mat(m + "A_log", NH); mat(m + "D", NH); mat(m + "norm.weight", DI)
            mat(m + "out_proj.weight", H, DI)
        elif kind == "attention":
            mat(m + "q_proj.weight", NQ * HD, H); mat(m + "k_proj.weight", NKV * HD, H)
            mat(m + "v_proj.weight", NKV * HD, H); mat(m + "o_proj.weight", H, NQ * HD)
        else:
            mat(m + "gate.weight", X, H); mat(m + "gate.e_score_correction_bias", X, dt="F32")
            mat(m + "fc1_latent_proj.weight", LAT, H); mat(m + "fc2_latent_proj.weight", H, LAT)
            mat(m + "shared_experts.up_proj.weight", SFF, H); mat(m + "shared_experts.down_proj.weight", H, SFF)
            for e in range(X):
                mat(m + f"experts.{e}.up_proj.weight", ff, LAT); mat(m + f"experts.{e}.down_proj.weight", LAT, ff)
    mat(BB + "embeddings.weight", V, H); mat(BB + "norm_f.weight", H); mat(LM + "lm_head.weight", V, H)
    for b, kind in enumerate(BLOCKS):
        block(f"{BB}layers.{b}.", kind)
    block(f"{MTPP}layers.0.", "attention"); block(f"{MTPP}layers.1.", "moe")
    mat(MTPP + "layers.0.enorm.weight", H); mat(MTPP + "layers.0.hnorm.weight", H)
    mat(MTPP + "layers.0.eh_proj.weight", H, 2 * H); mat(MTPP + "layers.1.final_layernorm.weight", H)
    mat("vision_model.radio_model.model.blocks.0.attn.qkv.weight", 96, 32); mat("mlp1.0.weight", 128)
    mat("vision_projector.vision_final_layernorm.weight", 32)
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


DENSE = ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "fc1_latent_proj", "fc2_latent_proj",
         "shared_experts.up_proj", "shared_experts.down_proj")


def make_exl3(root, hf_t, rng, K=4, experts=False, prefixed=False):
    """The EXL3 pack of the trunk's dense Linears and shared experts -- and its routed experts when `experts` -- keyed
    as exllamav3's NemotronH Omni adapter writes them: the HF names minus `language_model.` and `.weight` (`prefixed`:
    under the HF names, which the builder must refuse)."""
    t = {}
    for name, (_dt, shape, _b) in hf_t.items():
        routed = experts and ".mixer.experts." in name
        if name.startswith(BB + "layers.") and (routed or any(name.endswith(f"mixer.{d}.weight") for d in DENSE)):
            out, inp = shape
            key = name[:-len(".weight")] if prefixed else name[len(LM):-len(".weight")]
            for s, v in exl3_parts(rng, inp, (out + 127) // 128 * 128, K).items():
                t[f"{key}.{s}"] = v
    os.makedirs(root, exist_ok=True)
    write_st(os.path.join(root, "model-00001-of-00001.safetensors"), t)
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
    json.dump({"recipe": f"test-{label}", "model_type": "nemotron_h_omni", "rows": rows}, open(p, "w"))
    return p


def build_grade(tmp, label, args, n_files):
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
    check(same and len(files) == n_files + 1 + len(SIDE),
          f"{label}: emit --all twice, byte-identical ({len(files)} files: shards, index, side files)")
    ok, out = run("verify", *args, "--out", o1, "--all")
    check(ok and "failing: 0" in out and out.count("side file ") == len(SIDE) and "-> FAIL" not in out,
          f"{label}: verify --all PASS (the side files byte-equal)")
    ok, out = run("audit", "--out", o1)
    check(ok and "AUDIT PASS" in out, f"{label}: audit PASS")
    return o1


def main():
    rng = np.random.default_rng(296)
    tmp = tempfile.mkdtemp(prefix="nemotron-builder-test-")
    try:
        hf = os.path.join(tmp, "hf")
        t = make_hf(hf, rng)
        rbf = os.path.join(HERE, "format-maps", "nemotron_h-bf16.json")
        n_files = 1 + 3 + 1 + 1

        print("BF16 reference build:")
        o = build_grade(tmp, "bf16", ["--hf", hf, "--recipe", rbf], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["bf16"], f"GOLDEN bf16: {got}")
        shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
        hv, _ = header(os.path.join(o, shards[0]))
        kv = {e["key"]: e["value"] for e in json.loads(hv["__metadata__"]["pulsar.kv"])}
        check(kv["general.architecture"] == "nemotron_h" and kv["nemotron_h.layers_block_type"]["v"] == BLOCKS
              and kv["nemotron_h.mtp_layers_block_type"]["v"] == ["attention", "moe"]
              and kv["nemotron_h.time_step_limit.min"] == 0.0 and kv["nemotron_h.time_step_limit.max_inf"] is True
              and "nemotron_h.time_step_limit" not in kv and kv["nemotron_h.moe_latent_size"] == LAT
              and kv["pulsar.mtp_present"] is True and kv["pulsar.vision_present"] is False
              and not any(k.startswith("nemotron_h.vision") for k in kv)
              and [k for k in hv if k != "__metadata__"] == [],
              "pulsar.kv: llm_config verbatim, time_step_limit as min 0 + max_inf, presence; the primary shard holds no "
              "tensor")
        allnames = set()
        decl = {}
        fams = {}
        for f in shards:
            h, _ = header(os.path.join(o, f))
            allnames |= set(h) - {"__metadata__"}
            decl.update(json.loads(h["__metadata__"]["pulsar.tensors"]))
            fams[h["__metadata__"]["pulsar.shard_key"]] = json.loads(h["__metadata__"]["pulsar.experts"])
        check(not any(n.startswith(("vision_model.", "vision_projector.", "mlp1.")) for n in allnames),
              "no vision tensor in the container (consumed by name)")
        m0, m4 = BB + "layers.0.mixer.", BB + "layers.4.mixer."
        check(decl[m0 + "conv1d.weight"]["dims_ne"] == [CK, CD] and decl[m0 + "conv1d.weight"]["role"] == "ssm_conv"
              and decl[BB + "layers.0.norm.weight"]["role"] == "ssm_pre_norm"
              and decl[BB + "layers.1.norm.weight"] == {"layout": "bf16", "dims_ne": [H], "role": "ffn_norm",
                                                        "gguf_name": BB + "layers.1.norm.weight", "block": "layers",
                                                        "layer": 0}
              and decl[BB + "layers.2.mixer.in_proj.weight"]["layer"] == 1
              and decl[BB + "layers.3.mixer.q_proj.weight"]["role"] == "attn_q"
              and decl[BB + "layers.3.norm.weight"]["role"] == "attn_norm"
              and decl[m4 + "gate.e_score_correction_bias"]["layout"] == "f32"
              and decl[m4 + "gate.e_score_correction_bias"]["role"] == "router_select_bias"
              and decl[m4 + "gate.e_score_correction_bias"]["layer"] == 2
              and decl[m4 + "fc1_latent_proj.weight"]["role"] == "latent_in"
              and decl[m4 + "shared_experts.down_proj.weight"]["index"] == 0
              and decl[m4 + "shared_experts.down_proj.weight"]["role"] == "shared_down",
              "records: role / block / PULSAR layer (blocks 0+1 -> 0, 2 -> 1, 3+4 -> 2); conv declared [K, C]; the "
              "shared expert index 0")
        f2 = fams["layers.2"]
        check(sorted(x["part"] for x in f2) == ["down_proj", "up_proj"]
              and all(x["layout"] == "bf16" and x["layer"] == 2 and x["n_experts"] == X and x["block"] == "layers"
                      and x["entry_name"] == m4 + "experts.{e}." + x["part"] + ".weight"
                      and x["role"] == "expert_" + x["part"].split("_")[0]
                      and x["dims_per_expert_ne"] == ([LAT, FF] if x["part"] == "up_proj" else [FF, LAT]) for x in f2)
              and fams["layers.1"] == [],
              "routed families: up / down bf16 [latent <-> ff], entry_name + layer + role + block; layer 1 has none")
        f_l2 = [f for f in shards if m4 + "experts.3.up_proj.weight" in header(os.path.join(o, f))[0]]
        check(bool(f_l2) and entry_bytes(o, f_l2[0], m4 + "experts.3.up_proj.weight") == t[m4 + "experts.3.up_proj.weight"][2],
              "routed expert 3 up == its HF tensor's bytes")
        md = MTPP + "layers."
        check(decl[md + "0.eh_proj.weight"]["role"] == "draft_in_proj" and decl[md + "0.eh_proj.weight"]["block"] == "mtp"
              and decl[md + "0.eh_proj.weight"]["dims_ne"] == [2 * H, H]
              and decl[md + "1.final_layernorm.weight"]["role"] == "draft_out_norm"
              and decl[md + "1.norm.weight"]["role"] == "ffn_norm" and decl[md + "1.norm.weight"]["layer"] == 0
              and header(os.path.join(o, shards[-1]))[0]["__metadata__"]["pulsar.shard_key"] == "mtp.0",
              "drafter: mtp.0 last; eh_proj draft_in_proj [2H -> H]; final_layernorm draft_out_norm; its layer 0")

        print("EXL3 dense build (dense arm K4; routed BF16):")
        ex = os.path.join(tmp, "exl3")
        ext = make_exl3(ex, t, np.random.default_rng(7))
        rk4 = recipe(tmp, "k4", [
            {"role": "dense", "block": "layers", "format": "exl3m_k4"}, {"role": "dense", "block": "mtp", "format": "native"},
            {"role": "shared_expert", "block": "layers", "format": "exl3m_k4"},
            {"role": "shared_expert", "block": "mtp", "format": "native"},
            *[{"role": r, "format": "native"} for r in ("expert_up", "expert_down", "router", "norm", "embed", "head",
                                                       "other")]])
        o4 = build_grade(tmp, "k4", ["--hf", hf, "--exl3", ex, "--recipe", rk4], n_files)
        key = BB + "layers.2.mixer.in_proj"
        f_l1 = [f for f in sorted(os.listdir(o4)) if f.endswith(".safetensors")
                and key + ".weight" in header(os.path.join(o4, f))[0]]
        d1 = json.loads(header(os.path.join(o4, f_l1[0]))[0]["__metadata__"]["pulsar.tensors"]) if f_l1 else {}
        check(bool(f_l1) and d1[key + ".weight"]["layout"] == "exl3m_k4" and d1[key + ".weight"]["dims_ne"] == [H, DI + CD + NH]
              and entry_bytes(o4, f_l1[0], key + ".weight") == b"".join(ext[f"{key[len(LM):]}.{s}"][2]
                                                                        for s in ("trellis", "suh", "svh")),
              "ssm_in exl3m_k4 at its logical width, bytes == the pack's [trellis | suh | svh]")

        print("full EXL3 build (L296: the up-only routed experts from the pack too; expert width 128):")
        c128 = config()
        c128["llm_config"]["moe_intermediate_size"] = c128["llm_config"]["intermediate_size"] = 128
        hf128 = os.path.join(tmp, "hf128")
        t128 = make_hf(hf128, np.random.default_rng(11), cfg=c128)
        ex128 = os.path.join(tmp, "exl3-128")
        ext128 = make_exl3(ex128, t128, np.random.default_rng(12), experts=True)
        rfull = recipe(tmp, "full", [
            *[{"role": r, "block": "layers", "format": "exl3m_k4"} for r in ("dense", "shared_expert", "expert_up",
                                                                              "expert_down")],
            *[{"role": r, "block": "mtp", "format": "native"} for r in ("dense", "shared_expert", "expert_up",
                                                                         "expert_down")],
            *[{"role": r, "format": "native"} for r in ("router", "norm", "embed", "head", "other")]])
        of = build_grade(tmp, "full", ["--hf", hf128, "--exl3", ex128, "--recipe", rfull], n_files)
        ek = BB + "layers.4.mixer.experts.3.up_proj"
        ff = [f for f in sorted(os.listdir(of)) if f.endswith(".safetensors") and ek + ".weight" in header(os.path.join(of, f))[0]]
        fam = [x for x in json.loads(header(os.path.join(of, ff[0]))[0]["__metadata__"]["pulsar.experts"])
               if x["part"] == "up_proj"] if ff else []
        check(bool(ff) and fam and fam[0]["layout"] == "exl3m_k4" and fam[0]["n_experts"] == X
              and entry_bytes(of, ff[0], ek + ".weight") == b"".join(ext128[f"{ek[len(LM):]}.{s}"][2]
                                                                      for s in ("trellis", "suh", "svh")),
              "routed up expert 3 exl3m_k4, bytes == the pack's [trellis | suh | svh]; its family exl3m_k4 x 4 experts")
        exp = os.path.join(tmp, "exl3-prefixed")
        make_exl3(exp, t128, np.random.default_rng(12), experts=True, prefixed=True)
        ok, _ = run("plan", "--hf", hf128, "--exl3", exp, "--recipe", rfull, expect_fail=True,
                    contains="missing from the EXL3 checkpoint")
        check(ok, "a pack keyed under the HF names (language_model.) -> refused (not the exllamav3 layout)")

        print("drafter omitted:")
        rom = recipe(tmp, "omit", [
            *[{"role": r, "block": "layers", "format": "native"} for r in ("dense", "norm", "other", "shared_expert",
                                                                            "expert_up", "expert_down", "router")],
            *[{"role": r, "block": "mtp", "format": "omit"} for r in ("dense", "norm", "shared_expert", "expert_up",
                                                                       "expert_down", "router")],
            {"name": BB + "norm_f.weight", "format": "native"},
            *[{"role": r, "format": "native"} for r in ("embed", "head")]])
        oo = os.path.join(tmp, "omit")
        ok, _ = run("emit", "--hf", hf, "--recipe", rom, "--out", oo, "--all")
        files = sorted(f for f in os.listdir(oo) if f.endswith(".safetensors")) if ok else []
        kvo = {e["key"]: e["value"] for e in json.loads(header(os.path.join(oo, files[0]))[0]["__metadata__"]
                                                         ["pulsar.kv"])} if files else {}
        check(ok and len(files) == 1 + 3 + 1 and kvo.get("pulsar.mtp_present") is False,
              "no mtp shard, pulsar.mtp_present false")
        ok, out = run("verify", "--hf", hf, "--recipe", rom, "--out", oo, "--all")
        check(ok and "failing: 0" in out, "verify PASS (drafter omitted)")

        print("refusals:")
        cases = [
            ("an unknown tensor", None, lambda t: t.__setitem__(BB + "layers.0.mixer.extra.weight", t[BB + "norm_f.weight"]),
             "does not map"),
            ("a missing tensor", None, lambda t: t.pop(BB + "layers.2.mixer.D"), "missing from"),
            ("a missing expert", None, lambda t: t.pop(BB + "layers.4.mixer.experts.3.down_proj.weight"), "missing from"),
            ("a wrong shape", None, lambda t: t.__setitem__(BB + "layers.3.mixer.k_proj.weight",
                                                            t[BB + "layers.3.mixer.q_proj.weight"]),
             "disagree with the declared shape"),
        ]
        for label, key, val, msg in (("relu2 -> silu experts", "mlp_hidden_act", "silu", "assumes 'relu2'"),
                                     ("a dt clamp", "time_step_limit", [0.001, {"__float__": "Infinity"}], "[0, inf]"),
                                     ("a moe with no mixer before it", "layers_block_type",
                                      ["moe", "mamba", "moe", "attention", "moe"], "no mixer before it"),
                                     ("an mlp block", "layers_block_type", ["mamba", "mlp", "mamba", "attention", "moe"],
                                      "not a block kind"),
                                     ("a second shared expert", "n_shared_experts", 2, "assumes 1")):
            c = config()
            c["llm_config"][key] = val
            cases.append((label, c, None, msg))
        for i, (label, cfg, edit, msg) in enumerate(cases):
            d = os.path.join(tmp, f"bad-{i}")
            make_hf(d, np.random.default_rng(1), cfg=copy.deepcopy(cfg) if cfg else None, edit=edit)
            ok, _ = run("plan", "--hf", d, "--recipe", rbf, expect_fail=True, contains=msg)
            check(ok, f"{label} -> refused")
        r = json.load(open(rbf))
        hp = recipe(tmp, "half-mtp", [x for x in r["rows"] if x["role"] != "router"] + [
            {"role": "router", "block": "layers", "format": "native"}, {"role": "router", "block": "mtp", "format": "omit"}])
        ok, _ = run("plan", "--hf", hf, "--recipe", hp, expect_fail=True, contains="all of it or none")
        check(ok, "half the drafter omitted -> refused")
        rr = recipe(tmp, "omit-router", [x for x in r["rows"] if x["role"] != "router"] + [
            {"role": "router", "block": "layers", "format": "omit"}, {"role": "router", "block": "mtp", "format": "native"}])
        ok, _ = run("plan", "--hf", hf, "--recipe", rr, expect_fail=True, contains="the Nemotron-H loader binds")
        check(ok, "an omitted main-model tensor (the router) -> refused")
        print("side files:")
        bare = os.path.join(tmp, "hf-bare")
        shutil.copytree(hf, bare, ignore=shutil.ignore_patterns("tokenizer.json"))
        ok, _ = run("emit", "--hf", bare, "--recipe", rbf, "--out", os.path.join(tmp, "bare"), "--all",
                    expect_fail=True, contains="--tokenizer")
        check(ok and not os.path.exists(os.path.join(tmp, "bare")), "a checkpoint without tokenizer.json -> refused "
                                                                     "naming --tokenizer, before any shard is written")
    finally:
        shutil.rmtree(tmp)
    print(f"test_nemotron_h: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

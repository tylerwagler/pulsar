#!/usr/bin/env python3
"""The Inkling family of the direct builder (L294 / L295 E1), end to end on a SYNTHETIC miniature checkpoint -- no
mounts, no GPU, seconds.  The real checkpoint's names, dtypes and fused layouts at toy shapes (hidden 256, 3 layers:
one dense + two MoE, 4 routed + 2 shared experts, window / full attention, 2 drafter layers, a 3-linear hmlp tower,
the audio tower), and an exllamav3-layout EXL3 pack of random trellis / suh / svh under the L294 key names.  What it
pins:

  * the Source view: every routed / shared / dense gate and up projection == the even / odd rows of its fused w13
    (the row interleave exllamav3's frange_slice reads, the L294 BF16 parity), every down == its stack slice
  * plan -> emit twice -> byte-identical shards -> verify PASS -> audit PASS, for the BF16 reference recipe and a
    K3 EXL3 recipe (EXL3 entries byte-identical to the pack's [trellis | suh | svh])
  * the container: model.audio.* nowhere; short convs declared [channels, kernel]; every tensor and family record
    carrying its bind role, block and layer; routed families with entry_name; pulsar.kv (text / vision / mtp
    configs verbatim, eos, mtp / vision presence)
  * the drafter and the tower omitted whole -> no mtp shards, pulsar.mtp_present false
  * refusals by name: an unknown tensor, a missing tensor, a tensor whose shape disagrees with the declaration, an
    architecture switch the role table does not describe, a moe_intermediate_size that is not intermediate_size,
    half a drafter omitted, an omitted main-model tensor, routed experts at a rate no routed arm reads (K8; K5 plans
    since L295 F7)
  * L295 F7: the pack's K6 head -- stored at the 128 multiple, declared at its logical width, bytes the pack's
  * GOLDEN: the SHA-256 of every emitted shard and the index of the BF16 and the K3 build (the byte-identity
    instrument, rule 7)
  * the side files (L294 E6): emit --all writes the tokenizer beside the shards byte-equal to the checkpoint's
    (tiktoken/tokenizer.model as tokenizer.model, tokenizer_config.json, config.json), verify --all checks them; a
    checkpoint without them refuses naming --tokenizer, and --tokenizer DIR supplies them
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
import exl3_rates  # noqa: E402

FAILS: list[str] = []
# sha256 over (file name, sha256(file)) of every emitted file but the side files (test_deepseek.digest's rule); fixed
# when the family landed (L295 E1)
GOLDEN = {
    "bf16": "9ad19c8f08c6befd6bef34b1a2126fd00809ee0d9e270a70eb9ad73345947818",
    "k3": "7716f2f45a7ce7f45bf375db989dbfb1e6e136a1eb70d505db78fb2ba753e5ff",
}
H, V, NQ, NKV, HD, DREL, EXT, WIN, CK = 256, 64, 8, 4, 32, 16, 32, 16, 4
L, NDENSE, DFF, FF, E, S, NMTP = 3, 1, 256, 128, 4, 2, 2
LLM, MTP = "model.llm.", "model.mtp."
# the side files (inkling.side_files) and the synthetic checkpoint's tokenizer bytes they copy
SIDE = ("config.json", "tokenizer.model", "tokenizer_config.json")
TOK_MODEL = b"IQ== 0\nIg== 1\n"
TOK_CONFIG = b'{"added_tokens_decoder": {}}\n'


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def config():
    return {
        "architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model", "eos_token_id": 61,
        "text_config": {
            "model_max_length": 4096, "torch_dtype": "bfloat16", "hidden_size": H, "num_hidden_layers": L,
            "vocab_size": V, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD, "d_rel": DREL,
            "rel_extent": EXT, "q_bias": False, "o_bias": False, "log_scaling_n_floor": 128000,
            "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06, "use_embed_norm": True, "local_layer_ids": [0, 1],
            "dense_mlp_idx": NDENSE, "use_sconv": True, "sconv_kernel_size": CK, "unpadded_vocab_size": 60,
            "logits_mup_width_multiplier": 16.0, "final_logit_softcapping": None, "swa_head_dim": HD,
            "swa_num_attention_heads": NQ, "swa_num_key_value_heads": NKV, "sliding_window_size": WIN,
            "n_routed_experts": E, "num_experts_per_tok": 2, "n_shared_experts": S, "shared_expert_sink": True,
            "dense_intermediate_size": DFF, "intermediate_size": FF, "route_scale": 8.0, "use_gate_bias": True,
            "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True},
        "audio_config": {"decoder_dmodel": H, "n_mel_bins": 80, "audio_mode": "dmel"},
        "vision_config": {"vision_encoder_type": "hmlp", "decoder_dmodel": H, "patch_size": 4,
                          "temporal_patch_size": 2, "n_channels": 3, "n_layers": 3, "use_vision_norm": True},
        "mtp_config": {"num_nextn_predict_layers": NMTP, "chain_hidden_post_norm": False, "local_layer_ids": [0]},
    }


def make_hf(root, rng, cfg=None, edit=None):
    """The synthetic checkpoint; `edit(t)` mutates the tensor dict before it is written (the refusal cases)."""
    os.makedirs(os.path.join(root, "tiktoken"), exist_ok=True)
    json.dump(cfg or config(), open(os.path.join(root, "config.json"), "w"))
    open(os.path.join(root, "tiktoken", "tokenizer.model"), "wb").write(TOK_MODEL)
    open(os.path.join(root, "tokenizer_config.json"), "wb").write(TOK_CONFIG)
    t = {}

    def mat(name, *shape, dt="BF16"):
        a = rng.standard_normal(shape).astype(np.float32) * 0.05
        t[name] = (dt, list(shape), bf16(a).tobytes() if dt == "BF16" else a.tobytes())

    def block(p, window, dense):
        ext = WIN if window else EXT
        mat(p + "attn_norm.weight", H); mat(p + "mlp_norm.weight", H)
        mat(p + "attn_sconv.weight", H, 1, CK); mat(p + "mlp_sconv.weight", H, 1, CK)
        a = p + "attn."
        mat(a + "wq_du.weight", NQ * HD, H); mat(a + "wk_dv.weight", NKV * HD, H); mat(a + "wv_dv.weight", NKV * HD, H)
        mat(a + "wr_du.weight", NQ * DREL, H); mat(a + "wo_ud.weight", H, NQ * HD)
        mat(a + "q_norm.weight", HD); mat(a + "k_norm.weight", HD); mat(a + "rel_logits_proj.proj", DREL, ext)
        mat(a + "k_sconv.weight", NKV * HD, 1, CK); mat(a + "v_sconv.weight", NKV * HD, 1, CK)
        m = p + "mlp."
        if dense:
            mat(m + "w13_dn.weight", 2 * DFF, H); mat(m + "w2_md.weight", H, DFF); mat(m + "global_scale", 1)
        else:
            mat(m + "gate.weight", E + S, H); mat(m + "gate.bias", E, dt="F32"); mat(m + "gate.global_scale", 1, dt="F32")
            mat(m + "experts.w13_weight", E, 2 * FF, H); mat(m + "experts.w2_weight", E, H, FF)
            mat(m + "shared_experts.shared_w13_weight", S, 2 * FF, H)
            mat(m + "shared_experts.shared_w2_weight", S, H, FF)
    mat(LLM + "embed.weight", V, H); mat(LLM + "embed_norm.weight", H); mat(LLM + "norm.weight", H)
    mat(LLM + "unembed.weight", V, H)
    for i in range(L):
        block(f"{LLM}layers.{i}.", i in (0, 1), i < NDENSE)
    for n in range(NMTP):
        p = f"{MTP}layers.{n}."
        mat(p + "embed_norm.weight", H); mat(p + "hidden_norm.weight", H); mat(p + "input_proj.weight", H, 2 * H)
        block(p + "transformer_block.", n == 0, True)
    mat("model.visual.layers.linear_0.weight", 8, 12); mat("model.visual.layers.norm_0.weight", 8)
    mat("model.visual.layers.linear_1.weight", 16, 32); mat("model.visual.layers.norm_1.weight", 16)
    mat("model.visual.layers.linear_2.weight", H, 64); mat("model.visual.final_norm.weight", H)
    mat("model.audio.encoder.weight", 32, H); mat("model.audio.final_norm.weight", H)
    if edit:
        edit(t)
    names = sorted(t)
    mtp = [n for n in names if n.startswith(MTP)]
    main = [n for n in names if not n.startswith(MTP)]
    half = len(main) // 2
    wm = {}
    for fn, part in (("model-00001-of-00002.safetensors", main[:half]), ("model-00002-of-00002.safetensors", main[half:]),
                     ("mtp.safetensors", mtp)):
        write_st(os.path.join(root, fn), {n: t[n] for n in part})
        wm.update({n: fn for n in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    return t


def make_exl3(root, rng, K=3, expert_K=None, head_K=None):
    """The EXL3 pack of the main layers, under the L294 key names (attention as stored; the w13 split); with head_K,
    the head too, its V outputs stored at the 128 multiple as exllamav3 pads them (L295 F7)."""
    t = {}

    def lin(key, k, n, kk):
        for s, v in exl3_parts(rng, k, n, kk).items():
            t[f"{key}.{s}"] = v
    if head_K:
        lin(LLM + "unembed", H, exl3_rates.out_stored(V), head_K)
    for i in range(L):
        p = f"{LLM}layers.{i}."
        for nm, out, inp in (("wq_du", NQ * HD, H), ("wk_dv", NKV * HD, H), ("wv_dv", NKV * HD, H),
                             ("wr_du", NQ * DREL, H), ("wo_ud", H, NQ * HD)):
            lin(p + "attn." + nm, inp, out, K)
        if i < NDENSE:
            lin(p + "mlp.gate_proj", H, DFF, K); lin(p + "mlp.up_proj", H, DFF, K); lin(p + "mlp.w2_md", DFF, H, K)
            continue
        for j in range(S):
            q = f"{p}mlp.shared_experts.{j}."
            lin(q + "gate_proj", H, FF, K); lin(q + "up_proj", H, FF, K); lin(q + "down_proj", FF, H, K)
        for e in range(E):
            q = f"{p}mlp.experts.{e}."
            ek = expert_K or K
            lin(q + "gate_proj", H, FF, ek); lin(q + "up_proj", H, FF, ek); lin(q + "down_proj", FF, H, ek)
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
    """test_deepseek.digest over the shards and the index (the side files are checked byte-equal on their own)."""
    h = hashlib.sha256()
    for f in sorted(x for x in os.listdir(d) if x not in SIDE):
        h.update(f.encode() + b"\0" + hashlib.sha256(open(os.path.join(d, f), "rb").read()).digest())
    return h.hexdigest()


def sides_equal(o, tok_dir, hf):
    src = {"tokenizer.model": os.path.join(tok_dir, "tiktoken", "tokenizer.model"),
           "tokenizer_config.json": os.path.join(tok_dir, "tokenizer_config.json"),
           "config.json": os.path.join(hf, "config.json")}
    return all(os.path.isfile(os.path.join(o, n)) and open(os.path.join(o, n), "rb").read() == open(p, "rb").read()
               for n, p in src.items())


def recipe(tmp, label, rows):
    p = os.path.join(tmp, f"recipe-{label}.json")
    json.dump({"recipe": f"test-{label}", "model_type": "inkling_mm_model", "rows": rows}, open(p, "w"))
    return p


def build_grade(tmp, label, args, n_files):
    o1, o2 = os.path.join(tmp, f"{label}-1"), os.path.join(tmp, f"{label}-2")
    ok, out = run("plan", *args)
    check(ok, f"{label}: plan")
    if ok:
        print("    " + "\n    ".join(out.strip().splitlines()[-3:]))
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
    rng = np.random.default_rng(294)
    tmp = tempfile.mkdtemp(prefix="inkling-builder-test-")
    try:
        hf = os.path.join(tmp, "hf")
        t = make_hf(hf, rng)
        ex = os.path.join(tmp, "exl3")
        ext = make_exl3(ex, np.random.default_rng(7))
        rbf = os.path.join(HERE, "format-maps", "inkling-bf16.json")
        n_files = 1 + L + 1 + NMTP

        print("BF16 reference build:")
        o = build_grade(tmp, "bf16", ["--hf", hf, "--recipe", rbf], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["bf16"], f"GOLDEN bf16: {got}")
        check(sides_equal(o, hf, hf), "side files: tokenizer.model == tiktoken/tokenizer.model, tokenizer_config.json, "
                                      "config.json, as shipped")
        shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
        f_l1, f_l0 = shards[2], shards[1]

        def rows(raw, n_rows, start):
            return np.frombuffer(raw, np.uint8).reshape(n_rows, -1)[start::2].tobytes()
        w13 = t[LLM + "layers.1.mlp.experts.w13_weight"][2]
        per = len(w13) // E
        sw13 = t[LLM + "layers.1.mlp.shared_experts.shared_w13_weight"][2]
        sper = len(sw13) // S
        w2 = t[LLM + "layers.1.mlp.experts.w2_weight"][2]
        dn = t[LLM + "layers.0.mlp.w13_dn.weight"][2]
        p1 = LLM + "layers.1.mlp."
        check(entry_bytes(o, f_l1, p1 + "experts.2.gate_proj.weight") == rows(w13[2 * per:3 * per], 2 * FF, 0)
              and entry_bytes(o, f_l1, p1 + "experts.2.up_proj.weight") == rows(w13[2 * per:3 * per], 2 * FF, 1)
              and entry_bytes(o, f_l1, p1 + "experts.3.down_proj.weight") == w2[3 * len(w2) // E:],
              "routed expert 2 gate / up == the even / odd rows of its w13; expert 3 down == its w2 slice")
        check(entry_bytes(o, f_l1, p1 + "shared_experts.1.up_proj.weight") == rows(sw13[sper:], 2 * FF, 1)
              and entry_bytes(o, f_l0, LLM + "layers.0.mlp.gate_proj.weight") == rows(dn, 2 * DFF, 0)
              and entry_bytes(o, f_l0, LLM + "layers.0.mlp.up_proj.weight") == rows(dn, 2 * DFF, 1),
              "shared expert 1 up and the dense gate / up: the same interleave")
        h0, _ = header(os.path.join(o, shards[0]))
        meta0 = h0["__metadata__"]
        kv = {e["key"]: e["value"] for e in json.loads(meta0["pulsar.kv"])}
        check(kv["general.architecture"] == "inkling" and kv["inkling.local_layer_ids"]["v"] == [0, 1]
              and kv["inkling.mtp.num_nextn_predict_layers"] == NMTP and kv["inkling.vision.n_layers"] == 3
              and kv["inkling.eos_token_id"] == 61 and kv["pulsar.mtp_present"] is True
              and kv["pulsar.vision_present"] is True and not any(k.startswith("inkling.audio") for k in kv),
              "pulsar.kv: text / vision / mtp configs verbatim, eos, presence; no audio")
        allnames = set()
        for f in shards:
            allnames |= set(header(os.path.join(o, f))[0])
        check(not any(n.startswith("model.audio.") or "w13" in n or "w2_weight" in n for n in allnames),
              "no model.audio.* and no fused w13 / w2 stack in the container")
        h1, _ = header(os.path.join(o, f_l1))
        decl = json.loads(h1["__metadata__"]["pulsar.tensors"])
        fams = json.loads(h1["__metadata__"]["pulsar.experts"])
        a1 = LLM + "layers.1.attn."
        check(decl[a1 + "k_sconv.weight"]["dims_ne"] == [CK, NKV * HD]
              and decl[a1 + "k_sconv.weight"]["role"] == "short_conv_k"
              and decl[a1 + "wr_du.weight"]["role"] == "attn_rel_proj"
              and decl[a1 + "rel_logits_proj.proj"]["dims_ne"] == [WIN, DREL]
              and decl[p1 + "gate.bias"] == {"layout": "f32", "dims_ne": [E], "gguf_name": p1 + "gate.bias",
                                             "role": "router_select_bias", "block": "layers", "layer": 1}
              and decl[p1 + "shared_experts.1.down_proj.weight"]["index"] == 1,
              "tensor records: role / block / layer (index for a shared expert); short conv declared [C, K]")
        check(sorted(x["part"] for x in fams) == ["down_proj", "gate_proj", "up_proj"]
              and all(x["layout"] == "bf16" and x["layer"] == 1 and x["n_experts"] == E
                      and x["entry_name"] == p1 + "experts.{e}." + x["part"] + ".weight"
                      and x["role"] == "expert_" + x["part"].split("_")[0] for x in fams)
              and h1["__metadata__"]["pulsar.expert_dtype"] == "bf16",
              "routed families: gate / up / down, bf16, entry_name + layer + role; expert_dtype bf16")
        hm, _ = header(os.path.join(o, shards[-1]))
        check(hm["__metadata__"]["pulsar.shard_key"] == "mtp.1" and
              json.loads(hm["__metadata__"]["pulsar.tensors"])[MTP + "layers.1.input_proj.weight"]["role"]
              == "draft_in_proj", "drafter shards mtp.0 / mtp.1 last; draft_in_proj")

        print("K3 EXL3 build:")
        rk3 = recipe(tmp, "k3", [
            {"role": "dense", "block": "layers", "format": "exl3m_k3"}, {"role": "dense", "block": "mtp", "format": "native"},
            *[{"role": r, "format": "exl3m_k3"} for r in ("shared_expert", "expert_gate", "expert_up", "expert_down")],
            *[{"role": r, "format": "native"} for r in ("router", "norm", "embed", "head", "other", "vision")]])
        o = build_grade(tmp, "k3", ["--hf", hf, "--exl3", ex, "--recipe", rk3], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["k3"], f"GOLDEN k3: {got}")
        key = p1 + "experts.2.up_proj"
        key_d = LLM + "layers.0.mlp.w2_md"
        check(entry_bytes(o, f_l1, key + ".weight") == b"".join(ext[f"{key}.{s}"][2] for s in ("trellis", "suh", "svh"))
              and entry_bytes(o, f_l0, key_d + ".weight") == b"".join(ext[f"{key_d}.{s}"][2]
                                                                      for s in ("trellis", "suh", "svh")),
              "EXL3 entries == the pack's [trellis | suh | svh]")
        fams = json.loads(header(os.path.join(o, f_l1))[0]["__metadata__"]["pulsar.experts"])
        check(all(x["layout"] == "exl3m_k3" for x in fams), "routed families exl3m_k3")

        print("drafter and tower omitted:")
        rom = recipe(tmp, "omit", [*[{"role": r, "block": "layers", "format": "native"} for r in (
            "dense", "norm", "other")], *[{"role": r, "block": "mtp", "format": "omit"} for r in ("dense", "norm", "other")],
            *[{"role": r, "format": "native"} for r in ("shared_expert", "expert_gate", "expert_up", "expert_down",
                                                        "router", "embed", "head")],
            # the top-level norms belong to no block: name rows say them
            {"name": LLM + "embed_norm.weight", "format": "native"}, {"name": LLM + "norm.weight", "format": "native"},
            {"role": "vision", "format": "omit"}])
        o = os.path.join(tmp, "omit")
        ok, out = run("emit", "--hf", hf, "--recipe", rom, "--out", o, "--all")
        files = sorted(f for f in os.listdir(o) if f.endswith(".safetensors")) if ok else []
        kvo = {e["key"]: e["value"] for e in json.loads(header(os.path.join(o, files[0]))[0]["__metadata__"]
                                                         ["pulsar.kv"])} if files else {}
        check(ok and len(files) == 1 + L + 1 and kvo.get("pulsar.mtp_present") is False
              and kvo.get("pulsar.vision_present") is False and "inkling.vision.n_layers" not in kvo,
              "no mtp shards, pulsar.mtp_present / vision_present false")
        ok, out = run("verify", "--hf", hf, "--recipe", rom, "--out", o, "--all")
        check(ok and "failing: 0" in out, "verify PASS (omitted blocks)")

        print("refusals:")
        cases = [
            ("an unknown tensor", None, lambda t: t.__setitem__(LLM + "layers.0.attn.extra.weight", t[LLM + "norm.weight"]),
             "does not map"),
            ("a missing tensor", None, lambda t: t.pop(LLM + "layers.2.attn.q_norm.weight"), "missing from"),
            ("a global layer's relative basis at the window extent", None,
             lambda t: t.__setitem__(LLM + "layers.2.attn.rel_logits_proj.proj",
                                     t[LLM + "layers.1.attn.rel_logits_proj.proj"]), "disagree with the declared shape"),
            ("a missing fused stack", None, lambda t: t.pop(LLM + "layers.2.mlp.experts.w2_weight"), "missing from"),
        ]
        c = config()
        c["text_config"]["use_sconv"] = False
        cases.append(("use_sconv false", c, None, "assumes True"))
        c = config()
        c["text_config"]["moe_intermediate_size"] = 3 * FF
        cases.append(("moe_intermediate_size != intermediate_size", c, None, "moe_intermediate_size"))
        c = config()
        c["text_config"]["local_layer_ids"] = [0, 1, 2]
        cases.append(("a layer kind the tensors contradict", c, None, "disagree with the declared shape"))
        for i, (label, cfg, edit, msg) in enumerate(cases):
            d = os.path.join(tmp, f"bad-{i}")
            make_hf(d, np.random.default_rng(1), cfg=copy.deepcopy(cfg) if cfg else None, edit=edit)
            ok, _ = run("plan", "--hf", d, "--recipe", rbf, expect_fail=True, contains=msg)
            check(ok, f"{label} -> refused")
        r = json.load(open(rbf))
        b1 = MTP + "layers.1.transformer_block."
        rows = [x for x in r["rows"] if x["role"] != "dense"] + [
            {"role": "dense", "block": "layers", "format": "native"},
            {"role": "dense", "block": "mtp", "layers": [0], "format": "native"},
            {"name": MTP + "layers.1.input_proj.weight", "format": "omit"},
            {"name": b1 + "attn.w*", "format": "native"}, {"name": b1 + "mlp.*_proj.weight", "format": "native"},
            {"name": b1 + "mlp.w2_md.weight", "format": "native"}]
        hp = recipe(tmp, "half-mtp2", rows)
        ok, _ = run("plan", "--hf", hf, "--recipe", hp, expect_fail=True, contains="all of it or none")
        check(ok, "half the drafter omitted -> refused")
        rr = recipe(tmp, "omit-router", [x for x in r["rows"] if x["role"] != "router"] + [
            {"role": "router", "format": "omit"}])
        ok, _ = run("plan", "--hf", hf, "--recipe", rr, expect_fail=True, contains="the Inkling loader binds")
        check(ok, "an omitted main-model tensor (the router) -> refused")
        ex5 = os.path.join(tmp, "exl3-k5")
        make_exl3(ex5, np.random.default_rng(7), expert_K=5)
        r5 = recipe(tmp, "k5", [{"role": "dense", "block": "layers", "format": "exl3m_k3"},
                                {"role": "dense", "block": "mtp", "format": "native"},
                                {"role": "shared_expert", "format": "exl3m_k3"},
                                *[{"role": r, "format": "exl3m_k5"} for r in ("expert_gate", "expert_up", "expert_down")],
                                *[{"role": r, "format": "native"} for r in ("router", "norm", "embed", "head", "other",
                                                                             "vision")]])
        ok, _ = run("plan", "--hf", hf, "--exl3", ex5, "--recipe", r5)
        check(ok, "routed gate / up at K5 (split: the pair arm's K5, L295 F7) -> planned")
        ex8 = os.path.join(tmp, "exl3-k8")
        make_exl3(ex8, np.random.default_rng(8), expert_K=8)
        r8 = recipe(tmp, "k8", [x if not x["role"].startswith("expert_") else {**x, "format": "exl3m_k8"}
                                for x in json.load(open(r5))["rows"]])
        ok, _ = run("plan", "--hf", hf, "--exl3", ex8, "--recipe", r8, expect_fail=True, contains="exl3_arm_has_rate")
        check(ok, "routed experts at K8 (no routed arm reads K8) -> refused by the rate table")
        print("the pack's K6 head (L295 F7: stored at the 128 multiple, the logical width declared):")
        exh = os.path.join(tmp, "exl3-head")
        exth = make_exl3(exh, np.random.default_rng(11), head_K=6)
        rh = recipe(tmp, "k6head", [x if x["role"] != "head" else {**x, "format": "exl3m_k6"}
                                    for x in json.load(open(rk3))["rows"]])
        oh = build_grade(tmp, "k6head", ["--hf", hf, "--exl3", exh, "--recipe", rh], n_files)
        fh = [f for f in sorted(os.listdir(oh)) if f.endswith(".safetensors")
              and LLM + "unembed.weight" in header(os.path.join(oh, f))[0]]
        hd = json.loads(header(os.path.join(oh, fh[0]))[0]["__metadata__"]["pulsar.tensors"])[LLM + "unembed.weight"] \
            if fh else {}
        check(bool(fh) and hd.get("layout") == "exl3m_k6" and hd.get("dims_ne") == [H, V]
              and entry_bytes(oh, fh[0], LLM + "unembed.weight") == b"".join(exth[f"{LLM}unembed.{s}"][2]
                                                                          for s in ("trellis", "suh", "svh")),
              f"head exl3m_k6, declared [{H}, {V}] (logical), its bytes the pack's [trellis | suh | svh] at the stored "
              f"{exl3_rates.out_stored(V)} outputs")
        print("side files:")
        bare = os.path.join(tmp, "hf-no-tokenizer")   # the mini's case: a checkpoint without tiktoken/
        shutil.copytree(hf, bare, ignore=shutil.ignore_patterns("tiktoken", "tokenizer_config.json"))
        tok = os.path.join(tmp, "tok")
        os.makedirs(os.path.join(tok, "tiktoken"))
        open(os.path.join(tok, "tiktoken", "tokenizer.model"), "wb").write(TOK_MODEL + b"Iw== 2\n")
        open(os.path.join(tok, "tokenizer_config.json"), "wb").write(TOK_CONFIG)
        o = os.path.join(tmp, "bare")
        ok, _ = run("emit", "--hf", bare, "--recipe", rom, "--out", o, "--all", expect_fail=True, contains="--tokenizer")
        check(ok and not os.path.exists(o), "a checkpoint without its tokenizer -> refused naming --tokenizer, before "
                                            "any shard is written")
        ok, _ = run("emit", "--hf", bare, "--recipe", rom, "--tokenizer", tok, "--out", o, "--all")
        check(ok and sides_equal(o, tok, bare), "--tokenizer DIR: the tokenizer files from DIR, config.json the "
                                                "checkpoint's")
        open(os.path.join(o, "tokenizer.model"), "ab").write(b"x")
        ok, out = run("verify", "--hf", bare, "--recipe", rom, "--tokenizer", tok, "--out", o, "--all",
                      expect_fail=True, contains="side file tokenizer.model: MISSING or DIFFERS")
        check(ok, "verify --all: a side file that differs -> FAIL")
    finally:
        shutil.rmtree(tmp)
    print(f"test_inkling: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

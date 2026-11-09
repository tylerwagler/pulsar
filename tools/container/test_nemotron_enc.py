#!/usr/bin/env python3
"""test_nemotron_enc.py -- the Nemotron-3.5 retrieval encoders in the artifact builder (nemotron_enc.py, L321), on
synthetic Embed and Rerank checkpoints (tiny dims, the real layout: text_config nested, `language_model.` (Embed) /
`model.language_model.` (Rerank) prefixes, the Reranker's `language_model.score.weight`, vision tensors):

  * Embed, BF16: plan, emit --all twice byte-identical, verify --all (side files byte-equal), audit; GOLDEN
  * the records: attn_norm / attn_q..o / ffn_norm / ffn_gate..down per layer (block layers), embed / output_norm at the
    top; pulsar.kv: text_config verbatim (the rope flattened), pulsar.encoder / pooling / normalize; no vision tensor
  * Embed, EXL3 (every trunk Linear exl3m_k4) from a ONE-FILE pack (model.safetensors) keyed as the staged trunk is
    (`model.layers.N.*`): every Linear's bytes == the pack's [trellis | suh | svh]
  * Rerank, EXL3: the score head `language_model.score.weight` bf16 as stored, role score; pulsar.encoder rerank,
    pulsar.normalize false
  * refusals by name: an unknown tensor, a missing tensor, a wrong shape, a switch (is_causal true), the rope (YaRN
    factor 8), a pack keyed under the checkpoint's names, a recipe row that matches nothing (a head row on Embed)
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
# sha256 over (file name, sha256(file)) of every emitted file but the side files; fixed when the family landed (L321)
GOLDEN = {
    "embed-bf16": "36c18dd50435dea9e686f016d4aaed27ca7dc2f8e8e4a919662c9cbfb170649c",
}
H, V, L, NQ, NKV, HD, FF = 128, 64, 2, 4, 2, 32, 256
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
LMP = {"embed": "language_model.", "rerank": "model.language_model."}


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def config(kind):
    rope = {"apply_yarn_scaling": False, "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0,
            "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384, "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"}
    text = {"architectures": ["Ministral3Model"], "model_type": "ministral3", "is_causal": False, "hidden_act": "silu",
            "hidden_size": H, "intermediate_size": FF, "num_hidden_layers": L, "num_attention_heads": NQ,
            "num_key_value_heads": NKV, "head_dim": HD, "rms_norm_eps": 1e-5, "vocab_size": V,
            "max_position_embeddings": 262144, "tie_word_embeddings": False, "sliding_window": None,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 11, "pooling": "avg", "use_cache": False,
            "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16384}, "rope_parameters": rope}
    top = {"architectures": ["Mistral3Model" if kind == "embed" else "Mistral3ForSequenceClassification"],
           "model_type": "mistral3", "is_causal": False, "pooling": "avg", "image_token_index": 10,
           "spatial_merge_size": 2, "tie_word_embeddings": False, "text_config": text,
           "vision_config": {"model_type": "pixtral", "hidden_size": 32, "architectures": None}}
    return top


def make_hf(root, rng, kind, cfg=None, edit=None):
    os.makedirs(root, exist_ok=True)
    json.dump(cfg or config(kind), open(os.path.join(root, "config.json"), "w"))
    for f in SIDE[:-1]:
        open(os.path.join(root, f), "w").write(f"{{\"synthetic\": \"{f}\"}}\n")
    t = {}

    def mat(name, *shape, dt="BF16"):
        a = rng.standard_normal(shape).astype(np.float32) * 0.05
        t[name] = (dt, list(shape), bf16(a).tobytes() if dt == "BF16" else a.tobytes())

    p = LMP[kind]
    for il in range(L):
        q = f"{p}layers.{il}."
        mat(q + "input_layernorm.weight", H); mat(q + "post_attention_layernorm.weight", H)
        mat(q + "self_attn.q_proj.weight", NQ * HD, H); mat(q + "self_attn.k_proj.weight", NKV * HD, H)
        mat(q + "self_attn.v_proj.weight", NKV * HD, H); mat(q + "self_attn.o_proj.weight", H, NQ * HD)
        mat(q + "mlp.gate_proj.weight", FF, H); mat(q + "mlp.up_proj.weight", FF, H); mat(q + "mlp.down_proj.weight", H, FF)
    mat(p + "embed_tokens.weight", V, H); mat(p + "norm.weight", H)
    if kind == "rerank":
        mat("language_model.score.weight", 1, H)
    vp = "model." if kind == "rerank" else ""
    mat(vp + "vision_tower.transformer.layers.0.attention.q_proj.weight", 32, 32)
    mat(vp + "multi_modal_projector.linear_1.weight", 64, 32)
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


LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def make_exl3(root, hf_t, rng, kind, K=4, hf_keys=False):
    """The staged trunk's EXL3 pack (retriever/exl3/stage_text_trunk.py + exllamav3): every trunk Linear keyed
    `model.layers.N.<linear>` (`hf_keys`: under the checkpoint's own names, which the builder must refuse), in ONE
    model.safetensors as exllamav3 writes a small pack."""
    t = {}
    p = LMP[kind]
    for name, (_dt, shape, _b) in hf_t.items():
        if name.startswith(p + "layers.") and any(name.endswith(f"{d}.weight") for d in LINEARS):
            out, inp = shape
            key = name[:-len(".weight")] if hf_keys else "model." + name[len(p):-len(".weight")]
            for s, v in exl3_parts(rng, inp, (out + 127) // 128 * 128, K).items():
                t[f"{key}.{s}"] = v
    os.makedirs(root, exist_ok=True)
    write_st(os.path.join(root, "model.safetensors"), t)
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
    json.dump({"recipe": f"test-{label}", "model_type": "mistral3", "rows": rows}, open(p, "w"))
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


def records(o):
    shards = sorted(f for f in os.listdir(o) if f.endswith(".safetensors"))
    decl, names = {}, set()
    for f in shards:
        h, _ = header(os.path.join(o, f))
        names |= set(h) - {"__metadata__"}
        decl.update(json.loads(h["__metadata__"]["pulsar.tensors"]))
    hv, _ = header(os.path.join(o, shards[0]))
    kv = {e["key"]: e["value"] for e in json.loads(hv["__metadata__"]["pulsar.kv"])}
    return shards, decl, names, kv, hv


def main():
    rng = np.random.default_rng(321)
    tmp = tempfile.mkdtemp(prefix="nemotron-enc-builder-test-")
    try:
        n_files = 1 + L + 1                 # vision (kv only) + layers + top
        ROWS_BF16 = [{"role": "dense", "format": "native"}, {"role": "norm", "format": "native"},
                     {"role": "embed", "format": "native"}]
        ROWS_K4 = [{"role": "dense", "format": "exl3m_k4"}, {"role": "norm", "format": "native"},
                   {"role": "embed", "format": "native"}]

        print("Embed, BF16:")
        hf = os.path.join(tmp, "embed-hf")
        t = make_hf(hf, rng, "embed")
        o = build_grade(tmp, "embed-bf16", ["--hf", hf, "--recipe", recipe(tmp, "e-bf16", ROWS_BF16)], n_files)
        got = shard_digest(o)
        check(got == GOLDEN["embed-bf16"], f"GOLDEN embed-bf16: {got}")
        shards, decl, names, kv, hv = records(o)
        p = LMP["embed"]
        check(decl[p + "layers.1.self_attn.q_proj.weight"]["role"] == "attn_q"
              and decl[p + "layers.1.self_attn.q_proj.weight"]["layer"] == 1
              and decl[p + "layers.1.self_attn.q_proj.weight"]["block"] == "layers"
              and decl[p + "layers.0.input_layernorm.weight"]["role"] == "attn_norm"
              and decl[p + "layers.0.post_attention_layernorm.weight"]["role"] == "ffn_norm"
              and decl[p + "layers.0.mlp.down_proj.weight"]["role"] == "ffn_down"
              and decl[p + "layers.0.mlp.down_proj.weight"]["dims_ne"] == [FF, H]
              and decl[p + "embed_tokens.weight"]["role"] == "embed" and "block" not in decl[p + "embed_tokens.weight"]
              and decl[p + "norm.weight"]["role"] == "output_norm",
              "records: attn / ffn roles per layer (block layers, layer N), embed / output_norm at the top")
        check(kv["general.architecture"] == "nemotron_enc" and kv["pulsar.encoder"] == "embed"
              and kv["pulsar.pooling"] == "mean" and kv["pulsar.normalize"] is True
              and kv["nemotron_enc.is_causal"] is False and kv["nemotron_enc.rope_parameters.factor"] == 16.0
              and kv["nemotron_enc.rope_parameters.original_max_position_embeddings"] == 16384
              and kv["nemotron_enc.llama_4_scaling.beta"] == float(np.float32(0.1)) and kv["pulsar.vision_present"] is False
              and [k for k in hv if k != "__metadata__"] == [],
              "pulsar.kv: text_config verbatim (rope flattened), encoder embed / mean / normalize; the primary shard "
              "holds no tensor")
        check(not any(".vision_tower." in n or n.startswith(("vision_tower.", "multi_modal_projector."))
                      for n in names), "no vision tensor in the container (consumed by name)")

        print("Embed, EXL3 K4 from a one-file pack keyed as the staged trunk:")
        pk = os.path.join(tmp, "embed-pack")
        pt = make_exl3(pk, t, rng, "embed")
        o = build_grade(tmp, "embed-k4", ["--hf", hf, "--exl3", pk, "--recipe", recipe(tmp, "e-k4", ROWS_K4)], n_files)
        shards, decl, _names, _kv, _hv = records(o)
        bad = 0
        for il in range(L):
            for lin in LINEARS:
                n = f"{p}layers.{il}.{lin}.weight"
                key = f"model.layers.{il}.{lin}"
                want = b"".join(pt[f"{key}.{s}"][2] for s in ("trellis", "suh", "svh"))
                f = next(x for x in shards if n in header(os.path.join(o, x))[0])
                bad += decl[n]["layout"] != "exl3m_k4" or entry_bytes(o, f, n) != want
        check(bad == 0, f"every trunk Linear exl3m_k4, bytes == the pack's [trellis | suh | svh] ({L * len(LINEARS)})")

        print("Rerank, EXL3 K4:")
        hr = os.path.join(tmp, "rerank-hf")
        tr = make_hf(hr, rng, "rerank")
        pr = os.path.join(tmp, "rerank-pack")
        make_exl3(pr, tr, rng, "rerank")
        rows = ROWS_K4 + [{"role": "head", "format": "native"}]
        o = build_grade(tmp, "rerank-k4", ["--hf", hr, "--exl3", pr, "--recipe", recipe(tmp, "r-k4", rows)], n_files)
        shards, decl, _names, kv, _hv = records(o)
        sc = "language_model.score.weight"
        f = next(x for x in shards if sc in header(os.path.join(o, x))[0])
        check(decl[sc]["role"] == "score" and decl[sc]["layout"] == "bf16" and decl[sc]["dims_ne"] == [H, 1]
              and entry_bytes(o, f, sc) == tr[sc][2] and kv["pulsar.encoder"] == "rerank"
              and kv["pulsar.normalize"] is False
              and decl["model.language_model.layers.0.self_attn.k_proj.weight"]["layout"] == "exl3m_k4",
              "score head bf16 as stored, role score [H -> 1]; pulsar.encoder rerank, normalize false")

        print("refusals:")
        rbf = recipe(tmp, "e-bf16b", ROWS_BF16)

        def refused(label, kind, contains, cfg=None, edit=None, extra=(), rec=None):
            d = os.path.join(tmp, "bad-" + label.replace(" ", "-"))
            make_hf(d, np.random.default_rng(7), kind, cfg=cfg, edit=edit)
            ok, _ = run("plan", "--hf", d, "--recipe", rec or rbf, *extra, expect_fail=True, contains=contains)
            check(ok, f"refused: {label}")
            shutil.rmtree(d, ignore_errors=True)

        def put(name, *shape):
            def e(t):
                t[name] = ("BF16", list(shape), bf16(np.zeros(shape, np.float32)).tobytes())
            return e

        def drop(name):
            def e(t):
                del t[name]
            return e

        def shp(name, *shape):
            return put(name, *shape)

        refused("an unknown tensor", "embed", "does not map", edit=put(p + "layers.0.mlp.extra.weight", 4))
        refused("a missing tensor", "embed", "missing", edit=drop(p + "layers.1.mlp.up_proj.weight"))
        refused("a wrong shape", "embed", "declared shape", edit=shp(p + "layers.0.self_attn.k_proj.weight", NQ * HD, H))
        c = config("embed"); c["text_config"]["is_causal"] = True
        refused("a causal trunk (is_causal true)", "embed", "is_causal", cfg=c)
        c = config("embed"); c["text_config"]["rope_parameters"]["factor"] = 8.0
        refused("the rope (YaRN factor 8)", "embed", "factor", cfg=c)
        bad_pack = os.path.join(tmp, "pack-hfkeys")
        make_exl3(bad_pack, t, rng, "embed", hf_keys=True)
        ok, _ = run("plan", "--hf", hf, "--exl3", bad_pack, "--recipe", recipe(tmp, "e-k4b", ROWS_K4), expect_fail=True)
        check(ok, "refused: a pack keyed under the checkpoint's names (not the staged trunk's)")
        ok, _ = run("plan", "--hf", hf, "--recipe", recipe(tmp, "e-head", ROWS_BF16 + [{"role": "head", "format": "native"}]),
                    expect_fail=True)
        check(ok, "refused: a recipe row that matches nothing (a head row on Embed)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'test_nemotron_enc: PASS' if not FAILS else 'test_nemotron_enc: FAIL (' + str(len(FAILS)) + ')'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

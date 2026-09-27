"""qwen.py -- the qwen4_exp family in the direct builder (L251 S6): Qwen3.8-Flash-Next -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 HF checkpoint (the authority for every BF16 tensor, every MXFP8 tensor's input, the config and
           the tokenizer)
  --exl3   an exllamav3-layout checkpoint holding the EXL3 tensors the recipe names: <linear>.{trellis,suh,svh,mul1}
           under the HF Linear name (dense: minus ".weight"; routed experts: mlp.experts.<e>.<part>).  Ours
           (research/l251/container/qwen_exl3_quant.py: gate_up fused) or turboderp's (gate / up separate).
  --exl3-experts   optional: a DIFFERENT exllamav3 checkpoint for the routed experts only (e.g. turboderp's K4
           experts beside our K5 dense).  Named explicitly; nothing falls back from one source to the other.
  --recipe the per-tensor format map (format-maps/qwen38fn-*.json): every checkpoint tensor matches EXACTLY ONE row
           (fnmatch), every row matches something; a tensor no row names refuses.
  --ple-rows  the PLE row-file manifest (ple_rows.py build): its header facts become pulsar.kv entries so the
           loader can refuse a mismatched table.

What the container holds (the loader's contract, published in pulsar-notes research/l251/container/README.md):
  * names ARE the HF names.  Routed experts fold to one U8 entry per expert-projection,
    `model.language_model.layers.L.mlp.experts.<e>.<part>.weight`, contiguous in expert order, part in
    {gate_up_proj} (fused) or {gate_proj, up_proj} (split) plus {down_proj}; each family in pulsar.experts carries
    `entry_name` (the per-expert name with "{e}") and `layer`, because the DeepSeek rule the engine derives names
    by (`layers.%d.ffn.experts.%llu.%s.weight` from `blk.N.`) does not describe these names.
  * gguf_name == the HF name (dense) / the HF stack name `...mlp.experts.<part>` (families): the Qwen family binds
    by the checkpoint's own names -- no second name table.  A family materialises as one tensor of dims
    dims_per_expert_ne + [n_experts] = the HF stack shape reversed.
  * pulsar.kv (S1's contract, src/engine/family_qwen.h section 2): general.architecture = "qwen4_exp"; the HF
    text_config VERBATIM under `qwen4_exp.` (config_kvs); the PLE int64 buffers as u64 arrays
    qwen4_exp.ple_{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}; the builder's own facts under
    `pulsar.*` (recipe + sha256, sources, expert_gate_up, mtp/vision presence, the PLE row file's name, rows,
    record bytes, dtype and SHA-256); the tokenizer from the checkpoint's own files.
  * layouts: bf16 (native), mxfp8_lt (U8, E4M3 [out][in] + swizzled E8M0, padded per bytes_for), exl3m_k4 /
    exl3m_k5 (U8 [trellis | suh | svh], dims_ne [in, out]).  Declared shapes are the SOURCE shapes (no reshapes).
  * shards: `vision` (primary: pulsar.kv + the vision tower, BF16 native), `layers.0..47`, `top` (embed, head,
    top-level mixer).  MTP is omitted by the recipe (no lane yet); pulsar.kv says so.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import struct

from hf_source import EXL3_MUL1  # noqa: F401  (the codebook the EXL3 sources are checked against)
import kv as KV
import producers as PR

FAMILY = "qwen4_exp"
PFX = "model.language_model."
EXL3_RATES = {64: "exl3m_k4", 80: "exl3m_k5"}          # words per 16x16 tile -> layout (16 K words)
FORMATS = {"bf16", "mxfp8_lt", "exl3m_k4", "exl3m_k5", "ple_rows", "kv", "omit"}
EXPERT_STACKS = ("gate_up_proj", "down_proj")


def is_qwen(hf) -> bool:
    return hf.config["top_level"].get("model_type") == FAMILY


# ---------------------------------------------------------------------------
# the recipe
# ---------------------------------------------------------------------------
class Recipe:
    def __init__(self, path):
        self.path = path
        r = json.load(open(path))
        if r.get("model_type") != FAMILY:
            raise SystemExit(f"{path}: recipe is for {r.get('model_type')!r}, not {FAMILY}")
        self.name = r["recipe"]
        self.gate_up = r["expert_gate_up"]
        if self.gate_up not in ("fused", "split"):
            raise SystemExit(f"{path}: expert_gate_up {self.gate_up!r} is neither 'fused' nor 'split'")
        self.rows = [tuple(x) for x in r["rows"]]
        for pat, fmt in self.rows:
            if fmt not in FORMATS:
                raise SystemExit(f"{path}: row {pat!r} names format {fmt!r} ({sorted(FORMATS)})")
        self.sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()

    def resolve(self, names):
        """name -> format for every checkpoint name, refusing unnamed / doubly-named tensors and dead rows."""
        out, used = {}, set()
        for n in names:
            hits = [(p, f) for p, f in self.rows if fnmatch.fnmatchcase(n, p)]
            if not hits:
                raise SystemExit(f"{n}: no row of {self.path} names this tensor's format -- refusing")
            if len(hits) > 1:
                raise SystemExit(f"{n}: {len(hits)} rows of {self.path} match ({[p for p, _ in hits]}) -- refusing")
            out[n] = hits[0][1]
            used.add(hits[0][0])
        dead = [p for p, _ in self.rows if p not in used]
        if dead:
            raise SystemExit(f"{self.path}: rows that match no checkpoint tensor: {dead}")
        return out


# ---------------------------------------------------------------------------
# names -> shards
# ---------------------------------------------------------------------------
_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def shard_order(n_layer):
    return ["vision"] + [f"layers.{i}" for i in range(n_layer)] + ["top"]


def shard_file(order, shard):
    return f"model-{order.index(shard) + 1:05d}-of-{len(order):05d}.safetensors"


def shard_of(name):
    m = _LAYER.match(name)
    if m:
        return f"layers.{int(m.group(1))}"
    if name.startswith("model.visual."):
        return "vision"
    if name in ("lm_head.weight", f"{PFX}embed_tokens.weight") or name.startswith(f"{PFX}hyper_connection_mixer."):
        return "top"
    raise SystemExit(f"{name}: no shard for this name")


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _exl3_entry(src, key, k, n, want_layout):
    ranges, words = src.linear(key, k, n, EXL3_RATES)
    if EXL3_RATES[words] != want_layout:
        raise SystemExit(f"{key}: the recipe names {want_layout}, the EXL3 source holds {EXL3_RATES[words]} -- refusing")
    nbytes = PR.bytes_for(want_layout, [k, n])
    if sum(r[2] for r in ranges) != nbytes:
        raise SystemExit(f"{key}: EXL3 source spans {sum(r[2] for r in ranges)} bytes, {want_layout} on [{k}, {n}] is {nbytes}")
    return ranges, nbytes


def plan(hf, exl3, exl3_experts, recipe, tokenizer_dir, ple_manifest):
    cfg = hf.config
    n_layer = int(cfg["num_hidden_layers"])
    order = shard_order(n_layer)
    files = {s: shard_file(order, s) for s in order}
    shards = {s: {"entries": [], "tensors": {}, "experts": []} for s in order}
    fmt = recipe.resolve(hf.names())
    consumed = {"ple_rows": 0, "kv": 0, "omit": 0}
    for name in hf.names():
        f = fmt[name]
        if f in consumed:
            consumed[f] += 1
            continue
        shard = shard_of(name)
        dtype, shape = hf.dtype(name), hf.shape(name)
        if f == "bf16":
            if dtype != "BF16":
                raise SystemExit(f"{name}: recipe says bf16, the checkpoint holds {dtype}")
            path, off, n = hf.span(name)
            entry = {"name": name, "layout": "bf16", "gguf_name": name, "dtype": "BF16", "shape": list(shape),
                     "nbytes": n, "src": ("ranges", [(path, off, n)])}
            dims_ne = list(reversed(shape))
        elif f == "mxfp8_lt":
            if dtype != "BF16" or len(shape) != 2:
                raise SystemExit(f"{name}: mxfp8_lt from a BF16 matrix, the checkpoint holds {dtype} {shape}")
            out, inp = shape
            dims_ne = [inp, out]
            nbytes = PR.bytes_for("mxfp8_lt", dims_ne)
            entry = {"name": name, "layout": "mxfp8_lt", "gguf_name": name, "dtype": "U8", "shape": [nbytes],
                     "nbytes": nbytes, "bf16_src": name,
                     "src": ("produce", (lambda w=name, o=out, i=inp: PR.mxfp8_lt_from_bf16(hf.raw(w), o, i)))}
        elif f.startswith("exl3m_") and ".mlp.experts." not in name:
            if exl3 is None:
                raise SystemExit(f"{name}: the recipe names {f}; pass --exl3")
            if dtype != "BF16" or len(shape) != 2 or not name.endswith(".weight"):
                raise SystemExit(f"{name}: EXL3 dense Linear from a BF16 [out, in] .weight, got {dtype} {shape}")
            out, inp = shape
            dims_ne = [inp, out]
            ranges, nbytes = _exl3_entry(exl3, name[:-len(".weight")], inp, out, f)
            entry = {"name": name, "layout": f, "gguf_name": name, "dtype": "U8", "shape": [nbytes],
                     "nbytes": nbytes, "src": ("ranges", ranges)}
        elif f.startswith("exl3m_"):
            _plan_experts(hf, exl3_experts or exl3, recipe, name, f, shards[shard])
            continue
        else:
            raise SystemExit(f"{name}: format {f} has no producer")
        shards[shard]["entries"].append(entry)
        shards[shard]["tensors"][name] = {"layout": entry["layout"], "dims_ne": dims_ne, "gguf_name": name}

    kvs = build_kv(hf, recipe, tokenizer_dir, ple_manifest, exl3, exl3_experts)
    kv_arch = [k for k in kvs if not k["key"].startswith("tokenizer.")]
    for s in order:
        p = shards[s]
        exp_layouts = {e["layout"] for e in p["experts"]}
        p["meta"] = {
            "format": "pt",
            "pulsar.format": "pulsar-safetensors-v1",
            "pulsar.family": FAMILY,
            "pulsar.alignment": "32",
            "pulsar.shard": files[s],
            "pulsar.shard_key": s,
            "pulsar.n_shards": str(len(order)),
            "pulsar.primary": files["vision"],
            "pulsar.tensors": json.dumps(p["tensors"], separators=(",", ":"), sort_keys=True),
            "pulsar.experts": json.dumps(p["experts"], separators=(",", ":")),
            "pulsar.kv_arch": json.dumps(kv_arch, separators=(",", ":")),
            "pulsar.expert_dtype": "none" if not exp_layouts else ("exl3" if all(
                x.startswith("exl3m_") for x in exp_layouts) else "mixed"),
        }
        if s == "vision":
            p["meta"]["pulsar.kv"] = json.dumps(kvs, separators=(",", ":"))
    shape = {"family": FAMILY, "n_layer": n_layer, "recipe": recipe.name}
    return shape, order, files, shards, sum(consumed.values())


def _plan_experts(hf, src, recipe, stack_name, layout, shard):
    """One HF expert stack ([E, out, in] BF16) -> its EXL3 families: gate_up_proj (fused) or gate_proj + up_proj
    (split, the recipe's word), down_proj."""
    if src is None:
        raise SystemExit(f"{stack_name}: the recipe names {layout}; pass --exl3 (or --exl3-experts)")
    m = re.match(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(gate_up_proj|down_proj)$", stack_name)
    if not m:
        raise SystemExit(f"{stack_name}: not a routed-expert stack")
    pre, layer, stack = m.group(1), int(m.group(2)), m.group(3)
    E, out, inp = hf.shape(stack_name)
    if stack == "gate_up_proj" and recipe.gate_up == "split":
        if out % 2:
            raise SystemExit(f"{stack_name}: odd gate_up width {out}")
        parts = [("gate_proj", out // 2), ("up_proj", out // 2)]
    else:
        parts = [(stack, out)]
    for part, n in parts:
        eb = PR.bytes_for(layout, [inp, n])
        fam_name = pre + part
        entry_name = pre + "{e}." + part + ".weight"
        for e in range(E):
            ranges, nbytes = _exl3_entry(src, f"{pre}{e}.{part}", inp, n, layout)
            assert nbytes == eb
            name = entry_name.replace("{e}", str(e))
            shard["entries"].append({"name": name, "layout": layout, "gguf_name": fam_name, "dtype": "U8",
                                     "shape": [eb], "nbytes": eb, "src": ("ranges", ranges)})
        shard["experts"].append({"gguf_name": fam_name, "part": part, "n_experts": E, "expert_bytes": eb,
                                 "layout": layout, "contiguous": True, "dims_per_expert_ne": [inp, n],
                                 "entry_name": entry_name, "layer": layer})


# ---------------------------------------------------------------------------
# pulsar.kv
# ---------------------------------------------------------------------------
def config_kvs(prefix, value, out):
    """The HF text_config -> (key, type, value) triples, mechanically (S1's contract, family_qwen.h section 2):
    every key verbatim under `prefix`, nested dicts flattened with '.', None and empty lists skipped (nothing to
    type); bool -> bool, int -> u32 (u64 past 2^32, i64 if negative), float -> f32, str -> string, a list of str
    -> string array, a list of non-negative int -> u32 array (u64 past 2^32).  Anything else refuses."""
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            config_kvs(f"{prefix}.{k}", v, out)
    elif isinstance(value, bool):
        out.append((prefix, "bool", value))
    elif isinstance(value, int):
        out.append((prefix, "u32" if 0 <= value <= 0xFFFFFFFF else "u64" if value >= 0 else "i64", value))
    elif isinstance(value, float):
        out.append((prefix, "f32", value))
    elif isinstance(value, str):
        out.append((prefix, "string", value))
    elif isinstance(value, list):
        if not value:
            return
        if all(isinstance(v, str) for v in value):
            out.append((prefix, "array", ("string", list(value))))
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in value) and min(value) >= 0:
            out.append((prefix, "array", ("u64" if max(value) > 0xFFFFFFFF else "u32", list(value))))
        else:
            raise SystemExit(f"{prefix}: unsupported list {value!r}")
    else:
        raise SystemExit(f"{prefix}: unsupported config value {value!r}")


# the three int64 buffers HF keeps under ple.ple_embedding: config, not weights (S1's contract names)
PLE_BUFFERS = {"layer_multipliers": "ple_layer_multipliers",
               "ngram_heads_vocab_sizes": "ple_ngram_heads_vocab_sizes",
               "ngram_heads_offsets": "ple_ngram_heads_offsets"}


def build_kv(hf, recipe, tokenizer_dir, ple_manifest, exl3, exl3_experts):
    """pulsar.kv for qwen4_exp: `general.*`; the text_config verbatim under `qwen4_exp.` (config_kvs); the PLE
    buffers as u64 arrays; the builder's own facts under `pulsar.*` (recipe, sources, the PLE row file the
    table must match); the tokenizer from the checkpoint's own files."""
    top = hf.config["top_level"]
    text = top["text_config"]
    A = FAMILY
    L = int(text["num_hidden_layers"])
    types = list(text["layer_types"])
    if len(types) != L or set(types) - {"linear_attention", "full_attention"}:
        raise SystemExit(f"layer_types: {len(types)} entries of {sorted(set(types))} for {L} layers")
    ple_ids = list(text["ple_layer_ids"])
    if len(ple_ids) != 1:
        raise SystemExit(f"ple_layer_ids {ple_ids}: one PLE layer expected")
    ple_layer = ple_ids[0] - 1          # 1-based in the config
    if ple_manifest is None:
        raise SystemExit("the PLE row file is part of the model: pass --ple-rows MANIFEST (ple_rows.py build)")
    man = json.load(open(ple_manifest))
    pre = f"{PFX}layers.{ple_layer}.ple.ple_embedding."
    kvs = [
        ("general.architecture", "string", A),
        ("general.type", "string", "model"),
        ("general.name", "string", "Qwen3.8-Flash-Next"),
    ]
    config_kvs(A, text, kvs)
    bufs = {}
    for hf_key, key in PLE_BUFFERS.items():
        name = pre + hf_key
        if hf.dtype(name) != "I64":
            raise SystemExit(f"{name}: dtype {hf.dtype(name)}, expected I64")
        raw = hf.raw(name)
        vals = list(struct.unpack(f"<{len(raw) // 8}q", raw))
        if min(vals) < 0:
            raise SystemExit(f"{name}: a negative value; the contract carries these as u64")
        bufs[hf_key] = vals
        kvs.append((f"{A}.{key}", "array", ("u64", vals)))
    for mk, hk in (("head_offsets", "ngram_heads_offsets"), ("head_vocab_sizes", "ngram_heads_vocab_sizes"),
                   ("layer_multipliers", "layer_multipliers")):
        if man.get(mk) != bufs[hk]:
            raise SystemExit(f"{ple_manifest}: {mk} {man.get(mk)} is not the checkpoint's {bufs[hk]} -- a different table")
    if man.get("layer") != ple_layer:
        raise SystemExit(f"{ple_manifest}: layer {man.get('layer')}; the checkpoint's PLE layer is {ple_layer} -- a different table")
    kvs += [
        ("pulsar.recipe", "string", recipe.name),
        ("pulsar.recipe.sha256", "string", recipe.sha256),
        ("pulsar.expert_gate_up", "string", recipe.gate_up),
        ("pulsar.mtp_present", "bool", False),
        ("pulsar.vision_present", "bool", "vision_config" in top),
        ("pulsar.ple_rows.file", "string", os.path.basename(ple_manifest)[:-len(".json")] + ".rows"),
        ("pulsar.ple_rows.layer", "u32", man["layer"]),
        ("pulsar.ple_rows.n_rows", "u64", man["n_rows"]),
        ("pulsar.ple_rows.row_bytes", "u32", man["row_bytes"]),
        ("pulsar.ple_rows.value_dtype", "string", man["value_dtype"]),
        ("pulsar.ple_rows.sha256", "string", man["sha256"]),
        ("general.sampling.top_p", "f32", hf.generation_config.get("top_p", 1.0)),
        ("general.sampling.top_k", "u32", hf.generation_config.get("top_k", 0)),
        ("general.sampling.temp", "f32", hf.generation_config.get("temperature", 1.0)),
    ]
    for tag, src in (("exl3", exl3), ("exl3_experts", exl3_experts)):
        if src is not None:
            kvs.append((f"pulsar.source.{tag}", "string", os.path.basename(os.path.normpath(src.dir))))
    kvs.append(("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))))
    kvs += tokenizer_kvs(tokenizer_dir)
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out


TT_NORMAL, TT_CONTROL, TT_USER_DEFINED = 1, 3, 4


def tokenizer_kvs(tok_dir):
    """Mechanical, from the checkpoint's own files: tokens by id (added tokens overlay), merges, the chat template
    verbatim.  Token type: added special -> CONTROL, added non-special -> USER_DEFINED, else NORMAL (record-only;
    the engine reads tokens + merges).  S5 (the Qwen renderer) owns anything beyond this."""
    tok = json.load(open(os.path.join(tok_dir, "tokenizer.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(tok_dir, "tokenizer_config.json"), encoding="utf-8"))
    if tok["model"]["type"] != "BPE":
        raise SystemExit(f"tokenizer model {tok['model']['type']}: BPE expected")
    by_id = {i: t for t, i in tok["model"]["vocab"].items()}
    types = {i: TT_NORMAL for i in by_id}
    for a in tok["added_tokens"]:
        by_id[a["id"]] = a["content"]
        types[a["id"]] = TT_CONTROL if a["special"] else TT_USER_DEFINED
    n = max(by_id) + 1
    if len(by_id) != n:
        raise SystemExit(f"tokenizer id space has {n - len(by_id)} holes")
    merges = tok["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [" ".join(m) for m in merges]

    def tid(key):
        t = cfg.get(key)
        if t is None:
            return None
        for a in tok["added_tokens"]:
            if a["content"] == t:
                return a["id"]
        raise SystemExit(f"tokenizer_config {key} {t!r} is not an added token")
    tmpl = open(os.path.join(tok_dir, "chat_template.jinja"), encoding="utf-8").read()
    out = [
        ("tokenizer.ggml.model", "string", "gpt2"),
        ("tokenizer.ggml.tokens", "array", ("string", [by_id[i] for i in range(n)])),
        ("tokenizer.ggml.token_type", "array", ("i32", [types[i] for i in range(n)])),
        ("tokenizer.ggml.merges", "array", ("string", merges)),
        ("tokenizer.pretokenize_regex", "string", cfg["pretokenize_regex"]),
        ("tokenizer.ggml.eos_token_id", "u32", tid("eos_token")),
        ("tokenizer.ggml.padding_token_id", "u32", tid("pad_token")),
        ("tokenizer.ggml.add_bos_token", "bool", bool(cfg.get("add_bos_token", False))),
        ("tokenizer.ggml.add_eos_token", "bool", bool(cfg.get("add_eos_token", False))),
        ("tokenizer.chat_template", "string", tmpl),
    ]
    if cfg.get("bos_token") is not None:
        out.append(("tokenizer.ggml.bos_token_id", "u32", tid("bos_token")))
    return out

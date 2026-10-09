"""qwen.py -- the qwen4_exp family in the direct builder (L251 S6): Qwen3.8-Flash-Next -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 HF checkpoint (the authority for every BF16 tensor, every MXFP8 tensor's input, the config and
           the tokenizer)
  --exl3   an exllamav3-layout checkpoint holding the EXL3 tensors the recipe names: <linear>.{trellis,suh,svh,mul1}
           under the HF Linear name (dense: minus ".weight"; routed experts: mlp.experts.<e>.<part>).  Ours
           (research/l251/container/qwen_exl3_quant.py: gate_up fused) or turboderp's (gate / up separate).
  --exl3-experts   optional: a DIFFERENT exllamav3 checkpoint for the routed experts only (e.g. turboderp's K4
           experts beside our K5 dense).  Named explicitly; nothing falls back from one source to the other.
  --recipe a pulsar.recipe.v1 (recipe.py; format-maps/qwen38fn-*.json, name-pattern rows): every checkpoint tensor
           matches EXACTLY ONE row, every row matches something; a tensor no row names refuses.  Its family setting
           expert_gate_up (fused | split) says how the routed gate_up stack is written.
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
    k5 / k6 / k8 (U8 [trellis | suh | svh], dims_ne [in, out]; K6 / K8 are turboderp's packs, L266).  Declared shapes are the SOURCE shapes (no reshapes).
  * shards: `vision` (primary: pulsar.kv + the vision tower, BF16 native), `layers.0..47`, `top` (embed, head,
    top-level mixer), and `mtp` (the drafter's mtp.* tensors, its expert families marked "block": "mtp") when the
    recipe writes them (L279; the L251 sidecar's rows) -- pulsar.mtp_present says which.
"""
from __future__ import annotations

import json
import os
import re
import struct

import entries as EN
import exl3_rates
import kv as KV
from tokenizer import core as TOK
import producers as PR
import recipe as R
from names import Mapped

FAMILY = "qwen4_exp"
PFX = "model.language_model."
META = {"pulsar.family": FAMILY}
SETTINGS = ("expert_gate_up",)  # the recipe's family setting: the routed gate_up stack fused or split
EXPERT_STACKS = True            # an HF expert tensor is the whole [E, out, in] stack; its families plan in walk order
GATE_UP_PARTS = ("gate_up_proj", "gate_proj", "up_proj")
PART_ROLE = {"gate_up_proj": "expert_gate_up", "gate_proj": "expert_gate", "up_proj": "expert_up",
             "down_proj": "expert_down"}
# what the qwen4_exp loader binds (any role; the EXL3 rate by the engine's arm, exl3_rates.admit)
ADMITS = {"bf16", "mxfp8_lt", "tessera", *exl3_rates.K2, *R.CONSUMED}
# the tokenizer (tokenizer/core.py's walk), mechanical from the checkpoint's own files: HF-special added tokens
# CONTROL, the rest USER_DEFINED; the checkpoint's chat_template.jinja; its pretokenize_regex; bos only when the
# config names one (it names none).  S5 (the Qwen renderer) owns anything beyond this.
TOKENIZER = TOK.Settings(added_type=lambda a: TOK.TT_CONTROL if a["special"] else TOK.TT_USER_DEFINED,
                         pretokenize_regex=True, bos_optional=True)


# ---------------------------------------------------------------------------
# the naming table: HF name -> (role, shard, the expert stack's per-part entry names).  The container name and the
# gguf_name ARE the HF name (the Qwen family binds by the checkpoint's own names); what the table adds is what
# each tensor IS (its role, the recipe's vocabulary) and where it lives.  A name the table lacks refuses.
# ---------------------------------------------------------------------------
# the suffix after `model.language_model.layers.N.` (and the drafter's `mtp.layers.N.`) -> role
BLOCK_ROLES = {
    **{f"{hc}_hyper_connection.{t}": r for hc in ("attn", "mlp") for t, r in (
        ("block_inject_weight.weight", "other"), ("hc_norm.weight", "norm"),
        ("input_mix_weight_down.weight", "dense"), ("input_mix_weight_up.weight", "dense"))},
    "linear_attn.A_log": "other", "linear_attn.dt_bias": "other", "linear_attn.conv1d.weight": "other",
    "linear_attn.norm.weight": "norm",
    **{f"linear_attn.{p}.weight": "dense" for p in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")},
    **{f"self_attn.{p}.weight": "dense" for p in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")},
    **{f"self_attn.{p}.weight": "norm" for p in ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm")},
    "mlp.gate.weight": "router",
    "mlp.experts.gate_up_proj": "expert_gate_up",
    "mlp.experts.down_proj": "expert_down",
    **{f"mlp.shared_expert.{p}.weight": "shared_expert" for p in ("gate_proj", "up_proj", "down_proj")},
    "mlp.shared_expert_gate.weight": "router",
    "ple.key_proj.weight": "dense", "ple.value_proj.weight": "dense", "ple.conv1d.weight": "other",
    **{f"ple.{p}.weight": "norm" for p in ("norm_conv", "norm_key", "norm_query")},
    **{f"ple.ple_embedding.{p}": "ple_buffer" for p in ("layer_multipliers", "ngram_heads_offsets",
                                                       "ngram_heads_vocab_sizes")},
}
TOP_ROLES = {
    "lm_head.weight": "head",
    f"{PFX}embed_tokens.weight": "embed",
    f"{PFX}hyper_connection_mixer.hc_norm.weight": "norm",
    f"{PFX}hyper_connection_mixer.input_mix_weight_down.weight": "other",
    f"{PFX}hyper_connection_mixer.input_mix_weight_up.weight": "other",
}
# the drafter's own (outside mtp.layers.N): no lane reads it yet; it maps so a recipe can omit it by name
MTP_ROLES = {"mtp.fc_embedding.weight": "dense", "mtp.fc_hidden.weight": "dense",
             "mtp.pre_fc_norm_embedding.weight": "norm", "mtp.pre_fc_norm_hidden.weight": "norm",
             "mtp.hyper_connection_mixer.hc_norm.weight": "norm",
             "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "other",
             "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "other"}
_BLOCK = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.(.+)$")
_NGRAM = re.compile(r"^ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")


def map_hf(name: str, shape=None) -> Mapped | None:
    """The naming table's row for an HF name, or None (the caller refuses)."""
    m = _BLOCK.match(name)
    if m:
        mtp, layer, rest = m.group(1) == "mtp.layers", int(m.group(2)), m.group(3)
        role = "ple_table" if _NGRAM.match(rest) else BLOCK_ROLES.get(rest)
        if role is None:
            return None
        part = rest.rsplit(".", 1)[1] if role.startswith("expert_") else None
        return Mapped(name, name, "mtp" if mtp else "layer", "mtp" if mtp else f"layers.{layer}", layer, None, part,
                      False, role=role)
    if name.startswith("model.visual."):
        return Mapped(name, name, "vision", "vision", None, None, None, False, role="vision")
    if name in TOP_ROLES:
        return Mapped(name, name, "top", "top", None, None, None, False, role=TOP_ROLES[name])
    if name in MTP_ROLES:
        return Mapped(name, name, "mtp", "mtp", None, None, None, False, role=MTP_ROLES[name])
    return None


def expert_names(m: Mapped, part: str) -> tuple[str, str]:
    """(family gguf_name, per-expert entry name with "{e}") of one part of the expert stack `m`."""
    pre = m.container_name[:-len(m.part)]
    return pre + part, pre + "{e}." + part + ".weight"


def shape(hf, ctx):
    """n_layer, the recipe, and whether the recipe writes the MTP drafter (an `mtp` shard after `top`)."""
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    out = {"family": FAMILY, "n_layer": int(hf.config["num_hidden_layers"]), "recipe": ctx.recipe.name}
    res = ctx.recipe.resolve({n: map_hf(n) for n in hf.names()})
    if any(f not in R.CONSUMED for n, (f, _s) in res.items() if map_hf(n).shard == "mtp"):
        out["mtp"] = True
    return out


def shard_order(shape):
    return ["vision"] + [f"layers.{i}" for i in range(shape["n_layer"])] + ["top"] + (["mtp"] if shape.get("mtp") else [])


def declared_shape(m, hshape):
    return list(hshape)                 # no reshapes: the declared shape is the source's


def admit(hf, m, fmt, name):
    if fmt not in ADMITS:
        raise SystemExit(f"{name}: the qwen4_exp loader binds {sorted(ADMITS)}, not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit("qwen4_exp: pass --recipe (tools/container/format-maps/qwen38fn-*.json)")


def check_recipe(recipe):
    if recipe.settings.get("expert_gate_up") not in ("fused", "split"):
        raise SystemExit(f"{recipe.path}: expert_gate_up {recipe.settings.get('expert_gate_up')!r} is neither "
                         "'fused' nor 'split'")


def stack_families(hf, m, fmt, ctx):
    """One HF expert stack ([E, out, in] BF16) -> its families: gate_up_proj (fused) or gate_proj + up_proj (split,
    the recipe's setting), down_proj; each expert-projection from the EXL3 source by its Linear key."""
    stack_name = m.container_name
    layout, src = fmt
    if layout == "tessera":
        return PR.produce(PR.spec(PR.producer_for("tessera", layout, stack_name), []), ctx.sources[src])
    if layout not in exl3_rates.K2:
        raise SystemExit(f"{stack_name}: the recipe names {layout}; a routed stack is written from an EXL3 source")
    E, out, inp = hf.shape(stack_name)
    if m.part == "gate_up_proj" and ctx.recipe.settings["expert_gate_up"] == "split":
        if out % 2:
            raise SystemExit(f"{stack_name}: odd gate_up width {out}")
        parts = [("gate_proj", out // 2), ("up_proj", out // 2)]
    else:
        parts = [(m.part, out)]
    fams = []
    for part, n in parts:
        fam_name, entry_name = expert_names(m, part)
        names = [entry_name.replace("{e}", str(e)) for e in range(E)]
        srcs = [("ranges", EN.exl3_ranges(ctx.sources[src], x[:-len(".weight")], layout, inp, n)[0]) for x in names]
        fams.append(dict(gguf_name=fam_name, part=part, role=PART_ROLE[part], layout=layout, inp=inp, n=n,
                         entry_names=names, srcs=srcs, extras={"entry_name": entry_name, "layer": m.layer,
                                                               **({"block": "mtp"} if m.shard == "mtp" else {})}))
    return fams


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


def build_kv(hf, ctx):
    """pulsar.kv for qwen4_exp: `general.*`; the text_config verbatim under `qwen4_exp.` (config_kvs); the PLE
    buffers as u64 arrays; the builder's own facts under `pulsar.*` (recipe, sources, the PLE row file the
    table must match); the tokenizer from the checkpoint's own files."""
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    recipe, ple_manifest = ctx.recipe, ctx.ple_rows
    exl3, exl3_experts = ctx.sources.get("exl3"), ctx.sources.get("exl3_experts")
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
        ("pulsar.expert_gate_up", "string", recipe.settings["expert_gate_up"]),
        ("pulsar.mtp_present", "bool", bool(ctx.shape.get("mtp"))),
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
    kvs += TOK.kvs(ctx.tokenizer_dir, TOKENIZER)
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out

"""nemotron_parse.py -- NVIDIA Nemotron-3.5 Parse in the one artifact builder (L321): config.json architectures
["RadioMBartPrefixForCausalLM"], model_type "nemotron_parse" -- a C-RADIOv4-SO400M ViT (27 blocks, 1152 wide, 16 heads
of 72, MLP 4304, 10 class tokens, a learned 128 x 128 position field resampled per page), a neck (1x1 conv 1152 ->
1024 + LayerNorm; a (1, 4) stride-4 conv merging four adjacent patches + LayerNorm; a summary token from class tokens 0
and 1, Linear 2304 -> 1024 + LayerNorm), and a decoder-only mBART (10 layers, 1024 wide, 16 query / 8 KV heads of 64,
GELU MLP 4096, pre-LayerNorm, no positional encoding, embeddings x sqrt(1024), tied head over 72512 ids) reading the
neck's rows as a CAUSAL visual prefix before the prompt (hf_nemotron_parse_modeling.py, the checkpoint's own code).

Source: --hf, the F32 checkpoint (one model.safetensors) + its preprocessor_config.json and generation_config.json.
Recipe (required, format-maps/nemotron_parse-f16.json): every Linear and the tied embedding f16 (the Turing lane's
dense format: fp16 tensor cores, L321), every norm, bias, class token and the position field f32 as stored.

What the module checks (rule 1 -- refuse, never guess): every config value the forward depends on is the one the
engine implements (pinned below: the causal visual prefix, GQA 16 / 8, NoPE, tied head, scaled embeddings, GELU,
the canvas, the patch grid, the CLIP normalisation, the content-normalised position field, the greedy recipe's
repetition penalty), summary_idxs == [0, 1]; the names both ways; every dtype and shape.

Consumed by name, written nowhere: the input conditioner's mean / std (the processor normalises; the forward never
reads them), summary_idxs (pinned), and the one-step MTP tensors (decoder.extra_heads / extra_proj: no drafter in v1).

The convolutions are written as the Linears they are (Mapped.axes): conv1 [1024, 1152, 1] -> [1024, 1152]; conv2
[1024, 1024, 1, 4] -> [out, kw, ch] = [1024, 4096], so four consecutive patch rows ARE its input row.

The container: shards `vision` (primary: pulsar.kv, the ViT and the neck), `layers.0..9` (the decoder), `top` (the
embedding, the embedding and final norms).  pulsar.kv: general.*, the pinned facts under `nemotron_parse.`; side
file tokenizer.json (the engine's decode table and the prompt's added tokens).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

import kv as KV
from names import Mapped

FAMILY = "nemotron_parse"
MODEL_TYPE = "nemotron_parse"
ARCH = "RadioMBartPrefixForCausalLM"
META = {"pulsar.family": FAMILY}
SETTINGS = ()
EXPERT_STACKS = False
GATE_UP_PARTS = ()

VIT = "encoder.model_encoder.radio_model.model."
CONSUMED = ("encoder.model_encoder.radio_model.input_conditioner.", "encoder.model_encoder.radio_model.summary_idxs",
            "decoder.extra_heads.", "decoder.extra_proj.")

# the values the engine implements: (where, key) -> value
PINS = {
    ("top", "is_encoder_decoder"): False, ("top", "legacy_causal_vision_prefix"): True,
    ("top", "tie_word_embeddings"): True, ("top", "max_sequence_length"): 16384, ("top", "image_size"): [2048, 1664],
    ("top", "vision_spatial_dim"): 1152, ("top", "vision_summary_dim"): 2304, ("top", "num_extra_heads"): 1,
    ("top", "navit_packing"): True, ("top", "navit_position_embedding_mode"): "content_normalized_2d_interpolate",
    ("top", "navit_coordinate_frame"): "content_normalized", ("top", "navit_disable_cpe_augmentation"): True,
    ("top", "eos_token_id"): 2, ("top", "pad_token_id"): 1, ("top", "bos_token_id"): 0,
    ("dec", "d_model"): 1024, ("dec", "decoder_layers"): 10, ("dec", "decoder_attention_heads"): 16,
    ("dec", "self_attention_kv_heads"): 8, ("dec", "cross_attention_kv_heads"): 0, ("dec", "decoder_ffn_dim"): 4096,
    ("dec", "vocab_size"): 72512, ("dec", "activation_function"): "gelu", ("dec", "scale_embedding"): True,
    ("dec", "add_final_layer_norm"): True, ("dec", "causal_vision_prefix"): True, ("dec", "tie_word_embeddings"): True,
    ("enc", "patch_size"): 16, ("enc", "max_resolution"): 2048,
    ("pre", "do_normalize"): True, ("pre", "do_rescale"): True, ("pre", "final_size"): [2048, 1664],
    ("pre", "navit_packing"): True, ("pre", "navit_padding_mode"): "top_left_white",
    ("pre", "navit_coordinate_frame"): "content_normalized",
    ("pre", "image_mean"): [0.48145466, 0.4578275, 0.40821073], ("pre", "image_std"): [0.26862954, 0.26130258, 0.27577711],
    ("gen", "do_sample"): False, ("gen", "num_beams"): 1, ("gen", "repetition_penalty"): 1.1,
    ("gen", "max_new_tokens"): 9000, ("gen", "eos_token_id"): 2,
}
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"   # the card's default

# per ViT block: HF suffix -> (bind role, recipe role, shape)
VIT_BLOCK = {
    "norm1.weight": ("vit_norm1", "norm", [1152]), "norm1.bias": ("vit_norm1_bias", "norm", [1152]),
    "attn.qkv.weight": ("vit_qkv", "dense", [3456, 1152]), "attn.qkv.bias": ("vit_qkv_bias", "norm", [3456]),
    "attn.proj.weight": ("vit_proj", "dense", [1152, 1152]), "attn.proj.bias": ("vit_proj_bias", "norm", [1152]),
    "norm2.weight": ("vit_norm2", "norm", [1152]), "norm2.bias": ("vit_norm2_bias", "norm", [1152]),
    "mlp.fc1.weight": ("vit_fc1", "dense", [4304, 1152]), "mlp.fc1.bias": ("vit_fc1_bias", "norm", [4304]),
    "mlp.fc2.weight": ("vit_fc2", "dense", [1152, 4304]), "mlp.fc2.bias": ("vit_fc2_bias", "norm", [1152]),
}
# the tower's and the neck's own tensors: HF name -> (bind role, recipe role, source shape, declared shape, axes)
VISION_TOP = {
    VIT + "patch_generator.embedder.weight": ("vit_patch", "dense", [1152, 768], None, ()),
    VIT + "patch_generator.cls_token.token": ("vit_cls", "norm", [10, 1152], None, ()),
    VIT + "patch_generator.pos_embed": ("vit_pos", "norm", [1, 16384, 1152], [16384, 1152], ()),
    "encoder.conv1.weight": ("neck_proj", "dense", [1024, 1152, 1], [1024, 1152], ()),
    "encoder.conv1.bias": ("neck_proj_bias", "norm", [1024], None, ()),
    "encoder.layer_norm1.weight": ("neck_norm1", "norm", [1024], None, ()),
    "encoder.layer_norm1.bias": ("neck_norm1_bias", "norm", [1024], None, ()),
    "encoder.conv2.weight": ("neck_merge", "dense", [1024, 1024, 1, 4], [1024, 4096], (0, 2, 3, 1)),
    "encoder.layer_norm2.weight": ("neck_norm2", "norm", [1024], None, ()),
    "encoder.layer_norm2.bias": ("neck_norm2_bias", "norm", [1024], None, ()),
    "encoder.sum_proj.weight": ("neck_summary", "dense", [1024, 2304], None, ()),
    "encoder.sum_proj.bias": ("neck_summary_bias", "norm", [1024], None, ()),
    "encoder.layer_norm3.weight": ("neck_norm3", "norm", [1024], None, ()),
    "encoder.layer_norm3.bias": ("neck_norm3_bias", "norm", [1024], None, ()),
}
# per decoder layer: HF suffix -> (bind role, recipe role, shape)
DEC_LAYER = {
    "self_attn_layer_norm.weight": ("attn_norm", "norm", [1024]), "self_attn_layer_norm.bias": ("attn_norm_bias", "norm", [1024]),
    "self_attn.q_proj.weight": ("attn_q", "dense", [1024, 1024]), "self_attn.q_proj.bias": ("attn_q_bias", "norm", [1024]),
    "self_attn.k_proj.weight": ("attn_k", "dense", [512, 1024]), "self_attn.k_proj.bias": ("attn_k_bias", "norm", [512]),
    "self_attn.v_proj.weight": ("attn_v", "dense", [512, 1024]), "self_attn.v_proj.bias": ("attn_v_bias", "norm", [512]),
    "self_attn.out_proj.weight": ("attn_o", "dense", [1024, 1024]), "self_attn.out_proj.bias": ("attn_o_bias", "norm", [1024]),
    "final_layer_norm.weight": ("ffn_norm", "norm", [1024]), "final_layer_norm.bias": ("ffn_norm_bias", "norm", [1024]),
    "fc1.weight": ("ffn_up", "dense", [4096, 1024]), "fc1.bias": ("ffn_up_bias", "norm", [4096]),
    "fc2.weight": ("ffn_down", "dense", [1024, 4096]), "fc2.bias": ("ffn_down_bias", "norm", [1024]),
}
DEC_TOP = {
    "decoder.embed_tokens.weight": ("embed", "embed", [72512, 1024]),
    "decoder.layernorm_embedding.weight": ("embed_norm", "norm", [1024]),
    "decoder.layernorm_embedding.bias": ("embed_norm_bias", "norm", [1024]),
    "decoder.layer_norm.weight": ("output_norm", "norm", [1024]),
    "decoder.layer_norm.bias": ("output_norm_bias", "norm", [1024]),
}
N_VIT, N_DEC = 27, 10


def _read_json(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        raise SystemExit(f"{p}: the Parse checkpoint's {name} is part of its declaration -- missing")
    return json.load(open(p))


def _check(where, d, key, want):
    if key not in d:
        raise SystemExit(f"{where}: no {key!r} -- the Parse declaration reads it")
    got = d[key]
    same = (abs(got - want) <= 1e-9 * max(1.0, abs(want)) if isinstance(want, float) else
            [float(x) for x in got] == [float(x) for x in want] if isinstance(want, list) and want and
            isinstance(want[0], float) else got == want)
    if not same:
        raise SystemExit(f"{where}.{key} = {got!r}; the engine implements {want!r} -- refusing")


@dataclass(frozen=True)
class Shape:
    recipe: str = ""
    table: dict = field(default=None, compare=False, repr=False)

    def __str__(self):
        return (f"nemotron_parse: RADIO ViT {N_VIT} x 1152 (16 x 72), neck 1152 -> 1024 (x4 merge + summary), "
                f"mBART decoder {N_DEC} x 1024 (16q / 8kv), causal visual prefix, recipe {self.recipe}")


def declare(hf):
    top = hf.config["top_level"]
    if top.get("model_type") != MODEL_TYPE or top.get("architectures") != [ARCH]:
        raise SystemExit(f"config.json: model_type {top.get('model_type')!r} / architectures "
                         f"{top.get('architectures')!r}; nemotron_parse.py reads {MODEL_TYPE} / [{ARCH}]")
    blocks = {"top": top, "dec": top.get("decoder") or {}, "enc": top.get("encoder") or {},
              "pre": _read_json(hf.dir, "preprocessor_config.json"), "gen": hf.generation_config}
    for (where, key), want in PINS.items():
        _check({"top": "config.json", "dec": "config.json decoder", "enc": "config.json encoder",
                "pre": "preprocessor_config.json", "gen": "generation_config.json"}[where], blocks[where], key, want)
    args = blocks["enc"].get("args") or {}
    if args.get("model") != "vit_so400m_patch16_224" or args.get("register_multiple") != 10:
        raise SystemExit(f"config.json encoder.args: {args.get('model')!r} / register_multiple "
                         f"{args.get('register_multiple')!r}; the engine implements vit_so400m_patch16_224 with 10 "
                         f"class tokens -- refusing")
    return Shape()


def expected():
    """{HF name: (dtype, source shape, Mapped)} for every tensor the declaration writes."""
    out = {}

    def add(name, shape, bind, role, shard, block=None, layer=None, dshape=None, axes=()):
        rec = (("role", bind),) + ((("block", block), ("layer", layer)) if block is not None else ())
        fam = "layer" if layer is not None else "top"
        out[name] = ("F32", shape, Mapped(name, name, fam, shard, layer, None, None, False, role=role, bind=rec,
                                          axes=tuple(axes)), dshape or shape)

    for name, (bind, role, shp, dshp, axes) in VISION_TOP.items():
        add(name, shp, bind, role, "vision", dshape=dshp, axes=axes)
    for il in range(N_VIT):
        for s, (bind, role, shp) in VIT_BLOCK.items():
            add(f"{VIT}blocks.{il}.{s}", shp, bind, role, "vision", "vit", il)
    for il in range(N_DEC):
        for s, (bind, role, shp) in DEC_LAYER.items():
            add(f"decoder.layers.{il}.{s}", shp, bind, role, f"layers.{il}", "layers", il)
    for name, (bind, role, shp) in DEC_TOP.items():
        add(name, shp, bind, role, "top")
    return out


def table(hf):
    """{name: Mapped} for every HF tensor, checked both ways and dtype / shape against the declaration."""
    want = expected()
    mapped, unknown, wrong = {}, [], []
    for n in hf.names():
        if n.startswith(CONSUMED):
            mapped[n] = Mapped(n, n, "consumed", "vision", None, None, None, False, emit=False, role="consumed")
            continue
        if n not in want:
            unknown.append(n)
            continue
        dt, shp, m, _ = want[n]
        if hf.dtype(n) != dt or hf.shape(n) != shp:
            wrong.append(f"{n}: {hf.dtype(n)} {hf.shape(n)}, the declaration says {dt} {shp}")
            continue
        mapped[n] = m
    if unknown:
        raise SystemExit(f"{len(unknown)} tensors the Parse naming table does not map -- refusing: {unknown[:5]}")
    if wrong:
        raise SystemExit(f"{len(wrong)} tensors disagree with the declared shape -- refusing: {wrong[:3]}")
    missing = sorted(n for n in want if n not in mapped)
    if missing:
        raise SystemExit(f"{len(missing)} tensors the declaration expects are missing from {hf.dir} -- refusing: "
                         f"{missing[:5]}")
    idx = hf.raw("encoder.model_encoder.radio_model.summary_idxs")
    if list(np.frombuffer(idx, dtype="<i8")) != [0, 1]:
        raise SystemExit("encoder...summary_idxs is not [0, 1]: the engine's summary token reads class tokens 0 and 1")
    return mapped


# ---------------------------------------------------------------------------
# the family hooks (build.plan)
# ---------------------------------------------------------------------------
_DECLARED = {n: d for n, (_, _, _, d) in expected().items()}


def source(hf):
    hf.decl = declare(hf)
    hf.mapped = table(hf)
    return hf


def map_hf(name: str, shape: Shape) -> Mapped | None:
    return shape.table.get(name)


def shape(hf, ctx):
    if ctx.sources.get("exl3") is not None or ctx.sources.get("exl3_experts") is not None:
        raise SystemExit("--exl3: Parse's dense format is f16 (the Turing lane decodes token by token)")
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    return replace(hf.decl, recipe=ctx.recipe.name, table=hf.mapped)


def shard_order(shape):
    return ["vision"] + [f"layers.{i}" for i in range(N_DEC)] + ["top"]


def declared_shape(m, hshape):
    return list(_DECLARED[m.container_name])


def admit(hf, m, fmt, name):
    allowed = {"f16"} if m.role in ("dense", "embed") else {"native"}
    if fmt not in allowed:
        raise SystemExit(f"{name}: the Parse loader binds this {m.role} tensor as {sorted(allowed)}, not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit(f"{MODEL_TYPE}: pass --recipe (tools/container/format-maps/nemotron_parse-f16.json)")


def gate_up_key(rec):
    return rec["gguf_name"].rsplit(".", 1)[0]


# ---------------------------------------------------------------------------
# pulsar.kv: the pinned facts the engine's shape table pins again
# ---------------------------------------------------------------------------
def build_kv(hf, ctx):
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    P = "nemotron_parse."
    kvs = [("general.architecture", "string", FAMILY), ("general.type", "string", "model"),
           ("general.name", "string", "Nemotron-3.5-Parse"),
           (P + "vit.width", "u32", 1152), (P + "vit.depth", "u32", N_VIT), (P + "vit.heads", "u32", 16),
           (P + "vit.mlp", "u32", 4304), (P + "vit.patch", "u32", 16), (P + "vit.n_cls", "u32", 10),
           (P + "vit.pos_grid", "u32", 128), (P + "vit.norm_eps", "f32", 1e-6),
           (P + "neck.width", "u32", 1024), (P + "neck.merge", "u32", 4), (P + "neck.norm_eps", "f32", 1e-6),
           (P + "dec.width", "u32", 1024), (P + "dec.layers", "u32", N_DEC), (P + "dec.heads", "u32", 16),
           (P + "dec.kv_heads", "u32", 8), (P + "dec.ffn", "u32", 4096), (P + "dec.vocab", "u32", 72512),
           (P + "dec.norm_eps", "f32", 1e-5), (P + "dec.embed_scale", "f32", 32.0),
           (P + "max_sequence", "u32", 16384), (P + "bos", "u32", 0), (P + "eos", "u32", 2), (P + "pad", "u32", 1),
           (P + "image.canvas_h", "u32", 2048), (P + "image.canvas_w", "u32", 1664),
           (P + "image.align_h", "u32", 16), (P + "image.align_w", "u32", 64)]
    pre = _read_json(hf.dir, "preprocessor_config.json")
    for i in range(3):
        kvs += [(P + f"image.mean.{i}", "f32", float(pre["image_mean"][i])),
                (P + f"image.std.{i}", "f32", float(pre["image_std"][i]))]
    kvs += [(P + "gen.repetition_penalty", "f32", 1.1), (P + "gen.max_new_tokens", "u32", 9000),
            (P + "gen.prompt", "string", PROMPT),
            ("pulsar.recipe", "string", ctx.recipe.name), ("pulsar.recipe.sha256", "string", ctx.recipe.sha256 or ""),
            ("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir)))]
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out


SIDE = ("tokenizer.json",)


def side_files(hf, ctx):
    files = [(n, os.path.join(ctx.tokenizer_dir, n)) for n in SIDE]
    missing = [src for _, src in files if not os.path.isfile(src)]
    if missing:
        raise SystemExit(f"the Parse container carries the checkpoint's tokenizer: {', '.join(missing)} missing")
    return files

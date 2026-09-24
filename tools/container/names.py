"""names.py -- the ONE table from an HF checkpoint name to the container entry (L247).

The container's tensor names ARE the HF checkpoint's names (DeepSeek's native
namespace); the engine binds each entry by the `gguf_name` it carries
(src/engine/safetensors.cpp, weights.cpp).  Everything a builder needs to know
about a name is answered here and nowhere else:

  container_name  the safetensors key the entry is written under
  gguf_name       what the engine binds: blk.N.*, dspark.N.*, token_embd.weight,
                  output_norm.weight, output.weight, output_hc_*.weight; the
                  vision tower and Engram projections bind under their HF names
  family          top | layer | vision | mtp | expert
  shard           vision | layers.N | top | mtp.N  (one file per key, shard_order())
  is_scale        this HF tensor is a COMPANION of the entry named container_name
                  (an FP8/FP4 `.scale`, an EXL3 `.suh/.svh/.mul1`): it is consumed
                  together with the primary and never emitted on its own
  emit            False = the builder consumes the name but writes nothing to the
                  container (the three drop rules below); the entry still maps,
                  so a checkpoint walk stays fail-closed over every name

Container key == HF key everywhere except two folds forced by structure:
  (a) routed experts: `layers.L.ffn.experts.E.w{1,2,3}.{weight,scale}` -> ONE U8
      entry `layers.L.ffn.experts.E.w{1,2,3}.weight` (the scale is inside the
      layout); the drafter's `mtp.N.ffn.experts.*` fold the same way;
  (c) EXL3 sources: `<proj>.{trellis,suh,svh,mul1}` (and the block-diagonal
      `wo_a.slice.S.*`) -> the entry `<proj>.weight`, dense and expert alike.
The V4.1 drafter head (research doc s3 deviation (b)) is NOT a key deviation:
V4.1 spells `mtp.N.markov_head.{embed,head}.weight`, Vision-Exp spells
`mtp.N.markov_head.markov_w{1,2}.weight`, and each keeps its own HF key in the
container.  Only the gguf_name is pinned -- both bind as
`dspark.N.markov_head.markov_w{1,2}.weight` (weights.cpp dspark_weights_bind),
embed = w1 (vocab -> rank) and head = w2 (rank -> vocab; the k-major
fp8_e4m3_soa_k tensor, policy.py).

Drop rules (emit=False), each the served artifact's behaviour:
  1. `layers.N.ffn.gate.bias` on a hash-routed layer (N < n_hash): Vision-Exp
     allocates the router bias there but the forward never reads it (text
     tokens route through tid2eid, image tokens through bias_vl).
  2. `mtp.N.ffn.gate.bias_vl`: drafts are text; dspark_weights_bind has no
     binding for an image-token bias.
  3. `layers.N.engram.embed.{weight,scale}`: the Engram row table (L242) is a
     SIDE artifact, never a container tensor; `wkv`, `q_weight`, `k_weight` are
     ordinary layer-shard tensors under their HF names.

Unknown names return None and the caller refuses (rule 1 / rule 9); nothing
is skipped silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """What the name table needs from config.json.  V4 keeps the text stack at
    the top level; V4.1 nests it under `text_config` and deleted hash routing
    (no `num_hash_layers` key -- zero is the checkpoint's word, not a default)."""
    n_layer: int
    n_mtp: int
    has_vision: bool
    v41: bool
    n_hash: int

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelShape":
        v41 = cfg.get("model_type") == "deepseek_v41"
        if v41:
            tc = cfg["text_config"]
            if "num_hash_layers" in tc:
                raise ValueError("a deepseek_v41 config names num_hash_layers; V4.1 has no hash routing")
            n_hash = 0
            has_vision = "vision_config" in cfg
        else:
            if cfg.get("model_type") != "deepseek_v4":
                raise ValueError(f"unknown model_type {cfg.get('model_type')!r}")
            tc = cfg
            n_hash = int(tc["num_hash_layers"])
            has_vision = int(tc.get("vision_n_layers", 0)) > 0
        return cls(n_layer=int(tc["num_hidden_layers"]),
                   n_mtp=int(tc["num_nextn_predict_layers"]),
                   has_vision=has_vision, v41=v41, n_hash=n_hash)


@dataclass(frozen=True)
class Mapped:
    container_name: str
    gguf_name: str
    family: str
    shard: str
    layer: int | None
    expert: int | None
    part: str | None
    is_scale: bool
    emit: bool = True


def shard_order(shape: ModelShape) -> list[str]:
    """One shard per key: the vision/primary shard first (it carries pulsar.kv
    even when the tower is absent), one per layer, the top, one per drafter
    layer.  Files are model-{i:05d}-of-{n:05d}.safetensors in this order."""
    return (["vision"] + [f"layers.{i}" for i in range(shape.n_layer)] + ["top"]
            + [f"mtp.{i}" for i in range(shape.n_mtp)])


def shard_file(shape: ModelShape, shard: str) -> str:
    order = shard_order(shape)
    return f"model-{order.index(shard) + 1:05d}-of-{len(order):05d}.safetensors"


# ---------------------------------------------------------------------------
# The tables.  HF suffix (after `layers.N.` / `mtp.N.`) -> engine suffix (after
# `blk.N.` / `dspark.N.`).  Every row is a tensor some checkpoint carries;
# presence is always the checkpoint's word, never an index list.
# ---------------------------------------------------------------------------
TOP = {
    "embed.weight":  "token_embd.weight",
    "norm.weight":   "output_norm.weight",
    "head.weight":   "output.weight",
    # Vision-Exp only: the head's own HC mix (shape profile hc_head_mix).
    "hc_head_base":  "output_hc_base.weight",
    "hc_head_fn":    "output_hc_fn.weight",
    "hc_head_scale": "output_hc_scale.weight",
}

LAYER = {
    "hc_attn_base":                         "hc_attn_base.weight",
    "hc_attn_fn":                           "hc_attn_fn.weight",
    "hc_attn_scale":                        "hc_attn_scale.weight",
    "hc_ffn_base":                          "hc_ffn_base.weight",
    "hc_ffn_fn":                            "hc_ffn_fn.weight",
    "hc_ffn_scale":                         "hc_ffn_scale.weight",
    "attn.attn_sink":                       "attn_sinks.weight",
    "attn.wq_a.weight":                     "attn_q_a.weight",
    "attn.wq_b.weight":                     "attn_q_b.weight",
    "attn.q_norm.weight":                   "attn_q_a_norm.weight",
    "attn.wkv.weight":                      "attn_kv.weight",
    "attn.kv_norm.weight":                  "attn_kv_a_norm.weight",
    "attn.wo_a.weight":                     "attn_output_a.weight",
    "attn.wo_b.weight":                     "attn_output_b.weight",
    # CSA/CSA2: the compressor on every kv source (ape: 0731/Vision-Exp only;
    # wgate: ratio > 1 only), the indexer q/score weights on every index source.
    "attn.compressor.ape":                  "attn_compressor_ape.weight",
    "attn.compressor.wkv.weight":           "attn_compressor_kv.weight",
    "attn.compressor.wgate.weight":         "attn_compressor_gate.weight",
    "attn.compressor.norm.weight":          "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight":             "indexer.attn_q_b.weight",
    "attn.indexer.weights_proj.weight":     "indexer.proj.weight",
    # V4.1: the index key projects from the kv source's latent ...
    "attn.indexer.wk.weight":               "indexer.attn_k.weight",
    "attn.indexer.k_norm.weight":           "indexer.k_norm.weight",
    # ... 0731/Vision-Exp: the indexer compresses its own index key.
    "attn.indexer.compressor.ape":          "indexer_compressor_ape.weight",
    "attn.indexer.compressor.wkv.weight":   "indexer_compressor_kv.weight",
    "attn.indexer.compressor.wgate.weight": "indexer_compressor_gate.weight",
    "attn.indexer.compressor.norm.weight":  "indexer_compressor_norm.weight",
    "attn_norm.weight":                     "attn_norm.weight",
    "ffn_norm.weight":                      "ffn_norm.weight",
    "ffn.shared_experts.w1.weight":         "ffn_gate_shexp.weight",
    "ffn.shared_experts.w3.weight":         "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight":         "ffn_down_shexp.weight",
    "ffn.gate.weight":                      "ffn_gate_inp.weight",
    "ffn.gate.bias":                        "exp_probs_b.bias",
    "ffn.gate.bias_vl":                     "exp_probs_b_vl.bias",     # Vision-Exp: router bias for IMAGE tokens
    "ffn.gate.tid2eid":                     "ffn_gate_tid2eid.weight", # 0731/Vision-Exp hash layers only
}

# Drafter-only rows (mtp.N.*).  The two un-numbered engine names are the
# drafter's entry projection, which the checkpoint keeps on mtp.0.
DRAFTER = {
    "main_proj.weight":              "main_proj.weight",
    "main_norm.weight":              "main_norm.weight",
    "norm.weight":                   "norm.weight",
    "confidence_head.proj.weight":   "confidence_head.proj.weight",
    "markov_head.markov_w1.weight":  "markov_head.markov_w1.weight",   # Vision-Exp spelling
    "markov_head.markov_w2.weight":  "markov_head.markov_w2.weight",
    "markov_head.embed.weight":      "markov_head.markov_w1.weight",   # V4.1 spelling (DSparkMarkovHead)
    "markov_head.head.weight":       "markov_head.markov_w2.weight",
    "hc_head_base":                  "hc_head_base.weight",            # Vision-Exp: block 2's head mix
    "hc_head_fn":                    "hc_head_fn.weight",
    "hc_head_scale":                 "hc_head_scale.weight",
}
DRAFTER_UNNUMBERED = {"main_proj.weight", "main_norm.weight"}

# Engram (L242): projections ride in the layer shard under their HF names; the
# engine has no `blk.` spelling for them (slice 5 binds by these names or not at
# all -- inventing a rename here would be a second authority).  The row table
# is the side artifact.
ENGRAM_TENSORS = {"engram.wkv.weight", "engram.q_weight", "engram.k_weight"}
ENGRAM_SIDECAR = {"engram.embed.weight"}

EXPERT_PART_GGUF = {"w1": "gate", "w2": "down", "w3": "up"}
EXPERT_GGUF_PART = {v: k for k, v in EXPERT_PART_GGUF.items()}

VISION_PREFIX = ("vision.", "aligner.")
VISION_TOP = {"image_start", "image_end", "image_newline", "image_pad"}

# The companions an FP8/FP4 source (`.scale`) or an EXL3 source folds into the
# `.weight` entry.  `.mcg` is the codebook pulsar does not read: it maps to
# nothing, so the caller refuses the checkpoint (safetensors_lane Exl3Checkpoint).
COMPANION = {"scale", "suh", "svh", "mul1"}
PRIMARY = {"weight", "trellis"}

_EXPERT = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale|trellis|suh|svh|mul1)$")
_SLICE = re.compile(r"^(.+)\.slice\.\d+\.(trellis|suh|svh|mul1)$")
_LEAF = re.compile(r"^(.+)\.(weight|scale|trellis|suh|svh|mul1)$")
_BLOCK = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")


def _split_leaf(rest: str):
    """`<proj>.<leaf>` -> (container suffix `<proj>.weight`, is_companion), or
    (rest, False) for a bare name (hc_*, attn_sink, ape, q_weight ...)."""
    m = _SLICE.match(rest)
    if m:
        return m.group(1) + ".weight", m.group(2) in COMPANION
    m = _LEAF.match(rest)
    if m:
        return m.group(1) + ".weight", m.group(2) in COMPANION
    return rest, False


def _map_block(ns: str, idx: int, rest: str, shape: ModelShape) -> Mapped | None:
    is_mtp = ns == "mtp"
    if is_mtp:
        if idx >= shape.n_mtp:
            return None
        shard, prefix, family = f"mtp.{idx}", f"dspark.{idx}.", "mtp"
    else:
        if idx >= shape.n_layer:
            return None
        shard, prefix, family = f"layers.{idx}", f"blk.{idx}.", "layer"

    m = _EXPERT.match(rest)
    if m:
        e, part, leaf = int(m.group(1)), m.group(2), m.group(3)
        return Mapped(container_name=f"{ns}.{idx}.ffn.experts.{e}.{part}.weight",
                      gguf_name=f"{prefix}ffn_{EXPERT_PART_GGUF[part]}_exps.weight",
                      family="expert", shard=shard, layer=idx, expert=e, part=part,
                      is_scale=leaf in COMPANION)

    suffix, companion = _split_leaf(rest)
    if not is_mtp and suffix in ENGRAM_SIDECAR:
        return Mapped(f"{ns}.{idx}.{suffix}", f"{ns}.{idx}.{suffix}", family, shard, idx,
                      None, None, companion, emit=False)
    if not is_mtp and suffix in ENGRAM_TENSORS:
        return Mapped(f"{ns}.{idx}.{suffix}", f"{ns}.{idx}.{suffix}", family, shard, idx,
                      None, None, companion)

    if suffix in LAYER:
        gguf = prefix + LAYER[suffix]
        emit = True
        if not is_mtp and suffix == "ffn.gate.bias" and idx < shape.n_hash:
            emit = False                                   # drop rule 1
        if is_mtp and suffix == "ffn.gate.bias_vl":
            emit = False                                   # drop rule 2
    elif is_mtp and suffix in DRAFTER:
        gguf = ("dspark." if suffix in DRAFTER_UNNUMBERED else prefix) + DRAFTER[suffix]
        emit = True
    else:
        return None
    return Mapped(container_name=f"{ns}.{idx}.{suffix}", gguf_name=gguf, family=family,
                  shard=shard, layer=idx, expert=None, part=None, is_scale=companion, emit=emit)


def map_hf(hf_name: str, shape: ModelShape) -> Mapped | None:
    if hf_name.startswith(VISION_PREFIX) or hf_name in VISION_TOP:
        return Mapped(hf_name, hf_name, "vision", "vision", None, None, None, False)
    m = _BLOCK.match(hf_name)
    if m:
        return _map_block(m.group(1), int(m.group(2)), m.group(3), shape)
    suffix, companion = _split_leaf(hf_name)
    if suffix in TOP:
        return Mapped(suffix, TOP[suffix], "top", "top", None, None, None, companion)
    return None


# ---------------------------------------------------------------------------
# The inverse, for re-keying GGUF-era per-tensor maps (policy.rekey_format_map).
# Built from the same tables, so the two directions cannot disagree.
# ---------------------------------------------------------------------------
_TOP_INV = {v: k for k, v in TOP.items()}
_LAYER_INV = {v: k for k, v in LAYER.items()}
_DRAFTER_INV = {v: k for k, v in DRAFTER.items()
                if k not in ("markov_head.embed.weight", "markov_head.head.weight")}
_EXP_GGUF = re.compile(r"^(blk|dspark)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_GGUF_BLOCK = re.compile(r"^(blk|dspark)\.(\d+)\.(.+)$")


def hf_for_gguf(gguf_name: str, shape: ModelShape) -> str:
    """The HF/container name (or, for an expert stack, the fnmatch pattern over
    its per-expert entries) an engine name denotes.  Raises KeyError for a name
    no table row produces."""
    if gguf_name in _TOP_INV:
        return _TOP_INV[gguf_name]
    if gguf_name.startswith(VISION_PREFIX) or gguf_name in VISION_TOP:
        return gguf_name
    m = _EXP_GGUF.match(gguf_name)
    if m:
        ns = "layers" if m.group(1) == "blk" else "mtp"
        return f"{ns}.{int(m.group(2))}.ffn.experts.*.{EXPERT_GGUF_PART[m.group(3)]}.weight"
    m = _GGUF_BLOCK.match(gguf_name)
    if m:
        ns = "layers" if m.group(1) == "blk" else "mtp"
        idx, rest = int(m.group(2)), m.group(3)
        if rest in _LAYER_INV:
            hf = f"{ns}.{idx}.{_LAYER_INV[rest]}"
        elif ns == "mtp" and rest in _DRAFTER_INV:
            hf = f"{ns}.{idx}.{_DRAFTER_INV[rest]}"
        else:
            raise KeyError(gguf_name)
        got = map_hf(hf, shape)
        if got is None or got.gguf_name != gguf_name:
            raise KeyError(gguf_name)
        return hf
    if gguf_name.startswith("dspark.") and gguf_name[len("dspark."):] in DRAFTER_UNNUMBERED:
        return "mtp.0." + gguf_name[len("dspark."):]
    raise KeyError(gguf_name)

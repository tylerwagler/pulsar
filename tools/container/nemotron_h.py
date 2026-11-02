"""nemotron_h.py -- the Nemotron-H family in the one artifact builder (L296 N1): NVIDIA Nemotron 3.5 Super VL (the
Omni checkpoint: config.json architectures ["NemotronH_Omni_Reasoning_V3"], model_type "nemotron_h_omni", the
language model a NemotronHForCausalLM under `llm_config`) -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 Omni checkpoint: the authority for the config and every tensor.  Its language-model tensors sit
           under `language_model.` (`backbone.layers.B.`, `backbone.embeddings`, `backbone.norm_f`, `lm_head`, `mtp.`);
           the vision tower (`vision_model.`, `vision_projector.`, `mlp1.`, L331) is written by the recipe's `vision`
           / `vision_norm` rows or omitted whole by them (see "the vision tower" below).
  --exl3   an exllamav3 pack of the same checkpoint (exllamav3 1.6's NemotronH adapter: Linear keys = the HF names
           minus `language_model.` and `.weight`, e.g. `backbone.layers.1.mixer.experts.7.up_proj`): every Linear --
           the Mamba in / out projections, attention, the latent fc1 / fc2, the shared expert, the routed experts
           (UP-ONLY: one up and one down slice per expert, no gate; the engine's EXL3 routed front door runs them,
           moe.cpp PULSAR_MOE_ARM_ROWS_UP), the drafter's eh_proj -- and the head, byte copies of the pack (PackView).
           Norms, the conv, A_log / D / dt_bias, the router (the pack's is an F16 copy) and its selection bias are
           read from --hf as stored.
  --recipe a pulsar.recipe.v1 (format-maps/nemotron_h-*.json): required.
  --drafter lilicorr DIR   NVIDIA's LiLiCorr block drafter (ModelOpt W4A16_NVFP4, one model.safetensors), written
           into the container in place of the MTP module (the recipe omits it); see "the drafter input" below (L298).

The model (plan 121 section 1).  `llm_config.layers_block_type` lists the HF blocks, each ONE pre-norm residual
sublayer `x += mixer(RMSNorm(x))`: `mamba` (Mamba-2), `attention` (GQA, NoPE), `moe` (latent MoE).  Pulsar groups a
mixer with the MoE that follows it into one PULSAR_LAYER_BLOCKS layer (the 88 blocks of the full model are 48
layers): {mamba | attention} x {moe | none}.  A moe with no mixer before it, two moes in a row, and any other block
kind (`mlp`) are refused -- no such layer is declared.  The MTP module (`mtp_layers_block_type`, applied recursively
by the card) groups the same way under block `mtp`.

What the module checks (rule 1 -- refuse, never guess):
  * the config: every key the declaration reads is present; every switch the role table assumes is the value it
    assumes (no biases but the conv's, relu2 experts, silu Mamba activation, latent MoE, one shared expert, n_group 1,
    renormalised sigmoid top-k, no sliding window, untied head, time_step_limit [0, inf] -- the no-op vLLM serves);
  * the names, both ways: every HF tensor is a row of the declaration or under a vision prefix (consumed by name),
    and every row the declaration expects is present;
  * every tensor's dtype and shape against the declaration.

Roles.  Each tensor carries the RECIPE role (dense | shared_expert | expert_up / _down | router | norm | embed | head |
other) and the BIND role -- its identity for the loader (role / block / layer [/ index]), family-neutral words for
the blocks the engine composes (a Mamba-2 mixer `ssm_*`, a GQA mixer `attn_*`, a latent MoE `router`, `latent_*`,
`expert_*`, `shared_*`, the drafter's `draft_*`); `layer` is the PULSAR layer (the grouped one), not the HF block.
The table is ROLES below; nothing in it is named after Nemotron.

The container:
  * names ARE the HF names; routed experts fold to one entry per expert-projection (the HF per-expert tensors as
    they are), contiguous in expert order, part in {up_proj, down_proj}; each family in pulsar.experts carries
    entry_name, layer, role and block.  gguf_name of a family: `<prefix>.mixer.experts.<part>`.
  * layouts: bf16 / f32 native; exl3m_k* for a dense Linear, a routed expert-projection or the head from an EXL3
    pack (the rate per engine arm, exl3_rates.admit); mxfp8_lt for a dense Linear or the head from BF16.  The conv1d weight is declared [channels, kernel] (the HF [channels, 1, kernel]
    minus its unit dim; the same bytes).
  * shards: `vision` (primary: pulsar.kv, and the tower when written -- L331), `layers.0..L-1` (pulsar layers), `top` (embed,
    norm_f, head), `drafter` with --drafter, `mtp.0..N-1` when the recipe writes the MTP module (all of it or none;
    never beside --drafter: one drafter per container).
  * pulsar.kv: general.architecture = "nemotron_h"; llm_config verbatim under `nemotron_h.` (kv.config_kvs) except
    time_step_limit, which the declaration pins to [0, inf] and writes as `nemotron_h.time_step_limit.min` (f32 0)
    + `nemotron_h.time_step_limit.max_inf` (true); the builder's facts under `pulsar.*` (mtp_present, drafter --
    "lilicorr" or "" --, vision_present); with the tower its facts under `nemotron_h_vision.*`; with --drafter the
    drafter config under `lilicorr.*` and pulsar.source.drafter.
  * nvfp4 (the drafter's ModelOpt Linears): one U8 entry per Linear, named `X.weight`, declared [in, out] -- the
    nibbles, the 128x4-tiled E4M3 scales, the f32 global scale (producers.nvfp4).
  * the vision tower (L331; the engine's declaration is src/families/nemotron_h/nemotron_h.h section 2b): C-RADIO
    ViT-H/16 + the projector, BF16 as stored, in the `vision` shard.  Bind records: vit_patch, vit_cls, vit_pos (the
    [1, G*G, D] table declared [G*G, D]), vit_final_norm(_bias), proj_norm (mlp1.0), proj_fc1 (mlp1.1), proj_fc2
    (mlp1.3) at the top; per ViT block (block `vit`, layer i) vit_norm1 .. vit_fc2_bias -- the role words Parse's
    tower binds.  The LayerNorm weights and every bias are recipe role `vision_norm`, written f32 (f32_from_bf16:
    the shared row ops read f32), everything else role `vision`, written as stored.  Consumed by name and written
    nowhere: the video embedder (video is not served), the input conditioner (the processor normalises; its mean /
    std are checked against the config's), summary_idxs (the summary is not used).  The recipe writes all of it or
    omits all of it (pulsar.vision_present).  The facts the engine pins (nemotron_h_vision.*) are read from the config,
    each switch the tower's code assumes checked first (qkv bias, GELU MLP, no SwiGLU, no LayerScale, no drop path,
    pixel-shuffle v2 at ratio 0.5, the final LayerNorm present because the language model has MTP).
  * side files (emit --all writes, verify --all checks byte-equal): tokenizer.json, tokenizer_config.json,
    chat_template.jinja, generation_config.json (eos_token_id [2, 11]: the stop ids) and config.json, as shipped.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace

import drafters as DR
import entries as EN
import exl3_rates
import kv as KV
import producers as PR
import recipe as R
from names import Mapped

FAMILY = "nemotron_h"
MODEL_TYPE = "nemotron_h_omni"
ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
META = {"pulsar.family": FAMILY}
SETTINGS = ()
EXPERT_STACKS = False           # one HF tensor per expert-projection; grouped after the walk
GATE_UP_PARTS = ("up_proj",)    # up-only experts: the "gate / up" layout check sees the up stack alone
LM = "language_model."
BACKBONE = LM + "backbone."
MTP = LM + "mtp."
VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")
RADIO = "vision_model.radio_model."
EXL3 = set(exl3_rates.K2)
MIXERS = ("mamba", "attention")

# ---------------------------------------------------------------------------
# the role table: HF suffix (after the block's `mixer.`) -> (bind role, recipe role).  The block's own `norm.weight`
# is the mixer's pre-norm: ssm_pre_norm / attn_norm, or ffn_norm when the block is the layer's MoE.
# ---------------------------------------------------------------------------
MAMBA = {
    "in_proj.weight": ("ssm_in", "dense"),          # [E -> z | x | B | C | dt]
    "conv1d.weight": ("ssm_conv", "other"),         # [conv channels, 1, K] -> declared [channels, K]
    "conv1d.bias": ("ssm_conv_bias", "other"),
    "dt_bias": ("ssm_dt_bias", "other"),
    "A_log": ("ssm_a_log", "other"),
    "D": ("ssm_d", "other"),
    "norm.weight": ("ssm_out_norm", "norm"),        # the gated grouped RMSNorm: norm(y * silu(z)), group d_inner / G
    "out_proj.weight": ("ssm_out", "dense"),
}
ATTN = {
    "q_proj.weight": ("attn_q", "dense"),
    "k_proj.weight": ("attn_k", "dense"),
    "v_proj.weight": ("attn_v", "dense"),
    "o_proj.weight": ("attn_o", "dense"),
}
MOE = {
    "gate.weight": ("router", "router"),                            # [E -> n_expert], f32 logits
    "gate.e_score_correction_bias": ("router_select_bias", "router"),   # F32 [n_expert]: selection only
    "fc1_latent_proj.weight": ("latent_in", "dense"),               # E -> latent
    "fc2_latent_proj.weight": ("latent_out", "dense"),              # latent -> E
    "shared_experts.up_proj.weight": ("shared_up", "shared_expert"),
    "shared_experts.down_proj.weight": ("shared_down", "shared_expert"),
}
PART_ROLE = {"up_proj": ("expert_up", "expert_up"), "down_proj": ("expert_down", "expert_down")}
PRE_NORM = {"mamba": "ssm_pre_norm", "attention": "attn_norm", "moe": "ffn_norm"}
DRAFT_IN = {                                         # the drafter's first sublayer's own (the MTP input side)
    "enorm.weight": ("draft_embed_norm", "norm"),
    "hnorm.weight": ("draft_hidden_norm", "norm"),
    "eh_proj.weight": ("draft_in_proj", "dense"),    # [2E -> E] over [enorm(embed(t+1)) | hnorm(h_t)] (embed first)
}
DRAFT_OUT = {"final_layernorm.weight": ("draft_out_norm", "norm")}   # the drafter's last sublayer's own
TOP = {
    BACKBONE + "embeddings.weight": ("embed", "embed"),
    BACKBONE + "norm_f.weight": ("output_norm", "norm"),
    LM + "lm_head.weight": ("head", "head"),
}
_EXPERT = re.compile(r"^experts\.(\d+)\.(up_proj|down_proj)\.weight$")

# the switches the role table assumes: key -> the one value the declaration implements
ASSUME = {
    "attention_bias": False, "mlp_bias": False, "use_bias": False, "mamba_proj_bias": False, "use_conv_bias": True,
    "mlp_hidden_act": "relu2", "mamba_hidden_act": "silu", "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
    "norm_topk_prob": True, "sliding_window": None, "tie_word_embeddings": False, "residual_in_fp32": False,
    "moe_shared_expert_overlap": False,
}


def _need(d, key, what):
    if key not in d:
        raise SystemExit(f"config.json {what} has no {key!r} -- the Nemotron-H declaration reads it")
    return d[key]


def _assume(d, key, want, what):
    got = _need(d, key, what)
    if got != want:
        raise SystemExit(f"config.json {what}.{key} = {got!r}; the Nemotron-H role table assumes {want!r} -- refusing")


def _is_inf(v):
    return (isinstance(v, float) and math.isinf(v) and v > 0) or v == {"__float__": "Infinity"}


def group_layers(types, what):
    """HF block kinds -> pulsar layers: [(mixer kind, mixer block, moe block or None)].  A mixer opens a layer; a moe
    closes the open one; anything else refuses."""
    out = []
    for b, t in enumerate(types):
        if t in MIXERS:
            out.append([t, b, None])
        elif t == "moe":
            if not out or out[-1][2] is not None:
                raise SystemExit(f"config.json {what}[{b}] = 'moe' with no mixer before it (a layer is a mixer and "
                                 f"the MoE after it; no {{none, moe}} layer is declared) -- refusing")
            out[-1][2] = b
        else:
            raise SystemExit(f"config.json {what}[{b}] = {t!r}: not a block kind the declaration implements "
                             f"({list(MIXERS) + ['moe']}) -- refusing")
    return tuple(tuple(x) for x in out)


@dataclass(frozen=True)
class Shape:
    """The declared shape (the engine-side shape table refuses a container that disagrees)."""
    hidden: int
    vocab: int
    n_blocks: int
    layers: tuple               # ((mixer kind, mixer block, moe block | None), ...): the pulsar layers
    mtp_layers: tuple           # the drafter's, over its own sublayer indices
    n_mtp: int                  # num_nextn_predict_layers (the drafter modules; 1)
    attn: tuple                 # (n_head, n_kv, head_dim)
    ssm: tuple                  # (n_heads, head_dim, state, groups, conv_k)
    n_expert: int
    moe_ff: int
    latent: int
    shared_ff: int
    top_k: int
    recipe: str = ""
    mtp: bool = False           # the recipe writes the drafter
    drafter: str = ""           # the drafter input written (--drafter): "" or "lilicorr"
    vision: tuple = ()          # L331: the tower's facts ((key, kind, value), ...), declare_vision
    vision_written: bool = False   # the recipe writes the tower
    table: dict = field(default=None, compare=False, repr=False)   # {name: Mapped} (map_hf answers here)

    @property
    def n_layer(self):
        return len(self.layers)

    @property
    def d_inner(self):
        return self.ssm[0] * self.ssm[1]

    @property
    def conv_dim(self):
        return self.d_inner + 2 * self.ssm[3] * self.ssm[2]

    def __str__(self):
        kinds = "".join(("M" if m == "mamba" else "*") + ("E" if e is not None else "-") for m, _b, e in self.layers)
        return (f"nemotron_h: {self.n_blocks} blocks -> {self.n_layer} layers [{kinds}], {self.n_expert} experts "
                f"top-{self.top_k} on a {self.latent} latent + 1 shared, mtp {self.n_mtp}"
                f"{' written' if self.mtp else ' not written'}, drafter {self.drafter or 'none'}, vision {'written' if self.vision_written else 'not written'}, "
                f"recipe {self.recipe}")


def declare(top: dict) -> Shape:
    if top.get("model_type") != MODEL_TYPE or top.get("architectures") != [ARCHITECTURE]:
        raise SystemExit(f"config.json: model_type {top.get('model_type')!r} / architectures "
                         f"{top.get('architectures')!r}; nemotron_h.py reads {MODEL_TYPE} / [{ARCHITECTURE}]")
    c = _need(top, "llm_config", "the top level")
    w = "llm_config"
    _assume(c, "model_type", "nemotron_h", w)
    for k, v in ASSUME.items():
        _assume(c, k, v, w)
    tsl = _need(c, "time_step_limit", w)
    if not (isinstance(tsl, list) and len(tsl) == 2 and tsl[0] == 0 and _is_inf(tsl[1])):
        raise SystemExit(f"config.json llm_config.time_step_limit = {tsl!r}; the declaration implements [0, inf] "
                         f"(no dt clamp, as vLLM serves it) -- refusing")
    if _need(c, "moe_intermediate_size", w) != _need(c, "intermediate_size", w):
        raise SystemExit("config.json llm_config: moe_intermediate_size != intermediate_size (the experts' width)")
    types = list(_need(c, "layers_block_type", w))
    if "num_hidden_layers" in c and int(c["num_hidden_layers"]) != len(types):
        raise SystemExit(f"config.json llm_config.num_hidden_layers {c['num_hidden_layers']} != "
                         f"len(layers_block_type) {len(types)}")
    n_mtp = int(_need(c, "num_nextn_predict_layers", w))
    if n_mtp not in (0, 1):
        raise SystemExit(f"config.json llm_config.num_nextn_predict_layers = {n_mtp}: the drafter is one module "
                         f"(applied recursively) or none")
    mtp_types = list(_need(c, "mtp_layers_block_type", w)) if n_mtp else []
    sh = Shape(
        hidden=int(_need(c, "hidden_size", w)), vocab=int(_need(c, "vocab_size", w)), n_blocks=len(types),
        layers=group_layers(types, "llm_config.layers_block_type"),
        mtp_layers=group_layers(mtp_types, "llm_config.mtp_layers_block_type") if n_mtp else (), n_mtp=n_mtp,
        attn=(int(_need(c, "num_attention_heads", w)), int(_need(c, "num_key_value_heads", w)),
              int(_need(c, "head_dim", w))),
        ssm=(int(_need(c, "mamba_num_heads", w)), int(_need(c, "mamba_head_dim", w)), int(_need(c, "ssm_state_size", w)),
             int(_need(c, "n_groups", w)), int(_need(c, "conv_kernel", w))),
        n_expert=int(_need(c, "n_routed_experts", w)), moe_ff=int(_need(c, "moe_intermediate_size", w)),
        latent=int(_need(c, "moe_latent_size", w)), shared_ff=int(_need(c, "moe_shared_expert_intermediate_size", w)),
        top_k=int(_need(c, "num_experts_per_tok", w)))
    if sh.ssm[0] * sh.ssm[1] != int(_need(c, "expand", w)) * sh.hidden or sh.ssm[0] % sh.ssm[3]:
        raise SystemExit(f"config.json llm_config: mamba heads x head_dim {sh.d_inner} != expand x hidden, or "
                         f"{sh.ssm[0]} heads not a multiple of {sh.ssm[3]} groups")
    if not 0 < sh.top_k <= sh.n_expert:
        raise SystemExit(f"config.json llm_config: top-{sh.top_k} of {sh.n_expert}")
    sh = replace(sh, vision=declare_vision(top, sh.hidden, n_mtp))
    if n_mtp and len(sh.mtp_layers) != 1:
        raise SystemExit(f"config.json llm_config.mtp_layers_block_type {mtp_types}: one {{mixer, moe}} layer "
                         f"(the declaration's drafter)")
    return sh


def declare_vision(top: dict, llm_hidden: int, n_mtp: int):
    """The tower's facts from config.json (the engine pins each to its profile), every switch the tower's code assumes
    checked first."""
    v = _need(top, "vision_config", "the top level")
    w = "vision_config"
    for k, want in (("model_type", "radio"), ("qkv_bias", True), ("use_swiglu_ffn", False), ("hidden_act", "gelu"),
                    ("layerscale_value", 1.0), ("drop_path_rate", 0.0)):
        _assume(v, k, want, w)
    for k, want in (("downsample_ratio", 0.5), ("ps_version", "v2")):
        _assume(top, k, want, "the top level")
    D = int(_need(v, "hidden_size", w))
    if int(_need(top, "vit_hidden_size", "the top level")) != D:
        raise SystemExit("config.json: vit_hidden_size != vision_config.hidden_size")
    if not n_mtp:
        raise SystemExit("config.json: no MTP module, so no vision_final_layernorm (modeling_nemotron_h_omni.py "
                         "builds it only with MTP); the declaration has the norm -- refusing")
    heads, P = int(_need(v, "num_attention_heads", w)), int(_need(v, "patch_size", w))
    mlp = D * float(_need(v, "mlp_ratio", w))
    if D % heads or mlp != int(mlp) or int(_need(v, "max_img_size", w)) % P:
        raise SystemExit(f"config.json vision_config: {heads} heads over {D}, MLP {mlp}, max_img_size not a patch "
                         f"multiple")
    mean, std = list(_need(v, "norm_mean", w)), list(_need(v, "norm_std", w))
    facts = [("vit.width", "u32", D), ("vit.depth", "u32", int(_need(v, "num_hidden_layers", w))),
             ("vit.heads", "u32", heads), ("vit.head_dim", "u32", D // heads), ("vit.mlp", "u32", int(mlp)),
             ("vit.patch", "u32", P),
             ("vit.prefix", "u32", int(_need(v, "num_cls_tokens", w)) + int(_need(v, "num_registers", w))),
             ("vit.pos_grid", "u32", int(_need(v, "max_img_size", w)) // P),
             ("vit.norm_eps", "f32", float(_need(v, "layer_norm_eps", w))),
             ("proj.merge", "u32", 2), ("proj.hidden", "u32", int(_need(top, "projector_hidden_size", "the top level"))),
             ("proj.norm_eps", "f32", 1e-5),   # NemotronH_Omni_Reasoning_V3MLP's RMSNorm default (no config key)
             ("image.min_patches", "u32", int(_need(v, "min_num_patches", w))),
             ("image.max_patches", "u32", int(_need(v, "max_num_patches", w)))]
    facts += [(f"image.mean.{i}", "f32", float(x)) for i, x in enumerate(mean)]
    facts += [(f"image.std.{i}", "f32", float(x)) for i, x in enumerate(std)]
    if len(mean) != 3 or len(std) != 3:
        raise SystemExit("config.json vision_config: norm_mean / norm_std are 3 channels")
    return tuple(facts) + (("_llm_hidden", "u32", llm_hidden),)


def _vfact(sh: Shape, key: str):
    return next(v for k, _t, v in sh.vision if k == key)


def vision_expected(sh: Shape):
    """{HF name: (dtype, shape, Mapped)} of the tower: what is written (shard `vision`) and what is consumed by name."""
    D, L, M, P, G = (_vfact(sh, k) for k in ("vit.width", "vit.depth", "vit.mlp", "vit.patch", "vit.pos_grid"))
    NP, H, S = _vfact(sh, "vit.prefix"), _vfact(sh, "proj.hidden"), 4 * _vfact(sh, "vit.width")
    out = {}

    def add(name, dtype, shape, bind, role, layer=None):
        rec = (("role", bind),) + ((("block", "vit"), ("layer", layer)) if layer is not None else ())
        out[name] = (dtype, shape, Mapped(name, name, "vision", "vision", layer, None, None, False, role=role,
                                          bind=rec))

    def skip(name, dtype, shape):
        out[name] = (dtype, shape, Mapped(name, name, "vision", "vision", None, None, None, False, emit=False,
                                          role="vision"))
    pg = RADIO + "model.patch_generator."
    add(pg + "embedder.weight", "BF16", [D, 3 * P * P], "vit_patch", "vision")
    add(pg + "cls_token.token", "BF16", [NP, D], "vit_cls", "vision")
    add(pg + "pos_embed", "BF16", [1, G * G, D], "vit_pos", "vision")
    skip(pg + "video_embedder.weight", "BF16", [D, 2 * 3 * P * P])
    skip(RADIO + "input_conditioner.norm_mean", "F32", [3, 1, 1])
    skip(RADIO + "input_conditioner.norm_std", "F32", [3, 1, 1])
    skip("vision_model.summary_idxs", "I64", [2])
    for i in range(L):
        b = f"{RADIO}model.blocks.{i}."
        for s, shp, bind, role in (("norm1.weight", [D], "vit_norm1", "vision_norm"),
                                   ("norm1.bias", [D], "vit_norm1_bias", "vision_norm"),
                                   ("attn.qkv.weight", [3 * D, D], "vit_qkv", "vision"),
                                   ("attn.qkv.bias", [3 * D], "vit_qkv_bias", "vision_norm"),
                                   ("attn.proj.weight", [D, D], "vit_proj", "vision"),
                                   ("attn.proj.bias", [D], "vit_proj_bias", "vision_norm"),
                                   ("norm2.weight", [D], "vit_norm2", "vision_norm"),
                                   ("norm2.bias", [D], "vit_norm2_bias", "vision_norm"),
                                   ("mlp.fc1.weight", [M, D], "vit_fc1", "vision"),
                                   ("mlp.fc1.bias", [M], "vit_fc1_bias", "vision_norm"),
                                   ("mlp.fc2.weight", [D, M], "vit_fc2", "vision"),
                                   ("mlp.fc2.bias", [D], "vit_fc2_bias", "vision_norm")):
            add(b + s, "BF16", shp, bind, role, i)
    add("vision_projector.vision_final_layernorm.weight", "BF16", [D], "vit_final_norm", "vision_norm")
    add("vision_projector.vision_final_layernorm.bias", "BF16", [D], "vit_final_norm_bias", "vision_norm")
    add("mlp1.0.weight", "BF16", [S], "proj_norm", "vision")
    add("mlp1.1.weight", "BF16", [H, S], "proj_fc1", "vision")
    add("mlp1.3.weight", "BF16", [_vfact(sh, "_llm_hidden"), H], "proj_fc2", "vision")
    return out


def _mixer_table(sh: Shape, kind: str):
    """mixer suffix -> (dtype, HF shape, bind role, recipe role)."""
    H = sh.hidden
    t = {}
    if kind == "mamba":
        nh, _hd, N, G, K = sh.ssm
        DI, CD = sh.d_inner, sh.conv_dim
        for s, shp, dt in (("in_proj.weight", [DI + CD + nh, H], "BF16"), ("conv1d.weight", [CD, 1, K], "BF16"),
                           ("conv1d.bias", [CD], "BF16"), ("dt_bias", [nh], "BF16"), ("A_log", [nh], "BF16"),
                           ("D", [nh], "BF16"), ("norm.weight", [DI], "BF16"), ("out_proj.weight", [H, DI], "BF16")):
            t[s] = (dt, shp, *MAMBA[s])
    else:
        nq, nkv, hd = sh.attn
        for s, shp in (("q_proj.weight", [nq * hd, H]), ("k_proj.weight", [nkv * hd, H]),
                       ("v_proj.weight", [nkv * hd, H]), ("o_proj.weight", [H, nq * hd])):
            t[s] = ("BF16", shp, *ATTN[s])
    return t


def _moe_table(sh: Shape):
    H, X, F, Lt = sh.hidden, sh.n_expert, sh.moe_ff, sh.latent
    t = {}
    for s, shp, dt in (("gate.weight", [X, H], "BF16"), ("gate.e_score_correction_bias", [X], "F32"),
                       ("fc1_latent_proj.weight", [Lt, H], "BF16"), ("fc2_latent_proj.weight", [H, Lt], "BF16"),
                       ("shared_experts.up_proj.weight", [sh.shared_ff, H], "BF16"),
                       ("shared_experts.down_proj.weight", [H, sh.shared_ff], "BF16")):
        t[s] = (dt, shp, *MOE[s])
    for e in range(X):
        t[f"experts.{e}.up_proj.weight"] = ("BF16", [F, Lt], *PART_ROLE["up_proj"])
        t[f"experts.{e}.down_proj.weight"] = ("BF16", [Lt, F], *PART_ROLE["down_proj"])
    return t


def expected(sh: Shape):
    """{HF name: (dtype, shape, Mapped)} -- every language-model tensor the declaration says the checkpoint holds."""
    out = {}

    def add(name, dtype, shape, bind, role, shard, layer=None, block=None, expert=None, part=None, index=None):
        rec = (("role", bind),) + ((("block", block), ("layer", layer)) if block else ()) + \
              ((("index", index),) if index is not None else ())
        fam = "expert" if expert is not None else ("mtp" if block == "mtp" else "layer" if block else "top")
        out[name] = (dtype, shape, Mapped(name, name, fam, shard, layer, expert, part, False, role=role, bind=rec))

    def layer(prefix_of, il, grp, block, shard):
        kind, mb, eb = grp
        p = prefix_of(mb)
        add(p + "norm.weight", "BF16", [sh.hidden], PRE_NORM[kind], "norm", shard, il, block)
        for s, (dt, shp, bind, role) in _mixer_table(sh, kind).items():
            add(p + "mixer." + s, dt, shp, bind, role, shard, il, block)
        if eb is None:
            return
        p = prefix_of(eb)
        add(p + "norm.weight", "BF16", [sh.hidden], PRE_NORM["moe"], "norm", shard, il, block)
        for s, (dt, shp, bind, role) in _moe_table(sh).items():
            m = _EXPERT.match(s)
            idx = 0 if bind.startswith("shared_") else None
            add(p + "mixer." + s, dt, shp, bind, role, shard, il, block, expert=int(m.group(1)) if m else None,
                part=m.group(2) if m else None, index=idx)
    for il, grp in enumerate(sh.layers):
        layer(lambda b: f"{BACKBONE}layers.{b}.", il, grp, "layers", f"layers.{il}")
    if sh.n_mtp:
        grp = sh.mtp_layers[0]
        layer(lambda b: f"{MTP}layers.{b}.", 0, grp, "mtp", "mtp.0")
        first, last = grp[1], grp[2] if grp[2] is not None else grp[1]
        for s, (bind, role) in DRAFT_IN.items():
            shp = [sh.hidden, 2 * sh.hidden] if s == "eh_proj.weight" else [sh.hidden]
            add(f"{MTP}layers.{first}.{s}", "BF16", shp, bind, role, "mtp.0", 0, "mtp")
        for s, (bind, role) in DRAFT_OUT.items():
            add(f"{MTP}layers.{last}.{s}", "BF16", [sh.hidden], bind, role, "mtp.0", 0, "mtp")
    for name, (bind, role) in TOP.items():
        add(name, "BF16", [sh.vocab, sh.hidden] if role in ("embed", "head") else [sh.hidden], bind, role, "top")
    return out


def table(hf, decl: Shape):
    """{name: Mapped} for every HF tensor, checked both ways and dtype / shape against the declaration."""
    want = {**expected(decl), **vision_expected(decl)}
    mapped, unknown, wrong = {}, [], []
    for n in hf.names():
        if n not in want:
            unknown.append(n)
            continue
        dt, shp, m = want[n]
        if hf.dtype(n) != dt or hf.shape(n) != shp:
            wrong.append(f"{n}: {hf.dtype(n)} {hf.shape(n)}, the declaration says {dt} {shp}")
            continue
        mapped[n] = m
    if unknown:
        raise SystemExit(f"{len(unknown)} tensors the Nemotron-H naming table does not map -- refusing: {unknown[:5]}")
    if wrong:
        raise SystemExit(f"{len(wrong)} tensors disagree with the declared shape -- refusing: {wrong[:3]}")
    missing = sorted(n for n in want if n not in mapped)
    if missing:
        raise SystemExit(f"{len(missing)} tensors the declaration expects are missing from {hf.dir} -- refusing: "
                         f"{missing[:5]}")
    return mapped


# ---------------------------------------------------------------------------
# the drafter input (--drafter lilicorr DIR, L298): the shared block-drafter input (drafters.py) against this target
# ---------------------------------------------------------------------------
DRAFTERS = ("lilicorr",)


def with_drafter(hf, kind, path):
    """build.py --drafter KIND DIR: the checkpoint view carrying the drafter (drafters.py; refused by name unless KIND
    is one Nemotron-H's declaration reads and its config, tensors and quantization agree with it and the target)."""
    if kind not in DRAFTERS:
        raise SystemExit(f"--drafter {kind}: the Nemotron-H build reads {list(DRAFTERS)}")
    return DR.with_drafter(hf, kind, path, DR.Target(hf.decl.hidden, hf.decl.vocab, hf.decl.n_blocks))


# ---------------------------------------------------------------------------
# the family hooks (build.plan)
# ---------------------------------------------------------------------------
def source(hf):
    """The checkpoint as is: every tensor the plan writes is an HF tensor by its own name."""
    hf.decl = declare(hf.config["top_level"])
    hf.mapped = table(hf, hf.decl)
    return hf


def map_hf(name: str, shape: Shape) -> Mapped | None:
    return shape.table.get(name)


def _mtp_written(hf, recipe):
    res = recipe.resolve(hf.mapped)
    fm = {n: f for n, (f, _s) in res.items() if hf.mapped[n].shard.startswith("mtp.")}
    omitted = {n for n, f in fm.items() if f in R.CONSUMED}
    if omitted and len(omitted) != len(fm):
        raise SystemExit(f"{recipe.path}: writes part of the drafter and omits {sorted(omitted)[0]} -- all of it or "
                         f"none")
    return bool(fm) and not omitted


class PackView:
    """The --exl3 pack under the checkpoint's names: exllamav3 keys a NemotronH Linear without the Omni wrapper's
    `language_model.` prefix.  Answers the Exl3Checkpoint calls build.py / entries.py make for an HF Linear key."""

    def __init__(self, pack):
        self.pack = pack
        self.dir = pack.dir

    @staticmethod
    def key(k):
        if not k.startswith(LM):
            raise SystemExit(f"{k}: an EXL3 Linear outside {LM!r} -- the pack holds the language model only")
        return k[len(LM):]

    def names(self):
        return self.pack.names()

    def has_linear(self, key):
        return self.pack.has_linear(self.key(key))

    def linear(self, key, k, n, rates):
        return self.pack.linear(self.key(key), k, n, rates)


def _vision_written(hf, recipe):
    """The recipe writes the whole tower or none of it (pulsar.vision_present)."""
    res = recipe.resolve(hf.mapped)
    fm = {n: f for n, (f, _s) in res.items() if hf.mapped[n].family == "vision"}
    omitted = {n for n, f in fm.items() if f in R.CONSUMED}
    if omitted and len(omitted) != len(fm):
        raise SystemExit(f"{recipe.path}: writes part of the vision tower and omits {sorted(omitted)[0]} -- all of it "
                         f"or none")
    if not fm:
        raise SystemExit(f"{recipe.path}: no row for the vision tower (roles vision / vision_norm: native / f32, or "
                         f"omit)")
    return not omitted


def _check_conditioner(hf):
    """The checkpoint's input conditioner is the processor's normalisation: the same mean / std as the config's."""
    import numpy as np
    v = hf.config["top_level"]["vision_config"]
    for k in ("norm_mean", "norm_std"):
        got = np.frombuffer(hf.raw(f"{RADIO}input_conditioner.{k}"), dtype="<f4")
        if not np.array_equal(got, np.asarray(v[k], dtype=np.float32)):
            raise SystemExit(f"{RADIO}input_conditioner.{k} {got.tolist()} != vision_config.{k} {v[k]}")


def shape(hf, ctx):
    src = ctx.sources.get("exl3")
    if src is not None and not isinstance(src, PackView):
        ctx.sources["exl3"] = PackView(src)
    if ctx.sources.get("exl3_experts") is not None:
        raise SystemExit("--exl3-experts: the Nemotron-H build reads one pack (--exl3)")
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    mtp = _mtp_written(hf, ctx.recipe)
    drafter = hf.drafter.kind if isinstance(hf, DR.WithDrafter) else ""
    if drafter and mtp:
        raise SystemExit(f"--drafter {drafter}: {ctx.recipe.path} writes the MTP module too; a container carries one "
                         f"drafter -- omit the MTP rows (format-maps/nemotron_h-k4-lilicorr.json)")
    vision = _vision_written(hf, ctx.recipe)
    if vision:
        _check_conditioner(hf)
    return replace(hf.decl, recipe=ctx.recipe.name, mtp=mtp, drafter=drafter, vision_written=vision, table=hf.mapped)


def shard_order(shape):
    return (["vision"] + [f"layers.{i}" for i in range(shape.n_layer)] + ["top"]
            + (["drafter"] if shape.drafter else [])
            + ([f"mtp.{i}" for i in range(shape.n_mtp)] if shape.mtp else []))


def declared_shape(m, hshape):
    """The conv1d weight is declared [channels, kernel]: the HF [channels, 1, kernel] minus its unit dim (the same
    bytes, the kernel's view).  Everything else as the source."""
    if dict(m.bind).get("role") == "ssm_conv":
        c, one, k = hshape
        if one != 1:
            raise SystemExit(f"{m.container_name}: conv1d {hshape}, expected [channels, 1, kernel]")
        return [c, k]
    if dict(m.bind).get("role") == "vit_pos":   # L331: the [1, G*G, D] table minus its unit dim
        one, gg, d = hshape
        if one != 1:
            raise SystemExit(f"{m.container_name}: pos_embed {hshape}, expected [1, positions, width]")
        return [gg, d]
    if m.role == "drafter_quantized":           # NVFP4: the LOGICAL [out, in] of the U8 [out, in/2] nibble plane
        return list(PR.matrix_dims("U8+nvfp4", hshape))
    return list(hshape)


# what the loader binds, by recipe role: the dense Linears and the head EXL3 (dense arm) / mxfp8_lt / BF16; the
# routed experts BF16 or EXL3 (the up-only routed arm, L296; the rate by the engine's arm, exl3_rates.admit);
# everything else in its source dtype.  `omit` only for the drafter (_mtp_written keeps it whole).
def admit(hf, m, fmt, name):
    if m.shard == "drafter":                    # the --drafter input: ModelOpt's NVFP4 Linears nvfp4, the rest as stored
        allowed = {"nvfp4"} if m.role == "drafter_quantized" else {"native", PR.NATIVE_LAYOUT.get(hf.dtype(name))}
        if fmt not in allowed:
            raise SystemExit(f"{name}: the drafter's {m.role} tensor is written {sorted(a for a in allowed if a)}, "
                             f"not {fmt}")
        return
    native = PR.NATIVE_LAYOUT.get(hf.dtype(name))
    if m.role in ("vision", "vision_norm"):     # L331: the tower as stored, its LayerNorm / bias vectors f32; or omitted
        allowed = {"native", native, "omit"} if m.role == "vision" else {"f32", "omit"}
        if fmt not in allowed:
            raise SystemExit(f"{name}: the Nemotron-H loader binds this {m.role} tensor as "
                             f"{sorted(a for a in allowed if a)}, not {fmt}")
        return
    allowed = {"native", native}
    if m.role in ("dense", "shared_expert", "head"):
        allowed |= EXL3 | {"mxfp8_lt"}
    if m.role in ("expert_up", "expert_down"):
        allowed |= EXL3
    if m.shard.startswith("mtp."):
        allowed.add("omit")
    if fmt not in allowed:
        raise SystemExit(f"{name}: the Nemotron-H loader binds this {m.role} tensor as "
                         f"{sorted(a for a in allowed if a)}, not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit(f"{MODEL_TYPE}: pass --recipe (tools/container/format-maps/nemotron_h-*.json)")


def group_key(m):
    return (m.shard, m.layer, m.part)


def gate_up_key(rec):
    return rec["gguf_name"].rsplit(".", 1)[0]


def group_families(hf, key, slots, fmt, ctx):
    """One routed family from the per-expert HF tensors {e: (name, mapped)}: EXL3 ranges from the pack, or their
    BF16 bytes as stored."""
    shard, layer, part = key
    n_exp = max(slots) + 1
    if sorted(slots) != list(range(n_exp)) or n_exp != hf.decl.n_expert:
        raise SystemExit(f"{shard} {part}: experts {sorted(slots)[:3]}..., the declaration has {hf.decl.n_expert}")
    name0, m0 = slots[0]
    fmt0 = fmt[name0]
    odd = [slots[e][0] for e in range(n_exp) if fmt[slots[e][0]] != fmt0]
    if odd:
        raise SystemExit(f"{shard} {part}: experts of one family share a format and source; {odd[0]} is "
                         f"{fmt[odd[0]]}, expert 0 {fmt0}")
    layout, src = fmt0
    if layout == "native":
        layout = PR.NATIVE_LAYOUT[hf.dtype(name0)]
    out, inp = hf.shape(name0)
    names = [slots[e][0] for e in range(n_exp)]
    if layout in EXL3:
        srcs = [("ranges", EN.exl3_ranges(ctx.sources[src], x[:-len(".weight")], layout, inp, out)[0]) for x in names]
    elif layout == "bf16":
        srcs = [hf.copy_src(x)[0] for x in names]
    else:
        raise SystemExit(f"{name0}: a routed expert is written EXL3 or bf16, not {layout}")
    pre = name0[:name0.index(".experts.") + len(".experts.")]
    rec = dict(m0.bind)
    return [dict(gguf_name=pre + part, part=part, role=m0.role, layout=layout, inp=inp, n=out, entry_names=names,
                 srcs=srcs,
                 extras={"entry_name": f"{pre}{{e}}.{part}.weight", "layer": layer, "role": rec["role"],
                         "block": rec["block"]})]


# ---------------------------------------------------------------------------
# pulsar.kv
# ---------------------------------------------------------------------------
def build_kv(hf, ctx):
    """general.*; llm_config verbatim (kv.config_kvs) with time_step_limit as its two checked facts; the builder's."""
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    c = dict(hf.config["top_level"]["llm_config"])
    c.pop("time_step_limit")        # declare() pinned it to [0, inf]
    kvs = [("general.architecture", "string", FAMILY), ("general.type", "string", "model"),
           ("general.name", "string", "Nemotron-H")]
    KV.config_kvs(FAMILY, c, kvs)
    kvs += [(f"{FAMILY}.time_step_limit.min", "f32", 0.0), (f"{FAMILY}.time_step_limit.max_inf", "bool", True),
            ("pulsar.recipe", "string", ctx.recipe.name),
            ("pulsar.recipe.sha256", "string", ctx.recipe.sha256 or ""),
            ("pulsar.mtp_present", "bool", ctx.shape.mtp),
            ("pulsar.drafter", "string", ctx.shape.drafter),
            ("pulsar.vision_present", "bool", ctx.shape.vision_written)]
    if ctx.shape.vision_written:
        kvs += [(f"nemotron_h_vision.{k}", t, v) for k, t, v in ctx.shape.vision if not k.startswith("_")]
    if ctx.shape.drafter:
        kvs += hf.drafter.kvs
    src = ctx.sources.get("exl3")
    if src is not None:
        kvs.append(("pulsar.source.exl3", "string", os.path.basename(os.path.normpath(src.dir))))
    kvs.append(("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))))
    if ctx.shape.drafter:
        kvs.append(("pulsar.source.drafter", "string", os.path.basename(os.path.normpath(hf.drafter.dir))))
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out


# ---------------------------------------------------------------------------
# side files
# ---------------------------------------------------------------------------
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def side_files(hf, ctx):
    """[(container name, source path)]: the tokenizer and template the engine loads beside the shards
    (src/families/nemotron_h/nemotron_h_tokenizer.cpp), from --tokenizer DIR (default: the HF dir), and config.json
    the checkpoint's own."""
    files = [(n, os.path.join(ctx.tokenizer_dir, n)) for n in SIDE] + [("config.json", os.path.join(hf.dir, "config.json"))]
    missing = [src for _, src in files if not os.path.isfile(src)]
    if missing:
        raise SystemExit(f"the Nemotron-H container carries the checkpoint's tokenizer: {', '.join(missing)} missing "
                         f"(--tokenizer DIR names a directory with {', '.join(SIDE)})")
    return files

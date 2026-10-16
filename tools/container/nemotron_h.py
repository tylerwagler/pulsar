"""nemotron_h.py -- the Nemotron-H family in the one artifact builder (L296 N1): NVIDIA Nemotron 3.5 Super VL (the
Omni checkpoint: config.json architectures ["NemotronH_Omni_Reasoning_V3"], model_type "nemotron_h_omni", the
language model a NemotronHForCausalLM under `llm_config`) -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 Omni checkpoint: the authority for the config and every tensor.  Its language-model tensors sit
           under `language_model.` (`backbone.layers.B.`, `backbone.embeddings`, `backbone.norm_f`, `lm_head`, `mtp.`);
           the vision tower (`vision_model.`, `vision_projector.`, `mlp1.`) is consumed by name and written nowhere in
           v1 (plan 121 decision 3: text first).
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
  * shards: `vision` (primary: pulsar.kv only -- no tower in v1), `layers.0..L-1` (pulsar layers), `top` (embed,
    norm_f, head), `drafter` with --drafter, `mtp.0..N-1` when the recipe writes the MTP module (all of it or none;
    never beside --drafter: one drafter per container).
  * pulsar.kv: general.architecture = "nemotron_h"; llm_config verbatim under `nemotron_h.` (kv.config_kvs) except
    time_step_limit, which the declaration pins to [0, inf] and writes as `nemotron_h.time_step_limit.min` (f32 0)
    + `nemotron_h.time_step_limit.max_inf` (true); the builder's facts under `pulsar.*` (mtp_present, drafter --
    "lilicorr" or "" --, vision_present false); with --drafter the drafter config under `lilicorr.*` and
    pulsar.source.drafter.
  * nvfp4 (the drafter's ModelOpt Linears): one U8 entry per Linear, named `X.weight`, declared [in, out] -- the
    nibbles, the 128x4-tiled E4M3 scales, the f32 global scale (producers.nvfp4).
  * side files (emit --all writes, verify --all checks byte-equal): tokenizer.json, tokenizer_config.json,
    chat_template.jinja, generation_config.json (eos_token_id [2, 11]: the stop ids) and config.json, as shipped.
"""
from __future__ import annotations

import fnmatch
import math
import os
import re
from dataclasses import dataclass, field, replace

import entries as EN
import exl3_rates
import kv as KV
import producers as PR
import recipe as R
from hf_source import HFCheckpoint
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
VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")   # consumed by name in v1
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
                f"{' written' if self.mtp else ' not written'}, drafter {self.drafter or 'none'}, vision not written (v1), "
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
    if n_mtp and len(sh.mtp_layers) != 1:
        raise SystemExit(f"config.json llm_config.mtp_layers_block_type {mtp_types}: one {{mixer, moe}} layer "
                         f"(the declaration's drafter)")
    return sh


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
    want = expected(decl)
    mapped, unknown, wrong = {}, [], []
    for n in hf.names():
        if n.startswith(VISION_PREFIXES):
            mapped[n] = Mapped(n, n, "vision", "vision", None, None, None, False, emit=False, role="vision")
            continue
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
# the drafter input (--drafter lilicorr DIR, L298): NVIDIA's LiLiCorr block drafter for Nemotron 3.5 Super -- a DFlash
# block drafter (a 5-layer Qwen3-style non-causal SWA backbone with DFlash2's grouped convolutions, fed by the target's
# hidden states at dflash_config.target_layer_ids through `fc`) plus the LiLiCorr candidate-lattice head.  Reference
# semantics: vLLM's lilicorr.py / qwen3_dflash.py / qwen3_dflash2.py.  The checkpoint is ModelOpt W4A16_NVFP4: a
# quantized Linear is `X.weight` U8 [out, in/2] + `X.weight_scale` F8_E4M3 [out, in/16] + `X.weight_scale_2` F32 [],
# written as ONE `nvfp4` entry (producers.nvfp4) named `X.weight`; everything else BF16, written as stored.
#
# Container names are the drafter's HF names under DRAFTER (`drafter.layers.0.mlp.gate_proj.weight`), all in shard
# `drafter` (after `top`).  Bind records: the backbone block `drafter` (layer = drafter layer for per-layer tensors,
# none for the four top-level ones), the lattice head block `lilicorr` (layer = lattice layer for its layers' tensors).
# Recipe roles: `drafter_quantized` (the NVFP4 Linears; admitted as nvfp4 only) and `drafter_native` (the rest;
# native only).
#
# The role tables below: HF suffix -> (bind role, NVFP4?).  Shapes are declared from the drafter config (_lili_table).
# ---------------------------------------------------------------------------
DRAFTERS = ("lilicorr",)
DRAFTER = "drafter."
LILI_TOP = {
    "fc.weight": ("fc", True),                       # [H <- n_taps * H]: the concatenated target hidden states
    "hidden_norm.weight": ("hidden_norm", False),
    "norm.weight": ("out_norm", False),
    "lm_head.weight": ("head", True),                # [V <- H]: has_own_lm_head
}
LILI_LAYER = {
    "input_layernorm.weight": ("in_norm", False),
    "post_attention_layernorm.weight": ("post_norm", False),
    "self_attn.q_proj.weight": ("attn_q", False),
    "self_attn.k_proj.weight": ("attn_k", False),
    "self_attn.v_proj.weight": ("attn_v", False),
    "self_attn.o_proj.weight": ("attn_o", False),
    "self_attn.q_norm.weight": ("q_norm", False),
    "self_attn.k_norm.weight": ("k_norm", False),
    "attention_conv.base_kernel": ("attn_conv_base", False),            # [2, taps, H]
    "attention_conv.kernel_projection.weight": ("attn_conv_proj", False),   # [2 * taps * H / group <- H]
    "mlp_conv.base_kernel": ("mlp_conv_base", False),
    "mlp_conv.kernel_projection.weight": ("mlp_conv_proj", False),
    "mlp.gate_proj.weight": ("mlp_gate", True),
    "mlp.up_proj.weight": ("mlp_up", True),
    "mlp.down_proj.weight": ("mlp_down", True),
}
LILI_HEAD = {                                        # under `lilicorr.`; token_proj only when lilicorr_hidden != H
    "token_proj.weight": ("token_proj", True), "token_proj.bias": ("token_proj_bias", False),
    "pass_hidden_proj.weight": ("pass_hidden_proj", True), "pass_hidden_proj.bias": ("pass_hidden_proj_bias", False),
    "context_proj.weight": ("context_proj", True), "context_proj.bias": ("context_proj_bias", False),
    "feature_norm.weight": ("feature_norm", False), "feature_norm.bias": ("feature_norm_bias", False),   # LayerNorm(5)
    "feature_mlp.up_proj.weight": ("feature_up", False), "feature_mlp.up_proj.bias": ("feature_up_bias", False),
    "feature_mlp.down_proj.weight": ("feature_down", False),
    "feature_mlp.down_proj.bias": ("feature_down_bias", False),
    "slot_embedding": ("slot_embedding", False),             # [1, 1, block - 1, 1, L]
    "rank_embedding": ("rank_embedding", False),             # [1, 1, 1, topk, L]
    "relative_slot_bias": ("relative_slot_bias", False),     # [heads, 2 * block - 1]
    "same_slot_bias": ("same_slot_bias", False),             # [heads]
    "output_norm.weight": ("output_norm", False),
    "anchor_norm.weight": ("anchor_norm", False),
    "factor_input_proj.weight": ("factor_in", False), "factor_input_proj.bias": ("factor_in_bias", False),
    "out_head.weight": ("out_head", False), "out_head.bias": ("out_head_bias", False),
    "in_head.weight": ("in_head", False), "in_head.bias": ("in_head_bias", False),
    "anchor_out_head.weight": ("anchor_out_head", False), "anchor_out_head.bias": ("anchor_out_head_bias", False),
}
LILI_HEAD_LAYER = {                                  # under `lilicorr.layers.i.`
    "attn_norm.weight": ("attn_norm", False),
    "attn.in_proj_weight": ("attn_in", False), "attn.in_proj_bias": ("attn_in_bias", False),   # fused q | k | v
    "attn.out_proj.weight": ("attn_out", True), "attn.out_proj.bias": ("attn_out_bias", False),
    "mlp_norm.weight": ("mlp_norm", False),
    "mlp.up_proj.weight": ("mlp_up", True), "mlp.up_proj.bias": ("mlp_up_bias", False),
    "mlp.down_proj.weight": ("mlp_down", True), "mlp.down_proj.bias": ("mlp_down_bias", False),
}

# The drafter config.json: every top-level key is WRITTEN under `lilicorr.<key>` (kv.config_kvs), IGNORED for the
# reason given, or refused.  dflash_config is written flattened as `lilicorr.dflash.<key>`; every key of it is written.
LILI_WRITE = ("block_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "intermediate_size", "hidden_act", "rms_norm_eps", "vocab_size", "max_position_embeddings",
              "attention_bias", "rope_theta", "tie_word_embeddings", "num_target_layers", "layer_types",
              "sliding_window", "use_sliding_window", "max_window_layers", "has_own_lm_head", "draft_vocab_size")
LILI_IGNORE = {
    "architectures": "pinned [LiLiCorrDraftModel]; pulsar.drafter names the drafter",
    "model_type": "pinned qwen3 (the backbone's HF class); pulsar.drafter names the drafter",
    "dflash_config": "written flattened as lilicorr.dflash.<key>",
    "quantization_config": "consumed: checked against the header both ways; the nvfp4 layout carries it",
    "initializer_range": "training only",
    "attention_dropout": "training only (pinned 0.0)",
    "torch_dtype": "every tensor's dtype is checked against the declaration instead",
    "rope_scaling": "pinned null: no RoPE scaling",
}
LILI_DFLASH = ("mask_token_id", "target_layer_ids", "causal", "projector_type", "lilicorr_enabled",
               "lilicorr_candidate_topk", "lilicorr_hidden_size", "lilicorr_num_layers", "lilicorr_num_heads",
               "lilicorr_mlp_ratio", "lilicorr_factor_dim", "lilicorr_vector_eps", "lilicorr_logit_scale",
               "conv_kernel_size", "conv_group_size", "use_swa", "swa_window_size")
# the switches the drafter declaration implements: (block, key) -> the one value
LILI_PIN = {
    ("", "architectures"): ["LiLiCorrDraftModel"], ("", "model_type"): "qwen3", ("", "hidden_act"): "silu",
    ("", "attention_bias"): False, ("", "tie_word_embeddings"): False, ("", "has_own_lm_head"): True,
    ("", "attention_dropout"): 0.0, ("", "rope_scaling"): None,
    ("dflash_config", "causal"): False, ("dflash_config", "projector_type"): "lilicorr",
    ("dflash_config", "lilicorr_enabled"): True, ("dflash_config", "use_swa"): True,
    ("quantization_config", "quant_algo"): "W4A16_NVFP4", ("quantization_config", "group_size"): 16,
    ("quantization_config", "quant_method"): "modelopt_fp4", ("quantization_config", "kv_cache_quant_algo"): None,
}
LILI_QUANT_KEYS = ("quant_algo", "kv_cache_quant_algo", "group_size", "exclude_modules", "quant_method", "ignore")


def _lili_need(d, key, what):
    if key not in d:
        raise SystemExit(f"drafter config.json {what} has no {key!r} -- the LiLiCorr declaration reads it")
    return d[key]


def _lili_int(d, key, what, lo=1):
    v = _lili_need(d, key, what)
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise SystemExit(f"drafter config.json {what}.{key} = {v!r}: an integer >= {lo}")
    return v


def lili_config(top: dict, target: Shape):
    """The drafter config checked (every key known, the switches pinned, the target agreeing) -> (dims, kvs)."""
    unknown = sorted(set(top) - set(LILI_WRITE) - set(LILI_IGNORE))
    if unknown:
        raise SystemExit(f"drafter config.json: key {unknown[0]!r} is neither written nor ignored by the LiLiCorr "
                         f"declaration -- refusing (all unknown: {unknown})")
    dfl = _lili_need(top, "dflash_config", "the top level")
    q = _lili_need(top, "quantization_config", "the top level")
    for blk, d, known in (("dflash_config", dfl, LILI_DFLASH), ("quantization_config", q, LILI_QUANT_KEYS)):
        bad = sorted(set(d) - set(known))
        if bad:
            raise SystemExit(f"drafter config.json {blk}: key {bad[0]!r} is not one the LiLiCorr declaration reads "
                             f"-- refusing (all unknown: {bad})")
    for (blk, k), want in LILI_PIN.items():
        d = {"": top, "dflash_config": dfl, "quantization_config": q}[blk]
        got = _lili_need(d, k, blk or "the top level")
        if got != want or type(got) is not type(want):
            raise SystemExit(f"drafter config.json {blk + '.' if blk else ''}{k} = {got!r}; the LiLiCorr declaration "
                             f"implements {want!r} -- refusing")
    w = "the top level"
    H, nl = _lili_int(top, "hidden_size", w), _lili_int(top, "num_hidden_layers", w)
    nq, nkv, hd = (_lili_int(top, "num_attention_heads", w), _lili_int(top, "num_key_value_heads", w),
                   _lili_int(top, "head_dim", w))
    V, nt = _lili_int(top, "vocab_size", w), _lili_int(top, "num_target_layers", w)
    if H != target.hidden or V != target.vocab or _lili_need(top, "draft_vocab_size", w) != V:
        raise SystemExit(f"drafter config.json: hidden_size {H} / vocab_size {V} / draft_vocab_size "
                         f"{top['draft_vocab_size']}; the target's hidden is {target.hidden} and vocab {target.vocab} "
                         f"(LiLiCorr candidates need the full target vocabulary) -- refusing")
    if nt != target.n_blocks:
        raise SystemExit(f"drafter config.json num_target_layers {nt}; the target has {target.n_blocks} HF blocks")
    lt = _lili_need(top, "layer_types", w)
    if lt != ["sliding_attention"] * nl:
        raise SystemExit(f"drafter config.json layer_types {lt}: the declaration implements {nl} sliding_attention "
                         f"layers (use_swa)")
    tl = _lili_need(dfl, "target_layer_ids", "dflash_config")
    if (not isinstance(tl, list) or not tl or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < nt
                                                  for x in tl)):
        raise SystemExit(f"drafter config.json dflash_config.target_layer_ids {tl!r}: HF block indices < {nt}")
    if q["exclude_modules"] != q["ignore"]:
        raise SystemExit("drafter config.json quantization_config: exclude_modules != ignore -- one exclude list")
    d = "dflash_config"
    taps, group = _lili_int(dfl, "conv_kernel_size", d), _lili_int(dfl, "conv_group_size", d)
    L = _lili_int(dfl, "lilicorr_hidden_size", d)
    nh, ratio = _lili_int(dfl, "lilicorr_num_heads", d), _lili_need(dfl, "lilicorr_mlp_ratio", d)
    if H % group or L % nh or float(L * ratio) != int(L * ratio):
        raise SystemExit(f"drafter config.json: conv_group_size {group} must divide hidden {H}, lilicorr heads {nh} "
                         f"divide {L}, and {L} x mlp_ratio {ratio} be whole")
    for k in ("mask_token_id", "swa_window_size", "lilicorr_factor_dim", "lilicorr_num_layers",
              "lilicorr_candidate_topk"):
        _lili_int(dfl, k, d, lo=0 if k == "mask_token_id" else 1)
    if not dfl["mask_token_id"] < V:
        raise SystemExit(f"drafter config.json dflash_config.mask_token_id {dfl['mask_token_id']} >= vocab {V}")
    dims = dict(H=H, n_layer=nl, nq=nq, nkv=nkv, hd=hd, inter=_lili_int(top, "intermediate_size", w), V=V,
                n_taps=len(tl), taps=taps, group=group, L=L, nh=nh, mlp=int(L * ratio),
                F=dfl["lilicorr_factor_dim"], n_lat=dfl["lilicorr_num_layers"], block=_lili_int(top, "block_size", w),
                topk=dfl["lilicorr_candidate_topk"], exclude=list(q["exclude_modules"]))
    kvs = []
    for k in LILI_WRITE:
        n0 = len(kvs)
        KV.config_kvs(f"lilicorr.{k}", _lili_need(top, k, w), kvs)
        if len(kvs) == n0:
            raise SystemExit(f"drafter config.json {k} = {top[k]!r}: nothing to write (null / empty) -- refusing")
    for k in LILI_DFLASH:
        KV.config_kvs(f"lilicorr.dflash.{k}", dfl[k], kvs)
    return dims, kvs


def _lili_table(dm):
    """{drafter HF name: (dtype, shape, bind role, block, layer, nvfp4?)} -- every tensor the declaration expects."""
    H, L, hd = dm["H"], dm["L"], dm["hd"]
    kp = [2 * dm["taps"] * H // dm["group"], H]
    lay = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
           "self_attn.q_proj.weight": [dm["nq"] * hd, H], "self_attn.k_proj.weight": [dm["nkv"] * hd, H],
           "self_attn.v_proj.weight": [dm["nkv"] * hd, H], "self_attn.o_proj.weight": [H, dm["nq"] * hd],
           "self_attn.q_norm.weight": [hd], "self_attn.k_norm.weight": [hd],
           "attention_conv.base_kernel": [2, dm["taps"], H], "attention_conv.kernel_projection.weight": kp,
           "mlp_conv.base_kernel": [2, dm["taps"], H], "mlp_conv.kernel_projection.weight": kp,
           "mlp.gate_proj.weight": [dm["inter"], H], "mlp.up_proj.weight": [dm["inter"], H],
           "mlp.down_proj.weight": [H, dm["inter"]]}
    top = {"fc.weight": [H, dm["n_taps"] * H], "hidden_norm.weight": [H], "norm.weight": [H],
           "lm_head.weight": [dm["V"], H]}
    head = {"token_proj.weight": [L, H], "token_proj.bias": [L], "pass_hidden_proj.weight": [L, H],
            "pass_hidden_proj.bias": [L], "context_proj.weight": [L, H], "context_proj.bias": [L],
            "feature_norm.weight": [5], "feature_norm.bias": [5], "feature_mlp.up_proj.weight": [L, 5],
            "feature_mlp.up_proj.bias": [L], "feature_mlp.down_proj.weight": [L, L], "feature_mlp.down_proj.bias": [L],
            "slot_embedding": [1, 1, dm["block"] - 1, 1, L], "rank_embedding": [1, 1, 1, dm["topk"], L],
            "relative_slot_bias": [dm["nh"], 2 * dm["block"] - 1], "same_slot_bias": [dm["nh"]],
            "output_norm.weight": [L], "anchor_norm.weight": [L], "factor_input_proj.weight": [L, 3 * L],
            "factor_input_proj.bias": [L], "out_head.weight": [dm["F"], L], "out_head.bias": [dm["F"]],
            "in_head.weight": [dm["F"], L], "in_head.bias": [dm["F"]], "anchor_out_head.weight": [dm["F"], L],
            "anchor_out_head.bias": [dm["F"]]}
    if L == H:                                       # vLLM: token_proj is the identity (no tensors)
        del head["token_proj.weight"], head["token_proj.bias"]
    hl = {"attn_norm.weight": [L], "attn.in_proj_weight": [3 * L, L], "attn.in_proj_bias": [3 * L],
          "attn.out_proj.weight": [L, L], "attn.out_proj.bias": [L], "mlp_norm.weight": [L],
          "mlp.up_proj.weight": [dm["mlp"], L], "mlp.up_proj.bias": [dm["mlp"]],
          "mlp.down_proj.weight": [L, dm["mlp"]], "mlp.down_proj.bias": [L]}
    out = {}

    def add(name, shape, role_q, block, layer):
        out[name] = ("BF16", shape, role_q[0], block, layer, role_q[1])
    for s, shp in top.items():
        add(s, shp, LILI_TOP[s], "drafter", None)
    for i in range(dm["n_layer"]):
        for s, shp in lay.items():
            add(f"layers.{i}.{s}", shp, LILI_LAYER[s], "drafter", i)
    for s, shp in head.items():
        add(f"lilicorr.{s}", shp, LILI_HEAD[s], "lilicorr", None)
    for i in range(dm["n_lat"]):
        for s, shp in hl.items():
            add(f"lilicorr.layers.{i}.{s}", shp, LILI_HEAD_LAYER[s], "lilicorr", i)
    return out


def lili_check_exclude(table, exclude):
    """quantization_config's exclude list against the declaration, both ways: every Linear the declaration holds NVFP4
    matches no exclude pattern, every BF16 Linear matches one."""
    bad = []
    for name, (_dt, shape, _r, _b, _l, q) in table.items():
        if not name.endswith(".weight") or len(shape) != 2:
            continue                                  # not a Linear (norms, embeddings, the fused in_proj_weight)
        mod = name[:-len(".weight")]
        hit = [p for p in exclude if fnmatch.fnmatchcase(mod, p)]
        if q and hit:
            bad.append(f"{mod} is NVFP4 in the declaration but excluded by {hit[0]!r}")
        elif not q and not hit:
            bad.append(f"{mod} is BF16 in the declaration but no exclude pattern names it (ModelOpt would quantize it)")
    if bad:
        raise SystemExit(f"drafter config.json quantization_config.exclude_modules disagrees with the LiLiCorr "
                         f"declaration -- refusing: {bad[:3]}")


class Drafter:
    """The checked drafter: its checkpoint, its {container name: Mapped}, its kv triples."""

    def __init__(self, kind, path, target: Shape):
        if kind not in DRAFTERS:
            raise SystemExit(f"--drafter {kind}: the Nemotron-H build reads {list(DRAFTERS)}")
        self.kind, self.dir = kind, path
        self.ckpt = HFCheckpoint(path)
        dims, self.kvs = lili_config(self.ckpt.config["top_level"], target)
        table = _lili_table(dims)
        lili_check_exclude(table, dims["exclude"])
        want = {}
        for n, (dt, shp, bind, block, layer, q) in table.items():
            rec = (("role", bind), ("block", block)) + ((("layer", layer),) if layer is not None else ())
            role = "drafter_quantized" if q else "drafter_native"
            if q:
                out, inp = shp
                if inp % 64:
                    raise SystemExit(f"{n}: NVFP4 in={inp} is not a multiple of 64 (the nvfp4 layout's scale tiles)")
                want[n] = ("U8", [out, inp // 2], Mapped(DRAFTER + n, DRAFTER + n, "drafter", "drafter", layer, None,
                                                        None, False, role=role, bind=rec))
                for sfx, sdt, sshp in (("_scale", "F8_E4M3", [out, inp // 16]), ("_scale_2", "F32", [])):
                    want[n + sfx] = (sdt, sshp, Mapped(DRAFTER + n + sfx, DRAFTER + n + sfx, "drafter", "drafter",
                                                       layer, None, None, True, role=role))
            else:
                want[n] = (dt, shp, Mapped(DRAFTER + n, DRAFTER + n, "drafter", "drafter", layer, None, None, False,
                                           role=role, bind=rec))
        have = self.ckpt.names()
        unknown = [n for n in have if n not in want]
        if unknown:
            raise SystemExit(f"{len(unknown)} drafter tensors the LiLiCorr declaration does not map -- refusing: "
                             f"{unknown[:5]}")
        missing = sorted(n for n in want if n not in set(have))
        if missing:
            raise SystemExit(f"{len(missing)} tensors the LiLiCorr declaration expects are missing from {path} -- "
                             f"refusing: {missing[:5]}")
        wrong = [f"{n}: {self.ckpt.dtype(n)} {self.ckpt.shape(n)}, the declaration says {dt} {shp}"
                 for n, (dt, shp, _m) in want.items() if (self.ckpt.dtype(n), self.ckpt.shape(n)) != (dt, shp)]
        if wrong:
            raise SystemExit(f"{len(wrong)} drafter tensors disagree with the declaration -- refusing: {wrong[:3]}")
        self.mapped = {m.container_name: m for _dt, _s, m in want.values()}


class WithDrafter:
    """The checkpoint view with the drafter's tensors beside the target's, under DRAFTER-prefixed names: what
    build.py and entries.py read (names / has / shape / dtype / span / raw / copy_src); the rest is the target's."""

    def __init__(self, hf, drafter: Drafter):
        self.hf, self.drafter = hf, drafter
        self.dir, self.config, self.generation_config = hf.dir, hf.config, hf.generation_config
        self.decl = hf.decl
        self.mapped = {**hf.mapped, **drafter.mapped}

    def _of(self, name):
        return (self.drafter.ckpt, name[len(DRAFTER):]) if name.startswith(DRAFTER) else (self.hf, name)

    def names(self):
        return self.hf.names() + sorted(self.drafter.mapped)

    def has(self, name):
        src, n = self._of(name)
        return src.has(n)

    def shape(self, name):
        src, n = self._of(name)
        return src.shape(n)

    def dtype(self, name):
        src, n = self._of(name)
        return src.dtype(n)

    def span(self, name):
        src, n = self._of(name)
        return src.span(n)

    def raw(self, name):
        src, n = self._of(name)
        return src.raw(n)

    def copy_src(self, name):
        src, n = self._of(name)
        return src.copy_src(n)


def with_drafter(hf, kind, path):
    """build.py --drafter KIND DIR: the checkpoint view carrying the drafter (refused by name unless KIND is one of
    DRAFTERS and its config, tensors and quantization agree with the declaration and the target)."""
    return WithDrafter(hf, Drafter(kind, path, hf.decl))


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


def shape(hf, ctx):
    src = ctx.sources.get("exl3")
    if src is not None and not isinstance(src, PackView):
        ctx.sources["exl3"] = PackView(src)
    if ctx.sources.get("exl3_experts") is not None:
        raise SystemExit("--exl3-experts: the Nemotron-H build reads one pack (--exl3)")
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    mtp = _mtp_written(hf, ctx.recipe)
    drafter = hf.drafter.kind if isinstance(hf, WithDrafter) else ""
    if drafter and mtp:
        raise SystemExit(f"--drafter {drafter}: {ctx.recipe.path} writes the MTP module too; a container carries one "
                         f"drafter -- omit the MTP rows (format-maps/nemotron_h-k4-lilicorr.json)")
    return replace(hf.decl, recipe=ctx.recipe.name, mtp=mtp, drafter=drafter, table=hf.mapped)


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
            ("pulsar.vision_present", "bool", False)]
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

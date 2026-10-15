"""inkling.py -- the Inkling family in the one artifact builder (L294 / L295 E1): Thinking Machines' Inkling
(config.json architectures ["InklingForConditionalGeneration"], model_type "inkling_mm_model") -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 HF checkpoint: the authority for the config, every BF16 tensor and every non-EXL3 entry.  Read
           through this module's Source view (below).
  --exl3   an exllamav3 EXL3 pack from our fork's architecture module (pulsar/exllamav3 branch `inkling`,
           exllamav3/architecture/inkling.py).  Its Linear keys are exactly the view's names minus `.weight`
           (L294's tensor-name contract): attention keeps the checkpoint names (wq_du, wk_dv, wv_dv, wr_du, wo_ud);
           the fused gate/up tensors are split into `...mlp.{gate,up}_proj` (dense), `...mlp.experts.<e>.
           {gate,up,down}_proj` (routed), `...mlp.shared_experts.<j>.{gate,up,down}_proj` (shared); the dense down
           keeps `...mlp.w2_md`; the head is `model.llm.unembed`, which no recipe can read from the pack yet: the
           role `head` runs no EXL3 arm (exl3_rates.ROLE_ARM), so the head is written from --hf.
  --recipe a pulsar.recipe.v1 (format-maps/inkling-*.json): required, as for qwen4_exp.

The Source view.  Inkling's checkpoint fuses what pulsar binds separately: `w13` tensors hold gate and up
ROW-INTERLEAVED (gate on the even rows, up on the odd: exllamav3 Linear.frange_slice with finterleaved, the BF16
parity of L294), and the routed / shared experts are [E, out, in] stacks.  The view names each projection on its own
-- the same names the EXL3 pack uses -- so a BF16 build and an EXL3 build write one container layout and the
loader never de-interleaves (rule 3).  A derived tensor's own bytes are a slice of a stack (a copy range) or the
rows_strided producer over one (verbatim rows, no arithmetic).  Every other HF tensor passes through by name.

What the module checks (rule 1 -- refuse, never guess):
  * the config: every key the declaration reads is present; every architecture switch the role table assumes is the
    value it assumes (use_sconv, sigmoid gate + selection bias, sink shared experts, global scale, no q/o bias, embed
    norm, no logit softcap, hmlp vision); moe_intermediate_size, when the config names it, equals intermediate_size
    (Transformers 5.19 mis-parses the published config, L294);
  * the names, both ways: every HF tensor is a row of the table below (or under `model.audio.`, consumed by name:
    audio is out of scope, plan 120 decisions), and every row the declaration expects is present (the drafter's
    rows when config.json has an mtp_config, the tower's when the checkpoint holds any model.visual.* tensor);
  * every tensor's dtype and shape against the declaration (attention geometry per layer kind, extents, conv width,
    expert widths, the router's sink rows, the MTP input projection).

Roles.  Each tensor carries two words: the RECIPE role (the existing vocabulary recipe rows select by: dense |
shared_expert | expert_gate / _up / _down | router | norm | embed | head | vision | other) and the BIND role, its
identity for the loader, written into its pulsar.tensors / pulsar.experts record together with its block and layer
(names.Mapped.bind).  The bind roles are family-neutral words for the blocks L295 builds (F1 GQA attention with a
relative-bias score mod, F2 router with sinks, F3 state kinds); nothing here is named after Inkling.

The container:
  * names ARE the view's names (= the HF names, or the EXL3 Linear key + `.weight` for the derived ones); routed
    experts fold to one U8 entry per expert-projection `model.llm.layers.L.mlp.experts.<e>.<part>.weight`,
    contiguous in expert order, part in {gate_proj, up_proj, down_proj}; each family in pulsar.experts carries
    entry_name, layer and role.  The gguf_name of a family is `model.llm.layers.L.mlp.experts.<part>`.
  * layouts: bf16 / f32 native; exl3m_k* for the dense / shared / routed Linears from the EXL3 pack (the rate per
    engine arm, exl3_rates.admit); mxfp8_lt for a dense Linear from BF16.  The short-conv weights are declared
    [channels, kernel] (the HF [channels, 1, kernel] minus its unit dim; the same bytes).
  * shards: `vision` (primary: pulsar.kv + the hmlp tower), `layers.0..L-1`, `top` (embed, embed norm, final norm,
    head), `mtp.0..N-1` when the recipe writes the drafter (all of it or none).
  * pulsar.kv: general.architecture = "inkling"; text_config verbatim under `inkling.`, vision_config under
    `inkling.vision.`, mtp_config under `inkling.mtp.` (kv.config_kvs); eos_token_id; the builder's own facts under
    `pulsar.*`.  No tokenizer.* entries yet: Inkling's tokenizer is tiktoken / a BPE whose specials are not named
    in tokenizer_config.json, and the frontend kit (L295 F5, plan 120 E6) defines what the engine reads.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

import entries as EN
import exl3_rates
import kv as KV
import producers as PR
import recipe as R
from names import Mapped

FAMILY = "inkling"
MODEL_TYPE = "inkling_mm_model"
ARCHITECTURE = "InklingForConditionalGeneration"
META = {"pulsar.family": FAMILY}
SETTINGS = ()
EXPERT_STACKS = False           # the view names one tensor per expert-projection; grouped after the walk
GATE_UP_PARTS = ("gate_proj", "up_proj")
LLM, MTP, VIS, AUDIO = "model.llm.", "model.mtp.", "model.visual.", "model.audio."
EXL3 = set(exl3_rates.K2)

# ---------------------------------------------------------------------------
# the role table: suffix -> (bind role, recipe role).  Block suffixes follow `model.llm.layers.L.` and the drafter's
# `model.mtp.layers.N.transformer_block.`; the drafter's own follow `model.mtp.layers.N.`.
# ---------------------------------------------------------------------------
ATTN = {
    "attn_norm.weight": ("attn_norm", "norm"),
    "attn.wq_du.weight": ("attn_q", "dense"),
    "attn.wk_dv.weight": ("attn_k", "dense"),
    "attn.wv_dv.weight": ("attn_v", "dense"),
    "attn.wr_du.weight": ("attn_rel_proj", "dense"),          # hidden -> r (d_rel per head): the relative-bias query
    "attn.wo_ud.weight": ("attn_o", "dense"),
    "attn.q_norm.weight": ("attn_q_norm", "norm"),
    "attn.k_norm.weight": ("attn_k_norm", "norm"),
    "attn.rel_logits_proj.proj": ("attn_rel_basis", "other"),  # P [d_rel, extent]: bias(q, h, dist) = r . P[:, dist]
    "attn.k_sconv.weight": ("short_conv_k", "other"),
    "attn.v_sconv.weight": ("short_conv_v", "other"),
    "attn_sconv.weight": ("short_conv_attn_out", "other"),
    "mlp_norm.weight": ("ffn_norm", "norm"),
    "mlp_sconv.weight": ("short_conv_ffn_out", "other"),
}
DENSE_MLP = {
    "mlp.gate_proj.weight": ("ffn_gate", "dense"),             # derived: the even rows of mlp.w13_dn
    "mlp.up_proj.weight": ("ffn_up", "dense"),                 # derived: the odd rows
    "mlp.w2_md.weight": ("ffn_down", "dense"),
    "mlp.global_scale": ("ffn_out_scale", "other"),            # BF16 [1]: the MLP output's multiplier
}
MOE = {
    "mlp.gate.weight": ("router", "router"),                   # [E + n_shared, H]: routed logits, then the sink rows
    "mlp.gate.bias": ("router_select_bias", "router"),         # F32 [E]: selection only
    "mlp.gate.global_scale": ("router_out_scale", "router"),   # F32 [1]: x route_scale
}
PART_ROLE = {"gate_proj": ("expert_gate", "expert_gate"), "up_proj": ("expert_up", "expert_up"),
             "down_proj": ("expert_down", "expert_down")}
SHARED_ROLE = {"gate_proj": "shared_gate", "up_proj": "shared_up", "down_proj": "shared_down"}
DRAFT = {
    "embed_norm.weight": ("draft_embed_norm", "norm"),
    "hidden_norm.weight": ("draft_hidden_norm", "norm"),
    "input_proj.weight": ("draft_in_proj", "dense"),           # [H, 2H]: [hidden_norm(h) | embed_norm(e)] -> H
}
TOP = {
    LLM + "embed.weight": ("embed", "embed"),
    LLM + "embed_norm.weight": ("embed_norm", "norm"),
    LLM + "norm.weight": ("output_norm", "norm"),
    LLM + "unembed.weight": ("head", "head"),
}
# the fused HF tensors the view splits (Source._split): the dense w13 into gate / up rows; a routed or shared stack's
# w13 into per-index gate / up rows, its w2 into per-index slices
FUSED_DENSE = "mlp.w13_dn.weight"
FUSED_ROUTED = {"mlp.experts.w13_weight": "w13", "mlp.experts.w2_weight": "w2"}
FUSED_SHARED = {"mlp.shared_experts.shared_w13_weight": "w13", "mlp.shared_experts.shared_w2_weight": "w2"}

_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_SHARED = re.compile(r"^mlp\.shared_experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


# ---------------------------------------------------------------------------
# the declaration: config.json -> the shape, refusing what the role table does not describe
# ---------------------------------------------------------------------------
def _need(d, key, what):
    if key not in d:
        raise SystemExit(f"config.json {what} has no {key!r} -- the Inkling declaration reads it")
    return d[key]


def _assume(d, key, want, what):
    got = _need(d, key, what)
    if got != want:
        raise SystemExit(f"config.json {what}.{key} = {got!r}; the Inkling role table assumes {want!r} -- refusing")


@dataclass(frozen=True)
class Shape:
    """The declared shape (the engine-side shape struct refuses a container that disagrees): every dimension the
    role table checks a tensor against, the layer kinds, and what the recipe writes."""
    n_layer: int
    hidden: int
    vocab: int
    vocab_real: int
    attn_full: tuple            # (n_head, n_kv, head_dim, rel_extent) of a global layer
    attn_window: tuple          # (n_head, n_kv, head_dim, window): a local layer's relative extent is its window
    d_rel: int
    conv_k: int
    n_dense: int                # layers 0 .. n_dense-1 run the dense MLP
    dense_ff: int
    moe_ff: int
    n_expert: int
    n_shared: int
    top_k: int
    window_layers: frozenset
    n_mtp: int
    mtp_window_layers: frozenset
    vision_layers: int          # the hmlp tower's linears (0: none in the checkpoint)
    recipe: str = ""
    mtp: bool = False           # the recipe writes the drafter
    vision: bool = False        # the recipe writes the tower
    full_layers: tuple = field(default=(), compare=False)
    table: dict = field(default=None, compare=False, repr=False)   # the view's {name: Mapped} (map_hf answers here)

    def geometry(self, window: bool):
        return self.attn_window if window else self.attn_full

    def __str__(self):
        return (f"inkling: {self.n_layer} layers (full {list(self.full_layers)}), dense {self.n_dense}, "
                f"{self.n_expert} experts top-{self.top_k} + {self.n_shared} shared, mtp {self.n_mtp}"
                f"{' written' if self.mtp else ' not written'}, vision {'written' if self.vision else 'not written'}, "
                f"recipe {self.recipe}")


def declare(top: dict) -> Shape:
    if top.get("model_type") != MODEL_TYPE or top.get("architectures") != [ARCHITECTURE]:
        raise SystemExit(f"config.json: model_type {top.get('model_type')!r} / architectures "
                         f"{top.get('architectures')!r}; inkling.py reads {MODEL_TYPE} / [{ARCHITECTURE}]")
    t = _need(top, "text_config", "the top level")
    w = "text_config"
    for k, v in (("use_sconv", True), ("q_bias", False), ("o_bias", False), ("use_embed_norm", True),
                 ("gate_activation", "sigmoid"), ("use_gate_bias", True), ("shared_expert_sink", True),
                 ("norm_after_topk", True), ("use_global_scale", True), ("final_logit_softcapping", None)):
        _assume(t, k, v, w)
    n_layer = int(_need(t, "num_hidden_layers", w))
    moe_ff = int(_need(t, "intermediate_size", w))
    if "moe_intermediate_size" in t and int(t["moe_intermediate_size"]) != moe_ff:
        raise SystemExit(f"config.json text_config: moe_intermediate_size {t['moe_intermediate_size']} != "
                         f"intermediate_size {moe_ff} (the experts are intermediate_size wide; L294)")
    local = [int(x) for x in _need(t, "local_layer_ids", w)]
    if sorted(set(local)) != local or (local and not 0 <= local[0] <= local[-1] < n_layer):
        raise SystemExit(f"config.json text_config.local_layer_ids {local}: ascending, distinct, in [0, {n_layer})")
    n_dense = int(_need(t, "dense_mlp_idx", w))
    if not 0 <= n_dense <= n_layer:
        raise SystemExit(f"config.json text_config.dense_mlp_idx {n_dense} outside [0, {n_layer}]")
    mtp = top.get("mtp_config")
    n_mtp, mtp_local = 0, []
    if mtp is not None:
        n_mtp = int(_need(mtp, "num_nextn_predict_layers", "mtp_config"))
        _assume(mtp, "chain_hidden_post_norm", False, "mtp_config")
        mtp_local = [int(x) for x in _need(mtp, "local_layer_ids", "mtp_config")]
        if sorted(set(mtp_local)) != mtp_local or (mtp_local and not 0 <= mtp_local[0] <= mtp_local[-1] < n_mtp):
            raise SystemExit(f"config.json mtp_config.local_layer_ids {mtp_local}: ascending, distinct, in [0, {n_mtp})")
    vis = top.get("vision_config")
    n_vis = 0
    if vis is not None:
        _assume(vis, "vision_encoder_type", "hmlp", "vision_config")
        n_vis = int(_need(vis, "n_layers", "vision_config"))
        if int(_need(vis, "decoder_dmodel", "vision_config")) != int(_need(t, "hidden_size", w)):
            raise SystemExit("config.json vision_config.decoder_dmodel != text_config.hidden_size")
    sh = Shape(
        n_layer=n_layer, hidden=int(_need(t, "hidden_size", w)), vocab=int(_need(t, "vocab_size", w)),
        vocab_real=int(_need(t, "unpadded_vocab_size", w)),
        attn_full=(int(_need(t, "num_attention_heads", w)), int(_need(t, "num_key_value_heads", w)),
                   int(_need(t, "head_dim", w)), int(_need(t, "rel_extent", w))),
        attn_window=(int(_need(t, "swa_num_attention_heads", w)), int(_need(t, "swa_num_key_value_heads", w)),
                     int(_need(t, "swa_head_dim", w)), int(_need(t, "sliding_window_size", w))),
        d_rel=int(_need(t, "d_rel", w)), conv_k=int(_need(t, "sconv_kernel_size", w)),
        n_dense=n_dense, dense_ff=int(_need(t, "dense_intermediate_size", w)), moe_ff=moe_ff,
        n_expert=int(_need(t, "n_routed_experts", w)), n_shared=int(_need(t, "n_shared_experts", w)),
        top_k=int(_need(t, "num_experts_per_tok", w)), window_layers=frozenset(local), n_mtp=n_mtp,
        mtp_window_layers=frozenset(mtp_local), vision_layers=n_vis,
        full_layers=tuple(i for i in range(n_layer) if i not in local))
    if sh.vocab_real > sh.vocab or not 0 < sh.top_k <= sh.n_expert:
        raise SystemExit(f"config.json text_config: unpadded_vocab_size {sh.vocab_real} / vocab_size {sh.vocab}, "
                         f"top-{sh.top_k} of {sh.n_expert}")
    return sh


def _block_table(sh: Shape, window: bool, dense: bool, moe: bool):
    """suffix -> (dtype, shape, bind role, recipe role) of one transformer block, from the declaration."""
    H = sh.hidden
    nq, nkv, hd, extent = sh.geometry(window)
    t = {}

    def put(table, suffix, dtype, shape):
        bind, role = table[suffix]
        t[suffix] = (dtype, list(shape), bind, role)
    for s, shp in (("attn_norm.weight", [H]), ("mlp_norm.weight", [H]), ("attn.wq_du.weight", [nq * hd, H]),
                   ("attn.wk_dv.weight", [nkv * hd, H]), ("attn.wv_dv.weight", [nkv * hd, H]),
                   ("attn.wr_du.weight", [nq * sh.d_rel, H]), ("attn.wo_ud.weight", [H, nq * hd]),
                   ("attn.q_norm.weight", [hd]), ("attn.k_norm.weight", [hd]),
                   ("attn.rel_logits_proj.proj", [sh.d_rel, extent]),
                   ("attn.k_sconv.weight", [nkv * hd, 1, sh.conv_k]), ("attn.v_sconv.weight", [nkv * hd, 1, sh.conv_k]),
                   ("attn_sconv.weight", [H, 1, sh.conv_k]), ("mlp_sconv.weight", [H, 1, sh.conv_k])):
        put(ATTN, s, "BF16", shp)
    if dense:
        put(DENSE_MLP, "mlp.gate_proj.weight", "BF16", [sh.dense_ff, H])
        put(DENSE_MLP, "mlp.up_proj.weight", "BF16", [sh.dense_ff, H])
        put(DENSE_MLP, "mlp.w2_md.weight", "BF16", [H, sh.dense_ff])
        put(DENSE_MLP, "mlp.global_scale", "BF16", [1])
    if moe:
        put(MOE, "mlp.gate.weight", "BF16", [sh.n_expert + sh.n_shared, H])
        put(MOE, "mlp.gate.bias", "F32", [sh.n_expert])
        put(MOE, "mlp.gate.global_scale", "F32", [1])
        for part, shp in (("gate_proj", [sh.moe_ff, H]), ("up_proj", [sh.moe_ff, H]), ("down_proj", [H, sh.moe_ff])):
            for e in range(sh.n_expert):
                bind, role = PART_ROLE[part]
                t[f"mlp.experts.{e}.{part}.weight"] = ("BF16", shp, bind, role)
            for j in range(sh.n_shared):
                t[f"mlp.shared_experts.{j}.{part}.weight"] = ("BF16", shp, SHARED_ROLE[part], "shared_expert")
    return t


def _fused_table(sh: Shape, dense: bool, moe: bool):
    """The fused HF tensors of one block: suffix -> (dtype, shape)."""
    H = sh.hidden
    t = {}
    if dense:
        t[FUSED_DENSE] = ("BF16", [2 * sh.dense_ff, H])
    if moe:
        t["mlp.experts.w13_weight"] = ("BF16", [sh.n_expert, 2 * sh.moe_ff, H])
        t["mlp.experts.w2_weight"] = ("BF16", [sh.n_expert, H, sh.moe_ff])
        t["mlp.shared_experts.shared_w13_weight"] = ("BF16", [sh.n_shared, 2 * sh.moe_ff, H])
        t["mlp.shared_experts.shared_w2_weight"] = ("BF16", [sh.n_shared, H, sh.moe_ff])
    return t


def expected(sh: Shape, vision: bool):
    """{view name: (dtype, shape, Mapped)} -- every tensor the declaration says the checkpoint holds, as the view
    names it (the fused HF tensors are listed by fused_expected)."""
    out = {}

    def add(name, dtype, shape, bind, role, shard, layer=None, block=None, expert=None, part=None, index=None):
        rec = (("role", bind),) + ((("block", block), ("layer", layer)) if block else ()) + \
              ((("index", index),) if index is not None else ())
        fam = "expert" if expert is not None else ("mtp" if block == "mtp" else "layer" if block else
                                                    "vision" if shard == "vision" else "top")
        out[name] = (dtype, shape, Mapped(name, name, fam, shard, layer, expert, part, False, role=role, bind=rec))
    for L in range(sh.n_layer):
        dense = L < sh.n_dense
        for s, (dt, shp, bind, role) in _block_table(sh, L in sh.window_layers, dense, not dense).items():
            m = _EXPERT.match(s)
            j = _SHARED.match(s)
            add(f"{LLM}layers.{L}.{s}", dt, shp, bind, role, f"layers.{L}", L, "layers",
                expert=int(m.group(1)) if m else None, part=m.group(2) if m else None,
                index=int(j.group(1)) if j else None)
    for n in range(sh.n_mtp):
        for s, (dt, shp, bind, role) in _block_table(sh, n in sh.mtp_window_layers, True, False).items():
            add(f"{MTP}layers.{n}.transformer_block.{s}", dt, shp, bind, role, f"mtp.{n}", n, "mtp")
        for s, shp in (("embed_norm.weight", [sh.hidden]), ("hidden_norm.weight", [sh.hidden]),
                       ("input_proj.weight", [sh.hidden, 2 * sh.hidden])):
            add(f"{MTP}layers.{n}.{s}", "BF16", shp, *DRAFT[s], f"mtp.{n}", n, "mtp")
    for name, (bind, role) in TOP.items():
        add(name, "BF16", [sh.vocab, sh.hidden] if role in ("embed", "head") else [sh.hidden], bind, role, "top")
    if vision:
        V = sh.vision_layers
        for i in range(V):
            add(f"{VIS}layers.linear_{i}.weight", "BF16", None, "vision_linear", "vision", "vision", index=i)
        for i in range(V - 1):
            add(f"{VIS}layers.norm_{i}.weight", "BF16", None, "vision_norm", "vision", "vision", index=i)
        add(f"{VIS}final_norm.weight", "BF16", [sh.hidden], "vision_final_norm", "vision", "vision")
    return out


def fused_expected(sh: Shape):
    """{HF name: (dtype, shape, prefix, suffix)} of the fused tensors the view splits."""
    out = {}
    for L in range(sh.n_layer):
        dense = L < sh.n_dense
        for s, v in _fused_table(sh, dense, not dense).items():
            out[f"{LLM}layers.{L}.{s}"] = (*v, f"{LLM}layers.{L}.", s)
    for n in range(sh.n_mtp):
        for s, v in _fused_table(sh, True, False).items():
            out[f"{MTP}layers.{n}.transformer_block.{s}"] = (*v, f"{MTP}layers.{n}.transformer_block.", s)
    return out


# ---------------------------------------------------------------------------
# the Source view
# ---------------------------------------------------------------------------
class Source:
    """The HF checkpoint as the Inkling plan reads it: the passthrough tensors by their HF names, the fused ones as
    the projections they hold (named as the EXL3 pack names them), the audio tower by name (mapped, never
    written).  Construction checks every name, dtype and shape against the declaration, both ways."""

    def __init__(self, hf):
        self.hf, self.dir = hf, hf.dir
        self.config, self.generation_config = hf.config, hf.generation_config
        decl = declare(hf.config["top_level"])
        names = hf.names()
        has_vision = any(n.startswith(VIS) for n in names)
        if has_vision and not decl.vision_layers:
            raise SystemExit(f"{VIS}*: tensors present, but config.json has no vision_config")
        self.decl = decl
        want = expected(decl, has_vision)
        fused = fused_expected(decl)
        self.mapped, self._derived = {}, {}
        unknown, wrong = [], []
        for n in names:
            if n.startswith(AUDIO):          # out of scope (plan 120): consumed by name, written nowhere
                self.mapped[n] = Mapped(n, n, "audio", "audio", None, None, None, False, emit=False, role="audio")
                continue
            if n in fused:
                dt, shp, pre, suf = fused[n]
                if hf.dtype(n) != dt or hf.shape(n) != shp:
                    wrong.append(f"{n}: {hf.dtype(n)} {hf.shape(n)}, the declaration says {dt} {shp}")
                    continue
                self._split(n, pre, suf, shp)
                continue
            if n not in want:
                unknown.append(n)
                continue
            dt, shp, _m = want[n]
            got = hf.shape(n)
            if n.startswith(VIS) and shp is None:   # the hmlp tower's inner widths are its own
                shp = got
            if hf.dtype(n) != dt or got != shp:
                wrong.append(f"{n}: {hf.dtype(n)} {got}, the declaration says {dt} {shp}")
        if unknown:
            raise SystemExit(f"{len(unknown)} tensors the Inkling naming table does not map -- refusing: "
                             f"{unknown[:5]}")
        if wrong:
            raise SystemExit(f"{len(wrong)} tensors disagree with the declared shape -- refusing: {wrong[:3]}")
        both = sorted(set(names) & set(self._derived))
        if both:
            raise SystemExit(f"{both[0]}: in the checkpoint both on its own and inside its fused tensor -- refusing")
        present = set(names) | set(self._derived)
        missing = sorted(n for n in fused if n not in present) + sorted(n for n in want if n not in present)
        if missing:
            raise SystemExit(f"{len(missing)} tensors the declaration expects are missing from {hf.dir} -- "
                             f"refusing: {missing[:5]}")
        for n, (_dt, _shp, m) in want.items():
            self.mapped[n] = m
        self._names = sorted(self.mapped)

    def _split(self, name, pre, suf, shp):
        """The projections one fused HF tensor holds, each with its copy source."""
        hf = self.hf
        if suf == FUSED_DENSE:
            rows, cols = shp
            for part, start in (("gate_proj", 0), ("up_proj", 1)):
                self._derived[f"{pre}mlp.{part}.weight"] = (
                    [rows // 2, cols], ("produce", PR.spec("rows_strided", [name], n_rows=rows, row_bytes=2 * cols,
                                                           start=start, step=2)), rows // 2 * cols * 2)
            return
        stack = "experts" if suf in FUSED_ROUTED else "shared_experts"
        kind = (FUSED_ROUTED if suf in FUSED_ROUTED else FUSED_SHARED)[suf]
        n, rows, cols = shp
        for i in range(n):
            sl = f"{name}#{i}"
            if kind == "w2":
                path, off, nb = hf.span(sl)
                self._derived[f"{pre}mlp.{stack}.{i}.down_proj.weight"] = ([rows, cols], ("ranges", [(path, off, nb)]), nb)
                continue
            for part, start in (("gate_proj", 0), ("up_proj", 1)):
                self._derived[f"{pre}mlp.{stack}.{i}.{part}.weight"] = (
                    [rows // 2, cols], ("produce", PR.spec("rows_strided", [sl], n_rows=rows, row_bytes=2 * cols,
                                                           start=start, step=2)), rows // 2 * cols * 2)

    # -- the HFCheckpoint interface ---------------------------------------
    def names(self):
        return self._names

    def has(self, name):
        return name in self.mapped

    def shape(self, name):
        d = self._derived.get(name)
        return list(d[0]) if d else self.hf.shape(name)

    def dtype(self, name):
        return "BF16" if name in self._derived else self.hf.dtype(name)

    def span(self, name):
        d = self._derived.get(name)
        if d is None:
            return self.hf.span(name)
        if d[1][0] != "ranges":
            raise SystemExit(f"{name}: rows of a row-interleaved fused tensor -- no single copy range")
        return d[1][1][0]

    def copy_src(self, name):
        d = self._derived.get(name)
        return (d[1], d[2]) if d else self.hf.copy_src(name)

    def raw(self, name):
        """A tensor's own bytes: the HF tensor's (or an `X#i` slice), or a derived projection's."""
        d = self._derived.get(name)
        if d is None:
            return self.hf.raw(name)
        kind, src = d[1]
        if kind == "produce":
            return PR.produce(src, self.hf)
        path, off, n = src[0]
        with open(path, "rb") as f:
            f.seek(off)
            buf = f.read(n)
        if len(buf) != n:
            raise SystemExit(f"{name}: short read from {path}")
        return buf


def source(hf):
    return Source(hf)


# ---------------------------------------------------------------------------
# the family hooks (build.plan)
# ---------------------------------------------------------------------------
def map_hf(name: str, shape: Shape) -> Mapped | None:
    """The naming table's row for a name of the view (the table the Source built and checked), or None."""
    return shape.table.get(name)


def _resolve_written(hf, recipe):
    """Whether the recipe writes the drafter and the tower: all of each or none (a half-written block refuses)."""
    res = recipe.resolve(hf.mapped)
    out = {}
    for what, test in (("mtp", lambda m: m.shard.startswith("mtp.")), ("vision", lambda m: m.shard == "vision")):
        fm = {n: f for n, (f, _s) in res.items() if test(hf.mapped[n])}
        omitted = {n for n, f in fm.items() if f in R.CONSUMED}
        if omitted and len(omitted) != len(fm):
            raise SystemExit(f"{recipe.path}: writes part of the {what} block and omits {sorted(omitted)[0]} -- "
                             f"all of it or none")
        out[what] = bool(fm) and not omitted
    return out


def shape(hf, ctx):
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    w = _resolve_written(hf, ctx.recipe)
    return replace(hf.decl, recipe=ctx.recipe.name, mtp=w["mtp"], vision=w["vision"], table=hf.mapped)


def shard_order(shape):
    return (["vision"] + [f"layers.{i}" for i in range(shape.n_layer)] + ["top"]
            + ([f"mtp.{i}" for i in range(shape.n_mtp)] if shape.mtp else []))


def declared_shape(m, hshape):
    """The short-conv weights are declared [channels, kernel]: the HF [channels, 1, kernel] minus its unit dim (the
    same bytes, the kernel's view).  Everything else as the source."""
    if dict(m.bind).get("role", "").startswith("short_conv_"):
        c, one, k = hshape
        if one != 1:
            raise SystemExit(f"{m.container_name}: short conv {hshape}, expected [channels, 1, kernel]")
        return [c, k]
    return list(hshape)


# what the Inkling loader binds, by recipe role (EXL3 rates further by the engine arm, exl3_rates.admit): the routed
# / shared / dense Linears EXL3 or BF16 (dense Linears also MXFP8); everything else in its source dtype.  `omit` only
# for the optional blocks (the drafter, the tower; _resolve_written keeps each whole).
LINEAR_ROLES = {"dense", "shared_expert", "expert_gate", "expert_up", "expert_down"}


def admit(hf, m, fmt, name):
    native = PR.NATIVE_LAYOUT.get(hf.dtype(name))
    allowed = {"native", native}
    if m.role in LINEAR_ROLES:
        allowed |= EXL3
        if m.role in ("dense", "shared_expert"):
            allowed.add("mxfp8_lt")
    if m.shard.startswith("mtp.") or m.shard == "vision":
        allowed.add("omit")
    if fmt not in allowed:
        raise SystemExit(f"{name}: the Inkling loader binds this {m.role} tensor as {sorted(a for a in allowed if a)}, "
                         f"not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit(f"{MODEL_TYPE}: pass --recipe (tools/container/format-maps/inkling-*.json)")


def group_key(m):
    return (m.shard, m.layer, m.part)


def gate_up_key(rec):
    """The layer a routed family belongs to, for the gate / up layout check: the family name's prefix."""
    return rec["gguf_name"].rsplit(".", 1)[0]


def group_families(hf, key, slots, fmt, ctx):
    """One routed family from the view's per-expert tensors {e: (name, mapped)}: EXL3 ranges from the pack, or
    the BF16 bytes from the checkpoint (a slice of the stack, or its interleaved rows)."""
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
                 srcs=srcs, extras={"entry_name": f"{pre}{{e}}.{part}.weight", "layer": layer, "role": rec["role"],
                                    "block": rec["block"]})]


# ---------------------------------------------------------------------------
# pulsar.kv
# ---------------------------------------------------------------------------
def build_kv(hf, ctx):
    """general.*; the text / vision / mtp configs verbatim (kv.config_kvs); eos; the builder's facts."""
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    if os.path.normpath(ctx.tokenizer_dir) != os.path.normpath(hf.dir):
        raise SystemExit("--tokenizer: the Inkling container carries no tokenizer.* entries yet (the frontend kit, "
                         "L295 F5 / plan 120 E6, defines them)")
    top = hf.config["top_level"]
    sh = ctx.shape
    kvs = [("general.architecture", "string", FAMILY), ("general.type", "string", "model"),
           ("general.name", "string", "Inkling")]
    KV.config_kvs(FAMILY, top["text_config"], kvs)
    if sh.vision:
        KV.config_kvs(f"{FAMILY}.vision", top["vision_config"], kvs)
    if "mtp_config" in top:
        KV.config_kvs(f"{FAMILY}.mtp", top["mtp_config"], kvs)
    kvs += [(f"{FAMILY}.eos_token_id", "u32", int(_need(top, "eos_token_id", "the top level"))),
            ("pulsar.recipe", "string", ctx.recipe.name),
            ("pulsar.recipe.sha256", "string", ctx.recipe.sha256 or ""),
            ("pulsar.mtp_present", "bool", sh.mtp),
            ("pulsar.vision_present", "bool", sh.vision)]
    for tag in ("exl3", "exl3_experts"):
        src = ctx.sources.get(tag)
        if src is not None:
            kvs.append((f"pulsar.source.{tag}", "string", os.path.basename(os.path.normpath(src.dir))))
    kvs.append(("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))))
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out

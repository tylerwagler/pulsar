"""gemma4.py -- Google Gemma 4 26B-A4B (config.json architectures ["Gemma4ForConditionalGeneration"], model_type
"gemma4") in the one artifact builder (L338): a 30-layer decoder, five blocks of 5 sliding-window + 1 global
attention layers, every layer a dense GeGLU MLP AND 128 top-8 routed GeGLU experts side by side -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 checkpoint: the authority for the config and every tensor the recipe writes native (norms, the
           router, the per-layer scalar, the embedding, for lookups).  The vision tower (`model.vision_tower.`,
           `model.embed_vision.`) is consumed by name and written nowhere in v1 (text first).
  --exl3   the exllamav3 pack of the same checkpoint (exllamav3 >= 1.6.5 `architecture/gemma4.py`, mul1 codebook): every
           Linear byte-copied, under the checkpoint's own names -- dense `...layers.N.{self_attn,mlp}.<proj>`, routed
           `...layers.N.experts.<e>.{gate,up,down}_proj` (the pack splits HF's fused gate_up stack per expert), and
           the head: HF ties it to the embedding, exllamav3 quantizes one from it as `lm_head` (6 bpw by default), and
           the Turing build runs that one (Tyler 2026-11-01: pulsar's BF16 head GEMM needs sm_80+); Source names it
           `lm_head.weight`.
  --recipe a pulsar.recipe.v1 (format-maps/gemma4-*.json): required.

The stored widths (L338 pack census).  exllamav3 pads every Linear to the 128 Hadamard block on BOTH sides
(modules/linear.py pad_to = 128): the routed expert width 704 is stored 768 and the dense MLP's 2112 is stored 2176.
The padded OUTPUT columns of gate / up carry svh = 0, so they decode to exactly 0; gelu_tanh(0) * 0 = 0 feeds the
padded INPUT rows of down, whose own weights were zero before quantization.  The stored width is therefore the
model's width with exact zeros in the pad, and the container declares the stored widths: the engine runs a 768-wide
expert and a 2176-wide MLP, no pad anywhere (the pack census measured svh pad max |0.0| on gate / up of experts and
MLP, layer 0; L338).  Everything else is a 128 multiple as stored (hidden 2816, heads 16 x 256 | 16 x 512, kv 8 x 256
| 2 x 512).

What the module checks (rule 1 -- refuse, never guess):
  * the config: every key the declaration reads is present, and every switch the engine implements is the value it
    implements -- gelu_pytorch_tanh, K = V on the global layers, the MoE block on, no per-layer inputs, no shared-KV
    layers, no double-wide MLP, tied embeddings, no attention bias, final logit soft-capping, the two ropes (sliding:
    default at its theta over the full 256 head; global: proportional, partial factor 0.25, over 512), the sliding /
    full pattern from `layer_types`, causal text (`use_bidirectional_attention` "vision" = bidirectional for vision
    tokens only, which v1 does not carry);
  * the names, both ways: every HF tensor is a row of the declaration or under a vision prefix (consumed by name), and
    every row the declaration expects is present; every tensor's dtype and shape against the declaration.

Roles: the recipe role (dense | norm | router | other | embed | expert_gate_up | expert_down) and the BIND role, per
layer (block `layers`): attn_norm, attn_q / _k / _v / _o, attn_q_norm, attn_k_norm, attn_post_norm; ffn_pre_norm,
ffn_gate / _up / _down (the dense MLP), ffn_post_norm_1 (its output norm); moe_pre_norm, router, router_scale,
router_expert_scale, expert_gate / _up / _down, ffn_post_norm_2 (the experts' output norm); ffn_post_norm (of the
sum); layer_scalar.  Global layers have no attn_v (K = V).  At the top: embed, output_norm, head (the pack's
EXL3 head).

The container:
  * names ARE the HF names; routed experts fold to one U8 entry per expert-projection,
    `model.language_model.layers.N.experts.<e>.{gate,up,down}_proj.weight`, families in pulsar.experts carrying
    `entry_name`, `layer`, `role`, `block`.  Layouts: bf16 native; exl3m_k* for every Linear from the pack, the head included.
  * shards: `vision` (primary: pulsar.kv only -- no tower in v1), `layers.0..29`, `top`.
  * pulsar.kv: general.architecture = "gemma4"; text_config verbatim under `gemma4.` (kv.config_kvs); the builder's
    facts (recipe, sources, pulsar.vision_present false).
  * side files: tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json and config.json,
    as shipped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import adapters as AD
import entries as EN
import exl3_rates
import kv as KV
import producers as PR
from names import Mapped

FAMILY = "gemma4"
MODEL_TYPE = "gemma4"
ARCH = "Gemma4ForConditionalGeneration"
LM = "model.language_model."
META = {"pulsar.family": FAMILY}
SETTINGS = ()
EXPERT_STACKS = True            # an HF expert tensor is the whole [E, out, in] stack; split per expert from the pack
GATE_UP_PARTS = ("gate_proj", "up_proj")
VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.")
HEAD = "lm_head.weight"         # the pack's EXL3 head, under the name exllamav3 keys it (Source)
EXL3 = set(exl3_rates.K2)

# the switches the engine implements: text_config key -> the one value it implements
ASSUME = {"model_type": "gemma4_text", "hidden_activation": "gelu_pytorch_tanh", "attention_k_eq_v": True,
          "enable_moe_block": True, "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
          "use_double_wide_mlp": False, "tie_word_embeddings": True, "attention_bias": False,
          "use_bidirectional_attention": "vision"}
ROPE = {"sliding_attention": {"rope_type": "default"},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25}}
LAYER_TYPES = ("sliding_attention", "full_attention")


def _need(d, key, what):
    if key not in d:
        raise SystemExit(f"config.json {what} has no {key!r} -- the Gemma 4 declaration reads it")
    return d[key]


def _assume(d, key, want, what):
    got = _need(d, key, what)
    if got != want:
        raise SystemExit(f"config.json {what}.{key} = {got!r}; the Gemma 4 declaration implements {want!r} -- refusing")


@dataclass(frozen=True)
class Shape:
    """The declared shape (the engine-side shape table refuses a container that disagrees)."""
    hidden: int
    vocab: int
    n_layer: int
    full: tuple                 # per layer: True = full (global) attention, False = sliding window
    window: int
    n_head: int
    sliding_kv: tuple           # (n_kv, head_dim) of a sliding layer
    full_kv: tuple              # (n_kv, head_dim) of a global layer (K = V: no v_proj)
    ffn: int                    # the dense MLP width, as HF declares it
    n_expert: int
    top_k: int
    expert_ffn: int             # the routed expert width, as HF declares it
    eps: float
    softcap: float
    recipe: str = ""
    table: dict = field(default=None, compare=False, repr=False)   # {name: Mapped} (map_hf answers here)

    @property
    def ffn_stored(self):
        return exl3_rates.out_stored(self.ffn)

    @property
    def expert_ffn_stored(self):
        return exl3_rates.out_stored(self.expert_ffn)

    def __str__(self):
        nf = sum(self.full)
        return (f"gemma4: {self.n_layer} layers ({self.n_layer - nf} sliding w{self.window} {self.n_head}q/"
                f"{self.sliding_kv[0]}kv/{self.sliding_kv[1]}, {nf} global {self.n_head}q/{self.full_kv[0]}kv/"
                f"{self.full_kv[1]} K=V), {self.hidden} wide, dense GeGLU {self.ffn} (stored {self.ffn_stored}) + "
                f"{self.n_expert} experts top-{self.top_k} x {self.expert_ffn} (stored {self.expert_ffn_stored}), "
                f"softcap {self.softcap}, EXL3 head (the pack's), vision not written (v1), recipe {self.recipe}")


def declare(top: dict) -> Shape:
    arch = top.get("architectures")
    if top.get("model_type") != MODEL_TYPE or arch != [ARCH]:
        raise SystemExit(f"config.json: model_type {top.get('model_type')!r} / architectures {arch!r}; gemma4.py reads "
                         f"{MODEL_TYPE} / [{ARCH!r}]")
    c = _need(top, "text_config", "the top level")
    w = "text_config"
    for k, v in ASSUME.items():
        _assume(c, k, v, w)
    rp = _need(c, "rope_parameters", w)
    if sorted(rp) != sorted(LAYER_TYPES):
        raise SystemExit(f"config.json {w}.rope_parameters names {sorted(rp)}; the declaration has {list(LAYER_TYPES)}")
    for t, want in ROPE.items():
        for k, v in want.items():
            _assume(rp[t], k, v, f"{w}.rope_parameters.{t}")
        _need(rp[t], "rope_theta", f"{w}.rope_parameters.{t}")
    extra = {t: sorted(set(rp[t]) - set(ROPE[t]) - {"rope_theta"}) for t in LAYER_TYPES}
    if any(extra.values()):
        raise SystemExit(f"config.json {w}.rope_parameters carries {extra}: rope keys the declaration does not implement")
    types = _need(c, "layer_types", w)
    L = int(_need(c, "num_hidden_layers", w))
    if len(types) != L or set(types) - set(LAYER_TYPES):
        raise SystemExit(f"config.json {w}.layer_types: {len(types)} entries of {sorted(set(types))} for {L} layers")
    softcap = _need(c, "final_logit_softcapping", w)
    if not isinstance(softcap, (int, float)) or softcap <= 0:
        raise SystemExit(f"config.json {w}.final_logit_softcapping = {softcap!r}; the declaration implements a positive cap")
    sh = Shape(hidden=int(_need(c, "hidden_size", w)), vocab=int(_need(c, "vocab_size", w)), n_layer=L,
               full=tuple(t == "full_attention" for t in types), window=int(_need(c, "sliding_window", w)),
               n_head=int(_need(c, "num_attention_heads", w)),
               sliding_kv=(int(_need(c, "num_key_value_heads", w)), int(_need(c, "head_dim", w))),
               full_kv=(int(_need(c, "num_global_key_value_heads", w)), int(_need(c, "global_head_dim", w))),
               ffn=int(_need(c, "intermediate_size", w)), n_expert=int(_need(c, "num_experts", w)),
               top_k=int(_need(c, "top_k_experts", w)), expert_ffn=int(_need(c, "moe_intermediate_size", w)),
               eps=float(_need(c, "rms_norm_eps", w)), softcap=float(softcap))
    for nkv, hd in (sh.sliding_kv, sh.full_kv):
        if sh.n_head % nkv or hd % 128:
            raise SystemExit(f"config.json {w}: {sh.n_head} q heads over {nkv} kv heads of {hd}")
    if not 1 <= L <= 64 or not any(sh.full) or not 0 < sh.top_k <= sh.n_expert or sh.hidden % 128:
        raise SystemExit(f"config.json {w}: {L} layers ({sum(sh.full)} global), top-{sh.top_k} of {sh.n_expert} "
                         f"experts, hidden {sh.hidden}")
    return sh


# per layer: HF suffix -> (bind role, recipe role)
LAYER = {
    "input_layernorm.weight": ("attn_norm", "norm"),
    "self_attn.q_proj.weight": ("attn_q", "dense"),
    "self_attn.k_proj.weight": ("attn_k", "dense"),
    "self_attn.v_proj.weight": ("attn_v", "dense"),
    "self_attn.o_proj.weight": ("attn_o", "dense"),
    "self_attn.q_norm.weight": ("attn_q_norm", "norm"),
    "self_attn.k_norm.weight": ("attn_k_norm", "norm"),
    "post_attention_layernorm.weight": ("attn_post_norm", "norm"),
    "pre_feedforward_layernorm.weight": ("ffn_pre_norm", "norm"),
    "mlp.gate_proj.weight": ("ffn_gate", "dense"),
    "mlp.up_proj.weight": ("ffn_up", "dense"),
    "mlp.down_proj.weight": ("ffn_down", "dense"),
    "post_feedforward_layernorm_1.weight": ("ffn_post_norm_1", "norm"),
    "pre_feedforward_layernorm_2.weight": ("moe_pre_norm", "norm"),
    "router.proj.weight": ("router", "router"),
    "router.scale": ("router_scale", "other"),
    "router.per_expert_scale": ("router_expert_scale", "other"),
    "experts.gate_up_proj": ("expert_gate_up", "expert_gate_up"),
    "experts.down_proj": ("expert_down", "expert_down"),
    "post_feedforward_layernorm_2.weight": ("ffn_post_norm_2", "norm"),
    "post_feedforward_layernorm.weight": ("ffn_post_norm", "norm"),
    "layer_scalar": ("layer_scalar", "other"),
}
# the dense MLP Linears: declared at the pack's stored width (module docstring)
MLP_STORED = {"mlp.gate_proj.weight": "out", "mlp.up_proj.weight": "out", "mlp.down_proj.weight": "in"}
# the split parts of the routed stacks: HF stack suffix -> [(part, bind role)]
EXPERT_PARTS = {"experts.gate_up_proj": [("gate_proj", "expert_gate"), ("up_proj", "expert_up")],
                "experts.down_proj": [("down_proj", "expert_down")]}


def expected(sh: Shape):
    """{HF name: (dtype, shape, Mapped)} -- every text tensor the declaration expects."""
    H, F, E, Fe = sh.hidden, sh.ffn, sh.n_expert, sh.expert_ffn
    out = {}

    def add(name, shape, bind, role, shard, layer=None, part=None):
        rec = (("role", bind),) + ((("block", "layers"), ("layer", layer)) if layer is not None else ())
        fam = "layer" if layer is not None else "top"
        out[name] = ("BF16", shape, Mapped(name, name, fam, shard, layer, None, part, False, role=role, bind=rec))

    for il in range(sh.n_layer):
        nkv, hd = sh.full_kv if sh.full[il] else sh.sliding_kv
        dims = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
                "self_attn.q_proj.weight": [sh.n_head * hd, H], "self_attn.k_proj.weight": [nkv * hd, H],
                "self_attn.v_proj.weight": [nkv * hd, H], "self_attn.o_proj.weight": [H, sh.n_head * hd],
                "self_attn.q_norm.weight": [hd], "self_attn.k_norm.weight": [hd],
                "pre_feedforward_layernorm.weight": [H], "mlp.gate_proj.weight": [F, H], "mlp.up_proj.weight": [F, H],
                "mlp.down_proj.weight": [H, F], "post_feedforward_layernorm_1.weight": [H],
                "pre_feedforward_layernorm_2.weight": [H], "router.proj.weight": [E, H], "router.scale": [H],
                "router.per_expert_scale": [E], "experts.gate_up_proj": [E, 2 * Fe, H],
                "experts.down_proj": [E, H, Fe], "post_feedforward_layernorm_2.weight": [H],
                "post_feedforward_layernorm.weight": [H], "layer_scalar": [1]}
        for s, (bind, role) in LAYER.items():
            if s == "self_attn.v_proj.weight" and sh.full[il]:
                continue                                    # K = V: a global layer has no v_proj
            part = s.rsplit(".", 1)[1] if role.startswith("expert_") else None
            add(f"{LM}layers.{il}.{s}", dims[s], bind, role, f"layers.{il}", il, part)
    add(LM + "embed_tokens.weight", [sh.vocab, H], "embed", "embed", "top")
    add(LM + "norm.weight", [H], "output_norm", "norm", "top")
    add(HEAD, [sh.vocab, H], "head", "head", "top")
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
        raise SystemExit(f"{len(unknown)} tensors the Gemma 4 naming table does not map -- refusing: {unknown[:5]}")
    if wrong:
        raise SystemExit(f"{len(wrong)} tensors disagree with the declared shape -- refusing: {wrong[:3]}")
    missing = sorted(n for n in want if n not in mapped)
    if missing:
        raise SystemExit(f"{len(missing)} tensors the declaration expects are missing from {hf.dir} -- refusing: "
                         f"{missing[:5]}")
    return mapped


# ---------------------------------------------------------------------------
# the family hooks (build.plan)
# ---------------------------------------------------------------------------
class Source:
    """The checkpoint plus the head (L338, Tyler 2026-11-01: the Turing build runs the pack's EXL3 head).  HF ties the
    head to the embedding and stores no `lm_head.weight`; exllamav3 quantizes one from the embedding into the pack as
    `lm_head`.  This view names it -- HEAD, with the embedding's BF16 [vocab, hidden] shape and dtype (the [out, in] the
    plan sizes the EXL3 Linear by) -- so the head is an entry like any other; its bytes come from the pack only (admit
    takes the head EXL3, never native), so the embedding's bytes are never written twice."""

    def __init__(self, hf):
        self.hf = hf
        self.embed = LM + "embed_tokens.weight"

    def __getattr__(self, k):
        return getattr(self.hf, k)

    def _n(self, name):
        return self.embed if name == HEAD else name

    def names(self):
        return self.hf.names() + [HEAD]

    def has(self, name):
        return name == HEAD or self.hf.has(name)

    def shape(self, name):
        return self.hf.shape(self._n(name))

    def dtype(self, name):
        return self.hf.dtype(self._n(name))

    def raw(self, name):
        if name == HEAD:
            raise SystemExit(f"{HEAD}: the head's bytes are the pack's (EXL3); the BF16 view is not read")
        return self.hf.raw(name)

    def copy_src(self, name):
        if name == HEAD:
            raise SystemExit(f"{HEAD}: the head's bytes are the pack's (EXL3); the BF16 view is not copied")
        return self.hf.copy_src(name)


def source(hf):
    """The checkpoint plus the head entry (Source): every other tensor the plan writes is an HF tensor by its own name."""
    hf = Source(hf)
    hf.decl = declare(hf.config["top_level"])
    hf.mapped = table(hf, hf.decl)
    return hf


def with_adapter(hf, kind, path):
    """build.py --adapter KIND DIR: the checkpoint view carrying the adapter (adapters.py), checked against this
    declaration's per-layer projections at their LOGICAL widths (the dense MLP 2112: the engine pads to the stored)."""
    sh = hf.decl
    shapes = {}
    for il in range(sh.n_layer):
        nkv, hd = sh.full_kv if sh.full[il] else sh.sliding_kv
        q, kv = sh.n_head * hd, nkv * hd
        shapes[(il, "self_attn.q_proj")] = [q, sh.hidden]
        shapes[(il, "self_attn.k_proj")] = [kv, sh.hidden]
        shapes[(il, "self_attn.v_proj")] = None if sh.full[il] else [kv, sh.hidden]   # K = V: no v_proj
        shapes[(il, "self_attn.o_proj")] = [sh.hidden, q]
        shapes[(il, "mlp.gate_proj")] = [sh.ffn, sh.hidden]
        shapes[(il, "mlp.up_proj")] = [sh.ffn, sh.hidden]
        shapes[(il, "mlp.down_proj")] = [sh.hidden, sh.ffn]
    return AD.with_adapter(hf, kind, path, AD.Target(sh.hidden, sh.n_layer, shapes))


def map_hf(name: str, shape: Shape) -> Mapped | None:
    return shape.table.get(name)


def shape(hf, ctx):
    if ctx.sources.get("exl3_experts") is not None:
        raise SystemExit("--exl3-experts: the Gemma 4 build reads every Linear from the one --exl3 pack")
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    pack = ctx.sources.get("exl3")
    if pack is None:
        raise SystemExit("gemma4: pass --exl3 (the dense MLP and the experts are written from the pack only)")
    sh = hf.decl
    for il in range(sh.n_layer):            # the dense MLP's gate / up: the pad columns decode to zero
        for p in ("gate_proj", "up_proj"):
            require_zero_pad(pack, f"{LM}layers.{il}.mlp.{p}", sh.ffn, sh.ffn_stored)
    return replace(sh, recipe=ctx.recipe.name, table=hf.mapped)


def shard_order(shape):
    adapter = any(m.shard == "adapter" for m in shape.table.values())
    return ["vision"] + [f"layers.{i}" for i in range(shape.n_layer)] + ["top"] + (["adapter"] if adapter else [])


def _suffix(m):
    return m.container_name.split(f"layers.{m.layer}.", 1)[1]


def declared_shape(m, hshape):
    """The dense MLP Linears at the pack's stored width (module docstring); everything else as the source."""
    side = MLP_STORED.get(_suffix(m)) if m.layer is not None else None
    if side is None:
        return list(hshape)
    o, i = hshape
    return [exl3_rates.out_stored(o), i] if side == "out" else [o, exl3_rates.out_stored(i)]


# what the loader binds, by recipe role: every Linear EXL3 (the arm its role runs, exl3_rates.admit); the dense MLP
# EXL3 only (its declared width is the pack's stored width, which only the pack's bytes have); every other tensor in
# its source dtype
def admit(hf, m, fmt, name):
    native = PR.NATIVE_LAYOUT.get(hf.dtype(name))
    allowed = {"native", native}
    if m.role == "adapter" and fmt != "f16":        # the runtime LoRA's pairs: the fp16 tensor-core operand (L338)
        raise SystemExit(f"{name}: the adapter's LoRA pairs are written f16 (the runtime LoRA's operand), not {fmt}")
    if m.role == "adapter":
        return
    if m.role == "adapter_head" and fmt not in allowed:   # the decision head: its F32 as stored (read on the host)
        raise SystemExit(f"{name}: the decision head is written as stored ({native}), not {fmt}")
    if m.role == "dense":
        allowed |= EXL3
        if m.layer is not None and _suffix(m) in MLP_STORED:
            allowed = set(EXL3)
    if m.role in ("expert_gate_up", "expert_down", "head"):   # the head: the pack's (Source)
        allowed = set(EXL3)
    if fmt not in allowed:
        raise SystemExit(f"{name}: the Gemma 4 loader binds this {m.role} tensor as {sorted(a for a in allowed if a)}, "
                         f"not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit(f"{MODEL_TYPE}: pass --recipe (tools/container/format-maps/gemma4-*.json)")


def gate_up_key(rec):
    return rec["gguf_name"].rsplit(".", 1)[0]


def require_zero_pad(src, key, n, n_stored):
    """The exactness the stored widths rest on (module docstring), checked, not assumed: the gate / up Linear `key`'s
    svh is exactly zero on its padded output columns [n, n_stored), so those columns decode to 0.  Reads the pack's
    svh plane by its header entry (Exl3Checkpoint.entries)."""
    if n_stored == n:
        return
    path, off, nbytes, dt, shp = src.entries[f"{key}.svh"]
    if dt != "F16" or shp != [n_stored]:
        raise SystemExit(f"{key}.svh: {dt} {shp}, expected F16 [{n_stored}]")
    with open(path, "rb", buffering=0) as f:
        f.seek(off + 2 * n)
        pad = f.read(2 * (n_stored - n))
    if len(pad) != 2 * (n_stored - n) or any(b & 0x7f for b in pad[1::2]) or any(pad[0::2]):
        raise SystemExit(f"{key}.svh: nonzero on the padded columns [{n}, {n_stored}) -- the stored width would not "
                         f"equal the model's; refusing")


def stack_families(hf, m, fmt, ctx):
    """One HF expert stack ([E, out, in] BF16) -> its families, every expert-projection from the pack by its Linear key
    (`...layers.N.experts.<e>.<part>`), at the stored widths: gate / up [H -> Fe_stored], down [Fe_stored -> H]."""
    layout, src = fmt
    sh = hf.decl
    E, H, Fs = sh.n_expert, sh.hidden, sh.expert_ffn_stored
    pre = m.container_name[:-len(m.part)]                     # "...layers.N.experts."
    rec = dict(m.bind)
    fams = []
    for part, bind in EXPERT_PARTS[_suffix(m)]:
        inp, n = (H, Fs) if part != "down_proj" else (Fs, H)
        entry_name = pre + "{e}." + part + ".weight"
        names = [entry_name.replace("{e}", str(e)) for e in range(E)]
        if part != "down_proj":
            for x in names:
                require_zero_pad(ctx.sources[src], x[:-len(".weight")], sh.expert_ffn, Fs)
        srcs = [("ranges", EN.exl3_ranges(ctx.sources[src], x[:-len(".weight")], layout, inp, n)[0]) for x in names]
        fams.append(dict(gguf_name=pre + part, part=part, role=bind, layout=layout, inp=inp, n=n, entry_names=names,
                         srcs=srcs, extras={"entry_name": entry_name, "layer": m.layer, "role": bind,
                                            "block": rec["block"]}))
    return fams


# ---------------------------------------------------------------------------
# pulsar.kv
# ---------------------------------------------------------------------------
def build_kv(hf, ctx):
    """general.*; text_config verbatim (kv.config_kvs); the builder's facts."""
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    kvs = [("general.architecture", "string", FAMILY), ("general.type", "string", "model"),
           ("general.name", "string", os.path.basename(os.path.normpath(hf.dir)))]
    KV.config_kvs(FAMILY, dict(hf.config["top_level"]["text_config"]), kvs)
    kvs += [("pulsar.recipe", "string", ctx.recipe.name),
            ("pulsar.recipe.sha256", "string", ctx.recipe.sha256 or ""),
            ("pulsar.vision_present", "bool", False)]
    src = ctx.sources.get("exl3")
    if src is not None:
        kvs.append(("pulsar.source.exl3", "string", os.path.basename(os.path.normpath(src.dir))))
    kvs.append(("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))))
    if isinstance(hf, AD.WithAdapter):
        kvs += hf.adapter.kvs
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
    """[(container name, source path)]: the tokenizer, template and stop ids the engine loads beside the shards, from
    --tokenizer DIR (default: the HF dir), and config.json the checkpoint's own."""
    files = [(n, os.path.join(ctx.tokenizer_dir, n)) for n in SIDE] + [("config.json", os.path.join(hf.dir, "config.json"))]
    missing = [src for _, src in files if not os.path.isfile(src)]
    if missing:
        raise SystemExit(f"the Gemma 4 container carries the checkpoint's tokenizer: {', '.join(missing)} missing "
                         f"(--tokenizer DIR names a directory with {', '.join(SIDE)})")
    if isinstance(hf, AD.WithAdapter):
        files += hf.adapter.side
    return files

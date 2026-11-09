"""nemotron_enc.py -- the Nemotron-3.5 retrieval ENCODERS in the one artifact builder (L321): NVIDIA
Nemotron-3.5-Embed-8B (config.json architectures ["Mistral3Model"]) and Nemotron-3.5-Rerank-8B (["Mistral3ForSequence
Classification"]), both model_type "mistral3": a Pixtral tower + projector and a Ministral-3 text trunk run
BIDIRECTIONALLY (is_causal false), whose last hidden states are mean-pooled -- L2-normalised into a 4096-d embedding
(Embed), or scored by one Linear(4096 -> 1) (Rerank) -- -> pulsar's container.

Sources (no GGUF anywhere, L247):
  --hf     the BF16 checkpoint: the authority for the config and every tensor.  The trunk sits under `language_model.`
           (Embed) or `model.language_model.` (Rerank); the Reranker's score head is `language_model.score.weight` [1, E];
           the vision tower and projector (`vision_tower.`, `multi_modal_projector.`, with or without `model.`) are
           consumed by name and written nowhere in v1 (text first).
  --exl3   an exllamav3 pack of the trunk alone, staged as a Ministral3ForCausalLM (retriever/exl3/stage_text_trunk.py:
           keys `model.layers.N.*`, `model.embed_tokens`, `model.norm`) -- its Linears, byte copies (PackView maps the
           checkpoint's prefix to `model.`).  The pack's tied head (exllamav3 quantizes one) is never read.
  --recipe a pulsar.recipe.v1 (format-maps/nemotron_enc-*.json): required.

What the module checks (rule 1 -- refuse, never guess):
  * the config: every key the declaration reads is present; every switch the engine implements is the value it
    implements -- bidirectional trunk, silu, no sliding window, untied, mean pooling, and THE rope: YaRN at theta 1e6,
    factor 16 over an original 16384, beta_fast 32 / beta_slow 1, mscale = mscale_all_dim = 1 (attention factor exactly
    1), and transformers' llama-4 q scale (beta 0.1 over 16384: exactly 1 below position 16384, where the engine caps);
  * the names, both ways: every HF tensor is a row of the declaration or under a vision prefix (consumed by name), and
    every row the declaration expects is present; every tensor's dtype and shape against the declaration.

Roles: the recipe role (dense | norm | embed | head) and the BIND role -- attn_norm, attn_q / _k / _v / _o, ffn_norm,
ffn_gate / _up / _down per layer (block `layers`); embed, output_norm and (Rerank) score at the top.

The container:
  * names ARE the HF names; layouts: bf16 native, exl3m_k* for a dense Linear from the pack (the dense arm's rates,
    exl3_rates.admit).  The score head is bf16 as stored (the engine widens it to f32 for the dot).
  * shards: `vision` (primary: pulsar.kv only -- no tower in v1), `layers.0..L-1`, `top`.
  * pulsar.kv: general.architecture = "nemotron_enc"; text_config verbatim under `nemotron_enc.` (kv.config_kvs); the
    builder's facts: pulsar.encoder ("embed" | "rerank"), pulsar.pooling ("mean"), pulsar.normalize (Embed), the
    recipe and the sources.
  * side files: tokenizer.json, tokenizer_config.json, chat_template.jinja and config.json, as shipped (the checkpoints
    carry no generation_config.json: an encoder never stops, so it needs no stop ids).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import exl3_rates
import kv as KV
import producers as PR
from names import Mapped

FAMILY = "nemotron_enc"
MODEL_TYPE = "mistral3"
KINDS = {"Mistral3Model": "embed", "Mistral3ForSequenceClassification": "rerank"}
LM_PREFIX = {"embed": "language_model.", "rerank": "model.language_model."}
SCORE = "language_model.score.weight"
PACK_PREFIX = "model."
META = {"pulsar.family": FAMILY}
SETTINGS = ()
EXPERT_STACKS = False
GATE_UP_PARTS = ()
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "model.vision_tower.", "model.multi_modal_projector.")
EXL3 = set(exl3_rates.K2)

# the switches the engine implements: text_config key -> the one value it implements
ASSUME = {"model_type": "ministral3", "is_causal": False, "hidden_act": "silu", "tie_word_embeddings": False,
          "sliding_window": None}
ROPE = {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0, "original_max_position_embeddings": 16384,
        "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1}
TOP_ASSUME = {"is_causal": False, "pooling": "avg"}

# per layer: HF suffix -> (bind role, recipe role); the shape is filled in by expected()
LAYER = {
    "input_layernorm.weight": ("attn_norm", "norm"),
    "self_attn.q_proj.weight": ("attn_q", "dense"),
    "self_attn.k_proj.weight": ("attn_k", "dense"),
    "self_attn.v_proj.weight": ("attn_v", "dense"),
    "self_attn.o_proj.weight": ("attn_o", "dense"),
    "post_attention_layernorm.weight": ("ffn_norm", "norm"),
    "mlp.gate_proj.weight": ("ffn_gate", "dense"),
    "mlp.up_proj.weight": ("ffn_up", "dense"),
    "mlp.down_proj.weight": ("ffn_down", "dense"),
}


def _need(d, key, what):
    if key not in d:
        raise SystemExit(f"config.json {what} has no {key!r} -- the Nemotron-3.5 encoder declaration reads it")
    return d[key]


def _assume(d, key, want, what):
    got = _need(d, key, what)
    if got != want:
        raise SystemExit(f"config.json {what}.{key} = {got!r}; the encoder declaration implements {want!r} -- refusing")


@dataclass(frozen=True)
class Shape:
    """The declared shape (the engine-side shape table refuses a container that disagrees)."""
    kind: str                   # "embed" | "rerank"
    hidden: int
    vocab: int
    n_layer: int
    attn: tuple                 # (n_head, n_kv, head_dim)
    ffn: int
    eps: float
    recipe: str = ""
    table: dict = field(default=None, compare=False, repr=False)   # {name: Mapped} (map_hf answers here)

    @property
    def lm(self):
        return LM_PREFIX[self.kind]

    def __str__(self):
        nq, nkv, hd = self.attn
        return (f"nemotron_enc ({self.kind}): {self.n_layer} bidirectional layers, {self.hidden} wide, {nq}q/{nkv}kv/"
                f"{hd}, ffn {self.ffn}, yarn rope, mean pool -> "
                f"{'L2-normalised embedding' if self.kind == 'embed' else 'score'}, vision not written (v1), "
                f"recipe {self.recipe}")


def declare(top: dict) -> Shape:
    arch = top.get("architectures")
    if top.get("model_type") != MODEL_TYPE or not (isinstance(arch, list) and len(arch) == 1 and arch[0] in KINDS):
        raise SystemExit(f"config.json: model_type {top.get('model_type')!r} / architectures {arch!r}; nemotron_enc.py "
                         f"reads {MODEL_TYPE} / one of {sorted(KINDS)}")
    kind = KINDS[arch[0]]
    for k, v in TOP_ASSUME.items():
        _assume(top, k, v, "the top level")
    c = _need(top, "text_config", "the top level")
    w = "text_config"
    for k, v in ASSUME.items():
        _assume(c, k, v, w)
    rp = _need(c, "rope_parameters", w)
    for k, v in ROPE.items():
        _assume(rp, k, v, w + ".rope_parameters")
    ls = _need(c, "llama_4_scaling", w)
    if ls.get("beta") != ROPE["llama_4_scaling_beta"] or \
            ls.get("original_max_position_embeddings") != ROPE["original_max_position_embeddings"]:
        raise SystemExit(f"config.json text_config.llama_4_scaling = {ls!r}; the declaration implements beta 0.1 over "
                         f"16384 -- refusing")
    sh = Shape(kind=kind, hidden=int(_need(c, "hidden_size", w)), vocab=int(_need(c, "vocab_size", w)),
               n_layer=int(_need(c, "num_hidden_layers", w)),
               attn=(int(_need(c, "num_attention_heads", w)), int(_need(c, "num_key_value_heads", w)),
                     int(_need(c, "head_dim", w))),
               ffn=int(_need(c, "intermediate_size", w)), eps=float(_need(c, "rms_norm_eps", w)))
    nq, nkv, hd = sh.attn
    if nq % nkv or not 1 <= sh.n_layer <= 64:
        raise SystemExit(f"config.json text_config: {nq} q heads over {nkv} kv heads, {sh.n_layer} layers")
    return sh


def expected(sh: Shape):
    """{HF name: (dtype, shape, Mapped)} -- every trunk tensor (and the Reranker's head) the declaration expects."""
    H, F = sh.hidden, sh.ffn
    nq, nkv, hd = sh.attn
    dims = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
            "self_attn.q_proj.weight": [nq * hd, H], "self_attn.k_proj.weight": [nkv * hd, H],
            "self_attn.v_proj.weight": [nkv * hd, H], "self_attn.o_proj.weight": [H, nq * hd],
            "mlp.gate_proj.weight": [F, H], "mlp.up_proj.weight": [F, H], "mlp.down_proj.weight": [H, F]}
    out = {}

    def add(name, shape, bind, role, shard, layer=None):
        rec = (("role", bind),) + ((("block", "layers"), ("layer", layer)) if layer is not None else ())
        fam = "layer" if layer is not None else "top"
        out[name] = ("BF16", shape, Mapped(name, name, fam, shard, layer, None, None, False, role=role, bind=rec))

    for il in range(sh.n_layer):
        for s, (bind, role) in LAYER.items():
            add(f"{sh.lm}layers.{il}.{s}", dims[s], bind, role, f"layers.{il}", il)
    add(sh.lm + "embed_tokens.weight", [sh.vocab, H], "embed", "embed", "top")
    add(sh.lm + "norm.weight", [H], "output_norm", "norm", "top")
    if sh.kind == "rerank":
        add(SCORE, [1, H], "score", "head", "top")
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
        raise SystemExit(f"{len(unknown)} tensors the encoder naming table does not map -- refusing: {unknown[:5]}")
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
def source(hf):
    """The checkpoint as is: every tensor the plan writes is an HF tensor by its own name."""
    hf.decl = declare(hf.config["top_level"])
    hf.mapped = table(hf, hf.decl)
    return hf


def map_hf(name: str, shape: Shape) -> Mapped | None:
    return shape.table.get(name)


class PackView:
    """The --exl3 pack under the checkpoint's names: the staged trunk keys a Linear `model.layers.N.*` where the
    checkpoint says `language_model.layers.N.*` (Embed) or `model.language_model.layers.N.*` (Rerank)."""

    def __init__(self, pack, lm):
        self.pack = pack
        self.dir = pack.dir
        self.lm = lm

    def key(self, k):
        if not k.startswith(self.lm):
            raise SystemExit(f"{k}: an EXL3 Linear outside {self.lm!r} -- the pack holds the trunk only")
        return PACK_PREFIX + k[len(self.lm):]

    def names(self):
        return self.pack.names()

    def has_linear(self, key):
        return self.pack.has_linear(self.key(key))

    def linear(self, key, k, n, rates):
        return self.pack.linear(self.key(key), k, n, rates)


def shape(hf, ctx):
    src = ctx.sources.get("exl3")
    if src is not None and not isinstance(src, PackView):
        ctx.sources["exl3"] = PackView(src, hf.decl.lm)
    if ctx.sources.get("exl3_experts") is not None:
        raise SystemExit("--exl3-experts: the encoder has no routed experts")
    if ctx.recipe is None:
        default_recipe(hf, None, ctx)
    return replace(hf.decl, recipe=ctx.recipe.name, table=hf.mapped)


def shard_order(shape):
    return ["vision"] + [f"layers.{i}" for i in range(shape.n_layer)] + ["top"]


def declared_shape(m, hshape):
    return list(hshape)


# what the loader binds, by recipe role: a dense Linear EXL3 (the dense arm's rates) or bf16; every other tensor in
# its source dtype (the score head bf16: the engine widens it to f32 for the dot)
def admit(hf, m, fmt, name):
    native = PR.NATIVE_LAYOUT.get(hf.dtype(name))
    allowed = {"native", native}
    if m.role == "dense":
        allowed |= EXL3
    if fmt not in allowed:
        raise SystemExit(f"{name}: the encoder loader binds this {m.role} tensor as {sorted(a for a in allowed if a)}, "
                         f"not {fmt}")


def default_recipe(hf, mapped, ctx):
    raise SystemExit(f"{MODEL_TYPE}: pass --recipe (tools/container/format-maps/nemotron_enc-*.json)")


def gate_up_key(rec):
    return rec["gguf_name"].rsplit(".", 1)[0]


# ---------------------------------------------------------------------------
# pulsar.kv
# ---------------------------------------------------------------------------
def build_kv(hf, ctx):
    """general.*; text_config verbatim (kv.config_kvs); the builder's facts (kind, pooling, normalisation)."""
    if ctx.reap_map:
        raise SystemExit("--reap-map: REAP survivor maps are a DeepSeek table (validate_reap_metadata)")
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    kind = ctx.shape.kind
    kvs = [("general.architecture", "string", FAMILY), ("general.type", "string", "model"),
           ("general.name", "string", "Nemotron-3.5-Embed-8B" if kind == "embed" else "Nemotron-3.5-Rerank-8B")]
    KV.config_kvs(FAMILY, dict(hf.config["top_level"]["text_config"]), kvs)
    kvs += [("pulsar.encoder", "string", kind), ("pulsar.pooling", "string", "mean"),
            ("pulsar.normalize", "bool", kind == "embed"),
            ("pulsar.recipe", "string", ctx.recipe.name),
            ("pulsar.recipe.sha256", "string", ctx.recipe.sha256 or ""),
            ("pulsar.vision_present", "bool", False)]
    src = ctx.sources.get("exl3")
    if src is not None:
        kvs.append(("pulsar.source.exl3", "string", os.path.basename(os.path.normpath(src.dir))))
    kvs.append(("pulsar.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))))
    out = [KV.entry(k, t, v) for k, t, v in kvs]
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate kv key: " + ", ".join(sorted(k for k in set(keys) if keys.count(k) > 1)))
    return out


# ---------------------------------------------------------------------------
# side files
# ---------------------------------------------------------------------------
SIDE = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def side_files(hf, ctx):
    """[(container name, source path)]: the tokenizer and template the engine loads beside the shards, from
    --tokenizer DIR (default: the HF dir), and config.json the checkpoint's own."""
    files = [(n, os.path.join(ctx.tokenizer_dir, n)) for n in SIDE] + [("config.json", os.path.join(hf.dir, "config.json"))]
    missing = [src for _, src in files if not os.path.isfile(src)]
    if missing:
        raise SystemExit(f"the encoder container carries the checkpoint's tokenizer: {', '.join(missing)} missing "
                         f"(--tokenizer DIR names a directory with {', '.join(SIDE)})")
    return files

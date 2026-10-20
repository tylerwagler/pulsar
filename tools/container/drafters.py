"""drafters.py -- the block-drafter input of the direct builder (L298 LiLiCorr, L299 DFlash2): one drafter checkpoint
beside a family's target checkpoint, written into the container as shard `drafter` -- the engine's block-drafter lane
(src/engine/spec_block.h) reads it whatever the family.

  build.py ... --drafter KIND DIR     the family's with_drafter(hf, KIND, DIR) -> with_drafter(hf, kind, dir, Target)

KINDS (the engine's k_kinds table, spec_block.cpp, reads each one's config under `<kind>.`):
  lilicorr   NVIDIA's LiLiCorr drafter for Nemotron 3.5 Super (ModelOpt W4A16_NVFP4, its own lm_head) -- below
  dflash2    z-lab's DFlash2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2: BF16, binds the TARGET's lm_head, a
             CandidateSelector head) -- the DFlash2 section

Container names are the drafter's HF names under DRAFTER (`drafter.layers.0.mlp.gate_proj.weight`), all in shard
`drafter`; every tensor carries its bind record (role / block / layer): the backbone block `drafter`, the chain head's
block (`lilicorr`, `selector`).  pulsar.drafter names the kind; the drafter's config is written verbatim under
`<kind>.` (kv.config_kvs) and pulsar.source.drafter names the checkpoint.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import kv as KV
import producers as PR
from hf_source import HFCheckpoint
from names import Mapped


@dataclass(frozen=True)
class Target:
    """What a drafter is checked against and reads from its target: the hidden width, the vocabulary, the HF blocks;
    for a drafter that binds the target's head (DFlash2): the target checkpoint, its head's name and the head's
    rows (a draft vocabulary's ids, ascending; None = the whole vocabulary)."""
    hidden: int
    vocab: int
    n_blocks: int
    hf: object = None
    head_name: str = ""
    head_ids: tuple | None = None


# ---------------------------------------------------------------------------
# LiLiCorr (L298): NVIDIA's LiLiCorr block drafter for Nemotron 3.5 Super -- a DFlash
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


def lili_config(top: dict, target: Target):
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
    """The checked drafter: its checkpoint, its {container name: Mapped}, its kv triples, and (DFlash2) the tensors
    derived from the target (`derived`: container name -> (dtype, shape, producer descriptor))."""

    def __init__(self, kind, path, target: Target):
        if kind not in KINDS:
            raise SystemExit(f"--drafter {kind}: the builder reads {sorted(KINDS)}")
        self.kind, self.dir = kind, path
        self.ckpt = HFCheckpoint(path)
        self.derived = {}
        if kind == "dflash2":
            dflash2_build(self, target)
            return
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
    build.py and entries.py read (names / has / shape / dtype / span / raw / copy_src); the rest is the target's.  A
    derived drafter tensor (DFlash2's head rows of the target's lm_head, their ids) is produced from the target."""

    def __init__(self, hf, drafter: Drafter):
        self.hf, self.drafter = hf, drafter
        self.dir, self.config, self.generation_config = hf.dir, hf.config, hf.generation_config
        if hasattr(hf, "decl"):                       # a family view's own (Nemotron-H's declaration and table)
            self.decl = hf.decl
            self.mapped = {**hf.mapped, **drafter.mapped}

    def _of(self, name):
        return (self.drafter.ckpt, name[len(DRAFTER):]) if name.startswith(DRAFTER) else (self.hf, name)

    def names(self):
        return self.hf.names() + sorted(self.drafter.mapped)

    def has(self, name):
        if name in self.drafter.derived:
            return True
        src, n = self._of(name)
        return src.has(n)

    def shape(self, name):
        if name in self.drafter.derived:
            return list(self.drafter.derived[name][1])
        src, n = self._of(name)
        return src.shape(n)

    def dtype(self, name):
        if name in self.drafter.derived:
            return self.drafter.derived[name][0]
        src, n = self._of(name)
        return src.dtype(n)

    def span(self, name):
        if name in self.drafter.derived:
            raise SystemExit(f"{name}: a derived tensor has no span (its bytes are produced from the target)")
        src, n = self._of(name)
        return src.span(n)

    def raw(self, name):
        if name in self.drafter.derived:
            return PR.produce(self.drafter.derived[name][2], self.hf)
        src, n = self._of(name)
        return src.raw(n)

    def copy_src(self, name):
        if name in self.drafter.derived:
            d = self.drafter.derived[name]
            return ("produce", d[2]), PR.bytes_for("native", list(reversed(d[1])), d[0])
        src, n = self._of(name)
        return src.copy_src(n)


def with_drafter(hf, kind, path, target: Target):
    """build.py --drafter KIND DIR: the checkpoint view carrying the drafter (refused by name unless KIND is one this
    builder reads and its config and tensors agree with the declaration and the target)."""
    return WithDrafter(hf, Drafter(kind, path, target))


# ---------------------------------------------------------------------------
# DFlash2 (L299): z-lab's DFlash2 block drafter (dflash/model.py DFlash2DraftModel @07ebd93; incoai's for
# Qwen3.8-Flash-Next) -- the LiLiCorr backbone (5 Qwen3-style non-causal SWA layers with the grouped dynamic
# convolutions, fed by the target's hidden states at dflash_config.target_layer_ids through `fc`) ending in the TARGET's
# lm_head and a CandidateSelector (a rank-R predecessor / successor factorisation over the vocabulary re-scoring each
# slot's top-k against the token chosen before it).  The checkpoint is BF16 throughout; the formats are the recipe's
# (format-maps/qwen38fn-*-dflash2.json: the Linears nvfp4 from BF16 -- producers.nvfp4_from_bf16 --, the rest BF16).
#
# The head: the drafter has no lm_head of its own; the builder writes one -- the target's lm_head rows at
# Target.head_ids (a draft vocabulary: the cheaper read per draft pass) as `drafter.lm_head.weight` (producers.rows_gather
# of the target's tensor) and their ids as `drafter.lm_head.ids` (I32), or the whole head with no ids.
#
# Recipe roles: `drafter_linear` (nvfp4 or bf16), `drafter_native` (as stored), `drafter_ids` (i32).
# ---------------------------------------------------------------------------
DF2_TOP = {"fc.weight": ("fc", "drafter", True), "hidden_norm.weight": ("hidden_norm", "drafter", False),
           "norm.weight": ("out_norm", "drafter", False),
           "candidate_selector.hidden_projection.weight": ("hidden_proj", "selector", True),
           "candidate_selector.predecessor_codebook": ("pred_codebook", "selector", False),
           "candidate_selector.successor_codebook": ("succ_codebook", "selector", False)}
DF2_LAYER = {k: (v[0], "drafter", v[0] in ("attn_q", "attn_k", "attn_v", "attn_o", "attn_conv_proj", "mlp_conv_proj",
                                           "mlp_gate", "mlp_up", "mlp_down"))
             for k, v in LILI_LAYER.items()}
# The config.json: every top-level key WRITTEN under `dflash2.<key>` (kv.config_kvs: dflash_config and rope_parameters
# flattened), IGNORED for the reason given, or refused.
DF2_WRITE = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
             "intermediate_size", "hidden_act", "rms_norm_eps", "vocab_size", "max_position_embeddings", "attention_bias",
             "rope_parameters", "tie_word_embeddings", "num_target_layers", "layer_types", "sliding_window",
             "use_sliding_window", "max_window_layers", "is_causal", "dflash_config", "bos_token_id", "eos_token_id")
DF2_IGNORE = {
    "architectures": "pinned [DFlash2DraftModel]; pulsar.drafter names the drafter",
    "model_type": "pinned qwen3 (the backbone's HF class); pulsar.drafter names the drafter",
    "dtype": "every tensor's dtype is checked against the declaration instead",
    "initializer_range": "training only",
    "attention_dropout": "training only (pinned 0.0)",
    "pad_token_id": "pinned null: no padding in a draft block",
    "transformers_version": "provenance of the checkpoint's writer",
    "use_cache": "an HF generate() switch; the engine keeps the drafter's KV as its declared state",
}
DF2_DFLASH = ("block_size", "conv_group_size", "conv_kernel_size", "mask_token_id", "selector_rank", "selector_top_k",
              "target_layer_ids")
DF2_PIN = {("", "architectures"): ["DFlash2DraftModel"], ("", "model_type"): "qwen3", ("", "hidden_act"): "silu",
           ("", "attention_bias"): False, ("", "tie_word_embeddings"): False, ("", "is_causal"): False,
           ("", "use_sliding_window"): True, ("", "attention_dropout"): 0.0, ("", "pad_token_id"): None,
           ("rope_parameters", "rope_type"): "default"}


def _df2_int(d, key, what, lo=1):
    if key not in d:
        raise SystemExit(f"drafter config.json {what} has no {key!r} -- the DFlash2 declaration reads it")
    v = d[key]
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise SystemExit(f"drafter config.json {what}.{key} = {v!r}: an integer >= {lo}")
    return v


def dflash2_build(self: Drafter, target: Target):
    """The DFlash2 checkpoint checked against the declaration and the target -> self.kvs, self.mapped,
    self.derived."""
    top = self.ckpt.config["top_level"]
    unknown = sorted(set(top) - set(DF2_WRITE) - set(DF2_IGNORE))
    if unknown:
        raise SystemExit(f"drafter config.json: key {unknown[0]!r} is neither written nor ignored by the DFlash2 "
                         f"declaration -- refusing (all unknown: {unknown})")
    dfl, rope = top.get("dflash_config"), top.get("rope_parameters")
    if not isinstance(dfl, dict) or not isinstance(rope, dict):
        raise SystemExit("drafter config.json: dflash_config and rope_parameters must be objects")
    bad = sorted(set(dfl) - set(DF2_DFLASH))
    if bad:
        raise SystemExit(f"drafter config.json dflash_config: key {bad[0]!r} is not one the DFlash2 declaration reads")
    if sorted(rope) != ["rope_theta", "rope_type"]:
        raise SystemExit(f"drafter config.json rope_parameters {sorted(rope)}: the declaration reads rope_theta and "
                         f"rope_type (default: no scaling)")
    for (blk, k), want in DF2_PIN.items():
        d = {"": top, "rope_parameters": rope}[blk]
        got = d.get(k, "<absent>") if k != "pad_token_id" else d.get(k)
        if got != want or type(got) is not type(want):
            raise SystemExit(f"drafter config.json {blk + '.' if blk else ''}{k} = {got!r}; the DFlash2 declaration "
                             f"implements {want!r} -- refusing")
    w = "the top level"
    H, nl = _df2_int(top, "hidden_size", w), _df2_int(top, "num_hidden_layers", w)
    nq, nkv, hd = _df2_int(top, "num_attention_heads", w), _df2_int(top, "num_key_value_heads", w), _df2_int(top, "head_dim", w)
    V, nt, ff = _df2_int(top, "vocab_size", w), _df2_int(top, "num_target_layers", w), _df2_int(top, "intermediate_size", w)
    if H != target.hidden or V != target.vocab or nt != target.n_blocks:
        raise SystemExit(f"drafter config.json: hidden_size {H} / vocab_size {V} / num_target_layers {nt}; the target's "
                         f"are {target.hidden} / {target.vocab} / {target.n_blocks} -- refusing")
    if top.get("layer_types") != ["sliding_attention"] * nl or _df2_int(top, "max_window_layers", w) != nl:
        raise SystemExit(f"drafter config.json layer_types / max_window_layers: the declaration implements {nl} "
                         f"sliding_attention layers")
    _df2_int(top, "sliding_window", w)
    d = "dflash_config"
    taps, group = _df2_int(dfl, "conv_kernel_size", d), _df2_int(dfl, "conv_group_size", d)
    rank, topk, block = _df2_int(dfl, "selector_rank", d), _df2_int(dfl, "selector_top_k", d), _df2_int(dfl, "block_size", d)
    if not _df2_int(dfl, "mask_token_id", d, lo=0) < V or H % group:
        raise SystemExit(f"drafter config.json: mask_token_id < vocab {V} and conv_group_size {group} dividing {H}")
    tl = dfl.get("target_layer_ids")
    if (not isinstance(tl, list) or not tl or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < nt
                                                  for x in tl)):
        raise SystemExit(f"drafter config.json dflash_config.target_layer_ids {tl!r}: HF block indices < {nt}")
    del block, topk
    self.kvs = []
    for k in DF2_WRITE:
        n0 = len(self.kvs)
        KV.config_kvs(f"dflash2.{k}", top[k], self.kvs)
        if len(self.kvs) == n0:
            raise SystemExit(f"drafter config.json {k} = {top[k]!r}: nothing to write (null / empty) -- refusing")
    # the declaration's tensors: {HF name: (shape, bind role, block, layer, Linear?)}
    kp = [2 * taps * H // group, H]
    lay = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
           "self_attn.q_proj.weight": [nq * hd, H], "self_attn.k_proj.weight": [nkv * hd, H],
           "self_attn.v_proj.weight": [nkv * hd, H], "self_attn.o_proj.weight": [H, nq * hd],
           "self_attn.q_norm.weight": [hd], "self_attn.k_norm.weight": [hd],
           "attention_conv.base_kernel": [2, taps, H], "attention_conv.kernel_projection.weight": kp,
           "mlp_conv.base_kernel": [2, taps, H], "mlp_conv.kernel_projection.weight": kp,
           "mlp.gate_proj.weight": [ff, H], "mlp.up_proj.weight": [ff, H], "mlp.down_proj.weight": [H, ff]}
    tops = {"fc.weight": [H, len(tl) * H], "hidden_norm.weight": [H], "norm.weight": [H],
            "candidate_selector.hidden_projection.weight": [rank, H],
            "candidate_selector.predecessor_codebook": [V, rank], "candidate_selector.successor_codebook": [V, rank]}
    table = {n: (s, *DF2_TOP[n]) for n, s in tops.items()}
    for i in range(nl):
        for sfx, s in lay.items():
            table[f"layers.{i}.{sfx}"] = (s, *DF2_LAYER[sfx][:2], DF2_LAYER[sfx][2])
    self.mapped = {}
    for n, (shp, bind, block, linear) in table.items():
        layer = int(n.split(".")[1]) if n.startswith("layers.") else None
        rec = (("role", bind), ("block", block)) + ((("layer", layer),) if layer is not None else ())
        self.mapped[DRAFTER + n] = Mapped(DRAFTER + n, DRAFTER + n, "drafter", "drafter", layer, None, None, False,
                                         role="drafter_linear" if linear else "drafter_native", bind=rec)
    have = set(self.ckpt.names())
    unknown = sorted(have - set(table))
    missing = sorted(set(table) - have)
    if unknown or missing:
        raise SystemExit(f"the DFlash2 checkpoint {self.dir} and the declaration disagree -- refusing: not declared "
                         f"{unknown[:5]}, missing {missing[:5]}")
    wrong = [f"{n}: {self.ckpt.dtype(n)} {self.ckpt.shape(n)}, the declaration says BF16 {s}"
             for n, (s, *_r) in table.items() if (self.ckpt.dtype(n), self.ckpt.shape(n)) != ("BF16", s)]
    if wrong:
        raise SystemExit(f"{len(wrong)} drafter tensors disagree with the declaration -- refusing: {wrong[:3]}")
    # the head: the target's lm_head (rows at head_ids, or all of it)
    hf, hn = target.hf, target.head_name
    if hf is None or not hf.has(hn) or hf.dtype(hn) != "BF16" or hf.shape(hn) != [V, H]:
        raise SystemExit(f"DFlash2 binds the target's head: the target checkpoint holds no BF16 [{V}, {H}] {hn!r}")
    ids = list(target.head_ids) if target.head_ids is not None else None
    if ids is not None and (not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= V):
        raise SystemExit("the draft vocabulary's ids must be ascending, distinct and inside the vocabulary")
    head = DRAFTER + "lm_head.weight"
    rows = len(ids) if ids is not None else V
    self.derived[head] = ("BF16", [rows, H], PR.spec("rows_gather", [hn], row_bytes=2 * H,
                                                      rows=ids if ids is not None else list(range(V))))
    self.mapped[head] = Mapped(head, head, "drafter", "drafter", None, None, None, False, role="drafter_linear",
                               bind=(("role", "head"), ("block", "drafter")))
    if ids is not None:
        hid = DRAFTER + "lm_head.ids"
        self.derived[hid] = ("I32", [rows], PR.spec("i32_values", [], values=ids))
        self.mapped[hid] = Mapped(hid, hid, "drafter", "drafter", None, None, None, False, role="drafter_ids",
                                  bind=(("role", "head_ids"), ("block", "drafter")))


KINDS = ("lilicorr", "dflash2")

"""deepseek.py -- the DeepSeek family's hooks for the one plan (build.plan; L279): V4 Flash / Vision-Exp
(model_type deepseek_v4) and V4.1 (deepseek_v41).

What is the family's own and nothing else: the naming table (names.py: HF name -> container / engine name, role,
shard, the drop rules), the shard order, the declared-shape quirks (policy.declared_shape), which formats its
loader binds (admit), the DEFAULT recipe when none is given (default_recipe, from policy.layout_for and
the EXL3 source), how routed experts group (per-expert HF tensors fold into one family per (block, layer, part)),
and the arch KV (kv.py).  The plan, the recipe, the producer table, the writer and the source readers are shared.
"""
from __future__ import annotations

import exl3_rates
import kv as KV
import names as N
import policy as P
import producers as PR
import entries as EN
import recipe as R

META = {}                       # no family key: a DeepSeek container predates pulsar.family
SETTINGS = ()                   # no family recipe settings
EXPERT_STACKS = False           # routed experts arrive one HF tensor per expert-projection, grouped after the walk
GATE_UP_PARTS = ("w1", "w3")    # a layer's gate and up families must share one layout (the pair kernels read both)
EXL3 = set(exl3_rates.K2)
EXPERT_FORMATS = {"cutlass_mxfp4", *EXL3}   # a routed expert: CUTLASS MXFP4 or EXL3 (the rate by its arm)


def shape(hf, ctx):
    """From config.json alone (names.py's rule): the checkpoint's word, not a guess from its names; a layer-subset
    fixture (--layers) keeps the source layers it names (names.ModelShape.subset)."""
    full = N.ModelShape.from_config(hf.config["top_level"])
    if not ctx.keep_layers:
        return full
    return full.subset(ctx.keep_layers, KV.text_config(hf).get("dspark_target_layer_ids", ()))


shard_order = N.shard_order
map_hf = N.map_hf
declared_shape = P.declared_shape


def default_format(hf, m, name):
    """policy.layout_for as a recipe format: `native` where the layout is the source dtype's own."""
    layout = P.layout_for(m, hf.dtype(name), hf.shape(name))
    return "native" if layout == PR.NATIVE_LAYOUT.get(hf.dtype(name)) else layout


def admit(hf, m, fmt, name):
    """What the DeepSeek loader binds (weights.cpp): a routed expert as CUTLASS MXFP4 or EXL3; every other tensor
    only as its source dtype says (policy.layout_for) -- a recipe may restate that, never change it."""
    allowed = EXPERT_FORMATS if m.role.startswith("expert_") else {default_format(hf, m, name)}
    if fmt in allowed:
        return
    if fmt in EXL3 and m.role in ("dense", "shared_expert"):
        raise SystemExit(f"{name}: DeepSeek EXL3 dense refused -- the DeepSeek loader admits its dense / attention / "
                         "shared-expert linears at the MX slot's E4M3 only (weights.cpp tensor_expect_mxfp8, "
                         "PULSAR_ACT_SLOT_E4M3); the EXL3 dense arm reads bf16 rows (weight_format.cpp "
                         "pulsar_dense_arm_for), which the DeepSeek forward does not emit there")
    raise SystemExit(f"{name}: the DeepSeek loader binds this {m.role} tensor as {sorted(allowed)}, not {fmt}")


def default_recipe(hf, mapped, ctx):
    """Today's policy as a pulsar.recipe.v1: every tensor in the layout its source dtype says (policy.layout_for;
    the native dtypes as `native`), the routed experts of every main layer the EXL3 checkpoint holds at the rate its
    trellis holds (source exl3), the drafter's experts from the HF checkpoint."""
    exl3 = ctx.sources.get("exl3")
    held = set(N.exl3_layers(exl3.names())) if exl3 else set()
    assign, words = {}, {}
    for name, m in mapped.items():
        if not m.emit or m.is_scale:
            continue
        if m.role.startswith("expert_") and R.block_of(m) == "layers" and m.layer in held:
            key = (m.layer, m.part)
            if key not in words:
                o, i = PR.matrix_dims(EN.kind_of(hf, name), hf.shape(name))
                words[key] = exl3.linear(N.exl3_expert_key("layers", m.layer, 0, m.part), i, o,
                                         exl3_rates.LAYOUT_BY_WORDS)[1]
            assign[name] = (exl3_rates.LAYOUT_BY_WORDS[words[key]], "exl3")
        else:
            assign[name] = (default_format(hf, m, name), None)
    mt = hf.config["top_level"]["model_type"]
    return R.Recipe({"recipe": f"{mt}-default", "model_type": mt,
                     "about": "generated: DeepSeek's default (policy.layout_for + the EXL3 checkpoint's rates)",
                     "rows": R.rows_for(mapped, assign)}, f"<{mt} default recipe>")


def group_key(m):
    return (m.shard, m.layer, m.part)


def group_families(hf, key, slots, fmt, ctx):
    """One routed family from its per-expert HF tensors {e: (name, mapped)}: from an EXL3 checkpoint (verbatim
    ranges) or the HF FP4 source (cutlass_mxfp4)."""
    shard, layer, part = key
    n_exp = max(slots) + 1
    if sorted(slots) != list(range(n_exp)):
        raise SystemExit(f"{shard} {part}: experts not contiguous 0..{n_exp - 1}")
    name0, m0 = slots[0]
    layout, src = fmt[name0]
    odd = [slots[e][0] for e in range(n_exp) if fmt[slots[e][0]] != (layout, src)]
    if odd:
        raise SystemExit(f"{shard} {part}: experts of one family must share a format and source; {odd[0]} is "
                         f"{fmt[odd[0]]}, expert 0 {(layout, src)}")
    out, inp = PR.matrix_dims(EN.kind_of(hf, name0), hf.shape(name0))
    if layout in EXL3:
        block = shard.split(".")[0]
        srcs = [("ranges", EN.exl3_ranges(ctx.sources[src], N.exl3_expert_key(block, layer, e, part), layout, inp,
                                           out)[0]) for e in range(n_exp)]
    else:
        srcs = [EN.expert_src(hf, slots[e][0], layout, out, inp) for e in range(n_exp)]
    # A fixture's blocks are renumbered (names.py drop rule 4) while its entries keep the HF names, so the record
    # names its entries (the loader's declared form, safetensors.cpp) instead of the blk.N -> layers.N default.
    block_name = shard.split(".")[0]
    extras = ({"entry_name": f"{block_name}.{layer}.ffn.experts.{{e}}.{part}.weight"}
              if ctx.shape.keep is not None and block_name == "layers" else {})
    return [dict(gguf_name=m0.gguf_name, part=part, role=m0.role, layout=layout, inp=inp, n=out,
                 entry_names=[slots[e][1].container_name for e in range(n_exp)], srcs=srcs, extras=extras)]


def build_kv(hf, ctx):
    if ctx.ple_rows:
        raise SystemExit("--ple-rows: the PLE row file is a qwen4_exp table")
    return KV.build_kv(hf, ctx.tokenizer_dir, ctx.reap_map, keep=ctx.shape.keep, engram_layout=ctx.engram_layout)

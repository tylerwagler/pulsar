"""deepseek.py -- the DeepSeek family's hooks for the one plan (build.plan; L279 step 4): V4 Flash / Vision-Exp
(model_type deepseek_v4) and V4.1 (deepseek_v41).

What is the family's own and nothing else: the naming table (names.py: HF name -> container / engine name, shard,
the drop rules), the shard order, the declared-shape quirks (policy.declared_shape), the format each source is
written in (policy.layout_for), how routed experts group (per-expert HF tensors fold into one family per
(layer, part)), and the arch KV (kv.py).  The plan, the producer table, the writer and the EXL3 reader are shared.
"""
from __future__ import annotations

import kv as KV
import names as N
import policy as P
import producers as PR
import entries as EN

META = {}                       # no family key: a DeepSeek container predates pulsar.family
EXPERT_STACKS = False           # routed experts arrive one HF tensor per expert-projection, grouped after the walk
EXL3_RATES = P.EXL3_WORDS
GATE_UP_PARTS = ("w1", "w3")    # a layer's gate and up families must share one layout (the pair kernels read both)


def shape(hf, ctx):
    """From config.json alone (names.py's rule): the checkpoint's word, not a guess from its names."""
    return N.ModelShape.from_config(hf.config["top_level"])


shard_order = N.shard_order
map_hf = N.map_hf
declared_shape = P.declared_shape


def exl3_source(m, ctx):
    return ctx.exl3


def formats(hf, mapped, ctx):
    """name -> layout for every emitted primary tensor: the source's dtype decides (policy.layout_for, --format-map
    overrides consulted); a routed expert of a layer the EXL3 checkpoint supplies takes the rate its trellis holds."""
    out, words = {}, {}
    for name, m in mapped.items():
        if not m.emit or m.is_scale:
            continue
        if (m.role.startswith("expert_") and ctx.exl3 is not None and m.shard.startswith("layers.")
                and m.layer in ctx.exl3_layers):
            key = (m.layer, m.part)
            if key not in words:
                o, i = PR.matrix_dims(EN.kind_of(hf, name), hf.shape(name))
                words[key] = ctx.exl3.linear(N.exl3_expert_key(m.layer, 0, m.part), i, o, EXL3_RATES)[1]
            out[name] = EXL3_RATES[words[key]]
        else:
            out[name] = P.layout_for(m, hf.dtype(name), hf.shape(name), ctx.overrides)
    return out


def group_key(m):
    return (m.shard, m.layer, m.part)


def group_families(hf, key, slots, fmt, ctx):
    """One routed family from its per-expert HF tensors {e: (name, mapped)}: from the EXL3 checkpoint (verbatim
    ranges) or the HF FP4 source (cutlass_mxfp4)."""
    shard, layer, part = key
    n_exp = max(slots) + 1
    if sorted(slots) != list(range(n_exp)):
        raise SystemExit(f"{shard} {part}: experts not contiguous 0..{n_exp - 1}")
    name0, m0 = slots[0]
    layout = fmt[name0]
    out, inp = PR.matrix_dims(EN.kind_of(hf, name0), hf.shape(name0))
    if layout.startswith("exl3m_"):
        srcs = [("ranges", EN.exl3_ranges(ctx.exl3, N.exl3_expert_key(layer, e, part), layout, inp, out,
                                           EXL3_RATES)[0]) for e in range(n_exp)]
    else:
        srcs = [EN.expert_src(hf, slots[e][0], layout, out, inp) for e in range(n_exp)]
    return [dict(gguf_name=m0.gguf_name, part=part, layout=layout, inp=inp, n=out,
                 entry_names=[slots[e][1].container_name for e in range(n_exp)], srcs=srcs, extras={})]


def build_kv(hf, ctx):
    return KV.build_kv(hf, ctx.tokenizer_dir, ctx.reap_map)

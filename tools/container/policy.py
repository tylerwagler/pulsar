"""policy.py -- DeepSeek's DEFAULT recipe: the layout each container entry is written in when no --recipe names it
(L247; since L279 step 5 it generates the default pulsar.recipe.v1 rows, deepseek.default_recipe, and is no longer
consulted when a recipe is given).

Checked against the checkpoints' own shard headers (not config.json's
torch_dtype), the GGUF-era type policy (`suffix_type`, `policy_type`,
`dspark_type_flags.txt`, archived at tag archive/gguf-tooling-2026-09-24)
collapses to FOUR decisions -- every other table it carried (NORM_BF16,
BF16_GROUP, F32_SOURCE, the drafter's bf16/f32 flags) restated the source's
dtype and is not a decision at all:

  1. F8_E4M3 2-D weights (attention projections, shared experts, indexer wq_b,
     the drafter's main_proj, Engram's wkv) -> `mxfp8_lt` (their `.scale`
     companion folds in; producers.mxfp8_lt).
  2. Routed experts -> the expert SOURCE's layout: I8 nibbles + E8M0 scale
     (the QAT FP4 checkpoint) -> `cutlass_mxfp4`; an EXL3 trellis names its
     own rate in its shape ([k/16, n/16, words]; deepseek.default_recipe).
  3. The drafter's `markov_w2` (bf16 source) -> `fp8_e4m3_soa_k`, k-major
     (L213; the one lossy-by-design row; the engine refuses the v-major dims).
  4. `ffn.gate.tid2eid` -> `i32`: the source stores expert ids as I64 (torch's
     default), the engine binds PULSAR_TENSOR_I32 (weights.cpp
     tensor_expect_layout), and every value is an expert id < 256.  The ONLY
     dtype narrowing in the builder; any other I64 refuses.
  5. The declared SHAPE (declared_shape; dims_ne is its reverse) is the shape
     of what the container HOLDS, which is the source's except twice: a
     `[1, n]` matrix is declared as the vector `[n]` -- the drafter's
     `confidence_head.proj.weight` is stored `[1, E+256]` by both checkpoints
     and bound rank-1 by weights.cpp (`tensor_expect_f32_or_bf16(..., 1,
     E + 256, ...)`) -- and `markov_w2` is declared transposed, `[256, V]`
     (dims_ne `[V, 256]`), because decision 3's producer writes it k-major
     and weights.cpp binds those dims (`tensor_expect_layout(..., 2, V, 256,
     0)`; "an artifact still carrying the v-major markov_w2 ... refuses to
     load").  A rank-1 `[1]` (hc_head_scale) stays rank-1.  Nothing else
     reshapes.

Everything else keeps its native dtype: BF16 -> `bf16`, F32 -> `f32`, I32 ->
`i32`.  A dtype the engine has no layout for (F16, U8, a lone F8_E8M0 ...)
refuses; there is no "closest" format.
"""
from __future__ import annotations

import producers as PR
from names import Mapped


class PolicyError(ValueError):
    pass


def _default_layout(m: Mapped, dtype: str, shape: list[int]) -> str:
    if m.family == "expert":
        if dtype == "I8" and len(shape) == 2:
            return "cutlass_mxfp4"
        raise PolicyError(f"{m.container_name}: routed expert source {dtype} {shape} has no layout")
    if m.gguf_name.endswith(".markov_head.markov_w2.weight"):
        if dtype != "BF16" or len(shape) != 2:
            raise PolicyError(f"{m.container_name}: markov_w2 must be a BF16 matrix, got {dtype} {shape}")
        return "fp8_e4m3_soa_k"
    if m.gguf_name.endswith(".ffn_gate_tid2eid.weight"):
        if dtype not in ("I64", "I32") or len(shape) != 2:
            raise PolicyError(f"{m.container_name}: tid2eid must be an I64/I32 table, got {dtype} {shape}")
        return "i32"
    if dtype == "F8_E4M3":
        if len(shape) != 2:
            raise PolicyError(f"{m.container_name}: F8_E4M3 with shape {shape}; mxfp8_lt is a 2-D layout")
        return "mxfp8_lt"
    if dtype in PR.NATIVE_LAYOUT:
        return PR.NATIVE_LAYOUT[dtype]
    raise PolicyError(f"{m.container_name}: source dtype {dtype} has no engine layout")


def declared_shape(m: Mapped, shape: list[int]) -> list[int]:
    """The row-major shape the container declares for the dense entry `m`
    whose source has `shape` (decision 5); dims_ne is its reverse."""
    if m.gguf_name.endswith(".markov_head.markov_w2.weight"):
        rows, cols = shape
        return [cols, rows]
    if len(shape) == 2 and shape[0] == 1:
        return [shape[1]]
    return list(shape)


def layout_for(m: Mapped, dtype: str, shape: list[int]) -> str:
    """The default layout for the entry `m` whose PRIMARY source tensor has this dtype/shape (the `.weight`;
    companions never decide)."""
    if m.is_scale:
        raise PolicyError(f"{m.container_name}: layout_for takes the primary tensor, not a companion")
    if not m.emit:
        raise PolicyError(f"{m.container_name}: not a container tensor")
    return _default_layout(m, dtype, shape)

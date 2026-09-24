"""policy.py -- the layout each container entry is written in (L247).

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
     own rate in its shape ([k/16, n/16, words] -> exl3m_k2 / k2h / k3).  A
     per-layer format map (overrides) may choose among the expert layouts.
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

Overrides ({container-name or fnmatch pattern: layout}) are consulted first
and must name a layout the source can be written in: a map that asks a bf16
tensor for mxfp8_lt, or a FP4 expert for a layout the builder has no producer
for, refuses instead of quietly falling back to the default.
"""
from __future__ import annotations

from fnmatch import fnmatchcase

from names import Mapped, ModelShape, hf_for_gguf

# The engine's layout vocabulary (src/engine/model.cpp pulsar_layout_names)
# restricted to what the builder produces (README, order of work step 1).
NATIVE = {"BF16": "bf16", "F32": "f32", "I32": "i32"}
EXPERT_LAYOUTS = {"cutlass_mxfp4", "exl3m_k2", "exl3m_k2h", "exl3m_k3"}
EXL3_WORDS = {32: "exl3m_k2", 40: "exl3m_k2h", 48: "exl3m_k3"}
LAYOUTS = set(NATIVE.values()) | EXPERT_LAYOUTS | {"mxfp8_lt", "fp8_e4m3_soa_k"}

# GGUF-era spellings a prisma format map uses -> the engine's.  IQ2 is named so
# a map row can be REPORTED; layout_for refuses it because the builder has no
# producer (the type-44 permutation lived in the archived repack_iq2_mmq.py).
GGUF_FORMAT_NAMES = {
    "MXFP8_LT": "mxfp8_lt", "CUTLASS_MXFP4": "cutlass_mxfp4", "IQ2_XXS_MMQ": "iq2_xxs_mmq_k",
    "BF16": "bf16", "F32": "f32", "I32": "i32", "FP8_E4M3_SOA_K": "fp8_e4m3_soa_k",
    "EXL3M_K2": "exl3m_k2", "EXL3M_K2H": "exl3m_k2h", "EXL3M_K3": "exl3m_k3",
}


class PolicyError(ValueError):
    pass


def _override_for(m: Mapped, overrides: dict) -> str | None:
    hit = overrides.get(m.container_name)
    if hit is not None:
        return hit
    for pat, layout in overrides.items():
        if fnmatchcase(m.container_name, pat):
            return layout
    return None


def _default_layout(m: Mapped, dtype: str, shape: list[int]) -> str:
    if m.family == "expert":
        if dtype == "I8" and len(shape) == 2:
            return "cutlass_mxfp4"
        if dtype == "I16" and len(shape) == 3:
            words = shape[2]
            if words not in EXL3_WORDS:
                raise PolicyError(f"{m.container_name}: {words} words per EXL3 tile is not a rate pulsar reads")
            return EXL3_WORDS[words]
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
    if dtype in NATIVE:
        return NATIVE[dtype]
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


def layout_for(m: Mapped, dtype: str, shape: list[int], overrides: dict) -> str:
    """The layout name for the entry `m` whose PRIMARY source tensor has this
    dtype/shape (the `.weight` / `.trellis`; companions never decide)."""
    if m.is_scale:
        raise PolicyError(f"{m.container_name}: layout_for takes the primary tensor, not a companion")
    if not m.emit:
        raise PolicyError(f"{m.container_name}: not a container tensor")
    default = _default_layout(m, dtype, shape)
    want = _override_for(m, overrides)
    if want is None:
        return default
    if want not in LAYOUTS:
        raise PolicyError(f"{m.container_name}: override names {want!r}, which the builder does not produce")
    if m.family == "expert":
        if want not in EXPERT_LAYOUTS:
            raise PolicyError(f"{m.container_name}: {want!r} is not a routed-expert layout")
        if want.startswith("exl3m_") != default.startswith("exl3m_"):
            raise PolicyError(f"{m.container_name}: override {want!r} but the expert source is {default}")
        return want
    if want != default:
        raise PolicyError(f"{m.container_name}: override {want!r} but the source ({dtype}) is written as {default}")
    return want


def rekey_format_map(gguf_map: dict, shape: ModelShape) -> dict:
    """A prisma per-tensor map keyed by GGUF names ({"blk.10.ffn_down_exps.weight":
    "IQ2_XXS_MMQ", ...}) -> {container name or expert pattern: engine layout}."""
    out = {}
    for gguf_name, fmt in gguf_map.items():
        if fmt not in GGUF_FORMAT_NAMES:
            raise PolicyError(f"format map: {gguf_name}: unknown format {fmt!r}")
        try:
            key = hf_for_gguf(gguf_name, shape)
        except KeyError:
            raise PolicyError(f"format map: {gguf_name}: no container name for this engine name") from None
        if key in out:
            raise PolicyError(f"format map: {gguf_name} re-keys to {key}, already set")
        out[key] = GGUF_FORMAT_NAMES[fmt]
    return out

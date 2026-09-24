"""Format producers for the direct container builder: bytes in, bytes out, no I/O.

Every producer here turns an HF-checkpoint tensor's raw bytes into the exact
bytes one of the engine's layouts holds on disk, and `bytes_for` is the
builder's copy of the engine's byte model.  The authorities these agree with:

  * E4M3 codec / per-32 E8M0 re-derivation: the archived C quantizer,
    `git show archive/gguf-tooling-2026-09-24:gguf-tools/quants_fp.c`
    (`ds4q_f32_to_e4m3` :23-50, `ds4q_quantize_fp8_e4m3` :55-77), and the
    FP8 source decode `gguf-tools/quantize/dsq_codecs.c` (`e4m3fn_to_f32`
    :14-25 -- NaN codes decode to 0.0 -- and `dequant_fp8_weight` :59-99).
  * The block-scale swizzle: `pulsar_mx_sfoff` in src/cuda/pulsar_cuda_mx.cuh
    (`mx_sfoff` below is its line-for-line port; the test grades the
    transpose-based plane builder against it).
  * mxfp8_lt: `ds4q_pack_mxfp8_lt` quants_fp.c:226-255 (refuses shapes that
    are not 128-aligned); byte model `st_bytes_for` src/engine/safetensors.cpp.
  * cutlass_mxfp4: `ds4q_pack_cutlass_mxfp4` quants_fp.c:186-204; byte model
    `cutlass_mxfp4_expert_layout` src/engine/model.cpp.
  * fp8_e4m3_soa_k: `ds4q_fp8_e4m3_soa_k_repack` quants_fp.c:285-297 after the
    k-major transpose in `dsq_generate.c` :269-283; byte model `st_bytes_for`.
  * exl3m_*: `exl3_expert_layout` src/engine/exl3_trellis.h.

Every producer is graded byte-identical against the served source-precision
repack by tools/container/test_producers.py.  numpy only; no torch.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# E4M3 codec: the C reference, vectorised.
# --------------------------------------------------------------------------

_E4M3_SAT = 0x7E  # +448, the largest finite E4M3 magnitude


def _e4m3_decode_table() -> np.ndarray:
    """`e4m3fn_to_f32` (dsq_codecs.c:14-25) for all 256 codes.

    Note the reference's two quirks, kept on purpose: both NaN codes (0x7f, 0xff)
    decode to +0.0, and 0x80 decodes to -0.0 (which the encoder then writes back
    as 0x00 -- see `f32_to_e4m3`)."""
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        a = c & 0x7F
        if a == 0 or a == 0x7F:
            t[c] = -0.0 if (c & 0x80 and a == 0) else 0.0
            continue
        e = (c >> 3) & 0xF
        m = c & 0x7
        v = np.ldexp(np.float32(m), -9) if e == 0 else np.ldexp(np.float32(1.0 + m / 8.0), e - 7)
        t[c] = -v if c & 0x80 else v
    return t


E4M3_VALUE = _e4m3_decode_table()


def f32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """`ds4q_f32_to_e4m3` (quants_fp.c:23-50) over an f32 array, bit-exact.

    Sign follows `x < 0` (so -0.0 encodes as 0x00), magnitudes >= 448 saturate
    to 0x7e, the subnormal arm is lrintf(a * 512) capped at the minimum normal,
    the normal arm is round-to-nearest-even on the top 3 mantissa bits with a
    carry into the exponent and saturation past 448.  NaN is refused: the C
    dies on it ("source tensor is corrupt") instead of encoding 0x7f."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if np.isnan(x).any():
        raise ValueError("f32_to_e4m3: NaN in the input; the source tensor is corrupt")
    sign = np.where(x < 0, np.uint8(0x80), np.uint8(0))
    a = np.abs(x)
    bits = a.view(np.uint32)
    exp = (bits >> np.uint32(23)).astype(np.int32) - 127
    sig = (bits & np.uint32(0x7FFFFF)) | np.uint32(1 << 23)
    e8 = exp + 7

    # subnormal arm: code = lrintf(a * 512), > 7 rounds up to the min normal.
    # (Only read where e8 <= 0, i.e. a < 2^-6; the min() keeps a*512 finite for
    # the inf inputs the saturation arm owns.)
    sub_code = np.rint(np.minimum(a, np.float32(1.0)) * np.float32(512.0)).astype(np.int32)
    sub_code = np.where(sub_code > 7, 8, sub_code).astype(np.uint8)

    # normal arm
    mant3 = ((sig >> np.uint32(20)) & np.uint32(7)).astype(np.int32)
    rem = sig & np.uint32((1 << 20) - 1)
    half = np.uint32(1 << 19)
    round_up = (rem > half) | ((rem == half) & ((mant3 & 1) == 1))
    mant3 = mant3 + round_up.astype(np.int32)
    carry = mant3 == 8
    mant3 = np.where(carry, 0, mant3)
    e8n = e8 + carry.astype(np.int32)
    sat = (e8n > 15) | ((e8n == 15) & (mant3 >= 7))
    norm_code = ((e8n & 0xF) << 3) | (mant3 & 0x7)
    norm_code = np.where(sat, _E4M3_SAT, norm_code).astype(np.uint8)

    code = np.where(e8 <= 0, sub_code, norm_code)
    code = np.where(a == 0, np.uint8(0), code)
    code = np.where(a >= np.float32(448.0), np.uint8(_E4M3_SAT), code)
    return (sign | code).astype(np.uint8)


def _floor_log2_table() -> np.ndarray:
    """floor(log2 |v|) per E4M3 code as int8; -128 marks a zero-valued code
    (0x00, 0x80 and the two NaN codes, which the reference decodes to 0)."""
    t = np.full(256, -128, dtype=np.int8)
    for c in range(256):
        v = abs(float(E4M3_VALUE[c]))
        if v != 0.0:
            t[c] = np.frexp(v)[1] - 1
    return t


_E4M3_FLOOR_LOG2 = _floor_log2_table()

# Re-derivation shifts a whole 32-group by 2^d, d = S_source - S_new.  The new
# exponent is floor(log2 amax) - 7 (max element -> [128, 256)), so with
# floor(log2 |v|) in [-9, 8] for E4M3 magnitudes, d in [-1, 16]; the E8M0 clamps
# only pull d towards 0.  RESHIFT[c, d + 1] is the code |v| * 2^d encodes to.
_D_MIN, _D_MAX = -1, 16


def _reshift_table() -> np.ndarray:
    d = np.arange(_D_MIN, _D_MAX + 1, dtype=np.int32)
    vals = np.ldexp(E4M3_VALUE[:, None], d[None, :]).astype(np.float32)  # exact: power-of-two scaling
    return f32_to_e4m3(vals).reshape(256, d.size)


_RESHIFT = _reshift_table()
_RESHIFT_FLAT = np.ascontiguousarray(_RESHIFT.ravel())


# --------------------------------------------------------------------------
# The block-scale swizzle (pulsar_mx_sfoff) and its plane builder.
# --------------------------------------------------------------------------

def mx_sfoff(row, kb, kbp):
    """Line-for-line port of `pulsar_mx_sfoff` (src/cuda/pulsar_cuda_mx.cuh:57-60)
    == `ds4q_mx_sfoff` (quants_fp.c:169-174).  Works on ints or int arrays."""
    return (((row // 128) * (kbp // 4) + (kb // 4)) * 512
            + (row % 32) * 16 + ((row % 128) // 32) * 4 + (kb % 4))


def _rup(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def swizzle_sf(scale: np.ndarray, rows: int, kb_n: int) -> np.ndarray:
    """The E8M0 plane `sf[mx_sfoff(r, kb, kbp)] = scale[r, kb]`, zero-padded to
    rup(rows, 128) x rup(kb_n, 4), as one transpose.

    mx_sfoff factors as [row/128][kb/4][row%32][(row%128)/32][kb%4], so the
    padded [rows_pad, kb_pad] plane viewed as
    [R, 4 (row%128 / 32), 32 (row%32), KBp/4, 4 (kb%4)] and transposed to
    [R, KBp/4, 32, 4, 4] IS the swizzled plane.  The test asserts this against
    the scalar `mx_sfoff` on every shape it touches."""
    if scale.shape != (rows, kb_n):
        raise ValueError(f"swizzle_sf: scale plane is {scale.shape}, expected {(rows, kb_n)}")
    rows_pad, kb_pad = _rup(rows, 128), _rup(kb_n, 4)
    if (rows_pad, kb_pad) != (rows, kb_n):
        padded = np.zeros((rows_pad, kb_pad), dtype=np.uint8)
        padded[:rows, :kb_n] = scale
        scale = padded
    v = scale.reshape(rows_pad // 128, 4, 32, kb_pad // 4, 4)
    return np.ascontiguousarray(v.transpose(0, 3, 2, 1, 4)).reshape(-1)


def unswizzle_sf(sf: np.ndarray, rows: int, kb_n: int) -> np.ndarray:
    """Inverse of swizzle_sf: the [rows, kb_n] scale plane out of the padded
    swizzled bytes (used by the test's decoder)."""
    rows_pad, kb_pad = _rup(rows, 128), _rup(kb_n, 4)
    if sf.size != rows_pad * kb_pad:
        raise ValueError(f"unswizzle_sf: plane holds {sf.size} bytes, expected {rows_pad * kb_pad}")
    v = sf.reshape(rows_pad // 128, kb_pad // 4, 32, 4, 4).transpose(0, 3, 2, 1, 4)
    return np.ascontiguousarray(v).reshape(rows_pad, kb_pad)[:rows, :kb_n]


# --------------------------------------------------------------------------
# Producers.
# --------------------------------------------------------------------------

def native(raw: bytes) -> bytes:
    """Passthrough: the container keeps the checkpoint's own dtype and bytes."""
    return bytes(raw)


def i64_to_i32(raw: bytes) -> bytes:
    """The one native-dtype conversion the C did (`i64_to_i32`, dsq_generate.c
    :78-90): the checkpoint's I64 index tables (ffn.gate.tid2eid) are stored
    I32; a value outside int32 refuses."""
    v = np.frombuffer(raw, dtype="<i8")
    if v.size and (v.min() < np.iinfo(np.int32).min or v.max() > np.iinfo(np.int32).max):
        raise ValueError("i64_to_i32: value out of int32 range")
    return v.astype("<i4").tobytes()


def _check_fp8_inputs(w_fp8: bytes, scale_e8m0: bytes, out: int, inp: int, block: int) -> None:
    if block not in (32, 128):
        raise ValueError(f"mxfp8_lt: FP8 scale blocks are {block}x{block}, neither 32x32 nor 128x128")
    if out <= 0 or inp <= 0 or out % block or inp % block:
        raise ValueError(f"mxfp8_lt: [out={out}, in={inp}] is not tiled by {block}x{block} blocks")
    if len(w_fp8) != out * inp:
        raise ValueError(f"mxfp8_lt: weight holds {len(w_fp8)} bytes, shape [{out}, {inp}] needs {out * inp}")
    if len(scale_e8m0) != (out // block) * (inp // block):
        raise ValueError(f"mxfp8_lt: scale holds {len(scale_e8m0)} bytes, expected {(out // block) * (inp // block)}")


def _check_lt_shape(out: int, inp: int) -> int:
    """`ds4q_pack_mxfp8_lt`'s refusals (quants_fp.c:230-243): in%32, and the
    128-alignment (out%128, KB%4) the pre-stored layout was defined on."""
    if inp % 32:
        raise ValueError(f"mxfp8_lt: ncols {inp} is not divisible by 32")
    kb_n = inp // 32
    if out % 128 or kb_n % 4:
        raise ValueError(f"mxfp8_lt: shape [in={inp},out={out}] is not 128-aligned "
                         f"(out%128={out % 128}, KB%4={kb_n % 4})")
    return kb_n


def _group_scale_exponents(scale_e8m0: bytes, out: int, inp: int, block: int) -> np.ndarray:
    """The source E8M0 exponent (byte - 127) per [out, in/32] group, as int16."""
    s = np.frombuffer(scale_e8m0, dtype=np.uint8).reshape(out // block, inp // block)
    if s.max() == 255:
        raise ValueError("mxfp8_lt: E8M0 scale byte 255 (Inf/NaN) in the source scale plane")
    per_group = np.repeat(np.repeat(s, block, axis=0), block // 32, axis=1)
    return per_group.astype(np.int16) - 127


def mxfp8_lt(w_fp8: bytes, scale_e8m0: bytes, out: int, inp: int, block: int, mode: str) -> bytes:
    """HF F8_E4M3 [out, in] + F8_E8M0 [out/block, in/block] -> the engine's
    `mxfp8_lt` bytes: E4M3 data row-major [out][in], then the swizzled E8M0 plane
    (rup(out,128) x rup(in/32,4)).

    mode 'rederive' is the archived C codec exactly: decode E4M3 x E8M0 to f32
    (dequant_fp8_weight), per-32 amax -> E8M0 = floor(log2 amax) - 7 clamped to
    [-127, 127] (ds4q_quantize_fp8_e4m3), E4M3 RNE re-encode, pack.  Because
    every step is a power-of-two rescale, it reduces to a per-group shift d and
    a 256 x 18 code table -- bit-identical to the f32 path, without it.

    mode 'verbatim' keeps the source block scale, broadcast per 32-group, and
    copies the codes: no f32 round trip.  It decodes to the same values as
    'rederive' except (a) where re-derivation RAISES a group's exponent by one
    (max |code| in 0x78..0x7e, i.e. 256..448; d = -1): every code halves, and a
    code below 0x10 with an odd LSB -- the odd subnormals 0x01..0x07 and the odd
    minimum-normal codes 0x09..0x0f, which halve onto the subnormal grid -- loses
    that bit to round-to-nearest-even; and (b) the three codes the reference
    decodes to zero (0x80 -> 0x00, NaN codes 0x7f/0xff -> 0x00).  Lowering the
    exponent (d > 0) doubles codes exactly.  Measured on the Vision-Exp dense
    tensors by test_producers.py: ~1e-5 of elements differ in value."""
    if mode not in ("rederive", "verbatim"):
        raise ValueError(f"mxfp8_lt: mode must be 'rederive' or 'verbatim', not {mode!r}")
    _check_fp8_inputs(w_fp8, scale_e8m0, out, inp, block)
    kb_n = _check_lt_shape(out, inp)
    w = np.frombuffer(w_fp8, dtype=np.uint8).reshape(out, inp)
    s_exp = _group_scale_exponents(scale_e8m0, out, inp, block)  # [out, kb_n] int16

    if mode == "verbatim":
        codes = w
        sf = swizzle_sf((s_exp + 127).astype(np.uint8), out, kb_n)
        return codes.tobytes() + sf.tobytes()

    # rederive: floor(log2 amax) per group from the codes alone (monotone in |v|)
    fl = _E4M3_FLOOR_LOG2[w].reshape(out, kb_n, 32).max(axis=2).astype(np.int16)
    zero_group = fl == -128
    if int(s_exp.max()) + 9 > 128:
        raise ValueError("mxfp8_lt: source scale so large that E4M3 x E8M0 overflows f32 in the reference codec")
    t_exp = np.clip(s_exp + fl - 7, -127, 127)
    t_exp = np.where(zero_group, np.int16(-127), t_exp)
    d = np.where(zero_group, np.int16(0), s_exp - t_exp)
    if d.min() < _D_MIN or d.max() > _D_MAX:
        raise AssertionError(f"mxfp8_lt: re-derivation shift {d.min()}..{d.max()} outside the table's [{_D_MIN}, {_D_MAX}]")
    idx = (w.astype(np.uint16) * np.uint16(_RESHIFT.shape[1])
           + np.repeat((d - _D_MIN).astype(np.uint16), 32, axis=1))
    codes = np.take(_RESHIFT_FLAT, idx)
    sf = swizzle_sf((t_exp + 127).astype(np.uint8), out, kb_n)
    return codes.tobytes() + sf.tobytes()


def cutlass_mxfp4(w_i8: bytes, scale_e8m0: bytes, out: int, inp: int) -> bytes:
    """HF I8 [out, in/2] E2M1 nibbles + F8_E8M0 [out, in/32] -> one expert's
    `cutlass_mxfp4` bytes: the nibbles VERBATIM (out*in/2), then the swizzled
    E8M0 plane (rup(out,128) x rup(in/32,4)).  `ds4q_pack_cutlass_mxfp4`
    quants_fp.c:186-204."""
    if inp % 32:
        raise ValueError(f"cutlass_mxfp4: ncols {inp} is not divisible by 32")
    kb_n = inp // 32
    if len(w_i8) != out * inp // 2:
        raise ValueError(f"cutlass_mxfp4: weight holds {len(w_i8)} bytes, [out={out}, in={inp}] needs {out * inp // 2}")
    if len(scale_e8m0) != out * kb_n:
        raise ValueError(f"cutlass_mxfp4: scale holds {len(scale_e8m0)} bytes, expected {out * kb_n}")
    scale = np.frombuffer(scale_e8m0, dtype=np.uint8).reshape(out, kb_n)
    return bytes(w_i8) + swizzle_sf(scale, out, kb_n).tobytes()


def _quantize_fp8_e4m3_planes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`ds4q_quantize_fp8_e4m3` (quants_fp.c:55-77) on an f32 [nrows, ncols]
    array, returned as (E8M0 [nrows, ncols/32], E4M3 [nrows, ncols]) planes."""
    nrows, ncols = x.shape
    if ncols % 32:
        raise ValueError(f"fp8_e4m3: ncols {ncols} is not divisible by 32")
    if np.isnan(x).any():
        raise ValueError("fp8_e4m3: NaN weight in the input; the source tensor is corrupt")
    g = x.reshape(nrows, ncols // 32, 32)
    amax = np.abs(g).max(axis=2)
    e = np.frexp(amax)[1].astype(np.int32)  # amax = m * 2^e, m in [0.5, 1)
    scale_exp = np.where(amax > 0, (e - 1) - 7, -127)
    scale_exp = np.clip(scale_exp, -127, 127)
    inv = np.ldexp(np.float32(1.0), -scale_exp).astype(np.float32)
    codes = f32_to_e4m3(g * inv[:, :, None]).reshape(nrows, ncols)
    return (scale_exp + 127).astype(np.uint8), codes


def fp8_e4m3_soa_k_from_bf16(w_bf16: bytes, rows: int, cols: int) -> bytes:
    """The drafter's markov_w2: HF BF16 [rows, cols] -> transposed k-major
    [cols][rows] (dsq_generate.c:269-283: out[c*rows + r] = in[r*cols + c]),
    quantised E4M3/32 along `rows`, stored as `fp8_e4m3_soa_k` (quants_fp.c
    :285-297): the E8M0 plane [cols][rows/32] FIRST, then the E4M3 payload
    [cols][rows].  dims_ne = [rows, cols], i.e. the SOURCE order (the one
    tensor the template did not reverse).  bf16 -> f32 is exact; the E4M3
    encode is lossy by design (L213)."""
    if len(w_bf16) != rows * cols * 2:
        raise ValueError(f"fp8_e4m3_soa_k: weight holds {len(w_bf16)} bytes, [{rows}, {cols}] bf16 needs {rows * cols * 2}")
    if rows % 32:
        raise ValueError(f"fp8_e4m3_soa_k: ncols {rows} (the k-major row length) is not divisible by 32")
    bf = np.frombuffer(w_bf16, dtype="<u2").reshape(rows, cols)
    f32 = (bf.astype(np.uint32) << np.uint32(16)).view(np.float32)
    t = np.ascontiguousarray(f32.T)  # [cols][rows]
    sc, pay = _quantize_fp8_e4m3_planes(t)
    return sc.tobytes() + pay.tobytes()


# --------------------------------------------------------------------------
# The byte model: == st_bytes_for / routed_expert_side_layout.
# --------------------------------------------------------------------------

_NATIVE_DTYPE_LAYOUT = {"F32": "f32", "I32": "i32", "BF16": "bf16"}
_EXL3_K2 = {"exl3m_k2": 4, "exl3m_k2h": 5, "exl3m_k3": 6}
EXL3_HAD_BLOCK = 128


def _prod(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def bytes_for(layout: str, dims_ne: list, dtype: str | None = None) -> int:
    """Exact on-disk bytes of a declared tensor: `st_bytes_for`
    (src/engine/safetensors.cpp) for the fixed-rate layouts,
    `cutlass_mxfp4_expert_layout` (model.cpp) and `exl3_expert_layout`
    (exl3_trellis.h) for the per-expert ones.  dims_ne is the engine's ne order
    (dims_ne[0] = in = k, dims_ne[1] = out = n); a third entry multiplies
    (an expert stack).  'native' is resolved through the file dtype, as the
    engine does (`st_layout_type`)."""
    if layout == "native":
        if dtype not in _NATIVE_DTYPE_LAYOUT:
            raise ValueError(f"bytes_for: native tensor with unsupported dtype {dtype!r}")
        layout = _NATIVE_DTYPE_LAYOUT[dtype]
    if not dims_ne or any(int(d) <= 0 for d in dims_ne):
        raise ValueError(f"bytes_for: {layout} has a bad shape {dims_ne}")
    elements = _prod(dims_ne)
    if layout in ("f32", "i32"):
        return elements * 4
    if layout == "bf16":
        return elements * 2
    if layout == "iq2_xxs_mmq_k":
        if len(dims_ne) < 2 or elements % 256:
            raise ValueError(f"bytes_for: IQ2 tensor has a bad shape {dims_ne}")
        return elements // 256 * 66
    if layout in ("mxfp8_lt", "fp8_e4m3_soa_k"):
        if len(dims_ne) < 2:
            raise ValueError(f"bytes_for: MXFP8 tensor has a bad shape {dims_ne}")
        cols, rows = int(dims_ne[0]), int(dims_ne[1])
        if cols % 32:
            raise ValueError(f"bytes_for: MXFP8 tensor has a non-multiple-of-32 input dim {cols}")
        rest = _prod(dims_ne[2:])
        kb_pad = _rup(cols // 32, 4)
        if layout == "mxfp8_lt":
            return (rows * cols + _rup(rows, 128) * kb_pad) * rest
        return (rows * cols + rows * (cols // 32)) * rest
    if layout == "cutlass_mxfp4":
        if len(dims_ne) < 2:
            raise ValueError("bytes_for: cutlass_mxfp4 tensor must be at least 2D")
        k, n = int(dims_ne[0]), int(dims_ne[1])
        if k % 32:
            raise ValueError(f"bytes_for: cutlass_mxfp4 k={k} is not divisible by 32")
        per_expert = n * k // 2 + (_rup(n, 128) // 32) * _rup(k, 128)
        return per_expert * _prod(dims_ne[2:])
    if layout in _EXL3_K2:
        if len(dims_ne) < 2:
            raise ValueError(f"bytes_for: {layout} tensor must be at least 2D")
        k, n = int(dims_ne[0]), int(dims_ne[1])
        if k % EXL3_HAD_BLOCK or n % EXL3_HAD_BLOCK:
            raise ValueError(f"bytes_for: {layout} refuses dims [{k}, {n}] (both must be multiples of {EXL3_HAD_BLOCK})")
        k2 = _EXL3_K2[layout]
        words = 16 * (k2 >> 1) + (8 if k2 & 1 else 0)
        per_expert = (k // 16) * (n // 16) * words * 2 + (k + n) * 2
        return per_expert * _prod(dims_ne[2:])
    raise ValueError(f"bytes_for: unknown layout {layout!r}")

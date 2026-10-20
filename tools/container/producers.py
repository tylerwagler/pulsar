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
  * nvfp4 (ModelOpt W4A16_NVFP4 Linears, PULSAR_TENSOR_NVFP4): the nibbles verbatim, the E4M3 scales through the
    same 128x4 tile swizzle as the E8M0 planes, the global scale; byte model `st_bytes_for`.  (L299) from a BF16
    source: `nvfp4_from_bf16`, ModelOpt's max-based recipe (global amax / (6 x 448), per-16 E4M3 RNE of the block
    amax / 6, E2M1 RNE of the scaled values), then the same packer.

Every producer is graded byte-identical against the served source-precision
repack by tools/container/test_producers.py.  numpy only; no torch.
"""

from __future__ import annotations

import numpy as np

import exl3_rates

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


# The nvfp4 entry's tail: the f32 global scale, then zeros to NVFP4_TAIL bytes -- a multiple of 32, so an nvfp4 entry
# (out*in/2 and rup(out,128)*in/16 are multiples of 32 when in % 64 == 0) keeps the container's 32-byte order-based
# alignment (build.py ALIGN) when several sit back to back.
NVFP4_TAIL = 32


def nvfp4(w_u8: bytes, scale_e4m3: bytes, scale2_f32: bytes, out: int, inp: int) -> bytes:
    """A ModelOpt NVFP4 Linear (W4A16_NVFP4: U8 [out, in/2] E2M1 nibbles, low nibble = even input; F8_E4M3
    [out, in/16] block scales, row-major; F32 [] global scale; w = e2m1 * e4m3 * scale_2) -> the engine's `nvfp4`
    entry: the nibbles VERBATIM (out*in/2), the E4M3 scale plane in the 128x4-tiled MMA layout (`swizzle_sf`, i.e.
    pulsar_mx_sfoff with kbp = in/16; rows zero-padded to rup(out,128)), then the f32 global scale and zeros to
    NVFP4_TAIL bytes.  Requires in % 64 == 0 (whole 4-block scale tiles, no K padding)."""
    if inp % 64:
        raise ValueError(f"nvfp4: in={inp} is not divisible by 64")
    if len(w_u8) != out * inp // 2:
        raise ValueError(f"nvfp4: weight holds {len(w_u8)} bytes, [out={out}, in={inp}] needs {out * inp // 2}")
    if len(scale_e4m3) != out * (inp // 16):
        raise ValueError(f"nvfp4: weight_scale holds {len(scale_e4m3)} bytes, expected {out * (inp // 16)}")
    if len(scale2_f32) != 4:
        raise ValueError(f"nvfp4: weight_scale_2 holds {len(scale2_f32)} bytes, expected one f32")
    if not np.isfinite(np.frombuffer(scale2_f32, dtype="<f4")).all():
        raise ValueError("nvfp4: weight_scale_2 is not finite")
    scale = np.frombuffer(scale_e4m3, dtype=np.uint8).reshape(out, inp // 16)
    return bytes(w_u8) + swizzle_sf(scale, out, inp // 16).tobytes() + bytes(scale2_f32) + bytes(NVFP4_TAIL - 4)


E2M1_VALUE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_MID = (E2M1_VALUE[1:] + E2M1_VALUE[:-1]) / 2


def f32_to_e2m1(x: np.ndarray) -> np.ndarray:
    """|x| <= 6 (the caller's scaling) -> E2M1 codes (bit 3 the sign, bits 0-2 the magnitude's index in E2M1_VALUE),
    round to nearest, a midpoint to the even code."""
    a = np.abs(x).astype(np.float32)
    if (a > 6.0).any() or np.isnan(a).any():
        raise ValueError("f32_to_e2m1: a value past 6 or NaN -- the block scale did not bound it")
    idx = np.searchsorted(_E2M1_MID, a, side="left")          # a <= mid[idx]: the lower neighbour idx
    tie = (idx < 7) & (a == _E2M1_MID[np.minimum(idx, 6)])
    idx = np.where(tie & (idx % 2 == 1), idx + 1, idx)
    return (np.where(x < 0, 8, 0) | idx).astype(np.uint8)


def nvfp4_from_bf16(w_bf16: bytes, out: int, inp: int) -> bytes:
    """A BF16 [out, in] Linear -> the `nvfp4` entry (L299: our own quantisation, the DFlash2 drafter's Linears):
    s2 = amax(|w|) / (6 * 448) (f32; 1 for an all-zero weight), each 16-wide block's scale the E4M3 RNE of
    amax_block / (6 * s2), each value the E2M1 RNE of w / (scale * s2) (0 for a zero-scale block), packed by
    `nvfp4` (low nibble = even input).  The engine reads w = e2m1 * e4m3 * s2."""
    if inp % 64:
        raise ValueError(f"nvfp4_from_bf16: in={inp} is not divisible by 64")
    if len(w_bf16) != out * inp * 2:
        raise ValueError(f"nvfp4_from_bf16: weight holds {len(w_bf16)} bytes, [{out}, {inp}] bf16 needs {out * inp * 2}")
    w = (np.frombuffer(w_bf16, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(out, inp)
    if not np.isfinite(w).all():
        raise ValueError("nvfp4_from_bf16: a non-finite weight; the source tensor is corrupt")
    amax = np.float32(np.abs(w).max())
    s2 = np.float32(amax / np.float32(6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    blk = w.reshape(out, inp // 16, 16)
    bmax = np.abs(blk).max(axis=2)
    sc_code = f32_to_e4m3((bmax / (np.float32(6.0) * s2)).astype(np.float32))
    den = (E4M3_VALUE[sc_code] * s2).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(den[:, :, None] > 0, blk / den[:, :, None], np.float32(0.0)).astype(np.float32)
    code = f32_to_e2m1(np.clip(x, -6.0, 6.0)).reshape(out, inp)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return nvfp4(packed.tobytes(), sc_code.tobytes(), np.float32(s2).astype("<f4").tobytes(), out, inp)


def nvfp4_decode(buf: bytes, out: int, inp: int) -> np.ndarray:
    """An `nvfp4` entry -> its f32 [out, in] values (e2m1 * e4m3 * scale_2): the grading inverse."""
    vals = np.frombuffer(buf, dtype=np.uint8, count=out * inp // 2).reshape(out, inp // 2)
    kb_n = inp // 16
    sf_n = _rup(out, 128) * kb_n
    sf = unswizzle_sf(np.frombuffer(buf, dtype=np.uint8, count=sf_n, offset=out * inp // 2), out, kb_n)
    s2 = np.frombuffer(buf, dtype="<f4", count=1, offset=out * inp // 2 + sf_n)[0]
    code = np.empty((out, inp), dtype=np.uint8)
    code[:, 0::2] = vals & 15
    code[:, 1::2] = vals >> 4
    e2 = np.where(code & 8, -1.0, 1.0).astype(np.float32) * E2M1_VALUE[code & 7]
    return e2 * (E4M3_VALUE[sf].astype(np.float32) * np.float32(s2)).repeat(16, axis=1)


def rows_gather(buf: bytes, row_bytes: int, rows: list) -> bytes:
    """Rows `rows` (in that order) of a row-major [n][row_bytes] buffer, verbatim -- a head over a subset of the
    vocabulary read out of the full head (L299, the drafter's draft-vocabulary head).  Not a format conversion: the
    family's source view names the subset as a tensor of its own."""
    if row_bytes <= 0 or len(buf) % row_bytes:
        raise ValueError(f"rows_gather: {len(buf)} bytes are not whole {row_bytes}-byte rows")
    a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, row_bytes)
    ix = np.asarray(rows, dtype=np.int64)
    if ix.size == 0 or ix.min() < 0 or ix.max() >= a.shape[0]:
        raise ValueError(f"rows_gather: rows outside [0, {a.shape[0]})")
    return a[ix].tobytes()


def i32_values(values: list) -> bytes:
    """A small I32 table the family's view states itself (L299: the drafter head's vocabulary ids), little-endian."""
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0 or v.min() < -(1 << 31) or v.max() >= (1 << 31):
        raise ValueError("i32_values: empty, or a value outside int32")
    return v.astype("<i4").tobytes()


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


def mxfp8_lt_from_bf16(w_bf16: bytes, out: int, inp: int) -> bytes:
    """HF BF16 [out, in] -> the engine's `mxfp8_lt` bytes, quantised by the same codec as every other E4M3
    weight the builder writes (`_quantize_fp8_e4m3_planes` == ds4q_quantize_fp8_e4m3: per-32 group along `in`,
    E8M0 = floor(log2 amax) - 7, so the group max lands in [128, 256) and nothing saturates; E4M3 RNE).
    Layout: E4M3 codes row-major [out][in], then the swizzled E8M0 plane zero-padded to rup(out,128) x
    rup(in/32,4) -- the padding `bytes_for` models.  Unlike the FP8-source `mxfp8_lt`, the 128-alignment is
    NOT required of the data plane (Qwen's gated-residual low-rank [320, 10240] / [10240, 320] and DeltaNet
    a/b [48, 2560] are not tiled by 128); only in % 32 is.

    The L251 full-model KL graded these tensors with `qwen_screen.mxfp8_qdq` (floor(log2 amax) - 8, clamp at
    448): one binade lower, so a group whose max is in [1.75, 2) x 2^e saturates there.  Ours is pulsar's
    weight codec -- one authority for E4M3 weights; the difference is recorded in the S6 notes."""
    if inp % 32:
        raise ValueError(f"mxfp8_lt_from_bf16: in={inp} is not divisible by 32")
    if len(w_bf16) != out * inp * 2:
        raise ValueError(f"mxfp8_lt_from_bf16: weight holds {len(w_bf16)} bytes, [{out}, {inp}] bf16 needs {out * inp * 2}")
    bf = np.frombuffer(w_bf16, dtype="<u2").reshape(out, inp)
    f32 = (bf.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(f32).all():
        raise ValueError("mxfp8_lt_from_bf16: non-finite weight in the source")
    sc, codes = _quantize_fp8_e4m3_planes(f32)
    return codes.tobytes() + swizzle_sf(sc, out, inp // 32).tobytes()


def mxfp8_lt_decode(buf: bytes, out: int, inp: int) -> np.ndarray:
    """The f32 [out, in] values an `mxfp8_lt` tensor holds (E4M3 x 2^(E8M0 - 127)); the round-trip check's decoder."""
    kb_n = inp // 32
    codes = np.frombuffer(buf, dtype=np.uint8, count=out * inp).reshape(out, inp)
    sf = np.frombuffer(buf, dtype=np.uint8, offset=out * inp)
    scale = unswizzle_sf(sf, out, kb_n).astype(np.int32) - 127
    v = E4M3_VALUE[codes].reshape(out, kb_n, 32)
    return np.ldexp(v, scale[:, :, None]).reshape(out, inp).astype(np.float32)


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
# A plan entry names its producer as data -- {'producer', 'inputs', 'args'}: the producer's name, the source
# tensors whose raw bytes are its positional inputs, and its keyword arguments -- so a plan can be written out
# (build.py plan --dump) and two plans compared without producing a byte (L279).
# --------------------------------------------------------------------------

def tessera_planes(*_a, **_k) -> bytes:
    """(Tessera bundle -> the kernel-ready planes the Qwen MoE's Tessera arm binds): the slot is the table's; the
    producer is not on dev.  It needs L255's tools/tessera/planes.py (fused_stack + stack_planes: Tessera's own
    load-time prep, torch + a CUDA device) and the engine binding (family_qwen.h PULSAR_QWEN_TESS_*)."""
    raise SystemExit("tessera: the tessera producer lands with L255 (tools/tessera/planes.py; the engine's "
                     "PULSAR_QWEN_TESS_* binding) -- not on dev; refusing")


def rows_strided(buf: bytes, n_rows: int, row_bytes: int, start: int, step: int) -> bytes:
    """Rows start, start + step, ... of a row-major [n_rows][row_bytes] buffer, verbatim: one half of a
    row-interleaved fused gate/up tensor (gate on the even rows, up on the odd -- gpt-oss, Inkling's w13).  Not a
    format conversion: a family's source view (inkling.Source) names each half as a tensor of its own, and this is
    how that tensor's bytes are read out of the checkpoint."""
    if len(buf) != n_rows * row_bytes or not 0 <= start < step or n_rows % step:
        raise ValueError(f"rows_strided: {len(buf)} bytes as {n_rows} x {row_bytes}, rows {start}::{step}")
    return np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, row_bytes)[start::step].tobytes()


PRODUCERS = {f.__name__: f for f in (mxfp8_lt, mxfp8_lt_from_bf16, cutlass_mxfp4, nvfp4, nvfp4_from_bf16,
                                      fp8_e4m3_soa_k_from_bf16, i64_to_i32, tessera_planes, rows_strided, rows_gather,
                                      i32_values)}
# the engine's native layouts, by the stored dtype (`st_layout_type`): the recipe's `native` format
NATIVE_LAYOUT = {"BF16": "bf16", "F32": "f32", "I32": "i32"}

# The ONE producer table (L279 step 1): (source kind, target format) -> the producer that writes it, COPY for the
# source's own bytes verbatim.  A source kind is the stored dtype, `+scale` with a block-scale companion, `+nvfp4`
# with ModelOpt NVFP4's two (`X.weight_scale` E4M3 + `X.weight_scale_2` F32 beside `X.weight` U8), `exl3`
# (an exllamav3 Linear's trellis | suh | svh) or `tessera` (a Tessera bundle's unit blobs).  Both families' plans ask here; a pair that is not a row refuses by
# name (producer_for) -- there is no nearest format.
COPY = "copy"
PRODUCER_FOR = {
    ("BF16", "bf16"): COPY,
    ("F32", "f32"): COPY,
    ("I32", "i32"): COPY,
    ("I64", "i32"): "i64_to_i32",
    ("F8_E4M3+scale", "mxfp8_lt"): "mxfp8_lt",
    ("BF16", "mxfp8_lt"): "mxfp8_lt_from_bf16",
    ("BF16", "fp8_e4m3_soa_k"): "fp8_e4m3_soa_k_from_bf16",
    ("I8+scale", "cutlass_mxfp4"): "cutlass_mxfp4",
    ("U8+nvfp4", "nvfp4"): "nvfp4",
    ("BF16", "nvfp4"): "nvfp4_from_bf16",
    **{("exl3", layout): COPY for layout in exl3_rates.K2},
    ("tessera", "tessera"): "tessera_planes",
}


def source_kind(dtype: str, companion: str | None) -> str:
    """`companion`: None, "scale" (an FP8 / FP4 block scale `X.scale`) or "nvfp4" (ModelOpt's two NVFP4 scales)."""
    return f"{dtype}+{companion}" if companion else dtype


def producer_for(kind: str, fmt: str, what: str) -> str:
    """The producer that writes `fmt` from a `kind` source, or a refusal naming the pair."""
    prod = PRODUCER_FOR.get((kind, fmt))
    if prod is None:
        have = sorted(f for k, f in PRODUCER_FOR if k == kind)
        raise SystemExit(f"{what}: no producer writes {fmt} from a {kind} source (from {kind}: {have or 'nothing'})")
    return prod


def matrix_dims(kind: str, shape: list) -> tuple:
    """(out, in) of a source matrix: the FP4 sources store two E2M1 nibbles per byte (I8 MXFP4, U8 NVFP4)."""
    out, cols = shape
    return out, cols * 2 if kind in ("I8+scale", "U8+nvfp4") else cols


def spec(producer: str, inputs: list, **args) -> dict:
    """The serializable producer descriptor of one entry."""
    if producer not in PRODUCERS:
        raise ValueError(f"{producer}: not a producer ({sorted(PRODUCERS)})")
    return {"producer": producer, "inputs": list(inputs), "args": args}


def produce(desc: dict, source) -> bytes:
    """Run a descriptor against `source` (anything with .raw(name) -> bytes, the HF checkpoint)."""
    return PRODUCERS[desc["producer"]](*[source.raw(n) for n in desc["inputs"]], **desc["args"])


# --------------------------------------------------------------------------
# The byte model: == st_bytes_for / routed_expert_side_layout.
# --------------------------------------------------------------------------

# twice the rate K (exl3m_k2h is K = 2.5).  k4 / k5 are the Qwen lane's rates (L251: routed experts K4,
# EXL3-able dense Linears K5); k6 / k8 are turboderp's (L266).  Every rate has the same [trellis | suh | svh] byte model.
_EXL3_K2 = exl3_rates.K2
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
        if dtype not in NATIVE_LAYOUT:
            raise ValueError(f"bytes_for: native tensor with unsupported dtype {dtype!r}")
        layout = NATIVE_LAYOUT[dtype]
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
    if layout == "nvfp4":
        if len(dims_ne) != 2:
            raise ValueError(f"bytes_for: nvfp4 is a 2-D [in, out] Linear, got {dims_ne}")
        k, n = int(dims_ne[0]), int(dims_ne[1])
        if k % 64:
            raise ValueError(f"bytes_for: nvfp4 in={k} is not divisible by 64")
        return n * k // 2 + _rup(n, 128) * (k // 16) + NVFP4_TAIL
    if layout in _EXL3_K2:
        if len(dims_ne) < 2:
            raise ValueError(f"bytes_for: {layout} tensor must be at least 2D")
        k, n = int(dims_ne[0]), int(dims_ne[1])
        if len(dims_ne) == 2:   # a dense Linear: its logical out, stored padded (exl3_linear_layout; L295 F7)
            n = exl3_rates.out_stored(n)
        if k % EXL3_HAD_BLOCK or n % EXL3_HAD_BLOCK:
            raise ValueError(f"bytes_for: {layout} refuses dims [{k}, {n}] (both must be multiples of {EXL3_HAD_BLOCK})")
        words = exl3_rates.words_for(layout)
        per_expert = (k // 16) * (n // 16) * words * 2 + (k + n) * 2
        return per_expert * _prod(dims_ne[2:])
    raise ValueError(f"bytes_for: unknown layout {layout!r}")

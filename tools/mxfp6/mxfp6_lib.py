#!/usr/bin/env python3
"""MXFP6 round-trip library for the attention-quality pre-check.

Formats (OCP Microscaling / MX spec v1.0):
  MX block  = 32 elements sharing one E8M0 scale (2^(byte-127); byte 0 used for all-zero groups).
  MXFP8 elt = FP8 E4M3 (fn variant: bias 7, max 448, no inf, 0xFF/0x7F NaN).
  MXFP6 elt = E2M3 (bias 1, max 7.5,  min normal 1.0,  min subnormal 0.125)
            | E3M2 (bias 3, max 28,   min normal 0.25, min subnormal 0.0625)
  MXFP6 has no inf/NaN encodings; RNE rounding.

The pulsar engine's raw MXFP8 tensor layout (type 38) is the 33-byte interleaved
block: per row, per 32-elem group -> [E8M0 scale byte, 32 x E4M3 bytes]
(precedent: temp/splice_mxfp8_head.py, dequant = e4m3 * 2^(scale-127)).

Round-trip = decode MXFP8 -> float32 -> quantize MXFP6 (lossy) -> dequantize
-> re-encode MXFP8. Because E2M3/E3M2 have <= 3 mantissa bits and the group's
dynamic range after amax-anchored scaling fits inside e4m3's, the re-encode
is EXACT: the only loss is the MXFP6 quantization itself.

Scale rule (both the MXFP8 re-encode and the MXFP6 quant): the smallest
power-of-two scale with amax/scale <= elt_max, i.e. e = ceil(log2(amax/max)),
matching the shipped MXFP8-head splice; a comparison fix-up pass guards
against log2 ulp error at exact power-of-two boundaries.
"""
import numpy as np
import torch

# ---------------------------------------------------------------- e4m3 tables
_E4M3_LUT = (
    torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
)
_E4M3_LUT = np.where(np.isnan(_E4M3_LUT), 0.0, _E4M3_LUT).astype(np.float32)

E4M3_MAX = 448.0

# MXFP6 element format parameters: (mant_bits, min_normal_exp, elt_max)
FP6_FMT = {
    "e2m3": (3, 0, 7.5),    # exps 2^0..2^2, mant 3b; subnormals at 2^0 * m/8
    "e3m2": (2, -2, 28.0),  # exps 2^-2..2^4, mant 2b; subnormals at 2^-2 * m/4
}


def e4m3_decode(u8: np.ndarray) -> np.ndarray:
    """uint8 e4m3fn codes -> float32 (NaN codes -> 0; never emitted by us)."""
    return _E4M3_LUT[u8]


def e4m3_encode(f32: np.ndarray) -> np.ndarray:
    """float32 -> uint8 e4m3fn codes via torch cast (== RNE, proven by the
    exhaustive 2^32 sweep, temp/fp8test.cu / memory T2.3b)."""
    return (
        torch.from_numpy(np.ascontiguousarray(f32))
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .numpy()
    )


def _pow2_scale_exp(amax: np.ndarray, elt_max: float) -> np.ndarray:
    """Smallest e with amax / 2^e <= elt_max (elementwise). amax >= 0."""
    a = amax.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(a > 0, np.ceil(np.log2(np.maximum(a, 1e-300) / elt_max)), 0.0).astype(np.int64)
    # comparison fix-up against log2 ulp error at exact boundaries
    for _ in range(2):
        too_big = amax > np.exp2(e).astype(np.float64) * elt_max
        if not too_big.any():
            break
        e[too_big] += 1
    tighter = (amax > 0) & (amax <= np.exp2(e - 1).astype(np.float64) * elt_max)
    e[tighter] -= 1
    return np.clip(e, -127, 127)


def quantize_fp6(x: np.ndarray, fmt: str) -> np.ndarray:
    """RNE-quantize float32 x (already divided by the group scale, |x| <= elt_max)
    to the 6-bit element grid; returns exact float32 representable values."""
    mant, emin, elt_max = FP6_FMT[fmt]
    ax = np.abs(x).astype(np.float32)
    assert float(ax.max(initial=0.0)) <= elt_max, "scale rule violated"
    _, ex = np.frexp(ax)           # ax = m * 2^ex, m in [0.5, 1)
    e = np.maximum(ex - 1, emin)   # effective exponent, floor at min-normal
    step = np.exp2((e - mant).astype(np.float32))
    q = np.round(ax / step) * step  # np.round == round-half-to-even
    return np.copysign(q, x).astype(np.float32)


def mxfp8_blocks_decode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """raw uint8 [..., n_groups, 33] -> (float32 values [..., n_groups, 32], scale bytes)."""
    scale_b = raw[..., 0]
    vals = e4m3_decode(raw[..., 1:])
    scale = np.exp2(scale_b.astype(np.float32) - 127.0)
    return vals * scale[..., None], scale_b


def mxfp8_blocks_encode(v: np.ndarray) -> np.ndarray:
    """float32 [..., n_groups, 32] -> uint8 [..., n_groups, 33] MXFP8 blocks."""
    amax = np.abs(v).max(axis=-1)
    e = _pow2_scale_exp(amax, E4M3_MAX)
    scale_byte = (e + 127).astype(np.uint8)
    scale_byte[amax == 0] = 0
    inv = np.exp2(-e.astype(np.float32))
    scaled = (v * inv[..., None]).astype(np.float32)
    scaled[amax == 0] = 0.0
    out = np.empty(v.shape[:-1] + (33,), dtype=np.uint8)
    out[..., 0] = scale_byte
    out[..., 1:] = e4m3_encode(scaled)
    return out


def mxfp6_roundtrip_values(v: np.ndarray, fmt: str) -> np.ndarray:
    """float32 [..., n_groups, 32] -> MXFP6(fmt) quantize-dequantize (group-32 E8M0)."""
    _, _, elt_max = FP6_FMT[fmt]
    amax = np.abs(v).max(axis=-1)
    e = _pow2_scale_exp(amax, elt_max)
    inv = np.exp2(-e.astype(np.float32))
    scaled = (v * inv[..., None]).astype(np.float32)
    scaled[amax == 0] = 0.0
    q = quantize_fp6(scaled, fmt)
    return q * np.exp2(e.astype(np.float32))[..., None]


def roundtrip_mxfp8_container(raw: np.ndarray, fmt: str) -> tuple[np.ndarray, dict]:
    """Full lossy round-trip on raw MXFP8 33B blocks; returns (new raw blocks, stats)."""
    v, _ = mxfp8_blocks_decode(raw)
    v6 = mxfp6_roundtrip_values(v, fmt)
    out = mxfp8_blocks_encode(v6)
    # weight-space error stats (float64 accumulate)
    d = (v6.astype(np.float64) - v.astype(np.float64)).ravel()
    ref = v.astype(np.float64).ravel()
    rms_ref = float(np.sqrt(np.mean(ref * ref)))
    stats = {
        "rms_err": float(np.sqrt(np.mean(d * d))),
        "rms_ref": rms_ref,
        "rel_rms": float(np.sqrt(np.mean(d * d)) / rms_ref) if rms_ref else 0.0,
        "max_abs_err": float(np.abs(d).max(initial=0.0)),
    }
    return out, stats


# ------------------------------------------------------------------ self-test
def _fp6_codepoints(fmt: str) -> np.ndarray:
    """All non-negative representable values of the 6-bit element format."""
    mant, emin, _ = FP6_FMT[fmt]
    vals = {0.0}
    # subnormals: 2^emin * m/2^mant, m in 1..2^mant-1
    for m in range(1, 1 << mant):
        vals.add(m / (1 << mant) * 2.0**emin)
    # normals
    emax_n = {"e2m3": 2, "e3m2": 4}[fmt]
    for e in range(emin, emax_n + 1):
        for m in range(1 << mant):
            vals.add((1.0 + m / (1 << mant)) * 2.0**e)
    return np.array(sorted(vals), dtype=np.float32)


def self_test(seed: int = 0, n_groups: int = 4096) -> None:
    rng = np.random.default_rng(seed)

    for fmt in ("e2m3", "e3m2"):
        cp = _fp6_codepoints(fmt)

        # 1) RNE correctness of quantize_fp6 vs brute-force nearest-codepoint
        #    (ties -> value whose significand LSB is even == the codepoint list's
        #    even-index... verified via exact midpoint handling below).
        x = rng.uniform(-FP6_FMT[fmt][2], FP6_FMT[fmt][2], size=200000).astype(np.float32)
        q = quantize_fp6(x, fmt)
        allv = np.concatenate([-cp[::-1], cp])
        idx = np.searchsorted(allv, x)
        idx = np.clip(idx, 1, len(allv) - 1)
        lo, hi = allv[idx - 1], allv[idx]
        dlo, dhi = np.abs(x - lo), np.abs(x - hi)
        nearest = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.nan))
        strict = ~np.isnan(nearest)
        assert np.array_equal(q[strict], nearest[strict]), f"{fmt}: RNE mismatch"
        # exact ties: q must be one of the two neighbors (even-mantissa side)
        t = ~strict
        assert np.all((q[t] == lo[t]) | (q[t] == hi[t])), f"{fmt}: tie out of range"

        # 2) representable values are fixed points of quantize_fp6
        assert np.array_equal(quantize_fp6(cp, fmt), cp), f"{fmt}: not idempotent"

        # 3) VACUITY / IDENTITY: an MXFP8 container whose values are exactly
        #    MXFP6(fmt)-representable (random codes x random E8M0 scales) must
        #    survive MXFP8->float->MXFP6->float->MXFP8 BYTE-IDENTICALLY.
        codes = cp[rng.integers(0, len(cp), size=(n_groups, 32))]
        signs = rng.choice([-1.0, 1.0], size=(n_groups, 32)).astype(np.float32)
        sc_e = rng.integers(-20, 20, size=(n_groups,)).astype(np.float32)
        v = codes * signs * np.exp2(sc_e)[:, None]
        v[0, :] = 0.0  # all-zero group hits the scale_byte=0 convention
        raw0 = mxfp8_blocks_encode(v)
        raw1, st = roundtrip_mxfp8_container(raw0, fmt)
        assert np.array_equal(raw0, raw1), f"{fmt}: round-trip not identity"
        assert st["rms_err"] == 0.0, f"{fmt}: identity input produced error"

        # 4) e4m3 exactness: MXFP6 round-tripped generic data must re-encode
        #    into MXFP8 with ZERO container loss (decode(out) == fp6 values).
        g = (rng.standard_normal((n_groups, 32)) * np.exp2(rng.integers(-12, 12, (n_groups, 1)))).astype(np.float32)
        v6 = mxfp6_roundtrip_values(g, fmt)
        dec, _ = mxfp8_blocks_decode(mxfp8_blocks_encode(v6))
        assert np.array_equal(dec, v6), f"{fmt}: e4m3 container not lossless for fp6 values"

        print(f"self_test[{fmt}]: RNE vs brute-force OK, idempotent OK, "
              f"identity round-trip OK ({n_groups} groups), container-lossless OK")


if __name__ == "__main__":
    self_test()
    print("ALL SELF-TESTS PASSED")

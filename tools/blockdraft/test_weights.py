"""test_weights.py -- weights.py's decoders against their authorities (no model needed):
  * EXL3: every tests/test-vectors/exl3/*.exl3v block (exllamav3's own reconstruct on a GPU): W_hat byte-exact, the
    rotated weight within exllamav3's fused-kernel tolerance (2e-3 of the block's max, exl3_dequant_gate.cpp); the
    mul1 table against codebook-mul1.bin, all 65536 states
  * NVFP4: nvfp4_planes x s2 == producers.nvfp4_decode on a produced entry, and the planes' scale unswizzle against the
    scalar mx_sfoff

    python3 tools/blockdraft/test_weights.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import weights as W  # noqa: E402

VEC = os.path.join(HERE, "..", "..", "tests", "test-vectors", "exl3")
fails = 0


def check(ok, what):
    global fails
    print(("  ok   " if ok else "  FAIL ") + what)
    fails += 0 if ok else 1


def exl3_vectors():
    cb = np.fromfile(os.path.join(VEC, "codebook-mul1.bin"), dtype=np.uint16)
    check(cb.size == 65536 and np.array_equal(W.MUL1.view(np.uint16), cb), "mul1 codebook: 65536 states byte-exact")
    names = sorted(f for f in os.listdir(VEC) if f.endswith(".exl3v"))
    for n in names:
        b = open(os.path.join(VEC, n), "rb").read()
        assert b[:8] == b"EXL3VEC1"
        kt, nt, words, k2, flags, _r = np.frombuffer(b, dtype="<u4", count=6, offset=8)
        rows, cols = 16 * kt, 16 * nt
        p = np.frombuffer(b, dtype="<u2", offset=32)
        o = 0
        tr = p[o:o + kt * nt * words].view(np.int16).reshape(kt, nt, words); o += kt * nt * words
        suh = p[o:o + rows].view(np.float16); o += rows
        svh = p[o:o + cols].view(np.float16); o += cols
        what = p[o:o + rows * cols].reshape(rows, cols); o += rows * cols
        fused = p[o:o + rows * cols].view(np.float16).astype(np.float64).reshape(rows, cols)
        mine = W.exl3_what(tr)
        check(np.array_equal(mine.view(np.uint16), what), f"{n}: W_hat byte-exact (K2 {k2})")
        w = W.exl3_weight(tr, suh, svh).numpy().T.astype(np.float64)    # back to [in][out]
        err = np.abs(w - fused).max()
        check(err <= 2e-3 * np.abs(fused).max(), f"{n}: rotated weight max|d| {err:.2e} vs fused (tol "
                                                 f"{2e-3 * np.abs(fused).max():.2e})")


def nvfp4_roundtrip():
    rng = np.random.default_rng(7)
    for out, inp in ((256, 128), (200, 192), (640, 2560)):
        w = (rng.standard_normal((out, inp)) * rng.uniform(0.01, 0.2, (out, 1))).astype(np.float32)
        b = (w.view(np.uint32) >> 16).astype("<u2").tobytes()
        e = W.PR.nvfp4_from_bf16(b, out, inp)
        Wp, s2 = W.nvfp4_planes(e, out, inp)
        ref = W.PR.nvfp4_decode(e, out, inp)
        exact = Wp.numpy().astype(np.float64) * s2          # e2m1 x e4m3 is exact in f32; one rounding left
        rel = np.abs(ref - exact).max() / np.abs(exact).max()
        check(rel <= 2.0 ** -23, f"nvfp4 [{out}, {inp}]: planes x s2 == nvfp4_decode to f32 rounding (rel {rel:.1e})")
        # the scale plane through the scalar swizzle (pulsar_mx_sfoff), not the transpose
        kb_n = inp // 16
        sf = np.frombuffer(e, dtype=np.uint8, offset=out * inp // 2, count=W.PR._rup(out, 128) * kb_n)
        r, kb = np.meshgrid(np.arange(out), np.arange(kb_n), indexing="ij")
        scal = sf[W.PR.mx_sfoff(r, kb, kb_n)]
        e4 = W.PR.E4M3_VALUE[scal].astype(np.float32).repeat(16, axis=1)
        dec = Wp.numpy() / np.where(e4 > 0, e4, 1)
        check(np.isin(np.abs(dec[e4 > 0]), W.PR.E2M1_VALUE).all() and (Wp.numpy()[e4 == 0] == 0).all(),
              f"nvfp4 [{out}, {inp}]: every value = e2m1 x the mx_sfoff-addressed scale")


exl3_vectors()
nvfp4_roundtrip()
print("FAIL" if fails else "PASS", f"({fails} failure(s))")
sys.exit(1 if fails else 0)

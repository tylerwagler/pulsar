/* pulsar_cuda_libm.cuh -- the host libm's expf, exp and log, bit for bit, on the device (L302).
 *
 * The device sampler (pulsar_cuda_sampler.cu) must produce the bytes the host sampler produced -- every probability,
 * every race score, every logprob -- and those were computed with glibc's expf / exp / log.  CUDA's own expf / exp /
 * log are different approximations (and --use_fast_math swaps in __expf), so a sampler written with them draws a
 * different token whenever two race scores sit within an ulp.  These are the glibc 2.43 aarch64 routines (Arm's
 * optimized-routines) transcribed from the SHIPPED BINARY's instruction sequence, fused multiply-adds where the
 * compiler contracted them and nowhere else, over the tables read out of the same binary
 * (pulsar_cuda_libm_tables.inc).  Every operation is an explicit _rn intrinsic, so nvcc can neither contract nor
 * reassociate them; the file that includes this is built without --use_fast_math (no FTZ: a probability may be
 * subnormal, and the host keeps it).
 *
 * Domain: each routine reproduces the host on every input, special cases included (the sampler feeds expf
 * non-positive arguments, exp non-positive doubles, log uniforms in (0, 1) and sums >= 1).  The libm gate
 * (tests/sampler_device_gate.cpp --libm) checks expf over every float and exp / log over 2^28 random doubles each,
 * plus their edge classes, against the host's libm. */
#ifndef PULSAR_CUDA_LIBM_CUH
#define PULSAR_CUDA_LIBM_CUH

#include <stdint.h>

#include "pulsar_cuda_libm_tables.inc"

__device__ __forceinline__ double plm_d(uint64_t b) { return __longlong_as_double((long long)b); }
__device__ __forceinline__ uint64_t plm_u(double d) { return (uint64_t)__double_as_longlong(d); }

/* expf (glibc sysdeps/ieee754/flt-32/e_expf.c): x * 32/ln2 split into k + r in double, 2^(k/32) from the table, a
 * cubic in r.  aarch64: frinta (round half away) and fcvtas for k; three fmas. */
__device__ __forceinline__ float plm_expf(float x) {
    const uint32_t ix = __float_as_uint(x);
    const uint32_t abstop = (ix >> 20) & 0x7ffu;
    if (abstop > 0x42au) {                       /* |x| >= 88, or x is nan / inf */
        if (ix == 0xff800000u) return 0.0f;      /* -inf */
        if (abstop > 0x7f7u) return __fadd_rn(x, x);
        if (x > __uint_as_float(0x42b17217u)) return __uint_as_float(0x7f800000u);   /* overflow */
        if (x < __uint_as_float(0xc2cff1b4u)) return 0.0f;                           /* underflow */
        if (x < __uint_as_float(0xc2ce8ecfu)) return __uint_as_float(1u);            /* may underflow: 2^-149 */
    }
    const double xd = (double)x;
    const double z = __dmul_rn(xd, plm_d(PLM_EXPF_C[0]));
    const double kd = round(z);                               /* frinta */
    const long long ki = llround(z);                          /* fcvtas */
    const double r = __dsub_rn(z, kd);
    const uint64_t t = PLM_EXPF_T[ki & 31] + ((uint64_t)ki << 47);
    const double s = plm_d(t);
    const double r2 = __dmul_rn(r, r);
    const double zc = __fma_rn(plm_d(PLM_EXPF_C[1]), r, plm_d(PLM_EXPF_C[2]));
    double y = __fma_rn(plm_d(PLM_EXPF_C[3]), r, 1.0);
    y = __fma_rn(zc, r2, y);
    y = __dmul_rn(y, s);
    return __double2float_rn(y);
}

/* log (glibc sysdeps/ieee754/dbl-64/e_log.c): near 1 a degree-11 polynomial with a split r; elsewhere a 128-entry
 * (invc, logc) table and a quintic.  Subnormal and special inputs as glibc (the sampler never sends them). */
__device__ __forceinline__ double plm_log(double x) {
    uint64_t ix = plm_u(x);
    if (ix - 0x3fee000000000000ull <= 0x000308ffffffffffull) {   /* |x - 1| < ~0x1.09p-4 */
        if (ix == 0x3ff0000000000000ull) return 0.0;
        const double r = __dsub_rn(x, 1.0);
        const double r2 = __dmul_rn(r, r);
        const double r3 = __dmul_rn(r, r2);
        auto K = [](int i) { return plm_d(PLM_LOG_K[7 + i]); };   /* B[i] */
        double p1 = __fma_rn(K(2), r, K(1));
        p1 = __fma_rn(K(3), r2, p1);
        double p2 = __fma_rn(K(5), r, K(4));
        p2 = __fma_rn(K(6), r2, p2);
        double p3 = __fma_rn(K(8), r, K(7));
        p3 = __fma_rn(K(9), r2, p3);
        p3 = __fma_rn(K(10), r3, p3);
        p2 = __fma_rn(p3, r3, p2);
        p1 = __fma_rn(p2, r3, p1);
        /* rhi = r + w - w with w = r * 2^27, both contracted */
        const double t27 = 134217728.0;
        double rhi = __fma_rn(r, t27, r);
        rhi = __fma_rn(-r, t27, rhi);
        const double rhi2 = __dmul_rn(rhi, rhi);
        const double rlo = __dsub_rn(r, rhi);
        const double hi = __fma_rn(rhi2, K(0), r);
        double lo = __dsub_rn(r, hi);
        const double rs = __dadd_rn(r, rhi);
        lo = __fma_rn(rhi2, K(0), lo);
        lo = __fma_rn(__dmul_rn(K(0), rlo), rs, lo);
        const double y = __fma_rn(p1, r3, lo);
        return __dadd_rn(hi, y);
    }
    const uint32_t top = (uint32_t)(ix >> 48);
    if (top - 0x0010u >= 0x7ff0u - 0x0010u) {
        if ((ix << 1) == 0) return -__longlong_as_double(0x7ff0000000000000ll);   /* log(+-0) = -inf */
        if (ix == 0x7ff0000000000000ull) return x;                                /* log(inf) = inf */
        if ((top & 0x8000u) || (top & 0x7ff0u) == 0x7ff0u) return __longlong_as_double(0x7ff8000000000000ll);
        ix = plm_u(__dmul_rn(x, 4503599627370496.0)) - (52ull << 52);              /* subnormal: normalise */
    }
    const uint64_t tmp = ix - 0x3fe6000000000000ull;
    const uint32_t i = (uint32_t)((tmp >> 45) & 127u);
    const int k = (int)((long long)tmp >> 52);
    const uint64_t iz = ix - (tmp & 0xfff0000000000000ull);
    const double invc = plm_d(PLM_LOG_T[2 * i]), logc = plm_d(PLM_LOG_T[2 * i + 1]);
    const double z = plm_d(iz);
    const double r = __fma_rn(invc, z, -1.0);
    const double kd = (double)k;
    const double w = __fma_rn(plm_d(PLM_LOG_K[0]), kd, logc);
    const double hi = __dadd_rn(r, w);
    double lo = __dsub_rn(w, hi);
    lo = __dadd_rn(lo, r);
    lo = __fma_rn(plm_d(PLM_LOG_K[1]), kd, lo);
    const double r2 = __dmul_rn(r, r);
    const double r3 = __dmul_rn(r, r2);
    double y = __fma_rn(plm_d(PLM_LOG_K[2]), r2, lo);
    double q = __fma_rn(plm_d(PLM_LOG_K[4]), r, plm_d(PLM_LOG_K[3]));
    const double q2 = __fma_rn(plm_d(PLM_LOG_K[6]), r, plm_d(PLM_LOG_K[5]));
    q = __fma_rn(q2, r2, q);
    y = __fma_rn(r3, q, y);
    return __dadd_rn(y, hi);
}

/* exp (glibc sysdeps/ieee754/dbl-64/e_exp.c): k = round(x * 128/ln2), r = x - k ln2/128 in two fmas, a quintic, the
 * scale from the table; |x| >= 512 through the subnormal-safe special case. */
__device__ __forceinline__ double plm_exp(double x) {
    const uint64_t ux = plm_u(x);
    uint32_t abstop = (uint32_t)(ux >> 52) & 0x7ffu;
    if (abstop - 0x3c9u > 0x3eu) {
        if ((int32_t)(abstop - 0x3c9u) < 0) return __dadd_rn(1.0, x);   /* |x| < 2^-54 */
        if (abstop > 0x408u) {                                          /* |x| >= 1024 */
            if (ux == 0xfff0000000000000ull) return 0.0;
            if (abstop == 0x7ffu) return __dadd_rn(1.0, x);
            return (ux >> 63) ? 0.0 : __longlong_as_double(0x7ff0000000000000ll);
        }
        abstop = 0;                                                     /* 512 <= |x| < 1024: special case below */
    }
    const double z = __dmul_rn(x, plm_d(PLM_EXP_K[0]));
    const double kd = round(z);
    const long long ki = llround(z);
    double r = __fma_rn(plm_d(PLM_EXP_K[1]), kd, x);
    r = __fma_rn(plm_d(PLM_EXP_K[2]), kd, r);
    const uint32_t idx = (uint32_t)(ki & 127) * 2u;
    const double tail = plm_d(PLM_EXP_T[idx]);
    const uint64_t sbits = PLM_EXP_T[idx + 1] + ((uint64_t)ki << 45);
    const double c23 = __fma_rn(plm_d(PLM_EXP_K[4]), r, plm_d(PLM_EXP_K[3]));
    const double r2 = __dmul_rn(r, r);
    double t = __dadd_rn(r, tail);
    t = __fma_rn(c23, r2, t);
    const double r4 = __dmul_rn(r2, r2);
    const double c45 = __fma_rn(plm_d(PLM_EXP_K[6]), r, plm_d(PLM_EXP_K[5]));
    t = __fma_rn(r4, c45, t);
    if (abstop != 0) {
        const double scale = plm_d(sbits);
        return __fma_rn(t, scale, scale);
    }
    if (!(ki & 0x80000000ll)) {                                        /* k > 0: overflow side */
        const double scale = plm_d(sbits - (1009ull << 52));
        return __dmul_rn(__fma_rn(t, scale, scale), plm_d(0x7f00000000000000ull));
    }
    const double scale = plm_d(sbits + (1022ull << 52));
    const double st = __dmul_rn(t, scale);
    const double y = __dadd_rn(scale, st);
    if (!(y < 1.0)) return __dmul_rn(y, 2.2250738585072014e-308);
    const double hi = __dadd_rn(y, 1.0);
    double lo = __dsub_rn(scale, y);
    lo = __dadd_rn(lo, st);
    double l2 = __dsub_rn(1.0, hi);
    l2 = __dadd_rn(l2, y);
    double yy = __dadd_rn(l2, lo);
    yy = __dadd_rn(yy, hi);
    yy = __dsub_rn(yy, 1.0);
    if (yy == 0.0) return 0.0;
    return __dmul_rn(yy, 2.2250738585072014e-308);
}

#endif

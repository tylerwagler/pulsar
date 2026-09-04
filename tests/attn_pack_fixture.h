/* Host-side PULSAR_ATTN_PACK fixture helpers, shared by the attention gates.
 *
 * WHY A FIXTURE ENCODES RATHER THAN DRAWS BYTES.  When the raw KV caches moved
 * to the packed 584 B row (2026-08-17), these gates were updated to build
 * packed rows by filling the E4M3 payload with uniform random BYTES.  That is
 * not a uniform distribution of values: a uniform byte is uniform in EXPONENT,
 * so the decoded magnitudes span 2^-9..448 and are dominated by the large end.
 * A 512-wide q.k dot over such rows produces scores tens of thousands apart,
 * and the softmax collapses to one-hot.
 *
 * A one-hot softmax silently removes what these gates exist to measure:
 *   - attn_f16_kernel_test: median per-(tok,head) rel L2 fell to 4e-12 (exact
 *     agreement on a degenerate input) while a few sink-dominated heads with
 *     |ref| ~ 3e-4 reported 6.4e-2 and failed the gate on nothing but a
 *     small-norm denominator.
 *   - the split-KV merge gate (deleted with the f32 decode kernel, L166):
 *     residuals of 1.3e-49 and 0.0e+00, because when one split held all the
 *     weight there was no reassociation to measure and the softmax MERGE --
 *     the entire subject of that test -- never ran.
 *
 * Encoding a draw against the block scale fixes both, and has a useful
 * property: the decoded magnitude tracks the DRAW, not the scale byte, so the
 * scale band can still vary per block (exercising scale addressing) without
 * moving the data's magnitude.  Keep the band within about 2^-3..2^2 anyway --
 * a scale far from the data wastes E4M3's range and re-introduces quantisation
 * noise from the wrong end.
 *
 * The cost is that the fixture no longer sweeps every byte pattern.  That
 * coverage was defending a property of the decoder's bit math -- the row decode
 * is total -- which holds by inspection and does not
 * need a random walk to assert, whereas a non-degenerate softmax cannot be
 * recovered by inspection at all.
 */
#ifndef PULSAR_TESTS_ATTN_PACK_FIXTURE_H
#define PULSAR_TESTS_ATTN_PACK_FIXTURE_H

#include <cmath>
#include <cstdint>
#include <cstring>

/* Host NV row codec, mirroring the device row exactly:
 *   [n_nope/2 e2m1 nibbles][n_nope/16 e4m3 scale codes][f32 row scale]
 *   [n_rot bf16 rope]  (384 B at head_dim 512)
 * Encoders are inverse-searches of the decode tables so the two cannot
 * drift; this is fixture code and clarity beats speed. */
static inline float host_e4m3_mag(uint8_t code) {
    const uint32_t e = (code >> 3) & 15u, m = code & 7u;
    if (e == 15u && m == 7u) return NAN;
    if (e == 0u) return (float)m * 0.001953125f;
    return std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
}
static inline uint8_t host_e4m3_encode_pos(float x) {
    if (x >= 448.0f) return 126u;
    uint32_t best = 0u; float bd = std::fabs(x - host_e4m3_mag(0));
    for (uint32_t c = 1u; c <= 126u; c++) {
        const float d = std::fabs(x - host_e4m3_mag((uint8_t)c));
        if (d < bd || (d == bd && (c & 1u) == 0u && (best & 1u) != 0u)) { best = c; bd = d; }
    }
    return (uint8_t)best;
}
static inline float host_e2m1_value(uint32_t c) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[c & 7u];
}
static inline uint8_t host_e2m1_encode(float x) {
    const float ax = std::fmin(std::fabs(x), 6.0f);
    uint32_t best = 0u; float bd = std::fabs(ax - host_e2m1_value(0));
    for (uint32_t i = 1u; i < 8u; i++) {
        const float d = std::fabs(ax - host_e2m1_value(i));
        if (d < bd || (d == bd && (i & 1u) == 0u && (best & 1u) != 0u)) { best = i; bd = d; }
    }
    return (uint8_t)(best | (std::signbit(x) ? 0x8u : 0u));
}
static inline uint16_t host_bf16_rtn(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float host_bf16_widen(uint16_t bits) {
    const uint32_t u = (uint32_t)bits << 16;
    float f; std::memcpy(&f, &u, sizeof f); return f;
}

/* Pack one head_dim-wide float row into the NV layout; when `dec` is non-NULL
 * also write the DECODED values, so an oracle and the kernel look at the same
 * numbers.  Recipe mirrors attn_pack_store_kernel: per-16 amax, row scale
 * amax_row/(6*448) with the 1e-4 floor, e4m3 rtn scale codes, e2m1 rtn
 * tie-to-even data codes, bf16 rtn rope. */
static inline void host_nv_pack_row(const float *vals, uint8_t *row, float *dec,
                                    uint32_t head_dim) {
    const uint32_t n_rot = 64u, n_nope = head_dim - n_rot;
    const uint32_t nib = n_nope / 2u, nblk = n_nope / 16u;
    float ra = 0.0f;
    for (uint32_t d = 0; d < n_nope; d++) ra = std::fmax(ra, std::fabs(vals[d]));
    const float rs = std::fmax(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
    std::memcpy(row + nib + nblk, &rs, sizeof rs);
    for (uint32_t b = 0; b < nblk; b++) {
        float amax = 0.0f;
        for (uint32_t d = b * 16u; d < (b + 1u) * 16u; d++) amax = std::fmax(amax, std::fabs(vals[d]));
        const float t = std::fmin(448.0f, amax * (1.0f / 6.0f) / rs);
        row[nib + b] = host_e4m3_encode_pos(t);
        const float scale = host_e4m3_mag(row[nib + b]) * rs;
        for (uint32_t d = b * 16u; d < (b + 1u) * 16u; d += 2u) {
            const uint8_t v0 = scale > 0.0f ? host_e2m1_encode(vals[d] / scale) : 0u;
            const uint8_t v1 = scale > 0.0f ? host_e2m1_encode(vals[d + 1u] / scale) : 0u;
            row[d >> 1] = (uint8_t)(v0 | (v1 << 4));
            if (dec) {
                dec[d]      = ((v0 & 8u) ? -1.0f : 1.0f) * host_e2m1_value(v0) * scale;
                dec[d + 1u] = ((v1 & 8u) ? -1.0f : 1.0f) * host_e2m1_value(v1) * scale;
            }
        }
    }
    uint16_t *rope = (uint16_t *)(row + nib + nblk + 4u);
    for (uint32_t d = 0; d < n_rot; d++) {
        rope[d] = host_bf16_rtn(vals[n_nope + d]);
        if (dec) dec[n_nope + d] = host_bf16_widen(rope[d]);
    }
}

/* ---- Q upload in the engine's stored element type -------------------------
 *
 * Fixtures must NOT assume Q is f32.  pulsar_q_t narrowed to __half (L045), and
 * a fixture that cudaMemcpys floats into that buffer keeps compiling and
 * running while handing the kernel garbage.  The quiet variant is worse: the
 * decode kernels are TEMPLATED on the Q type, so a `float *` argument deduces
 * QT=float and the gate certifies an instantiation the engine never launches --
 * green, and measuring nothing.
 *
 * Going through here ties the fixture to pulsar_q_t by construction. */
static inline void fixture_q_set(float &d, float v)  { d = v; }
static inline void fixture_q_set(__half &d, float v) { d = __float2half(v); }

static inline pulsar_q_t *fixture_upload_q(const std::vector<float> &q) {
    pulsar_q_t *d = NULL;
    if (cudaMalloc(&d, q.size() * sizeof(pulsar_q_t)) != cudaSuccess) return NULL;
    std::vector<pulsar_q_t> h(q.size());
    for (size_t i = 0; i < q.size(); i++) fixture_q_set(h[i], q[i]);
    if (cudaMemcpy(d, h.data(), h.size() * sizeof(pulsar_q_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return NULL;
    }
    return d;
}

#endif /* PULSAR_TESTS_ATTN_PACK_FIXTURE_H */

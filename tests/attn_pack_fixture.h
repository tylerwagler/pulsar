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
 *   - attn_decode_split_test: residuals of 1.3e-49 and 0.0e+00, because when
 *     one split holds all the weight there is no reassociation to measure and
 *     the softmax MERGE -- the entire subject of that test -- is never run.
 *
 * Encoding a draw against the block scale fixes both, and has a useful
 * property: the decoded magnitude tracks the DRAW, not the scale byte, so the
 * scale band can still vary per block (exercising scale addressing) without
 * moving the data's magnitude.  Keep the band within about 2^-3..2^2 anyway --
 * a scale far from the data wastes E4M3's range and re-introduces quantisation
 * noise from the wrong end.
 *
 * The cost is that the fixture no longer sweeps every byte pattern.  That
 * coverage was defending a property of the decoder's bit math -- attn_pack_e4m3
 * is total, with no NaN encodings -- which holds by inspection and does not
 * need a random walk to assert, whereas a non-degenerate softmax cannot be
 * recovered by inspection at all.
 */
#ifndef PULSAR_TESTS_ATTN_PACK_FIXTURE_H
#define PULSAR_TESTS_ATTN_PACK_FIXTURE_H

#include <cmath>
#include <cstdint>
#include <cstring>

/* Decode only, mirroring attn_pack_e4m3 / attn_comp_pack_ld in
 * src/cuda/pulsar_cuda_internal.h; there is nothing here that can drift from
 * the device's rounding. */
static inline float host_e4m3_decode(uint8_t b, float scale) {
    const uint32_t e = (b >> 3) & 15u;
    const uint32_t m = b & 7u;
    const float v = e ? std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7)
                      : (float)m * 0.001953125f;      /* 2^-9 * m, subnormal */
    const float sv = v * scale;
    return (b & 0x80u) ? -sv : sv;
}

/* Nearest E4M3, defined as the inverse-search of the decoder above so the two
 * cannot drift.  128 magnitudes searched per element: this is fixture code and
 * clarity beats speed. */
static inline uint8_t host_e4m3_encode(float v, float scale) {
    const uint8_t sign = (v < 0.0f) ? 0x80u : 0x00u;
    const float x = std::fabs(v / scale);
    int best = 0; float bd = INFINITY;
    for (int b = 0; b < 128; b++) {
        const float d = std::fabs(host_e4m3_decode((uint8_t)b, 1.0f) - x);
        if (d < bd) { bd = d; best = b; }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

/* The E8M0 scale byte for nope element d of a row whose scale bytes are
 * already written.  Callers write scales BEFORE the payload -- the payload
 * byte is an encode against its own block scale, so the scale has to exist
 * before the value does. */
static inline float host_pack_block_scale(const uint8_t *row, uint32_t n_nope,
                                          uint32_t d, uint32_t block) {
    const uint32_t sb = (uint32_t)row[n_nope + (d / block)];
    float scale; const uint32_t su = sb << 23;
    std::memcpy(&scale, &su, sizeof scale);
    return scale;
}

static inline float host_bf16_widen(uint16_t bits) {
    const uint32_t u = (uint32_t)bits << 16;
    float f; std::memcpy(&f, &u, sizeof f); return f;
}

#endif /* PULSAR_TESTS_ATTN_PACK_FIXTURE_H */

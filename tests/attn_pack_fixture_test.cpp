/* Host check for the restored 0731 unified NVFP4 row codec.
 *
 * WHY THIS EXISTS BEFORE THE KERNEL.  The device packer for that row is the
 * highest-consequence code in the engine -- a wrong recipe silently corrupts
 * every KV cache -- and it cannot be run here (no GPU).  So the oracle lands
 * first and the kernel lands against it, which is how this tree already works
 * (gates before landings).  tests/attn_pack_fixture.h is a HOST replica of the
 * device row; this binary exercises it with no device at all.
 *
 * What it pins is the GEOMETRY and the RECIPE's arithmetic, not the kernel:
 * that the row is 384 B at head_dim 512 with the rope tail at +256, that the
 * row scale is amax/(6*448) with the 1e-4 floor, that a zero row encodes zero
 * codes and zero row scale rather than NaN, that the rope tail is exact bf16,
 * and that the whole thing is deterministic.  A device/kernel mismatch is what
 * tests/kv4_pack_gate.cpp is for, once the kernel exists.
 *
 * Build/run: make attn-pack-fixture-check */

#include "pulsar_gpu.h"          /* the row geometry macros the codec codes against */
#include "attn_pack_fixture.h"

#include <cstdio>
#include <cstring>

static int g_fail = 0;

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL  %s\n", what); g_fail++; }
    else     { std::fprintf(stderr, "ok    %s\n", what); }
}

int main(void) {
    const uint32_t HD = 512u, NROT = 64u, NNOPE = HD - NROT;
    const uint32_t NIB = NNOPE / 2u, NBLK = NNOPE / 16u;

    /* ---- geometry: the layout every reader indexes by ---- */
    check(PULSAR_ATTN_PACK_ROWBYTES(HD) == 384u, "row is 384 B at head_dim 512");
    check(PULSAR_ATTN_PACK_NROT == NROT, "n_rot is the macro's");
    check(NIB == 224u && NBLK == 28u, "224 nibble bytes + 28 scale codes");
    /* the f32 row scale must be 4-aligned inside the row, and the rope tail
     * starts 4 B after the last scale code */
    check((NIB + NBLK) % 4u == 0u, "row scale sits 4-aligned at +252");
    check((uint32_t)PULSAR_ATTN_PACK_ROWBYTES(HD) == NIB + NBLK + 4u + NROT * 2u,
          "rope tail at +256, 64 bf16 = 128 B, ends the row");

    /* ---- recipe: a known row ---- */
    static float vals[HD];
    static uint8_t row[384], row2[384];
    static float dec[HD];

    for (uint32_t d = 0; d < HD; d++) {
        /* a deterministic, non-degenerate draw; keep it away from zero so the
         * 1e-4 floor is not what is being measured */
        vals[d] = 0.75f * std::sin((float)d * 0.11f) + 0.25f * std::cos((float)d * 0.017f);
    }
    host_nv_pack_row(vals, row, dec, HD);

    /* row scale = max|nope| / (6*448); independently recomputed here */
    float amax = 0.0f;
    for (uint32_t d = 0; d < NNOPE; d++) amax = std::fmax(amax, std::fabs(vals[d]));
    float rs = 0.0f;
    std::memcpy(&rs, row + NIB + NBLK, sizeof rs);
    const float want_rs = amax * (1.0f / (6.0f * 448.0f));
    check(rs > 0.0f && std::fabs(rs - want_rs) <= 1e-6f * want_rs,
          "row scale is amax/(6*448)");

    /* the rope tail is exact bf16 of the input, untouched by the quantiser */
    {
        const uint16_t *rope = (const uint16_t *)(row + NIB + NBLK + 4u);
        bool exact = true;
        for (uint32_t d = 0; d < NROT; d++) {
            if (rope[d] != host_bf16_rtn(vals[NNOPE + d])) exact = false;
        }
        check(exact, "rope tail is exact bf16 of the input");
    }

    /* quantisation is bounded: every nope dim decodes within one block step */
    {
        float worst = 0.0f;
        for (uint32_t d = 0; d < NNOPE; d++) {
            const float err = std::fabs(dec[d] - host_bf16_widen(host_bf16_rtn(vals[d])));
            worst = std::fmax(worst, err);
        }
        /* E2M1's largest step is 6 -> 4, i.e. 1/3 of the block scale, and the
         * block scale is at most amax; so <= amax/3, and a bit of slack. */
        check(worst <= amax * 0.34f + 1e-6f, "nope dims decode within an E2M1 step");
    }

    /* determinism: the same input packs to the same bytes */
    host_nv_pack_row(vals, row2, NULL, HD);
    check(std::memcmp(row, row2, sizeof row) == 0, "packing is deterministic");

    /* ---- the zero row: codes 0, NOT NaN ---- */
    {
        static float zero[HD];
        static uint8_t zrow[384];
        std::memset(zero, 0, sizeof zero);
        host_nv_pack_row(zero, zrow, NULL, HD);
        float zrs = 0.0f;
        std::memcpy(&zrs, zrow + NIB + NBLK, sizeof zrs);
        bool codes_zero = true;
        for (uint32_t i = 0; i < NIB; i++) if (zrow[i] != 0u) codes_zero = false;
        check(codes_zero, "a zero row encodes zero nibble codes");
        check(zrs == (1.0e-4f / (6.0f * 448.0f)), "a zero row still carries the 1e-4 floor");
    }

    std::fprintf(stderr, g_fail ? "\nattn-pack fixture check: %d FAILED\n"
                                : "\nattn-pack fixture check: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}

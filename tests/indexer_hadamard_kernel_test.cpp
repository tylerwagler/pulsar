/* Device gate for V4's indexer rotation (L218 s35).
 *
 * The reference's `rotate=True` compressor runs
 *     rotate_activation(x) = hadamard_transform(x, scale = d**-0.5)
 * before the same fp4 quant, and V4's indexer runs it on BOTH its own
 * compressor's index key and its q.  L218 deleted it with the 0731 checkpoint
 * ("0731's 128-point Hadamard rotation before the quant is gone with that
 * checkpoint"), which is why V4's indexer had no key path at all.  Restoring a
 * rotation is the one kind of restore that cannot be eyeballed: a different
 * ORTHOGONAL matrix still produces plausible scores and plausible text, so the
 * convention has to be pinned, not argued.
 *
 * WHAT IS PINNED, and how:
 *   - ORDER (natural/Hadamard, not sequency).  The host oracle below builds the
 *     transform from the textbook recursion, sharing no arithmetic with the
 *     kernel, and two cases have a closed form that discriminates the order: an
 *     all-ones row must transform to [sqrt(128), 0, 0, ...] and a unit impulse
 *     to a FLAT row of 1/sqrt(128).  A sequency-ordered transform swaps which of
 *     the two is sparse.  Both are asserted AND printed.
 *   - NORMALISATION (1/sqrt(128)).  It is what sets the per-32-block absmax, so
 *     it shows up in every E8M0 scale byte.
 *   - The block-absmax -> E8M0 -> E2M1 tail: compared BYTE FOR BYTE (all 68
 *     bytes of every row, scale bytes included) against a host mirror of the
 *     reference definition plus the tree's published codec.
 *
 * The fused rope+q kernel is graded against the two-launch sequence it replaces
 * (rope_tail then this pack), byte for byte -- that is its stated contract.
 *
 * No model needed.  build+run: make indexer-hadamard-kernel-check CUDA_ARCH=sm_120f
 */
#include "pulsar_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HAD_DIM 128u
#define HAD_ROWBYTES 68u          /* 64 nibble bytes + 4 E8M0 bytes */
static const float HAD_SCALE = 0.08838834764831845f;   /* 128 ** -0.5, fp32 */

/* ---- the host oracle ---------------------------------------------------- */

/* The natural-order (Hadamard) Walsh-Hadamard transform, written as the
 * textbook recursion rather than as the kernel's index arithmetic, so a bug in
 * either is not mirrored in the other. */
static void host_hadamard(const float *x, float *v) {
    for (uint32_t i = 0; i < HAD_DIM; i++) v[i] = x[i];
    for (uint32_t half = 1; half < HAD_DIM; half <<= 1) {
        for (uint32_t base = 0; base < HAD_DIM; base += 2u * half) {
            for (uint32_t j = 0; j < half; j++) {
                const float a = v[base + j], b = v[base + half + j];
                v[base + j]        = a + b;
                v[base + half + j] = a - b;
            }
        }
    }
    for (uint32_t i = 0; i < HAD_DIM; i++) v[i] *= HAD_SCALE;
}

/* The reference's fast_round_scale: 2^ceil(log2(y)) read off y's IEEE exponent
 * field (pulsar_cuda_internal.h -- NOT log2f/ceilf, which round differently at
 * the boundaries). */
static uint32_t host_e8m0(float y) {
    uint32_t b;
    memcpy(&b, &y, sizeof b);
    const uint32_t e8 = ((b >> 23) & 0xFFu) + (((b & 0x7FFFFFu) != 0u) ? 1u : 0u);
    return e8 > 254u ? 254u : e8;
}
static float host_e8m0_scale(uint32_t b) {
    const uint32_t f = b << 23;
    float s;
    memcpy(&s, &f, sizeof s);
    return s;
}
/* The OCP E2M1 encode: nearest of {0,.5,1,1.5,2,3,4,6}, ties to the even code. */
static uint8_t host_encode(float x) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - mag[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - mag[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return (uint8_t)((best & 7) | ((x < 0.0f) ? 0x8u : 0u));
}

/* One 128-value row -> its 68-byte MXKV FP4 row: the whole reference pipeline. */
static void host_ref_row(const float *x, uint8_t *out) {
    float v[HAD_DIM];
    host_hadamard(x, v);
    uint8_t nib[HAD_DIM];
    for (uint32_t blk = 0; blk < 4u; blk++) {
        float amax = 7.052966104933725e-38f;   /* the kernel's 6 * 2^-126 floor */
        for (uint32_t i = 0; i < 32u; i++) amax = fmaxf(amax, fabsf(v[blk * 32u + i]));
        const uint32_t e8 = host_e8m0(amax * (1.0f / 6.0f));
        const float scale = host_e8m0_scale(e8);
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t d = blk * 32u + i;
            nib[d] = host_encode(fminf(6.0f, fmaxf(-6.0f, v[d] / scale)));
        }
        out[64u + blk] = (uint8_t)e8;
    }
    for (uint32_t i = 0; i < 64u; i++) out[i] = (uint8_t)(nib[2u * i] | (nib[2u * i + 1u] << 4));
}

/* ------------------------------------------------------------------------- */

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static const uint32_t N_ROWS = 8u;

static float frand(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*s >> 40) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}

static void make_inputs(std::vector<float> &x) {
    x.assign((size_t)N_ROWS * HAD_DIM, 0.0f);
    for (uint32_t d = 0; d < HAD_DIM; d++) x[0 * HAD_DIM + d] = 1.0f;                     /* all ones     */
    x[1 * HAD_DIM + 0] = 1.0f;                                                            /* impulse      */
    for (uint32_t d = 0; d < HAD_DIM; d++) x[2 * HAD_DIM + d] = (d & 1u) ? -1.0f : 1.0f;   /* alternating  */
    for (uint32_t d = 0; d < HAD_DIM; d++) x[3 * HAD_DIM + d] = (d == 63u) ? 40.0f : 0.0f; /* single spike */
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (uint32_t r = 4; r < N_ROWS; r++)
        for (uint32_t d = 0; d < HAD_DIM; d++) x[(size_t)r * HAD_DIM + d] = frand(&s) * 3.0f;
}

int main(void) {
    std::vector<float> x;
    make_inputs(x);

    pulsar_gpu_tensor *xd = pulsar_gpu_tensor_alloc(x.size() * sizeof(float));
    pulsar_gpu_tensor *pd = pulsar_gpu_tensor_alloc((size_t)N_ROWS * HAD_ROWBYTES);
    CHECK(xd && pd, "tensor alloc failed");
    if (!xd || !pd) return 1;
    CHECK(pulsar_gpu_tensor_write(xd, 0, x.data(), x.size() * sizeof(float)), "input write");

    /* ---- A. the Hadamard pack vs the host oracle, byte for byte ---------- */
    CHECK(pulsar_gpu_dsv4_indexer_qat_pack_tensor(xd, pd, 0, N_ROWS, 128u, false), "qat pack launch");
    pulsar_gpu_synchronize();
    std::vector<uint8_t> got((size_t)N_ROWS * HAD_ROWBYTES), want((size_t)N_ROWS * HAD_ROWBYTES);
    CHECK(pulsar_gpu_tensor_read(pd, 0, got.data(), got.size()), "packed read");

    uint32_t row_mismatch = 0, byte_mismatch = 0;
    for (uint32_t r = 0; r < N_ROWS; r++) {
        host_ref_row(&x[(size_t)r * HAD_DIM], &want[(size_t)r * HAD_ROWBYTES]);
        uint32_t bad = 0;
        for (uint32_t b = 0; b < HAD_ROWBYTES; b++)
            if (got[(size_t)r * HAD_ROWBYTES + b] != want[(size_t)r * HAD_ROWBYTES + b]) { bad++; byte_mismatch++; }
        if (bad) row_mismatch++;
    }
    CHECK(row_mismatch == 0, "indexer hadamard pack vs host oracle: %u/%u rows differ (%u bytes)",
          row_mismatch, N_ROWS, byte_mismatch);
    printf("hadamard+fp4 pack: %u rows x %u B, byte-exact vs the host oracle (incl. all %u E8M0 scale bytes)\n",
           N_ROWS, HAD_ROWBYTES, 4u * N_ROWS);

    /* ---- B. the closed forms, PRINTED: this is the ORDER claim ----------- */
    {
        float v[HAD_DIM];
        host_hadamard(&x[0], v);                     /* all ones -> [sqrt(128), 0, ...] */
        const float expect0 = sqrtf(128.0f);
        CHECK(fabsf(v[0] - expect0) < 1e-4f, "all-ones row: v[0] %g, want %g", (double)v[0], (double)expect0);
        float tail_max = 0.0f;
        for (uint32_t d = 1; d < HAD_DIM; d++) tail_max = fmaxf(tail_max, fabsf(v[d]));
        CHECK(tail_max < 1e-4f, "all-ones row: tail not zero (max %g) -- wrong transform ORDER", (double)tail_max);
        printf("order check: all-ones -> v[0]=%.4f (want sqrt(128)=%.4f), max |tail|=%.2e\n",
               (double)v[0], (double)expect0, (double)tail_max);

        host_hadamard(&x[HAD_DIM], v);               /* impulse -> flat 1/sqrt(128) */
        float flat_max = 0.0f, flat_min = 1e30f;
        for (uint32_t d = 0; d < HAD_DIM; d++) { flat_max = fmaxf(flat_max, v[d]); flat_min = fminf(flat_min, v[d]); }
        CHECK(fabsf(flat_max - HAD_SCALE) < 1e-6f && fabsf(flat_min - HAD_SCALE) < 1e-6f,
              "impulse row not flat: [%g, %g], want %g", (double)flat_min, (double)flat_max, (double)HAD_SCALE);
        printf("order check: impulse -> flat row [%.6f, %.6f] (want 1/sqrt(128)=%.6f)\n",
               (double)flat_min, (double)flat_max, (double)HAD_SCALE);

        /* The all-ones row's own scales: sqrt(128) in block 0, the amax floor in
         * the other three.  The floor's BYTE is not asserted to a literal --
         * amax is exactly 6*2^-126, and 6*2^-126 * (1/6) lands a hair above
         * 2^-126 in fp32, which is a legitimate 2 rather than 1; part A already
         * grades it byte-for-byte against the oracle.  What IS asserted here is
         * that the three zero blocks agree with each other and that block 0 is
         * the 2^1 the closed form demands. */
        CHECK(got[64] == (uint8_t)128u, "all-ones row: block 0 scale byte %u, want 128 (2^1)",
              (unsigned)got[64]);
        CHECK(got[65] == got[66] && got[66] == got[67],
              "all-ones row: the three zero blocks disagree (%u/%u/%u)",
              (unsigned)got[65], (unsigned)got[66], (unsigned)got[67]);
        printf("           all-ones row scales: block0=%u (2^1), zero blocks=%u (the amax floor)\n",
               (unsigned)got[64], (unsigned)got[65]);
    }

    /* ---- C. the fused rope+q kernel == rope then pack -------------------- */
    {
        const uint32_t n_tok = 2u, n_head = 3u, n_rows = n_tok * n_head, n_rot = 64u;
        std::vector<float> xq((size_t)n_rows * HAD_DIM);
        uint64_t s = 0x1234567ull;
        for (auto &v : xq) v = frand(&s) * 2.0f;
        pulsar_gpu_tensor *a  = pulsar_gpu_tensor_alloc(xq.size() * sizeof(float));
        pulsar_gpu_tensor *b  = pulsar_gpu_tensor_alloc(xq.size() * sizeof(float));
        pulsar_gpu_tensor *pr = pulsar_gpu_tensor_alloc((size_t)n_rows * HAD_ROWBYTES);
        pulsar_gpu_tensor *pf = pulsar_gpu_tensor_alloc((size_t)n_rows * HAD_ROWBYTES);
        CHECK(a && b && pr && pf, "q-path alloc failed");
        if (!a || !b || !pr || !pf) return 1;
        CHECK(pulsar_gpu_tensor_write(a, 0, xq.data(), xq.size() * sizeof(float)), "q write a");
        CHECK(pulsar_gpu_tensor_write(b, 0, xq.data(), xq.size() * sizeof(float)), "q write b");

        const float fb = 10000.0f, fs = 1.0f, ef = 0.0f, af = 1.0f, bf = 32.0f, bs = 1.0f;
        const uint32_t pos0 = 7u, n_ctx = 4096u;
        CHECK(pulsar_gpu_rope_tail_tensor(a, n_tok, n_head, HAD_DIM, n_rot, pos0, n_ctx, false,
                                          fb, fs, ef, af, bf, bs, NULL), "rope_tail launch");
        CHECK(pulsar_gpu_dsv4_indexer_qat_pack_tensor(a, pr, 0, n_rows, HAD_DIM, false), "pack-after-rope launch");
        CHECK(pulsar_gpu_dsv4_indexer_rope_qat_tensor(b, pf, n_tok, n_head, HAD_DIM, n_rot, pos0, n_ctx, false,
                                                      fb, fs, ef, af, bf, bs, NULL), "fused rope+qat launch");
        pulsar_gpu_synchronize();
        std::vector<uint8_t> r1((size_t)n_rows * HAD_ROWBYTES), r2((size_t)n_rows * HAD_ROWBYTES);
        CHECK(pulsar_gpu_tensor_read(pr, 0, r1.data(), r1.size()), "read r1");
        CHECK(pulsar_gpu_tensor_read(pf, 0, r2.data(), r2.size()), "read r2");
        uint32_t d2 = 0;
        for (size_t i = 0; i < r1.size(); i++) d2 += (r1[i] != r2[i]);
        CHECK(d2 == 0, "fused rope+hadamard+pack differs from rope-then-pack: %u/%zu bytes", d2, r1.size());
        printf("fused q kernel == rope_tail + hadamard pack: %u rows x %u B byte-identical\n", n_rows, HAD_ROWBYTES);
    }

    /* ---- D. V4's indexer compressor == the explicit composition ----------
     * The composite's ONLY content is the order of its parts and the rope
     * argument that order implies (`pos0 + ratio - 1`, stride `ratio`), so the
     * gate is: the composite's rows and state lane must be byte-identical to
     * those parts driven by hand with independently written arguments.  A wrong
     * rope position or stride changes the bytes; a wrong ape placement changes
     * them too, because the ape is folded BEFORE the pooling softmax. */
    {
        const uint32_t ratio = 4u, head_dim = 128u, coff = 2u, width = coff * head_dim;
        const uint32_t n_tok = 11u, pos0 = 4u, n_groups = n_tok / ratio, n_rot = 64u;
        std::vector<float> kv((size_t)n_tok * width), sc((size_t)n_tok * width), ape((size_t)ratio * width);
        uint64_t s = 0xbeef1234ull;
        for (auto &v : kv)  v = frand(&s) * 3.0f;
        for (auto &v : sc)  v = frand(&s) * 4.0f;
        for (auto &v : ape) v = frand(&s) * 0.5f;

        /* A fake model map holding the compressor's ape followed by its RMSNorm
         * weight -- the two model-mapped tables the composite reads.  It must be
         * a HOST buffer: the real cuda_model_range_ptr copies OUT of the mmap'd
         * model into a device staging copy, so handing it a device pointer makes
         * it memcpy from device memory.  (The standalone csa2 probe stubs that
         * function to base+offset and never notices.) */
        std::vector<float> wmap((size_t)ratio * width + head_dim);
        for (uint32_t i = 0; i < (uint32_t)((size_t)ratio * width); i++) wmap[i] = ape[i];
        for (uint32_t i = 0; i < head_dim; i++) wmap[(size_t)ratio * width + i] = 0.5f + frand(&s) * 0.5f;
        const uint64_t ape_offset = 0ull;
        const uint64_t norm_offset = (uint64_t)ratio * width * sizeof(float);
        const uint64_t map_bytes = (uint64_t)wmap.size() * sizeof(float);
        const void *model_map = wmap.data();

        pulsar_gpu_tensor *kvA = pulsar_gpu_tensor_alloc((size_t)n_tok * width * sizeof(float));
        pulsar_gpu_tensor *scA = pulsar_gpu_tensor_alloc((size_t)n_tok * width * sizeof(float));
        pulsar_gpu_tensor *scB = pulsar_gpu_tensor_alloc((size_t)n_tok * width * sizeof(float));
        pulsar_gpu_tensor *latA = pulsar_gpu_tensor_alloc((size_t)n_groups * head_dim * sizeof(float));
        pulsar_gpu_tensor *latB = pulsar_gpu_tensor_alloc((size_t)n_groups * head_dim * sizeof(float));
        const uint64_t lane_bytes = (uint64_t)coff * ratio * width * sizeof(float);
        pulsar_gpu_tensor *sA_kv = pulsar_gpu_tensor_alloc(lane_bytes), *sA_sc = pulsar_gpu_tensor_alloc(lane_bytes);
        pulsar_gpu_tensor *sB_kv = pulsar_gpu_tensor_alloc(lane_bytes), *sB_sc = pulsar_gpu_tensor_alloc(lane_bytes);
        pulsar_gpu_tensor *pkA = pulsar_gpu_tensor_alloc((size_t)n_groups * HAD_ROWBYTES);
        pulsar_gpu_tensor *pkB = pulsar_gpu_tensor_alloc((size_t)n_groups * HAD_ROWBYTES);
        CHECK(kvA && scA && scB && latA && latB && sA_kv && sA_sc && sB_kv && sB_sc && pkA && pkB,
              "compressor alloc failed");
        if (g_fail) return 1;
        CHECK(pulsar_gpu_tensor_write(kvA, 0, kv.data(), kv.size() * sizeof(float)), "kv write");
        CHECK(pulsar_gpu_tensor_write(scA, 0, sc.data(), sc.size() * sizeof(float)), "sc write A");
        CHECK(pulsar_gpu_tensor_write(scB, 0, sc.data(), sc.size() * sizeof(float)), "sc write B");
        
        /* pos0 != 0 means the lane is LIVE: the reference reads the incoming
         * carry for group 0 rather than padding, and the csa2 prefill
         * deliberately does NOT clear it (clearing would destroy that carry).
         * Both routes must therefore start from the SAME state, or they would
         * each read whatever their own fresh allocation happened to hold. */
        {
            std::vector<float> z(lane_bytes / sizeof(float), 0.0f);
            std::vector<float> ninf(lane_bytes / sizeof(float), -INFINITY);
            CHECK(pulsar_gpu_tensor_write(sA_kv, 0, z.data(), lane_bytes), "init sA_kv");
            CHECK(pulsar_gpu_tensor_write(sB_kv, 0, z.data(), lane_bytes), "init sB_kv");
            CHECK(pulsar_gpu_tensor_write(sA_sc, 0, ninf.data(), lane_bytes), "init sA_sc");
            CHECK(pulsar_gpu_tensor_write(sB_sc, 0, ninf.data(), lane_bytes), "init sB_sc");
        }

        const float fb = 10000.0f, fs = 1.0f, ef = 0.0f, af = 1.0f, bf = 32.0f, bs = 1.0f;
        const uint32_t n_ctx = 4096u;
        const float eps = 1e-6f;

        /* route A: the composite */
        CHECK(pulsar_gpu_indexer_compressor_prefill_tensor(pkA, latA, sA_kv, sA_sc, scA, kvA,
                                                           model_map, map_bytes, ape_offset, 0u,
                                                           norm_offset, 0u, 0u,
                                                           head_dim, ratio, pos0, n_tok, n_rot, n_ctx,
                                                           fb, fs, ef, af, bf, bs, eps),
              "indexer compressor composite launch");
        /* route B: the parts, by hand, with the arguments written out here */
        CHECK(pulsar_gpu_csa2_comp_ape_add_tensor(scB, model_map, map_bytes, ape_offset, 0u,
                                                  width, ratio, pos0, n_tok), "ape add B");
        CHECK(pulsar_gpu_csa2_compressor_prefill_tensor(latB, kvA, scB, sB_kv, sB_sc,
                                                        model_map, map_bytes, 0ull, 0u,
                                                        head_dim, ratio, pos0, n_tok, eps), "pool B");
        CHECK(pulsar_gpu_rope_tail_strided_tensor(latB, n_groups, head_dim, n_rot,
                                                  pos0 + ratio - 1u, ratio, n_ctx,
                                                  fb, fs, ef, af, bf, bs), "rope B");
        CHECK(pulsar_gpu_dsv4_indexer_qat_pack_tensor(latB, pkB, 0, n_groups, head_dim, false), "pack B");
        pulsar_gpu_synchronize();

        std::vector<uint8_t> ra((size_t)n_groups * HAD_ROWBYTES), rb((size_t)n_groups * HAD_ROWBYTES);
        CHECK(pulsar_gpu_tensor_read(pkA, 0, ra.data(), ra.size()), "read pkA");
        CHECK(pulsar_gpu_tensor_read(pkB, 0, rb.data(), rb.size()), "read pkB");
        uint32_t dd = 0;
        for (size_t i = 0; i < ra.size(); i++) dd += (ra[i] != rb[i]);
        CHECK(dd == 0, "indexer compressor composite vs its parts: %u/%zu row bytes differ", dd, ra.size());

        std::vector<float> sa_kv(lane_bytes / 4), sb_kv(lane_bytes / 4), sa_sc(lane_bytes / 4), sb_sc(lane_bytes / 4);
        CHECK(pulsar_gpu_tensor_read(sA_kv, 0, sa_kv.data(), lane_bytes), "read sA_kv");
        CHECK(pulsar_gpu_tensor_read(sB_kv, 0, sb_kv.data(), lane_bytes), "read sB_kv");
        CHECK(pulsar_gpu_tensor_read(sA_sc, 0, sa_sc.data(), lane_bytes), "read sA_sc");
        CHECK(pulsar_gpu_tensor_read(sB_sc, 0, sb_sc.data(), lane_bytes), "read sB_sc");
        uint32_t sd = 0;
        for (size_t i = 0; i < sa_kv.size(); i++) {
            sd += (memcmp(&sa_kv[i], &sb_kv[i], 4) != 0);
            sd += (memcmp(&sa_sc[i], &sb_sc[i], 4) != 0);
        }
        CHECK(sd == 0, "indexer compressor state lane differs from its parts: %u elements", sd);
        printf("indexer compressor: %u tokens -> %u index-K rows (%u B each), composite == its parts byte-for-byte; "
               "state lane %llu B identical\n",
               n_tok, n_groups, HAD_ROWBYTES, (unsigned long long)lane_bytes);

        /* A batch that closes NO group (n_tokens < ratio) emits no row, but the
         * lane still has to receive the trailing partial rows: the next chunk's
         * first group pools them in, and the reference stashes them
         * (`kv_state[...offset:offset+remainder]`, `+ self.ape[:remainder]`).
         * The composite used to return before the pool on this shape, which left
         * the lane at its zero/-inf prime and made the group spanning the two
         * chunks pool the wrong tokens -- silently, since the emitted rows still
         * looked like numbers.  There is no packed row to compare here, so the
         * LANE is the byte-exact witness. */
        {
            const uint32_t r_tok = 3u;   /* < ratio: no complete group */
            std::vector<float> z(lane_bytes / 4, 0.0f), ninf(lane_bytes / 4, -INFINITY);
            CHECK(pulsar_gpu_tensor_write(sA_kv, 0, z.data(), lane_bytes), "reset sA_kv");
            CHECK(pulsar_gpu_tensor_write(sB_kv, 0, z.data(), lane_bytes), "reset sB_kv");
            CHECK(pulsar_gpu_tensor_write(sA_sc, 0, ninf.data(), lane_bytes), "reset sA_sc");
            CHECK(pulsar_gpu_tensor_write(sB_sc, 0, ninf.data(), lane_bytes), "reset sB_sc");
            CHECK(pulsar_gpu_tensor_write(scA, 0, sc.data(), (size_t)r_tok * width * sizeof(float)), "sc write A");
            CHECK(pulsar_gpu_tensor_write(scB, 0, sc.data(), (size_t)r_tok * width * sizeof(float)), "sc write B");

            CHECK(pulsar_gpu_indexer_compressor_prefill_tensor(pkA, latA, sA_kv, sA_sc, scA, kvA,
                                                               model_map, map_bytes, ape_offset, 0u,
                                                               norm_offset, 0u, 0u,
                                                               head_dim, ratio, pos0, r_tok, n_rot, n_ctx,
                                                               fb, fs, ef, af, bf, bs, eps),
                  "remainder-only composite launch");
            CHECK(pulsar_gpu_csa2_comp_ape_add_tensor(scB, model_map, map_bytes, ape_offset, 0u,
                                                      width, ratio, pos0, r_tok), "remainder-only ape add B");
            CHECK(pulsar_gpu_csa2_compressor_prefill_tensor(latB, kvA, scB, sB_kv, sB_sc,
                                                            model_map, map_bytes, 0ull, 0u,
                                                            head_dim, ratio, pos0, r_tok, eps),
                  "remainder-only pool B");
            pulsar_gpu_synchronize();

            std::vector<float> ra_kv(lane_bytes / 4), rb_kv(lane_bytes / 4), ra_sc(lane_bytes / 4), rb_sc(lane_bytes / 4);
            CHECK(pulsar_gpu_tensor_read(sA_kv, 0, ra_kv.data(), lane_bytes), "read sA_kv");
            CHECK(pulsar_gpu_tensor_read(sB_kv, 0, rb_kv.data(), lane_bytes), "read sB_kv");
            CHECK(pulsar_gpu_tensor_read(sA_sc, 0, ra_sc.data(), lane_bytes), "read sA_sc");
            CHECK(pulsar_gpu_tensor_read(sB_sc, 0, rb_sc.data(), lane_bytes), "read sB_sc");
            uint32_t rd = 0;
            for (size_t i = 0; i < ra_kv.size(); i++) {
                rd += (memcmp(&ra_kv[i], &rb_kv[i], 4) != 0);
                rd += (memcmp(&ra_sc[i], &rb_sc[i], 4) != 0);
            }
            CHECK(rd == 0, "remainder-only batch: the lane differs from its parts: %u elements", rd);
            /* and the lane is NOT the untouched prime, or the comparison above
             * would be two empty lanes agreeing with each other */
            uint32_t nz = 0;
            for (size_t i = 0; i < ra_kv.size(); i++) nz += (ra_kv[i] != 0.0f);
            CHECK(nz != 0, "remainder-only batch wrote nothing into the lane");
            printf("indexer compressor: %u tokens (< ratio %u) -> lane %u/%zu kv elements non-empty, "
                   "composite == its parts\n", r_tok, ratio, nz, ra_kv.size());
        }

        /* the lane is the size the struct note claims: coff*ratio x coff*head_dim */
        CHECK(lane_bytes == (uint64_t)coff * ratio * coff * head_dim * sizeof(float),
              "index lane geometry is not coff*ratio x coff*head_dim (%llu B)", (unsigned long long)lane_bytes);
    }

    printf("INDEXER HADAMARD KERNEL TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}

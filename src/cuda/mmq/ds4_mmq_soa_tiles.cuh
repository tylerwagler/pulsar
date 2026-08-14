#pragma once

// ds4-owned SoA tile loaders (L008, 2026-08-14).
//
// These are the aligned row-pair-SoA twins of upstream's q2_K / iq2_xxs tile
// loaders.  They live HERE, not in upstream's mmq-load-tiles.cuh, so that file
// re-syncs by copy.  Only the template wrapper was adapted to master's shape
// (mmq_y -> I, need_check -> fallback, nwarps derived from
// ggml_cuda_mmq_get_nthreads); the byte-decoding bodies are unchanged from the
// 5c0e946-era originals and remain bit-identical to the standard loaders.
//
// They take one extra argument the upstream loader typedef has no room for --
// soa_npair, the artifact's total pair count -- which is why mmq.cuh threads
// x_soa/soa_blocks down to the load call instead of routing through
// ggml_cuda_mmq_get_load_tiles().

// ds4 (P4 Inc3): aligned row-pair-SoA twin of load_tiles_q2_K.  x points at
// the weight-server artifact [uint2 dm2 | int4 sc4 | uint2 qs2] (row pairs
// interleaved; q2_k_aligned_derepack_kernel in ds4_mmq.cu is the byte-level
// spec this inverts) and soa_npair is the total pair count = nblk/2, from
// which the section offsets derive.  Every tile_x element is the same value
// the standard loader reads from the raw 84-byte block, moved through
// integer selects only, so the output is bit-identical and vec_dot is
// untouched.  No expert geometry is needed: rows per expert is even, so the
// artifact's pair index ((e*(M/2))+r/2)*nb_row+b equals
// (r_global/2)*nb_row+b for the flat row r_global = e*M+r.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void load_tiles_q2_K_soa(
    const char * __restrict__ x, const int64_t soa_npair, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

    const uint64_t dm_bytes = ((uint64_t)soa_npair *  8u + 63u) & ~63ull;
    const uint64_t sc_bytes = ((uint64_t)soa_npair * 32u + 63u) & ~63ull;
    const uint2 * dm2 = (const uint2 *)  x;
    const int4  * sc4 = (const int4  *) (x + dm_bytes);
    const uint2 * qs2 = (const uint2 *) (x + dm_bytes + sc_bytes);

    // kbx0 = offset_x + kb0 is a multiple of stride (the row part; experts
    // and row tiles are whole-row offsets) plus the block column kb0 <
    // stride, so the (row, block-column) split hoists out of the row loop.
    const int r0 = kbx0 / stride;
    const int b  = kbx0 % stride;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const int      rg   = r0 + i;
        // Element indices stay 32-bit: the largest artifact (V4-Pro down)
        // has ~16.5M pairs -> qs2 index < 2^28.
        const uint32_t pblk = (uint32_t)(rg >> 1)*(uint32_t)stride + (uint32_t)b;
        const uint32_t par  = (uint32_t)rg & 1u;

        const uint2 q = qs2[pblk*16u + kqsx];
        const int x_ql_0 = (int)(par ? q.y : q.x);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        // scales byte kqsx of the raw block = word w = kqsx/4; the artifact
        // packs scale words {row0_w0, row0_w1, row1_w0, row1_w1} as two int4
        // per pair, so (w>>1) picks the int4 and (par, w&1) the component.
        const int w = kqsx >> 2;
        const int4 s = sc4[pblk*2u + (uint32_t)(w >> 1)];
        const uint32_t sw = par ? ((w & 1) ? (uint32_t)s.w : (uint32_t)s.z)
                                : ((w & 1) ? (uint32_t)s.y : (uint32_t)s.x);
        const int sc_m = (int)((sw >> (8*(kqsx & 3))) & 0xFFu);

        const uint2 dmp = dm2[pblk];
        const uint32_t dm_bits = par ? dmp.y : dmp.x;
        half2 dm;
        *reinterpret_cast<uint32_t *>(&dm) = dm_bits;
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i*(MMQ_TILE_NE_K + 1)   + kqsx] = x_dm_ik;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}


// ds4 (P4 Inc3): aligned-SoA twin of load_tiles_iq2_xxs.  x points at the
// weight-server artifact [__half dq | uint2 qs] (iq2_xxs_aligned_derepack_kernel
// in ds4_mmq.cu is the byte-level spec) and soa_nblk is the total block
// count.  Each lane's (q2, aux32) int pair is exactly one artifact uint2, so
// the loads are 8-byte aligned and fully coalesced; the unpack math is the
// standard loader's, making tile_x bit-identical.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void load_tiles_iq2_xxs_soa(
    const char * __restrict__ x, const int64_t soa_nblk, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

    const uint64_t dq_bytes = ((uint64_t)soa_nblk * 2u + 63u) & ~63ull;
    const half  * dq = (const half  *)  x;
    const uint2 * qs = (const uint2 *) (x + dq_bytes);

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        // Flat block index; < 2^25 for both model shapes, so 32-bit element
        // indexing is safe.
        const uint32_t blk = (uint32_t)(kbx0 + i*stride);

        const uint2 v = qs[blk*8u + kqsx];
        const int q2 = (int)v.x;
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = v.y;

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = dq[blk];
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0   + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}



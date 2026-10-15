/* Short convolution block timings at Inkling-Small's shapes (L295 E3).  Not a
 * gate.  Per layer Inkling runs three launches: K + V (two 1024-channel streams
 * read strided from the fused [q 4096 | k 1024 | v 1024 | r 512] row, one
 * launch) and the attention-output and MLP-output deltas (4096 channels each).
 * Lanes: decode (1 and 16 sequences x 1 row), a 6-row verify of 16 sequences
 * with per-row capture, and a 4096-row prefill chunk.  Each timing is the mean
 * of 200 back-to-back launches on the same buffers (inputs and slots stay
 * L2-resident for the decode lanes; the prefill's 32 MB a stream does not), with
 * the bytes each launch must move and the rate that implies. */
#include "cuda/pulsar_cuda_sconv.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Lane { const char *name; uint32_t seqs, rows_per_seq; bool capture; };

int main() {
    const uint32_t QKVR = 6656, H = 4096, K = 4, MAXR = 4096;
    uint16_t *w, *qkvr, *res, *y, *pool, *cap;
    CK(cudaMalloc(&w, (size_t)(2 * 1024 + 2 * H) * K * 2));
    CK(cudaMemset(w, 0x3c, (size_t)(2 * 1024 + 2 * H) * K * 2));
    CK(cudaMalloc(&qkvr, (size_t)MAXR * QKVR * 2)); CK(cudaMemset(qkvr, 0x3c, (size_t)MAXR * QKVR * 2));
    CK(cudaMalloc(&res, (size_t)MAXR * H * 2)); CK(cudaMemset(res, 0x3c, (size_t)MAXR * H * 2));
    CK(cudaMalloc(&y, (size_t)MAXR * H * 2 * 2));
    const uint64_t ss = (uint64_t)(K - 1) * H;   /* one slot of the widest stream */
    CK(cudaMalloc(&pool, 4 * 16 * ss * 2)); CK(cudaMemset(pool, 0, 4 * 16 * ss * 2));
    CK(cudaMalloc(&cap, (size_t)4 * 16 * 6 * ss * 2));
    int32_t *first, *slot;
    CK(cudaMalloc(&first, 17 * 4)); CK(cudaMalloc(&slot, 16 * 4));
    pulsar_sconv_decl kd{1024, K, 1, w}, vd{1024, K, 1, w + 1024 * K}, ad{H, K, 1, w + 2048 * K}, md{H, K, 1, w + (2048 + H) * K};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const Lane lanes[] = {{"decode, 1 seq x 1 row", 1, 1, false},
                          {"decode, 16 seqs x 1 row", 16, 1, false},
                          {"verify, 16 seqs x 6 rows + capture", 16, 6, true},
                          {"prefill, 1 seq x 4096 rows", 1, 4096, false}};
    printf("sconv bench (L295 E3), Inkling-Small shapes, K = 4, 200 launches each\n");
    printf("%-36s %-18s %10s %10s %9s\n", "lane", "launch", "us", "MB moved", "GB/s");
    for (const Lane &L : lanes) {
        std::vector<int32_t> hf(L.seqs + 1), hs(L.seqs);
        for (uint32_t s = 0; s <= L.seqs; ++s) hf[s] = (int32_t)(s * L.rows_per_seq);
        for (uint32_t s = 0; s < L.seqs; ++s) hs[s] = (int32_t)s;
        CK(cudaMemcpy(first, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(slot, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
        const uint32_t rows = L.seqs * L.rows_per_seq;
        const pulsar_sconv_rows rw{L.seqs, rows, L.rows_per_seq, first, slot};
        auto mk = [&](const pulsar_sconv_decl *d, const uint16_t *x, uint64_t ldx, uint16_t *yy, int si) {
            pulsar_sconv_stream s{};
            s.decl = d; s.x = x; s.ld_x = ldx; s.y = yy; s.ld_y = d->channels;
            s.state = pool + (uint64_t)si * 16 * ss; s.state_stride = ss;
            if (L.capture) { s.capture = cap + (uint64_t)si * 16 * 6 * ss; s.capture_stride = ss; }
            return s;
        };
        const pulsar_sconv_stream kv[2] = {mk(&kd, qkvr + 4096, QKVR, y, 0), mk(&vd, qkvr + 5120, QKVR, y + (size_t)MAXR * 1024, 1)};
        const pulsar_sconv_stream at = mk(&ad, res, H, y, 2), ml = mk(&md, res, H, y + (size_t)MAXR * H, 3);
        struct { const char *nm; const pulsar_sconv_stream *s; uint32_t n; uint32_t C; } runs[] = {
            {"K+V (2 x 1024)", kv, 2, 2048}, {"attn delta (4096)", &at, 1, H}, {"mlp delta (4096)", &ml, 1, H}};
        double layer_us = 0;
        for (auto &r : runs) {
            for (int i = 0; i < 10; ++i)
                if (pulsar_sconv_forward(r.s, r.n, &rw, 0)) return 1;
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 200; ++i) pulsar_sconv_forward(r.s, r.n, &rw, 0);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / 200.0;
            const double bytes = (double)rows * r.C * 2 * 2                      /* x in, y out */
                               + (double)L.seqs * (K - 1) * r.C * 2 * 2         /* the slot read and written */
                               + (L.capture ? (double)rows * (K - 1) * r.C * 2 : 0.0)
                               + (double)r.C * K * 2;                           /* weights */
            printf("%-36s %-18s %10.2f %10.3f %9.1f\n", L.name, r.nm, us, bytes / 1e6, bytes / us / 1e3);
            layer_us += us;
        }
        printf("%-36s %-18s %10.2f   (x 42 layers = %.3f ms)\n", "", "a layer's 3 launches", layer_us, layer_us * 42 / 1000);
    }
    return 0;
}

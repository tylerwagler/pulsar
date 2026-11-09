/* Correctness gate for the CSA2 candidate-pool kernels (L218).
 *
 * #includes the shipped .cu so it drives the REAL kernels.  The oracle is the
 * reference's select_candidate_blocks written out in C: block max over
 * `block_size` positions with -inf padding, the row's newest block pinned to
 * +inf, the topk_blocks largest kept, -inf picks dropped.  Ties are resolved
 * the way the kernel documents (position-first), so the oracle sorts by
 * (score desc, position asc) and the comparison is exact.
 *
 * Shapes: n_comp 4100 (513 blocks, a partial last block), block 8, k 100 --
 * and k 2048 over 4100 (fewer blocks than k: every reachable block kept, the
 * unreachable ones not).  Rows at several reaches, including a row whose
 * reach ends mid-block (the pin) and a row with reach 0.  Then the mask
 * kernel: every masked score is -inf, every unmasked one untouched.
 *
 * build+run (on the GPU box):
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/cand_kt tests/candidate_kernel_test.cu
 */
#include "../src/cuda/pulsar_cuda_candidates.cu"

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return 0;
}
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    (void)bytes; (void)what; return (const char *)model_map + offset;
}

static pulsar_gpu_tensor *dev_tensor(size_t bytes) {
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof(*t));
    if (cudaMalloc(&t->ptr, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu\n", bytes); exit(1); }
    t->bytes = bytes; return t;
}
static float frand(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*s >> 40) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}

/* the reference, with the documented tie rule */
static void oracle_mask(const float *sr, uint32_t n_comp, uint32_t vis, uint32_t bs, uint32_t k, std::vector<uint8_t> &keep) {
    const uint32_t nb = (n_comp + bs - 1) / bs;
    std::vector<float> bscore(nb);
    for (uint32_t b = 0; b < nb; b++) {
        float m = -INFINITY;
        for (uint32_t c = b * bs; c < std::min(n_comp, (b + 1) * bs); c++) m = fmaxf(m, sr[c]);
        bscore[b] = m;
    }
    if (vis) bscore[(vis - 1) / bs] = INFINITY;
    std::vector<uint32_t> order(nb);
    for (uint32_t b = 0; b < nb; b++) order[b] = b;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return bscore[a] > bscore[b]; });
    keep.assign(nb, 0);
    const uint32_t kk = std::min(k, nb);
    for (uint32_t i = 0; i < kk; i++) if (bscore[order[i]] > -INFINITY) keep[order[i]] = 1;
}

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static void run_case(uint32_t n_comp, uint32_t bs, uint32_t k, uint32_t ratio, const std::vector<uint32_t> &pos, uint64_t *seed, bool ties) {
    const uint32_t n_rows = (uint32_t)pos.size();
    const uint32_t nb = (n_comp + bs - 1) / bs, mw = pulsar_gpu_candidate_mask_words(n_comp, bs);
    std::vector<float> scores((size_t)n_rows * n_comp);
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t vis = std::min(n_comp, (pos[r] + 1u) / ratio);
        for (uint32_t c = 0; c < n_comp; c++) {
            float v = c < vis ? 3.0f * frand(seed) : -INFINITY;
            if (ties && c < vis) v = floorf(v * 4.0f) / 4.0f;   /* quantised scores: many exact ties */
            scores[(size_t)r * n_comp + c] = v;
        }
    }
    pulsar_gpu_tensor *sc_d = dev_tensor(scores.size() * 4), *mask_d = dev_tensor((size_t)n_rows * mw * 4),
                      *scr_d = dev_tensor((size_t)n_rows * nb * 4), *pos_d = dev_tensor(n_rows * 4);
    cudaMemcpy(sc_d->ptr, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice);
    std::vector<int32_t> posi(pos.begin(), pos.end());
    cudaMemcpy(pos_d->ptr, posi.data(), n_rows * 4, cudaMemcpyHostToDevice);
    CHECK(pulsar_gpu_candidate_blocks_tensor(mask_d, scr_d, sc_d, n_comp, n_rows, mw, bs, k, 0u, ratio, pos_d),
          "candidate blocks launch (n_comp %u k %u)", n_comp, k);
    cudaDeviceSynchronize();
    std::vector<uint32_t> mask((size_t)n_rows * mw);
    cudaMemcpy(mask.data(), mask_d->ptr, mask.size() * 4, cudaMemcpyDeviceToHost);
    int bad = 0, kept_total = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        std::vector<uint8_t> keep;
        const uint32_t vis = std::min(n_comp, (pos[r] + 1u) / ratio);
        oracle_mask(&scores[(size_t)r * n_comp], n_comp, vis, bs, k, keep);
        for (uint32_t b = 0; b < nb; b++) {
            const int got = (mask[(size_t)r * mw + (b >> 5)] >> (b & 31)) & 1;
            if (got != keep[b]) { if (bad < 5) printf("  row %u block %u: kernel %d oracle %d (vis %u)\n", r, b, got, keep[b], vis); bad++; }
            kept_total += got;
        }
    }
    CHECK(bad == 0, "candidate mask vs oracle: %d block(s) differ (n_comp %u, k %u, ties %d)", bad, n_comp, k, ties);
    /* level two: masked scores */
    CHECK(pulsar_gpu_candidate_mask_scores_tensor(sc_d, mask_d, n_comp, n_rows, mw, bs), "mask scores launch");
    cudaDeviceSynchronize();
    std::vector<float> masked(scores.size());
    cudaMemcpy(masked.data(), sc_d->ptr, masked.size() * 4, cudaMemcpyDeviceToHost);
    int mbad = 0;
    for (uint32_t r = 0; r < n_rows; r++)
        for (uint32_t c = 0; c < n_comp; c++) {
            const int in = (mask[(size_t)r * mw + ((c / bs) >> 5)] >> ((c / bs) & 31)) & 1;
            const float want = in ? scores[(size_t)r * n_comp + c] : -INFINITY;
            if (masked[(size_t)r * n_comp + c] != want && !(isinf(want) && isinf(masked[(size_t)r * n_comp + c]))) mbad++;
        }
    CHECK(mbad == 0, "masked scores: %d elements wrong", mbad);
    printf("n_comp %u block %u k %u rows %u ties %d: %d blocks kept, mask == oracle: %s, scores masked: %s\n",
           n_comp, bs, k, n_rows, ties, kept_total, bad ? "NO" : "yes", mbad ? "NO" : "yes");
}

int main(void) {
    uint64_t seed = 0xc0ffee;
    /* rows: reach 4100 (all), 4093 (mid-block pin at block 511), 8 (one block), 1, 0 (nothing reachable) */
    std::vector<uint32_t> pos = { 4099u, 4092u, 7u, 0u, 0u };
    run_case(4100u, 8u, 100u, 1u, pos, &seed, false);
    run_case(4100u, 8u, 2048u, 1u, pos, &seed, false);
    run_case(4100u, 8u, 100u, 1u, pos, &seed, true);
    /* ratio 2: reach is (pos+1)/2 */
    std::vector<uint32_t> pos2 = { 8199u, 8185u, 15u, 1u };
    run_case(4100u, 8u, 300u, 2u, pos2, &seed, true);
    printf("CANDIDATE KERNEL TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}

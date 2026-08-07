/* mmq_align_roundtrip.cu -- prove the OFFLINE aligned artifact produced by
 * gguf-tools/repack_iq2_mmq.py is byte-identical to what the vendored device
 * repack builds, and that it inverts cleanly.
 *
 * Three checks, all byte-exact, on a REAL tensor lifted out of the model:
 *   A. size      : ds4_mmq_iq2_xxs_aligned_bytes(M,K,E) == raw stream bytes
 *                  == the offline artifact's file size.
 *   B. layout    : ds4_repack_iq2_aligned_device(raw) == offline artifact.
 *                  This is the check that actually pins the layout -- a
 *                  round-trip alone would pass for ANY self-consistent
 *                  permutation, including a wrong one.
 *   C. round-trip: ds4_mmq_iq2_xxs_aligned_derepack(offline artifact) == raw.
 *
 * Usage: mmq_align_roundtrip RAW.bin ALIGNED.bin in_dim out_dim group_count
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#include "ds4_repack.h"
#include "ds4_mmq.h"

/* Single-token decode hook the adapter references but prefill never reaches. */
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, const void **q81) {
    (void)src; (void)in_dim; (void)q81; return 0;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2);} } while (0)

static unsigned char *slurp(const char *path, uint64_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(2); }
    fseeko(f, 0, SEEK_END);
    uint64_t n = (uint64_t)ftello(f);
    fseeko(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc(n);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(2); }
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    *n_out = n;
    return p;
}

/* Report the first differing byte and the total, not just a pass/fail bit. */
static int cmp_report(const char *tag, const unsigned char *a,
                      const unsigned char *b, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            uint64_t diff = 0;
            for (uint64_t j = 0; j < n; j++) diff += (a[j] != b[j]);
            fprintf(stderr, "%s: MISMATCH at byte %llu (%02x vs %02x); "
                    "%llu of %llu bytes differ (%.4f%%)\n", tag,
                    (unsigned long long)i, a[i], b[i],
                    (unsigned long long)diff, (unsigned long long)n,
                    100.0 * (double)diff / (double)n);
            return 1;
        }
    }
    printf("%s: OK (%llu bytes identical)\n", tag, (unsigned long long)n);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s RAW.bin ALIGNED.bin in_dim out_dim group_count\n", argv[0]);
        return 1;
    }
    const uint64_t in_dim = strtoull(argv[3], NULL, 10);
    const uint64_t out_dim = strtoull(argv[4], NULL, 10);
    const uint32_t groups = (uint32_t)strtoul(argv[5], NULL, 10);

    uint64_t raw_n = 0, ali_n = 0;
    unsigned char *raw = slurp(argv[1], &raw_n);
    unsigned char *ali = slurp(argv[2], &ali_n);

    /* aligned_bytes takes (M, K, n_experts) = (out_dim, in_dim, groups). */
    const uint64_t want = ds4_mmq_iq2_xxs_aligned_bytes((int)out_dim, (int)in_dim, (int)groups);
    const uint64_t nblk = (uint64_t)groups * out_dim * (in_dim / 256u);
    printf("in_dim(K)=%llu out_dim(M)=%llu experts=%u nblk=%llu\n",
           (unsigned long long)in_dim, (unsigned long long)out_dim, groups,
           (unsigned long long)nblk);
    printf("raw file     = %llu bytes\n", (unsigned long long)raw_n);
    printf("aligned file = %llu bytes\n", (unsigned long long)ali_n);
    printf("ds4_mmq_iq2_xxs_aligned_bytes(M=%llu,K=%llu,E=%u) = %llu\n",
           (unsigned long long)out_dim, (unsigned long long)in_dim, groups,
           (unsigned long long)want);

    int rc = 0;
    if (raw_n != nblk * 66ull) {
        fprintf(stderr, "A size: raw file %llu != nblk*66 %llu\n",
                (unsigned long long)raw_n, (unsigned long long)(nblk * 66ull));
        rc = 1;
    }
    if (want != raw_n || ali_n != raw_n) {
        fprintf(stderr, "A size: MISMATCH aligned_bytes=%llu raw=%llu file=%llu delta=%lld\n",
                (unsigned long long)want, (unsigned long long)raw_n,
                (unsigned long long)ali_n, (long long)(want - raw_n));
        rc = 1;
    } else {
        printf("A size: OK (aligned == raw == file == %llu, delta 0)\n",
               (unsigned long long)raw_n);
    }

    unsigned char *d_raw = NULL, *d_ali = NULL, *d_back = NULL;
    CK(cudaMalloc(&d_raw, raw_n));
    CK(cudaMalloc(&d_ali, want));
    CK(cudaMalloc(&d_back, raw_n));
    CK(cudaMemcpy(d_raw, raw, raw_n, cudaMemcpyHostToDevice));
    /* Poison the destination so any byte the repack fails to write shows up. */
    CK(cudaMemset(d_ali, 0xA5, want));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    if (!ds4_repack_iq2_aligned_device(d_ali, d_raw, in_dim, out_dim, groups, 0)) {
        fprintf(stderr, "ds4_repack_iq2_aligned_device returned false\n");
        return 2;
    }
    CK(cudaEventRecord(t1));
    CK(cudaDeviceSynchronize());
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("device repack: %.3f ms\n", ms);

    unsigned char *h_dev = (unsigned char *)malloc(want);
    CK(cudaMemcpy(h_dev, d_ali, want, cudaMemcpyDeviceToHost));
    rc |= cmp_report("B layout (offline artifact vs device repack)", ali, h_dev, want);

    /* Derepack the OFFLINE artifact (not the device one) -- that is the object
     * whose invertibility we care about. */
    CK(cudaMemcpy(d_ali, ali, want, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_back, 0x5A, raw_n));
    if (ds4_mmq_iq2_xxs_aligned_derepack(d_ali, d_back, (int)out_dim, (int)in_dim,
                                         (int)groups, 0) != 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_aligned_derepack returned nonzero\n");
        return 2;
    }
    CK(cudaDeviceSynchronize());
    unsigned char *h_back = (unsigned char *)malloc(raw_n);
    CK(cudaMemcpy(h_back, d_back, raw_n, cudaMemcpyDeviceToHost));
    rc |= cmp_report("C round-trip (derepack(offline) vs raw)", raw, h_back, raw_n);

    printf(rc ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return rc;
}

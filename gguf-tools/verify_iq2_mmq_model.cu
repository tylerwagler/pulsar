/* verify_model_mmq.cu -- payload proof for the shipped repacked GGUF.
 *
 * For EVERY converted span listed in the manifest (name off bytes d0 d1 d2):
 *   1. pread the aligned span out of the repacked model and the raw span out
 *      of the source model,
 *   2. device-repack the SOURCE raw bytes and require the result to equal the
 *      shipped aligned bytes (this is what pins the layout -- see B below),
 *   3. device-derepack the SHIPPED aligned bytes and require the result to
 *      equal the source raw bytes (the artifact inverts to the original model).
 *
 * Both directions are checked because either alone is weaker: a round-trip is
 * self-consistent for any invertible permutation, and a forward match alone
 * would not catch a lossy write.
 *
 * Usage: verify_model_mmq SRC.gguf DST.gguf MANIFEST
 */
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#include "ds4_repack.h"
#include "ds4_mmq.h"

extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, const void **q81) {
    (void)src; (void)in_dim; (void)q81; return 0;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2);} } while (0)

static int pread_full(int fd, void *buf, uint64_t n, uint64_t off) {
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) return 0;
        p += r; off += (uint64_t)r; n -= (uint64_t)r;
    }
    return 1;
}

static uint64_t first_diff(const uint8_t *a, const uint8_t *b, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return ~0ull;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s SRC.gguf DST.gguf MANIFEST\n", argv[0]);
        return 1;
    }
    int fs = open(argv[1], O_RDONLY);
    int fd = open(argv[2], O_RDONLY);
    if (fs < 0 || fd < 0) { fprintf(stderr, "open failed\n"); return 2; }
    FILE *mf = fopen(argv[3], "r");
    if (!mf) { fprintf(stderr, "open manifest failed\n"); return 2; }

    char name[512];
    unsigned long long off, nbytes, d0, d1, d2;
    uint64_t cap = 0;
    uint8_t *h_raw = NULL, *h_ali = NULL, *h_tmp = NULL;
    uint8_t *d_raw = NULL, *d_ali = NULL, *d_out = NULL;

    int n_ok = 0, n_fail = 0;
    uint64_t total = 0;
    while (fscanf(mf, "%511s %llu %llu %llu %llu %llu",
                  name, &off, &nbytes, &d0, &d1, &d2) == 6) {
        if (nbytes > cap) {
            free(h_raw); free(h_ali); free(h_tmp);
            if (d_raw) { CK(cudaFree(d_raw)); CK(cudaFree(d_ali)); CK(cudaFree(d_out)); }
            cap = nbytes;
            h_raw = (uint8_t *)malloc(cap);
            h_ali = (uint8_t *)malloc(cap);
            h_tmp = (uint8_t *)malloc(cap);
            CK(cudaMalloc(&d_raw, cap));
            CK(cudaMalloc(&d_ali, cap));
            CK(cudaMalloc(&d_out, cap));
        }
        if (!pread_full(fs, h_raw, nbytes, off) || !pread_full(fd, h_ali, nbytes, off)) {
            fprintf(stderr, "%s: read failed\n", name); return 2;
        }

        /* The aligned artifact must be exactly as large as the raw stream. */
        const uint64_t want = ds4_mmq_iq2_xxs_aligned_bytes((int)d1, (int)d0, (int)d2);
        if (want != nbytes) {
            fprintf(stderr, "%s: SIZE aligned_bytes=%llu != raw %llu (delta %lld)\n",
                    name, (unsigned long long)want, nbytes, (long long)(want - nbytes));
            n_fail++; continue;
        }

        /* B: device-repack(source raw) == shipped aligned */
        CK(cudaMemcpy(d_raw, h_raw, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_ali, 0xA5, nbytes));
        if (!ds4_repack_iq2_aligned_device(d_ali, d_raw, d0, d1, (uint32_t)d2, 0)) {
            fprintf(stderr, "%s: device repack failed\n", name); n_fail++; continue;
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_tmp, d_ali, nbytes, cudaMemcpyDeviceToHost));
        uint64_t i = first_diff(h_tmp, h_ali, nbytes);
        if (i != ~0ull) {
            fprintf(stderr, "%s: LAYOUT mismatch vs device repack at byte %llu "
                    "(%02x vs %02x)\n", name, (unsigned long long)i, h_tmp[i], h_ali[i]);
            n_fail++; continue;
        }

        /* C: derepack(shipped aligned) == source raw */
        CK(cudaMemcpy(d_ali, h_ali, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_out, 0x5A, nbytes));
        if (ds4_mmq_iq2_xxs_aligned_derepack(d_ali, d_out, (int)d1, (int)d0,
                                             (int)d2, 0) != 0) {
            fprintf(stderr, "%s: derepack failed\n", name); n_fail++; continue;
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_tmp, d_out, nbytes, cudaMemcpyDeviceToHost));
        i = first_diff(h_tmp, h_raw, nbytes);
        if (i != ~0ull) {
            fprintf(stderr, "%s: ROUND-TRIP mismatch vs source raw at byte %llu "
                    "(%02x vs %02x)\n", name, (unsigned long long)i, h_tmp[i], h_raw[i]);
            n_fail++; continue;
        }

        n_ok++;
        total += nbytes;
        printf("  OK %-40s %llu bytes\n", name, nbytes);
        fflush(stdout);
    }
    printf("\nconverted spans verified: %d OK, %d FAIL, %llu bytes (%.2f GB)\n",
           n_ok, n_fail, (unsigned long long)total, (double)total / 1e9);
    printf(n_fail ? "PAYLOAD: FAIL\n" : "PAYLOAD: PASS\n");
    return n_fail ? 1 : 0;
}

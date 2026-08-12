/* iq2_mmq_emit — synthesize a deterministic raw IQ2_XXS (type 16) block stream
 * and write both it and the native ds4q_pack_iq2_xxs_mmq() output, so
 * tests/iq2_mmq_ref.py can run the reference permutation on the SAME raw bytes
 * and the two files can be diffed (make test-iq2-mmq). Proves the native
 * type-43 packer is byte-identical to gguf-tools/repack_iq2_mmq.py, which is
 * the layout's reference implementation and the thing the whole-artifact
 * post-pass used to do. CPU-only; no model, no GPU.
 *
 * The packer is a pure permutation, so any byte pattern exercises it fully --
 * there is no value range to respect, unlike the E8M0 clamp in the mxfp8_lt
 * twin of this test. */
#include "quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s NBLK RAW16_FILE MMQ_FILE\n", argv[0]);
        return 2;
    }
    const int64_t nblk = atoll(argv[1]);
    if (nblk <= 0 || nblk % 32 != 0) {
        fprintf(stderr, "bad nblk %lld (must be a positive multiple of 32 for "
                "the size-preserving layout)\n", (long long)nblk);
        return 2;
    }

    const size_t raw_bytes = (size_t)nblk * 66;
    uint8_t *raw = malloc(raw_bytes);
    if (!raw) { perror("malloc"); return 1; }
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < raw_bytes; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        raw[i] = (uint8_t)(s >> 40);
    }

    const size_t mmq_bytes = ds4q_iq2_xxs_mmq_bytes(nblk);
    if (mmq_bytes != raw_bytes) {
        fprintf(stderr, "size identity broken: mmq %zu != raw %zu\n", mmq_bytes, raw_bytes);
        return 1;
    }
    uint8_t *mmq = malloc(mmq_bytes);
    if (!mmq) { perror("malloc"); return 1; }
    ds4q_pack_iq2_xxs_mmq(raw, mmq, nblk);

    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(raw, 1, raw_bytes, f) != raw_bytes) { perror("write raw"); return 1; }
    fclose(f);
    f = fopen(argv[3], "wb");
    if (!f || fwrite(mmq, 1, mmq_bytes, f) != mmq_bytes) { perror("write mmq"); return 1; }
    fclose(f);

    printf("emit: nblk=%lld raw=%zu mmq=%zu\n", (long long)nblk, raw_bytes, mmq_bytes);
    free(raw);
    free(mmq);
    return 0;
}

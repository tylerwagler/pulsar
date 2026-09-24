/* HOST gate for the EXL3 trellis arithmetic (src/engine/exl3_trellis.h, L245).
 *
 * No model, no device.  The header is pulsar's one authority for the EXL3
 * tile bit layout, the mul1 codebook and the Hadamard basis; this binary
 * grades it against exllamav3 itself:
 *
 *   1. the packer round trip.  A port of `pack_trellis_kernel` (the function
 *      that DEFINES the bit order) packs random states for every integer K,
 *      and the header's state windows must read them back -- checked against
 *      an independent bit-array construction of the stream, not against the
 *      packer's own arithmetic.  Runs with no vectors at all.
 *   2. W_hat byte-exact.  Every `*.exl3v` vector in the directory holds a
 *      128x128 block: the trellis words as stored, suh, svh, and `what` = what
 *      exllamav3's `reconstruct` produced on a GPU for those words (fp16).
 *      Real blocks come from the public MiaAI V4.1 checkpoint (K=3 and K=2),
 *      random-word blocks cover every rate the extension accepts.  A single
 *      differing fp16 is a FAIL.
 *   3. the codebook table, when `codebook-mul1.bin` is present: all 65536
 *      states' fp16 values, byte-exact.
 *   4. the rotation.  W = diag(suh) H128 W_hat H128 diag(svh) computed here in
 *      double is compared with the vector's `w_fused` (exllamav3's fused
 *      reconstruct_had_slice) at the tolerance exllamav3's own test holds that
 *      kernel to (2e-3 of the block's max).  This is the reference the device
 *      rotations will be graded against; it is a tolerance, not a byte check,
 *      because their kernel accumulates in fp16.
 *   5. a mutation: one flipped stream bit in a real block must change W_hat.
 *      A gate that passes an all-zero comparison is measuring nothing.
 *
 * Vector format (little-endian; producer: pulsar-notes research, the
 * exl3_goldens.py script on sparky): 32-byte header {char magic[8]="EXL3VEC1";
 * u32 k_tiles, n_tiles, words_per_tile, k_x2, flags, reserved}; then trellis
 * i16[k_tiles][n_tiles][words], suh f16[k_tiles*16], svh f16[n_tiles*16],
 * what f16[k_tiles*16][n_tiles*16], w_fused (same).  flags bit0 = real
 * checkpoint block, bit1 = mul1 codebook (the only one pulsar accepts).
 */

#include <dirent.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "exl3_trellis.h"

static unsigned g_checks, g_failures;

static void fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("  FAIL: ", stdout);
    vprintf(fmt, ap);
    fputc('\n', stdout);
    va_end(ap);
    g_failures++;
}

/* ------------------------------------------------------------------------ */
/* 1. the packer round trip                                                  */

/* Port of exllamav3 quant/pack.cu pack_trellis_kernel<K>: 16 spans of 16
 * positions, each position's low K bits appended MSB-first into uint16 words,
 * then the halves of every uint32 swapped on store. */
static void pack_tile_reference(const uint16_t states[EXL3_TILE_WEIGHTS], int K, uint16_t *packed) {
    const int packed_size = 256 * K / 16;
    uint16_t s_packed[128];
    for (int t = 0; t < 16; t++) {
        int i = 16 * t, j = K * t, k = 32;
        uint32_t buf = 0;
        for (int n = 0; n < 16; n++) {
            uint32_t v = states[i] & ((1u << K) - 1u);
            k -= K;
            buf |= v << k;
            if (k <= 16) {
                s_packed[j] = (uint16_t)(buf >> 16);
                buf <<= 16;
                k += 16;
                j++;
            }
            i++;
        }
    }
    for (int t = 0; t < packed_size / 2; t++) { /* SWAP16 */
        packed[2 * t] = s_packed[2 * t + 1];
        packed[2 * t + 1] = s_packed[2 * t];
    }
}

static uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}

static void check_pack_round_trip(void) {
    for (int K = 1; K <= 8; K++) {
        const int k2 = 2 * K;
        uint16_t states[EXL3_TILE_WEIGHTS], packed[128];
        uint8_t stream[256 * 8];
        uint32_t rng = 0x9E3779B9u * (uint32_t)K;
        for (int p = 0; p < EXL3_TILE_WEIGHTS; p++) states[p] = (uint16_t)xorshift32(&rng);
        pack_tile_reference(states, K, packed);
        /* independent construction: stream bit p*K + b = bit (K-1-b) of state p */
        for (int p = 0; p < EXL3_TILE_WEIGHTS; p++)
            for (int b = 0; b < K; b++) stream[p * K + b] = (uint8_t)((states[p] >> (K - 1 - b)) & 1u);
        const int total = 256 * K;
        unsigned bad = 0;
        for (int p = 0; p < EXL3_TILE_WEIGHTS; p++) {
            uint32_t want = 0;
            for (int b = 0; b < 16; b++) {
                int s = (p + 1) * K - 16 + b;
                s = ((s % total) + total) % total;
                want = (want << 1) | stream[s];
            }
            const uint32_t got = exl3_tile_state(packed, k2, p);
            if (got != want || (got & ((1u << K) - 1u)) != (states[p] & ((1u << K) - 1u))) bad++;
        }
        g_checks++;
        if (exl3_words_per_tile(k2) != 256 * K / 16 || exl3_k2_from_words(256 * K / 16) != k2)
            fail("K=%d: word count %d / rate %d disagree with the packer", K,
                 exl3_words_per_tile(k2), exl3_k2_from_words(256 * K / 16));
        if (bad) fail("K=%d: %u of 256 state windows differ from the packed stream", K, bad);
        else printf("  pack round trip K=%d: 256/256 windows\n", K);
    }
    /* the half-integer widths and the refusals */
    g_checks++;
    if (exl3_k2_from_words(40) != 5 || exl3_k2_from_words(24) != 3 || exl3_k2_from_words(56) != 7 ||
        exl3_k2_from_words(72) != 0 || exl3_k2_from_words(144) != 0 || exl3_k2_from_words(20) != 0 ||
        exl3_k2_from_words(0) != 0)
        fail("rate-from-width table wrong (40->5 24->3 56->7; 72, 144, 20, 0 -> refused)");
    /* the half-integer window: K=2.5 spends 5 bits per pair, even ends at 2, odd at 5 */
    if (exl3_state_end_bit(5, 0) != 2 || exl3_state_end_bit(5, 1) != 5 || exl3_state_end_bit(5, 2) != 7 ||
        exl3_state_end_bit(5, 255) != 128 * 5 || exl3_stream_bits(5) != 128 * 5)
        fail("half-integer window ends wrong");
}

/* ------------------------------------------------------------------------ */
/* 2..5. the vectors                                                         */

struct vec {
    char name[256];
    uint32_t k_tiles, n_tiles, words, k2, flags;
    const uint16_t *trellis, *suh, *svh, *what, *w_fused;
    uint8_t *buf;
    size_t size;
};

static bool load_vec(const char *dir, const char *name, struct vec *v) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fail("%s: cannot open", name); return false; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 32) { fail("%s: %ld bytes, no header", name, n); fclose(f); return false; }
    v->buf = (uint8_t *)malloc((size_t)n);
    v->size = (size_t)n;
    if (fread(v->buf, 1, (size_t)n, f) != (size_t)n) { fail("%s: short read", name); fclose(f); return false; }
    fclose(f);
    snprintf(v->name, sizeof v->name, "%s", name);
    if (memcmp(v->buf, "EXL3VEC1", 8) != 0) { fail("%s: bad magic", name); return false; }
    uint32_t h[6];
    memcpy(h, v->buf + 8, sizeof h);
    v->k_tiles = h[0]; v->n_tiles = h[1]; v->words = h[2]; v->k2 = h[3]; v->flags = h[4];
    if (!(v->flags & 2u)) { fail("%s: not the mul1 codebook (flags 0x%x)", name, v->flags); return false; }
    if (!exl3_k2_valid((int)v->k2) || exl3_words_per_tile((int)v->k2) != (int)v->words ||
        exl3_k2_from_words((int)v->words) != (int)v->k2) {
        fail("%s: k2=%u words=%u is not an EXL3 rate/width pair", name, v->k2, v->words);
        return false;
    }
    const size_t tiles = (size_t)v->k_tiles * v->n_tiles;
    const size_t rows = 16u * v->k_tiles, cols = 16u * v->n_tiles;
    const size_t want = 32 + 2 * (tiles * v->words + rows + cols + 2 * rows * cols);
    if (want != v->size) { fail("%s: size %zu, expected %zu", name, v->size, want); return false; }
    const uint16_t *p = (const uint16_t *)(v->buf + 32);
    v->trellis = p; p += tiles * v->words;
    v->suh = p; p += rows;
    v->svh = p; p += cols;
    v->what = p; p += rows * cols;
    v->w_fused = p;
    return true;
}

/* W_hat for the whole block, row-major (rows, cols), fp16 bits. */
static void dequant_block(const struct vec *v, uint16_t *out) {
    const int cols = 16 * (int)v->n_tiles;
    for (uint32_t kt = 0; kt < v->k_tiles; kt++)
        for (uint32_t nt = 0; nt < v->n_tiles; nt++) {
            uint16_t tile[EXL3_TILE_WEIGHTS];
            exl3_tile_dequant(v->trellis + ((size_t)kt * v->n_tiles + nt) * v->words, (int)v->k2, tile);
            for (int r = 0; r < 16; r++)
                memcpy(out + ((size_t)kt * 16 + r) * cols + nt * 16, tile + r * 16, 16 * sizeof(uint16_t));
        }
}

static void check_vec(struct vec *v, bool *mutated_once) {
    const size_t rows = 16u * v->k_tiles, cols = 16u * v->n_tiles;
    uint16_t *what = (uint16_t *)malloc(rows * cols * sizeof(uint16_t));
    dequant_block(v, what);

    /* 2. byte-exact W_hat */
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < rows * cols; i++)
        if (what[i] != v->what[i]) { if (!bad) first = i; bad++; }
    g_checks++;
    if (bad)
        fail("%s: W_hat differs in %zu of %zu weights; first at (%zu,%zu): ours 0x%04x theirs 0x%04x",
             v->name, bad, rows * cols, first / cols, first % cols, what[first], v->what[first]);

    /* 4. the rotation, in double, against their fused kernel */
    if (rows % EXL3_HAD_BLOCK == 0 && cols % EXL3_HAD_BLOCK == 0) {
        double *w = (double *)malloc(rows * cols * sizeof(double));
        for (size_t i = 0; i < rows * cols; i++) w[i] = f16_to_f32(v->what[i]);
        double col_v[EXL3_HAD_BLOCK];
        for (size_t c = 0; c < cols; c++)                 /* H on the left: down each column, per 128-row block */
            for (size_t r0 = 0; r0 < rows; r0 += EXL3_HAD_BLOCK) {
                for (int i = 0; i < EXL3_HAD_BLOCK; i++) col_v[i] = w[(r0 + i) * cols + c];
                exl3_had128(col_v);
                for (int i = 0; i < EXL3_HAD_BLOCK; i++) w[(r0 + i) * cols + c] = col_v[i];
            }
        for (size_t r = 0; r < rows; r++)                 /* H on the right: along each row, per 128-col block */
            for (size_t c0 = 0; c0 < cols; c0 += EXL3_HAD_BLOCK) exl3_had128(w + r * cols + c0);
        double max_abs = 0, max_err = 0;
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++) {
                const double ref = w[r * cols + c] * f16_to_f32(v->suh[r]) * f16_to_f32(v->svh[c]);
                const double theirs = f16_to_f32(v->w_fused[r * cols + c]);
                const double e = fabs(ref - theirs);
                if (fabs(ref) > max_abs) max_abs = fabs(ref);
                if (e > max_err) max_err = e;
            }
        const double rel = max_abs > 0 ? max_err / max_abs : 0;
        g_checks++;
        if (rel >= 2e-3) fail("%s: rotation vs their fused kernel: rel %.2e (limit 2e-3)", v->name, rel);
        printf("  %-32s K=%g %s W_hat %s  rotation rel %.2e\n", v->name, v->k2 / 2.0,
               (v->flags & 1u) ? "real  " : "random", bad ? "DIFFERS" : "exact  ", rel);
        free(w);
    } else {
        printf("  %-32s K=%g %s W_hat %s  (no 128-block, rotation skipped)\n", v->name, v->k2 / 2.0,
               (v->flags & 1u) ? "real  " : "random", bad ? "DIFFERS" : "exact  ");
    }

    /* 5. one mutation on the first real block: a flipped stream bit must move W_hat */
    if (!*mutated_once && (v->flags & 1u) && !bad) {
        *mutated_once = true;
        uint16_t *mut = (uint16_t *)malloc(rows * cols * sizeof(uint16_t));
        ((uint16_t *)v->trellis)[0] ^= 0x0100u; /* stream bit 23 of tile (0,0) */
        dequant_block(v, mut);
        ((uint16_t *)v->trellis)[0] ^= 0x0100u;
        size_t moved = 0;
        for (size_t i = 0; i < 256; i++) /* the flipped bit lives in tile (0,0) */
            if (mut[(i / 16) * cols + (i % 16)] != what[(i / 16) * cols + (i % 16)]) moved++;
        g_checks++;
        if (!moved) fail("%s: mutation did not change W_hat -- the comparison is degenerate", v->name);
        else printf("  mutation on %s: one flipped bit moved %zu weights of the tile\n", v->name, moved);
        free(mut);
    }
    free(what);
}

static void check_codebook(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/codebook-mul1.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("  codebook-mul1.bin absent -- table check skipped\n"); return; }
    static uint16_t theirs[65536];
    const size_t n = fread(theirs, sizeof(uint16_t), 65536, f);
    fclose(f);
    g_checks++;
    if (n != 65536) { fail("codebook-mul1.bin: %zu entries, want 65536", n); return; }
    unsigned bad = 0, first = 0;
    for (unsigned x = 0; x < 65536; x++)
        if (exl3_mul1_decode(x) != theirs[x]) { if (!bad) first = x; bad++; }
    if (bad) fail("codebook: %u of 65536 states differ; first x=0x%04x ours 0x%04x theirs 0x%04x",
                  bad, first, exl3_mul1_decode(first), theirs[first]);
    else printf("  codebook mul1: 65536/65536 states exact\n");
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : NULL;
    printf("exl3-dequant-gate: src/engine/exl3_trellis.h vs exllamav3 (%s)\n", dir ? dir : "no vectors");
    check_pack_round_trip();
    unsigned n_vec = 0, n_real = 0;
    if (dir) {
        check_codebook(dir);
        DIR *d = opendir(dir);
        if (!d) { fail("%s: cannot open directory", dir); }
        else {
            char *names[1024];
            unsigned n = 0;
            struct dirent *e;
            while ((e = readdir(d)) && n < 1024) {
                const size_t len = strlen(e->d_name);
                if (len > 6 && strcmp(e->d_name + len - 6, ".exl3v") == 0) names[n++] = strdup(e->d_name);
            }
            closedir(d);
            qsort(names, n, sizeof names[0], cmp_str);
            bool mutated = false;
            for (unsigned i = 0; i < n; i++) {
                struct vec v;
                memset(&v, 0, sizeof v);
                if (load_vec(dir, names[i], &v)) {
                    check_vec(&v, &mutated);
                    n_vec++;
                    if (v.flags & 1u) n_real++;
                }
                free(v.buf);
                free(names[i]);
            }
            g_checks++;
            if (n_vec == 0) fail("%s: no *.exl3v vectors", dir);
            else if (n_real == 0) fail("%s: no REAL checkpoint block among %u vectors", dir, n_vec);
            else if (!mutated) fail("mutation check never ran");
        }
    }
    printf("exl3-dequant-gate: %u checks, %u failures, %u vectors (%u real): %s\n",
           g_checks, g_failures, n_vec, n_real, g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}

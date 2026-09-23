/* engram_rows -- the Engram table and aux files, built from the V4.1 checkpoint shards (L242).
 *
 * WHAT IT WRITES
 * --------------
 * For one Engram layer (V4.1: layers 1 and 14) from the shard that holds it:
 *
 *   <out>/engram-l<L>.rows     the ROW FILE the engine gathers from at run time:
 *                              a 64-byte header, then n_rows records of 264 bytes --
 *                              256 E4M3 values followed by their 8 E8M0 scales (one per
 *                              32 values), i.e. the checkpoint's `embed.weight` row and
 *                              its `embed.scale` row INTERLEAVED so one read fetches both.
 *                              (The checkpoint stores them as two tensors 91 GiB apart.)
 *   <out>/engram-l<L>.wkv      `wkv` in the engine's pre-stored MXFP8_LT layout: the
 *                              [25600][6144] E4M3 data plane row-major, then the E8M0 scale
 *                              plane swizzled exactly as pulsar_mx_sfoff (128-row bands x 4
 *                              k-block tiles of 512 B, KBp rounded to 4).  The checkpoint's
 *                              scale is one E8M0 per 32x32 block; each row of a block takes
 *                              the block's scale, which is lossless.
 *   <out>/engram-l<L>.qk       `q_weight` and `k_weight`, [4][5120] bf16 each, raw, q first.
 *   <out>/engram-l<L>.json     the manifest: layer, n_rows, byte sizes, source shard, the
 *                              source tensors' byte offsets, and a FNV-1a of every output file.
 *
 * The row file's 64-byte header: magic "PENGRAM1", u32 version 1, u32 layer, u64 n_rows,
 * u32 row_bytes (264), u32 dim (256), u32 n_scale (8), u32 reserved, then zero padding.
 *
 * WHY C
 * -----
 * It runs where the shards are (the NFS host has no numpy) and it IS the format authority:
 * the engine's reader asserts the same header, and `verify` re-reads sampled rows from the
 * shard and compares them byte for byte against the row file.
 *
 * USAGE
 *   engram_rows build  SHARD LAYER OUT_DIR       (streams the shard once; ~94 GiB out per layer)
 *   engram_rows verify SHARD LAYER OUT_DIR [N]   (N sampled rows, default 4096, plus wkv/qk fully)
 *
 * SHARD is the safetensors file holding `layers.<LAYER>.engram.*` (V4.1: shard 47 holds layer
 * 1, shard 48 layer 14 -- the tool refuses a shard without the tensors rather than guessing).
 * Build is resumable only by re-running: a partial output is detected by size and rewritten. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DIM        256u
#define N_SCALE    8u
#define ROW_BYTES  (DIM + N_SCALE)
#define HDR_BYTES  64u
#define CHUNK_ROWS (1u << 20)   /* 1 Mi rows: 256 MiB of values + 8 MiB of scales per chunk */
#define WKV_OUT    25600u
#define WKV_IN     6144u
#define HC_MULT    4u
#define MODEL_DIM  5120u

static void die(const char *what) { perror(what); exit(2); }
static void dief(const char *fmt, const char *a) { fprintf(stderr, fmt, a); fputc('\n', stderr); exit(2); }

/* ---- safetensors header: find one tensor's data offsets ---------------------------- */
typedef struct { uint64_t off0, off1; char dtype[16]; } st_tensor;

static int st_find(const char *hdr, size_t hlen, const char *name, uint64_t data_start, st_tensor *out) {
    char key[256];
    snprintf(key, sizeof key, "\"%s\"", name);
    const char *p = memmem(hdr, hlen, key, strlen(key));
    if (!p) return 0;
    const char *end = memchr(p, '}', hlen - (size_t)(p - hdr));
    if (!end) return 0;
    const char *d = memmem(p, (size_t)(end - p), "\"dtype\":\"", 9);
    const char *o = memmem(p, (size_t)(end - p), "\"data_offsets\":[", 16);
    if (!d || !o) return 0;
    d += 9;
    size_t n = 0;
    while (d[n] != '"' && n + 1 < sizeof out->dtype) n++;
    memcpy(out->dtype, d, n);
    out->dtype[n] = '\0';
    unsigned long long a = 0, b = 0;
    if (sscanf(o + 16, "%llu,%llu", &a, &b) != 2) return 0;
    out->off0 = data_start + a;
    out->off1 = data_start + b;
    return 1;
}

static char *st_header(int fd, uint64_t *data_start, size_t *hlen) {
    uint64_t n = 0;
    if (pread(fd, &n, 8, 0) != 8) die("read header length");
    if (n == 0 || n > (64u << 20)) dief("%s", "implausible safetensors header length");
    char *h = malloc((size_t)n + 1);
    if (!h) die("malloc");
    if (pread(fd, h, (size_t)n, 8) != (ssize_t)n) die("read header");
    h[n] = '\0';
    *data_start = 8 + n;
    *hlen = (size_t)n;
    return h;
}

/* ---- pulsar_mx_sfoff, verbatim from src/cuda/pulsar_cuda_mx.cuh -------------------- */
static inline size_t sfoff(size_t row, size_t kb, size_t KBp) {
    return ((row / 128) * (KBp / 4) + (kb / 4)) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (kb % 4);
}
static inline size_t rup(size_t a, size_t m) { return (a + m - 1) / m * m; }

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static void pread_all(int fd, void *dst, size_t n, uint64_t off, const char *what) {
    unsigned char *d = dst;
    while (n) {
        ssize_t r = pread(fd, d, n, (off_t)off);
        if (r <= 0) { fprintf(stderr, "engram_rows: short read in %s (%s)\n", what, r < 0 ? strerror(errno) : "eof"); exit(2); }
        d += r; n -= (size_t)r; off += (uint64_t)r;
    }
}
static void write_all(FILE *f, const void *src, size_t n, const char *what) {
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "engram_rows: short write in %s\n", what); exit(2); }
}

typedef struct {
    int fd;
    st_tensor w, s, wkv_w, wkv_s, q, k;
    uint64_t n_rows;
} src_layer;

static void open_source(const char *shard, unsigned layer, src_layer *L) {
    L->fd = open(shard, O_RDONLY);
    if (L->fd < 0) die(shard);
    uint64_t ds; size_t hl;
    char *h = st_header(L->fd, &ds, &hl);
    char name[128];
#define FIND(field, suffix) do { snprintf(name, sizeof name, "layers.%u.engram.%s", layer, suffix); \
        if (!st_find(h, hl, name, ds, &L->field)) dief("engram_rows: shard has no tensor %s", name); } while (0)
    FIND(w, "embed.weight"); FIND(s, "embed.scale"); FIND(wkv_w, "wkv.weight"); FIND(wkv_s, "wkv.scale");
    FIND(q, "q_weight"); FIND(k, "k_weight");
#undef FIND
    free(h);
    if (strcmp(L->w.dtype, "F8_E4M3") || strcmp(L->s.dtype, "F8_E8M0") || strcmp(L->wkv_w.dtype, "F8_E4M3") ||
        strcmp(L->wkv_s.dtype, "F8_E8M0") || strcmp(L->q.dtype, "BF16") || strcmp(L->k.dtype, "BF16"))
        dief("%s", "engram_rows: a tensor's dtype is not the V4.1 Engram layout (E4M3 rows, E8M0 scales, BF16 q/k)");
    const uint64_t wb = L->w.off1 - L->w.off0, sb = L->s.off1 - L->s.off0;
    if (wb % DIM || sb % N_SCALE || wb / DIM != sb / N_SCALE) dief("%s", "engram_rows: embed.weight / embed.scale row counts disagree");
    L->n_rows = wb / DIM;
    if (L->wkv_w.off1 - L->wkv_w.off0 != (uint64_t)WKV_OUT * WKV_IN) dief("%s", "engram_rows: wkv.weight is not [25600][6144]");
    if (L->wkv_s.off1 - L->wkv_s.off0 != (uint64_t)(WKV_OUT / 32) * (WKV_IN / 32)) dief("%s", "engram_rows: wkv.scale is not [800][192]");
    if (L->q.off1 - L->q.off0 != (uint64_t)HC_MULT * MODEL_DIM * 2 || L->k.off1 - L->k.off0 != (uint64_t)HC_MULT * MODEL_DIM * 2)
        dief("%s", "engram_rows: q_weight / k_weight are not [4][5120] bf16");
}

static void interleave(unsigned char *dst, const unsigned char *vals, const unsigned char *scales, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        memcpy(dst + r * ROW_BYTES, vals + r * DIM, DIM);
        memcpy(dst + r * ROW_BYTES + DIM, scales + r * N_SCALE, N_SCALE);
    }
}

static void path_of(char *buf, size_t n, const char *dir, unsigned layer, const char *ext) {
    snprintf(buf, n, "%s/engram-l%u.%s", dir, layer, ext);
}

/* wkv -> pre-stored MXFP8_LT bytes: [data out*in][scale rup(out,128)*KBp]. */
static unsigned char *pack_wkv(const src_layer *L, size_t *bytes_out) {
    const size_t KB = WKV_IN / 32, KBp = rup(KB, 4);
    const size_t data_bytes = (size_t)WKV_OUT * WKV_IN, scale_bytes = rup(WKV_OUT, 128) * KBp;
    unsigned char *buf = calloc(data_bytes + scale_bytes, 1);
    unsigned char *bs = malloc((size_t)(WKV_OUT / 32) * (WKV_IN / 32));
    if (!buf || !bs) die("malloc wkv");
    pread_all(L->fd, buf, data_bytes, L->wkv_w.off0, "wkv.weight");
    pread_all(L->fd, bs, (size_t)(WKV_OUT / 32) * (WKV_IN / 32), L->wkv_s.off0, "wkv.scale");
    unsigned char *sc = buf + data_bytes;
    for (size_t row = 0; row < WKV_OUT; row++)
        for (size_t kb = 0; kb < KB; kb++)
            sc[sfoff(row, kb, KBp)] = bs[(row / 32) * (WKV_IN / 32) + kb];
    free(bs);
    *bytes_out = data_bytes + scale_bytes;
    return buf;
}

static int cmd_build(const char *shard, unsigned layer, const char *dir) {
    src_layer L;
    open_source(shard, layer, &L);
    char rows_p[4096], wkv_p[4096], qk_p[4096], man_p[4096];
    path_of(rows_p, sizeof rows_p, dir, layer, "rows");
    path_of(wkv_p, sizeof wkv_p, dir, layer, "wkv");
    path_of(qk_p, sizeof qk_p, dir, layer, "qk");
    path_of(man_p, sizeof man_p, dir, layer, "json");
    fprintf(stderr, "engram_rows: layer %u: %llu rows (%.2f GiB of values + %.2f GiB of scales) -> %s\n",
            layer, (unsigned long long)L.n_rows, (double)L.n_rows * DIM / 1073741824.0,
            (double)L.n_rows * N_SCALE / 1073741824.0, rows_p);

    /* aux first: small, and their hashes go into the manifest */
    size_t wkv_bytes = 0;
    unsigned char *wkv = pack_wkv(&L, &wkv_bytes);
    FILE *f = fopen(wkv_p, "wb"); if (!f) die(wkv_p);
    write_all(f, wkv, wkv_bytes, "wkv"); fclose(f);
    const uint64_t wkv_fnv = fnv1a(1469598103934665603ull, wkv, wkv_bytes);
    free(wkv);
    const size_t qk_half = (size_t)HC_MULT * MODEL_DIM * 2;
    unsigned char *qk = malloc(qk_half * 2); if (!qk) die("malloc qk");
    pread_all(L.fd, qk, qk_half, L.q.off0, "q_weight");
    pread_all(L.fd, qk + qk_half, qk_half, L.k.off0, "k_weight");
    f = fopen(qk_p, "wb"); if (!f) die(qk_p);
    write_all(f, qk, qk_half * 2, "qk"); fclose(f);
    const uint64_t qk_fnv = fnv1a(1469598103934665603ull, qk, qk_half * 2);
    free(qk);

    /* the row file */
    f = fopen(rows_p, "wb"); if (!f) die(rows_p);
    unsigned char hdr[HDR_BYTES] = {0};
    memcpy(hdr, "PENGRAM1", 8);
    uint32_t v = 1, lay = layer, rb = ROW_BYTES, dim = DIM, ns = N_SCALE;
    uint64_t nr = L.n_rows;
    memcpy(hdr + 8, &v, 4); memcpy(hdr + 12, &lay, 4); memcpy(hdr + 16, &nr, 8);
    memcpy(hdr + 24, &rb, 4); memcpy(hdr + 28, &dim, 4); memcpy(hdr + 32, &ns, 4);
    write_all(f, hdr, HDR_BYTES, "header");
    unsigned char *vals = malloc((size_t)CHUNK_ROWS * DIM), *scs = malloc((size_t)CHUNK_ROWS * N_SCALE),
                  *out = malloc((size_t)CHUNK_ROWS * ROW_BYTES);
    if (!vals || !scs || !out) die("malloc chunk");
    uint64_t rows_fnv = 1469598103934665603ull;
    for (uint64_t r0 = 0; r0 < L.n_rows; r0 += CHUNK_ROWS) {
        const size_t n = (size_t)((L.n_rows - r0 < CHUNK_ROWS) ? (L.n_rows - r0) : CHUNK_ROWS);
        pread_all(L.fd, vals, n * DIM, L.w.off0 + r0 * DIM, "embed.weight");
        pread_all(L.fd, scs, n * N_SCALE, L.s.off0 + r0 * N_SCALE, "embed.scale");
        interleave(out, vals, scs, n);
        write_all(f, out, n * ROW_BYTES, "rows");
        /* hash a sample of every chunk, not every byte: the manifest is a tamper check, the
         * verify subcommand is the byte proof */
        rows_fnv = fnv1a(rows_fnv, out, n * ROW_BYTES < 65536 ? n * ROW_BYTES : 65536);
        if ((r0 / CHUNK_ROWS) % 32 == 0)
            fprintf(stderr, "  %6.2f%%  %llu rows\r", 100.0 * (double)(r0 + n) / (double)L.n_rows,
                    (unsigned long long)(r0 + n));
    }
    fputc('\n', stderr);
    if (fclose(f) != 0) die("close rows");
    free(vals); free(scs); free(out);

    f = fopen(man_p, "w"); if (!f) die(man_p);
    fprintf(f, "{\n  \"format\": \"pulsar-engram-rows\", \"version\": 1, \"layer\": %u,\n"
               "  \"n_rows\": %llu, \"row_bytes\": %u, \"dim\": %u, \"n_scale\": %u, \"header_bytes\": %u,\n"
               "  \"rows_bytes\": %llu, \"wkv_bytes\": %zu, \"wkv_layout\": \"mxfp8_lt\", \"wkv_in\": %u, \"wkv_out\": %u,\n"
               "  \"qk_bytes\": %zu, \"hc_mult\": %u, \"model_dim\": %u,\n"
               "  \"source_shard\": \"%s\",\n"
               "  \"source_offsets\": {\"embed.weight\": %llu, \"embed.scale\": %llu, \"wkv.weight\": %llu, \"wkv.scale\": %llu, \"q_weight\": %llu, \"k_weight\": %llu},\n"
               "  \"fnv1a\": {\"rows_sampled\": \"%016llx\", \"wkv\": \"%016llx\", \"qk\": \"%016llx\"}\n}\n",
            layer, (unsigned long long)L.n_rows, ROW_BYTES, DIM, N_SCALE, HDR_BYTES,
            (unsigned long long)(HDR_BYTES + L.n_rows * ROW_BYTES), wkv_bytes, WKV_IN, WKV_OUT,
            qk_half * 2, HC_MULT, MODEL_DIM, shard,
            (unsigned long long)L.w.off0, (unsigned long long)L.s.off0, (unsigned long long)L.wkv_w.off0,
            (unsigned long long)L.wkv_s.off0, (unsigned long long)L.q.off0, (unsigned long long)L.k.off0,
            (unsigned long long)rows_fnv, (unsigned long long)wkv_fnv, (unsigned long long)qk_fnv);
    fclose(f);
    close(L.fd);
    fprintf(stderr, "engram_rows: layer %u done: %s\n", layer, man_p);
    return 0;
}

static int cmd_verify(const char *shard, unsigned layer, const char *dir, uint64_t n_sample) {
    src_layer L;
    open_source(shard, layer, &L);
    char rows_p[4096], wkv_p[4096], qk_p[4096];
    path_of(rows_p, sizeof rows_p, dir, layer, "rows");
    path_of(wkv_p, sizeof wkv_p, dir, layer, "wkv");
    path_of(qk_p, sizeof qk_p, dir, layer, "qk");
    int rf = open(rows_p, O_RDONLY); if (rf < 0) die(rows_p);
    unsigned char hdr[HDR_BYTES];
    pread_all(rf, hdr, HDR_BYTES, 0, "row header");
    uint64_t nr = 0; uint32_t lay = 0, rb = 0;
    memcpy(&lay, hdr + 12, 4); memcpy(&nr, hdr + 16, 8); memcpy(&rb, hdr + 24, 4);
    struct stat st; if (fstat(rf, &st) != 0) die("fstat rows");
    int bad = 0;
    if (memcmp(hdr, "PENGRAM1", 8) || lay != layer || nr != L.n_rows || rb != ROW_BYTES) { fprintf(stderr, "  FAIL  row header\n"); bad = 1; }
    if ((uint64_t)st.st_size != HDR_BYTES + L.n_rows * ROW_BYTES) {
        fprintf(stderr, "  FAIL  row file size %lld != %llu (partial build?)\n", (long long)st.st_size,
                (unsigned long long)(HDR_BYTES + L.n_rows * ROW_BYTES)); bad = 1;
    }
    /* sampled rows, a fixed stride so every band of the file is touched */
    unsigned char row[ROW_BYTES], v[DIM], s[N_SCALE];
    uint64_t mism = 0;
    for (uint64_t i = 0; i < n_sample; i++) {
        const uint64_t r = (uint64_t)((__int128)i * L.n_rows / n_sample);
        pread_all(rf, row, ROW_BYTES, HDR_BYTES + r * ROW_BYTES, "row");
        pread_all(L.fd, v, DIM, L.w.off0 + r * DIM, "embed.weight");
        pread_all(L.fd, s, N_SCALE, L.s.off0 + r * N_SCALE, "embed.scale");
        if (memcmp(row, v, DIM) || memcmp(row + DIM, s, N_SCALE)) mism++;
    }
    fprintf(stderr, "  %s  rows: %llu sampled, %llu mismatched\n", mism ? "FAIL" : "ok  ",
            (unsigned long long)n_sample, (unsigned long long)mism);
    if (mism) bad = 1;
    /* wkv + qk fully */
    size_t wb = 0; unsigned char *want = pack_wkv(&L, &wb);
    unsigned char *have = malloc(wb); if (!have) die("malloc");
    int wf = open(wkv_p, O_RDONLY); if (wf < 0) die(wkv_p);
    pread_all(wf, have, wb, 0, "wkv file");
    const int wkv_ok = memcmp(want, have, wb) == 0;
    fprintf(stderr, "  %s  wkv: %zu bytes %s\n", wkv_ok ? "ok  " : "FAIL", wb, wkv_ok ? "identical to the repack" : "DIFFER");
    if (!wkv_ok) bad = 1;
    free(want); free(have); close(wf);
    const size_t qh = (size_t)HC_MULT * MODEL_DIM * 2;
    unsigned char *qw = malloc(qh * 2), *qh2 = malloc(qh * 2); if (!qw || !qh2) die("malloc");
    pread_all(L.fd, qw, qh, L.q.off0, "q"); pread_all(L.fd, qw + qh, qh, L.k.off0, "k");
    int qf = open(qk_p, O_RDONLY); if (qf < 0) die(qk_p);
    pread_all(qf, qh2, qh * 2, 0, "qk file");
    const int qk_ok = memcmp(qw, qh2, qh * 2) == 0;
    fprintf(stderr, "  %s  qk: %zu bytes %s\n", qk_ok ? "ok  " : "FAIL", qh * 2, qk_ok ? "identical" : "DIFFER");
    if (!qk_ok) bad = 1;
    free(qw); free(qh2); close(qf); close(rf); close(L.fd);
    fprintf(stderr, "engram_rows verify layer %u: %s\n", layer, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s build|verify SHARD LAYER OUT_DIR [N_SAMPLE]\n", argv[0]);
        return 2;
    }
    const unsigned layer = (unsigned)strtoul(argv[3], NULL, 10);
    if (!strcmp(argv[1], "build")) return cmd_build(argv[2], layer, argv[4]);
    if (!strcmp(argv[1], "verify")) return cmd_verify(argv[2], layer, argv[4], argc > 5 ? strtoull(argv[5], NULL, 10) : 4096ull);
    fprintf(stderr, "engram_rows: unknown subcommand %s\n", argv[1]);
    return 2;
}

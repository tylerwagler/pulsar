/*
 * dsq_sha256.c — SHA-256 (FIPS 180-4), self-contained.
 *
 * Here so the quantizer keeps its "plain C, no external deps" property. Used to
 * identify the imatrix by CONTENT in the artifact metadata: the GGUF used to
 * record the imatrix's PATH, which says where someone's disk was rather than
 * which imatrix it was, so it could not distinguish a real reproduction from a
 * coincidence — and it guaranteed a metadata diff between any two machines.
 */
#include "dsq_internal.h"

typedef struct {
    uint32_t h[8];
    uint64_t len;      /* total message bytes fed so far */
    uint8_t  buf[64];
    size_t   buf_len;
} sha256_ctx;

static const uint32_t sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_block(sha256_ctx *c, const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
               ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror32(w[i-15], 7) ^ ror32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ror32(w[i-2], 17) ^ ror32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c->h[0], b = c->h[1], cc = c->h[2], d = c->h[3];
    uint32_t e = c->h[4], f = c->h[5], g = c->h[6], h = c->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + sha256_k[i] + w[i];
        uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d;
    c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}

static void sha256_init(sha256_ctx *c) {
    c->h[0] = 0x6a09e667u; c->h[1] = 0xbb67ae85u; c->h[2] = 0x3c6ef372u; c->h[3] = 0xa54ff53au;
    c->h[4] = 0x510e527fu; c->h[5] = 0x9b05688cu; c->h[6] = 0x1f83d9abu; c->h[7] = 0x5be0cd19u;
    c->len = 0;
    c->buf_len = 0;
}

static void sha256_update(sha256_ctx *c, const uint8_t *p, size_t n) {
    c->len += n;
    while (n) {
        if (c->buf_len == 0 && n >= 64) {
            sha256_block(c, p);
            p += 64; n -= 64;
            continue;
        }
        size_t take = 64 - c->buf_len;
        if (take > n) take = n;
        memcpy(c->buf + c->buf_len, p, take);
        c->buf_len += take;
        p += take; n -= take;
        if (c->buf_len == 64) { sha256_block(c, c->buf); c->buf_len = 0; }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    /* Snapshot the length BEFORE padding: the padding bytes go through
     * sha256_update, which would otherwise fold them into the length field. */
    const uint64_t bits = c->len * 8;
    static const uint8_t pad0 = 0x80, zero = 0x00;
    sha256_update(c, &pad0, 1);
    while (c->buf_len != 56) sha256_update(c, &zero, 1);
    for (int i = 0; i < 8; i++) c->buf[56 + i] = (uint8_t)(bits >> (56 - 8*i));
    sha256_block(c, c->buf);
    c->buf_len = 0;
    for (int i = 0; i < 8; i++) {
        out[4*i + 0] = (uint8_t)(c->h[i] >> 24);
        out[4*i + 1] = (uint8_t)(c->h[i] >> 16);
        out[4*i + 2] = (uint8_t)(c->h[i] >> 8);
        out[4*i + 3] = (uint8_t)(c->h[i]);
    }
}

static void sha256_hex(const uint8_t digest[32], char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(out + 2*i, 3, "%02x", digest[i]);
    out[64] = '\0';
}

/* FIPS 180-4 vectors, checked once before the first real hash. A silently wrong
 * SHA-256 would stamp a confident, meaningless identity into every artifact. */
static void sha256_selftest(void) {
    static bool done = false;
    if (done) return;
    struct { const char *msg; const char *want; } v[] = {
        { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    };
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
        sha256_ctx c;
        sha256_init(&c);
        sha256_update(&c, (const uint8_t *)v[i].msg, strlen(v[i].msg));
        uint8_t d[32];
        char hex[65];
        sha256_final(&c, d);
        sha256_hex(d, hex);
        if (strcmp(hex, v[i].want) != 0) {
            fprintf(stderr, "error: sha256 self-test failed\n  got  %s\n  want %s\n",
                    hex, v[i].want);
            exit(1);
        }
    }
    /* One multi-block case, to exercise the buffered path rather than only
     * messages that fit in a single 64-byte block. */
    {
        sha256_ctx c;
        sha256_init(&c);
        uint8_t chunk[1000];
        memset(chunk, 'a', sizeof(chunk));
        for (int i = 0; i < 1000; i++) sha256_update(&c, chunk, sizeof(chunk));
        uint8_t d[32];
        char hex[65];
        sha256_final(&c, d);
        sha256_hex(d, hex);
        const char *want = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
        if (strcmp(hex, want) != 0) {
            fprintf(stderr, "error: sha256 self-test failed (1M 'a')\n  got  %s\n  want %s\n",
                    hex, want);
            exit(1);
        }
    }
    done = true;
}

char *sha256_file_hex(const char *path) {
    sha256_selftest();
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open for sha256", path);
    sha256_ctx c;
    sha256_init(&c);
    uint8_t *buf = xmalloc(1u << 20);
    size_t n;
    while ((n = fread(buf, 1, 1u << 20, fp)) > 0) sha256_update(&c, buf, n);
    if (ferror(fp)) die_errno("read for sha256", path);
    free(buf);
    fclose(fp);
    uint8_t digest[32];
    sha256_final(&c, digest);
    char *hex = xmalloc(65);
    sha256_hex(digest, hex);
    return hex;
}

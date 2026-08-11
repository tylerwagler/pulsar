#include "dsq_internal.h"

void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

char *path_join(const char *a, const char *b) {
    const size_t na = strlen(a);
    const size_t nb = strlen(b);
    const bool slash = na && a[na - 1] == '/';
    char *out = xmalloc(na + (slash ? 0 : 1) + nb + 1);
    memcpy(out, a, na);
    size_t pos = na;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, nb + 1);
    return out;
}

bool str_starts(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

bool str_ends(const char *s, const char *suffix) {
    const size_t ns = strlen(s);
    const size_t nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open", path);
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("seek", path);
    off_t n = ftello(fp);
    if (n < 0) die_errno("tell", path);
    if (fseeko(fp, 0, SEEK_SET) != 0) die_errno("seek", path);
    char *buf = xmalloc((size_t)n + 1);
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("read", path);
    buf[n] = '\0';
    fclose(fp);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

/* Bytes between the cursor and EOF.  File-declared lengths must be checked
 * against this before allocation (upstream ds4 a968c08/c7689db): a corrupt
 * length near UINT64_MAX otherwise wraps the +1 in the alloc and the
 * following fread heap-overflows a tiny buffer with the rest of the file. */
uint64_t bytes_remaining_fp(FILE *fp, const char *what) {
    off_t pos = ftello(fp);
    if (pos < 0) die_errno("ftello", what);
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("fseeko", what);
    off_t end = ftello(fp);
    if (end < 0) die_errno("ftello", what);
    if (fseeko(fp, pos, SEEK_SET) != 0) die_errno("fseeko", what);
    return end >= pos ? (uint64_t)(end - pos) : 0;
}

uint64_t read_checked_len_fp(FILE *fp, const char *what) {
    uint64_t n = read_u64_le_fp(fp, what);
    if (n > bytes_remaining_fp(fp, what)) {
        fprintf(stderr, "error: %s %llu exceeds the remaining file bytes\n",
                what, (unsigned long long)n);
        exit(1);
    }
    return n;
}

uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

int32_t read_i32_fp(FILE *fp, const char *what) {
    int32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

int64_t load_i64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return (int64_t)v;
}


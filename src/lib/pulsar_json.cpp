/* pulsar_json.cpp -- see pulsar_json.h.
 *
 * Moved from src/server/util.cpp so the engine can use it too.  The only change
 * is the scratch buffer: the original accumulated into the server's `buf`,
 * which carries chat-text span tracking and belongs to the server, so this uses
 * a local growable buffer instead.  Behaviour is otherwise identical, including
 * the two hardening rules the server relies on: *out is nulled before any work
 * (a caller that reparses in place must not keep a dangling pointer), and the
 * decoded bytes are malloc'd and owned by the caller.
 */
#include "pulsar_json.h"

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

/* ---- local scratch buffer (replaces the server's `buf`) ---- */
typedef struct {
    char *p;
    size_t len;
    size_t cap;
} json_buf;

static bool jb_reserve(json_buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + extra + 1) {
        if (cap > (size_t)-1 / 2) return false;
        cap *= 2;
    }
    char *np = (char *)realloc(b->p, cap);
    if (!np) return false;
    b->p = np;
    b->cap = cap;
    return true;
}

static bool jb_putc(json_buf *b, char c) {
    if (!jb_reserve(b, 1)) return false;
    b->p[b->len++] = c;
    return true;
}

static void utf8_put(json_buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        (void)jb_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        (void)jb_putc(b, (char)(0xc0 | (cp >> 6)));
        (void)jb_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        (void)jb_putc(b, (char)(0xe0 | (cp >> 12)));
        (void)jb_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        (void)jb_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        (void)jb_putc(b, (char)(0xf0 | (cp >> 18)));
        (void)jb_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        (void)jb_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        (void)jb_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

bool json_string_n(const char **p, char **out, size_t *out_len) {
    /* Null *out up front: callers that reparse in place (`free(x);
     * json_string(&p, &x)`) must not be left holding a dangling pointer on a
     * failure path (upstream ds4 3196149's double-free hardening). */
    *out = NULL;
    if (out_len) *out_len = 0;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    json_buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            if (!jb_putc(&b, (char)c)) goto fail;
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': if (!jb_putc(&b, '"')) goto fail; break;
        case '\\': if (!jb_putc(&b, '\\')) goto fail; break;
        case '/': if (!jb_putc(&b, '/')) goto fail; break;
        case 'b': if (!jb_putc(&b, '\b')) goto fail; break;
        case 'f': if (!jb_putc(&b, '\f')) goto fail; break;
        case 'n': if (!jb_putc(&b, '\n')) goto fail; break;
        case 'r': if (!jb_putc(&b, '\r')) goto fail; break;
        case 't': if (!jb_putc(&b, '\t')) goto fail; break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    if (!jb_reserve(&b, 0)) goto fail;
    b.p[b.len] = '\0';
    if (out_len) *out_len = b.len;
    *out = b.p;
    return true;
fail:
    free(b.p);
    return false;
}

bool json_string(const char **p, char **out) {
    return json_string_n(p, out, NULL);
}

bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    /* strtod accepts nan/inf lexemes; (int)NaN is UB, so fold NaN with the
     * negatives (upstream ds4 8340f35). The INT_MAX clamp handles +inf. */
    if (!(v >= 0)) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) { *out = true; return true; }
    if (json_lit(p, "false")) { *out = false; return true; }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') { (*p)++; return true; }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') { (*p)++; return true; }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

bool json_raw_value(const char **p, char **out) {
    /* Null *out up front, like json_string_n: callers that reparse a
     * duplicate key in place (`free(x); json_raw_value(&p, &x)`) must not be
     * left holding a dangling pointer when the second value is malformed. */
    *out = NULL;
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = (char *)malloc(n + 1);
    if (!s) return false;
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

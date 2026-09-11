#include "pulsar_server_internal.h"

#include <charconv>
#include <cmath>

#include <sys/random.h>


volatile sig_atomic_t g_stop_requested = 0;


volatile sig_atomic_t g_listen_fd = -1;


void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}



void die(const char *msg) {
    fprintf(stderr, "pulsar-server: %s\n", msg);
    exit(1);
}



void *server_xmalloc(size_t n) {
    void *p = (void *)malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}



void *server_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}



char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = (char *)server_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}



bool random_bytes(void *dst, size_t len) {
    unsigned char *p = (unsigned char *)dst;
    /* getrandom() has no fd/chroot failure modes; /dev/urandom stays as the
     * fallback for kernels without the syscall. */
    while (len) {
        ssize_t n = getrandom(p, len, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) break;
        p += (size_t)n;
        len -= (size_t)n;
    }
    if (len == 0) return true;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            return false;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    close(fd);
    return true;
}



char *xstrndup(const char *s, size_t n) {
    char *p = (char *)server_xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}



static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    b->ptr = (char *)server_xrealloc(b->ptr, cap);
    b->cap = cap;
}



void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}



void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}



void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}



void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}



char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}



void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}



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



static void utf8_put(buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
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
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
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
    if (out_len) *out_len = b.len;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
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
    if (json_lit(p, "true")) {
        *out = true;
        return true;
    }
    if (json_lit(p, "false")) {
        *out = false;
        return true;
    }
    return false;
}


static bool json_skip_value_depth(const char **p, int depth);



static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
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
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
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
    char *s = (char *)server_xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}



char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;

    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            buf_putc(&b, (char)c);
        }
    }
    return buf_take(&b);
}



/* ---- Python json.dumps emulation ----------------------------------------
 *
 * The V4.1 reference encoder writes every non-string DSML parameter value
 * with `json.dumps(value, ensure_ascii=False)` after `json.loads`.  Those
 * bytes are what the model was trained on, and a prompt that differs only in
 * ", " versus "," tokenises differently, so the renderer reproduces the
 * spelling exactly: dict/list separators ", " and ": ", strings escaped for
 * backslash, quote and the C0 controls only (\b \f \n \r \t short forms,
 * \u00XX otherwise; non-ASCII raw), integer literals verbatim, floats as
 * repr(): shortest round-trip digits, fixed notation for decimal exponents in
 * (-4, 16], ".0" appended to an integral value, else "d.ddde+XX". */
static void py_json_string(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        default:
            if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
            else buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

static void py_float_repr(buf *b, double v) {
    if (v == 0.0) {
        buf_puts(b, std::signbit(v) ? "-0.0" : "0.0");
        return;
    }
    /* shortest round-trip digits in scientific form: [-]d[.ddd]e[+-]XX */
    char tmp[64];
    auto r = std::to_chars(tmp, tmp + sizeof tmp, v, std::chars_format::scientific);
    if (r.ec != std::errc()) pulsar_die("py_float_repr: to_chars failed");
    *r.ptr = '\0';
    const char *m = tmp;
    if (*m == '-') { buf_putc(b, '-'); m++; }
    const char *e = strchr(m, 'e');
    if (!e) pulsar_die("py_float_repr: no exponent in scientific form");
    char digits[40];
    size_t nd = 0;
    for (const char *q = m; q < e; q++) if (*q != '.') digits[nd++] = *q;
    digits[nd] = '\0';
    const int exp10 = atoi(e + 1);
    const int decpt = exp10 + 1;             /* digits d1..dn stand for 0.d1..dn x 10^decpt */
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            buf_puts(b, "0.");
            for (int i = 0; i < -decpt; i++) buf_putc(b, '0');
            buf_puts(b, digits);
        } else if ((size_t)decpt >= nd) {
            buf_puts(b, digits);
            for (size_t i = nd; i < (size_t)decpt; i++) buf_putc(b, '0');
            buf_puts(b, ".0");
        } else {
            for (size_t i = 0; i < nd; i++) {
                if (i == (size_t)decpt) buf_putc(b, '.');
                buf_putc(b, digits[i]);
            }
        }
    } else {
        buf_putc(b, digits[0]);
        if (nd > 1) { buf_putc(b, '.'); buf_puts(b, digits + 1); }
        buf_printf(b, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    }
}

static bool py_json_value(const char **p, buf *b) {
    json_ws(p);
    const char c = **p;
    if (c == '{') {
        (*p)++;
        buf_putc(b, '{');
        json_ws(p);
        bool first = true;
        while (**p && **p != '}') {
            char *key = NULL;
            size_t klen = 0;
            if (!json_string_n(p, &key, &klen)) return false;
            if (!first) buf_puts(b, ", ");
            first = false;
            py_json_string(b, key, klen);
            free(key);
            json_ws(p);
            if (**p != ':') return false;
            (*p)++;
            buf_puts(b, ": ");
            if (!py_json_value(p, b)) return false;
            json_ws(p);
            if (**p == ',') { (*p)++; json_ws(p); }
        }
        if (**p != '}') return false;
        (*p)++;
        buf_putc(b, '}');
        return true;
    }
    if (c == '[') {
        (*p)++;
        buf_putc(b, '[');
        json_ws(p);
        bool first = true;
        while (**p && **p != ']') {
            if (!first) buf_puts(b, ", ");
            first = false;
            if (!py_json_value(p, b)) return false;
            json_ws(p);
            if (**p == ',') { (*p)++; json_ws(p); }
        }
        if (**p != ']') return false;
        (*p)++;
        buf_putc(b, ']');
        return true;
    }
    if (c == '"') {
        char *str = NULL;
        size_t n = 0;
        if (!json_string_n(p, &str, &n)) return false;
        py_json_string(b, str, n);
        free(str);
        return true;
    }
    if (json_lit(p, "true"))  { buf_puts(b, "true");  return true; }
    if (json_lit(p, "false")) { buf_puts(b, "false"); return true; }
    if (json_lit(p, "null"))  { buf_puts(b, "null");  return true; }
    /* number: an integer literal is Python int -> printed verbatim (modulo
     * "-0" -> "0"); anything with a fraction or exponent is a float -> repr */
    const char *start = *p;
    const char *q = start;
    if (*q == '-') q++;
    bool is_float = false;
    while ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' || *q == '+' || *q == '-') {
        if (*q == '.' || *q == 'e' || *q == 'E') is_float = true;
        q++;
    }
    if (q == start || (q == start + 1 && *start == '-')) return false;
    if (!is_float) {
        if (q - start == 2 && start[0] == '-' && start[1] == '0') buf_putc(b, '0');
        else buf_append(b, start, (size_t)(q - start));
    } else {
        char *end = NULL;
        const double v = strtod(start, &end);
        if (end != q) return false;
        py_float_repr(b, v);
    }
    *p = q;
    return true;
}

char *json_python_dumps_raw_value(const char *json) {
    const char *p = json ? json : "null";
    buf b = {0};
    if (!py_json_value(&p, &b)) {
        buf_free(&b);
        return xstrdup(json ? json : "null");
    }
    json_ws(&p);
    if (*p) {
        buf_free(&b);
        return xstrdup(json);
    }
    return buf_take(&b);
}



bool json_content(const char **p, char **out) {
    /* Null *out up front (see json_string_n): a reparse-in-place caller must
     * not double-free when a duplicate key's second value fails to parse. */
    *out = NULL;
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }

    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) goto fail;
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    if (!json_string(p, &s)) {
                        free(key);
                        goto fail;
                    }
                    buf_puts(&b, s);
                    free(s);
                } else if (!json_skip_value(p)) {
                    free(key);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') goto fail;
            (*p)++;
        } else if (!json_skip_value(p)) {
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}


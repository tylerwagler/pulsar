/** @file
 *  Small server utilities: allocation that dies on failure, the growable buffer, JSON helpers, random ids and
 *  the stop signal.
 */
#include "pulsar_server_internal.h"
#include "pulsar_json.h"

#include <charconv>
#include <cmath>

#include <sys/random.h>


/// Set by stop_signal_handler on the first SIGINT / SIGTERM: the server drains and stops (read by every thread).
volatile sig_atomic_t g_stop_requested = 0;


/// The listening socket for stop_signal_handler to close (-1: none, or already closed).
volatile sig_atomic_t g_listen_fd = -1;


/// @brief SIGINT / SIGTERM handler: the first signal sets g_stop_requested and closes the listener; a second exits
/// at once (status 130).  Async-signal-safe.
/// @param sig the signal (unused)
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



/// @brief Print "pulsar-server: msg" to stderr and exit(1).
/// @param msg the reason
void die(const char *msg) {
    fprintf(stderr, "pulsar-server: %s\n", msg);
    exit(1);
}



/// @brief malloc that dies on failure ("out of memory"); n == 0 allocates 1 byte.
/// @param n bytes
/// @return the block (never NULL; free() it)
void *server_xmalloc(size_t n) {
    void *p = (void *)malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}



/// @brief realloc that dies on failure; n == 0 reallocates to 1 byte.
/// @param p the block (may be NULL)
/// @param n the new size in bytes
/// @return the block (never NULL)
void *server_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}



/// @brief A malloc'd copy of a C string (dies on out of memory, like server_xmalloc).
/// @param s the string (not NULL)
/// @return the copy (never NULL)
char *server_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = (char *)server_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}



/// @brief Fill `dst` with cryptographic randomness: getrandom(), falling back to /dev/urandom.
/// @param dst the buffer
/// @param len bytes to fill
/// @return false when neither source delivered all `len` bytes
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



/// @brief A malloc'd, NUL-terminated copy of the first `n` bytes of `s` (dies on out of memory).
/// @param s the bytes (at least `n`)
/// @param n bytes to copy
/// @return the copy (never NULL)
char *server_xstrndup(const char *s, size_t n) {
    char *p = (char *)server_xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}



/// @brief Grow `b` (doubling, from 256) so `add` more bytes plus the NUL fit; dies on overflow.
/// @param b the buffer
/// @param add bytes about to be appended
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



/// @brief Append `n` bytes and keep the buffer NUL-terminated.
/// @param b the buffer
/// @param p the bytes
/// @param n how many
void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}



/// @brief Append one byte.
/// @param b the buffer
/// @param c the byte
void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}



/** See the declaration in pulsar_server_internal.h; doxygen applies \hidecallergraph per entity.
 *  \hidecallergraph */
/// @param b the buffer
/// @param s a NUL-terminated string, appended without its NUL
void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}



/// @brief Append printf-formatted text (dies when vsnprintf fails).
/// @param b the buffer
/// @param fmt the format, then its arguments
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



/// @brief Take the bytes out of `b` and reset it; the span records are dropped (copy them out first).
/// @param b the buffer (zeroed afterwards)
/// @return the NUL-terminated text, owned by the caller ("" malloc'd for an empty buffer)
char *buf_take(buf *b) {
    /* L223: the client-data ranges are useless without the bytes they describe,
     * so taking the bytes drops them.  A caller that needs them (the chat
     * renderer) copies them out BEFORE this call. */
    free(b->spans);
    if (!b->ptr) { memset(b, 0, sizeof(*b)); return server_xstrdup(""); }
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}



/// @brief Free the bytes and the span records, and zero `b`.
/// @param b the buffer
void buf_free(buf *b) {
    free(b->ptr);
    free(b->spans);
    memset(b, 0, sizeof(*b));
}



/* json_ws / json_lit / json_string[_n] / json_number / json_int / json_bool /
 * json_skip_value / json_raw_value and their helpers moved to
 * src/lib/pulsar_json.cpp so the engine's safetensors reader can share ONE
 * scanner instead of a second, drifting copy.  The only change is the scratch
 * buffer: this file accumulated into `buf`, which carries chat-text span
 * tracking, so the shared version uses a local growable buffer. */


/// @brief One JSON value with every whitespace byte outside strings removed.
/// @param json the value's text (NULL reads as "null")
/// @return malloc'd text; a copy of the input when it does not scan as a value
char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return server_xstrdup(json ? json : "null");
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
/// @brief Append `n` bytes as a JSON string the way Python's json.dumps(ensure_ascii=False) spells it.
/// @param b the buffer
/// @param s the string's bytes (UTF-8, unescaped)
/// @param n bytes at `s`
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

/// @brief Append a double as Python's repr() spells it (shortest round-trip digits).
/// @param b the buffer
/// @param v the value
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

/// @brief Re-spell one JSON value at *p as Python's json.dumps would, appending it to `b`.
/// @param p the cursor; advanced past the value on success
/// @param b the buffer
/// @return false on malformed JSON
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
        return server_xstrdup(json ? json : "null");
    }
    json_ws(&p);
    if (*p) {
        buf_free(&b);
        return server_xstrdup(json);
    }
    return buf_take(&b);
}



/// @brief Read a message's "content": a string, null (""), an array of strings and {"text": ...} parts
/// (concatenated, other members and parts skipped), or any other value skipped as "".
/// @param p the cursor; advanced past the value
/// @param out receives the malloc'd text (NULL until it succeeds)
/// @return false on malformed JSON
bool json_content(const char **p, char **out) {
    /* Null *out up front (see json_string_n): a reparse-in-place caller must
     * not double-free when a duplicate key's second value fails to parse. */
    *out = NULL;
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = server_xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = server_xstrdup("");
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



/* RFC 4648 standard-alphabet value, or -1. */
/// @brief The 6-bit value of a standard-alphabet base64 character.
/// @param c the character
/// @return 0..63, or -1 for any other byte ('=' included)
static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}



uint8_t *base64_decode(const char *in, size_t in_len, size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!in || in_len == 0 || (in_len % 4) != 0) return NULL;
    uint8_t *out = (uint8_t *)server_xmalloc(in_len / 4 * 3 + 1);
    size_t n = 0;
    for (size_t i = 0; i < in_len; i += 4) {
        int v[4];
        int pad = 0;
        for (int k = 0; k < 4; k++) {
            const unsigned char c = (unsigned char)in[i + k];
            if (c == '=') {
                /* Padding closes the stream: only the final one or two slots
                 * of the final quad may be '=', and nothing may follow it. */
                if (k < 2 || i + 4 != in_len) {
                    free(out);
                    return NULL;
                }
                pad++;
                v[k] = 0;
                continue;
            }
            if (pad) {
                free(out);
                return NULL;
            }
            const int d = base64_value(c);
            if (d < 0) {
                free(out);
                return NULL;
            }
            v[k] = d;
        }
        const uint32_t x = ((uint32_t)v[0] << 18) | ((uint32_t)v[1] << 12) |
                           ((uint32_t)v[2] << 6) | (uint32_t)v[3];
        out[n++] = (uint8_t)(x >> 16);
        if (pad < 2) out[n++] = (uint8_t)(x >> 8);
        if (pad < 1) out[n++] = (uint8_t)x;
    }
    if (out_len) *out_len = n;
    return out;
}


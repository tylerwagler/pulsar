/* pyjson.cpp -- see pyjson.h. */
#include "pyjson.h"

#include <charconv>
#include <cmath>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

#include "pulsar_utf8.h"

namespace {

constexpr int kMaxDepth = 256;

struct parser {
    const char *begin;
    const char *p;
    const char *end;
    char *err;
    size_t errlen;
    bool failed = false;

    bool fail(const char *fmt, ...) {
        if (!failed && err && errlen) {
            va_list ap;
            va_start(ap, fmt);
            char msg[160];
            vsnprintf(msg, sizeof msg, fmt, ap);
            va_end(ap);
            snprintf(err, errlen, "json: %s at byte %zu", msg, (size_t)(p - begin));
        }
        failed = true;
        return false;
    }

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool u16(uint32_t *out) {
        if (end - p < 6 || p[0] != '\\' || p[1] != 'u') return false;
        uint32_t v = 0;
        for (int i = 2; i < 6; i++) {
            const int h = hex(p[i]);
            if (h < 0) return false;
            v = v * 16 + (uint32_t)h;
        }
        p += 6;
        *out = v;
        return true;
    }

    static void put_utf8(std::string *s, uint32_t cp) {
        if (cp < 0x80) {
            s->push_back((char)cp);
        } else if (cp < 0x800) {
            s->push_back((char)(0xc0 | (cp >> 6)));
            s->push_back((char)(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            s->push_back((char)(0xe0 | (cp >> 12)));
            s->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            s->push_back((char)(0x80 | (cp & 0x3f)));
        } else {
            s->push_back((char)(0xf0 | (cp >> 18)));
            s->push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
            s->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            s->push_back((char)(0x80 | (cp & 0x3f)));
        }
    }

    bool string(std::string *out) {
        if (p >= end || *p != '"') return fail("expected a string");
        p++;
        out->clear();
        for (;;) {
            if (p >= end) return fail("unterminated string");
            const unsigned char c = (unsigned char)*p;
            if (c == '"') { p++; return true; }
            if (c < 0x20) return fail("raw control character 0x%02x in a string", c);
            if (c == '\\') {
                if (end - p < 2) return fail("unterminated escape");
                const char e = p[1];
                switch (e) {
                case '"': out->push_back('"'); p += 2; continue;
                case '\\': out->push_back('\\'); p += 2; continue;
                case '/': out->push_back('/'); p += 2; continue;
                case 'b': out->push_back('\b'); p += 2; continue;
                case 'f': out->push_back('\f'); p += 2; continue;
                case 'n': out->push_back('\n'); p += 2; continue;
                case 'r': out->push_back('\r'); p += 2; continue;
                case 't': out->push_back('\t'); p += 2; continue;
                case 'u': {
                    uint32_t cp = 0;
                    if (!u16(&cp)) return fail("bad \\u escape");
                    if (cp >= 0xdc00 && cp <= 0xdfff)
                        return fail("lone low surrogate \\u%04x (not a Unicode scalar value)", cp);
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        uint32_t lo = 0;
                        if (!u16(&lo) || lo < 0xdc00 || lo > 0xdfff)
                            return fail("lone high surrogate \\u%04x (not a Unicode scalar value)", cp);
                        cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                    }
                    put_utf8(out, cp);
                    continue;
                }
                default:
                    return fail("bad escape \\%c", e);
                }
            }
            const int n = utf8_seq_ok((const unsigned char *)p, (size_t)(end - p));
            if (n == 0) return fail("invalid UTF-8 in a string");
            out->append(p, (size_t)n);
            p += n;
        }
    }

    bool number(pyjson_value *v) {
        const char *s = p;
        if (p < end && *p == '-') p++;
        if (p >= end || !(*p >= '0' && *p <= '9')) return fail("bad number");
        if (*p == '0') {
            p++;
        } else {
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        bool is_float = false;
        if (p < end && *p == '.') {
            is_float = true;
            p++;
            if (p >= end || !(*p >= '0' && *p <= '9')) return fail("bad fraction");
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            is_float = true;
            p++;
            if (p < end && (*p == '+' || *p == '-')) p++;
            if (p >= end || !(*p >= '0' && *p <= '9')) return fail("bad exponent");
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        if (!is_float) {
            v->kind = pyjson_value::INT;
            if (p - s == 2 && s[0] == '-' && s[1] == '0') v->num = "0";
            else v->num.assign(s, (size_t)(p - s));
            return true;
        }
        std::string tmp(s, (size_t)(p - s));
        v->kind = pyjson_value::FLOAT;
        v->f = strtod(tmp.c_str(), NULL);   /* +-HUGE_VAL on overflow, as float() */
        return true;
    }

    bool lit(const char *word) {
        const size_t n = strlen(word);
        if ((size_t)(end - p) < n || memcmp(p, word, n)) return false;
        p += n;
        return true;
    }

    bool value(pyjson_value *v, int depth) {
        if (depth > kMaxDepth) return fail("nesting deeper than %d", kMaxDepth);
        ws();
        if (p >= end) return fail("unexpected end");
        const char c = *p;
        if (c == '{') {
            p++;
            v->kind = pyjson_value::OBJ;
            std::unordered_map<std::string, size_t> index;
            ws();
            if (p < end && *p == '}') { p++; return true; }
            for (;;) {
                ws();
                std::string key;
                if (!string(&key)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                p++;
                pyjson_value item;
                if (!value(&item, depth + 1)) return false;
                auto it = index.find(key);
                if (it != index.end()) {
                    v->o[it->second].second = std::move(item);   /* dict: first position, last value */
                } else {
                    index.emplace(key, v->o.size());
                    v->o.emplace_back(std::move(key), std::move(item));
                }
                ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            p++;
            v->kind = pyjson_value::ARR;
            ws();
            if (p < end && *p == ']') { p++; return true; }
            for (;;) {
                pyjson_value item;
                if (!value(&item, depth + 1)) return false;
                v->a.push_back(std::move(item));
                ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v->kind = pyjson_value::STR;
            return string(&v->s);
        }
        if (lit("true")) { v->kind = pyjson_value::BOOL; v->b = true; return true; }
        if (lit("false")) { v->kind = pyjson_value::BOOL; v->b = false; return true; }
        if (lit("null")) { v->kind = pyjson_value::NUL; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return fail("unexpected character '%c'", c);
    }
};

void float_repr(double v, std::string *out) {
    if (std::isinf(v)) {
        out->append(v < 0 ? "-Infinity" : "Infinity");
        return;
    }
    /* Zero is tested on the BITS: under -ffast-math (no signed zeros) the
     * compiler may replace v by +0.0 inside any `v == 0.0` branch. */
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    if (!(bits << 1)) {
        out->append(bits >> 63 ? "-0.0" : "0.0");
        return;
    }
    char tmp[64];
    auto r = std::to_chars(tmp, tmp + sizeof tmp - 1, v, std::chars_format::scientific);
    *r.ptr = '\0';
    const char *m = tmp;
    if (*m == '-') { out->push_back('-'); m++; }
    const char *e = strchr(m, 'e');
    char digits[40];
    size_t nd = 0;
    for (const char *q = m; q < e; q++)
        if (*q != '.') digits[nd++] = *q;
    digits[nd] = '\0';
    const int exp10 = atoi(e + 1);
    const int decpt = exp10 + 1;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out->append("0.");
            out->append((size_t)-decpt, '0');
            out->append(digits);
        } else if ((size_t)decpt >= nd) {
            out->append(digits);
            out->append((size_t)decpt - nd, '0');
            out->append(".0");
        } else {
            out->append(digits, (size_t)decpt);
            out->push_back('.');
            out->append(digits + decpt);
        }
    } else {
        out->push_back(digits[0]);
        if (nd > 1) {
            out->push_back('.');
            out->append(digits + 1);
        }
        char ex[16];
        snprintf(ex, sizeof ex, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
        out->append(ex);
    }
}

}  // namespace

const pyjson_value *pyjson_value::get(const char *key) const {
    if (kind != OBJ) return NULL;
    for (const auto &kv : o)
        if (kv.first == key) return &kv.second;
    return NULL;
}

bool pyjson_parse(const char *text, size_t n, pyjson_value *out, char *err, size_t errlen) {
    parser ps{text, text, text + n, err, errlen};
    *out = pyjson_value();
    if (!ps.value(out, 0)) return false;
    ps.ws();
    if (ps.p != ps.end) {
        if (err && errlen) snprintf(err, errlen, "json: trailing data at byte %zu", (size_t)(ps.p - text));
        return false;
    }
    return true;
}

void pyjson_dump_string(const char *s, size_t n, std::string *out) {
    out->push_back('"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': out->append("\\\""); break;
        case '\\': out->append("\\\\"); break;
        case '\b': out->append("\\b"); break;
        case '\f': out->append("\\f"); break;
        case '\n': out->append("\\n"); break;
        case '\r': out->append("\\r"); break;
        case '\t': out->append("\\t"); break;
        default:
            if (c < 0x20) {
                char u[8];
                snprintf(u, sizeof u, "\\u%04x", c);
                out->append(u);
            } else {
                out->push_back((char)c);
            }
        }
    }
    out->push_back('"');
}

void pyjson_dump(const pyjson_value &v, std::string *out) {
    switch (v.kind) {
    case pyjson_value::NUL: out->append("null"); return;
    case pyjson_value::BOOL: out->append(v.b ? "true" : "false"); return;
    case pyjson_value::INT: out->append(v.num); return;
    case pyjson_value::FLOAT: float_repr(v.f, out); return;
    case pyjson_value::STR: pyjson_dump_string(v.s.data(), v.s.size(), out); return;
    case pyjson_value::ARR:
        out->push_back('[');
        for (size_t i = 0; i < v.a.size(); i++) {
            if (i) out->append(", ");
            pyjson_dump(v.a[i], out);
        }
        out->push_back(']');
        return;
    case pyjson_value::OBJ:
        out->push_back('{');
        for (size_t i = 0; i < v.o.size(); i++) {
            if (i) out->append(", ");
            pyjson_dump_string(v.o[i].first.data(), v.o[i].first.size(), out);
            out->append(": ");
            pyjson_dump(v.o[i].second, out);
        }
        out->push_back('}');
        return;
    }
}

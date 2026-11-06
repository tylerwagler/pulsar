/** @file
 *  The chat renderers' shared kit (chat_render_kit.h). */
/* chat_render_kit.cpp -- the span-carrying text and Python's value formatting every Jinja-following renderer shares
 * (moved out of nemotron_chat.cpp, L338; byte-identical behaviour). */
#include "chat_render_kit.h"

#include <math.h>
#include <stdio.h>

#include "chat_refuse.h"
#include "pulsar_utf8.h"

void chat_piece::client(const std::string &t) {
    if (t.empty()) return;
    spans.push_back({(uint32_t)s.size(), (uint32_t)(s.size() + t.size())});
    s += t;
}

void chat_piece::add(const chat_piece &p) {
    const uint32_t base = (uint32_t)s.size();
    for (const pulsar_text_span &sp : p.spans) {
        const pulsar_text_span v = {base + sp.lo, base + sp.hi};
        if (!spans.empty() && spans.back().hi == v.lo) spans.back().hi = v.hi;
        else spans.push_back(v);
    }
    s += p.s;
}

chat_piece chat_piece::sub(size_t lo, size_t hi) const {
    chat_piece out;
    out.s = s.substr(lo, hi - lo);
    for (const pulsar_text_span &sp : spans) {
        const size_t a = sp.lo > lo ? sp.lo : lo, b = sp.hi < hi ? sp.hi : hi;
        if (a < b) out.spans.push_back({(uint32_t)(a - lo), (uint32_t)(b - lo)});
    }
    return out;
}

chat_piece chat_piece::trimmed() const {
    size_t lo = 0, hi = 0;
    py_strip_bounds(s.data(), s.size(), &lo, &hi);
    return sub(lo, hi);
}

void chat_writer::client(const std::string &t) {
    if (t.empty()) return;
    const uint32_t lo = (uint32_t)out->text.size();
    out->text += t;
    out->spans.push_back({lo, (uint32_t)out->text.size()});
}

void chat_writer::add(const chat_piece &p) {
    const uint32_t base = (uint32_t)out->text.size();
    for (const pulsar_text_span &sp : p.spans) {
        const pulsar_text_span v = {base + sp.lo, base + sp.hi};
        if (!out->spans.empty() && out->spans.back().hi == v.lo) out->spans.back().hi = v.hi;
        else out->spans.push_back(v);
    }
    out->text += p.s;
}

bool chat_render_finish(const chat_render_out *out, char *err, size_t errlen) {
    if (out->text.size() > UINT32_MAX)
        return chat_refuse(err, errlen, "prompt_too_large", "%zu bytes do not fit the 32-bit span map", out->text.size());
    return true;
}

namespace {

/// L354: whether CPython's repr() escapes a code point (str.isprintable() is False).
/// @param cp the code point
/// @return true when it does
bool py_unprintable(uint32_t cp) {
    static const uint32_t ranges[][2] = {
#include "py_unprintable.inc"
    };
    size_t lo = 0, hi = sizeof ranges / sizeof ranges[0];
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp > ranges[mid][1]) lo = mid + 1;
        else if (cp < ranges[mid][0]) hi = mid;
        else return true;
    }
    return false;
}

/// Python's repr() of a str: single quotes unless the text has a ' and no ", escapes as CPython writes them.
/// @param s   the string (UTF-8; an invalid byte is copied as is)
/// @param out receives the repr
void py_repr_str(const std::string &s, std::string *out) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    out->push_back(q);
    const unsigned char *u = (const unsigned char *)s.data();
    for (size_t i = 0; i < s.size();) {
        const int k = utf8_seq_ok(u + i, s.size() - i);
        const uint32_t cp = k ? utf8_cp_at(u + i, k) : u[i];
        char esc[16] = "";
        if (cp == '\\') snprintf(esc, sizeof esc, "\\\\");
        else if (cp == (uint32_t)q) snprintf(esc, sizeof esc, "\\%c", q);
        else if (cp == '\n') snprintf(esc, sizeof esc, "\\n");
        else if (cp == '\r') snprintf(esc, sizeof esc, "\\r");
        else if (cp == '\t') snprintf(esc, sizeof esc, "\\t");
        else if (py_unprintable(cp)) {
            /* L354: all of Python's repr(), not only the C0 and C1 controls: U+00A0 is \\xa0, U+2028 is \\u2028 */
            if (cp < 0x100) snprintf(esc, sizeof esc, "\\x%02x", cp);
            else if (cp < 0x10000) snprintf(esc, sizeof esc, "\\u%04x", cp);
            else snprintf(esc, sizeof esc, "\\U%08x", cp);
        }
        if (esc[0]) out->append(esc);
        else out->append(s, i, k ? (size_t)k : 1);
        i += k ? (size_t)k : 1;
    }
    out->push_back(q);
}

/// Python's repr() of a float.
/// @param f   the value
/// @param out receives the repr ("inf" / "-inf", else the shortest round-trip digits)
void py_float(double f, std::string *out) {
    if (isinf(f)) {
        out->append(f > 0 ? "inf" : "-inf");
        return;
    }
    pyjson_value v;
    v.kind = pyjson_value::FLOAT;
    v.f = f;
    pyjson_dump(v, out);   /* repr(): the shortest round-trip digits, Python's notation */
}

}  // namespace

void py_repr(const pyjson_value &v, std::string *out) {
    switch (v.kind) {
    case pyjson_value::NUL: out->append("None"); return;
    case pyjson_value::BOOL: out->append(v.b ? "True" : "False"); return;
    case pyjson_value::INT: out->append(v.num); return;
    case pyjson_value::FLOAT: py_float(v.f, out); return;
    case pyjson_value::STR: py_repr_str(v.s, out); return;
    case pyjson_value::ARR:
        out->push_back('[');
        for (size_t i = 0; i < v.a.size(); i++) {
            if (i) out->append(", ");
            py_repr(v.a[i], out);
        }
        out->push_back(']');
        return;
    case pyjson_value::OBJ:
        out->push_back('{');
        for (size_t i = 0; i < v.o.size(); i++) {
            if (i) out->append(", ");
            py_repr_str(v.o[i].first, out);
            out->append(": ");
            py_repr(v.o[i].second, out);
        }
        out->push_back('}');
        return;
    }
}

std::string py_str(const pyjson_value &v) {
    if (v.kind == pyjson_value::STR) return v.s;
    std::string out;
    py_repr(v, &out);
    return out;
}

std::string py_tojson(const pyjson_value &v) {
    std::string out;
    pyjson_dump(v, &out);
    return out;
}

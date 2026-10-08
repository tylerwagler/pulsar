/* qwen_output.cpp -- the Qwen3.8-Flash-Next output parser.  See qwen_chat.h. */
#include "qwen_chat.h"

#include <stdio.h>
#include <string.h>

#include "pulsar_utf8.h"

namespace {

const char kThinkEnd[] = "</think>";
const char kCallOpen[] = "<tool_call>";
const char kCallClose[] = "</tool_call>";

/* Length of the longest suffix of s that is a proper prefix of `marker`: the
 * bytes that must wait for the next chunk before they can be called text. */
size_t marker_tail(const std::string &s, const char *marker) {
    const size_t m = strlen(marker);
    for (size_t k = std::min(s.size(), m - 1); k > 0; k--)
        if (!s.compare(s.size() - k, k, marker, k)) return k;
    return 0;
}

uint32_t cp_at(const unsigned char *p, int n) {
    switch (n) {
    case 1: return p[0];
    case 2: return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return ((uint32_t)(p[0] & 0x0f) << 12) | ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    default: return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3f) << 12) |
                    ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    }
}

}  // namespace

bool qwen_output_parser::init(bool thinking, const char *tools_json, char *err, size_t errlen) {
    *this = qwen_output_parser();
    mode_ = thinking ? M_REASONING : M_CONTENT;
    if (tools_json) {
        char jerr[160] = "not a JSON array";
        if (!pyjson_parse(tools_json, strlen(tools_json), &tools_, jerr, sizeof jerr) ||
            tools_.kind != pyjson_value::ARR) {
            if (err && errlen) snprintf(err, errlen, "tools_not_array: %s", jerr);
            return false;
        }
    }
    return true;
}

/* Stripped streaming: leading whitespace of a section is dropped, interior
 * whitespace waits for the next non-space, trailing whitespace is dropped by
 * end_section.  Works on whole codepoints; a split sequence waits. */
void qwen_output_parser::emit_text(bool reasoning, const char *p, size_t n, std::vector<qwen_out_event> *ev) {
    section &s = sec_[reasoning ? 1 : 0];
    std::string buf = s.partial + std::string(p, n);
    s.partial.clear();
    std::string out;
    const unsigned char *u = (const unsigned char *)buf.data();
    for (size_t i = 0; i < buf.size();) {
        int k = utf8_seq_ok(u + i, buf.size() - i);
        bool space = k && qwen_py_isspace(cp_at(u + i, k));
        if (!k) {
            /* an incomplete sequence at the end waits; an invalid byte is text */
            const int want = utf8_seq_len(u[i]);
            if (want > 1 && i + (size_t)want > buf.size()) {
                bool prefix_ok = true;
                for (size_t j = i + 1; j < buf.size(); j++) prefix_ok &= (u[j] & 0xc0) == 0x80;
                if (prefix_ok) { s.partial.assign(buf, i, std::string::npos); break; }
            }
            k = 1;
            space = false;
        }
        if (space) {
            if (s.started) s.ws.append(buf, i, (size_t)k);
        } else {
            out += s.ws;
            s.ws.clear();
            out.append(buf, i, (size_t)k);
            s.started = true;
        }
        i += (size_t)k;
    }
    if (out.empty()) return;
    (reasoning ? reasoning_ : content_) += out;
    qwen_out_event e;
    e.kind = reasoning ? qwen_out_event::REASONING : qwen_out_event::CONTENT;
    e.text = std::move(out);
    ev->push_back(std::move(e));
}

void qwen_output_parser::end_section(bool reasoning) {
    section &s = sec_[reasoning ? 1 : 0];
    s.ws.clear();
    s.started = false;
    /* a dangling partial sequence at a section end is not text either side can own */
    s.partial.clear();
}

/* The type the schema declares for `key` of function `fn`: "string" when the
 * declared type is or includes string, "other" for any other declaration,
 * NULL when the tool or the property is not declared. */
const char *qwen_output_parser::param_type(const std::string &fn, const std::string &key) const {
    for (const pyjson_value &tool : tools_.a) {
        const pyjson_value *f = tool.get("function");
        if (!f) f = &tool;
        const pyjson_value *name = f->get("name");
        if (!name || name->kind != pyjson_value::STR || name->s != fn) continue;
        const pyjson_value *params = f->get("parameters");
        const pyjson_value *props = params ? params->get("properties") : NULL;
        const pyjson_value *prop = props ? props->get(key.c_str()) : NULL;
        if (!prop) return NULL;
        const pyjson_value *type = prop->get("type");
        if (!type) return "other";
        if (type->kind == pyjson_value::STR) return type->s == "string" ? "string" : "other";
        if (type->kind == pyjson_value::ARR)
            for (const pyjson_value &t : type->a)
                if (t.kind == pyjson_value::STR && t.s == "string") return "string";
        return "other";
    }
    return NULL;
}

/* One "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i]: 1 = read (*key, the typed *v,
 * *i past it), 0 = not complete yet, -1 = not a parameter; *why names what is missing or wrong.  The one
 * reading close_call and the stream share. */
int qwen_output_parser::read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key,
                                   pyjson_value *v, const char **why) const {
    static const char kParam[] = "<parameter=";
    const size_t n = strlen(kParam), rem = b.size() - *i;
    if (b.compare(*i, std::min(rem, n), kParam, std::min(rem, n))) {
        *why = "expected <parameter= or </function>";
        return -1;
    }
    *why = "unterminated <parameter=KEY>";
    if (rem < n) return 0;
    const size_t at = *i + n;
    const size_t key_end = b.find('>', at);
    if (b.find('\n', at) < key_end) return -1;
    if (key_end == std::string::npos) return 0;
    *why = "unterminated parameter value";
    const size_t close = b.find("</parameter>", key_end + 1);
    if (close == std::string::npos) return 0;
    *key = b.substr(at, key_end - at);
    size_t vlo = key_end + 1, vhi = close;
    if (vlo < vhi && b[vlo] == '\n') vlo++;
    if (vhi > vlo && b[vhi - 1] == '\n') vhi--;
    const std::string text = b.substr(vlo, vhi - vlo);
    *i = close + 12;
    /* A value is JSON only when the schema does not declare it a string
     * AND the text is exactly what tojson would have written for it: the
     * template writes every non-string with tojson, so text in any other
     * spelling can only have been a raw string. */
    *v = pyjson_value();
    const char *type = param_type(fn, *key);
    char jerr[8];
    bool as_json = !(type && !strcmp(type, "string")) && pyjson_parse(text.data(), text.size(), v, jerr, sizeof jerr);
    if (as_json) {
        std::string canon;
        pyjson_dump(*v, &canon);
        as_json = v->kind != pyjson_value::STR && canon == text;   /* strings are written raw, never tojson */
    }
    if (!as_json) {
        *v = pyjson_value();
        v->kind = pyjson_value::STR;
        v->s = text;
    }
    return 1;
}

/* The open call's parameters that closed since the last call, as TOOL_ARGS fragments. */
void qwen_output_parser::stream_params(std::vector<qwen_out_event> *ev) {
    static const char kFnEnd[] = "</function>";
    while (!arg_stop_) {
        size_t i = arg_pos_;
        while (i < block_.size() && (block_[i] == ' ' || block_[i] == '\n' || block_[i] == '\t' || block_[i] == '\r')) i++;
        const size_t rem = block_.size() - i;
        if (!rem || !block_.compare(i, std::min(rem, strlen(kFnEnd)), kFnEnd, std::min(rem, strlen(kFnEnd))))
            return;   /* the call's close is close_call's */
        std::string key;
        pyjson_value v;
        const char *why = NULL;
        const int r = read_param(block_, &i, fn_, &key, &v, &why);
        if (r == 0) return;
        if (r < 0) {
            arg_stop_ = true;   /* close_call names the fault */
            return;
        }
        pyjson_value one;
        one.kind = pyjson_value::OBJ;
        one.o.emplace_back(key, std::move(v));
        std::string d;
        pyjson_dump(one, &d);   /* {"key": value} */
        qwen_out_event e;
        e.kind = qwen_out_event::TOOL_ARGS;
        e.index = index_;
        e.text = arg_n_ ? ", " + d.substr(1, d.size() - 2) : d.substr(0, d.size() - 1);
        ev->push_back(std::move(e));
        arg_n_++;
        arg_pos_ = i;
    }
}

/* Parse the body of one <tool_call> ... </tool_call>. */
void qwen_output_parser::close_call(std::vector<qwen_out_event> *ev) {
    const std::string &b = block_;
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < b.size() && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r')) i++;
    };
    auto fail = [&](const char *why) {
        qwen_out_event e;
        e.kind = qwen_out_event::ERROR;
        char msg[200];
        snprintf(msg, sizeof msg, "malformed tool call %d: %s at byte %zu of the block", index_, why, i);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
        index_++;
    };
    skip_ws();
    if (b.compare(i, 10, "<function=")) return fail("expected <function=");
    i += 10;
    const size_t name_end = b.find('>', i);
    if (name_end == std::string::npos || b.find('\n', i) < name_end) return fail("unterminated <function=NAME>");
    const std::string name = b.substr(i, name_end - i);
    i = name_end + 1;
    pyjson_value args;
    args.kind = pyjson_value::OBJ;
    for (;;) {
        skip_ws();
        if (!b.compare(i, 11, "</function>")) {
            i += 11;
            break;
        }
        std::string key;
        pyjson_value v;
        const char *why = NULL;
        if (read_param(b, &i, name, &key, &v, &why) != 1) return fail(why);
        bool replaced = false;   /* a repeated key: first position, last value */
        for (auto &kv : args.o)
            if (kv.first == key) { kv.second = v; replaced = true; }
        if (!replaced) args.o.emplace_back(key, std::move(v));
    }
    skip_ws();
    if (i != b.size()) return fail("text after </function>");
    qwen_out_call call;
    call.name = name;
    pyjson_dump(args, &call.arguments);
    if (!began_) {
        qwen_out_event e;
        e.kind = qwen_out_event::TOOL_BEGIN;
        e.index = index_;
        e.name = name;
        ev->push_back(std::move(e));
    }
    qwen_out_event e;
    e.kind = qwen_out_event::TOOL_END;
    e.index = index_;
    e.name = name;
    e.arguments = call.arguments;
    ev->push_back(std::move(e));
    calls_.push_back(std::move(call));
    index_++;
}

/* Advance pay_ over block_'s new bytes (the open call's body so far; hold_ keeps a partial
 * "</tool_call>" out of it).  A parameter value starts past "<parameter=KEY>" and ends at the first
 * "</parameter>" -- close_call's reading. */
void qwen_output_parser::scan_payload() {
    static const char kParam[] = "<parameter=", kParamEnd[] = "</parameter>";
    payload_scan &ps = pay_;
    const std::string &b = block_;
    for (;;) {
        if (!ps.value) {
            const size_t at = b.find(kParam, ps.pos);
            if (at == std::string::npos) {
                ps.pos = b.size() > strlen(kParam) ? std::max(ps.pos, b.size() - strlen(kParam)) : ps.pos;
                break;
            }
            const size_t gt = b.find('>', at);
            if (gt == std::string::npos) {
                ps.pos = at;
                break;
            }
            const char *type = param_type(fn_, b.substr(at + strlen(kParam), gt - at - strlen(kParam)));
            ps.value = true;
            ps.is_string = type && !strcmp(type, "string");
            ps.json_str = ps.esc = false;
            ps.pos = ps.vstart = gt + 1;
            continue;
        }
        bool closed = false;
        while (ps.pos < b.size()) {
            if (!b.compare(ps.pos, strlen(kParamEnd), kParamEnd)) {
                ps.pos += strlen(kParamEnd);
                ps.value = false;
                closed = true;
                break;
            }
            if (b[ps.pos] == '<' && b.size() - ps.pos < strlen(kParamEnd) &&
                !b.compare(ps.pos, b.size() - ps.pos, kParamEnd, b.size() - ps.pos))
                break;   /* a partial closer waits for its next bytes */
            const char c = b[ps.pos++];
            if (ps.is_string) continue;
            if (ps.json_str) {
                if (ps.esc) ps.esc = false;
                else if (c == '\\') ps.esc = true;
                else if (c == '"') ps.json_str = false;
            } else if (c == '"') {
                ps.json_str = true;
            }
        }
        if (!closed) break;
    }
    /* the payload: inside a string value or a JSON string, past the template's leading newline, not on
     * "</" of a closer (this block's or the held "</tool_call>") */
    const size_t pending = b.size() - std::min(ps.pos, b.size());
    ps.payload = ps.value && (ps.is_string || ps.json_str) && b.size() > ps.vstart && pending < 2 && hold_.size() < 2;
}

bool qwen_output_parser::raw_span(size_t *lo, size_t *hi) const {
    if (calls_.empty()) return false;
    *lo = raw_lo_;
    *hi = raw_hi_;
    return true;
}

void qwen_output_parser::feed(const char *p, size_t n, std::vector<qwen_out_event> *ev) {
    hold_.append(p, n);
    fed_ += n;
    for (;;) {
        if (mode_ == M_REASONING) {
            const size_t at = hold_.find(kThinkEnd);
            if (at == std::string::npos) {
                const size_t keep = marker_tail(hold_, kThinkEnd);
                emit_text(true, hold_.data(), hold_.size() - keep, ev);
                hold_.erase(0, hold_.size() - keep);
                return;
            }
            emit_text(true, hold_.data(), at, ev);
            end_section(true);
            hold_.erase(0, at + strlen(kThinkEnd));
            mode_ = M_CONTENT;
            continue;
        }
        if (mode_ == M_CONTENT || mode_ == M_AFTER_TOOL) {
            const size_t at = hold_.find(kCallOpen);
            if (at == std::string::npos) {
                const size_t keep = marker_tail(hold_, kCallOpen);
                emit_text(false, hold_.data(), hold_.size() - keep, ev);
                hold_.erase(0, hold_.size() - keep);
                return;
            }
            emit_text(false, hold_.data(), at, ev);
            end_section(false);
            if (!block_seen_) {   /* hold_[0] sits at stream offset fed_ - hold_.size() */
                raw_lo_ = fed_ - hold_.size() + at;
                block_seen_ = true;
            }
            hold_.erase(0, at + strlen(kCallOpen));
            block_.clear();
            began_ = false;
            fn_.clear();
            pay_ = payload_scan();
            mode_ = M_TOOL;
            continue;
        }
        /* M_TOOL */
        const size_t at = hold_.find(kCallClose);
        const size_t keep = at == std::string::npos ? marker_tail(hold_, kCallClose) : 0;
        const size_t take = at == std::string::npos ? hold_.size() - keep : at;
        block_.append(hold_, 0, take);
        if (!began_) {
            /* announce the call as soon as its name is complete */
            size_t i = 0;
            while (i < block_.size() && (block_[i] == ' ' || block_[i] == '\n' || block_[i] == '\t' || block_[i] == '\r')) i++;
            if (!block_.compare(i, 10, "<function=")) {
                const size_t e = block_.find('>', i + 10);
                const size_t nl = block_.find('\n', i + 10);
                if (e != std::string::npos && nl > e) {
                    qwen_out_event ev_begin;
                    ev_begin.kind = qwen_out_event::TOOL_BEGIN;
                    ev_begin.index = index_;
                    ev_begin.name = block_.substr(i + 10, e - i - 10);
                    fn_ = ev_begin.name;
                    ev->push_back(std::move(ev_begin));
                    began_ = true;
                    arg_pos_ = e + 1;
                    arg_n_ = 0;
                    arg_stop_ = false;
                }
            }
        }
        if (at == std::string::npos) {
            hold_.erase(0, take);
            if (began_) {
                scan_payload();
                stream_params(ev);
            }
            return;
        }
        raw_hi_ = fed_ - hold_.size() + at + strlen(kCallClose);
        hold_.erase(0, at + strlen(kCallClose));
        if (began_) stream_params(ev);   /* a last parameter closed in this piece goes out before the end */
        close_call(ev);
        mode_ = M_AFTER_TOOL;
    }
}

void qwen_output_parser::finish(std::vector<qwen_out_event> *ev) {
    if (mode_ == M_TOOL) {
        block_ += hold_;
        hold_.clear();
        qwen_out_event e;
        e.kind = qwen_out_event::ERROR;
        char msg[120];
        snprintf(msg, sizeof msg, "unterminated tool call %d (the turn ended inside <tool_call>)", index_);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
        index_++;
        return;
    }
    /* a held marker prefix at the end was text after all */
    const bool reasoning = mode_ == M_REASONING;
    emit_text(reasoning, hold_.data(), hold_.size(), ev);
    hold_.clear();
    end_section(reasoning);
}

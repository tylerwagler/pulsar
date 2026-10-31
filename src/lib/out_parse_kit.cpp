/** @file
 *  The output-parser kit's implementation and the shared JSON {"name", "args"} call body. */
/* out_parse_kit.cpp -- the output-parser kit.  See out_parse_kit.h. */
#include "out_parse_kit.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "pulsar_utf8.h"
#include "pyjson.h"

namespace {

/* Length of the longest suffix of s that is a proper prefix of `marker`: the
 * bytes that must wait for the next chunk before they can be called text. */
/// @param s      the held bytes
/// @param marker a marker the current state could take
/// @return the length of the longest suffix of `s` that is a proper prefix of `marker` (0 if none)
size_t marker_tail(const std::string &s, const std::string &marker) {
    const size_t m = marker.size();
    for (size_t k = std::min(s.size(), m - 1); k > 0; k--)
        if (!s.compare(s.size() - k, k, marker, 0, k)) return k;
    return 0;
}

/// Decode one UTF-8 sequence already validated as `n` bytes long.
/// @param p the sequence's lead byte
/// @param n its length in bytes (1..4)
/// @return the codepoint
uint32_t cp_at(const unsigned char *p, int n) {
    switch (n) {
    case 1: return p[0];
    case 2: return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return ((uint32_t)(p[0] & 0x0f) << 12) | ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    default: return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3f) << 12) |
                    ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    }
}

/* The end of the whole UTF-8 sequences in b[lo, hi): hi, or the start of a sequence hi cuts. */
/// @param b  the bytes
/// @param lo the start of the range (a lead byte is not looked for before it)
/// @param hi one past the range's end
/// @return `hi`, or the offset of the incomplete sequence `hi` cuts
size_t utf8_whole_end(const std::string &b, size_t lo, size_t hi) {
    size_t k = hi;
    while (k > lo && ((unsigned char)b[k - 1] & 0xc0) == 0x80) k--;
    if (k == lo) return hi;   /* no lead byte in reach: not ours to judge */
    const int want = utf8_seq_len((unsigned char)b[k - 1]);
    return want > 1 && (k - 1) + (size_t)want > hi ? k - 1 : hi;
}

}  // namespace

/* ---- the parser ---------------------------------------------------------------------------- */

void out_parser::init(const out_grammar *g, out_call_body *body, out_state initial) {
    g_ = g;
    body_ = body;
    state_ = initial;
    hold_.clear();
    fed_ = 0;
    header_.clear();
    header_at_ = 0;
    block_seen_ = false;
    raw_lo_ = raw_hi_ = 0;
    sec_[0] = sec_[1] = section();
    block_.clear();
    index_ = errors_ = 0;
    reasoning_.clear();
    content_.clear();
    calls_.clear();
}

void out_parser::send(bool reasoning, std::string &&text, std::vector<out_event> *ev) {
    if (text.empty()) return;
    (reasoning ? reasoning_ : content_) += text;
    out_event e;
    e.kind = reasoning ? out_event::REASONING : out_event::CONTENT;
    e.text = std::move(text);
    ev->push_back(std::move(e));
}

/* A text state's bytes under its section's policy.  Works on whole codepoints; a split sequence waits; an
 * invalid byte is text.  HEADER buffers. */
void out_parser::emit_text(out_state st, const char *p, size_t n, std::vector<out_event> *ev) {
    if (st == OUT_HEADER) {
        header_.append(p, n);
        return;
    }
    const bool reasoning = st == OUT_REASONING;
    const out_text_policy policy = reasoning ? g_->reasoning : g_->content;
    section &s = sec_[reasoning ? 1 : 0];
    std::string buf = s.partial + std::string(p, n);
    s.partial.clear();
    std::string out;
    const unsigned char *u = (const unsigned char *)buf.data();
    for (size_t i = 0; i < buf.size();) {
        int k = utf8_seq_ok(u + i, buf.size() - i);
        bool space = k && py_isspace(cp_at(u + i, k));
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
        switch (policy) {
        case OUT_TEXT_STRIP:
            if (space) {
                if (s.started) s.ws.append(buf, i, (size_t)k);
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
                s.started = true;
            }
            break;
        case OUT_TEXT_RSTRIP:
            if (space) {
                s.ws.append(buf, i, (size_t)k);
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
            }
            break;
        case OUT_TEXT_KEEP_UNLESS_BLANK:
            if (s.started) {
                out.append(buf, i, (size_t)k);
            } else {
                s.ws.append(buf, i, (size_t)k);
                if (!space) {
                    s.started = true;
                    out += s.ws;
                    s.ws.clear();
                }
            }
            break;
        case OUT_TEXT_TRIM_AT_CALLS:
            if (space) {
                if (s.started || !block_seen_) s.ws.append(buf, i, (size_t)k);   /* after a call: leading dropped */
            } else {
                out += s.ws;
                s.ws.clear();
                out.append(buf, i, (size_t)k);
                s.started = true;
            }
            break;
        }
        i += (size_t)k;
    }
    send(reasoning, std::move(out), ev);
}

/* A section ends (its state is left for another).  A dangling partial sequence is not text either side can
 * own.  STRIP starts the next section afresh; the other policies run over the whole turn. */
void out_parser::end_section(out_state st) {
    if (st != OUT_CONTENT && st != OUT_REASONING) return;
    const bool reasoning = st == OUT_REASONING;
    section &s = sec_[reasoning ? 1 : 0];
    s.partial.clear();
    if ((reasoning ? g_->reasoning : g_->content) == OUT_TEXT_STRIP) {
        s.ws.clear();
        s.started = false;
    }
}

/* The turn ended: what a policy still holds. */
void out_parser::finish_section(out_state st, std::vector<out_event> *ev) {
    const bool reasoning = st == OUT_REASONING;
    section &s = sec_[reasoning ? 1 : 0];
    s.partial.clear();
    const out_text_policy policy = reasoning ? g_->reasoning : g_->content;
    if (policy == OUT_TEXT_KEEP_UNLESS_BLANK && !s.started && calls_.empty()) send(reasoning, std::move(s.ws), ev);
    if (policy == OUT_TEXT_TRIM_AT_CALLS && !block_seen_) send(reasoning, std::move(s.ws), ev);
    s.ws.clear();
}

size_t out_parser::find_marker(const out_transition **tr, size_t *len, size_t *keep) const {
    size_t best = std::string::npos;
    *tr = nullptr;
    *len = 0;
    *keep = 0;
    auto consider = [&](const std::string &m, const out_transition *t) {
        const size_t at = hold_.find(m);
        if (at != std::string::npos && (at < best || (at == best && m.size() > *len))) {
            best = at;
            *tr = t;
            *len = m.size();
        }
        *keep = std::max(*keep, marker_tail(hold_, m));
    };
    for (const out_transition &t : g_->transitions)
        if (t.from == state_) consider(g_->markers[(size_t)t.marker], &t);
    for (const std::string &m : g_->dropped) consider(m, nullptr);
    return best;
}

void out_parser::close_call(std::vector<out_event> *ev) {
    out_call call;
    const char *why = "";
    size_t at = 0;
    if (body_->close(block_, index_, ev, &call, &why, &at)) {
        out_event e;
        e.kind = out_event::TOOL_END;
        e.index = index_;
        e.name = call.name;
        e.arguments = call.arguments;
        ev->push_back(std::move(e));
        calls_.push_back(std::move(call));
    } else {
        out_event e;
        e.kind = out_event::ERROR;
        char msg[200];
        snprintf(msg, sizeof msg, "malformed tool call %d: %s at byte %zu of the block", index_, why, at);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
    }
    index_++;
}

void out_parser::feed(const char *p, size_t n, std::vector<out_event> *ev) {
    hold_.append(p, n);
    fed_ += n;
    for (;;) {
        const out_transition *tr = nullptr;
        size_t len = 0, keep = 0;
        const size_t at = find_marker(&tr, &len, &keep);
        if (at != std::string::npos && !tr) {   /* a dropped marker: the bytes around it join */
            if (state_ == OUT_CALL) block_.append(hold_, 0, at);
            else emit_text(state_, hold_.data(), at, ev);
            hold_.erase(0, at + len);
            continue;
        }
        if (state_ == OUT_CALL) {
            const size_t take = at == std::string::npos ? hold_.size() - keep : at;
            block_.append(hold_, 0, take);
            if (at == std::string::npos) {
                hold_.erase(0, take);
                body_->grow(block_, hold_.size(), false, index_, ev);
                return;
            }
            raw_hi_ = fed_ - hold_.size() + at + len;
            const size_t marker_at = fed_ - hold_.size() + at;
            hold_.erase(0, at + len);
            body_->grow(block_, 0, true, index_, ev);
            close_call(ev);
            if (tr->to == OUT_HEADER) header_at_ = marker_at + len;
            state_ = tr->to;
            continue;
        }
        if (at == std::string::npos) {
            emit_text(state_, hold_.data(), hold_.size() - keep, ev);
            hold_.erase(0, hold_.size() - keep);
            return;
        }
        emit_text(state_, hold_.data(), at, ev);
        const size_t marker_at = fed_ - hold_.size() + at;
        if (state_ == OUT_HEADER) {
            /* a header is a call's name (dropped) or, closed with no content kind, text */
            if (tr->act & OUT_ACT_HEADER_TEXT) emit_text(OUT_CONTENT, header_.data(), header_.size(), ev);
            header_.clear();
        } else if (tr->to != state_) {
            end_section(state_);
        }
        if (tr->to == OUT_CALL) {
            section &c = sec_[0];
            if (g_->content == OUT_TEXT_KEEP_UNLESS_BLANK && !c.started) c.ws.clear();   /* blank before a call */
            if (g_->content == OUT_TEXT_TRIM_AT_CALLS) {   /* trailing whitespace before a call; strip afresh after */
                c.ws.clear();
                c.started = false;
            }
            if (!block_seen_) {
                raw_lo_ = state_ == OUT_HEADER ? header_at_ : marker_at;
                block_seen_ = true;
            }
            block_.clear();
            body_->open();
        }
        if (tr->to == OUT_HEADER) header_at_ = marker_at + len;
        hold_.erase(0, at + len);
        state_ = tr->to;
    }
}

void out_parser::finish(std::vector<out_event> *ev) {
    if (state_ == OUT_CALL) {
        block_ += hold_;
        hold_.clear();
        out_event e;
        e.kind = out_event::ERROR;
        char msg[160];
        snprintf(msg, sizeof msg, "unterminated tool call %d (the turn ended inside %s)", index_, g_->call_open);
        e.text = msg;
        e.index = index_;
        ev->push_back(std::move(e));
        errors_++;
        index_++;
    } else {
        /* a held marker prefix at the end was text after all; a header the turn ended in is text */
        emit_text(state_, hold_.data(), hold_.size(), ev);
        hold_.clear();
        if (state_ == OUT_HEADER) {
            emit_text(OUT_CONTENT, header_.data(), header_.size(), ev);
            header_.clear();
        }
        end_section(state_);
    }
    finish_section(OUT_REASONING, ev);
    finish_section(OUT_CONTENT, ev);
}

bool out_parser::raw_span(size_t *lo, size_t *hi) const {
    if (calls_.empty()) return false;
    *lo = raw_lo_;
    *hi = raw_hi_;
    return true;
}

/* ---- the shared JSON call body ----------------------------------------------------------------- */

void out_json_call_body::open() {
    pos_ = 0;
    depth_ = 0;
    in_str_ = esc_ = false;
    str_lo_ = 0;
    key_.clear();
    want_value_ = false;
    value_key_.clear();
    name_done_ = false;
    name_.clear();
    args_lo_ = args_hi_ = std::string::npos;
    args_sent_ = 0;
    began_ = payload_ = held_ = false;
}

/* Advance over the body's new bytes: the depth-1 members' keys, "name"'s string value, where the "args"
 * object starts and ends. */
void out_json_call_body::scan(const std::string &b) {
    for (; pos_ < b.size(); pos_++) {
        const char c = b[pos_];
        if (in_str_) {
            if (esc_) {
                esc_ = false;
            } else if (c == '\\') {
                esc_ = true;
            } else if (c == '"') {
                in_str_ = false;
                if (depth_ == 1 && want_value_) {
                    if (value_key_ == "name" && !name_done_) {
                        pyjson_value v;
                        char jerr[8];
                        if (pyjson_parse(b.data() + str_lo_, pos_ + 1 - str_lo_, &v, jerr, sizeof jerr)) {
                            name_ = v.s;
                            name_done_ = true;
                        }
                    }
                    want_value_ = false;
                } else if (depth_ == 1) {
                    key_.assign(b, str_lo_ + 1, pos_ - str_lo_ - 1);
                }
            }
            continue;
        }
        switch (c) {
        case '"':
            in_str_ = true;
            str_lo_ = pos_;
            break;
        case ':':
            if (depth_ == 1) {
                want_value_ = true;
                value_key_ = key_;
            }
            break;
        case ',':
            if (depth_ == 1) {
                want_value_ = false;
                key_.clear();
            }
            break;
        case '{':
        case '[':
            if (depth_ == 1 && want_value_) {
                if (c == '{' && value_key_ == "args" && args_lo_ == std::string::npos) args_lo_ = pos_;
                want_value_ = false;
            }
            depth_++;
            break;
        case '}':
        case ']':
            depth_--;
            if (depth_ == 1 && args_lo_ != std::string::npos && args_hi_ == std::string::npos && pos_ > args_lo_)
                args_hi_ = pos_ + 1;   /* the first container closed back to depth 1 after "args" opened is it */
            break;
        default:
            break;
        }
    }
}

/* The args object's bytes so far, whole UTF-8 sequences, its final "}" held back. */
void out_json_call_body::stream_args(const std::string &b, bool closing, int index, std::vector<out_event> *ev) {
    if (!began_ || args_lo_ == std::string::npos) return;
    const size_t from = args_lo_ + args_sent_;
    size_t end = args_hi_ != std::string::npos ? args_hi_ - 1 : b.size();
    if (!closing && args_hi_ == std::string::npos) end = utf8_whole_end(b, from, end);
    if (end <= from) return;
    out_event e;
    e.kind = out_event::TOOL_ARGS;
    e.index = index;
    e.text.assign(b, from, end - from);
    ev->push_back(std::move(e));
    args_sent_ = end - args_lo_;
}

void out_json_call_body::grow(const std::string &b, size_t held, bool closing, int index,
                              std::vector<out_event> *ev) {
    scan(b);
    held_ = held > 0;
    if (!began_ && name_done_ && !name_.empty()) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name_;
        ev->push_back(std::move(e));
        began_ = true;
    }
    stream_args(b, closing, index, ev);
    payload_ = !closing && !held_ && in_str_ && depth_ >= 2;
}

bool out_json_call_body::close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
                               const char **why, size_t *at) {
    pyjson_value v;
    char jerr[160];
    *at = 0;
    if (!pyjson_parse(b.data(), b.size(), &v, jerr, sizeof jerr) || v.kind != pyjson_value::OBJ) {
        *why = "the body is not one JSON object";
        return false;
    }
    const pyjson_value *name = v.get("name");
    if (!name || name->kind != pyjson_value::STR || name->s.empty()) {
        *why = "no \"name\" string";
        return false;
    }
    const pyjson_value *args = v.get("args");
    if (args && args->kind != pyjson_value::OBJ) {
        *why = "\"args\" is not an object";
        *at = args_lo_ == std::string::npos ? 0 : args_lo_;
        return false;
    }
    if (!began_) {
        out_event e;
        e.kind = out_event::TOOL_BEGIN;
        e.index = index;
        e.name = name->s;
        ev->push_back(std::move(e));
    }
    call->name = name->s;
    call->arguments = args && args_hi_ != std::string::npos ? b.substr(args_lo_, args_hi_ - args_lo_) : "{}";
    return true;
}

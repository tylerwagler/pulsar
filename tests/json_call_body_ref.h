/** @file
 *  The shared JSON call body as it read before L437 typed its offsets, kept for the differential test. */
/* json_call_body_ref.h -- src/lib/out_parse_kit.{h,cpp}'s out_json_call_body at fb522dfb, the commit before L437 made
 * its cursors BodyOff and its lengths GenBytes: the body's offsets are the bare size_t they were, and its arithmetic
 * is that commit's.  The code is that commit's but for the names (`_ref`), `inline`, and the two hooks whose
 * signatures the kit changed (grow, close): they are shims around the reference's own reading.
 *
 * It is the REFERENCE of src/server/server_tests.cpp's test_l437_json_call_body_matches_reference and is not to be
 * edited with the live body.  Delete it with that test when the body's reading changes on purpose. */
#pragma once

#include <string.h>

#include <string>
#include <vector>

#include "out_parse_kit.h"
#include "pulsar_utf8.h"
#include "pyjson.h"

/** The shared JSON call body: one object `{"name": NAME, "args": {...}}` (other members ignored; no "args"
 * = {}).  The call is announced when "name" closes; the args object streams verbatim as it arrives (whole
 * UTF-8 sequences, its final "}" held back), and the call's arguments are that object's bytes as sampled.
 * The payload is any string inside the args value. */
class out_json_call_body_ref : public out_call_body {
public:
    void open() override;
    /// L437 plumbing: the kit's typed hook; the reference's own reading (grow_ref) takes the bare count.
    void grow(const std::string &b, GenBytes held, bool closing, int index, std::vector<out_event> *ev) override {
        grow_ref(b, held.raw(), closing, index, ev);
    }
    /// out_call_body::grow as it read at the reference's commit.
    /// @param b       the call's body so far
    /// @param held    bytes the kit holds back as a possible closing marker
    /// @param closing the closing marker was found
    /// @param index   the call's index
    /// @param ev      receives the events
    void grow_ref(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev);
    bool in_payload() const override { return payload_; }
    /// L437 plumbing: the kit's typed hook; the reference's own reading (close_ref) reports a bare offset.
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               BodyOff *at) override {
        size_t bare = 0;
        const bool ok = close_ref(b, index, ev, call, why, &bare);
        *at = BodyOff{bare};
        return ok;
    }
    /// out_call_body::close as it read at the reference's commit.
    /// @param b     the call's complete body
    /// @param index the call's index
    /// @param ev    receives TOOL_BEGIN if grow never sent it
    /// @param call  receives the call on success
    /// @param why   on failure, what is missing or wrong
    /// @param at    on failure, the byte of the body where reading stopped
    /// @return true when the body is a well-formed call
    bool close_ref(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
                   size_t *at);

private:
    /// Advance the scanner over `b`'s bytes past pos_: the depth-1 member keys, "name"'s string value, and
    /// where the "args" object starts (args_lo_) and ends (args_hi_).
    /// @param b the call's body so far
    void scan(const std::string &b);
    /// Send the args object's bytes not yet streamed as one TOOL_ARGS fragment: whole UTF-8 sequences only
    /// (unless `closing`), its final "}" held back.  Nothing before TOOL_BEGIN.
    /// @param b       the call's body so far
    /// @param closing the body is complete
    /// @param index   the call's index, for the event
    /// @param ev      receives the fragment
    void stream_args(const std::string &b, bool closing, int index, std::vector<out_event> *ev);

    size_t pos_ = 0;            ///< bytes of the body scanned
    int depth_ = 0;             ///< object / array depth at pos_
    bool in_str_ = false;       ///< inside a string literal
    bool esc_ = false;          ///< the previous byte was a backslash in it
    size_t str_lo_ = 0;         ///< where the open string literal starts (its quote)
    std::string key_;           ///< the last depth-1 string closed (a member key until its value starts)
    bool want_value_ = false;   ///< a depth-1 ':' was seen: the next value is key_'s
    std::string value_key_;     ///< the member whose value is being read at depth 1
    bool name_done_ = false;    ///< "name"'s string value closed
    std::string name_;          ///< its decoded text
    size_t args_lo_ = std::string::npos;  ///< where the "args" object starts
    size_t args_hi_ = std::string::npos;  ///< one past its closing "}"
    size_t args_sent_ = 0;      ///< args bytes streamed (from args_lo_)
    bool began_ = false;        ///< TOOL_BEGIN sent
    /// The next byte is inside a string at depth >= 2 (the args value), with no closer held: payload.
    bool payload_ = false;
    bool held_ = false;         ///< the kit holds a possible closer
};


/* The end of the whole UTF-8 sequences in b[lo, hi): hi, or the start of a sequence hi cuts. */
/// @param b  the bytes
/// @param lo the start of the range (a lead byte is not looked for before it)
/// @param hi one past the range's end
/// @return `hi`, or the offset of the incomplete sequence `hi` cuts
inline size_t utf8_whole_end_ref(const std::string &b, size_t lo, size_t hi) {
    size_t k = hi;
    while (k > lo && ((unsigned char)b[k - 1] & 0xc0) == 0x80) k--;
    if (k == lo) return hi;   /* no lead byte in reach: not ours to judge */
    const int want = utf8_seq_len((unsigned char)b[k - 1]);
    return want > 1 && (k - 1) + (size_t)want > hi ? k - 1 : hi;
}


inline void out_json_call_body_ref::open() {
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
inline void out_json_call_body_ref::scan(const std::string &b) {
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
inline void out_json_call_body_ref::stream_args(const std::string &b, bool closing, int index, std::vector<out_event> *ev) {
    if (!began_ || args_lo_ == std::string::npos) return;
    const size_t from = args_lo_ + args_sent_;
    size_t end = args_hi_ != std::string::npos ? args_hi_ - 1 : b.size();
    if (!closing && args_hi_ == std::string::npos) end = utf8_whole_end_ref(b, from, end);
    if (end <= from) return;
    out_event e;
    e.kind = out_event::TOOL_ARGS;
    e.index = index;
    e.text.assign(b, from, end - from);
    ev->push_back(std::move(e));
    args_sent_ = end - args_lo_;
}

inline void out_json_call_body_ref::grow_ref(const std::string &b, size_t held, bool closing, int index,
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

inline bool out_json_call_body_ref::close_ref(const std::string &b, int index, std::vector<out_event> *ev, out_call *call,
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


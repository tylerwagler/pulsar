/** @file
 *  The chat renderers' shared kit: text with its client-data span map (piece / writer), and Python's str() / repr() /
 *  tojson of a json.loads value -- what a Jinja template's filters write. */
/* chat_render_kit.h -- what every chat renderer that follows a Jinja template shares (L338, out of nemotron_chat):
 *   chat_piece   text plus the client-data ranges inside it: what the template's string operations (concatenation,
 *                split, trim) act on, carrying the spans along;
 *   chat_writer  appends to a render's output, recording client-data spans (abutting spans merge);
 *   py_repr / py_str / py_tojson  Python's repr(), the template's `| string` and `| tojson` of a json.loads value. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "pulsar.h"
#include "pyjson.h"

/// A render's result: the prompt text and the client-data span map.
struct chat_render_out {
    /// The rendered text (UTF-8; at most UINT32_MAX bytes, else prompt_too_large).
    std::string text;
    std::vector<pulsar_text_span> spans;   ///< client-data ranges, ascending, disjoint
};

/// Text plus the client-data spans inside it.
struct chat_piece {
    /// The text.
    std::string s;
    /// Client-data ranges in `s`, ascending, disjoint.
    std::vector<pulsar_text_span> spans;

    /// Append template text.
    /// @param t the literal
    void lit(const std::string &t) { s += t; }
    /// Append client data, recording its span.
    /// @param t the client's text (nothing is recorded when empty)
    void client(const std::string &t);
    /// Append another piece, its spans rebased (an abutting span merges).
    /// @param p the piece
    void add(const chat_piece &p);
    /// @param lo the first byte kept
    /// @param hi one past the last byte kept
    /// @return [lo, hi) of this piece, its spans clipped
    chat_piece sub(size_t lo, size_t hi) const;
    /// @return the piece with Python str.strip() whitespace removed from both ends, spans clipped
    chat_piece trimmed() const;
    /// @param t a NUL-terminated substring
    /// @return true when the text contains `t`
    bool contains(const char *t) const { return s.find(t) != std::string::npos; }
};

/// Appends to a render's output, recording client-data spans as it goes.
struct chat_writer {
    /// The output being written (not owned).
    chat_render_out *out;
    /// Append template text.
    /// @param t the literal
    void lit(const char *t) { out->text += t; }
    /// Append template text.
    /// @param t the literal
    void lit(const std::string &t) { out->text += t; }
    /// Append client data, recording its span.
    /// @param t the client's text (nothing is recorded when empty)
    void client(const std::string &t);
    /// Append a piece, its spans merged with an abutting previous one (the spans stay ascending and disjoint).
    /// @param p the piece
    void add(const chat_piece &p);
};

/// Refuse a render too large for the 32-bit span map.
/// @param out    the finished render
/// @param err    the error buffer
/// @param errlen its size
/// @return true when the text fits (<= UINT32_MAX bytes)
bool chat_render_finish(const chat_render_out *out, char *err, size_t errlen);

/// Python's repr() of a json.loads value: None, True/False, numbers, quoted strings, [..], {..}.
/// @param v   the value
/// @param out receives the repr
void py_repr(const pyjson_value &v, std::string *out);
/// The template's `x | string`: a string as itself, anything else its repr.
/// @param v the value
/// @return Python's str(v)
std::string py_str(const pyjson_value &v);
/// The template's `| tojson`.
/// @param v the value
/// @return pyjson_dump of `v`
std::string py_tojson(const pyjson_value &v);

/** @file
 *  A JSON document with Python's json.loads / json.dumps semantics, for byte-exact chat templates. */
/* pyjson.h -- a JSON document with PYTHON's semantics, for the Qwen chat
 * template (L251 S5).
 *
 * The Qwen3.8-Flash-Next template writes JSON with `tool | tojson`, which in
 * transformers is `json.dumps(x, ensure_ascii=False)` over the value
 * `json.loads` produced.  A prompt that differs from those bytes in one
 * separator tokenises differently, so the renderer needs the round trip
 * exactly:
 *
 *   parse:  strict RFC 8259 (no NaN/Infinity literals, no raw control
 *           characters in strings, no trailing garbage).  An object keeps its
 *           keys in document order; a DUPLICATE key keeps its FIRST position
 *           and takes the LAST value (a Python dict built from the pairs).  A
 *           number with no fraction or exponent is a Python int (kept as its
 *           digits, any size; "-0" is 0); anything else is a float
 *           (IEEE double; out-of-range becomes +-inf, as float() does).
 *           A string must be valid UTF-8 after unescaping: a lone surrogate
 *           escape is refused by name (Python would keep it, and HF's
 *           tokenizer then fails on it -- there is no prompt to match).
 *   dump:   ", " and ": " separators, keys in stored order, non-ASCII raw,
 *           `"` `\` and the C0 controls escaped (\b \f \n \r \t short, the
 *           rest \u00XX, lowercase hex), ints verbatim, floats as repr()
 *           (shortest round-trip digits; fixed notation for decimal exponents
 *           in (-4, 16], ".0" on an integral value, else d.ddde+XX), +-inf as
 *           Infinity / -Infinity.
 *
 * This is NOT the server's JSON scanner (pulsar_json.h): that one walks a
 * request without materialising it.  This one builds a tree, because the
 * template iterates a value's items and re-serialises parts of it. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

/** One JSON value as json.loads would build it; only the field(s) of `kind` are meaningful. */
struct pyjson_value {
    /** The value's type: null, bool, Python int, float, string, array, object. */
    enum kind_t { NUL, BOOL, INT, FLOAT, STR, ARR, OBJ };
    /// Which field holds the value
    kind_t kind = NUL;
    /// BOOL
    bool b = false;
    std::string num;   ///< INT: the canonical decimal digits
    double f = 0.0;    ///< FLOAT
    std::string s;     ///< STR: UTF-8
    std::vector<pyjson_value> a;                          ///< ARR
    std::vector<std::pair<std::string, pyjson_value>> o;  ///< OBJ, document order

    /** Object member by key, or NULL (also NULL when this is not an object). */
    /// @param key NUL-terminated member name, compared bytewise
    /// @return the member's value (owned by this object), or NULL
    const pyjson_value *get(const char *key) const;
};

/** Parse exactly one JSON value spanning `text[0..n)` (surrounding whitespace
 * allowed).  On failure returns false and writes a one-line reason. */
/// @param text   the JSON text (need not be NUL-terminated)
/// @param n      its length in bytes
/// @param out    receives the parsed value
/// @param err    receives the one-line reason on failure (may be NULL)
/// @param errlen size of `err`
/// @return true on success
bool pyjson_parse(const char *text, size_t n, pyjson_value *out, char *err, size_t errlen);

/** Append `json.dumps(v, ensure_ascii=False)` to `out`. */
/// @param v   the value
/// @param out the serialization is appended here
void pyjson_dump(const pyjson_value &v, std::string *out);

/** Append a JSON string literal for the UTF-8 bytes s[0..n), Python-escaped. */
/// @param s   the UTF-8 bytes
/// @param n   their count
/// @param out the literal (with its quotes) is appended here
void pyjson_dump_string(const char *s, size_t n, std::string *out);

/** Append `json.dumps(sort(v), ensure_ascii=False, separators=(",", ":"), allow_nan=False)` -- every
 * object's keys sorted, recursively (the canonical JSON vLLM's Inkling renderer writes).  false = a
 * non-finite float (Python raises). */
/// @param v   the value
/// @param out the canonical serialization is appended here
/// @return true on success, false when `v` holds a non-finite float
bool pyjson_dump_canonical(const pyjson_value &v, std::string *out);

/** Python's truthiness of the value json.loads produced (`x or default`), with one departure kept as it is: an
 *  integer is false only when its text is exactly `0`, so `-0` (Python: int 0, false) is true.  A float -0.0 is
 *  false, as in Python. */
/// @param v the value
/// @return false for null, false, 0, 0.0, -0.0, "", [] and {}; true otherwise (`-0` included)
bool pyjson_truthy(const pyjson_value &v);

/** Python's str.isspace() for one codepoint -- the whitespace str.strip() removes (and Jinja's |trim): the
 * one definition the renderers and the output parsers share. */
/// @param cp the codepoint
/// @return true when Python's str.isspace() holds for it
bool py_isspace(uint32_t cp);

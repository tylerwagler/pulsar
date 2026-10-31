/** @file
 *  The one JSON scanner, shared by the server's API parsing and the engine's safetensors reader. */
/* pulsar_json.h -- the ONE JSON scanner, shared by the server's API parsing and
 * by the engine's safetensors reader.
 *
 * The engine reads a safetensors shard's header and its __metadata__
 * declarations, both of which are JSON, so the scanner cannot stay private to
 * the server.  It lives in src/lib beside the other shared headers, and its
 * object rides in CORE_OBJS (see the Makefile) because the ENGINE objects
 * reference it -- so one implementation serves both modules instead of two that
 * are free to drift.
 *
 * A scanner, not a document model: it walks a NUL-terminated buffer and hands
 * back malloc'd strings.  Values the caller does not want are skipped rather
 * than materialised, which is what makes it usable on a 7 MB metadata blob.
 */
#ifndef PULSAR_JSON_H
#define PULSAR_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/// Deepest array/object nesting json_skip_value accepts (a guard against stack exhaustion); overridable at build.
#ifndef JSON_MAX_NESTING
#define JSON_MAX_NESTING 256
#endif

/// Skip whitespace (isspace) at `*p`.
/// @param p cursor into a NUL-terminated buffer; advanced past the whitespace
void json_ws(const char **p);
/// Match the literal `lit` exactly at `*p` (no whitespace skipped).
/// @param p   cursor; advanced past `lit` on a match
/// @param lit the literal to match
/// @return true when `*p` starts with `lit`
bool json_lit(const char **p, const char *lit);
/// Decode a JSON string (leading whitespace skipped); json_string_n without the length.
/// @param p   cursor; advanced past the closing quote on success
/// @param out receives the decoded bytes, malloc'd and NUL-terminated (caller frees); NULL on failure
/// @return true on success
bool json_string(const char **p, char **out);
/// Decode a JSON string (leading whitespace skipped): escapes resolved, \\u surrogate pairs joined, a lone
/// surrogate becomes U+FFFD.
/// @param p       cursor; advanced past the closing quote on success
/// @param out     receives the decoded bytes, malloc'd and NUL-terminated (caller frees); nulled first, NULL on failure
/// @param out_len receives the decoded length (may hold embedded NULs); may be NULL
/// @return true on success, false on a missing quote, a bad escape or allocation failure
bool json_string_n(const char **p, char **out, size_t *out_len);
/// Read a number with strtod (leading whitespace skipped).
/// @param p   cursor; advanced past the number on success
/// @param out receives the value
/// @return true when a number was read
bool json_number(const char **p, double *out);
/// Read a number and convert it to int: negatives and NaN become 0, values above INT_MAX clamp to INT_MAX.
/// @param p   cursor; advanced past the number on success
/// @param out receives the value
/// @return true when a number was read
bool json_int(const char **p, int *out);
/// Read `true` or `false` (leading whitespace skipped).
/// @param p   cursor; advanced past the literal on success
/// @param out receives the value
/// @return true when a boolean was read
bool json_bool(const char **p, bool *out);
/// Skip one JSON value of any type, nested at most JSON_MAX_NESTING deep.
/// @param p cursor; advanced past the value on success
/// @return true when a well-formed value was skipped
bool json_skip_value(const char **p);
/// Copy one JSON value's raw text (after leading whitespace) without decoding it.
/// @param p   cursor; advanced past the value on success
/// @param out receives the value's text, malloc'd and NUL-terminated (caller frees); nulled first
/// @return true on success
bool json_raw_value(const char **p, char **out);

#endif /* PULSAR_JSON_H */

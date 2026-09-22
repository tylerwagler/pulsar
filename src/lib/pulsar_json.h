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

#ifndef JSON_MAX_NESTING
#define JSON_MAX_NESTING 256
#endif

void json_ws(const char **p);
bool json_lit(const char **p, const char *lit);
bool json_string(const char **p, char **out);
bool json_string_n(const char **p, char **out, size_t *out_len);
bool json_number(const char **p, double *out);
bool json_int(const char **p, int *out);
bool json_bool(const char **p, bool *out);
bool json_skip_value(const char **p);
bool json_raw_value(const char **p, char **out);

#endif /* PULSAR_JSON_H */

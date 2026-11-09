/** @file
 *  The chat renderers' one refusal writer (qwen_chat, nemotron_chat, inkling_chat): "<key>: <detail>" into the
 *  caller's error buffer. */
#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

/** Write "<key>: <fmt...>" into `err` (when given) and fail: every renderer refusal's shape, so a caller reads the
 *  refusal's name as the text before the first colon. */
/// @param err    the caller's error buffer, or NULL
/// @param errlen its size in bytes
/// @param key    the refusal's name (no_messages, unexpected_role, ...)
/// @param fmt    printf format of the detail
/// @return false, always
static inline bool chat_refuse(char *err, size_t errlen, const char *key, const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));
static inline bool chat_refuse(char *err, size_t errlen, const char *key, const char *fmt, ...) {
    if (err && errlen) {
        char msg[300];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        snprintf(err, errlen, "%s: %s", key, msg);
    }
    return false;
}

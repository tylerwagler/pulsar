/** @file log.cpp
 *  @brief Fatal-exit and colorized logging. */
#include "pulsar_engine_internal.h"

/* Fatal-exit and colorized logging. Split out of util.cpp in the C++ port:
 * everything here is about reporting, nothing about compute. */

/* L272: every family's loader has one failure policy.  A refusal is said where it is found and counted
 * here (moved from family.cpp, L335: model_layout.cpp's refusals reach host-only tools); the stage that found it finishes (so one load reports every problem it can see), and the family's
 * load fails at its next stage boundary -- pulsar_engine_open returns an error, the process never exits on
 * a bad artifact.  Loads run one at a time (engine open is not concurrent). */
/// Refusals counted since the last reset (pulsar_load_refuse); single-threaded, loads are serial.
static uint32_t g_load_refusals;
void pulsar_load_refuse(void) { g_load_refusals++; }
uint32_t pulsar_load_refusals(void) { return g_load_refusals; }
void pulsar_load_refusals_reset(void) { g_load_refusals = 0; }



void pulsar_die(const char *msg) {
    fprintf(stderr, "pulsar: %s\n", msg);
    exit(1);
}



void pulsar_die_errno(const char *what, const char *path) {
    fprintf(stderr, "pulsar: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}



/// @param type the log line's type
/// @return its ANSI color escape ("" for none)
static const char *pulsar_log_color_code(pulsar_log_type type) {
    switch (type) {
    case PULSAR_LOG_PREFILL:
    case PULSAR_LOG_TIMING:
        return "\x1b[36m";
    case PULSAR_LOG_GENERATION:
    case PULSAR_LOG_OK:
        return "\x1b[32m";
    case PULSAR_LOG_KVCACHE:
        return "\x1b[33m";
    case PULSAR_LOG_TOOL:
        return "\x1b[90m";
    case PULSAR_LOG_WARNING:
        return "\x1b[38;5;208m";
    case PULSAR_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}



bool pulsar_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}



/// vfprintf, colored by type when fp is a terminal (PULSAR_LOG_DEFAULT: never).
/// @param fp the stream
/// @param type the line's type
/// @param fmt printf format
/// @param ap its arguments
static void pulsar_vlog(FILE *fp, pulsar_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != PULSAR_LOG_DEFAULT && pulsar_log_is_tty(fp);
    if (colorize) fputs(pulsar_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}



void pulsar_log(FILE *fp, pulsar_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pulsar_vlog(fp, type, fmt, ap);
    va_end(ap);
}

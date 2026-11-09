/** @file
 *  Shared CLI argument parsers for the pulsar, pulsar-bench, pulsar-eval and pulsar-agent front ends. */
#ifndef PULSAR_ARGPARSE_H
#define PULSAR_ARGPARSE_H

/* Shared CLI argument helpers for the pulsar / pulsar-bench / pulsar-eval /
 * pulsar-agent front-ends. These were copied into all four and had drifted in
 * wording and naming; this is the single source. The error prefix is the
 * binary's own name via program_invocation_short_name (glibc), so each tool's
 * diagnostics read exactly as before. All exit(2) on a bad value -- the CLI
 * convention. */

#include <errno.h>   /* program_invocation_short_name */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** The value after option `opt`: advances *i past it, or exits 2 when argv ends first.
 *  @param i    in: the option's index; out: the value's index
 *  @param argc argument count
 *  @param argv argument vector
 *  @param opt  the option's name, for the message
 *  @return the value's string (argv[*i]) */
static inline const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s: missing value for %s\n", program_invocation_short_name, opt);
        exit(2);
    }
    return argv[++(*i)];
}

/* Strictly positive int (rejects 0 and negatives). */
/// @param s   the value's text
/// @param opt the option's name, for the message
/// @return the value in [1, INT_MAX]; exits 2 on anything else
static inline int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", program_invocation_short_name, opt, s);
        exit(2);
    }
    return (int)v;
}

/* Int in [min, max]. */
/// @param s   the value's text
/// @param opt the option's name, for the message
/// @param min the smallest value accepted
/// @param max the largest value accepted
/// @return the value in [min, max]; exits 2 on anything else
static inline int parse_int_range(const char *s, const char *opt, int min, int max) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < min || v > max) {
        fprintf(stderr, "%s: invalid value for %s: %s (expected %d..%d)\n",
                program_invocation_short_name, opt, s, min, max);
        exit(2);
    }
    return (int)v;
}

/* Non-negative int (allows 0). */
/// @param s   the value's text
/// @param opt the option's name, for the message
/// @return the value in [0, INT_MAX]; exits 2 on anything else
static inline int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", program_invocation_short_name, opt, s);
        exit(2);
    }
    return (int)v;
}

/* Strictly positive uint64.  strtoull negates a '-' value (wrapping "-1" to 2^64-1) and saturates past 2^64-1 with
 * ERANGE: both refuse. */
/// @param s   the value's text
/// @param opt the option's name, for the message
/// @return the nonzero value; exits 2 on an empty, non-numeric, signed-negative, out-of-range or zero value
static inline uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0 || errno == ERANGE || strchr(s, '-')) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", program_invocation_short_name, opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

/** Finite float in [min, max].
 *  @param s   the value's text
 *  @param opt the option's name, for the message
 *  @param min the smallest value accepted
 *  @param max the largest value accepted
 *  @return the value; exits 2 on anything else */
static inline float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", program_invocation_short_name, opt, s);
        exit(2);
    }
    return v;
}

/** Any finite double.
 *  @param s   the value's text
 *  @param opt the option's name, for the message
 *  @return the value; exits 2 on an empty, non-numeric or non-finite value */
static inline double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "%s: invalid value for %s: %s\n", program_invocation_short_name, opt, s);
        exit(2);
    }
    return v;
}

#endif /* PULSAR_ARGPARSE_H */

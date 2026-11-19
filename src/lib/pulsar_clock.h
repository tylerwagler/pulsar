/** @file
 *  The process's clocks, one reader each, and the named conversions between time units (L442). */
/* One clock per purpose, read in ONE place (L442; the rule-4 consolidation L334 chose "instead of types" for durations).
 *
 * Before L442 sixteen file-local helpers read the clock -- server_now_sec, the engine's now_sec, tp_now_sec,
 * agent_now_sec, cli_now_sec, bench_now_sec, the evaluator's now_sec, front_now_s, cuda_wall_sec, seg_now, two
 * now_ms, pulsar_stall_now_ms, pulsar_metrics_stream_now_ms, engram_now_ns, and genmsg's wall_ms -- and about a
 * hundred expressions converted between units with bare 1e3 / 1e6 / 1000.0 factors.  The survey (L442's entry):
 *
 *   CLOCK_MONOTONIC  every interval, rate, timeout and stage timer: all the helpers but one, in seconds (double),
 *                    milliseconds (double, or whole int64), or nanoseconds (uint64).  std::chrono::steady_clock is
 *                    this clock on Linux (constrain.cpp, pulsar_grammar_admit.h time with it locally).
 *   CLOCK_REALTIME   wall time: printed timestamps (/metrics/stream's "t", the agent trace), the absolute deadlines a
 *                    default-clock pthread_cond_timedwait takes, time(NULL)'s created / uptime seconds -- and genmsg's
 *                    wall_ms, the stream writer's stall deadline and idle clock (gettimeofday): an INTERVAL on the wall
 *                    clock (L442's clock finding; it compares only with itself).
 *
 * Nothing compared or subtracted a reading of one clock with a reading of the other.  So: one reader per clock
 * (pulsar_mono_ts, pulsar_wall_ts), the unit accessors built on it, and named conversions.  A duration stays a plain
 * double or integer in its unit -- the series preferred this to a duration type, and the survey found no function
 * boundary where two units of time meet in one value. */
#pragma once

#include <stdint.h>
#include <time.h>

/** The monotonic clock now (CLOCK_MONOTONIC): the one reading every interval is taken from.
 *  @return the reading */
static inline struct timespec pulsar_mono_ts(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

/** The wall clock now (CLOCK_REALTIME): printed timestamps and a default-clock condition variable's deadlines only.
 *  @return the reading */
static inline struct timespec pulsar_wall_ts(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts;
}

/* ---- named conversions: a value in one unit of time as another (the factor is in the name) ---------------------- */

/** Seconds as milliseconds: `s * 1000`, in s's type.
 *  @param s seconds */
/// @return milliseconds
template <class T>
constexpr T pulsar_ms_of_sec(T s) { return s * T(1000); }
/** Seconds as microseconds: `s * 1000000`.
 *  @param s seconds */
/// @return microseconds
template <class T>
constexpr T pulsar_us_of_sec(T s) { return s * T(1000000); }
/** Seconds as nanoseconds: `s * 1000000000`.
 *  @param s seconds */
/// @return nanoseconds
template <class T>
constexpr T pulsar_ns_of_sec(T s) { return s * T(1000000000); }
/** Milliseconds as seconds: `ms / 1000` (an integer's whole seconds, truncated).
 *  @param ms milliseconds */
/// @return seconds
template <class T>
constexpr T pulsar_sec_of_ms(T ms) { return ms / T(1000); }
/** Milliseconds as microseconds: `ms * 1000`.
 *  @param ms milliseconds */
/// @return microseconds
template <class T>
constexpr T pulsar_us_of_ms(T ms) { return ms * T(1000); }
/** Milliseconds as nanoseconds: `ms * 1000000`.
 *  @param ms milliseconds */
/// @return nanoseconds
template <class T>
constexpr T pulsar_ns_of_ms(T ms) { return ms * T(1000000); }
/** Microseconds as milliseconds: `us / 1000`.
 *  @param us microseconds */
/// @return milliseconds
template <class T>
constexpr T pulsar_ms_of_us(T us) { return us / T(1000); }
/** Nanoseconds as milliseconds: `ns / 1000000` (an integer's whole milliseconds, truncated).
 *  @param ns nanoseconds */
/// @return milliseconds
template <class T>
constexpr T pulsar_ms_of_ns(T ns) { return ns / T(1000000); }
/** Nanoseconds as seconds, as a double: `ns * 1e-9` (the clock readers' own formula).
 *  @param ns nanoseconds */
/// @return seconds
constexpr double pulsar_sec_of_ns(double ns) { return ns * 1e-9; }
/** A count per millisecond as a count per second -- a rate from `n` events in `ms` milliseconds: `1000 * n / ms`.
 *  @param n  the events (tokens, rows)
 *  @param ms the milliseconds they took (> 0) */
/// @return events per second
constexpr double pulsar_per_sec(double n, double ms) { return 1000.0 * n / ms; }

/** A reading in seconds (double).
 *  @param ts the reading */
/// @return seconds since the clock's origin
static inline double pulsar_ts_sec(struct timespec ts) { return (double)ts.tv_sec + pulsar_sec_of_ns((double)ts.tv_nsec); }
/** A reading in whole milliseconds (truncated).
 *  @param ts the reading */
/// @return milliseconds since the clock's origin
static inline int64_t pulsar_ts_ms(struct timespec ts) {
    return pulsar_ms_of_sec((int64_t)ts.tv_sec) + pulsar_ms_of_ns((int64_t)ts.tv_nsec);
}
/** The milliseconds from reading `a` to reading `b` (of one clock), as a double.
 *  @param a the earlier reading
 *  @param b the later one */
/// @return b - a, milliseconds
static inline double pulsar_ms_between(struct timespec a, struct timespec b) {
    return pulsar_ms_of_sec((double)(b.tv_sec - a.tv_sec)) + (double)(b.tv_nsec - a.tv_nsec) / 1e6;
}

/** CLOCK_MONOTONIC in seconds: the timers' one helper (intervals, rates, timeouts). */
/// @return seconds since an arbitrary fixed point
static inline double pulsar_mono_sec(void) { return pulsar_ts_sec(pulsar_mono_ts()); }
/** CLOCK_MONOTONIC in milliseconds, as a double (a stage timing). */
/// @return milliseconds since an arbitrary fixed point
static inline double pulsar_mono_ms_f(void) {
    const struct timespec ts = pulsar_mono_ts();
    return pulsar_ms_of_sec((double)ts.tv_sec) + (double)ts.tv_nsec * 1e-6;
}
/** CLOCK_MONOTONIC in whole milliseconds (the stall marks, the metrics stream's pacing). */
/// @return milliseconds since an arbitrary fixed point
static inline int64_t pulsar_mono_ms(void) { return pulsar_ts_ms(pulsar_mono_ts()); }
/** CLOCK_MONOTONIC in nanoseconds (the engram pool's latencies). */
/// @return nanoseconds since an arbitrary fixed point
static inline uint64_t pulsar_mono_ns(void) {
    const struct timespec ts = pulsar_mono_ts();
    return pulsar_ns_of_sec((uint64_t)ts.tv_sec) + (uint64_t)ts.tv_nsec;
}
/** CLOCK_REALTIME in whole milliseconds. */
/// @return milliseconds since the epoch
static inline int64_t pulsar_wall_ms(void) { return pulsar_ts_ms(pulsar_wall_ts()); }
/** The CLOCK_REALTIME deadline `seconds` from now, for pthread_cond_timedwait on a default-clock condition.
 *  @param seconds how long, >= 0 */
/// @return the absolute deadline
static inline struct timespec pulsar_wall_deadline_after(double seconds) {
    struct timespec d = pulsar_wall_ts();
    const long ns = (long)pulsar_ns_of_sec(seconds);
    d.tv_sec += ns / 1000000000L;
    d.tv_nsec += ns % 1000000000L;
    if (d.tv_nsec >= 1000000000L) {
        d.tv_sec += 1;
        d.tv_nsec -= 1000000000L;
    }
    return d;
}

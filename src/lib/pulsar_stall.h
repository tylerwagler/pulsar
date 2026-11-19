/** @file
 *  The worker's process-global progress marks, the stall watchdog's input (src/server/watchdog.cpp). */
/* The worker's progress marks (the stall watchdog's input; src/server/watchdog.cpp).
 *
 * A production server hung under memory pressure 2026-10-29 (Nemotron 3.5 Super, 16 banks, ~190k context): the worker spun inside libcuda for 36
 * minutes, /health said "ok", and nothing in the log said where it was.  The marks below are the "where am I": every
 * CUDA-heavy section the worker enters -- a prefill chunk, a grid-checkpoint capture, a segment's device reads, a
 * decode quantum, a disk-KV persist -- names itself here first, and each mark is a heartbeat.  A worker that has
 * marked nothing for N seconds while jobs are queued or running is stalled, and the watchdog says so, once, with the
 * last mark it set.
 *
 * Process-global and lock-free (one worker per server; the TP worker rank marks its own process).  A mark is two
 * relaxed atomic stores and an increment: nothing on the normal path reads them.  The name must be a string literal
 * (its pointer is published, never copied). */
#pragma once

#include <atomic>
#include <stdint.h>
#include <time.h>

#include "lib/pulsar_clock.h"

/** The process-global progress marks (one instance, g_pulsar_stall); written by pulsar_stall_mark, read by the watchdog. */
struct pulsar_stall_marks {
    std::atomic<uint64_t> beats{0};            ///< every mark bumps it: the heartbeat
    std::atomic<const char *> where{nullptr};  ///< the section last entered (a string literal)
    std::atomic<int64_t> where_ms{0};          ///< CLOCK_MONOTONIC ms when it was entered
};

/// The process's one set of marks (C++17 inline variable: one instance across every TU).
inline pulsar_stall_marks g_pulsar_stall;

/** Enter a section: publish its name and the time, and beat. */
/// @param where the section's name; must be a string literal (its pointer is published)
static inline void pulsar_stall_mark(const char *where) {
    g_pulsar_stall.where.store(where, std::memory_order_relaxed);
    g_pulsar_stall.where_ms.store(pulsar_mono_ms(), std::memory_order_relaxed);
    g_pulsar_stall.beats.fetch_add(1, std::memory_order_relaxed);
}

/** @file
 *  The /metrics/stream change feed (L432): the generation counter any thread marks, and the rule by which one
 *  subscriber is sent frames -- rate-bounded, never stale past a tick, refreshed while quiet. */
#ifndef PULSAR_METRICS_STREAM_H
#define PULSAR_METRICS_STREAM_H

/* L432: who says the metrics moved, and when a subscriber hears of it.  Free of the server and of sockets, so the
 * rule is tested with a clock the test owns (the plan) and with real threads under ThreadSanitizer (the wait):
 * tests/metrics_stream_test.cpp.
 *
 * THE FEED is one counter, the generation.  A thread that has changed something /metrics reports MARKS it
 * (pulsar_metrics_feed_mark): one relaxed atomic add, no lock, no wake -- so a request path, or the worker between
 * two decode steps, pays nothing for a subscriber being there.  The mark comes AFTER the change, so a subscriber
 * that reads /metrics on a frame carrying that generation reads the change.  A publisher that already holds the
 * subscribers' mutex (the worker's snapshot) may broadcast their condition too; that only makes the frame prompt.
 *
 * THE RULE, per subscriber (pulsar_metrics_stream_plan):
 *   - The first frame is sent at once: a subscriber learns the current generation without waiting for a change.
 *   - RATE.  Two frames are at least `min_interval_ms` apart, whatever marks arrive: at most
 *     1000 / min_interval_ms frames a second.  Marks in between coalesce into the next frame, which carries the
 *     generation read when it is sent -- the last one, never an earlier one.
 *   - STALENESS.  A subscriber waits at most `min_interval_ms` at a time while it has nothing to send, and looks at
 *     the counter after every wait.  So a mark is followed by a frame carrying a generation at least as new within
 *     `min_interval_ms`, with no wake at all: a mark that lands between a subscriber's look and its wait is not
 *     lost, it is seen at the next look.  That frame is the final one of a burst: nothing is owed after it.
 *   - REFRESH.  A subscriber sent nothing for `refresh_ms` is sent a frame with the generation unchanged.  It says
 *     "nothing was marked, read again if you show what is sampled when /metrics is read" (memory, disk space, a
 *     stall's seconds), and it is the connection's proof of life.
 *
 * LOCKS.  pulsar_metrics_stream_next holds the caller's mutex to read the stop flag and to wait, and returns with
 * it released; the frame is written by the caller off the lock.  A mark takes nothing. */

#include <pthread.h>
#include <time.h>

/** The pace of one subscriber's frames.  `refresh_ms >= min_interval_ms > 0`. */
struct pulsar_metrics_stream_pace {
    int min_interval_ms;   ///< two frames are at least this far apart; also the longest a mark waits for its frame
    int refresh_ms;        ///< a subscriber sent nothing for this long is sent the unchanged generation again
};

/** Where one subscriber stands: what it was last sent, and when. */
struct pulsar_metrics_stream_cursor {
    bool sent;                       ///< a frame was sent (false: the first is owed at once)
    unsigned long long generation;   ///< the generation that frame carried
    long long sent_ms;               ///< when it was decided, on the monotonic clock, ms
};

/** What a subscriber does next. */
enum pulsar_metrics_stream_event {
    PULSAR_METRICS_STREAM_WAIT,      ///< nothing to send yet (the plan's answer only; `next` waits it out)
    PULSAR_METRICS_STREAM_CHANGE,    ///< send a frame: the first one, or the generation moved
    PULSAR_METRICS_STREAM_REFRESH,   ///< send a frame: nothing moved for `refresh_ms`
    PULSAR_METRICS_STREAM_STOP,      ///< the stop flag is set: end the stream
};

/** One decision of the rule. */
struct pulsar_metrics_stream_step {
    pulsar_metrics_stream_event event;   ///< WAIT, CHANGE or REFRESH
    int wait_ms;                         ///< with WAIT: how long before deciding again (> 0)
};

/** Say the metrics moved: bump the generation.  Any thread, any lock held or none; call it after the change.
 *  @param generation the feed's counter */
static inline void pulsar_metrics_feed_mark(unsigned long long *generation) {
    __atomic_fetch_add(generation, 1ull, __ATOMIC_RELAXED);
}

/** @param generation the feed's counter
 *  @return its value now (any thread) */
static inline unsigned long long pulsar_metrics_feed_read(const unsigned long long *generation) {
    return __atomic_load_n(generation, __ATOMIC_RELAXED);
}

/** The rule (the block comment above), as a pure function of the subscriber's cursor, the feed and the clock.
 *  @param c           the subscriber's cursor
 *  @param generation  the feed's generation now
 *  @param now_ms      the monotonic clock now, ms
 *  @param pace        the pace
 *  @return CHANGE or REFRESH when a frame is due now, else WAIT with the time to the next decision */
static inline pulsar_metrics_stream_step pulsar_metrics_stream_plan(const pulsar_metrics_stream_cursor &c,
                                                                    unsigned long long generation, long long now_ms,
                                                                    const pulsar_metrics_stream_pace &pace) {
    if (!c.sent) return {PULSAR_METRICS_STREAM_CHANGE, 0};
    const long long since = now_ms - c.sent_ms;
    if (generation != c.generation) {
        if (since >= pace.min_interval_ms) return {PULSAR_METRICS_STREAM_CHANGE, 0};
        return {PULSAR_METRICS_STREAM_WAIT, (int)(pace.min_interval_ms - since)};   /* the rate: not before then */
    }
    if (since >= pace.refresh_ms) return {PULSAR_METRICS_STREAM_REFRESH, 0};
    /* nothing to send: look again after a tick (a mark wakes nobody), or when the refresh is due if that is sooner */
    const long long to_refresh = pace.refresh_ms - since;
    return {PULSAR_METRICS_STREAM_WAIT, (int)(to_refresh < pace.min_interval_ms ? to_refresh : pace.min_interval_ms)};
}

/** @return the monotonic clock, ms */
static inline long long pulsar_metrics_stream_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000ll + ts.tv_nsec / 1000000l;
}

/** Block until the subscriber is owed a frame or the stream stops, and move its cursor to that frame.  The wait is
 *  on the monotonic clock (pthread_cond_clockwait): a wall-clock step neither holds a frame back nor releases one.
 *  @param c           the subscriber's cursor (zeroed before the first call)
 *  @param mu          guards `stop`; taken here, released on return
 *  @param cv          broadcast when `stop` is set, and by any publisher that wants its frame prompt
 *  @param generation  the feed's counter
 *  @param stop        set, under `mu`, to end every stream
 *  @param pace        the pace
 *  @param out         receives the generation the frame carries (CHANGE, REFRESH)
 *  @return CHANGE, REFRESH or STOP -- never WAIT */
static inline pulsar_metrics_stream_event pulsar_metrics_stream_next(pulsar_metrics_stream_cursor *c,
                                                                     pthread_mutex_t *mu, pthread_cond_t *cv,
                                                                     const unsigned long long *generation,
                                                                     const bool *stop,
                                                                     const pulsar_metrics_stream_pace *pace,
                                                                     unsigned long long *out) {
    pthread_mutex_lock(mu);
    for (;;) {
        if (*stop) {
            pthread_mutex_unlock(mu);
            return PULSAR_METRICS_STREAM_STOP;
        }
        const unsigned long long gen = pulsar_metrics_feed_read(generation);
        const long long now = pulsar_metrics_stream_now_ms();
        const pulsar_metrics_stream_step step = pulsar_metrics_stream_plan(*c, gen, now, *pace);
        if (step.event != PULSAR_METRICS_STREAM_WAIT) {
            c->sent = true;
            c->generation = gen;
            c->sent_ms = now;
            *out = gen;
            pthread_mutex_unlock(mu);
            return step.event;
        }
        struct timespec deadline;
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += step.wait_ms / 1000;
        deadline.tv_nsec += (long)(step.wait_ms % 1000) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec += 1;
            deadline.tv_nsec -= 1000000000L;
        }
        /* a broadcast, the deadline and a spurious wake all go round: the plan decides again */
        pthread_cond_clockwait(cv, mu, CLOCK_MONOTONIC, &deadline);
    }
}

#endif

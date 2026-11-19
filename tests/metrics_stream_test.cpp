/* L432: the /metrics/stream change feed (src/lib/pulsar_metrics_stream.h) -- the mark any thread makes and the rule
 * by which one subscriber is sent frames.  Built twice from this source (make metrics-stream-gate): plain, and
 * under ThreadSanitizer, where the marks race the subscribers' waits on purpose.
 *
 * The rule's legs run on a clock the test owns (pulsar_metrics_stream_plan is pure), so the two bounds -- frames a
 * second, and how long a mark waits for its frame -- are checked exactly, not against a sleep.  The thread legs
 * use the real wait (pulsar_metrics_stream_next) with a short window and check the same bounds on what each
 * subscriber was actually sent. */
#include "lib/pulsar_metrics_stream.h"

#include <atomic>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

static int g_failed = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            fprintf(stderr, "metrics-stream: %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c);                      \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

/* The server's pace (PULSAR_METRICS_STREAM_MIN_INTERVAL_MS / _REFRESH_MS), for the legs on the test's clock. */
static const pulsar_metrics_stream_pace k_pace = {200, 5000};

/* One frame a simulated subscriber was sent. */
struct sim_frame {
    long long at_ms;
    unsigned long long generation;
    bool refresh;
};

/* A subscriber on the test's clock: `marks_at` are the times (ms, ascending) the feed is marked; the subscriber
 * decides, waits exactly what the plan says, and decides again, until `end_ms`.  A wait is cut short at `wake_at`
 * times (a publisher's broadcast), which may or may not coincide with a mark. */
static std::vector<sim_frame> simulate(const std::vector<long long> &marks_at, const std::vector<long long> &wake_at,
                                       long long end_ms, const pulsar_metrics_stream_pace &pace) {
    std::vector<sim_frame> out;
    pulsar_metrics_stream_cursor c = {};
    long long now = 0;
    size_t wi = 0;
    while (now <= end_ms) {
        unsigned long long gen = 0;
        for (long long m : marks_at) gen += m <= now;   /* a mark at `now` is made before the look */
        const pulsar_metrics_stream_step s = pulsar_metrics_stream_plan(c, gen, now, pace);
        if (s.event != PULSAR_METRICS_STREAM_WAIT) {
            c.sent = true;
            c.generation = gen;
            c.sent_ms = now;
            out.push_back({now, gen, s.event == PULSAR_METRICS_STREAM_REFRESH});
            continue;
        }
        CHECK(s.wait_ms > 0);
        if (s.wait_ms <= 0) break;
        long long next = now + s.wait_ms;
        while (wi < wake_at.size() && wake_at[wi] <= now) wi++;
        if (wi < wake_at.size() && wake_at[wi] < next) next = wake_at[wi];
        now = next;
    }
    return out;
}

/* Both bounds on a simulated run: frames at least a window apart (so at most 1000 / window a second, in any
 * second), generations never going back, and every mark followed within a window by a frame at least as new. */
static void check_bounds(const std::vector<sim_frame> &f, const std::vector<long long> &marks_at,
                         const pulsar_metrics_stream_pace &pace) {
    for (size_t i = 1; i < f.size(); i++) {
        CHECK(f[i].at_ms - f[i - 1].at_ms >= pace.min_interval_ms);
        CHECK(f[i].generation >= f[i - 1].generation);
        CHECK(f[i].refresh == (f[i].generation == f[i - 1].generation));
        if (f[i].refresh) CHECK(f[i].at_ms - f[i - 1].at_ms == pace.refresh_ms);
    }
    const size_t per_s = (size_t)(1000 / pace.min_interval_ms);
    for (size_t i = 0; i < f.size(); i++) {   /* any one-second window holds at most per_s frames */
        size_t n = 0;
        for (size_t k = i; k < f.size() && f[k].at_ms < f[i].at_ms + 1000; k++) n++;
        CHECK(n <= per_s);
    }
    for (size_t m = 0; m < marks_at.size(); m++) {
        bool framed = false;
        for (const sim_frame &x : f)
            if (x.at_ms >= marks_at[m] && x.at_ms <= marks_at[m] + pace.min_interval_ms && x.generation >= m + 1)
                framed = true;
        CHECK(framed);
    }
}

static void test_the_first_frame_is_sent_at_once(void) {
    pulsar_metrics_stream_cursor c = {};
    const pulsar_metrics_stream_step s = pulsar_metrics_stream_plan(c, 7, 123456, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_CHANGE);
}

static void test_the_plan_at_its_edges(void) {
    pulsar_metrics_stream_cursor c = {true, 7, 1000};
    pulsar_metrics_stream_step s;
    /* changed inside the window: wait out exactly the rest of it */
    s = pulsar_metrics_stream_plan(c, 8, 1001, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 199);
    s = pulsar_metrics_stream_plan(c, 8, 1199, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 1);
    s = pulsar_metrics_stream_plan(c, 8, 1200, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_CHANGE);
    /* unchanged: a tick at a time, never past the refresh */
    s = pulsar_metrics_stream_plan(c, 7, 1000, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 200);
    s = pulsar_metrics_stream_plan(c, 7, 1130, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 200);
    s = pulsar_metrics_stream_plan(c, 7, 5900, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 100);
    s = pulsar_metrics_stream_plan(c, 7, 5999, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_WAIT && s.wait_ms == 1);
    s = pulsar_metrics_stream_plan(c, 7, 6000, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_REFRESH);
    /* a change long after the last frame goes at once */
    s = pulsar_metrics_stream_plan(c, 9, 4000, k_pace);
    CHECK(s.event == PULSAR_METRICS_STREAM_CHANGE);
}

/* An idle feed: the first frame, then the unchanged generation once a refresh interval and nothing else. */
static void test_an_idle_feed_is_refreshed_and_nothing_else(void) {
    const std::vector<sim_frame> f = simulate({}, {}, 60000, k_pace);
    CHECK(f.size() == 13);   /* t = 0, 5 s, .. 60 s */
    for (size_t i = 0; i < f.size(); i++) {
        CHECK(f[i].generation == 0);
        CHECK(f[i].at_ms == (long long)i * k_pace.refresh_ms);
        CHECK(f[i].refresh == (i > 0));
    }
    check_bounds(f, {}, k_pace);
}

/* A burst: 1,000 marks in one second (stage-1 calls arrive tens a second; this is a thousand).  Five frames a
 * second at most, and the last frame carries the last mark, within a window of it. */
static void test_a_burst_is_coalesced_and_ends_on_the_last_mark(void) {
    for (const bool woken : {false, true}) {   /* marks alone, and marks that each wake the subscriber */
        std::vector<long long> marks;
        for (int i = 0; i < 1000; i++) marks.push_back(1000 + i);
        const std::vector<sim_frame> f = simulate(marks, woken ? marks : std::vector<long long>{}, 4000, k_pace);
        check_bounds(f, marks, k_pace);
        size_t during = 0;
        const sim_frame *last_change = NULL;
        for (const sim_frame &x : f) {
            during += x.at_ms >= 1000 && x.at_ms < 2000;
            if (!x.refresh) last_change = &x;
        }
        CHECK(during <= 5);
        CHECK(last_change && last_change->generation == 1000);
        CHECK(last_change && last_change->at_ms >= 1999 && last_change->at_ms <= 1999 + k_pace.min_interval_ms);
    }
    /* all thousand in the same millisecond, right after a frame: one frame, a window later */
    {
        const std::vector<long long> marks(1000, 1);
        const std::vector<sim_frame> f = simulate(marks, {}, 1000, k_pace);
        CHECK(f.size() == 2 && f[1].generation == 1000 && f[1].at_ms == 200);
        check_bounds(f, marks, k_pace);
    }
}

/* Marks at every phase of the subscriber's tick, alone and with wakes at unrelated times: both bounds hold for
 * each, on the test's clock. */
static void test_a_mark_at_any_phase_is_framed_within_the_window(void) {
    unsigned seed = 432;
    auto rnd = [&](unsigned n) {
        seed = seed * 1103515245u + 12345u;
        return (seed >> 8) % n;
    };
    for (int trial = 0; trial < 400; trial++) {
        std::vector<long long> marks, wakes;
        long long t = 0;
        const int n = 1 + (int)rnd(40);
        for (int i = 0; i < n; i++) marks.push_back(t += rnd(4) == 0 ? rnd(7000) : rnd(250));
        long long w = 0;
        for (int i = 0; i < (int)rnd(30); i++) wakes.push_back(w += 1 + rnd(900));
        check_bounds(simulate(marks, wakes, t + 6000, k_pace), marks, k_pace);
    }
}

/* ---- the real wait, with threads ---- */

/* What one subscriber thread was sent. */
struct sub_log {
    std::vector<unsigned long long> generations;
    std::vector<long long> at_ms;                    /* the cursor's decision time */
    std::atomic<unsigned long long> last{0};         /* the newest generation it was sent */
    std::atomic<long> frames{0};
    bool stopped = false;
};

struct feed {
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    unsigned long long generation = 0;
    bool stop = false;
};

static void subscriber(feed *f, const pulsar_metrics_stream_pace *pace, sub_log *log) {
    pulsar_metrics_stream_cursor c = {};
    for (;;) {
        unsigned long long gen = 0;
        const pulsar_metrics_stream_event e =
            pulsar_metrics_stream_next(&c, &f->mu, &f->cv, &f->generation, &f->stop, pace, &gen);
        if (e == PULSAR_METRICS_STREAM_STOP) break;
        CHECK(e == PULSAR_METRICS_STREAM_CHANGE || e == PULSAR_METRICS_STREAM_REFRESH);
        log->generations.push_back(gen);
        log->at_ms.push_back(c.sent_ms);
        log->last.store(gen);
        log->frames.fetch_add(1);
    }
    log->stopped = true;
}

static void stop_feed(feed *f) {
    pthread_mutex_lock(&f->mu);
    f->stop = true;
    pthread_cond_broadcast(&f->cv);
    pthread_mutex_unlock(&f->mu);
}

/* Poll a state for up to `ms`; false when it never came. */
template <class P> static bool until(P pred, int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

/* A mark racing a subscriber's wait, one mark at a time and at every phase the scheduler gives: the mark is made
 * with no lock and no wake, exactly as a request thread makes it, while the subscriber is somewhere between
 * looking at the counter and sleeping.  Each one must be framed, and within the window (plus the machine's
 * margin: the leg waits on the frame, and only the margin is a duration). */
static void test_no_mark_is_lost_to_a_subscribers_wait(void) {
    const pulsar_metrics_stream_pace pace = {4, 60000};
    feed f;
    sub_log log;
    std::thread sub(subscriber, &f, &pace, &log);
    CHECK(until([&] { return log.frames.load() == 1; }, 10000));
    long worst_us = 0;
    for (unsigned long long i = 1; i <= 400; i++) {
        if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(137 * (i % 29)));
        const auto t0 = std::chrono::steady_clock::now();
        pulsar_metrics_feed_mark(&f.generation);
        const bool got = until([&] { return log.last.load() == i; }, 10000);
        CHECK(got);
        if (!got) break;
        const long us = (long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
                            .count();
        if (us > worst_us) worst_us = us;
    }
    CHECK(worst_us < (pace.min_interval_ms + 500) * 1000L);
    stop_feed(&f);
    sub.join();
    CHECK(log.stopped);
    CHECK(log.generations.size() == 401);   /* the first frame and one per mark: none lost, none sent twice */
    for (size_t i = 1; i < log.at_ms.size(); i++) CHECK(log.at_ms[i] - log.at_ms[i - 1] >= pace.min_interval_ms);
    printf("metrics-stream: 400 lone marks each framed; the slowest took %.1f ms at a %d ms window\n",
           worst_us / 1e3, pace.min_interval_ms);
}

/* The whole shape under load: request threads marking with no lock, a worker publishing under the lock with a
 * broadcast, and several subscribers.  Each subscriber's frames are a window apart, its generations never go back,
 * and once the writers are done each one ends on the final generation -- the burst's last state -- and is then
 * sent nothing more until the refresh. */
static void test_writers_and_subscribers_together(void) {
    const pulsar_metrics_stream_pace pace = {10, 60000};
    feed f;
    const int n_subs = 4, n_markers = 3, marks_each = 1000;
    std::vector<sub_log> logs(n_subs);
    std::vector<std::thread> subs, writers;
    for (int i = 0; i < n_subs; i++) subs.emplace_back(subscriber, &f, &pace, &logs[(size_t)i]);
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_markers; t++)
        writers.emplace_back([&] {
            for (int i = 0; i < marks_each; i++) {
                pulsar_metrics_feed_mark(&f.generation);
                if (i % 50 == 49) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    writers.emplace_back([&] {   /* the worker: its snapshot's mark and broadcast, under the lock */
        for (int i = 0; i < marks_each; i++) {
            pthread_mutex_lock(&f.mu);
            pulsar_metrics_feed_mark(&f.generation);
            pthread_cond_broadcast(&f.cv);
            pthread_mutex_unlock(&f.mu);
            if (i % 50 == 49) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    for (std::thread &w : writers) w.join();
    const unsigned long long final_gen = (unsigned long long)(n_markers + 1) * marks_each;
    CHECK(pulsar_metrics_feed_read(&f.generation) == final_gen);
    for (sub_log &l : logs) CHECK(until([&] { return l.last.load() == final_gen; }, 10000));
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(5 * pace.min_interval_ms));   /* nothing follows it */
    stop_feed(&f);
    for (std::thread &s : subs) s.join();
    long most = 0;
    for (const sub_log &l : logs) {
        CHECK(l.stopped);
        CHECK(!l.generations.empty() && l.generations.back() == final_gen);
        for (size_t i = 1; i < l.generations.size(); i++) {
            CHECK(l.generations[i] > l.generations[i - 1]);   /* no refresh here: every frame is a change */
            CHECK(l.at_ms[i] - l.at_ms[i - 1] >= pace.min_interval_ms);
        }
        /* the rate bound on what was sent: one frame a window over the run, and the first */
        CHECK((double)l.generations.size() <= 2.0 + elapsed_ms / pace.min_interval_ms);
        if ((long)l.generations.size() > most) most = (long)l.generations.size();
    }
    printf("metrics-stream: %llu marks from %d threads in %.0f ms -> at most %ld frames a subscriber (window %d ms)\n",
           final_gen, n_markers + 1, elapsed_ms, most, pace.min_interval_ms);
}

/* A stop is noticed at once, not at the end of a wait (the server's shutdown drains its subscribers). */
static void test_stop_ends_a_waiting_subscriber_at_once(void) {
    const pulsar_metrics_stream_pace pace = {30000, 60000};   /* a subscriber that would sleep half a minute */
    feed f;
    sub_log log;
    std::thread sub(subscriber, &f, &pace, &log);
    CHECK(until([&] { return log.frames.load() == 1; }, 10000));
    const auto t0 = std::chrono::steady_clock::now();
    stop_feed(&f);
    sub.join();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    CHECK(log.stopped && log.generations.size() == 1);
}

int main() {
    test_the_first_frame_is_sent_at_once();
    test_the_plan_at_its_edges();
    test_an_idle_feed_is_refreshed_and_nothing_else();
    test_a_burst_is_coalesced_and_ends_on_the_last_mark();
    test_a_mark_at_any_phase_is_framed_within_the_window();
    test_no_mark_is_lost_to_a_subscribers_wait();
    test_writers_and_subscribers_together();
    test_stop_ends_a_waiting_subscriber_at_once();
    if (g_failed) {
        fprintf(stderr, "metrics-stream: %d checks FAILED\n", g_failed);
        return 1;
    }
    printf("metrics-stream: PASS\n");
    return 0;
}

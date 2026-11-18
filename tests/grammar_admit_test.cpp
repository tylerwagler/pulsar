/* L427: grammar compile admission (src/lib/pulsar_grammar_admit.h) -- the slot queue, the pending set and the
 * per-caller deadline, driven with compiles that block until the test releases them, so "slow" is a flag and not
 * a minute-long schema.  Built twice (make grammar-admit-gate): plain with GRAMMAR_ADMIT_REAL, which adds the legs
 * through the real xgrammar boundary (pulsar_grammar_compile), and under ThreadSanitizer without it (the admission
 * is the concurrent code; xgrammar's own cache is upstream's).
 *
 * Every wait in here is for a STATE (a compile entered, N callers waiting), never for a duration, except the short
 * deadlines the give-up legs are about. */
#include "lib/pulsar_grammar_admit.h"

#include <atomic>
#include <chrono>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

using admit = pulsar_grammar_admit<std::string>;
using sclock = std::chrono::steady_clock;

static int g_failed = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            fprintf(stderr, "grammar-admit: %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c);                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

/* Poll a state for up to 10 s; a state that never comes fails the leg instead of hanging it. */
template <class P> static bool until(P pred) {
    const auto end = sclock::now() + std::chrono::seconds(10);
    while (!pred()) {
        if (sclock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

/* The compiles: each records that it started (in order), then blocks until its key is released. */
struct fake {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::string> started;   /* keys, in the order their compiles began */
    std::set<std::string> released;
    bool release_all = false;
    int finished = 0;

    admit::work_fn work(const std::string &k, bool fail = false) {
        return [this, k, fail](std::string *e) -> std::optional<std::string> {
            std::unique_lock<std::mutex> lk(mu);
            started.push_back(k);
            cv.wait(lk, [&] { return release_all || released.count(k); });
            finished++;
            if (fail) {
                *e = "bad " + k;
                return std::nullopt;
            }
            return "G:" + k;
        };
    }
    void release(const std::string &k) {
        std::lock_guard<std::mutex> lk(mu);
        released.insert(k);
        cv.notify_all();
    }
    void open() {
        std::lock_guard<std::mutex> lk(mu);
        release_all = true;
        cv.notify_all();
    }
    int starts(const std::string &k = "") {
        std::lock_guard<std::mutex> lk(mu);
        if (k.empty()) return (int)started.size();
        int n = 0;
        for (const std::string &s : started) n += s == k;
        return n;
    }
    int done() {
        std::lock_guard<std::mutex> lk(mu);
        return finished;
    }
    std::vector<std::string> order() {
        std::lock_guard<std::mutex> lk(mu);
        return started;
    }
};

/* One caller on its own thread. */
struct call {
    std::thread th;
    std::optional<std::string> out;
    pulsar_grammar_compile_info info = {PULSAR_GRAMMAR_ENDS, false, 0.0};
    std::string err;
    double wall_s = 0.0;
    std::atomic<bool> back{false};

    void go(admit *a, fake *f, const std::string &k, double deadline_s = 30.0, bool fail = false) {
        th = std::thread([=, this] {
            const auto t0 = sclock::now();
            out = a->run(f, k, f->work(k, fail), deadline_s, &info, &err);
            wall_s = std::chrono::duration<double>(sclock::now() - t0).count();
            back.store(true);
        });
    }
    void join() {
        if (th.joinable()) th.join();
    }
};

static bool idle(admit &a) {
    return until([&] {
        const pulsar_grammar_compile_stats s = a.stats();
        return s.running == 0 && s.waiting == 0;
    });
}

/* (f) a free slot and nothing pending: the compile starts at once, on a worker, with no wait recorded. */
static void test_fast_path(void) {
    admit a(4, 32);
    fake f;
    f.open();
    for (int i = 0; i < 3; i++) {   /* the same key again: nothing is pending between calls, so each one compiles */
        call c;
        c.go(&a, &f, "k");
        c.join();
        CHECK(c.out && *c.out == "G:k");
        CHECK(c.info.end == PULSAR_GRAMMAR_COMPILED && !c.info.shared && c.info.slot_wait_s == 0.0);
        CHECK(c.wall_s < 5.0);
    }
    CHECK(f.starts() == 3);
    const pulsar_grammar_compile_stats s = a.stats();
    CHECK(s.slot_waits == 0 && s.shared == 0 && s.ends[PULSAR_GRAMMAR_COMPILED] == 3);
    CHECK(idle(a));
    /* with three slots busy the fourth distinct key still starts at once */
    fake g;
    call busy[3], c;
    for (int i = 0; i < 3; i++) busy[i].go(&a, &g, "b" + std::to_string(i));
    CHECK(until([&] { return g.starts() == 3; }));
    g.release("free");
    c.go(&a, &g, "free");
    c.join();
    CHECK(c.out && c.info.slot_wait_s == 0.0 && !c.info.shared);
    g.open();
    for (call &b : busy) b.join();
    CHECK(idle(a));
}

/* (a) five distinct keys, four slots: the fifth waits and compiles when a slot frees. */
static void test_fifth_waits(void) {
    admit a(4, 32);
    fake f;
    call c[5];
    for (int i = 0; i < 4; i++) c[i].go(&a, &f, "k" + std::to_string(i));
    CHECK(until([&] { return f.starts() == 4; }));
    c[4].go(&a, &f, "k4");
    CHECK(until([&] { return a.stats().waiting == 5; }));
    CHECK(a.stats().running == 4 && f.starts() == 4 && !c[4].back.load());   /* waiting, not refused, not started */
    f.release("k4");   /* its own compile would return at once: it is the slot it lacks */
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(f.starts() == 4 && !c[4].back.load());
    f.release("k1");
    c[1].join();
    c[4].join();
    CHECK(c[4].out && *c[4].out == "G:k4");
    CHECK(c[4].info.end == PULSAR_GRAMMAR_COMPILED && !c[4].info.shared && c[4].info.slot_wait_s > 0.0);
    CHECK(f.starts() == 5 && f.order()[4] == "k4");
    f.open();
    for (call &x : c) x.join();
    for (int i = 0; i < 4; i++) CHECK(c[i].out && c[i].info.slot_wait_s == 0.0 && !c[i].info.shared);
    const pulsar_grammar_compile_stats s = a.stats();
    CHECK(s.slot_waits == 1 && s.slot_wait_s > 0.0 && s.ends[PULSAR_GRAMMAR_COMPILED] == 5);
    CHECK(idle(a));
}

/* (b) eleven identical requests: one compile, one slot, eleven results. */
static void test_fanout_shares(void) {
    admit a(4, 32);
    fake f;
    call c[11];
    for (call &x : c) x.go(&a, &f, "same");
    CHECK(until([&] { return a.stats().waiting == 11; }));
    CHECK(a.stats().running == 1 && f.starts() == 1);
    /* the other three slots are free for other keys while eleven wait on one */
    call other;
    f.release("other");
    other.go(&a, &f, "other");
    other.join();
    CHECK(other.out && !other.info.shared && other.info.slot_wait_s == 0.0);
    /* a different compiler (scope) with the same text is another cache entry: it does not share */
    fake f2;
    f2.open();
    call scoped;
    scoped.go(&a, &f2, "same");
    scoped.join();
    CHECK(scoped.out && !scoped.info.shared && f2.starts() == 1);
    f.open();
    int shared = 0;
    for (call &x : c) {
        x.join();
        CHECK(x.out && *x.out == "G:same" && x.info.end == PULSAR_GRAMMAR_COMPILED && x.info.slot_wait_s == 0.0);
        shared += x.info.shared;
    }
    CHECK(shared == 10 && f.starts("same") == 1);
    CHECK(a.stats().shared == 10);
    CHECK(idle(a));
}

/* (c) a caller whose deadline passes leaves alone; the compile runs on for the others; the last caller leaving a
 * running compile neither stops nor leaks its worker; the last caller leaving a queued one removes it. */
static void test_give_up(void) {
    admit a(1, 32);
    fake f;
    call owner, patient, hasty;
    owner.go(&a, &f, "slow");
    CHECK(until([&] { return f.starts() == 1; }));
    patient.go(&a, &f, "slow");
    hasty.go(&a, &f, "slow", 0.05);
    hasty.join();
    CHECK(!hasty.out && hasty.info.end == PULSAR_GRAMMAR_LATE && hasty.info.shared && hasty.info.slot_wait_s == 0.0);
    CHECK(!owner.back.load() && !patient.back.load() && a.stats().running == 1);
    f.release("slow");
    owner.join();
    patient.join();
    CHECK(owner.out && patient.out && *patient.out == "G:slow" && f.starts() == 1);
    CHECK(idle(a));

    /* the only caller gives up on a RUNNING compile: the worker is still there, finishes, and frees its slot */
    call alone;
    alone.go(&a, &f, "late", 0.05);
    alone.join();
    CHECK(!alone.out && alone.info.end == PULSAR_GRAMMAR_LATE && !alone.info.shared);
    CHECK(a.stats().running == 1 && a.stats().waiting == 0 && f.done() == 1);
    /* ... and while it runs, the same request joins it instead of starting a second compile */
    call again;
    again.go(&a, &f, "late");
    CHECK(until([&] { return a.stats().waiting == 1; }));
    CHECK(f.starts("late") == 1);
    /* a queued compile whose only caller gives up is removed: it never runs and holds no place */
    call queued;
    queued.go(&a, &f, "never", 0.05);
    queued.join();
    CHECK(!queued.out && queued.info.end == PULSAR_GRAMMAR_NO_SLOT && queued.info.slot_wait_s >= 0.05);
    /* a queued compile with two callers survives the first one leaving */
    call stay, leave;
    stay.go(&a, &f, "kept");
    CHECK(until([&] { return a.stats().waiting == 2; }));
    leave.go(&a, &f, "kept", 0.05);
    leave.join();
    CHECK(leave.info.end == PULSAR_GRAMMAR_NO_SLOT && leave.info.shared);
    f.open();
    again.join();
    stay.join();
    CHECK(again.out && again.info.shared && stay.out && *stay.out == "G:kept" && stay.info.slot_wait_s > 0.0);
    CHECK(f.starts("late") == 1 && f.starts("never") == 0 && f.starts("kept") == 1);
    CHECK(idle(a));   /* no worker left behind */
    const pulsar_grammar_compile_stats s = a.stats();
    CHECK(s.ends[PULSAR_GRAMMAR_LATE] == 2 && s.ends[PULSAR_GRAMMAR_NO_SLOT] == 2);
}

/* (d) a compile that fails reaches every caller with the same reason, and is not remembered by the admission: the
 * next identical request compiles again (in production "again" is xgrammar's cached failure; the real leg below). */
static void test_failure(void) {
    admit a(4, 32);
    fake f;
    call c[5];
    for (call &x : c) x.go(&a, &f, "bad", 30.0, true);
    CHECK(until([&] { return a.stats().waiting == 5; }));
    f.release("bad");
    for (call &x : c) {
        x.join();
        CHECK(!x.out && x.info.end == PULSAR_GRAMMAR_INVALID && x.err == "bad bad");
    }
    CHECK(f.starts() == 1 && idle(a));
    call retry;
    retry.go(&a, &f, "bad", 30.0, true);
    retry.join();
    CHECK(!retry.out && retry.info.end == PULSAR_GRAMMAR_INVALID && !retry.info.shared && retry.err == "bad bad");
    CHECK(f.starts() == 2);
    call good;   /* and the key is not poisoned: the same key compiles when the compile succeeds */
    good.go(&a, &f, "bad");
    good.join();
    CHECK(good.out && *good.out == "G:bad" && f.starts() == 3);
    CHECK(a.stats().ends[PULSAR_GRAMMAR_INVALID] == 6);
    CHECK(idle(a));
}

/* (e) slots are granted in arrival order, and a later arrival does not pass the queue. */
static void test_fairness(void) {
    admit a(4, 32);
    fake f;
    call run[4], q[6];
    for (int i = 0; i < 4; i++) run[i].go(&a, &f, "r" + std::to_string(i));
    CHECK(until([&] { return f.starts() == 4; }));
    for (int i = 0; i < 6; i++) {   /* one at a time, so the arrival order is known */
        q[i].go(&a, &f, "q" + std::to_string(i));
        CHECK(until([&] { return a.stats().waiting == 5 + i; }));
    }
    /* free one slot at a time: each goes to the oldest queued key, which then holds it */
    const char *freed[6] = {"r2", "r0", "q0", "r3", "q1", "r1"};
    const char *next[6] = {"q0", "q1", "q2", "q3", "q4", "q5"};
    for (int i = 0; i < 6; i++) {
        f.release(freed[i]);
        CHECK(until([&] { return f.starts() == 5 + i; }));
        CHECK(f.order().back() == next[i]);
        CHECK(a.stats().running == 4);
    }
    f.open();
    for (call &x : run) x.join();
    for (call &x : q) {
        x.join();
        CHECK(x.out && x.info.slot_wait_s > 0.0);
    }
    CHECK(idle(a));
}

/* The bound: with waiters_max callers waiting, one more is refused at once, whatever it asks for. */
static void test_waiter_bound(void) {
    admit a(1, 3);
    fake f;
    call c[3];
    c[0].go(&a, &f, "x");
    CHECK(until([&] { return f.starts() == 1; }));
    c[1].go(&a, &f, "x");
    c[2].go(&a, &f, "y");
    CHECK(until([&] { return a.stats().waiting == 3; }));
    for (const char *k : {"x", "y", "z"}) {
        call over;
        over.go(&a, &f, k);
        over.join();
        CHECK(!over.out && over.info.end == PULSAR_GRAMMAR_CROWDED && over.wall_s < 5.0);
    }
    f.open();
    for (call &x : c) {
        x.join();
        CHECK(x.out);
    }
    CHECK(a.stats().ends[PULSAR_GRAMMAR_CROWDED] == 3 && f.starts() == 2);
    CHECK(idle(a));
}

/* The admission object may go while a compile nobody waits for is still running: the worker keeps what it needs
 * and finishes. */
static void test_outlives(void) {
    fake f;
    {
        admit a(4, 32);
        call c;
        c.go(&a, &f, "orphan", 0.02);
        c.join();
        CHECK(c.info.end == PULSAR_GRAMMAR_LATE && a.stats().running == 1);
    }
    CHECK(f.done() == 0);
    f.open();
    CHECK(until([&] { return f.done() == 1; }));
}

/* A storm for ThreadSanitizer: many callers, few keys, compiles that return at once or fail, short and long
 * deadlines. */
static void test_storm(void) {
    std::atomic<int> compiles{0}, got{0}, refused{0};
    {
        admit a(4, 32);
        std::vector<std::thread> th;
        for (int t = 0; t < 24; t++)
            th.emplace_back([&, t] {
                for (int i = 0; i < 200; i++) {
                    const int k = (t * 7 + i) % 9;
                    const std::string key = "k" + std::to_string(k);
                    pulsar_grammar_compile_info info;
                    std::string err;
                    const std::optional<std::string> out = a.run(
                        &compiles, key,
                        [&compiles, k, key](std::string *e) -> std::optional<std::string> {
                            compiles.fetch_add(1);
                            if (k % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(300));
                            if (k == 4) {
                                *e = "bad";
                                return std::nullopt;
                            }
                            return "G:" + key;
                        },
                        i % 5 == 0 ? 0.0002 : 30.0, &info, &err);
                    if (out) {
                        if (*out != "G:" + key) abort();
                        got.fetch_add(1);
                    } else {
                        if (info.end == PULSAR_GRAMMAR_INVALID && (k != 4 || err != "bad")) abort();
                        refused.fetch_add(1);
                    }
                }
            });
        for (std::thread &t : th) t.join();
        const pulsar_grammar_compile_stats s = a.stats();
        uint64_t ends = 0;
        for (uint64_t e : s.ends) ends += e;
        CHECK(ends == 24u * 200u && s.waiting == 0 && s.running <= 4);
        CHECK(s.ends[PULSAR_GRAMMAR_COMPILED] == (uint64_t)got.load());
        CHECK(idle(a));   /* the late compiles too: their functions point into this frame */
    }
    printf("grammar-admit: storm: %d compiles for %d calls (%d results, %d refused or failed)\n", compiles.load(),
           24 * 200, got.load(), refused.load());
    CHECK(compiles.load() > 0 && compiles.load() <= 24 * 200 && got.load() > 0);
}

#ifdef GRAMMAR_ADMIT_REAL
/* The real boundary: bytes 0..255 as tokens plus a stop token. */
static pulsar_grammar_compiler *real_compiler(void) {
    std::vector<std::string> tok;
    for (int b = 0; b < 256; b++) tok.push_back(std::string(1, (char)b));
    tok.push_back("<EOS>");
    std::vector<const char *> p;
    std::vector<size_t> l;
    for (const std::string &t : tok) {
        p.push_back(t.data());
        l.push_back(t.size());
    }
    const int32_t stop = 256;
    char err[400] = "";
    pulsar_grammar_compiler *c = pulsar_grammar_compiler_new(p.data(), l.data(), 257, 257, &stop, 1, 64LL << 20, err,
                                                             sizeof err);
    if (!c) {
        fprintf(stderr, "grammar-admit: compiler: %s\n", err);
        exit(2);
    }
    return c;
}

/* Every mask row a grammar gives along `text`, then whether it ended: two grammars that agree here constrain
 * that output identically. */
static std::vector<uint32_t> walk(const pulsar_grammar *g, const char *text) {
    char err[400] = "";
    std::vector<uint32_t> rows;
    const uint32_t W = pulsar_grammar_mask_words(257);
    pulsar_grammar_matcher *m = pulsar_grammar_matcher_new(g, true, err, sizeof err);
    CHECK(m != NULL);
    if (!m) return rows;
    for (const char *b = text;; b++) {
        rows.resize(rows.size() + W);
        CHECK(pulsar_grammar_matcher_fill(m, rows.data() + rows.size() - W, err, sizeof err));
        if (!*b) break;
        CHECK(pulsar_grammar_matcher_accept(m, (uint8_t)*b, err, sizeof err));
    }
    rows.push_back(pulsar_grammar_matcher_terminated(m));
    pulsar_grammar_matcher_free(m);
    return rows;
}

static const char k_tag[] = "{\"type\":\"structural_tag\",\"format\":{\"type\":\"json_schema\",\"json_schema\":"
                            "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"n\":{\"type\":"
                            "\"integer\"}},\"required\":[\"name\",\"n\"],\"additionalProperties\":false}}}";
static const char k_doc[] = "{\"name\":\"a\",\"n\":7}";

/* Threads compiling the same real schema at once get grammars that mask identically -- and identically to a
 * compile made alone, before and after (a cache hit). */
static void test_real_same_schema(void) {
    pulsar_grammar_compiler *c = real_compiler();
    pulsar_grammar_compiler *ref_c = real_compiler();   /* a second compiler: its compile shares nothing */
    char err[400] = "";
    pulsar_grammar_compile_info info;
    pulsar_grammar *ref = pulsar_grammar_compile(ref_c, k_tag, 30.0, &info, err, sizeof err);
    CHECK(ref != NULL && info.end == PULSAR_GRAMMAR_COMPILED && !info.shared && info.slot_wait_s == 0.0);
    if (!ref) {
        fprintf(stderr, "grammar-admit: %s\n", err);
        return;
    }
    const std::vector<uint32_t> want = walk(ref, k_doc);
    CHECK(want.back() == 1u);   /* the document completes the grammar */
    enum { N = 8 };
    pulsar_grammar *g[N] = {};
    std::vector<std::thread> th;
    std::atomic<int> go{0};
    for (int i = 0; i < N; i++)
        th.emplace_back([&, i] {
            while (!go.load()) std::this_thread::yield();
            char e[400] = "";
            g[i] = pulsar_grammar_compile(c, k_tag, 30.0, NULL, e, sizeof e);
            if (!g[i]) fprintf(stderr, "grammar-admit: thread %d: %s\n", i, e);
        });
    go.store(1);
    for (std::thread &t : th) t.join();
    for (int i = 0; i < N; i++) {
        CHECK(g[i] != NULL);
        if (g[i]) CHECK(walk(g[i], k_doc) == want);
        pulsar_grammar_free(g[i]);
    }
    pulsar_grammar *hit = pulsar_grammar_compile(c, k_tag, 30.0, &info, err, sizeof err);
    CHECK(hit != NULL && info.end == PULSAR_GRAMMAR_COMPILED && !info.shared);
    if (hit) CHECK(walk(hit, k_doc) == want);
    pulsar_grammar_free(hit);
    pulsar_grammar_free(ref);
    pulsar_grammar_compiler_free(c);
    pulsar_grammar_compiler_free(ref_c);
}

/* An invalid tag: every concurrent caller gets xgrammar's reason, a retry gets the same reason again, and a valid
 * tag on the same compiler still compiles. */
static void test_real_invalid(void) {
    pulsar_grammar_compiler *c = real_compiler();
    static const char bad[] = "{\"type\":\"structural_tag\",\"format\":{\"type\":\"no_such_format\"}}";
    enum { N = 6 };
    std::string errs[N];
    pulsar_grammar_compile_info infos[N];
    std::vector<std::thread> th;
    for (int i = 0; i < N; i++)
        th.emplace_back([&, i] {
            char e[400] = "";
            pulsar_grammar *g = pulsar_grammar_compile(c, bad, 30.0, &infos[i], e, sizeof e);
            CHECK(g == NULL);
            pulsar_grammar_free(g);
            errs[i] = e;
        });
    for (std::thread &t : th) t.join();
    for (int i = 0; i < N; i++) {
        CHECK(infos[i].end == PULSAR_GRAMMAR_INVALID && !errs[i].empty() && errs[i] == errs[0]);
        CHECK(errs[i].rfind("grammar compile: ", 0) == 0);
    }
    char e[400] = "";
    pulsar_grammar_compile_info info;
    CHECK(pulsar_grammar_compile(c, bad, 30.0, &info, e, sizeof e) == NULL);
    CHECK(info.end == PULSAR_GRAMMAR_INVALID && !info.shared && errs[0] == e);
    pulsar_grammar *ok = pulsar_grammar_compile(c, k_tag, 30.0, &info, e, sizeof e);
    CHECK(ok != NULL);
    pulsar_grammar_free(ok);
    pulsar_grammar_compiler_free(c);
    printf("grammar-admit: real invalid tag: \"%s\"\n", errs[0].c_str());
    const pulsar_grammar_compile_stats s = pulsar_grammar_compile_stats_get();
    CHECK(s.waiting == 0 && s.ends[PULSAR_GRAMMAR_INVALID] == N + 1u);
}
#endif

int main() {
    test_fast_path();
    test_fifth_waits();
    test_fanout_shares();
    test_give_up();
    test_failure();
    test_fairness();
    test_waiter_bound();
    test_outlives();
    test_storm();
#ifdef GRAMMAR_ADMIT_REAL
    test_real_same_schema();
    test_real_invalid();
    const char *legs = "admission + the real xgrammar boundary";
#else
    const char *legs = "admission";
#endif
    if (g_failed) {
        printf("GRAMMAR ADMIT GATE: FAIL (%d checks, %s)\n", g_failed, legs);
        return 1;
    }
    printf("GRAMMAR ADMIT GATE: PASS (%s)\n", legs);
    return 0;
}

/** @file
 *  Grammar compile admission (L427): a slot queue, one compile per pending input, a deadline per waiting request. */
#ifndef PULSAR_GRAMMAR_ADMIT_H
#define PULSAR_GRAMMAR_ADMIT_H

/* L427: who compiles, who waits, and for how long.  pulsar_grammar.cpp runs every xgrammar compile through ONE
 * pulsar_grammar_admit; the class knows nothing of xgrammar (the compile is a function it is handed), so the rules
 * below are tested with compiles that block on a flag (tests/grammar_admit_test.cpp, and under ThreadSanitizer).
 *
 * THE RULES.
 *   - A compile runs on a detached worker thread, never on the caller's and never under the lock.  At most `slots`
 *     workers are alive; a worker whose callers have all given up still counts (a compile cannot be stopped) and
 *     still delivers: its result is what the next caller of the same input finds in the compiler's own cache.
 *   - PENDING SET.  A job is keyed by (scope, text): the compiler it runs on and the exact input, which is what the
 *     compiler's cache keys on (pulsar_grammar.cpp says why that is all of it).  A caller whose key is pending --
 *     queued or running -- JOINS that job: no thread, no slot.  The job leaves the set when its compile returns, so
 *     a failure is delivered to every caller that waited on it and is not remembered here: the next caller of that
 *     key asks the compiler again.
 *   - SLOT QUEUE.  A job that finds every slot taken waits in a FIFO; a finishing worker takes the queue's head and
 *     runs it on its own thread, so a slot is handed over, never released while a job waits: a later arrival
 *     cannot pass an earlier one (queue non-empty implies running == slots, under the lock).
 *   - DEADLINE.  Each caller waits until its OWN deadline, counted from its call: time in the queue and time
 *     compiling are the same clock.  A caller that gives up leaves alone.  The last caller leaving a job that is
 *     still QUEUED removes it (nothing ran, nothing to cache); leaving a RUNNING job changes nothing.
 *   - BOUND.  At most `waiters_max` callers wait at once, whatever they wait for; one more is refused at once.
 *
 * LOCKS.  One mutex guards the pending set, the queue and the counters.  It is held for map and queue operations
 * and for starting a thread, never across a compile or a wait (a waiting caller sleeps on its job's condition
 * variable).  The class takes no other lock and calls nothing but the compile function, off the lock.
 *
 * No exception is thrown or caught here (threads start through pthread_create), so the header compiles in any TU;
 * the compile function must not throw. */

#include "pulsar_grammar.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <string.h>
#include <string>
#include <utility>

/** Compile admission over results of type `Result` (copyable: every caller of a shared job takes a copy). */
template <class Result> class pulsar_grammar_admit {
public:
    /// One compile: the result, or nullopt with the reason in `*err`.  Runs on a worker thread; must not throw.
    using work_fn = std::function<std::optional<Result>(std::string *err)>;

    /** An admission with `slots` workers and room for `waiters_max` waiting callers.
     *  @param slots       compiles that may run at once
     *  @param waiters_max callers that may wait at once */
    pulsar_grammar_admit(int slots, int waiters_max) : st_(std::make_shared<state>()) {
        st_->slots = slots;
        st_->waiters_max = waiters_max;
    }

    /** The result of compiling (scope, text), waiting at most `deadline_s` seconds for a slot and the compile.
     *  `work` is called at most once, and only when no job for the key is pending.
     *  @param scope      the compiler the input is compiled on (identity only; `work` keeps it alive)
     *  @param text       the exact input
     *  @param work       the compile
     *  @param deadline_s the caller's wall-clock budget, seconds
     *  @param info       receives how the call ended, whether it shared a job, and its slot wait
     *  @param err        receives the compile's own reason (INVALID) or the thread's (NO_THREAD); else untouched
     *  @return the result; nullopt unless info->end is PULSAR_GRAMMAR_COMPILED */
    std::optional<Result> run(const void *scope, const std::string &text, const work_fn &work, double deadline_s,
                              pulsar_grammar_compile_info *info, std::string *err) {
        const auto t0 = clock::now();
        const auto deadline = t0 + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(
                                       std::clamp(deadline_s, 0.0, 86400.0)));
        pulsar_grammar_compile_info met = {PULSAR_GRAMMAR_COMPILED, false, 0.0};
        std::optional<Result> out;
        const std::shared_ptr<state> st = st_;
        std::shared_ptr<job> j;   /* before the lock: a job this call drops dies after the lock is released */
        std::unique_lock<std::mutex> lk(st->mu);
        key_type key(scope, text);
        const auto it = st->pending.find(key);
        if (st->waiting >= st->waiters_max) {
            met.end = PULSAR_GRAMMAR_CROWDED;
        } else if (it != st->pending.end()) {
            j = it->second;
            met.shared = true;
        } else if (st->running < st->slots) {
            /* a free slot means an empty queue (the handover rule): this is the arrival-order head */
            j = std::make_shared<job>();
            j->key = key;
            const int rc = start_worker(st, j, work);
            if (rc) {
                met.end = PULSAR_GRAMMAR_NO_THREAD;
                *err = strerror(rc);
                j.reset();
            } else {
                st->running++;
                st->pending.emplace(std::move(key), j);
            }
        } else {
            j = std::make_shared<job>();
            j->key = key;
            j->work = work;
            j->queued = true;
            st->queue.push_back(j);
            st->pending.emplace(std::move(key), j);
        }
        bool slot_waited = false;
        if (j) {
            slot_waited = j->queued;
            st->waiting++;
            j->waiters++;
            const bool done = j->cv.wait_until(lk, deadline, [&] { return j->done; });
            st->waiting--;
            j->waiters--;
            if (slot_waited)
                met.slot_wait_s = std::chrono::duration<double>((j->queued ? clock::now() : j->started) - t0).count();
            if (done) {
                out = j->out;
                if (!out) {
                    met.end = PULSAR_GRAMMAR_INVALID;
                    *err = j->err;
                }
            } else if (j->queued) {
                met.end = PULSAR_GRAMMAR_NO_SLOT;
                if (!j->waiters) {   /* nobody is left to want it and nothing ran: it gives its place up */
                    st->queue.erase(std::find(st->queue.begin(), st->queue.end(), j));
                    st->pending.erase(j->key);
                }
            } else {
                met.end = PULSAR_GRAMMAR_LATE;   /* the worker runs on, for the other callers and the cache */
            }
        }
        st->counts.ends[met.end]++;
        if (met.shared) st->counts.shared++;
        if (slot_waited) {
            st->counts.slot_waits++;
            st->counts.slot_wait_s += met.slot_wait_s;
        }
        lk.unlock();
        *info = met;
        return out;
    }

    /** The counters, with the workers alive and the callers waiting now.
     *  @return a snapshot taken under the lock */
    pulsar_grammar_compile_stats stats() const {
        std::lock_guard<std::mutex> lk(st_->mu);
        pulsar_grammar_compile_stats s = st_->counts;
        s.running = st_->running;
        s.waiting = st_->waiting;
        return s;
    }

private:
    /// The deadline clock (monotonic).
    using clock = std::chrono::steady_clock;
    /// A job's identity: the compiler and the exact input.
    using key_type = std::pair<const void *, std::string>;

    /** One compile and the callers waiting on it.  Every field is guarded by state::mu. */
    struct job {
        /// The pending-set key
        key_type key;
        /// The compile, held here while the job is queued; the worker that takes the job takes it
        work_fn work;
        /// Signalled once, when the compile returned
        std::condition_variable cv;
        /// In the slot queue: no worker has taken it yet
        bool queued = false;
        /// The compile returned; `out` / `err` are final
        bool done = false;
        /// When a worker took it off the queue (meaningful once a queued job is no longer queued)
        clock::time_point started;
        /// Callers waiting on it now
        int waiters = 0;
        /// The result on success
        std::optional<Result> out;
        /// The reason when `out` is empty
        std::string err;
    };

    /** Everything shared by the callers and the workers; a worker keeps it alive past the admission object. */
    struct state {
        /// Guards every field below and every job's fields
        std::mutex mu;
        /// Compiles that may run at once
        int slots = 0;
        /// Callers that may wait at once
        int waiters_max = 0;
        /// Workers alive
        int running = 0;
        /// Callers waiting, on any job
        int waiting = 0;
        /// Jobs waiting for a slot, in arrival order
        std::deque<std::shared_ptr<job>> queue;
        /// Jobs queued or running, by key
        std::map<key_type, std::shared_ptr<job>> pending;
        /// The counters (running / waiting are filled by stats())
        pulsar_grammar_compile_stats counts = {};
    };

    /** What a new worker thread is handed. */
    struct start {
        /// The shared state
        std::shared_ptr<state> st;
        /// Its first job
        std::shared_ptr<job> j;
        /// That job's compile
        work_fn work;
    };

    /** Start a detached worker on job `j` (called under the lock; the worker needs the lock only once its compile
     *  has returned).
     *  @param st   the shared state
     *  @param j    the job
     *  @param work its compile
     *  @return 0, or pthread_create's error number (nothing was started) */
    static int start_worker(const std::shared_ptr<state> &st, const std::shared_ptr<job> &j, const work_fn &work) {
        start *go = new start{st, j, work};
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_t th;
        const int rc = pthread_create(&th, &attr, worker_main, go);
        pthread_attr_destroy(&attr);
        if (rc) delete go;
        return rc;
    }

    /** A worker: runs its job, publishes the result, then takes the queue's head on the same slot until the queue
     *  is empty.
     *  @param p the `start` record (owned)
     *  @return NULL */
    static void *worker_main(void *p) {
        std::unique_ptr<start> go((start *)p);
        const std::shared_ptr<state> st = std::move(go->st);
        std::shared_ptr<job> j = std::move(go->j);
        work_fn work = std::move(go->work);
        go.reset();
        for (;;) {
            std::string e;
            std::optional<Result> out = work(&e);   /* no lock held */
            work = nullptr;                         /* its captures (the compiler) are dropped off the lock too */
            std::shared_ptr<job> next;
            {
                std::lock_guard<std::mutex> lk(st->mu);
                j->out = std::move(out);
                j->err = std::move(e);
                j->done = true;
                st->pending.erase(j->key);
                j->cv.notify_all();
                if (st->queue.empty()) {
                    st->running--;
                    return NULL;
                }
                next = std::move(st->queue.front());
                st->queue.pop_front();
                next->queued = false;
                next->started = clock::now();
                work = std::move(next->work);
                next->work = nullptr;
            }
            j = std::move(next);
        }
    }

    /// The shared state
    std::shared_ptr<state> st_;
};

#endif

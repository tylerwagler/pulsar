#include "pulsar_engine_internal.h"

/** Persistent CPU worker pool. Split out of util.cpp in the C++ port.
 *
 * Decode reuses these threads instead of creating pthreads in the token
 * loop. The row partitioning below must stay exactly as the C
 * implementation had it: CPU kernels rely on the same disjoint row split,
 * and the bit-exact gate compares outputs against the pre-port binary. */

namespace pulsar {

namespace {

/** Nested parallel_for calls run serially on the calling thread. */
thread_local int g_parallel_depth = 0;

} // namespace

/** Persistent row-parallel worker pool for the CPU kernels.
 *
 * A GENERATION COUNTER, not a work queue. There is exactly one job in flight:
 * `parallel_for_min_rows` publishes (fn, ctx, n_rows), bumps `generation_`,
 * and broadcasts; every worker wakes, sees a generation it has not run,
 * computes its OWN row range from its thread id, and reports completion by
 * incrementing `done_`. No per-job allocation, no queue, and the row split is
 * derived identically on every thread rather than handed out.
 *
 * That last property is load-bearing: the bit-exact gate compares CPU kernel
 * output against the pre-port binary, and floating-point summation is not
 * associative, so the row partition has to stay exactly what the C
 * implementation used.
 *
 * The calling thread takes the first row range itself rather than idling, so
 * an N-thread pool runs N ranges with N-1 workers.
 */
class ThreadPool {
public:
    /* Lazy init: created on first parallel_for. A shutdown() returns the
     * pool to the never-initialized state, so a later use re-creates it —
     * the same lifecycle the C implementation had. */
    void ensure_init() {
        if (initialized_) return;

        pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

        uint32_t n_threads = 12;
        const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (online_cpus > 0) {
            n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
        }

        const char *env = getenv("PULSAR_THREADS");
        if (env && env[0]) {
            long v = strtol(env, NULL, 10);
            if (v > 0) n_threads = (uint32_t)v;
        }
        if (g_requested_threads > 0) n_threads = g_requested_threads;
        if (n_threads > PULSAR_MAX_THREADS) n_threads = PULSAR_MAX_THREADS;
        if (n_threads == 0) n_threads = 1;

        pthread_mutex_init(&mutex_, NULL);
        pthread_cond_init(&work_cond_, NULL);
        pthread_cond_init(&done_cond_, NULL);
        n_threads_ = n_threads;
        n_workers_ = n_threads > 0 ? n_threads - 1 : 0;
        generation_ = 0;
        done_ = 0;
        shutdown_ = false;
        initialized_ = true;

        for (uint32_t i = 1; i < n_threads; i++) {
            if (pthread_create(&threads_[i], NULL, worker_trampoline,
                               (void *)(uintptr_t)i) != 0) {
                pulsar_die("failed to create worker thread");
            }
        }
    }

    /** Stop the workers, destroy the sync primitives, and zero the pool back to
     * its never-initialized state so a later parallel_for re-creates it. */
    void shutdown() {
        if (!initialized_) return;

        pthread_mutex_lock(&mutex_);
        shutdown_ = true;
        generation_++;
        pthread_cond_broadcast(&work_cond_);
        pthread_mutex_unlock(&mutex_);

        for (uint32_t i = 1; i < n_threads_; i++) {
            pthread_join(threads_[i], NULL);
        }

        pthread_cond_destroy(&done_cond_);
        pthread_cond_destroy(&work_cond_);
        pthread_mutex_destroy(&mutex_);
        memset(this, 0, sizeof(*this));
    }

    /* Run a row-parallel CPU kernel, falling back to serial execution for
     * small jobs or nested calls where spawning more work would only add
     * latency. */
    void parallel_for_min_rows(uint64_t n_rows, pulsar_parallel_fn fn, void *ctx,
                               uint64_t min_parallel_rows) {
        ensure_init();

        if (g_parallel_depth > 0 || n_threads_ <= 1 || n_rows < min_parallel_rows) {
            fn(ctx, 0, n_rows);
            return;
        }

        pthread_mutex_lock(&mutex_);
        fn_ = fn;
        ctx_ = ctx;
        n_rows_ = n_rows;
        done_ = 0;
        generation_++;
        pthread_cond_broadcast(&work_cond_);

        const uint64_t rows_per_thread = (n_rows + n_threads_ - 1) / n_threads_;
        uint64_t main_row1 = rows_per_thread;
        if (main_row1 > n_rows) main_row1 = n_rows;
        pthread_mutex_unlock(&mutex_);

        if (main_row1 > 0) {
            g_parallel_depth++;
            fn(ctx, 0, main_row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&mutex_);
        while (done_ < n_workers_) {
            pthread_cond_wait(&done_cond_, &mutex_);
        }
        pthread_mutex_unlock(&mutex_);
    }

private:
    /** pthread entry point: unpacks the thread id from `arg` and enters
     * worker_main() on the singleton pool. */
    static void *worker_trampoline(void *arg);

    /** Worker loop: wait for a new generation, compute this thread's row range
     * from `tid`, run the kernel, count in as done. Exits on shutdown. */
    void worker_main(uint32_t tid) {
        uint32_t seen_generation = 0;

        for (;;) {
            pthread_mutex_lock(&mutex_);
            while (seen_generation == generation_ && !shutdown_) {
                pthread_cond_wait(&work_cond_, &mutex_);
            }
            if (shutdown_) {
                pthread_mutex_unlock(&mutex_);
                return;
            }

            seen_generation = generation_;
            pulsar_parallel_fn fn = fn_;
            void *ctx = ctx_;
            const uint64_t n_rows = n_rows_;
            const uint32_t n_threads = n_threads_;
            pthread_mutex_unlock(&mutex_);

            const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
            const uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                g_parallel_depth++;
                fn(ctx, row0, row1);
                g_parallel_depth--;
            }

            pthread_mutex_lock(&mutex_);
            done_++;
            if (done_ == n_workers_) {
                pthread_cond_signal(&done_cond_);
            }
            pthread_mutex_unlock(&mutex_);
        }
    }

    pthread_t threads_[PULSAR_MAX_THREADS] = {};  ///< worker threads; [0] is unused (the caller is thread 0)
    pthread_mutex_t mutex_ = {};        ///< guards every field below
    pthread_cond_t work_cond_ = {};     ///< broadcast to release workers into a new generation
    pthread_cond_t done_cond_ = {};     ///< signalled when the last worker finishes
    uint32_t n_threads_ = 0;            ///< total row ranges, including the caller's
    uint32_t n_workers_ = 0;            ///< spawned threads: n_threads_ - 1
    /** Bumped once per dispatch. A worker compares it against the generation it
     * last ran, which is what lets one broadcast release everyone exactly once
     * with no queue and no per-worker flag. */
    uint32_t generation_ = 0;
    uint32_t done_ = 0;                 ///< workers finished this generation
    bool initialized_ = false;          ///< the pool is up (lazy init on first use)
    bool shutdown_ = false;             ///< workers should exit their loop
    pulsar_parallel_fn fn_ = nullptr;   ///< the kernel for the in-flight job
    void *ctx_ = nullptr;               ///< its argument bundle
    uint64_t n_rows_ = 0;               ///< rows to divide across the ranges
};

static ThreadPool g_pool;

void *ThreadPool::worker_trampoline(void *arg) {
    g_pool.worker_main((uint32_t)(uintptr_t)arg);
    return NULL;
}

} // namespace pulsar



uint32_t g_requested_threads;



void pulsar_threads_shutdown(void) {
    pulsar::g_pool.shutdown();
}



void pulsar_parallel_for_min_rows(uint64_t n_rows, pulsar_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    pulsar::g_pool.parallel_for_min_rows(n_rows, fn, ctx, min_parallel_rows);
}



void pulsar_parallel_for(uint64_t n_rows, pulsar_parallel_fn fn, void *ctx) {
    pulsar_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

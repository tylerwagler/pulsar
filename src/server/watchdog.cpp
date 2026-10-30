/* The stall watchdog (pulsar_server_internal.h, "The stall watchdog").
 *
 * 2026-10-29, the work box: the worker spun inside libcuda for 36 minutes (a GPU fault on a managed KV page that
 * nvidia-uvm could not service at the kernel's min watermark) while /health said "ok" and the log's last line was a
 * prefill chunk.  This thread turns that into one loud dump and a /health verdict.  It never touches the engine or a
 * worker-owned structure: it reads the worker's marks (pulsar_stall.h), the snapshot the worker publishes under mu
 * (publish_metrics_snapshot), /proc, and -- once per stall -- asks the worker for its own backtrace by signal. */
#include "pulsar_server_internal.h"

#include <dirent.h>
#include <execinfo.h>

bool server_stall_limit_parse(const char *v, int *limit_s, char *err, size_t errlen) {
    *limit_s = PULSAR_STALL_WATCHDOG_DEFAULT_S;
    if (!v || !v[0]) return true;
    char *end = NULL;
    errno = 0;
    const long n = strtol(v, &end, 10);
    if (!isdigit((unsigned char)v[0]) || errno != 0 || end == v || *end != '\0' || n < 0 || n > 86400) {
        snprintf(err, errlen, "'%s' is not a whole number of seconds in 0..86400 (0 = off)", v);
        return false;
    }
    *limit_s = (int)n;
    return true;
}

int64_t server_stall_seconds(int64_t now_ms, int64_t beat_ms, bool busy, int limit_s) {
    if (limit_s <= 0 || !busy || now_ms < beat_ms) return 0;
    const int64_t s = (now_ms - beat_ms) / 1000;
    return s >= limit_s ? s : 0;
}

namespace {

/* The worker's backtrace, written by the worker itself from the signal (backtrace_symbols_fd allocates nothing).
 * backtrace() is warmed at start so its first call (which loads the unwinder) never runs inside the handler. */
void stall_backtrace_handler(int) {
    void *frames[64];
    const int n = backtrace(frames, 64);
    static const char head[] = "pulsar-server: STALL: the worker's own backtrace (from its signal handler):\n";
    if (write(STDERR_FILENO, head, sizeof head - 1) < 0) return;
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
}

/* "/proc/self/task/<tid>/stat": the name, the state and the user / system CPU ticks of one thread */
bool thread_stat(int tid, char *comm, size_t commlen, char *state, unsigned long long *utime,
                 unsigned long long *stime) {
    char path[64], buf[1024];
    snprintf(path, sizeof path, "/proc/self/task/%d/stat", tid);
    FILE *fp = fopen(path, "r");
    if (!fp) return false;
    const size_t n = fread(buf, 1, sizeof buf - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    const char *lp = strchr(buf, '('), *rp = strrchr(buf, ')');
    if (!lp || !rp || rp < lp) return false;
    snprintf(comm, commlen, "%.*s", (int)(rp - lp - 1), lp + 1);
    /* after ") ": state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime */
    unsigned long long f[12] = {0};
    char st = '?';
    if (sscanf(rp + 2, "%c %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &st, &f[0], &f[1], &f[2],
               &f[3], &f[4], &f[5], &f[6], &f[7], &f[8], &f[9], &f[10], &f[11]) < 13)
        return false;
    *state = st;
    *utime = f[10];
    *stime = f[11];
    return true;
}

void thread_wchan(int tid, char *out, size_t outlen) {
    char path[64];
    snprintf(path, sizeof path, "/proc/self/task/%d/wchan", tid);
    out[0] = '\0';
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    const size_t n = fread(out, 1, outlen - 1, fp);
    fclose(fp);
    out[n] = '\0';
}

uint64_t meminfo_kib(const char *key) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long v = 0;
    const size_t kl = strlen(key);
    while (fgets(line, sizeof line, fp))
        if (!strncmp(line, key, kl) && sscanf(line + kl, " %llu", &v) == 1) break;
    fclose(fp);
    return v;
}

const char *const k_phase_names[] = {"idle", "restore", "prefill-cold", "prefill", "decode-init", "decode", "finish",
                                     "done"};

void *watchdog_main(void *arg) {
    server *s = (server *)arg;
    uint64_t seen = g_pulsar_stall.beats.load(std::memory_order_relaxed);
    int64_t beat_ms = pulsar_stall_now_ms();
    bool dumped = false, busy = false;
    while (!__atomic_load_n(&s->watchdog_quit, __ATOMIC_RELAXED)) {
        usleep(1000000);
        const int64_t now = pulsar_stall_now_ms();
        const uint64_t beats = g_pulsar_stall.beats.load(std::memory_order_relaxed);
        if (beats != seen) {
            seen = beats;
            beat_ms = now;
            if (dumped) {
                server_log(PULSAR_LOG_DEFAULT, "pulsar-server: STALL cleared: the worker marked \"%s\" again",
                           g_pulsar_stall.where.load(std::memory_order_relaxed));
                dumped = false;
            }
        }
        /* jobs queued or running?  The worker does not hold mu across a CUDA call, but if it ever does, a mu that
         * stays taken is itself a sign: keep the last answer, busy unless it was known idle */
        if (pthread_mutex_trylock(&s->mu) == 0) {
            busy = s->n_generating > 0 || s->n_queued > 0;
            pthread_mutex_unlock(&s->mu);
        }
        const int64_t stalled = server_stall_seconds(now, beat_ms, busy, s->stall_limit_s);
        __atomic_store_n(&s->stall_s, stalled, __ATOMIC_RELAXED);
        if (stalled > 0 && !dumped) {
            dumped = true;
            s->watchdog_dump(stalled);
        }
    }
    return NULL;
}

}  // namespace

bool server::watchdog_start() {
    if (stall_limit_s <= 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: stall watchdog OFF (PULSAR_STALL_WATCHDOG_S=0)");
        return true;
    }
    void *warm[4];
    (void)backtrace(warm, 4);   /* loads the unwinder now, never inside the handler */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = stall_backtrace_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR2, &sa, NULL) != 0) return false;
    __atomic_store_n(&watchdog_quit, false, __ATOMIC_RELAXED);
    if (pthread_create(&watchdog_thread, NULL, watchdog_main, this) != 0) return false;
    watchdog_running = true;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: stall watchdog ON: %d s without a worker mark while jobs wait is a stall "
               "(PULSAR_STALL_WATCHDOG_S; /health then reports \"stalled\")",
               stall_limit_s);
    return true;
}

void server::watchdog_stop() {
    if (!watchdog_running) return;
    __atomic_store_n(&watchdog_quit, true, __ATOMIC_RELAXED);
    pthread_join(watchdog_thread, NULL);
    watchdog_running = false;
}

void server::watchdog_dump(int64_t stalled_s) {
    auto *s = this;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const char *where = g_pulsar_stall.where.load(std::memory_order_relaxed);
    const int64_t where_ms = g_pulsar_stall.where_ms.load(std::memory_order_relaxed);
    const double where_age = where_ms ? (double)(pulsar_stall_now_ms() - where_ms) / 1000.0 : 0.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: STALL: the worker has made no progress for %lld s (limit %d s, PULSAR_STALL_WATCHDOG_S): "
               "last section entered \"%s\" %.1f s ago, %llu marks so far",
               (long long)stalled_s, s->stall_limit_s, where ? where : "(none yet)", where_age,
               (unsigned long long)g_pulsar_stall.beats.load(std::memory_order_relaxed));
    /* memory: what MemAvailable, swap and the KV figures read now */
    const uint64_t avail = server_mem_available_bytes();
    const uint64_t swap_total = meminfo_kib("SwapTotal:"), swap_free = meminfo_kib("SwapFree:");
    /* the worker's published snapshot (under mu; trylock for up to ~1 s -- the dump never blocks on a held lock) */
    bool locked = false;
    for (int i = 0; i < 100 && !locked; i++) {
        locked = pthread_mutex_trylock(&s->mu) == 0;
        if (!locked) usleep(10000);
    }
    if (!locked) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: STALL:   mu is held (by the worker?): no queue / slot snapshot");
    } else {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: STALL:   queue %d waiting, %d running; KV touched %.2f GiB, reserved %.2f GiB, "
                   "guard budget %.2f GiB (ceiling %.2f); MemAvailable %.2f GiB, floor %.2f GiB, swap used %.2f of "
                   "%.2f GiB",
                   s->n_queued, s->n_generating, (double)s->m_kv_touched.raw() / gib,
                   (double)s->m_kv_reserved.raw() / gib, (double)s->m_kv_budget.raw() / gib,
                   (double)s->guard_touched_ceiling.raw() / gib, (double)avail / gib,
                   (double)server_mem_floor().raw() / gib, (double)(swap_total - swap_free) / (1024.0 * 1024.0),
                   (double)swap_total / (1024.0 * 1024.0));
        for (int i = 0; i < s->n_slots; i++) {
            const int ph = s->m_slot_phase[i];
            if (ph == 0 && s->m_slot_pos[i] == 0) continue;   /* idle and empty */
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: STALL:   slot %d bank %d: %s, position %d of %d, prefill %d/%d%s%s", i,
                       s->slots[i].bank, ph >= 0 && ph < 8 ? k_phase_names[ph] : "?", s->m_slot_pos[i],
                       s->m_slot_ctx[i], s->m_slot_prefill_done[i], s->m_slot_prefill_total[i],
                       s->slots[i].spilled ? ", spilled" : "", s->slots[i].preempted ? ", preempted" : "");
        }
        pthread_mutex_unlock(&s->mu);
    }
    /* every thread: name, state, CPU ticks, where the kernel has it waiting */
    const int wtid = __atomic_load_n(&s->worker_tid, __ATOMIC_RELAXED);
    if (DIR *d = opendir("/proc/self/task")) {
        while (struct dirent *de = readdir(d)) {
            const int tid = atoi(de->d_name);
            if (tid <= 0) continue;
            char comm[64], wchan[128];
            char st = '?';
            unsigned long long ut = 0, stt = 0;
            if (!thread_stat(tid, comm, sizeof comm, &st, &ut, &stt)) continue;
            thread_wchan(tid, wchan, sizeof wchan);
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: STALL:   thread %d %-16s %c utime %llu stime %llu wchan %s%s",
                       tid, comm, st, ut, stt, wchan[0] ? wchan : "-", tid == wtid ? "  <- worker" : "");
        }
        closedir(d);
    }
    /* the worker's own stack: it prints it from SIGUSR2 (a thread spinning in the driver still takes a signal) */
    if (wtid > 0) (void)pthread_kill(s->worker_thread, SIGUSR2);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: STALL: dump done; /health reports status \"stalled\" until the worker marks again "
               "(gdb -p %d -batch -ex 'thread apply all bt' for every stack; dmesg for nvidia-uvm / page-allocation "
               "failures)",
               (int)getpid());
}

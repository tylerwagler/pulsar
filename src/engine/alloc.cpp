#include "pulsar_engine_internal.h"

/* Checked allocators and the decode-phase allocation guard. Split out of
 * util.cpp in the C++ port. The guard is process-wide state armed around
 * phases that must not allocate (decode reuses preallocated scratch). */

namespace pulsar {

/** Makes host allocation inside a guarded phase FATAL, to catch a decode step
 * that has stopped reusing its preallocated scratch.
 *
 * An opt-in developer tool, not a production safeguard: it arms only when
 * PULSAR_ALLOC_GUARD is set, read once, so the begin/end call sites on the hot
 * path are a hard no-op otherwise. Process-wide state, which is fine because it
 * is a single-threaded diagnostic and never on in a served build.
 */
class AllocGuard {
public:
    /* Only arms when PULSAR_ALLOC_GUARD is set in the environment, so the
     * begin/end call sites are a hard no-op in production and the guard is an
     * opt-in developer tool: `PULSAR_ALLOC_GUARD=1` makes any host allocation
     * inside a guarded phase fatal, to catch a decode step that has stopped
     * reusing its preallocated scratch. Read once. */
    /** Arm the guard for `phase`, whose name appears in the failure message.
     * No-op unless PULSAR_ALLOC_GUARD was set at first use. */
    void begin(const char *phase) {
        static const int armed = getenv("PULSAR_ALLOC_GUARD") != nullptr;
        if (!armed) return;
        phase_ = phase;
        enabled_ = true;
    }

    /** Disarm. Safe to call when never armed. */
    void end() {
        enabled_ = false;
        phase_ = nullptr;
    }

    /** Called from every checked allocator. Prints the offending operation and
     * EXITS when armed -- the point is to stop at the allocation, where the
     * stack still names the caller, rather than to report it later. */
    void check(const char *op, size_t size) const {
        if (!enabled_) return;
        fprintf(stderr,
                "pulsar: internal allocation during %s: %s(%zu). "
                "decode is expected to reuse preallocated scratch buffers.\n",
                phase_ ? phase_ : "guarded phase",
                op,
                size);
        exit(1);
    }

private:
    bool enabled_ = false;          ///< a guarded phase is currently open
    const char *phase_ = nullptr;   ///< its name, for the failure message; borrowed
};

static AllocGuard g_alloc_guard;

} // namespace pulsar



void pulsar_alloc_guard_begin(const char *phase) {
    pulsar::g_alloc_guard.begin(phase);
}



void pulsar_alloc_guard_end(void) {
    pulsar::g_alloc_guard.end();
}



void *xcalloc(size_t n, size_t size) {
    pulsar::g_alloc_guard.check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xmalloc(size_t size) {
    pulsar::g_alloc_guard.check("malloc", size);
    void *p = malloc(size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xrealloc(void *ptr, size_t size) {
    pulsar::g_alloc_guard.check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) pulsar_die("out of memory");
    return p;
}






char *pulsar_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = static_cast<char *>(xmalloc(n + 1));
    memcpy(p, s, n + 1);
    return p;
}

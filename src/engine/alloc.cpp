#include "pulsar_engine_internal.h"

/* Checked allocators and the decode-phase allocation guard. Split out of
 * util.cpp in the C++ port. The guard is process-wide state armed around
 * phases that must not allocate (decode reuses preallocated scratch). */

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xrealloc(void *ptr, size_t size) {
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

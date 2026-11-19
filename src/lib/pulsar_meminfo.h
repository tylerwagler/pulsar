/** @file
 *  The host's memory readings from a meminfo-format file: one field reader and the one MemAvailable reading (L442). */
/* /proc/meminfo, read in ONE place (L442).  Until L442 MemAvailable was parsed four times -- the server's admission
 * reading (server_sched.cpp), the engine's read-ahead admit (prefill_loop.cpp), /metrics' whole-box reconciliation
 * (http_server.cpp) and, as a generic field reader, the stall dump's swap line (watchdog.cpp) -- with three spellings
 * of the same parse.  They agreed on the field, its unit (KiB) and "0 when unreadable or missing"; only the generic
 * reader told a missing field from an unreadable file.  That reader is now the one (meminfo_kib), and MemAvailable
 * is pulsar_mem_available, the typed host reading (pulsar_units.h HostBytes) every caller takes.
 *
 * Never on a token or layer hot path: one open and a short scan a call (a provisioning attempt, a prefill quantum, a
 * persist, the guard's live budget at most every 0.25 s, a /metrics scrape, a read-ahead arm). */
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lib/pulsar_units.h"

/** What meminfo_kib found (L335: a missing key and an unreadable file are told apart). */
enum meminfo_read {
    MEMINFO_FOUND,       ///< the field was read
    MEMINFO_MISSING,     ///< the file has no such field
    MEMINFO_UNREADABLE,  ///< the file would not open
};

/** One field of a meminfo-format file: the first line that starts with `key` and has a number after it.
 *  @param path the meminfo file ("/proc/meminfo"; a test passes its own)
 *  @param key a field with its colon ("SwapTotal:", "MemAvailable:")
 *  @param kib receives the field's value in KiB (0 unless found)
 *  @return MEMINFO_FOUND, MEMINFO_MISSING (the file has no such field) or MEMINFO_UNREADABLE (it would not open) */
inline meminfo_read meminfo_kib(const char *path, const char *key, uint64_t *kib) {
    *kib = 0;
    FILE *fp = fopen(path, "r");
    if (!fp) return MEMINFO_UNREADABLE;
    char line[256];
    unsigned long long v = 0;
    const size_t kl = strlen(key);
    meminfo_read r = MEMINFO_MISSING;
    while (fgets(line, sizeof line, fp))
        if (!strncmp(line, key, kl) && sscanf(line + kl, " %llu", &v) == 1) {
            r = MEMINFO_FOUND;
            *kib = v;
            break;
        }
    fclose(fp);
    return r;
}

/** The host's MemAvailable from a meminfo-format file, in bytes; 0 when the file is unreadable or has no such field
 *  (every caller fails closed on 0).  Coarse by design: on the GB10, driver 610's UVM accounting lags MemAvailable and
 *  under UVM pressure MemTotal itself shrinks -- a belt-and-suspenders reading beside the ledger, not a gauge.
 *  @param path the meminfo file (a test passes its own) */
/// @return MemAvailable, bytes (0 = unreadable)
inline HostBytes pulsar_mem_available_from(const char *path) {
    uint64_t kib = 0;
    (void)meminfo_kib(path, "MemAvailable:", &kib);   /* 0 unless found */
    return HostBytes{kib * 1024ull};   /* the kernel's reading enters the host's unit here */
}

/** The host's MemAvailable now (pulsar_mem_available_from over /proc/meminfo): the one MemAvailable reading. */
/// @return MemAvailable, bytes (0 = unreadable)
inline HostBytes pulsar_mem_available(void) { return pulsar_mem_available_from("/proc/meminfo"); }

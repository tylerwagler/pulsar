/* Page-cache-neutral streaming writes (L261).
 *
 * A disk-KV snapshot is ~800 MB for a 300k-token conversation, written once as
 * the staged payload and again into the final .kv (and, on a TP pair, by every
 * rank).  Through the page cache that is ~1.6 GB of dirty pages per rank on a
 * box with ~16 GB free, and the kernel made room by swapping the server's own
 * memory out -- the bursts seconds before both L258 hangs.  This helper writes
 * the same bytes but hands every PULSAR_WRITEBACK_STEP of them to the disk and
 * drops them from the cache as it goes, so a store's footprint stays bounded.
 *
 * Header-only; the engine's payload writer and the segment store share it. */
#pragma once

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#define PULSAR_WRITEBACK_STEP (UINT64_C(64) << 20)

typedef struct {
    FILE *fp;        /* NULL: writeback off (a read stream, or not a regular file) */
    off_t synced;    /* bytes already pushed to disk and dropped from the cache */
} pulsar_writeback;

static inline void pulsar_writeback_init(pulsar_writeback *w, FILE *fp) {
    w->fp = fp;
    w->synced = fp ? ftello(fp) : 0;
    if (w->synced < 0) w->fp = NULL;
}

/* Push [synced, end) to disk and drop it from the cache.  `wait` blocks until
 * the range is written (required before DONTNEED can drop dirty pages). */
static inline void pulsar_writeback_range(pulsar_writeback *w, off_t end) {
    if (!w->fp || end <= w->synced) return;
    const int fd = fileno(w->fp);
    (void)sync_file_range(fd, w->synced, end - w->synced,
                          SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                          SYNC_FILE_RANGE_WAIT_AFTER);
    (void)posix_fadvise(fd, w->synced, end - w->synced, POSIX_FADV_DONTNEED);
    w->synced = end;
}

/* Call after writing: once a step's worth has accumulated, flush stdio and
 * write it back. */
static inline void pulsar_writeback_step(pulsar_writeback *w) {
    if (!w->fp) return;
    const off_t pos = ftello(w->fp);
    if (pos < 0 || (uint64_t)(pos - w->synced) < PULSAR_WRITEBACK_STEP) return;
    if (fflush(w->fp) != 0) return;
    pulsar_writeback_range(w, pos);
}

/* Write back and drop everything written so far (the caller still fsyncs). */
static inline void pulsar_writeback_finish(pulsar_writeback *w) {
    if (!w->fp) return;
    if (fflush(w->fp) != 0) return;
    const off_t pos = ftello(w->fp);
    if (pos > 0) pulsar_writeback_range(w, pos);
}

/* Drop a whole file's cached pages after its fsync. */
static inline void pulsar_writeback_drop_file(FILE *fp) {
    if (fp) (void)posix_fadvise(fileno(fp), 0, 0, POSIX_FADV_DONTNEED);
}

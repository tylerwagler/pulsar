/* Engram n-gram hashing, host side (PLAN 95 phase 4 / L218 phase 0c).
 *
 * Mirrors engram.py's `NgramHashState.forward` exactly.  The reference is
 * gate-baseline/l218-v41/engram_hash.py and the layout it emitted
 * (engram-layout-v41.json); tests/engram_hash_test.cpp pins this against vectors
 * generated from that layout for both engram layers.
 *
 * The host is the right side of the boundary for this: the ids are known before
 * the forward, so every bucket row a token needs is computable ahead of the GPU
 * work.  That prefetchability is the whole reason a ~189 GiB disk-resident table
 * is viable on one box.
 *
 * The arithmetic is deliberately plain `%` on non-negative int64 -- see the
 * invariant argued at the declaration in pulsar_engine_internal.h.  The one
 * thing this file must not do is let a corrupt compressed id through, because
 * a negative `tok` would set the sign bit of a product and the XOR below would
 * then produce a row index that is still in range but points at the wrong
 * bucket.  Wrong-and-plausible is the failure mode worth refusing. */

#include "pulsar_engine_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

void pulsar_engram_hash_pos(const pulsar_engram_layout *L, uint32_t layer,
                            const int32_t *ids, uint32_t n_ids, uint32_t pos,
                            uint32_t *cols) {
    /* Every one of these is a caller bug, and returning quietly would leave
     * `cols` uninitialised -- the caller would then read stale row indices that
     * still look like valid rows.  Refuse instead. */
    if (!L || !cols || !ids || L->n_vocab == 0 || L->compressed_vocab == 0 ||
        layer >= L->n_layers || layer >= PULSAR_ENGRAM_MAX_LAYERS || pos >= n_ids) {
        pulsar_die("engram hash: called with an out-of-range layout, layer or position");
    }
    const int64_t  *mult   = L->multipliers + (uint64_t)layer * PULSAR_ENGRAM_MAX_NGRAM;
    const uint32_t *primes = L->primes + (uint64_t)layer * (PULSAR_ENGRAM_MAX_NGRAM - 1u) * PULSAR_ENGRAM_N_HEADS;
    const uint64_t *offs   = L->offsets + (uint64_t)layer * PULSAR_ENGRAM_N_COLS;

    /* The n-gram window is the token at `pos` and the (max_ngram-1) before it;
     * positions before the sequence starts take the pad id.  `pos < i` is also
     * what keeps `pos - i` from wrapping -- both are uint32_t, so the pad branch
     * must stay on top of the subtraction rather than beside it. */
    int64_t rolling = 0;
    for (uint32_t i = 0; i < PULSAR_ENGRAM_MAX_NGRAM; i++) {
        int64_t tok;
        if (pos < i) {
            tok = (int64_t)L->pad_compressed_id;
        } else {
            const int32_t tid = ids[pos - i];
            if (tid < 0 || (uint32_t)tid >= L->n_vocab) {
                pulsar_die("engram hash: token id outside the layout's token map");
            }
            tok = (int64_t)L->token_map[tid];
        }
        if (tok < 0 || (uint64_t)tok >= L->compressed_vocab) {
            pulsar_die("engram hash: compressed id outside the layout's compressed vocab");
        }
        const int64_t prod = tok * mult[i];
        if (i == 0u) { rolling = prod; continue; }
        rolling ^= prod;
        const uint32_t base = (i - 1u) * PULSAR_ENGRAM_N_HEADS;
        for (uint32_t h = 0; h < PULSAR_ENGRAM_N_HEADS; h++) {
            cols[base + h] = (uint32_t)(rolling % (int64_t)primes[base + h]) +
                             (uint32_t)offs[base + h];
        }
    }
}


/* ---- the table -------------------------------------------------------------- */

int pulsar_engram_table_open(pulsar_engram_table *t, const char *path, uint32_t layer,
                             uint64_t n_rows_expected) {
    if (!t || !path) return 0;
    memset(t, 0, sizeof *t);
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "pulsar: engram table %s: %s\n", path, strerror(errno));
        return 0;
    }
    unsigned char hdr[PULSAR_ENGRAM_HDR_BYTES];
    if (pread(fd, hdr, sizeof hdr, 0) != (ssize_t)sizeof hdr || memcmp(hdr, "PENGRAM1", 8) != 0) {
        fprintf(stderr, "pulsar: engram table %s: not a PENGRAM1 row file -- refusing\n", path);
        close(fd);
        return 0;
    }
    uint32_t version = 0, file_layer = 0, row_bytes = 0, dim = 0, n_scale = 0;
    uint64_t n_rows = 0;
    memcpy(&version, hdr + 8, 4);
    memcpy(&file_layer, hdr + 12, 4);
    memcpy(&n_rows, hdr + 16, 8);
    memcpy(&row_bytes, hdr + 24, 4);
    memcpy(&dim, hdr + 28, 4);
    memcpy(&n_scale, hdr + 32, 4);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    const uint64_t want_size = PULSAR_ENGRAM_HDR_BYTES + n_rows * PULSAR_ENGRAM_ROW_BYTES;
    if (version != 1u || file_layer != layer || row_bytes != PULSAR_ENGRAM_ROW_BYTES || dim != 256u ||
        n_scale != 8u || n_rows != n_rows_expected || (uint64_t)st.st_size != want_size) {
        fprintf(stderr, "pulsar: engram table %s: header says version %u layer %u rows %llu record %u (%u+%u), "
                        "size %lld; this model wants layer %u rows %llu record %u size %llu -- refusing\n",
                path, version, file_layer, (unsigned long long)n_rows, row_bytes, dim, n_scale,
                (long long)st.st_size, layer, (unsigned long long)n_rows_expected,
                PULSAR_ENGRAM_ROW_BYTES, (unsigned long long)want_size);
        close(fd);
        return 0;
    }
    t->parts = (pulsar_engram_part *)xcalloc(1, sizeof *t->parts);
    t->parts[0].fd = fd;
    t->parts[0].base = PULSAR_ENGRAM_HDR_BYTES;
    t->n_parts = 1;
    t->rows_per_part = n_rows;
    t->row_bytes = PULSAR_ENGRAM_ROW_BYTES;
    t->layer = layer;
    t->n_rows = n_rows;
    t->path = pulsar_strdup(path);
    return 1;
}

int pulsar_engram_table_open_parts(pulsar_engram_table *t, uint32_t layer, uint32_t n_parts,
                                   const char *const *paths, const uint64_t *bases,
                                   uint64_t rows_per_part, uint32_t row_bytes, uint64_t n_rows) {
    if (!t) return 0;
    memset(t, 0, sizeof *t);
    if (!paths || !bases || n_parts == 0 || rows_per_part == 0 || row_bytes == 0 ||
        n_rows == 0 || (n_rows + rows_per_part - 1) / rows_per_part != n_parts) {
        fprintf(stderr, "pulsar: row table: %u parts of %llu rows cannot hold %llu rows -- refusing\n",
                n_parts, (unsigned long long)rows_per_part, (unsigned long long)n_rows);
        return 0;
    }
    t->parts = (pulsar_engram_part *)xcalloc(n_parts, sizeof *t->parts);
    for (uint32_t i = 0; i < n_parts; i++) t->parts[i].fd = -1;
    t->n_parts = n_parts;
    t->rows_per_part = rows_per_part;
    t->row_bytes = row_bytes;
    t->layer = layer;
    t->n_rows = n_rows;
    t->path = pulsar_strdup(paths[0]);
    for (uint32_t i = 0; i < n_parts; i++) {
        if (!paths[i]) { pulsar_engram_table_close(t); return 0; }
        for (uint32_t j = 0; j < i; j++)
            if (strcmp(paths[j], paths[i]) == 0) { t->parts[i].fd = t->parts[j].fd; break; }
        if (t->parts[i].fd < 0) {
            t->parts[i].fd = open(paths[i], O_RDONLY | O_CLOEXEC);
            if (t->parts[i].fd < 0) {
                fprintf(stderr, "pulsar: row table part %u %s: %s\n", i, paths[i], strerror(errno));
                pulsar_engram_table_close(t);
                return 0;
            }
        }
        const uint64_t rows = i + 1 < n_parts ? rows_per_part : n_rows - (uint64_t)i * rows_per_part;
        struct stat st;
        if (fstat(t->parts[i].fd, &st) != 0 || (uint64_t)st.st_size < bases[i] + rows * row_bytes) {
            fprintf(stderr, "pulsar: row table part %u %s: %llu rows of %u B at offset %llu run past the file "
                            "(%lld B) -- refusing\n", i, paths[i], (unsigned long long)rows, row_bytes,
                    (unsigned long long)bases[i], (long long)st.st_size);
            pulsar_engram_table_close(t);
            return 0;
        }
        t->parts[i].base = bases[i];
    }
    return 1;
}

void pulsar_engram_table_close(pulsar_engram_table *t) {
    if (!t) return;
    for (uint32_t i = 0; t->parts && i < t->n_parts; i++) {
        const int fd = t->parts[i].fd;
        if (fd < 0) continue;
        bool first = true;   /* parts share a file's fd: close each fd once */
        for (uint32_t j = 0; j < i; j++) first &= t->parts[j].fd != fd;
        if (first) close(fd);
    }
    free(t->parts);
    free(t->path);
    memset(t, 0, sizeof *t);
}

/* ---- the pread pool ----------------------------------------------------------
 *
 * A gather is split into slices of consecutive row ids; workers take slices off
 * one queue and pread each row of a slice into its place in the destination.
 * A slice is at most PULSAR_ENGRAM_IO_SLICE rows (a 98k-row prefill gather keeps
 * every thread busy without contending on the lock) and at least one, sized so a
 * gather spreads over the whole pool: a cold row is one NVMe read (~150 us), and
 * a 16-row Qwen decode gather in slices of 4 kept 4 of 16 threads busy, four reads
 * deep -- 1.2 ms the GPU sat idle at the PLE layer of every decode step (nsys,
 * qwen-perf 2026-10-11).  Failures are counted on the gather and reported once at
 * wait.
 *
 * L329 (b2): the pool has a second, BACKGROUND class -- the read-ahead of a prompt's
 * next chunk (pulsar_engram_read_ahead).  Strict priority, so a read-ahead never
 * delays a gather someone waits on (TensorFold left its read-ahead opt-in over
 * exactly that hazard: one FIFO, a decode gather queued behind a chunk's reads):
 *   - a worker takes a background row only when no gather or prefetch has rows left
 *     to take AND none is still being read (hi_busy 0): the device then serves the
 *     foreground's reads alone, but for the background reads already in flight;
 *   - at most PULSAR_ENGRAM_AHEAD_READERS workers read in the background at once,
 *     so a gather arriving finds the rest of the pool idle and at most that many
 *     background reads ahead of it at the device;
 *   - a background slice is ONE row, so a background reader re-checks the
 *     foreground before every read.
 * Background rows are read into a throwaway stack buffer: the point is the page
 * cache, and the real gather reads the same bytes again (the read-ahead changes
 * no byte anyone receives).  A tag (the issuer's session and bank) lets the
 * issuer drop the rows nobody took yet (pulsar_engram_read_ahead_cancel).
 * Why preads on bounded threads and not posix_fadvise(WILLNEED) / readahead(2):
 * both populate the cache too (measured on the NAS copy, 2026-10-27), but they
 * return at once and leave every row's read queued at the device -- 65,536 rows
 * at whatever depth the kernel picks, ahead of the next decode gather, with no
 * completion to count and nothing to cancel. */
#define PULSAR_ENGRAM_IO_SLICE 4u
/** The widest row a background read throws away (its stack buffer; Engram 264 B, Qwen PLE 320 B). */
#define PULSAR_ENGRAM_AHEAD_ROW_MAX 512u

struct pulsar_engram_gather {
    pulsar_engram_io *io;   ///< the pool it was issued on; wait blocks on it
    const pulsar_engram_table *table;
    const uint64_t *rows;
    uint32_t n_rows;        ///< rows to read (a cancelled read-ahead shrinks it to the rows already taken)
    uint32_t slice;         ///< rows a worker takes at once (1 .. PULSAR_ENGRAM_IO_SLICE)
    unsigned char *dst;     ///< NULL on a read-ahead: each row lands in the reader's throwaway buffer
    uint32_t next;          ///< next row index a worker takes (under the pool lock)
    uint32_t done;          ///< rows finished (ok or failed)
    uint32_t failed;        ///< rows whose read failed
    uint64_t first_bad;     ///< the first failing row id, for the message
    int      first_errno;
    int      detached;      ///< a prefetch (L289) or a read-ahead: nobody waits; the worker finishing its last row frees it
    uint64_t tag;           ///< L329 (b2): a read-ahead's issuer (pulsar_engram_read_ahead_cancel)
    pulsar_engram_gather *q_next;
};

struct pulsar_engram_io {
    pthread_mutex_t lock;
    pthread_cond_t  work;   ///< a gather has rows to take
    pthread_cond_t  idle;   ///< a gather finished a row (waiters re-check `done`)
    pulsar_engram_gather *head, *tail;         ///< gathers with rows left to take, FIFO
    pulsar_engram_gather *bg_head, *bg_tail;   ///< L329 (b2): read-aheads with rows left to take, FIFO
    uint32_t hi_busy;       ///< foreground slices being read
    uint32_t bg_busy;       ///< background rows being read (<= PULSAR_ENGRAM_AHEAD_READERS)
    uint64_t bg_pending;    ///< background rows queued, not yet taken (<= PULSAR_ENGRAM_AHEAD_PENDING_MAX)
    pulsar_read_ahead_stats ahead;   ///< the background class's counters (under the lock)
    pthread_t *threads;
    uint32_t n_threads;
    int stopping;
};

static uint64_t engram_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void engram_gather_free(pulsar_engram_gather *g) {
    free(g->dst);
    free((void *)g->rows);
    free(g);
}

/* pread row r of `tb` into d; false with errno set on a failure (0 for an id past the table) */
static bool engram_read_row(const pulsar_engram_table *tb, uint64_t r, unsigned char *d) {
    if (r >= tb->n_rows) { errno = 0; return false; }
    const uint32_t rb = tb->row_bytes;
    const pulsar_engram_part *pt = &tb->parts[r / tb->rows_per_part];
    const off_t off = (off_t)(pt->base + (r % tb->rows_per_part) * rb);
    size_t got = 0;
    while (got < rb) {
        const ssize_t n = pread(pt->fd, d + got, rb - got, off + (off_t)got);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        got += (size_t)n;
    }
    return true;
}

/* L329 (b2): may a worker take a background row now?  (under the lock) */
static bool engram_bg_ready(const pulsar_engram_io *io) {
    return io->bg_head && !io->head && io->hi_busy == 0 && io->bg_busy < PULSAR_ENGRAM_AHEAD_READERS;
}

/* drop g from the background take-queue (under the lock; prev = its predecessor or NULL) */
static void engram_bg_unlink(pulsar_engram_io *io, pulsar_engram_gather *prev, pulsar_engram_gather *g) {
    if (prev) prev->q_next = g->q_next; else io->bg_head = g->q_next;
    if (io->bg_tail == g) io->bg_tail = prev;
    g->q_next = NULL;
}

static void *engram_io_worker(void *arg) {
    pulsar_engram_io *io = (pulsar_engram_io *)arg;
    for (;;) {
        pthread_mutex_lock(&io->lock);
        while (!io->head && !io->stopping && !engram_bg_ready(io)) pthread_cond_wait(&io->work, &io->lock);
        if (io->stopping && !io->head) { pthread_mutex_unlock(&io->lock); return NULL; }   /* read-aheads are dropped */
        const bool bg = !io->head;   /* the foreground first, always */
        pulsar_engram_gather *g = bg ? io->bg_head : io->head;
        const uint32_t i0 = g->next;
        const uint32_t i1 = i0 + g->slice < g->n_rows ? i0 + g->slice : g->n_rows;
        g->next = i1;
        if (g->next >= g->n_rows) {   /* nothing left to take: drop it from its take-queue */
            if (bg) {
                engram_bg_unlink(io, NULL, g);
            } else {
                io->head = g->q_next;
                if (!io->head) io->tail = NULL;
            }
        }
        if (bg) { io->bg_busy++; io->bg_pending -= i1 - i0; } else io->hi_busy++;
        /* a prefetch or a read-ahead wakes one reader (L289); each reader wakes the next while slices remain */
        if (io->head || engram_bg_ready(io)) pthread_cond_signal(&io->work);
        pthread_mutex_unlock(&io->lock);

        uint32_t failed = 0, first_errno = 0;
        uint64_t first_bad = 0;
        const uint64_t t0 = bg ? engram_now_ns() : 0;
        unsigned char sink[PULSAR_ENGRAM_AHEAD_ROW_MAX];
        for (uint32_t i = i0; i < i1; i++) {
            unsigned char *d = g->dst ? g->dst + (uint64_t)i * g->table->row_bytes : sink;
            if (!engram_read_row(g->table, g->rows[i], d)) {
                if (!failed) { first_bad = g->rows[i]; first_errno = (uint32_t)errno; }
                failed++;
            }
        }
        pthread_mutex_lock(&io->lock);
        g->done += i1 - i0;
        if (failed) { if (!g->failed) { g->first_bad = first_bad; g->first_errno = (int)first_errno; } g->failed += failed; }
        if (bg) {
            io->bg_busy--;
            io->ahead.rows_read += i1 - i0 - failed;
            io->ahead.rows_failed += failed;   /* a hint's failure is the later gather's to report */
            io->ahead.bytes_read += (uint64_t)(i1 - i0 - failed) * g->table->row_bytes;
            io->ahead.read_ns += engram_now_ns() - t0;
            if (engram_bg_ready(io)) pthread_cond_signal(&io->work);
        } else if (--io->hi_busy == 0 && engram_bg_ready(io)) {
            pthread_cond_signal(&io->work);   /* the foreground went quiet: the background resumes */
        }
        /* a detached gather is out of the take-queue once its last slice is taken, so its last finisher is its last user */
        const bool reap = g->detached && g->done >= g->n_rows;
        if (!g->detached) pthread_cond_broadcast(&io->idle);
        pthread_mutex_unlock(&io->lock);
        if (reap) engram_gather_free(g);
    }
}

pulsar_engram_io *pulsar_engram_io_create(uint32_t n_threads) {
    if (n_threads == 0) return NULL;
    pulsar_engram_io *io = (pulsar_engram_io *)xcalloc(1, sizeof *io);
    pthread_mutex_init(&io->lock, NULL);
    pthread_cond_init(&io->work, NULL);
    pthread_cond_init(&io->idle, NULL);
    io->threads = (pthread_t *)xcalloc(n_threads, sizeof(pthread_t));
    for (uint32_t i = 0; i < n_threads; i++) {
        if (pthread_create(&io->threads[i], NULL, engram_io_worker, io) != 0) {
            fprintf(stderr, "pulsar: engram io: could not start reader thread %u of %u\n", i, n_threads);
            io->n_threads = i;
            pulsar_engram_io_destroy(io);
            return NULL;
        }
        io->n_threads = i + 1;
    }
    return io;
}

void pulsar_engram_io_destroy(pulsar_engram_io *io) {
    if (!io) return;
    pthread_mutex_lock(&io->lock);
    io->stopping = 1;
    pthread_cond_broadcast(&io->work);
    pthread_mutex_unlock(&io->lock);
    for (uint32_t i = 0; i < io->n_threads; i++) pthread_join(io->threads[i], NULL);
    /* the workers are gone: a read-ahead still queued has no row in flight (a taken row was finished) */
    for (pulsar_engram_gather *g = io->bg_head, *nx; g; g = nx) {
        nx = g->q_next;
        engram_gather_free(g);
    }
    pthread_mutex_destroy(&io->lock);
    pthread_cond_destroy(&io->work);
    pthread_cond_destroy(&io->idle);
    free(io->threads);
    free(io);
}

static pulsar_engram_gather *engram_gather_issue(pulsar_engram_io *io, const pulsar_engram_table *t,
                                                 const uint64_t *rows, uint32_t n_rows, unsigned char *dst,
                                                 int detached) {
    if (!io || !t || !t->parts || !rows || !dst || n_rows == 0) return NULL;
    pulsar_engram_gather *g = (pulsar_engram_gather *)xcalloc(1, sizeof *g);
    g->detached = detached;
    g->io = io;
    g->table = t;
    g->rows = rows;
    g->n_rows = n_rows;
    const uint32_t even = n_rows / io->n_threads;   /* the pool's share of the gather */
    g->slice = even < 1u ? 1u : even > PULSAR_ENGRAM_IO_SLICE ? PULSAR_ENGRAM_IO_SLICE : even;
    g->dst = dst;
    pthread_mutex_lock(&io->lock);
    if (io->tail) io->tail->q_next = g; else io->head = g;
    io->tail = g;
    /* a gather someone waits on wakes the whole pool now; a prefetch (L289) runs off the critical path, so its
     * issuer -- the draft chain, between MTP steps -- pays one wake and the readers wake each other */
    if (detached) pthread_cond_signal(&io->work);
    else pthread_cond_broadcast(&io->work);
    pthread_mutex_unlock(&io->lock);
    return g;
}

pulsar_engram_gather *pulsar_engram_gather_start(pulsar_engram_io *io, const pulsar_engram_table *t,
                                                 const uint64_t *rows, uint32_t n_rows,
                                                 unsigned char *dst) {
    return engram_gather_issue(io, t, rows, n_rows, dst, 0);
}

void pulsar_engram_prefetch(pulsar_engram_io *io, const pulsar_engram_table *t, const uint64_t *rows, uint32_t n_rows) {
    if (!io || !t || !t->parts || !rows || n_rows == 0) return;
    uint64_t *ids = (uint64_t *)xmalloc((size_t)n_rows * sizeof(uint64_t));
    memcpy(ids, rows, (size_t)n_rows * sizeof(uint64_t));
    unsigned char *dst = (unsigned char *)xmalloc((size_t)n_rows * t->row_bytes);
    (void)engram_gather_issue(io, t, ids, n_rows, dst, 1);   /* the pool owns ids and dst now */
}

uint32_t pulsar_engram_read_ahead(pulsar_engram_io *io, const pulsar_engram_table *t, const uint64_t *rows,
                                  uint32_t n_rows, uint64_t tag) {
    if (!io || !t || !t->parts || !rows || n_rows == 0) return 0;
    if (t->row_bytes > PULSAR_ENGRAM_AHEAD_ROW_MAX)
        pulsar_die("engram read-ahead: a row wider than the background reader's buffer");
    uint64_t *ids = (uint64_t *)xmalloc((size_t)n_rows * sizeof(uint64_t));
    memcpy(ids, rows, (size_t)n_rows * sizeof(uint64_t));
    pulsar_engram_gather *g = (pulsar_engram_gather *)xcalloc(1, sizeof *g);
    g->io = io;
    g->table = t;
    g->rows = ids;   /* the pool owns them now */
    g->n_rows = n_rows;
    g->slice = 1;
    g->detached = 1;
    g->tag = tag;
    pthread_mutex_lock(&io->lock);
    /* bounded: the rows queued and not yet read never pass the cap (each a cold row is about one page of cache) */
    if (io->bg_pending + n_rows > PULSAR_ENGRAM_AHEAD_PENDING_MAX) {
        io->ahead.rows_refused += n_rows;
        pthread_mutex_unlock(&io->lock);
        free(ids);
        free(g);
        return 0;
    }
    if (io->bg_tail) io->bg_tail->q_next = g; else io->bg_head = g;
    io->bg_tail = g;
    io->bg_pending += n_rows;
    io->ahead.issued++;
    io->ahead.rows_queued += n_rows;
    if (engram_bg_ready(io)) pthread_cond_signal(&io->work);
    pthread_mutex_unlock(&io->lock);
    return n_rows;
}

void pulsar_engram_read_ahead_skipped(pulsar_engram_io *io) {
    if (!io) return;
    pthread_mutex_lock(&io->lock);
    io->ahead.skipped++;
    pthread_mutex_unlock(&io->lock);
}

uint32_t pulsar_engram_read_ahead_cancel(pulsar_engram_io *io, uint64_t tag) {
    if (!io) return 0;
    uint32_t dropped = 0;
    pthread_mutex_lock(&io->lock);
    for (pulsar_engram_gather *g = io->bg_head, *prev = NULL, *nx; g; g = nx) {
        nx = g->q_next;
        if (g->tag != tag) { prev = g; continue; }
        dropped += g->n_rows - g->next;
        io->bg_pending -= g->n_rows - g->next;
        g->n_rows = g->next;   /* the rows already taken are the last it reads */
        engram_bg_unlink(io, prev, g);
        /* no row of it in flight: nobody else will finish it; else its last finisher reaps it */
        if (g->done >= g->n_rows) engram_gather_free(g);
    }
    io->ahead.rows_cancelled += dropped;
    pthread_mutex_unlock(&io->lock);
    return dropped;
}

void pulsar_engram_read_ahead_stats(pulsar_engram_io *io, pulsar_read_ahead_stats *out) {
    memset(out, 0, sizeof *out);
    if (!io) return;
    pthread_mutex_lock(&io->lock);
    *out = io->ahead;
    out->in_flight = io->bg_busy;
    out->rows_pending = io->bg_pending;
    pthread_mutex_unlock(&io->lock);
}

/* Blocks until every row is accounted for.  A worker adds its slice to `done`
 * only after its last touch of the handle, so once done == n_rows no thread
 * holds it and it can be freed here. */
int pulsar_engram_gather_wait(pulsar_engram_gather *g) {
    if (!g) return 0;
    pulsar_engram_io *io = g->io;
    pthread_mutex_lock(&io->lock);
    while (g->done < g->n_rows) pthread_cond_wait(&io->idle, &io->lock);
    pthread_mutex_unlock(&io->lock);
    const int ok = g->failed == 0;
    if (!ok)
        fprintf(stderr, "pulsar: engram gather from %s: %u of %u rows failed (first: row %llu, %s)\n",
                g->table->path, g->failed, g->n_rows, (unsigned long long)g->first_bad,
                g->first_errno ? strerror(g->first_errno) : "row id past the table");
    free(g);
    return ok;
}


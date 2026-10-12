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
#include <sys/uio.h>
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
 * A gather is split into slices of consecutive row ids; readers take slices off
 * one queue and pread each row of a slice into its place in the destination.
 * A slice is at most PULSAR_ENGRAM_IO_SLICE rows (a 98k-row prefill gather keeps
 * every thread busy without contending on the lock) and at least one, sized so a
 * gather spreads over the whole pool: a cold row is one NVMe read (~150 us), and
 * a 16-row Qwen decode gather in slices of 4 kept 4 of 16 threads busy, four reads
 * deep -- 1.2 ms the GPU sat idle at the PLE layer of every decode step (nsys,
 * qwen-perf 2026-10-11).  Failures are counted on the gather and reported once at
 * wait.
 *
 * L288 qwen-graph: the pool is for COLD rows.  A gather of at most
 * PULSAR_ENGRAM_INLINE_ROWS rows first copies every row the page cache already
 * holds on the issuing thread (preadv2 RWF_NOWAIT: a few us a row, never a disk
 * wait) and queues only the rest; a prefetch is a page-cache hint on the calling
 * thread (posix_fadvise WILLNEED), not a pool gather.  Every row a decode / verify
 * step gathered went through the pool before, and its lock convoyed (nsys, td305:
 * a reader's lock 454 us median against its 3 us page-cache pread; the waiter
 * blocked up to 10 ms): the verify's PLE wait was 1.7 ms median for 80 rows its
 * prefetch had already warmed, and at 4 streams the drafting host blocked 1.4 ms a
 * round queueing prefetches.  Who reads a row, and when, changes no byte of it. */
#define PULSAR_ENGRAM_IO_SLICE 4u

struct pulsar_engram_gather {
    pulsar_engram_io *io;   ///< the pool it was issued on; wait blocks on it
    const pulsar_engram_table *table;
    const uint64_t *rows;
    uint32_t n_rows;
    uint32_t *take;         ///< the rows the pool reads (indices into rows / dst), owned; NULL = all, in order
    uint32_t n_take;        ///< entries of take (n_rows when take is NULL)
    uint32_t slice;         ///< take entries a reader takes at once (1 .. PULSAR_ENGRAM_IO_SLICE)
    unsigned char *dst;
    uint32_t next;          ///< next take entry a reader takes (under the pool lock)
    uint32_t done;          ///< rows finished (ok or failed), the issuer's own included
    uint32_t failed;        ///< rows whose read failed
    uint64_t first_bad;     ///< the first failing row id, for the message
    int      first_errno;
    pulsar_engram_gather *q_next;
};

struct pulsar_engram_io {
    pthread_mutex_t lock;
    pthread_cond_t  work;   ///< a gather has rows to take
    pthread_cond_t  idle;   ///< a gather finished a row (waiters re-check `done`)
    pulsar_engram_gather *head, *tail;   ///< gathers with rows left to take, FIFO
    pthread_t *threads;
    uint32_t n_threads;
    int stopping;
};

/* Row i of g into its place in dst: 0 = read, else the errno (EAGAIN with `nowait`: not all of it is in the page
 * cache). */
static int engram_read_row(const pulsar_engram_gather *g, uint32_t i, bool nowait) {
    const uint64_t r = g->rows[i];
    const pulsar_engram_table *tb = g->table;
    const uint32_t rb = tb->row_bytes;
    unsigned char *d = g->dst + (uint64_t)i * rb;
    if (r >= tb->n_rows) return EINVAL;
    const pulsar_engram_part *pt = &tb->parts[r / tb->rows_per_part];
    const off_t off = (off_t)(pt->base + (r % tb->rows_per_part) * rb);
    size_t got = 0;
    while (got < rb) {
        struct iovec v = {d + got, rb - got};
        const ssize_t n = preadv2(pt->fd, &v, 1, off + (off_t)got, nowait ? RWF_NOWAIT : 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return errno;
        if (n == 0) return nowait ? EAGAIN : EIO;   /* the file ends inside the row */
        got += (size_t)n;
    }
    return 0;
}

static void *engram_io_worker(void *arg) {
    pulsar_engram_io *io = (pulsar_engram_io *)arg;
    for (;;) {
        pthread_mutex_lock(&io->lock);
        while (!io->head && !io->stopping) pthread_cond_wait(&io->work, &io->lock);
        if (io->stopping && !io->head) { pthread_mutex_unlock(&io->lock); return NULL; }
        pulsar_engram_gather *g = io->head;
        const uint32_t j0 = g->next;
        const uint32_t j1 = j0 + g->slice < g->n_take ? j0 + g->slice : g->n_take;
        g->next = j1;
        if (g->next >= g->n_take) {   /* nothing left to take: drop it from the take-queue */
            io->head = g->q_next;
            if (!io->head) io->tail = NULL;
        }
        pthread_mutex_unlock(&io->lock);

        uint32_t failed = 0;
        int first_errno = 0;
        uint64_t first_bad = 0;
        for (uint32_t j = j0; j < j1; j++) {
            const uint32_t i = g->take ? g->take[j] : j;
            const int err = engram_read_row(g, i, false);
            if (err) { if (!failed) { first_bad = g->rows[i]; first_errno = err; } failed++; }
        }
        pthread_mutex_lock(&io->lock);
        g->done += j1 - j0;
        if (failed) { if (!g->failed) { g->first_bad = first_bad; g->first_errno = first_errno; } g->failed += failed; }
        pthread_cond_broadcast(&io->idle);
        pthread_mutex_unlock(&io->lock);
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
    pthread_mutex_destroy(&io->lock);
    pthread_cond_destroy(&io->work);
    pthread_cond_destroy(&io->idle);
    free(io->threads);
    free(io);
}

pulsar_engram_gather *pulsar_engram_gather_start(pulsar_engram_io *io, const pulsar_engram_table *t,
                                                 const uint64_t *rows, uint32_t n_rows,
                                                 unsigned char *dst) {
    if (!io || !t || !t->parts || !rows || !dst || n_rows == 0) return NULL;
    pulsar_engram_gather *g = (pulsar_engram_gather *)xcalloc(1, sizeof *g);
    g->io = io;
    g->table = t;
    g->rows = rows;
    g->n_rows = n_rows;
    g->dst = dst;
    g->n_take = n_rows;
    if (n_rows <= PULSAR_ENGRAM_INLINE_ROWS) {
        /* the page cache's rows here and now; the pool reads the rest */
        g->take = (uint32_t *)xmalloc((size_t)n_rows * sizeof(uint32_t));
        g->n_take = 0;
        for (uint32_t i = 0; i < n_rows; i++)
            if (engram_read_row(g, i, true) != 0) g->take[g->n_take++] = i;
        g->done = n_rows - g->n_take;
        if (g->n_take == 0) return g;
    }
    const uint32_t even = g->n_take / io->n_threads;   /* the pool's share of the gather */
    g->slice = even < 1u ? 1u : even > PULSAR_ENGRAM_IO_SLICE ? PULSAR_ENGRAM_IO_SLICE : even;
    pthread_mutex_lock(&io->lock);
    if (io->tail) io->tail->q_next = g; else io->head = g;
    io->tail = g;
    /* as many readers as the gather has slices (a decode step's few cold rows wake a few, not the pool) */
    const uint32_t n_slices = (g->n_take + g->slice - 1u) / g->slice;
    if (n_slices >= io->n_threads) pthread_cond_broadcast(&io->work);
    else for (uint32_t k = 0; k < n_slices; k++) pthread_cond_signal(&io->work);
    pthread_mutex_unlock(&io->lock);
    return g;
}

void pulsar_engram_prefetch(pulsar_engram_io *io, const pulsar_engram_table *t, const uint64_t *rows, uint32_t n_rows) {
    if (!io || !t || !t->parts || !rows) return;
    for (uint32_t k = 0; k < n_rows; k++) {
        const uint64_t r = rows[k];
        if (r >= t->n_rows) continue;   /* the gather that reads it reports it */
        const pulsar_engram_part *pt = &t->parts[r / t->rows_per_part];
        (void)posix_fadvise(pt->fd, (off_t)(pt->base + (r % t->rows_per_part) * t->row_bytes), (off_t)t->row_bytes,
                            POSIX_FADV_WILLNEED);
    }
}

/* Blocks until every row is accounted for.  A reader adds its slice to `done`
 * only after its last touch of the handle, so once done == n_rows no thread
 * holds it and it can be freed here; a gather the issuer read whole never
 * entered the pool. */
int pulsar_engram_gather_wait(pulsar_engram_gather *g) {
    if (!g) return 0;
    pulsar_engram_io *io = g->io;
    if (g->n_take > 0) {
        pthread_mutex_lock(&io->lock);
        while (g->done < g->n_rows) pthread_cond_wait(&io->idle, &io->lock);
        pthread_mutex_unlock(&io->lock);
    }
    const int ok = g->failed == 0;
    if (!ok)
        fprintf(stderr, "pulsar: engram gather from %s: %u of %u rows failed (first: row %llu, %s)\n",
                g->table->path, g->failed, g->n_rows, (unsigned long long)g->first_bad,
                g->first_errno ? strerror(g->first_errno) : "row id past the table");
    free(g->take);
    free(g);
    return ok;
}


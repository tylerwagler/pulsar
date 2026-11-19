/** @file content_sample.cpp
 *  @brief L426: the sampled content digest of weights on disk -- the sampling rule, its threaded reads, and a row
 *  table's digest.  What the disk KV store identity's weights word is made of (pulsar_engine_kv_identity). */
#include "pulsar_engine_internal.h"

#include <unistd.h>
#include <atomic>
#include <thread>

void pulsar_content_sample(int fd, FileOff off, TensorBytes bytes_t, std::vector<pulsar_content_read> *plan) {
    const uint64_t B = PULSAR_CONTENT_SAMPLE_BLOCK, bytes = bytes_t.raw();   /* the rule's arithmetic, on the length */
    if (bytes <= 2u * B) {   /* the first and last block would cover it: every byte */
        for (uint64_t at = 0; at < bytes; at += B)
            plan->push_back({fd, off + TensorBytes{at}, (uint32_t)(bytes - at < B ? bytes - at : B)});
        return;
    }
    /* block j of n starts at j (bytes - B) / (n - 1): the first at 0, the last ending at the span's end */
    const uint64_t n = 2u + bytes / PULSAR_CONTENT_SAMPLE_STRIDE;
    for (uint64_t j = 0; j < n; j++)
        plan->push_back({fd, off + TensorBytes{(uint64_t)((unsigned __int128)j * (bytes - B) / (n - 1u))}, (uint32_t)B});
}

bool pulsar_content_sample_fold(const std::vector<pulsar_content_read> &plan, pulsar_kvdigest *d) {
    const size_t n = plan.size();
    std::vector<uint64_t> at(n + 1u, 0u);   /* where each read lands in `buf` */
    for (size_t i = 0; i < n; i++) at[i + 1u] = at[i] + plan[i].bytes;
    std::vector<uint8_t> buf(at[n]);
    std::atomic<size_t> next{0};
    std::atomic<int> failed{0};   /* the first failure's errno */
    /* a reader takes the next read until none is left; a read is whole or a failure (a short one is a file that
     * ends inside its own declaration) */
    const auto reader = [&] {
        for (size_t i; (i = next.fetch_add(1)) < n && !failed.load();) {
            for (uint64_t got = 0; got < plan[i].bytes;) {
                const ssize_t r = pread(plan[i].fd, buf.data() + at[i] + got, (size_t)(plan[i].bytes - got),
                                        (off_t)(plan[i].off + TensorBytes{got}).raw());
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    int none = 0;
                    failed.compare_exchange_strong(none, r == 0 ? ENODATA : errno);
                    return;
                }
                got += (uint64_t)r;
            }
        }
    };
    std::vector<std::thread> readers;
    for (uint32_t t = 1; t < PULSAR_CONTENT_SAMPLE_READERS && t < n; t++) readers.emplace_back(reader);
    reader();
    for (std::thread &t : readers) t.join();
    if (failed.load()) {
        errno = failed.load();
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        pulsar_kvdigest_update(d, &plan[i].bytes, sizeof plan[i].bytes);
        pulsar_kvdigest_update(d, buf.data() + at[i], plan[i].bytes);
    }
    return true;
}

int pulsar_engram_table_digest(const pulsar_engram_table *t, uint64_t *out) {
    std::vector<pulsar_content_read> plan;
    for (uint32_t i = 0; i < t->n_parts; i++) {
        const uint64_t rows = i + 1 < t->n_parts ? t->rows_per_part : t->n_rows - (uint64_t)i * t->rows_per_part;
        pulsar_content_sample(t->parts[i].fd, t->parts[i].base, elt_bytes(rows, t->row_bytes), &plan);
    }
    pulsar_kvdigest d;
    pulsar_kvdigest_init(&d);
    pulsar_kvdigest_update(&d, &t->n_rows, sizeof t->n_rows);
    pulsar_kvdigest_update(&d, &t->row_bytes, sizeof t->row_bytes);
    if (!pulsar_content_sample_fold(plan, &d)) {
        fprintf(stderr, "pulsar: row table %s cannot be read for the KV store identity: %s\n", t->path,
                strerror(errno));
        return 0;
    }
    *out = pulsar_kvdigest_final(&d);
    return 1;
}

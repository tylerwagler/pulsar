/* L353 no. 2: the store's index is the worker's, but a chat parse on a client thread walks its trailers
 * (kv_cache_restore_tool_memory_for_messages -> pulsar_segstore_foreach_trailer).  Under ThreadSanitizer (make
 * segstore-tsan) readers walk while the "worker" adds and evicts, so an unguarded walk reports a race. */
#include "lib/pulsar_segstore.h"

#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <unistd.h>
#include <vector>

static int write_payload(FILE *fp, void *, char *, size_t) {
    static const char zeros[256] = {0};
    return fwrite(zeros, 1, sizeof zeros, fp) == sizeof zeros ? 0 : 1;
}
static int write_trailer(FILE *fp, void *, char *, size_t) { return fwrite("TRAIL", 1, 5, fp) == 5 ? 0 : 1; }

int main() {
    char tmpl[] = "/tmp/pulsar-segstore-tsan.XXXXXX";
    char *dir = mkdtemp(tmpl);
    if (!dir) { perror("mkdtemp"); return 2; }
    pulsar_segstore *st = pulsar_segstore_open(dir, 0, 1, NULL, NULL);
    if (!st) { fprintf(stderr, "open failed\n"); return 2; }

    std::atomic<bool> stop{false};
    std::atomic<long> walks{0}, seen{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; t++)
        readers.emplace_back([&] {
            while (!stop.load()) {
                pulsar_segstore_foreach_trailer(st, [](FILE *fp, uint64_t bytes, const char *text, size_t len, void *ud) {
                    char b[5];
                    if (bytes != 5 || fread(b, 1, 5, fp) != 5 || memcmp(b, "TRAIL", 5) != 0 || len == 0 || text[0] != 'r') {
                        fprintf(stderr, "a walk read a wrong entry\n");
                        abort();
                    }
                    ((std::atomic<long> *)ud)->fetch_add(1);
                    return true;
                }, &seen);
                walks.fetch_add(1);
            }
        });

    /* the worker: a root and leaves appear and go, as commit and eviction do */
    char err[256], root[41], key[41];
    if (!pulsar_segstore_put(st, "", 0, 128, "root", 4, 256, write_payload, 5, write_trailer, NULL, root, err, sizeof err)) {
        fprintf(stderr, "put: %s\n", err);
        return 2;
    }
    int failures = 0;
    for (int i = 0; i < 4000; i++) {
        char text[32];
        const int n = snprintf(text, sizeof text, "r%d", i);
        if (!pulsar_segstore_put(st, root, 128, 256, text, (size_t)n, 256, write_payload, 5, write_trailer, NULL, key, err,
                                 sizeof err)) {
            failures++;
            continue;
        }
        if (i % 2) pulsar_segstore_drop(st, key);
        if (i % 64 == 63) {   /* keep the index small enough that every walk meets the churn */
            char *keys = NULL;
            const int k = pulsar_segstore_keys(st, &keys);
            for (int j = 0; j < k; j++) {
                char one[41];
                memcpy(one, keys + j * 40, 40);
                one[40] = '\0';
                if (strcmp(one, root) != 0) pulsar_segstore_drop(st, one);
            }
            free(keys);
        }
    }
    stop.store(true);
    for (auto &r : readers) r.join();
    pulsar_segstore_close(st);

    char cmd[128];
    snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", dir);
    if (system(cmd) != 0) fprintf(stderr, "cleanup failed: %s\n", dir);
    printf("segstore-tsan: %ld walks, %ld trailers read, %d failed puts\n", walks.load(), seen.load(), failures);
    return failures ? 1 : 0;
}

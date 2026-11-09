/** @file
 *  The disk KV cache's host memory pool: first fit over coalescing free extents (pulsar_kvpool.h). */
#include "pulsar_kvpool.h"

#include <iterator>
#include <map>
#include <unordered_map>

/** The pool: the region and its extents, by offset. */
struct pulsar_kvpool {
    uint8_t *base = nullptr;                          ///< the region (the caller's)
    uint64_t size = 0;                                ///< its usable bytes
    uint64_t used = 0;                                ///< bytes in allocated blocks
    std::map<uint64_t, uint64_t> free_ext;            ///< offset -> length of every free extent, disjoint, coalesced
    std::unordered_map<uint64_t, uint64_t> blocks;    ///< offset -> length of every allocated block
};

pulsar_kvpool *pulsar_kvpool_create(void *base, uint64_t bytes) {
    const uint64_t size = bytes / PULSAR_KVPOOL_ALIGN * PULSAR_KVPOOL_ALIGN;
    if (!base || (uintptr_t)base % PULSAR_KVPOOL_ALIGN != 0u || size == 0u) return nullptr;
    auto *p = new pulsar_kvpool();
    p->base = (uint8_t *)base;
    p->size = size;
    p->free_ext[0] = size;
    return p;
}

void pulsar_kvpool_destroy(pulsar_kvpool *p) { delete p; }

void *pulsar_kvpool_alloc(pulsar_kvpool *p, uint64_t bytes) {
    if (!p || bytes == 0u) return nullptr;
    const uint64_t n = pulsar_kvpool_round(bytes);
    for (auto it = p->free_ext.begin(); it != p->free_ext.end(); ++it) {
        if (it->second < n) continue;
        const uint64_t off = it->first, len = it->second;
        p->free_ext.erase(it);
        if (len > n) p->free_ext[off + n] = len - n;
        p->blocks[off] = n;
        p->used += n;
        return p->base + off;
    }
    return nullptr;
}

void pulsar_kvpool_free(pulsar_kvpool *p, void *blk) {
    if (!p || !blk) return;
    const uint64_t off = (uint64_t)((uint8_t *)blk - p->base);
    auto b = p->blocks.find(off);
    if (b == p->blocks.end()) return;
    uint64_t start = off, len = b->second;
    p->used -= len;
    p->blocks.erase(b);
    auto next = p->free_ext.lower_bound(start);
    if (next != p->free_ext.end() && next->first == start + len) {   /* the extent after it */
        len += next->second;
        next = p->free_ext.erase(next);
    }
    if (next != p->free_ext.begin()) {   /* the extent before it */
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) {
            start = prev->first;
            len += prev->second;
            p->free_ext.erase(prev);
        }
    }
    p->free_ext[start] = len;
}

uint64_t pulsar_kvpool_size(const pulsar_kvpool *p) { return p ? p->size : 0u; }
uint64_t pulsar_kvpool_used(const pulsar_kvpool *p) { return p ? p->used : 0u; }

uint64_t pulsar_kvpool_largest_free(const pulsar_kvpool *p) {
    uint64_t m = 0;
    if (p)
        for (const auto &e : p->free_ext) m = e.second > m ? e.second : m;
    return m;
}

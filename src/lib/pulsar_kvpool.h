/** @file
 *  The disk KV cache's host memory pool (L347 I3a, plans/124 D4 / D9): one region, sized once at startup, carved into
 *  4096-aligned blocks -- the copy-out staging buffers and, on a discrete-GPU host, the RAM tier. */
#ifndef PULSAR_KVPOOL_H
#define PULSAR_KVPOOL_H

/* The region is the caller's (the server pins it with pulsar_gpu_host_alloc, so a block is a DMA source and target
 * for the GPU copies; a test may hand any page-aligned memory).  Blocks are first-fit over the free extents, which
 * coalesce on free; every block starts and ends on PULSAR_KVPOOL_ALIGN, so it is also an O_DIRECT buffer.  Not
 * thread-safe: the segment store's thread (the server's worker) is its only user -- the writer thread reads and
 * writes the bytes of the blocks it was handed, never the pool. */

#include <stdint.h>

/** Block alignment and granule, bytes: a page, and the O_DIRECT block size the store writes and reads in. */
#define PULSAR_KVPOOL_ALIGN 4096u

/** An open pool (opaque). */
typedef struct pulsar_kvpool pulsar_kvpool;

/** Carve a pool over [base, base + bytes) (bytes rounded down to the granule).
 *  @param base  the region, PULSAR_KVPOOL_ALIGN-aligned; owned by the caller, who frees it after the pool
 *  @param bytes its size
 *  @return the pool, NULL when base is misaligned or the region holds no granule */
pulsar_kvpool *pulsar_kvpool_create(void *base, uint64_t bytes);
/** Destroy the pool (not the region).
 *  @param p the pool, or NULL */
void pulsar_kvpool_destroy(pulsar_kvpool *p);
/** A block of at least `bytes` (rounded up to the granule), first fit.
 *  @param p     the pool
 *  @param bytes the size wanted (> 0)
 *  @return the block, NULL when no free extent is that large */
void *pulsar_kvpool_alloc(pulsar_kvpool *p, uint64_t bytes);
/** Return a block from pulsar_kvpool_alloc (it coalesces with free neighbours).
 *  @param p   the pool
 *  @param blk the block, or NULL (no-op) */
void pulsar_kvpool_free(pulsar_kvpool *p, void *blk);
/** @param p the pool (NULL: 0)
 *  @return the region's usable size, bytes */
uint64_t pulsar_kvpool_size(const pulsar_kvpool *p);
/** @param p the pool (NULL: 0)
 *  @return bytes in allocated blocks */
uint64_t pulsar_kvpool_used(const pulsar_kvpool *p);
/** @param p the pool (NULL: 0)
 *  @return the largest block pulsar_kvpool_alloc could return now, bytes */
uint64_t pulsar_kvpool_largest_free(const pulsar_kvpool *p);
/** @param bytes a size
 *  @return it rounded up to the granule */
static inline uint64_t pulsar_kvpool_round(uint64_t bytes) {
    return (bytes + PULSAR_KVPOOL_ALIGN - 1u) / PULSAR_KVPOOL_ALIGN * PULSAR_KVPOOL_ALIGN;
}

#endif

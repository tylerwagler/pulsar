/** @file
 *  The disk KV payloads' 64-bit digest (L219; v2, four lanes, L347 I3a): one authority for the engine's payload
 *  streams, the segment store's writer and reader, and the TP workers' copies. */
#ifndef PULSAR_KVDIGEST_H
#define PULSAR_KVDIGEST_H

/* L219: a 64-bit digest over every payload byte, stored in the payload's last 8 bytes, so media bit-rot, a torn file
 * or an offline edit inside a multi-GB payload is a cache miss (re-prefill), not wrong attention.  Not cryptographic:
 * integrity against corruption, not tampering.
 *
 * v2 (L347 I3a): FOUR LANES.  The payload is read as little-endian 64-bit words; word i folds into lane i mod 4 (one
 * dependent multiply per word per lane, so the four chains run side by side: v1's single chain ran ~4 GB/s on the
 * E5-2699 v4, 83 ms per 318 MiB segment, on the worker).  The final value folds the four lanes in order and the byte
 * count, so swapping two lanes' words, or a trailing zero byte, changes it.  A trailing partial word is zero-padded
 * into the lane its word index names.
 *
 * The update is CHUNK-INDEPENDENT: a partial word is carried in `tail` and the word index in `words`, so a writer that
 * digests one buffer and a reader that digests it in ten pieces compute the same value. */

#include <stdbool.h>
#include <stdint.h>

/** Lanes of the v2 digest (word i folds into lane i mod PULSAR_KVDIGEST_LANES). */
#define PULSAR_KVDIGEST_LANES 4u
/** Bytes of a stored digest: a payload's last 8 bytes. */
#define PULSAR_KVDIGEST_BYTES 8u

/** A running digest (see above): init, update any number of times, final. */
typedef struct {
    uint64_t h[PULSAR_KVDIGEST_LANES];   ///< the lanes
    uint64_t words;                      ///< whole words folded so far (the next word's lane is words mod 4)
    uint64_t tail;                       ///< pending bytes of the next word, first-arrived at the low end
    uint32_t ntail;                      ///< 0..7 pending bytes
} pulsar_kvdigest;

/** Start a digest.
 *  @param d the digest to reset */
void pulsar_kvdigest_init(pulsar_kvdigest *d);
/** Fold bytes into the digest (chunk-independent).
 *  @param d     the digest
 *  @param ptr   the bytes
 *  @param bytes how many */
void pulsar_kvdigest_update(pulsar_kvdigest *d, const void *ptr, uint64_t bytes);
/** The digest's value: any pending tail folded, the lanes combined (the running state is not changed).
 *  @param d the digest
 *  @return the value */
uint64_t pulsar_kvdigest_final(const pulsar_kvdigest *d);
/** One-shot digest of a buffer.
 *  @param ptr   the bytes
 *  @param bytes how many
 *  @return pulsar_kvdigest_final over them */
uint64_t pulsar_kvdigest_of(const void *ptr, uint64_t bytes);
/** Seal a payload whose last PULSAR_KVDIGEST_BYTES are its digest slot: the digest of [0, bytes - 8) is stored at
 *  [bytes - 8, bytes), little-endian.  The segment store's writer calls it before a file write.
 *  @param payload the payload
 *  @param bytes   its size (>= 8) */
void pulsar_kvdigest_seal(uint8_t *payload, uint64_t bytes);
/** Does a payload's digest slot hold the digest of the rest?
 *  @param payload the payload
 *  @param bytes   its size
 *  @return true when bytes >= 8 and the stored digest matches */
bool pulsar_kvdigest_check(const uint8_t *payload, uint64_t bytes);

#endif

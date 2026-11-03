/** @file
 *  The model gates' disk segments through stdio (L347 I3a): a segment or restore point copied out of the engine
 *  (pulsar_session_copy_out_*), sealed and written to a FILE, and read back whole, its digest checked, into the
 *  engine's load from memory -- the bytes the server's segment store writes and reads, without the store. */
#pragma once

#include "pulsar.h"
#include "lib/pulsar_kvdigest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Write `bytes` the copy-out `fill` produces, sealed, at fp's position.
 *  @tparam F    a callable `int(uint8_t *dst)`: the copy-out, 0 on success
 *  @param fp    the file
 *  @param bytes the payload's size (0: the engine has none to give)
 *  @param fill  the copy-out
 *  @param err   receives a reason on failure (when the copy-out did not say one)
 *  @param errlen bytes at err
 *  @return 0, or 1 */
template <typename F>
static inline int gate_payload_save(FILE *fp, uint64_t bytes, F fill, char *err, size_t errlen) {
    uint8_t *buf = fp && bytes >= PULSAR_KVDIGEST_BYTES ? (uint8_t *)malloc((size_t)bytes) : NULL;
    if (!buf) {
        snprintf(err, errlen, "gate save: no stream, payload or memory (%llu bytes)", (unsigned long long)bytes);
        return 1;
    }
    int rc = fill(buf);
    if (rc == 0) {
        pulsar_kvdigest_seal(buf, bytes);
        if (fwrite(buf, 1, (size_t)bytes, fp) != bytes || fflush(fp) != 0) {
            snprintf(err, errlen, "gate save: short write");
            rc = 1;
        }
    }
    free(buf);
    return rc;
}

/** The segment [G_prev, G) of the installed bank to `fp` (what the removed pulsar_session_save_segment wrote).
 *  @param s the session  @param fp the file  @param G_prev the span's start  @param G its end
 *  @param err receives a reason  @param errlen bytes at err  @return 0, or 1 */
static inline int gate_save_segment(pulsar_session *s, FILE *fp, TokPos G_prev, TokPos G, char *err, size_t errlen) {
    const uint64_t bytes = pulsar_session_segment_bytes(s, G_prev, G, NULL);
    return gate_payload_save(fp, bytes, [&](uint8_t *dst) {
        return pulsar_session_copy_out_segment(s, G_prev, G, NULL, dst, bytes, err, errlen);
    }, err, errlen);
}

/** The restore point at G of the installed bank to `fp`.
 *  @param s the session  @param fp the file  @param G the grid position
 *  @param err receives a reason  @param errlen bytes at err  @return 0, or 1 */
static inline int gate_save_restore_point(pulsar_session *s, FILE *fp, TokPos G, char *err, size_t errlen) {
    const uint64_t bytes = pulsar_session_restore_point_bytes(s, G);
    return gate_payload_save(fp, bytes, [&](uint8_t *dst) {
        return pulsar_session_copy_out_restore_point(s, G, NULL, dst, bytes, err, errlen);
    }, err, errlen);
}

/** `bytes` from fp's position into a malloc'd buffer, its digest checked.
 *  @param fp the file (NULL: no part)  @param bytes the part's size  @param err receives a reason
 *  @param errlen bytes at err  @return the bytes (caller frees), NULL on a short read or a digest mismatch */
static inline uint8_t *gate_payload_read(FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = fp && bytes ? (uint8_t *)malloc((size_t)bytes) : NULL;
    if (!buf || fread(buf, 1, (size_t)bytes, fp) != bytes || !pulsar_kvdigest_check(buf, bytes)) {
        snprintf(err, errlen, "gate load: %s", !buf ? "no stream or memory" : "short read or digest mismatch");
        free(buf);
        return NULL;
    }
    return buf;
}

/** One segment (and, with `last`, its restore point) from files into the installed bank (what the removed
 *  pulsar_session_load_segment over streams did).  As a chain load does (pulsar_kvchain_load_slice), a part whose
 *  digest fails is refused before the engine reads it, and a failure leaves the bank invalidated.
 *  @param s the session  @param fp the segment's file  @param bytes its size  @param last the chain's last
 *  @param rp_fp the restore point's file when `last`  @param rp_bytes its size  @param G_out receives G
 *  @param err receives a reason  @param errlen bytes at err  @return 0, or nonzero */
static inline int gate_load_segment(pulsar_session *s, FILE *fp, uint64_t bytes, bool last, FILE *rp_fp,
                                    uint64_t rp_bytes, TokPos *G_out, char *err, size_t errlen) {
    if (G_out) *G_out = TokPos{0};
    uint8_t *seg = gate_payload_read(fp, bytes, err, errlen);
    uint8_t *rp = seg && last ? gate_payload_read(rp_fp, rp_bytes, err, errlen) : NULL;
    const int rc = seg && (!last || rp)
                       ? pulsar_session_load_segment(s, seg, bytes, last, rp, rp_bytes, G_out, NULL, err, errlen)
                       : 1;
    free(seg);
    free(rp);
    if (rc != 0) pulsar_session_invalidate(s);
    return rc;
}

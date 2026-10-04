#ifndef PULSAR_KVCHAIN_H
#define PULSAR_KVCHAIN_H

/* L264: a session's KV as a SEGMENT CHAIN in a pulsar_segstore -- the one
 * persist/restore both pulsar-server's disk KV cache and pulsar-agent's saved
 * sessions run.
 *
 * persist: extend the chain the store already holds for the installed bank's
 *   history by one segment per grid checkpoint past it, so a write costs only
 *   the new tokens.  The stored chain is this history only if the bank's own
 *   tokens up to its end render to exactly its text (same bytes at the same
 *   position, the L115 rule); otherwise a fresh chain starts at the root.
 * restore: load the deepest chain whose text is a byte prefix of `text`; the
 *   bank stands live at its end.  A segment that will not load is dropped with
 *   everything below it, and so is a chain whose text and tokens disagree at the
 *   boundary (L196), so the next persist rewrites it. */

#include "pulsar.h"
#include "pulsar_segstore.h"

#include <stddef.h>
#include <stdint.h>

/** The longest chain persisted or restored: one segment per retained grid
 *  checkpoint, so far above any bank's count. */
#define PULSAR_KVCHAIN_MAX 512

/** An optional per-segment trailer (the server's tool map).  `size` reports the
 *  trailer for a segment's text (0 bytes = none); `write` writes exactly it. */
typedef struct {
    void *ud;
    bool (*size)(void *ud, const char *text, uint64_t *bytes);
    int (*write)(FILE *fp, void *ud, const char *text, char *err, size_t errlen);
} pulsar_kvchain_trailer;

/** What a persist did. */
typedef struct {
    int written;        ///< new segments
    int end;            ///< the chain's end after the write (0: none)
    uint64_t bytes;     ///< payload + trailer bytes written
    char key[41];       ///< the chain's last key ("" when none)
    char err[256];      ///< why the write stopped early ("" when it did not)
} pulsar_kvchain_persist_result;

/** Persist the installed bank's history (see above).  Nothing is written when
 *  its deepest grid checkpoint is below `min_tokens`.  Returns `written`. */
int pulsar_kvchain_persist(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, int min_tokens,
                           const pulsar_kvchain_trailer *trailer, pulsar_kvchain_persist_result *out);

/** Restore the deepest stored chain for `text` into the installed bank (see
 *  above), at most `cap` segments reported in `chain` (*n_out of them).
 *  Returns the position it stands at, 0 when nothing was loaded (`err` says why
 *  when a stored chain was refused).  A chain ending below `min_tokens` is not
 *  loaded.  The chain is touched on success. */
int pulsar_kvchain_restore(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                           size_t text_len, int min_tokens, pulsar_segstore_seg *chain, int cap, int *n_out,
                           char *err, size_t errlen);

#endif

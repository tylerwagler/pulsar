#ifndef PULSAR_SEGSTORE_H
#define PULSAR_SEGSTORE_H

/* L264 S4: the server's disk KV cache as a content-addressed SEGMENT store.
 *
 * A segment holds one stretch of a conversation's KV: the engine's segment
 * payload (pulsar_session_save_segment: rows [G_prev, G) and the grid checkpoint
 * at G) plus the rendered text of those tokens.  Its key is
 *     sha1(parent_key || text)          (parent_key "" for a root, G_prev 0)
 * so a key names the WHOLE text from position 0 to its G, and two conversations
 * that share a prefix share its segments.  A lookup walks the chains from the
 * roots, hashing the request's own bytes against each child's key, and returns
 * the deepest chain whose text is a byte prefix of the request -- no text is
 * held in memory, only keys, parents and lengths.
 *
 * Storage: <dir>/<key>.seg = fixed header, text, payload, optional trailer
 * (the server's tool map).  Writes are durable (tmp + fsync + rename + dir
 * fsync) and leave no page cache (L261).  Eviction keeps the budget by removing
 * a LEAF: a segment with children is part of a longer chain and goes only after
 * them.  L323: a DOMINATED leaf goes first -- one whose text is a byte prefix of
 * a sibling's (the same conversation's older tail, superseded by a later tail or
 * by the grid segment that covers it) -- then the least recently used.
 *
 * L323: a payload may end in an optional STATE part (the engine's checkpoint at
 * G, pulsar_session_segment_bytes); the index records its size, so a lookup can
 * ask for a chain that ends where a state is stored and a load can price "every
 * segment's rows + the last one's state" without opening a file. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define PULSAR_SEGSTORE_MAGIC   0x31475350u   /* "PSG1" */
#define PULSAR_SEGSTORE_VERSION 2u            /* v2 (L323): the header carries the payload's state bytes */
#define PULSAR_SEGSTORE_FLAG_TRAILER 1u       /* a trailer follows the payload */

typedef struct pulsar_segstore pulsar_segstore;

/** One segment as the index knows it. */
typedef struct {
    char key[41];
    char parent[41];          ///< "" for a root
    uint32_t G_prev, G;       ///< token span
    uint64_t text_end;        ///< bytes of chain text from position 0 to G (index-derived)
    uint64_t text_len;        ///< this segment's own text bytes
    uint64_t payload_bytes;
    uint64_t state_bytes;     ///< L323: the payload's trailing state part (0 = rows only, no state at G)
    uint64_t trailer_bytes;
    uint64_t file_bytes;
    uint32_t hits;
    uint64_t last_used;       ///< unix seconds
    bool dominated;           ///< L323: a leaf whose text is a byte prefix of a sibling's (index-derived)
} pulsar_segstore_seg;

/** L323: which chain ends a lookup accepts.  `grid`: the last segment's G is a multiple of it (0 = any end);
 *  `state`: the last segment carries a state. */
typedef struct {
    uint32_t grid;
    bool state;
} pulsar_segstore_end;

typedef void (*pulsar_segstore_log_fn)(void *ud, const char *msg);
/** Writes exactly the bytes promised into fp; 0 on success. */
typedef int (*pulsar_segstore_write_fn)(FILE *fp, void *ud, char *err, size_t errlen);

/** What a store's segments were computed by: the model AND its routed-expert
 * format (pulsar_engine_routed_quant_bits -- the IQ2, MXFP4 and EXL3 builds of
 * one model share a model id and a KV row layout, so a segment's shape cannot
 * tell them apart, but their KV differs).  The one rule both writers and
 * readers key with. */
static inline uint32_t pulsar_segstore_identity(uint32_t model_id, uint32_t routed_quant_bits) {
    return model_id << 8 | (routed_quant_bits & 0xffu);
}

/** Open (create) the store and index its segments.  budget_bytes 0 = unbounded.
 * Segments of another identity (pulsar_segstore_identity), of another format,
 * or unreadable are skipped and left alone -- except a segment of an OLDER store
 * version, which no build of this format reads: it is removed (L323). */
pulsar_segstore *pulsar_segstore_open(const char *dir, uint64_t budget_bytes, uint32_t identity,
                                      pulsar_segstore_log_fn log, void *log_ud);
void pulsar_segstore_close(pulsar_segstore *st);

/** The key rule, the one authority for it. */
void pulsar_segstore_child_key(const char *parent, const char *text, size_t len, char out[41]);

/** The deepest chain (by G, then by text) whose text is a byte prefix of text[0, len) and whose last segment `want`
 * accepts (NULL: any), root first, at most `cap` long.  Returns its length (0: none). */
int pulsar_segstore_lookup(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                           pulsar_segstore_seg *chain, int cap);
bool pulsar_segstore_contains(pulsar_segstore *st, const char key[41]);
/** The index entry of `key` into *out; false when it is not stored. */
bool pulsar_segstore_get(pulsar_segstore *st, const char key[41], pulsar_segstore_seg *out);

/** Write one segment.  `parent` must already be stored (or be "" for a root);
 * makes room first by evicting leaves -- never the parent.  The key is derived
 * from parent and text here, so a caller cannot store a segment under the
 * wrong name.  The written key is returned in key_out.  The same key twice is one
 * segment: a put of a stored key writes nothing -- unless the stored copy has no
 * state and this one does (`state_bytes`, the payload's trailing part), which
 * replaces it in place (same key, same children). */
bool pulsar_segstore_put(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G,
                         const char *text, size_t text_len,
                         uint64_t payload_bytes, uint64_t state_bytes, pulsar_segstore_write_fn write_payload,
                         uint64_t trailer_bytes, pulsar_segstore_write_fn write_trailer,
                         void *ud, char key_out[41], char *err, size_t errlen);

/** Open a segment's payload for reading, positioned at its first byte. */
FILE *pulsar_segstore_open_payload(pulsar_segstore *st, const char key[41], uint64_t *payload_bytes);
/** Open a segment's trailer for reading, positioned at its first byte; NULL when
 * the segment carries none. */
FILE *pulsar_segstore_open_trailer(pulsar_segstore *st, const char key[41], uint64_t *trailer_bytes);
/** Record a use of every segment of a chain (hit count + last_used). */
void pulsar_segstore_touch(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n);
/** Called with every key the store removes (drop, release, eviction, an orphan
 *  pruned at open) -- the TP leader tells its workers to delete their copies.
 *  Set it before anything can be removed; NULL to clear. */
typedef void (*pulsar_segstore_removed_fn)(void *ud, const char key[41]);
void pulsar_segstore_set_removed_hook(pulsar_segstore *st, pulsar_segstore_removed_fn fn, void *ud);
/** Remove a segment and every segment below it (a payload that would not load). */
void pulsar_segstore_drop(pulsar_segstore *st, const char key[41]);
/** Release the stretch of a chain only `tip` uses: walk up from a LEAF tip while
 * each parent has no other child, stopping below `stop` (a chain to keep, e.g. a
 * shared system prompt; NULL for none), and drop from there down.  A tip with
 * children is shared by a longer chain and nothing goes.  Returns the bytes
 * released. */
uint64_t pulsar_segstore_release(pulsar_segstore *st, const char tip[41], const char *stop);

/** Visit every segment that carries a trailer: fp is positioned at the trailer,
 * and `text` is the segment's own text.  The visitor returns false to stop. */
typedef bool (*pulsar_segstore_trailer_fn)(FILE *fp, uint64_t trailer_bytes, const char *text, size_t text_len,
                                           void *ud);
void pulsar_segstore_foreach_trailer(pulsar_segstore *st, pulsar_segstore_trailer_fn fn, void *ud);

/** Every key, 40 hex chars each, concatenated (caller frees); the count returned. */
int pulsar_segstore_keys(pulsar_segstore *st, char **keys);
/** Bytes the store holds, and the count of segments. */
uint64_t pulsar_segstore_used_bytes(const pulsar_segstore *st);
int pulsar_segstore_count(const pulsar_segstore *st);

#endif

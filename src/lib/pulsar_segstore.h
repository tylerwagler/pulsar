/** @file
 *  The server's disk KV cache as a content-addressed segment store (L264 S4): keys, chains, budget, lookup. */
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

/// Magic of a .seg file's header: "PSG1".
#define PULSAR_SEGSTORE_MAGIC   0x31475350u   /* "PSG1" */
/// Store format version; v2 (L323) carries the payload's state bytes in the header.  Older versions are removed at open.
#define PULSAR_SEGSTORE_VERSION 2u            /* v2 (L323): the header carries the payload's state bytes */
/// Header flag: a trailer follows the payload.
#define PULSAR_SEGSTORE_FLAG_TRAILER 1u       /* a trailer follows the payload */

/// An open segment store (opaque): its directory, budget, identity and in-memory index.
typedef struct pulsar_segstore pulsar_segstore;

/** One segment as the index knows it. */
typedef struct {
    /// This segment's key: 40 hex chars of sha1(parent || text) + NUL
    char key[41];
    char parent[41];          ///< "" for a root
    /** Token span [G_prev, G): G_prev is the parent's end (0 for a root), G this segment's end. */
    uint32_t G_prev, G;       ///< token span
    uint64_t text_end;        ///< bytes of chain text from position 0 to G (index-derived)
    uint64_t text_len;        ///< this segment's own text bytes
    /// Engine payload bytes (rows + optional state), pulsar_session_save_segment's output
    uint64_t payload_bytes;
    uint64_t state_bytes;     ///< L323: the payload's trailing state part (0 = rows only, no state at G)
    /// Optional trailer bytes after the payload (0 = none)
    uint64_t trailer_bytes;
    /// The whole .seg file: header + text + payload + trailer; what the budget counts
    uint64_t file_bytes;
    /// Lookups that used this segment (pulsar_segstore_touch)
    uint32_t hits;
    uint64_t last_used;       ///< unix seconds
    bool dominated;           ///< L323: a leaf whose text is a byte prefix of a sibling's (index-derived)
} pulsar_segstore_seg;

/** L323: which chain ends a lookup accepts.  `grid`: the last segment's G is a multiple of it (0 = any end);
 *  `state`: the last segment carries a state. */
typedef struct {
    /// In tokens: the chain's end G must be a multiple of it (0 = any end)
    uint32_t grid;
    /// The last segment must carry a state part (state_bytes > 0)
    bool state;
} pulsar_segstore_end;

/** Log sink for the store's messages.
 *  @param ud  the log_ud given to pulsar_segstore_open
 *  @param msg one NUL-terminated message line */
typedef void (*pulsar_segstore_log_fn)(void *ud, const char *msg);
/** Writes exactly the bytes promised into fp; 0 on success. */
/// @param fp     the segment file, positioned where the bytes go
/// @param ud     the caller context passed to pulsar_segstore_put
/// @param err    receives a reason on failure
/// @param errlen size of `err`
typedef int (*pulsar_segstore_write_fn)(FILE *fp, void *ud, char *err, size_t errlen);

/// @param model_id          the model id (low 16 bits used)
/// @param routed_quant_bits the routed-expert format's bits (low 8 bits used)
/// @param rope_id           the family's rope_id; 0 for the rope the model id implies
/// @return the 32-bit store identity: model_id << 8 | bits, with a nonzero rope fold in the top byte when rope_id != 0
/** What a store's segments were computed by: the model, its routed-expert
 * format (pulsar_engine_routed_quant_bits -- the IQ2, MXFP4 and EXL3 builds of
 * one model share a model id and a KV row layout, so a segment's shape cannot
 * tell them apart, but their KV differs) AND its rope where the model id does
 * not fix it (L280: the family's rope_id, 0 for the rope the model id implies;
 * Qwen under YaRN writes other K bytes into the same layout).  The one rule
 * both writers and readers key with; callers take it from
 * pulsar_engine_kv_identity.
 *
 * A rope_id of 0 leaves the identity as it was before L280 (model_id << 8 |
 * bits), so stores written under a model's own rope stay readable.  Any other
 * rope_id sets the top byte to a nonzero fold of it: such a store never shares
 * an identity with a rope_id-0 one.  Two scaled ropes whose folds collide (1 in
 * 255) share an index, and the engine's payload layout digest -- which mixes the
 * whole rope_id -- still refuses every segment of the other at load (and the
 * server drops it), so a fold collision costs a refused load, never another
 * rope's KV. */
static inline uint32_t pulsar_segstore_identity(uint32_t model_id, uint32_t routed_quant_bits, uint64_t rope_id) {
    uint32_t id = (model_id & 0xffffu) << 8 | (routed_quant_bits & 0xffu);
    if (rope_id) {
        uint8_t f = 0;
        for (int i = 0; i < 8; i++) f ^= (uint8_t)(rope_id >> (8 * i));
        id |= (uint32_t)(f ? f : 1u) << 24;
    }
    return id;
}

/// @param dir          the store directory (created when missing; must be writable)
/// @param budget_bytes byte budget for the .seg files; 0 = unbounded
/// @param identity     pulsar_segstore_identity of the writer; segments of another identity are skipped
/// @param log          message sink, or NULL
/// @param log_ud       context for `log`
/// @return the open store (close with pulsar_segstore_close), or NULL when the directory cannot be created, written or listed
/** Open (create) the store and index its segments.  budget_bytes 0 = unbounded.
 * Segments of another identity (pulsar_segstore_identity), of another format,
 * or unreadable are skipped and left alone -- except a segment of an OLDER store
 * version, which no build of this format reads: it is removed (L323). */
pulsar_segstore *pulsar_segstore_open(const char *dir, uint64_t budget_bytes, uint32_t identity,
                                      pulsar_segstore_log_fn log, void *log_ud);
/** Close the store and free its index; the segment files stay on disk.
 *  @param st the store, or NULL */
void pulsar_segstore_close(pulsar_segstore *st);

/// @param parent the parent's key ("" for a root)
/// @param text   the segment's own rendered text
/// @param len    bytes of `text`
/// @param out    receives the 40 hex chars of sha1(parent || text) and a NUL
/** The key rule, the one authority for it. */
void pulsar_segstore_child_key(const char *parent, const char *text, size_t len, char out[41]);

/// @param st    the store
/// @param text  the request's rendered text
/// @param len   bytes of `text`
/// @param want  which chain ends are acceptable, or NULL for any
/// @param chain receives the chain's entries, root first
/// @param cap   capacity of `chain`
/// @return the chain's length, 0 when no chain matches
/** The deepest chain (by G, then by text) whose text is a byte prefix of text[0, len) and whose last segment `want`
 * accepts (NULL: any), root first, at most `cap` long.  Returns its length (0: none). */
int pulsar_segstore_lookup(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                           pulsar_segstore_seg *chain, int cap);
/** Is `key` stored?
 *  @param st  the store, or NULL (false)
 *  @param key 40 hex chars
 *  @return true when the index holds the key */
bool pulsar_segstore_contains(pulsar_segstore *st, const char key[41]);
/// @param st  the store
/// @param key 40 hex chars
/// @param out receives the index entry
/// @return true when the key is stored
/** The index entry of `key` into *out; false when it is not stored. */
bool pulsar_segstore_get(pulsar_segstore *st, const char key[41], pulsar_segstore_seg *out);

/// @param st            the store
/// @param parent        the parent's key, "" for a root
/// @param G_prev        the segment's first token position (the parent's G)
/// @param G             the segment's end position
/// @param text          the segment's own rendered text
/// @param text_len      bytes of `text`
/// @param payload_bytes bytes write_payload will write
/// @param state_bytes   of those, the payload's trailing state part (0 = rows only)
/// @param write_payload writes the payload
/// @param trailer_bytes bytes write_trailer will write (0 = none)
/// @param write_trailer writes the trailer, or NULL with trailer_bytes 0
/// @param ud            context for both writers
/// @param key_out       receives the stored key
/// @param err           receives a reason on failure
/// @param errlen        size of `err`
/// @return true when the segment is stored (or already was)
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

/// @param st            the store
/// @param key           40 hex chars
/// @param payload_bytes receives the payload's size, or NULL
/// @return the open file (caller fcloses), NULL when the key is not stored or the file cannot be read
/** Open a segment's payload for reading, positioned at its first byte. */
FILE *pulsar_segstore_open_payload(pulsar_segstore *st, const char key[41], uint64_t *payload_bytes);
/// @param st            the store
/// @param key           40 hex chars
/// @param trailer_bytes receives the trailer's size, or NULL
/// @return the open file (caller fcloses), NULL when the key is not stored or carries no trailer
/** Open a segment's trailer for reading, positioned at its first byte; NULL when
 * the segment carries none. */
FILE *pulsar_segstore_open_trailer(pulsar_segstore *st, const char key[41], uint64_t *trailer_bytes);
/// @param st          the store
/// @param key         40 hex chars
/// @param payload_len payload bytes to probe, from the payload's start
/// @param trailer     also probe the trailer
/// @return bytes of that range resident in the page cache
/** L327: how many bytes of a segment's first `payload_len` payload bytes (and, with `trailer`, of its trailer) are in
 *  the page cache now (mincore; 0 for a key not stored or a file that will not map) -- a read of them costs memory
 *  bandwidth, not the disk's. */
uint64_t pulsar_segstore_cached_bytes(pulsar_segstore *st, const char key[41], uint64_t payload_len, bool trailer);
/// @param st    the store
/// @param chain the chain's entries, as pulsar_segstore_lookup returned them
/// @param n     entries in `chain`
/** Record a use of every segment of a chain (hit count + last_used). */
void pulsar_segstore_touch(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n);
/** Called with every key the store removes (drop, release, eviction, an orphan
 *  pruned at open) -- the TP leader tells its workers to delete their copies.
 *  Set it before anything can be removed; NULL to clear. */
/** @param ud  the context given to pulsar_segstore_set_removed_hook
 *  @param key the removed key, 40 hex chars */
typedef void (*pulsar_segstore_removed_fn)(void *ud, const char key[41]);
/** Set (or clear, fn NULL) the hook called with every key the store removes.
 *  @param st the store, or NULL (no-op)
 *  @param fn the hook
 *  @param ud context for `fn` */
void pulsar_segstore_set_removed_hook(pulsar_segstore *st, pulsar_segstore_removed_fn fn, void *ud);
/// @param st  the store
/// @param key 40 hex chars of the segment to remove with its subtree
/** Remove a segment and every segment below it (a payload that would not load). */
void pulsar_segstore_drop(pulsar_segstore *st, const char key[41]);
/// @param st   the store
/// @param tip  the chain's leaf key
/// @param stop key of a segment to keep (the walk stops below it), or NULL
/// @return bytes of segment files removed
/** Release the stretch of a chain only `tip` uses: walk up from a LEAF tip while
 * each parent has no other child, stopping below `stop` (a chain to keep, e.g. a
 * shared system prompt; NULL for none), and drop from there down.  A tip with
 * children is shared by a longer chain and nothing goes.  Returns the bytes
 * released. */
uint64_t pulsar_segstore_release(pulsar_segstore *st, const char tip[41], const char *stop);

/** Visit every segment that carries a trailer: fp is positioned at the trailer,
 * and `text` is the segment's own text.  The visitor returns false to stop. */
/** @param fp            the segment file, positioned at its trailer (the store closes it)
 *  @param trailer_bytes the trailer's size
 *  @param text          the segment's own text
 *  @param text_len      bytes of `text`
 *  @param ud            context given to pulsar_segstore_foreach_trailer
 *  @return false to stop the walk */
typedef bool (*pulsar_segstore_trailer_fn)(FILE *fp, uint64_t trailer_bytes, const char *text, size_t text_len,
                                           void *ud);
/** Run `fn` over every segment that carries a trailer, in index order.
 *  @param st the store, or NULL (no-op)
 *  @param fn the visitor
 *  @param ud context for `fn` */
void pulsar_segstore_foreach_trailer(pulsar_segstore *st, pulsar_segstore_trailer_fn fn, void *ud);

/// @param st   the store
/// @param keys receives the malloc'd keys (NULL when there are none)
/// @return the number of keys
/** Every key, 40 hex chars each, concatenated (caller frees); the count returned. */
int pulsar_segstore_keys(pulsar_segstore *st, char **keys);
/// @param st the store, or NULL (0)
/// @return bytes of the .seg files the store holds
/** Bytes the store holds, and the count of segments. */
uint64_t pulsar_segstore_used_bytes(const pulsar_segstore *st);
/** The number of segments the store holds.
 *  @param st the store, or NULL (0)
 *  @return the segment count */
int pulsar_segstore_count(const pulsar_segstore *st);

#endif

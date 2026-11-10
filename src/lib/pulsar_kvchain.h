/** @file
 *  A session's KV persisted and restored as a segment chain in a pulsar_segstore (L264, the canonical grid of L323). */
#ifndef PULSAR_KVCHAIN_H
#define PULSAR_KVCHAIN_H

/* L264: a session's KV as a SEGMENT CHAIN in a pulsar_segstore -- the one
 * persist/restore both pulsar-server's disk KV cache and pulsar-agent's saved
 * sessions run.
 *
 * L323: the CANONICAL GRID.  A persisted chain is cut only at absolute multiples
 * of the disk grid (pulsar_kvchain_grid: the session's prefill chunk, where every
 * prefill captures a checkpoint) plus ONE leaf -- the tail from the last grid
 * point to the chain's end -- so every copy and every branch of a conversation
 * shares identical segments, by key, up to the grid point below its divergence,
 * wherever their turns ended.  A tail is never a parent: the next persist of a
 * longer history writes the grid segment that covers it and a new tail, and the
 * store evicts the superseded tail first (a dominated leaf, pulsar_segstore.h).
 * Before L323 the cuts were wherever the writer's checkpoints fell, so copies
 * deduped only when cut alike, and every segment carried the ~133 MiB state.
 * L347: a segment is rows only; the state at a chain's end is the end key's
 * RESTORE POINT (pulsar_segstore_put_rp).  L347 I2: restore points exist only where
 * a restore starts -- a chain's end when its bank is about to lose it, a shared
 * prefix's anchor, and every PULSAR_KVCHAIN_RP_EVERY_CELLS grid cells -- while rows
 * are written by every persist.
 *
 * persist: extend the deepest grid-ended chain the store already holds for the
 *   installed bank's history by the grid segments past it up to the bank's
 *   deepest checkpoint, so a write costs only the new tokens (a segment the store
 *   has is not written again).  What else it writes is the caller's request
 *   (PULSAR_KVCHAIN_TAIL / _RP / _ANCHOR): the tail to that checkpoint, the restore
 *   point at the chain's end, flagged an anchor;
 *   and, whatever it asked, a periodic restore point at each grid point it passes
 *   that pulsar_kvchain_rp_periodic names, when the bank holds its checkpoint there
 *   (pulsar_kvchain_place_rps).  Older restore points stay when the chain goes on
 *   (a branch from an older end resumes there); the store's eviction takes the
 *   unflagged ones first, the oldest first.  L347 I3a: every file -- segment or restore point -- is COPIED OUT on the calling
 *   thread into its image (pulsar_session_copy_out_*: an event-fenced copy, the
 *   bank moves on) and handed to the writer (pulsar_kvchain_writer), which seals,
 *   writes and commits it.  The
 *   stored chain is this history only if the bank's own tokens up to its end
 *   render to exactly its text (same bytes at the same position, the L115 rule);
 *   otherwise a fresh chain starts at the root.
 * restore: load the deepest chain whose text is a byte prefix of `text` and whose
 *   last segment has a restore point -- every segment's rows, the last one's
 *   restore point -- so the bank stands live at the nearest stored restore point
 *   at or below the request's divergence and the sync prefills the rest.  A chain whose text and tokens disagree at
 *   the boundary (L196) is dropped from its last segment, so the next persist rewrites it.  L347 I3b (plans/124 D7):
 *   a part that will not load drops only what its fault invalidates -- a CONTENT fault (a part missing, unreadable,
 *   of the wrong size or header, its digest wrong) in a segment's rows drops the key and everything below it, in a
 *   restore point that restore point alone -- and a TRANSIENT failure (no RAM room, the bank, the device, a TP rank's
 *   copy) drops nothing.  pulsar_kvchain_verify_slice checks every part's bytes before a load writes the bank, and
 *   pulsar_kvchain_pin holds the chain against eviction while it is read.
 *
 * L281: a chain carries IMAGE rows.  An image block renders as its family's placeholder text (so a request carrying
 * the image finds the chain by the text it was rendered with), and each segment's payload carries the records of the
 * blocks inside it (image_identity.cpp): persist goes past a block only when the session records its image, and
 * restore loads past a block only when the request brings the same image there -- by content, in order -- stopping,
 * said by name, at the last segment below the first block that differs.  Every rank's copy carries the records. */

#include "pulsar.h"
#include "pulsar_segstore.h"

#include <stddef.h>
#include <stdint.h>

/** The longest chain persisted or restored: one segment per disk-grid cell plus the tail (L323) -- a 262k context is
 *  65 at the production grid, so far below it. */
#define PULSAR_KVCHAIN_MAX 512

/** L347 I2: a persist writes a restore point at every multiple of this many disk-grid cells it passes (32768 tokens at
 *  the production grid of 4096), when the bank holds its checkpoint there, so a crash or a lost restore point costs at
 *  most this many cells of state to rebuild, never the whole history (Tyler 2026-11-03: "yes, 8 cells is fine"). */
#define PULSAR_KVCHAIN_RP_EVERY_CELLS 8

/** L347 I2: what a persist writes besides the rows up to the deepest grid point (pulsar_kvchain_persist's `what`). */
/// Also the tail, from the last grid point to the deepest grid checkpoint (closing the chain there)
#define PULSAR_KVCHAIN_TAIL 1u
/// And the restore point at the chain's end (the history is about to be lost: evict, spill, abandon, shutdown)
#define PULSAR_KVCHAIN_RP 2u
/// Which is a shared-prefix anchor (PULSAR_SEGSTORE_RP_ANCHOR); only with PULSAR_KVCHAIN_RP
#define PULSAR_KVCHAIN_ANCHOR 4u
/// L347 I3b: and a pinned one (PULSAR_SEGSTORE_RP_PINNED: X-Pulsar-KV pin-anchor); only with PULSAR_KVCHAIN_ANCHOR
#define PULSAR_KVCHAIN_PIN 8u

/** An optional per-segment trailer (the server's tool map).  `size` reports the
 *  trailer for a segment's text (0 bytes = none); `write` writes exactly it. */
typedef struct {
    /// Caller context passed back to `size` and `write`
    void *ud;
    /** Report the trailer's size for a segment whose rendered text is `text` into *bytes; false = no trailer. */
    bool (*size)(void *ud, const char *text, uint64_t *bytes);
    /** Write exactly the bytes `size` reported for `text` into fp; 0 on success, else a message in err. */
    int (*write)(FILE *fp, void *ud, const char *text, char *err, size_t errlen);
} pulsar_kvchain_trailer;

/** L338 / L347 I3a: where a persist's files go once copied out -- the server's background writer.  A file (a segment
 *  or a restore point) is reserved in the store with its image (pulsar_segstore_reserve / _reserve_rp), copied out into
 *  it on the calling thread (the bank moves on), then handed to the writer, which seals and writes it
 *  (pulsar_segstore_job_write) and commits it back on the store's thread (pulsar_segstore_commit). */
typedef struct {
    /// context for both hooks
    void *ud;
    /** The store's pool has no room for an image of a `file_bytes` file: wait for the writer to finish (and commit)
     *  what it holds until pulsar_segstore_ram_make_room makes it; false = nothing in flight can make it (the file is
     *  not written by this persist, which says so: pulsar_kvchain_persist_result::no_room). */
    bool (*wait_room)(void *ud, uint64_t file_bytes);
    /** Take the reserved, filled job: the writer owns it from here, writes it and commits it. */
    void (*submit)(void *ud, pulsar_segstore_job *job);
} pulsar_kvchain_writer;

/** L338: a writer that writes and commits each job on the calling thread before the persist goes on -- for the gates
 *  and tools that persist and load in one breath (the server writes in the background). */
/// @param st the store the persist writes to
/// @return the writer (its ud is `st`)
pulsar_kvchain_writer pulsar_kvchain_writer_inline(pulsar_segstore *st);

/** What a persist did. */
typedef struct {
    int written;        ///< new segments, copied out and handed to the writer (L338: committed later, on its thread)
    int end;            ///< the chain's end after the write (0: none)
    uint64_t bytes;     ///< payload + trailer bytes copied out
    uint64_t rp_bytes;  ///< L347: bytes of the restore points this persist copied out (chain end and periodic; 0: none)
    int rp_written;     ///< L347 I2: how many restore points it handed to the writer
    /// L347 I3a: a file could not be copied out: the store's RAM pool had no room and nothing in flight could make it
    /// (a spill that sees this does not free its bank)
    bool no_room;
    char key[41];       ///< the chain's last key ("" when none)
    char err[256];      ///< why the write stopped early ("" when it did not)
} pulsar_kvchain_persist_result;

/** L323: the disk grid of `s`'s chains, the ONE authority for where a persisted chain is cut: the session's prefill
 *  chunk (pulsar_session_prefill_cap, 4096 in production -- PULSAR_PREFILL_CHUNK_DEFAULT), because every prefill
 *  lands a chunk, and captures a grid checkpoint, on each absolute multiple of it (prefill_loop.cpp). */
/// @param s the session
/// @return the grid in tokens: the session's prefill chunk
int pulsar_kvchain_grid(pulsar_session *s);

/** L323: where the segments of a chain stored to `prev` (a grid point, or 0) end when the history's deepest grid
 *  checkpoint is `deepest`: every multiple of `grid` in (prev, deepest], then, with `tail`, the tail to `deepest`
 *  when it is not one.  Writes them ascending to `ends` (room for `cap`); returns how many. */
/// @param prev    the stored chain's end (a grid point, or 0)
/// @param deepest the history's deepest grid checkpoint
/// @param grid    the disk grid (pulsar_kvchain_grid)
/// @param tail    also end with the tail to `deepest` when it is not a grid point
/// @param ends    receives the segment ends, ascending
/// @param cap     capacity of `ends`
/// @return the number of ends written
int pulsar_kvchain_segment_ends(int prev, int deepest, int grid, bool tail, int *ends, int cap);

/** L347 I2: is `G` a periodic restore point -- a positive multiple of PULSAR_KVCHAIN_RP_EVERY_CELLS cells of `grid`?
 *  @param G    a segment end
 *  @param grid the disk grid (pulsar_kvchain_grid)
 *  @return true when a persist passing G writes a restore point there */
bool pulsar_kvchain_rp_periodic(int G, int grid);

/** L347 I2: one segment of a persisted chain: its key and its end. */
typedef struct {
    char key[41];   ///< the segment's key
    int G;          ///< its end
} pulsar_kvchain_link;

/** L347 I2: where a persist's restore points come from -- the installed bank's checkpoints in pulsar_kvchain_persist,
 *  a fixture in the unit tests. */
typedef struct {
    /// context for both hooks
    void *ud;
    /** The restore point's bytes at G; 0 = there is no checkpoint at G (nothing is written there). */
    uint64_t (*bytes)(void *ud, int G);
    /** L347 I3a: copy exactly those bytes for G, the restore point of segment `key`, into `dst` (its digest slot is
     *  the writer's); 0 on success, else err. */
    int (*copy)(void *ud, int G, const char *key, uint8_t *dst, uint64_t bytes, char *err, size_t errlen);
} pulsar_kvchain_rp_source;

/** L347 I2: the restore point policy of a persist that stored (or found stored) the segments `links` (ascending; the
 *  last is the chain's end): a periodic one (PULSAR_SEGSTORE_RP_PERIODIC) at each link pulsar_kvchain_rp_periodic
 *  names, and with PULSAR_KVCHAIN_RP in `what` one at the chain's end (PULSAR_SEGSTORE_RP_ANCHOR with
 *  PULSAR_KVCHAIN_ANCHOR) -- each where `src` holds a checkpoint, one write per key (a key that has one gains the
 *  flags), copied out and handed to `writer`.  Older restore points of the chain stay (eviction takes the unflagged ones
 *  first, the oldest first).  Counts into r (rp_bytes, rp_written, no_room); the first failure is r->err (when it is empty).
 *  @param st     the store
 *  @param links  the chain's segments this persist passed, ascending
 *  @param n      how many
 *  @param grid   the disk grid
 *  @param what   the persist's request (PULSAR_KVCHAIN_*)
 *  @param src    the checkpoints
 *  @param writer where the copied-out restore points go
 *  @param r      the persist's result */
void pulsar_kvchain_place_rps(pulsar_segstore *st, const pulsar_kvchain_link *links, int n, int grid, unsigned what,
                              const pulsar_kvchain_rp_source *src, const pulsar_kvchain_writer *writer,
                              pulsar_kvchain_persist_result *r);

/** Persist the installed bank's history (see above), never past
 *  pulsar_session_persist_end (L281: the first block whose image the session does not record).  `what` (L347 I2):
 *  PULSAR_KVCHAIN_TAIL ends the chain with the tail to the deepest grid checkpoint (pulsar_kvchain_segment_ends),
 *  PULSAR_KVCHAIN_RP leaves the restore point at the chain's end, PULSAR_KVCHAIN_ANCHOR (with _RP) flags it an
 *  anchor; the periodic restore points are written whatever it asks (pulsar_kvchain_place_rps).  A grid point an image
 *  block lies across is not a cut (the segment runs on to the next one).  Nothing is written when the deepest grid
 *  checkpoint is below `min_tokens`.  Returns `written` (segments the store did not hold). */
/// @param st         the store
/// @param e          the engine
/// @param s          the session whose installed bank is persisted
/// @param min_tokens write nothing when the deepest grid checkpoint is below this
/// @param what       PULSAR_KVCHAIN_TAIL / _RP / _ANCHOR (0: the rows to the last grid point, and periodic points)
/// @param trailer    the per-segment trailer, or NULL for none
/// @param writer     where the copied-out files go (required)
/// @param out        receives what the persist did, or NULL
/// @return the number of segments written
int pulsar_kvchain_persist(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, int min_tokens, unsigned what,
                           const pulsar_kvchain_trailer *trailer, const pulsar_kvchain_writer *writer,
                           pulsar_kvchain_persist_result *out);

/** L310: the chain a restore of `text` would load, without loading it -- the deepest whose text is a byte prefix of
 *  `text` ending at a restore point, cut at the request's images exactly as pulsar_kvchain_restore cuts it (and back
 *  to the last restore point at or below the cut) -- into `chain` (root first), and *bytes what a load reads
 *  (pulsar_kvchain_load_bytes).  Returns its length; 0 when there is none or it ends below `min_tokens`.  `err` says
 *  why a stored chain was cut short. */
/// @param st           the store
/// @param text         the request's rendered text
/// @param text_len     bytes of `text`
/// @param image_hashes content hashes of the request's images in order, or NULL
/// @param n_images     entries in `image_hashes`
/// @param min_tokens   report no chain ending below this
/// @param chain        receives the chain, root first
/// @param cap          capacity of `chain`
/// @param bytes        receives what a load reads, or NULL
/// @param err          receives why a stored chain was cut short
/// @param errlen       size of `err`
/// @return the chain's length; 0 when there is none
int pulsar_kvchain_find(pulsar_segstore *st, const char *text, size_t text_len, const uint64_t *image_hashes,
                        int n_images, int min_tokens, pulsar_segstore_seg *chain, int cap, uint64_t *bytes, char *err,
                        size_t errlen);
/** L323: the bytes a load of `chain` (n segments) reads -- every segment's rows and trailer, the last one's restore
 *  point file too (L347) -- the read pulsar_kvchain_find prices against the prefill it saves. */
/// @param chain the chain, root first
/// @param n     its segment count
/// @return bytes a load of it reads
uint64_t pulsar_kvchain_load_bytes(const pulsar_segstore_seg *chain, int n);
/** L347 I3a: how many of those bytes are in the store's RAM copies now (pulsar_segstore_resident_bytes over the same
 *  parts). */
/// @param st    the store
/// @param chain the chain, root first
/// @param n     its segment count
/// @return bytes of those parts resident in RAM
uint64_t pulsar_kvchain_resident_bytes(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n);
/** L330: what a load of segment `i` of `chain` (n segments) reads: its rows and trailer, and its restore point file
 *  when it is the last (pulsar_kvchain_load_bytes is the sum over the chain). */
/// @param chain the chain, root first
/// @param n     its segment count
/// @param i     the segment's index in `chain`
/// @return bytes a load of segment `i` reads
uint64_t pulsar_kvchain_segment_load_bytes(const pulsar_segstore_seg *chain, int n, int i);
/** L330: one segment's load into whatever `ud` names (pulsar_session_load_segment's contract: `last` restores the
 *  state at the segment's end from its restore point; PULSAR_SEGMENT_OK = loaded, *G the position it stands at; L347
 *  I3b: else the refusal's class).  L347 I3a: from memory -- the parts as pulsar_segstore_map_part mapped them,
 *  digests already checked. */
/// @param ud       the loader's context
/// @param seg      the segment's payload
/// @param bytes    the payload's size
/// @param last     restore the state at its end too (the chain's last segment)
/// @param rp       L347: the segment's restore point when `last`, else NULL
/// @param rp_bytes its size (0 when not `last`)
/// @param G        receives the position the bank stands at, or NULL
/// @param key      the segment's key
/// @param err      receives a reason on failure
/// @param errlen   size of `err`
typedef pulsar_segment_status (*pulsar_kvchain_segment_loader)(void *ud, const uint8_t *seg, uint64_t bytes, bool last,
                                                               const uint8_t *rp, uint64_t rp_bytes, int *G,
                                                               const char *key, char *err, size_t errlen);
/** L347 I3b: one segment's check, nothing written (pulsar_session_check_segment's contract). */
/// @param ud       the checker's context
/// @param seg      the segment's payload
/// @param bytes    the payload's size
/// @param last     the chain's last segment (its restore point is checked too)
/// @param rp       its restore point when `last`, else NULL
/// @param rp_bytes its size (0 when not `last`)
/// @param err      receives the fault
/// @param errlen   size of `err`
typedef pulsar_segment_status (*pulsar_kvchain_segment_checker)(void *ud, const uint8_t *seg, uint64_t bytes,
                                                                bool last, const uint8_t *rp, uint64_t rp_bytes,
                                                                char *err, size_t errlen);
/** L330: load the next slice of a chain -- segments [*next, *next + k) of the `n` pulsar_kvchain_find returned, cut at
 *  the chain's end -- in order through `load` (the chain's last segment with its restore point, L347), advancing
 *  *next past each segment loaded.  The whole chain in one slice
 *  is the one-shot load (pulsar_kvchain_load); the server reads a long chain a slice per worker turn.  1 on a content
 *  fault: what it invalidates is dropped from the store (a segment's rows: it and everything below it, by key; a
 *  restore point: it alone) and `err` says which; *next stays at it.  L347 I3a / I3b (plans/124 D7): 2 on a transient
 *  failure (no room in the RAM pool to read a part now, the bank, the device, a TP rank's copy, a bad slice): nothing
 *  is dropped.  L375 (O25): 3 when a part's file could not be read now (pulsar_segstore_map_part's 3): nothing is
 *  dropped either, and unlike 2 nothing the store's writer finishes changes it -- a caller that waits for the
 *  writer's commits on 2 does not on 3.  A part whose read keeps failing becomes a content fault in the store
 *  (PULSAR_SEGSTORE_READ_FAILS_MAX) and is dropped here as one.  Bytes and order are the one-shot's whatever `k` is: a slice ends at a segment boundary, where the bank
 *  stands at the next segment's start. */
/// @param st     the store (a failed segment is dropped from it)
/// @param chain  the chain pulsar_kvchain_find returned
/// @param n      its segment count
/// @param next   in: the first segment to load; out: past the last one loaded
/// @param k      segments in the slice (cut at the chain's end)
/// @param load   the per-segment loader
/// @param ud     context for `load`
/// @param err    receives which segment failed
/// @param errlen size of `err`
/// @return 0 when the slice loaded, 1 on a content fault (dropped), 2 on a transient failure (kept), 3 on a part's
///         read that failed now (kept)
int pulsar_kvchain_load_slice_with(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int *next, int k,
                                   pulsar_kvchain_segment_loader load, void *ud, char *err, size_t errlen);
/** L347 I3b: check the next slice of a chain before any of it is loaded -- segments [*next, *next + k): each part mapped
 *  (from RAM, or read off disk and digest-checked) and its bytes checked by `check`, nothing written -- advancing *next
 *  past each.  Faults as pulsar_kvchain_load_slice_with (1: content, dropped; 2: transient, kept; 3: a read that
 *  failed now, kept).  A restore that runs
 *  it over the whole chain first leaves the bank untouched by any content fault. */
/// @param st     the store (a faulty part is dropped from it)
/// @param chain  the chain pulsar_kvchain_find returned
/// @param n      its segment count
/// @param next   in: the first segment to check; out: past the last one checked
/// @param k      segments in the slice (cut at the chain's end)
/// @param check  the per-segment check
/// @param ud     context for `check`
/// @param err    receives which segment failed
/// @param errlen size of `err`
/// @return 0 when the slice checked, 1 on a content fault (dropped), 2 on a transient failure (kept)
int pulsar_kvchain_verify_slice_with(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, int *next, int k,
                                     pulsar_kvchain_segment_checker check, void *ud, char *err, size_t errlen);
/** L347 I3b: pulsar_kvchain_verify_slice_with against the installed bank of `s` (pulsar_session_check_segment); the
 *  bank is not touched. */
/// @param st     the store
/// @param s      the session whose installed bank's layout the bytes must be
/// @param chain  the chain
/// @param n      its segment count
/// @param next   in: the first segment to check; out: past the last one checked
/// @param k      segments in the slice
/// @param err    receives which segment failed
/// @param errlen size of `err`
/// @return as pulsar_kvchain_verify_slice_with
int pulsar_kvchain_verify_slice(pulsar_segstore *st, pulsar_session *s, const pulsar_segstore_seg *chain, int n,
                                int *next, int k, char *err, size_t errlen);
/** L347 I3b: hold every segment of `chain` against eviction (pulsar_segstore_pin) while a restore
 *  reads it, or end that hold -- one call each way.
 *  @param st    the store
 *  @param chain the chain
 *  @param n     its segment count
 *  @param pin   take the hold (true) or end it (false) */
void pulsar_kvchain_pin(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n, bool pin);
/** pulsar_kvchain_load_slice_with into the installed bank of `s` (pulsar_session_load_segment); a failed slice leaves
 *  the bank invalidated. */
/// @param st     the store
/// @param s      the session whose installed bank is loaded
/// @param chain  the chain pulsar_kvchain_find returned
/// @param n      its segment count
/// @param next   in: the first segment to load; out: past the last one loaded
/// @param k      segments in the slice
/// @param err    receives which segment failed
/// @param errlen size of `err`
/// @return 0 when the slice loaded, else pulsar_kvchain_load_slice_with's code (the bank is then invalidated)
int pulsar_kvchain_load_slice(pulsar_segstore *st, pulsar_session *s, const pulsar_segstore_seg *chain, int n,
                              int *next, int k, char *err, size_t errlen);
/** L330: a loaded chain's last word (L196): the bank's tokens end at the chain's end and agree with `text` at its
 *  boundary.  Returns the position (the chain is touched); 0 when they disagree (the chain's last segment is dropped,
 *  the bank invalidated, `err` says so). */
/// @param st     the store
/// @param e      the engine
/// @param s      the session whose installed bank holds the loaded chain
/// @param text   the request's rendered text
/// @param chain  the loaded chain
/// @param n      its segment count
/// @param err    receives the disagreement
/// @param errlen size of `err`
/// @return the bank's position, 0 when tokens and text disagree
int pulsar_kvchain_load_check(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                              const pulsar_segstore_seg *chain, int n, char *err, size_t errlen);
/** L310: load the `n` segments pulsar_kvchain_find returned for `text` into the installed bank -- the whole chain as
 *  one slice, then the boundary check.  Returns the position the bank stands at; 0 when a segment would not load or
 *  the chain disagrees with its tokens at the boundary (that chain is dropped, the bank invalidated, `err` says
 *  which).  The chain is touched on success. */
/// @param st     the store
/// @param e      the engine
/// @param s      the session whose installed bank is loaded
/// @param text   the request's rendered text
/// @param chain  the chain pulsar_kvchain_find returned
/// @param n      its segment count
/// @param err    receives the reason on failure
/// @param errlen size of `err`
/// @return the position the bank stands at; 0 on failure
int pulsar_kvchain_load(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                        const pulsar_segstore_seg *chain, int n, char *err, size_t errlen);

/** Restore the deepest stored chain for `text` into the installed bank (see
 *  above), at most `cap` segments reported in `chain` (*n_out of them).  `image_hashes` are the content hashes
 *  (pulsar_image_hash) of the images the history must hold, in order -- a request's images, or a spilled bank's own
 *  records (pulsar_session_image_hashes); NULL / 0 for none: the chain loads only as far as its records match them.
 *  Returns the position it stands at, 0 when nothing was loaded; `err` says why when a stored chain was refused or
 *  stopped short of its end.  A chain ending below `min_tokens` is not loaded.  The chain is touched on success. */
/// @param st           the store
/// @param e            the engine
/// @param s            the session whose installed bank is restored
/// @param text         the request's rendered text
/// @param text_len     bytes of `text`
/// @param image_hashes content hashes of the images the history must hold, in order, or NULL
/// @param n_images     entries in `image_hashes`
/// @param min_tokens   load no chain ending below this
/// @param chain        receives the chain's segments
/// @param cap          capacity of `chain`
/// @param n_out        receives the number of segments reported
/// @param err          receives why a stored chain was refused or stopped short
/// @param errlen       size of `err`
/// @return the position the bank stands at, 0 when nothing was loaded
int pulsar_kvchain_restore(pulsar_segstore *st, pulsar_engine *e, pulsar_session *s, const char *text,
                           size_t text_len, const uint64_t *image_hashes, int n_images, int min_tokens,
                           pulsar_segstore_seg *chain, int cap, int *n_out, char *err, size_t errlen);

#endif

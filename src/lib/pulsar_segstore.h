/** @file
 *  The server's disk KV cache as a content-addressed segment store (L264 S4): keys, chains, budget, lookup. */
#ifndef PULSAR_SEGSTORE_H
#define PULSAR_SEGSTORE_H

/* L264 S4: the server's disk KV cache as a content-addressed SEGMENT store.
 *
 * A segment holds one stretch of a conversation's KV: the engine's segment
 * payload (pulsar_session_copy_out_segment: rows [G_prev, G)) plus the rendered
 * text of those tokens.  Its key is
 *     sha1(parent_key || text)          (parent_key "" for a root, G_prev 0)
 * so a key names the WHOLE text from position 0 to its G, and two conversations
 * that share a prefix share its segments.  A lookup walks the chains from the
 * roots, hashing the request's own bytes against each child's key, and returns
 * the deepest chain whose text is a byte prefix of the request -- no text is
 * held in memory, only keys, parents and lengths.
 *
 * Storage: <dir>/<key>.seg = fixed header, text, payload, optional trailer
 * (the server's tool map).  L347 I3a: every file is first an IMAGE in host memory
 * (the file's bytes, padded to the O_DIRECT granule), written with O_DIRECT (tmp +
 * rename, no fsync: the size check at open and the payload's digest at a read
 * reject a torn file) and read back with O_DIRECT -- no page cache either way.  A
 * PAYLOAD's last 8 bytes are its digest (lib/pulsar_kvdigest.h), sealed by the
 * file write and checked by every read off disk; the store holds no other view of
 * the payload's format.  Eviction keeps the budget by removing
 * a LEAF: a segment with children is part of a longer chain and goes only after
 * them.  L323: a DOMINATED leaf goes before other leaves -- one whose text is a
 * byte prefix of a sibling's (the same conversation's older tail, superseded by a
 * later tail or by the grid segment that covers it) -- then the least recently
 * used (L347 I2: after the unflagged restore points, below).
 *
 * L347: the state at a segment's end G lives apart, in the key's RESTORE POINT
 * <dir>/<key>.rp = fixed header + the engine's restore point
 * (pulsar_session_copy_out_restore_point).  The index records its file's size
 * (rp_bytes), so a lookup can ask for a chain that ends at a restore point and
 * a load can price "every segment's rows + the last one's restore point"
 * without opening a file.  The budget counts both files.
 *
 * L347 I2: a restore point carries FLAGS (PULSAR_SEGSTORE_RP_ANCHOR, _PERIODIC: why it was written, kept in the .rp
 * header so they survive a restart).  Eviction takes, in order (oldest first within each class, never a key pinned
 * or pending): (0) a restore point that is not flagged and not on a branch point (a key with >= 2 children); (1) a
 * dominated leaf; (2) any other leaf, LRU; (3) a flagged or branch-point restore point.  A leaf's restore point so
 * always goes before the leaf.  The dominated rule applies only to a leaf without a restore point (so unflagged).
 * An older end of a chain keeps its restore point when the chain goes on (a later branch resumes there -- an edited or
 * regenerated turn); retention alone makes room: class (0) takes the unflagged ones first, the oldest first.
 *
 * L347 I3a (plans/124 D9): the RAM TIER.  With a pool attached (pulsar_segstore_set_ram), images are pool blocks; with
 * `retain` the image of every written file stays as its entry's RAM copy (residency: disk, or disk and RAM -- an entry
 * is indexed only once its file is written), a read of a resident part is a pointer into it (no disk I/O), and a part
 * read off disk becomes its entry's RAM copy.  RAM copies are evicted to make room in the pool in the disk eviction's
 * order (an unflagged off-branch restore point, a dominated leaf's rows, other rows, a flagged or branch-point restore
 * point; LRU within a class), never a part mapped for a load; the file stays.  Without `retain` (a unified-memory
 * host, where host RAM is the GPU's) an image lives only while it is written or read.
 *
 * L347 I3b (plans/124 D7, D10): PINNED anchors and restore pins.  A restore point flagged PULSAR_SEGSTORE_RP_PINNED (a
 * judge's preamble, X-Pulsar-KV: pin-anchor) is never evicted from the disk or the RAM tier -- and so neither is any segment of its chain (its key's rows go only after its restore point, its
 * ancestors are not leaves).  A pin is taken away (logged, counted) only when the store needs room and nothing else
 * may go, or when the pinned anchors hold more than their budget (pulsar_segstore_set_pinned_max): the least recently
 * used first.  Apart from that, pulsar_segstore_pin holds a key against eviction while a restore
 * reads its chain (the same count as a write in flight's parent). */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/// Magic of a .seg file's header: "PSG1".
#define PULSAR_SEGSTORE_MAGIC   0x31475350u   /* "PSG1" */
/// Store format version; v3 (L347): rows-only payloads, the state in the key's .rp.  Older versions are removed at open.
#define PULSAR_SEGSTORE_VERSION 5u            /* v5 (L353): a key covers the span's image content hashes; v2 (L323): the header carries the payload's state bytes; v3 (L347): no state part (the word is 0), the state at G is the key's .rp; v4 (L347 I3a): the payloads' digest is pulsar_kvdigest v2, sealed and checked by the store */
/// Magic of a .rp file's header: "PRP1" (L347).
#define PULSAR_SEGSTORE_RP_MAGIC   0x31505250u   /* "PRP1" */
/// Restore point file format version; v2 (L347 I2): the header's word at 12 is the restore point's flags; v3 (L347
/// I3a): the payload's digest is pulsar_kvdigest v2.  Older versions are removed at open.
#define PULSAR_SEGSTORE_RP_VERSION 3u
/// L375 (O25): a part whose file read fails this many times in a row with a persistent errno (EIO, EACCES, EISDIR,
/// ...; never ENOENT, which is a fault at once, and never a passing shortage -- EMFILE, ENFILE, ENOMEM, ENOBUFS,
/// EAGAIN -- which counts nothing: L353 no. 13) is reported as a content fault and dropped by its reader.
#define PULSAR_SEGSTORE_READ_FAILS_MAX 3
/// Restore point flag (L347 I2): written at a request's shared-prefix anchor (the second sighting of its preamble)
#define PULSAR_SEGSTORE_RP_ANCHOR 1u
/// Restore point flag (L347 I2): written at a periodic grid point (pulsar_kvchain_rp_periodic)
#define PULSAR_SEGSTORE_RP_PERIODIC 2u
/// Restore point flag (L347 I3b): a pinned anchor -- never evicted (disk or RAM); its chain neither.
/// The header's flags word holds it (no layout change, so the .rp version stays)
#define PULSAR_SEGSTORE_RP_PINNED 4u
/// L347 I3a: a key's segment file (pulsar_segstore_map_part, pulsar_segstore_job_kind)
#define PULSAR_SEGSTORE_PART_ROWS 0
/// L347 I3a: a key's restore point file
#define PULSAR_SEGSTORE_PART_RP 1
/// Header flag: a trailer follows the payload.
#define PULSAR_SEGSTORE_FLAG_TRAILER 1u       /* a trailer follows the payload */

/// An open segment store (opaque): its directory, budget, identity and in-memory index.
typedef struct pulsar_segstore pulsar_segstore;
/// L347 I3a: the host memory pool images come from (lib/pulsar_kvpool.h).
typedef struct pulsar_kvpool pulsar_kvpool;

/** One segment as the index knows it. */
typedef struct {
    /// This segment's key: 40 hex chars of sha1(parent || text) + NUL
    char key[41];
    char parent[41];          ///< "" for a root
    /** Token span [G_prev, G): G_prev is the parent's end (0 for a root), G this segment's end. */
    uint32_t G_prev, G;       ///< token span
    uint64_t text_end;        ///< bytes of chain text from position 0 to G (index-derived)
    uint64_t text_len;        ///< this segment's own text bytes
    /// Engine payload bytes (rows only, L347), pulsar_session_copy_out_segment's output
    uint64_t payload_bytes;
    uint64_t rp_bytes;        ///< L347: bytes of the key's .rp file, the restore point at G (0 = none)
    uint32_t rp_flags;        ///< L347 I2: its PULSAR_SEGSTORE_RP_* flags (0 without one)
    /// Optional trailer bytes after the payload (0 = none)
    uint64_t trailer_bytes;
    /// The whole .seg file: header + text + payload + trailer; what the budget counts
    uint64_t file_bytes;
    /// Lookups that used this segment (pulsar_segstore_touch)
    uint32_t hits;
    uint64_t last_used;       ///< unix seconds
    /// L323: its text is a byte prefix of a sibling's (index-derived); eviction counts it only for a leaf without a
    /// restore point (L347 I2)
    bool dominated;
    /// L378: its rows were mapped for a load since the store opened (pulsar_segstore_map_part); index only, never on
    /// disk, so a file kept from an earlier run starts unread
    bool rows_read;
    /// L378: ... and its restore point (cleared when the restore point is removed or replaced)
    bool rp_read;
} pulsar_segstore_seg;

/** L323: which chain ends a lookup accepts.  `grid`: the last segment's G is a multiple of it (0 = any end);
 *  `state`: the last segment has a restore point (L347). */
typedef struct {
    /// In tokens: the chain's end G must be a multiple of it (0 = any end)
    uint32_t grid;
    /// The last segment must have a restore point (rp_bytes > 0)
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
/// @param budget_bytes byte budget for the .seg and .rp files; 0 = unbounded
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
/** Close the store and free its index and its RAM copies (no job may be out); the segment files stay on disk.
 *  @param st the store, or NULL */
void pulsar_segstore_close(pulsar_segstore *st);
/** L347 I3a: attach the host memory pool images are carved from, and turn the RAM tier on (`retain`: a written or
 *  read image stays as its entry's RAM copy) -- right after the open, before any write.  Without a pool images are
 *  posix_memalign'd per job and never kept (the agent, the gates).
 *  @param st     the store
 *  @param pool   the pool (the caller's; it outlives the store), or NULL
 *  @param retain keep RAM copies (a discrete-GPU host); false on a unified-memory host */
void pulsar_segstore_set_ram(pulsar_segstore *st, pulsar_kvpool *pool, bool retain);

/// @param parent the parent's key ("" for a root)
/// @param text   the segment's own rendered text
/// @param len    bytes of `text`
/// @param out    receives the 40 hex chars of sha1(parent || text) and a NUL
/** The key rule, the one authority for it. */
void pulsar_segstore_child_key(const char *parent, const char *text, size_t len, char out[41]);

/** L353: a segment's key: its parent, its text, and the content hashes of the image blocks in its span.  An image block
 *  renders as a fixed placeholder, so without the hashes two same-size images share every key: "the same key is the
 *  same state" needs them.  With n == 0 it is pulsar_segstore_child_key.
 *  @param parent parent key, or "" for a root
 *  @param text   the span's text
 *  @param len    bytes of `text`
 *  @param hashes [n] the span's image content hashes, in position order
 *  @param n      hashes
 *  @param out    receives the 40 hex digits and a NUL */
void pulsar_segstore_child_key_images(const char *parent, const char *text, size_t len, const uint64_t *hashes,
                                      uint32_t n, char out[41]);

/** L353: pulsar_segstore_lookup for a request carrying images: each child's key is reproduced from the request's bytes
 *  and its next unconsumed image hashes, so a chain passes a block only for that block's image, and stops at it for any
 *  other (or for a request with none).  One rule for the persist's parent, a restore and a text request.
 *  @param st     the store
 *  @param text   the request's text
 *  @param len    bytes of `text`
 *  @param want   which ends are accepted, or NULL for any
 *  @param hashes [n] the request's image content hashes, in position order; NULL with n == 0 for none
 *  @param n      hashes
 *  @param chain  receives the deepest accepted chain, root first
 *  @param cap    entries `chain` holds
 *  @return segments in `chain`, 0 when none */
int pulsar_segstore_lookup_images(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                                  const uint64_t *hashes, uint32_t n, pulsar_segstore_seg *chain, int cap);

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
 * segment: a put of a stored key writes nothing. */
bool pulsar_segstore_put(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G,
                         const char *text, size_t text_len,
                         uint64_t payload_bytes, pulsar_segstore_write_fn write_payload,
                         uint64_t trailer_bytes, pulsar_segstore_write_fn write_trailer,
                         void *ud, char key_out[41], char *err, size_t errlen);

/** L338: one put in three steps, so the file write can run on another thread.  pulsar_segstore_reserve makes the
 *  room and names the segment (the store's thread); L347 I3a: it allocates the file's image, writes its header and
 *  text there, and the caller fills the payload (pulsar_segstore_job_payload: the engine copies out into it) and the
 *  trailer; pulsar_segstore_job_write seals the payload's digest and writes the image and reads no store state (any
 *  thread); pulsar_segstore_commit indexes it -- its image kept as the entry's RAM copy with the RAM tier, else freed --
 *  or drops it (the store's thread).  While a job is out its key is pending (a reserve of it writes nothing, eviction
 *  skips it, a child may name it as parent) and its parent is never evicted; lookups do not see it until the commit.
 *  pulsar_segstore_put is the three in a row, its parts written through memory streams. */
typedef struct pulsar_segstore_job pulsar_segstore_job;
/** L353: pulsar_segstore_reserve for a span holding image blocks: the key covers their content hashes.
 *  @param hashes [n_hashes] the span's image content hashes, in position order; the rest as pulsar_segstore_reserve
 *  @param n_hashes hashes
 *  @return as pulsar_segstore_reserve */
/// @param st            the store
/// @param parent        parent key, or "" for a root
/// @param G_prev        the span's start
/// @param G             the span's end
/// @param text          the span's text
/// @param text_len      bytes of `text`
/// @param payload_bytes the rows' bytes
/// @param trailer_bytes the trailer's bytes
/// @param key_out       receives the key
/// @param job_out       receives the job
/// @param err           receives a reason on failure
/// @param errlen        size of `err`
int pulsar_segstore_reserve_images(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G, const char *text,
                                   size_t text_len, const uint64_t *hashes, uint32_t n_hashes, uint64_t payload_bytes,
                                   uint64_t trailer_bytes, char key_out[41], pulsar_segstore_job **job_out, char *err,
                                   size_t errlen);

/// @param st            the store
/// @param parent        the parent's key, "" for a root (stored, or pending)
/// @param G_prev        the segment's first token position (the parent's G)
/// @param G             the segment's end position
/// @param text          the segment's own rendered text (copied)
/// @param text_len      bytes of `text`
/// @param payload_bytes bytes the payload writer will write
/// @param trailer_bytes bytes the trailer writer will write (0 = none)
/// @param key_out       receives the segment's key
/// @param job_out       receives the job on 1, NULL otherwise
/// @param err           receives a reason on -1
/// @param errlen        size of `err`
/// @return 1: a job to write and commit; 0: the store holds (or is writing) what this write would; -1: refused; -2:
///         no room in the RAM pool now (nothing changed but evictions; the caller may wait for its writer and retry)
int pulsar_segstore_reserve(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G, const char *text,
                            size_t text_len, uint64_t payload_bytes, uint64_t trailer_bytes, char key_out[41],
                            pulsar_segstore_job **job_out, char *err, size_t errlen);
/** L347 I3a: a restore point's reserve -- the restore point of stored (or pending) segment `key`, as a job like a
 *  segment's: room made (never `key`), its image allocated with its header, the key pinned until the commit; the
 *  caller fills the payload, pulsar_segstore_job_write writes it, pulsar_segstore_commit indexes it (a pending
 *  segment's joins it at that segment's commit, and goes if its write fails).  A stored restore point is replaced at
 *  the commit (its flags joined).  `flags` (PULSAR_SEGSTORE_RP_*) join those of one in flight or waiting for its
 *  segment, which then writes nothing.
 *  @param st       the store
 *  @param key      40 hex chars of a stored or pending segment
 *  @param rp_bytes the restore point's bytes (the engine's restore point; its last 8 are its digest)
 *  @param flags    why it is written (PULSAR_SEGSTORE_RP_*; 0 for a chain end)
 *  @param job_out  receives the job on 1, NULL otherwise
 *  @param err      receives a reason on -1 / -2
 *  @param errlen   size of `err`
 *  @return 1: a job; 0: one is in flight or waiting (it gained the flags); -1: refused; -2: no room in the RAM pool */
int pulsar_segstore_reserve_rp(pulsar_segstore *st, const char key[41], uint64_t rp_bytes, uint32_t flags,
                               pulsar_segstore_job **job_out, char *err, size_t errlen);
/// @param job           the job (pulsar_segstore_reserve / _reserve_rp), its payload (and trailer) filled
/// @param err           receives a reason on failure
/// @param errlen        size of `err`
/// @return true when the file is in place (sealed, written with O_DIRECT, renamed); false leaves no file
bool pulsar_segstore_job_write(pulsar_segstore_job *job, char *err, size_t errlen);
/// @param st      the store the job was reserved in
/// @param job     the job (freed here)
/// @param written its pulsar_segstore_job_write succeeded
/// @return true when the segment is indexed (false: the write failed, or its parent's did and the file is removed)
bool pulsar_segstore_commit(pulsar_segstore *st, pulsar_segstore_job *job, bool written);
/// @param job the job
/// @return its key (40 hex chars)
const char *pulsar_segstore_job_key(const pulsar_segstore_job *job);
/// @param job the job
/// @return PULSAR_SEGSTORE_PART_ROWS (a segment) or _RP (a restore point)
int pulsar_segstore_job_kind(const pulsar_segstore_job *job);
/// @param job the job
/// @return its first token position (0 for a restore point)
uint32_t pulsar_segstore_job_G_prev(const pulsar_segstore_job *job);
/// @param job the job
/// @return its end position (a restore point: its segment's, 0 while that one is pending)
uint32_t pulsar_segstore_job_G(const pulsar_segstore_job *job);
/// @param job the job
/// @return the file's bytes
uint64_t pulsar_segstore_job_bytes(const pulsar_segstore_job *job);
/// @param job the job
/// @return its image's bytes (pulsar_segstore_image_bytes of the file's)
uint64_t pulsar_segstore_job_image_bytes(const pulsar_segstore_job *job);
/// @param job the job
/// @return where its payload goes in the image (pulsar_segstore_job_payload_bytes of them; the last 8 are sealed)
uint8_t *pulsar_segstore_job_payload(pulsar_segstore_job *job);
/// @param job the job
/// @return its payload's bytes
uint64_t pulsar_segstore_job_payload_bytes(const pulsar_segstore_job *job);
/// @param job the job
/// @return where its trailer goes in the image, NULL when it has none
uint8_t *pulsar_segstore_job_trailer(pulsar_segstore_job *job);
/** L347 I3a: a segment file's bytes (header, text, payload, trailer).
 *  @param text_len      its text's bytes
 *  @param payload_bytes its payload's bytes
 *  @param trailer_bytes its trailer's bytes
 *  @return the file's bytes */
uint64_t pulsar_segstore_seg_file_bytes(uint64_t text_len, uint64_t payload_bytes, uint64_t trailer_bytes);
/** L347 I3a: a restore point file's bytes (header, payload).
 *  @param rp_bytes the restore point's bytes
 *  @return the file's bytes */
uint64_t pulsar_segstore_rp_file_bytes(uint64_t rp_bytes);
/** L347 I3a: the image a file of `file_bytes` takes in the pool: one byte past the file (a memory stream's NUL),
 *  rounded up to the O_DIRECT granule.
 *  @param file_bytes the file's bytes
 *  @return the image's bytes */
uint64_t pulsar_segstore_image_bytes(uint64_t file_bytes);
/** L347 I3a: make room in the pool for an image of a `file_bytes` file now, evicting RAM copies (never a mapped part).
 *  @param st         the store
 *  @param file_bytes the file's bytes
 *  @return true when such an image fits now (always without a pool) */
bool pulsar_segstore_ram_make_room(pulsar_segstore *st, uint64_t file_bytes);

/** L347: write the restore point of stored segment `key` (the state at its G) as `<dir>/<key>.rp` now --
 *  pulsar_segstore_reserve_rp, the payload through `write_rp`, the write and the commit in a row -- replacing one it
 *  has.  Makes room first by evicting (never `key` itself); the budget counts
 *  the file.  A key still being written (reserved, not committed) takes it too: the restore point joins the segment at
 *  its commit, and is removed if the segment's write fails.  L347 I2: `flags` (PULSAR_SEGSTORE_RP_*) are joined to
 *  those of a restore point it replaces -- the same key is the same state, its reasons to be kept only accumulate.
 *  @param st       the store
 *  @param key      40 hex chars of a stored (or pending) segment
 *  @param rp_bytes bytes write_rp will write (the engine's restore point)
 *  @param flags    why it is written (PULSAR_SEGSTORE_RP_*; 0 for a chain end)
 *  @param write_rp writes them
 *  @param ud       context for write_rp
 *  @param err      receives a reason on failure
 *  @param errlen   size of `err`
 *  @return true when the restore point is in place (false leaves any older one as it was) */
bool pulsar_segstore_put_rp(pulsar_segstore *st, const char key[41], uint64_t rp_bytes, uint32_t flags,
                            pulsar_segstore_write_fn write_rp, void *ud, char *err, size_t errlen);
/** L347 I2: join `flags` to the flags of stored segment `key`'s restore point, in its index entry and its file's
 *  header (in place: the state is the same).  L347 I3b: a new PULSAR_SEGSTORE_RP_PINNED marks it used now and holds
 *  the pinned budget (pulsar_segstore_set_pinned_max).
 *  @param st    the store
 *  @param key   40 hex chars
 *  @param flags PULSAR_SEGSTORE_RP_* to add
 *  @return true when the key has a restore point and its header took the flags */
bool pulsar_segstore_flag_rp(pulsar_segstore *st, const char key[41], uint32_t flags);
/** L347 I2: does the store hold the rows of text[0, len)?  True when a stored chain's text runs through byte `len`:
 *  one ends exactly there, or its next segment's own text starts with the rest of `text` (a tail written by an earlier
 *  request that went on past `len`; its text is read from its file only when the first bytes the index caches
 *  match).  *rp_at_len says whether a stored chain ending exactly at `len` has a restore point.
 *  @param st        the store
 *  @param text      the text
 *  @param len       bytes of it to cover
 *  @param rp_at_len receives whether a chain ending at `len` has a restore point
 *  @return true when stored rows cover text[0, len) */
bool pulsar_segstore_covers(pulsar_segstore *st, const char *text, size_t len, bool *rp_at_len);
/** L347 I3a: a stored part's payload for a load: its RAM copy when it has one, else its file read whole with O_DIRECT
 *  into an image, checked (size, header, the payload's digest) -- kept as its RAM copy with the RAM tier, else lent
 *  until the unmap.  The part stays mapped (never evicted from RAM) until pulsar_segstore_unmap_part.
 *  @param st        the store
 *  @param key       40 hex chars
 *  @param part      PULSAR_SEGSTORE_PART_ROWS or _RP
 *  @param payload   receives the payload's bytes
 *  @param bytes     receives their count
 *  @param from_disk receives whether the file was read (NULL: not wanted)
 *  @param err       receives the reason on failure
 *  @param errlen    size of `err`
 *  @return 0 mapped; 1 the part is not stored, or its file is missing or bad, or (L375) its read has now failed
 *          PULSAR_SEGSTORE_READ_FAILS_MAX times in a row with an errno that is not a passing shortage (a content
 *          fault: drop it); 2 no room in the RAM pool to read it now (images come back at the writer's commits: a
 *          wait for them can help); 3 (L375) its file is there and could not be read now (EMFILE, ENOMEM, EIO, ...):
 *          nothing a wait on the writer changes, and nothing to drop */
int pulsar_segstore_map_part(pulsar_segstore *st, const char key[41], int part, const uint8_t **payload,
                             uint64_t *bytes, bool *from_disk, char *err, size_t errlen);
/** L347 I3a: end a pulsar_segstore_map_part (a lent image is freed).
 *  @param st   the store
 *  @param key  40 hex chars
 *  @param part the part mapped */
void pulsar_segstore_unmap_part(pulsar_segstore *st, const char key[41], int part);
/** L347: remove a segment's restore point (the segment stays).  The removed hook is called with `rp_only`.
 *  @param st  the store
 *  @param key 40 hex chars
 *  @return true when there was one to remove */
bool pulsar_segstore_drop_rp(pulsar_segstore *st, const char key[41]);

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
/** L347 I3a (replaces L327's page-cache probe: the store's O_DIRECT reads and writes leave nothing there): how many of
 *  a load's bytes of `key` are in its RAM copies now -- with `rows` its payload and trailer when its rows are
 *  resident, with `rp` its restore point file when that is -- a read of them costs an H2D copy, not the disk's.
 *  @param st   the store
 *  @param key  40 hex chars
 *  @param rows count its rows
 *  @param rp   count its restore point
 *  @return the resident bytes (0 for a key not stored) */
uint64_t pulsar_segstore_resident_bytes(pulsar_segstore *st, const char key[41], bool rows, bool rp);
/** L347 I3a: the pool bytes the RAM copies hold.
 *  @param st the store, or NULL (0)
 *  @return bytes */
uint64_t pulsar_segstore_ram_bytes(const pulsar_segstore *st);
/// @param st    the store
/// @param chain the chain's entries, as pulsar_segstore_lookup returned them
/// @param n     entries in `chain`
/** Record a use of every segment of a chain (hit count + last_used). */
void pulsar_segstore_touch(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n);
/** Called with every key the store removes (drop, release, eviction, an orphan
 *  pruned at open) -- the TP leader tells its workers to delete their copies --
 *  and (L347, `rp_only`) with every restore point removed apart from its segment.
 *  Set it before anything can be removed; NULL to clear. */
/** @param ud      the context given to pulsar_segstore_set_removed_hook
 *  @param key     the removed key, 40 hex chars
 *  @param rp_only only the key's restore point went (false: the segment and its restore point) */
typedef void (*pulsar_segstore_removed_fn)(void *ud, const char key[41], bool rp_only);
/** Set (or clear, fn NULL) the hook called with every key the store removes.
 *  @param st the store, or NULL (no-op)
 *  @param fn the hook
 *  @param ud context for `fn` */
void pulsar_segstore_set_removed_hook(pulsar_segstore *st, pulsar_segstore_removed_fn fn, void *ud);
/// @param st  the store
/// @param key 40 hex chars of the segment to remove with its subtree
/** Remove a segment and every segment below it, restore points included (a payload that would not load). */
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
/** Run `fn` over every segment that carries a trailer, in index order.  L353: the one call safe off the store's thread
 *  (a chat parse's): it copies the entries under a lock and reads the files without it, so one evicted meanwhile is
 *  skipped.
 *  @param st the store, or NULL (no-op)
 *  @param fn the visitor
 *  @param ud context for `fn` */
void pulsar_segstore_foreach_trailer(pulsar_segstore *st, pulsar_segstore_trailer_fn fn, void *ud);

/// @param st   the store
/// @param keys receives the malloc'd keys (NULL when there are none)
/// @return the number of keys
/** Every key, 40 hex chars each, concatenated (caller frees); the count returned. */
int pulsar_segstore_keys(pulsar_segstore *st, char **keys);
/** L347 I3b: hold `key` against eviction -- a restore reading its chain -- until as many
 *  pulsar_segstore_unpin.  The key need not be stored (an unpin of a dropped key only ends the count).
 *  @param st  the store, or NULL (no-op)
 *  @param key 40 hex chars */
void pulsar_segstore_pin(pulsar_segstore *st, const char key[41]);
/** L347 I3b: end one pulsar_segstore_pin.
 *  @param st  the store, or NULL (no-op)
 *  @param key 40 hex chars */
void pulsar_segstore_unpin(pulsar_segstore *st, const char key[41]);
/** L347 I3b: is `key` held (a restore's pin, a write in flight's parent, a restore point job's key)?
 *  @param st  the store, or NULL (false)
 *  @param key 40 hex chars
 *  @return true while held */
bool pulsar_segstore_is_pinned(const pulsar_segstore *st, const char key[41]);
/** L347 I3b: the PINNED anchors' budget: past it the least recently used loses its pin (logged) -- now, and whenever an
 *  anchor is pinned.  Counted: each anchor's restore point and every segment of its chain, once.
 *  @param st    the store, or NULL (no-op)
 *  @param bytes the budget (0 = unbounded) */
void pulsar_segstore_set_pinned_max(pulsar_segstore *st, uint64_t bytes);
/** L347 I3b: how many anchors are pinned, and the bytes they hold (as the budget counts them).
 *  @param st    the store, or NULL (0)
 *  @param bytes receives the bytes, or NULL
 *  @return the pinned anchors */
int pulsar_segstore_pinned_anchors(const pulsar_segstore *st, uint64_t *bytes);
/** L347 I3b: pins taken away, ever (over the budget, or the store out of room).
 *  @param st the store, or NULL (0)
 *  @return the count */
uint64_t pulsar_segstore_pins_released(const pulsar_segstore *st);
/// @param st the store, or NULL (0)
/// @return bytes of the .seg and .rp files the store holds
/** Bytes the store holds, and the count of segments. */
uint64_t pulsar_segstore_used_bytes(const pulsar_segstore *st);
/** L276: the byte budget the store was opened with (--kv-disk-space-mb).
 *  @param st the store, or NULL (0)
 *  @return bytes; 0 = unbounded */
uint64_t pulsar_segstore_budget_bytes(const pulsar_segstore *st);
/** L276: what the store has removed, ever: segments taken out of the index (evicted for room, released, dropped as
 *  bad) and, in `rps`, restore points removed without their segment.
 *  @param st  the store, or NULL (0)
 *  @param rps receives the restore points removed, or NULL
 *  @return the segments removed */
uint64_t pulsar_segstore_removed(const pulsar_segstore *st, uint64_t *rps);
/** L353 no. 16: whether a part read from disk stays as its RAM copy (pulsar_segstore_set_ram's retain): a second read
 *  of it then costs nothing.  @param st the store, or NULL  @return true when it does */
bool pulsar_segstore_retains(const pulsar_segstore *st);
/** L378: what the store wrote, served and removed since it opened -- the disk tier's ledger (bytes are file bytes:
 *  header, text, payload and trailer of a .seg; header and payload of a .rp). */
typedef struct {
    uint64_t segs_written;         ///< .seg files committed
    uint64_t rows_bytes_written;   ///< ... their bytes
    uint64_t rps_written;          ///< .rp files committed
    uint64_t rp_bytes_written;     ///< ... their bytes
    uint64_t segs_read;            ///< .seg parts mapped for a load (from the RAM tier or the disk), each map
    uint64_t rows_bytes_read;      ///< ... their file bytes
    uint64_t rps_read;             ///< .rp parts mapped for a load, each map
    uint64_t rp_bytes_read;        ///< ... their file bytes
    uint64_t bytes_removed;        ///< bytes of the files removed, any reason (evicted, dropped as bad, released, replaced)
    uint64_t budget_bytes_removed; ///< ... of them, evicted for room (the budget or a full disk)
    /// ... of bytes_removed, files no load had mapped since the store opened (pulsar_segstore_seg::rows_read / rp_read)
    uint64_t unread_bytes_removed;
} pulsar_segstore_io;
/** L378: read the store's I/O ledger.  @param st the store, or NULL (zeros)  @param out receives it */
void pulsar_segstore_io_stats(const pulsar_segstore *st, pulsar_segstore_io *out);
/** L353: files copied out whose write failed, ever.  @param st the store, or NULL  @return the count (0 for NULL) */
uint64_t pulsar_segstore_writes_failed(const pulsar_segstore *st);
/** L353: reads the disk's free bytes.  @param ud context  @return the bytes */
typedef uint64_t (*pulsar_segstore_disk_free_fn)(void *ud);
/** L353: read the disk's free bytes through `fn` in place of statvfs (a test's small disk).
 *  @param st the store, or NULL (no-op)  @param fn the reader (NULL = statvfs)  @param ud context for `fn` */
void pulsar_segstore_set_disk_free(pulsar_segstore *st, pulsar_segstore_disk_free_fn fn, void *ud);
/** The number of segments the store holds.
 *  @param st the store, or NULL (0)
 *  @return the segment count */
int pulsar_segstore_count(const pulsar_segstore *st);

#endif

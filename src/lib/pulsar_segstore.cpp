/** @file
 *  The content-addressed segment store's implementation (pulsar_segstore.h). */
/* L264 S4: the content-addressed segment store (pulsar_segstore.h). */
#include "pulsar_segstore.h"
#include "pulsar_kvdigest.h"
#include "pulsar_kvpool.h"
#include "sha1.hpp"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

/* Fixed header, little-endian:
 *   0 magic, 4 version, 8 identity, 12 flags, 16 key[40], 56 parent[40],
 *   96 G_prev, 100 G, 104 text_len u64, 112 payload_bytes u64,
 *   120 trailer_bytes u64, 128 hits u32, 132 pad, 136 created u64,
 *   144 last_used u64, 152 pad u64 (v2's state_bytes; 0 since v3, L347)  -> 160 bytes. */
/// Bytes of the fixed .seg header (layout above); the text starts at this offset.
constexpr size_t HDR = 160;
/* L347: a restore point file's header, little-endian:
 *   0 magic, 4 version, 8 identity, 12 flags (PULSAR_SEGSTORE_RP_*, v2), 16 key[40], 56 payload_bytes u64  -> 64
 *   bytes; the engine's restore point payload follows. */
/// Bytes of the fixed .rp header (layout above); the payload starts at this offset.
constexpr size_t RP_HDR = 64;
/** L347 I2: the bytes of a segment's text the index keeps (pulsar_segstore::heads), so pulsar_segstore_covers reads a
 *  segment's file only when they match: a store of one-shot prompts holds many root tails that share a template's
 *  first bytes and part soon after them. */
constexpr size_t HEAD_BYTES = 256;

/** Store v little-endian at p[0..4). @param p destination @param v value */
void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
/** Store v little-endian at p[0..8). @param p destination @param v value */
void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
/** @param p 4 little-endian bytes @return their value */
uint32_t get32(const uint8_t *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
/** @param p 8 little-endian bytes @return their value */
uint64_t get64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }

/** @param s at least 40 chars
 *  @return true when s[0..40) is all lowercase hex (a well-formed key) */
bool hex40(const char *s) {
    for (int i = 0; i < 40; i++) {
        const char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

}  // namespace

/** The open store: its directory and the in-memory index of every segment in it (keys, parents, lengths; no text). */
struct pulsar_segstore {
    /// The store's directory; segments are `<dir>/<key>.seg`
    std::string dir;
    /// Bytes of .seg and .rp files the store may hold (0 = unbounded)
    uint64_t budget = 0;
    /// The store identity (pulsar_segstore_identity) a header must carry
    uint32_t identity = 0;
    /// Log sink (NULL = silent)
    pulsar_segstore_log_fn log = nullptr;
    /// Context for `log`
    void *log_ud = nullptr;
    /// Called with every key removed (pulsar_segstore_set_removed_hook)
    pulsar_segstore_removed_fn removed = nullptr;
    /// Context for `removed`
    void *removed_ud = nullptr;
    /// Key -> index entry, every stored segment
    std::unordered_map<std::string, pulsar_segstore_seg> segs;
    /// L353: guards the shape of `segs` against pulsar_segstore_foreach_trailer, the one call made off the store's
    /// thread (a chat parse on a client thread): the store's thread holds it to add or erase an entry and reads `segs`
    /// without it; the walk holds it only to copy what it needs, never across a file read
    std::mutex segs_mu;
    std::unordered_map<std::string, std::vector<std::string>> children;   ///< parent key ("" = roots) -> child keys
    /// Sum of file_bytes and rp_bytes over segs, plus pending_rp: what the budget is checked against
    uint64_t used = 0;
    /// L338: file bytes of the reserved writes not yet committed (pulsar_segstore_reserve): the budget counts them
    uint64_t reserved = 0;
    /// L338: keys being written (reserved, not committed): a second reserve of one writes nothing
    std::unordered_set<std::string> pending;
    /// L338: parents of the writes in flight, the restore point jobs' keys and (L347 I3b) the chains being restored
    /// (pulsar_segstore_pin), with their count: eviction never takes one
    std::unordered_map<std::string, int> pinned;
    /** L347 I3a: a file's image in host memory -- the file's bytes from offset 0, padded to the O_DIRECT granule. */
    struct image {
        uint8_t *p = nullptr;   ///< the bytes (a pool block, or posix_memalign'd without a pool)
        uint64_t cap = 0;       ///< bytes allocated
    };
    /** L347: a restore point written for a pending key (its .seg not committed yet). */
    struct pending_point {
        uint64_t bytes = 0;   ///< its .rp file's bytes
        uint32_t flags = 0;   ///< its PULSAR_SEGSTORE_RP_* flags
        image img;            ///< L347 I3a: its image, kept as the entry's RAM copy at the segment's commit (retain)
    };
    /// L347: restore points written for pending keys, moved onto the entry at its commit (or removed with a failed one)
    std::unordered_map<std::string, pending_point> pending_rp;
    /// L347 I2: key -> the first HEAD_BYTES of its own text (all of it when shorter), for pulsar_segstore_covers
    std::unordered_map<std::string, std::string> heads;
    /// L347 I3a: the pool images come from (pulsar_segstore_set_ram); NULL: posix_memalign'd, never kept
    pulsar_kvpool *pool = nullptr;
    /// L347 I3a: the RAM tier: a written file's image stays as its entry's RAM copy (D9; off on a unified host)
    bool retain = false;
    /// L347 I3a: the RAM tier's residency, the one authority for it: key -> the image of its .seg
    std::unordered_map<std::string, image> ram_rows;
    /// L347 I3a: ... and key -> the image of its .rp
    std::unordered_map<std::string, image> ram_rp;
    /// L347 I3a: a .seg image read from disk for a load and not kept (no RAM tier), until pulsar_segstore_unmap_part
    std::unordered_map<std::string, image> loan_rows;
    /// L347 I3a: ... and a .rp image so read
    std::unordered_map<std::string, image> loan_rp;
    /// L347 I3a: parts mapped for a load (pulsar_segstore_map_part; hold_name), with their count: RAM eviction skips them
    std::unordered_map<std::string, int> held;
    /** @param key a key  @param rp its restore point (else its rows)  @return the part's name in `held` */
    static std::string hold_name(const std::string &key, bool rp) { return rp ? key + ".rp" : key; }
    /// L347 I3a: restore point writes in flight (pulsar_segstore_reserve_rp): key -> flags joined while it flies
    std::unordered_map<std::string, uint32_t> rp_jobs;
    /// L347 I3b: the PULSAR_SEGSTORE_RP_PINNED anchors' budget, bytes (pulsar_segstore_set_pinned_max; 0 = unbounded)
    uint64_t pinned_max = 0;
    /// L347 I3b: pins taken away, ever: over the budget, or the store out of room (pulsar_segstore_pins_released)
    uint64_t pins_released = 0;
    /// L276: segments taken out of the index, ever: evicted, released, dropped as bad (pulsar_segstore_removed)
    uint64_t segs_removed = 0;
    /// L353: files whose write failed (pulsar_segstore_commit with written false)
    uint64_t writes_failed = 0;
    /// L353: reads the disk's free bytes in place of statvfs (pulsar_segstore_set_disk_free; a test's)
    pulsar_segstore_disk_free_fn disk_free_fn = nullptr;
    /// Context for `disk_free_fn`
    void *disk_free_ud = nullptr;
    /// L276: restore points removed without their segment, ever (pulsar_segstore_removed)
    uint64_t rps_removed = 0;

    /** printf-style message to `log` (truncated to 511 bytes); no-op without a sink.
     *  @param fmt printf format */
    void logf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
        if (!log) return;
        char msg[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        log(log_ud, msg);
    }
    /** @param key a segment key
     *  @return the segment's file path, `<dir>/<key>.seg` */
    std::string path(const char *key) const { return dir + "/" + key + ".seg"; }
    /** @param key a segment key
     *  @return its restore point's file path, `<dir>/<key>.rp` (L347) */
    std::string rp_path(const char *key) const { return dir + "/" + key + ".rp"; }

    /** Read and validate a .seg header from fp's current position into *e (file_bytes, text_end and dominated are
     *  left 0; the caller derives them).
     *  @param fp stream at the header
     *  @param e receives the index entry
     *  @param flags receives the header flags (PULSAR_SEGSTORE_FLAG_*)
     *  @return false on a short read, another magic / version / identity, a malformed key, or an empty span */
    bool read_header(FILE *fp, pulsar_segstore_seg *e, uint32_t *flags) const {
        uint8_t h[HDR];
        return fread(h, 1, HDR, fp) == HDR && parse_header(h, e, flags);
    }
    /** L347 I3a: validate a .seg header in memory into *e (as read_header).
     *  @param h     HDR bytes
     *  @param e     receives the index entry
     *  @param flags receives the header flags
     *  @return false on another magic / version / identity, a malformed key, or an empty span */
    bool parse_header(const uint8_t *h, pulsar_segstore_seg *e, uint32_t *flags) const {
        if (get32(h) != PULSAR_SEGSTORE_MAGIC || get32(h + 4) != PULSAR_SEGSTORE_VERSION) return false;
        if (get32(h + 8) != identity) return false;
        memset(e, 0, sizeof *e);
        *flags = get32(h + 12);
        memcpy(e->key, h + 16, 40);
        memcpy(e->parent, h + 56, 40);
        if (!hex40(e->key)) return false;
        if (e->parent[0] && !hex40(e->parent)) return false;
        e->G_prev = get32(h + 96);
        e->G = get32(h + 100);
        e->text_len = get64(h + 104);
        e->payload_bytes = get64(h + 112);
        e->trailer_bytes = (*flags & PULSAR_SEGSTORE_FLAG_TRAILER) ? get64(h + 120) : 0u;
        e->hits = get32(h + 128);
        e->last_used = get64(h + 144);
        return e->G > e->G_prev;
    }
    /** L347: read and validate a .rp header from fp's start.
     *  @param fp            the open .rp file
     *  @param key           receives the key it names (40 hex + NUL)
     *  @param payload_bytes receives the payload's size
     *  @param flags         receives its PULSAR_SEGSTORE_RP_* flags
     *  @param older         receives true when it is a restore point of an older store version
     *  @return false on a short read, another magic / version / identity, or a malformed key */
    bool read_rp_header(FILE *fp, char key[41], uint64_t *payload_bytes, uint32_t *flags, bool *older) const {
        uint8_t h[RP_HDR];
        *older = false;
        return fread(h, 1, RP_HDR, fp) == RP_HDR && parse_rp_header(h, key, payload_bytes, flags, older);
    }
    /** L347 I3a: validate a .rp header in memory (as read_rp_header).
     *  @param h             RP_HDR bytes
     *  @param key           receives the key it names (40 hex + NUL)
     *  @param payload_bytes receives the payload's size
     *  @param flags         receives its PULSAR_SEGSTORE_RP_* flags
     *  @param older         receives true when it is a restore point of an older store version
     *  @return false on another magic / version / identity, or a malformed key */
    bool parse_rp_header(const uint8_t *h, char key[41], uint64_t *payload_bytes, uint32_t *flags, bool *older) const {
        *older = false;
        if (get32(h) != PULSAR_SEGSTORE_RP_MAGIC) return false;
        if (get32(h + 4) != PULSAR_SEGSTORE_RP_VERSION) {
            *older = get32(h + 4) < PULSAR_SEGSTORE_RP_VERSION;
            return false;
        }
        if (get32(h + 8) != identity) return false;
        memcpy(key, h + 16, 40);
        key[40] = '\0';
        *flags = get32(h + 12);
        *payload_bytes = get64(h + 56);
        return hex40(key);
    }
    /** L347 I3a: an image of `bytes` file bytes: a pool block (RAM images evicted to make room) or, without a pool,
     *  posix_memalign'd.  Always one byte past the file (a memory stream's closing NUL, pulsar_segstore_put) and
     *  padded to the granule, the pad zeroed by the caller.
     *  @param bytes the file's bytes
     *  @return the image (p NULL when there is no room) */
    image img_alloc(uint64_t bytes) {
        image im;
        im.cap = pulsar_kvpool_round(bytes + 1u);
        if (pool) {
            if (ram_make_room(im.cap)) im.p = (uint8_t *)pulsar_kvpool_alloc(pool, im.cap);
        } else {
            void *q = nullptr;
            if (posix_memalign(&q, PULSAR_KVPOOL_ALIGN, (size_t)im.cap) == 0) im.p = (uint8_t *)q;
        }
        if (!im.p) im.cap = 0;
        return im;
    }
    /** L347 I3a: free an image (a no-op for an empty one).  @param im the image */
    void img_free(const image &im) {
        if (!im.p) return;
        if (pool) pulsar_kvpool_free(pool, im.p);
        else free(im.p);
    }
    /** L347 I3a: a written file's image becomes its entry's RAM copy (retain), or is freed.
     *  @param map ram_rows or ram_rp
     *  @param key the entry
     *  @param im  the image */
    void img_keep(std::unordered_map<std::string, image> &map, const std::string &key, const image &im) {
        img_drop(map, key);   /* a replaced restore point's copy */
        if (!retain) { img_free(im); return; }
        map[key] = im;
    }
    /** L347 I3a: drop a key's RAM copy in `map`, if any (its file stays).  @param map ram_rows or ram_rp
     *  @param key the entry */
    void img_drop(std::unordered_map<std::string, image> &map, const std::string &key) {
        auto it = map.find(key);
        if (it == map.end()) return;
        img_free(it->second);
        map.erase(it);
    }
    /* L347 I3a (D9): RAM eviction in the disk eviction's value order -- (0) an unflagged restore point off a branch
     * point, (1) a dominated leaf's rows, (2) any other rows, (3) a flagged or branch-point restore point -- the least
     * recently used first within a class.  A copy that goes stays on disk: an entry is indexed only once its file is
     * written, so dropping its image loses nothing.  Never an image mapped for a load. */
    /* L347 I3b: a PINNED anchor's restore point and its chain's rows are never evicted from RAM while anything else
     * may go -- only when every other copy is gone (the pool smaller than the pinned budget), and then only the RAM
     * copy (the files stay, the pins too). */
    /** Free one RAM copy by the order above.  @return false when none may go */
    bool ram_evict_one() {
        const std::unordered_set<std::string> anchored = pinned_chain_keys();
        for (int pass = 0; pass < 2; pass++)
            if (ram_evict_one_pass(anchored, pass == 1)) return true;
        return false;
    }
    /** One pass of ram_evict_one.  @param anchored pinned_chain_keys()  @param anchors_too a pinned anchor's copies may
     *  go too  @return true when one copy went */
    bool ram_evict_one_pass(const std::unordered_set<std::string> &anchored, bool anchors_too) {
        const std::string *victim = nullptr;
        bool victim_rp = false;
        int victim_class = 4;
        uint64_t victim_used = 0;
        auto consider = [&](const std::string &k, bool rp) {
            if (held.count(hold_name(k, rp))) return;
            auto it = segs.find(k);
            if (it == segs.end()) return;
            if (!anchors_too && anchored.count(k) &&
                (!rp || (it->second.rp_flags & PULSAR_SEGSTORE_RP_PINNED) != 0u))
                return;
            const size_t kids = n_children(k);
            const int c = rp ? (it->second.rp_flags == 0u && kids < 2u ? 0 : 3)
                             : (it->second.dominated && kids == 0u ? 1 : 2);
            if (!victim || c < victim_class || (c == victim_class && it->second.last_used < victim_used)) {
                victim = &it->first;
                victim_rp = rp;
                victim_class = c;
                victim_used = it->second.last_used;
            }
        };
        for (auto &kv : ram_rp) consider(kv.first, true);
        for (auto &kv : ram_rows) consider(kv.first, false);
        if (!victim) return false;
        const std::string k = *victim;
        img_drop(victim_rp ? ram_rp : ram_rows, k);
        return true;
    }
    /** L347 I3a: make a free block of `bytes` in the pool by evicting RAM copies (true without a pool).
     *  @param bytes the block wanted
     *  @return true when the pool can hand out such a block now */
    bool ram_make_room(uint64_t bytes) {
        if (!pool) return true;
        const uint64_t want = pulsar_kvpool_round(bytes);
        while (pulsar_kvpool_largest_free(pool) < want)
            if (!ram_evict_one()) return false;
        return true;
    }
    /** Remove a stored segment's files, its .seg and its .rp (L347), from disk (not from the index).
     *  @param key the segment */
    void unlink_files(const std::string &key) const {
        ::unlink(path(key.c_str()).c_str());
        ::unlink(rp_path(key.c_str()).c_str());
    }

    /** Add e to the index: segs, its parent's children, used. Does not refresh dominance.
     *  @param e the entry (file_bytes set) */
    void link(const pulsar_segstore_seg &e) {
        {
            std::lock_guard<std::mutex> lk(segs_mu);
            segs[e.key] = e;
        }
        children[e.parent].push_back(e.key);
        used += e.file_bytes + e.rp_bytes;
    }
    /** Remove key from the index (not from disk): fires the removed hook, drops it from its parent's children, and
     *  re-derives the siblings' dominance.  A no-op for a key not stored.
     *  @param key the segment */
    void unlink_index(const std::string &key) {
        auto it = segs.find(key);
        if (it == segs.end()) return;
        if (removed) removed(removed_ud, key.c_str(), false);
        segs_removed++;   /* L276 */
        used -= it->second.file_bytes + it->second.rp_bytes;
        const std::string parent = it->second.parent;
        auto &sib = children[parent];
        for (size_t i = 0; i < sib.size(); i++)
            if (sib[i] == key) { sib.erase(sib.begin() + (long)i); break; }
        children.erase(key);
        heads.erase(key);
        img_drop(ram_rows, key);   /* L347 I3a: its RAM copies go with it */
        img_drop(ram_rp, key);
        {
            std::lock_guard<std::mutex> lk(segs_mu);
            segs.erase(it);
        }
        if (parent.empty() || segs.count(parent)) refresh_dominated(parent);
    }
    /** Read a stored segment's own text from its file.
     *  @param key the segment
     *  @param out receives text_len bytes
     *  @return false when the key is not stored or the read fails */
    bool read_text(const std::string &key, std::string *out) const {
        auto it = segs.find(key);
        if (it == segs.end()) return false;
        FILE *fp = fopen(path(key.c_str()).c_str(), "rb");
        if (!fp) return false;
        out->assign(it->second.text_len, '\0');
        const bool ok = fseek(fp, (long)HDR, SEEK_SET) == 0 &&
                        fread(out->empty() ? nullptr : &(*out)[0], 1, out->size(), fp) == out->size();
        fclose(fp);
        return ok;
    }
    /* L323: the ONE dominance rule.  A child of `parent` is dominated when its text is a strict byte prefix of a
     * sibling's: the conversation went on past it under the same parent (a later tail, or the grid segment that
     * covers it), so every request it serves to its end a sibling serves too -- only one that diverges inside the
     * sibling's stretch past it resumes deeper from it.  Re-derived whenever the parent's children change. */
    /** Recompute `dominated` for every child of parent (reads the children's texts from disk).
     *  @param parent the parent key ("" = the roots) */
    void refresh_dominated(const std::string &parent) {
        auto ch = children.find(parent);
        if (ch == children.end()) return;
        const std::vector<std::string> &kids = ch->second;
        std::vector<std::string> texts(kids.size());
        std::vector<bool> have(kids.size(), false);
        if (kids.size() > 1)
            for (size_t i = 0; i < kids.size(); i++) have[i] = read_text(kids[i], &texts[i]);
        for (size_t i = 0; i < kids.size(); i++) {
            bool dom = false;
            for (size_t k = 0; have[i] && !dom && k < kids.size(); k++)
                dom = k != i && have[k] && texts[k].size() > texts[i].size() &&
                      !memcmp(texts[k].data(), texts[i].data(), texts[i].size());
            segs[kids[i]].dominated = dom;
        }
    }
    /* A segment whose parent is gone cannot be reached by any lookup. */
    /** Delete (file and index) every segment whose parent is not stored, until none is left. */
    void prune_orphans() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = segs.begin(); it != segs.end(); ++it) {
                if (it->second.parent[0] && !segs.count(it->second.parent)) {
                    const std::string k = it->first;
                    logf("segment %s dropped: its parent %s is gone", k.c_str(), it->second.parent);
                    unlink_files(k);
                    unlink_index(k);
                    changed = true;
                    break;
                }
            }
        }
    }
    /* The chain text lengths are a property of the chain, derived once. */
    /** Set every reachable segment's text_end by walking down from the roots. */
    void derive_text_ends() {
        std::vector<std::string> stack;
        for (auto &k : children[""]) { segs[k].text_end = segs[k].text_len; stack.push_back(k); }
        while (!stack.empty()) {
            const std::string k = stack.back();
            stack.pop_back();
            for (auto &c : children[k]) {
                segs[c].text_end = segs[k].text_end + segs[c].text_len;
                stack.push_back(c);
            }
        }
    }
    /** L347: remove a stored segment's restore point (file and index); the segment stays.
     *  @param e   the segment's index entry (rp_bytes > 0)
     *  @param why the log line's reason */
    void drop_rp_of(pulsar_segstore_seg &e, const char *why) {
        logf("dropped restore point %s at %u, %llu bytes: %s", e.key, e.G, (unsigned long long)e.rp_bytes, why);
        ::unlink(rp_path(e.key).c_str());
        img_drop(ram_rp, e.key);   /* L347 I3a */
        used -= e.rp_bytes;
        e.rp_bytes = 0;
        e.rp_flags = 0;
        rps_removed++;   /* L276 */
        if (removed) removed(removed_ud, e.key, true);
    }
    /** @param key a stored key
     *  @return how many children the index holds for it */
    size_t n_children(const std::string &key) const {
        auto ch = children.find(key);
        return ch == children.end() ? 0u : ch->second.size();
    }
    /* L347 I2 (D5), L353: the eviction classes, lowest first -- (0) a dominated leaf (L323) and (1) any other leaf without
     * a restore point: rows that restore nothing past the chain's last restore point, and the next persist of a live
     * bank rewrites any it needs; (2) a restore point inside a chain, neither flagged nor on a branch point: past
     * classes 0 and 1 every leaf has a restore point, so a deeper one serves its chain; (3) a chain end's restore point:
     * where its conversation resumes, its rows useless without it; (4) a flagged (anchor, periodic) or branch-point
     * restore point: where restores of many requests start.  -1: not a candidate (a segment with children and no
     * restore point; L347 I3b: a PINNED anchor's restore point, and so its rows).  Until L353 class (0) was every
     * unflagged restore point, so a full store drained its conversations' resume points and kept their dead rows. */
    /** @param e      an index entry
     *  @param leaf   it has no children
     *  @param branch it has >= 2 children
     *  @return its eviction class (above), -1 for none */
    static int evict_class(const pulsar_segstore_seg &e, bool leaf, bool branch) {
        if (e.rp_bytes > 0u && (e.rp_flags & PULSAR_SEGSTORE_RP_PINNED) != 0u) return -1;   /* L347 I3b: pinned */
        if (e.rp_bytes > 0u) return e.rp_flags != 0u || branch ? 4 : leaf ? 3 : 2;
        if (!leaf) return -1;
        return e.dominated ? 0 : 1;
    }
    /** Evict one restore point (file and index), or one leaf (files and index), by the classes above: the lowest class
     *  first, the least recently used within it.  Never `keep` or `keep2`, a parent of a write in flight or a key being
     *  written (L338).
     *  @param keep a key to spare (NULL = none)
     *  @param keep2 a second key to spare (NULL = none)
     *  @return false when nothing may go */
    bool evict_one(const char *keep, const char *keep2) {
        const pulsar_segstore_seg *victim = nullptr;
        int victim_class = 5;
        for (auto &kv : segs) {
            if ((keep && kv.first == keep) || (keep2 && kv.first == keep2)) continue;
            if (pinned.count(kv.first) || pending.count(kv.first)) continue;   /* L338: a write in flight */
            const size_t kids = n_children(kv.first);
            const int c = evict_class(kv.second, kids == 0u, kids >= 2u);
            if (c < 0) continue;
            if (!victim || c < victim_class || (c == victim_class && kv.second.last_used < victim->last_used)) {
                victim = &kv.second;
                victim_class = c;
            }
        }
        /* L347 I3b: nothing else may go -- the oldest-used pinned anchor loses its pin (and goes by the classes next) */
        if (!victim) return unpin_oldest("the store needs room and nothing else may go", keep, keep2);
        const std::string k = victim->key;
        if (victim->rp_bytes > 0u) {
            drop_rp_of(segs[k], victim_class < 4 ? "store over budget"
                                                 : "store over budget (a flagged or branch point's)");
            return true;
        }
        logf("evicted segment %s [%u,%u) %llu bytes%s: store over budget", k.c_str(), victim->G_prev, victim->G,
             (unsigned long long)victim->file_bytes, victim->dominated ? " (dominated)" : "");
        unlink_files(k);
        unlink_index(k);
        return true;
    }
    /** L347 I3b: set a stored restore point's flags, in its index entry, its file's header (in place) and its RAM copy.
     *  @param e     the entry (rp_bytes > 0)
     *  @param flags the PULSAR_SEGSTORE_RP_* flags it carries from now
     *  @return false when the header could not be written (nothing changed) */
    bool write_rp_flags(pulsar_segstore_seg &e, uint32_t flags) {
        if (flags == e.rp_flags) return true;
        FILE *fp = fopen(rp_path(e.key).c_str(), "r+b");
        if (!fp) return false;
        uint8_t b[4];
        put32(b, flags);
        const bool ok = fseek(fp, 12, SEEK_SET) == 0 && fwrite(b, 1, 4, fp) == 4;
        const bool closed = fclose(fp) == 0;
        if (!ok || !closed) return false;
        e.rp_flags = flags;
        auto ram = ram_rp.find(e.key);   /* L347 I3a: its RAM copy's header too */
        if (ram != ram_rp.end()) memcpy(ram->second.p + 12, b, 4);
        return true;
    }
    /** L347 I3b: every key a PINNED anchor holds: the anchor's key and its ancestors (its chain's rows).
     *  @return the keys */
    std::unordered_set<std::string> pinned_chain_keys() const {
        std::unordered_set<std::string> keys;
        for (auto &kv : segs) {
            if (!(kv.second.rp_bytes > 0u && (kv.second.rp_flags & PULSAR_SEGSTORE_RP_PINNED))) continue;
            for (std::string k = kv.first; !k.empty();) {
                auto it = segs.find(k);
                if (it == segs.end() || !keys.insert(k).second) break;   /* a chain met already holds the rest */
                k = it->second.parent;
            }
        }
        return keys;
    }
    /** L347 I3b: the bytes the PINNED anchors hold on disk: their restore points and every segment of their chains
     *  (each once).  @param count receives how many anchors are pinned (NULL: not wanted)  @return the bytes */
    uint64_t pinned_bytes(int *count) const {
        uint64_t bytes = 0;
        int n = 0;
        for (const std::string &k : pinned_chain_keys()) {
            const pulsar_segstore_seg &e = segs.at(k);
            bytes += e.file_bytes;
            if (e.rp_bytes > 0u && (e.rp_flags & PULSAR_SEGSTORE_RP_PINNED)) { bytes += e.rp_bytes; n++; }
        }
        if (count) *count = n;
        return bytes;
    }
    /** L347 I3b: take the pin from the least recently used PINNED anchor (never `keep` / `keep2`), said in the log.
     *  @param why   the log's reason
     *  @param keep  a key to spare (NULL = none)
     *  @param keep2 a second key to spare (NULL = none)
     *  @return false when no anchor is pinned (or its header could not be written) */
    bool unpin_oldest(const char *why, const char *keep, const char *keep2) {
        pulsar_segstore_seg *victim = nullptr;
        for (auto &kv : segs) {
            if ((keep && kv.first == keep) || (keep2 && kv.first == keep2)) continue;
            if (!(kv.second.rp_bytes > 0u && (kv.second.rp_flags & PULSAR_SEGSTORE_RP_PINNED))) continue;
            if (!victim || kv.second.last_used < victim->last_used) victim = &kv.second;
        }
        if (!victim || !write_rp_flags(*victim, victim->rp_flags & ~PULSAR_SEGSTORE_RP_PINNED)) return false;
        pins_released++;
        logf("unpinned anchor %s at %u (%llu bytes, last used %llu): %s", victim->key, victim->G,
             (unsigned long long)victim->rp_bytes, (unsigned long long)victim->last_used, why);
        return true;
    }
    /** L347 I3b: hold the pinned anchors to pinned_max: while they hold more, the least recently used loses its pin. */
    void enforce_pinned_max() {
        while (pinned_max && pinned_bytes(nullptr) > pinned_max)
            if (!unpin_oldest("the pinned anchors are over their budget (--kv-pinned-max-mb)", nullptr, nullptr))
                break;
    }
};

pulsar_segstore *pulsar_segstore_open(const char *dir, uint64_t budget_bytes, uint32_t identity,
                                      pulsar_segstore_log_fn log, void *log_ud) {
    if (!dir || !dir[0]) return nullptr;
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return nullptr;
    /* Refuse a directory this process cannot write up front, instead of
     * failing every store later.  L347 I3a: every file is written and read with
     * O_DIRECT (the one I/O path): a filesystem that refuses it refuses the store. */
    {
        char probe[4096];
        snprintf(probe, sizeof probe, "%s/.probe.%ld", dir, (long)getpid());
        const int fd = open(probe, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
        const int saved = errno;
        if (fd >= 0) close(fd);
        ::unlink(probe);
        if (fd < 0) {
            if (log) {
                char msg[512];
                snprintf(msg, sizeof msg, "segment store %s refused: it cannot be written with O_DIRECT (%s)", dir,
                         strerror(saved));
                log(log_ud, msg);
            }
            return nullptr;
        }
    }
    auto *st = new pulsar_segstore();
    st->dir = dir;
    st->budget = budget_bytes;
    st->identity = identity;
    st->log = log;
    st->log_ud = log_ud;
    DIR *d = opendir(dir);
    if (!d) { delete st; return nullptr; }
    uint64_t retired = 0, retired_bytes = 0;
    std::vector<std::string> rps;   /* L347: the .rp files, indexed once every .seg is */
    while (const struct dirent *de = readdir(d)) {
        const size_t n = strlen(de->d_name);
        /* A writer that died between tmp and rename leaves "<key>.seg.tmp.<pid>" (or "<key>.rp.tmp.<pid>"):
         * no live writer exists at open, so every such file is an orphan. */
        if (strstr(de->d_name, ".seg.tmp.") || strstr(de->d_name, ".rp.tmp.")) {
            ::unlink((st->dir + "/" + de->d_name).c_str());
            continue;
        }
        if (n == 43 && strcmp(de->d_name + 40, ".rp") == 0 && hex40(de->d_name)) {
            rps.push_back(std::string(de->d_name, 40));
            continue;
        }
        if (n != 44 || strcmp(de->d_name + 40, ".seg") != 0 || !hex40(de->d_name)) continue;
        const std::string p = st->dir + "/" + de->d_name;
        FILE *fp = fopen(p.c_str(), "rb");
        if (!fp) continue;
        pulsar_segstore_seg e;
        uint32_t flags = 0;
        const bool ok = st->read_header(fp, &e, &flags) && !memcmp(e.key, de->d_name, 40);
        struct stat sb;
        const bool stat_ok = fstat(fileno(fp), &sb) == 0;
        const bool sized = ok && stat_ok && (uint64_t)sb.st_size == HDR + e.text_len + e.payload_bytes + e.trailer_bytes;
        std::string head(sized ? (e.text_len < HEAD_BYTES ? e.text_len : HEAD_BYTES) : 0u, '\0');
        const bool head_ok = head.empty() || fread(&head[0], 1, head.size(), fp) == head.size();
        /* L323: a segment of an OLDER store version can never load again (on-disk KV carries no back-compat, Tyler
         * 2026-10-13) and its bytes sit outside the budget: removed, whatever model wrote it */
        bool older = false;
        if (!ok) {
            uint8_t h[8];
            older = fseek(fp, 0, SEEK_SET) == 0 && fread(h, 1, 8, fp) == 8 && get32(h) == PULSAR_SEGSTORE_MAGIC &&
                    get32(h + 4) < PULSAR_SEGSTORE_VERSION;
        }
        fclose(fp);
        if (older && ::unlink(p.c_str()) == 0) {
            retired++;
            retired_bytes += stat_ok ? (uint64_t)sb.st_size : 0u;
        }
        if (!sized || !head_ok) continue;
        e.file_bytes = (uint64_t)sb.st_size;
        st->link(e);
        st->heads[e.key] = head;
    }
    closedir(d);
    if (retired)
        st->logf("segment store %s: removed %llu segments of an older store version (%llu MiB)", dir,
                 (unsigned long long)retired, (unsigned long long)(retired_bytes >> 20));
    /* L347: a restore point joins its indexed segment; one whose .seg file is gone, of an older store version, or not
     * readable beside an indexed segment can never load and is removed; one beside another identity's (or an
     * unreadable) segment is left alone, as that segment is */
    int rp_removed = 0;
    for (const std::string &k : rps) {
        const std::string p = st->rp_path(k.c_str());
        FILE *fp = fopen(p.c_str(), "rb");
        if (!fp) continue;
        char key[41];
        uint64_t payload = 0;
        uint32_t flags = 0;
        bool older = false;
        const bool ok = st->read_rp_header(fp, key, &payload, &flags, &older) && !memcmp(key, k.data(), 40);
        struct stat sb;
        const bool sized = ok && fstat(fileno(fp), &sb) == 0 && (uint64_t)sb.st_size == RP_HDR + payload;
        fclose(fp);
        auto it = st->segs.find(k);
        if (sized && it != st->segs.end()) {
            it->second.rp_bytes = (uint64_t)sb.st_size;
            it->second.rp_flags = flags;
            st->used += it->second.rp_bytes;
            continue;
        }
        struct stat seg_sb;
        const bool doomed = older || it != st->segs.end() || stat(st->path(k.c_str()).c_str(), &seg_sb) != 0;
        if (doomed && ::unlink(p.c_str()) == 0) rp_removed++;
    }
    if (rp_removed) st->logf("segment store %s: removed %d restore points that cannot load", dir, rp_removed);
    st->prune_orphans();
    st->derive_text_ends();
    {
        std::vector<std::string> parents;
        for (auto &kv : st->children) if (kv.second.size() > 1) parents.push_back(kv.first);
        for (auto &p : parents) st->refresh_dominated(p);
    }
    {
        size_t n_rp = 0;
        for (auto &kv : st->segs) n_rp += kv.second.rp_bytes > 0u;
        st->logf("segment store %s: %zu segments, %zu restore points, %llu MiB (budget %llu MiB)", dir, st->segs.size(),
                 n_rp, (unsigned long long)(st->used >> 20), (unsigned long long)(budget_bytes >> 20));
    }
    return st;
}

void pulsar_segstore_close(pulsar_segstore *st) {
    if (!st) return;
    /* L347 I3a: the RAM copies and any loaned image go back to the pool (no job may be out: the caller drained its
     * writer) */
    for (auto *m : {&st->ram_rows, &st->ram_rp, &st->loan_rows, &st->loan_rp})
        for (auto &kv : *m) st->img_free(kv.second);
    for (auto &kv : st->pending_rp) st->img_free(kv.second.img);
    delete st;
}

void pulsar_segstore_set_ram(pulsar_segstore *st, pulsar_kvpool *pool, bool retain) {
    if (!st) return;
    st->pool = pool;
    st->retain = pool != nullptr && retain;
}

void pulsar_segstore_child_key_images(const char *parent, const char *text, size_t len, const uint64_t *hashes,
                                      uint32_t n, char out[41]) {
    pulsar::Sha1 h;
    if (parent && parent[0]) h.update(parent, 40);
    h.update(text, len);
    for (uint32_t i = 0; i < n; i++) h.update(&hashes[i], sizeof hashes[i]);   /* L353 */
    uint8_t d[20];
    h.final(d);
    pulsar::Sha1::hex20(d, out);
}

void pulsar_segstore_child_key(const char *parent, const char *text, size_t len, char out[41]) {
    pulsar_segstore_child_key_images(parent, text, len, nullptr, 0u, out);
}

int pulsar_segstore_lookup(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                           pulsar_segstore_seg *chain, int cap) {
    return pulsar_segstore_lookup_images(st, text, len, want, nullptr, 0u, chain, cap);
}

int pulsar_segstore_lookup_images(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                                  const uint64_t *hashes, uint32_t n, pulsar_segstore_seg *chain, int cap) {
    if (!st || !text || cap <= 0) return 0;
    /* Depth-first over the children whose key the request's own bytes
     * reproduce; the deepest accepted end wins (by G, then by text). */
    struct frame { std::string key; uint64_t end; int depth; uint32_t used; };   /* used: image hashes consumed (L353) */
    std::vector<frame> stack;
    std::vector<std::string> path(cap), best_path;
    int best_depth = 0;
    uint32_t best_G = 0;
    uint64_t best_end = 0;
    stack.push_back({"", 0, 0, 0u});
    while (!stack.empty()) {
        const frame f = stack.back();
        stack.pop_back();
        if (f.depth > 0) path[f.depth - 1] = f.key;
        const pulsar_segstore_seg *s = f.depth > 0 ? &st->segs[f.key] : nullptr;
        const bool accepted = s && (!want || ((want->grid == 0u || s->G % want->grid == 0u) &&
                                              (!want->state || s->rp_bytes > 0u)));
        if (accepted && (s->G > best_G || (s->G == best_G && f.end > best_end))) {
            best_depth = f.depth;
            best_G = s->G;
            best_end = f.end;
            best_path.assign(path.begin(), path.begin() + f.depth);
        }
        if (f.depth >= cap) continue;
        auto ch = st->children.find(f.key);
        if (ch == st->children.end()) continue;
        for (const std::string &c : ch->second) {
            const pulsar_segstore_seg &cs = st->segs[c];
            if (f.end + cs.text_len > len) continue;
            /* L353: the span holds some run of the request's next images (none for a text span): the key says which */
            for (uint32_t m = 0; f.used + m <= n; m++) {
                char k[41];
                pulsar_segstore_child_key_images(f.key.c_str(), text + f.end, cs.text_len,
                                                 hashes ? hashes + f.used : nullptr, m, k);
                if (c == k) {
                    stack.push_back({c, f.end + cs.text_len, f.depth + 1, f.used + m});
                    break;
                }
            }
        }
    }
    for (int i = 0; i < best_depth; i++) chain[i] = st->segs[best_path[i]];
    return best_depth;
}

bool pulsar_segstore_contains(pulsar_segstore *st, const char key[41]) {
    return st && st->segs.count(std::string(key, 40)) != 0;
}

bool pulsar_segstore_get(pulsar_segstore *st, const char key[41], pulsar_segstore_seg *out) {
    if (!st) return false;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end()) return false;
    if (out) *out = it->second;
    return true;
}

/** Run one writer into a memory stream and check it wrote exactly what it promised.
 *  @param fp     the stream, positioned where the bytes go
 *  @param fn     the writer
 *  @param ud     its context
 *  @param bytes  the bytes it promised
 *  @param what   names the part in a failure
 *  @param err    receives a reason on failure
 *  @param errlen size of `err`
 *  @return 0, or 1 on a failed or mis-sized write */
static int write_part(FILE *fp, pulsar_segstore_write_fn fn, void *ud, uint64_t bytes, const char *what, char *err,
                      size_t errlen) {
    const long at = ftell(fp);
    if (fn(fp, ud, err, errlen) != 0) return 1;
    if (fflush(fp) != 0 || (uint64_t)(ftell(fp) - at) != bytes) {
        snprintf(err, errlen, "segment put: %s wrote %ld bytes, promised %llu", what, ftell(fp) - at,
                 (unsigned long long)bytes);
        return 1;
    }
    return 0;
}

/** Fill `bytes` of an image through a writer (a memory stream over them, one byte past: glibc's fmemopen writes a NUL
 *  there on every flush -- the image always has that byte, pulsar_segstore::img_alloc).
 *  @param at     where the bytes go
 *  @param bytes  how many
 *  @param fn     the writer
 *  @param ud     its context
 *  @param what   names the part in a failure
 *  @param err    receives a reason on failure
 *  @param errlen size of `err`
 *  @return 0, or 1 */
static int fill_part(uint8_t *at, uint64_t bytes, pulsar_segstore_write_fn fn, void *ud, const char *what, char *err,
                     size_t errlen) {
    FILE *m = fmemopen(at, (size_t)bytes + 1u, "wb");
    if (!m) {
        snprintf(err, errlen, "segment put: %s: no memory stream", what);
        return 1;
    }
    const int rc = write_part(m, fn, ud, bytes, what, err, errlen);
    fclose(m);
    return rc;
}

/** L338 / L347 I3a: a put in three steps, so the file write can leave the thread that owns the index.  The job
 * carries everything the write needs -- the file's whole IMAGE in host memory (header, text, payload, trailer, padded
 * to the O_DIRECT granule) -- so no store state is read while it runs; the store keeps the job's key PENDING (a second
 * reserve of it writes nothing) and its parent PINNED (eviction takes leaves only, and a pending child is not yet a
 * child in the index) until the commit.  A restore point job (pulsar_segstore_reserve_rp) pins its own key. */
struct pulsar_segstore_job {
    int kind;                ///< PULSAR_SEGSTORE_PART_ROWS (a .seg) or _RP (a .rp)
    char key[41];            ///< the segment's key
    char parent[41];         ///< its parent's key ("" for a root; rows only)
    uint32_t G_prev;         ///< the span's start (rows only)
    uint32_t G;              ///< the span's end (rows only)
    uint64_t text_len;       ///< the segment's own text bytes (rows only)
    uint64_t payload_off;    ///< where the payload starts in the image
    uint64_t payload_bytes;  ///< the payload's bytes (its last 8 are its digest, sealed by job_write)
    uint64_t trailer_bytes;  ///< the trailer's bytes (0 = none; rows only)
    uint32_t flags;          ///< the restore point's PULSAR_SEGSTORE_RP_* flags as its header carries them (rp only)
    uint64_t need;           ///< the file's bytes
    pulsar_segstore::image img;   ///< the file's image
    std::string final_path;  ///< `<dir>/<key>.seg` or `.rp`
};

/** Room in the budget for `need` more bytes, evicting by the classes (never `keep` / `keep2`).
 *  @param st    the store
 *  @param need  the bytes
 *  @param freed bytes the write releases when it commits (a replaced restore point)
 *  @param keep  a key to spare
 *  @param keep2 a second key to spare (NULL = none)
 *  @return true when they fit */
/** L353: the disk's free bytes for the store, or UINT64_MAX when unreadable.  @param st the store
 *  @return the bytes */
static uint64_t disk_free_bytes(const pulsar_segstore *st) {
    if (st->disk_free_fn) return st->disk_free_fn(st->disk_free_ud);
    struct statvfs fs;
    if (statvfs(st->dir.c_str(), &fs) != 0) return UINT64_MAX;
    return (uint64_t)fs.f_bavail * (uint64_t)fs.f_frsize;
}

static bool budget_room(pulsar_segstore *st, uint64_t need, uint64_t freed, const char *keep, const char *keep2) {
    while (st->budget && st->used + st->reserved - freed + need > st->budget)
        if (!st->evict_one(keep, keep2)) break;
    if (st->budget && st->used + st->reserved - freed + need > st->budget) return false;
    /* L353: the budget counts this store's own files; the disk also holds other stores' and anyone's.  A full disk
     * evicts like a full budget -- without this, every later persist copied out, failed and wrote nothing.  The writes
     * in flight count in full: the disk may already hold some of them, so this can evict a little early, never late. */
    bool said = false;
    for (;;) {
        const uint64_t avail = disk_free_bytes(st);
        if (avail == UINT64_MAX || st->reserved + need <= avail) return true;
        if (!said) {
            st->logf("the disk has %llu MiB free for a %llu MiB write (%llu MiB in flight): evicting",
                     (unsigned long long)(avail >> 20), (unsigned long long)(need >> 20),
                     (unsigned long long)(st->reserved >> 20));
            said = true;
        }
        if (!st->evict_one(keep, keep2)) return false;
    }
}

int pulsar_segstore_reserve(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G, const char *text,
                            size_t text_len, uint64_t payload_bytes, uint64_t trailer_bytes, char key_out[41],
                            pulsar_segstore_job **job_out, char *err, size_t errlen) {
    return pulsar_segstore_reserve_images(st, parent, G_prev, G, text, text_len, nullptr, 0u, payload_bytes, trailer_bytes,
                                          key_out, job_out, err, errlen);
}

int pulsar_segstore_reserve_images(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G, const char *text,
                                   size_t text_len, const uint64_t *hashes, uint32_t n_hashes, uint64_t payload_bytes,
                                   uint64_t trailer_bytes, char key_out[41], pulsar_segstore_job **job_out, char *err,
                                   size_t errlen) {
    *job_out = nullptr;
    if (!st || G <= G_prev || (G_prev == 0) != (!parent || !parent[0])) {
        snprintf(err, errlen, "segment put: a root starts at 0 and only a root has no parent");
        return -1;
    }
    if (payload_bytes < PULSAR_KVDIGEST_BYTES) {
        snprintf(err, errlen, "segment put: a payload ends with its %u-byte digest; %llu bytes cannot",
                 PULSAR_KVDIGEST_BYTES, (unsigned long long)payload_bytes);
        return -1;
    }
    if (parent && parent[0] && !pulsar_segstore_contains(st, parent) && !st->pending.count(std::string(parent, 40))) {
        snprintf(err, errlen, "segment put: parent %.40s is not stored", parent);
        return -1;
    }
    char key[41];
    pulsar_segstore_child_key_images(parent, text, text_len, hashes, n_hashes, key);
    memcpy(key_out, key, 41);
    /* the same text is the same segment: stored, or being written (its write carries what this one would) */
    if (st->pending.count(std::string(key, 40)) || pulsar_segstore_contains(st, key)) return 0;
    const uint64_t need = pulsar_segstore_seg_file_bytes(text_len, payload_bytes, trailer_bytes);
    if (st->budget && need > st->budget) {
        snprintf(err, errlen, "segment put: %llu bytes exceed the whole budget", (unsigned long long)need);
        return -1;
    }
    if (!budget_room(st, need, 0, parent, key)) {
        snprintf(err, errlen, "segment put: no room (only the chain being extended and the segments being written "
                              "remain)");
        return -1;
    }
    const pulsar_segstore::image img = st->img_alloc(need);
    if (!img.p) {
        snprintf(err, errlen, "segment put: no room in the RAM pool for a %.1f MiB image now",
                 (double)need / (1024.0 * 1024.0));
        return -2;
    }
    pulsar_segstore_job *j = new pulsar_segstore_job();
    j->kind = PULSAR_SEGSTORE_PART_ROWS;
    memcpy(j->key, key, 41);
    memset(j->parent, 0, sizeof j->parent);
    if (parent && parent[0]) memcpy(j->parent, parent, 40);
    j->G_prev = G_prev;
    j->G = G;
    j->text_len = text_len;
    j->payload_off = HDR + text_len;
    j->payload_bytes = payload_bytes;
    j->trailer_bytes = trailer_bytes;
    j->flags = 0;
    j->need = need;
    j->img = img;
    j->final_path = st->path(key);
    /* the header and the text now; the payload and the trailer are the caller's to fill */
    uint8_t *h = img.p;
    memset(h, 0, HDR);
    const uint64_t now = (uint64_t)time(nullptr);
    put32(h, PULSAR_SEGSTORE_MAGIC);
    put32(h + 4, PULSAR_SEGSTORE_VERSION);
    put32(h + 8, st->identity);
    put32(h + 12, trailer_bytes ? PULSAR_SEGSTORE_FLAG_TRAILER : 0u);
    memcpy(h + 16, key, 40);
    if (j->parent[0]) memcpy(h + 56, j->parent, 40);
    put32(h + 96, G_prev);
    put32(h + 100, G);
    put64(h + 104, text_len);
    put64(h + 112, payload_bytes);
    put64(h + 120, trailer_bytes);
    put64(h + 136, now);
    put64(h + 144, now);
    if (text_len) memcpy(h + HDR, text, text_len);
    memset(img.p + need, 0, (size_t)(img.cap - need));   /* the O_DIRECT pad */
    st->reserved += need;
    st->pending.insert(std::string(key, 40));
    if (j->parent[0]) st->pinned[std::string(j->parent, 40)]++;
    *job_out = j;
    return 1;
}

int pulsar_segstore_reserve_rp(pulsar_segstore *st, const char key[41], uint64_t rp_bytes, uint32_t flags,
                               pulsar_segstore_job **job_out, char *err, size_t errlen) {
    *job_out = nullptr;
    if (!st || !key || rp_bytes < PULSAR_KVDIGEST_BYTES) {
        snprintf(err, errlen, "restore point put: no store or key, or fewer bytes than its digest");
        return -1;
    }
    const std::string k(key, 40);
    auto it = st->segs.find(k);
    const bool pending = it == st->segs.end() && st->pending.count(k) != 0;
    if (it == st->segs.end() && !pending) {
        snprintf(err, errlen, "restore point put: segment %.40s is not stored", key);
        return -1;
    }
    /* one in flight, or one written for the pending segment: the same key is the same state -- it gains the flags */
    auto flying = st->rp_jobs.find(k);
    if (flying != st->rp_jobs.end()) {
        flying->second |= flags;
        return 0;
    }
    auto waiting = st->pending_rp.find(k);
    if (pending && waiting != st->pending_rp.end()) {
        waiting->second.flags |= flags;
        return 0;
    }
    /* a stored one is replaced (its flags joined); it leaves the count at the commit, the room is made for the new */
    const uint64_t had = pending ? 0u : it->second.rp_bytes;
    const uint32_t joined = (had ? it->second.rp_flags : 0u) | flags;
    const uint64_t need = pulsar_segstore_rp_file_bytes(rp_bytes);
    if (st->budget && need > st->budget) {
        snprintf(err, errlen, "restore point put: %llu bytes exceed the whole budget", (unsigned long long)need);
        return -1;
    }
    if (!budget_room(st, need, had, key, nullptr)) {
        snprintf(err, errlen, "restore point put: no room for %llu bytes", (unsigned long long)need);
        return -1;
    }
    const pulsar_segstore::image img = st->img_alloc(need);
    if (!img.p) {
        snprintf(err, errlen, "restore point put: no room in the RAM pool for a %.1f MiB image now",
                 (double)need / (1024.0 * 1024.0));
        return -2;
    }
    pulsar_segstore_job *j = new pulsar_segstore_job();
    j->kind = PULSAR_SEGSTORE_PART_RP;
    memcpy(j->key, key, 40);
    j->key[40] = '\0';
    memset(j->parent, 0, sizeof j->parent);
    j->G_prev = 0;
    j->G = pending ? 0u : it->second.G;
    j->text_len = 0;
    j->payload_off = RP_HDR;
    j->payload_bytes = rp_bytes;
    j->trailer_bytes = 0;
    j->flags = joined;
    j->need = need;
    j->img = img;
    j->final_path = st->rp_path(j->key);
    uint8_t *h = img.p;
    memset(h, 0, RP_HDR);
    put32(h, PULSAR_SEGSTORE_RP_MAGIC);
    put32(h + 4, PULSAR_SEGSTORE_RP_VERSION);
    put32(h + 8, st->identity);
    put32(h + 12, joined);
    memcpy(h + 16, key, 40);
    put64(h + 56, rp_bytes);
    memset(img.p + need, 0, (size_t)(img.cap - need));
    st->reserved += need;
    st->rp_jobs[k] = 0u;
    st->pinned[k]++;
    *job_out = j;
    return 1;
}

int pulsar_segstore_job_kind(const pulsar_segstore_job *job) { return job->kind; }
const char *pulsar_segstore_job_key(const pulsar_segstore_job *job) { return job->key; }
uint32_t pulsar_segstore_job_G_prev(const pulsar_segstore_job *job) { return job->G_prev; }
uint32_t pulsar_segstore_job_G(const pulsar_segstore_job *job) { return job->G; }
uint64_t pulsar_segstore_job_bytes(const pulsar_segstore_job *job) { return job->need; }
uint64_t pulsar_segstore_job_image_bytes(const pulsar_segstore_job *job) { return job->img.cap; }
uint8_t *pulsar_segstore_job_payload(pulsar_segstore_job *job) { return job->img.p + job->payload_off; }
uint64_t pulsar_segstore_job_payload_bytes(const pulsar_segstore_job *job) { return job->payload_bytes; }
uint8_t *pulsar_segstore_job_trailer(pulsar_segstore_job *job) {
    return job->trailer_bytes ? job->img.p + job->payload_off + job->payload_bytes : nullptr;
}

uint64_t pulsar_segstore_seg_file_bytes(uint64_t text_len, uint64_t payload_bytes, uint64_t trailer_bytes) {
    return HDR + text_len + payload_bytes + trailer_bytes;
}

uint64_t pulsar_segstore_rp_file_bytes(uint64_t rp_bytes) { return RP_HDR + rp_bytes; }

uint64_t pulsar_segstore_image_bytes(uint64_t file_bytes) { return pulsar_kvpool_round(file_bytes + 1u); }

/* L347 I3a: the one file write of the store.  The payload's digest is sealed here (pulsar_kvdigest_seal: its last 8
 * bytes, over the rest), off the store's thread; the image goes to "<final>.tmp.<pid>" with O_DIRECT -- straight from
 * the image, which is granule-aligned and padded, so the write is whole granules; the file is then cut to its true
 * size (ftruncate: the file's bytes are exactly the store's layout, no pad on disk) -- and renamed into place.
 * NO fsync and no directory fsync: an O_DIRECT write is on the device when it returns, and a file a crash tore (the
 * rename's metadata lost, a tmp left behind) is refused when the store opens (tmp files removed, every file's size
 * checked against its header) or when it is read (the payload's digest). */
bool pulsar_segstore_job_write(pulsar_segstore_job *job, char *err, size_t errlen) {
    pulsar_kvdigest_seal(job->img.p + job->payload_off, job->payload_bytes);
    char tmp[64];
    snprintf(tmp, sizeof tmp, ".tmp.%ld", (long)getpid());
    const std::string tmp_path = job->final_path + tmp;
    const int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
    if (fd < 0) {
        snprintf(err, errlen, "segment write: %s: %s", tmp_path.c_str(), strerror(errno));
        return false;
    }
    const uint64_t padded = pulsar_kvpool_round(job->need);
    uint64_t done = 0;
    bool ok = true;
    while (ok && done < padded) {
        const uint64_t step = padded - done < (UINT64_C(1) << 30) ? padded - done : (UINT64_C(1) << 30);
        const ssize_t w = write(fd, job->img.p + done, (size_t)step);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0 || (uint64_t)w % PULSAR_KVPOOL_ALIGN != 0u) {
            snprintf(err, errlen, "segment write: %s: %s", tmp_path.c_str(),
                     w < 0 ? strerror(errno) : "a short O_DIRECT write");
            ok = false;
            break;
        }
        done += (uint64_t)w;
    }
    if (ok && ftruncate(fd, (off_t)job->need) != 0) {
        snprintf(err, errlen, "segment write: truncate %s: %s", tmp_path.c_str(), strerror(errno));
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        snprintf(err, errlen, "segment write: close %s: %s", tmp_path.c_str(), strerror(errno));
        ok = false;
    }
    if (ok && rename(tmp_path.c_str(), job->final_path.c_str()) != 0) {
        snprintf(err, errlen, "segment write: rename: %s", strerror(errno));
        ok = false;
    }
    if (!ok) ::unlink(tmp_path.c_str());
    return ok;
}

/** L347 I3a: commit a restore point job (see pulsar_segstore_commit).
 *  @param st      the store
 *  @param job     the job (not freed here)
 *  @param written its write succeeded
 *  @return true when the restore point is indexed, or waits for its pending segment */
static bool commit_rp(pulsar_segstore *st, pulsar_segstore_job *job, bool written) {
    const std::string key(job->key, 40);
    st->reserved -= job->need;
    {
        auto p = st->pinned.find(key);
        if (p != st->pinned.end() && --p->second == 0) st->pinned.erase(p);
    }
    uint32_t extra = 0;
    {
        auto f = st->rp_jobs.find(key);
        if (f != st->rp_jobs.end()) { extra = f->second; st->rp_jobs.erase(f); }
    }
    if (!written) { st->img_free(job->img); return false; }
    const uint32_t flags = job->flags | extra;
    if (flags != job->flags) {   /* flags joined while it flew: its header takes them, file and image */
        uint8_t b[4];
        put32(b, flags);
        const int fd = open(job->final_path.c_str(), O_WRONLY | O_CLOEXEC);
        const bool ok = fd >= 0 && pwrite(fd, b, 4, 12) == 4;
        if (fd >= 0) close(fd);
        if (!ok) st->logf("restore point %s: its joined flags could not be written (kept as written)", job->key);
        else memcpy(job->img.p + 12, b, 4);
    }
    const uint32_t kept = flags;
    auto it = st->segs.find(key);
    if (it != st->segs.end()) {
        st->used = st->used - it->second.rp_bytes + job->need;
        it->second.rp_bytes = job->need;
        it->second.rp_flags = kept;
        st->img_keep(st->ram_rp, key, job->img);
        if (kept & PULSAR_SEGSTORE_RP_PINNED) {   /* L347 I3b */
            it->second.last_used = (uint64_t)time(nullptr);
            st->enforce_pinned_max();
        }
        return true;
    }
    if (st->pending.count(key)) {   /* its segment is still being written: the restore point joins it at its commit */
        pulsar_segstore::pending_point &pp = st->pending_rp[key];
        pp.bytes = job->need;
        pp.flags = kept;
        pp.img = job->img;
        st->used += job->need;
        return true;
    }
    st->logf("restore point %s dropped: its segment is gone", job->key);
    ::unlink(job->final_path.c_str());
    st->img_free(job->img);
    return false;
}

bool pulsar_segstore_commit(pulsar_segstore *st, pulsar_segstore_job *job, bool written) {
    if (!written) st->writes_failed++;
    if (job->kind == PULSAR_SEGSTORE_PART_RP) {
        const bool ok = commit_rp(st, job, written);
        delete job;
        return ok;
    }
    const std::string key(job->key, 40), parent(job->parent);
    st->reserved -= job->need;
    st->pending.erase(key);
    if (!parent.empty()) {
        auto p = st->pinned.find(parent);
        if (p != st->pinned.end() && --p->second == 0) st->pinned.erase(p);
    }
    /* L347: a restore point written while the segment was pending joins it, or goes with it */
    pulsar_segstore::pending_point rp;
    {
        auto r = st->pending_rp.find(key);
        if (r != st->pending_rp.end()) {
            rp = r->second;
            st->pending_rp.erase(r);
        }
    }
    /* a parent that was itself pending and whose write failed is not stored: its child cannot be named by a chain */
    const bool parent_ok = parent.empty() || st->segs.count(parent) != 0;
    bool indexed = false;
    if (written && !parent_ok) {
        st->logf("dropped segment %s [%u,%u): its parent %s was not stored", job->key, job->G_prev, job->G,
                 parent.c_str());
        ::unlink(job->final_path.c_str());
    } else if (written) {
        pulsar_segstore_seg e;
        memset(&e, 0, sizeof e);
        memcpy(e.key, job->key, 41);
        if (!parent.empty()) memcpy(e.parent, job->parent, 40);
        e.G_prev = job->G_prev;
        e.G = job->G;
        e.text_len = job->text_len;
        e.payload_bytes = job->payload_bytes;
        e.trailer_bytes = job->trailer_bytes;
        e.file_bytes = job->need;
        e.last_used = (uint64_t)time(nullptr);
        e.text_end = (parent.empty() ? 0u : st->segs[parent].text_end) + job->text_len;
        st->used -= rp.bytes;   /* link counts it again, on the entry */
        e.rp_bytes = rp.bytes;
        e.rp_flags = rp.bytes ? rp.flags : 0u;
        st->link(e);
        st->heads[e.key] = std::string((const char *)job->img.p + HDR,
                                       (size_t)(job->text_len < HEAD_BYTES ? job->text_len : HEAD_BYTES));
        st->refresh_dominated(e.parent);
        st->img_keep(st->ram_rows, key, job->img);
        if (rp.bytes) st->img_keep(st->ram_rp, key, rp.img);
        if (rp.bytes && (rp.flags & PULSAR_SEGSTORE_RP_PINNED)) st->enforce_pinned_max();   /* L347 I3b */
        indexed = true;
    }
    if (!indexed) {
        st->img_free(job->img);
        if (rp.bytes) {
            ::unlink(st->rp_path(key.c_str()).c_str());
            st->used -= rp.bytes;
            st->img_free(rp.img);
            st->rps_removed++;   /* L276 */
            if (st->removed) st->removed(st->removed_ud, key.c_str(), true);
        }
    }
    delete job;
    return indexed;
}

bool pulsar_segstore_put(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G,
                         const char *text, size_t text_len,
                         uint64_t payload_bytes, pulsar_segstore_write_fn write_payload,
                         uint64_t trailer_bytes, pulsar_segstore_write_fn write_trailer,
                         void *ud, char key_out[41], char *err, size_t errlen) {
    if (!write_payload) {
        snprintf(err, errlen, "segment put: no payload writer");
        return false;
    }
    pulsar_segstore_job *job = nullptr;
    const int rc = pulsar_segstore_reserve(st, parent, G_prev, G, text, text_len, payload_bytes,
                                           write_trailer ? trailer_bytes : 0u, key_out, &job, err, errlen);
    if (rc <= 0) return rc == 0;
    bool ok = fill_part(pulsar_segstore_job_payload(job), payload_bytes, write_payload, ud, "payload", err, errlen) == 0;
    if (ok && job->trailer_bytes)
        ok = fill_part(pulsar_segstore_job_trailer(job), job->trailer_bytes, write_trailer, ud, "trailer", err,
                       errlen) == 0;
    ok = ok && pulsar_segstore_job_write(job, err, errlen);
    return pulsar_segstore_commit(st, job, ok) && ok;
}

bool pulsar_segstore_put_rp(pulsar_segstore *st, const char key[41], uint64_t rp_bytes, uint32_t flags,
                            pulsar_segstore_write_fn write_rp, void *ud, char *err, size_t errlen) {
    if (!write_rp) {
        snprintf(err, errlen, "restore point put: no writer");
        return false;
    }
    pulsar_segstore_job *job = nullptr;
    const int rc = pulsar_segstore_reserve_rp(st, key, rp_bytes, flags, &job, err, errlen);
    if (rc <= 0) return rc == 0;
    bool ok = fill_part(pulsar_segstore_job_payload(job), rp_bytes, write_rp, ud, "restore point", err, errlen) == 0;
    ok = ok && pulsar_segstore_job_write(job, err, errlen);
    return pulsar_segstore_commit(st, job, ok) && ok;
}

bool pulsar_segstore_flag_rp(pulsar_segstore *st, const char key[41], uint32_t flags) {
    if (!st) return false;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end() || it->second.rp_bytes == 0u) return false;
    const bool pinning = (flags & ~it->second.rp_flags & PULSAR_SEGSTORE_RP_PINNED) != 0u;
    if (!st->write_rp_flags(it->second, it->second.rp_flags | flags)) return false;
    if (pinning) {
        it->second.last_used = (uint64_t)time(nullptr);   /* the newest pin is the last to lose it */
        st->enforce_pinned_max();
    }
    return true;
}

bool pulsar_segstore_covers(pulsar_segstore *st, const char *text, size_t len, bool *rp_at_len) {
    *rp_at_len = false;
    if (!st || !text || len == 0u) return false;
    /* the lookup's walk (children whose key the text's own bytes reproduce), plus, at each node it reaches, a child
     * that runs past `len` and starts with the rest of the text */
    struct frame { std::string key; uint64_t end; };
    std::vector<frame> stack{{"", 0}};
    bool covered = false;
    while (!stack.empty()) {
        const frame f = stack.back();
        stack.pop_back();
        if (!f.key.empty() && f.end == len) {
            covered = true;
            if (st->segs[f.key].rp_bytes > 0u) *rp_at_len = true;
            continue;
        }
        auto ch = st->children.find(f.key);
        if (ch == st->children.end()) continue;
        for (const std::string &c : ch->second) {
            const pulsar_segstore_seg &cs = st->segs[c];
            if (f.end + cs.text_len <= len) {
                char k[41];
                pulsar_segstore_child_key(f.key.c_str(), text + f.end, cs.text_len, k);
                if (c == k) stack.push_back({c, f.end + cs.text_len});
                continue;
            }
            if (covered) continue;
            const size_t rest = len - f.end;
            const std::string &head = st->heads[c];
            if (memcmp(head.data(), text + f.end, rest < head.size() ? rest : head.size()) != 0) continue;
            std::string t;
            covered = rest <= head.size() || (st->read_text(c, &t) && memcmp(t.data(), text + f.end, rest) == 0);
        }
    }
    return covered;
}

/* L347 I3a: a part's bytes for a load -- the RAM copy when the entry has one (no disk I/O), else the file read whole
 * with O_DIRECT into a fresh image (granule-sized reads into the pool's aligned block, the short last one at EOF), its
 * header and size checked and its payload's digest checked (pulsar_kvdigest_check): a mismatch is the file's fault.
 * With the RAM tier the image read stays as the entry's RAM copy; without it, it is the load's until the unmap. */
int pulsar_segstore_map_part(pulsar_segstore *st, const char key[41], int part, const uint8_t **payload,
                             uint64_t *bytes, bool *from_disk, char *err, size_t errlen) {
    *payload = nullptr;
    *bytes = 0;
    if (from_disk) *from_disk = false;
    if (!st) { snprintf(err, errlen, "no store"); return 1; }
    const std::string k(key, 40);
    auto it = st->segs.find(k);
    const bool rp = part == PULSAR_SEGSTORE_PART_RP;
    if (it == st->segs.end() || (rp && it->second.rp_bytes == 0u)) {
        snprintf(err, errlen, "%s not stored", rp ? "restore point" : "segment");
        return 1;
    }
    const pulsar_segstore_seg &e = it->second;
    const uint64_t off = rp ? RP_HDR : HDR + e.text_len;
    const uint64_t pbytes = rp ? e.rp_bytes - RP_HDR : e.payload_bytes;
    auto &ram = rp ? st->ram_rp : st->ram_rows;
    auto res = ram.find(k);
    if (res != ram.end()) {
        st->held[pulsar_segstore::hold_name(k, rp)]++;
        *payload = res->second.p + off;
        *bytes = pbytes;
        return 0;
    }
    const uint64_t file_bytes = rp ? e.rp_bytes : e.file_bytes;
    const pulsar_segstore::image img = st->img_alloc(file_bytes);
    if (!img.p) {
        snprintf(err, errlen, "no room in the RAM pool to read a %.1f MiB %s now", (double)file_bytes / (1024.0 * 1024.0),
                 rp ? "restore point" : "segment");
        return 2;
    }
    const std::string path = rp ? st->rp_path(k.c_str()) : st->path(k.c_str());
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    uint64_t got = 0;
    int rd_errno = 0;
    if (fd < 0) rd_errno = errno;
    const uint64_t padded = pulsar_kvpool_round(file_bytes);
    while (fd >= 0 && got < padded) {
        const uint64_t step = padded - got < (UINT64_C(1) << 30) ? padded - got : (UINT64_C(1) << 30);
        const ssize_t r = read(fd, img.p + got, (size_t)step);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) { rd_errno = errno; break; }
        if (r == 0) break;
        got += (uint64_t)r;
        if ((uint64_t)r % PULSAR_KVPOOL_ALIGN != 0u) break;   /* the short read at EOF */
    }
    if (fd >= 0) close(fd);
    const char *why = nullptr;
    if (rd_errno) why = strerror(rd_errno);
    else if (got != file_bytes) why = "its size is not the indexed size";
    else if (rp) {
        char hk[41];
        uint64_t hp = 0;
        uint32_t hf = 0;
        bool older = false;
        if (!st->parse_rp_header(img.p, hk, &hp, &hf, &older) || memcmp(hk, k.data(), 40) != 0 || hp != pbytes)
            why = "its header is not the indexed one";
    } else {
        pulsar_segstore_seg h;
        uint32_t hf = 0;
        if (!st->parse_header(img.p, &h, &hf) || memcmp(h.key, k.data(), 40) != 0 || h.payload_bytes != pbytes ||
            h.text_len != e.text_len)
            why = "its header is not the indexed one";
    }
    if (!why && !pulsar_kvdigest_check(img.p + off, pbytes))
        why = rp ? "digest mismatch (corrupt restore point)" : "digest mismatch (corrupt segment)";
    if (why) {
        snprintf(err, errlen, "%s %s: %s", rp ? "restore point" : "segment", path.c_str(), why);
        st->img_free(img);
        return 1;
    }
    if (st->retain) ram[k] = img;
    else (rp ? st->loan_rp : st->loan_rows)[k] = img;
    st->held[pulsar_segstore::hold_name(k, rp)]++;
    *payload = img.p + off;
    *bytes = pbytes;
    if (from_disk) *from_disk = true;
    return 0;
}

void pulsar_segstore_unmap_part(pulsar_segstore *st, const char key[41], int part) {
    if (!st) return;
    const std::string k(key, 40);
    const bool rp = part == PULSAR_SEGSTORE_PART_RP;
    auto h = st->held.find(pulsar_segstore::hold_name(k, rp));
    if (h == st->held.end()) return;
    if (--h->second > 0) return;   /* a second map still reads it */
    st->held.erase(h);
    st->img_drop(rp ? st->loan_rp : st->loan_rows, k);
}

bool pulsar_segstore_ram_make_room(pulsar_segstore *st, uint64_t bytes) {
    return st && st->ram_make_room(pulsar_segstore_image_bytes(bytes));
}

uint64_t pulsar_segstore_resident_bytes(pulsar_segstore *st, const char key[41], bool rows, bool rp) {
    if (!st) return 0u;
    const std::string k(key, 40);
    auto it = st->segs.find(k);
    if (it == st->segs.end()) return 0u;
    uint64_t got = 0;
    if (rows && st->ram_rows.count(k)) got += it->second.payload_bytes + it->second.trailer_bytes;
    if (rp && st->ram_rp.count(k)) got += it->second.rp_bytes;
    return got;
}

uint64_t pulsar_segstore_ram_bytes(const pulsar_segstore *st) {
    uint64_t got = 0;
    if (st)
        for (auto *m : {&st->ram_rows, &st->ram_rp})
            for (auto &kv : *m) got += kv.second.cap;
    return got;
}

bool pulsar_segstore_drop_rp(pulsar_segstore *st, const char key[41]) {
    if (!st) return false;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end() || it->second.rp_bytes == 0u) return false;
    st->drop_rp_of(it->second, "dropped by its user");
    return true;
}

FILE *pulsar_segstore_open_payload(pulsar_segstore *st, const char key[41], uint64_t *payload_bytes) {
    if (!st) return nullptr;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end()) return nullptr;
    FILE *fp = fopen(st->path(it->first.c_str()).c_str(), "rb");
    if (!fp) return nullptr;
    pulsar_segstore_seg e;
    uint32_t flags = 0;
    if (!st->read_header(fp, &e, &flags) || fseek(fp, (long)(HDR + e.text_len), SEEK_SET) != 0) {
        fclose(fp);
        return nullptr;
    }
    if (payload_bytes) *payload_bytes = e.payload_bytes;
    return fp;
}

FILE *pulsar_segstore_open_trailer(pulsar_segstore *st, const char key[41], uint64_t *trailer_bytes) {
    if (!st) return nullptr;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end() || !it->second.trailer_bytes) return nullptr;
    FILE *fp = fopen(st->path(it->first.c_str()).c_str(), "rb");
    if (!fp) return nullptr;
    const pulsar_segstore_seg &e = it->second;
    if (fseek(fp, (long)(HDR + e.text_len + e.payload_bytes), SEEK_SET) != 0) { fclose(fp); return nullptr; }
    if (trailer_bytes) *trailer_bytes = e.trailer_bytes;
    return fp;
}

void pulsar_segstore_touch(pulsar_segstore *st, const pulsar_segstore_seg *chain, int n) {
    if (!st) return;
    const uint64_t now = (uint64_t)time(nullptr);
    for (int i = 0; i < n; i++) {
        auto it = st->segs.find(chain[i].key);
        if (it == st->segs.end()) continue;
        it->second.hits++;
        it->second.last_used = now;
        FILE *fp = fopen(st->path(it->first.c_str()).c_str(), "r+b");
        if (!fp) continue;
        uint8_t b[16];
        put32(b, it->second.hits);
        put64(b + 8, now);
        if (fseek(fp, 128, SEEK_SET) == 0) (void)fwrite(b, 1, 4, fp);
        if (fseek(fp, 144, SEEK_SET) == 0) (void)fwrite(b + 8, 1, 8, fp);
        fclose(fp);
    }
}

void pulsar_segstore_drop(pulsar_segstore *st, const char key[41]) {
    if (!st) return;
    std::vector<std::string> stack{std::string(key, 40)};
    while (!stack.empty()) {
        const std::string k = stack.back();
        stack.pop_back();
        auto ch = st->children.find(k);
        if (ch != st->children.end()) for (auto &c : ch->second) stack.push_back(c);
        st->unlink_files(k);
        st->unlink_index(k);
    }
}

void pulsar_segstore_set_removed_hook(pulsar_segstore *st, pulsar_segstore_removed_fn fn, void *ud) {
    if (!st) return;
    st->removed = fn;
    st->removed_ud = ud;
}

uint64_t pulsar_segstore_release(pulsar_segstore *st, const char tip[41], const char *stop) {
    if (!st) return 0;
    std::string cur(tip, 40);
    if (!st->segs.count(cur) || (stop && cur == stop)) return 0;
    auto ch = st->children.find(cur);
    if (ch != st->children.end() && !ch->second.empty()) return 0;
    for (;;) {
        const std::string parent = st->segs[cur].parent;
        if (parent.empty() || (stop && parent == stop) || st->children[parent].size() != 1) break;
        cur = parent;
    }
    const uint64_t before = st->used;
    char key[41];
    memcpy(key, cur.c_str(), 41);
    pulsar_segstore_drop(st, key);
    return before - st->used;
}

void pulsar_segstore_foreach_trailer(pulsar_segstore *st, pulsar_segstore_trailer_fn fn, void *ud) {
    if (!st || !fn) return;
    /* A client thread's walk: copy the entries under the lock, read the files without it.  A segment evicted since
     * the copy fails its fopen and is skipped; one evicted after it stays readable through the open file.  These
     * fields never change once an entry is in the index. */
    struct walk_entry { std::string key; uint64_t text_len, payload_bytes, trailer_bytes; };
    std::vector<walk_entry> entries;
    {
        std::lock_guard<std::mutex> lk(st->segs_mu);
        for (auto &kv : st->segs)
            if (kv.second.trailer_bytes)
                entries.push_back({kv.first, kv.second.text_len, kv.second.payload_bytes, kv.second.trailer_bytes});
    }
    for (const walk_entry &e : entries) {
        FILE *fp = fopen(st->path(e.key.c_str()).c_str(), "rb");
        if (!fp) continue;
        std::string text(e.text_len, '\0');
        bool go = fseek(fp, (long)HDR, SEEK_SET) == 0 && fread(&text[0], 1, e.text_len, fp) == e.text_len &&
                  fseek(fp, (long)(HDR + e.text_len + e.payload_bytes), SEEK_SET) == 0;
        const bool stop = go && !fn(fp, e.trailer_bytes, text.data(), text.size(), ud);
        fclose(fp);
        if (stop) break;
    }
}

int pulsar_segstore_keys(pulsar_segstore *st, char **keys) {
    *keys = nullptr;
    if (!st || st->segs.empty()) return 0;
    char *out = (char *)malloc(st->segs.size() * 40 + 1);
    if (!out) return 0;
    size_t i = 0;
    for (auto &kv : st->segs) { memcpy(out + i * 40, kv.first.data(), 40); i++; }
    out[i * 40] = '\0';
    *keys = out;
    return (int)i;
}

void pulsar_segstore_pin(pulsar_segstore *st, const char key[41]) {
    if (st && key) st->pinned[std::string(key, 40)]++;
}

void pulsar_segstore_unpin(pulsar_segstore *st, const char key[41]) {
    if (!st || !key) return;
    auto p = st->pinned.find(std::string(key, 40));
    if (p != st->pinned.end() && --p->second == 0) st->pinned.erase(p);
}

bool pulsar_segstore_is_pinned(const pulsar_segstore *st, const char key[41]) {
    return st && key && st->pinned.count(std::string(key, 40)) != 0;
}

void pulsar_segstore_set_pinned_max(pulsar_segstore *st, uint64_t bytes) {
    if (!st) return;
    st->pinned_max = bytes;
    st->enforce_pinned_max();
}

int pulsar_segstore_pinned_anchors(const pulsar_segstore *st, uint64_t *bytes) {
    int n = 0;
    const uint64_t b = st ? st->pinned_bytes(&n) : 0u;
    if (bytes) *bytes = b;
    return n;
}

uint64_t pulsar_segstore_pins_released(const pulsar_segstore *st) { return st ? st->pins_released : 0u; }

uint64_t pulsar_segstore_used_bytes(const pulsar_segstore *st) { return st ? st->used : 0u; }
uint64_t pulsar_segstore_budget_bytes(const pulsar_segstore *st) { return st ? st->budget : 0u; }
uint64_t pulsar_segstore_removed(const pulsar_segstore *st, uint64_t *rps) {
    if (rps) *rps = st ? st->rps_removed : 0u;
    return st ? st->segs_removed : 0u;
}
uint64_t pulsar_segstore_writes_failed(const pulsar_segstore *st) { return st ? st->writes_failed : 0u; }
void pulsar_segstore_set_disk_free(pulsar_segstore *st, pulsar_segstore_disk_free_fn fn, void *ud) {
    if (!st) return;
    st->disk_free_fn = fn;
    st->disk_free_ud = ud;
}
int pulsar_segstore_count(const pulsar_segstore *st) { return st ? (int)st->segs.size() : 0; }

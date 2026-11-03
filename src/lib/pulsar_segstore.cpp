/** @file
 *  The content-addressed segment store's implementation (pulsar_segstore.h). */
/* L264 S4: the content-addressed segment store (pulsar_segstore.h). */
#include "pulsar_segstore.h"
#include "pulsar_writeback.h"
#include "sha1.hpp"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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
    std::unordered_map<std::string, std::vector<std::string>> children;   ///< parent key ("" = roots) -> child keys
    /// Sum of file_bytes and rp_bytes over segs, plus pending_rp: what the budget is checked against
    uint64_t used = 0;
    /// L338: file bytes of the reserved writes not yet committed (pulsar_segstore_reserve): the budget counts them
    uint64_t reserved = 0;
    /// L338: keys being written (reserved, not committed): a second reserve of one writes nothing
    std::unordered_set<std::string> pending;
    /// L338: parents of the writes in flight, with their count: eviction never takes one
    std::unordered_map<std::string, int> pinned;
    /** L347: a restore point written for a pending key (its .seg not committed yet). */
    struct pending_point {
        uint64_t bytes = 0;   ///< its .rp file's bytes
        uint32_t flags = 0;   ///< its PULSAR_SEGSTORE_RP_* flags
    };
    /// L347: restore points written for pending keys, moved onto the entry at its commit (or removed with a failed one)
    std::unordered_map<std::string, pending_point> pending_rp;
    /// L347 I2: key -> the first HEAD_BYTES of its own text (all of it when shorter), for pulsar_segstore_covers
    std::unordered_map<std::string, std::string> heads;

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
        if (fread(h, 1, HDR, fp) != HDR) return false;
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
        if (fread(h, 1, RP_HDR, fp) != RP_HDR || get32(h) != PULSAR_SEGSTORE_RP_MAGIC) return false;
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
    /** Remove a stored segment's files, its .seg and its .rp (L347), from disk (not from the index).
     *  @param key the segment */
    void unlink_files(const std::string &key) const {
        ::unlink(path(key.c_str()).c_str());
        ::unlink(rp_path(key.c_str()).c_str());
    }

    /** Add e to the index: segs, its parent's children, used. Does not refresh dominance.
     *  @param e the entry (file_bytes set) */
    void link(const pulsar_segstore_seg &e) {
        segs[e.key] = e;
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
        used -= it->second.file_bytes + it->second.rp_bytes;
        const std::string parent = it->second.parent;
        auto &sib = children[parent];
        for (size_t i = 0; i < sib.size(); i++)
            if (sib[i] == key) { sib.erase(sib.begin() + (long)i); break; }
        children.erase(key);
        heads.erase(key);
        segs.erase(it);
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
        used -= e.rp_bytes;
        e.rp_bytes = 0;
        e.rp_flags = 0;
        if (removed) removed(removed_ud, e.key, true);
    }
    /** @param key a stored key
     *  @return how many children the index holds for it */
    size_t n_children(const std::string &key) const {
        auto ch = children.find(key);
        return ch == children.end() ? 0u : ch->second.size();
    }
    /* L347 I2 (D5): the eviction classes, lowest first -- (0) a restore point neither flagged nor on a branch point (a
     * dead end's or a superseded one's: big, and a restore from it is the cheapest thing to lose); (1) a dominated leaf
     * (L323; only a leaf without a restore point, so never a flagged one); (2) any other leaf; (3) a flagged (anchor,
     * periodic) or branch-point restore point: where restores of many requests start.  -1: not a candidate (a segment
     * with children and no restore point). */
    /** @param e      an index entry
     *  @param leaf   it has no children
     *  @param branch it has >= 2 children
     *  @return its eviction class (above), -1 for none */
    static int evict_class(const pulsar_segstore_seg &e, bool leaf, bool branch) {
        if (e.rp_bytes > 0u) return e.rp_flags == 0u && !branch ? 0 : 3;
        if (!leaf) return -1;
        return e.dominated ? 1 : 2;
    }
    /** Evict one restore point (file and index), or one leaf (files and index), by the classes above: the lowest class
     *  first, the least recently used within it.  Never `keep` or `keep2`, a parent of a write in flight or a key being
     *  written (L338).
     *  @param keep a key to spare (NULL = none)
     *  @param keep2 a second key to spare (NULL = none)
     *  @return false when nothing may go */
    bool evict_one(const char *keep, const char *keep2) {
        const pulsar_segstore_seg *victim = nullptr;
        int victim_class = 4;
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
        if (!victim) return false;
        const std::string k = victim->key;
        if (victim->rp_bytes > 0u) {
            drop_rp_of(segs[k], victim_class == 0 ? "store over budget"
                                                  : "store over budget (a flagged or branch point's)");
            return true;
        }
        logf("evicted segment %s [%u,%u) %llu bytes%s: store over budget", k.c_str(), victim->G_prev, victim->G,
             (unsigned long long)victim->file_bytes, victim->dominated ? " (dominated)" : "");
        unlink_files(k);
        unlink_index(k);
        return true;
    }
};

pulsar_segstore *pulsar_segstore_open(const char *dir, uint64_t budget_bytes, uint32_t identity,
                                      pulsar_segstore_log_fn log, void *log_ud) {
    if (!dir || !dir[0]) return nullptr;
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return nullptr;
    /* Refuse a directory this process cannot write up front, instead of
     * failing every store later. */
    {
        char probe[4096];
        snprintf(probe, sizeof probe, "%s/.probe.%ld", dir, (long)getpid());
        const int fd = open(probe, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return nullptr;
        close(fd);
        ::unlink(probe);
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

void pulsar_segstore_close(pulsar_segstore *st) { delete st; }

void pulsar_segstore_child_key(const char *parent, const char *text, size_t len, char out[41]) {
    pulsar::Sha1 h;
    if (parent && parent[0]) h.update(parent, 40);
    h.update(text, len);
    uint8_t d[20];
    h.final(d);
    pulsar::Sha1::hex20(d, out);
}

int pulsar_segstore_lookup(pulsar_segstore *st, const char *text, size_t len, const pulsar_segstore_end *want,
                           pulsar_segstore_seg *chain, int cap) {
    if (!st || !text || cap <= 0) return 0;
    /* Depth-first over the children whose key the request's own bytes
     * reproduce; the deepest accepted end wins (by G, then by text). */
    struct frame { std::string key; uint64_t end; int depth; };
    std::vector<frame> stack;
    std::vector<std::string> path(cap), best_path;
    int best_depth = 0;
    uint32_t best_G = 0;
    uint64_t best_end = 0;
    stack.push_back({"", 0, 0});
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
            char k[41];
            pulsar_segstore_child_key(f.key.c_str(), text + f.end, cs.text_len, k);
            if (c == k) stack.push_back({c, f.end + cs.text_len, f.depth + 1});
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

/** Write all `n` bytes of `p` to `fp`.
 *  @param fp the open file
 *  @param p  the bytes
 *  @param n  their count
 *  @return 0 when every byte was written, 1 on a short write */
static int write_all(FILE *fp, const void *p, size_t n) { return fwrite(p, 1, n, fp) == n ? 0 : 1; }

/** Run one writer and check it wrote exactly what it promised.
 *  @param fp     the file, positioned where the bytes go
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
    if ((uint64_t)(ftell(fp) - at) != bytes) {
        snprintf(err, errlen, "segment put: %s wrote %ld bytes, promised %llu", what, ftell(fp) - at,
                 (unsigned long long)bytes);
        return 1;
    }
    return 0;
}

/* The one durable file write of the store: "<final>.tmp.<pid>", the header, `body`, fsync, out of the page cache
 * (L261), rename, directory fsync.  False leaves no file. */
/// @tparam F     a callable `int(FILE *)` writing the bytes after the header: 0 on success (err filled otherwise)
/// @param final_path the file's path
/// @param hdr        the fixed header
/// @param hdr_len    its bytes
/// @param body       writes the rest
/// @param err        receives a reason on failure
/// @param errlen     size of `err`
/// @return true when the file is in place
template <typename F>
static bool durable_write(const std::string &final_path, const uint8_t *hdr, size_t hdr_len, F body, char *err,
                          size_t errlen) {
    char tmp[64];
    snprintf(tmp, sizeof tmp, ".tmp.%ld", (long)getpid());
    const std::string tmp_path = final_path + tmp;
    FILE *fp = fopen(tmp_path.c_str(), "wb");
    if (!fp) { snprintf(err, errlen, "segment put: %s: %s", tmp_path.c_str(), strerror(errno)); return false; }
    pulsar_writeback wb;
    pulsar_writeback_init(&wb, fp);
    int rc = write_all(fp, hdr, hdr_len);
    if (rc) snprintf(err, errlen, "segment put: header: %s", strerror(errno));
    if (rc == 0) rc = body(fp);
    if (rc == 0 && (fflush(fp) != 0 || fsync(fileno(fp)) != 0)) {
        snprintf(err, errlen, "segment put: flush: %s", strerror(errno));
        rc = 1;
    }
    if (rc == 0) pulsar_writeback_finish(&wb);
    pulsar_writeback_drop_file(fp);
    fclose(fp);
    if (rc == 0 && rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        snprintf(err, errlen, "segment put: rename: %s", strerror(errno));
        rc = 1;
    }
    if (rc != 0) { ::unlink(tmp_path.c_str()); return false; }
    const std::string dir = final_path.substr(0, final_path.rfind('/'));
    const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) { (void)fsync(dfd); close(dfd); }
    return true;
}

/** L338: a put in three steps, so the file write can leave the thread that owns the index.  The job carries
 * everything the write needs (no store state is read while it runs); the store keeps the job's key PENDING (a second
 * reserve of it writes nothing) and its parent PINNED (eviction takes leaves only, and a pending child is not yet a
 * child in the index) until the commit. */
struct pulsar_segstore_job {
    char key[41];            ///< the segment's key
    char parent[41];         ///< its parent's key ("" for a root)
    uint32_t G_prev;         ///< the span's start
    uint32_t G;              ///< the span's end
    std::string text;        ///< the segment's own rendered text
    uint64_t payload_bytes;  ///< the payload writer's bytes
    uint64_t trailer_bytes;  ///< the trailer writer's bytes (0 = none)
    bool with_trailer;       ///< a trailer follows the payload
    uint64_t need;           ///< the file's bytes
    uint32_t identity;       ///< the store identity the header carries
    std::string final_path;  ///< `<dir>/<key>.seg`
};

int pulsar_segstore_reserve(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G, const char *text,
                            size_t text_len, uint64_t payload_bytes, uint64_t trailer_bytes, char key_out[41],
                            pulsar_segstore_job **job_out, char *err, size_t errlen) {
    *job_out = nullptr;
    if (!st || G <= G_prev || (G_prev == 0) != (!parent || !parent[0])) {
        snprintf(err, errlen, "segment put: a root starts at 0 and only a root has no parent");
        return -1;
    }
    if (parent && parent[0] && !pulsar_segstore_contains(st, parent) && !st->pending.count(std::string(parent, 40))) {
        snprintf(err, errlen, "segment put: parent %.40s is not stored", parent);
        return -1;
    }
    char key[41];
    pulsar_segstore_child_key(parent, text, text_len, key);
    memcpy(key_out, key, 41);
    /* the same text is the same segment: stored, or being written (its write carries what this one would) */
    if (st->pending.count(std::string(key, 40)) || pulsar_segstore_contains(st, key)) return 0;
    const uint64_t need = HDR + text_len + payload_bytes + trailer_bytes;
    if (st->budget && need > st->budget) {
        snprintf(err, errlen, "segment put: %llu bytes exceed the whole budget", (unsigned long long)need);
        return -1;
    }
    while (st->budget && st->used + st->reserved + need > st->budget)
        if (!st->evict_one(parent, key)) break;
    if (st->budget && st->used + st->reserved + need > st->budget) {
        snprintf(err, errlen, "segment put: no room (only the chain being extended and the segments being written "
                              "remain)");
        return -1;
    }
    pulsar_segstore_job *j = new pulsar_segstore_job();
    memcpy(j->key, key, 41);
    memset(j->parent, 0, sizeof j->parent);
    if (parent && parent[0]) memcpy(j->parent, parent, 40);
    j->G_prev = G_prev;
    j->G = G;
    j->text.assign(text, text_len);
    j->payload_bytes = payload_bytes;
    j->trailer_bytes = trailer_bytes;
    j->with_trailer = trailer_bytes > 0u;
    j->need = need;
    j->identity = st->identity;
    j->final_path = st->path(key);
    st->reserved += need;
    st->pending.insert(std::string(key, 40));
    if (j->parent[0]) st->pinned[std::string(j->parent, 40)]++;
    *job_out = j;
    return 1;
}

const char *pulsar_segstore_job_key(const pulsar_segstore_job *job) { return job->key; }
uint32_t pulsar_segstore_job_G_prev(const pulsar_segstore_job *job) { return job->G_prev; }
uint32_t pulsar_segstore_job_G(const pulsar_segstore_job *job) { return job->G; }
const char *pulsar_segstore_job_text(const pulsar_segstore_job *job) { return job->text.c_str(); }
uint64_t pulsar_segstore_job_bytes(const pulsar_segstore_job *job) { return job->need; }

bool pulsar_segstore_job_write(pulsar_segstore_job *job, pulsar_segstore_write_fn write_payload,
                               pulsar_segstore_write_fn write_trailer, void *ud, char *err, size_t errlen) {
    if (!write_payload || (job->with_trailer && !write_trailer)) {
        snprintf(err, errlen, "segment put: no writer for the payload or its trailer");
        return false;
    }
    uint8_t h[HDR];
    memset(h, 0, sizeof h);
    const uint64_t now = (uint64_t)time(nullptr);
    put32(h, PULSAR_SEGSTORE_MAGIC);
    put32(h + 4, PULSAR_SEGSTORE_VERSION);
    put32(h + 8, job->identity);
    put32(h + 12, job->with_trailer ? PULSAR_SEGSTORE_FLAG_TRAILER : 0u);
    memcpy(h + 16, job->key, 40);
    if (job->parent[0]) memcpy(h + 56, job->parent, 40);
    put32(h + 96, job->G_prev);
    put32(h + 100, job->G);
    put64(h + 104, job->text.size());
    put64(h + 112, job->payload_bytes);
    put64(h + 120, job->trailer_bytes);
    put64(h + 136, now);
    put64(h + 144, now);
    return durable_write(job->final_path, h, HDR, [&](FILE *fp) {
        if (write_all(fp, job->text.data(), job->text.size()) != 0) {
            snprintf(err, errlen, "segment put: text: %s", strerror(errno));
            return 1;
        }
        if (write_part(fp, write_payload, ud, job->payload_bytes, "payload", err, errlen) != 0) return 1;
        return job->with_trailer ? write_part(fp, write_trailer, ud, job->trailer_bytes, "trailer", err, errlen) : 0;
    }, err, errlen);
}

bool pulsar_segstore_commit(pulsar_segstore *st, pulsar_segstore_job *job, bool written) {
    const std::string key(job->key, 40), parent(job->parent);
    st->reserved -= job->need;
    st->pending.erase(key);
    if (!parent.empty()) {
        auto p = st->pinned.find(parent);
        if (p != st->pinned.end() && --p->second == 0) st->pinned.erase(p);
    }
    /* L347: a restore point written while the segment was pending joins it, or goes with it */
    uint64_t rp_bytes = 0;
    uint32_t rp_flags = 0;
    {
        auto r = st->pending_rp.find(key);
        if (r != st->pending_rp.end()) {
            rp_bytes = r->second.bytes;
            rp_flags = r->second.flags;
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
        e.text_len = job->text.size();
        e.payload_bytes = job->payload_bytes;
        e.trailer_bytes = job->trailer_bytes;
        e.file_bytes = job->need;
        e.last_used = (uint64_t)time(nullptr);
        e.text_end = (parent.empty() ? 0u : st->segs[parent].text_end) + job->text.size();
        st->used -= rp_bytes;   /* link counts it again, on the entry */
        e.rp_bytes = rp_bytes;
        e.rp_flags = rp_bytes ? rp_flags : 0u;
        st->link(e);
        st->heads[e.key] = job->text.substr(0, HEAD_BYTES);
        st->refresh_dominated(e.parent);
        indexed = true;
    }
    if (!indexed && rp_bytes) {
        ::unlink(st->rp_path(key.c_str()).c_str());
        st->used -= rp_bytes;
        if (st->removed) st->removed(st->removed_ud, key.c_str(), true);
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
    const bool ok = pulsar_segstore_job_write(job, write_payload, write_trailer, ud, err, errlen);
    return pulsar_segstore_commit(st, job, ok) && ok;
}

bool pulsar_segstore_put_rp(pulsar_segstore *st, const char key[41], uint64_t rp_bytes, uint32_t flags,
                            pulsar_segstore_write_fn write_rp, void *ud, char *err, size_t errlen) {
    if (!st || !key || !write_rp || rp_bytes == 0u) {
        snprintf(err, errlen, "restore point put: no store, key, writer or bytes");
        return false;
    }
    const std::string k(key, 40);
    auto it = st->segs.find(k);
    const bool pending = it == st->segs.end() && st->pending.count(k) != 0;
    if (it == st->segs.end() && !pending) {
        snprintf(err, errlen, "restore point put: segment %.40s is not stored", key);
        return false;
    }
    /* the restore point it replaces (if any) leaves the count first: the room is made for the new file alone.  The
     * slot stays valid through eviction, which never takes `key` (unordered_map nodes do not move) */
    uint64_t *slot = pending ? &st->pending_rp[k].bytes : &it->second.rp_bytes;
    uint32_t *slot_flags = pending ? &st->pending_rp[k].flags : &it->second.rp_flags;
    const uint32_t joined = (*slot ? *slot_flags : 0u) | flags;
    const uint64_t need = RP_HDR + rp_bytes;
    if (st->budget && need > st->budget) {
        snprintf(err, errlen, "restore point put: %llu bytes exceed the whole budget", (unsigned long long)need);
        if (pending && *slot == 0u) st->pending_rp.erase(k);
        return false;
    }
    while (st->budget && st->used + st->reserved - *slot + need > st->budget)
        if (!st->evict_one(key, nullptr)) break;
    if (st->budget && st->used + st->reserved - *slot + need > st->budget) {
        snprintf(err, errlen, "restore point put: no room for %llu bytes", (unsigned long long)need);
        if (pending && *slot == 0u) st->pending_rp.erase(k);
        return false;
    }
    uint8_t h[RP_HDR];
    memset(h, 0, sizeof h);
    put32(h, PULSAR_SEGSTORE_RP_MAGIC);
    put32(h + 4, PULSAR_SEGSTORE_RP_VERSION);
    put32(h + 8, st->identity);
    put32(h + 12, joined);
    memcpy(h + 16, key, 40);
    put64(h + 56, rp_bytes);
    const bool ok = durable_write(st->rp_path(k.c_str()), h, RP_HDR, [&](FILE *fp) {
        return write_part(fp, write_rp, ud, rp_bytes, "restore point", err, errlen);
    }, err, errlen);
    if (!ok) {
        /* a failed write left the old file in place (the rename never happened) */
        if (pending && *slot == 0u) st->pending_rp.erase(k);
        return false;
    }
    st->used = st->used - *slot + need;
    *slot = need;
    *slot_flags = joined;
    return true;
}

bool pulsar_segstore_flag_rp(pulsar_segstore *st, const char key[41], uint32_t flags) {
    if (!st) return false;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end() || it->second.rp_bytes == 0u) return false;
    const uint32_t joined = it->second.rp_flags | flags;
    if (joined == it->second.rp_flags) return true;
    FILE *fp = fopen(st->rp_path(it->first.c_str()).c_str(), "r+b");
    if (!fp) return false;
    uint8_t b[4];
    put32(b, joined);
    const bool ok = fseek(fp, 12, SEEK_SET) == 0 && fwrite(b, 1, 4, fp) == 4;
    const bool closed = fclose(fp) == 0;
    if (!ok || !closed) return false;
    it->second.rp_flags = joined;
    return true;
}

int pulsar_segstore_drop_superseded_rps(pulsar_segstore *st, const char key[41]) {
    if (!st) return 0;
    std::string cur(key, 40);
    if (!st->segs.count(cur)) return 0;
    /* may this key's restore point go?  Not flagged, not a branch point, no write in flight */
    auto droppable = [st](const std::string &k) {
        const pulsar_segstore_seg &e = st->segs[k];
        return e.rp_bytes > 0u && e.rp_flags == 0u && st->n_children(k) < 2u && !st->pinned.count(k) &&
               !st->pending.count(k);
    };
    int dropped = 0;
    for (;;) {
        const std::string parent = st->segs[cur].parent;
        /* an older tail of this conversation under the same parent: a leaf whose text is a byte prefix of cur's (so
         * dominated, the index's cheap test first; then the texts) */
        std::string cur_text;
        bool have_cur = false;
        auto sib = st->children.find(parent);
        if (sib != st->children.end())
            for (const std::string &c : sib->second) {
                const pulsar_segstore_seg &e = st->segs[c];
                if (c == cur || !e.dominated || st->n_children(c) != 0u || !droppable(c) ||
                    e.text_len >= st->segs[cur].text_len)
                    continue;
                if (!have_cur && !(have_cur = st->read_text(cur, &cur_text))) break;
                std::string t;
                if (!st->read_text(c, &t) || memcmp(t.data(), cur_text.data(), t.size()) != 0) continue;
                st->drop_rp_of(st->segs[c], "superseded by a later end of its chain");
                dropped++;
            }
        if (parent.empty()) break;
        if (droppable(parent)) {
            st->drop_rp_of(st->segs[parent], "superseded by a later end of its chain");
            dropped++;
        }
        cur = parent;
    }
    return dropped;
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

FILE *pulsar_segstore_open_rp(pulsar_segstore *st, const char key[41], uint64_t *bytes) {
    if (!st) return nullptr;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end() || it->second.rp_bytes == 0u) return nullptr;
    FILE *fp = fopen(st->rp_path(it->first.c_str()).c_str(), "rb");
    if (!fp) return nullptr;
    char k[41];
    uint64_t payload = 0;
    uint32_t flags = 0;
    bool older = false;
    if (!st->read_rp_header(fp, k, &payload, &flags, &older) || memcmp(k, it->first.data(), 40) != 0 ||
        RP_HDR + payload != it->second.rp_bytes) {
        fclose(fp);
        return nullptr;
    }
    if (bytes) *bytes = payload;
    return fp;
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

/* The bytes of [off, off + len) of an open file that sit in the page cache, by mincore over a read-only mapping of
 * the pages that span it (pages straddling the range's ends count by their overlap). */
/// Page-cache residency of a byte range of an open file (see the comment above).
/// @param fd  the open file
/// @param off the range's first byte
/// @param len the range's length in bytes
/// @return bytes of the range resident in the page cache; 0 when the mapping fails
static uint64_t segstore_resident(int fd, uint64_t off, uint64_t len) {
    if (len == 0u) return 0u;
    const long pg_l = sysconf(_SC_PAGESIZE);
    const uint64_t pg = pg_l > 0 ? (uint64_t)pg_l : 4096u;
    const uint64_t a0 = off / pg * pg, a1 = (off + len + pg - 1u) / pg * pg;
    void *map = mmap(nullptr, (size_t)(a1 - a0), PROT_READ, MAP_SHARED, fd, (off_t)a0);
    if (map == MAP_FAILED) return 0u;
    std::vector<unsigned char> vec((size_t)((a1 - a0) / pg));
    uint64_t got = 0u;
    if (mincore(map, (size_t)(a1 - a0), vec.data()) == 0)
        for (size_t i = 0; i < vec.size(); i++) {
            if (!(vec[i] & 1u)) continue;
            const uint64_t p0 = a0 + (uint64_t)i * pg, p1 = p0 + pg;
            const uint64_t lo = p0 > off ? p0 : off, hi = p1 < off + len ? p1 : off + len;
            got += hi > lo ? hi - lo : 0u;
        }
    munmap(map, (size_t)(a1 - a0));
    return got;
}

uint64_t pulsar_segstore_cached_bytes(pulsar_segstore *st, const char key[41], uint64_t payload_len, bool trailer,
                                      bool rp) {
    if (!st) return 0u;
    auto it = st->segs.find(std::string(key, 40));
    if (it == st->segs.end()) return 0u;
    const pulsar_segstore_seg &e = it->second;
    const int fd = open(st->path(it->first.c_str()).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0u;
    const uint64_t at = HDR + e.text_len;
    uint64_t got = segstore_resident(fd, at, payload_len < e.payload_bytes ? payload_len : e.payload_bytes);
    if (trailer) got += segstore_resident(fd, at + e.payload_bytes, e.trailer_bytes);
    close(fd);
    if (rp && e.rp_bytes) {   /* L347: the restore point file, whole */
        const int rfd = open(st->rp_path(it->first.c_str()).c_str(), O_RDONLY | O_CLOEXEC);
        if (rfd >= 0) {
            got += segstore_resident(rfd, 0, e.rp_bytes);
            close(rfd);
        }
    }
    return got;
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
    for (auto &kv : st->segs) {
        const pulsar_segstore_seg &e = kv.second;
        if (!e.trailer_bytes) continue;
        FILE *fp = fopen(st->path(kv.first.c_str()).c_str(), "rb");
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

uint64_t pulsar_segstore_used_bytes(const pulsar_segstore *st) { return st ? st->used : 0u; }
int pulsar_segstore_count(const pulsar_segstore *st) { return st ? (int)st->segs.size() : 0; }

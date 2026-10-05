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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace {

/* Fixed header, little-endian:
 *   0 magic, 4 version, 8 model_id, 12 flags, 16 key[40], 56 parent[40],
 *   96 G_prev, 100 G, 104 text_len u64, 112 payload_bytes u64,
 *   120 trailer_bytes u64, 128 hits u32, 132 pad, 136 created u64,
 *   144 last_used u64  -> 152 bytes. */
constexpr size_t HDR = 152;

void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
uint32_t get32(const uint8_t *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
uint64_t get64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }

bool hex40(const char *s) {
    for (int i = 0; i < 40; i++) {
        const char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

}  // namespace

struct pulsar_segstore {
    std::string dir;
    uint64_t budget = 0;
    uint32_t model_id = 0;
    pulsar_segstore_log_fn log = nullptr;
    void *log_ud = nullptr;
    pulsar_segstore_removed_fn removed = nullptr;
    void *removed_ud = nullptr;
    std::unordered_map<std::string, pulsar_segstore_seg> segs;
    std::unordered_map<std::string, std::vector<std::string>> children;   ///< parent key ("" = roots) -> child keys
    uint64_t used = 0;

    void logf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
        if (!log) return;
        char msg[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        log(log_ud, msg);
    }
    std::string path(const char *key) const { return dir + "/" + key + ".seg"; }

    bool read_header(FILE *fp, pulsar_segstore_seg *e, uint32_t *flags) const {
        uint8_t h[HDR];
        if (fread(h, 1, HDR, fp) != HDR) return false;
        if (get32(h) != PULSAR_SEGSTORE_MAGIC || get32(h + 4) != PULSAR_SEGSTORE_VERSION) return false;
        if (get32(h + 8) != model_id) return false;
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

    void link(const pulsar_segstore_seg &e) {
        segs[e.key] = e;
        children[e.parent].push_back(e.key);
        used += e.file_bytes;
    }
    void unlink_index(const std::string &key) {
        auto it = segs.find(key);
        if (it == segs.end()) return;
        if (removed) removed(removed_ud, key.c_str());
        used -= it->second.file_bytes;
        auto &sib = children[it->second.parent];
        for (size_t i = 0; i < sib.size(); i++)
            if (sib[i] == key) { sib.erase(sib.begin() + (long)i); break; }
        children.erase(key);
        segs.erase(it);
    }
    /* A segment whose parent is gone cannot be reached by any lookup. */
    void prune_orphans() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = segs.begin(); it != segs.end(); ++it) {
                if (it->second.parent[0] && !segs.count(it->second.parent)) {
                    const std::string k = it->first;
                    logf("segment %s dropped: its parent %s is gone", k.c_str(), it->second.parent);
                    ::unlink(path(k.c_str()).c_str());
                    unlink_index(k);
                    changed = true;
                    break;
                }
            }
        }
    }
    /* The chain text lengths are a property of the chain, derived once. */
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
    /* LRU among the leaves, never `keep`.  false when none can go. */
    bool evict_one(const char *keep) {
        const pulsar_segstore_seg *victim = nullptr;
        for (auto &kv : segs) {
            if (keep && kv.first == keep) continue;
            auto ch = children.find(kv.first);
            if (ch != children.end() && !ch->second.empty()) continue;
            if (!victim || kv.second.last_used < victim->last_used) victim = &kv.second;
        }
        if (!victim) return false;
        const std::string k = victim->key;
        logf("evicted segment %s [%u,%u) %llu bytes: store over budget", k.c_str(), victim->G_prev, victim->G,
             (unsigned long long)victim->file_bytes);
        ::unlink(path(k.c_str()).c_str());
        unlink_index(k);
        return true;
    }
};

pulsar_segstore *pulsar_segstore_open(const char *dir, uint64_t budget_bytes, uint32_t model_id,
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
    st->model_id = model_id;
    st->log = log;
    st->log_ud = log_ud;
    DIR *d = opendir(dir);
    if (!d) { delete st; return nullptr; }
    while (const struct dirent *de = readdir(d)) {
        const size_t n = strlen(de->d_name);
        /* A writer that died between tmp and rename leaves "<key>.seg.tmp.<pid>":
         * no live writer exists at open, so every such file is an orphan. */
        if (strstr(de->d_name, ".seg.tmp.")) { ::unlink((st->dir + "/" + de->d_name).c_str()); continue; }
        if (n != 44 || strcmp(de->d_name + 40, ".seg") != 0 || !hex40(de->d_name)) continue;
        const std::string p = st->dir + "/" + de->d_name;
        FILE *fp = fopen(p.c_str(), "rb");
        if (!fp) continue;
        pulsar_segstore_seg e;
        uint32_t flags = 0;
        const bool ok = st->read_header(fp, &e, &flags) && !memcmp(e.key, de->d_name, 40);
        struct stat sb;
        const bool sized = ok && fstat(fileno(fp), &sb) == 0 &&
                           (uint64_t)sb.st_size == HDR + e.text_len + e.payload_bytes + e.trailer_bytes;
        fclose(fp);
        if (!sized) continue;
        e.file_bytes = (uint64_t)sb.st_size;
        st->link(e);
    }
    closedir(d);
    st->prune_orphans();
    st->derive_text_ends();
    st->logf("segment store %s: %zu segments, %llu MiB (budget %llu MiB)", dir, st->segs.size(),
             (unsigned long long)(st->used >> 20), (unsigned long long)(budget_bytes >> 20));
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

int pulsar_segstore_lookup(pulsar_segstore *st, const char *text, size_t len,
                           pulsar_segstore_seg *chain, int cap) {
    if (!st || !text || cap <= 0) return 0;
    /* Depth-first over the children whose key the request's own bytes
     * reproduce; the deepest chain wins. */
    struct frame { std::string key; uint64_t end; int depth; };
    std::vector<frame> stack;
    std::vector<std::string> path(cap), best_path;
    int best_depth = 0;
    uint64_t best_end = 0;
    stack.push_back({"", 0, 0});
    while (!stack.empty()) {
        const frame f = stack.back();
        stack.pop_back();
        if (f.depth > 0) path[f.depth - 1] = f.key;
        if (f.depth > best_depth || (f.depth == best_depth && f.end > best_end)) {
            best_depth = f.depth;
            best_end = f.end;
            best_path.assign(path.begin(), path.begin() + f.depth);
        }
        if (f.depth >= cap) continue;
        auto ch = st->children.find(f.key);
        if (ch == st->children.end()) continue;
        for (const std::string &c : ch->second) {
            const pulsar_segstore_seg &s = st->segs[c];
            if (f.end + s.text_len > len) continue;
            char k[41];
            pulsar_segstore_child_key(f.key.c_str(), text + f.end, s.text_len, k);
            if (c == k) stack.push_back({c, f.end + s.text_len, f.depth + 1});
        }
    }
    for (int i = 0; i < best_depth; i++) chain[i] = st->segs[best_path[i]];
    return best_depth;
}

bool pulsar_segstore_contains(pulsar_segstore *st, const char key[41]) {
    return st && st->segs.count(std::string(key, 40)) != 0;
}

static int write_all(FILE *fp, const void *p, size_t n) { return fwrite(p, 1, n, fp) == n ? 0 : 1; }

bool pulsar_segstore_put(pulsar_segstore *st, const char *parent, uint32_t G_prev, uint32_t G,
                         const char *text, size_t text_len,
                         uint64_t payload_bytes, pulsar_segstore_write_fn write_payload,
                         uint64_t trailer_bytes, pulsar_segstore_write_fn write_trailer,
                         void *ud, char key_out[41], char *err, size_t errlen) {
    if (!st || !write_payload || G <= G_prev || (G_prev == 0) != (!parent || !parent[0])) {
        snprintf(err, errlen, "segment put: a root starts at 0 and only a root has no parent");
        return false;
    }
    if (parent && parent[0] && !pulsar_segstore_contains(st, parent)) {
        snprintf(err, errlen, "segment put: parent %.40s is not stored", parent);
        return false;
    }
    char key[41];
    pulsar_segstore_child_key(parent, text, text_len, key);
    memcpy(key_out, key, 41);
    if (pulsar_segstore_contains(st, key)) return true;   /* the same text is the same segment */
    const uint64_t need = HDR + text_len + payload_bytes + (write_trailer ? trailer_bytes : 0u);
    if (st->budget && need > st->budget) {
        snprintf(err, errlen, "segment put: %llu bytes exceed the whole budget", (unsigned long long)need);
        return false;
    }
    while (st->budget && st->used + need > st->budget)
        if (!st->evict_one(parent)) break;
    if (st->budget && st->used + need > st->budget) {
        snprintf(err, errlen, "segment put: no room (only the chain being extended remains)");
        return false;
    }
    char tmp[64];
    snprintf(tmp, sizeof tmp, ".tmp.%ld", (long)getpid());
    const std::string final_path = st->path(key), tmp_path = final_path + tmp;
    FILE *fp = fopen(tmp_path.c_str(), "wb");
    if (!fp) { snprintf(err, errlen, "segment put: %s: %s", tmp_path.c_str(), strerror(errno)); return false; }
    pulsar_writeback wb;
    pulsar_writeback_init(&wb, fp);
    uint8_t h[HDR];
    memset(h, 0, sizeof h);
    const uint64_t now = (uint64_t)time(nullptr);
    put32(h, PULSAR_SEGSTORE_MAGIC);
    put32(h + 4, PULSAR_SEGSTORE_VERSION);
    put32(h + 8, st->model_id);
    put32(h + 12, write_trailer ? PULSAR_SEGSTORE_FLAG_TRAILER : 0u);
    memcpy(h + 16, key, 40);
    if (parent && parent[0]) memcpy(h + 56, parent, 40);
    put32(h + 96, G_prev);
    put32(h + 100, G);
    put64(h + 104, text_len);
    put64(h + 112, payload_bytes);
    put64(h + 120, write_trailer ? trailer_bytes : 0u);
    put32(h + 128, 0);
    put64(h + 136, now);
    put64(h + 144, now);
    int rc = write_all(fp, h, HDR) || write_all(fp, text, text_len);
    if (rc == 0) {
        const long at = ftell(fp);
        rc = write_payload(fp, ud, err, errlen);
        if (rc == 0 && (uint64_t)(ftell(fp) - at) != payload_bytes) {
            snprintf(err, errlen, "segment put: payload wrote %ld bytes, promised %llu", ftell(fp) - at,
                     (unsigned long long)payload_bytes);
            rc = 1;
        }
    }
    if (rc == 0 && write_trailer) {
        const long at = ftell(fp);
        rc = write_trailer(fp, ud, err, errlen);
        if (rc == 0 && (uint64_t)(ftell(fp) - at) != trailer_bytes) {
            snprintf(err, errlen, "segment put: trailer size mismatch");
            rc = 1;
        }
    }
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
    {
        const int dfd = open(st->dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) { (void)fsync(dfd); close(dfd); }
    }
    pulsar_segstore_seg e;
    memset(&e, 0, sizeof e);
    memcpy(e.key, key, 41);
    if (parent && parent[0]) memcpy(e.parent, parent, 40);
    e.G_prev = G_prev;
    e.G = G;
    e.text_len = text_len;
    e.payload_bytes = payload_bytes;
    e.trailer_bytes = write_trailer ? trailer_bytes : 0u;
    e.file_bytes = need;
    e.last_used = now;
    e.text_end = (parent && parent[0] ? st->segs[parent].text_end : 0u) + text_len;
    st->link(e);
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
        ::unlink(st->path(k.c_str()).c_str());
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

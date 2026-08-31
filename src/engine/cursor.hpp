#ifndef PULSAR_CURSOR_HPP
#define PULSAR_CURSOR_HPP

#include "pulsar_engine_internal.h"

/* Bounds-checked byte cursor over the mmapped GGUF file (C++ port of the
 * pulsar_cursor free functions). The state stays in the C pulsar_cursor struct so
 * the still-C gguf.c can hold and pass cursors; this class is a typed view
 * over one. When gguf.c ports, the struct folds into the class. */

namespace pulsar {

/** Bounds-checked sequential reader over the mmapped GGUF file.
 *
 * A typed VIEW over a ::pulsar_cursor, not an owner: the state lives in the C
 * struct so the still-C loader can hold and pass cursors, and a Cursor is
 * constructed around one wherever typed reads are wanted.
 *
 * Every accessor bounds-checks first and returns false rather than reading out
 * of range. A failure does NOT disable the cursor -- each call reports its own
 * result and callers check each one. What is first-wins is the error MESSAGE,
 * so it names the earliest failure and the offset it happened at.
 */
class Cursor {
public:
    /** Wrap an existing cursor. The reference must outlive this view. */
    explicit Cursor(pulsar_cursor &c) : c_(c) {}

    /** Record `msg` with the current byte offset, but only if no error has been
     * recorded yet -- the first failure is the informative one. */
    void set_error(const char *msg) {
        if (c_.error[0] == '\0') {
            snprintf(c_.error, sizeof(c_.error), "%s at byte %" PRIu64, msg, c_.pos);
        }
    }

    /** Are `n` more bytes available? Records a truncation error if not.
     * Written to be overflow-safe: `pos > size - n` rather than `pos + n > size`. */
    bool has(uint64_t n) {
        if (n > c_.size || c_.pos > c_.size - n) {
            set_error("truncated GGUF file");
            return false;
        }
        return true;
    }

    /** Copy `n` bytes to `dst` and advance. @return false if out of range,
     * leaving `pos` unmoved. */
    bool read(void *dst, uint64_t n) {
        if (!has(n)) return false;
        memcpy(dst, c_.base + c_.pos, (size_t)n);
        c_.pos += n;
        return true;
    }

    /** Advance `n` bytes without reading them. @return false if out of range. */
    bool skip(uint64_t n) {
        if (!has(n)) return false;
        c_.pos += n;
        return true;
    }

    /** Read a little-endian uint32. */
    bool u32(uint32_t *v) { return read(v, sizeof(*v)); }
    /** Read a little-endian uint64. */
    bool u64(uint64_t *v) { return read(v, sizeof(*v)); }

    /** Read a length-prefixed GGUF string. `s` is left pointing INTO the
     * mapping -- borrowed, not copied, and valid only while the mapping is. */
    bool string(pulsar_str *s) {
        uint64_t len;
        if (!u64(&len)) return false;
        if (!has(len)) return false;
        s->ptr = (const char *)(c_.base + c_.pos);
        s->len = len;
        c_.pos += len;
        return true;
    }

private:
    pulsar_cursor &c_;  ///< the wrapped cursor state; borrowed
};

} // namespace pulsar

#endif /* PULSAR_CURSOR_HPP */

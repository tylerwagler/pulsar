#ifndef PULSAR_CURSOR_HPP
#define PULSAR_CURSOR_HPP

#include "pulsar_engine_internal.h"

/* Bounds-checked byte cursor over a metadata VALUE blob.  The state lives in
 * the pulsar_cursor struct so it can be held and passed as a value; this class
 * is a typed view over one, and the facade in cursor.cpp is its C entry. */

namespace pulsar {

/** Bounds-checked sequential reader over a metadata value blob.
 *
 * A typed VIEW over a ::pulsar_cursor, not an owner: the state lives in the
 * struct so a loader can hold and pass cursors, and a Cursor is constructed
 * around one wherever typed reads are wanted.
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
            set_error("truncated metadata value");
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

    /** Read a little-endian uint32. */
    bool u32(uint32_t *v) { return read(v, sizeof(*v)); }
    /** Read a little-endian uint64. */
    bool u64(uint64_t *v) { return read(v, sizeof(*v)); }

    /** Read a length-prefixed string. `s` is left pointing into the value blob
     * -- borrowed, not copied, and valid only while the blob is. */
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

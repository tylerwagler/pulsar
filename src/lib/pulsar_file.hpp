#pragma once

#include <cstdio>

/* RAII owner for a C FILE* (C++ port). Closes the handle on scope exit so that
 * early-return error paths cannot leak it — the recurring bug in the hand-rolled
 * "fclose only on the success path" idiom (e.g. `ok = nw == n && fclose(fp) == 0`,
 * which short-circuits past fclose on a short read/write).
 *
 * For files whose close-time flush error must be observed (write paths), call
 * close() explicitly and check its return — that returns fclose()'s status and
 * disarms the destructor, so behaviour is identical to the manual fclose while
 * the destructor remains a leak safety-net on the paths that forget it.
 *
 * Move-only; no exceptions (the project builds -fno-exceptions). */

namespace pulsar {

/** RAII owner for a C `FILE *` -- see the note above for the bug class it
 * exists to close.
 *
 * @warning On a WRITE path the destructor is a safety net, not the close you
 * want: it discards fclose()'s status, and that is where a buffered write's
 * flush error is reported. Call close() and check it. On read paths the
 * destructor is sufficient.
 */
class FileHandle {
public:
    /** Construct empty; holds no file. */
    FileHandle() = default;
    /** Take ownership of `fp`, which may be NULL. */
    explicit FileHandle(FILE *fp) : fp_(fp) {}
    /** Close the handle if still open, discarding any flush error. */
    ~FileHandle() { if (fp_) fclose(fp_); }

    FileHandle(const FileHandle &) = delete;
    FileHandle &operator=(const FileHandle &) = delete;
    /** Move-construct, leaving `o` empty. */
    FileHandle(FileHandle &&o) noexcept : fp_(o.fp_) { o.fp_ = nullptr; }
    /** Move-assign, closing any handle already held. */
    FileHandle &operator=(FileHandle &&o) noexcept {
        if (this != &o) { if (fp_) fclose(fp_); fp_ = o.fp_; o.fp_ = nullptr; }
        return *this;
    }

    /** The owned handle, or NULL. Borrowed -- do not fclose it. */
    FILE *get() const { return fp_; }
    /** True when a file is held. */
    explicit operator bool() const { return fp_ != nullptr; }

    /* Explicit close; returns fclose()'s status (0 on success). Idempotent: the
     * destructor will not double-close after this. */
    /** @return fclose()'s status (0 on success), or 0 if nothing was held. */
    int close() {
        if (!fp_) return 0;
        int rc = fclose(fp_);
        fp_ = nullptr;
        return rc;
    }

private:
    FILE *fp_ = nullptr;  ///< the owned handle; NULL once closed or moved from
};

} // namespace pulsar

#include "cursor.hpp"

/* C facade over pulsar::Cursor.  The cursor reads the metadata VALUES the
 * container declared, from the value blob the loader built. */

bool cursor_read(pulsar_cursor *c, void *dst, uint64_t n) {
    return pulsar::Cursor(*c).read(dst, n);
}



bool cursor_u32(pulsar_cursor *c, uint32_t *v) {
    return pulsar::Cursor(*c).u32(v);
}



bool cursor_u64(pulsar_cursor *c, uint64_t *v) {
    return pulsar::Cursor(*c).u64(v);
}



bool cursor_string(pulsar_cursor *c, pulsar_str *s) {
    return pulsar::Cursor(*c).string(s);
}

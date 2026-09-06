#ifndef PULSAR_CTXMEM_H
#define PULSAR_CTXMEM_H

#include <stdint.h>
#include <stdio.h>
#include "pulsar.h"

/** The one "context buffers" line every front end prints after engine open.
 *
 * Formats pulsar_context_memory_estimate() for `ctx_size` into `buf` as
 *
 *   `<prog>: context buffers 2.94 GiB (ctx=1048576, backend=cuda,
 *    prefill_chunk=4096, kv=2.44 GiB [raw 0.07 + comp/idx 2.38],
 *    scratch 0.50 GiB, raw_kv_rows=4352, compressed_kv_rows=262146)`
 *
 * with no trailing newline, so the caller hands it to its own sink (stderr,
 * server_log).  Call after pulsar_engine_open(): the estimate reads the
 * loaded model's shape and compress ratios.
 *
 * @param buf            output buffer
 * @param cap            bytes available in `buf` (the line fits in 256)
 * @param prog           program-name prefix, e.g. "pulsar-server"
 * @param backend        the engine backend the estimate is for
 * @param ctx_size       context size in tokens
 * @param prefill_chunk  prefill chunk width; 0 selects the engine default
 * @return `buf`
 */
static inline const char *pulsar_context_memory_line(char *buf,
                                                     size_t cap,
                                                     const char *prog,
                                                     pulsar_backend backend,
                                                     int ctx_size,
                                                     uint32_t prefill_chunk) {
    const pulsar_context_memory m =
        pulsar_context_memory_estimate(backend, ctx_size, prefill_chunk);
    const double GiB = 1073741824.0;
    snprintf(buf, cap,
             "%s: context buffers %.2f GiB (ctx=%d, backend=%s, prefill_chunk=%u, "
             "kv=%.2f GiB [raw %.2f + comp/idx %.2f], scratch %.2f GiB, "
             "raw_kv_rows=%u, compressed_kv_rows=%u)",
             prog,
             (double)m.total_bytes / GiB,
             ctx_size,
             pulsar_backend_name(backend),
             m.prefill_cap,
             (double)(m.raw_bytes + m.comp_index_bytes) / GiB,
             (double)m.raw_bytes / GiB,
             (double)m.comp_index_bytes / GiB,
             (double)m.scratch_bytes / GiB,
             m.raw_cap,
             m.comp_cap);
    return buf;
}

#endif

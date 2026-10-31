/** @file gpu_graph_alloc.cpp
 *  @brief DeepSeek graph allocation helpers: raw-ring spans per batch and env flags. */
#include "pulsar_engine_internal.h"









/// The raw-window rows a batch attends: its own plus the sliding window, capped by the positions so far and raw_cap.
/// @param g the graph
/// @param pos0 the batch's first position
/// @param n_tokens rows
/// @return rows (0 with no raw ring or no rows)
uint32_t gpu_graph_raw_span_for_batch(
        const pulsar_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : PULSAR_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}



/// @param g the graph
/// @param last_pos the span's last position
/// @param n_raw its raw rows
/// @return the ring row of the span's first raw position
uint32_t gpu_graph_raw_start_for_span(
        const pulsar_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}






/* =========================================================================
 * GPU Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The generation path uses the fused helpers below.  (The unfused reference
 * arms and their PULSAR_CUDA_DISABLE_*_FUSION switches were deleted in the
 * 133->108 flag purge -- an opt-out with no real caller gets deleted; this
 * header said otherwise until L106 K4.)
 */

/// An environment flag, read once into *cache: set, non-empty and not "0".
/// @param name the variable
/// @param cache the caller's cache (-1 = unread)
/// @return the flag
bool gpu_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}


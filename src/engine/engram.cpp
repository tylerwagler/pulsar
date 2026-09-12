/* Engram n-gram hashing, host side (PLAN 95 phase 4 / L218 phase 0c).
 *
 * Mirrors engram.py's `NgramHashState.forward` exactly.  The reference is
 * gate-baseline/l218-v41/engram_hash.py and the layout it emitted
 * (engram-layout-v41.json); tests/engram_hash_test.cpp pins this against vectors
 * generated from that layout for both engram layers.
 *
 * The host is the right side of the boundary for this: the ids are known before
 * the forward, so every bucket row a token needs is computable ahead of the GPU
 * work.  That prefetchability is the whole reason a ~189 GiB disk-resident table
 * is viable on one box.
 *
 * The arithmetic is deliberately plain `%` on non-negative int64 -- see the
 * invariant argued at the declaration in pulsar_engine_internal.h.  The one
 * thing this file must not do is let a corrupt compressed id through, because
 * a negative `tok` would set the sign bit of a product and the XOR below would
 * then produce a row index that is still in range but points at the wrong
 * bucket.  Wrong-and-plausible is the failure mode worth refusing. */

#include "pulsar_engine_internal.h"

void pulsar_engram_hash_pos(const pulsar_engram_layout *L, uint32_t layer,
                            const int32_t *ids, uint32_t n_ids, uint32_t pos,
                            uint32_t *cols) {
    /* Every one of these is a caller bug, and returning quietly would leave
     * `cols` uninitialised -- the caller would then read stale row indices that
     * still look like valid rows.  Refuse instead. */
    if (!L || !cols || !ids || L->n_vocab == 0 || L->compressed_vocab == 0 ||
        layer >= L->n_layers || layer >= PULSAR_ENGRAM_MAX_LAYERS || pos >= n_ids) {
        pulsar_die("engram hash: called with an out-of-range layout, layer or position");
    }
    const int64_t  *mult   = L->multipliers + (uint64_t)layer * PULSAR_ENGRAM_MAX_NGRAM;
    const uint32_t *primes = L->primes + (uint64_t)layer * (PULSAR_ENGRAM_MAX_NGRAM - 1u) * PULSAR_ENGRAM_N_HEADS;
    const uint64_t *offs   = L->offsets + (uint64_t)layer * PULSAR_ENGRAM_N_COLS;

    /* The n-gram window is the token at `pos` and the (max_ngram-1) before it;
     * positions before the sequence starts take the pad id.  `pos < i` is also
     * what keeps `pos - i` from wrapping -- both are uint32_t, so the pad branch
     * must stay on top of the subtraction rather than beside it. */
    int64_t rolling = 0;
    for (uint32_t i = 0; i < PULSAR_ENGRAM_MAX_NGRAM; i++) {
        int64_t tok;
        if (pos < i) {
            tok = (int64_t)L->pad_compressed_id;
        } else {
            const int32_t tid = ids[pos - i];
            if (tid < 0 || (uint32_t)tid >= L->n_vocab) {
                pulsar_die("engram hash: token id outside the layout's token map");
            }
            tok = (int64_t)L->token_map[tid];
        }
        if (tok < 0 || (uint64_t)tok >= L->compressed_vocab) {
            pulsar_die("engram hash: compressed id outside the layout's compressed vocab");
        }
        const int64_t prod = tok * mult[i];
        if (i == 0u) { rolling = prod; continue; }
        rolling ^= prod;
        const uint32_t base = (i - 1u) * PULSAR_ENGRAM_N_HEADS;
        for (uint32_t h = 0; h < PULSAR_ENGRAM_N_HEADS; h++) {
            cols[base + h] = (uint32_t)(rolling % (int64_t)primes[base + h]) +
                             (uint32_t)offs[base + h];
        }
    }
}

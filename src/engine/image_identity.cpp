/* image_identity.cpp -- L281: which images a session's KV holds, block by block, for every family.
 *
 * A family's image blocks are runs of sentinel ids whose VALUES only geometry decides (DeepSeek: vocab_size + role),
 * so two different images of one size give an identical token prefix.  What the KV holds is therefore said by a
 * record per block -- where it starts, where it ends (exclusive), and a hash of the image bytes merged there -- and
 * every reuse decision compares records, never token ids alone:
 *
 *   - the licence (a sync that extends live history admits its held images only when the live records are exactly
 *     the request's images whose blocks end at or below the shared prefix -- session.cpp);
 *   - an interrupted image prefill records the blocks it completed, so the next quantum resumes (L281 (b));
 *   - a rewind keeps the records of the blocks that survive it (before L281 it dropped the whole identity when one
 *     block was cut: L273's "known conservative case");
 *   - a disk chain persists past a block only when its record is known, carries the records of its span in the
 *     segment payload, and a restore loads past a block only when the request brings the same image there
 *     (L281 (c)).
 *
 * The family supplies the geometry (pulsar_family::vision: which ids are sentinels, how long the block starting at a
 * position is); a family with no images has no hook and no id is a sentinel.  Everything else is here, once. */
#include "pulsar_engine_internal.h"

static uint64_t fnv_u64(uint64_t h, uint64_t v) {
    for (int b = 0; b < 8; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 1099511628211ull; }
    return h;
}

uint64_t pulsar_image_content_hash(const pulsar_image_ref *img) {
    uint64_t h = fnv_u64(1469598103934665603ull, img && img->bytes ? (uint64_t)img->len : 0u);
    if (img && img->bytes)
        for (size_t i = 0; i < img->len; i++) { h ^= img->bytes[i]; h *= 1099511628211ull; }
    return h ? h : 1u;
}

static bool is_sentinel(const pulsar_family_vision *v, const pulsar_engine *e, int32_t id) {
    return v && v->is_sentinel(e, id);
}
static bool block_extent(const pulsar_family_vision *v, const pulsar_engine *e, const int32_t *ids, int n, int start,
                         int *len) {
    *len = 0;
    return v && v->block_extent(e, ids, n, start, len) && *len > 0;
}

bool pulsar_image_is_sentinel(const pulsar_engine *e, int32_t id) {
    return is_sentinel(e && e->family ? e->family->vision : NULL, e, id);
}

bool pulsar_image_block_extent(const pulsar_engine *e, const int32_t *ids, int n, int start, int *len) {
    return block_extent(e && e->family ? e->family->vision : NULL, e, ids, n, start, len);
}

bool pulsar_image_identity_build(const pulsar_family_vision *v, const pulsar_engine *e, const int32_t *ids, int n,
                                 const pulsar_image_ref *images, int n_images, uint32_t limit,
                                 pulsar_image_identity *out) {
    out->n = 0;
    for (int i = 0; i < n_images; i++) {
        int len = 0;
        if (!block_extent(v, e, ids, n, images[i].start_pos, &len)) {
            fprintf(stderr, "pulsar: image identity: image %d at %d names no image block\n", i, images[i].start_pos);
            return false;
        }
        const uint32_t s = (uint32_t)images[i].start_pos, end = s + (uint32_t)len;
        if (end > limit) continue;
        if (out->n == PULSAR_IMAGE_BLOCKS_MAX) {
            fprintf(stderr, "pulsar: image identity: more than %d image blocks\n", PULSAR_IMAGE_BLOCKS_MAX);
            return false;
        }
        if (out->n > 0 && s < out->b[out->n - 1].end) {
            fprintf(stderr, "pulsar: image identity: image %d's block at %u overlaps or precedes the last at %u\n", i,
                    s, out->b[out->n - 1].start);
            return false;
        }
        out->b[out->n++] = (pulsar_image_block){ s, end, pulsar_image_content_hash(&images[i]) };
    }
    return true;
}

void pulsar_image_identity_trim(pulsar_image_identity *id, uint32_t pos) {
    while (id->n > 0 && id->b[id->n - 1].end > pos) id->n--;
}

bool pulsar_image_identity_equal(const pulsar_image_identity *a, const pulsar_image_identity *b) {
    if (a->n != b->n) return false;
    for (uint32_t i = 0; i < a->n; i++)
        if (a->b[i].start != b->b[i].start || a->b[i].end != b->b[i].end || a->b[i].content != b->b[i].content)
            return false;
    return true;
}

uint32_t pulsar_image_identity_end(const pulsar_image_identity *id) {
    return id->n ? id->b[id->n - 1].end : 0u;
}

int pulsar_image_persist_end(const pulsar_family_vision *v, const pulsar_engine *e, const int32_t *ids, int n,
                             const pulsar_image_identity *id) {
    uint32_t k = 0;   /* the next record the walk expects */
    for (int i = 0; i < n; i++) {
        if (!is_sentinel(v, e, ids[i])) continue;
        int len = 0;
        if (!block_extent(v, e, ids, n, i, &len)) return -1;   /* a malformed block: persist nothing */
        /* a block the identity records (in order, at this place) may be persisted; the first one it does not --
         * rows whose image no record names -- ends the persistable prefix at its start */
        if (k < id->n && id->b[k].start == (uint32_t)i && id->b[k].end == (uint32_t)(i + len)) {
            k++;
            i += len - 1;
            continue;
        }
        return i;
    }
    return n;
}

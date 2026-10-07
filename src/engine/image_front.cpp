/* image_front.cpp -- L268: the image path every family shares.
 *
 * DeepSeek's image work (L216, L226, L261, L273, L281) built the path inside DeepSeek's code; Qwen's tower (L268)
 * would have twinned it.  Everything that is not the family's own lives here once, behind pulsar_family_vision
 * (family.h): the family says where its blocks are, what its renderer writes, how one image becomes block ids, and
 * how its tower turns an image into block rows.  The core does the rest:
 *
 *   - the placeholder expansion: one placeholder token per image, replaced by the image's block (start_pos set);
 *   - the block fit: every image names a block the prompt carries, and every block fits one prefill chunk;
 *   - the reuse licence: when a request's images let the live KV be kept (with image_identity.cpp's records);
 *   - the orphan refusal: a sentinel or placeholder id with no image to fill it;
 *   - the chunk merge: the rows of every block a prefill chunk owns, written through the family's carrier;
 *   - the tower-output cache: an image's tower runs once per process (L226), its position-dependent assembly every
 *     time. */
#include "pulsar_engine_internal.h"

#include <time.h>

static double front_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const pulsar_family_vision *front_of(const pulsar_engine *e) {
    return e && e->family ? e->family->vision : NULL;
}

/* ---- the tower-output cache (L226) --------------------------------------------------------------------------------
 *
 * What is cached is the family tower's output for one image: it depends on the image bytes only (and the family's
 * fixed preprocessing), never on where the block sits -- the position enters in the family's block assembly
 * (DeepSeek's compressor pads and N-layout; Qwen has none), which runs every time and costs memcpys.  So the key is
 * the image content, and a hit skips the tower but still assembles the exact block for this position.
 *
 * Byte-capped FIFO, process-global, no lock: the engine's GPU state is shared and merges are serial by construction
 * (prefill is serial), so the cache inherits the same discipline.  It is cleared when the engine changes --
 * pulsar_test opens several engines in one process, and another model's row would be a silently wrong embedding. */
#define IMAGE_TOWER_CACHE_BYTES (64u * 1024u * 1024u)
typedef struct {
    uint64_t key;
    size_t   src_len;   ///< the image's byte length, checked with the key
    int      rows;      ///< tower rows stored
    size_t   bytes;     ///< their size
    uint16_t *data;
} tower_cache_entry;
static tower_cache_entry g_tower_cache[16];
static int    g_tower_cache_n = 0;
static size_t g_tower_cache_bytes = 0;
static const pulsar_engine *g_tower_cache_owner = NULL;

static void tower_cache_reset(void) {
    for (int i = 0; i < g_tower_cache_n; i++) free(g_tower_cache[i].data);
    g_tower_cache_n = 0;
    g_tower_cache_bytes = 0;
}

static void tower_cache_drop_oldest(void) {
    g_tower_cache_bytes -= g_tower_cache[0].bytes;
    free(g_tower_cache[0].data);
    memmove(&g_tower_cache[0], &g_tower_cache[1], (size_t)(g_tower_cache_n - 1) * sizeof(g_tower_cache[0]));
    g_tower_cache_n--;
}

static const tower_cache_entry *tower_cache_get(uint64_t key, size_t src_len) {
    for (int i = 0; i < g_tower_cache_n; i++)
        if (g_tower_cache[i].key == key && g_tower_cache[i].src_len == src_len) return &g_tower_cache[i];
    return NULL;
}

/* Takes ownership of `data` (freed when it is not kept). */
static void tower_cache_put(uint64_t key, size_t src_len, uint16_t *data, int rows, size_t bytes) {
    if (rows <= 0 || bytes > IMAGE_TOWER_CACHE_BYTES) { free(data); return; }   /* not worth caching */
    while (g_tower_cache_n > 0 && g_tower_cache_bytes + bytes > IMAGE_TOWER_CACHE_BYTES) tower_cache_drop_oldest();
    if (g_tower_cache_n == (int)(sizeof(g_tower_cache) / sizeof(g_tower_cache[0]))) tower_cache_drop_oldest();
    tower_cache_entry *c = &g_tower_cache[g_tower_cache_n++];
    c->key = key;
    c->src_len = src_len;
    c->rows = rows;
    c->bytes = bytes;
    c->data = data;
    g_tower_cache_bytes += bytes;
}

bool pulsar_image_block_rows(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *ids, int block_len,
                             uint16_t *out, bool *cache_hit, char *err, size_t errlen) {
    const pulsar_family_vision *v = front_of(e);
    if (cache_hit) *cache_hit = false;
    if (!v || !v->block_rows || !img || !img->bytes || img->len == 0 || !ids || block_len <= 0 || !out) {
        snprintf(err, errlen, "image block rows: the family serves no images, or the request is incomplete");
        return false;
    }
    if (g_tower_cache_owner != e) {
        tower_cache_reset();
        g_tower_cache_owner = e;
    }
    const uint64_t key = pulsar_image_content_hash(img) ^ (uint64_t)(uintptr_t)e->family;
    const tower_cache_entry *hit = tower_cache_get(key, img->len);
    if (hit) {
        if (cache_hit) *cache_hit = true;
        return v->block_rows(e, img, ids, block_len, hit->data, hit->rows, NULL, NULL, out, err, errlen);
    }
    uint16_t *tower = NULL;
    int n_tower = 0;
    if (!v->block_rows(e, img, ids, block_len, NULL, 0, &tower, &n_tower, out, err, errlen)) {
        free(tower);
        return false;
    }
    if (tower) tower_cache_put(key, img->len, tower, n_tower,
                               (size_t)n_tower * v->row_width(e) * sizeof(uint16_t));
    return true;
}

bool pulsar_image_merge_chunk(const pulsar_engine *e, const int32_t *ids, int n_ids, const pulsar_image_ref *images,
                              int n_images, uint32_t pos0, uint32_t n_tokens, pulsar_image_row_writer write,
                              void *ud) {
    if (n_images <= 0) return true;   /* text: nothing to do */
    const pulsar_family_vision *v = front_of(e);
    if (!v || !ids || n_ids <= 0 || !images || !write) return false;
    const uint32_t width = v->row_width(e);
    for (int i = 0; i < n_images; i++) {
        const pulsar_image_ref *img = &images[i];
        if (!img->bytes || img->len == 0 || img->start_pos < 0) return false;
        int block_len = 0;
        if (!pulsar_image_block_extent(e, ids, n_ids, img->start_pos, &block_len)) {
            fprintf(stderr, "pulsar: image %d claims a block at token %d that the prompt does not carry\n", i,
                    img->start_pos);
            return false;
        }
        const uint32_t s0 = (uint32_t)img->start_pos;
        if (s0 < pos0 || s0 + (uint32_t)block_len > pos0 + n_tokens) continue;   /* another chunk owns it */
        uint16_t *rows = (uint16_t *)malloc((size_t)block_len * width * sizeof(uint16_t));
        if (!rows) return false;
        char err[256] = "";
        bool hit = false;
        const double t0 = front_now_s();
        const bool ok = pulsar_image_block_rows(e, img, ids + s0, block_len, rows, &hit, err, sizeof err) &&
                        write(ud, rows, (uint32_t)block_len, s0 - pos0, n_tokens);
        free(rows);
        fprintf(stderr, "pulsar: image %d: %s %.1f ms (%d block rows)\n", i,
                hit ? "CACHED tower (encode skipped)" : "tower encode+assemble", (front_now_s() - t0) * 1000.0,
                block_len);
        if (!ok) {
            fprintf(stderr, "pulsar: image %d at token %d failed to merge (block %d rows)%s%s\n", i, img->start_pos,
                    block_len, err[0] ? ": " : "", err);
            return false;
        }
    }
    return true;
}

bool pulsar_image_spans_fit(const pulsar_engine *e, const int32_t *ids, int n, const pulsar_image_ref *images,
                            int n_images, uint32_t chunk_cap, int *end_out, char *err, size_t errlen) {
    int end = 0;
    for (int i = 0; i < n_images; i++) {
        if (!images || !images[i].bytes || images[i].len == 0 || images[i].start_pos < 0) {
            snprintf(err, errlen, "image %d has no bytes or a bad span position", i);
            return false;
        }
        int len = 0;
        if (!pulsar_image_block_extent(e, ids, n, images[i].start_pos, &len)) {
            snprintf(err, errlen, "image %d at %d is not a sentinel block in this prompt", i, images[i].start_pos);
            return false;
        }
        if (len > (int)chunk_cap) {
            snprintf(err, errlen, "image %d's block is %d tokens but one prefill chunk holds only %u "
                                  "(send a smaller image)", i, len, chunk_cap);
            return false;
        }
        if (images[i].start_pos + len > end) end = images[i].start_pos + len;
    }
    if (end_out) *end_out = end;
    return true;
}

bool pulsar_image_expand(const pulsar_engine *e, const pulsar_tokens *prompt, pulsar_image_ref *images,
                         int n_images, pulsar_tokens *out, char *err, size_t errlen) {
    const pulsar_family_vision *v = front_of(e);
    if (!v) {
        snprintf(err, errlen, "the %s family serves no images", e ? pulsar_engine_family_name(e) : "?");
        return false;
    }
    /* The tower's absence is a client-visible condition (400), not an engine crash, so it is checked HERE rather
     * than left to the prefill. */
    if (n_images > 0 && !e->vision_ready) {
        snprintf(err, errlen, "this model has no vision tower bound; it cannot accept images");
        return false;
    }
    const int placeholder = v->placeholder_id(e);
    if (placeholder < 0) {
        snprintf(err, errlen, "the tokenizer has no image placeholder token (\"%s\")", v->placeholder_text);
        return false;
    }
    /* Count first so a mismatch reports both numbers. */
    int seen = 0;
    for (int i = 0; i < prompt->len; i++) seen += prompt->v[i] == placeholder;
    if (seen != n_images) {
        snprintf(err, errlen, "the prompt carries %d image placeholder(s) but the request has %d image(s)", seen,
                 n_images);
        return false;
    }
    int next = 0;
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] != placeholder) {
            pulsar_tokens_push(out, prompt->v[i]);
            continue;
        }
        pulsar_image_ref *img = &images[next];
        if (!img->bytes || img->len == 0) {
            snprintf(err, errlen, "image %d has no bytes", next);
            return false;
        }
        const int at = out->len;   /* the block start, before any of it is appended */
        if (!v->expand(e, img, out, err, errlen)) {
            if (!err[0]) snprintf(err, errlen, "image %d could not be decoded or is not one the tower accepts", next);
            return false;
        }
        int len = 0;
        if (!pulsar_image_block_extent(e, out->v, out->len, at, &len) || at + len != out->len) {
            snprintf(err, errlen, "image %d: the family's expansion is not one block", next);
            return false;
        }
        img->start_pos = at;
        next++;
    }
    return true;
}

bool pulsar_image_refuse_orphans(const pulsar_engine *e, const pulsar_tokens *prompt, char *err, size_t errlen) {
    /* A prompt carrying sentinel ids with no image to fill them would prefill rows whose embeddings never arrive
     * (only the merge puts anything there), so it is refused instead of silently serving a wrong answer.  The
     * PLACEHOLDER is refused too: it is a renderer artifact the expansion must have replaced. */
    const pulsar_family_vision *v = front_of(e);
    if (!v) return true;
    const int placeholder = v->placeholder_id(e);
    for (int i = 0; i < prompt->len; i++) {
        if (pulsar_image_is_sentinel(e, prompt->v[i]) || (placeholder >= 0 && prompt->v[i] == placeholder)) {
            snprintf(err, errlen, "prompt token %d is image sentinel id %d, but the request carries no images", i,
                     prompt->v[i]);
            return false;
        }
    }
    return true;
}

void pulsar_image_licence_decide(const pulsar_session *s, const pulsar_tokens *prompt,
                                 const pulsar_image_ref *images, int n_images, pulsar_image_licence *out) {
    memset(out, 0, sizeof *out);
    const pulsar_engine *e = s->engine;
    /* The live history this prompt can keep: its common token prefix with the checkpoint (an echo that stops short
     * of the live tail -- stripped reasoning, a rollback -- shares a prefix without extending it). */
    const uint32_t live_len = s->checkpoint_valid ? (uint32_t)s->checkpoint.len : 0u;
    uint32_t ck = 0;
    const uint32_t lim = live_len < (uint32_t)prompt->len ? live_len : (uint32_t)prompt->len;
    while (ck < lim && s->checkpoint.v[ck] == prompt->v[ck]) ck++;
    out->live_len = live_len;
    out->common = ck;
    /* The licence decides on an EXTENSION of the live tokens; a prompt that stops short or diverges is the family's
     * seam / rebuild path, which re-enters with an extension or prefills cold. */
    out->extends_live = s->checkpoint_valid && ck == live_len;
    if (!out->extends_live) return;
    for (int i = 0; i < n_images && !out->straddles; i++) {
        int len = 0;
        (void)pulsar_image_block_extent(e, prompt->v, prompt->len, images[i].start_pos, &len);
        const uint32_t bs = (uint32_t)images[i].start_pos, be = bs + (uint32_t)len;
        if (be > ck && bs >= ck) out->n_new++;
        else if (be > ck) out->straddles = true;
    }
    /* L281: the held images' records -- each block's rows and the bytes merged there -- must be exactly the live
     * ones */
    pulsar_image_identity held;
    const bool held_ok = pulsar_image_identity_build(front_of(e), e, prompt->v, prompt->len, images, n_images, ck,
                                                     &held);
    out->n_held = held_ok ? (int)held.n : 0;
    out->held_end = held_ok ? pulsar_image_identity_end(&held) : 0u;
    out->keep = !out->straddles && held_ok && pulsar_image_identity_equal(&s->live_images, &held);
}

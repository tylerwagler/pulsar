/* L216 gate: the MoE router's IMAGE-SLOT bias (bias_vl), graded against the
 * reference's own gating code.
 *
 * WHAT IS PINNED.  The reference's Gate.forward keys on an out-of-vocab image
 * mask (inference/model.py):
 *
 *     image_mask = (input_ids >= self.vocab_size) if self.bias_vl is not None else None
 *     if self.hash:
 *         if image_mask is None:
 *             indices = self.tid2eid[input_ids]
 *         else:
 *             indices = self.tid2eid[torch.where(image_mask, 0, input_ids)]
 *             vl_indices = (scores + self.bias_vl).topk(self.topk, dim=-1)[1]
 *             indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
 *     else:
 *         if image_mask is None:
 *             scores = scores + self.bias
 *         else:
 *             scores = scores + torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
 *         indices = scores.topk(self.topk, dim=-1)[1]
 *     weights = original_scores.gather(1, indices)   # UNBIASED
 *
 * so an image slot selects with bias_vl instead of the text bias, and on a hash
 * layer it ESCAPES tid2eid entirely -- that table has one row per real token
 * and no row for a sentinel.  The routing WEIGHTS come from the unbiased
 * scores, which is the pre-existing contract.
 *
 * THE ORACLE IS THE REFERENCE, NOT A RE-DERIVATION.  The fixture is produced by
 * tests/vision_router_goldens.py, which instantiates the snapshot's own `Gate`,
 * installs a synthetic layer, runs its forward, and dumps BOTH the layer and
 * the reference's (indices, weights).  This gate replays the dumped logits and
 * token ids through pulsar_gpu_router_select_batch_tensor and requires the
 * reference's answer.  Nothing here re-implements the gating arithmetic.
 *
 * The fixture is deterministic on purpose: the 256 experts get strictly
 * descending softplus scores with a 5e-3 step and every expert in TEXT_SET /
 * IMAGE_SET is ranked last, so a text slot must select TEXT_SET and an image
 * slot IMAGE_SET -- sets pairwise disjoint from each other, from the tid2eid
 * rows, and from the score leaders.  The gate asserts that per case from the
 * golden itself (`image_differs`), so a fixture that had collapsed to one
 * answer could not pass as discriminating, and it asserts the observed margin
 * at the top-6 boundary so a case whose answer the engine's fast-math softplus
 * could reorder is reported as inconclusive rather than as a pass.
 *
 * Needs a GPU, no model: the bias / bias_vl / tid2eid tensors are laid out in
 * one synthetic model_map, and every case gets its OWN slot in it.  That is not
 * tidiness: cuda_model_range_ptr caches a device copy per (model_map, offset),
 * so two cases sharing a host address and offset hand the second case the
 * first case's cached bytes.  With per-case offsets the cache key identifies
 * the content, which is the invariant the engine relies on (one mmap'd GGUF,
 * one offset per tensor, contents never changing).
 *
 * WHY INDICES ARE EXACT AND WEIGHTS ARE NOT.  The kernel is compiled with
 * --use_fast_math (approximate expf/log1pf on the device) while the reference
 * ran on the CPU in fp32, so the two softplus() implementations differ by an
 * ulp or so.  Expert INDICES are asserted exactly -- the construction puts a
 * >= 5e-3 gap at the top-6 boundary, and the case asserts it -- while the
 * WEIGHTS are asserted to a documented tolerance and the observed error is
 * printed (it has measured ~3e-8, i.e. float epsilon on weights of order 0.3).
 */
#include "pulsar_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { BIAS_OFF = 0, VL_OFF = 4096, HASH_OFF = 8192, NAME_LEN = 32 };
static const double WTOL = 1.0e-4;
static const float MIN_MARGIN = 1.0e-3f;

struct Golden {
    std::vector<uint8_t> raw;
    uint32_t n_cases = 0, n_expert = 0, topk = 0, n_vocab = 0, hash_rows = 0;
};

struct Case {
    char name[NAME_LEN + 1];
    uint32_t hash_mode, has_vl_bias, n_rows, has_text_bias, image_differs;
    std::vector<float> bias, bias_vl, logits, margin, w;
    std::vector<int32_t> tid2eid, tokens, idx;
};

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Case names are NUL-padded ASCII by the generator; treat a short field safely. */
static void copy_name(char *dst, const uint8_t *src) {
    memcpy(dst, src, NAME_LEN);
    dst[NAME_LEN] = '\0';
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "tests/test-vectors/vision-router-goldens.bin";
    if (pulsar_gpu_init() == 0) {
        fprintf(stderr, "vision_router_gate: no GPU\n");
        return 2;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vision_router_gate: cannot open %s\n", path);
        fprintf(stderr, "  (regenerate with: python3 tests/vision_router_goldens.py <snapshot-dir> "
                        "> tests/test-vectors/vision-router-goldens.bin)\n");
        return 2;
    }
    Golden g;
    {
        uint8_t tail[1];
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz < 28) { fprintf(stderr, "vision_router_gate: %s is truncated\n", path); fclose(f); return 2; }
        g.raw.resize((size_t)sz);
        if (fread(g.raw.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return 2; }
        (void)tail;
    }
    fclose(f);

    if (memcmp(g.raw.data(), "VRT1", 4) != 0) {
        fprintf(stderr, "vision_router_gate: %s is not a VRT1 fixture\n", path);
        return 2;
    }
    g.n_cases = rd_u32(&g.raw[4]);
    g.n_expert = rd_u32(&g.raw[8]);
    g.topk = rd_u32(&g.raw[12]);
    g.n_vocab = rd_u32(&g.raw[16]);
    g.hash_rows = rd_u32(&g.raw[20]);
    /* The kernel hardcodes these; a fixture that disagrees would be graded
     * against a shape the engine cannot represent, so say so instead. */
    if (g.n_expert != 256u || g.topk != 6u || g.n_vocab != 129280u || g.hash_rows != 64u) {
        fprintf(stderr, "vision_router_gate: fixture shape %ux%u vocab %u rows %u is not the "
                        "engine's 256x6 / 129280 / 64\n", g.n_expert, g.topk, g.n_vocab, g.hash_rows);
        return 2;
    }
    if (g.n_expert * sizeof(float) > VL_OFF || VL_OFF + g.n_expert * sizeof(float) > HASH_OFF) {
        fprintf(stderr, "vision_router_gate: fixture expert count does not fit the map layout\n");
        return 2;
    }

    size_t off = 28;
    const uint64_t hash_bytes = (uint64_t)g.n_vocab * g.topk * sizeof(int32_t);

    std::vector<Case> cases;
    for (uint32_t ci = 0; ci < g.n_cases; ci++) {
        Case c;
        const size_t b = off + NAME_LEN + 20;
        const size_t need = b + (size_t)g.n_expert * 4 * 2 +
                            (size_t)g.hash_rows * g.topk * 4;
        if (need > g.raw.size()) {
            printf("FAIL case %u: fixture truncated\n", ci);
            return 1;
        }
        copy_name(c.name, &g.raw[off]);
        off += NAME_LEN;
        c.hash_mode = rd_u32(&g.raw[off + 0]);
        c.has_vl_bias = rd_u32(&g.raw[off + 4]);
        c.n_rows = rd_u32(&g.raw[off + 8]);
        c.has_text_bias = rd_u32(&g.raw[off + 12]);
        c.image_differs = rd_u32(&g.raw[off + 16]);
        off += 20;

        const uint64_t rows = c.n_rows;
        const size_t tail = (size_t)g.n_expert * 4 * 2 +
                            (size_t)g.hash_rows * g.topk * 4 +
                            (size_t)rows * g.n_expert * 4 + (size_t)rows * 4 * 2 +
                            (size_t)rows * g.topk * 4 * 2;
        if (off + tail > g.raw.size()) {
            printf("FAIL %-32s: fixture truncated\n", c.name);
            return 1;
        }
        c.bias.resize(g.n_expert);
        memcpy(c.bias.data(), &g.raw[off], (size_t)g.n_expert * 4); off += (size_t)g.n_expert * 4;
        c.bias_vl.resize(g.n_expert);
        memcpy(c.bias_vl.data(), &g.raw[off], (size_t)g.n_expert * 4); off += (size_t)g.n_expert * 4;
        c.tid2eid.resize((size_t)g.hash_rows * g.topk);
        memcpy(c.tid2eid.data(), &g.raw[off], c.tid2eid.size() * 4); off += c.tid2eid.size() * 4;
        c.logits.resize((size_t)rows * g.n_expert);
        memcpy(c.logits.data(), &g.raw[off], c.logits.size() * 4); off += c.logits.size() * 4;
        c.tokens.resize(rows);
        memcpy(c.tokens.data(), &g.raw[off], c.tokens.size() * 4); off += c.tokens.size() * 4;
        c.margin.resize(rows);
        memcpy(c.margin.data(), &g.raw[off], c.margin.size() * 4); off += c.margin.size() * 4;
        c.idx.resize((size_t)rows * g.topk);
        memcpy(c.idx.data(), &g.raw[off], c.idx.size() * 4); off += c.idx.size() * 4;
        c.w.resize((size_t)rows * g.topk);
        memcpy(c.w.data(), &g.raw[off], c.w.size() * 4); off += c.w.size() * 4;
        cases.push_back(std::move(c));
    }

    /* ---- the synthetic layer: one slot per case --------------------------- */
    const uint64_t stride = ((HASH_OFF + hash_bytes) + 4095u) & ~(uint64_t)4095u;
    std::vector<uint8_t> map((size_t)(stride * cases.size()), 0);
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const Case &c = cases[ci];
        uint8_t *base = map.data() + stride * ci;
        memcpy(base + BIAS_OFF, c.bias.data(), (size_t)g.n_expert * 4);
        memcpy(base + VL_OFF, c.bias_vl.data(), (size_t)g.n_expert * 4);
        memcpy(base + HASH_OFF, c.tid2eid.data(), c.tid2eid.size() * 4);
    }

    int failures = 0, checked = 0;
    std::vector<int32_t> dev_idx[16];

    for (uint32_t ci = 0; ci < cases.size() && ci < 16u; ci++) {
        const Case &c = cases[ci];
        const uint64_t rows = c.n_rows;
        int bad = 0;

        /* ---- GUARD 1: the fixture must discriminate -----------------------
         * Taken from the golden, so it cannot be satisfied by a fixture whose
         * two biases happen to agree: where the reference used a DIFFERENT set
         * for an image slot, this case is the one that catches a kernel that
         * ignored bias_vl. */
        std::vector<uint32_t> img_rows, txt_rows;
        for (uint32_t r = 0; r < c.n_rows; r++) {
            if (c.has_vl_bias && c.tokens[r] >= (int32_t)g.n_vocab) img_rows.push_back(r);
            else txt_rows.push_back(r);
        }
        for (size_t i = 0; i < c.idx.size(); i++) {
            if (c.idx[i] < 0 || c.idx[i] >= (int32_t)g.n_expert) {
                printf("  fixture: selected expert %d out of range\n", c.idx[i]);
                bad++;
                break;
            }
        }
        if (!img_rows.empty() && !txt_rows.empty()) {
            const int32_t *a = &c.idx[(size_t)img_rows[0] * g.topk];
            const int32_t *t = &c.idx[(size_t)txt_rows[0] * g.topk];
            const int same = (memcmp(a, t, g.topk * 4) == 0);
            if (c.image_differs && same) {
                printf("  fixture claims image slots differ but the golden gives the same set\n");
                bad++;
            }
            if (!c.image_differs && !same) {
                printf("  fixture claims image slots match but the golden gives a different set\n");
                bad++;
            }
        }

        /* ---- GUARD 2: conclusive margins ---------------------------------- */
        double min_gap = INFINITY;
        for (uint32_t r = 0; r < c.n_rows; r++) {
            if (c.margin[r] < min_gap) min_gap = (double)c.margin[r];
            if (std::isfinite(c.margin[r]) && c.margin[r] < MIN_MARGIN) {
                printf("  row %u: top-6 boundary gap %.3e is below %.0e -- inconclusive\n",
                       r, (double)c.margin[r], (double)MIN_MARGIN);
                bad++;
            }
        }

        /* ---- the device against the reference ------------------------------ */
        const uint64_t logits_bytes = rows * g.n_expert * sizeof(float);
        pulsar_gpu_tensor *dlog = pulsar_gpu_tensor_alloc(logits_bytes);
        pulsar_gpu_tensor *dtok = pulsar_gpu_tensor_alloc(rows * sizeof(int32_t));
        pulsar_gpu_tensor *dsel = pulsar_gpu_tensor_alloc(rows * g.topk * sizeof(int32_t));
        pulsar_gpu_tensor *dw = pulsar_gpu_tensor_alloc(rows * g.topk * sizeof(float));
        std::vector<int32_t> idx((size_t)rows * g.topk);
        std::vector<float> w((size_t)rows * g.topk);
        const int launched =
            dlog && dtok && dsel && dw &&
            pulsar_gpu_tensor_write(dlog, 0, c.logits.data(), logits_bytes) &&
            pulsar_gpu_tensor_write(dtok, 0, c.tokens.data(), rows * sizeof(int32_t)) &&
            /* The merged signature: the branch's tid2eid arm makes "hash mode"
             * a ROW COUNT rather than a flag (0 = no table), and dev's image
             * bias rides at the end.  A hash case passes its table offset and row
             * count; a text case passes 0/0. */
            pulsar_gpu_router_select_batch_tensor(dsel, dw, NULL, map.data(), map.size(),
                                                  stride * ci + BIAS_OFF,
                                                  c.has_text_bias != 0,
                                                  c.hash_mode ? stride * ci + HASH_OFF : 0,
                                                  c.hash_mode ? g.hash_rows : 0,
                                                  dlog, dtok, g.n_expert, g.topk, 1.5f,
                                                  (uint32_t)rows,
                                                  c.has_vl_bias ? stride * ci + VL_OFF : 0u,
                                                  g.n_vocab, c.has_vl_bias != 0) &&
            pulsar_gpu_end_commands() &&
            pulsar_gpu_tensor_read(dsel, 0, idx.data(), idx.size() * 4) &&
            pulsar_gpu_tensor_read(dw, 0, w.data(), w.size() * 4);
        if (!launched) {
            printf("FAIL %-32s: launch/copy failed\n", c.name);
            failures++;
        }

        int idx_bad = 0;
        double max_werr = 0.0;
        for (uint32_t r = 0; launched && r < c.n_rows; r++) {
            if (memcmp(&idx[(size_t)r * g.topk], &c.idx[(size_t)r * g.topk], g.topk * 4) != 0) {
                if (idx_bad++ < 2) {
                    printf("  row %u idx: device", r);
                    for (uint32_t j = 0; j < g.topk; j++) printf(" %d", idx[(size_t)r * g.topk + j]);
                    printf("\n           ref   ");
                    for (uint32_t j = 0; j < g.topk; j++) printf(" %d", c.idx[(size_t)r * g.topk + j]);
                    printf("\n");
                }
            }
            for (uint32_t j = 0; j < g.topk; j++) {
                const double d = fabs((double)w[(size_t)r * g.topk + j] -
                                      (double)c.w[(size_t)r * g.topk + j]);
                if (d > max_werr) max_werr = d;
            }
        }
        if (idx_bad) { printf("  %d row(s) disagree with the reference\n", idx_bad); bad++; }
        if (launched && max_werr > WTOL) {
            printf("  weights differ from the reference by %.3e (> %.0e)\n", max_werr, WTOL);
            bad++;
        }

        if (launched) dev_idx[ci] = idx;
        printf("%s %-32s rows=%u min gap=%.2e max|dW|=%.2e\n", bad ? "FAIL" : "ok  ", c.name,
               c.n_rows, min_gap, max_werr);
        failures += bad;
        if (launched) checked += (int)c.n_rows;
        pulsar_gpu_tensor_free(dlog);
        pulsar_gpu_tensor_free(dtok);
        pulsar_gpu_tensor_free(dsel);
        pulsar_gpu_tensor_free(dw);
    }

    /* ---- the controlled comparison: the same batch, with and without ------ */
    struct Pair { uint32_t a, b; const char *why; };
    const Pair same[] = {
        {1, 0, "bias_vl present must not disturb a text-only batch (learned)"},
        {4, 3, "bias_vl present must not disturb a text-only batch (hash)"},
    };
    for (const Pair &pr : same) {
        if (pr.a >= cases.size() || pr.b >= cases.size()) continue;
        if (dev_idx[pr.a].empty() || dev_idx[pr.b].empty()) continue;
        if (dev_idx[pr.a] != dev_idx[pr.b]) {
            printf("FAIL %s\n", pr.why);
            failures++;
        } else {
            printf("ok   %s\n", pr.why);
        }
    }

    printf("VISION ROUTER GATE: %s (%d rows checked, %d failures)\n",
           failures ? "FAIL" : "PASS", checked, failures);
    return failures ? 1 : 0;
}

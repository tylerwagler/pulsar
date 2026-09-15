/* L216 gate: the merged image span lands in the HC carrier correctly.
 *
 * vision_merge_span() produces ONE bf16 embedding per span position; the
 * reference writes that block into h and only THEN expands to hc_mult copies
 * (`merge_image_embeddings` then `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`).
 * So every HC stream must carry the same merged row, and every row OUTSIDE the
 * span must be left exactly as the token embedder left it.
 *
 * This grades gpu_graph_write_vision_span(): the replication across all
 * PULSAR_N_HC streams, the offset (the span is not at row 0), and the refusals
 * (a span that would run off the end must write NOTHING rather than clip).
 *
 * It composes with vision-merge-gate, which grades vision_merge_span() itself
 * against the checkpoint's merge_image_embeddings: that gate pins the VALUES,
 * this one pins the PLACEMENT.  Neither alone would catch the other's failure --
 * a right-looking span in the wrong rows, or the right rows holding aligner
 * output for the wrong sentinel type.
 *
 * Needs a GPU, no model: the carrier is a plain tensor of raw bf16.
 */
#include "pulsar.h"
#include "pulsar_gpu.h"
#include "pulsar_engine_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const uint16_t FILL = 0xA5A5u;   /* a bit pattern no bf16 row can be mistaken for */

int main(void) {
    if (pulsar_gpu_init() == 0) {
        fprintf(stderr, "vision_hc_gate: no GPU\n");
        return 2;
    }

    const uint32_t n_tokens = 8, row0 = 2, n_rows = 3;
    const size_t row_elt = (size_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const size_t per_row = row_elt * PULSAR_HC_ELT_SIZE;
    const uint64_t carrier_bytes = (uint64_t)n_tokens * per_row;

    std::vector<uint16_t> rows((size_t)n_rows * PULSAR_N_EMBD);
    for (uint32_t r = 0; r < n_rows; r++)
        for (uint32_t d = 0; d < PULSAR_N_EMBD; d++)
            rows[(size_t)r * PULSAR_N_EMBD + d] = (uint16_t)(0x3F00u + r * 0x40u + (d & 0x3Fu));

    int failures = 0;

    /* Reads element (token, hc, d) out of the carrier. */
    auto elem = [&](const std::vector<uint16_t> &buf, uint32_t t, uint32_t h, uint32_t d) {
        const size_t base = (size_t)t * row_elt + (size_t)h * PULSAR_N_EMBD + d;
        return buf[base];
    };

    /* ---- the write itself ------------------------------------------------- */
    {
        std::vector<uint16_t> carrier(carrier_bytes / 2, FILL);
        std::vector<uint16_t> back(carrier_bytes / 2, 0);
        pulsar_gpu_tensor *d = pulsar_gpu_tensor_alloc(carrier_bytes);
        int bad = 0;
        if (!d || !pulsar_gpu_tensor_write(d, 0, carrier.data(), carrier_bytes) ||
            !gpu_graph_write_vision_span(d, rows.data(), n_rows, row0, n_tokens) ||
            !pulsar_gpu_end_commands() ||
            !pulsar_gpu_tensor_read(d, 0, back.data(), carrier_bytes)) {
            printf("FAIL launch/copy failed\n");
            failures++;
        } else {
            /* every HC stream of every span row carries the merged row */
            for (uint32_t r = 0; r < n_rows && !bad; r++) {
                for (uint32_t h = 0; h < PULSAR_N_HC && !bad; h++) {
                    for (uint32_t dd = 0; dd < PULSAR_N_EMBD; dd++) {
                        const uint16_t want = rows[(size_t)r * PULSAR_N_EMBD + dd];
                        if (elem(back, row0 + r, h, dd) != want) {
                            printf("FAIL row %u hc %u dim %u = 0x%04x, merged row has 0x%04x\n",
                                   row0 + r, h, dd, elem(back, row0 + r, h, dd), want);
                            bad++;
                            break;
                        }
                    }
                }
            }
            /* and nothing else moved */
            for (uint32_t t = 0; t < n_tokens && !bad; t++) {
                if (t >= row0 && t < row0 + n_rows) continue;
                if (elem(back, t, 0, 0) != FILL || elem(back, t, PULSAR_N_HC - 1, PULSAR_N_EMBD - 1) != FILL) {
                    printf("FAIL row %u outside the span was rewritten\n", t);
                    bad++;
                }
            }
            printf("%s span write      rows %u..%u of %u, all %u HC streams\n",
                   bad ? "FAIL" : "ok  ", row0, row0 + n_rows - 1, n_tokens, (unsigned)PULSAR_N_HC);
        }
        failures += bad;
        pulsar_gpu_tensor_free(d);
    }

    /* ---- the refusals ------------------------------------------------------
     * A span that does not fit must write NOTHING: clipping would leave a
     * half-merged image whose span ids still claim the whole block. */
    {
        std::vector<uint16_t> carrier(carrier_bytes / 2, FILL);
        std::vector<uint16_t> back(carrier_bytes / 2, 0);
        pulsar_gpu_tensor *d = pulsar_gpu_tensor_alloc(carrier_bytes);
        struct Refuse { const char *name; uint32_t rows_n, at, ntok; const uint16_t *r; bool want; };
        const Refuse cases[] = {
            {"span runs off the end",  n_rows, 7, n_tokens, rows.data(), false},
            {"row0 == n_tokens",       n_rows, n_tokens, n_tokens, rows.data(), false},
            {"zero rows",              0,      0, n_tokens, rows.data(), false},
            {"no rows pointer",        n_rows, 0, n_tokens, NULL,        false},
            {"exactly fills the tail", 2,      6, n_tokens, rows.data(), true},
        };
        int bad = 0;
        for (const Refuse &c : cases) {
            if (!pulsar_gpu_tensor_write(d, 0, carrier.data(), carrier_bytes)) { bad++; break; }
            const bool got = gpu_graph_write_vision_span(d, c.r, c.rows_n, c.at, c.ntok);
            pulsar_gpu_end_commands();
            if (got != c.want) {
                printf("FAIL %-24s returned %d, wanted %d\n", c.name, (int)got, (int)c.want);
                bad++;
                continue;
            }
            if (!got) {
                /* a refused call must not have touched the carrier */
                if (!pulsar_gpu_tensor_read(d, 0, back.data(), carrier_bytes)) { bad++; break; }
                for (size_t i = 0; i < back.size(); i++) {
                    if (back[i] != FILL) {
                        printf("FAIL %-24s wrote the carrier anyway (word %zu)\n", c.name, i);
                        bad++;
                        break;
                    }
                }
            } else {
                if (!pulsar_gpu_tensor_read(d, 0, back.data(), carrier_bytes)) { bad++; break; }
                if (elem(back, c.at, 0, 0) != rows[0]) {
                    printf("FAIL %-24s did not write row %u\n", c.name, c.at);
                    bad++;
                }
            }
            printf("%s %-24s refused=%d\n", bad ? "FAIL" : "ok  ", c.name, (int)!got);
        }
        failures += bad;
        pulsar_gpu_tensor_free(d);
    }

    /* ---- an undersized carrier is refused, not overrun --------------------- */
    {
        pulsar_gpu_tensor *small = pulsar_gpu_tensor_alloc(carrier_bytes - PULSAR_HC_ELT_SIZE);
        const bool got = small && gpu_graph_write_vision_span(small, rows.data(), n_rows, row0, n_tokens);
        if (!small || got) {
            printf("FAIL undersized carrier was accepted\n");
            failures++;
        } else {
            printf("ok   undersized carrier refused\n");
        }
        pulsar_gpu_tensor_free(small);
    }

    printf("VISION HC GATE: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

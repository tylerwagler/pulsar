/* Task #32 probe — does an mseq rewind leave ms_n_comp[bank] stale?
 *
 * pulsar_session::rewind clamps ONLY the scalar frontier counters
 * (grep -c ms_n_comp src/engine/session.cpp == 0). The served path validates
 * the NEXT multiseq step against the PER-BANK slots
 * (gpu_graph_multiseq_step_begin, capture_cur=false). The two are bridged only
 * by bank_state_save/_restore on a bank hand-off.
 *
 * In the server's round loop the hand-off ordering is:
 *   decode_mixed -> live_bank = -1 -> bank_switch(b) [restore, NO save]
 *   -> emit -> rewind [clamps SCALARS] -> next slot's bank_switch [SAVES]
 * With several live slots the clamp is published by the next switch-away. With
 * ONE live slot the loop ends first, and the following round restores from
 * ms_n_comp[b] -- which the rewind never touched.
 *
 * This probe runs that single-slot shape deterministically and prints both
 * representations at each step, so the answer is observed rather than argued.
 *
 * usage: ./tests/mseq_rewind_probe MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Report layer 2 (ratio 4 in this artifact -- the layer L120's production
 * signature named) in both representations. */
static void show(pulsar_session *s, const char *when, int pos) {
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t il = 2;
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    printf("%-34s pos=%-5d want=%-5u scalar_n_comp=%-5u ms_n_comp[0]=%-5u%s\n",
           when, pos, ratio ? (uint32_t)pos / ratio : 0u,
           g->layer_n_comp[il], g->ms_n_comp[0][il],
           (ratio && g->ms_n_comp[0][il] != (uint32_t)pos / ratio) ? "   <-- per-bank DIVERGED" : "");
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine *e = NULL;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "engine open failed\n"); return 2;
    }
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session create failed\n"); return 2; }

    /* deterministic token stream */
    static int toks[800];
    for (int i = 0; i < 800; i++) toks[i] = 1000 + (i % 97);
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = toks; p.len = p.cap = 600;
    char err[256];
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) {
        fprintf(stderr, "sync failed: %s\n", err); return 2;
    }
    show(s, "after prefill(600)", pulsar_session_pos(s));

    /* mimic admission: publish the prefilled frontier into bank 0 */
    pulsar_session_bank_state_save(s, 0u);
    show(s, "after bank_state_save(0)", pulsar_session_pos(s));

    static float *logits = (float *)malloc(sizeof(float) * (size_t)PULSAR_N_VOCAB);
    /* round 1: decode 6 rows through the SERVED entry, one row on bank 0 */
    for (int k = 0; k < 6; k++) {
        pulsar_multiseq_req req;
        req.bank = 0u; req.pos = (int32_t)pulsar_session_pos(s); req.token = toks[600 + k];
        uint32_t got = 0;
        const int rc = pulsar_session_decode_mixed(s, &req, 1u, logits,
                                                   (int)PULSAR_N_VOCAB, &got, 0u,
                                                   err, sizeof err);
        if (rc != 0) { printf("round1 step %d REJECTED rc=%d: %s\n", k, rc, err); return 1; }
        pulsar_session_note_committed_tokens(s, &toks[600 + k], 1);
    }
    show(s, "after mseq round 1 (6 rows)", pulsar_session_pos(s));

    /* ghost rewind, exactly as the server does when emission stops mid-batch */
    const int target = pulsar_session_pos(s) - 3;
    pulsar_session_rewind(s, target);
    show(s, "after rewind (single live slot)", target);

    /* round 2 on the SAME bank with no intervening switch-away -- the shape
     * that has no publisher for the clamp. Does step_begin reject? */
    pulsar_multiseq_req req2;
    req2.bank = 0u; req2.pos = (int32_t)pulsar_session_pos(s); req2.token = toks[700];
    uint32_t got2 = 0;
    const int rc2 = pulsar_session_decode_mixed(s, &req2, 1u, logits,
                                                (int)PULSAR_N_VOCAB, &got2, 0u,
                                                err, sizeof err);
    printf("\nround 2 on same bank after rewind: rc=%d%s\n", rc2,
           rc2 == 0 ? "  (ACCEPTED)" : "  <-- REJECTED, L120's shape");
    if (rc2 != 0) printf("   %s\n", err);

    pulsar_session_free(s);
    pulsar_engine_close(e);
    return 0;
}

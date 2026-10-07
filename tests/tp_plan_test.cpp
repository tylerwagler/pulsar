/* tp_plan_test.cpp -- L272 P4b: a family's tensor-parallel slices, recorded on the host.
 *
 * Opens a real model inspect-only as rank R of N (no GPU work, no transport): the open builds the rank's
 * plan (the family declares, the core chooses each operation by format), and this runs it in record mode
 * (src/engine/tp_slice.cpp) -- every slice the rank would build or register is written as one canonical
 * line to stdout.  A change to how slices are declared or done must reproduce the committed golden
 * (tests/tp-plan-golden/) line for line, on any box, before a pair runs it.
 *
 *   tp_plan_test MODEL RANK NRANKS > plan.txt
 */
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL RANK NRANKS\n", argv[0]);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = true;
    opt.tp_rank = atoi(argv[2]);
    opt.tp_nranks = atoi(argv[3]);
    opt.tp_peers = "record";   /* the rank and group only: an inspect-only open builds no transport */
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "tp_plan_test: %s does not open as rank %s of %s\n", argv[1], argv[2], argv[3]);
        return 1;
    }
    if (!e->tp_plan) {
        fprintf(stderr, "tp_plan_test: the open built no TP plan\n");
        pulsar_engine_close(e);
        return 1;
    }
    printf("# family %s rank %d of %u\n", e->family->name, e->model.tp_rank, e->model.tp_n_ranks);
    pulsar_tp_record_begin(stdout);
    const bool ok = pulsar_tp_plan_run(e);
    pulsar_tp_record_end();
    fflush(stdout);
    pulsar_engine_close(e);
    if (!ok) {
        fprintf(stderr, "tp_plan_test: the plan's run refused\n");
        return 1;
    }
    return 0;
}

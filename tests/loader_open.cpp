/* loader_open.cpp -- L278 (the loader contract): open a model inspect-only and say that the open RETURNED.
 *
 * The CLI cannot tell a refusal from a death: a loader that exits the process and an open that returns an error
 * both end the run with a non-zero status.  The contract is the second -- a broken artifact is refused by name, at a
 * stage boundary, and the caller (a server, a gate) keeps running.  This probe prints LOADER_OPEN_RETURNED after
 * pulsar_engine_open comes back, whatever it returned; tests/loader_contract.py grades its absence as "died inside the
 * loader".
 *
 *   loader_open MODEL     exit 0 = opened, 3 = refused (returned an error); no marker = the loader exited
 */
#include "pulsar.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = true;
    pulsar_engine *e = NULL;
    const int rc = pulsar_engine_open(&e, &opt);
    printf("LOADER_OPEN_RETURNED rc=%d engine=%s\n", rc, e ? "open" : "none");
    fflush(stdout);
    if (e) pulsar_engine_close(e);
    return rc == 0 && e ? 0 : 3;
}

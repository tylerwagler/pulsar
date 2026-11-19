/* loader_open.cpp -- L278 (the loader contract): open a model inspect-only and say that the open RETURNED.
 *
 * The CLI cannot tell a refusal from a death: a loader that exits the process and an open that returns an error
 * both end the run with a non-zero status.  The contract is the second -- a broken artifact is refused by name, at a
 * stage boundary, and the caller (a server, a gate) keeps running.  This probe prints LOADER_OPEN_RETURNED after
 * pulsar_engine_open comes back, whatever it returned; tests/loader_contract.py grades its absence as "died inside the
 * loader".
 *
 *   loader_open MODEL     exit 0 = opened, 3 = refused (returned an error); no marker = the loader exited
 *
 * L426: with a store directory it also opens a disk-KV segment store there under the opened model's identity
 * (pulsar_engine_kv_identity), as a server does, and says what the store answered -- so the gate can grade, on the
 * host, that a store one container wrote is not indexed under another's weights:
 *
 *   loader_open MODEL STORE_DIR [write]
 *     LOADER_OPEN_IDENTITY weights=<hex> tokenizer=<hex>     the identity's L426 words
 *     LOADER_OPEN_STORE_LOG <line>                           each line the store's open logged
 *     LOADER_OPEN_STORE indexed=<n> wrote=<0|1>              segments indexed at the open; with `write`, whether a
 *                                                            segment and its restore point were then added
 */
#include "pulsar.h"
#include "lib/pulsar_segstore.h"

#include <stdio.h>
#include <string.h>

/** A pulsar_segstore_write_fn: 256 bytes of a payload (the digest slot is the store's).
 *  @param fp the image  @return 0 written, 1 failed */
static int write_payload(FILE *fp, void *, char *, size_t) {
    static const char bytes[256] = {1};
    return fwrite(bytes, 1, sizeof bytes, fp) == sizeof bytes ? 0 : 1;
}

/** A pulsar_segstore_log_fn: the store's line, marked.  @param msg the line */
static void store_log(void *, const char *msg) { printf("LOADER_OPEN_STORE_LOG %s\n", msg); }

/** Open a segment store at `dir` under the engine's identity and say what it indexed; with `write`, add a segment.
 *  @param e     the opened engine
 *  @param dir   the store directory
 *  @param write put one segment and its restore point
 *  @return true when the store opened (and took the segment asked for) */
static bool store_probe(pulsar_engine *e, const char *dir, bool write) {
    const pulsar_segstore_ident id = pulsar_engine_kv_identity(e);
    printf("LOADER_OPEN_IDENTITY weights=%016llx tokenizer=%016llx\n", (unsigned long long)id.weights,
           (unsigned long long)id.tokenizer);
    pulsar_segstore *st = pulsar_segstore_open(dir, 0, id, store_log, NULL);
    if (!st) return false;
    const int indexed = pulsar_segstore_count(st);
    bool wrote = false;
    if (write) {
        char key[41], err[256] = "";
        wrote = pulsar_segstore_put(st, "", TokPos{0}, TokPos{128}, "SYSTEM:", TextBytes{7u}, 256, write_payload, 0, NULL,
                                    NULL, key, err, sizeof err) &&
                pulsar_segstore_put_rp(st, key, 256, 0, write_payload, NULL, err, sizeof err);
        if (!wrote) fprintf(stderr, "loader_open: the store took no segment: %s\n", err);
    }
    printf("LOADER_OPEN_STORE indexed=%d wrote=%d\n", indexed, wrote ? 1 : 0);
    pulsar_segstore_close(st);
    return wrote == write;
}

int main(int argc, char **argv) {
    const bool write = argc == 4 && !strcmp(argv[3], "write");
    if (argc < 2 || argc > 4 || (argc == 4 && !write)) {
        fprintf(stderr, "usage: %s MODEL [STORE_DIR [write]]\n", argv[0]);
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
    bool store_ok = true;
    if (e && rc == 0 && argc >= 3) store_ok = store_probe(e, argv[2], write);
    fflush(stdout);
    if (e) pulsar_engine_close(e);
    if (!store_ok) return 4;
    return rc == 0 && e ? 0 : 3;
}

/* The one seam between a model-dependent gate and the process it runs in.
 *
 * A gate is written as a program: it parses argv, opens the engine, runs,
 * returns an exit code.  Built on its own it IS that program.  Built into
 * tests/gates_runner.cpp (-DPULSAR_GATE_RUNNER -DGATE_ENTRY=gate_xxx_main) the
 * same source becomes one function among many, and the engine it "opens" is
 * the runner's: one 86 GB model load per engine configuration for the whole
 * battery instead of one per gate (L163: 28 loads x 18.6 s was a third of the
 * battery, bounded by the NVMe the model streams from).
 *
 * Contract for a gate that includes this header:
 *   - its entry is `int GATE_ENTRY(int argc, char **argv)`, never `main`;
 *   - it opens through gate_engine_open and closes through gate_engine_close,
 *     and treats the engine as borrowed: no pulsar_gpu_cleanup, no second
 *     live engine (the instance lock refuses one anyway);
 *   - it never calls exit(): every failure is a return, because the process
 *     may be the whole battery;
 *   - it resets its file-scope state at entry and frees every session and
 *     token buffer on every path -- a leaked session holds a full KV graph the
 *     next gate has to fit beside;
 *   - it asks for its bank-pool size through pulsar_engine_set_bank_pool
 *     before its first session, never by writing PULSAR_MSEQ_BANKS after the
 *     engine is open (the engine parses that variable once per process).
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "pulsar.h"

#ifndef GATE_ENTRY
#define GATE_ENTRY main
#endif

/* ---- a gate on a TP group (L240) --------------------------------------------
 *
 * A gate has no TP flags, and must not grow twenty-one copies of them.  The
 * group comes from the environment, read ONCE at open by the one seam every
 * gate already passes through:
 *
 *   PULSAR_TP_RANK    this rank, 0-based
 *   PULSAR_TP_NRANKS  group size (>= 2)
 *   PULSAR_TP_PEERS   every rank's "host:port", rank order, comma-separated
 *   PULSAR_TP_PORT    this rank's listen port (optional; engine default)
 *
 * All unset: one box, the options are the gate's alone.  Some set: refuse --
 * a half-configured group is a misconfiguration, never a single-box run.
 *
 * A WORKER rank's process is the receive loop and nothing else: the leader
 * drives every session operation and the worker applies it (slice 4e), so no
 * gate body may run on it.  gate_tp_worker_or_continue runs the loop when the
 * engine came up as a worker and exits the process with the loop's verdict.
 * That is the one sanctioned exit() behind this seam: on a worker rank there
 * is no battery to return to, and the leader's grade is the run's grade. */
static inline int gate_tp_options_from_env(pulsar_engine_options *opt) {
    const char *rank = getenv("PULSAR_TP_RANK"), *n = getenv("PULSAR_TP_NRANKS");
    const char *peers = getenv("PULSAR_TP_PEERS"), *port = getenv("PULSAR_TP_PORT");
    if (!rank && !n && !peers) return 0;
    if (!rank || !n || !peers) {
        fprintf(stderr, "gate: PULSAR_TP_RANK, PULSAR_TP_NRANKS and PULSAR_TP_PEERS must be set "
                        "together (set:%s%s%s) -- refusing a half-configured group\n",
                rank ? " RANK" : "", n ? " NRANKS" : "", peers ? " PEERS" : "");
        return 1;
    }
    opt->tp_rank = atoi(rank);
    opt->tp_nranks = atoi(n);
    opt->tp_peers = peers;
    opt->tp_port = port ? atoi(port) : 0;
    fprintf(stderr, "gate: TP group from the environment -- rank %d of %d, peers %s\n",
            opt->tp_rank, opt->tp_nranks, peers);
    return 0;
}

static inline void gate_tp_worker_or_continue(pulsar_engine *e) {
    if (!pulsar_engine_is_tp_worker(e)) return;
    char err[512] = {0};
    const int rc = pulsar_tp_worker_run(e, err, sizeof err);
    fprintf(stderr, "gate: TP worker loop ended rc=%d%s%s\n", rc, err[0] ? ": " : "", err);
    pulsar_engine_close(e);
    exit(rc == 0 ? 0 : 1);
}

#ifdef PULSAR_GATE_RUNNER
/* The runner's broker (tests/gates_runner.cpp): returns the live engine when
 * `opt` matches its configuration, otherwise closes the live one and opens
 * this configuration.  Close is the broker's decision, not the gate's. */
int  gate_engine_open(pulsar_engine **e, const pulsar_engine_options *opt);
void gate_engine_close(pulsar_engine *e);
#else
static inline int gate_engine_open(pulsar_engine **e, const pulsar_engine_options *opt) {
    pulsar_engine_options o = *opt;
    if (gate_tp_options_from_env(&o)) return 1;
    const int rc = pulsar_engine_open(e, &o);
    if (rc == 0) gate_tp_worker_or_continue(*e);
    return rc;
}
static inline void gate_engine_close(pulsar_engine *e) {
    pulsar_engine_close(e);
}
#endif

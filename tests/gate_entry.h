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

#include "pulsar.h"

#ifndef GATE_ENTRY
#define GATE_ENTRY main
#endif

#ifdef PULSAR_GATE_RUNNER
/* The runner's broker (tests/gates_runner.cpp): returns the live engine when
 * `opt` matches its configuration, otherwise closes the live one and opens
 * this configuration.  Close is the broker's decision, not the gate's. */
int  gate_engine_open(pulsar_engine **e, const pulsar_engine_options *opt);
void gate_engine_close(pulsar_engine *e);
#else
static inline int gate_engine_open(pulsar_engine **e, const pulsar_engine_options *opt) {
    return pulsar_engine_open(e, opt);
}
static inline void gate_engine_close(pulsar_engine *e) {
    pulsar_engine_close(e);
}
#endif

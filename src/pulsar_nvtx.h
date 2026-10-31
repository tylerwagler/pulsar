/* TEMPORARY profiling instrument (work/nvtx-prof, not for landing): NVTX
 * ranges so an nsys capture attributes the timeline to pulsar's phases.
 * nvtx3 is header-only; with no tool attached a push/pop is a few ns. */
/** @file pulsar_nvtx.h
 *  NVTX ranges for nsys captures: a scoped range object and the macros that open one. */
#pragma once
#include <nvtx3/nvToolsExt.h>

/** A scoped NVTX range: pushed on construction, popped on destruction.  Not copyable. */
struct pulsar_nvtx_range {
    /** Push the range.
     *  @param name the range's label (shown on the nsys timeline) */
    explicit pulsar_nvtx_range(const char *name) { nvtxRangePushA(name); }
    /** Pop the range. */
    ~pulsar_nvtx_range() { nvtxRangePop(); }
    pulsar_nvtx_range(const pulsar_nvtx_range &) = delete;
    pulsar_nvtx_range &operator=(const pulsar_nvtx_range &) = delete;
};
/** @def PULSAR_NVTX_CAT2
 *  Token paste (the inner step of PULSAR_NVTX_CAT). */
#define PULSAR_NVTX_CAT2(a, b) a##b
/** @def PULSAR_NVTX_CAT
 *  Token paste after expanding its arguments (gives each range a per-line name). */
#define PULSAR_NVTX_CAT(a, b) PULSAR_NVTX_CAT2(a, b)
/** @def PULSAR_NVTX
 *  Open a range named `name` until the end of the enclosing scope. */
#define PULSAR_NVTX(name) pulsar_nvtx_range PULSAR_NVTX_CAT(pulsar_nvtx_, __LINE__)(name)
/** @def PULSAR_NVTX_FN
 *  Open a range named after the enclosing function until the end of its scope. */
#define PULSAR_NVTX_FN() PULSAR_NVTX(__func__)

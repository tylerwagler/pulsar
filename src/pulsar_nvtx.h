/* TEMPORARY profiling instrument (work/nvtx-prof, not for landing): NVTX
 * ranges so an nsys capture attributes the timeline to pulsar's phases.
 * nvtx3 is header-only; with no tool attached a push/pop is a few ns. */
#pragma once
#include <nvtx3/nvToolsExt.h>

struct pulsar_nvtx_range {
    explicit pulsar_nvtx_range(const char *name) { nvtxRangePushA(name); }
    ~pulsar_nvtx_range() { nvtxRangePop(); }
    pulsar_nvtx_range(const pulsar_nvtx_range &) = delete;
    pulsar_nvtx_range &operator=(const pulsar_nvtx_range &) = delete;
};
#define PULSAR_NVTX_CAT2(a, b) a##b
#define PULSAR_NVTX_CAT(a, b) PULSAR_NVTX_CAT2(a, b)
#define PULSAR_NVTX(name) pulsar_nvtx_range PULSAR_NVTX_CAT(pulsar_nvtx_, __LINE__)(name)
#define PULSAR_NVTX_FN() PULSAR_NVTX(__func__)

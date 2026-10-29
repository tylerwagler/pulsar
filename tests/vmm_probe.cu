/* vmm_probe.cu -- can the KV pools be CUDA VMM on this device?  (work/kv-vmm, after the 2026-10-29 work hang)
 *
 * The hang: a GPU fault on a cudaMallocManaged KV page that nvidia-uvm could not service at the kernel's min
 * watermark (its uvm_va_block_context_t allocation failed) -- the GPU waited forever.  The replacement maps each
 * bank's pool physical explicitly with the driver's virtual memory management API (cuMemAddressReserve +
 * cuMemCreate + cuMemMap + cuMemSetAccess): no fault path, and an allocation that cannot be backed is a return code
 * on the host.  This probe asks the device whether that holds, one PASS / FAIL line per check:
 *
 *   attr        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, the granularity (min / recommended)
 *   reserve     16 x 8 GiB of VA (a 16-bank pool at --ctx 1M of 8 KiB a token)
 *   rw          every bank: two chunks mapped, a kernel writes a pattern, a kernel checks it, cudaMemcpy D2H into
 *               pinned and pageable memory matches, cudaMemset / D2D work
 *   remap       unmap a chunk and map the SAME handle again at the same VA: the data is intact; release it, create a
 *               new one at that VA: usable (a decommit / recommit)
 *   graph       a captured graph's kernel on a pool address replays correctly after that VA was decommitted and
 *               recommitted (graphs keep pointers; the VA must stay valid)
 *   memavail    mapping --memavail-gib GiB lowers MemAvailable by >= 90% of it; releasing it gives >= 90% back
 *   oversub     chunks are created while MemAvailable covers the next one over --floor-gib (the engine's commit
 *               gate): the gate must refuse cleanly (never more than a chunk under the floor, each create < 2 s, no new
 *               dmesg uvm / page-allocation / Xid line).  Then, INFO only, the ungated driver down to --stop-gib --
 *               on GB10 (gpuq #492) cuMemCreate did not refuse by itself, which is why every commit is gated
 *   timing      commit (create + map + set access) and decommit (unmap + release) per chunk, median / max, at 2 MiB
 *               and at --chunk-mib
 *
 * Exit 0 only when every check passed.  Model-free; needs the GPU to itself for the memavail / oversub checks (a
 * quiet gpuq job, production stopped).  Usage:
 *     ./tests/vmm_probe [--chunk-mib 64] [--memavail-gib 4] [--floor-gib 6] [--stop-gib 1.5] [--skip-oversub]
 *                       [--skip-ungated]
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

#define DRV(name) static PFN_##name p_##name = nullptr
typedef CUresult (*PFN_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*PFN_cuMemGetAllocationGranularity)(size_t *, const CUmemAllocationProp *,
                                                      CUmemAllocationGranularity_flags);
typedef CUresult (*PFN_cuMemAddressReserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*PFN_cuMemAddressFree)(CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemCreate)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *,
                                    unsigned long long);
typedef CUresult (*PFN_cuMemRelease)(CUmemGenericAllocationHandle);
typedef CUresult (*PFN_cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult (*PFN_cuMemUnmap)(CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
typedef CUresult (*PFN_cuGetErrorString)(CUresult, const char **);
DRV(cuDeviceGetAttribute);
DRV(cuMemGetAllocationGranularity);
DRV(cuMemAddressReserve);
DRV(cuMemAddressFree);
DRV(cuMemCreate);
DRV(cuMemRelease);
DRV(cuMemMap);
DRV(cuMemUnmap);
DRV(cuMemSetAccess);
DRV(cuGetErrorString);

int g_fails = 0;

void verdict(bool ok, const char *check, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
void verdict(bool ok, const char *check, const char *fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    printf("%s %-9s %s\n", ok ? "PASS" : "FAIL", check, msg);
    fflush(stdout);
    g_fails += !ok;
}

const char *cu_err(CUresult r) {
    const char *s = nullptr;
    if (p_cuGetErrorString && p_cuGetErrorString(r, &s) == CUDA_SUCCESS && s) return s;
    return "unknown CUresult";
}

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

uint64_t mem_available() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof line, fp))
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    fclose(fp);
    return (uint64_t)kib * 1024ull;
}

std::vector<std::string> dmesg_lines(bool *ok) {
    std::vector<std::string> out;
    *ok = false;
    FILE *p = popen("dmesg 2>/dev/null", "r");
    if (!p) return out;
    char line[1024];
    while (fgets(line, sizeof line, p)) out.emplace_back(line);
    *ok = pclose(p) == 0;
    return out;
}

template <class F>
bool load(F *fp, const char *name) {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || !p) {
        printf("FAIL load      the driver has no %s\n", name);
        return false;
    }
    *fp = (F)p;
    return true;
}

__global__ void fill_kernel(uint32_t *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] = seed ^ (uint32_t)(i * 2654435761u);
}

__global__ void check_kernel(const uint32_t *p, uint64_t n, uint32_t seed, unsigned long long *bad) {
    unsigned long long b = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        b += p[i] != (seed ^ (uint32_t)(i * 2654435761u));
    if (b) atomicAdd(bad, b);
}

__global__ void bump_kernel(uint32_t *p, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] += 1u;
}

CUmemAllocationProp g_prop;
CUmemAccessDesc g_access;

/* commit one chunk at va: create + map + set access; the handle into *h */
CUresult commit(CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle *h) {
    CUresult r = p_cuMemCreate(h, bytes, &g_prop, 0);
    if (r != CUDA_SUCCESS) return r;
    r = p_cuMemMap(va, bytes, 0, *h, 0);
    if (r != CUDA_SUCCESS) { p_cuMemRelease(*h); return r; }
    r = p_cuMemSetAccess(va, bytes, &g_access, 1);
    if (r != CUDA_SUCCESS) { p_cuMemUnmap(va, bytes); p_cuMemRelease(*h); }
    return r;
}

CUresult decommit(CUdeviceptr va, size_t bytes, CUmemGenericAllocationHandle h) {
    CUresult r = p_cuMemUnmap(va, bytes);
    CUresult r2 = p_cuMemRelease(h);
    return r != CUDA_SUCCESS ? r : r2;
}

/* fill + device check + D2H (pinned and pageable) of `bytes` at va with `seed`; false on any mismatch */
bool rw_ok(CUdeviceptr va, size_t bytes, uint32_t seed, uint32_t *pinned, uint32_t *pageable, unsigned long long *bad,
           char *why, size_t whylen) {
    const uint64_t n = bytes / 4;
    uint32_t *d = (uint32_t *)(uintptr_t)va;
    fill_kernel<<<256, 256>>>(d, n, seed);
    cudaMemset(bad, 0, sizeof *bad);
    check_kernel<<<256, 256>>>(d, n, seed, bad);
    unsigned long long hb = 0;
    if (cudaMemcpy(&hb, bad, sizeof hb, cudaMemcpyDeviceToHost) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        snprintf(why, whylen, "kernel / copy error: %s", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    if (hb) { snprintf(why, whylen, "%llu words wrong on the device", hb); return false; }
    const size_t probe = std::min<size_t>(bytes, 8u << 20);
    if (cudaMemcpy(pinned, d, probe, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(pageable, d, probe, cudaMemcpyDeviceToHost) != cudaSuccess) {
        snprintf(why, whylen, "D2H failed: %s", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    for (uint64_t i = 0; i < probe / 4; i++) {
        const uint32_t want = seed ^ (uint32_t)(i * 2654435761u);
        if (pinned[i] != want || pageable[i] != want) {
            snprintf(why, whylen, "D2H word %llu: pinned %08x pageable %08x want %08x", (unsigned long long)i,
                     pinned[i], pageable[i], want);
            return false;
        }
    }
    return true;
}

void stats(std::vector<double> v, double *med, double *mx) {
    std::sort(v.begin(), v.end());
    *med = v.empty() ? 0 : v[v.size() / 2];
    *mx = v.empty() ? 0 : v.back();
}

}  // namespace

int main(int argc, char **argv) {
    size_t chunk_mib = 64;
    double memavail_gib = 4.0, stop_gib = 1.5, floor_gib = 6.0;
    bool skip_oversub = false, skip_ungated = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--chunk-mib") && i + 1 < argc) chunk_mib = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--memavail-gib") && i + 1 < argc) memavail_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stop-gib") && i + 1 < argc) stop_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--floor-gib") && i + 1 < argc) floor_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--skip-oversub")) skip_oversub = true;
        else if (!strcmp(argv[i], "--skip-ungated")) skip_ungated = true;
        else {
            fprintf(stderr, "usage: %s [--chunk-mib N] [--memavail-gib G] [--floor-gib G] [--stop-gib G] [--skip-oversub] "
                            "[--skip-ungated]\n", argv[0]);
            return 2;
        }
    }
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) {
        printf("FAIL init      no CUDA device: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("vmm-probe: %s sm_%d%d, integrated %d, %.1f GiB, driver %d runtime %d\n", dp.name, dp.major, dp.minor,
           dp.integrated, (double)dp.totalGlobalMem / GiB, drv, rt);
    if (!load(&p_cuDeviceGetAttribute, "cuDeviceGetAttribute") || !load(&p_cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity") ||
        !load(&p_cuMemAddressReserve, "cuMemAddressReserve") || !load(&p_cuMemAddressFree, "cuMemAddressFree") ||
        !load(&p_cuMemCreate, "cuMemCreate") || !load(&p_cuMemRelease, "cuMemRelease") || !load(&p_cuMemMap, "cuMemMap") ||
        !load(&p_cuMemUnmap, "cuMemUnmap") || !load(&p_cuMemSetAccess, "cuMemSetAccess") ||
        !load(&p_cuGetErrorString, "cuGetErrorString"))
        return 1;

    /* ---- attr */
    int vmm = 0;
    p_cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0);
    memset(&g_prop, 0, sizeof g_prop);
    g_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    g_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    g_prop.location.id = 0;
    size_t gmin = 0, grec = 0;
    const CUresult rg1 = p_cuMemGetAllocationGranularity(&gmin, &g_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    const CUresult rg2 = p_cuMemGetAllocationGranularity(&grec, &g_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    verdict(vmm == 1 && rg1 == CUDA_SUCCESS && gmin > 0, "attr", "VMM supported %d, granularity min %zu KiB, recommended %zu KiB%s",
            vmm, gmin >> 10, grec >> 10, rg1 == CUDA_SUCCESS ? "" : cu_err(rg1));
    (void)rg2;
    if (vmm != 1 || gmin == 0) {
        printf("vmm-probe: STOP -- VMM is not usable on this device; the KV pools cannot move to it\n");
        return 1;
    }
    memset(&g_access, 0, sizeof g_access);
    g_access.location = g_prop.location;
    g_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    size_t chunk = chunk_mib << 20;
    chunk = (chunk + gmin - 1) / gmin * gmin;

    /* ---- reserve */
    const int NB = 16;
    const size_t bank_va = 8ull << 30;
    CUdeviceptr va[NB] = {};
    int reserved = 0;
    CUresult rr = CUDA_SUCCESS;
    for (int b = 0; b < NB; b++) {
        rr = p_cuMemAddressReserve(&va[b], bank_va, gmin, 0, 0);
        if (rr != CUDA_SUCCESS) break;
        reserved++;
    }
    verdict(reserved == NB, "reserve", "%d of %d x 8 GiB VA reserved%s%s", reserved, NB, rr ? ": " : "",
            rr ? cu_err(rr) : "");
    if (reserved != NB) return 1;

    uint32_t *pinned = nullptr;
    cudaMallocHost(&pinned, 8u << 20);
    std::vector<uint32_t> pageable((8u << 20) / 4);
    unsigned long long *bad = nullptr;
    cudaMalloc(&bad, sizeof *bad);

    /* ---- rw: two chunks per bank, the second at 4 GiB in (a sparse map) */
    CUmemGenericAllocationHandle h[NB][2] = {};
    bool rw = true;
    char why[256] = "";
    for (int b = 0; b < NB && rw; b++) {
        const CUdeviceptr at[2] = {va[b], va[b] + (4ull << 30)};
        for (int c = 0; c < 2 && rw; c++) {
            const CUresult r = commit(at[c], chunk, &h[b][c]);
            if (r != CUDA_SUCCESS) { snprintf(why, sizeof why, "commit bank %d chunk %d: %s", b, c, cu_err(r)); rw = false; break; }
            rw = rw_ok(at[c], chunk, 0x1000u * b + c, pinned, pageable.data(), bad, why, sizeof why);
        }
    }
    if (rw) {   /* cudaMemset + D2D across two banks' chunks */
        rw = cudaMemset((void *)(uintptr_t)va[1], 0x5a, chunk) == cudaSuccess &&
             cudaMemcpy((void *)(uintptr_t)va[2], (void *)(uintptr_t)va[1], chunk, cudaMemcpyDeviceToDevice) == cudaSuccess &&
             cudaMemcpy(pinned, (void *)(uintptr_t)va[2], 4096, cudaMemcpyDeviceToHost) == cudaSuccess &&
             ((const uint8_t *)pinned)[4095] == 0x5a;
        if (!rw) snprintf(why, sizeof why, "memset / D2D: %s", cudaGetErrorString(cudaGetLastError()));
    }
    verdict(rw, "rw", "%d banks x 2 chunks of %zu MiB: kernel write/check, D2H pinned + pageable, memset, D2D%s%s",
            NB, chunk >> 20, rw ? "" : " -- ", why);

    /* ---- remap: same handle, same VA, data intact; then release + a new handle at that VA */
    bool remap = rw;
    if (remap) {
        const CUdeviceptr v = va[3];
        remap = rw_ok(v, chunk, 0xabcd0003u, pinned, pageable.data(), bad, why, sizeof why) &&
                p_cuMemUnmap(v, chunk) == CUDA_SUCCESS && p_cuMemMap(v, chunk, 0, h[3][0], 0) == CUDA_SUCCESS &&
                p_cuMemSetAccess(v, chunk, &g_access, 1) == CUDA_SUCCESS;
        if (remap) {
            cudaMemset(bad, 0, sizeof *bad);
            check_kernel<<<256, 256>>>((const uint32_t *)(uintptr_t)v, chunk / 4, 0xabcd0003u, bad);
            unsigned long long hb = 1;
            cudaMemcpy(&hb, bad, sizeof hb, cudaMemcpyDeviceToHost);
            remap = hb == 0;
            if (!remap) snprintf(why, sizeof why, "%llu words changed across the unmap / map", hb);
        }
        if (remap) {
            remap = decommit(v, chunk, h[3][0]) == CUDA_SUCCESS && commit(v, chunk, &h[3][0]) == CUDA_SUCCESS &&
                    rw_ok(v, chunk, 0x77u, pinned, pageable.data(), bad, why, sizeof why);
            if (!remap && !why[0]) snprintf(why, sizeof why, "decommit / recommit at the same VA failed");
        }
    }
    verdict(remap, "remap", "same handle remapped at the same VA keeps its data; release + a new handle there is usable%s%s",
            remap ? "" : " -- ", why);

    /* ---- graph: a captured kernel on a pool address replays after its VA was decommitted and recommitted */
    bool graph_ok = remap;
    if (graph_ok) {
        const CUdeviceptr v = va[4];
        uint32_t *d = (uint32_t *)(uintptr_t)v;
        cudaStream_t st;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        cudaGraph_t g;
        cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        bump_kernel<<<256, 256, 0, st>>>(d, chunk / 4);
        cudaStreamEndCapture(st, &g);
        graph_ok = cudaGraphInstantiate(&ge, g, 0) == cudaSuccess;
        if (graph_ok) {
            graph_ok = decommit(v, chunk, h[4][0]) == CUDA_SUCCESS && commit(v, chunk, &h[4][0]) == CUDA_SUCCESS &&
                       cudaMemset(d, 0, chunk) == cudaSuccess && cudaGraphLaunch(ge, st) == cudaSuccess &&
                       cudaStreamSynchronize(st) == cudaSuccess &&
                       cudaMemcpy(pinned, d, 4096, cudaMemcpyDeviceToHost) == cudaSuccess && pinned[0] == 1u &&
                       pinned[1023] == 1u;
            cudaGraphExecDestroy(ge);
        }
        cudaGraphDestroy(g);
        cudaStreamDestroy(st);
        if (!graph_ok) snprintf(why, sizeof why, "replay after recommit: %s (word0 %u)",
                                cudaGetErrorString(cudaGetLastError()), pinned[0]);
    }
    verdict(graph_ok, "graph", "a graph captured on a pool VA replays after a decommit / recommit of it%s%s",
            graph_ok ? "" : " -- ", why);

    /* the rw / remap / graph chunks go before the checks that map from each bank's start */
    for (int bb = 0; bb < NB; bb++)
        for (int c = 0; c < 2; c++)
            if (h[bb][c]) decommit(c ? va[bb] + (4ull << 30) : va[bb], chunk, h[bb][c]);

    /* ---- memavail: mapped physical is system memory and MemAvailable says so */
    {
        const size_t n = (size_t)(memavail_gib * GiB) / chunk;
        std::vector<CUmemGenericAllocationHandle> hs(n);
        const uint64_t m0 = mem_available();
        size_t made = 0;
        for (; made < n; made++)
            if (commit(va[5] + made * chunk, chunk, &hs[made]) != CUDA_SUCCESS) break;
        cudaDeviceSynchronize();
        usleep(500000);
        const uint64_t m1 = mem_available();
        for (size_t i = 0; i < made; i++) decommit(va[5] + i * chunk, chunk, hs[i]);
        usleep(500000);
        const uint64_t m2 = mem_available();
        const double mapped = (double)made * chunk, dropped = (double)m0 - (double)m1, back = (double)m2 - (double)m1;
        verdict(made == n && dropped >= 0.9 * mapped && back >= 0.9 * mapped, "memavail",
                "mapped %.2f GiB: MemAvailable %.2f -> %.2f (-%.2f), released -> %.2f (+%.2f) GiB", mapped / GiB,
                m0 / GiB, m1 / GiB, dropped / GiB, m2 / GiB, back / GiB);
    }

    /* ---- timing: commit / decommit per chunk at the granularity and at the chunk size */
    for (size_t sz : {gmin, chunk}) {
        const int reps = sz == gmin ? 256 : 32;
        std::vector<double> tc, td;
        std::vector<CUmemGenericAllocationHandle> hs(reps);
        bool ok = true;
        for (int i = 0; i < reps && ok; i++) {
            const double t0 = now_ms();
            ok = commit(va[6] + (size_t)i * sz, sz, &hs[i]) == CUDA_SUCCESS;
            tc.push_back(now_ms() - t0);
        }
        for (int i = 0; i < reps && ok; i++) {
            const double t0 = now_ms();
            ok = decommit(va[6] + (size_t)i * sz, sz, hs[i]) == CUDA_SUCCESS;
            td.push_back(now_ms() - t0);
        }
        double cm, cx, dm, dx;
        stats(tc, &cm, &cx);
        stats(td, &dm, &dx);
        verdict(ok, "timing", "%zu KiB chunks x %d: commit median %.3f ms max %.3f, decommit median %.3f ms max %.3f",
                sz >> 10, reps, cm, cx, dm, dx);
    }

    /* ---- oversub (gpuq #492: GB10's cuMemCreate does NOT refuse before the kernel's margin -- it kept succeeding
     * down to 1.47 GiB, retrying inside the driver).  So the engine gates every commit by a floor (the runtime's
     * pulsar_gpu_set_commit_floor), and this check is that rule: chunks are created only while MemAvailable covers the
     * next chunk over --floor-gib.  PASS: the gate refuses cleanly (MemAvailable never under the floor by more than a
     * chunk, every create < 2 s, no new dmesg uvm / page-allocation / Xid line).  Then, informational, the ungated
     * driver from there to --stop-gib: does it refuse by itself? (INFO line, never a FAIL.) */
    if (skip_oversub) {
        printf("SKIP oversub   (--skip-oversub)\n");
    } else {
        bool dm_ok0 = false;
        const std::vector<std::string> d0 = dmesg_lines(&dm_ok0);
        std::vector<std::pair<CUdeviceptr, CUmemGenericAllocationHandle>> held;
        const size_t ov_bytes = ((size_t)dp.totalGlobalMem * 2 + gmin - 1) / gmin * gmin;
        CUdeviceptr ov = 0;
        const CUresult rv = p_cuMemAddressReserve(&ov, ov_bytes, gmin, 0, 0);
        size_t off = 0;
        double slowest = 0;
        uint64_t low = mem_available();
        bool gated_refused = false;
        CUresult last = CUDA_SUCCESS;
        for (; rv == CUDA_SUCCESS && off + chunk <= ov_bytes;) {
            const uint64_t av = mem_available();
            low = std::min(low, av);
            if ((double)av < floor_gib * GiB + (double)chunk) { gated_refused = true; break; }   /* the gate */
            CUmemGenericAllocationHandle hh;
            const double t0 = now_ms();
            last = commit(ov + off, chunk, &hh);
            slowest = std::max(slowest, now_ms() - t0);
            if (last != CUDA_SUCCESS) break;
            held.push_back({ov + off, hh});
            off += chunk;
        }
        low = std::min(low, mem_available());
        const size_t gated_chunks = held.size();
        bool dm_ok1 = false;
        const std::vector<std::string> d1 = dmesg_lines(&dm_ok1);
        int hits = 0;
        std::string first;
        if (dm_ok0 && dm_ok1)
            for (size_t i = d0.size(); i < d1.size(); i++)
                if (d1[i].find("page allocation failure") != std::string::npos ||
                    d1[i].find("uvm") != std::string::npos || d1[i].find("Xid") != std::string::npos) {
                    if (!hits++) first = d1[i];
                }
        verdict(gated_refused && (double)low >= floor_gib * GiB - (double)chunk && slowest < 2000.0 && hits == 0,
                "oversub", "a commit gated at a %.2f GiB floor refused cleanly after %zu chunks (%.2f GiB): MemAvailable "
                "low %.2f GiB, slowest create %.1f ms, dmesg %s%s%s", floor_gib, gated_chunks,
                (double)gated_chunks * chunk / GiB, low / GiB, slowest,
                !(dm_ok0 && dm_ok1) ? "unreadable (run with dmesg access to check)" : hits ? first.c_str() : "clean",
                last != CUDA_SUCCESS ? "; the driver refused first: " : "", last != CUDA_SUCCESS ? cu_err(last) : "");
        /* informational: the ungated driver, from the floor down to --stop-gib */
        if (gated_refused && !skip_ungated) {
            bool stopped = false;
            double slow_u = 0;
            CUresult lu = CUDA_SUCCESS;
            for (; off + chunk <= ov_bytes;) {
                if ((double)mem_available() < stop_gib * GiB) { stopped = true; break; }
                CUmemGenericAllocationHandle hh;
                const double t0 = now_ms();
                lu = commit(ov + off, chunk, &hh);
                slow_u = std::max(slow_u, now_ms() - t0);
                if (lu != CUDA_SUCCESS) break;
                held.push_back({ov + off, hh});
                off += chunk;
            }
            const std::string what = lu != CUDA_SUCCESS ? std::string("the driver refused: ") + cu_err(lu)
                                     : stopped ? std::string("the driver did NOT refuse before --stop-gib")
                                               : std::string("the VA filled");
            printf("INFO ungated   %zu more chunks past the floor: %s at MemAvailable %.2f GiB; slowest create %.1f ms "
                   "(why the engine gates every commit)\n", held.size() - gated_chunks, what.c_str(),
                   mem_available() / GiB, slow_u);
            fflush(stdout);
        }
        for (auto &x : held) decommit(x.first, chunk, x.second);
        if (rv == CUDA_SUCCESS) p_cuMemAddressFree(ov, ov_bytes);
    }

    for (int bb = 0; bb < NB; bb++) p_cuMemAddressFree(va[bb], bank_va);
    cudaFreeHost(pinned);
    cudaFree(bad);
    printf("vmm-probe: %s (%d failed)\n", g_fails ? "FAIL" : "PASS", g_fails);
    return g_fails ? 1 : 0;
}

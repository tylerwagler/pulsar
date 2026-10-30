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
 *   memavail    mapping --memavail-gib GiB lowers MemAvailable by the mapped bytes and releasing gives them back:
 *               5 rounds, the MEDIAN drop and return each within 15% (one reading is cache noise), nothing leaked
 *   oversub     chunks are created while MemAvailable covers the next one over --floor-gib (the engine's commit
 *               gate): the gate must refuse cleanly (never more than a chunk under the floor, each create < 2 s, no new
 *               dmesg uvm / page-allocation / Xid line).  Then, INFO only, the ungated driver down to --stop-gib --
 *               on GB10 (gpuq #492) cuMemCreate did not refuse by itself, which is why every commit is gated
 *   INFO        what taking the physical costs the box: /proc/vmstat reclaim (pgscan / pgsteal) and compaction
 *               (compact_stall / fail / success) deltas and PSI memory, for the VMM rounds, for the same bytes as one
 *               GPU-touched cudaMallocManaged buffer (the old pools' allocator), and for the gated oversubscription
 *   pool        at that floor, the chunks the gated phase holds are a pre-created pool: unmapped and mapped at
 *               another VA (a spill and a readmit) round after round, with no compaction stall (gpuq #504: the server
 *               creates its chunks once at startup and only maps them while serving)
 *   cache       (with --cache-file, INFO) a pool-sized reserve right after filling the page cache with that file,
 *               with its cache kept and after dropping it: time, compaction stalls, reclaim, PSI (gpuq #505)
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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>

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

/* the kernel's reclaim / compaction counters (/proc/vmstat) and its memory pressure (/proc/pressure/memory): what
 * taking physical costs the box -- gpuq #503's squeeze leg was killed by systemd-oomd on PSI with Pgscan 81M */
struct vmstat_snap {
    unsigned long long pgscan = 0, pgsteal = 0, compact_stall = 0, compact_fail = 0, compact_success = 0;
};
vmstat_snap vmstat_read() {
    vmstat_snap v;
    FILE *fp = fopen("/proc/vmstat", "r");
    if (!fp) return v;
    char k[128];
    unsigned long long x = 0;
    while (fscanf(fp, "%127s %llu", k, &x) == 2) {
        if (!strncmp(k, "pgscan_", 7)) v.pgscan += x;
        else if (!strncmp(k, "pgsteal_", 8)) v.pgsteal += x;
        else if (!strcmp(k, "compact_stall")) v.compact_stall = x;
        else if (!strcmp(k, "compact_fail")) v.compact_fail = x;
        else if (!strcmp(k, "compact_success")) v.compact_success = x;
    }
    fclose(fp);
    return v;
}
std::string psi_memory() {
    FILE *fp = fopen("/proc/pressure/memory", "r");
    if (!fp) return "unreadable";
    char line[256];
    std::string out;
    while (fgets(line, sizeof line, fp)) {
        line[strcspn(line, "\n")] = 0;
        if (!out.empty()) out += " | ";
        out += line;
    }
    fclose(fp);
    return out;
}
void vmstat_info(const char *what, const vmstat_snap &a, const vmstat_snap &b) {
    printf("INFO %-9s pgscan +%llu, pgsteal +%llu, compact stalls +%llu (fail +%llu, success +%llu); PSI memory: %s\n",
           what, b.pgscan - a.pgscan, b.pgsteal - a.pgsteal, b.compact_stall - a.compact_stall,
           b.compact_fail - a.compact_fail, b.compact_success - a.compact_success, psi_memory().c_str());
    fflush(stdout);
}

uint64_t mem_free() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof line, fp))
        if (sscanf(line, "MemFree: %llu kB", &kib) == 1) break;
    fclose(fp);
    return (uint64_t)kib * 1024ull;
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
    const char *cache_file = nullptr;
    double cache_reserve_gib = 20.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--chunk-mib") && i + 1 < argc) chunk_mib = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--memavail-gib") && i + 1 < argc) memavail_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stop-gib") && i + 1 < argc) stop_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--floor-gib") && i + 1 < argc) floor_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--skip-oversub")) skip_oversub = true;
        else if (!strcmp(argv[i], "--skip-ungated")) skip_ungated = true;
        else if (!strcmp(argv[i], "--cache-file") && i + 1 < argc) cache_file = argv[++i];
        else if (!strcmp(argv[i], "--cache-reserve-gib") && i + 1 < argc) cache_reserve_gib = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--chunk-mib N] [--memavail-gib G] [--floor-gib G] [--stop-gib G] [--skip-oversub] "
                            "[--skip-ungated] [--cache-file PATH] [--cache-reserve-gib G]\n", argv[0]);
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

    /* ---- memavail: mapped physical is system memory and MemAvailable says so.  One reading is noisy (gpuq #498:
     * -3.42 GiB for 4.00 mapped -- MemAvailable counts reclaimable cache the kernel shifts while we map), so the
     * property is measured over MEMAVAIL_REPS map / release rounds, each after a settle, and graded on the MEDIAN:
     * the drop and the return each within MEMAVAIL_TOL of the mapped bytes (MemFree is reported beside it), and
     * nothing leaks -- MemAvailable after the last release within MEMAVAIL_TOL of the mapped bytes of where it began. */
    {
        const int MEMAVAIL_REPS = 5;
        const double MEMAVAIL_TOL = 0.15;
        const size_t n = (size_t)(memavail_gib * GiB) / chunk;
        std::vector<CUmemGenericAllocationHandle> hs(n);
        std::vector<double> drop_a, back_a, drop_f;
        bool made_all = true;
        sync();
        usleep(1000000);
        const uint64_t start = mem_available();
        const vmstat_snap vs_start = vmstat_read();
        for (int rep = 0; rep < MEMAVAIL_REPS && made_all; rep++) {
            const uint64_t a0 = mem_available(), f0 = mem_free();
            size_t made = 0;
            for (; made < n; made++)
                if (commit(va[5] + made * chunk, chunk, &hs[made]) != CUDA_SUCCESS) break;
            made_all = made == n;
            cudaDeviceSynchronize();
            usleep(1000000);
            const uint64_t a1 = mem_available(), f1 = mem_free();
            for (size_t i = 0; i < made; i++) decommit(va[5] + i * chunk, chunk, hs[i]);
            usleep(1000000);
            const uint64_t a2 = mem_available();
            drop_a.push_back((double)a0 - (double)a1);
            back_a.push_back((double)a2 - (double)a1);
            drop_f.push_back((double)f0 - (double)f1);
        }
        const uint64_t end = mem_available();
        const vmstat_snap vs_end = vmstat_read();
        vmstat_info("vmm-cost", vs_start, vs_end);
        /* the same bytes as one cudaMallocManaged buffer the GPU touches (the pools' old allocator): what the managed
         * path took from the box, beside the VMM chunks' -- MemFree / MemAvailable and the same counters */
        {
            void *mp = nullptr;
            const size_t mb = (size_t)n * chunk;
            const uint64_t a0 = mem_available(), f0 = mem_free();
            const vmstat_snap m0 = vmstat_read();
            if (cudaMallocManaged(&mp, mb) == cudaSuccess) {
                fill_kernel<<<256, 256>>>((uint32_t *)mp, mb / 4, 7u);
                cudaDeviceSynchronize();
                usleep(1000000);
                const uint64_t a1 = mem_available(), f1 = mem_free();
                const vmstat_snap m1 = vmstat_read();
                cudaFree(mp);
                printf("INFO managed   the same %.2f GiB as cudaMallocManaged, GPU-touched: MemAvailable -%.2f, MemFree "
                       "-%.2f GiB\n", (double)mb / GiB, ((double)a0 - (double)a1) / GiB,
                       ((double)f0 - (double)f1) / GiB);
                vmstat_info("mgd-cost", m0, m1);
            } else {
                printf("INFO managed   cudaMallocManaged of %.2f GiB failed: %s\n", (double)mb / GiB,
                       cudaGetErrorString(cudaGetLastError()));
            }
        }
        double da, xa, ba, xb, df, xf;
        stats(drop_a, &da, &xa);
        stats(back_a, &ba, &xb);
        stats(drop_f, &df, &xf);
        const double mapped = (double)n * chunk;
        const auto within = [&](double v) { return v >= (1.0 - MEMAVAIL_TOL) * mapped && v <= (1.0 + MEMAVAIL_TOL) * mapped; };
        const double leak = (double)start - (double)end;
        verdict(made_all && within(da) && within(ba) && leak <= MEMAVAIL_TOL * mapped, "memavail",
                "mapped %.2f GiB x %d: median MemAvailable -%.2f (MemFree -%.2f), released +%.2f GiB (within %.0f%%); "
                "after the last release %.2f GiB from where it began", mapped / GiB, (int)drop_a.size(), da / GiB,
                df / GiB, ba / GiB, MEMAVAIL_TOL * 100.0, -leak / GiB);
        (void)xa;
        (void)xb;
        (void)xf;
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
        const vmstat_snap vs_ov0 = vmstat_read();
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
        vmstat_info("gated", vs_ov0, vmstat_read());
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
        /* ---- pool (gpuq #504): the server creates its chunks once, at startup, and from then on only MAPS them.  Here,
         * at the floor the gated phase left the box at, the chunks it holds are the pre-created pool: POOL_CYCLES
         * rounds unmap POOL_BATCH of them and map them again at another VA (a spill and a readmit), checking the data
         * on the new VA.  PASS: no compaction stall past what the idle box shows over as long (the counter is the
         * system's), every map < 50 ms (the reclaim scan is reported). */
        if (gated_refused && held.size() >= 64) {
            const int POOL_CYCLES = 50;
            const size_t POOL_BATCH = 32;
            CUdeviceptr pv = 0;
            const bool pv_ok = p_cuMemAddressReserve(&pv, POOL_BATCH * chunk, gmin, 0, 0) == CUDA_SUCCESS;
            const vmstat_snap p0 = vmstat_read();
            const double tc0 = now_ms();
            double slow_map = 0;
            bool pool_ok = pv_ok;
            for (int cyc = 0; pool_ok && cyc < POOL_CYCLES; cyc++) {
                for (size_t i = 0; pool_ok && i < POOL_BATCH; i++) {
                    auto &x = held[held.size() - 1 - i];
                    const double t0 = now_ms();
                    pool_ok = p_cuMemUnmap(x.first, chunk) == CUDA_SUCCESS &&
                              p_cuMemMap(pv + i * chunk, chunk, 0, x.second, 0) == CUDA_SUCCESS &&
                              p_cuMemSetAccess(pv + i * chunk, chunk, &g_access, 1) == CUDA_SUCCESS;
                    slow_map = std::max(slow_map, now_ms() - t0);
                }
                if (pool_ok && cyc == POOL_CYCLES - 1)
                    pool_ok = rw_ok(pv, chunk, 0x900du, pinned, pageable.data(), bad, why, sizeof why);
                for (size_t i = 0; pool_ok && i < POOL_BATCH; i++) {   /* back where they were */
                    auto &x = held[held.size() - 1 - i];
                    pool_ok = p_cuMemUnmap(pv + i * chunk, chunk) == CUDA_SUCCESS &&
                              p_cuMemMap(x.first, chunk, 0, x.second, 0) == CUDA_SUCCESS &&
                              p_cuMemSetAccess(x.first, chunk, &g_access, 1) == CUDA_SUCCESS;
                }
            }
            const vmstat_snap p1 = vmstat_read();
            const double took = now_ms() - tc0;
            if (pv_ok) p_cuMemAddressFree(pv, POOL_BATCH * chunk);
            /* gpuq #506: compact_stall is system-wide -- at the floor another process's allocations stall too (#506
             * read +15 here).  So the same span is measured again with the probe idle, still at the floor: the maps
             * cause a stall only past what the idle box shows in as long. */
            const vmstat_snap q0 = vmstat_read();
            usleep((useconds_t)(took * 1000.0));
            const vmstat_snap q1 = vmstat_read();
            const unsigned long long busy_st = p1.compact_stall - p0.compact_stall, idle_st = q1.compact_stall - q0.compact_stall;
            verdict(pool_ok && busy_st <= idle_st && slow_map < 50.0,
                    "pool", "%d rounds of %zu pre-created chunks unmapped and mapped at another VA at MemAvailable "
                    "%.2f GiB in %.2f s: compaction stalls +%llu (the idle box in as long: +%llu), pgscan +%llu, "
                    "slowest map %.2f ms%s%s", POOL_CYCLES, POOL_BATCH, mem_available() / GiB, took / 1e3, busy_st,
                    idle_st, p1.pgscan - p0.pgscan, slow_map, pool_ok ? "" : " -- ", pool_ok ? "" : why);
        } else {
            verdict(false, "pool", "no pre-created chunks to cycle (the gated phase did not reach its floor)");
        }
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

    /* ---- cache (gpuq #505: the 21.6 GiB pool reserve right after a 62 GB model load drove PSI to 80 % and oomd
     * killed the job): fill the page cache with --cache-file (a model's shards, ~60 GB), then create
     * --cache-reserve-gib of chunks the way the server's pool does, twice -- with the file's cache kept, and after
     * dropping it (posix_fadvise DONTNEED) -- reporting time, compaction stalls, reclaim and the highest PSI.
     * INFO only: the numbers that choose the reserve's order and pacing. */
    if (cache_file) {
        /* a file, or every regular file in a directory (a model's shards) */
        std::vector<std::string> files;
        if (DIR *d = opendir(cache_file)) {
            while (struct dirent *de = readdir(d))
                if (de->d_name[0] != '.') files.push_back(std::string(cache_file) + "/" + de->d_name);
            closedir(d);
        } else {
            files.push_back(cache_file);
        }
        for (int pass = 0; pass < 2; pass++) {
            std::vector<char> buf(64u << 20);
            uint64_t read_bytes = 0;
            for (const std::string &f : files) {
                const int fd = open(f.c_str(), O_RDONLY);
                if (fd < 0) continue;
                for (ssize_t r; (r = read(fd, buf.data(), buf.size())) > 0;) read_bytes += (uint64_t)r;
                if (pass == 1) (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
            if (read_bytes == 0) { printf("INFO cache     %s: nothing readable\n", cache_file); break; }
            usleep(1000000);
            const uint64_t a0 = mem_available();
            double want = cache_reserve_gib * GiB;
            const double cap = (double)a0 - (floor_gib + 2.0) * GiB;
            if (want > cap) want = cap > 0 ? cap : 0;
            const size_t n = (size_t)(want / (double)gmin);
            std::vector<CUmemGenericAllocationHandle> hs;
            hs.reserve(n);
            const vmstat_snap v0 = vmstat_read();
            const double t0 = now_ms();
            double psi_max = 0, slowest = 0;
            for (size_t i = 0; i < n; i++) {
                if (i % 128 == 0) {
                    const std::string ps = psi_memory();
                    double v = 0;
                    if (sscanf(ps.c_str(), "some avg10=%lf", &v) == 1 && v > psi_max) psi_max = v;
                }
                CUmemGenericAllocationHandle h;
                const double c0 = now_ms();
                if (p_cuMemCreate(&h, gmin, &g_prop, 0) != CUDA_SUCCESS) break;
                slowest = std::max(slowest, now_ms() - c0);
                hs.push_back(h);
            }
            const double took = now_ms() - t0;
            const vmstat_snap v1 = vmstat_read();
            for (auto h : hs) p_cuMemRelease(h);
            printf("INFO cache     %s cache (%.1f GiB read): %zu chunks (%.2f GiB) created in %.2f s, slowest %.1f ms; "
                   "compact stalls +%llu (fail +%llu), pgscan +%llu; PSI some avg10 at most %.1f; MemAvailable before "
                   "%.2f GiB\n", pass ? "after dropping the file's" : "with the file's", (double)read_bytes / GiB,
                   hs.size(), (double)hs.size() * gmin / GiB, took / 1e3, slowest,
                   v1.compact_stall - v0.compact_stall, v1.compact_fail - v0.compact_fail, v1.pgscan - v0.pgscan,
                   psi_max, a0 / GiB);
            fflush(stdout);
            usleep(5000000);   /* let the pressure average settle between the passes */
        }
    }

    for (int bb = 0; bb < NB; bb++) p_cuMemAddressFree(va[bb], bank_va);
    cudaFreeHost(pinned);
    cudaFree(bad);
    printf("vmm-probe: %s (%d failed)\n", g_fails ? "FAIL" : "PASS", g_fails);
    return g_fails ? 1 : 0;
}

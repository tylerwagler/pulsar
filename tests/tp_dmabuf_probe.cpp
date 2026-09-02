// tp_dmabuf_probe — GDR (GPUDirect RDMA) capability probe, verified at the box.
//
// Runs the full dma-buf GPUDirect sequence and reports how far the platform
// gets: VMM pinned allocation (cuMemCreate + cuMemMap + cuMemSetAccess), the
// CUDA-13 GPUDirect-RDMA allocation flag (allocFlags.gpuDirectRDMACapable),
// dma-buf export (cuMemExportToShareableHandle, POSIX fd), and HCA import
// (ibv_reg_dmabuf_mr, IBVERBS_1.12; the transport dlopens libibverbs, so this
// symbol resolves the same way).
//
// VERDICT on the pair (ca1070wk30007/008, driver 610.57.04, 2026-09-02):
//   VMM_SUPPORTED=1  GPU_DIRECT_RDMA_SUPPORTED=0  GDR_WITH_VMM_SUPPORTED=0
//   cuMemCreate rejects the GPURDMA flag ("invalid device ordinal"); the
//   plain-pinned dma-buf exports fine but the NIC refuses it (EINVAL).
//   GB10 does NOT expose GPUDirect RDMA in either form, so the slab class is
//   host-pinned (cudaHostRegister + ibv_reg_mr), which the bring-up probes
//   already use and verify.  Re-run this probe only if the nvidia driver ever
//   advertises CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
//   (=110) — that is the switch that would reopen the device-resident slab.
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cnd, ...)                                              \
    do {                                                             \
        if (!(cnd)) {                                                \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
            failures++;                                              \
        }                                                            \
    } while (0)

static const char *cuda_err(CUresult r) {
    const char *s = NULL;
    cuGetErrorString(r, &s);
    return s ? s : "unknown";
}

int main(void) {
    CUresult cr = cuInit(0);
    CHECK(cr == CUDA_SUCCESS, "cuInit: %s", cuda_err(cr));
    if (cr != CUDA_SUCCESS) return 1;

    CUdevice dev = 0;
    CHECK(cuDeviceGet(&dev, 0) == CUDA_SUCCESS, "cuDeviceGet");
    {
        int v = 0;
        cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
        std::printf("VMM_SUPPORTED=%d\n", v);
        cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
        std::printf("GPU_DIRECT_RDMA_SUPPORTED=%d\n", v);
        cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev);
        std::printf("GDR_WITH_VMM_SUPPORTED=%d\n", v);
    }
    CUcontext ctx = NULL;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS, "cuDevicePrimaryCtxRetain");
    CHECK(cuCtxSetCurrent(ctx) == CUDA_SUCCESS, "cuCtxSetCurrent");

    /* VMM pinned allocation, rounded up to the driver's allocation granularity */
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    /* CUDA 13: the GPUDirect RDMA flag (attr 110/116 supported) makes a pinned
     * allocation importable by the NIC -- cuMemExportToShareableHandle then
     * yields a dma-buf fd that ibv_reg_dmabuf_mr accepts. */
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t gran = 0;
    CHECK(cuMemGetAllocationGranularity(&gran, &prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS,
          "cuMemGetAllocationGranularity");
    size_t len = (size_t)(1u << 20); /* want 1 MiB */
    len = ((len + gran - 1) / gran) * gran;

    CUmemGenericAllocationHandle alloc = 0;
    cr = cuMemCreate(&alloc, len, &prop, 0);
    if (cr != CUDA_SUCCESS && prop.allocFlags.gpuDirectRDMACapable) {
        /* the GDR flag alone can be the problem; retry plain pinned to isolate */
        std::printf("cuMemCreate(+GDR): %s -- retrying plain pinned\n", cuda_err(cr));
        prop.allocFlags.gpuDirectRDMACapable = 0;
        cr = cuMemCreate(&alloc, len, &prop, 0);
    }
    CHECK(cr == CUDA_SUCCESS, "cuMemCreate(%zu): %s", len, cuda_err(cr));
    if (cr != CUDA_SUCCESS) return 1;

    CUdeviceptr dptr = 0;
    cr = cuMemAddressReserve(&dptr, len, 0, 0, 0);
    CHECK(cr == CUDA_SUCCESS, "cuMemAddressReserve: %s", cuda_err(cr));
    cr = cuMemMap((CUdeviceptr)dptr, len, 0, alloc, 0);
    CHECK(cr == CUDA_SUCCESS, "cuMemMap: %s", cuda_err(cr));
    {
        CUmemAccessDesc access;
        std::memset(&access, 0, sizeof(access));
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = dev;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cr = cuMemSetAccess((CUdeviceptr)dptr, len, &access, 1);
        CHECK(cr == CUDA_SUCCESS, "cuMemSetAccess: %s", cuda_err(cr));
    }
    /* touch the pages so they're resident before importing */
    CHECK(cuMemsetD8Async((CUdeviceptr)dptr, 0xAB, len, NULL) == CUDA_SUCCESS,
          "cuMemsetD8Async");
    cuCtxSynchronize();

    /* export a dma-buf fd for the whole allocation */
    int fd = -1;
    cr = cuMemExportToShareableHandle(&fd, alloc,
                                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    CHECK(cr == CUDA_SUCCESS, "cuMemExportToShareableHandle(POSIX_FD): %s", cuda_err(cr));
    if (cr != CUDA_SUCCESS) return 1;
    std::printf("exported dma-buf fd = %d (len %zu)\n", fd, len);

    /* open the HCA and register the dma-buf */
    const char *want = getenv("PULSAR_TP_RDMA_DEV");
    struct ibv_device **list = ibv_get_device_list(NULL);
    struct ibv_context *hca = NULL;
    if (list) {
        for (int i = 0; list[i] && !hca; i++) {
            if (!want || !std::strcmp(list[i]->name, want))
                hca = ibv_open_device(list[i]);
        }
    }
    CHECK(hca != NULL, "no verbs device (PULSAR_TP_RDMA_DEV=%s)", want ? want : "auto");
    if (hca) {
        struct ibv_pd *pd = ibv_alloc_pd(hca);
        CHECK(pd != NULL, "ibv_alloc_pd: %s", strerror(errno));
        if (pd) {
            struct ibv_mr *mr = ibv_reg_dmabuf_mr(
                pd, 0 /*offset*/, len /*length*/, (uint64_t)dptr /*addr*/, fd,
                IBV_ACCESS_LOCAL_WRITE);
            CHECK(mr != NULL,
                  "ibv_reg_dmabuf_mr (dma-buf fd %d): %s (errno %d) "
                  "[peermem loaded?]",
                  fd, strerror(errno), errno);
            if (mr) {
                std::printf("reg_dmabuf OK: mr addr %p len %zu\n", mr->addr,
                            mr->length);
                CHECK(ibv_dereg_mr(mr) == 0, "ibv_dereg_mr failed");
            }
            ibv_dealloc_pd(pd);
        }
        ibv_close_device(hca);
    }
    ibv_free_device_list(list);

    close(fd);
    cuMemUnmap(dptr, len);
    cuMemAddressFree(dptr, len);
    cuMemRelease(alloc);
    cuCtxDestroy(ctx);

    if (failures) {
        std::fprintf(stderr, "tp_dmabuf_probe: %d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("tp_dmabuf_probe: ok (VMM pinned -> dma-buf fd -> ibv_reg_dmabuf_mr)\n");
    return 0;
}

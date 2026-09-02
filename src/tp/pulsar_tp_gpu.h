#ifndef PULSAR_TP_GPU_H
#define PULSAR_TP_GPU_H

/*
 * GPU-side slab support for the two-rank TP port (slice 4b ground work).
 *
 * Callers: the engine worker thread (slice 4c), and the on-pair probes
 * (tests/tp_slab_gpu_probe).  Deliberately host C++ -- it only needs the CUDA
 * RUNTIME (cudaHostRegister/Mapped), so it compiles without nvcc and pulls no
 * device kernels in.  The pure-.cu gate kernels (pulsar_gpu_tp_{init,gate_...
 * encode,batch,big,kick,wait}) land in 4c WITH their engine callers.
 *
 * Slab class (runbook step 4, verified on the pair 2026-09-02): GB10 will not
 * HCA-register cudaMallocManaged pages (ibv_reg_mr -> EFAULT) and exposes no
 * GPUDirect RDMA (tests/tp_dmabuf_probe: attrs 110/116 = 0), so the slab is
 * page-aligned host RAM, page-locked and GPU-mapped with cudaHostRegister
 * (+Mapped).  ibv_reg_mr (called by pulsar_tp_attach_slab) accepts it, and
 * GB10 unified memory lets kernels write it directly -- no D2H/H2D bounce.
 */

#include <stddef.h>

/* Allocate a host-pinned, GPU-visible slab of `bytes` for the transport.
 * Returns 1 on success (base valid), 0 on failure with err set. */
int pulsar_tp_gpu_slab_alloc_hostpin(size_t bytes, void **base,
                                     char *err, size_t errlen);

/* Release a host-pinned slab from pulsar_tp_gpu_slab_alloc_hostpin. */
void pulsar_tp_gpu_slab_free_hostpin(void *base);

#endif

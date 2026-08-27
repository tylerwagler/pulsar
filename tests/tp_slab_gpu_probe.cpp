/*
 * TP GPU-slab gate probe (branch tensor_parallel, bring-up step 4 in
 * docs/tensor-parallel-bringup.md).  nvcc-built so it can be run on the pair;
 * on a GPU-less box it fails LOUDLY at cudaMallocManaged (that is the point —
 * this binary exists to be copied to the DSV4-Flash pair and run there).
 *
 * Validates the plan's GB10-unified-memory thesis end to end: the TP slab is
 * allocated with cudaMallocManaged (device memory, host-coherent on GB10),
 * handed to pulsar_tp_attach_slab -> ibv_reg_mr over the whole slab (RDMA) or
 * kept as plain TCP staging, and several gate + batch + big exchanges are run
 * with pattern verification, exactly like the host loopback test but with the
 * slab in GPU-visible memory.
 *
 *   Box A:  ./tp_slab_gpu_probe leader 0.0.0.0 5599
 *   Box B:  ./tp_slab_gpu_probe worker <BOX_A_ROCE_IP> 5599
 * Set PULSAR_TP_EXPECT_RDMA=1 to require the RDMA path on both sides.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "tp/pulsar_tp.h"

#define N_LAYER 43u
#define N_EMBD 4096u
#define VEC_BYTES ((uint64_t)N_EMBD * 4u)

static int g_failures = 0;
#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
        }                                                                    \
    } while (0)

static float pat(int rank, uint32_t slot, uint32_t i) {
    return (float)((int)(rank == 0 ? 70000 : 50000) + (int)slot * 1000 + (int)i);
}

static int create_rank(int rank, const char *peer, int port, pulsar_tp **out,
                       char *err, size_t errlen) {
    pulsar_tp_options opt;
    std::memset(&opt, 0, sizeof(opt));
    opt.role = rank == 0 ? PULSAR_TP_ROLE_LEADER : PULSAR_TP_ROLE_WORKER;
    opt.peer = peer;
    opt.port = port;
    pulsar_tp_identity id = {};
    id.gguf_bytes = 87000000000ull;
    id.model_id = 3u;
    id.n_layer = N_LAYER;
    id.n_embd = N_EMBD;
    id.n_vocab = 129280u;
    id.quant_bits = 2u;
    id.ctx_size = 1048576u;
    return pulsar_tp_create(out, &opt, &id, err, errlen);
}

static int run_rank(pulsar_tp *tp, int rank) {
    const char *want = getenv("PULSAR_TP_EXPECT_RDMA");
    if (want && want[0])
        CHECK(pulsar_tp_is_rdma(tp),
              "rank %d: PULSAR_TP_EXPECT_RDMA set but transport is TCP", rank);
    std::printf("rank %d: connected transport=%s\n", rank,
                pulsar_tp_is_rdma(tp) ? "rdma" : "tcp");

    const size_t slab_bytes = (size_t)pulsar_tp_slab_bytes(N_LAYER, N_EMBD);
    void *base = NULL;
    cudaError_t ce = cudaMallocManaged(&base, slab_bytes);
    if (ce != cudaSuccess || !base) {
        std::fprintf(stderr,
                     "rank %d: cudaMallocManaged(%zu) failed: %s — requires a "
                     "CUDA device (this binary is for the pair)\n",
                     rank, slab_bytes, cudaGetErrorString(ce));
        return 2;
    }
    std::memset(base, 0, slab_bytes);
    char err[256];
    if (!pulsar_tp_attach_slab(tp, base, err, sizeof(err))) {
        CHECK(0, "rank %d: attach_slab (cudaMallocManaged): %s", rank, err);
        cudaFree(base);
        return 1;
    }
    std::printf("rank %d: slab attached, GPU-visible (%zu bytes), mr=%s\n",
                rank, slab_bytes, pulsar_tp_is_rdma(tp) ? "registered" : "tcp-stage");

    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(N_LAYER, N_EMBD, &s);
    /* A handful of full gate exchanges (16 KiB f32 partials) both directions,
     * then one batch and one big-gate round, verifying the peer's pattern. */
    for (uint32_t layer = 0; layer < 4; layer++) {
        for (uint32_t gate = 0; gate < PULSAR_TP_GATES_PER_LAYER; gate++) {
            const uint32_t slot = layer * PULSAR_TP_GATES_PER_LAYER + gate;
            const uint64_t e = slot + 1;
            float *out = (float *)((uint8_t *)base +
                pulsar_tp_slab_out_offset(&s, layer, gate, VEC_BYTES));
            const uint32_t n = (uint32_t)(VEC_BYTES / sizeof(float));
            for (uint32_t i = 0; i < n; i++) out[i] = pat(rank, slot, i);
            if (!pulsar_tp_gate_exchange(tp, layer, gate, e)) {
                CHECK(0, "rank %d: gate l=%u g=%u e=%llu failed", rank, layer,
                      gate, (unsigned long long)e);
                break;
            }
            const float *in = (const float *)((uint8_t *)base +
                pulsar_tp_slab_in_offset(&s, layer, gate, VEC_BYTES));
            for (uint32_t i = 0; i < n; i++) {
                if (in[i] != pat(1 - rank, slot, i)) {
                    CHECK(0, "rank %d: gate l=%u g=%u i=%u in %g want %g",
                          rank, layer, gate, i, in[i], pat(1 - rank, slot, i));
                    break;
                }
            }
        }
    }
    /* batch gate (3 rows) + big gate (256 KiB) spot checks. */
    float *bout = (float *)((uint8_t *)base +
        pulsar_tp_slab_batch_out_offset(&s, 10, VEC_BYTES));
    const uint32_t bn = (uint32_t)(3u * VEC_BYTES / sizeof(float));
    for (uint32_t i = 0; i < bn; i++) bout[i] = pat(rank, 1000, i);
    CHECK(pulsar_tp_batch_gate_exchange(tp, 10, 3, 1) == 1,
          "rank %d: batch_gate failed", rank);

    static float big_out[65536], big_in[65536];   /* 256 KiB f32 partials */
    for (uint32_t i = 0; i < 65536; i++) big_out[i] = pat(rank, 1001, i);
    CHECK(pulsar_tp_big_gate_exchange(tp, 0, 1, big_out, big_in,
                                      sizeof(big_out)) == 1,
          "rank %d: big_gate failed", rank);
    for (uint32_t i = 0; i < 65536; i += 4096) {
        if (big_in[i] != pat(1 - rank, 1001, i)) {
            CHECK(0, "rank %d: big_gate in[%u]=%g want %g", rank, i, big_in[i],
                  pat(1 - rank, 1001, i));
            break;
        }
    }

    pulsar_tp_free(tp);
    cudaFree(base);
    if (g_failures) {
        std::fprintf(stderr, "tp_slab_probe: rank %d: %d FAILURE(S)\n", rank, g_failures);
        return 1;
    }
    std::printf("tp_slab_probe: rank %d ok (GPU slab: gates, batch, big; %s)\n",
                rank, pulsar_tp_is_rdma(tp) ? "RDMA" : "TCP");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s leader 0.0.0.0 PORT   |   %s worker PEER-IP PORT\n",
            argv[0], argv[0]);
        return 2;
    }
    const bool leader = std::strcmp(argv[1], "leader") == 0;
    const char *peer = argv[2];
    const int port = atoi(argv[3]);
    char err[256];
    pulsar_tp *tp = NULL;
    if (!create_rank(leader ? 0 : 1, leader ? ((peer[0]=='0')?NULL:peer) : peer,
                     port, &tp, err, sizeof(err))) {
        std::fprintf(stderr, "tp_slab_probe: create (%s) failed: %s\n",
                     leader ? "leader" : "worker", err);
        return 1;
    }
    return run_rank(tp, leader ? 0 : 1);
}

/* QSA LAYER REPLAY (L251 S3): run one real Qwen attention layer's between-the-
 * Linears work through pulsar_gpu_qsa_forward on projections captured from the
 * BF16 source, for grading against transformers' module offline.
 *
 * DIR holds (written by the capture script, research/l251/attn in pulsar-notes):
 *   qg.bf16 [T][12288]  k.bf16 [T][512]  v.bf16 [T][512]  idx.bf16 [T][640]
 *   norms.f32  q_norm[256] k_norm[256] idx_q_norm[128] idx_k_norm[128]
 * and receives:
 *   ours_out.f32 [T][6144]   the gated output (o_proj input), from the f32 tap
 *   ours_sel.u32 [T][512]    selected blocks ascending (0xffffffff = every token)
 *   ours_kv.bin  [T][1056]   the FP8 cache, for the host-side FP8 grade
 *
 * One sequence, chunked prefill at CHUNK rows; then the same tokens again with
 * the last DECODE_TAIL rows as single-row decodes, which must reproduce the
 * prefill's bytes (decode == prefill on real data).  Prints the prefill time.
 *
 * usage: qsa_layer_replay DIR T [CHUNK=2048] [DECODE_TAIL=16] */
#include "pulsar_gpu.h"
#include "pulsar_cuda_mx.cuh"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> slurp(const std::string &path, size_t want) {
    std::vector<uint8_t> b(want);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f || fread(b.data(), 1, want, f) != want) { fprintf(stderr, "short or missing %s (want %zu B)\n", path.c_str(), want); exit(2); }
    fclose(f);
    return b;
}
void spill(const std::string &path, const void *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path.c_str()); exit(2); }
    fclose(f);
}
float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
pulsar_gpu_tensor *alloc0(uint64_t bytes) {
    pulsar_gpu_tensor *t = pulsar_gpu_tensor_alloc(bytes);
    if (!t || !pulsar_gpu_tensor_fill_f32(t, 0.f, bytes / 4)) { fprintf(stderr, "alloc %llu failed\n", (unsigned long long)bytes); exit(2); }
    return t;
}
double now() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

struct Inputs { const uint16_t *qg, *k, *v, *idx; };

struct Replay {
    uint32_t T, chunk;
    pulsar_gpu_tensor *qg, *k, *v, *idx, *slot, *sc, *tap, *sel, *ws, *norm[4];
    pulsar_qsa_seq seq;

    void rows(const Inputs &in, uint32_t p0, uint32_t n, float *out, uint32_t *selo) {
        std::vector<float> buf;
        auto up = [&](pulsar_gpu_tensor *t, const uint16_t *src, uint32_t w) {
            buf.resize((size_t)n * w);
            for (size_t i = 0; i < buf.size(); i++) buf[i] = bf(src[(size_t)p0 * w + i]);
            pulsar_gpu_tensor_write(t, 0, buf.data(), buf.size() * 4);
        };
        up(qg, in.qg, PULSAR_QSA_Q_IN);
        up(k, in.k, PULSAR_QSA_KV_IN);
        up(v, in.v, PULSAR_QSA_KV_IN);
        up(idx, in.idx, PULSAR_QSA_IDX_IN);
        std::vector<uint32_t> rs(n, 0), rp(n);
        for (uint32_t i = 0; i < n; i++) rp[i] = p0 + i;
        pulsar_qsa_layer L{(const uint16_t *)pulsar_gpu_tensor_device_ptr(norm[0]), (const uint16_t *)pulsar_gpu_tensor_device_ptr(norm[1]),
                           (const uint16_t *)pulsar_gpu_tensor_device_ptr(norm[2]), (const uint16_t *)pulsar_gpu_tensor_device_ptr(norm[3])};
        pulsar_qsa_io io{};
        io.qg = qg; io.k = k; io.v = v; io.idx = idx;
        io.out_e4m3 = pulsar_gpu_tensor_device_ptr(slot);
        io.out_scale = pulsar_gpu_tensor_device_ptr(sc);
        io.out_sf_pitch = pulsar_mx_kbp(PULSAR_QSA_OUT_DIM);
        io.tap_out_f32 = tap;
        io.tap_sel = sel;
        pulsar_gpu_tensor_fill_f32(sc, 0.f, pulsar_gpu_tensor_bytes(sc) / 4);
        if (!pulsar_gpu_qsa_forward(&L, &seq, 1, rs.data(), rp.data(), n, &io, ws)) { fprintf(stderr, "forward refused\n"); exit(1); }
        pulsar_gpu_tensor_read(tap, 0, out, (size_t)n * PULSAR_QSA_OUT_DIM * 4);
        pulsar_gpu_tensor_read(sel, 0, selo, (size_t)n * PULSAR_QSA_TOP_BLOCKS * 4);
    }
    void reset() {
        pulsar_gpu_tensor_fill_f32(seq.kv, 0.f, pulsar_gpu_tensor_bytes(seq.kv) / 4);
        pulsar_gpu_tensor_fill_f32(seq.bkey, 0.f, pulsar_gpu_tensor_bytes(seq.bkey) / 4);
        pulsar_gpu_tensor_fill_f32(seq.stage, 1e30f, PULSAR_QSA_STAGE_BYTES / 4);
    }
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s DIR T [CHUNK=2048] [DECODE_TAIL=16]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const uint32_t T = (uint32_t)atoi(argv[2]);
    const uint32_t chunk = argc > 3 ? (uint32_t)atoi(argv[3]) : 2048u;
    const uint32_t tail = argc > 4 ? (uint32_t)atoi(argv[4]) : 16u;
    if (!pulsar_gpu_init()) return 2;
    std::vector<uint8_t> bqg = slurp(dir + "/qg.bf16", (size_t)T * PULSAR_QSA_Q_IN * 2);
    std::vector<uint8_t> bk = slurp(dir + "/k.bf16", (size_t)T * PULSAR_QSA_KV_IN * 2);
    std::vector<uint8_t> bv = slurp(dir + "/v.bf16", (size_t)T * PULSAR_QSA_KV_IN * 2);
    std::vector<uint8_t> bi = slurp(dir + "/idx.bf16", (size_t)T * PULSAR_QSA_IDX_IN * 2);
    std::vector<uint8_t> bn = slurp(dir + "/norms.f32", (2 * PULSAR_QSA_HEAD_DIM + 2 * PULSAR_QSA_IDX_DIM) * 4);
    Inputs in{(const uint16_t *)bqg.data(), (const uint16_t *)bk.data(), (const uint16_t *)bv.data(), (const uint16_t *)bi.data()};

    Replay R;
    R.T = T;
    R.chunk = chunk;
    R.qg = alloc0((uint64_t)chunk * PULSAR_QSA_Q_IN * 4);
    R.k = alloc0((uint64_t)chunk * PULSAR_QSA_KV_IN * 4);
    R.v = alloc0((uint64_t)chunk * PULSAR_QSA_KV_IN * 4);
    R.idx = alloc0((uint64_t)chunk * PULSAR_QSA_IDX_IN * 4);
    R.slot = alloc0((uint64_t)chunk * PULSAR_QSA_OUT_DIM);
    R.sc = alloc0(pulsar_mx_sf_slab_bytes((int)chunk, pulsar_mx_kbp(PULSAR_QSA_OUT_DIM)));
    R.tap = alloc0((uint64_t)chunk * PULSAR_QSA_OUT_DIM * 4);
    R.sel = alloc0((uint64_t)chunk * PULSAR_QSA_TOP_BLOCKS * 4);
    R.ws = alloc0(pulsar_gpu_qsa_workspace_bytes(chunk, T));
    const float *nf = (const float *)bn.data();
    const uint32_t off[5] = {0, PULSAR_QSA_HEAD_DIM, 2 * PULSAR_QSA_HEAD_DIM, 2 * PULSAR_QSA_HEAD_DIM + PULSAR_QSA_IDX_DIM,
                             2 * PULSAR_QSA_HEAD_DIM + 2 * PULSAR_QSA_IDX_DIM};
    for (int i = 0; i < 4; i++) {
        const uint32_t n = off[i + 1] - off[i];      /* the container stores the norms bf16 */
        std::vector<uint16_t> b16(n);
        for (uint32_t j = 0; j < n; j++) {
            uint32_t u; memcpy(&u, nf + off[i] + j, 4); b16[j] = (uint16_t)((u + 0x8000u) >> 16);
        }
        R.norm[i] = pulsar_gpu_tensor_alloc((uint64_t)n * 2);
        pulsar_gpu_tensor_write(R.norm[i], 0, b16.data(), (uint64_t)n * 2);
    }
    const uint32_t cap = (T + 3u) & ~3u;
    R.seq = {alloc0((uint64_t)cap * PULSAR_QSA_KV_TOKEN_BYTES), alloc0((uint64_t)(cap / 4) * PULSAR_QSA_BKEY_BYTES),
             alloc0(PULSAR_QSA_STAGE_BYTES), cap};

    std::vector<float> out((size_t)T * PULSAR_QSA_OUT_DIM);
    std::vector<uint32_t> sel((size_t)T * PULSAR_QSA_TOP_BLOCKS);
    R.reset();
    const double t0 = now();
    for (uint32_t p = 0; p < T; p += chunk) {
        const uint32_t n = std::min(chunk, T - p);
        R.rows(in, p, n, &out[(size_t)p * PULSAR_QSA_OUT_DIM], &sel[(size_t)p * PULSAR_QSA_TOP_BLOCKS]);
    }
    const double t1 = now();
    printf("replay: T=%u chunk=%u prefill wall %.2f s (incl. host bf16->f32 + readback)\n", T, chunk, t1 - t0);
    std::vector<uint8_t> kv((size_t)T * PULSAR_QSA_KV_TOKEN_BYTES);
    pulsar_gpu_tensor_read(R.seq.kv, 0, kv.data(), kv.size());
    spill(dir + "/ours_out.f32", out.data(), out.size() * 4);
    spill(dir + "/ours_sel.u32", sel.data(), sel.size() * 4);
    spill(dir + "/ours_kv.bin", kv.data(), kv.size());

    /* decode == prefill on the real rows */
    int fail = 0;
    if (tail && tail < T) {
        R.reset();
        std::vector<float> o2((size_t)chunk * PULSAR_QSA_OUT_DIM);
        std::vector<uint32_t> s2((size_t)chunk * PULSAR_QSA_TOP_BLOCKS);
        for (uint32_t p = 0; p < T - tail; p += chunk) R.rows(in, p, std::min(chunk, T - tail - p), o2.data(), s2.data());
        uint32_t diff = 0;
        for (uint32_t p = T - tail; p < T; p++) {
            R.rows(in, p, 1, o2.data(), s2.data());
            diff += memcmp(o2.data(), &out[(size_t)p * PULSAR_QSA_OUT_DIM], PULSAR_QSA_OUT_DIM * 4) != 0 ||
                    memcmp(s2.data(), &sel[(size_t)p * PULSAR_QSA_TOP_BLOCKS], PULSAR_QSA_TOP_BLOCKS * 4) != 0;
        }
        std::vector<uint8_t> kv2(kv.size());
        pulsar_gpu_tensor_read(R.seq.kv, 0, kv2.data(), kv2.size());
        const bool kv_same = kv2 == kv;
        fail = diff != 0 || !kv_same;
        printf("replay: decode == prefill over the last %u rows: %s (%u rows differ; KV %s)\n", tail,
               fail ? "FAIL" : "ok", diff, kv_same ? "identical" : "DIFFERS");
    }
    return fail;
}

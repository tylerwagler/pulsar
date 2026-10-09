/* tessera_kernel_gate -- pulsar's Tessera launcher against Tessera's own build, byte for byte (L255).
 *
 *   tests/tessera_kernel_gate FIXTURE [--bench N]
 *
 * FIXTURE comes from tools/tessera/kernel_fixture.py: Tessera's own torch extension (built from Tessera's
 * source by Tessera's loader) run on Qwen3.8-Flash-Next layer-12 weights, with every kernel input and the
 * output bytes recorded.  This gate feeds the same inputs to pulsar_tessera_dense_launch /
 * pulsar_tessera_moe_launch (src/cuda/mmq/pulsar_tessera.cu: the vendored device code under pulsar's host
 * side) and passes only if every output byte matches and pulsar's split-K choice equals Tessera's.
 * --bench N then times each passing case N times (CUDA events, median, after 5 warm-up launches): a HOT
 * microbenchmark -- consecutive launches leave the weights wherever the previous one did -- whose MoE scope is
 * the whole call (routing prep, both launches, token sum), the scope of Tessera's FusedRoutedWindowMoE.__call__.
 *
 * Made with Tessera by Robert Tand - https://github.com/RobTand/tessera */
#include "pulsar_tessera.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Rec {
    int dtype = 0;
    std::vector<int64_t> dims;
    std::vector<uint8_t> data;
};

bool read_fixture(const char *path, std::map<std::string, Rec> &out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char magic[8];
    uint32_t n = 0;
    bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "TSFX1\0\0\0", 8) == 0 && fread(&n, 4, 1, f) == 1;
    for (uint32_t i = 0; ok && i < n; ++i) {
        uint16_t nl = 0;
        uint8_t dt = 0, nd = 0;
        ok = fread(&nl, 2, 1, f) == 1;
        std::string name(nl, '\0');
        ok = ok && fread(&name[0], 1, nl, f) == nl && fread(&dt, 1, 1, f) == 1 && fread(&nd, 1, 1, f) == 1;
        Rec r;
        r.dtype = dt;
        r.dims.resize(nd);
        uint64_t nb = 0;
        ok = ok && (nd == 0 || fread(r.dims.data(), 8, nd, f) == nd) && fread(&nb, 8, 1, f) == 1;
        if (!ok) break;
        fseek(f, (long)((16 - ftell(f) % 16) % 16), SEEK_CUR);
        r.data.resize(nb);
        ok = fread(r.data.data(), 1, nb, f) == nb;
        fseek(f, (long)((16 - ftell(f) % 16) % 16), SEEK_CUR);
        out[name] = std::move(r);
    }
    fclose(f);
    if (!ok) fprintf(stderr, "%s: truncated or not a TSFX1 fixture\n", path);
    return ok;
}

bool cuda_ok(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Fixture {
    std::map<std::string, Rec> recs;
    std::vector<void *> device;

    ~Fixture() { for (void *p : device) cudaFree(p); }

    const Rec *get(const std::string &name) const {
        auto it = recs.find(name);
        if (it == recs.end()) { fprintf(stderr, "fixture has no record %s\n", name.c_str()); return nullptr; }
        return &it->second;
    }
    void *upload(const std::string &name) {
        const Rec *r = get(name);
        void *d = nullptr;
        if (!r || !cuda_ok(cudaMalloc(&d, r->data.size() ? r->data.size() : 16), "malloc")) return nullptr;
        device.push_back(d);
        if (!cuda_ok(cudaMemcpy(d, r->data.data(), r->data.size(), cudaMemcpyHostToDevice), "upload")) return nullptr;
        return d;
    }
    const int32_t *ints(const std::string &name) const {
        const Rec *r = get(name);
        return r ? (const int32_t *)r->data.data() : nullptr;
    }
    /* one projection's planes under prefix p, geometry from the caller */
    bool proj(const std::string &p, int E, int K, int N, int tile_words, int slot_words, pulsar_tessera_proj *o) {
        const Rec *w = get(p + "words");
        if (!w) return false;
        o->E = E;
        o->K = K;
        o->N = N;
        o->words = (const int32_t *)upload(p + "words");
        o->words_stride = (long)w->dims[1];
        o->table = (const uint16_t *)upload(p + "table");
        o->init = (const int32_t *)upload(p + "init");
        o->has_init = (const int32_t *)upload(p + "has_init");
        o->wscale = (const float *)upload(p + "wscale");
        o->runs = (const int32_t *)upload(p + "runs");
        o->bdesc = (const int32_t *)upload(p + "bdesc");
        o->tile_words = tile_words;
        o->slot_words = slot_words;
        return o->words && o->table && o->init && o->has_init && o->wscale && o->runs && o->bdesc;
    }
};

/* compare bf16 outputs; report the mismatch count and the largest difference */
bool same_bytes(const char *label, const std::vector<uint16_t> &got, const Rec &want) {
    if (want.data.size() != got.size() * 2) {
        printf("  %-28s FAIL size %zu vs %zu\n", label, got.size() * 2, want.data.size());
        return false;
    }
    const uint16_t *w = (const uint16_t *)want.data.data();
    size_t bad = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == w[i]) continue;
        ++bad;
        uint32_t a = (uint32_t)got[i] << 16, b = (uint32_t)w[i] << 16;
        float fa, fb;
        memcpy(&fa, &a, 4);
        memcpy(&fb, &b, 4);
        const float d = fa > fb ? fa - fb : fb - fa;
        if (d > worst) worst = d;
    }
    printf("  %-28s %s  (%zu values, %zu differ, max |diff| %.3g)\n", label, bad ? "FAIL" : "PASS", got.size(), bad,
           (double)worst);
    return bad == 0;
}

/* median microseconds of n launches of fn on stream, after 5 warm-up launches */
template <typename F> double median_us(F fn, int n, cudaStream_t stream) {
    for (int i = 0; i < 5; ++i) fn();
    std::vector<cudaEvent_t> ev(2 * (size_t)n);
    for (auto &e : ev) cudaEventCreate(&e);
    for (int i = 0; i < n; ++i) {
        cudaEventRecord(ev[2 * i], stream);
        fn();
        cudaEventRecord(ev[2 * i + 1], stream);
    }
    cudaStreamSynchronize(stream);
    std::vector<float> ms((size_t)n);
    for (int i = 0; i < n; ++i) cudaEventElapsedTime(&ms[i], ev[2 * i], ev[2 * i + 1]);
    for (auto &e : ev) cudaEventDestroy(e);
    std::sort(ms.begin(), ms.end());
    return 1000.0 * ms[(size_t)n / 2];
}

} // namespace

int main(int argc, char **argv) {
    const int bench = argc == 4 && strcmp(argv[2], "--bench") == 0 ? atoi(argv[3]) : 0;
    if (argc != 2 && bench <= 0) {
        fprintf(stderr, "usage: %s FIXTURE [--bench N] (tools/tessera/kernel_fixture.py)\n", argv[0]);
        return 2;
    }
    Fixture fx;
    if (!read_fixture(argv[1], fx.recs)) return 2;
    cudaStream_t stream;
    if (!cuda_ok(cudaStreamCreate(&stream), "stream")) return 2;
    int fails = 0, cases = 0;

    printf("tessera_kernel_gate: %s (%zu records)\n", argv[1], fx.recs.size());
    for (const char *role : {"qkv", "oproj"}) {
        const std::string p = std::string("dense.") + role + ".";
        if (!fx.recs.count(p + "geom")) {   // a --moe-only fixture: no dense roles to run
            printf("  (no dense %s role in this fixture)\n", role);
            continue;
        }
        const int32_t *g = fx.ints(p + "geom");   // rows, cols, tile_words, slot_words
        pulsar_tessera_proj w{};
        if (!g || !fx.proj(p, 1, g[1], g[0], g[2], g[3], &w)) return 2;
        for (const auto &kv : fx.recs) {
            const std::string pre = std::string("case.dense.") + role + ".M";
            if (kv.first.compare(0, pre.size(), pre) != 0 || kv.first.size() < 2 ||
                kv.first.compare(kv.first.size() - 2, 2, ".x") != 0)
                continue;
            const std::string c = kv.first.substr(0, kv.first.size() - 1);   // "case.dense.<role>.M<m>."
            const Rec &x = kv.second;
            const int M = (int)x.dims[0];
            const int32_t *ks = fx.ints(c + "ksplit");                       // Tessera's S, its SM count
            const int mine = pulsar_tessera_dense_k_split(M, w.N, w.K, ks[1], w.tile_words);
            char label[96];
            snprintf(label, sizeof label, "dense %s M=%d S=%d", role, M, ks[0]);
            ++cases;
            if (mine != ks[0]) {
                printf("  %-28s FAIL k_split: pulsar %d, Tessera %d\n", label, mine, ks[0]);
                ++fails;
                continue;
            }
            const uint16_t *dx = (const uint16_t *)fx.upload(c + "x");
            uint16_t *dy = nullptr;
            const size_t wsb = pulsar_tessera_dense_workspace_bytes(&w, M);
            void *ws = nullptr;
            if (!dx || !cuda_ok(cudaMalloc(&dy, (size_t)M * w.N * 2), "malloc y") ||
                !cuda_ok(cudaMalloc(&ws, wsb), "malloc ws"))
                return 2;
            fx.device.push_back(dy);
            fx.device.push_back(ws);
            if (pulsar_tessera_dense_launch(&w, dx, M, dy, w.N, ws, wsb, stream) != 0 ||
                !cuda_ok(cudaStreamSynchronize(stream), "dense launch")) {
                printf("  %-28s FAIL launch\n", label);
                ++fails;
                continue;
            }
            std::vector<uint16_t> got((size_t)M * w.N);
            if (!cuda_ok(cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost), "download")) return 2;
            const bool pass = same_bytes(label, got, *fx.get(c + "y"));
            fails += pass ? 0 : 1;
            if (pass && bench)
                printf("  %-28s bench %.1f us (median of %d, hot)\n", label,
                       median_us([&] { pulsar_tessera_dense_launch(&w, dx, M, dy, w.N, ws, wsb, stream); }, bench,
                                 stream), bench);
        }
    }

    const int32_t *mg = fx.ints("moe.geom");   // E, I, H, tile/slot gate-up, tile/slot down
    if (!mg) return 2;
    pulsar_tessera_proj gate{}, up{}, down{};
    if (!fx.proj("moe.gate.", mg[0], mg[2], mg[1], mg[3], mg[4], &gate) ||
        !fx.proj("moe.up.", mg[0], mg[2], mg[1], mg[3], mg[4], &up) ||
        !fx.proj("moe.down.", mg[0], mg[1], mg[2], mg[5], mg[6], &down))
        return 2;
    for (const auto &kv : fx.recs) {
        if (kv.first.compare(0, 10, "case.moe.T") != 0 || kv.first.compare(kv.first.size() - 2, 2, ".x") != 0) continue;
        const std::string c = kv.first.substr(0, kv.first.size() - 1);
        const int T = (int)kv.second.dims[0];
        const int top_k = (int)fx.get(c + "ids")->dims[1];
        char label[96];
        snprintf(label, sizeof label, "moe E=%d T=%d top_k=%d", mg[0], T, top_k);
        ++cases;
        const uint16_t *dx = (const uint16_t *)fx.upload(c + "x");
        const int32_t *ids = (const int32_t *)fx.upload(c + "ids");
        const float *wts = (const float *)fx.upload(c + "w");
        const size_t wsb = pulsar_tessera_moe_workspace_bytes(&gate, &down, T, top_k);
        uint16_t *dy = nullptr;
        void *ws = nullptr;
        if (!dx || !ids || !wts || !cuda_ok(cudaMalloc(&dy, (size_t)T * down.N * 2), "malloc y") ||
            !cuda_ok(cudaMalloc(&ws, wsb), "malloc ws"))
            return 2;
        fx.device.push_back(dy);
        fx.device.push_back(ws);
        if (pulsar_tessera_moe_launch(&gate, &up, &down, dx, T, top_k, ids, wts, dy, ws, wsb, stream) != 0 ||
            !cuda_ok(cudaStreamSynchronize(stream), "moe launch")) {
            printf("  %-28s FAIL launch\n", label);
            ++fails;
            continue;
        }
        std::vector<uint16_t> got((size_t)T * down.N);
        if (!cuda_ok(cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost), "download")) return 2;
        const bool pass = same_bytes(label, got, *fx.get(c + "y"));
        fails += pass ? 0 : 1;
        if (pass && bench)
            printf("  %-28s bench %.1f us (median of %d, hot, whole call)\n", label,
                   median_us([&] {
                       pulsar_tessera_moe_launch(&gate, &up, &down, dx, T, top_k, ids, wts, dy, ws, wsb, stream);
                   }, bench, stream), bench);
    }
    cudaStreamDestroy(stream);
    printf("tessera_kernel_gate: %d of %d cases byte-identical to Tessera's build -- %s\n", cases - fails, cases,
           fails ? "FAIL" : "PASS");
    return fails || !cases ? 1 : 0;
}

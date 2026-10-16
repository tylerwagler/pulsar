/* LILICORR HEAD GATE (L298): the candidate-lattice head + greedy walk (src/cuda/pulsar_cuda_lilicorr.h) against vLLM's
 * own draft rounds.
 *
 * Inputs: the drafter checkpoint (model.safetensors: the head's `lilicorr.*` tensors -- the NVFP4 ones packed into the
 * engine's nvfp4 slices, pulsar_nvfp4_layout, and run through the rows front door; the BF16 ones as stored) and the
 * head fixtures tools/lilicorr/reference.py `fixture` writes from vllm_dump.py's goldens: per prompt, every round's
 * head inputs exactly as vLLM fed them (candidate ids and log-probs, pass rows, anchor), the candidates' embedding rows
 * (a compact table; cand_local indexes it), vLLM's scores and chain, and the PyTorch reference's.
 *
 *   1. CHAIN: every round's chain == vLLM's, or -- the L298 grading rule -- they part at a slot where vLLM's own scores
 *      put the two picks within the head's bf16 noise floor (kFloor logits: vLLM's scores sit 0.43-0.68 logits from an
 *      all-f32 head on these rounds); the round is then reported, never silently passed.  Same check against the
 *      PyTorch reference's chain.
 *   2. SCORES (information): max |kernel - vLLM| and |kernel - reference| over every round, in logits.
 *   3. BANKS: the rounds run 16 banks a launch and each alone; scores and chains must be bit-identical (every Linear is
 *      M-independent, the lattice GEMMs run one algorithm whatever their batch count).
 *   4. refusals: 0 banks, 17 banks, a short workspace, a Linear of the wrong shape.
 *   5. timing (information): one launch at 1 bank and at 16.
 *
 * usage: ./tests/lilicorr_head_gate <drafter dir> <fixture dir> [prompt ...]
 */
#include "cuda/pulsar_cuda_lilicorr.h"
#include "cuda/pulsar_cuda_nvfp4.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "LILICORR-HEAD FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); \
                                       g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "LILICORR-HEAD FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const float kFloor = 0.5f;   /* logits: the head's bf16 noise floor (vLLM vs an all-f32 head, measured) */

/* ---- a minimal safetensors reader: the header's flat JSON, one tensor at a time by exact name ---- */
struct st_tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr;
    size_t bytes = 0;
};
struct st_file {
    const uint8_t *map = nullptr;
    size_t size = 0;
    std::string header;
    const uint8_t *body = nullptr;
    bool open(const std::string &path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat sb;
        fstat(fd, &sb);
        size = (size_t)sb.st_size;
        map = (const uint8_t *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) return false;
        uint64_t n;
        memcpy(&n, map, 8);
        header.assign((const char *)map + 8, n);
        body = map + 8 + n;
        return true;
    }
    bool has(const std::string &name) const { return header.find("\"" + name + "\":") != std::string::npos; }
    st_tensor get(const std::string &name) const {
        st_tensor t;
        const size_t at = header.find("\"" + name + "\":");
        if (at == std::string::npos) { fprintf(stderr, "LILICORR-HEAD FAIL: no tensor %s\n", name.c_str()); exit(1); }
        const size_t end = header.find('}', at);
        const std::string obj = header.substr(at, end - at);
        const size_t d = obj.find("\"dtype\":\"") + 9;
        t.dtype = obj.substr(d, obj.find('"', d) - d);
        size_t s = obj.find("\"shape\":[") + 9;
        const size_t se = obj.find(']', s);
        while (s < se) {
            char *e;
            t.shape.push_back(strtoll(obj.c_str() + s, &e, 10));
            s = (size_t)(e - obj.c_str());
            while (s < se && (obj[s] == ',' || obj[s] == ' ')) s++;
        }
        const size_t o = obj.find("\"data_offsets\":[") + 16;
        char *e;
        const uint64_t a = strtoull(obj.c_str() + o, &e, 10);
        const uint64_t b = strtoull(e + 1, nullptr, 10);
        t.data = body + a;
        t.bytes = b - a;
        return t;
    }
};

static void *upload(const void *p, size_t n) {
    void *d = nullptr;
    CUDA_OK(cudaMalloc(&d, n + 16));
    CUDA_OK(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
    return d;
}
static const uint16_t *bf16_dev(const st_file &f, const std::string &name) {
    const st_tensor t = f.get(name);
    if (t.dtype != "BF16") { fprintf(stderr, "LILICORR-HEAD FAIL: %s is %s, not BF16\n", name.c_str(), t.dtype.c_str()); exit(1); }
    return (const uint16_t *)upload(t.data, t.bytes);
}

/* One head Linear from the checkpoint: NVFP4 (weight U8 + weight_scale E4M3 + weight_scale_2) -> an nvfp4 slice
 * (values verbatim, the scales 128x4-tiled, the global in the tail); BF16 -> as stored. */
static pulsar_lilicorr_linear head_linear(const st_file &f, const std::string &name) {
    pulsar_lilicorr_linear L;
    memset(&L, 0, sizeof(L));
    const st_tensor w = f.get(name + ".weight");
    L.b = bf16_dev(f, name + ".bias");
    if (w.dtype == "U8") {
        const st_tensor sc = f.get(name + ".weight_scale"), g = f.get(name + ".weight_scale_2");
        const int out = (int)w.shape[0], in = (int)w.shape[1] * 2;
        uint64_t vb, sb, stride;
        if (!pulsar_nvfp4_layout(in, out, &vb, &sb, &stride)) { fprintf(stderr, "bad nvfp4 shape %s\n", name.c_str()); exit(1); }
        std::vector<uint8_t> blob(stride, 0);
        memcpy(blob.data(), w.data, vb);
        const int kb4 = in / 64;
        for (int r = 0; r < out; r++)
            for (int kb = 0; kb < in / 16; kb++)
                blob[vb + ((size_t)(r / 128) * kb4 + kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4] =
                    sc.data[(size_t)r * (in / 16) + kb];
        memcpy(&blob[vb + sb], g.data, 4);
        L.l.w = upload(blob.data(), blob.size());
        L.l.in = in;
        L.l.out = out;
        L.l.nvfp4 = true;
    } else {
        L.l.w = upload(w.data, w.bytes);
        L.l.out = (int)w.shape[0];
        L.l.in = (int)w.shape[1];
        L.l.bf16 = true;
    }
    return L;
}

static pulsar_lilicorr_head load_head(const st_file &f) {
    pulsar_lilicorr_head h;
    memset(&h, 0, sizeof(h));
    const std::string p = "lilicorr.";
    h.token_proj = head_linear(f, p + "token_proj");
    h.pass_hidden_proj = head_linear(f, p + "pass_hidden_proj");
    h.context_proj = head_linear(f, p + "context_proj");
    h.feature_norm_w = bf16_dev(f, p + "feature_norm.weight");
    h.feature_norm_b = bf16_dev(f, p + "feature_norm.bias");
    h.feature_mlp_up_w = bf16_dev(f, p + "feature_mlp.up_proj.weight");
    h.feature_mlp_up_b = bf16_dev(f, p + "feature_mlp.up_proj.bias");
    h.feature_mlp_down = head_linear(f, p + "feature_mlp.down_proj");
    h.slot_embedding = bf16_dev(f, p + "slot_embedding");
    h.rank_embedding = bf16_dev(f, p + "rank_embedding");
    h.relative_slot_bias = bf16_dev(f, p + "relative_slot_bias");
    h.same_slot_bias = bf16_dev(f, p + "same_slot_bias");
    for (int i = 0; i < (int)PULSAR_LILICORR_LAYERS; i++) {
        const std::string q = p + "layers." + std::to_string(i) + ".";
        pulsar_lilicorr_layer &L = h.layers[i];
        L.lat_attn_norm = bf16_dev(f, q + "attn_norm.weight");
        /* in_proj is stored as a bare parameter pair (in_proj_weight / in_proj_bias) */
        const st_tensor w = f.get(q + "attn.in_proj_weight");
        L.lat_attn_in.l.w = upload(w.data, w.bytes);
        L.lat_attn_in.l.out = (int)w.shape[0];
        L.lat_attn_in.l.in = (int)w.shape[1];
        L.lat_attn_in.l.bf16 = true;
        L.lat_attn_in.b = bf16_dev(f, q + "attn.in_proj_bias");
        L.lat_attn_out = head_linear(f, q + "attn.out_proj");
        L.lat_mlp_norm = bf16_dev(f, q + "mlp_norm.weight");
        L.lat_mlp_up = head_linear(f, q + "mlp.up_proj");
        L.lat_mlp_down = head_linear(f, q + "mlp.down_proj");
    }
    h.output_norm = bf16_dev(f, p + "output_norm.weight");
    h.anchor_norm = bf16_dev(f, p + "anchor_norm.weight");
    h.factor_input_proj = head_linear(f, p + "factor_input_proj");
    h.out_head = head_linear(f, p + "out_head");
    h.in_head = head_linear(f, p + "in_head");
    h.anchor_out_head = head_linear(f, p + "anchor_out_head");
    h.vector_eps = 1e-4f;    /* config.json dflash_config.lilicorr_vector_eps */
    h.logit_scale = 8.0f;    /* lilicorr_logit_scale */
    return h;
}

template <class T> static std::vector<T> host(const st_file &f, const std::string &name) {
    const st_tensor t = f.get(name);
    std::vector<T> v(t.bytes / sizeof(T));
    memcpy(v.data(), t.data, t.bytes);
    return v;
}

struct run_out {
    std::vector<float> scores;
    std::vector<int32_t> ids, rows;
};

/* Rounds [r0, r0 + B) as B banks of one launch. */
static run_out run(const pulsar_lilicorr_head &h, const st_file &fx, int r0, int B, float *ms = nullptr) {
    static std::map<std::string, void *> dev;   /* the fixture uploaded once per prompt file (keyed by data pointer) */
    auto D = [&](const char *name) -> const uint8_t * {
        const st_tensor t = fx.get(name);
        const std::string key = std::to_string((uintptr_t)t.data);
        if (!dev.count(key)) dev[key] = upload(t.data, t.bytes);
        return (const uint8_t *)dev[key];
    };
    const int32_t *cand = (const int32_t *)D("cand_local") + (size_t)r0 * 56;
    const float *lp = (const float *)D("cand_logp") + (size_t)r0 * 56;
    const uint16_t *table = (const uint16_t *)D("emb_rows");
    const uint16_t *ph = (const uint16_t *)D("pass_hidden") + (size_t)r0 * 7 * 4096;
    const uint16_t *ah = (const uint16_t *)D("anchor_hidden") + (size_t)r0 * 4096;
    const uint8_t *av = D("anchor_valid") + r0;
    const uint64_t wsb = pulsar_lilicorr_head_ws_bytes(B);
    void *ws;
    float *sc;
    int32_t *ids, *rows;
    CUDA_OK(cudaMalloc(&ws, wsb));
    CUDA_OK(cudaMalloc((void **)&sc, (size_t)B * 448 * 4));
    CUDA_OK(cudaMalloc((void **)&ids, (size_t)B * 7 * 4));
    CUDA_OK(cudaMalloc((void **)&rows, (size_t)B * 7 * 4));
    const int rc = pulsar_lilicorr_head_launch(&h, B, cand, lp, table, ph, ah, av, sc, ids, rows, ws, wsb, 0);
    CHECK(rc == 0, "launch at %d banks returned %d", B, rc);
    CUDA_OK(cudaDeviceSynchronize());
    if (ms) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 20; i++) pulsar_lilicorr_head_launch(&h, B, cand, lp, table, ph, ah, av, sc, ids, rows, ws, wsb, 0);
        cudaEventRecord(b);
        CUDA_OK(cudaEventSynchronize(b));
        cudaEventElapsedTime(ms, a, b);
        *ms /= 20.0f;
    }
    run_out o;
    o.scores.resize((size_t)B * 448);
    o.ids.resize((size_t)B * 7);
    o.rows.resize((size_t)B * 7);
    CUDA_OK(cudaMemcpy(o.scores.data(), sc, o.scores.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(o.ids.data(), ids, o.ids.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(o.rows.data(), rows, o.rows.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(ws); cudaFree(sc); cudaFree(ids); cudaFree(rows);
    return o;
}

/* Grade one round's chain (candidate indices `mine`) against an expected one (`want`) with its own scores `wsc`
 * [7][8][8]: equal, or parted where `wsc` puts the two picks within kFloor.  Returns 0 equal, 1 excused, 2 a miss;
 * *gap = the expected scores' margin at the parting slot. */
static int grade(const int32_t *mine, const int32_t *want, const float *wsc, float *gap) {
    int prev = 0;
    for (int s = 0; s < 7; s++) {
        if (mine[s] != want[s]) {
            *gap = wsc[(s * 8 + prev) * 8 + want[s]] - wsc[(s * 8 + prev) * 8 + mine[s]];
            return *gap <= kFloor ? 1 : 2;
        }
        prev = want[s];
    }
    *gap = 0.0f;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <drafter dir> <fixture dir> [prompt ...]\n", argv[0]);
        return 2;
    }
    st_file model;
    if (!model.open(std::string(argv[1]) + "/model.safetensors")) { fprintf(stderr, "cannot open the drafter\n"); return 2; }
    const pulsar_lilicorr_head h = load_head(model);
    std::vector<std::string> prompts;
    for (int i = 3; i < argc; i++) prompts.push_back(argv[i]);
    if (prompts.empty()) prompts = {"code", "prose", "tool", "edit", "long"};

    int tot = 0, eq_v = 0, ex_v = 0, miss_v = 0, eq_r = 0, ex_r = 0, miss_r = 0, bank_bad = 0;
    float max_dv = 0, max_dr = 0, max_bank = 0;
    for (const std::string &name : prompts) {
        st_file fx;
        if (!fx.open(std::string(argv[2]) + "/" + name + ".safetensors")) {
            CHECK(false, "no fixture %s", name.c_str());
            continue;
        }
        const std::vector<int32_t> ids = host<int32_t>(fx, "cand_ids"), vchain = host<int32_t>(fx, "vllm_chain");
        const std::vector<int32_t> rrows = host<int32_t>(fx, "ref_rows"), local = host<int32_t>(fx, "cand_local");
        const std::vector<float> vsc = host<float>(fx, "vllm_scores"), rsc = host<float>(fx, "ref_scores");
        const int R = (int)vchain.size() / 7;
        float dv = 0, dr = 0;
        for (int r0 = 0; r0 < R; r0 += 16) {
            const int B = std::min(16, R - r0);
            const run_out batched = run(h, fx, r0, B);
            for (int b = 0; b < B; b++) {
                const int r = r0 + b;
                const run_out alone = run(h, fx, r, 1);
                const float *s = &batched.scores[(size_t)b * 448];
                float bd = 0;
                for (int i = 0; i < 448; i++) {
                    dv = std::max(dv, std::fabs(s[i] - vsc[(size_t)r * 448 + i]));
                    dr = std::max(dr, std::fabs(s[i] - rsc[(size_t)r * 448 + i]));
                    bd = std::max(bd, std::fabs(s[i] - alone.scores[i]));
                }
                max_bank = std::max(max_bank, bd);
                if (memcmp(&batched.rows[(size_t)b * 7], alone.rows.data(), 28) != 0) bank_bad++;
                /* the chain as candidate indices (chain_rows); chain_ids are the cand_ids entries at them -- the
                 * fixture's local table rows here */
                const int32_t *mine = &batched.rows[(size_t)b * 7];
                int32_t vrows[7];
                for (int s2 = 0; s2 < 7; s2++) {   /* vLLM's chain as candidate indices */
                    int j = 0;
                    while (j < 8 && ids[((size_t)r * 7 + s2) * 8 + j] != vchain[(size_t)r * 7 + s2]) j++;
                    CHECK(j < 8, "%s r%03d: vLLM's slot %d token is not among its candidates", name.c_str(), r, s2);
                    vrows[s2] = j < 8 ? j : 0;
                    CHECK(mine[s2] >= 0 && mine[s2] < 8 &&
                              batched.ids[(size_t)b * 7 + s2] == local[((size_t)r * 7 + s2) * 8 + mine[s2]],
                          "%s r%03d: chain_ids[%d] is not cand_ids at chain_rows", name.c_str(), r, s2);
                }
                float gap;
                const int gv = grade(mine, vrows, &vsc[(size_t)r * 448], &gap);
                tot++;
                if (gv == 0) eq_v++;
                else if (gv == 1) { ex_v++; printf("  %s r%03d: parts from vLLM within the floor (vLLM margin %.4f)\n", name.c_str(), r, gap); }
                else { miss_v++; CHECK(false, "%s r%03d: chain parts from vLLM's where vLLM's margin is %.4f > %.2f", name.c_str(), r, gap, kFloor); }
                const int gr = grade(mine, &rrows[(size_t)r * 7], &rsc[(size_t)r * 448], &gap);
                if (gr == 0) eq_r++;
                else if (gr == 1) ex_r++;
                else { miss_r++; CHECK(false, "%s r%03d: chain parts from the reference's where its margin is %.4f", name.c_str(), r, gap); }
            }
        }
        max_dv = std::max(max_dv, dv);
        max_dr = std::max(max_dr, dr);
        printf("  %s: %d rounds, max |scores - vLLM| %.4f, |scores - reference| %.4f\n", name.c_str(), R, dv, dr);
    }
    printf("CHAIN vs vLLM: %d rounds: %d exact, %d parted within the floor (%.2f logits), %d misses\n", tot, eq_v, ex_v,
           kFloor, miss_v);
    printf("CHAIN vs reference: %d exact, %d within the floor, %d misses\n", eq_r, ex_r, miss_r);
    printf("SCORES: max |kernel - vLLM| %.4f, max |kernel - reference| %.4f (logits)\n", max_dv, max_dr);
    printf("BANKS: 16-bank launches vs one bank: chains differ in %d rounds, scores max diff %.3g\n", bank_bad, max_bank);
    CHECK(bank_bad == 0 && max_bank == 0.0f, "a round's scores depend on the banks beside it");

    /* refusals */
    {
        float *sc; int32_t *a, *b; void *ws;
        CUDA_OK(cudaMalloc((void **)&sc, 448 * 17 * 4)); CUDA_OK(cudaMalloc((void **)&a, 7 * 17 * 4));
        CUDA_OK(cudaMalloc((void **)&b, 7 * 17 * 4)); CUDA_OK(cudaMalloc(&ws, pulsar_lilicorr_head_ws_bytes(16)));
        const int32_t *ci = (const int32_t *)a;
        CHECK(pulsar_lilicorr_head_ws_bytes(0) == 0 && pulsar_lilicorr_head_ws_bytes(17) == 0, "ws bytes past the bank range");
        CHECK(pulsar_lilicorr_head_launch(&h, 0, ci, sc, (const uint16_t *)ws, (const uint16_t *)ws, (const uint16_t *)ws,
                                          (const uint8_t *)ws, sc, a, b, ws, pulsar_lilicorr_head_ws_bytes(16), 0) == -1, "0 banks accepted");
        CHECK(pulsar_lilicorr_head_launch(&h, 17, ci, sc, (const uint16_t *)ws, (const uint16_t *)ws, (const uint16_t *)ws,
                                          (const uint8_t *)ws, sc, a, b, ws, pulsar_lilicorr_head_ws_bytes(16), 0) == -1, "17 banks accepted");
        CHECK(pulsar_lilicorr_head_launch(&h, 2, ci, sc, (const uint16_t *)ws, (const uint16_t *)ws, (const uint16_t *)ws,
                                          (const uint8_t *)ws, sc, a, b, ws, pulsar_lilicorr_head_ws_bytes(2) - 1, 0) == -1, "short ws accepted");
        pulsar_lilicorr_head bad = h;
        bad.in_head.l.in = 2048;
        CHECK(pulsar_lilicorr_head_launch(&bad, 1, ci, sc, (const uint16_t *)ws, (const uint16_t *)ws, (const uint16_t *)ws,
                                          (const uint8_t *)ws, sc, a, b, ws, pulsar_lilicorr_head_ws_bytes(16), 0) == -1, "wrong Linear accepted");
        cudaFree(sc); cudaFree(a); cudaFree(b); cudaFree(ws);
    }
    /* timing (information) */
    {
        st_file fx;
        if (fx.open(std::string(argv[2]) + "/" + prompts[0] + ".safetensors")) {
            float t1 = 0, t16 = 0;
            run(h, fx, 0, 1, &t1);
            const int R = (int)(fx.get("vllm_chain").bytes / 28);
            if (R >= 16) run(h, fx, 0, 16, &t16);
            printf("TIMING: one launch %.3f ms at 1 bank, %.3f ms at 16 banks\n", t1, t16);
        }
    }
    printf(g_fail ? "LILICORR-HEAD GATE: FAIL\n" : "LILICORR-HEAD GATE: PASS\n");
    return g_fail;
}

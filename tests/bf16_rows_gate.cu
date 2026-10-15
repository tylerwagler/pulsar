/* BF16 ROWS GATE (L295 F7): BF16 weights against raw bf16 rows -- the dense arm and the routed arm
 * (src/cuda/pulsar_cuda_bf16_rows.cuh), through the production objects and the rows front doors.
 *
 * Model-free.  Random bf16 weights and rows (normal, a few outlier channels), graded against double arithmetic on
 * the same bf16 values.  bf16 x bf16 is exact in f32, so the only error is f32 accumulation: every output is held to
 * 2^-18 of its own sum |w x| (the envelope's slack over the tensor cores' fragment sums and the f32 adds).
 *
 *   1. dense, decode rows (the tensor-core rows kernel): Inkling-width shapes 4096 -> 2048 / 2048 -> 4096, an output
 *      width off 16 (4096 -> 1000) and a wide one (1024 -> 201024); M = 16 graded, and M = 1, 5, 9 and 40 (16-row
 *      blocks) BIT-IDENTICAL to the same rows at M = 16 (the decode contract: a row's bytes do not depend on the
 *      batch width);
 *   2. dense, prompt chunks (cuBLASLt, one fixed-order kernel per shape): M = 40 graded, and the rows of a prompt cut
 *      at 1 / 7 / 33 bit-identical to the M = 40 call's (chunk invariance);
 *   3. the front door: pulsar_rows_linear_launch with a BF16 reference == the arm, bit for bit; refusals (in % 64,
 *      a BF16 reference carrying an EXL3 rate);
 *   4. routed (Inkling's expert shape 4096 -> 2048 -> 4096, 16 experts, top-6; Zipf-like routing with dropped picks):
 *      gate / up z graded; the bf16 mid within one bf16 ulp of the SwiGLU of the device's own z (x the route weight);
 *      the down z graded on the device's mid; the sum == the slots' down z added in slot order on the host, bit for
 *      bit; every token's output at T = 16 (decode) and T = 40 (a prompt chunk) BIT-IDENTICAL to the token alone
 *      (T = 1); a NaN in a row raises the non-finite flag; the front door (pulsar_rows_moe_routed_launch with BF16
 *      stacks) == the arm; refusals (BF16 with Tessera, BF16 with a riding shared expert);
 *   4'. the same routed checks for relu^2 up-only experts (no gate stack: Nemotron-H's activation, pulsar_ffn_act
 *      RELU2), and the activation / stack refusals both ways;
 *   4''. (L296 N4) the non-gated shared expert through pulsar_rows_moe_shared_launch (BF16 4096 -> 5376 -> 4096,
 *      RELU2): up graded, the bf16 relu^2 rows within one ulp, the down graded on them, out == the routed sum + the
 *      expert's f32 output bit for bit, rows alone == batched;
 *   5. timing (information): the 4096 -> 201024 BF16 head at 1 / 4 / 16 rows.
 *
 * usage: ./tests/bf16_rows_gate
 */
#include "cuda/pulsar_cuda_bf16_rows.cuh"
#include "cuda/pulsar_cuda_qwen.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "BF16-ROWS FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); \
                                       g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "BF16-ROWS FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint64_t g_rng = 0xB16B00B5F00DULL;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static double rndu(void) { return ((double)rnd() + 0.5) / 4294967296.0; }

static uint16_t bf16_bits(double v) {
    const __nv_bfloat16 b = __float2bfloat16((float)v);
    uint16_t u;
    memcpy(&u, &b, 2);
    return u;
}
static double bf16_val(uint16_t u) {
    const uint32_t f = (uint32_t)u << 16;
    float x;
    memcpy(&x, &f, 4);
    return x;
}
/* n bf16 values, scale s, a few outlier channels (period `cols`) */
static std::vector<uint16_t> rnd_bf16(size_t n, double s, int cols) {
    std::vector<double> ch(cols);
    for (int c = 0; c < cols; c++) ch[c] = rnd() % 64 == 0 ? 8.0 : 1.0;
    std::vector<uint16_t> v(n);
    /* a cheap near-normal (four uniforms): the stacks are hundreds of millions of values */
    for (size_t i = 0; i < n; i++) v[i] = bf16_bits((rndu() + rndu() + rndu() + rndu() - 2.0) * 1.7 * s * ch[i % cols]);
    return v;
}
template <class T> static T *upload(const std::vector<T> &v) {
    T *d = nullptr;
    CUDA_OK(cudaMalloc((void **)&d, v.size() * sizeof(T) + 16));
    CUDA_OK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> download(const T *d, size_t n) {
    std::vector<T> v(n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

/* y[r][o] in double and its envelope sum |w x| for rows [r0, r1) of x [*][in] against W [out][in] */
static void ref_linear(const uint16_t *W, const uint16_t *x, int r0, int r1, int in, int out,
                       std::vector<double> &y, std::vector<double> &env) {
    y.assign((size_t)(r1 - r0) * out, 0.0);
    env.assign(y.size(), 0.0);
    std::vector<double> xr(in);
    for (int r = r0; r < r1; r++) {
        for (int k = 0; k < in; k++) xr[k] = bf16_val(x[(size_t)r * in + k]);
        for (int o = 0; o < out; o++) {
            const uint16_t *w = W + (size_t)o * in;
            double s = 0, a = 0;
            for (int k = 0; k < in; k++) {
                const double p = bf16_val(w[k]) * xr[k];
                s += p;
                a += fabs(p);
            }
            y[(size_t)(r - r0) * out + o] = s;
            env[(size_t)(r - r0) * out + o] = a;
        }
    }
}
/* outputs past 2^-18 of their envelope; the worst err / envelope */
static size_t over_env(const float *got, const std::vector<double> &want, const std::vector<double> &env, size_t n,
                       double *worst) {
    size_t over = 0;
    *worst = 0;
    for (size_t i = 0; i < n; i++) {
        const double e = fabs((double)got[i] - want[i]), lim = ldexp(env[i], -18) + 1e-30;
        over += !(e <= lim);
        *worst = fmax(*worst, e / fmax(env[i], 1e-30));
    }
    return over;
}

static void section_dense(void) {
    struct shape { int in, out; const char *what; };
    const shape shapes[] = {{4096, 2048, "4096 -> 2048"}, {2048, 4096, "2048 -> 4096"}, {4096, 1000, "out off 16"},
                            {1024, 201024, "wide (a head's width)"}};
    const int MR = 40;
    for (const shape &sh : shapes) {
        const int in = sh.in, out = sh.out;
        const std::vector<uint16_t> W = rnd_bf16((size_t)out * in, 0.02, in), X = rnd_bf16((size_t)MR * in, 1.0, in);
        uint16_t *dW = upload(W), *dX = upload(X);
        float *dY = nullptr;
        CUDA_OK(cudaMalloc((void **)&dY, (size_t)MR * out * sizeof(float)));
        auto run = [&](int M, bool prompt, int r0) {
            CUDA_OK(cudaMemset(dY, 0xff, (size_t)M * out * sizeof(float)));
            const int rc = pulsar_bf16_rows_linear_launch(dW, dX + (size_t)r0 * in, M, in, out, dY, prompt, 0);
            CHECK(rc == 0, "%s: M=%d prompt=%d refused (rc=%d)", sh.what, M, prompt, rc);
            return download(dY, (size_t)M * out);
        };
        std::vector<double> ref, env;
        const int nref = out > 100000 ? 16 : MR;   /* the wide shape: the decode rows only */
        ref_linear(W.data(), X.data(), 0, nref, in, out, ref, env);
        /* 1. decode rows */
        const std::vector<float> y16 = run(16, false, 0);
        double worst = 0;
        const size_t over = over_env(y16.data(), ref, env, (size_t)16 * out, &worst);
        size_t diff = 0;
        for (int M : {1, 5, 9}) {
            const std::vector<float> y = run(M, false, 0);
            diff += memcmp(y.data(), y16.data(), y.size() * sizeof(float)) != 0;
        }
        if (out < 100000) {   /* decode rows past 16 (a wide verify): 16-row blocks, each row a one-row call's bytes */
            const std::vector<float> y40 = run(MR, false, 0);
            diff += memcmp(y40.data(), y16.data(), y16.size() * sizeof(float)) != 0;
            for (int r : {16, 23, MR - 1}) {
                const std::vector<float> y1 = run(1, false, r);
                diff += memcmp(y1.data(), y40.data() + (size_t)r * out, (size_t)out * sizeof(float)) != 0;
            }
        }
        CHECK(over == 0, "%s decode: %zu outputs past the f32 envelope (worst err / sum|wx| %.2e)", sh.what, over, worst);
        CHECK(diff == 0, "%s: %zu decode calls (M = 1, 5, 9; M = 40 in 16-row blocks, and its rows 16, 23, 39 alone) differ "
              "from the same rows of the M = 16 call", sh.what, diff);
        printf("  dense %-22s %5d -> %-6d decode M=16: worst err/sum|wx| %.2e; M=1,5,9,40 %s\n", sh.what, in, out, worst,
               diff ? "DIFFER" : "bit-identical to M=16's rows");
        if (out > 100000) {
            cudaFree(dW); cudaFree(dX); cudaFree(dY);
            continue;
        }
        /* 2. prompt chunks */
        const std::vector<float> yp = run(MR, true, 0);
        double pworst = 0;
        const size_t pover = over_env(yp.data(), ref, env, (size_t)MR * out, &pworst);
        size_t pdiff = 0, same16 = memcmp(yp.data(), y16.data(), y16.size() * sizeof(float)) == 0;
        for (int cut : {1, 7, 33}) {
            const std::vector<float> a = run(cut, true, 0), b = run(MR - cut, true, cut);
            pdiff += memcmp(a.data(), yp.data(), a.size() * sizeof(float)) != 0;
            pdiff += memcmp(b.data(), yp.data() + (size_t)cut * out, b.size() * sizeof(float)) != 0;
        }
        CHECK(pover == 0, "%s prompt: %zu outputs past the f32 envelope (worst %.2e)", sh.what, pover, pworst);
        CHECK(pdiff == 0, "%s prompt: %zu of 6 chunk halves differ from the M = 40 call's rows", sh.what, pdiff);
        printf("  dense %-22s prompt M=40 (cuBLASLt): worst %.2e; cut at 1 / 7 / 33 %s; rows 0..15 %s the rows "
               "kernel's\n", sh.what, pworst, pdiff ? "DIFFERS" : "bit-identical", same16 ? "equal to" : "rounding-close to");
        /* 3. the front door, once per shape */
        pulsar_rows_linear l{};
        l.w = dW; l.in = in; l.out = out; l.bf16 = true;
        CUDA_OK(cudaMemset(dY, 0xff, (size_t)16 * out * sizeof(float)));
        CHECK(pulsar_rows_linear_launch(&l, dX, 16, dY, nullptr, 0, 0) == 0, "%s: the front door refused", sh.what);
        const std::vector<float> yf = download(dY, (size_t)16 * out);
        CHECK(memcmp(yf.data(), y16.data(), yf.size() * sizeof(float)) == 0, "%s: the front door != the arm", sh.what);
        l.k2 = 8;
        CHECK(pulsar_rows_linear_launch(&l, dX, 16, dY, nullptr, 0, 0) == -1, "%s: a BF16 reference with a rate accepted",
              sh.what);
        cudaFree(dW); cudaFree(dX); cudaFree(dY);
    }
    CHECK(pulsar_bf16_rows_linear_launch((void *)256, (const uint16_t *)256, 4, 4096 + 32, 64, (float *)256, false, 0) == -1,
          "in %% 64 != 0 accepted");
    printf("  dense refusals: in %% 64, a BF16 reference carrying an EXL3 rate -- refused\n");
}

struct moe_fix {
    int E, H, mid, k;
    pulsar_ffn_act act = PULSAR_FFN_ACT_SWIGLU;   /* RELU2: up-only experts, dG NULL */
    std::vector<uint16_t> G, U, D;
    uint16_t *dG = nullptr, *dU = nullptr, *dD = nullptr;
};

/* the routed arm on T rows: returns out [T][H] and (optionally) the intermediates */
struct moe_run {
    std::vector<float> out, gz, uz, dz;
    std::vector<uint16_t> mid;
    uint32_t nf = 0;
};
static moe_run run_moe(const moe_fix &f, const std::vector<uint16_t> &X, const std::vector<int32_t> &sel,
                       const std::vector<float> &wts, int T, int r0, bool front_door) {
    const int k = f.k, H = f.H, M = f.mid;
    const size_t P = (size_t)T * k;
    std::vector<uint16_t> xs(X.begin() + (size_t)r0 * H, X.begin() + (size_t)(r0 + T) * H);
    std::vector<int32_t> ss(sel.begin() + (size_t)r0 * k, sel.begin() + (size_t)(r0 + T) * k);
    std::vector<float> ws(wts.begin() + (size_t)r0 * k, wts.begin() + (size_t)(r0 + T) * k);
    uint16_t *dX = upload(xs);
    int32_t *dS = upload(ss);
    float *dWt = upload(ws), *dO = nullptr, *dgz = nullptr, *duz = nullptr, *ddz = nullptr;
    uint16_t *dmid = nullptr;
    uint32_t *dnf = nullptr;
    CUDA_OK(cudaMalloc((void **)&dO, (size_t)T * H * 4));
    CUDA_OK(cudaMalloc((void **)&dnf, 4));
    CUDA_OK(cudaMemset(dnf, 0, 4));
    CUDA_OK(cudaMemset(dO, 0xff, (size_t)T * H * 4));
    moe_run r;
    if (!front_door) {
        CUDA_OK(cudaMalloc((void **)&dgz, P * M * 4));
        CUDA_OK(cudaMalloc((void **)&duz, P * M * 4));
        CUDA_OK(cudaMalloc((void **)&ddz, P * H * 4));
        CUDA_OK(cudaMalloc((void **)&dmid, P * M * 2));
        const pulsar_bf16_moe w{f.dG, f.dU, f.dD, f.E, H, M, H, f.act};
        const int rc = pulsar_bf16_moe_routed_launch(w, dS, dWt, T, k, 0.0f, dX, dgz, duz, dmid, ddz, dO, dnf, 7u, 0);
        CHECK(rc == 0, "routed T=%d: the arm refused (rc=%d)", T, rc);
        r.gz = download(dgz, P * M);
        r.uz = download(duz, P * M);
        r.dz = download(ddz, P * H);
        r.mid = download(dmid, P * M);
        cudaFree(dgz); cudaFree(duz); cudaFree(ddz); cudaFree(dmid);
    } else {
        pulsar_rows_moe w{};
        w.bf16_gate = f.dG; w.bf16_up = f.dU; w.bf16_down = f.dD; w.act = f.act;
        w.n_expert = f.E; w.top_k = k; w.in_dim = H; w.mid_dim = M; w.out_dim = H; w.ex_lo = 0; w.n_local = f.E;
        const size_t wsb = pulsar_rows_moe_routed_workspace_bytes(T, k, M, H);
        void *dws = nullptr;
        CUDA_OK(cudaMalloc(&dws, wsb));
        const int rc = pulsar_rows_moe_routed_launch(&w, dS, dWt, dX, T, dO, dws, wsb, dnf, 7u, 0);
        CHECK(rc == 0, "routed T=%d: the front door refused (rc=%d)", T, rc);
        cudaFree(dws);
    }
    r.out = download(dO, (size_t)T * H);
    const std::vector<uint32_t> nf = download(dnf, 1);
    r.nf = nf[0];
    cudaFree(dX); cudaFree(dS); cudaFree(dWt); cudaFree(dO); cudaFree(dnf);
    return r;
}

static void section_routed(pulsar_ffn_act act) {
    moe_fix f;
    const bool gated = act == PULSAR_FFN_ACT_SWIGLU;
    f.E = 16; f.H = 4096; f.mid = 2048; f.k = 6; f.act = act;
    if (gated) f.G = rnd_bf16((size_t)f.E * f.mid * f.H, 0.02, f.H);
    f.U = rnd_bf16((size_t)f.E * f.mid * f.H, 0.02, f.H);
    f.D = rnd_bf16((size_t)f.E * f.H * f.mid, 0.02, f.mid);
    f.dG = gated ? upload(f.G) : nullptr; f.dU = upload(f.U); f.dD = upload(f.D);
    printf("  -- routed, %s\n", gated ? "SwiGLU (gate + up)" : "relu^2 (up-only, no gate stack)");
    const int T = 40, k = f.k, H = f.H, M = f.mid;
    const std::vector<uint16_t> X = rnd_bf16((size_t)T * H, 1.0, H);
    std::vector<int32_t> sel((size_t)T * k);
    std::vector<float> wts((size_t)T * k);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < k; j++) {
            int e;
            bool dup;
            do {   /* Zipf-like: low ids hot */
                e = (int)(f.E * pow(rndu(), 2.5));
                dup = false;
                for (int i = 0; i < j; i++) dup |= sel[(size_t)t * k + i] == e;
            } while (dup || e >= f.E);
            sel[(size_t)t * k + j] = (t % 7 == 3 && j == 4) ? -1 : e;   /* a dropped pick (another rank's) */
            wts[(size_t)t * k + j] = (float)(0.05 + rndu() * 0.3);
        }
    /* the arm at T = 16 (decode) with its intermediates, graded */
    const moe_run r = run_moe(f, X, sel, wts, 16, 0, false);
    size_t zover = 0, mid_bad = 0, dover = 0, sum_bad = 0, pairs = 0;
    double zworst = 0, dworst = 0;
    std::vector<double> y, env;
    for (int t = 0; t < 16; t++)
        for (int j = 0; j < k; j++) {
            const int e = sel[(size_t)t * k + j];
            const size_t p = (size_t)t * k + j;
            if (e < 0) continue;
            pairs++;
            double w = 0;
            for (int pr = gated ? 0 : 1; pr < 2; pr++) {   /* gate (SwiGLU only), up */
                const std::vector<uint16_t> &S = pr ? f.U : f.G;
                ref_linear(S.data() + (size_t)e * M * H, X.data(), t, t + 1, H, M, y, env);
                zover += over_env((pr ? r.uz : r.gz).data() + p * M, y, env, M, &w);
                zworst = fmax(zworst, w);
            }
            for (int i = 0; i < M; i++) {   /* the mid: the activation of the device's own z, within one bf16 ulp */
                const double u = r.uz[p * M + i], g = gated ? r.gz[p * M + i] : 0.0, ru = u > 0 ? u : 0.0;
                const double h = gated ? g / (1.0 + exp(-g)) * u * wts[p] : ru * ru * wts[p];
                const double got = bf16_val(r.mid[p * M + i]);
                const double ulp = ldexp(1.0, (int)floor(log2(fmax(fabs(h), 1e-30))) - 7);
                mid_bad += !(fabs(got - h) <= ulp);
            }
            ref_linear(f.D.data() + (size_t)e * H * M, r.mid.data(), (int)p, (int)p + 1, M, H, y, env);
            dover += over_env(r.dz.data() + p * H, y, env, H, &w);
            dworst = fmax(dworst, w);
        }
    for (int t = 0; t < 16; t++)
        for (int o = 0; o < H; o++) {
            float acc = 0.0f;
            for (int j = 0; j < k; j++)
                if (sel[(size_t)t * k + j] >= 0) acc += r.dz[((size_t)t * k + j) * H + o];
            sum_bad += memcmp(&acc, &r.out[(size_t)t * H + o], 4) != 0;
        }
    CHECK(zover == 0, "routed gate / up: %zu outputs past the f32 envelope (worst %.2e)", zover, zworst);
    CHECK(mid_bad == 0, "routed mid: %zu values past one bf16 ulp of the SwiGLU of the device's z", mid_bad);
    CHECK(dover == 0, "routed down: %zu outputs past the f32 envelope (worst %.2e)", dover, dworst);
    CHECK(sum_bad == 0, "routed sum: %zu outputs differ from the slots' down z added in slot order", sum_bad);
    CHECK(r.nf == 0, "routed: the non-finite flag raised on finite rows (%u)", r.nf);
    printf("  routed %d -> %d -> %d, %d experts, top-%d, T=16 (%zu pairs, dropped picks skipped): gate / up worst %.2e, "
           "mid within 1 bf16 ulp, down worst %.2e, sum == slot order %s\n", H, M, H, f.E, k, pairs, zworst, dworst,
           sum_bad ? "DIFFERS" : "bit for bit");
    /* batch / chunk invariance: every token of a T = 16 and a T = 40 call == the token alone */
    const moe_run r40 = run_moe(f, X, sel, wts, T, 0, false);
    size_t inv_bad = 0;
    for (int t = 0; t < T; t++) {
        const moe_run r1 = run_moe(f, X, sel, wts, 1, t, false);
        inv_bad += memcmp(r1.out.data(), r40.out.data() + (size_t)t * H, (size_t)H * 4) != 0;
        if (t < 16) inv_bad += memcmp(r1.out.data(), r.out.data() + (size_t)t * H, (size_t)H * 4) != 0;
    }
    CHECK(inv_bad == 0, "routed: %zu token outputs at T = 16 / 40 differ from the token alone", inv_bad);
    /* the front door == the arm */
    const moe_run rf = run_moe(f, X, sel, wts, T, 0, true);
    CHECK(memcmp(rf.out.data(), r40.out.data(), rf.out.size() * 4) == 0, "routed: the front door != the arm");
    /* a NaN row raises the flag */
    std::vector<uint16_t> Xn = X;
    Xn[(size_t)5 * H + 17] = 0x7fc0;
    const moe_run rn = run_moe(f, Xn, sel, wts, 8, 0, false);
    CHECK(rn.nf == 7u, "routed: a NaN row did not raise the non-finite flag (%u)", rn.nf);
    /* refusals at the front door */
    {
        pulsar_rows_moe w{};
        w.bf16_gate = f.dG; w.bf16_up = f.dU; w.bf16_down = f.dD; w.act = f.act;
        w.n_expert = f.E; w.top_k = k; w.in_dim = H; w.mid_dim = M; w.out_dim = H; w.n_local = f.E;
        pulsar_rows_moe_shared sh{};
        w.shared = &sh; w.n_shared = 1;
        const size_t wsb = pulsar_rows_moe_routed_workspace_bytes(1, k, M, H);
        void *dws = nullptr;
        CUDA_OK(cudaMalloc(&dws, wsb));
        int32_t *dS = upload(std::vector<int32_t>(sel.begin(), sel.begin() + k));
        float *dWt = upload(std::vector<float>(wts.begin(), wts.begin() + k)), *dO = nullptr;
        uint16_t *dX = upload(std::vector<uint16_t>(X.begin(), X.begin() + H));
        uint32_t *dnf = nullptr;
        CUDA_OK(cudaMalloc((void **)&dO, (size_t)H * 4));
        CUDA_OK(cudaMalloc((void **)&dnf, 4));
        CHECK(pulsar_rows_moe_routed_launch(&w, dS, dWt, dX, 1, dO, dws, wsb, dnf, 7u, 0) == -1,
              "routed: BF16 stacks with a riding shared expert accepted");
        w.shared = nullptr; w.n_shared = 0;
        w.down_table = reinterpret_cast<const void *const *>(dws);
        CHECK(pulsar_rows_moe_routed_launch(&w, dS, dWt, dX, 1, dO, dws, wsb, dnf, 7u, 0) == -1,
              "routed: BF16 stacks with an EXL3 table accepted");
        w.down_table = nullptr;
        /* the activation names the stacks: relu^2 with a gate stack, SwiGLU without one */
        w.act = gated ? PULSAR_FFN_ACT_RELU2 : PULSAR_FFN_ACT_SWIGLU;
        CHECK(pulsar_rows_moe_routed_launch(&w, dS, dWt, dX, 1, dO, dws, wsb, dnf, 7u, 0) == -1,
              "routed: %s accepted", gated ? "relu^2 with a gate stack" : "SwiGLU without a gate stack");
        const pulsar_bf16_moe bad{f.dU, f.dU, f.dD, f.E, H, M, H, PULSAR_FFN_ACT_RELU2};
        CHECK(pulsar_bf16_moe_routed_launch(bad, dS, dWt, 1, k, 0.0f, dX, nullptr, dO, (uint16_t *)dws, dO, dO, dnf, 7u, 0)
              == -1, "routed: the arm took a gate stack for relu^2");
        cudaFree(dws); cudaFree(dS); cudaFree(dWt); cudaFree(dO); cudaFree(dX); cudaFree(dnf);
    }
    printf("  routed: T=16 and T=40 tokens bit-identical to each token alone (%d tokens); the front door == the arm "
           "%s; a NaN row raises the flag %s; refusals (a riding shared expert, an EXL3 table beside BF16)\n", T,
           memcmp(rf.out.data(), r40.out.data(), rf.out.size() * 4) ? "NO" : "yes", rn.nf == 7u ? "yes" : "NO");
    cudaFree(f.dG); cudaFree(f.dU); cudaFree(f.dD);
}

/* L296 N4: the non-gated shared expert (RELU2) through the shared launcher, on BF16 linears at Nemotron-H's width */
static void section_shared_relu2(void) {
    const int H = 4096, M = 5376, T = 16;
    const std::vector<uint16_t> U = rnd_bf16((size_t)M * H, 0.02, H), D = rnd_bf16((size_t)H * M, 0.02, M);
    const std::vector<uint16_t> X = rnd_bf16((size_t)T * H, 1.0, H);
    std::vector<float> pre((size_t)T * H);
    for (auto &v : pre) v = (float)(rndu() - 0.5);
    uint16_t *dU = upload(U), *dD = upload(D);
    struct out { std::vector<float> o, yu, ys; std::vector<uint16_t> h; int rc; };
    auto run = [&](int rows, int r0) {
        uint16_t *dX = upload(std::vector<uint16_t>(X.begin() + (size_t)r0 * H, X.begin() + (size_t)(r0 + rows) * H));
        float *dO = upload(std::vector<float>(pre.begin() + (size_t)r0 * H, pre.begin() + (size_t)(r0 + rows) * H));
        float *dyu = nullptr, *dys = nullptr;
        uint16_t *dh = nullptr;
        CUDA_OK(cudaMalloc((void **)&dyu, (size_t)rows * M * 4));
        CUDA_OK(cudaMalloc((void **)&dys, (size_t)rows * H * 4));
        CUDA_OK(cudaMalloc((void **)&dh, (size_t)rows * M * 2));
        pulsar_rows_moe_shared q{};
        q.act = PULSAR_FFN_ACT_RELU2;
        q.up = pulsar_rows_linear{dU, nullptr, 0, H, M, false, true};
        q.down = pulsar_rows_linear{dD, nullptr, 0, M, H, false, true};
        q.yu = dyu; q.ys = dys; q.h = dh;   /* no gate, no yg */
        out r;
        r.rc = pulsar_rows_moe_shared_launch(&q, dX, rows, dO, 0);
        r.o = download(dO, (size_t)rows * H);
        r.yu = download(dyu, (size_t)rows * M);
        r.ys = download(dys, (size_t)rows * H);
        r.h = download(dh, (size_t)rows * M);
        cudaFree(dX); cudaFree(dO); cudaFree(dyu); cudaFree(dys); cudaFree(dh);
        return r;
    };
    const out r = run(T, 0);
    CHECK(r.rc == 0, "shared relu^2: the launcher refused (rc=%d)", r.rc);
    std::vector<double> y, env;
    double uworst = 0, dworst = 0;
    size_t uover = 0, hbad = 0, dover = 0, obad = 0;
    ref_linear(U.data(), X.data(), 0, T, H, M, y, env);
    uover = over_env(r.yu.data(), y, env, (size_t)T * M, &uworst);
    for (size_t i = 0; i < (size_t)T * M; i++) {
        const double u = r.yu[i], rl = u > 0 ? u : 0.0, h = rl * rl, got = bf16_val(r.h[i]);
        const double ulp = ldexp(1.0, (int)floor(log2(fmax(fabs(h), 1e-30))) - 7);
        hbad += !(fabs(got - h) <= ulp);
    }
    ref_linear(D.data(), r.h.data(), 0, T, M, H, y, env);
    dover = over_env(r.ys.data(), y, env, (size_t)T * H, &dworst);
    for (size_t i = 0; i < (size_t)T * H; i++) {
        const float want = pre[i] + r.ys[i];
        obad += memcmp(&want, &r.o[i], 4) != 0;
    }
    size_t inv = 0;
    for (int t = 0; t < T; t += 5) {
        const out r1 = run(1, t);
        inv += memcmp(r1.o.data(), r.o.data() + (size_t)t * H, (size_t)H * 4) != 0;
    }
    CHECK(uover == 0 && hbad == 0 && dover == 0 && obad == 0 && inv == 0,
          "shared relu^2: up %zu / rows %zu / down %zu outside, out != routed + shared at %zu, %zu rows differ alone",
          uover, hbad, dover, obad, inv);
    printf("  shared relu^2 (BF16 %d -> %d -> %d, %d rows): up worst %.2e, relu^2 rows within 1 bf16 ulp, down worst "
           "%.2e, out == the routed sum + the expert bit for bit, rows alone == batched\n", H, M, H, T, uworst, dworst);
    cudaFree(dU); cudaFree(dD);
}

static void section_time(void) {
    const int in = 4096, out = 201024;
    uint16_t *dW = nullptr, *dX = nullptr;
    float *dY = nullptr;
    CUDA_OK(cudaMalloc((void **)&dW, (size_t)out * in * 2));
    CUDA_OK(cudaMemset(dW, 0x3c, (size_t)out * in * 2));
    CUDA_OK(cudaMalloc((void **)&dX, (size_t)16 * in * 2));
    CUDA_OK(cudaMemset(dX, 0x3c, (size_t)16 * in * 2));
    CUDA_OK(cudaMalloc((void **)&dY, (size_t)16 * out * 4));
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    printf("  time, the BF16 head 4096 -> 201024 (1.65 GB):");
    for (const int M : {1, 4, 16}) {
        for (int i = 0; i < 2; i++) (void)pulsar_bf16_rows_linear_launch(dW, dX, M, in, out, dY, false, 0);
        CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < 10; i++) (void)pulsar_bf16_rows_linear_launch(dW, dX, M, in, out, dY, false, 0);
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        printf("  %2d rows %.2f ms (%.0f GB/s)", M, ms / 10, (double)out * in * 2 / (ms / 10 * 1e-3) / 1e9);
    }
    printf("\n");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("bf16-rows-gate: BF16 weights x raw bf16 rows (dense and routed) vs double\n");
    section_dense();
    section_routed(PULSAR_FFN_ACT_SWIGLU);
    section_routed(PULSAR_FFN_ACT_RELU2);
    section_shared_relu2();
    section_time();
    printf(g_fail ? "BF16-ROWS GATE FAIL\n" : "BF16-ROWS GATE PASS\n");
    return g_fail;
}

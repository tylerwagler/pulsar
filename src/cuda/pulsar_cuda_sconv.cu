/* Short convolution block (L295 E3): the standalone launch of the arithmetic in
 * pulsar_cuda_sconv.cuh.  Contract and state layout in pulsar_cuda_sconv.h.
 *
 * Schedule.  One CTA per (channel tile of 1024, sequence, row tile of 16, stream);
 * a thread owns 8 channels (one 16-byte load a row) and walks its tile's rows
 * with the K-1 previous inputs in registers.  A row tile after the first starts
 * its window from the call's own input rows (immutable for the launch); only the
 * FIRST row tile of a sequence reads the slot, and that same thread -- the only
 * reader of those channels of that slot -- writes the advanced state after its
 * rows, so the in-place update has no hazard.  The schedule changes which thread
 * computes a (row, channel), never what it computes. */
#include "pulsar_cuda_sconv.h"
#include "pulsar_cuda_sconv.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace {

constexpr int kThreads = 128;           /* per CTA */
constexpr int kVec = 8;                 /* channels per thread: one uint4 of bf16 */
constexpr int kTile = kThreads * kVec;  /* channels per CTA */
constexpr int kRows = 16;               /* rows per CTA; >= KMAX - 1, so only tile 0 reads the slot */
static_assert(kRows >= PULSAR_SCONV_KMAX - 1, "a later row tile must find its whole window in the call's rows");

struct stream_args {
    const uint16_t *w, *x;
    uint16_t *y, *state, *cap;
    uint64_t ld_x, ld_y, state_stride, cap_stride;
    uint32_t C;
    int residual;
};

struct call_args {
    stream_args s[PULSAR_SCONV_STREAMS_MAX];
    const int32_t *seq_first, *seq_slot;
    uint32_t tiles_per_seq;
};

__device__ __forceinline__ void unpack8(uint4 v, float *f) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = pulsar_sconv_bf2f((uint16_t)(u[i] & 0xffffu));
        f[2 * i + 1] = pulsar_sconv_bf2f((uint16_t)(u[i] >> 16));
    }
}

/* bf16 values held in f32 (exact) back to their bits */
__device__ __forceinline__ uint4 pack8_exact(const float *f) {
    uint32_t u[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        u[i] = (uint32_t)pulsar_sconv_f2bf(f[2 * i]) | ((uint32_t)pulsar_sconv_f2bf(f[2 * i + 1]) << 16);
    return make_uint4(u[0], u[1], u[2], u[3]);
}

template <int K>
__global__ void __launch_bounds__(kThreads) sconv_kernel(const call_args a) {
    const stream_args &S = a.s[blockIdx.z];
    const uint32_t c0 = (blockIdx.x * kThreads + threadIdx.x) * kVec;
    if (c0 >= S.C) return;
    const uint32_t seq = blockIdx.y / a.tiles_per_seq;
    const int32_t t0 = (int32_t)(blockIdx.y % a.tiles_per_seq) * kRows;
    const int32_t f0 = a.seq_first[seq];
    const int32_t n = a.seq_first[seq + 1] - f0;
    if (t0 >= n) return;
    const int32_t t1 = min(n, t0 + kRows);

    float w[kVec][K];
    {
        const uint16_t *wp = S.w + (uint64_t)c0 * K;   /* [C][K]: this thread's 8 channels are 8K contiguous taps */
#pragma unroll
        for (int j = 0; j < kVec; ++j)
#pragma unroll
            for (int i = 0; i < K; ++i) w[j][i] = pulsar_sconv_bf2f(wp[j * K + i]);
    }
    const uint16_t *xs = S.x + (uint64_t)f0 * S.ld_x + c0;   /* the sequence's row 0, this thread's channels */
    uint16_t *slot = S.state + (uint64_t)a.seq_slot[seq] * S.state_stride + c0;

    /* hist[i][j]: the input at row t - (K-1) + i, channel c0 + j, for the next row t */
    float hist[K - 1][kVec];
    uint4 old[K - 1];
    if (t0 == 0) {
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
            old[i] = *(const uint4 *)(slot + (uint64_t)i * S.C);
            unpack8(old[i], hist[i]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < K - 1; ++i) unpack8(*(const uint4 *)(xs + (uint64_t)(t0 - (K - 1) + i) * S.ld_x), hist[i]);
    }

    for (int32_t t = t0; t < t1; ++t) {
        float xr[kVec];
        unpack8(*(const uint4 *)(xs + (uint64_t)t * S.ld_x), xr);
        uint32_t o[kVec / 2];
#pragma unroll
        for (int j = 0; j < kVec; j += 2) {
            float h0[K - 1], h1[K - 1];
#pragma unroll
            for (int i = 0; i < K - 1; ++i) { h0[i] = hist[i][j]; h1[i] = hist[i][j + 1]; }
            const uint16_t lo = pulsar_sconv_out(pulsar_sconv_taps<K>(w[j], h0, xr[j]), xr[j], S.residual);
            const uint16_t hi = pulsar_sconv_out(pulsar_sconv_taps<K>(w[j + 1], h1, xr[j + 1]), xr[j + 1], S.residual);
            o[j / 2] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
        *(uint4 *)(S.y + (uint64_t)(f0 + t) * S.ld_y + c0) = make_uint4(o[0], o[1], o[2], o[3]);
#pragma unroll
        for (int i = 0; i < K - 2; ++i)
#pragma unroll
            for (int j = 0; j < kVec; ++j) hist[i][j] = hist[i + 1][j];
#pragma unroll
        for (int j = 0; j < kVec; ++j) hist[K - 2][j] = xr[j];
        if (S.cap) {
            uint16_t *cp = S.cap + (uint64_t)(f0 + t) * S.cap_stride + c0;
#pragma unroll
            for (int i = 0; i < K - 1; ++i) *(uint4 *)(cp + (uint64_t)i * S.C) = pack8_exact(hist[i]);
        }
    }

    if (t0 == 0) {
        /* the state after the sequence's last row: positions n - (K-1) .. n-1 of [old state | the call's rows] */
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
            const int32_t idx = n + i;   /* index into the K-1 old rows followed by the n new ones */
            uint4 v = old[K - 2];
            if (idx >= K - 1) {
                v = *(const uint4 *)(xs + (uint64_t)(idx - (K - 1)) * S.ld_x);
            } else {
#pragma unroll
                for (int j = 0; j < K - 2; ++j)   /* a select, so `old` stays in registers */
                    if (idx == j) v = old[j];
            }
            *(uint4 *)(slot + (uint64_t)i * S.C) = v;
        }
    }
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

bool refuse(const char *what, uint32_t i) {
    fprintf(stderr, "pulsar: sconv: stream %u: %s -- refusing\n", i, what);
    return false;
}

/* [lo, hi) bytes of `rows` rows of `C` bf16 at stride ld */
void span(const uint16_t *p, uint64_t ld, uint32_t rows, uint32_t C, uintptr_t *lo, uintptr_t *hi) {
    *lo = (uintptr_t)p;
    *hi = (uintptr_t)(p + (uint64_t)(rows - 1) * ld + C);
}

bool stream_ok(const pulsar_sconv_stream &s, uint32_t i, uint32_t n_rows, uint32_t K0) {
    const pulsar_sconv_decl *d = s.decl;
    if (!d) return refuse("no declaration", i);
    if (d->channels == 0 || d->channels % kVec) return refuse("channels must be a positive multiple of 8", i);
    if (d->kernel < 2 || d->kernel > PULSAR_SCONV_KMAX) return refuse("kernel must be 2 .. PULSAR_SCONV_KMAX", i);
    if (d->kernel != K0) return refuse("every stream of one call has the first stream's kernel", i);
    if (d->residual != 0 && d->residual != 1) return refuse("residual must be 0 or 1", i);
    if (!d->weight || !aligned16(d->weight)) return refuse("no weight, or a weight not 16-byte aligned", i);
    if (!s.x || !s.y || !s.state || !aligned16(s.x) || !aligned16(s.y) || !aligned16(s.state))
        return refuse("x, y and state must be set and 16-byte aligned", i);
    const uint64_t se = pulsar_sconv_state_elems(d);
    if (s.ld_x < d->channels || s.ld_y < d->channels || s.ld_x % kVec || s.ld_y % kVec)
        return refuse("ld_x / ld_y must be >= channels and multiples of 8", i);
    if (s.state_stride < se || s.state_stride % kVec) return refuse("state_stride must be >= (K-1) * C and a multiple of 8", i);
    if (s.capture && (!aligned16(s.capture) || s.capture_stride < se || s.capture_stride % kVec))
        return refuse("capture must be 16-byte aligned with a stride >= (K-1) * C, a multiple of 8", i);
    uintptr_t xl, xh, yl, yh;
    span(s.x, s.ld_x, n_rows, d->channels, &xl, &xh);
    span(s.y, s.ld_y, n_rows, d->channels, &yl, &yh);
    if (xl < yh && yl < xh) return refuse("the output overlaps the input rows (a later row reads an earlier row's input)", i);
    return true;
}

template <int K>
cudaError_t launch(const call_args &a, dim3 grid, cudaStream_t stream) {
    sconv_kernel<K><<<grid, kThreads, 0, stream>>>(a);
    return cudaGetLastError();
}

} // namespace

extern "C" uint64_t pulsar_sconv_state_elems(const pulsar_sconv_decl *decl) {
    return decl && decl->kernel >= 1 ? (uint64_t)(decl->kernel - 1) * decl->channels : 0;
}

extern "C" int pulsar_sconv_forward(const pulsar_sconv_stream *streams, uint32_t n_streams, const pulsar_sconv_rows *rows,
                                    cudaStream_t stream) {
    if (!streams || n_streams < 1 || n_streams > PULSAR_SCONV_STREAMS_MAX || !rows) {
        fprintf(stderr, "pulsar: sconv: %u streams (1 .. %d) or no rows -- refusing\n", n_streams, PULSAR_SCONV_STREAMS_MAX);
        return -1;
    }
    if (rows->n_seqs == 0 || rows->n_rows == 0 || rows->max_seq_rows == 0 || rows->max_seq_rows > rows->n_rows ||
        rows->n_seqs > rows->n_rows || !rows->seq_first || !rows->seq_slot) {
        fprintf(stderr, "pulsar: sconv: %u sequences, %u rows, longest %u, or no row descriptors -- refusing\n",
                rows->n_seqs, rows->n_rows, rows->max_seq_rows);
        return -1;
    }
    const uint32_t tiles = (rows->max_seq_rows + kRows - 1) / kRows;
    if ((uint64_t)rows->n_seqs * tiles > 65535u) {
        fprintf(stderr, "pulsar: sconv: %u sequences x %u row tiles exceed one grid column -- refusing\n", rows->n_seqs, tiles);
        return -1;
    }
    const uint32_t K = streams[0].decl ? streams[0].decl->kernel : 0;
    call_args a{};
    uint32_t cmax = 0;
    for (uint32_t i = 0; i < n_streams; ++i) {
        const pulsar_sconv_stream &s = streams[i];
        if (!stream_ok(s, i, rows->n_rows, K)) return -1;
        a.s[i] = stream_args{s.decl->weight, s.x, s.y, s.state, s.capture, s.ld_x, s.ld_y, s.state_stride,
                             s.capture_stride, s.decl->channels, s.decl->residual};
        if (s.decl->channels > cmax) cmax = s.decl->channels;
    }
    a.seq_first = rows->seq_first;
    a.seq_slot = rows->seq_slot;
    a.tiles_per_seq = tiles;
    const dim3 grid((cmax + kTile - 1) / kTile, rows->n_seqs * tiles, n_streams);
    cudaError_t e = cudaErrorInvalidValue;
    switch (K) {
        case 2: e = launch<2>(a, grid, stream); break;
        case 3: e = launch<3>(a, grid, stream); break;
        case 4: e = launch<4>(a, grid, stream); break;
        case 5: e = launch<5>(a, grid, stream); break;
        case 6: e = launch<6>(a, grid, stream); break;
        case 7: e = launch<7>(a, grid, stream); break;
        case 8: e = launch<8>(a, grid, stream); break;
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: sconv: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

extern "C" int pulsar_sconv_restore(const pulsar_sconv_stream *s, uint32_t slot, uint32_t row, cudaStream_t stream) {
    if (!s || !s->decl || !s->state || !s->capture || s->state_stride < pulsar_sconv_state_elems(s->decl) ||
        s->capture_stride < pulsar_sconv_state_elems(s->decl)) {
        fprintf(stderr, "pulsar: sconv restore: no stream, state, capture, or a stride below one record -- refusing\n");
        return -1;
    }
    const cudaError_t e = cudaMemcpyAsync(s->state + (uint64_t)slot * s->state_stride, s->capture + (uint64_t)row * s->capture_stride,
                                          pulsar_sconv_state_elems(s->decl) * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: sconv restore: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

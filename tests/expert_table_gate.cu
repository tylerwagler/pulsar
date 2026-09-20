/* EXPERT TABLE GATE -- PLAN 94 phase 1 (L217).
 *
 * The per-expert ADDRESS TABLE is the one place an expert's address is computed
 * (pulsar_cuda_expert_table.cu).  Every consumer -- the grouped CUTLASS MXFP4
 * setup, three GEMV arms, and both type-44 planes -- reads it instead of doing
 * `base + e*stride` itself.  That indirection exists so phase 2 can point an
 * expert at a slot the cache assigns, moves and evicts at runtime.
 *
 * WHY THIS IS NOT OPTIONAL (plan s4.1).  A wrong expert ADDRESS has the same
 * signature as the mis-ordered router that `cuda-reap-router-audit` exists for:
 * fluent output, no crash.  The byte gates catch it on the artifacts they run,
 * but they only catch it WHERE THEY RUN.  This gate pins the rule itself,
 * model-free and in milliseconds, so phase 2 can change what FILLS the table
 * and be graded on the contract rather than on a logits diff that happens to
 * cover the shape.
 *
 * It checks three things:
 *   1. the arithmetic -- entries[e] (and the [d,q] pair for type 44) is exactly
 *      what the consumers used to derive;
 *   2. the CACHE identity -- one table per stack, not one per call, because
 *      "the table is the authority" is false if two callers get two tables;
 *   3. the REAL-ARTIFACT arithmetic -- the type-44 layout for the shapes of the
 *      shipped artifact reproduces its own tensor byte size (553,648,128 B),
 *      which is the number the ledger's fit tables are built from.  That is the
 *      one assertion here that cannot be satisfied by a self-consistent mistake:
 *      nb/M here are the artifact's, the byte count is the artifact's, and they
 *      were derived independently.
 *
 * usage: ./tests/expert_table_gate
 */
#include "../src/cuda/pulsar_cuda_expert_table.cu"

#include <cstdio>
#include <vector>

/* The table TU reaches cuda_ok() only from the flag-free paths; the engine's is
 * in pulsar_cuda_runtime.cu.  Mirrors its contract and prints CUDA's message. */
int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "expert-table: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "EXPERT-TABLE FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)

/* Read a device pointer array back and compare against a hand-computed one. */
static void check_table(const char *what, const void *const *dev, const void *base,
                        uint32_t n_total, uint32_t planes,
                        uint64_t stride0, uint64_t off1, uint64_t stride1) {
    std::vector<const void *> got((size_t)n_total * planes, (const void *)0x1);
    cudaError_t e = cudaMemcpy(got.data(), dev, got.size() * sizeof(void *), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { CHECK(false, "%s: reading the table back: %s", what, cudaGetErrorString(e)); return; }
    for (uint32_t i = 0; i < n_total; i++) {
        const char *want_d = (const char *)base + (size_t)i * stride0;
        CHECK(got[(size_t)i * planes] == (const void *)want_d,
              "%s: entry %u plane 0 = %p, expected %p (base + %u*%llu)",
              what, i, got[(size_t)i * planes], (const void *)want_d, i,
              (unsigned long long)stride0);
        if (planes > 1) {
            const char *want_q = (const char *)base + off1 + (size_t)i * stride1;
            CHECK(got[(size_t)i * planes + 1] == (const void *)want_q,
                  "%s: entry %u plane 1 = %p, expected %p (base + %llu + %u*%llu)",
                  what, i, got[(size_t)i * planes + 1], (const void *)want_q,
                  (unsigned long long)off1, i, (unsigned long long)stride1);
        }
    }
}

int main(void) {
    /* A synthetic arena: any distinguishable address works, the table does not
     * dereference it. */
    std::vector<char> arena(1u << 20);
    const void *base = arena.data();

    /* 1. type 40: one plane, base + e*stride. */
    {
        const uint32_t n = 256;
        const uint64_t stride = 553648128u / 256u;      /* the artifact's per-expert bytes */
        const void *const *t = (const void *const *)mxfp4_expert_table(base, stride, n);
        CHECK(t != NULL, "mxfp4_expert_table returned NULL for a valid stack");
        if (t) {
            check_table("type 40", t, base, n, 1, stride, 0, 0);
            /* 2. the cache identity: the SAME stack must yield the SAME table. */
            const void *const *again = (const void *const *)mxfp4_expert_table(base, stride, n);
            CHECK(again == t, "a second call for the same stack returned a different table (%p vs %p)"
                              " -- the table is not the single authority", (const void *)again, (const void *)t);
            printf("type 40: 256 entries, base + e*%llu, one table per stack\n", (unsigned long long)stride);
        }
    }

    /* 3. type 44: the [d,q] pair, and the REAL artifact's geometry.  nb = K/256
     * with K = 4096, M = 2048, E = 256 is one shipped expert tensor, whose own
     * byte size the ledger's fit tables already record as 553,648,128. */
    {
        const uint32_t E = 256, nb = 16, M = 2048;
        const uint64_t stride0 = (uint64_t)nb * M * 2ull;                 /* 65,536   */
        const uint64_t off1    = (stride0 * E + 63ull) & ~63ull;          /* 16,777,216 */
        const uint64_t stride1 = (uint64_t)nb * M * 8ull * sizeof(uint2); /* 2,097,152 */
        CHECK(stride0 == 65536ull, "type 44 d-plane stride = %llu, expected 65536", (unsigned long long)stride0);
        CHECK(off1 == 16777216ull, "type 44 q-plane offset = %llu, expected 16777216", (unsigned long long)off1);
        CHECK(stride1 == 2097152ull, "type 44 q-plane stride = %llu, expected 2097152", (unsigned long long)stride1);
        const uint64_t total = off1 + (uint64_t)E * stride1;
        CHECK(total == 553648128ull,
              "type 44 tensor bytes = %llu, but the artifact's own expert tensor is 553,648,128",
              (unsigned long long)total);

        const void *const *t = iq2_expert_table(base, E, nb, M);
        CHECK(t != NULL, "iq2_expert_table returned NULL for a valid stack");
        if (t) {
            check_table("type 44", t, base, E, 2, stride0, off1, stride1);
            CHECK(iq2_expert_table(base, E, nb, M) == t, "a second call returned a different type-44 table");
            printf("type 44: 256 [d,q] pairs, %llu B/expert x 256 = %llu B (the artifact's tensor)\n",
                   (unsigned long long)(stride0 + stride1), (unsigned long long)total);
        }
    }

    /* 4. distinct stacks must NOT share a table (a key that ignores the base
     * would hand one model's addresses to another). */
    {
        const void *other = arena.data() + 4096;
        const void *const *a = (const void *const *)mxfp4_expert_table(base, 4096, 8);
        const void *const *b = (const void *const *)mxfp4_expert_table(other, 4096, 8);
        CHECK(a && b && a != b, "two different bases share one table -- the key is not the stack");
        const void *const *c = (const void *const *)mxfp4_expert_table(base, 8192, 8);
        CHECK(a && c && a != c, "two different strides share one table -- the key is not the stack");
        printf("keying: base and stride both distinguish stacks\n");
    }

    /* 5. bad input refuses rather than fabricating a table. */
    {
        CHECK(mxfp4_expert_table(NULL, 4096, 8) == NULL, "a NULL base must refuse");
        CHECK(mxfp4_expert_table(base, 0, 8) == NULL, "a zero stride must refuse");
        CHECK(mxfp4_expert_table(base, 4096, 0) == NULL, "a zero expert count must refuse");
        CHECK(iq2_expert_table(base, 0, 16, 2048) == NULL, "a zero expert count must refuse (type 44)");
        printf("bad input refuses\n");
    }

    mxfp4_expert_tables_clear();   /* the cleanup contract: must not crash, must reset */
    if (g_fail) { fprintf(stderr, "EXPERT-TABLE GATE FAIL\n"); return 1; }
    printf("EXPERT-TABLE GATE PASS\n");
    return 0;
}

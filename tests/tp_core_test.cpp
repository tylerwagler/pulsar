/*
 * TP transport-core unit test (branch tensor_parallel, docs/
 * tensor-parallel-port.md slice 1).  Host-only — no CUDA, no sockets — so it
 * runs anywhere g++ does.  Pins the slab layout and identity contract that
 * the full transport (slice 3) will build on; a deliberate layout change
 * fails here first.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "tp/pulsar_tp.h"

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
#define CHECK_S(e, kind)                                                     \
    do {                                                                     \
        std::string msg = kind;                                              \
        CHECK(e, "%s", msg.c_str());                                         \
    } while (0)

static void test_slab_layout(void) {
    /* n_layer=43, n_embd=4096 => vec=16384, slots=86.  Expected offsets are
     * hand-computed from the upstream ds4_tp_slab_layout formula; a change in
     * layout order or stride fails these hard numbers. */
    const uint64_t expect_slab = 14091976ull;
    const uint64_t expect_in = 1409024ull;
    const uint64_t expect_in_flags = 2818048ull;
    const uint64_t expect_token = 2818736ull;
    const uint64_t expect_gpu_flags = 2819440ull;
    const uint64_t expect_batch_in = 8455880ull;

    pulsar_tp_slab s;
    pulsar_tp_slab_layout_init(43u, 4096u, &s);
    CHECK(s.slab_bytes == expect_slab, "slab_bytes=%llu want %llu",
          (unsigned long long)s.slab_bytes, (unsigned long long)expect_slab);
    CHECK(s.out_off == 0, "out_off=%llu want 0", (unsigned long long)s.out_off);
    CHECK(s.in_off == expect_in, "in_off=%llu want %llu",
          (unsigned long long)s.in_off, (unsigned long long)expect_in);
    CHECK(s.in_flags_off == expect_in_flags, "in_flags_off=%llu want %llu",
          (unsigned long long)s.in_flags_off, (unsigned long long)expect_in_flags);
    CHECK(s.token_off == expect_token, "token_off=%llu want %llu",
          (unsigned long long)s.token_off, (unsigned long long)expect_token);
    CHECK(s.gpu_flags_off == expect_gpu_flags, "gpu_flags_off=%llu want %llu",
          (unsigned long long)s.gpu_flags_off, (unsigned long long)expect_gpu_flags);
    CHECK(s.batch_in_off == expect_batch_in, "batch_in_off=%llu want %llu",
          (unsigned long long)s.batch_in_off, (unsigned long long)expect_batch_in);
    CHECK(pulsar_tp_slab_bytes(43u, 4096u) == expect_slab,
          "pulsar_tp_slab_bytes=%llu want %llu",
          (unsigned long long)pulsar_tp_slab_bytes(43u, 4096u),
          (unsigned long long)expect_slab);

    /* Region order and non-overlap, every slot endpoint. */
    const uint64_t vec = (uint64_t)4096 * sizeof(float);
    const uint64_t slots = 86u;
    const uint64_t batch = 43u * 8u * vec;
    CHECK(s.out_off + slots * vec <= s.in_off, "out/in overlap");
    CHECK(s.in_off + slots * vec <= s.in_flags_off, "in/flags overlap");
    CHECK(s.in_flags_off + slots * 8 <= s.token_off, "flags/token overlap");
    CHECK(s.token_off + 16 <= s.out_flags_off, "token/outflags overlap");
    CHECK(s.out_flags_off + slots * 8 <= s.gpu_flags_off, "outflags/gpuflags overlap");
    CHECK(s.gpu_flags_off + slots * 4 <= s.batch_out_off, "gpuflags/batch overlap");
    CHECK(s.batch_out_off + batch <= s.batch_in_off, "batch out/in overlap");
    CHECK(s.batch_in_off + batch == s.slab_bytes, "batch-in tail != slab_bytes");

    /* Per-(layer,gate) slots are monotone and fit their region. */
    for (uint32_t l = 0; l < 43u; l++) {
        for (uint32_t g = 0; g < PULSAR_TP_GATES_PER_LAYER; g++) {
            const uint64_t o = pulsar_tp_slab_out_offset(&s, l, g, vec);
            const uint64_t i = pulsar_tp_slab_in_offset(&s, l, g, vec);
            CHECK(o >= s.out_off && o + vec <= s.in_off, "out slot %u/%u out of range", l, g);
            CHECK(i >= s.in_off && i + vec <= s.in_flags_off, "in slot %u/%u out of range", l, g);
            const uint64_t bo = pulsar_tp_slab_batch_out_offset(&s, l, vec);
            const uint64_t bi = pulsar_tp_slab_batch_in_offset(&s, l, vec);
            CHECK(bo >= s.batch_out_off && bo + 8u * vec <= s.batch_in_off,
                  "batch-out row block %u out of range", l);
            CHECK(bi >= s.batch_in_off && bi + 8u * vec <= s.slab_bytes,
                  "batch-in row block %u out of range", l);
        }
    }
    /* Monotonic in slot index: FFN(after ATTN) sits after it. */
    CHECK(pulsar_tp_slab_out_offset(&s, 0, PULSAR_TP_GATE_ATTN, vec) <
          pulsar_tp_slab_out_offset(&s, 0, PULSAR_TP_GATE_FFN, vec),
          "ATTN slot must precede FFN slot");
}

static void test_hello_wire(void) {
    /* Fixed size: 4x u32 header + u64 gguf_bytes + 9x u32 identity + pad. */
    CHECK(sizeof(pulsar_tp_hello_fixed) == 64u,
          "hello_fixed size=%zu want 64", sizeof(pulsar_tp_hello_fixed));

    pulsar_tp_identity id = {
        87000000000ull, 3u, 43u, 4096u, 129280u, 2u, 1048576u, 0u, 0u, 86u,
    };
    pulsar_tp_hello_fixed h = {};
    h.magic = PULSAR_TP_MAGIC;
    h.version = PULSAR_TP_PROTOCOL_VERSION;
    h.role = PULSAR_TP_ROLE_LEADER;
    h.rdma_ok = 1;
    h.gguf_bytes = id.gguf_bytes;
    h.model_id = id.model_id;
    h.n_layer = id.n_layer;
    h.n_embd = id.n_embd;
    h.n_vocab = id.n_vocab;
    h.quant_bits = id.quant_bits;
    h.ctx_size = id.ctx_size;
    h.gate_slot_start = id.gate_slot_start;
    h.gate_slot_step = id.gate_slot_step;
    h.gates_per_token = id.gates_per_token;

    /* Wire round-trip is a plain memcpy of the fixed struct. */
    unsigned char raw[sizeof(pulsar_tp_hello_fixed)];
    std::memcpy(raw, &h, sizeof(raw));
    pulsar_tp_hello_fixed back;
    std::memcpy(&back, raw, sizeof(back));
    CHECK(std::memcmp(&h, &back, sizeof(h)) == 0, "hello wire round-trip differs");
    CHECK(back.magic == PULSAR_TP_MAGIC && back.version == PULSAR_TP_PROTOCOL_VERSION,
          "magic/version lost on wire");
}

static void test_identity_check(void) {
    pulsar_tp_identity mine = {
        87000000000ull, 3u, 43u, 4096u, 129280u, 2u, 1048576u, 0u, 0u, 86u,
    };
    char err[256];

    /* Same identity falls under every compared field. */
    CHECK(pulsar_tp_identity_check(&mine, &mine, err, sizeof(err)) == 0,
          "same identity rejected");

    /* ctx_size differs => still accept (exchanged, not compared — upstream). */
    pulsar_tp_identity peer = mine;
    peer.ctx_size = 65536u;
    CHECK(pulsar_tp_identity_check(&mine, &peer, err, sizeof(err)) == 0,
          "ctx_size must not be compared");

    /* Every compared field rejects with a diagnostic. */
    struct { const char *name; void (*mut)(pulsar_tp_identity *); } muts[] = {
        { "gguf_bytes", [](pulsar_tp_identity *i) { i->gguf_bytes++; } },
        { "model_id",   [](pulsar_tp_identity *i) { i->model_id++; } },
        { "n_layer",    [](pulsar_tp_identity *i) { i->n_layer++; } },
        { "n_embd",     [](pulsar_tp_identity *i) { i->n_embd++; } },
        { "n_vocab",    [](pulsar_tp_identity *i) { i->n_vocab++; } },
        { "quant_bits", [](pulsar_tp_identity *i) { i->quant_bits++; } },
        { "gate_slot_start", [](pulsar_tp_identity *i) { i->gate_slot_start++; } },
        { "gate_slot_step",  [](pulsar_tp_identity *i) { i->gate_slot_step++; } },
        { "gates_per_token", [](pulsar_tp_identity *i) { i->gates_per_token++; } },
    };
    for (auto &m : muts) {
        peer = mine;
        m.mut(&peer);
        err[0] = '\0';
        CHECK(pulsar_tp_identity_check(&mine, &peer, err, sizeof(err)) == -1,
              "field mismatch accepted: %s", m.name);
        CHECK(err[0] != '\0', "no diagnostic for field mismatch: %s", m.name);
    }

    /* err==NULL / errlen==0 is tolerated (transport passes optional buffers). */
    CHECK(pulsar_tp_identity_check(&mine, &mine, NULL, 0) == 0, "NULL err not tolerated");
}

static void test_gate_schedule(void) {
    /* Identity mapping: per_token 0 -> slot = (seq-1) % n_slots. */
    for (uint64_t seq = 1; seq <= 400; seq++)
        CHECK(pulsar_tp_gate_slot(86u, seq, 0, 0, 0) ==
              (uint32_t)((seq - 1) % 86u),
              "identity slot mismatch at seq %llu", (unsigned long long)seq);
    CHECK(pulsar_tp_gate_slot(2u, 1000, 0, 0, 0) == 1u,
          "identity wrap (1000-1)%%2 != 1");

    /* Schedule: start + ((seq-1) % per_token) * step, wrapping each token. */
    CHECK(pulsar_tp_gate_slot(86u, 1, 4, 2, 3) == 4u, "sched seq1");
    CHECK(pulsar_tp_gate_slot(86u, 2, 4, 2, 3) == 6u, "sched seq2");
    CHECK(pulsar_tp_gate_slot(86u, 3, 4, 2, 3) == 8u, "sched seq3");
    CHECK(pulsar_tp_gate_slot(86u, 4, 4, 2, 3) == 4u, "sched seq4 (wrap)");
    /* Sparse GLM-style schedule: one FFN gate on selected layers. */
    CHECK(pulsar_tp_gate_slot(172u, 1, 2, 43, 4) == 2u, "sparse seq1");
    CHECK(pulsar_tp_gate_slot(172u, 2, 2, 43, 4) == 45u, "sparse seq2");
    CHECK(pulsar_tp_gate_slot(172u, 5, 2, 43, 4) == 2u, "sparse seq5 (wrap)");
}

static void test_identity_defaults(void) {
    pulsar_tp_identity id;
    pulsar_tp_identity_init_defaults(&id, 87000000000ull, 3u, 43u, 4096u,
                                     129280u, 2u, 1048576u);
    CHECK(id.gguf_bytes == 87000000000ull && id.model_id == 3u && id.n_layer == 43u,
          "identity defaults: artifact shape");
    CHECK(id.n_embd == 4096u && id.n_vocab == 129280u && id.quant_bits == 2u &&
          id.ctx_size == 1048576u, "identity defaults: fields");
    /* DS: every layer gates ATTN then FFN -> identity slot mapping. */
    CHECK(id.gates_per_token == 0 && id.gate_slot_start == 0 &&
          id.gate_slot_step == 0, "identity defaults: DS gate schedule");
    char err[128];
    CHECK(pulsar_tp_identity_check(&id, &id, err, sizeof(err)) == 0,
          "identity defaults: self-compat through identity_check");
}

int main(void) {
    test_slab_layout();
    test_hello_wire();
    test_identity_check();
    test_identity_defaults();
    test_gate_schedule();
    if (g_failures) {
        std::fprintf(stderr, "tp_core_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("tp_core_test: ok (slab layout, hello wire, identity check, identity defaults, gate schedule)\n");
    return 0;
}

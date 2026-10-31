/*
 * TP command-free test (L335).  Host-only, built with AddressSanitizer: for
 * every frame kind pulsar_tp_recv_command parses, a command with EVERY
 * malloc'd array set (tokens, items, logits, spec_banks, spec_rngs,
 * spec_steps, images, image_bytes, keys) goes through pulsar_tp_command_free,
 * and LeakSanitizer must find nothing left after each kind.  The free also
 * resets the command: every pointer NULL, every count 0, type ERROR.
 *
 * Before L335 the free zeroed the command after tokens / items and freed the
 * rest through NULL pointers: every SET_LOGITS, speculative batch, SYNC_MM
 * and SEGMENT_RECONCILE frame a worker received leaked.
 *
 * Exit: 0 on success, 1 on a leak or a field the free left set.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sanitizer/lsan_interface.h>

#include "tp/pulsar_tp.h"

/// Mismatches seen so far.
static int g_failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            g_failures++;                                             \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

/// Every frame kind pulsar_tp_recv_command parses into a command.
static const pulsar_tp_frame_type k_kinds[] = {
    PULSAR_TP_FRAME_SYNC, PULSAR_TP_FRAME_VERIFY, PULSAR_TP_FRAME_NOTE_COMMITTED,
    PULSAR_TP_FRAME_NOTE_PREFILLED, PULSAR_TP_FRAME_SPEC_NEXT_BASE, PULSAR_TP_FRAME_SPEC_ROUND_BEGIN,
    PULSAR_TP_FRAME_SPEC_ARM_CAPTURE, PULSAR_TP_FRAME_SPEC_ROUND_END, PULSAR_TP_FRAME_SPEC_ROUND_ABORT,
    PULSAR_TP_FRAME_SPEC_REDRAFT_BATCH, PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT, PULSAR_TP_FRAME_VERIFY_PROFILE,
    PULSAR_TP_FRAME_GENERATE_SPECULATIVE, PULSAR_TP_FRAME_SPEC_ASSEMBLE_BATCH,
    PULSAR_TP_FRAME_SPEC_ROUND_END_BATCH, PULSAR_TP_FRAME_SPEC_REDRAFT_COMMIT_BATCH,
    PULSAR_TP_FRAME_SET_LOGITS, PULSAR_TP_FRAME_SYNC_MM, PULSAR_TP_FRAME_SYNC_CHECK,
    PULSAR_TP_FRAME_BANK_FREE_PHYSICAL, PULSAR_TP_FRAME_BANK_ALLOC_PHYSICAL, PULSAR_TP_FRAME_BANK_STATE_SAVE,
    PULSAR_TP_FRAME_BANK_STATE_RESTORE, PULSAR_TP_FRAME_BANK_REPOINT, PULSAR_TP_FRAME_SESSION_CREATE,
    PULSAR_TP_FRAME_REWIND, PULSAR_TP_FRAME_SEGMENT_SAVE, PULSAR_TP_FRAME_SEGMENT_LOAD,
    PULSAR_TP_FRAME_SEGMENT_DROP, PULSAR_TP_FRAME_SEGMENT_RECONCILE, PULSAR_TP_FRAME_SESSION_DESTROY,
    PULSAR_TP_FRAME_INVALIDATE, PULSAR_TP_FRAME_EVAL, PULSAR_TP_FRAME_EVAL_BATCH,
    PULSAR_TP_FRAME_MIXED_BATCH, PULSAR_TP_FRAME_FUSED_BATCH, PULSAR_TP_FRAME_STOP,
};

/// A command of `kind` with every malloc'd array set, as the widest frames fill them.
/// @param kind the frame kind
/// @return the command (each array malloc'd; release with pulsar_tp_command_free)
static pulsar_tp_command full_command(pulsar_tp_frame_type kind) {
    const uint32_t n = 3;
    pulsar_tp_command c;
    std::memset(&c, 0, sizeof(c));
    c.type = kind;
    c.session_id = 99;
    c.tokens = static_cast<int *>(std::malloc(n * sizeof(*c.tokens)));
    c.n_tokens = n;
    c.items = static_cast<pulsar_tp_batch_item *>(std::calloc(n, sizeof(*c.items)));
    c.n_items = n;
    c.logits = static_cast<float *>(std::malloc(n * sizeof(*c.logits)));
    c.n_logits = n;
    c.spec.count = n;
    c.spec_banks = static_cast<uint32_t *>(std::malloc(n * sizeof(*c.spec_banks)));
    c.spec_rngs = static_cast<uint64_t *>(std::malloc(n * sizeof(*c.spec_rngs)));
    c.spec_steps = static_cast<pulsar_tp_spec_command *>(std::calloc(n, sizeof(*c.spec_steps)));
    c.images = static_cast<pulsar_image_ref *>(std::calloc(n, sizeof(*c.images)));
    c.n_images = n;
    c.image_bytes = static_cast<uint8_t *>(std::malloc(64));
    for (uint32_t i = 0; i < n; i++) c.images[i].bytes = c.image_bytes + 16u * i;
    c.keys = static_cast<char *>(std::malloc(n * 40u + 1u));
    c.n_keys = n;
    return c;
}

int main(void) {
    for (const pulsar_tp_frame_type kind : k_kinds) {
        pulsar_tp_command c = full_command(kind);
        pulsar_tp_command_free(&c);
        CHECK(c.type == PULSAR_TP_FRAME_ERROR, "kind %d: type %d after free", (int)kind, (int)c.type);
        CHECK(!c.tokens && !c.items && !c.logits && !c.spec_banks && !c.spec_rngs && !c.spec_steps &&
                  !c.images && !c.image_bytes && !c.keys,
              "kind %d: a pointer survived the free", (int)kind);
        CHECK(!c.n_tokens && !c.n_items && !c.n_logits && !c.n_images && !c.n_keys && !c.spec.count &&
                  !c.session_id,
              "kind %d: a count survived the free", (int)kind);
        /* Per kind, so a leak names the kind that left it. */
        CHECK(__lsan_do_recoverable_leak_check() == 0, "kind %d: the free leaked (report above)", (int)kind);
    }
    pulsar_tp_command_free(NULL);   /* NULL is a no-op */
    if (g_failures) {
        std::fprintf(stderr, "tp_command_free_test: FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("tp_command_free_test: ok (%zu frame kinds, every array set, no leak under LeakSanitizer)\n",
                sizeof(k_kinds) / sizeof(k_kinds[0]));
    return 0;
}

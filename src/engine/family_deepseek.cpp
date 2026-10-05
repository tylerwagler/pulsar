/* family_deepseek.cpp -- the DeepSeek V4 family descriptor (L251 S1).
 *
 * The first family: DeepSeek V4 Flash (0731), Vision-Exp and V4.1 Flash, one
 * architecture with two shape profiles (shape_profiles.cpp).  Every entry
 * points at the engine as it stood at L250 -- the load and device steps moved
 * out of pulsar_engine::open, the session create/destroy/price bodies, and the
 * pulsar_session methods -- so a DeepSeek lane runs the same code, in the same
 * order, as before the interface existed.  The graph (pulsar_session::graph)
 * is this family's session state. */
#include "pulsar_engine_internal.h"

static uint32_t ds4_logits_width(const pulsar_engine *) { return PULSAR_N_VOCAB; }

static void ds4_tp_shape(const pulsar_engine *, uint32_t *n_layer, uint32_t *n_embd, uint32_t *n_vocab) {
    *n_layer = PULSAR_N_LAYER;
    *n_embd = PULSAR_N_EMBD;
    *n_vocab = PULSAR_N_VOCAB;
}

static const char *ds4_model_name(const pulsar_engine *) { return PULSAR_MODEL_SHAPE_NAME; }

/* The conversation format follows the profile: 0731 has its own template,
 * Vision-Exp and V4.1 share one (pulsar_shape::variant). */
static pulsar_chat_format ds4_chat_format(const pulsar_engine *) {
    return g_pulsar_shape.variant == PULSAR_VARIANT_V4 ? PULSAR_CHAT_DS4_V4 : PULSAR_CHAT_DS4_V41;
}

/* 0 and 1 are the two profiles' ids since before families existed; disk-KV
 * files carry them. */
static int ds4_model_id(const pulsar_engine *) { return (int)PULSAR_MODEL_VARIANT; }

static int ds4_sync(pulsar_session *s, const pulsar_tokens *prompt,
                    const pulsar_image_ref *images, int n_images, char *err, size_t errlen) {
    return s->sync(prompt, images, n_images, err, errlen);
}

static int ds4_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    return s->eval(token, err, errlen);
}

static int ds4_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n,
                               float *logits, int logits_cap, char *err, size_t errlen) {
    return s->decode_multiseq(reqs, n, logits, logits_cap, err, errlen);
}

static int ds4_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows,
                            float *logits, int logits_cap, uint32_t *out_n_rows,
                            uint32_t max_head_runs, char *err, size_t errlen) {
    return s->decode_mixed(reqs, n_rows, logits, logits_cap, out_n_rows, max_head_runs, err, errlen);
}

static void ds4_invalidate(pulsar_session *s) { s->invalidate(); }

static const pulsar_family_session_ops k_ds4_session_ops = {
    /* .create          = */ pulsar_ds4_session_create,
    /* .destroy         = */ pulsar_ds4_session_destroy,
    /* .cost_bytes      = */ pulsar_ds4_session_cost_bytes,
    /* .sync            = */ ds4_sync,
    /* .eval            = */ ds4_eval,
    /* .decode_multiseq = */ ds4_decode_multiseq,
    /* .decode_mixed    = */ ds4_decode_mixed,
    /* .invalidate      = */ ds4_invalidate,
};

const pulsar_family PULSAR_FAMILY_DEEPSEEK4 = {
    /* .id           = */ PULSAR_FAMILY_ID_DEEPSEEK4,
    /* .arch         = */ "deepseek4",
    /* .name         = */ "DeepSeek V4",
    /* .drafter      = */ PULSAR_DRAFTER_DSPARK,
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SPEC | PULSAR_FAMILY_CAP_PAYLOAD |
                          PULSAR_FAMILY_CAP_REWIND | PULSAR_FAMILY_CAP_VISION | PULSAR_FAMILY_CAP_TP |
                          PULSAR_FAMILY_CAP_IMATRIX | PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_GENERATE |
                          PULSAR_FAMILY_CAP_SEGMENTS,
    /* .load         = */ pulsar_ds4_family_load,
    /* .after_gpu    = */ pulsar_ds4_family_after_gpu,
    /* .logits_width = */ ds4_logits_width,
    /* .model_name   = */ ds4_model_name,
    /* .chat_format  = */ ds4_chat_format,
    /* .model_id     = */ ds4_model_id,
    /* .tp_shape     = */ ds4_tp_shape,
    /* .session      = */ &k_ds4_session_ops,
};

/* qwen_forward.h -- the Qwen MTP layer's forward (family_qwen.cpp), for its drafter (spec_qwen.cpp, L272 P1).  The
 * trunk's forward is the core driver's (step_driver.cpp, L295 F4).  Include after pulsar_engine_internal.h and
 * family_qwen.h.
 *
 * qwen_mtp_forward: the MTP layer over the trunk stacks staged in mtp_h, `head_n` rows through the draft head
 * (n_draft logits a row); `decode_rows` (L288) takes a PREFILL step's rows -- runs of any number of banks -- as
 * decode rows (the speculation's absorb).  With `pick` (L289; L288: pick[head_n], every headed row) each head row's
 * greedy draft is reduced on the device -- its vocabulary id and softmax probability, the draft's confidence -- and
 * logits_out (NULL allowed) is read back only when a caller needs the rows themselves (a sampled draft's q);
 * qwen_mtp_dist: the sampled q over the draft vocabulary, ids mapped. */
#ifndef PULSAR_QWEN_FORWARD_H
#define PULSAR_QWEN_FORWARD_H

#include "pulsar_engine_internal.h"
#include "family_qwen.h"

/** A Qwen session's state: the driver's buffers are its base (L295 F4). */
static inline pulsar_qwen_state *qwen_st(const pulsar_session *s) {
    return static_cast<pulsar_qwen_state *>(s->driver);
}
/** A draft row's greedy pick (L289): the vocabulary id of the draft head's first maximum and its softmax
 *  probability over the draft head. */
struct qwen_mtp_pick {
    int32_t id = -1;
    float conf = 0.0f;
};
bool qwen_mtp_forward(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                      const int32_t *bank, uint32_t n_rows, uint32_t head_row0, uint32_t head_n, float *logits_out,
                      qwen_mtp_pick *pick = NULL, bool decode_rows = false);
bool qwen_mtp_dist(pulsar_session *s, const float *row, float temperature, int top_k, float top_p, float min_p,
                   pulsar_sample_dist *q);

#endif /* PULSAR_QWEN_FORWARD_H */

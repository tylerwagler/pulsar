/* qwen_forward.h -- the Qwen trunk's and MTP layer's forwards (family_qwen.cpp), for the speculation seam
 * (spec_qwen.cpp, L272 P1).  Include after pulsar_engine_internal.h and family_qwen.h.
 *
 * qwen_forward: one step of the trunk over `n_rows` rows (DECODE: one row per bank; PREFILL: runs of one bank
 * each -- a verify's [0, n_verify), then the fused step's prompt runs), the `heads` rows headed into `logits_out`
 * (host, n_vocab a row); n_verify > 0 arms the per-row state capture pulsar_qwen_s4_spec_rollback reads over
 * the verify rows.  qwen_mtp_forward: the MTP layer over the
 * trunk stacks staged in mtp_h, `head_n` rows through the draft head (n_draft logits a row).
 * qwen_mtp_argmax: the draft-vocabulary argmax of one MTP row, mapped to the vocabulary (*prob: its
 * softmax, the draft's confidence); qwen_mtp_dist: the sampled q over the draft vocabulary, ids mapped;
 * qwen_argmax: the argmax of a row of n floats.  qwen_verify_rows: a verify's rows (one run a bank, each at its bank's
 * next position) as the forward's token / position / bank arrays, false with `err` naming the first row that is
 * not. */
#ifndef PULSAR_QWEN_FORWARD_H
#define PULSAR_QWEN_FORWARD_H

#include "pulsar_engine_internal.h"
#include "family_qwen.h"

/** The step rows a forward heads, in logits-row order: logits row i is step row row[i] (L284 #2: a fused step
 *  heads its verify rows, then its prompt runs' last rows).  n > PULSAR_QWEN_HEAD_ROWS_MAX is refused. */
struct qwen_heads {
    uint32_t n = 0;
    uint32_t row[PULSAR_QWEN_HEAD_ROWS_MAX] = {};
};
/** Step rows [row0, row0 + n). */
qwen_heads qwen_heads_span(uint32_t row0, uint32_t n);
bool qwen_forward(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens, const int32_t *pos,
                  const int32_t *bank, uint32_t n_rows, const qwen_heads &heads, float *logits_out,
                  uint32_t n_verify = 0);
bool qwen_mtp_forward(pulsar_session *s, pulsar_qwen_step_mode mode, const int32_t *tokens, const int32_t *pos,
                      const int32_t *bank, uint32_t n_rows, uint32_t head_row0, uint32_t head_n, float *logits_out);
int32_t qwen_mtp_argmax(const pulsar_engine *e, const float *row, float *prob = NULL);
bool qwen_mtp_dist(pulsar_session *s, const float *row, float temperature, int top_k, float top_p, float min_p,
                   pulsar_sample_dist *q);
uint32_t qwen_argmax(const float *v, uint32_t n);
bool qwen_verify_rows(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, int32_t *tok, int32_t *pos,
                      int32_t *bk, char *err, size_t errlen);

#endif /* PULSAR_QWEN_FORWARD_H */

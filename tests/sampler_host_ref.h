/* sampler_host_ref.h -- the frozen host sampler (sampler_host_ref.cpp): the device sampler's reference. */
#ifndef PULSAR_SAMPLER_HOST_REF_H
#define PULSAR_SAMPLER_HOST_REF_H

#include <stdint.h>

namespace sref {

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

typedef struct {
    sample_candidate *cand;
    uint64_t *keys;
    uint64_t *tmp;
    sample_candidate *cand2;
    uint32_t cap;
} pulsar_sample_scratch;

typedef struct {
    int *ids;
    float *probs;
    uint32_t n;
} pulsar_sample_dist;

typedef struct {
    int id;
    float logit;
    float logprob;
} pulsar_token_score;

int sample_argmax(const float *logits, uint32_t n_vocab);
uint64_t pulsar_sample_key(uint64_t rng, uint32_t ahead);
void pulsar_sample_advance(uint64_t *rng, uint32_t n);
void pulsar_sample_scratch_free(pulsar_sample_scratch *s);
int pulsar_sample_dist_build(const float *logits, uint32_t n_vocab, float temperature, int top_k, float top_p,
                             float min_p, pulsar_sample_scratch *scratch, pulsar_sample_dist *out);
int pulsar_sample_dist_build_prefiltered(const int32_t *ids, const float *vals, uint32_t n_cand, uint32_t n_vocab,
                                         float max_logit, float temperature, float min_p,
                                         pulsar_sample_scratch *scratch, pulsar_sample_dist *out);
void pulsar_sample_dist_free(pulsar_sample_dist *d);
float pulsar_sample_dist_prob(const pulsar_sample_dist *d, int token);
int pulsar_sample_dist_draw(const pulsar_sample_dist *d, uint64_t key);
int sample_top_p_min_p(const float *logits, uint32_t n_vocab, float temperature, int top_k, float top_p, float min_p,
                       uint64_t key, pulsar_sample_scratch *scratch);
int pulsar_logits_top_logprobs(const float *logits, int n_vocab, pulsar_token_score *out, int k);
int pulsar_logits_token_logprob(const float *logits, int n_vocab, int token, pulsar_token_score *out);

}  // namespace sref

#endif

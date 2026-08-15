#include "pulsar_engine_internal.h"



uint32_t pulsar_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = PULSAR_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}



uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    if (requested_chunk != 0) {
        cap = requested_chunk;
    } else {
        const char *env = getenv("PULSAR_CUDA_PREFILL_CHUNK");
        if (env && env[0]) {
            char *endp = NULL;
            const long v = strtol(env, &endp, 10);
            if (endp != env) {
                if (v <= 0) return cap;
                cap = (uint32_t)v;
            }
        } else if (prompt_len > 4096) {
            cap = 4096u;
        }
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}









/* Collapse final HC streams into the ordinary embedding vector before the
 * output norm and vocabulary projection. */
static void output_hc_head_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc) {
    const uint32_t n_hc = PULSAR_N_HC;
    const uint64_t hc_dim = (uint64_t)PULSAR_N_EMBD * n_hc;
    float *flat = (float *)xmalloc((size_t)hc_dim * sizeof(flat[0]));
    float *pre = (float *)xmalloc((size_t)n_hc * sizeof(pre[0]));
    float *w = (float *)xmalloc((size_t)n_hc * sizeof(w[0]));

    rms_norm_no_weight(flat, inp_hc, hc_dim, PULSAR_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);

    const float *scale = (const float *)tensor_data(model, weights->output_hc_scale);
    const float *base = (const float *)tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + PULSAR_HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, w, PULSAR_N_EMBD, n_hc);

    free(w);
    free(pre);
    free(flat);
}



/* Final language-model head: HC collapse, RMSNorm, and Q8_0 vocab projection. */
void output_logits_one(
        float             * logits,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc) {
    float *embd = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(embd[0]));
    float *norm = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(norm[0]));

    output_hc_head_one(embd, model, weights, inp_hc);
    rms_norm_weight(norm, embd, (const float *)tensor_data(model, weights->output_norm), PULSAR_N_EMBD, PULSAR_RMS_EPS);

    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
    free(embd);
}



int sample_argmax(const float *logits, uint32_t n_vocab);



/* =========================================================================
 * GPU Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the GPU executor.
 */

float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}


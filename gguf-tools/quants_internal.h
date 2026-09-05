/*
 * quants_internal.h — shared internal header for the quants_*.c TUs.
 * Carries the shared prologue (macros, numeric helpers, K-quant search
 * kernels) and the per-format entry points the dispatch tail in
 * quants_common.c calls across TU boundaries. Split from quants.c.
 */
#ifndef QUANTS_INTERNAL_H
#define QUANTS_INTERNAL_H

#include "quants.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256
#define DS4Q_GROUP_MAX_EPS 1e-15f
#define DS4Q_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DS4Q_MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    const char *name;
    int64_t block_size;
    size_t type_size;
    bool can_quantize;
    bool requires_imatrix;  /* cannot be quantized without one */
    bool uses_imatrix;      /* the quantizer reads one when given (superset of requires) */
} ds4q_traits;

extern const ds4q_traits ds4q_type_traits[DS4Q_TYPE_COUNT];

/* quants_common.c — numeric helpers + K-quant search kernels */
extern pthread_mutex_t ds4q_init_mutex;
float ds4q_f32_from_bits(uint32_t bits);
uint32_t ds4q_f32_to_bits(float f);
uint16_t ds4q_f32_to_f16(float f);
int ds4q_nearest_int(float fval);
float ds4q_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                            uint8_t *L, float *the_min, uint8_t *Laux,
                            float rmin, float rdelta, int nstep, bool use_mad);
float ds4q_make_qkx3_quants(int n, int nmax, const float *x, const float *weights,
                            uint8_t *L, float *the_min, uint8_t *Laux,
                            float rmin, float rdelta, int nstep, bool use_mad);
float ds4q_make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights);

/* quants_fp.c — MXFP8/MXFP4 codecs */
size_t ds4q_quantize_fp8_e4m3(const float *src, void *dst, int64_t start,
                              int64_t nrows, int64_t ncols);
size_t ds4q_quantize_mxfp4(const float *src, void *dst, int64_t start,
                           int64_t nrows, int64_t ncols);

/* quants_kquants.c — Q2_K + IQ2_XXS */
size_t ds4q_quantize_q2_k(const float *src, void *dst, int64_t start,
                          int64_t nrows, int64_t ncols, const float *quant_weights);
size_t ds4q_quantize_iq2_xxs(const float *src, void *dst, int64_t start,
                             int64_t nrows, int64_t ncols, const float *quant_weights);
void ds4q_iq2_xxs_init(void);

#endif /* QUANTS_INTERNAL_H */

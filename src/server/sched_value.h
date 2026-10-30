/* L312: the fused lane's value-per-time round selection (sched_value.cpp has the rule and its rationale).
 *
 * A selection layer only: it reads the engine's two measured cost authorities (the L304 / L327 prompt chunk cost fit,
 * the L301 verify curve) and the riders' service prices, and says which prompt chunks the next spec round carries.  It knows
 * nothing of slots, banks or routing; server_sched.cpp's spec quantum gathers its inputs and applies its choice. */
/** @file
 *  The fused lane's value-per-time round selection (L312), the interface server_sched.cpp calls.
 */
#ifndef PULSAR_SCHED_VALUE_H
#define PULSAR_SCHED_VALUE_H

#include "pulsar.h"

#include <stdint.h>

/** The most prompts one round may weigh (one chunk each), = PULSAR_SESSION_POOL_CAP (asserted where it is used). */
#define SCHED_PROMPTS_MAX 16

/** A non-finishing chunk ends on a multiple of this (L260: the widest compressor group, so a prompt's chunking and
 *  its numerics never depend on the load beside it). */
#define SCHED_CHUNK_ALIGN 128u

/** The preferences the selection maximises (sched_value.cpp says what each one means).  Value is counted in prompt
 *  tokens: one prompt token at its base price is worth 1. */
typedef struct {
    double out_value;   ///< an output token's value (decode over prefill: Tyler 2026-10-15)
    double fairness;    ///< token fairness f, value per ms: a request's price is its base + f w / (its service rate)
    double prompt_weight;   ///< the prefilling prompts' weight as a class (the decoding streams' is 1)
    double age_price;   ///< c, value per ms^2 per decoding stream: what holding a stream's next token costs
    double adapt_ms;    ///< tau: how fast a service price follows the service it receives
} sched_prefs;

/** One prefilling prompt the round may carry. */
typedef struct {
    uint32_t p0;          ///< its bank frontier (rows already computed)
    uint32_t len;         ///< its prompt length
    uint32_t grid;        ///< the resume grid: a chunk reaching `len` is a candidate only with no grid point inside it
    double inv_ms;        ///< its service price over f (ms per token, sched_service_observe); <= 0: not yet priced
    uint64_t last_ride;   ///< when it last rode (any monotone clock): the unpriced rule serves the oldest first
} sched_prompt;

/** The decode side of the round (every live decoder rides every round, one base row each). */
typedef struct {
    uint32_t n_live;      ///< decoding banks
    uint32_t n_dec;       ///< every bank of the quantum (a finishing chunk needs a head row beyond these)
    uint32_t heads_max;   ///< the family's head rows a step may carry
    uint32_t rows;        ///< the decode-only round's verify rows (the verify pick's)
    double tokens[SCHED_PROMPTS_MAX];   ///< each live decoder's expected tokens this round (1 + its verified survivals)
    double inv_ms[SCHED_PROMPTS_MAX];   ///< each live decoder's service price over f (<= 0: not yet priced)
    double age_ms;        ///< the decoders' summed time since their last output
} sched_round;

/** One round's selection (sched_choose's output), indexed like its `pr` array. */
typedef struct {
    uint32_t q[SCHED_PROMPTS_MAX];   ///< rows each prompt rides (0: it waits)
    bool fin[SCHED_PROMPTS_MAX];     ///< its chunk finishes it (a head row)
    bool priced;                     ///< the costs were measured (else the unpriced rule chose)
    double t_ms;                     ///< the chosen round's predicted wall
    double t_dec_ms;                 ///< the decode-only round's predicted wall
    double score, score_dec;         ///< their value per time, net of the decoders' waiting
} sched_choice;
/** @var sched_choice::score
 *  The chosen round's score (value in prompt tokens per ms of wall, net of the decoders' waiting); 0 unpriced. */

/** The choice for one round.  room_rows bounds the round's prompt rows (the fused step's widest run);
 *  first_full_round: this is the first round of a full decode quantum (the unpriced rule's only use of it);
 *  probe_rows: the continuation size while the costs are unmeasured. */
void sched_choose(const pulsar_engine *e, const sched_prefs *p, const sched_round *r, const sched_prompt *pr,
                  int n, uint32_t room_rows, bool first_full_round, uint32_t probe_rows, sched_choice *out);

/** A prompt's individually feasible service rate, tokens per ms: its widest chunk alone on the measured costs
 *  (0: unmeasured).  What its price adapts against. */
double sched_prompt_rate_max(const pulsar_engine *e, const sched_round *r, const sched_prompt *pr,
                             uint32_t room_rows);
/** A decoder's: `tokens` expected from a round of its own `rows` rows on the verify curve (0: no curve). */
double sched_decode_rate_max(const pulsar_engine *e, double tokens, uint32_t rows);

/** Flare's service price, held over one round of `t_ms` in which the request received `tokens` (0: none) of the
 *  `max_rate` it could have: dp/dt = (f - p y) / (m tau) integrated exactly for the held action (inv = p / f). */
void sched_service_observe(double *inv_ms, double t_ms, double tokens, double max_rate, double adapt_ms);

#endif

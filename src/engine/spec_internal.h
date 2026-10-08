/* spec_internal.h -- the round API's private types, shared by the core (session_spec.cpp) and the
 * target / drafter implementations (spec_dspark.cpp; Qwen's in L272 P1 S3).  Not a standalone header:
 * include after pulsar_engine_internal.h. */
#ifndef PULSAR_SPEC_INTERNAL_H
#define PULSAR_SPEC_INTERNAL_H

#include "pulsar_engine_internal.h"
#include "spec_ops.h"

/* L150: what a bank's redraft needs, recorded by round_end in the batched lane
 * (which defers drafting), consumed by pulsar_session_spec_redraft_batch. */
typedef struct {
    bool     valid;         ///< round_end reached the drafting point with the drafter's features ready
    int32_t  next_base;     ///< the carry: batch position 0 of the next round
    uint32_t n_draft;       ///< the bank's depth at round_end (pulsar_spec_cur_depth), <= 16
    uint32_t n_batch;
    int      commit;
    float    temperature, top_p, min_p;
    int      top_k;
    /* results, filled by the batched redraft; committed into the bank's shadow
     * by pulsar_session_spec_redraft_commit under the server's bank switch */
    bool     done;
    bool     sample_drafts;
    uint32_t keep;
    bool     have_conf;
    int32_t  refined[17];
    float    conf[16];
    float    q_drawn[16];                                  ///< q(pend[i]) per position (sampled)
    uint32_t qn[16];                                       ///< stored proposal dist size (0 = row fallback)
    int32_t  qids[16][PULSAR_DSPARK_QDIST_CAP];
    float    qprobs[16][PULSAR_DSPARK_QDIST_CAP];
    float   *qrows;                                        ///< [16][logits width] fallback rows, lazily owned
} spec_redraft_req;

typedef struct pulsar_spec_round {
    int32_t base;           ///< the round's base token (verify row 0, at saved_len)
    uint32_t K;             ///< draft depth this round: tokens the drafter proposed
    uint32_t n_batch;       ///< rows the verify forward evaluates (K + the base row)
    int saved_len;          ///< checkpoint length before the round, to trim back to on rejection
    bool pend_sampled;      ///< the pendings below were sampled rather than left from a prior round
    int32_t pend[16];       ///< the drafted token ids, in draft order
    float pend_conf[16];    ///< confidence head's score per draft position
    int row_tops[16];       ///< the target's argmax per verify row; the accept comparison
    /** The target's state snapshot taken before the verify (pulsar_spec_target_ops snapshot / restore /
     * release own its contents; malloc'd, freed with the round), so a partial accept can roll back. */
    void *snap;
    spec_redraft_req redraft;   ///< L150: batched-lane deferred redraft (see the typedef)
} pulsar_spec_round;

/** Round-local row i (i = 0 the base row, i = j the row that judges draft j) -> its full target logits. */
typedef bool (*pulsar_spec_row_read_fn)(void *ud, uint32_t row, float *out);

/** How a round reads its verify rows: a reader plus what it reads from -- the caller's logits block at
 * the round's offset (the core's reader) or the target's own device rows (a target reader). */
typedef struct pulsar_spec_rows {
    pulsar_spec_row_read_fn read;
    void *ud;               ///< what `read` takes: `&block` or `&target`
    /** Compact candidate rows for the walk (PULSAR_DSPARK_PREFILTER_ROW_I32 ints a row, indexed by the
     *  row's ABSOLUTE index in the forward), or NULL = the walk builds from full rows. */
    const int32_t *compact;
    struct { const float *rows; uint32_t row0; uint32_t vocab; } block;   ///< the core's reader
    struct { void *g; uint32_t row0; } target;                            ///< a target reader's handle
} pulsar_spec_rows;

/* ---- the core's helpers the implementations use ---- */
/** The width of every host logits row the round API reads or stores (the engine's logits width). */
static inline uint32_t spec_vocab(const pulsar_session *s) { return (uint32_t)s->engine->logits_width(); }
/** The session's current draft depth (the adaptive controller's, else the starting depth), 1..16. */
uint32_t pulsar_spec_cur_depth(const pulsar_session *s);
/** The draft stop threshold in force: --spec-tau / PULSAR_SPEC_TAU, else the drafter's; 0 = no stop. */
float pulsar_spec_tau(const pulsar_engine *e);
/** THE stop rule (spec_ops.h): a draft whose confidence is under tau is its chain's last (tau <= 0: none is). */
static inline bool pulsar_spec_conf_stops(float conf, float tau) { return tau > 0.0f && conf < tau; }
/** The stop rule over a drafted chain: how many of its n drafts are kept -- up to and including the first that
 *  stops it. */
static inline uint32_t pulsar_spec_conf_keep(const float *conf, uint32_t n, float tau) {
    for (uint32_t k = 0; k < n; k++)
        if (pulsar_spec_conf_stops(conf[k], tau)) return k + 1u;
    return n;
}
/** A draft record for the installed bank's next chain after `next_base` at the session's depth and these
 *  params -- what a drafter's draft / draft_batch fills (the rest of the record is the caller's). */
void pulsar_spec_draft_req_init(pulsar_session *s, spec_redraft_req *q, int next_base, float temperature, int top_k,
                                float top_p, float min_p);
/** Keep sampled draft `pos`'s proposal q for the walk's residual: its support when it fits the compact
 *  store (true), else false and q->qn[pos] = 0 -- the caller keeps q's full row at q->qrows + pos * vocab,
 *  from which the walk rebuilds q under the record's params. */
bool pulsar_spec_q_record(spec_redraft_req *q, uint32_t pos, const pulsar_sample_dist *qd);
/** The core's reader over `rows->block`. */
bool pulsar_spec_row_read_block(void *ud, uint32_t row, float *out);
/** Stamp a drafter's result record into the installed bank's pendings (the one way pendings are written). */
void pulsar_spec_redraft_stamp(pulsar_session *s, spec_redraft_req *q);
/** `bank`'s speculative shadow for a read: the live state for the installed bank, the saved carry's for
 *  another, NULL when nothing is saved. */
const pulsar_spec_carry_state *pulsar_spec_bank_shadow(pulsar_session *s, uint32_t bank);
/** The adaptive draft controller's CURRENT depth for `bank` (0 = no drafter or nothing valid). */
int pulsar_spec_bank_depth(pulsar_session *s, uint32_t bank);
/** A redraft record's lazily-owned full q rows ([16][vocab]); false = out of memory. */
bool pulsar_spec_redraft_qrows_reserve(spec_redraft_req *q, uint32_t vocab);
/** Save / restore the session's speculative shadow (s->spec and the q rows its sampled pendings read)
 *  into / from a bank carry's copy -- every family's bank save and restore call these. */
void pulsar_spec_shadow_save(pulsar_session *s, pulsar_spec_carry_state *dst, float **dst_rows, uint32_t *dst_cap);
void pulsar_spec_shadow_restore(pulsar_session *s, const pulsar_spec_carry_state *src, const float *src_rows,
                                uint32_t src_cap);
static inline const pulsar_spec_target_ops *spec_target(const pulsar_session *s) { return s->engine->family->spec; }
static inline const pulsar_drafter_ops *spec_drafter(const pulsar_session *s) { return s->engine->drafter_ops; }
/** The bank save / restore the batched lane runs per step: the family's bank ops or DeepSeek's members. */
static inline bool spec_bank_restore(pulsar_session *s, uint32_t bank) {
    return FAMILY_BANKS(s) ? FAMILY_BANKS(s)->restore(s, bank) : s->bank_state_restore(bank);
}
static inline void spec_bank_save(pulsar_session *s, uint32_t bank) {
    if (FAMILY_BANKS(s)) FAMILY_BANKS(s)->save(s, bank); else s->bank_state_save(bank);
}

/* ---- the implementations ---- */
extern const pulsar_spec_target_ops k_ds4_spec_target;   ///< DeepSeek V4's verify hooks (spec_dspark.cpp)
extern const pulsar_drafter_ops k_dspark_drafter;        ///< the DSpark drafter (spec_dspark.cpp)
extern const pulsar_spec_target_ops k_qwen_spec_target;  ///< Qwen4-exp's verify hooks (spec_qwen.cpp)
extern const pulsar_drafter_ops k_mtp_drafter;           ///< Qwen's MTP drafter (spec_qwen.cpp)

#endif /* PULSAR_SPEC_INTERNAL_H */

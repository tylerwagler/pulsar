/* spec_ops.h -- the speculation seam (L272 P1 S2): what a TARGET family provides so any drafter can run
 * against its forward, and what a DRAFTER provides -- two plugins, one round API (session_spec.cpp) and
 * one server lane between them.
 *
 * The drafter is not the family.  A target family can carry several drafters (Qwen3.8 ships an MTP
 * head and has DSpark and DFlash2 drafters trained for it; a Flash-Next DFlash has been served on two
 * DGX Sparks), so the family's part is only the verify side of the round -- snapshot and roll back the
 * state a verify run mutates, arm the forward's per-row captures and its logits readback, read a
 * bank's rows out of the shared forward, leave the bank's state at the committed position, and cut the
 * history on the L155 trim -- and the drafter's part is its own weights and state: prime its context
 * from the prompt's captured features, absorb a committed row, draft a chain of up to K ids with
 * per-position confidences (sampled drafts with their proposal q recorded), finish a deferred device
 * draft, and the banked forms of those.  The core owns the round object, the per-bank pendings / carry /
 * depth / quench / counters, the accept walk, the EOS clamp, the emit, the L155 trim decision and the
 * batch plumbing (assemble / round_end / redraft, one step per bank, one verdict per phase on a pair).
 *
 * A verify row carries its position in the chain today (parent = row - 1); a tree verifier (DDTree,
 * TreeSpark) would add ancestor masks to the target's attention and per-branch recurrent state, and
 * that is a target capability this seam does not declare. */
#ifndef PULSAR_SPEC_OPS_H
#define PULSAR_SPEC_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "spec_depth.h"

struct pulsar_session;
struct pulsar_engine;
struct pulsar_spec_round;
struct pulsar_spec_rows;

/** The TARGET family's verify hooks.  A hook marked optional may be NULL: the core then does nothing
 *  there (a family whose verify captures its own undo state inside the forward has no snapshot). */
typedef struct pulsar_spec_target_ops {
    /** Optional.  Snapshot, on the installed bank, the state verify rows mutate and position-addressed
     *  KV cannot undo (DeepSeek: the compressor and indexer pending groups and the n_comp counters).  The
     *  round owns the handle (`pulsar_spec_round::snap`, malloc'd; the core frees it with the round).
     *  false with `err` = refuse the round. */
    bool (*snapshot)(struct pulsar_session *s, struct pulsar_spec_round *r, char *err, size_t errlen);
    /** Optional.  Put the snapshot back on the installed bank (an abort, a failed walk). */
    bool (*restore)(struct pulsar_session *s, struct pulsar_spec_round *r);
    /** Optional.  The snapshot is no longer needed (the round ended or aborted). */
    void (*release)(struct pulsar_session *s, struct pulsar_spec_round *r);
    /** A round begun with these sampling params will ride the next shared forward: the target may
     *  choose a cheaper logits readback for the step (DeepSeek: per-row argmaxes when every round is
     *  greedy, compact min-p candidates when every round is in the sparse contract).  Optional. */
    void (*round_note)(struct pulsar_session *s, float temperature, int top_k, float top_p, float min_p);
    /** Optional.  Before the shared verify forward over n_rows rows (0 = disarm after it): keep each
     *  row's drafter features and state inputs, and select the logits readback the rounds noted.  A
     *  family whose decode_mixed treats a heads-on-every-row step as the verify needs none. */
    void (*arm_verify)(struct pulsar_session *s, uint32_t n_rows);
    /** The single lane's verify: ONE forward over the round's rows [base, pend...] at saved_len on the
     *  installed bank, features and state inputs kept, r->row_tops filled for the K drafts, and `out`
     *  set to read the rows.  false with `err` = the round is undone by the core. */
    bool (*verify_single)(struct pulsar_session *s, struct pulsar_spec_round *r, struct pulsar_spec_rows *out,
                          char *err, size_t errlen);
    /** Optional.  The batched lane: this round's rows [row0, row0 + n_batch) of the last shared
     *  forward, whose caller-visible logits block is `block` (rows the target read back another way are
     *  not in it): fill r->row_tops for the K drafts and set `out` to read the rows.  NULL = the block
     *  holds every row: the core takes the argmaxes from it and reads it. */
    bool (*verify_rows)(struct pulsar_session *s, struct pulsar_spec_round *r, const float *block, uint32_t row0,
                        struct pulsar_spec_rows *out, char *err, size_t errlen);
    /** The walk accepted `commit` of the K drafts: the verify advanced the installed bank's state by all
     *  n_batch rows; leave it at saved_len + 1 + commit -- nothing to do on a full accept (DeepSeek), a
     *  snapshot restore plus a roll-forward of the recurrent state from the inputs saved during the
     *  verify on a partial one; a recurrent family also moves its position counters here. */
    bool (*commit)(struct pulsar_session *s, struct pulsar_spec_round *r, uint32_t commit, uint32_t row0);
    /** The L155 trim: the bank's history ends at `pos`, below what the round committed (DeepSeek: rewind;
     *  a recurrent family invalidates or restores a grid checkpoint). */
    void (*cut)(struct pulsar_session *s, int pos);
    /** The adaptive depth controller's constants, or NULL = a fixed depth (the drafter's default). */
    const pulsar_spec_depth_policy *depth;
    /* The rows one shared verify forward carries -- and so the decoders it carries, one base row each -- are the
     * family's session fact pulsar_family_session_ops::fused_heads_max (L284), not a field here. */
} pulsar_spec_target_ops;

/** A DRAFTER: its own weights and per-bank context, behind the round API. */
typedef struct pulsar_drafter_ops {
    const char *name;
    /** The depth a fresh session drafts at (the adaptive controller moves it from here). */
    uint32_t (*depth_default)(const struct pulsar_engine *e);
    /** Optional.  A fresh context: prime it from the prompt's captured features before the first draft.
     *  0 = done or nothing to do, -1 with `err` = refuse the round. */
    int (*prime)(struct pulsar_session *s, char *err, size_t errlen);
    /** Optional.  The installed bank's context advances by one committed row of the last verify (`row`
     *  is the row's absolute index in that forward's capture; `next` the token that follows it -- the
     *  next committed token, or the round's carry after the last one).  NULL = the drafter absorbs a
     *  round's rows together (absorb_banked), in the single lane too. */
    bool (*absorb)(struct pulsar_session *s, uint32_t row, int32_t next);
    /** Every deferred row in one pass, no bank switch: rows[i] of banks[i], followed by next[i]. */
    bool (*absorb_banked)(struct pulsar_session *s, const uint32_t *rows, const uint32_t *banks,
                          const int32_t *next, uint32_t n);
    /** Draft the installed bank's next chain after `next_base`: up to the session's depth, each draft with
     *  a confidence (or -1), sampled drafts (temperature > 0) drawn from the request-filtered proposal q
     *  with q(id) and q's support recorded -- stamped into the bank's pendings.  Returns the pending
     *  count kept.  `features_ready` = the drafter's conditioning features are those of the last
     *  committed row. */
    uint32_t (*draft)(struct pulsar_session *s, int next_base, bool features_ready, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
    /** Optional.  Finish a deferred device draft into the installed bank's pendings. */
    void (*harvest)(struct pulsar_session *s);
    /** N banks' drafts in one pass, no bank switch, each result into rounds[i]->redraft; byte-exact with
     *  `draft` for each bank.  0, or -1 with `err` on a device failure. */
    int (*draft_batch)(struct pulsar_session *s, struct pulsar_spec_round **rounds, const uint32_t *banks,
                       uint64_t **rngs, int n, char *err, size_t errlen);
} pulsar_drafter_ops;

#endif /* PULSAR_SPEC_OPS_H */

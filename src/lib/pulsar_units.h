/** @file
 *  Typed quantities that share a raw integer type but must never mix (L333, L334): KvRows, KvBytes, TokCount,
 *  PromptToks, TokPos, the P2b index spaces, TokId, BankId, TensorBytes, TextBytes, the layer index spaces (L436),
 *  the output path's text offsets (L437), slab, file and model-map offsets (L438), host and device memory (L442) */
#ifndef PULSAR_UNITS_H
#define PULSAR_UNITS_H

/* pulsar_units.h -- quantities that share a raw integer type but must never mix, as distinct types (L333, L334).
 *
 * L333, the KV accounting.  The guard, the admission verdicts, the memory floor and the families' touched / growth
 * prices once carried every quantity as a bare uint64_t, and whether a number was token rows or bytes lived only in
 * comments.  L332's VMM series switched the guard's "touched" from the rows high-water to mapped chunk bytes and
 * nothing complained; admission then took paths the managed run never took (a 500, a decoding bank counted as 0).
 *
 *   KvRows   token rows of KV: a bank's high-water, a prompt's length, a decode quantum, a page of the disk grid --
 *            what a family's pools grow by, one row per token.
 *   KvBytes  bytes of KV the KV accounting prices: a bank's touched pages, a growth price, the guard's budget and
 *            ceiling.  The memory readings and floors it is compared with are HostBytes / DeviceBytes since L442
 *            (that section, at the end of this file, says which memory a KV price lives in).
 *
 * L334 phase 1, the token seam's two currencies (pulsar.h pulsar_prefix_match, L115).  A client re-sends sampled
 * turns in canonical spelling, so across a seam the bank's live history and the request's prompt spend a different
 * NUMBER of tokens on the same bytes.  A count on one side is no count on the other; the two once shared `int` and
 * a comment that said mixing them was "the hazard this struct exists to make impossible".
 *
 *   TokCount    tokens of the live side: a length or distance on a bank's history (its KV rows, cut points) and on
 *               the stitched prompt a sync runs, which keeps the live tokens below the cut -- the distance of the
 *               live side's positions (TokPos, below).
 *   PromptToks  tokens of the request's prompt as the client's bytes tokenised (accounting, prefill bounds, the
 *               prompt tokens a bank's state serves).
 *   Both count from position 0.  The one conversion is the stitch's (pulsar_prefix_match_stitched_len): past the cut
 *   the stitched prompt carries the prompt's own tokens, so a prompt-side distance there is a live-side one.
 *
 * L334 phase 2a, positions vs counts (pulsar.h's position API: a session's and a bank's position, the prefill
 * frontier, rewind targets, grid checkpoints G, resume and adopt points, a batched row's position, a segment's span).
 * They were all `int`, and an int that meant "where" was added to, compared with and passed for an int that meant
 * "how many" (f7e05077 a batch row as a position, c9dd9499 a compressed row as one).  An affine pair:
 *
 *   TokPos      a point on the live sequence (pulsar_point<TokCount>): the position a token sits at, a frontier, a
 *               checkpoint, a resume point.  none() names "no position" (it is -1, the value the int fields carried).
 *   TokCount    the distance between two of them, or a length.
 *   pos + count -> pos, pos - count -> pos, pos - pos -> count, count +/- count -> count, count.times(n) where a
 *   scale is meant; pos + pos, a scaled position and every mix of the two (or with another unit) do not compile.
 *   The origin is position 0: TokPos::after(n) is the position after the first n tokens and p.before() the tokens
 *   before p -- the one crossing between a point and a length (a history's length is the position its next token
 *   takes).  Arithmetic and ordering on none() abort, and so does arithmetic that would land on it (-1 is never a
 *   computed position): fail closed; == / != none() and is_none() are how "no position" is asked.  Otherwise the
 *   arithmetic is int's, negative results included.  A count becomes KV rows only through pulsar_kv_rows (asserts >= 0).
 *   PromptToks' point type is PromptPos (L439, at the end of this file).
 *
 * The rules, enforced by the compiler:
 *   - no implicit conversion in or out: a value enters by an explicit constructor (KvBytes{n}) where it is MEASURED
 *     or CONFIGURED (a /proc read, the ledger's figure, a constant, a token array's length) and leaves by .raw() only
 *     where it leaves the type system (a printf, /metrics, JSON, an array index, an untyped caller);
 *   - arithmetic and comparison only within a type (+, -, the six comparisons), with exactly the representation's
 *     semantics (`Rep`: uint64_t's unsigned modular arithmetic for the KV accounting, int's for the token currencies,
 *     the fields they replaced), so a typed expression computes what the raw one did (a position's too, its
 *     operands not none());
 *   - a count of identical quantities scales by name (q.times(n): n banks' growth, n pages);
 *   - rows become bytes only through the family's one authority, which takes the rows and its own factor -- the
 *     session's pools (pulsar_session_quantum_growth_bytes_per_bank, family.h growth_bytes) or the declared plan
 *     (state_kinds.h pulsar_state_pool_bytes_at).  There is no scalar bytes-per-row: each pool rounds its own
 *     ceil(rows / tokens-per-row) and DeepSeek each layer its ceil(rows / compress ratio), so a "rows * bpr" helper
 *     would misstate the price; none exists, and the inverse (bytes to rows) is asked nowhere.
 *
 * Zero cost: one `Rep`, trivially copyable, standard layout (the static_asserts below), so a struct holding one may
 * still be zeroed with memset / calloc and a default-constructed value is as uninitialised as the integer it
 * replaced (value-initialise, KvBytes{}, for 0).  Header-only C++17 (nvcc's default, for pulsar.h's .cu host
 * includes); no exceptions, no RTTI. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <type_traits>
#include <utility>

/** One quantity of the unit named by `Tag`, held as a `Rep`: an integer that converts to nothing and from nothing
 *  implicitly. */
template <class Tag, class Rep = uint64_t>
class pulsar_unit {
    static_assert(std::is_integral<Rep>::value && !std::is_same<Rep, bool>::value, "a unit is held as an integer");

public:
    /** Uninitialised, as the integer it replaces (value-initialise for 0). */
    pulsar_unit() = default;
    /** Enter the unit from a raw integer (explicit only; bool and floating point are refused at compile time).
     *  @param v the raw count, converted to `Rep` exactly as an assignment to the old field was */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_unit(I v) : v_(static_cast<Rep>(v)) {}
    /** The raw count, where the value leaves the type system (printf, /metrics, JSON, an index). */
    /// @return the count as `Rep`
    constexpr Rep raw() const { return v_; }
    /** The largest representable quantity (the saturation bound the overflow checks compare against). */
    /// @return the quantity holding numeric_limits<Rep>::max()
    static constexpr pulsar_unit max() { return pulsar_unit(std::numeric_limits<Rep>::max()); }
    /** `n` copies of this quantity (n banks' growth, n pages): `Rep` multiplication.
     *  @param n how many */
    /// @return this quantity times n, in the same unit
    constexpr pulsar_unit times(Rep n) const { return pulsar_unit(v_ * n); }

    /** Sum (as `Rep`: modular for an unsigned one). */
    friend constexpr pulsar_unit operator+(pulsar_unit a, pulsar_unit b) { return pulsar_unit(a.v_ + b.v_); }
    /** Difference (as `Rep`: modular for an unsigned one, so callers that must not wrap compare first, as before). */
    friend constexpr pulsar_unit operator-(pulsar_unit a, pulsar_unit b) { return pulsar_unit(a.v_ - b.v_); }
    /** Add in place. */
    /// @param b the quantity to add
    /// @return `*this`
    constexpr pulsar_unit &operator+=(pulsar_unit b) { v_ += b.v_; return *this; }
    /** Subtract in place. */
    /// @param b the quantity to subtract
    /// @return `*this`
    constexpr pulsar_unit &operator-=(pulsar_unit b) { v_ -= b.v_; return *this; }
    /** Equal. */
    friend constexpr bool operator==(pulsar_unit a, pulsar_unit b) { return a.v_ == b.v_; }
    /** Not equal. */
    friend constexpr bool operator!=(pulsar_unit a, pulsar_unit b) { return a.v_ != b.v_; }
    /** Less. */
    friend constexpr bool operator<(pulsar_unit a, pulsar_unit b) { return a.v_ < b.v_; }
    /** Less or equal. */
    friend constexpr bool operator<=(pulsar_unit a, pulsar_unit b) { return a.v_ <= b.v_; }
    /** Greater. */
    friend constexpr bool operator>(pulsar_unit a, pulsar_unit b) { return a.v_ > b.v_; }
    /** Greater or equal. */
    friend constexpr bool operator>=(pulsar_unit a, pulsar_unit b) { return a.v_ >= b.v_; }

private:
    Rep v_;  ///< the count, in the unit `Tag` names
};

/** Tag of KvRows (never instantiated). */
struct pulsar_kv_rows_tag;
/** Tag of KvBytes (never instantiated). */
struct pulsar_kv_bytes_tag;

/** Token rows of KV (one row per token a bank's pools hold or will hold). */
typedef pulsar_unit<pulsar_kv_rows_tag> KvRows;
/** Bytes of KV the KV accounting prices (in the device's memory: pulsar_kv_device_bytes, L442). */
typedef pulsar_unit<pulsar_kv_bytes_tag> KvBytes;

/** Tag of TokCount (never instantiated). */
struct pulsar_tok_count_tag;
/** Tag of PromptToks (never instantiated). */
struct pulsar_prompt_toks_tag;

/** Tokens of the live side: a length or distance on a bank's history and on the stitched prompt a sync runs (L334). */
typedef pulsar_unit<pulsar_tok_count_tag, int> TokCount;
/** Tokens of the request's prompt, the client's bytes as tokenised, on the other side of a token seam (L334). */
typedef pulsar_unit<pulsar_prompt_toks_tag, int> PromptToks;

/** A typed position's misuse -- arithmetic or ordering on none(), or arithmetic that lands on it: a programmer error,
 *  fatal (rule 9).
 *  @param what the operation refused
 *  @param why "on" (an operand is none()) or "lands on" (the result would be) */
[[noreturn]] inline void pulsar_point_fail(const char *what, const char *why) {
    fprintf(stderr, "pulsar: %s %s a position that is none()\n", what, why);
    abort();
}

/** A point on a sequence whose distances are `Dist` (a pulsar_unit with a signed Rep): the affine half of the pair
 *  (L334).  Held as Dist's Rep, the integer it replaces; -1 is none(), "no position". */
template <class Dist>
class pulsar_point {
    /// The signed integer a position is held as: Dist's Rep.
    typedef decltype(Dist{}.raw()) Rep;
    static_assert(std::is_signed<Rep>::value, "a position's sentinel is -1: it is held signed");
    /// The raw value of none().
    static constexpr Rep kNone = Rep(-1);

    /** Abort when this is none(). @param what the operation */
    constexpr void live(const char *what) const {
        if (v_ == kNone) pulsar_point_fail(what, "on");
    }
    /** A computed position, refused when it lands on none(): arithmetic never forges the sentinel.
     *  @param v the result
     *  @param what the operation */
    /// @return the position v
    static constexpr pulsar_point moved(Rep v, const char *what) {
        return v == kNone ? (pulsar_point_fail(what, "lands on"), pulsar_point(v)) : pulsar_point(v);
    }

public:
    /** Uninitialised, as the integer it replaces (value-initialise for position 0). */
    pulsar_point() = default;
    /** Enter from a raw integer (explicit only; bool and floating point are refused; -1 enters as none(), the value
     *  the int field carried).
     *  @param v the raw position */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_point(I v) : v_(static_cast<Rep>(v)) {}
    /** No position. */
    /// @return the none() sentinel (raw -1)
    static constexpr pulsar_point none() { return pulsar_point(kNone); }
    /** Is this none()? */
    /// @return true for none()
    constexpr bool is_none() const { return v_ == kNone; }
    /** The raw position, where the value leaves the type system (printf, JSON, an index, a kernel's launch, an
     *  untyped engine body); none() leaves as -1. */
    /// @return the position as `Rep`
    constexpr Rep raw() const { return v_; }
    /** The position after the first `n` tokens (position 0 plus n): where a length ends (a count of -1 would land on
     *  none(): fatal).
     *  @param n the tokens before it */
    /// @return position n
    static constexpr pulsar_point after(Dist n) { return moved(n.raw(), "after(count)"); }
    /** The tokens before this position (it minus position 0). */
    /// @return the distance from position 0 (fatal on none())
    constexpr Dist before() const { return live("before()"), Dist(v_); }

    /** A position moved forward by a distance. */
    friend constexpr pulsar_point operator+(pulsar_point p, Dist d) {
        return p.live("pos + count"), moved(p.v_ + d.raw(), "pos + count");
    }
    /** A position moved back by a distance. */
    friend constexpr pulsar_point operator-(pulsar_point p, Dist d) {
        return p.live("pos - count"), moved(p.v_ - d.raw(), "pos - count");
    }
    /** The distance from b to a. */
    friend constexpr Dist operator-(pulsar_point a, pulsar_point b) {
        return a.live("pos - pos"), b.live("pos - pos"), Dist(a.v_ - b.v_);
    }
    /** Move forward in place. */
    /// @param d the distance to move
    /// @return `*this`
    constexpr pulsar_point &operator+=(Dist d) { return *this = *this + d; }
    /** Move back in place. */
    /// @param d the distance to move
    /// @return `*this`
    constexpr pulsar_point &operator-=(Dist d) { return *this = *this - d; }
    /** Equal (none() equals none() only). */
    friend constexpr bool operator==(pulsar_point a, pulsar_point b) { return a.v_ == b.v_; }
    /** Not equal. */
    friend constexpr bool operator!=(pulsar_point a, pulsar_point b) { return a.v_ != b.v_; }
    /** Earlier (neither may be none()). */
    friend constexpr bool operator<(pulsar_point a, pulsar_point b) {
        return a.live("pos < pos"), b.live("pos < pos"), a.v_ < b.v_;
    }
    /** Earlier or the same (neither may be none()). */
    friend constexpr bool operator<=(pulsar_point a, pulsar_point b) {
        return a.live("pos <= pos"), b.live("pos <= pos"), a.v_ <= b.v_;
    }
    /** Later (neither may be none()). */
    friend constexpr bool operator>(pulsar_point a, pulsar_point b) {
        return a.live("pos > pos"), b.live("pos > pos"), a.v_ > b.v_;
    }
    /** Later or the same (neither may be none()). */
    friend constexpr bool operator>=(pulsar_point a, pulsar_point b) {
        return a.live("pos >= pos"), b.live("pos >= pos"), a.v_ >= b.v_;
    }

private:
    Rep v_;  ///< the position, a count of tokens from position 0; -1 = none()
};

/** A point on the live sequence: a bank's history, the stitched prompt a sync runs (L334). */
typedef pulsar_point<TokCount> TokPos;

/** The KV rows a token count is, one row per token: the one crossing from a count to the KV accounting's unit.
 *  A negative count is no row count (fatal).
 *  @param n the tokens, >= 0 */
/// @return `n` as KvRows, one row per token
inline KvRows pulsar_kv_rows(TokCount n) {
    if (n.raw() < 0) {
        fprintf(stderr, "pulsar: %d tokens are no KV row count\n", n.raw());
        abort();
    }
    return KvRows{(uint32_t)n.raw()};
}

static_assert(sizeof(KvRows) == sizeof(uint64_t) && sizeof(KvBytes) == sizeof(uint64_t), "a unit is one uint64_t");
static_assert(std::is_trivially_copyable<KvBytes>::value && std::is_trivial<KvBytes>::value &&
                  std::is_standard_layout<KvBytes>::value,
              "a unit is as cheap as the uint64_t it replaces (memset / calloc-safe)");
static_assert(sizeof(TokCount) == sizeof(int) && sizeof(PromptToks) == sizeof(int) &&
                  std::is_trivial<TokCount>::value && std::is_standard_layout<PromptToks>::value,
              "a token currency is as cheap as the int it replaces");
static_assert(sizeof(TokPos) == sizeof(int) && std::is_trivial<TokPos>::value &&
                  std::is_standard_layout<TokPos>::value,
              "a position is as cheap as the int it replaces (memset / calloc-safe)");

/* ---- L334 phase 2b: the engine's other index spaces -------------------------------------------------------------
 * A position on the token sequence is not a row of a batched step, and neither is a row of a compressed KV cache; all
 * three were uint32 / int side by side in the same signatures (gpu_graph_index_comp_update took a batch row, a
 * position and a compressed row, three uint32_t in a row -- f7e05077 passed the row for the position, c9dd9499 a
 * compressed row for one).  Each space is its own affine pair, as TokPos / TokCount:
 *
 *   StepRow / StepRows   a row of one batched step (the step's row map gives each its position and bank);
 *   HeadRow / HeadRows   a headed row of a step (the logits rows: a verify capture, the draw), a SUBSET of the step's
 *                        rows -- head row h is step row heads.row[h], equal to h only where every row is headed (a
 *                        verify prefix), and that identity is crossed by name, never assumed;
 *   CompRow / CompRows   a row of a compressed KV cache or a pool that holds several tokens a row;
 *   TokPerRow            the factor between tokens and such rows (a compress ratio, a pool's tokens per row): nonzero,
 *                        not a count of anything -- a ratio can no longer be passed where a row or a position goes.
 *
 * The crossings are named and take their factor (below): a position's compressed row, the first position of a
 * compressed row, the rows a history holds or covers, the history a row count serves, and a run's row map. */

/** Tag of StepRows (never instantiated). */
struct pulsar_step_rows_tag;
/** Tag of HeadRows (never instantiated). */
struct pulsar_head_rows_tag;
/** Tag of CompRows (never instantiated). */
struct pulsar_comp_rows_tag;
/** Tag of TokPerRow (never instantiated). */
struct pulsar_tok_per_row_tag;

/** Rows of a batched step: a count, or the distance between two of its rows (L334 P2b). */
typedef pulsar_unit<pulsar_step_rows_tag, int> StepRows;
/** A row of a batched step (none() = no row). */
typedef pulsar_point<StepRows> StepRow;
/** Headed rows of a step (its logits rows): a count or a distance (L334 P2b). */
typedef pulsar_unit<pulsar_head_rows_tag, int> HeadRows;
/** A headed row of a step: an index into its logits / verify capture (none() = no row). */
typedef pulsar_point<HeadRows> HeadRow;
/** Rows of a compressed KV cache or of a pool holding several tokens a row: a count or a distance (L334 P2b). */
typedef pulsar_unit<pulsar_comp_rows_tag, int> CompRows;
/** A row of a compressed cache / multi-token pool (none() = no row). */
typedef pulsar_point<CompRows> CompRow;
/** Tokens a row of a compressed cache or pool holds: a compress ratio, a pool's tokens per row (nonzero). */
typedef pulsar_unit<pulsar_tok_per_row_tag, uint32_t> TokPerRow;

/** A factor of 0 tokens a row is no factor (fatal): the crossings below divide by it.
 *  @param r the factor */
inline void pulsar_tok_per_row_check(TokPerRow r) {
    if (r.raw() == 0u) {
        fprintf(stderr, "pulsar: 0 tokens a row is no compress ratio\n");
        abort();
    }
}

/** The compressed row a position's token lands in: floor(pos / r) (the position law; fatal on none()).
 *  @param p the position
 *  @param r tokens a row */
/// @return its row
inline CompRow pulsar_comp_row_of(TokPos p, TokPerRow r) {
    pulsar_tok_per_row_check(r);
    return CompRow{(int)((uint32_t)p.before().raw() / r.raw())};
}
/** The first position of a compressed row: row * r (a group's start: where its rope / its tokens begin).
 *  @param row the row (fatal on none())
 *  @param r   tokens a row */
/// @return its first position
inline TokPos pulsar_comp_row_pos(CompRow row, TokPerRow r) {
    pulsar_tok_per_row_check(r);
    return TokPos{(int)((uint32_t)row.before().raw() * r.raw())};
}
/** The rows a history of `len` tokens has completed: floor(len / r) (a compressed cache's frontier).
 *  @param len the tokens
 *  @param r   tokens a row */
/// @return complete rows
inline CompRows pulsar_comp_rows_held(TokCount len, TokPerRow r) {
    pulsar_tok_per_row_check(r);
    return CompRows{(int)((uint32_t)len.raw() / r.raw())};
}
/** The rows that cover `len` tokens: ceil(len / r) (a pool's rows for a partial last row; pricing).
 *  @param len the tokens
 *  @param r   tokens a row */
/// @return covering rows
inline CompRows pulsar_comp_rows_cover(TokCount len, TokPerRow r) {
    pulsar_tok_per_row_check(r);
    return CompRows{(int)(((uint32_t)len.raw() + r.raw() - 1u) / r.raw())};
}
/** The longest history `rows` complete rows serve: rows * r + r - 1 tokens (the deepest length whose floor(len / r)
 *  is `rows` -- what a lowered compressed frontier still serves).
 *  @param rows complete rows
 *  @param r    tokens a row */
/// @return the tokens
inline TokCount pulsar_comp_rows_serve(CompRows rows, TokPerRow r) {
    pulsar_tok_per_row_check(r);
    return TokCount{(int)((uint32_t)rows.raw() * r.raw() + r.raw() - 1u)};
}

/** One run of a batched step: consecutive rows from `first` holding one bank's consecutive positions from `pos0`, one
 *  token a row -- the row map's one shape (the driver's run_first / run_pos0, a verify round, a prefill chunk).  Its
 *  two crossings are the only way between a step row and a position. */
struct pulsar_step_run {
    StepRow first;  ///< the run's first row in the step
    TokPos pos0;    ///< the position its first row holds
    StepRows n;     ///< rows in the run
    /** The position row `row` holds: pos0 + (row - first).
     *  @param row a row of this run */
    /// @return its position
    TokPos pos_of(StepRow row) const { return pos0 + TokCount{(row - first).raw()}; }
    /** The row that holds position `p`: first + (p - pos0).
     *  @param p a position of this run */
    /// @return its row
    StepRow row_of(TokPos p) const { return first + StepRows{(p - pos0).raw()}; }
};

static_assert(sizeof(StepRow) == sizeof(int) && sizeof(HeadRow) == sizeof(int) && sizeof(CompRow) == sizeof(int) &&
                  sizeof(TokPerRow) == sizeof(uint32_t) && std::is_trivial<StepRow>::value &&
                  std::is_trivial<CompRow>::value && std::is_trivial<pulsar_step_run>::value,
              "the P2b index spaces are as cheap as the ints they replace (memset / calloc-safe)");

/* ---- L334 phase 3: vocabulary token ids ---------------------------------------------------------------------------
 * A token id names WHICH token; a position or a count says where / how many.  All three were int side by side
 * (L251 passed a count of 1 where eval took an id; spec_round_begin(int first_token, int max_tokens, ...)).  An id
 * is an identity, not a quantity: it compares equal or not and nothing else -- no arithmetic, no ordering, no vocab
 * of its own.  Which ids are valid is a question for an AUTHORITY (the tokenizer's size, the logits width, a family's
 * image sentinels at or past the vocab), asked by name where it applies; the type only refuses the confusions. */

/** An identity of the token vocabulary: == / != only; -1 is none() ("no token": a refused sample, an absent marker). */
template <class Tag, class Rep>
class pulsar_id {
    static_assert(std::is_signed<Rep>::value, "an id's sentinel is -1: it is held signed");
    /// The raw value of none().
    static constexpr Rep kNone = Rep(-1);

public:
    /** Uninitialised, as the integer it replaces (value-initialise for id 0). */
    pulsar_id() = default;
    /** Enter from a raw integer (explicit only; bool and floating point are refused; -1 enters as none()).
     *  @param v the raw id */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_id(I v) : v_(static_cast<Rep>(v)) {}
    /** No token. */
    /// @return the none() sentinel (raw -1)
    static constexpr pulsar_id none() { return pulsar_id(kNone); }
    /** Is this none()? */
    /// @return true for none()
    constexpr bool is_none() const { return v_ == kNone; }
    /** Is this a token at all -- not negative (none(), or a negative a caller forged)?  The vocab bound is an
     *  authority's (below()), not this. */
    /// @return true for a non-negative id
    constexpr bool is_token() const { return v_ >= 0; }
    /** Is this a token below `n` -- an authority's vocab bound (the tokenizer's size, the logits width)?
     *  @param n the bound */
    /// @return true when 0 <= id < n
    constexpr bool below(uint32_t n) const { return v_ >= 0 && (uint32_t)v_ < n; }
    /** The raw id, where it leaves the type system: a device upload, a table index, printf, the wire. */
    /// @return the id as `Rep` (none() as -1)
    constexpr Rep raw() const { return v_; }
    /** The same token (none() equals none() only). */
    friend constexpr bool operator==(pulsar_id a, pulsar_id b) { return a.v_ == b.v_; }
    /** Another token. */
    friend constexpr bool operator!=(pulsar_id a, pulsar_id b) { return a.v_ != b.v_; }

private:
    Rep v_;  ///< the id; -1 = none()
};

/** Tag of TokId (never instantiated). */
struct pulsar_tok_id_tag;
/** A vocabulary token id (L334 P3). */
typedef pulsar_id<pulsar_tok_id_tag, int32_t> TokId;

static_assert(sizeof(TokId) == sizeof(int32_t) && std::is_trivial<TokId>::value &&
                  std::is_standard_layout<TokId>::value,
              "a token id is as cheap as the int32 it replaces (memset / calloc-safe; arrays of it upload as int32)");

/** L334 P3b: an id array as the int32 array a kernel launch on host memory or a wire struct reads -- the ONE crossing
 *  of a TokId array into raw ids (TokId is exactly an int32: same size, trivial, standard layout).  Byte copies
 *  (uploads, memcpy, fwrite, digests) take the TokId array as it is and need no crossing.
 *  @param ids the id array */
/// @return the same memory as int32 ids
inline const int32_t *pulsar_tok_ids_raw(const TokId *ids) { return reinterpret_cast<const int32_t *>(ids); }
/** The writable form of pulsar_tok_ids_raw (a wire struct or kernel output filled in place).
 *  @param ids the id array */
/// @return the same memory as int32 ids
inline int32_t *pulsar_tok_ids_raw(TokId *ids) { return reinterpret_cast<int32_t *>(ids); }
/** The reverse crossing: raw int32 ids (a received wire frame, a kernel's host output) viewed as TokIds, in place --
 *  no copy (a sync's prompt can be a million ids).
 *  @param raw the int32 ids */
/// @return the same memory as token ids
inline const TokId *pulsar_tok_ids_of(const int32_t *raw) { return reinterpret_cast<const TokId *>(raw); }

/* ---- L334 phase 4: bank ids -------------------------------------------------------------------------------------
 * A bank is one conversation's KV / state slot in a session's pool.  Its id sat as uint32_t beside a layer index
 * (gpu_graph_n_comp(g, bank, il) vs the views' (g, il, bank)), a checkpoint slot, a decl id, a step row and the
 * server's slot index (fae579f0: slot i == bank i taken for granted, and a slots[] bound taken from the bank count).
 * An id like TokId: == / != only, none() = -1 (no bank installed), below(n) against an authority's count (the pool,
 * PULSAR_MSEQ_MAX); is_token() reads "a real bank" here. */
/** Tag of BankId (never instantiated). */
struct pulsar_bank_tag;
/** A bank id in a session's pool (L334 P4). */
typedef pulsar_id<pulsar_bank_tag, int32_t> BankId;

/** A bank as the index of a per-bank array (fatal on none(): indexing with "no bank" is always a bug).
 *  @param b the bank */
/// @return its index
inline uint32_t pulsar_bank_index(BankId b) {
    if (!b.is_token()) {
        fprintf(stderr, "pulsar: bank %d is no bank -- refusing to index with it\n", b.raw());
        abort();
    }
    return (uint32_t)b.raw();
}

/** L334 P4 S5: a bank array as the int32 array a kernel launch on host memory or a wire struct reads -- the named
 *  crossing (BankId is exactly an int32).  Byte copies (uploads, memcpy) take BankId arrays as they are.
 *  @param banks the bank array */
/// @return the same memory as int32 bank ids
inline const int32_t *pulsar_bank_ids_raw(const BankId *banks) { return reinterpret_cast<const int32_t *>(banks); }
/** The reverse, in place: raw int32 bank ids viewed as BankIds.  @param raw the int32 ids */
/// @return the same memory as bank ids
inline const BankId *pulsar_bank_ids_of(const int32_t *raw) { return reinterpret_cast<const BankId *>(raw); }

static_assert(sizeof(BankId) == sizeof(int32_t) && std::is_trivial<BankId>::value &&
                  std::is_standard_layout<BankId>::value,
              "a bank id is as cheap as the int32 it replaces (arrays of it upload as int32)");

/* ---- L334 phase 5: tensor bytes -----------------------------------------------------------------------------------
 * A tensor's extent or an offset into it, in BYTES -- beside element counts, row counts and float counts of the same
 * uint64_t in the GPU host API (f3a617af / d0b2fdb5: a launcher took the allocation's bytes for its live rows;
 * 22c92f71: element size x width priced where a packed row's bytes belonged; read / read_f32, write / write_q_f32,
 * alloc / alloc_elt are byte and element twins of the same shape).  Its own unit, not KvBytes (that is what the KV
 * accounting prices; kv_bytes_of crosses by name).  Offsets are TensorBytes too: a distance from the tensor's base.
 * Element counts stay raw at the element APIs -- the byte APIs no longer take them. */
/** Tag of TensorBytes (never instantiated). */
struct pulsar_tensor_bytes_tag;
/** Bytes of a tensor's extent, or a byte offset into one (L334 P5). */
typedef pulsar_unit<pulsar_tensor_bytes_tag> TensorBytes;

/** The bytes of `n` elements of T: the named form of `n * sizeof(T)`.
 *  @param n the elements */
/// @return their bytes
template <class T>
constexpr TensorBytes bytes_of(uint64_t n) { return TensorBytes{n * (uint64_t)sizeof(T)}; }
/** The bytes of `n` elements of `esz` bytes each (a format's element size: *_ELT_SIZE, alloc_elt).
 *  @param n   the elements
 *  @param esz bytes per element */
/// @return their bytes
constexpr TensorBytes elt_bytes(uint64_t n, uint64_t esz) { return TensorBytes{n * esz}; }
/** Tensor bytes as what the KV accounting prices -- the one crossing into KvBytes (a pool's resident extent).
 *  @param b the tensor bytes */
/// @return the same bytes as KvBytes
constexpr KvBytes kv_bytes_of(TensorBytes b) { return KvBytes{b.raw()}; }

static_assert(sizeof(TensorBytes) == sizeof(uint64_t) && std::is_trivial<TensorBytes>::value,
              "tensor bytes are as cheap as the uint64_t they replace");


/* ---- L334 phase 7: rendered-text bytes ----------------------------------------------------------------------------
 * The disk KV chains (pulsar_segstore.h, pulsar_kvchain.h) and the server's KV cache key a history by the bytes its
 * tokens RENDER to (pulsar_kvtext_render_tokens_text), so a chain's text length sat beside its token span G -- a
 * uint64_t / size_t of text beside a uint32_t / int of tokens, in the same signatures (pulsar_segstore_put took
 * (G_prev, G, text, text_len) in a row).  The bugs were semantic, not arithmetic: a chain's text carrying an EOS its
 * tokens lacked (27d5bff6: a text end taken for a token end), and byte spans cut from the wrong text (9637c76b).
 *
 *   TextBytes   a length of rendered text in bytes -- a segment's own text, a chain's text from byte 0 to its G, a
 *               request's rendered prompt -- or an offset into such a text, which IS the length up to it: no point
 *               type and no sentinel (an offset is never "none").
 *
 * The rules are the P1 ones: no implicit conversion, arithmetic and ordering only within the unit (uint64_t's), .raw()
 * where the bytes leave the type system (a memcmp / memcpy, a hash's update, a printf, a file offset).  A token
 * position or count never becomes a byte length: the two meet only where a chain is checked (a stored text_end against
 * the length its G tokens render to).  A renderer's client-data span (pulsar_text_span, below) holds offsets of its
 * own type since L440 (PromptOff); a length enters one by name (PromptOff::after).  Out of scope: a segment file's
 * header / payload / trailer offsets (the file format's arithmetic, HDR + text_len + payload via .raw()), byte offsets
 * inside the tokenizer and BPE, JSON body offsets. */
/** Tag of TextBytes (never instantiated). */
struct pulsar_text_bytes_tag;
/** Bytes of rendered text: a length, or an offset from byte 0 (the length up to it) (L334 P7). */
typedef pulsar_unit<pulsar_text_bytes_tag> TextBytes;

static_assert(sizeof(TextBytes) == sizeof(uint64_t) && std::is_trivial<TextBytes>::value,
              "text bytes are as cheap as the uint64_t they replace");

/* ---- L436: layer index spaces -------------------------------------------------------------------------------------
 * "Layer" names several numberings that were all uint32_t `il` / `li` / `l` / `layer` / `slot`, side by side in the
 * same signatures and the same loops (L236#2: a drafter's own layer 0..2 indexed the TRUNK's expert-count table).
 * Each is its own index type; none converts to another, to a unit above or to an integer without a name:
 *
 *   TrunkLayer  a layer of the model's layer plan (family.h pulsar_layer_plan: 0 .. n_layer - 1) -- what the step
 *               driver dispatches on, a family's shape bit masks and per-layer weights, a drafter's tap point.
 *   LayerSlot   a layer slot of the declared state (state_kinds.h pulsar_state_plan): the trunk's layers in order,
 *               then a drafter's ([0, n_trunk) and [n_trunk, n_slots)).  What a state tensor, a KV pool and a
 *               capture are addressed by, and what a forward that runs a drafter's layer through the trunk's code
 *               carries (Qwen's MTP layer, Nemotron-H's MTP module).
 *   DraftLayer  a drafter's OWN layer or module (a block drafter's layer, the chained-MTP lane's module, DeepSeek's
 *               DSpark layer): it starts at 0 again, so it is never a trunk layer.
 *   SrcBlock    a block in the source checkpoint's own numbering, as a drafter's config names its taps: equal to
 *               the trunk layer where the checkpoint's blocks are the plan's layers (Qwen, Gemma 4), a table where
 *               the plan groups blocks into layers (Nemotron-H: a mixer block and its MoE block are one layer).
 *   KindOrd     a layer's ordinal among the layers of one KIND (the n-th attention layer of a Mamba / attention
 *               interleave, the n-th slot carrying a state declaration, the n-th Engram layer): the index of
 *               per-kind storage that is compacted over the kind.  One type for every kind: each ordinal lives in
 *               one module beside the one table that produces it, and no two kinds' ordinals meet.
 *   VisLayer    a block of a vision tower (the shared ViT encoder, Qwen's and DeepSeek's towers): its own depth, its
 *               own weights, beside a decoder's trunk layers in the same family files.
 *   TpSlot      a layer's exchange slot in the tensor-parallel slab: what pulsar_tp.h's calls take, down to the
 *               gate frame's 16-bit field, where it leaves as the raw number the wire always carried.
 *
 * An index, not a quantity: no sum of two, no scaling.  It orders (a loop's bound, "the next layer"), steps by one
 * (++, next()), and is asked against a COUNT by name (below(n): the bound is the owner's -- the plan's n_layer, the
 * state plan's n_slots; last_of(n)).  none() is the unsigned all-ones the uint32 fields carried for "no layer"
 * (UINT32_MAX); ordering and stepping on it abort (rule 9), == / != / is_none() ask it.  The crossings between
 * spaces are named functions owned by whoever owns the relation: the state layout (state_kinds.h
 * pulsar_layer_slot_of, pulsar_drafter_slot, pulsar_slot_trunk_layer), a family's shape (its SrcBlock -> TrunkLayer,
 * its kind ordinals), the step driver (pulsar_tp_slot_of).  .raw() is for where the number leaves the type system:
 * an array subscript in the space's own table, a kernel launch, a tensor's name, a printf, a file or wire field. */

/** An index's misuse -- ordering or stepping on none(): a programmer error, fatal (rule 9).
 *  @param what the operation refused */
[[noreturn]] inline void pulsar_index_fail(const char *what) {
    fprintf(stderr, "pulsar: %s on a layer index that is none()\n", what);
    abort();
}

/** An index into the space named by `Tag`, held as an unsigned `Rep` (the uint32_t it replaces): explicit in, .raw()
 *  out, ordered and stepped within its own space only (L436). */
template <class Tag, class Rep = uint32_t>
class pulsar_index {
    static_assert(std::is_unsigned<Rep>::value && !std::is_same<Rep, bool>::value,
                  "an index is held unsigned: none() is all ones");
    /// The raw value of none().
    static constexpr Rep kNone = std::numeric_limits<Rep>::max();

    /** Abort when this is none(). @param what the operation */
    constexpr void live(const char *what) const {
        if (v_ == kNone) pulsar_index_fail(what);
    }

public:
    /** Uninitialised, as the integer it replaces (value-initialise for index 0). */
    pulsar_index() = default;
    /** Enter from a raw integer (explicit only; bool and floating point are refused; all ones enters as none()).
     *  @param v the raw index, converted to `Rep` exactly as an assignment to the old field was */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_index(I v) : v_(static_cast<Rep>(v)) {}
    /** No index (UINT32_MAX, the value the uint32 fields carried). */
    /// @return the none() sentinel
    static constexpr pulsar_index none() { return pulsar_index(kNone); }
    /** Is this none()? */
    /// @return true for none()
    constexpr bool is_none() const { return v_ == kNone; }
    /** The raw index, where it leaves the type system: a subscript of the space's own table, a kernel launch, a
     *  tensor's name, printf, a file or wire field. */
    /// @return the index as `Rep` (none() as all ones)
    constexpr Rep raw() const { return v_; }
    /** Is this an index below `n` -- the owner's count (a plan's n_layer, a drafter's depth)?  The loop bound and the
     *  range check: `il < n`.  none() is below nothing.
     *  @param n the count */
    /// @return true when index < n
    constexpr bool below(Rep n) const { return v_ != kNone && v_ < n; }
    /** Is this the last of `n` -- `il + 1 == n`?  (fatal on none())
     *  @param n the count */
    /// @return true when this is index n - 1
    constexpr bool last_of(Rep n) const { return live("last_of(n)"), v_ + Rep(1) == n; }
    /** The index after this one in the same space: `il + 1` (fatal on none()). */
    /// @return the next index
    constexpr pulsar_index next() const { return live("next()"), pulsar_index(v_ + Rep(1)); }
    /** Step to the next index in place: `il++` (fatal on none()). */
    /// @return `*this`
    constexpr pulsar_index &operator++() { return live("++"), ++v_, *this; }

    /** The same index (none() equals none() only). */
    friend constexpr bool operator==(pulsar_index a, pulsar_index b) { return a.v_ == b.v_; }
    /** Another index. */
    friend constexpr bool operator!=(pulsar_index a, pulsar_index b) { return a.v_ != b.v_; }
    /** Earlier (neither may be none()). */
    friend constexpr bool operator<(pulsar_index a, pulsar_index b) {
        return a.live("index < index"), b.live("index < index"), a.v_ < b.v_;
    }
    /** Earlier or the same (neither may be none()). */
    friend constexpr bool operator<=(pulsar_index a, pulsar_index b) {
        return a.live("index <= index"), b.live("index <= index"), a.v_ <= b.v_;
    }
    /** Later (neither may be none()). */
    friend constexpr bool operator>(pulsar_index a, pulsar_index b) {
        return a.live("index > index"), b.live("index > index"), a.v_ > b.v_;
    }
    /** Later or the same (neither may be none()). */
    friend constexpr bool operator>=(pulsar_index a, pulsar_index b) {
        return a.live("index >= index"), b.live("index >= index"), a.v_ >= b.v_;
    }

private:
    Rep v_;  ///< the index; all ones = none()
};

/** Tag of TrunkLayer (never instantiated). */
struct pulsar_trunk_layer_tag;
/** Tag of LayerSlot (never instantiated). */
struct pulsar_layer_slot_tag;
/** Tag of DraftLayer (never instantiated). */
struct pulsar_draft_layer_tag;
/** Tag of SrcBlock (never instantiated). */
struct pulsar_src_block_tag;
/** Tag of KindOrd (never instantiated). */
struct pulsar_kind_ord_tag;
/** Tag of VisLayer (never instantiated). */
struct pulsar_vis_layer_tag;
/** Tag of TpSlot (never instantiated). */
struct pulsar_tp_slot_tag;

/** A layer of the model's layer plan, 0 .. n_layer - 1 (L436). */
typedef pulsar_index<pulsar_trunk_layer_tag> TrunkLayer;
/** A layer slot of the declared state: the trunk's layers, then a drafter's (L436). */
typedef pulsar_index<pulsar_layer_slot_tag> LayerSlot;
/** A drafter's own layer or module, from 0 (L436). */
typedef pulsar_index<pulsar_draft_layer_tag> DraftLayer;
/** A block in the source checkpoint's own numbering (a drafter's tap ids) (L436). */
typedef pulsar_index<pulsar_src_block_tag> SrcBlock;
/** A layer's ordinal among the layers of one kind: the index of storage compacted over the kind (L436). */
typedef pulsar_index<pulsar_kind_ord_tag> KindOrd;
/** A block of a vision tower (L436). */
typedef pulsar_index<pulsar_vis_layer_tag> VisLayer;
/** A layer's exchange slot in the tensor-parallel slab (L436). */
typedef pulsar_index<pulsar_tp_slot_tag> TpSlot;

static_assert(sizeof(TrunkLayer) == sizeof(uint32_t) && sizeof(LayerSlot) == sizeof(uint32_t) &&
                  sizeof(DraftLayer) == sizeof(uint32_t) && sizeof(SrcBlock) == sizeof(uint32_t) &&
                  sizeof(KindOrd) == sizeof(uint32_t) && sizeof(VisLayer) == sizeof(uint32_t) &&
                  sizeof(TpSlot) == sizeof(uint32_t) && std::is_trivial<TrunkLayer>::value &&
                  std::is_standard_layout<TrunkLayer>::value && std::is_trivial<LayerSlot>::value &&
                  std::is_standard_layout<LayerSlot>::value,
              "a layer index is as cheap as the uint32 it replaces (memset / calloc-safe; arrays of it are uint32 "
              "arrays)");

/* ---- L436: tables indexed by one space ---------------------------------------------------------------------------
 * A table that one index space subscripts, as a type that takes NO OTHER subscript: `plan.kind[il]` with a layer
 * slot, a drafter's layer or a bare integer does not compile.  Two forms:
 *
 *   pulsar_indexed<T, Index, N>     a fixed array T[N] -- exactly the array it replaces (one member, standard layout,
 *                                   an aggregate: memset / calloc / brace initialisation and its place in a struct are
 *                                   what they were; T may itself be an array, for a table whose FIRST subscript is the
 *                                   space's);
 *   pulsar_indexed_view<T, Index>   a pointer and its count, for a heap table or a caller's array.
 *
 * operator[](Index) only, bounds checked (fatal: an index past the table, none() included, is a programmer error --
 * rule 9); data() / size() are for where the table leaves the type system whole (memcpy, memset, a kernel launch, a
 * digest, a walk by another counter in the owner's own module). */

/** A subscript past an indexed table: fatal (rule 9).
 *  @param i the raw index
 *  @param n the table's entries */
[[noreturn]] inline void pulsar_indexed_fail(uint64_t i, uint64_t n) {
    fprintf(stderr, "pulsar: index %llu is outside a table of %llu -- refusing the subscript\n", (unsigned long long)i,
            (unsigned long long)n);
    abort();
}

/** The fixed array T[N], subscripted by `Index` and nothing else (L436). */
template <class T, class Index, size_t N>
struct pulsar_indexed {
    T v_[N];  ///< the entries (public so the type stays an aggregate; use operator[] / data())

    /** Entry `i` (fatal past the table).
     *  @param i the index, in this table's space */
    /// @return the entry
    constexpr T &operator[](Index i) {
        static_assert(sizeof(pulsar_indexed) == sizeof(T[N]) && alignof(pulsar_indexed) == alignof(T[N]),
                      "an indexed table has the size and alignment of the array it replaces");
        return i.raw() < N ? v_[i.raw()] : (pulsar_indexed_fail(i.raw(), N), v_[0]);
    }
    /** Entry `i`, read-only (fatal past the table).
     *  @param i the index, in this table's space */
    /// @return the entry
    constexpr const T &operator[](Index i) const {
        static_assert(sizeof(pulsar_indexed) == sizeof(T[N]) && alignof(pulsar_indexed) == alignof(T[N]),
                      "an indexed table has the size and alignment of the array it replaces");
        return i.raw() < N ? v_[i.raw()] : (pulsar_indexed_fail(i.raw(), N), v_[0]);
    }
    /** The array itself, where it leaves the type system whole. */
    /// @return the first entry's address
    constexpr T *data() { return v_; }
    /** The array itself, read-only. */
    /// @return the first entry's address
    constexpr const T *data() const { return v_; }
    /** The table's entries. */
    /// @return N
    static constexpr size_t size() { return N; }
};

/** A pointer to `n` entries of T, subscripted by `Index` and nothing else (L436). */
template <class T, class Index>
struct pulsar_indexed_view {
    T *p_;        ///< the entries (borrowed, or the owner's heap block)
    uint32_t n_;  ///< how many

    /** Entry `i` (fatal past the table).
     *  @param i the index, in this table's space */
    /// @return the entry
    constexpr T &operator[](Index i) const { return i.raw() < n_ ? p_[i.raw()] : (pulsar_indexed_fail(i.raw(), n_), p_[0]); }
    /** The entries, where they leave the type system whole. */
    /// @return the first entry's address
    constexpr T *data() const { return p_; }
    /** The table's entries. */
    /// @return the count
    constexpr uint32_t size() const { return n_; }
};

static_assert(sizeof(pulsar_indexed<uint32_t, TrunkLayer, 7>) == sizeof(uint32_t[7]) &&
                  alignof(pulsar_indexed<uint32_t, TrunkLayer, 7>) == alignof(uint32_t) &&
                  sizeof(pulsar_indexed<void *[3], LayerSlot, 5>) == sizeof(void *[5][3]) &&
                  alignof(pulsar_indexed<void *[3], LayerSlot, 5>) == alignof(void *) &&
                  sizeof(pulsar_indexed<uint8_t, SrcBlock, 9>) == 9 &&
                  std::is_trivial<pulsar_indexed<uint32_t, TrunkLayer, 7>>::value &&
                  std::is_standard_layout<pulsar_indexed<uint32_t, TrunkLayer, 7>>::value &&
                  std::is_aggregate<pulsar_indexed<uint32_t, TrunkLayer, 7>>::value,
              "an indexed table is the array it replaces: same size, alignment and layout (memset / calloc-safe)");

/* ==== L437: offsets into the texts of the OUTPUT path (begin) ========================================================
 * A turn's generated bytes are read by several readers, each counting from its own byte 0: the turn's generated text
 * (the server's gen_state::text of one decode attempt: the release cursor, the stop scan's resume point, the cut where
 * a turn ends) and the stream an output parser was fed since its init (out_parse_kit.h: where a call's marker was, the
 * calls' sampled span).  All were size_t side by side, and a parser's stream offset was used as an offset into the
 * turn's text (L419's first_call_at, the raw span): equal only because every parser is fed the text from its first
 * byte, which nothing checked (L435 S2-9).  Each text is its own offset type over one shared length:
 *
 *   GenBytes    a length of generated text in bytes: a token's piece, a marker, the bytes fed to a reader, the distance
 *               between two offsets of one text.  Not TextBytes: that is RENDERED text (a prompt, a chain), which also
 *               serves as its own offsets -- a prompt's length or offset must not move a generated text's offset.
 *   GenOff      an offset into the turn's generated text (gen_state::text, from the attempt's first byte).
 *   StreamOff   an offset into the bytes ONE output parser was fed since its init().
 *   BodyOff     an offset into an open tool call's BODY: the bytes between the marker that opened the call and the
 *               one that closes it, as an out_call_body reader is handed them (it is not a stretch of the stream:
 *               dropped markers are cut out of it).
 *
 * off + len -> off, off - len -> off, off - off -> len (within one text), ordering and equality within one text; an
 * offset of one text is no offset of another, not even explicitly.  The only way from a parser's stream to the turn's
 * text is the conversion of the object that feeds the parser and so owns the relation (the server's kit_gen::text_at:
 * the text offset is the text cursor less the distance to the stream's end -- right for a parser started anywhere in
 * the text).  none() is "no offset" (raw SIZE_MAX, std::string::npos: the value the size_t fields carried -- no call
 * opened, no tail); arithmetic and ordering on it, and arithmetic that would land on it, are fatal, as a position's.
 * Otherwise the arithmetic is size_t's.  The origin is offset 0: Off::after(n) is where the first n bytes end and
 * off.before() the bytes before it -- the one crossing between an offset and a length (a text's length is where its
 * next byte goes).  Zero cost: one size_t.
 *
 * Left untyped, each inside one object and crossing no boundary as an offset: the parser's held bytes (hold_: offsets
 * local to out_parser::feed / find_marker, converted by out_parser::stream_at), the Markdown scanner (run lengths
 * only, no offset) and the grammar gate's trigger matchers (KMP states: prefix lengths of a PATTERN).  The scanners
 * under DeepSeek's DSML walk and under the call-body readers take a (text, index) pair like any string function; a
 * typed cursor becomes that index in one place each (walk_ix, body_ix).
 *
 * The PROMPT side has one typed offset space of its own, with rendered text's TextBytes for its lengths:
 *
 *   ContentOff  an offset into ONE chat message's content text as the server holds it (chat_msg::content): where an
 *               image's placeholder sits, where a tool reference was written.  Not an offset of the rendered prompt
 *               the content is later copied into, nor of the content a family renderer is handed with its markers
 *               stripped.
 *
 * The rendered prompt's own offsets: TextBytes where a chain's or a prompt's text is measured (L334 P7), and the
 * renderer's pulsar_text_span {lo, hi}, typed in the L440 section below. */

/** Tag of GenBytes (never instantiated). */
struct pulsar_gen_bytes_tag;
/** Bytes of generated text: a length, or the distance between two offsets of one text (L437). */
typedef pulsar_unit<pulsar_gen_bytes_tag, size_t> GenBytes;

/** An offset into the text named by `Tag`, in bytes from its first byte, with `Dist` distances -- GenBytes for the
 *  output path's texts, TextBytes for the prompt side's (L437).  Held as the size_t it replaces; SIZE_MAX
 *  (std::string::npos) is none(), "no offset". */
template <class Tag, class Dist = GenBytes>
class pulsar_text_off {
    /// The raw value of none().
    static constexpr size_t kNone = std::numeric_limits<size_t>::max();

    /** Abort when this is none(). @param what the operation */
    constexpr void live(const char *what) const {
        if (v_ == kNone) pulsar_point_fail(what, "on");
    }
    /** A computed offset, refused when it lands on none(): arithmetic never forges the sentinel.
     *  @param v the result
     *  @param what the operation */
    /// @return the offset v
    static constexpr pulsar_text_off moved(size_t v, const char *what) {
        return v == kNone ? (pulsar_point_fail(what, "lands on"), pulsar_text_off(v)) : pulsar_text_off(v);
    }

public:
    /** Uninitialised, as the integer it replaces (value-initialise for offset 0). */
    pulsar_text_off() = default;
    /** Enter from a raw integer (explicit only; bool and floating point are refused; SIZE_MAX enters as none(), the
     *  value the size_t field carried).
     *  @param v the raw offset */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_text_off(I v) : v_(static_cast<size_t>(v)) {}
    /** No offset. */
    /// @return the none() sentinel (raw SIZE_MAX)
    static constexpr pulsar_text_off none() { return pulsar_text_off(kNone); }
    /** Is this none()? */
    /// @return true for none()
    constexpr bool is_none() const { return v_ == kNone; }
    /** The raw offset, where the value leaves the type system (a buffer index, printf, a protocol sink); none() leaves
     *  as SIZE_MAX. */
    /// @return the offset as size_t
    constexpr size_t raw() const { return v_; }
    /** The offset past the first `n` bytes (offset 0 plus n): where a text of that length ends.
     *  @param n the bytes before it */
    /// @return offset n
    static constexpr pulsar_text_off after(Dist n) { return moved(n.raw(), "after(len)"); }
    /** The bytes before this offset (it minus offset 0). */
    /// @return the distance from offset 0 (fatal on none())
    constexpr Dist before() const { return live("before()"), Dist(v_); }

    /** An offset moved forward by a length. */
    friend constexpr pulsar_text_off operator+(pulsar_text_off p, Dist d) {
        return p.live("off + len"), moved(p.v_ + d.raw(), "off + len");
    }
    /** An offset moved back by a length. */
    friend constexpr pulsar_text_off operator-(pulsar_text_off p, Dist d) {
        return p.live("off - len"), moved(p.v_ - d.raw(), "off - len");
    }
    /** The distance from b to a (size_t's: callers that must not wrap compare first, as before). */
    friend constexpr Dist operator-(pulsar_text_off a, pulsar_text_off b) {
        return a.live("off - off"), b.live("off - off"), Dist(a.v_ - b.v_);
    }
    /** Move forward in place. */
    /// @param d the length to move by
    /// @return `*this`
    constexpr pulsar_text_off &operator+=(Dist d) { return *this = *this + d; }
    /** Move back in place. */
    /// @param d the length to move by
    /// @return `*this`
    constexpr pulsar_text_off &operator-=(Dist d) { return *this = *this - d; }
    /** Equal (none() equals none() only). */
    friend constexpr bool operator==(pulsar_text_off a, pulsar_text_off b) { return a.v_ == b.v_; }
    /** Not equal. */
    friend constexpr bool operator!=(pulsar_text_off a, pulsar_text_off b) { return a.v_ != b.v_; }
    /** Earlier (neither may be none()). */
    friend constexpr bool operator<(pulsar_text_off a, pulsar_text_off b) {
        return a.live("off < off"), b.live("off < off"), a.v_ < b.v_;
    }
    /** Earlier or the same (neither may be none()). */
    friend constexpr bool operator<=(pulsar_text_off a, pulsar_text_off b) {
        return a.live("off <= off"), b.live("off <= off"), a.v_ <= b.v_;
    }
    /** Later (neither may be none()). */
    friend constexpr bool operator>(pulsar_text_off a, pulsar_text_off b) {
        return a.live("off > off"), b.live("off > off"), a.v_ > b.v_;
    }
    /** Later or the same (neither may be none()). */
    friend constexpr bool operator>=(pulsar_text_off a, pulsar_text_off b) {
        return a.live("off >= off"), b.live("off >= off"), a.v_ >= b.v_;
    }

private:
    size_t v_;  ///< the offset, in bytes from the text's first byte; SIZE_MAX = none()
};

/** Tag of GenOff (never instantiated). */
struct pulsar_gen_off_tag;
/** Tag of StreamOff (never instantiated). */
struct pulsar_stream_off_tag;
/** Tag of BodyOff (never instantiated). */
struct pulsar_body_off_tag;

/** An offset into the turn's generated text: gen_state::text of one decode attempt (L437). */
typedef pulsar_text_off<pulsar_gen_off_tag> GenOff;
/** An offset into the bytes one output parser was fed since its init() (L437). */
typedef pulsar_text_off<pulsar_stream_off_tag> StreamOff;
/** An offset into an open tool call's body, as its out_call_body reader is handed it (L437). */
typedef pulsar_text_off<pulsar_body_off_tag> BodyOff;
/** Tag of ContentOff (never instantiated). */
struct pulsar_content_off_tag;
/** An offset into one chat message's content text (chat_msg::content), with TextBytes distances (L437). */
typedef pulsar_text_off<pulsar_content_off_tag, TextBytes> ContentOff;

static_assert(sizeof(GenBytes) == sizeof(size_t) && sizeof(GenOff) == sizeof(size_t) &&
                  sizeof(StreamOff) == sizeof(size_t) && std::is_trivial<GenBytes>::value &&
                  std::is_trivial<GenOff>::value && std::is_standard_layout<StreamOff>::value,
              "an output-path offset or length is as cheap as the size_t it replaces");
static_assert(sizeof(BodyOff) == sizeof(size_t) && std::is_trivial<BodyOff>::value &&
                  !std::is_constructible<BodyOff, GenOff>::value && !std::is_constructible<GenOff, BodyOff>::value &&
                  !std::is_constructible<BodyOff, StreamOff>::value &&
                  !std::is_constructible<StreamOff, BodyOff>::value &&
                  !std::is_constructible<BodyOff, GenBytes>::value && !std::is_constructible<BodyOff, TextBytes>::value &&
                  !std::is_convertible<size_t, BodyOff>::value && !std::is_convertible<BodyOff, size_t>::value,
              "a call body's offset is as cheap as a size_t, and is no offset of the stream or the text, no length");
static_assert(!std::is_constructible<GenOff, StreamOff>::value && !std::is_constructible<StreamOff, GenOff>::value &&
                  !std::is_constructible<GenOff, GenBytes>::value && !std::is_constructible<GenBytes, GenOff>::value &&
                  !std::is_constructible<StreamOff, GenBytes>::value &&
                  !std::is_constructible<GenBytes, StreamOff>::value,
              "an offset of one text is no offset of another and no length, not even explicitly");
static_assert(!std::is_constructible<GenOff, TextBytes>::value && !std::is_constructible<StreamOff, TextBytes>::value &&
                  !std::is_constructible<GenBytes, TextBytes>::value &&
                  !std::is_constructible<TextBytes, GenBytes>::value &&
                  !std::is_constructible<TextBytes, GenOff>::value && !std::is_constructible<GenOff, TokPos>::value &&
                  !std::is_constructible<GenOff, TokCount>::value && !std::is_constructible<GenBytes, TokCount>::value &&
                  !std::is_constructible<StreamOff, TokPos>::value && !std::is_constructible<GenOff, TokId>::value,
              "rendered-text bytes and the token types never enter the output path's offsets, nor leave as them");
static_assert(!std::is_convertible<size_t, GenOff>::value && !std::is_convertible<GenOff, size_t>::value &&
                  !std::is_convertible<size_t, StreamOff>::value && !std::is_convertible<StreamOff, size_t>::value &&
                  !std::is_convertible<size_t, GenBytes>::value && !std::is_convertible<GenBytes, size_t>::value &&
                  !std::is_convertible<int, GenOff>::value && !std::is_convertible<int, GenBytes>::value,
              "no implicit way in or out of a bare integer");
static_assert(sizeof(ContentOff) == sizeof(size_t) && std::is_trivial<ContentOff>::value &&
                  !std::is_constructible<ContentOff, GenOff>::value && !std::is_constructible<GenOff, ContentOff>::value &&
                  !std::is_constructible<ContentOff, BodyOff>::value &&
                  !std::is_constructible<ContentOff, StreamOff>::value &&
                  !std::is_constructible<ContentOff, TextBytes>::value &&
                  !std::is_constructible<TextBytes, ContentOff>::value &&
                  !std::is_constructible<ContentOff, GenBytes>::value &&
                  !std::is_convertible<size_t, ContentOff>::value && !std::is_convertible<ContentOff, size_t>::value,
              "a message content's offset is no offset of generated text and no length, and enters by name only");
/* ==== L437: offsets into the texts of the OUTPUT path (end) ======================================================== */

/* ==== L439: a position on a PROMPT (begin) ==========================================================================
 * PromptToks had no point type (L334: "no prompt-side position crosses the API yet").  One does now: the shared-
 * prefix wait cuts a prefill at a token of the prompt its sync was handed, the fused prefill lanes read that prompt
 * at the bank's position, and both carried the index as the `int` that also held a position on the bank's history.
 * Across a token seam (pulsar.h pulsar_prefix_match) the two are different numbers for the same byte of text:
 *
 *   PromptPos   a point on a prompt -- the request's own tokens, or the prompt a sync is handed
 *               (pulsar_point<PromptToks>): an index into THAT token array, an end, a cut.  none() = no position.
 *   PromptToks  the distance between two of them, or a length (as before).
 *
 * It mixes with nothing on the live side: a TokPos becomes a PromptPos only through the match that relates the two
 * sequences (pulsar_prefix_match's stitch and its inverse, or a proof that no seam lies below: the server's
 * fuse_witness), never by constructing one from the other's raw value. */

/** A point on a prompt's own tokens: an index into the request's (or a sync prompt's) token array (L439). */
typedef pulsar_point<PromptToks> PromptPos;

static_assert(sizeof(PromptPos) == sizeof(int) && std::is_trivial<PromptPos>::value &&
                  std::is_standard_layout<PromptPos>::value,
              "a prompt position is as cheap as the int it replaces (memset / calloc-safe)");
static_assert(!std::is_constructible<PromptPos, TokPos>::value && !std::is_constructible<TokPos, PromptPos>::value &&
                  !std::is_constructible<PromptPos, TokCount>::value &&
                  !std::is_constructible<PromptPos, PromptToks>::value &&
                  !std::is_constructible<PromptToks, PromptPos>::value &&
                  !std::is_convertible<int, PromptPos>::value && !std::is_convertible<PromptPos, int>::value,
              "a prompt position is no live position and no count, not even explicitly, and no bare integer");
/* ==== L439: a position on a PROMPT (end) ============================================================================ */
/* ==== L440: offsets into the RENDERED PROMPT and into a renderer's input content (begin) ============================
 * A chat renderer writes one text -- the rendered prompt, the string the tokenizer consumes -- and records the byte
 * ranges it copied from client data (pulsar_text_span), where the tokenizers match no control token.  Those offsets
 * were a bare uint32 pair, beside three other texts' offsets that were bare too: a message's own content (ContentOff,
 * L437), the content a family renderer is handed with the parser's image markers cut out of it (qwen_msg_in /
 * nemotron_msg_in::image_at), and the generated text (GenOff).  None is an offset of another: a message's content
 * lands in the prompt wherever the template has got to, a different place for each message.
 *
 *   PromptOff     an offset into ONE rendered text, in bytes from its first byte: the rendered prompt, or a part of it
 *                 that is rendered on its own and has its own byte 0 (a renderer's piece, a system block, a live
 *                 turn's suffix, the tail of a prompt past a restored chain).  Distances are TextBytes (L334 P7: a
 *                 length of rendered text).  Held in 32 bits, so pulsar_text_span keeps its layout; arithmetic is
 *                 uint32_t's, exactly the `(uint32_t)` casts it replaces (a kit render past UINT32_MAX bytes is
 *                 refused by chat_render_finish).  No sentinel: an offset of a rendered text is never "none".
 *   StrippedOff   an offset into the content a family renderer is HANDED for one message (its `content` field): the
 *                 server's chat_msg::content with the parser's image markers removed.  Where an image sits
 *                 (image_at).  Not a ContentOff -- each marker before it shortens the text by the marker's length --
 *                 and not a PromptOff.
 *
 * The relations, each with one name:
 *   - a part's offset -> the enclosing text's: pulsar_text_span::placed(at), `at` being where the part's first byte
 *     landed -- known only to the renderer that appended it (chat_piece::add, chat_writer::add, buf_spans_carry,
 *     buf_puts_spanned: at = prompt_end() of what is written so far);
 *   - the enclosing text's -> a part's: pulsar_text_span::slice(skip, len) (clipped to the part, rebased to its first
 *     byte: chat_piece::sub, pulsar_text_spans_slice) and cut_to(n) (the text cut to its first n bytes);
 *   - a length of rendered text -> the offset where it ends: PromptOff::after(len), off.before() back.  A chain's
 *     text is a prefix of the prompt it serves, so its TextBytes length is the prompt offset its tail starts at;
 *     TextBytes keeps that doubling (P7) and the span slice takes the prefix as the LENGTH it is;
 *   - ContentOff -> StrippedOff: only chat_strip_image_markers (the server's chat_family.cpp), which cuts the
 *     markers out and so knows what is left before each image;
 *   - PromptOff -> a token index: no arithmetic, only through a tokenizer reading the text with its spans.
 * An offset becomes a buffer index with .raw(), where a tokenizer's byte cursor (a size_t over the whole text) is
 * compared with it: the comparison stays the size_t one it was. */

/** An offset into the text named by `Tag`, held in 32 bits, with TextBytes distances and no sentinel (L440): the
 *  uint32_t a renderer's span map stored, with uint32_t's (modular) arithmetic. */
template <class Tag>
class pulsar_text_off32 {
public:
    /** Uninitialised, as the integer it replaces (value-initialise for offset 0). */
    pulsar_text_off32() = default;
    /** Enter from a raw integer (explicit only; bool and floating point are refused), converted as the `(uint32_t)`
     *  cast at the old field was.
     *  @param v the raw offset */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_text_off32(I v) : v_(static_cast<uint32_t>(v)) {}
    /** The raw offset, where the value leaves the type system (a buffer index, printf, a golden file). */
    /// @return the offset as uint32_t
    constexpr uint32_t raw() const { return v_; }
    /** The offset past the first `n` bytes: where a text of that length ends, and where the next byte written lands.
     *  @param n the bytes before it */
    /// @return offset n (as uint32_t)
    static constexpr pulsar_text_off32 after(TextBytes n) { return pulsar_text_off32(n.raw()); }
    /** The bytes before this offset (it minus offset 0). */
    /// @return the distance from offset 0
    constexpr TextBytes before() const { return TextBytes{v_}; }

    /** An offset moved forward by a length. */
    friend constexpr pulsar_text_off32 operator+(pulsar_text_off32 p, TextBytes d) {
        return pulsar_text_off32(p.v_ + d.raw());
    }
    /** An offset moved back by a length. */
    friend constexpr pulsar_text_off32 operator-(pulsar_text_off32 p, TextBytes d) {
        return pulsar_text_off32(p.v_ - d.raw());
    }
    /** The distance from b to a (uint32_t's: callers that must not wrap compare first, as before). */
    friend constexpr TextBytes operator-(pulsar_text_off32 a, pulsar_text_off32 b) {
        return TextBytes{static_cast<uint32_t>(a.v_ - b.v_)};
    }
    /** Move forward in place. */
    /// @param d the length to move by
    /// @return `*this`
    constexpr pulsar_text_off32 &operator+=(TextBytes d) { return *this = *this + d; }
    /** Move back in place. */
    /// @param d the length to move by
    /// @return `*this`
    constexpr pulsar_text_off32 &operator-=(TextBytes d) { return *this = *this - d; }
    /** Equal. */
    friend constexpr bool operator==(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ == b.v_; }
    /** Not equal. */
    friend constexpr bool operator!=(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ != b.v_; }
    /** Earlier. */
    friend constexpr bool operator<(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ < b.v_; }
    /** Earlier or the same. */
    friend constexpr bool operator<=(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ <= b.v_; }
    /** Later. */
    friend constexpr bool operator>(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ > b.v_; }
    /** Later or the same. */
    friend constexpr bool operator>=(pulsar_text_off32 a, pulsar_text_off32 b) { return a.v_ >= b.v_; }

private:
    uint32_t v_;  ///< the offset, in bytes from the text's first byte
};

/** Tag of PromptOff (never instantiated). */
struct pulsar_prompt_off_tag;
/** Tag of StrippedOff (never instantiated). */
struct pulsar_stripped_off_tag;
/** An offset into one rendered text -- the rendered prompt, or a part of it rendered on its own (L440). */
typedef pulsar_text_off32<pulsar_prompt_off_tag> PromptOff;
/** An offset into the content a family renderer is handed for one message: chat_msg::content less the parser's image
 *  markers (L440). */
typedef pulsar_text_off32<pulsar_stripped_off_tag> StrippedOff;

/** A half-open [lo, hi) byte range of CLIENT-supplied text in a rendered prompt (the engine's public span type: its
 *  tokenizer API is in pulsar.h).  The renderer produces these; the tokeniser treats them as plain text.  The uint32
 *  pair it always was (the static_asserts below), an aggregate: `{lo, hi}`, memcpy, malloc'd arrays as before. */
struct pulsar_text_span {
    PromptOff lo;   ///< first byte
    PromptOff hi;   ///< one past the last byte

    /** A span from raw byte offsets (a test's literal, a fixture's numbers).
     *  @param lo the first byte
     *  @param hi one past the last byte */
    /// @return the span {lo, hi}
    template <class I, class J>
    static constexpr pulsar_text_span of(I lo, J hi) { return {PromptOff{lo}, PromptOff{hi}}; }
    /** The span's length (uint32_t's hi - lo: empty() first where it could wrap). */
    /// @return the bytes from lo to hi
    constexpr TextBytes len() const { return hi - lo; }
    /** Is there no byte in it? */
    /// @return true when hi <= lo
    constexpr bool empty() const { return hi <= lo; }
    /** Is the byte at `p` client data of this span?
     *  @param p an offset of the same text */
    /// @return true when lo <= p < hi
    constexpr bool contains(PromptOff p) const { return lo <= p && p < hi; }
    /** This span of a part, as a span of the text the part was appended to.
     *  @param at where the part's first byte landed in that text */
    /// @return the span moved by the bytes before `at`
    constexpr pulsar_text_span placed(PromptOff at) const { return {lo + at.before(), hi + at.before()}; }
    /** This span after its text is cut to its first `n` bytes.
     *  @param n   the bytes kept
     *  @param out receives the span, its end clipped to n (may be `this`; untouched on false) */
    /// @return false when the span starts at or past the cut: nothing of it is left
    constexpr bool cut_to(TextBytes n, pulsar_text_span *out) const {
        if (lo.before() >= n) return false;
        *out = {lo, hi.before() > n ? PromptOff::after(n) : hi};
        return true;
    }
    /** This span within the part [skip, skip + len) of its text, as a span of that part: clipped to it and rebased
     *  to its first byte.
     *  @param skip the bytes of the text before the part
     *  @param len  the part's length
     *  @param out  receives the span (untouched on false) */
    /// @return false when no byte of the span is in the part
    constexpr bool slice(TextBytes skip, TextBytes len, pulsar_text_span *out) const {
        const TextBytes end = skip + len;
        TextBytes a = lo.before(), b = hi.before();
        if (b <= skip || a >= end) return false;
        if (a < skip) a = skip;
        if (b > end) b = end;
        if (b <= a) return false;
        *out = {PromptOff::after(a - skip), PromptOff::after(b - skip)};
        return true;
    }
};

static_assert(sizeof(PromptOff) == sizeof(uint32_t) && alignof(PromptOff) == alignof(uint32_t) &&
                  std::is_trivial<PromptOff>::value && std::is_standard_layout<PromptOff>::value &&
                  sizeof(StrippedOff) == sizeof(uint32_t) && std::is_trivial<StrippedOff>::value,
              "a prompt or stripped-content offset is the uint32_t it replaces");
static_assert(sizeof(pulsar_text_span) == 2 * sizeof(uint32_t) && alignof(pulsar_text_span) == alignof(uint32_t) &&
                  offsetof(pulsar_text_span, lo) == 0 && offsetof(pulsar_text_span, hi) == sizeof(uint32_t) &&
                  std::is_standard_layout<pulsar_text_span>::value && std::is_trivial<pulsar_text_span>::value &&
                  std::is_aggregate<pulsar_text_span>::value,
              "pulsar_text_span is the {uint32_t lo, hi} pair it was: same size, alignment and member offsets");
static_assert(!std::is_constructible<PromptOff, ContentOff>::value && !std::is_constructible<ContentOff, PromptOff>::value &&
                  !std::is_constructible<PromptOff, StrippedOff>::value &&
                  !std::is_constructible<StrippedOff, PromptOff>::value &&
                  !std::is_constructible<StrippedOff, ContentOff>::value &&
                  !std::is_constructible<ContentOff, StrippedOff>::value &&
                  !std::is_constructible<PromptOff, GenOff>::value && !std::is_constructible<GenOff, PromptOff>::value &&
                  !std::is_constructible<PromptOff, StreamOff>::value &&
                  !std::is_constructible<StreamOff, PromptOff>::value &&
                  !std::is_constructible<PromptOff, BodyOff>::value && !std::is_constructible<BodyOff, PromptOff>::value &&
                  !std::is_constructible<StrippedOff, GenOff>::value && !std::is_constructible<GenOff, StrippedOff>::value,
              "an offset of one text is no offset of another, not even explicitly");
static_assert(!std::is_constructible<PromptOff, TextBytes>::value && !std::is_constructible<TextBytes, PromptOff>::value &&
                  !std::is_constructible<PromptOff, GenBytes>::value &&
                  !std::is_constructible<StrippedOff, TextBytes>::value &&
                  !std::is_constructible<PromptOff, TokPos>::value && !std::is_constructible<TokPos, PromptOff>::value &&
                  !std::is_constructible<PromptOff, TokCount>::value &&
                  !std::is_constructible<TokCount, PromptOff>::value &&
                  !std::is_constructible<PromptOff, PromptToks>::value &&
                  !std::is_constructible<PromptOff, TokId>::value && !std::is_constructible<TokId, PromptOff>::value,
              "a length enters a prompt offset by name only (after / before); a token position, count or id never");
static_assert(!std::is_convertible<uint32_t, PromptOff>::value && !std::is_convertible<PromptOff, uint32_t>::value &&
                  !std::is_convertible<size_t, PromptOff>::value && !std::is_convertible<PromptOff, size_t>::value &&
                  !std::is_convertible<int, PromptOff>::value && !std::is_convertible<uint32_t, StrippedOff>::value &&
                  !std::is_convertible<StrippedOff, uint32_t>::value && !std::is_convertible<int, StrippedOff>::value,
              "no implicit way in or out of a bare integer");
/* ==== L440: offsets into the RENDERED PROMPT and into a renderer's input content (end) ============================== */

/* ==== L438: byte offsets -- into a slab, into a file, into a model map (begin) =====================================
 * L334 P5 gave tensor BYTES one unit, TensorBytes, for a size and for an offset alike ("a distance from the tensor's
 * base"), and that is what the GPU host API takes.  It left the structs that HOLD such numbers raw, so every use wrapped
 * them (`TensorBytes{L.ids}`, ~400 constructions), and in those structs an offset and a size sit side by side, passed
 * positionally: a scratch arena's `ws, ws_bytes` (carved as `base + p.ws, p.ws_bytes`), a checkpoint slot's
 * `off, bytes`, the content sampler's `(fd, off, bytes)`.  A stored offset is its own type here, over the one length:
 *
 *   SlabOff   a byte offset into ONE tensor, slab or scratch arena, from its first byte: where a layout puts a
 *             buffer, where a checkpoint slot or a capture block sits, a walk's cursor.
 *   FileOff   a byte offset into ONE file, from its first byte: where a shard's tensor data starts, where a row
 *             table's rows start, the position a pread takes.
 *   MapOff    a byte offset into ONE model map the GPU host API resolves weights in (pulsar_gpu_set_model_map,
 *             pulsar_gpu_cache_model_range, the register_* keys): a mapped file's (a shard, an overlay donor) -- what
 *             pulsar_tensor::abs_offset holds and every weight-offset parameter of pulsar_gpu.h takes -- or a key
 *             under a map the engine registers built weights by (a TP half's, pulsar_tp_*_key_offset), which no file
 *             holds.  ~90 parameters of pulsar_gpu.h took it as a bare uint64_t beside sizes, widths and counts.
 *   All three measure in TensorBytes: the bytes a file stores for a tensor ARE the tensor's bytes (weights are mapped
 *   or read in place, never converted), so a tensor's extent is the one length every offset moves by.  A file mapped
 *   whole from its first byte has one number for both of its offsets; the model layer, which maps it, owns that
 *   crossing (pulsar_engine_internal.h pulsar_map_off_of_file / pulsar_file_off_of_map).
 *
 * off + bytes -> off, off - bytes -> off, off - off -> bytes, ordering and equality within one space; an offset is no
 * size and no offset of the other space, not even explicitly (pulsar_text_off's rules: the same affine pair as a
 * text's offsets, one size_t).  The crossings:
 *   off.before()            the bytes before the offset: what the GPU host API's offset parameter is (P5: an offset
 *                           there is a distance from the base), and what a refusal prints;
 *   Off::after(bytes)       the offset past that many bytes (a cursor from a running size);
 *   pulsar_slab_at(base, o) the address `o` names in a mapped slab -- the ONE place a slab offset meets a pointer (a
 *                           kernel launch's carve of its scratch arena);
 *   .raw()                  where the number leaves the type system: pread's off_t, a printf, a digest, a file field.
 * none() (raw SIZE_MAX) is "no offset", as the text offsets': arithmetic and ordering on it, or landing on it, abort.
 * No field typed here held that value as a sentinel (a layout's absent buffer is a zero SIZE, never an offset).
 *
 * Left raw on purpose: a struct that is serialized, hashed by its fields' bytes or handed to a kernel keeps the integer
 * its layout names (pulsar_content_read::bytes is digested as a uint32; a kernel's descriptor is the kernel's); the
 * segment store's file arithmetic (L334 P7: out of scope by decision); a payload's own length, which is a stream's. */

/** Tag of SlabOff (never instantiated). */
struct pulsar_slab_off_tag;
/** Tag of FileOff (never instantiated). */
struct pulsar_file_off_tag;
/** A byte offset into one tensor, slab or scratch arena, with TensorBytes distances (L438). */
typedef pulsar_text_off<pulsar_slab_off_tag, TensorBytes> SlabOff;
/** A byte offset into one file, with TensorBytes distances (L438). */
typedef pulsar_text_off<pulsar_file_off_tag, TensorBytes> FileOff;
/** Tag of MapOff (never instantiated). */
struct pulsar_map_off_tag;
/** A byte offset into one model map the GPU host API resolves weights in, with TensorBytes distances (L438). */
typedef pulsar_text_off<pulsar_map_off_tag, TensorBytes> MapOff;

/** The address a slab offset names in a mapped slab: `base + off`, as a `T *` -- the one crossing from a slab offset to
 *  a pointer (a launch's carve of its scratch arena; fatal on none()).
 *  @param base the slab's first byte (a device or host mapping)
 *  @param off  the offset into it */
/// @return the address, as `T *`
template <class T = uint8_t>
inline T *pulsar_slab_at(void *base, SlabOff off) {
    return reinterpret_cast<T *>(static_cast<uint8_t *>(base) + off.before().raw());
}
/** pulsar_slab_at on a read-only mapping.
 *  @param base the slab's first byte
 *  @param off  the offset into it */
/// @return the address, as `const T *`
template <class T = uint8_t>
inline const T *pulsar_slab_at(const void *base, SlabOff off) {
    return reinterpret_cast<const T *>(static_cast<const uint8_t *>(base) + off.before().raw());
}

/** `b` rounded up to a multiple of `align` (a power of two): the named form of `(b + align - 1) & ~(align - 1)`, what
 *  the layouts' 256-byte slots and the checkpoint slots are sized by.
 *  @param b     the bytes
 *  @param align the alignment, a power of two */
/// @return the rounded bytes
constexpr TensorBytes pulsar_bytes_align(TensorBytes b, uint64_t align) {
    return TensorBytes{(b.raw() + (align - 1u)) & ~(align - 1u)};
}

static_assert(sizeof(size_t) == sizeof(uint64_t) && sizeof(SlabOff) == sizeof(uint64_t) &&
                  sizeof(FileOff) == sizeof(uint64_t) && alignof(SlabOff) == alignof(uint64_t) &&
                  alignof(FileOff) == alignof(uint64_t) && sizeof(MapOff) == sizeof(uint64_t) &&
                  alignof(MapOff) == alignof(uint64_t) && std::is_trivial<MapOff>::value &&
                  std::is_standard_layout<MapOff>::value && std::is_trivial<SlabOff>::value &&
                  std::is_standard_layout<SlabOff>::value && std::is_trivial<FileOff>::value &&
                  std::is_standard_layout<FileOff>::value,
              "a slab or file offset is as cheap as the uint64_t it replaces (memset / calloc-safe)");
static_assert(!std::is_constructible<SlabOff, FileOff>::value && !std::is_constructible<FileOff, SlabOff>::value &&
                  !std::is_constructible<SlabOff, TensorBytes>::value &&
                  !std::is_constructible<TensorBytes, SlabOff>::value &&
                  !std::is_constructible<FileOff, TensorBytes>::value &&
                  !std::is_constructible<TensorBytes, FileOff>::value &&
                  !std::is_constructible<SlabOff, KvBytes>::value && !std::is_constructible<FileOff, KvBytes>::value &&
                  !std::is_constructible<SlabOff, TextBytes>::value && !std::is_constructible<FileOff, TextBytes>::value &&
                  !std::is_constructible<SlabOff, GenOff>::value && !std::is_constructible<FileOff, ContentOff>::value &&
                  !std::is_constructible<GenOff, SlabOff>::value && !std::is_constructible<ContentOff, FileOff>::value &&
                  !std::is_constructible<MapOff, FileOff>::value && !std::is_constructible<FileOff, MapOff>::value &&
                  !std::is_constructible<MapOff, SlabOff>::value && !std::is_constructible<SlabOff, MapOff>::value &&
                  !std::is_constructible<MapOff, TensorBytes>::value &&
                  !std::is_constructible<TensorBytes, MapOff>::value,
              "an offset of one space is no offset of another and no size, not even explicitly");
static_assert(!std::is_convertible<uint64_t, SlabOff>::value && !std::is_convertible<SlabOff, uint64_t>::value &&
                  !std::is_convertible<uint64_t, FileOff>::value && !std::is_convertible<FileOff, uint64_t>::value &&
                  !std::is_convertible<int, SlabOff>::value && !std::is_convertible<int, FileOff>::value &&
                  !std::is_convertible<uint64_t, MapOff>::value && !std::is_convertible<MapOff, uint64_t>::value,
              "no implicit way in or out of a bare integer");
/* P5 / P7 closed with no such assert for the byte units themselves (L435: "TensorBytes / TextBytes have no
 * non-convertibility assert"): the three byte units never convert to one another or to a row count, even explicitly --
 * kv_bytes_of is the one crossing, and it is a function */
static_assert(!std::is_constructible<TensorBytes, KvBytes>::value && !std::is_constructible<KvBytes, TensorBytes>::value &&
                  !std::is_constructible<TensorBytes, TextBytes>::value &&
                  !std::is_constructible<TextBytes, TensorBytes>::value &&
                  !std::is_constructible<KvBytes, TextBytes>::value && !std::is_constructible<TextBytes, KvBytes>::value &&
                  !std::is_constructible<TensorBytes, KvRows>::value && !std::is_constructible<KvRows, TensorBytes>::value &&
                  !std::is_constructible<TensorBytes, GenBytes>::value &&
                  !std::is_constructible<TensorBytes, TokCount>::value &&
                  !std::is_constructible<TensorBytes, CompRows>::value &&
                  !std::is_convertible<uint64_t, TensorBytes>::value && !std::is_convertible<TensorBytes, uint64_t>::value &&
                  !std::is_convertible<uint64_t, TextBytes>::value && !std::is_convertible<TextBytes, uint64_t>::value,
              "tensor, KV and text bytes are three units: none converts to another, to a row count or to an integer");
/* ==== L438: byte offsets -- into a slab, into a file, into a model map (end) ======================================= */

/* ==== L442: host and device memory ================================================================================
 * KvBytes carried three memories until L442 (L391's note, L435 S2-5): the host's (MemAvailable, the zones' high
 * watermarks, the memory floor, a persist's dirty pages, a discrete card's pinned checkpoints), the device's
 * (cudaMemGetInfo's free memory, a discrete card's working set, margin and guard floor) and the KV accounting's own
 * prices.  On the GB10 the three are one pool of unified memory; on a discrete card (CT 130's RTX 6000, CT 131's
 * RTX 5000) the host and the device are two, and a KV need tested against the host's MemAvailable read the wrong one
 * (L391's S1-1: provision_bank and the three per-chunk prefill checks).  Each memory is now its own unit:
 *
 *   HostBytes    bytes of the host's RAM: a /proc/meminfo reading, the memory floor and its terms, what a persist or
 *                a pinned host slab takes, a host mapping (the weights' mmap).
 *   DeviceBytes  bytes of the device's memory: cudaMemGetInfo's free / total, a discrete card's working set, its
 *                admission margin and its guard floor; the free memory a KV headroom check reads.
 *   KvBytes      (above) the KV accounting: a price, a budget, the ledger, touched and reserved KV.
 *
 * Where a KV price lives: in the DEVICE's memory, on every platform -- a discrete card's VRAM, or on the GB10 the
 * unified pool the device allocates the KV from (cudaMallocManaged).  So a KV quantity meets a memory reading only as
 * DeviceBytes, through the one relation pulsar_kv_device_bytes (a price as the device memory it takes) and its
 * inverse pulsar_kv_room_bytes (device memory as room for KV); a KV price never meets a host reading.
 * Where host and device are one pool (unified memory: pulsar_gpu_integrated() != 0) a host reading IS the device's:
 * pulsar_unified_device_bytes / pulsar_unified_host_bytes are that one crossing, and are called only on unified
 * memory's paths (each call site says why it is one).  On a discrete card nothing converts between the two.
 * Every crossing is the identity on the value: the split changes no number, only which comparisons compile. */
/** Tag of HostBytes (never instantiated). */
struct pulsar_host_bytes_tag;
/** Tag of DeviceBytes (never instantiated). */
struct pulsar_device_bytes_tag;
/** Bytes of the host's RAM: a MemAvailable reading, the memory floor, a persist's or a pinned slab's bytes (L442). */
typedef pulsar_unit<pulsar_host_bytes_tag> HostBytes;
/** Bytes of the device's memory: cudaMemGetInfo's readings, a discrete card's working set and floors (L442). */
typedef pulsar_unit<pulsar_device_bytes_tag> DeviceBytes;

/** A KV price as the device memory it occupies -- the KV lives in the device's memory on every platform (on the GB10
 *  the unified pool).  The one way a KV quantity meets a memory reading.
 *  @param b the KV bytes */
/// @return the same bytes as DeviceBytes
constexpr DeviceBytes pulsar_kv_device_bytes(KvBytes b) { return DeviceBytes{b.raw()}; }
/** Device memory as room for KV: the inverse of pulsar_kv_device_bytes (a headroom reading turned into a KV budget).
 *  @param b the device bytes */
/// @return the same bytes as KvBytes
constexpr KvBytes pulsar_kv_room_bytes(DeviceBytes b) { return KvBytes{b.raw()}; }
/** A host reading as the device's, where the two are ONE pool: unified memory only (pulsar_gpu_integrated() != 0;
 *  on the GB10 MemAvailable is what the device can still allocate).  Never on a discrete card's path.
 *  @param b the host bytes */
/// @return the same bytes as DeviceBytes
constexpr DeviceBytes pulsar_unified_device_bytes(HostBytes b) { return DeviceBytes{b.raw()}; }
/** The inverse of pulsar_unified_device_bytes, under the same condition: unified memory only.
 *  @param b the device bytes */
/// @return the same bytes as HostBytes
constexpr HostBytes pulsar_unified_host_bytes(DeviceBytes b) { return HostBytes{b.raw()}; }

/** A host slab's tensor bytes as the host memory they take (a pinned checkpoint slab's extent): the one crossing
 *  from TensorBytes into HostBytes, beside kv_bytes_of.
 *  @param b the tensor bytes, of a host allocation */
/// @return the same bytes as HostBytes
constexpr HostBytes host_bytes_of(TensorBytes b) { return HostBytes{b.raw()}; }

/** Bytes per GiB: the one constant every GiB figure in a log line or a test is divided or built by (L442; the series
 *  chose one shared constant over a type for GiB doubles). */
constexpr double PULSAR_GIB = 1024.0 * 1024.0 * 1024.0;
/** Bytes per MiB, the same for MiB figures. */
constexpr double PULSAR_MIB = 1024.0 * 1024.0;
/** A byte count in GiB, for a log line: `(double)b / PULSAR_GIB`.
 *  @param b the bytes */
/// @return GiB
constexpr double pulsar_gib(uint64_t b) { return (double)b / PULSAR_GIB; }
/** A byte unit's quantity in GiB, for a log line (KvBytes, HostBytes, DeviceBytes, TensorBytes alike).
 *  @param b the bytes */
/// @return GiB
template <class Tag>
constexpr double pulsar_gib(pulsar_unit<Tag, uint64_t> b) { return (double)b.raw() / PULSAR_GIB; }
/** A byte count in MiB, for a log line: `(double)b / PULSAR_MIB`.
 *  @param b the bytes */
/// @return MiB
constexpr double pulsar_mib(uint64_t b) { return (double)b / PULSAR_MIB; }
/** A byte unit's quantity in MiB, for a log line.
 *  @param b the bytes */
/// @return MiB
template <class Tag>
constexpr double pulsar_mib(pulsar_unit<Tag, uint64_t> b) { return (double)b.raw() / PULSAR_MIB; }

static_assert(sizeof(HostBytes) == sizeof(uint64_t) && sizeof(DeviceBytes) == sizeof(uint64_t) &&
                  std::is_trivial<HostBytes>::value && std::is_standard_layout<HostBytes>::value &&
                  std::is_trivial<DeviceBytes>::value && std::is_standard_layout<DeviceBytes>::value,
              "host and device bytes are as cheap as the uint64_t they replace (memset / calloc-safe)");
static_assert(!std::is_constructible<HostBytes, DeviceBytes>::value &&
                  !std::is_constructible<DeviceBytes, HostBytes>::value &&
                  !std::is_constructible<HostBytes, KvBytes>::value && !std::is_constructible<KvBytes, HostBytes>::value &&
                  !std::is_constructible<DeviceBytes, KvBytes>::value &&
                  !std::is_constructible<KvBytes, DeviceBytes>::value &&
                  !std::is_constructible<HostBytes, TensorBytes>::value &&
                  !std::is_constructible<DeviceBytes, TensorBytes>::value &&
                  !std::is_constructible<HostBytes, KvRows>::value && !std::is_constructible<DeviceBytes, KvRows>::value,
              "host, device and KV bytes are three memories: none converts to another except by a named function");
static_assert(!std::is_convertible<uint64_t, HostBytes>::value && !std::is_convertible<HostBytes, uint64_t>::value &&
                  !std::is_convertible<uint64_t, DeviceBytes>::value &&
                  !std::is_convertible<DeviceBytes, uint64_t>::value,
              "no implicit way in or out of a bare integer");
/* a host reading compared with a device reading, or a KV price with either, does not compile: the operators are
 * per unit (pulsar_unit's friends take two of the same type) */
/** Whether `A < B` compiles (the static_assert below): false unless an operator takes the two. */
template <class A, class B, class = void>
struct pulsar_units_comparable_ : std::false_type {};
/** `A < B` compiles. */
template <class A, class B>
struct pulsar_units_comparable_<A, B, decltype((void)(std::declval<A>() < std::declval<B>()))> : std::true_type {};
static_assert(!pulsar_units_comparable_<HostBytes, DeviceBytes>::value &&
                  !pulsar_units_comparable_<KvBytes, HostBytes>::value &&
                  !pulsar_units_comparable_<KvBytes, DeviceBytes>::value &&
                  pulsar_units_comparable_<DeviceBytes, DeviceBytes>::value,
              "a reading is compared only with a reading of the same memory");
/* ==== L442: host and device memory (end) ========================================================================== */

#endif

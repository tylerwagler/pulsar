/** @file
 *  Typed quantities that share a raw integer type but must never mix (L333, L334): KvRows, KvBytes, TokCount, PromptToks, TokPos */
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
 *   KvBytes  bytes of the unified memory the KV accounting prices: a bank's touched pages, a growth price, the
 *            guard's budget and ceiling, the memory floor and the MemAvailable reading the guard turns into KV room.
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
 *   PromptToks has no point type: no prompt-side position crosses the API yet.
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <type_traits>

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
/** Bytes of the unified memory the KV accounting prices. */
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

#endif

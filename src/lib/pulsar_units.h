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
 *   LiveToks    tokens of the live side: a bank's history (its KV rows, cut points, rewind targets, checkpoints) and
 *               the stitched prompt a sync runs, which keeps the live tokens below the cut.
 *   PromptToks  tokens of the request's prompt as the client's bytes tokenised (accounting, prefill bounds, the
 *               prompt tokens a bank's state serves).
 *   Both count from position 0, so a count is also the position it ends at.  The one conversion is the stitch's
 *   (pulsar_prefix_match_stitched_len): past the cut the stitched prompt carries the prompt's own tokens, so a
 *   prompt-side distance there is a live-side one.
 *
 * The rules, enforced by the compiler:
 *   - no implicit conversion in or out: a value enters by an explicit constructor (KvBytes{n}) where it is MEASURED
 *     or CONFIGURED (a /proc read, the ledger's figure, a constant, a token array's length) and leaves by .raw() only
 *     where it leaves the type system (a printf, /metrics, JSON, an array index, an untyped caller);
 *   - arithmetic and comparison only within a type (+, -, the six comparisons), with exactly the representation's
 *     semantics (`Rep`: uint64_t's unsigned modular arithmetic for the KV accounting, int's for the token currencies,
 *     the fields they replaced), so a typed expression computes what the raw one did;
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
    constexpr Rep raw() const { return v_; }
    /** The largest representable quantity (the saturation bound the overflow checks compare against). */
    static constexpr pulsar_unit max() { return pulsar_unit(std::numeric_limits<Rep>::max()); }
    /** `n` copies of this quantity (n banks' growth, n pages): `Rep` multiplication.
     *  @param n how many */
    constexpr pulsar_unit times(Rep n) const { return pulsar_unit(v_ * n); }

    /** Sum (as `Rep`: modular for an unsigned one). */
    friend constexpr pulsar_unit operator+(pulsar_unit a, pulsar_unit b) { return pulsar_unit(a.v_ + b.v_); }
    /** Difference (as `Rep`: modular for an unsigned one, so callers that must not wrap compare first, as before). */
    friend constexpr pulsar_unit operator-(pulsar_unit a, pulsar_unit b) { return pulsar_unit(a.v_ - b.v_); }
    /** Add in place. */
    constexpr pulsar_unit &operator+=(pulsar_unit b) { v_ += b.v_; return *this; }
    /** Subtract in place. */
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

/** Tag of LiveToks (never instantiated). */
struct pulsar_live_toks_tag;
/** Tag of PromptToks (never instantiated). */
struct pulsar_prompt_toks_tag;

/** Tokens of the live side of a token seam: a bank's history and the stitched prompt a sync runs (L334). */
typedef pulsar_unit<pulsar_live_toks_tag, int> LiveToks;
/** Tokens of the request's prompt, the client's bytes as tokenised, on the other side of a token seam (L334). */
typedef pulsar_unit<pulsar_prompt_toks_tag, int> PromptToks;

static_assert(sizeof(KvRows) == sizeof(uint64_t) && sizeof(KvBytes) == sizeof(uint64_t), "a unit is one uint64_t");
static_assert(std::is_trivially_copyable<KvBytes>::value && std::is_trivial<KvBytes>::value &&
                  std::is_standard_layout<KvBytes>::value,
              "a unit is as cheap as the uint64_t it replaces (memset / calloc-safe)");
static_assert(sizeof(LiveToks) == sizeof(int) && sizeof(PromptToks) == sizeof(int) &&
                  std::is_trivial<LiveToks>::value && std::is_standard_layout<PromptToks>::value,
              "a token currency is as cheap as the int it replaces");

#endif

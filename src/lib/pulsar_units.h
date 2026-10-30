#ifndef PULSAR_UNITS_H
#define PULSAR_UNITS_H

/* pulsar_units.h -- the KV accounting's units as distinct types (L333).
 *
 * The guard, the admission verdicts, the memory floor and the families' touched / growth prices once carried every
 * quantity as a bare uint64_t, and whether a number was token rows or bytes lived only in comments.  L332's VMM
 * series switched the guard's "touched" from the rows high-water to mapped chunk bytes and nothing complained;
 * admission then took paths the managed run never took (a 500, a decoding bank counted as 0).  Here a quantity's
 * unit is its type:
 *
 *   KvRows   token rows of KV: a bank's high-water, a prompt's length, a decode quantum, a page of the disk grid --
 *            what a family's pools grow by, one row per token.
 *   KvBytes  bytes of the unified memory the KV accounting prices: a bank's touched pages, a growth price, the
 *            guard's budget and ceiling, the memory floor and the MemAvailable reading the guard turns into KV room.
 *
 * The rules, enforced by the compiler:
 *   - no implicit conversion in or out: a value enters by an explicit constructor (KvBytes{n}) where it is MEASURED
 *     or CONFIGURED (a /proc read, the ledger's figure, a constant) and leaves by .raw() only where it leaves the
 *     type system (a printf, /metrics, JSON);
 *   - arithmetic and comparison only within a type (+, -, the six comparisons), with exactly uint64_t's semantics
 *     (unsigned, modular: every caller's range checks are its own, as before), so a typed expression computes the
 *     bytes the raw one did;
 *   - a count of identical quantities scales by name (q.times(n): n banks' growth, n pages);
 *   - rows become bytes only through the family's one authority, which takes the rows and its own factor -- the
 *     session's pools (pulsar_session_quantum_growth_bytes_per_bank, family.h growth_bytes) or the declared plan
 *     (state_kinds.h pulsar_state_pool_bytes_at).  There is no scalar bytes-per-row: each pool rounds its own
 *     ceil(rows / tokens-per-row) and DeepSeek each layer its ceil(rows / compress ratio), so a "rows * bpr" helper
 *     would misstate the price; none exists, and the inverse (bytes to rows) is asked nowhere.
 *
 * Zero cost: one uint64_t, trivially copyable, standard layout (the static_asserts below), so a struct holding one
 * may still be zeroed with memset / calloc and a default-constructed value is as uninitialised as the uint64_t it
 * replaced (value-initialise, KvBytes{}, for 0).  Header-only C++17 (nvcc's default, for pulsar.h's .cu host
 * includes); no exceptions, no RTTI. */

#include <stdint.h>
#include <type_traits>

/** One quantity of the unit named by `Tag`: a uint64_t that converts to nothing and from nothing implicitly. */
template <class Tag>
class pulsar_unit {
public:
    /** Uninitialised, as the uint64_t it replaces (value-initialise for 0). */
    pulsar_unit() = default;
    /** Enter the unit from a raw integer (explicit only; bool and floating point are refused at compile time).
     *  @param v the raw count, converted to uint64_t exactly as an assignment to the old uint64_t field was */
    template <class I, typename std::enable_if<std::is_integral<I>::value && !std::is_same<I, bool>::value, int>::type = 0>
    constexpr explicit pulsar_unit(I v) : v_(static_cast<uint64_t>(v)) {}
    /** The raw count, where the value leaves the type system (printf, /metrics, JSON). */
    constexpr uint64_t raw() const { return v_; }
    /** The largest representable quantity (the saturation bound the overflow checks compare against). */
    static constexpr pulsar_unit max() { return pulsar_unit(UINT64_MAX); }
    /** `n` copies of this quantity (n banks' growth, n pages): uint64_t multiplication.
     *  @param n how many */
    constexpr pulsar_unit times(uint64_t n) const { return pulsar_unit(v_ * n); }

    /** Sum (modular, as uint64_t). */
    friend constexpr pulsar_unit operator+(pulsar_unit a, pulsar_unit b) { return pulsar_unit(a.v_ + b.v_); }
    /** Difference (modular, as uint64_t: callers that must not wrap compare first, as before). */
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
    uint64_t v_;  ///< the count, in the unit `Tag` names
};

/** Tag of KvRows (never instantiated). */
struct pulsar_kv_rows_tag;
/** Tag of KvBytes (never instantiated). */
struct pulsar_kv_bytes_tag;

/** Token rows of KV (one row per token a bank's pools hold or will hold). */
typedef pulsar_unit<pulsar_kv_rows_tag> KvRows;
/** Bytes of the unified memory the KV accounting prices. */
typedef pulsar_unit<pulsar_kv_bytes_tag> KvBytes;

static_assert(sizeof(KvRows) == sizeof(uint64_t) && sizeof(KvBytes) == sizeof(uint64_t), "a unit is one uint64_t");
static_assert(std::is_trivially_copyable<KvBytes>::value && std::is_trivial<KvBytes>::value &&
                  std::is_standard_layout<KvBytes>::value,
              "a unit is as cheap as the uint64_t it replaces (memset / calloc-safe)");

#endif

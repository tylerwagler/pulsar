/* sampler_host_ref.cpp -- THE HOST SAMPLER as it stood on dev c81332fe, frozen: the reference the device sampler
 * (src/cuda/pulsar_cuda_sampler.cu, L302) is held to by tests/sampler_device_gate.cpp.
 *
 * The functions below are tokenizer.cpp's sampler and session.cpp's logprob scorers VERBATIM (only renamed into
 * namespace sref, with the few engine helpers they called defined beside them), built with the engine's own
 * CXXFLAGS (-O3 -ffast-math -march=native): the bytes the host produced came from THAT compilation -- its
 * reciprocal multiplies, its contractions, glibc's expf / exp / log -- and the device reproduces them.  When the
 * host path left the engine (L302), this file kept its compiled behaviour as the oracle; the gate's first run
 * proved it byte-identical to the engine's own sampler on every case before the engine's copy was deleted.
 *
 * Do not "fix" or modernise anything here: it is a record of what the engine did. */
#include "sampler_host_ref.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace sref {

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) abort();
    return p;
}

#define PULSAR_SAMPLE_SPARSE_MINP_MIN 0.02f
#define PULSAR_NEG_INF (-1.0e30f)
static inline bool pulsar_sample_sparse_minp(float min_p, uint32_t n_vocab) {
    const float band = 2.5f * 5.9604645e-8f * (float)n_vocab;
    return min_p >= (band > PULSAR_SAMPLE_SPARSE_MINP_MIN ? band : PULSAR_SAMPLE_SPARSE_MINP_MIN);
}

static inline bool sample_finite_bits(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof(b));
    return (b & 0x7f800000u) != 0x7f800000u;
}

int sample_argmax(const float *logits, uint32_t n_vocab) {
    constexpr uint32_t BLOCK = 256u, LANES = 8u;
    int best = -1;
    float best_v = 0.0f;
    for (uint32_t b0 = 0; b0 < n_vocab; b0 += BLOCK) {
        const uint32_t b1 = n_vocab - b0 < BLOCK ? n_vocab : b0 + BLOCK;
        float acc[LANES];
        for (uint32_t k = 0; k < LANES; k++) acc[k] = -INFINITY;
        uint32_t i = b0;
        for (; i + LANES <= b1; i += LANES) {
            for (uint32_t k = 0; k < LANES; k++) {
                const float v = logits[i + k];
                const float f = sample_finite_bits(v) ? v : -INFINITY;
                acc[k] = f > acc[k] ? f : acc[k];
            }
        }
        for (; i < b1; i++) {
            const float v = logits[i];
            const float f = sample_finite_bits(v) ? v : -INFINITY;
            acc[0] = f > acc[0] ? f : acc[0];
        }
        float m = acc[0];
        for (uint32_t k = 1; k < LANES; k++) m = acc[k] > m ? acc[k] : m;
        if (m == -INFINITY || (best >= 0 && !(m > best_v))) continue;   /* no finite, or no rise */
        for (uint32_t j = b0; j < b1; j++) {
            if (logits[j] == m) {
                best = (int)j;
                best_v = m;
                break;
            }
        }
    }
    return best;
}



/* The request's sampler stream (L288 sdet): a Weyl state.  The KEY of the token `ahead` places past the stream's
 * next undrawn one is mix(state + (ahead + 1) * STEP) -- a function of (seed, token index) only, so a lane reads a
 * token's key without consuming anything, and a token's draw is the same whichever lane makes it, whatever was
 * drafted around it (session_spec.cpp's walk is why that matters).  Every committed token advances the state one
 * STEP (pulsar_sample_advance).
 *
 * STEP is deliberately NOT SplitMix64's golden-ratio gamma: two seeds that differ by k steps share their streams
 * shifted by k tokens, and seeds spaced by that gamma are a common idiom -- the spec-sampling oracle seeds its
 * trajectories gamma * (t + 1), which made trajectory t's token i + 1 draw at trajectory t + 1's token i key and
 * inflated its chi-squares with correlated samples.  An arbitrary odd constant puts any seed family nobody builds
 * from it ~2^63 steps apart. */
#define SAMPLE_STREAM_STEP 0xa0761d6478bd642fULL

static uint64_t sample_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t pulsar_sample_key(uint64_t rng, uint32_t ahead) {
    return sample_mix64(rng + ((uint64_t)ahead + 1u) * SAMPLE_STREAM_STEP);
}

void pulsar_sample_advance(uint64_t *rng, uint32_t n) {
    *rng += (uint64_t)n * SAMPLE_STREAM_STEP;
}



/* IEEE-754 float32 -> uint32 whose UNSIGNED ASCENDING order is the float's
 * DESCENDING order: flip negatives entirely, set the sign bit on
 * non-negatives (the standard monotonic transform, giving float-ascending),
 * then complement to reverse it. Only finite values reach here — the callers
 * filter !isfinite — so NaN ordering is not a concern.
 *
 * -0.0f is canonicalized to +0.0f first: it is a distinct bit pattern but
 * compares EQUAL to +0.0f, so the comparator this replaces called them a
 * tie, and by the tie note below, ordering what it tied can change the
 * nucleus. Real, not hypothetical: logits arrive by memcpy from the GPU, and
 * device code is built --use_fast_math (FTZ), so it writes exactly this.
 *
 * Subnormals are deliberately NOT flushed, and the reasoning is worth
 * recording because it is a trap. -ffast-math would set FPCR.FZ (flushing
 * subnormal FCMP inputs, making the comparator tie them) — but ONLY via
 * crtfastmath.o, which the LINKER pulls in, and every binary carrying this
 * code (ds4, pulsar-server, pulsar_test) is linked by nvcc, not by gcc -ffast-math.
 * Measured in the real link config: FPCR = 0x0, FZ = 0, and the comparator
 * ORDERS subnormals. So keying them strictly is what matches. A gcc-linked
 * probe of the same source reports FPCR = 0x1000000 and the opposite answer —
 * do not test this outside the real linkage. tests/pulsar_test.cpp --sampler
 * covers it (shape "subnormals + zeros (FZ range)"). */
static inline uint32_t sample_desc_key(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if (u == 0x80000000u) u = 0u;   /* -0.0 == +0.0 to the comparator */
    const uint32_t asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ~asc;
}



/* Stable LSD radix sort of packed (sample_desc_key << 32 | id) on the key
 * half — 4 byte passes, O(n), no comparator calls. Replaces a qsort over the
 * whole 129k-entry vocab that cost ~10.6 ms per call (~97% of dist_build) and
 * ran once per accepted position in the sampled speculative walk.
 *
 * Order is bit-for-bit what the qsort produced, ties included.
 *
 * STABILITY IS LOAD-BEARING — do not swap in an unstable sort (parallel,
 * in-place, MSD) on the theory that ties are harmless. They are not. It is
 * true that permuting tied candidates permutes IDENTICAL prob values, so
 * `sum`, `min_prob`, `filtered`, `filtered_sum` and every out->probs entry
 * are bit-invariant under any tie order. But `filtered` is a COUNT, and a tie
 * group straddling the cutoff changes which ids fall inside it — i.e. nucleus
 * MEMBERSHIP, not just order within the nucleus. Concretely, all-equal logits
 * with top_p=0.5 give filtered=64640 of 129280: ascending ties yield
 * ids [0..64639], descending ties yield ids [64640..129279] — disjoint.
 * A draw at the same key then races a different candidate set, so the
 * speculative walk would REJECT a draft the other order ACCEPTS, changing the
 * emitted token stream.
 *
 * So: the caller fills `a` in ascending-id order and this sort is stable,
 * making ascending-id the canonical tie order. That also matches what the
 * replaced qsort produced at this size (glibc takes its stable msort_with_tmp
 * path for a 129280 x 12B array -- verified), but nothing here depends on that
 * unspecified detail: this sort is stable by construction and libc-independent.
 * tests/pulsar_test.cpp --sampler pins the order explicitly (and catches a
 * tie-order flip on a realistic shape -- ties are common, not adversarial).
 *
 * Requires n >= 1. */
static void sample_radix_sort_desc(uint64_t *a, uint64_t *tmp, uint32_t n) {
    uint32_t hist[4][256];
    memset(hist, 0, sizeof(hist));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t k = (uint32_t)(a[i] >> 32);
        hist[0][k & 0xffu]++;
        hist[1][(k >> 8) & 0xffu]++;
        hist[2][(k >> 16) & 0xffu]++;
        hist[3][(k >> 24) & 0xffu]++;
    }
    uint64_t *src = a;
    uint64_t *dst = tmp;
    for (int pass = 0; pass < 4; pass++) {
        const int sh = 32 + pass * 8;
        /* Whole pass is a no-op when every element shares this key byte —
         * e.g. the all-equal-logits degenerate case skips all four. */
        if (hist[pass][(uint32_t)(src[0] >> sh) & 0xffu] == n) continue;
        uint32_t off[256];
        uint32_t run = 0;
        for (int b = 0; b < 256; b++) {
            off[b] = run;
            run += hist[pass][b];
        }
        for (uint32_t i = 0; i < n; i++)
            dst[off[(uint32_t)(src[i] >> sh) & 0xffu]++] = src[i];
        uint64_t *swap = src;
        src = dst;
        dst = swap;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(*a));
}



/* =====
 * min-p prefilter threshold (pulsar_sample_dist_build's full-vocab path; the
 * device prefilter's contract, pulsar_sample_dist_build_prefiltered, is the
 * same superset argument in logit units).
 *
 * The min-p cutoff every path applies post-sort keeps candidate i (i > 0) iff
 *
 *     fl(p_i) >= fl(p_max * min_p)
 *
 * where p_max = expf((max - max)/T) = expf(0.0f) == 1.0f EXACTLY, so the
 * threshold IS min_p and the decision depends on nothing but p_i.
 *
 * L149 (2026-09-02): until then the comparison was
 *     fl(p_i / sum) >= fl(fl(p_max / sum) * min_p)
 * with `sum` the full-vocab normaliser -- the same condition in exact
 * arithmetic, but three roundings wide, and it made every min-p decision
 * depend on a 129k-term sum that existed only to cancel out. On the sampled
 * speculative path that sum was the 129k host expf calls the GPU idled
 * behind for ~630 us per draft position. The division-free form decides the
 * boundary from p_i alone, which is what lets a DEVICE prefilter hand the
 * sampler the few candidates above the floor instead of the full row
 * (pulsar_sample_dist_build_prefiltered). Rounding-level change: a candidate
 * whose p_i sits within ~3 ulp of min_p could be classed differently than
 * before; the emitted distribution is otherwise identical.
 *
 * A logit-side prefilter that keeps p_i >= min_p * (1 - 4e-6) keeps a
 * SUPERSET of whatever the cutoff keeps. Because probs are monotone in the
 * logit, that superset is a PREFIX of the descending sort -- so the cutoff
 * loop, run unchanged over the sorted survivors, walks exactly the
 * candidates it would have walked over the full sorted vocab and trims the
 * boundary with the SAME float comparison. Membership is never decided by
 * the prefilter.
 *
 * The max candidate is kept unconditionally (mirrors the loop's i > 0
 * exemption; also covers min_p > 1 and a NaN temperature poisoning p).
 *
 * DOMAIN: min_p is NOT range-checked server-side; both paths take the
 * prefilter only for min_p > SAMPLE_MINP_PREFILTER_MIN and fall back to the
 * exact full sort below it. Requests with 0 < min_p <= 1e-30 are absurd but
 * stay correct. */
#define SAMPLE_MINP_PREFILTER_SLACK (1.0f - 4e-6f)
#define SAMPLE_MINP_PREFILTER_MIN 1e-30f



static void sample_scratch_reserve(pulsar_sample_scratch *s, uint32_t cap) {
    if (s->cap >= cap) return;
    pulsar_sample_scratch_free(s);
    s->cand = (sample_candidate *)xmalloc((size_t)cap * sizeof(*s->cand));
    s->keys = (uint64_t *)xmalloc((size_t)cap * sizeof(*s->keys));
    s->tmp = (uint64_t *)xmalloc((size_t)cap * sizeof(*s->tmp));
    s->cand2 = (sample_candidate *)xmalloc((size_t)cap * sizeof(*s->cand2));
    s->cap = cap;
}



void pulsar_sample_scratch_free(pulsar_sample_scratch *s) {
    free(s->cand);
    free(s->keys);
    free(s->tmp);
    free(s->cand2);
    memset(s, 0, sizeof(*s));
}



/* =====
 * THE sampler: the one authority for the candidate set (L186).
 *
 * pulsar_sample_dist_build turns a logits row plus (temperature, top-k,
 * top-p, min-p) into the filtered, renormalised nucleus every lane draws
 * from.  The plain per-token samplers (sample_top_p_min_p, behind
 * pulsar_sample_logits / pulsar_session_sample) are build -> draw; the
 * speculative walk builds the same object for every verified row and draws
 * the target's own token from it.  One candidate order -- descending
 * logit, ascending id on ties, the order pulsar_sample_dist_draw walks -- so
 * a given key yields the same token whether or not speculation is
 * armed; tests/pulsar_test.cpp --sampler pins plain == draw(build) under
 * fixed seeds.  Before L186 the plain sampler carried its own cutoff loops
 * and, at top_p >= 1, walked candidates in VOCAB order: the same marginals,
 * a different token for the same rng word, so quenching the drafter
 * mid-request changed the sampled sequence under a fixed seed.
 *
 * Degenerate rows refuse.  No finite logit, or a candidate mass that is not
 * a positive finite number (a NaN temperature), returns 0 through
 * sample_dist_refuse -- one message, `out` zeroed -- and the plain samplers
 * return -1.  No point mass at token 0, no best-effort candidate.
 *
 * NOTE on optimizing this: `sum` is taken over ALL n candidates BEFORE any
 * cutoff, and the top-p cutoff is relative to it (filtered_sum / sum).
 * Dropping candidates from `sum` -- the NAIVE min-p pre-filter -- therefore
 * changes `sum`, hence `filtered`, hence `filtered_sum`, hence EVERY output
 * probability. It is not an equivalent rewrite; tests/pulsar_test.cpp
 * --sampler catches it (n 6 != 5). The full-vocab sum is load-bearing.
 *
 * What the top_k <= 0, min_p > 0 path DOES do (the LEGAL prefilter): the sum
 * still covers every finite candidate, but is accumulated in VOCAB-INDEX
 * order in the same pass that computes each prob once and collects only the
 * prefilter survivors (see SAMPLE_MINP_PREFILTER_SLACK: a strict superset of
 * the min-p cutoff's survivors under ANY sum, forming a prefix of the
 * descending sort). Only the survivors are sorted -- tens instead of 129k at
 * the server-default min_p = 0.05 -- and the byte-exact cutoff then trims
 * the boundary with the same comparisons as ever, evaluated against that
 * sum. Distribution-preserving by construction; NOT stream-preserving:
 * summing in index order instead of the old sorted-descending order rounds
 * differently (~1e-7 relative), so every output prob moves by that much and
 * rng draws near a bucket edge can flip. Survivor membership and order are
 * unchanged (tests/pulsar_test.cpp --sampler-prefilter pins set/order
 * identity against the old-sum reference and characterizes the prob delta;
 * --sampler is byte-exact against the re-derived index-order-sum reference).
 * The min_p <= 0 full sort and the top_k > 0 preselect keep the old
 * sorted-order sum and remain byte-identical to the pre-prefilter build.
 *
 * ALIASING CONTRACT: `out`'s ids/probs must never point into `scratch`. The
 * spec walk (session_spec.cpp) holds one dist at a time while reusing the
 * scratch across accepted positions, so handing `out` a scratch pointer to
 * save the malloc would be silently wrong the moment two dists overlap --
 * and the --sampler gate builds one dist at a time, so it would NOT catch it.
 */

/* THE refusal for a row no distribution can be drawn from.  Every builder
 * fails through here and the plain samplers propagate the 0 as -1, so a
 * degenerate row refuses in one place (fail closed, loudly, once). */
static int sample_dist_refuse(pulsar_sample_dist *out, uint32_t n_vocab, const char *why) {
    memset(out, 0, sizeof(*out));
    static int said = 0;
    if (!said) {
        said = 1;
        fprintf(stderr,
                "pulsar: sampler: refusing a %u-wide logits row: %s -- no token is drawn "
                "(said once)\n",
                n_vocab, why);
    }
    return 0;
}

/* THE nucleus cutoff and emit, shared by every builder.  `n` >= 1 candidates
 * in descending (logit, ascending id) order are read through prob(i) / id(i)
 * (two callers keep two candidate layouts; the order is what is shared).
 * min-p is division-free (L149: prob(0) == expf(0) == 1.0f exactly, so the
 * floor is prob(0) * min_p and the decision depends on prob(i) alone); top-p
 * is relative to `sum`, the mass the caller accumulated over ALL its
 * candidates, and includes the crossing candidate.  A caller whose contract
 * has proven the top-p test inert and that holds no full-row mass (the
 * device-prefiltered build) passes sum = +inf, which makes the test false
 * without a second loop.  Emits the nucleus renormalised over itself, in
 * the order pulsar_sample_dist_draw walks. */
template <class ProbAt, class IdAt>
static void sample_nucleus_emit(uint32_t n, float sum, float top_p, float min_p,
                                ProbAt prob, IdAt id, pulsar_sample_dist *out) {
    const float min_e = prob(0) * min_p;
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = prob(i);
        if (i > 0 && p < min_e) break;
        filtered_sum += p;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    /* filtered >= 1: i == 0 never breaks on min-p, and the top-p break
     * follows the increment. */
    out->ids = (int *)xmalloc((size_t)filtered * sizeof(int));
    out->probs = (float *)xmalloc((size_t)filtered * sizeof(float));
    out->n = filtered;
    for (uint32_t i = 0; i < filtered; i++) {
        out->ids[i] = id(i);
        out->probs[i] = prob(i) / filtered_sum;   /* renormalized nucleus */
    }
}

int pulsar_sample_dist_build(const float *logits, uint32_t n_vocab,
                          float temperature, int top_k, float top_p, float min_p,
                          pulsar_sample_scratch *scratch, pulsar_sample_dist *out) {
    memset(out, 0, sizeof(*out));
    /* Every knob must be finite.  A NaN fails every comparison-based clamp
     * below (and the greedy `<= 0` test); +-Inf temperature or top_p produces
     * a FINITE mass (p = expf((v-max)/+-inf) = 1, sum = n), so neither the
     * clamps nor the mass guard would catch it -- each would silently become a
     * uniform or unfiltered draw.  The server parser refuses these at the
     * surface; this is the engine's own refusal, so the API/CLI path cannot
     * reach the sampler with one either (review B2).  Fail closed, loudly,
     * once, for all three. */
    if (!isfinite(temperature) || !isfinite(top_p) || !isfinite(min_p))
        return sample_dist_refuse(out, n_vocab, "a sampling knob is not finite");
    if (temperature <= 0.0f) {
        const int best = sample_argmax(logits, n_vocab);
        if (best < 0) return sample_dist_refuse(out, n_vocab, "no finite logit");
        out->ids = (int *)xmalloc(sizeof(int));
        out->probs = (float *)xmalloc(sizeof(float));
        out->ids[0] = best;
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > 1024) top_k = top_k <= 0 ? 0 : 1024;

    /* collect candidates: full vocab, or top-k preselect.
     * Buffers come from the caller's scratch and are reused across calls --
     * the sampled speculative walk calls this per accepted position. */
    uint32_t cap = top_k > 0 ? (uint32_t)top_k : n_vocab;
    sample_scratch_reserve(scratch, cap);
    sample_candidate *cand = scratch->cand;
    /* `sc` is the descending-order candidate view the cutoff tail walks. The
     * prefilter path gathers into scratch->cand2 (probs and sum already
     * computed, index-order sum); the other paths sort `cand` in place and
     * compute probs + sum post-sort, in sorted order, exactly as before. */
    sample_candidate *sc = cand;
    /* LAZY-MATERIALIZATION view for the full-vocab (top_k<=0, min_p small)
     * path. That path used to materialize a 12-byte sample_candidate for
     * EVERY finite vocab entry (129,280 x 12 B = 1.55 MB of strided stores,
     * each pulling logits[id] through a random gather) and then walk the same
     * array AGAIN to fill in .prob — while the cutoff loop below almost always
     * breaks within a few dozen entries and only `filtered` entries are ever
     * read out. Instead it now makes ONE fused pass that writes only the
     * probabilities, PACKED (4 B/entry, into cand's storage — cand itself is
     * unused on that path), and leaves the ids where the radix sort already
     * put them: the low 32 bits of keys[i]. `pv`/`idk` are the resulting
     * lazy view; the prob/id readers below read whichever representation is
     * live. Arithmetic is UNCHANGED: same max_logit (logits[keys[0]] IS
     * cand[0].logit), same expf inputs in the same sorted order, same `sum`
     * accumulation order, hence the same cutoff, `filtered` set and output
     * probabilities. */
    const float *pv = NULL;         /* packed probs, sorted order (or NULL) */
    const uint64_t *idk = NULL;     /* sorted keys; id = low 32 bits (or NULL) */
    uint32_t n = 0;
    int have_probs = 0;
    float sum = 0.0f;
    bool sum_inert = false;   /* the full-nucleus fast arm hands the emit +inf */
    if (top_k > 0) {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == (uint32_t)top_k && v <= cand[n - 1].logit) continue;
            uint32_t j = n < (uint32_t)top_k ? n++ : n - 1;
            while (j > 0 && cand[j - 1].logit < v) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j].id = (int)i;
            cand[j].logit = v;
        }
    } else if (min_p > SAMPLE_MINP_PREFILTER_MIN) {
        /* min-p prefilter path (see the block comment above; the MIN floor
         * keeps the cutoff arithmetic normal — below it the exact full sort
         * runs instead). Pass 1: the row max by THE rule (sample_argmax):
         * first finite value seeds, lowest id on ties — the same candidate
         * the stable descending sort puts at cand[0]. */
        const int best = sample_argmax(logits, n_vocab);
        const uint32_t max_id = best >= 0 ? (uint32_t)best : 0u;
        const float max_logit = best >= 0 ? logits[max_id] : 0.0f;
        const uint32_t finite = best >= 0 ? 1u : 0u;   /* "any finite logit" is all pass 2 asks */
        /* Pass 2: one prob per finite candidate, computed ONCE and carried
         * through the sort; `sum` over ALL of them, in this (index) order;
         * survivors collected in ascending-id order so the stable radix keeps
         * ascending-id as the canonical tie order.
         *
         * L219 fast path -- the full-nucleus shape skips the below-floor expf.
         * With top_p == 1.0 the emitted set is the min-p cut and every prob is
         * renormalised over it, so the only role `sum` plays is the
         * `filtered_sum / sum >= 1.0` break.  While a survivor remains that
         * ratio is at least prefilter/sum away from one, and with
         * pulsar_sample_sparse_minp(min_p, n_vocab) that is far more than an
         * ulp, so the break can only coincide with the min-p
         * boundary.  Hand the emit +inf instead (the trick
         * pulsar_sample_dist_build_prefiltered documents) and members need no
         * normaliser at all.  An entry below the device prefilter's logit floor
         * (slack 1e-3, a strict superset of the survivors) can never satisfy
         * `p >= prefilter`, so it needs no expf.  At the production shape that
         * is ~129k expf per sampled token -- the pass the L149 comment measures
         * at ~630 us.  Membership is still decided by the SAME comparison
         * below; the floor only bounds it.  Below the sparse floor the full sum
         * is still computed and the general arithmetic runs. */
        if (finite > 0) {
            uint64_t *keys = scratch->keys;
            const float prefilter = min_p * SAMPLE_MINP_PREFILTER_SLACK;
            const bool full_nucleus = top_p == 1.0f && pulsar_sample_sparse_minp(min_p, n_vocab);
            const float floor_logit = full_nucleus
                ? max_logit + temperature * (logf(min_p) - 1e-3f)
                : -INFINITY;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                if (!isfinite(v)) continue;
                if (v < floor_logit && i != max_id) continue;
                const float p = expf((v - max_logit) / temperature);
                if (!full_nucleus) sum += p;   /* inert on the fast arm (sum = +inf below) */
                if (p >= prefilter || i == max_id) {
                    keys[n] = ((uint64_t)sample_desc_key(v) << 32) | n;
                    cand[n] = (sample_candidate){
                        .id = (int)i, .logit = v, .prob = p};
                    n++;
                }
            }
            if (full_nucleus) {
                sum = INFINITY;
                sum_inert = true;
            }
            /* n >= 1: the max candidate is always kept. */
            sample_radix_sort_desc(keys, scratch->tmp, n);
            for (uint32_t i = 0; i < n; i++)
                scratch->cand2[i] = cand[(uint32_t)keys[i]];
            sc = scratch->cand2;
            have_probs = 1;
        }
    } else {
        uint64_t *keys = scratch->keys;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            keys[n++] = ((uint64_t)sample_desc_key(v) << 32) | (uint32_t)i;
        }
        if (n) {
            sample_radix_sort_desc(keys, scratch->tmp, n);
            /* Fused prob pass over the packed view (see the note at `pv`).
             * Byte-identical to the old materialize-then-!have_probs-loop:
             * max_logit, the expf inputs and the `sum` order all match. */
            float *probs = (float *)cand;   /* n*4 <= cap*sizeof(sample_candidate) */
            const float max_logit = logits[(uint32_t)keys[0]];
            for (uint32_t i = 0; i < n; i++) {
                const float p =
                    expf((logits[(uint32_t)keys[i]] - max_logit) / temperature);
                probs[i] = p;
                sum += p;
            }
            pv = probs;
            idk = keys;
            have_probs = 1;
        }
    }
    if (n == 0) return sample_dist_refuse(out, n_vocab, "no finite logit");

    if (!have_probs) {
        const float max_logit = sc[0].logit;
        for (uint32_t i = 0; i < n; i++) {
            sc[i].prob = expf((sc[i].logit - max_logit) / temperature);
            sum += sc[i].prob;
        }
    }
    /* prob(0) == expf(0) == 1 exactly, so the mass is >= 1 for any finite
     * temperature and finite knobs: the entry guard above refuses anything
     * else, so this is now an arithmetic-fault catch, not the NaN-temperature
     * door it was.  The full-nucleus arm replaces `sum` with +inf on purpose
     * (see its comment), so the guard does not apply there: the emitted probs
     * are renormalised by filtered_sum and every kept prob is a finite
     * positive number. */
    if (!sum_inert && (sum <= 0.0f || !isfinite(sum)))
        return sample_dist_refuse(out, n_vocab, "candidate mass is not a positive finite number");
    sample_nucleus_emit(n, sum, top_p, min_p,
                        [&](uint32_t i) { return pv ? pv[i] : sc[i].prob; },
                        [&](uint32_t i) { return idk ? (int)(uint32_t)idk[i] : sc[i].id; },
                        out);
    return 1;
}

/* L149: the min-p prefilter path of pulsar_sample_dist_build, fed a
 * device-selected candidate subset instead of the full row.
 *
 * CONTRACT (the caller's device kernel, pulsar_gpu_minp_prefilter_rows):
 * `ids`/`vals` list, in ASCENDING id order, every finite logit v of the row
 * with v >= max_logit + delta, where max_logit is the row's first finite
 * maximum and delta <= T * ln(min_p) (a margin below the floor). That set is
 * a superset of the min-p survivors {i : expf((v_i - max)/T) >= min_p} and
 * contains the max, so -- probs being monotone in the logit -- the sorted
 * survivors are a prefix of the sorted candidates and the shared cutoff
 * (sample_nucleus_emit) walks exactly what pulsar_sample_dist_build walks
 * over the full row.
 *
 * Byte-identical to pulsar_sample_dist_build(row, n_vocab, T, 0, 1.0f, min_p)
 * for pulsar_sample_sparse_minp(min_p, n_vocab), n_vocab the width of the row
 * the candidates came from:
 *   - same expf inputs (v - max)/T, same descending (logit, ascending id)
 *     order via the same radix keys (the ordinal is monotone in id in both),
 *     same min-p comparison (division-free, so no normaliser needed);
 *   - top_p == 1.0f is redundant given the min-p floor: the full path's
 *     `filtered_sum / sum >= 1.0f` can only fire once no further survivor
 *     exists (sum >= filtered_sum + min_p while one does, and
 *     min_p / filtered_sum > 2 * 2^-24 for filtered_sum <= n_vocab under
 *     the predicate's floor -- the proof at pulsar_sample_sparse_minp), and when it fires the next candidate
 *     fails min-p anyway -- same `filtered`, same probs.  This path holds
 *     no full-row mass, so it hands the shared cutoff sum = +inf: the same
 *     loop, the test provably inert.
 * Anything outside the contract returns 0 with `out` untouched; the caller
 * decides whether that is a shape outside the sparse contract (read the
 * full row) or a device result that broke it (refuse) -- see
 * session_spec.cpp (pulsar_test --sampler pins the identity and its
 * sensitivity). */
int pulsar_sample_dist_build_prefiltered(const int32_t *ids, const float *vals, uint32_t n_cand,
                                         uint32_t n_vocab, float max_logit, float temperature, float min_p,
                                         pulsar_sample_scratch *scratch, pulsar_sample_dist *out) {
    if (!ids || !vals || !scratch || !out || n_cand == 0 || n_cand > n_vocab || !(temperature > 0.0f) ||
        !isfinite(max_logit) || !pulsar_sample_sparse_minp(min_p, n_vocab) || !(min_p <= 1.0f))
        return 0;
    sample_scratch_reserve(scratch, n_cand);
    sample_candidate *cand = scratch->cand;
    uint64_t *keys = scratch->keys;
    uint32_t n = 0;
    for (uint32_t c = 0; c < n_cand; c++) {
        const float v = vals[c];
        if (!isfinite(v)) continue;
        keys[n] = ((uint64_t)sample_desc_key(v) << 32) | n;
        cand[n] = (sample_candidate){
            .id = (int)ids[c], .logit = v, .prob = expf((v - max_logit) / temperature)};
        n++;
    }
    if (n == 0) return 0;
    sample_radix_sort_desc(keys, scratch->tmp, n);
    sample_candidate *sc = scratch->cand2;
    for (uint32_t i = 0; i < n; i++) sc[i] = cand[(uint32_t)keys[i]];
    /* The sorted head must be the max the caller named, or the list is not
     * the superset the contract promises: refuse rather than sample from it. */
    if (sc[0].logit != max_logit) return 0;
    memset(out, 0, sizeof(*out));
    sample_nucleus_emit(n, INFINITY, 1.0f, min_p,
                        [&](uint32_t i) { return sc[i].prob; },
                        [&](uint32_t i) { return sc[i].id; },
                        out);
    return 1;
}

void pulsar_sample_dist_free(pulsar_sample_dist *d) {
    free(d->ids);
    free(d->probs);
    memset(d, 0, sizeof(*d));
}

float pulsar_sample_dist_prob(const pulsar_sample_dist *d, int token) {
    for (uint32_t i = 0; i < d->n; i++)
        if (d->ids[i] == token) return d->probs[i];
    return 0.0f;
}

/* The draw at `key`: an exponential race (the Gumbel-max trick) -- every candidate v gets its own uniform u_v, a
 * hash of (key, v), and the token is the argmin of -ln(u_v) / p_v, exactly a draw from p.  The race is what lets
 * two distributions share their randomness token by token: the speculative walk draws the target's token from p
 * and the drafter its draft from q at the same key, and the two coincide with probability at least
 * (1 - TV) / (1 + TV) whatever order either keeps its candidates in -- the drafter-invariant coupling (Daliri et
 * al. 2024).  An inverse-CDF draw at one shared uniform matched only where p and q rank their candidates alike
 * (two near-tied leaders in swapped order shared almost no mass).  Ties go to the earlier candidate in the stored
 * order. */
int pulsar_sample_dist_draw(const pulsar_sample_dist *d, uint64_t key) {
    int best = d->ids[0];
    double best_s = INFINITY;
    for (uint32_t i = 0; i < d->n; i++) {
        if (!(d->probs[i] > 0.0f)) continue;
        const uint64_t h = sample_mix64(key ^ ((uint64_t)(uint32_t)d->ids[i] * 0xd1b54a32d192ed03ULL));
        const double u = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);   /* (0, 1): 53 bits, never 0 */
        const double sc = -log(u) / (double)d->probs[i];
        if (sc < best_s) {
            best_s = sc;
            best = d->ids[i];
        }
    }
    return best;
}



int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t     key,
        pulsar_sample_scratch *scratch) {
    /* build -> draw, nothing else: the candidate set, its order and every
     * clamp live in pulsar_sample_dist_build (L186).  A caller with no
     * session (pulsar_sample_logits) gets a call-local scratch -- the same
     * path, the buffers' lifetime is the only difference. */
    pulsar_sample_scratch local;
    if (!scratch) {
        memset(&local, 0, sizeof(local));
        scratch = &local;
    }
    int tok = -1;
    pulsar_sample_dist d;
    if (pulsar_sample_dist_build(logits, n_vocab, temperature, top_k, top_p, min_p,
                                 scratch, &d)) {
        /* Greedy is a point mass: the key is not read (the token still holds
         * its index in the stream; the caller advances past it like any other). */
        tok = temperature <= 0.0f ? d.ids[0] : pulsar_sample_dist_draw(&d, key);
        pulsar_sample_dist_free(&d);
    }
    if (scratch == &local) pulsar_sample_scratch_free(&local);
    return tok;
}







int pulsar_logits_top_logprobs(const float *logits, int n_vocab,
                            pulsar_token_score *out, int k) {
    if (!logits || !out || k <= 0 || n_vocab <= 0) return 0;
    if (k > n_vocab) k = n_vocab;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = PULSAR_NEG_INF;
        out[i].logprob = PULSAR_NEG_INF;
    }

    /* Row max by THE rule (sample_argmax): the first finite value seeds.  A
     * -1e30 seed is finite, so an all-non-finite row used to pass the
     * isfinite(max_logit) check below and score against log(0). */
    float max_logit = 0.0f;
    bool have_max = false;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (!have_max || v > max_logit) {
            max_logit = v;
            have_max = true;
        }
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!have_max) return 0;

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : PULSAR_NEG_INF;
    }
    return k;
}


int pulsar_logits_token_logprob(const float *logits, int n_vocab, int token,
                             pulsar_token_score *out) {
    if (!logits || !out || n_vocab <= 0 || token < 0 || token >= n_vocab) return 0;

    float max_logit = 0.0f;
    bool have_max = false;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (!have_max || v > max_logit) {
            max_logit = v;
            have_max = true;
        }
    }
    if (!have_max) return 0;

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : PULSAR_NEG_INF;
    return 1;
}



}  // namespace sref

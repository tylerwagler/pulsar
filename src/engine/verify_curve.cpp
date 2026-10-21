/* L301 -- the spec lane's verify pick: ONE rule for how many draft rows a round verifies, every drafter.
 *
 * After DeepInfer's MTPVarlen (csrc/engine/mtp_varlen.hpp, Apache-2.0), ported in design, not in code.  A round
 * verifies n_seqs base rows (one a decoding bank, never cut) and some of the drafts its banks carry.  A draft is worth
 * its survival -- the product of the confidences up to it, the chance the walk reaches and accepts it -- so the
 * round's expected tokens with its best T drafts verified are n_seqs + the top-T survivals, and its wall is the
 * round's cost at n_seqs + T rows.  The pick is the T that maximises their ratio:
 *
 *     T* = argmax_T (n_seqs + sum of the top-T survivals) / round_ms(n_seqs, n_seqs + T)
 *
 * over every (bank, draft position) ranked by survival -- a bank's verified drafts are then a prefix of its chain,
 * survival being monotone along it.  round_ms is MEASURED, never a constant of a deployment:
 *
 *   round_ms(n, r) = base[r] + slope * r + host(n) + draft(n)
 *
 *   - base[r]: the verify forward of an r-row step, profiled once per (container, drafter, bank pool, context,
 *     build) at startup (pulsar_session_verify_curve_open) and cached on disk under that key.  A per-row-count
 *     TABLE, not a line: the decode arms run 16-row blocks, so the cost is a staircase, flat inside a block and a
 *     step at 17 -- L288's 16 / 32 width probes priced two points of it, the linear lane fit none.
 *   - slope: the online correction (DeepInfer's ForwardRuntimeEstimator).  The profile ran at one depth; a served
 *     round's forward drifts from it with the context it attends and with the box.  Each round contributes its
 *     residual per row, (forward_ms - base[rows]) / rows; the medians per row count (counts with >= 10 samples,
 *     each count equal weight) are reduced by their median over a window of the last 5000 rounds -- robust to a
 *     stray round (a page-fault burst, a GC pause on the host) and to which row counts dominate the window.
 *     Applied as max(0, slope): the profile is the floor, a negative residual is not extrapolated to wider rows.
 *   - host(n) and draft(n): the round's wall outside the forward -- planning, the walk, emission, and the next
 *     round's drafting -- the median of the last 9 rounds at n banks (valid from 5), else of the last 9 at any n
 *     (DeepInfer's SamplerRuntimeEstimator).  They do not depend on T, but they set how much a wider forward costs
 *     against the whole round.
 *
 * Every consumer reads this one curve (rule 4): the server's per-round allocation (server_sched.cpp), the drafters'
 * keep (block drafters, DSpark, copy chains: pulsar_spec_keep_picked) and the MTP chain's lockstep continuation
 * (pulsar_spec_chain_step, which adds the next drafter step's measured cost to the round).  It replaces L288's
 * marginal threshold (row_us / token_ms), its 16 / 32 width probes and the drafters' token_ms price.
 *
 * No curve (a tensor-parallel group -- its ranks must draft alike and the clocks are the leader's -- a classic
 * one-bank server, or before the profile): every pending draft the row budget holds is verified, by survival -- the
 * row cap alone, "no number, no decision" as before.  Which drafts are verified is all the pick moves: every emitted
 * token is the target's (spec == plain), greedy or keyed.
 *
 * The lag rule (DeepInfer's N-2 / N-1 comment): a planner that runs ahead of the round's results (overlapped
 * scheduling) must take T from the survivals one round older than those it allocates with, so that draft i's
 * admission never depends on confidences i+1.. computed later.  The scheduler here plans after the results, so
 * both are the round's own. */
#include "pulsar_engine_internal.h"
#include "spec_internal.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace {

/* A median over a multiset that changes one sample at a time: the smaller half in `lo`, the larger in `hi`,
 * |lo| == |hi| or |lo| == |hi| + 1. */
struct median_set {
    std::multiset<double> lo, hi;
    void rebalance() {
        if (lo.size() > hi.size() + 1) {
            hi.insert(*lo.rbegin());
            lo.erase(std::prev(lo.end()));
        } else if (lo.size() < hi.size()) {
            lo.insert(*hi.begin());
            hi.erase(hi.begin());
        }
    }
    void insert(double v) {
        if (lo.empty() || v <= *lo.rbegin()) lo.insert(v);
        else hi.insert(v);
        rebalance();
    }
    void erase(double v) {
        auto it = lo.find(v);
        if (it != lo.end()) lo.erase(it);
        else if ((it = hi.find(v)) != hi.end()) hi.erase(it);
        rebalance();
    }
    size_t size() const { return lo.size() + hi.size(); }
    double get() const { return lo.size() == hi.size() ? 0.5 * (*lo.rbegin() + *hi.begin()) : *lo.rbegin(); }
};

/* The median of the last VC_RECENT observations, a number from VC_RECENT_MIN of them. */
#define VC_RECENT 9u
#define VC_RECENT_MIN 5u
struct recent_median {
    double v[VC_RECENT];
    uint32_t n, next;
    double med;
    void add(double x) {
        v[next] = x;
        next = (next + 1u) % VC_RECENT;
        if (n < VC_RECENT) n++;
        double s[VC_RECENT];
        for (uint32_t i = 0; i < n; i++) {   /* insertion: n <= 9 */
            uint32_t j = i;
            for (; j > 0 && s[j - 1u] > v[i]; j--) s[j] = s[j - 1u];
            s[j] = v[i];
        }
        med = n % 2u ? s[n / 2u] : 0.5 * (s[n / 2u - 1u] + s[n / 2u]);
    }
    bool get(double *out) const {
        if (n < VC_RECENT_MIN) return false;
        *out = med;
        return true;
    }
};

/* The residual per row of the forward against the profile: the median per row count over a window of rounds, then
 * the median of those (counts with VC_SLOPE_MIN samples or more, each count one vote). */
#define VC_SLOPE_WINDOW 5000u
#define VC_SLOPE_MIN 10u
struct residual_slope {
    std::deque<std::pair<uint32_t, double>> win;
    std::map<uint32_t, median_set> by_rows;
    median_set medians;
    void update(uint32_t rows, double v, bool insert) {
        median_set &m = by_rows[rows];
        if (m.size() >= VC_SLOPE_MIN) medians.erase(m.get());
        if (insert) m.insert(v);
        else m.erase(v);
        if (m.size() >= VC_SLOPE_MIN) medians.insert(m.get());
        else if (m.size() == 0) by_rows.erase(rows);
    }
    void add(uint32_t rows, double v) {
        if (win.size() == VC_SLOPE_WINDOW) {
            update(win.front().first, win.front().second, false);
            win.pop_front();
        }
        win.emplace_back(rows, v);
        update(rows, v, true);
    }
    bool get(double *out) const {
        if (medians.size() == 0) return false;
        *out = medians.get();
        return true;
    }
};

}  // namespace

/* The engine's verify curve (pulsar_engine::verify_curve; NULL = none). */
struct pulsar_verify_curve {
    uint32_t rows_max;                              ///< the widest verify step (the family's fused_heads_max)
    double base_ms[PULSAR_SPEC_LOGITS_ROWS + 1];    ///< [r]: the profiled r-row verify forward, ms ([0] unused)
    uint64_t key;                                   ///< the cache key it was profiled / loaded under
    bool cached;                                    ///< loaded from the cache (else profiled this process)
    residual_slope slope;
    recent_median host[PULSAR_SPEC_LOGITS_ROWS + 1], host_any;
    recent_median draft[PULSAR_SPEC_LOGITS_ROWS + 1], draft_any;
    uint64_t observed;                              ///< rounds observed
    double ann_slope;                               ///< the residual last announced (rule 5)
    bool ann_slope_valid;
    uint64_t ann_at;                                ///< `observed` at that announcement
};

pulsar_verify_curve *pulsar_verify_curve_new(uint32_t rows_max, const double *base_ms) {
    if (rows_max == 0u || rows_max > PULSAR_SPEC_LOGITS_ROWS) return NULL;
    pulsar_verify_curve *c = new pulsar_verify_curve();
    c->rows_max = rows_max;
    for (uint32_t r = 1; r <= rows_max; r++) c->base_ms[r] = base_ms[r];
    return c;
}

void pulsar_verify_curve_free(pulsar_verify_curve *c) { delete c; }

static double vc_recent(const recent_median *by_n, const recent_median *any, uint32_t n) {
    double v = 0.0;
    if (n <= PULSAR_SPEC_LOGITS_ROWS && by_n[n].get(&v)) return v;
    return any->get(&v) ? v : 0.0;
}

static double vc_slope(const pulsar_verify_curve *c) {
    double v = 0.0;
    return c->slope.get(&v) && v > 0.0 ? v : 0.0;
}

/* round_ms(n, rows): see the head of the file.  rows in [1, rows_max]. */
static double vc_round_ms(const pulsar_verify_curve *c, uint32_t n, uint32_t rows) {
    return c->base_ms[rows] + vc_slope(c) * (double)rows + vc_recent(c->host, &c->host_any, n) +
           vc_recent(c->draft, &c->draft_any, n);
}

/* The curve a pick may read: none on a tensor-parallel group (its ranks draft alike from the same numbers, and the
 * clocks are the leader's), else the engine's. */
static const pulsar_verify_curve *vc_of(const pulsar_engine *e) {
    return e && !e->tp ? e->verify_curve : NULL;
}

/* The pick over survivals sorted descending: T, the drafts verified beside the n base rows, within cap_rows rows;
 * `extra_ms` is added to every candidate's round (a drafter step still to be paid).  *ratio = the best tokens/ms
 * (0 with no curve). */
uint32_t pulsar_verify_curve_pick(const pulsar_verify_curve *c, uint32_t n, const float *surv, uint32_t m,
                                  uint32_t cap_rows, double extra_ms, double *ratio) {
    *ratio = 0.0;
    if (c && cap_rows > c->rows_max) cap_rows = c->rows_max;
    const uint32_t room = cap_rows > n ? std::min(cap_rows - n, m) : 0u;
    if (!c || n == 0u || room == 0u) return room;
    double cur = (double)n;
    double best = cur / (vc_round_ms(c, n, n) + extra_ms);
    uint32_t best_t = 0;
    for (uint32_t t = 1; t <= room; t++) {
        cur += surv[t - 1u] > 0.0f ? (double)surv[t - 1u] : 0.0;
        const double r = cur / (vc_round_ms(c, n, n + t) + extra_ms);
        if (r > best) {
            best = r;
            best_t = t;
        }
    }
    *ratio = best;
    return best_t;
}

uint32_t pulsar_engine_verify_pick(const pulsar_engine *e, uint32_t n_seqs, const float *surv, uint32_t m,
                                   uint32_t cap_rows) {
    double ratio;
    return pulsar_verify_curve_pick(vc_of(e), n_seqs, surv, m, cap_rows, 0.0, &ratio);
}

/* A draft's survival from its chain's confidences: a confidence below 0 is none (a drafter without a head, the
 * stop rule off) and counts as 1 -- the draft is ranked as sure, as DeepInfer pads a chain with no rates. */
static inline float vc_conf(float c) { return c < 0.0f ? 1.0f : c; }

void pulsar_spec_keep_picked(const pulsar_session *s, uint32_t n_chains, spec_redraft_req *const *qs, uint32_t nq) {
    const pulsar_engine *e = s->engine;
    struct cand { float surv; uint32_t q, k; };
    std::vector<cand> c;
    for (uint32_t i = 0; i < nq; i++) {
        float run = 1.0f;
        for (uint32_t k = 0; k < qs[i]->keep; k++) c.push_back({run *= vc_conf(qs[i]->conf[k]), i, k});
    }
    /* by survival, then chain, then position: a chain's admitted drafts are a prefix */
    std::sort(c.begin(), c.end(), [](const cand &a, const cand &b) {
        return a.surv != b.surv ? a.surv > b.surv : a.q != b.q ? a.q < b.q : a.k < b.k;
    });
    std::vector<float> surv(c.size());
    for (size_t i = 0; i < c.size(); i++) surv[i] = c[i].surv;
    double ratio;
    const uint32_t T = pulsar_verify_curve_pick(vc_of(e), n_chains, surv.data(), (uint32_t)surv.size(),
                                                pulsar_engine_fused_heads_max(e), 0.0, &ratio);
    for (uint32_t i = 0; i < nq; i++) qs[i]->keep = 0;
    for (uint32_t t = 0; t < T; t++) qs[c[t].q]->keep++;
}

uint32_t pulsar_verify_chain_pick(const pulsar_verify_curve *c, const pulsar_lane_cost *step, uint32_t cap,
                                  uint32_t n_chains, const float *drafted, uint32_t n_drafted, const float *run,
                                  uint32_t n_live, double spent_ms, bool *go) {
    const uint32_t budget = cap > n_chains ? cap - n_chains : 0u;
    /* the live chains by survival, best first */
    uint32_t idx[PULSAR_SPEC_LOGITS_ROWS];
    const uint32_t nl = std::min(n_live, (uint32_t)PULSAR_SPEC_LOGITS_ROWS);
    for (uint32_t i = 0; i < nl; i++) idx[i] = i;
    std::sort(idx, idx + nl, [&](uint32_t a, uint32_t b) { return run[a] != run[b] ? run[a] > run[b] : a < b; });
    for (uint32_t i = 0; i < n_live; i++) go[i] = false;
    if (!c) {
        /* no curve: a chain goes on while its next draft (worth at most its survival) would still rank among the
         * drafts the row budget verifies */
        uint32_t m = 0;
        for (uint32_t x = 0; x < nl; x++) {
            uint32_t ahead = 0;
            for (uint32_t j = 0; j < n_drafted; j++) ahead += drafted[j] >= run[idx[x]] ? 1u : 0u;
            if (ahead >= budget) continue;
            go[idx[x]] = true;
            m++;
        }
        return m;
    }
    /* the round as drafted so far, then with the k best live chains' next drafts in it (each valued at its chain's
     * survival, the most it can be worth) and the step that drafts them paid -- the drafter's measured step at k
     * chains (flat + slope * k; one width only, a lone stream: its mean there; 0 until measured): the k with the
     * best ratio goes on */
    std::vector<float> d(drafted, drafted + n_drafted);
    std::sort(d.begin(), d.end(), [](float a, float b) { return a > b; });
    double best = 0.0;
    uint32_t best_k = 0;
    std::vector<float> merged;
    merged.reserve(d.size() + nl);
    for (uint32_t k = 0; k <= nl; k++) {
        merged.assign(d.begin(), d.end());
        for (uint32_t x = 0; x < k; x++) merged.push_back(run[idx[x]]);
        std::inplace_merge(merged.begin(), merged.begin() + (long)d.size(), merged.end(),
                           [](float a, float b) { return a > b; });
        const double step_ms = k ? pulsar_lane_cost_ms(step, (double)k) : 0.0;
        double ratio;
        (void)pulsar_verify_curve_pick(c, n_chains, merged.data(), (uint32_t)merged.size(), cap, spent_ms + step_ms,
                                       &ratio);
        if (k == 0u || ratio > best) {
            best = ratio;
            best_k = k;
        }
    }
    for (uint32_t x = 0; x < best_k; x++) go[idx[x]] = true;
    return best_k;
}

uint32_t pulsar_spec_chain_step(const pulsar_session *s, uint32_t n_chains, const float *drafted, uint32_t n_drafted,
                                const float *run, uint32_t n_live, double spent_ms, bool *go) {
    const pulsar_engine *e = s->engine;
    const pulsar_lane_cost step = lane_cost_fit_view(&e->draft_step_cost);
    return pulsar_verify_chain_pick(vc_of(e), &step, pulsar_engine_fused_heads_max(e), n_chains, drafted, n_drafted,
                                    run, n_live, spent_ms, go);
}

/* ---- the online correction --------------------------------------------------------------------------------- */

void pulsar_engine_verify_curve_observe(pulsar_engine *e, uint32_t n_seqs, uint32_t rows, double forward_ms,
                                        double host_ms, double draft_ms) {
    pulsar_verify_curve *c = e ? e->verify_curve : NULL;
    if (!c || e->tp || n_seqs == 0u || rows == 0u || rows > c->rows_max || !(forward_ms > 0.0)) return;
    c->slope.add(rows, (forward_ms - c->base_ms[rows]) / (double)rows);
    if (host_ms >= 0.0) {
        if (n_seqs <= PULSAR_SPEC_LOGITS_ROWS) c->host[n_seqs].add(host_ms);
        c->host_any.add(host_ms);
    }
    if (draft_ms >= 0.0) {
        if (n_seqs <= PULSAR_SPEC_LOGITS_ROWS) c->draft[n_seqs].add(draft_ms);
        c->draft_any.add(draft_ms);
    }
    c->observed++;
    /* rule 5: the correction announces itself when it first has a number and, at most once in 4096 rounds, when it
     * has moved by more than 0.1 ms a row since -- the round cost the pick prices with is in the log */
    double sl;
    if (!c->slope.get(&sl)) return;
    if (!c->ann_slope_valid || (c->observed - c->ann_at >= 4096u && fabs(sl - c->ann_slope) > 0.1)) {
        fprintf(stderr, "pulsar: verify curve correction%s: %+.3f ms a row past the profile (applied %.3f), host %.1f "
                        "ms, draft %.1f ms at %u banks (%llu rounds)\n",
                c->ann_slope_valid ? " moved" : "", sl, vc_slope(c), vc_recent(c->host, &c->host_any, n_seqs),
                vc_recent(c->draft, &c->draft_any, n_seqs), n_seqs, (unsigned long long)c->observed);
        c->ann_slope = sl;
        c->ann_slope_valid = true;
        c->ann_at = c->observed;
    }
}

void pulsar_engine_verify_curve_view(const pulsar_engine *e, pulsar_verify_curve_view *out) {
    memset(out, 0, sizeof *out);
    const pulsar_verify_curve *c = e ? e->verify_curve : NULL;
    if (!c) return;
    out->valid = true;
    out->cached = c->cached;
    out->rows_max = c->rows_max;
    for (uint32_t r = 1; r <= c->rows_max; r++) out->base_ms[r] = (float)c->base_ms[r];
    double sl = 0.0;
    out->slope_valid = c->slope.get(&sl);
    out->slope_ms = (float)sl;
    out->host_ms = (float)vc_recent(c->host, &c->host_any, PULSAR_SPEC_LOGITS_ROWS + 1u);
    out->draft_ms = (float)vc_recent(c->draft, &c->draft_any, PULSAR_SPEC_LOGITS_ROWS + 1u);
    out->observed = c->observed;
    out->key = c->key;
}

/* ---- the profile and its cache ----------------------------------------------------------------------------- */

/* What a profile depends on: the container (and an external drafter's), the drafter, the widest step, the bank pool
 * and context the session was built with, and the build that runs the kernels. */
static uint64_t vc_key(const pulsar_session *s, const char *build_id) {
    const pulsar_engine *e = s->engine;
    uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&](const void *p, size_t n) {
        const uint8_t *b = (const uint8_t *)p;
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t)b[i];
            h *= 0x100000001b3ull;
        }
    };
    const char tag[] = "pulsar-verify-curve 1";
    mix(tag, sizeof tag);
    mix(&e->model.container_digest, sizeof e->model.container_digest);
    if (e->dspark_external) mix(&e->dspark_model.container_digest, sizeof e->dspark_model.container_digest);
    const char *dn = e->drafter_ops && e->drafter_ops->name ? e->drafter_ops->name : "";
    mix(dn, strlen(dn) + 1u);
    const uint32_t v[3] = {pulsar_engine_fused_heads_max(e), (uint32_t)pulsar_session_bank_count((pulsar_session *)s),
                           (uint32_t)pulsar_session_ctx((pulsar_session *)s)};
    mix(v, sizeof v);
    const char *b = build_id ? build_id : "";
    mix(b, strlen(b) + 1u);
    /* the executable itself: a rebuild the build id does not name (a dirty tree, an object the id was not
     * recompiled into) is another build */
    struct stat st;
    if (stat("/proc/self/exe", &st) == 0) {
        const int64_t id[2] = {(int64_t)st.st_size, (int64_t)st.st_mtime};
        mix(id, sizeof id);
    }
    return h;
}

/* A cached curve older than this is profiled again (a driver or clock change the key cannot see). */
#define VC_CACHE_TTL_S (7 * 24 * 3600)

static bool vc_cache_read(const char *path, uint64_t key, uint32_t rows_max, double *base) {
    struct stat st;
    if (stat(path, &st) != 0 || time(NULL) - st.st_mtime > VC_CACHE_TTL_S) return false;
    FILE *f = fopen(path, "r");
    if (!f) return false;
    unsigned version = 0, rows = 0;
    unsigned long long k = 0;
    bool ok = fscanf(f, "pulsar-verify-curve %u key %llx rows %u", &version, &k, &rows) == 3 && version == 1u &&
              k == key && rows == rows_max;
    for (uint32_t r = 1; ok && r <= rows_max; r++) {
        unsigned rr = 0;
        double ms = 0.0;
        ok = fscanf(f, "%u %lf", &rr, &ms) == 2 && rr == r && ms > 0.0 && ms < 1e5;
        if (ok) base[r] = ms;
    }
    fclose(f);
    return ok;
}

static void vc_cache_write(const char *path, uint64_t key, uint32_t rows_max, const double *base) {
    char tmp[4200];
    if (snprintf(tmp, sizeof tmp, "%s.%d.tmp", path, (int)getpid()) >= (int)sizeof tmp) return;
    FILE *f = fopen(tmp, "w");
    if (!f) {
        fprintf(stderr, "pulsar: verify curve: cannot write the cache %s (%s); it is profiled again next start\n", tmp,
                strerror(errno));
        return;
    }
    fprintf(f, "pulsar-verify-curve 1 key %016llx rows %u\n", (unsigned long long)key, rows_max);
    for (uint32_t r = 1; r <= rows_max; r++) fprintf(f, "%u %.4f\n", r, base[r]);
    const bool ok = fflush(f) == 0 && fsync(fileno(f)) == 0;
    fclose(f);
    if (!ok || rename(tmp, path) != 0) {
        fprintf(stderr, "pulsar: verify curve: the cache %s was not written (%s)\n", path, strerror(errno));
        unlink(tmp);
    }
}

/* The profile (DeepInfer's profile_decode_perf): on the installed bank at its frontier, an r-row verify step for
 * every r from the widest down -- the production verify (heads on every row, the target's verify capture armed),
 * the target's frontier snapshot taken before it and put back after, exactly a round's begin and abort, so the bank
 * is where it was after each -- untimed until the width is WARM, then VC_PROFILE_REPS timed; the mean of the timed
 * ones less the fastest and the slowest.
 *
 * L306: the same walk is the startup warmup's catalog of the verify widths, so it runs on every start and times only
 * when `base` is non-NULL (no cached curve).  A width is warm at its first rep after the first that ran every graph
 * segment it entered from a captured graph -- the segment cache's misses did not move (pulsar_gpu_seg_stats) -- and
 * that rep is the first timed one.  Inside the warmup's catalog window a shape's first use warms its scratch and its
 * second captures it (DeepSeek's output head alternates two carrier parities: four uses), so the timed reps are the
 * replays production runs, not the eager first uses (before L306 every width ran six times under the live capture
 * threshold of twelve: a graphed lane's curve priced its eager forward).  A lane with no segment is warm at its second
 * rep -- one untimed, as before; with no timing wanted, a width whose first rep entered no segment is done (DeepSeek's
 * head is graphed at 16 rows and below only). */
#define VC_PROFILE_REPS 5u
#define VC_WARM_MAX 8u
static bool vc_profile(pulsar_session *s, uint32_t rows_max, double *base, char *err, size_t errlen) {
    const uint32_t bank = pulsar_session_live_bank(s);
    const int P = pulsar_session_pos(s);
    const uint32_t V = spec_vocab(s);
    if (P < 1 || (int64_t)P + rows_max > (int64_t)pulsar_session_ctx(s)) {
        snprintf(err, errlen, "the bank holds %d tokens: no room for a %u-row verify", P, rows_max);
        return false;
    }
    const pulsar_spec_target_ops *t = spec_target(s);
    std::vector<float> logits((size_t)rows_max * V);
    pulsar_multiseq_req reqs[PULSAR_SPEC_LOGITS_ROWS];
    pulsar_spec_round *r = pulsar_spec_round_new();
    bool ok = true;
    pulsar_session_bank_state_save(s, bank);
    for (uint32_t rows = rows_max; ok && rows >= 1u; rows--) {
        for (uint32_t k = 0; k < rows; k++) reqs[k] = {bank, (int32_t)(P + (int)k), s->checkpoint.v[k % (uint32_t)P]};
        double ms[VC_PROFILE_REPS];
        uint32_t timed = 0;
        bool warm = false;
        bool segments = true;   /* this width enters a graph segment (known after its first rep) */
        for (uint32_t rep = 0;
             ok && segments && rep < VC_WARM_MAX + VC_PROFILE_REPS && (base ? timed < VC_PROFILE_REPS : !warm); rep++) {
            pulsar_gpu_seg_stats g0, g1;
            pulsar_gpu_seg_stats_get(&g0);
            ok = pulsar_session_bank_state_restore(s, bank) &&
                 (!t->snapshot || t->snapshot(s, r, err, errlen));
            if (!ok) break;
            pulsar_session_bank_state_save(s, bank);
            pulsar_session_spec_arm_capture(s, rows);
            uint32_t got = 0;
            const double t0 = now_sec();
            ok = pulsar_session_decode_mixed(s, reqs, rows, logits.data(), (int)((size_t)rows * V), &got,
                                             PULSAR_MSEQ_HEAD_ALL_ROWS, err, errlen) == 0 && got == rows;
            const double t1 = now_sec();
            pulsar_session_spec_arm_capture(s, 0u);
            if (!ok) break;
            pulsar_gpu_seg_stats_get(&g1);
            if (!warm && rep >= 1u && g1.misses == g0.misses) warm = true;
            if (warm && base) {
                ms[timed++] = (t1 - t0) * 1e3;
            } else if (!warm && rep + 1u == VC_WARM_MAX) {
                /* rule 5: a width whose segments never all replay is timed as it runs, and says so */
                warm = true;
                fprintf(stderr, "pulsar: verify curve: the %u-row verify still missed the graph cache after %u reps "
                                "(%llu captured, %llu live): timed as it runs\n",
                        rows, VC_WARM_MAX, (unsigned long long)g1.captures, (unsigned long long)g1.live_captures);
            }
            if (!base && rep == 0u && g1.misses == g0.misses) segments = false;   /* nothing to capture here */
            ok = pulsar_session_bank_state_restore(s, bank) && (!t->restore || t->restore(s, r));
            pulsar_session_bank_state_save(s, bank);
            if (!ok) snprintf(err, errlen, "the bank could not be put back after a %u-row verify", rows);
        }
        if (!ok || !base) continue;
        std::sort(ms, ms + VC_PROFILE_REPS);
        double sum = 0.0;
        for (uint32_t i = 1; i + 1u < VC_PROFILE_REPS; i++) sum += ms[i];
        base[rows] = sum / (double)(VC_PROFILE_REPS - 2u);
    }
    if (t->release) t->release(s, r);
    pulsar_spec_round_free(r);
    (void)pulsar_session_bank_state_restore(s, bank);
    return ok;
}

int pulsar_session_verify_curve_open(pulsar_session *s, const char *cache_dir, const char *build_id, char *err,
                                     size_t errlen) {
    pulsar_engine *e = s ? s->engine : NULL;
    if (!e || e->tp || !pulsar_engine_has_spec_rounds(e) || !pulsar_engine_has_fused_step(e) ||
        pulsar_session_bank_count(s) < 2) {
        snprintf(err, errlen, "no batched speculative lane to price (tensor-parallel, no drafter, or no bank pool)");
        return -1;
    }
    const uint32_t rows_max = pulsar_engine_fused_heads_max(e);
    if (rows_max == 0u || rows_max > PULSAR_SPEC_LOGITS_ROWS) {
        snprintf(err, errlen, "a verify width of %u rows is outside the landing buffers (%u)", rows_max,
                 PULSAR_SPEC_LOGITS_ROWS);
        return -1;
    }
    double base[PULSAR_SPEC_LOGITS_ROWS + 1] = {0};
    const uint64_t key = vc_key(s, build_id);
    char path[4096] = "";
    if (cache_dir && cache_dir[0])
        snprintf(path, sizeof path, "%s/%016llx.curve", cache_dir, (unsigned long long)key);
    const double t0 = now_sec();
    const bool cached = path[0] && vc_cache_read(path, key, rows_max, base);
    /* L306: the walk runs on every start -- it is the warmup's catalog of the verify widths -- and times them only
     * without a cached curve; then the drafter's own captured shapes (its draft pass's bank counts and bounds) */
    if (!vc_profile(s, rows_max, cached ? NULL : base, err, errlen)) return -1;
    if (!cached && path[0]) vc_cache_write(path, key, rows_max, base);
    if (e->drafter_ops && e->drafter_ops->warm && e->drafter_ops->warm(s, err, errlen) != 0) return -1;
    pulsar_verify_curve *c = pulsar_verify_curve_new(rows_max, base);
    c->key = key;
    c->cached = cached;
    pulsar_verify_curve_free(e->verify_curve);
    e->verify_curve = c;
    /* rule 5: the curve the pick prices with, and where it came from */
    char line[1024];
    int at = 0;
    for (uint32_t r = 1; r <= rows_max && at < (int)sizeof line - 16; r++)
        at += snprintf(line + at, sizeof line - (size_t)at, " %.1f", c->base_ms[r]);
    fprintf(stderr, "pulsar: verify curve %s (key %016llx, %.1f s): ms at 1..%u rows:%s\n",
            c->cached ? "loaded from the cache" : "profiled", (unsigned long long)c->key, now_sec() - t0, rows_max,
            line);
    return 0;
}

/* ---- confidence calibration (rule 5 for the confidences every pick ranks by) --------------------------------- */

void pulsar_spec_calibration_note(pulsar_engine *e, bool copy, const float *conf, const float *label,
                                  const uint8_t *hit, uint32_t n) {
    pulsar_spec_calibration &a = e->spec_calibration;
    const uint32_t d = copy ? 1u : 0u;
    for (uint32_t i = 0; i < n && i < 16u; i++) {
        if (conf[i] < 0.0f) continue;   /* no confidence: nothing to calibrate */
        const double c = conf[i] > 1.0f ? 1.0 : (double)conf[i], y = (double)label[i];
        a.n[d][i]++;
        a.conf[d][i] += c;
        a.label[d][i] += y;
        a.hit[d][i] += hit[i] ? 1.0 : 0.0;
        a.abs_err[d][i] += fabs(c - y);
        a.sq_err[d][i] += (c - y) * (c - y);
        const uint32_t b = c >= 1.0 ? 9u : (uint32_t)(c * 10.0);
        a.bin_n[d][b]++;
        a.bin_conf[d][b] += c;
        a.bin_label[d][b] += y;
    }
}

/* ---- L304: the fused step's prompt cost ------------------------------------------------------------------------
 *
 * After DeepInfer's prefill load balancer (doc/inference_deployment_qa.md, "within a single GPU"): a chunk of a
 * prompt that does not finish it is not a fixed width but the widest that fits a time budget, solved from a fitted
 * cost.  A fused step's forward carries the decode banks' verify rows and each riding prompt's chunk of q rows over
 * the c rows of it already computed; the chunk's rows each pay the dense and expert work (linear in q) and attend
 * over c + their own causal prefix (q * (c + q / 2) pairs).  So
 *
 *     forward_ms = a + alpha * sum q + beta * sum q (c + q / 2)
 *
 * with a the verify rows and the step's fixed cost.  An exponentially weighted least-squares fit over the fused steps
 * the server reports (weight 1 - 1/64 a step: a prompt's chunks move c through the window); beta clamps at 0 (a
 * short context's attention is lost in the noise: the two-term fit then).  No number until 8 steps are in and the
 * terms are positive. */
#define FC_DECAY (1.0 - 1.0 / 64.0)
#define FC_MIN_OBS 8u

static bool fc_solve(const pulsar_fused_cost_fit *f, double *a, double *alpha, double *beta) {
    /* the normal equations of [1, x1, x2] */
    const double m[3][3] = {{f->w, f->s1, f->s2}, {f->s1, f->s11, f->s12}, {f->s2, f->s12, f->s22}};
    const double v[3] = {f->sy, f->s1y, f->s2y};
    const auto det3 = [](const double (&x)[3][3]) {
        return x[0][0] * (x[1][1] * x[2][2] - x[1][2] * x[2][1]) - x[0][1] * (x[1][0] * x[2][2] - x[1][2] * x[2][0]) +
               x[0][2] * (x[1][0] * x[2][1] - x[1][1] * x[2][0]);
    };
    const double d = det3(m);
    if (fabs(d) > 1e-9 * (f->w * f->s11 * f->s22 + 1.0)) {
        double r[3];
        for (int k = 0; k < 3; k++) {
            double mk[3][3];
            memcpy(mk, m, sizeof mk);
            for (int i = 0; i < 3; i++) mk[i][k] = v[i];
            r[k] = det3(mk) / d;
        }
        if (r[1] > 0.0 && r[2] >= 0.0) {
            *a = r[0];
            *alpha = r[1];
            *beta = r[2];
            return true;
        }
    }
    /* the two-term fit [1, x1] */
    const double d2 = f->w * f->s11 - f->s1 * f->s1;
    if (!(d2 > 1e-9 * (f->w * f->s11 + 1.0))) return false;
    *alpha = (f->w * f->s1y - f->s1 * f->sy) / d2;
    *a = (f->sy - *alpha * f->s1) / f->w;
    *beta = 0.0;
    return *alpha > 0.0;
}

void pulsar_engine_fused_cost_observe(pulsar_engine *e, uint32_t n_runs, const uint32_t *q, const uint32_t *c,
                                      double forward_ms) {
    if (!e || e->tp || n_runs == 0u || !(forward_ms > 0.0)) return;
    double x1 = 0.0, x2 = 0.0;
    for (uint32_t i = 0; i < n_runs; i++) {
        x1 += (double)q[i];
        x2 += (double)q[i] * ((double)c[i] + 0.5 * (double)q[i]);
    }
    x2 *= 1e-6;   /* in M pairs: the normal equations stay well scaled */
    pulsar_fused_cost_fit *f = &e->fused_cost;
    f->w = f->w * FC_DECAY + 1.0;
    f->s1 = f->s1 * FC_DECAY + x1;
    f->s2 = f->s2 * FC_DECAY + x2;
    f->s11 = f->s11 * FC_DECAY + x1 * x1;
    f->s12 = f->s12 * FC_DECAY + x1 * x2;
    f->s22 = f->s22 * FC_DECAY + x2 * x2;
    f->sy = f->sy * FC_DECAY + forward_ms;
    f->s1y = f->s1y * FC_DECAY + x1 * forward_ms;
    f->s2y = f->s2y * FC_DECAY + x2 * forward_ms;
    f->n++;
    double a = 0.0, al = 0.0, be = 0.0;
    f->valid = f->n >= FC_MIN_OBS && fc_solve(f, &a, &al, &be);
    if (!f->valid) return;
    f->a_ms = a > 0.0 ? a : 0.0;
    f->alpha_ms = al;
    f->beta_ms = be * 1e-6;
    /* rule 5: the fit announces itself when it arms and, at most once in 256 steps, when a term moved by half */
    const bool first = f->ann_n == 0u;
    const bool moved = !first && f->n - f->ann_n >= 256u &&
                       (fabs(f->alpha_ms - f->ann_alpha) * 2.0 > f->ann_alpha ||
                        fabs(f->beta_ms - f->ann_beta) * 2.0 > f->ann_beta + 1e-12);
    if (first || moved) {
        f->ann_n = f->n;
        f->ann_alpha = f->alpha_ms;
        f->ann_beta = f->beta_ms;
        fprintf(stderr, "pulsar: fused step cost measured: %.1f + %.4f x rows + %.3f x M pairs ms (%u steps%s)\n",
                f->a_ms, f->alpha_ms, f->beta_ms * 1e6, f->n, first ? "" : ", moved");
    }
}

double pulsar_engine_fused_cost_ms(const pulsar_engine *e, uint32_t q, uint32_t c) {
    const pulsar_fused_cost_fit *f = e ? &e->fused_cost : NULL;
    if (!f || !f->valid) return 0.0;
    return f->alpha_ms * (double)q + f->beta_ms * (double)q * ((double)c + 0.5 * (double)q);
}

bool pulsar_engine_fused_chunk(const pulsar_engine *e, uint32_t c, double spent_ms, double budget_ms, uint32_t *rows) {
    const pulsar_fused_cost_fit *f = e ? &e->fused_cost : NULL;
    *rows = 0u;
    if (!f || !f->valid) return false;
    const double room = budget_ms - f->a_ms - spent_ms;
    if (!(room > 0.0)) return true;
    /* the largest q with alpha q + beta q (c + q / 2) <= room */
    const double lin = f->alpha_ms + f->beta_ms * (double)c;
    const double q = f->beta_ms > 0.0 ? (-lin + sqrt(lin * lin + 2.0 * f->beta_ms * room)) / f->beta_ms : room / lin;
    *rows = q >= 4294967295.0 ? 4294967295u : (uint32_t)q;
    return true;
}

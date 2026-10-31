/* L312 -- the fused lane's round selection: value per time, with a decode-age price and token-fairness prices.
 *
 * After NVIDIA Flare's scheduler (docs/scheduling-utility.md, helios/src/scheduler/{policy,utility}.rs), ported in
 * design, not in code.  Every spec round the server may carry chunks of the prefilling prompts beside the decoders'
 * verify rows (L260 fusion).  Which chunks, and how wide, is one rule: of every candidate round -- decode only, or
 * decode plus some prompts' chunks -- run the one with the most value per time, net of what the round makes the
 * running streams wait:
 *
 *     score(a) = (V(a) + c * T_dec * sum_i A_i) / T(a) - c / 2 * n * T(a)
 *
 *   - V(a): each stream's expected tokens (1 + the survivals the verify pick verifies; the same in every candidate)
 *     at its price out_value + f_d * inv_i, a finishing chunk's first token at out_value, and each prompt row at its
 *     price 1 + f_p * inv_j.  Value is counted in prompt tokens.
 *   - T(a): the round's predicted wall on the engine's MEASURED costs, the one authority for each: a decode-only
 *     round on the L301 verify curve, round_ms(n, rows); a fused round as that round plus what its prompt chunks add
 *     to it (L327: the prompt chunk cost fit's fused terms, pulsar_engine_fused_prompt_ms -- the fit takes the verify
 *     rows' forward off each fused step it observes, on the same curve).  No cost model of its own, no constant of a
 *     deployment.
 *   - The age terms (Flare's decode-age continuation): c prices a stream's waiting per ms^2, so holding n streams
 *     for T costs c (A T + T^2 / 2) each; dividing by T and dropping what every candidate shares leaves the two terms
 *     above (A_i: stream i's time since its last output, T_dec: the reference round, the decode-only one, since
 *     every live decoder rides every round).  This is what bounds the decoders' gaps: there is no deadline and no
 *     chunk ceiling -- a wider chunk rides exactly while its prompt's value pays for the longer wait it causes.
 *   - inv: a request's token-fairness price over its coefficient (sched_service_observe): Flare's service-weighted
 *     price, the mirror-descent flow dp/dt = (f - p y) / (m tau) of the log-utility sum f log(y) integrated exactly
 *     over each round.  A request that waits sees its price grow linearly with the time it waits; one that is served
 *     sees it settle to f / (its service rate).  So a waiting prompt's chunk always comes to outbid the decode-only
 *     round (no prompt starves -- L277: a second long prompt got 0 rows for 233 s in slot order), two long prompts
 *     share the rows by their prices, and a stream that a long prefill slows sees its own price rise against it.
 *   - The coefficients, per class: the fairness f is split evenly among the decoding streams (f_d = f / n) and among
 *     the prefilling prompts (f_p = f w / n_p), w the prompts' weight as a class.  Flare prices every request alike;
 *     here one long prompt beside four streams would then get a fifth of the device, and four prompts beside one
 *     stream four fifths -- a class split keeps decode against prefill where the preference puts it whatever the
 *     count on either side.
 *
 * Flare's delivery prices drop out: every live decoder rides every round (its base row is never cut), so each
 * candidate delivers to every stream once and there is no decode cohort to choose.
 *
 * The selection: the decode-only round is the baseline; then greedy moves, each the single change (one prompt's
 * chunk to a wider candidate end, from none included) that raises the score most, until none does.  A prompt's
 * candidate ends are the ones the fused step admits (L260, L264): any multiple of SCHED_CHUNK_ALIGN inside the
 * prompt, and its end when no resume-grid point lies inside the chunk (the bank keeps a checkpoint there); within
 * the round's prompt rows and, for a finishing chunk, a head row left.  The first move carries the fused step's
 * fixed cost, which is why it may jump to any end.
 *
 * Unpriced (no fit yet -- the first fused steps -- or no verify curve -- none opened): no number, no
 * decision (L301's rule): a prompt's rest rides whole when it fits the round, and a continuation rides
 * probe_rows on the first round of a full decode quantum (L260's decode share), the prompt that rode least recently
 * first.  Each such round is an observation the fit is made of; the server alternates two probe widths so the fit
 * can separate its fixed term from its per-row one. */
/** @file
 *  The fused lane's value-per-time round selection (L312): which prompt chunks ride the next spec round.
 */
#include "sched_value.h"

#include <math.h>
#include <string.h>

#include <algorithm>

namespace {

/** One candidate chunk of a prompt. */
struct cand {
    /// Rows the chunk carries (from the prompt's frontier p0).
    uint32_t q;
    /// The chunk reaches the prompt's end (it needs a head row).
    bool fin;
};

/* The prompt's candidate ends, narrowest first, within `room` rows. */
/** @param p the prompt
 *  @param room the round's prompt rows left
 *  @param out the candidates, narrowest first
 *  @param cap entries at `out`
 *  @return the candidates written (0: the prompt is done or there is no room) */
static int sched_cands(const sched_prompt *p, uint32_t room, cand *out, int cap) {
    int k = 0;
    if (p->len <= p->p0 || room == 0u) return 0;
    const uint32_t left = p->len - p->p0;
    for (uint32_t e = (p->p0 / SCHED_CHUNK_ALIGN + 1u) * SCHED_CHUNK_ALIGN; e < p->len && k < cap;
         e += SCHED_CHUNK_ALIGN) {
        if (e - p->p0 > room) break;
        out[k++] = {e - p->p0, false};
    }
    const uint32_t g = p->grid ? p->len / p->grid * p->grid : 0u;
    const bool grid_inside = g > p->p0 && g < p->len;
    if (left <= room && !grid_inside && k < cap) out[k++] = {left, true};
    return k;
}

/** Room for one prompt's candidate chunks (sched_cands' cap). */
#define SCHED_CANDS_MAX 64

}  // namespace

double sched_prompt_rate_max(const pulsar_engine *e, const sched_round *r, const sched_prompt *pr,
                             uint32_t room_rows) {
    cand c[SCHED_CANDS_MAX];
    const int k = sched_cands(pr, room_rows, c, SCHED_CANDS_MAX);
    if (k == 0) return 0.0;
    double outside = 0.0;
    const double base =
            pulsar_engine_verify_round_ms(e, r->n_live ? r->n_live : 1u, r->n_live ? r->n_live : 1u, &outside);
    if (!(base > 0.0)) return 0.0;
    const uint32_t q = c[k - 1].q, at = pr->p0;
    double ms = 0.0;
    if (!pulsar_engine_fused_prompt_ms(e, 1u, &q, &at, &ms)) return 0.0;
    return (double)q / (base + ms);
}

double sched_decode_rate_max(const pulsar_engine *e, double tokens, uint32_t rows) {
    double outside = 0.0;
    const double t = pulsar_engine_verify_round_ms(e, 1u, rows ? rows : 1u, &outside);
    return t > 0.0 ? tokens / t : 0.0;
}

void sched_service_observe(double *inv_ms, double t_ms, double tokens, double max_rate, double adapt_ms) {
    if (!(*inv_ms > 0.0) || !(t_ms > 0.0) || !(max_rate > 0.0) || !(adapt_ms > 0.0)) return;
    if (tokens > 0.0) {
        const double d = tokens / max_rate / adapt_ms;
        *inv_ms = *inv_ms * exp(-d) + t_ms / tokens * -expm1(-d);
    } else {
        *inv_ms += t_ms / (max_rate * adapt_ms);
    }
}

void sched_choose(const pulsar_engine *e, const sched_prefs *p, const sched_round *r, const sched_prompt *pr,
                  int n, uint32_t room_rows, bool first_full_round, uint32_t probe_rows, sched_choice *out) {
    memset(out, 0, sizeof *out);
    if (n > SCHED_PROMPTS_MAX) n = SCHED_PROMPTS_MAX;
    const uint32_t max_fins = r->heads_max > r->n_dec ? r->heads_max - r->n_dec : 0u;
    double outside = 0.0;
    const double t_dec = r->n_live ? pulsar_engine_verify_round_ms(e, r->n_live, r->rows, &outside) : 0.0;
    double probe_ms = 0.0;
    const uint32_t zq = 0u, zc = 0u;
    out->priced = t_dec > 0.0 && pulsar_engine_fused_prompt_ms(e, 1u, &zq, &zc, &probe_ms);
    out->t_dec_ms = t_dec;

    if (!out->priced) {
        /* no number, no decision: the rest whole, a probe continuation on a full quantum's first round */
        int order[SCHED_PROMPTS_MAX];
        for (int i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order, order + n, [&](int a, int b) { return pr[a].last_ride < pr[b].last_ride; });
        uint32_t room = room_rows, fins = 0u;
        for (int oi = 0; oi < n; oi++) {
            const int i = order[oi];
            cand c[SCHED_CANDS_MAX];
            const int k = sched_cands(&pr[i], room, c, SCHED_CANDS_MAX);
            if (k == 0) continue;
            const uint32_t left = pr[i].len - pr[i].p0;
            int pick = -1;
            if (left <= room) {
                /* the rest fits: it rides -- to its last grid point when one lies inside it (L264: the bank keeps a
                 * checkpoint there; the tail rides the next round), else whole, with a head row */
                const uint32_t g = pr[i].grid ? pr[i].len / pr[i].grid * pr[i].grid : 0u;
                if (!(g > pr[i].p0 && g < pr[i].len)) pick = fins < max_fins ? k - 1 : -1;
                else
                    for (int j = 0; j < k; j++)
                        if (!c[j].fin && pr[i].p0 + c[j].q <= g) pick = j;
            } else if (first_full_round) {
                for (int j = 0; j < k && c[j].q <= probe_rows; j++) pick = j;
                if (pick < 0 && room == room_rows) pick = 0;   /* alone, its narrowest end still moves it */
            }
            if (pick < 0) continue;
            out->q[i] = c[pick].q;
            out->fin[i] = c[pick].fin;
            room -= c[pick].q;
            fins += c[pick].fin ? 1u : 0u;
        }
        return;
    }

    const double f_dec = r->n_live ? p->fairness / (double)r->n_live : 0.0;
    const double f_pr = n ? p->fairness * p->prompt_weight / (double)n : 0.0;
    double dec_value = 0.0;
    for (uint32_t i = 0; i < r->n_live && i < SCHED_PROMPTS_MAX; i++)
        dec_value += (p->out_value + f_dec * (r->inv_ms[i] > 0.0 ? r->inv_ms[i] : 0.0)) * r->tokens[i];
    const double age_credit = p->age_price * t_dec * r->age_ms;
    const double hold = 0.5 * p->age_price * (double)r->n_live;
    const auto score = [&](double v, double t) { return (v + age_credit) / t - hold * t; };
    out->score_dec = score(dec_value, t_dec);
    out->t_ms = t_dec;
    out->score = out->score_dec;

    cand cs[SCHED_PROMPTS_MAX][SCHED_CANDS_MAX];
    int nc[SCHED_PROMPTS_MAX], sel[SCHED_PROMPTS_MAX];
    double price[SCHED_PROMPTS_MAX];
    for (int i = 0; i < n; i++) {
        nc[i] = sched_cands(&pr[i], room_rows, cs[i], SCHED_CANDS_MAX);
        sel[i] = -1;
        price[i] = 1.0 + f_pr * (pr[i].inv_ms > 0.0 ? pr[i].inv_ms : 0.0);
    }
    uint32_t rows = 0u, fins = 0u;
    double value = dec_value;
    for (;;) {
        int bi = -1, bk = -1;
        double best = out->score, best_t = 0.0;
        for (int i = 0; i < n; i++) {
            const uint32_t q0 = sel[i] >= 0 ? cs[i][sel[i]].q : 0u;
            const bool f0 = sel[i] >= 0 && cs[i][sel[i]].fin;
            const double v0 = sel[i] >= 0 ? price[i] * q0 + (f0 ? p->out_value : 0.0) : 0.0;
            for (int k = sel[i] + 1; k < nc[i]; k++) {
                const cand &c = cs[i][k];
                if (rows - q0 + c.q > room_rows) break;
                if (c.fin && !f0 && fins >= max_fins) continue;
                uint32_t qq[SCHED_PROMPTS_MAX], cc[SCHED_PROMPTS_MAX], m = 0u;
                for (int j = 0; j < n; j++) {
                    const uint32_t qj = j == i ? c.q : sel[j] >= 0 ? cs[j][sel[j]].q : 0u;
                    if (!qj) continue;
                    qq[m] = qj;
                    cc[m++] = pr[j].p0;
                }
                double add = 0.0;
                (void)pulsar_engine_fused_prompt_ms(e, m, qq, cc, &add);
                const double t = t_dec + add;
                const double v = value - v0 + price[i] * c.q + (c.fin ? p->out_value : 0.0);
                const double s = score(v, t);
                if (s > best) {
                    best = s;
                    best_t = t;
                    bi = i;
                    bk = k;
                }
            }
        }
        if (bi < 0) break;
        const cand &was = sel[bi] >= 0 ? cs[bi][sel[bi]] : cand{0u, false};
        const cand &now = cs[bi][bk];
        value += price[bi] * ((double)now.q - (double)was.q) + ((now.fin ? 1.0 : 0.0) - (was.fin ? 1.0 : 0.0)) *
                 p->out_value;
        rows += now.q - was.q;
        fins += (now.fin ? 1u : 0u) - (was.fin ? 1u : 0u);
        sel[bi] = bk;
        out->score = best;
        out->t_ms = best_t;
    }
    for (int i = 0; i < n; i++) {
        if (sel[i] < 0) continue;
        out->q[i] = cs[i][sel[i]].q;
        out->fin[i] = cs[i][sel[i]].fin;
    }
}

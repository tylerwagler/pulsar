#!/usr/bin/env python3
"""L286: propose block allocations of the routed experts from the measured response curves.  CPU only.

    propose.py --table table.json --fullkl NOTES/research/l251/full-kl --out allocs.json

Each block of layers takes one gate_up K in {4, 5} (the engine's gate_up_fused arm reads only those) and one down K in
{3, 4, 5, 6} (down arm; K2 is admitted but left out: it sits at the far end of the measured curve).  Predicted cost of
a block's choice = curve_gu(x(K_gu)) + curve_dn(x(K_dn)), x = the block's proxy change in units of its U proxy.
Bytes: unit = one down bit on one layer = 512 x (640/16)(2560/16) x 16 x 2 B = 104.86 MB; gate_up K4 -> K5 is 2 units
per layer.  Exact multiple-choice knapsack over byte units.

A step below K4 is admitted only where the block's measured point at that step has a CI clear of 0: the early
blocks' K3 costs sit inside the chaotic floor, so they never drop below uniform on an unresolved number (L286 rule).

Bases:
  raw    the curves' point estimates
  fit    the point estimates with the two-scale sub-additivity correction fitted on L251's recipes (analyze.py):
         positive terms x alpha_costs, negative terms (improvements) x alpha_gains
  safe   every block's term at its CI upper bound: a block leaves K4 only when even the pessimistic end of the
         measurement says the move pays (upgrades must be resolved as gains, downgrades as cheap).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MD                                                             # noqa: E402

UNIT = 512 * (640 // 16) * (2560 // 16) * 16 * 2
OPTS_GU, OPTS_DN = (4, 5), (3, 4, 5, 6)


def solve(costs, budget):
    """min sum cost s.t. sum units <= budget; costs[block] = {(gu, dn): (cost, units)}."""
    best = {0: (0.0, [])}
    for b in sorted(costs):
        nxt = {}
        for u0, (c0, ch) in best.items():
            for opt, (c, u) in costs[b].items():
                cand = (c0 + c, ch + [(b, opt)])
                if u0 + u not in nxt or cand[0] < nxt[u0 + u][0]:
                    nxt[u0 + u] = cand
        best = nxt
    c, u, ch = min((v[0], u, v[1]) for u, v in best.items() if u <= budget)
    return c, u, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--fullkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewer-gb", type=float, default=2.0, help="the fewer-bytes proposal saves at least this much")
    a = ap.parse_args()
    T = json.load(open(a.table))
    rec = json.load(open(f"{a.fullkl}/results.json"))["record"]["layers"]
    R = T["ratios"]
    R["expert_vs_K4"] = {int(k): v for k, v in R["expert_vs_K4"].items()}
    curves = T["curves"]

    V = T["validation"]
    a_c, a_g = V["alpha_costs"], V["alpha_gains"]

    def term(c, x, basis):
        if basis == "safe":
            return max(MD.curve_eval(c, x, 2), MD.curve_eval(c, x, 3))
        v = MD.curve_eval(c, x, 1)
        return v if basis == "raw" else v * (a_c if v > 0 else a_g)

    def resolved_down(c, x):
        """The measured point nearest above x (the step's own measurement) has its CI clear of 0."""
        above = [p for p in c if p[0] >= x and p[0] > 0]
        return bool(above) and above[0][2] > 0

    def costs(basis):
        out = {}
        for key, c in curves.items():
            x0, x1 = map(int, key.split("-"))
            n = x1 - x0 + 1
            opts = {}
            for gu in OPTS_GU:
                for dn in OPTS_DN:
                    cost = 0.0
                    resolved = all(K >= 4 or resolved_down(c[f], MD.expert_x(rec, R, x0, x1, f, {
                        L: K for L in range(x0, x1 + 1)})) for f, K in (("gu", gu), ("dn", dn)))
                    if not resolved:
                        continue            # a step below K4 whose cost the measurement does not resolve: refused
                    for f, K in (("gu", gu), ("dn", dn)):
                        if K != 4:
                            x = MD.expert_x(rec, R, x0, x1, f, {L: K for L in range(x0, x1 + 1)})
                            cost += term(c[f], x, basis)
                    opts[(gu, dn)] = (cost, n * (2 * (gu == 5) + dn - 4))
            out[x0] = opts
        return out

    point = costs("raw")
    fit = costs("fit")
    props = {}
    fewer = -int(a.fewer_gb * 1e9 / UNIT + 0.999)
    print(f"two-scale correction from the known recipes: costs x {a_c:.3f}, gains x {a_g:.3f}")
    for name, basis, budget in (("A-raw", "raw", 0), ("A-fit", "fit", 0), ("A-safe", "safe", 0),
                                ("B-fit", "fit", fewer), ("B-safe", "safe", fewer)):
        c, u, ch = solve(costs(basis), budget)
        pred = sum(point[b][opt][0] for b, opt in ch)
        pred_fit = sum(fit[b][opt][0] for b, opt in ch)
        blocks = {b: opt for b, opt in ch}
        gu, dn = {}, {}
        for key in curves:
            x0, x1 = map(int, key.split("-"))
            for L in range(x0, x1 + 1):
                g, d = blocks[x0]
                if g != 4:
                    gu[L] = g
                if d != 4:
                    dn[L] = d
        props[name] = {"gate_up": gu, "down": dn, "units": u, "bytes_delta": u * UNIT, "predicted_dkl": pred, "predicted_dkl_fit": pred_fit,
                       "objective": c, "blocks": {str(b): list(opt) for b, opt in ch}}
        print(f"{name:8s} {u:+4d} units ({u * UNIT / 1e9:+.2f} GB)  predicted dKL raw {pred:+.3e} fit {pred_fit:+.3e}  "
              f"objective {c:+.3e}")
        print("          blocks (gate_up K, down K):", " ".join(f"{b}:{g}/{d}" for b, (g, d) in ch))
    json.dump({n: {"gate_up": p["gate_up"], "down": p["down"]} for n, p in props.items()}, open(a.out, "w"), indent=1)
    json.dump(props, open(os.path.splitext(a.out)[0] + ".detail.json", "w"), indent=1)


if __name__ == "__main__":
    main()

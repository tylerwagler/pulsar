#!/usr/bin/env python3
"""L286: gains table + the cost model checked against L251's measured recipes.  CPU only.

    analyze.py --gains-work DIR --fullkl NOTES/research/l251/full-kl [--block 4] [--out table.json]

Reads every gains_*.json / pertok_*.pt in DIR.  A branch is (first layer, last layer, family, s); its dKL is
KL(REF||branch) - KL(REF||U) per token, with a 95% bootstrap CI over the 48 sequences (paired).  For the chosen block
size it builds each (block, family)'s response curve (model.py) from every measured s, then predicts L251's recipes
R55 / R62 / R67 / R72 (`configs.json`) and compares with their measured full-model KL (`results.json`, paired ratios),
next to the old surrogate's prediction.
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MD                                                             # noqa: E402

SURROGATE = {"R72": 0.50, "R67": 0.62, "R62": 0.86, "R55": 1.51}   # L251 full-kl README, predicted / U
NAME = re.compile(r"^L(\d+)(?:-(\d+))?:(\w+):s([0-9.e-]+)$")


def load(work):
    import torch
    br, dense_local, noise, u_ref = {}, {}, {}, {}
    for p in sorted(glob.glob(f"{work}/gains_*.json")):
        if p.endswith(".partial.json"):
            continue
        r = json.load(open(p))
        pt = torch.load(f"{work}/pertok_{r['tag']}.pt")
        su = [float(x.sum()) for x in pt["ref_u"]]
        Ts = [x.numel() for x in pt["ref_u"]]
        for n in r["summary"]:
            m = NAME.match(n)
            a, b, f, s = int(m[1]), int(m[2] or m[1]), m[3], float(m[4])
            d = [float(x.sum()) - y for x, y in zip(pt[f"ref_{n}"], su)]
            pt_, lo, hi = MD.bootstrap_mean_diff(d, Ts)
            ent = {"dkl": pt_, "ci": [lo, hi], "kl_u_b": sum(float(x.sum()) for x in pt[f"u_{n}"]) / sum(Ts),
                   "src": r["tag"]}
            if f == "nz":
                noise[a] = ent
            elif s != 1.0:
                br[(a, b, f, s)] = ent
        for k, v in r["dense_local"].items():
            if k != "unhooked":
                dense_local[k] = v["rho_heldout"]
        for L, v in r["per_layer"].items():
            if v.get("u_vs_ref") is not None:
                u_ref[int(L)] = v["u_vs_ref"]
    return br, dense_local, noise, u_ref


def block_proxy(rec, dense_local, x0, x1, f):
    if f == "dense":
        return sum(MD.dense_proxy(dense_local, L) for L in range(x0, x1 + 1))
    return sum(MD.expert_proxy(rec, L, f) for L in range(x0, x1 + 1))


def dense_x(rec, dense_local, R, C, n, x0, x1):
    """The block's dense proxy change (units of its U proxy) under recipe n."""
    d = 0.0
    for L in range(x0, x1 + 1):
        for k, v in dense_local.items():
            if k.startswith(f"{MD.PFX}{L}."):
                fu, fr = C["U-e4-d5"][k], C[n][k]
                if fr in ("BF16", "F32"):
                    d -= v
                elif fr != fu:
                    assert fu == "EXL3_K5" and fr.startswith("EXL3"), (k, fu, fr)
                    d += v * (R["dense_vs_K5"][int(fr[-1])] - 1)
    return d / block_proxy(rec, dense_local, x0, x1, "dense")


def span(v):
    """[point, low, high] of a curve term (a CI bound evaluated at negative x swaps sides)."""
    return [v[0], min(v[1:]), max(v[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains-work", required=True)
    ap.add_argument("--fullkl", required=True)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = json.load(open(f"{a.fullkl}/results.json"))
    rec = R["record"]["layers"]
    C = json.load(open(f"{a.fullkl}/configs.json"))["configs"]
    br, dense_local, noise, u_ref = load(a.gains_work)
    ratios = MD.ratios(rec)
    print("universal proxy ratios:", json.dumps({k: ratios[k] for k in ("expert_vs_K4", "dense_vs_K5")}))
    print("  cv:", json.dumps(ratios["cv"]), "n:", json.dumps(ratios["n"]))
    print("\nevery branch: dKL = KL(REF||b) - KL(REF||U) per token [95% CI]; slope = dKL / x, x = s^2 - 1")
    rows = []
    for (x0, x1, f, s), e in sorted(br.items(), key=lambda t: (t[0][1] - t[0][0], t[0][2], t[0][3], t[0][0])):
        sig = "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else " "
        rows.append({"layers": [x0, x1], "family": f, "s": s, **e})
        print(f"  L{x0:2d}-{x1:2d} {f:5s} s={s:<4g} {e['dkl']:+.3e} [{e['ci'][0]:+.3e}, {e['ci'][1]:+.3e}]{sig} "
              f"KL(U||b) {e['kl_u_b']:.3e}  slope {e['dkl'] / (s * s - 1):+.2e}")
    for L, e in sorted(noise.items()):
        print(f"  floor control at L{L} (W_U x (1 + 5e-4 N)): {e['dkl']:+.3e} [{e['ci'][0]:+.3e}, {e['ci'][1]:+.3e}]"
              f"  KL(U||b) {e['kl_u_b']:.3e}")
    blocks = [(i, i + a.block - 1) for i in range(0, 48, a.block)]
    fams = [f for f in ("gu", "dn", "dense") if all(any(b[:3] == (x0, x1, f) for b in br) for x0, x1 in blocks)]
    curves = {f"{x0}-{x1}": {f: MD.curve_points(rows, x0, x1, f) for f in fams} for x0, x1 in blocks}
    out = {"ratios": ratios, "branches": rows, "noise": {str(k): v for k, v in noise.items()}, "u_vs_ref": u_ref,
           "block": a.block, "families": fams, "curves": curves,
           "proxy_u": {f"{x0}-{x1}": {f: block_proxy(rec, dense_local, x0, x1, f) for f in fams} for x0, x1 in blocks}}
    print(f"\nresponse curves, {a.block}-layer blocks: dKL at x (s^2 - 1)")
    for k, c in curves.items():
        print(f"  L{k:6s} " + "  ".join(f"{f}: " + " ".join(f"{x:+g}:{d:+.2e}" for x, d, _, _ in c[f] if x)
                                        for f in fams))
    if {"gu", "dn"} <= set(fams):
        kl_u = R["U-e4-d5"]["kl_mean"]
        su = R["per_seq"]["U-e4-d5"]["kl_sum"]
        print(f"\ncost model vs L251's measured full-model KL (L251's U-e4-d5 KL/token {kl_u:.4e})"
              + ("" if "dense" in fams else "  [EXPERTS ONLY: the recipes' dense moves are not priced]"))
        print(f"{'recipe':7} {'measured / U [CI]':>22} {'meas dKL':>10} {'model dKL':>10} {'[lo, hi]':>22} "
              f"{'experts':>10} {'dense':>10} {'surrogate':>9}")
        pts = []
        for n in ("R55", "R62", "R67", "R72"):
            terms = {}
            for x0, x1 in blocks:
                for f in ("gu", "dn"):
                    Ks = {L: int(C[n][f"{MD.PFX}{L}.{MD.EXPERT[f]}"][-1]) for L in range(x0, x1 + 1)}
                    x = MD.expert_x(rec, ratios, x0, x1, f, Ks)
                    terms[(x0, f)] = span([MD.curve_eval(curves[f"{x0}-{x1}"][f], x, w) for w in (1, 2, 3)])
                if "dense" in fams:
                    x = dense_x(rec, dense_local, ratios, C, n, x0, x1)
                    terms[(x0, "dense")] = span([MD.curve_eval(curves[f"{x0}-{x1}"]["dense"], x, w) for w in (1, 2, 3)])
            tot = [sum(v[i] for v in terms.values()) for i in range(3)]
            te = sum(v[0] for (b, f), v in terms.items() if f != "dense")
            r, lo, hi = MD.bootstrap_ratio(R["per_seq"][n]["kl_sum"], su)
            meas = (r - 1) * kl_u
            pts.append({"recipe": n, "measured_ratio": [r, lo, hi], "measured_dkl": meas, "model_dkl": tot[0],
                        "model_ci_sum": [tot[1], tot[2]], "experts": te, "dense": tot[0] - te,
                        "terms": {f"{b}:{f}": v[0] for (b, f), v in terms.items()}})
            print(f"{n:7} {r:6.3f} [{lo:.3f}, {hi:.3f}] {meas:+10.2e} {tot[0]:+10.2e} [{tot[1]:+.2e}, {tot[2]:+.2e}] "
                  f"{te:+10.2e} {tot[0] - te:+10.2e} {SURROGATE[n]:9.2f}")
        meas = [p["measured_dkl"] for p in pts]
        pred = [p["model_dkl"] for p in pts]
        sur = [SURROGATE[p["recipe"]] for p in pts]
        alpha = sum(x * y for x, y in zip(pred, meas)) / sum(x * x for x in pred)
        print(f"Spearman vs measured: model {MD.spearman(pred, meas):+.2f}, old surrogate {MD.spearman(sur, meas):+.2f}; "
              f"model sign right {sum((x > 0) == (y > 0) for x, y in zip(pred, meas))}/4, "
              f"old surrogate {sum((x > 1) == (y > 0) for x, y in zip(sur, meas))}/4")
        print(f"scale alpha = {alpha:.3f} (measured = alpha x model, least squares through 0): " +
              ", ".join(f"{p['recipe']} {alpha * p['model_dkl']:+.2e} vs {p['measured_dkl']:+.2e}" for p in pts))
        # sub-additivity, two scales: costs (positive terms) and gains (negative terms) saturate differently
        P = [sum(v for v in p["terms"].values() if v > 0) for p in pts]
        N = [sum(v for v in p["terms"].values() if v < 0) for p in pts]
        spp, snn, spn = sum(x * x for x in P), sum(x * x for x in N), sum(x * y for x, y in zip(P, N))
        smp, smn = sum(x * y for x, y in zip(meas, P)), sum(x * y for x, y in zip(meas, N))
        det = spp * snn - spn * spn
        a_c, a_g = (smp * snn - smn * spn) / det, (smn * spp - smp * spn) / det
        print(f"two-scale fit: measured = {a_c:.3f} x costs + {a_g:.3f} x gains: " +
              ", ".join(f"{p['recipe']} {a_c * x + a_g * y:+.2e} (costs {x:+.2e}, gains {y:+.2e})"
                        for p, x, y in zip(pts, P, N)))
        out["validation"] = {"points": pts, "alpha": alpha, "alpha_costs": a_c, "alpha_gains": a_g,
                             "spearman_model": MD.spearman(pred, meas), "spearman_surrogate": MD.spearman(sur, meas)}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

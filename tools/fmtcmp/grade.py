#!/usr/bin/env python3
"""Grade fmtcmp captures: KL(BF16 || X) per token, NLL, top-1 agreement, per class, with document-bootstrap CIs.

Every capture is a directory of ds4-fidelity-jsonl-v1 files (one per suite document, from
tools/fidelity/capture_vllm_prompt_logprobs.py or capture_exl3.py), all teacher-forcing the frozen suite's ids.

  grade.py --suite SUITE --ref caps/bf16 nvfp4=caps/nvfp4 k4=caps/exl3-k4 k5=caps/exl3-k5
  grade.py --suite SUITE nvfp4=caps/nvfp4            (no reference yet: absolute NLL only)
  grade.py --suite SUITE --ref caps/nvfp4 k4=...     (any capture can stand in as the reference; say so)

Numbers:
  - KL: tools/fidelity/fidelity_compare.trunc_kl (union of both top-K supports + one tail bucket; the one authority
    for top-K KL in this repo).  A lower bound on the full-vocab KL; every arm goes through the same estimator at
    the same K (64), so the arms compare like for like.  "miss" counts positions where the reference's top-1 token
    is outside the candidate's top-K (the estimator is weakest there).
  - NLL: -logprob of the actual next token (vLLM returns it even outside top-K; capture_exl3 always writes it).
  - top-1: argmax agreement with the reference.
  - Means are per token (L251's KL/token).  CIs: bootstrap over documents (10,000 draws), token-weighted; ratios
    and differences between arms are paired (the same draws).
Groups: chat48 (the L251 held-out instrument), each chat class, code, story (with depth buckets), all.
Refuses (exit 1) when any capture's token ids differ from the suite's: a mis-paired position is not a result.
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fidelity"))
from fidelity_compare import load, trunc_kl  # noqa: E402

STORY_BUCKETS = [2048, 8192, 16384]


def doc_stats(suite_ids, ref, cand):
    """Per-position arrays for one document.  ref may be None (NLL only)."""
    ch, crows = cand
    if ch.get("tokens") != suite_ids:
        return None, "candidate token ids differ from the suite"
    out = {"pos": [], "nll_c": [], "nll_r": [], "kl": [], "top1": [], "miss": []}
    if ref is not None:
        rh, rrows = ref
        if rh.get("tokens") != suite_ids:
            return None, "reference token ids differ from the suite"
    for p in range(1, len(suite_ids)):
        c = crows.get(p)
        if c is None or c.get("logprob") is None:
            return None, f"candidate has no row / forced logprob at position {p}"
        out["pos"].append(p)
        out["nll_c"].append(-c["logprob"])
        if ref is not None:
            r = rrows.get(p)
            if r is None or r.get("logprob") is None:
                return None, f"reference has no row / forced logprob at position {p}"
            out["nll_r"].append(-r["logprob"])
            out["kl"].append(trunc_kl(r["topk"], c["topk"]))
            out["top1"].append(1 if r["topk"][0][0] == c["topk"][0][0] else 0)
            out["miss"].append(0 if r["topk"][0][0] in {t for t, _ in c["topk"]} else 1)
    return out, None


def boot(docs, key, draws, seed, fn=None):
    """Token-weighted mean over docs, with a bootstrap CI over documents.  fn(doc) -> (sum, count)."""
    fn = fn or (lambda d: (sum(d[key]), len(d[key])))
    parts = [fn(d) for d in docs]
    tot = sum(s for s, _ in parts) / max(1, sum(n for _, n in parts))
    if len(parts) < 2:
        return tot, None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        pick = [parts[rng.randrange(len(parts))] for _ in parts]
        vals.append(sum(s for s, _ in pick) / max(1, sum(n for _, n in pick)))
    vals.sort()
    return tot, vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def paired_ratio(da, db, key, draws, seed):
    """mean_a / mean_b over the same documents, paired bootstrap."""
    pa = [(sum(d[key]), len(d[key])) for d in da]
    pb = [(sum(d[key]), len(d[key])) for d in db]
    r0 = (sum(s for s, _ in pa) / sum(n for _, n in pa)) / max(1e-30, sum(s for s, _ in pb) / sum(n for _, n in pb))
    if len(pa) < 2:
        return r0, None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        idx = [rng.randrange(len(pa)) for _ in pa]
        a = sum(pa[i][0] for i in idx) / sum(pa[i][1] for i in idx)
        b = sum(pb[i][0] for i in idx) / sum(pb[i][1] for i in idx)
        vals.append(a / max(1e-30, b))
    vals.sort()
    return r0, vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def fmt_ci(v, lo, hi, f="{:.3e}"):
    return f.format(v) + ("" if lo is None else f" [{f.format(lo)}, {f.format(hi)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ref", default=None, help="reference capture dir (BF16)")
    ap.add_argument("cands", nargs="+", help="name=capture_dir")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    ap.add_argument("--only", default=None, help="comma list of document names (self-checks)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.suite, "manifest.json")))
    docs = man["files"]
    if a.only:
        docs = [d for d in docs if d["name"] in set(a.only.split(","))]
    ids = {d["name"]: json.load(open(os.path.join(a.suite, d["tokens"]))) for d in docs}
    cls = {d["name"]: d["class"] for d in docs}
    cands = [c.split("=", 1) for c in a.cands]

    ref = {}
    if a.ref:
        for d in docs:
            p = os.path.join(a.ref, d["name"] + ".jsonl")
            if not os.path.exists(p):
                sys.exit(f"reference is missing {p}")
            ref[d["name"]] = load(p)

    per = {}   # cand -> name -> stats
    bad = []
    for cname, cdir in cands:
        per[cname] = {}
        for d in docs:
            p = os.path.join(cdir, d["name"] + ".jsonl")
            if not os.path.exists(p):
                bad.append(f"{cname}: missing {d['name']}")
                continue
            st, err = doc_stats(ids[d["name"]], ref.get(d["name"]) if a.ref else None, load(p))
            if err:
                bad.append(f"{cname}/{d['name']}: {err}")
                continue
            st["class"] = cls[d["name"]]
            per[cname][d["name"]] = st
    if bad:
        sys.exit("refusing to grade:\n  " + "\n  ".join(bad))

    groups = [("chat48", lambda n: n.startswith("chat-"))]
    for c in sorted({cls[n] for n in cls if n.startswith("chat-")}):
        groups.append((f"  {c}", lambda n, c=c: n.startswith("chat-") and cls[n] == c))
    groups += [("code", lambda n: n == "code"), ("story", lambda n: n == "story"), ("all", lambda n: True)]

    result = {"suite": man["version"], "ref": a.ref, "cands": dict(cands), "groups": {}}
    hdr = "| group | arm | tokens | NLL arm |" + (" NLL ref | dNLL | KL/token [95% CI] | KL median | top-1 | ref-top1 miss |"
                                                  if a.ref else "")
    print(f"# fmtcmp grade -- suite {man['version']} ({len(docs)} of {man['n_docs']} docs, "
          f"{sum(d['n_tokens'] for d in docs)} tokens)"
          + (f", reference {a.ref}" if a.ref else ", NO reference (absolute NLL only)"))
    print()
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for gname, sel in groups:
        for cname, _ in cands:
            ds = [per[cname][n] for n in per[cname] if sel(n)]
            if not ds:
                continue
            ntok = sum(len(d["nll_c"]) for d in ds)
            nll = boot(ds, "nll_c", a.draws, a.seed)
            row = {"tokens": ntok, "nll": nll}
            line = f"| {gname} | {cname} | {ntok} | {fmt_ci(*nll, f='{:.4f}')} |"
            if a.ref:
                nllr = boot(ds, "nll_r", a.draws, a.seed)
                dn = boot(ds, None, a.draws, a.seed,
                          fn=lambda d: (sum(c - r for c, r in zip(d["nll_c"], d["nll_r"])), len(d["nll_c"])))
                kl = boot(ds, "kl", a.draws, a.seed)
                allkl = sorted(k for d in ds for k in d["kl"])
                med = allkl[len(allkl) // 2]
                t1 = sum(sum(d["top1"]) for d in ds) / ntok
                miss = sum(sum(d["miss"]) for d in ds)
                row.update({"nll_ref": nllr, "dnll": dn, "kl": kl, "kl_median": med, "top1": t1, "miss": miss})
                line += (f" {nllr[0]:.4f} | {fmt_ci(*dn, f='{:+.4f}')} | {fmt_ci(*kl)} | {med:.2e} | {100 * t1:.2f}% "
                         f"| {miss} |")
            print(line)
            result["groups"].setdefault(gname.strip(), {})[cname] = row

    if a.ref and "story" in per[cands[0][0]]:
        print("\n## story by depth (KL/token, top-1)\n")
        print("| depth | " + " | ".join(c for c, _ in cands) + " |")
        print("|---|" + "---|" * len(cands))
        bounds = STORY_BUCKETS + [None]
        lo = 0
        for b in bounds:
            cells = []
            for cname, _ in cands:
                s = per[cname]["story"]
                sel = [i for i, p in enumerate(s["pos"]) if p >= lo and (b is None or p < b)]
                if not sel:
                    cells.append("-")
                    continue
                cells.append(f"{sum(s['kl'][i] for i in sel) / len(sel):.3e} / "
                             f"{100 * sum(s['top1'][i] for i in sel) / len(sel):.1f}%")
            print(f"| {lo}-{b if b else 'end'} | " + " | ".join(cells) + " |")
            lo = b or lo

    if a.ref and len(cands) > 1:
        print("\n## paired KL ratios on chat48 (row / column, document bootstrap)\n")
        names = [c for c, _ in cands]
        print("| | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for x in names:
            cells = []
            for y in names:
                if x == y:
                    cells.append("1")
                    continue
                dx = [per[x][n] for n in sorted(per[x]) if n.startswith("chat-")]
                dy = [per[y][n] for n in sorted(per[y]) if n.startswith("chat-")]
                r = paired_ratio(dx, dy, "kl", a.draws, a.seed)
                cells.append(fmt_ci(*r, f="{:.3f}"))
                result.setdefault("ratios", {})[f"{x}/{y}"] = r
            print(f"| {x} | " + " | ".join(cells) + " |")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()

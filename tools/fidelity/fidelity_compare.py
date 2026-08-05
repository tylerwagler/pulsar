#!/usr/bin/env python3
"""Compare two teacher-forced logprob captures (Plan 90 A2/A3).

Inputs are ds4-fidelity-jsonl-v1 files (from capture_vllm_prompt_logprobs.py
or pulsar_kldump_to_jsonl.py) for the SAME suite document.  Reports, per depth
bucket and overall:

  - truncated KL(ref || cand): both top-K distributions are renormalized over
    the union of their top-K token sets plus a shared tail bucket carrying the
    residual mass.  A lower bound on true KL; exact when both K cover ~all
    mass.  (Exact full-vocab KL between pulsar variants comes from the
    engine's own --kl-score path instead.)
  - top-1 agreement %
  - forced-token logprob delta (mean |Δ|), where both sides scored it

Alignment: token id arrays from both headers are aligned by searching a small
BOS/prefix offset (-4..+4) maximizing id agreement; positions that still
mismatch ids are skipped and counted.

Usage:
  fidelity_compare.py REF.jsonl CAND.jsonl [--buckets 2048,9216,38912]
  fidelity_compare.py --dir REF_DIR CAND_DIR   (all common *.jsonl files)
"""
import argparse
import json
import math
import os
import sys


def load(path):
    with open(path) as f:
        header = json.loads(f.readline())
        rows = {}
        for line in f:
            r = json.loads(line)
            rows[r["pos"]] = r
    return header, rows


def best_offset(a_tokens, b_tokens):
    """Offset o such that a[i] ~ b[i+o] agrees most often."""
    if not a_tokens or not b_tokens:
        return 0, 0.0
    best, best_rate = 0, -1.0
    n = min(len(a_tokens), len(b_tokens), 4096)
    for o in range(-4, 5):
        hit = tot = 0
        for i in range(n):
            j = i + o
            if 0 <= j < len(b_tokens):
                tot += 1
                hit += a_tokens[i] == b_tokens[j]
        rate = hit / tot if tot else 0.0
        if rate > best_rate:
            best, best_rate = o, rate
    return best, best_rate


def trunc_kl(p_topk, q_topk):
    """KL(p || q) over union-of-supports + tail bucket, renormalized."""
    p = {t: lp for t, lp in p_topk}
    q = {t: lp for t, lp in q_topk}
    support = set(p) | set(q)
    p_mass = sum(math.exp(lp) for lp in p.values())
    q_mass = sum(math.exp(lp) for lp in q.values())
    p_tail = max(1e-12, 1.0 - p_mass)
    q_tail = max(1e-12, 1.0 - q_mass)
    # Unseen-in-one-side tokens are charged to that side's tail, spread evenly.
    p_missing = [t for t in support if t not in p]
    q_missing = [t for t in support if t not in q]
    p_fill = p_tail / max(1, len(p_missing) + 1)
    q_fill = q_tail / max(1, len(q_missing) + 1)
    kl = 0.0
    tot_p = 0.0
    for t in support:
        pp = math.exp(p[t]) if t in p else p_fill
        qq = math.exp(q[t]) if t in q else q_fill
        kl += pp * math.log(pp / qq)
        tot_p += pp
    # shared tail bucket for both distributions' remaining mass
    pp = max(1e-12, 1.0 - tot_p)
    qq = max(1e-12, q_fill)
    kl += pp * math.log(pp / qq)
    return kl


def bucket_of(pos, bounds):
    for i, b in enumerate(bounds):
        if pos < b:
            return i
    return len(bounds)


def bucket_name(i, bounds):
    lo = 0 if i == 0 else bounds[i - 1]
    hi = bounds[i] if i < len(bounds) else None
    return f"{lo}-{hi}" if hi else f"{lo}+"


def compare_file(ref_path, cand_path, bounds):
    ref_h, ref_rows = load(ref_path)
    cand_h, cand_rows = load(cand_path)
    off, rate = best_offset(ref_h.get("tokens") or [], cand_h.get("tokens") or [])
    stats = {}
    skipped = 0
    for pos, r in ref_rows.items():
        c = cand_rows.get(pos + off)
        if not c:
            continue
        if r.get("token") is not None and c.get("token") is not None \
           and r["token"] != c["token"]:
            skipped += 1
            continue
        b = bucket_of(pos, bounds)
        s = stats.setdefault(b, {"n": 0, "kl": [], "top1": 0, "dlp": []})
        s["n"] += 1
        s["kl"].append(trunc_kl(r["topk"], c["topk"]))
        if r["topk"] and c["topk"] and r["topk"][0][0] == c["topk"][0][0]:
            s["top1"] += 1
        if r.get("logprob") is not None and c.get("logprob") is not None:
            s["dlp"].append(abs(r["logprob"] - c["logprob"]))
    return {"offset": off, "align_rate": rate, "skipped": skipped,
            "buckets": stats}


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report(name, res, bounds):
    print(f"\n## {name}  (align offset {res['offset']}, "
          f"rate {res['align_rate']:.3f}, {res['skipped']} pos skipped)")
    print("| depth | n | KL med | KL p95 | top-1 % | |Δlogprob| |")
    print("|---|---|---|---|---|---|")
    total = {"n": 0, "kl": [], "top1": 0, "dlp": []}
    for b in sorted(res["buckets"]):
        s = res["buckets"][b]
        total["n"] += s["n"]
        total["kl"] += s["kl"]
        total["top1"] += s["top1"]
        total["dlp"] += s["dlp"]
        print(f"| {bucket_name(b, bounds)} | {s['n']} "
              f"| {pct(s['kl'], 0.5):.3e} | {pct(s['kl'], 0.95):.3e} "
              f"| {100.0 * s['top1'] / max(1, s['n']):.2f} "
              f"| {sum(s['dlp']) / max(1, len(s['dlp'])):.3e} |")
    s = total
    print(f"| **all** | {s['n']} | {pct(s['kl'], 0.5):.3e} "
          f"| {pct(s['kl'], 0.95):.3e} "
          f"| {100.0 * s['top1'] / max(1, s['n']):.2f} "
          f"| {sum(s['dlp']) / max(1, len(s['dlp'])):.3e} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--dir", action="store_true",
                    help="ref/cand are directories; compare common *.jsonl")
    # Plan 90 depth points are 2k/9k/38k/64k: the 64k bound must be its own
    # bucket boundary or everything past 38912 collapses into one bin — and
    # depth is exactly where the competitor gap grows (2.4x@2k -> 3.3x@64k),
    # so the deepest bucket is the most diagnostic one we have.
    ap.add_argument("--buckets", default="2048,9216,38912,65536")
    a = ap.parse_args()
    bounds = [int(x) for x in a.buckets.split(",")]

    if a.dir:
        names = sorted(set(os.listdir(a.ref)) & set(os.listdir(a.cand)))
        names = [n for n in names if n.endswith(".jsonl")]
        if not names:
            sys.exit("no common .jsonl files")
        for n in names:
            report(n, compare_file(os.path.join(a.ref, n),
                                   os.path.join(a.cand, n), bounds), bounds)
    else:
        report(os.path.basename(a.cand),
               compare_file(a.ref, a.cand, bounds), bounds)


if __name__ == "__main__":
    main()

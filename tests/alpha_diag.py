#!/usr/bin/env python3
"""Prose-vs-structured spec-acceptance decomposition via /metrics deltas.

Runs 2 domains x 3 temperatures x 3 generations of 256 tokens against a live
server and prints alpha, accepted/round, tokens/round and the per-position
acceptance waterfall + conditionals per cell, from /metrics counter deltas.

This is the harness behind L075 (2026-08-19): it separated "sampling-rule
problem" from "model-level disagreement" in one 6-minute run -- prose alpha
was temperature-INDEPENDENT (greedy as bad as sampled), no positional cliff,
structured at/above spec, which convicts drafter-vs-quantized-main
distributional distance and selects distillation as the fix. Rerun it after
any change that might move acceptance (numerics campaigns, drafter work):
    python3 tests/alpha_diag.py          # server on 127.0.0.1:8000
"""
import json, urllib.request, sys

BASE = "http://127.0.0.1:8000"

PROSE = ("Continue this essay in flowing prose, no lists or headings.\n\n"
         "The history of long-distance navigation is a history of borrowed "
         "certainties. Sailors trusted stars they could not reach, clocks "
         "they could not repair at sea, and charts drawn by men who had "
         "died before they were born. ")
STRUCT = ("Output a JSON array of 30 objects, each with fields "
          "\"name\", \"age\", \"city\", \"occupation\" for fictional people. "
          "JSON only, no prose.\n")


def metrics():
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            if "spec_decode" not in line:
                continue
            key, _, val = line.rpartition(" ")
            out[key] = float(val)
    return out


def gen(prompt, temp, n):
    body = json.dumps({"model": "deepseek-v4-flash", "prompt": prompt,
                       "max_tokens": n, "temperature": temp, "seed": 7}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read())


def key1(m, frag):
    for k in m:
        if frag in k and "per_pos" not in k:
            return k
    raise SystemExit("missing " + frag)


for domain, prompt in (("prose", PROSE), ("struct", STRUCT)):
    for temp in (0.0, 0.6, 1.0):
        m0 = metrics()
        for _ in range(3):
            gen(prompt, temp, 256)
        m1 = metrics()
        d = {k: m1[k] - m0.get(k, 0.0) for k in m1}
        drafted = d[key1(m1, "num_draft_tokens_total")]
        accepted = d[key1(m1, "num_accepted_tokens_total")]
        rounds = d[key1(m1, "num_drafts_total")]
        spec_gen = d.get("pulsar:spec_decode_gen_tokens_total", 0.0)
        alpha = accepted / drafted if drafted else 0.0
        pos = []
        for i in range(16):
            for k in d:
                if 'position="%d"' % i in k:
                    pos.append(d[k] / rounds if rounds else 0.0)
        cond = []
        prev = 1.0
        for p in pos:
            if p <= 0:
                break
            cond.append(p / prev if prev > 0 else 0.0)
            prev = p
        wf = " ".join("%.2f" % p for p in pos[:8])
        cd = " ".join("%.2f" % c for c in cond[:8])
        print("%s temp=%.1f: alpha=%.3f acc/round=%.2f tok/round=%.2f "
              "rounds=%d drafted/round=%.2f" % (
                  domain, temp, alpha,
                  accepted / rounds if rounds else 0.0,
                  spec_gen / rounds if rounds else 0.0,
                  rounds,
                  drafted / rounds if rounds else 0.0))
        print("  waterfall  = " + wf)
        print("  conditional= " + cd)
        sys.stdout.flush()
print("DIAG COMPLETE")

#!/usr/bin/env python3
"""Context-coherence probe: can the model actually use what is in its window?

Plants three distinctive facts at the BEGINNING, MIDDLE and END of a long
context and asks for them back, at real depths rather than a token count
chosen to fit in a demo.

Two design choices are deliberate, and both are corrections of the version
this replaces:

  * FREE SCHEMA -- no constrained decoding, no JSON grammar, no const-pinned
    fields.  Under a grammar mask the model is forced into the shape the
    scorer expects, so a wrong answer can still look structurally perfect and
    the probe cannot fail in the way that matters.  A masked probe is
    unfalsifiable; this one asks in prose and reads the prose back.

  * A CHECKSUM question.  Per-fact recall can be passed by luck or by a model
    that latches one fact and pattern-matches the rest.  The checksum needs
    all three values at once (their sum), so it cannot be answered from a
    partial retrieval.  It is scored separately and is the honest headline.

The planted facts are deterministic in --seed, so a run is reproducible, and
they are lexically unlike the filler so a hit cannot come from the padding.

    ./tests/context_coherence_probe.py --port 8000 --depths 2048,8192,32768

Exit status: 0 all depths met --min-score, 1 a depth fell below it,
2 harness/transport error.
"""

import argparse
import http.client
import json
import os
import random
import re
import sys

DEFAULT_FILLER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "speed-bench", "promessi_sposi.txt",
)

# Distinctive, unlikely to collide with filler prose or with each other.
CODEWORDS = ["harbor-lantern", "copper-mantis", "vellum-signal", "quartz-ferry",
             "obsidian-kite", "tallow-compass", "cinder-almanac", "gossamer-anvil"]


def plant_facts(rng):
    """Three (label, codeword, value) triples. Values are 4-digit so the sum
    is unambiguous and cannot be confused with a position or a depth."""
    words = rng.sample(CODEWORDS, 3)
    return [
        {"position": "beginning", "codeword": words[0], "value": rng.randint(1000, 9999)},
        {"position": "middle", "codeword": words[1], "value": rng.randint(1000, 9999)},
        {"position": "end", "codeword": words[2], "value": rng.randint(1000, 9999)},
    ]


def fact_sentence(fact):
    return ("Record for later: the %s calibration code is %s and its reading is %d."
            % (fact["position"], fact["codeword"], fact["value"]))


def build_context(filler_words, target_words, facts):
    """Interleave the three facts into filler at begin/middle/end."""
    body = filler_words[:target_words]
    if len(body) < 60:
        raise RuntimeError("filler too short for the requested depth")
    mid = len(body) // 2
    # Insert from the back so earlier indices stay valid.
    out = list(body)
    out.insert(len(out) - 20, fact_sentence(facts[2]))
    out.insert(mid, fact_sentence(facts[1]))
    out.insert(20, fact_sentence(facts[0]))
    return " ".join(out)


QUESTION = (
    "\n\n---\n"
    "Three calibration records were embedded in the text above. "
    "For each one, state its codeword and its numeric reading. "
    "Then state the sum of the three readings. "
    "Answer in plain prose; do not restate the passage."
)


def score_answer(answer, facts):
    """Substring/word-boundary scoring. No structure is required of the model."""
    low = answer.lower()
    per_fact = []
    for f in facts:
        cw_hit = f["codeword"].lower() in low
        # \b around a 4-digit number: avoids crediting 1234 inside 51234.
        val_hit = re.search(r"(?<!\d)%d(?!\d)" % f["value"], answer) is not None
        # Both together is the only honest "retrieved": a codeword with the
        # wrong number, or a number with no codeword, is not a retrieval.
        per_fact.append({
            "position": f["position"],
            "codeword": f["codeword"],
            "value": f["value"],
            "codeword_found": cw_hit,
            "value_found": val_hit,
            "retrieved": bool(cw_hit and val_hit),
        })
    total = sum(f["value"] for f in facts)
    checksum_ok = re.search(r"(?<!\d)%d(?!\d)" % total, answer) is not None
    n_ok = sum(1 for f in per_fact if f["retrieved"])
    return {
        "facts": per_fact,
        "n_retrieved": n_ok,
        "expected_sum": total,
        "checksum_ok": checksum_ok,
        # Headline: all three retrieved AND the sum right. The sum alone can be
        # arithmetic luck; the facts alone can be partial latching.
        "score": round(n_ok / 3.0, 4),
        "strict_pass": bool(n_ok == 3 and checksum_ok),
    }


def run_depth(args, filler_words, depth_words, rng):
    facts = plant_facts(rng)
    context = build_context(filler_words, depth_words, facts)
    prompt = context + QUESTION

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text = stream_text(args, payload)
    scored = score_answer(text, facts)
    scored["depth_words"] = depth_words
    scored["prompt_chars"] = len(prompt)
    scored["answer_chars"] = len(text)
    scored["answer"] = text if args.keep_answers else None
    return scored


def stream_text(args, payload):
    """Concatenate VISIBLE content deltas only.

    reasoning_content is deliberately excluded: that is where the model thinks
    about the answer, not where it gives one.  Scoring it would let a model
    pass by musing about a codeword it never actually commits to."""
    conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read(2048).decode("utf-8", "replace")
        conn.close()
        raise RuntimeError("HTTP %d: %s" % (resp.status, detail))
    parts = []
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.strip()
        if not line or line.startswith(b":"):
            continue  # blank or ": ping" heartbeat comment
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
    conn.close()
    return "".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="ds4flash")
    p.add_argument("--depths", default="2048,8192,32768",
                   help="approximate context depths in WORDS (not tokens), comma separated")
    p.add_argument("--filler", default=DEFAULT_FILLER)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--min-score", type=float, default=1.0,
                   help="minimum per-depth fact score to pass (default 1.0 = all three)")
    p.add_argument("--allow-checksum-miss", dest="require_checksum",
                   action="store_false", default=True,
                   help="score facts only; by default a wrong sum fails the depth")
    p.add_argument("--keep-answers", action="store_true",
                   help="include raw model answers in the JSON (verbose)")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    try:
        with open(args.filler, encoding="utf-8", errors="replace") as f:
            filler_words = f.read().split()
    except OSError as e:
        print("cannot read filler %s: %s" % (args.filler, e), file=sys.stderr)
        return 2

    try:
        depths = [int(d) for d in args.depths.split(",") if d.strip()]
    except ValueError:
        print("bad --depths", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    results = []
    failed = False
    for depth in depths:
        if depth > len(filler_words):
            print("depth %d exceeds filler length %d words -- skipping"
                  % (depth, len(filler_words)), file=sys.stderr)
            continue
        try:
            r = run_depth(args, filler_words, depth, rng)
        except (OSError, RuntimeError) as e:
            print("harness error at depth %d: %s" % (depth, e), file=sys.stderr)
            return 2
        results.append(r)
        ok = r["score"] >= args.min_score and (r["checksum_ok"] or not args.require_checksum)
        if not ok:
            failed = True
        print("  depth %-7d retrieved=%d/3 checksum=%s -> %s"
              % (depth, r["n_retrieved"], "ok" if r["checksum_ok"] else "MISS",
                 "pass" if ok else "FAIL"), file=sys.stderr)

    record = {
        "schema": "pulsar.context_coherence_probe.v1",
        "host": "%s:%d" % (args.host, args.port),
        "model": args.model,
        "seed": args.seed,
        "min_score": args.min_score,
        "require_checksum": bool(args.require_checksum),
        "depths": results,
        "verdict": "fail" if failed else "pass",
    }
    text = json.dumps(record, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

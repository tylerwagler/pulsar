#!/usr/bin/env python3
"""Prefill and decode speed of one OpenAI-compatible completions endpoint (fmtcmp, L296).  Stdlib only.

  speed.py --endpoint http://127.0.0.1:8300 --suite /mnt/models/fmtcmp/suite-v1 --out speed.json

- prefill: a 4,096- and a 32,768-token prompt (the suite's story ids, then code ids, cut), max_tokens 1, streamed;
  prefill tok/s = prompt tokens / time to the first token.  One warm-up request first, each length measured
  --reps times, and the cache-busting first token differs per request (the capture flags disable prefix caching
  already; the distinct first token keeps the numbers honest on a server that caches).
- decode: --gen tokens with ignore_eos at temperature 0 after a 512-token prompt, streamed;
  decode tok/s = (tokens - 1) / (last token time - first token time).  1 stream, then --streams concurrent
  streams (aggregate and per stream).
The server runs whatever speculation its flags say; report the flags beside the numbers.
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request


def stream(endpoint, model, prompt, max_tokens, ignore_eos=False):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": ignore_eos, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, usage = [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                if c.get("text"):
                    times.append(time.perf_counter())
    return t0, times, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="4096,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--streams", type=int, default=4)
    a = ap.parse_args()

    model = json.loads(urllib.request.urlopen(a.endpoint.rstrip("/") + "/v1/models").read())["data"][0]["id"]
    man = json.load(open(os.path.join(a.suite, "manifest.json")))
    by = {f["name"]: f for f in man["files"]}
    pool = []
    for n in ("story", "code"):
        pool += json.load(open(os.path.join(a.suite, by[n]["tokens"])))
    res = {"endpoint": a.endpoint, "model": model, "prefill": {}, "decode": {}}

    def prompt(n, salt):
        return [pool[(salt * 7919) % 1000 + 1]] + pool[: n - 1]   # a distinct first token per request

    stream(a.endpoint, model, prompt(1024, 0), 8)                   # warm-up
    salt = 1
    for n in [int(x) for x in a.lengths.split(",")]:
        rates = []
        for _ in range(a.reps):
            t0, times, usage = stream(a.endpoint, model, prompt(n, salt), 1)
            salt += 1
            rates.append(n / (times[0] - t0))
        res["prefill"][n] = {"tok_s": statistics.median(rates), "runs": rates}
        print(f"prefill {n}: {statistics.median(rates):.0f} tok/s  {['%.0f' % r for r in rates]}", flush=True)

    def one(out, i):
        t0, times, usage = stream(a.endpoint, model, prompt(512, 100 + i), a.gen, ignore_eos=True)
        out[i] = (times, (usage or {}).get("completion_tokens") or len(times))

    for s in (1, a.streams):
        outs = [None] * s
        th = [threading.Thread(target=one, args=(outs, i)) for i in range(s)]
        [t.start() for t in th]
        [t.join() for t in th]
        # tokens from usage (a speculative server streams several tokens a chunk); the first chunk is excluded
        per = [(n - 1) / (t[-1] - t[0]) for t, n in outs]
        span = max(t[-1] for t, _ in outs) - min(t[0] for t, _ in outs)
        agg = sum(n - 1 for _, n in outs) / span
        res["decode"][s] = {"per_stream_tok_s": per, "aggregate_tok_s": agg,
                            "tokens": [n for _, n in outs], "chunks": [len(t) for t, _ in outs]}
        print(f"decode {s} stream(s): aggregate {agg:.1f} tok/s, per stream {['%.1f' % p for p in per]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

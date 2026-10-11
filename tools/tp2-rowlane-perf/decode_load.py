#!/usr/bin/env python3
"""Decode load for the row-lane A/B (stdlib only; the pair is air-gapped).

N concurrent greedy streaming completions against a running pulsar-server, each with its own fixed prompt
(~PP tokens, identical across builds), up to TG generated tokens (a raw continuation; the server has no ignore_eos, gen_tokens says what ran).  Prints one JSON line:

  decode_tps      aggregate decode tokens/s over the window where ALL streams decode
                  (last first-token .. first stream end)
  per_stream_tps  median over streams of (tokens - 1) / (end - first token)
  step_ms_med/p90 the step wall: chunks of one stream arriving within 2 ms of each other are one engine step
                  (a speculative step can stream several tokens), so this is the gap between steps
  tok_per_step    tokens per step (1.0 without drafts)

usage: decode_load.py PORT N PP TG
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

port, n, pp, tg = (int(a) for a in sys.argv[1:5])
WORDS = ("the pair moves rows between two sparks over rdma while each rank owns half of every expert and "
         "the proxy thread posts sends and reaps completions so the stream never waits on the host ").split()


def prompt(i):
    # Distinct per stream (no shared prefix), the same bytes for every build; ~1.3 tokens per word.
    words = [WORDS[(i * 7 + k * 13) % len(WORDS)] for k in range(int(pp / 1.3))]
    return f"Stream {i}. Continue this text in your own words:\n" + " ".join(words)


res = [None] * n


def run(i):
    body = {"model": "m", "prompt": prompt(i), "max_tokens": tg, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    stamps, ntok = [], None
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode(errors="replace").strip()   # a token may stream part of a UTF-8 sequence
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            ntok = d["usage"].get("completion_tokens")
        ch = d.get("choices") or []
        if ch and ch[0].get("text"):
            stamps.append(time.time())
    res[i] = (t0, stamps, ntok if ntok else len(stamps))


start = time.time()
threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
bad = [i for i, r in enumerate(res) if r is None or len(r[1]) < 2]
if bad:
    sys.exit(f"decode_load: stream(s) {bad} failed or produced < 2 chunks -- no measurement")

firsts = [r[1][0] for r in res]
ends = [r[1][-1] for r in res]
w0, w1 = max(firsts), min(ends)
in_window = sum(sum(1 for s in r[1] if w0 < s <= w1) for r in res)
chunks_per_tok = sum(len(r[1]) for r in res) / max(1, sum(r[2] for r in res))
decode_tps = in_window / chunks_per_tok / (w1 - w0) if w1 > w0 else 0.0
per_stream = [(r[2] - 1) / (r[1][-1] - r[1][0]) for r in res if len(r[1]) > 1]
gaps, steps, toks = [], 0, 0
for r in res:
    st = r[1]
    burst_starts = [st[0]] + [b for a, b in zip(st, st[1:]) if b - a > 0.002]
    gaps += [b - a for a, b in zip(burst_starts, burst_starts[1:])]
    steps += len(burst_starts)
    toks += r[2]
gaps.sort()
print(json.dumps({
    "streams": n, "pp": pp, "tg": tg, "gen_tokens": sum(r[2] for r in res),
    "decode_tps": round(decode_tps, 2),
    "per_stream_tps": round(statistics.median(per_stream), 2) if per_stream else 0,
    "step_ms_med": round(1e3 * statistics.median(gaps), 2) if gaps else 0,
    "step_ms_p90": round(1e3 * gaps[int(0.9 * (len(gaps) - 1))], 2) if gaps else 0,
    "tok_per_step": round(toks / max(1, steps), 2),
    "ttft_max_s": round(max(f - r[0] for f, r in zip(firsts, res)), 2),
    "wall_s": round(time.time() - start, 1),
}))

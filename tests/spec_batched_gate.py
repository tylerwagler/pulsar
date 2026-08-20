#!/usr/bin/env python3
"""plan-34 inc-6: the spec-batched lane gate (L076).

Scripted oracle drive through lane 3 -- the checks the W4 commit message
requires before the lane lands on dev:

  1. ENGAGEMENT  lane "spec-batched" is observed while two spec-capable
                 decodes overlap, and the spec counters ADVANCE during the
                 overlap (drafted > 0, 0 < accepted <= drafted).
  2. CONTRACT    the engine's spec_decode_gen_tokens delta equals the tokens
                 the server actually emitted (counter/emission agreement).
  3. ISOLATION   stream A (greedy, temp 0) produces BYTE-IDENTICAL text when
                 co-scheduled with partner B and with a DIFFERENT partner C.
                 A's rounds may interleave with either partner's arbitrarily;
                 if any of A's bits depend on who shares the forward, the
                 banks are not isolated. A is kept SHORTER than its partners
                 so its whole generation runs inside lane 3 (never the solo
                 lane tail, which is a different arithmetic and would fail
                 this check spuriously -- L043).
  4. HEALTH      no degenerate repetition in any stream; finishes are sane.

Cross-LANE byte equality is deliberately NOT asserted (L043): the solo run
is captured and printed as information only.

Usage:
  spec_batched_gate.py --server ./pulsar-server --model /path/model.gguf
                       [--port 8098]
The gate owns the server lifecycle (launches on --port, kills on exit), so
run it on a box where that port is free and the GPU can take a model load.
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

A_TOKENS, PARTNER_TOKENS = 48, 160
PROMPT_A = "The chemistry of tidal pools rewards patient observation, because"
PROMPT_B = "def topological_sort(graph):\n    \"\"\"Return the vertices of a DAG in topological order.\"\"\"\n"
PROMPT_C = "The economic history of the Hanseatic League illustrates how"


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
        return r.read().decode()


def counters(txt):
    out = {}
    for line in txt.splitlines():
        for k in ("vllm:spec_decode_num_draft_tokens_total",
                  "vllm:spec_decode_num_accepted_tokens_total",
                  "pulsar:spec_decode_gen_tokens_total"):
            if line.startswith(k):
                out[k.split(":")[1]] = int(float(line.rsplit(None, 1)[1]))
    return out


def active_lane(txt):
    for line in txt.splitlines():
        if line.startswith("pulsar:decode_lane{") and line.rstrip().endswith(" 1"):
            return line.split('lane="')[1].split('"')[0]
    return "?"


def complete(base, prompt, max_tokens, slot):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read().decode())
    slot["text"] = j["choices"][0]["text"]
    slot["n"] = j.get("usage", {}).get("completion_tokens", -1)
    slot["finish"] = j["choices"][0].get("finish_reason")


def run_pair(base, prompt_partner):
    a, p = {}, {}
    lanes, stop = {}, [False]

    def poll():
        while not stop[0]:
            try:
                lane = active_lane(metrics(base))
                lanes[lane] = lanes.get(lane, 0) + 1
            except Exception:
                pass
            time.sleep(0.2)

    c0 = counters(metrics(base))
    poller = threading.Thread(target=poll)
    poller.start()
    ta = threading.Thread(target=complete, args=(base, PROMPT_A, A_TOKENS, a))
    tp = threading.Thread(target=complete, args=(base, prompt_partner, PARTNER_TOKENS, p))
    ta.start(); tp.start(); ta.join(); tp.join()
    stop[0] = True; poller.join()
    c1 = counters(metrics(base))
    delta = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    return a, p, lanes, delta


def degenerate(text):
    words = text.split()
    if len(words) < 12:
        return False
    shingles = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
    return len(set(shingles)) < len(shingles) * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8098)
    args = ap.parse_args()
    base = "http://127.0.0.1:%d" % args.port

    srv = subprocess.Popen(
        [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=open("spec_batched_gate.server.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    ok = False
    try:
        for _ in range(120):
            if srv.poll() is not None:
                print("spec-batched gate: SERVER DIED during load"); return 2
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2); ok = True; break
            except Exception:
                time.sleep(5)
        if not ok:
            print("spec-batched gate: server never became healthy"); return 2

        failures = []
        # warm the prefixes so both measured pairs decode-overlap immediately
        run_pair(base, PROMPT_B)

        a1, p1, lanes1, d1 = run_pair(base, PROMPT_B)
        a2, p2, lanes2, d2 = run_pair(base, PROMPT_C)

        for tag, lanes, d in (("A+B", lanes1, d1), ("A+C", lanes2, d2)):
            if lanes.get("spec-batched", 0) < 3:
                failures.append("%s: lane spec-batched barely/never engaged (%s)" % (tag, lanes))
            drafted = d.get("spec_decode_num_draft_tokens_total", 0)
            accepted = d.get("spec_decode_num_accepted_tokens_total", 0)
            if drafted <= 0:
                failures.append("%s: no draft tokens counted during concurrency" % tag)
            if not (0 < accepted <= drafted):
                failures.append("%s: accepted/drafted contract violated (%d/%d)" % (tag, accepted, drafted))

        emitted1 = a1.get("n", 0) + p1.get("n", 0)
        gen1 = d1.get("spec_decode_gen_tokens_total", 0)
        if gen1 != emitted1:
            failures.append("A+B: spec_gen_tokens %d != emitted %d" % (gen1, emitted1))

        if a1.get("text") != a2.get("text"):
            i = next((k for k, (x, y) in enumerate(zip(a1.get("text", ""), a2.get("text", ""))) if x != y),
                     min(len(a1.get("text", "")), len(a2.get("text", ""))))
            failures.append("BANK ISOLATION: A's greedy text depends on its partner (first divergence at char %d)" % i)

        for tag, s in (("A1", a1), ("A2", a2), ("B", p1), ("C", p2)):
            if degenerate(s.get("text", "")):
                failures.append("%s: degenerate repetition" % tag)
            if s.get("finish") not in ("length", "stop"):
                failures.append("%s: bad finish_reason %r" % (tag, s.get("finish")))

        solo = {}
        complete(base, PROMPT_A, A_TOKENS, solo)
        print("informational: solo-vs-batched A %s (cross-lane equality NOT asserted, L043)"
              % ("IDENTICAL" if solo.get("text") == a1.get("text") else "differs"))
        print("lanes A+B=%s A+C=%s | counters A+B=%s" % (lanes1, lanes2, d1))

        if failures:
            for f in failures:
                print("FAIL:", f)
            print("spec-batched lane gate: FAIL (%d)" % len(failures))
            return 1
        print("spec-batched lane gate: PASS (engagement, counter contract, "
              "bank isolation across partners, health)")
        return 0
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

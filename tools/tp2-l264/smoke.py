#!/usr/bin/env python3
"""L264 TP2 served smoke: the KV state model on a tensor-parallel pair.

Run by tools/tp2-l264/run.sh on the head (rank 0); stdlib only.  Each PHASE is
one invocation; state between phases (the requests and the replies they got)
lives in STATE (a JSON file), so a phase after a restart can resend the same
request and require the same reply.

Every request is greedy (temperature 0) and non-streaming, so a reply is a
function of the KV it ran on: a resume from a grid checkpoint, a restore from
the disk segment chain and a cold prefill must all give the same text (the
chunk-neutrality law the single-Spark gates prove, here through the pair).

  phase1  A cold long turn; B the next turn, echoed WITHOUT the reasoning (the
          L264 client shape) -- must resume from a grid checkpoint, not 0;
          C B resent exactly -- same reply, cache hit; D turn 3 -- reply kept
  phase2  (after a pair restart) E turn 3 resent -- from the DISK chain on both
          ranks: cached > 0 and the same reply as D; F four new conversations
          at once -- every one 200 (mirrored stores under concurrency)
  miss    (after a worker copy was corrupted) G turn 3 resent -- still the same
          reply (the miss falls back to prefill on every rank)
  guard   (relaunched with a small KV budget) H six conversations of ~18k
          tokens two at a time, then each resent -- same replies (banks
          spilled to the segment chain and restored)

Exit status: 0 = every leg passed; 1 = a leg failed (named on stdout).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("SMOKE_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("SMOKE_MODEL", "deepseek-v4-flash")
TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "1800"))

failures = []


def leg(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def filler(seed, n_lines):
    """Deterministic prose, distinct per seed: ~11 tokens a line."""
    words = ["river", "stone", "lantern", "harbor", "copper", "meadow", "signal", "orchard",
             "thunder", "ledger", "violet", "compass", "granite", "willow", "beacon", "saddle"]
    out = []
    for i in range(n_lines):
        w = [words[(seed * 7 + i * 3 + k * 5) % len(words)] for k in range(6)]
        out.append(f"Line {seed}.{i}: the {w[0]} and the {w[1]} met by the {w[2]} near {w[3]}, {w[4]} {w[5]}.")
    return "\n".join(out)


def chat(messages, max_tokens=768):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            status, data = r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"status": e.code, "text": "", "error": e.read().decode(errors="replace")[:400],
                "secs": time.time() - t0}
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    # The whole output: thinking mode puts most of a short budget in the
    # reasoning, and a greedy reply is compared on everything it generated.
    out = (msg.get("reasoning_content") or "") + "\n--\n" + (msg.get("content") or "")
    return {"status": status, "text": out if out.strip("\n-") else "",
            "answer": msg.get("content") or "",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "secs": time.time() - t0}


def show(tag, r):
    print(f"    {tag}: http {r['status']} prompt={r.get('prompt_tokens')} cached={r.get('cached')} "
          f"{r['secs']:.1f}s answer={r.get('answer', '')[:60]!r}", flush=True)


def conversation(seed, n_lines):
    return [{"role": "system", "content": "You are a careful assistant. Answer briefly."},
            {"role": "user", "content": filler(seed, n_lines) +
             f"\n\nQuestion {seed}: which word appears in line {seed}.3? Answer in one sentence."}]


def phase1(st):
    msgs = conversation(1, 560)                     # ~6.5k tokens: several grid points, two prefill chunks
    a = chat(msgs); show("A cold", a)
    leg("A cold turn answers", a["status"] == 200 and a["text"] != "")
    msgs += [{"role": "assistant", "content": a["answer"]},   # the client echoes the visible reply only
             {"role": "user", "content": "Now name the word in line 1.7, in one sentence."}]
    b = chat(msgs); show("B next turn", b)
    leg("B resumes from a grid checkpoint (cached >= 90% of A's prompt)",
        b["status"] == 200 and b["cached"] >= 0.9 * a.get("prompt_tokens", 1),
        f"cached {b.get('cached')} of A's {a.get('prompt_tokens')}")
    c = chat(msgs); show("C exact resend", c)
    leg("C exact resend: same reply", c["status"] == 200 and c["text"] == b["text"])
    leg("C exact resend: cache hit", c.get("cached", 0) >= 0.9 * b.get("prompt_tokens", 1))
    msgs += [{"role": "assistant", "content": b["answer"]},
             {"role": "user", "content": "And line 1.11? One sentence."}]
    d = chat(msgs); show("D turn 3", d)
    leg("D turn 3 answers", d["status"] == 200 and d["text"] != "")
    st["turn3"] = msgs
    st["turn3_reply"] = d["text"]
    st["turn3_prompt"] = d.get("prompt_tokens", 0)


def phase2(st):
    e = chat(st["turn3"]); show("E turn 3 after restart", e)
    leg("E restores from the disk segment chain (cached > 50% of the prompt)",
        e["status"] == 200 and e["cached"] > 0.5 * st["turn3_prompt"],
        f"cached {e.get('cached')} of {st['turn3_prompt']}")
    leg("E same reply as before the restart", e["text"] == st["turn3_reply"],
        "" if e["text"] == st["turn3_reply"] else f"before {st['turn3_reply'][:60]!r}")
    with ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(lambda s: chat(conversation(s, 300)), [11, 12, 13, 14]))
    for i, r in enumerate(rs):
        show(f"F conversation {11 + i}", r)
    leg("F four concurrent conversations all answer", all(r["status"] == 200 and r["text"] for r in rs))


def phase_miss(st):
    g = chat(st["turn3"]); show("G turn 3 with a corrupt worker copy", g)
    leg("G answers through the miss", g["status"] == 200 and g["text"] != "")
    leg("G same reply as before", g["text"] == st["turn3_reply"])


def phase_guard(st):
    # The guard checks from the BATCHED decode step (two or more banks decoding
    # at once), so the conversations run two at a time: the banks idle while a
    # pair decodes are its spill victims, and the resends restore them.
    seeds = list(range(21, 27))
    with ThreadPoolExecutor(2) as ex:
        first = dict(zip(seeds, ex.map(lambda s: chat(conversation(s, 850)), seeds)))
    for s in seeds:
        show(f"H conversation {s}", first[s])
    leg("H every conversation answers", all(r["status"] == 200 and r["text"] for r in first.values()))
    with ThreadPoolExecutor(2) as ex:
        again = dict(zip(seeds, ex.map(lambda s: chat(conversation(s, 850)), seeds)))
    for s in seeds:
        r = again[s]
        show(f"H resend {s}", r)
        # The answer must match.  The whole output is reported, not graded: these
        # decode two banks at a time, and batched decode is not bitwise
        # deterministic by design (a speed tradeoff) -- see README "Reasoning
        # drift under concurrency".
        leg(f"H resend {s}: same answer after the guard's spill/restore", r["status"] == 200 and
            r["answer"] == first[s]["answer"])
        if r["text"] != first[s]["text"]:
            print(f"  [WARN] H resend {s}: the reasoning differs from the first run", flush=True)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("phase1", "phase2", "miss", "guard"):
        print(__doc__)
        return 2
    phase, path = sys.argv[1], sys.argv[2]
    st = json.load(open(path)) if os.path.exists(path) else {}
    print(f"== smoke {phase} against {BASE}", flush=True)
    {"phase1": phase1, "phase2": phase2, "miss": phase_miss, "guard": phase_guard}[phase](st)
    json.dump(st, open(path, "w"))
    print(f"== smoke {phase}: {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

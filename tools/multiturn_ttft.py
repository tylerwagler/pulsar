"""Multi-turn warm-state benchmark: N conversations x T turns, round-robin.

Turn 1 is cold prefill.  Turns 2..T replay the growing history plus a new
question — with a warm bank the prefix is resident and TTFT is small; with
fewer banks than conversations, older banks get evicted and every turn pays
a cold re-prefill.  Reports per-turn TTFT percentiles and total wall.
"""
import json, sys, time, urllib.request

BASE = "http://sparky.defense.lan:8099"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

TOPICS = ["Redis persistence", "B-tree splits", "TCP congestion control",
          "Rust borrow checking", "DNS resolution", "RAID levels",
          "Unix pipes", "Git internals", "CPU branch prediction",
          "SQL isolation levels", "Bloom filters", "NUMA effects",
          "TLS handshakes", "LSM trees", "io_uring", "cache coherency"]

def chat(messages, max_tokens=160):
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps({"model": "deepseek-v4-flash", "messages": messages,
                    "max_tokens": max_tokens, "temperature": 0,
                    "stream": True}).encode(),
        {"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=900)
    ttft = None
    text = []
    for line in resp:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        if ttft is None:
            ttft = time.time() - t0
        try:
            d = json.loads(line[6:])
            c = d["choices"][0]["delta"].get("content")
            if c: text.append(c)
        except Exception:
            pass
    return ttft or (time.time() - t0), "".join(text), time.time() - t0

convs = []
for i in range(N):
    filler = (f"Background context, item {i}: " +
              "the quick brown fox jumps over the lazy dog. " * 220)
    convs.append([{"role": "user",
                   "content": filler +
                   f"\n\nNow: explain {TOPICS[i % len(TOPICS)]} in two sentences."}])

wall0 = time.time()
ttfts = {t: [] for t in range(1, TURNS + 1)}
for turn in range(1, TURNS + 1):
    for i in range(N):
        ttft, text, dur = chat(convs[i])
        ttfts[turn].append(ttft)
        convs[i].append({"role": "assistant", "content": text})
        convs[i].append({"role": "user",
                         "content": f"Good. Now add one more detail about "
                                    f"{TOPICS[i % len(TOPICS)]}, briefly."})
    med = sorted(ttfts[turn])[len(ttfts[turn]) // 2]
    print(f"turn {turn}: TTFT med {med:.2f}s  "
          f"min {min(ttfts[turn]):.2f}  max {max(ttfts[turn]):.2f}", flush=True)
print(f"total wall: {time.time() - wall0:.1f}s  (N={N}, turns={TURNS})")

#!/usr/bin/env python3
"""stall_live.py -- L312's stall check: a long prompt is not starved behind decoding streams.

    python3 tests/stall_live.py MODEL [--binary ./pulsar-server] [--port 8306] [--streams 4] [--long-chars 240000]
                                      [--max-s 240]

STREAMS chat streams decode -- each a loop of 512-token greedy turns re-sent until the long prompt's first token
(or MAX_S, the cap that shows a stall) -- and 4 s in a ~53k-token raw prompt arrives (the story fixture behind a
nonce: no cache serves it; 8 tokens out).  Before L312's value-per-time round selection a fused lane could hold the
prompt until the streams stopped; now it must finish while they decode.  PASS: its TTFT is under MAX_S - 5 s.  Rule 5:
the long prompt is the long lane (>= 40k prompt tokens) and the streams decoded inside the window (gaps measured).
Reported: TTFT, the streams' decode tok/s and their gaps (p50 / p99 / max) inside [the prompt's arrival, its first
token].

From ~/l312w/tbt2.py (LOOP_STREAMS=1 NO_ALONE=1, one rep) and the Flare battery's check (L317: in the repo, a leg of
tests/live_legs.py on Qwen's shared server, run first while its banks are free).  Alone (this command line) it runs on
an instance of its own.  Exit 0 only on PASS."""
import os, random, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools", "serve-bench"))
from live_server import Leg  # noqa: E402
from serve_matrix import CONC_TASKS, request, story_text  # noqa: E402


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))] if v else 0.0


class StallLeg(Leg):
    NAME = "l312-stall"

    def __init__(self, g, work, streams=4, long_chars=240000, max_s=240.0):
        Leg.__init__(self, g, work)
        self.S, self.LONG, self.MAX_S = streams, long_chars, max_s

    def warm(self, srv):
        res, longp, done = {}, {}, threading.Event()

        def stream(i):
            parts, k = [], 0
            t_end = time.perf_counter() + self.MAX_S
            while not done.is_set() and time.perf_counter() < t_end:
                parts.append(request(srv.port, "chat", CONC_TASKS[(i + k) % len(CONC_TASKS)], 512, 0.0,
                                     1234 + i + 97 * k, "m"))
                k += 1
            res[i] = [p["stamps"] for p in parts]

        th = [threading.Thread(target=stream, args=(i,)) for i in range(self.S)]
        for x in th:
            x.start()
        time.sleep(4.0)
        text = "[%.12f stall] " % random.random()
        story = story_text()
        while len(text) < self.LONG:
            text += story
        longp["t_send"] = time.perf_counter()
        longp["r"] = request(srv.port, "raw", text[:self.LONG], 8, 0.0, 7, "m")
        done.set()
        for x in th:
            x.join()
        t_a, t_b = longp["t_send"], longp["r"]["first_evt"]
        gaps, toks = [], 0
        for i in range(self.S):
            for st in res.get(i, []):
                rounds = [st[0]] if st else []
                for t in st[1:]:
                    if t - rounds[-1] > 0.002:   # a 2 ms burst is one round
                        rounds.append(t)
                gaps += [(b - a) * 1e3 for a, b in zip(rounds, rounds[1:]) if a >= t_a and b <= (t_b or t_a)]
                toks += sum(1 for t in st if t_a <= t <= (t_b or t_a))
        ttft = (t_b - t_a) if t_b else None
        pt = longp["r"].get("prompt_tokens") or 0
        win = max((t_b or t_a) - t_a, 1e-9)
        self.g.note("TTFT %s s, prompt %d tokens, decode %.1f tok/s in the window, gap p50/p99/max %.0f/%.0f/%.0f ms "
                    "over %d gaps" % ("%.1f" % ttft if ttft is not None else "-", pt, toks / win, pct(gaps, 50),
                                      pct(gaps, 99), max(gaps) if gaps else 0.0, len(gaps)))
        self.g.check(pt >= 40000 and gaps, "the lane under test ran: a %d-token prompt beside %d decoding streams "
                     "(%d gaps in the window)" % (pt, self.S, len(gaps)))
        self.g.check(ttft is not None and 0 < ttft < self.MAX_S - 5.0,
                     "L312 STALL CHECK: the long prompt's first token after %s s, under the %.0f s stream cap"
                     % ("%.1f" % ttft if ttft is not None else "never", self.MAX_S))


def _args(ap):
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--long-chars", type=int, default=240000)
    ap.add_argument("--max-s", type=float, default=240.0)


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(StallLeg, sys.argv[1:], ctx=65536, banks=8, port=8306, add_args=_args))

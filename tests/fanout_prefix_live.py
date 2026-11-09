#!/usr/bin/env python3
"""fanout_prefix_live.py -- L329 (a): requests that arrive together on a NEW shared prefix prefill it once (the
shared-prefix wait, PULSAR_SERVER_PREFIX_WAIT_MS), on a real pulsar-server.

    python3 tests/fanout_prefix_live.py MODEL --binary ./pulsar-server --work DIR [--port 8329] [--siblings 10]
                                        [--preamble 90000] [--stagger 0.25] [--ctx 65536] [--tag NAME] [--ab]

Claude Code's fan-out (production 2026-10-23 16:08, Qwen: 11 subagents within 30 s over ~25k-token prompts sharing
20-24k): before L329 every sibling bound at once and adopted only what the first had prefilled so far (12288), then
prefilled the rest of the shared prefix itself -- 108k prompt tokens for ~68k of distinct work, the first's prompt done
after 100.6 s.  With the wait a sibling stays queued until the first stands at the shared cut, then adopts it.

SIBLINGS OpenAI /v1/chat/completions requests sent STAGGER seconds apart from one thread each: one system prompt
([tag] + the first PREAMBLE characters of the story, ~20k tokens at 90000 on Qwen) and a task of its own as the user
message; greedy, max_tokens 1, logprobs with the top 5 -- the reply is the prompt's last row (no decode round runs, so
no batched decode's nondeterminism), and its logprobs are the prefill's logits to the bit.  Checks:
  * warm (the wait at its default): every request answered; the shared prefix prefilled ONCE -- exactly one request
    reports fewer cached tokens than CUT (the resume-grid floor of min prompt - TAIL, a lower bound of the shared
    prefix: TAIL = 200 tokens over-covers a sibling's task and the template's close), and the prompt tokens prefilled
    in all (prompt - cached) are at most the largest prompt plus each other sibling's part past CUT; the waits
    (/metrics pulsar:prefix_waits_total) started and every one REACHED its cut (none left or timed out); the log shows
    the leader standing at the shared cut (rule 5).  The last first token's time is reported.
  * fresh: each request ALONE, one after another (the first cold: 0 cached tokens): its reply token and top-5 logprobs
    == the burst's, bit for bit (the cut the leader stopped at, the adopt and the delta are the cold prefill's bytes).
  * --ab (alone only): a third instance with PULSAR_SERVER_PREFIX_WAIT_MS=0 runs the same burst: no wait starts, its
    replies == the burst's, it prefills more prompt tokens; the last first token with the wait on vs off is printed.

A leg of tests/live_legs.py (L317 Leg shape: warm + fresh; --ab adds the hooks instance).  Alone it runs on instances of
its own, PULSAR_MSEQ_BANKS=16.  Exit 0 only when every check held."""
import json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, cached_tokens, check_fresh  # noqa: E402
from prefix_turns_live import STORY  # noqa: E402

GRID = 128   # the resume grid (Qwen, DeepSeek): the wait's cut is a multiple of it
TAIL = 200   # tokens: over-covers a sibling's own task, the user turn and the template's close
TASKS = [
    "Your sub-task: list the characters who appear in the first chapter.",
    "Your sub-task: list the places the first chapter describes.",
    "Your sub-task: name the first conflict of the story.",
    "Your sub-task: name the first object the story describes.",
    "Your sub-task: say in what season the story opens.",
    "Your sub-task: say who speaks first in the story.",
    "Your sub-task: count the named animals in the story.",
    "Your sub-task: name the story's narrator, if it has one.",
    "Your sub-task: quote the story's first line of dialogue.",
    "Your sub-task: say where the story's first scene takes place.",
    "Your sub-task: name a colour the story mentions early on.",
    "Your sub-task: say what time of day the story opens at.",
]
WAITS = ("started", "reached", "left", "timeout")


def waits(m):
    return {k: m.get('pulsar:prefix_waits_total{outcome="%s"}' % k, 0.0) for k in WAITS}


class FanoutPrefixLeg(Leg):
    NAME = "fanout-prefix"
    fresh = True

    def __init__(self, g, work, siblings=10, preamble=90000, stagger=0.25, tag="fanout", ab=False):
        Leg.__init__(self, g, work)
        self.n, self.stagger, self.ab = min(siblings, len(TASKS)), stagger, ab
        self.system = ("[%s] You are one of several sub-agents reading the story below; each of you has its own "
                       "sub-task, given after it.\n\n" % tag) + STORY[:preamble]
        self.on = None   # the warm burst's results

    def body(self, k):
        return {"model": "m", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": TASKS[k]}]}

    @staticmethod
    def reply(d):
        """the reply token and its top-5 logprobs, as the server printed them (%.9g of the f32 values)"""
        c = (d.get("choices") or [{}])[0]
        lp = ((c.get("logprobs") or {}).get("content") or [{}])[0]
        return json.dumps({"text": (c.get("message") or {}).get("content"), "token": lp.get("token"),
                           "logprob": lp.get("logprob"),
                           "top": [(t.get("token"), t.get("logprob")) for t in lp.get("top_logprobs", [])]},
                          sort_keys=True)

    def burst(self, srv, what):
        """SIBLINGS requests, STAGGER s apart, each on a thread of its own: [(status, json, sent_s, done_s)]"""
        out = [None] * self.n
        t0 = time.time()

        def one(k):
            time.sleep(k * self.stagger)
            sent = time.time() - t0
            st, d = srv.post_json("/v1/chat/completions", self.body(k), timeout=1800)
            out[k] = (st, d, sent, time.time() - t0)

        th = [threading.Thread(target=one, args=(k,)) for k in range(self.n)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        res = {"status": [o[0] for o in out], "reply": [self.reply(o[1]) for o in out],
               "prompt": [(o[1].get("usage") or {}).get("prompt_tokens", 0) for o in out],
               "cached": [cached_tokens(o[1].get("usage")) or 0 for o in out],
               "sent": [o[2] for o in out], "done": [o[3] for o in out]}
        res["prefilled"] = sum(p - c for p, c in zip(res["prompt"], res["cached"]))
        res["last"] = max(res["done"])
        for k in range(self.n):
            print("# [%s] %s request %d: sent %.2f s, answered %.2f s, prompt %d, cached %d" % (
                self.NAME, what, k, res["sent"][k], res["done"][k], res["prompt"][k], res["cached"][k]), flush=True)
        print("# [%s] %s: %d prompt tokens prefilled, the last first token at %.1f s" % (
            self.NAME, what, res["prefilled"], res["last"]), flush=True)
        return res

    def warm(self, srv):
        w0, mark = waits(srv.metrics()), srv.mark()
        r = self.burst(srv, "burst (wait on)")
        w = waits(srv.metrics())
        d = {k: w[k] - w0[k] for k in WAITS}
        self.g.check(all(s == 200 for s in r["status"]), "every request answered: %s" % r["status"])
        cut = (min(r["prompt"]) - TAIL) // GRID * GRID
        short = [k for k in range(self.n) if r["cached"][k] < cut]
        self.g.check(len(short) == 1, "the shared prefix prefilled once: %d request(s) resumed below the cut %d "
                     "(cached %s)" % (len(short), cut, r["cached"]))
        bound = max(r["prompt"]) + sum(p - cut for p in r["prompt"]) - (max(r["prompt"]) - cut)
        self.g.check(r["prefilled"] <= bound, "%d prompt tokens prefilled in all <= %d (the largest prompt + each "
                     "other request's part past the cut)" % (r["prefilled"], bound))
        self.g.check(d["started"] >= 1 and d["reached"] == d["started"] and d["left"] == 0 and d["timeout"] == 0,
                     "the waits: %s (every started wait reached its cut)" % d)
        stood = [l for l in srv.since(mark).splitlines() if "the shared cut a queued request waits for" in l]
        self.g.check(len(stood) >= 1, "the leader stood at the shared cut: %s" % (stood[:2] or "no such line"))
        self.on = r

    def cold(self, srv):
        if not self.on:
            self.g.check(False, "no warm burst to compare with")
            return
        for k in range(self.n):
            st, d = srv.post_json("/v1/chat/completions", self.body(k), timeout=1800)
            if k == 0:
                check_fresh(self.g, "the first solo request", d.get("usage"))
            self.g.check(st == 200 and self.reply(d) == self.on["reply"][k],
                         "request %d alone == in the burst: %s" % (k, self.reply(d) if st == 200 else st))

    def hook_env(self):
        return {"PULSAR_SERVER_PREFIX_WAIT_MS": "0"} if self.ab else None

    def hooked(self, srv):
        w0 = waits(srv.metrics())
        r = self.burst(srv, "burst (wait off)")
        w = waits(srv.metrics())
        self.g.check(w["started"] == w0["started"], "PULSAR_SERVER_PREFIX_WAIT_MS=0: no wait started")
        if not self.on:
            return
        self.g.check(r["reply"] == self.on["reply"], "every reply with the wait off == with it on")
        self.g.check(r["prefilled"] > self.on["prefilled"], "the wait off prefills more: %d vs %d prompt tokens" % (
            r["prefilled"], self.on["prefilled"]))
        print("# [%s] A/B: %d siblings over a %d-token prompt: prefilled %d -> %d tokens, last first token "
              "%.1f -> %.1f s (wait off -> on)" % (self.NAME, self.n, min(r["prompt"]), r["prefilled"],
                                                  self.on["prefilled"], r["last"], self.on["last"]), flush=True)


def _args(ap):
    ap.add_argument("--siblings", type=int, default=10)
    ap.add_argument("--preamble", type=int, default=90000)
    ap.add_argument("--stagger", type=float, default=0.25)
    ap.add_argument("--tag", default="fanout")
    ap.add_argument("--ab", action="store_true", help="also run the burst with the wait off (a third instance)")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(FanoutPrefixLeg, sys.argv[1:], ctx=65536, banks=16, port=8329, add_args=_args))

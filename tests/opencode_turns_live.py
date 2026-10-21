#!/usr/bin/env python3
"""opencode_turns_live.py -- L311: opencode's conversation shape on a thinking-on model whose template truncates
history reasoning (Nemotron's truncate_history_thinking), measured on a real pulsar-server.

    python3 tests/opencode_turns_live.py MODEL --binary ./pulsar-server --work DIR --capture REQ.json [--tag T]

opencode sends exact appends: the captured request's system prompt and its tools, then per turn the user message,
the previous assistant turn coming back with its `reasoning_content` -- which the template re-renders as
"<think></think>answer", so the bank (holding the reply as decoded) and the prompt part right after the previous
generation header.  Here the user messages are the capture's own, the replies the model's (greedy, streamed,
max_tokens --gen), the client pausing --pause s between turns.

Per turn, from the server's log for that request: the bank it ran on and whether it is the previous turn's, where
its sync resumed (restored tokens) and the tokens it prefilled; and its TTFT.  REUSE == COLD: the last turn sent
again alone to the warm server and to a fresh one (an empty disk tier) -- the same greedy reply.

L317: a leg of tests/live_legs.py -- the turns on the model's shared server (warm), the reference on its fresh
instance (0 cached tokens asserted).  Alone (this command line) it runs on instances of its own, PULSAR_MSEQ_BANKS=4.
Exit 0 only when reuse == cold held."""
import json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_server import Leg, check_fresh  # noqa: E402

ROUTE = re.compile(r"route: best bank (\d+) .* -> (in place|fresh preferred)")
PROV = re.compile(r"provisioned bank (\d+)")
RESUME = re.compile(r"bank (\d+) (?:resumes from its (prompt end|message endpoint|checkpoint) at (\d+)|"
                    r"resumes at its prompt end (\d+)|adopts bank \d+'s [a-z ]+ at (\d+))")


def turn_facts(log_part):
    """(bank, restored) of one request from its log lines"""
    bank, restored, routed = None, 0, False
    for l in log_part.splitlines():
        m = ROUTE.search(l)
        if m and m.group(2) == "in place":
            bank, routed = int(m.group(1)), True
        m = PROV.search(l)
        if m:
            bank, routed = int(m.group(1)), True
        m = RESUME.search(l) if routed else None   # a resume before the route is the previous turn's terminal step
        if m:
            bank = int(m.group(1)) if bank is None else bank
            restored = int(next(g for g in m.groups()[2:] if g))
    return bank, restored


class OpencodeLeg(Leg):
    NAME = "opencode-turns"
    fresh = True

    def __init__(self, g, work, capture=os.path.join(HERE, "test-vectors", "opencode-nemotron-turn3.json"), gen=400,
                 pause=2.0, tag="run"):
        Leg.__init__(self, g, work)
        cap = json.load(open(capture))
        cap = cap.get("body", cap)
        self.SYSTEM = [m for m in cap["messages"] if m["role"] == "system"]
        self.USERS = [m["content"] for m in cap["messages"] if m["role"] == "user"]
        self.TOOLS = cap.get("tools") or []
        self.gen, self.pause, self.tag = gen, pause, tag
        self.last = self.warm_again = None

    def chat(self, srv, msgs):
        body = {"model": "m", "messages": msgs, "tools": self.TOOLS, "stream": True, "max_tokens": self.gen,
                "temperature": 0, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(srv.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        t = time.time()
        content, reasoning, ttft, usage = "", "", None, {}
        for raw in urllib.request.urlopen(req, timeout=3600):
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
        return {"content": content, "reasoning": reasoning, "ttft": ttft or time.time() - t, "usage": usage}

    def warm(self, srv):
        rows = []
        msgs = list(self.SYSTEM)
        prev_bank = None
        for t, u in enumerate(self.USERS, 1):
            msgs = msgs + [{"role": "user", "content": u}]
            mark = srv.mark()
            r = self.chat(srv, msgs)
            time.sleep(0.5)
            prompt = (r["usage"] or {}).get("prompt_tokens", 0)
            bank, restored = turn_facts(srv.since(mark))
            same = prev_bank is not None and bank == prev_bank
            rows.append((t, bank, same, restored, prompt - restored, r["ttft"]))
            print("%s\tturn %d\tbank %s\tsame %s\trestored %d\tprefilled %d\tttft %.3f" %
                  (self.tag, t, bank, "-" if prev_bank is None else ("yes" if same else "NO"), restored,
                   prompt - restored, r["ttft"]), flush=True)
            prev_bank = bank
            self.last = list(msgs)
            msgs = msgs + [{"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}]
            time.sleep(self.pause)
        self.warm_again = self.chat(srv, self.last)
        later = [r for r in rows if r[0] > 1]
        if later:
            self.g.note("%s turns>1: same bank %d/%d, mean restored %.0f, mean prefilled %.0f, mean ttft %.3f s" %
                        (self.tag, sum(1 for r in later if r[2]), len(later), sum(r[3] for r in later) / len(later),
                         sum(r[4] for r in later) / len(later), sum(r[5] for r in later) / len(later)))

    def cold(self, srv):
        if not self.warm_again:
            self.g.check(False, "no warm re-send to compare (the warm phase did not finish)")
            return
        cold_r = self.chat(srv, self.last)
        check_fresh(self.g, "the last turn", cold_r["usage"])
        w = self.warm_again
        same = (w["content"], w["reasoning"]) == (cold_r["content"], cold_r["reasoning"])
        self.g.check(same, "reuse == cold: the last turn re-sent alone == a fresh server's")
        if not same:
            print("  warm %r\n  cold %r" % ((w["reasoning"] + "|" + w["content"])[:300],
                                            (cold_r["reasoning"] + "|" + cold_r["content"])[:300]))


def _args(ap):
    ap.add_argument("--capture", required=True, help="an opencode request (its body's messages and tools)")
    ap.add_argument("--gen", type=int, default=400)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--tag", default="run")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(OpencodeLeg, sys.argv[1:], ctx=65536, banks=4, port=8312, add_args=_args))

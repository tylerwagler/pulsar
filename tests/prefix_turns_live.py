#!/usr/bin/env python3
"""prefix_turns_live.py -- L310 / L311 / L313 / L316: an agent loop's prefix reuse, measured on a real pulsar-server.

    python3 tests/prefix_turns_live.py MODEL --binary ./pulsar-server --work DIR [--port 8311] [--turns 8]
                                       [--side] [--reply N] [--ctx 65536] [--tag NAME]

An Anthropic /v1/messages tool loop (Claude Code's shape): a system prompt, a task, then TURNS rounds of a forced
`read_file` call (greedy) answered by a tool_result of ~1.2k tokens of story text, the client pausing PAUSE seconds
between the model's turn and its result (the tool's run time).  Per turn: the prompt's tokens, the cache read the
server reports (where the sync resumed), recomputed = prompt - cached, and the request's TTFT (the response's
timings block).  With --side, after each tool result a SIDE request shares the main conversation through its last
message and adds a "Describe your most recent action in 3-5 words" user message while the main turn is still being
answered (Claude Code's agent summary) -- its TTFT and recompute are reported apart.

REUSE == COLD (must-pass): the last main request is sent again to a FRESH server (an empty disk tier, cold banks)
and its greedy reply must equal the warm one byte for byte; with --side the last side request too.  The client keeps
the replies' thinking blocks in its history (an Anthropic client echoes them inside a tool loop): the warm bank's
live continuation holds the reasoning it generated, so a history without it is another conversation to a fresh
server (L316: 10587 vs 10093 tokens, the in-loop check failing on every pack).  Each graded pair must also be one
prompt (the same prompt token count), so a check cannot pass or fail on two different conversations.

L317: a leg of tests/live_legs.py -- the loop on the model's shared server (warm), the reference requests on its fresh
instance, which must report 0 cached tokens.  The system prompt opens with the leg's tag, so two prefix legs (or any
other leg) never share a prefix on either instance.  Alone (this command line) it runs on instances of its own,
PULSAR_MSEQ_BANKS=4.  Prints one TSV line a request (tag, kind, turn, prompt, cached, recomputed, ttft_s, e2e_s) and a
summary; exit 0 only when every check held."""
import json, os, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, check_fresh  # noqa: E402

STORY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_context_story_prompt.txt"),
             encoding="utf-8", errors="replace").read()
TOOLS = [{"name": "read_file", "description": "Read a numbered part of the project's notes file.",
          "input_schema": {"type": "object", "properties": {"part": {"type": "integer"}}, "required": ["part"]}}]
SYSTEM = ("You are a careful assistant working through a long notes file one part at a time with the read_file tool. "
          "Read the parts in order, starting at part 1.  Before each read_file call, write a detailed summary of "
          "the part you just read.  " + STORY[:6000])


def post(srv, body):
    """One streamed /v1/messages request: (the message as a non-streamed reply would carry it -- content + usage --,
    its e2e seconds); d["ttft_s"] = seconds to the first content delta, d["prompt_tokens"] its whole prompt."""
    body = dict(body, stream=True)
    req = urllib.request.Request(srv.url + "/v1/messages", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    r = urllib.request.urlopen(req, timeout=1200)
    msg, blocks, parts, ttft = {}, [], [], None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        ty = ev.get("type")
        if ty == "message_start":
            msg = ev["message"]
        elif ty == "content_block_start":
            blocks.append(dict(ev["content_block"]))
            parts.append("")
        elif ty == "content_block_delta":
            if ttft is None:
                ttft = time.time() - t
            dl = ev["delta"]
            if dl.get("type") == "text_delta":
                blocks[-1]["text"] = blocks[-1].get("text", "") + dl["text"]
            elif dl.get("type") == "input_json_delta":
                parts[-1] += dl["partial_json"]
            elif dl.get("type") == "thinking_delta":
                blocks[-1]["thinking"] = blocks[-1].get("thinking", "") + dl["thinking"]
            elif dl.get("type") == "signature_delta":
                blocks[-1]["signature"] = blocks[-1].get("signature", "") + dl["signature"]
        elif ty == "content_block_stop":
            if blocks and blocks[-1].get("type") == "tool_use":
                blocks[-1]["input"] = json.loads(parts[-1]) if parts[-1] else {}
    # L316: the thinking blocks stay in the reply and so in the client's history, as an Anthropic client echoes them
    # inside a tool loop.  Dropping them made the fresh server render a different conversation (an empty think block
    # where the warm bank's live continuation holds the reasoning it generated): 10093 vs 10587 prompt tokens.
    msg["content"] = blocks
    msg["ttft_s"] = ttft if ttft is not None else time.time() - t
    u = msg.get("usage", {})
    msg["prompt_tokens"] = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) +
                            u.get("cache_creation_input_tokens", 0))
    return msg, time.time() - t


def reply_text(d):
    """The reply's content, thinking included, without the server's random tool_use ids and thinking signatures (each
    server names its calls and stamps its thinking blocks with a response id afresh)."""
    return json.dumps([{k: v for k, v in c.items() if k not in ("id", "signature")} for c in d.get("content", [])],
                      sort_keys=True)


class PrefixTurnsLeg(Leg):
    NAME = "prefix-turns"
    fresh = True

    def __init__(self, g, work, turns=8, pause=2.0, side=False, reply=0, tag="run"):
        Leg.__init__(self, g, work)
        self.turns, self.pause, self.side, self.reply, self.tag = turns, pause, side, reply, tag
        self.system = "[%s] " % tag + SYSTEM   # L317: no other leg's conversation shares this one's prefix
        self.rows, self.last_main = [], None

    def body(self, msgs, forced, max_tokens):
        b = {"model": "m", "max_tokens": max_tokens, "temperature": 0, "system": self.system, "tools": TOOLS,
             "messages": msgs}
        if forced:
            b["tool_choice"] = {"type": "tool", "name": "read_file"}
        return b

    def measure(self, srv, kind, turn, b):
        d, e2e = post(srv, b)
        prompt = d["prompt_tokens"]
        cached = d.get("usage", {}).get("cache_read_input_tokens", 0)
        row = (self.tag, kind, turn, prompt, cached, prompt - cached, d["ttft_s"], e2e)
        self.rows.append(row)
        print("%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f" % row, flush=True)
        return d

    def loop(self, srv):
        msgs = [{"role": "user", "content": "Summarise the notes file. Begin with read_file part 1."}]
        last_main = last_side = None
        th = None
        for t in range(1, self.turns + 1):
            # --reply N: the turn writes a reply of up to N tokens (the decode rows a next turn re-prefilled, L275's
            # ~969), then calls the tool; else a bare forced call
            b = self.body(msgs, self.reply == 0, self.reply if self.reply else 96)
            d = self.measure(srv, "main", t, b)
            if th:
                th.join()   # the side request rode beside this main turn
                th = None
            last_main = (b, d)
            calls = [c for c in d.get("content", []) if c.get("type") == "tool_use"]
            msgs = msgs + [{"role": "assistant", "content": d["content"] or [{"type": "text", "text": "."}]}]
            time.sleep(self.pause)   # the tool runs
            part = STORY[6000 + t * 4800: 6000 + (t + 1) * 4800]
            if calls:
                msgs = msgs + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"],
                                                             "content": part}]}]
            else:
                msgs = msgs + [{"role": "user", "content": "Part %d of the notes:\n%s\n\nSummarise it in detail, then "
                                                           "read the next part." % (t + 1, part)}]
            if self.side and t < self.turns:
                sb = self.body(msgs + [{"role": "user", "content": "Describe your most recent action in 3-5 words "
                                                                   "using present tense (-ing)."}], False, 24)
                th = threading.Thread(target=lambda sb=sb, t=t: self.measure(srv, "side", t, sb))
                th.start()
                time.sleep(0.2)   # the next main turn arrives while the side request is being answered
                last_side = (sb, None)
        if th:
            th.join()
        return last_main, last_side

    def warm(self, srv):
        mark = srv.mark()
        self.last_main, self.last_side = self.loop(srv)
        new = srv.since(mark).splitlines()
        term = [l for l in new if "terminal checkpoint" in l]
        disk = [l for l in new if "restore-vs-recompute" in l]
        self.g.note("terminal checkpoint lines: %d; restore-vs-recompute lines: %d" % (len(term), len(disk)))
        for l in term[-3:] + disk[-3:]:
            print("#   " + l[-200:])
        self.main_again = self.side_warm = None
        if self.last_main:
            time.sleep(self.pause)
            self.main_again, _ = post(srv, self.last_main[0])   # alone on the idle warm server (its prompt end resumes it)
        if self.last_side:
            self.side_warm, _ = post(srv, self.last_side[0])
        mains = [r for r in self.rows if r[1] == "main" and r[2] > 1]
        sides = [r for r in self.rows if r[1] == "side"]
        for name, rs in (("main", mains), ("side", sides)):
            if rs:
                self.g.note("%s turns>1: n=%d mean recomputed %.0f tokens, mean ttft %.3f s" %
                            (name, len(rs), sum(r[5] for r in rs) / len(rs), sum(r[6] for r in rs) / len(rs)))

    def cold(self, srv):
        if not self.last_main:
            self.g.check(False, "no last main turn to compare (the warm phase did not finish)")
            return
        dc, _ = post(srv, self.last_main[0])
        check_fresh(self.g, "the last main turn", dc.get("usage"))
        # in the loop beside a side request the batched decode is nondeterministic by design: there only the re-send,
        # alone, is graded
        for what, warm in (("in the loop", None if self.side else self.last_main[1]), ("re-sent alone", self.main_again)):
            if warm is None:
                continue
            # L316: the instrument proves it compared ONE conversation -- the warm request's effective prompt (a live
            # continuation keeps the bank's tokens) and the fresh server's render are the same length, or the replies
            # say nothing about reuse
            self.g.check(warm["prompt_tokens"] == dc["prompt_tokens"],
                         "the last main turn (%s) and the fresh server's are one prompt: %d vs %d tokens" %
                         (what, warm["prompt_tokens"], dc["prompt_tokens"]))
            same = reply_text(dc) == reply_text(warm)
            self.g.check(same, "reuse == cold: the last main turn (%s) == a fresh server's" % what)
            if not same:
                print("  warm %s\n  cold %s" % (reply_text(warm)[:400], reply_text(dc)[:400]))
        if self.side_warm is not None:
            ds, _ = post(srv, self.last_side[0])
            self.g.check(reply_text(ds) == reply_text(self.side_warm),
                         "reuse == cold: the last side request's greedy reply on a fresh server")


def _args(ap):
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--side", action="store_true")
    ap.add_argument("--reply", type=int, default=0, help="a turn writes up to this many tokens before its call (0: a "
                    "bare forced call)")
    ap.add_argument("--tag", default="run")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(PrefixTurnsLeg, sys.argv[1:], ctx=65536, banks=4, port=8311, add_args=_args))

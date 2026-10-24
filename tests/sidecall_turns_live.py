#!/usr/bin/env python3
"""sidecall_turns_live.py -- L322: a side call that extends a conversation's bank does not move the conversation's next
turn to another bank, on a real pulsar-server.

    python3 tests/sidecall_turns_live.py MODEL --binary ./pulsar-server --work DIR [--port 8322] [--rounds 3]
                                         [--ctx 65536] [--tag NAME]

Claude Code's shape (production, 2026-10-24): after every finished turn it sends a SUGGESTION MODE side call -- the
same conversation plus ~340 tokens (a user turn and its generation header).  The side call carries everything the
bank prefilled, so it runs in place and extends the bank; the next REAL turn carries the conversation through the real
turn's end and the next user marker, then diverges inside the side call's user turn.  Before L322 that turn took a
fresh bank (L311's route anchor was the deeper of the bank's last user marker and its generation header, so the side
call's header hid the last-turn rule), and with the adopt capped at a token seam it reloaded the whole chain from disk:
22-25 s a turn at 178-193k.

Here: an Anthropic /v1/messages conversation (greedy; a system prompt opening with the leg's tag, then ROUNDS user
questions), each answered turn followed by a side call of ~340 tokens, the client pausing --pause s between requests
(the server's terminal checkpoint runs then, as in production).  Per real turn after the first, from the server's log
for that request:
  * IN PLACE: the route line says `-> in place` on the bank the side call (and the previous real turn) ran on, and the
    resume decision is the bank's own (`-> own`), no disk chain loaded;
  * DELTA: the sync resumed within one resume grid (128 tokens) of the route's match -- the bank rewound to its own
    checkpoint below the divergence and prefilled the delta, not the conversation.
REUSE == COLD (must-pass): the last real turn sent to a FRESH server (an empty disk tier, which must report 0 cached
tokens) gives the same greedy reply, on the same prompt token count.

A leg of tests/live_legs.py (L317 Leg shape: warm on the model's shared server, cold on its fresh instance).  Alone
(this command line) it runs on instances of its own, PULSAR_MSEQ_BANKS=4.  Exit 0 only when every check held."""
import os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, check_fresh  # noqa: E402
from prefix_turns_live import STORY, post, reply_text  # noqa: E402

ROUTE = re.compile(r"route: best bank (\d+) common (\d+) resume (-?\d+) frontier (\d+) prefilled (\d+) anchor (-?\d+) "
                   r"prompt (\d+) -> (in place|fresh preferred)")
RESUME = re.compile(r"resume: bank (\d+) own -?\d+ adopt -?\d+(?: \(none\))? disk -?\d+ of \d+ \(match -?\d+\) -> (\w+)")
START = re.compile(r"chat ctx=(\d+)\.\.(\d+):\d+ .*prompt start")
QUESTIONS = [
    "Who is the main character of the story, and what do they want? Answer in two sentences.",
    "Name two places the story describes and what happens at each, briefly.",
    "Which conflict drives the middle of the story? Answer in a few sentences.",
    "How does the story end? Summarise the resolution briefly.",
    "What is the most important object in the story, and why?",
]
# Claude Code's prompt-suggestion instruction, paraphrased and padded to its ~340-token size
SUGGESTION = ("[SUGGESTION MODE: Suggest what the user might naturally type next into this conversation.]\n\n"
              "Look at the conversation so far and predict the user's most likely next message. Reply with ONLY the "
              "suggestion text, in the user's own voice, at most twelve words, with no quotes, no preamble and no "
              "explanation. If nothing obvious would follow, reply with an empty line.\n\nRules:\n" +
              "".join("- Rule %d: keep the suggestion short, specific to the last answer, and phrased as the user "
                      "would phrase it; never answer the question yourself.\n" % i for i in range(1, 12)))


def facts(part):
    """(route groups, resume pick and bank, (resumed at, prompt end), disk loaded) of one request's log part"""
    r = ROUTE.search(part)
    p = RESUME.search(part)
    s = START.search(part)
    return (r.groups() if r else None, (int(p.group(1)), p.group(2)) if p else None,
            (int(s.group(1)), int(s.group(2))) if s else None, "kv cache hit chain" in part)


class SidecallLeg(Leg):
    NAME = "sidecall-turns"
    fresh = True

    def __init__(self, g, work, rounds=3, pause=1.5, tag="sidecall"):
        Leg.__init__(self, g, work)
        self.rounds, self.pause, self.tag = rounds, pause, tag
        self.system = ("[%s] You answer questions about the story below, briefly and precisely.\n\n" % tag +
                       STORY[:8000])
        self.last = self.last_warm = None

    def body(self, msgs, max_tokens):
        return {"model": "m", "max_tokens": max_tokens, "temperature": 0, "system": self.system, "messages": msgs}

    def request(self, srv, b):
        m = srv.mark()
        d, _ = post(srv, b)
        return d, facts(srv.since(m))

    def warm(self, srv):
        msgs, conv_bank = [], None   # the bank the conversation's last request ran on
        for k in range(1, self.rounds + 1):
            msgs = msgs + [{"role": "user", "content": QUESTIONS[(k - 1) % len(QUESTIONS)]}]
            b = self.body(msgs, 160)
            d, (route, pick, start, disk) = self.request(srv, b)
            print("# [%s] real turn %d: prompt %d, route %s, resume %s, sync %s, disk %s"
                  % (self.tag, k, d["prompt_tokens"], route, pick, start, disk), flush=True)
            if k > 1:
                self.g.check(route is not None and route[7] == "in place" and int(route[0]) == conv_bank,
                             "real turn %d routes in place on the side call's bank %s (route %s)" % (k, conv_bank,
                                                                                                     route))
                self.g.check(pick is not None and pick[1] == "own" and not disk,
                             "real turn %d resumes from the bank's own state, no disk chain (resume %s, disk %s)"
                             % (k, pick, disk))
                self.g.check(route is not None and start is not None and start[0] >= int(route[1]) - 128,
                             "real turn %d resumed at %s, within one grid (128) of the match %s -- prefilled %s of %s"
                             % (k, start and start[0], route and route[1], start and start[1] - start[0],
                                start and start[1]))
            self.last, self.last_warm = b, d
            conv_bank = pick[0] if pick else None
            msgs = msgs + [{"role": "assistant", "content": d["content"] or [{"type": "text", "text": "."}]}]
            if k == self.rounds:
                break
            time.sleep(self.pause)   # the client renders the answer; the server's terminal checkpoint runs
            sd, (sroute, spick, sstart, sdisk) = self.request(srv, self.body(msgs + [{"role": "user",
                                                                                    "content": SUGGESTION}], 24))
            print("# [%s] side call %d: prompt %d, route %s, resume %s" % (self.tag, k, sd["prompt_tokens"], sroute,
                                                                         spick), flush=True)
            self.g.check(sroute is not None and sroute[7] == "in place" and spick is not None and
                         int(sroute[0]) == conv_bank and spick[0] == conv_bank,
                         "side call %d extends the conversation's bank %s in place (route %s, resume %s)"
                         % (k, conv_bank, sroute, spick))
            time.sleep(self.pause)

    def cold(self, srv):
        if not self.last:
            self.g.check(False, "no last real turn to compare (the warm phase did not finish)")
            return
        dc, _ = post(srv, self.last)
        check_fresh(self.g, "the last real turn", dc.get("usage"))
        self.g.check(dc["prompt_tokens"] == self.last_warm["prompt_tokens"],
                     "the last real turn and the fresh server's are one prompt: %d vs %d tokens"
                     % (self.last_warm["prompt_tokens"], dc["prompt_tokens"]))
        same = reply_text(dc) == reply_text(self.last_warm)
        self.g.check(same, "reuse == cold: the last real turn's greedy reply == a fresh server's")
        if not same:
            print("  warm %s\n  cold %s" % (reply_text(self.last_warm)[:400], reply_text(dc)[:400]))


def _args(ap):
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--tag", default="sidecall")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(SidecallLeg, sys.argv[1:], ctx=65536, banks=4, port=8322, add_args=_args))

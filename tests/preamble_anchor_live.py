#!/usr/bin/env python3
"""preamble_anchor_live.py -- L353: new conversations of one client share their system prompt, and a restore point at its
end serves them, on a real pulsar-server.

    python3 tests/preamble_anchor_live.py MODEL --binary ./pulsar-server --work DIR [--port 8325] [--preamble 40000] [--block 20000]
                                          [--ctx 65536] [--tag NAME]

Claude Code's new-session shape: a long system prompt, a first user message (shared reminders, then the typed text), a
second user message (a shared block) -- so sessions part at the typed text, past the system prompt and before the
last user marker.  Three sessions, one after another, greedy, that differ only in the typed text.  From the log:
  * FIRST SIGHTING: the anchor names a cut, its rows are not stored, nothing is written;
  * SECOND SIGHTING: the rows are stored and no restore point stands there: written once (reason=anchor), at the same
    cut, though the session resumed from a shallower point another bank holds;
  * THIRD: the restore point is stored, no write; the session resumes at or past the cut;
  * THE CUT is a function of the prompt: the same on all three, below the typed text;
  * REUSE == COLD: the third session's reply is byte for byte a fresh server's on the same request.

A leg of tests/live_legs.py (L317 Leg shape: warm + a fresh instance).  Exit 0 only when every check held."""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, cached_tokens, check_fresh  # noqa: E402
from prefix_turns_live import STORY, post, reply_text  # noqa: E402

ANCHOR = re.compile(r"kv cache anchor at (\d+): rows (not stored|stored)[^,]*, restore point (none|stored) -> (no write|write it)")
WROTE = re.compile(r"persisted reason=anchor chain to (\d+)")
TYPED = ["Name the first character of the story.", "Name the first place of the story.",
         "Name the first object of the story."]


class PreambleAnchorLeg(Leg):
    NAME = "preamble-anchor"
    fresh = True

    def __init__(self, g, work, preamble=40000, block=20000, tag="preamble"):
        Leg.__init__(self, g, work)
        self.system = ("[%s] You are a careful reader of the story below.\n\n" % tag) + STORY[:preamble]
        # as in Claude Code: a long shared block AFTER the typed text, so a bank's grid is thinned where sessions part
        # and no bank holds a checkpoint near the system prompt's end by chance
        self.shared = "<reminder>Answer in one short sentence.</reminder>\n"
        self.block = "<environment>Working directory: /tmp/%s\n%s</environment>" % (tag, STORY[preamble:preamble + block])
        self.reply = None

    def body(self, k):
        return {"model": "m", "max_tokens": 48, "temperature": 0, "system": self.system,
                "messages": [{"role": "user", "content": self.shared + TYPED[k]},
                             {"role": "user", "content": self.block}]}

    def warm(self, srv):
        cuts = []
        for k in range(len(TYPED)):
            mark = srv.mark()
            d, _ = post(srv, self.body(k))
            log = srv.since(mark)
            a = ANCHOR.search(log)
            wrote = WROTE.findall(log)
            print("# [%s] session %d: %d prompt tokens, anchor %s, wrote %s, cached %s" % (
                self.NAME, k + 1, d["prompt_tokens"], a.groups() if a else None, wrote,
                cached_tokens(d.get("usage"))), flush=True)
            if k == 0:
                self.g.check(a is not None and a.group(2) == "not stored" and not wrote,
                             "first sighting: the rows are not stored and nothing is written")
            elif k == 1:
                self.g.check(a is not None and a.group(2) == "stored" and a.group(4) == "write it",
                             "second sighting: the rows are stored, no restore point stands there: write it")
                self.g.check(len(wrote) == 1 and a is not None and int(wrote[0]) == int(a.group(1)),
                             "the anchor is written once, at the cut: %s" % wrote)
            else:
                self.g.check(not wrote, "third sighting: nothing is written: %s" % wrote)
                self.g.check(a is None or a.group(3) == "stored",
                             "third sighting: the restore point stands, or the session resumed at or past the cut")
                self.reply = reply_text(d)
                c = cached_tokens(d.get("usage"))
                self.g.check(c is not None and cuts and c >= cuts[0],
                             "third session resumes at or past the cut: %s cached, cut %s" % (c, cuts))
            if a:
                cuts.append(int(a.group(1)))
        self.g.check(cuts and len(set(cuts)) == 1, "the cut is the same on every session that named one: %s" % cuts)
        self.g.check(cuts and cuts[0] % 128 == 0, "the cut sits on the resume grid: %s" % cuts)

    def cold(self, srv):
        d, _ = post(srv, self.body(len(TYPED) - 1))
        check_fresh(self.g, "the third session's request", d.get("usage"))
        self.g.check(self.reply is not None and reply_text(d) == self.reply,
                     "reuse == cold: the third session's reply is a fresh server's, byte for byte")


def _args(ap):
    ap.add_argument("--preamble", type=int, default=40000)
    ap.add_argument("--block", type=int, default=20000)
    ap.add_argument("--tag", default="preamble")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(PreambleAnchorLeg, sys.argv[1:], ctx=65536, banks=4, port=8325, add_args=_args))

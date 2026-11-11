#!/usr/bin/env python3
"""resume_deepest_live.py -- L318: an Open WebUI conversation resumes from the DEEPEST valid state, and a prefill its
client abandons keeps its work for the re-send.

    python3 tests/resume_deepest_live.py MODEL [--binary ./pulsar-server] [--work DIR] [--port 8318] [--turns 5]
                                         [--ctx 262144] [--banks 16] [--segment 4096] [--report-only]

The shape (production, 2026-10-21): Open WebUI injects its tool / web-search results into the request it sends but
stores the plain message, so every request rewrites the history INSIDE the previous turn -- before the bank's last user
marker, where neither the bank's prompt-end record nor its message endpoint serves it.  Here: a ~26k-token system
document, then TURNS user questions; request k carries the previous answer with a ~4k-token <details> tool-result
block in front of it, and request k+1 carries that answer plain.  Each prompt is ~30k+ tokens on Qwen.  Greedy, no
thinking, OpenAI /v1/chat/completions.

Per turn it prints one TSV row (turn, prompt, resumed-at, prefilled, ttft_s, source, route common, disk) -- the resume
point is the request's own `chat ctx=A..B ... prompt start` log line, the source and `disk` (the depth of the disk
chain the server found for the request) the log's resume decision (or, for a binary before L318, its adopt / disk-hit
lines), `common` the route's byte match against the bank that served the previous turn.

Checks (must-pass; --report-only prints the warm and restart ones as notes, for measuring a binary without the fix):
  * DEEPEST: every warm turn after the first resumes within one segment (--segment, the prefill chunk the disk
    chain's segments span) of `common`, and at or above the disk chain the server found for it (below it only when the
    server priced the read out: `recompute`).  Which tier served a warm turn is printed, not asserted: memory serves
    them whenever a bank still holds a record or a grid checkpoint at the shared prefix, and what it holds depends on
    whether the previous turns' terminal checkpoints ran (L380: they wait out an idle grace and are skipped for a
    conversation that leaves its replies behind -- before that they ran after every turn, thinned the grid checkpoint
    at the shared prefix away, and the disk tier served turn 3 as a side effect).
  * DISK (L380): the disk tier serves the conversation on the warm instance's RESTART (live_legs' hooks instance: the
    same disk store, every bank empty), where no bank can.  Its next turn resumes from the disk chain within one
    segment of the prefix every turn shares, and the re-send's request, sent again, resumes at or past the abandoned
    prefill's stored chain and answers with the warm instance's reply (rule 5: the lane ran, and what it restored is
    the state).
  * ABANDON: a request whose system prompt is rewritten at its start (nothing stored serves it) is abandoned by its
    client once half its prefill is done; the server logs the chain it stored for it, and the identical re-send resumes at or past
    that chain's end -- the abandoned prefix is not prefilled again.
  * REUSE == COLD: the last turn's and the re-send's greedy replies equal a FRESH server's (an empty disk tier, which
    must report 0 cached tokens) byte for byte.
  * PRICED (L327): the conversation's prefills run solo (one client), and they measure the prompt chunk cost fit; no
    restore-vs-recompute decision after that is `unpriced`; on the restart, which reads the fit back, at least one of
    its two disk hits is priced (load / recompute -- a server's first hit may be `unmeasured`, its second is not);
    and the fresh instance (same binary) reads the fit back from beside its verify curve too.

L317 (tests/live_legs.py): ResumeDeepestLeg has the Leg shape -- warm(srv) on the model's shared server, hooked(srv) on
the hooks instance (for the restart alone: it sets no hook and calls no test route), cold(srv) on the fresh instance
(fresh = True) -- and uses only srv.url / mark / since / post_json; its LEGS row is
("resume-deepest", "resume_deepest_live", "ResumeDeepestLeg", {}, ("qwen",)).  Alone (this command line) it runs the
same phases on three instances of its own (live_legs.standalone).  Exit 0 only on PASS."""
import json, os, random, re, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STORY = open(os.path.join(HERE, "long_context_story_prompt.txt"), encoding="utf-8", errors="replace").read()
QUESTIONS = [
    "Who is the main character of the reference document, and what do they want?",
    "Name three places the document describes and what happens at each.",
    "Which conflict drives the middle of the document? Answer in a few sentences.",
    "List the secondary characters you can find and one trait of each.",
    "How does the document end? Summarise the resolution briefly.",
    "What is the single most important object in the document, and why?",
    "Quote one sentence that captures the mood of the document and explain it.",
]


def sources(turn, n_words):
    """Open WebUI's injected tool / web-search result for a turn: deterministic, unlike any other turn's."""
    r = random.Random(318000 + turn)
    vocab = ["harbor", "signal", "ledger", "copper", "meridian", "lantern", "archive", "basalt", "cipher", "drift",
             "estuary", "foundry", "garrison", "hollow", "isotope", "juniper", "keystone", "lattice", "monsoon",
             "nickel", "orchard", "parapet", "quarry", "rampart", "sextant", "tundra", "umber", "vantage", "willow"]
    words = " ".join(r.choice(vocab) + str(r.randrange(100)) for _ in range(n_words))
    return ('<details type="tool_calls" done="true">\n<summary>Tool Executed: search_web</summary>\n'
            "Result %d: %s\n</details>\n" % (turn, words))


class ResumeDeepestLeg:
    NAME = "resume-deepest"
    fresh = True         # cold(srv) runs on live_legs' fresh instance
    test_routes = True   # hooked(srv) runs on live_legs' hooks instance with no hook of its own: the warm one's restart

    def __init__(self, g, work, turns=5, segment=4096, report_only=False):
        self.g, self.work, self.turns, self.segment, self.report_only = g, work, turns, segment, report_only
        self.rows, self.refs = [], []
        # what the restart phase sends, kept by the warm one: the conversation, the prefix every turn of it shares
        # (turn 1's prompt), the abandoned prefill's stored chain and the re-send (its messages, its reply)
        self.history, self.shared, self.stored, self.resend = None, 0, None, None
        os.makedirs(work, exist_ok=True)

    hook_env = staticmethod(lambda: None)

    # ---- requests ----
    def count_tokens(self, srv, text):
        st, d = srv.post_json("/v1/messages/count_tokens", {"model": "m", "messages": [{"role": "user", "content": text}]})
        if st != 200:
            raise RuntimeError("count_tokens: HTTP %s %s" % (st, d))
        return d["input_tokens"]

    def body(self, msgs):
        return {"model": "m", "max_tokens": 160, "temperature": 0, "enable_thinking": False, "messages": msgs}

    def chat(self, srv, msgs):
        st, d = srv.post_json("/v1/chat/completions", self.body(msgs), timeout=1800)
        if st != 200:
            raise RuntimeError("chat: HTTP %s %s" % (st, json.dumps(d)[:400]))
        return d

    @staticmethod
    def reply(d):
        return d["choices"][0]["message"].get("content") or ""

    def build(self, srv):
        """The system document (~26k tokens) and the per-turn source size (~4k tokens), sized in THIS tokenizer."""
        probe = STORY[:20000]
        per_char = self.count_tokens(srv, probe) / len(probe)
        n_doc = min(len(STORY), int(26000 / per_char))
        self.doc = STORY[:n_doc]
        self.system = ("You are a helpful assistant in a chat app. Answer questions about this reference document "
                       "briefly and precisely.\n\n" + self.doc)
        probe_src = sources(0, 1000)
        self.src_words = int(1000 * 4000 / max(1, self.count_tokens(srv, probe_src)))
        self.g.note("system document %d chars (~%d tokens), sources %d words (~4k tokens) per turn"
                    % (n_doc, int(n_doc * per_char), self.src_words))

    def request(self, history, k):
        """Request k (1-based): the stored history plain, except the previous answer, which carries its tool result."""
        msgs = [{"role": "system", "content": self.system}]
        for i, (q, a) in enumerate(history):
            rich = i == len(history) - 1
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": (sources(i + 1, self.src_words) if rich else "") + a}]
        msgs.append({"role": "user", "content": QUESTIONS[(k - 1) % len(QUESTIONS)]})
        return msgs

    # ---- the log ----
    def parse(self, text):
        start = re.findall(r"chat ctx=(\d+)\.\.(\d+):(\d+) .*prompt start", text)
        commons = re.findall(r"route: best bank \d+ common (\d+) resume (-?\d+)", text)
        pick = re.findall(r"resume: bank \d+ own -?\d+ adopt -?\d+(?: \(none\))? disk (-?\d+) of \d+ \(match -?\d+\) -> (\w+)",
                          text)
        if pick:
            source = pick[-1][1]
        elif "kv cache hit chain" in text:
            source = "disk"
        elif re.search(r"bank \d+ adopted bank", text):
            source = "adopt"
        else:
            source = "own"
        return (int(start[-1][0]) if start else -1, int(start[-1][2]) if start else -1,
                int(commons[-1][0]) if commons else -1, source, int(pick[-1][0]) if pick else -1)

    @staticmethod
    def verdicts(text):
        """(offset, verdict) of each restore-vs-recompute decision in `text`"""
        return [(d.start(), d.group(1)) for d in re.finditer(r"kv cache restore-vs-recompute: .*-> (\w+)", text)]

    def turn(self, srv, k, msgs):
        m = srv.mark()
        t = time.time()
        d = self.chat(srv, msgs)
        e2e = time.time() - t
        tm = d.get("timings", {})
        text = srv.since(m)
        at, prefilled, common, source, disk = self.parse(text)
        row = (k, tm.get("prompt_n", -1), at, prefilled, tm.get("ttft_s", e2e), source, common, disk)
        self.rows.append(row)
        print("turn\t%d\tprompt %d\tresumed %d\tprefilled %d\tttft %.3f\t%s\tcommon %d\tdisk %d" % row, flush=True)
        return d, row, [v for _, v in self.verdicts(text)]

    def check(self, cond, msg):
        if self.report_only:
            self.g.note(("(would PASS) " if cond else "(would FAIL) ") + msg)
        else:
            self.g.check(cond, msg)
        return bool(cond)

    def priced(self, srv, m0, disk_phase):
        """L327 PRICED: the restore-vs-recompute decisions this leg caused, against the server's prompt chunk cost fit.
        A single-client conversation prefills SOLO, and before L327 only fused rounds fed the fit, so every decision
        here said `unpriced` (L320 section 3).  Once the log says the solo terms are measured (or loaded from the
        cache), no decision may be unpriced.  disk_phase (the restart, whose turns only the disk tier serves): at least
        one must be priced -- load or recompute (the first disk hit of a server may be `unmeasured`: no load has
        measured the read yet).  The warm phase may decide nothing: a turn memory serves deeper prices no read."""
        text = srv.since(0)
        fit = re.search(r"prompt chunk cost (?:measured|loaded from the cache|moved): solo [\d.]+ ", text)
        fit_at = fit.start() if fit else None
        verdicts = [(at, v) for at, v in self.verdicts(text) if at >= m0]
        after = [v for at, v in verdicts if fit_at is not None and at > fit_at]
        self.g.note("restore decisions: %s; solo fit %s" % (",".join(v for _, v in verdicts) or "none",
                                                           "at byte %d of the log" % fit_at if fit else "never"))
        self.check(fit is not None, "the solo prefills measured the prompt chunk cost fit (%s)"
                   % (fit.group(0).strip() if fit else "no 'prompt chunk cost ... solo' line"))
        self.check("unpriced" not in after, "no restore decision after the solo fit was unpriced (%s)" % ",".join(after))
        if disk_phase:
            self.check(any(v in ("load", "recompute") for v in after),
                       "a restore decision was priced: load or recompute (%s)" % (",".join(after) or "none after the fit"))

    # ---- the phases ----
    def warm(self, srv):
        m0 = srv.mark()
        self.build(srv)
        history = []
        for k in range(1, self.turns + 1):
            msgs = self.request(history, k)
            d, row, verdicts = self.turn(srv, k, msgs)
            if k == self.turns:
                self.refs.append(("the last turn", msgs, self.reply(d)))
            history.append((msgs[-1]["content"], self.reply(d) or "."))
            if k == 1:
                self.shared = row[1]   # every later request carries turn 1's prompt whole: the system document, the
                                       # first question and the generation header
            else:
                _, prompt, at, prefilled, _, source, common, disk = row
                self.check(at >= 0 and common > 0 and at >= common - self.segment,
                           "turn %d resumed at %d, within one segment (%d) of common %d (prompt %d, prefilled %d, %s)"
                           % (k, at, self.segment, common, prompt, prefilled, source))
                # whichever tier served it, nothing the disk tier held was deeper -- unless the server priced that
                # read out (server_resume_pick: a chain loads when it is past memory's depth and not `recompute`)
                self.check(disk >= 0 and (at >= disk or "recompute" in verdicts),
                           "turn %d resumed at %d, at or above the disk chain the server found for it (%s; %s%s)"
                           % (k, at, "to %d" % disk if disk >= 0 else "no `resume:` line", source,
                              ", the read priced out: recompute" if "recompute" in verdicts else ""))
        later = self.rows[1:]
        if later:
            self.g.note("turns > 1: sources %s (the disk tier served %d of %d: memory serves a turn it holds deeper; "
                        "the restart phase is where the disk tier must); mean prefilled %.0f tokens, mean ttft %.3f s"
                        % (",".join(r[5] for r in later), sum(r[5] == "disk" for r in later), len(later),
                           sum(r[3] for r in later) / len(later), sum(r[4] for r in later) / len(later)))
        self.history = history
        self.abandon(srv, history)
        self.priced(srv, m0, disk_phase=False)

    def hooked(self, srv):
        """L380 DISK: the warm instance's restart -- its disk store, every bank empty -- so the disk chain is the only
        state that holds any of the conversation.  Turn A is its next turn (the request after the last warm one: the
        history rewritten inside the previous turn again): the chain the warm turns stored reaches the last segment
        end inside the prefix they all share.  Turn B is the re-send's request, sent again: its chain reaches past the
        abandoned prefill's.  A's read may be `unmeasured` (a server's first load is what measures the read rate); B's
        is priced, against A's load and the fit the instance read back at its start."""
        if not self.check(self.history is not None and self.resend is not None and self.shared > 0,
                          "the warm phase left its conversation and its re-send for the restart"):
            return
        m0 = srv.mark()
        k = len(self.history) + 1
        self.g.note("restart: the conversation's turn %d, then the re-send's request again" % k)
        _, row, _ = self.turn(srv, k, self.request(self.history, k))
        _, prompt, at, prefilled, _, source, _, disk = row
        self.check(source == "disk" and at >= self.shared - self.segment,
                   "on the restart the disk tier served the conversation's next turn: resumed at %d from %s (disk chain "
                   "to %d), within one segment (%d) of the %d tokens every turn shares (prompt %d, prefilled %d)"
                   % (at, source, disk, self.segment, self.shared, prompt, prefilled))
        msgs, warm = self.resend
        d, row, _ = self.turn(srv, k, msgs)
        self.check(self.stored is not None and row[5] == "disk" and row[2] >= self.stored[0],
                   "on the restart the disk tier served the re-send's request: resumed at %d from %s, at or past the "
                   "abandoned prefill's stored chain (%s) -- prefilled %d"
                   % (row[2], row[5], self.stored[0] if self.stored else "none", row[3]))
        same = self.reply(d) == warm
        self.check(same, "reuse == restore: the re-send's greedy reply off the disk chain equals the warm instance's%s"
                   % ("" if same else "\n  warm %r\n  disk %r" % (warm[:300], self.reply(d)[:300])))
        self.priced(srv, m0, disk_phase=True)

    def abandon(self, srv, history):
        """The conversation with its system prompt rewritten at the start (a new document edition): the whole prompt to
        prefill, sharing nothing the warm turns stored -- so the fresh server's reference is cold too.  Hang up once
        half of it is done, re-send the identical request."""
        msgs = self.request(history, len(history) + 1)
        msgs[0] = dict(msgs[0], content="Revised edition of the reference. " + self.system)
        body = json.dumps(dict(self.body(msgs), stream=True)).encode()
        m = srv.mark()
        host, port = srv.url.split("//")[1].split(":")
        s = socket.create_connection((host, int(port)))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body)
        t0, half = time.time(), False
        while time.time() - t0 < 600:
            prog = re.findall(r"prefill chunk (\d+)/(\d+)", srv.since(m))
            if prog and int(prog[-1][1]) > 0 and int(prog[-1][0]) * 2 >= int(prog[-1][1]):
                half = int(prog[-1][0]) < int(prog[-1][1])
                break
            time.sleep(0.05)
        s.close()
        t1 = time.time()
        stored = None
        while time.time() - t1 < 120 and stored is None:
            hit = re.findall(r"abandoned prefill of bank \d+: chain stored to (\d+) \(resumed at (\d+), prompt (\d+)\)",
                             srv.since(m))
            if hit:
                stored = tuple(int(x) for x in hit[-1])
            elif "prefill SSE error" in srv.since(m) and "abandoned prefill of bank" not in srv.since(m) and \
                    time.time() - t1 > 10:
                break
            time.sleep(0.2)
        ab_at = self.parse(srv.since(m))[0]
        self.check(half, "the long request was abandoned mid-prefill (resumed at %d)" % ab_at)
        self.check(stored is not None and stored[0] >= stored[1] + self.segment,
                   "the abandon stored its computed prefix: %s" %
                   ("chain to %d (the prefill resumed at %d, prompt %d)" % stored if stored else "no store logged"))
        d, row, _ = self.turn(srv, len(history) + 1, msgs)
        self.check(stored is not None and row[2] >= stored[0],
                   "the re-send resumed at %d, at or past the abandoned prefill's stored chain (%s) -- prefilled %d"
                   % (row[2], stored[0] if stored else "none", row[3]))
        self.refs.append(("the re-send", msgs, self.reply(d)))
        self.stored, self.resend = stored, (msgs, self.reply(d))

    def cold(self, srv):
        # L327: the fresh instance (same binary, same verify-curve cache) starts with the warm instance's fit
        self.check(re.search(r"prompt chunk cost loaded from the cache: solo [\d.]+ ", srv.since(0)) is not None,
                   "the fresh instance read the prompt chunk cost fit back from beside its verify curve")
        for what, msgs, warm in self.refs:
            d = self.chat(srv, msgs)
            cached = d.get("timings", {}).get("cached_n", -1)
            self.g.check(cached == 0, "the fresh server served %s cold (cached %d)" % (what, cached))
            same = self.reply(d) == warm
            self.g.check(same, "reuse == cold: %s's greedy reply equals a fresh server's%s"
                         % (what, "" if same else "\n  warm %r\n  cold %r" % (warm[:300], self.reply(d)[:300])))


def _args(ap):
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--segment", type=int, default=4096)
    ap.add_argument("--report-only", action="store_true", help="measure only: the warm and restart checks print as notes")


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(ResumeDeepestLeg, sys.argv[1:], ctx=262144, banks=16, port=8318, add_args=_args))

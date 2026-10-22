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

Per turn it prints one TSV row (turn, prompt, resumed-at, prefilled, ttft_s, source, route common) -- the resume point
is the request's own `chat ctx=A..B ... prompt start` log line, the source the log's resume decision (or, for a
binary before L318, its adopt / disk-hit lines), `common` the route's byte match against the bank that served the
previous turn.

Checks (must-pass; --report-only prints them and exits 0, for measuring a binary without the fix):
  * DEEPEST: every turn after the first resumes within one segment (--segment, the prefill chunk the disk chain's
    segments span) of `common`, and the disk tier served at least one of them (rule 5: the lane ran).
  * ABANDON: a request whose system prompt is rewritten at its start (nothing stored serves it) is abandoned by its
    client once half its prefill is done; the server logs the chain it stored for it, and the identical re-send resumes at or past
    that chain's end -- the abandoned prefix is not prefilled again.
  * REUSE == COLD: the last turn's and the re-send's greedy replies equal a FRESH server's (an empty disk tier, which
    must report 0 cached tokens) byte for byte.

L317 (tests/live_legs.py): ResumeDeepestLeg has the Leg shape -- warm(srv) on the model's shared server, cold(srv) on
the fresh instance (fresh = True) -- and uses only srv.url / mark / since / post_json; its LEGS row would be
("resume-deepest", "resume_deepest_live", "ResumeDeepestLeg", {}, ("qwen",)).  Alone (this command line) it runs on
two instances of its own.  Exit 0 only on PASS."""
import argparse, json, os, random, re, shutil, socket, subprocess, sys, time, urllib.error, urllib.request

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
    fresh = True   # cold(srv) runs on live_legs' fresh instance

    def __init__(self, g, work, turns=5, segment=4096, report_only=False):
        self.g, self.work, self.turns, self.segment, self.report_only = g, work, turns, segment, report_only
        self.rows, self.refs = [], []
        os.makedirs(work, exist_ok=True)

    hook_env = staticmethod(lambda: None)
    hooked = None

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
        pick = re.findall(r"resume: bank \d+ own -?\d+ adopt -?\d+(?: \(none\))? disk -?\d+ of \d+ \(match -?\d+\) -> (\w+)",
                          text)
        if pick:
            source = pick[-1]
        elif "kv cache hit chain" in text:
            source = "disk"
        elif re.search(r"bank \d+ adopted bank", text):
            source = "adopt"
        else:
            source = "own"
        return (int(start[-1][0]) if start else -1, int(start[-1][2]) if start else -1,
                int(commons[-1][0]) if commons else -1, source)

    def turn(self, srv, k, msgs):
        m = srv.mark()
        t = time.time()
        d = self.chat(srv, msgs)
        e2e = time.time() - t
        tm = d.get("timings", {})
        at, prefilled, common, source = self.parse(srv.since(m))
        row = (k, tm.get("prompt_n", -1), at, prefilled, tm.get("ttft_s", e2e), source, common)
        self.rows.append(row)
        print("turn\t%d\tprompt %d\tresumed %d\tprefilled %d\tttft %.3f\t%s\tcommon %d" % row, flush=True)
        return d, row

    def check(self, cond, msg):
        if self.report_only:
            self.g.note(("(would PASS) " if cond else "(would FAIL) ") + msg)
        else:
            self.g.check(cond, msg)

    # ---- the phases ----
    def warm(self, srv):
        self.build(srv)
        history = []
        for k in range(1, self.turns + 1):
            msgs = self.request(history, k)
            d, row = self.turn(srv, k, msgs)
            if k == self.turns:
                self.refs.append(("the last turn", msgs, self.reply(d)))
            history.append((msgs[-1]["content"], self.reply(d) or "."))
            if k > 1:
                _, prompt, at, prefilled, _, source, common = row
                self.check(at >= 0 and common > 0 and at >= common - self.segment,
                           "turn %d resumed at %d, within one segment (%d) of common %d (prompt %d, prefilled %d, %s)"
                           % (k, at, self.segment, common, prompt, prefilled, source))
        later = self.rows[1:]
        self.check(any(r[5] == "disk" for r in later),
                   "the disk tier served %d of %d later turns (sources: %s)"
                   % (sum(r[5] == "disk" for r in later), len(later), ",".join(r[5] for r in later)))
        if later:
            self.g.note("turns > 1: mean prefilled %.0f tokens, mean ttft %.3f s"
                        % (sum(r[3] for r in later) / len(later), sum(r[4] for r in later) / len(later)))
        self.abandon(srv, history)

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
        ab_at, _, _, _ = self.parse(srv.since(m))
        self.check(half, "the long request was abandoned mid-prefill (resumed at %d)" % ab_at)
        self.check(stored is not None and stored[0] >= stored[1] + self.segment,
                   "the abandon stored its computed prefix: %s" %
                   ("chain to %d (the prefill resumed at %d, prompt %d)" % stored if stored else "no store logged"))
        d, row = self.turn(srv, len(history) + 1, msgs)
        self.check(stored is not None and row[2] >= stored[0],
                   "the re-send resumed at %d, at or past the abandoned prefill's stored chain (%s) -- prefilled %d"
                   % (row[2], stored[0] if stored else "none", row[3]))
        self.refs.append(("the re-send", msgs, self.reply(d)))

    def cold(self, srv):
        for what, msgs, warm in self.refs:
            d = self.chat(srv, msgs)
            cached = d.get("timings", {}).get("cached_n", -1)
            self.g.check(cached == 0, "the fresh server served %s cold (cached %d)" % (what, cached))
            same = self.reply(d) == warm
            self.g.check(same, "reuse == cold: %s's greedy reply equals a fresh server's%s"
                         % (what, "" if same else "\n  warm %r\n  cold %r" % (warm[:300], self.reply(d)[:300])))


# ---- standalone: the leg on two instances of its own ----
class Grader:
    def __init__(self, leg):
        self.leg, self.ok, self.n = leg, True, 0

    def check(self, c, msg):
        self.ok &= bool(c)
        self.n += 1
        print("%s [%s] %s" % ("PASS" if c else "FAIL", self.leg, msg), flush=True)
        return bool(c)

    def note(self, msg):
        print("# [%s] %s" % (self.leg, msg), flush=True)


class Server:
    def __init__(self, a, name):
        self.url = "http://127.0.0.1:%d" % a.port
        self.dir = os.path.join(a.work, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        env = dict(os.environ, PULSAR_MSEQ_BANKS=str(a.banks))
        cmd = [a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port", str(a.port),
               "--kv-disk-dir", os.path.join(self.dir, "kv"), "--kv-disk-space-mb", "65536"]
        subprocess.run(["sync"], check=False)
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit("FAIL %s exited during startup (rc %s): %s" % (name, self.p.returncode, self.tail()))
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2).read()
                break
            except Exception:
                pass
            if time.time() - t0 > 900:
                self.stop()
                raise SystemExit("FAIL %s did not listen within 900 s" % name)
            time.sleep(2)
        print("# server %s up in %.0f s: %s, ctx %d, banks %d, disk KV on (pid %d)"
              % (name, time.time() - t0, a.model, a.ctx, a.banks, self.p.pid), flush=True)

    def text(self):
        self.log.flush()
        return open(self.log_path, "rb").read().decode("utf-8", "replace")

    def tail(self, n=3000):
        return self.text()[-n:]

    def mark(self):
        return len(self.text())

    def since(self, mark):
        return self.text()[mark:]

    def post_json(self, path, body, timeout=900):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(txt)
            except Exception:
                return e.code, {"error": txt}

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if not self.log.closed:
            self.log.close()
        shutil.rmtree(os.path.join(self.dir, "kv"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--work", default=os.path.join(os.environ.get("TMPDIR") or "/var/tmp", "pulsar-l318-%d" % os.getpid()))
    ap.add_argument("--port", type=int, default=8318)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--banks", type=int, default=16)
    ap.add_argument("--segment", type=int, default=4096)
    ap.add_argument("--report-only", action="store_true", help="measure only: print the checks, exit 0")
    ap.add_argument("--no-cold", action="store_true", help="skip the fresh-server comparison")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    g = Grader(ResumeDeepestLeg.NAME)
    leg = ResumeDeepestLeg(g, os.path.join(a.work, "leg"), turns=a.turns, segment=a.segment, report_only=a.report_only)
    for name, fn in (("warm", leg.warm), ("cold", None if a.no_cold else leg.cold)):
        if fn is None:
            continue
        srv = Server(a, name)
        try:
            fn(srv)
        except Exception as e:
            import traceback
            traceback.print_exc()
            g.check(False, "the %s phase raised %s: %s" % (name, type(e).__name__, e))
        finally:
            srv.stop()
        if not g.ok and not a.report_only:
            break
    print("ROWS\tturn\tprompt\tresumed\tprefilled\tttft_s\tsource\tcommon")
    for r in leg.rows:
        print("ROW\t%d\t%d\t%d\t%d\t%.3f\t%s\t%d" % r)
    ok = g.ok and g.n > 0
    print("resume-deepest-live: %s (%s)" % ("PASS" if ok else "FAIL", a.binary))
    return 0 if ok or a.report_only else 1


if __name__ == "__main__":
    sys.exit(main())

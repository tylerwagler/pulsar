#!/usr/bin/env python3
"""resume_fixes_live.py -- L355: what a request resumes from after a disconnect, a spill and a restart, on a real pulsar-server.

    python3 tests/resume_fixes_live.py MODEL --binary ./pulsar-server --work DIR [--port 8355] [--ctx 65536] [--banks 4]

A live leg (tests/live_legs.py runs it on Qwen's shared instances, L357): disconnect, spill's frame and boundary's first
half on the warm instance; boundary's second half (the warm instance's shutdown persist read on its restart, the hooks
instance) and the spill (the guard's budget lowered through the test route) on the hooks instance.  Alone it runs the
same phases on instances of its own.  Each case opens with a line of its own: the instances are shared.

Each case states the behaviour L353 fixed and reads it from the server's own log and the reply's usage:

  disconnect (no. 9)  A client hangs up mid-prefill and re-sends.  The bank keeps what it prefilled: the re-send resumes from
                      it, with no chain read.
  spill (nos. 8, 21)  A bank spilled under a low KV budget.  A request sharing only the start of its history doesn't read its
                      whole chain; a request continuing it does, to full depth straight after the spill.
  boundary (no. 15)   A prompt of exactly a grid multiple, persisted, re-sent on a fresh server.  A chain ending at the prompt's
                      end can't be resumed: the pick comes from a point below it (one on a shared system prompt), and the reply's cached count says so.

Exit 0 only when every check passed."""
import json
import os
import re
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Grader, Leg, cached_tokens  # noqa: E402
from preempt_live import probe  # noqa: E402

STORY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_context_story_prompt.txt"),
             encoding="utf-8", errors="replace").read()
GRID = 128   # the resume grid: a segment's restore point stands on it


def chat(srv, messages, max_tokens=1):
    st, d = srv.post_json("/v1/chat/completions", {"model": "m", "max_tokens": max_tokens, "temperature": 0,
                                                   "chat_template_kwargs": {"enable_thinking": False},
                                                   "messages": messages}, timeout=1800)
    if st != 200:
        raise RuntimeError("chat: HTTP %s %s" % (st, json.dumps(d)[:300]))
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def user(text):
    return [{"role": "user", "content": text}]


def resumes(log):
    """(bank, own, disk, of, pick) of each `resume:` line"""
    return [(int(a), int(b), int(c), int(d), e) for a, b, c, d, e in re.findall(
        r"resume: bank (\d+) own (\d+) adopt \d+(?: \(none\))? disk (\d+) of (\d+) \(match \d+\) -> (\w+)", log)]


def phase_prefilling(srv):
    return any(k.startswith("pulsar:slot_phase") and "prefill" in k and v == 1 for k, v in srv.metrics().items())


OP = "[resume-fixes %s]\n"   # each case's first line: no other leg, and no other case, sends it


def evicted(log):
    """the banks the guard spilled in `log`"""
    return [int(b) for b in re.findall(r"guard EVICTED bank (\d+)", log)]


class ResumeFixesLeg(Leg):
    NAME = "resume-fixes"
    fresh = False
    test_routes = True

    def __init__(self, g, work):
        Leg.__init__(self, g, work)
        self.frame = None          # spill: (budget, la, lb), framed on the warm instance
        self.boundary = None       # boundary: (system, text), the prompt the warm instance persisted at shutdown

    def warm(self, srv):
        print("=== disconnect ===", flush=True)
        self.disconnect(srv)
        print("=== spill: the frame ===", flush=True)
        self.spill_frame(srv)
        print("=== boundary: the persisted prompt ===", flush=True)
        self.boundary_persist(srv)

    def hooked(self, srv):
        print("=== boundary: the re-send after the restart ===", flush=True)
        self.boundary_resend(srv)
        print("=== spill ===", flush=True)
        self.spill(srv)

    # ---- disconnect (no. 9)
    def disconnect(self, srv):
        g = self.g
        text = OP % "disconnect" + STORY[:120000]
        # the same request as the re-send's, template options too: another render shares no prefix with it
        body = json.dumps({"model": "m", "max_tokens": 1, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}, "messages": user(text)}).encode()
        mark = srv.mark()
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " +
                  str(len(body)).encode() + b"\r\n\r\n" + body)
        t0 = time.time()
        while time.time() - t0 < 120 and not phase_prefilling(srv):
            time.sleep(0.05)
        if not g.check(phase_prefilling(srv), "the long prompt reached prefill"):
            s.close()
            return
        # past the first grid checkpoints, so there is something to keep: a hang-up before them keeps nothing, rightly
        t0 = time.time()
        while time.time() - t0 < 300 and "prefill chunk 8192/" not in srv.since(mark):
            time.sleep(0.05)
        if not g.check("prefill chunk 8192/" in srv.since(mark), "the prefill passed 8192 before the hang-up"):
            s.close()
            return
        s.close()
        t0 = time.time()
        while time.time() - t0 < 120 and "client disconnected during prefill" not in srv.since(mark):
            time.sleep(0.1)
        if not g.check("client disconnected during prefill" in srv.since(mark), "the server saw the hang-up in prefill"):
            return
        mark = srv.mark()
        _, usage = chat(srv, user(text))
        log = srv.since(mark)
        r = resumes(log)
        print("# re-send: %s, usage %s" % (r[-1:] or "no resume line", usage), flush=True)
        g.check(r and r[-1][1] > 0 and r[-1][4] == "own", "the re-send resumes from the bank's own prefix: %s" % (r[-1:],))
        g.check((cached_tokens(usage) or 0) > 0, "the reply reports what it reused: %s" % cached_tokens(usage))
        g.check("kv cache hit" not in log, "no chain was read for it")

    # ---- spill (nos. 8, 21)
    def texts(self, case):
        return OP % case + STORY[:48000], OP % case + STORY[60000:120000]

    def spill_frame(self, srv):
        pr = probe(srv, srv.ctx)
        if not self.g.check(pr is not None, "the server reports its guard budget and one bank's KV bytes"):
            return
        eager, per_tok = pr
        # counted on texts of the same length with another first line: the real A and B find nothing of theirs on disk
        a_text, b_text = self.texts("spill-frame")
        _, ua = chat(srv, user(a_text))
        _, ub = chat(srv, user(b_text))
        la, lb = ua["prompt_tokens"], ub["prompt_tokens"]
        self.frame = (int(eager + per_tok * (lb + 0.5 * la)), la, lb)   # B fits only with A out
        self.g.note("eager %.2f GiB, %.0f B a token, A %d, B %d: budget %.2f GiB" %
                    (eager / 2**30, per_tok, la, lb, self.frame[0] / 2**30))

    def spill(self, srv):
        g = self.g
        if not g.check(self.frame is not None, "the warm phase framed the spill"):
            return
        budget, la, lb = self.frame
        a_text, b_text = self.texts("spill")
        mark = srv.mark()
        srv.set_guard_budget(budget)
        g.check("KV budget OVERRIDDEN" in srv.since(mark), "the guard runs the lowered budget (test route)")
        mark = srv.mark()
        reply, _ = chat(srv, user(a_text), max_tokens=8)
        ra = resumes(srv.since(mark))
        if not g.check(bool(ra), "A said its bank: %s" % (ra[-1:],)):
            return
        bank_a = ra[-1][0]
        # the guard spills idle banks least recent first: on a shared instance an earlier leg's go before A's, so the
        # check is A's bank, by number
        mark = srv.mark()
        chat(srv, user(b_text))
        if not g.check(bank_a in evicted(srv.since(mark)), "B spilled A's bank %d (spilled %s; A %d tokens, B %d, budget "
                       "%.2f GiB)" % (bank_a, evicted(srv.since(mark)), la, lb, budget / 2**30)):
            return
        # no. 21: straight after the spill, the request that continues A restores it to depth
        mark = srv.mark()
        _, usage = chat(srv, user(a_text) + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Go on."}])
        log = srv.since(mark)
        print("# continuing A: %s, usage %s" % (resumes(log)[-1:], usage), flush=True)
        g.check((cached_tokens(usage) or 0) >= la - 2 * 4096,
                "the request continuing A resumed near its end: %s of %d" % (cached_tokens(usage), la))
        ra = resumes(log)
        bank_a = ra[-1][0] if ra else bank_a
        # no. 8: spill it again, and a request that parts from A early reads no chain of A's
        mark = srv.mark()
        chat(srv, user(b_text))
        if not g.check(bank_a in evicted(srv.since(mark)), "B spilled A's bank %d again (spilled %s)" %
                       (bank_a, evicted(srv.since(mark)))):
            return
        mark = srv.mark()
        _, usage = chat(srv, user(a_text[:2000 + len(OP % "spill")] + " A different story starts here. " +
                                  STORY[122000:128000]))
        log = srv.since(mark)
        hits = re.findall(r"kv cache hit.* chain of \d+ to (\d+)", log)
        print("# parting early from A: %s, chain reads to %s, usage %s" % (resumes(log)[-1:], hits, usage), flush=True)
        g.check(not any(int(h) > 4096 for h in hits), "no chain past the shared start was read: %s" % hits)

    # ---- boundary (no. 15)
    def boundary_persist(self, srv):
        g = self.g
        system, tail = OP % "boundary" + STORY[:30000], STORY[30000:34000]

        def msgs(sys_text, t):
            return [{"role": "system", "content": sys_text}, {"role": "user", "content": t}]

        def count(sys_text, t):
            st, d = srv.post_json("/v1/messages/count_tokens", {"model": "m", "system": sys_text, "messages": user(t)},
                                  timeout=60)
            if st != 200:
                raise RuntimeError("count_tokens: HTTP %s" % st)
            return d["input_tokens"]
        # count_tokens counts its own render, so the chat prompt's offset from it is measured first -- under another
        # system prompt, whose history serves nothing of the real one (alone, this was a restart)
        other = OP % "boundary-offset" + STORY[60000:90000]
        text = tail
        offset = chat(srv, msgs(other, text))[1]["prompt_tokens"] - count(other, text)
        # the first request leaves a restore point on the system prompt's grid, the stored point below the prompt's end
        chat(srv, msgs(system, "Name the first character."))
        mark = srv.mark()
        chat(srv, msgs(system, "Name the last character."))
        r = resumes(srv.since(mark))
        if not g.check(bool(r) and r[-1][2] > 0, "a stored point stands on the shared system prompt: %s" % (r[-1:],)):
            return
        for _ in range(400):
            if (count(system, text) + offset) % GRID == 0:
                break
            text = text[:text.rfind(" ")]
        _, usage = chat(srv, msgs(system, text))
        if not g.check(usage["prompt_tokens"] % GRID == 0, "a prompt of exactly a grid multiple: %d" % usage["prompt_tokens"]):
            return
        self.boundary = (system, text)   # the warm instance's shutdown persist closes the chain at the prompt's end

    def boundary_resend(self, srv):
        g = self.g
        if not g.check(self.boundary is not None, "the warm phase persisted the boundary prompt"):
            return
        system, text = self.boundary
        mark = srv.mark()
        _, usage = chat(srv, [{"role": "system", "content": system}, {"role": "user", "content": text}])
        r = resumes(srv.since(mark))
        print("# exact re-send on the restarted server: %s, usage %s" % (r[-1:], usage), flush=True)
        L = usage["prompt_tokens"]
        if not g.check(bool(r), "the server said what it resumed from"):
            return
        disk = r[-1][2]
        g.check(0 < disk < L, "the pick comes from a stored point below the prompt's end: %d of %d" % (disk, L))
        c = cached_tokens(usage) or 0
        g.check(c < L and (r[-1][4] != "disk" or c == disk),
                "the reply reports what really resumed: %d cached, %s at %d of %d" % (c, r[-1][4], disk, L))


def main():
    from live_legs import standalone
    return standalone(ResumeFixesLeg, sys.argv[1:], ctx=65536, banks=4, port=8355)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""resume_fixes_live.py -- L355: what a request resumes from after a disconnect, a spill and a restart, on a real pulsar-server.

    python3 tests/resume_fixes_live.py MODEL --binary ./pulsar-server --work DIR [--port 8355] [--ctx 65536] [--banks 4]

Each case states the behaviour L353 fixed and reads it from the server's own log and the reply's usage:

  disconnect (no. 9)  A client hangs up mid-prefill and re-sends.  The bank keeps what it prefilled: the re-send resumes from
                      it, with no chain read.
  spill (nos. 8, 21)  A bank spilled under a low KV budget.  A request sharing only the start of its history doesn't read its
                      whole chain; a request continuing it does, to full depth straight after the spill.
  boundary (no. 15)   A prompt of exactly a grid multiple, persisted, re-sent on a fresh server.  A chain ending at the prompt's
                      end can't be resumed: the pick comes from a point below it (one on a shared system prompt), and the reply's cached count says so.

Exit 0 only when every check passed."""
import argparse
import json
import os
import re
import shutil
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Grader, Server, cached_tokens  # noqa: E402
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


def case_disconnect(a, g, work):
    srv = Server(a.binary, a.model, work, "disconnect", a.port, a.ctx, banks=a.banks)
    try:
        text = STORY[:120000]
        # the same request as the re-send's, template options too: another render shares no prefix with it
        body = json.dumps({"model": "m", "max_tokens": 1, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}, "messages": user(text)}).encode()
        mark = srv.mark()
        s = socket.create_connection(("127.0.0.1", a.port), timeout=10)
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
    finally:
        srv.stop()


def case_spill(a, g, work):
    probe_srv = Server(a.binary, a.model, work, "spill-probe", a.port, a.ctx, banks=a.banks)
    try:
        pr = probe(probe_srv, a.ctx)
        if not g.check(pr is not None, "the server reports its guard budget and one bank's KV bytes"):
            return
        eager, per_tok = pr
        a_text, b_text = STORY[:48000], STORY[60000:120000]
        _, ua = chat(probe_srv, user(a_text))
        _, ub = chat(probe_srv, user(b_text))
        la, lb = ua["prompt_tokens"], ub["prompt_tokens"]
    finally:
        probe_srv.stop()
    budget = int(eager + per_tok * (lb + 0.5 * la))   # B fits only with A's bank out
    srv = Server(a.binary, a.model, work, "spill", a.port, a.ctx, banks=a.banks,
                 env={"PULSAR_SERVER_KV_BUDGET_OVERRIDE": str(budget)})
    try:
        reply, _ = chat(srv, user(a_text), max_tokens=8)
        mark = srv.mark()
        chat(srv, user(b_text))
        spilled = "guard EVICTED bank" in srv.since(mark)
        if not g.check(spilled, "B spilled A's bank (A %d tokens, B %d, budget %.2f GiB)" % (la, lb, budget / 2**30)):
            return
        # no. 21: straight after the spill, the request that continues A restores it to depth
        mark = srv.mark()
        _, usage = chat(srv, user(a_text) + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Go on."}])
        log = srv.since(mark)
        print("# continuing A: %s, usage %s" % (resumes(log)[-1:], usage), flush=True)
        g.check((cached_tokens(usage) or 0) >= la - 2 * 4096,
                "the request continuing A resumed near its end: %s of %d" % (cached_tokens(usage), la))
        # no. 8: spill it again, and a request that parts from A early reads no chain of A's
        mark = srv.mark()
        chat(srv, user(b_text))
        if not g.check("guard EVICTED bank" in srv.since(mark), "B spilled A's bank again"):
            return
        mark = srv.mark()
        _, usage = chat(srv, user(a_text[:2000] + " A different story starts here. " + STORY[122000:128000]))
        log = srv.since(mark)
        hits = re.findall(r"kv cache hit.* chain of \d+ to (\d+)", log)
        print("# parting early from A: %s, chain reads to %s, usage %s" % (resumes(log)[-1:], hits, usage), flush=True)
        g.check(not any(int(h) > 4096 for h in hits), "no chain past the shared start was read: %s" % hits)
    finally:
        srv.stop()


def case_boundary(a, g, work):
    kv = os.path.join(work, "boundary-kv")
    shutil.rmtree(kv, ignore_errors=True)
    extra = ("--kv-disk-dir", kv)
    system, tail = STORY[:30000], STORY[30000:34000]
    def msgs(t):
        return [{"role": "system", "content": system}, {"role": "user", "content": t}]
    srv = Server(a.binary, a.model, work, "boundary-1", a.port, a.ctx, banks=a.banks, extra=extra)
    try:
        # trim the text a word at a time to a prompt of exactly a grid multiple.  count_tokens counts its own render, so the
        # chat prompt's offset from it is measured first
        def count(t):
            st, d = srv.post_json("/v1/messages/count_tokens", {"model": "m", "system": system, "messages": user(t)}, timeout=60)
            if st != 200:
                raise RuntimeError("count_tokens: HTTP %s" % st)
            return d["input_tokens"]
        text = tail
        offset = chat(srv, msgs(text))[1]["prompt_tokens"] - count(text)
        srv.stop()   # that request's history would serve the real one: a clean store for it
        shutil.rmtree(kv, ignore_errors=True)
        srv = Server(a.binary, a.model, work, "boundary-1", a.port, a.ctx, banks=a.banks, extra=extra)
        # the first request leaves a restore point on the system prompt's grid, the stored point below the prompt's end
        chat(srv, msgs("Name the first character."))
        mark = srv.mark()
        chat(srv, msgs("Name the last character."))
        r = resumes(srv.since(mark))
        if not g.check(bool(r) and r[-1][2] > 0, "a stored point stands on the shared system prompt: %s" % (r[-1:],)):
            return
        for _ in range(400):
            if (count(text) + offset) % GRID == 0:
                break
            text = text[:text.rfind(" ")]
        _, usage = chat(srv, msgs(text))
        if not g.check(usage["prompt_tokens"] % GRID == 0, "a prompt of exactly a grid multiple: %d" % usage["prompt_tokens"]):
            return
    finally:
        srv.stop()   # the shutdown persist closes the chain at the prompt's end
    srv = Server(a.binary, a.model, work, "boundary-2", a.port, a.ctx, banks=a.banks, extra=extra)
    try:
        mark = srv.mark()
        _, usage = chat(srv, msgs(text))
        r = resumes(srv.since(mark))
        print("# exact re-send on a fresh server: %s, usage %s" % (r[-1:], usage), flush=True)
        L = usage["prompt_tokens"]
        if not g.check(bool(r), "the server said what it resumed from"):
            return
        disk = r[-1][2]
        g.check(0 < disk < L, "the pick comes from a stored point below the prompt's end: %d of %d" % (disk, L))
        c = cached_tokens(usage) or 0
        g.check(c < L and (r[-1][4] != "disk" or c == disk),
                "the reply reports what really resumed: %d cached, %s at %d of %d" % (c, r[-1][4], disk, L))
    finally:
        srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--work", default=os.path.join(os.environ.get("TMPDIR") or "/var/tmp", "pulsar-resume-fixes-%d" % os.getpid()))
    ap.add_argument("--port", type=int, default=8355)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--banks", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    g = Grader("resume-fixes")
    for case in (case_disconnect, case_spill, case_boundary):
        print("=== %s ===" % case.__name__, flush=True)
        case(a, g, a.work)
    print("resume-fixes-live: %s" % ("PASS" if g.ok else "FAIL"))
    return 0 if g.ok else 1


if __name__ == "__main__":
    sys.exit(main())

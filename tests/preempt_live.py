#!/usr/bin/env python3
"""preempt_live.py -- L314: a running decoder preempted through its terminal checkpoint and readmitted, on a real
pulsar-server.

    python3 tests/preempt_live.py MODEL exact    [--binary ./pulsar-server] [--work DIR] [--port 8314] [--margin 1.5]
    python3 tests/preempt_live.py MODEL pressure [--ctx 1048576 --banks 16 --n 12 --prompt 48000 --gen 128 --fit 3]

exact     The guard's eager floor and one bank's KV bytes a token are read from a server of this --ctx / --banks
          (/metrics kv_eager_floor_bytes, L326; kv_bank_bytes over the context), and the prompts are counted on it; a
          server whose KV budget is overridden (PULSAR_SERVER_KV_BUDGET_OVERRIDE, the guard's test hook) so request A
          (a long prompt decoding 1500 greedy tokens) fits alone, B (a longer prompt, 1 token) fits alone, and both do
          not.
          B arrives while A decodes: A must be PREEMPTED (the server's log), B answered, A READMITTED and finished --
          and A's text must equal A's on a fresh server, uninterrupted, byte for byte (a single stream on both sides:
          greedy decode is deterministic there).  Every request asks for logprobs, which keeps it on the plain lane
          (one M-independent row a step): a reloaded bank's speculative rounds start without the pendings it had, and
          past 16 verify rows a round's numerics are graded, not byte-equal (ENGINEERING-RULES 7), so on the spec
          lanes the instrument is the resume point, not the text.
pressure  N concurrent requests of L-token prompts decoding G tokens each on a --ctx / --banks server with its real
          budget; reports the 503s, the admission refusals by reason (/metrics), preemptions and readmissions, and
          the wall time -- run once per binary to compare.  A measurement, not a battery leg.

L317: `exact` is a leg of tests/live_legs.py: the probe and the counts on the model's shared server (warm), the
preemption on its hooks instance (the budget override), A uninterrupted on its fresh instance (0 cached tokens
asserted).  The servers run with their disk-KV dirs under the work dir (never /tmp).  Exit 0 only when every check
passed."""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, Server, check_fresh  # noqa: E402

STORY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_context_story_prompt.txt"),
             encoding="utf-8", errors="replace").read()
A_TEXT, B_TEXT, W_TEXT = STORY[:24000], STORY[30000:66000], STORY[70000:78000]
GEN_A = 1500


def chat(srv, text, max_tokens, logprobs=True):
    """(status, reasoning + text, usage) of one greedy request continuing `text`"""
    st, d = srv.post_json("/v1/chat/completions", {
        "model": "m", "max_tokens": max_tokens, "temperature": 0, "logprobs": logprobs,
        "messages": [{"role": "user", "content": text + "\n\nContinue this story in your own words."}]}, timeout=3600)
    if st != 200:
        return st, str(d), {}
    m = d["choices"][0]["message"]
    return 200, (m.get("reasoning_content") or "") + "\n--\n" + (m.get("content") or ""), d.get("usage", {})


def guard(srv):
    """(touched budget, eager floor) in bytes as the server holds them (/metrics, L326), or None with the guard off.
    The touched budget is min(kv budget - eager, touched + MemAvailable - floor post-warmup) (cli_main.cpp, L319)."""
    m = srv.metrics()
    t, e = m.get("pulsar:kv_touched_budget_bytes", 0.0), m.get("pulsar:kv_eager_floor_bytes", 0.0)
    return (t, e) if t > 0 and e > 0 else None


def probe(srv, ctx):
    """(the guard's eager floor, one bank's KV bytes a token) of this server's --ctx / --banks"""
    gl = guard(srv)
    bank_bytes = srv.metrics().get("pulsar:kv_bank_bytes", 0.0)
    if not gl or bank_bytes <= 0:
        return None
    return gl[1], bank_bytes / ctx


def slot_decoding(srv):
    return any(k.startswith("pulsar:slot_phase") and 'phase="decode"' in k and v == 1 for k, v in srv.metrics().items())


class PreemptLeg(Leg):
    NAME = "preempt-exact"
    fresh = True

    def __init__(self, g, work, margin=1.5, tag=None):   # tag: the old command line's, kept for its callers
        Leg.__init__(self, g, work)
        self.margin, self.budget, self.res = margin, None, {}

    def warm(self, srv):
        pr = probe(srv, srv.ctx)
        if not self.g.check(pr is not None, "the shared server reports its guard budget and one bank's KV bytes"):
            return
        eager, per_tok = pr
        # the prompts counted on the shared server: the budget holds B alone (and A alone: A's prompt + its generation
        # is below B's prompt) with a margin, never both.  Bank 0 is never preempted (the guard keeps it resident): an
        # unrelated conversation W takes it first, so A lands on a bank of its own and decodes alone.
        lens = {}
        for k, t in (("B", B_TEXT), ("A", A_TEXT), ("W", W_TEXT)):
            st, _, u = chat(srv, t, 1)
            lens[k] = u.get("prompt_tokens", 0) if st == 200 else 0
        LA, LB, LW = lens["A"] + GEN_A, lens["B"], lens["W"]
        if not self.g.check(0 < LA < LB and LW > 0, "the counted prompts (A + generation %d, B %d, W %d) frame a "
                            "preemption" % (LA, LB, LW)):
            return
        self.budget = int(eager + per_tok * (LW + self.margin * LB))
        self.g.note("eager %.2f GiB, %.0f B a token: budget override %.2f GiB" % (eager / 2**30, per_tok,
                                                                                    self.budget / 2**30))

    def hook_env(self):
        return {"PULSAR_SERVER_KV_BUDGET_OVERRIDE": str(self.budget)} if self.budget else None

    def hooked(self, srv):
        over = srv.grep("KV budget OVERRIDDEN")
        self.g.check(over, "the hooks instance runs the overridden budget: %s" % (over[0][-80:] if over else "no line"))
        # the override sizes the ledger's share (kv budget - eager); the ceiling is the smaller of it and the measured
        # headroom (L319), so the frame holds only when the ledger's share binds: touched budget == override - eager
        gl = guard(srv)
        share = self.budget - gl[1] if gl else 0
        self.g.check(gl is not None and abs(gl[0] - share) < 2**20,
                     "the overridden ledger share binds the touched budget: %s" %
                     ("%.2f GiB (ledger share %.2f - eager %.2f = %.2f)" %
                      (gl[0] / 2**30, self.budget / 2**30, gl[1] / 2**30, share / 2**30) if gl else "guard off"))
        mark = srv.mark()
        chat(srv, W_TEXT, 1)
        th = threading.Thread(target=lambda: self.res.__setitem__("A", chat(srv, A_TEXT, GEN_A)))
        th.start()
        t0 = time.time()   # A decoding: a slot's phase reads decode on /metrics
        while time.time() - t0 < 600 and not slot_decoding(srv):
            time.sleep(0.25)
        time.sleep(2.0)
        self.res["B"] = chat(srv, B_TEXT, 1)
        th.join()
        log = srv.since(mark)
        mt = srv.metrics()
        self.g.check(self.res["B"][0] == 200, "B (the prompt that did not fit) answered: %d" % self.res["B"][0])
        self.g.check("PREEMPTED" in log and "READMITTED" in log,
                     "A preempted and readmitted (preemptions %d, readmissions %d)" %
                     (mt.get("pulsar:decoder_preemptions_total", -1), mt.get("pulsar:decoder_readmissions_total", -1)))
        for l in [l for l in log.splitlines() if "PREEMPTED" in l or "READMITTED" in l][:4]:
            print("#   " + l[-220:])

    def cold(self, srv):
        if "A" not in self.res:
            self.g.check(False, "no preempted A to compare (the hooks phase did not run)")
            return
        st, alone, u = chat(srv, A_TEXT, GEN_A)
        check_fresh(self.g, "A uninterrupted", u)
        got = self.res["A"]
        self.g.check(got[0] == 200 and st == 200 and got[1] == alone,
                     "A's %d chars after its preemption == A uninterrupted on a fresh server" % len(got[1]))
        if got[1] != alone:
            print("  preempted %r\n  alone     %r" % (got[1][:300], alone[:300]))


def pressure(a):
    """N concurrent requests on a --ctx / --banks server, the budget overridden to `fit` requests (0: the real one)."""
    ok = True
    env = {}
    if a.fit > 0:
        p = Server(a.binary, a.model, a.work, "pressure-probe", a.port, a.ctx, a.banks)
        try:
            pr = probe(p, a.ctx)
        finally:
            p.stop()
        if not pr:
            print("FAIL the probe server reported no guard budget / bank bytes")
            return 1
        eager, per_tok = pr
        budget = int(eager + per_tok * (a.prompt + a.gen) * a.fit)
        env = {"PULSAR_SERVER_KV_BUDGET_OVERRIDE": str(budget)}
        print("# eager %.2f GiB, %.0f B a token: budget override %.2f GiB (%.1f requests)" %
              (eager / 2**30, per_tok, budget / 2**30, a.fit))
    srv = Server(a.binary, a.model, a.work, "pressure", a.port, a.ctx, a.banks, env)
    codes = []
    t0 = time.time()
    try:
        chars = int(a.prompt * 3.6)

        def one(i):
            off = (i * 7919) % max(1, len(STORY) - 1000)
            text = (STORY[off:] + STORY)[:chars] if chars < len(STORY) else (STORY * (chars // len(STORY) + 1))[:chars]
            codes.append(chat(srv, "Request %d.\n" % i + text, a.gen, logprobs=False)[0])
        ths = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
        for t in ths:
            t.start()
            time.sleep(0.5)
        for t in ths:
            t.join()
        mt = srv.metrics()
    finally:
        srv.stop()
    refusals = {k: v for k, v in mt.items() if k.startswith("pulsar:admission_refusals_total") and v}
    print("# %s: %d requests of ~%d tokens x %d generated on %d banks at ctx %d in %.0f s" %
          (a.tag, a.n, a.prompt, a.gen, a.banks, a.ctx, time.time() - t0))
    print("# %s: 200s %d, 503s %d, other %s" % (a.tag, codes.count(200), codes.count(503),
                                                [c for c in codes if c not in (200, 503)]))
    print("# %s: refusals %s" % (a.tag, refusals))
    print("# %s: preemptions %d, readmissions %d, spills %d" % (a.tag, mt.get("pulsar:decoder_preemptions_total", 0),
                                                              mt.get("pulsar:decoder_readmissions_total", 0),
                                                              mt.get("pulsar:bank_spills_total", 0)))
    ok = all(c in (200, 503) for c in codes)
    print(("PASS" if ok else "FAIL") + " every request answered 200 or refused 503")
    print("preempt-live %s pressure: %s" % (a.tag, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "pressure":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("model")
        ap.add_argument("mode")
        ap.add_argument("--binary", default="./pulsar-server")
        ap.add_argument("--work", required=True)
        ap.add_argument("--port", type=int, default=8314)
        ap.add_argument("--ctx", type=int, default=65536)
        ap.add_argument("--banks", type=int, default=4)
        ap.add_argument("--n", type=int, default=12)
        ap.add_argument("--prompt", type=int, default=48000, help="approximate prompt tokens a request")
        ap.add_argument("--gen", type=int, default=128)
        ap.add_argument("--fit", type=float, default=3.0, help="the KV budget overridden to this many requests (0: real)")
        ap.add_argument("--tag", default="run")
        sys.exit(pressure(ap.parse_args()))
    argv = [x for x in sys.argv[1:] if x != "exact"]
    from live_legs import standalone
    sys.exit(standalone(PreemptLeg, argv, ctx=65536, banks=4, port=8314,
                        add_args=lambda ap: (ap.add_argument("--margin", type=float, default=1.5),
                                             ap.add_argument("--tag", default="run"))))

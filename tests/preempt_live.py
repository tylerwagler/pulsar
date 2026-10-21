#!/usr/bin/env python3
"""preempt_live.py -- L314: a running decoder preempted through its terminal checkpoint and readmitted, on a real
pulsar-server.

    python3 tests/preempt_live.py MODEL --binary ./pulsar-server --work DIR exact    [--port 8314]
    python3 tests/preempt_live.py MODEL --binary ./pulsar-server --work DIR pressure [--ctx 1048576 --banks 16 ...]

exact     A probe server reads the guard's eager floor and one bank's KV bytes a token (/metrics kv_bank_bytes over
          the context); the measured server's KV budget is then overridden (PULSAR_SERVER_KV_BUDGET_OVERRIDE, the
          guard's test hook) so request A (a long prompt decoding 1500 greedy tokens) fits alone, B (a longer prompt, 1
          token) fits alone, and both do not.  B arrives while A decodes: A must be PREEMPTED (the server's log),
          B answered, A READMITTED and finished -- and A's text must equal A's on a fresh server, uninterrupted, byte
          for byte (a single stream on both sides: greedy decode is deterministic there).
pressure  N concurrent requests of L-token prompts decoding G tokens each on a --ctx / --banks server with its real
          budget; reports the 503s, the admission refusals by reason (/metrics), preemptions and readmissions, and
          the wall time -- run once per binary to compare.

The server runs with its own disk-KV dir under DIR (never /tmp).  Exit 0 only when every check passed."""
import argparse, json, os, re, shutil, subprocess, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("mode", choices=["exact", "pressure"])
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--work", required=True)
ap.add_argument("--port", type=int, default=8314)
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--banks", type=int, default=4)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--prompt", type=int, default=48000, help="pressure: approximate prompt tokens a request")
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--tag", default="run")
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port
os.makedirs(a.work, exist_ok=True)
STORY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_context_story_prompt.txt"),
             encoding="utf-8", errors="replace").read()


class Server:
    def __init__(self, name, env_extra=None):
        self.dir = os.path.join(a.work, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        env = dict(os.environ, PULSAR_MSEQ_BANKS=str(a.banks), **(env_extra or {}))
        self.p = subprocess.Popen([a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port",
                                   str(a.port), "--kv-disk-dir", os.path.join(self.dir, "kv")],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit("FAIL server exited during startup: " + self.text()[-3000:])
            try:
                urllib.request.urlopen(B + "/health", timeout=2).read()
                break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise SystemExit("FAIL server did not listen")
            time.sleep(2)
        print("# %s up in %.0f s" % (name, time.time() - t0), flush=True)

    def text(self):
        self.log.flush()
        return open(self.log_path, "rb").read().decode("utf-8", "replace")

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def chat(text, max_tokens):
    body = {"model": "m", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": text + "\n\nContinue this story in your own words."}]}
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode())
        return 200, d["choices"][0]["message"].get("content") or ""
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def metrics():
    out = {}
    for line in urllib.request.urlopen(B + "/metrics", timeout=10).read().decode().splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


ok = True


def check(c, msg):
    global ok
    ok &= bool(c)
    print(("PASS " if c else "FAIL ") + msg, flush=True)


if a.mode == "exact":
    A_TEXT, B_TEXT = STORY[:24000], STORY[30000:66000]
    probe = Server(a.tag + "-probe")
    try:
        m = re.search(r"touched budget ([0-9.]+) GiB \(kv budget ([0-9.]+) . eager ([0-9.]+) GiB\)", probe.text())
        bank_bytes = metrics().get("pulsar:kv_bank_bytes", 0.0)
    finally:
        probe.stop()
    if not m or bank_bytes <= 0:
        print("FAIL the probe server reported no guard budget / bank bytes")
        sys.exit(1)
    eager = float(m.group(3)) * 2**30
    per_tok = bank_bytes / a.ctx
    GEN_A = 1500
    LA, LB = 6000 + GEN_A, 9000   # A's prompt + generation, B's prompt (approximate tokens)
    budget = int(eager + per_tok * (LA + 0.7 * LB))
    print("# eager %.2f GiB, %.0f B a token: budget override %.2f GiB" % (eager / 2**30, per_tok, budget / 2**30))
    srv = Server(a.tag + "-preempt", {"PULSAR_SERVER_KV_BUDGET_OVERRIDE": str(budget)})
    res = {}
    try:
        th = threading.Thread(target=lambda: res.__setitem__("A", chat(A_TEXT, GEN_A)))
        th.start()
        t0 = time.time()   # A decoding: a slot's phase reads decode on /metrics
        while time.time() - t0 < 600 and not any(k.startswith("pulsar:slot_phase") and 'phase="decode"' in k and v == 1
                                                 for k, v in metrics().items()):
            time.sleep(0.25)
        time.sleep(2.0)
        res["B"] = chat(B_TEXT, 1)
        th.join()
        log = srv.text()
        mt = metrics()
    finally:
        srv.stop()
    check(res["B"][0] == 200, "B (the prompt that did not fit) answered: %d" % res["B"][0])
    check("PREEMPTED" in log and "READMITTED" in log,
          "A preempted and readmitted (preemptions %d, readmissions %d)" %
          (mt.get("pulsar:decoder_preemptions_total", -1), mt.get("pulsar:decoder_readmissions_total", -1)))
    for l in [l for l in log.splitlines() if "PREEMPTED" in l or "READMITTED" in l][:4]:
        print("#   " + l[-220:])
    cold = Server(a.tag + "-alone")
    try:
        alone = chat(A_TEXT, GEN_A)
    finally:
        cold.stop()
    check(res["A"][0] == 200 and res["A"] == alone,
          "A's %d chars after its preemption == A uninterrupted on a fresh server" % len(res["A"][1]))
    if res["A"] != alone:
        print("  preempted %r\n  alone     %r" % (res["A"][1][:300], alone[1][:300]))
else:
    srv = Server(a.tag + "-pressure")
    codes = []
    t0 = time.time()
    try:
        chars = int(a.prompt * 3.6)
        def one(i):
            off = (i * 7919) % max(1, len(STORY) - 1000)
            text = (STORY[off:] + STORY)[:chars] if chars < len(STORY) else (STORY * (chars // len(STORY) + 1))[:chars]
            codes.append(chat("Request %d.\n" % i + text, a.gen)[0])
        ths = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
        for t in ths:
            t.start()
            time.sleep(0.5)
        for t in ths:
            t.join()
        mt = metrics()
    finally:
        srv.stop()
    refusals = {k: v for k, v in mt.items() if k.startswith("pulsar:admission_refusals_total") and v}
    print("# %s: %d requests of ~%d tokens x %d generated on %d banks at ctx %d in %.0f s" %
          (a.tag, a.n, a.prompt, a.gen, a.banks, a.ctx, time.time() - t0))
    print("# %s: 200s %d, 503s %d, other %s" % (a.tag, codes.count(200), codes.count(503),
                                                [c for c in codes if c not in (200, 503)]))
    print("# %s: refusals %s" % (a.tag, json.dumps(refusals)))
    print("# %s: preemptions %d, readmissions %d, spills %d" % (a.tag, mt.get("pulsar:decoder_preemptions_total", 0),
                                                              mt.get("pulsar:decoder_readmissions_total", 0),
                                                              mt.get("pulsar:bank_spills_total", 0)))
    check(all(c in (200, 503) for c in codes), "every request answered 200 or refused 503")
print("preempt-live %s %s: %s" % (a.tag, a.mode, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

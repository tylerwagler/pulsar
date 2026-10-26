#!/usr/bin/env python3
"""server_kv_admission_live.py -- L280: a prompt's KV is admitted against the 2b guard's budget BEFORE it lands.

    python3 tests/server_kv_admission_live.py MODEL [--binary ./pulsar-server] [--port 8197] [--ctx 32768]
                                              [--tokens 6000] [--log PATH]

The defect (L280 window C, --ctx 1M): the guard watched decode quanta only, so a third 1M chain loaded into a
fresh bank with the resident KV already past the budget and no idle bank to spill; the box ran out of memory and
earlyoom killed the server mid-request (the client saw the connection close without a response).  This drives a
real server on a small model at a small --ctx, with the guard's budget lowered through its test hook
(PULSAR_SERVER_KV_BUDGET_OVERRIDE) to 2.2 conversations of --tokens tokens, so three cannot all stay resident:

  phase 0  a probe server: the eager floor (/metrics pulsar:kv_eager_floor_bytes, L326), KV bytes per token and the
           tokenizer's tokens per word
  S  SPILL   chat A, chat B (resident, within the budget), then chat C on a fresh bank: the server spills an idle
             bank BEFORE C binds ("kv admission: ... spilling 1 idle bank(s) first"); every answer 200
  W  PREEMPT (a fresh server) A and B as two long streamed completions, so two banks are busy, then C: no idle
             bank to spill.  Since L314 the server does not queue C behind a busy bank: the NEWEST decoder (B) is
             PREEMPTED through its terminal checkpoint, C answers 200 while B is parked, and B is READMITTED and
             streams all its tokens (L326: this phase was L280's WAIT and was not rerun after L314 changed the rule;
             the WAIT arm is now the one for a decoder that cannot be preempted)
  R  REFUSE  a prompt whose KV alone is past the budget: 503 with the numbers, at once; the server keeps serving
  P  REUSE   chat B asked again (spilled to disk in S) answers exactly as B did cold -- the prefix cache unchanged
  never      the decode guard's "NO idle victim -- back-pressure" (proceeding past its budget), in any phase; a
             violation names the phase and prints the line with the admission / preempt / guard lines before it.
             KNOWN_L328: today W reproduces it (gpuq #472: READMITTED 0.0 s after PREEMPTED, then the line), so it
             is a STRICT expected failure: reported as XFAIL (L328), and a run that does NOT reproduce it FAILS
             ("XPASS: ... drop KNOWN_L328") -- the fix in L328 drops the marker in the same commit

Family-neutral: the bank price and the spill are the engine's bank ops.  Exit 0 only when every check passed.
Scratch (the disk-KV dirs) goes under $TMPDIR; --log keeps every guarded server's log (one file, by phase)."""
import argparse, json, os, random, re, shutil, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8197)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--tokens", type=int, default=6000, help="one conversation's prompt (tokens)")
ap.add_argument("--log", default=None, help="keep the guarded servers' logs here")
a = ap.parse_args()
URL = "http://127.0.0.1:%d" % a.port
fails = 0
logs = []   # (phase, the server's log) per stopped server
tags = {}   # log path -> phase


def check(ok, what):
    global fails
    fails += not ok
    print("  %s %s" % ("ok  " if ok else "FAIL", what), flush=True)


def start(env_extra, tag):
    kv = tempfile.mkdtemp(prefix="kvadm-")
    log = os.path.join(kv, "server.log")
    env = dict(os.environ, PULSAR_MSEQ_BANKS="4", **env_extra)
    f = open(log, "w")
    p = subprocess.Popen([a.binary, "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port), "--ctx", str(a.ctx),
                          "--kv-disk-dir", kv, "--kv-disk-space-mb", "4096"], stdout=f, stderr=subprocess.STDOUT,
                         env=env)
    tags[log] = tag
    for _ in range(900):
        if "listening on" in open(log).read():
            return p, kv, log
        if p.poll() is not None:
            break
        time.sleep(1)
    print(open(log).read()[-3000:])
    sys.exit("server %s did not start" % tag)


def stop(p, kv, log):
    p.terminate()
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill()
    logs.append((tags.get(log, "?"), open(log).read()))
    shutil.rmtree(kv, ignore_errors=True)


# the scheduler's KV decisions, for a failure's context
KV_LINE = re.compile(r"kv admission|PREEMPTED|READMITTED|preemption|guard|spill", re.I)


def kv_lines(text, upto=None, n=14):
    """the last `n` KV-decision lines of a log (before line index `upto`)"""
    lines = text.splitlines()[:upto]
    return [l for l in lines if KV_LINE.search(l)][-n:]


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line and not line.startswith("#"):
                k, _, v = line.rpartition(" ")
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, r.read().decode(errors="replace")   # a small model may emit broken UTF-8
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def chat(text):
    """greedy, 16 tokens, with each token's logprob: (code, the answer and its per-token logprobs, usage)"""
    code, raw = post("/v1/chat/completions", {"model": "m", "temperature": 0, "max_tokens": 16, "logprobs": True,
                                             "top_logprobs": 1, "messages": [
        {"role": "user", "content": text + "\n\nSay only the last number above."}]})
    if code != 200:
        return code, raw, None
    out = json.loads(raw)
    ch = out["choices"][0]
    lp = [(c.get("token"), c.get("logprob")) for c in ((ch.get("logprobs") or {}).get("content") or [])]
    return code, json.dumps([ch["message"].get("content", ""), lp]), out.get("usage", {})


def long_stream(text, first, result, key):
    """a streamed raw completion of 300 tokens: `first` is set when its first token arrives"""
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(
        {"model": "m", "prompt": text, "max_tokens": 300, "temperature": 0, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if line.startswith(b"data: {"):
                    n += 1
                    first.set()
            result[key] = (r.status, n, time.time())
    except urllib.error.HTTPError as e:
        first.set()
        result[key] = (e.code, n, time.time())


def words(seed, n):
    r = random.Random(seed)
    return " ".join("%s%d" % (r.choice(("alpha", "bravo", "cedar", "delta", "ember", "fjord")), r.randrange(10**6))
                    for _ in range(n))


# ---- phase 0: the eager floor, the KV price, the tokens per word
p, kv, log = start({}, "probe")
try:
    m = metrics()
    # the server's own eager floor (L326): ledger - touched budget is only that while the ledger's share binds; the
    # touched budget is min(ledger - eager, touched + MemAvailable - floor) since L319
    eager = m["pulsar:kv_eager_floor_bytes"]
    check(eager > 0, "the probe server reports its eager floor (%.2f GiB)" % (eager / 2**30))
    per_token = m["pulsar:kv_bank_bytes"] / a.ctx
    code, _, usage = chat(words(1, 400))
    tpw = usage["prompt_tokens"] / 400.0 if usage else 3.0
finally:
    stop(p, kv, log)
logs.clear()
T = a.tokens
nw = lambda tok: int(tok / tpw)
budget_touched = int(2.2 * T * per_token)
env = {"PULSAR_SERVER_KV_BUDGET_OVERRIDE": str(int(eager + budget_touched))}
print("phase 0: eager %.2f GiB, %.0f B of KV per token, %.2f tokens per word; touched budget %.1f MiB = 2.2 x %d "
      "tokens" % (eager / 2**30, per_token, tpw, budget_touched / 2**20, T), flush=True)
A, Bt, C, D = (words(11, nw(T)), words(12, nw(T)), words(13, nw(T)), words(15, nw(2.5 * T)))

p, kv, log = start(env, "S")
try:
    m = metrics()
    check(abs(m["pulsar:kv_touched_budget_bytes"] - budget_touched) < 2**20,
          "the guarded server's touched budget is %.1f MiB" % (m["pulsar:kv_touched_budget_bytes"] / 2**20))
    print("S: A and B resident, then C on a fresh bank", flush=True)
    ra, rb = chat(A), chat(Bt)
    check(ra[0] == 200 and rb[0] == 200, "A %s (%s prompt tokens), B %s" % (ra[0], ra[2] and ra[2].get("prompt_tokens"),
                                                                            rb[0]))
    rc = chat(C)
    txt = open(log).read()
    check(rc[0] == 200, "C answered %s" % rc[0])
    check("spilling 1 idle bank(s) first" in txt, "the server spilled an idle bank BEFORE C bound")
    print("R: a prompt past the budget by itself", flush=True)
    t0 = time.time()
    rd = chat(D)
    check(rd[0] == 503 and "not enough KV memory" in rd[1], "D refused %s in %.1f s: %s" % (rd[0], time.time() - t0,
                                                                                          rd[1][:200]))
    print("P: reuse == cold", flush=True)
    rb2 = chat(Bt)
    check(rb2[0] == 200 and rb2[1] == rb[1] and len(json.loads(rb[1])[1]) > 0,
          "B again (restored from its spill) answers as B cold, token for token with the same logprobs (%d tokens)"
          % len(json.loads(rb[1])[1]))
    txt = open(log).read()
    check(txt.count("kv admission REFUSED") == 1, "one refusal line for the one refused request")
finally:
    stop(p, kv, log)

p, kv, log = start(env, "W")
try:
    print("W: A and B streaming on two banks, then C: B preempted for C, then readmitted (L314)", flush=True)
    res = {}
    fa, fb = threading.Event(), threading.Event()
    ta = threading.Thread(target=long_stream, args=(A, fa, res, "a"))
    ta.start()
    fa.wait(600)
    tb = threading.Thread(target=long_stream, args=(Bt, fb, res, "b"))
    tb.start()
    fb.wait(600)
    rc = chat(C)
    c_end = time.time()
    ta.join()
    tb.join()
    txt = open(log).read()
    check(res.get("a", (0,))[0] == 200 and res.get("b", (0,))[0] == 200,
          "A and B streamed %s / %s tokens" % (res.get("a", (0, 0))[1], res.get("b", (0, 0))[1]))
    check(rc[0] == 200, "C answered %s" % rc[0])
    npre, nre = txt.count("PREEMPTED bank"), txt.count("READMITTED")
    nwait = txt.count("queued until a busy bank's job ends")
    ok = npre == 1 and nre == 1 and nwait == 0
    check(ok, "C took the newest decoder's bank: preempted %d, readmitted %d, queued %d (want 1, 1, 0)"
          % (npre, nre, nwait))
    b_end = res.get("b", (0, 0, 0))[2]
    check(b_end > 0 and c_end <= b_end, "C finished while B was parked, before B's stream ended (%.1f s before)"
          % (b_end - c_end if b_end else 0))
    if not ok:
        for l in kv_lines(txt):
            print("#   W: " + l[-220:])
finally:
    stop(p, kv, log)
# L328 owns this: KV admission (L314's readmission among it) ignores admitted-but-unprefilled prompts, so W's parked B
# is readmitted before C prefills and the guard ends over budget with no victim.  Its fix sets this False.
KNOWN_L328 = True
bp = [(ph, i, l) for ph, t in logs for i, l in enumerate(t.splitlines()) if "NO idle victim" in l]
where = "; %d in phase(s) %s" % (len(bp), ",".join(sorted({ph for ph, _, _ in bp}))) if bp else ""
if KNOWN_L328:
    # a STRICT expected failure: reproducing it is not counted; NOT reproducing it is (the marker cannot outlive the bug)
    if bp:
        print("  XFAIL (L328) the decode guard proceeded past its budget (back-pressure line%s): L314's readmission "
              "fits a parked decoder back before the prompt it was preempted for has prefilled" % where, flush=True)
    else:
        check(False, "XPASS: L328 no longer reproduces -- drop KNOWN_L328")
else:
    check(not bp, "the decode guard never proceeded past its budget (no back-pressure line%s)" % where)
for ph, i, l in bp[:2]:   # the first ones, with the decisions that led there
    text = dict(logs)[ph]
    for c in kv_lines(text, upto=i):
        print("#   %s: %s" % (ph, c[-220:]))
    print("#   %s: >> %s" % (ph, l[-220:]))
if a.log:
    open(a.log, "w").write("\n".join("==== phase %s\n%s" % (ph, t) for ph, t in logs))
    print("server logs: %s" % a.log)
print("server-kv-admission-live: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(1 if fails else 0)

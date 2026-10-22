#!/usr/bin/env python3
"""server_kv_headroom_live.py -- L319: the touched-KV ceiling is the box's measured headroom, so long sessions spill
or wait at the floor instead of running the machine out of memory.

    python3 tests/server_kv_headroom_live.py MODEL [--binary ./pulsar-server] [--port 8198] [--ctx 1048576]
                                             [--headroom-gib 0.75] [--timeout 1500] [--log PATH]

The defect (L319, 2026-10-22, e80e7a66): the 2b guard's ceiling was the ledger's share (kv budget - eager), and in
pooled mode the ledger's budget is not physical (a startup clamp lifts it to slot 0 + one session estimate).  On
Nemotron 3.5 Super at --ctx 1M it allowed 9.05 GiB of touched KV against 5.16 GiB above the floor; the guard never
spilled, MemAvailable reached zero under long opencode sessions, and the GPU stalled on unbacked pages at work (96%
busy, no XID) while earlyoom killed the server at home.  This reproduces that box at a small scale: a balloon takes
all but the floor plus --headroom-gib of the box's memory BEFORE the server starts (so its post-warmup reading is
the tight one), then conversations whose KV adds up to ~2.4x the headroom land on fresh banks one by one.

  phase 0  a probe server, no balloon: its own memory cost (MemAvailable before - after its warmup), the KV bytes a
           token, the tokens a word
  H        the guarded server's startup line: its touched budget is at most touched + MemAvailable - floor at its
           post-warmup reading (the measured headroom), not the ledger's share
  F        conversations of 0.6x the headroom each on fresh banks: every one answers 200, the server spills or waits
           (an admission line), and MemAvailable -- sampled every 0.25 s throughout -- never falls more than
           --slack-gib under the floor.  A sample past --abort-gib under the floor ends the run AT ONCE as a FAIL
           (well before earlyoom's 2%): the old ceiling lets the banks eat the floor, and this stops it there.
  never    the decode guard's "NO idle victim -- back-pressure" line

The whole run is under --timeout: a hang is a FAIL, never a stuck gate.  Needs the box to itself (it measures the
system's MemAvailable): a quiet gpuq job with production stopped.  Exit 0 only when every check passed."""
import argparse, json, mmap, os, random, re, shutil, signal, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8198)
ap.add_argument("--ctx", type=int, default=1048576)
ap.add_argument("--banks", type=int, default=6)
ap.add_argument("--headroom-gib", type=float, default=0.75, help="MemAvailable above the floor the server starts with")
ap.add_argument("--slack-gib", type=float, default=0.5, help="how far under the floor MemAvailable may dip (unpriced)")
ap.add_argument("--abort-gib", type=float, default=1.0, help="a dip this far under the floor ends the run as a FAIL")
ap.add_argument("--timeout", type=int, default=1500)
ap.add_argument("--log", default=os.path.join(os.environ.get("TMPDIR", "/var/tmp"), "server-kv-headroom-live.log"),
                help="the guarded server's log is kept here")
a = ap.parse_args()
URL = "http://127.0.0.1:%d" % a.port
GiB = float(1 << 30)
FLOOR = 4 * GiB   # PULSAR_SERVER_MEM_FLOOR_BYTES
fails = 0
procs = []        # every child to kill on the way out (servers)
balloon = []      # the balloon's mapping
scratch = []


def check(ok, what):
    global fails
    fails += not ok
    print("  %s %s" % ("ok  " if ok else "FAIL", what), flush=True)


def cleanup():
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    for m in balloon:
        m.close()
    balloon.clear()
    for d in scratch:
        shutil.rmtree(d, ignore_errors=True)


def on_timeout(signum, frame):
    print("  FAIL the run passed its %d s timeout (a hang)" % a.timeout, flush=True)
    cleanup()
    print("server-kv-headroom-live: FAIL (timeout)", flush=True)
    os._exit(1)


signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(a.timeout)


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def start(tag):
    kv = tempfile.mkdtemp(prefix="kvhead-")
    scratch.append(kv)
    log = os.path.join(kv, "server.log")
    env = dict(os.environ, PULSAR_MSEQ_BANKS=str(a.banks))
    env.pop("PULSAR_SERVER_KV_BUDGET_OVERRIDE", None)   # the ceiling under test is the server's own
    f = open(log, "w")
    p = subprocess.Popen([a.binary, "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port), "--ctx", str(a.ctx),
                          "--kv-disk-dir", kv, "--kv-disk-space-mb", "16384"], stdout=f, stderr=subprocess.STDOUT,
                         env=env)
    procs.append(p)
    for _ in range(900):
        txt = open(log).read()
        if "listening on" in txt:
            return p, log
        if p.poll() is not None:
            break
        time.sleep(1)
    print(open(log).read()[-3000:])
    check(False, "server %s started" % tag)
    cleanup()
    sys.exit(1)


def stop(p):
    p.terminate()
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


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


def chat(text, max_tokens=8):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(
        {"model": "m", "temperature": 0, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": text + "\n\nSay only the last number above."}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read().decode(errors="replace")).get("usage", {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


def words(seed, n):
    r = random.Random(seed)
    return " ".join("%s%d" % (r.choice(("alpha", "bravo", "cedar", "delta", "ember", "fjord")), r.randrange(10**6))
                    for _ in range(n))


def post_warmup_avail(txt):
    m = re.search(r"MemAvailable ([0-9.]+) GiB post-warmup", txt)
    return float(m.group(1)) * GiB if m else 0.0


try:
    # ---- phase 0: the server's own cost, the KV price, the tokens per word
    a0 = mem_available()
    p, log = start("probe")
    m = metrics()
    per_token = m["pulsar:kv_bank_bytes"] / a.ctx
    code, usage = chat(words(1, 400))
    tpw = usage.get("prompt_tokens", 1200) / 400.0
    a1 = post_warmup_avail(open(log).read())
    stop(p)
    cost = a0 - a1
    print("phase 0: MemAvailable %.2f GiB, the server costs %.2f GiB to its post-warmup reading; %.0f B of KV a token, "
          "%.2f tokens a word" % (a0 / GiB, cost / GiB, per_token, tpw), flush=True)
    check(a1 > 0 and cost > 0, "the probe measured its cost")

    # ---- the balloon: all but the floor + headroom, before the guarded server starts
    time.sleep(5)
    want = mem_available() - cost - FLOOR - a.headroom_gib * GiB
    check(want > 0, "room for a balloon of %.2f GiB" % (want / GiB))
    if want > 0:
        mb = mmap.mmap(-1, int(want), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_POPULATE", 0))
        balloon.append(mb)
        chunk = b"\x01" * (64 << 20)
        for off in range(0, len(mb) - len(chunk) + 1, len(chunk)):   # resident whatever MAP_POPULATE did
            mb[off:off + len(chunk)] = chunk
    print("balloon %.2f GiB: MemAvailable now %.2f GiB" % (want / GiB, mem_available() / GiB), flush=True)

    # ---- H: the guarded server's ceiling is its measured headroom
    p, log = start("guarded")
    txt = open(log).read()
    av = post_warmup_avail(txt)
    mb_ = re.search(r"touched budget ([0-9.]+) GiB", txt)
    mt = re.search(r"touched ([0-9.]+) \+ MemAvailable", txt)
    ceiling = float(mb_.group(1)) * GiB if mb_ else -1.0
    touched0 = float(mt.group(1)) * GiB if mt else 0.0
    headroom = av - FLOOR
    print("H: post-warmup MemAvailable %.2f GiB (headroom %.2f GiB over the floor), touched budget %.2f GiB"
          % (av / GiB, headroom / GiB, ceiling / GiB), flush=True)
    check(ceiling >= 0, "the server announced its touched budget")
    check(ceiling <= touched0 + headroom + 0.02 * GiB,
          "the touched budget %.2f GiB is within touched %.2f + the measured headroom %.2f GiB"
          % (ceiling / GiB, touched0 / GiB, headroom / GiB))

    # ---- F: fill past the headroom on fresh banks; MemAvailable sampled throughout
    low = [mem_available()]
    stop_sampling = threading.Event()
    aborted = threading.Event()

    def sample():
        while not stop_sampling.is_set():
            v = mem_available()
            low[0] = min(low[0], v)
            if v < FLOOR - a.abort_gib * GiB:
                aborted.set()
                print("  FAIL MemAvailable %.2f GiB: %.2f GiB under the floor -- stopping the run before the box "
                      "runs dry" % (v / GiB, (FLOOR - v) / GiB), flush=True)
                for q in procs:
                    if q.poll() is None:
                        q.kill()
                return
            time.sleep(0.25)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    conv_tokens = int(min(0.6 * max(headroom, 0.1 * GiB) / per_token, 0.9 * a.ctx))
    n_conv = max(a.banks, int(2.4 * max(headroom, 0.1 * GiB) / (conv_tokens * per_token)) + 1)
    n_conv = min(n_conv, a.banks + 2)
    print("F: %d conversations of ~%d tokens (%.2f GiB of KV each) on fresh banks" % (
        n_conv, conv_tokens, conv_tokens * per_token / GiB), flush=True)
    for i in range(n_conv):
        if aborted.is_set():
            break
        t0 = time.time()
        code, usage = chat(words(100 + i, int(conv_tokens / tpw)))
        print("  conversation %d: %s in %.1f s (%s prompt tokens), MemAvailable %.2f GiB (low %.2f)" % (
            i, code, time.time() - t0, usage.get("prompt_tokens"), mem_available() / GiB, low[0] / GiB), flush=True)
        if not aborted.is_set():
            check(code == 200, "conversation %d answered %s" % (i, code))
    stop_sampling.set()
    th.join()
    txt = open(log).read() if os.path.exists(log) else ""
    check(not aborted.is_set(), "the run was not stopped at the floor")
    check(low[0] <= av - 0.5 * headroom, "the fill pressed: MemAvailable fell %.2f GiB from its post-warmup %.2f GiB "
          "(at least half the headroom)" % ((av - low[0]) / GiB, av / GiB))
    check(low[0] >= FLOOR - a.slack_gib * GiB, "MemAvailable stayed within %.2f GiB of the floor (low %.2f GiB)"
          % (a.slack_gib, low[0] / GiB))
    adm = txt.count("spilling") + txt.count("queued until a busy bank") + txt.count("guard: spilled")
    check(adm > 0, "the ceiling bound: %d spill / wait line(s)" % adm)
    check("NO idle victim" not in txt, "the decode guard never proceeded past its budget (no back-pressure line)")
    if p.poll() is None:
        stop(p)
    shutil.copyfile(log, a.log)
    print("  (the guarded server's log: %s)" % a.log, flush=True)
finally:
    signal.alarm(0)
    cleanup()
print("server-kv-headroom-live: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails), flush=True)
sys.exit(1 if fails else 0)

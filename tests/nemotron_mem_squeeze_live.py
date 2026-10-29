#!/usr/bin/env python3
"""nemotron_mem_squeeze_live.py -- the 2026-10-29 work hang, as a leg: long conversations grown by ~15k-token
continued prefills toward ~200k while MemAvailable sits near the floor, and the server neither hangs nor eats the
floor.

    python3 tests/nemotron_mem_squeeze_live.py MODEL [--binary ./pulsar-server] [--port 8199] [--ctx 1048576]
        [--banks 16] [--convs 3] [--target 200000] [--turn 15000] [--headroom-gib 3.0] [--slack-gib 0.25]
        [--req-timeout 900] [--timeout 7200] [--log PATH]

The incident (work box, dev 65d463a5, Nemotron 3.5 Super K4 + LiLiCorr, 16 banks at --ctx 1M, opencode): the 2b
guard's ceiling was cut once at warmup and priced all the post-warmup headroom as KV's, so memory nobody accounts for
took the 4 GiB floor; the box reached the kernel's min watermark, nvidia-uvm failed to allocate a
uvm_va_block_context_t while servicing a GPU fault on a managed KV pool (dmesg), the GPU never got its page and the
worker spun in save_segment's device drain for 36 minutes.  Sparky's #490 run of the same shape (no balloon) fell to
3.9 GB MemAvailable -- under the floor -- with touched KV still under the guard's budget.

  B   a balloon takes all but (the derived floor + --headroom-gib) of the box's memory BEFORE the server starts, so the
      post-warmup reading is the tight one (the floor is read from a probe server's "memory floor" line)
  G   --convs conversations grow in parallel, each by --turn-token user turns (continued prefills on their own banks,
      a "continued" persist at every grid cut) toward --target tokens -- together ~2x the headroom of KV
  S   MemAvailable is sampled every 0.25 s, /health every 2 s

PASS (exit 0) only when every check holds:
  1. no hang: every request returns within --req-timeout, the whole run within --timeout, the server is alive at
     the end (a hung worker is a FAIL here, never a stuck gate)
  2. the watchdog stays quiet: /health never says "stalled", the log has no "STALL:" line (PULSAR_STALL_WATCHDOG_S is
     set to 60 for the run, so a real stall would show)
  3. MemAvailable never falls more than --slack-gib under the derived floor
  4. persists happen or are skipped loudly: at least one "kv cache persisted reason=continued" line, and the
     "persist SKIPPED" lines appear iff pulsar:kv_persist_skipped_total > 0; no "chain not fully stored" line
  5. every answer is 200, or a 503 that names the memory refusal ("not enough KV memory" / "not enough memory to
     continue"); never a 500
  6. the run pressed: MemAvailable came within 1 GiB of the floor, or the guard spilled / refused (else it proved
     nothing)
  7. dmesg (when readable) has no new page-allocation failure or nvidia-uvm line

Needs the box to itself (it measures system MemAvailable): a quiet gpuq job, production stopped.  ~1-2 h at the
defaults (three 200k conversations at ~550 t/s of continued prefill)."""
import argparse, json, mmap, os, random, re, shutil, signal, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8199)
ap.add_argument("--ctx", type=int, default=1048576)
ap.add_argument("--banks", type=int, default=16)
ap.add_argument("--convs", type=int, default=3)
ap.add_argument("--target", type=int, default=200000, help="tokens each conversation grows to")
ap.add_argument("--turn", type=int, default=15000, help="tokens a continued prefill adds")
ap.add_argument("--headroom-gib", type=float, default=3.0, help="MemAvailable above the derived floor at start")
ap.add_argument("--slack-gib", type=float, default=0.25, help="how far under the floor MemAvailable may dip")
ap.add_argument("--req-timeout", type=int, default=900)
ap.add_argument("--timeout", type=int, default=7200)
ap.add_argument("--log", default=os.path.join(os.environ.get("TMPDIR", "/var/tmp"), "nemotron-mem-squeeze.log"))
a = ap.parse_args()
URL = "http://127.0.0.1:%d" % a.port
GiB = float(1 << 30)
fails = 0
procs, balloon, scratch = [], [], []


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
    print("nemotron-mem-squeeze-live: FAIL (timeout)", flush=True)
    os._exit(1)


signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(a.timeout)


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def dmesg_lines():
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines() if out.returncode == 0 else None
    except Exception:
        return None


def start(tag):
    kv = tempfile.mkdtemp(prefix="memsq-", dir=os.environ.get("TMPDIR") or "/var/tmp")
    scratch.append(kv)
    log = os.path.join(kv, "server.log")
    env = dict(os.environ, PULSAR_MSEQ_BANKS=str(a.banks), PULSAR_STALL_WATCHDOG_S="60")
    env.pop("PULSAR_SERVER_KV_BUDGET_OVERRIDE", None)
    f = open(log, "w")
    p = subprocess.Popen([a.binary, "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port), "--ctx", str(a.ctx),
                          "--kv-disk-dir", kv, "--kv-disk-space-mb", "65536"], stdout=f, stderr=subprocess.STDOUT,
                         env=env)
    procs.append(p)
    for _ in range(1800):
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
        p.wait(300)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def get(path, timeout=30):
    try:
        with urllib.request.urlopen(URL + path, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:   # noqa: BLE001 -- a dead or wedged server is an answer here
        return 0, str(e)


def metrics():
    out = {}
    for line in get("/metrics")[1].splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def chat(messages, max_tokens=16):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(
        {"model": "m", "temperature": 0, "max_tokens": max_tokens, "messages": messages}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=a.req_timeout) as r:
            out = json.loads(r.read().decode(errors="replace"))
            return r.status, out.get("usage", {}), out["choices"][0]["message"].get("content") or ""
    except urllib.error.HTTPError as e:
        return e.code, {}, e.read().decode(errors="replace")[:400]
    except Exception as e:   # noqa: BLE001 -- a timeout is the hang this leg exists for
        return -1, {}, str(e)


def words(seed, n):
    r = random.Random(seed)
    return " ".join("%s%d" % (r.choice(("alpha", "bravo", "cedar", "delta", "ember", "fjord")), r.randrange(10**6))
                    for _ in range(n))


REFUSALS = ("not enough KV memory", "not enough memory to continue")

try:
    dmesg0 = dmesg_lines()
    # ---- probe: the server's own cost, the derived floor, tokens a word
    a0 = mem_available()
    p, log = start("probe")
    code, usage, _ = chat([{"role": "user", "content": words(1, 400) + "\nSay only the last number."}])
    tpw = usage.get("prompt_tokens", 1200) / 400.0
    txt = open(log).read()
    m1 = re.search(r"MemAvailable ([0-9.]+) GiB post-warmup", txt)
    mf = re.search(r"memory floor ([0-9.]+) MiB", txt)
    a1 = float(m1.group(1)) * GiB if m1 else 0.0
    floor = float(mf.group(1)) * (1 << 20) if mf else 4 * GiB
    stop(p)
    cost = a0 - a1
    print("probe: the server costs %.2f GiB; memory floor %.2f GiB; %.2f tokens a word" % (cost / GiB, floor / GiB,
                                                                                          tpw), flush=True)
    check(a1 > 0 and cost > 0 and mf is not None, "the probe measured its cost and announced its floor")

    # ---- B: the balloon
    time.sleep(10)
    want = mem_available() - cost - floor - a.headroom_gib * GiB
    if want > 0:
        mb = mmap.mmap(-1, int(want), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_POPULATE", 0))
        balloon.append(mb)
        chunk = b"\x01" * (64 << 20)
        for off in range(0, len(mb) - len(chunk) + 1, len(chunk)):
            mb[off:off + len(chunk)] = chunk
    print("B: balloon %.2f GiB, MemAvailable now %.2f GiB" % (max(want, 0) / GiB, mem_available() / GiB), flush=True)

    p, log = start("squeezed")
    low = [mem_available()]
    stalled_seen, health_bad = [], []
    done = threading.Event()

    def sample():
        n = 0
        while not done.is_set():
            low[0] = min(low[0], mem_available())
            if n % 8 == 0:
                code, body = get("/health", timeout=10)
                if code != 200 or '"stalled":0' not in body:
                    (stalled_seen if '"stalled"' in body and '"status":"stalled"' in body else health_bad).append(
                        (time.time(), code, body[:200]))
            n += 1
            time.sleep(0.25)

    th = threading.Thread(target=sample, daemon=True)
    th.start()

    # ---- G: the conversations
    results = []   # (conv, turn, code, seconds, prompt_tokens, reply)

    def converse(c):
        msgs, have, t = [], 0, 0
        while have < a.target:
            msgs.append({"role": "user", "content": words(1000 * c + t, int(a.turn / tpw)) +
                         "\nSay only the last number above."})
            t0 = time.time()
            code, usage, reply = chat(msgs)
            results.append((c, t, code, time.time() - t0, usage.get("prompt_tokens"), reply))
            print("  conv %d turn %d: %s in %.1f s (%s prompt tokens), MemAvailable %.2f GiB (low %.2f)" % (
                c, t, code, time.time() - t0, usage.get("prompt_tokens"), mem_available() / GiB, low[0] / GiB),
                flush=True)
            if code != 200:
                return
            have = usage.get("prompt_tokens", have + a.turn)
            msgs.append({"role": "assistant", "content": reply})
            t += 1

    ths = [threading.Thread(target=converse, args=(c,)) for c in range(a.convs)]
    for x in ths:
        x.start()
    for x in ths:
        x.join()
    done.set()
    th.join()
    txt = open(log).read()
    m = metrics()

    # ---- the pass rule
    hung = [r for r in results if r[2] == -1]
    check(not hung and p.poll() is None,
          "1 no hang: every request answered within %d s and the server is alive (%d hung)" % (a.req_timeout, len(hung)))
    check(not stalled_seen and "STALL:" not in txt,
          "2 the watchdog stayed quiet (%d stalled /health answers, %d STALL lines)" % (len(stalled_seen),
                                                                                        txt.count("STALL:")))
    check(not health_bad, "2b /health answered 200 throughout (%d other answers)" % len(health_bad))
    check(low[0] >= floor - a.slack_gib * GiB,
          "3 MemAvailable stayed within %.2f GiB of the %.2f GiB floor (low %.2f GiB)" % (a.slack_gib, floor / GiB,
                                                                                         low[0] / GiB))
    persisted = len(re.findall(r"kv cache persisted reason=continued", txt))
    skipped_lines = txt.count("kv cache persist SKIPPED")
    skipped = m.get("pulsar:kv_persist_skipped_total", 0.0)
    check(persisted > 0 and (skipped_lines > 0) == (skipped > 0) and "chain not fully stored" not in txt,
          "4 persists happened (%d continued) or were skipped loudly (%d skipped, %d lines)" % (persisted, skipped,
                                                                                               skipped_lines))
    bad = [r for r in results if r[2] not in (200, 503, -1) or (r[2] == 503 and not any(x in r[5] for x in REFUSALS))]
    check(not bad, "5 every answer was 200 or a named memory refusal: %s" % ([(r[0], r[1], r[2], r[5][:120])
                                                                              for r in bad][:3] or "yes"))
    pressed = low[0] <= floor + 1.0 * GiB
    acted = (txt.count("guard EVICTED") + txt.count("kv admission REFUSED") + txt.count("prefill REFUSED") +
             txt.count("PREEMPTED") + int(skipped))
    check(pressed or acted > 0, "6 the run pressed (low %.2f GiB vs floor %.2f; %d guard / refusal actions)" % (
        low[0] / GiB, floor / GiB, acted))
    dmesg1 = dmesg_lines()
    if dmesg0 is not None and dmesg1 is not None:
        new = dmesg1[len(dmesg0):]
        hits = [x for x in new if "page allocation failure" in x or "uvm" in x.lower() or "Xid" in x]
        check(not hits, "7 dmesg: no page-allocation failure / nvidia-uvm / Xid line (%s)" % (hits[:2] or "none"))
    else:
        print("  (7 dmesg unreadable here: not checked)", flush=True)
    if p.poll() is None:
        stop(p)
    shutil.copyfile(log, a.log)
    print("  (the squeezed server's log: %s)" % a.log, flush=True)
finally:
    signal.alarm(0)
    cleanup()
print("nemotron-mem-squeeze-live: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails), flush=True)
sys.exit(1 if fails else 0)

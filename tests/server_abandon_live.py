#!/usr/bin/env python3
"""server_abandon_live.py -- L282: a prefill its client abandons while the decode lane runs is released, not held.

    python3 tests/server_abandon_live.py MODEL [--binary ./pulsar-server] [--port 8197]

A real pulsar-server (its own disk-KV dir under /tmp): two long streaming decodes fill the decode lane, a ~9k-token
prompt starts prefilling and its client hangs up, and for 30 s the slot phases are read from /metrics.  Before the
fix the abandoned slot sat in "finish" until the lane emptied (the pair, 2026-10-07: six slots for 23 minutes, two
prompts starved behind them).  PASS: the abandon was logged (the lane was exercised) and no slot is in "finish" after
the first 5 s while both decodes still run.  Exit 0 only on PASS.  Run by `make server-live-gate`."""
import argparse, json, os, re, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8197)
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port
d = tempfile.mkdtemp(prefix="pulsar-abandon-live.")
log_path = os.path.join(d, "server.log")
log = open(log_path, "wb")
srv = subprocess.Popen([a.binary, "-m", a.model, "--ctx", "32768", "--host", "127.0.0.1", "--port", str(a.port),
                        "--kv-disk-dir", os.path.join(d, "kv")], stdout=log, stderr=subprocess.STDOUT)


def stop(rc, why):
    print("SERVER ABANDON LIVE: %s -- %s" % ("PASS" if rc == 0 else "FAIL", why))
    if srv.poll() is None:
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
    log.close()
    if rc:
        print("--- server log tail ---\n" + open(log_path, "rb").read()[-2000:].decode("utf-8", "replace"))
    shutil.rmtree(d, ignore_errors=True)
    sys.stdout.flush()   # os._exit skips the interpreter's flush
    os._exit(rc)         # the two decode threads are still streaming


t0 = time.time()
while True:
    if srv.poll() is not None:
        stop(1, "the server exited during startup")
    try:
        urllib.request.urlopen(B + "/health", timeout=2).read()
        break
    except Exception:
        if time.time() - t0 > 900:
            stop(1, "the server did not listen within 900 s")
        time.sleep(2)


def phases():
    out = {}
    for ln in urllib.request.urlopen(B + "/metrics", timeout=5).read().decode().splitlines():
        if ln.startswith("pulsar:slot_phase{") and ln.endswith(" 1"):
            ph = re.search(r'phase="(\w+)"', ln).group(1)
            out[ph] = out.get(ph, 0) + 1
    return out


done = []


def decode_stream(tag):
    body = {"model": "m", "max_tokens": 3000, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content": "Write a very long, detailed story about a lighthouse keeper "
                                                     "(%s). Do not stop early." % tag}]}
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"content-type": "application/json"}), timeout=900)
    for _ in r:
        pass
    done.append(tag)


for tag in ("A", "B"):
    threading.Thread(target=decode_stream, args=(tag,), daemon=True).start()
for _ in range(120):
    if phases().get("decode", 0) >= 2:
        break
    time.sleep(1)
else:
    stop(1, "two decodes never ran together: %s" % phases())

big = " ".join("filler%d" % i for i in range(9000))
body = json.dumps({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": big}]}).encode()
s = socket.create_connection(("127.0.0.1", a.port))
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
          b"Content-Length: %d\r\n\r\n" % len(body) + body)
time.sleep(2.0)
before = phases()
s.close()
held = []
for i in range(6):
    time.sleep(5)
    held.append(phases().get("finish", 0))
abandoned = open(log_path, "rb").read().count(b"client disconnected during prefill, abandoning")
if done:
    stop(1, "a decode ended before the 30 s window closed (%s): the lane was not running throughout" % done)
if abandoned < 1:
    stop(1, "no abandon logged (phases at hang-up %s): the case did not exercise the lane's abandon" % before)
if any(held[1:]):
    stop(1, "a slot stayed in finish while the lane ran (finish counts every 5 s: %s)" % held)
stop(0, "abandoned prefill released while 2 decodes ran (finish counts every 5 s: %s)" % held)

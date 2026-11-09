#!/usr/bin/env python3
"""server_abandon_live.py -- L282: a prefill its client abandons while the decode lane runs is released, not held.

    python3 tests/server_abandon_live.py MODEL [--binary ./pulsar-server] [--port 8197]

Two long streaming decodes fill the decode lane, a ~16k-token prompt starts prefilling and its client hangs up, and
for 30 s the slot phases are read from /metrics.  Before the fix the abandoned slot sat in "finish" until the lane
emptied (the pair, 2026-10-07: six slots for 23 minutes, two prompts starved behind them).  PASS: the abandon was
logged (the lane was exercised) and no slot is in "finish" after the first 5 s while both decodes still run.

L317: a leg of tests/live_legs.py (warm phase, on the model's shared server): the two decodes' clients hang up when the
window closes and the leg waits for the decode lane to drain, so the next leg finds it idle.  Alone (this command
line) it runs on an instance of its own.  Exit 0 only on PASS."""
import json, os, re, socket, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg  # noqa: E402


class AbandonLeg(Leg):
    NAME = "server-abandon"

    def phases(self):
        out = {}
        for ln in urllib.request.urlopen(self.srv.url + "/metrics", timeout=5).read().decode().splitlines():
            if ln.startswith("pulsar:slot_phase{") and ln.endswith(" 1"):
                ph = re.search(r'phase="(\w+)"', ln).group(1)
                out[ph] = out.get(ph, 0) + 1
        return out

    def count_tokens(self, text):
        r = urllib.request.urlopen(urllib.request.Request(self.srv.url + "/v1/messages/count_tokens", json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": text}]}).encode(),
            {"content-type": "application/json"}), timeout=60)
        return json.loads(r.read())["input_tokens"]

    def warm(self, srv):
        self.srv = srv
        done, conns, stop = [], [], threading.Event()

        def decode_stream(tag):
            body = {"model": "m", "max_tokens": 3000, "temperature": 0, "stream": True,
                    "messages": [{"role": "user", "content": "Write a very long, detailed story about a lighthouse "
                                                             "keeper (%s). Do not stop early." % tag}]}
            try:
                r = urllib.request.urlopen(urllib.request.Request(srv.url + "/v1/chat/completions",
                                                                  json.dumps(body).encode(),
                                                                  {"content-type": "application/json"}), timeout=900)
                conns.append(r)
                for _ in r:
                    if stop.is_set():
                        break
            except Exception:
                pass
            if not stop.is_set():
                done.append(tag)

        th = [threading.Thread(target=decode_stream, args=(tag,), daemon=True) for tag in ("A", "B")]
        for t in th:
            t.start()
        try:
            self.window(srv, done)
        finally:
            stop.set()
            for r in conns:   # the clients hang up: the decodes are abandoned too, or run out their 3000 tokens
                try:
                    r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
            t0 = time.time()
            while time.time() - t0 < 300 and self.phases().get("decode", 0) + self.phases().get("finish", 0):
                time.sleep(1)
            self.g.note("decode lane idle %.0f s after the window (phases %s)" % (time.time() - t0, self.phases()))

    def window(self, srv, done):
        for _ in range(120):
            if self.phases().get("decode", 0) >= 2:
                break
            time.sleep(1)
        else:
            self.g.check(False, "two decodes never ran together: %s" % self.phases())
            return
        # ~16k tokens in THIS model's tokenizer (long enough to be caught mid-prefill; it fits a 32k context alone)
        n_words = 2000
        n_words = int(n_words * 16000 / max(1, self.count_tokens(" ".join("filler%d" % i for i in range(n_words)))))
        big = " ".join("filler%d" % i for i in range(n_words))
        body = json.dumps({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": big}]}).encode()
        mark = srv.mark()
        s = socket.create_connection(("127.0.0.1", srv.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body)
        t0 = time.time()
        while True:   # hang up the moment the prompt is prefilling (a fixed delay missed it on Qwen: queued, or done)
            before = self.phases()
            if before.get("prefill_main", 0) + before.get("prefill_cold", 0) >= 1:
                break
            if time.time() - t0 > 120:
                s.settimeout(2)
                try:
                    reply = s.recv(400).decode("utf-8", "replace")
                except OSError:
                    reply = "(no reply)"
                s.close()
                self.g.check(False, "the long prompt never reached prefill beside the decodes (phases %s; server: %r)"
                             % (before, reply))
                return
            time.sleep(0.2)
        s.close()
        held = []
        for i in range(6):
            time.sleep(5)
            held.append(self.phases().get("finish", 0))
        abandoned = srv.since(mark).count("client disconnected during prefill, abandoning")
        self.g.check(not done, "both decodes ran through the 30 s window (ended early: %s)" % done)
        self.g.check(abandoned >= 1, "the abandon was logged (%d; phases at hang-up %s): the case exercised the lane"
                     % (abandoned, before))
        self.g.check(not any(held[1:]), "abandoned prefill released while 2 decodes ran (finish counts every 5 s: %s)"
                     % held)


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(AbandonLeg, sys.argv[1:], ctx=32768, port=8197))

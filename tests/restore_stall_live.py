#!/usr/bin/env python3
"""restore_stall_live.py -- L330: a disk KV restore no longer freezes every decoding stream for the whole read.

    python3 tests/restore_stall_live.py MODEL [--binary ./pulsar-server] [--work DIR] [--port 8330] [--ctx 262144]
                                        [--banks 4] [--tokens 200000] [--slice-ms 250] [--min-gib 1.5]

Before L330 a request whose resume pick was the disk chain read the whole chain inside one worker turn
(gen_begin -> kv_cache_load -> pulsar_kvchain_load): every other stream stood still for the read (the archived
10-22..10-24 production log: 24 loads over 5 s, the longest 24.0 s, the log silent through it).  Now the chain loads
a slice of segments per worker turn, the job in GEN_RESTORE between decode rounds.

The shape, on one server with the disk tier (its hooks instance: PULSAR_SERVER_KV_RESTORE_SLICE_MS=--slice-ms, the
slice's test hook, so a chain a few GiB long -- what --ctx 262144 holds -- reads in many slices; production's slice is
one prefill chunk's priced time, which a 1M-token chain spans several of):
  C   a ~--tokens raw prompt (the story fixture behind a nonce, repeated: nothing stored serves it), its client hangs up
      once 90% of its prefill is done -- the abandon stores the computed prefix as a chain (L318) and leaves the bank
      invalid, so the identical re-send can only resume from the DISK chain
  D   a greedy chat stream decoding 1500 tokens; the page cache is dropped (the read comes from the disk), and once D
      has decoded 40 tokens, C is re-sent (1 token out)
  /metrics is polled at 50 Hz throughout: the restore window is where a slot's phase reads "restore", the reservation
  is kv_reserved_bytes - kv_touched_bytes there.

Checks (must-pass):
  * LANE (rule 5): the server announces the sliced restore and the hook's slice; C's re-send loads its chain from disk
    ("kv cache hit chain ... in S slices"), the chain at least --min-gib and S >= 4.
  * BOUND: D's longest inter-token gap inside the restore window <= the longest slice (/metrics
    kv_restore_stall_max_seconds) + one decode round (max(0.5 s, 4 x D's median gap outside the window)) -- about one
    slice, not the whole read.
  * NON-VACUOUS: the whole read (the hit line's load=) is past that bound -- a one-shot read would have failed it.
  * RESERVED: while C restores, the admission base holds at least half the chain's rows above what is touched (the job
    reserves its chain's room from bind, L328's base).
  * C's re-send and D answer 200, D with all 1500 tokens.

Alone (this command line, make restore-stall-live) it runs on an instance of its own.  Exit 0 only on PASS."""
import os, random, re, socket, json, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools", "serve-bench"))
from live_server import Leg, drop_caches  # noqa: E402
from serve_matrix import request, story_text  # noqa: E402

GEN_D = 1500


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))] if v else 0.0


class RestoreStallLeg(Leg):
    NAME = "restore-stall"

    def __init__(self, g, work, tokens=200000, slice_ms=250, min_gib=1.5, tag=None):
        Leg.__init__(self, g, work)
        self.tokens, self.slice_ms, self.min_gib = tokens, slice_ms, min_gib

    def hook_env(self):
        return {"PULSAR_SERVER_KV_RESTORE_SLICE_MS": str(self.slice_ms)}

    def prompt(self):
        text = "[%.12f restore] " % random.random()
        story, k = story_text(), 0
        while len(text) < self.tokens * 4:   # ~4 characters a token on the story fixture
            text += "\n\nPart %d.\n\n" % k + story
            k += 1
        return text[:self.tokens * 4]

    def abandon(self, srv, text):
        """C streamed; hang up at 90% of its prefill; the stored chain's end (or None)."""
        body = json.dumps({"model": "m", "prompt": text, "max_tokens": 1, "temperature": 0, "stream": True}).encode()
        m = srv.mark()
        host, port = srv.url.split("//")[1].split(":")
        s = socket.create_connection((host, int(port)))
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body)
        t0, at = time.time(), None
        while time.time() - t0 < 1800:
            prog = re.findall(r"prefill chunk (\d+)/(\d+)", srv.since(m))
            if prog and int(prog[-1][1]) > 0 and int(prog[-1][0]) * 10 >= int(prog[-1][1]) * 9:
                at = (int(prog[-1][0]), int(prog[-1][1]))
                break
            time.sleep(0.05)
        s.close()
        t1 = time.time()
        while time.time() - t1 < 180:
            hit = re.findall(r"abandoned prefill of bank \d+: chain stored to (\d+)", srv.since(m))
            if hit:
                return int(hit[-1]), at
            time.sleep(0.2)
        return None, at

    def hooked(self, srv):
        log0 = srv.text()
        self.g.check("restores are SLICED between rounds" in log0 and "%d ms reads" % self.slice_ms in log0,
                     "the server announces the sliced restore at the hook's %d ms slice" % self.slice_ms)
        per_tok = srv.metrics().get("pulsar:kv_bank_bytes", 0.0) / max(1, srv.ctx)
        text = self.prompt()
        stored, at = self.abandon(srv, text)
        if not self.g.check(stored is not None, "C abandoned at %s of its prefill stored its prefix: chain to %s"
                            % ("%d/%d" % at if at else "?", stored)):
            return
        # D decodes; the page cache dropped, so the restore reads the disk
        drop_caches()
        res, samples, stop = {}, [], threading.Event()
        td = threading.Thread(target=lambda: res.__setitem__("D", request(
            srv.port, "chat", "Write a long, detailed story about a lighthouse keeper and a storm, chapter by chapter.",
            GEN_D, 0.0, 11, "m")))

        def poll():
            while not stop.is_set():
                try:
                    mt = srv.metrics()
                    restoring = any(v == 1 and 'phase="restore"' in k for k, v in mt.items()
                                    if k.startswith("pulsar:slot_phase"))
                    samples.append((time.perf_counter(), restoring, mt.get("pulsar:kv_reserved_bytes", 0.0) -
                                    mt.get("pulsar:kv_touched_bytes", 0.0)))
                except OSError:
                    pass
                time.sleep(0.02)
        tp = threading.Thread(target=poll)
        tp.start()
        td.start()
        t0 = time.time()
        # D's first tokens: its stream is steady before C comes back (request() fills stamps only at its end, so the
        # server's slot phase says when D decodes)
        while time.time() - t0 < 600 and not any(
                v == 1 and 'phase="decode"' in k for k, v in srv.metrics().items() if k.startswith("pulsar:slot_phase")):
            time.sleep(0.1)
        time.sleep(3.0)
        m = srv.mark()
        mt0 = srv.metrics()
        rc = request(srv.port, "raw", text, 1, 0.0, 7, "m")
        td.join()
        stop.set()
        tp.join()
        log = srv.since(m)
        mt1 = srv.metrics()
        d = res.get("D") or {}
        self.g.check(len(d.get("stamps", [])) > 0 and d.get("tokens", 0) == GEN_D,
                     "D decoded all its %d tokens (%d)" % (GEN_D, d.get("tokens", 0)))
        self.g.check(rc.get("first_evt") is not None, "C's re-send answered")
        # LANE
        st = re.findall(r"restores the request's chain of (\d+) segments to (\d+) \(([\d.]+) MiB", log)
        hit = re.findall(r"kv cache hit chain of (\d+) to (\d+) .*load=([\d.]+) ms in (\d+) slices? \(longest ([\d.]+) ms\)",
                         log)
        slices = int(mt1.get("pulsar:kv_restore_slices_total", 0) - mt0.get("pulsar:kv_restore_slices_total", 0))
        gib = float(st[-1][2]) / 1024.0 if st else 0.0
        self.g.check(bool(st) and bool(hit) and int(hit[-1][1]) >= stored and gib >= self.min_gib and slices >= 4,
                     "the re-send restored its disk chain in slices: %s" %
                     ("%s segments to %s, %.2f GiB, %s slices (%d on /metrics), longest %s ms, read %s ms" %
                      (st[-1][0], st[-1][1], gib, hit[-1][3], slices, hit[-1][4], hit[-1][2]) if st and hit else
                      "no restore / hit line"))
        # the window and D's gaps in it
        win = [t for t, r, _ in samples if r]
        if not self.g.check(len(win) > 0, "the restore window was seen on /metrics (%d samples)" % len(win)):
            return
        w0, w1 = min(win) - 0.05, max(win) + 0.05
        stamps = d.get("stamps", [])
        # the gaps wholly inside the window: the slices and the decode rounds between them (the gap that runs on into
        # the prompt's first prefill chunk after the restore is the prefill's, not the restore's)
        gaps_in = [b - a for a, b in zip(stamps, stamps[1:]) if a >= w0 and b <= w1]
        gaps_out = [b - a for a, b in zip(stamps, stamps[1:]) if b < w0 or a > w1]
        stall_max = mt1.get("pulsar:kv_restore_stall_max_seconds", 0.0)
        bound = stall_max + max(0.5, 4.0 * pct(gaps_out, 50))
        worst = max(gaps_in) if gaps_in else None
        read_s = float(hit[-1][2]) / 1000.0 if hit else 0.0
        self.g.note("restore window %.2f s; D's gaps in it: %d, max %s s (p50 outside %.3f s); longest slice %.3f s; "
                    "the whole read %.2f s -- before L330 D would have stood still for all of it" %
                    (w1 - w0, len(gaps_in), "%.3f" % worst if worst is not None else "-", pct(gaps_out, 50),
                     stall_max, read_s))
        self.g.check(worst is not None and worst <= bound,
                     "BOUND: D's longest gap during the restore %s s <= one slice %.3f s + one round = %.3f s" %
                     ("%.3f" % worst if worst is not None else "(no gap in the window)", stall_max, bound))
        self.g.check(read_s > bound, "NON-VACUOUS: the whole read %.2f s is past the bound %.3f s" % (read_s, bound))
        held = max([r for t, x, r in samples if x] or [0.0])
        rows = (int(hit[-1][1]) if hit else 0) * per_tok
        self.g.check(rows > 0 and held >= 0.5 * rows,
                     "RESERVED: while restoring, the admission base held %.2f GiB above touched (the chain's rows %.2f "
                     "GiB)" % (held / 2**30, rows / 2**30))


def _args(ap):
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--slice-ms", type=int, default=250)
    ap.add_argument("--min-gib", type=float, default=1.5)


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(RestoreStallLeg, sys.argv[1:], ctx=262144, banks=4, port=8330, add_args=_args))

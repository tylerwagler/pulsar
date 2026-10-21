#!/usr/bin/env python3
"""prefix_turns_live.py -- L310 / L311 / L313: an agent loop's prefix reuse, measured on a real pulsar-server.

    python3 tests/prefix_turns_live.py MODEL --binary ./pulsar-server --work DIR [--port 8311] [--turns 8]
                                       [--side] [--ctx 65536] [--tag NAME]

An Anthropic /v1/messages tool loop (Claude Code's shape): a system prompt, a task, then TURNS rounds of a forced
`read_file` call (greedy) answered by a tool_result of ~1.2k tokens of story text, the client pausing PAUSE seconds
between the model's turn and its result (the tool's run time).  Per turn: the prompt's tokens, the cache read the
server reports (where the sync resumed), recomputed = prompt - cached, and the request's TTFT (the response's
timings block).  With --side, after each tool result a SIDE request shares the main conversation through its last
message and adds a "Describe your most recent action in 3-5 words" user message while the main turn is still being
answered (Claude Code's agent summary) -- its TTFT and recompute are reported apart.

REUSE == COLD (must-pass): the last main request is sent again to a FRESH server (an empty disk tier, cold banks)
and its greedy reply must equal the warm one byte for byte; with --side the last side request too.

The server runs with its own disk-KV dir under DIR (never /tmp), PULSAR_MSEQ_BANKS=4.  Prints one TSV line a request
(tag, kind, turn, prompt, cached, recomputed, ttft_s, e2e_s) and a summary; exit 0 only when reuse == cold held."""
import argparse, json, os, shutil, subprocess, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--work", required=True)
ap.add_argument("--port", type=int, default=8311)
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--turns", type=int, default=8)
ap.add_argument("--pause", type=float, default=2.0)
ap.add_argument("--side", action="store_true")
ap.add_argument("--reply", type=int, default=0, help="a turn writes up to this many tokens before its call (0: a bare forced call)")
ap.add_argument("--tag", default="run")
ap.add_argument("--start-timeout", type=float, default=900.0)
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port
os.makedirs(a.work, exist_ok=True)
STORY = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_context_story_prompt.txt"),
             encoding="utf-8", errors="replace").read()


class Server:
    def __init__(self, name):
        self.dir = os.path.join(a.work, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        env = dict(os.environ, PULSAR_MSEQ_BANKS="4")
        self.p = subprocess.Popen([a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port",
                                   str(a.port), "--kv-disk-dir", os.path.join(self.dir, "kv")],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit("FAIL server exited during startup (rc %s): %s" % (self.p.returncode, self.tail()))
            try:
                urllib.request.urlopen(B + "/health", timeout=2).read()
                break
            except Exception:
                pass
            if time.time() - t0 > a.start_timeout:
                raise SystemExit("FAIL server did not listen")
            time.sleep(2)
        print("# %s up in %.0f s" % (name, time.time() - t0), flush=True)

    def tail(self, n=3000):
        self.log.flush()
        return open(self.log_path, "rb").read()[-n:].decode("utf-8", "replace")

    def grep(self, word):
        self.log.flush()
        return [l for l in open(self.log_path, "rb").read().decode("utf-8", "replace").splitlines() if word in l]

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def post(body):
    """One streamed /v1/messages request: (the message as a non-streamed reply would carry it -- content + usage --,
    its e2e seconds); d["ttft_s"] = seconds to the first content delta."""
    body = dict(body, stream=True)
    req = urllib.request.Request(B + "/v1/messages", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    r = urllib.request.urlopen(req, timeout=1200)
    msg, blocks, parts, ttft = {}, [], [], None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        ty = ev.get("type")
        if ty == "message_start":
            msg = ev["message"]
        elif ty == "content_block_start":
            blocks.append(dict(ev["content_block"]))
            parts.append("")
        elif ty == "content_block_delta":
            if ttft is None:
                ttft = time.time() - t
            dl = ev["delta"]
            if dl.get("type") == "text_delta":
                blocks[-1]["text"] = blocks[-1].get("text", "") + dl["text"]
            elif dl.get("type") == "input_json_delta":
                parts[-1] += dl["partial_json"]
            elif dl.get("type") == "thinking_delta":
                blocks[-1]["thinking"] = blocks[-1].get("thinking", "") + dl["thinking"]
        elif ty == "content_block_stop":
            if blocks and blocks[-1].get("type") == "tool_use":
                blocks[-1]["input"] = json.loads(parts[-1]) if parts[-1] else {}
    msg["content"] = [b for b in blocks if b.get("type") != "thinking"]
    msg["ttft_s"] = ttft if ttft is not None else time.time() - t
    return msg, time.time() - t


TOOLS = [{"name": "read_file", "description": "Read a numbered part of the project's notes file.",
          "input_schema": {"type": "object", "properties": {"part": {"type": "integer"}}, "required": ["part"]}}]
SYSTEM = ("You are a careful assistant working through a long notes file one part at a time with the read_file tool. "
          "Read the parts in order, starting at part 1.  Before each read_file call, write a detailed summary of "
          "the part you just read.  " + STORY[:6000])


def body(msgs, forced, max_tokens):
    b = {"model": "m", "max_tokens": max_tokens, "temperature": 0, "system": SYSTEM, "tools": TOOLS, "messages": msgs}
    if forced:
        b["tool_choice"] = {"type": "tool", "name": "read_file"}
    return b


rows = []


def measure(srv, kind, turn, b):
    d, e2e = post(b)
    u = d.get("usage", {})
    prompt = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    cached = u.get("cache_read_input_tokens", 0)
    row = (a.tag, kind, turn, prompt, cached, prompt - cached, d["ttft_s"], e2e)
    rows.append(row)
    print("%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f" % row, flush=True)
    return d


def reply_text(d):
    """The reply's content without the server's random tool_use ids (each server names its calls afresh)."""
    return json.dumps([{k: v for k, v in c.items() if k != "id"} for c in d.get("content", [])], sort_keys=True)


def loop(srv):
    msgs = [{"role": "user", "content": "Summarise the notes file. Begin with read_file part 1."}]
    last_main = last_side = None
    th = None
    for t in range(1, a.turns + 1):
        # --reply N: the turn writes a reply of up to N tokens (the decode rows a next turn re-prefilled, L275's
        # ~969), then calls the tool; else a bare forced call
        b = body(msgs, a.reply == 0, a.reply if a.reply else 96)
        d = measure(srv, "main", t, b)
        if th:
            th.join()   # the side request rode beside this main turn
            th = None
        last_main = (b, d)
        calls = [c for c in d.get("content", []) if c.get("type") == "tool_use"]
        msgs = msgs + [{"role": "assistant", "content": d["content"] or [{"type": "text", "text": "."}]}]
        time.sleep(a.pause)   # the tool runs
        part = STORY[6000 + t * 4800: 6000 + (t + 1) * 4800]
        if calls:
            msgs = msgs + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"],
                                                         "content": part}]}]
        else:
            msgs = msgs + [{"role": "user", "content": "Part %d of the notes:\n%s\n\nSummarise it in detail, then "
                                                       "read the next part." % (t + 1, part)}]
        if a.side and t < a.turns:
            sb = body(msgs + [{"role": "user", "content": "Describe your most recent action in 3-5 words using "
                                                          "present tense (-ing)."}], False, 24)
            th = threading.Thread(target=lambda sb=sb, t=t: measure(srv, "side", t, sb))
            th.start()
            time.sleep(0.2)   # the next main turn arrives while the side request is being answered
            last_side = (sb, None)
    if th:
        th.join()
    return last_main, last_side


srv = Server(a.tag + "-warm")
ok = True
try:
    last_main, last_side = loop(srv)
    term = srv.grep("terminal checkpoint")
    disk = srv.grep("restore-vs-recompute")
    print("# terminal checkpoint lines: %d; restore-vs-recompute lines: %d" % (len(term), len(disk)))
    for l in term[-3:] + disk[-3:]:
        print("#   " + l[-200:])
    side_warm = main_again = None
    if last_main:
        time.sleep(a.pause)
        main_again, _ = post(last_main[0])   # alone on the idle warm server (its prompt end resumes it)
    if last_side:
        side_warm, _ = post(last_side[0])
finally:
    srv.stop()
if not last_main:
    sys.exit(1)

mains = [r for r in rows if r[1] == "main" and r[2] > 1]
sides = [r for r in rows if r[1] == "side"]
for name, rs in (("main", mains), ("side", sides)):
    if rs:
        print("# %s turns>1: n=%d mean recomputed %.0f tokens, mean ttft %.3f s" %
              (name, len(rs), sum(r[5] for r in rs) / len(rs), sum(r[6] for r in rs) / len(rs)))

# REUSE == COLD on a fresh server
cold = Server(a.tag + "-cold")
try:
    dc, _ = post(last_main[0])
    # in the loop beside a side request the batched decode is nondeterministic by design: there only the re-send,
    # alone, is graded
    for what, warm in (("in the loop", None if a.side else last_main[1]), ("re-sent alone", main_again)):
        if warm is None:
            continue
        same = reply_text(dc) == reply_text(warm)
        print(("PASS" if same else "FAIL") + " reuse == cold: the last main turn (%s) == a fresh server's" % what)
        if not same:
            print("  warm %s\n  cold %s" % (reply_text(warm)[:400], reply_text(dc)[:400]))
        ok &= same
    if side_warm is not None:
        ds, _ = post(last_side[0])
        same = reply_text(ds) == reply_text(side_warm)
        print(("PASS" if same else "FAIL") + " reuse == cold: the last side request's greedy reply on a fresh server")
        ok &= same
finally:
    cold.stop()
print("prefix-turns-live %s: %s" % (a.tag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""opencode_turns_live.py -- L311: opencode's conversation shape on a thinking-on model whose template truncates
history reasoning (Nemotron's truncate_history_thinking), measured on a real pulsar-server.

    python3 tests/opencode_turns_live.py MODEL --binary ./pulsar-server --work DIR --capture REQ.json [--tag T]

opencode sends exact appends: the captured request's system prompt and its tools, then per turn the user message,
the previous assistant turn coming back with its `reasoning_content` -- which the template re-renders as
"<think></think>answer", so the bank (holding the reply as decoded) and the prompt part right after the previous
generation header.  Here the user messages are the capture's own, the replies the model's (greedy, streamed,
max_tokens --gen), the client pausing --pause s between turns.

Per turn, from the server's log for that request: the bank it ran on and whether it is the previous turn's, where
its sync resumed (restored tokens) and the tokens it prefilled; and its TTFT.  REUSE == COLD: the last turn sent
again alone to the warm server and to a fresh one (an empty disk tier) -- the same greedy reply.
Exit 0 only when reuse == cold held."""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--work", required=True)
ap.add_argument("--capture", required=True, help="an opencode request (its body's messages and tools)")
ap.add_argument("--port", type=int, default=8312)
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--gen", type=int, default=400)
ap.add_argument("--pause", type=float, default=2.0)
ap.add_argument("--tag", default="run")
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port
os.makedirs(a.work, exist_ok=True)
cap = json.load(open(a.capture))
cap = cap.get("body", cap)
SYSTEM = [m for m in cap["messages"] if m["role"] == "system"]
USERS = [m["content"] for m in cap["messages"] if m["role"] == "user"]
TOOLS = cap.get("tools") or []


class Server:
    def __init__(self, name):
        self.dir = os.path.join(a.work, name)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        self.p = subprocess.Popen([a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port",
                                   str(a.port), "--kv-disk-dir", os.path.join(self.dir, "kv")],
                                  stdout=self.log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PULSAR_MSEQ_BANKS="4"))
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


def chat(msgs):
    body = {"model": "m", "messages": msgs, "tools": TOOLS, "stream": True, "max_tokens": a.gen, "temperature": 0,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t = time.time()
    content, reasoning, ttft, usage = "", "", None, {}
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                ttft = time.time() - t
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
    return {"content": content, "reasoning": reasoning, "ttft": ttft or time.time() - t, "usage": usage}


ROUTE = re.compile(r"route: best bank (\d+) .* -> (in place|fresh preferred)")
PROV = re.compile(r"provisioned bank (\d+)")
RESUME = re.compile(r"bank (\d+) (?:resumes from its (prompt end|message endpoint|checkpoint) at (\d+)|"
                    r"resumes at its prompt end (\d+)|adopts bank \d+'s [a-z ]+ at (\d+))")


def turn_facts(log_part, prompt):
    """(bank, restored, prefilled) of one request from its log lines"""
    bank, restored = None, 0
    for l in log_part.splitlines():
        m = ROUTE.search(l)
        if m and m.group(2) == "in place":
            bank = int(m.group(1))
        m = PROV.search(l)
        if m:
            bank = int(m.group(1))
        m = RESUME.search(l)
        if m:
            bank = int(m.group(1)) if bank is None else bank
            restored = int(next(g for g in m.groups()[2:] if g))
    return bank, restored


srv = Server(a.tag + "-warm")
ok = True
rows = []
try:
    msgs = list(SYSTEM)
    prev_bank, last = None, None
    for t, u in enumerate(USERS, 1):
        msgs = msgs + [{"role": "user", "content": u}]
        mark = len(srv.text())
        r = chat(msgs)
        time.sleep(0.5)
        part = srv.text()[mark:]
        prompt = (r["usage"] or {}).get("prompt_tokens", 0)
        bank, restored = turn_facts(part, prompt)
        same = prev_bank is not None and bank == prev_bank
        rows.append((t, bank, same, restored, prompt - restored, r["ttft"]))
        print("%s\tturn %d\tbank %s\tsame %s\trestored %d\tprefilled %d\tttft %.3f" %
              (a.tag, t, bank, "-" if prev_bank is None else ("yes" if same else "NO"), restored, prompt - restored,
               r["ttft"]), flush=True)
        prev_bank = bank
        last = list(msgs)
        msgs = msgs + [{"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}]
        time.sleep(a.pause)
    warm_again = chat(last)
finally:
    srv.stop()
cold = Server(a.tag + "-cold")
try:
    cold_r = chat(last)
finally:
    cold.stop()
same = (warm_again["content"], warm_again["reasoning"]) == (cold_r["content"], cold_r["reasoning"])
print(("PASS" if same else "FAIL") + " reuse == cold: the last turn re-sent alone == a fresh server's")
if not same:
    print("  warm %r\n  cold %r" % ((warm_again["reasoning"] + "|" + warm_again["content"])[:300],
                                    (cold_r["reasoning"] + "|" + cold_r["content"])[:300]))
ok &= same
later = [r for r in rows if r[0] > 1]
if later:
    print("# %s turns>1: same bank %d/%d, mean restored %.0f, mean prefilled %.0f, mean ttft %.3f s" %
          (a.tag, sum(1 for r in later if r[2]), len(later), sum(r[3] for r in later) / len(later),
           sum(r[4] for r in later) / len(later), sum(r[5] for r in later) / len(later)))
print("opencode-turns-live %s: %s" % (a.tag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

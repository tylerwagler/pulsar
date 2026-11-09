#!/usr/bin/env python3
"""agent_tools_gate.py -- L365: pulsar-agent's tool loop, live, on each model given.

On 2026-11-09 pulsar-agent's tool loop failed on Qwen: the agent described its tools as DeepSeek's DSML and read
only DSML back, so on every other family the model's calls were never run.  The agent now declares its tools in the
family's own template (pulsar_chat_open_tools) and reads calls with the family's output parser -- DeepSeek keeps
DSML.  Nothing exercised that loop end to end: this gate does.

One headless run per model (`pulsar-agent -m MODEL --non-interactive -p ...`, greedy) in a scratch workspace holding
notes/secret.txt.  The prompt asks for a read of that file and a write of its code word into answer.txt, so the turn
needs two tool rounds -- each call rendered by the template, read by the parser, run, its result fed back.
Asserts:
  1. the agent exits 0;
  2. answer.txt holds exactly the code word (the read's result reached the model, the write ran);
  3. the trace shows the read and the write ran as tool calls and no call was malformed.

usage: agent_tools_gate.py PULSAR_AGENT MODEL [MODEL ...]
"""
import os, random, re, shutil, subprocess, sys, tempfile, time

TIMEOUT_S = 1200   # a cold load of the largest container plus two short tool rounds

PROMPT = ("Use your tools. First read the file notes/secret.txt. Then write a file named answer.txt whose entire "
          "content is the code word from it, nothing else. When answer.txt is written, reply DONE.")


def run(binary, model):
    word = "".join(random.Random(model).choice("bcdfghjklmnpqrstvwxz") for _ in range(10))
    work = tempfile.mkdtemp(prefix="agent-tools-gate-")
    ok = False
    try:
        os.makedirs(os.path.join(work, "notes"))
        with open(os.path.join(work, "notes", "secret.txt"), "w") as f:
            f.write("Some notes.\nThe code word is: %s\nEnd of notes.\n" % word)
        trace = os.path.join(work, "trace.log")
        home = os.path.join(work, "home")
        os.makedirs(home)
        cmd = [binary, "-m", model, "--non-interactive", "-p", PROMPT, "--chdir", work, "--trace", trace,
               "--temp", "0", "-n", "4096"]
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_S, env=dict(os.environ, HOME=home))
            rc, out, err = p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired as e:
            rc, out, err = "timeout", (e.stdout or b"").decode(errors="replace"), (e.stderr or b"").decode(errors="replace")
        dt = time.monotonic() - t0
        tr = open(trace, errors="replace").read() if os.path.exists(trace) else ""
        answer = os.path.join(work, "answer.txt")
        got = open(answer, errors="replace").read().strip() if os.path.exists(answer) else None
        calls = re.findall(r"(?:family call \d+|dsml done calls=\d+).*", tr)
        family = "family call" in tr
        ran = lambda name: (re.search(r"family call \d+ %s\b" % name, tr) is not None) if family else True
        malformed = re.findall(r"(?:family call error|dsml error) .*", tr)
        fails = []
        if rc != 0: fails.append("exit %s" % rc)
        if got != word: fails.append("answer.txt is %r, want %r" % (got, word))
        if not ran("read"): fails.append("no read call ran")
        if not ran("write"): fails.append("no write call ran")
        if malformed: fails.append("malformed call: %s" % malformed[0])
        name = os.path.basename(model.rstrip("/"))
        if fails:
            print("FAIL agent tools [%s] %.0f s: %s" % (name, dt, "; ".join(fails)))
            print("---- stdout (tail)\n%s\n---- stderr (tail)\n%s" % (out[-3000:], err[-2000:]))
            print("---- trace calls: %s" % (calls[:8],))
        else:
            ok = True
            print("PASS agent tools [%s] %.0f s: %d tool rounds, answer.txt == the code word" % (name, dt, len(calls)))
    finally:
        if ok: shutil.rmtree(work, ignore_errors=True)
        else: print("workspace kept: %s" % work)
    return ok


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    binary, models = sys.argv[1], sys.argv[2:]
    bad = [m for m in models if not run(binary, m)]
    print("agent_tools_gate: %d models, %d failed -- %s" % (len(models), len(bad), "FAIL" if bad else "PASS"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

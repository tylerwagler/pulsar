#!/usr/bin/env python3
"""L325 prefix-stability replay gate: a client that changes its request only in cache-preserving ways must find its
previous request's KV in the next one's prompt.

    python3 tests/prefix_stability_gate.py [--pulsar-test ./pulsar_test] [--shape NAME ...] [--family F ...]
                                           [--keep DIR] [--list]

Each SHAPE is a sequence of request bodies (Anthropic Messages or OpenAI chat) with, per request, its PARENT (the
earlier request it continues; default the one before).  Every body is rendered by the server's real renderer
(`pulsar_test --render-cases`: the protocol parser + the family's render, no model, no GPU) for every family the shape
names, and for each (parent N, request M) the gate asserts that M's rendered prompt extends N's up to N's STABLE POINT:

  - the end of N's last message that carries a `cache_control` breakpoint (the client's own statement of what it
    expects to be cached), else the end of N's last message;
  - or, where a shape declares `stable_through` = k, the end of N's message k (a family template that by its own
    definition re-renders later turns: Nemotron's truncate_history_thinking drops the reasoning of turns before the
    last user message, L311).

"The end of message k" is measured, not guessed: N is rendered once more with a sentinel appended to message k, and
the stable point is where the sentinel lands, capped by N's own bytes (min(sentinel offset, lcp(N, probe))).  A failure
names the shape, the family, the pair, the first differing byte of the two renders and the bytes around it.

Shapes:
  claude-code-2.1.310-toolsearch  the Part-1 captures (tests/prefix-stability/...): Claude Code 2.1.310's three main-
                                  thread requests around a ToolSearch load (select:WebFetch): WebFetch joins `tools`
                                  with defer_loading, and the tool_result carries a tool_reference block (L325)
  toolsearch-load                 synthetic: a deferred tool loaded by tool_reference, then called (L325)
  reminders-last-user             <system-reminder> text blocks in the last user turn, re-sent as history (L195 shape)
  mid-system                      role "system" messages mid-conversation, the last one carrying the breakpoint (L113)
  suggestion-side-call            Claude Code's SUGGESTION MODE side call after a turn, then the real next turn (L322)
  nemotron-truncate-thinking      opencode on Nemotron: reasoning in history, truncate_history_thinking (L311)

Exit status non-zero on any violation or refusal."""
import argparse, copy, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "prefix-stability")
SENTINEL = "@@L325CUT@@"
ALL_FAMILIES = ["deepseek-v41", "qwen", "nemotron", "inkling"]
CC = {"type": "ephemeral"}


# ---- shapes ----------------------------------------------------------------------------------------------------------

def tool(name, desc, props=None, deferred=False):
    t = {"name": name, "description": desc,
         "input_schema": {"type": "object", "properties": props or {}, "required": sorted((props or {}).keys())}}
    if deferred:
        t["defer_loading"] = True
    return t


BASH = tool("Bash", "Run a shell command.", {"command": {"type": "string", "description": "The command."}})
TOOLSEARCH = tool("ToolSearch", "Fetches full schema definitions for deferred tools so they can be called.",
                  {"query": {"type": "string", "description": "select:<name> or keywords."},
                   "max_results": {"type": "number", "description": "How many."}})
PLACEHOLDER = tool("DeferredToolPlaceholder", "Reserved placeholder that keeps deferred tool loading active; never "
                   "call this tool.", deferred=True)
TASKSTOP = tool("TaskStop", "Stops a running background task by its ID.",
                {"task_id": {"type": "string", "description": "The ID of the background task to stop."}}, deferred=True)
SYSTEM = [{"type": "text", "text": "You are a coding agent.", "cache_control": CC}]
DEFERRED_NOTE = ("<system-reminder>\nThe following deferred tools are now available via ToolSearch:\nTaskStop\n"
                 "</system-reminder>")


def anth(messages, tools=None, system=SYSTEM):
    b = {"model": "claude-sonnet-5", "max_tokens": 1024, "stream": True, "system": system, "messages": messages}
    if tools is not None:
        b["tools"] = tools
    return b


def with_cc_last(msgs):
    """The messages with Claude Code's moving breakpoint: cache_control on the last message's last block."""
    out = copy.deepcopy(msgs)
    last = out[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"][-1]["cache_control"] = CC
    return out


def shape_toolsearch_load():
    m1 = [{"role": "user", "content": [{"type": "text", "text": DEFERRED_NOTE}, {"type": "text", "text": "Stop task 7."}]}]
    call = {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "ToolSearch",
                                              "input": {"query": "select:TaskStop", "max_results": 1}}]}
    loaded = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                           "content": [{"type": "tool_reference", "tool_name": "TaskStop"}]},
                                          {"type": "text", "text": "Tool loaded."}]}
    m2 = m1 + [call, loaded]
    use = {"role": "assistant", "content": [{"type": "text", "text": "Stopping it."},
                                            {"type": "tool_use", "id": "toolu_02", "name": "TaskStop",
                                             "input": {"task_id": "7"}}]}
    done = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_02", "content": "stopped"}]}
    m3 = m2 + [use, done]
    before = [BASH, TOOLSEARCH, PLACEHOLDER]
    after = [BASH, TASKSTOP, TOOLSEARCH, PLACEHOLDER]   # Claude Code inserts the loaded tool in name order
    return {"api": "anthropic", "requests": [{"body": anth(with_cc_last(m1), before)},
                                             {"body": anth(with_cc_last(m2), after)},
                                             {"body": anth(with_cc_last(m3), after)}]}


def shape_reminders_last_user():
    m1 = [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nToday is Friday.\n</system-reminder>"},
                                       {"type": "text", "text": "Summarize the repo."}]}]
    m2 = m1 + [{"role": "assistant", "content": [{"type": "text", "text": "It is an inference engine."}]},
               {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nA file changed.\n"
                                                                      "</system-reminder>"},
                                            {"type": "text", "text": "And the tests?"}]}]
    return {"api": "anthropic", "requests": [{"body": anth(with_cc_last(m1), [BASH])},
                                             {"body": anth(with_cc_last(m2), [BASH])}]}


def shape_mid_system():
    m1 = [{"role": "user", "content": "Fix the build."},
          {"role": "system", "content": [{"type": "text", "text": "<system-reminder>\nTodo list is empty.\n"
                                                                   "</system-reminder>", "cache_control": CC}]}]
    m2 = copy.deepcopy(m1)
    m2[1]["content"] = m2[1]["content"][0]["text"]   # the breakpoint moves on: the block re-sent as a plain string
    m2 += [{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_11", "name": "Bash",
                                               "input": {"command": "make"}}]},
           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_11", "content": "ok"}]},
           {"role": "system", "content": [{"type": "text", "text": "<system-reminder>\nTodo list updated.\n"
                                                                    "</system-reminder>", "cache_control": CC}]}]
    return {"api": "anthropic", "requests": [{"body": anth(m1, [BASH])}, {"body": anth(m2, [BASH])}]}


def shape_suggestion_side_call():
    c = [{"role": "user", "content": "Explain the router."}]
    reply = {"role": "assistant", "content": [{"type": "text", "text": "It picks the bank with the deepest resume."}]}
    side = c + [reply, {"role": "user", "content": "[SUGGESTION MODE: Suggest what the user might want to do next.]"}]
    real = c + [reply, {"role": "user", "content": "Now explain adopt."}]
    return {"api": "anthropic", "requests": [{"body": anth(with_cc_last(c), [BASH])},
                                             {"body": anth(with_cc_last(side), [BASH]), "parent": 0},
                                             {"body": anth(with_cc_last(real), [BASH]), "parent": 0},
                                             # the real turn shares the side call's history through the reply
                                             {"body": anth(with_cc_last(real), [BASH]), "parent": 1,
                                              "stable_through": 1}]}


def shape_nemotron_truncate_thinking():
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run a command.",
                                               "parameters": {"type": "object", "properties": {
                                                   "command": {"type": "string"}}, "required": ["command"]}}}]
    sys_m = {"role": "system", "content": "You are opencode."}
    u1 = {"role": "user", "content": "List the files."}
    a1 = {"role": "assistant", "reasoning_content": "I should run ls.", "content": "",
          "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash",
                                                                           "arguments": "{\"command\": \"ls\"}"}}]}
    t1 = {"role": "tool", "tool_call_id": "call_1", "content": "a.c b.c"}
    a2 = {"role": "assistant", "reasoning_content": "Two files.", "content": "a.c and b.c."}
    u2 = {"role": "user", "content": "Thanks."}

    def oai(msgs):
        return {"model": "nemotron", "messages": msgs, "tools": tools, "stream": True}
    return {"api": "openai", "families": ["nemotron"],
            "requests": [{"body": oai([sys_m, u1])},
                         {"body": oai([sys_m, u1, a1, t1])},
                         # the template drops a1's reasoning once u2 follows: stable through u1 (L311's endpoint)
                         {"body": oai([sys_m, u1, a1, t1, a2, u2]), "stable_through": 1}]}


def shape_captures(name):
    d = os.path.join(FIXTURES, name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    return {"api": "anthropic", "requests": [{"body": json.load(open(os.path.join(d, f)))} for f in files]}


SHAPES = {
    "claude-code-2.1.310-toolsearch": lambda: shape_captures("claude-code-2.1.310-toolsearch"),
    "toolsearch-load": shape_toolsearch_load,
    "reminders-last-user": shape_reminders_last_user,
    "mid-system": shape_mid_system,
    "suggestion-side-call": shape_suggestion_side_call,
    "nemotron-truncate-thinking": shape_nemotron_truncate_thinking,
}


# ---- the gate --------------------------------------------------------------------------------------------------------

def breakpoint_message(body):
    """Index of the last message carrying a cache_control breakpoint, else the last message."""
    msgs = body["messages"]
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("cache_control"):
            return i
        if isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("cache_control")
                                                      for b in m["content"]):
            return i
    return len(msgs) - 1


def with_sentinel(body, k):
    b = copy.deepcopy(body)
    m = b["messages"][k]
    c = m.get("content")
    if isinstance(c, list):
        c.append({"type": "text", "text": SENTINEL})
    else:
        m["content"] = (c or "") + SENTINEL
    return b


def render_all(pulsar_test, bodies):
    """{id: text or ('REFUSED', why)} for a list of (id, api, family, body)."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for cid, api, fam, body in bodies:
            line = dict({"_id": cid, "_api": api, "_family": fam}, **body)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        path = f.name
    try:
        p = subprocess.run([pulsar_test, "--render-cases"], env=dict(os.environ, PULSAR_RENDER_CASES=path),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        os.unlink(path)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        raise SystemExit("prefix-stability-gate: pulsar_test --render-cases failed (exit %d)" % p.returncode)
    out, cur, buf = {}, None, []
    for line in p.stdout.decode("utf-8", "surrogateescape").split("\n"):
        if cur is None and line.startswith("===CASE "):
            head = line[len("===CASE "):-3]
            cid, _, rest = head.partition(" ")
            if rest.startswith("REFUSED"):
                out[int(cid)] = ("REFUSED", rest[len("REFUSED "):])
            else:
                cur, buf = int(cid), []
        elif cur is not None and line == "===END===":
            out[cur] = "\n".join(buf)
            cur = None
        elif cur is not None:
            buf.append(line)
    return out


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    step = 4096
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def show(s, at, w=160):
    return repr(s[max(0, at - w):at]) + " | " + repr(s[at:at + w])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulsar-test", default=os.path.join(ROOT, "pulsar_test"))
    ap.add_argument("--shape", action="append", help="run only these shapes")
    ap.add_argument("--family", action="append", help="run only these families")
    ap.add_argument("--keep", help="write every render to this directory")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(SHAPES))
        return 0
    names = a.shape or list(SHAPES)
    for n in names:
        if n not in SHAPES:
            raise SystemExit("prefix-stability-gate: unknown shape %r (--list)" % n)
    print("prefix-stability-gate: %s (sentinel probe, render-only, no model)" % os.path.realpath(a.pulsar_test))
    jobs, cases, cid = [], [], 0
    for n in names:
        sh = SHAPES[n]()
        fams = [f for f in sh.get("families", ALL_FAMILIES) if not a.family or f in a.family]
        reqs = sh["requests"]
        for fam in fams:
            ids = []
            for r in reqs:
                cases.append((cid, sh["api"], fam, r["body"]))
                ids.append(cid)
                cid += 1
            for i, r in enumerate(reqs):
                if i == 0 and "parent" not in r:
                    continue
                p = r.get("parent", i - 1)
                k = r.get("stable_through", breakpoint_message(reqs[p]["body"]))
                cases.append((cid, sh["api"], fam, with_sentinel(reqs[p]["body"], k)))
                jobs.append((n, fam, p, i, ids[p], ids[i], cid, k))
                cid += 1
    out = render_all(a.pulsar_test, cases)
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
        for c, api, fam, _ in cases:
            v = out.get(c)
            with open(os.path.join(a.keep, "%04d-%s.txt" % (c, fam)), "w", encoding="utf-8",
                      errors="surrogateescape") as f:
                f.write(v if isinstance(v, str) else "REFUSED %s\n" % (v[1] if v else "missing"))
    fails = 0
    for n, fam, p, i, a_id, b_id, probe_id, k in jobs:
        A, B, P = out.get(a_id), out.get(b_id), out.get(probe_id)
        tag = "%s [%s] %d->%d" % (n, fam, p, i)
        bad = [(w, v) for w, v in (("parent", A), ("request", B), ("probe", P)) if not isinstance(v, str)]
        if bad:
            for w, v in bad:
                print("FAIL %s: the %s render was refused: %s" % (tag, w, v[1] if v else "missing"))
            fails += 1
            continue
        s = P.find(SENTINEL)
        if s < 0:
            print("FAIL %s: the sentinel in message %d did not reach the render" % (tag, k))
            fails += 1
            continue
        need = min(s, lcp(A, P))
        got = lcp(A, B)
        if got >= need:
            print("ok   %s: extends %d of %d bytes (stable point %d, end of message %d)" % (tag, got, len(A), need, k))
        else:
            print("FAIL %s: diverges at byte %d, before the stable point %d (end of message %d; parent %d bytes)"
                  % (tag, got, need, k, len(A)))
            print("       parent : %s" % show(A, got))
            print("       request: %s" % show(B, got))
            fails += 1
    print("prefix-stability-gate: %d pairs, %d failed" % (len(jobs), fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

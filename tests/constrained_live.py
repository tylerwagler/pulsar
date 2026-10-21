#!/usr/bin/env python3
"""constrained_live.py -- L303's live gate: structured outputs on a real pulsar-server, every family alike.

    python3 tests/constrained_live.py MODEL [--binary ./pulsar-server] [--port 8197] [--seeds 6] [--think-seeds 2]
                                         [--fail-after 12]

Starts the server (speculation on, its own disk-KV dir under /tmp, removed after), runs the checks, stops it.

  1. VALID: every constrained output is valid against its grammar, over SEEDS seeds at T 0 and T 0.8 with thinking
     off (THINK_SEEDS with it on, for the json answer and the strict and forced calls): a json_schema answer (a small validator below), json_object, a regex, strict tool calls (each call names
     a declared tool and its arguments satisfy the tool's schema), a forced call over three tools (L272's case: a
     declared name, and arguments by the schema), the Responses API's text.format and forced tool_choice, and
     Anthropic's output_config.format.
  2. SPEC == PLAIN: each seeded constrained request is sent twice -- once as is (it speculates: the spec lane, the
     masks on every verify row) and once with logprobs:true (a logprobs request decodes WITHOUT speculation: the
     plain lane, the mask on each draw) -- and the two answers must be byte-identical.  Requests run one at a time.
  3. COST: the tok/s and the spec acceptance of the constrained json answer against the same prompt unconstrained
     (the response's timings), printed (graded by a person, not by this gate).
  4. ISOLATION (--fail-after N > 0): a second server with PULSAR_GRAMMAR_FAIL_AFTER=N -- every gate fails once a
     region has taken N tokens.  Concurrently: a long JSON answer (must fail, by name), a 2-token regex answer (must
     succeed, constrained) and an unconstrained answer (must succeed); then a request on the recovered server.

Exit 0 only when every check passed.  One model load per family (two with --fail-after)."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8197)
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--think-seeds", type=int, default=2, help="seeds for the thinking-on runs (json_schema, strict and "
                "forced tools: the answer after a reasoning block)")
ap.add_argument("--fail-after", type=int, default=12)
ap.add_argument("--start-timeout", type=float, default=900.0)
ap.add_argument("--extra", default="", help="extra server arguments, space separated")
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port
ok = True


def check(c, msg):
    global ok
    ok &= bool(c)
    print(("PASS " if c else "FAIL ") + msg)
    sys.stdout.flush()


class Server:
    def __init__(self, env_extra=None):
        self.dir = tempfile.mkdtemp(prefix="pulsar-constrained-live.")
        self.log_path = os.path.join(self.dir, "server.log")
        self.log = open(self.log_path, "wb")
        env = dict(os.environ)
        env.update(env_extra or {})
        cmd = [a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port", str(a.port),
               "--kv-disk-dir", os.path.join(self.dir, "kv")] + a.extra.split()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                self.log.flush()
                tail = open(self.log_path, "rb").read()[-3000:].decode("utf-8", "replace")
                raise RuntimeError("server exited during startup (rc %s)\n%s" % (self.p.returncode, tail))
            try:
                urllib.request.urlopen(B + "/health", timeout=2).read()
                break
            except Exception:
                pass
            if time.time() - t0 > a.start_timeout:
                raise RuntimeError("server did not listen within %.0f s" % a.start_timeout)
            time.sleep(2)
        print("server up on %s in %.0f s (%s%s)" % (B, time.time() - t0, a.model,
                                                     ", env %s" % env_extra if env_extra else ""))

    def stop(self, failed):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()
        if failed:
            print("--- server log tail ---\n" + open(self.log_path, "rb").read()[-4000:].decode("utf-8", "replace"))
        shutil.rmtree(self.dir, ignore_errors=True)


def post(path, body):
    req = urllib.request.Request(B + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except Exception:
            return e.code, {"error": txt}


# ---- a JSON-schema validator for the subset the gate's schemas use ------------------------------------------
def valid(v, s, path="$"):
    if s is True or s == {}:
        return None
    t = s.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer":
        if not (isinstance(v, int) and not isinstance(v, bool)):
            return "%s: %r is not an integer" % (path, v)
    elif t == "number":
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            return "%s: %r is not a number" % (path, v)
    elif t in types and not isinstance(v, types[t]):
        return "%s: %r is not %s" % (path, v, t)
    if "enum" in s and v not in s["enum"]:
        return "%s: %r not in %s" % (path, v, s["enum"])
    if "minimum" in s and v < s["minimum"]:
        return "%s: %r < %s" % (path, v, s["minimum"])
    if "maximum" in s and v > s["maximum"]:
        return "%s: %r > %s" % (path, v, s["maximum"])
    if t == "string" and "pattern" in s and not re.fullmatch(s["pattern"], v):
        return "%s: %r does not match %s" % (path, v, s["pattern"])
    if t == "array":
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", 1 << 30):
            return "%s: %d items" % (path, len(v))
        for i, x in enumerate(v):
            e = valid(x, s.get("items", True), "%s[%d]" % (path, i))
            if e:
                return e
    if t == "object":
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in v:
                return "%s: missing %s" % (path, k)
        for k, x in v.items():
            if k in props:
                e = valid(x, props[k], path + "." + k)
                if e:
                    return e
            elif s.get("additionalProperties", True) is False:
                return "%s: undeclared %s" % (path, k)
    return None


DOC = {"type": "object", "additionalProperties": False, "required": ["city", "country", "population", "tags", "capital"],
       "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                      "population": {"type": "integer", "minimum": 0},
                      "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                      "capital": {"type": "boolean"}}}
WEATHER = {"type": "object", "additionalProperties": False, "required": ["city", "unit"],
           "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}
SEARCH = {"type": "object", "additionalProperties": False, "required": ["query", "limit"],
          "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}}
TIME = {"type": "object", "additionalProperties": False, "required": ["timezone"],
        "properties": {"timezone": {"type": "string"}}}
SCHEMAS = {"get_weather": WEATHER, "search": SEARCH, "get_time": TIME}


def tools(strict):
    return [{"type": "function", "function": {"name": n, "description": d, "strict": strict, "parameters": SCHEMAS[n]}}
            for n, d in (("get_weather", "Current weather for a city"), ("search", "Web search"),
                         ("get_time", "The time in a timezone"))]


def user(text):
    return [{"role": "user", "content": text}]


def thinking(on):
    return {} if on else {"reasoning_effort": "none"}


def chat(body, seed, temp, plain):
    b = dict(body, model="m", seed=seed, temperature=temp)
    if plain:
        b["logprobs"] = True
    return post("/v1/chat/completions", b)


def outcome(j):
    """What the two lanes must agree on: the text, the calls (their ids are per-response) and the finish."""
    ch = j["choices"][0]
    m = ch["message"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
    return (m.get("content") or "", json.dumps(calls, sort_keys=True), ch.get("finish_reason"))


def grade_calls(m, forced):
    calls = m.get("tool_calls") or []
    if forced and not calls:
        return "a forced turn made no call"
    for c in calls:
        f = c["function"]
        if f["name"] not in SCHEMAS:
            return "undeclared tool %r" % f["name"]
        try:
            args = json.loads(f["arguments"])
        except Exception as e:
            return "arguments are not JSON: %r (%s)" % (f["arguments"], e)
        e = valid(args, SCHEMAS[f["name"]])
        if e:
            return "%s: %s" % (f["name"], e)
    return None


CASES = [
    # (name, body, grader(message) -> error or None)
    ("json_schema", lambda th: dict(max_tokens=600 if th else 200, messages=user(
        "Describe Lyon as a JSON record."), response_format={"type": "json_schema", "json_schema": {
        "name": "city", "strict": True, "schema": DOC}}, **thinking(th)),
     lambda m: (lambda c: valid(json.loads(c), DOC) if c.strip().startswith("{") else "not JSON: %r" % c[:80])(
         m.get("content") or "")),
    ("json_object", lambda th: dict(max_tokens=600 if th else 160, messages=user(
        "Give me three facts about Mars as a JSON object."), response_format={"type": "json_object"}, **thinking(th)),
     lambda m: None if isinstance(json.loads(m.get("content") or "null"), dict) else "not an object"),
    ("regex", lambda th: dict(max_tokens=400 if th else 24, messages=user(
        "Invent a flight number."), response_format={"type": "regex", "pattern": "[A-Z]{2}[0-9]{3,4}"},
        **thinking(th)),
     lambda m: None if re.fullmatch(r"[A-Z]{2}[0-9]{3,4}", (m.get("content") or "").strip()) else
     "%r does not match" % m.get("content")),
    ("strict tools", lambda th: dict(max_tokens=600 if th else 200, tools=tools(True), messages=user(
        "What's the weather in Paris right now, in celsius? Use a tool."), **thinking(th)),
     lambda m: grade_calls(m, False)),
    ("forced required", lambda th: dict(max_tokens=200, tools=tools(True), tool_choice="required",
                                        messages=user("Tell me a joke."), **thinking(th)),
     lambda m: grade_calls(m, True)),
    ("forced non-strict", lambda th: dict(max_tokens=200, tools=tools(False), tool_choice="required",
                                          messages=user("What time is it in Tokyo?"), **thinking(th)),
     lambda m: (lambda c: "no call" if not c else None if c[0]["function"]["name"] in SCHEMAS else "undeclared")(
         m.get("tool_calls") or [])),
]


CONTROL = {"diff": 0, "n": 0}


def run_control():
    """The same long answer UNCONSTRAINED, spec against plain: whether this model's speculative lane is byte-exact
    against its plain lane without any grammar.  It is not on every model -- past the M-independent width (16 rows)
    a verify row may take the faster GEMM arms (ENGINEERING-RULES 7, graded, not byte-gated), and a draw at a near
    tie can then differ -- so a constrained mismatch is the grammar's only where the control is exact."""
    for seed in range(1, a.seeds + 1):
        for temp in (0.0, 0.8):
            body = dict(max_tokens=160, messages=user("Give me three facts about Mars as a JSON object."),
                        **thinking(False))
            st, j = chat(body, seed, temp, False)
            st2, j2 = chat(body, seed, temp, True)
            CONTROL["n"] += 1
            if st != 200 or st2 != 200 or outcome(j) != outcome(j2):
                CONTROL["diff"] += 1
    print("CONTROL unconstrained spec == plain %d/%d%s" % (CONTROL["n"] - CONTROL["diff"], CONTROL["n"],
          "" if not CONTROL["diff"] else " -- this model's spec lane is not byte-exact against plain without a grammar"))


def run_valid_and_exact():
    run_control()
    for name, body_fn, grade in CASES:
        for th in (False, True):
            if th and name not in ("json_schema", "strict tools", "forced required"):
                continue
            bad = 0
            diff = 0
            n = 0
            cut = 0
            accepts = []   # rule 5: the spec responses say they speculated (their acceptance), or the lane did not run
            for seed in range(1, (a.think_seeds if th else a.seeds) + 1):
                for temp in (0.0, 0.8):
                    body = body_fn(th)
                    st, j = chat(body, seed, temp, False)
                    if st != 200:
                        check(False, "%s think=%s seed=%d T=%.1f -> HTTP %d %s" % (name, th, seed, temp, st,
                                                                                  json.dumps(j)[:200]))
                        bad += 1
                        continue
                    n += 1
                    if "spec_accept_rate" in j.get("timings", {}):
                        accepts.append(j["timings"]["spec_accept_rate"])
                    m = j["choices"][0]["message"]
                    try:
                        e = grade(m)
                    except Exception as ex:
                        e = "%s: %s" % (type(ex).__name__, ex)
                    if e and j["choices"][0].get("finish_reason") == "length":
                        cut += 1   # cut by max_tokens: a prefix, not a verdict (a fixture's babble never closes)
                    elif e:
                        bad += 1
                        print("  invalid %s think=%s seed=%d T=%.1f: %s | %r %r" % (
                            name, th, seed, temp, e, (m.get("content") or "")[:160], m.get("tool_calls")))
                    st2, j2 = chat(body, seed, temp, True)
                    if st2 != 200 or outcome(j2) != outcome(j):
                        diff += 1
                        print("  spec != plain %s think=%s seed=%d T=%.1f:\n    spec  %r\n    plain %r" % (
                            name, th, seed, temp, outcome(j), outcome(j2) if st2 == 200 else (st2, j2)))
            check(bad == 0, "%s think=%s: %d/%d valid over seeds x T{0,0.8} (%d cut by max_tokens)" % (
                name, th, n - bad - cut, n, cut))
            check(diff == 0 or CONTROL["diff"] > 0,
                  "%s think=%s: spec == plain byte-exact %d/%d (speculated in %d, mean acceptance %s)%s" % (
                      name, th, n - diff, n, len(accepts), "%.3f" % (sum(accepts) / len(accepts)) if accepts else "-",
                      " -- not graded: the unconstrained control differs too" if diff and CONTROL["diff"] else ""))


def json_valid(txt, schema):
    try:
        return valid(json.loads(txt), schema)
    except Exception as e:
        return "%s: %r" % (type(e).__name__, txt[:120])


def run_protocols():
    st, j = post("/v1/responses", {"model": "m", "input": "Describe Lyon as a JSON record.", "reasoning": {
        "effort": "none"}, "max_output_tokens": 200, "text": {"format": {"type": "json_schema", "name": "city",
                                                                          "strict": True, "schema": DOC}}})
    txt = "".join(c.get("text", "") for o in j.get("output", []) if o.get("type") == "message"
                  for c in o.get("content", []))
    e = json_valid(txt, DOC) if st == 200 else "HTTP %d %r" % (st, json.dumps(j)[:120])
    check(e is None or j.get("status") == "incomplete", "responses text.format json_schema -> %s" % (e or txt[:100]))
    st, j = post("/v1/responses", {"model": "m", "input": "Hi", "reasoning": {"effort": "none"},
                                   "max_output_tokens": 200, "tools": [
                                       {"type": "function", "name": "search", "strict": True, "parameters": SEARCH}],
                                   "tool_choice": {"type": "function", "name": "search"}})
    fc = [o for o in j.get("output", []) if o.get("type") == "function_call"]
    e = (json_valid(fc[0]["arguments"], SEARCH) if fc[0]["name"] == "search" else "named %s" % fc[0]["name"]) if fc \
        else "HTTP %d %s" % (st, json.dumps(j)[:200])
    check(e is None or j.get("status") == "incomplete", "responses forced tool_choice -> %s" % (e or fc[0]["arguments"]))
    st, j = post("/v1/messages", {"model": "m", "max_tokens": 200, "messages": user("Describe Lyon as a JSON record."),
                                  "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": DOC}}})
    txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    e = json_valid(txt, DOC) if st == 200 else "HTTP %d %r" % (st, json.dumps(j)[:120])
    check(e is None or j.get("stop_reason") == "max_tokens", "anthropic output_config.format -> %s" % (e or txt[:100]))


def run_cost():
    """The constrained answer against the same prompt unconstrained: decode tok/s and spec acceptance."""
    rows = []
    for label, rf in (("unconstrained", None), ("json_schema", {"type": "json_schema", "json_schema": {
            "name": "city", "schema": DOC}})):
        tps, acc = [], []
        for seed in range(1, 4):
            body = dict(model="m", seed=seed, temperature=0.8, max_tokens=200, reasoning_effort="none",
                        messages=user("Describe Lyon as a JSON record with city, country, population, tags, capital."))
            if rf:
                body["response_format"] = rf
            st, j = post("/v1/chat/completions", body)
            t = j.get("timings", {}) if st == 200 else {}
            if "predicted_per_second" in t:
                tps.append(t["predicted_per_second"])
            if "spec_accept_rate" in t:
                acc.append(t["spec_accept_rate"])
        rows.append((label, sum(tps) / max(len(tps), 1), sum(acc) / max(len(acc), 1)))
    for label, tps, acc in rows:
        print("COST %-14s decode %.2f tok/s  spec accept %.3f" % (label, tps, acc))


def run_isolation():
    srv = Server({"PULSAR_GRAMMAR_FAIL_AFTER": str(a.fail_after)})
    res = {}

    def go(key, body):
        res[key] = post("/v1/chat/completions", dict(body, model="m", seed=7, temperature=0.8))

    long_json = dict(max_tokens=300, reasoning_effort="none", messages=user("Describe Lyon as a JSON record."),
                     response_format={"type": "json_schema", "json_schema": {"name": "c", "schema": DOC}})
    short_rx = dict(max_tokens=24, reasoning_effort="none", messages=user("Pick a digit."),
                    response_format={"type": "regex", "pattern": "[0-9]"})
    plain = dict(max_tokens=60, reasoning_effort="none", messages=user("Name three rivers in France."))
    th = [threading.Thread(target=go, args=(k, b)) for k, b in (("long", long_json), ("short", short_rx),
                                                                 ("plain", plain))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    st, j = res["long"]
    check(st >= 400 and "injected" in json.dumps(j), "isolation: the failing grammar's request fails by name (HTTP %d %s)"
          % (st, json.dumps(j)[:160]))
    st, j = res["short"]
    check(st == 200 and re.fullmatch("[0-9]", (j["choices"][0]["message"].get("content") or "").strip()),
          "isolation: a concurrent constrained request completes -> %s" % (json.dumps(j)[:120]))
    st, j = res["plain"]
    check(st == 200 and len(j["choices"][0]["message"].get("content") or "") > 10,
          "isolation: a concurrent unconstrained request completes")
    st, j = post("/v1/chat/completions", dict(plain, model="m", seed=3, temperature=0.0))
    check(st == 200 and len(j["choices"][0]["message"].get("content") or "") > 10,
          "isolation: the server serves after the failure")
    srv.stop(not ok)


failed = False
try:
    srv = Server()
    try:
        run_valid_and_exact()
        run_protocols()
        run_cost()
    except Exception as e:
        check(False, "raised %s: %s" % (type(e).__name__, e))
    srv.stop(not ok)
    if a.fail_after > 0:
        run_isolation()
except Exception as e:
    check(False, "server: %s" % e)
print("constrained-live: %s (%s)" % ("PASS" if ok else "FAIL", a.model))
sys.exit(0 if ok else 1)

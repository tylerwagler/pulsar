#!/usr/bin/env python3
"""constrained_live.py -- L303's live gate: structured outputs on a real pulsar-server, every family alike.

    python3 tests/constrained_live.py MODEL [--binary ./pulsar-server] [--port 8197] [--seeds 6] [--think-seeds 2]
                                         [--fail-after 12]

  1. VALID: every constrained output is valid against its grammar, over SEEDS seeds at T 0 and T 0.8 with thinking
     off (THINK_SEEDS with it on, for the json answer and the strict and forced calls): a json_schema answer (a small
     validator below), json_object, a regex, strict tool calls (each call names a declared tool and its arguments
     satisfy the tool's schema), a forced call over three tools (L272's case: a declared name, and arguments by the
     schema), the Responses API's text.format and forced tool_choice, and Anthropic's output_config.format.
  2. SPEC == PLAIN: each seeded constrained request is sent twice -- once as is (it speculates: the spec lane, the
     masks on every verify row) and once with logprobs:true (a logprobs request decodes WITHOUT speculation: the
     plain lane, the mask on each draw) -- and the two answers must be byte-identical.  Requests run one at a time.
  3. COST: the tok/s and the spec acceptance of the constrained json answer against the same prompt unconstrained
     (the response's timings), printed (graded by a person, not by this gate).
  4. ISOLATION (--fail-after N > 0): on a server with PULSAR_GRAMMAR_FAIL_AFTER=N -- every gate fails once a region
     has taken N tokens.  Concurrently: a long JSON answer (must fail, by name), a 2-token regex answer (must
     succeed, constrained) and an unconstrained answer (must succeed); then a request on the recovered server.

L317: a leg of tests/live_legs.py -- 1-3 on the model's shared server (warm), 4 on the model's hooks instance (the
one server-wide test hook it needs).  Alone (this command line) it runs on instances of its own.  Exit 0 only when
every check passed."""
import json, os, re, sys, threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg  # noqa: E402


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


def chat(srv, body, seed, temp, plain):
    b = dict(body, model="m", seed=seed, temperature=temp)
    if plain:
        b["logprobs"] = True
    return srv.post_json("/v1/chat/completions", b)


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



def json_valid(txt, schema):
    try:
        return valid(json.loads(txt), schema)
    except Exception as e:
        return "%s: %r" % (type(e).__name__, txt[:120])


class ConstrainedLeg(Leg):
    NAME = "constrained"

    def __init__(self, g, work, seeds=6, think_seeds=2, fail_after=12):
        Leg.__init__(self, g, work)
        self.seeds, self.think_seeds, self.fail_after = seeds, think_seeds, fail_after
        self.control = {"diff": 0, "n": 0}

    def warm(self, srv):
        self.srv = srv
        self.run_valid_and_exact()
        self.run_protocols()
        self.run_cost()
        if not self.g.ok:
            print("--- server log tail ---\n" + srv.tail(4000))

    def hook_env(self):
        return {"PULSAR_GRAMMAR_FAIL_AFTER": str(self.fail_after)} if self.fail_after > 0 else None

    def run_control(self):
        """The same long answer UNCONSTRAINED, spec against plain: whether this model's speculative lane is byte-exact
        against its plain lane without any grammar.  It is not on every model -- past the M-independent width (16 rows)
        a verify row may take the faster GEMM arms (ENGINEERING-RULES 7, graded, not byte-gated), and a draw at a near
        tie can then differ -- so a constrained mismatch is the grammar's only where the control is exact."""
        for seed in range(1, self.seeds + 1):
            for temp in (0.0, 0.8):
                body = dict(max_tokens=160, messages=user("Give me three facts about Mars as a JSON object."),
                            **thinking(False))
                st, j = chat(self.srv, body, seed, temp, False)
                st2, j2 = chat(self.srv, body, seed, temp, True)
                self.control["n"] += 1
                if st != 200 or st2 != 200 or outcome(j) != outcome(j2):
                    self.control["diff"] += 1
        c = self.control
        print("CONTROL unconstrained spec == plain %d/%d%s" % (c["n"] - c["diff"], c["n"], "" if not c["diff"] else
              " -- this model's spec lane is not byte-exact against plain without a grammar"))

    def run_valid_and_exact(self):
        self.run_control()
        for name, body_fn, grade in CASES:
            for th in (False, True):
                if th and name not in ("json_schema", "strict tools", "forced required"):
                    continue
                bad = diff = n = cut = 0
                accepts = []   # rule 5: the spec responses say they speculated (their acceptance), or the lane did not run
                for seed in range(1, (self.think_seeds if th else self.seeds) + 1):
                    for temp in (0.0, 0.8):
                        body = body_fn(th)
                        st, j = chat(self.srv, body, seed, temp, False)
                        if st != 200:
                            self.g.check(False, "%s think=%s seed=%d T=%.1f -> HTTP %d %s" % (
                                name, th, seed, temp, st, json.dumps(j)[:200]))
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
                        st2, j2 = chat(self.srv, body, seed, temp, True)
                        if st2 != 200 or outcome(j2) != outcome(j):
                            diff += 1
                            print("  spec != plain %s think=%s seed=%d T=%.1f:\n    spec  %r\n    plain %r" % (
                                name, th, seed, temp, outcome(j), outcome(j2) if st2 == 200 else (st2, j2)))
                self.g.check(bad == 0, "%s think=%s: %d/%d valid over seeds x T{0,0.8} (%d cut by max_tokens)" % (
                    name, th, n - bad - cut, n, cut))
                cd = self.control["diff"]
                self.g.check(diff == 0 or cd > 0,
                             "%s think=%s: spec == plain byte-exact %d/%d (speculated in %d, mean acceptance %s)%s" % (
                                 name, th, n - diff, n, len(accepts),
                                 "%.3f" % (sum(accepts) / len(accepts)) if accepts else "-",
                                 " -- not graded: the unconstrained control differs too" if diff and cd else ""))

    def run_protocols(self):
        post = self.srv.post_json
        st, j = post("/v1/responses", {"model": "m", "input": "Describe Lyon as a JSON record.", "reasoning": {
            "effort": "none"}, "max_output_tokens": 200, "text": {"format": {"type": "json_schema", "name": "city",
                                                                              "strict": True, "schema": DOC}}})
        txt = "".join(c.get("text", "") for o in j.get("output", []) if o.get("type") == "message"
                      for c in o.get("content", []))
        e = json_valid(txt, DOC) if st == 200 else "HTTP %d %r" % (st, json.dumps(j)[:120])
        self.g.check(e is None or j.get("status") == "incomplete", "responses text.format json_schema -> %s" % (e or txt[:100]))
        st, j = post("/v1/responses", {"model": "m", "input": "Hi", "reasoning": {"effort": "none"},
                                       "max_output_tokens": 200, "tools": [
                                           {"type": "function", "name": "search", "strict": True, "parameters": SEARCH}],
                                       "tool_choice": {"type": "function", "name": "search"}})
        fc = [o for o in j.get("output", []) if o.get("type") == "function_call"]
        e = (json_valid(fc[0]["arguments"], SEARCH) if fc[0]["name"] == "search" else "named %s" % fc[0]["name"]) if fc \
            else "HTTP %d %s" % (st, json.dumps(j)[:200])
        self.g.check(e is None or j.get("status") == "incomplete",
                     "responses forced tool_choice -> %s" % (e or fc[0]["arguments"]))
        st, j = post("/v1/messages", {"model": "m", "max_tokens": 200, "messages": user("Describe Lyon as a JSON record."),
                                      "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": DOC}}})
        txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        e = json_valid(txt, DOC) if st == 200 else "HTTP %d %r" % (st, json.dumps(j)[:120])
        self.g.check(e is None or j.get("stop_reason") == "max_tokens",
                     "anthropic output_config.format -> %s" % (e or txt[:100]))

    def run_cost(self):
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
                st, j = self.srv.post_json("/v1/chat/completions", body)
                t = j.get("timings", {}) if st == 200 else {}
                if "predicted_per_second" in t:
                    tps.append(t["predicted_per_second"])
                if "spec_accept_rate" in t:
                    acc.append(t["spec_accept_rate"])
            rows.append((label, sum(tps) / max(len(tps), 1), sum(acc) / max(len(acc), 1)))
        for label, tps, acc in rows:
            print("COST %-14s decode %.2f tok/s  spec accept %.3f" % (label, tps, acc))

    def hooked(self, srv):
        """ISOLATION: on the instance whose every grammar fails after fail_after tokens (PULSAR_GRAMMAR_FAIL_AFTER), a
        long JSON answer fails by name while a short regex answer and an unconstrained one, concurrent, complete; then
        the server serves on."""
        res = {}

        def go(key, body):
            res[key] = srv.post_json("/v1/chat/completions", dict(body, model="m", seed=7, temperature=0.8))

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
        self.g.check(st >= 400 and "injected" in json.dumps(j),
                     "isolation: the failing grammar's request fails by name (HTTP %d %s)" % (st, json.dumps(j)[:160]))
        st, j = res["short"]
        self.g.check(st == 200 and re.fullmatch("[0-9]", (j["choices"][0]["message"].get("content") or "").strip()),
                     "isolation: a concurrent constrained request completes -> %s" % (json.dumps(j)[:120]))
        st, j = res["plain"]
        self.g.check(st == 200 and len(j["choices"][0]["message"].get("content") or "") > 10,
                     "isolation: a concurrent unconstrained request completes")
        st, j = srv.post_json("/v1/chat/completions", dict(plain, model="m", seed=3, temperature=0.0))
        self.g.check(st == 200 and len(j["choices"][0]["message"].get("content") or "") > 10,
                     "isolation: the server serves after the failure")


def _args(ap):
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--think-seeds", type=int, default=2, help="seeds for the thinking-on runs (json_schema, strict "
                    "and forced tools: the answer after a reasoning block)")
    ap.add_argument("--fail-after", type=int, default=12)


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(ConstrainedLeg, sys.argv[1:], ctx=16384, port=8197, add_args=_args))

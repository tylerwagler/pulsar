import json, urllib.request, time

BASE = "http://sparky.defense.lan:8099"

def chat(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)

# --- Smoke 1: canonicalization -> cross-spelling warm-prefix hits ---------
# Same toolset, two JSON spellings.  With canonical rendering both requests
# produce byte-identical tool schema prompts; request B should find request
# A's committed prefix (visible in the server's route log as a large common).
tool_compact = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get weather — current conditions",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
msgs = [{"role": "user", "content": "Say READY and stop."}]
r1 = json.load(chat({"model": "deepseek-v4-flash", "messages": msgs,
                     "tools": [tool_compact], "max_tokens": 30,
                     "temperature": 0, "reasoning_effort": "none"}))
print("A finish:", r1["choices"][0]["finish_reason"])

# spelling B: reordered whitespace via manual serialization is hard from
# python (json.dumps normalizes) -- so build the raw body string by hand
# with extra spaces inside the tools array.
raw_body = ('{"model":"deepseek-v4-flash","max_tokens":30,"temperature":0,'
            '"reasoning_effort":"none",'
            '"messages":[{"role":"user","content":"Say READY and stop."}],'
            '"tools":[ { "type" : "function", "function" : { '
            '"name" : "get_weather", '
            '"description" : "Get weather \\u2014 current conditions", '
            '"parameters" : { "type" : "object", "properties" : '
            '{ "city" : { "type" : "string" } }, "required" : [ "city" ] } } } ]}')
req = urllib.request.Request(BASE + "/v1/chat/completions", raw_body.encode(),
                             {"Content-Type": "application/json"})
r2 = json.load(urllib.request.urlopen(req, timeout=300))
print("B finish:", r2["choices"][0]["finish_reason"])

# --- Smoke 2: truncated streamed tool call closes its args JSON -----------
body = {"model": "deepseek-v4-flash",
        "messages": [{"role": "user",
                      "content": "Use the bash tool to list every file under "
                                 "/var/log with full details, long form."}],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}],
        "stream": True, "max_tokens": 21, "temperature": 0,
        "reasoning_effort": "none"}
resp = chat(body)
events = resp.read().decode()
frags = []
finish = None
for line in events.splitlines():
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    ch = d["choices"][0]
    if ch.get("finish_reason"):
        finish = ch["finish_reason"]
    for tc in (ch["delta"].get("tool_calls") or []):
        f = tc.get("function", {})
        if "arguments" in f:
            frags.append(f["arguments"])
args = "".join(frags)
print("streamed args:", repr(args))
print("finish:", finish)
try:
    parsed = json.loads(args)
    print("ARGS JSON: WELL-FORMED ->", parsed)
except Exception as e:
    print("ARGS JSON: BROKEN:", e)

import json, urllib.request

BASE = "http://sparky.defense.lan:8099"
body = {"model": "deepseek-v4-flash",
        "messages": [{"role": "user",
                      "content": "Use the bash tool to list every file under "
                                 "/var/log with full details, long form."}],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}],
        "stream": True, "max_tokens": 34, "temperature": 0,
        "reasoning_effort": "none"}
req = urllib.request.Request(BASE + "/v1/chat/completions",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
events = urllib.request.urlopen(req, timeout=300).read().decode()
frags = []
for line in events.splitlines():
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    for tc in (d["choices"][0]["delta"].get("tool_calls") or []):
        f = tc.get("function", {})
        if "arguments" in f:
            frags.append(f["arguments"])
for i, f in enumerate(frags):
    print(i, repr(f), f.encode("utf-8").hex())
src = "</｜DSML｜"
print("expected long-close prefix:", src.encode("utf-8").hex())

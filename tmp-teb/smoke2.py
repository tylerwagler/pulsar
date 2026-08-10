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
        "stream": True, "temperature": 0, "reasoning_effort": "none"}

for mt in (26, 30, 34, 38):
    body["max_tokens"] = mt
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    events = urllib.request.urlopen(req, timeout=300).read().decode()
    frags, finish = [], None
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
    try:
        json.loads(args)
        ok = "WELL-FORMED"
    except Exception:
        ok = "BROKEN"
    print(f"max_tokens={mt}: args={args!r} frags={len(frags)} finish={finish} -> {ok}")

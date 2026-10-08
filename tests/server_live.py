#!/usr/bin/env python3
"""server_live.py -- the live server tier (L278 contract 6): a real pulsar-server on a model, driven over HTTP.

    python3 tests/server_live.py MODEL [--binary ./pulsar-server] [--port 8199] [--ctx 32768]

Starts the server on 127.0.0.1:PORT (its own disk-KV dir under /tmp, removed after), waits for it to listen, runs
the checks below, stops it.  Every check is family-neutral -- the same requests for DeepSeek and Qwen; the
family's own renderer and parser are what is being graded.  Born as L272's ~/p3_live.py (the probe that found
the undeclared-tool hole), here with its own server so it is reproducible.

  1. a plain chat answer ("7+5" -> "12"), non-streamed
  2. Anthropic forced tool_choice, non-streamed: a declared tool_use with its argument
  3. the same streamed: a tool_use block start + input_json_delta events carrying the arguments
  4. the tool_result continuation uses the result
  5. OpenAI tool_choice="required" with one tool: a call to the declared tool
  6. OpenAI named tool_choice for an UNDECLARED tool: refused 400 by name
  7. "required" with two tools, 5 sampled turns: every call names a declared tool
  8. (L268) an image: OpenAI image_url -- the model names the shape's colour
  9. (L268) an image inside an Anthropic tool_result (an agent's screenshot) -- the same answer
 10. (L268) the same screenshot after a SAMPLED call: the tool_result image continues the live KV (the server
     logs the anthropic live continuation and places the image on the live history), and the answer is right
 11. (L268/L281) a disk chain over an image serves the next request that brings the SAME image on a fresh bank:
     the server logs a chain hit past the image block (its text is the placeholder's, every family) and answers
 12. (L284) /v1/completions continues the prompt raw (no template, no thinking); think:true there is refused 400
 13. (L284) every effort name of the one table (and an integer) serves; thinking on + effort none is refused 400
 14. (L284) an OpenAI streamed call's arguments arrive in fragments as read, adding up to the object

Not in `make gates` (one server load per family): `make server-live-gate` runs it on FRONTIER_MODEL and the
hosted Qwen model.  Exit 0 only when every check passed."""
import argparse, base64, json, os, shutil, struct, subprocess, sys, tempfile, time, urllib.error, urllib.request, zlib

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--binary", default="./pulsar-server")
ap.add_argument("--port", type=int, default=8199)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--start-timeout", type=float, default=900.0)
a = ap.parse_args()
B = "http://127.0.0.1:%d" % a.port

kvdir = tempfile.mkdtemp(prefix="pulsar-server-live.")
log_path = os.path.join(kvdir, "server.log")
log = open(log_path, "wb")
srv = subprocess.Popen([a.binary, "-m", a.model, "--ctx", str(a.ctx), "--host", "127.0.0.1", "--port", str(a.port),
                        "--kv-disk-dir", os.path.join(kvdir, "kv")], stdout=log, stderr=subprocess.STDOUT)


def stop(rc):
    if srv.poll() is None:
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
    log.close()
    if rc != 0:
        tail = open(log_path, "rb").read()[-3000:].decode("utf-8", "replace")
        print("--- server log tail ---\n" + tail)
    shutil.rmtree(kvdir, ignore_errors=True)
    sys.exit(rc)


t0 = time.time()
while True:
    if srv.poll() is not None:
        print("FAIL server exited during startup (rc %s)" % srv.returncode)
        stop(1)
    try:
        urllib.request.urlopen(B + "/health", timeout=2).read()
        break
    except Exception:
        pass
    if time.time() - t0 > a.start_timeout:
        print("FAIL server did not listen within %.0f s" % a.start_timeout)
        stop(1)
    time.sleep(2)
print("server up on %s in %.0f s (%s)" % (B, time.time() - t0, a.model))


def post(path, body):
    req = urllib.request.Request(B + path, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    return r.read().decode(), time.time() - t


ok = True


def check(c, msg):
    global ok
    ok &= bool(c)
    print(("PASS " if c else "FAIL ") + msg)
    sys.stdout.flush()


def guarded(name, fn):
    try:
        fn()
    except Exception as e:  # a check that throws is a failed check, said by name
        check(False, "%s raised %s: %s" % (name, type(e).__name__, e))


def red_circle_png(w=320, h=240):
    """A red disc on white, as a PNG (zlib only) -- an image whose content a sane vision model cannot miss."""
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            inside = (x - w / 2) ** 2 + (y - h / 2) ** 2 < (min(w, h) * 0.35) ** 2
            row += bytes((220, 20, 20) if inside else (255, 255, 255))
        rows.append(bytes(row))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


IMG_B64 = base64.b64encode(red_circle_png()).decode()
TOOLS = [{"name": "get_weather", "description": "Current weather for a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
OT = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                        "parameters": TOOLS[0]["input_schema"]}}]
OT2 = OT + [{"type": "function", "function": {"name": "search", "description": "Web search", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
msgs = [{"role": "user", "content": "I'm heading to Paris tomorrow."}]
state = {}


def c1():
    d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 64, "temperature": 0,
                                          "messages": [{"role": "user",
                                                        "content": "What is 7+5? Answer with just the number."}]})
    m = json.loads(d)["choices"][0]["message"]
    check("12" in (m.get("content") or ""), "plain chat answer -> %r (%.1fs)" % ((m.get("content") or "")[:80], dt))


def c2():
    d, dt = post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": TOOLS, "messages": msgs,
                                  "tool_choice": {"type": "tool", "name": "get_weather"}})
    j = json.loads(d)
    tu = [b for b in j["content"] if b["type"] == "tool_use"]
    state["j"], state["tu"] = j, tu
    check(tu and tu[0]["name"] == "get_weather" and "city" in tu[0]["input"],
          "anthropic forced tool_choice -> %s stop=%s (%.1fs)" % (tu[0]["input"] if tu else j["content"],
                                                                   j.get("stop_reason"), dt))


def c3():
    d, dt = post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": TOOLS, "messages": msgs, "stream": True,
                                  "tool_choice": {"type": "tool", "name": "get_weather"}})
    ev = [json.loads(l[6:]) for l in d.splitlines() if l.startswith("data: ")]
    starts = [e for e in ev if e.get("type") == "content_block_start" and e["content_block"]["type"] == "tool_use"]
    deltas = [e for e in ev if e.get("type") == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
    args = "".join(e["delta"]["partial_json"] for e in deltas)
    check(starts and deltas and "city" in json.loads(args or "{}"),
          "anthropic stream: tool_use start + %d input_json_delta -> %s" % (len(deltas), args))


def c4():
    if not state.get("tu"):
        check(False, "tool_result continuation: no call from check 2 to continue")
        return
    call = state["tu"][0]
    msgs2 = msgs + [{"role": "assistant", "content": state["j"]["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                                  "content": "18C, light rain"}]}]
    d, dt = post("/v1/messages", {"model": "m", "max_tokens": 600, "tools": TOOLS, "messages": msgs2})
    j2 = json.loads(d)
    txt = "".join(b.get("text", "") for b in j2["content"] if b["type"] == "text")
    check("18" in txt or "rain" in txt.lower(),
          "anthropic tool_result continuation -> %r (%.1fs, usage %s)" % (txt[:120], dt, j2.get("usage")))


def c5():
    d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT, "tool_choice": "required",
                                          "messages": [{"role": "user", "content": "Tell me a joke."}]})
    ch = json.loads(d)["choices"][0]
    tc = ch["message"].get("tool_calls") or []
    check(tc and tc[0]["function"]["name"] == "get_weather",
          "openai tool_choice=required -> %s finish=%s" % (tc[0]["function"] if tc else ch["message"],
                                                           ch["finish_reason"]))


def c6():
    try:
        post("/v1/chat/completions", {"model": "m", "max_tokens": 50, "tools": OT,
                                      "tool_choice": {"type": "function", "function": {"name": "reply"}},
                                      "messages": [{"role": "user", "content": "Hi"}]})
        check(False, "named undeclared tool_choice should refuse")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        check(e.code == 400 and "not a declared tool" in body, "named undeclared tool_choice -> %d %s" % (e.code, body[:120]))


def c7():
    names = []
    for _ in range(5):
        d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT2, "tool_choice": "required",
                                              "temperature": 1.0,
                                              "messages": [{"role": "user", "content": "Tell me a joke."}]})
        ch = json.loads(d)["choices"][0]
        tc = ch["message"].get("tool_calls") or []
        names.append(tc[0]["function"]["name"] if tc else "(none:" + ch["finish_reason"] + ")")
    check(all(n in ("get_weather", "search") for n in names), "required, two tools, 5 sampled turns -> %s" % names)


def c8():
    d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 1024, "temperature": 0, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + IMG_B64}},
            {"type": "text", "text": "What color is the circle in this image? Answer with one word."}]}]})
    txt = json.loads(d)["choices"][0]["message"].get("content") or ""
    check("red" in txt.lower(), "openai image_url -> %r (%.1fs)" % (txt[:80], dt))


def c9():
    shot_tool = [{"name": "screenshot", "description": "Take a screenshot",
                  "input_schema": {"type": "object", "properties": {}}}]
    msgs9 = [{"role": "user", "content": "Take a screenshot and tell me the colour of the circle in one word."},
             {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_shot1", "name": "screenshot",
                                                "input": {}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_shot1", "content": [
                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}]}]}]
    d, dt = post("/v1/messages", {"model": "m", "max_tokens": 200, "tools": shot_tool, "messages": msgs9})
    j = json.loads(d)
    txt = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
    check("red" in txt.lower(), "anthropic tool_result screenshot -> %r (%.1fs)" % (txt[:80], dt))


def c10():
    shot_tool = [{"name": "screenshot", "description": "Take a screenshot",
                  "input_schema": {"type": "object", "properties": {}}}]
    m10 = [{"role": "user", "content": "Take a screenshot and tell me the colour of the circle in one word."}]
    d, _ = post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": shot_tool,
                                 "tool_choice": {"type": "tool", "name": "screenshot"}, "messages": m10})
    j = json.loads(d)
    tu = [b for b in j["content"] if b["type"] == "tool_use"]
    if not tu:
        check(False, "live image continuation: no sampled screenshot call to continue")
        return
    m10 += [{"role": "assistant", "content": j["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"], "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}]}]}]
    seen = os.path.getsize(log_path)
    d, dt = post("/v1/messages", {"model": "m", "max_tokens": 1024, "tools": shot_tool, "messages": m10})
    j2 = json.loads(d)
    txt = "".join(b.get("text", "") for b in j2["content"] if b["type"] == "text")
    with open(log_path, "rb") as f:
        f.seek(seen)
        new = f.read().decode("utf-8", "replace")
    live = "anthropic live continuation" in new and "cannot place its images" not in new
    check(live and "red" in txt.lower(),
          "live tool_result image continuation (live=%s) -> %r (%.1fs)" % (live, txt[:80], dt))


def c11():
    # A persists a chain to the grid point below its prompt's end; B repeats A's message and asks on, so B's prefix
    # holds that whole chain.  A's long answer moves its bank's frontier past the protect floor: B routes to a fresh
    # bank, where only the disk chain can serve it.
    shared = ("Reference notes: " + " ".join("entry%d" % i for i in range(700)) +
              "\nDescribe the picture in about 150 words.")

    def ask(text):
        return {"model": "m", "max_tokens": 2048, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + IMG_B64}},
            {"type": "text", "text": text}]}]}

    post("/v1/chat/completions", ask(shared))
    seen = os.path.getsize(log_path)
    d, dt = post("/v1/chat/completions", ask(shared + " Then, on its own line, name the circle's colour in one word."))
    j = json.loads(d)
    txt = j["choices"][0]["message"].get("content") or ""
    cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    with open(log_path, "rb") as f:
        f.seek(seen)
        new = f.read().decode("utf-8", "replace")
    hit = "kv cache hit" in new
    check(hit and cached >= 1024 and "red" in txt.lower(),
          "image disk chain on a fresh bank (hit=%s, cached %d) -> %r (%.1fs)" % (hit, cached, txt[-60:], dt))


def c12():
    """L284 P2: /v1/completions continues the prompt raw on every family -- no template, no thinking block."""
    d, dt = post("/v1/completions", {"model": "m", "max_tokens": 16, "temperature": 0,
                                     "prompt": "1, 2, 3, 4, 5, 6,"})
    txt = json.loads(d)["choices"][0]["text"]
    refused = 0
    try:
        post("/v1/completions", {"model": "m", "max_tokens": 4, "prompt": "x", "think": True})
    except urllib.error.HTTPError as e:
        refused = e.code
    check("7" in txt and "</think>" not in txt and refused == 400,
          "completions raw continuation -> %r; think:true -> %d (%.1fs)" % (txt[:60], refused, dt))


def c13():
    """L284 P3: one effort-name table -- every name and an integer serve on every family; thinking on + none is 400."""
    codes = []
    for eff in ("minimal", "low", "medium", "high", "xhigh", "max", 60):
        d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 1, "reasoning_effort": eff,
                                              "messages": [{"role": "user", "content": "Hi"}]})
        codes.append(200)
    refused = 0
    try:
        post("/v1/chat/completions", {"model": "m", "max_tokens": 1, "think": True, "reasoning_effort": "none",
                                      "messages": [{"role": "user", "content": "Hi"}]})
    except urllib.error.HTTPError as e:
        refused = e.code
    check(refused == 400, "effort names %s -> 200 each; think:true + none -> %d" % (codes, refused))


def c14():
    """L284 P5: an OpenAI streamed call's arguments arrive as they are read and add up to the object."""
    d, dt = post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT, "stream": True,
                                          "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                          "messages": [{"role": "user", "content": "Weather in Paris?"}]})
    ev = [json.loads(l[6:]) for l in d.splitlines() if l.startswith("data: {")]
    frags = [tc["function"].get("arguments", "") for e in ev for ch in e.get("choices", [])
             for tc in (ch.get("delta") or {}).get("tool_calls") or [] if "function" in tc]
    args = "".join(frags)
    check(len([f for f in frags if f]) >= 2 and "city" in json.loads(args or "{}"),
          "openai stream: %d argument fragments -> %s" % (len([f for f in frags if f]), args))


for n, f in (("plain", c1), ("forced", c2), ("forced-stream", c3), ("continuation", c4), ("required", c5),
             ("undeclared", c6), ("required-two", c7), ("image", c8), ("image-tool-result", c9),
             ("image-live-continuation", c10), ("image-disk-chain", c11), ("completions-raw", c12),
             ("effort-names", c13), ("stream-args", c14)):
    guarded(n, f)
print("SERVER LIVE GATE: " + ("PASS" if ok else "FAIL"))
stop(0 if ok else 1)

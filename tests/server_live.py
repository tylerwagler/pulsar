#!/usr/bin/env python3
"""server_live.py -- the live server tier (L278 contract 6): a real pulsar-server on a model, driven over HTTP.

    python3 tests/server_live.py MODEL [--binary ./pulsar-server] [--port 8199] [--ctx 32768]

Every check is family-neutral -- the same requests for DeepSeek and Qwen; the family's own renderer and parser are
what is being graded.  Born as L272's ~/p3_live.py (the probe that found the undeclared-tool hole).

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

L317: a leg of tests/live_legs.py (the battery's shared server per model; warm phase only).  Alone (this command line)
it runs on an instance of its own.  `make server-live-gate` runs it alone on FRONTIER_MODEL and the hosted Qwen.
Exit 0 only when every check passed."""
import base64, json, os, struct, sys, time, urllib.error, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg  # noqa: E402


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
MSGS = [{"role": "user", "content": "I'm heading to Paris tomorrow."}]
SHOT_TOOL = [{"name": "screenshot", "description": "Take a screenshot",
              "input_schema": {"type": "object", "properties": {}}}]


class ServerLiveLeg(Leg):
    NAME = "server-live"

    def warm(self, srv):
        self.srv, self.state = srv, {}
        for n, f in (("plain", self.c1), ("forced", self.c2), ("forced-stream", self.c3), ("continuation", self.c4),
                     ("required", self.c5), ("undeclared", self.c6), ("required-two", self.c7), ("image", self.c8),
                     ("image-tool-result", self.c9), ("image-live-continuation", self.c10),
                     ("image-disk-chain", self.c11), ("completions-raw", self.c12), ("effort-names", self.c13),
                     ("stream-args", self.c14)):
            try:
                f()
            except Exception as e:   # a check that throws is a failed check, said by name
                self.g.check(False, "%s raised %s: %s" % (n, type(e).__name__, e))
        if not self.g.ok:
            print("--- server log tail ---\n" + srv.tail())

    def post(self, path, body):
        req = urllib.request.Request(self.srv.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        t = time.time()
        r = urllib.request.urlopen(req, timeout=600)
        return r.read().decode(), time.time() - t

    def c1(self):
        d, dt = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 64, "temperature": 0, "messages": [
            {"role": "user", "content": "What is 7+5? Answer with just the number."}]})
        m = json.loads(d)["choices"][0]["message"]
        self.g.check("12" in (m.get("content") or ""), "plain chat answer -> %r (%.1fs)" % ((m.get("content") or "")[:80], dt))

    def c2(self):
        d, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": TOOLS, "messages": MSGS,
                                           "tool_choice": {"type": "tool", "name": "get_weather"}})
        j = json.loads(d)
        tu = [b for b in j["content"] if b["type"] == "tool_use"]
        self.state["j"], self.state["tu"] = j, tu
        self.g.check(tu and tu[0]["name"] == "get_weather" and "city" in tu[0]["input"],
                     "anthropic forced tool_choice -> %s stop=%s (%.1fs)" % (tu[0]["input"] if tu else j["content"],
                                                                              j.get("stop_reason"), dt))

    def c3(self):
        d, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": TOOLS, "messages": MSGS,
                                           "stream": True, "tool_choice": {"type": "tool", "name": "get_weather"}})
        ev = [json.loads(l[6:]) for l in d.splitlines() if l.startswith("data: ")]
        starts = [e for e in ev if e.get("type") == "content_block_start" and e["content_block"]["type"] == "tool_use"]
        deltas = [e for e in ev if e.get("type") == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
        args = "".join(e["delta"]["partial_json"] for e in deltas)
        self.g.check(starts and deltas and "city" in json.loads(args or "{}"),
                     "anthropic stream: tool_use start + %d input_json_delta -> %s" % (len(deltas), args))

    def c4(self):
        if not self.state.get("tu"):
            self.g.check(False, "tool_result continuation: no call from check 2 to continue")
            return
        call = self.state["tu"][0]
        msgs2 = MSGS + [{"role": "assistant", "content": self.state["j"]["content"]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                                      "content": "18C, light rain"}]}]
        d, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 600, "tools": TOOLS, "messages": msgs2})
        j2 = json.loads(d)
        txt = "".join(b.get("text", "") for b in j2["content"] if b["type"] == "text")
        self.g.check("18" in txt or "rain" in txt.lower(),
                     "anthropic tool_result continuation -> %r (%.1fs, usage %s)" % (txt[:120], dt, j2.get("usage")))

    def c5(self):
        d, dt = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT,
                                                   "tool_choice": "required",
                                                   "messages": [{"role": "user", "content": "Tell me a joke."}]})
        ch = json.loads(d)["choices"][0]
        tc = ch["message"].get("tool_calls") or []
        self.g.check(tc and tc[0]["function"]["name"] == "get_weather",
                     "openai tool_choice=required -> %s finish=%s" % (tc[0]["function"] if tc else ch["message"],
                                                                      ch["finish_reason"]))

    def c6(self):
        try:
            self.post("/v1/chat/completions", {"model": "m", "max_tokens": 50, "tools": OT,
                                               "tool_choice": {"type": "function", "function": {"name": "reply"}},
                                               "messages": [{"role": "user", "content": "Hi"}]})
            self.g.check(False, "named undeclared tool_choice should refuse")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            self.g.check(e.code == 400 and "not a declared tool" in body,
                         "named undeclared tool_choice -> %d %s" % (e.code, body[:120]))

    def c7(self):
        names = []
        for _ in range(5):
            d, dt = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT2,
                                                       "tool_choice": "required", "temperature": 1.0,
                                                       "messages": [{"role": "user", "content": "Tell me a joke."}]})
            ch = json.loads(d)["choices"][0]
            tc = ch["message"].get("tool_calls") or []
            names.append(tc[0]["function"]["name"] if tc else "(none:" + ch["finish_reason"] + ")")
        self.g.check(all(n in ("get_weather", "search") for n in names), "required, two tools, 5 sampled turns -> %s" % names)

    def c8(self):
        d, dt = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 1024, "temperature": 0, "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + IMG_B64}},
                {"type": "text", "text": "What color is the circle in this image? Answer with one word."}]}]})
        txt = json.loads(d)["choices"][0]["message"].get("content") or ""
        self.g.check("red" in txt.lower(), "openai image_url -> %r (%.1fs)" % (txt[:80], dt))

    def c9(self):
        msgs9 = [{"role": "user", "content": "Take a screenshot and tell me the colour of the circle in one word."},
                 {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_shot1", "name": "screenshot",
                                                    "input": {}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_shot1", "content": [
                     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}]}]}]
        d, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 200, "tools": SHOT_TOOL, "messages": msgs9})
        j = json.loads(d)
        txt = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
        self.g.check("red" in txt.lower(), "anthropic tool_result screenshot -> %r (%.1fs)" % (txt[:80], dt))

    def c10(self):
        m10 = [{"role": "user", "content": "Take a screenshot and tell me the colour of the circle in one word."}]
        d, _ = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "tools": SHOT_TOOL,
                                          "tool_choice": {"type": "tool", "name": "screenshot"}, "messages": m10})
        j = json.loads(d)
        tu = [b for b in j["content"] if b["type"] == "tool_use"]
        if not tu:
            self.g.check(False, "live image continuation: no sampled screenshot call to continue")
            return
        m10 += [{"role": "assistant", "content": j["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"], "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}]}]}]
        seen = self.srv.mark()
        d, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 1024, "tools": SHOT_TOOL, "messages": m10})
        j2 = json.loads(d)
        txt = "".join(b.get("text", "") for b in j2["content"] if b["type"] == "text")
        new = self.srv.since(seen)
        live = "anthropic live continuation" in new and "cannot place its images" not in new
        self.g.check(live and "red" in txt.lower(),
                     "live tool_result image continuation (live=%s) -> %r (%.1fs)" % (live, txt[:80], dt))

    def c11(self):
        # A persists a chain to the grid point below its prompt's end; B repeats A's message and asks on, so B's prefix
        # holds that whole chain.  A's long answer moves its bank's frontier past the protect floor: B routes to a fresh
        # bank, where only the disk chain can serve it.
        shared = ("Reference notes: " + " ".join("entry%d" % i for i in range(700)) +
                  "\nDescribe the picture in about 150 words.")

        def ask(text):
            return {"model": "m", "max_tokens": 2048, "temperature": 0, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + IMG_B64}},
                {"type": "text", "text": text}]}]}

        self.post("/v1/chat/completions", ask(shared))
        seen = self.srv.mark()
        d, dt = self.post("/v1/chat/completions", ask(shared + " Then, on its own line, name the circle's colour in one word."))
        j = json.loads(d)
        txt = j["choices"][0]["message"].get("content") or ""
        cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        hit = "kv cache hit" in self.srv.since(seen)
        self.g.check(hit and cached >= 1024 and "red" in txt.lower(),
                     "image disk chain on a fresh bank (hit=%s, cached %d) -> %r (%.1fs)" % (hit, cached, txt[-60:], dt))

    def c12(self):
        """L284 P2: /v1/completions continues the prompt raw on every family -- no template, no thinking block."""
        d, dt = self.post("/v1/completions", {"model": "m", "max_tokens": 16, "temperature": 0,
                                              "prompt": "1, 2, 3, 4, 5, 6,"})
        txt = json.loads(d)["choices"][0]["text"]
        refused = 0
        try:
            self.post("/v1/completions", {"model": "m", "max_tokens": 4, "prompt": "x", "think": True})
        except urllib.error.HTTPError as e:
            refused = e.code
        self.g.check("7" in txt and "</think>" not in txt and refused == 400,
                     "completions raw continuation -> %r; think:true -> %d (%.1fs)" % (txt[:60], refused, dt))

    def c13(self):
        """L284 P3: one effort-name table -- every name and an integer serve on every family; thinking on + none is 400."""
        codes = []
        for eff in ("minimal", "low", "medium", "high", "xhigh", "max", 60):
            self.post("/v1/chat/completions", {"model": "m", "max_tokens": 1, "reasoning_effort": eff,
                                               "messages": [{"role": "user", "content": "Hi"}]})
            codes.append(200)
        refused = 0
        try:
            self.post("/v1/chat/completions", {"model": "m", "max_tokens": 1, "think": True, "reasoning_effort": "none",
                                               "messages": [{"role": "user", "content": "Hi"}]})
        except urllib.error.HTTPError as e:
            refused = e.code
        self.g.check(refused == 400, "effort names %s -> 200 each; think:true + none -> %d" % (codes, refused))

    def c14(self):
        """L284 P5: an OpenAI streamed call's arguments arrive as they are read and add up to the object."""
        d, dt = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 400, "tools": OT, "stream": True,
                                                   "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                                   "messages": [{"role": "user", "content": "Weather in Paris?"}]})
        ev = [json.loads(l[6:]) for l in d.splitlines() if l.startswith("data: {")]
        frags = [tc["function"].get("arguments", "") for e in ev for ch in e.get("choices", [])
                 for tc in (ch.get("delta") or {}).get("tool_calls") or [] if "function" in tc]
        args = "".join(frags)
        self.g.check(len([f for f in frags if f]) >= 2 and "city" in json.loads(args or "{}"),
                     "openai stream: %d argument fragments -> %s" % (len([f for f in frags if f]), args))


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(ServerLiveLeg, sys.argv[1:], ctx=32768, port=8199))

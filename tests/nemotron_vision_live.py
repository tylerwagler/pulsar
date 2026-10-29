#!/usr/bin/env python3
"""nemotron_vision_live.py -- L331: Nemotron 3.5 Super VL's images, live, on a real pulsar-server over HTTP.

Synthetic images (zlib-only PNGs: no PIL on the box) whose content a working vision path cannot miss, asked with
thinking off at temperature 0; each check passes when the answer NAMES the content:
  V1  a red disc                         "red"
  V2  a blue square beside a yellow triangle    "blue" for the square's colour; "triangle" for the right-hand shape
  V3  the digits 42 drawn as seven-segment bars  "42"
  V4  a green background                 "green"  (a tile the processor upscales: 100 x 80 px)
  V5  a wide banner, black on white "7"  "7"      (an extreme aspect: 1200 x 150 px)
  V6  two images in one message, numbered <image 1> / <image 2>: "which one is red" -> "first" / "1"
  V7  the prefix cache with an image (the core's image identity): the conversation extended after an image answer
      resumes PAST the image block (cached tokens >= the first prompt's, the image's block among them); the same
      text with a DIFFERENT image does not resume past the image's start (cached < the block's start + its length)
  V8  an Anthropic tool_result screenshot (the agent shape): the disc's colour, "red"
--refuse: the model is a container WITHOUT the tower -- every image request is refused 400, by name ("no vision tower").

    python3 tests/nemotron_vision_live.py MODEL [--binary ./pulsar-server] [--refuse]
Exit 0 only when every check passed."""
import base64, json, os, struct, sys, time, urllib.error, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, cached_tokens  # noqa: E402


def png(w, h, paint):
    """paint(x, y) -> (r, g, b); a PNG from zlib alone"""
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            row += bytes(paint(x, y))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def disc(w=512, h=512, col=(220, 20, 20)):
    return png(w, h, lambda x, y: col if (x - w / 2) ** 2 + (y - h / 2) ** 2 < (min(w, h) * 0.33) ** 2 else WHITE)


def square_triangle(w=640, h=360):
    def p(x, y):
        if w * 0.08 < x < w * 0.42 and h * 0.2 < y < h * 0.8:
            return (30, 60, 210)
        cx, top, bot, half = w * 0.75, h * 0.15, h * 0.85, w * 0.18
        if top < y < bot and abs(x - cx) < half * (y - top) / (bot - top):
            return (240, 210, 20)
        return WHITE
    return png(w, h, p)


SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg"}


def digits(text, w, h, ox=0.1, scale=0.8):
    """seven-segment digits, black bars on white"""
    n = len(text)
    cw, ch = w * scale / n, h * scale
    t = max(2, int(min(cw, ch) * 0.12))

    def p(x, y):
        for i, d in enumerate(text):
            x0, y0 = w * ox + i * cw + cw * 0.15, h * (1 - scale) / 2
            x1, ym, y1 = x0 + cw * 0.6, y0 + ch / 2, y0 + ch
            segs = {"a": (x0, y0, x1, y0 + t), "g": (x0, ym - t / 2, x1, ym + t / 2), "d": (x0, y1 - t, x1, y1),
                    "f": (x0, y0, x0 + t, ym), "b": (x1 - t, y0, x1, ym), "e": (x0, ym, x0 + t, y1),
                    "c": (x1 - t, ym, x1, y1)}
            for s in SEG[d]:
                a, b, c, e = segs[s]
                if a <= x < c and b <= y < e:
                    return BLACK
        return WHITE
    return png(w, h, p)


def b64(p):
    return base64.b64encode(p).decode()


def url_part(p):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(p)}}


class NemotronVisionLive(Leg):
    NAME = "nemotron-vision-live"

    def __init__(self, g, work, refuse=False):
        super().__init__(g, work)
        self.refuse = refuse

    def warm(self, srv):
        self.srv = srv
        checks = (("refuse", self.refusal),) if self.refuse else (
            ("V1", self.v1), ("V2", self.v2), ("V3", self.v3), ("V4", self.v4), ("V5", self.v5), ("V6", self.v6),
            ("V7", self.v7), ("V8", self.v8))
        for n, f in checks:
            try:
                f()
            except Exception as e:   # a check that throws is a failed check, said by name
                self.g.check(False, "%s raised %s: %s" % (n, type(e).__name__, e))
        if not self.g.ok:
            print("--- server log tail ---\n" + srv.tail())

    def post(self, path, body):
        req = urllib.request.Request(self.srv.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        t = time.time()
        r = urllib.request.urlopen(req, timeout=900)
        return json.loads(r.read().decode()), time.time() - t

    def ask(self, parts, max_tokens=64, history=()):
        body = {"model": "m", "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none",
                "messages": list(history) + [{"role": "user", "content": parts}]}
        j, dt = self.post("/v1/chat/completions", body)
        return (j["choices"][0]["message"].get("content") or ""), j, dt

    def expect(self, label, parts, words):
        txt, _j, dt = self.ask(parts)
        hit = any(w in txt.lower() for w in words)
        self.g.check(hit, "%s -> %r (%.1fs; wants one of %s)" % (label, txt[:100], dt, words))
        return txt

    def v1(self):
        self.expect("V1 red disc", [url_part(disc()), {"type": "text", "text":
                    "What colour is the circle? Answer with one word."}], ["red"])

    def v2(self):
        p = square_triangle()
        self.expect("V2 square colour", [url_part(p), {"type": "text", "text":
                    "What colour is the square on the left? Answer with one word."}], ["blue"])
        self.expect("V2 right-hand shape", [url_part(p), {"type": "text", "text":
                    "What shape is on the right? Answer with one word."}], ["triangle"])

    def v3(self):
        self.expect("V3 digits", [url_part(digits("42", 480, 240)), {"type": "text", "text":
                    "What number is written in this image? Answer with the number only."}], ["42"])

    def v4(self):
        self.expect("V4 small green tile (upscaled)", [url_part(png(100, 80, lambda x, y: (20, 170, 40))),
                    {"type": "text", "text": "What colour is this image? Answer with one word."}], ["green"])

    def v5(self):
        self.expect("V5 wide banner", [url_part(digits("7", 1200, 150, ox=0.45, scale=0.7)), {"type": "text", "text":
                    "Which digit is shown? Answer with the digit only."}], ["7"])

    def v6(self):
        self.expect("V6 two images (numbered)", [url_part(disc(col=(220, 20, 20))), url_part(disc(col=(30, 60, 210))),
                    {"type": "text", "text": "Which image shows a red circle, the first or the second? Answer with "
                                             "one word."}], ["first", "1"])

    def v7(self):
        q = {"type": "text", "text": "Describe the colour of the circle in one word."}
        a = disc(col=(220, 20, 20))
        t1, j1, _ = self.ask([url_part(a), q])
        p1 = j1["usage"]["prompt_tokens"]
        hist = [{"role": "user", "content": [url_part(a), q]}, {"role": "assistant", "content": t1}]
        t2, j2, _ = self.ask([{"type": "text", "text": "And what shape is it? One word."}], history=hist)
        c2 = cached_tokens(j2.get("usage")) or 0
        self.g.check(c2 >= p1 - 8 and "circle" in t2.lower(),
                     "V7 the same image resumes past its block: cached %d of the first prompt's %d -> %r" % (c2, p1,
                                                                                                         t2[:60]))
        b = disc(col=(30, 60, 210))
        t3, j3, _ = self.ask([url_part(b), q])
        c3 = cached_tokens(j3.get("usage")) or 0
        blk = p1 - 30   # the image block alone is >= 256 of the first prompt's tokens; the text around it < 30
        self.g.check(c3 < p1 - 256 and c3 <= p1 - blk + 30 and "blue" in t3.lower(),
                     "V7 a different image does not resume past its start: cached %d (prompt %d) -> %r" % (c3, p1,
                                                                                                        t3[:60]))

    def v8(self):
        shot = [{"name": "screenshot", "description": "Take a screenshot", "input_schema": {"type": "object",
                                                                                              "properties": {}}}]
        msgs = [{"role": "user", "content": "Take a screenshot and tell me the colour of the circle, in one word."},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "screenshot", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(disc())}}]}]}]
        j, dt = self.post("/v1/messages", {"model": "m", "max_tokens": 64, "tools": shot, "messages": msgs,
                                           "thinking": {"type": "disabled"}})
        txt = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
        self.g.check("red" in txt.lower(), "V8 anthropic tool_result screenshot -> %r (%.1fs)" % (txt[:80], dt))

    def refusal(self):
        code, body = 0, ""
        try:
            self.ask([url_part(disc()), {"type": "text", "text": "What colour?"}])
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read().decode()
        self.g.check(code == 400 and "no vision tower" in body,
                     "a tower-less container refuses an image by name -> %d %r" % (code, body[:120]))


if __name__ == "__main__":
    from live_legs import standalone
    sys.exit(standalone(NemotronVisionLive, sys.argv[1:], ctx=32768, port=8331,
                        add_args=lambda ap: ap.add_argument("--refuse", action="store_true")))

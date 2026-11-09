#!/usr/bin/env python3
"""model_smoke_live.py -- L357: every published model starts and answers one request on a real pulsar-server.

    python3 tests/model_smoke_live.py MODEL_DIR... --binary ./pulsar-server --work DIR [--port 8357] [--ctx 8192]

One model at a time: a pulsar-server on it alone (tests/live_server.py's Server: drop_caches, /health, TERM), then the
one request its kind serves -- the kind read from the server's own /health (an encoder / parser server lists its
models with "encoder" / "parser"; anything else is a chat model):
  chat     "What is 7+5? Answer with the number only." (temperature 0, thinking off through chat_template_kwargs, the
           control every family's template takes) -- the reply's content contains the number 12;
  embed    /v1/embeddings, one query -- a vector of the dimension /health advertises, finite, not all zero;
  rerank   /v1/rerank, "capital of France" over ["Bananas are yellow.", "Paris is the capital of France."] -- Paris
           ranks first;
  parse    /v1/parse, a small generated PNG with no text (a grey square on white) -- 200 and a non-empty JSON body.
A model whose server does not start is a FAIL naming it; there is no SKIP.  One PASS / FAIL line per model; exit 0
only when every model passed.  DeepSeek V4 is smoke-only since 2026-11-07: this is its release check.

L357: the models live-legs already runs (Qwen, Nemotron) take the same chat check as a leg on its warm instance
(SmokeLeg), so the release tier starts no server of theirs for it; make model-smoke's SMOKE_MODELS leaves them out."""
import argparse, base64, json, math, os, re, struct, sys, time, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import Leg, Server, StartError  # noqa: E402

CHAT_Q = "What is 7+5? Answer with the number only."
RERANK_Q = "capital of France"
RERANK_DOCS = ["Bananas are yellow.", "Paris is the capital of France."]   # the right answer second: no order bias


def png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def blank_page_png(w=128, h=128):
    """An RGB PNG, white with a grey square in the middle: an image with no text in it."""
    rows = []
    for y in range(h):
        row = bytearray(b"\0")
        for x in range(w):
            inside = w // 4 <= x < 3 * w // 4 and h // 4 <= y < 3 * h // 4
            row += b"\x80\x80\x80" if inside else b"\xff\xff\xff"
        rows.append(bytes(row))
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + png_chunk(b"IEND", b""))


def health(srv):
    return json.loads(urllib.request.urlopen(srv.url + "/health", timeout=10).read().decode("utf-8", "replace"))


def kind_of(h):
    """chat, or the served encoder / parser entry's kind and dimension"""
    for m in h.get("models") or []:
        if isinstance(m, dict) and m.get("parser"):
            return "parse", None
        if isinstance(m, dict) and m.get("encoder"):
            return m["encoder"], m.get("dim")
    return "chat", None


def smoke_chat(srv):
    st, j = srv.post_json("/v1/chat/completions", {
        "model": "m", "messages": [{"role": "user", "content": CHAT_Q}], "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}})
    if st != 200:
        return False, "chat: HTTP %d %s" % (st, json.dumps(j)[:300])
    content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return re.search(r"\b12\b", content) is not None, "chat answered %r" % content[:120]


class SmokeLeg(Leg):
    """The chat smoke on live-legs' warm instance (production's flags): the model started, and it answers"""
    NAME = "model-smoke"

    def warm(self, srv):
        kind, _ = kind_of(health(srv))
        if not self.g.check(kind == "chat", "the warm instance serves a chat model: %s" % kind):
            return
        ok, what = smoke_chat(srv)
        self.g.check(ok, "%s: %s" % (srv.model, what))


def smoke_embed(srv, dim):
    st, j = srv.post_json("/v1/embeddings", {"input": [RERANK_Q], "input_type": "query"})
    if st != 200:
        return False, "embeddings: HTTP %d %s" % (st, json.dumps(j)[:300])
    v = ((j.get("data") or [{}])[0]).get("embedding") or []
    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    nonzero = any(x != 0 for x in v) if finite else False
    ok = bool(dim) and len(v) == dim and finite and nonzero
    return ok, "embedding of %d floats (advertised %s), finite %s, non-zero %s" % (len(v), dim, finite, nonzero)


def smoke_rerank(srv):
    st, j = srv.post_json("/v1/rerank", {"query": RERANK_Q, "documents": RERANK_DOCS})
    if st != 200:
        return False, "rerank: HTTP %d %s" % (st, json.dumps(j)[:300])
    res = j.get("results") or []
    scores = {r.get("index"): r.get("relevance_score") for r in res}
    ok = len(res) == len(RERANK_DOCS) and res[0].get("index") == 1
    return ok, "rerank scores Paris %s, Bananas %s (best first: index %s)" % (
        scores.get(1), scores.get(0), res[0].get("index") if res else None)


def smoke_parse(srv):
    url = "data:image/png;base64," + base64.b64encode(blank_page_png()).decode()
    st, j = srv.post_json("/v1/parse", {"image": url, "max_tokens": 64})
    ok = st == 200 and isinstance(j, dict) and bool(j)
    return ok, "parse: HTTP %d, %d JSON members, text %r" % (st, len(j) if isinstance(j, dict) else 0,
                                                            (j.get("text") or "")[:80] if isinstance(j, dict) else "")


def smoke(model, a, n):
    """(ok, kind, what) for one model on its own server"""
    try:
        srv = Server(a.binary, model, a.work, "smoke-%d" % n, a.port, a.ctx, kv_disk=False,
                     start_timeout=a.start_timeout)
    except StartError as e:
        return False, "?", "did not start: %s" % str(e)[-600:]
    try:
        kind, dim = kind_of(health(srv))
        t0 = time.time()
        if kind == "chat":
            ok, what = smoke_chat(srv)
        elif kind == "embed":
            ok, what = smoke_embed(srv, dim)
        elif kind == "rerank":
            ok, what = smoke_rerank(srv)
        elif kind == "parse":
            ok, what = smoke_parse(srv)
        else:
            ok, what = False, "an encoder kind this smoke does not know: %r" % kind
        what += " (%.0f s load, %.1f s request)" % (srv.load_s, time.time() - t0)
        if not srv.alive():
            ok, what = False, what + "; the server died: " + srv.tail(600)
        return ok, kind, what
    except Exception as e:   # a dropped connection or a bad body is this model's FAIL, not the run's end
        return False, "?", "%s: %s" % (type(e).__name__, e)
    finally:
        srv.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--binary", default="./pulsar-server")
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=8357)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--start-timeout", type=float, default=900.0)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    lines, fails = [], 0
    for n, model in enumerate(a.models):
        ok, kind, what = smoke(model, a, n)
        fails += not ok
        line = "%s model-smoke %s [%s]: %s" % ("PASS" if ok else "FAIL", model, kind, what)
        lines.append(line)
        print(line, flush=True)
    print("\n===== model-smoke: %d model(s) =====" % len(a.models))
    for line in lines:
        print("  " + line)
    print("MODEL SMOKE: " + ("PASS" if not fails else "FAIL (%d of %d)" % (fails, len(a.models))))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

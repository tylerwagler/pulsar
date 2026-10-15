#!/usr/bin/env python3
"""A minimal OpenAI-compatible /v1/chat/completions server over exllamav3, for the EXL3 arms' agentic eval (L296).

pulsar does not serve Nemotron yet, so tool-eval-bench reaches the EXL3 packs through this shim.  It does what the
card's vLLM server does for a request, and nothing more:
  - render: the checkpoint's own chat template through transformers' apply_chat_template (tools, reasoning
    history, `chat_template_kwargs` such as enable_thinking); tool-call arguments decoded once from their JSON
    string, as vLLM does (build_suite.normalize_messages, the one implementation of that rule here);
  - generate: exllamav3's Generator, the request's temperature / top_p / top_k / min_p / seed / max_tokens, stop at
    the generation EOS ids (generation_config.json), one request at a time (the card serves --max-num-seqs 1);
  - parse: reasoning = the text before </think> (the template opens <think>), content after it; tool calls in the
    qwen3_coder XML (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`), parameter
    values typed by the request's tool schema as vLLM's qwen3_coder parser types them.
  Not done: tool_choice="required" enforcement (tool-eval-bench then excludes TC-45, as it does for any endpoint
  that does not enforce it), images, logprobs.  stream=true is answered as one delta chunk + [DONE].

  python exl3_openai_shim.py --model /srv/models/nemotron35-exl3/k4 --port 8301 [--max-ctx 65536]
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_suite import normalize_messages  # noqa: E402

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def typed(value, schema):
    """vLLM qwen3_coder: strings stay strings; numbers, booleans, objects and arrays are parsed."""
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if value.strip().lower() == "null":
        return None
    try:
        if t in ("integer", "int"):
            return int(value.strip())
        if t in ("number", "float"):
            v = float(value.strip())
            return int(v) if v.is_integer() and "." not in value else v
        if t in ("boolean", "bool"):
            return value.strip().lower() == "true"
        if t in ("object", "array") or t is None and value.strip()[:1] in "[{":
            return json.loads(value)
    except ValueError:
        pass
    return value


def parse_output(text, tools):
    reasoning, content = None, text
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip("\n")
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties", {})
    calls = []
    for m in CALL_RE.finditer(content):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {k.strip(): typed(v, props.get(k.strip())) for k, v in PARAM_RE.findall(m.group(2))}
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if calls:
        content = content[: content.index("<tool_call>")]
    content = content.strip() or None
    return reasoning, content, calls


class Engine:
    def __init__(self, path, max_ctx):
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
        from transformers import AutoTokenizer
        self.config = Config.from_directory(path)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=max_ctx, max_batch_size=1)
        self.model.load(progressbar=True)
        self.tok = Tokenizer.from_config(self.config)
        self.hf = AutoTokenizer.from_pretrained(path)
        self.gen = Generator(model=self.model, cache=self.cache, tokenizer=self.tok)
        gc = json.load(open(os.path.join(path, "generation_config.json")))
        eos = gc.get("eos_token_id")
        self.stop = eos if isinstance(eos, list) else [eos]
        self.max_ctx = max_ctx
        self.name = os.path.basename(path.rstrip("/"))
        self.lock = threading.Lock()

    def chat(self, req):
        import torch
        from exllamav3 import ComboSampler, Job
        tools = req.get("tools") or None
        kw = dict(req.get("chat_template_kwargs") or {})
        ids = self.hf.apply_chat_template(normalize_messages(req["messages"]), tools=tools, tokenize=True,
                                          add_generation_prompt=True, return_dict=False, **kw)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        n_in = len(ids)
        max_new = req.get("max_completion_tokens") or req.get("max_tokens") or 4096
        max_new = min(max_new, self.max_ctx - n_in - 1)
        if max_new <= 0:
            return None, f"prompt of {n_in} tokens exceeds the context ({self.max_ctx})"
        sampler = ComboSampler(temperature=float(req.get("temperature", 1.0)), top_p=float(req.get("top_p", 1.0)),
                               top_k=int(req.get("top_k", 0) or 0), min_p=float(req.get("min_p", 0.0) or 0.0))
        with self.lock:
            job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=max_new, sampler=sampler,
                      seed=req.get("seed"), stop_conditions=self.stop, decode_special_tokens=True)
            self.gen.enqueue(job)
            text, n_out, reason = "", 0, "stop"
            while self.gen.num_remaining_jobs():
                for r in self.gen.iterate():
                    if r.get("stage") != "streaming":
                        continue
                    text += r.get("text", "")
                    if r.get("eos"):
                        n_out = r.get("new_tokens", n_out)
                        reason = "length" if r.get("eos_reason") == "max_new_tokens" else "stop"
        # the template's generation prompt ends in "<think>\n" (or "<think></think>" with thinking off)
        prompt_tail = self.hf.decode(ids[-4:])
        if "<think>" in prompt_tail and "</think>" not in prompt_tail.split("<think>")[-1]:
            text = "<think>" + text
        text = text.replace("<|im_end|>", "")
        reasoning, content, calls = parse_output(text, tools)
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
            msg["reasoning"] = reasoning
        if calls:
            msg["tool_calls"] = calls
            reason = "tool_calls"
        return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": self.name, "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}, None


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj, sse=False):
            if sse:
                self.send_response(code)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                c = obj["choices"][0]
                delta = dict(c["message"])
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(delta["tool_calls"])]
                chunk = {"id": obj["id"], "object": "chat.completion.chunk", "created": obj["created"],
                         "model": obj["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                end = dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": c["finish_reason"]}],
                           usage=obj["usage"])
                for part in (chunk, end):
                    self.wfile.write(b"data: " + json.dumps(part).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": engine.name, "object": "model",
                                                             "owned_by": "exllamav3", "max_model_len": engine.max_ctx}]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            try:
                out, err = engine.chat(req)
            except Exception as e:  # a bad request must not take the server down
                return self._send(400, {"error": {"message": f"{type(e).__name__}: {e}", "type": "invalid_request"}})
            if err:
                return self._send(400, {"error": {"message": err, "type": "invalid_request"}})
            self._send(200, out, sse=bool(req.get("stream")))

        def log_message(self, fmt, *args):
            sys.stderr.write("[shim] " + fmt % args + "\n")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8301)
    ap.add_argument("--max-ctx", type=int, default=65536)
    a = ap.parse_args()
    if a.port == 8000:
        sys.exit("refusing port 8000 (production)")
    engine = Engine(a.model, a.max_ctx)
    print(f"[shim] {engine.name} on http://{a.host}:{a.port}", file=sys.stderr, flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()

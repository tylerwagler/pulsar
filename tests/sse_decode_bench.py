#!/usr/bin/env python3
"""Client-side decode metric measured off the SSE stream.

Wall-clock t/s (total tokens / total request time) folds prefill, scheduling
delay and the network into the decode number.  A user watching tokens appear
experiences something different: how fast deltas arrive *once they start*.
This measures that, and reports it ALONGSIDE the wall-clock figure rather than
replacing it -- when the two disagree the gap is itself the finding (a slot
starved behind another job's prefill shows a healthy inter-delta rate and a
poor wall-clock one).

Stdlib only, matching speed-bench/plot_speed.py.

    ./tests/sse_decode_bench.py --port 8080 --prompt "Write a haiku" --max-tokens 128

Emits a JSON record on stdout (--json-out to also write a file).

Two measurement hazards this handles explicitly, both live in our own server:

  * SSE comment lines.  The decode heartbeat (43ddb54) writes ": ping" on the
    OpenAI and Responses surfaces.  A comment is not an event; counting one as
    a delta would inflate the rate and, worse, a heartbeat is emitted exactly
    when the slot is NOT producing -- the moment the metric most needs to stay
    honest.
  * reasoning_content deltas are decode steps too.  Counting only `content`
    makes a thinking model look stalled for the whole reasoning phase.

Disk KV state is recorded in the output because it changes what is being
measured: a run that hits a warm checkpoint skips prefill entirely, so TTFT
and wall-clock t/s are not comparable against a cold run.  Pass
--kv-disk-state to label the run; prefer running the server with --no-kv-disk
for anything you intend to compare across builds.
"""

import argparse
import http.client
import json
import sys
import time


def stream_once(host, port, payload, timeout):
    """POST a streaming chat completion; return timing + token facts."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    t_request = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, headers)
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read(2048).decode("utf-8", "replace")
        conn.close()
        raise RuntimeError("HTTP %d from /v1/chat/completions: %s" % (resp.status, detail))

    # Arrival time of every delta that carries generated text.  index 0 is the
    # first token, so ttft = arrivals[0] - t_request.
    arrivals = []
    n_content = 0
    n_reasoning = 0
    n_comments = 0
    usage = None
    finish_reason = None

    while True:
        raw = resp.readline()
        if not raw:
            break
        now = time.perf_counter()
        line = raw.strip()
        if not line:
            continue
        # SSE comment (our ": ping" heartbeat). Not an event -- count and skip.
        if line.startswith(b":"):
            n_comments += 1
            continue
        if not line.startswith(b"data:"):
            continue  # event: lines and any field we don't consume
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue

        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]

        for choice in obj.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            produced = False
            if isinstance(delta.get("content"), str) and delta["content"]:
                n_content += 1
                produced = True
            if isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
                n_reasoning += 1
                produced = True
            if produced:
                arrivals.append(now)

    t_end = time.perf_counter()
    conn.close()

    return {
        "t_request": t_request,
        "t_end": t_end,
        "arrivals": arrivals,
        "n_content_deltas": n_content,
        "n_reasoning_deltas": n_reasoning,
        "n_sse_comments": n_comments,
        "usage": usage,
        "finish_reason": finish_reason,
    }


def summarize(raw):
    """Derive the two rates. Returns None-valued fields when undeterminable."""
    arrivals = raw["arrivals"]
    n_deltas = raw["n_content_deltas"] + raw["n_reasoning_deltas"]
    wall_s = raw["t_end"] - raw["t_request"]

    out = {
        "ttft_ms": None,
        "sse_decode_tps": None,
        "wall_tps": None,
        "wall_s": round(wall_s, 6),
        "decode_span_s": None,
        "n_deltas": n_deltas,
        "n_content_deltas": raw["n_content_deltas"],
        "n_reasoning_deltas": raw["n_reasoning_deltas"],
        "n_sse_comments": raw["n_sse_comments"],
        "finish_reason": raw["finish_reason"],
        "token_source": None,
        "completion_tokens": None,
    }
    if not arrivals:
        return out

    out["ttft_ms"] = round((arrivals[0] - raw["t_request"]) * 1000.0, 3)

    # Prefer the server's own completion_tokens: one delta is not guaranteed to
    # be one token, so counting deltas would quietly mis-scale the rate.
    usage = raw["usage"] or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        out["completion_tokens"] = completion_tokens
        out["token_source"] = "usage"
    else:
        out["completion_tokens"] = n_deltas
        out["token_source"] = "delta-count"

    # Decode span excludes prefill: first delta -> last delta.  With n tokens
    # there are n-1 inter-token gaps, so divide by (n-1) to avoid crediting
    # decode with the token that arrived before the span opened.
    span = arrivals[-1] - arrivals[0]
    out["decode_span_s"] = round(span, 6)
    if span > 0 and out["completion_tokens"] > 1:
        out["sse_decode_tps"] = round((out["completion_tokens"] - 1) / span, 3)
    if wall_s > 0:
        out["wall_tps"] = round(out["completion_tokens"] / wall_s, 3)
    return out


def run(args):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    runs = []
    for i in range(args.repeat):
        raw = stream_once(args.host, args.port, payload, args.timeout)
        summary = summarize(raw)
        summary["run"] = i
        runs.append(summary)
        if args.repeat > 1 and not args.quiet:
            print(
                "run %d: ttft=%s ms  sse=%s t/s  wall=%s t/s  tokens=%s (%s)"
                % (
                    i,
                    summary["ttft_ms"],
                    summary["sse_decode_tps"],
                    summary["wall_tps"],
                    summary["completion_tokens"],
                    summary["token_source"],
                ),
                file=sys.stderr,
            )

    record = {
        "schema": "pulsar.sse_decode_bench.v1",
        "host": "%s:%d" % (args.host, args.port),
        "model": args.model,
        "label": args.label,
        "max_tokens": args.max_tokens,
        "kv_disk_state": args.kv_disk_state,
        "prompt_chars": len(args.prompt),
        "runs": runs,
    }

    ok = [r for r in runs if r["sse_decode_tps"] is not None]
    if ok:
        # min, not mean: a floor is what a user is guaranteed, and one stalled
        # run is the signal, not an outlier to average away.
        record["min_sse_decode_tps"] = min(r["sse_decode_tps"] for r in ok)
        record["min_wall_tps"] = min(r["wall_tps"] for r in ok if r["wall_tps"])
        record["max_ttft_ms"] = max(r["ttft_ms"] for r in ok if r["ttft_ms"])
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="ds4flash")
    p.add_argument("--prompt", default="Write a short technical paragraph about cache coherence.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--label", default="", help="free-form tag recorded in the output")
    p.add_argument("--kv-disk-state", default="unknown",
                   choices=["unknown", "disabled", "enabled"],
                   help="what the server was started with; recorded, not enforced")
    p.add_argument("--json-out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    record = run(args)
    text = json.dumps(record, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

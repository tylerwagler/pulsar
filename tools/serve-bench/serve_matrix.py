#!/usr/bin/env python3
"""Serving speed matrix for any OpenAI-compatible server (pulsar-server, TensorFold, vLLM), stdlib only.

One client, one set of prompt bytes, so two engines on the same box are measured under the same conditions.
Three modes, one JSON line per cell on stdout (and --out FILE appends them):

  single   one stream at a time: every --prompts x --temps cell, a warm-up then --reps runs (seeds 1234..).
           decode_tps = (completion_tokens - 1) / (last chunk - first chunk), the span TensorFold's
           tools/bench_openai.py times, so its published numbers compare directly.  tok_per_burst: chunks of one
           stream arriving within 2 ms of each other are one engine step (a speculative round streams several
           tokens), so this is tokens per round as the client sees it (1.0 without drafts).
  conc     --streams N concurrent streams, each its own prompt (no shared prefix).  decode_tps is the aggregate
           over the window where ALL streams decode (last first-chunk .. first stream end), per_stream_tps the
           median of each stream's own (tokens - 1) / span -- tools/tp2-rowlane-perf/decode_load.py's math.
  prefill  TTFT of a cold prompt of exactly --lengths tokens (counted with --tokenizer, a tokenizer.json): the
           story text below, cut by token count, behind a per-run nonce so no prefix cache can serve it.
           prefill_tps = prompt tokens / TTFT (one generated token, so TTFT is the prefill plus one step).

Neither pulsar nor TensorFold's Flash Next CUDA engine honours ignore_eos, so a reply can end early: every cell
reports the tokens it actually ran.

Against a server with pulsar's /metrics, every single / conc cell also reports the server's own speculation counters
over its timed runs: srv_tok_per_round (accepted drafts + the base token, per round that carried drafts) and
srv_drafts_per_round.  TensorFold's /metrics has no round counter; with --server-log (its server's stdout) the
cell sums the per-request "done ... rounds=R accepted=A/D" lines into the same two fields.  pulsar also publishes a
decode-only round's host wall by phase (pulsar:spec_round_phase_ms_total), reported as srv_phase_ms (ms a round).  With either,
srv_ms_per_round = 1000 x srv_tok_per_round / decode_tps: a round's wall price as the client sees it (conc: per
stream, from the median stream's rate).

  scoreboard  the fixed L288 cell set against one engine: single code T0 / prose T0 / code T0.7 (median of --reps x
           --tokens), then --streams-set (default 4,8) concurrent streams at T0 (--conc-tokens each).
           tools/serve-bench/scoreboard.sh starts either engine (pulsar-server or TensorFold) and runs it plus
           TensorFold's own public client (tools/bench_openai.py); `serve_matrix.py table A.jsonl B.jsonl` compares.

  serve_matrix.py single  --port 8000 --prompts code,codex,prose,story --temps 0,0.7,1.0 --tokens 512 --reps 3
  serve_matrix.py conc    --port 8000 --streams 4 --temps 0 --tokens 256
  serve_matrix.py prefill --port 8000 --lengths 1024,8192,32768 --tokenizer /srv/models/.../tokenizer.json
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STORY_FILE = os.path.join(HERE, "..", "..", "tests", "long_context_story_prompt.txt")

# kind chat: /v1/chat/completions with thinking off; kind raw: /v1/completions (a raw continuation).
PROMPTS = {
    # TensorFold's two public fixtures (tools/bench_openai.py), byte for byte, so its published table compares.
    "codex": ("raw", "Write a short Python function that computes the Fibonacci sequence and explain it."),
    "prose": ("chat", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical "
                      "example."),
    # A longer code task (L272's LRU-cache flavour): a chat reply that stays code for 512 tokens.
    "code": ("chat", "Write a Python LRU cache class with O(1) get and put using a dict and a doubly linked list. "
                     "Include type hints, docstrings, and a set of unittest test cases."),
    # A raw mid-document continuation (L271's losing workload: low MTP acceptance).
    "story": ("raw", None),
}

CONC_TASKS = [
    "Write a Python LRU cache class with O(1) get and put, with type hints and unittest cases.",
    "Write a C function that parses an IPv4 dotted-quad string into a uint32_t, rejecting malformed input.",
    "Explain how a B-tree keeps itself balanced on insert, step by step, with a small example.",
    "Write a Rust function that merges k sorted vectors into one sorted vector using a binary heap.",
    "Describe the life cycle of a star like the Sun, from nebula to white dwarf, in plain English.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes.",
    "Explain the difference between TCP and UDP and when you would choose each, with examples.",
    "Write a JavaScript debounce function with a leading/trailing option and explain how it works.",
]


def story_text():
    with open(STORY_FILE, encoding="utf-8") as f:
        t = f.read()
    # The fixture is a DeepSeek render; keep the user's story text only.
    for marker in ("<｜User｜>",):
        if marker in t:
            t = t.split(marker, 1)[1]
    return t


def story_prompt():
    t = story_text()
    return t[:2400].rsplit(" ", 1)[0]       # ~550 tokens, ends mid-story: a raw continuation


def request(port, kind, text, tokens, temp, seed, model, host="127.0.0.1"):
    body = {"model": model, "max_tokens": tokens, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    if temp > 0:
        body.update(top_k=20, top_p=0.95)
    if kind == "chat":
        url = f"http://{host}:{port}/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": text}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = f"http://{host}:{port}/v1/completions"
        body["prompt"] = text
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage, out, first_evt = [], None, [], None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and first_evt is None:
                first_evt = time.perf_counter()   # the first choice event, text or not: prefill's TTFT
            for ch in d.get("choices") or []:
                piece = ch.get("text") or (ch.get("delta") or {}).get("content") \
                    or (ch.get("delta") or {}).get("reasoning_content") or ""
                if piece:
                    stamps.append(time.perf_counter())
                    out.append(piece)
    n = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") else len(stamps)
    pt = int(usage["prompt_tokens"]) if usage and usage.get("prompt_tokens") else None
    return {"t0": t0, "first_evt": first_evt, "stamps": stamps, "tokens": n, "prompt_tokens": pt, "text": "".join(out)}


def spec_counters(port, host="127.0.0.1"):
    """The server's speculation counters (pulsar's /metrics, vLLM names): drafted / accepted tokens and draft rounds.
    None when the server has no such endpoint (TensorFold), so a cell then reports the client's view only."""
    want = {"vllm:spec_decode_num_draft_tokens_total": "drafted", "vllm:spec_decode_num_accepted_tokens_total":
            "accepted", "vllm:spec_decode_num_drafts_total": "rounds"}
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=10) as resp:
            text = resp.read().decode(errors="replace")
    except Exception:
        return None
    out, phases = {}, {}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in want:
            out[want[name]] = int(float(line.rsplit(" ", 1)[1]))
        elif name == "pulsar:spec_round_phase_ms_total":       # L288: a round's wall by phase
            phases[line.split('phase="', 1)[1].split('"', 1)[0]] = float(line.rsplit(" ", 1)[1])
        elif name == "pulsar:spec_round_phase_rounds_total":
            out["phase_rounds"] = int(line.rsplit(" ", 1)[1])
    out["phases"] = phases
    return out if all(k in out for k in want.values()) else None


def spec_delta(m0, m1):
    """Server-side tokens per round over a cell: (drafts accepted + one base token) per draft round, and drafts per
    round -- the rounds that carried drafts (a plain step has none)."""
    if not m0 or not m1:
        return {}
    r = m1["rounds"] - m0["rounds"]
    if r <= 0:
        return {"srv_rounds": 0}
    rec = {"srv_rounds": r, "srv_tok_per_round": round(1 + (m1["accepted"] - m0["accepted"]) / r, 3),
           "srv_drafts_per_round": round((m1["drafted"] - m0["drafted"]) / r, 3)}
    pr = m1.get("phase_rounds", 0) - m0.get("phase_rounds", 0)
    if pr > 0:   # pulsar: ms per decode-only round by phase (host wall, untraced)
        rec["srv_phase_ms"] = {k: round((v - m0["phases"].get(k, 0.0)) / pr, 3) for k, v in m1["phases"].items()}
    return rec


def log_mark(a):
    """Where --server-log ends now (None without one): log_spec reads the requests finished after it."""
    return os.path.getsize(a.server_log) if a.server_log else None


def log_spec(a, mark):
    """The same two fields from TensorFold's per-request log line ("[tensorfold] done ... rounds=R accepted=A/D"),
    summed over the requests that finished since `mark`; {} without --server-log or such lines."""
    if mark is None:
        return {}
    time.sleep(0.3)   # the line is printed as the reply closes
    with open(a.server_log, errors="replace") as f:
        f.seek(mark)
        lines = [l for l in f.read().splitlines() if "] done " in l and " rounds=" in l]
    r = acc = dr = 0
    for l in lines:
        kv = dict(t.split("=", 1) for t in l.split() if "=" in t)
        r += int(kv["rounds"])
        a_, d_ = kv["accepted"].split("/")
        acc, dr = acc + int(a_), dr + int(d_)
    if r <= 0:
        return {}
    return {"srv_rounds": r, "srv_tok_per_round": round(1 + acc / r, 3), "srv_drafts_per_round": round(dr / r, 3)}


def round_price(rec, tps):
    """srv_ms_per_round from a stream's decode rate and its tokens per round."""
    tpr = rec.get("srv_tok_per_round")
    if tps and tpr:
        rec["srv_ms_per_round"] = round(1000.0 * tpr / tps, 2)
    return rec


def bursts(stamps):
    return [stamps[0]] + [b for a, b in zip(stamps, stamps[1:]) if b - a > 0.002] if stamps else []


def one_stats(r):
    st = r["stamps"]
    span = st[-1] - st[0] if len(st) > 1 else 0.0
    b = bursts(st)
    return {"decode_tps": (r["tokens"] - 1) / span if span > 0 else None,
            "ttft_s": st[0] - r["t0"] if st else None,
            "tok_per_burst": r["tokens"] / max(1, len(b))}


def emit(rec, out):
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def run_single(a):
    for name in a.prompts.split(","):
        kind, text = PROMPTS[name]
        text = text if text is not None else story_prompt()
        for temp in [float(t) for t in a.temps.split(",")]:
            request(a.port, kind, text, min(a.tokens, 64), temp, 1233, a.model)          # warm-up
            m0, lm = spec_counters(a.port), log_mark(a)
            runs = [request(a.port, kind, text, a.tokens, temp, 1234 + i, a.model) for i in range(a.reps)]
            m1 = spec_counters(a.port)
            stats = [one_stats(r) for r in runs]
            tps = [s["decode_tps"] for s in stats if s["decode_tps"]]
            med = statistics.median(tps) if tps else None
            emit(round_price({"mode": "single", "label": a.label, "prompt": name, "temp": temp,
                              "max_tokens": a.tokens, "tokens": [r["tokens"] for r in runs],
                              "decode_tps_median": round(med, 2) if med else None,
                              "decode_tps_all": [round(x, 2) for x in tps],
                              "tok_per_burst": round(statistics.median(s["tok_per_burst"] for s in stats), 2),
                              "ttft_s_median": round(statistics.median(s["ttft_s"] for s in stats), 3),
                              **(spec_delta(m0, m1) or log_spec(a, lm)),
                              "sample": runs[0]["text"][:100]}, med), a.out)


def run_conc(a):
    for temp in [float(t) for t in a.temps.split(",")]:
        conc_cell(a, a.streams, temp, a.tokens)


def conc_cell(a, n, temp, tokens):
        res = [None] * n

        def go(i):
            res[i] = request(a.port, "chat", CONC_TASKS[i % len(CONC_TASKS)] + f" (variant {i})", tokens,
                             temp, 1234 + i, a.model)

        th = [threading.Thread(target=go, args=(i,)) for i in range(n)]
        m0, lm = spec_counters(a.port), log_mark(a)
        start = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        m1 = spec_counters(a.port)
        bad = [i for i, r in enumerate(res) if r is None or len(r["stamps"]) < 2]
        if bad:
            sys.exit(f"serve_matrix: stream(s) {bad} failed or produced < 2 chunks -- no measurement")
        firsts = [r["stamps"][0] for r in res]
        ends = [r["stamps"][-1] for r in res]
        w0, w1 = max(firsts), min(ends)
        chunks_per_tok = sum(len(r["stamps"]) for r in res) / max(1, sum(r["tokens"] for r in res))
        in_win = sum(sum(1 for s in r["stamps"] if w0 < s <= w1) for r in res)
        agg = in_win / chunks_per_tok / (w1 - w0) if w1 > w0 else 0.0
        per = [one_stats(r)["decode_tps"] for r in res]
        per_med = statistics.median(p for p in per if p)
        emit(round_price({"mode": "conc", "label": a.label, "streams": n, "temp": temp, "max_tokens": tokens,
                          "tokens": [r["tokens"] for r in res],
                          "decode_tps": round(agg, 2),
                          "per_stream_tps": round(per_med, 2),
                          "per_stream_min": round(min(p for p in per if p), 2),
                          "end_to_end_tps": round(sum(r["tokens"] for r in res) / (max(ends) - start), 2),
                          "tok_per_burst": round(statistics.median(one_stats(r)["tok_per_burst"] for r in res), 2),
                          "ttft_max_s": round(max(f - r["t0"] for f, r in zip(firsts, res)), 2),
                          **(spec_delta(m0, m1) or log_spec(a, lm))}, per_med), a.out)


def run_prefill(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ids = tok.encode(story_text(), add_special_tokens=False).ids
    for n in [int(x) for x in a.lengths.split(",")]:
        for rep in range(a.reps):
            nonce = f"[run {time.time_ns()} {rep}] "
            head = tok.encode(nonce, add_special_tokens=False).ids
            body = (ids * (1 + n // max(1, len(ids))))[: n - len(head)]
            text = nonce + tok.decode(body)
            r = request(a.port, "raw", text, 1, 0.0, None, a.model)
            ttft = r["first_evt"] - r["t0"] if r["first_evt"] else None
            pt = r["prompt_tokens"]
            emit({"mode": "prefill", "label": a.label, "target_tokens": n, "prompt_tokens": pt,
                  "ttft_s": round(ttft, 3) if ttft else None,
                  "prefill_tps": round((pt or n) / ttft, 1) if ttft else None}, a.out)


# The L288 scoreboard's single-stream cells (prompt, temperature); the concurrent cells run at T0.
SCOREBOARD_SINGLE = [("code", 0.0), ("prose", 0.0), ("code", 0.7)]


def run_scoreboard(a):
    for name, temp in SCOREBOARD_SINGLE:
        b = argparse.Namespace(**vars(a))
        b.prompts, b.temps = name, str(temp)
        run_single(b)
    for n in [int(x) for x in a.streams_set.split(",") if x]:
        conc_cell(a, n, 0.0, a.conc_tokens)


def run_table(a):
    """One row per cell, one column per label (file order): tok/s (tokens/round, ms/round), and the first two
    labels' tok/s ratio.  Concurrent cells: aggregate tok/s; tokens/round and ms/round per stream."""
    cells, labels = {}, []
    for path in a.files:
        for line in open(path):
            d = json.loads(line)
            if d["mode"] not in ("single", "conc"):
                continue
            key = f"{d['prompt']} T{d['temp']:g}" if d["mode"] == "single" else f"{d['streams']} streams T{d['temp']:g}"
            lab = d.get("label") or os.path.basename(path)
            if lab not in labels:
                labels.append(lab)
            cells.setdefault(key, {})[lab] = d
    two = len(labels) >= 2
    print("| cell | " + " | ".join(f"{l} tok/s (tok/round, ms/round)" for l in labels)
          + (f" | {labels[0]} / {labels[1]}" if two else "") + " |")
    print("|---|" + "---:|" * (len(labels) + (1 if two else 0)))
    for key, by in cells.items():
        row, tps = [], []
        for l in labels:
            d = by.get(l)
            t = (d.get("decode_tps_median", d.get("decode_tps")) if d else None)
            tps.append(t)
            row.append(f"{t} ({d.get('srv_tok_per_round', '-')}, {d.get('srv_ms_per_round', '-')})" if d else "")
        if two:
            row.append(f"{tps[0] / tps[1]:.3f}" if tps[0] and tps[1] else "")
        print(f"| {key} | " + " | ".join(row) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["single", "conc", "prefill", "scoreboard", "table"])
    p.add_argument("files", nargs="*", help="table: the .jsonl files to compare (labels in file order)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="m")
    p.add_argument("--label", default="")
    p.add_argument("--out")
    p.add_argument("--prompts", default="code,codex,prose,story")
    p.add_argument("--temps", default="0")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--lengths", default="1024,8192,32768")
    p.add_argument("--tokenizer")
    p.add_argument("--server-log", help="TensorFold: its server's log, read for the per-request round counts")
    p.add_argument("--streams-set", default="4,8", help="scoreboard: the concurrent-stream cells")
    p.add_argument("--conc-tokens", type=int, default=256, help="scoreboard: tokens per concurrent stream")
    a = p.parse_args()
    {"single": run_single, "conc": run_conc, "prefill": run_prefill, "scoreboard": run_scoreboard,
     "table": run_table}[a.mode](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture teacher-forced per-token top-K logprobs from a vLLM server.

Plan 90 Track A1: run this AGAINST THE WORK RIG (full-fat DSv4-Flash-0731 on
vLLM) to produce the source-truth reference for the fidelity ledger.  Output is
the cross-rig common format consumed by fidelity_compare.py:

  header line: {"format":"ds4-fidelity-jsonl-v1","source":...,"model":...,
                "file":...,"tokens":[...]}          (token ids, prompt order)
  then one line per scored prompt position:
      {"pos":P,"token":ID,"logprob":F_or_null,"topk":[[ID,LOGPROB],...]}

vLLM must be launched with --max-logprobs >= --topk (default cap is 20; ask
for --max-logprobs 64 on the rig if possible, but 20 is usable).

Stdlib only; resumable (skips suite files whose output already exists).

Usage:
  capture_vllm_prompt_logprobs.py --endpoint http://rig:8000 --suite SUITE_DIR \
      [--model auto] [--topk 20] [--out OUT_DIR]
"""
import argparse
import json
import os
import sys
import urllib.request


def api(endpoint, path, payload=None, timeout=600):
    url = endpoint.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pick_model(endpoint, model):
    if model != "auto":
        return model
    models = api(endpoint, "/v1/models")["data"]
    if not models:
        sys.exit("no models served")
    return models[0]["id"]


def tokenize(endpoint, model, text):
    """vLLM /tokenize endpoint; falls back to None if absent."""
    try:
        r = api(endpoint, "/tokenize",
                {"model": model, "prompt": text, "add_special_tokens": True})
        return r.get("tokens")
    except Exception:
        return None


def parse_prompt_logprobs(entries):
    """vLLM prompt_logprobs: list aligned to prompt tokens; first entry null.
    Each non-null entry maps token-id (as str) -> {logprob, rank, ...}; the
    rank-0 entry is the actual prompt token.  Some builds emit lists of
    {token/id/logprob} objects instead; handle both."""
    out = []
    for pos, e in enumerate(entries or []):
        topk = []
        if isinstance(e, dict):
            for tid, info in e.items():
                lp = info.get("logprob") if isinstance(info, dict) else info
                topk.append((int(tid), float(lp)))
            topk.sort(key=lambda t: -t[1])
        elif isinstance(e, list):
            for info in e:
                tid_i = int(info.get("id", info.get("token_id", -1)))
                topk.append((tid_i, float(info["logprob"])))
            topk.sort(key=lambda t: -t[1])
        out.append((pos, topk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--suite", required=True, help="suite dir with manifest.json")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default=None, help="default: <suite>/ref-<model-tag>")
    ap.add_argument("--tokens-dir", default=None,
                    help="directory of <doc>.json pre-tokenized id arrays. "
                         "Posts ids instead of text so BOTH rigs score the "
                         "identical sequence -- required for cross-rig KL, "
                         "since pulsar and vLLM tokenize the same bytes "
                         "differently on 8 of 9 calib docs (measured "
                         "2026-08-05, +1..+64 ids from <think>/DSML folding)")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.suite, "manifest.json")))
    model = pick_model(a.endpoint, a.model)
    tag = model.split("/")[-1].replace(" ", "_")
    out_dir = a.out or os.path.join(a.suite, f"ref-{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"model={model} out={out_dir}", file=sys.stderr)

    for entry in manifest["files"]:
        name = entry["name"]
        out_path = os.path.join(out_dir, name + ".jsonl")
        if os.path.exists(out_path):
            print(f"skip {name} (exists)", file=sys.stderr)
            continue
        pre = None
        if a.tokens_dir:
            tp = os.path.join(a.tokens_dir, name + ".json")
            if not os.path.exists(tp):
                sys.exit(f"{name}: --tokens-dir given but {tp} is missing. "
                         "Every document must be pre-tokenized or none, "
                         "otherwise the reference mixes two conventions.")
            pre = json.load(open(tp))
            if not isinstance(pre, list) or not pre:
                sys.exit(f"{tp}: expected a non-empty JSON array of token ids")

        if pre is not None:
            tokens = pre
            prompt_field = pre          # vLLM accepts a list[int] prompt
            print(f"capture {name}: {len(pre)} PRE-TOKENIZED ids",
                  file=sys.stderr)
        else:
            text = open(os.path.join(a.suite, entry["file"]),
                        encoding="utf-8").read()
            tokens = tokenize(a.endpoint, model, text)
            prompt_field = text
            print(f"capture {name}: {len(text)} chars, "
                  f"{len(tokens) if tokens else '?'} tokens", file=sys.stderr)

        resp = api(a.endpoint, "/v1/completions", {
            "model": model,
            "prompt": prompt_field,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": a.topk,
        }, timeout=3600)
        entries = resp["choices"][0].get("prompt_logprobs")
        if entries is None:
            sys.exit(f"{name}: server returned no prompt_logprobs — "
                     "does this vLLM build support the prompt_logprobs "
                     "sampling param over the API?")
        rows = parse_prompt_logprobs(entries)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps({
                "format": "ds4-fidelity-jsonl-v1",
                "source": "vllm",
                "model": model,
                "file": name,
                "topk": a.topk,
                "tokens": tokens,
                "pretokenized": pre is not None,
            }) + "\n")
            for pos, topk in rows:
                if not topk:
                    continue
                tok = tokens[pos] if tokens and pos < len(tokens) else None
                lp = None
                for tid, tlp in topk:
                    if tok is not None and tid == tok:
                        lp = tlp
                        break
                f.write(json.dumps({
                    "pos": pos,
                    "token": tok,
                    "logprob": lp,
                    "topk": [[t, round(l, 6)] for t, l in topk],
                }) + "\n")
        os.replace(tmp, out_path)
        print(f"wrote {out_path}", file=sys.stderr)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

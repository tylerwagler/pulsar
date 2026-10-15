#!/usr/bin/env python3
"""Teacher-forced logprob capture for an EXL3 pack through exllamav3's own forward (the EXL3 arms of L296).

pulsar does not run Nemotron yet, so the EXL3 arms are graded on exllamav3's forward: the format's own reference
implementation.  Writes the same ds4-fidelity-jsonl-v1 files as tools/fidelity/capture_vllm_prompt_logprobs.py,
with the same top-K and the same "actual token appended when outside the top-K" convention vLLM's prompt_logprobs
has, so grade.py runs one estimator over every arm.

Per document: the suite's frozen ids, forwarded in --chunk-token chunks through a paged cache with the recurrent
(Mamba-2) state carried between chunks, logits taken at every position (log_softmax in fp32 over the real vocab).

  python capture_exl3.py --model /srv/models/nemotron35-exl3/k4 --suite /mnt/models/fmtcmp/suite-v1 \
      --out ~/fmtcmp/caps/exl3-k4 [--selfcheck] [--only chat-00,story]

--selfcheck (run it once per pack, ~1 min): the first --selfcheck-len tokens of the first document forwarded
two ways -- one cache-less pass (attn_mode flash_attn_nc, no recurrent cache) and the chunked cached path at a
deliberately small chunk -- must agree (top-1 >= 99.5%, mean |d logprob(actual)| <= 2e-2).  It proves the chunk
plumbing (KV pages, recurrent-state carry) before any capture is trusted.  Exit 1 when it fails.

Resumable: documents whose output exists are skipped.  Run under gpuq on Sparky.
"""
import argparse
import json
import os
import sys
import time

import torch

TOPK = 64


def load_model(path, max_tokens):
    from exllamav3 import Cache, Config, Model
    config = Config.from_directory(path)
    model = Model.from_config(config)
    pages = (max_tokens + 255) // 256 * 256
    cache = Cache(model, max_num_tokens=pages, max_batch_size=1)
    model.load(progressbar=True)
    return config, model, cache, pages


def forward_cached(model, cache, pages, ids, chunk, vocab):
    """Log-probs [T, vocab] (fp32, on GPU in chunks, returned per chunk through a generator)."""
    T = ids.shape[1]
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, pages)}
    try:
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            params["past_len"] = s
            logits = model.forward(ids[:, s:e], params)
            yield s, torch.log_softmax(logits[0, :, :vocab].float(), dim=-1)
            del logits
    finally:
        for rs in params.get("recurrent_states") or []:
            rs.free()


def forward_nc(model, ids, vocab):
    logits = model.forward(ids, {"attn_mode": "flash_attn_nc"})
    return torch.log_softmax(logits[0, :, :vocab].float(), dim=-1)


def rows_from(lp, s, ids_list, T):
    """lp[i] predicts token s+i+1.  Returns jsonl rows in capture_vllm_prompt_logprobs.py's format."""
    out = []
    vals, idx = lp.topk(TOPK, dim=-1)
    vals, idx = vals.cpu().tolist(), idx.cpu().tolist()
    tgt_pos = [s + i + 1 for i in range(lp.shape[0]) if s + i + 1 < T]
    if tgt_pos:
        tgt = torch.tensor([ids_list[p] for p in tgt_pos], device=lp.device)
        tlp = lp[torch.arange(len(tgt_pos), device=lp.device), tgt].cpu().tolist()
    for i, p in enumerate(tgt_pos):
        top = [[t, round(v, 6)] for t, v in zip(idx[i], vals[i])]
        tok = ids_list[p]
        if tok not in idx[i]:
            top.append([tok, round(tlp[i], 6)])
        out.append({"pos": p, "token": tok, "logprob": tlp[i], "topk": top})
    return out


def selfcheck(model, cache, pages, ids_list, n, chunk, vocab):
    ids = torch.tensor([ids_list[:n]], dtype=torch.long)
    ref = forward_nc(model, ids, vocab)
    a_top = ref.argmax(-1)
    a_tgt = ref[torch.arange(n - 1), ids[0, 1:].to(ref.device)]
    agree, dsum = 0, 0.0
    for s, lp in forward_cached(model, cache, pages, ids, chunk, vocab):
        e = s + lp.shape[0]
        agree += (lp.argmax(-1) == a_top[s:e]).sum().item()
        m = min(e, n - 1) - s
        if m > 0:
            b_tgt = lp[torch.arange(m, device=lp.device), ids[0, s + 1:s + 1 + m].to(lp.device)]
            dsum += (b_tgt - a_tgt[s:s + m]).abs().sum().item()
    t1, dlp = agree / n, dsum / (n - 1)
    ok = t1 >= 0.995 and dlp <= 2e-2
    print(f"[selfcheck] {n} tokens: cache-less vs chunked(cache, chunk {chunk}): top-1 {100 * t1:.2f}%, "
          f"mean |d logprob| {dlp:.3e} -> {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="EXL3 pack directory")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--only", default=None, help="comma list of document names")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--selfcheck-len", type=int, default=1024)
    ap.add_argument("--selfcheck-chunk", type=int, default=160)
    ap.add_argument("--deadline", type=float, default=0, help="unix time: stop before starting a doc past it")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.suite, "manifest.json")))
    docs = man["files"]
    if a.only:
        keep = set(a.only.split(","))
        docs = [d for d in docs if d["name"] in keep]
    max_len = max(d["n_tokens"] for d in man["files"])
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    config, model, cache, pages = load_model(a.model, max_len)
    vocab = config.vocab_size
    print(f"[exl3] loaded {a.model} in {time.time() - t0:.0f}s; vocab {vocab}, cache {pages} tokens",
          file=sys.stderr)

    with torch.inference_mode():
        if a.selfcheck:
            first = json.load(open(os.path.join(a.suite, man["files"][0]["tokens"])))
            if not selfcheck(model, cache, pages, first, min(a.selfcheck_len, len(first)), a.selfcheck_chunk, vocab):
                sys.exit(1)
        for d in docs:
            outp = os.path.join(a.out, d["name"] + ".jsonl")
            if os.path.exists(outp):
                continue
            if a.deadline and time.time() > a.deadline:
                print("[exl3] deadline reached; rerun to resume", file=sys.stderr)
                break
            ids_list = json.load(open(os.path.join(a.suite, d["tokens"])))
            T = len(ids_list)
            ids = torch.tensor([ids_list], dtype=torch.long)
            t1 = time.time()
            rows = []
            for s, lp in forward_cached(model, cache, pages, ids, a.chunk, vocab):
                rows += rows_from(lp, s, ids_list, T)
            tmp = outp + ".tmp"
            with open(tmp, "w") as f:
                f.write(json.dumps({"format": "ds4-fidelity-jsonl-v1", "source": "exllamav3", "model": a.model,
                                    "file": d["name"], "topk": TOPK, "tokens": ids_list, "pretokenized": True,
                                    "chunk": a.chunk}) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, outp)
            print(f"[exl3] {d['name']}: {T} tokens in {time.time() - t1:.1f}s", file=sys.stderr)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

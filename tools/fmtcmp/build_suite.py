#!/usr/bin/env python3
"""Build the frozen L296 format-comparison suite: one token-id sequence per document.

Every arm (BF16 on the H100s, NVIDIA's NVFP4 on vLLM, our EXL3 K4 / K5 on exllamav3) teacher-forces the SAME
ids, so per-position KL pairs exactly.  The suite is built once, here, and shipped frozen (manifest.json carries
the sha256 of every file); a change is a new suite version, never an in-place rebuild.

Documents (suite-v1):
  - chat-NN  The L251 held-out instrument: the 48 rows of calib-qwen38-v1.jsonl that L251's full-model KL graded
             (capture_meta.json "heldout_rows"), mapped back to their source conversations in the reap corpus
             cases and re-rendered through Nemotron's own chat template (tools, reasoning, tool results), then cut
             to the first --seq-len tokens, as L251 cut them.  Classes: agentic glaive / hermes-fc /
             hermes-reasoning / swe, code-opencode, longprose-wikitext, prose-ultrachat.
  - story    The decode-reference "story" prompt (Bellwether harbour, ~31k tokens, assignment recall at the end):
             the long-context sample.
  - code     The decode-reference "code" prompt (the generated C audit log).
  story / code are the system + user messages of the ref-qwen38fn-q fixtures, rendered with the generation
  prompt and thinking off, as their manifests record for Qwen.

Run on Sparky (needs transformers + the BF16 checkpoint's tokenizer; CPU only):
  ~/inkling/.venv/bin/python build_suite.py --out /mnt/models/fmtcmp/suite-v1
"""
import argparse
import hashlib
import json
import os
import re
import sys

DEF_TOKENIZER = "/mnt/models/nemotron35-super-vl-bf16"
DEF_CALIB = "/mnt/models/qwen38-calib/calib-qwen38-v1.jsonl"
DEF_META = os.path.expanduser("~/fmtcmp/inputs/l251-capture_meta.json")
DEF_CASES = "/mnt/models/reap-corpus/cases"
DEF_REFDIR = "/mnt/models/sparky-parked/home-claude/ref-qwen38fn-q"

QWEN_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def first_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def match_case(row_text, cases):
    """The source conversation of a calibration row: the case whose message contents appear in the row most often,
    ties broken by the number of user turns.  Refuses a non-unique best."""
    n_user = row_text.count("<|im_start|>user")
    scored = []
    for ci, c in enumerate(cases):
        hits = 0
        for m in c["messages"]:
            t = first_text(m.get("content")).strip()[:200]
            if t and t in row_text:
                hits += 1
        users = sum(1 for m in c["messages"] if m["role"] == "user")
        scored.append(((hits, users == n_user), ci))
    scored.sort(reverse=True)
    best, runner = scored[0], scored[1] if len(scored) > 1 else None
    if best[0][0] == 0:
        sys.exit("no source case matches a held-out row")
    if runner is not None and runner[0] == best[0]:
        # identical score: the cases must render identically up to the cut or the choice is ambiguous
        return best[1], True
    return best[1], False


def normalize_messages(msgs):
    """Tool-call arguments are JSON strings in the corpus; the template iterates them as a mapping (vLLM decodes them
    once the same way).  A string that is not a JSON object stays a string."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        v = json.loads(a)
                        if isinstance(v, dict):
                            fn["arguments"] = v
                    except ValueError:
                        pass
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=DEF_TOKENIZER)
    ap.add_argument("--calib", default=DEF_CALIB)
    ap.add_argument("--meta", default=DEF_META, help="L251 per-expert-kl capture_meta.json (held-out row list)")
    ap.add_argument("--cases", default=DEF_CASES)
    ap.add_argument("--refdir", default=DEF_REFDIR, help="dir with story.rendered.txt / code.rendered.txt")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--version", default="suite-v1")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    if os.path.exists(os.path.join(a.out, "manifest.json")):
        sys.exit(f"{a.out} already holds a suite; suites are immutable (build a new version)")
    os.makedirs(os.path.join(a.out, "tokens"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "text"), exist_ok=True)

    rows = [json.loads(l) for l in open(a.calib)]
    meta = json.load(open(a.meta))
    files = []

    def emit(name, cls, text, ids, extra):
        tp = os.path.join("tokens", name + ".json")
        xp = os.path.join("text", name + ".txt")
        with open(os.path.join(a.out, tp), "w") as f:
            json.dump(ids, f)
        with open(os.path.join(a.out, xp), "w", encoding="utf-8") as f:
            f.write(text)
        e = {"name": name, "class": cls, "file": xp, "tokens": tp, "n_tokens": len(ids),
             "sha256_tokens": sha256_file(os.path.join(a.out, tp)),
             "sha256_text": sha256_file(os.path.join(a.out, xp))}
        e.update(extra)
        files.append(e)
        print(f"{name:10s} {cls:26s} {len(ids):6d} tokens", file=sys.stderr)

    # chat-NN: the L251 held-out rows, re-rendered for Nemotron
    case_cache = {}
    for k, (ri, src) in enumerate(zip(meta["heldout_rows"], meta["heldout_sources"])):
        row = rows[ri]
        if row["source"] != src:
            sys.exit(f"row {ri}: source {row['source']} != capture_meta {src}")
        if src not in case_cache:
            case_cache[src] = json.load(open(os.path.join(a.cases, src)))
        cases = case_cache[src]
        ci, tie = match_case(row["text"], cases)
        case = cases[ci]
        kw = {"enable_thinking": case.get("thinking_mode", "thinking") == "thinking"}
        if case.get("tools"):
            kw["tools"] = case["tools"]
        text = tok.apply_chat_template(normalize_messages(case["messages"]), tokenize=False,
                                       add_generation_prompt=False, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"][: a.seq_len]
        cls = src.rsplit("-", 1)[0]
        emit(f"chat-{k:02d}", cls, text, ids,
             {"l251_row": ri, "source": src, "case": ci, "case_tie": tie, "full_tokens":
              len(tok(text, add_special_tokens=False)["input_ids"])})

    # story / code, rendered as their fixtures were: story = system + user through the chat template with the
    # generation prompt and thinking off; code = raw text (its manifest: "raw text, add_special_tokens=False").
    for name in ("code", "story"):
        src_path = os.path.join(a.refdir, f"{name}.rendered.txt")
        fx = json.load(open(os.path.join(a.refdir, f"{name}_manifest.json")))
        rendered = open(src_path, encoding="utf-8").read()
        if fx["render"].startswith("raw text"):
            text, render = rendered, "raw"
        elif fx["render"].startswith("apply_chat_template(system+user"):
            msgs = [{"role": r, "content": c} for r, c in QWEN_TURN.findall(rendered)]
            if [m["role"] for m in msgs] != ["system", "user"]:
                sys.exit(f"{name}: expected system+user turns, got {[m['role'] for m in msgs]}")
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            render = "chat, generation prompt, thinking off"
        else:
            sys.exit(f"{name}: unknown fixture render {fx['render']!r}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        emit(name, name, text, ids, {"source": src_path, "sha256_source": sha256_file(src_path), "render": render})

    manifest = {
        "format": "fmtcmp-suite-v1",
        "version": a.version,
        "tokenizer": a.tokenizer,
        "sha256_tokenizer_json": sha256_file(os.path.join(a.tokenizer, "tokenizer.json")),
        "sha256_chat_template": sha256_file(os.path.join(a.tokenizer, "chat_template.jinja")),
        "transformers": transformers.__version__,
        "calib": a.calib, "sha256_calib": sha256_file(a.calib),
        "meta": a.meta, "sha256_meta": sha256_file(a.meta),
        "seq_len": a.seq_len,
        "n_docs": len(files),
        "n_tokens": sum(f["n_tokens"] for f in files),
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(files)} docs, {manifest['n_tokens']} tokens -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

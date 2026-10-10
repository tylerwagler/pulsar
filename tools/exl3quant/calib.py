#!/usr/bin/env python3
"""The calibration rows for V4.1's EXL3 quantization: coding-heavy, rendered in V4.1's own prompt format.

The model serves Claude Code-style agentic coding (long multi-turn sessions whose tokens are mostly tool
results -- file contents, command output -- around short tool calls and code), so the Hessians are taken over
that distribution, not exllamav3's default web/wiki mix.  Every conversation is rendered by the snapshot's own
`encoding/encoding.py` (V4.1's DSML tool-call format, its <think> handling, BOS) and tokenized with the snapshot's
tokenizer, so the rows are what the served model sees.

The mix (rows of --cols tokens; the default 64 x 4096 = 262,144 tokens, ~91% code / agentic):

  source        rows  what
  agentic-swe     20  SWE-agent trajectories: an autonomous programmer in a shell over real GitHub repos --
                      file views, searches, edits, test runs (reap-corpus agentic-swe, 104 cases)
  repo-reads      16  synthetic coding-agent sessions over LOCAL source trees (this engine's C++/CUDA,
                      exllamav3, llama.cpp, vLLM): Grep -> Read -> Read -> Edit tool calls whose results are
                      the real files, `cat -n` numbered -- the bulk of a Claude Code context
  code-opencode   12  code generation / explanation (OpenCodeInstruct)
  agentic-tools   10  function calling with and without reasoning (hermes reasoning-tool-use, thinking mode;
                      hermes-fc), tool schemas in the system prompt
  prose            6  ultrachat dialogue + wikitext, so the chat / explanation text around the code is covered

A conversation of at least one row gives up to --max-rows-per-case of its whole-row windows, picked at random, so
a single long trajectory cannot dominate and rows are not all the same opening system prompt -- a mid-session row
starts without BOS, as a long served context does past its first chunk.  Shorter conversations (each starting at
BOS) are packed back to back.  --holdout rows are drawn the same way (from cases no calibration row used) and are
forwarded by v41_stream.py but kept out of every Hessian: they are the held-out activations grade_layer.py
scores the quantized experts on.

    .venv/bin/python tools/exl3quant/calib.py --ref $V41 --corpus /mnt/models/reap-corpus \\
        --repo engine=$PWD/src --repo exllamav3=~/exllamav3/exllamav3 ... --out calib-v41-code-v1.safetensors

Output (safetensors, numpy): rows I64 [R, cols], holdout BOOL [R], source I32 [R] (index into the metadata's
`sources` list); metadata: the mix, cols, seed, the encoder and tokenizer SHA-256s.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random

import numpy as np

MIX = (("agentic-swe", 20), ("repo-reads", 16), ("code-opencode", 12), ("agentic-tools", 10), ("prose", 6))
HOLDOUT = (("agentic-swe", 1), ("repo-reads", 1), ("code-opencode", 1), ("agentic-tools", 1))
CORPUS_FILES = {
    "agentic-swe": (("agentic-swe.jsonl", "chat"),),
    "code-opencode": (("code-opencode.jsonl", "chat"),),
    "agentic-tools": (("agentic-hermes-reasoning.jsonl", "thinking"), ("agentic-hermes-fc.jsonl", "chat")),
    "prose": (("prose-ultrachat.jsonl", "chat"), ("longprose-wikitext.jsonl", "chat")),
}
REPO_EXT = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".rs", ".go", ".ts", ".md", ".txt", ".toml",
            ".cmake", ".sh", ".json", ".yaml", ".yml")
READ_LINES = 400        # a Read tool result: the first 400 numbered lines, as a coding agent's Read returns

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "Read", "description": "Read a file from the local filesystem. Returns the contents with line "
        "numbers (cat -n format), up to 400 lines from offset.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute path of the file to read"},
            "offset": {"type": "integer", "description": "Line to start reading from"},
            "limit": {"type": "integer", "description": "Number of lines to read"}},
            "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "Grep", "description": "Search file contents with a regular expression (ripgrep). Returns "
        "matching lines as path:line:text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "The regular expression to search for"},
            "path": {"type": "string", "description": "Directory or file to search in"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "Edit", "description": "Replace an exact string in a file with a new string.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
            "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
AGENT_SYSTEM = ("You are an expert software engineer working as a coding agent in the repository at {root}. "
                "Use the tools to inspect and change the code. Read the code before changing it, keep changes "
                "minimal, and explain what you found.")
TASKS = ("Where is `{sym}` defined and how is it used? Walk me through it.",
         "I think there's a bug around `{sym}` in {file}. Can you take a look?",
         "Explain what {file} does, focusing on `{sym}`.",
         "Add a short comment documenting `{sym}` in {file}.",
         "Review {file} -- is `{sym}` handled correctly at the edges?")


def load_encoder(ref_dir):
    path = os.path.join(ref_dir, "encoding", "encoding.py")
    spec = importlib.util.spec_from_file_location("v41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256_file(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_cases(corpus, source, rng):
    """[(messages, thinking_mode)] of a corpus-backed source, shuffled; tools ride on the first message."""
    cases = []
    for fname, mode in CORPUS_FILES[source]:
        for line in open(os.path.join(corpus, fname)):
            d = json.loads(line)
            msgs = d["messages"]
            if d.get("tools"):
                if msgs[0]["role"] != "system":
                    msgs = [{"role": "system", "content": ""}] + msgs
                msgs[0] = dict(msgs[0], tools=d["tools"])
            cases.append((msgs, mode))
    rng.shuffle(cases)
    return cases


def repo_files(repos):
    """{repo name: (root, [relative paths])} of the text source files under each root."""
    out = {}
    for name, root in repos:
        root = os.path.abspath(os.path.expanduser(root))
        files = []
        for d, dirs, fs in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in ("build", "node_modules",
                                                                                     "__pycache__", "third_party"))
            for f in sorted(fs):
                p = os.path.join(d, f)
                if f.endswith(REPO_EXT) and 2_000 <= os.path.getsize(p) <= 400_000:
                    files.append(os.path.relpath(p, root))
        if not files:
            raise SystemExit(f"--repo {name}={root}: no source files")
        out[name] = (root, files)
    return out


def numbered(text, start=1):
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(text.splitlines(), start))


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def identifiers(text, rng):
    import re
    ids = [m for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{5,40}\b", text) if not m.isupper() or "_" in m]
    return rng.choice(ids) if ids else "main"


def repo_case(repos, rng):
    """One synthetic coding-agent session over a real source tree: Grep for a symbol, Read the file it lives in,
    Read a neighbour, Edit (a comment inserted above real lines), a short answer.  The tool results are the real
    files; the assistant turns are short and templated (their tokens are a small share of the row)."""
    name = rng.choice(sorted(repos))
    root, files = repos[name]
    vroot = f"/work/{name}"
    f1 = rng.choice(files)
    t1 = read_text(os.path.join(root, f1))
    if t1 is None:
        return None
    sym = identifiers(t1, rng)
    same_dir = [f for f in files if os.path.dirname(f) == os.path.dirname(f1) and f != f1]
    f2 = rng.choice(same_dir or files)
    t2 = read_text(os.path.join(root, f2)) or ""
    lines1 = t1.splitlines()
    hits = [f"{f1}:{i + 1}:{line.strip()[:160]}" for i, line in enumerate(lines1) if sym in line][:12]
    hits += [f"{f2}:{i + 1}:{line.strip()[:160]}" for i, line in enumerate(t2.splitlines()) if sym in line][:6]
    at = next((i for i, line in enumerate(lines1) if sym in line), 0)
    old = "\n".join(lines1[at:at + 3])
    cm = "#" if f1.endswith((".py", ".sh", ".toml", ".yaml", ".yml", ".cmake")) else "//"
    indent = lines1[at][:len(lines1[at]) - len(lines1[at].lstrip())] if lines1 else ""
    new = f"{indent}{cm} {sym}: see the call sites in {os.path.basename(f2)}\n{old}"

    def call(i, fn, **args):
        return {"id": f"call_{i}", "type": "function", "function": {"name": fn, "arguments": json.dumps(args)}}
    msgs = [
        {"role": "system", "content": AGENT_SYSTEM.format(root=vroot), "tools": AGENT_TOOLS},
        {"role": "user", "content": rng.choice(TASKS).format(sym=sym, file=f1)},
        {"role": "assistant", "content": f"Let me find where `{sym}` appears.",
         "tool_calls": [call(1, "Grep", pattern=sym, path=vroot)]},
        {"role": "tool", "tool_call_id": "call_1", "content": "\n".join(hits) or "No matches found"},
        {"role": "assistant", "content": "", "tool_calls": [call(2, "Read", file_path=f"{vroot}/{f1}")]},
        {"role": "tool", "tool_call_id": "call_2", "content": numbered("\n".join(lines1[:READ_LINES]))},
        {"role": "assistant", "content": f"`{sym}` is used in {f1}. Let me check {f2} as well.",
         "tool_calls": [call(3, "Read", file_path=f"{vroot}/{f2}")]},
        {"role": "tool", "tool_call_id": "call_3", "content": numbered("\n".join(t2.splitlines()[:READ_LINES]))},
        {"role": "assistant", "content": "",
         "tool_calls": [call(4, "Edit", file_path=f"{vroot}/{f1}", old_string=old, new_string=new)]},
        {"role": "tool", "tool_call_id": "call_4", "content": f"The file {vroot}/{f1} has been updated."},
        {"role": "assistant", "content": f"`{sym}` is defined in `{f1}` (line {at + 1}) and referenced from "
                                         f"`{f2}`. I added a comment pointing at its call sites."},
    ]
    return msgs, "chat"


class Packer:
    """Rendered conversations -> rows of `cols` tokens.  A conversation of at least `cols` tokens gives up to `cap`
    of its whole-row windows, chosen at random (not always its opening, which for a trajectory corpus is the same
    long system prompt every time); shorter ones are packed back to back."""

    def __init__(self, enc, tok, cols, cap, rng):
        self.enc, self.tok, self.cols, self.cap, self.rng = enc, tok, cols, cap, rng
        self.ready, self.buf = [], []
        self.rendered = self.failed = 0

    def add(self, msgs, mode):
        try:
            text = self.enc.encode_messages(msgs, thinking_mode=mode)
        except Exception:  # noqa: BLE001 -- a malformed corpus case (bad tool-call JSON etc.) is skipped, counted
            self.failed += 1
            return
        self.rendered += 1
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if len(ids) < self.cols:
            self.buf += ids
            while len(self.buf) >= self.cols:
                self.ready.append(self.buf[:self.cols])
                self.buf = self.buf[self.cols:]
            return
        windows = [ids[i:i + self.cols] for i in range(0, len(ids) - self.cols + 1, self.cols)]
        self.ready += [windows[i] for i in sorted(self.rng.sample(range(len(windows)), min(self.cap, len(windows))))]

    def take(self, n):
        out, self.ready = self.ready[:n], self.ready[n:]
        return out

    def reset(self):
        self.ready, self.buf = [], []


def build(ref, corpus, repos, cols, mix, holdout, seed, cap):
    from tokenizers import Tokenizer
    enc, enc_sha = load_encoder(ref)
    tok = Tokenizer.from_file(os.path.join(ref, "tokenizer.json"))
    rng = random.Random(seed)
    trees = repo_files(repos) if repos else {}
    names = [s for s, _ in mix]
    rows, src, held, report = [], [], [], {}
    for s, n_cal in mix:
        n_hold = dict(holdout).get(s, 0)
        if s == "repo-reads":
            if not trees:
                raise SystemExit("repo-reads in the mix: pass --repo name=root (one or more)")
            cases = iter(lambda: repo_case(trees, rng), object())
        else:
            cases = iter(corpus_cases(corpus, s, rng))
        pk = Packer(enc, tok, cols, cap, rng)
        got = []
        for want in (n_cal, n_hold):        # holdout rows come after, from conversations no calibration row used
            pk.reset()
            part = []
            while len(part) < want:
                c = next(cases, None)
                if c is None:
                    raise SystemExit(f"{s}: the corpus ran out at {len(part)} of {want} rows")
                if c is not None:          # a repo file that is not UTF-8 text yields no session
                    pk.add(*c)
                part += pk.take(want - len(part))
            got.append(part)
        for part, is_hold in zip(got, (False, True)):
            rows += part
            src += [names.index(s)] * len(part)
            held += [is_hold] * len(part)
        report[s] = {"rows": len(got[0]), "holdout": len(got[1]), "conversations": pk.rendered,
                     "render_failures": pk.failed}
        print(f"calib: {s}: {report[s]}", flush=True)
    meta = {"sources": json.dumps(names), "mix": json.dumps(report), "cols": str(cols), "seed": str(seed),
            "max_rows_per_case": str(cap), "encoder_sha256": enc_sha,
            "tokenizer_sha256": sha256_file(os.path.join(ref, "tokenizer.json")),
            "repos": json.dumps({k: v[0] for k, v in trees.items()})}
    return (np.asarray(rows, dtype=np.int64), np.asarray(held, dtype=np.bool_), np.asarray(src, dtype=np.int32),
            meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="the V4.1 snapshot (encoding/encoding.py, tokenizer.json)")
    ap.add_argument("--corpus", required=True, help="the reap-corpus directory (agentic-swe.jsonl, ...)")
    ap.add_argument("--repo", action="append", default=[], metavar="NAME=ROOT",
                    help="a local source tree for the repo-reads sessions (repeatable)")
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source's row count (and holdout)")
    ap.add_argument("--max-rows-per-case", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repos = [tuple(r.split("=", 1)) for r in a.repo]
    mix = [(s, max(1, round(n * a.scale))) for s, n in MIX]
    hold = [(s, max(1, round(n * a.scale))) for s, n in HOLDOUT]
    rows, held, src, meta = build(a.ref, a.corpus, repos, a.cols, mix, hold, a.seed, a.max_rows_per_case)
    from safetensors.numpy import save_file
    save_file({"rows": rows, "holdout": held, "source": src}, a.out, metadata=meta)
    print(f"calib: {rows.shape[0]} rows x {rows.shape[1]} ({int((~held).sum())} calibration, {int(held.sum())} "
          f"held out) -> {a.out}")


if __name__ == "__main__":
    main()

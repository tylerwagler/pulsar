#!/usr/bin/env python3
"""Assemble the fidelity suite (Plan 90 A1): a versioned set of plain-text
documents both rigs teacher-force identically.

Contents (suite-v1):
  - short-XX.txt: 12 diverse calibration texts (~1-4 KB each) from
    calib-diverse-ds4-v1.jsonl (deterministic stride over the 220 entries)
  - long-mixed.txt: concatenation of further calib texts to ~256 KB — ONE long
    document whose positions give the whole drift-vs-depth curve at compare
    time (buckets 0-2k / 2k-9k / 9k-38k / 38k+ tokens)
  - dsml-tools.txt: a static DSML tool-call transcript in the DS4 chat wire
    format (numerics probe for the tool-format token distributions; content is
    synthetic and versioned with the suite)

The suite is content-addressed: manifest.json records sha256 of every file.
Never regenerate in place — a changed suite is a NEW suite version.

Usage: build_suite.py --calib /path/calib-diverse-ds4-v1.jsonl --out SUITE_DIR
"""
import argparse
import hashlib
import json
import os

DSML_SAMPLE = (
    "<｜User｜>Sample transcript for numeric probing of tool-format "
    "regions follows.\nWhat is the disk usage of /var/log on this host?"
    "<｜Assistant｜></think>I'll check the disk usage for you.\n\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"bash\">\n"
    "<｜DSML｜parameter name=\"command\" string=\"true\">du -sh /var/log"
    "</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    "<｜User｜><tool_result>1.2G\t/var/log</tool_result>"
    "<｜Assistant｜></think>/var/log is using 1.2 GiB. The largest "
    "contributors are usually journal files; you can trim them with "
    "journalctl --vacuum-size=500M if needed.<｜end▁of▁sentence｜>"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-short", type=int, default=12)
    ap.add_argument("--long-bytes", type=int, default=256 * 1024)
    a = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(a.calib, encoding="utf-8")]
    os.makedirs(a.out, exist_ok=False)  # refuse to overwrite a suite

    files = []

    # Deterministic diverse picks: stride over the calib entries that clear
    # the teacher-forcing minimum (the engine KL harness scores nothing under
    # its 32-token prefix; small docs also bucket poorly), then fill the long
    # doc from the remaining entries in order.
    eligible = [i for i, t in enumerate(texts) if len(t.encode("utf-8")) >= 2000]
    stride = max(1, len(eligible) // a.n_short)
    short_idx = eligible[::stride][:a.n_short]
    for i, idx in enumerate(short_idx):
        name = f"short-{i:02d}"
        path = os.path.join(a.out, name + ".txt")
        open(path, "w", encoding="utf-8").write(texts[idx])
        files.append({"name": name, "file": name + ".txt",
                      "kind": "short", "calib_index": idx})

    long_parts = []
    size = 0
    for idx, t in enumerate(texts):
        if idx in short_idx:
            continue
        long_parts.append(t)
        size += len(t.encode("utf-8")) + 2
        if size >= a.long_bytes:
            break
    path = os.path.join(a.out, "long-mixed.txt")
    open(path, "w", encoding="utf-8").write("\n\n".join(long_parts))
    files.append({"name": "long-mixed", "file": "long-mixed.txt",
                  "kind": "long", "parts": len(long_parts)})

    path = os.path.join(a.out, "dsml-tools.txt")
    open(path, "w", encoding="utf-8").write(DSML_SAMPLE)
    files.append({"name": "dsml-tools", "file": "dsml-tools.txt",
                  "kind": "dsml"})

    for f in files:
        f["sha256"] = sha256(os.path.join(a.out, f["file"]))
    manifest = {
        "suite": os.path.basename(a.out.rstrip("/")),
        "calib_source": os.path.basename(a.calib),
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"suite {manifest['suite']}: {len(files)} files")


if __name__ == "__main__":
    main()

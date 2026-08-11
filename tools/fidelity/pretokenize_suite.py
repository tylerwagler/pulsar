#!/usr/bin/env python3
"""Freeze one canonical token-id sequence per suite document.

WHY THIS EXISTS. Cross-rig KL pairs position p on both sides, so the two
engines must walk the SAME id sequence. They do not agree on text: measured
2026-08-05 against the full-fat 0731 reference, pulsar and vLLM tokenized 8 of
9 calib documents differently (+1, +5, +13, +14, +19, +31, +36, +64 ids), and
long-mixed by +363. Neither is wrong -- pulsar's tokenize_rendered_chat folds
<think>/DSML markers into single special ids exactly as the served model does,
and vLLM's /tokenize does not. But the disagreement silently destroys position
alignment, and the resulting table still renders plausible numbers.

Once a suite is frozen this way, both rigs consume ids:
    pulsar : --kl-tokens tokens/<doc>.json
    vLLM   : capture_vllm_prompt_logprobs.py --tokens-dir tokens/

WHICH TOKENIZATION TO FREEZE. Default is pulsar's, taken from a capture's
sidecars or jsonl headers, because it is what the SERVED model actually sees --
so the measurement describes production rather than a convention no engine
runs. Teacher forcing scores whatever sequence it is handed, so the reference
rig scoring pulsar's ids is not a distortion: both sides score one fixed
sequence, which is exactly the definition of the comparison we want.

Suites are immutable -- freezing tokens makes a new suite version. Do not
re-freeze in place once reference captures exist against it.
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def ids_from_jsonl(path):
    with open(path) as f:
        return json.loads(f.readline()).get("tokens")


def ids_from_sidecar(path):
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--from-capture", required=True,
                    help="directory of <doc>.jsonl captures, or of "
                         "<doc>.klrf.tokens.json sidecars, to take ids from")
    ap.add_argument("--out", default=None,
                    help="default: <suite>/tokens")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen set (makes every "
                         "reference captured against it invalid)")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.suite, "manifest.json")))
    out_dir = a.out or os.path.join(a.suite, "tokens")
    if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "*.json")) \
            and not a.force:
        sys.exit(f"{out_dir} already holds a frozen tokenization. Freezing "
                 "again invalidates every reference captured against it -- "
                 "make a new suite version, or pass --force if you are sure.")
    os.makedirs(out_dir, exist_ok=True)

    frozen = {}
    for entry in manifest["files"]:
        name = entry["name"]
        jl = os.path.join(a.from_capture, name + ".jsonl")
        sc = os.path.join(a.from_capture, name + ".klrf.tokens.json")
        if os.path.exists(jl):
            ids = ids_from_jsonl(jl)
        elif os.path.exists(sc):
            ids = ids_from_sidecar(sc)
        else:
            sys.exit(f"{name}: no ids found in {a.from_capture} "
                     f"(looked for {name}.jsonl and {name}.klrf.tokens.json)")
        if not ids:
            sys.exit(f"{name}: capture carried no token array")
        p = os.path.join(out_dir, name + ".json")
        json.dump(ids, open(p, "w"))
        frozen[name] = {"n_tokens": len(ids),
                        "sha256": hashlib.sha256(
                            json.dumps(ids).encode()).hexdigest()}
        print(f"froze {name}: {len(ids)} ids")

    meta = {"suite": os.path.basename(os.path.abspath(a.suite)),
            "source": os.path.abspath(a.from_capture),
            "convention": "pulsar tokenize_rendered_chat (served-model "
                          "special-token handling)",
            "documents": frozen}
    json.dump(meta, open(os.path.join(out_dir, "FROZEN.json"), "w"), indent=1)
    print(f"\nwrote {len(frozen)} documents to {out_dir}")
    print("both rigs must now be driven from these ids:")
    print(f"  pulsar: --kl-tokens {out_dir}/<doc>.json")
    print(f"  vLLM  : capture_vllm_prompt_logprobs.py --tokens-dir {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A one-document copy of a frozen fmtcmp suite (the R1 smoke): one_doc_suite.py SUITE DOC OUT_DIR."""
import json
import os
import shutil
import sys

suite, doc, out = sys.argv[1:4]
man = json.load(open(os.path.join(suite, "manifest.json")))
files = [f for f in man["files"] if f["name"] == doc]
if len(files) != 1:
    sys.exit(f"{doc} is not in {suite}")
for sub in ("tokens", "text"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
shutil.copy(os.path.join(suite, files[0]["tokens"]), os.path.join(out, files[0]["tokens"]))
shutil.copy(os.path.join(suite, files[0]["file"]), os.path.join(out, files[0]["file"]))
man.update(files=files, n_docs=1, n_tokens=files[0]["n_tokens"], version=man["version"] + f"-only-{doc}")
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)

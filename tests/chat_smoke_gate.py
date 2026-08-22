#!/usr/bin/env python3
"""Chat-decode smoke gate: the two-minute check that was missing.

On 2026-08-22 every one of 17 gates was green while serving emitted BOS-token
salad: the n=1 decode GEMV wrote f32 rows into the f16 Q buffer, and every
decode gate compared that garbage to itself.  The reference gate anchors
PREFILL only; nothing semantic ever looked at a generated token.

This gate greedy-generates one short chat answer through the ENGINE (CLI, no
server) and asserts three things no us-vs-us comparison can fake:
  1. the answer contains the arithmetic result ("12");
  2. no BOS token appears in the continuation;
  3. generation terminated by stop, not by drowning in one repeated token.

Model-dependent, ~2 minutes.  If this gate fails, do not trust any decode
number from the same build -- including benchmarks, which measure garbage at
full speed.

usage: chat_smoke_gate.py MODEL [PULSAR_BIN]
"""
import subprocess, sys, re

model = sys.argv[1]
binary = sys.argv[2] if len(sys.argv) > 2 else "./pulsar"
try:
    out = subprocess.run(
        [binary, "-m", model, "-p", "What is 7+5? Answer with just the number."],
        capture_output=True, text=True, timeout=600).stdout
except subprocess.TimeoutExpired:
    print("CHAT SMOKE GATE: FAIL (generation did not terminate in 600s)")
    sys.exit(1)

tail = out[-4000:]
bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
bos_lit = bos.encode().decode("unicode_escape")
fails = []
if "12" not in tail:
    fails.append("expected answer '12' not found")
if bos_lit in tail:
    fails.append("BOS token present in continuation (degenerate decode)")
words = re.findall(r"\S+", tail)
if words and len(set(words)) < max(2, len(words) // 8):
    fails.append("continuation is one token repeated (degenerate decode)")

if fails:
    print("CHAT SMOKE GATE: FAIL")
    for f in fails: print("  -", f)
    print("  tail:", repr(tail[-300:]))
    sys.exit(1)
print("CHAT SMOKE GATE: PASS (greedy chat answer is semantically sane)")

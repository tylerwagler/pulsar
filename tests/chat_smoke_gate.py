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

L220 -- WHY THIS IS NOT ONE 600 s SUBPROCESS TIMEOUT ANY MORE.
A single total budget conflates two different things: a COLD LOAD of the 92 GB
artifact, which is slow but is READING THE DISK, and a genuine HANG, which
makes no progress at all.  The one flake this gate has produced was the first:
the child had read 92.8 GB at 96% GPU when 600 s expired, so the budget had
been spent on the load and the generation never got its turn -- one whole
battery re-run.  Raising the number would hide a real hang for even longer, so
the two phases now have independent, measured liveness rules:

  * LOAD phase -- process start until the first token byte on stdout (the CLI
    flushes every token: src/cli/pulsar_cli.cpp token_printer_write_text +
    fflush), covering open + prefill.  Liveness is PROGRESS, not a clock: the
    /proc/<pid>/io read_bytes counter is sampled every LOAD_STALL_WINDOW_S and
    the phase continues while at least MIN_LOAD_PROGRESS_BYTES were really
    fetched from storage.  MEASURED on sparky (2026-09-12, GB10): after
    `sync; echo 3 > /proc/sys/vm/drop_caches`, a cold load runs at ~1.7 GB/s
    and first-token lands at 29.4 s (total 30.7 s), i.e. ~300 MB per 180 s
    window -- 9x the MIN threshold.  The observed flake rate (92.8 GB while
    still not generating at 600 s) is ~155 MB/s, ~30 MB/s per window, still
    far above the threshold; a box that has genuinely stopped (0 bytes/window,
    which is what a hang looks like) fails after one window.  LOAD_CEILING_S
    is a backstop for a perpetual crawl and for kernels where /proc io is
    unreadable.
  * GENERATION phase -- once tokens flow, every byte resets a short stall
    watchdog (GEN_STALL_BUDGET_S).  A real decode hang stops emitting and fails
    in seconds, long before any load budget could hide it.

Both rules fail closed: a process that stops making progress in either phase
fails, and the diagnostic names which phase stalled.

usage: chat_smoke_gate.py MODEL [PULSAR_BIN]
"""
import os, re, subprocess, sys, tempfile, threading, time

model = sys.argv[1]
binary = sys.argv[2] if len(sys.argv) > 2 else "./pulsar"

# Measured, not guessed (see the header): a real cold load of this artifact on
# sparky is ~30 s at ~1.7 GB/s.  A load is allowed to be arbitrarily slow as
# long as it keeps fetching bytes; these thresholds only catch a stall.
LOAD_STALL_WINDOW_S = 180.0
MIN_LOAD_PROGRESS_BYTES = 32 * 1024 * 1024   # ~0.18 MB/s sustained floor
LOAD_CEILING_S = 1800.0                      # backstop for a bounded run
# Normal decode emits a token in well under a second (spec batches included);
# no token at all for this long is a hang, not a slow box.
GEN_STALL_BUDGET_S = 120.0

out = bytearray()
first_output_at = threading.Event()
last_activity = [time.monotonic()]
reader_done = threading.Event()
errfile = tempfile.TemporaryFile()

proc = subprocess.Popen(
    [binary, "-m", model, "-p", "What is 7+5? Answer with just the number."],
    stdout=subprocess.PIPE, stderr=errfile)


def read_bytes():
    """Bytes really fetched from the storage layer, mmap page-ins included --
    the load's liveness.  None when the kernel does not expose it."""
    try:
        with open("/proc/%d/io" % proc.pid) as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split(":")[1])
    except OSError:
        return None
    return None


def drain(fd, sink):
    """Read until EOF.  os.read returns as soon as bytes are available -- unlike
    a buffered .read(n), it cannot sit through a stall holding a token."""
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        sink += chunk
        last_activity[0] = time.monotonic()
        first_output_at.set()
    reader_done.set()


threading.Thread(target=drain, args=(proc.stdout.fileno(), out), daemon=True).start()

start = time.monotonic()
last_activity[0] = start
window_start = start
window_bytes = read_bytes()
failure = None
while True:
    if proc.poll() is not None:
        break
    now = time.monotonic()
    if first_output_at.is_set():
        if now - last_activity[0] > GEN_STALL_BUDGET_S:
            failure = ("generation stalled: no token for %.0f s after output began "
                       "(genuine decode hang, not a load)" % (now - last_activity[0]))
            break
    else:
        if now - start > LOAD_CEILING_S:
            failure = ("load exceeded the %.0f s ceiling (read %d bytes of stdout)"
                       % (LOAD_CEILING_S, len(out)))
            break
        if now - window_start >= LOAD_STALL_WINDOW_S:
            rb = read_bytes()
            if rb is not None and window_bytes is not None:
                if rb - window_bytes < MIN_LOAD_PROGRESS_BYTES:
                    failure = ("load made no storage progress for %.0f s (%d bytes fetched, "
                               "need %d) -- the child is stuck, not loading"
                               % (now - window_start, rb - window_bytes, MIN_LOAD_PROGRESS_BYTES))
                    break
                window_bytes = rb
            window_start = now
    time.sleep(0.25)

if failure:
    proc.kill()
    proc.wait()
    print("CHAT SMOKE GATE: FAIL (%s)" % failure)
    errfile.seek(0)
    tail_err = errfile.read()[-2000:].decode("utf-8", "replace")
    if tail_err.strip():
        print("  child stderr tail:", repr(tail_err[-500:]))
    sys.exit(1)

# The child exited; let the reader drain whatever is left in the pipe.
reader_done.wait(timeout=5.0)
errfile.close()

tail = out.decode("utf-8", "replace")[-4000:]
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

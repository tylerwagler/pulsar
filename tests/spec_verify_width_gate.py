#!/usr/bin/env python3
"""Spec-verify width gate: spec decoding must be OUTPUT-IDENTICAL to greedy at
EVERY verify width, not just the narrow ones the other gates happen to exercise.

Why this exists.  On 2026-08-14 three commits carrying a production-reachable
silent-wrong-logits bug passed all fourteen gates.  The activation cache is
keyed on (ptr, n_tok, in_dim); a layer encode armed it on `batch_ffn_norm` and
never disarmed, and the batch output head reuses that very buffer as its
output_norm scratch at the same n_tok and in_dim -- so the vocab GEMM hit a
stale entry and computed every verify row's logits from the last layer's FFN
activations.

Nothing caught it, for a reason worth stating: the prefill gate's frontier head
is a single-row head with a dedicated buffer that is never keyed, and the spec
gates run verify widths <= 4, where the mxfp8 dispatch takes a custom-NT kernel
that reads f32 directly and never consults the cache.  The cache is only
consulted at n_tok 5..8 -- the same 5..8-remainder shape hole that
pulsar_cuda_matmul.cu:1286-1294 already documents in a comment.

So this gate sweeps the depths the others skip.  Greedy (temp 0) decoding is
deterministic, and speculative decoding is an exact-output optimisation: it may
change SPEED but never TEXT.

*** THIS IS A DIAGNOSTIC, NOT A GATE.  Do not add it to GATE_TARGETS. ***

Byte-equality against greedy is NOT an invariant of this engine, and the first
two versions of this script assumed it was.  Measured on clean dev (cf33e3e),
96 and 160 tokens, two prompts:

  original prompt, n=96   -> diverging depths {4,5}
  original prompt, n=160  -> diverging depths {4,5}
  different prompt, n=96  -> diverging depths {1,2,3,4,5,6,7,8}   <-- ALL

So the set is stable across generation LENGTH but changes with the PROMPT, and
under some prompts EVERY depth diverges.  A fixed --known-diverging list is
therefore meaningless and would fail on main.  Not even depth 1 is structurally
safe: spec verify goes through gpu_graph_encode_output_head_batch while non-spec
decode uses the single-row gpu_graph_encode_output_head with its own buffer, so
no depth runs the same kernels as sequential decode.

Why divergence exists at all: the batched verify GEMMs and the single-row decode
GEMMs are different kernels, and above the mxfp8 gemv cap they are different
ARITHMETIC (see L041).  A small logit difference only changes text when it flips
an argmax, which is why it looks like luck -- it is luck, conditioned on whether
a near-tie occurs in that particular generation.

What this IS good for: A/B against a KNOWN-GOOD BINARY at a FIXED prompt.  That
is what proved the L035 act-cache bug -- dev {4,5}, pre-fix {4,5,6,7,8}, post-fix
{4,5}, same prompt throughout, so the delta was unambiguous.

To make it a real gate it needs a recorded per-depth baseline blob from a
reference commit, compared byte-for-byte, on the cuda-prefill-gate pattern.
"""
import argparse
import subprocess
import sys


def run(binary, model, prompt, ntok, extra):
    cmd = [binary, "-m", model, "-p", prompt, "--temp", "0", "-n", str(ntok)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("  harness error: %s exited %d\n%s" % (binary, p.returncode, p.stderr[-2000:]))
        return None
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="./pulsar")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--known-diverging", default="4,5",
                    help="draft depths already divergent on dev via the documented "
                         "NT-vs-cuBLASLt dispatch crossing; not a regression")
    ap.add_argument("--prompt",
                    default="Explain in detail how a speculative decoding "
                            "verify step works, step by step.")
    a = ap.parse_args()

    widths = [int(w) for w in a.widths.split(",") if w.strip()]
    known = {int(w) for w in a.known_diverging.split(",") if w.strip()}
    print("spec verify width gate: depths=%s, %d tokens, greedy (known-diverging: %s)"
          % (widths, a.tokens, sorted(known) or "none"))

    ref = run(a.binary, a.model, a.prompt, a.tokens, ["--no-dspark"])
    if ref is None:
        print("SPEC WIDTH GATE: HARNESS ERROR (reference run failed)")
        return 2
    print("  reference (--no-dspark): %d bytes" % len(ref))

    bad = []
    for w in widths:
        out = run(a.binary, a.model, a.prompt, a.tokens, ["--dspark-draft", str(w)])
        if out is None:
            bad.append((w, "run failed"))
            continue
        if out == ref:
            print("  depth %d: identical to greedy OK%s"
                  % (w, "  (known-diverging depth now matches)" if w in known else ""))
        else:
            # First divergent character makes the failure actionable.
            n = min(len(out), len(ref))
            at = next((i for i in range(n) if out[i] != ref[i]), n)
            if w in known:
                print("  depth %d: differs at byte %d -- KNOWN (dispatch crossing), not a regression" % (w, at))
            else:
                print("  depth %d: DIFFERS from greedy at byte %d (%d vs %d bytes) FAIL"
                      % (w, at, len(out), len(ref)))
                bad.append((w, "diverged at byte %d" % at))

    if bad:
        print("SPEC WIDTH GATE: FAIL (%s)"
              % ", ".join("w=%d %s" % (w, why) for w, why in bad))
        return 1
    print("SPEC WIDTH GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

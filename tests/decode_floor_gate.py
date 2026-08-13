#!/usr/bin/env python3
"""Five-workload decode floor gate -- min, not mean.

A mean across workloads hides the case that actually hurts: one prompt shape
that decodes badly.  Averaging it against four healthy ones yields a number
that still looks fine while the regression ships.  This gate scores the
WORST workload and compares that against a recorded floor.

The floor comes from OUR OWN baseline, captured with --write-baseline on a
known-good build.  It is deliberately not somebody else's published number:
a foreign figure measured on a different harness, context depth and sampler
is not a floor, it is a rumour.

Five workloads at 512 tokens, concurrency 1.  The languages are not
decoration -- token mix shifts the decode profile (dense punctuation and
short identifiers tokenize differently from prose), so a single prompt shape
would leave most of the surface unmeasured.

    # on a known-good build, once:
    ./tests/decode_floor_gate.py --port 8080 --write-baseline

    # thereafter, per release:
    ./tests/decode_floor_gate.py --port 8080 --json-out run.json

Exit status: 0 pass, 1 regression below floor, 2 harness/transport error.

Run the server with --no-kv-disk. A warm checkpoint skips prefill, which
makes wall-clock t/s incomparable between runs; the gate records the state
you declare and refuses to write a baseline from a run declared 'enabled'.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sse_decode_bench import stream_once, summarize  # noqa: E402

DEFAULT_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test-vectors", "decode_floor_baseline.json"
)

# Kept deliberately terse and self-contained: the prompt must not depend on
# repo state, or the gate stops being reproducible across checkouts.
WORKLOADS = [
    ("python", "Write a Python module that implements an LRU cache with a TTL per entry. "
               "Include type hints and docstrings. Output code only."),
    ("rust", "Write a Rust module implementing a lock-free single-producer single-consumer "
             "ring buffer using atomics. Include the unsafe justification comments. Output code only."),
    ("typescript", "Write a TypeScript module implementing a typed event emitter with "
                   "wildcard subscriptions and once() semantics. Output code only."),
    ("cuda", "Write a CUDA kernel that performs a warp-level segmented reduction over a "
             "ragged array, plus its launch wrapper. Output code only."),
    ("go", "Write a Go package implementing a bounded worker pool with context cancellation "
           "and graceful drain. Output code only."),
]


def run_workloads(args):
    results = []
    for name, prompt in WORKLOADS:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        raw = stream_once(args.host, args.port, payload, args.timeout)
        summary = summarize(raw)
        summary["workload"] = name
        results.append(summary)
        print(
            "  %-11s sse=%-8s wall=%-8s ttft=%-9s tokens=%s"
            % (
                name,
                summary["sse_decode_tps"],
                summary["wall_tps"],
                summary["ttft_ms"],
                summary["completion_tokens"],
            ),
            file=sys.stderr,
        )
    return results


def score(results):
    """The gate value is the worst workload, and we name which one it was."""
    usable = [r for r in results if r["sse_decode_tps"] is not None]
    if not usable:
        return None, None
    worst = min(usable, key=lambda r: r["sse_decode_tps"])
    return worst["sse_decode_tps"], worst["workload"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="ds4flash")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--write-baseline", action="store_true",
                   help="record this run as the floor instead of gating against it")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="fractional slack below the floor before failing (default 0.10)")
    p.add_argument("--kv-disk-state", default="disabled",
                   choices=["unknown", "disabled", "enabled"])
    p.add_argument("--json-out", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    if args.write_baseline and args.kv_disk_state != "disabled":
        print("refusing to write a baseline from a run with kv-disk state '%s': "
              "a warm checkpoint skips prefill and the floor would be unreachable "
              "on a cold run" % args.kv_disk_state, file=sys.stderr)
        return 2

    print("decode floor gate: %d workloads, %d tokens, C1, kv-disk=%s"
          % (len(WORKLOADS), args.max_tokens, args.kv_disk_state), file=sys.stderr)
    try:
        results = run_workloads(args)
    except (OSError, RuntimeError) as e:
        print("harness error: %s" % e, file=sys.stderr)
        return 2

    worst_tps, worst_name = score(results)
    record = {
        "schema": "pulsar.decode_floor_gate.v1",
        "host": "%s:%d" % (args.host, args.port),
        "model": args.model,
        "label": args.label,
        "max_tokens": args.max_tokens,
        "concurrency": 1,
        "kv_disk_state": args.kv_disk_state,
        "results": results,
        "min_sse_decode_tps": worst_tps,
        "min_workload": worst_name,
    }

    if worst_tps is None:
        print("FAIL: no workload produced a usable decode rate", file=sys.stderr)
        record["verdict"] = "no-data"
        emit(record, args.json_out)
        return 2

    if args.write_baseline:
        os.makedirs(os.path.dirname(args.baseline), exist_ok=True)
        with open(args.baseline, "w") as f:
            json.dump(
                {
                    "schema": "pulsar.decode_floor_baseline.v1",
                    "min_sse_decode_tps": worst_tps,
                    "min_workload": worst_name,
                    "max_tokens": args.max_tokens,
                    "model": args.model,
                    "kv_disk_state": args.kv_disk_state,
                    "per_workload": {r["workload"]: r["sse_decode_tps"] for r in results},
                },
                f,
                indent=2,
            )
            f.write("\n")
        print("baseline written: %s (floor %.3f t/s, worst=%s)"
              % (args.baseline, worst_tps, worst_name), file=sys.stderr)
        record["verdict"] = "baseline-written"
        emit(record, args.json_out)
        return 0

    try:
        with open(args.baseline) as f:
            base = json.load(f)
    except (OSError, ValueError) as e:
        print("no usable baseline at %s (%s) -- run once with --write-baseline "
              "on a known-good build" % (args.baseline, e), file=sys.stderr)
        record["verdict"] = "no-baseline"
        emit(record, args.json_out)
        return 2

    floor = base.get("min_sse_decode_tps")
    if not isinstance(floor, (int, float)) or floor <= 0:
        print("baseline %s has no usable min_sse_decode_tps" % args.baseline, file=sys.stderr)
        record["verdict"] = "bad-baseline"
        emit(record, args.json_out)
        return 2

    if base.get("max_tokens") != args.max_tokens:
        print("baseline was captured at max_tokens=%s but this run used %s; rates are "
              "not comparable across generation lengths"
              % (base.get("max_tokens"), args.max_tokens), file=sys.stderr)
        record["verdict"] = "baseline-mismatch"
        emit(record, args.json_out)
        return 2

    threshold = floor * (1.0 - args.tolerance)
    record["baseline_min_sse_decode_tps"] = floor
    record["threshold_tps"] = round(threshold, 3)
    record["ratio_to_baseline"] = round(worst_tps / floor, 4)

    if worst_tps < threshold:
        record["verdict"] = "fail"
        emit(record, args.json_out)
        print("FAIL: worst workload '%s' at %.3f t/s is below the floor %.3f t/s "
              "(baseline %.3f, tolerance %.0f%%)"
              % (worst_name, worst_tps, threshold, floor, args.tolerance * 100),
              file=sys.stderr)
        return 1

    record["verdict"] = "pass"
    emit(record, args.json_out)
    print("PASS: worst workload '%s' at %.3f t/s, floor %.3f t/s (%.1f%% of baseline)"
          % (worst_name, worst_tps, threshold, 100.0 * worst_tps / floor), file=sys.stderr)
    return 0


def emit(record, path):
    text = json.dumps(record, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Five-workload decode floor gate: each workload against its own baseline, per served model (L367).

Five workloads at 512 tokens, concurrency 1.  The languages are not decoration -- token mix shifts the decode profile
(dense punctuation and short identifiers tokenize differently from prose; speculative decoding accepts differently),
so a single prompt shape would leave most of the surface unmeasured.

REPEATABLE.  Requests are greedy (temperature 0), so a build decodes the same text every time and speculative
acceptance -- which moves with the text -- does not move between runs; each workload is run --repeats times (default
3) and its median decode rate is the one graded.  (Before L367 the requests sampled: between v0.5.0 and v0.5.1, with no
decode-path change, Qwen's python workload measured 75.27 then 53.95 tok/s.)

PER MODEL, PER WORKLOAD.  The baseline file holds one entry per served model id (the server's /v1/models), recorded
on a release build with --write-baseline --release vX.Y.Z.  A run is graded workload by workload: each median against
that workload's baseline; the gate fails when any falls below (1 - tolerance) of it, and names the worst.  A regression
in one workload can no longer hide behind the slowest one.  A model with no baseline is not graded (exit 2), never
compared against another model's numbers.  Each entry carries its own tolerance -- one number, or one per workload --
set from that model's measured run-to-run spread (`tolerance_why` says how); --tolerance overrides it.

    # on the release build, once per model (the server serving that model, --no-kv-disk):
    ./tests/decode_floor_gate.py --port 8080 --write-baseline --release v0.5.1 --engine 52bc947e

    # thereafter:
    ./tests/decode_floor_gate.py --port 8080 --json-out run.json

    ./tests/decode_floor_gate.py --self-test     # the grading logic, no server

Exit status: 0 pass (or baseline written), 1 a workload below its floor, 2 harness / baseline error.

Run the server with --no-kv-disk.  A warm checkpoint skips prefill, which makes wall-clock t/s incomparable between
runs; the gate records the state you declare and refuses to write a baseline from a run declared 'enabled'.
"""

import argparse
import datetime
import http.client
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test-vectors", "decode_floor_baseline.json"
)
BASELINE_SCHEMA = "pulsar.decode_floor_baseline.v2"
DEFAULT_TOLERANCE = 0.05
TEMPERATURE = 0.0

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


def served_model(host, port, timeout):
    """The id the server serves (its first /v1/models entry)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError("HTTP %d from /v1/models" % resp.status)
    data = json.loads(body).get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError("/v1/models lists no model")
    return data[0]["id"]


def run_workloads(args, model):
    from sse_decode_bench import stream_once, summarize
    results = []
    for name, prompt in WORKLOADS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        runs = []
        for _ in range(args.repeats):
            summary = summarize(stream_once(args.host, args.port, payload, args.timeout))
            runs.append(summary)
        rates = [r["sse_decode_tps"] for r in runs if r["sse_decode_tps"] is not None]
        med = round(statistics.median(rates), 3) if rates else None
        results.append({"workload": name, "sse_decode_tps": med, "runs": runs,
                        "spread": round(max(rates) / min(rates) - 1.0, 4) if len(rates) > 1 else 0.0})
        print("  %-11s median sse=%-8s runs=%s tokens=%s"
              % (name, med, [r["sse_decode_tps"] for r in runs], [r["completion_tokens"] for r in runs]),
              file=sys.stderr)
    return results


def tol_of(tolerance, name):
    """A workload's tolerance: one number for every workload, or a {workload: number} map (DEFAULT_TOLERANCE where
    a workload has none)."""
    return tolerance.get(name, DEFAULT_TOLERANCE) if isinstance(tolerance, dict) else tolerance


def grade(per_workload, base_per_workload, tolerance):
    """Each workload's rate against its own baseline, at its own tolerance (tol_of).  Returns (verdict, rows, worst)
    where rows are (workload, rate, baseline, ratio) and worst is the row furthest below its floor (else the lowest
    ratio); verdict is 'pass', 'fail' or 'incomplete' (a baseline workload missing from the run, or without a rate)."""
    rows = []
    for name, floor in base_per_workload.items():
        rate = per_workload.get(name)
        if rate is None or not isinstance(floor, (int, float)) or floor <= 0:
            return "incomplete", rows, (name, rate, floor, None)
        rows.append((name, rate, floor, round(rate / floor, 4)))
    if not rows:
        return "incomplete", rows, None
    worst = min(rows, key=lambda r: r[3] - (1.0 - tol_of(tolerance, r[0])))
    return ("fail" if worst[3] < 1.0 - tol_of(tolerance, worst[0]) else "pass"), rows, worst


def load_baselines(path):
    with open(path) as f:
        base = json.load(f)
    if base.get("schema") != BASELINE_SCHEMA:
        raise ValueError("schema %r, want %r (per-model baselines, L367)" % (base.get("schema"), BASELINE_SCHEMA))
    return base


def write_baseline(path, model, entry, max_tokens, repeats):
    """Record (or replace) one model's entry; every other model's entry is kept."""
    try:
        base = load_baselines(path)
    except (OSError, ValueError):
        base = {"schema": BASELINE_SCHEMA, "max_tokens": max_tokens, "repeats": repeats,
                "temperature": TEMPERATURE, "models": {}}
    if base.get("max_tokens") != max_tokens or base.get("temperature") != TEMPERATURE:
        raise ValueError("the file's runs are max_tokens=%s temperature=%s; this run's are %s / %s"
                         % (base.get("max_tokens"), base.get("temperature"), max_tokens, TEMPERATURE))
    base["repeats"] = repeats
    base["models"][model] = entry
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(base, f, indent=2, sort_keys=False)
        f.write("\n")


def self_test():
    """The grading and the per-model file, no server."""
    import tempfile
    base = {"python": 75.0, "rust": 54.0, "go": 56.0}
    v, rows, worst = grade({"python": 74.0, "rust": 55.0, "go": 56.0}, base, 0.05)
    assert v == "pass" and worst[0] == "python", (v, worst)
    # a 20% drop in the FASTEST workload fails although the slowest is unchanged (the old min-only gate passed it)
    v, rows, worst = grade({"python": 60.0, "rust": 54.0, "go": 56.0}, base, 0.05)
    assert v == "fail" and worst[0] == "python" and worst[3] == 0.8, (v, worst)
    v, _, worst = grade({"python": 75.0, "rust": 54.0}, base, 0.05)
    assert v == "incomplete" and worst[0] == "go", (v, worst)
    # per-workload tolerances: python's 15% drop passes at its own 20%, rust's 6% drop fails at the default 5%
    v, _, worst = grade({"python": 64.0, "rust": 50.7, "go": 56.0}, base, {"python": 0.20})
    assert v == "fail" and worst[0] == "rust", (v, worst)
    v, _, worst = grade({"python": 64.0, "rust": 54.0, "go": 56.0}, base, {"python": 0.20})
    assert v == "pass", (v, worst)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.json")
        write_baseline(p, "qwen", {"per_workload": {"python": 75.0}}, 512, 3)
        write_baseline(p, "deepseek", {"per_workload": {"python": 30.0}}, 512, 3)
        write_baseline(p, "qwen", {"per_workload": {"python": 76.0}}, 512, 3)
        b = load_baselines(p)
        assert set(b["models"]) == {"qwen", "deepseek"} and b["models"]["qwen"]["per_workload"]["python"] == 76.0
        try:
            write_baseline(p, "qwen", {}, 256, 3)
            raise AssertionError("a different max_tokens was accepted")
        except ValueError:
            pass
    # the shipped file is a v2 file whose entries name their release and cover every workload
    b = load_baselines(DEFAULT_BASELINE)
    for m, e in b["models"].items():
        assert e.get("release") and set(e["per_workload"]) == {w for w, _ in WORKLOADS}, m
        t = e.get("tolerance", 0)
        assert e.get("tolerance_why") and all(0 < x < 0.5 for x in (t.values() if isinstance(t, dict) else [t])), m
    print("decode_floor_gate self-test: PASS (%d models in the shipped baselines)" % len(b["models"]))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default=None, help="the model id to request (default: the server's /v1/models)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3, help="runs per workload; the median is graded (default 3)")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--write-baseline", action="store_true",
                   help="record this run as the model's baseline instead of grading it")
    p.add_argument("--release", default="", help="the release the baseline is recorded on (required to write)")
    p.add_argument("--engine", default="", help="the engine commit of the build (recorded with a baseline)")
    p.add_argument("--server-args", default="", help="the server's flags (recorded with a baseline)")
    p.add_argument("--tolerance", type=float, default=None,
                   help="fractional slack below each workload's baseline before failing (default: the model's "
                        "recorded tolerance, else %.2f)" % DEFAULT_TOLERANCE)
    p.add_argument("--kv-disk-state", default="disabled", choices=["unknown", "disabled", "enabled"])
    p.add_argument("--json-out", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--self-test", action="store_true", help="check the grading logic and the shipped file; no server")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if args.write_baseline and args.kv_disk_state != "disabled":
        print("refusing to write a baseline from a run with kv-disk state '%s': a warm checkpoint skips prefill and "
              "the floor would be unreachable on a cold run" % args.kv_disk_state, file=sys.stderr)
        return 2
    if args.write_baseline and not args.release:
        print("--write-baseline needs --release (the release the floor belongs to)", file=sys.stderr)
        return 2

    try:
        model = args.model or served_model(args.host, args.port, args.timeout)
    except (OSError, RuntimeError, ValueError) as e:
        print("harness error: %s" % e, file=sys.stderr)
        return 2
    print("decode floor gate: %s, %d workloads x %d (median), %d tokens, greedy, C1, kv-disk=%s"
          % (model, len(WORKLOADS), args.repeats, args.max_tokens, args.kv_disk_state), file=sys.stderr)
    try:
        results = run_workloads(args, model)
    except (OSError, RuntimeError) as e:
        print("harness error: %s" % e, file=sys.stderr)
        return 2

    per_workload = {r["workload"]: r["sse_decode_tps"] for r in results}
    usable = {k: v for k, v in per_workload.items() if v is not None}
    record = {
        "schema": "pulsar.decode_floor_gate.v2",
        "host": "%s:%d" % (args.host, args.port),
        "model": model,
        "label": args.label,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "temperature": TEMPERATURE,
        "concurrency": 1,
        "kv_disk_state": args.kv_disk_state,
        "results": results,
        "per_workload": per_workload,
        "min_sse_decode_tps": min(usable.values()) if usable else None,
        "min_workload": min(usable, key=usable.get) if usable else None,
    }
    if len(usable) != len(WORKLOADS):
        print("FAIL: a workload produced no usable decode rate", file=sys.stderr)
        record["verdict"] = "no-data"
        emit(record, args.json_out)
        return 2

    if args.write_baseline:
        entry = {"release": args.release, "engine": args.engine, "server_args": args.server_args,
                 "recorded": datetime.date.today().isoformat(), "per_workload": per_workload,
                 "tolerance": args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE,
                 "spread": {r["workload"]: r["spread"] for r in results}}
        try:
            write_baseline(args.baseline, model, entry, args.max_tokens, args.repeats)
        except ValueError as e:
            print("baseline not written: %s" % e, file=sys.stderr)
            return 2
        print("baseline written for %s (%s): %s" % (model, args.release, per_workload), file=sys.stderr)
        record["verdict"] = "baseline-written"
        emit(record, args.json_out)
        return 0

    try:
        base = load_baselines(args.baseline)
    except (OSError, ValueError) as e:
        print("no usable baseline file at %s (%s)" % (args.baseline, e), file=sys.stderr)
        record["verdict"] = "no-baseline"
        emit(record, args.json_out)
        return 2
    entry = base["models"].get(model)
    if not entry:
        print("no baseline for model %r (have: %s) -- record one on a release build with --write-baseline --release"
              % (model, ", ".join(sorted(base["models"])) or "none"), file=sys.stderr)
        record["verdict"] = "no-baseline"
        emit(record, args.json_out)
        return 2
    if base.get("max_tokens") != args.max_tokens:
        print("the baselines were recorded at max_tokens=%s, this run used %s" % (base.get("max_tokens"),
              args.max_tokens), file=sys.stderr)
        record["verdict"] = "baseline-mismatch"
        emit(record, args.json_out)
        return 2

    tol = args.tolerance if args.tolerance is not None else entry.get("tolerance", DEFAULT_TOLERANCE)
    verdict, rows, worst = grade(per_workload, entry["per_workload"], tol)
    record["baseline"] = {"release": entry.get("release"), "engine": entry.get("engine"),
                          "per_workload": entry["per_workload"]}
    record["tolerance"] = tol
    record["ratios"] = {name: ratio for name, _, _, ratio in rows}
    record["worst_workload"] = worst[0] if worst else None
    record["worst_ratio"] = worst[3] if worst else None
    record["verdict"] = verdict
    emit(record, args.json_out)
    for name, rate, floor, ratio in rows:
        print("  %-11s %8.3f t/s  baseline %8.3f  %6.1f%%%s" % (name, rate, floor, 100 * ratio,
              "  <-- below" if ratio < 1.0 - tol_of(tol, name) else ""), file=sys.stderr)
    if verdict == "incomplete":
        print("FAIL: workload %r has no rate or no baseline" % (worst[0] if worst else None), file=sys.stderr)
        return 2
    if verdict == "fail":
        print("FAIL: %s's '%s' at %.1f%% of its %s baseline (tolerance %.0f%%)"
              % (model, worst[0], 100 * worst[3], entry.get("release"), 100 * tol_of(tol, worst[0])), file=sys.stderr)
        return 1
    print("PASS: %s, every workload within its tolerance of its %s baseline (closest: '%s' %.1f%%, tolerance %.0f%%)"
          % (model, entry.get("release"), worst[0], 100 * worst[3], 100 * tol_of(tol, worst[0])), file=sys.stderr)
    return 0


def emit(record, path):
    text = json.dumps(record, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())

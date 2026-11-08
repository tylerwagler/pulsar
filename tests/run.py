#!/usr/bin/env python3
"""run.py -- runs a test tier from tests/manifest.toml and says exactly what ran (L357).

    tests/run.py check|kernels|release [--only a,b] [--strict] [--dry-run]
    tests/run.py --list [TIER]
    tests/run.py --verify        every manifest test is named once and is a make target
    tests/run.py --names         every test's name (the Makefile declares them phony from it)

A test is a make target; a target's exit code is its verdict.  A log line starting SKIP means part of it did not run:
the summary names each one, and --strict fails it.  N/A is not a skip: it names a leg the loaded family cannot have
(a capability it lacks, e.g. step graphs on Qwen), so there is nothing to run.  Each run leaves its logs and results.json in .build/tests/<tier>-<sha>/,
and exits 0 only when every test passed."""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = ("check", "kernels", "release")
GPU_TIERS = ("kernels", "release")


def manifest():
    with open(os.path.join(ROOT, "tests", "manifest.toml"), "rb") as f:
        m = tomllib.load(f)
    for t in m["test"]:
        if t["tier"] not in TIERS:
            sys.exit("manifest: %s has an unknown tier %r" % (t["name"], t["tier"]))
    return m


def verify(m):
    """The manifest is the one list (L357): every test is named once and is a make target (a name with no rule would
    fail only when its tier runs)."""
    text = open(os.path.join(ROOT, "Makefile"), encoding="utf-8", errors="replace").read()
    rules = set(re.findall(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", text, re.M))
    names = [t["name"] for t in m["test"]]
    bad = ["%s is listed %d times" % (n, names.count(n)) for n in sorted(set(names)) if names.count(n) > 1]
    bad += ["%s has no make rule" % n for n in names if n not in rules]
    for b in bad:
        print("MANIFEST FAIL: " + b)
    if not bad:
        print("manifest: %d tests, each a make target" % len(names))
    return not bad


def make_cmd(target):
    cmd = ["make", "--no-print-directory", "-j%s" % os.environ.get("GATE_JOBS", str(os.cpu_count() or 8)), target, "CUDA_ARCH=sm_120f"]
    for v in ("FRONTIER_MODEL", "SPEC_GATE_MODEL", "CUTLASS_DIR"):
        if os.environ.get(v):
            cmd.append("%s=%s" % (v, os.environ[v]))
    return cmd


def git_sha():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "-uno"], capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() or "unknown", bool(dirty)


class Run:
    def __init__(self, out, env, stream_lock):
        self.out, self.env, self.lock = out, env, stream_lock
        self.results = {}

    def one(self, t, stream):
        name = t["name"]
        log = os.path.join(self.out, name + ".log")
        t0 = time.time()
        if stream:
            print("\n=== %s ===" % name, flush=True)
        with open(log, "wb") as f:
            p = subprocess.Popen(make_cmd(name), cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout:
                f.write(line)
                if stream:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
            rc = p.wait()
        text = open(log, encoding="utf-8", errors="replace").read()
        skips = sorted({x.strip()[:120] for x in text.splitlines() if re.match(r"\s*SKIP(PED)?\b", x)})
        self.results[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "skips": skips, "budget": t.get("budget")}


def summary(m, run, tier, total, strict):
    sections = {}
    sec = run.env.get("PULSAR_GATES_SECTIONS")
    if sec and os.path.exists(sec):
        for line in open(sec):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                sections[parts[1]] = int(parts[0])
    print("\n" + "=" * 20 + " %s " % tier.upper() + "=" * 20)
    print("  %-9s %7s %7s  %s" % ("", "secs", "budget", "test"))
    failed, flagged, skipped = [], [], []
    for name, r in run.results.items():
        verdict = "PASS" if r["rc"] == 0 else "FAIL"
        if r["skips"]:
            skipped.append(name)
            if strict and r["rc"] == 0:
                verdict = "FAIL"
        if verdict == "FAIL":
            failed.append(name)
        over = r["budget"] is not None and r["seconds"] > r["budget"]
        if over:
            flagged.append(name)
        print("  %-9s %7.0f %7s  %s%s" % (verdict, r["seconds"], r["budget"] if r["budget"] is not None else "-", name, "  (over budget)" if over else ""))
    for name, s in sections.items():
        b = m.get("sections", {}).get(name)
        if b is not None and s > b:
            flagged.append(name)
        print("  %-9s %7d %7s  %s%s" % ("", s, b if b is not None else "-", name, "  (over budget)" if (b is not None and s > b) else ""))
    if skipped:
        print("\nNOT FULLY RUN:")
        for name in skipped:
            for s in run.results[name]["skips"]:
                print("  %s: %s" % (name, s))
    limit = m.get("limits", {}).get(tier)
    print("\n  total %.0f s%s" % (total, " of %d" % limit if limit else ""))
    if limit and total > limit:
        flagged.append("total")
    if flagged:
        print("  over budget, not a failure: " + ", ".join(flagged))
    print("\n%s" % ("%s PASS" % tier.upper() if not failed else "%s FAILED: %s" % (tier.upper(), " ".join(failed))))
    return failed, sections


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", nargs="?", choices=TIERS)
    ap.add_argument("--only", help="comma-separated test names, from any tier")
    ap.add_argument("--strict", action="store_true", help="a SKIP line fails its test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--names", action="store_true", help="every test's name, one line (the Makefile's .PHONY)")
    a = ap.parse_args()
    m = manifest()
    if a.names:
        print("\n".join(t["name"] for t in m["test"]))
        return 0
    if a.verify:
        return 0 if verify(m) else 1
    tests = m["test"]
    if a.list:
        for t in tests:
            if not a.tier or t["tier"] == a.tier:
                print("%-8s %-34s %s" % (t["tier"], t["name"], t.get("budget", "-")))
        return 0
    if not a.tier:
        ap.error("name a tier")
    if a.tier != "release":
        tests = [t for t in tests if t["tier"] == a.tier]
    if a.only:
        want = a.only.split(",")
        unknown = [w for w in want if w not in {t["name"] for t in m["test"]}]
        if unknown:
            sys.exit("run.py: not in the manifest: %s" % ", ".join(unknown))
        tests = [t for t in m["test"] if t["name"] in want]
    if not tests:
        sys.exit("run.py: nothing selected")
    if a.dry_run:
        for t in tests:
            print(" ".join(make_cmd(t["name"])))
        return 0
    if not verify(m):
        return 1
    sha, dirty = git_sha()
    out = os.path.join(ROOT, ".build", "tests", "%s-%s" % (a.tier, sha))
    os.makedirs(out, exist_ok=True)
    print("%s at %s%s, %d tests, logs in %s" % (a.tier, sha, " (DIRTY TREE)" if dirty else "", len(tests), out), flush=True)
    env = dict(os.environ)
    sections = os.path.join(out, "sections.tsv")
    open(sections, "w").close()
    env["PULSAR_GATES_SECTIONS"] = sections
    if a.tier in GPU_TIERS and subprocess.run(make_cmd("gates-preflight"), cwd=ROOT, env=env).returncode != 0:
        return 1
    if a.tier == "release" and not a.only and subprocess.run(make_cmd("test-bins"), cwd=ROOT, env=env).returncode != 0:
        print("run.py: the test binaries did not build")
        return 1
    run = Run(out, env, threading.Lock())
    t0 = time.time()
    background = []
    overlap_now = a.tier == "release" and not a.only
    late = [t for t in tests if t.get("overlap")] if overlap_now else []
    for t in tests:
        if overlap_now and t.get("overlap"):
            continue
        if overlap_now and t["name"] == "cuda-runner-gate":
            for h in late:
                th = threading.Thread(target=run.one, args=(h, False))
                th.start()
                background.append(th)
        run.one(t, True)
    for th in background:
        th.join()
    total = time.time() - t0
    failed, secs = summary(m, run, a.tier, total, a.strict)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"tier": a.tier, "sha": sha, "dirty": dirty, "total_s": round(total), "tests": run.results, "sections": secs, "failed": failed}, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""battery_budget.py -- L317: `make gates` prints each section's wall time against its budget, and flags (never fails)
a section over it, so a row that adds battery time says so.

    python3 tests/battery_budget.py --sections FILE --host FILE --total SECONDS --budget tests/manifest.toml

The budgets live in ONE place, tests/manifest.toml (L357): each test's `budget`, the per-model `[sections]`, the
`[limits]` release total, and for the host-only section the longest budget of the tests that overlap.  The measured sections come from make
gates: one `SECONDS<TAB>NAME` line per serial target, the runner's per-model lines (cuda-runner-gate [<family>]) and
the live legs' (live-legs [<model>]); the host-only gates overlap the runner, so they are one section, their longest
("host-only gates (longest)").  A measured section with no budget row is flagged (add its row); a budget row nothing
measured is listed as not run.  Exit 0 always: the report is a flag, the battery's verdict is the gates'."""
import argparse, sys, tomllib

HOST = "host-only gates (longest)"
TOTAL = "suite total"


def read_budget(path):
    with open(path, "rb") as f:
        m = tomllib.load(f)
    rows = [(t["name"], int(t["budget"])) for t in m.get("test", []) if "budget" in t and not t.get("overlap")]
    rows += [(k, int(v)) for k, v in m.get("sections", {}).items()]
    host = [int(t["budget"]) for t in m.get("test", []) if t.get("overlap") and "budget" in t]
    if host:
        rows.append((HOST, max(host)))
    if "release" in m.get("limits", {}):
        rows.append((TOTAL, int(m["limits"]["release"])))
    return rows


def read_tsv(path):
    out = []
    try:
        for line in open(path, encoding="utf-8"):
            secs, _, name = line.rstrip("\n").partition("\t")
            if name:
                out.append((name, int(float(secs))))
    except FileNotFoundError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--host", default="")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--budget", required=True)
    a = ap.parse_args()
    budget = read_budget(a.budget)
    measured = read_tsv(a.sections)
    host = read_tsv(a.host) if a.host else []
    if host:
        measured.append((HOST, max(s for _, s in host)))
    measured.append((TOTAL, a.total))
    want = dict(budget)
    got = dict(measured)
    print("\n  battery wall time against its budget (tests/manifest.toml):")
    print("    %6s  %6s  %s" % ("secs", "budget", "section"))
    over = []
    for name, secs in sorted(measured, key=lambda r: (r[0] == TOTAL, -r[1])):
        b = want.get(name)
        flag = ""
        if b is None:
            flag = "  <-- NO BUDGET: add one to tests/manifest.toml"
            over.append(name)
        elif secs > b:
            flag = "  <-- OVER BUDGET by %d s" % (secs - b)
            over.append(name)
        print("    %6d  %6s  %s%s" % (secs, b if b is not None else "-", name, flag))
    for name, b in budget:
        if name not in got:
            print("    %6s  %6d  %s  (not run)" % ("-", b, name))
    if host:
        print("\n  host-only gates (overlapping the runner), slowest first:")
        for name, secs in sorted(host, key=lambda r: -r[1]):
            print("    %6d  %s" % (secs, name))
    print("\n  suite total: %d s" % a.total)
    if over:
        print("  BUDGET FLAGGED (not a failure): %s -- a row that adds battery time says so in its ledger row, and "
              "moves the manifest" % ", ".join(over))
    else:
        print("  every section within its budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())

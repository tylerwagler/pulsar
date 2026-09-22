#!/usr/bin/env python3
"""merge_parent_costs.py — combine per-parent expert_empirical_cost runs into
one allocator cost table, and VERIFY the combination.

WHY A MERGE AND NOT TWO CHAINED RUNS.  `expert_empirical_cost --merge-base`
REPLACES every expert row it finds in the base
(`merge_cost_payloads(..., replace_experts=True)` pops the overlap, then
updates).  Chaining is therefore asymmetric and silent: merge a gate_up run
first, then merge a down run on top of it, and the second merge overwrites the
measured gate/up rows with its own `unmeasured_parent_this_run` markers.  The
two runs have to be combined by explicit overlay instead -- which is what this
does -- and the result is checked, not trusted:

  * a row is taken from a run only if that run MEASURED it
    (`cost_source == 'empirical_unit_kl'`); marker rows are never copied;
  * a name measured by BOTH runs is an error -- the pair is complementary by
    construction, so an overlap means the pair is mis-specified;
  * every expert name either run covers must be overlaid, and the overlay is
    reported per role;
  * each run's per-layer, per-role `predicted_dloss` sums are compared against
    that run's own `provenance.unit_kls`.  That is the measurement re-assembled
    out of the per-member rows, so it is the one check that proves the
    n_params split booked the unit KL exactly once.

The per-parent patch (fork `dsv4-vexp` @ fb141ebd) quantizes only the selected
packed parent, so this is what turns "the split is proportional to n_params"
into "the split is measured".

usage:
  merge_parent_costs.py --base cost.pkl --gu gu.pkl --down down.pkl \
      --out cost-parents.pkl [--report parents.json] [--allow-partial]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

MEASURED = "empirical_unit_kl"
MARKER = "unmeasured_parent_this_run"

# Promotion budget deltas, in GiB, per (layer, role).  One projection copy of
# one layer is 256 experts x (4096x2048 + 2048x4096)/2 ... i.e. the measured
# iq2_xxs -> mxfp4 byte delta: 0.58720256 GB = 0.546875 GiB exactly
# (L216, "whole layer 1.641 GiB vs gu 1.094 vs down 0.547").
PROJ_GIB = 0.58720256e9 / 2**30
ROLE_GIB = {"gate_up": 2 * PROJ_GIB, "down": PROJ_GIB}

ROLE_OF = {"gate_proj": "gate_up", "up_proj": "gate_up", "down_proj": "down"}
EXPERT_RE = re.compile(
    r"^(?P<layer>.+?)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    with path.open("rb") as fh:
        return pickle.load(fh)


def measured_rows(payload: dict) -> dict[str, dict]:
    """Rows this run actually measured (any format measured counts)."""
    out: dict[str, dict] = {}
    for name, row in (payload.get("costs") or {}).items():
        if isinstance(row, dict) and any(
            isinstance(r, dict) and r.get("cost_source") == MEASURED
            for r in row.values()
        ):
            out[name] = row
    return out


def covered_rows(payload: dict) -> set[str]:
    """Every name the run emitted a row for, measured or marked."""
    return set((payload.get("costs") or {}).keys())


def role_sums(rows: dict[str, dict], fmt: str) -> dict[tuple[str, str], float]:
    """{(unit, role): summed predicted_dloss} over the given rows.

    `unit` is the routed-expert unit name (`model.layers.L.mlp.experts`), which
    is the key `provenance.unit_kls` uses -- not the bare layer prefix.
    """
    acc: dict[tuple[str, str], float] = {}
    for name, row in rows.items():
        m = EXPERT_RE.match(name)
        if not m:
            continue
        cell = row.get(fmt)
        if not isinstance(cell, dict) or cell.get("cost_source") != MEASURED:
            continue
        key = (m.group("layer") + ".mlp.experts", ROLE_OF[m.group("proj")])
        acc[key] = acc.get(key, 0.0) + float(cell.get("predicted_dloss", 0.0))
    return acc


def verify_against_unit_kls(run: dict, rows: dict[str, dict], fmt: str,
                            rtol: float = 2e-5) -> list[str]:
    """Re-assemble each run's unit KL from its own per-member rows."""
    problems: list[str] = []
    unit_kls = (run.get("provenance") or {}).get("unit_kls") or {}
    sums = role_sums(rows, fmt)
    if not unit_kls:
        return ["run carries no provenance.unit_kls -- nothing to verify"]
    for layer, per_fmt in unit_kls.items():
        if fmt not in per_fmt:
            continue
        want = float(per_fmt[fmt])
        got = sum(v for (lay, _role), v in sums.items() if lay == layer)
        if want == 0.0:
            if got != 0.0:
                problems.append(f"{layer}: unit KL {fmt}=0 but rows sum {got!r}")
            continue
        if abs(got - want) > rtol * abs(want):
            problems.append(
                f"{layer}: rows re-assemble {got!r}, provenance says {want!r} "
                f"({(got - want) / want:+.2%})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="AURA/body cost table to overlay expert rows into")
    ap.add_argument("--gu", required=True, help="gate_up_proj run pkl")
    ap.add_argument("--down", required=True, help="down_proj run pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="write the per-layer split table here (JSON)")
    ap.add_argument("--fmt", default="IQ2_XXS",
                    help="format whose predicted_dloss is the KL currency")
    ap.add_argument("--allow-partial", action="store_true",
                    help="warn instead of failing when a covered expert name is "
                         "left unmeasured (e.g. a single-layer smoke pair)")
    args = ap.parse_args()

    # A parent's measurement is routinely SHARDED across rented boxes (each box
    # owns a layer range), so --gu/--down take a comma list.  The shards of one
    # parent must be disjoint, carry the same calibration and the same
    # quantize_parents, and are verified individually.
    paths = {
        "base": [Path(p) for p in args.base.split(",") if p.strip()],
        "gate_up": [Path(p) for p in args.gu.split(",") if p.strip()],
        "down": [Path(p) for p in args.down.split(",") if p.strip()],
    }
    if len(paths["base"]) != 1:
        print("refusing: --base takes exactly one cost table", file=sys.stderr)
        return 2
    base = load(paths["base"][0])
    runs = {label: [load(p) for p in paths[label]] for label in ("gate_up", "down")}

    proj_of = {"gate_up": "gate_up_proj", "down": "down_proj"}
    for label, shards in runs.items():
        calibs = set()
        for shard in shards:
            p = shard.get("provenance") or {}
            if p.get("quantize_parents") != [proj_of[label]]:
                print(f"refusing: a --{label if label == 'down' else 'gu'} shard has "
                      f"quantize_parents={p.get('quantize_parents')!r}, expected "
                      f"[{proj_of[label]!r}]", file=sys.stderr)
                return 2
            calibs.add((p.get("calib_sha256"), p.get("n_calib_samples"),
                        p.get("calib_seqlen"), p.get("calib_seed"),
                        p.get("calib_batch")))
        if len(calibs) != 1:
            print(f"refusing: the {label} shards do not share one calibration: "
                  f"{sorted(calibs)}", file=sys.stderr)
            return 2
        print(f"  {label}: {len(shards)} shard(s), calibration "
              f"{sorted(calibs)[0][0][:16] if sorted(calibs)[0][0] else 'none'}")

    measured = {label: {} for label in ("gate_up", "down")}
    for label, shards in runs.items():
        for shard in shards:
            rows = measured_rows(shard)
            dupes = set(rows) & set(measured[label])
            if dupes:
                print(f"refusing: {len(dupes)} name(s) measured by two {label} "
                      f"shards, e.g. {sorted(dupes)[:3]} -- the shards must be "
                      f"disjoint", file=sys.stderr)
                return 2
            measured[label].update(rows)
    overlap = set(measured["gate_up"]) & set(measured["down"])
    if overlap:
        print(f"refusing: {len(overlap)} name(s) measured by BOTH parents, e.g. "
              f"{sorted(overlap)[:3]} -- the pair is not complementary",
              file=sys.stderr)
        return 2

    # Every expert name any shard covers must end up measured by exactly one.
    covered = set()
    for label, shards in runs.items():
        for shard in shards:
            covered |= covered_rows(shard)
    overlaid = set(measured["gate_up"]) | set(measured["down"])
    missing = sorted(covered - overlaid)
    if missing and not args.allow_partial:
        print(f"refusing: {len(missing)} expert name(s) covered but not measured "
              f"by either run, e.g. {missing[:3]}", file=sys.stderr)
        return 2
    if missing:
        print(f"WARNING: {len(missing)} covered name(s) left unmeasured "
              f"(partial pair), e.g. {missing[:3]}")

    # Verification: each shard's rows must re-assemble that shard's own unit KLs.
    problems: list[str] = []
    for label, shards in runs.items():
        for i, shard in enumerate(shards):
            problems += [f"[{label} shard {i}] {p}" for p in
                         verify_against_unit_kls(shard, measured_rows(shard), args.fmt)]
    if problems:
        print("VERIFICATION FAILED -- not writing:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 3

    # Per-label stats across that label's shards (each measured row's stats come
    # from the shard that measured it).
    run_stats = {label: {} for label in ("gate_up", "down")}
    for label, shards in runs.items():
        for shard in shards:
            run_stats[label].update(shard.get("stats") or {})

    # Overlay.
    out = dict(base)
    costs = dict(base.get("costs") or {})
    stats = dict(base.get("stats") or {})
    replaced: list[str] = []
    added: list[str] = []
    for label in ("gate_up", "down"):
        for name, row in measured[label].items():
            # A measured expert row the base never priced is ADDED, not refused.
            # Real case, 2026-09-22: the fresh body cost table carries 765 rows
            # for layer 14 instead of 768 -- all three projections of expert 187
            # are absent, while the probe has all three.  Refusing would block a
            # merge whose whole purpose is to price exactly those units, and
            # skipping would leave three routed-expert Linears unpriced in the
            # table the allocator reads.  The count is reported so the base's own
            # gap does not go unnoticed.
            if name not in costs:
                added.append(name)
                if not EXPERT_RE.match(name):
                    print(f"WARNING: {name} measured by the {label} run, absent "
                          f"from the base, and NOT a routed-expert member -- "
                          f"added anyway, but check the base", file=sys.stderr)
            else:
                replaced.append(name)
            costs[name] = row
            src_stats = run_stats[label].get(name)
            if src_stats is not None:
                stats[name] = src_stats
    out["costs"] = costs
    if stats:
        out["stats"] = stats
    prov = dict(out.get("provenance") or {})
    prov["per_parent_merge"] = {
        "schema": "pulsar.per_parent_cost_merge.v1",
        "base": {"path": str(paths["base"][0]), "sha256": sha256(paths["base"][0])},
        "runs": {
            label: {
                "shards": [
                    {
                        "path": str(p),
                        "sha256": sha256(p),
                        "quantize_parents": (shard.get("provenance") or {}).get("quantize_parents"),
                        "calib_sha256": (shard.get("provenance") or {}).get("calib_sha256"),
                        "n_calib_samples": (shard.get("provenance") or {}).get("n_calib_samples"),
                        "calib_seqlen": (shard.get("provenance") or {}).get("calib_seqlen"),
                        "calib_batch": (shard.get("provenance") or {}).get("calib_batch"),
                        "measured_rows": len(measured_rows(shard)),
                    }
                    for p, shard in zip(paths[label], runs[label])
                ],
                "overlaid_rows": len(measured[label]),
            }
            for label in ("gate_up", "down")
        },
        "overlaid_rows_total": len(replaced),
        "added_rows_absent_from_base": added,
    }
    out["provenance"] = prov

    # The deliverable: per-layer, per-role measured KL and cost per GiB.
    gu_sums = role_sums(measured["gate_up"], args.fmt)
    dn_sums = role_sums(measured["down"], args.fmt)
    layers = sorted({unit for unit, _ in gu_sums | dn_sums},
                    key=lambda s: (len(s), s))
    table = []
    for unit in layers:
        layer = unit[: -len(".mlp.experts")] if unit.endswith(".mlp.experts") else unit
        gu = gu_sums.get((unit, "gate_up"))
        dn = dn_sums.get((unit, "down"))
        row = {"layer": layer, "unit": unit, "gu_kl": gu, "down_kl": dn}
        if gu is not None:
            row["gu_kl_per_gib"] = gu / ROLE_GIB["gate_up"]
        if dn is not None:
            row["down_kl_per_gib"] = dn / ROLE_GIB["down"]
        if gu and dn:
            row["down_over_gu"] = dn / gu
            row["down_more_efficient"] = (dn / ROLE_GIB["down"]) > (gu / ROLE_GIB["gate_up"])
        table.append(row)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"merged {len(replaced)} measured expert rows into {paths['base'][0].name} "
          f"({len(costs)} rows total) -> {dest}")
    if added:
        print(f"  ADDED {len(added)} measured row(s) the base never priced "
              f"(base coverage gap): {sorted(added)[:4]}"
              f"{' ...' if len(added) > 4 else ''}")
    print(f"  gu measured rows: {len(measured['gate_up'])}, "
          f"down measured rows: {len(measured['down'])}, "
          f"covered/unmeasured: {len(covered)}/{len(missing)}")
    if table:
        print(f"  {'layer':<32} {'gu_kl':>12} {'down_kl':>12} {'down/gu':>8} "
              f"{'gu/GiB':>10} {'down/GiB':>10}")
        for row in table:
            print(f"  {row['layer']:<32} "
                  f"{row['gu_kl'] if row['gu_kl'] is not None else float('nan'):>12.6g} "
                  f"{row['down_kl'] if row['down_kl'] is not None else float('nan'):>12.6g} "
                  f"{row.get('down_over_gu', float('nan')):>8.3f} "
                  f"{row.get('gu_kl_per_gib', float('nan')):>10.3f} "
                  f"{row.get('down_kl_per_gib', float('nan')):>10.3f}")
        n_dn = sum(1 for r in table if r.get("down_more_efficient"))
        print(f"  down is the more efficient unit (KL/GiB) on {n_dn} of "
              f"{sum(1 for r in table if 'down_more_efficient' in r)} layers")
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({
            "schema": "pulsar.per_parent_split.v1",
            "format": args.fmt,
            "role_gib": ROLE_GIB,
            "base": str(paths["base"][0]),
            "runs": prov["per_parent_merge"]["runs"],
            "layers": table,
        }, indent=1, sort_keys=True) + "\n")
        print(f"  report -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

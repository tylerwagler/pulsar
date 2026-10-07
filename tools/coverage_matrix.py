#!/usr/bin/env python3
"""coverage_matrix.py -- the test system's contract x family coverage, checked (L278).

Reads tests/contracts.tsv (every gate's contract, family and tier) and the Makefile, then:
  - checks every listed gate exists (a make target, or a runner sub-gate name in tests/gates_runner.cpp) and is in
    the tier it claims (battery = in GATE_TARGETS / HOST_GATE_TARGETS / the runner; dev = selected by gates-dev);
  - prints the contract x family matrix (battery gates per cell);
  - FAILS on a contract x family cell that has no battery gate and is not a declared `gap` row naming the ledger
    row that closes it -- and on a `gap` row whose cell is in fact covered (a closed gap must be deleted).

Exit 0 = the matrix is what the file says.  Run from the engine root (make host-checks does).
"""
import re
import sys

CONTRACTS = ('session', 'loader', 'spec', 'spec-dist', 'reference', 'chat', 'server', 'tp')
FAMILIES = ('deepseek4', 'qwen4_exp')


def make_list(text, var):
    m = re.search(r'^' + var + r'\s*[:?]?=\s*((?:.*\\\n)*.*)$', text, re.M)
    if not m:
        sys.exit(f'coverage_matrix: {var} not found in the Makefile')
    return set(m.group(1).replace('\\\n', ' ').split())


def main():
    mk = open('Makefile').read()
    runner = open('tests/gates_runner.cpp').read()
    battery = make_list(mk, 'GATE_TARGETS') | make_list(mk, 'HOST_GATE_TARGETS')
    targets = set(re.findall(r'^([A-Za-z0-9_.-]+):', mk, re.M))
    runner_gates = set(re.findall(r'\{"([a-z][a-z0-9-]+)",\s*gate_\w+_main', runner)) | \
        set(re.findall(r'gate_spec \w+ = \{"([a-z][a-z0-9-]+)"', runner))
    m = re.search(r'ref_names\[\d+\] = \{([^}]*)\}', runner)   # the per-model reference gates' names
    if m:
        runner_gates |= set(re.findall(r'"([a-z][a-z0-9-]+)"', m.group(1)))
    rows, gaps, bad = [], [], []
    for n, line in enumerate(open('tests/contracts.tsv'), 1):
        line = line.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        f = line.split('\t')
        if f[0] == 'gap':
            if len(f) < 4 or f[1] not in CONTRACTS or f[2] not in FAMILIES or not re.match(r'L\d+', f[3]):
                bad.append(f'contracts.tsv:{n}: a gap row is: gap, contract, family, Lnnn, note')
                continue
            gaps.append((f[1], f[2], f[3], f[4] if len(f) > 4 else ''))
            continue
        if len(f) < 4:
            bad.append(f'contracts.tsv:{n}: needs target, contract, family, tier')
            continue
        target, contract, family, tier = f[:4]
        contracts = contract.split(',') if not contract.startswith('kernel:') else [contract]
        for c in contracts:
            if not (c in CONTRACTS or c.startswith('kernel:')):
                bad.append(f'contracts.tsv:{n}: unknown contract {c!r}')
        if family not in FAMILIES + ('all',):
            bad.append(f'contracts.tsv:{n}: unknown family {family!r}')
        if target.startswith('runner:'):
            name = target[len('runner:'):]
            exists, in_battery = name in runner_gates, 'cuda-runner-gate' in battery
        else:
            exists, in_battery = target in targets, target in battery
        if not exists:
            bad.append(f'contracts.tsv:{n}: {target} is not a make target / runner sub-gate')
        elif tier == 'battery' and not in_battery:
            bad.append(f'contracts.tsv:{n}: {target} says battery but make gates does not run it')
        elif tier != 'battery' and in_battery:
            bad.append(f'contracts.tsv:{n}: {target} says {tier} but make gates runs it -- say battery')
        rows.append((target, contracts, family, tier))

    print(f'  {"":11s}' + ''.join(f'{fam:>28s}' for fam in FAMILIES))
    for c in CONTRACTS:
        cells = []
        for fam in FAMILIES:
            hits = [t for t, cc, ff, tier in rows if c in cc and ff in (fam, 'all') and tier == 'battery']
            gap = [g for g in gaps if g[0] == c and g[1] == fam]
            if hits and gap:
                bad.append(f'{c} x {fam} is covered ({hits[0]}) but still declared a gap ({gap[0][2]}) -- delete the gap row')
            if not hits and not gap:
                bad.append(f'{c} x {fam} has no battery gate and no declared gap')
            cells.append(f'{len(hits)} gate(s)' if hits else (f'GAP ({gap[0][2]})' if gap else 'MISSING'))
        print(f'  {c:11s}' + ''.join(f'{x:>28s}' for x in cells))
    for g in gaps:
        print(f'  gap: {g[0]} x {g[1]} -- {g[2]}: {g[3]}')
    for b in bad:
        print(f'  FAIL  {b}')
    print('COVERAGE MATRIX ' + ('PASS' if not bad else f'FAIL ({len(bad)})'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

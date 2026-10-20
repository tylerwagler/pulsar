"""Single-stream end-to-end estimate from L288's measured round costs (C2, td305 code, 2026-10-14):
round = T0 + ROW x (verify rows - 1) + drafter, T0 = 29.0 ms (a 1-row step incl. head + round end), ROW = 2.2 ms a verify
row (C2 2.18 wall; 2.49 in the 10-12 nsys -> ROW_HI), MTP step 1.2 ms + absorb 0.8 ms a round, DFlash2 forward +
absorb D ms a round (UNMEASURED: 2 ms = NVFP4 drafter + small head, 4 ms = full head).  Check: C2's code round
(5.02 rows, 3.9 MTP steps) -> 29 + 2.2 x 4.02 + 3.9 x 1.2 + 0.8 = 43.3 ms vs 42.8 measured."""
import json
import sys

T0, MTP_STEP, MTP_ABS = 29.0, 1.2, 0.8


def mtp_trace(path, row):
    rows = [tuple(map(float, l.split()[:2])) for l in open(path) if l.strip()]
    ms = sum(T0 + row * nv + MTP_STEP * nv + MTP_ABS for nv, _ in rows)
    tok = sum(nk + 1 for _, nk in rows)
    return tok / len(rows), sum(nv for nv, _ in rows) / len(rows), ms / tok


def mtp_fixed(path, row):
    rows = [tuple(map(int, l.split()[:2])) for l in open(path) if l.strip()]
    D = max(nv for nv, _ in rows)
    best = None
    for k in range(1, D + 1):
        full = [r for r in rows if r[0] == D]
        tok = sum(min(nk, k) + 1 for _, nk in full) / len(full)
        ms = (T0 + (row + MTP_STEP) * k + MTP_ABS) / tok
        if best is None or ms < best[2]:
            best = (k, tok, ms)
    return best


for row in (2.2, 2.5):
    print(f"== ROW {row} ms")
    for fx, served, unst in [("code", "taps/mtp-code-d8-off.trace", "p1/ours-td305-code-d6-off.trace"),
                             ("prose", "p1/ours-td305-prose-d4-0.7.trace", "p1/ours-td305-prose-d6-off.trace"),
                             ("story", "p1/ours-td305-story-d4-0.7.trace", "p1/ours-td305-story-d6-off.trace"),
                             ("edit", "taps/mtp-edit-d8-off.trace", None)]:
        tpr, nv, ms = mtp_trace(served, row)
        line = f"{fx:5s} MTP served ({served.split('/')[-1]}): {tpr:.2f} tok/rnd, {nv:.2f} drafts, {ms:.2f} ms/tok"
        if unst:
            k, t, m = mtp_fixed(unst, row)
            line += f" | MTP best fixed k={k}: {t:.2f} tok/rnd {m:.2f} ms/tok"
        print(line)
        res = json.load(open(f"results/{fx}.json"))
        for D in (2.0, 4.0):
            cells = []
            for B in (4, 6, 8):
                s = res[f"B{B}-sel"]
                m = (T0 + row * (B - 1) + D) / s["tokens_per_round"]
                cells.append(f"B{B} {s['tokens_per_round']:.2f}/{m:.2f}ms ({ms / m - 1:+.0%})")
            print(f"      DFlash2 D={D:.0f}ms: " + "  ".join(cells))

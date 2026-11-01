#!/usr/bin/env python3
"""L338 P2.3: grade the runtime JEV adapter (tests/gemma4_jev_gate.cpp's capture) against the BF16 CPU reference.

  jev_grade.py prep  <decision_ref.pt> <cases.bin>
  jev_grade.py grade <decision_ref.pt> <out.bin> <JEV snapshot dir>

decision_ref.pt (the L338 P0 reference, make_refs.py): meta {Q_NOUL, Q_SCORE, T, template} and cases, each with
noul_ids / noul_h (final-norm last-token hidden, fp32) / noul_p, and for the 18 judge cases score_ids / score_h /
score_p.  prep writes every noul prompt in order, then every score prompt (the gate syncs them in that order); grade
reads one f32 [hidden] row per prompt back and runs the decision head on it exactly as the reference does:
z = 30 tanh((W h + b) / 30), softmax over the kind's slot range / T[kind].

Pass: every judge case keeps its expected verdict at P(block) = 0.5, |dP(block)| <= 0.10 on them, |dE[score]| <= 0.5.
Reported, not graded: the replay cases' verdict agreement with the reference and the hidden's cosine (the EXL3 4 bpw
backbone against BF16 is the fidelity being shown, not a bit-exact claim)."""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors.numpy import load_file

P_TOL, SCORE_TOL, CUT = 0.10, 0.5, 0.5


def load(ref):
    return torch.load(ref, weights_only=False)


def prompts(dec):
    out = [("noul", i, c["noul_ids"]) for i, c in enumerate(dec["cases"])]
    out += [("score", i, c["score_ids"]) for i, c in enumerate(dec["cases"]) if "score_ids" in c]
    return out


def prep(ref, path):
    ps = prompts(load(ref))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ps)))
        for _, _, ids in ps:
            a = np.asarray(ids, dtype=np.int32)
            f.write(struct.pack("<I", len(a)))
            f.write(a.tobytes())
    print(f"jev-grade prep: {len(ps)} prompts ({sum(len(p[2]) for p in ps)} tokens) -> {path}")


def grade(ref, outbin, snap):
    dec = load(ref)
    ps = prompts(dec)
    jc = json.load(open(os.path.join(snap, "judge_config.json")))
    T = json.load(open(os.path.join(snap, "calibration_gold.json")))["per_kind"]
    head = load_file(os.path.join(snap, "head.safetensors"))
    W, b = head["proj.weight"].astype(np.float64), head["proj.bias"].astype(np.float64)
    E = W.shape[1]
    H = np.fromfile(outbin, dtype=np.float32)
    if H.size != len(ps) * E:
        sys.exit(f"jev-grade: {outbin} holds {H.size} floats, want {len(ps)} x {E}")
    H = H.reshape(len(ps), E).astype(np.float64)

    def probs(h, kind, n):
        z = 30.0 * np.tanh((W @ h + b) / 30.0)
        s = jc["slots"]["ranges"][kind][0]
        x = z[s:s + n] / T[kind]
        x = np.exp(x - x.max())
        return x / x.sum()

    fails, rows = 0, {}
    for k, (kind, i, _) in enumerate(ps):
        c = dec["cases"][i]
        h_ref = c[f"{kind}_h"].double().numpy()
        cos = float(H[k] @ h_ref / (np.linalg.norm(H[k]) * np.linalg.norm(h_ref)))
        p = probs(H[k], kind, 2 if kind == "noul" else 6)
        rows.setdefault(i, {})[kind] = (cos, p, c[f"{kind}_p"].double().numpy())
    n_agree = n_rep = 0
    print(f"{'group':8s} {'label':28s} {'exp':6s} {'P(block) ref':>12s} {'pulsar':>8s} {'cos':>8s} {'E[s] ref':>8s} {'pulsar':>8s}")
    for i, c in enumerate(dec["cases"]):
        cos, p, pr = rows[i]["noul"]
        line = f"{c['group']:8s} {c['label'][:28]:28s} {c['expect'][:6]:6s} {pr[1]:12.4f} {p[1]:8.4f} {cos:8.5f}"
        same = (p[1] >= CUT) == (pr[1] >= CUT)
        if c["group"] == "judge18":
            ok = (p[1] >= CUT) == (c["expect"] == "block") and abs(p[1] - pr[1]) <= P_TOL
            if "score" in rows[i]:
                _, ps_, psr = rows[i]["score"]
                es, esr = float(np.arange(6) @ ps_), float(np.arange(6) @ psr)
                ok = ok and abs(es - esr) <= SCORE_TOL
                line += f" {esr:8.3f} {es:8.3f}"
            fails += not ok
            line += "" if ok else "   FAIL"
        else:
            n_rep += 1
            n_agree += same
            line += "" if same else "   (verdict flips vs ref)"
        print(line)
    j = [rows[i]["noul"] for i, c in enumerate(dec["cases"]) if c["group"] == "judge18"]
    dps = [abs(p[1] - pr[1]) for _, p, pr in j]
    coss = [rows[i][k][0] for i in rows for k in rows[i]]
    print(f"judge18: max |dP(block)| {max(dps):.4f}, mean {np.mean(dps):.4f}; replay verdicts agree {n_agree}/{n_rep}; "
          f"hidden cosine min {min(coss):.5f} mean {np.mean(coss):.5f}")
    print(f"JEV GRADE {'FAIL' if fails else 'PASS'} ({fails} judge cases out of tolerance)")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "grade":
        sys.exit(grade(sys.argv[2], sys.argv[3], sys.argv[4]))
    else:
        sys.exit(__doc__)

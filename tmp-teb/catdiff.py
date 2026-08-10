import json, sys
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
names = sys.argv[1:]
runs = {n: json.load(open(f"{base}/{n}.json")) for n in names}
cats = {n: {c["category"]: c for c in runs[n]["scores"]["category_scores"]} for n in names}
first = names[0]
hdr = f"{'cat':4s} {'label':28s}" + "".join(f" {n[-8:]:>10s}" for n in names)
print(hdr)
for k in sorted(cats[first]):
    row = f"{k:4s} {cats[first][k]['label'][:28]:28s}"
    vals = [f"{cats[n][k]['earned']}/{cats[n][k]['max']}" for n in names]
    diff = len(set(vals)) > 1
    row += "".join(f" {v:>10s}" for v in vals) + ("   <--" if diff else "")
    print(row)
for n in names:
    print(n, "final:", runs[n]["final_score"])

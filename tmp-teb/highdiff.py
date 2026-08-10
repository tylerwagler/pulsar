import json
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
r = json.load(open(f"{base}/pin-high-s42.json"))
print("PINNED HIGH final:", r["final_score"], r.get("rating"))
low = json.load(open(f"{base}/pin-full-s42.json"))
cl = {c["category"]: c for c in low["scores"]["category_scores"]}
ch = {c["category"]: c for c in r["scores"]["category_scores"]}
for k in sorted(ch):
    a, b = cl[k], ch[k]
    mark = "  <--" if a["earned"] != b["earned"] else ""
    print(f"{k:3s} {b['label'][:26]:26s} low {a['earned']:2d}/{a['max']:<2d}  "
          f"high {b['earned']:2d}/{b['max']:<2d}{mark}")

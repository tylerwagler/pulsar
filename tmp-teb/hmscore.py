import json, sys, glob
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
for f in sorted(glob.glob(base + "/hm-*.json")):
    r = json.load(open(f))
    p = [c for c in r["scores"]["category_scores"] if c["category"] == "P"][0]
    print(f.split("/")[-1], f"P={p['earned']}/{p['max']}",
          f"pass={p['pass_count']} partial={p['partial_count']} fail={p['fail_count']}",
          "final:", r["final_score"])

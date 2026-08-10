import json
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
for n in ("ours-seed43", "his-seed43"):
    r = json.load(open(f"{base}/{n}.json"))
    c = r.get("config", {})
    keys = ("temperature", "seed", "top_p", "top_k", "min_p", "backend",
            "model", "no_think", "max_turns")
    print(n, {k: c.get(k) for k in keys if k in c})

import json
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
ids = []
for line in open(f"{base}/pin-full-s42.progress"):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("event") == "scenario_result" and e.get("status") != "pass":
        ids.append((e["scenario_id"], e.get("status")))
print(" ".join(i for i, _ in ids))
for i, s in ids:
    print(i, s)

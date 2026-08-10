import glob, re, os
base = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/runs/2026/08"
for f in sorted(glob.glob(base + "/*.md")):
    text = open(f).read()
    if "pp2048" not in text:
        continue
    rows = [l.strip() for l in text.splitlines() if re.search(r"pp2048.*c[124]", l)]
    if not rows:
        continue
    print("==", os.path.basename(f), "mtime", os.path.getmtime(f))
    for r in rows:
        print("  ", r)

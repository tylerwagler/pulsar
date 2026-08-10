import json, time

D = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
seen = {"pressure-75": 0, "pressure-0": 0}
done = set()
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    for tag in ("pressure-75", "pressure-0"):
        try:
            lines = open(f"{D}/{tag}.progress").read().splitlines()
        except FileNotFoundError:
            continue
        results = []
        final = None
        for line in lines:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "scenario_result":
                results.append(e)
            if e.get("event") == "benchmark_complete":
                final = e.get("final_score")
        for e in results[seen[tag]:]:
            print(f"[{tag}] {e['scenario_id']}: {e.get('status')} "
                  f"({e.get('duration_seconds', 0):.0f}s)", flush=True)
        seen[tag] = len(results)
        if final is not None and tag not in done:
            done.add(tag)
            print(f"[{tag}] COMPLETE: final_score={final}", flush=True)
    if len(done) == 2:
        break
    time.sleep(30)

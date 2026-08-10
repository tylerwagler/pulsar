import json, sys, time, os

D = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
old = {"TC-14": "partial", "TC-38": "partial", "TC-43": "partial",
       "TC-45": "partial", "TC-51": "partial", "TC-57": "partial",
       "TC-58": "partial", "TC-60": "fail", "TC-62": "partial",
       "TC-66": "partial", "TC-69": "partial", "TC-71": "fail",
       "TC-72": "fail", "TC-73": "partial", "TC-74": "partial",
       "TC-75": "fail", "TC-76": "partial", "TC-80": "fail",
       "TC-82": "partial", "TC-83": "fail", "TC-84": "partial"}
seen = 0
deadline = time.time() + 3500
while time.time() < deadline:
    try:
        lines = open(f"{D}/retry-fails.progress").read().splitlines()
    except FileNotFoundError:
        lines = []
    results = []
    done = False
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "scenario_result":
            results.append(e)
        if e.get("event") == "benchmark_complete":
            done = True
    for e in results[seen:]:
        sid = e["scenario_id"]
        was = old.get(sid, "?")
        now = e.get("status")
        arrow = "=" if was == now else ("IMPROVED" if now == "pass" or
                (was == "fail" and now == "partial") else "WORSE")
        print(f"{sid}: {was} -> {now}  [{arrow}]  "
              f"({e.get('duration_seconds', 0):.0f}s)", flush=True)
    seen = len(results)
    if done or (seen >= len(old)):
        print(f"retry complete: {seen} scenarios", flush=True)
        break
    time.sleep(20)

import json, time

D = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
old = {}
for line in open(f"{D}/pin-full-s42.progress"):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("event") == "scenario_result":
        old[e["scenario_id"]] = e.get("status")
seen = 0
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    try:
        lines = open(f"{D}/confirm-full-s42.progress").read().splitlines()
    except FileNotFoundError:
        lines = []
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
    for e in results[seen:]:
        sid = e["scenario_id"]
        was = old.get(sid, "?")
        now = e.get("status")
        tag = "=" if was == now else ("IMPROVED" if now == "pass" or
              (was == "fail" and now == "partial") else "WORSE")
        print(f"{sid} [{e.get('index', 0) + 1}/84]: {was} -> {now}  [{tag}]",
              flush=True)
    seen = len(results)
    if final is not None:
        print(f"CONFIRM RUN COMPLETE: final_score={final} "
              f"(pinned baseline 84, Entrpi 85)", flush=True)
        break
    time.sleep(20)

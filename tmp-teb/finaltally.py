import json
D = "/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb"
r = json.load(open(f"{D}/confirm-full-s42.json"))
print("FINAL:", r["final_score"], r.get("rating"))
for line in open(f"{D}/confirm-full-s42.progress"):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("event") == "scenario_result" and e["scenario_id"] == "TC-84":
        print("TC-84:", e.get("status"))
old = json.load(open(f"{D}/pin-full-s42.json"))
co = {c["category"]: c for c in old["scores"]["category_scores"]}
cn = {c["category"]: c for c in r["scores"]["category_scores"]}
for k in sorted(cn):
    if co[k]["earned"] != cn[k]["earned"]:
        print(f"{k} {cn[k]['label'][:24]}: {co[k]['earned']}/{co[k]['max']}"
              f" -> {cn[k]['earned']}/{cn[k]['max']}")

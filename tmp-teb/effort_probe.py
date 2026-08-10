import json, urllib.request
P = ("A farmer has 3 fields. Field A yields 40 bushels per acre on 12 acres. "
     "Field B yields 15% more per acre than A on 8 acres. Field C yields 20% "
     "less per acre than B on 20 acres. Grain sells at 6.20 per bushel but "
     "the co-op takes 4% commission and hauling costs 180 per field. What is "
     "the farmer's net revenue? Show only the final number.")
for e in ("none", "low", "high", "max"):
    req = urllib.request.Request(
        "http://sparky.defense.lan:8099/v1/chat/completions",
        json.dumps({"model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": P}],
                    "max_tokens": 3000, "temperature": 0,
                    "reasoning_effort": e}).encode(),
        {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    print(e, "reasoning:", len(m.get("reasoning_content") or ""),
          "| answer:", (m.get("content") or "").strip()[:60])

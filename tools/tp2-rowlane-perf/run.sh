#!/bin/bash
# Row-lane decode A/B on the pair (L269-perf).  Build A = the row lane as the pair runs it today (combine kernels
# spin on the done word); build B = the new build (the stream waits on the done word, no SM held).  Measures decode
# tok/s and the per-step wall at 1/2/4 streams on the V4 0731 production model, rounds interleaved A,B / B,A; then,
# for each layer-subset fixture present on BOTH nodes, a 256-token greedy TP=2 CLI decode per build (ms/token) with
# the logprobs compared byte for byte.  Prints one table to paste back.
#
#   tools/tp2-rowlane-perf/run.sh <build A dir> <build B dir>
#
# Both dirs at the same paths on both nodes, each built with `make pulsar pulsar-server`.  Run on the HEAD
# (ca1070wk30008) as the serving user.  Settings (env): STREAMS ("1 2 4"), PP (512), TG (256), ROUNDS (2),
# V4FIX (~/models/vexp-tp-fixture), V41FIX (~/models/v41-tp1-fixture-mx), MODEL (~/models/DeepSeek-v4-Flash).
# See docs/history/tp2-rowlane-perf-README.md.  The pair plumbing is tools/tp2/lib.sh.
set -u
A=${1:?usage: run.sh <build A dir> <build B dir>}
B=${2:?usage: run.sh <build A dir> <build B dir>}
KIT=rowlane-ab
MODEL=${MODEL:-$HOME/models/DeepSeek-v4-Flash}
STREAMS=${STREAMS:-1 2 4}
PP=${PP:-512}
TG=${TG:-256}
ROUNDS=${ROUNDS:-2}
V4FIX=${V4FIX:-$HOME/models/vexp-tp-fixture}
V41FIX=${V41FIX:-$HOME/models/v41-tp1-fixture-mx}
CTX=${CTX:-262144}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../tp2/lib.sh"
RES=$OUT/results.jsonl
: > "$RES"

proto_of() { grep -o 'PULSAR_TP_PROTOCOL_VERSION [0-9]*u' "$1/src/tp/pulsar_tp.h" | grep -o '[0-9]*'; }
lane_lines() {   # what each rank's bring-up said about the lane (and the wait, if the build announces one)
    grep -h -o -E "row lane (ON|OFF)[^)]*\)|bulk lane armed[^)]*\)" "$OUT/head-$1.log" | sort -u | head -3 | sed 's/^/    /'
}

note "row-lane A/B: A $A, B $B, worker $WORKER, model $MODEL, streams [$STREAMS] pp $PP tg $TG, $ROUNDS rounds, out $OUT"
for D in "$A" "$B"; do
    PROTO=$(proto_of "$D")
    preflight
    [ -x "$D/pulsar" ] && ssh "$WORKER" "test -x $D/pulsar" || note "$D: no CLI binary 'pulsar' on both nodes -- fixture legs skip it"
done
[ -e "$MODEL/model.safetensors.index.json" ] && ssh "$WORKER" "test -e $MODEL/model.safetensors.index.json" ||
    { verdict "model $(basename "$MODEL")" "FAIL (missing on a node)"; final_verdict "ROW-LANE A/B"; }

# ---- 1. the production model, served: decode at 1/2/4 streams, per build, rounds interleaved ------------------
serve_leg() {   # $1 = arm (A|B), $2 = round
    local tag="$1$2" T
    D=$([ "$1" = A ] && echo "$A" || echo "$B")
    PROTO=$(proto_of "$D")
    ssh "$WORKER" "rm -rf $KV"; rm -rf "$KV"
    reclaim
    T=$(start_pair "") || { stop_pair; journals "$T" "$tag"; verdict "bring-up $tag" FAIL; return 1; }
    python3 "$HERE/decode_load.py" 8000 1 "$PP" 32 > /dev/null 2>&1       # warm: first-use allocations, not measured
    for c in $STREAMS; do
        local j
        j=$(python3 "$HERE/decode_load.py" 8000 "$c" "$PP" "$TG" 2> "$OUT/load-$tag-c$c.err") ||
            { verdict "load $tag c$c" "FAIL ($OUT/load-$tag-c$c.err)"; continue; }
        echo "{\"leg\": \"serve\", \"arm\": \"$1\", \"round\": $2, \"load\": $j}" >> "$RES"
        note "$tag c$c $j"
    done
    stop_pair; journals "$T" "$tag"; identity "$tag"; alarms "$tag"
    lane_lines "$tag"
}
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) = 1 ]; then serve_leg A "$r"; serve_leg B "$r"; else serve_leg B "$r"; serve_leg A "$r"; fi
done

# ---- 2. fixtures (optional): 256 greedy tokens through the TP=2 CLI, ms/token per build, logprobs A == B -------
# A 7-layer fixture is where the lane's cost is largest relative to the work (L269: on sparky's loopback the V4
# fixture went 5.15 -> 15.5 s per 256 tokens with the lane on).  Build A predating v26 runs these fixtures with the
# lane OFF (the gate ring was too small for < 12 layers, and V4.1's 20 KB rows exceeded the old 16 KB cap); build B
# runs them with it ON -- the comparison is "lane off vs lane on" for A < v26, "spin vs wait" for A >= v26.
PROMPT="Explain how a C pointer differs from an array in one paragraph."
cli_leg() {   # $1 = fixture tag, $2 = fixture dir, $3 = arm
    local tag="$1-$3" T rc
    D=$([ "$3" = A ] && echo "$A" || echo "$B")
    [ -x "$D/pulsar" ] && ssh "$WORKER" "test -x $D/pulsar" || { verdict "fixture $tag" "SKIPPED (no CLI binary)"; return; }
    reclaim
    T=$(date '+%Y-%m-%d %H:%M:%S')
    local args="-m $2 -c 4096 --nothink --temp 0 -n 257 -p \"$PROMPT\""
    local unit="sudo -n systemd-run --uid=$(id -un) --gid=$(id -gn) -p LimitMEMLOCK=infinity -p WorkingDirectory=$D \
--setenv=HOME=$HOME --setenv=PULSAR_TP_RDMA_DEV=$RDMA_DEV --setenv=PULSAR_TP_RDMA_DEV2=$RDMA_DEV2"
    ssh "$WORKER" "$unit --unit=$UNIT $D/pulsar $args --tp-rank 1 --tp-nranks 2 --tp-peers $PEERS --tp-port 5590 \
--dump-logprobs $HOME/$KIT-worker-lp.json" > /dev/null
    sleep 5
    eval "$unit --wait --pipe $D/pulsar $args --tp-rank 0 --tp-nranks 2 --tp-peers $PEERS --tp-port 5590 \
--dump-logprobs $OUT/lp-$tag.json" 2>&1 < /dev/null |
        while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$OUT/cli-$tag.log"
    stop_pair
    ssh "$WORKER" "sudo -n journalctl -u $UNIT --since '$T' --no-pager" > "$OUT/worker-cli-$tag.log" 2>&1
    # decode window: the first decode-tier line (prefill is done) .. the run's closing identity tally
    python3 - "$OUT/cli-$tag.log" "$1" "$3" >> "$RES" <<'PY'
import json, sys
t0 = t1 = None
lane = "?"
for l in open(sys.argv[1], errors="replace"):
    ts, _, msg = l.partition(" ")
    if "row lane ON" in msg: lane = "ON"
    if "row lane OFF" in msg and lane == "?": lane = "OFF"
    if t0 is None and "decode attention =" in msg: t0 = float(ts)
    if "cross-rank logits identity" in msg: t1 = float(ts)
ms = round((t1 - t0) * 1e3 / 256, 2) if t0 and t1 else None
print(json.dumps({"leg": "fixture", "fixture": sys.argv[2], "arm": sys.argv[3], "lane": lane, "ms_per_token": ms}))
PY
    tail -1 "$RES" | sed 's/^/  /'
    grep -q "cross-rank logits identity: \([0-9]*\)/\1 " "$OUT/cli-$tag.log" && verdict "fixture $tag identity" PASS ||
        verdict "fixture $tag identity" "FAIL ($OUT/cli-$tag.log)"
}
for f in "v4fix:$V4FIX" "v41fix:$V41FIX"; do
    name=${f%%:*}; dir=${f#*:}
    if [ -e "$dir/model.safetensors.index.json" ] && ssh "$WORKER" "test -e $dir/model.safetensors.index.json"; then
        cli_leg "$name" "$dir" A
        cli_leg "$name" "$dir" B
        if cmp -s "$OUT/lp-$name-A.json" "$OUT/lp-$name-B.json"; then verdict "fixture $name logprobs A == B" PASS
        else verdict "fixture $name logprobs A == B" "FAIL (cmp $OUT/lp-$name-A.json $OUT/lp-$name-B.json)"; fi
    else
        verdict "fixture $name" "SKIPPED ($dir absent on a node)"
    fi
done

# ---- 3. the table ---------------------------------------------------------------------------------------------
python3 - "$RES" "$A" "$B" <<'PY' | tee "$OUT/table.txt"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"\nROW-LANE A/B   A = {sys.argv[2]}   B = {sys.argv[3]}")
print("\n| streams | A decode tok/s | B decode tok/s | B/A | A per-stream | B per-stream | A step ms (p90) | B step ms (p90) | tok/step A/B |")
print("|---|---|---|---|---|---|---|---|---|")
by = {}
for r in rows:
    if r["leg"] == "serve": by.setdefault((r["load"]["streams"], r["arm"]), []).append(r["load"])
def m(xs, k): return statistics.mean(x[k] for x in xs) if xs else float("nan")
for c in sorted({k[0] for k in by}):
    a, b = by.get((c, "A"), []), by.get((c, "B"), [])
    ra = m(b, "decode_tps") / m(a, "decode_tps") if a and b and m(a, "decode_tps") else float("nan")
    print(f"| {c} | {m(a,'decode_tps'):.1f} | {m(b,'decode_tps'):.1f} | {ra:.3f} | {m(a,'per_stream_tps'):.1f} | "
          f"{m(b,'per_stream_tps'):.1f} | {m(a,'step_ms_med'):.1f} ({m(a,'step_ms_p90'):.1f}) | "
          f"{m(b,'step_ms_med'):.1f} ({m(b,'step_ms_p90'):.1f}) | {m(a,'tok_per_step'):.2f}/{m(b,'tok_per_step'):.2f} |")
print("\nper round (decode tok/s): " + "; ".join(
    f"{r['arm']}{r['round']} c{r['load']['streams']} {r['load']['decode_tps']}" for r in rows if r["leg"] == "serve"))
fx = [r for r in rows if r["leg"] == "fixture"]
if fx:
    print("\n| fixture | A lane | A ms/token | B lane | B ms/token |")
    print("|---|---|---|---|---|")
    for name in sorted({r["fixture"] for r in fx}):
        a = next((r for r in fx if r["fixture"] == name and r["arm"] == "A"), {})
        b = next((r for r in fx if r["fixture"] == name and r["arm"] == "B"), {})
        print(f"| {name} | {a.get('lane')} | {a.get('ms_per_token')} | {b.get('lane')} | {b.get('ms_per_token')} |")
PY

final_verdict "ROW-LANE A/B"

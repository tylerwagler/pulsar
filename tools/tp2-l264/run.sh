#!/bin/bash
# L264 TP2 validation: the KV state model (grid checkpoints, segment chains,
# their TP mirroring -- protocol v23) on the pair.  Run on the HEAD
# (ca1070wk30008) as the serving user, from the engine tree built on BOTH nodes
# at the same path:
#
#   tools/tp2-l264/run.sh /home/tyler.wagler.local/pulsar-l264
#
# See tools/tp2-l264/README.md for what each step proves and how to read a
# failure.  Every step logs to $OUT; the last lines are the verdict.
set -u
D=${1:?usage: run.sh <build dir, same path on both nodes>}
KIT=l264-tp2
MODEL=${MODEL:-$HOME/models/DeepSeek-v4-Flash}
GUARD_BUDGET=${GUARD_BUDGET:-}   # bytes; default: the server's eager floor + 0.1 GiB (step 5 must spill)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../tp2/lib.sh"
STATE=$OUT/smoke-state.json

note "L264 TP2 validation: build $D, worker $WORKER, out $OUT"

# ---- 0. preflight ----------------------------------------------------------
preflight

# ---- 1. host tests (no GPU, no pair) ---------------------------------------
host_tests

# ---- 2. fresh pair, phase 1: resume across the stripped-reasoning echo ------
stop_pair
ssh "$WORKER" "rm -rf $KV"; rm -rf "$KV"
reclaim
T=$(start_pair "") || { journals "$T" boot1; verdict bring-up FAIL; exit 1; }
python3 "$HERE/smoke.py" phase1 "$STATE" > "$OUT/smoke-phase1.log" 2>&1 && verdict "phase1 (resume)" PASS || verdict "phase1 (resume)" "FAIL ($OUT/smoke-phase1.log)"
cat "$OUT/smoke-phase1.log"

# ---- 3. restart with a planted orphan copy; phase 2: restore from disk ------
ORPHAN=$KV/tp-seg/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tpseg
ssh "$WORKER" "mkdir -p $KV/tp-seg && echo orphan > $ORPHAN"
stop_pair; journals "$T" run1; identity run1; alarms run1
LEADER_SEGS=$(ls "$KV"/*.seg 2>/dev/null | wc -l)
reclaim
T=$(start_pair "") || { journals "$T" boot2; verdict bring-up2 FAIL; exit 1; }
sleep 5
ssh "$WORKER" "test -e $ORPHAN" && verdict reconcile "FAIL (the planted orphan copy survived the bring-up)" || verdict reconcile PASS
WORKER_SEGS=$(ssh "$WORKER" "ls $KV/tp-seg/*.tpseg 2>/dev/null | wc -l")
[ "$LEADER_SEGS" -gt 0 ] && [ "$LEADER_SEGS" = "$WORKER_SEGS" ] && verdict "copies" "PASS ($LEADER_SEGS segments on each rank)" ||
    verdict "copies" "FAIL (leader $LEADER_SEGS segments, worker $WORKER_SEGS copies)"
python3 "$HERE/smoke.py" phase2 "$STATE" > "$OUT/smoke-phase2.log" 2>&1 && verdict "phase2 (disk restore)" PASS || verdict "phase2 (disk restore)" "FAIL ($OUT/smoke-phase2.log)"
cat "$OUT/smoke-phase2.log"

# ---- 4. corrupt worker copies are a clean miss on every rank ---------------
# Every copy, so turn 3's chain is certain to hit one: the leader's own files
# stay intact, only the worker's no longer reach the leader's state.
ssh "$WORKER" "for f in $KV/tp-seg/*.tpseg; do truncate -s 4096 \$f; done"
python3 "$HERE/smoke.py" miss "$STATE" > "$OUT/smoke-miss.log" 2>&1 && verdict "miss" PASS || verdict "miss" "FAIL ($OUT/smoke-miss.log)"
cat "$OUT/smoke-miss.log"
stop_pair; journals "$T" run2; identity run2
MISS=$(count "kv segment load miss" "$OUT/worker-run2.log")
[ "$MISS" -ge 1 ] && verdict "miss seen" "PASS ($MISS worker miss lines)" || verdict "miss seen" "FAIL (no worker miss logged -- did G restore from disk at all?)"
# run2's alarms: the miss must not have failed the pair
FAILED=$(cat "$OUT/head-run2.log" "$OUT/worker-run2.log" | grep -c -i -E "marked failed|pair is marked")
[ "$FAILED" = 0 ] && verdict "alarms run2" PASS || verdict "alarms run2" "FAIL ($FAILED lines)"

# ---- 5. the eviction guard spills to the segment chain and restores --------
# The guard keeps touched KV under (budget - eager floor); an ~18k-token bank
# touches ~60 MiB, so 0.1 GiB above the floor holds two and the rest spill.
if [ -z "$GUARD_BUDGET" ]; then
    EAGER=$(grep -o "eager [0-9.]*" "$OUT/head-run1.log" | head -1 | awk '{print $2}')
    GUARD_BUDGET=$(python3 -c "print(int((float('${EAGER:-6.0}') + 0.1) * 2**30))")
    note "guard budget: eager ${EAGER:-?} GiB + 0.1 GiB = $GUARD_BUDGET bytes"
fi
reclaim
T=$(start_pair "--setenv=PULSAR_SERVER_KV_BUDGET_OVERRIDE=$GUARD_BUDGET") || { journals "$T" boot3; verdict bring-up3 FAIL; exit 1; }
python3 "$HERE/smoke.py" guard "$STATE" > "$OUT/smoke-guard.log" 2>&1 && verdict "guard" PASS || verdict "guard" "FAIL ($OUT/smoke-guard.log)"
cat "$OUT/smoke-guard.log"
stop_pair; journals "$T" run3; identity run3; alarms run3
RESTORED=$(count "guard RESTORED bank" "$OUT/head-run3.log")
[ "$RESTORED" -ge 1 ] && verdict "guard restores" "PASS ($RESTORED)" || verdict "guard restores" "FAIL (no spill fired -- lower GUARD_BUDGET)"

# ---- verdict ----------------------------------------------------------------
final_verdict "L264 TP2"

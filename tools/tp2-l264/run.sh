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
WORKER=${WORKER:-ca1070wk30007}
PEERS=${PEERS:-192.168.9.13:5590,192.168.9.12:5590}     # rank 0 (this host) first
MODEL=${MODEL:-$HOME/models/DeepSeek-v4-Flash}
RDMA_DEV=${RDMA_DEV:-mlx5_3}
RDMA_DEV2=${RDMA_DEV2:-mlx5_1}
KV=${KV:-$HOME/l264-tp2-kv}                               # a fresh disk KV dir on BOTH nodes
GUARD_BUDGET=${GUARD_BUDGET:-}   # bytes; default: the server's eager floor + 0.1 GiB (step 5 must spill)
OUT=${OUT:-$HOME/l264-tp2-$(date +%m%d-%H%M)}
UNIT=pulsar-tp-exp
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
STATE=$OUT/smoke-state.json
verdicts=()
note() { echo "[$(date +%H:%M:%S)] $*"; }
verdict() { verdicts+=("$1: $2"); note "VERDICT $1: $2"; }

STOP="sudo -n systemctl stop pulsar-tp pulsar-tp-prof $UNIT 2>/dev/null; sudo -n systemctl reset-failed pulsar-tp-prof $UNIT 2>/dev/null; true"
run_unit() {   # $1 = rank, $2 = extra --setenv args
    echo "sudo -n systemd-run --unit=$UNIT --uid=$(id -un) --gid=$(id -gn) -p LimitMEMLOCK=infinity \
-p WorkingDirectory=$D --setenv=HOME=$HOME --setenv=PULSAR_TP_RDMA_DEV=$RDMA_DEV \
--setenv=PULSAR_TP_RDMA_DEV2=$RDMA_DEV2 $2 $D/pulsar-server -m $MODEL --ctx 1048576 \
--tp-rank $1 --tp-nranks 2 --tp-peers $PEERS --tp-port 5590 --kv-disk-dir $KV"
}
reclaim() {    # both hosts: the GB10 teardown (drop_caches, >= 100 GiB available)
    for h in local "$WORKER"; do
        local cmd='sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"; for i in $(seq 1 120); do a=$(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo); [ "$a" -ge 100 ] && break; sleep 2; done; echo "$(hostname) MemAvailable $a GiB"'
        if [ "$h" = local ]; then bash -c "$cmd"; else ssh "$WORKER" "$cmd"; fi
    done
}
stop_pair() { eval "$STOP"; ssh "$WORKER" "$STOP"; sleep 3; }
start_pair() { # $1 = extra env for both ranks
    local t0; t0=$(date '+%Y-%m-%d %H:%M:%S')
    ssh "$WORKER" "$(run_unit 1 "$1")" >/dev/null
    sleep 5
    eval "$(run_unit 0 "$1")" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && { note "pair up ($((i * 5)) s)" >&2; echo "$t0"; return 0; }
        sleep 5
    done
    note "pair did not come up" >&2; echo "$t0"; return 1
}
journals() {   # $1 = since, $2 = tag: both ranks' logs for this stretch
    sudo -n journalctl -u $UNIT --since "$1" --no-pager > "$OUT/head-$2.log" 2>&1
    ssh "$WORKER" "sudo -n journalctl -u $UNIT --since '$1' --no-pager" > "$OUT/worker-$2.log" 2>&1
}
count() { grep -c -E "$1" "$2" 2>/dev/null || true; }
identity() {   # the engine prints the cross-rank tally when it closes: M/N must match
    local line; line=$(grep -h "cross-rank logits identity" "$OUT/head-$1.log" | tail -1)
    if [[ "$line" =~ ([0-9]+)/([0-9]+)\ worker ]] && [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[2]}" ]; then
        verdict "identity $1" "PASS (${BASH_REMATCH[1]}/${BASH_REMATCH[2]})"
    else
        verdict "identity $1" "FAIL (${line:-no tally line})"
    fi
}
alarms() {     # divergences and refusals: none expected outside the miss step
    local n; n=$(cat "$OUT/head-$1.log" "$OUT/worker-$1.log" | grep -c -i -E "diverg|marked failed|pair is marked|refusing to apply")
    [ "$n" = 0 ] && verdict "alarms $1" "PASS" || verdict "alarms $1" "FAIL ($n lines: grep -i -E 'diverg|marked failed' $OUT/*-$1.log)"
}

note "L264 TP2 validation: build $D, worker $WORKER, out $OUT"

# ---- 0. preflight: one protocol, one tree on both nodes --------------------
proto() { grep -o 'PULSAR_TP_PROTOCOL_VERSION [0-9]*u' "$1/src/tp/pulsar_tp.h"; }
P0=$(proto "$D"); P1=$(ssh "$WORKER" "grep -o 'PULSAR_TP_PROTOCOL_VERSION [0-9]*u' $D/src/tp/pulsar_tp.h")
S0=$(cd "$D" && cat src/tp/*.cpp src/tp/*.h src/engine/*.cpp src/engine/*.h src/lib/*.cpp src/lib/*.h | sha256sum | cut -c1-16)
S1=$(ssh "$WORKER" "cd $D && cat src/tp/*.cpp src/tp/*.h src/engine/*.cpp src/engine/*.h src/lib/*.cpp src/lib/*.h | sha256sum | cut -c1-16")
note "protocol head=$P0 worker=$P1; source digest head=$S0 worker=$S1"
if [ "$P0" != "PULSAR_TP_PROTOCOL_VERSION 23u" ] || [ "$P0" != "$P1" ] || [ "$S0" != "$S1" ] ||
   [ ! -x "$D/pulsar-server" ] || ! ssh "$WORKER" "test -x $D/pulsar-server"; then
    verdict preflight "FAIL (protocol/source/binary differ or missing -- build the same tree on both nodes)"
    exit 1
fi
verdict preflight PASS

# ---- 1. host tests (no GPU, no pair) ---------------------------------------
( cd "$D" && make -s CUDA_ARCH=sm_120f tests/tp_mirror_test tests/tp_transport_test tests/tp_mesh_test >/dev/null &&
  PULSAR_TP_TIMEOUT_SEC=1 timeout 60 ./tests/tp_mirror_test && PULSAR_TP_RDMA_DEV=none ./tests/tp_transport_test &&
  ./tests/tp_mesh_test ) > "$OUT/host-tests.log" 2>&1 && verdict "host tests" PASS || verdict "host tests" "FAIL ($OUT/host-tests.log)"

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
reclaim
echo
echo "================ L264 TP2 VERDICT ($OUT) ================"
printf '  %s\n' "${verdicts[@]}"
if printf '%s\n' "${verdicts[@]}" | grep -q FAIL; then echo "  => FAIL"; exit 1; fi
echo "  => ALL PASS"
echo "The pair is stopped; restart the production service with: sudo systemctl start pulsar-tp (both nodes)."

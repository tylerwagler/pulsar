#!/bin/bash
# The TP2 pair kits' shared plumbing (tp2-l264, tp2-l266): settings, the transient unit, the GB10 teardown,
# bring-up, the journals, the identity / alarm verdicts and the one-tree preflight.  Sourced by a kit after it
# sets D (the build dir, same path on both nodes), KIT (its name) and, where it differs from the defaults
# below, MODEL / CTX / KV / PROTO.  Every helper reads these; nothing here is kit-specific.
WORKER=${WORKER:-ca1070wk30007}
PEERS=${PEERS:-192.168.9.13:5590,192.168.9.12:5590}     # rank 0 (this host) first
RDMA_DEV=${RDMA_DEV:-mlx5_3}
RDMA_DEV2=${RDMA_DEV2:-mlx5_1}
CTX=${CTX:-1048576}
PROTO=${PROTO:-23}                                        # the TP protocol both nodes must speak
KV=${KV:-$HOME/$KIT-kv}                                   # a fresh disk KV dir on BOTH nodes
OUT=${OUT:-$HOME/$KIT-$(date +%m%d-%H%M)}
UNIT=pulsar-tp-exp
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
verdicts=()
note() { echo "[$(date +%H:%M:%S)] $*"; }
verdict() { verdicts+=("$1: $2"); note "VERDICT $1: $2"; }

STOP="sudo -n systemctl stop pulsar-tp pulsar-tp-prof $UNIT 2>/dev/null; sudo -n systemctl reset-failed pulsar-tp-prof $UNIT 2>/dev/null; true"
run_unit() {   # $1 = rank, $2 = extra --setenv args
    echo "sudo -n systemd-run --unit=$UNIT --uid=$(id -un) --gid=$(id -gn) -p LimitMEMLOCK=infinity \
-p WorkingDirectory=$D --setenv=HOME=$HOME --setenv=PULSAR_TP_RDMA_DEV=$RDMA_DEV \
--setenv=PULSAR_TP_RDMA_DEV2=$RDMA_DEV2 $2 $D/pulsar-server -m $MODEL --ctx $CTX \
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

preflight() {   # one protocol, one tree on both nodes
    proto() { grep -o 'PULSAR_TP_PROTOCOL_VERSION [0-9]*u' "$1/src/tp/pulsar_tp.h"; }
    P0=$(proto "$D"); P1=$(ssh "$WORKER" "grep -o 'PULSAR_TP_PROTOCOL_VERSION [0-9]*u' $D/src/tp/pulsar_tp.h")
    S0=$(cd "$D" && cat src/tp/*.cpp src/tp/*.h src/engine/*.cpp src/engine/*.h src/lib/*.cpp src/lib/*.h | sha256sum | cut -c1-16)
    S1=$(ssh "$WORKER" "cd $D && cat src/tp/*.cpp src/tp/*.h src/engine/*.cpp src/engine/*.h src/lib/*.cpp src/lib/*.h | sha256sum | cut -c1-16")
    note "protocol head=$P0 worker=$P1; source digest head=$S0 worker=$S1"
    if [ "$P0" != "PULSAR_TP_PROTOCOL_VERSION ${PROTO}u" ] || [ "$P0" != "$P1" ] || [ "$S0" != "$S1" ] ||
       [ ! -x "$D/pulsar-server" ] || ! ssh "$WORKER" "test -x $D/pulsar-server"; then
        verdict preflight "FAIL (protocol/source/binary differ or missing -- build the same tree on both nodes)"
        exit 1
    fi
    verdict preflight PASS
}

host_tests() {   # the TP transport / mirror / mesh host tests (no GPU, no pair)
    ( cd "$D" && make -s CUDA_ARCH=sm_120f tests/tp_mirror_test tests/tp_transport_test tests/tp_mesh_test >/dev/null &&
      PULSAR_TP_TIMEOUT_SEC=1 timeout 60 ./tests/tp_mirror_test && PULSAR_TP_RDMA_DEV=none ./tests/tp_transport_test &&
      ./tests/tp_mesh_test ) > "$OUT/host-tests.log" 2>&1 && verdict "host tests" PASS || verdict "host tests" "FAIL ($OUT/host-tests.log)"
}

final_verdict() {   # $1 = the kit's title
    reclaim
    echo
    echo "================ $1 VERDICT ($OUT) ================"
    printf '  %s\n' "${verdicts[@]}"
    if printf '%s\n' "${verdicts[@]}" | grep -q FAIL; then echo "  => FAIL"; exit 1; fi
    echo "  => ALL PASS"
    echo "The pair is stopped; restart the production service with: sudo systemctl start pulsar-tp (both nodes)."
}

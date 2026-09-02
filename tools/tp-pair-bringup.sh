#!/usr/bin/env bash
# TP pair bring-up one-shot (branch tensor_parallel; docs/tensor-parallel-
# bringup.md steps 0-5 made push-button).
#
# Builds the host tests + GPU-slab probe LOCALLY, copies them to both pair hosts
# over ssh/scp, then runs, in order:
#   1. the remote-leader/remote-worker transport test over the TCP fallback,
#   2. the same over real RDMA (PULSAR_TP_EXPECT_RDMA=1, MUST-RDMA),
#   3. the GPU-visible slab probe (cudaMallocManaged + attach + exchanges).
# Every leg is guarded by a timeout; any failing leg aborts the script.
# Exit codes are aggregated per leg, so a stuck pair prints which leg died.
#
# No positional args.  Env:
#   PULSAR_PAIR_A / PULSAR_PAIR_B   ssh/scp targets of the two boxes (required
#                                   for the pair legs; see --local-only)
#   PULSAR_PAIR_IP_A                A's RoCE/control IP for B to dial
#                                   (default: $PULSAR_PAIR_A)
#   PULSAR_PAIR_PORT                control port (default 5599)
#   PULSAR_TP_RDMA_DEV              HCA device, set IDENTICALLY on both ranks
#                                   (runbook F1; unset = auto, prints warning)
#   PULSAR_PAIR_REMOTE_DIR          staging dir on the pair (~/tp-pair-bringup)
#   PULSAR_PAIR_SSH                 extra ssh/scp args, e.g. -i key, -p 22
#   PULSAR_PAIR_SKIP_RDMA=1         stop after the TCP leg
#   PULSAR_PAIR_TIMEOUT             per-leg seconds (default 60)
#
#   --help          this usage
#   --local-only    build + run the loopback transport test here (no pair)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A="${PULSAR_PAIR_A:-}"
B="${PULSAR_PAIR_B:-}"
IP_A="${PULSAR_PAIR_IP_A:-$A}"
PORT="${PULSAR_PAIR_PORT:-5599}"
RDIR="${PULSAR_PAIR_REMOTE_DIR:-~/tp-pair-bringup}"
SSHX="${PULSAR_PAIR_SSH:-}"
TIMEOUT="${PULSAR_PAIR_TIMEOUT:-60}"
BINARIES=(tp_transport_test tp_slab_gpu_probe)
LOG="$(mktemp -d /tmp/tp-bringup.XXXXXX)"

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

rsh() { # rsh HOST CMD...  (run remotely, echo locally)
    local h="$1"; shift
    echo "[$h] $*"
    ssh $SSHX "$h" "cd $RDIR && $*"
}
rcp() {
    local h="$1"; shift
    ssh $SSHX "$h" "mkdir -p $RDIR" || return 1
    scp $SSHX "$@" "$h:$RDIR/" || return 1
}

step() { echo; echo "### $*"; }

build_local() {
    step "build (host tests + GPU probe)"
    make -C "$ROOT" tp-transport-test tp-slab-probe
}

# Return nonzero unless RDMA is expected to be exercisable.
leg_timeout() { timeout "$TIMEOUT" "$@"; }

run_local_loopback() {
    step "local sanity (loopback transport test)"
    ( cd "$ROOT" && ./tests/tp_transport_test )
    echo "local loopback: ok"
}

run_pair() { # run_pair LEG_NAME EXPECT_RDMA
    local leg="$1" expect="$2" log="$LOG/$1"
    local env_rdma=""
    [ "$expect" = 1 ] && env_rdma="PULSAR_TP_EXPECT_RDMA=1 "
    [ -n "${PULSAR_TP_RDMA_DEV:-}" ] && env_rdma+="PULSAR_TP_RDMA_DEV=$PULSAR_TP_RDMA_DEV "
    if [ -z "${PULSAR_TP_RDMA_DEV:-}" ] && [ "$expect" = 1 ]; then
        echo "WARNING: PULSAR_TP_RDMA_DEV unset — both ranks must auto-pick the SAME"
        echo "         cable (runbook F1). Set it identically on both ranks to be safe."
    fi

    step "$leg (leader A, worker B, transport${expect:+ RDMA-required})"
    rsh "$A" "$env_rdma ./tp_transport_test remote-leader 0.0.0.0 $PORT" >"$log.leader" 2>&1 &
    local lead_pid=$!
    sleep 2
    rsh "$B" "$env_rdma ./tp_transport_test remote-worker $IP_A $PORT" >"$log.worker" 2>&1
    wait "$lead_pid"
    echo "$leg: ok"
}

run_slab_probe() {
    local log="$LOG/slab"
    local env_rdma=""
    [ -n "${PULSAR_TP_RDMA_DEV:-}" ] && env_rdma="PULSAR_TP_RDMA_DEV=$PULSAR_TP_RDMA_DEV "
    if [ -z "${PULSAR_PAIR_SKIP_RDMA:-}" ]; then
        env_rdma+="PULSAR_TP_EXPECT_RDMA=1 "
    fi
    step "GPU slab probe (cudaMallocManaged + attach + gate/batch/big)"
    rsh "$A" "$env_rdma ./tp_slab_gpu_probe leader 0.0.0.0 $PORT" >"$log.leader" 2>&1 &
    local lead_pid=$!
    sleep 2
    rsh "$B" "$env_rdma ./tp_slab_gpu_probe worker $IP_A $PORT" >"$log.worker" 2>&1
    wait "$lead_pid"
    echo "slab probe: ok"
}

deploy() {
    step "deploy binaries to the pair"
    rcp "$A" "$ROOT"/tests/"${BINARIES[@]}"
    rcp "$B" "$ROOT"/tests/"${BINARIES[@]}"
}

interconnect_sanity() {
    step "interconnect sanity (informational; runbook step 1)"
    echo "  On each box, confirm the verbs port is ACTIVE on the device named by"
    echo "  PULSAR_TP_RDMA_DEV before expecting RDMA to work:"
    echo "      rdma link show              # look for 'state ACTIVE'"
    echo "      ibstat | grep -A1 Active    # CX-7 port state"
    echo "  Two live cables? Pin the SAME cable on both ranks (runbook F1)."
}

if [ "${1:-}" = "--help" ]; then usage; exit 0; fi
if [ "${1:-}" = "--local-only" ]; then build_local; run_local_loopback; exit 0; fi

if [ -z "$A" ] || [ -z "$B" ]; then
    echo "PULSAR_PAIR_A / PULSAR_PAIR_B are required for pair bring-up."
    echo "Run with --local-only to just build + loopback-test here."
    echo
    usage
    exit 2
fi

build_local
deploy
interconnect_sanity
run_pair "tcp-fallback" 0
if [ "${PULSAR_PAIR_SKIP_RDMA:-}" != 1 ]; then
    run_pair "rdma" 1
fi
run_slab_probe

echo
echo "bring-up: ALL LEGS OK — logs in $LOG"
exit 0

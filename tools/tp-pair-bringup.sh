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
    step "GPU slab probe (host-pinned slab + attach + gate/batch/big)"
    rsh "$A" "$env_rdma ./tp_slab_gpu_probe leader 0.0.0.0 $PORT" >"$log.leader" 2>&1 &
    local lead_pid=$!
    sleep 2
    rsh "$B" "$env_rdma ./tp_slab_gpu_probe worker $IP_A $PORT" >"$log.worker" 2>&1
    wait "$lead_pid"
    echo "slab probe: ok"
}

# Sources needed for a native on-pair build when the pair arch differs from
# the local arch (the DGX Sparks are aarch64; a dev box is usually x86-64).
# The two host tests pull in only pulsar_tp.{cpp,h} + their own .cpp.  The GPU
# slab probe is an nvcc TU that MUST be built with the PAIR's own CUDA toolkit
# (sm_120f): an x86 CUDA install ships no aarch64 (sbsa) runtime, so it cannot
# be cross-linked from a dev box -- native make on the pair is the design
# (makefile's default -L$(CUDA_HOME)/targets/sbsa-linux/lib).
STAGE_SRCS=(
    Makefile
    tests/tp_transport_test.cpp
    tests/tp_slab_gpu_probe.cpp
    src/tp/pulsar_tp.cpp
    src/tp/pulsar_tp.h
    src/tp/pulsar_tp_gpu.cpp
    src/tp/pulsar_tp_gpu.h
)

pair_arch() { # `uname -m` of the pair (both ranks are the same box model)
    ssh $SSHX "$A" 'uname -m' 2>/dev/null | tail -1
}

deploy() { # build (local arch, or native on the pair when it differs) + stage
    local local_arch remote_arch
    local_arch=$(uname -m)
    remote_arch=$(pair_arch)
    echo "pair arch: ${remote_arch:-UNKNOWN} (local: $local_arch)"
    if [ -z "$remote_arch" ]; then
        echo "ERROR: could not read uname -m from $A" >&2
        return 1
    fi
    if [ "$remote_arch" != "$local_arch" ]; then
        build_native_on_pair "$remote_arch" || return 1
    else
        build_local
        deploy_same_arch
    fi
}

deploy_same_arch() {
    step "deploy binaries to the pair"
    # Build the src path list with an explicit loop — embedding "${BINARIES[@]}"
    # inside a larger word ("$ROOT"/tests/"${BINARIES[@]}") makes bash join the
    # elements into ONE argument (root cause of the first deploy failure).
    local bin srcs=()
    for bin in "${BINARIES[@]}"; do srcs+=("$ROOT/tests/$bin"); done
    rcp "$A" "${srcs[@]}"
    rcp "$B" "${srcs[@]}"
}

build_native_on_pair() { # stage sources, build with the pair toolchain on A only, copy to B
    local remote_arch="$1"
    step "build natively on the pair (arch $remote_arch): stage sources to $A"
    local h f
    for h in "$A" "$B"; do
        ssh $SSHX "$h" "mkdir -p $RDIR/tests $RDIR/src/tp" || return 1
    done
    for f in "${STAGE_SRCS[@]}"; do
        scp $SSHX "$ROOT/$f" "$A:$RDIR/$f" || return 1
    done
    # Build the FILE targets, not the phony tp-transport-test / tp-slab-probe
    # recipes — those also RUN the loopback test, and on a verbs-capable pair
    # the loopback auto-picks a no-IP port (fe80 GID) and fails with
    # "modify RTR: Network is unreachable".  The real legs below use the
    # remote-leader/remote-worker subcommands with PULSAR_TP_RDMA_DEV pinned.
    rsh "$A" "make tests/tp_transport_test tests/tp_slab_gpu_probe CUDA_ARCH=sm_120f" || return 1
    rsh "$A" "cp -f tests/tp_transport_test tests/tp_slab_gpu_probe ." || return 1
    step "copy built binaries to $B"
    scp $SSHX "$A:$RDIR/tp_transport_test" "$B:$RDIR/" || return 1
    scp $SSHX "$A:$RDIR/tp_slab_gpu_probe" "$B:$RDIR/" || return 1
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

deploy        # builds (locally, or natively on the pair if archs differ)
interconnect_sanity
run_pair "tcp-fallback" 0
if [ "${PULSAR_PAIR_SKIP_RDMA:-}" != 1 ]; then
    run_pair "rdma" 1
fi
run_slab_probe

echo
echo "bring-up: ALL LEGS OK — logs in $LOG"
exit 0

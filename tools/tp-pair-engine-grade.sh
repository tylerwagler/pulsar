#!/usr/bin/env bash
# TP ENGINE-PAIR GRADING -- runbook step 6 made push-button and SELF-GRADING.
#
# Brings up the real engine on a TP group (2..n boxes, docs/tensor-parallel-
# bringup.md step 6), runs the SAME greedy prompt on every rank, and grades the
# result.  It splits into two legs:
#
#   LEG A (the strong one, needs no reference): every rank must produce
#   BYTE-IDENTICAL logprobs.  The vocab split's contract is that each rank
#   computes only its own vocab range, all-gathers the group's ranges and
#   assembles the FULL vector -- so every rank must end with the same logits.
#   If the gather, the range partition or the assembly is wrong, the ranks
#   disagree and this leg fails by name.  This is the "instrument proves it ran
#   the lane" check for slice 4d.
#
#   LEG B (optional): grade one rank against a SINGLE-BOX baseline.  TP is NOT
#   byte-exact (partials are summed in a new order), so this leg is a tolerance
#   report, never an equality assert -- rule 3.
#
# No positional args.  Environment:
#   PULSAR_TP_HOSTS    REQUIRED. ssh targets in RANK ORDER, rank 0 first, at
#                      least two.  e.g. "sparky workrig"
#   PULSAR_TP_MODEL    checkpoint dir (default /srv/models/vexp-safetensors)
#   PULSAR_TP_BIN      engine binary on each host, relative to $HOME or
#                      absolute (default ~/pulsar)
#   PULSAR_TP_PORT     TP control port (default 5590)
#   PULSAR_TP_PROMPT   prompt (default: a short deterministic one)
#   PULSAR_TP_TOKENS   generated tokens (default 32)
#   PULSAR_TP_CTX      context (default 4096)
#   PULSAR_TP_ARM      --tp-arm value (default prefill)
#   PULSAR_TP_SSH      extra ssh args, e.g. "-p 2222"
#   PULSAR_TP_TIMEOUT  per-rank seconds (default 900)
#   PULSAR_TP_STAGGER  seconds between rank launches (default 5; the mesh dials
#                      downward and accepts upward, so rank order is the safe
#                      start order)
#   PULSAR_TP_BASELINE single-box logprobs JSON to grade against (LEG B)
#   PULSAR_TP_WORKDIR  staging dir on each host (default ~/tp-pair-grade)
#   PULSAR_TP_DRYRUN=1 print the plan and exit without running anything
#
# Exit: 0 only if every rank exited 0, no rank refused/desynced, and LEG A
# (all ranks byte-identical) passed.

set -u

HOSTS=${PULSAR_TP_HOSTS:-}
MODEL=${PULSAR_TP_MODEL:-/srv/models/vexp-safetensors}
BIN=${PULSAR_TP_BIN:-'$HOME/pulsar'}
PORT=${PULSAR_TP_PORT:-5590}
PROMPT=${PULSAR_TP_PROMPT:-"Explain how a C pointer differs from an array in one paragraph."}
TOKENS=${PULSAR_TP_TOKENS:-32}
CTX=${PULSAR_TP_CTX:-4096}
ARM=${PULSAR_TP_ARM:-prefill}
SSH_ARGS=${PULSAR_TP_SSH:-}
TIMEOUT=${PULSAR_TP_TIMEOUT:-900}
STAGGER=${PULSAR_TP_STAGGER:-5}
BASELINE=${PULSAR_TP_BASELINE:-}
WORKDIR=${PULSAR_TP_WORKDIR:-tp-pair-grade}
DRYRUN=${PULSAR_TP_DRYRUN:-0}

die() { echo "tp-pair-engine-grade: $*" >&2; exit 2; }

[ -n "$HOSTS" ] || die "PULSAR_TP_HOSTS is required (rank order, rank 0 first, >= 2 hosts)"
read -r -a RANKS <<< "$HOSTS"
N=${#RANKS[@]}
[ "$N" -ge 2 ] || die "need at least two hosts, got $N"

# The mesh takes the whole group's listen addresses up front.
PEERS=""
for r in "${RANKS[@]}"; do
    PEERS="${PEERS}${PEERS:+,}${r}:${PORT}"
done

echo "tp-pair-engine-grade: n=$N ranks"
for i in "${!RANKS[@]}"; do echo "  rank $i -> ${RANKS[$i]}"; done
echo "  peers: $PEERS"
echo "  model: $MODEL   arm: $ARM   ctx: $CTX   tokens: $TOKENS"

# ---- plan -------------------------------------------------------------------
run_rank_cmd() {   # $1 = rank index
    local r=$1
    printf 'cd %s && PULSAR_LOCK_FILE=/tmp/tp-grade-lock-%d PULSAR_TP_RANK=%d %s -m %s ' \
           "$WORKDIR" "$r" "$r" "$BIN" "$MODEL"
    printf -- '--tp-rank %d --tp-nranks %d --tp-peers %s --tp-port %d --tp-arm %s ' \
           "$r" "$N" "$PEERS" "$PORT" "$ARM"
    printf -- '-c %d --nothink --temp 0 -n %d --dump-logprobs rank%d.lp.json -p %q' \
           "$CTX" "$TOKENS" "$r" "$PROMPT"
}

if [ "$DRYRUN" != 0 ]; then
    echo "--- plan (dry run) ---"
    for r in $(seq 0 $((N - 1))); do
        echo "rank $r: ssh $SSH_ARGS ${RANKS[$r]} \"$(run_rank_cmd "$r") &\""
    done
    exit 0
fi

# ---- launch -----------------------------------------------------------------
pids=()
for r in $(seq 0 $((N - 1))); do
    h=${RANKS[$r]}
    ssh $SSH_ARGS -o BatchMode=yes "$h" "mkdir -p $WORKDIR && rm -f $WORKDIR/rank$r.*" \
        || die "rank $r host $h unreachable / cannot stage $WORKDIR"
    # Per-rank lock file: the instance lock is per MACHINE, so a same-host dry
    # run needs its own, and it is harmless when the ranks are on real boxes.
    ssh $SSH_ARGS -o BatchMode=yes "$h" \
        "cd $WORKDIR && ($(run_rank_cmd "$r") > rank$r.out 2> rank$r.err; echo \$? > rank$r.rc) &" \
        || die "rank $r launch failed"
    echo "  launched rank $r on $h"
    [ "$r" -lt $((N - 1)) ] && sleep "$STAGGER"
done

# ---- wait + collect ---------------------------------------------------------
echo "tp-pair-engine-grade: waiting up to ${TIMEOUT}s per rank"
for r in $(seq 0 $((N - 1))); do
    h=${RANKS[$r]}
    ssh $SSH_ARGS -o BatchMode=yes "$h" \
        "for i in \$(seq 1 $TIMEOUT); do [ -f $WORKDIR/rank$r.rc ] && break; sleep 1; done; \
         [ -f $WORKDIR/rank$r.rc ] || echo TIMEOUT > $WORKDIR/rank$r.rc" \
        || true   # an unreachable host leaves no rc file, which the read below fails on
done

fail=0
echo "--- per-rank rc ---"
for r in $(seq 0 $((N - 1))); do
    h=${RANKS[$r]}
    rc=$(ssh $SSH_ARGS -o BatchMode=yes "$h" "cat $WORKDIR/rank$r.rc 2>/dev/null" | tr -d '[:space:]')
    echo "  rank $r ($h): rc=${rc:-?}"
    [ "${rc:-}" = "0" ] || fail=1
done

# ---- refusal/desync scan ----------------------------------------------------
echo "--- refusal / desync scan ---"
for r in $(seq 0 $((N - 1))); do
    h=${RANKS[$r]}
    bad=$(ssh $SSH_ARGS -o BatchMode=yes "$h" \
          "grep -icE 'tensor parallelism bring-up failed|desync|refus|no channel to rank|cannot honor expert ownership|owned-expert range refused|vocab range refused' $WORKDIR/rank$r.err" \
          | tr -d '[:space:]')
    if [ "${bad:-0}" != "0" ]; then
        echo "  rank $r: $bad refusal/desync line(s):"
        ssh $SSH_ARGS -o BatchMode=yes "$h" \
            "grep -iE 'tensor parallelism bring-up failed|desync|refus|no channel to rank|cannot honor expert ownership|owned-expert range refused|vocab range refused' $WORKDIR/rank$r.err | head -5"
        fail=1
    else
        echo "  rank $r: clean"
    fi
done

# ---- LEG A: every rank must agree byte-for-byte -----------------------------
echo "--- LEG A: cross-rank logprobs identity (the vocab gather's contract) ---"
for r in $(seq 1 $((N - 1))); do
    if ssh $SSH_ARGS -o BatchMode=yes "${RANKS[0]}" "cat $WORKDIR/rank0.lp.json" > /tmp/.tp-grade-r0.json 2>/dev/null \
       && ssh $SSH_ARGS -o BatchMode=yes "${RANKS[$r]}" "cat $WORKDIR/rank$r.lp.json" > /tmp/.tp-grade-rr.json 2>/dev/null \
       && [ -s /tmp/.tp-grade-r0.json ] && [ -s /tmp/.tp-grade-rr.json ]; then
        if cmp -s /tmp/.tp-grade-r0.json /tmp/.tp-grade-rr.json; then
            echo "  rank 0 == rank $r: BYTE-IDENTICAL ($(wc -c < /tmp/.tp-grade-r0.json) bytes)"
        else
            echo "  rank 0 != rank $r: DIFFER -- the assembly disagreed across ranks"
            diff <(head -c 400 /tmp/.tp-grade-r0.json) <(head -c 400 /tmp/.tp-grade-rr.json) | head -6
            fail=1
        fi
    else
        echo "  rank $r: logprobs missing or empty -- cannot grade (a rank that never"
        echo "           reached the head produces no file, which is itself a failure)"
        fail=1
    fi
done

# ---- LEG B (optional, tolerance only): grade against a single-box baseline ---
if [ -n "$BASELINE" ]; then
    echo "--- LEG B: vs single-box baseline (TOLERANCE, TP is not byte-exact) ---"
    if ssh $SSH_ARGS -o BatchMode=yes "${RANKS[0]}" "cat $WORKDIR/rank0.lp.json" > /tmp/.tp-grade-b.json 2>/dev/null \
       && [ -s /tmp/.tp-grade-b.json ]; then
        python3 - "$BASELINE" /tmp/.tp-grade-b.json <<'PY' || fail=1
import json, sys
base = json.load(open(sys.argv[1]))
tp   = json.load(open(sys.argv[2]))
bs, ts = base.get("steps", []), tp.get("steps", [])
print(f"  steps: baseline {len(bs)} vs tp {len(ts)}")
n = min(len(bs), len(ts))
worst = 0.0
flips = 0
for i in range(n):
    b = {t["token"]["id"]: t["logprob"] for t in bs[i]["top_logprobs"]}
    t = {t["token"]["id"]: t["logprob"] for t in ts[i]["top_logprobs"]}
    if bs[i]["selected"]["id"] != ts[i]["selected"]["id"]:
        flips += 1
    for k, v in b.items():
        if k in t:
            worst = max(worst, abs(v - t[k]))
print(f"  greedy-token disagreements: {flips}/{n}")
print(f"  worst |logprob delta| over shared top-k: {worst:.6f}")
if flips == 0 and worst < 1e-3:
    print("  LEG B: within tolerance")
else:
    print("  LEG B: OUTSIDE tolerance -- inspect before trusting the split")
    sys.exit(1)
PY
    else
        echo "  baseline or rank0 logprobs unavailable -- LEG B skipped"
    fi
fi

echo "===================== tp-pair-engine-grade ====================="
if [ "$fail" = 0 ]; then
    echo "  PASS: all ranks exited 0, no refusal/desync, all ranks byte-identical"
    exit 0
fi
echo "  FAIL: see the legs above"
exit 1

#!/bin/bash
# MXFP6-attention quality gate orchestrator (overnight, detached).
#
# Sequence (disk fits only ONE 91GB splice at a time next to the retrain
# agent's artifacts, so the two variants are strictly sequential):
#   1. wait for the already-running e2m3 splice to finish
#   2. [flock] baseline dump x2 (determinism proof) + e2m3 dump
#   3. delete the e2m3 artifact, build the e3m2 splice (CPU, lock released)
#   4. [flock] e3m2 dump
#   5. drift tables -> $WT/temp/mxfp6-gate-results.txt
#
# All model loads run inside gate_phase.sh under
# flock $LOCK.  Settings (env, required): WT the built tree, ART the artifact dir, LOCK the GPU lock file,
# PROD the production model, PY the interpreter with the splice tool's deps.
set -u
WT=${WT:?set WT to the built tree}
ART=${ART:?set ART to the artifact directory}
LOCK=${LOCK:?set LOCK to the GPU lock file}
PROD=${PROD:?set PROD to the production model}
PY=${PY:?set PY to the interpreter with the splice dependencies}
RES="$WT/temp/mxfp6-gate-results.txt"
LOG="$ART/gate.log"

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

say "=== MXFP6 gate starting (pid $$) ==="

# --- 1. wait for the e2m3 splice (launched before this script) ---------------
say "waiting for e2m3 splice to finish..."
ok=""
for _ in $(seq 720); do
    if grep -q '^sha256(' "$ART/splice-e2m3.log" 2>/dev/null; then ok=1; break; fi
    if ! pgrep -f 'splice_mxfp6_attn.py' >/dev/null; then
        sleep 5
        grep -q '^sha256(' "$ART/splice-e2m3.log" 2>/dev/null && { ok=1; break; }
        say "FATAL: e2m3 splice died without sha; see splice-e2m3.log"; exit 1
    fi
    sleep 60
done
[ -n "$ok" ] || { say "FATAL: e2m3 splice timeout"; exit 1; }
say "e2m3 splice done: $(grep '^sha256(' "$ART/splice-e2m3.log")"

# --- 2. baseline x2 + e2m3 dumps under one lock hold --------------------------
say "acquiring GPU lock for baseline+e2m3 (may wait hours)..."
flock -w 28800 "$LOCK" bash "$WT/tools/mxfp6/gate_phase.sh" "$LOG" \
    "$PROD:$ART/logits-base-a" \
    "$PROD:$ART/logits-base-b" \
    "$ART/v5mx-attn-mxfp6-e2m3.gguf:$ART/logits-e2m3"
rc=$?
if [ $rc -ne 0 ]; then say "FATAL: phase-1 GPU runs failed rc=$rc"; exit 1; fi

# --- 3. swap artifacts: rm e2m3, build e3m2 (CPU, no lock) --------------------
say "deleting e2m3 artifact, building e3m2 splice..."
rm -f "$ART/v5mx-attn-mxfp6-e2m3.gguf"
$PY "$WT/tools/mxfp6/splice_mxfp6_attn.py" "$PROD" \
    "$ART/v5mx-attn-mxfp6-e3m2.gguf" --fmt e3m2 --set attn+shexp \
    > "$ART/splice-e3m2.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "FATAL: e3m2 splice failed rc=$rc"; exit 1; fi
say "e3m2 splice done: $(grep '^sha256(' "$ART/splice-e3m2.log")"

# --- 4. e3m2 dump --------------------------------------------------------------
say "acquiring GPU lock for e3m2..."
flock -w 28800 "$LOCK" bash "$WT/tools/mxfp6/gate_phase.sh" "$LOG" \
    "$ART/v5mx-attn-mxfp6-e3m2.gguf:$ART/logits-e3m2"
rc=$?
if [ $rc -ne 0 ]; then say "FATAL: e3m2 GPU run failed rc=$rc"; exit 1; fi

# --- 5. drift tables -----------------------------------------------------------
say "computing drift tables..."
{
    echo "MXFP6 attention pre-check drift tables ($(date))"
    echo "model: $PROD"
    echo "spliced set: 344 tensors (43 layers x {attn_q_a,q_b,kv,output_a,output_b,ffn_{gate,up,down}_shexp}), 5.86 GB"
    echo
    echo "-- determinism proof: baseline run B vs run A (must be 0.000) --"
    $PY "$WT/tools/mxfp6/compare_frontier_logits.py" \
        "$ART/logits-base-a" "$ART/logits-base-b" --label "base-b (determinism)"
    echo
    $PY "$WT/tools/mxfp6/compare_frontier_logits.py" \
        "$ART/logits-base-a" "$ART/logits-e2m3" --label "MXFP6-E2M3 attention"
    echo
    $PY "$WT/tools/mxfp6/compare_frontier_logits.py" \
        "$ART/logits-base-a" "$ART/logits-e3m2" --label "MXFP6-E3M2 attention"
    echo
    echo "calibration: MXFP8 head 1.87% shipped | NVFP4 head 6.76% rejected | MXFP4 attn 33% dead"
    echo "read: <3% promising | 3-7% borderline | >10% dead"
} > "$RES" 2>&1
say "=== gate COMPLETE; results in $RES ==="

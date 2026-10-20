#!/bin/bash
# scoreboard.sh ENGINE MODEL OUTDIR LABEL [server args...] -- the L288 cross-engine scoreboard, one engine at a time.
#
# ENGINE is `pulsar` (./pulsar-server of PULSAR_TREE, default the tree this script sits in) or `tensorfold`
# (TF_DIR/.venv/bin/tensorfold serve, default ~/tf-run, `--parallel $TF_PARALLEL`, default 8, so concurrent
# streams batch; TF_PARALLEL= serves without it).  STREAMS_SET= skips the concurrent cells, NO_PUB=1 the public client.  The server
# runs as a direct child in its own process group, killed on every exit path; the script refuses a busy port and
# never leaves a listener.  Run it under the GPU lock with the lock fd closed, e.g. on sparky:
#
#   flock -o ~/gpu.lock -c 'timeout 1800 tools/serve-bench/scoreboard.sh pulsar /srv/models/X ~/sb ours'
#   flock -o ~/gpu.lock -c 'timeout 1800 tools/serve-bench/scoreboard.sh tensorfold /srv/models/tf-exl3-305 ~/sb tf'
#   tools/serve-bench/serve_matrix.py table ~/sb/ours.jsonl ~/sb/tf.jsonl
#
# Cells (serve_matrix.py scoreboard, untraced): single code T0 / prose T0 / code T0.7 (median of 3 x 512 tokens), edit T0
# (1024 tokens: a quoted ~165-line module re-stated with a rename -- the copy-draft workload),
# 4 and 8 concurrent streams T0 (256 tokens each; one discarded warm run, then the median [min-max] of 5); each with tok/s, tokens/round and ms/round from the server's own
# counters (pulsar /metrics, TensorFold's per-request "done ... rounds=" log lines).  Then TensorFold's public client
# (tools/bench_openai.py of its tree, TF_BENCH; its two fixtures, 64 tokens, T1.0 and T0, REPS_PUB=10 runs) into
# LABEL-pub.json, so both engines also compare with TensorFold's published table.  Never under nsys: tracing slows
# the host and inflates waits (L288 #52), so cross-engine ratios come from this script only.
ENGINE=$1; MODEL=$2; OUT=$3; LABEL=$4; shift 4
PORT=${PORT:-8099}
HERE=$(cd "$(dirname "$0")" && pwd)
SM=$HERE/serve_matrix.py
TF_DIR=${TF_DIR:-$HOME/tf-run}
TF_BENCH=${TF_BENCH:-$TF_DIR/src/tools/bench_openai.py}
TF_PARALLEL=${TF_PARALLEL-8}   # TensorFold's CUDA server runs concurrent requests one at a time without --parallel
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
# a battery's marker refuses -- unless it is gpuq's marker for the quiet-class job this runs in (GPUQ_JOB_ID)
[ -e ~/BATTERY_RUNNING ] && ! grep -q "^gpuq job ${GPUQ_JOB_ID:-none} " ~/BATTERY_RUNNING && {
  echo "battery running -- refusing"; exit 9; }
listener() { ss -ltnpH "sport = :$PORT" | grep -o "pid=[0-9]*" | cut -d= -f2 | sort -u; }
SP=
cleanup() {
  [ -n "$SP" ] || return 0
  kill -TERM -$SP 2>/dev/null; for i in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -KILL -$SP 2>/dev/null; wait $SP 2>/dev/null
  for p in $(listener); do kill -KILL $p 2>/dev/null; done
  SP=
}
trap cleanup EXIT
trap "exit 130" INT TERM HUP
if [ -n "$(listener)" ]; then echo "port $PORT busy -- refusing"; exit 5; fi
sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null
case $ENGINE in
pulsar)
  TREE=${PULSAR_TREE:-$(cd "$HERE/../.." && pwd)}
  cd "$TREE" || exit 2
  WHAT="pulsar $(git log --oneline -1 | cut -c1-60); dirty $(git status --short -uno | wc -l); binary $(stat -c %y pulsar-server | cut -c1-19)"
  setsid env PULSAR_MSEQ_BANKS=16 ./pulsar-server -m "$MODEL" --ctx 262144 --host 127.0.0.1 --port $PORT --no-kv-disk "$@" \
      > "$OUT/server-$LABEL.log" 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
  SP=$! ;;
tensorfold)
  cd "$TF_DIR" || exit 2
  WHAT="tensorfold $("$TF_DIR/.venv/bin/python" -c 'import tensorfold; print(tensorfold.__version__)' 2>/dev/null)"
  setsid env CC=gcc-13 CXX=g++-13 TORCH_EXTENSIONS_DIR=$TF_DIR/ext TRITON_CACHE_DIR=$TF_DIR/triton \
      "$TF_DIR/.venv/bin/tensorfold" serve "$MODEL" --host 127.0.0.1 --port $PORT --name m ${TF_PARALLEL:+--parallel $TF_PARALLEL} "$@" \
      > "$OUT/server-$LABEL.log" 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
  SP=$! ;;
*) echo "ENGINE must be pulsar or tensorfold"; exit 2 ;;
esac
for i in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 "$OUT/server-$LABEL.log"; exit 3; }
  sleep 2
done
echo "[$LABEL] up after ~$((i*2)) s ($WHAT) model $MODEL args: $*"
python3 "$SM" single --port $PORT --prompts code,prose --temps 0,0.7 --tokens 128 --reps 1 > /dev/null 2>&1   # warm-up
python3 "$SM" conc --port $PORT --streams 4 --temps 0 --tokens 64 > /dev/null 2>&1
rm -f "$OUT/$LABEL.jsonl"
LOGARG=(); [ "$ENGINE" = tensorfold ] && LOGARG=(--server-log "$OUT/server-$LABEL.log")
python3 "$SM" scoreboard "${LOGARG[@]}" --streams-set "${STREAMS_SET-4,8}" --port $PORT --label "$LABEL" --out "$OUT/$LABEL.jsonl" > /dev/null || exit 4
if [ -f "$TF_BENCH" ] && [ -z "$NO_PUB" ]; then
  python3 "$TF_BENCH" http://127.0.0.1:$PORT m --tokens 64 --reps ${REPS_PUB:-10} --temperatures 1.0,0 \
      --label "$LABEL" --output "$OUT/$LABEL-pub.json" | sed "s/^/[pub] /"
fi
python3 "$SM" table "$OUT/$LABEL.jsonl"

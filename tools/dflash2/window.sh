#!/bin/bash
# L292 DFlash2 gate, the GPU window (production :8000 stopped by Tyler first).  td305 = the model the L288 MTP table
# was measured on; the code/prose/story replies are ~/qaccept/p1's (the table's own tokens).
#   1. code tap dump (teacher-forced prefill of prompt + reply, layers 5/15/25/35/45 streams) -> validated, abort if bad
#   2. prose, story tap dumps
#   3. edit: greedy reply (1024) + unstopped depth-8 MTP trace, then its tap dump
#   4. code: unstopped depth-8 MTP trace on today's build (drift check of the L288 table; cut first if short of time)
# run: gpuq submit --class quiet --mem 60G --timeout 25m --label dflash2-taps -- bash ~/dflash2/window.sh
set -u
D=$HOME/dflash2; G=$D/engine/tests/qwen_generate; M=/srv/models/qwen38fn-td305; P1=$HOME/qaccept/p1
OUT=$D/taps; mkdir -p $OUT
T0=$(date +%s)
el() { echo "[$(( $(date +%s) - T0 ))s]"; }
fresh() { sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
  local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "$(el) MemAvailable ${a} GiB"; [ "$a" -lt 60 ] && { echo "MemAvailable ${a} GiB < 60 -- refusing"; exit 6; }; }
[ -e ~/BATTERY_RUNNING ] && { echo "battery running -- stopping"; exit 9; }
pgrep -f "[p]ulsar-server-" >/dev/null && { echo "a pulsar-server is running -- the window is not open"; exit 7; }
echo "$(el) tree $(git -C $D/engine log --oneline -1 | cut -c1-60), binary $(stat -c %y $G)"

dump() {  # fixture tokens.bin
  local fx=$1 ids=$2; rm -rf $OUT/$fx; mkdir -p $OUT/$fx
  fresh
  env PULSAR_CUDA_GRAPH_DUMP_PREFIX=$OUT/$fx/t PULSAR_CUDA_GRAPH_DUMP_NAME=qwen_h \
      PULSAR_CUDA_GRAPH_DUMP_LAYER=5,15,25,35,45 \
      $G $M $ids 1 $OUT/$fx/check.ids > $OUT/$fx/dump.log 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-
  local rc=$?
  echo "$(el) dump $fx rc=$rc tokens $(( $(stat -c %s $ids) / 4 )) files $(ls $OUT/$fx | grep -c '^t_') bytes $(du -sb $OUT/$fx | cut -f1)"
  return $rc
}
validate() {  # fixture n_tokens: every tapped layer covers positions 0..n-1 exactly once, rows of 4 x 2560 f32
  python3 - "$OUT/$1" "$2" <<'EOF'
import os, re, sys
d, n = sys.argv[1], int(sys.argv[2]); row = 4 * 2560 * 4; cov = {}
for f in os.listdir(d):
    m = re.match(r"t_qwen_h-(\d+)_pos(\d+)(?:_c(\d+))?\.bin$", f)
    if not m: continue
    il, last, c = int(m[1]), int(m[2]), int(m[3] or 1)
    sz = os.path.getsize(os.path.join(d, f)); assert sz % row == 0, (f, sz)
    r = sz // row; cov.setdefault(il, []).append((last - r + 1, last, c))
ok = sorted(cov) == [5, 15, 25, 35, 45]
for il, spans in sorted(cov.items()):
    pos = sorted(p for a, b, c in spans if c == 1 for p in range(a, b + 1))
    good = pos == list(range(n)); ok &= good
    print(f"  layer {il}: {len(spans)} files, calls {sorted(set(c for *_, c in spans))}, covers 0..{n-1}: {good}")
print("VALID" if ok else "INVALID"); sys.exit(0 if ok else 1)
EOF
}

dump code $P1/ours-td305-code-d6-off.ids || exit 1
validate code $(( $(stat -c %s $P1/ours-td305-code-d6-off.ids) / 4 )) || { echo "$(el) code dump INVALID -- stopping"; exit 1; }
dump prose $P1/ours-td305-prose-d6-off.ids
dump story $P1/ours-td305-story-d6-off.ids

fresh
env QWEN_SPEC=1 QWEN_SPEC_K=8 PULSAR_SPEC_TAU=off QWEN_SPEC_TRACE=$OUT/mtp-edit-d8-off.trace \
    $G $M $D/ids-edit.bin 1024 $OUT/edit.ids > $OUT/mtp-edit.log 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-
echo "$(el) edit spec rc=$? $(grep -E 'SPEC [0-9]+ rounds|identical' $OUT/mtp-edit.log | tr '\n' ' ' | cut -c1-220)"
dump edit $OUT/edit.ids

fresh
env QWEN_SPEC=1 QWEN_SPEC_K=8 PULSAR_SPEC_TAU=off QWEN_SPEC_TRACE=$OUT/mtp-code-d8-off.trace \
    $G $M $D/ids-code.bin 512 $OUT/code-today.ids > $OUT/mtp-code.log 2>&1 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-
echo "$(el) code spec rc=$? $(grep -E 'SPEC [0-9]+ rounds|identical' $OUT/mtp-code.log | tr '\n' ' ' | cut -c1-220)"
for fx in prose story edit; do validate $fx $(( $(stat -c %s $( [ $fx = edit ] && echo $OUT/edit.ids || echo $P1/ours-td305-$fx-d6-off.ids) ) / 4 )); done
echo "$(el) WINDOW DONE"

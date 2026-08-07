#!/bin/bash
# Does __launch_bounds__(512,2) on the prefill attention kernel change NUMERICS?
# It should not -- it constrains register allocation, not arithmetic -- but
# register pressure can change instruction selection and FMA contraction, so
# this is measured rather than assumed.  Same model, same prompt, same build
# except for -DPULSAR_ATTN_MIN_BLOCKS=2; anything but a byte-identical logits
# dump means the occupancy knob moved the math.
set -u
M=/srv/models/v5mx4-0731-mmqaligned.gguf
P=/home/claude/pulsar-work/speed-bench/promessi_sposi.txt
OUT=/home/claude/ab-results
export PULSAR_LOCK_FILE=$HOME/pulsar-claude.lock
rm -rf "$OUT/lg-lb" "$OUT/lg-nolb"; mkdir -p "$OUT/lg-lb" "$OUT/lg-nolb"

run() {
  sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 3
  "$1" -m "$M" --prompt-file "$P" --ctx-start 4096 --ctx-max 4096 \
       --gen-tokens 0 --prefill-chunk 4096 --dump-frontier-logits-dir "$2" \
       > "$2/run.log" 2>&1
  echo "$1 rc=$?"
}
run /home/claude/pb-nolb "$OUT/lg-nolb"
run /home/claude/pb-lb2  "$OUT/lg-lb"

echo "=== byte compare ==="
if cmp -s "$OUT/lg-nolb/frontier_004096.logits.json" "$OUT/lg-lb/frontier_004096.logits.json"; then
  echo "BYTE-IDENTICAL -- launch_bounds is numerically neutral"
else
  echo "DIFFERS -- launch_bounds moved the math, quantify before shipping"
fi
echo "LB_CHECK_DONE"
